#include "strata/core/conversation_memory.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace strata::core {

namespace fs = std::filesystem;

namespace {

std::atomic<uint64_t> g_memory_limit_bytes{0};

// "<key> <n> kB", exactly once, nothing else on the line; the value in bytes.  Anything else is unknown.
std::optional<uint64_t> meminfo_kb_bytes(std::istream& meminfo, const char* wanted) {
    std::optional<uint64_t> result;
    std::string line;
    while (std::getline(meminfo, line)) {
        std::istringstream fields(line);
        std::string key, value, unit, extra;
        if (!(fields >> key) || key != wanted) continue;
        if (result || !(fields >> value >> unit) || unit != "kB" || (fields >> extra)) return {};
        uint64_t kb = 0;
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), kb);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
            kb > std::numeric_limits<uint64_t>::max() / 1024) return {};
        result = kb * 1024;
    }
    if (meminfo.bad() || (meminfo.fail() && !meminfo.eof())) return {};
    return result;
}

// One cgroup control file: absent (no such file), a number, "max", or unusable (unreadable, empty, not a number).
struct Value {
    enum Kind { absent, bad, max, number } kind = absent;
    uint64_t n = 0;
};

Value read_value(const fs::path& path) {
    std::error_code ec;
    const bool there = fs::exists(path, ec);
    if (ec) return {Value::bad, 0};
    if (!there) return {};
    std::ifstream file(path);
    if (!file) return {Value::bad, 0};
    char buf[64];
    file.read(buf, sizeof buf);
    if (file.bad()) return {Value::bad, 0};
    size_t len = (size_t) file.gcount();
    if (len == sizeof buf) return {Value::bad, 0};
    while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r' || buf[len - 1] == ' ' || buf[len - 1] == '\t')) --len;
    if (len == 0) return {Value::bad, 0};
    if (len == 3 && std::string(buf, 3) == "max") return {Value::max, 0};
    Value v{Value::number, 0};
    const auto parsed = std::from_chars(buf, buf + len, v.n);
    if (parsed.ec != std::errc{} || parsed.ptr != buf + len) return {Value::bad, 0};
    return v;
}

// The one reclaimable kind that is credited: clean inactive file cache, from the same group's memory.stat
// (v2 inactive_file, v1 total_inactive_file), less its dirty and writeback pages when those are listed, and never
// more than the group's charged usage.  active_file (it holds mlocked pages) and shmem (pinned complements, the KV
// pool) are not credited.  A memory.stat that is missing or unparsable gives no credit - not an unknown sample.
uint64_t reclaimable_cache(const fs::path& dir, uint64_t current, bool v1) {
    std::ifstream stat(dir / "memory.stat");
    if (!stat) return 0;
    const char* inactive_key = v1 ? "total_inactive_file" : "inactive_file";
    const char* dirty_key = v1 ? "total_dirty" : "file_dirty";
    const char* writeback_key = v1 ? "total_writeback" : "file_writeback";
    std::optional<uint64_t> inactive;
    uint64_t dirty = 0, writeback = 0;
    std::string line;
    while (std::getline(stat, line)) {
        std::istringstream fields(line);
        std::string key, value, extra;
        if (!(fields >> key >> value) || (fields >> extra)) continue;
        uint64_t* target = key == dirty_key ? &dirty : key == writeback_key ? &writeback : nullptr;
        uint64_t n = 0;
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), n);
        const bool ok = parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size();
        if (key == inactive_key) { if (!ok || inactive) return 0; inactive = n; }
        else if (target != nullptr) { if (!ok) return 0; *target = n; }
    }
    if (!inactive) return 0;
    uint64_t credit = std::min(*inactive, current);
    credit = dirty >= credit ? 0 : credit - dirty;
    credit = writeback >= credit ? 0 : credit - writeback;
    return credit;
}

// What the cgroup files say: every finite limit with the usage charged against it (and the cache credit of that
// group), and the usage and credit of the topmost visible group (the root of this namespace - what a container's own
// limit is measured over).
struct CgroupScan {
    struct Limit { uint64_t limit, current, credit; };
    std::vector<Limit> limits;
    std::optional<uint64_t> root_current;
    uint64_t root_credit = 0;
    bool any_files = false;
};

// `dirs` runs from the top visible group down to the process's own.  False when a file that is there cannot be used.
bool scan_levels(const std::vector<fs::path>& dirs, const char* const* limit_names, size_t n_limits,
                 const char* current_name, bool v1, CgroupScan& out) {
    for (size_t i = 0; i < dirs.size(); ++i) {
        const Value current = read_value(dirs[i] / current_name);
        std::vector<Value> limits;
        for (size_t k = 0; k < n_limits; ++k) limits.push_back(read_value(dirs[i] / limit_names[k]));
        bool unusable = current.kind == Value::bad || current.kind == Value::max;
        bool finite_limit = false;
        for (const Value& v : limits) {
            unusable = unusable || v.kind == Value::bad;
            finite_limit = finite_limit || v.kind == Value::number;
        }
        if (unusable) return false;
        if (current.kind == Value::absent) {
            if (finite_limit) return false;   // a limit with no usage beside it: the room cannot be told
            continue;                          // no memory controller at this level (e.g. the host's root group)
        }
        out.any_files = true;
        const uint64_t credit = reclaimable_cache(dirs[i], current.n, v1);
        if (i == 0) { out.root_current = current.n; out.root_credit = credit; }
        for (const Value& v : limits) {
            if (v.kind != Value::number) continue;
            if (v1 && v.n >= (1ull << 62)) continue;   // v1's "unlimited" is a number near 2^63
            out.limits.push_back({v.n, current.n, credit});
        }
    }
    return true;
}

// "/a/b/c" -> root, root/a, root/a/b, root/a/b/c, cut at the first one that is not a directory here: a container
// without its own cgroup namespace sees its group's host path in /proc/self/cgroup, but only its own subtree under
// the mount, so the deepest one that exists (at worst the mount itself) is the nearest group we can read.
bool group_dirs(const fs::path& root, const std::string& group, std::vector<fs::path>& dirs) {
    dirs.clear();
    std::vector<std::string> names;
    std::istringstream parts(group);
    for (std::string part; std::getline(parts, part, '/');) {
        if (part.empty()) continue;
        if (part == "." || part == "..") return false;   // never walk out of the mount, nor trust a path that tries
        names.push_back(part);
    }
    std::error_code ec;
    if (!fs::is_directory(root, ec)) return true;   // nothing mounted: no cgroup terms
    dirs.push_back(root);
    fs::path at = root;
    for (const std::string& part : names) {
        at /= part;
        if (!fs::is_directory(at, ec)) break;
        dirs.push_back(at);
    }
    return true;
}

bool scan_cgroups(const MemoryProbePaths& paths, CgroupScan& out) {
    std::optional<std::string> v2_group, v1_group;
    std::ifstream groups(paths.self_cgroup);
    if (!groups) {
        v2_group = "/";   // no /proc/self/cgroup: the mount's own top group is still readable
    } else {
        std::string line;
        while (std::getline(groups, line)) {
            const size_t c1 = line.find(':'), c2 = c1 == std::string::npos ? c1 : line.find(':', c1 + 1);
            if (c2 == std::string::npos) continue;
            const std::string controllers = line.substr(c1 + 1, c2 - c1 - 1), group = line.substr(c2 + 1);
            if (line.compare(0, c1, "0") == 0 && controllers.empty()) { v2_group = group; continue; }
            std::istringstream names(controllers);
            for (std::string name; std::getline(names, name, ',');)
                if (name == "memory") v1_group = group;
        }
    }
    std::vector<fs::path> dirs;
    if (v2_group) {
        if (!group_dirs(paths.cgroup_root, *v2_group, dirs)) return false;
        static const char* const names[] = {"memory.max", "memory.high"};
        if (!scan_levels(dirs, names, 2, "memory.current", false, out)) return false;
    }
    if (!out.any_files && v1_group) {   // cgroup v1: the memory controller's own tree
        if (!group_dirs(fs::path(paths.cgroup_root) / "memory", *v1_group, dirs)) return false;
        static const char* const names[] = {"memory.limit_in_bytes"};
        if (!scan_levels(dirs, names, 1, "memory.usage_in_bytes", true, out)) return false;
    }
    return true;
}

std::optional<HostMemoryReading> combine(std::optional<uint64_t> available, std::optional<uint64_t> total,
                                         const CgroupScan& cg, uint64_t explicit_limit) {
    if (!available) return {};
    HostMemoryReading r;
    r.mem_available = *available;
    r.available = *available;
    r.source = MemorySource::meminfo;
    r.limit = total.value_or(0);
    r.current = total && *total >= *available ? *total - *available : 0;
    for (const CgroupScan::Limit& l : cg.limits) {
        const uint64_t used = l.current - l.credit;   // credit <= current
        const uint64_t room = l.limit > used ? l.limit - used : 0;
        if (room < r.available) {
            r.available = room;
            r.source = MemorySource::cgroup;
            r.limit = l.limit;
            r.current = l.current;
            r.credit = l.credit;
        }
    }
    if (explicit_limit != 0) {
        uint64_t used = 0, credit = 0;
        if (cg.root_current) { used = *cg.root_current; credit = cg.root_credit; }
        else if (total && *total >= *available) used = *total - *available;
        else return {};
        const uint64_t net = used - credit;
        const uint64_t room = explicit_limit > net ? explicit_limit - net : 0;
        if (room <= r.available) {   // a tie goes to the operator's number: it is the one that names the real cap
            r.available = room;
            r.source = MemorySource::flag;
            r.limit = explicit_limit;
            r.current = used;
            r.credit = credit;
        }
    }
    return r;
}

} // namespace

const char* memory_source_name(MemorySource source) {
    switch (source) {
    case MemorySource::flag: return "flag";
    case MemorySource::cgroup: return "cgroup";
    default: return "meminfo";
    }
}

std::optional<uint64_t> conversation_mem_available(std::istream& meminfo) {
    return meminfo_kb_bytes(meminfo, "MemAvailable:");
}

void set_memory_limit_bytes(uint64_t bytes) { g_memory_limit_bytes.store(bytes, std::memory_order_relaxed); }
uint64_t memory_limit_bytes() { return g_memory_limit_bytes.load(std::memory_order_relaxed); }

std::optional<HostMemoryReading> sample_host_memory(const MemoryProbePaths& paths, uint64_t explicit_limit) {
    std::ifstream avail_file(paths.meminfo);
    if (!avail_file) return {};
    const std::optional<uint64_t> available = conversation_mem_available(avail_file);
    if (!available) return {};
    std::ifstream total_file(paths.meminfo);
    std::optional<uint64_t> total;
    if (total_file) total = meminfo_kb_bytes(total_file, "MemTotal:");
    CgroupScan cg;
    if (!scan_cgroups(paths, cg)) {
        // Without an operator cap the cgroup files are only an extra: the guard falls back to MemAvailable alone (what
        // 0.1.41 used), once-warned.  With a cap the usage cannot be told, so the sample stays unknown.
        if (explicit_limit != 0) return {};
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true))
            std::fprintf(stderr, "[strata] memory guard: the cgroup memory files could not be read or parsed, using "
                                 "MemAvailable; set --memory-limit-mib to the container's limit\n");
        cg = CgroupScan{};
    }
    return combine(available, total, cg, explicit_limit);
}

std::optional<HostMemoryReading> sample_host_memory() {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof status;
    // #1607: the commit limit (RAM + page file) refuses an allocation while physical RAM is still free
    if (!GlobalMemoryStatusEx(&status)) return {};
    return combine(std::min<uint64_t>(status.ullAvailPhys, status.ullAvailPageFile), status.ullTotalPhys, CgroupScan{}, memory_limit_bytes());
#elif defined(__linux__)
    return sample_host_memory(MemoryProbePaths{}, memory_limit_bytes());
#else
    return {};
#endif
}

std::optional<uint64_t> available_host_bytes() {
    const auto reading = sample_host_memory();
    if (!reading) return {};
    return reading->available;
}

} // namespace strata::core
