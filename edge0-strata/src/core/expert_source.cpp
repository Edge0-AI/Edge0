// src/core/expert_source.cpp - the adapter.  See the header for the three clauses of the contract.
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include "strata/core/latent_share.hpp"
#include "strata/core/expert_nibble.hpp"
#include "strata/core/foresight_swap.hpp"
#include "strata/core/on_device.hpp"
#include "strata/platform/aux_cpus.hpp"
#include "strata/core/remote_experts.hpp"
#include "strata/core/peer_experts.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/window_limits.hpp"
#include "strata/artifact/gguf_reader.hpp"

#include "strata/core/pinned.hpp"
#include "strata/platform/memory.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/cpu/kq_avx1.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>
#ifndef _WIN32
#include <signal.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX   // std::numeric_limits<T>::max() below
#endif
#include <windows.h>
#else
#include <fcntl.h>
#if defined(__APPLE__)
#include <mach/mach.h>   // host_statistics64: the available memory
#endif
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#if defined(__linux__)
#include <cerrno>
#include <linux/aio_abi.h>   // the kernel's own asynchronous reads (io_submit): no library, allowed by Docker's seccomp
#include <sched.h>
#include <sys/syscall.h>
#endif

// a 64-bit seek (as in pinned.cu): the 32-bit `fseek` wraps past 4 GiB, and the spelling differs per platform
#ifndef STRATA_FSEEK64
#ifdef _WIN32
#define STRATA_FSEEK64(f, o) _fseeki64((f), (long long) (o), SEEK_SET)
#else
#define STRATA_FSEEK64(f, o) fseeko((f), (off_t) (o), SEEK_SET)
#endif
#endif

// The pid of this process and whether a pid still exists (the shared arena's reader election, both OSes)
static uint64_t arena_this_pid() {
#ifdef _WIN32
    return (uint64_t) GetCurrentProcessId();
#else
    return (uint64_t) getpid();
#endif
}
static bool arena_pid_alive(uint64_t pid) {
#ifdef _WIN32
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, (DWORD) pid);
    if (h == nullptr) return GetLastError() == ERROR_ACCESS_DENIED;     // (denied: it exists; invalid parameter: it does not)
    DWORD code = 0;
    const bool alive = GetExitCodeProcess(h, &code) && code == STILL_ACTIVE;
    CloseHandle(h);
    return alive;
#else
    return !(kill((pid_t) pid, 0) != 0 && errno == ESRCH);
#endif
}

namespace strata::core {

namespace detail {

bool cgroup_available_bytes(uint64_t limit, const CgroupMemoryStat& stat, uint64_t& bytes) {
    bytes = 0;
    if (!stat.valid) return false;

    // memory.stat's counters can race memory.current, so bound them to charged usage first.  The whole clean page
    // cache is reclaimable (149: MemAvailable counts active file pages too); the kernel keeps a low-watermark share.
    uint64_t reclaimable = std::min(std::max(stat.file, stat.inactive_file), stat.current);
    reclaimable = stat.file_dirty >= reclaimable ? 0 : reclaimable - stat.file_dirty;
    reclaimable = stat.file_writeback >= reclaimable ? 0 : reclaimable - stat.file_writeback;
    reclaimable -= std::min(reclaimable / 2, limit / 64);

    // Reclaiming clean file pages reduces usage; saturating subtraction also handles a transient over-limit read.
    const uint64_t usage_after_reclaim = stat.current - reclaimable;
    bytes = usage_after_reclaim < limit ? limit - usage_after_reclaim : 0;
    return true;
}

uint64_t clamp_resident_budget(uint64_t requested, uint64_t physical, uint64_t commit, uint64_t headroom) {
    const uint64_t available = std::min(physical, commit);
    const uint64_t room = available > headroom ? available - headroom : 0;
    if (requested <= room) return requested;
    const uint64_t margin = 256ull << 20;
    return room > margin ? room - margin : 0;
}

bool pin_depends_on_reclaim(uint64_t free_now, uint64_t bytes, uint64_t reserve) {
    return bytes > 0 && free_now < bytes + reserve;
}

bool wait_for_ram(uint64_t need, double max_wait_s, double step_s, double note_s,
                  const std::function<bool(uint64_t&)>& sample, const std::function<void(double)>& sleep,
                  const std::function<void(double, uint64_t)>& progress, double& waited, uint64_t& available) {
    waited = 0.0;
    if (!sample(available)) return false;
    double next_note = 0.0;
    while (available < need && waited < max_wait_s) {
        if (progress && waited >= next_note) {
            progress(waited, available);
            next_note = waited + note_s;
        }
        sleep(step_s);
        waited += step_s;
        if (!sample(available)) return false;
    }
    return available >= need;
}

bool pin_step_fits(uint64_t available, uint64_t step, uint64_t reserve) {
    return available >= step + reserve;
}

uint64_t meminfo_bytes(const std::string& meminfo_text, const std::string& key) {
    std::istringstream in(meminfo_text);
    std::string line;
    const std::string want = key + ":";
    while (std::getline(in, line)) {
        std::istringstream fields(line);
        std::string k, unit;
        uint64_t value = 0;
        if (fields >> k >> value && k == want) {
            fields >> unit;
            return unit == "kB" ? value * 1024 : value;
        }
    }
    return 0;
}

uint64_t numa_pin_free(uint64_t global_free, const std::vector<uint64_t>& node_free) {
    if (node_free.size() < 2) return global_free;
    const uint64_t smallest = *std::min_element(node_free.begin(), node_free.end());
    return std::min<uint64_t>(global_free, smallest * (uint64_t) node_free.size());
}

std::array<std::pair<uint64_t, uint64_t>, 3> numa_slice_ranges(uint64_t slot_off, const uint64_t role_bytes[3],
                                                               int slice, uint64_t page) {
    assert(slice == 0 || slice == 1);
    std::array<std::pair<uint64_t, uint64_t>, 3> out{};
    uint64_t start = slot_off;
    for (int r = 0; r < 3; ++r) {
        const uint64_t mid = start + role_bytes[r] / 2, end = start + role_bytes[r];
        if (slice == 0) {
            out[r] = {start / page * page, (mid + page - 1) / page * page};
        } else {
            const uint64_t lo = (mid + page - 1) / page * page, hi = end / page * page;
            out[r] = hi > lo ? std::make_pair(lo, hi) : std::make_pair(lo, lo);
        }
        start = end;
    }
    return out;
}

std::string page_cache_hint(const std::string& meminfo_text) {
    const uint64_t cache = meminfo_bytes(meminfo_text, "Cached") + meminfo_bytes(meminfo_text, "Buffers");
    if (cache < (1ull << 30)) return "";
    char msg[420];
    std::snprintf(msg, sizeof msg,
                  "; %.1f GiB of the RAM is file cache (earlier runs' model files, which Linux gives back on demand - "
                  "'free' showing little is normal). If the start still fails, free it with: "
                  "sync; echo 1 | sudo tee /proc/sys/vm/drop_caches   (or start strata serve with "
                  "STRATA_DROP_CACHE_ON_EXIT=1 so it releases its model files when it stops)",
                  (double) cache / 1073741824.0);
    return msg;
}

bool make_cache_complement_plan(
    int64_t n_layers, int64_t n_expert, const std::vector<uint64_t>& layer_blob_bytes,
    const std::vector<std::pair<int32_t, int32_t>>& primary_gpu_pairs,
    const std::vector<std::pair<int32_t, int32_t>>& additional_gpu_pairs,
    std::vector<uint64_t>& offsets, uint64_t& bytes, std::string& err) {
    offsets.clear();
    bytes = 0;
    err.clear();
    if (n_layers <= 0 || n_expert <= 0 || layer_blob_bytes.size() != (size_t) n_layers) {
        err = "FileExpertSource: invalid geometry for the cache complement plan";
        return false;
    }
    if ((uint64_t) n_layers > (uint64_t) std::numeric_limits<size_t>::max() / (uint64_t) n_expert) {
        err = "FileExpertSource: cache complement index table is too large";
        return false;
    }
    const size_t count = (size_t) n_layers * (size_t) n_expert;
    std::vector<uint8_t> omitted(count, 0);
    auto mark_pairs = [&](const std::vector<std::pair<int32_t, int32_t>>& pairs, uint8_t bit,
                          const char* label) -> bool {
        for (const auto& pair : pairs) {
            if (pair.first < 0 || pair.second < 0 || pair.first >= n_layers || pair.second >= n_expert) {
                err = std::string("FileExpertSource: ") + label + " pair is outside the expert geometry";
                return false;
            }
            const size_t index = (size_t) pair.first * (size_t) n_expert + (size_t) pair.second;
            if ((omitted[index] & bit) != 0) {
                err = std::string("FileExpertSource: duplicate ") + label + " pair in the cache complement plan";
                return false;
            }
            if (bit == 2 && (omitted[index] & 1) != 0) {
                err = "FileExpertSource: the primary and additional GPU expert tiers overlap";
                return false;
            }
            omitted[index] |= bit;
        }
        return true;
    };
    if (!mark_pairs(primary_gpu_pairs, 1, "primary GPU") ||
        !mark_pairs(additional_gpu_pairs, 2, "additional GPU")) return false;
    for (uint64_t blob_bytes : layer_blob_bytes) {
        if (blob_bytes == 0) {
            err = "FileExpertSource: cache complement layer has zero-sized expert blobs";
            return false;
        }
    }

    offsets.assign(count, kNoCacheComplement);
    for (int64_t layer = 0; layer < n_layers; ++layer) {
        const uint64_t blob_bytes = layer_blob_bytes[(size_t) layer];
        for (int64_t expert = 0; expert < n_expert; ++expert) {
            const size_t index = (size_t) layer * (size_t) n_expert + (size_t) expert;
            if (omitted[index] != 0) continue;
            if (bytes > std::numeric_limits<uint64_t>::max() - blob_bytes) {
                offsets.clear();
                bytes = 0;
                err = "FileExpertSource: cache complement size overflows";
                return false;
            }
            offsets[index] = bytes;
            bytes += blob_bytes;
        }
    }
    if (bytes > (uint64_t) std::numeric_limits<size_t>::max()) {
        offsets.clear();
        bytes = 0;
        err = "FileExpertSource: cache complement exceeds the host address space";
        return false;
    }
    return true;
}

const uint8_t* cache_complement_blob_or_fallback(
    size_t index, const std::vector<uint64_t>& offsets, const uint8_t* complement_host,
    const uint8_t* mapped_fallback) {
    if (complement_host != nullptr && index < offsets.size() && offsets[index] != kNoCacheComplement)
        return complement_host + (size_t) offsets[index];
    return mapped_fallback;
}

int64_t choose_resident_keep_from(const std::vector<uint64_t>& slot_bytes, uint64_t base_bytes, uint64_t budget,
                                  int64_t lend_from) {
    if (base_bytes > budget) return -1;
    const int64_t slots = (int64_t) slot_bytes.size();
    if (lend_from < 0 || lend_from > slots) lend_from = slots;   // no lend region: only the experts no slot holds
    int64_t keep = slots;
    uint64_t bytes = base_bytes;
    while (keep > lend_from) {
        const uint64_t b = slot_bytes[(size_t) keep - 1];
        if (b > budget - bytes) break;
        bytes += b;
        --keep;
    }
    return keep;
}

int64_t append_stage_lend_regions(int64_t n_layers, int64_t n_expert, const std::vector<uint64_t>& layer_blob_bytes,
                                  std::vector<uint64_t>& offsets, uint64_t& bytes, uint64_t cap_bytes,
                                  const std::vector<std::vector<std::pair<int32_t, int32_t>>>& regions,
                                  std::string& err) {
    const size_t blobs = (size_t) (n_layers * n_expert);
    if (n_layers <= 0 || n_expert <= 0 || layer_blob_bytes.size() != (size_t) n_layers || offsets.size() != blobs) {
        err = "append_stage_lend_regions: the plan does not match the geometry";
        return -1;
    }
    std::vector<std::pair<int32_t, int32_t>> sorted;
    for (const auto& region : regions) {   // validate the whole request before touching the plan
        for (const auto& pair : region)
            if (pair.first < 0 || pair.first >= n_layers || pair.second < 0 || pair.second >= n_expert) {
                err = "append_stage_lend_regions: a lend-region expert is outside the model";
                return -1;
            }
        sorted = region;
        std::sort(sorted.begin(), sorted.end());
        for (size_t i = 1; i < sorted.size(); ++i)
            if (sorted[i] == sorted[i - 1]) {
                err = "append_stage_lend_regions: a stage's lend region repeats an expert";
                return -1;
            }
    }
    int64_t kept = 0;
    for (const auto& region : regions) {
        for (const auto& pair : region) {
            const size_t index = (size_t) pair.first * (size_t) n_expert + (size_t) pair.second;
            if (offsets[index] != kNoCacheComplement) continue;   // already in the copy: the complement, or a
                                                                  // CUDA0 lend slot the region is asked to skip
            const uint64_t b = layer_blob_bytes[(size_t) pair.first];
            if (b > cap_bytes - bytes) break;                     // this stage's rest keeps the file fallback
            offsets[index] = bytes;
            bytes += b;
            ++kept;
        }
    }
    return kept;
}

bool exchange_cache_complement(std::vector<uint64_t>& offsets, size_t in, size_t out) {
    if (in == out || in >= offsets.size() || out >= offsets.size() || offsets[in] == kNoCacheComplement ||
        offsets[out] != kNoCacheComplement) return false;
    offsets[out] = offsets[in];
    offsets[in] = kNoCacheComplement;
    return true;
}

}  // namespace detail

namespace {

#if defined(__linux__)
bool read_cgroup_memory_stat(const std::filesystem::path& path, uint64_t current,
                             detail::CgroupMemoryStat& stat) {
    std::ifstream input(path / "memory.stat");
    if (!input) return false;

    bool inactive_file = false, file_dirty = false, file_writeback = false, file = false;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream fields(line);
        std::string key;
        uint64_t value = 0;
        if (!(fields >> key >> value)) return false;
        fields >> std::ws;
        if (!fields.eof()) return false;

        if (key == "inactive_file") {
            if (inactive_file) return false;
            inactive_file = true;
            stat.inactive_file = value;
        } else if (key == "file_dirty") {
            if (file_dirty) return false;
            file_dirty = true;
            stat.file_dirty = value;
        } else if (key == "file_writeback") {
            if (file_writeback) return false;
            file_writeback = true;
            stat.file_writeback = value;
        } else if (key == "file") {   // 149: optional (every v2 kernel has it); 0 = inactive_file alone
            if (file) return false;
            file = true;
            stat.file = value;
        }
    }
    if (!input.eof() || !inactive_file || !file_dirty || !file_writeback) return false;
    stat.current = current;
    stat.valid = true;
    return true;
}
#endif

}  // namespace

namespace detail {

bool host_available_memory(HostMemory& m, const std::string& meminfo, const std::string& self_cgroup,
                           const std::string& cgroup_root) {
    m = HostMemory{};
#if defined(_WIN32)
    (void) meminfo; (void) self_cgroup; (void) cgroup_root;
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) return false;
    m.available = (uint64_t) status.ullAvailPhys;
    m.commit = (uint64_t) status.ullAvailPageFile;
    return m.available > 0;
#elif defined(__linux__)
    // MemAvailable includes reclaimable page cache, unlike _SC_AVPHYS_PAGES.
    std::ifstream info(meminfo);
    std::string line;
    uint64_t bytes = 0;
    while (std::getline(info, line)) {
        std::istringstream fields(line);
        std::string key, unit;
        uint64_t value = 0;
        if (fields >> key >> value >> unit && key == "MemAvailable:" && unit == "kB" &&
            value <= std::numeric_limits<uint64_t>::max() / 1024) bytes = value * 1024;
    }
    if (bytes == 0) return false;
    // Account for the tightest cgroup ancestor limit when its normal mount is visible.
    // This is a point-in-time guard, not a reservation against concurrent allocations.
    std::ifstream groups(self_cgroup);
    if (!groups) return false;
    const std::filesystem::path root(cgroup_root);
    bool v2 = false;
    std::string v1_path;   // the memory controller's group (cgroup v1), when there is no v2 line
    while (std::getline(groups, line)) {
        if (line.rfind("0::/", 0) != 0) {
            // v1: "N:controller[,controller]:/path"
            const size_t c1 = line.find(':'), c2 = c1 == std::string::npos ? c1 : line.find(':', c1 + 1);
            if (c2 == std::string::npos) continue;
            std::istringstream ctl(line.substr(c1 + 1, c2 - c1 - 1));
            for (std::string c; std::getline(ctl, c, ',');)
                if (c == "memory") v1_path = line.substr(c2 + 1);
            continue;
        }
        auto path = (root / line.substr(4)).lexically_normal();
        if (path.string().rfind(root.string(), 0) != 0 || !std::filesystem::is_directory(path)) return false;
        v2 = true;
        while (path.string().rfind(root.string(), 0) == 0) {
            std::ifstream limit_file(path / "memory.max"), current_file(path / "memory.current");
            std::string limit;
            uint64_t current = 0;
            const bool readable = bool(limit_file >> limit) && bool(current_file >> current);
            // The host's root cgroup has no memory.max; ordinary child groups must expose their limits.
            if (!readable && !(path == root && !std::filesystem::exists(path / "memory.max") &&
                               std::filesystem::exists(path / "cgroup.controllers"))) return false;
            if (readable && limit != "max") {
                try {
                    size_t consumed = 0;
                    const uint64_t cap = std::stoull(limit, &consumed);
                    if (consumed != limit.size()) return false;
                    CgroupMemoryStat stat;
                    if (!read_cgroup_memory_stat(path, current, stat)) return false;
                    uint64_t cgroup_available = 0;
                    if (!cgroup_available_bytes(cap, stat, cgroup_available)) return false;
                    bytes = std::min(bytes, cgroup_available);
                    m.cgroup_limit = std::min(m.cgroup_limit, cap);
                } catch (...) { return false; }
            }
            if (path == root) break;
            path = path.parent_path();
        }
    }
    if (!v2 && !v1_path.empty()) {
        // #633: cgroup v1 (older Docker hosts, RHEL 7/8): the memory controller's group and its ancestors.  An
        // unlimited group says a number near 2^63 (rounded to its page size); a group whose files are not
        // visible (no mount in this namespace) is skipped - MemAvailable alone, as with no cgroup at all.
        const std::filesystem::path mroot = root / "memory";
        auto path = (mroot / v1_path.substr(v1_path.rfind('/', 0) == 0 ? 1 : 0)).lexically_normal();
        while (path.string().rfind(mroot.string(), 0) == 0) {
            std::ifstream limit_file(path / "memory.limit_in_bytes"), usage_file(path / "memory.usage_in_bytes");
            uint64_t cap = 0, usage = 0;
            if (limit_file >> cap && usage_file >> usage && cap < (1ull << 62)) {
                bytes = std::min(bytes, usage < cap ? cap - usage : 0);
                m.cgroup_limit = std::min(m.cgroup_limit, cap);
            }
            if (path == mroot) break;
            path = path.parent_path();
        }
    }
    m.available = bytes;
    return true;
#elif defined(__APPLE__)
    // macOS: free + inactive + purgeable pages (what the OS can hand out without swapping), as vm_stat reports them
    (void) meminfo; (void) self_cgroup; (void) cgroup_root;
    vm_statistics64_data_t vs{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_size_t page_bytes = 0;
    if (host_page_size(mach_host_self(), &page_bytes) != KERN_SUCCESS ||
        host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &vs, &count) != KERN_SUCCESS) return false;
    m.available = ((uint64_t) vs.free_count + vs.inactive_count + vs.purgeable_count) * (uint64_t) page_bytes;
    return m.available > 0;
#else
    (void) meminfo; (void) self_cgroup; (void) cgroup_root;
    const long pages = sysconf(_SC_AVPHYS_PAGES);
    const long page_bytes = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_bytes <= 0 ||
        (uint64_t) pages > std::numeric_limits<uint64_t>::max() / (uint64_t) page_bytes) return false;
    m.available = (uint64_t) pages * (uint64_t) page_bytes;
    return m.available > 0;
#endif
}

}  // namespace detail

namespace {

bool available_memory_bytes(uint64_t& bytes, uint64_t* commit = nullptr) {
    detail::HostMemory m;
    if (!detail::host_available_memory(m)) return false;
    bytes = m.available;
    if (commit != nullptr) *commit = m.commit;
    return bytes > 0;
}

/// #1250: the pages that are really free (MemFree), not the reclaimable cache MemAvailable counts; false when unknown.
bool free_memory_bytes(uint64_t& bytes) {
#if defined(__linux__)
    std::ifstream info("/proc/meminfo");
    if (!info) return false;
    std::stringstream text;
    text << info.rdbuf();
    bytes = detail::meminfo_bytes(text.str(), "MemFree");
    if (bytes == 0) return false;
    // 0.1.42: a multi-socket host's nodes each have their own free pages; judge by the emptiest one too
    std::vector<uint64_t> nodes;
    for (int n = 0; n < 64; ++n) {
        std::ifstream node("/sys/devices/system/node/node" + std::to_string(n) + "/meminfo");
        if (!node) break;
        std::string line;
        while (std::getline(node, line)) {
            std::istringstream f(line);
            std::string a, id, key;
            uint64_t kb = 0;
            if (f >> a >> id >> key >> kb && key == "MemFree:") { nodes.push_back(kb * 1024); break; }
        }
    }
    bytes = detail::numa_pin_free(bytes, nodes);
    return bytes > 0;
#else
    bytes = 0;
    return false;
#endif
}

void log_complement_memory(const char* stage, uint64_t requested) {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) {
        std::fprintf(stderr, "FileExpertSource: %s: GlobalMemoryStatusEx failed (error %lu)\n", stage, GetLastError());
        return;
    }
    std::fprintf(stderr, "FileExpertSource: %s: requested %.2f GiB, available RAM %.2f GiB, available commit %.2f GiB "
                         "(limit %.2f GiB), available virtual address space %.2f GiB\n",
                 stage, (double) requested / 1073741824.0, (double) status.ullAvailPhys / 1073741824.0,
                 (double) status.ullAvailPageFile / 1073741824.0, (double) status.ullTotalPageFile / 1073741824.0,
                 (double) status.ullAvailVirtual / 1073741824.0);
#else
    (void) stage;
    (void) requested;
#endif
}

#if defined(__linux__)
/// STRATA_NUMA_SPLIT step B: each NUMA slice of every complement slot (numa_slice_ranges) is first-touched by a helper
/// thread pinned to one core of that slice's node, so the default local policy puts its pages there (no mbind: ~113k
/// ranges would split the arena into more VMAs than vm.max_map_count allows).  Inactive (`place` false) unless the
/// live pool split on real nodes; then the paced fault loop hands it each step to touch.
struct NumaComplementPlacement {
    bool place = false;
    int nodes[2] = {-1, -1};
    int core[2] = {-1, -1};
    std::vector<std::pair<uint64_t, uint64_t>> ranges[2];   ///< page-aligned, sorted, coalesced, arena offsets
    uint64_t slice_bytes[2] = {0, 0};
    std::vector<uint64_t> free_before;                      ///< MemFree per node before the allocation
    std::vector<uint64_t> free_after;                       ///< the same after the model files' pages are released
    std::atomic<bool> pin_failed{false};
    std::chrono::steady_clock::time_point t0;

    static std::vector<uint64_t> node_free() {
        // CBA: stops at the first missing node directory (as free_memory_bytes does), sparse node ids give a false
        // "at risk 0.00" line only; list the node directory if a sparse-id host ever shows it
        std::vector<uint64_t> out;
        for (int n = 0; n < 64; ++n) {
            std::ifstream node("/sys/devices/system/node/node" + std::to_string(n) + "/meminfo");
            if (!node) break;
            std::string line;
            uint64_t kb = 0;
            while (std::getline(node, line)) {
                std::istringstream f(line);
                std::string a, id, key;
                if (f >> a >> id >> key >> kb && key == "MemFree:") break;
                kb = 0;
            }
            out.push_back(kb * 1024);
        }
        return out;
    }
    static std::string free_text(const std::vector<uint64_t>& free) {
        std::string s;
        char buf[64];
        for (size_t n = 0; n < free.size(); ++n) {
            std::snprintf(buf, sizeof buf, "%snode %zu %.2f GiB", n ? ", " : "", n, (double) free[n] / 1073741824.0);
            s += buf;
        }
        return s;
    }

    /// Decides from the live pool's plan; false with a logged reason when the split is on but placement cannot run.
    void init(const std::vector<uint64_t>& offsets, int64_t n_expert, const std::vector<uint64_t>& blob_bytes,
              const std::vector<uint64_t>& role_bytes) {
        // relies on the pool being built first (generate.cpp: the ExpertPool at ~5632, the complement at ~7690)
        const auto& plan = strata::kernels::cpu::active_numa_split();
        if (!plan.on || plan.test) return;
        free_before = node_free();
        const char* why = nullptr;
        if (role_bytes.size() != 3 * blob_bytes.size()) why = "this pack has no per-role byte sizes (not a GGUF-mapped native pack)";
        for (size_t l = 0; why == nullptr && l < blob_bytes.size(); ++l)
            if (role_bytes[3 * l] + role_bytes[3 * l + 1] + role_bytes[3 * l + 2] != blob_bytes[l])
                why = "the role sizes do not add up to the expert blob";
        for (int k = 0; why == nullptr && k < 2; ++k)
            if (plan.cores[k].empty()) why = "a slice has no core";
            else if (plan.nodes[k] < 0 || plan.nodes[k] >= 64) why = "a node id is out of range";
        if (why != nullptr) {
            std::fprintf(stderr, "FileExpertSource: WARNING: NUMA placement FAILED: %s; the complement is placed by "
                                 "first touch as usual (the split runs with remote reads)\n", why);
            return;
        }
        for (int k = 0; k < 2; ++k) {
            nodes[k] = plan.nodes[k];
            core[k] = plan.cores[k].front();   // workers come before the host core in the plan
            std::vector<std::pair<uint64_t, uint64_t>> all;
            for (size_t i = 0; i < offsets.size(); ++i) {
                if (offsets[i] == detail::kNoCacheComplement) continue;
                const size_t l = i / (size_t) n_expert;
                for (const auto& r : detail::numa_slice_ranges(offsets[i], &role_bytes[3 * l], k))
                    if (r.second > r.first) all.push_back(r);
            }
            std::sort(all.begin(), all.end());
            for (const auto& r : all) {   // coalesce: ends become monotonic, so a step finds its first range by bisection
                if (!ranges[k].empty() && r.first <= ranges[k].back().second)
                    ranges[k].back().second = std::max(ranges[k].back().second, r.second);
                else ranges[k].push_back(r);
            }
            for (const auto& r : ranges[k]) slice_bytes[k] += r.second - r.first;
        }
        place = true;
        t0 = std::chrono::steady_clock::now();
        std::fprintf(stderr, "FileExpertSource: NUMA placement of the complement: slice 0 -> node %d (%.2f GiB, toucher "
                             "CPU %d), slice 1 -> node %d (%.2f GiB, toucher CPU %d), 4 KiB pages; free before: %s\n",
                     nodes[0], (double) slice_bytes[0] / 1073741824.0, core[0], nodes[1],
                     (double) slice_bytes[1] / 1073741824.0, core[1], free_text(free_before).c_str());
    }

    /// Touches one byte per page of each slice's ranges clipped to [lo, hi), one pinned thread per slice; then the
    /// calling thread touches the step's other pages (outside every slice: slot tails, the arena end).  The caller
    /// must not re-touch slice pages: with automatic NUMA balancing on, its access faults would migrate them to its node.
    void touch_step(uint8_t* arena, uint64_t lo, uint64_t hi) {
        // CBA: two threads spawned per 1 GiB step (~50 spawns per start); long-lived touchers if the spawn ever shows
        auto touch = [&](int k) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(core[k], &set);
            if (sched_setaffinity(0, sizeof set, &set) != 0) pin_failed = true;
            const auto& rs = ranges[k];
            auto it = std::partition_point(rs.begin(), rs.end(), [&](const auto& r) { return r.second <= lo; });
            for (; it != rs.end() && it->first < hi; ++it)
                for (uint64_t at = std::max(it->first, lo); at < std::min(it->second, hi); at += 4096)
                    ((volatile uint8_t*) arena)[at] = 0;
        };
        // CBA: a failed spawn terminates; fall back to an unpinned touch if it ever happens
        std::thread a(touch, 0), b(touch, 1);
        a.join();
        b.join();
        size_t c[2] = {0, 0};   // the pages no slice holds, as the default loop touches them
        for (uint64_t at = lo; at < hi; at += 4096) {
            bool covered = false;
            for (int k = 0; k < 2; ++k) {
                while (c[k] < ranges[k].size() && ranges[k][c[k]].second <= at) ++c[k];
                covered = covered || (c[k] < ranges[k].size() && ranges[k][c[k]].first <= at);
            }
            if (!covered) ((volatile uint8_t*) arena)[at] = 0;
        }
    }

    /// After the drop: a node whose free memory cannot take its slice plus `reserve_share` will spill to the other node
    /// (no local reclaim); named up front, the 95 % check stays the verdict.
    void check_room(uint64_t reserve_share) const {
        for (int k = 0; k < 2; ++k) {
            const uint64_t room = (size_t) nodes[k] < free_after.size() ? free_after[(size_t) nodes[k]] : 0;
            if (room < slice_bytes[k] + reserve_share)
                std::fprintf(stderr, "FileExpertSource: WARNING: NUMA placement at risk: node %d has %.2f GiB free for its "
                                     "%.2f GiB slice plus %.2f GiB of reserve\n",
                             nodes[k], (double) room / 1073741824.0, (double) slice_bytes[k] / 1073741824.0,
                             (double) reserve_share / 1073741824.0);
        }
    }

    /// Samples one page in 64 of each slice with move_pages (null node list: returns each page's node) and logs the
    /// share on the intended node, once the arena is registered (its final state); `got < bytes` means the paced loop
    /// stopped early, `registered < bytes` that the driver did not page-lock all of it: the placement then fails.
    void report(const uint8_t* arena, uint64_t got, uint64_t registered, uint64_t bytes, const char* stop) const {
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        uint64_t on_node[2] = {0, 0}, sampled[2] = {0, 0}, per_node[2][64] = {}, absent[2] = {0, 0};
        constexpr int kNotSampled = -100000;   // outside any per-page status (a node id, or -errno)
        int sys_errno[2] = {0, 0};              // first errno of a failed move_pages batch, per slice
        size_t batches[2] = {0, 0}, failed[2] = {0, 0};
        std::vector<void*> pages;
        std::vector<int> status;
        for (int k = 0; k < 2; ++k) {
            uint64_t n = 0;
            pages.clear();
            for (const auto& r : ranges[k])
                for (uint64_t at = r.first; at < r.second; at += 4096, ++n)
                    if (n % 64 == 0) pages.push_back((void*) (arena + at));
            status.assign(pages.size(), -1);
            for (size_t b = 0; b < pages.size(); b += 4096) {
                const size_t cnt = std::min<size_t>(4096, pages.size() - b);
                ++batches[k];
                if (syscall(SYS_move_pages, 0, (unsigned long) cnt, pages.data() + b, nullptr, status.data() + b, 0) != 0) {
                    if (sys_errno[k] == 0) sys_errno[k] = errno;
                    ++failed[k];
                    for (size_t i = b; i < b + cnt; ++i) status[i] = kNotSampled;   // the syscall failed: no sample, unlike a page's own -errno
                }
            }
            for (int s : status) {
                if (s == kNotSampled) continue;
                ++sampled[k];
                if (s >= 0 && s < 64) ++per_node[k][s];
                else ++absent[k];
            }
            on_node[k] = nodes[k] >= 0 && nodes[k] < 64 ? per_node[k][nodes[k]] : 0;
        }
        double share[2];
        for (int k = 0; k < 2; ++k) share[k] = sampled[k] > 0 ? 100.0 * (double) on_node[k] / (double) sampled[k] : 0.0;
        char what[2][96];   // a slice whose move_pages batches all failed has no share to print
        for (int k = 0; k < 2; ++k) {
            if (batches[k] > 0 && failed[k] == batches[k]) std::snprintf(what[k], sizeof what[k], "not verified");
            else std::snprintf(what[k], sizeof what[k], "%.2f GiB (%.1f %% of its slices)",
                               share[k] / 100.0 * (double) slice_bytes[k] / 1073741824.0, share[k]);
        }
        std::fprintf(stderr, "FileExpertSource: NUMA placement: node %d %s, node %d %s; free before: %s; after the drop: %s; "
                             "pages taken and page-locked in %.1f s since the placement started\n",
                     nodes[0], what[0], nodes[1], what[1], free_text(free_before).c_str(), free_text(free_after).c_str(), secs);
        if (got < bytes)
            std::fprintf(stderr, "FileExpertSource: WARNING: NUMA placement FAILED: the paced loop stopped at %.2f of %.2f "
                                 "GiB (%s); the rest is first-touched by unpinned copy threads (memory not placed on its node: the split runs with remote reads)\n",
                         (double) got / 1073741824.0, (double) bytes / 1073741824.0, stop != nullptr ? stop : "?");
        else if (registered < bytes)
            std::fprintf(stderr, "FileExpertSource: WARNING: NUMA placement FAILED: only %.2f of %.2f GiB page-locked (%s); "
                                 "the unregistered pages can be migrated by NUMA balancing (memory not placed on its node: the split runs with remote reads)\n",
                         (double) registered / 1073741824.0, (double) bytes / 1073741824.0, stop != nullptr ? stop : "?");
        if (pin_failed)
            std::fprintf(stderr, "FileExpertSource: WARNING: NUMA placement: a toucher could not be pinned to its core\n");
        for (int k = 0; k < 2; ++k) {
            if (batches[k] > 0 && failed[k] == batches[k]) {
                std::fprintf(stderr, "FileExpertSource: WARNING: NUMA placement could not be verified: move_pages failed "
                                     "(errno %d: %s) for slice %d (node %d)\n", sys_errno[k], std::strerror(sys_errno[k]), k, nodes[k]);
                continue;
            }
            if (failed[k] > 0)
                std::fprintf(stderr, "FileExpertSource: WARNING: NUMA placement: move_pages failed (errno %d: %s) for %zu of %zu "
                                     "batches of slice %d; judged on the other samples\n", sys_errno[k],
                             std::strerror(sys_errno[k]), failed[k], batches[k], k);
            if (share[k] < 95.0)
                std::fprintf(stderr, "FileExpertSource: WARNING: NUMA placement FAILED: slice %d has %.1f %% of its pages "
                                     "on node %d (< 95 %%; of %llu sampled: node %d %llu, node %d %llu, not present %llu)\n",
                             k, share[k], nodes[k], (unsigned long long) sampled[k], nodes[0],
                             (unsigned long long) per_node[k][nodes[0]], nodes[1], (unsigned long long) per_node[k][nodes[1]],
                             (unsigned long long) absent[k]);
        }
    }
};
#endif

/// #286: an unbuffered file handle - a HANDLE on Windows, an O_DIRECT descriptor (as a pointer-sized integer) on Linux.
void close_direct(void* h) {
#if defined(_WIN32)
    CloseHandle((HANDLE) h);
#else
    ::close((int) (intptr_t) h);
#endif
}

#if defined(__linux__)
/// #3237: how fast this drive serves the experts' reads the two ways the file tier can take them - O_DIRECT preads of
/// 1 MiB at random places on 8 threads (the unbuffered tier's I/O threads) against first touches of the pages of random
/// 1 MiB pieces of a mapping on 3 threads (the cached tier: the pool's threads faulting the page cache in).  Regions the
/// page cache already holds are skipped, and the probed ranges are dropped from the cache again.  The best of two
/// rounds each; a few hundred MiB read, under two seconds per pass.  Done once per process.  false: not measurable.
struct TierProbeResult { bool done = false, ok = false; double direct_bps = 0, cached_bps = 0; };

bool file_tier_probe(const std::vector<std::string>& paths, double& direct_bps, double& cached_bps) {
    static std::mutex mu;
    static TierProbeResult memo;
    std::lock_guard<std::mutex> lk(mu);
    if (memo.done) { direct_bps = memo.direct_bps; cached_bps = memo.cached_bps; return memo.ok; }
    memo.done = true;
    const std::string* best = nullptr;
    uint64_t size = 0;
    for (const std::string& p : paths) {
        struct stat st{};
        if (::stat(p.c_str(), &st) == 0 && (uint64_t) st.st_size > size) { size = (uint64_t) st.st_size; best = &p; }
    }
    constexpr uint64_t kRegion = 1ull << 20;
    constexpr int kPerPass = 96, kRounds = 2;
    constexpr int kDirectThreads = 8, kCachedThreads = 3;
    if (best == nullptr || size < 8ull * kPerPass * kRegion * 2) return false;
    const int fd_d = ::open(best->c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    const int fd_b = ::open(best->c_str(), O_RDONLY | O_CLOEXEC);
    void* map = fd_b >= 0 ? mmap(nullptr, (size_t) size, PROT_READ, MAP_SHARED, fd_b, 0) : MAP_FAILED;
    auto cleanup = [&] {
        if (map != MAP_FAILED) munmap(map, (size_t) size);
        if (fd_d >= 0) ::close(fd_d);
        if (fd_b >= 0) ::close(fd_b);
    };
    if (fd_d < 0 || map == MAP_FAILED) { cleanup(); return false; }
    const uint8_t* base = static_cast<const uint8_t*>(map);
    uint64_t seed = (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count() * 6364136223846793005ull + 1442695040888963407ull;
    std::vector<uint64_t> used;
    auto pick = [&](std::vector<uint64_t>& out) {   // kPerPass distinct 1 MiB pieces the page cache does not hold
        out.clear();
        std::vector<unsigned char> vec(kRegion / 4096);
        for (int tries = 0; (int) out.size() < kPerPass && tries < kPerPass * 40; ++tries) {
            seed = seed * 6364136223846793005ull + 1442695040888963407ull;
            const uint64_t off = ((seed >> 20) % (size / kRegion)) * kRegion;
            if (std::find(used.begin(), used.end(), off) != used.end()) continue;
            if (mincore((void*) (base + off), (size_t) kRegion, vec.data()) != 0) continue;
            int res = 0;
            for (unsigned char c : vec) res += c & 1;
            if (res * 4 > (int) vec.size()) continue;
            used.push_back(off);
            out.push_back(off);
        }
        return (int) out.size() >= kPerPass / 2;
    };
    auto preadn = [&](uint8_t* dst, uint64_t off) {
        uint64_t done = 0;
        while (done < kRegion) {
            const ssize_t r = ::pread(fd_d, dst + done, (size_t) (kRegion - done), (off_t) (off + done));
            if (r < 0) { if (errno == EINTR) continue; return false; }
            if (r == 0) return false;
            done += (uint64_t) r;
        }
        return true;
    };
    double best_d = 0, best_c = 0;
    bool ok = true;
    for (int round = 0; round < kRounds && ok; ++round) {
        for (int mode = 0; mode < 2 && ok; ++mode) {   // 0: direct, 1: cached
            std::vector<uint64_t> offs;
            if (!pick(offs)) { ok = false; break; }
            std::atomic<size_t> next{0};
            std::atomic<uint64_t> bytes{0};
            std::atomic<int> failed{0};
            const auto t0 = std::chrono::steady_clock::now();
            const auto deadline = t0 + std::chrono::milliseconds(1500);
            std::vector<std::thread> th;
            const int nthreads = mode == 0 ? kDirectThreads : kCachedThreads;
            for (int t = 0; t < nthreads; ++t) {
                th.emplace_back([&] {
                    void* buf = nullptr;
                    if (mode == 0 && posix_memalign(&buf, 4096, kRegion) != 0) { failed = 1; return; }
                    volatile uint8_t sink = 0;
                    for (;;) {
                        const size_t i = next.fetch_add(1);
                        if (i >= offs.size() || std::chrono::steady_clock::now() > deadline) break;
                        if (mode == 0) {
                            if (!preadn((uint8_t*) buf, offs[i])) { failed = 1; break; }
                        } else {
                            for (uint64_t pg = 0; pg < kRegion; pg += 4096) sink = (uint8_t) (sink + base[offs[i] + pg]);
                        }
                        bytes.fetch_add(kRegion);
                    }
                    std::free(buf);
                });
            }
            for (std::thread& x : th) x.join();
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (failed.load() != 0 || bytes.load() == 0 || secs <= 0) { ok = false; break; }
            const double bps = (double) bytes.load() / secs;
            (mode == 0 ? best_d : best_c) = std::max(mode == 0 ? best_d : best_c, bps);
            for (uint64_t off : offs) {   // leave the page cache as it was
                (void) madvise((void*) (base + off), (size_t) kRegion, MADV_DONTNEED);
                (void) posix_fadvise(fd_b, (off_t) off, (off_t) kRegion, POSIX_FADV_DONTNEED);
            }
        }
    }
    cleanup();
    memo.ok = ok && best_d > 0 && best_c > 0;
    memo.direct_bps = best_d;
    memo.cached_bps = best_c;
    direct_bps = best_d;
    cached_bps = best_c;
    return memo.ok;
}
#endif

/// #286/#577: whether the file tier reads its `read_bytes` unbuffered, beside `arena_bytes` of RAM copy still to be
/// taken.  Windows times probe reads too (experts_unbuffered); on Linux the same rule as its `cache_counts = false`
/// arm decides: whether the file cache could keep those bytes at all, from MemAvailable and the tightest cgroup
/// limit (host_available_memory).  STRATA_UNBUFFERED_LOAD=1 / 0 forces it either way, as on Windows.
bool file_tier_unbuffered(const std::vector<std::string>& paths, uint64_t arena_bytes, uint64_t read_bytes,
                          std::string& why) {
#if defined(_WIN32)
    return experts_unbuffered(paths, arena_bytes, why, /*cache_counts=*/false, read_bytes);
#elif defined(__linux__)
    (void) paths;
    if (const char* env = std::getenv("STRATA_UNBUFFERED_LOAD"); env != nullptr && env[0] != '\0') {
        why = std::string("STRATA_UNBUFFERED_LOAD=") + env;
        return env[0] != '0';
    }
    uint64_t avail = 0;
    if (!available_memory_bytes(avail)) {
        why = "the available RAM is unknown";
        return false;
    }
    // #1194: the experts read from the files are a skewed set: a cache holding a twentieth of them (at least 1.5 GiB)
    // serves most of the repeats, and beat the unbuffered reads at every room measured (it reads 2 to 7 times less from
    // the drive).  `avail` already has what this process and the cgroup hold taken off, so 1 GiB is left for the rest.
    // #1752 (KevinX8): ROCm queues are suspended while the kernel reclaims around the process's HMM-tracked arena
    // (#1691) on an MI50 (gfx906): the cached reads churn the page cache and reclaim unmaps the arena, so the gfx906
    // backend keeps the old room >= read_bytes rule.  Other HIP cards (gfx1201 measured: not affected) keep the new
    // rule; STRATA_FILE_CACHE_OLD_RULE=1 / 0 forces the old rule on / off on any Linux build.
    bool old_rule = false;
#if defined(STRATA_HIP_GFX906)
    old_rule = true;
#endif
    if (const char* r = std::getenv("STRATA_FILE_CACHE_OLD_RULE"); r != nullptr && r[0] != '\0') old_rule = r[0] != '0';
    const bool keepable = old_rule
        ? strata::platform::file_cache_keeps(avail, arena_bytes, read_bytes)
        : strata::platform::file_cache_keeps(avail, arena_bytes, read_bytes, 1ull << 30, 0.05, 3ull << 29);
    char msg[256];
    std::snprintf(msg, sizeof msg, "%.1f GiB available, %.1f GiB of it still to be taken by the RAM copy, %.1f GiB "
                  "of experts read from the files: the file cache %s keep the ones that come back",
                  (double) avail / (1ull << 30), (double) arena_bytes / (1ull << 30), (double) read_bytes / (1ull << 30),
                  keepable ? "can" : "cannot");
    why = msg;
    // #3237: the room rule says the cache can keep the experts that come back, but when it holds under half of the read
    // set, whether it pays depends on the drive: on a PCIe 5 NVMe (8 GB/s direct) the cached path read prompts at half
    // the speed and decoded 3-4x slower than unbuffered reads, on a 1.7-2.5 GB/s NVMe the cache decodes faster.  So time
    // both paths once at start and read unbuffered only when the drive is fast and serves direct reads clearly faster
    // (platform::file_tier_prefers_direct).  STRATA_FILE_TIER_PROBE=0 keeps the room rule alone.
    if (keepable && !old_rule && read_bytes > 0) {
        const char* pe = std::getenv("STRATA_FILE_TIER_PROBE");
        const uint64_t margin = 1ull << 30;
        const double room = avail > arena_bytes + margin ? (double) (avail - arena_bytes - margin) : 0.0;
        const double held = room / (double) read_bytes;
        if (!(pe != nullptr && pe[0] == '0') && held < 0.5) {
            double d = 0, c = 0;
            char pm[256];
            if (file_tier_probe(paths, d, c)) {
                const bool direct = strata::platform::file_tier_prefers_direct(held, d, c);
                std::snprintf(pm, sizeof pm, "; the cache holds %.0f%% of the read set, direct reads %.2f GB/s, cached-path reads "
                              "%.2f GB/s (%.2fx): %s", held * 100.0, d / 1e9, c / 1e9, d / c,
                              direct ? "the file tier reads unbuffered" : "the file cache stays");
                why += pm;
                return direct;
            }
            why += "; the drive probe could not run, the file cache stays";
        }
    }
    return !keepable;
#else
    (void) paths; (void) arena_bytes; (void) read_bytes;
    why = "through the file cache (no unbuffered reads on this platform)";
    return false;
#endif
}

}  // namespace

// ================================ THE FILE-BACKED SOURCE ================================

FileExpertSource::~FileExpertSource() { close(); }

#if defined(__linux__)
namespace {
/// Bytes of [p, p + n) whose pages are in the page cache (mincore, page-granular); 0 when the call fails.
uint64_t resident_in(const uint8_t* p, uint64_t n) {
    if (n == 0) return 0;
    static thread_local std::vector<unsigned char> vec;
    const uintptr_t a = (uintptr_t) p & ~(uintptr_t) 4095, e = ((uintptr_t) p + (uintptr_t) n + 4095) & ~(uintptr_t) 4095;
    vec.resize((size_t) ((e - a) / 4096));
    if (mincore((void*) a, (size_t) (e - a), vec.data()) != 0) return 0;
    uint64_t pages = 0;
    for (unsigned char c : vec) pages += c & 1;
    return std::min<uint64_t>(n, pages * 4096);
}
/// A cheap residency probe: the first, last and two inner pages; the bytes of the range when all four are cached, else 0.
uint64_t resident_sampled(const uint8_t* p, uint64_t n) {
    if (n == 0) return 0;
    const uintptr_t a = (uintptr_t) p & ~(uintptr_t) 4095, e = ((uintptr_t) p + (uintptr_t) n - 1) & ~(uintptr_t) 4095;
    const uintptr_t pages = (e - a) / 4096;
    const uintptr_t at[4] = {a, a + (pages / 3) * 4096, a + (pages * 2 / 3) * 4096, e};
    for (uintptr_t x : at) {
        unsigned char v = 0;
        if (mincore((void*) x, 4096, &v) != 0 || (v & 1) == 0) return 0;
    }
    return n;
}
bool resident_sampled_n(const uint8_t* p, uint64_t n) { return resident_sampled(p, n) >= n; }
/// The whole range with pread (a short read continues); false on an error or an end of file.
bool pread_full(int fd, uint8_t* dst, uint64_t n, uint64_t off) {
    uint64_t done = 0;
    while (done < n) {
        const ssize_t r = ::pread(fd, dst + done, (size_t) (n - done), (off_t) (off + done));
        if (r < 0) { if (errno == EINTR) continue; return false; }
        if (r == 0) return false;
        done += (uint64_t) r;
    }
    return true;
}
}  // namespace
#endif

int FileExpertSource::spans(int64_t layer, int64_t expert, Span* out) const {
    if (layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return 0;
    if (!role_ptr_.empty()) {
        if (unmapped_) return 0;   // the GGUF views are closed (drop_mapping)
        uint64_t at = 0;
        for (int r = 0; r < 3; ++r) {
            const size_t i = (size_t) (3 * layer + r);
            const uint64_t per = role_bytes_[i];
            const uint8_t* p = role_ptr_[i] + (size_t) ((uint64_t) expert * per);
            const size_t f = (size_t) role_file_[i];
            out[r] = {p, per, f < io_fds_.size() ? io_fds_[f] : -1, (uint64_t) (p - maps_[f].base), at};
            at += per;
        }
        return 3;
    }
    const uint8_t* b = mapped_blob(layer, expert);
    if (b == nullptr) return 0;
    out[0] = {b, layer_blob_bytes_[(size_t) layer], io_fds_.empty() ? -1 : io_fds_[0], (uint64_t) (b - base_), 0};
    return 1;
}

uint64_t FileExpertSource::cached_bytes(int64_t layer, int64_t expert, uint64_t& total, bool exact) const {
    total = 0;
#if defined(__linux__)
    Span sp[3];
    const int ns = spans(layer, expert, sp);
    uint64_t res = 0;
    for (int i = 0; i < ns; ++i) {
        total += sp[i].n;
        res += exact ? resident_in(sp[i].p, sp[i].n) : resident_sampled(sp[i].p, sp[i].n);
    }
    return res;
#else
    (void) layer; (void) expert;
    return 0;
#endif
}

FileExpertSource::IoCounters FileExpertSource::io_counters() const {
    IoCounters c;
    c.cached_bytes = io_cached_.load(std::memory_order_relaxed);
    c.uncached_bytes = io_uncached_.load(std::memory_order_relaxed);
    c.pread_bytes = io_pread_bytes_.load(std::memory_order_relaxed);
    c.pread_us = io_pread_us_.load(std::memory_order_relaxed);
    c.pread_n = io_pread_n_.load(std::memory_order_relaxed);
    c.pf_jobs = io_pf_jobs_.load(std::memory_order_relaxed);
    c.pf_demand = io_pf_demand_.load(std::memory_order_relaxed);
    c.pf_read_blobs = io_pf_blobs_.load(std::memory_order_relaxed);
    c.pf_resident_skips = io_pf_skips_.load(std::memory_order_relaxed);
    c.pf_dropped = io_pf_dropped_.load(std::memory_order_relaxed);
    c.pf_used = io_pf_used_.load(std::memory_order_relaxed);
    c.pf_unused = io_pf_unused_.load(std::memory_order_relaxed);
    c.crit_us = io_crit_us_.load(std::memory_order_relaxed);
    c.crit_n = io_crit_n_.load(std::memory_order_relaxed);
    c.pf_bytes = io_pf_bytes_.load(std::memory_order_relaxed);
    c.pf_used_bytes = io_pf_used_bytes_.load(std::memory_order_relaxed);
    c.pf_late = io_pf_late_.load(std::memory_order_relaxed);
    c.pf_rej_conf = io_pf_rej_conf_.load(std::memory_order_relaxed);
    c.pf_rej_budget = io_pf_rej_budget_.load(std::memory_order_relaxed);
    if (io_sched_) {
        std::lock_guard<std::mutex> lk(const_cast<std::mutex&>(io_mu_));
        c.pf_budget_bytes = io_budget_now();
    }
    return c;
}

// 149-prefetch: the speculative bytes a decode window may read ahead.  Fixed (STRATA_IO_PF_BUDGET_MB), or what the workers
// measured they can read in one window (one worker's rate x the workers x the window's time) times STRATA_IO_PF_BUDGET_FRAC;
// 256 MB until both rates are known.
uint64_t FileExpertSource::io_budget_now() const {
    if (io_budget_fixed_ > 0) return io_budget_fixed_;
    if (io_bw_ <= 0.0 || io_win_us_ <= 0.0) return 256ull << 20;
    const double b = io_bw_ * (double) std::max(1, io_threads_) * io_win_us_ * io_budget_frac_ * io_scale_;
    return (uint64_t) std::max(16.0 * 1048576.0, b);
}

void FileExpertSource::io_window_tick() {
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(io_mu_);
    if (io_win_t0_.time_since_epoch().count() != 0) {
        const double us = std::chrono::duration<double, std::micro>(now - io_win_t0_).count();
        io_win_us_ = io_win_us_ <= 0.0 ? us : 0.8 * io_win_us_ + 0.2 * us;
    }
    io_win_t0_ = now;
    io_win_bytes_ = 0;
    // feedback: the share of the read-ahead bytes that a layer then used (the used count lags the reads by a window or
    // so; an EWMA over windows evens that out).  Below STRATA_IO_PF_TARGET the budget shrinks (x0.7 a window, to 1/20),
    // above it it grows back (x1.1, to 1): a box whose page cache cannot hold what is read ahead stops paying for it.
    const uint64_t pf = io_pf_bytes_.load(std::memory_order_relaxed), used = io_pf_used_bytes_.load(std::memory_order_relaxed);
    const uint64_t dpf = pf - io_prev_pf_, du = used - io_prev_used_;
    io_prev_pf_ = pf;
    io_prev_used_ = used;
    if (io_target_ > 0.0 && dpf > 0) {
        const double u = std::min(1.0, (double) du / (double) dpf);
        io_useful_ = io_useful_ < 0.0 ? u : 0.8 * io_useful_ + 0.2 * u;
        io_scale_ = io_useful_ < io_target_ ? std::max(0.05, io_scale_ * 0.7) : std::min(1.0, io_scale_ * 1.1);
    }
}

// 149-prefetch: admit the predicted experts that are not resident.  An expert's bar is the base confidence, raised with
// the queue's fill against the window budget (a busy drive takes only the surer guesses); the window's speculative bytes
// stay within the budget; each job carries the layer clock by which it must have started (else the worker drops it).
void FileExpertSource::warm_scored(int64_t layer, const int64_t* experts, const float* conf, int64_t n, int ahead) {
    if (!io_sched_ || !io_pf_ || !direct_.empty() || !io_ahead_ || !gate_open()) { warm(layer, experts, n); return; }
    if (n <= 0 || layer < 0 || layer >= n_layers_) return;
    const uint64_t bytes = layer_blob_bytes_[(size_t) layer];
    const uint64_t deadline = io_epoch_.load(std::memory_order_relaxed) + (uint64_t) std::max(1, ahead);
    bool any = false;
    {
        std::lock_guard<std::mutex> lk(io_mu_);
        const uint64_t budget = io_budget_now();
        for (int64_t j = 0; j < n; ++j) {
            const int64_t e = experts[j];
            if (e < 0 || e >= n_expert_) continue;
            const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) e;
            if (complement_ready_ && resident_blob(index) != nullptr) continue;   // in the RAM copy
            warm_count_.fetch_add(1, std::memory_order_relaxed);
            const double pressure = std::min(1.0, (double) io_q_bytes_ / (double) std::max<uint64_t>(1, budget));
            const double bar = (double) io_minconf_ + (1.0 - (double) io_minconf_) * pressure;
            const float c = conf != nullptr ? conf[j] : 1.0f;
            if ((double) c < bar) { io_pf_rej_conf_.fetch_add(1, std::memory_order_relaxed); continue; }
            if (io_win_bytes_ + bytes > budget) { io_pf_rej_budget_.fetch_add(1, std::memory_order_relaxed); continue; }
            bool dup = false;
            for (const IoJob& q : io_jobs_)
                if (q.layer == (int32_t) layer && q.expert == (int32_t) e) { dup = true; break; }
            if (dup) continue;
            io_jobs_.push_back({(int32_t) layer, (int32_t) e, c, deadline, bytes});
            io_q_bytes_ += bytes;
            io_win_bytes_ += bytes;
            any = true;
        }
    }
    if (any) io_cv_.notify_all();
}

void FileExpertSource::set_io_prefetch(bool on, int threads) {
    io_stop();
#if defined(__linux__)
    if (!on) return;
    if (base_ == nullptr || paths_.empty() || !direct_.empty()) return;   // unbuffered readers batch their own reads
    std::vector<int> fds;
    for (const std::string& path : paths_) {
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            for (int f : fds) ::close(f);
            return;
        }
        (void) posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);   // read exactly the range asked for, no read-ahead beyond it
        fds.push_back(fd);
    }
    io_fds_ = std::move(fds);
    {
        std::lock_guard<std::mutex> lk(stage_mu_);
        for (uint64_t b : layer_blob_bytes_) stage_blob_ = std::max(stage_blob_, b);
    }
    if (!warm_stamp_) warm_stamp_.reset(new std::atomic<uint32_t>[(size_t) (n_layers_ * n_expert_)]());
    io_threads_ = threads > 0 ? threads : 8;
    if (const char* v = std::getenv("STRATA_IO_PF_AHEAD")) io_ahead_ = std::atoi(v) != 0;
    {   // experts.bin: fill the page cache from the workers and let the kernels read the mapping (STRATA_IO_PF_STAGE=1: stage)
        const char* v = std::getenv("STRATA_IO_PF_STAGE");
        io_fill_ = role_ptr_.empty() && !(v != nullptr && std::atoi(v) != 0);
    }
    io_pf_ = true;
    io_stats_ = true;
    io_quit_ = false;
    {   // 149-prefetch: STRATA_IO_PF_SCHED=1 - deadlines, confidence admission and a per-window byte budget (off: as before)
        const char* v = std::getenv("STRATA_IO_PF_SCHED");
        io_sched_ = v != nullptr && std::atoi(v) != 0;
        if (const char* m = std::getenv("STRATA_IO_PF_MINCONF")) io_minconf_ = (float) std::clamp(std::atof(m), 0.0, 1.0);
        if (const char* m = std::getenv("STRATA_IO_PF_BUDGET_MB")) io_budget_fixed_ = (uint64_t) std::max(0.0, std::atof(m) * 1048576.0);
        if (const char* m = std::getenv("STRATA_IO_PF_BUDGET_FRAC")) io_budget_frac_ = std::clamp(std::atof(m), 0.01, 4.0);
        if (const char* m = std::getenv("STRATA_IO_PF_TARGET")) io_target_ = std::clamp(std::atof(m), 0.0, 1.0);
        gate_on_ = true;   // 0.2.0: the decode gate comes with io prefetch; STRATA_IO_PF_GATE=0 turns it off
        if (const char* m = std::getenv("STRATA_IO_PF_GATE"); m != nullptr && m[0] != '\0') gate_on_ = std::atoi(m) != 0;
        if (const char* m = std::getenv("STRATA_IO_PF_GATE_LO")) gate_lo_ = std::clamp(std::atof(m), 0.0, 1.0);
        if (const char* m = std::getenv("STRATA_IO_PF_GATE_HI")) gate_hi_ = std::clamp(std::atof(m), gate_lo_, 1.0);
        if (const char* m = std::getenv("STRATA_IO_PF_GATE_USEFUL")) gate_min_useful_ = std::clamp(std::atof(m), 0.0, 1.0);
    }
    if (!pf_mark_) {
        pf_mark_.reset(new std::atomic<uint8_t>[(size_t) (n_layers_ * n_expert_)]);
        for (int64_t i = 0; i < n_layers_ * n_expert_; ++i) pf_mark_[(size_t) i].store(0, std::memory_order_relaxed);
    }
    for (int i = 0; i < io_threads_; ++i) io_workers_.emplace_back([this] { io_worker(); });
#else
    (void) on; (void) threads;
#endif
}

void FileExpertSource::io_stop() {
    {
        std::lock_guard<std::mutex> lk(io_mu_);
        io_quit_ = true;
        io_q_.clear();
        io_jobs_.clear();
        io_q_bytes_ = 0;
    }
    io_cv_.notify_all();
    for (std::thread& t : io_workers_)
        if (t.joinable()) t.join();
    io_workers_.clear();
#if defined(__linux__)
    for (int f : io_fds_) ::close(f);
#endif
    io_fds_.clear();
    io_pf_ = false;
    io_quit_ = false;
}

void FileExpertSource::io_enqueue(int64_t layer, int64_t expert, bool ahead) {
    {
        std::lock_guard<std::mutex> lk(io_mu_);
        if (io_q_.size() >= 512) {   // the predictions run ahead of the reads: the oldest are the stalest
            io_q_.pop_front();
            io_pf_dropped_.fetch_add(1, std::memory_order_relaxed);
        }
        io_q_.emplace_back((int32_t) layer, (int32_t) expert | (ahead ? kIoAhead : 0));
    }
    io_cv_.notify_one();
}

// A read-ahead worker: takes a predicted (layer, expert), skips it when its pages are cached or it is already
// staged, and otherwise reads the whole blob into a staging buffer (copy_from_files preads what is not cached).
void FileExpertSource::io_worker() {
    strata::platform::name_this_thread("strata-io");
    for (;;) {
        int32_t layer, expert;
        bool ahead = true;   // a prediction (marked for the used / unused count), not a layer's own demand read
        {
            std::unique_lock<std::mutex> lk(io_mu_);
            io_cv_.wait(lk, [&] { return io_quit_ || !io_q_.empty() || !io_jobs_.empty(); });
            if (io_quit_) return;
            if (io_sched_ && io_q_.empty() && !io_jobs_.empty()) {   // (the layers' own reads in io_q_ go first)
                // 149-prefetch: drop the jobs whose layer the decode has reached (its demand read covers them); then the
                // earliest deadline, the surest guess first among equals
                const uint64_t now = io_epoch_.load(std::memory_order_relaxed);
                size_t best = io_jobs_.size();
                for (size_t i = 0; i < io_jobs_.size();) {
                    if (io_jobs_[i].deadline <= now) {
                        io_q_bytes_ -= std::min(io_q_bytes_, io_jobs_[i].bytes);
                        io_jobs_[i] = io_jobs_.back();
                        io_jobs_.pop_back();
                        io_pf_late_.fetch_add(1, std::memory_order_relaxed);
                        continue;
                    }
                    ++i;
                }
                best = io_jobs_.size();
                for (size_t i = 0; i < io_jobs_.size(); ++i)
                    if (best == io_jobs_.size() || io_jobs_[i].deadline < io_jobs_[best].deadline ||
                        (io_jobs_[i].deadline == io_jobs_[best].deadline && io_jobs_[i].conf > io_jobs_[best].conf))
                        best = i;
                if (best == io_jobs_.size()) continue;   // all were late
                layer = io_jobs_[best].layer;
                expert = io_jobs_[best].expert;
                io_q_bytes_ -= std::min(io_q_bytes_, io_jobs_[best].bytes);
                io_jobs_[best] = io_jobs_.back();
                io_jobs_.pop_back();
            } else {
                if (io_q_.empty()) continue;
                layer = io_q_.back().first;   // the newest prediction first: it is the layer that comes next
                expert = io_q_.back().second & ~kIoAhead;
                ahead = (io_q_.back().second & kIoAhead) != 0;
                io_q_.pop_back();
            }
        }
        io_pf_jobs_.fetch_add(1, std::memory_order_relaxed);
        const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
        if (complement_ready_ && resident_blob(index) != nullptr) continue;
        if (!override_.empty() && override_[index] != nullptr) continue;
        const auto tjob = std::chrono::steady_clock::now();
        // 149-prefetch: one worker's read rate (bytes per microsecond), for the scheduler's window budget
        auto note_rate = [&](uint64_t b) {
            if (b == 0) return;
            const double us = std::max(1.0, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - tjob).count());
            io_pf_bytes_.fetch_add(b, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lk(io_mu_);
            const double r = (double) b / us;
            io_bw_ = io_bw_ <= 0.0 ? r : 0.9 * io_bw_ + 0.1 * r;
        };
        if (io_fill_) {   // read what is not cached into a scratch buffer: the pages stay in the page cache for the kernels
#if defined(__linux__)
            static thread_local std::vector<uint8_t> scratch;
            Span sp[3];
            const int ns = spans(layer, expert, sp);
            bool any = false;
            uint64_t got = 0;
            for (int i = 0; i < ns; ++i) {
                if (sp[i].fd < 0 || resident_sampled_n(sp[i].p, sp[i].n)) continue;
                if (scratch.size() < sp[i].n) scratch.resize((size_t) sp[i].n);
                const auto t0 = std::chrono::steady_clock::now();
                if (pread_full(sp[i].fd, scratch.data(), sp[i].n, sp[i].off)) {
                    io_pread_bytes_.fetch_add(sp[i].n, std::memory_order_relaxed);
                    io_pread_us_.fetch_add((uint64_t) std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count(),
                                           std::memory_order_relaxed);
                    io_pread_n_.fetch_add(1, std::memory_order_relaxed);
                    any = true;
                    got += sp[i].n;
                }
            }
            if (any) {
                io_pf_blobs_.fetch_add(1, std::memory_order_relaxed);
                if (pf_mark_ && ahead) pf_mark_[index].store(1, std::memory_order_relaxed);   // read ahead: scored at the demand
                note_rate(got);
            }
            else io_pf_skips_.fetch_add(1, std::memory_order_relaxed);
#endif
            continue;
        }
        uint64_t total = 0;
        if (cached_bytes(layer, expert, total, false) >= total) {
            io_pf_skips_.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        size_t v = 0;
        bool fill = false;
        uint8_t* dst = nullptr;
        {
            std::lock_guard<std::mutex> lk(stage_mu_);
            if (stage_blob_ == 0) continue;
            const bool have = claim_stage((int64_t) layer * n_expert_ + expert, v, fill, /*ahead=*/true);
            if (have || !fill) continue;   // already staged or being read, or no buffer
            stage_pf_[v] = 1;
            dst = stage_buf_[v].get();
        }
        const bool ok = copy_from_files(layer, expert, dst);
        io_pf_blobs_.fetch_add(1, std::memory_order_relaxed);
        if (ok) file_read_bytes_.fetch_add(layer_blob_bytes_[(size_t) layer], std::memory_order_relaxed);
        if (ok) note_rate(layer_blob_bytes_[(size_t) layer]);
        {
            std::lock_guard<std::mutex> lk(stage_mu_);
            stage_busy_[v] = 0;
            if (!ok) {
                stage_of_.erase(stage_key_[v]);
                stage_key_[v] = -1;
                stage_pf_[v] = 0;
            }
        }
        stage_cv_.notify_all();
    }
}

// The decode's hand-out of a blob in the experts.bin file tier with io prefetch on: what a layer's prediction read
// ahead is staged; a blob whose pages are all cached is the mapping itself (no copy); the rest is read with one
// pread into a staging buffer, not faulted in by the kernels one small request at a time.
const uint8_t* FileExpertSource::io_blob(int64_t layer, int64_t expert) {
    const int64_t key = layer * n_expert_ + expert;
    if (io_fill_) {
        const uint8_t* b = mapped_blob(layer, expert);
        if (b != nullptr) {
            const uint64_t total = layer_blob_bytes_[(size_t) layer];
            file_read_bytes_.fetch_add(total, std::memory_order_relaxed);
        }
        return b;
    }
    bool have;
    {
        std::lock_guard<std::mutex> lk(stage_mu_);
        have = stage_of_.find(key) != stage_of_.end();
    }
    if (!have) {
        // prefetch() (every decode layer calls it first) staged what was not cached: what is not staged is cached
        const uint8_t* b = mapped_blob(layer, expert);
        if (b != nullptr) {
            const uint64_t total = layer_blob_bytes_[(size_t) layer];
            io_cached_.fetch_add(total, std::memory_order_relaxed);
            file_read_bytes_.fetch_add(total, std::memory_order_relaxed);
            return b;
        }
    }
    return staged_blob(layer, expert);
}

bool FileExpertSource::open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err) {
    close();
    if (n_layers <= 0 || n_expert <= 0) { err = "FileExpertSource: the geometry is empty"; return false; }
    const auto& layout = strata::kernels::cpu::expert_layout();
    if (layout.n_layers != n_layers || layout.n_expert != n_expert) {
        err = "FileExpertSource: the requested geometry does not match the loaded expert layout";
        return false;
    }
    if ((uint64_t) n_layers > (uint64_t) std::numeric_limits<int64_t>::max() / (uint64_t) n_expert) {
        err = "FileExpertSource: the expert count overflows";
        return false;
    }
    const uint64_t blob_count = (uint64_t) n_layers * (uint64_t) n_expert;
    if (blob_count > (uint64_t) std::numeric_limits<int64_t>::max() ||
        (uint64_t) n_layers > (uint64_t) std::numeric_limits<size_t>::max()) {
        err = "FileExpertSource: the expert count overflows";
        return false;
    }

    std::vector<uint64_t> layer_offsets((size_t) n_layers), layer_blob_bytes((size_t) n_layers);
    const uint64_t want = layout.total;
    if (want == 0 || want > (uint64_t) std::numeric_limits<size_t>::max()) {
        err = "FileExpertSource: the loaded expert layout has an invalid size";
        return false;
    }
    if (!layout.native) {
        if (blob_count > std::numeric_limits<uint64_t>::max() / (uint64_t) strata::kernels::cpu::BLOB) {
            err = "FileExpertSource: the canonical expert size overflows";
            return false;
        }
        const uint64_t canonical_size = blob_count * (uint64_t) strata::kernels::cpu::BLOB;
        if (want != canonical_size) {
            err = "FileExpertSource: the canonical expert layout has an inconsistent size";
            return false;
        }
        const uint64_t bytes = (uint64_t) strata::kernels::cpu::BLOB;
        const uint64_t layer_bytes = (uint64_t) n_expert * bytes;
        for (int64_t layer = 0; layer < n_layers; ++layer) {
            layer_offsets[(size_t) layer] = (uint64_t) layer * layer_bytes;
            layer_blob_bytes[(size_t) layer] = bytes;
        }
    } else {
        if (layout.offset.size() != (size_t) n_layers || layout.bytes.size() != (size_t) n_layers ||
            layout.fmt.size() != (size_t) n_layers) {
            err = "FileExpertSource: the native expert layout is incomplete";
            return false;
        }
        uint64_t at = 0;
        for (int64_t layer = 0; layer < n_layers; ++layer) {
            const size_t i = (size_t) layer;
            const uint64_t bytes = (uint64_t) layout.fmt[i].bytes;
            if (layout.offset[i] != at || bytes == 0 || layout.bytes[i] != bytes ||
                bytes > std::numeric_limits<uint64_t>::max() / (uint64_t) n_expert) {
                err = "FileExpertSource: the native expert layout is invalid at layer " + std::to_string(layer);
                return false;
            }
            const uint64_t layer_bytes = bytes * (uint64_t) n_expert;
            if (at > want || layer_bytes > want - at) {
                err = "FileExpertSource: the native expert layout exceeds its declared size at layer " +
                      std::to_string(layer);
                return false;
            }
            layer_offsets[i] = layout.offset[i];
            layer_blob_bytes[i] = bytes;
            at += layer_bytes;
        }
        if (at != want) {
            err = "FileExpertSource: the native expert layout has an inconsistent size";
            return false;
        }
    }
    const std::string path = pack_dir + "/experts.bin";
    if (layout.native && !gguf_.empty() && !std::filesystem::exists(path)) {
        // CS-T: no experts.bin - the model's GGUF shards, read in place
        blobs_ = (int64_t) blob_count;
        n_layers_ = n_layers;
        n_expert_ = n_expert;
        layer_offsets_ = std::move(layer_offsets);
        layer_blob_bytes_ = std::move(layer_blob_bytes);
        if (!open_gguf(err)) { close(); return false; }
        record_inputs(path, gguf_, layout, true);
        return true;
    }

#if defined(_WIN32)
    // UTF-8 -> UTF-16: the pack may live under a path with non-ASCII characters, and `CreateFileA` would
    // silently mangle it into a file-not-found.
    const int wide = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::vector<wchar_t> wpath((size_t) (wide > 0 ? wide : 1));
    if (wide > 0) MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wide);
    // **`FILE_FLAG_RANDOM_ACCESS` WAS HERE AND IT COST 14x.**
    //
    // The design depends on the OS page cache holding the whole 34 GB expert set, because this machine has
    // 64 GB of DDR5 and `L9` measured the CPU path at 44.14 GB/s from DRAM.  `FILE_FLAG_RANDOM_ACCESS` tells
    // the cache manager the opposite: it disables read-ahead AND it lets the manager drop the pages again
    // quickly, on the assumption that a large randomly-accessed file will not be re-read.  Measured, on
    // `strata generate --max-new 24`: **1.93 GB/s** - disk speed, 344 ms/token, and it never warmed up over 25
    // tokens, because the pages were being evicted as fast as they were faulted in.
    //
    // The correct flag is NO flag.  The access pattern IS random (10 of 512 experts per layer, a different 10
    // each layer), but every byte read is read again on the next token, so retention is the whole game.
    HANDLE f = CreateFileW(wpath.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) {
        err = "FileExpertSource: cannot open " + path;
        return false;
    }
    LARGE_INTEGER sz{};
    if (!GetFileSizeEx(f, &sz)) {
        CloseHandle(f);
        err = "FileExpertSource: cannot size " + path;
        return false;
    }
    if ((uint64_t) sz.QuadPart != want) {
        char buf[400];
        std::snprintf(buf, sizeof buf,
                      "FileExpertSource: %s is %llu B but the loaded expert layout requires %llu B - this is not "
                      "the pack this geometry came from",
                      path.c_str(), (unsigned long long) sz.QuadPart, (unsigned long long) want);
        CloseHandle(f);
        err = buf;
        return false;
    }
    HANDLE m = CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (m == nullptr) {
        CloseHandle(f);
        err = "FileExpertSource: CreateFileMapping failed on " + path;
        return false;
    }
    void* view = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    if (view == nullptr) {
        CloseHandle(m);
        CloseHandle(f);
        err = "FileExpertSource: MapViewOfFile failed on " + path;
        return false;
    }
    file_ = f;
    mapping_ = m;
    base_ = (const uint8_t*) view;
    paths_.assign(1, path);
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { err = "FileExpertSource: cannot open " + path; return false; }
    struct stat st{};
    if (fstat(fd, &st) != 0) { ::close(fd); err = "FileExpertSource: cannot stat " + path; return false; }
    if (st.st_size < 0 || (uint64_t) st.st_size != want) {
        char buf[400];
        std::snprintf(buf, sizeof buf,
                      "FileExpertSource: %s is %llu B but the loaded expert layout requires %llu B - this is not "
                      "the pack this geometry came from",
                      path.c_str(), (unsigned long long) (st.st_size < 0 ? 0 : st.st_size),
                      (unsigned long long) want);
        ::close(fd);
        err = buf;
        return false;
    }
    void* view = mmap(nullptr, (size_t) want, PROT_READ, MAP_SHARED, fd, 0);
    if (view == MAP_FAILED) { ::close(fd); err = "FileExpertSource: mmap failed on " + path; return false; }
    fd_ = fd;
    base_ = (const uint8_t*) view;
    paths_.assign(1, path);   // #286: open_direct opens it again with O_DIRECT
#endif
    blobs_ = (int64_t) blob_count;
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    mapped_bytes_ = want;
    layer_offsets_ = std::move(layer_offsets);
    layer_blob_bytes_ = std::move(layer_blob_bytes);
    record_inputs(path, gguf_, layout, false);
    return true;
}

void FileExpertSource::close() {
    prefetch_end();
    ram_stop();
    io_stop();
    inputs_.clear();
    if (complement_arena_ != nullptr) {
        if (complement_pinned_ && !complement_registered_) (void) cudaFreeHost(complement_arena_);
        else {
            if (complement_registered_) (void) cudaHostUnregister(complement_arena_);
            if (complement_locked_ > 0)
                strata::platform::unlock_resident((uint8_t*) complement_arena_ + complement_lock_off_, complement_locked_);
            std::free(complement_arena_);
        }
    }
    if (xstage_ != nullptr) {
        if (xstage_pinned_) (void) cudaFreeHost(xstage_);
        else std::free(xstage_);
    }
    xstage_ = nullptr;
    xstage_pinned_ = false;
    xstage_cap_ = 0;
    xstage_blob_ = 0;
    override_.clear();
    staged_.clear();
    exchange_storage_.clear();
    exchanges_ = 0;
    file_reads_.store(0);
    complement_arena_ = nullptr;
    complement_host_ = nullptr;
    complement_device_ = nullptr;
    complement_bytes_ = 0;
    complement_offsets_.clear();
    complement_pinned_ = false;
    complement_partial_ = false;
    complement_registered_ = false;
    complement_pin_limit_ = 0;
    complement_lock_off_ = 0;
    complement_ready_ = false;
    complement_locked_ = 0;
    complement_lent_slots_ = 0;
    if (!maps_.empty()) {
        for (Map& m : maps_) {
#if defined(_WIN32)
            if (m.base != nullptr && !m.closed) UnmapViewOfFile((LPCVOID) m.base);
            if (m.mapping != nullptr) CloseHandle((HANDLE) m.mapping);
            if (m.file != nullptr) CloseHandle((HANDLE) m.file);
#else
            if (m.base != nullptr && !m.closed) munmap((void*) m.base, (size_t) m.bytes);
            if (m.fd >= 0) ::close(m.fd);
#endif
        }
        maps_.clear();
        base_ = nullptr;   // one of the views above
        unmapped_ = false;
    }
    role_ptr_.clear();
    role_bytes_.clear();
    role_file_.clear();
    paths_.clear();
    for (void* h : direct_) close_direct(h);
    direct_.clear();
    for (void* h : pdirect_) close_direct(h);
    pdirect_.clear();
    {
        std::lock_guard<std::mutex> lk(stage_mu_);
        // the buffers' last copies end before they are freed (#1237)
        if (!stage_ev_.empty()) (void) cudaDeviceSynchronize();
        stage_free_events();
        stage_buf_.clear();
        stage_key_.clear();
        stage_epoch_.clear();
        stage_used_.clear();
        stage_busy_.clear();
        stage_pf_.clear();
        stage_of_.clear();
        stage_blob_ = 0;
        stage_seq_ = 0;
        epoch_ = 0;
        last_layer_ = -1;
        stage_grew_ = false;
    }
    ram_reads_.store(0);
    warm_stamp_.reset();
    warm_hits_.store(0);
    warm_count_.store(0);
    file_read_bytes_.store(0);
    file_blob_bytes_.store(0);
    file_us_.store(0);
    io_cached_.store(0); io_uncached_.store(0); io_pread_bytes_.store(0); io_pread_us_.store(0); io_pread_n_.store(0);
    io_pf_jobs_.store(0); io_pf_blobs_.store(0); io_pf_skips_.store(0); io_pf_dropped_.store(0);
    io_pf_used_.store(0); io_pf_unused_.store(0); io_crit_us_.store(0); io_crit_n_.store(0);
#if defined(_WIN32)
    if (base_ != nullptr && !unmapped_) UnmapViewOfFile((LPCVOID) base_);
    unmapped_ = false;
    if (mapping_ != nullptr) CloseHandle((HANDLE) mapping_);
    if (file_ != nullptr) CloseHandle((HANDLE) file_);
    mapping_ = nullptr;
    file_ = nullptr;
#else
    if (base_ != nullptr) munmap((void*) base_, (size_t) mapped_bytes_);
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
#endif
    base_ = nullptr;
    blobs_ = 0;
    n_layers_ = 0;
    n_expert_ = 0;
    mapped_bytes_ = 0;
    layer_offsets_.clear();
    layer_blob_bytes_.clear();
    reads_ = 0;
}


bool ExpertSource::copy_blob(int64_t layer, int64_t expert, uint8_t* dst) {
    const uint8_t* b = blob(layer, expert);
    if (b == nullptr || dst == nullptr) return false;
    std::memcpy(dst, b, (size_t) strata::kernels::cpu::expert_layout().blob_bytes(layer));
    return true;
}

bool ExpertSource::copy_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (!copy_blob(layers[i], experts[i], dst[i])) return false;
    return true;
}

// ================================ CS-T: THE GGUF SHARDS IN PLACE ================================
//
// A native pack without experts.bin: every file native_experts.txt names is mapped (MapViewOfFile / mmap, no
// flag - the same retention argument as experts.bin above), and an expert's blob [gate rows | up rows | down rows]
// is three slices of three tensors, possibly in two shards (UD-Q4_K_XL's layer 11).  Nothing is read at open; a
// blob is assembled when it is asked for (`blob`, into a small pool of buffers) or copied where it is needed
// (`copy_blob`: the RAM copy, the prompt path's pinned stager buffers).
bool FileExpertSource::open_gguf(std::string& err) {
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (!check_experts_gguf(gguf_, lay, err)) { err = "FileExpertSource: " + err; return false; }
    const size_t cut = gguf_.find_last_of("/\\");
    const std::string dir = cut == std::string::npos ? std::string() : gguf_.substr(0, cut + 1);
    std::map<std::string, size_t> index;
    role_ptr_.assign((size_t) (3 * n_layers_), nullptr);
    role_bytes_.assign((size_t) (3 * n_layers_), 0);
    role_file_.assign((size_t) (3 * n_layers_), 0);
    for (int64_t l = 0; l < n_layers_; ++l) {
        const auto& fm = lay.fmt[(size_t) l];
        const uint64_t per[3] = {fm.up_off, fm.up_off, lay.bytes[(size_t) l] - fm.down_off};
        for (int r = 0; r < 3; ++r) {
            const size_t i = (size_t) (3 * l + r);
            const std::string path = lay.gguf_file.size() > i && !lay.gguf_file[i].empty() ? dir + lay.gguf_file[i]
                                                                                            : gguf_;
            auto it = index.find(path);
            if (it == index.end()) {
                Map m;
#if defined(_WIN32)
                const int wide = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
                std::vector<wchar_t> wpath((size_t) (wide > 0 ? wide : 1));
                if (wide > 0) MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wide);
                HANDLE f = CreateFileW(wpath.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                       FILE_ATTRIBUTE_NORMAL, nullptr);
                LARGE_INTEGER sz{};
                if (f == INVALID_HANDLE_VALUE || !GetFileSizeEx(f, &sz)) {
                    if (f != INVALID_HANDLE_VALUE) CloseHandle(f);
                    err = "FileExpertSource: cannot open " + path;
                    return false;
                }
                HANDLE mh = CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
                void* view = mh != nullptr ? MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0) : nullptr;
                if (view == nullptr) {
                    if (mh != nullptr) CloseHandle(mh);
                    CloseHandle(f);
                    err = "FileExpertSource: cannot map " + path;
                    return false;
                }
                m.file = f;
                m.mapping = mh;
                m.bytes = (uint64_t) sz.QuadPart;
                m.base = (const uint8_t*) view;
#else
                const int fd = ::open(path.c_str(), O_RDONLY);
                struct stat st{};
                if (fd < 0 || fstat(fd, &st) != 0 || st.st_size <= 0) {
                    if (fd >= 0) ::close(fd);
                    err = "FileExpertSource: cannot open " + path;
                    return false;
                }
                void* view = mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
                if (view == MAP_FAILED) { ::close(fd); err = "FileExpertSource: cannot map " + path; return false; }
                m.fd = fd;
                m.bytes = (uint64_t) st.st_size;
                m.base = (const uint8_t*) view;
#endif
                maps_.push_back(m);
                paths_.push_back(path);
                it = index.emplace(path, maps_.size() - 1).first;
            }
            const Map& m = maps_[it->second];
            role_file_[i] = (int) it->second;
            const uint64_t at = lay.gguf_off[i], bytes = per[r] * (uint64_t) n_expert_;
            if (at > m.bytes || bytes > m.bytes - at) {   // check_experts_gguf proved it; the mapping must agree
                err = "FileExpertSource: an expert span runs past the end of " + path;
                return false;
            }
            role_ptr_[i] = m.base + (size_t) at;
            role_bytes_[i] = per[r];
        }
    }
    base_ = maps_.front().base;       // "opened"; mapped_blob answers nullptr in this mode
    warm_stamp_.reset(new std::atomic<uint32_t>[(size_t) (n_layers_ * n_expert_)]());
    for (uint64_t b : layer_blob_bytes_) stage_blob_ = std::max(stage_blob_, b);
    return true;
}

bool FileExpertSource::copy_from_files(int64_t layer, int64_t expert, uint8_t* dst) const {
    if (dst == nullptr || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return false;
    if (!direct_.empty()) {   // #286: from the drive; a failed read falls back to the mapping below
        const Fill f{0, layer, expert, dst};
        if (read_direct(&f, 1)) return true;
        direct_fallbacks_.fetch_add(1, std::memory_order_relaxed);
    }
#if defined(__linux__)
    if (io_pf_ || io_stats_) {   // count what the page cache holds; with io prefetch, read what it does not with pread
        Span sp[3];
        const int ns = spans(layer, expert, sp);
        for (int i = 0; i < ns; ++i) {
            const uint64_t res = resident_in(sp[i].p, sp[i].n);
            io_cached_.fetch_add(res, std::memory_order_relaxed);
            io_uncached_.fetch_add(sp[i].n - res, std::memory_order_relaxed);
            const bool dec = in_decode();
            if (gate_on_ && dec) {
                gate_c_.fetch_add(res, std::memory_order_relaxed);
                gate_u_.fetch_add(sp[i].n - res, std::memory_order_relaxed);
            }
            if (io_pf_ && res < sp[i].n && sp[i].fd >= 0 && !(dec && !gate_open())) {
                const auto t0 = std::chrono::steady_clock::now();
                const bool ok = pread_full(sp[i].fd, dst + sp[i].at, sp[i].n, sp[i].off);
                io_pread_us_.fetch_add((uint64_t) std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count(),
                                       std::memory_order_relaxed);
                io_pread_n_.fetch_add(1, std::memory_order_relaxed);
                if (ok) {
                    io_pread_bytes_.fetch_add(sp[i].n, std::memory_order_relaxed);
                    continue;
                }
            }
            std::memcpy(dst + sp[i].at, sp[i].p, (size_t) sp[i].n);
        }
        if (ns > 0) return true;
    }
#endif
    if (!role_ptr_.empty()) {
        if (unmapped_) return false;   // drop_mapping closed the GGUF views: only the unbuffered reads remain
        uint64_t at = 0;
        for (int r = 0; r < 3; ++r) {
            const size_t i = (size_t) (3 * layer + r);
            const uint64_t per = role_bytes_[i];
            std::memcpy(dst + at, role_ptr_[i] + (size_t) ((uint64_t) expert * per), (size_t) per);
            at += per;
        }
        return true;
    }
    const uint8_t* b = mapped_blob(layer, expert);
    if (b == nullptr) return false;
    std::memcpy(dst, b, (size_t) layer_blob_bytes_[(size_t) layer]);
    return true;
}

// Pinned stage buffers (PR 1237): opt-in (STRATA_STAGE_PIN=1).  On by default they corrupted IQ3_S decode after a 4096-token
// prompt in the 0.1.41 release gate (48 tokens of id 0; off = identical to 0.1.40.4): with page-locked buffers the cache-fill
// copy is truly asynchronous, so a recycled stage buffer can be overwritten before its copy finishes.  Off until that is fixed.
#define STAGE_PIN_DEFAULT 0
constexpr uint64_t kStagePinFloor = 3ull << 30;   // RAM left available after a pinned stage buffer
void FileExpertSource::StageBufFree::operator()(uint8_t* p) const noexcept {
    if (pinned) (void) cudaFreeHost(p);
    else delete[] p;
}

// A blob assembled from the three role slices.  The buffer of a (layer, expert) is reused for another only once
// its blob has not been asked for during `kStageAge` layers (begin_layer) or 256 assemblies, whichever comes
// first, and never while it is being filled: the pool computes a layer's misses before it starts the next, and a
// fill (the GPU cache at startup, an adaptive swap, a helper GPU) copies the blob right away.
//
// `claim_stage` finds or reserves the buffer of `key` (stage_mu_ held): true when the blob is already there (or
// being filled by another thread - the caller then waits), false when the caller must fill buffer `v`.
bool FileExpertSource::claim_stage(int64_t key, size_t& v, bool& fill, bool ahead) {
    constexpr uint64_t kStageSeq = 256;
    const uint64_t seq = ++stage_seq_;
    auto it = stage_of_.find(key);
    fill = false;
    if (it != stage_of_.end()) {
        v = it->second;
        stage_epoch_[v] = epoch_;
        stage_used_[v] = seq;
        if (!ahead && stage_pf_[v]) {   // a layer asked for what was read ahead
            stage_pf_[v] = 0;
            io_pf_used_.fetch_add(1, std::memory_order_relaxed);
            io_pf_used_bytes_.fetch_add(layer_blob_bytes_[(size_t) (key / n_expert_)], std::memory_order_relaxed);
        }
        return true;
    }
    v = stage_buf_.size();
    uint64_t oldest = std::numeric_limits<uint64_t>::max();
    for (size_t i = 0; i < stage_buf_.size(); ++i)
        if (!stage_busy_[i] && (stage_epoch_[i] + kStageAge <= epoch_ || stage_used_[i] + kStageSeq <= seq) &&
            stage_used_[i] < oldest) {
            oldest = stage_used_[i];
            v = i;
        }
    if (v == stage_buf_.size()) {
        // Pinned, where the driver allows: a stage buffer is the source of the cache fill's cudaMemcpyAsync,
        // and a pageable source is staged through the driver's bounce buffer - an extra copy at a fraction of
        // the transfer rate, on the calling thread.  A failure falls back to pageable for this buffer.
        // STRATA_STAGE_PIN: 0 keeps them pageable (as before 0.1.41), 1 always tries to pin, unset (the default) pins while
        // the host has kStagePinFloor of RAM to spare beside this buffer (the pool grows to ~257 buffers, 0.4-0.6 GiB):
        // a low-RAM box keeps pageable ones instead of paying for the pinning (recommend, never force).
        static const int pin_mode = [] { const char* v = std::getenv("STRATA_STAGE_PIN"); return v == nullptr ? -1 : (std::atoi(v) != 0 ? 1 : 0); }();
        bool want_pin = pin_mode != 0;
        if (pin_mode < 0) {
            uint64_t avail = 0;
            want_pin = STAGE_PIN_DEFAULT && available_memory_bytes(avail) && avail >= (uint64_t) stage_blob_ + kStagePinFloor;
        }
        void* p = nullptr;
        cudaError_t pin_err = cudaSuccess;
        if (want_pin) pin_err = cudaHostAlloc(&p, (size_t) stage_blob_, cudaHostAllocDefault);
        const bool pinned = want_pin && pin_err == cudaSuccess && p != nullptr;
        if (!pinned) (void) cudaGetLastError();   // the failed alloc's sticky error is ours, not the caller's
        uint8_t* raw = pinned ? (uint8_t*) p : new (std::nothrow) uint8_t[(size_t) stage_blob_];
        if (raw == nullptr) return false;
        stage_buf_.emplace_back(raw, StageBufFree{pinned});
        stage_idx_[raw] = stage_buf_.size() - 1;
        stage_ev_.push_back(nullptr);
        stage_ev_live_.push_back(0);
        if (pinned && !stage_pin_said_) {
            stage_pin_said_ = true;
            std::fprintf(stderr, "FileExpertSource: the stage buffers are pinned (cudaHostAlloc)\n");
        } else if (!pinned && want_pin && !stage_pin_failed_said_) {
            stage_pin_failed_said_ = true;
            std::fprintf(stderr, "FileExpertSource: cudaHostAlloc failed for a stage buffer (%s) - pageable it is\n",
                         cudaGetErrorString(pin_err));
        }
        stage_key_.push_back(-1);
        stage_epoch_.push_back(0);
        stage_used_.push_back(0);
        stage_busy_.push_back(0);
        stage_pf_.push_back(0);
        if (stage_buf_.size() == 512 && !stage_grew_) {
            stage_grew_ = true;
            std::fprintf(stderr, "FileExpertSource: %zu blobs assembled from the GGUF are in use at once (%.2f GiB)\n",
                         stage_buf_.size(), (double) stage_buf_.size() * (double) stage_blob_ / 1073741824.0);
        }
    } else {
        stage_of_.erase(stage_key_[v]);
        if (stage_pf_[v]) { stage_pf_[v] = 0; io_pf_unused_.fetch_add(1, std::memory_order_relaxed); }   // read ahead, never asked for
        // #1237: a page-locked buffer is read by the DMA after the copy call returned: it is refilled only once the
        // last copy queued out of it (note_async_read) has finished.  Rare: the pool is sized so that a buffer asked
        // for has long been consumed; a wait here means the stream is more than a pool behind.
        if (v < stage_ev_live_.size() && stage_ev_live_[v]) {
            stage_ev_live_[v] = 0;
            if (stage_ev_[v] != nullptr) {
                if (cudaEventQuery((cudaEvent_t) stage_ev_[v]) == cudaErrorNotReady) {
                    stage_fence_waits_.fetch_add(1, std::memory_order_relaxed);
                    (void) cudaEventSynchronize((cudaEvent_t) stage_ev_[v]);
                }
                (void) cudaGetLastError();
            }
        }
    }
    stage_key_[v] = key;
    stage_epoch_[v] = epoch_;
    stage_used_[v] = seq;
    stage_busy_[v] = 1;
    stage_of_[key] = v;
    fill = true;
    return false;
}

// 149 STRATA_IO_PF_GATE: whether this thread reads for the decode (the dispatch's thread, and the fetch threads it starts)
namespace { thread_local bool tl_decode = false; }
FileExpertSource::DecodeScope::DecodeScope() : prev(tl_decode) { tl_decode = true; }
FileExpertSource::DecodeScope::~DecodeScope() { tl_decode = prev; }
bool FileExpertSource::in_decode() { return tl_decode; }

// once a decode window: the share of the window's demand bytes the page cache held, an EWMA, and the gate's state
void FileExpertSource::gate_tick() {
    if (!gate_on_) return;
    const uint64_t c = gate_c_.load(std::memory_order_relaxed), u = gate_u_.load(std::memory_order_relaxed);
    const uint64_t dc = c - gate_prev_c_, du = u - gate_prev_u_;
    gate_prev_c_ = c;
    gate_prev_u_ = u;
    if (dc + du < (64ull << 20)) {   // too few bytes read this window to judge; the state stays
        if (!gate_open_.load(std::memory_order_relaxed)) ++gate_closed_w_;
        return;
    }
    const double h = (double) dc / (double) (dc + du);
    gate_ewma_ = gate_ewma_ < 0.0 ? h : 0.8 * gate_ewma_ + 0.2 * h;   // reported only
    if (!gate_open_.load(std::memory_order_relaxed)) ++gate_closed_w_;
}

namespace {
// The room the page cache has for the experts: in a memory-limited cgroup (v2; the tightest limit of the group and its
// ancestors) the limit less what is not page cache (the group's memory.current - its "file" pages); otherwise
// MemAvailable (free + reclaimable cache).  0 when unknown.
uint64_t gate_cache_room() {
#if defined(__linux__)
    auto read_u64 = [](const std::string& path, uint64_t& v) {
        std::ifstream f(path);
        std::string s;
        if (!(f >> s) || s == "max") return false;
        v = std::strtoull(s.c_str(), nullptr, 10);
        return v > 0;
    };
    std::string rel;
    {
        std::ifstream g("/proc/self/cgroup");
        std::string line;
        while (std::getline(g, line))
            if (line.rfind("0::", 0) == 0) { rel = line.substr(3); break; }
    }
    uint64_t best = 0;
    if (!rel.empty()) {
        const std::string base = "/sys/fs/cgroup";
        for (std::string at = rel;; ) {
            const std::string dir = base + (at == "/" ? std::string() : at);
            uint64_t mx = 0, cur = 0, file = 0;
            if (read_u64(dir + "/memory.max", mx) && read_u64(dir + "/memory.current", cur)) {
                std::ifstream st(dir + "/memory.stat");
                std::string k;
                uint64_t v = 0;
                while (st >> k >> v)
                    if (k == "file") { file = v; break; }
                const uint64_t used = cur > file ? cur - file : 0;
                const uint64_t room = mx > used ? mx - used : 0;
                if (best == 0 || room < best) best = room;
            }
            if (at.empty() || at == "/") break;
            const size_t cut = at.find_last_of('/');
            at = cut == 0 || cut == std::string::npos ? std::string("/") : at.substr(0, cut);
        }
    }
    if (best > 0) return best;
    std::ifstream info("/proc/meminfo");
    std::stringstream text;
    text << info.rdbuf();
    return detail::meminfo_bytes(text.str(), "MemAvailable");
#else
    return 0;
#endif
}
}

double FileExpertSource::cache_room_ratio() const {
    const uint64_t total = expert_bytes(), room = gate_cache_room();
    return total == 0 || room == 0 ? 0.0 : (double) room / (double) total;
}

void FileExpertSource::gate_capacity() {
    if (!gate_on_ || (gate_win_++ % 16) != 0) return;
    const uint64_t total = expert_bytes(), room = gate_cache_room();
    if (total == 0 || room == 0) return;   // unknown: the state stays
    gate_ratio_ = (double) room / (double) total;
    // and the read-ahead must pay: the share of its bytes a layer then used, over these 16 windows (64 MB at least read
    // ahead to judge).  Below STRATA_IO_PF_GATE_USEFUL (0.10) it closes (a RAM tier holds what it would read: an 18 GiB
    // RAM tier, 0% used, -9..-15% decode); closed for that reason, it opens again for a probe every 256 windows.
    const uint64_t pf = io_pf_bytes_.load(std::memory_order_relaxed), used = io_pf_used_bytes_.load(std::memory_order_relaxed);
    const uint64_t dpf = pf - gate_prev_pf_, dused = used - gate_prev_used_;
    gate_prev_pf_ = pf;
    gate_prev_used_ = used;
    if (dpf >= (64ull << 20)) gate_useful_ = (double) dused / (double) dpf;
    const bool open = gate_open_.load(std::memory_order_relaxed);
    const bool room_ok = open ? gate_ratio_ >= gate_lo_ : gate_ratio_ > gate_hi_;
    bool want = room_ok;
    // too little to read ahead (< 128 MB in 16 windows: the RAM and VRAM tiers hold what is predicted) costs the look-ahead's
    // second layer and the I/O threads for nothing (an 18 GiB RAM tier: 2.5 GB read ahead in a run, -9..-16% decode)
    if (room_ok && open && gate_win_ > 16 && (dpf < (128ull << 20) || gate_useful_ < gate_min_useful_)) {
        want = false;
        gate_useless_ = true;
        gate_useless_since_ = gate_win_;
    } else if (room_ok && !open && gate_useless_) {
        want = gate_win_ - gate_useless_since_ >= 256;   // the probe
        if (want) gate_useless_ = false;
    }
    if (!room_ok) gate_useless_ = false;
    if (want != open) { gate_open_.store(want, std::memory_order_relaxed); ++gate_flips_; }
}

// Fills buffer `v` (reserved by claim_stage) outside the lock, then publishes it.
bool FileExpertSource::fill_stage(size_t v, int64_t layer, int64_t expert, uint8_t* dst) {
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = copy_from_files(layer, expert, dst);
    publish_stage(v, layer, ok, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    return ok;
}

void FileExpertSource::publish_stage(size_t v, int64_t layer, bool ok, double us) {
    file_us_.fetch_add((uint64_t) us, std::memory_order_relaxed);
    if (ok) {
        file_read_bytes_.fetch_add(layer_blob_bytes_[(size_t) layer], std::memory_order_relaxed);
        file_blob_bytes_.fetch_add(layer_blob_bytes_[(size_t) layer], std::memory_order_relaxed);
    }
    {
        std::lock_guard<std::mutex> lk(stage_mu_);
        stage_busy_[v] = 0;
        if (!ok) {
            stage_of_.erase(stage_key_[v]);
            stage_key_[v] = -1;
        }
    }
    stage_cv_.notify_all();
}

void FileExpertSource::note_async_read(const uint8_t* blob, void* stream) {
    if (blob == nullptr) return;
    std::lock_guard<std::mutex> lk(stage_mu_);
    const auto it = stage_idx_.find(blob);
    if (it == stage_idx_.end()) return;                 // not a stage buffer: the mapping or the resident arena
    const size_t v = it->second;
    if (!stage_buf_[v].get_deleter().pinned) return;    // pageable: the driver copied it before the call returned
    if (stage_ev_[v] == nullptr) {
        cudaEvent_t ev = nullptr;
        if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) {
            (void) cudaGetLastError();
            (void) cudaStreamSynchronize((cudaStream_t) stream);   // no event: wait for the copy itself
            return;
        }
        stage_ev_[v] = ev;
    }
    if (cudaEventRecord((cudaEvent_t) stage_ev_[v], (cudaStream_t) stream) != cudaSuccess) {
        (void) cudaGetLastError();
        (void) cudaStreamSynchronize((cudaStream_t) stream);
        return;
    }
    stage_ev_live_[v] = 1;
}

void FileExpertSource::stage_free_events() {
    for (void*& e : stage_ev_)
        if (e != nullptr) { (void) cudaEventDestroy((cudaEvent_t) e); e = nullptr; }
    stage_ev_.clear();
    stage_ev_live_.clear();
    stage_idx_.clear();
}

const uint8_t* FileExpertSource::staged_blob(int64_t layer, int64_t expert) {
    const int64_t key = layer * n_expert_ + expert;
    size_t v = 0;
    bool fill = false;
    uint8_t* dst = nullptr;
    const auto t0 = std::chrono::steady_clock::now();
    auto crit = [&] {
        if (!io_pf_) return;
        io_crit_us_.fetch_add((uint64_t) std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count(),
                              std::memory_order_relaxed);
        io_crit_n_.fetch_add(1, std::memory_order_relaxed);
    };
    {
        std::unique_lock<std::mutex> lk(stage_mu_);
        const bool have = claim_stage(key, v, fill);
        if (!have && !fill) return nullptr;
        dst = stage_buf_[v].get();
        if (have) {
            // another thread (a prefetch, the adaptive tier) is filling it: wait for that
            stage_cv_.wait(lk, [&] { return !stage_busy_[v] || stage_key_[v] != key; });
            if (stage_key_[v] != key) return nullptr;   // its fill failed
            lk.unlock();
            crit();
            return dst;
        }
    }
    const bool ok = fill_stage(v, layer, expert, dst);
    crit();
    return ok ? dst : nullptr;
}

void FileExpertSource::prefetch(int64_t layer, const int64_t* experts, int64_t n) {
    if (!staged() || n <= 0 || layer < 0 || layer >= n_layers_) return;
    if (io_fill_ && io_pf_ && direct_.empty()) {   // hand the layer's uncached blobs to the workers and go on: the kernels
        for (int64_t i = 0; i < n; ++i) {          // that reach one before it lands wait on its pages, the rest compute
            const int64_t e = experts[i];
            if (e < 0 || e >= n_expert_) continue;
            const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) e;
            if (pf_mark_ && pf_mark_[index].exchange(0, std::memory_order_relaxed)) {   // 149-prefetch: read ahead, now asked for
                io_pf_used_.fetch_add(1, std::memory_order_relaxed);
                io_pf_used_bytes_.fetch_add(layer_blob_bytes_[(size_t) layer], std::memory_order_relaxed);
            }
            if (complement_ready_ && resident_blob(index) != nullptr) continue;
            if (!override_.empty() && override_[index] != nullptr) continue;
            uint64_t total = 0;
            if (cached_bytes(layer, e, total, false) >= total) continue;
            io_pf_demand_.fetch_add(1, std::memory_order_relaxed);
            io_enqueue(layer, e);
        }
        return;
    }
    std::vector<Fill> todo;
    const auto t0 = std::chrono::steady_clock::now();
    claim_todo(layer, experts, n, todo, nullptr);
    fill_many(todo);
    if (io_pf_ && !todo.empty()) {
        io_crit_us_.fetch_add((uint64_t) std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count(),
                              std::memory_order_relaxed);
        io_crit_n_.fetch_add(1, std::memory_order_relaxed);
    }
}

void FileExpertSource::claim_todo(int64_t layer, const int64_t* experts, int64_t n, std::vector<Fill>& todo, uint8_t* pending,
                                  bool zc) {
    // 149-cold mode 3: the GGUF in place, mapped and buffered: an expert the page cache holds whole is computed from
    // the mapping (its three tensors' rows), not assembled into a stage buffer first.  Decided here, before the lock.
    zc_keys_.clear();
    std::vector<char> zc_ok;
    if (zc && !role_ptr_.empty() && !unmapped_ && direct_.empty() && strata::kernels::cpu::expert_layout().native &&
        strata::kernels::cpu::expert_layout().fmt[(size_t) layer].tail_off == 0) {
        zc_ok.assign((size_t) n, 0);
        for (int64_t i = 0; i < n; ++i) {
            const int64_t e = experts[i];
            if (e < 0 || e >= n_expert_) continue;
            const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) e;
            if (complement_ready_ && resident_blob(index) != nullptr) continue;   // in the RAM copy: no residency query
            if (!override_.empty() && override_[index] != nullptr) continue;
            uint64_t total = 0;
            const uint64_t c = cached_bytes(layer, e, total, true);
            zc_ok[(size_t) i] = total > 0 && c >= total ? 1 : 0;   // no residency query (not Linux): never
        }
    }
    // experts.bin with io prefetch: a blob whose pages are cached is handed out as the mapping (io_blob); only the
    // uncached ones are read here, together, on the fetch threads
    const bool only_uncached = io_pf_ && role_ptr_.empty() && direct_.empty();
    std::vector<char> cold;
    if (only_uncached) {
        cold.assign((size_t) n, 0);
        for (int64_t i = 0; i < n; ++i) {
            const int64_t e = experts[i];
            if (e < 0 || e >= n_expert_) continue;
            uint64_t total = 0;
            cold[(size_t) i] = cached_bytes(layer, e, total, false) < total ? 1 : 0;
        }
    }
    {
        std::lock_guard<std::mutex> lk(stage_mu_);
        for (int64_t i = 0; i < n; ++i) {
            const int64_t e = experts[i];
            if (e < 0 || e >= n_expert_) continue;
            const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) e;
            if (complement_ready_ && resident_blob(index) != nullptr)
                continue;                                     // in the RAM copy
            if (!override_.empty() && override_[index] != nullptr) continue;
            if (only_uncached && !cold[(size_t) i] && stage_of_.find(layer * n_expert_ + e) == stage_of_.end()) continue;
            if (!zc_ok.empty() && zc_ok[(size_t) i] && stage_of_.find(layer * n_expert_ + e) == stage_of_.end()) {
                zc_keys_.push_back(layer * n_expert_ + e);   // read from the mapping by the kernels themselves
                continue;
            }
            size_t v = 0;
            bool fill = false;
            if (!claim_stage(layer * n_expert_ + e, v, fill) && fill) {
                todo.push_back({v, layer, e, stage_buf_[v].get()});
                if (pending != nullptr) pending[i] = 1;
                if (warm_stamp_) {
                    const uint32_t s = warm_stamp_[index].load(std::memory_order_relaxed);
                    if (s != 0 && (uint64_t) s + 3 >= epoch_ + 1) warm_hits_.fetch_add(1, std::memory_order_relaxed);
                }
            } else if (pending != nullptr && fill == false) {
                // being filled by another thread (a read ahead, the adaptive tier): not ready yet either
                const auto it = stage_of_.find(layer * n_expert_ + e);
                if (it != stage_of_.end() && stage_busy_[it->second]) pending[i] = 1;
            }
        }
    }
}

// 149-cold STRATA_COLD_OVERLAP=1: the claimed blobs are read on background threads (the decode thread computes the
// ready experts meanwhile); the same reads as fill_many, on as many threads, each blob published as it lands.
int64_t FileExpertSource::prefetch_begin(int64_t layer, const int64_t* experts, int64_t n, uint8_t* pending) {
    for (int64_t i = 0; i < n; ++i) pending[i] = 0;
    if (!staged() || n <= 0 || layer < 0 || layer >= n_layers_) return 0;
    if (io_fill_ && io_pf_ && direct_.empty()) {   // the page-fill mode: nothing is staged here, the kernels wait on pages
        prefetch(layer, experts, n);
        return 0;
    }
    prefetch_end();
    async_todo_.clear();
    async_fill_us_ = 0;
    static const int mode_env = [] {   // as the dispatch reads it (0.2.0: unset = 4 for the GGUF read in place on Linux)
        const char* v = std::getenv("STRATA_COLD_OVERLAP");
        return v == nullptr || v[0] == '\0' ? -1 : std::atoi(v);
    }();
    const int mode = mode_env >= 0 ? mode_env : (linux_file_defaults() ? 4 : 0);
    claim_todo(layer, experts, n, async_todo_, pending, mode >= 3);
    if (mode == 4) {   // zero-copy only: the blobs that must be read are read here, as `prefetch` reads them
        const auto t0 = std::chrono::steady_clock::now();
        fill_many(async_todo_);
        if (io_pf_ && !async_todo_.empty()) {
            io_crit_us_.fetch_add((uint64_t) std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count(),
                                  std::memory_order_relaxed);
            io_crit_n_.fetch_add(1, std::memory_order_relaxed);
        }
        async_todo_.clear();
        for (int64_t i = 0; i < n; ++i) pending[i] = 0;   // one another thread still fills: `blob` waits for it
        return 0;
    }
    int64_t np = 0;
    for (int64_t i = 0; i < n; ++i) np += pending[i] ? 1 : 0;
    if (async_todo_.empty()) return np;
    async_t0_ = std::chrono::steady_clock::now();
    async_next_.store(0, std::memory_order_relaxed);
    async_end_ns_.store(0, std::memory_order_relaxed);
    const bool direct = !direct_.empty();
    const bool dec = tl_decode;
    auto work = [this, direct, dec] {
        tl_decode = dec;
        const std::vector<Fill>& todo = async_todo_;
        if (direct) {
            for (size_t at; (at = async_next_.fetch_add(16)) < todo.size();) {
                const size_t k = std::min<size_t>(16, todo.size() - at);
                const auto t0 = std::chrono::steady_clock::now();
                const bool ok = read_direct(todo.data() + at, k);
                const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                for (size_t i = at; i < at + k; ++i) {
                    const Fill& f = todo[i];
                    if (ok) publish_stage(f.v, f.layer, true, us / (double) k);
                    else (void) fill_stage(f.v, f.layer, f.e, f.dst);
                }
            }
        } else {
            for (size_t i; (i = async_next_.fetch_add(1)) < todo.size();)
                (void) fill_stage(todo[i].v, todo[i].layer, todo[i].e, todo[i].dst);
        }
        const int64_t ns = (int64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - async_t0_).count();
        int64_t cur = async_end_ns_.load(std::memory_order_relaxed);
        while (ns > cur && !async_end_ns_.compare_exchange_weak(cur, ns, std::memory_order_relaxed)) {}
    };
    const size_t nt = direct ? std::min<size_t>(4, (async_todo_.size() + 15) / 16)
                             : std::min<size_t>(async_todo_.size(), (size_t) std::max(1, fetch_threads_));
    for (size_t t = 0; t < nt; ++t) async_th_.emplace_back(work);
    return np;
}

int64_t FileExpertSource::wait_landed(int64_t layer, const int64_t* experts, int64_t n, uint8_t* state) {
    std::unique_lock<std::mutex> lk(stage_mu_);
    int64_t got = 0;
    stage_cv_.wait(lk, [&] {
        bool left = false;
        for (int64_t i = 0; i < n; ++i) {
            if (state[i] != 1) continue;
            const auto it = stage_of_.find(layer * n_expert_ + experts[i]);
            // published (or failed: then `blob` reads it again on this thread, as before)
            if (it == stage_of_.end() || !stage_busy_[it->second]) { state[i] = 2; ++got; }
            else left = true;
        }
        return got > 0 || !left;
    });
    return got;
}

bool FileExpertSource::zc_roles(int64_t layer, int64_t expert, const uint8_t** p) {
    const int64_t key = layer * n_expert_ + expert;
    if (std::find(zc_keys_.begin(), zc_keys_.end(), key) == zc_keys_.end()) return false;
    for (int r = 0; r < 3; ++r) {
        const size_t i = (size_t) (3 * layer + r);
        p[r] = role_ptr_[i] + (size_t) ((uint64_t) expert * role_bytes_[i]);
    }
    file_read_bytes_.fetch_add(layer_blob_bytes_[(size_t) layer], std::memory_order_relaxed);
    if (complement_ready_) file_reads_.fetch_add(1, std::memory_order_relaxed);
    reads_.fetch_add(1, std::memory_order_relaxed);
    if (io_stats_) io_cached_.fetch_add(layer_blob_bytes_[(size_t) layer], std::memory_order_relaxed);
    if (gate_on_ && tl_decode) gate_c_.fetch_add(layer_blob_bytes_[(size_t) layer], std::memory_order_relaxed);
    return true;
}

void FileExpertSource::prefetch_end() {
    if (async_th_.empty()) return;
    for (std::thread& t : async_th_) t.join();
    async_th_.clear();
    async_fill_us_ = (double) async_end_ns_.load(std::memory_order_relaxed) / 1e3;
    if (io_pf_ && !async_todo_.empty()) {   // the reads' wall time, as the synchronous fill counts it
        io_crit_us_.fetch_add((uint64_t) async_fill_us_, std::memory_order_relaxed);
        io_crit_n_.fetch_add(1, std::memory_order_relaxed);
    }
    async_todo_.clear();
}

void FileExpertSource::prefetch_pairs(const std::pair<int32_t, int32_t>* pairs, int64_t n) {
    if (direct_.empty() || pairs == nullptr || n <= 0) return;   // unbuffered only: the mapped fill stays as it was
    std::vector<Fill> todo;
    {
        std::lock_guard<std::mutex> lk(stage_mu_);
        for (int64_t i = 0; i < std::min<int64_t>(n, 64); ++i) {
            const int64_t l = pairs[i].first, e = pairs[i].second;
            if (l < 0 || e < 0 || l >= n_layers_ || e >= n_expert_) continue;
            size_t v = 0;
            bool fill = false;
            if (!claim_stage(l * n_expert_ + e, v, fill) && fill) todo.push_back({v, l, e, stage_buf_[v].get()});
        }
    }
    fill_many(todo);
}

bool FileExpertSource::advise_pairs(const std::pair<int32_t, int32_t>* pairs, int64_t n) const {
    if (base_ == nullptr || unmapped_ || !direct_.empty() || !strata::platform::read_ahead_enabled()) return false;
    for (int64_t i = 0; pairs != nullptr && i < n; ++i) {
        const int64_t l = pairs[i].first, e = pairs[i].second;
        if (l < 0 || e < 0 || l >= n_layers_ || e >= n_expert_) continue;
        if (role_ptr_.empty()) {   // experts.bin: the blob is one contiguous range
            if (const uint8_t* b = mapped_blob(l, e)) strata::platform::advise_willneed(b, layer_blob_bytes_[(size_t) l]);
            continue;
        }
        for (int r = 0; r < 3; ++r) {
            const size_t k = (size_t) (3 * l + r);
            strata::platform::advise_willneed(role_ptr_[k] + (size_t) ((uint64_t) e * role_bytes_[k]), role_bytes_[k]);
        }
    }
    return true;
}

void FileExpertSource::fill_many(const std::vector<Fill>& todo) {
    if (todo.empty()) return;
    if (!direct_.empty()) {
        // #286: overlapped batches - every role window of 16 blobs in flight at once; a big batch (the profile
        // fill) on up to 4 threads, a decode layer's few misses on this one
        std::atomic<size_t> next{0};
        auto work = [&] {
            for (size_t at; (at = next.fetch_add(16)) < todo.size();) {
                const size_t k = std::min<size_t>(16, todo.size() - at);
                const auto t0 = std::chrono::steady_clock::now();
                const bool ok = read_direct(todo.data() + at, k);
                const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                for (size_t i = at; i < at + k; ++i) {
                    const Fill& f = todo[i];
                    // a failed batch: each blob again on its own (copy_from_files falls back to the mapping)
                    if (ok) publish_stage(f.v, f.layer, true, us / (double) k);
                    else (void) fill_stage(f.v, f.layer, f.e, f.dst);
                }
            }
        };
        const size_t nt = std::min<size_t>(4, (todo.size() + 15) / 16);
        std::vector<std::thread> th;
        for (size_t t = 1; t < nt; ++t) th.emplace_back(work);
        work();
        for (auto& t : th) t.join();
        return;
    }
#if defined(_WIN32)
    // One PrefetchVirtualMemory call for every slice about to be copied: the memory manager reads them in large
    // requests, all queued at once, where the copies' page faults would read a few clusters each.  The copies below
    // then find the pages resident (or in flight).  STRATA_FETCH_PVM=0 is the A/B arm.
    {
        using Pvm = BOOL(WINAPI*)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
        static const Pvm pvm = [] {
            const char* v = std::getenv("STRATA_FETCH_PVM");
            if (v != nullptr && std::atoi(v) == 0) return (Pvm) nullptr;
            return (Pvm) (void*) GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory");
        }();
        if (pvm != nullptr) {
            std::vector<WIN32_MEMORY_RANGE_ENTRY> ranges;
            ranges.reserve(todo.size() * 3);
            for (const Fill& f : todo)
                for (int r = 0; r < 3; ++r) {
                    const size_t i = (size_t) (3 * f.layer + r);
                    ranges.push_back({(PVOID) (role_ptr_[i] + (size_t) ((uint64_t) f.e * role_bytes_[i])),
                                      (SIZE_T) role_bytes_[i]});
                }
            (void) pvm(GetCurrentProcess(), (ULONG_PTR) ranges.size(), ranges.data(), 0);
        }
    }
#endif
    // the page faults of a mapped read are one outstanding request each: several threads keep the SSD's queue full
    std::atomic<size_t> next{0};
    const bool dec = tl_decode;
    auto work = [&] {
        tl_decode = dec;
        for (size_t i; (i = next.fetch_add(1)) < todo.size();)
            (void) fill_stage(todo[i].v, todo[i].layer, todo[i].e, todo[i].dst);
    };
    const size_t nt = std::min<size_t>(todo.size(), (size_t) fetch_threads_);
    std::vector<std::thread> th;
    for (size_t t = 1; t < nt; ++t) th.emplace_back(work);
    work();
    for (auto& t : th) t.join();
}

uint64_t FileExpertSource::expert_bytes() const {
    uint64_t total = 0;
    for (uint64_t b : layer_blob_bytes_) total += b * (uint64_t) n_expert_;
    return total;
}

bool FileExpertSource::open_direct(std::string& why) {
#if defined(_WIN32)
    for (const std::string& path : paths_) {
        const int wide = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
        std::vector<wchar_t> w((size_t) (wide > 0 ? wide : 1), L'\0');
        if (wide > 0) MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, w.data(), wide);
        HANDLE h = CreateFileW(w.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                               FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
        if (h == INVALID_HANDLE_VALUE) {
            why += "; cannot open " + path + " unbuffered (error " + std::to_string((unsigned long long) GetLastError()) +
                   "), read through the file cache";
            for (void* d : direct_) CloseHandle((HANDLE) d);
            direct_.clear();
            return false;
        }
        direct_.push_back(h);
    }
    if (role_ptr_.empty()) {   // experts.bin: blob() now assembles into the stage buffers, sized for the largest blob
        std::lock_guard<std::mutex> lk(stage_mu_);
        for (uint64_t b : layer_blob_bytes_) stage_blob_ = std::max(stage_blob_, b);
    }
    return true;
#elif defined(__linux__)
    // O_DIRECT: the reads bypass the page cache, as FILE_FLAG_NO_BUFFERING does (4 KiB-aligned windows, read_direct)
    for (const std::string& path : paths_) {
        const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
        if (fd < 0) {
            why += "; cannot open " + path + " with O_DIRECT (" + std::strerror(errno) + "), read through the file cache";
            for (void* d : direct_) close_direct(d);
            direct_.clear();
            return false;
        }
        direct_.push_back((void*) (intptr_t) fd);
    }
    if (role_ptr_.empty()) {   // experts.bin: blob() now assembles into the stage buffers, sized for the largest blob
        std::lock_guard<std::mutex> lk(stage_mu_);
        for (uint64_t b : layer_blob_bytes_) stage_blob_ = std::max(stage_blob_, b);
    }
    return true;
#else
    (void) why;
    return false;
#endif
}

bool FileExpertSource::open_prompt_direct(std::string& why) {
#if defined(__linux__)
    if (base_ == nullptr || paths_.empty()) { why = "no expert files open"; return false; }
    if (!direct_.empty()) { why = "the file tier reads unbuffered already"; return false; }
    if (!pdirect_.empty()) return true;
    for (const std::string& path : paths_) {
        const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
        if (fd < 0) {
            why = "cannot open " + path + " with O_DIRECT (" + std::strerror(errno) + ")";
            for (void* d : pdirect_) close_direct(d);
            pdirect_.clear();
            return false;
        }
        pdirect_.push_back((void*) (intptr_t) fd);
    }
    return true;
#else
    why = "Linux only (NTFS serves unbuffered reads of a mapped file one at a time)";
    return false;
#endif
}

bool FileExpertSource::set_unbuffered(uint64_t ram_bytes, std::string& why) {
#if defined(_WIN32) || defined(__linux__)
    if (base_ == nullptr || paths_.empty() || !direct_.empty()) {
        why = !direct_.empty() ? "already unbuffered" : "no expert files open";
        return !direct_.empty();
    }
    // #577: the file tier reads the experts outside the RAM copy, not every byte of the shards (dense weights, the
    // PLE table), and the copy holds at most every expert - the budget asked for can be more than that
    const uint64_t experts = expert_bytes();
    const uint64_t arena = std::min(ram_bytes, experts);
    if (!file_tier_unbuffered(paths_, arena, experts - arena, why)) return false;
    return open_direct(why);
#else
    (void) ram_bytes;
    why = "through the file cache (no unbuffered reads on this platform)";
    return false;
#endif
}

bool FileExpertSource::recheck_unbuffered(std::string& why) {
#if defined(_WIN32) || defined(__linux__)
    if (const char* env = std::getenv("STRATA_UNBUFFERED_LOAD"); env != nullptr && env[0] != '\0') {
        why = std::string("STRATA_UNBUFFERED_LOAD=") + env;
        return !direct_.empty();
    }
    if (base_ == nullptr || paths_.empty()) {
        why = "no expert files open";
        return false;
    }
    // #577: the RAM copy is built (and already out of the available RAM), so what the file cache would have to keep
    // is exactly the experts outside it - the GPU cache's (refilled after a prompt borrowed their slots) and the
    // ones neither holds
    const uint64_t experts = expert_bytes();
    const uint64_t read = experts > complement_bytes_ ? experts - complement_bytes_ : 0;
    std::string w;
    const bool ub = file_tier_unbuffered(paths_, 0, read, w);
    char head[96];
    std::snprintf(head, sizeof head, "re-checked with the RAM copy built (%.2f GiB): ",
                  (double) complement_bytes_ / 1073741824.0);
    why = head + w;
    if (ub == !direct_.empty()) return ub;
    if (ub) return open_direct(why);
    // through the file cache after all: the mapped reads take over (staged() stays true for the GGUF in place)
    for (void* d : direct_) close_direct(d);
    direct_.clear();
    return false;
#else
    why = "through the file cache (no unbuffered reads on this platform)";
    return false;
#endif
}

bool FileExpertSource::drop_mapping(std::string& why) {
#if defined(_WIN32)
    if (direct_.empty()) { why = "the reads are not unbuffered"; return false; }
    if (!role_ptr_.empty() || !maps_.empty()) {
        // the GGUF in place: nothing in the engine reads the shards through these views any more (the token embedding is
        // copied into pinned host memory at load, the dense weights and the head into VRAM, the PLE table has its own
        // reader), so every shard goes: with one mapped, NTFS runs the unbuffered reads of that file one at a time
        if (unmapped_) { why = "already closed"; return true; }
        for (Map& m : maps_) {
            if (m.closed) continue;
            if (m.base != nullptr && !UnmapViewOfFile((LPCVOID) m.base)) { why = "UnmapViewOfFile failed"; return false; }
            m.closed = true;   // m.base stays: the read offsets are taken from it
            if (m.mapping != nullptr) CloseHandle((HANDLE) m.mapping);
            m.mapping = nullptr;
            if (m.file != nullptr) CloseHandle((HANDLE) m.file);
            m.file = nullptr;
        }
        unmapped_ = true;
        char note[160];
        std::snprintf(note, sizeof note, "%zu GGUF shard%s unmapped; every expert outside RAM and VRAM is read unbuffered",
                      maps_.size(), maps_.size() == 1 ? "" : "s");
        why = note;
        return true;
    }
    if (base_ == nullptr || unmapped_) { why = base_ == nullptr ? "not open" : "already closed"; return unmapped_; }
    if (!UnmapViewOfFile((LPCVOID) base_)) { why = "UnmapViewOfFile failed"; return false; }
    unmapped_ = true;
    if (mapping_ != nullptr) CloseHandle((HANDLE) mapping_);
    mapping_ = nullptr;
    // the cached handle too: only the unbuffered handles (direct_) read the file from here on
    if (file_ != nullptr) CloseHandle((HANDLE) file_);
    file_ = nullptr;
    why = "every expert outside RAM and VRAM is read unbuffered";
    return true;
#else
    // Linux's page cache has no such lock: only the tests close the GGUF views (STRATA_TEST_UNMAP_GGUF=1), to run the
    // unmapped path the Windows file tier takes
    const char* t = std::getenv("STRATA_TEST_UNMAP_GGUF");
    if (t == nullptr || t[0] == '\0' || std::atoi(t) == 0 || role_ptr_.empty()) { why = "not Windows"; return false; }
    if (direct_.empty()) { why = "the reads are not unbuffered"; return false; }
    if (unmapped_) { why = "already closed"; return true; }
    for (Map& m : maps_) {
        if (m.closed) continue;
        if (m.base != nullptr && munmap((void*) m.base, (size_t) m.bytes) != 0) { why = "munmap failed"; return false; }
        m.closed = true;
        if (m.fd >= 0) ::close(m.fd);
        m.fd = -1;
    }
    unmapped_ = true;
    why = std::to_string(maps_.size()) + " GGUF shard(s) unmapped (STRATA_TEST_UNMAP_GGUF); every expert outside RAM and VRAM is read unbuffered";
    return true;
#endif
}

bool FileExpertSource::begin_startup_unbuffered(std::string& why) {
#if defined(_WIN32)
    if (const char* v = std::getenv("STRATA_STARTUP_UNBUFFERED"); v != nullptr && v[0] != '\0' && std::atoi(v) == 0) {
        why = "STRATA_STARTUP_UNBUFFERED=0";
        return false;
    }
    if (base_ == nullptr || unmapped_ || startup_ub_ || !role_ptr_.empty() || !maps_.empty() || paths_.size() != 1 ||
        layer_blob_bytes_.empty()) {
        why = "experts.bin only";
        return false;
    }
    if (!direct_.empty()) { why = "the file tier reads unbuffered already"; return false; }
    const auto t0 = std::chrono::steady_clock::now();
    const uint8_t* m = mapped_blob(0, 0);   // the first blob as the mapping reads it, for the check below
    if (m == nullptr) { why = "the first blob is not mapped"; return false; }
    const std::vector<uint8_t> ref(m, m + (size_t) layer_blob_bytes_[0]);
    // NTFS runs the unbuffered reads of a file one at a time while it is mapped or open buffered (drop_mapping), so
    // the view, the section and the handle all go.  The first unbuffered open with no buffered handle left purges the
    // file's cached pages: 1.2-2.5 s after a mapped start had left IQ2_XS's 33 GiB experts.bin on the standby list,
    // 3 ms after an unbuffered start.  A buffered handle held open instead keeps the cache and serializes the reads
    UnmapViewOfFile((LPCVOID) base_);
    unmapped_ = true;
    if (mapping_ != nullptr) CloseHandle((HANDLE) mapping_);
    mapping_ = nullptr;
    if (file_ != nullptr) CloseHandle((HANDLE) file_);
    file_ = nullptr;
    std::string fail;
    if (open_direct(fail)) {
        // the first blob both ways: a drive that does not read it unbuffered keeps the mapped copy
        std::vector<uint8_t> probe(ref.size());
        const Fill f{0, 0, 0, probe.data()};
        if (read_direct(&f, 1) && std::memcmp(probe.data(), ref.data(), ref.size()) == 0) {
            startup_ub_ = true;
            startup_blob_bytes_ = file_blob_bytes_.load();
            startup_us_ = file_us_.load();
            char note[96];
            std::snprintf(note, sizeof note, "the view closed until the RAM copy is built; %.0f ms",
                          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            why = note;
            return true;
        }
        fail = "an unbuffered read of the first blob failed or differed from the mapping";
        for (void* d : direct_) close_direct(d);
        direct_.clear();
        std::lock_guard<std::mutex> lk(stage_mu_);
        stage_blob_ = 0;
    }
    std::string again;
    why = map_view(again) ? fail : fail + "; " + again;
    return false;
#else
    why = "Windows only";
    return false;
#endif
}

bool FileExpertSource::end_startup_unbuffered(std::string& why) {
    if (!startup_ub_) return true;
    startup_ub_ = false;
    // the view first: closing the unbuffered handles beside a buffered one purges nothing
    if (!map_view(why)) {
        why += ": the file tier reads it unbuffered";
        return false;   // as after drop_mapping: every read goes through read_direct
    }
    for (void* d : direct_) close_direct(d);
    direct_.clear();
    {   // the mapped tier hands out the mapping: the startup's stage buffers go
        std::lock_guard<std::mutex> lk(stage_mu_);
        stage_buf_.clear();
        stage_key_.clear();
        stage_epoch_.clear();
        stage_used_.clear();
        stage_busy_.clear();
        stage_pf_.clear();
        stage_of_.clear();
        stage_blob_ = 0;
    }
    file_blob_bytes_.store(startup_blob_bytes_);
    file_us_.store(startup_us_);
    return true;
}

bool FileExpertSource::map_view(std::string& why) {
#if defined(_WIN32)
    // open's mapping of experts.bin, again
    const std::string& path = paths_.front();
    const int wide = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::vector<wchar_t> wpath((size_t) (wide > 0 ? wide : 1), L'\0');
    if (wide > 0) MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wide);
    HANDLE f = CreateFileW(wpath.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                           nullptr);
    LARGE_INTEGER sz{};
    const bool same = f != INVALID_HANDLE_VALUE && GetFileSizeEx(f, &sz) && (uint64_t) sz.QuadPart == mapped_bytes_;
    HANDLE m = same ? CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr) : nullptr;
    void* view = m != nullptr ? MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0) : nullptr;
    if (view == nullptr) {
        why = f != INVALID_HANDLE_VALUE && !same
                  ? std::string("experts.bin changed size since it was opened")
                  : "experts.bin could not be mapped again (error " + std::to_string((unsigned long long) GetLastError()) + ")";
        if (m != nullptr) CloseHandle(m);
        if (f != INVALID_HANDLE_VALUE) CloseHandle(f);
        return false;
    }
    file_ = f;
    mapping_ = m;
    base_ = (const uint8_t*) view;
    unmapped_ = false;
    return true;
#else
    why = "Windows only";
    return false;
#endif
}

bool FileExpertSource::read_direct(const Fill* fills, size_t n) const {
#if defined(_WIN32)
    // NTFS runs the unbuffered reads of a file one at a time while the file is mapped or cached anywhere (the engine
    // maps the GGUF shards for the token embedding and the PLE table): measured on a PCIe 5 drive, 512 KiB reads at
    // queue depth 48 make 10.4 GB/s on a file nobody maps and 3.4 GB/s on a mapped one, 8 MiB reads 8.6 GB/s.  So
    // windows less than kGap apart (the same role of nearby experts) are merged into requests of up to kMerge bytes.
    constexpr uint64_t kSector = 4096, kGap = 1ull << 20, kMerge = 32ull << 20;
    struct Window { int file; uint64_t a0, size, skip, n, at; uint8_t* dst; size_t req; uint64_t in_req; };
    struct Req { HANDLE h; uint64_t a0, size, pos; };
    // this thread's aligned buffer and events, kept for its next batch
    struct Scratch {
        uint8_t* buf = nullptr;
        size_t cap = 0;
        std::vector<HANDLE> ev;
        std::vector<OVERLAPPED> ov;
        std::vector<Window> win;
        std::vector<size_t> order;
        std::vector<Req> req;
        std::vector<DWORD> got;
        ~Scratch() {
            if (buf != nullptr) VirtualFree(buf, 0, MEM_RELEASE);
            for (HANDLE e : ev) CloseHandle(e);
        }
    };
    thread_local Scratch sc;
    sc.win.clear();
    for (size_t k = 0; k < n; ++k) {
        const Fill& f = fills[k];
        if (f.dst == nullptr || f.layer < 0 || f.e < 0 || f.layer >= n_layers_ || f.e >= n_expert_) return false;
        if (role_ptr_.empty()) {   // experts.bin: the blob is one contiguous range
            const uint64_t per = layer_blob_bytes_[(size_t) f.layer];
            const uint64_t off = layer_offsets_[(size_t) f.layer] + (uint64_t) f.e * per;
            const uint64_t a0 = off / kSector * kSector, a1 = (off + per + kSector - 1) / kSector * kSector;
            sc.win.push_back({0, a0, a1 - a0, off - a0, per, 0, f.dst, 0, 0});
            continue;
        }
        uint64_t at = 0;
        for (int r = 0; r < 3; ++r) {
            const size_t i = (size_t) (3 * f.layer + r);
            const uint64_t per = role_bytes_[i];
            const Map& m = maps_[(size_t) role_file_[i]];
            const uint64_t off = (uint64_t) (role_ptr_[i] - m.base) + (uint64_t) f.e * per;
            const uint64_t a0 = off / kSector * kSector, a1 = (off + per + kSector - 1) / kSector * kSector;
            sc.win.push_back({role_file_[i], a0, a1 - a0, off - a0, per, at, f.dst, 0, 0});
            at += per;
        }
    }
    sc.order.resize(sc.win.size());
    for (size_t w = 0; w < sc.win.size(); ++w) sc.order[w] = w;
    std::sort(sc.order.begin(), sc.order.end(), [&](size_t a, size_t b) {
        return sc.win[a].file != sc.win[b].file ? sc.win[a].file < sc.win[b].file : sc.win[a].a0 < sc.win[b].a0;
    });
    sc.req.clear();
    uint64_t total = 0;
    int last_file = -1;
    for (size_t w : sc.order) {
        Window& x = sc.win[w];
        if (!sc.req.empty() && x.file == last_file) {
            Req& q = sc.req.back();
            const uint64_t end = q.a0 + q.size, xend = x.a0 + x.size;
            if (x.a0 <= end + kGap && std::max(end, xend) - q.a0 <= kMerge) {
                if (xend > end) {
                    total += xend - end;
                    q.size = xend - q.a0;
                }
                x.req = sc.req.size() - 1;
                x.in_req = x.a0 - q.a0;
                continue;
            }
        }
        sc.req.push_back({(HANDLE) direct_[(size_t) x.file], x.a0, x.size, total});
        total += x.size;
        last_file = x.file;
        x.req = sc.req.size() - 1;
        x.in_req = 0;
    }
    if (total > sc.cap) {
        if (sc.buf != nullptr) VirtualFree(sc.buf, 0, MEM_RELEASE);
        sc.cap = (size_t) ((total + (1u << 20) - 1) >> 20 << 20);
        sc.buf = (uint8_t*) VirtualAlloc(nullptr, sc.cap, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (sc.buf == nullptr) { sc.cap = 0; return false; }
    }
    while (sc.ev.size() < sc.req.size()) {
        HANDLE e = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (e == nullptr) return false;
        sc.ev.push_back(e);
    }
    sc.ov.assign(sc.req.size(), OVERLAPPED{});
    // every request in flight before the first wait: the drive sees the whole batch as one queue
    size_t issued = 0;
    bool ok = true;
    for (size_t q = 0; q < sc.req.size(); ++q) {
        const Req& r = sc.req[q];
        OVERLAPPED& o = sc.ov[q];
        o.Offset = (DWORD) r.a0;
        o.OffsetHigh = (DWORD) (r.a0 >> 32);
        o.hEvent = sc.ev[q];
        if (!ReadFile(r.h, sc.buf + r.pos, (DWORD) r.size, nullptr, &o) && GetLastError() != ERROR_IO_PENDING) {
            ok = false;
            break;
        }
        ++issued;
    }
    std::vector<DWORD>& got = sc.got;
    got.assign(sc.req.size(), 0);
    for (size_t q = 0; q < issued; ++q)
        if (!GetOverlappedResult(sc.req[q].h, &sc.ov[q], &got[q], TRUE)) ok = false;
    if (!ok || issued < sc.req.size()) return false;
    for (const Window& x : sc.win) {
        // a request may run past the end of the file: only the role's own bytes have to arrive
        if ((uint64_t) got[x.req] < x.in_req + x.skip + x.n) return false;
        std::memcpy(x.dst + x.at, sc.buf + sc.req[x.req].pos + x.in_req + x.skip, (size_t) x.n);
    }
    return true;
#elif defined(__linux__)
    // Linux: O_DIRECT descriptors (open_direct) and the kernel's asynchronous reads - every window of the batch is
    // queued with one io_submit before the first wait, so the drive sees the batch as one queue, as the overlapped
    // ReadFiles above do.  No NTFS here: windows are merged only where they touch (the 4 KiB rounding of adjacent
    // slices, consecutive experts of a role), never across a gap that would be read for nothing.  Each thread has its
    // own context and aligned buffer, kept for its next batch, so concurrent batches never see each other's events.
    constexpr uint64_t kSector = 4096, kMerge = 32ull << 20;
    constexpr long kDepth = 256;   // requests in flight per thread; a larger batch is read in waves of this many
    struct Window { int file; uint64_t a0, size, skip, n, at; uint8_t* dst; size_t req; uint64_t in_req; };
    struct Req { int fd; uint64_t a0, size, pos; };
    struct Scratch {
        uint8_t* buf = nullptr;
        size_t cap = 0;
        aio_context_t ctx = 0;
        std::vector<Window> win;
        std::vector<size_t> order;
        std::vector<Req> req;
        std::vector<int64_t> got;
        std::vector<iocb> cb;
        std::vector<iocb*> cbp;
        std::vector<io_event> ev;
        ~Scratch() {
            std::free(buf);
            if (ctx != 0) (void) syscall(SYS_io_destroy, ctx);
        }
    };
    thread_local Scratch sc;
    if (sc.ctx == 0 && syscall(SYS_io_setup, kDepth, &sc.ctx) != 0) {
        sc.ctx = 0;
        return false;   // no AIO (a seccomp profile, aio-max-nr): the caller falls back to the mapped copy
    }
    sc.win.clear();
    for (size_t k = 0; k < n; ++k) {
        const Fill& f = fills[k];
        if (f.dst == nullptr || f.layer < 0 || f.e < 0 || f.layer >= n_layers_ || f.e >= n_expert_) return false;
        if (role_ptr_.empty()) {   // experts.bin: the blob is one contiguous range
            const uint64_t per = layer_blob_bytes_[(size_t) f.layer];
            const uint64_t off = layer_offsets_[(size_t) f.layer] + (uint64_t) f.e * per;
            const uint64_t a0 = off / kSector * kSector, a1 = (off + per + kSector - 1) / kSector * kSector;
            sc.win.push_back({0, a0, a1 - a0, off - a0, per, 0, f.dst, 0, 0});
            continue;
        }
        uint64_t at = 0;
        for (int r = 0; r < 3; ++r) {
            const size_t i = (size_t) (3 * f.layer + r);
            const uint64_t per = role_bytes_[i];
            const Map& m = maps_[(size_t) role_file_[i]];
            const uint64_t off = (uint64_t) (role_ptr_[i] - m.base) + (uint64_t) f.e * per;
            const uint64_t a0 = off / kSector * kSector, a1 = (off + per + kSector - 1) / kSector * kSector;
            sc.win.push_back({role_file_[i], a0, a1 - a0, off - a0, per, at, f.dst, 0, 0});
            at += per;
        }
    }
    sc.order.resize(sc.win.size());
    for (size_t w = 0; w < sc.win.size(); ++w) sc.order[w] = w;
    std::sort(sc.order.begin(), sc.order.end(), [&](size_t a, size_t b) {
        return sc.win[a].file != sc.win[b].file ? sc.win[a].file < sc.win[b].file : sc.win[a].a0 < sc.win[b].a0;
    });
    sc.req.clear();
    uint64_t total = 0;
    int last_file = -1;
    for (size_t w : sc.order) {
        Window& x = sc.win[w];
        if (!sc.req.empty() && x.file == last_file) {
            Req& q = sc.req.back();
            const uint64_t end = q.a0 + q.size, xend = x.a0 + x.size;
            if (x.a0 <= end && std::max(end, xend) - q.a0 <= kMerge) {
                if (xend > end) {
                    total += xend - end;
                    q.size = xend - q.a0;
                }
                x.req = sc.req.size() - 1;
                x.in_req = x.a0 - q.a0;
                continue;
            }
        }
        sc.req.push_back({(int) (intptr_t) (direct_.empty() ? pdirect_ : direct_)[(size_t) x.file], x.a0, x.size, total});
        total += x.size;
        last_file = x.file;
        x.req = sc.req.size() - 1;
        x.in_req = 0;
    }
    if (total > sc.cap) {
        std::free(sc.buf);
        sc.cap = (size_t) ((total + (1u << 20) - 1) >> 20 << 20);
        void* p = nullptr;
        sc.buf = posix_memalign(&p, (size_t) kSector, sc.cap) == 0 ? (uint8_t*) p : nullptr;
        if (sc.buf == nullptr) { sc.cap = 0; return false; }
    }
    sc.got.assign(sc.req.size(), -1);
    sc.cb.resize(std::min<size_t>(sc.req.size(), (size_t) kDepth));
    sc.cbp.resize(sc.cb.size());
    sc.ev.resize(sc.cb.size());
    bool ok = true;
    for (size_t w0 = 0; ok && w0 < sc.req.size(); w0 += (size_t) kDepth) {
        const size_t wn = std::min<size_t>((size_t) kDepth, sc.req.size() - w0);
        for (size_t j = 0; j < wn; ++j) {
            const Req& r = sc.req[w0 + j];
            iocb& c = sc.cb[j];
            std::memset(&c, 0, sizeof c);
            c.aio_lio_opcode = IOCB_CMD_PREAD;
            c.aio_fildes = (uint32_t) r.fd;
            c.aio_buf = (uint64_t) (uintptr_t) (sc.buf + r.pos);
            c.aio_nbytes = r.size;
            c.aio_offset = (int64_t) r.a0;
            c.aio_data = (uint64_t) (w0 + j);
            sc.cbp[j] = &c;
        }
        // every request of the wave queued; a refused submit ends the batch once what was queued has completed
        // (the kernel still writes into the buffer until then)
        size_t queued = 0;
        while (queued < wn) {
            const long r = syscall(SYS_io_submit, sc.ctx, (long) (wn - queued), sc.cbp.data() + queued);
            if (r > 0) { queued += (size_t) r; continue; }
            if (r < 0 && errno == EINTR) continue;
            ok = false;
            break;
        }
        for (size_t done = 0; done < queued;) {
            const long r = syscall(SYS_io_getevents, sc.ctx, 1L, (long) (queued - done), sc.ev.data(), nullptr);
            if (r < 0) {
                if (errno == EINTR) continue;
                // the context is in an unknown state: drop it (io_destroy waits for what is still in flight)
                (void) syscall(SYS_io_destroy, sc.ctx);
                sc.ctx = 0;
                return false;
            }
            for (long j = 0; j < r; ++j) {
                const io_event& e = sc.ev[(size_t) j];
                sc.got[(size_t) e.data] = e.res;
                if (e.res < 0) ok = false;
            }
            done += (size_t) r;
        }
    }
    if (!ok) return false;
    for (const Window& x : sc.win) {
        // a request may run past the end of the file: only the role's own bytes have to arrive
        if (sc.got[x.req] < 0 || (uint64_t) sc.got[x.req] < x.in_req + x.skip + x.n) return false;
        std::memcpy(x.dst + x.at, sc.buf + sc.req[x.req].pos + x.in_req + x.skip, (size_t) x.n);
    }
    return true;
#else
    (void) fills; (void) n;
    return false;
#endif
}

void FileExpertSource::warm(int64_t layer, const int64_t* experts, int64_t n) {
    if ((role_ptr_.empty() && !io_pf_) || n <= 0 || layer < 0 || layer >= n_layers_) return;
    uint32_t stamp;
    {
        std::lock_guard<std::mutex> lk(stage_mu_);
        stamp = (uint32_t) epoch_ + 1;
    }
#if defined(_WIN32)
    using Pvm = BOOL(WINAPI*)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
    static const Pvm pvm = (Pvm) (void*) GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory");
    std::vector<WIN32_MEMORY_RANGE_ENTRY> ranges;
#endif
    for (int64_t j = 0; j < n; ++j) {
        const int64_t e = experts[j];
        if (e < 0 || e >= n_expert_) continue;
        const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) e;
        if (complement_ready_ && resident_blob(index) != nullptr)
            continue;                                                    // in the RAM copy
        if (warm_stamp_) warm_stamp_[index].store(stamp, std::memory_order_relaxed);
        warm_count_.fetch_add(1, std::memory_order_relaxed);
        if (io_pf_ && direct_.empty() && gate_open()) {   // read it ahead with pread on the I/O threads (instead of an madvise hint)
            if (io_ahead_) io_enqueue(layer, e, true);
            continue;
        }
        if (role_ptr_.empty() || unmapped_) continue;
        for (int r = 0; r < 3; ++r) {
            const size_t i = (size_t) (3 * layer + r);
            const uint8_t* p = role_ptr_[i] + (size_t) ((uint64_t) e * role_bytes_[i]);
#if defined(_WIN32)
            ranges.push_back({(PVOID) p, (SIZE_T) role_bytes_[i]});
#else
            const uintptr_t pg = 4096, a = (uintptr_t) p & ~(pg - 1);
            (void) madvise((void*) a, (size_t) ((uintptr_t) p + role_bytes_[i] - a), MADV_WILLNEED);
#endif
        }
    }
#if defined(_WIN32)
    if (pvm != nullptr && !ranges.empty()) (void) pvm(GetCurrentProcess(), (ULONG_PTR) ranges.size(), ranges.data(), 0);
#endif
}

static inline int la_pc64(uint64_t x) {   // (portable: no compiler builtin)
    int n = 0;
    for (; x != 0; x &= x - 1) ++n;
    return n;
}

RouterLookahead::~RouterLookahead() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        quit_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

void RouterLookahead::set_foresight(ForesightSwap* fs) {
    for (;;) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (fs != nullptr || !busy_) { fs_ = fs; return; }
        }
        // detaching: a prediction in progress finishes first (the swap space is about to go away)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

bool RouterLookahead::start(std::vector<std::vector<uint16_t>> routers, int64_t n_embd, int64_t n_expert, int k,
                            ExpertSource* src, std::string& err, bool allow_cold_source) {
    if (src == nullptr || (!allow_cold_source && !src->warms())) { err = "RouterLookahead: the expert source does not warm"; return false; }
    if (n_embd % 8 != 0) { err = "RouterLookahead: n_embd is not a multiple of 8"; return false; }
    for (const auto& r : routers)
        if (r.size() != (size_t) (n_embd * n_expert)) { err = "RouterLookahead: a router of another shape"; return false; }
    routers_ = std::move(routers);
    n_embd_ = n_embd;
    n_expert_ = n_expert;
    k_ = k < 1 ? 1 : k > (int) n_expert ? (int) n_expert : k;
    src_ = src;
    x_.assign((size_t) (8 * n_embd), 0.f);
    thread_ = std::thread([this] {
        strata::platform::name_this_thread("strata-lookahead");
        strata::aux_cpus::pin_current_thread();
        run();
    });
    return true;
}

void RouterLookahead::submit(int64_t layer, const float* x, int64_t n_tok, const int32_t* host_res) {
    if (layer + 1 >= (int64_t) routers_.size() || n_tok <= 0 || x == nullptr) return;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (busy_ || pending_) { skipped_.fetch_add(1, std::memory_order_relaxed); return; }
        n_tok_ = std::min<int64_t>(n_tok, 8);
        std::memcpy(x_.data(), x, (size_t) (n_tok_ * n_embd_) * sizeof(float));
        layer_ = layer + 1;
        host_res_ = host_res;
        sub_gen_ = gen_.load(std::memory_order_relaxed);
        pending_ = true;
    }
    cv_.notify_one();
}

void RouterLookahead::enable_stats(int64_t n_layers) {
    words_ = (n_expert_ + 63) / 64;
    pmask_.reset(new std::atomic<uint64_t>[(size_t) (n_layers * 3 * words_)]);
    pstamp_.reset(new std::atomic<uint64_t>[(size_t) n_layers]);
    for (int64_t i = 0; i < n_layers * 3 * words_; ++i) pmask_[(size_t) i].store(0, std::memory_order_relaxed);
    for (int64_t i = 0; i < n_layers; ++i) pstamp_[(size_t) i].store(0, std::memory_order_relaxed);
    pvote_.reset(new std::atomic<int32_t>[(size_t) (n_layers * kVoteN)]);
    for (int64_t i = 0; i < n_layers * kVoteN; ++i) pvote_[(size_t) i].store(0, std::memory_order_relaxed);
    stats_on_ = true;
}

// the dispatch thread: score the prediction made for `layer` against the experts it routes outside the GPU cache
void RouterLookahead::stats_score(int64_t layer, const int32_t* ids, int64_t n, const int32_t* host_res, int pcie_num) {
    if (!stats_on_ || layer < 1 || layer >= (int64_t) routers_.size()) return;
    ++st_layers_;
    if (pstamp_[(size_t) layer].load(std::memory_order_acquire) != gen_.load(std::memory_order_relaxed) + 1) {
        ++st_nopred_;
        return;
    }
    std::vector<uint64_t> act((size_t) words_, 0);
    for (int64_t i = 0; i < n; ++i) {
        const int32_t e = ids[i];
        if (e < 0 || e >= n_expert_) continue;
        if (host_res != nullptr && host_res[(size_t) (layer * n_expert_ + e)] >= 0) continue;
        act[(size_t) (e >> 6)] |= 1ull << (e & 63);
    }
    uint64_t na = 0;
    for (int64_t w = 0; w < words_; ++w) na += (uint64_t) la_pc64(act[(size_t) w]);
    st_actual_ += na;
    {   // the vote-ranked top M (the experts most tokens predict) against the misses, and against the PCIe share
        static const int vm[3] = {4, 8, 12};
        int32_t vt[kVoteN];
        int nv = 0;
        for (int i = 0; i < kVoteN; ++i) {
            const int32_t v = pvote_[(size_t) (layer * kVoteN + i)].load(std::memory_order_relaxed);
            if (v == 0) break;
            vt[nv++] = v - 1;
        }
        // the misses in routing order (first appearance), the PCIe share = the last (n_miss * pcie_num / 256) of them
        std::vector<int32_t> order_miss;
        for (int64_t i = 0; i < n; ++i) {
            const int32_t e = ids[i];
            if (e < 0 || e >= n_expert_) continue;
            if (host_res != nullptr && host_res[(size_t) (layer * n_expert_ + e)] >= 0) continue;
            if (std::find(order_miss.begin(), order_miss.end(), e) == order_miss.end()) order_miss.push_back(e);
        }
        const int m = ((int) order_miss.size() * pcie_num) >> 8;
        st_pch_ += (uint64_t) m;
        for (int s = 0; s < 3; ++s) {
            const int top = std::min(vm[s], nv);
            st_vpred_[s] += (uint64_t) top;
            for (int j = 0; j < top; ++j) {
                if ((act[(size_t) (vt[j] >> 6)] >> (vt[j] & 63)) & 1ull) ++st_vhit_[s];
                const auto it = std::find(order_miss.begin(), order_miss.end(), vt[j]);
                if (it != order_miss.end() && (int) (it - order_miss.begin()) >= (int) order_miss.size() - m) ++st_pcov_[s];
            }
        }
    }
    for (int s = 0; s < 3; ++s) {
        uint64_t hit = 0, np = 0;
        for (int64_t w = 0; w < words_; ++w) {
            const uint64_t pm = pmask_[(size_t) ((layer * 3 + s) * words_ + w)].load(std::memory_order_relaxed);
            hit += (uint64_t) la_pc64(pm & act[(size_t) w]);
            np += (uint64_t) la_pc64(pm);
        }
        st_hit_[s] += hit;
        st_pred_[s] += np;
    }
    if (layer == (int64_t) routers_.size() - 1 && (++st_windows_ % 100) == 0) {
        const uint64_t scored = st_layers_ - st_nopred_;
        std::fprintf(stderr, "strata lookahead stats: %llu windows, %llu layers scored, %llu without a prediction (%.1f%%); "
                     "%.2f actual non-resident experts/layer;",
                     (unsigned long long) st_windows_, (unsigned long long) scored, (unsigned long long) st_nopred_,
                     st_layers_ ? 100.0 * (double) st_nopred_ / (double) st_layers_ : 0.0,
                     scored ? (double) st_actual_ / (double) scored : 0.0);
        std::fprintf(stderr, " predictor thread %.2f ms/window;",
                     (double) busy_us_.load(std::memory_order_relaxed) / 1000.0 / (double) st_windows_);
        static const int add[3] = {0, 4, 10};
        for (int s = 0; s < 3; ++s)
            std::fprintf(stderr, " k=%d: recall %.3f precision %.3f (%.2f predicted/layer);", k_ + add[s],
                         st_actual_ ? (double) st_hit_[s] / (double) st_actual_ : 0.0,
                         st_pred_[s] ? (double) st_hit_[s] / (double) st_pred_[s] : 0.0,
                         scored ? (double) st_pred_[s] / (double) scored : 0.0);
        static const int vm[3] = {4, 8, 12};
        for (int s = 0; s < 3; ++s)
            std::fprintf(stderr, " votes top%d: %.2f/layer, covers %.3f of the misses (precision %.3f), %.3f of the PCIe share (%.2f/layer);",
                         vm[s], scored ? (double) st_vpred_[s] / (double) scored : 0.0,
                         st_actual_ ? (double) st_vhit_[s] / (double) st_actual_ : 0.0,
                         st_vpred_[s] ? (double) st_vhit_[s] / (double) st_vpred_[s] : 0.0,
                         st_pch_ ? (double) st_pcov_[s] / (double) st_pch_ : 0.0,
                         scored ? (double) st_pch_ / (double) scored : 0.0);
        std::fprintf(stderr, "\n");
    }
}

void RouterLookahead::run() {
    std::vector<float> logits((size_t) (8 * n_expert_));
    std::vector<int32_t> order((size_t) n_expert_);
    std::vector<int64_t> want;
    std::vector<float> wconf;   // 149-prefetch: per wanted expert, its best rank's confidence (warm_scored)
    for (;;) {
        int64_t layer, nt;
        const int32_t* host_res;
        ForesightSwap* fs;
        uint64_t sgen;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return quit_ || pending_; });
            if (quit_) return;
            pending_ = false;
            busy_ = true;
            layer = layer_;
            nt = n_tok_;
            host_res = host_res_;
            fs = fs_;
            sgen = sub_gen_;
        }
        const auto t0 = std::chrono::steady_clock::now();
        const int64_t layer0 = layer;
        const int depth = src_ != nullptr && !src_->read_ahead_open() ? 1 : depth_;   // 149: the gate closed
        for (int dj = 0; dj < depth && layer0 + dj < (int64_t) routers_.size(); ++dj) {
        layer = layer0 + dj;
        want.clear();
        wconf.clear();
        const bool scored = src_->wants_scores();
        // The router dot is AVX2 (kq_avx2.cpp); a CPU without AVX2 (the experimental older-CPU builds) takes the AVX1
        // one (kq_avx1.cpp) or, without AVX, the same sums in plain C++.  From the Strata_Dirigo fork (rwkeyes): an
        // AVX-only Xeon E5-2687W died here (vpmovzxwd) on its first request.  An estimate only (which experts to
        // prefetch); the order of the additions differs between the three, the output does not depend on it.
#if defined(__aarch64__) && !defined(STRATA_GPU_METAL)   // Metal (Apple silicon) keeps the plain loop below: its bf16_rows_dot_multi is a stub
        strata::kernels::cpu::bf16_rows_dot_multi(routers_[(size_t) layer].data(), (int) n_expert_, (int) n_embd_,
                                                  x_.data(), (int) nt, logits.data());
#else
        if (strata::kernels::cpu::cpu_avx2_ok()) {
            strata::kernels::cpu::bf16_rows_dot_multi(routers_[(size_t) layer].data(), (int) n_expert_, (int) n_embd_,
                                                      x_.data(), (int) nt, logits.data());
        } else if (strata::kernels::cpu::cpu_avx1_ok()) {
            strata::kernels::cpu::bf16_rows_dot_multi_avx1(routers_[(size_t) layer].data(), (int) n_expert_,
                                                           (int) n_embd_, x_.data(), (int) nt, logits.data());
        } else {
            // a mul and an add, not std::fma: without an FMA instruction that is a libm call per element (74x
            // slower on a Xeon E5-2665, measured by the fork)
            const uint16_t* rw = routers_[(size_t) layer].data();
            for (int64_t r = 0; r < n_expert_; ++r) {
                const uint16_t* wr = rw + (size_t) r * (size_t) n_embd_;
                for (int64_t t = 0; t < nt; ++t) {
                    const float* xr = x_.data() + (size_t) t * (size_t) n_embd_;
                    float acc = 0.0f;
                    for (int64_t c = 0; c < n_embd_; ++c) {
                        const uint32_t bits = (uint32_t) wr[c] << 16;
                        float wf;
                        std::memcpy(&wf, &bits, sizeof wf);
                        acc += wf * xr[c];
                    }
                    logits[(size_t) t * (size_t) n_expert_ + (size_t) r] = acc;
                }
            }
        }
#endif
        const int kmax = stats_on_ ? (int) std::min<int64_t>(k_ + 10, n_expert_) : k_;
        const int ks[3] = {k_, (int) std::min<int64_t>(k_ + 4, n_expert_), kmax};
        std::vector<uint64_t> sm;
        if (stats_on_) sm.assign((size_t) (3 * words_), 0);
        for (int64_t t = 0; t < nt; ++t) {
            const float* lt = logits.data() + (size_t) (t * n_expert_);
            for (int64_t e = 0; e < n_expert_; ++e) order[(size_t) e] = (int32_t) e;
            std::partial_sort(order.begin(), order.begin() + kmax, order.end(),
                              [&](int32_t a, int32_t b) { return lt[(size_t) a] > lt[(size_t) b]; });
            for (int j = 0; j < kmax; ++j) {
                const int64_t e = order[(size_t) j];
                if (host_res != nullptr && host_res[(size_t) (layer * n_expert_ + e)] >= 0) continue;   // on the GPU
                if (j < k_) {
                    const auto it = std::find(want.begin(), want.end(), e);
                    // confidence: 1 for a token's top pick down to 1/k_ for its k_-th, the best over the window's tokens,
                    // halved for every layer further ahead than the next (a deeper guess is less often right)
                    const float c = (float) (k_ - j) / (float) k_ * (dj == 0 ? 1.0f : std::pow(0.5f, (float) dj));
                    if (it == want.end()) { want.push_back(e); wconf.push_back(c); }
                    else if (scored && c > wconf[(size_t) (it - want.begin())]) wconf[(size_t) (it - want.begin())] = c;
                }
                if (stats_on_ && dj == 0)
                    for (int s = 0; s < 3; ++s)
                        if (j < ks[s]) sm[(size_t) (s * words_ + (e >> 6))] |= 1ull << (e & 63);
            }
        }
        if (stats_on_ && dj == 0) {
            // votes: how many tokens put each non-resident expert in their top k_
            std::vector<std::pair<int, int32_t>> votes;   // (sum of 64 - rank, expert)
            for (int64_t t = 0; t < nt; ++t) {
                const float* lt = logits.data() + (size_t) (t * n_expert_);
                for (int64_t e = 0; e < n_expert_; ++e) order[(size_t) e] = (int32_t) e;
                std::partial_sort(order.begin(), order.begin() + k_, order.end(),
                                  [&](int32_t a, int32_t b) { return lt[(size_t) a] > lt[(size_t) b]; });
                for (int j = 0; j < k_; ++j) {
                    const int32_t e = order[(size_t) j];
                    if (host_res != nullptr && host_res[(size_t) (layer * n_expert_ + e)] >= 0) continue;
                    auto it = std::find_if(votes.begin(), votes.end(), [&](const auto& v) { return v.second == e; });
                    if (it == votes.end()) votes.emplace_back(64 - j, e);
                    else it->first += 64 - j;
                }
            }
            std::sort(votes.begin(), votes.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            for (int i = 0; i < kVoteN; ++i)
                pvote_[(size_t) (layer * kVoteN + i)].store(i < (int) votes.size() ? votes[(size_t) i].second + 1 : 0,
                                                            std::memory_order_relaxed);
            for (int64_t i = 0; i < 3 * words_; ++i)
                pmask_[(size_t) (layer * 3 * words_ + i)].store(sm[(size_t) i], std::memory_order_relaxed);
            pstamp_[(size_t) layer].store(sgen + 1, std::memory_order_release);
        }
        if (scored) src_->warm_scored(layer, want.data(), wconf.data(), (int64_t) want.size(), dj + 1);
        else src_->warm(layer, want.data(), (int64_t) want.size());
        if (fs != nullptr && dj == 0)   // the next layer only: a deeper guess is too often wrong to spend a copy on
            for (const int64_t e : want) fs->predict(src_, layer, (int32_t) e);
        predicted_.fetch_add((int64_t) want.size(), std::memory_order_relaxed);
        }
        busy_us_.fetch_add((uint64_t) std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count(),
                           std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(mu_);
            busy_ = false;
        }
    }
}

void FileExpertSource::begin_layer(int64_t layer, const int32_t* ids, int64_t k) {
    (void) ids;
    (void) k;
    if (!staged()) return;
    bool new_window = false;
    {
        std::lock_guard<std::mutex> lk(stage_mu_);
        if (layer != last_layer_) {
            ++epoch_;
            new_window = layer < last_layer_;
            last_layer_ = layer;
            io_epoch_.store(epoch_, std::memory_order_relaxed);   // 149-prefetch: the workers' deadline clock
        }
    }
    if (new_window && io_sched_) io_window_tick();
    if (new_window && gate_on_ && tl_decode) { gate_capacity(); gate_tick(); }
}

bool FileExpertSource::transient(int64_t layer, int64_t expert) const {
    // the io-prefetch staging of the experts.bin tier is the decode's: the prompt path keeps reading the mapping
    // (its own read-ahead hints), so this answers as it did without io prefetch
    if ((role_ptr_.empty() && direct_.empty()) || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return false;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    if (complement_ready_ && resident_blob(index) != nullptr)
        return false;
    return override_.empty() || override_[index] == nullptr;
}

double FileExpertSource::cached_share(int64_t samples) const {
#if defined(__linux__)
    if (role_ptr_.empty() || unmapped_ || samples <= 0 || n_layers_ <= 0 || n_expert_ <= 0) return -1.0;
    // an even spread over the (layer, expert) grid, every page of each of the expert's three slices through mincore
    const int64_t total = (int64_t) n_layers_ * (int64_t) n_expert_;
    const int64_t n = std::min<int64_t>(samples, total);
    uint64_t pages = 0, resident = 0;
    std::vector<unsigned char> vec;
    for (int64_t s = 0; s < n; ++s) {
        const int64_t idx = std::min<int64_t>(total - 1, (s * total) / n + (s * 7919) % std::max<int64_t>(1, total / n));
        const int64_t layer = idx / n_expert_, expert = idx % n_expert_;
        if (!transient(layer, expert)) continue;
        for (int r = 0; r < 3; ++r) {
            const size_t i = (size_t) (3 * layer + r);
            if (role_ptr_[i] == nullptr || role_bytes_[i] == 0) continue;
            const uint8_t* p = role_ptr_[i] + (size_t) ((uint64_t) expert * role_bytes_[i]);
            const uintptr_t pg = 4096, a = (uintptr_t) p & ~(pg - 1);
            const size_t len = (size_t) ((uintptr_t) p + role_bytes_[i] - a), np = (len + pg - 1) / pg;
            vec.assign(np, 0);
            if (mincore((void*) a, len, vec.data()) != 0) return -1.0;
            pages += np;
            for (unsigned char v : vec) resident += v & 1;
        }
    }
    return pages ? (double) resident / (double) pages : -1.0;
#else
    (void) samples;
    return -1.0;   // no cheap residency query for a file mapping here: the caller keeps its SSD assumption
#endif
}

bool FileExpertSource::copy_blob(int64_t layer, int64_t expert, uint8_t* dst) {
    if (base_ == nullptr || dst == nullptr || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_)
        return false;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    const uint64_t bytes = layer_blob_bytes_[(size_t) layer];
    if (complement_ready_) {
        const uint8_t* held =
            resident_blob(index);
        if (held == nullptr && !override_.empty()) held = override_[index];
        if (held != nullptr) {
            std::memcpy(dst, held, (size_t) bytes);
            return true;
        }
    }
    const auto t0 = std::chrono::steady_clock::now();
    if (!copy_from_files(layer, expert, dst)) return false;
    file_us_.fetch_add((uint64_t) std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count(),
                       std::memory_order_relaxed);
    file_read_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    return true;
}

// pp-opt: the prompt path streams a chunk's experts in file order, so a stager thread's run of jobs is mostly
// neighbouring blobs of experts.bin.  One read_direct batch merges them into requests of up to 32 MiB (kMerge) where
// copy_blob made one ~2 MB request each - and NTFS serves the unbuffered reads of a mapped file one at a time, so the
// request size is what sets the rate (see read_direct).  The blobs the RAM copy holds are copied as before.
bool FileExpertSource::copy_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n) {
    if ((direct_.empty() && pdirect_.empty()) || n <= 1) return ExpertSource::copy_blobs(layers, experts, dst, n);
    thread_local std::vector<Fill> fills;
    fills.clear();
    uint64_t bytes = 0;
    for (size_t i = 0; i < n; ++i) {
        const int64_t l = layers[i], e = experts[i];
        if (base_ == nullptr || dst[i] == nullptr || l < 0 || e < 0 || l >= n_layers_ || e >= n_expert_) return false;
        const size_t index = (size_t) l * (size_t) n_expert_ + (size_t) e;
        if (complement_ready_ &&
            (resident_blob(index) != nullptr || (!override_.empty() && override_[index] != nullptr))) {
            if (!copy_blob(l, e, dst[i])) return false;
            continue;
        }
        fills.push_back({0, l, e, dst[i]});
        bytes += layer_blob_bytes_[(size_t) l];
    }
    if (fills.empty()) return true;
    const auto t0 = std::chrono::steady_clock::now();
    if (!read_direct(fills.data(), fills.size()))
        for (const Fill& f : fills)   // a failed batch: each blob on its own (copy_from_files falls back to the mapping)
            if (!copy_from_files(f.layer, f.e, f.dst)) return false;
    file_us_.fetch_add((uint64_t) std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count(),
                       std::memory_order_relaxed);
    file_read_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    return true;
}

const uint8_t* FileExpertSource::mapped_blob(int64_t layer, int64_t expert) const {
    if (!role_ptr_.empty()) return nullptr;   // the GGUF in place: no contiguous blob in any file
    if (unmapped_) return nullptr;            // pp-opt: drop_mapping closed the view; read_direct reads them
    if (base_ == nullptr || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return nullptr;
    const size_t i = (size_t) layer;
    if (i >= layer_offsets_.size() || i >= layer_blob_bytes_.size()) return nullptr;
    const uint64_t blob_bytes = layer_blob_bytes_[i];
    if (blob_bytes == 0 || (uint64_t) expert > std::numeric_limits<uint64_t>::max() / blob_bytes) return nullptr;
    const uint64_t expert_offset = (uint64_t) expert * blob_bytes;
    const uint64_t layer_offset = layer_offsets_[i];
    if (layer_offset > mapped_bytes_ || expert_offset > mapped_bytes_ - layer_offset) return nullptr;
    const uint64_t offset = layer_offset + expert_offset;
    if (blob_bytes > mapped_bytes_ - offset) return nullptr;
    return base_ + (size_t) offset;
}

uint64_t FileExpertSource::release(int64_t layer, int64_t expert) {
#if defined(_WIN32)
    static const bool enabled = [] {
        const char* v = std::getenv("STRATA_FILE_RELEASE");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    if (!enabled || base_ == nullptr || unmapped_ || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_)
        return 0;
    static const uint64_t page = [] {
        SYSTEM_INFO info{};
        GetSystemInfo(&info);
        return (uint64_t) info.dwPageSize;
    }();
    // These read-only file views are not CUDA-registered. blob() may instead return pinned RAM or an exchange
    // buffer, so resolve the file spans directly. A shared boundary page stays for the neighboring expert.
    auto trim = [&](const uint8_t* p, uint64_t bytes) -> uint64_t {
        const uint64_t misalignment = (uintptr_t) p % page;
        const uint64_t skip = misalignment == 0 ? 0 : page - misalignment;
        if (bytes <= skip) return 0;
        const uint64_t whole = (bytes - skip) / page * page;
        if (whole == 0) return 0;
        SetLastError(ERROR_SUCCESS);
        if (!VirtualUnlock((LPVOID) (p + (size_t) skip), (SIZE_T) whole)) {
            const DWORD error = GetLastError();
            // VirtualUnlock trims an unlocked range and reports FALSE / ERROR_NOT_LOCKED for that success.
            if (error != ERROR_NOT_LOCKED) {
                std::fprintf(stderr, "FileExpertSource: cannot trim mapped pages for layer %lld expert %lld "
                                     "(%llu bytes, Windows error %lu)\n",
                             (long long) layer, (long long) expert, (unsigned long long) whole,
                             (unsigned long) error);
                std::fflush(stderr);
                return 0;
            }
        }
        return whole;   // bytes advised, not a measurement of physical RAM reclaimed
    };
    if (role_ptr_.empty()) {
        const uint8_t* p = mapped_blob(layer, expert);
        return p == nullptr ? 0 : trim(p, layer_blob_bytes_[(size_t) layer]);
    }
    uint64_t bytes = 0;
    for (int r = 0; r < 3; ++r) {
        const size_t i = (size_t) (3 * layer + r);
        bytes += trim(role_ptr_[i] + (size_t) ((uint64_t) expert * role_bytes_[i]), role_bytes_[i]);
    }
    return bytes;
#else
    (void) layer; (void) expert;
    return 0;
#endif
}

bool FileExpertSource::pin_cache_complement(
    const ExpertCache& cache, std::string& err, bool pin,
    const std::vector<std::pair<int32_t, int32_t>>& additional_gpu_pairs, int64_t lend_from_slot,
    uint64_t headroom_bytes, uint64_t budget_bytes, const std::vector<std::pair<int32_t, int32_t>>* rank) {
    err.clear();
    const auto pin_t0 = std::chrono::steady_clock::now();
    if (base_ == nullptr) { err = "FileExpertSource: open the mapped experts before pinning a complement"; return false; }
    if (complement_ready_) { err = "FileExpertSource: the cache complement is already pinned"; return false; }
    if (!cache.valid()) { err = "FileExpertSource: the GPU expert cache is not open"; return false; }
    if (cache.fills() != cache.resident()) {
        err = "FileExpertSource: the GPU expert cache is not fully filled";
        return false;
    }
    const cudaError_t sync = cudaDeviceSynchronize();
    if (sync != cudaSuccess) {
        err = std::string("FileExpertSource: GPU expert cache is not ready: ") + cudaGetErrorString(sync);
        (void) cudaGetLastError();
        return false;
    }

    // The GPU cache's experts, and the bytes each slot's expert takes here (for the lend region below).
    const int64_t n_slots = cache.slots();
    std::vector<std::pair<int32_t, int32_t>> primary_gpu_pairs;
    std::vector<int32_t> pair_slot;
    std::vector<uint64_t> slot_bytes((size_t) std::max<int64_t>(n_slots, 0), 0);
    primary_gpu_pairs.reserve((size_t) cache.resident());
    pair_slot.reserve((size_t) cache.resident());
    for (int64_t layer = 0; layer < n_layers_; ++layer) {
        for (int64_t expert = 0; expert < n_expert_; ++expert) {
            const int32_t slot = cache.slot_of(layer, expert);
            if (slot == kNotResident) continue;
            primary_gpu_pairs.emplace_back((int32_t) layer, (int32_t) expert);
            pair_slot.push_back(slot);
            if (slot >= 0 && slot < n_slots) slot_bytes[(size_t) slot] = layer_blob_bytes_[(size_t) layer];
        }
    }
    std::vector<uint64_t> offsets;
    uint64_t bytes = 0;
    if (!detail::make_cache_complement_plan(n_layers_, n_expert_, layer_blob_bytes_, primary_gpu_pairs,
                                            additional_gpu_pairs, offsets, bytes, err)) return false;
#if defined(_WIN32)
    // #467: the GPU cache's pre-fill touched its experts through the mapping (~19 GiB on a 24 GB card), and Windows
    // counts those file pages in this process's working set, not as available: a 32 GB PC read 0.44 GiB here
    // (20.7 GiB before the start).  Trimmed, they move to the standby list (still cached, counted as available).
    // Resident mode only: nothing else calls this function.  Locked/pinned pages stay; the rest fault back softly.
    {
        uint64_t before = 0, after = 0;
        const bool read_before = available_memory_bytes(before);
        (void) SetProcessWorkingSetSize(GetCurrentProcess(), (SIZE_T) -1, (SIZE_T) -1);
        if (read_before && available_memory_bytes(after))
            std::fprintf(stderr, "FileExpertSource: available RAM %.2f GiB, %.2f GiB after the mapped experts left the "
                                 "process working set (#467)\n",
                         (double) before / 1073741824.0, (double) after / 1073741824.0);
    }
#endif
    const bool what_fits = budget_bytes == kResidentWhatFits;   // #467: the soft mode's second try
    int64_t budget_lent = 0, budget_lendable = 0;   // the lend region's slots kept by a budget (below), and its size
    uint64_t budget_physical = 0;   // #403: reuse the memory snapshot a budget was sized from
    uint64_t budget_commit = std::numeric_limits<uint64_t>::max();
    if (budget_bytes > 0) {
        // CS-T: a RAM budget.  The complement's experts in `rank` order (the expert profile, hottest first) while
        // they fit, the rest left on the mapped files; Windows commit capacity also limits the allocation (#730).
        uint64_t physical = 0;
        if (!available_memory_bytes(physical, &budget_commit)) {
            err = "FileExpertSource: cannot determine available memory for --resident-budget-gib";
            return false;
        }
        budget_physical = physical;
        const uint64_t available = std::min(physical, budget_commit);
        const char* limit = budget_commit < physical ? "commit" : "RAM";
        const uint64_t clamped = detail::clamp_resident_budget(budget_bytes, physical, budget_commit, headroom_bytes);
        if (budget_bytes > clamped) {
            // #403: 256 MiB under the room, so the engine's own allocations after this reading still leave the
            // headroom (a budget clamped to exactly the room failed the safety check below on a reading a few MB
            // lower).  An unclamped budget is unchanged.
            if (what_fits)
                std::fprintf(stderr, "FileExpertSource: room for the complement: %.2f GiB (%.2f GiB available %s "
                                     "minus %.0f GiB headroom and a 0.25 GiB margin)\n",
                             (double) clamped / 1073741824.0, (double) available / 1073741824.0, limit,
                             (double) headroom_bytes / 1073741824.0);
            else
                std::fprintf(stderr, "FileExpertSource: --resident-budget-gib %.2f exceeds available %s "
                                     "(%.2f GiB minus %.0f GiB headroom and a 0.25 GiB margin): %.2f GiB\n",
                             (double) budget_bytes / 1073741824.0, limit, (double) available / 1073741824.0,
                             (double) headroom_bytes / 1073741824.0, (double) clamped / 1073741824.0);
            budget_bytes = clamped;
        }
        std::vector<uint64_t> ranked(offsets.size(), kNoComplement);
        uint64_t at = 0;
        int64_t held = 0;
        if (rank != nullptr)
            for (const auto& pr : *rank) {
                if (pr.first < 0 || pr.second < 0 || pr.first >= n_layers_ || pr.second >= n_expert_) continue;
                const size_t i = (size_t) pr.first * (size_t) n_expert_ + (size_t) pr.second;
                if (offsets[i] == kNoComplement || ranked[i] != kNoComplement) continue;   // on a GPU, or twice
                const uint64_t b = layer_blob_bytes_[(size_t) pr.first];
                if (b > budget_bytes - at) continue;
                ranked[i] = at;
                at += b;
                ++held;
            }
        if (what_fits && held == 0) {   // #467: nothing to keep - the caller's plain mmap fallback, not an empty copy
            err = "FileExpertSource: available memory cannot hold any expert of the complement";
            return false;
        }
        std::fprintf(stderr, "FileExpertSource: RAM budget %.2f GiB: %lld of the %.2f GiB of experts the GPU cache does "
                             "not hold, by profile rank; the rest are read from the files\n",
                     (double) budget_bytes / 1073741824.0, (long long) held, (double) bytes / 1073741824.0);
        // The prompt path's lend region under a budget.  Until now a budget turned the loan off (`lend_from_slot = -1`
        // below), so every lent slot's expert was read from the pack during every prompt chunk and again when the slot
        // was refilled after the prompt - even with room left in the budget.  Measured on an RX 6900 XT 16 GB with
        // --resident-budget-gib 71 and UD-Q4_K_XL: the copy kept all 63.47 GiB of the experts outside the cache, 7.5
        // GiB of the budget stayed unused, and every agent turn read 0.5-2.6 GB of the 1,231 lent slots' experts (3.6
        // GiB) from the GGUF (a fresh 17K-token prompt 24 GB).  As the unbudgeted path does, take the lend region's
        // experts from the last slot down (the order the borrowing takes them) while they fit in what the budget has
        // left; the ones past it keep the file fallback, and `rank` keeps its precedence (the hottest experts first).
        if (lend_from_slot >= 0 && lend_from_slot < n_slots && additional_gpu_pairs.empty()) {
            budget_lendable = n_slots - lend_from_slot;
            std::vector<size_t> order;   // the pairs in the lend region, highest slot first
            for (size_t i = 0; i < primary_gpu_pairs.size(); ++i)
                if (pair_slot[i] >= lend_from_slot && pair_slot[i] < n_slots) order.push_back(i);
            std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return pair_slot[a] > pair_slot[b]; });
            const uint64_t at0 = at;
            for (size_t i : order) {
                const auto& pr = primary_gpu_pairs[i];
                const size_t idx = (size_t) pr.first * (size_t) n_expert_ + (size_t) pr.second;
                const uint64_t b = layer_blob_bytes_[(size_t) pr.first];
                if (b > budget_bytes - at) break;   // a contiguous tail of the region: a short prompt lends only the last few
                if (ranked[idx] != kNoComplement) continue;
                ranked[idx] = at;
                at += b;
                ++budget_lent;
            }
            std::fprintf(stderr, "FileExpertSource: the prompt path's lend region: %lld of its %lld slots keep their experts "
                                 "in RAM too (%.2f GiB of the budget's remaining %.2f GiB)\n",
                         (long long) budget_lent, (long long) budget_lendable, (double) (at - at0) / 1073741824.0,
                         (double) (budget_bytes - at0) / 1073741824.0);
        }
        offsets.swap(ranked);
        bytes = at;
        lend_from_slot = -1;
    }

    const bool lend = lend_from_slot >= 0 && lend_from_slot < n_slots && additional_gpu_pairs.empty();
    uint64_t budget = std::numeric_limits<uint64_t>::max();
    if (bytes > 0 || lend) {
        // #403: with a budget, the reading it was sized from - a second reading a few MB lower (the engine's own
        // allocations, the file cache) failed a budget the first one had clamped.  (A budget turns `lend` off.)
        uint64_t physical = budget_physical;
        uint64_t commit = budget_commit;
        if (physical == 0 && !available_memory_bytes(physical, &commit)) {
            err = "FileExpertSource: cannot determine available memory for the resident-memory safety check";
            return false;
        }
        const uint64_t available = std::min(physical, commit);
        budget = available > headroom_bytes ? available - headroom_bytes : 0;
        if (bytes > budget) {
            char message[320];
            std::snprintf(message, sizeof message,
                          "FileExpertSource: resident complement %.2f GiB exceeds available %s (%.2f GiB) minus the "
                          "%.0f GiB safety headroom",
                          (double) bytes / 1073741824.0, commit < physical ? "commit" : "RAM",
                          (double) available / 1073741824.0, (double) headroom_bytes / 1073741824.0);
            err = message;
#if defined(__linux__)
            if (std::ifstream mi("/proc/meminfo"); mi) {
                std::stringstream mt;
                mt << mi.rdbuf();
                err += detail::page_cache_hint(mt.str());
            }
#endif
            return false;
        }
    }
    // The prompt path's lend region: its slots' experts are streamed from here during a prompt and copied back into
    // their slots after it, so the ones that fit are kept here too (from the last slot down: a short prompt lends
    // only the last few).  The rest keep the mapped-file fallback.
    int64_t keep_from = n_slots;
    if (lend) {
        keep_from = detail::choose_resident_keep_from(slot_bytes, bytes, budget, lend_from_slot);
        if (keep_from < 0) keep_from = n_slots;
        if (keep_from < n_slots) {
            std::vector<std::pair<int32_t, int32_t>> core;
            core.reserve(primary_gpu_pairs.size());
            for (size_t i = 0; i < primary_gpu_pairs.size(); ++i)
                if (pair_slot[i] < keep_from) core.push_back(primary_gpu_pairs[i]);
            if (!detail::make_cache_complement_plan(n_layers_, n_expert_, layer_blob_bytes_, core,
                                                    additional_gpu_pairs, offsets, bytes, err)) return false;
        }
    }

    // A layer split: generate hands the copy every stage's lend region (`stage_lend_regions`, set before the
    // call).  The prompt path borrows the tail slots of EVERY part's cache, and with `additional_gpu_pairs`
    // non-empty the single lend region above is off - so without this the borrowed experts sat in neither
    // VRAM nor RAM during a prompt and came back through page faults from the mapped shards: measured
    // 2026-10-06 on an RTX 2000 Ada + 5060 Ti rig (17/48 split, IQ3_S mapped from the GGUF, --prefill auto,
    // one 73K-token prompt) 3.7 GiB read from the NVMe for the one prompt (arena mode reads 1.1 GiB), and one
    // 8192-token chunk spent 1.5 s of its 3.9 s on host staging (arena: 0.37 s).  The regions walk each
    // stage's highest slots first, as the borrowing does; the whole copy - complement and regions
    // together - stays within one budget, and experts past it keep the file fallback they had.
    int64_t stage_lent = 0;
    uint64_t stage_lent_bytes = 0;
    if (!stage_lend_regions_.empty()) {
        if (budget == std::numeric_limits<uint64_t>::max()) {   // #403: the one reading, taken now
            uint64_t physical = budget_physical, commit = budget_commit;
            if (physical == 0 && !available_memory_bytes(physical, &commit)) {
                err = "FileExpertSource: cannot determine available memory for the stage lend regions";
                return false;
            }
            const uint64_t available = std::min(physical, commit);
            budget = available > headroom_bytes ? available - headroom_bytes : 0;
        }
        uint64_t cap = budget;   // an explicit --resident-budget-gib caps the whole copy, regions included
        if (budget_bytes > 0 && budget_bytes != kResidentWhatFits && budget_bytes < cap) cap = budget_bytes;
        const uint64_t before = bytes;
        stage_lent = detail::append_stage_lend_regions(n_layers_, n_expert_, layer_blob_bytes_, offsets, bytes, cap,
                                                       stage_lend_regions_, err);
        if (stage_lent < 0) return false;
        stage_lent_bytes = bytes - before;
    }

    void* arena = nullptr;
    const uint8_t* host = nullptr;
    const uint8_t* device = nullptr;
    bool pinned_ok = false;
    bool paced = false;         ///< #1250: the pages are taken in steps and page-locked afterwards
    uint64_t locked = 0;
    uint64_t partial_pin = 0;   ///< CS-T: a registered prefix of a locked arena
    uint64_t lock_off = 0;      ///< where the working-set lock starts (after the registered prefix)
    std::string note;
#if defined(__linux__)
    NumaComplementPlacement numa;   ///< STRATA_NUMA_SPLIT step B; inert unless the live pool split on real nodes
#endif
    auto release = [&]() {
        if (arena == nullptr) return;
        if (pinned_ok) (void) cudaFreeHost(arena);
        else {
            if (partial_pin > 0) (void) cudaHostUnregister(arena);
            if (locked > 0) strata::platform::unlock_resident((uint8_t*) arena + lock_off, locked);
            std::free(arena);
        }
        arena = nullptr;
    };
#if defined(__linux__)
    // #1250: the cached pages of the model files this source maps (the engine read them itself), given back
    auto drop_file_cache = [&]() {
        auto drop = [](const uint8_t* at, uint64_t len, int fd) {
            if (at != nullptr && len > 0) (void) madvise((void*) at, (size_t) len, MADV_DONTNEED);
            if (fd >= 0) (void) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        };
        if (maps_.empty()) drop(base_, mapped_bytes_, fd_);
        else for (const Map& m : maps_) drop(m.closed ? nullptr : m.base, m.bytes, m.fd);
        // the files read at start outside this source (set_extra_drop_files): their pages are already copied to VRAM
        // or RAM, and they sat on whichever node read them (~2.2 GiB on node 0 measured on the 3090 box)
        // CBA: whole files by path; a file read at start but not listed stays cached (the per-node free line shows
        // it), a registry of the files the loaders open if that ever bites
        for (const std::string& p : extra_drop_files_) {
            const int fd = ::open(p.c_str(), O_RDONLY | O_CLOEXEC);
            if (fd >= 0) {
                (void) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
                ::close(fd);
            }
        }
    };
#endif
    if (bytes > 0) {
        std::fprintf(stderr, "FileExpertSource: allocating %.2f GiB %s cache complement\n",
                     (double) bytes / 1073741824.0, pin ? "page-locked" : "pageable resident");
        log_complement_memory("before complement allocation", bytes);
        std::fflush(stderr);
        // #1250: the driver takes page-locked pages in one go and cannot wait for the kernel to reclaim file cache: with
        // the cache of the model files the engine read itself in the way (MemFree low, MemAvailable high) a host was
        // OOM-killed here.  First give those pages back (the engine's own mappings); when what is really free still
        // does not cover the complement, register it in steps, each only while the RAM left stays above the reserve.
        // STRATA_PIN_GUARD=0 pages-locks in one go as before, =2 always registers in steps (a test); STRATA_PIN_RESERVE_GIB
        // is the RAM that must stay above every step (default: the resident headroom, at least 2 GiB)
        // STRATA_NUMA_SPLIT placement forces the drop and the paced path whatever the guard (STRATA_PIN_GUARD=0 included)
        const uint64_t pace_reserve = [&] {
            const char* v = std::getenv("STRATA_PIN_RESERVE_GIB");
            return v != nullptr && std::atof(v) > 0 ? (uint64_t) (std::atof(v) * 1073741824.0)
                                                     : std::max<uint64_t>(headroom_bytes, 2ull << 30);
        }();
#if defined(__linux__)
        if (pin) {
            static const int guard = [] {
                const char* v = std::getenv("STRATA_PIN_GUARD");
                return v == nullptr ? 1 : std::atoi(v);
            }();
            uint64_t free_now = 0;
            numa.init(offsets, n_expert_, layer_blob_bytes_, role_bytes_);   // STRATA_NUMA_SPLIT: placed by the paced loop
            if (numa.place) {
                // the engine's own mapped model pages sit on whichever node read them, and a toucher on a full node
                // gets its pages from the other one (no local reclaim): give them back first, as the guard=1 path does
                drop_file_cache();
                numa.free_after = NumaComplementPlacement::node_free();
                std::fprintf(stderr, "FileExpertSource: NUMA placement: the model files' cached pages were released; "
                                     "free after the drop: %s\n",
                             NumaComplementPlacement::free_text(numa.free_after).c_str());
                // CBA: half the pace reserve as each node's share of the headroom; a per-node reserve if one node
                // carries the host's other allocations
                numa.check_room(pace_reserve / 2);
                if (guard == 0)
                    std::fprintf(stderr, "FileExpertSource: STRATA_PIN_GUARD=0 overridden: NUMA placement needs the "
                                         "paced path\n");
            }
            if (guard == 2 || numa.place) {
                paced = true;
                std::fprintf(stderr, "FileExpertSource: %s: page-locking the complement in steps\n",
                             guard == 2 ? "STRATA_PIN_GUARD=2" : "STRATA_NUMA_SPLIT");
            } else if (guard != 0 && free_memory_bytes(free_now) &&
                       detail::pin_depends_on_reclaim(free_now, bytes, pace_reserve)) {
                drop_file_cache();
                const uint64_t before = free_now;
                if (free_memory_bytes(free_now))
                    std::fprintf(stderr, "FileExpertSource: %.2f GiB really free for a %.2f GiB page-locked complement; "
                                         "the model files' cached pages were released: %.2f GiB free\n",
                                 (double) before / 1073741824.0, (double) bytes / 1073741824.0,
                                 (double) free_now / 1073741824.0);
                paced = detail::pin_depends_on_reclaim(free_now, bytes, pace_reserve);
                if (paced)
                    std::fprintf(stderr, "FileExpertSource: the rest comes from reclaimed cache of other files: "
                                         "page-locking it in steps, as far as the RAM allows (#1250)\n");
            }
        } else if (strata::kernels::cpu::active_numa_split().on && !strata::kernels::cpu::active_numa_split().test) {
            std::fprintf(stderr, "FileExpertSource: WARNING: NUMA placement FAILED: the complement is not page-locked, "
                                 "nothing placed\n");
        }
#endif
        if (pin && !paced) {
            const cudaError_t allocated = cudaHostAlloc(&arena, (size_t) bytes,
                                                         cudaHostAllocMapped | cudaHostAllocPortable);
            if (allocated == cudaSuccess) {
                void* alias = nullptr;
                const cudaError_t aliased = cudaHostGetDevicePointer(&alias, arena, 0);
                if (aliased == cudaSuccess && alias != nullptr) {
                    device = (const uint8_t*) alias;
                    pinned_ok = true;
                    note = "page-locked and mapped";
                } else {
                    note = std::string("no device alias (") + cudaGetErrorString(aliased) + ")";
                    (void) cudaGetLastError();
                    (void) cudaFreeHost(arena);
                    arena = nullptr;
                }
            } else {
                // Refused (the driver's page-locked limit): the same bytes in ordinary memory, locked in the working
                // set instead, as the arena does - resident either way, only copied by the CPU instead of by DMA.
                note = std::string("page-locking refused (") + cudaGetErrorString(allocated) + ")";
                (void) cudaGetLastError();
                arena = nullptr;
            }
        }
        if (arena == nullptr) {
            if (!note.empty()) {
                std::fprintf(stderr, "FileExpertSource: %s; trying pageable memory\n", note.c_str());
                log_complement_memory("before pageable allocation", bytes);
                std::fflush(stderr);
            }
            errno = 0;
            if (paced) {   // page-aligned: the steps are registered on page boundaries
#if !defined(_WIN32)
                void* aligned = nullptr;
                const uint64_t page = 4096;
                if (posix_memalign(&aligned, (size_t) page, (size_t) ((bytes + page - 1) / page * page)) == 0) arena = aligned;
#endif
#if defined(__linux__)
                // STRATA_NUMA_SPLIT: a 2 MiB huge page would put several slices on its toucher's node, so the arena
                // takes 4 KiB pages (one call, no VMA split)
                if (arena != nullptr && numa.place) {
                    if (madvise(arena, (size_t) ((bytes + page - 1) / page * page), MADV_NOHUGEPAGE) != 0)
                        std::fprintf(stderr, "FileExpertSource: WARNING: madvise(MADV_NOHUGEPAGE) on the complement "
                                             "failed (errno %d)\n", errno);
                }
#endif
            } else {
                arena = std::malloc((size_t) bytes);
            }
            if (arena == nullptr) {
                const int allocation_errno = errno;
                err = "FileExpertSource: pageable resident complement allocation failed (malloc errno " +
                      std::to_string(allocation_errno) + ": " + std::strerror(allocation_errno) + ")";
                if (!note.empty()) err += "; " + note;
                log_complement_memory("after pageable allocation failure", bytes);
                return false;
            }
            if (pin) {
                // CS-T, a RAM budget: its bytes are in profile order, hottest first, so the driver is asked to
                // register the largest prefix it takes (from the cap down in 2 GiB steps).  Those experts can be
                // read by the GPU over PCIe (--pcie-frac) and copied by DMA; only the rest is locked in the working
                // set (the registered prefix is page-locked by the driver already - locking it twice made the next
                // device allocation fail).
                // opt-in (STRATA_PARTIAL_PIN=1): on the RTX 5070 PC the GPU's PCIe share of the misses measured no
                // faster than the CPU computing them (7.30 / 7.44 tok/s with 24 / 16 GiB registered against 7.05-7.74
                // unpinned at a 40 GiB budget), and registering adds startup time and driver memory pressure
                static const bool partial_on = [] {
                    const char* v = std::getenv("STRATA_PARTIAL_PIN");
                    return v != nullptr && std::atoi(v) != 0;
                }();
                // at most STRATA_PARTIAL_PIN_GIB (default 24): registering 30 GiB of a 40 GiB arena left the driver
                // unable to page-lock the prompt path's small buffers afterwards (RTX 5070, WDDM)
                static const uint64_t pin_cap = [] {
                    const char* v = std::getenv("STRATA_PARTIAL_PIN_GIB");
                    return (uint64_t) ((v != nullptr && std::atof(v) > 0 ? std::atof(v) : 24.0) * 1073741824.0);
                }();
                if (paced) {
                    // #1250: the pages are taken in 1 GiB steps by ordinary page faults (which reclaim file cache as
                    // they need to), each only while the RAM available (the cgroup's room included) stays above the
                    // reserve; then the pages that are there are page-locked in ONE registration (nothing is
                    // allocated then, the driver only pins them), so a blob never straddles two registrations.  The
                    // hottest experts come first in the arena: they are the pinned ones; the rest is locked pageable
                    // memory, as when the driver refuses the page-locking.
                    constexpr uint64_t kStep = 1ull << 30, kPage = 4096;
                    uint64_t got = 0;
                    const char* stop = nullptr;
                    while (got < bytes) {
                        const uint64_t step = std::min<uint64_t>(kStep, bytes - got);
                        uint64_t avail_now = 0;
                        if (!available_memory_bytes(avail_now) || !detail::pin_step_fits(avail_now, step, pace_reserve)) {
                            stop = "the RAM left would fall under the reserve";
                            break;
                        }
#if defined(__linux__)
                        if (numa.place) numa.touch_step((uint8_t*) arena, got, got + step);   // slices on their nodes
                        else
#endif
                            for (uint64_t at = got; at < got + step; at += kPage) ((volatile uint8_t*) arena)[at] = 0;
                        got += step;
                    }
                    // cut at an expert boundary: a blob that runs past the registered range would be refused by a
                    // cudaMemcpyAsync as not page-locked
                    uint64_t w = got;
                    for (size_t i = 0; i < offsets.size() && w < bytes; ++i) {
                        if (offsets[i] == kNoComplement) continue;
                        const uint64_t b = layer_blob_bytes_[i / (size_t) n_expert_];
                        if (offsets[i] < got && offsets[i] + b > got) { w = offsets[i]; break; }
                    }
                    if (w > 0) {
                        if (cudaHostRegister(arena, (size_t) w, cudaHostRegisterMapped | cudaHostRegisterPortable) ==
                            cudaSuccess) {
                            void* alias = nullptr;
                            if (cudaHostGetDevicePointer(&alias, arena, 0) == cudaSuccess && alias != nullptr) {
                                device = (const uint8_t*) alias;
                                partial_pin = w;
                            } else {
                                (void) cudaGetLastError();
                                (void) cudaHostUnregister(arena);
                                if (stop == nullptr) stop = "no device alias";
                            }
                        } else {
                            (void) cudaGetLastError();
                            if (stop == nullptr) stop = "the driver refused to page-lock it";
                        }
                    }
                    lock_off = partial_pin;
                    char msg[240];
                    if (stop != nullptr)
                        std::snprintf(msg, sizeof msg, "WARNING: only %.2f of %.2f GiB could be page-locked (%s); the rest is "
                                      "resident but pageable, so its copies to the GPU go through the CPU (#1250)",
                                      (double) partial_pin / 1073741824.0, (double) bytes / 1073741824.0, stop);
                    else
                        std::snprintf(msg, sizeof msg, "%.2f GiB page-locked after taking the pages in steps",
                                      (double) partial_pin / 1073741824.0);
                    note += std::string(note.empty() ? "" : "; ") + msg;
                    if (stop != nullptr) std::fprintf(stderr, "FileExpertSource: %s\n", msg);
#if defined(__linux__)
                    // judged on the final state: registered pages no longer move; before the copy fills them
                    if (numa.place) numa.report((const uint8_t*) arena, got, partial_pin, bytes, stop);
#endif
                } else if (budget_bytes > 0 && partial_on) {
                    const uint64_t step = 2ull << 30;
                    for (uint64_t want = std::min(bytes, pin_cap); want >= step; want = want > step ? want - step : 0) {
                        // cut at an expert boundary: a blob that started inside the registered range and ran past it
                        // would be taken as page-locked by a cudaMemcpyAsync and refused ("adaptive refill failed")
                        uint64_t w = want;
                        for (size_t i = 0; i < offsets.size(); ++i) {
                            if (offsets[i] == kNoComplement) continue;
                            const uint64_t b = layer_blob_bytes_[i / (size_t) n_expert_];
                            if (offsets[i] < want && offsets[i] + b > want) { w = offsets[i]; break; }
                        }
                        if (w == 0) break;
                        if (cudaHostRegister(arena, (size_t) w, cudaHostRegisterMapped | cudaHostRegisterPortable) ==
                            cudaSuccess) {
                            void* alias = nullptr;
                            if (cudaHostGetDevicePointer(&alias, arena, 0) == cudaSuccess && alias != nullptr) {
                                device = (const uint8_t*) alias;
                                partial_pin = w;
                            } else {
                                (void) cudaGetLastError();
                                (void) cudaHostUnregister(arena);
                            }
                            break;
                        }
                        (void) cudaGetLastError();
                        if (want <= step) break;
                    }
                    char msg[160];
                    std::snprintf(msg, sizeof msg, "%.2f GiB of it registered for the GPU (the hottest)",
                                  (double) partial_pin / 1073741824.0);
                    note += std::string("; ") + msg;
                }
                if (!paced) lock_off = partial_pin;
                // (paced: lock_off was set above)
                const strata::platform::LockResult lr =
                    lock_off < bytes ? strata::platform::lock_resident((uint8_t*) arena + lock_off, bytes - lock_off)
                                     : strata::platform::LockResult{true, 0, ""};
                locked = lr.locked_bytes;
                note += (note.empty() ? "" : "; ") + lr.note;
            }
        }
        host = (const uint8_t*) arena;
    }

    // Copied layer by layer on a few threads: the page faults of the mapped file are the cost, and they overlap.
#if !defined(_WIN32)
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        err = "FileExpertSource: cannot determine page size for mapped-page release";
        release();
        return false;
    }
#endif
    std::atomic<int64_t> next_layer{0}, layers_done{0};
    std::atomic<uint64_t> copied{0};
    std::atomic<bool> failed{false};
    std::mutex fail_mu;
    const auto copy_t0 = std::chrono::steady_clock::now();
    std::string fail_msg;
    auto fail = [&](const std::string& m) {
        std::lock_guard<std::mutex> lock(fail_mu);
        if (fail_msg.empty()) fail_msg = m;
        failed.store(true);
    };
    auto worker = [&]() {
        for (;;) {
            const int64_t layer = next_layer.fetch_add(1);
            if (layer >= n_layers_ || failed.load()) return;
            const uint64_t blob_bytes = layer_blob_bytes_[(size_t) layer];
            std::vector<Fill> batch;   // #286, unbuffered: 32 blobs' reads in flight at once, merged where near
            auto flush = [&]() {
                if (batch.empty()) return true;
                bool ok = read_direct(batch.data(), batch.size());
                for (size_t i = 0; !ok && i < batch.size(); ++i)
                    if (!copy_from_files(batch[i].layer, batch[i].e, batch[i].dst)) return false;
                copied.fetch_add(blob_bytes * (uint64_t) batch.size());
                batch.clear();
                return true;
            };
            if (direct_.empty()) {   // mapped reads: ask for the layer's blobs before copying them
                std::vector<std::pair<int32_t, int32_t>> ahead;
                for (int64_t expert = 0; expert < n_expert_; ++expert)
                    if (offsets[(size_t) layer * (size_t) n_expert_ + (size_t) expert] != kNoComplement)
                        ahead.emplace_back((int32_t) layer, (int32_t) expert);
                (void) advise_pairs(ahead.data(), (int64_t) ahead.size());
            }
            for (int64_t expert = 0; expert < n_expert_; ++expert) {
                const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
                const uint64_t offset = offsets[index];
                if (offset == kNoComplement) continue;
                if (offset > bytes || blob_bytes > bytes - offset) {
                    fail("FileExpertSource: invalid blob bounds while building the cache complement");
                    return;
                }
                uint8_t* dst = (uint8_t*) host + (size_t) offset;
                if (!direct_.empty()) {
                    batch.push_back({0, layer, expert, dst});
                    if (batch.size() == 32 && !flush()) {
                        fail("FileExpertSource: an expert could not be read while building the cache complement");
                        return;
                    }
                    continue;
                }
                if (!copy_from_files(layer, expert, dst)) {
                    fail("FileExpertSource: invalid blob bounds while building the cache complement");
                    return;
                }
                copied.fetch_add(blob_bytes);
            }
            if (!flush()) {
                fail("FileExpertSource: an expert could not be read while building the cache complement");
                return;
            }
#if !defined(_WIN32)
            if (role_ptr_.empty()) {   // experts.bin; the GGUF in place leaves its pages to the OS
            const uint64_t layer_offset = layer_offsets_[(size_t) layer];
            const uint64_t layer_bytes = blob_bytes * (uint64_t) n_expert_;
            const uint64_t layer_end = layer_offset + layer_bytes;
            const uint64_t page = (uint64_t) page_size;
            const uint64_t advice_start = layer_offset - layer_offset % page;
            const uint64_t end_remainder = layer_end % page;
            const uint64_t extra = end_remainder == 0 ? 0 : page - end_remainder;
            const uint64_t advice_end = extra > mapped_bytes_ - layer_end ? mapped_bytes_ : layer_end + extra;
            if (advice_end > advice_start &&
                madvise((void*) (base_ + (size_t) advice_start), (size_t) (advice_end - advice_start), MADV_DONTNEED) != 0) {
                fail("FileExpertSource: madvise could not release mapped expert layer " + std::to_string(layer));
                return;
            }
#if !defined(__APPLE__)   // macOS has no posix_fadvise; the madvise above already released the mapping
            if (posix_fadvise(fd_, (off_t) layer_offset, (off_t) layer_bytes, POSIX_FADV_DONTNEED) != 0) {
                fail("FileExpertSource: posix_fadvise could not release expert layer " + std::to_string(layer));
                return;
            }
#endif
            }
#endif
            const int64_t done = layers_done.fetch_add(1) + 1;
            if (done % 8 == 0 || done == n_layers_)
                std::fprintf(stderr, "FileExpertSource: copied cache complement through layer %lld/%lld (%.2f GiB, %.0f s)\n",
                             (long long) done, (long long) n_layers_, (double) copied.load() / 1073741824.0,
                             std::chrono::duration<double>(std::chrono::steady_clock::now() - copy_t0).count());
        }
    };
    {
        const int threads = (int) std::max<int64_t>(1, std::min<int64_t>(6, n_layers_));
        std::vector<std::thread> pool;
        for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
        worker();
        for (auto& t : pool) t.join();
    }
    std::fflush(stderr);
    if (failed.load()) {
        err = fail_msg;
        release();
        return false;
    }
#if defined(_WIN32)
    // The mapped pages this process touched (the GPU cache's fill and this copy) leave its working set for the
    // standby list: VirtualUnlock on pages that are not locked does exactly that (it then reports ERROR_NOT_LOCKED).
    // Not with the view closed (begin_startup_unbuffered): its range may hold other memory by now.
    if (maps_.empty() && !unmapped_) (void) VirtualUnlock((LPVOID) base_, (SIZE_T) mapped_bytes_);
    for (const Map& m : maps_) if (!m.closed) (void) VirtualUnlock((LPVOID) m.base, (SIZE_T) m.bytes);
#endif
    if (std::getenv("STRATA_VERIFY_COMPLEMENT") != nullptr && host != nullptr && bytes > 0) {
        // tests: the copy's checksum (FNV-1a of each 1 GiB piece on its own thread, then of those), any reader
        constexpr uint64_t kPiece = 1ull << 30;
        const size_t pieces = (size_t) ((bytes + kPiece - 1) / kPiece);
        std::vector<uint64_t> ph(pieces, 0);
        std::atomic<size_t> next_piece{0};
        auto hash_work = [&] {
            for (size_t i; (i = next_piece.fetch_add(1)) < pieces;)
                ph[i] = fnv1a64(host + (size_t) (i * kPiece), std::min<uint64_t>(kPiece, bytes - i * kPiece));
        };
        std::vector<std::thread> hth;
        for (int t = 1; t < 8; ++t) hth.emplace_back(hash_work);
        hash_work();
        for (auto& t : hth) t.join();
        uint64_t h = fnv1a64((const uint8_t*) offsets.data(), (uint64_t) (offsets.size() * sizeof(uint64_t)));
        h = fnv1a64((const uint8_t*) ph.data(), (uint64_t) (ph.size() * sizeof(uint64_t)), h);
        std::fprintf(stderr, "FileExpertSource: cache complement checksum %016llx (%.2f GiB, its offsets included)\n",
                     (unsigned long long) h, (double) bytes / 1073741824.0);
    }

    complement_arena_ = arena;
    complement_host_ = host;
    complement_device_ = device;
    complement_bytes_ = bytes;
    complement_offsets_ = std::move(offsets);
    complement_pinned_ = (pinned_ok || partial_pin > 0) && bytes > 0;
    complement_pin_limit_ = pinned_ok ? bytes : partial_pin;
    complement_registered_ = !pinned_ok && partial_pin > 0;
    // a paced registration that covers the whole arena is as good as a cudaHostAlloc'd one
    complement_partial_ = complement_registered_ && !(paced && partial_pin >= bytes);
    complement_locked_ = locked;
    complement_lock_off_ = lock_off;
    complement_lent_slots_ = lend ? n_slots - keep_from : budget_lent;   // a budget keeps its own tail of the region
    complement_lent_slots_ += stage_lent;   // and the split stages' regions: pairs the copy kept of them
    complement_ready_ = true;
    std::fprintf(stderr, "FileExpertSource: %s cache complement ready: resident %.2f GiB, pinned %.2f GiB in %.0f s%s%s\n",
                 complement_pinned_ ? "mapped pinned" : pin ? "locked resident" : "pageable resident",
                 (double) resident_bytes() / 1073741824.0, (double) pinned_bytes() / 1073741824.0,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - pin_t0).count(),
                 note.empty() ? "" : "; ", note.c_str());
    const int64_t lendable_all = lend ? n_slots - lend_from_slot : budget_lendable;   // 0: no lend region at all
    if (lendable_all > 0)
        std::fprintf(stderr, "FileExpertSource: %lld of the prompt path's %lld lendable slots keep their experts in RAM "
                             "too%s\n", (long long) (lend ? n_slots - keep_from : budget_lent), (long long) lendable_all,
                     (lend ? n_slots - keep_from : budget_lent) < lendable_all
                         ? " (the others are read from the file when lent: not enough RAM for them)" : "");
    if (lendable_all > 0 && (lend ? n_slots - keep_from : budget_lent) < lendable_all) {
        // #1389: below full coverage the uncovered experts are read from the pack during a long prompt (a 74K prompt
        // routes through nearly all experts, so the misses repeat). The reporter measured 18% uncovered = -26% prompt
        // speed on a gfx1100; short prompts barely touch them. A warning, not a refusal.
        const long long lendable = (long long) lendable_all;
        const long long missing = lendable - (long long) (lend ? n_slots - keep_from : budget_lent);
        std::fprintf(stderr, "FileExpertSource: WARNING: %lld of %lld lendable slots (%.0f%%) will read their experts from the pack "
                             "during a long prompt; expect a slower prompt read on 50K+ token prompts (one measurement: 18%% uncovered "
                             "cost 26%%). %s and restart; the line above should read N of N.\n", missing, lendable,
                     100.0 * (double) missing / (double) std::max<long long>(1, lendable),
                     lend ? "Free RAM or lower STRATA_RESIDENT_HEADROOM_GIB (now the RAM left free at start)"
                          : "Raise --resident-budget-gib (the lend region comes after the ranked experts)");
    }
    if (!stage_lend_regions_.empty()) {
        int64_t offered = 0;
        for (const auto& region : stage_lend_regions_) offered += (int64_t) region.size();
        std::fprintf(stderr, "FileExpertSource: the split stages lend %lld slots to the prompt path: %lld keep "
                             "their experts in RAM too (%.2f GiB)%s\n",
                     (long long) offered, (long long) stage_lent, (double) stage_lent_bytes / 1073741824.0,
                     stage_lent < offered ? " (the rest are read from the file when lent: not enough RAM for them)" : "");
    }
    if (!additional_gpu_pairs.empty()) {
        std::fprintf(stderr, "FileExpertSource: %zu verified additional-GPU experts remain on the mmap fallback\n",
                     additional_gpu_pairs.size());
    }
    std::fflush(stderr);
    return true;
}

const uint8_t* FileExpertSource::resident_blob(size_t index) const {
    if (exchange_storage_.active()) return exchange_storage_.resident(index).host;
    return detail::cache_complement_blob_or_fallback(index, complement_offsets_, complement_host_, nullptr);
}

bool FileExpertSource::has_resident(int64_t layer, int64_t expert) const {
    if (!complement_ready_ || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return false;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    return resident_blob(index) != nullptr;
}

bool FileExpertSource::reserve_exchanges(int64_t n, std::string& err, int64_t bounce) {
    err.clear();
    if (n <= xstage_cap_) return true;
    if (!staged_.empty()) { err = "FileExpertSource: exchange buffers are in use"; return false; }
    if (exchange_storage_.active()) {
        err = "FileExpertSource: reserve maximum exchange capacity before enabling rotation";
        return false; // the original exchange allocation can now contain live resident experts
    }
    uint64_t blob = 0;
    for (const uint64_t b : layer_blob_bytes_) blob = std::max(blob, b);
    if (blob == 0 || n <= 0) { err = "FileExpertSource: no expert geometry for the exchange buffers"; return false; }
    if (xstage_ != nullptr) {
        if (xstage_pinned_) (void) cudaFreeHost(xstage_);
        else std::free(xstage_);
        xstage_ = nullptr;
        xstage_cap_ = 0;
    }
    if ((uint64_t)n > std::numeric_limits<size_t>::max() / blob) {
        err = "FileExpertSource: exchange allocation size overflow"; return false;
    }
    const char* rotate_env = std::getenv("STRATA_EXCHANGE_ROTATE");
    const bool requested = rotate_env && std::strcmp(rotate_env, "1") == 0;
    // layers of several block sizes (a GGUF's mixed i-quants): spares of every size, see ExchangeStorage
    bounce = std::max<int64_t>(0, std::min(bounce, n));
    const uint64_t rotate_total = detail::ExchangeStorage::exchange_bytes_needed(layer_blob_bytes_, (size_t) n, (size_t) bounce);
    const bool eligible = requested && rotate_total != 0 && rotate_total <= std::numeric_limits<size_t>::max() &&
                          complement_pinned_ && !complement_partial_ &&
                          complement_host_ && complement_device_ && complement_bytes_ > 0;
    const size_t total = eligible ? (size_t) rotate_total : (size_t) n * (size_t) blob;
    void* p = nullptr;
    if (cudaHostAlloc(&p, total, eligible ? cudaHostAllocMapped : cudaHostAllocDefault) == cudaSuccess && p != nullptr) {
        xstage_pinned_ = true;
    } else {
        (void) cudaGetLastError();
        p = std::malloc(total);
        xstage_pinned_ = false;
        if (p == nullptr) { err = "FileExpertSource: cannot allocate the exchange buffers"; return false; }
    }
    xstage_ = (uint8_t*) p;
    xstage_cap_ = n;
    xstage_blob_ = blob;
    if (requested) {
        std::string reason = "requires a fully mapped/pinned RAM complement";
        void* device = nullptr;
        if (eligible && xstage_pinned_ && cudaHostGetDevicePointer(&device, p, 0) == cudaSuccess && device) {
            try {
                std::vector<uint64_t> sizes(complement_offsets_.size(), 0);
                for (size_t i = 0; i < sizes.size(); ++i) {
                    const size_t layer = n_expert_ > 0 ? i / (size_t) n_expert_ : 0;
                    sizes[i] = layer < layer_blob_bytes_.size() ? layer_blob_bytes_[layer] : 0;
                }
                if (exchange_storage_.initialize_sized(complement_offsets_, sizes, (uint8_t*)complement_host_, complement_device_,
                        complement_bytes_, xstage_, (const uint8_t*)device, (uint64_t)total, (size_t)n, (size_t)bounce, reason)) {
                    std::fprintf(stderr, "FileExpertSource: exchange buffer rotation enabled: %lld buffers in %zu block sizes "
                                         "(largest %llu bytes, %.2f GiB page-locked); no host commit memcpy\n",
                                 (long long)n, exchange_storage_.block_sizes(), (unsigned long long)blob,
                                 (double)total / 1073741824.0);
                }
            } catch (const std::bad_alloc&) { reason = "buffer metadata allocation failed"; }
        } else if (eligible) {
            (void) cudaGetLastError();
            reason = "exchange buffers could not be pinned and mapped";
        }
        if (!exchange_storage_.active())
            std::fprintf(stderr, "FileExpertSource: exchange rotation unavailable; retaining copy path: %s\n", reason.c_str());
    }
    return true;
}

uint8_t* FileExpertSource::exchange_buffer(int64_t q, int64_t layer) const {
    if (xstage_ == nullptr || q < 0 || q >= xstage_cap_) return nullptr;
    if (exchange_storage_.active())
        return layer >= 0 && layer < n_layers_ ? exchange_storage_.spare((size_t)q, (size_t)layer_blob_bytes_[(size_t)layer]).host
                                             : exchange_storage_.spare((size_t)q).host;
    return xstage_ + (size_t) q * (size_t) xstage_blob_;
}

bool FileExpertSource::stage_exchange(int64_t layer, int64_t in, int64_t out, int64_t q) {
    if (out < 0 || out >= n_expert_ || !has_resident(layer, in) || has_resident(layer, out) ||
        exchange_buffer(q, layer) == nullptr) return false;
    const size_t i_in = (size_t) layer * (size_t) n_expert_ + (size_t) in;
    const size_t i_out = (size_t) layer * (size_t) n_expert_ + (size_t) out;
    if (override_.empty()) override_.assign((size_t) blobs_, nullptr);
    if (override_[i_out] != nullptr) return false;
    for (const Exchange& x : staged_)
        if (x.in == i_in || x.q == q) return false;
    override_[i_out] = exchange_buffer(q, layer);
    staged_.push_back({i_in, i_out, q, layer_blob_bytes_[(size_t) layer]});
    return true;
}

bool FileExpertSource::soft_commit_exchanges() {
    if (staged_.empty()) return true;
    if (exchange_storage_.active() || !xstage_pinned_ || xstage_ == nullptr || !complement_pinned_ || complement_device_ == nullptr)
        return false;
    if (xstage_dev_ == nullptr) {
        void* d = nullptr;
        if (cudaHostGetDevicePointer(&d, xstage_, 0) != cudaSuccess || d == nullptr) { (void) cudaGetLastError(); return false; }
        xstage_dev_ = (const uint8_t*) d;
    }
    if (soft_in_.empty()) soft_in_.assign((size_t) blobs_, SIZE_MAX);
    for (const Exchange& x : staged_) soft_in_[x.out] = x.in;
    return true;
}

void FileExpertSource::fs_hold_staged(bool hold) {
    ForesightSwap* fs = fs_swap_.load(std::memory_order_acquire);
    if (fs == nullptr || n_expert_ <= 0) return;
    for (const Exchange& x : staged_)
        for (const size_t i : {x.in, x.out}) {
            const int64_t layer = (int64_t) (i / (size_t) n_expert_);
            const int32_t expert = (int32_t) (i % (size_t) n_expert_);
            if (hold) fs->hold(layer, expert);
            else fs->unhold(layer, expert);
        }
}

int64_t FileExpertSource::commit_exchanges() {
    int64_t n = 0;
    fs_hold_staged(true);   // #2134: both experts' bytes move (in's place takes out's bytes, out leaves its buffer)
    for (const Exchange& x : staged_)
        if (!soft_in_.empty()) soft_in_[x.out] = SIZE_MAX;
    for (const Exchange& x : staged_) {
        const uint8_t* src = override_[x.out];
        if (exchange_storage_.active()) {
            if (!exchange_storage_.commit(x.in, x.out, (size_t)x.q, src, (size_t)x.bytes)) {
                // Admitting the GPU swap after a failed RAM ownership update would lose an expert.
                std::fprintf(stderr, "FileExpertSource: invalid exchange rotation commit; refusing corrupt residency\n");
                std::abort();
            }
            ++n;
            override_[x.out] = nullptr;
            continue;
        }
        const uint64_t at = complement_offsets_[x.in];
        if (src != nullptr && at != kNoComplement && at <= complement_bytes_ && x.bytes <= complement_bytes_ - at &&
            complement_host_ != nullptr) {
            std::memcpy((uint8_t*) complement_host_ + (size_t) at, src, (size_t) x.bytes);
            if (detail::exchange_cache_complement(complement_offsets_, x.in, x.out)) ++n;
        }
        override_[x.out] = nullptr;
    }
    fs_hold_staged(false);
    staged_.clear();
    exchanges_ += n;
    return n;
}

bool FileExpertSource::replace_resident(int64_t layer, int64_t in, int64_t out) {
    if (!complement_ready_ || complement_host_ == nullptr || exchange_storage_.active() || !staged_.empty() ||
        layer < 0 || layer >= n_layers_ || in < 0 || out < 0 || in >= n_expert_ || out >= n_expert_ || in == out)
        return false;
    const size_t i_in = (size_t) layer * (size_t) n_expert_ + (size_t) in;
    const size_t i_out = (size_t) layer * (size_t) n_expert_ + (size_t) out;
    if (i_in >= complement_offsets_.size() || i_out >= complement_offsets_.size()) return false;
    if (!override_.empty() && (override_[i_in] != nullptr || override_[i_out] != nullptr)) return false;
    const uint64_t at = complement_offsets_[i_out], bytes = layer_blob_bytes_[(size_t) layer];
    if (at == kNoComplement || complement_offsets_[i_in] != kNoComplement || at > complement_bytes_ ||
        bytes > complement_bytes_ - at) return false;
    (void) at;
    // read beside the copy first: a failed read leaves `out` where it was
    if (ram_adapt_buf_.size() < (size_t) bytes) ram_adapt_buf_.resize((size_t) bytes);
    const auto t0 = std::chrono::steady_clock::now();
    if (!copy_from_files(layer, in, ram_adapt_buf_.data())) return false;
    file_us_.fetch_add((uint64_t) std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count(),
                       std::memory_order_relaxed);
    file_read_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    return install_resident(layer, in, out, ram_adapt_buf_.data());
}

bool FileExpertSource::replace_resident_staged(int64_t layer, int64_t in, int64_t out) {
    if (!complement_ready_ || complement_host_ == nullptr || exchange_storage_.active() || !staged_.empty() ||
        layer < 0 || layer >= n_layers_ || in < 0 || out < 0 || in >= n_expert_ || out >= n_expert_ || in == out)
        return false;
    const size_t i_in = (size_t) layer * (size_t) n_expert_ + (size_t) in;
    const size_t i_out = (size_t) layer * (size_t) n_expert_ + (size_t) out;
    if (i_in >= complement_offsets_.size() || i_out >= complement_offsets_.size()) return false;
    if (!override_.empty() && (override_[i_in] != nullptr || override_[i_out] != nullptr)) return false;
    const uint64_t at = complement_offsets_[i_out], bytes = layer_blob_bytes_[(size_t) layer];
    if (at == kNoComplement || complement_offsets_[i_in] != kNoComplement || at > complement_bytes_ ||
        bytes > complement_bytes_ - at || bytes > stage_blob_) return false;
    ForesightSwap* fs = fs_swap_.load(std::memory_order_acquire);
    bool moved = false;
    {
        // held across the copy: claim_stage cannot hand the buffer to another blob meanwhile
        std::lock_guard<std::mutex> lk(stage_mu_);
        const auto it = stage_of_.find((int64_t) i_in);
        if (it == stage_of_.end() || stage_busy_[it->second] || stage_key_[it->second] != (int64_t) i_in) return false;
        if (fs != nullptr) fs->hold(layer, (int32_t) out);
        std::memcpy((uint8_t*) complement_host_ + (size_t) at, stage_buf_[it->second].get(), (size_t) bytes);
        moved = detail::exchange_cache_complement(complement_offsets_, i_out, i_in);
        if (fs != nullptr) fs->unhold(layer, (int32_t) out);
    }
    if (moved) { ++ram_adapts_; ++ram_staged_; }
    return moved;
}

// `in`'s bytes (src) into `out`'s place of the compact copy, and the two offsets swapped - the checks again, as the caller
// may have read `src` a while ago (149-prefetch: ram_install)
bool FileExpertSource::install_resident(int64_t layer, int64_t in, int64_t out, const uint8_t* src) {
    if (!complement_ready_ || complement_host_ == nullptr || exchange_storage_.active() || !staged_.empty() ||
        layer < 0 || layer >= n_layers_ || in < 0 || out < 0 || in >= n_expert_ || out >= n_expert_ || in == out || src == nullptr)
        return false;
    const size_t i_in = (size_t) layer * (size_t) n_expert_ + (size_t) in;
    const size_t i_out = (size_t) layer * (size_t) n_expert_ + (size_t) out;
    if (i_in >= complement_offsets_.size() || i_out >= complement_offsets_.size()) return false;
    if (!override_.empty() && (override_[i_in] != nullptr || override_[i_out] != nullptr)) return false;
    const uint64_t at = complement_offsets_[i_out], bytes = layer_blob_bytes_[(size_t) layer];
    if (at == kNoComplement || complement_offsets_[i_in] != kNoComplement || at > complement_bytes_ ||
        bytes > complement_bytes_ - at) return false;
    // #2134: a Foresight slot may be copying out of `out`'s place right now; it is dropped (never served) before the place is rewritten
    ForesightSwap* fs = fs_swap_.load(std::memory_order_acquire);
    if (fs != nullptr) fs->hold(layer, (int32_t) out);
    std::memcpy((uint8_t*) complement_host_ + (size_t) at, src, (size_t) bytes);
    const bool moved = detail::exchange_cache_complement(complement_offsets_, i_out, i_in);
    if (fs != nullptr) fs->unhold(layer, (int32_t) out);
    if (!moved) return false;
    ++ram_adapts_;
    return true;
}

// ---- 149-prefetch: staged promotions.  The read of `in` runs on this source's own thread (one at a time: the drive's
// queue stays the decode's first); the install is the decode thread's (ram_install, at a window boundary).
bool FileExpertSource::ram_stage(int64_t layer, int64_t in, int64_t out) {
    if (!complement_ready_ || complement_host_ == nullptr || layer < 0 || layer >= n_layers_ || in < 0 || out < 0 ||
        in >= n_expert_ || out >= n_expert_ || in == out)
        return false;
    {
        std::lock_guard<std::mutex> lk(ram_mu_);
        static const size_t cap = [] {
            const char* v = std::getenv("STRATA_RAM_ADAPT_QUEUE");
            return v != nullptr ? (size_t) std::max(1, std::atoi(v)) : (size_t) 64;
        }();
        if (ram_jobs_.size() >= cap) return false;
        for (const auto& j : ram_jobs_)
            if (j->layer == layer && (j->in == in || j->out == out || j->in == out || j->out == in)) return false;
        auto j = std::make_unique<RamJob>();
        j->layer = layer; j->in = in; j->out = out; j->state = 0;
        ram_jobs_.push_back(std::move(j));
        if (!ram_thread_.joinable()) {
            ram_quit_ = false;
            ram_thread_ = std::thread([this] { ram_worker(); });
        }
    }
    ram_cv_.notify_one();
    return true;
}

bool FileExpertSource::ram_queued(int64_t layer, int64_t expert) const {
    std::lock_guard<std::mutex> lk(ram_mu_);
    for (const auto& j : ram_jobs_)
        if (j->layer == layer && (j->in == expert || j->out == expert)) return true;
    return false;
}

void FileExpertSource::ram_worker() {
    strata::platform::name_this_thread("strata-ram");
    for (;;) {
        RamJob* job = nullptr;
        {
            std::unique_lock<std::mutex> lk(ram_mu_);
            ram_cv_.wait(lk, [&] {
                if (ram_quit_) return true;
                for (const auto& j : ram_jobs_)
                    if (j->state == 0) return true;
                return false;
            });
            if (ram_quit_) return;
            for (const auto& j : ram_jobs_)
                if (j->state == 0) { job = j.get(); break; }
            job->state = 1;
        }
        const uint64_t bytes = layer_blob_bytes_[(size_t) job->layer];
        job->buf.resize((size_t) bytes);
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = copy_from_files(job->layer, job->in, job->buf.data());
        file_us_.fetch_add((uint64_t) std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count(),
                           std::memory_order_relaxed);
        if (ok) {
            file_read_bytes_.fetch_add(bytes, std::memory_order_relaxed);
            ram_staged_reads_.fetch_add(1, std::memory_order_relaxed);
        }
        std::lock_guard<std::mutex> lk(ram_mu_);
        job->state = ok ? 2 : 3;
    }
}

int64_t FileExpertSource::ram_install(const int32_t* host_res, int64_t n_expert) {
    std::vector<std::unique_ptr<RamJob>> done;
    {
        std::lock_guard<std::mutex> lk(ram_mu_);
        for (auto it = ram_jobs_.begin(); it != ram_jobs_.end();) {
            if ((*it)->state >= 2) { done.push_back(std::move(*it)); it = ram_jobs_.erase(it); }
            else ++it;
        }
    }
    int64_t n = 0;
    for (const auto& j : done) {
        if (j->state != 2) { ram_install_skips_.fetch_add(1, std::memory_order_relaxed); continue; }
        if (host_res != nullptr && (host_res[(size_t) (j->layer * n_expert + j->in)] >= 0 ||
                                    host_res[(size_t) (j->layer * n_expert + j->out)] >= 0)) {
            ram_install_gpu_.fetch_add(1, std::memory_order_relaxed);   // the GPU tier took one of them meanwhile
            continue;
        }
        if (install_resident(j->layer, j->in, j->out, j->buf.data())) ++n;
        else ram_install_skips_.fetch_add(1, std::memory_order_relaxed);
    }
    return n;
}

void FileExpertSource::ram_stop() {
    {
        std::lock_guard<std::mutex> lk(ram_mu_);
        ram_quit_ = true;
    }
    ram_cv_.notify_all();
    if (ram_thread_.joinable()) ram_thread_.join();
    std::lock_guard<std::mutex> lk(ram_mu_);
    ram_jobs_.clear();
    ram_quit_ = false;
}

int FileExpertSource::tier_of(int64_t layer, int64_t expert) const {
    if (layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return 2;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    if (complement_ready_ && resident_blob(index) != nullptr) return 0;
#if defined(__linux__)
    uint64_t total = 0;
    const uint64_t c = cached_bytes(layer, expert, total, false);
    return total > 0 && c >= total ? 1 : 2;
#else
    return 2;
#endif
}

void FileExpertSource::commit_copies() {
    fs_hold_staged(true);   // #2134: held until commit_flip has pointed `out` at its new place
    if (exchange_storage_.active()) return;   // STRATA_EXCHANGE_ROTATE: ownership moves in commit_flip, nothing to copy
    for (const Exchange& x : staged_) {
        const uint8_t* src = override_.empty() ? nullptr : override_[x.out];
        const uint64_t at = complement_offsets_[x.in];
        if (src != nullptr && at != kNoComplement && at <= complement_bytes_ && x.bytes <= complement_bytes_ - at &&
            complement_host_ != nullptr)
            std::memcpy((uint8_t*) complement_host_ + (size_t) at, src, (size_t) x.bytes);
    }
}

int64_t FileExpertSource::commit_flip() {
    int64_t n = 0;
    fs_hold_staged(true);   // (already held by commit_copies; idempotent)
    for (const Exchange& x : staged_) {
        const uint8_t* src = override_[x.out];
        if (exchange_storage_.active()) {   // STRATA_EXCHANGE_ROTATE: the same ownership transfer as commit_exchanges
            if (!exchange_storage_.commit(x.in, x.out, (size_t) x.q, src, (size_t) x.bytes)) {
                std::fprintf(stderr, "FileExpertSource: invalid exchange rotation commit; refusing corrupt residency\n");
                std::abort();
            }
            ++n;
            override_[x.out] = nullptr;
            continue;
        }
        const uint64_t at = complement_offsets_[x.in];
        // the same test as commit_copies: an exchange whose bytes were copied is the one that flips
        if (src != nullptr && at != kNoComplement && at <= complement_bytes_ && x.bytes <= complement_bytes_ - at &&
            complement_host_ != nullptr && detail::exchange_cache_complement(complement_offsets_, x.in, x.out))
            ++n;
        override_[x.out] = nullptr;
    }
    fs_hold_staged(false);
    staged_.clear();
    exchanges_ += n;
    return n;
}

const uint8_t* FileExpertSource::resident_blob(int64_t layer, int64_t expert) const {
    if (!complement_ready_ || complement_host_ == nullptr || layer < 0 || expert < 0 || layer >= n_layers_ ||
        expert >= n_expert_) return nullptr;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    if (index >= complement_offsets_.size()) return nullptr;
    return resident_blob(index);   // through the rotation's ownership table when STRATA_EXCHANGE_ROTATE is on
}

const uint8_t* FileExpertSource::blob_stable(int64_t layer, int64_t expert) {
    if (io_pf_ && role_ptr_.empty() && direct_.empty()) {
        if (complement_ready_ && layer >= 0 && expert >= 0 && layer < n_layers_ && expert < n_expert_) {
            const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
            if (resident_blob(index) != nullptr || (!override_.empty() && override_[index] != nullptr)) return blob(layer, expert);
        }
        const uint8_t* b = mapped_blob(layer, expert);
        if (b != nullptr) file_read_bytes_.fetch_add(layer_blob_bytes_[(size_t) layer], std::memory_order_relaxed);
        return b;
    }
    return blob(layer, expert);
}

int FileExpertSource::blob_parts(int64_t layer, int64_t expert, BlobPart* out) {
    // the GGUF mapped in place only (three tensors per expert); the RAM copy, an exchange override, the unbuffered
    // reads, a closed mapping and the I/O prefetch (STRATA_IO_PREFETCH: it preads the ranges the page cache lacks, which a copy out
    // of the mapping would fault in one page at a time) keep going through blob(), which knows them
    if (role_ptr_.empty() || !direct_.empty() || io_pf_ || unmapped_ || base_ == nullptr || layer < 0 || expert < 0 ||
        layer >= n_layers_ || expert >= n_expert_)
        return 0;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    if (complement_ready_ && (resident_blob(index) != nullptr || (!override_.empty() && override_[index] != nullptr))) return 0;
    Span sp[3];
    const int ns = spans(layer, expert, sp);
    for (int i = 0; i < ns; ++i) out[i] = {sp[i].p, sp[i].n, sp[i].at};
    if (ns > 0) {
        file_read_bytes_.fetch_add(layer_blob_bytes_[(size_t) layer], std::memory_order_relaxed);
        ++reads_;
    }
    return ns;
}

const uint8_t* FileExpertSource::blob(int64_t layer, int64_t expert) {
    if (base_ == nullptr || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return nullptr;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    const uint8_t* result = nullptr;
    bool from_files = true;
    if (complement_ready_) {
        result = resident_blob(index);
        if (result != nullptr) {
            ram_reads_.fetch_add(1, std::memory_order_relaxed);
            from_files = false;
        } else if (!override_.empty() && override_[index] != nullptr) {
            result = override_[index];
            from_files = false;
        }
    }
    if (from_files) {
        if (io_pf_ && role_ptr_.empty() && direct_.empty()) {
            result = io_blob(layer, expert);              // staged when it had to be read, the mapping when cached
        } else if (staged()) {
            result = staged_blob(layer, expert);          // counts its bytes
        } else {
            result = mapped_blob(layer, expert);
            if (result != nullptr) {
                file_read_bytes_.fetch_add(layer_blob_bytes_[(size_t) layer], std::memory_order_relaxed);
#if defined(__linux__)
                if (io_stats_) {   // STRATA_IO_STATS=1: what the page cache held when the kernels were handed the blob
                    uint64_t total = 0;
                    const uint64_t c = cached_bytes(layer, expert, total);
                    io_cached_.fetch_add(c, std::memory_order_relaxed);
                    io_uncached_.fetch_add(total - c, std::memory_order_relaxed);
                }
#endif
            }
        }
        if (complement_ready_ && result != nullptr) file_reads_.fetch_add(1, std::memory_order_relaxed);
    }
    if (result != nullptr) reads_.fetch_add(1, std::memory_order_relaxed);
    return result;
}

bool FileExpertSource::pinned(int64_t layer, int64_t expert) const {
    if (!complement_ready_ || !complement_pinned_ || complement_host_ == nullptr || layer < 0 || expert < 0 ||
        layer >= n_layers_ || expert >= n_expert_) return false;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    if (exchange_storage_.active()) return exchange_storage_.resident(index).host != nullptr;
    // soft_commit_exchanges: a staged `out` is judged by the place it will take (`in`'s), as after the commit
    const size_t at_index = !soft_in_.empty() && soft_in_[index] != SIZE_MAX ? soft_in_[index] : index;
    if (at_index >= complement_offsets_.size() || complement_offsets_[at_index] == kNoComplement) return false;
    // a partial pin (CS-T): only the registered prefix
    return !complement_partial_ ||
           complement_offsets_[at_index] + layer_blob_bytes_[(size_t) layer] <= complement_pin_limit_;
}

const uint8_t* FileExpertSource::device_alias(int64_t layer, int64_t expert) const {
    if (!pinned(layer, expert) || complement_device_ == nullptr) return nullptr;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    if (exchange_storage_.active()) return exchange_storage_.resident(index).device;
    if (!soft_in_.empty() && soft_in_[index] != SIZE_MAX)   // soft_commit_exchanges: out is read from its exchange buffer
        return override_.empty() || override_[index] == nullptr || xstage_dev_ == nullptr
                   ? nullptr : xstage_dev_ + (size_t) (override_[index] - xstage_);
    return complement_device_ + (size_t) complement_offsets_[index];
}

bool FileExpertSource::pcie_layer(int64_t layer) const {
    if (complement_ready_ && complement_pinned_ && complement_device_ != nullptr)
        return layer >= 0 && layer < n_layers_;
    return device_alias(layer, 0) != nullptr;
}

// ================================ THE ADAPTER ================================

void expert_pool_dispatch(void* user, const float* x_f, const int32_t* ids, const float* weights, int64_t n_embd,
                          int64_t k, float* out) {
    (void) weights;   // clause 2: `moe_combine` applies it on the device.  Not an oversight.
    ExpertDispatch& d = *(ExpertDispatch*) user;
    if (d.failed) return;   // a previous layer already failed; do not make it worse

    using namespace strata::kernels::cpu;
    // Clause 3: the blob's internal offsets are compile-time constants, so a mismatched geometry does not
    // produce a wrong answer - it produces a walk off the end of the blob into the next expert's bytes, which
    // is finite and plausible.  Refuse, name the number, and let the driver report it.
    if (n_embd != H) {
        d.failed = true;
        d.fail = "the expert kernel is compiled for a 2560-wide activation";
        d.fail_layer = d.layers;
        return;
    }
    if (expert_layout().native) {
        // plan v0.3 P6: a native pack runs its experts in verify windows only (the driver guarantees it)
        d.failed = true;
        d.fail = "the single-token expert path does not take a native (IQ) pack";
        d.fail_layer = d.layers;
        return;
    }
    if (k > (int64_t) d.jobs.size()) d.jobs.resize((size_t) k);

    d.src->begin_layer(d.layers, ids, k);

    // Clause 1: rebuilt from `x_f` on EVERY call.  `x_f` is mapped pinned memory whose address never changes,
    // so anything cached against it would be layer 0's activation reused 48 times.
    act_quant_q8_1(x_f, H, d.act);

    // ---- R4.2c: THE POOL'S HALF OF THE SPLIT.  **IT DOES NOT DECIDE ANYTHING - `Launch` ALREADY DID.**
    //
    // The decision has to be made on THIS layer's ids, and `Launch` is the only callback that runs before the
    // pool while the ids are known (the doorbell publishes them when the ring fires).  So `expert_hit_run`
    // decides, and this consumes `d.is_hit`.  The first version decided here instead, which meant `Launch`
    // computed the PREVIOUS layer's experts into this layer's rows: C1 went from mean KL 9.69e-02 to 1.03e+00.
    //
    // `njobs` indexes the JOB ARRAY and `i` indexes the OUTPUT - they are the same only when nothing is a hit.
    const bool graph_hits = d.host_res != nullptr;
    const bool use_hits = graph_hits || (d.hits_ready() && d.decided);
    int64_t njobs = 0;

    if (d.remote_count > 0) {
        int32_t kind[32];
        if (k > 32) {
            d.failed = true; d.fail = "remote experts: routing width exceeds 32"; return;
        }
        for (int64_t i = 0; i < k; ++i)
            kind[i] = use_hits && ids[i] >= 0 && ids[i] < d.n_expert && (graph_hits
                ? d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) ids[i]] >= 0
                : d.is_hit[(size_t) i] != 0) ? 0 : -1;
        static thread_local std::string remote_error;
        for (int r = 0; r < d.remote_count; ++r)
            if (!d.remote[r]->begin(d.layers, x_f, ids, 1, k, kind, d.host_res, remote_error)) {
                d.failed = true; d.fail = remote_error.c_str(); d.fail_layer = d.layers; return;
            }
    }

    for (int64_t i = 0; i < k; ++i) {
        const int64_t e = ids[i];
        if (e < 0 || e >= d.n_expert) {
            d.failed = true;
            d.fail = "a routed expert id is out of range";
            d.fail_layer = d.layers;
            d.fail_expert = e;
            return;
        }
        const uint8_t* b = d.src->blob(d.layers, e);
        if (b == nullptr) {
            // The one failure the loop cannot see.  Leaving `out` at its previous contents would feed the NEXT
            // layer a stale expert vector, which `moe_combine` would weight and add - the token would still be
            // finite and would still be wrong, 48 layers deep.
            d.failed = true;
            d.fail = "the expert source could not produce a blob";
            d.fail_layer = d.layers;
            d.fail_expert = e;
            ++d.missing;
            return;
        }
        // A hit's row was zeroed by `Launch` and belongs to the GPU; the pool must not touch it.
        if (use_hits && (graph_hits ? d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e] >= 0
                                    : d.is_hit[(size_t) i] != 0)) {
            if (graph_hits) ++d.cache_hits;
            // The GPU owns this row and `hit_out` is zeroed, so the CPU's contribution is zero - but
            // `y_miss` is a REUSED pinned buffer, so the row must be written, not merely skipped.
            std::memset(out + (size_t) i * (size_t) n_embd, 0, (size_t) n_embd * sizeof(float));
            continue;
        }
        if (graph_hits) ++d.cache_refused;   // token graph: a miss (nothing is admitted during a token)

        bool remote_owns = false;
        for (int r = 0; r < d.remote_count; ++r) remote_owns |= d.remote[r]->owns(i);
        if (remote_owns) {
            std::memset(out + (size_t) i * (size_t) n_embd, 0, (size_t) n_embd * sizeof(float));
            continue;
        }

        // `njobs` indexes the JOB ARRAY and `i` indexes the OUTPUT - they are the same only when nothing is a
        // hit, and using one for the other is how a hit's row would get two experts summed into it.
        ExpertJob& j = d.jobs[(size_t) njobs++];
        j.blob = b;
        j.act = &d.act;             // SHARED across the batch: one conversion serves all ten experts
        j.out = out + (size_t) i * (size_t) n_embd;
        j.weight = 1.0f;            // clause 2: a diagnostic field, NOT the router weight
        j.slot = (int) i;
    }

    // Plan v0.3 P4: rows of every expert across all threads (bitwise the same as `run`).
    if (d.split_rows) d.pool->run_split(d.jobs.data(), (int) njobs);
    else d.pool->run(d.jobs.data(), (int) njobs);
    if (d.remote_count > 0) {
        static thread_local std::string remote_error;
        for (int r = 0; r < d.remote_count; ++r)
            if (!d.remote[r]->finish(out, remote_error)) {
                d.failed = true; d.fail = remote_error.c_str(); d.fail_layer = d.layers; return;
            }
    }
    ++d.layers;
    d.experts += k;
}

namespace {
// the verify window's per-entry tables in `expert_pool_dispatch_multi` (`kind`, `distinct`, `first_of`)
// are fixed arrays of this many entries: a window's rows (kWindowMaxT) of the model's 10 routed experts must fit, and a
// larger k is refused at run time rather than written past them.  128 for the 8-row window (as it was), 256 for 16.
constexpr int64_t kMaxWindowEntries = strata::kernels::kWindowMaxT <= 8 ? 128 : 256;
static_assert(strata::kernels::kWindowMaxT * 10 <= kMaxWindowEntries, "a verify window's entries overflow the tables");
}  // namespace

namespace {
// STRATA_ROUTE_TAIL_SKIP: the rank from which a missed expert that every token of the window routes that low is skipped.  An
// explicit environment value is read once; the serve start-up (generate.cpp) sets the CUDA default where it applies.
std::atomic<int> g_tail_skip_rank{-1};   // -1: not resolved yet
}  // namespace
int tail_skip_rank() {
    int r = g_tail_skip_rank.load(std::memory_order_relaxed);
    if (r < 0) {
        const char* v = std::getenv("STRATA_ROUTE_TAIL_SKIP");
        r = v != nullptr ? std::atoi(v) : 0;
        r = r > 0 && r < 10 ? r : 0;
        g_tail_skip_rank.store(r, std::memory_order_relaxed);
    }
    return r;
}
void set_tail_skip_rank(int r) { g_tail_skip_rank.store(r > 0 && r < 10 ? r : 0, std::memory_order_relaxed); }

static void lat_quant_q8_1(const float* x, int n, uint8_t* o) {   // = quantize_q8_1_kernel, per 32: d, sum, qs
    const int nb = n / 32;
    for (int b = 0; b < nb; ++b) {
        const float* xb = x + b * 32;
        uint8_t* ob = o + b * 36;
        float amax = 0.0f, sum = 0.0f;
        for (int j = 0; j < 32; ++j) {
            const float v = xb[j];
            amax = fmaxf(amax, fabsf(v));
            sum += v;
        }
        const float d32 = amax / 127.0f;
        const float d = fabsf(d32) > 65504.0f ? copysignf(65504.0f, d32) : d32;
        const uint16_t dh = strata::kernels::f16_from_f32(d);
        const uint16_t sh = strata::kernels::f16_from_f32(sum);
        ob[0] = (uint8_t)(dh & 0xFFu); ob[1] = (uint8_t)(dh >> 8);
        ob[2] = (uint8_t)(sh & 0xFFu); ob[3] = (uint8_t)(sh >> 8);
        int8_t* q = (int8_t*)(ob + 4);
        // amax/sum stay scalar above (their order feeds the f16 d/sum - a SIMD reduction would
        // reorder it).  The qs loop is per element, no reduction: the AVX2 copy in kq_avx2.cpp.
        // THE 32-float SNAPSHOT is the race fix (probe-proven 2026-11-04): lh has a writer that
        // races this read - consuming it at a different moment changed the window's output (the
        // day's whole "parity red" chain).  The copy pins what the quant consumes, a few ns/block.
        // STRATA_LH_WHO=1 double-reads to catch the writer in the act.
        static const bool qs_simd_off = std::getenv("STRATA_NO_QS_SIMD") != nullptr;
        static const bool lh_who = std::getenv("STRATA_LH_WHO") != nullptr;
        float snap[32];
        std::memcpy(snap, xb, 32 * sizeof(float));
        if (lh_who) {   // read it again right away: a racing writer shows up as a delta
            static std::atomic<int> lh_diffs{0};
            for (int j = 0; j < 32; ++j)
                if (snap[j] != xb[j]) {
                    if (lh_diffs.fetch_add(1) < 8)
                        std::fprintf(stderr, "LHWHO b=%d j=%d: before %g after %g\n", b, j, snap[j], xb[j]);
                    break;
                }
        }
        if (amax != 0.0f && !qs_simd_off && strata::kernels::cpu::kq256_supported(8))
            strata::kernels::cpu::lat_quant_q8_1_qs_avx2(snap, &d, q);
        else {
            for (int j = 0; j < 32; ++j) {
                float v;
                if (amax == 0.0f) v = 0.0f;
                else v = roundf(xb[j] / d);
                v = v > 127.0f ? 127.0f : (v < -127.0f ? -127.0f : v);
                q[j] = (int8_t) v;
            }
        }
    }
}

// ---- latent MoE: the CPU pool's expert arithmetic (edge0).  The GPU computes the whole window's D stage
// and its result (the q8_1 image of xl, row t*G + g) is in mapped host memory - the pool never recomputes D.
// Per CPU-served entry (kind == -1): the A blob computes h = A[e] . xl[t,g] (I then ELL, the pool's native
// row phases over the GPU's own quantized xl rows), then contribution = U[g] . q8_1(h) as a mode-7 row dot
// spread over the workers, written into the entry's mapped row (out + i*H).
static void latent_cpu_run(ExpertDispatch& d, const float* x_f, const int32_t* ids,
                           int64_t n_tok, int64_t k, const int32_t* kind, float* out) {
    using namespace strata::kernels::cpu;
    const int64_t l = d.layers;
    const NativeFmt& f = expert_layout().fmt[(size_t) l];
    const int64_t ELL = d.lat_ELL, H = d.lat_H;
    const int G = d.lat_G, E = d.lat_E;
    const strata::core::LatentDense& dd = *d.lat_dense;
    const uint8_t* U = dd.u(l);
    const int32_t* labels = d.lat_labels + (size_t) l * (size_t) E;
    const size_t u_row = dd.u_row_bytes();
    const size_t xlq_row = (size_t) (ELL / 32) * 36;
    const double t_enter_ms = (double) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count() / 1e6;   // PROBE-OTHER: entry

    // the CPU entries: j -> (entry index i, token, expert, group); the groups used
    static thread_local std::vector<int64_t> ent_i;
    static thread_local std::vector<int> ent_t, ent_e, ent_g;
    ent_i.clear(); ent_t.clear(); ent_e.clear(); ent_g.clear();
    bool used[64] = {};
    for (int64_t i = 0; i < n_tok * k; ++i) {
        if (kind[i] != -1) continue;
        const int32_t e = ids[i];
        if (e < 0 || e >= E) { d.failed = true; d.fail = "a routed expert id is out of range"; d.fail_layer = l; d.fail_expert = e; return; }
        const int g = labels[(size_t) e];
        if (g < 0 || g >= G) { d.failed = true; d.fail = "latent: a bad group label"; d.fail_layer = l; d.fail_expert = e; return; }
        ent_i.push_back(i); ent_t.push_back((int) (i / k)); ent_e.push_back(e); ent_g.push_back(g);
        if (g < 64) used[g] = true;
    }
    if (ent_t.empty()) return;
    static const bool s_pool_trace = std::getenv("STRATA_POOL_TRACE") != nullptr;
    if (s_pool_trace) std::fprintf(stderr, "POOL-ENTRY l=%d\n", (int) l);
    const float* xl = (const float*) d.lat_xlq;   // the D stage's f32 rows (mapped), row t*G + g
    if (xl == nullptr) { d.failed = true; d.fail = "latent CPU share: no D result for this window"; d.fail_layer = l; return; }
    double t_xlseq_ms = 0.0;
    if (d.lat_xlseq != nullptr) {   // the D stage's doorbell: the rows of layer l are in when it reaches l+1
        const uint32_t want = (uint32_t) (l + 1);
        if (s_pool_trace) std::fprintf(stderr, "POOL-DOORBELL l=%d want=%u got=%u\n", (int) l, want, *(volatile const uint32_t*) d.lat_xlseq);
        const auto t0 = std::chrono::steady_clock::now();
        while (*(volatile const uint32_t*) d.lat_xlseq < want) {
            if (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() > 20000.0) {
                d.failed = true; d.fail = "latent CPU share: the D rows never arrived for this layer";
                d.fail_layer = l;
                return;
            }
            std::this_thread::yield();
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        t_xlseq_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (s_pool_trace) std::fprintf(stderr, "POOL-QUANT l=%d\n", (int) l);
    }

    // ---- A: the blob experts on the pool's native row phases.  The activation is quantized ON THE HOST from
    // the D stage's f32 rows (native_quant_act -> the blobs' vec_dot_type, Q8_0 for NVFP4); the layer's
    // Hadamard fold stays off - the D output is already in the blob's own basis (v1's verified fix).
    const double t_ew_ms = (double) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count() / 1e6;   // PROBE-OTHER: ew end
    static thread_local std::vector<float> lh;
    static thread_local std::vector<uint8_t> act;   // one Q8_0 activation row per CPU entry
    static thread_local std::vector<ExpertJobMulti> jobs;
    static thread_local std::vector<char> taken;
    lh.assign((size_t) ent_t.size() * (size_t) ELL, 0.f);
    jobs.clear();
    taken.assign(ent_t.size(), 0);
    NativeFmt f_nr = f; f_nr.had_gu = false;   // no rotation: the D output is the blob's own basis
    const size_t act_row = f.act_bytes;        // per-entry activation bytes (the gu vec_dot_type, Q8_0)
    // DESIGN-4: the act rows arrive GPU-made (mapped, bit-exact - probe_std's zero diff; the D
    // stage quantizes them before the doorbell, so xlseq covers them).  The jobs point straight
    // at the mapped rows - zero host compute, zero copy.  STRATA_ACT_FALLBACK=1 keeps the old
    // host quantization (the A/B arm); STRATA_ACT_CHECK memcmps GPU rows vs the host reference
    // (zero-row guard included - the mod5 lesson).
    static const bool act_fallback = std::getenv("STRATA_ACT_FALLBACK") != nullptr;
    static const bool act_check = std::getenv("STRATA_ACT_CHECK") != nullptr;
    const uint8_t* act0 = (const uint8_t*) d.lat_act0;
    // DESIGN-4's GPU rows are ALWAYS Q8_0 (verify.cpp quantize_q8_0_rows), but this layer's
    // blobs may dot against another activation format (Q3_K -> ggml-cpu's Q8_K, 292 B per
    // 256 values): a non-Q8_0 layer must not consume them - the bytes would be misread and
    // the row stride would bleed into the next row.  Only a layer whose own act_row is the
    // Q8_0 one takes the GPU-made rows; the rest fall back to the host quantizer.
    const size_t act_q8_0_row = (size_t) (ELL / 32) * 34;
    const bool act_gpu = act0 != nullptr && !act_fallback && act_row == act_q8_0_row;
    act.assign((size_t) ent_t.size() * act_row, 0);
    if (act_gpu) {
        for (size_t j = 0; j < ent_t.size(); ++j) {
            const uint8_t* src = act0 + ((size_t) ent_t[j] * (size_t) G + (size_t) ent_g[j]) * act_row;
            if (act_check) {   // the GPU-vs-host probe, first diff only, zero-head guard
                static std::atomic<int> act_diffs{0};
                static thread_local std::vector<uint8_t> act_ref;
                act_ref.assign(act_row, 0);
                native_quant_act(f_nr, xl + ((size_t) ent_t[j] * (size_t) G + (size_t) ent_g[j]) * (size_t) ELL,
                                 act_ref.data());
                const uint8_t* a = act_ref.data();
                if (a[0] == 0 && a[1] == 0 && a[2] == 0 && a[3] == 0)
                    if (act_diffs.fetch_add(1) < 8)
                        std::fprintf(stderr, "ACTCHECK j=%zu: GPU ZERO HEAD\n", j);
                if (std::memcmp(a, src, act_row) != 0)
                    for (size_t k = 0; k < act_row; ++k)
                        if (a[k] != src[k]) {
                            if (act_diffs.fetch_add(1) < 8)
                                std::fprintf(stderr, "ACTCHECK j=%zu byte %zu: host %02x vs gpu %02x\n", j, k, a[k], src[k]);
                            break;
                        }
            }
            std::memcpy(act.data() + j * act_row, src, act_row);
        }
    } else {
        auto* const tr = ggml_get_type_traits_cpu((ggml_type) f_nr.gu_act);
        auto const ff = tr->from_float;
        for (size_t j = 0; j < ent_t.size(); ++j) {
            const float* src = xl + ((size_t) ent_t[j] * (size_t) G + (size_t) ent_g[j]) * (size_t) ELL;
            ff(src, act.data() + j * act_row, (int64_t) ELL);
        }
    }
    const double t_quant_ms = (double) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count() / 1e6;   // DIAGA: relative marker
    int64_t t_blob_ns = 0;
    for (size_t i0 = 0; i0 < ent_t.size(); ++i0) {
        if (taken[i0]) continue;
        const int e = ent_e[i0];
        ExpertJobMulti jb{};
        const auto tb0 = std::chrono::steady_clock::now();
        jb.blob = d.src->blob(l, e);
        t_blob_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - tb0).count();
        if (jb.blob == nullptr) { d.failed = true; d.fail = "the expert source could not produce a blob"; d.fail_layer = l; d.fail_expert = e; ++d.missing; return; }
        for (size_t i = i0; i < ent_t.size() && jb.nt < MAXT; ++i) {
            if (taken[i] || ent_e[i] != e) continue;
            taken[i] = 1;
            jb.nact[jb.nt] = act.data() + i * act_row;
            jb.out[jb.nt] = lh.data() + i * (size_t) ELL;
            ++jb.nt;
        }
        jobs.push_back(jb);
    }
    if (std::getenv("STRATA_DIAG_A") != nullptr) {
        static std::atomic<int64_t> dwin{0}, djobs{0}, dent{0};
        int64_t ent_sum = 0;
        for (const auto& jb : jobs) ent_sum += jb.nt;
        dwin.fetch_add(1, std::memory_order_relaxed);
        djobs.fetch_add((int64_t) jobs.size(), std::memory_order_relaxed);
        dent.fetch_add(ent_sum, std::memory_order_relaxed);
        if (dwin.load(std::memory_order_relaxed) % 48 == 0)
            std::fprintf(stderr, "DIAGA win=%lld jobs=%.1f entries=%.1f\n",
                         (long long) dwin.load(), (double) djobs.load() / dwin.load(),
                         (double) dent.load() / dwin.load());
    }
    const double t_a0_ms = (double) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count() / 1e6;   // PROBE-OTHER: jb end
    d.pool->run_split_multi_native(f, jobs.data(), (int) jobs.size(), NativeArithmetic::Default);
    const double t_a1_ms = (double) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count() / 1e6;   // DIAGA: A wall
    if (s_pool_trace) std::fprintf(stderr, "POOL-A-END l=%d\n", (int) l);

    // ---- U: contribution[j] = U[g] . q8_1(h_j): one matvec per used group over its entries (mode 7)
    static thread_local std::vector<uint8_t> hq;
    static thread_local std::vector<const void*> flat_act;
    static thread_local std::vector<float*> flat_out;
    static thread_local std::vector<ExpertPool::LatentMatvec> mvs;
    hq.resize(xlq_row * ent_t.size());
    for (size_t j = 0; j < ent_t.size(); ++j) lat_quant_q8_1(lh.data() + j * (size_t) ELL, (int) ELL, hq.data() + j * xlq_row);
    if (s_pool_trace) std::fprintf(stderr, "POOL-HQ-END l=%d\n", (int) l);
    // knife-0 (STRATA_LATENT_U_GPU=1): skip the CPU U matvec (mode 7); the GPU computes U from this hq.
    static const bool u_gpu = [] { const char* v = std::getenv("STRATA_LATENT_U_GPU"); return v != nullptr && std::atoi(v) != 0; }();
    { static std::atomic<int> pn{0}; if (pn.fetch_add(1, std::memory_order_relaxed) < 3)
      std::fprintf(stderr, "K0PROBE u_gpu=%d LUGPU=%s nent=%zu n_tok=%lld k=%lld\n",
                   (int) u_gpu, std::getenv("STRATA_LATENT_U_GPU") ? std::getenv("STRATA_LATENT_U_GPU") : "(null)",
                   ent_t.size(), (long long) n_tok, (long long) k); std::fflush(stderr); }
    if (const char* hqd = std::getenv("STRATA_HQ_DUMP")) {   // knife-0 debug: the hq rows this group computed
        static std::atomic<int> hqn{0};
        if (hqn.fetch_add(1, std::memory_order_relaxed) < 2) {
            FILE* f = std::fopen(hqd, "ab");
            if (f != nullptr) {
                std::fwrite(hq.data(), 1, hq.size(), f);
                std::fclose(f);
            }
        }
    }
    if (u_gpu) {
        uint8_t* lhq = (uint8_t*) out;   // m_ymiss_ reinterpreted: Q8_1, routed order (xlq_row per entry)
        { static std::atomic<int> env_n{0}; if (env_n.fetch_add(1, std::memory_order_relaxed) < 2)
          std::fprintf(stderr, "DIAGHQ-ENV u_gpu=1 LUGPU=%s DIAGHQ=%s n_tok=%lld k=%lld\n",
                       std::getenv("STRATA_LATENT_U_GPU") ? std::getenv("STRATA_LATENT_U_GPU") : "(null)",
                       std::getenv("STRATA_DIAG_HQ") ? std::getenv("STRATA_DIAG_HQ") : "(null)",
                       (long long) n_tok, (long long) k); std::fflush(stderr); }
        std::memset(lhq, 0, (size_t) n_tok * (size_t) k * xlq_row);   // skipped (kind=-2) rows -> zero hq -> zero U
        for (size_t j = 0; j < ent_t.size(); ++j) {
            const int r = ent_i[j];
            std::memcpy(lhq + (size_t) r * xlq_row, hq.data() + j * xlq_row, xlq_row);
        }
        static std::atomic<int> diaghq_n{0};
        if (diaghq_n.fetch_add(1, std::memory_order_relaxed) < 5 && !ent_t.empty()) {
            const uint8_t* src = hq.data();
            const uint8_t* dst = lhq + (size_t) ent_i[0] * xlq_row;
            const int cmp = std::memcmp(src, dst, xlq_row);
            const uint16_t d = *(const uint16_t*) dst, s = *(const uint16_t*) (dst + 2);
            const bool nan_d = (d & 0x7C00) == 0x7C00 && (d & 0x3FF) != 0;
            const bool nan_s = (s & 0x7C00) == 0x7C00 && (s & 0x3FF) != 0;
            std::fprintf(stderr, "DIAGHQ nent=%zu out=%p tb_off_bytes=%lld ent_i0=%d cmp=%d d=0x%04x(nan=%d) s=0x%04x(nan=%d) q0..3=%d,%d,%d,%d\n",
                         ent_t.size(), (const void*) out, (long long) (ent_i[0] * xlq_row), ent_i[0], cmp,
                         d, (int) nan_d, s, (int) nan_s,
                         (int) (int8_t) dst[4], (int) (int8_t) dst[5], (int) (int8_t) dst[6], (int) (int8_t) dst[7]);
            std::fflush(stderr);
        }
        d.multi_entries += (int64_t) ent_t.size();
        return;
    }
    mvs.clear();
    flat_act.reserve(ent_t.size());
    flat_out.reserve(ent_t.size());
    int mv_base[64];
    for (int g = 0; g < G; ++g) {
        mv_base[g] = (int) flat_act.size();
        if (g >= 64 || !used[g]) continue;
        for (size_t j = 0; j < ent_t.size(); ++j) {
            if (ent_g[j] != g) continue;
            flat_act.push_back(hq.data() + j * xlq_row);
            flat_out.push_back(out + (size_t) ent_i[j] * (size_t) H);
        }
    }
    for (int g = 0; g < G; ++g) {
        if (g >= 64 || !used[g]) continue;
        const int i0 = mv_base[g];
        const int i1 = g + 1 < G ? mv_base[g + 1] : (int) flat_act.size();
        if (i1 <= i0) continue;
        ExpertPool::LatentMatvec mv{};
        mv.w = U + (size_t) g * (size_t) H * u_row;
        mv.row_bytes = u_row;
        mv.nrows = (int) H;
        mv.ncols = (int) ELL;
        mv.nt = i1 - i0;
        mv.act = flat_act.data() + i0;
        mv.out = flat_out.data() + i0;
        mvs.push_back(mv);
    }
    const double t_u1_ms = (double) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count() / 1e6;   // DIAGA: U wall
    static std::atomic<double> s_ur{0};
    const auto ur0 = std::chrono::steady_clock::now();
    d.pool->run_latent_rows(mvs.data(), (int) mvs.size());
    s_ur.fetch_add((double) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - ur0).count() / 1e6, std::memory_order_relaxed);
    d.multi_entries += (int64_t) ent_t.size();
    if (std::getenv("STRATA_DIAG_A") != nullptr) {
        static std::atomic<double> s_xlseq{0}, s_a{0}, s_u{0}, s_blob{0};
        static std::atomic<int64_t> s_calls{0};
        static std::atomic<double> s_ew{0}, s_jb{0}, s_am{0};   // PROBE-OTHER: the glue walls
        s_ew.fetch_add(t_ew_ms - t_enter_ms, std::memory_order_relaxed);
        s_jb.fetch_add(t_a0_ms - t_quant_ms, std::memory_order_relaxed);
        s_am.fetch_add(t_quant_ms - t_ew_ms, std::memory_order_relaxed);
        s_xlseq.fetch_add(t_xlseq_ms, std::memory_order_relaxed);
        s_a.fetch_add(t_a1_ms - t_a0_ms, std::memory_order_relaxed);
        s_u.fetch_add(t_u1_ms - t_a1_ms, std::memory_order_relaxed);
        s_blob.fetch_add((double) t_blob_ns / 1e6, std::memory_order_relaxed);
        if (s_calls.fetch_add(1, std::memory_order_relaxed) % 48 == 0)
            std::fprintf(stderr, "DIAGP ms/call: xlseq %.3f blob %.3f A %.3f U %.3f UR %.3f EW %.3f JB %.3f AM %.3f (n %lld)\n",
                         s_xlseq.load() / (s_calls.load() + 1), s_blob.load() / (s_calls.load() + 1),
                         s_a.load() / (s_calls.load() + 1), s_u.load() / (s_calls.load() + 1),
                         s_ur.load() / (s_calls.load() + 1),
                         s_ew.load() / (s_calls.load() + 1), s_jb.load() / (s_calls.load() + 1),
                         s_am.load() / (s_calls.load() + 1),
                         (long long) (s_calls.load() + 1));
    }
}

void expert_pool_dispatch_multi(ExpertDispatch& d, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k,
                                float* out, strata::kernels::cpu::NativeArithmetic arithmetic) {
    using namespace strata::kernels::cpu;
    if (d.failed) return;
    d.lat_xlq = strata::core::latent_xl_current();      // this window's D rows (null: not latent)
    d.lat_xlseq = strata::core::latent_xlseq_current();
    d.lat_act0 = strata::core::latent_act0_current();   // DESIGN-4: the D rows as Q8_0, GPU-made
    if (n_tok < 1 || n_tok > strata::kernels::kWindowMaxT) {
        d.failed = true;
        d.fail = "a verify window has more tokens than the multi-token expert kernel takes";
        d.fail_layer = d.layers;
        return;
    }
    if (d.lookahead != nullptr) {
        if (d.lookahead->stats_on()) {
            if (d.layers == 0) d.lookahead->stats_window_start();
            d.lookahead->stats_score(d.layers, ids, n_tok * k, d.host_res, d.pcie_num);
        }
        d.lookahead->submit(d.layers, x_f, n_tok, d.host_res);   // CS-T: warm layer + 1
    }
    if (k < 1 || n_tok * k > kMaxWindowEntries) {
        d.failed = true;
        d.fail = "a verify window routes more entries than the expert pool's window tables hold";
        d.fail_layer = d.layers;
        return;
    }
    if ((int64_t) d.act_multi.size() < n_tok) d.act_multi.resize((size_t) strata::kernels::kWindowMaxT);
    const ExpertLayout& lay = expert_layout();
    const bool native = lay.native;
    if (native && d.nact_multi.size() < (size_t) strata::kernels::kWindowMaxT * kNativeActBytes)
        d.nact_multi.resize((size_t) strata::kernels::kWindowMaxT * kNativeActBytes);
    if (d.job_of.size() != (size_t) d.n_expert) d.job_of.assign((size_t) d.n_expert, (int16_t) -1);
    if (d.jobs_multi.size() < (size_t) (n_tok * k)) d.jobs_multi.resize((size_t) (strata::kernels::kWindowMaxT * k));
    static const bool ptrace = std::getenv("STRATA_POOL_TRACE") != nullptr;
    auto pt = [&](const char* what, long long a = -1) {
        if (ptrace) { std::fprintf(stderr, "pool trace: layer %lld %s %lld\n", (long long) d.layers, what, a); std::fflush(stderr); }
    };
    const auto c0 = std::chrono::steady_clock::now();
    const FileExpertSource::DecodeScope decode_scope;   // 149 STRATA_IO_PF_GATE: this thread reads for the decode
    // 149-cold STRATA_COLD_OVERLAP: 0 off, 1 overlap, 2 measure only, 3 overlap + zero-copy, 4 zero-copy only.
    // 0.2.0: unset, 4 for the GGUF read in place on Linux (zero-copy needs the page cache's residency query), else 0
    static const int cold_env = [] {
        const char* v = std::getenv("STRATA_COLD_OVERLAP");
        return v == nullptr || v[0] == '\0' ? -1 : std::clamp(std::atoi(v), 0, 4);
    }();
    d.cold_mode = cold_env >= 0 ? cold_env : (d.src != nullptr && d.src->linux_file_defaults() ? 4 : 0);
    pt("begin");
    const double t_b0_ms = (double) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count() / 1e6;   // DIAGB2: before begin_layer
    d.src->begin_layer(d.layers, ids, n_tok * k);
    pt("begun");
    if (!d.usage.empty())
        for (int64_t i = 0; i < n_tok * k; ++i)
            if (ids[i] >= 0 && ids[i] < d.n_expert) d.usage[(size_t) d.layers * (size_t) d.n_expert + (size_t) ids[i]] += 1.0f;
    // ---- plan v0.3 P6: the GPU's share, decided and published FIRST so the GPU starts while the CPU works.
    // Distinct experts in routing order; resident ones and the last pcie_num/256 of the missed ones go to the GPU.
    const int64_t n = n_tok * k;
    const double t_c0_ms = (double) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count() / 1e6;   // DIAGB: dispatch entry
    double t_d1_ms = 0, t_d2_ms = 0, t_k1_ms = 0, t_k2_ms = 0;   // DIAGB2 stamps, assigned inside the plan block
    static std::atomic<int64_t> s_pinned_ns{0}, s_blob_ns{0}, s_pinned_calls{0};
    static std::atomic<int64_t> s_publish_ns{0}, s_fetch_ns{0}, s_pub_calls{0};
    int32_t kind[kMaxWindowEntries];       // per entry: -1 CPU, 0 VRAM, 1 PCIe
    if (d.plan != nullptr && n <= kMaxWindowEntries && n <= d.plan->cap) {
        int64_t distinct[kMaxWindowEntries], first_of[kMaxWindowEntries];
        int nd = 0, nmiss = 0;
        // a helper GPU (RemoteExperts) holding the expert computes it in its own VRAM once begin() claims the row: like
        // the peer tier's, such an expert is no miss of this plan and never part of the PCIe share, which would read it
        // over the primary's link instead (2x RX 6900 XT, IQ3_S, --remote-expert-opt with the probed share 0.39: ~4
        // helper experts per layer-window moved to PCIe, 59 ms per window against 34 ms)
        auto helper_holds = [&](int32_t e) {
            for (int r = 0; r < d.remote_count; ++r)
                if (d.remote[r]->holds(d.layers, e)) return true;
            return false;
        };
        for (int64_t i = 0; i < n; ++i) {
            first_of[i] = i;
            for (int64_t j = 0; j < i; ++j)
                if (ids[j] == ids[i]) { first_of[i] = first_of[j]; break; }
            if (first_of[i] == i) {
                distinct[nd++] = i;
                const int32_t e = ids[i];
                if (e >= 0 && e < d.n_expert && d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e] < 0 &&
                    !(d.peer != nullptr && d.peer->has(d.layers, e)) && !helper_holds(e)) ++nmiss;
            }
        }
        t_d1_ms = (double) std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count() / 1e6;   // DIAGB2: distinct done
        // STRATA_ROUTE_TAIL_SKIP=R (default 7 on CUDA serve, =0 off; changes the output): a missed expert that EVERY token of the window routes
        // only at rank >= R (the smallest weights of the top-10) is neither copied nor computed; its contribution is
        // zero, without renormalisation (the others keep the weight the router gave them).  Less PCIe and CPU work.
        const int tail_rank = tail_skip_rank();
        bool skip_d[kMaxWindowEntries];
        for (int q = 0; q < nd; ++q) skip_d[q] = false;
        if (tail_rank > 0 && d.peer == nullptr && d.remote_count == 0) {
            int minr[kMaxWindowEntries];
            for (int q = 0; q < nd; ++q) minr[q] = 99;
            for (int64_t i = 0; i < n; ++i) {
                int q = 0;
                while (q < nd && distinct[q] != first_of[i]) ++q;
                if (q < nd) minr[q] = std::min(minr[q], (int) (i % k));
            }
            for (int q = 0; q < nd; ++q) {
                const int32_t e = ids[distinct[q]];
                if (e < 0 || e >= d.n_expert || minr[q] < tail_rank) continue;
                if (d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e] >= 0) continue;   // resident
                skip_d[q] = true;
                --nmiss;
                ++d.tail_experts;
            }
        }
        t_d2_ms = (double) std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count() / 1e6;   // DIAGB2: tail_skip done
        // STRATA_PCIE_BALANCE (opt-in, #1548): the count comes from the measured CPU and GPU costs instead of the share
        const bool pcie_ok = (d.pcie_num > 0 || d.pcie_model.on) && d.src->pcie_layer(d.layers);
        int m = pcie_ok ? d.pcie_model.pick(nmiss, (nmiss * d.pcie_num) >> 8) : 0;
        // STRATA_PCIE_MIN1 (opt-in): the share rounds to 0 for 1-2 misses per layer, so the PCIe lane idles on the typical
        // decode step; with >= 2 misses give it at least one (the last in routing order).  It only flies if its blob is
        // pinned (the fetch below checks), else it stays a CPU miss exactly as before.  Read once.
        static const bool pcie_min1 = [] { const char* v = std::getenv("STRATA_PCIE_MIN1"); return v && std::atoi(v) != 0; }();
        if (pcie_min1 && pcie_ok && m < 1 && nmiss >= 2) m = 1;
        int miss_rank = 0, groups = 0, entries = 0, fetches = 0;
        GpuPlanSink& P = *d.plan;
        const uint8_t* dma_src[64];
        int64_t pcie_i0[64];
        for (int q = 0; q < nd; ++q) {
            const int64_t i0 = distinct[q];
            const int32_t e = ids[i0];
            int kd = -1;
            unsigned long long ptr = 0;
            if (skip_d[q]) {                       // STRATA_ROUTE_TAIL_SKIP: neither copied nor computed
                for (int64_t i = i0; i < n; ++i)
                    if (first_of[i] == i0) kind[i] = -2;
                continue;
            }
            if (e >= 0 && e < d.n_expert) {
                const int32_t slot = d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e];
                if (slot >= 0) {
                    kd = 0;
                    ptr = (unsigned long long) (d.cache_base + (d.cache_slot_off ? (size_t) d.cache_slot_off[slot]
                                                                                 : (size_t) slot * (size_t) d.cache_blob));
                } else if (d.peer != nullptr && d.peer->has(d.layers, e)) {
                    kd = 2;                        // multi-GPU: the second GPU computes it
                } else if (helper_holds(e)) {
                    // left to the helper: kind -1 until RemoteExperts::begin() claims the row (see above)
                } else if (d.fs != nullptr && d.fs->take(d.layers, e, ptr)) {
                    kd = 0;                        // Foresight: a landed swap-space slot holds it - a GPU group like a hit
                } else {
                    if (d.fs != nullptr) d.fs->note_miss(d.src, d.layers, e);
                    if (miss_rank >= nmiss - m && fetches < P.staging_cap && fetches < 64) {
                        const uint8_t* src;
                        {   // DIAGB2: time pinned() and blob() separately (PCIe-share miss only)
                            const auto tp0 = std::chrono::steady_clock::now();
                            const bool p = d.src->pinned(d.layers, e);
                            const auto tp1 = std::chrono::steady_clock::now();
                            src = p ? d.src->blob(d.layers, e) : nullptr;
                            const auto tp2 = std::chrono::steady_clock::now();
                            s_pinned_ns.fetch_add((int64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(tp1 - tp0).count(), std::memory_order_relaxed);
                            s_blob_ns.fetch_add((int64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(tp2 - tp1).count(), std::memory_order_relaxed);
                            s_pinned_calls.fetch_add(1, std::memory_order_relaxed);
                        }
                        if (src != nullptr) {
                            kd = 1;
                            dma_src[fetches] = src;
                            pcie_i0[fetches] = i0;
                            ++fetches;
                        }
                    }
                    ++miss_rank;
                }
            }
            for (int64_t i = i0; i < n; ++i)
                if (first_of[i] == i0) kind[i] = kd;
            if (kd != 0) continue;                 // the VRAM groups first; the PCIe groups below
            P.ptr[groups] = ptr;
            P.start[groups] = entries;
            for (int64_t i = i0; i < n; ++i)
                if (first_of[i] == i0) {
                    P.dst[entries] = (int32_t) i;
                    P.tok[entries] = (int32_t) (i / k);
                    ++entries;
                }
            ++groups;
        }
        t_k1_ms = (double) std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count() / 1e6;   // DIAGB2: kind loop done
        P.start[groups] = entries;
        const uint64_t bb = lay.blob_bytes(d.layers);
        for (int q = 0; q < fetches; ++q) {       // the PCIe groups: staging slot q, entries after the VRAM ones
            const int64_t i0 = pcie_i0[q];
            P.ptr2[q] = P.pcie_mode != 0 ? (unsigned long long) d.src->device_alias(d.layers, ids[i0])
                                 : P.staging + (unsigned long long) q * (unsigned long long) bb;
            P.start2[q] = entries;
            for (int64_t i = i0; i < n; ++i)
                if (first_of[i] == i0) {
                    P.dst[entries] = (int32_t) i;
                    P.tok[entries] = (int32_t) (i / k);
                    ++entries;
                }
            ++d.pcie_experts;
        }
        P.start2[fetches] = entries;
        P.counts[0] = groups;
        P.counts[1] = entries;
        P.counts[2] = fetches;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        pt("publish", fetches);
        const auto pp0 = std::chrono::steady_clock::now();
        if (P.publish) P.publish(P.ctx);
        const auto pp1 = std::chrono::steady_clock::now();
        pt("fetch", fetches);
        if (P.fetch) P.fetch(P.ctx, dma_src, P.pcie_mode != 0 ? 0 : fetches, (size_t) bb);   // the copy engine, beside the CPU's work
        const auto pp2 = std::chrono::steady_clock::now();
        s_publish_ns.fetch_add((int64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(pp1 - pp0).count(), std::memory_order_relaxed);
        s_fetch_ns.fetch_add((int64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(pp2 - pp1).count(), std::memory_order_relaxed);
        s_pub_calls.fetch_add(1, std::memory_order_relaxed);
        t_k2_ms = (double) std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count() / 1e6;   // DIAGB2: publish+fetch done
    } else {
        for (int64_t i = 0; i < n; ++i) {
            const int32_t e = ids[i];
            kind[i] = (e >= 0 && e < d.n_expert && d.host_res != nullptr &&
                       d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e] >= 0) ? 0
                    : (e >= 0 && e < d.n_expert && d.peer != nullptr && d.peer->has(d.layers, e)) ? 2 : -1;
        }
    }
    if (d.peer != nullptr) {                 // multi-GPU: start the second GPU's share before the CPU's own work
        std::string perr;
        if (!d.peer->launch(d.layers, x_f, ids, n_tok, k, kind, perr, out)) {
            std::fprintf(stderr, "strata: %s (layer %lld)\n", perr.c_str(), (long long) d.layers);
            d.failed = true;
            d.fail = "the peer GPU's experts could not be launched";
            d.fail_layer = d.layers;
            return;
        }
    }
    if (d.remote_count > 0) {
        static thread_local std::string remote_error;
        for (int r = 0; r < d.remote_count; ++r) {
            if (!d.remote[r]->begin(d.layers, x_f, ids, n_tok, k, kind, d.host_res, remote_error)) {
                d.failed = true; d.fail = remote_error.c_str(); d.fail_layer = d.layers; return;
            }
            for (int64_t i = 0; i < n; ++i) if (d.remote[r]->owns(i)) kind[i] = 2;
        }
    }
    static const bool dec_batch = [] { const char* v = std::getenv("STRATA_DEC_BATCH"); return v == nullptr || std::atoi(v) != 0; }();
    // the GPU's copy_rows_from_mapped zeroes the rows it computed itself (dec_batch); STRATA_VERIFY_DEVICE_PLAN's
    // copy_or_zero_from_mapped copies every row, so there the host still zeroes them
    static const bool device_plan = [] { const char* v = std::getenv("STRATA_VERIFY_DEVICE_PLAN"); return v != nullptr && std::atoi(v) != 0; }();
    const bool gpu_zeroes_hits = (d.plan != nullptr && n <= kMaxWindowEntries && n <= d.plan->cap && dec_batch && !device_plan);
    bool any_cpu = false;
    for (int64_t i = 0; i < n; ++i)
        if (kind[i] == -1) { any_cpu = true; break; }   // -2: skipped (STRATA_ROUTE_TAIL_SKIP), no work
    if (d.dres_check != nullptr && any_cpu) {   // STRATA_ADAPT_CHECKRES
        bool published = false;
        for (int64_t i = 0; i < n && !published; ++i)
            published = ids[i] < 0 || ids[i] >= d.n_expert ||
                        d.dres_check[(size_t) d.layers * (size_t) d.n_expert + (size_t) ids[i]] < 0;
        if (!published) ++d.stale_x;
    }
    const double t_kb_ms = (double) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count() / 1e6;   // DIAGB: K wall ends
    if (d.latent && any_cpu) {   // latent MoE: the CPU share runs the A-blob experts on the host (D -> A -> U)
        static thread_local std::vector<int64_t> lmiss;
        static thread_local std::vector<uint8_t> lmiss_mark;   // one pass, no std::find scan
        lmiss.clear();
        if (lmiss_mark.size() < (size_t) d.n_expert) lmiss_mark.assign((size_t) d.n_expert, 0);
        for (int64_t i = 0; i < n; ++i)
            if (kind[i] == -1) {
                const int32_t e = ids[i];
                if (e < 0 || e >= d.n_expert || lmiss_mark[(size_t) e]) continue;
                lmiss_mark[(size_t) e] = 1;
                lmiss.push_back(e);
            }
        if (!lmiss.empty()) d.src->prefetch(d.layers, lmiss.data(), (int64_t) lmiss.size());
        for (const int64_t e : lmiss) lmiss_mark[(size_t) e] = 0;   // the scratch resets for the next call
        for (int64_t i = 0; i < n; ++i)   // STRATA_ROUTE_TAIL_SKIP: a skipped entry's row is zero (the merge copies it)
            if (kind[i] == -2) std::memset(out + (size_t) i * (size_t) d.lat_H, 0, (size_t) d.lat_H * sizeof(float));
        const auto l0 = std::chrono::steady_clock::now();
        const double t_bb_ms = (double) std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count() / 1e6;   // DIAGB: B wall ends
        latent_cpu_run(d, x_f, ids, n_tok, k, kind, out);
        const double t_run_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - l0).count();
        d.ms_run += t_run_ms;
        {   // DIAGB2: fine dispatch-bridge walls (ms/call cumulative mean). RUN = full latent_cpu_run wall.
            static std::atomic<double> s_beg{0}, s_dist{0}, s_tailw0{0}, s_kind{0}, s_pcie{0}, s_glue{0}, s_bwall{0}, s_run{0}, s_tailw{0};
            static std::atomic<int64_t> s_bn{0};
            const double t_tb_ms = (double) std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count() / 1e6;
            s_beg.fetch_add(t_c0_ms - t_b0_ms, std::memory_order_relaxed);            // begin_layer + usage
            s_dist.fetch_add(t_d1_ms - t_c0_ms, std::memory_order_relaxed);           // distinct + nmiss
            s_tailw0.fetch_add(t_d2_ms - t_d1_ms, std::memory_order_relaxed);         // tail_skip
            s_kind.fetch_add(t_k1_ms - t_d2_ms, std::memory_order_relaxed);           // kind loop (incl pinned/blob)
            s_pcie.fetch_add(t_k2_ms - t_k1_ms, std::memory_order_relaxed);           // PCIe group fill + publish + fetch
            s_glue.fetch_add(t_kb_ms - t_k2_ms, std::memory_order_relaxed);           // peer/remote/any_cpu/dres_check glue
            s_bwall.fetch_add(t_bb_ms - t_kb_ms, std::memory_order_relaxed);          // lmiss + prefetch + memset(-2)
            s_run.fetch_add(t_run_ms, std::memory_order_relaxed);                     // latent_cpu_run full wall
            s_tailw.fetch_add(t_tb_ms - t_bb_ms - t_run_ms, std::memory_order_relaxed); // tail bookkeeping
            const int64_t bn = s_bn.fetch_add(1, std::memory_order_relaxed);
            if (std::getenv("STRATA_DIAG_A") != nullptr && bn % 48 == 0)
                std::fprintf(stderr,
                    "DIAGB2 ms/call: beg %.4f dist %.4f tail %.4f kind %.4f pcie %.4f glue %.4f B %.4f RUN %.4f tailw %.4f | pub %.4f fetch %.4f pinned %.4f blob %.4f | pub_c %.2f pin_c %.2f (n %lld)\n",
                    s_beg.load() / (double) (bn + 1), s_dist.load() / (double) (bn + 1),
                    s_tailw0.load() / (double) (bn + 1), s_kind.load() / (double) (bn + 1),
                    s_pcie.load() / (double) (bn + 1), s_glue.load() / (double) (bn + 1),
                    s_bwall.load() / (double) (bn + 1), s_run.load() / (double) (bn + 1),
                    s_tailw.load() / (double) (bn + 1),
                    (double) s_publish_ns.load() / 1e6 / (double) (bn + 1),
                    (double) s_fetch_ns.load() / 1e6 / (double) (bn + 1),
                    (double) s_pinned_ns.load() / 1e6 / (double) (bn + 1),
                    (double) s_blob_ns.load() / 1e6 / (double) (bn + 1),
                    (double) s_pub_calls.load() / (double) (bn + 1),
                    (double) s_pinned_calls.load() / (double) (bn + 1),
                    (long long) (bn + 1));
        }
        if (d.plan != nullptr) d.plan->cpu_jobs = (int) lmiss.size();
        d.multi_misses += (int64_t) lmiss.size();
        for (int64_t i = 0; i < n; ++i) {
            const int64_t e = ids[i];
            if (e >= 0 && e < d.n_expert) d.job_of[(size_t) e] = -1;
        }
        ++d.layers;
        d.experts += n_tok * k;
        return;
    }
    const auto c1 = std::chrono::steady_clock::now();
    if (any_cpu) {
        // #578 --remote-expert-opt: a token whose experts all run on a GPU (CUDA0 or a helper) needs no CPU activation
        const bool ep = d.remote_count > 0 && d.remote[0]->optimized_decode();
        for (int64_t t = 0; t < n_tok; ++t) {
            if (ep && std::all_of(kind + t * k, kind + (t + 1) * k, [](int32_t v) { return v >= 0; })) continue;
            if (native && strata::kernels::cpu::q2_native_kernels(lay.fmt[(size_t) d.layers].gu_type))   // a native Q2_0 pack: the Q2_0 kernels' activations
                act_quant_any(x_f + (size_t) t * H, H, d.act_multi[(size_t) t]);
            else if (native)
                native_quant_act(lay.fmt[(size_t) d.layers], x_f + (size_t) t * H, d.nact_multi.data() + (size_t) t * kNativeActBytes);
            else
                act_quant_q8_1(x_f + (size_t) t * H, H, d.act_multi[(size_t) t]);
        }
    }
    const auto c2 = std::chrono::steady_clock::now();
    // 149-cold STRATA_COLD_OVERLAP (d.cold_mode): which of the CPU's experts still have a blob to read.  Mode 1 reads
    // them in the background and computes the ready ones first, then each read one as it lands; every expert keeps
    // its own token rows (the pool splits rows, never a row's sum), and the weighted sum over a token's experts is
    // the GPU's, after the dispatch - so the result is the same bits in any batch order.  The stage buffers stay
    // claimed (busy, then young) until the dispatch has joined its reads and run every batch.
    static thread_local std::vector<int64_t> miss;
    static thread_local std::vector<uint8_t> mstate;   // per miss: 0 ready, 1 being read, 2 landed, 3 computed
    miss.clear();
    int64_t npend = 0;
    double cold_r_ms = 0;
    if (any_cpu) {   // CS-T: the experts the CPU computes, fetched together (the GGUF in place reads them on several threads)
        for (int64_t i = 0; i < n_tok * k; ++i)
            if (kind[i] == -1 && ids[i] >= 0 && ids[i] < d.n_expert &&
                std::find(miss.begin(), miss.end(), (int64_t) ids[i]) == miss.end())
                miss.push_back(ids[i]);
        if (d.cold_mode == 0) {
            d.src->prefetch(d.layers, miss.data(), (int64_t) miss.size());
        } else {
            mstate.assign(miss.size(), 0);
            const auto r0 = std::chrono::steady_clock::now();
            npend = d.src->prefetch_begin(d.layers, miss.data(), (int64_t) miss.size(), mstate.data());
            if (npend > 0 && d.cold_mode == 2) {   // measure: every read first, as mode 0 does
                d.src->prefetch_end();
                for (uint8_t& s : mstate) if (s == 1) s = 2;
                cold_r_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - r0).count();
                d.ms_cold_wait += cold_r_ms;
            }
        }
    }
    // the CPU's experts this batch takes: all (mode 0, or no read), else by their state (want 0: the ready ones,
    // want 2: those that landed since the last batch)
    auto take = [&](int64_t e, uint8_t want) {
        if (npend == 0) return true;
        for (size_t q = 0; q < miss.size(); ++q)
            if (miss[q] == e) return mstate[q] == want;
        return want == 0;
    };
    int njobs = 0, njobs_all = 0;
    bool first_pass = true;
    uint8_t want = 0;
    double run_ms_extra = 0, cr_ms = 0;
    for (;;) {
    for (int64_t t = 0; t < n_tok; ++t)
        for (int64_t j = 0; j < k; ++j) {
            const int64_t i = t * k + j;
            const int64_t e = ids[i];
            float* row = out + (size_t) i * H;
            if (!first_pass && (e < 0 || e >= d.n_expert || kind[i] != -1 || !take(e, want)))
                continue;   // a later batch: only the CPU entries of the experts that just landed
            if (first_pass && kind[i] == -1 && e >= 0 && e < d.n_expert && !take(e, 0)) continue;   // being read: a later batch
            if (e < 0 || e >= d.n_expert) {
                d.failed = true;
                d.fail = "a routed expert id is out of range";
                d.fail_layer = d.layers;
                d.fail_expert = e;
                return;
            }
            if (kind[i] == -2) {            // STRATA_ROUTE_TAIL_SKIP: a skipped tail expert contributes zero
                std::memset(row, 0, (size_t) H * sizeof(float));
                ++d.tail_skipped;
                continue;
            }
            if (kind[i] >= 0) {             // CUDA0, PCIe, or a remote/peer result staged into this row below
                if (kind[i] == 0) ++d.cache_hits;
                else ++d.offload_entries;                       // #588: PCIe or another GPU
                if (kind[i] == 2 && d.peer != nullptr) ++d.peer_entries;
                // multi-GPU: a direct peer launch is writing this row right now - zeroing it would race it;
                // dec_batch: copy_rows_from_mapped_kernel already zeroes kind 0/1 rows on the GPU
                if (!((kind[i] == 2 && d.peer != nullptr && d.peer->launched_direct()) ||
                      (gpu_zeroes_hits && (kind[i] == 0 || kind[i] == 1))))
                    std::memset(row, 0, (size_t) H * sizeof(float));
                continue;
            }
            ++d.cache_refused;
            int16_t& jo = d.job_of[(size_t) e];
            if (jo < 0) {
                const uint8_t* zp[3] = {nullptr, nullptr, nullptr};
                const bool zc = d.cold_mode >= 3 && native && d.src->zc_roles(d.layers, e, zp);
                const uint8_t* b = zc ? zp[0] : d.src->blob(d.layers, e);
                if (b == nullptr) {
                    d.failed = true;
                    d.fail = "the expert source could not produce a blob";
                    d.fail_layer = d.layers;
                    d.fail_expert = e;
                    ++d.missing;
                    return;
                }
                jo = (int16_t) njobs++;
                ExpertJobMulti& nj = d.jobs_multi[(size_t) jo];
                nj.blob = b;
                nj.nt = 0;
                nj.fmt = nullptr;
                if (zc) {   // 149-cold mode 3: the gate rows are the blob; up and down at their distance in the mapping
                    if (d.zc_fmt.size() < d.jobs_multi.size()) d.zc_fmt.resize(d.jobs_multi.size());
                    strata::kernels::cpu::NativeFmt& f = d.zc_fmt[(size_t) jo];
                    f = lay.fmt[(size_t) d.layers];
                    f.up_off = (size_t) ((uintptr_t) zp[1] - (uintptr_t) zp[0]);     // modular: the tensors' order
                    f.down_off = (size_t) ((uintptr_t) zp[2] - (uintptr_t) zp[0]);   // in the file does not matter
                    nj.fmt = &f;
                    ++d.cold_zc;
                }
            }
            if (d.jobs_multi[(size_t) jo].nt == MAXT) {
                // a window of 9..16 rows routes more than MAXT tokens to this expert: the kernels and the pool's scratch
                // take MAXT tokens a job, so the rest are a further job over the same blob (each token's rows are its own,
                // so the split changes no value; the blob is read once more)
                const int16_t prev = jo;
                jo = (int16_t) njobs++;
                ExpertJobMulti& nj = d.jobs_multi[(size_t) jo];
                nj.blob = d.jobs_multi[(size_t) prev].blob;
                nj.fmt = d.jobs_multi[(size_t) prev].fmt;
                nj.nt = 0;
            }
            ExpertJobMulti& jb = d.jobs_multi[(size_t) jo];
            jb.act[jb.nt] = &d.act_multi[(size_t) t];
            jb.nact[jb.nt] = native ? d.nact_multi.data() + (size_t) t * kNativeActBytes : nullptr;
            jb.out[jb.nt] = row;
            ++jb.nt;
            ++d.multi_entries;
        }
    if (npend == 0) break;
    // 149-cold: run this batch now (the ready experts, then each landed group), then wait for the next landing
    {
        const auto b0 = std::chrono::steady_clock::now();
        if (njobs > 0) {
            if (native) d.pool->run_split_multi_native(lay.fmt[(size_t) d.layers], d.jobs_multi.data(), njobs, arithmetic);
            else d.pool->run_split_multi(d.jobs_multi.data(), njobs);
            ++d.cold_batches;
        }
        const auto b1 = std::chrono::steady_clock::now();
        const double bm = std::chrono::duration<double, std::milli>(b1 - b0).count();
        run_ms_extra += bm;
        if (first_pass) { d.ms_cold_cr += bm; cr_ms = bm; }
        else d.ms_cold_cc += bm;
        if (first_pass) d.cold_ready += njobs;
        // the computed ones' job slots are free again (an expert is in one batch only)
        for (int jx = 0; jx < njobs; ++jx) d.jobs_multi[(size_t) jx].nt = 0;
        for (int64_t i = 0; i < n_tok * k; ++i)
            if (ids[i] >= 0 && ids[i] < d.n_expert) d.job_of[(size_t) ids[i]] = -1;
        if (!first_pass) for (uint8_t& s : mstate) if (s == 2) s = 3;
        njobs_all += njobs;
        njobs = 0;
        first_pass = false;
        want = 2;
        bool left = false;
        for (uint8_t s : mstate) left = left || s == 1 || s == 2;
        if (!left) break;
        if (std::none_of(mstate.begin(), mstate.end(), [](uint8_t s) { return s == 2; })) {
            const auto w0 = std::chrono::steady_clock::now();
            (void) d.src->wait_landed(d.layers, miss.data(), (int64_t) miss.size(), mstate.data());
            d.ms_cold_wait += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
        }
    }
    }
    if (npend > 0) {
        d.src->prefetch_end();
        const FileExpertSource* fs = dynamic_cast<const FileExpertSource*>(d.src);
        const double r_ms = d.cold_mode == 2 ? cold_r_ms : (fs != nullptr ? fs->async_fill_us() / 1e3 : 0.0);
        ++d.cold_layers;
        d.cold_pending += npend;
        d.ms_cold_r += r_ms;
        d.ms_cold_hide += std::min(r_ms, cr_ms);   // what an overlap can hide of this layer's reads at most
        njobs = njobs_all;
        d.ms_run += run_ms_extra;    // the batches ran inside the job loop: the pool's time stays the pool's
        d.ms_jobs -= run_ms_extra;
    }
    const auto c3 = std::chrono::steady_clock::now();
    if (d.plan != nullptr) d.plan->cpu_jobs = njobs;
    pt("run", njobs);
    if (njobs > 0 && npend == 0) {
        if (native) d.pool->run_split_multi_native(lay.fmt[(size_t) d.layers], d.jobs_multi.data(), njobs, arithmetic);
        else d.pool->run_split_multi(d.jobs_multi.data(), njobs);
    }
    if (d.remote_count > 0) {
        static thread_local std::string remote_error;
        for (int r = 0; r < d.remote_count; ++r)
            if (!d.remote[r]->finish(out, remote_error)) {
                d.failed = true; d.fail = remote_error.c_str(); d.fail_layer = d.layers; return;
            }
    }
    if (d.peer != nullptr) {                 // multi-GPU: the second GPU's rows, into the same mapped rows
        std::string perr;
        if (!d.peer->finish(out, perr)) {
            std::fprintf(stderr, "strata: %s (layer %lld)\n", perr.c_str(), (long long) d.layers);
            d.failed = true;
            d.fail = "the peer GPU's experts failed";
            d.fail_layer = d.layers;
            return;
        }
    }
    const auto c4 = std::chrono::steady_clock::now();
    pt("ran");
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    d.ms_plan += ms(c0, c1);
    d.ms_actq += ms(c1, c2);
    d.ms_jobs += ms(c2, c3);
    d.ms_run += ms(c3, c4);
    for (int64_t i = 0; i < n_tok * k; ++i) {
        const int64_t e = ids[i];
        if (e >= 0 && e < d.n_expert) d.job_of[(size_t) e] = -1;
    }
    d.multi_misses += njobs;
    ++d.layers;
    d.experts += n_tok * k;
}

void expert_hit_run(void* user, void* stream, HitPhase phase, const int32_t* ids, int64_t k) {
    ExpertDispatch& d = *(ExpertDispatch*) user;
    if (d.failed) return;
    cudaStream_t cs = (cudaStream_t) stream;

    if (phase == HitPhase::Launch) {
        d.decided = false;
        d.hit_pending = false;
        if (!d.hits_ready() || ids == nullptr || k <= 0) return;
        if ((int64_t) d.is_hit.size() < k) d.is_hit.resize((size_t) k);
        // STRATA_EXPERT_NIBBLE=1: this split computes a resident expert with the GGUF-block kernels and admits + fills
        // a missed one with its GGUF blob - neither may meet a layer whose slots hold 4-bit codes (a native pack's
        // experts run in the verify windows, so this is the backstop, never the path)
#if STRATA_EXPERT_NIBBLE_BUILT   // compiled out of every other build: nothing is added to the hit split there
        if (strata::core::nibble::layer(d.layers).on)
            strata::core::nibble::fatal(d.layers, "the expert source's hit split (expert_hit_run)",
                                        "it computes resident experts as GGUF blocks and fills slots with GGUF blobs");
#endif

        // ================================ THE DECISION, ONCE, ON THIS LAYER'S IDS ================================
        //
        // Every routed expert is asked of the cache.  Resident -> the GPU computes it.  Not resident -> it is
        // admitted and filled if there is room (which makes it a hit on THIS call, because the fill and the
        // kernel are on one stream in that order), and otherwise it stays a miss for the CPU.
        d.n_hits = 0;
        for (int64_t i = 0; i < k; ++i) {
            const int64_t e = ids[i];
            d.is_hit[(size_t) i] = 0;
            if (e < 0 || e >= d.n_expert) continue;   // out of range: the pool refuses it, with a message
            int32_t slot = d.cache->slot_of(d.layers, e);
            if (slot == kNotResident) {
                const int32_t cand = d.cache->admit(d.layers, e);
                if (cand == kNotResident) {
                    ++d.cache_refused;
                    continue;
                }
                // `blob` is asked ONLY for an expert about to be filled, so the source's read counter stays a
                // count of distinct experts moved rather than of looks.
                const uint8_t* b = d.src->blob(d.layers, e);
                std::string ferr;
                if (b == nullptr || !d.cache->fill_slot(cand, b, cs, ferr, (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(d.layers))) {
                    d.failed = true;
                    d.fail = "the expert cache could not fill a slot";
                    d.fail_layer = d.layers;
                    d.fail_expert = e;
                    return;
                }
                d.src->note_async_read(b, cs);   // #1237: a page-locked stage buffer is not recycled under the queued copy
                ++d.cache_admitted;
                slot = cand;
            } else {
                ++d.cache_hits;
            }
            d.is_hit[(size_t) i] = 1;
            d.h_slot[(size_t) d.n_hits] = slot;
            d.h_dst[(size_t) d.n_hits] = (int32_t) i;
            ++d.n_hits;
        }
        d.decided = true;
        if (d.n_hits <= 0) return;   // nothing resident yet: no GPU work, and nothing for `Combine` to add

        const size_t list_bytes = (size_t) d.n_hits * sizeof(int32_t);
        // `hit_out` is ZEROED rather than overwritten: the kernel writes only the rows this layer's hits own,
        // so a row that was a hit last layer and a miss this one would still hold last layer's expert and
        // `add_inplace` would sum it in.  Finite, plausible, wrong.
        if (cudaMemsetAsync(d.hit_out, 0, (size_t) d.parts_elems * sizeof(float), cs) != cudaSuccess ||
            cudaMemcpyAsync(d.d_slot, d.h_slot.data(), list_bytes, cudaMemcpyHostToDevice, cs) != cudaSuccess ||
            cudaMemcpyAsync(d.d_dst, d.h_dst.data(), list_bytes, cudaMemcpyHostToDevice, cs) != cudaSuccess) {
            d.hit_fail = "the hit list could not be staged";
            d.failed = true;
            d.fail = d.hit_fail;
            return;
        }
        // The activation is quantized HERE rather than reused from `s.moe.x_q8_0`, which `post[l-1]` wrote from
        // the PREVIOUS layer's `mixed`.  `pre[l]` has since overwritten `mixed`, so that buffer is a layer stale
        // - and a stale activation produces a perfectly finite expert for the wrong input.
        // **R4.2h: THE SCALED QUANTIZER, SO A HIT REPRODUCES A MISS.**  The CPU pool quantizes this same
        // activation with `act_quant_q8_1` and multiplies by the fp32 `ActQ::scale`; `quantize_q8_0` writes
        // an fp16 `d` instead, and `bench/micro/act_quant_parity.cu` measured **80 of 80 chunks differing by
        // up to 4.761e-04 relative**.  `quantize_q8_0_scaled` adopts the CPU's rule and scale, and the kernel
        // takes the fp32 array.  Falling back to the old path would silently reintroduce the divergence, so
        // the scales are required here rather than optional.
        if (d.x_q8_0_hit_scale == nullptr) {
            d.failed = true;
            d.fail = "the hit path has no fp32 activation scales (R4.2h)";
            return;
        }
        strata::kernels::quantize_q8_0_scaled(d.mixed, d.x_q8_0_hit, d.x_q8_0_hit_scale, strata::kernels::cpu::H,
                                              cs);
        if (d.hit_cpu_order)
            strata::kernels::moe_hit_grouped_s2_cpu_order(d.cache_base, d.d_slot, d.d_dst, d.n_hits,
                d.cache_blob, d.x_q8_0_hit, d.hit_scratch, d.hit_out, cs, d.x_q8_0_hit_scale);
        else
            strata::kernels::moe_hit_grouped_s2(d.cache_base, d.d_slot, d.d_dst, d.n_hits, d.cache_blob,
                d.x_q8_0_hit, d.hit_scratch, d.hit_out, cs, d.x_q8_0_hit_scale);
        d.hit_pending = true;
        if (d.hit_done != nullptr) cudaEventRecord((cudaEvent_t) d.hit_done, cs);
        // The A/B arm: ONE driver entry here, and nothing else changes.  If the work was waiting for the host
        // to enter the driver, this is what lets it start while the pool runs.
        if (d.hit_poke && d.hit_done != nullptr) (void) cudaEventQuery((cudaEvent_t) d.hit_done);
        return;
    }

    // Combine: `parts += hit_out`, stream-ordered after the misses were copied into `parts`.
    if (!d.hit_pending) return;
    d.hit_pending = false;
    // Did the GPU get the hit work done while the CPU was in the pool?  This query is itself a driver entry,
    // so it is the LAST chance to observe a late start: a NOT-READY here means the work had not finished by the
    // time the pool returned, and with no poke in front of it that can only be because it began after.
    if (d.hit_done != nullptr) {
        if (cudaEventQuery((cudaEvent_t) d.hit_done) == cudaSuccess) ++d.hit_ready;
        else ++d.hit_late;
    }
    strata::kernels::add_inplace(d.parts_out, d.hit_out, d.parts_elems, cs);
}

// ================================ THE RESIDENT ARENA (R2.1) ================================

namespace {

void hash_u64(uint64_t& h, uint64_t v) {
    h = fnv1a64((const uint8_t*) &v, sizeof v, h);
}

void hash_text(uint64_t& h, const std::string& s) {
    h = fnv1a64((const uint8_t*) s.data(), (uint64_t) s.size(), h);
}

bool hash_small_file(const std::filesystem::path& path, uint64_t& h, std::string& err) {
    hash_text(h, path.filename().string());
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        hash_u64(h, 0);
        return true;
    }
    hash_u64(h, 1);
    std::vector<uint8_t> buf(64u << 10);
    for (;;) {
        f.read((char*) buf.data(), (std::streamsize) buf.size());
        const std::streamsize n = f.gcount();
        if (n > 0) h = fnv1a64(buf.data(), (uint64_t) n, h);
        if (f.eof()) break;
        if (!f) {
            err = "ArenaExpertSource: cannot hash pack metadata " + path.string();
            return false;
        }
    }
    return true;
}

bool hash_sampled_file(const std::filesystem::path& path, uint64_t& h, std::string& err) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        err = "ArenaExpertSource: cannot sample pack source " + path.string();
        return false;
    }
    const std::streamoff end = f.tellg();
    if (end < 0) {
        err = "ArenaExpertSource: cannot size pack source " + path.string();
        return false;
    }
    const uint64_t bytes = (uint64_t) end;
    hash_text(h, path.filename().string());
    hash_u64(h, bytes);
    constexpr uint64_t sample = 64u << 10;
    const uint64_t starts[3] = {0, bytes / 2, bytes > sample ? bytes - sample : 0};
    std::vector<uint8_t> buf((size_t) std::min<uint64_t>(sample, bytes));
    for (uint64_t off : starts) {
        if (buf.empty()) break;
        const uint64_t at = std::min<uint64_t>(off, bytes - (uint64_t) buf.size());
        f.clear();
        f.seekg((std::streamoff) at);
        f.read((char*) buf.data(), (std::streamsize) buf.size());
        if ((size_t) f.gcount() != buf.size()) {
            err = "ArenaExpertSource: short read while hashing pack source " + path.string();
            return false;
        }
        hash_u64(h, at);
        h = fnv1a64(buf.data(), (uint64_t) buf.size(), h);
    }
    return true;
}

bool shared_arena_pack_hash(const std::string& pack_dir, const std::string& experts_path,
                            const std::string& gguf, const strata::kernels::cpu::ExpertLayout& lay,
                            uint64_t& out, std::string& err) {
    uint64_t h = 1469598103934665603ull;
    hash_text(h, "strata-shared-expert-arena-pack-v1");
    hash_u64(h, (uint64_t) lay.n_layers);
    hash_u64(h, (uint64_t) lay.n_expert);
    hash_u64(h, lay.total);
    hash_u64(h, lay.max_blob);
    hash_u64(h, lay.native ? 1 : 0);

    const std::filesystem::path pack(pack_dir);
    for (const char* name : {"manifest.json", "index.txt", "native_experts.txt"}) {
        if (!hash_small_file(pack / name, h, err)) return false;
    }

    if (std::filesystem::exists(experts_path)) {
        if (!hash_sampled_file(experts_path, h, err)) return false;
    } else if (!gguf.empty()) {
        // Native packs may read experts straight from one or more GGUF shards.  Sample every distinct source
        // file named by native_experts.txt; this keeps the fingerprint cheap while still tying it to the model
        // bytes rather than only to an equal-size layout.
        const std::filesystem::path first(gguf);
        std::vector<std::filesystem::path> sources{first};
        for (const std::string& name : lay.gguf_file) {
            if (name.empty()) continue;
            const std::filesystem::path p = first.parent_path() / name;
            if (std::find(sources.begin(), sources.end(), p) == sources.end()) sources.push_back(p);
        }
        for (const auto& p : sources) {
            if (!hash_sampled_file(p, h, err)) return false;
        }
    }

    out = h == 0 ? 1 : h;
    return true;
}

}  // namespace

namespace {
/// The GGUF file that holds role `r` (0 gate, 1 up, 2 down) of layer `l`: a name beside the --native shard
/// (native_experts.txt v3 per layer, v4 per role), or the --native shard itself.
std::string expert_gguf_file(const std::string& gguf, const strata::kernels::cpu::ExpertLayout& lay, int64_t l, int r) {
    const size_t i = (size_t) (3 * l + r);
    if (lay.gguf_file.size() <= i || lay.gguf_file[i].empty()) return gguf;
    const size_t cut = gguf.find_last_of("/\\");
    return (cut == std::string::npos ? std::string() : gguf.substr(0, cut + 1)) + lay.gguf_file[i];
}
}  // namespace

void ExpertSource::record_inputs(const std::string& pack_experts, const std::string& gguf,
                                 const strata::kernels::cpu::ExpertLayout& lay, bool from_gguf) {
    static const char* roles[3] = {"gate", "up", "down"};
    inputs_.clear();
    if (!from_gguf) { inputs_.push_back({"pack experts.bin", pack_experts}); return; }
    // the same resolution open_gguf / load_experts_gguf use: every (layer, role) tensor and the file it is read from
    for (int64_t l = 0; l < lay.n_layers; ++l)
        for (int r = 0; r < 3; ++r)
            inputs_.push_back({"expert blk." + std::to_string(l) + ".ffn_" + roles[r], expert_gguf_file(gguf, lay, l, r)});
}

bool check_experts_gguf(const std::string& gguf, const strata::kernels::cpu::ExpertLayout& lay, std::string& err) {
    static const char* roles[3] = {"gate", "up", "down"};
    if (lay.gguf_off.size() != (size_t) (3 * lay.n_layers)) {
        err = "native_experts.txt has no GGUF offsets (a pack older than v2): repack it with tools/iq_pack.py";
        return false;
    }
    try {
        std::map<std::string, std::unique_ptr<strata::GgufFile>> files;
        for (int64_t l = 0; l < lay.n_layers; ++l) {
            const auto& fm = lay.fmt[(size_t) l];
            const uint64_t blob = lay.bytes[(size_t) l];
            const uint64_t per[3] = {fm.up_off, fm.up_off, blob - fm.down_off};
            for (int r = 0; r < 3; ++r) {
                const std::string path = expert_gguf_file(gguf, lay, l, r);
                auto& f = files[path];
                if (!f) f = std::make_unique<strata::GgufFile>(path);
                const std::string name = "blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight";
                const strata::TensorInfo* t = f->find(name);
                const uint64_t want_type = (uint64_t) (r < 2 ? fm.gu_type : fm.d_type);
                // GGUF order: dim 0 is the row (the input), dim 1 the rows, dim 2 the experts
                const uint64_t cols = (uint64_t) (r < 2 ? fm.n_embd : fm.n_ff);
                const uint64_t rows = (uint64_t) (r < 2 ? fm.n_ff : fm.n_embd);
                const uint64_t bytes = per[r] * (uint64_t) lay.n_expert;
                const uint64_t payload = f->file_size() - f->data_start();
                std::string why;
                if (t == nullptr) why = "is not in it";
                else if (t->type != want_type)
                    why = std::string("is ") + t->type_name() + ", the pack says type " + std::to_string(want_type);
                else if (t->shape.size() != 3 || t->shape[0] != cols || t->shape[1] != rows ||
                         t->shape[2] != (uint64_t) lay.n_expert)
                    why = "is not [" + std::to_string(cols) + ", " + std::to_string(rows) + ", " +
                          std::to_string(lay.n_expert) + "]";
                else if (strata::tensor_payload_bytes(*t) != bytes)
                    why = "is not " + std::to_string(per[r]) + " B per expert";
                else if (f->data_start() + t->offset != lay.gguf_off[(size_t) (3 * l + r)])
                    why = "starts at byte " + std::to_string(f->data_start() + t->offset) + ", the pack says " +
                          std::to_string(lay.gguf_off[(size_t) (3 * l + r)]);
                else if (t->offset > payload || bytes > payload - t->offset)
                    why = "runs past the end of the file (a truncated shard?)";
                if (!why.empty()) {
                    err = "the pack's native_experts.txt does not match the model: " + name + " in " + path + " " +
                          why + " - repack with tools/iq_pack.py from this model's shards";
                    return false;
                }
            }
        }
        return true;
    } catch (const std::exception& e) {
        err = std::string("native experts from the GGUF: ") + e.what();
        return false;
    }
}

// Plan v0.3 P6: the arena from the model's GGUF shards.  Each layer's gate, up and down tensors hold the 512
// experts one after another; they are read in chunks and each expert's slice lands at its place in the blob
// [gate rows | up rows | down rows] - the layout tools/iq_pack.py would have written to experts.bin.  Each role
// is read from its own file (native_experts.txt v4: a shard boundary can fall inside a layer; per role as in
// #255, gopinath87607).  The caller checks the spans first (check_experts_gguf).
// `unbuffered` (Windows, experts_unbuffered): each chunk's 4 KiB-aligned window is read with FILE_FLAG_NO_BUFFERING into
// an aligned buffer and scattered into the blobs - no copy through the file cache when the drive is read anyway.
LoadStats load_experts_gguf(const std::string& gguf, uint8_t* dst, const strata::kernels::cpu::ExpertLayout& lay,
                            int threads, bool unbuffered, const std::atomic<int>* ready, const PadMap* pad) {
    // where blob (l, e) goes in `dst`: packed (experts.bin) or the padded layout of pin by need
    auto dst_blob = [&lay, pad](int64_t l, int64_t e) -> uint64_t {
        return pad != nullptr ? pad->blob_off(l, e) : lay.blob_offset(l, e);
    };
    // `ready`: layer l is written only once *ready > l + 1 (a PinnedArena registering its slices meanwhile; the next
    // slice too, since its registration starts on the page that may hold this layer's tail)
    auto wait_ready = [ready](int64_t l) {
        if (ready != nullptr)
            while (ready->load(std::memory_order_acquire) <= l + 1) std::this_thread::yield();
    };
    LoadStats st;
    st.layers = (uint64_t) lay.n_layers;
    const auto t0 = std::chrono::steady_clock::now();
    std::atomic<int64_t> next{0};
    std::atomic<bool> bad{false};
#if defined(_WIN32)
    if (unbuffered) {
        uint64_t max_chunk = 0;
        for (int64_t l = 0; l < lay.n_layers; ++l) {
            const auto& fm = lay.fmt[(size_t) l];
            max_chunk = std::max<uint64_t>(max_chunk, std::max<uint64_t>(fm.up_off, lay.bytes[(size_t) l] - fm.down_off) * 16);
        }
        std::mutex err_mu;
        std::string err;
        auto worker = [&]() {
            constexpr uint64_t kSector = 4096;
            const uint64_t cap = (max_chunk + 2 * kSector + kSector - 1) / kSector * kSector;
            uint8_t* buf = (uint8_t*) VirtualAlloc(nullptr, (size_t) cap, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            HANDLE h = INVALID_HANDLE_VALUE;
            std::string open_name;
            auto fail = [&](const std::string& what) {
                std::lock_guard<std::mutex> g(err_mu);
                if (err.empty()) err = what;
                bad = true;
            };
            if (buf == nullptr) fail("cannot allocate a read buffer");
            for (;;) {
                const int64_t l = next.fetch_add(1);
                if (l >= lay.n_layers || bad) break;
                wait_ready(l);
                const auto& fm = lay.fmt[(size_t) l];
                const uint64_t blob = lay.bytes[(size_t) l];
                const uint64_t per[3] = {fm.up_off, fm.up_off, blob - fm.down_off};
                const uint64_t at[3] = {0, fm.up_off, fm.down_off};
                for (int r = 0; r < 3 && !bad; ++r) {
                    // the handle is kept while consecutive roles share a file (every layer of a v3 pack)
                    const std::string name = expert_gguf_file(gguf, lay, l, r);
                    if (name != open_name) {
                        if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
                        const int wide = MultiByteToWideChar(CP_UTF8, 0, name.c_str(), -1, nullptr, 0);
                        std::vector<wchar_t> w((size_t) std::max(wide, 1), L'\0');
                        if (wide > 0) MultiByteToWideChar(CP_UTF8, 0, name.c_str(), -1, w.data(), wide);
                        h = CreateFileW(w.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                        FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
                        if (h == INVALID_HANDLE_VALUE) {
                            fail("cannot open " + name + " (error " + std::to_string((unsigned long long) GetLastError()) + ")");
                            break;
                        }
                        open_name = name;
                    }
                    const uint64_t src = lay.gguf_off[(size_t) (3 * l + r)];
                    const uint64_t total = per[r] * (uint64_t) lay.n_expert;
                    const uint64_t chunk = per[r] * 16;           // 16 experts per read
                    for (uint64_t done = 0; done < total; done += chunk) {
                        const uint64_t n = std::min<uint64_t>(chunk, total - done);
                        const uint64_t a0 = (src + done) / kSector * kSector;
                        const uint64_t a1 = (src + done + n + kSector - 1) / kSector * kSector;
                        OVERLAPPED ov{};
                        ov.Offset = (DWORD) a0;
                        ov.OffsetHigh = (DWORD) (a0 >> 32);
                        DWORD got = 0;
                        // the window may run past the end of the file: only the tensor's own bytes have to arrive
                        if (!ReadFile(h, buf, (DWORD) (a1 - a0), &got, &ov) || (uint64_t) got < src + done - a0 + n) {
                            fail("short unbuffered read of layer " + std::to_string(l) + " in " + name + " (error " +
                                 std::to_string((unsigned long long) GetLastError()) + ")");
                            break;
                        }
                        const uint8_t* q = buf + (src + done - a0);
                        for (uint64_t k = 0; k < n / per[r]; ++k) {
                            const uint64_t e = done / per[r] + k;
                            std::memcpy(dst + dst_blob(l, (int64_t) e) + at[r], q + k * per[r], (size_t) per[r]);
                        }
                    }
                }
            }
            if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
            if (buf != nullptr) VirtualFree(buf, 0, MEM_RELEASE);
        };
        std::vector<std::thread> pool;
        for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
        worker();
        for (auto& t : pool) t.join();
        if (bad) {
            st.seconds = -1.0;
            st.ok = false;
            st.error = err.empty() ? "unreadable shard while reading the experts from the GGUF" : err;
            return st;
        }
        st.bytes = lay.total;
        st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return st;
    }
#else
    (void) unbuffered;
#endif
    auto worker = [&]() {
        // #230: `fread` on a `FILE*`, as load_experts_ranges (#89): MSVC's `std::ifstream::read` splits a request
        // into 4095-byte freads, which took this path to 0.02 GiB/s on a Windows install without experts.bin.
        // The guard closes the handle on every return.
        struct Closer {
            FILE* f = nullptr;
            ~Closer() { if (f != nullptr) std::fclose(f); }
        } file;
        std::string open_name;
        std::vector<uint8_t> buf;
        for (;;) {
            const int64_t l = next.fetch_add(1);
            if (l >= lay.n_layers || bad) break;
            wait_ready(l);
            const auto& fm = lay.fmt[(size_t) l];
            const uint64_t blob = lay.bytes[(size_t) l];
            const uint64_t per[3] = {fm.up_off, fm.up_off, blob - fm.down_off};
            const uint64_t at[3] = {0, fm.up_off, fm.down_off};
            for (int r = 0; r < 3; ++r) {
                // the handle is kept while consecutive roles share a file (every layer of a v3 pack)
                const std::string name = expert_gguf_file(gguf, lay, l, r);
                if (name != open_name) {
                    if (file.f != nullptr) std::fclose(file.f);
                    file.f = std::fopen(name.c_str(), "rb");
                    if (file.f == nullptr) { bad = true; return; }
                    open_name = name;
                }
                FILE* const f = file.f;
                const uint64_t src = lay.gguf_off[(size_t) (3 * l + r)];
                const uint64_t total = per[r] * (uint64_t) lay.n_expert;
                const uint64_t chunk = per[r] * 16;           // 16 experts per read
                buf.resize((size_t) chunk);
                for (uint64_t done = 0; done < total; done += chunk) {
                    const uint64_t n = std::min<uint64_t>(chunk, total - done);
                    // 64-bit seek: a shard is tens of GB
                    if (STRATA_FSEEK64(f, src + done) != 0) { bad = true; return; }
                    if (std::fread(buf.data(), 1, (size_t) n, f) != (size_t) n) { bad = true; return; }
                    for (uint64_t k = 0; k < n / per[r]; ++k) {
                        const uint64_t e = done / per[r] + k;
                        std::memcpy(dst + dst_blob(l, (int64_t) e) + at[r], buf.data() + k * per[r], (size_t) per[r]);
                    }
                }
            }
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();
    if (bad) {
        st.seconds = -1.0;
        st.ok = false;
        st.error = "short read or unreadable shard while reading the experts from the GGUF";
        return st;
    }
    st.bytes = lay.total;
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return st;
}

ArenaExpertSource::~ArenaExpertSource() { close(); }

bool ArenaExpertSource::open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, int threads,
                             std::string& err, uint64_t max_pinned_bytes,
                             const std::string& shared_arena_file) {
    close();
    const std::string path = pack_dir + "/experts.bin";
    // plan v0.3 P6: the layout (canonical, or a native pack's per-layer blobs) was loaded by the driver
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    if (lay.n_layers != n_layers || lay.n_expert != n_expert) {
        err = "ArenaExpertSource: the expert layout was loaded for a different geometry";
        return false;
    }
    const int64_t blob = (int64_t) lay.max_blob;
    const uint64_t want = lay.total;

    // plan v0.3 P6: no experts.bin in a native pack -> the experts come straight from the GGUF
    const bool from_gguf = !std::ifstream(path, std::ios::binary) && lay.native && !lay.gguf_off.empty() && !gguf_.empty();
    // SIZE CHECK BEFORE THE ALLOCATION, not after.  A wrong pack should name the two numbers rather than spend
    // 34 GB and a minute of loading first.
    if (from_gguf) {
        // every (file, offset) of native_experts.txt must be the tensor it claims, of the pack's type and
        // dimensions and inside its file - before the allocation, so a pack of another model or a truncated
        // shard is a message rather than an arena of plausible wrong experts
        if (!check_experts_gguf(gguf_, lay, err)) { err = "ArenaExpertSource: " + err; return false; }
    } else {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) { err = "ArenaExpertSource: cannot open " + path; return false; }
        const uint64_t got = (uint64_t) f.tellg();
        if (got != want && got != want + (uint64_t) blob) {   // + one blob: STRATA_ARENA_MMAP's padded file
            char buf[400];
            std::snprintf(buf, sizeof buf,
                          "ArenaExpertSource: %s is %llu B but %lld layers x %lld experts (blobs up to %lld B) "
                          "make %llu B - this is not the pack this geometry came from",
                          path.c_str(), (unsigned long long) got, (long long) n_layers, (long long) n_expert,
                          (long long) blob, (unsigned long long) want);
            err = buf;
            return false;
        }
    }

    // STRATA_ARENA_MMAP=1 (a small-RAM machine whose GPUs hold most experts): the arena is the pack's experts.bin
    // mapped READ-ONLY - not locked, not pinned.  The page cache keeps what the CPU pool and the prompt path
    // actually read (the experts no VRAM cache holds) and gives back the rest under memory pressure; a page
    // dropped is re-read from the file, so a result never depends on what is resident.  The GPUs then get no
    // mapped alias: run with --pcie-frac 0.  The first start writes experts.bin (arena layout, padded by one blob
    // so a whole-slot copy may start at any expert), later ones map it.
#if defined(_WIN32)
    static const bool arena_mmap = false;   // POSIX mmap/madvise: Linux only for now
#else
    static const bool arena_mmap = [] { const char* v = std::getenv("STRATA_ARENA_MMAP"); return v && v[0] == '1'; }();
    if (arena_mmap) {
        const uint64_t file_bytes = want + (uint64_t) blob;
        uint64_t have = 0;
        {
            std::ifstream f(path, std::ios::binary | std::ios::ate);
            if (f) have = (uint64_t) f.tellg();
        }
        if (have == file_bytes) {
            const int fd = ::open(path.c_str(), O_RDONLY);
            void* v = fd >= 0 ? mmap(nullptr, (size_t) file_bytes, PROT_READ, MAP_SHARED, fd, 0) : MAP_FAILED;
            if (fd >= 0) ::close(fd);
            if (v == MAP_FAILED) { err = "ArenaExpertSource: mmap of " + path + " failed"; return false; }
            map_ = v;
            map_bytes_ = file_bytes;
            base_ = (const uint8_t*) v;
            pinned_bytes_ = 0;
            dev_slice_.clear();
            slice_bytes_ = 0;
            ranges_ = false;
            blobs_ = n_layers * n_expert;
            n_expert_ = n_expert;
            reads_ = 0;
            note_ = "mapped read-only from " + path + " (STRATA_ARENA_MMAP: not locked, not pinned)";
            gib_per_s_ = 0.0;
            load_seconds_ = load_read_s_ = load_copy_s_ = 0.0;
            record_inputs(path, gguf_, lay, from_gguf);   // session files: the mapped experts.bin is a model input
            return true;
        }
    }
#endif

    // #633: THE RAM BEFORE THE ALLOCATION.  On Linux the arena is an anonymous mapping that succeeds whatever the host
    // has; its pages are committed as the load writes them, so a container whose memory limit is below the arena was
    // killed by the OOM killer part-way through the load, without a message.  A hard cgroup limit below it is
    // certain to end that way: refused, with the numbers.  Less RAM available than the arena (another program, an
    // engine still exiting) is only a warning - the OS may make room - per the recommend-not-force rule.
    ram_warning_.clear();
    // (a shared arena, --shared-expert-arena, may already be in RAM for another engine: not checked)
    if (detail::HostMemory hm; shared_arena_file.empty() && detail::host_available_memory(hm)) {
        const uint64_t need = want + (uint64_t) blob;
        const double gib = 1073741824.0;
        char buf[512];
        if (hm.cgroup_limit < need) {
            std::snprintf(buf, sizeof buf,
                          "ArenaExpertSource: the expert arena needs %.2f GiB of RAM but this process's memory limit "
                          "(cgroup memory.max / memory.limit_in_bytes) is %.2f GiB: it would be killed while loading. "
                          "Raise the container's limit, or run with less RAM: --mmap-experts with "
                          "--resident-budget-gib N keeps only the hottest experts in RAM",
                          (double) need / gib, (double) hm.cgroup_limit / gib);
            err = buf;
            return false;
        }
        if (hm.available < need) {
            // #4051: the shortfall is often an engine that was still exiting (the code's own example), a few seconds
            // from giving its RAM back: wait for it, not for ever, and say so.  STRATA_ARENA_RAM_WAIT_S changes the
            // seconds (0 loads at once, as before this wait).  The load then goes on either way - it only warns.
            double wait_s = 20.0;
            if (const char* w = std::getenv("STRATA_ARENA_RAM_WAIT_S"); w != nullptr && w[0] != '\0') {
                char* end = nullptr;
                const double v = std::strtod(w, &end);
                if (end != w && *end == '\0' && v >= 0.0 && v <= 3600.0) wait_s = v;
            }
            double waited = 0.0;
            uint64_t avail = hm.available;
            if (wait_s > 0.0) {
                const bool came = detail::wait_for_ram(
                    need, wait_s, 0.25, 5.0,
                    [](uint64_t& a) { detail::HostMemory h; if (!detail::host_available_memory(h)) return false; a = h.available; return true; },
                    [](double s) { std::this_thread::sleep_for(std::chrono::duration<double>(s)); },
                    [&](double w, uint64_t a) {
                        std::fprintf(stderr, "strata generate: the expert arena needs %.2f GiB of RAM and %.2f GiB is "
                                             "available; waiting up to %.0f s for it (another engine still exiting?) - "
                                             "%.0f s so far (STRATA_ARENA_RAM_WAIT_S changes this)\n",
                                     (double) need / gib, (double) a / gib, wait_s, w);
                    },
                    waited, avail);
                if (came)
                    std::fprintf(stderr, "strata generate: %.2f GiB of RAM is available now (after %.1f s): loading the "
                                         "expert arena\n", (double) avail / gib, waited);
                hm.available = avail;
            }
            if (hm.available < need) {
                std::snprintf(buf, sizeof buf,
                              "the expert arena needs %.2f GiB of RAM but %.2f GiB is available (%.2f GiB short%s): the "
                              "load may swap or be stopped by the OS%s. Close other programs, or run with less RAM: "
                              "--mmap-experts with --resident-budget-gib N",
                              (double) need / gib, (double) hm.available / gib, (double) (need - hm.available) / gib,
                              waited > 0.0 ? ", after waiting" : "",
                              hm.available < need / 2 ? " - with this little room the whole machine can stall in "
                                                        "memory reclaim until it is restarted (#4051); on Linux a "
                                                        "systemd MemoryHigh= below the RAM keeps one program from "
                                                        "taking all of it" : "");
                ram_warning_ = buf;
            }
        }
    }

    uint64_t pack_hash = 0;
    if (!shared_arena_file.empty() &&
        !shared_arena_pack_hash(pack_dir, path, gguf_, lay, pack_hash, err)) return false;

    // one layer per registration slice, so no expert straddles two registrations.  The arena is one blob
    // longer than the file: a copy of a whole VRAM slot (the largest blob) may then start at any expert.
    std::vector<uint64_t> bounds, loff, lbytes;
    for (int64_t l = 0; l < n_layers; ++l) {
        bounds.push_back(lay.layer_offset(l));
        loff.push_back(lay.layer_offset(l));
        lbytes.push_back(lay.blob_bytes(l) * (uint64_t) n_expert);
    }
    bounds.push_back(want);
    // Pin by need: the arena gets a page-aligned blob STRIDE (blob rounded up to 4 KiB, ~+0.15 %), so every blob starts
    // on a page and any set of blobs can be registered exactly (a canonical blob is 337.5 pages: packed, an isolated
    // blob could never be registered without its neighbours' bytes).  Not with the mapped arena file (STRATA_ARENA_MMAP
    // writes experts.bin from the arena).  The file offsets (loff) stay packed; only the destination moves.
    padded_ = false;
    pad_ = PadMap{};
    if (pin_by_need_ && !pin_pairs_.empty() && !arena_mmap && shared_arena_file.empty()) {   // (a shared arena file has one layout for every process: the packed one)
        padded_ = true;
        uint64_t at = 0;
        for (int64_t l = 0; l < n_layers; ++l) {
            const uint64_t b = lay.blob_bytes(l), s = (b + 4095) & ~(uint64_t) 4095;
            pad_.dst_off.push_back(at);
            pad_.blob.push_back(b);
            pad_.stride.push_back(s);
            at += s * (uint64_t) n_expert;
        }
        pad_.total = at;
    }
    const uint64_t arena_bytes = padded_ ? pad_.total : want;   // the spare blob below comes on top, as before
    // Pin by need: the pairs set_pin_priority gave, as byte ranges in priority order.  Neighbours (adjacent blobs) merge
    // into one range that takes the best rank of its members, so the hottest experts are registered first and the
    // number of cudaHostRegister calls stays far below the number of pairs.
    std::vector<std::pair<uint64_t, uint64_t>> need_ranges;
    const bool by_need = padded_;
    if (by_need) {
        // every blob starts on a page and owns its whole stride (pad included), so a run of adjacent blobs is one exact
        // range: [first start, last start + stride).  The best rank of a run's members is its priority.
        struct Ext { uint64_t off, len; size_t rank; int tier; };
        std::vector<Ext> ex;
        ex.reserve(pin_pairs_.size());
        std::vector<uint8_t> seen((size_t) (n_layers * n_expert), 0);
        for (size_t i = 0; i < pin_pairs_.size(); ++i) {
            const int64_t l = pin_pairs_[i].first, e = pin_pairs_[i].second;
            if (l < 0 || l >= n_layers || e < 0 || e >= n_expert || seen[(size_t) (l * n_expert + e)]) continue;
            seen[(size_t) (l * n_expert + e)] = 1;
            ex.push_back({arena_off(l, e), pad_.stride[(size_t) l], i, i < pin_first_ ? 0 : 1});
        }
        std::sort(ex.begin(), ex.end(), [](const Ext& a, const Ext& b) { return a.off < b.off; });
        std::vector<Ext> merged;
        for (const Ext& x : ex) {
            if (!merged.empty() && merged.back().off + merged.back().len == x.off && merged.back().tier == x.tier) {
                merged.back().len += x.len;
                merged.back().rank = std::min(merged.back().rank, x.rank);
            } else {
                merged.push_back(x);
            }
        }
        std::stable_sort(merged.begin(), merged.end(), [](const Ext& a, const Ext& b) { return a.rank < b.rank; });
        for (const Ext& x : merged) need_ranges.emplace_back(x.off, x.len);
    }
    // #285: on Windows the per-layer registration runs on a thread ahead of the readers, which wait for their
    // layer's slice (the whole arena registered before the load cost ~2.5 s at 33 GiB of 4 KB pages, not hidden).
    // Only where the whole arena would be registered: no cap (STRATA_ARENA_PIN_GIB, multi-GPU), no shared arena.
    // Linux keeps the whole-arena pin (#253). OPT-IN (STRATA_DEFERRED_REGISTER=1): on the RTX 5070 / Windows the
    // arena registered per layer before it is filled decoded ~10% slower than the old order (whole-arena try, sliced
    // fallback after the load) with identical output; the start-time gain does not pay for that.
#ifdef _WIN32
    const char* defer_env = std::getenv("STRATA_DEFERRED_REGISTER");
    const bool deferred = !by_need && max_pinned_bytes == 0 && arena_pin_cap_gib() == -1 && shared_arena_file.empty() &&
                          (defer_env != nullptr && defer_env[0] != 0 && defer_env[0] != '0');
#else
    const bool deferred = false;
#endif
    PinnedArena* a = by_need ? new PinnedArena(arena_bytes + (uint64_t) blob, PinnedArena::ByNeed{}, need_ranges,
                                               max_pinned_bytes, shared_arena_file, pack_hash)
                   : deferred ? new PinnedArena(want + (uint64_t) blob, bounds, PinnedArena::Deferred{})
                              : new PinnedArena(want + (uint64_t) blob, bounds, max_pinned_bytes,
                                                shared_arena_file, pack_hash);
    if (!a->valid()) {
        const std::string why = a->note;
        delete a;
        err = "ArenaExpertSource: the arena could not be reserved (" +
              std::to_string(want + (uint64_t) blob) + " B)" +
              (why.empty() ? std::string{} : ": " + why);
        return false;
    }
    // #285: unbuffered when the drive is read anyway and the file cache could not keep the experts for the next
    // start either (a 64 GB PC); otherwise the buffered readers, which a warm restart serves from the cache
    std::vector<std::string> files;
    if (from_gguf) {
        for (int64_t l = 0; l < n_layers; ++l)
            for (int r = 0; r < 3; ++r) {
                const std::string f = expert_gguf_file(gguf_, lay, l, r);
                if (std::find(files.begin(), files.end(), f) == files.end()) files.push_back(f);
            }
    } else {
        files.push_back(path);
    }
    std::string why;
    const bool unbuffered = experts_unbuffered(files, want + (uint64_t) blob, why);
    const int readers = unbuffered ? std::max(threads, 16) : threads;   // 16 keep a PCIe 5 drive's queue full
    LoadStats st;
    // A shared arena (--shared-expert-arena, the replicas): ONE engine reads the experts into it, the others wait and skip the
    // read.  The state word in the file's header says who: 0 nobody, a pid loading, ~0 filled.  An engine that finds a
    // pid that is gone takes the load over (the previous one died mid-read: the arena is read again from the start), so a
    // dead loader never leaves the others waiting, and an engine that waits too long loads it itself.
    constexpr uint64_t kArenaFilled = ~uint64_t{0};
    std::atomic<uint64_t>* arena_state = padded_ ? nullptr : a->shared_state;
    bool arena_leader = false, arena_skip = false;
    if (arena_state != nullptr) {
        const uint64_t me = arena_this_pid();
        const auto t_begin = std::chrono::steady_clock::now();
        bool said = false;
        for (;;) {
            uint64_t cur = arena_state->load(std::memory_order_acquire);
            if (cur == kArenaFilled) { arena_skip = true; break; }
            if (cur == 0 || cur == me) {
                if (arena_state->compare_exchange_strong(cur, me, std::memory_order_acq_rel)) { arena_leader = true; break; }
                continue;
            }
            const bool alive = arena_pid_alive(cur);
            if (!alive) {
                if (arena_state->compare_exchange_strong(cur, me, std::memory_order_acq_rel)) {
                    std::fprintf(stderr, "strata generate: shared expert arena: engine %llu died while reading it; reading it again\n",
                                 (unsigned long long) cur);
                    arena_leader = true;
                    break;
                }
                continue;
            }
            if (std::chrono::steady_clock::now() - t_begin > std::chrono::minutes(45)) break;   // (loads it on its own)
            if (!said) {
                said = true;
                std::fprintf(stderr, "strata generate: shared expert arena: engine %llu is reading the experts; waiting for it "
                                     "instead of reading them again\n", (unsigned long long) cur);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }
    std::atomic<int> ready{0};
    std::thread reg;
    if (deferred) {
        int dev = 0;
        cudaGetDevice(&dev);
        reg = std::thread([a, &ready, dev] {
            cudaSetDevice(dev);
            a->register_slices(ready);
        });
    }
    const std::atomic<int>* rp = deferred ? &ready : nullptr;
    if (arena_skip) {
        st.ok = true;
        st.bytes = want;
        st.seconds = 0.0;
        std::fprintf(stderr, "strata generate: shared expert arena: already filled by another engine; the read is skipped\n");
    } else if (from_gguf) {
        st = load_experts_gguf(gguf_, a->data(), lay, readers, unbuffered, rp, padded_ ? &pad_ : nullptr);
    } else {
        if (unbuffered && !padded_) st = load_experts_direct(path, a->data(), loff, lbytes, readers, /*chunk=*/8u << 20, rp);
        if (!unbuffered || padded_ || (!st.ok && st.error.empty()))   // unaligned ranges / the padded layout: the buffered reader
            st = load_experts_ranges(path, a->data(), loff, lbytes, threads, /*chunk=*/8u << 20, rp, padded_ ? &pad_ : nullptr);
    }
    if (reg.joinable()) reg.join();
    std::fprintf(stderr, "strata generate: expert arena read %s (%s)\n", unbuffered ? "unbuffered" : "through the file cache",
                 why.c_str());
    if (!st.ok) {
        if (arena_leader) arena_state->store(0, std::memory_order_release);   // (somebody else may try)
        delete a;
        err = "ArenaExpertSource: the expert load was refused: " + (st.error.empty() ? std::string("unknown") : st.error);
        return false;
    }
    if (st.bytes != want) {
        if (arena_leader) arena_state->store(0, std::memory_order_release);
        delete a;
        err = "ArenaExpertSource: the load read " + std::to_string(st.bytes) + " B of " + std::to_string(want);
        return false;
    }
    if (arena_leader) arena_state->store(kArenaFilled, std::memory_order_release);   // the others may use (and skip reading) it
    if (std::getenv("STRATA_VERIFY_ARENA") != nullptr) {   // tests: the loaded arena's checksum, any loader
        std::fprintf(stderr, "strata generate: expert arena checksum %016llx (%.2f GiB)\n",
                     (unsigned long long) fnv1a64(a->data(), want), (double) want / (1ull << 30));
    }
    // the first start: write experts.bin for the mapped starts after this one - only when the drive has room for it
    // and 2 GiB more (a full drive fails other writes too); otherwise this start says so and runs pinned as before
    std::error_code space_ec;
    const uint64_t free_disk = arena_mmap && from_gguf ? (uint64_t) std::filesystem::space(
        std::filesystem::path(path).parent_path(), space_ec).available : 0;
    const bool room = !space_ec && free_disk >= want + (uint64_t) blob + (2ull << 30);
    if (arena_mmap && from_gguf && !room)
        std::fprintf(stderr, "strata generate: STRATA_ARENA_MMAP: NOT writing %s - %.1f GiB free on that drive, it needs "
                             "%.1f GiB plus 2 GiB to spare; this start keeps the arena in RAM\n", path.c_str(),
                     (double) free_disk / 1073741824.0, (double) (want + (uint64_t) blob) / 1073741824.0);
    if (arena_mmap && from_gguf && room) {
        const std::string tmp = path + ".tmp";
        std::FILE* f = std::fopen(tmp.c_str(), "wb");
        bool ok = f != nullptr && std::fwrite(a->data(), 1, (size_t) want, f) == (size_t) want;
        if (ok) {
            std::vector<uint8_t> pad((size_t) blob, 0);
            ok = std::fwrite(pad.data(), 1, pad.size(), f) == pad.size();
        }
        if (f) ok = std::fclose(f) == 0 && ok;
        if (ok) ok = std::rename(tmp.c_str(), path.c_str()) == 0;
        if (!ok) std::remove(tmp.c_str());
        std::fprintf(stderr, "strata generate: STRATA_ARENA_MMAP: %s %s (the next start maps it)\n",
                     ok ? "wrote" : "could NOT write", path.c_str());
    }
    arena_ = a;
    base_ = a->data();
    pinned_bytes_ = a->registered_bytes;
    // plan v0.3 P6: device aliases of the mapped registration, for the PCIe share of the misses
    dev_slice_.clear();
    slice_bytes_ = a->slice_bytes;
    pin_start_.clear(); pin_end_.clear(); pin_dev_.clear(); layer_pcie_.clear();
    ranges_ = false;
    if (a->by_need) {
        // device alias of each registered range's start (device_alias adds the blob's distance from it); one
        // refusal drops the whole table: no PCIe share rather than a wrong pointer
        ranges_ = true;
        for (const auto& r : a->pin_ranges) {
            void* d = nullptr;
            if (cudaHostGetDevicePointer(&d, (void*) (base_ + r.first), 0) != cudaSuccess) {
                (void) cudaGetLastError();
                ranges_ = false;
                break;
            }
            pin_start_.push_back(r.first);
            pin_end_.push_back(r.first + r.second);
            pin_dev_.push_back((const uint8_t*) d);
        }
        if (!ranges_) {
            pin_start_.clear(); pin_end_.clear(); pin_dev_.clear();
            pinned_bytes_ = 0;
        } else {
            layer_pcie_.assign((size_t) n_layers, 0);
            n_expert_ = n_expert;   // pinned() reads it; the common tail below sets the same value
            for (int64_t l = 0; l < n_layers; ++l)
                for (int64_t e = 0; e < n_expert && !layer_pcie_[(size_t) l]; ++e)
                    layer_pcie_[(size_t) l] = pinned(l, e) ? 1 : 0;
        }
    } else if (a->registered_bytes > 0) {
        std::vector<uint64_t> starts = a->slice_bytes > 0 ? a->slice_starts : std::vector<uint64_t>{0};
        for (uint64_t off : starts) {
            void* d = nullptr;
            if (cudaHostGetDevicePointer(&d, (void*) (base_ + off), 0) != cudaSuccess) {
                (void) cudaGetLastError();
                dev_slice_.clear();
                break;
            }
            dev_slice_.push_back((const uint8_t*) d);
        }
    }
    blobs_ = n_layers * n_expert;
    n_expert_ = n_expert;
    reads_ = 0;
    note_ = a->note;
    gib_per_s_ = st.gib_per_second();
    load_seconds_ = st.seconds;
    load_read_s_ = st.read_seconds;
    load_copy_s_ = st.copy_seconds;
    record_inputs(path, gguf_, lay, from_gguf);
    return true;
}

void ArenaExpertSource::close() {
#if !defined(_WIN32)
    if (map_ != nullptr) munmap(map_, (size_t) map_bytes_);
#endif
    map_ = nullptr;
    map_bytes_ = 0;
    inputs_.clear();
    if (arena_ != nullptr) {
        delete (PinnedArena*) arena_;
        arena_ = nullptr;
    }
    base_ = nullptr;
    blobs_ = 0;
    n_expert_ = 0;
}

void ArenaExpertSource::prefetch(int64_t layer, int64_t expert) {
#if defined(_WIN32)
    (void) layer; (void) expert;
#else
    if (map_ == nullptr || layer < 0 || expert < 0 || expert >= n_expert_) return;
    const auto& lay = strata::kernels::cpu::expert_layout();
    const uint64_t off = lay.blob_offset(layer, expert) & ~(uint64_t) 4095;
    const uint64_t end = lay.blob_offset(layer, expert) + lay.blob_bytes(layer);
    madvise((uint8_t*) map_ + off, (size_t) (end - off), MADV_WILLNEED);
#endif
}

// Only the pages wholly inside the blob: a page shared with a neighbour the CPU may still read stays.  MADV_DONTNEED
// unmaps them from this process; they stay in the page cache as clean, unmapped pages - the first the kernel takes
// back under pressure, and a minor fault away when the CPU needs the expert again (an adaptive swap evicts it).
// Dropping them from the cache too (POSIX_FADV_DONTNEED) made every eviction a re-read from the disk: ~200 major
// faults a second in decode, +8 ms a window.  The memory the arena used to hold was never the cache itself but
// ROCclr's pin-in-place locks on it (see main) - locked pages are the ones the kernel cannot take back.
uint64_t ArenaExpertSource::release(int64_t layer, int64_t expert) {
#if defined(_WIN32)
    (void) layer; (void) expert;
    return 0;
#else
    if (map_ == nullptr || layer < 0 || expert < 0 || expert >= n_expert_) return 0;
    const auto& lay = strata::kernels::cpu::expert_layout();
    const uint64_t off = (lay.blob_offset(layer, expert) + 4095) & ~(uint64_t) 4095;
    const uint64_t end = (lay.blob_offset(layer, expert) + lay.blob_bytes(layer)) & ~(uint64_t) 4095;
    if (end <= off) return 0;
    if (madvise((uint8_t*) map_ + off, (size_t) (end - off), MADV_DONTNEED) != 0) return 0;
    return end - off;
#endif
}

int64_t ArenaExpertSource::pin_range_of(uint64_t off, uint64_t len) const {
    // last range starting at or before the blob; it must also end after it (a blob is inside ONE range: merged
    // neighbours are contiguous, and a blob split by a trimmed page is reported unpinned - the safe answer)
    const auto it = std::upper_bound(pin_start_.begin(), pin_start_.end(), off);
    if (it == pin_start_.begin()) return -1;
    const size_t i = (size_t) (it - pin_start_.begin()) - 1;
    return off + len <= pin_end_[i] ? (int64_t) i : -1;
}

uint64_t ArenaExpertSource::arena_off(int64_t layer, int64_t expert) const {
    return padded_ ? pad_.blob_off(layer, expert) : strata::kernels::cpu::expert_layout().blob_offset(layer, expert);
}

bool ArenaExpertSource::pcie_layer(int64_t layer) const {
    if (!ranges_) return device_alias(layer, 0) != nullptr;   // the prefix rule (ExpertSource's default)
    return layer >= 0 && (size_t) layer < layer_pcie_.size() && layer_pcie_[(size_t) layer] != 0;
}

bool ArenaExpertSource::pinned(int64_t layer, int64_t expert) const {
    if (base_ == nullptr || layer < 0 || expert < 0 || expert >= n_expert_) return false;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (ranges_) return pin_range_of(arena_off(layer, expert), lay.blob_bytes(layer)) >= 0;   // the whole blob inside ONE registered range
    return lay.blob_offset(layer, expert) + lay.blob_bytes(layer) <= pinned_bytes_;
}

const uint8_t* ArenaExpertSource::device_alias(int64_t layer, int64_t expert) const {
    if (ranges_) {   // pin by need: the range's device base + the blob's offset inside it
        if (base_ == nullptr || layer < 0 || expert < 0 || expert >= n_expert_) return nullptr;
        const auto& rl = strata::kernels::cpu::expert_layout();
        const uint64_t off = arena_off(layer, expert);
        const int64_t i = pin_range_of(off, rl.blob_bytes(layer));
        return i < 0 ? nullptr : pin_dev_[(size_t) i] + (off - pin_start_[(size_t) i]);
    }
    if (dev_slice_.empty() || !pinned(layer, expert)) return nullptr;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (slice_bytes_ == 0) return dev_slice_[0] + lay.blob_offset(layer, expert);
    // one registration slice per layer
    if ((size_t) layer >= dev_slice_.size()) return nullptr;
    return dev_slice_[(size_t) layer] + (uint64_t) expert * lay.blob_bytes(layer);
}

const uint8_t* ArenaExpertSource::blob(int64_t layer, int64_t expert) {
    if (base_ == nullptr) return nullptr;
    if (layer < 0 || expert < 0 || expert >= n_expert_) return nullptr;
    const int64_t idx = layer * n_expert_ + expert;
    if (idx < 0 || idx >= blobs_) return nullptr;
    reads_.fetch_add(1, std::memory_order_relaxed);
    // Pointer arithmetic into resident memory.  No fault, no copy, no mapping - which is the entire point of
    // this class over `FileExpertSource`.
    return base_ + arena_off(layer, expert);
}

const uint8_t* ArenaExpertSource::blob_at(int64_t layer, int64_t expert) const {
    if (base_ == nullptr || layer < 0 || expert < 0 || expert >= n_expert_) return nullptr;
    const int64_t idx = layer * n_expert_ + expert;
    if (idx < 0 || idx >= blobs_) return nullptr;
    return base_ + arena_off(layer, expert);   // (the padded layout of pin by need too)
}

}  // namespace strata::core

// ---- Foresight swap space (see include/strata/core/foresight_swap.hpp) ------------------------------------------
namespace strata::core {

bool ForesightSwap::init(int slots_per_layer, int64_t layers, int64_t experts, const std::vector<int>& card_of_layer,
                         const std::vector<int>& devices, std::string& err) {
    slots = slots_per_layer;
    n_layers = layers;
    n_expert = experts;
    const size_t ns = (size_t) (layers * slots);
    slot.reset(new Slot[ns]);
    where.assign((size_t) (layers * experts), -1);
    miss_cnt.assign((size_t) (layers * experts), 0);
    miss_at.assign((size_t) (layers * experts), 0);
    layer_card = card_of_layer;
    layer_dev.resize((size_t) layers);
    card_dev = devices;
    card_stream.assign(devices.size(), nullptr);
    card_mem.assign(devices.size(), nullptr);
    card_round.assign(devices.size(), ~0ull);
    card_used.assign(devices.size(), 0);
    const auto& lay = strata::kernels::cpu::expert_layout();
    for (size_t c = 0; c < devices.size(); ++c) {
        uint64_t bytes = 0;
        for (int64_t l = 0; l < layers; ++l)
            if (card_of_layer[(size_t) l] == (int) c) bytes += (uint64_t) slots * (uint64_t) lay.blob_bytes(l);
        if (bytes == 0) continue;
        const OnDevice on(devices[c]);
        cudaStream_t s = nullptr;
        if (cudaMalloc(&card_mem[c], bytes) != cudaSuccess ||
            cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) {
            cudaGetLastError();
            err = "cannot allocate " + std::to_string(bytes >> 20) + " MiB on CUDA" + std::to_string(devices[c]) +
                  " (raise --vram-reserve-mib by about that much)";
            return false;
        }
        card_stream[c] = s;
        uint64_t off = 0;
        for (int64_t l = 0; l < layers; ++l) {
            if (card_of_layer[(size_t) l] != (int) c) continue;
            layer_dev[(size_t) l] = devices[c];
            const uint64_t bb = (uint64_t) lay.blob_bytes(l);
            for (int s2 = 0; s2 < slots; ++s2) {
                Slot& sl = slot[(size_t) (l * slots + s2)];
                sl.dptr = (unsigned long long) ((uint8_t*) card_mem[c] + off);
                sl.bytes = bb;
                cudaEvent_t ev = nullptr;
                if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) {
                    cudaGetLastError();
                    err = "cannot create the copy events";
                    return false;
                }
                sl.ev = ev;
                off += bb;
            }
        }
    }
    filler_ = std::thread([this] { fill_loop(); });
    return true;
}

ForesightSwap::~ForesightSwap() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    if (filler_.joinable()) filler_.join();
    for (size_t c = 0; c < card_dev.size(); ++c) {
        const OnDevice on(card_dev[c]);
        if (card_stream[c]) cudaStreamSynchronize((cudaStream_t) card_stream[c]);
        if (card_mem[c]) cudaFree(card_mem[c]);
        if (card_stream[c]) cudaStreamDestroy((cudaStream_t) card_stream[c]);
    }
    if (slot)
        for (int64_t i = 0; i < n_layers * slots; ++i)
            if (slot[(size_t) i].ev) cudaEventDestroy((cudaEvent_t) slot[(size_t) i].ev);
}

void ForesightSwap::fill_loop() {
    strata::platform::name_this_thread("strata-fill");
    int cur_dev = -1;
    for (;;) {
        int64_t idx;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return stop_ || !queue_.empty(); });
            if (stop_) return;
            idx = queue_.front();
            queue_.pop_front();
        }
        Slot& sl = slot[(size_t) idx];
        const int64_t l = idx / slots;
        const int dev = layer_dev[(size_t) l];
        if (dev != cur_dev) { cudaSetDevice(dev); cur_dev = dev; }
        cudaStream_t s = (cudaStream_t) card_stream[(size_t) layer_card[(size_t) l]];
        if (cudaMemcpyAsync((void*) sl.dptr, sl.src, sl.bytes, cudaMemcpyHostToDevice, s) != cudaSuccess ||
            cudaEventRecord((cudaEvent_t) sl.ev, s) != cudaSuccess) {
            cudaGetLastError();
            if (!failed.exchange(true)) std::fprintf(stderr, "strata serve: foresight swap: a copy failed - swap space off\n");
            continue;   // never marked issued: the slot is never used
        }
        sl.issued.store(true, std::memory_order_release);
    }
}

bool ForesightSwap::take(int64_t layer, int32_t expert, unsigned long long& ptr) {
    std::lock_guard<std::mutex> smu_lk(smu_);
    const int s = where[(size_t) (layer * n_expert + expert)];
    if (s < 0) return false;
    Slot& sl = slot[(size_t) (layer * slots + s)];
    if (!sl.ready) {
        if (!sl.loading || !sl.issued.load(std::memory_order_acquire)) { ++pending; return false; }
        const cudaError_t q = cudaEventQuery((cudaEvent_t) sl.ev);
        if (q != cudaSuccess) {
            if (q != cudaErrorNotReady) cudaGetLastError();
            ++pending;
            return false;
        }
        sl.ready = true;
        sl.loading = false;
        if (verify) {   // test mode: the bytes the GPU will read must be the expert's own
            std::vector<uint8_t> back((size_t) sl.bytes);
            const OnDevice on(layer_dev[(size_t) layer]);
            const bool same = cudaMemcpy(back.data(), (const void*) sl.dptr, (size_t) sl.bytes, cudaMemcpyDeviceToHost) ==
                                  cudaSuccess && std::memcmp(back.data(), sl.src, (size_t) sl.bytes) == 0;
            if (same) ++verify_ok;
            else {
                cudaGetLastError();
                ++verify_bad;
                sl.ready = false;                           // never serve it
                where[(size_t) (layer * n_expert + expert)] = -1;
                sl.expert = -1;
                return false;
            }
        }
    }
    sl.last_ref = completed.load(std::memory_order_acquire);
    ptr = sl.dptr;
    ++hits;
    return true;
}

void ForesightSwap::hold(int64_t layer, int32_t expert) {
    if (layer < 0 || layer >= n_layers || expert < 0 || expert >= n_expert) return;
    std::lock_guard<std::mutex> smu_lk(smu_);
    const size_t key = (size_t) (layer * n_expert + expert);
    const int s = where[key];
    where[key] = kHeld;
    if (s < 0) return;
    Slot& sl = slot[(size_t) (layer * slots + s)];
    sl.expert = kDropped;   // `loading` and `last_ref` stay: a copy in flight and a window that read it are waited for
    sl.ready = false;
    ++dropped;
}

void ForesightSwap::unhold(int64_t layer, int32_t expert) {
    if (layer < 0 || layer >= n_layers || expert < 0 || expert >= n_expert) return;
    std::lock_guard<std::mutex> smu_lk(smu_);
    int16_t& w = where[(size_t) (layer * n_expert + expert)];
    if (w == kHeld) w = -1;
}

void ForesightSwap::note_miss(ExpertSource* src, int64_t layer, int32_t expert) {
    std::lock_guard<std::mutex> smu_lk(smu_);
    request_locked(src, layer, expert, false);
}

void ForesightSwap::predict(ExpertSource* src, int64_t layer, int32_t expert) {
    if (layer < 0 || layer >= n_layers || expert < 0 || expert >= n_expert) return;
    std::lock_guard<std::mutex> smu_lk(smu_);
    request_locked(src, layer, expert, true);
}

void ForesightSwap::request_locked(ExpertSource* src, int64_t layer, int32_t expert, bool predicted_copy) {
    if (failed.load(std::memory_order_relaxed)) return;
    const size_t key = (size_t) (layer * n_expert + expert);
    if (where[key] != -1) return;                       // already on its way, or held by an exchange (#2134)
    const uint64_t now = completed.load(std::memory_order_acquire);
    if (admit > 1 && !predicted_copy) {
        if (now > miss_at[key] + 4) miss_cnt[key] = 0;
        miss_at[key] = now;
        if (miss_cnt[key] < 255) ++miss_cnt[key];
        if (miss_cnt[key] < admit) return;
    }
    const int c = layer_card[(size_t) layer];
    if (card_round[(size_t) c] != now) { card_round[(size_t) c] = now; card_used[(size_t) c] = 0; }
    if (card_used[(size_t) c] >= budget) { ++over_budget; return; }
    if (src == nullptr || !src->pinned(layer, expert)) return;   // async copies need the pinned arena
    const uint8_t* b = src->blob(layer, expert);
    if (b == nullptr) return;
    int best = -1;
    uint64_t best_ref = ~0ull;
    for (int s = 0; s < slots; ++s) {   // an empty slot, else the one used longest ago that no window can still read
        Slot& sl = slot[(size_t) (layer * slots + s)];
        if (sl.loading) {   // copied in but never asked for since: landed long ago -> an ordinary candidate
            if (sl.last_ref + (uint64_t) depth > now || !sl.issued.load(std::memory_order_acquire)) continue;
            const cudaError_t q = cudaEventQuery((cudaEvent_t) sl.ev);
            if (q != cudaSuccess) { if (q != cudaErrorNotReady) cudaGetLastError(); continue; }
            sl.loading = false;
            sl.ready = true;
        }
        if (sl.expert == -1) { best = s; break; }      // never used (a dropped slot, kDropped, waits like a used one)
        if (sl.last_ref + (uint64_t) depth > now) continue;
        if (sl.last_ref < best_ref) { best_ref = sl.last_ref; best = s; }
    }
    if (best < 0) { ++busy; return; }
    Slot& sl = slot[(size_t) (layer * slots + best)];
    if (sl.expert >= 0) where[(size_t) (layer * n_expert + sl.expert)] = -1;
    sl.expert = expert;
    sl.ready = false;
    sl.loading = true;
    sl.issued.store(false, std::memory_order_relaxed);
    sl.src = b;
    sl.last_ref = now;
    where[key] = (int16_t) best;
    ++card_used[(size_t) c];
    ++copies;
    if (predicted_copy) ++predicted;
    {
        std::lock_guard<std::mutex> lk(mu_);
        queue_.push_back(layer * slots + best);
    }
    cv_.notify_one();
}

std::string ForesightSwap::report() const {
    std::lock_guard<std::mutex> smu_lk(const_cast<std::mutex&>(smu_));
    char buf[320];
    std::snprintf(buf, sizeof buf, "foresight swap: %llu experts served from the swap space, %llu copies (%llu from the "
                                   "router look-ahead), %llu not landed yet, %llu no free slot, %llu over budget%s",
                  (unsigned long long) hits, (unsigned long long) copies, (unsigned long long) predicted,
                  (unsigned long long) pending, (unsigned long long) busy, (unsigned long long) over_budget, verify ? "" : "");
    std::string r = buf;
    if (dropped > 0) r += ", " + std::to_string(dropped) + " dropped by exchanges";
    if (verify) r += ", verified " + std::to_string(verify_ok) + " ok / " + std::to_string(verify_bad) + " BAD";
    return r;
}

ForesightSwap* foresight_swap_from_env(int64_t layers, int64_t experts, const std::vector<int>& card_of_layer,
                                       const std::vector<int>& devices) {
    const char* v = std::getenv("STRATA_FS_SLOTS");
    const int slots = v ? std::atoi(v) : 0;
    if (slots <= 0) return nullptr;
    if (const char* dp = std::getenv("STRATA_VERIFY_DEVICE_PLAN"); dp && std::atoi(dp) != 0) {
        std::fprintf(stderr, "strata serve: foresight swap: off beside STRATA_VERIFY_DEVICE_PLAN (the device would plan "
                             "resident layers without the host)\n");
        return nullptr;
    }
    auto* f = new ForesightSwap();
    auto env_int = [](const char* name, int def, int lo) { const char* e = std::getenv(name); const int x = e ? std::atoi(e) : def; return x < lo ? lo : x; };
    f->budget = env_int("STRATA_FS_BUDGET", 32, 1);
    f->admit = env_int("STRATA_FS_ADMIT", 1, 1);
    f->depth = env_int("STRATA_FS_DEPTH", 4, 2);
    f->verify = env_int("STRATA_FS_VERIFY", 0, 0) != 0;
    std::string err;
    if (!f->init(std::min(slots, 64), layers, experts, card_of_layer, devices, err)) {
        std::fprintf(stderr, "strata serve: foresight swap: %s - swap space off\n", err.c_str());
        delete f;
        return nullptr;
    }
    std::fprintf(stderr, "strata serve: foresight swap: %d slots per layer on %zu card(s), up to %d copies per card and "
                         "window, admit after %d miss(es), reuse after %d windows\n",
                 f->slots, devices.size(), f->budget, f->admit, f->depth);
    return f;
}

}  // namespace strata::core
