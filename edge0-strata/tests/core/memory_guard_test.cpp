// available_host_bytes(): MemAvailable, the cgroup limits above this process, and the operator's cap, on fixture
// directories standing in for /proc/meminfo, /proc/self/cgroup and /sys/fs/cgroup.  CPU only, no GPU.
#include "strata/core/conversation_memory.hpp"

#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {

int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

constexpr uint64_t MiB = 1ull << 20;
constexpr uint64_t GiB = 1ull << 30;

struct Fixture {
    fs::path dir;
    Fixture() {
        static int n = 0;
        dir = fs::temp_directory_path() / ("strata_memory_guard_" + std::to_string((long) ::getpid()) + "_" + std::to_string(n++));
        fs::remove_all(dir);
        fs::create_directories(dir / "cgroup");
    }
    ~Fixture() { std::error_code ec; fs::remove_all(dir, ec); }
    void put(const std::string& rel, const std::string& text) {
        const fs::path p = dir / rel;
        fs::create_directories(p.parent_path());
        std::ofstream(p, std::ios::binary) << text;
    }
    void put(const std::string& rel, uint64_t n) { put(rel, std::to_string(n) + "\n"); }
    void meminfo(uint64_t total, uint64_t available) {
        put("meminfo", "MemTotal:       " + std::to_string(total / 1024) + " kB\nMemFree: 1 kB\nMemAvailable:   " +
                           std::to_string(available / 1024) + " kB\n");
    }
    void self(const std::string& text) { put("self_cgroup", text + "\n"); }
    MemoryProbePaths paths() const { return {(dir / "meminfo").string(), (dir / "self_cgroup").string(), (dir / "cgroup").string()}; }
    std::optional<HostMemoryReading> sample(uint64_t explicit_limit = 0) const { return sample_host_memory(paths(), explicit_limit); }
};

}  // namespace

int main() {
    // no cgroup limit anywhere ("max" at every level, and the root without memory files): plain MemAvailable
    {
        Fixture f;
        f.meminfo(120 * GiB, 25 * GiB);
        f.self("0::/a/b");
        f.put("cgroup/memory.max", "max\n"); f.put("cgroup/memory.current", 90 * GiB);
        f.put("cgroup/a/memory.max", "max\n"); f.put("cgroup/a/memory.current", 60 * GiB);
        f.put("cgroup/a/b/memory.max", "max\n"); f.put("cgroup/a/b/memory.current", 50 * GiB);
        const auto r = f.sample();
        check(r && r->available == 25 * GiB && r->source == MemorySource::meminfo, "max everywhere is plain MemAvailable");
        check(r->limit == 120 * GiB && r->current == 95 * GiB && r->mem_available == 25 * GiB, "meminfo reading: MemTotal and used");
    }
    // nested v2 limits: the tightest room of the group and its ancestors wins, "max" levels are skipped
    {
        Fixture f;
        f.meminfo(120 * GiB, 80 * GiB);
        f.self("0::/a/b");
        f.put("cgroup/memory.max", 100 * GiB); f.put("cgroup/memory.current", 90 * GiB);   // 10 GiB of room
        f.put("cgroup/a/memory.max", 60 * GiB); f.put("cgroup/a/memory.current", 58 * GiB);  // 2 GiB of room
        f.put("cgroup/a/b/memory.max", "max\n"); f.put("cgroup/a/b/memory.current", 50 * GiB);
        auto r = f.sample();
        check(r && r->available == 2 * GiB && r->source == MemorySource::cgroup, "tightest ancestor wins");
        check(r->limit == 60 * GiB && r->current == 58 * GiB, "reading names that group's limit and usage");
        f.put("cgroup/a/memory.current", 59 * GiB + 512 * MiB);
        r = f.sample();
        check(r && r->available == 512 * MiB, "the room follows memory.current");
        f.put("cgroup/a/memory.current", 61 * GiB);
        r = f.sample();
        check(r && r->available == 0 && r->source == MemorySource::cgroup, "usage over the limit is a known zero");
        f.put("cgroup/a/memory.current", 10 * GiB);
        r = f.sample();
        check(r && r->available == 10 * GiB && r->limit == 100 * GiB, "the root's room wins once the group has more");
        check(r->source == MemorySource::cgroup, "source cgroup");
        // the room under a limit larger than MemAvailable is bounded by MemAvailable
        f.meminfo(120 * GiB, 4 * GiB);
        r = f.sample();
        check(r && r->available == 4 * GiB && r->source == MemorySource::meminfo, "MemAvailable bounds a roomy cgroup");
    }
    // memory.high below memory.max counts, and a high of "max" does not
    {
        Fixture f;
        f.meminfo(120 * GiB, 80 * GiB);
        f.self("0::/g");
        f.put("cgroup/g/memory.max", 100 * GiB); f.put("cgroup/g/memory.high", 99 * GiB + 900 * MiB);
        f.put("cgroup/g/memory.current", 99 * GiB);
        auto r = f.sample();
        check(r && r->available == 900 * MiB && r->limit == 99 * GiB + 900 * MiB, "memory.high lower than memory.max");
        f.put("cgroup/g/memory.high", "max\n");
        r = f.sample();
        check(r && r->available == 1 * GiB && r->limit == 100 * GiB, "memory.high max leaves memory.max");
        f.put("cgroup/g/memory.max", "max\n"); f.put("cgroup/g/memory.high", 90 * GiB);
        r = f.sample();
        check(r && r->available == 0, "memory.high alone, already exceeded");
    }
    // the operator's cap: the container cannot see its real limit (memory.max is "max" at the namespace root)
    {
        Fixture f;
        f.meminfo(120 * GiB, 25 * GiB);   // the stale view: MemAvailable says 25 GiB
        f.self("0::/system.slice/strata.service");
        f.put("cgroup/memory.max", "max\n"); f.put("cgroup/memory.high", "max\n");
        f.put("cgroup/memory.current", 99 * GiB + 724 * MiB);
        f.put("cgroup/system.slice/strata.service/memory.max", "max\n");
        f.put("cgroup/system.slice/strata.service/memory.current", 97 * GiB);
        check(f.sample() && f.sample()->available == 25 * GiB && f.sample()->source == MemorySource::meminfo,
              "without the cap the stale MemAvailable is believed");
        auto r = f.sample(100 * GiB);
        check(r && r->available == 300 * MiB && r->source == MemorySource::flag, "the cap minus the root's memory.current");
        check(r->limit == 100 * GiB && r->current == 99 * GiB + 724 * MiB, "reading names the cap and the usage");
        r = f.sample(99 * GiB);
        check(r && r->available == 0 && r->source == MemorySource::flag, "usage over the cap is a known zero");
        r = f.sample(200 * GiB);
        check(r && r->available == 25 * GiB && r->source == MemorySource::meminfo, "a cap above everything else changes nothing");
        // a real limit visible and tighter than the cap: the tighter one
        f.put("cgroup/system.slice/memory.max", 98 * GiB); f.put("cgroup/system.slice/memory.current", 97 * GiB + 512 * MiB);
        r = f.sample(100 * GiB);
        check(r && r->available == 300 * MiB && r->source == MemorySource::flag, "cap tighter than the visible cgroup limit");
        r = f.sample(150 * GiB);
        check(r && r->available == 512 * MiB && r->source == MemorySource::cgroup, "visible cgroup limit tighter than the cap");
        // equal rooms: the operator's number is the one reported
        r = f.sample(99 * GiB + 724 * MiB + 512 * MiB);
        check(r && r->available == 512 * MiB && r->source == MemorySource::flag, "a tie goes to the operator's cap");
        // no memory.current at the root: used = MemTotal - MemAvailable
        fs::remove(f.dir / "cgroup/system.slice/memory.max");
        fs::remove(f.dir / "cgroup/memory.current");
        r = f.sample(100 * GiB);
        check(r && r->available == 5 * GiB && r->source == MemorySource::flag && r->current == 95 * GiB,
              "without the root's memory.current the meminfo-derived usage is used");
        // ... and with no MemTotal either, the usage is unknown: unknown, not a guess
        f.put("meminfo", "MemAvailable: 1024 kB\n");
        check(!f.sample(100 * GiB), "cap with unknown usage is unknown");
        check(f.sample().has_value(), "no cap, no usage needed");
    }
    // malformed or unusable files: without a cap the guard falls back to MemAvailable alone (meminfo source, once-warned);
    // with a cap the sample is unknown (admission refuses), never a guess
    {
        const char* bad_values[] = {"garbage\n", "", "\n", "-5\n", "+5\n", "12x\n", "1 2\n", "18446744073709551616\n",
                                    "0x10\n", "MAX\n"};
        for (const char* file : {"memory.max", "memory.high", "memory.current"}) {
            for (const char* text : bad_values) {
                Fixture f;
                f.meminfo(120 * GiB, 25 * GiB);
                f.self("0::/g");
                f.put("cgroup/g/memory.max", 100 * GiB); f.put("cgroup/g/memory.high", "max\n");
                f.put("cgroup/g/memory.current", 50 * GiB);
                f.put(std::string("cgroup/g/") + file, text);
                {
                    const auto r = f.sample();
                    check(r && r->available == 25 * GiB && r->source == MemorySource::meminfo,
                          "malformed cgroup file without a cap: MemAvailable alone");
                }
                check(!f.sample(100 * GiB), "malformed cgroup file is unknown with the cap too");
            }
        }
        Fixture f;
        f.self("0::/g");
        f.put("cgroup/g/memory.max", 100 * GiB); f.put("cgroup/g/memory.current", 50 * GiB);
        f.meminfo(120 * GiB, 25 * GiB);
        check(f.sample() && f.sample()->available == 25 * GiB, "the fixture itself is sound");
        fs::remove(f.dir / "cgroup/g/memory.current");
        check(f.sample() && f.sample()->available == 25 * GiB && f.sample()->source == MemorySource::meminfo,
              "a finite limit with no usage beside it, no cap: MemAvailable alone");
        check(!f.sample(100 * GiB), "... and with the cap it is unknown");
        f.put("cgroup/g/memory.max", "max\n");
        check(f.sample() && f.sample()->available == 25 * GiB, "no finite limit and no usage: the level is skipped");
        f.put("cgroup/g/memory.current", "max\n");
        check(f.sample() && f.sample()->source == MemorySource::meminfo, "memory.current is never \"max\": MemAvailable alone");
        check(!f.sample(100 * GiB), "... unknown with the cap");
        f.put("cgroup/g/memory.current", 1ull);
        for (const char* text : {"", "MemAvailable: 12 MB\n", "MemAvailable: 1x kB\n", "MemTotal: 5 kB\n",
                                 "MemAvailable: 5 kB\nMemAvailable: 6 kB\n"}) {
            f.put("meminfo", text);
            check(!f.sample(), "malformed MemAvailable is unknown");
        }
        fs::remove(f.dir / "meminfo");
        check(!f.sample(), "no meminfo is unknown");
        f.meminfo(120 * GiB, 25 * GiB);
        f.self("0::/a/../g");
        check(f.sample() && f.sample()->source == MemorySource::meminfo, "a .. component in the cgroup path: MemAvailable alone");
        check(!f.sample(100 * GiB), "a .. component with the cap is refused");
        f.self("garbage without colons");
        check(f.sample() && f.sample()->available == 25 * GiB, "unparsable /proc/self/cgroup lines are ignored");
    }
    // a path from /proc/self/cgroup that this mount does not show: the deepest group that is there (at worst the root)
    {
        Fixture f;
        f.meminfo(120 * GiB, 25 * GiB);
        f.self("0::/lxc/105/ns/deep");
        f.put("cgroup/memory.max", 100 * GiB); f.put("cgroup/memory.current", 99 * GiB);
        auto r = f.sample();
        check(r && r->available == 1 * GiB && r->source == MemorySource::cgroup, "an invisible group falls back to the mount's root");
        f.put("cgroup/lxc/memory.max", 100 * GiB); f.put("cgroup/lxc/memory.current", 99 * GiB + 256 * MiB);
        r = f.sample();
        check(r && r->available == 1 * GiB - 256 * MiB, "the deepest visible prefix is read too");
    }
    // no /proc/self/cgroup file: the mount's root group; no mount: MemAvailable alone
    {
        Fixture f;
        f.meminfo(120 * GiB, 25 * GiB);
        f.put("cgroup/memory.max", 100 * GiB); f.put("cgroup/memory.current", 98 * GiB);
        auto r = f.sample();
        check(r && r->available == 2 * GiB, "no cgroup file: the root group");
        fs::remove_all(f.dir / "cgroup");
        r = f.sample();
        check(r && r->available == 25 * GiB && r->source == MemorySource::meminfo, "no cgroup mount: MemAvailable");
        check(f.sample(100 * GiB) && f.sample(100 * GiB)->source == MemorySource::flag &&
                  f.sample(100 * GiB)->available == 5 * GiB, "no cgroup mount, a cap: meminfo-derived usage");
    }
    // cgroup v1
    {
        Fixture f;
        f.meminfo(120 * GiB, 80 * GiB);
        f.self("12:cpu,cpuacct:/x\n4:memory:/m/n\n1:name=systemd:/x");
        f.put("cgroup/memory/memory.limit_in_bytes", 9223372036854771712ull);
        f.put("cgroup/memory/memory.usage_in_bytes", 90 * GiB);
        f.put("cgroup/memory/m/memory.limit_in_bytes", 64 * GiB); f.put("cgroup/memory/m/memory.usage_in_bytes", 60 * GiB);
        f.put("cgroup/memory/m/n/memory.limit_in_bytes", 9223372036854771712ull);
        f.put("cgroup/memory/m/n/memory.usage_in_bytes", 1 * GiB);
        auto r = f.sample();
        check(r && r->available == 4 * GiB && r->source == MemorySource::cgroup && r->limit == 64 * GiB, "v1: the tightest limit, unlimited skipped");
        r = f.sample(100 * GiB);
        check(r && r->available == 4 * GiB && r->source == MemorySource::cgroup, "v1 with a looser cap");
        r = f.sample(92 * GiB);
        check(r && r->available == 2 * GiB && r->source == MemorySource::flag && r->current == 90 * GiB, "v1 with a tighter cap: the v1 root's usage");
        f.put("cgroup/memory/m/memory.limit_in_bytes", "junk\n");
        check(f.sample() && f.sample()->source == MemorySource::meminfo, "v1 malformed limit, no cap: MemAvailable alone");
        check(!f.sample(100 * GiB), "v1 malformed limit with the cap is unknown");
    }
    // hybrid: a v2 line, but the memory controller lives in the v1 tree
    {
        Fixture f;
        f.meminfo(120 * GiB, 80 * GiB);
        f.self("5:memory:/g\n0::/g");
        f.put("cgroup/cgroup.controllers", "cpu\n");
        f.put("cgroup/g/cpu.max", "max 100000\n");
        f.put("cgroup/memory/g/memory.limit_in_bytes", 10 * GiB); f.put("cgroup/memory/g/memory.usage_in_bytes", 9 * GiB);
        auto r = f.sample();
        check(r && r->available == 1 * GiB && r->source == MemorySource::cgroup, "hybrid falls through to v1");
        f.put("cgroup/memory.max", "max\n"); f.put("cgroup/memory.current", 1 * GiB);
        r = f.sample();
        check(r && r->available == 80 * GiB, "v2 memory files win over v1 when both exist");
    }
    // the one credited kind of reclaimable memory: clean inactive file cache from the same group's memory.stat
    {
        const auto stat = [](uint64_t inactive, uint64_t active, uint64_t shmem, uint64_t dirty = 0, uint64_t wb = 0) {
            return "anon 1\nfile 2\nshmem " + std::to_string(shmem) + "\ninactive_file " + std::to_string(inactive) +
                   "\nactive_file " + std::to_string(active) + "\nfile_dirty " + std::to_string(dirty) +
                   "\nfile_writeback " + std::to_string(wb) + "\nunevictable 5\n";
        };
        Fixture f;
        f.meminfo(120 * GiB, 25 * GiB);
        f.self("0::/system.slice/strata.service");
        f.put("cgroup/memory.max", "max\n"); f.put("cgroup/memory.high", "max\n");
        f.put("cgroup/memory.current", 98 * GiB + 324 * MiB);
        f.put("cgroup/memory.stat", stat(3 * GiB + 800 * MiB, 27 * GiB, 65 * GiB));
        f.put("cgroup/system.slice/strata.service/memory.max", "max\n");
        f.put("cgroup/system.slice/strata.service/memory.current", 90 * GiB);
        // the operator's cap: the root's memory.current less the root's inactive_file; active_file and shmem are not credited
        auto r = f.sample(99 * GiB);
        check(r && r->source == MemorySource::flag && r->available == 99 * GiB - (98 * GiB + 324 * MiB - 3 * GiB - 800 * MiB),
              "the cap path credits the root's inactive_file");
        check(r->credit == 3 * GiB + 800 * MiB && r->current == 98 * GiB + 324 * MiB, "reading shows the credit and the raw usage");
        f.put("cgroup/memory.stat", stat(0, 27 * GiB, 65 * GiB));
        r = f.sample(99 * GiB);
        check(r && r->available == 700 * MiB && r->credit == 0, "only inactive_file counts: active_file and shmem give nothing");
        // dirty and writeback pages are not clean: taken off the credit
        f.put("cgroup/memory.stat", stat(3 * GiB, 0, 0, 1 * GiB, 512 * MiB));
        r = f.sample(99 * GiB);
        check(r && r->credit == 1 * GiB + 512 * MiB && r->available == 700 * MiB + 1 * GiB + 512 * MiB, "dirty and writeback come off");
        f.put("cgroup/memory.stat", stat(3 * GiB, 0, 0, 2 * GiB, 2 * GiB));
        r = f.sample(99 * GiB);
        check(r && r->credit == 0 && r->available == 700 * MiB, "dirty plus writeback over inactive_file leaves no credit");
        // inactive_file larger than the usage: the credit stops at the usage, headroom never exceeds the cap
        f.put("cgroup/memory.stat", stat(500 * GiB, 0, 0));
        f.meminfo(300 * GiB, 200 * GiB);   // so the cap's room (not MemAvailable) is the smallest term
        r = f.sample(99 * GiB);
        check(r && r->credit == 98 * GiB + 324 * MiB && r->available == 99 * GiB && r->source == MemorySource::flag,
              "inactive_file above the usage cannot lift headroom over the cap");
        // no memory.stat, or one that cannot be read: no credit, still a sample
        fs::remove(f.dir / "cgroup/memory.stat");
        r = f.sample(99 * GiB);
        check(r && r->available == 700 * MiB && r->credit == 0, "no memory.stat: no credit");
        for (const char* text : {"", "garbage\n", "inactive_file\n", "inactive_file -1\n", "inactive_file 12x\n",
                                 "inactive_file 1\ninactive_file 2\n", "inactive_file 99999999999999999999\n",
                                 "inactive_file 4096\nfile_dirty zz\n"}) {
            f.put("cgroup/memory.stat", text);
            r = f.sample(99 * GiB);
            check(r && r->available == 700 * MiB && r->credit == 0, "unparsable memory.stat: no credit, not unknown");
        }
        // a cgroup limit's own group is credited from its own memory.stat
        f.put("cgroup/memory.stat", stat(0, 0, 0));
        f.put("cgroup/system.slice/strata.service/memory.max", 92 * GiB);
        f.put("cgroup/system.slice/strata.service/memory.stat", stat(1 * GiB, 0, 0));
        r = f.sample();
        check(r && r->source == MemorySource::cgroup && r->available == 3 * GiB && r->credit == 1 * GiB &&
                  r->limit == 92 * GiB && r->current == 90 * GiB, "a cgroup limit credits its own group's inactive_file");
        f.put("cgroup/system.slice/strata.service/memory.stat", stat(500 * GiB, 0, 0));
        f.meminfo(300 * GiB, 200 * GiB);
        r = f.sample();
        check(r && r->available == 92 * GiB && r->credit == 90 * GiB, "cgroup term: the credit stops at the usage too");
        // cgroup v1: total_inactive_file (and total_dirty / total_writeback)
        Fixture g;
        g.meminfo(120 * GiB, 80 * GiB);
        g.self("4:memory:/m");
        g.put("cgroup/memory/m/memory.limit_in_bytes", 64 * GiB); g.put("cgroup/memory/m/memory.usage_in_bytes", 62 * GiB);
        g.put("cgroup/memory/memory.limit_in_bytes", 9223372036854771712ull); g.put("cgroup/memory/memory.usage_in_bytes", 62 * GiB);
        g.put("cgroup/memory/m/memory.stat", "cache 1\ninactive_file 7\ntotal_inactive_file " + std::to_string(3 * GiB) +
                                                  "\ntotal_dirty " + std::to_string(1 * GiB) + "\ntotal_writeback 0\n");
        r = g.sample();
        check(r && r->available == 2 * GiB + 2 * GiB && r->credit == 2 * GiB && r->source == MemorySource::cgroup,
              "v1 credits total_inactive_file less total_dirty");
        g.put("cgroup/memory/m/memory.stat", "inactive_file " + std::to_string(3 * GiB) + "\n");
        r = g.sample();
        check(r && r->available == 2 * GiB && r->credit == 0, "v1 ignores the non-total inactive_file");
    }
    // the process-wide setting and the real files
    {
        check(memory_limit_bytes() == 0, "no cap by default");
        set_memory_limit_bytes(123 * MiB);
        check(memory_limit_bytes() == 123 * MiB, "cap round trip");
#if defined(__linux__)
        set_memory_limit_bytes(1);   // one byte: whatever this machine uses, nothing is left
        const auto r = sample_host_memory();
        check(!r || (r->available == 0 && r->source == MemorySource::flag && available_host_bytes() == 0u),
              "the real files honour the process-wide cap");
#endif
        set_memory_limit_bytes(0);
        const auto none = available_host_bytes();
        check(!none || conversation_memory_admit(none, 0, 0), "the real files without a cap: bytes or unknown");
        check(std::string(memory_source_name(MemorySource::flag)) == "flag" &&
                  std::string(memory_source_name(MemorySource::cgroup)) == "cgroup" &&
                  std::string(memory_source_name(MemorySource::meminfo)) == "meminfo", "source names");
    }
    std::printf("memory_guard_test: %d checks passed\n", checks);
    return 0;
}
