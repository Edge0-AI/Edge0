// src/platform/memory_test.cpp - plan v0.3 P0.1: lock_resident on a 256 MiB region (CPU only, no GPU, no model).
#include "strata/platform/memory.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#endif

namespace {
uint64_t gib(double g) { return (uint64_t) (g * 1073741824.0); }

// #577: the file tier's unbuffered choice (true = the file cache keeps its reads, so they stay buffered)
int file_cache_keeps_cases() {
    using strata::platform::file_cache_keeps;
    struct Case { const char* what; double avail, arena, read; bool keeps; };
    const double experts = 71.7;   // UD-Q4_K_XL's routed experts
    const Case cases[] = {
        // the reporter's 96 GB PC (83.0 GiB available at start): 0.1.38 passed the budget asked for and every
        // shard's bytes (103.7 GiB) and went unbuffered
        {"96 GB, 0.1.38's inputs (budget 72, all shards)", 83.0, 72.0, 103.7, false},
        {"96 GB, budget 72, before the copy (arena = all experts)", 83.0, experts, 0.0, true},
        {"96 GB, budget 72, copy built (48.94 resident)", 83.0 - 48.94, 0.0, experts - 48.94, true},
        {"96 GB, budget 40, before the copy", 83.0, 40.0, experts - 40.0, true},
        {"96 GB, budget 40, copy built", 83.0 - 40.0, 0.0, experts - 40.0, true},
        // #357/#362's low-RAM PCs stay unbuffered
        {"64 GB, budget 40, before the copy", 55.0, 40.0, experts - 40.0, false},
        {"64 GB, budget 40, copy built", 15.0, 0.0, experts - 40.0, false},
        {"32 GB, budget 16, before the copy", 26.0, 16.0, experts - 16.0, false},
        {"32 GB, budget 16, copy built", 10.0, 0.0, experts - 16.0, false},
        {"32 GB, Q2_0 pack (34 GiB experts), budget 8", 26.0, 8.0, 34.0 - 8.0, false},
        {"nothing available", 0.0, 0.0, 1.0, false},
        {"arena larger than the RAM", 10.0, 20.0, 0.0, true},
    };
    int fail = 0;
    for (const Case& c : cases) {
        const bool got = file_cache_keeps(gib(c.avail), gib(c.arena), gib(c.read));
        std::printf("file_cache_keeps %-58s -> %s%s\n", c.what, got ? "buffered" : "unbuffered",
                    got == c.keeps ? "" : "   <-- WRONG");
        fail |= got != c.keeps;
    }
    // #1194, the Linux file tier: a skewed read set, so a twentieth of it (at least 1.5 GiB) is room enough
    const Case linux_cases[] = {
        {"#1194: 31 GB + 12 GB GPU, budget 18 built (6.4 avail, 13.6 read)", 6.4, 0.0, 13.6, true},
        {"#1194: before the copy (avail 25, arena 18, read 13.6)", 25.0, 18.0, 13.6, true},
        {"32 GB cgroup, budget 18, copy built (12.1 avail, 22 read)", 12.1, 0.0, 22.0, true},
        {"32 GB cgroup, budget 24, copy built (3.8 avail, 16 read)", 3.8, 0.0, 16.0, true},
        {"20 GB cgroup, budget 12, copy built (4.7 avail, 28 read)", 4.7, 0.0, 28.0, true},
        {"12 GB cgroup, budget 6, copy built (3.5 avail, 34 read)", 3.5, 0.0, 34.0, true},
        {"no room beside the margin (1 avail, 47 read)", 1.0, 0.0, 47.0, false},
        {"room under the 1.5 GiB floor (2.4 avail, 47 read)", 2.4, 0.0, 47.0, false},
        {"room a twentieth of the reads (3.4 avail, 47 read)", 3.4, 0.0, 47.0, true},
        {"room just under that (3.3 avail, 47 read)", 3.3, 0.0, 47.0, false},
        {"a big read set needs its twentieth (4 avail, 200 read)", 4.0, 0.0, 200.0, false},
        {"a big read set with that room (12 avail, 200 read)", 12.0, 0.0, 200.0, true},
        {"read less than the floor (2.5 avail, 1 read)", 2.5, 0.0, 1.0, true},
        {"nothing available", 0.0, 0.0, 1.0, false},
        {"arena larger than the RAM", 10.0, 20.0, 5.0, false},
    };
    for (const Case& c : linux_cases) {
        const bool got = file_cache_keeps(gib(c.avail), gib(c.arena), gib(c.read), 1ull << 30, 0.05, 3ull << 29);
        std::printf("file_cache_keeps (Linux file tier) %-52s -> %s%s\n", c.what, got ? "buffered" : "unbuffered",
                    got == c.keeps ? "" : "   <-- WRONG");
        fail |= got != c.keeps;
    }
    return fail;
}
}  // namespace

// #3237: the drive probe's decision
int file_tier_prefers_direct_cases() {
    using strata::platform::file_tier_prefers_direct;
    struct Case { const char* what; double held, direct, cached; bool direct_wins; };
    const Case cases[] = {
        {"#3237: 13% held, 8.18 vs 4.93 GB/s", 0.13, 8.18e9, 4.93e9, true},
        {"P1004 (measured): 1.65 vs 0.81 GB/s, 19% held: a slow drive keeps the cache", 0.19, 1.65e9, 0.81e9, false},
        {"a fast drive, the same ratio as P1004 (8 vs 4 GB/s)", 0.19, 8.0e9, 4.0e9, true},
        {"just under the 4 GB/s floor", 0.10, 3.9e9, 1.0e9, false},
        {"a third held, direct 1.4x: not enough", 0.35, 5.6e9, 4.0e9, false},
        {"a third held, direct 1.6x", 0.35, 6.4e9, 4.0e9, true},
        {"under 30% held, direct 1.3x", 0.20, 5.2e9, 4.0e9, true},
        {"under 30% held, direct 1.2x", 0.20, 4.8e9, 4.0e9, false},
        {"half held or more: the cache, however fast the drive", 0.5, 9e9, 1e9, false},
        {"no cached-path figure", 0.1, 5e9, 0.0, false},
        {"no direct figure", 0.1, 0.0, 5e9, false},
        {"a hard disk (direct 2x, under the floor): the cache", 0.1, 0.2e9, 0.1e9, false},
    };
    int fail = 0;
    for (const Case& c : cases) {
        const bool got = file_tier_prefers_direct(c.held, c.direct, c.cached);
        std::printf("file_tier_prefers_direct %-62s -> %s%s\n", c.what, got ? "unbuffered" : "cache",
                    got == c.direct_wins ? "" : "   <-- WRONG");
        fail |= got != c.direct_wins;
    }
    return fail;
}

int main() {
    int fail_keeps = file_cache_keeps_cases() | file_tier_prefers_direct_cases();
    const uint64_t bytes = 256ull << 20;
    void* p = std::malloc(bytes);
    if (p == nullptr) return 2;
    std::memset(p, 1, bytes);
    const strata::platform::LockResult r = strata::platform::lock_resident(p, bytes);
    std::printf("lock_resident: ok=%d locked=%llu MiB (%s)\n", (int) r.ok, (unsigned long long) (r.locked_bytes >> 20),
                r.note.c_str());
    int fail = !(r.ok && r.locked_bytes == bytes) | fail_keeps;
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc);
    std::printf("working set %llu MiB\n", (unsigned long long) (pmc.WorkingSetSize >> 20));
    fail |= pmc.WorkingSetSize < bytes;
#endif
    strata::platform::unlock_resident(p, bytes);
    std::free(p);
    std::printf("platform_memory_test: %s\n", fail ? "FAILED" : "OK");
    return fail;
}
