// metal/src/engine/arena_budget.h - how big the SSD-streaming expert arena may be (stream.h), and how much the engine may
// hold resident in all, as plain arithmetic so it can be tested without a Mac (metal/tests/arena_budget_test.cpp).
//
// The arena is anonymous memory the GPU needs resident while a command buffer runs.  It must leave room for macOS and the
// desktop, and for the other big memory the engine itself uses while it runs:
//   keep      macOS and the desktop: max(7 GB, 15% of the RAM) - STRATA_METAL_KEEP_FREE_GB sets it.  The M5 Max
//             (128 GB) froze with an 89 GB arena and 7 GB kept beside a 50 GB n-gram table; with 15% it keeps 20.6 GB.
//             The 32 GB M4 (IQ3_XXS after a 6K prompt, 5 interleaved runs, memory_pressure every 3 s, vm_stat
//             swapouts): 7 GB kept + 2 GB reserve 5.93 tok/s (arena ~8.5 GB, least free 50%), 7 + 0.5 6.59 (~10 GB,
//             49%), 5 + 0.5 7.47 (~12 GB, 43%); no swap-out in any run.  But a 34-minute server soak with 5 GB kept
//             swapped out 93 MB while its 11.7 GB arena filled (none after): 7 GB again.
//   ple       the n-gram table: mapped whole in RAM (--ple-io ram) it is all of it; read row by row (the default,
//             F_NOCACHE: no file cache) only its bounded row cache, 1M rows of at most 192 bytes: 0.5 GB reserved.
//   GPU       the Metal working set less what is already allocated, less 1 GB.
//   footprint the engine's WHOLE anonymous footprint - arena + every Metal buffer + what the process holds beside them
//             (host memory) - stays under footprint_cap() (below): on a big Mac 60% of the RAM.  The 67% rule it replaces
//             counted only the Metal buffers (cx.allocated(), 7.5 GB) while the process held 11-15 GB, and 67% was too
//             much anyway (see kLeaveShare).
// STRATA_METAL_ARENA_GB sets the arena outright (the engine says when that goes over the cap; it is the user's choice).
// The arena is placed just before it is filled, and the memory macOS can give is asked again then: when it has dropped
// (another program), the arena shrinks (arena_shrink).
// "Available" (arena_avail) counts free and purgeable pages and the file cache macOS drops without writing anything:
// inactive pages, but no more of them than there are file-backed pages.  Inactive anonymous memory (other programs')
// is left out: macOS frees it only by compressing or swapping it (the soak's 93 MB swapped out while an arena planned
// with every inactive page filled).  Free + purgeable alone is too little: at the engine's start the model file's own
// pages fill the cache (6.4 GB free + purgeable on the 32 GB M4 against 15 GB a minute later).  The arena is filled in
// steps (STRATA_METAL_ARENA_STEP_GB, default 1.5): before each step the free memory is asked again, and the fill stops -
// the rest of the arena's slots unused - when a step would take macOS below what it keeps (arena_grow).
#pragma once

#include <algorithm>
#include <cstddef>

namespace strata::metal_engine {

struct ArenaMem {
    double phys = 0;          // bytes of RAM
    double working_set = 0;   // the Metal device's recommended working set
    double allocated = 0;     // already allocated by the engine (Metal buffers)
    double available = 0;     // what macOS can give now without paging anything out (arena_avail)
    double ple_mapped = 0;    // the n-gram table's bytes when it is mapped whole in RAM, else 0
    double proc_footprint = 0;   // what the process holds right now, whatever it is (task_info phys_footprint); 0: unknown
    double filecache_min = 0;    // sysctl vm.vm_page_filecache_min in bytes: the file cache macOS keeps before it compresses (0: unknown)
    double keep_gb = -1;      // STRATA_METAL_KEEP_FREE_GB (< 0: unset)
    double arena_gb = -1;     // STRATA_METAL_ARENA_GB (< 0: unset)
    double leave_share = -1;  // STRATA_METAL_LEAVE_SHARE: the share of the RAM a big Mac leaves to macOS (< 0: kLeaveShare)
};

constexpr double kGB = 1e9;

// Why a cap at all.  macOS keeps a floor of file cache (vm.vm_page_filecache_min: 37.2 GB, 27% of the 128 GB M5 Max) and,
// when anonymous memory wants more than the rest, compresses the anonymous pages (then swaps) instead of dropping file
// cache below it.  The arena is anonymous and the GPU wires it only while a command buffer runs: while it is being filled,
// or idle, it is ordinary pageable memory.  Measured on the M5 Max (UD-Q6_K_XL streaming, memwatch every second):
//   the old build, a 100 GB arena (what it planned after a fresh boot): 19 GB of the arena compressed in 7 s, 29,000 pages
//     swapped out, pressure "warn" - the signature of the two panics (compressor at 3.9M pages, free memory 13 MB, the
//     engine 95.3 and 89.9 GiB resident, watchdogd starved for 92 s);
//   the old build, auto arena 69 GB, a 40 GB file write and a 16 GB program beside it: compressor 21k -> 4.07M pages in
//     10 s, swapouts +29,000, pressure warn (stopped by the watchdog);
//   footprint cap at 70% of the RAM (wired 94.5 GB), 40 GB file write: compressor 1.8M pages, swapouts +12,000, stopped;
//   35% left (89 GB): clear with the write alone (compressor 0.28M pages, 316 pages swapped), not with more beside it;
//   40% left (cap 82.5 GB, footprint 77 GB): 40 GB write + 16 GB program, swapouts 0, compressor 0.3M pages, decode and
//     prompt speed equal to the 80 GB arena's (51 tok/s story, 1,020 tok/s for 15K).
// And the arena is wired (mlock, stream.cpp) once it is filled: unwired it is pageable whenever the GPU is idle (the GPU's own wiring
// ends seconds after its last command buffer; a Metal residency set did not hold it either), and a 20 GB file written beside the
// idle engine took tens of GB of the arena into the compressor and froze the Mac (a third watchdog panic, in a control run of this
// work: wired 2.4 GB, file cache 50 GB, compressor 1.9M pages, 45M compressions).  Wired, the same write (full speed, 20 GB in 2.5 s)
// raised the compressor by 0.7M pages (the ~11 GB of the engine that is not arena), swapped out 408 pages and changed nothing else.
// A pack that stays RESIDENT is different: its bytes are clean file pages (macOS drops them, never compresses them) and they
// count as file cache, not against the floor.  UD-Q4_K_XL resident on the M5 Max (wired 91.7 GB, 67%), 40 GB write + 16 GB
// program beside it: swapouts 0, compressor 5,500 pages, 63 tok/s.  So two caps:
//   footprint_cap    anonymous memory (the streaming arena, Metal buffers, host memory): kLeaveShare = 40% of the RAM left
//                    on a big Mac (at least the file-cache floor + 12 GB)
//   resident_cap     a resident pack's mapped bytes: kResidentLeaveShare = 30% left (the old rule, measured as above)
constexpr double kLeaveShare = 0.40;
constexpr double kResidentLeaveShare = 0.30;
constexpr double kBigMac = 64.0 * kGB;
constexpr double kFileCacheHeadroom = 12.0 * kGB;   // beside the file-cache floor: macOS itself (~6 GB) and one more program

inline double arena_keep(const ArenaMem& m) {
    return m.keep_gb >= 0 ? m.keep_gb * kGB : std::max(7.0 * kGB, 0.15 * m.phys);
}

/// The most the engine may hold as anonymous memory, all counted (arena + Metal buffers + host memory).
inline double footprint_cap(const ArenaMem& m) {
    const double share = m.leave_share >= 0 ? m.leave_share : kLeaveShare;
    double leave = arena_keep(m);
    if (m.phys >= kBigMac) leave = std::max({leave, share * m.phys, m.filecache_min > 0 ? m.filecache_min + kFileCacheHeadroom : 0.0});
    return std::min(m.working_set, m.phys - leave);
}

/// The most a resident pack (its mapped bytes + the engine's buffers) may hold.
inline double resident_cap(const ArenaMem& m) {
    double leave = arena_keep(m);
    if (m.phys >= kBigMac) leave = std::max(leave, kResidentLeaveShare * m.phys);
    return std::min(m.working_set, m.phys - leave);
}

/// What the engine holds before the arena: the Metal buffers (alloc + wrap) or, when the process holds more, all of it
/// (host memory, mapped tables).
inline double footprint_base(const ArenaMem& m) { return std::max(m.allocated, m.proc_footprint); }

/// Bytes macOS can give without compressing or swapping: free + purgeable + the clean file cache (inactive and
/// speculative pages, at most the file-backed ones).
inline double arena_avail(double free_b, double purgeable_b, double inactive_b, double speculative_b, double file_b) {
    return free_b + purgeable_b + std::min(inactive_b + speculative_b, file_b);
}

inline double arena_ple_reserve(const ArenaMem& m) { return m.ple_mapped > 0 ? m.ple_mapped : 0.5 * kGB; }

/// The arena's bytes (at least 1 GB: a smaller arena still runs, at a low hit rate).
inline double arena_budget(const ArenaMem& m) {
    if (m.arena_gb >= 0) return m.arena_gb * kGB;
    const double gpu = m.working_set - footprint_base(m) - 1.0 * kGB;
    const double ram = m.available - arena_keep(m) - arena_ple_reserve(m);
    const double foot = footprint_cap(m) - footprint_base(m) - arena_ple_reserve(m);   // the whole footprint under the cap
    return std::max(1.0 * kGB, std::min(std::min(gpu, ram), foot));
}

/// The plan as numbers, for the log line and the tests: what the engine will hold with `arena` bytes of arena.
struct FootprintPlan {
    double arena = 0, base = 0, ple = 0, total = 0, cap = 0;
    bool over() const { return total > cap + 1e6; }
};
inline FootprintPlan footprint_plan(const ArenaMem& m, double arena) {
    FootprintPlan p;
    p.arena = arena;
    p.base = footprint_base(m);
    p.ple = arena_ple_reserve(m);
    p.total = p.arena + p.base + p.ple;
    p.cap = footprint_cap(m);
    return p;
}

/// A pack that stays resident (no SSD streaming) holds its mapped bytes plus `overhead` (the prompt buffers, KV, the draft
/// layer, ...): does that fit under the resident cap?  (stream.cpp: ExpertStream::wanted)
inline bool resident_fits(double phys, double working_set, double model_bytes, double overhead) {
    ArenaMem m;
    m.phys = phys;
    m.working_set = working_set;
    return model_bytes + overhead <= resident_cap(m);
}

/// Just before the arena is filled: the bytes it may still have when macOS can now give `available_now` (it shrinks,
/// never grows; the explicit STRATA_METAL_ARENA_GB is kept as it is).
inline double arena_shrink(const ArenaMem& m, double planned, double available_now) {
    if (m.arena_gb >= 0) return planned;
    ArenaMem now = m;
    now.available = available_now;
    return std::min(planned, arena_budget(now));
}

/// The stepped fill: may the next `step` bytes of the arena be filled while macOS has `available_now` free + purgeable?
/// (the explicit STRATA_METAL_ARENA_GB always fills)
inline bool arena_grow(const ArenaMem& m, double available_now, double step) {
    if (m.arena_gb >= 0) return true;
    return available_now - step >= arena_keep(m) + arena_ple_reserve(m);
}

/// The fill as a plan, for the test: the pairs' bytes in rank order, steps of `step` bytes, `avail(filled)` the free
/// memory once `filled` bytes are in.  Returns how many pairs are filled.
template <class Avail>
inline size_t arena_fill_plan(const ArenaMem& m, const double* cost, size_t n, double step, Avail avail) {
    size_t i = 0;
    double filled = 0;
    while (i < n) {
        double s = 0;
        size_t j = i;
        while (j < n && (s < step || j == i)) s += cost[j++];
        if (!arena_grow(m, avail(filled), s)) break;
        filled += s;
        i = j;
    }
    return i;
}

}  // namespace strata::metal_engine
