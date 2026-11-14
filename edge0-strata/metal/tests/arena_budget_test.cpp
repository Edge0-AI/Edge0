// metal/tests/arena_budget_test.cpp - the SSD-streaming arena's size rule (src/engine/arena_budget.h), on the sizes of
// the Macs it was measured on.  No GPU: plain arithmetic.  Exit 0 = every check passed.
#include "../src/engine/arena_budget.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace strata::metal_engine;

static int failures = 0;
static void check(bool ok, const char* what, double got) {
    std::printf("%s %-70s %.2f GB\n", ok ? "ok  " : "FAIL", what, got / kGB);
    failures += !ok;
}

int main() {
    // the M5 Max as it froze: 137.4 GB RAM, 115.4 GB working set, 7.0 GB allocated, 97.7 GB available
    ArenaMem m5{137.4 * kGB, 115.4 * kGB, 7.0 * kGB, 97.7 * kGB};
    check(std::fabs(arena_keep(m5) - 0.15 * 137.4 * kGB) < 1e6, "M5 Max: macOS keeps 15% of the RAM (20.6 GB)", arena_keep(m5));
    const double b5 = arena_budget(m5);
    check(b5 <= 97.7 * kGB - 20.6 * kGB - 0.5 * kGB + 1e6, "M5 Max: the arena leaves >= 20 GB + the n-gram reserve", b5);
    check(b5 > 70 * kGB, "M5 Max: still a large arena (UD-Q5's 98 GB of experts mostly cached)", b5);
    // the n-gram table mapped whole (--ple-io ram, 50.7 GB): it comes out of the arena
    ArenaMem m5r = m5;
    m5r.ple_mapped = 50.7 * kGB;
    check(arena_budget(m5r) <= b5 - 48.0 * kGB, "M5 Max, table in RAM: the arena gives up the table's 50.7 GB", arena_budget(m5r));
    // the M4 Mac mini, 32 GB: the 7 GB floor (15% would be 5.16 GB)
    ArenaMem m4{34.4 * kGB, 26.8 * kGB, 3.3 * kGB, 17.8 * kGB};
    check(std::fabs(arena_keep(m4) - 7.0 * kGB) < 1e6, "M4 32 GB: macOS keeps the 7 GB floor", arena_keep(m4));
    check(std::fabs(arena_budget(m4) - (17.8 - 7.0 - 0.5) * kGB) < 1e6, "M4 32 GB: available - 7 - 0.5", arena_budget(m4));
    ArenaMem m16{17.2 * kGB, 11 * kGB, 3 * kGB, 9 * kGB};
    check(std::fabs(arena_keep(m16) - 7.0 * kGB) < 1e6, "a 16 GB Mac: the 7 GB floor", arena_keep(m16));
    // the overrides
    ArenaMem o = m4;
    o.keep_gb = 12.0;
    check(std::fabs(arena_budget(o) - (17.8 - 12.0 - 0.5) * kGB) < 1e6, "STRATA_METAL_KEEP_FREE_GB=12", arena_budget(o));
    o.arena_gb = 10.0;
    check(std::fabs(arena_budget(o) - 10.0 * kGB) < 1e6, "STRATA_METAL_ARENA_GB=10 wins", arena_budget(o));
    check(std::fabs(arena_shrink(o, 10.0 * kGB, 1.0 * kGB) - 10.0 * kGB) < 1e6, "an explicit arena is not shrunk", 10.0 * kGB);
    // never below 1 GB; the GPU working set bounds it too
    ArenaMem tiny{16 * kGB, 10 * kGB, 9.5 * kGB, 8 * kGB};
    check(std::fabs(arena_budget(tiny) - 1.0 * kGB) < 1e6, "a 16 GB Mac with nothing left: 1 GB (low hit rate)", arena_budget(tiny));
    // the caps.  Anonymous memory (arena, Metal buffers, host memory): 40% of the RAM left on a big Mac; a resident pack: 30%.
    check(std::fabs(footprint_cap(m5) - 0.60 * 137.4 * kGB) < 1e6, "M5 Max: the footprint cap is 60% of the RAM (82.4 GB; 40% left)", footprint_cap(m5));
    check(std::fabs(resident_cap(m5) - 0.70 * 137.4 * kGB) < 1e6, "M5 Max: a resident pack may hold 70% (30% left)", resident_cap(m5));
    check(std::fabs(footprint_cap(m4) - 26.8 * kGB) < 1e6, "M4 32 GB: the cap is its 26.8 GB working set (7 GB left, as measured)", footprint_cap(m4));
    check(std::fabs(resident_cap(m4) - 26.8 * kGB) < 1e6, "M4 32 GB: same for a resident pack", resident_cap(m4));
    ArenaMem fc = m5;   // the kernel's file-cache floor (37.2 GB here) + 12 GB: never less than that is left
    fc.filecache_min = 37.2 * kGB;
    check(std::fabs(footprint_cap(fc) - 0.60 * 137.4 * kGB) < 1e6, "M5 Max with its 37.2 GB file-cache floor: 49.2 GB < 55 GB, the share binds", footprint_cap(fc));
    fc.filecache_min = 50.0 * kGB;   // a macOS that keeps more: 62 GB left
    check(std::fabs(footprint_cap(fc) - (137.4 - 62.0) * kGB) < 1e6, "a larger file-cache floor (50 GB): the cap follows it (75.4 GB)", footprint_cap(fc));
    ArenaMem gpu = m5;
    gpu.available = 200 * kGB;
    check(std::fabs(arena_budget(gpu) - (0.60 * 137.4 - 7.0 - 0.5) * kGB) < 1e6, "plenty free: the footprint cap bounds it (cap - 7 GB held - 0.5 reserve)", arena_budget(gpu));
    ArenaMem q6 = m5;   // UD-Q6_K_XL after a reboot: 7.5 GB allocated, 125 GB "available" (cache); 84 GB ran, 90 GB swapped
    q6.allocated = 7.5 * kGB;
    q6.available = 125 * kGB;
    check(arena_budget(q6) <= 74.5 * kGB && arena_budget(q6) > 74.0 * kGB, "M5 Max, UD-Q6 with 125 GB available: 74.4 GB, not 104", arena_budget(q6));
    // the process holds more than the Metal buffers (host memory): the larger count is the base.  UD-Q6 on the M5 held 11.6-15.1 GB
    // against 7.5 GB of Metal buffers; the arena gives up the difference, so the total stays at the cap
    ArenaMem held = q6;
    held.proc_footprint = 15.1 * kGB;
    check(std::fabs(arena_budget(held) - (arena_budget(q6) - 7.6 * kGB)) < 1e6, "the process holds 15.1 GB against 7.5 GB of Metal buffers: the arena gives up the 7.6 GB", arena_budget(held));
    FootprintPlan fp = footprint_plan(held, arena_budget(held));
    check(!fp.over() && std::fabs(fp.total - fp.cap) < 1e6, "the planned total (arena + held + n-gram reserve) ends at the cap, not over it", fp.total);
    check(137.4 * kGB - fp.total >= 0.40 * 137.4 * kGB - 1e6, "RAM less the planned total is at least 40% of the RAM", 137.4 * kGB - fp.total);
    check(std::fabs(fp.total - 82.4 * kGB) < 0.1 * kGB && std::fabs(fp.arena - 66.8 * kGB) < 0.1 * kGB, "= the measured 66.8 GB arena / 82.4 GB total of the clear run", fp.total);
    // a footprint smaller than the Metal buffers never makes the arena grow
    ArenaMem less = q6;
    less.proc_footprint = 1.0 * kGB;
    check(std::fabs(arena_budget(less) - arena_budget(q6)) < 1e6, "a smaller process count than the Metal buffers changes nothing", arena_budget(less));
    // little free memory (a 24 GB program already running): the free-memory rule binds below the cap
    ArenaMem busy = held;
    busy.available = 60 * kGB;
    check(arena_budget(busy) < arena_budget(held) && std::fabs(arena_budget(busy) - (60.0 - 20.61 - 0.5) * kGB) < 0.1 * kGB, "a program already holds memory: the free-memory rule binds (38.9 GB)", arena_budget(busy));
    // the explicit arena is kept (never forced) but the plan says it is over the cap
    ArenaMem ex = q6;
    ex.arena_gb = 90.0;
    fp = footprint_plan(ex, arena_budget(ex));
    check(std::fabs(fp.arena - 90.0 * kGB) < 1e6 && fp.over(), "STRATA_METAL_ARENA_GB=90 on the M5: kept, and flagged over the cap", fp.total);
    ex.arena_gb = 60.0;
    check(!footprint_plan(ex, arena_budget(ex)).over(), "STRATA_METAL_ARENA_GB=60: under the cap", footprint_plan(ex, arena_budget(ex)).total);
    ArenaMem sh = q6;   // STRATA_METAL_LEAVE_SHARE (for measuring)
    sh.leave_share = 0.30;
    check(std::fabs(footprint_cap(sh) - 0.70 * 137.4 * kGB) < 1e6, "STRATA_METAL_LEAVE_SHARE=0.30: the old 30% (96.2 GB)", footprint_cap(sh));
    // the working set still binds where it is tighter than the RAM share
    ArenaMem m64{68.7 * kGB, 48.0 * kGB, 5.0 * kGB, 55.0 * kGB};
    check(std::fabs(footprint_cap(m64) - 41.2 * kGB) < 0.1 * kGB, "64 GB Mac: 60% of the RAM (41.2 GB), under the 48 GB working set", footprint_cap(m64));
    check(std::fabs(resident_cap(m64) - 48.0 * kGB) < 1e6, "64 GB Mac: a resident pack: the 48 GB working set (30% left would give 48.1)", resident_cap(m64));
    ArenaMem m96{103.1 * kGB, 77.3 * kGB, 6.0 * kGB, 90.0 * kGB};
    check(std::fabs(footprint_cap(m96) - 0.60 * 103.1 * kGB) < 1e6, "96 GB Mac: 60% of the RAM (61.9 GB)", footprint_cap(m96));
    check(std::fabs(resident_cap(m96) - 0.70 * 103.1 * kGB) < 1e6, "96 GB Mac: resident 70% (72.2 GB), under its 77.3 GB working set", resident_cap(m96));
    // a resident pack: its bytes + 12 GiB must fit the resident cap (stream.cpp: wanted)
    const double G12 = 12.0 * 1073741824.0;
    check(resident_fits(137.4 * kGB, 115.4 * kGB, 54.8 * kGB, G12), "M5 Max: IQ3_S (54.8 GB) stays resident", 54.8 * kGB);
    check(resident_fits(137.4 * kGB, 115.4 * kGB, 82.5 * kGB, G12), "M5 Max: UD-Q4_K_XL (82.5 GB) stays resident (measured: wired 91.7 GB, nothing compressed)", 82.5 * kGB);
    check(!resident_fits(137.4 * kGB, 115.4 * kGB, 103.9 * kGB, G12), "M5 Max: UD-Q5_K_XL (103.9 GB) streams", 103.9 * kGB);
    check(!resident_fits(34.4 * kGB, 26.8 * kGB, 37.6 * kGB, G12), "M4 32 GB: Q2_0 (37.6 GB) streams", 37.6 * kGB);
    // memory taken between planning and committing: it shrinks, never grows
    const double s1 = arena_shrink(m5, b5, 60.0 * kGB);
    check(s1 < b5 && std::fabs(s1 - (60.0 - 20.61 - 0.5) * kGB) < 0.1 * kGB, "less free when committing: it shrinks", s1);
    check(std::fabs(arena_shrink(m5, b5, 130.0 * kGB) - b5) < 1e6, "more free when committing: it stays as planned", b5);
    // the stepped fill (arena_grow / arena_fill_plan): 1.5 GB steps of 1.75 MB experts on the M4 (7 GB kept + 0.5)
    ArenaMem s4 = m4;
    s4.available = 17.8 * kGB;
    check(arena_grow(s4, 17.8 * kGB, 1.5 * kGB), "a step with room: it fills", 17.8 * kGB);
    check(!arena_grow(s4, 8.9 * kGB, 1.5 * kGB), "a step that would leave < 7.5 GB: it stops", 8.9 * kGB);
    check(arena_grow(s4, 9.0 * kGB, 1.5 * kGB), "a step that leaves exactly 7.5 GB: it fills", 9.0 * kGB);
    ArenaMem pin = s4;
    pin.arena_gb = 12.0;
    check(arena_grow(pin, 1.0 * kGB, 1.5 * kGB), "STRATA_METAL_ARENA_GB: always fills", 1.0 * kGB);
    std::vector<double> cost(6000, 1.75e6);   // 10.5 GB of experts planned
    // macOS gives exactly what is filled: the plan fills all of it while free stays >= 7.5 GB
    const size_t all = arena_fill_plan(s4, cost.data(), cost.size(), 1.5 * kGB, [](double f) { return 18.5 * kGB - f; });
    check(all == cost.size(), "free memory follows the fill: every planned expert is filled", all * 1.75e6);
    // another program takes 4 GB while the arena fills (after 3 GB): the fill stops at the floor, a step short
    const size_t part = arena_fill_plan(s4, cost.data(), cost.size(), 1.5 * kGB,
                                        [](double f) { return 18.5 * kGB - f - (f >= 3 * kGB ? 4 * kGB : 0); });
    const double left = 18.5 * kGB - part * 1.75e6 - 4 * kGB;
    check(part < cost.size() && left >= 7.5 * kGB && left - 1.5 * kGB < 7.5 * kGB + 1.75e6,
          "4 GB taken meanwhile: it stops with >= 7.5 GB free, less than a step above", part * 1.75e6);
    // no room even for the first step: nothing is filled (the min slots still exist, unfilled)
    check(arena_fill_plan(s4, cost.data(), cost.size(), 1.5 * kGB, [](double) { return 8.0 * kGB; }) == 0,
          "no room for one step: nothing filled", 0);
    // what counts as available: the M4 at the engine's start (16 KiB pages: 6.3 GB free + purgeable, 10.4 GB
    // inactive, 14 GB file-backed) - the clean file cache counts; another program's inactive memory does not
    check(std::fabs(arena_avail(6.3 * kGB, 0.07 * kGB, 10.4 * kGB, 0.1 * kGB, 14.1 * kGB) - 16.87 * kGB) < 1e6,
          "inactive file cache counts", arena_avail(6.3 * kGB, 0.07 * kGB, 10.4 * kGB, 0.1 * kGB, 14.1 * kGB));
    check(std::fabs(arena_avail(6.3 * kGB, 0.07 * kGB, 10.4 * kGB, 0.1 * kGB, 2.0 * kGB) - 8.37 * kGB) < 1e6,
          "inactive anonymous memory does not (only 2 GB file-backed)", arena_avail(6.3 * kGB, 0.07 * kGB, 10.4 * kGB, 0.1 * kGB, 2.0 * kGB));
    std::printf("%s\n", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
