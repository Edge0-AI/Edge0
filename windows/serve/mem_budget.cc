// serve/mem_budget.cc — process memory budget (hard working-set cap on Windows).
// Semantics: budget = N MB ⇒ resident set is capped at N; beyond that the OS evicts
// our pages (mmap'd model-file pages first, so the next touch faults a real re-read).
// This is the direct implementation of the low-RAM product tiers (8 GB / 16 GB machines).
#include "mem_budget.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cstdio>
#include <cstdlib>

E0_MB_API bool edge0_apply_mem_budget(int mb) {
    if (mb <= 0) {                                        // debug-compat env fallback
        const char * e = getenv("E0_MEM_BUDGET_MB");
        if (e && *e) mb = (int) atol(e);
    }
    if (mb <= 0) return false;

    const SIZE_T budget = (SIZE_T) mb * 1024 * 1024;
    SIZE_T cur_min = 0, cur_max = 0; DWORD cur_flags = 0;
    SIZE_T want_min = budget / 8;                 // soft floor = 12.5%: a residency base so the OS doesn't evict everything
    if (GetProcessWorkingSetSizeEx(GetCurrentProcess(), &cur_min, &cur_max, &cur_flags) && cur_max) {
        if (want_min > cur_min) want_min = cur_min < budget ? cur_min : budget;  // never raise the original minimum
    }
    // max working set = hard cap: the OS trims immediately on overflow (unlike the
    // default ~1.5 TB soft limit, which only trims under global memory pressure)
    if (!SetProcessWorkingSetSizeEx(GetCurrentProcess(), want_min, budget,
                                    QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_ENABLE)) {
        fprintf(stderr, "[mem-budget] SetProcessWorkingSetSizeEx failed err=%lu (budget %dMB NOT applied)\n",
                GetLastError(), mb);
        return false;
    }
    fprintf(stderr, "[mem-budget] applied: max working set = %d MB (min %llu MB, hard-trim on)\n",
            mb, (unsigned long long) (want_min / (1024 * 1024)));
    fflush(stderr);
    return true;
}
