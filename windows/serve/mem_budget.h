// serve/mem_budget.h — process memory budget (first-class product feature).
// Independent of the prefetch router: it only caps physical residency; no prediction.
#pragma once
#ifdef _WIN32
#  define E0_MB_API __declspec(dllexport)
#else
#  define E0_MB_API __attribute__((visibility("default")))
#endif

// Primary channel: CLI flag (--mem-budget-mb → mb > 0); when mb <= 0, falls back to
// the E0_MEM_BUDGET_MB env var (debug compat). mb > 0 pins this process's maximum
// working set to that size — the OS then evicts pages beyond the cap, so cold model
// reads genuinely occur. Returns whether the cap was applied.
E0_MB_API bool edge0_apply_mem_budget(int mb);
