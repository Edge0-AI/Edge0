#pragma once
#include <cstddef>

#define E0_MB_API __attribute__((visibility("default")))

// Positive CLI MiB overrides E0_MEM_BUDGET_MB; zero/unset is dormant.
// Linux has no unprivileged hard RSS rlimit. This cooperative monitor reclaims
// clean read-only model pages and reports when the measured RSS cannot be met.
E0_MB_API bool edge0_apply_mem_budget(int mib);
E0_MB_API bool edge0_mem_budget_enabled();
E0_MB_API bool edge0_mem_budget_check();
E0_MB_API void edge0_mem_budget_shutdown();

// llama_mmap lifecycle hooks: ONLY clean, read-only file mappings belong here.
// Unmap and reclamation share a lock so recycled addresses cannot be advised.
E0_MB_API void edge0_mem_budget_mapped(void * address, size_t length);
E0_MB_API int edge0_mem_budget_munmap(void * address, size_t length);
