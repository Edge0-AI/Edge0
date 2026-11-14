// metal/include/metal_compat/linux/mman.h - the Linux mmap flags src/core/pinned.cu asks for, on macOS.  There are no
// hugetlb pages and no transparent huge pages: MAP_HUGETLB is spelled as MAP_FIXED at address 0, which macOS refuses
// (the 4 GiB __PAGEZERO), and MADV_HUGEPAGE as an advice madvise refuses, so pinned.cu takes its 4 KB-page fallback
// and reports it, exactly as on a Linux box without a hugetlb pool.
#pragma once
#include <sys/mman.h>
#ifndef MAP_HUGETLB
#define MAP_HUGETLB MAP_FIXED
#endif
#ifndef MADV_HUGEPAGE
#define MADV_HUGEPAGE (-1)
#endif
