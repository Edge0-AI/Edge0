// metal/include/metal_compat/immintrin.h - the x86 intrinsics the host loops use, on arm64: the spin-wait hint and
// the store fence before a host flag is raised (verify.cpp, session.cpp, mtp.cpp, graph.cpp).
#pragma once
static inline void _mm_pause() { __asm__ __volatile__("yield"); }
static inline void _mm_sfence() { __asm__ __volatile__("dmb ishst" ::: "memory"); }
