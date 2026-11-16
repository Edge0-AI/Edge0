// Strata. The gfx11/gfx12 WMMA fragments one warp's 16x16x16 tile is fed and read through (the expert kernels'
// mx_set_wmma_kernel and the attention kernel's matrix-core path share this form).  Each lane gives its row's (or
// column's) 16 values of a k-step - on gfx12 its half of them - and holds 8 of the 16x16 results: row set_wmma_row(i, hi),
// column lane % 16.
#pragma once
#include <cstdint>
#if defined(STRATA_USE_HIP)
#include <hip/hip_runtime.h>

using h16v = _Float16 __attribute__((ext_vector_type(16)));
using f8v = float __attribute__((ext_vector_type(8)));

inline __device__ __forceinline__ f8v set_wmma(h16v a, h16v b, f8v c) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#elif defined(__gfx1200__) || defined(__gfx1201__)
    using h8 = _Float16 __attribute__((ext_vector_type(8)));
    const int k0 = (threadIdx.x & 31) / 16 * 8;
    h8 aa, bb;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        aa[i] = a[k0 + i];
        bb[i] = b[k0 + i];
    }
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(aa, bb, c);
#else
    __builtin_trap();
    return c;
#endif
}
inline __device__ __forceinline__ int set_wmma_row(int i, int hi) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    return 8 * hi + i;
#else
    return 2 * i + hi;
#endif
}
#endif
