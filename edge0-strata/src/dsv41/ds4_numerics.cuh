// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_numerics.cuh - the BF16 and FP8 (E4M3) conversions of maya-ds4's kernels, on CUDA through its
// intrinsics (cuda_bf16.h / cuda_fp8.h) and on HIP in plain arithmetic that gives the same bits: BF16 round to
// nearest even; E4M3 (OCP "fn": no infinities, max 448) decoded exactly, rounded to nearest even and saturated.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#if !defined(STRATA_USE_HIP)
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#endif

namespace strata::ds4 {

/// F32 -> BF16 bits, round to nearest even (a NaN stays a NaN).
__device__ __forceinline__ uint16_t bf16_rn(float f) {
#if !defined(STRATA_USE_HIP)
    return __bfloat16_as_ushort(__float2bfloat16_rn(f));
#else
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x40u);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
#endif
}

/// An E4M3 code's value (exact in F32).
__device__ __forceinline__ float e4m3_value(uint8_t q) {
#if !defined(STRATA_USE_HIP)
    const __half_raw h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t) q, __NV_E4M3);
    return __half2float(__half(h));
#else
    const uint32_t e = (q >> 3) & 15u, m = q & 7u;
    float v;
    if (e == 15u && m == 7u) v = __uint_as_float(0x7fc00000u);                         // NaN
    else if (e == 0u) v = (float) m * 0.001953125f;                                      // m 2^-9
    else v = __uint_as_float(((e + 120u) << 23) | (m << 20));                           // (1 + m / 8) 2^(e - 7)
    return (q & 0x80u) ? -v : v;
#endif
}

/// v rounded to the nearest E4M3 value (ties to even), saturated at +-448.
__device__ __forceinline__ float e4m3_round(float v) {
#if !defined(STRATA_USE_HIP)
    const __nv_fp8_storage_t q = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    const __half_raw h = __nv_cvt_fp8_to_halfraw(q, __NV_E4M3);
    return __half2float(__half(h));
#else
    const float a = fabsf(v);
    if (!(a == a)) return v;
    float r;
    if (a >= 448.0f) r = 448.0f;
    else {
        const int e = (int) ((__float_as_uint(a) >> 23) & 0xffu) - 127;   // a's binade (a > 0)
        const float step = e < -6 ? 0.001953125f : __uint_as_float((uint32_t) (e - 3 + 127) << 23);
        r = a == 0.0f ? 0.0f : rintf(a / step) * step;                      // (a power of two: exact)
        r = fminf(r, 448.0f);
    }
    return v < 0.0f ? -r : r;
#endif
}

}  // namespace strata::ds4
