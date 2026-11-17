// metal/include/metal_compat/cuda_bf16.h - the handful of CUDA bf16/fp8 names the dsv41 sources use, on Metal.
//
// Only what the engine touches (found by compiling src/dsv41/*.cu against this directory, 2026-10-23): the storage typedef
// and four conversions.  On a Mac the bf16 storage is a uint16_t and the conversion is the standard round-to-nearest-even
// truncation CUDA's __float2bfloat16_rn does; fp8 is E4M3 with the same rounding.
#pragma once

#include "cuda_fp16.h"   // __half_raw, which the fp8 conversions return
#include <cstdint>
#include <cstring>

typedef uint16_t __nv_bfloat16;
typedef uint16_t __nv_bfloat162;
typedef uint8_t __nv_fp8_storage_t;

static inline uint16_t strata_f32_to_bf16_rn(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x0040u);   // NaN keeps a payload
    uint32_t r = (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
    return (uint16_t) r;
}

static inline __nv_bfloat16 __float2bfloat16_rn(float f) { return strata_f32_to_bf16_rn(f); }
static inline uint16_t __bfloat16_as_ushort(__nv_bfloat16 v) { return v; }

// E4M3 (no infinities, max 448), round to nearest even - the inverse of the half conversion the engine also uses
static inline __nv_fp8_storage_t __nv_cvt_float_to_fp8(float f, int, int) {
    if (f != f) return 0x7f;
    float a = f < 0 ? -f : f;
    int sign = f < 0 ? 0x80 : 0;
    if (a == 0.0f) return (__nv_fp8_storage_t) sign;
    if (a >= 448.0f) return (__nv_fp8_storage_t) (sign | 0x7e);
    int e = 0;
    while (a >= 2.0f) { a *= 0.5f; ++e; }
    while (a < 1.0f) { a *= 2.0f; --e; }
    // a in [1,2): 3 mantissa bits, bias 7
    float scaled = (a - 1.0f) * 8.0f;
    int m = (int) (scaled + 0.5f);
    int biased = e + 7;
    if (m == 8) { m = 0; ++biased; }
    if (biased <= 0) return (__nv_fp8_storage_t) sign;                       // subnormal -> zero here
    if (biased >= 15) return (__nv_fp8_storage_t) (sign | 0x7e);             // clamp to 448
    return (__nv_fp8_storage_t) (sign | (biased << 3) | m);
}
static inline uint16_t __nv_cvt_fp8_to_halfraw(__nv_fp8_storage_t v, int) {
    // half (IEEE binary16) from E4M3: the engine only reads these back for the vision tower's tables
    const int sign = (v & 0x80) ? 0x8000 : 0;
    const int e = (v >> 3) & 0xf, m = v & 0x7;
    if (e == 0 && m == 0) return (uint16_t) sign;
    int he, hm;
    if (e == 0) { he = 0; hm = m << 7; }
    else { he = e - 7 + 15; hm = m << 7; }
    return (uint16_t) (sign | (he << 10) | hm);
}

// the conversion modes __nv_cvt_float_to_fp8 / __nv_cvt_fp8_to_halfraw take (CUDA spells them as an enum)
enum __nv_fp8_interpretation_t { __NV_E4M3 = 0, __NV_E5M2 = 1 };
enum __nv_saturation_t { __NV_NOSAT = 0, __NV_SATFINITE = 1 };
