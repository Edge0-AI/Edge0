// metal/include/metal_compat/cuda_fp16.h - the half-precision surface the dsv41 sources use, on a Mac.
//
// The engine touches exactly these (counted by compiling src/dsv41/*.cu against this directory, M5 Max 2026-10-23):
//   __half 16, __half2 11, __half_raw 3, __half_as_ushort 8,
//   __float2half_rn 17, __half2float 18, __half22float2 6,
//   the round-to-nearest float helpers (__fmul_rn, __fadd_rn, __fmaf_rn, __frcp_rn, __fsqrt_rn),
//   and the plain half/half2 typedefs ggml-common.h uses.
// On Apple Silicon half is IEEE binary16 too, so the storage is a uint16_t and the conversions are the standard
// round-to-nearest-even ones - the same bits CUDA produces.
#pragma once

#include "vector_types.h"

#include <cstdint>
#include <cstring>

struct __half_raw {
    uint16_t x;
    __half_raw() = default;
    constexpr __half_raw(uint16_t bits) : x(bits) {}
};

struct __half {
    uint16_t x;
    __half() = default;
    __half(const __half_raw& r) : x(r.x) {}
    operator __half_raw() const { return __half_raw{x}; }
};

struct __half2 {
    uint16_t x, y;
};

typedef __half half;
typedef __half2 half2;

static inline uint16_t strata_f32_to_f16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    const uint32_t sign = (u >> 16) & 0x8000u;
    const uint32_t mag = u & 0x7fffffffu;
    if (mag >= 0x7f800000u) return (uint16_t) (sign | (mag > 0x7f800000u ? 0x7e00u : 0x7c00u));
    if (mag >= 0x477ff000u) return (uint16_t) (sign | 0x7c00u);
    const uint32_t r = (mag + 0x0fffu + ((mag >> 13) & 1u)) >> 13;
    if (r >= 0x7c00u) return (uint16_t) (sign | 0x7c00u);
    return (uint16_t) (sign | r);
}

static inline float strata_f16_bits_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) bits = sign;
        else {
            const float v = (float) m * (1.0f / 16777216.0f);
            return sign ? -v : v;
        }
    } else if (e == 31) {
        bits = sign | 0x7f800000u | (m << 13);
    } else {
        bits = sign | ((e - 15 + 127) << 23) | (m << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof out);
    return out;
}

static inline __half __float2half_rn(float f) { __half h; h.x = strata_f32_to_f16_bits(f); return h; }
static inline __half __float2half(float f) { return __float2half_rn(f); }
static inline float __half2float(__half h) { return strata_f16_bits_to_f32(h.x); }
static inline uint16_t __half_as_ushort(__half h) { return h.x; }
static inline __half __ushort_as_half(uint16_t u) { __half h; h.x = u; return h; }
static inline float2 __half22float2(__half2 h) { return make_float2(strata_f16_bits_to_f32(h.x), strata_f16_bits_to_f32(h.y)); }
static inline __half2 __floats2half2_rn(float a, float b) {
    __half2 h;
    h.x = strata_f32_to_f16_bits(a);
    h.y = strata_f32_to_f16_bits(b);
    return h;
}

// round-to-nearest float helpers: the FP ops here are IEEE and un-fused, which is what these mean
static inline float __fmul_rn(float a, float b) { return a * b; }
static inline float __fadd_rn(float a, float b) { return a + b; }
static inline float __fmaf_rn(float a, float b, float c) { return a * b + c; }
static inline float __frcp_rn(float a) { return 1.0f / a; }
static inline float __fsqrt_rn(float a) { return __builtin_sqrtf(a); }

// the half2 pack/unpack and arithmetic helpers the trellis and vision paths use
static inline __half2 __halves2half2(__half lo, __half hi) { __half2 h; h.x = lo.x; h.y = hi.x; return h; }
static inline __half2 __half2half2(__half v) { __half2 h; h.x = v.x; h.y = v.x; return h; }
static inline __half __low2half(__half2 h) { __half o; o.x = h.x; return o; }
static inline __half __high2half(__half2 h) { __half o; o.x = h.y; return o; }
static inline float __low2float(__half2 h) { return strata_f16_bits_to_f32(h.x); }
static inline float __high2float(__half2 h) { return strata_f16_bits_to_f32(h.y); }
static inline __half2 __hadd2(__half2 a, __half2 b) { return __floats2half2_rn(__half2float(__low2half(a)) + __half2float(__low2half(b)), __half2float(__high2half(a)) + __half2float(__high2half(b))); }
static inline __half2 __hmul2(__half2 a, __half2 b) { return __floats2half2_rn(__half2float(__low2half(a)) * __half2float(__low2half(b)), __half2float(__high2half(a)) * __half2float(__high2half(b))); }
static inline __half2 __hfma2(__half2 a, __half2 b, __half2 c) { return __floats2half2_rn(__half2float(__low2half(a)) * __half2float(__low2half(b)) + __half2float(__low2half(c)), __half2float(__high2half(a)) * __half2float(__high2half(b)) + __half2float(__high2half(c))); }
static inline __half2 __hneg2(__half2 a) { __half2 h; h.x = a.x ^ 0x8000u; h.y = a.y ^ 0x8000u; return h; }
