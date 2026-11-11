// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_iq.cu - a GGUF's low-bit routed experts on the device (see ds4_iq.cuh).  The block layouts and grids are
// ggml's (ggml-common.h, MIT, third_party/ggml/LICENSE); the sign and grid tricks are llama.cpp's vecdotq.cuh.
#include "ds4_launch.h"   // strata_ds4_launch: <<<>>> on CUDA and HIP, a dispatch on Metal
#include "ds4_iq.cuh"
#if defined(STRATA_METAL)
#include <ds4_metal_host.h>
#endif
#include "ds4_numerics.cuh"
#include "ds4_experts_rule.hpp"

#include <cuda_fp16.h>
#if !defined(STRATA_USE_HIP)
#include <mma.h>
#endif

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <map>
#include <mutex>
#include <unordered_map>

namespace strata::ds4::gguf::iq {
namespace {

constexpr int kWarps = 8, kThreads = kWarps * 32;

// ---------------------------------------------------------------- the types
template <int T>
struct Ty;
template <> struct Ty<kIQ2_XXS> { static constexpr int B = 66, TAB = 512, MIN = 0; };
template <> struct Ty<kIQ2_XS> { static constexpr int B = 74, TAB = 1024, MIN = 0; };
template <> struct Ty<kIQ2_S> { static constexpr int B = 82, TAB = 2048, MIN = 0; };
template <> struct Ty<kIQ3_XXS> { static constexpr int B = 98, TAB = 256, MIN = 0; };
template <> struct Ty<kIQ3_S> { static constexpr int B = 110, TAB = 512, MIN = 0; };
template <> struct Ty<kIQ4_XS> { static constexpr int B = 136, TAB = 4, MIN = 0; };
template <> struct Ty<kQ4_K> { static constexpr int B = 144, TAB = 0, MIN = 1; };
// The 1.5- and 1.75-bit types (IQ1_S, IQ1_M: an 11-bit index into iq1s_grid_gpu, whose 8 values are 4-bit codes) and
// IQ4_NL (32 values a block: its codebook's 16 int8 values are the first 4 words of its table)
template <> struct Ty<kIQ1_S> { static constexpr int B = 50, TAB = 2048, MIN = 0; };
template <> struct Ty<kIQ1_M> { static constexpr int B = 56, TAB = 2048, MIN = 0; };
template <> struct Ty<kIQ4_NL> { static constexpr int B = 18, TAB = 16, MIN = 0; };
template <> struct Ty<kQ2_K> { static constexpr int B = 84, TAB = 0, MIN = 1; };
template <> struct Ty<kQ3_K> { static constexpr int B = 110, TAB = 0, MIN = 0; };
// the dense matrices' 5- and 6-bit K-quants (GEMV only: ds4_iq's mv; not an expert type)
template <> struct Ty<kQ5_K> { static constexpr int B = 176, TAB = 0, MIN = 1; };
template <> struct Ty<kQ6_K> { static constexpr int B = 210, TAB = 0, MIN = 0; };
// MXFP4 (ds4_gguf.cu's own kernels everywhere else): a block of 32 values, an E8M0 scale then 16 code bytes; the light
// sets take it too
constexpr int kMXFP4 = 39;
template <> struct Ty<kMXFP4> { static constexpr int B = 17, TAB = 0, MIN = 0; };
template <int T> constexpr int kQK = T == kMXFP4 || T == kIQ4_NL ? 32 : 256;   // values a block
static_assert(sizeof(block_iq2_xxs) == 66 && sizeof(block_iq2_xs) == 74 && sizeof(block_iq2_s) == 82 &&
                  sizeof(block_iq3_xxs) == 98 &&
                  sizeof(block_iq3_s) == 110 && sizeof(block_iq4_xs) == 136 && sizeof(block_q4_K) == 144 &&
                  sizeof(block_q2_K) == 84 && sizeof(block_q3_K) == 110 && sizeof(block_iq1_s) == 50 &&
                  sizeof(block_iq1_m) == 56 && sizeof(block_iq4_nl) == 18,
              "ggml's block sizes");

template <int T>
__host__ __device__ __forceinline__ size_t row_bytes_t(int k) { return (size_t) (k / kQK<T>) * Ty<T>::B; }

// the type's codebook into shared memory (u32 words; IQ4_XS: its 16 values, from bytes - an int8 table's alignment)
template <int T>
__device__ __forceinline__ void load_tab(uint32_t* tab) {
    if constexpr (T == kIQ4_XS || T == kIQ4_NL) {
        if (threadIdx.x < 4) {
            const uint8_t* kv = (const uint8_t*) kvalues_iq4nl + 4 * threadIdx.x;
            tab[threadIdx.x] = (uint32_t) kv[0] | (uint32_t) kv[1] << 8 | (uint32_t) kv[2] << 16 | (uint32_t) kv[3] << 24;
        }
    } else if constexpr (Ty<T>::TAB > 0) {
        const uint32_t* src = T == kIQ2_XXS  ? (const uint32_t*) iq2xxs_grid
                              : T == kIQ2_XS ? (const uint32_t*) iq2xs_grid
                              : T == kIQ2_S  ? (const uint32_t*) iq2s_grid
                              : T == kIQ3_XXS ? (const uint32_t*) iq3xxs_grid
                              : T == kIQ1_S || T == kIQ1_M ? (const uint32_t*) iq1s_grid_gpu
                                              : (const uint32_t*) iq3s_grid;
        for (int i = threadIdx.x; i < Ty<T>::TAB; i += blockDim.x) tab[i] = src[i];
    }
}

__device__ __forceinline__ uint32_t ld2(const uint8_t* p) {   // 4 bytes from a 2-byte aligned address
    const uint16_t* q = (const uint16_t*) p;
    return (uint32_t) q[0] | ((uint32_t) q[1] << 16);
}
__device__ __forceinline__ uint32_t ld4(const uint8_t* p) { return *(const uint32_t*) p; }
__device__ __forceinline__ float h2f(const uint8_t* p) { return __half2float(__ushort_as_half(*(const uint16_t*) p)); }
// 7 sign bits (bits 0..6 of v) -> the 8 signs, bit 7 their parity (ggml's ksigns_iq2xs)
__device__ __forceinline__ uint32_t ksigns8(uint32_t v) {
    v &= 127u;
    return v | ((__popc(v) & 1u) << 7);
}
// 4 grid bytes with the signs of bits 0..3 of b applied (bit i: byte i negative) - llama.cpp's HIP form, cheap where the
// SIMD byte instructions (__vcmpne4 / __vsub4) are emulated (Volta, RDNA): every grid byte is nonzero and <= 62, so
// (g ^ 0xff) + 1 = -g in each byte with no carry into the next
__device__ __forceinline__ int sgn4(uint32_t g, uint32_t b) {
    const uint32_t lo = (((b & 15u) * 0x10204080u) & 0x80808080u) >> 7;
    return (int) ((g ^ (lo * 255u)) + lo);
}
__device__ __forceinline__ int2 tab16(uint32_t q4, const uint32_t* t) {   // vecdotq's get_int_from_table_16
    uint32_t tmp[2];
    const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t sh = 16 * i;
        const uint32_t lo = __byte_perm(t[0], t[1], q4 >> sh), hi = __byte_perm(t[2], t[3], q4 >> sh);
        tmp[i] = __byte_perm(lo, hi, sel >> sh);
    }
    return make_int2((int) __byte_perm(tmp[0], tmp[1], 0x6420), (int) __byte_perm(tmp[0], tmp[1], 0x7531));
}
// iq1s_grid_gpu's 8 values in 4 bytes, 4 bits each (low nibbles: values 0..3, high: 4..7; 0 is -1, 2 is +1) -> the 4
// values as int8 codes 8 (n - 1) + 1, or - 1 when the group's delta is negative: ggml's weight dl ((n - 1) +- 0.125), whose
// dl (2 sc + 1) / 8 is this kernel's per-sub-block scale, is then d * (s0 lo + s1 hi) exactly
__device__ __forceinline__ uint32_t iq1_codes(uint32_t n, bool neg) {
    const uint32_t c = (uint32_t) __vsub4((int) (n * 8u), 0x07070707);   // (n * 8 - 7) in each byte: no carry (n <= 15)
    return neg ? (uint32_t) __vsub4((int) c, 0x02020202) : c;
}
// IQ1_M has no d field: its F16 scale is the top nibbles of its four u16 scales (ggml's dequantize_row_iq1_m)
__device__ __forceinline__ uint16_t iq1m_d(const uint8_t* b) {
    const uint32_t s0 = (uint32_t) b[48] | (uint32_t) b[49] << 8, s1 = (uint32_t) b[50] | (uint32_t) b[51] << 8;
    const uint32_t s2 = (uint32_t) b[52] | (uint32_t) b[53] << 8, s3 = (uint32_t) b[54] | (uint32_t) b[55] << 8;
    return (uint16_t) ((s0 >> 12) | ((s1 >> 8) & 0x00f0u) | ((s2 >> 4) & 0x0f00u) | (s3 & 0xf000u));
}
__device__ __forceinline__ int q3k_scale(const uint8_t* sc, int is) {
    const int lo = is < 8 ? sc[is] & 15 : sc[is - 8] >> 4;
    const int hi = (sc[8 + (is & 3)] >> (2 * (is >> 2))) & 3;
    return (lo | (hi << 4)) - 32;
}
__device__ __forceinline__ void q4k_scale(const uint8_t* q, int j, int& d, int& m) {   // ggml's get_scale_min_k4
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// a 32-value sub-block (kb: its index in the row) as 32 int8 codes and its scales: weight i = d s q_i - m, s and m of
// its half (i < 16: s0, m0; else s1, m1)
struct Sub {
    int q[8];
    float d, m0, m1;
    int s0, s1;
};
// (b: the block's address, ib: the sub-block in it (0 for the 32-value types); the token kernels keep ib fixed a lane)
template <int T>
__device__ __forceinline__ void decode_b(const uint8_t* __restrict__ b, int ib, const uint32_t* tab, Sub& s) {
    if constexpr (T == kMXFP4) {   // codes: value j in byte j's low nibble, j + 16 in its high one (any alignment)
        const uint32_t t[4] = {0x03020100u, 0x0C080604u, 0xFDFEFF00u, 0xF4F8FAFCu};   // ggml's kvalues_mxfp4
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint8_t* q = b + 1 + 4 * i;
            const int2 v = tab16((uint32_t) q[0] | (uint32_t) q[1] << 8 | (uint32_t) q[2] << 16 | (uint32_t) q[3] << 24, t);
            s.q[i] = v.x;
            s.q[4 + i] = v.y;
        }
        const uint32_t e = b[0];   // 2^(e - 128): ggml's GGML_E8M0_TO_FP32_HALF (the codes are the E2M1 values doubled)
        s.d = __uint_as_float(e < 2 ? 0x00200000u << e : (e - 1) << 23);
        s.s0 = s.s1 = 1;
        (void) ib;
        (void) tab;
    } else if constexpr (T == kIQ2_XXS) {
        const uint8_t* q = b + 2 + 8 * ib;
        const uint32_t a0 = ld2(q), a1 = ld2(q + 4);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 g = ((const uint2*) tab)[(a0 >> (8 * l)) & 255u];
            const uint32_t sg = ksigns8(a1 >> (7 * l));
            s.q[2 * l] = sgn4(g.x, sg);
            s.q[2 * l + 1] = sgn4(g.y, sg >> 4);
        }
        s.s0 = s.s1 = 2 * (int) (a1 >> 28) + 1;
        s.d = h2f(b) * 0.125f;
    } else if constexpr (T == kIQ2_XS) {   // d, qs[32] u16 (9-bit grid index, 7-bit sign index), scales[8] (two 4-bit)
        const uint8_t* q = b + 2 + 8 * ib;
        const uint32_t a0 = ld2(q), a1 = ld2(q + 4);
        const uint32_t v[4] = {a0 & 0xffffu, a0 >> 16, a1 & 0xffffu, a1 >> 16};
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 g = ((const uint2*) tab)[v[l] & 511u];
            const uint32_t sg = ksigns8(v[l] >> 9);
            s.q[2 * l] = sgn4(g.x, sg);
            s.q[2 * l + 1] = sgn4(g.y, sg >> 4);
        }
        const uint8_t sc = b[66 + ib];
        s.s0 = 2 * (int) (sc & 15) + 1;
        s.s1 = 2 * (int) (sc >> 4) + 1;
        s.d = h2f(b) * 0.125f;
    } else if constexpr (T == kIQ2_S) {
        const uint8_t* qs = b + 2;
        const uint32_t i4 = ld2(qs + 4 * ib), sg4 = ld2(qs + 32 + 4 * ib), h = b[66 + ib], sc = b[74 + ib];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 g = ((const uint2*) tab)[((i4 >> (8 * l)) & 255u) | ((h << (8 - 2 * l)) & 0x300u)];
            const uint32_t sg = sg4 >> (8 * l);
            s.q[2 * l] = sgn4(g.x, sg);
            s.q[2 * l + 1] = sgn4(g.y, sg >> 4);
        }
        s.s0 = 2 * (int) (sc & 15) + 1;
        s.s1 = 2 * (int) (sc >> 4) + 1;
        s.d = h2f(b) * 0.125f;
    } else if constexpr (T == kIQ3_XXS) {
        const uint8_t* qs = b + 2;
        const uint32_t w0 = ld2(qs + 8 * ib), w1 = ld2(qs + 8 * ib + 4), aux = ld2(qs + 64 + 4 * ib);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t w = l < 2 ? w0 : w1, sh = 16 * (l & 1);
            const uint32_t sg = ksigns8(aux >> (7 * l));
            s.q[2 * l] = sgn4(tab[(w >> sh) & 255u], sg);
            s.q[2 * l + 1] = sgn4(tab[(w >> (sh + 8)) & 255u], sg >> 4);
        }
        s.s0 = s.s1 = 2 * (int) (aux >> 28) + 1;
        s.d = h2f(b) * 0.25f;
    } else if constexpr (T == kIQ3_S) {
        const uint8_t* qs = b + 2;
        const uint32_t w0 = ld2(qs + 8 * ib), w1 = ld2(qs + 8 * ib + 4), h = b[66 + ib], s4 = ld2(b + 74 + 4 * ib);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t w = l < 2 ? w0 : w1, sh = 16 * (l & 1);
            const uint32_t i1 = ((w >> sh) & 255u) | ((h << (8 - 2 * l)) & 256u);
            const uint32_t i2 = ((w >> (sh + 8)) & 255u) | ((h << (7 - 2 * l)) & 256u);
            const uint32_t sg = s4 >> (8 * l);
            s.q[2 * l] = sgn4(tab[i1], sg);
            s.q[2 * l + 1] = sgn4(tab[i2], sg >> 4);
        }
        s.s0 = s.s1 = 1 + 2 * (int) ((b[106 + (ib >> 1)] >> (4 * (ib & 1))) & 15);
        s.d = h2f(b);
    } else if constexpr (T == kIQ4_XS) {
        const uint32_t sh = *(const uint16_t*) (b + 2);
        const int ls = ((b[4 + (ib >> 1)] >> (4 * (ib & 1))) & 15) | (((sh >> (2 * ib)) & 3) << 4);
        const uint8_t* qs = b + 8 + 16 * ib;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int2 v = tab16(ld4(qs + 4 * t), tab);
            s.q[t] = v.x;
            s.q[4 + t] = v.y;
        }
        s.s0 = s.s1 = ls - 32;
        s.d = h2f(b);
    } else if constexpr (T == kIQ1_S) {   // d, qs[32] (4 a 32-value sub-block), qh[8] u16: 11-bit grid, 3-bit scale, delta
        const uint8_t* qs = b + 2;
        const uint16_t qh = ((const uint16_t*) (b + 34))[ib];
        const bool neg = (qh & 0x8000u) != 0;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t g = tab[qs[4 * ib + l] | (((qh >> (3 * l)) & 7u) << 8)];
            s.q[2 * l] = (int) iq1_codes(g & 0x0F0F0F0Fu, neg);
            s.q[2 * l + 1] = (int) iq1_codes((g >> 4) & 0x0F0F0F0Fu, neg);
        }
        s.s0 = s.s1 = 2 * (int) ((qh >> 12) & 7u) + 1;
        s.d = h2f(b) * 0.125f;
    } else if constexpr (T == kIQ1_M) {   // qs[32], qh[16], scales[8]: no d field; 11-bit grid, two 3-bit scales a 32
        const uint8_t* qs = b + 4 * ib;
        const uint8_t qh0 = b[32 + 2 * ib], qh1 = b[33 + 2 * ib];
        const uint16_t sc = ((const uint16_t*) (b + 48))[ib >> 1];
        const int sh = 6 * (ib & 1);
        const uint32_t ii[4] = {qs[0] | ((uint32_t) (qh0 << 8) & 0x700u), qs[1] | ((uint32_t) (qh0 << 4) & 0x700u),
                                qs[2] | ((uint32_t) (qh1 << 8) & 0x700u), qs[3] | ((uint32_t) (qh1 << 4) & 0x700u)};
        const bool neg[4] = {(qh0 & 0x08u) != 0, (qh0 & 0x80u) != 0, (qh1 & 0x08u) != 0, (qh1 & 0x80u) != 0};
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t g = tab[ii[l]];
            s.q[2 * l] = (int) iq1_codes(g & 0x0F0F0F0Fu, neg[l]);
            s.q[2 * l + 1] = (int) iq1_codes((g >> 4) & 0x0F0F0F0Fu, neg[l]);
        }
        s.s0 = 2 * (int) ((sc >> sh) & 7u) + 1;
        s.s1 = 2 * (int) ((sc >> (sh + 3)) & 7u) + 1;
        s.d = __half2float(__ushort_as_half(iq1m_d(b))) * 0.125f;
    } else if constexpr (T == kIQ4_NL) {   // d, qs[16]: a block of 32 values, its codebook's int8 a nibble
        const uint8_t* qs = b + 2;   // (18-byte blocks: even, so ld2's 2-byte alignment - not ld4's 4)
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int2 v = tab16(ld2(qs + 4 * t), tab);
            s.q[t] = v.x;
            s.q[4 + t] = v.y;
        }
        s.s0 = s.s1 = 1;
        s.d = h2f(b);
    } else if constexpr (T == kQ4_K) {
        int sc, m;
        q4k_scale(b + 4, ib, sc, m);
        const uint8_t* qs = b + 16 + 32 * (ib >> 1);
        const int sh = 4 * (ib & 1);
#pragma unroll
        for (int t = 0; t < 8; ++t) s.q[t] = (int) ((ld4(qs + 4 * t) >> sh) & 0x0f0f0f0fu);
        s.s0 = s.s1 = sc;
        s.d = h2f(b);
        s.m0 = s.m1 = h2f(b + 2) * (float) m;
    } else if constexpr (T == kQ5_K) {   // d, dmin, scales[12], qh[32], qs[128]: weight = d sc (nibble + 16 hbit) - dmin m
        int sc, m;
        q4k_scale(b + 4, ib, sc, m);
        const uint8_t* qs = b + 48 + 32 * (ib >> 1);
        const uint8_t* qh = b + 16;
        const int sh = 4 * (ib & 1);
#pragma unroll
        for (int t = 0; t < 8; ++t)
            s.q[t] = (int) (((ld4(qs + 4 * t) >> sh) & 0x0f0f0f0fu) | (((ld4(qh + 4 * t) >> ib) & 0x01010101u) << 4));
        s.s0 = s.s1 = sc;
        s.d = h2f(b);
        s.m0 = s.m1 = h2f(b + 2) * (float) m;
    } else if constexpr (T == kQ6_K) {   // ql[128], qh[64], scales int8[16], d: weight = d sc (q - 32), q 6 bits
        const int h = ib >> 2, quarter = ib & 3;
        const uint8_t* ql = b + 64 * h + (quarter & 1 ? 32 : 0);
        const uint8_t* qh = b + 128 + 32 * h;
        const int sh = 4 * (quarter >> 1);
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const uint32_t x = ((ld2(ql + 4 * t) >> sh) & 0x0f0f0f0fu) | (((ld2(qh + 4 * t) >> (2 * quarter)) & 0x03030303u) << 4);
            s.q[t] = (int) __vsub4(x, 0x20202020u);
        }
        const int8_t* scl = (const int8_t*) (b + 192) + 8 * h + 2 * quarter;
        s.s0 = scl[0];
        s.s1 = scl[1];
        s.d = h2f(b + 208);
    } else if constexpr (T == kQ2_K) {
        const int n = ib >> 2, j = ib & 3;
        const uint8_t* qs = b + 16 + 32 * n;
#pragma unroll
        for (int t = 0; t < 8; ++t) s.q[t] = (int) ((ld4(qs + 4 * t) >> (2 * j)) & 0x03030303u);
        const int c0 = b[8 * n + 2 * j], c1 = b[8 * n + 2 * j + 1];
        const float dmin = h2f(b + 82);
        s.s0 = c0 & 15;
        s.s1 = c1 & 15;
        s.d = h2f(b + 80);
        s.m0 = dmin * (float) (c0 >> 4);
        s.m1 = dmin * (float) (c1 >> 4);
    } else {   // Q3_K
        const int n = ib >> 2, j = ib & 3;
        const uint8_t* qs = b + 32 + 32 * n;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const uint32_t ql = (ld2(qs + 4 * t) >> (2 * j)) & 0x03030303u;
            const uint32_t neg = ((~ld2(b + 4 * t)) >> (4 * n + j)) & 0x01010101u;
            s.q[t] = (int) (ql | (neg * 0xFCu));   // (q - 4 in a byte: q + 0xFC, no carry for q <= 3)
        }
        s.s0 = q3k_scale(b + 96, 8 * n + 2 * j);
        s.s1 = q3k_scale(b + 96, 8 * n + 2 * j + 1);
        s.d = h2f(b + 108);
    }
}
template <int T>
__device__ __forceinline__ void decode(const uint8_t* __restrict__ row, int kb, const uint32_t* tab, Sub& s) {
    decode_b<T>(row + (size_t) (kQK<T> == 32 ? kb : kb >> 3) * Ty<T>::B, kb & 7, tab, s);
}
// the sub-block against 32 int8 activations (xa: values 0..15, xb: 16..31), before their scale: an exact integer dot,
// its scales in F32 (explicit roundings: every instantiation the same bits)
template <int T>
__device__ __forceinline__ float sub_dot(const Sub& s, const int4& xa, const int4& xb) {
    int lo = __dp4a(s.q[0], xa.x, 0);
    lo = __dp4a(s.q[1], xa.y, lo);
    lo = __dp4a(s.q[2], xa.z, lo);
    lo = __dp4a(s.q[3], xa.w, lo);
    int hi = __dp4a(s.q[4], xb.x, 0);
    hi = __dp4a(s.q[5], xb.y, hi);
    hi = __dp4a(s.q[6], xb.z, hi);
    hi = __dp4a(s.q[7], xb.w, hi);
    float v = __fmul_rn(s.d, (float) (s.s0 * lo + s.s1 * hi));
    if constexpr (Ty<T>::MIN) {
        int sl = __dp4a(0x01010101, xa.x, 0);
        sl = __dp4a(0x01010101, xa.y, sl);
        sl = __dp4a(0x01010101, xa.z, sl);
        sl = __dp4a(0x01010101, xa.w, sl);
        int sh = __dp4a(0x01010101, xb.x, 0);
        sh = __dp4a(0x01010101, xb.y, sh);
        sh = __dp4a(0x01010101, xb.z, sh);
        sh = __dp4a(0x01010101, xb.w, sh);
        v = __fsub_rn(v, __fadd_rn(__fmul_rn(s.m0, (float) sl), __fmul_rn(s.m1, (float) sh)));
    }
    return v;
}

// values [8 p, 8 p + 8) of sub-block kb, as F32 (the prompt sets' tiles): ggml's dequantize_row_* values
template <int T>
__device__ __forceinline__ void deq8(const uint8_t* __restrict__ row, int kb, int p, const uint32_t* tab, float* w) {
    const int ib = kb & 7;
    const uint8_t* b = row + (size_t) (kQK<T> == 32 ? kb : kb >> 3) * Ty<T>::B;
    if constexpr (T == kIQ2_XXS) {
        const uint8_t* q = b + 2 + 8 * ib;
        const uint32_t a1 = ld2(q + 4);
        const uint2 g = ((const uint2*) tab)[q[p]];
        const uint32_t v = (a1 >> (7 * p)) & 127u, sg = v | ((__popc(v) & 1u) << 7);
        const float dl = h2f(b) * (float) (2 * (int) (a1 >> 28) + 1) * 0.125f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float gv = (float) ((j < 4 ? g.x >> (8 * j) : g.y >> (8 * (j - 4))) & 255u);
            w[j] = (sg >> j) & 1u ? -(dl * gv) : dl * gv;
        }
    } else if constexpr (T == kIQ2_XS) {
        const uint16_t v = ((const uint16_t*) (b + 2 + 8 * ib))[p];
        const uint2 g = ((const uint2*) tab)[v & 511u];
        const uint32_t sg = ksigns8(v >> 9);
        const uint8_t sc = b[66 + ib];
        const float dl = h2f(b) * (float) (2 * (int) (p < 2 ? sc & 15 : sc >> 4) + 1) * 0.125f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float gv = (float) ((j < 4 ? g.x >> (8 * j) : g.y >> (8 * (j - 4))) & 255u);
            w[j] = (sg >> j) & 1u ? -(dl * gv) : dl * gv;
        }
    } else if constexpr (T == kIQ2_S) {
        const uint8_t* qs = b + 2;
        const uint32_t idx = qs[4 * ib + p] | ((((uint32_t) b[66 + ib]) << (8 - 2 * p)) & 0x300u);
        const uint32_t sg = qs[32 + 4 * ib + p], sc = b[74 + ib];
        const uint2 g = ((const uint2*) tab)[idx];
        const float dl = h2f(b) * (float) (2 * (int) (p < 2 ? sc & 15 : sc >> 4) + 1) * 0.125f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float gv = (float) ((j < 4 ? g.x >> (8 * j) : g.y >> (8 * (j - 4))) & 255u);
            w[j] = (sg >> j) & 1u ? -(dl * gv) : dl * gv;
        }
    } else if constexpr (T == kIQ3_XXS || T == kIQ3_S) {
        const uint8_t* qs = b + 2;
        uint32_t g1, g2, sg;
        float dl;
        if constexpr (T == kIQ3_XXS) {
            const uint32_t aux = ld2(qs + 64 + 4 * ib), v = (aux >> (7 * p)) & 127u;
            g1 = tab[qs[8 * ib + 2 * p]];
            g2 = tab[qs[8 * ib + 2 * p + 1]];
            sg = v | ((__popc(v) & 1u) << 7);
            dl = h2f(b) * (float) (2 * (int) (aux >> 28) + 1) * 0.25f;
        } else {
            const uint32_t h = b[66 + ib];
            g1 = tab[qs[8 * ib + 2 * p] | ((h << (8 - 2 * p)) & 256u)];
            g2 = tab[qs[8 * ib + 2 * p + 1] | ((h << (7 - 2 * p)) & 256u)];
            sg = b[74 + 4 * ib + p];
            dl = h2f(b) * (float) (1 + 2 * (int) ((b[106 + (ib >> 1)] >> (4 * (ib & 1))) & 15));
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float gv = (float) ((j < 4 ? g1 >> (8 * j) : g2 >> (8 * (j - 4))) & 255u);
            w[j] = (sg >> j) & 1u ? -(dl * gv) : dl * gv;
        }
    } else if constexpr (T == kIQ4_XS) {
        const uint32_t sh = *(const uint16_t*) (b + 2);
        const int ls = ((b[4 + (ib >> 1)] >> (4 * (ib & 1))) & 15) | (((sh >> (2 * ib)) & 3) << 4);
        const float dl = h2f(b) * (float) (ls - 32);
        const uint8_t* qs = b + 8 + 16 * ib + 8 * (p & 1);
        const int8_t* kv = (const int8_t*) tab;
#pragma unroll
        for (int j = 0; j < 8; ++j) w[j] = dl * (float) kv[p < 2 ? qs[j] & 15 : qs[j] >> 4];
    } else if constexpr (T == kIQ1_S) {
        const uint16_t qh = ((const uint16_t*) (b + 34))[ib];
        const uint32_t g = tab[b[2 + 4 * ib + p] | (((qh >> (3 * p)) & 7u) << 8)];
        const uint32_t c0 = iq1_codes(g & 0x0F0F0F0Fu, (qh & 0x8000u) != 0);
        const uint32_t c1 = iq1_codes((g >> 4) & 0x0F0F0F0Fu, (qh & 0x8000u) != 0);
        const float dl = h2f(b) * (float) (2 * (int) ((qh >> 12) & 7u) + 1) * 0.125f;
#pragma unroll
        for (int j = 0; j < 4; ++j) w[j] = dl * (float) (int8_t) (c0 >> (8 * j));
#pragma unroll
        for (int j = 0; j < 4; ++j) w[4 + j] = dl * (float) (int8_t) (c1 >> (8 * j));
    } else if constexpr (T == kIQ1_M) {
        const uint8_t* qs = b + 4 * ib;
        const uint8_t qh0 = b[32 + 2 * ib], qh1 = b[33 + 2 * ib], qh = p < 2 ? qh0 : qh1;
        const uint16_t sc = ((const uint16_t*) (b + 48))[ib >> 1];
        const uint32_t g = tab[qs[p] | ((uint32_t) (qh << (p & 1 ? 4 : 8)) & 0x700u)];
        const bool neg = (p & 1 ? qh & 0x80u : qh & 0x08u) != 0;
        const uint32_t c0 = iq1_codes(g & 0x0F0F0F0Fu, neg), c1 = iq1_codes((g >> 4) & 0x0F0F0F0Fu, neg);
        const int sh = (int) ((sc >> (6 * (ib & 1) + (p < 2 ? 0 : 3))) & 7u);
        const float dl = __half2float(__ushort_as_half(iq1m_d(b))) * (float) (2 * sh + 1) * 0.125f;
#pragma unroll
        for (int j = 0; j < 4; ++j) w[j] = dl * (float) (int8_t) (c0 >> (8 * j));
#pragma unroll
        for (int j = 0; j < 4; ++j) w[4 + j] = dl * (float) (int8_t) (c1 >> (8 * j));
    } else if constexpr (T == kIQ4_NL) {
        const float dl = h2f(b);
        const uint8_t* qs = b + 2 + 8 * (p & 1);
        const int8_t* kv = (const int8_t*) tab;
#pragma unroll
        for (int j = 0; j < 8; ++j) w[j] = dl * (float) kv[p < 2 ? qs[j] & 15 : qs[j] >> 4];
    } else if constexpr (T == kQ4_K) {
        int sc, m;
        q4k_scale(b + 4, ib, sc, m);
        const float d1 = h2f(b) * (float) sc, m1 = h2f(b + 2) * (float) m;
        const uint8_t* qs = b + 16 + 32 * (ib >> 1) + 8 * p;
#pragma unroll
        for (int j = 0; j < 8; ++j) w[j] = d1 * (float) ((ib & 1) ? qs[j] >> 4 : qs[j] & 15) - m1;
    } else if constexpr (T == kQ2_K) {
        const int n = ib >> 2, jj = ib & 3;
        const int c = b[8 * n + 2 * jj + (p >> 1)];
        const float dl = h2f(b + 80) * (float) (c & 15), ml = h2f(b + 82) * (float) (c >> 4);
        const uint8_t* qs = b + 16 + 32 * n + 8 * p;
#pragma unroll
        for (int j = 0; j < 8; ++j) w[j] = dl * (float) ((qs[j] >> (2 * jj)) & 3) - ml;
    } else {   // Q3_K
        const int n = ib >> 2, jj = ib & 3;
        const float dl = h2f(b + 108) * (float) q3k_scale(b + 96, 8 * n + 2 * jj + (p >> 1));
        const uint8_t* qs = b + 32 + 32 * n + 8 * p;
        const uint8_t* hm = b + 8 * p;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            w[j] = dl * (float) ((int) ((qs[j] >> (2 * jj)) & 3) - (((hm[j] >> (4 * n + jj)) & 1) ? 0 : 4));
    }
}


// ---------------------------------------------------------------- the token kernels' faster decode (CUDA only)
// The token kernels (gate_up_kernel / down_kernel) are bound by instructions per weight, not by memory, so on CUDA a type may
// decode through a larger shared-memory table that folds more of the work into one lookup.  The arithmetic is unchanged
// (the same integer codes into the same dp4a sums, the same F32 scales in the same order), so the results are the same bits.
#if !defined(STRATA_USE_HIP)
template <int T> struct Fast { static constexpr int WORDS = 0; };   // 0: this type decodes as decode<T>
// IQ1_M: the 12-bit (neg, 11-bit grid index) -> its 8 int8 codes (iq1_codes with the sub-block delta's sign), 2 words
template <> struct Fast<kIQ1_M> { static constexpr int WORDS = 8192; };
__device__ __align__(16) const uint2 kIq1mCodes[4096] = {
#include "ds4_iq1m_codes.inc"
};
#else
template <int T> struct Fast { static constexpr int WORDS = 0; };
#endif
// F: whether the fast form is wanted (the kernels of many rows a pass are bound by the dot products and their registers, and the
// larger codebook's shared memory costs them occupancy: they keep decode<T>)
template <int T, bool F = true> constexpr int kTabWords = F && Fast<T>::WORDS > 0 ? Fast<T>::WORDS : (Ty<T>::TAB > 0 ? Ty<T>::TAB : 1);

// Persistent grids (blocks looping over work items) only for the types with a codebook worth keeping in the block (8 KiB and
// more); the others run a block (tile, group) as before - their blocks leave at once when the group is empty.
#if defined(STRATA_USE_HIP)
template <int T>
constexpr bool kPersist = false;   // (HIP keeps the release grids)
#else
template <int T>
constexpr bool kPersist = kTabWords<T> * 4 >= 8192;
#endif

template <int T, bool F = true>
__device__ __forceinline__ void load_tab_fast(uint32_t* tab) {
    if constexpr (!F || Fast<T>::WORDS == 0) {
        load_tab<T>(tab);
#if !defined(STRATA_USE_HIP)
    } else if constexpr (T == kIQ1_M) {   // a copy of the generated table (ds4_iq1m_codes.inc): 2048 x 16 bytes
        const uint4* src = (const uint4*) kIq1mCodes;
        uint4* dst = (uint4*) tab;
        for (int j = threadIdx.x; j < 2048; j += blockDim.x) dst[j] = src[j];
#endif
    }
}
template <int T, bool F = true>
__device__ __forceinline__ void decode_fast_b(const uint8_t* __restrict__ b, int ib, const uint32_t* tab, Sub& s) {
    if constexpr (!F || Fast<T>::WORDS == 0) {
        decode_b<T>(b, ib, tab, s);
    } else if constexpr (T == kIQ1_M) {   // (56-byte blocks in 1120-byte rows at 256-aligned offsets: 8-byte aligned)
        const uint32_t qsw = *(const uint32_t*) (b + 4 * ib);
        const uint32_t qhw = *(const uint16_t*) (b + 32 + 2 * ib);
        const uint2 sw = *(const uint2*) (b + 48);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t i = ((qsw >> (8 * l)) & 255u) | ((l < 3 ? qhw << (8 - 4 * l) : qhw >> 4) & 0xF00u);
            const uint2 c = ((const uint2*) tab)[i];
            s.q[2 * l] = (int) c.x;
            s.q[2 * l + 1] = (int) c.y;
        }
        const uint32_t word = ib < 4 ? sw.x : sw.y;
        const uint32_t sc16 = (ib & 2 ? word >> 16 : word) & 0xffffu;
        const int sh = 6 * (ib & 1);
        s.s0 = 2 * (int) ((sc16 >> sh) & 7u) + 1;
        s.s1 = 2 * (int) ((sc16 >> (sh + 3)) & 7u) + 1;
        const uint32_t d16 = (sw.x >> 12 & 0xfu) | ((sw.x >> 24) & 0xf0u) | ((sw.y >> 4) & 0xf00u) | ((sw.y >> 16) & 0xf000u);
        s.d = __half2float(__ushort_as_half((uint16_t) d16)) * 0.125f;
    }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float swiglu(float gv, float uv, float limit) {   // (ds4_gguf.cu's)
    if (limit > 0.0f) {
        gv = fminf(gv, limit);
        uv = fminf(fmaxf(uv, -limit), limit);
    }
    return __fmul_rn(__fdiv_rn(gv, __fadd_rn(1.0f, expf(-gv))), uv);
}
__device__ __forceinline__ bool group(const Args& a, int grp, const uint8_t*& blob, int& m0, int& m1) {
    if (a.grp_ptr == nullptr) {
        const unsigned long long p = a.plan_ptr[grp];
        if (p == 0ull) return false;
        blob = (const uint8_t*) p;
        m0 = grp;
        m1 = grp + 1;
        return true;
    }
    if (grp >= *a.n_groups) return false;
    blob = (const uint8_t*) a.grp_ptr[grp];
    m0 = a.grp_start[grp];
    m1 = a.grp_start[grp + 1];
    return true;
}

// ---------------------------------------------------------------- token / window experts
// RW rows a warp, 32 / RW lanes a row (a lane a sub-block a step), their sums over those lanes
template <int RW>
__device__ __forceinline__ float lanes_sum(float v) {
#pragma unroll
    for (int o = 16 / RW; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
constexpr int kGuRW = 1, kDnRW = 2;   // rows a warp: gate / up (n_embd 5120: 10 sub-blocks a lane), down (2304: 9)

// grid (n_ff / (8 RW), groups): RW rows of the gate and of the up a warp, NE of the group's entries a pass (each
// sub-block decoded once for them) -> h = the clamped swiglu
template <int T, int NE, int RW = kGuRW>
__global__ void __launch_bounds__(kThreads) gate_up_kernel(const Args a, int groups) {
    constexpr int LPR = 32 / RW;
    constexpr int kBPS = kQK<T> == 32 ? 1 : 8, kStep = LPR / kBPS;   // sub-blocks a block; blocks a lane steps
    static_assert(LPR % kBPS == 0, "a lane keeps one sub-block position");
    __shared__ __align__(16) uint32_t tab[kTabWords<T>];
    __shared__ int es[NE], xr[NE];
    bool loaded = false;   // (the codebook once a block, when its first live item comes: a block of empty groups leaves at once)
    const int lane32 = threadIdx.x & 31;
    const size_t rb = row_bytes_t<T>(a.n_embd);
    const int nsb = a.n_embd / 32, nblk = nsb / kBPS;
    const int ntile = a.n_ff / (kWarps * RW);
    for (int item = kPersist<T> ? blockIdx.x : 0; item < (kPersist<T> ? groups * ntile : 1); item += kPersist<T> ? gridDim.x : 1) {
    const int grp = kPersist<T> ? item / ntile : blockIdx.y, tile = kPersist<T> ? item % ntile : blockIdx.x;
    const uint8_t* blob;
    int m0, m1;
    if (!group(a, grp, blob, m0, m1)) continue;   // (uniform)
    if (!loaded) { load_tab_fast<T>(tab); loaded = true; }
    for (int mm = m0; mm < m1; mm += NE) {
        const int ne = min(NE, m1 - mm);
        __syncthreads();
        if ((int) threadIdx.x < ne) {
            const int e = a.grp_ptr != nullptr ? a.grp_ent[mm + threadIdx.x] : mm + threadIdx.x;
            es[threadIdx.x] = e;
            xr[threadIdx.x] = a.grp_ptr != nullptr ? e >> 3 : 0;
        }
        __syncthreads();
        {
            const int row = (tile * kWarps + (threadIdx.x >> 5)) * RW + lane32 / LPR, lane = lane32 % LPR;
            const int ib0 = kBPS == 8 ? lane & 7 : 0, blk0 = kBPS == 8 ? lane >> 3 : lane;
            const uint8_t* g = blob + a.tr[0] + (size_t) row * rb;
            const uint8_t* u = blob + a.tr[1] + (size_t) row * rb;
            float ag[NE], au[NE];
#pragma unroll
            for (int i = 0; i < NE; ++i) ag[i] = au[i] = 0.0f;
            // a lane's sub-blocks: kb = lane + LPR j = 8 blk + ib with ib fixed (the 32-value types: blk = kb)
            for (int blk = blk0; blk < nblk; blk += kStep) {
                const int kb = blk * kBPS + ib0;
                Sub sg, su;
                decode_fast_b<T>(g + (size_t) blk * Ty<T>::B, ib0, tab, sg);
                decode_fast_b<T>(u + (size_t) blk * Ty<T>::B, ib0, tab, su);
#pragma unroll
                for (int i = 0; i < NE; ++i)
                    if (i < ne) {
                        const int4* xp = (const int4*) (a.xq + (size_t) xr[i] * a.n_embd + (size_t) kb * 32);
                        const int4 xa = xp[0], xb = xp[1];
                        const float dx = a.xd[(size_t) xr[i] * nsb + kb];
                        ag[i] = fmaf(dx, sub_dot<T>(sg, xa, xb), ag[i]);
                        au[i] = fmaf(dx, sub_dot<T>(su, xa, xb), au[i]);
                    }
            }
#pragma unroll
            for (int i = 0; i < NE; ++i)
                if (i < ne) {
                    const float gv = lanes_sum<RW>(ag[i]), uv = lanes_sum<RW>(au[i]);
                    if (lane == 0) a.h[(size_t) es[i] * a.n_ff + row] = swiglu(gv, uv, a.limit);
                }
        }
    }
    }
}

// grid (n_embd / (8 RW), groups): RW rows of the down a warp for NE entries a pass, against their int8 h -> part
template <int T, int NE, int RW = kDnRW>
__global__ void __launch_bounds__(kThreads) down_kernel(const Args a, int groups) {
    constexpr int LPR = 32 / RW;
    constexpr int kBPS = kQK<T> == 32 ? 1 : 8, kStep = LPR / kBPS;   // sub-blocks a block; blocks a lane steps
    static_assert(LPR % kBPS == 0, "a lane keeps one sub-block position");
    __shared__ __align__(16) uint32_t tab[kTabWords<T>];
    __shared__ int es[NE];
    bool loaded = false;
    const int lane32 = threadIdx.x & 31;
    const int nsb = a.n_ff / 32, nblk = nsb / kBPS;
    const int ntile = a.n_embd / (kWarps * RW);
    for (int item = kPersist<T> ? blockIdx.x : 0; item < (kPersist<T> ? groups * ntile : 1); item += kPersist<T> ? gridDim.x : 1) {
    const int grp = kPersist<T> ? item / ntile : blockIdx.y, tile = kPersist<T> ? item % ntile : blockIdx.x;
    const uint8_t* blob;
    int m0, m1;
    if (!group(a, grp, blob, m0, m1)) continue;
    if (!loaded) { load_tab_fast<T>(tab); loaded = true; }
    for (int mm = m0; mm < m1; mm += NE) {
        const int ne = min(NE, m1 - mm);
        __syncthreads();
        if ((int) threadIdx.x < ne) es[threadIdx.x] = a.grp_ptr != nullptr ? a.grp_ent[mm + threadIdx.x] : mm + threadIdx.x;
        __syncthreads();
        {
            const int row = (tile * kWarps + (threadIdx.x >> 5)) * RW + lane32 / LPR, lane = lane32 % LPR;
            const int ib0 = kBPS == 8 ? lane & 7 : 0, blk0 = kBPS == 8 ? lane >> 3 : lane;
            const uint8_t* d = blob + a.tr[2] + (size_t) row * row_bytes_t<T>(a.n_ff);
            float ad[NE];
#pragma unroll
            for (int i = 0; i < NE; ++i) ad[i] = 0.0f;
            for (int blk = blk0; blk < nblk; blk += kStep) {
                const int kb = blk * kBPS + ib0;
                Sub sd;
                decode_fast_b<T>(d + (size_t) blk * Ty<T>::B, ib0, tab, sd);
#pragma unroll
                for (int i = 0; i < NE; ++i)
                    if (i < ne) {
                        const int4* hp = (const int4*) (a.hq + (size_t) es[i] * a.n_ff + (size_t) kb * 32);
                        const int4 ha = hp[0], hb = hp[1];
                        ad[i] = fmaf(a.hd[(size_t) es[i] * nsb + kb], sub_dot<T>(sd, ha, hb), ad[i]);
                    }
            }
#pragma unroll
            for (int i = 0; i < NE; ++i)
                if (i < ne) {
                    const float v = lanes_sum<RW>(ad[i]);
                    if (lane == 0) a.part[(size_t) es[i] * a.n_embd + row] = v;
                }
        }
    }
    }
}

// ---------------------------------------------------------------- the prompt path's sets
constexpr int BM = 64, BN = 64, TM = 4, TN = 4;
static_assert((BM / TM) * (BN / TN) == kThreads, "one output tile a thread");

// F32 FMAs (any device): a block RM rows (64, or 16 for experts of a few rows: a short prompt's) x BN outputs, a 32-value
// sub-block of K a step, the weights dequantised into shared memory (each thread 8 values of a row), each thread RT x TN
// outputs.  (deq8 is the tile loader a tensor-core form takes too.)
template <int T, bool GATE_UP, int RM = BM>
__global__ void __launch_bounds__(kThreads) set_kernel(const SetArgs a) {
    constexpr int RT = RM / (kThreads / (BN / TN));
    static_assert(RT >= 1 && RT * (kThreads / (BN / TN)) == RM, "the block's rows");
    __shared__ uint32_t tab[Ty<T>::TAB > 0 ? Ty<T>::TAB : 1];
    __shared__ float wg[32][BN + 1];
    __shared__ float wu[GATE_UP ? 32 : 1][BN + 1];
    __shared__ float xs[RM][33];
    __shared__ int src[RM];
    const int e = blockIdx.z;
    const int lo = a.bounds[e] - a.row0, hi = a.bounds[e + 1] - a.row0;
    const int r0 = lo + blockIdx.x * RM;
    if (r0 >= hi || hi - lo <= a.skip_le) return;   // (uniform; skip_le: the light kernels' experts)
    if (a.skip_ge > 0 && hi - lo >= a.skip_ge) return;   // (the caller's FP16 GEMMs')
    load_tab<T>(tab);
    const int nr = min(RM, hi - r0);
    const int c0 = blockIdx.y * BN;
    const uint8_t* blob = a.wbase + (size_t) e * a.stride;
    const int K = GATE_UP ? a.n_embd : a.n_ff;
    const size_t rb = row_bytes_t<T>(K);
    const int tid = (int) threadIdx.x;
    if (tid < RM) src[tid] = tid < nr ? (GATE_UP ? a.idx[r0 + tid] : r0 + tid) : -1;
    __syncthreads();
    const int ty = tid / (BN / TN), tx = tid % (BN / TN);
    const int tr = tid >> 2, tq = tid & 3;
    float ag[RT][TN], au[RT][TN];
#pragma unroll
    for (int i = 0; i < RT; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) ag[i][j] = au[i][j] = 0.0f;
    for (int kb = 0; kb < K / 32; ++kb) {
        float w8[8];
        deq8<T>(blob + a.tr[GATE_UP ? 0 : 2] + (size_t) (c0 + tr) * rb, kb, tq, tab, w8);
#pragma unroll
        for (int j = 0; j < 8; ++j) wg[tq * 8 + j][tr] = w8[j];
        if (GATE_UP) {
            deq8<T>(blob + a.tr[1] + (size_t) (c0 + tr) * rb, kb, tq, tab, w8);
#pragma unroll
            for (int j = 0; j < 8; ++j) wu[tq * 8 + j][tr] = w8[j];
        }
        for (int i = tid; i < RM * 32; i += kThreads) {
            const int r = i / 32, c = i % 32, sr = src[r];
            float v = 0.0f;
            if (sr >= 0) {
                const size_t at = (size_t) sr * a.ldx + (size_t) kb * 32 + c;
                v = GATE_UP ? a.xf[at] : __half2float(__ushort_as_half(a.xh[at]));
            }
            xs[r][c] = v;
        }
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < 32; ++kk) {
            float xv[RT], gw[TN], uw[TN];
#pragma unroll
            for (int i = 0; i < RT; ++i) xv[i] = xs[ty * RT + i][kk];
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                gw[j] = wg[kk][tx * TN + j];
                if (GATE_UP) uw[j] = wu[kk][tx * TN + j];
            }
#pragma unroll
            for (int i = 0; i < RT; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    ag[i][j] = fmaf(xv[i], gw[j], ag[i][j]);
                    if (GATE_UP) au[i][j] = fmaf(xv[i], uw[j], au[i][j]);
                }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < RT; ++i) {
        const int r = ty * RT + i;
        if (r >= nr) continue;
        const size_t lr = (size_t) (r0 + r);
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int c = c0 + tx * TN + j;
            if (GATE_UP) a.h[lr * a.ldo + c] = __half_as_ushort(__float2half_rn(swiglu(ag[i][j], au[i][j], a.limit)));
            else a.y[lr * a.ldo + c] = ag[i][j];
        }
    }
}

#if !defined(STRATA_USE_HIP)
// the same on the tensor cores (sm_70+): 8 warps each 16 rows x 32 outputs, the tiles F16 in shared memory
namespace wm = nvcuda::wmma;
constexpr int kLdT = 40;
template <int T, bool GATE_UP>
__global__ void __launch_bounds__(kThreads) set_tc_kernel(const SetArgs a) {
    struct Tiles {
        __half x[BM][kLdT], wg[BN][kLdT], wu[GATE_UP ? BN : 1][kLdT];
    };
    struct Out {
        float g[BM][BN + 4], u[GATE_UP ? BM : 1][BN + 4];
    };
    __shared__ __align__(32) unsigned char smem[sizeof(Tiles) > sizeof(Out) ? sizeof(Tiles) : sizeof(Out)];
    __shared__ uint32_t tab[Ty<T>::TAB > 0 ? Ty<T>::TAB : 1];
    __shared__ int src[BM];
    Tiles& Tl = *reinterpret_cast<Tiles*>(smem);
    Out& O = *reinterpret_cast<Out*>(smem);
    const int e = blockIdx.z;
    const int lo = a.bounds[e] - a.row0, hi = a.bounds[e + 1] - a.row0;
    const int r0 = lo + blockIdx.x * BM;
    if (r0 >= hi || hi - lo <= a.skip_le) return;   // (uniform; skip_le: the light kernels' experts)
    if (a.skip_ge > 0 && hi - lo >= a.skip_ge) return;   // (the caller's FP16 GEMMs')
    load_tab<T>(tab);
    const int nr = min(BM, hi - r0);
    const int c0 = blockIdx.y * BN;
    const uint8_t* blob = a.wbase + (size_t) e * a.stride;
    const int K = GATE_UP ? a.n_embd : a.n_ff;
    const size_t rb = row_bytes_t<T>(K);
    const int tid = (int) threadIdx.x, warp = tid >> 5;
    if (tid < BM) src[tid] = tid < nr ? (GATE_UP ? a.idx[r0 + tid] : r0 + tid) : -1;
    __syncthreads();
    const int wm_ = warp >> 1, wn = (warp & 1) * 2;
    wm::fragment<wm::accumulator, 16, 16, 16, float> fg[2], fu[2];
    for (int j = 0; j < 2; ++j) {
        wm::fill_fragment(fg[j], 0.0f);
        wm::fill_fragment(fu[j], 0.0f);
    }
    const int tr = tid >> 2, tq = tid & 3;
    for (int kb = 0; kb < K / 32; ++kb) {
        for (int m = 0; m < (GATE_UP ? 2 : 1); ++m) {
            float w8[8];
            deq8<T>(blob + a.tr[GATE_UP ? m : 2] + (size_t) (c0 + tr) * rb, kb, tq, tab, w8);
            __half* dst = m == 0 ? &Tl.wg[tr][tq * 8] : &Tl.wu[tr][tq * 8];
#pragma unroll
            for (int j = 0; j < 8; ++j) dst[j] = __float2half_rn(w8[j]);
        }
        const int sr = src[tr];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            __half v = __float2half_rn(0.0f);
            if (sr >= 0) {
                const size_t at = (size_t) sr * a.ldx + (size_t) kb * 32 + tq * 8 + j;
                v = GATE_UP ? __float2half_rn(a.xf[at]) : __ushort_as_half(a.xh[at]);
            }
            Tl.x[tr][tq * 8 + j] = v;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 32; kk += 16) {
            wm::fragment<wm::matrix_a, 16, 16, 16, __half, wm::row_major> fa;
            wm::load_matrix_sync(fa, &Tl.x[wm_ * 16][kk], kLdT);
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                wm::fragment<wm::matrix_b, 16, 16, 16, __half, wm::col_major> fb;
                wm::load_matrix_sync(fb, &Tl.wg[(wn + j) * 16][kk], kLdT);
                wm::mma_sync(fg[j], fa, fb, fg[j]);
                if (GATE_UP) {
                    wm::load_matrix_sync(fb, &Tl.wu[(wn + j) * 16][kk], kLdT);
                    wm::mma_sync(fu[j], fa, fb, fu[j]);
                }
            }
        }
        __syncthreads();
    }
    for (int j = 0; j < 2; ++j) {
        wm::store_matrix_sync(&O.g[wm_ * 16][(wn + j) * 16], fg[j], BN + 4, wm::mem_row_major);
        if (GATE_UP) wm::store_matrix_sync(&O.u[wm_ * 16][(wn + j) * 16], fu[j], BN + 4, wm::mem_row_major);
    }
    __syncthreads();
    for (int i = tid; i < BM * BN; i += kThreads) {
        const int r = i / BN, c = i % BN;
        if (r >= nr) continue;
        const size_t lr = (size_t) (r0 + r);
        if (GATE_UP) a.h[lr * a.ldo + c0 + c] = __half_as_ushort(__float2half_rn(swiglu(O.g[r][c], O.u[r][c], a.limit)));
        else a.y[lr * a.ldo + c0 + c] = O.g[r][c];
    }
}
#endif

// ---------------------------------------------------------------- GEMVs
constexpr int kMaxJobs = 8;
struct Jobs {
    MvJob j[kMaxJobs];
};
template <int T, int NC>
__global__ void __launch_bounds__(kThreads) mv_kernel(const Jobs J) {
    __shared__ uint32_t tab[Ty<T>::TAB > 0 ? Ty<T>::TAB : 1];
    load_tab<T>(tab);
    __syncthreads();
    const MvJob& b = J.j[blockIdx.y];
    const int row = blockIdx.x * kWarps + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= b.n) return;
    const uint8_t* w = b.w + (size_t) row * b.ld;
    const int nsb = b.k / 32;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
    for (int kb = lane; kb < nsb; kb += 32) {
        Sub s;
        decode<T>(w, kb, tab, s);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (c < b.ncol) {
                const int4* xp = (const int4*) (b.xq + (size_t) c * b.k + (size_t) kb * 32);
                acc[c] = fmaf(b.xd[(size_t) c * nsb + kb], sub_dot<T>(s, xp[0], xp[1]), acc[c]);
            }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c)
        if (c < b.ncol) {
            const float v = warp_sum(acc[c]);
            if (lane == 0) b.y[c * b.ycs + row] = b.alpha * v + (b.bias != nullptr ? b.bias[row] : 0.0f);
        }
}

// ---------------------------------------------------------------- the prompt sets' light experts
// A set's experts of a few rows (<= kLightRows: a prompt of a few hundred tokens spreads ~10 over most of 384 experts)
// in the token kernels' form: each weight sub-block decoded once and dotted with every row's int8 input (rounded per 32
// values as the token path rounds them) - the weights read once at the memory's speed, where a 64-row GEMM tile did 64
// rows' work for them.  Each expert by its own rows: the light kernels take the experts of <= kLightRows rows, the tiles
// the others (SetArgs::skip_le: they skip the light ones) - every expert's sums in one fixed order whatever the set.
// Expert e's row i is input row e kLightRows + i of the scratch, output (local) row bounds[e] - row0 + i.
constexpr int kLightRows = 16;
constexpr int kLightMaxExp = 512;   // a set's experts at most (the scratch: kLightMaxExp kLightRows rows)

// grid (k / 32 / 8, n_exp kLightRows): the light experts' rows of k values (gate / up: F32 x[idx[lr]]; down: F16 h[lr],
// ldx apart) -> int8 q[e kLightRows + i][k], scales d[..][k / 32]
__global__ void __launch_bounds__(kThreads) light_q8_kernel(const float* __restrict__ xf, const uint16_t* __restrict__ xh,
                                                           const int* __restrict__ idx, int ldx, int k,
                                                           const int* __restrict__ bounds, int row0,
                                                           int8_t* __restrict__ q, float* __restrict__ d) {
    const int e = blockIdx.y / kLightRows, i = blockIdx.y % kLightRows;
    const int rows = bounds[e + 1] - bounds[e];
    const int b = blockIdx.x * kWarps + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (i >= rows || rows > kLightRows || b * 32 >= k) return;
    const int lr = bounds[e] - row0 + i;
    const size_t at = (size_t) b * 32 + lane, qr = (size_t) blockIdx.y;
    const float v = xf != nullptr ? xf[(size_t) idx[lr] * ldx + at] : __half2float(__ushort_as_half(xh[(size_t) lr * ldx + at]));
    float am = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    const float sc = am / 127.0f;   // (ds4_gguf.cu's q8_quant: the same operations)
    q[qr * k + at] = (int8_t) (am > 0.0f ? __float2int_rn(v / sc) : 0);
    if (lane == 0) d[qr * (k / 32) + b] = sc;
}

// grid (n_ff / 8 or n_embd / 8, n_exp): a light expert's rows, a warp a row of the weights (gate and up, or down), every
// row of the expert a pass -> h16 (the F16 swiglu) or y
// one work item: the rows [tile kWarps, ...) of expert e (see light_kernel)
template <int T, int NE, bool GATE_UP, bool F>
__device__ __forceinline__ void light_item(const SetArgs& a, const int8_t* __restrict__ xq, const float* __restrict__ xd,
                                           const uint32_t* tab, int item, int ntile) {
    constexpr int kBPS = kQK<T> == 32 ? 1 : 8, kStep = 32 / kBPS;   // (as the token kernels: a lane keeps its sub-block position)
    const int K = GATE_UP ? a.n_embd : a.n_ff, N = GATE_UP ? a.n_ff : a.n_embd, nsb = K / 32, nblk = nsb / kBPS;
    const int lane = threadIdx.x & 31;
    const int ib0 = kBPS == 8 ? lane & 7 : 0, blk0 = kBPS == 8 ? lane >> 3 : lane;
    const int e = item / ntile, row = (item % ntile) * kWarps + (threadIdx.x >> 5);
    const int lo = a.bounds[e] - a.row0, ne = a.bounds[e + 1] - a.bounds[e];
    if (ne <= 0 || ne > NE || row >= N) return;   // (ne: uniform, the tiles take an expert of more rows)
    const size_t rb = row_bytes_t<T>(K);
    const uint8_t* blob = a.wbase + (size_t) e * a.stride;
    const size_t q0 = (size_t) e * kLightRows;
    const uint8_t* w0 = blob + a.tr[GATE_UP ? 0 : 2] + (size_t) row * rb;
    const uint8_t* w1 = blob + a.tr[1] + (size_t) row * rb;
    float a0[NE], a1[NE];
#pragma unroll
    for (int i = 0; i < NE; ++i) a0[i] = a1[i] = 0.0f;
    for (int blk = blk0; blk < nblk; blk += kStep) {
        const int kb = blk * kBPS + ib0;
        Sub s0, s1;
        decode_fast_b<T, F>(w0 + (size_t) blk * Ty<T>::B, ib0, tab, s0);
        if (GATE_UP) decode_fast_b<T, F>(w1 + (size_t) blk * Ty<T>::B, ib0, tab, s1);
#pragma unroll
        for (int i = 0; i < NE; ++i)
            if (i < ne) {
                const int4* xp = (const int4*) (xq + (q0 + i) * K + (size_t) kb * 32);
                const int4 xa = xp[0], xb = xp[1];
                const float dx = xd[(q0 + i) * nsb + kb];
                a0[i] = fmaf(dx, sub_dot<T>(s0, xa, xb), a0[i]);
                if (GATE_UP) a1[i] = fmaf(dx, sub_dot<T>(s1, xa, xb), a1[i]);
            }
    }
#pragma unroll
    for (int i = 0; i < NE; ++i)
        if (i < ne) {
            const float v0 = warp_sum(a0[i]);
            const float v1 = GATE_UP ? warp_sum(a1[i]) : 0.0f;
            if (lane == 0) {
                const size_t lr = (size_t) (lo + i);
                if (GATE_UP) a.h[lr * a.ldo + row] = __half_as_ushort(__float2half_rn(swiglu(v0, v1, a.limit)));
                else a.y[lr * a.ldo + row] = v0;
            }
        }
}

template <int T, int NE, bool GATE_UP>
__global__ void __launch_bounds__(kThreads) light_kernel(const SetArgs a, const int8_t* __restrict__ xq,
                                                                          const float* __restrict__ xd) {
    constexpr bool kF = NE <= 4;   // (the fast codebook only for the few-row kernels, which loop over work items round the grid)
    __shared__ __align__(16) uint32_t tab[kTabWords<T, kF>];
    const int N = GATE_UP ? a.n_ff : a.n_embd, ntile = (N + kWarps - 1) / kWarps;
    bool loaded = false;   // (the codebook when the first live item comes)
    for (int item = kPersist<T> ? blockIdx.x : 0; item < (kPersist<T> ? a.n_exp * ntile : 1); item += kPersist<T> ? gridDim.x : 1) {
        const int e = kPersist<T> ? item / ntile : blockIdx.y, ne = a.bounds[e + 1] - a.bounds[e];
        if (ne <= 0 || ne > NE) continue;   // (uniform)
        if (!loaded) { load_tab_fast<T, kF>(tab); __syncthreads(); loaded = true; }
        light_item<T, NE, GATE_UP, kF>(a, xq, xd, tab, kPersist<T> ? item : (int) (blockIdx.y * ntile + blockIdx.x), ntile);
    }
}

// The release's form, for the experts of more rows (5 .. 16: compute bound, registers tuned): grid (n_ff / 8 or n_embd / 8, n_exp), a light expert's rows, a warp a row of the weights (gate and up, or down), every
// row of the expert a pass -> h16 (the F16 swiglu) or y
template <int T, int NE, bool GATE_UP>
__global__ void __launch_bounds__(kThreads) light_kernel_rows(const SetArgs a, const int8_t* __restrict__ xq,
                                                          const float* __restrict__ xd) {
    __shared__ uint32_t tab[Ty<T>::TAB > 0 ? Ty<T>::TAB : 1];
    const int e = blockIdx.y;
    const int lo = a.bounds[e] - a.row0, ne = a.bounds[e + 1] - a.bounds[e];
    if (ne <= 0 || ne > NE) return;   // (uniform: the tiles take an expert of more rows)
    load_tab<T>(tab);
    __syncthreads();
    const int K = GATE_UP ? a.n_embd : a.n_ff, N = GATE_UP ? a.n_ff : a.n_embd, nsb = K / 32;
    const int row = blockIdx.x * kWarps + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= N) return;
    const uint8_t* blob = a.wbase + (size_t) e * a.stride;
    const size_t rb = row_bytes_t<T>(K);
    const uint8_t* w0 = blob + a.tr[GATE_UP ? 0 : 2] + (size_t) row * rb;
    const uint8_t* w1 = blob + a.tr[1] + (size_t) row * rb;
    const size_t q0 = (size_t) e * kLightRows;
    float a0[NE], a1[NE];
#pragma unroll
    for (int i = 0; i < NE; ++i) a0[i] = a1[i] = 0.0f;
    for (int kb = lane; kb < nsb; kb += 32) {
        Sub s0, s1;
        decode<T>(w0, kb, tab, s0);
        if (GATE_UP) decode<T>(w1, kb, tab, s1);
#pragma unroll
        for (int i = 0; i < NE; ++i)
            if (i < ne) {
                const int4* xp = (const int4*) (xq + (q0 + i) * K + (size_t) kb * 32);
                const int4 xa = xp[0], xb = xp[1];
                const float dx = xd[(q0 + i) * nsb + kb];
                a0[i] = fmaf(dx, sub_dot<T>(s0, xa, xb), a0[i]);
                if (GATE_UP) a1[i] = fmaf(dx, sub_dot<T>(s1, xa, xb), a1[i]);
            }
    }
#pragma unroll
    for (int i = 0; i < NE; ++i)
        if (i < ne) {
            const float v0 = warp_sum(a0[i]);
            const float v1 = GATE_UP ? warp_sum(a1[i]) : 0.0f;
            if (lane == 0) {
                const size_t lr = (size_t) (lo + i);
                if (GATE_UP) a.h[lr * a.ldo + row] = __half_as_ushort(__float2half_rn(swiglu(v0, v1, a.limit)));
                else a.y[lr * a.ldo + row] = v0;
            }
        }
}

bool tc_ok() {   // as ds4_gguf.cu's: sm_70 and later, STRATA_DS4_NO_TC=1 keeps the F32 kernel
#if defined(STRATA_USE_HIP)
    return false;
#else
    // (per device: the cards of a multi-GPU run need not be the same generation)
    static std::atomic<int> state[16];   // 0: unknown, 1: tensor cores, 2: the F32 kernel
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16) return false;
    int v = state[dev].load(std::memory_order_relaxed);
    if (v == 0) {
        int major = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        const char* e = std::getenv("STRATA_DS4_NO_TC");
        v = major >= 7 && !(e != nullptr && e[0] == '1') ? 1 : 2;
        state[dev].store(v, std::memory_order_relaxed);
    }
    return v == 1;
#endif
}

// ---------------------------------------------------------------- dispatch
// F(T) for the type t at run time; false when it is not one of these
#define DS4_IQ_SWITCH(t, F)          \
    switch (t) {                     \
        case kIQ2_XXS: F(kIQ2_XXS); break; \
        case kIQ2_XS: F(kIQ2_XS); break;   \
        case kIQ2_S: F(kIQ2_S); break;     \
        case kIQ3_XXS: F(kIQ3_XXS); break; \
        case kIQ3_S: F(kIQ3_S); break;     \
        case kIQ4_XS: F(kIQ4_XS); break;   \
        case kIQ1_S: F(kIQ1_S); break;     \
        case kIQ1_M: F(kIQ1_M); break;     \
        case kIQ4_NL: F(kIQ4_NL); break;   \
        case kQ4_K: F(kQ4_K); break;       \
        case kQ2_K: F(kQ2_K); break;       \
        case kQ3_K: F(kQ3_K); break;       \
        default: return false;       \
    }

// CUDA: a persistent grid (the resident blocks of the device, split over the groups), each block looping over row tiles with
// its codebook loaded once; HIP keeps the full grid (one tile a block)
template <auto Kernel>
int resident_ctas() {   // (a static a kernel: the occupancy differs by the codebook's shared memory)
#if defined(STRATA_USE_HIP) || defined(STRATA_METAL)
    return 1 << 30;
#else
    static std::atomic<int> cache[16];
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16) return 1 << 30;
    int v = cache[dev].load(std::memory_order_relaxed);
    if (v == 0) {
        int sms = 0, per = 0;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, Kernel, kThreads, 0);
        if (const char* e = std::getenv("DS4_IQ_MAX_SMS")) sms = std::min(sms, std::max(1, std::atoi(e)));   // (benchmarking: a smaller card)
        if (const char* e = std::getenv("DS4_IQ_CTA_PER_SM")) per = std::min(per, std::max(1, std::atoi(e)));
        v = std::max(1, sms * std::max(per, 1));
        cache[dev].store(v, std::memory_order_relaxed);
    }
    return v;
#endif
}
// The grid: CUDA takes the device's resident blocks (at most one a work item) and every block loops over the items (group, row
// tile) round the grid (group by group, so that the blocks work on one expert's rows together): whichever groups are active,
// they share all the blocks, and a block loads its codebook once.  HIP: a block an item (the full grid).
template <int T>
inline int items_grid(int items, int resident) { return std::max(1, std::min(items, resident)); }
template <int T, int NE>
void launch_gu(const Args& a, int groups, cudaStream_t s) {
    const int tiles = a.n_ff / (kWarps * kGuRW);
    if constexpr (kPersist<T>)
        strata_ds4_launch(gate_up_kernel<T, NE>, items_grid<T>(groups * tiles, resident_ctas<gate_up_kernel<T, NE, kGuRW>>()), kThreads, 0, s, a, groups);
    else
        strata_ds4_launch(gate_up_kernel<T, NE>, dim3((unsigned) tiles, (unsigned) groups), kThreads, 0, s, a, groups);
}
template <int T, int NE>
void launch_dn(const Args& a, int groups, cudaStream_t s) {
    const int tiles = a.n_embd / (kWarps * kDnRW);
    if constexpr (kPersist<T>)
        strata_ds4_launch(down_kernel<T, NE>, items_grid<T>(groups * tiles, resident_ctas<down_kernel<T, NE, kDnRW>>()), kThreads, 0, s, a, groups);
    else
        strata_ds4_launch(down_kernel<T, NE>, dim3((unsigned) tiles, (unsigned) groups), kThreads, 0, s, a, groups);
}
template <int T>
void gu_ne(const Args& a, int groups, int ne, cudaStream_t s) {
    if (ne <= 1) launch_gu<T, 1>(a, groups, s);
    else if (ne <= 4) launch_gu<T, 4>(a, groups, s);
    else if (ne <= 8) launch_gu<T, 8>(a, groups, s);
    else launch_gu<T, 16>(a, groups, s);
}
template <int T>
void dn_ne(const Args& a, int groups, int ne, cudaStream_t s) {
    if (ne <= 1) launch_dn<T, 1>(a, groups, s);
    else if (ne <= 4) launch_dn<T, 4>(a, groups, s);
    else if (ne <= 8) launch_dn<T, 8>(a, groups, s);
    else launch_dn<T, 16>(a, groups, s);
}
#include "ds4_iq_hip.inc"
template <int T, bool GU>
void launch_set(const SetArgs& a, dim3 grid, cudaStream_t s) {
#if defined(STRATA_USE_HIP)
    if (hipk::launch<T, GU>(a, s)) return;   // (gfx12, STRATA_DS4_HIPK=1: the matrix-core tiles)
#endif
#if !defined(STRATA_USE_HIP)
    if (tc_ok()) {
        strata_ds4_launch(set_tc_kernel<T, GU>, grid, kThreads, 0, s, a);
        return;
    }
#endif
    if (a.max_rows <= 16) {   // (a block of 64 rows would do 64 rows' FMAs for the few there are)
        grid.x = (unsigned) ((a.max_rows + 15) / 16);
        strata_ds4_launch(set_kernel<T, GU, 16>, grid, kThreads, 0, s, a);
        return;
    }
    strata_ds4_launch(set_kernel<T, GU>, grid, kThreads, 0, s, a);
}
// the light sets' int8 rows, per stream (kLightMaxExp kLightRows rows of k)
struct LightBuf {
    int8_t* q = nullptr;
    float* d = nullptr;
    size_t cap = 0;
};
std::mutex g_lmu;
std::map<std::pair<int, cudaStream_t>, LightBuf> g_light;   // (device, stream): the default stream 0 is one handle on every device
LightBuf* light_buf(cudaStream_t s, int k) {
    std::lock_guard<std::mutex> lk(g_lmu);
    int dev = 0;
    cudaGetDevice(&dev);
    LightBuf& b = g_light[{dev, s}];
    const size_t need = (size_t) kLightMaxExp * kLightRows * (size_t) std::max(k, 8192);
    if (b.cap < need) {
        if (b.q != nullptr) cudaFree(b.q);
        if (b.d != nullptr) cudaFree(b.d);
        b = LightBuf{};
        if (cudaMalloc(&b.q, need) != cudaSuccess || cudaMalloc(&b.d, need / 32 * sizeof(float)) != cudaSuccess) {
            std::fprintf(stderr, "ds4 iq: the light sets' scratch (%zu MB) failed\n", need >> 20);
            return nullptr;
        }
        b.cap = need;
    }
    return &b;
}
bool light_on() {   // STRATA_DS4_NO_LIGHT=1: every set through the GEMM tiles
    static const bool on = [] {
        const char* v = std::getenv("STRATA_DS4_NO_LIGHT");
        return !(v != nullptr && v[0] == '1');
    }();
    return on;
}
template <int T, bool GU>
void launch_light(const SetArgs& a, const LightBuf& b, cudaStream_t s) {
    const int tiles = ((GU ? a.n_ff : a.n_embd) + kWarps - 1) / kWarps;
    if (a.max_rows <= 4) {   // few rows: work items round a persistent grid (the codebook per block is then the cost)
        if constexpr (kPersist<T>)
            strata_ds4_launch(light_kernel<T, 4, GU>, items_grid<T>(a.n_exp * tiles, resident_ctas<light_kernel<T, 4, GU>>()), kThreads, 0, s, a, b.q, b.d);
        else   // (a small codebook: the release's kernel, which is the faster there)
            strata_ds4_launch(light_kernel_rows<T, 4, GU>, dim3((unsigned) tiles, (unsigned) a.n_exp), kThreads, 0, s, a, b.q, b.d);
    } else {   // more rows: compute bound, the release's kernel and grid
        const dim3 grid((unsigned) tiles, (unsigned) a.n_exp);
        if (a.max_rows <= 8) strata_ds4_launch(light_kernel_rows<T, 8, GU>, grid, kThreads, 0, s, a, b.q, b.d);
        else strata_ds4_launch(light_kernel_rows<T, kLightRows, GU>, grid, kThreads, 0, s, a, b.q, b.d);
    }
}
bool light_set(const SetArgs& a, bool gu, cudaStream_t s) {
    const int K = gu ? a.n_embd : a.n_ff;
    LightBuf* b = light_buf(s, K);
    if (b == nullptr) return false;
    const dim3 gq((unsigned) ((K / 32 + kWarps - 1) / kWarps), (unsigned) (a.n_exp * kLightRows));
    strata_ds4_launch(light_q8_kernel, gq, kThreads, 0, s, gu ? a.xf : nullptr, gu ? nullptr : a.xh, a.idx, a.ldx, K, a.bounds,
                                            a.row0, b->q, b->d);
    if (gu) {
        if (a.tg == kMXFP4) {
            launch_light<kMXFP4, true>(a, *b, s);
            return true;
        }
#define LG_(T) launch_light<T, true>(a, *b, s)
        DS4_IQ_SWITCH(a.tg, LG_)
#undef LG_
    } else {
        if (a.td == kMXFP4) {
            launch_light<kMXFP4, false>(a, *b, s);
            return true;
        }
#define LD_(T) launch_light<T, false>(a, *b, s)
        DS4_IQ_SWITCH(a.td, LD_)
#undef LD_
    }
    return true;
}

template <int T>
void mv_nc(const Jobs& J, int nj, int max_n, int max_c, cudaStream_t s) {
    const dim3 grid((unsigned) ((max_n + kWarps - 1) / kWarps), (unsigned) nj);
    if (max_c <= 1) strata_ds4_launch(mv_kernel<T, 1>, grid, kThreads, 0, s, J);
    else if (max_c <= 4) strata_ds4_launch(mv_kernel<T, 4>, grid, kThreads, 0, s, J);
    else strata_ds4_launch(mv_kernel<T, 16>, grid, kThreads, 0, s, J);
}


// ---------------------------------------------------------------- whole matrices as 16-bit rows (the chunked prompt's GEMMs)
// n rows of k values from `raw` (ld bytes a row): one thread a 32-value sub-block, the sub-block's codes and scales from decode<>
// (value = d s q - m of its half, ggml's dequantize_row_*), rounded to BF16 (bf16) or F16.
// The 32 values go through shared memory and are written coalesced (a warp's 32 threads cover 1024 contiguous values): a
// thread writing its own 32 values straight to dst wrote 2 bytes every 64 (a warp's store touches 32 cache lines), which
// measured 0.5 G weights/s on the R9700 - the 8K prompt's q_b stage 4 s.  DS4_DQ_OLD=1 keeps the old kernel for A/B.
template <int T>
__global__ void __launch_bounds__(kThreads) dq_rows_kernel2(const uint8_t* __restrict__ raw, size_t ld, int k, int n,
                                                            uint16_t* __restrict__ dst, bool bf16) {
    __shared__ uint32_t tab[Ty<T>::TAB > 0 ? Ty<T>::TAB : 1];
    __shared__ uint16_t sm[kThreads * 32];
    load_tab<T>(tab);
    __syncthreads();
    const int nsb = k / 32;
    const size_t total = (size_t) n * (size_t) nsb;   // 32-value sub-blocks
    const size_t vals = (size_t) n * (size_t) k;
    for (size_t base = (size_t) blockIdx.x * kThreads; base < total; base += (size_t) gridDim.x * kThreads) {
        const size_t i = base + threadIdx.x;
        uint16_t* o = sm + threadIdx.x * 32;
        if (i < total) {
            const size_t r = i / (size_t) nsb;
            const int kb = (int) (i - r * (size_t) nsb);
            Sub sb;
            sb.m0 = sb.m1 = 0.0f;
            decode<T>(raw + r * ld, kb, tab, sb);
            const float d0 = __fmul_rn(sb.d, (float) sb.s0), d1 = __fmul_rn(sb.d, (float) sb.s1);
#pragma unroll
            for (int j = 0; j < 32; ++j) {
                const int q = (int) (int8_t) ((uint32_t) sb.q[j >> 2] >> (8 * (j & 3)));
                const float v = j < 16 ? __fsub_rn(__fmul_rn(d0, (float) q), sb.m0) : __fsub_rn(__fmul_rn(d1, (float) q), sb.m1);
                o[j] = bf16 ? strata::ds4::bf16_rn(v) : __half_as_ushort(__float2half_rn(v));
            }
        } else {
#pragma unroll
            for (int j = 0; j < 32; ++j) o[j] = 0;
        }
        __syncthreads();
        // sm[m] is the block's value m (the sub-blocks are contiguous in the row-major output, rows included): write it
        // a thread's 32 values apart, so a warp's stores are 512 contiguous bytes each
        const size_t go = base * 32;
        uint16_t* d = dst + go;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            const size_t m = (size_t) j * kThreads + threadIdx.x;
            if (go + m < vals) d[m] = sm[m];
        }
        __syncthreads();
    }
}

template <int T>
__global__ void __launch_bounds__(kThreads) dq_rows_kernel(const uint8_t* __restrict__ raw, size_t ld, int k, int n,
                                                           uint16_t* __restrict__ dst, bool bf16) {
    __shared__ uint32_t tab[Ty<T>::TAB > 0 ? Ty<T>::TAB : 1];
    load_tab<T>(tab);
    __syncthreads();
    const int nsb = k / 32;
    const size_t total = (size_t) n * (size_t) nsb;
    // A warp stages its 32 sub-blocks in shared through contiguous uint4 loads and decodes from there: a lane reading its own
    // 176-byte block at that stride touched 32 scattered sectors per instruction - the dequant's remaining cost once the stores
    // were vectorised (15.4 s of a 4K prompt, 23 GB of quantised input at ~1.5 GB/s).  Each lane decodes with the index relative
    // to the staged buffer (kb - kb0), which is exactly equivalent because kb0 is a multiple of the 8 sub-blocks in a block.
    constexpr int kQKv = kQK<T>;
    constexpr int kSpan = (kQKv == 32 ? 32 : 4) * Ty<T>::B;      // bytes of a warp's 32 sub-blocks
    constexpr bool kCanStage = Ty<T>::TAB == 0;                  // (a codebook table already fills shared for the i-quants)
    const bool staged = (nsb % 32 == 0) && kCanStage;
    __shared__ uint4 sin4[kCanStage ? (kThreads / 32) * 44 : 1];
    const int lane32 = threadIdx.x & 31, wq = threadIdx.x >> 5;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t) gridDim.x * blockDim.x) {
        const size_t r = i / (size_t) nsb;
        const int kb = (int) (i - r * (size_t) nsb);
        Sub sb;
        sb.m0 = sb.m1 = 0.0f;
        if (staged) {
            const size_t wbase = i - (size_t) lane32;   // (uniform in the warp: blockDim and the stride are multiples of 32)
            const size_t r0 = wbase / (size_t) nsb;
            const int kb0 = (int) (wbase - r0 * (size_t) nsb);
            const uint8_t* srcp = raw + r0 * ld + (size_t) (kb0 * Ty<T>::B) / (kQKv / 32);
            uint4* st = sin4 + wq * 44;
            for (int c = lane32; c < kSpan / 16; c += 32) st[c] = ((const uint4*) srcp)[c];
            __threadfence_block();
            __syncwarp();
            decode<T>((const uint8_t*) st, kb - kb0, tab, sb);
        } else
            decode<T>(raw + r * ld, kb, tab, sb);
        const float d0 = __fmul_rn(sb.d, (float) sb.s0), d1 = __fmul_rn(sb.d, (float) sb.s1);
        uint16_t* o = dst + r * (size_t) k + (size_t) kb * 32;
        // (the 32 halves packed into four 16-byte stores: 32 two-byte stores a lane, each lane 64 bytes from the next, write
        // ~16x the bytes they carry in DRAM sectors - measured as the dense dequant's whole cost: 6.6 GB to FP16 in 20.0 s)
        uint32_t hw[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int q0 = (int) (int8_t) ((uint32_t) sb.q[j >> 1] >> (8 * ((2 * j) & 3)));
            const int q1 = (int) (int8_t) ((uint32_t) sb.q[j >> 1] >> (8 * ((2 * j + 1) & 3)));
            const bool lo = (2 * j) < 16;
            const float v0 = lo ? __fsub_rn(__fmul_rn(d0, (float) q0), sb.m0) : __fsub_rn(__fmul_rn(d1, (float) q0), sb.m1);
            const bool lo1 = (2 * j + 1) < 16;
            const float v1 = lo1 ? __fsub_rn(__fmul_rn(d0, (float) q1), sb.m0) : __fsub_rn(__fmul_rn(d1, (float) q1), sb.m1);
            const uint32_t h0 = bf16 ? strata::ds4::bf16_rn(v0) : __half_as_ushort(__float2half_rn(v0));
            const uint32_t h1 = bf16 ? strata::ds4::bf16_rn(v1) : __half_as_ushort(__float2half_rn(v1));
            hw[j] = h0 | (h1 << 16);
        }
        uint4* o4 = (uint4*) o;
        o4[0] = make_uint4(hw[0], hw[1], hw[2], hw[3]);
        o4[1] = make_uint4(hw[4], hw[5], hw[6], hw[7]);
        o4[2] = make_uint4(hw[8], hw[9], hw[10], hw[11]);
        o4[3] = make_uint4(hw[12], hw[13], hw[14], hw[15]);
    }
}

}  // namespace

// Which dense dequantizer: dq_rows_kernel2 (the coalesced transpose, 0.2.7 DS Support: 566 -> 592 ppts on the R9700, ids identical) or the
// row-per-lane dq_rows_kernel, which main's fd33cb6ee gave 16-byte stores and a staged input: the Strix Halo (gfx11) 4K
// prompt 79.5 -> 88.5 tok/s with it, so a gfx11 device defaults to it.  DS4_DQ_OLD=1 forces it anywhere, DS4_DQ_OLD=0 forces kernel2
// anywhere.  The detection is the device's gcnArchName (HIP only; a CUDA build has no gfx11).  Decided once per device.
static bool dq_use_old_kernel() {
    const char* dq_env = std::getenv("DS4_DQ_OLD");
    const int env = dq_env != nullptr && *dq_env != '\0' ? (std::atoi(dq_env) != 0 ? 1 : 0) : -1;
    const bool v026 = strata::ds4::defaults_026();   // DS4_DEFAULTS=026: 0.2.6's row-per-lane dequantizer (the same values, bit for bit)
    if (env >= 0 || v026) return strata::ds4::dq_old_default(env, false, v026);
#if defined(STRATA_USE_HIP)
    static std::atomic<int> cache[64];   // 0 unknown, 1 no, 2 yes
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) return false;
    int c = cache[dev].load();
    if (c == 0) {
        cudaDeviceProp prop{};
        const bool yes = cudaGetDeviceProperties(&prop, dev) == cudaSuccess && std::strncmp(prop.gcnArchName, "gfx11", 5) == 0;
        c = yes ? 2 : 1;
        cache[dev].store(c);
    }
    return c == 2;
#else
    return false;
#endif
}

bool dequant_rows(int type, const void* raw, size_t ld, int k, int n, uint16_t* dst, bool bf16, cudaStream_t s) {
    if (raw == nullptr || k % 256 != 0 || n <= 0) return false;
    const unsigned blocks = (unsigned) std::min<size_t>(((size_t) n * (size_t) (k / 32) + kThreads - 1) / kThreads, 65535);
    const bool old = dq_use_old_kernel();
#define DS4_DQ(T) do {                                                                                    \
        if (old)                                                                                          \
            strata_ds4_launch(dq_rows_kernel<T>, blocks, kThreads, 0, s, (const uint8_t*) raw, ld, k, n, dst, bf16);      \
        else                                                                                              \
            strata_ds4_launch(dq_rows_kernel2<T>, blocks, kThreads, 0, s, (const uint8_t*) raw, ld, k, n, dst, bf16);     \
    } while (0)
    switch (type) {
        case kIQ2_XXS: DS4_DQ(kIQ2_XXS); break;
        case kIQ2_XS: DS4_DQ(kIQ2_XS); break;
        case kIQ2_S: DS4_DQ(kIQ2_S); break;
        case kIQ3_XXS: DS4_DQ(kIQ3_XXS); break;
        case kIQ3_S: DS4_DQ(kIQ3_S); break;
        case kIQ4_XS: DS4_DQ(kIQ4_XS); break;
        case kIQ1_S: DS4_DQ(kIQ1_S); break;
        case kIQ1_M: DS4_DQ(kIQ1_M); break;
        case kQ4_K: DS4_DQ(kQ4_K); break;
        case kQ5_K: DS4_DQ(kQ5_K); break;
        case kQ6_K: DS4_DQ(kQ6_K); break;
        case kQ2_K: DS4_DQ(kQ2_K); break;
        case kQ3_K: DS4_DQ(kQ3_K); break;
        default: return false;
    }
#undef DS4_DQ
    return cudaGetLastError() == cudaSuccess;
}

bool supported(int t) {
#if defined(STRATA_METAL)
    // The MSL kernels (metal/src/engine/msl/ds4_iq.metal) implement the 2.0-bit types plus IQ1_M; IQ1_S and IQ4_NL
    // have CUDA kernels here but no Metal ones.  Saying so makes those layers "cpu only" (ds4_model.cu, expert_layout),
    // so their experts stay on the CPU lane instead of stopping in "kernel launch with no Metal dispatch".
    if (t == kIQ4_NL) return false;   // (IQ1_S: ds4_iq1s_mvn + the tile kernel since the Unsloth sets)
#endif
    // the types the kernels below are instantiated for (DS4_IQ_SWITCH + the trellis ones + the dense K-quants)
    switch (t) {
        case kIQ2_XXS: case kIQ2_XS: case kIQ2_S: case kIQ3_XXS: case kIQ3_S: case kIQ4_XS:
        case kIQ1_S: case kIQ1_M: case kIQ4_NL:
        case kQ4_K: case kQ5_K: case kQ6_K: case kQ2_K: case kQ3_K:
            return true;
        default:
            return is_tq(t);
    }
}

bool known(int t) { return block_bytes(t) > 0; }

int block_values(int t) {
    if (t == kIQ4_NL) return 32;   // (ggml's QK4_NL: the only type here whose block is not 256 values)
    return is_tq(t) ? 128 : block_bytes(t) > 0 ? 256 : 0;
}

int block_bytes(int t) {
    if (is_tq(t)) return 2 + 4 * tq_kbits(t);
    switch (t) {
        case kIQ2_XXS: return Ty<kIQ2_XXS>::B;
        case kIQ2_XS: return Ty<kIQ2_XS>::B;
        case kIQ2_S: return Ty<kIQ2_S>::B;
        case kIQ3_XXS: return Ty<kIQ3_XXS>::B;
        case kIQ3_S: return Ty<kIQ3_S>::B;
        case kIQ4_XS: return Ty<kIQ4_XS>::B;
        case kQ4_K: return Ty<kQ4_K>::B;
        case kQ5_K: return Ty<kQ5_K>::B;
        case kQ6_K: return Ty<kQ6_K>::B;
        case kQ2_K: return Ty<kQ2_K>::B;
        case kQ3_K: return Ty<kQ3_K>::B;
        case kIQ1_S: return Ty<kIQ1_S>::B;
        case kIQ1_M: return Ty<kIQ1_M>::B;
        case kIQ4_NL: return Ty<kIQ4_NL>::B;
        default: return 0;
    }
}

const char* name(int t) {
    switch (t) {
        case kTQK6: return "TQK6";
        case kTQK7: return "TQK7";
        case kTQK5X: return "TQK5X";
        case kTQK4X: return "TQK4X";
        case kTQK5T: return "TQK5T";
        case kTQK4T: return "TQK4T";
        case kIQ2_XXS: return "IQ2_XXS";
        case kIQ2_XS: return "IQ2_XS";
        case kIQ2_S: return "IQ2_S";
        case kIQ3_XXS: return "IQ3_XXS";
        case kIQ3_S: return "IQ3_S";
        case kIQ4_XS: return "IQ4_XS";
        case kQ4_K: return "Q4_K";
        case kQ5_K: return "Q5_K";
        case kQ6_K: return "Q6_K";
        case kQ2_K: return "Q2_K";
        case kQ3_K: return "Q3_K";
        case kIQ1_S: return "IQ1_S";
        case kIQ1_M: return "IQ1_M";
        case kIQ4_NL: return "IQ4_NL";
        default: return "?";
    }
}

bool gate_up(const Args& a, int groups, int ne_max, cudaStream_t s) {
#if defined(STRATA_METAL)
    // gate_up_kernel on the Metal dispatch (the prompt / window path's expert GEMM); the launch below is the fallback
    if (::strata::dsv41::metal_host::iq_gu(a.plan_ptr, a.grp_ptr, a.grp_start, a.grp_ent, a.n_groups, a.tr, a.n_embd,
                                           a.n_ff, a.limit, a.xq, a.xd, a.h, a.tg, groups))
        return true;
#endif
    if (is_tq(a.tg)) return tq::gate_up(a, groups, ne_max, s);
    if (a.n_embd % 256 != 0 || a.n_ff % (kWarps * kGuRW) != 0) return false;
#define GU_(T) gu_ne<T>(a, groups, ne_max, s)
    DS4_IQ_SWITCH(a.tg, GU_)
#undef GU_
    return true;
}

bool down(const Args& a, int groups, int ne_max, cudaStream_t s) {
#if defined(STRATA_METAL)
    // down_kernel on the Metal dispatch; the launch below is the fallback
    if (::strata::dsv41::metal_host::iq_dn(a.plan_ptr, a.grp_ptr, a.grp_start, a.grp_ent, a.n_groups, a.tr, a.n_embd,
                                           a.n_ff, a.hq, a.hd, a.part, a.td, groups))
        return true;
#endif
    if (is_tq(a.td)) return tq::down(a, groups, ne_max, s);
    if (a.n_ff % 256 != 0 || a.n_embd % (kWarps * kDnRW) != 0) return false;
#define DN_(T) dn_ne<T>(a, groups, ne_max, s)
    DS4_IQ_SWITCH(a.td, DN_)
#undef DN_
    return true;
}

bool light_gate_up(const SetArgs& a, cudaStream_t s) {
    if (is_tq(a.tg)) return false;   // (the trellis sets take the tiles for every expert)
    if (a.n_exp > kLightMaxExp || !light_on() || a.n_embd % 256 != 0 || a.n_ff % 256 != 0) return false;
    return light_set(a, true, s);
}
bool light_down(const SetArgs& a, cudaStream_t s) {
    if (is_tq(a.td)) return false;
    if (a.n_exp > kLightMaxExp || !light_on() || a.n_embd % 256 != 0 || a.n_ff % 256 != 0) return false;
    return light_set(a, false, s);
}

bool hipk_force(int mode) {
#if defined(STRATA_USE_HIP)
    hipk::force_mode().store(mode, std::memory_order_relaxed);
    return mode < 0 || hipk::ok_device() || mode == 0;
#else
    (void) mode;
    return false;
#endif
}
long long hipk_launches() {
#if defined(STRATA_USE_HIP)
    return hipk::launches().load(std::memory_order_relaxed);
#else
    return -1;
#endif
}

bool set_gate_up(const SetArgs& a, cudaStream_t s) {
#if defined(STRATA_METAL)
    if (a.hbounds != nullptr && ::strata::dsv41::metal_host::iq_set_gu(a.wbase, a.stride, a.tr, a.hbounds, a.n_exp, a.total_rows, a.n_embd, a.n_ff, a.limit, a.xf, a.ldx, a.idx, a.h, a.ldo, a.tg, a.td))
        return true;
#endif
    if (is_tq(a.tg)) return tq::set_gate_up(a, s);
    if (a.n_embd % 256 != 0 || a.n_ff % BN != 0) return false;
    SetArgs t = a;
    if (light_gate_up(a, s)) {   // the experts of a few rows; the tiles the others
        if (a.max_rows <= kLightRows) return true;
        t.skip_le = kLightRows;
    }
    const dim3 grid((unsigned) ((a.max_rows + BM - 1) / BM), (unsigned) (a.n_ff / BN), (unsigned) a.n_exp);
#define SG_(T) launch_set<T, true>(t, grid, s)
    DS4_IQ_SWITCH(a.tg, SG_)
#undef SG_
    return true;
}

bool set_down(const SetArgs& a, cudaStream_t s) {
#if defined(STRATA_METAL)
    if (a.hbounds != nullptr && ::strata::dsv41::metal_host::iq_set_dn(a.wbase, a.stride, a.tr, a.hbounds, a.n_exp, a.total_rows, a.n_embd, a.n_ff, a.limit, a.xh, a.ldx, a.y, a.ldo, a.tg, a.td))
        return true;
#endif
    if (is_tq(a.td)) return tq::set_down(a, s);
    if (a.n_ff % 256 != 0 || a.n_embd % BN != 0) return false;
    SetArgs t = a;
    if (light_down(a, s)) {
        if (a.max_rows <= kLightRows) return true;
        t.skip_le = kLightRows;
    }
    const dim3 grid((unsigned) ((a.max_rows + BM - 1) / BM), (unsigned) (a.n_embd / BN), (unsigned) a.n_exp);
#define SD_(T) launch_set<T, false>(t, grid, s)
    DS4_IQ_SWITCH(a.td, SD_)
#undef SD_
    return true;
}

// ---------------------------------------------------------------- the busy experts' FP16 weights (f16_rows)
// 8 values a thread: row r's sub-block kb, piece p - deq8's values rounded as the tensor-core tiles round them
template <int T>
__global__ void __launch_bounds__(kThreads) f16_rows_kernel(const uint8_t* __restrict__ w, int k, int n,
                                                             uint16_t* __restrict__ dst, int ldd) {
    __shared__ uint32_t tab[Ty<T>::TAB > 0 ? Ty<T>::TAB : 1];
    load_tab<T>(tab);
    __syncthreads();
    const int nq = k / 8;
    const size_t rb = row_bytes_t<T>(k);
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < (long long) n * nq;
         i += (long long) gridDim.x * blockDim.x) {
        const int r = (int) (i / nq), q = (int) (i % nq);
        float w8[8];
        deq8<T>(w + (size_t) r * rb, q >> 2, q & 3, tab, w8);
        union {
            uint4 v;
            uint16_t h[8];
        } u;
#pragma unroll
        for (int j = 0; j < 8; ++j) u.h[j] = __half_as_ushort(__float2half_rn(w8[j]));
        *reinterpret_cast<uint4*>(dst + (size_t) r * ldd + (size_t) q * 8) = u.v;
    }
}
__global__ void __launch_bounds__(kThreads) gather_f16_kernel(const uint16_t* __restrict__ x, int ldx,
                                                               const int* __restrict__ idx, int n, int k,
                                                               uint16_t* __restrict__ dst) {
    const int nq = k / 8;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < (long long) n * nq;
         i += (long long) gridDim.x * blockDim.x) {
        const int r = (int) (i / nq), q = (int) (i % nq);
        *reinterpret_cast<uint4*>(dst + (size_t) r * k + (size_t) q * 8) =
            *reinterpret_cast<const uint4*>(x + (size_t) idx[r] * ldx + (size_t) q * 8);
    }
}
__global__ void __launch_bounds__(kThreads) swiglu_f16_kernel(const float* __restrict__ gu, int ld, int n, int ff,
                                                               float limit, uint16_t* __restrict__ h, int ldh) {
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < (long long) n * ff;
         i += (long long) gridDim.x * blockDim.x) {
        const int r = (int) (i / ff), c = (int) (i % ff);
        const float* g = gu + (size_t) r * ld;
        h[(size_t) r * ldh + c] = __half_as_ushort(__float2half_rn(swiglu(g[c], g[ff + c], limit)));
    }
}
static unsigned grid_for(long long items) {
    return (unsigned) std::max<long long>(1, std::min<long long>(4096, (items + kThreads - 1) / kThreads));
}
bool f16_rows_ok(int type) {   // (the DS4_IQ_SWITCH list)
    switch (type) {
        case kIQ2_XXS: case kIQ2_XS: case kIQ2_S: case kIQ3_XXS: case kIQ3_S: case kIQ4_XS: case kIQ1_S: case kIQ1_M:
        case kIQ4_NL: case kQ4_K: case kQ2_K: case kQ3_K:
            return true;
        default:
            return false;
    }
}
bool f16_rows(int type, const uint8_t* w, int k, int n, uint16_t* dst, int ldd, cudaStream_t s) {
    if (k % 256 != 0 || n < 1 || ldd % 8 != 0 || ((uintptr_t) dst & 15) != 0) return false;
    const unsigned blocks = grid_for((long long) n * (k / 8));
#define FR_(T) strata_ds4_launch(f16_rows_kernel<T>, blocks, kThreads, 0, s, w, k, n, dst, ldd)
    DS4_IQ_SWITCH(type, FR_)
#undef FR_
    return cudaGetLastError() == cudaSuccess;
}
void gather_f16(const uint16_t* x, int ldx, const int* idx, int n, int k, uint16_t* dst, cudaStream_t s) {
    strata_ds4_launch(gather_f16_kernel, grid_for((long long) n * (k / 8)), kThreads, 0, s, x, ldx, idx, n, k, dst);
}
void swiglu_f16(const float* gu, int ld, int n, int ff, float limit, uint16_t* h, int ldh, cudaStream_t s) {
    strata_ds4_launch(swiglu_f16_kernel, grid_for((long long) n * ff), kThreads, 0, s, gu, ld, n, ff, limit, h, ldh);
}

bool mv(const MvJob* jobs, int n, cudaStream_t s) {
#if defined(STRATA_METAL)
    if (n > 0 && n <= 64 && ::strata::dsv41::metal_host::iq_mv((const void*) jobs, n)) return true;
#endif
    {   // the trellis jobs go to their kernels, the others on
        MvJob tqj[64], oth[64];
        int nt = 0, no = 0;
        if (n > 64) return false;
        for (int i = 0; i < n; ++i) (is_tq(jobs[i].type) ? tqj[nt++] : oth[no++]) = jobs[i];
        if (nt > 0) {
            if (!tq::mv(tqj, nt, s)) return false;
            if (no == 0) return true;
            return mv(oth, no, s);
        }
    }
    // the jobs by type, up to kMaxJobs a launch
    bool done[64] = {};
    if (n > 64) return false;
    for (int i = 0; i < n; ++i)
        while (!done[i]) {
            const int t = jobs[i].type;
            Jobs J{};
            int nj = 0, max_n = 0, max_c = 1;
            for (int k = i; k < n && nj < kMaxJobs; ++k) {
                if (done[k] || jobs[k].type != t) continue;
                if (jobs[k].k % 256 != 0 || jobs[k].ncol < 1 || jobs[k].ncol > 16) return false;
                J.j[nj++] = jobs[k];
                max_n = std::max(max_n, jobs[k].n);
                max_c = std::max(max_c, jobs[k].ncol);
                done[k] = true;
            }
#define MV_(T) mv_nc<T>(J, nj, max_n, max_c, s)
            switch (t) {   // the experts' types and the dense matrices' 5- and 6-bit ones
                case kQ5_K: MV_(kQ5_K); break;
                case kQ6_K: MV_(kQ6_K); break;
                default: DS4_IQ_SWITCH(t, MV_)
            }
#undef MV_
        }
    return true;
}

}  // namespace strata::ds4::gguf::iq
