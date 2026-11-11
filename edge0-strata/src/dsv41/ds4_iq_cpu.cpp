// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_iq_cpu.cpp - a GGUF's low-bit experts on the CPU lane (see ds4_gguf_cpu.hpp): the GLM CPU lane's native
// experts (src/kernels/cpu/native_expert.cpp), whose arithmetic is ggml-cpu's - and for IQ3_XXS / IQ3_S (Maya-S's
// down, Maya-M's gate / up / down) rows of our own, the same numbers bit for bit, faster.
#include "ds4_gguf_cpu.hpp"
// Strata port: the low-bit (IQ) experts have no CPU lane here (they need Maya's native_expert split-base rows); the stub
// branch below answers for them.  MXFP4, the source format, has its own CPU rows (ds4_gguf_cpu.cpp).
#undef STRATA_NATIVE_EXPERTS
#if defined(STRATA_NATIVE_EXPERTS)
#include "strata/kernels/cpu/native_expert.hpp"  // (only with STRATA_NATIVE_EXPERTS)

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include "ggml-cpu.h"
#include "ggml.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#define IQ_X86 1
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif
#if defined(IQ_X86) && (defined(__GNUC__) || defined(__clang__))
#define IQ_AVX2 __attribute__((target("avx2,fma")))
#else
#define IQ_AVX2
#endif

namespace strata::ds4::gguf {

#if defined(STRATA_NATIVE_EXPERTS)
namespace kc = strata::dsv41k::cpu;
namespace {
kc::NativeFmt native(const IqCpu& f) {   // (the fields the rows read; no lookups)
    kc::NativeFmt n;
    n.gu_type = f.tg;
    n.d_type = f.td;
    n.gu_act = f.gu_act;
    n.d_act = f.d_act;
    n.n_embd = f.n_embd;
    n.n_ff = f.n_ff;
    n.gu_row = f.gu_row;
    n.d_row = f.d_row;
    n.up_off = f.gu_row * (size_t) f.n_ff;
    n.down_off = 2 * n.up_off;
    n.bytes = n.down_off + f.d_row * (size_t) f.n_embd;
    n.act_bytes = f.act_bytes;
    n.h_bytes = f.h_bytes;
    return n;
}

// ---- what this CPU runs: AVX2 + FMA (the OS saving the YMM state), and AVX-512 F / BW + VBMI with BMI2 (the ZMM state)
struct IqCpuIsa {
    bool avx2 = false, w512 = false;
};
IqCpuIsa iq_isa() {
    IqCpuIsa c;
#if defined(IQ_X86)
    unsigned r[4] = {0, 0, 0, 0};
    auto cpuid = [&](unsigned leaf, unsigned sub) {
#if defined(_MSC_VER)
        int x[4];
        __cpuidex(x, (int) leaf, (int) sub);
        for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
        __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
    };
    cpuid(0, 0);
    if (r[0] < 7) return c;
    cpuid(1, 0);
    if (!((r[2] >> 12) & 1u) || !((r[2] >> 27) & 1u) || !((r[2] >> 28) & 1u)) return c;   // FMA, OSXSAVE, AVX
#if defined(_MSC_VER)
    const unsigned long long xcr0 = _xgetbv(0);
#else
    unsigned lo = 0, hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
    if ((xcr0 & 0x6) != 0x6) return c;
    cpuid(7, 0);
    c.avx2 = (r[1] >> 5) & 1u;
    c.w512 = c.avx2 && (xcr0 & 0xE6) == 0xE6 && ((r[1] >> 16) & 1u) && ((r[1] >> 30) & 1u) && ((r[2] >> 1) & 1u) &&
             ((r[1] >> 8) & 1u);   // AVX-512 F, BW, VBMI; BMI2
#endif
    return c;
}
const IqCpuIsa& isa() {
    static const IqCpuIsa c = iq_isa();
    return c;
}

// ---- IQ3_XXS / IQ3_S rows of our own.  ggml-cpu's AVX2 dots build each 32-value chunk's grid magnitudes from eight
// scalar table loads and inserts, and redo it for every token; here the grid comes in with one vpgatherdd from the
// same table, the signs are built in registers (the GLM rows' sgn7_vec / sgn_vec), and every token of a call applies
// the decoded chunk.  The arithmetic is ggml's to the bit: the same exact int32 lanes per block (lane l: the chunks'
// grid entries l, each times its 2 s + 1), then for each block fma(d_x d_y, lanes, acc) and hsum_float_8 (IQ3_XXS:
// times 0.25) - checked against ggml-cpu's own dot once per type (iq3_check), else ggml's rows are used.
inline float h2f(uint16_t h) {   // F16 -> F32, exactly
    const uint32_t s = (uint32_t) (h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    uint32_t b;
    if (e == 0) {
        const float f = (float) m * 5.9604644775390625e-8f;   // m 2^-24 (exact)
        std::memcpy(&b, &f, 4);
        b |= s;
    } else if (e == 31) {
        b = s | 0x7f800000u | (m << 13);
    } else {
        b = s | ((e + 112u) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}
inline uint32_t u32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

#if defined(IQ_X86)
IQ_AVX2 inline float hsum_ggml(__m256 x) {   // ggml's hsum_float_8, the same additions in the same order
    __m128 r = _mm256_extractf128_ps(x, 1);
    r = _mm_add_ps(r, _mm256_castps256_ps128(x));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}
// IQ2_S / IQ3_S: 32 explicit sign bits -> 32 bytes of -1 / +1 (ggml's bit selector)
IQ_AVX2 inline __m256i sgn32(uint32_t m) {
    const __m256i sh = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,   //
                                        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i sel = _mm256_set1_epi64x((long long) 0x8040201008040201ull);
    const __m256i b = _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int) m), sh), sel);
    return _mm256_or_si256(_mm256_cmpeq_epi8(b, sel), _mm256_set1_epi8(1));
}
// table entries into a vector without the shuffle port (ggml's _mm256_set_epi64x / _epi32 of table loads ends up in
// inserts, one shuffle each; here each entry is a broadcast load and the lanes are put together by dword blends)
IQ_AVX2 inline __m256i bq(const void* p) { return _mm256_castpd_si256(_mm256_broadcast_sd((const double*) p)); }
IQ_AVX2 inline __m256i bd(const void* p) { return _mm256_castps_si256(_mm256_broadcast_ss((const float*) p)); }
IQ_AVX2 inline __m256i tab4(const uint64_t* t, unsigned a, unsigned b, unsigned c, unsigned d) {   // qwords t[a] t[b] t[c] t[d]
    return _mm256_blend_epi32(_mm256_blend_epi32(bq(t + a), bq(t + b), 0x0C), _mm256_blend_epi32(bq(t + c), bq(t + d), 0xC0),
                              0xF0);
}
IQ_AVX2 inline __m256i tab8(const uint32_t* t, const unsigned* i) {   // dwords t[i[0]] .. t[i[7]]
    const __m256i a = _mm256_blend_epi32(bd(t + i[0]), bd(t + i[1]), 0x02), b = _mm256_blend_epi32(bd(t + i[2]), bd(t + i[3]), 0x08);
    const __m256i c = _mm256_blend_epi32(bd(t + i[4]), bd(t + i[5]), 0x20), e = _mm256_blend_epi32(bd(t + i[6]), bd(t + i[7]), 0x80);
    return _mm256_blend_epi32(_mm256_blend_epi32(a, b, 0x0C), _mm256_blend_epi32(c, e, 0xC0), 0xF0);
}
// the 8 sign bytes (-1 / +1) of each 7-bit sign index with its parity bit: ggml's keven_signs_q2xs, as +-1 bytes
alignas(64) uint64_t g_ks64[128];
void build_ks64() {
    for (int i = 0; i < 128; ++i) {
        int par = 0;
        for (int b = 0; b < 7; ++b) par ^= (i >> b) & 1;
        const int s = i | (par << 7);
        uint64_t v = 0;
        for (int j = 0; j < 8; ++j) v |= (uint64_t) (((s >> j) & 1) ? 0xFFu : 0x01u) << (8 * j);
        g_ks64[i] = v;
    }
}
IQ_AVX2 inline __m256i sgn7t(uint32_t w) {   // a chunk's four 7-bit sign indices (bits 0..27 of w)
    return tab4(g_ks64, w & 127u, (w >> 7) & 127u, (w >> 14) & 127u, (w >> 21) & 127u);
}
IQ_AVX2 inline __m256i sc16(int a, int b) {   // int16 lanes 0-7 a (values 0-15), 8-15 b (values 16-31)
    return _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_set1_epi16((short) a)), _mm_set1_epi16((short) b), 1);
}
bool g_gather18 = false, g_gather21 = false;   // the IQ3_XXS / IQ3_S grids by vpgatherdd (own(): if faster here)
// IQ2_XXS's grid entry with its signs applied, for every grid index g and sign index i, at g << 7 | i (256 KB, built on
// first use): one lookup instead of two; the products are then (q8 + 128) x signed grid less 128 x the grid's pair sums
// (vpmaddubsw takes u8 x s8), the same integers as ggml's |grid| x signed q8
alignas(64) uint64_t g_ct2x[256 * 128];
bool g_ct16 = false;   // (own(): if faster here)
void build_ct2x() {
    for (int g = 0; g < 256; ++g)
        for (int i = 0; i < 128; ++i) {
            const uint64_t sg = g_ks64[i];   // (+-1 bytes)
            uint64_t v = 0;
            for (int j = 0; j < 8; ++j) {
                const int gb = (int) ((iq2xxs_grid[g] >> (8 * j)) & 255u);
                const int sb = ((sg >> (8 * j)) & 255u) == 0xFFu ? -gb : gb;
                v |= (uint64_t) (uint8_t) (int8_t) sb << (8 * j);
            }
            g_ct2x[(size_t) g << 7 | (size_t) i] = v;
        }
}

// rows [r0, r1) (out[t][r - r0]) of NT tokens; TY 16 = IQ2_XXS, 22 = IQ2_S, 18 = IQ3_XXS, 21 = IQ3_S.  A pair of
// 32-value chunks a step: grid magnitudes g1 / g2, sign vectors s1 / s2 (-1 / +1), scales c1 / c2 - decoded once - then
// every token: q8 times the signs, vpmaddubsw against the grid, vpmaddwd by the scales, into ggml's 8 int32 lanes
template <int TY, int NT>
IQ_AVX2 void iq_rows_avx2(const uint8_t* w, size_t rb, int nb, const block_q8_K* const* y, float* const* out, int r0,
                          int r1) {
    const __m256i sub = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const bool gat = TY == 18 ? g_gather18 : TY == 21 ? g_gather21 : false;
    const bool ct = TY == 16 && g_ct16;
    const __m256i k80 = _mm256_set1_epi8((char) 0x80);
    constexpr float K = TY == 18 ? 0.25f : TY == 21 ? 1.0f : 0.125f;
    constexpr size_t BB = TY == 16 ? sizeof(block_iq2_xxs) : TY == 22 ? sizeof(block_iq2_s) :
                          TY == 18 ? sizeof(block_iq3_xxs) : sizeof(block_iq3_s);
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * rb;
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* blk = row + (size_t) i * BB;
            __m256i sumi[NT];
            for (int t = 0; t < NT; ++t) sumi[t] = _mm256_setzero_si256();
            uint16_t dh;
            std::memcpy(&dh, blk, 2);
            for (int ib = 0; ib < QK_K / 32; ib += 2) {
                __m256i g1, g2, s1, s2, c1, c2;
                if (TY == 16) {   // IQ2_XXS: per chunk a word of 4 grid indices, a word of 4 sign indices + the scale
                    const uint8_t* q = blk + 2 + 8 * ib;
                    const uint32_t a0 = u32(q), a1 = u32(q + 4), b0 = u32(q + 8), b1 = u32(q + 12);
                    if (ct) {   // (g1 / g2 the signed grids; s1 / s2 their 128 x pair sums)
                        g1 = tab4(g_ct2x, (a0 & 255u) << 7 | (a1 & 127u), ((a0 >> 8) & 255u) << 7 | ((a1 >> 7) & 127u),
                                  ((a0 >> 16) & 255u) << 7 | ((a1 >> 14) & 127u), (a0 >> 24) << 7 | ((a1 >> 21) & 127u));
                        g2 = tab4(g_ct2x, (b0 & 255u) << 7 | (b1 & 127u), ((b0 >> 8) & 255u) << 7 | ((b1 >> 7) & 127u),
                                  ((b0 >> 16) & 255u) << 7 | ((b1 >> 14) & 127u), (b0 >> 24) << 7 | ((b1 >> 21) & 127u));
                        s1 = _mm256_maddubs_epi16(k80, g1);
                        s2 = _mm256_maddubs_epi16(k80, g2);
                    } else {
                        g1 = tab4(iq2xxs_grid, a0 & 255u, (a0 >> 8) & 255u, (a0 >> 16) & 255u, a0 >> 24);
                        g2 = tab4(iq2xxs_grid, b0 & 255u, (b0 >> 8) & 255u, (b0 >> 16) & 255u, b0 >> 24);
                        s1 = sgn7t(a1);
                        s2 = sgn7t(b1);
                    }
                    c1 = _mm256_set1_epi16((short) (2 * (a1 >> 28) + 1));
                    c2 = _mm256_set1_epi16((short) (2 * (b1 >> 28) + 1));
                } else if (TY == 22) {   // IQ2_S: 10-bit grid indices (qs | 2 bits of qh), explicit signs, 2 scales a chunk
                    const uint8_t* qs = blk + 2 + 4 * ib;
                    const uint8_t* qh = blk + 2 + QK_K / 4;
                    const uint8_t* sg = blk + 2 + QK_K / 8 + 4 * ib;
                    const uint8_t* scl = blk + 2 + QK_K / 4 + QK_K / 32;
                    const unsigned h0 = qh[ib], h1 = qh[ib + 1];
                    g1 = tab4(iq2s_grid, qs[0] | ((h0 << 8) & 0x300u), qs[1] | ((h0 << 6) & 0x300u), qs[2] | ((h0 << 4) & 0x300u),
                              qs[3] | ((h0 << 2) & 0x300u));
                    g2 = tab4(iq2s_grid, qs[4] | ((h1 << 8) & 0x300u), qs[5] | ((h1 << 6) & 0x300u), qs[6] | ((h1 << 4) & 0x300u),
                              qs[7] | ((h1 << 2) & 0x300u));
                    s1 = sgn32(u32(sg));
                    s2 = sgn32(u32(sg + 4));
                    c1 = sc16(2 * (scl[ib] & 0xf) + 1, 2 * (scl[ib] >> 4) + 1);
                    c2 = sc16(2 * (scl[ib + 1] & 0xf) + 1, 2 * (scl[ib + 1] >> 4) + 1);
                } else if (TY == 18) {   // IQ3_XXS: 8 grid indices a chunk, then the chunks' sign / scale words
                    const uint8_t* q3 = blk + 2 + 8 * ib;
                    const uint8_t* gas = blk + 2 + QK_K / 4 + 4 * ib;
                    if (gat) {
                        g1 = _mm256_i32gather_epi32((const int*) iq3xxs_grid, _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) q3)), 4);
                        g2 = _mm256_i32gather_epi32((const int*) iq3xxs_grid,
                                                    _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (q3 + 8))), 4);
                    } else {
                        unsigned i1[8], i2[8];
                        for (int k = 0; k < 8; ++k) {
                            i1[k] = q3[k];
                            i2[k] = q3[8 + k];
                        }
                        g1 = tab8(iq3xxs_grid, i1);
                        g2 = tab8(iq3xxs_grid, i2);
                    }
                    const uint32_t a0 = u32(gas), a1 = u32(gas + 4);
                    s1 = sgn7t(a0);
                    s2 = sgn7t(a1);
                    c1 = _mm256_set1_epi16((short) (2 * (a0 >> 28) + 1));
                    c2 = _mm256_set1_epi16((short) (2 * (a1 >> 28) + 1));
                } else {   // IQ3_S: 9-bit grid indices (qs | a bit of qh), explicit signs
                    const uint8_t* qs = blk + 2 + 8 * ib;
                    const uint8_t* qh = blk + 2 + QK_K / 4;
                    const uint8_t* sg = blk + 2 + QK_K / 4 + QK_K / 32 + 4 * ib;
                    const int sc = blk[2 + QK_K / 4 + QK_K / 32 + QK_K / 8 + ib / 2];
                    const unsigned h0 = qh[ib], h1 = qh[ib + 1];
                    if (gat) {
                        const __m256i hb1 = _mm256_slli_epi32(_mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int) h0), sub), _mm256_set1_epi32(1)), 8);
                        const __m256i hb2 = _mm256_slli_epi32(_mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int) h1), sub), _mm256_set1_epi32(1)), 8);
                        g1 = _mm256_i32gather_epi32((const int*) iq3s_grid,
                                                    _mm256_or_si256(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) qs)), hb1), 4);
                        g2 = _mm256_i32gather_epi32((const int*) iq3s_grid,
                                                    _mm256_or_si256(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (qs + 8))), hb2), 4);
                    } else {
                        unsigned i1[8], i2[8];
                        for (int k = 0; k < 8; ++k) {
                            i1[k] = qs[k] | (((h0 >> k) & 1u) << 8);
                            i2[k] = qs[8 + k] | (((h1 >> k) & 1u) << 8);
                        }
                        g1 = tab8(iq3s_grid, i1);
                        g2 = tab8(iq3s_grid, i2);
                    }
                    s1 = sgn32(u32(sg));
                    s2 = sgn32(u32(sg + 4));
                    c1 = _mm256_set1_epi16((short) (2 * (sc & 0xf) + 1));
                    c2 = _mm256_set1_epi16((short) (2 * (sc >> 4) + 1));
                }
                if (ct) {
                    for (int t = 0; t < NT; ++t) {
                        const int8_t* q8 = y[t][i].qs + 32 * ib;
                        const __m256i u1 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i*) q8), k80);   // (q8 + 128)
                        const __m256i u2 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i*) (q8 + 32)), k80);
                        const __m256i d1 = _mm256_sub_epi16(_mm256_maddubs_epi16(u1, g1), s1);
                        const __m256i d2 = _mm256_sub_epi16(_mm256_maddubs_epi16(u2, g2), s2);
                        sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(_mm256_madd_epi16(d1, c1), _mm256_madd_epi16(d2, c2)));
                    }
                    continue;
                }
                for (int t = 0; t < NT; ++t) {
                    const int8_t* q8 = y[t][i].qs + 32 * ib;
                    const __m256i d1 = _mm256_maddubs_epi16(g1, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*) q8), s1));
                    const __m256i d2 = _mm256_maddubs_epi16(g2, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*) (q8 + 32)), s2));
                    sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(_mm256_madd_epi16(d1, c1), _mm256_madd_epi16(d2, c2)));
                }
            }
            const float dx = h2f(dh);
            for (int t = 0; t < NT; ++t)
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(sumi[t]), accf[t]);
        }
        for (int t = 0; t < NT; ++t) out[t][r - r0] = K * hsum_ggml(accf[t]);
    }
}

// ---- the same rows in 512-bit lanes (AVX-512 BW + VBMI, BMI2: Ice Lake and later Intel, Zen 4 / Zen 5), the GLM
// lane's one-token IQ2_XXS scheme (src/kernels/cpu/iq_avx512_one.cpp) for 3-bit grids: every grid byte is one of 8
// levels, so an entry is four 3-bit codes (a u16) and the grid's codes fit in registers - IQ3_XXS's 256 in 8 zmm,
// IQ3_S's 512 in 16; a block's 64 indices are looked up 32 at a time with two-register word permutes and blends, each
// 64 values' 16 codes spread to their bytes by a word permute, a multishift, an and and a byte shuffle; IQ3_XXS's signs
// come from pdep and a parity fold, IQ3_S's are a mask register's bits.  Two chunks a register, folded to ggml's 8
// lanes, then ggml's float steps: the same numbers to the bit (own_check).
#if defined(__GNUC__) || defined(__clang__)
#define IQ_512 __attribute__((target("avx512f,avx512bw,avx512vbmi,bmi2,avx2,fma")))
#else
#define IQ_512
#endif
alignas(64) uint16_t g_code3x[256], g_code3s[512];
bool g_codes_ok = false;

IQ_512 inline uint32_t signs4(uint32_t w) {   // ksigns_iq2xs for the four 7-bit fields of w: i | parity(i) << 7
    const uint64_t s = _pdep_u64(w & 0x0FFFFFFFu, 0x7F7F7F7Full);
    uint64_t p = s ^ ((s >> 4) & 0x0F0F0F0Full);
    p ^= (p >> 2) & 0x03030303ull;
    p ^= (p >> 1) & 0x01010101ull;
    return (uint32_t) (s | ((p & 0x01010101ull) << 7));
}
// each grid entry as four 3-bit level codes; false if a byte is not one of the 8 levels (a grid this file does not know)
bool build_codes3() {
    static const int lx[8] = {4, 12, 20, 28, 36, 44, 52, 62}, ls[8] = {1, 3, 5, 7, 9, 11, 13, 15};
    auto code = [](uint32_t e, const int* lv) -> int {
        int c = 0;
        for (int k = 0; k < 4; ++k) {
            const int b = (int) ((e >> (8 * k)) & 255u);
            int l = -1;
            for (int j = 0; j < 8; ++j)
                if (lv[j] == b) l = j;
            if (l < 0) return -1;
            c |= l << (3 * k);
        }
        return c;
    };
    for (int i = 0; i < 256; ++i) {
        const int c = code(iq3xxs_grid[i], lx);
        if (c < 0) return false;
        g_code3x[i] = (uint16_t) c;
    }
    for (int i = 0; i < 512; ++i) {
        const int c = code(iq3s_grid[i], ls);
        if (c < 0) return false;
        g_code3s[i] = (uint16_t) c;
    }
    return true;
}
IQ_512 inline __m256i fold(__m512i a) {   // a block's 16 int32 lanes -> ggml's 8 (lane k + lane k + 8), exact
    return _mm256_add_epi32(_mm512_castsi512_si256(a), _mm512_extracti64x4_epi64(a, 1));
}
IQ_512 inline __m512i scales4(int s0, int s1, int s2, int s3) {   // int16 lanes 0-7 s0, 8-15 s1, 16-23 s2, 24-31 s3
    const __m512i idx = _mm512_set_epi16(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,   //
                                         0, 0, 0, 0, 0, 0, 0, 0);
    const uint64_t packed = (uint64_t) (uint16_t) s0 | ((uint64_t) (uint16_t) s1 << 16) |
                            ((uint64_t) (uint16_t) s2 << 32) | ((uint64_t) (uint16_t) s3 << 48);
    return _mm512_permutexvar_epi16(idx, _mm512_set1_epi64((long long) packed));
}
// the codes of 32 word indices: 64 entries a register pair, NP pairs, blended by the index's bits 6, 7 (and 8)
template <int NP>
IQ_512 inline __m512i lookup(const __m512i* T, __m512i idx) {
    __m512i r[NP];
    for (int j = 0; j < NP; ++j) r[j] = _mm512_permutex2var_epi16(T[2 * j], idx, T[2 * j + 1]);
    const __mmask32 m6 = _mm512_test_epi16_mask(idx, _mm512_set1_epi16(64));
    const __mmask32 m7 = _mm512_test_epi16_mask(idx, _mm512_set1_epi16(128));
    __m512i a = _mm512_mask_blend_epi16(m7, _mm512_mask_blend_epi16(m6, r[0], r[1]), _mm512_mask_blend_epi16(m6, r[2], r[3]));
    if (NP == 8) {
        const __mmask32 m8 = _mm512_test_epi16_mask(idx, _mm512_set1_epi16(256));
        const __m512i b = _mm512_mask_blend_epi16(m7, _mm512_mask_blend_epi16(m6, r[NP > 4 ? 4 : 0], r[NP > 5 ? 5 : 0]),
                                                  _mm512_mask_blend_epi16(m6, r[NP > 6 ? 6 : 0], r[NP > 7 ? 7 : 0]));
        a = _mm512_mask_blend_epi16(m8, a, b);
    }
    return a;
}

template <int TY, int NT>
IQ_512 void iq3_rows_512(const uint8_t* w, size_t rb, int nb, const block_q8_K* const* y, float* const* out, int r0,
                         int r1) {
    constexpr int NTAB = TY == 18 ? 8 : 16;
    constexpr size_t BB = TY == 18 ? sizeof(block_iq3_xxs) : sizeof(block_iq3_s);
    __m512i T[NTAB];
    for (int k = 0; k < NTAB; ++k)
        T[k] = _mm512_load_si512((const void*) ((TY == 18 ? g_code3x : g_code3s) + 32 * k));
    // code -> grid byte (IQ3_XXS 4 .. 62, IQ3_S 1 .. 15) in every 128-bit lane; byte b of an output qword <- 3 bits at
    // [0, 3, 6, 9] (its first entry's code, word 0) or [32, 35, 38, 41] (the second's, word 2)
    const __m512i lut = TY == 18 ? _mm512_set4_epi32(0, 0, 0x3E342C24, 0x1C140C04) : _mm512_set4_epi32(0, 0, 0x0F0D0B09, 0x07050301);
    const __m512i ctl = _mm512_set1_epi64(0x2926232009060300LL);
    const __m512i seven = _mm512_set1_epi8(7), zero = _mm512_setzero_si512();
    __m512i pidx[2];   // a 64-value group h of 32 codes: output qword m <- entries 16 h + 2 m (word 0), + 1 (word 2)
    for (int h = 0; h < 2; ++h) {
        alignas(64) uint16_t p[32];
        for (int m = 0; m < 8; ++m) {
            p[4 * m] = p[4 * m + 1] = (uint16_t) (16 * h + 2 * m);
            p[4 * m + 2] = p[4 * m + 3] = (uint16_t) (16 * h + 2 * m + 1);
        }
        pidx[h] = _mm512_load_si512((const void*) p);
    }
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * rb;
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* blk = row + (size_t) i * BB;
            __m512i codes[2];
            if (TY == 18) {
                const __m512i q = _mm512_loadu_si512((const void*) (blk + 2));
                codes[0] = lookup<4>(T, _mm512_cvtepu8_epi16(_mm512_castsi512_si256(q)));
                codes[1] = lookup<4>(T, _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(q, 1)));
            } else {
                const uint8_t* qh = blk + 2 + QK_K / 4;
                const __m512i hi = _mm512_set1_epi16(256);
                const __m512i ia = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*) (blk + 2)));
                const __m512i ib = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*) (blk + 34)));
                // (index k of chunk c: qs[8 c + k] | bit k of qh[c] << 8 - the 32 bits of qh[0..3] are chunks 0..3's)
                codes[0] = lookup<8>(T, _mm512_or_si512(ia, _mm512_maskz_mov_epi16((__mmask32) u32(qh), hi)));
                codes[1] = lookup<8>(T, _mm512_or_si512(ib, _mm512_maskz_mov_epi16((__mmask32) u32(qh + 4), hi)));
            }
            __m512i acc[NT];
            for (int t = 0; t < NT; ++t) acc[t] = _mm512_setzero_si512();
            for (int p = 0; p < 4; ++p) {
                const __m512i rep = _mm512_permutexvar_epi16(pidx[p & 1], codes[p >> 1]);
                const __m512i g = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_multishift_epi64_epi8(ctl, rep), seven));
                uint64_t sm;
                int sa, sb;
                if (TY == 18) {
                    const uint8_t* sw = blk + 2 + QK_K / 4 + 8 * p;
                    const uint32_t wa = u32(sw), wb = u32(sw + 4);
                    sm = (uint64_t) signs4(wa) | ((uint64_t) signs4(wb) << 32);
                    sa = 2 * (int) (wa >> 28) + 1;
                    sb = 2 * (int) (wb >> 28) + 1;
                } else {
                    std::memcpy(&sm, blk + 2 + QK_K / 4 + QK_K / 32 + 8 * p, 8);
                    const int sc = blk[2 + QK_K / 4 + QK_K / 32 + QK_K / 8 + p];
                    sa = 2 * (sc & 0xf) + 1;
                    sb = 2 * (sc >> 4) + 1;
                }
                const __mmask64 m = _cvtu64_mask64(sm);
                const __m512i S = scales4(sa, sa, sb, sb);
                for (int t = 0; t < NT; ++t) {
                    const __m512i yv = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * p));
                    const __m512i ys = _mm512_mask_sub_epi8(yv, m, zero, yv);
                    acc[t] = _mm512_add_epi32(acc[t], _mm512_madd_epi16(_mm512_maddubs_epi16(g, ys), S));
                }
            }
            uint16_t dh;
            std::memcpy(&dh, blk, 2);
            const float dx = h2f(dh);
            for (int t = 0; t < NT; ++t)
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(fold(acc[t])), accf[t]);
        }
        for (int t = 0; t < NT; ++t) out[t][r - r0] = (TY == 18 ? 0.25f : 1.0f) * hsum_ggml(accf[t]);
    }
}

template <int TY>
void iq_rows_n(const uint8_t* w, size_t rb, int n, const void* const* act, int nt, float* const* out, int r0, int r1) {
    static const bool w512 = [] {
        const char* v = std::getenv("DS4_IQ_AVX2");   // (=1: the AVX2 rows on an AVX-512 CPU, to compare)
        return (TY == 18 || TY == 21) && g_codes_ok && isa().w512 && !(v != nullptr && std::atoi(v) != 0);
    }();
    const int nb = n / QK_K;
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const block_q8_K* y[4];
        float* o[4];
        const int c = nt - t0 < 4 ? nt - t0 : 4;
        for (int t = 0; t < c; ++t) {
            y[t] = (const block_q8_K*) act[t0 + t];
            o[t] = out[t0 + t];
        }
        if constexpr (TY == 18 || TY == 21) {
            if (w512) {
                switch (c) {
                    case 1: iq3_rows_512<TY, 1>(w, rb, nb, y, o, r0, r1); break;
                    case 2: iq3_rows_512<TY, 2>(w, rb, nb, y, o, r0, r1); break;
                    case 3: iq3_rows_512<TY, 3>(w, rb, nb, y, o, r0, r1); break;
                    default: iq3_rows_512<TY, 4>(w, rb, nb, y, o, r0, r1); break;
                }
                continue;
            }
        }
        switch (c) {
            case 1: iq_rows_avx2<TY, 1>(w, rb, nb, y, o, r0, r1); break;
            case 2: iq_rows_avx2<TY, 2>(w, rb, nb, y, o, r0, r1); break;
            case 3: iq_rows_avx2<TY, 3>(w, rb, nb, y, o, r0, r1); break;
            default: iq_rows_avx2<TY, 4>(w, rb, nb, y, o, r0, r1); break;
        }
    }
}
#endif  // IQ_X86

// out[t][r - r0] for the rows of `type` (an own type only)
void own_rows(int type, const uint8_t* w, size_t rb, int n, const void* const* act, int nt, float* const* out, int r0,
              int r1) {
#if defined(IQ_X86)
    if (type == GGML_TYPE_IQ2_XXS) return iq_rows_n<16>(w, rb, n, act, nt, out, r0, r1);
    if (type == GGML_TYPE_IQ2_S) return iq_rows_n<22>(w, rb, n, act, nt, out, r0, r1);
    if (type == GGML_TYPE_IQ3_XXS) return iq_rows_n<18>(w, rb, n, act, nt, out, r0, r1);
    if (type == GGML_TYPE_IQ3_S) return iq_rows_n<21>(w, rb, n, act, nt, out, r0, r1);
#endif
    (void) type, (void) w, (void) rb, (void) n, (void) act, (void) nt, (void) out, (void) r0, (void) r1;
}

// our rows against ggml-cpu's dot, bit for bit: random blocks of the type (its F16 scale a small normal value), Q8_K
// activations from ggml's quantizer, 1, 2, 3, 5 and 9 tokens, 13 rows of 8 blocks
bool own_check(int type) {
    ggml_cpu_init();
    const ggml_type t = (ggml_type) type;
    const int n = 2048, rows = 13;
    const size_t rb = ggml_row_size(t, n), bb = ggml_type_size(t), ab = ggml_row_size(GGML_TYPE_Q8_K, n);
    uint64_t st = 0x9E3779B97F4A7C15ull;
    auto next = [&st] {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        return st;
    };
    std::vector<uint8_t> w(rb * rows);
    for (auto& b : w) b = (uint8_t) next();
    for (size_t o = 0; o < w.size(); o += bb) {
        const ggml_fp16_t d = ggml_fp32_to_fp16(0.002f + 0.001f * (float) (next() % 16));
        std::memcpy(&w[o], &d, sizeof d);
    }
    const auto* q8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    const auto dot = ggml_get_type_traits_cpu(t)->vec_dot;
    for (int nt : {1, 2, 3, 5, 9}) {
        std::vector<uint8_t> a(ab * (size_t) nt);
        std::vector<float> x((size_t) n), got((size_t) nt * rows), ref((size_t) nt * rows);
        std::vector<const void*> ap((size_t) nt);
        std::vector<float*> op((size_t) nt);
        for (int k = 0; k < nt; ++k) {
            for (auto& v : x) v = ((float) (next() % 2001) - 1000.0f) * 1e-3f;
            q8->from_float(x.data(), a.data() + ab * (size_t) k, n);
            ap[(size_t) k] = a.data() + ab * (size_t) k;
            op[(size_t) k] = got.data() + (size_t) k * rows;
        }
        own_rows(type, w.data(), rb, n, ap.data(), nt, op.data(), 0, rows);
        for (int k = 0; k < nt; ++k)
            for (int r = 0; r < rows; ++r) dot(n, &ref[(size_t) k * rows + r], 0, w.data() + (size_t) r * rb, 0, ap[(size_t) k], 0, 1);
        if (std::memcmp(got.data(), ref.data(), got.size() * sizeof(float)) != 0) return false;
    }
    return true;
}

// the time (us, best of 5 rounds) of 32 rows of 16 blocks of the type against one token: ours with the IQ3 grids as
// broadcasts + blends (variant 1) or by vpgatherdd (2), or the native lane's rows (0: the GLM lane's / ggml-cpu's)
double time_variant(int type, int variant) {
    const ggml_type t = (ggml_type) type;
    const int n = 4096, rows = 32;
    const size_t rb = ggml_row_size(t, n), bb = ggml_type_size(t), ab = ggml_row_size(GGML_TYPE_Q8_K, n);
    std::vector<uint8_t> w(rb * rows), a(ab);
    uint32_t s = 12345u;
    for (auto& b : w) b = (uint8_t) ((s = s * 1664525u + 1013904223u) >> 24);
    for (size_t o = 0; o < w.size(); o += bb) {
        const ggml_fp16_t d = ggml_fp32_to_fp16(0.003f);
        std::memcpy(&w[o], &d, sizeof d);
    }
    std::vector<float> x((size_t) n), y((size_t) rows);
    for (auto& v : x) v = (float) ((int) ((s = s * 1664525u + 1013904223u) >> 16) - 32768) / 32768.0f;
    ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(x.data(), a.data(), n);
    const void* ap[1] = {a.data()};
    float* yp[1] = {y.data()};
    kc::NativeFmt nf;   // (the native rows timed as a gate / up pair of the same rows: 2 x rows dots, as ours below)
    std::string err;
    if (variant == 0 && !kc::native_fmt(type, type, n, QK_K, nf, err)) return 1e30;
    bool* gflag = type == GGML_TYPE_IQ3_XXS ? &g_gather18 : type == GGML_TYPE_IQ3_S ? &g_gather21 :
                  type == GGML_TYPE_IQ2_XXS ? &g_ct16 : nullptr;
    const bool keep = gflag != nullptr && *gflag;
    if (gflag != nullptr) *gflag = variant >= 2;
    double best = 1e30;
    for (int round = 0; round < 6; ++round) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < 4; ++rep) {
            if (variant == 0) {
                kc::native_gu_rows_split(nf, w.data(), w.data(), ap, 1, yp, 0, rows, std::numeric_limits<float>::infinity());
            } else {
                own_rows(type, w.data(), rb, n, ap, 1, yp, 0, rows);
                own_rows(type, w.data(), rb, n, ap, 1, yp, 0, rows);
            }
        }
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        if (round > 0) best = std::min(best, us);
    }
    if (gflag != nullptr) *gflag = keep;
    return best;
}

// our rows for this type, here: IQ3_XXS / IQ3_S on AVX2 or AVX-512 VBMI, IQ2_XXS / IQ2_S on AVX2 - checked once
// against ggml-cpu's dot (each variant), then the fastest of ours and the native lane's measured on this CPU (they
// give the same bits, so the choice changes no answer).  DS4_IQ_OWN=0: the native lane's rows; =1: ours without
// measuring (DS4_IQ_GATHER=1: the IQ3 grids by vpgatherdd); DS4_IQ_AVX2=1: our AVX2 rows on an AVX-512 CPU too.
bool own(int type) {
#if defined(IQ_X86)
    int k;
    switch (type) {
        case GGML_TYPE_IQ3_XXS: k = 0; break;
        case GGML_TYPE_IQ3_S: k = 1; break;
        case GGML_TYPE_IQ2_XXS: k = 2; break;
        case GGML_TYPE_IQ2_S: k = 3; break;
        default: return false;
    }
    static int mode = -1;   // DS4_IQ_OWN: -1 measured, 0 never, 1 always
    static bool force_avx2 = false, force_gather = false;
    static const bool allowed = [] {
        const char* v = std::getenv("DS4_IQ_OWN");
        mode = v == nullptr ? -1 : std::atoi(v) != 0 ? 1 : 0;
        const char* a2 = std::getenv("DS4_IQ_AVX2");
        force_avx2 = a2 != nullptr && std::atoi(a2) != 0;
        const char* gt = std::getenv("DS4_IQ_GATHER");
        force_gather = gt != nullptr && std::atoi(gt) != 0;
        g_codes_ok = build_codes3();   // (the 512-bit rows' tables; without them those rows are not used)
        build_ks64();
        build_ct2x();
        return mode != 0 && isa().avx2;
    }();
    // (IQ2 on an AVX-512 CPU: the GLM lane's AVX-512 rows serve them, unless DS4_IQ_AVX2)
    if (!allowed || (k >= 2 && isa().w512 && !force_avx2)) return false;
    static std::once_flag once[4];
    static bool ok[4];
    std::call_once(once[k], [&] {
        bool* gflag = k == 0 ? &g_gather18 : k == 1 ? &g_gather21 : k == 2 ? &g_ct16 : nullptr;   // (the second form)
        bool good = own_check(type);   // (the blend form; the 512-bit rows where they serve)
        bool good_g = false;
        if (gflag != nullptr) {
            *gflag = true;
            good_g = own_check(type);
            *gflag = false;
        }
        if (mode == 1) {
            if (gflag != nullptr) *gflag = force_gather && good_g;
            ok[k] = good || (force_gather && good_g);
            return;
        }
        // the variants timed in turns, the best of each kept (other work on the machine - the load's read-ahead, the
        // prompt - then hits them alike), and ours taken only when clearly faster (10 %): the choice changes no answer,
        // a wrong one costs time (Uranus' Golden Cove: one-shot timings 30-80 % apart run to run, the choice flipping)
        double tn = 1e30, tb = 1e30, tg = 1e30;
        for (int it = 0; it < 5; ++it) {
            tn = std::min(tn, time_variant(type, 0));
            if (good) tb = std::min(tb, time_variant(type, 1));
            if (good_g) tg = std::min(tg, time_variant(type, 2));
        }
        if (gflag != nullptr) *gflag = tg < tb;
        ok[k] = std::min(tb, tg) < 0.9 * tn;
        if (std::getenv("DS4_IQ_VERBOSE") != nullptr)
            std::fprintf(stderr, "ds4: CPU lane %s rows: native %.0f us, ours %.0f us%s -> %s\n", ggml_type_name((ggml_type) type),
                         tn, tb, gflag != nullptr ? (std::string(" (second form: gather / one table ") + std::to_string((int) tg) + " us)").c_str() : "",
                         ok[k] ? (gflag != nullptr && *gflag ? "ours (second form)" : "ours") : "native");
    });
    return ok[k];
#else
    (void) type;
    return false;
#endif
}
}  // namespace

bool iq_cpu_fmt(int tg, int td, int n_embd, int n_ff, IqCpu& f, std::string& err) {
    kc::NativeFmt n;
    if (!kc::native_fmt(tg, td, n_embd, n_ff, n, err)) return false;
    f.tg = tg;
    f.td = td;
    f.n_embd = n_embd;
    f.n_ff = n_ff;
    f.gu_act = n.gu_act;
    f.d_act = n.d_act;
    f.gu_row = n.gu_row;
    f.d_row = n.d_row;
    f.act_bytes = n.act_bytes;
    f.h_bytes = n.h_bytes;
    return true;
}

void iq_cpu_quant(int act, const float* x, int k, void* dst) {
    kc::NativeFmt n;
    n.gu_act = act;
    n.n_embd = k;
    kc::native_quant_act(n, x, dst);
}

void iq_cpu_gate_up(const IqCpu& f, const uint8_t* gate, const uint8_t* up, const void* const* act, int nt,
                    float* const* ff, int r0, int r1, float limit) {
    if (r1 > r0 && own(f.tg)) {   // (native_gu_rows_split's epilogue, on the same values)
        const int nr = r1 - r0;
        thread_local std::vector<float> buf;
        thread_local std::vector<float*> go, uo;
        buf.resize((size_t) 2 * nt * nr);
        go.resize((size_t) nt);
        uo.resize((size_t) nt);
        for (int t = 0; t < nt; ++t) {
            go[(size_t) t] = buf.data() + (size_t) t * nr;
            uo[(size_t) t] = buf.data() + (size_t) (nt + t) * nr;
        }
        own_rows(f.tg, gate, f.gu_row, f.n_embd, act, nt, go.data(), r0, r1);
        own_rows(f.tg, up, f.gu_row, f.n_embd, act, nt, uo.data(), r0, r1);
        for (int t = 0; t < nt; ++t)
            for (int r = 0; r < nr; ++r) {
                float g = go[(size_t) t][r], u = uo[(size_t) t][r];
                if (limit > 0.0f) {
                    g = std::fmin(g, limit);
                    u = std::fmin(std::fmax(u, -limit), limit);
                }
                ff[t][r0 + r] = (g / (1.f + std::exp(-g))) * u;
            }
        return;
    }
    kc::native_gu_rows_split(native(f), gate, up, act, nt, ff, r0, r1,
                             limit > 0.0f ? limit : std::numeric_limits<float>::infinity());
}

void iq_cpu_down(const IqCpu& f, const uint8_t* down, const void* const* hq, int nt, float* const* out, int r0, int r1) {
    if (r1 > r0 && own(f.td)) {
        thread_local std::vector<float*> o;
        o.resize((size_t) nt);
        for (int t = 0; t < nt; ++t) o[(size_t) t] = out[t] + r0;
        own_rows(f.td, down, f.d_row, f.n_ff, hq, nt, o.data(), r0, r1);
        return;
    }
    kc::native_down_rows_split(native(f), down, hq, nt, out, r0, r1);
}

bool iq_cpu_own_rows(int type) { return own(type); }
#else
// (no ggml-cpu lane for the low-bit types here; the trellis types have their own: ds4_tq_cpu.cpp)
bool iq_cpu_fmt(int tg, int td, int n_embd, int n_ff, IqCpu& f, std::string& err) {
    if (tq_cpu_type(tg) && tq_cpu_type(td)) return tq_cpu_fmt(tg, td, n_embd, n_ff, f, err);
    if (gg_cpu_type(tg) && gg_cpu_type(td)) return gg_cpu_fmt(tg, td, n_embd, n_ff, f, err);
    err = "this build has no ggml-cpu (STRATA_NATIVE_EXPERTS=OFF): the low-bit experts CPU lane is not in it";
    return false;
}
void iq_cpu_quant(int act, const float* x, int k, void* dst) {
    if (act == tq_cpu_act()) tq_cpu_quant(x, k, dst);
    else gg_cpu_quant(act, x, k, dst);
}
void iq_cpu_gate_up(const IqCpu& f, const uint8_t* gate, const uint8_t* up, const void* const* act, int nt, float* const* ff,
                    int r0, int r1, float limit) {
    if (tq_cpu_type(f.tg)) tq_cpu_gate_up(f, gate, up, act, nt, ff, r0, r1, limit);
    else gg_cpu_gate_up(f, gate, up, act, nt, ff, r0, r1, limit);
}
void iq_cpu_down(const IqCpu& f, const uint8_t* down, const void* const* hq, int nt, float* const* out, int r0, int r1) {
    if (tq_cpu_type(f.td)) tq_cpu_down(f, down, hq, nt, out, r0, r1);
    else gg_cpu_down(f, down, hq, nt, out, r0, r1);
}
bool iq_cpu_own_rows(int) { return false; }
#endif

}  // namespace strata::ds4::gguf
