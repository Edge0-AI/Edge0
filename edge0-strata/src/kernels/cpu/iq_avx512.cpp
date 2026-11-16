// src/kernels/cpu/iq_avx512.cpp - plan v0.3 P6: the i-quant expert rows in 512-bit lanes, several tokens at once.
//
// ggml-cpu's x86 dot products for these formats are AVX2 and single-token: every token re-decodes the weights
// and a sign is applied with five instructions per 32 values.  Here 64 values are decoded once (8 or 16 grid
// lookups, one 64-bit sign mask, one vector of scales) and every token applies them with a masked subtract, a
// `maddubs` and a `madd`.  The arithmetic is ggml's (ggml-cpu/quants.c, the `_generic` references): integer sums
// per block, times d_x * d_y * the format's constant - only the order of the float additions differs.
//
// Formats: IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18), IQ3_S (21), IQ2_S (22).  IQ1_M stays on ggml-cpu (no shipped
// model has IQ1_M expert rows: the 'Coder IQ1_M' pack's gate/up are IQ2_S / IQ3_XXS / IQ3_S).
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace strata::kernels::cpu {
namespace {

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }
inline uint32_t u32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint16_t u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline uint64_t u64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

// E-2: the IQ3 grids by one AVX-512 gather (the same words, the same results) - measured 3-5% SLOWER than the scalar
// lookups on a Ryzen 5 7600 (Zen 4 gathers are microcoded), so opt-in: STRATA_IQ_GATHER=1
// 0.2.5: DEFAULT on where gathers are fast (Intel server and HEDT cores with AVX-512: cpu_gather_fast), off on Zen 4 (microcoded);
// STRATA_IQ_GATHER=1 / 0 force it either way.  The same words are loaded, so the results are bit for bit the scalar decode's.
const bool gather = [] {
    const char* v = std::getenv("STRATA_IQ_GATHER");
    return v != nullptr && v[0] != '\0' ? std::atoi(v) != 0 : cpu_gather_fast();
}();
// the grid lookups by one 64-bit gather of eight entries (IQ2_XXS / IQ2_XS / IQ2_S): `idx` = eight 32-bit indices
inline __m512i gather64(__m256i idx, const void* grid) { return _mm512_i32gather_epi64(idx, grid, 8); }
// E-2: software prefetch distance in bytes (STRATA_IQ_PREFETCH; 0 = off): -2-3% gate/up time on IQ3_S decode
// (the rows are read at ~30 GB/s by 6 cores, near what DDR5 with 4 KB pages gives: the pool is memory-bound)
const int prefetch_ahead = [] {
    const char* v = std::getenv("STRATA_IQ_PREFETCH");
    return v ? std::atoi(v) : 2048;
}();

// lanes 0-7 -> s0, 8-15 -> s1, 16-23 -> s2, 24-31 -> s3 (int16 lanes of a maddubs result: two values each)
inline __m512i scales4(int s0, int s1, int s2, int s3) {
    const __m512i idx = _mm512_set_epi16(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
                                         0, 0, 0, 0, 0, 0, 0, 0);
    const uint64_t packed = (uint64_t) (uint16_t) s0 | ((uint64_t) (uint16_t) s1 << 16) |
                            ((uint64_t) (uint16_t) s2 << 32) | ((uint64_t) (uint16_t) s3 << 48);
    return _mm512_permutexvar_epi16(idx, _mm512_set1_epi64((long long) packed));
}

// ---- per format: 64 values (chunk j of a 256-value block) -> grid magnitudes, sign mask, scales
template <int TY> struct Fmt;

template <> struct Fmt<16> {   // IQ2_XXS: d, qs[32] u16
    static constexpr int bytes = 66;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        const uint32_t a0 = u32(q), a1 = u32(q + 4), b0 = u32(q + 8), b1 = u32(q + 12);
        g = gather ? gather64(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long) ((uint64_t) a0 | ((uint64_t) b0 << 32)))), iq2xxs_grid)
                   : _mm512_set_epi64((long long) iq2xxs_grid[b0 >> 24], (long long) iq2xxs_grid[(b0 >> 16) & 255],
                             (long long) iq2xxs_grid[(b0 >> 8) & 255], (long long) iq2xxs_grid[b0 & 255],
                             (long long) iq2xxs_grid[a0 >> 24], (long long) iq2xxs_grid[(a0 >> 16) & 255],
                             (long long) iq2xxs_grid[(a0 >> 8) & 255], (long long) iq2xxs_grid[a0 & 255]);
        const uint64_t s = (uint64_t) ksigns_iq2xs[a1 & 127] | ((uint64_t) ksigns_iq2xs[(a1 >> 7) & 127] << 8) |
                           ((uint64_t) ksigns_iq2xs[(a1 >> 14) & 127] << 16) | ((uint64_t) ksigns_iq2xs[(a1 >> 21) & 127] << 24) |
                           ((uint64_t) ksigns_iq2xs[b1 & 127] << 32) | ((uint64_t) ksigns_iq2xs[(b1 >> 7) & 127] << 40) |
                           ((uint64_t) ksigns_iq2xs[(b1 >> 14) & 127] << 48) | ((uint64_t) ksigns_iq2xs[(b1 >> 21) & 127] << 56);
        m = _cvtu64_mask64(s);
        const int sa = 2 * (int) (a1 >> 28) + 1, sb = 2 * (int) (b1 >> 28) + 1;
        sc = scales4(sa, sa, sb, sb);
    }
};

template <> struct Fmt<17> {   // IQ2_XS: d, qs[32] u16 (9-bit grid index + 7-bit sign index), scales[8]
    static constexpr int bytes = 74;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        uint16_t v[8];
        std::memcpy(v, q, 16);
        g = gather ? gather64(_mm256_and_si256(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*) q)), _mm256_set1_epi32(511)), iq2xs_grid)
                   : _mm512_set_epi64((long long) iq2xs_grid[v[7] & 511], (long long) iq2xs_grid[v[6] & 511],
                             (long long) iq2xs_grid[v[5] & 511], (long long) iq2xs_grid[v[4] & 511],
                             (long long) iq2xs_grid[v[3] & 511], (long long) iq2xs_grid[v[2] & 511],
                             (long long) iq2xs_grid[v[1] & 511], (long long) iq2xs_grid[v[0] & 511]);
        uint64_t s = 0;
        for (int l = 0; l < 8; ++l) s |= (uint64_t) ksigns_iq2xs[v[l] >> 9] << (8 * l);
        m = _cvtu64_mask64(s);
        const uint8_t s0 = b[66 + 2 * j], s1 = b[66 + 2 * j + 1];
        sc = scales4(2 * (s0 & 15) + 1, 2 * (s0 >> 4) + 1, 2 * (s1 & 15) + 1, 2 * (s1 >> 4) + 1);
    }
};

template <> struct Fmt<22> {   // IQ2_S: d, qs[64] (32 grid bytes, 32 sign bytes), qh[8], scales[8]
    static constexpr int bytes = 82;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* qs = b + 2 + 8 * j;
        const uint8_t h0 = b[66 + 2 * j], h1 = b[66 + 2 * j + 1];
        if (gather) {   // the 10-bit indices: the byte plus two high bits from qh (h0 for lanes 0-3, h1 for 4-7), one 64-bit gather
            const __m256i hi = _mm256_and_si256(_mm256_sllv_epi32(_mm256_setr_epi32(h0, h0, h0, h0, h1, h1, h1, h1),
                                                                   _mm256_setr_epi32(8, 6, 4, 2, 8, 6, 4, 2)), _mm256_set1_epi32(0x300));
            g = gather64(_mm256_or_si256(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) qs)), hi), iq2s_grid);
        } else
        g = _mm512_set_epi64((long long) iq2s_grid[qs[7] | ((h1 << 2) & 0x300)], (long long) iq2s_grid[qs[6] | ((h1 << 4) & 0x300)],
                             (long long) iq2s_grid[qs[5] | ((h1 << 6) & 0x300)], (long long) iq2s_grid[qs[4] | ((h1 << 8) & 0x300)],
                             (long long) iq2s_grid[qs[3] | ((h0 << 2) & 0x300)], (long long) iq2s_grid[qs[2] | ((h0 << 4) & 0x300)],
                             (long long) iq2s_grid[qs[1] | ((h0 << 6) & 0x300)], (long long) iq2s_grid[qs[0] | ((h0 << 8) & 0x300)]);
        m = _cvtu64_mask64(u64(b + 2 + 32 + 8 * j));
        const uint8_t s0 = b[74 + 2 * j], s1 = b[74 + 2 * j + 1];
        sc = scales4(2 * (s0 & 15) + 1, 2 * (s0 >> 4) + 1, 2 * (s1 & 15) + 1, 2 * (s1 >> 4) + 1);
    }
};

template <> struct Fmt<18> {   // IQ3_XXS: d, qs[64] grid bytes, 8 x u32 (4 x 7-bit sign index + 4-bit scale)
    static constexpr int bytes = 98;
    static constexpr float K = 0.25f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        // E-2: one gather of the 16 grid words (the scalar lookups assembled with set_epi32 cost ~30 uops)
        g = gather ? _mm512_i32gather_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) q)), iq3xxs_grid, 4)
                   : _mm512_set_epi32((int) iq3xxs_grid[q[15]], (int) iq3xxs_grid[q[14]], (int) iq3xxs_grid[q[13]], (int) iq3xxs_grid[q[12]],
                             (int) iq3xxs_grid[q[11]], (int) iq3xxs_grid[q[10]], (int) iq3xxs_grid[q[9]], (int) iq3xxs_grid[q[8]],
                             (int) iq3xxs_grid[q[7]], (int) iq3xxs_grid[q[6]], (int) iq3xxs_grid[q[5]], (int) iq3xxs_grid[q[4]],
                             (int) iq3xxs_grid[q[3]], (int) iq3xxs_grid[q[2]], (int) iq3xxs_grid[q[1]], (int) iq3xxs_grid[q[0]]);
        const uint32_t a = u32(b + 2 + 64 + 8 * j), c = u32(b + 2 + 64 + 8 * j + 4);
        const uint64_t s = (uint64_t) ksigns_iq2xs[a & 127] | ((uint64_t) ksigns_iq2xs[(a >> 7) & 127] << 8) |
                           ((uint64_t) ksigns_iq2xs[(a >> 14) & 127] << 16) | ((uint64_t) ksigns_iq2xs[(a >> 21) & 127] << 24) |
                           ((uint64_t) ksigns_iq2xs[c & 127] << 32) | ((uint64_t) ksigns_iq2xs[(c >> 7) & 127] << 40) |
                           ((uint64_t) ksigns_iq2xs[(c >> 14) & 127] << 48) | ((uint64_t) ksigns_iq2xs[(c >> 21) & 127] << 56);
        m = _cvtu64_mask64(s);
        const int sa = 2 * (int) (a >> 28) + 1, sb = 2 * (int) (c >> 28) + 1;
        sc = scales4(sa, sa, sb, sb);
    }
};

template <> struct Fmt<21> {   // IQ3_S: d, qs[64], qh[8], signs[32], scales[4]
    static constexpr int bytes = 110;
    static constexpr float K = 1.0f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        const uint32_t h0 = b[66 + 2 * j], h1 = b[66 + 2 * j + 1];
        if (gather) {   // E-2: the 9-bit indices (the high bit from qh under a mask), one gather
            __m512i idx = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) q));
            idx = _mm512_mask_add_epi32(idx, (__mmask16) (h0 | (h1 << 8)), idx, _mm512_set1_epi32(256));
            g = _mm512_i32gather_epi32(idx, iq3s_grid, 4);
        } else {
#define G3(k, h, kk) (int) iq3s_grid[q[k] | (((h >> kk) & 1) << 8)]
        g = _mm512_set_epi32(G3(15, h1, 7), G3(14, h1, 6), G3(13, h1, 5), G3(12, h1, 4),
                             G3(11, h1, 3), G3(10, h1, 2), G3(9, h1, 1), G3(8, h1, 0),
                             G3(7, h0, 7), G3(6, h0, 6), G3(5, h0, 5), G3(4, h0, 4),
                             G3(3, h0, 3), G3(2, h0, 2), G3(1, h0, 1), G3(0, h0, 0));
#undef G3
        }
        m = _cvtu64_mask64(u64(b + 74 + 8 * j));
        const uint8_t s = b[106 + j];
        const int sa = 2 * (s & 15) + 1, sb = 2 * (s >> 4) + 1;
        sc = scales4(sa, sa, sb, sb);
    }
};

template <bool Singleton> struct RowAccumulator;
template <> struct RowAccumulator<false> { using Type = __m512; };
template <> struct RowAccumulator<true> { using Type = __m256; };

template <int TY, int NT, bool Singleton = false>
inline void row_dot(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    // Decode weights once for the group; select the accumulation order at compile time.
    using Acc = typename RowAccumulator<Singleton>::Type;
    Acc accf[NT];
    for (int t = 0; t < NT; ++t) {
        if constexpr (Singleton) accf[t] = _mm256_setzero_ps();
        else accf[t] = _mm512_setzero_ps();
    }
    const __m512i zero = _mm512_setzero_si512();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * Fmt<TY>::bytes;
        // E-2: the weight rows stream from DRAM (4 KB pages when large pages are refused): ask for the bytes a few
        // blocks ahead (the next row's start included) before the decode needs them
        if (prefetch_ahead > 0) {
            const uint8_t* ahead = blk + (size_t) prefetch_ahead;
            _mm_prefetch((const char*) ahead, _MM_HINT_T0);
            _mm_prefetch((const char*) ahead + 64, _MM_HINT_T0);
        }
        __m512i acci[NT];
        for (int t = 0; t < NT; ++t) acci[t] = _mm512_setzero_si512();
        for (int j = 0; j < 4; ++j) {
            __m512i g, sc;
            __mmask64 m;
            Fmt<TY>::decode(blk, j, g, m, sc);
            for (int t = 0; t < NT; ++t) {
                const __m512i yv = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * j));
                const __m512i ys = _mm512_mask_sub_epi8(yv, m, zero, yv);
                acci[t] = _mm512_add_epi32(acci[t], _mm512_madd_epi16(_mm512_maddubs_epi16(g, ys), sc));
            }
        }
        if constexpr (Singleton) {
            const float dx = h2f(u16(blk));
            for (int t = 0; t < NT; ++t) {
                // Preserve ggml AVX2's eight integer lanes BEFORE floating conversion.
                const __m256i folded = _mm256_add_epi32(_mm512_castsi512_si256(acci[t]),
                                                       _mm512_extracti64x4_epi64(acci[t], 1));
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d),
                                         _mm256_cvtepi32_ps(folded), accf[t]);
            }
        } else {
            const float dx = h2f(u16(blk)) * Fmt<TY>::K;
            for (int t = 0; t < NT; ++t)
                accf[t] = _mm512_fmadd_ps(_mm512_set1_ps(dx * y[t][i].d), _mm512_cvtepi32_ps(acci[t]), accf[t]);
        }
    }
    if constexpr (Singleton) {
        for (int t = 0; t < NT; ++t) {
            __m128 sum = _mm256_extractf128_ps(accf[t], 1);
            sum = _mm_add_ps(sum, _mm256_castps256_ps128(accf[t]));
            sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
            sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
            res[t] = Fmt<TY>::K * _mm_cvtss_f32(sum);
        }
    } else {
        for (int t = 0; t < NT; ++t) res[t] = _mm512_reduce_add_ps(accf[t]);
    }
}

template <int TY, int NT, bool Singleton = false>
void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, float* const* ff,
             int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    const int nb = n / QK_K;
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<TY, NT, Singleton>(blob + (size_t) r * gu_row, nb, y, g);
        row_dot<TY, NT, Singleton>(blob + up_off + (size_t) r * gu_row, nb, y, u);
        for (int t = 0; t < NT; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
    }
}

template <int TY, int NT>
void dot_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<TY, NT>(w + (size_t) r * row_bytes, n / QK_K, y, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

template <int TY, bool Singleton = false>
void gu_rows_nt(int nt, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                float* const* ff, int r0, int r1) {
    switch (nt) {
        case 1: gu_rows<TY, 1, Singleton>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 2: gu_rows<TY, 2, Singleton>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 3: gu_rows<TY, 3, Singleton>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 4: gu_rows<TY, 4, Singleton>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 5: gu_rows<TY, 5, Singleton>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 6: gu_rows<TY, 6, Singleton>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 7: gu_rows<TY, 7, Singleton>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: gu_rows<TY, 8, Singleton>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    }
}

template <int TY>
void dot_rows_nt(int nt, const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0,
                 int r1) {
    switch (nt) {
        case 1: dot_rows<TY, 1>(w, row_bytes, n, act, out, r0, r1); break;
        case 2: dot_rows<TY, 2>(w, row_bytes, n, act, out, r0, r1); break;
        case 3: dot_rows<TY, 3>(w, row_bytes, n, act, out, r0, r1); break;
        case 4: dot_rows<TY, 4>(w, row_bytes, n, act, out, r0, r1); break;
        default: for (int t0 = 0; t0 < nt; t0 += 4) {
            const int k = nt - t0 < 4 ? nt - t0 : 4;
            dot_rows_nt<TY>(k, w, row_bytes, n, act + t0, out + t0, r0, r1);
        }
    }
}


// The sign masks of IQ2_XXS / IQ2_XS / IQ3_XXS come from a 7-bit index whose 8th bit is the parity of the other seven (ksigns_iq2xs[i] =
// i | parity(i) << 7).  The table needs a load and a few shifts and ors a group of eight values; popcount does it for all eight groups
// of 64 values in one vector: measured on Zen 5, one-token IQ2_XXS rows 33 -> 45-50 GB/s (16 threads, DRAM-sized), cache-resident 16.8 -> 13.4 ns a block.
// (AVX512_VPOPCNTDQ + VL: Zen 4, Zen 5, Ice Lake and later; iq512x_supported() asks for it.)
inline __mmask64 sign_mask_from_idx(__m256i si) {
    const __m256i par = _mm256_slli_epi32(_mm256_and_si256(_mm256_popcnt_epi32(si), _mm256_set1_epi32(1)), 7);
    return _cvtu64_mask64((uint64_t) _mm_cvtsi128_si64(_mm256_cvtepi32_epi8(_mm256_or_si256(si, par))));
}
inline __m256i sign_idx_xxs(uint32_t lo4, uint32_t hi4) {   // four 7-bit fields of each of two words -> eight dwords
    const __m256i w = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_set1_epi32((int) lo4)), _mm_set1_epi32((int) hi4), 1);
    return _mm256_and_si256(_mm256_srlv_epi32(w, _mm256_setr_epi32(0, 7, 14, 21, 0, 7, 14, 21)), _mm256_set1_epi32(127));
}

template <int TY> struct FmtX : Fmt<TY> {};
template <> struct FmtX<16> : Fmt<16> {
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        const uint32_t a0 = u32(q), a1 = u32(q + 4), b0 = u32(q + 8), b1 = u32(q + 12);
        g = gather ? gather64(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long) ((uint64_t) a0 | ((uint64_t) b0 << 32)))), iq2xxs_grid)
                   : _mm512_set_epi64((long long) iq2xxs_grid[b0 >> 24], (long long) iq2xxs_grid[(b0 >> 16) & 255],
                             (long long) iq2xxs_grid[(b0 >> 8) & 255], (long long) iq2xxs_grid[b0 & 255],
                             (long long) iq2xxs_grid[a0 >> 24], (long long) iq2xxs_grid[(a0 >> 16) & 255],
                             (long long) iq2xxs_grid[(a0 >> 8) & 255], (long long) iq2xxs_grid[a0 & 255]);
        m = sign_mask_from_idx(sign_idx_xxs(a1, b1));
        const int sa = 2 * (int) (a1 >> 28) + 1, sb = 2 * (int) (b1 >> 28) + 1;
        sc = scales4(sa, sa, sb, sb);
    }
};
template <> struct FmtX<17> : Fmt<17> {
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        uint16_t v[8];
        std::memcpy(v, q, 16);
        g = gather ? gather64(_mm256_and_si256(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*) q)), _mm256_set1_epi32(511)), iq2xs_grid)
                   : _mm512_set_epi64((long long) iq2xs_grid[v[7] & 511], (long long) iq2xs_grid[v[6] & 511],
                             (long long) iq2xs_grid[v[5] & 511], (long long) iq2xs_grid[v[4] & 511],
                             (long long) iq2xs_grid[v[3] & 511], (long long) iq2xs_grid[v[2] & 511],
                             (long long) iq2xs_grid[v[1] & 511], (long long) iq2xs_grid[v[0] & 511]);
        m = sign_mask_from_idx(_mm256_srli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*) q)), 9));
        const uint8_t s0 = b[66 + 2 * j], s1 = b[66 + 2 * j + 1];
        sc = scales4(2 * (s0 & 15) + 1, 2 * (s0 >> 4) + 1, 2 * (s1 & 15) + 1, 2 * (s1 >> 4) + 1);
    }
};
template <> struct FmtX<18> : Fmt<18> {
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        g = gather ? _mm512_i32gather_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) q)), iq3xxs_grid, 4)
                   : _mm512_set_epi32((int) iq3xxs_grid[q[15]], (int) iq3xxs_grid[q[14]], (int) iq3xxs_grid[q[13]], (int) iq3xxs_grid[q[12]],
                             (int) iq3xxs_grid[q[11]], (int) iq3xxs_grid[q[10]], (int) iq3xxs_grid[q[9]], (int) iq3xxs_grid[q[8]],
                             (int) iq3xxs_grid[q[7]], (int) iq3xxs_grid[q[6]], (int) iq3xxs_grid[q[5]], (int) iq3xxs_grid[q[4]],
                             (int) iq3xxs_grid[q[3]], (int) iq3xxs_grid[q[2]], (int) iq3xxs_grid[q[1]], (int) iq3xxs_grid[q[0]]);
        const uint32_t a = u32(b + 2 + 64 + 8 * j), c = u32(b + 2 + 64 + 8 * j + 4);
        m = sign_mask_from_idx(sign_idx_xxs(a, c));
        const int sa = 2 * (int) (a >> 28) + 1, sb = 2 * (int) (c >> 28) + 1;
        sc = scales4(sa, sa, sb, sb);
    }
};

// ---- the DeepSeek CPU lane's exact rows (iq512x_rows): the same 64-value decode, but summed as ggml's AVX2 dots sum, so a
// token's row is ggml-cpu's vec_dot bit for bit at any number of tokens: the integer sums of a block fold to ggml's eight lanes
// (low half = the even 32-value chunk, high half = the odd one), the block scale is f16(d) * y.d, eight float lanes accumulate
// across blocks, and K * ggml's hsum comes last.  vpdpwssd is the madd and the add in one.
inline __m256i fold256(__m512i a) { return _mm256_add_epi32(_mm512_castsi512_si256(a), _mm512_extracti64x4_epi64(a, 1)); }

inline float hsum8_ggml(__m256 x) {
    __m128 s = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

template <int TY, int NT>
inline void row_dot_x(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    const __m512i zero = _mm512_setzero_si512();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * FmtX<TY>::bytes;
        if (prefetch_ahead > 0) {
            const uint8_t* ahead = blk + (size_t) prefetch_ahead;
            _mm_prefetch((const char*) ahead, _MM_HINT_T0);
            _mm_prefetch((const char*) ahead + 64, _MM_HINT_T0);
        }
        __m512i acci[NT];
        for (int t = 0; t < NT; ++t) acci[t] = _mm512_setzero_si512();
        for (int j = 0; j < 4; ++j) {
            __m512i g, sc;
            __mmask64 m;
            FmtX<TY>::decode(blk, j, g, m, sc);
            for (int t = 0; t < NT; ++t) {
                const __m512i yv = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * j));
                const __m512i ys = _mm512_mask_sub_epi8(yv, m, zero, yv);
                acci[t] = _mm512_dpwssd_epi32(acci[t], _mm512_maddubs_epi16(g, ys), sc);
            }
        }
        const float dx = h2f(u16(blk));
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(fold256(acci[t])), accf[t]);
    }
    for (int t = 0; t < NT; ++t) res[t] = FmtX<TY>::K * hsum8_ggml(accf[t]);
}

template <int TY, int NT>
void dot_rows_x(const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot_x<TY, NT>(w + (size_t) r * row_bytes, n / QK_K, y, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

// many tokens an expert (the prompt): signed int8 weights decoded once per block, activations as u8 (q8 + 128, once a call),
// sum(w * q) = sum(w * (q + 128)) - 128 * sum(w) with the second sum once a weight block; tokens in tiles of 16 and rows in
// groups of 8 so the tile's slice of a block and the group's accumulators stay in L1 (the AVX2 form's layout)
template <int TY>
void dot_rows_wide(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int T, float* const* out, int r0, int r1) {
    constexpr int kTile = 16, kRows = 8;
    const int nb = n / QK_K;
    static thread_local std::vector<uint8_t> ubuf;
    static thread_local std::vector<float> accb;
    ubuf.resize((size_t) (T < kTile ? T : kTile) * n);
    accb.resize((size_t) kRows * kTile * 8);
    const __m512i zero = _mm512_setzero_si512(), one8 = _mm512_set1_epi8(1), flip = _mm512_set1_epi8((char) 0x80);
    for (int t0 = 0; t0 < T; t0 += kTile) {
        const int tt = T - t0 < kTile ? T - t0 : kTile;
        const block_q8_K* y[kTile];
        for (int t = 0; t < tt; ++t) {
            y[t] = (const block_q8_K*) act[t0 + t];
            uint8_t* u = ubuf.data() + (size_t) t * n;
            for (int i = 0; i < nb; ++i)
                for (int j = 0; j < 4; ++j)
                    _mm512_storeu_si512((void*) (u + (size_t) i * 256 + 64 * j),
                                        _mm512_xor_si512(_mm512_loadu_si512((const void*) (y[t][i].qs + 64 * j)), flip));
        }
        for (int rg = r0; rg < r1; rg += kRows) {
            const int rr = r1 - rg < kRows ? r1 - rg : kRows;
            for (int k = 0; k < rr * tt; ++k) _mm256_storeu_ps(&accb[(size_t) k * 8], _mm256_setzero_ps());
            for (int i = 0; i < nb; ++i)
                for (int q = 0; q < rr; ++q) {
                    const uint8_t* blk = w + (size_t) (rg + q) * row_bytes + (size_t) i * FmtX<TY>::bytes;
                    if (prefetch_ahead > 0) {
                        _mm_prefetch((const char*) blk + prefetch_ahead, _MM_HINT_T0);
                        _mm_prefetch((const char*) blk + prefetch_ahead + 64, _MM_HINT_T0);
                    }
                    __m512i w8[4], sc[4];
                    __m512i corr = zero;
                    for (int j = 0; j < 4; ++j) {
                        __m512i g, s;
                        __mmask64 m;
                        FmtX<TY>::decode(blk, j, g, m, s);
                        w8[j] = _mm512_mask_sub_epi8(g, m, zero, g);
                        sc[j] = s;
                        corr = _mm512_dpwssd_epi32(corr, _mm512_maddubs_epi16(one8, w8[j]), s);
                    }
                    const __m256i corr128 = _mm256_slli_epi32(fold256(corr), 7);
                    const float dx = h2f(u16(blk));
                    float* accq = &accb[(size_t) q * tt * 8];
                    for (int t = 0; t < tt; ++t) {
                        const uint8_t* u = ubuf.data() + (size_t) t * n + (size_t) i * 256;
                        __m512i a = zero;
                        for (int j = 0; j < 4; ++j)
                            a = _mm512_dpwssd_epi32(a, _mm512_maddubs_epi16(_mm512_loadu_si512((const void*) (u + 64 * j)), w8[j]), sc[j]);
                        const __m256i s = _mm256_sub_epi32(fold256(a), corr128);
                        float* ap = accq + (size_t) t * 8;
                        _mm256_storeu_ps(ap, _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(s), _mm256_loadu_ps(ap)));
                    }
                }
            for (int q = 0; q < rr; ++q)
                for (int t = 0; t < tt; ++t)
                    out[t0 + t][rg + q] = FmtX<TY>::K * hsum8_ggml(_mm256_loadu_ps(&accb[((size_t) q * tt + t) * 8]));
        }
    }
}

template <int TY>
void dot_rows_x_nt(int nt, const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0, int r1) {
    switch (nt) {
        case 1: dot_rows_x<TY, 1>(w, row_bytes, n, act, out, r0, r1); break;
        case 2: dot_rows_x<TY, 2>(w, row_bytes, n, act, out, r0, r1); break;
        case 3: dot_rows_x<TY, 3>(w, row_bytes, n, act, out, r0, r1); break;
        case 4: dot_rows_x<TY, 4>(w, row_bytes, n, act, out, r0, r1); break;
        case 5: dot_rows_x<TY, 5>(w, row_bytes, n, act, out, r0, r1); break;
        case 6: dot_rows_x<TY, 6>(w, row_bytes, n, act, out, r0, r1); break;
        default: dot_rows_wide<TY>(w, row_bytes, n, act, nt, out, r0, r1); break;
    }
}


// ---- IQ1_M (29), AVX-512, 1..6 tokens: ggml's dot lane for lane.  The grid indices come out of two pdep (as ggml's BMI2 path), the
// +-1 delta signs out of one pext into an 8-bit mask a 64-value step; the signed grid bytes are split into |g| and a negative mask;
// the delta dot is the pair sums of the activation (made once a call) times +-scale.  d is f16 in the top nibbles of the scales.
inline void iq1m_pre(const block_q8_K* y, int nb, int16_t* p) {
    const __m256i one8 = _mm256_set1_epi8(1);
    for (int i = 0; i < nb; ++i)
        for (int c = 0; c < 8; ++c)
            _mm256_storeu_si256((__m256i*) (p + ((size_t) i * 8 + c) * 16),
                                _mm256_maddubs_epi16(one8, _mm256_loadu_si256((const __m256i*) (y[i].qs + 32 * c))));
}

template <int NT>
void rows_m(const uint8_t* w, size_t rb, int n, const block_q8_K* const* y, const int16_t* const* P, float* const* out, int r0, int r1) {
    const int nb = n / QK_K;
    const __m256i mask = _mm256_set1_epi16(0x7), mone = _mm256_set1_epi16(1), mtwo8 = _mm256_set1_epi8(2);
    const __m256i scales_shift = _mm256_set_epi64x(9, 3, 6, 0);
    const __m512i zero = _mm512_setzero_si512();
    // the 32 scale words a 64-weight step (8 each of chunks 4 it .. 4 it + 3) out of the 16 decoded scales (their order: 4 e + j for
    // shift e, word j): one vpermw a step instead of two vpshufb + adds + an insert
    __m512i sidx[4];
    for (int it = 0; it < 4; ++it)
        sidx[it] = _mm512_set_epi16(12 + it, 12 + it, 12 + it, 12 + it, 12 + it, 12 + it, 12 + it, 12 + it, 4 + it, 4 + it, 4 + it, 4 + it, 4 + it, 4 + it,
                                    4 + it, 4 + it, 8 + it, 8 + it, 8 + it, 8 + it, 8 + it, 8 + it, 8 + it, 8 + it, it, it, it, it, it, it, it, it);
    for (int r = r0; r < r1; ++r) {
        __m256 a1[NT], a2[NT];
        for (int t = 0; t < NT; ++t) a1[t] = a2[t] = _mm256_setzero_ps();
        const uint8_t* row = w + (size_t) r * rb;
        for (int i = 0; i < nb; ++i) {
            const uint8_t* blk = row + (size_t) i * 56;
            const uint8_t* qs = blk;
            const uint8_t* qh = blk + 32;
            const uint8_t* scb = blk + 48;
            if (prefetch_ahead > 0) {
                _mm_prefetch((const char*) blk + prefetch_ahead, _MM_HINT_T0);
                _mm_prefetch((const char*) blk + prefetch_ahead + 64, _MM_HINT_T0);
            }
            const uint16_t sc0 = u16(scb), sc1 = u16(scb + 2), sc2 = u16(scb + 4), sc3 = u16(scb + 6);
            const uint16_t d16 = (uint16_t) ((sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000));
            __m256i scales = _mm256_set1_epi64x((long long) u64(scb));
            scales = _mm256_srlv_epi64(scales, scales_shift);
            scales = _mm256_add_epi16(_mm256_slli_epi16(_mm256_and_si256(scales, mask), 1), mone);
            const __m512i sctab = _mm512_castsi256_si512(scales);
            __m512i s1[NT], s2[NT];
            for (int t = 0; t < NT; ++t) s1[t] = s2[t] = zero;
            for (int it = 0; it < 4; ++it) {
                const uint32_t qh32 = u32(qh + 4 * it);
                const uint64_t p1 = _pdep_u64(u32(qs + 8 * it), 0x00ff00ff00ff00ffULL) | _pdep_u64(qh32 & 0x7777, 0x0f000f000f000f00ULL);
                const uint64_t p2 = _pdep_u64(u32(qs + 8 * it + 4), 0x00ff00ff00ff00ffULL) | _pdep_u64((qh32 >> 16) & 0x7777, 0x0f000f000f000f00ULL);
                uint16_t ix[8];
                std::memcpy(ix, &p1, 8);
                std::memcpy(ix + 4, &p2, 8);
                const __m512i g = _mm512_set_epi64((long long) iq1s_grid[ix[7]], (long long) iq1s_grid[ix[6]], (long long) iq1s_grid[ix[5]],
                                                   (long long) iq1s_grid[ix[4]], (long long) iq1s_grid[ix[3]], (long long) iq1s_grid[ix[2]],
                                                   (long long) iq1s_grid[ix[1]], (long long) iq1s_grid[ix[0]]);
                const __m512i sc = _mm512_permutexvar_epi16(sidx[it], sctab);
                // the delta sign of each 8-weight group (bit 3 of its qh nibble) negates that group's 4 scale words
                const __mmask32 dneg = (__mmask32) (_pdep_u32(_pext_u32(qh32, 0x88888888u), 0x11111111u) * 15u);
                const __m512i ssc = _mm512_mask_sub_epi16(sc, dneg, zero, sc);
                const __m512i ag = _mm512_abs_epi8(g);
                const __mmask64 gneg = _mm512_movepi8_mask(g);
                for (int t = 0; t < NT; ++t) {
                    const __m512i q8 = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * it));
                    const __m512i pp = _mm512_loadu_si512((const void*) (P[t] + ((size_t) i * 8 + 2 * it) * 16));
                    const __m512i ys = _mm512_mask_sub_epi8(q8, gneg, zero, q8);
                    s1[t] = _mm512_dpwssd_epi32(s1[t], _mm512_maddubs_epi16(ag, ys), sc);
                    s2[t] = _mm512_dpwssd_epi32(s2[t], pp, ssc);
                }
            }
            const float dx = h2f(d16);
            for (int t = 0; t < NT; ++t) {
                const __m256 d = _mm256_set1_ps(y[t][i].d * dx);
                a1[t] = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(fold256(s1[t])), a1[t]);
                a2[t] = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(fold256(s2[t])), a2[t]);
            }
        }
        for (int t = 0; t < NT; ++t) out[t][r] = hsum8_ggml(a1[t]) + 0.125f * hsum8_ggml(a2[t]);
    }
}

void rows_m_nt(int nt, const uint8_t* w, size_t rb, int n, const void* const* act, float* const* out, int r0, int r1) {
    const int nb = n / QK_K;
    static thread_local std::vector<int16_t> pre;
    for (int t0 = 0; t0 < nt; t0 += 6) {
        const int k = nt - t0 < 6 ? nt - t0 : 6;
        const block_q8_K* y[6];
        const int16_t* P[6];
        float* o[6];
        pre.resize((size_t) k * n);
        for (int t = 0; t < k; ++t) {
            y[t] = (const block_q8_K*) act[t0 + t];
            iq1m_pre(y[t], nb, pre.data() + (size_t) t * n);
            P[t] = pre.data() + (size_t) t * n;
            o[t] = out[t0 + t];
        }
        switch (k) {
            case 1: rows_m<1>(w, rb, n, y, P, o, r0, r1); break;
            case 2: rows_m<2>(w, rb, n, y, P, o, r0, r1); break;
            case 3: rows_m<3>(w, rb, n, y, P, o, r0, r1); break;
            case 4: rows_m<4>(w, rb, n, y, P, o, r0, r1); break;
            case 5: rows_m<5>(w, rb, n, y, P, o, r0, r1); break;
            default: rows_m<6>(w, rb, n, y, P, o, r0, r1); break;
        }
    }
}

}  // namespace

void iq512_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1) {
    switch (type) {
        case 16: gu_rows_nt<16>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 17: gu_rows_nt<17>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 18: gu_rows_nt<18>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 21: gu_rows_nt<21>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 22: gu_rows_nt<22>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: break;
    }
}

// The linked ggml AVX2 singleton order, for the formats validated against its production dot.
void iq512_reference_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                             const void* const* act, int nt, float* const* ff, int r0, int r1) {
    switch (type) {
        case 18: gu_rows_nt<18, true>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 21: gu_rows_nt<21, true>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 22: gu_rows_nt<22, true>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
    }
}

void iq512_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                int r0, int r1) {
    switch (type) {
        case 16: dot_rows_nt<16>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 17: dot_rows_nt<17>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 18: dot_rows_nt<18>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 21: dot_rows_nt<21>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 22: dot_rows_nt<22>(nt, w, row_bytes, n, act, out, r0, r1); break;
        default: break;
    }
}

static bool has_vpopcnt() {
    static const bool ok = [] {
#if defined(_MSC_VER)
        int r[4];
        __cpuidex(r, 7, 0);
        return ((r[2] >> 14) & 1) != 0;   // AVX512_VPOPCNTDQ
#else
        unsigned a, b, c, d;
        __cpuid_count(7, 0, a, b, c, d);
        return ((c >> 14) & 1u) != 0;
#endif
    }();
    return ok;
}

bool iq512_gather_on() noexcept { return gather; }

bool iq512x_supported(int type) noexcept {
    return has_vpopcnt() && (type == 16 || type == 17 || type == 18 || type == 21 || type == 22 || type == 29);
}

void iq512x_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                 int r0, int r1) {
    switch (type) {
        case 16: dot_rows_x_nt<16>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 17: dot_rows_x_nt<17>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 18: dot_rows_x_nt<18>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 21: dot_rows_x_nt<21>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 22: dot_rows_x_nt<22>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 29: rows_m_nt(nt, w, row_bytes, n, act, out, r0, r1); break;
        default: break;
    }
}

}  // namespace strata::kernels::cpu
