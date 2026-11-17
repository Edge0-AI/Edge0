#if !(defined(__x86_64__) || defined(_M_X64))   // (x86 defines it below, inside its guard)
// the band GEMM's chunk width, defined here because the uses are outside the AVX-512 guard
constexpr int kGemmK = 256;
#endif
// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/kernels/cpu/exl3_cpu.cpp - EXL3 routed experts on the CPU (see include/strata/kernels/cpu/exl3_cpu.hpp).
//
// The vector kernels follow exllamav3's CPU expert kernel (MIT, turboderp - exllamav3_ext/cpu/moe_mul1.cpp, its
// AVX-512 BW and AVX2 tiers): a tile row's 16 states come out of the tile's words with cross-lane permutes and a
// per-lane funnel shift from compile-time tables, the codebook product's bytes are added with vpmaddubsw against ones
// and vpmaddwd against ones.  Unlike there, the activations stay F32 (no int8 rows) and the affine offset is the
// device's FP16 constant, so the CPU lane computes what the device kernels compute up to the FP16 rounding of each
// weight.  The AVX-512 and AVX2 tiers do the same FMAs in the same order: they give the same bits (the bands,
// the split-K pieces, the GEMM's sums and the vectorised transforms alike).
#include "strata/dsv41/exl3_cpu.hpp"

#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#define EXL3_X86 1
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

#if defined(EXL3_X86) && (defined(__GNUC__) || defined(__clang__))
#define EXL3_AVX512 __attribute__((target("avx512f,avx512bw,fma")))
#define EXL3_AVX2 __attribute__((target("avx2,fma,f16c")))
#define EXL3_AVX2_INLINE __attribute__((target("avx2,fma,f16c"), always_inline)) inline
#elif defined(_MSC_VER)
#define EXL3_AVX512
#define EXL3_AVX2
#define EXL3_AVX2_INLINE __forceinline
#else
#define EXL3_AVX512
#define EXL3_AVX2
#define EXL3_AVX2_INLINE inline
#endif

namespace strata::dsv41k::cpu {
// Strata's cpu_avx2_ok lives in strata_kernels_cpu, which this path does not link: AVX2 + FMA here
static bool cpu_avx2_ok() {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}
using strata::kernels::f32_from_f16;
using strata::kernels::f16_from_f32;
namespace {

constexpr uint32_t kMul1 = 0x83DCD12Du;

float k_inv() { return f32_from_f16(0x1eee); }
float k_zero() { return 1024.0f * f32_from_f16(0x1eee) + f32_from_f16(0xc931); }   // w = s * k_inv + k_zero

// the 128-point Sylvester transform over sqrt(128), in place
void had128(float* v) {
    for (int h = 1; h < 128; h <<= 1)
        for (int i = 0; i < 128; i += 2 * h)
            for (int j = i; j < i + h; ++j) {
                const float a = v[j], b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
    const float s = 0.088388347648f;
    for (int i = 0; i < 128; ++i) v[i] *= s;
}

// step p of a tile: row (p/8 % 4) * 2 + (p & 1) + 8 (p/2 & 1), column p/32 + 8 (p/4 & 1); inv[row * 16 + col] = p
constexpr std::array<uint16_t, 256> make_inv() {
    std::array<uint16_t, 256> inv{};
    for (int p = 0; p < 256; ++p) {
        const int lane = p >> 3, j = p & 7;
        const int r = (lane & 3) * 2 + (j & 1) + 8 * ((j >> 1) & 1);
        const int c = (lane >> 2) + 8 * (j >> 2);
        inv[(size_t) (r * 16 + c)] = (uint16_t) p;
    }
    return inv;
}

// a tile row's 16 states: column c's 16-bit window ends at bit b1 = (p + 1 + 256) K of the tile's MSB-first stream;
// its words i0 (the window's first bit) and i1 (its last), and the right shift of (w[i0] : w[i1]) that aligns it
template <int K>
struct RowTab {
    int32_t i0[16][16] = {}, i1[16][16] = {}, sr[16][16] = {}, sl[16][16] = {};
    uint16_t hi0[16] = {}, hi1[16] = {};   // columns whose word is in the second register pair (>= 32)
};
template <int K>
constexpr RowTab<K> make_rowtab() {
    RowTab<K> t{};
    constexpr int W = 8 * K;
    const auto inv = make_inv();
    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 16; ++c) {
            const int p = inv[(size_t) (r * 16 + c)];
            const int b1 = (p + 1 + 256) * K;
            const int w1 = ((b1 - 1) >> 5) % W, w0 = ((b1 - 16) >> 5) % W;
            const int sh = (((b1 - 1) >> 5) + 1) * 32 - b1;
            t.i0[r][c] = w0;
            t.i1[r][c] = w1;
            t.sr[r][c] = sh;
            t.sl[r][c] = 32 - sh;   // a shift by 32 clears the lane: a window inside one word
            if (w0 >= 32) t.hi0[r] = (uint16_t) (t.hi0[r] | (1u << c));
            if (w1 >= 32) t.hi1[r] = (uint16_t) (t.hi1[r] | (1u << c));
        }
    return t;
}

// ---------------------------------------------------------------- scalar
template <int K>
void band_scalar(const Exl3Part& m, const float* xh, int tc0, int ks0, int ks1, float* acc /* 128 */) {
    constexpr int W = 8 * K;
    static constexpr RowTab<K> T = make_rowtab<K>();
    const int ld = m.n / 16;
    for (int i = 0; i < 128; ++i) acc[i] = 0.0f;
    for (int ks = ks0; ks < ks1; ++ks)
        for (int t = 0; t < 8; ++t) {
            const uint32_t* w = m.trellis + ((size_t) ks * ld + tc0 + t) * W;
            for (int r = 0; r < 16; ++r) {
                const float xv = xh[ks * 16 + r];
                for (int c = 0; c < 16; ++c) {
                    const uint64_t a = w[T.i0[r][c]], b = w[T.i1[r][c]];
                    const uint32_t st = (uint32_t) (((a << 32) | b) >> T.sr[r][c]) & 0xffffu;
                    const uint32_t x = st * kMul1;
                    const int s = (int) ((x & 0xffu) + ((x >> 8) & 0xffu) + ((x >> 16) & 0xffu) + (x >> 24));
                    acc[t * 16 + c] += (float) s * xv;
                }
            }
        }
}

// band_gemm's chunk of inputs (see below); outside the x86 section, as exl3_cpu_gemm_scratch() sizes by it on any CPU
constexpr int kGemmK = 256;

// ---------------------------------------------------------------- AVX-512
#if defined(EXL3_X86)
template <int K>
EXL3_AVX512 inline __m512i gather_words(const RowTab<K>& T, const int32_t (&idx)[16][16], const uint16_t* hi, int r,
                                        __m512i p0, __m512i p1, __m512i p2, __m512i p3) {
    const __m512i ix = _mm512_loadu_si512((const void*) idx[r]);
    __m512i v = _mm512_permutex2var_epi32(p0, ix, p1);
    if constexpr (8 * K > 32) {
        if (hi[r] != 0) v = _mm512_mask_blend_epi32((__mmask16) hi[r], v, _mm512_permutex2var_epi32(p2, ix, p3));
    }
    (void) T;
    return v;
}

template <int K>
EXL3_AVX512 void band_avx512(const Exl3Part& m, const float* xh, int tc0, int ks0, int ks1, float* out /* 128 */) {
    constexpr int W = 8 * K;
    static constexpr RowTab<K> T = make_rowtab<K>();
    constexpr auto ld_mask = [](int n) -> __mmask16 {
        return n >= 16 ? (__mmask16) 0xffffu : n <= 0 ? (__mmask16) 0 : (__mmask16) ((1u << n) - 1u);
    };
    constexpr __mmask16 m0 = ld_mask(W), m1 = ld_mask(W - 16), m2 = ld_mask(W - 32), m3 = ld_mask(W - 48);
    const int ld = m.n / 16;
    const size_t row_words = (size_t) ld * W;
    const __m512i mult = _mm512_set1_epi32((int) kMul1);
    const __m512i ones8 = _mm512_set1_epi8(1), ones16 = _mm512_set1_epi16(1);
    const __m512i low16 = _mm512_set1_epi32(0xffff);
    __m512 acc[8];
    for (int t = 0; t < 8; ++t) acc[t] = _mm512_setzero_ps();
    const uint32_t* row = m.trellis + (size_t) ks0 * row_words + (size_t) tc0 * W;
    for (int ks = ks0; ks < ks1; ++ks, row += row_words) {
        // the band's 8 tiles of this slice are contiguous; the next slices' are one row of tiles on
        const char* pf = (const char*) (row + 4 * row_words);
        for (int l = 0; l < 8 * W * 4; l += 64) _mm_prefetch(pf + l, _MM_HINT_T0);
        const float* xk = xh + ks * 16;
        for (int t = 0; t < 8; ++t) {
            const uint32_t* w = row + t * W;
            const __m512i p0 = _mm512_maskz_loadu_epi32(m0, w);
            const __m512i p1 = m1 ? _mm512_maskz_loadu_epi32(m1, w + 16) : _mm512_setzero_si512();
            const __m512i p2 = m2 ? _mm512_maskz_loadu_epi32(m2, w + 32) : _mm512_setzero_si512();
            const __m512i p3 = m3 ? _mm512_maskz_loadu_epi32(m3, w + 48) : _mm512_setzero_si512();
            __m512 a = acc[t];
#pragma GCC unroll 16
            for (int r = 0; r < 16; ++r) {
                const __m512i wa = gather_words<K>(T, T.i0, T.hi0, r, p0, p1, p2, p3);
                const __m512i wb = gather_words<K>(T, T.i1, T.hi1, r, p0, p1, p2, p3);
                const __m512i st = _mm512_and_si512(
                    _mm512_or_si512(_mm512_srlv_epi32(wb, _mm512_loadu_si512((const void*) T.sr[r])),
                                    _mm512_sllv_epi32(wa, _mm512_loadu_si512((const void*) T.sl[r]))),
                    low16);
                const __m512i bs = _mm512_madd_epi16(_mm512_maddubs_epi16(_mm512_mullo_epi32(st, mult), ones8), ones16);
                a = _mm512_fmadd_ps(_mm512_cvtepi32_ps(bs), _mm512_set1_ps(xk[r]), a);
            }
            acc[t] = a;
        }
    }
    for (int t = 0; t < 8; ++t) _mm512_storeu_ps(out + t * 16, acc[t]);
}

// band_avx512 for N activations at once (xh[i] -> out[i]), as exllamav3's CPU kernel takes up to 4 token rows: each
// tile row's 16 states are pulled out and byte-summed once, then one FMA per activation - activation i's FMAs are
// band_avx512's for it alone, in its order, so out[i] is the same bit for bit.  An extra activation costs one FMA of
// the row's ~12 instructions.  (The tiles' accumulators live in memory, loaded once a tile and slice as there: only
// the N of the tile in hand take registers.)
template <int K, int N>
EXL3_AVX512 void band_avx512_n(const Exl3Part& m, const float* const* xh, int tc0, int ks0, int ks1,
                               float* const* out /* N x 128 */) {
    constexpr int W = 8 * K;
    static constexpr RowTab<K> T = make_rowtab<K>();
    constexpr auto ld_mask = [](int n) -> __mmask16 {
        return n >= 16 ? (__mmask16) 0xffffu : n <= 0 ? (__mmask16) 0 : (__mmask16) ((1u << n) - 1u);
    };
    constexpr __mmask16 m0 = ld_mask(W), m1 = ld_mask(W - 16), m2 = ld_mask(W - 32), m3 = ld_mask(W - 48);
    const int ld = m.n / 16;
    const size_t row_words = (size_t) ld * W;
    const __m512i mult = _mm512_set1_epi32((int) kMul1);
    const __m512i ones8 = _mm512_set1_epi8(1), ones16 = _mm512_set1_epi16(1);
    const __m512i low16 = _mm512_set1_epi32(0xffff);
    __m512 acc[8][N];
    for (int t = 0; t < 8; ++t)
        for (int i = 0; i < N; ++i) acc[t][i] = _mm512_setzero_ps();
    const uint32_t* row = m.trellis + (size_t) ks0 * row_words + (size_t) tc0 * W;
    for (int ks = ks0; ks < ks1; ++ks, row += row_words) {
        const char* pf = (const char*) (row + 4 * row_words);
        for (int l = 0; l < 8 * W * 4; l += 64) _mm_prefetch(pf + l, _MM_HINT_T0);
        const float* xk[N];
        for (int i = 0; i < N; ++i) xk[i] = xh[i] + ks * 16;
        for (int t = 0; t < 8; ++t) {
            const uint32_t* w = row + t * W;
            const __m512i p0 = _mm512_maskz_loadu_epi32(m0, w);
            const __m512i p1 = m1 ? _mm512_maskz_loadu_epi32(m1, w + 16) : _mm512_setzero_si512();
            const __m512i p2 = m2 ? _mm512_maskz_loadu_epi32(m2, w + 32) : _mm512_setzero_si512();
            const __m512i p3 = m3 ? _mm512_maskz_loadu_epi32(m3, w + 48) : _mm512_setzero_si512();
            __m512 a[N];
#pragma GCC unroll 8
            for (int i = 0; i < N; ++i) a[i] = acc[t][i];
#pragma GCC unroll 16
            for (int r = 0; r < 16; ++r) {
                const __m512i wa = gather_words<K>(T, T.i0, T.hi0, r, p0, p1, p2, p3);
                const __m512i wb = gather_words<K>(T, T.i1, T.hi1, r, p0, p1, p2, p3);
                const __m512i st = _mm512_and_si512(
                    _mm512_or_si512(_mm512_srlv_epi32(wb, _mm512_loadu_si512((const void*) T.sr[r])),
                                    _mm512_sllv_epi32(wa, _mm512_loadu_si512((const void*) T.sl[r]))),
                    low16);
                const __m512 f = _mm512_cvtepi32_ps(
                    _mm512_madd_epi16(_mm512_maddubs_epi16(_mm512_mullo_epi32(st, mult), ones8), ones16));
#pragma GCC unroll 8
                for (int i = 0; i < N; ++i) a[i] = _mm512_fmadd_ps(f, _mm512_set1_ps(xk[i][r]), a[i]);
            }
#pragma GCC unroll 8
            for (int i = 0; i < N; ++i) acc[t][i] = a[i];
        }
    }
    for (int i = 0; i < N; ++i)
        for (int t = 0; t < 8; ++t) _mm512_storeu_ps(out[i] + t * 16, acc[t][i]);
}

// rows_avx512_n: slices [ks0, ks1) of the tile columns [t0, t1) - a split-K piece, or a run of its columns - for N
// activations (xh[i] -> acc[i], indexed by column; columns 16 t0 .. 16 t1 overwritten).  The tile rows are read in
// memory order, where a band's tiles are 1 KB runs a whole tile row apart: 40 threads of a 2-socket Xeon 6152 read a K4
// expert from DRAM in 0.14 ms in pieces, 0.21 in bands (~90 vs ~60 GB/s), the kernel the same.  A tile's 16 sums per
// activation live in acc between slices; each row's decode and FMAs are band_avx512's, so acc[i] is the N = 1 call's
// for xh[i] alone, bit for bit, whatever the columns' run.  The prefetch runs 16 tiles ahead in the job's own order
// (into the next slice past the run's end, not the next job's columns)
template <int K, int N>
EXL3_AVX512 void rows_avx512_n(const Exl3Part& m, const float* const* xh, int ks0, int ks1, int t0, int t1,
                               float* const* acc) {
    constexpr int W = 8 * K;
    static constexpr RowTab<K> T = make_rowtab<K>();
    constexpr auto ld_mask = [](int n) -> __mmask16 {
        return n >= 16 ? (__mmask16) 0xffffu : n <= 0 ? (__mmask16) 0 : (__mmask16) ((1u << n) - 1u);
    };
    constexpr __mmask16 m0 = ld_mask(W), m1 = ld_mask(W - 16), m2 = ld_mask(W - 32), m3 = ld_mask(W - 48);
    const int ld = m.n / 16;
    const size_t row_words = (size_t) ld * W;
    const __m512i mult = _mm512_set1_epi32((int) kMul1);
    const __m512i ones8 = _mm512_set1_epi8(1), ones16 = _mm512_set1_epi16(1);
    const __m512i low16 = _mm512_set1_epi32(0xffff);
    for (int i = 0; i < N; ++i)
        for (int c = 16 * t0; c < 16 * t1; c += 16) _mm512_storeu_ps(acc[i] + c, _mm512_setzero_ps());
    const int span = t1 - t0;
    const uint32_t* row = m.trellis + (size_t) ks0 * row_words;
    for (int ks = ks0; ks < ks1; ++ks, row += row_words) {
        float xk[N][16];
        for (int i = 0; i < N; ++i)
            for (int r = 0; r < 16; ++r) xk[i][r] = xh[i][ks * 16 + r];
        for (int t = t0; t < t1; ++t) {
            const uint32_t* w = row + (size_t) t * W;
            const uint32_t* pf = t + 16 < t1 || span == ld ? w + 16 * W : row + row_words + (size_t) (t + 16 - span) * W;
            for (int l = 0; l < W * 4; l += 64) _mm_prefetch((const char*) pf + l, _MM_HINT_T0);
            const __m512i p0 = _mm512_maskz_loadu_epi32(m0, w);
            const __m512i p1 = m1 ? _mm512_maskz_loadu_epi32(m1, w + 16) : _mm512_setzero_si512();
            const __m512i p2 = m2 ? _mm512_maskz_loadu_epi32(m2, w + 32) : _mm512_setzero_si512();
            const __m512i p3 = m3 ? _mm512_maskz_loadu_epi32(m3, w + 48) : _mm512_setzero_si512();
            __m512 a[N];
#pragma GCC unroll 8
            for (int i = 0; i < N; ++i) a[i] = _mm512_loadu_ps(acc[i] + t * 16);
#pragma GCC unroll 16
            for (int r = 0; r < 16; ++r) {
                const __m512i wa = gather_words<K>(T, T.i0, T.hi0, r, p0, p1, p2, p3);
                const __m512i wb = gather_words<K>(T, T.i1, T.hi1, r, p0, p1, p2, p3);
                const __m512i st = _mm512_and_si512(
                    _mm512_or_si512(_mm512_srlv_epi32(wb, _mm512_loadu_si512((const void*) T.sr[r])),
                                    _mm512_sllv_epi32(wa, _mm512_loadu_si512((const void*) T.sl[r]))),
                    low16);
                const __m512 f = _mm512_cvtepi32_ps(
                    _mm512_madd_epi16(_mm512_maddubs_epi16(_mm512_mullo_epi32(st, mult), ones8), ones16));
#pragma GCC unroll 8
                for (int i = 0; i < N; ++i) a[i] = _mm512_fmadd_ps(f, _mm512_set1_ps(xk[i][r]), a[i]);
            }
#pragma GCC unroll 8
            for (int i = 0; i < N; ++i) _mm512_storeu_ps(acc[i] + t * 16, a[i]);
        }
    }
}

// band_gemm: a band's byte-sum products for many activations - each chunk of kGemmK inputs decoded ONCE into F32
// (wc, in L2), then multiplied as a GEMM: tiles of TB activations x 64 columns, the sums in registers, two FMAs a
// load.  The prompt path's CPU rows, where an expert serves tens to hundreds of tokens: one decode (~6 instructions a
// 16-weight row) for all of them instead of one per kExl3CpuMaxN.  (kGemmK is defined above the AVX-512 section.)

template <int K>
EXL3_AVX512 void decode_chunk(const Exl3Part& m, int tc0, int ks0, int ks1, float* wc /* (ks1 - ks0) 16 x 128 */) {
    constexpr int W = 8 * K;
    static constexpr RowTab<K> T = make_rowtab<K>();
    constexpr auto ld_mask = [](int n) -> __mmask16 {
        return n >= 16 ? (__mmask16) 0xffffu : n <= 0 ? (__mmask16) 0 : (__mmask16) ((1u << n) - 1u);
    };
    constexpr __mmask16 m0 = ld_mask(W), m1 = ld_mask(W - 16), m2 = ld_mask(W - 32), m3 = ld_mask(W - 48);
    const size_t row_words = (size_t) (m.n / 16) * W;
    const __m512i mult = _mm512_set1_epi32((int) kMul1);
    const __m512i ones8 = _mm512_set1_epi8(1), ones16 = _mm512_set1_epi16(1);
    const __m512i low16 = _mm512_set1_epi32(0xffff);
    const uint32_t* row = m.trellis + (size_t) ks0 * row_words + (size_t) tc0 * W;
    for (int ks = ks0; ks < ks1; ++ks, row += row_words) {
        const char* pf = (const char*) (row + 4 * row_words);
        for (int l = 0; l < 8 * W * 4; l += 64) _mm_prefetch(pf + l, _MM_HINT_T0);
        float* wr = wc + (size_t) (ks - ks0) * 16 * 128;
        for (int t = 0; t < 8; ++t) {
            const uint32_t* w = row + t * W;
            const __m512i p0 = _mm512_maskz_loadu_epi32(m0, w);
            const __m512i p1 = m1 ? _mm512_maskz_loadu_epi32(m1, w + 16) : _mm512_setzero_si512();
            const __m512i p2 = m2 ? _mm512_maskz_loadu_epi32(m2, w + 32) : _mm512_setzero_si512();
            const __m512i p3 = m3 ? _mm512_maskz_loadu_epi32(m3, w + 48) : _mm512_setzero_si512();
#pragma GCC unroll 16
            for (int r = 0; r < 16; ++r) {
                const __m512i wa = gather_words<K>(T, T.i0, T.hi0, r, p0, p1, p2, p3);
                const __m512i wb = gather_words<K>(T, T.i1, T.hi1, r, p0, p1, p2, p3);
                const __m512i st = _mm512_and_si512(
                    _mm512_or_si512(_mm512_srlv_epi32(wb, _mm512_loadu_si512((const void*) T.sr[r])),
                                    _mm512_sllv_epi32(wa, _mm512_loadu_si512((const void*) T.sl[r]))),
                    low16);
                const __m512i bs =
                    _mm512_madd_epi16(_mm512_maddubs_epi16(_mm512_mullo_epi32(st, mult), ones8), ones16);
                _mm512_store_ps(wr + r * 128 + t * 16, _mm512_cvtepi32_ps(bs));
            }
        }
    }
}

// acc[t][0..128) += sum_{k < kr} x[t][k] wc[k][..] for TB activations
template <int TB>
EXL3_AVX512 void gemm_tile(const float* wc, int kr, const float* const* x, float* const* acc) {
    for (int cb = 0; cb < 128; cb += 64) {
        __m512 a[TB][4];
        for (int t = 0; t < TB; ++t)
            for (int j = 0; j < 4; ++j) a[t][j] = _mm512_loadu_ps(acc[t] + cb + j * 16);
        const float* w = wc + cb;
        for (int k = 0; k < kr; ++k, w += 128) {
            const __m512 w0 = _mm512_load_ps(w), w1 = _mm512_load_ps(w + 16), w2 = _mm512_load_ps(w + 32),
                         w3 = _mm512_load_ps(w + 48);
#pragma GCC unroll 8
            for (int t = 0; t < TB; ++t) {
                const __m512 xv = _mm512_set1_ps(x[t][k]);
                a[t][0] = _mm512_fmadd_ps(xv, w0, a[t][0]);
                a[t][1] = _mm512_fmadd_ps(xv, w1, a[t][1]);
                a[t][2] = _mm512_fmadd_ps(xv, w2, a[t][2]);
                a[t][3] = _mm512_fmadd_ps(xv, w3, a[t][3]);
            }
        }
        for (int t = 0; t < TB; ++t)
            for (int j = 0; j < 4; ++j) _mm512_storeu_ps(acc[t] + cb + j * 16, a[t][j]);
    }
}

// nb bands from tile column tc0 (acc[t]: the activation's 128 nb outputs from there): each chunk of the inputs is
// taken for every band in turn, so the activations' chunk (n x kGemmK floats) comes from L2 for the nb bands - an
// expert's rows are its own tokens, and with a job a band they came from DRAM again for every band (~16x the
// expert's weights in bytes)
template <int K>
EXL3_AVX512 void band_gemm_avx512(const Exl3Part& m, const float* const* xh, int n, int tc0, int nb,
                                  float* const* acc, float* wc) {
    constexpr int KS = kGemmK / 16;
    constexpr int TB = 6;   // 24 sums + 4 weights + the activation: 29 of the 32 registers
    for (int t = 0; t < n; ++t) std::memset(acc[t], 0, (size_t) nb * 128 * sizeof(float));
    std::vector<float*> ab((size_t) n);
    for (int ks0 = 0; ks0 < m.k / 16; ks0 += KS) {
        const int ks1 = std::min(m.k / 16, ks0 + KS), kr = (ks1 - ks0) * 16;
        for (int bb = 0; bb < nb; ++bb) {
            decode_chunk<K>(m, tc0 + 8 * bb, ks0, ks1, wc);
            for (int t = 0; t < n; ++t) ab[(size_t) t] = acc[t] + bb * 128;
            float* const* ap = ab.data();
            int t = 0;
            for (; t + TB <= n; t += TB) {
                const float* xp[TB];
                for (int i = 0; i < TB; ++i) xp[i] = xh[t + i] + ks0 * 16;
                gemm_tile<TB>(wc, kr, xp, ap + t);
            }
            if (t < n) {   // the last 1..TB-1 activations in one pass too
                const float* xp[TB];
                for (int i = 0; i < n - t; ++i) xp[i] = xh[t + i] + ks0 * 16;
                switch (n - t) {
                    case 1: gemm_tile<1>(wc, kr, xp, ap + t); break;
                    case 2: gemm_tile<2>(wc, kr, xp, ap + t); break;
                    case 3: gemm_tile<3>(wc, kr, xp, ap + t); break;
                    case 4: gemm_tile<4>(wc, kr, xp, ap + t); break;
                    default: gemm_tile<5>(wc, kr, xp, ap + t); break;
                }
            }
        }
    }
}

// had128 in 8 registers: every butterfly takes the operands the scalar loop takes (a lane's partner by a permute; the
// lower lane a + b, the upper b's lane t - r = a - b), so the result is had128's bit for bit
EXL3_AVX512 inline __m512 bfly(__m512 r, __m512 t, __mmask16 upper) {
    return _mm512_mask_sub_ps(_mm512_add_ps(r, t), upper, t, r);
}
EXL3_AVX512 void had128_avx512(float* v) {
    __m512 r[8];
    for (int i = 0; i < 8; ++i) r[i] = _mm512_loadu_ps(v + 16 * i);
    for (int i = 0; i < 8; ++i) {
        r[i] = bfly(r[i], _mm512_permute_ps(r[i], 0b10110001), (__mmask16) 0xAAAA);              // h = 1
        r[i] = bfly(r[i], _mm512_permute_ps(r[i], 0b01001110), (__mmask16) 0xCCCC);              // h = 2
        r[i] = bfly(r[i], _mm512_shuffle_f32x4(r[i], r[i], 0b10110001), (__mmask16) 0xF0F0);   // h = 4
        r[i] = bfly(r[i], _mm512_shuffle_f32x4(r[i], r[i], 0b01001110), (__mmask16) 0xFF00);   // h = 8
    }
    for (int h = 1; h < 8; h <<= 1)   // h = 16, 32, 64: whole registers
        for (int i = 0; i < 8; i += 2 * h)
            for (int j = i; j < i + h; ++j) {
                const __m512 a = r[j], b = r[j + h];
                r[j] = _mm512_add_ps(a, b);
                r[j + h] = _mm512_sub_ps(a, b);
            }
    const __m512 s = _mm512_set1_ps(0.088388347648f);
    for (int i = 0; i < 8; ++i) _mm512_storeu_ps(v + 16 * i, _mm512_mul_ps(r[i], s));
}

// exl3_cpu_prepare_block / exl3_cpu_band_out with the FP16 scales converted 16 at a time (vcvtph2ps: exact, as
// f32_from_f16) and the block's sum in eight double lanes
EXL3_AVX512 float prepare_block_avx512(const uint16_t* suh, const float* x, float* xh) {
    for (int i = 0; i < 128; i += 16) {
        const __m512 sc = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (suh + i)));
        _mm512_storeu_ps(xh + i, _mm512_mul_ps(_mm512_loadu_ps(x + i), sc));
    }
    had128_avx512(xh);
    __m512d s0 = _mm512_setzero_pd(), s1 = _mm512_setzero_pd();
    for (int i = 0; i < 128; i += 16) {
        s0 = _mm512_add_pd(s0, _mm512_cvtps_pd(_mm256_loadu_ps(xh + i)));
        s1 = _mm512_add_pd(s1, _mm512_cvtps_pd(_mm256_loadu_ps(xh + i + 8)));
    }
    return (float) _mm512_reduce_add_pd(_mm512_add_pd(s0, s1));
}
EXL3_AVX512 void band_out_avx512(const float* acc, float ki, float kz, const uint16_t* svh, float* y) {
    const __m512 vk = _mm512_set1_ps(ki), vz = _mm512_set1_ps(kz);
    for (int i = 0; i < 128; i += 16) _mm512_storeu_ps(y + i, _mm512_fmadd_ps(_mm512_loadu_ps(acc + i), vk, vz));
    had128_avx512(y);
    for (int i = 0; i < 128; i += 16) {
        const __m512 sc = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (svh + i)));
        _mm512_storeu_ps(y + i, _mm512_mul_ps(_mm512_loadu_ps(y + i), sc));
    }
}

// ---------------------------------------------------------------- AVX2
// The AVX-512 tier on 8-lane registers, a tile row's 16 columns as two halves of 8 (after exllamav3's AVX2 tier).
// Column c' of a half reads the tile's words K c' + base (mod 8K) - its 8 words lie in one residue class of the word
// index mod K, and its funnel shift is the same for all 8 (the layout's row-shift invariant).  So each tile's K
// registers are regrouped once into its K classes, and every half's gather is then ONE vpermd of a class register
// (none where the lanes are in order already) instead of a permute and a blend per register the half reads: odd K
// takes each class lane from exactly one register (blends only), K 2 / 4 / 8 are shuffles / a 4x4 / an 8x8 transpose,
// K 6 a permute and a blend per register.  exllamav3 gets the same single permute from a planar order repacked at load
// (`_copy_repacked`); here the regrouping costs a few register ops a tile and the RAM tier and the GPUs keep the
// tiles as they are.  The shifts are immediates, and the byte sum and the FMA are the AVX-512 tier's, lane for lane
// in the same (slice, row) order: the two tiers give the same bits.

// the lane of tile word w in its class register (class w % K; avx2_classes builds them)
constexpr int avx2_slot(int K, int w) {
    return (K & 1)    ? (w & 7)
           : K == 2   ? (w & 4) | ((w >> 2) & 2) | ((w >> 1) & 1)
           : K == 4   ? (w >> 3) + 4 * ((w >> 2) & 1)
           : K == 8   ? (w >> 3)
                      : w / K;
}

// per row and half: the right shift, whether the 16-bit windows sit in one word (no second gather), and for the
// window's first and last word the class register and the vpermd lanes that put its 8 words in column order
template <int K>
struct Avx2Tab {
    int sh[16][2] = {};
    bool one[16][2] = {};
    int cls[16][2][2] = {};
    bool ident[16][2][2] = {};
    int32_t idx[16][2][2][8] = {};
    bool ok = true;   // the invariants above hold for every row and half
};
template <int K>
constexpr Avx2Tab<K> make_avx2tab() {
    Avx2Tab<K> t{};
    const RowTab<K> rt = make_rowtab<K>();
    for (int r = 0; r < 16; ++r)
        for (int h = 0; h < 2; ++h) {
            const int c0 = 8 * h;
            t.sh[r][h] = rt.sr[r][c0];
            t.one[r][h] = true;
            for (int c = c0; c < c0 + 8; ++c) {
                if (rt.sr[r][c] != t.sh[r][h]) t.ok = false;
                if (rt.i0[r][c] != rt.i1[r][c]) t.one[r][h] = false;
            }
            if (t.one[r][h] != (t.sh[r][h] <= 16)) t.ok = false;
            for (int wd = 0; wd < 2; ++wd) {
                const auto& ix = wd == 0 ? rt.i0 : rt.i1;
                t.cls[r][h][wd] = ix[r][c0] % K;
                t.ident[r][h][wd] = true;
                for (int c = 0; c < 8; ++c) {
                    if (ix[r][c0 + c] % K != t.cls[r][h][wd]) t.ok = false;
                    t.idx[r][h][wd][c] = avx2_slot(K, ix[r][c0 + c]);
                    if (t.idx[r][h][wd][c] != c) t.ident[r][h][wd] = false;
                }
            }
        }
    return t;
}
template <int K>
inline constexpr Avx2Tab<K> kAvx2Tab = make_avx2tab<K>();

// odd K: lane l of class rho is register j's, the one j with (8 j + l) % K == rho
template <int K, int RHO, int J>
constexpr int avx2_odd_mask() {
    int m = 0;
    for (int l = 0; l < 8; ++l)
        if ((8 * J + l) % K == RHO) m |= 1 << l;
    return m;
}
template <int K, int RHO, int J = 1>
EXL3_AVX2_INLINE __m256i avx2_odd_class(const __m256i (&R)[K], __m256i acc) {
    if constexpr (J == K) {
        return acc;
    } else {
        constexpr int m = avx2_odd_mask<K, RHO, J>();   // (clang's _mm256_blend_epi32 is a macro: no template commas)
        return avx2_odd_class<K, RHO, J + 1>(R, _mm256_blend_epi32(acc, R[J], m));
    }
}
template <int K, int... RHO>
EXL3_AVX2_INLINE void avx2_odd_classes(const __m256i (&R)[K], __m256i (&C)[K], std::integer_sequence<int, RHO...>) {
    ((C[RHO] = avx2_odd_class<K, RHO>(R, R[0])), ...);
}
// any K (used for K 6): class rho's word rho + K s at lane s - each register's words permuted there, then blended
template <int K, int RHO, int J>
constexpr int avx2_gen_mask() {
    int m = 0;
    for (int s = 0; s < 8; ++s)
        if ((RHO + K * s) / 8 == J) m |= 1 << s;
    return m;
}
template <int K, int RHO, int J, int S>
constexpr int avx2_gen_lane() {
    return (RHO + K * S) / 8 == J ? (RHO + K * S) % 8 : 0;
}
template <int K, int RHO, int J = 0>
EXL3_AVX2_INLINE __m256i avx2_gen_class(const __m256i (&R)[K], __m256i acc) {
    if constexpr (J == K) {
        return acc;
    } else if constexpr (avx2_gen_mask<K, RHO, J>() == 0) {
        return avx2_gen_class<K, RHO, J + 1>(R, acc);
    } else {
        const __m256i p = _mm256_permutevar8x32_epi32(
            R[J], _mm256_setr_epi32(avx2_gen_lane<K, RHO, J, 0>(), avx2_gen_lane<K, RHO, J, 1>(),
                                    avx2_gen_lane<K, RHO, J, 2>(), avx2_gen_lane<K, RHO, J, 3>(),
                                    avx2_gen_lane<K, RHO, J, 4>(), avx2_gen_lane<K, RHO, J, 5>(),
                                    avx2_gen_lane<K, RHO, J, 6>(), avx2_gen_lane<K, RHO, J, 7>()));
        constexpr int m = avx2_gen_mask<K, RHO, J>();   // (a macro under clang, as above)
        return avx2_gen_class<K, RHO, J + 1>(R, _mm256_blend_epi32(acc, p, m));
    }
}
template <int K, int... RHO>
EXL3_AVX2_INLINE void avx2_gen_classes(const __m256i (&R)[K], __m256i (&C)[K], std::integer_sequence<int, RHO...>) {
    ((C[RHO] = avx2_gen_class<K, RHO>(R, _mm256_setzero_si256())), ...);
}
// a 4x4 transpose in each 128-bit half: o[q] = lanes q and q + 4 of r0..r3
EXL3_AVX2_INLINE void avx2_t4(__m256i r0, __m256i r1, __m256i r2, __m256i r3, __m256i (&o)[4]) {
    const __m256i t0 = _mm256_unpacklo_epi32(r0, r1), t1 = _mm256_unpackhi_epi32(r0, r1);
    const __m256i t2 = _mm256_unpacklo_epi32(r2, r3), t3 = _mm256_unpackhi_epi32(r2, r3);
    o[0] = _mm256_unpacklo_epi64(t0, t2);
    o[1] = _mm256_unpackhi_epi64(t0, t2);
    o[2] = _mm256_unpacklo_epi64(t1, t3);
    o[3] = _mm256_unpackhi_epi64(t1, t3);
}
// a tile's 8K words as its K class registers (word w in C[w % K] at lane avx2_slot(K, w))
template <int K>
EXL3_AVX2_INLINE void avx2_classes(const uint32_t* w, __m256i (&C)[K]) {
    __m256i R[K];
    for (int j = 0; j < K; ++j) R[j] = _mm256_loadu_si256((const __m256i*) (w + 8 * j));
    if constexpr (K == 1) {
        C[0] = R[0];
    } else if constexpr (K % 2 == 1) {
        avx2_odd_classes<K>(R, C, std::make_integer_sequence<int, K>{});
    } else if constexpr (K == 2) {
        const __m256 a = _mm256_castsi256_ps(R[0]), b = _mm256_castsi256_ps(R[1]);
        C[0] = _mm256_castps_si256(_mm256_shuffle_ps(a, b, 0x88));   // words 0 2 8 10 | 4 6 12 14
        C[1] = _mm256_castps_si256(_mm256_shuffle_ps(a, b, 0xDD));   // words 1 3 9 11 | 5 7 13 15
    } else if constexpr (K == 4) {
        avx2_t4(R[0], R[1], R[2], R[3], C);
    } else if constexpr (K == 8) {
        __m256i A[4], B[4];
        avx2_t4(R[0], R[1], R[2], R[3], A);
        avx2_t4(R[4], R[5], R[6], R[7], B);
        for (int q = 0; q < 4; ++q) {
            C[q] = _mm256_permute2x128_si256(A[q], B[q], 0x20);
            C[q + 4] = _mm256_permute2x128_si256(A[q], B[q], 0x31);
        }
    } else {
        avx2_gen_classes<K>(R, C, std::make_integer_sequence<int, K>{});
    }
}

// the 8 words (first or last of each window) of row R's half H, in column order
template <int K, int R, int H, int WD>
EXL3_AVX2_INLINE __m256i avx2_gather(const __m256i (&C)[K]) {
    constexpr const Avx2Tab<K>& T = kAvx2Tab<K>;
    if constexpr (T.ident[R][H][WD]) {
        return C[T.cls[R][H][WD]];
    } else {
        return _mm256_permutevar8x32_epi32(C[T.cls[R][H][WD]],
                                           _mm256_loadu_si256((const __m256i*) T.idx[R][H][WD]));
    }
}

struct Avx2Consts {
    __m256i mult, ones8, ones16, low16;
};
EXL3_AVX2_INLINE Avx2Consts avx2_consts() {
    return {_mm256_set1_epi32((int) kMul1), _mm256_set1_epi8(1), _mm256_set1_epi16(1), _mm256_set1_epi32(0xffff)};
}

// row R's half H: its 8 states, and their codebook byte sums as F32
template <int K, int R, int H>
EXL3_AVX2_INLINE __m256 avx2_sums(const __m256i (&C)[K], const Avx2Consts& k) {
    constexpr const Avx2Tab<K>& T = kAvx2Tab<K>;
    constexpr int sh = T.sh[R][H];
    const __m256i b = avx2_gather<K, R, H, 1>(C);
    __m256i st;
    if constexpr (T.one[R][H] && sh == 16) {
        st = _mm256_srli_epi32(b, 16);
    } else if constexpr (T.one[R][H]) {
        st = _mm256_and_si256(_mm256_srli_epi32(b, sh), k.low16);
    } else {
        const __m256i a = avx2_gather<K, R, H, 0>(C);
        st = _mm256_and_si256(_mm256_or_si256(_mm256_srli_epi32(b, sh), _mm256_slli_epi32(a, 32 - sh)), k.low16);
    }
    return _mm256_cvtepi32_ps(
        _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_mullo_epi32(st, k.mult), k.ones8), k.ones16));
}

template <int K, int R>
EXL3_AVX2_INLINE void avx2_rows(const __m256i (&C)[K], const float* xk, __m256& lo, __m256& hi, const Avx2Consts& k) {
    if constexpr (R < 16) {
        const __m256 xv = _mm256_broadcast_ss(xk + R);
        lo = _mm256_fmadd_ps(avx2_sums<K, R, 0>(C, k), xv, lo);
        hi = _mm256_fmadd_ps(avx2_sums<K, R, 1>(C, k), xv, hi);
        avx2_rows<K, R + 1>(C, xk, lo, hi, k);
    }
}

template <int K>
EXL3_AVX2 void band_avx2(const Exl3Part& m, const float* xh, int tc0, int ks0, int ks1, float* out /* 128 */) {
    static_assert(kAvx2Tab<K>.ok, "EXL3 tile layout: a half row's words must share a residue class and a shift");
    constexpr int W = 8 * K;
    const int ld = m.n / 16;
    const size_t row_words = (size_t) ld * W;
    const Avx2Consts k = avx2_consts();
    __m256 acc[8][2];   // the band's 8 tiles (16 registers: in memory, a tile's two taken at a time)
    for (int t = 0; t < 8; ++t) acc[t][0] = acc[t][1] = _mm256_setzero_ps();
    const uint32_t* row = m.trellis + (size_t) ks0 * row_words + (size_t) tc0 * W;
    for (int ks = ks0; ks < ks1; ++ks, row += row_words) {
        // the band's 8 tiles of this slice are contiguous; the next slices' are one row of tiles on
        const char* pf = (const char*) (row + 4 * row_words);
        for (int l = 0; l < 8 * W * 4; l += 64) _mm_prefetch(pf + l, _MM_HINT_T0);
        const float* xk = xh + ks * 16;
        for (int t = 0; t < 8; ++t) {
            __m256i C[K];
            avx2_classes<K>(row + t * W, C);
            __m256 lo = acc[t][0], hi = acc[t][1];
            avx2_rows<K, 0>(C, xk, lo, hi, k);
            acc[t][0] = lo;
            acc[t][1] = hi;
        }
    }
    for (int t = 0; t < 8; ++t) {
        _mm256_storeu_ps(out + t * 16, acc[t][0]);
        _mm256_storeu_ps(out + t * 16 + 8, acc[t][1]);
    }
}

// band_avx2 for N activations at once, as band_avx512_n: a half row's states summed once, then one FMA per
// activation - activation i's FMAs are band_avx2's for it alone, in its order, so out[i] is the same bit for bit.
// Up to 3 activations both halves of a tile at once (2N accumulators, one broadcast a row and activation); more, a
// tile's halves one after the other (N accumulators, not 2N: 16 registers) - 2 activations 1.55 -> 1.34 ms an
// expert-token on one Zen 3 core, 3: 1.18 -> 1.11
template <int N, int... I>
EXL3_AVX2_INLINE void avx2_fma_n(__m256 f, const float* const (&xk)[N], int r, __m256 (&a)[N],
                                 std::integer_sequence<int, I...>) {
    ((a[I] = _mm256_fmadd_ps(f, _mm256_broadcast_ss(xk[I] + r), a[I])), ...);
}
template <int K, int N, int H, int R>
EXL3_AVX2_INLINE void avx2_rows_n(const __m256i (&C)[K], const float* const (&xk)[N], __m256 (&a)[N],
                                  const Avx2Consts& k) {
    if constexpr (R < 16) {
        avx2_fma_n<N>(avx2_sums<K, R, H>(C, k), xk, R, a, std::make_integer_sequence<int, N>{});
        avx2_rows_n<K, N, H, R + 1>(C, xk, a, k);
    }
}
template <int N, int... I>
EXL3_AVX2_INLINE void avx2_fma_n2(__m256 f0, __m256 f1, const float* const (&xk)[N], int r, __m256 (&lo)[N],
                                  __m256 (&hi)[N], std::integer_sequence<int, I...>) {
    ((lo[I] = _mm256_fmadd_ps(f0, _mm256_broadcast_ss(xk[I] + r), lo[I]),
      hi[I] = _mm256_fmadd_ps(f1, _mm256_broadcast_ss(xk[I] + r), hi[I])),
     ...);
}
template <int K, int N, int R>
EXL3_AVX2_INLINE void avx2_rows_n2(const __m256i (&C)[K], const float* const (&xk)[N], __m256 (&lo)[N],
                                   __m256 (&hi)[N], const Avx2Consts& k) {
    if constexpr (R < 16) {
        avx2_fma_n2<N>(avx2_sums<K, R, 0>(C, k), avx2_sums<K, R, 1>(C, k), xk, R, lo, hi,
                       std::make_integer_sequence<int, N>{});
        avx2_rows_n2<K, N, R + 1>(C, xk, lo, hi, k);
    }
}
template <int K, int N>
EXL3_AVX2 void band_avx2_n(const Exl3Part& m, const float* const* xh, int tc0, int ks0, int ks1,
                           float* const* out /* N x 128 */) {
    static_assert(kAvx2Tab<K>.ok, "EXL3 tile layout: a half row's words must share a residue class and a shift");
    constexpr int W = 8 * K;
    const int ld = m.n / 16;
    const size_t row_words = (size_t) ld * W;
    const Avx2Consts k = avx2_consts();
    __m256 acc[8][2][N];
    for (int t = 0; t < 8; ++t)
        for (int h = 0; h < 2; ++h)
            for (int i = 0; i < N; ++i) acc[t][h][i] = _mm256_setzero_ps();
    const uint32_t* row = m.trellis + (size_t) ks0 * row_words + (size_t) tc0 * W;
    for (int ks = ks0; ks < ks1; ++ks, row += row_words) {
        const char* pf = (const char*) (row + 4 * row_words);
        for (int l = 0; l < 8 * W * 4; l += 64) _mm_prefetch(pf + l, _MM_HINT_T0);
        const float* xk[N];
        for (int i = 0; i < N; ++i) xk[i] = xh[i] + ks * 16;
        for (int t = 0; t < 8; ++t) {
            __m256i C[K];
            avx2_classes<K>(row + t * W, C);
            if constexpr (N <= 3) {
                __m256 lo[N], hi[N];
                for (int i = 0; i < N; ++i) lo[i] = acc[t][0][i], hi[i] = acc[t][1][i];
                avx2_rows_n2<K, N, 0>(C, xk, lo, hi, k);
                for (int i = 0; i < N; ++i) acc[t][0][i] = lo[i], acc[t][1][i] = hi[i];
            } else {
                __m256 a[N];
                for (int i = 0; i < N; ++i) a[i] = acc[t][0][i];
                avx2_rows_n<K, N, 0, 0>(C, xk, a, k);
                for (int i = 0; i < N; ++i) acc[t][0][i] = a[i], a[i] = acc[t][1][i];
                avx2_rows_n<K, N, 1, 0>(C, xk, a, k);
                for (int i = 0; i < N; ++i) acc[t][1][i] = a[i];
            }
        }
    }
    for (int i = 0; i < N; ++i)
        for (int t = 0; t < 8; ++t) {
            _mm256_storeu_ps(out[i] + t * 16, acc[t][0][i]);
            _mm256_storeu_ps(out[i] + t * 16 + 8, acc[t][1][i]);
        }
}

// #188's split-K piece (rows_avx512_n) on the AVX2 tier: slices [ks0, ks1) of EVERY column for N activations, the
// tile rows in memory order, a tile's sums in acc between slices; each row's decode and FMAs are band_avx2's, so acc[i]
// is the band kernel's for its columns and xh[i] alone, bit for bit (and rows_avx512_n's)
template <int K, int N>
EXL3_AVX2 void rows_avx2_n(const Exl3Part& m, const float* const* xh, int ks0, int ks1, int t0, int t1,
                           float* const* acc) {
    static_assert(kAvx2Tab<K>.ok, "EXL3 tile layout: a half row's words must share a residue class and a shift");
    constexpr int W = 8 * K;
    const int ld = m.n / 16;
    const size_t row_words = (size_t) ld * W;
    const Avx2Consts k = avx2_consts();
    for (int i = 0; i < N; ++i)
        for (int c = 16 * t0; c < 16 * t1; c += 8) _mm256_storeu_ps(acc[i] + c, _mm256_setzero_ps());
    const int span = t1 - t0;
    const uint32_t* row = m.trellis + (size_t) ks0 * row_words;
    for (int ks = ks0; ks < ks1; ++ks, row += row_words) {
        const float* xk[N];
        for (int i = 0; i < N; ++i) xk[i] = xh[i] + ks * 16;
        for (int t = t0; t < t1; ++t) {
            const uint32_t* w = row + (size_t) t * W;
            const uint32_t* pf = t + 16 < t1 || span == ld ? w + 16 * W : row + row_words + (size_t) (t + 16 - span) * W;
            for (int l = 0; l < W * 4; l += 64) _mm_prefetch((const char*) pf + l, _MM_HINT_T0);
            __m256i C[K];
            avx2_classes<K>(w, C);
            if constexpr (N == 1) {
                __m256 lo = _mm256_loadu_ps(acc[0] + t * 16), hi = _mm256_loadu_ps(acc[0] + t * 16 + 8);
                avx2_rows<K, 0>(C, xk[0], lo, hi, k);
                _mm256_storeu_ps(acc[0] + t * 16, lo);
                _mm256_storeu_ps(acc[0] + t * 16 + 8, hi);
            } else if constexpr (N <= 3) {
                __m256 lo[N], hi[N];
                for (int i = 0; i < N; ++i)
                    lo[i] = _mm256_loadu_ps(acc[i] + t * 16), hi[i] = _mm256_loadu_ps(acc[i] + t * 16 + 8);
                avx2_rows_n2<K, N, 0>(C, xk, lo, hi, k);
                for (int i = 0; i < N; ++i)
                    _mm256_storeu_ps(acc[i] + t * 16, lo[i]), _mm256_storeu_ps(acc[i] + t * 16 + 8, hi[i]);
            } else {
                __m256 a[N];
                for (int i = 0; i < N; ++i) a[i] = _mm256_loadu_ps(acc[i] + t * 16);
                avx2_rows_n<K, N, 0, 0>(C, xk, a, k);
                for (int i = 0; i < N; ++i) _mm256_storeu_ps(acc[i] + t * 16, a[i]), a[i] = _mm256_loadu_ps(acc[i] + t * 16 + 8);
                avx2_rows_n<K, N, 1, 0>(C, xk, a, k);
                for (int i = 0; i < N; ++i) _mm256_storeu_ps(acc[i] + t * 16 + 8, a[i]);
            }
        }
    }
}

// #188's band GEMM (decode_chunk + gemm_tile) on the AVX2 tier: a chunk of a band's byte sums decoded once into F32,
// then TB activations x 32 columns a pass, the sums in registers (12 of the 16 with TB 3) - each column's FMAs in the
// band kernel's order (k ascending; x w + a as band_avx2's s x + a), so the same bits as gemm_tile on AVX-512
template <int K, int R>
EXL3_AVX2_INLINE void avx2_decode_rows(const __m256i (&C)[K], float* wr, int t, const Avx2Consts& k) {
    if constexpr (R < 16) {
        _mm256_store_ps(wr + R * 128 + t * 16, avx2_sums<K, R, 0>(C, k));
        _mm256_store_ps(wr + R * 128 + t * 16 + 8, avx2_sums<K, R, 1>(C, k));
        avx2_decode_rows<K, R + 1>(C, wr, t, k);
    }
}
template <int K>
EXL3_AVX2 void decode_chunk_avx2(const Exl3Part& m, int tc0, int ks0, int ks1, float* wc /* (ks1 - ks0) 16 x 128 */) {
    constexpr int W = 8 * K;
    const size_t row_words = (size_t) (m.n / 16) * W;
    const Avx2Consts k = avx2_consts();
    const uint32_t* row = m.trellis + (size_t) ks0 * row_words + (size_t) tc0 * W;
    for (int ks = ks0; ks < ks1; ++ks, row += row_words) {
        const char* pf = (const char*) (row + 4 * row_words);
        for (int l = 0; l < 8 * W * 4; l += 64) _mm_prefetch(pf + l, _MM_HINT_T0);
        float* wr = wc + (size_t) (ks - ks0) * 16 * 128;
        for (int t = 0; t < 8; ++t) {
            __m256i C[K];
            avx2_classes<K>(row + t * W, C);
            avx2_decode_rows<K, 0>(C, wr, t, k);
        }
    }
}
// acc[t][0..128) += sum_{k < kr} x[t][k] wc[k][..] for TB activations
template <int TB>
EXL3_AVX2 void gemm_tile_avx2(const float* wc, int kr, const float* const* x, float* const* acc) {
    for (int cb = 0; cb < 128; cb += 32) {
        __m256 a[TB][4];
        for (int t = 0; t < TB; ++t)
            for (int j = 0; j < 4; ++j) a[t][j] = _mm256_loadu_ps(acc[t] + cb + j * 8);
        const float* w = wc + cb;
        for (int q = 0; q < kr; ++q, w += 128) {
            const __m256 w0 = _mm256_load_ps(w), w1 = _mm256_load_ps(w + 8), w2 = _mm256_load_ps(w + 16),
                         w3 = _mm256_load_ps(w + 24);
            for (int t = 0; t < TB; ++t) {
                const __m256 xv = _mm256_broadcast_ss(x[t] + q);
                a[t][0] = _mm256_fmadd_ps(xv, w0, a[t][0]);
                a[t][1] = _mm256_fmadd_ps(xv, w1, a[t][1]);
                a[t][2] = _mm256_fmadd_ps(xv, w2, a[t][2]);
                a[t][3] = _mm256_fmadd_ps(xv, w3, a[t][3]);
            }
        }
        for (int t = 0; t < TB; ++t)
            for (int j = 0; j < 4; ++j) _mm256_storeu_ps(acc[t] + cb + j * 8, a[t][j]);
    }
}
template <int K>
EXL3_AVX2 void band_gemm_avx2(const Exl3Part& m, const float* const* xh, int n, int tc0, int nb, float* const* acc,
                              float* wc) {
    constexpr int KS = kGemmK / 16;
    constexpr int TB = 3;   // 12 sums + 4 weights of the 16 registers
    for (int t = 0; t < n; ++t) std::memset(acc[t], 0, (size_t) nb * 128 * sizeof(float));
    std::vector<float*> ab((size_t) n);
    for (int ks0 = 0; ks0 < m.k / 16; ks0 += KS) {
        const int ks1 = std::min(m.k / 16, ks0 + KS), kr = (ks1 - ks0) * 16;
        for (int bb = 0; bb < nb; ++bb) {
            decode_chunk_avx2<K>(m, tc0 + 8 * bb, ks0, ks1, wc);
            for (int t = 0; t < n; ++t) ab[(size_t) t] = acc[t] + bb * 128;
            float* const* ap = ab.data();
            int t = 0;
            for (; t + TB <= n; t += TB) {
                const float* xp[TB];
                for (int i = 0; i < TB; ++i) xp[i] = xh[t + i] + ks0 * 16;
                gemm_tile_avx2<TB>(wc, kr, xp, ap + t);
            }
            if (t < n) {
                const float* xp[TB];
                for (int i = 0; i < n - t; ++i) xp[i] = xh[t + i] + ks0 * 16;
                if (n - t == 1) gemm_tile_avx2<1>(wc, kr, xp, ap + t);
                else gemm_tile_avx2<2>(wc, kr, xp, ap + t);
            }
        }
    }
}

// had128 on 16 registers (exllamav3's hadamard_128_avx2): every butterfly takes the scalar loop's operands (a lane's
// partner by a permute; the lower lane a + b, the upper t - r = a - b), so the result is had128's bit for bit
EXL3_AVX2 void had128_avx2(float* v) {
    __m256 r[16];
    for (int i = 0; i < 16; ++i) r[i] = _mm256_loadu_ps(v + 8 * i);
    for (int i = 0; i < 16; ++i) {
        __m256 t = _mm256_permute_ps(r[i], 0b10110001);   // h = 1
        r[i] = _mm256_blend_ps(_mm256_add_ps(r[i], t), _mm256_sub_ps(t, r[i]), 0b10101010);
        t = _mm256_permute_ps(r[i], 0b01001110);          // h = 2
        r[i] = _mm256_blend_ps(_mm256_add_ps(r[i], t), _mm256_sub_ps(t, r[i]), 0b11001100);
        t = _mm256_permute2f128_ps(r[i], r[i], 0x01);     // h = 4
        r[i] = _mm256_blend_ps(_mm256_add_ps(r[i], t), _mm256_sub_ps(t, r[i]), 0b11110000);
    }
    for (int h = 1; h < 16; h <<= 1)   // h = 8 .. 64: whole registers
        for (int i = 0; i < 16; i += 2 * h)
            for (int j = i; j < i + h; ++j) {
                const __m256 a = r[j], b = r[j + h];
                r[j] = _mm256_add_ps(a, b);
                r[j + h] = _mm256_sub_ps(a, b);
            }
    const __m256 s = _mm256_set1_ps(0.088388347648f);
    for (int i = 0; i < 16; ++i) _mm256_storeu_ps(v + 8 * i, _mm256_mul_ps(r[i], s));
}

// prepare_block_avx512 / band_out_avx512 on the AVX2 tier, the same operations in the same order: the FP16 scales by
// vcvtph2ps (exact), the block's sum in the same eight double lanes (lanes 0-7 of the 16-float steps' first half, 8-15
// of the second, added) and reduced as GCC's _mm512_reduce_add_pd does (upper four + lower four, then the two pairs,
// then the last two), band_out's FMA - the same bits as the AVX-512 tier
EXL3_AVX2 float prepare_block_avx2(const uint16_t* suh, const float* x, float* xh) {
    for (int i = 0; i < 128; i += 8) {
        const __m256 sc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (suh + i)));
        _mm256_storeu_ps(xh + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), sc));
    }
    had128_avx2(xh);
    __m256d s0a = _mm256_setzero_pd(), s0b = _mm256_setzero_pd(), s1a = _mm256_setzero_pd(), s1b = _mm256_setzero_pd();
    for (int i = 0; i < 128; i += 16) {
        s0a = _mm256_add_pd(s0a, _mm256_cvtps_pd(_mm_loadu_ps(xh + i)));
        s0b = _mm256_add_pd(s0b, _mm256_cvtps_pd(_mm_loadu_ps(xh + i + 4)));
        s1a = _mm256_add_pd(s1a, _mm256_cvtps_pd(_mm_loadu_ps(xh + i + 8)));
        s1b = _mm256_add_pd(s1b, _mm256_cvtps_pd(_mm_loadu_ps(xh + i + 12)));
    }
    const __m256d lo = _mm256_add_pd(s0a, s1a), hi = _mm256_add_pd(s0b, s1b);   // the 8 lanes of s0 + s1
    const __m256d t3 = _mm256_add_pd(hi, lo);
    const __m128d t6 = _mm_add_pd(_mm256_extractf128_pd(t3, 1), _mm256_castpd256_pd128(t3));
    return (float) (_mm_cvtsd_f64(t6) + _mm_cvtsd_f64(_mm_unpackhi_pd(t6, t6)));
}
EXL3_AVX2 void band_out_avx2(const float* acc, float ki, float kz, const uint16_t* svh, float* y) {
    const __m256 vk = _mm256_set1_ps(ki), vz = _mm256_set1_ps(kz);
    for (int i = 0; i < 128; i += 8) _mm256_storeu_ps(y + i, _mm256_fmadd_ps(_mm256_loadu_ps(acc + i), vk, vz));
    had128_avx2(y);
    for (int i = 0; i < 128; i += 8) {
        const __m256 sc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (svh + i)));
        _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(y + i), sc));
    }
}

// AVX-512 F and BW with FMA, the OS saving the state (STRATA_FORCE_AVX2 is the tier choice's, below)
bool cpu_avx512_exl3() {
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
    if (r[0] < 7) return false;
    cpuid(1, 0);
    if (!((r[2] >> 27) & 1u) || !((r[2] >> 12) & 1u)) return false;   // OSXSAVE, FMA
#if defined(_MSC_VER)
    const unsigned long long xcr0 = _xgetbv(0);
#else
    unsigned lo = 0, hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
    if ((xcr0 & 0xE6) != 0xE6) return false;   // the OS saves the AVX-512 state
    cpuid(7, 0);
    return ((r[1] >> 16) & 1u) && ((r[1] >> 30) & 1u);   // AVX-512F, AVX-512BW
}
// F16C (the AVX2 tier's FP16 scale conversions; every AVX2 CPU has it, asked anyway)
bool cpu_f16c() {
    unsigned r[4] = {0, 0, 0, 0};
#if defined(_MSC_VER)
    int x[4];
    __cpuidex(x, 1, 0);
    for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
    __cpuid_count(1, 0, r[0], r[1], r[2], r[3]);
#endif
    return ((r[2] >> 29) & 1u) != 0;
}
#endif

// the tiers this CPU runs (bit t: tier t - 0 scalar, 1 AVX2 + FMA + F16C, 2 AVX-512 F/BW), and the one the lane takes:
// the widest, but AVX2 on an AVX-512 CPU with STRATA_FORCE_AVX2=1 (to compare the two vector tiers)
int tier_mask() {
    static const int mask = [] {
        int t = 1;
#if defined(EXL3_X86)
        if (cpu_avx2_ok() && cpu_f16c()) t |= 2;
        if (cpu_avx512_exl3()) t |= 4;
#endif
        return t;
    }();
    return mask;
}
int detect_tier() {
    const int t = tier_mask();
    const char* f = std::getenv("STRATA_FORCE_AVX2");
    if ((t & 4) && !(f != nullptr && f[0] == '1')) return 2;
    return (t & 2) ? 1 : 0;
}

template <int K>
void band_on(int tier, const Exl3Part& m, const float* xh, int tc0, int ks0, int ks1, float* acc) {
#if defined(EXL3_X86)
    if (tier == 2) {
        band_avx512<K>(m, xh, tc0, ks0, ks1, acc);
        return;
    }
    if (tier == 1) {
        band_avx2<K>(m, xh, tc0, ks0, ks1, acc);
        return;
    }
#endif
    band_scalar<K>(m, xh, tc0, ks0, ks1, acc);
}

// n activations (1..kExl3CpuMaxN) on one band
template <int K>
void band_on_n(int tier, const Exl3Part& m, const float* const* xh, int n, int tc0, int ks0, int ks1,
               float* const* acc) {
#if defined(EXL3_X86)
    if (tier == 2 && n > 1) {
        switch (n) {
            case 2: band_avx512_n<K, 2>(m, xh, tc0, ks0, ks1, acc); return;
            case 3: band_avx512_n<K, 3>(m, xh, tc0, ks0, ks1, acc); return;
            case 4: band_avx512_n<K, 4>(m, xh, tc0, ks0, ks1, acc); return;
            case 5: band_avx512_n<K, 5>(m, xh, tc0, ks0, ks1, acc); return;
            case 6: band_avx512_n<K, 6>(m, xh, tc0, ks0, ks1, acc); return;
            case 7: band_avx512_n<K, 7>(m, xh, tc0, ks0, ks1, acc); return;
            default: band_avx512_n<K, 8>(m, xh, tc0, ks0, ks1, acc); return;
        }
    }
    if (tier == 1 && n > 1) {
        switch (n) {
            case 2: band_avx2_n<K, 2>(m, xh, tc0, ks0, ks1, acc); return;
            case 3: band_avx2_n<K, 3>(m, xh, tc0, ks0, ks1, acc); return;
            case 4: band_avx2_n<K, 4>(m, xh, tc0, ks0, ks1, acc); return;
            case 5: band_avx2_n<K, 5>(m, xh, tc0, ks0, ks1, acc); return;
            case 6: band_avx2_n<K, 6>(m, xh, tc0, ks0, ks1, acc); return;
            case 7: band_avx2_n<K, 7>(m, xh, tc0, ks0, ks1, acc); return;
            default: band_avx2_n<K, 8>(m, xh, tc0, ks0, ks1, acc); return;
        }
    }
#endif
    for (int i = 0; i < n; ++i) band_on<K>(tier, m, xh[i], tc0, ks0, ks1, acc[i]);
}

// a split-K piece's tile columns [t0, t1) for n activations (1..kExl3CpuMaxN); the scalar loop: each band's
// band_scalar (t0, t1 multiples of 8 there)
template <int K>
void rows_on_n(int tier, const Exl3Part& m, const float* const* xh, int n, int ks0, int ks1, int t0, int t1,
               float* const* acc) {
#if defined(EXL3_X86)
    if (tier == 2) {
        switch (n) {
            case 1: rows_avx512_n<K, 1>(m, xh, ks0, ks1, t0, t1, acc); return;
            case 2: rows_avx512_n<K, 2>(m, xh, ks0, ks1, t0, t1, acc); return;
            case 3: rows_avx512_n<K, 3>(m, xh, ks0, ks1, t0, t1, acc); return;
            case 4: rows_avx512_n<K, 4>(m, xh, ks0, ks1, t0, t1, acc); return;
            case 5: rows_avx512_n<K, 5>(m, xh, ks0, ks1, t0, t1, acc); return;
            case 6: rows_avx512_n<K, 6>(m, xh, ks0, ks1, t0, t1, acc); return;
            case 7: rows_avx512_n<K, 7>(m, xh, ks0, ks1, t0, t1, acc); return;
            default: rows_avx512_n<K, 8>(m, xh, ks0, ks1, t0, t1, acc); return;
        }
    }
    if (tier == 1) {
        switch (n) {
            case 1: rows_avx2_n<K, 1>(m, xh, ks0, ks1, t0, t1, acc); return;
            case 2: rows_avx2_n<K, 2>(m, xh, ks0, ks1, t0, t1, acc); return;
            case 3: rows_avx2_n<K, 3>(m, xh, ks0, ks1, t0, t1, acc); return;
            case 4: rows_avx2_n<K, 4>(m, xh, ks0, ks1, t0, t1, acc); return;
            case 5: rows_avx2_n<K, 5>(m, xh, ks0, ks1, t0, t1, acc); return;
            case 6: rows_avx2_n<K, 6>(m, xh, ks0, ks1, t0, t1, acc); return;
            case 7: rows_avx2_n<K, 7>(m, xh, ks0, ks1, t0, t1, acc); return;
            default: rows_avx2_n<K, 8>(m, xh, ks0, ks1, t0, t1, acc); return;
        }
    }
#endif
    for (int i = 0; i < n; ++i)
        for (int c0 = 16 * t0; c0 < 16 * t1; c0 += 128) band_scalar<K>(m, xh[i], c0 / 16, ks0, ks1, acc[i] + c0);
}

void band_acc_tier(int tier, const Exl3Part& m, const float* xh, int c0, int ks0, int ks1, float* acc) {
    const int tc0 = c0 / 16;
    switch (m.K) {
        case 1: band_on<1>(tier, m, xh, tc0, ks0, ks1, acc); break;
        case 2: band_on<2>(tier, m, xh, tc0, ks0, ks1, acc); break;
        case 3: band_on<3>(tier, m, xh, tc0, ks0, ks1, acc); break;
        case 4: band_on<4>(tier, m, xh, tc0, ks0, ks1, acc); break;
        case 5: band_on<5>(tier, m, xh, tc0, ks0, ks1, acc); break;
        case 6: band_on<6>(tier, m, xh, tc0, ks0, ks1, acc); break;
        case 7: band_on<7>(tier, m, xh, tc0, ks0, ks1, acc); break;
        case 8: band_on<8>(tier, m, xh, tc0, ks0, ks1, acc); break;
        default:
            for (int i = 0; i < 128; ++i) acc[i] = 0.0f;
    }
}

void band_acc_n_tier(int tier, const Exl3Part& m, const float* const* xh, int n, int c0, int ks0, int ks1,
                     float* const* acc) {
    const int tc0 = c0 / 16;
    for (int i0 = 0; i0 < n; i0 += kExl3CpuMaxN) {
        const int c = std::min(kExl3CpuMaxN, n - i0);
        switch (m.K) {
            case 1: band_on_n<1>(tier, m, xh + i0, c, tc0, ks0, ks1, acc + i0); break;
            case 2: band_on_n<2>(tier, m, xh + i0, c, tc0, ks0, ks1, acc + i0); break;
            case 3: band_on_n<3>(tier, m, xh + i0, c, tc0, ks0, ks1, acc + i0); break;
            case 4: band_on_n<4>(tier, m, xh + i0, c, tc0, ks0, ks1, acc + i0); break;
            case 5: band_on_n<5>(tier, m, xh + i0, c, tc0, ks0, ks1, acc + i0); break;
            case 6: band_on_n<6>(tier, m, xh + i0, c, tc0, ks0, ks1, acc + i0); break;
            case 7: band_on_n<7>(tier, m, xh + i0, c, tc0, ks0, ks1, acc + i0); break;
            case 8: band_on_n<8>(tier, m, xh + i0, c, tc0, ks0, ks1, acc + i0); break;
            default:
                for (int i = 0; i < c; ++i)
                    for (int j = 0; j < 128; ++j) acc[i0 + i][j] = 0.0f;
        }
    }
}

}  // namespace

int exl3_cpu_tier() {
    static const int tier = detect_tier();
    return tier;
}

bool exl3_cpu_vector() { return exl3_cpu_tier() > 0; }

float exl3_cpu_prepare_block(const uint16_t* suh, const float* x, float* xh) {
#if defined(EXL3_X86)
    if (exl3_cpu_tier() == 2) return prepare_block_avx512(suh, x, xh);
    if (exl3_cpu_tier() == 1) return prepare_block_avx2(suh, x, xh);
#endif
    for (int i = 0; i < 128; ++i) xh[i] = x[i] * f32_from_f16(suh[i]);
    had128(xh);
    double sum = 0.0;
    for (int i = 0; i < 128; ++i) sum += xh[i];
    return (float) sum;
}

float exl3_cpu_prepare(const Exl3Part& m, const float* x, float* xh) {
    double sum = 0.0;
    for (int b = 0; b < m.k; b += 128) sum += exl3_cpu_prepare_block(m.suh + b, x + b, xh + b);
    return (float) sum;
}

void exl3_cpu_band_acc(const Exl3Part& m, const float* xh, int c0, int ks0, int ks1, float* acc) {
    band_acc_tier(exl3_cpu_tier(), m, xh, c0, ks0, ks1, acc);
}

void exl3_cpu_band_acc_n(const Exl3Part& m, const float* const* xh, int n, int c0, int ks0, int ks1,
                         float* const* acc) {
    band_acc_n_tier(exl3_cpu_tier(), m, xh, n, c0, ks0, ks1, acc);
}

bool exl3_cpu_band_acc_on(int tier, const Exl3Part& m, const float* const* xh, int n, int c0, int ks0, int ks1,
                          float* const* acc) {
    if (tier < 0 || tier > 2 || ((tier_mask() >> tier) & 1) == 0 || n < 1) return false;
    if (n == 1) band_acc_tier(tier, m, xh[0], c0, ks0, ks1, acc[0]);
    else band_acc_n_tier(tier, m, xh, n, c0, ks0, ks1, acc);
    return true;
}

void exl3_cpu_rows_acc_cols_n(const Exl3Part& m, const float* const* xh, int n, int ks0, int ks1, int c0, int c1,
                              float* const* acc) {
    const int tier = exl3_cpu_tier();
    const int t0 = c0 / 16, t1 = c1 / 16;
    for (int i0 = 0; i0 < n; i0 += kExl3CpuMaxN) {
        const int c = std::min(kExl3CpuMaxN, n - i0);
        switch (m.K) {
            case 1: rows_on_n<1>(tier, m, xh + i0, c, ks0, ks1, t0, t1, acc + i0); break;
            case 2: rows_on_n<2>(tier, m, xh + i0, c, ks0, ks1, t0, t1, acc + i0); break;
            case 3: rows_on_n<3>(tier, m, xh + i0, c, ks0, ks1, t0, t1, acc + i0); break;
            case 4: rows_on_n<4>(tier, m, xh + i0, c, ks0, ks1, t0, t1, acc + i0); break;
            case 5: rows_on_n<5>(tier, m, xh + i0, c, ks0, ks1, t0, t1, acc + i0); break;
            case 6: rows_on_n<6>(tier, m, xh + i0, c, ks0, ks1, t0, t1, acc + i0); break;
            case 7: rows_on_n<7>(tier, m, xh + i0, c, ks0, ks1, t0, t1, acc + i0); break;
            case 8: rows_on_n<8>(tier, m, xh + i0, c, ks0, ks1, t0, t1, acc + i0); break;
            default:
                for (int i = 0; i < c; ++i)
                    for (int j = c0; j < c1; ++j) acc[i0 + i][j] = 0.0f;
        }
    }
}

void exl3_cpu_rows_acc_n(const Exl3Part& m, const float* const* xh, int n, int ks0, int ks1, float* const* acc) {
    exl3_cpu_rows_acc_cols_n(m, xh, n, ks0, ks1, 0, m.n, acc);
}

size_t exl3_cpu_gemm_scratch() { return (size_t) kGemmK * 128; }

void exl3_cpu_band_gemm(const Exl3Part& m, const float* const* xh, int n, int c0, int nb, float* const* acc,
                        float* wc) {
#if defined(EXL3_X86)
    if (exl3_cpu_tier() == 2) {
        switch (m.K) {
            case 1: band_gemm_avx512<1>(m, xh, n, c0 / 16, nb, acc, wc); return;
            case 2: band_gemm_avx512<2>(m, xh, n, c0 / 16, nb, acc, wc); return;
            case 3: band_gemm_avx512<3>(m, xh, n, c0 / 16, nb, acc, wc); return;
            case 4: band_gemm_avx512<4>(m, xh, n, c0 / 16, nb, acc, wc); return;
            case 5: band_gemm_avx512<5>(m, xh, n, c0 / 16, nb, acc, wc); return;
            case 6: band_gemm_avx512<6>(m, xh, n, c0 / 16, nb, acc, wc); return;
            case 7: band_gemm_avx512<7>(m, xh, n, c0 / 16, nb, acc, wc); return;
            case 8: band_gemm_avx512<8>(m, xh, n, c0 / 16, nb, acc, wc); return;
            default: break;
        }
    }
    if (exl3_cpu_tier() == 1) {
        switch (m.K) {
            case 1: band_gemm_avx2<1>(m, xh, n, c0 / 16, nb, acc, wc); return;
            case 2: band_gemm_avx2<2>(m, xh, n, c0 / 16, nb, acc, wc); return;
            case 3: band_gemm_avx2<3>(m, xh, n, c0 / 16, nb, acc, wc); return;
            case 4: band_gemm_avx2<4>(m, xh, n, c0 / 16, nb, acc, wc); return;
            case 5: band_gemm_avx2<5>(m, xh, n, c0 / 16, nb, acc, wc); return;
            case 6: band_gemm_avx2<6>(m, xh, n, c0 / 16, nb, acc, wc); return;
            case 7: band_gemm_avx2<7>(m, xh, n, c0 / 16, nb, acc, wc); return;
            case 8: band_gemm_avx2<8>(m, xh, n, c0 / 16, nb, acc, wc); return;
            default: break;
        }
    }
#endif
    (void) wc;
    for (int i = 0; i < n; ++i)
        for (int b = 0; b < nb; ++b) exl3_cpu_band_acc(m, xh[i], c0 + 128 * b, 0, m.k / 16, acc[i] + 128 * b);
}

void exl3_cpu_rows_acc(const Exl3Part& m, const float* xh, int ks0, int ks1, float* acc) {
    exl3_cpu_rows_acc_n(m, &xh, 1, ks0, ks1, &acc);
}

void exl3_cpu_band_out(const Exl3Part& m, const float* acc, float xsum, int c0, float* y) {
    const float ki = k_inv(), kz = k_zero() * xsum;
#if defined(EXL3_X86)
    if (exl3_cpu_tier() == 2) {
        band_out_avx512(acc, ki, kz, m.svh + c0, y);
        return;
    }
    if (exl3_cpu_tier() == 1) {
        band_out_avx2(acc, ki, kz, m.svh + c0, y);
        return;
    }
#endif
    for (int i = 0; i < 128; ++i) y[i] = acc[i] * ki + kz;
    had128(y);
    for (int i = 0; i < 128; ++i) y[i] *= f32_from_f16(m.svh[c0 + i]);
}

void exl3_cpu_band(const Exl3Part& m, const float* xh, float xsum, int c0, float* y) {
    float acc[128];
    exl3_cpu_band_acc(m, xh, c0, 0, m.k / 16, acc);
    exl3_cpu_band_out(m, acc, xsum, c0, y);
}

}  // namespace strata::dsv41k::cpu
