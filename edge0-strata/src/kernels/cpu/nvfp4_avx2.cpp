// src/kernels/cpu/nvfp4_avx2.cpp - see include/strata/kernels/cpu/nvfp4_avx2.hpp.
//
// Per 64-value block, once: its E2M1 values in element order as two 32-byte registers (ggml's reordering: sub-block
// s keeps value j in the low nibble of qs[8 s + j] and value j + 8 in the high one), their magnitudes, and the four
// sub-block scales. Per token, as ggml_vec_dot_nvfp4_q8_0's AVX2 path: vpsignb moves the sign onto the activation,
// vpmaddubsw + vpmaddwd give 8 int32 lanes (four per 16-value sub-block), one FMA per 32 values applies
// ue4m3(d[s]) * d_q8, and hsum_float_8's order sums the lanes.
//
// Needs AVX2, FMA and F16C - what cpu_avx2_ok() checks before anything here is called.
#include "strata/kernels/cpu/nvfp4_avx2.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

// A compile-time table, not a constructor: this file is compiled for a wide ISA, and a namespace-scope object with a
// constructor runs at program start on EVERY CPU - the compiler vectorises the loop with AVX/FMA, and a CPU without them
// (Westmere, Sandy Bridge, ...) died with an illegal instruction before main (#2324).  Powers of two: exact.
constexpr float pow2i(int e) {
    float r = 1.0f;
    for (; e > 0; --e) r *= 2.0f;
    for (; e < 0; ++e) r *= 0.5f;
    return r;
}
struct Ue4m3Table {   // ggml_ue4m3_to_fp32 (ggml-impl.h): halved for the doubled kvalues_fp4, 0x7F read as 0
    float v[256];
    constexpr Ue4m3Table() : v() {
        for (int x = 0; x < 256; ++x) {
            if (x == 0 || x == 0x7F) { v[x] = 0.0f; continue; }
            const int e = (x >> 3) & 0xF, m = x & 0x7;
            const float raw = e == 0 ? (float) m * pow2i(-9) : (1.0f + (float) m / 8.0f) * pow2i(e - 7);
            v[x] = raw * 0.5f;
        }
    }
};
constexpr Ue4m3Table kUe4m3;

// software prefetch distance in bytes (STRATA_NVFP4_PREFETCH, 0 = off), as the AVX-512 rows
// (read on first use, behind the CPU check: a static initialiser here would run on every CPU, see above)
inline int prefetch_distance() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_NVFP4_PREFETCH");
        return e ? std::atoi(e) : 2048;
    }();
    return v;
}

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }

// sub-blocks 2 h and 2 h + 1 (16 bytes of qs) as their 32 signed doubled E2M1 values, in element order
inline __m256i values32(const uint8_t* qs, const __m128i kv) {
    const __m128i q = _mm_loadu_si128((const __m128i*) qs);
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m128i lo = _mm_shuffle_epi8(kv, _mm_and_si128(q, m4));
    const __m128i hi = _mm_shuffle_epi8(kv, _mm_and_si128(_mm_srli_epi16(q, 4), m4));
    return _mm256_set_m128i(_mm_unpackhi_epi64(lo, hi), _mm_unpacklo_epi64(lo, hi));
}

inline float hsum8(const __m256 x) {   // ggml's hsum_float_8
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

constexpr int kMaxBlocks = 64;   // rows of up to 4096 values (n_embd 2560)
// tokens per pass: the pool's MAXT, so a window is one pass. Spilled accumulators cost less than another pass over
// the weights: one expert's gate+up, 8 tokens, 0.415 ms in passes of 4 + 4, 0.352 in one of 8 (5 6 7 alike)
constexpr int kMaxW = 8;

struct Acts {   // per token and block, the two Q8_0 blocks' scales; built once per call, reused by every row
    int nt = 0, nb = 0;
    const block_q8_0* y[kMaxW] = {};
    float dy[kMaxW * kMaxBlocks * 2];
    Acts(const void* const* act, int nt_, int nb_) : nt(nt_), nb(nb_) {
        for (int t = 0; t < nt; ++t) {
            y[t] = (const block_q8_0*) act[t];
            for (int ib = 0; ib < 2 * nb; ++ib) dy[(size_t) t * 2 * nb + ib] = h2f(y[t][ib].d);
        }
    }
};

template <int NT>
inline void row_dot(const uint8_t* row, const Acts& a, float* res) {
    const __m128i kv = _mm_loadu_si128((const __m128i*) kvalues_fp4);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    const block_nvfp4* x = (const block_nvfp4*) row;
    const int prefetch_ahead = prefetch_distance();
    for (int ib = 0; ib < a.nb; ++ib) {
        if (prefetch_ahead > 0) _mm_prefetch((const char*) (x + ib) + prefetch_ahead, _MM_HINT_T0);
        const __m256i v01 = values32(x[ib].qs, kv), v23 = values32(x[ib].qs + 16, kv);
        const __m256i a01 = _mm256_sign_epi8(v01, v01), a23 = _mm256_sign_epi8(v23, v23);
        // lanes 0-3 ue4m3(d[0]), 4-7 d[1] (and d[2], d[3]): times a broadcast d_q8, each lane is ggml's scalar
        // ue4m3(d[s]) * d_q8, rounded the same
        const __m256 u01 = _mm256_set_m128(_mm_set1_ps(kUe4m3.v[x[ib].d[1]]), _mm_set1_ps(kUe4m3.v[x[ib].d[0]]));
        const __m256 u23 = _mm256_set_m128(_mm_set1_ps(kUe4m3.v[x[ib].d[3]]), _mm_set1_ps(kUe4m3.v[x[ib].d[2]]));
        for (int t = 0; t < NT; ++t) {
            const block_q8_0* yb = a.y[t] + 2 * ib;
            const __m256i q01 = _mm256_loadu_si256((const __m256i*) yb[0].qs);
            const __m256i q23 = _mm256_loadu_si256((const __m256i*) yb[1].qs);
            const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(a01, _mm256_sign_epi8(q01, v01)), ones);
            const __m256i p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(a23, _mm256_sign_epi8(q23, v23)), ones);
            const float* dy = a.dy + (size_t) t * 2 * a.nb + 2 * ib;
            const __m256 s01 = _mm256_mul_ps(u01, _mm256_broadcast_ss(dy));
            const __m256 s23 = _mm256_mul_ps(u23, _mm256_broadcast_ss(dy + 1));
            acc[t] = _mm256_fmadd_ps(s01, _mm256_cvtepi32_ps(p1), acc[t]);
            acc[t] = _mm256_fmadd_ps(s23, _mm256_cvtepi32_ps(p2), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum8(acc[t]);
}

template <int NT>
void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, const Acts& a, float* const* ff, int r0, int r1,
             float sg, float su) {
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<NT>(blob + (size_t) r * gu_row, a, g);
        row_dot<NT>(blob + up_off + (size_t) r * gu_row, a, u);
        for (int t = 0; t < NT; ++t) {   // native_gu_rows' ggml path, operation for operation
            const float gs = g[t] * sg, us = u[t] * su;
            ff[t][r] = (gs / (1.f + std::exp(-gs))) * us;
        }
    }
}

template <int NT>
void dot_rows(const uint8_t* w, size_t row_bytes, const Acts& a, float* const* out, int r0, int r1, float scale) {
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<NT>(w + (size_t) r * row_bytes, a, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t] * scale;
    }
}

// windows wider than kMaxW (not from the pool) run in balanced slices of at most kMaxW
template <typename F>
inline void slices(int nt, F&& run) {
    const int k = (nt + kMaxW - 1) / kMaxW;
    for (int i = 0, t0 = 0; i < k; ++i) {
        const int w = nt / k + (i < nt % k ? 1 : 0);
        run(t0, w);
        t0 += w;
    }
}

}  // namespace

bool nvfp4_256_fits(int n) { return n % QK_NVFP4 == 0 && n / QK_NVFP4 <= kMaxBlocks; }

void nvfp4_256_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                       float* const* ff, int r0, int r1, float s_gate, float s_up) {
    slices(nt, [&](int t0, int w) {
        const Acts a(act + t0, w, n / QK_NVFP4);
        float* const* f = ff + t0;
        switch (w) {
            case 1: gu_rows<1>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            case 2: gu_rows<2>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            case 3: gu_rows<3>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            case 4: gu_rows<4>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            case 5: gu_rows<5>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            case 6: gu_rows<6>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            case 7: gu_rows<7>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            default: gu_rows<8>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
        }
    });
}

void nvfp4_256_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                    int r0, int r1, float scale) {
    slices(nt, [&](int t0, int k) {
        const Acts a(act + t0, k, n / QK_NVFP4);
        float* const* o = out + t0;
        switch (k) {
            case 1: dot_rows<1>(w, row_bytes, a, o, r0, r1, scale); break;
            case 2: dot_rows<2>(w, row_bytes, a, o, r0, r1, scale); break;
            case 3: dot_rows<3>(w, row_bytes, a, o, r0, r1, scale); break;
            case 4: dot_rows<4>(w, row_bytes, a, o, r0, r1, scale); break;
            case 5: dot_rows<5>(w, row_bytes, a, o, r0, r1, scale); break;
            case 6: dot_rows<6>(w, row_bytes, a, o, r0, r1, scale); break;
            case 7: dot_rows<7>(w, row_bytes, a, o, r0, r1, scale); break;
            default: dot_rows<8>(w, row_bytes, a, o, r0, r1, scale); break;
        }
    });
}

}  // namespace strata::kernels::cpu
