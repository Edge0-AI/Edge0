// src/kernels/cpu/nvfp4_avx512.cpp - see include/strata/kernels/cpu/nvfp4_avx512.hpp.
//
// Per 64-value block, once: the 64 E2M1 codes in element order, their magnitudes from |kvalues_fp4| (0..12, the
// doubled E2M1 values), a sign mask (code bit 3), and the four sub-block scales. Per token: the sign moves onto the
// activation (a masked subtract - Q8_0 is quantized to [-127, 127], so negation never overflows), one vpdpbusd gives
// 16 int32 lanes = four per 16-value sub-block, and one FMA applies ue4m3(d[s]) * d_q8. The UE4M3 decode is ggml's
// CPU one (ggml-impl.h ggml_ue4m3_to_fp32): halved for the doubled table, 0x7F read as 0.
//
// Needs AVX512F/BW/VL and VNNI - what cpu_avx512_ok() checks before anything here is called.
#include "strata/kernels/cpu/nvfp4_avx512.hpp"

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

// software prefetch distance in bytes (STRATA_NVFP4_PREFETCH, 0 = off): the rows stream from DRAM through 4 KB
// pages, where the hardware prefetchers stop at every page boundary; the IQ kernel's E-2 knob, same default
// (read on first use, behind the CPU check: a static initialiser here would run on every CPU, see above)
inline int prefetch_distance() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_NVFP4_PREFETCH");
        return e ? std::atoi(e) : 2048;
    }();
    return v;
}

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }

// The block's 64 codes in element order. Sub-block s keeps its codes in qs[8 s .. 8 s + 7]: byte j holds value j
// in its low nibble and value j + 8 in its high one, so a sub-block is [low nibbles, high nibbles] of its 8 bytes.
inline __m512i codes64(const uint8_t* qs) {
    const __m256i q = _mm256_loadu_si256((const __m256i*) qs);
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m256i lo = _mm256_and_si256(q, m4);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
    const __m256i ul = _mm256_unpacklo_epi64(lo, hi);                 // [lo 0-7, hi 0-7 | lo 16-23, hi 16-23]
    const __m256i uh = _mm256_unpackhi_epi64(lo, hi);                 // [lo 8-15, hi 8-15 | lo 24-31, hi 24-31]
    const __m256i s01 = _mm256_permute2x128_si256(ul, uh, 0x20);      // sub-blocks 0, 1
    const __m256i s23 = _mm256_permute2x128_si256(ul, uh, 0x31);      // sub-blocks 2, 3
    return _mm512_inserti64x4(_mm512_castsi256_si512(s01), s23, 1);
}

// lanes 4 s .. 4 s + 3 = ue4m3(d[s])
inline __m512 sub_scales(const uint8_t* d) {
    const __m128 s4 = _mm_setr_ps(kUe4m3.v[d[0]], kUe4m3.v[d[1]], kUe4m3.v[d[2]], kUe4m3.v[d[3]]);
    const __m512i idx = _mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
    return _mm512_permutexvar_ps(idx, _mm512_castps128_ps512(s4));
}

// Per token and block, the activation's scales as lanes: 0-7 -> the first Q8_0 block's d (sub-blocks 0, 1),
// 8-15 -> the second's (sub-blocks 2, 3). Built once per call, reused by every row.
constexpr int kMaxBlocks = 64;                                        // rows of up to 4096 values (n_embd 2560)
struct Acts {
    int nt = 0, nb = 0;
    const block_q8_0* y[7] = {};
    __m512 dy[7 * kMaxBlocks];                                         // [t * nb + ib]; on the stack, no allocation
    Acts(const void* const* act, int nt_, int nb_) : nt(nt_), nb(nb_) {
        for (int t = 0; t < nt; ++t) {
            y[t] = (const block_q8_0*) act[t];
            for (int ib = 0; ib < nb; ++ib) {
                const block_q8_0* yb = y[t] + 2 * ib;
                dy[(size_t) t * nb + ib] = _mm512_mask_blend_ps((__mmask16) 0xFF00, _mm512_set1_ps(h2f(yb[0].d)),
                                                                 _mm512_set1_ps(h2f(yb[1].d)));
            }
        }
    }
};

template <int NT>
inline void row_dot(const uint8_t* row, const Acts& a, float* res) {
    const __m512i mag = _mm512_broadcast_i32x4(_mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, 1, 2, 3, 4, 6, 8, 12));
    const __m512i eight = _mm512_set1_epi8(8);
    const __m512i zero = _mm512_setzero_si512();
    __m512 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm512_setzero_ps();
    const block_nvfp4* x = (const block_nvfp4*) row;
    const int prefetch_ahead = prefetch_distance();
    for (int ib = 0; ib < a.nb; ++ib) {
        if (prefetch_ahead > 0) _mm_prefetch((const char*) (x + ib) + prefetch_ahead, _MM_HINT_T0);
        const __m512i codes = codes64(x[ib].qs);
        const __m512i g = _mm512_shuffle_epi8(mag, codes);
        const __mmask64 neg = _mm512_test_epi8_mask(codes, eight);
        const __m512 sw = sub_scales(x[ib].d);
        for (int t = 0; t < NT; ++t) {
            const block_q8_0* yb = a.y[t] + 2 * ib;
            const __m512i yv = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*) yb[0].qs)),
                                                  _mm256_loadu_si256((const __m256i*) yb[1].qs), 1);
            const __m512i ys = _mm512_mask_sub_epi8(yv, neg, zero, yv);
            const __m512i p = _mm512_dpbusd_epi32(zero, g, ys);
            acc[t] = _mm512_fmadd_ps(_mm512_mul_ps(sw, a.dy[(size_t) t * a.nb + ib]), _mm512_cvtepi32_ps(p), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = _mm512_reduce_add_ps(acc[t]);
}

template <int NT>
void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, const Acts& a, float* const* ff, int r0, int r1,
             float sg, float su) {
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<NT>(blob + (size_t) r * gu_row, a, g);
        row_dot<NT>(blob + up_off + (size_t) r * gu_row, a, u);
        for (int t = 0; t < NT; ++t) {
            const float gs = g[t] * sg;
            ff[t][r] = (gs / (1.f + std::exp(-gs))) * (u[t] * su);
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

}  // namespace

bool nvfp4_512_fits(int n) { return n % QK_NVFP4 == 0 && n / QK_NVFP4 <= kMaxBlocks; }

// Windows wider than 7 run in balanced slices of at most 7: at 8 the accumulators no longer fit beside the loads and
// MSVC spills (measured 0.32 ms for 8 tokens against 2 x 0.13 for 4 + 4).
template <typename F>
inline void slices(int nt, F&& run) {
    const int k = (nt + 6) / 7;
    for (int i = 0, t0 = 0; i < k; ++i) {
        const int w = nt / k + (i < nt % k ? 1 : 0);
        run(t0, w);
        t0 += w;
    }
}

void nvfp4_512_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
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
            default: gu_rows<7>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
        }
    });
}

void nvfp4_512_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
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
            default: dot_rows<7>(w, row_bytes, a, o, r0, r1, scale); break;
        }
    });
}

}  // namespace strata::kernels::cpu
