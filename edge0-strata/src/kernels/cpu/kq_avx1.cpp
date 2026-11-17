// src/kernels/cpu/kq_avx1.cpp - the AVX1 router dot for older CPUs (see kq_avx1.hpp).  From rwkeyes' Strata_Dirigo
// fork (MIT), its AVX1 floor.
//
// Measured on the hardware this exists for (Xeon E5-2665, 2x8 cores, AVX only) for one layer's router
// [512 experts x 5120 embeddings]: the scalar std::fma fallback it replaces took 362 ms/layer, a scalar
// mul+add 4.9 ms, and this kernel 2.6 ms - 137x, with no extra memory (the weights stay bf16).
//
// Compiled for AVX (/arch:AVX, -mavx; CMakeLists.txt sets that per source); it is only ever called behind
// cpu_avx1_ok(), so no CPU that lacks AVX can reach it.
#include "strata/kernels/cpu/kq_avx1.hpp"

#include <immintrin.h>

#include <cstring>

#if !defined(__AVX__)
#error "kq_avx1.cpp must be compiled with AVX enabled (see the per-source flags in CMakeLists.txt)"
#endif

namespace strata::kernels::cpu {
namespace {

// 8 floats -> one float, the same reduction order the AVX2 kernel's hsum_float_8 uses.
inline float hsum_float_8_avx1(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x55));
    return _mm_cvtss_f32(lo);
}

// 8 bf16 -> 8 f32: a bf16 is the top half of its f32, so widen in 128-bit halves (AVX1 has no
// 256-bit integer ops) and shift each 32-bit lane into the high half.
inline __m256 bf16x8_to_f32x8_avx1(const uint16_t* p) {
    const __m128i v = _mm_loadu_si128((const __m128i*) p);
    const __m128i z = _mm_setzero_si128();
    const __m128 lo = _mm_castsi128_ps(_mm_slli_epi32(_mm_unpacklo_epi16(v, z), 16));
    const __m128 hi = _mm_castsi128_ps(_mm_slli_epi32(_mm_unpackhi_epi16(v, z), 16));
    return _mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1);
}

}  // namespace

void bf16_rows_dot_multi_avx1(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out) {
    // each row is read once for all `nt` (<= 8) tokens
    for (int r = 0; r < rows; ++r) {
        const uint16_t* wr = w + (size_t) r * (size_t) cols;
        __m256 acc[8];
        for (int t = 0; t < nt; ++t) acc[t] = _mm256_setzero_ps();
        int c = 0;
        for (; c + 8 <= cols; c += 8) {
            const __m256 wf = bf16x8_to_f32x8_avx1(wr + c);
            for (int t = 0; t < nt; ++t) {
                const __m256 xv = _mm256_loadu_ps(x + (size_t) t * (size_t) cols + c);
                acc[t] = _mm256_add_ps(acc[t], _mm256_mul_ps(wf, xv));   // no FMA on AVX1: mul, then add
            }
        }
        for (int t = 0; t < nt; ++t) {
            float s = hsum_float_8_avx1(acc[t]);
            for (; c < cols; ++c) {                                      // cols % 8 != 0 tail (cols % 8 == 0 today)
                const uint32_t bits = (uint32_t) wr[c] << 16;
                float wf; std::memcpy(&wf, &bits, sizeof wf);
                s += wf * x[(size_t) t * (size_t) cols + c];
            }
            out[(size_t) t * (size_t) rows + r] = s;
        }
    }
}

}  // namespace strata::kernels::cpu
