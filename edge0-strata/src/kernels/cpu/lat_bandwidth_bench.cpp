// src/kernels/cpu/lat_bandwidth_bench.cpp
// Step 5.2: is the U matvec memory-bound?  Measures the maddubs kernel's throughput in two regimes:
//   hot  = one group (H x ELL Q8_0 = ~4.87 MB), re-read every iteration (fits L3)  -> compute-bound ceiling
//   cold = K groups (> 4x L3), each read once and cycled (streamed from RAM)        -> the arena's streaming rate
// Reports GFLOP/s and the weight stream GB/s for both, so (a) bandwidth vs (b) dilution is decided.
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static inline float h2f(uint16_t h) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h)));
}
static inline float hsum_float_8(__m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}
template <int NT>
static void dot_maddubs(const block_q8_0* x, int nb, const void* const* act, float* res) {
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        const float xd = h2f(x[ib].d);
        const __m256i qx = _mm256_loadu_si256((const __m256i*) x[ib].qs);
        const __m256i ax = _mm256_sign_epi8(qx, qx);
        for (int t = 0; t < NT; ++t) {
            const uint8_t* yb = (const uint8_t*) act[t] + (size_t) ib * 36;
            const float dy = h2f(*(const uint16_t*) (yb + 0));
            const __m256 d = _mm256_set1_ps(xd * dy);
            const __m256i qy = _mm256_loadu_si256((const __m256i*) (yb + 4));
            const __m256i sy = _mm256_sign_epi8(qy, qx);
            const __m256 q = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(ax, sy)));
            acc[t] = _mm256_fmadd_ps(q, d, acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum_float_8(acc[t]);
}
static void rows_maddubs(const block_q8_0* w, int R, int nb, const void* const* act, int nt, float* out) {
    for (int r = 0; r < R; ++r) {
        float s[4];
        switch (nt) {
            case 1: dot_maddubs<1>(w + (size_t) r * nb, nb, act, s); out[r] = s[0]; break;
            case 2: dot_maddubs<2>(w + (size_t) r * nb, nb, act, s); out[r] = s[0]; out[R + r] = s[1]; break;
            default: dot_maddubs<4>(w + (size_t) r * nb, nb, act, s); for (int t = 0; t < 4; ++t) out[(size_t) t * R + r] = s[t]; break;
        }
    }
}

int main() {
    constexpr int R = 2560, N = 1792, NB = N / 32;
    const size_t group_bytes = (size_t) R * NB * 34;   // one group's Q8_0 weights
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> qd(-127, 127);
    const uint16_t scale_half = 0x2000u;

    const int K = 32;   // cold: 32 groups = 32 * 4.87 MB ~ 156 MB > 4x L3 (36 MB)
    std::vector<block_q8_0> w((size_t) K * R * NB);
    for (auto& b : w) { b.d = scale_half; for (int j = 0; j < 32; ++j) b.qs[j] = (int8_t) qd(rng); }
    std::vector<uint8_t> actb((size_t) 4 * NB * 36);
    for (size_t p = 0; p < actb.size(); p += 36) { std::memcpy(actb.data() + p, &scale_half, 2); for (int j = 0; j < 32; ++j) actb[p + 4 + j] = (uint8_t)(int8_t) qd(rng); }
    const void* actp[4];
    for (int t = 0; t < 4; ++t) actp[t] = actb.data() + (size_t) t * NB * 36;

    std::vector<float> out((size_t) 4 * R);
    std::fill(out.begin(), out.end(), 0.f);
    volatile float sink = 0.f;

    for (int nt : {1, 4}) {
        const double flops = 2.0 * (double) R * (double) N * (double) nt;
        const int it = 40;

        // hot: group 0 re-read every iteration
        {
            double best = 1e30;
            for (int i = 0; i < it; ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                rows_maddubs(w.data(), R, NB, actp, nt, out.data());
                const auto t1 = std::chrono::steady_clock::now();
                best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            std::printf("nt=%d HOT   (1 group, L3-resident): %.3f ms = %.1f GFLOP/s  (weight stream %.1f GB/s)\n",
                        nt, best, flops / best / 1e6, (double) group_bytes / best / 1e6);
            sink += out[0];
        }

        // cold: cycle K groups, each read once per pass
        {
            double best = 1e30;
            for (int i = 0; i < it; ++i) {
                const int g = i % K;
                const block_q8_0* gw = w.data() + (size_t) g * R * NB;
                const auto t0 = std::chrono::steady_clock::now();
                rows_maddubs(gw, R, NB, actp, nt, out.data());
                const auto t1 = std::chrono::steady_clock::now();
                best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            std::printf("nt=%d COLD  (%d groups cycled, streamed): %.3f ms = %.1f GFLOP/s  (weight stream %.1f GB/s)\n",
                        nt, K, best, flops / best / 1e6, (double) group_bytes / best / 1e6);
            sink += out[0];
        }
    }
    (void) sink;
    return 0;
}
