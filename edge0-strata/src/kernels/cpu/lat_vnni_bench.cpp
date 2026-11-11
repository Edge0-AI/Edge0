// src/kernels/cpu/lat_vnni_bench.cpp
// Step 5.1 microbenchmark: the latent U matvec (Q8_0 weights x Q8_1 acts, H=2560 x ELL=1792) on two paths -
// the current AVX2 maddubs dot (kq_avx2.cpp lat_q8_0_q8_1_dot) and an AVX-VNNI dpbusd dot (target("avxvnni")),
// against a double reference.  Reports throughput (GFLOP/s) and the output delta (max/mean/p99).
// No GPU, no model: random Q8_0/Q8_1 blocks of the model's U shape.
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <cmath>
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

// ---- the current maddubs inner dot (kq_avx2.cpp lat_q8_0_q8_1_dot), verbatim ----
template <int NT>
static void dot_maddubs(const block_q8_0* x, int nb, const void* const* act, float* res) {
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        const float xd = h2f(x[ib].d);
        const __m256i qx = _mm256_loadu_si256((const __m256i*) x[ib].qs);
        const __m256i ax = _mm256_sign_epi8(qx, qx);   // |weight|
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

// ---- the AVX-VNNI dpbusd inner dot (per-block zero-src + cvt + fmadd, like expert.cpp row_dot_multi) ----
template <int NT>
__attribute__((target("avxvnni")))
static void dot_vnni(const block_q8_0* x, int nb, const void* const* act, float* res) {
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        const float xd = h2f(x[ib].d);
        const __m256i qx = _mm256_loadu_si256((const __m256i*) x[ib].qs);
        const __m256i ax = _mm256_sign_epi8(qx, qx);   // |weight| (unsigned operand of dpbusd)
        for (int t = 0; t < NT; ++t) {
            const uint8_t* yb = (const uint8_t*) act[t] + (size_t) ib * 36;
            const float dy = h2f(*(const uint16_t*) (yb + 0));
            const __m256 d = _mm256_set1_ps(xd * dy);
            const __m256i qy = _mm256_loadu_si256((const __m256i*) (yb + 4));
            const __m256i sy = _mm256_sign_epi8(qy, qx);
            const __m256 q = _mm256_cvtepi32_ps(_mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), ax, sy));
            acc[t] = _mm256_fmadd_ps(q, d, acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum_float_8(acc[t]);
}

// one full U matvec: R rows x (nb*32) cols, nt tokens; out[t*R + r].
static void rows_maddubs(const block_q8_0* w, int R, int nb, const void* const* act, int nt, float* out) {
    for (int r = 0; r < R; ++r) {
        const block_q8_0* wr = w + (size_t) r * nb;
        float s[4];
        switch (nt) {
            case 1: dot_maddubs<1>(wr, nb, act, s); out[r] = s[0]; break;
            case 2: dot_maddubs<2>(wr, nb, act, s); out[r] = s[0]; out[R + r] = s[1]; break;
            case 3: dot_maddubs<3>(wr, nb, act, s); for (int t = 0; t < 3; ++t) out[(size_t) t * R + r] = s[t]; break;
            default: dot_maddubs<4>(wr, nb, act, s); for (int t = 0; t < 4; ++t) out[(size_t) t * R + r] = s[t]; break;
        }
    }
}
__attribute__((target("avxvnni")))
static void rows_vnni(const block_q8_0* w, int R, int nb, const void* const* act, int nt, float* out) {
    for (int r = 0; r < R; ++r) {
        const block_q8_0* wr = w + (size_t) r * nb;
        float s[4];
        switch (nt) {
            case 1: dot_vnni<1>(wr, nb, act, s); out[r] = s[0]; break;
            case 2: dot_vnni<2>(wr, nb, act, s); out[r] = s[0]; out[R + r] = s[1]; break;
            case 3: dot_vnni<3>(wr, nb, act, s); for (int t = 0; t < 3; ++t) out[(size_t) t * R + r] = s[t]; break;
            default: dot_vnni<4>(wr, nb, act, s); for (int t = 0; t < 4; ++t) out[(size_t) t * R + r] = s[t]; break;
        }
    }
}

int main() {
    constexpr int R = 2560, N = 1792, NB = N / 32;   // the U matvec shape H x ELL
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> qd(-127, 127);
    const uint16_t scale_half = 0x2000u;   // 0.0078125, a valid f16 block scale

    std::vector<block_q8_0> w((size_t) R * NB);
    for (auto& b : w) {
        b.d = scale_half;
        for (int j = 0; j < 32; ++j) b.qs[j] = (int8_t) qd(rng);
    }
    // acts as raw bytes: block_q8_1 = 36 B/block (d +0, s +2 [unused], qs +4)
    std::vector<uint8_t> actb((size_t) 4 * NB * 36);
    for (int t = 0; t < 4; ++t)
        for (int ib = 0; ib < NB; ++ib) {
            uint8_t* p = actb.data() + ((size_t) t * NB + ib) * 36;
            std::memcpy(p + 0, &scale_half, 2);
            *(uint16_t*)(p + 2) = 0;
            for (int j = 0; j < 32; ++j) p[4 + j] = (uint8_t)(int8_t) qd(rng);
        }

    for (int nt : {1, 2, 4}) {
        const void* actp[4];
        for (int t = 0; t < nt; ++t) actp[t] = actb.data() + (size_t) t * NB * 36;

        // double reference
        std::vector<double> ref((size_t) nt * R, 0.0);
        for (int t = 0; t < nt; ++t)
            for (int r = 0; r < R; ++r) {
                double s = 0;
                const block_q8_0* wr = &w[(size_t) r * NB];
                const uint8_t* ar = actb.data() + (size_t) t * NB * 36;
                for (int ib = 0; ib < NB; ++ib) {
                    int dot = 0;
                    const int8_t* wq = wr[ib].qs;
                    const int8_t* aq = (const int8_t*)(ar + (size_t) ib * 36 + 4);
                    for (int j = 0; j < 32; ++j) dot += (int) wq[j] * (int) aq[j];
                    s += (double) h2f(wr[ib].d) * (double) h2f(*(const uint16_t*)(ar + (size_t) ib * 36 + 0)) * (double) dot;
                }
                ref[(size_t) t * R + r] = s;
            }

        std::vector<float> om((size_t) nt * R), ov((size_t) nt * R);
        std::fill(om.begin(), om.end(), 0.f);
        std::fill(ov.begin(), ov.end(), 0.f);

        rows_maddubs(w.data(), R, NB, actp, nt, om.data());
        rows_vnni(w.data(), R, NB, actp, nt, ov.data());

        auto rel = [&](const std::vector<float>& o) {
            double num = 0, den = 0;
            for (size_t i = 0; i < o.size(); ++i) { num += std::fabs(o[i] - ref[i]); den += std::fabs(ref[i]); }
            return num / (den + 1e-30);
        };
        std::printf("nt=%d  rel-vs-ref  maddubs %.2e  vnni %.2e\n", nt, rel(om), rel(ov));

        std::vector<double> dd((size_t) nt * R);
        for (size_t i = 0; i < dd.size(); ++i) dd[i] = std::fabs((double) om[i] - (double) ov[i]);
        std::sort(dd.begin(), dd.end());
        double dmax = dd.back(), dsum = 0;
        for (double v : dd) dsum += v;
        std::printf("  delta maddubs-vs-vnni: max %.3e  mean %.3e  p99 %.3e\n",
                    dmax, dsum / (double) dd.size(), dd[(size_t) (0.99 * (dd.size() - 1))]);

        const int it = 40;
        auto bench = [&](void (*fn)(const block_q8_0*, int, int, const void* const*, int, float*), float* o) {
            double best = 1e30;
            for (int i = 0; i < it; ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                fn(w.data(), R, NB, actp, nt, o);
                const auto t1 = std::chrono::steady_clock::now();
                best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            return best;
        };
        const double tm = bench(rows_maddubs, om.data());
        const double tv = bench(rows_vnni, ov.data());
        const double flops = 2.0 * (double) R * (double) N * (double) nt;
        std::printf("  throughput (best of %d): maddubs %.3f ms = %.1f GFLOP/s | vnni %.3f ms = %.1f GFLOP/s | speedup %.3fx\n",
                    it, tm, flops / tm / 1e6, tv, flops / tv / 1e6, tm / tv);
    }
    return 0;
}
