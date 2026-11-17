// src/kernels/cpu/agg_bandwidth_bench.cpp
// Step 5.3 (v2, cold-corrected): the TRUE cold aggregate ceiling of the engine's 15 workers.
//   A cold stream : 15 threads each do ONE pass over a fresh 64 MB slice (960 MB total >> L3) -> raw cold read ceiling
//   B cold matvec : 15 threads each run the U matvec over FRESH groups (60 groups = 292 MB, read once) -> the engine shape, cold
// Both pinned to the engine's exact worker set: P-cores 2,4,6,8,10,12,14 + E-cores 16..23.
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>
#ifdef __linux__
#include <sched.h>
#endif

static const int cores[15] = {2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23};
static void pin_to(int cpu) {
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    sched_setaffinity(0, sizeof(set), &set);
#else
    (void) cpu;
#endif
}

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
        dot_maddubs<1>(w + (size_t) r * nb, nb, act, s);
        out[r] = s[0];
    }
}

int main() {
    constexpr int R = 2560, N = 1792, NB = N / 32, T = 15;
    const size_t group_bytes = (size_t) R * NB * 34;   // 4.87 MB per group
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> qd(-127, 127);
    const uint16_t scale_half = 0x2000u;

    // ---- A: cold raw stream, 15 threads x 64 MB (960 MB, read once) ----
    {
        const size_t per = (size_t) 64 * 1024 * 1024;
        std::vector<uint8_t> buf((size_t) T * per);
        for (size_t i = 0; i < buf.size(); ++i) buf[i] = (uint8_t)(i & 0xff);
        std::vector<uint64_t> sums(T, 0);
        std::atomic<int> go{0};
        std::vector<std::thread> th;
        for (int t = 0; t < T; ++t)
            th.emplace_back([&, t]() {
                pin_to(cores[t]);
                const uint8_t* p = buf.data() + (size_t) t * per;
                uint64_t s = 0;
                while (go.load() == 0) {}
                const auto t0 = std::chrono::steady_clock::now();
                for (size_t j = 0; j < per / 8; ++j) s += ((const uint64_t*) p)[j];
                const auto t1 = std::chrono::steady_clock::now();
                sums[t] = s;
                (void) t1;
            });
        const auto w0 = std::chrono::steady_clock::now();
        go.store(1);
        for (auto& x : th) x.join();
        const auto w1 = std::chrono::steady_clock::now();
        const double wall = std::chrono::duration<double, std::milli>(w1 - w0).count();
        volatile uint64_t sink = 0; for (auto s : sums) sink += s; (void) sink;
        std::printf("A cold raw-stream  15x64MB read-once: wall %.3f ms = %.1f GB/s aggregate\n", wall, (double) T * per / wall / 1e6);
    }

    // ---- B: cold matvec, 15 threads x 4 fresh groups each (60 groups = 292 MB, read once) ----
    {
        const int K = 60;   // 4 groups per thread, disjoint
        std::vector<block_q8_0> w((size_t) K * R * NB);
        for (auto& b : w) { b.d = scale_half; for (int j = 0; j < 32; ++j) b.qs[j] = (int8_t) qd(rng); }
        std::vector<uint8_t> actb((size_t) NB * 36);
        for (size_t p = 0; p < actb.size(); p += 36) { std::memcpy(actb.data() + p, &scale_half, 2); for (int j = 0; j < 32; ++j) actb[p + 4 + j] = (uint8_t)(int8_t) qd(rng); }
        const void* actp = actb.data();
        std::vector<float> out((size_t) T * R, 0.f);
        std::atomic<int> go{0};
        std::vector<std::thread> th;
        for (int t = 0; t < T; ++t)
            th.emplace_back([&, t]() {
                pin_to(cores[t]);
                float* o = out.data() + (size_t) t * R;
                while (go.load() == 0) {}
                const auto t0 = std::chrono::steady_clock::now();
                for (int g = 0; g < 4; ++g) {   // thread t reads groups t, t+15, t+30, t+45 (disjoint, cold)
                    const block_q8_0* gw = w.data() + (size_t) (t + g * T) * R * NB;
                    rows_maddubs(gw, R, NB, &actp, 1, o);
                }
                const auto t1 = std::chrono::steady_clock::now();
                (void) t1;
            });
        const auto w0 = std::chrono::steady_clock::now();
        go.store(1);
        for (auto& x : th) x.join();
        const auto w1 = std::chrono::steady_clock::now();
        const double wall = std::chrono::duration<double, std::milli>(w1 - w0).count();
        const double bytes = (double) K * group_bytes;
        const double flops = (double) K * R * N * 2;
        volatile float sink = 0; for (int t = 0; t < T; ++t) sink += out[(size_t) t * R]; (void) sink;
        std::printf("B cold matvec      15 workers x 4 groups read-once: wall %.3f ms = %.1f GB/s aggregate = %.1f GFLOP/s\n",
                    wall, bytes / wall / 1e6, flops / wall / 1e6);
    }
    return 0;
}
