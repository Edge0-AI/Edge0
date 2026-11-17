// src/kernels/cpu/router_dot_parity.cpp - the router lookahead's BF16 dot on each rung this CPU has: the AVX2 kernel
// (kq_avx2.cpp) and the AVX1 one for older CPUs (kq_avx1.cpp), against a double-precision reference, plus the time
// of one layer's router.  No GPU, no model: random BF16 rows of the model's router shape [512 x 2560], 1..8 tokens.
//
//     router_dot_parity            (STRATA_FORCE_ISA=avx leaves out the AVX2 kernel, as on an AVX-only CPU)
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/kq_avx1.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace cpu = strata::kernels::cpu;

int main() {
    constexpr int R = 512, C = 2560, MAXT = 8;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 0.05f);
    std::vector<uint16_t> w((size_t) R * C);
    for (auto& v : w) {
        const float f = nd(rng);
        uint32_t b;
        std::memcpy(&b, &f, 4);
        v = (uint16_t) (b >> 16);
    }
    std::vector<float> x((size_t) MAXT * C);
    for (auto& v : x) v = nd(rng) * 20.f;
    std::vector<double> ref((size_t) MAXT * R);
    for (int t = 0; t < MAXT; ++t)
        for (int r = 0; r < R; ++r) {
            double s = 0;
            for (int c = 0; c < C; ++c) {
                const uint32_t b = (uint32_t) w[(size_t) r * C + c] << 16;
                float f;
                std::memcpy(&f, &b, 4);
                s += (double) f * x[(size_t) t * C + c];
            }
            ref[(size_t) t * R + r] = s;
        }
    int failures = 0;
    auto run = [&](const char* tag, void (*fn)(const uint16_t*, int, int, const float*, int, float*)) {
        std::vector<float> out((size_t) MAXT * R);
        for (int nt = 1; nt <= MAXT; ++nt) {
            std::fill(out.begin(), out.end(), 0.f);
            fn(w.data(), R, C, x.data(), nt, out.data());
            double num = 0, den = 0;
            for (int i = 0; i < nt * R; ++i) { num += std::fabs(out[(size_t) i] - ref[(size_t) i]); den += std::fabs(ref[(size_t) i]); }
            const double rel = num / (den + 1e-30);
            if (nt == 1 || nt == MAXT) std::printf("  %-5s nt=%d rel %.2e vs the double reference\n", tag, nt, rel);
            if (!(rel < 1e-5)) { std::printf("  %-5s nt=%d MISMATCH\n", tag, nt); ++failures; }
        }
        const int it = 20;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < it; ++i) fn(w.data(), R, C, x.data(), 4, out.data());
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / it;
        std::printf("  %-5s one layer's router, 4 tokens: %.3f ms\n", tag, ms);
    };
    std::printf("router_dot_parity: %s (avx2 %d, avx %d)\n", cpu::cpu_name().c_str(), (int) cpu::cpu_avx2_ok(),
                (int) cpu::cpu_avx1_ok());
    if (cpu::cpu_avx2_ok()) run("avx2", cpu::bf16_rows_dot_multi);
    if (cpu::cpu_avx1_ok()) run("avx", cpu::bf16_rows_dot_multi_avx1);
    std::printf("router_dot_parity: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
