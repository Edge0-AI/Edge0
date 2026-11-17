// src/kernels/cpu/expert_multi_test.cpp - plan v0.3 P6: one expert, several tokens (CPU only, synthetic data).
//
//   expert_multi_test              bitwise check: s2_expert_vnni_multi vs s2_expert_vnni_q per token, n = 1..8
//   expert_multi_test --bench [E]  time per expert for n = 1..5 over E synthetic experts (default 256 = 354 MB,
//                                  far beyond L3, so the blob comes from DRAM as it does in the engine); 1 thread
//
// The bench measures the speculation model's `extra_use_cost`: the CPU time of each extra token routed to an
// expert, as a fraction of reading the expert once (tools/spec_economics.py assumes 0.2).
#include "strata/kernels/cpu/expert.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace c = strata::kernels::cpu;

namespace {
// the pool's path: the gate/up rows, the second activation quantization, the down rows (what the engine's expert
// jobs run), for one expert and n tokens
void rows_path(const uint8_t* blob, const c::ActQ* const* a1, int n, float* const* out, c::ExpertScratchMulti& ws) {
    float* ffp[c::MAXT];
    const c::ActQ* a2[c::MAXT];
    for (int t = 0; t < n; ++t) ffp[t] = ws.ff[t];
    c::s2_expert_gu_rows_multi(blob, a1, n, ffp, 0, c::FF);
    for (int t = 0; t < n; ++t) {
        c::act_quant_q8_1(ws.ff[t], c::FF, ws.a2[t]);
        a2[t] = &ws.a2[t];
    }
    c::s2_expert_down_rows_multi(blob, a2, n, out, 0, c::H);
}

void make_blob(uint8_t* b, std::mt19937& rng) {
    for (size_t i = 0; i < c::O_GU_SCALES; ++i) b[i] = (uint8_t) rng();              // codes
    for (size_t i = c::O_GU_SCALES; i < c::BLOB; i += 2) {                            // fp16 scales ~0.004-0.016
        const uint16_t h = (uint16_t) (0x1C00 + rng() % 0x0800);
        std::memcpy(b + i, &h, 2);
    }
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

int main(int argc, char** argv) {
    if (const c::CpuFeatures feat = c::cpu_features(); !feat.usable()) {   // the AVX-512 kernel: nothing to test here
        std::printf("expert_multi_test: CPU lacks %s: SKIPPED\n", feat.reason());
        return 77;
    }
    std::mt19937 rng(9);
    const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    const int E = bench ? (argc > 2 ? std::atoi(argv[2]) : 256) : 4;
    std::vector<uint8_t> blobs((size_t) E * c::BLOB);
    for (int e = 0; e < E; ++e) make_blob(&blobs[(size_t) e * c::BLOB], rng);
    std::normal_distribution<float> nd(0.f, 1.f);
    static c::ActQ acts[c::MAXT];
    static float xs[c::MAXT][c::H];
    for (int t = 0; t < c::MAXT; ++t) {
        for (float& v : xs[t]) v = nd(rng);
        c::act_quant_q8_1(xs[t], c::H, acts[t]);
    }
    const c::ActQ* a1[c::MAXT];
    for (int t = 0; t < c::MAXT; ++t) a1[t] = &acts[t];
    static float out_multi[c::MAXT][c::H], out_single[c::H];
    float* outs[c::MAXT];
    for (int t = 0; t < c::MAXT; ++t) outs[t] = out_multi[t];
    static c::ExpertScratchMulti wsm;
    static c::ExpertScratch ws;

    if (!bench) {
        int fail = 0;
        for (int e = 0; e < E; ++e) {
            const uint8_t* blob = &blobs[(size_t) e * c::BLOB];
            for (int n = 1; n <= c::MAXT; ++n) {
                c::s2_expert_vnni_multi(blob, a1, n, outs, wsm);
                for (int t = 0; t < n; ++t) {
                    c::s2_expert_vnni_q(blob, acts[t], out_single, ws);
                    if (std::memcmp(out_single, out_multi[t], sizeof out_single) != 0) {
                        std::fprintf(stderr, "FAIL expert %d n=%d token %d differs\n", e, n, t);
                        ++fail;
                    }
                }
                static float out_rows[c::MAXT][c::H];
                float* orow[c::MAXT];
                for (int t = 0; t < c::MAXT; ++t) orow[t] = out_rows[t];
                rows_path(blob, a1, n, orow, wsm);
                for (int t = 0; t < n; ++t)
                    if (std::memcmp(out_rows[t], out_multi[t], sizeof out_rows[t]) != 0) {
                        std::fprintf(stderr, "FAIL expert %d n=%d token %d: rows path differs from the multi kernel\n", e, n, t);
                        ++fail;
                    }
            }
        }
        std::printf("expert_multi_test: %s\n", fail ? "FAILED" : "OK (every token bitwise equal to the single-token kernel)");
        return fail ? 1 : 0;
    }

    std::printf("%d experts x %.2f MB, one thread\n", E, c::BLOB / 1e6);
    double base = 0;
    for (int n = 1; n <= 5; ++n) {
        c::s2_expert_vnni_multi(&blobs[0], a1, n, outs, wsm);                     // warm the code path
        const double t0 = now_ms();
        for (int e = 0; e < E; ++e) c::s2_expert_vnni_multi(&blobs[(size_t) e * c::BLOB], a1, n, outs, wsm);
        const double us = 1000.0 * (now_ms() - t0) / E;
        if (n == 1) base = us;
        std::printf("tokens per expert %d: %7.1f us per expert (%.2fx one token; extra token = %.2f of one read), %.1f GB/s\n",
                    n, us, us / base, n > 1 ? (us / base - 1.0) / (n - 1) : 0.0, c::BLOB / (us * 1e3));
        rows_path(&blobs[0], a1, n, outs, wsm);
        const double r0 = now_ms();
        for (int e = 0; e < E; ++e) rows_path(&blobs[(size_t) e * c::BLOB], a1, n, outs, wsm);
        const double rus = 1000.0 * (now_ms() - r0) / E;
        std::printf("   rows path (the pool's): %7.1f us per expert, %.1f GB/s\n", rus, c::BLOB / (rus * 1e3));
    }
    return 0;
}
