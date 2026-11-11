// src/prefill/lat_q3k_swiglu_test.cpp - M2b of the q3k latent A stage: SwiGLU + the q8_1 of H.
// Two gates, kept apart so a failure says which half is wrong:
//   1. SwiGLU: the float H the kernel computed against a double silu(g) * u   (rel_mean <= 1e-6)
//   2. the quantizer: the kernel's staged codes and d's against q8_1_store's rule applied on the host to the SAME
//      float H the kernel wrote out   (codes byte for byte, d's bit for bit)
#include "strata/prefill/lat_q3k_swiglu.hpp"

#include <cuda_runtime.h>
#include <ggml.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(e) do { const cudaError_t r_ = (e); if (r_ != cudaSuccess) { \
    std::printf("FAIL %s: %s\n", #e, cudaGetErrorString(r_)); std::exit(1); } } while (0)

int main() {
    constexpr int64_t FF = strata::prefill::lat_q3k_swiglu_ff;
    constexpr int64_t HQ = strata::prefill::lat_q3k_swiglu_hq_row;
    constexpr int E = 512;                        // entries (the engine's 4K shape: 512 entries a launch)
    constexpr int NB = (int) (FF / 32);           // 20 q8_1 blocks per entry (n_ff / 32 - the width that bit me in M1)
    std::printf("M2b: %d entries x %lld features, hq %lld B an entry (%d blocks)\n", E, (long long) FF, (long long) HQ, NB);

    // gate/up as M2a writes them: real products, entry-major, a spread of magnitudes so the amax varies per block
    std::mt19937 rng(23);
    std::normal_distribution<float> ng(0.f, 2.0f), nu(0.f, 1.5f);
    std::vector<float> gate((size_t) E * FF), up((size_t) E * FF);
    for (size_t i = 0; i < gate.size(); ++i) { gate[i] = ng(rng); up[i] = nu(rng); }
    // a few hard blocks: all zeros, one huge value, a negative gate, a near-tie (rounding boundary)
    gate[0] = 0.f; up[0] = 0.f;
    for (int f = 0; f < 32; ++f) { gate[(size_t) FF + f] = 0.f; up[(size_t) FF + f] = 0.f; }        // an all-zero block
    gate[(size_t) 2 * FF] = 12.5f; up[(size_t) 2 * FF] = 3.0f;                                       // one big value
    for (int f = 0; f < 32; ++f) gate[(size_t) 3 * FF + f] = -8.0f + 0.01f * f;                      // silu of negatives

    float* dgate = nullptr; float* dup = nullptr; uint8_t* dhq = nullptr; float* dhout = nullptr;
    CK(cudaMalloc(&dgate, gate.size() * 4)); CK(cudaMalloc(&dup, up.size() * 4));
    CK(cudaMalloc(&dhq, (size_t) E * HQ)); CK(cudaMalloc(&dhout, gate.size() * 4));
    CK(cudaMemcpy(dgate, gate.data(), gate.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dup, up.data(), up.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(dhq, 0xAA, (size_t) E * HQ));       // so an unwritten byte shows
    strata::prefill::lat_q3k_swiglu_q8(dgate, dup, E, dhq, dhout, nullptr);
    CK(cudaDeviceSynchronize());
    std::vector<float> hout(gate.size());
    std::vector<uint8_t> hq((size_t) E * HQ);
    CK(cudaMemcpy(hout.data(), dhout, hout.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hq.data(), dhq, hq.size(), cudaMemcpyDeviceToHost));

    // ---- gate 1: the SwiGLU floats against double silu(g) * u ------------------------------------------------
    {
        double sabs = 0, sref = 0, num = 0, aa = 0, bb = 0;
        float dmax = 0.f;
        int bad = 0;
        for (size_t i = 0; i < hout.size(); ++i) {
            const double g = gate[i], u = up[i];
            const double ref = (g / (1.0 + std::exp(-g))) * u;
            const double d = std::fabs((double) hout[i] - ref);
            sabs += d; sref += std::fabs(ref); dmax = std::fmax(dmax, (float) d);
            num += (double) hout[i] * ref; aa += (double) hout[i] * hout[i]; bb += ref * ref;
            if (d > 1e-6 * std::fabs(ref) + 1e-12) ++bad;
        }
        std::printf("swiglu: rel_mean %.3e  max|d| %.6e  cos %.9f  values off by >1e-6 rel: %d\n",
                    sref > 0 ? sabs / sref : 0.0, (double) dmax, num / (std::sqrt(aa) * std::sqrt(bb) + 1e-30), bad);
        if (!((sref > 0 ? sabs / sref : 1.0) <= 1e-6)) { std::printf("FAIL swiglu\n"); return 1; }
    }

    // ---- gate 2: the quantizer against q8_1_store's rule on the host, same floats ----------------------------
    {
        int code_bad = 0, d_bad = 0, blocks = 0, zeros = 0;
        double dmax_rel = 0;
        for (int e = 0; e < E; ++e) {
            const uint8_t* row = hq.data() + (size_t) e * HQ;
            for (int b = 0; b < NB; ++b) {
                const float* hv = hout.data() + (size_t) e * FF + 32 * b;
                float am = 0.f;
                for (int v = 0; v < 32; ++v) am = std::fmax(am, std::fabs(hv[v]));
                float d = am / 127.0f;                                  // q8_1_finite: the clamp is > 65504
                if (std::fabs(d) > 65504.0f) d = std::copysign(65504.0f, d);
                if (am == 0.f) ++zeros;
                const uint8_t* ch = row + (size_t) (b >> 1) * 80;
                uint16_t dh;
                std::memcpy(&dh, ch + 64 + 2 * (b & 1), 2);
                const float dmine = ggml_fp16_to_fp32(dh);
                dmax_rel = std::max(dmax_rel, std::fabs((double) dmine - (double) d) / (std::fabs((double) d) + 1e-30));
                if (dh != ggml_fp32_to_fp16(d)) ++d_bad;
                for (int v = 0; v < 32; ++v) {
                    const float q = am == 0.f ? 0.f : roundf(hv[v] / d);
                    const int8_t qc = (int8_t) (q > 127.f ? 127.f : q < -127.f ? -127.f : q);
                    const uint8_t got = ch[16 * (2 * (b & 1) + (v >> 4)) + 4 * (v & 3) + ((v & 15) >> 2)];
                    if ((int8_t) got != qc) ++code_bad;
                }
                ++blocks;
            }
        }
        std::printf("q8_1: %d blocks (%d all-zero)  code mismatches %d  d mismatches %d  max d rel %.3e\n", blocks, zeros,
                    code_bad, d_bad, dmax_rel);
        if (code_bad != 0 || d_bad != 0) { std::printf("FAIL quantizer\n"); return 1; }
    }

    // ---- informational: the dequantized H against the float H (the q8_1's own noise) ------------------------
    {
        double sabs = 0, sref = 0;
        for (int e = 0; e < E; ++e) {
            const uint8_t* row = hq.data() + (size_t) e * HQ;
            for (int b = 0; b < NB; ++b) {
                const uint8_t* ch = row + (size_t) (b >> 1) * 80;
                uint16_t dh;
                std::memcpy(&dh, ch + 64 + 2 * (b & 1), 2);
                const double d = ggml_fp16_to_fp32(dh);
                for (int v = 0; v < 32; ++v) {
                    const uint8_t c = ch[16 * (2 * (b & 1) + (v >> 4)) + 4 * (v & 3) + ((v & 15) >> 2)];
                    const double deq = d * (double) (int8_t) c;
                    const double x = hout[(size_t) e * FF + 32 * b + v];
                    sabs += std::fabs(deq - x); sref += std::fabs(x);
                }
            }
        }
        std::printf("hq dequantized vs the float H: rel_mean %.3e (q8_1's own resolution)\n", sabs / sref);
    }

    // ---- timing (informational) ------------------------------------------------------------------------------
    {
        cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
        strata::prefill::lat_q3k_swiglu_q8(dgate, dup, E, dhq, nullptr, nullptr);
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(t0));
        for (int i = 0; i < 100; ++i) strata::prefill::lat_q3k_swiglu_q8(dgate, dup, E, dhq, nullptr, nullptr);
        CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, t0, t1));
        std::printf("M2b swiglu+q8_1: %.3f ms / 100 = %.4f ms  %.3f us/entry (%d entries)\n", ms, ms / 100.0,
                    ms * 1000.0 / 100.0 / E, E);
        CK(cudaEventDestroy(t0)); CK(cudaEventDestroy(t1));
    }

    std::printf("PASS: M2b (SwiGLU + q8_1 of H)\n");
    return 0;
}
