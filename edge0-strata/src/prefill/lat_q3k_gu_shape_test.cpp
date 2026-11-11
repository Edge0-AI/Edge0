// src/prefill/lat_q3k_gu_shape_test.cpp - M2c step 1: the gu half on the same four shapes the down half was measured
// on (epg 4 / 32 / 200 / 1400, ~4000 entries a call), mine against native_expert_grouped's gate/up only.  No
// optimization: this is only here to tell a structural slowdown from a shape artefact.
#include "strata/prefill/lat_q3k_gu.hpp"
#include "strata/kernels/iq_kernels.hpp"

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
    constexpr int64_t ELL = 1280, FF = 640;
    constexpr int64_t GUB = 1280 * 550, DNB = ELL * FF / 32 * 18;
    constexpr int kMaxE = 4500, kMaxG = 1024, kXq = (int) (ELL / 32) * 36;   // 1440 B an entry
    constexpr int G = 128;

    // one real expert blob, replicated (distinct addresses: the reads are DRAM traffic)
    std::mt19937 rng(31);
    std::normal_distribution<float> nw(0.f, 0.05f);
    std::vector<uint8_t> blob((size_t) (GUB + DNB));
    {
        std::vector<float> w((size_t) 1280 * ELL);
        for (auto& v : w) v = nw(rng);
        if ((size_t) ggml_quantize_chunk(GGML_TYPE_Q3_K, w.data(), blob.data(), 0, 1280, ELL, nullptr) != (size_t) GUB) {
            std::printf("FAIL gu quantize\n"); return 1;
        }
        std::vector<float> wd((size_t) ELL * FF);
        for (auto& v : wd) v = nw(rng);
        if ((size_t) ggml_quantize_chunk(GGML_TYPE_IQ4_NL, wd.data(), blob.data() + GUB, 0, ELL, FF, nullptr) !=
            (size_t) DNB) { std::printf("FAIL dn quantize\n"); return 1; }
    }
    uint8_t* dall = nullptr;
    CK(cudaMalloc(&dall, (size_t) G * blob.size()));
    {
        std::vector<uint8_t> all((size_t) G * blob.size());
        for (int g = 0; g < G; ++g) std::memcpy(all.data() + (size_t) g * blob.size(), blob.data(), blob.size());
        CK(cudaMemcpy(dall, all.data(), all.size(), cudaMemcpyHostToDevice));
    }
    std::vector<const uint8_t*> bp((size_t) kMaxG);
    for (int g = 0; g < kMaxG; ++g) bp[(size_t) g] = dall + (size_t) (g % G) * blob.size();
    const uint8_t** dblob = nullptr;
    CK(cudaMalloc(&dblob, (size_t) kMaxG * sizeof(uint8_t*)));
    CK(cudaMemcpy(dblob, bp.data(), (size_t) kMaxG * sizeof(uint8_t*), cudaMemcpyHostToDevice));

    // a valid q8_1 input row per entry (40 blocks; any codes will do - the shape is what is measured)
    std::vector<uint8_t> xq((size_t) kMaxE * kXq);
    {
        std::uniform_int_distribution<int> u8(-127, 127);
        for (size_t e = 0; e < (size_t) kMaxE; ++e)
            for (int b = 0; b < (int) (ELL / 32); ++b) {
                uint8_t* blk = xq.data() + e * kXq + 36 * b;
                const uint16_t d16 = ggml_fp32_to_fp16(0.01f + 1e-5f * (e % 199));
                std::memcpy(blk, &d16, 2);
                const uint16_t s16 = 0;
                std::memcpy(blk + 2, &s16, 2);
                for (int i = 0; i < 32; ++i) blk[4 + i] = (uint8_t) u8(rng);
            }
    }
    uint8_t* dxq = nullptr;
    CK(cudaMalloc(&dxq, xq.size()));
    CK(cudaMemcpy(dxq, xq.data(), xq.size(), cudaMemcpyHostToDevice));

    float* dgate = nullptr; float* dup = nullptr;
    CK(cudaMalloc(&dgate, (size_t) kMaxE * FF * 4));
    CK(cudaMalloc(&dup, (size_t) kMaxE * FF * 4));
    const auto L = strata::kernels::native_expert_layout(GGML_TYPE_Q3_K, GGML_TYPE_IQ4_NL, ELL, FF);
    void* scr = nullptr;
    CK(cudaMalloc(&scr, strata::kernels::native_expert_scratch_bytes(kMaxE, FF)));
    int32_t *dgs = nullptr, *ded = nullptr, *det = nullptr, *dng = nullptr;
    CK(cudaMalloc(&dgs, (size_t) (kMaxG + 1) * 4));
    CK(cudaMalloc(&ded, (size_t) kMaxE * 4));
    CK(cudaMalloc(&det, (size_t) kMaxE * 4));
    CK(cudaMalloc(&dng, 4));

    std::printf("the gu shape sweep (gate/up only, us/entry):\n");
    const int shapes[][2] = {{4, 1000}, {32, 128}, {200, 20}, {1400, 3}};
    double worst = 0;
    for (const auto& sh : shapes) {
        const int epg = sh[0], G2 = sh[1], E2 = epg * G2;
        std::vector<int32_t> g2((size_t) G2 + 1), id((size_t) E2), ngv(1, G2);
        for (int g = 0; g <= G2; ++g) g2[(size_t) g] = g * epg;
        for (int e = 0; e < E2; ++e) id[(size_t) e] = e;
        CK(cudaMemcpy(dgs, g2.data(), g2.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(ded, id.data(), id.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(det, id.data(), id.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dng, ngv.data(), 4, cudaMemcpyHostToDevice));
        const int iters = E2 >= 4000 ? 20 : 50;
        strata::kernels::native_expert_set_mode(-1, 3);   // gate/up only
        auto time_mine = [&] {
            strata::prefill::lat_q3k_gu(dblob, dgs, G2, dxq, dgate, dup, FF, nullptr);
            CK(cudaDeviceSynchronize());
            cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
            CK(cudaEventRecord(t0));
            for (int i = 0; i < iters; ++i)
                strata::prefill::lat_q3k_gu(dblob, dgs, G2, dxq, dgate, dup, FF, nullptr);
            CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
            float ms = 0; CK(cudaEventElapsedTime(&ms, t0, t1));
            CK(cudaEventDestroy(t0)); CK(cudaEventDestroy(t1));
            return ms / iters;
        };
        auto time_native = [&] {
            strata::kernels::native_expert_grouped(L, (const unsigned long long*) dblob, dgs, dng, ded, det, G2, E2,
                                                   dxq, scr, dgate, nullptr, 0);
            CK(cudaDeviceSynchronize());
            cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
            CK(cudaEventRecord(t0));
            for (int i = 0; i < iters; ++i)
                strata::kernels::native_expert_grouped(L, (const unsigned long long*) dblob, dgs, dng, ded, det, G2, E2,
                                                       dxq, scr, dgate, nullptr, 0);
            CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
            float ms = 0; CK(cudaEventElapsedTime(&ms, t0, t1));
            CK(cudaEventDestroy(t0)); CK(cudaEventDestroy(t1));
            return ms / iters;
        };
        const double tm = time_mine(), tn = time_native();
        const double r = tn > 0 ? tm / tn : 0.0;
        worst = std::max(worst, r);
        std::printf("  epg %4d  groups %4d  E %5d | mine %.3f ms %6.3f us/entry | native %.3f ms %6.3f us/entry | "
                    "mine/native %.2fx\n", epg, G2, E2, tm, tm * 1000.0 / E2, tn, tn * 1000.0 / E2, r);
    }
    std::printf("worst shape: mine/native %.2fx\n", worst);
    std::printf("DONE: gu shape sweep\n");
    return 0;
}
