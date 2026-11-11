// src/kernels/latent_bench.cpp - D/A/U timing through the real latent packs.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/latent_dense.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace k = strata::kernels;

#define ck(x, m) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", m, cudaGetErrorString(e_)); std::exit(2); } } while (0)

int main(int argc, char** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage: %s pack_dir dense.bin layer T\n", argv[0]); return 2; }
    const int layer = std::atoi(argv[3]);
    const int T = std::atoi(argv[4]);
    const int K = 10, G = 8, E = 512, H = 2560, ELL = 1792, I = 640;
    const int cap = T * K;
    std::string err;
    if (!strata::kernels::cpu::expert_layout_load(argv[1], 48, 512, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
    strata::core::FileExpertSource src;
    if (!src.open(argv[1], 48, 512, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
    strata::core::LatentDense dense;
    if (!dense.open(argv[2], 48, G, H, ELL, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
    const k::NativeExpertLayout L = k::native_expert_layout(40, 40, ELL, I);

    std::vector<float> x((size_t) T * H);
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) v = nd(rng);
    std::vector<int32_t> topk((size_t) T * K), labels(E), start(E + 1, 0), pos(E), dst(cap), tok(cap), counts(E, 0);
    for (int e = 0; e < E; ++e) labels[e] = e % G;
    for (int i = 0; i < T * K; ++i) topk[i] = (int32_t) (i % E);
    for (int r = 0; r < cap; ++r) counts[topk[r]]++;
    for (int e = 0; e < E; ++e) start[e + 1] = start[e] + counts[e];
    pos = start;
    for (int r = 0; r < cap; ++r) { const int e = topk[r]; const int p = pos[e]++; dst[p] = r; tok[p] = (r / K) * G + labels[e]; }

    float *dx = nullptr, *dD = nullptr, *dU = nullptr, *dXl = nullptr, *dYl = nullptr, *dY = nullptr;
    float *dUIn = nullptr, *dUOut = nullptr;
    void *dxq = nullptr, *dxlq = nullptr, *dUq = nullptr, *dscr = nullptr;
    uint8_t* dA = nullptr;
    unsigned long long* dPtr = nullptr;
    int32_t *dStart = nullptr, *dN = nullptr, *dDst = nullptr, *dTok = nullptr;
    ck(cudaMalloc(&dx, (size_t) T * H * 4), "x");
    ck(cudaMalloc(&dxq, (size_t) T * (H / 32) * 36), "xq");
    ck(cudaMalloc(&dD, dense.d_bytes()), "D");
    ck(cudaMalloc(&dU, dense.u_bytes()), "U");
    ck(cudaMalloc(&dXl, (size_t) T * G * ELL * 4), "xl");
    ck(cudaMalloc(&dxlq, (size_t) T * G * (ELL / 32) * 36), "xlq");
    ck(cudaMalloc(&dA, (size_t) E * L.bytes), "A");
    ck(cudaMalloc(&dPtr, (size_t) E * 8), "ptr");
    ck(cudaMalloc(&dStart, (size_t) (E + 1) * 4), "start");
    ck(cudaMalloc(&dN, 4), "n");
    ck(cudaMalloc(&dDst, (size_t) cap * 4), "dst");
    ck(cudaMalloc(&dTok, (size_t) cap * 4), "tok");
    ck(cudaMalloc(&dYl, (size_t) cap * ELL * 4), "yl");
    ck(cudaMalloc(&dY, (size_t) cap * H * 4), "y");
    ck(cudaMalloc(&dUIn, (size_t) cap * ELL * 4), "uin");
    ck(cudaMalloc(&dUOut, (size_t) cap * H * 4), "uout");
    ck(cudaMalloc(&dUq, (size_t) cap * (ELL / 32) * 36), "uq");
    ck(cudaMalloc(&dscr, k::native_expert_scratch_bytes(cap, I)), "scratch");

    ck(cudaMemcpy(dx, x.data(), (size_t) T * H * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(dD, dense.d(layer), dense.d_bytes(), cudaMemcpyHostToDevice), "D");
    ck(cudaMemcpy(dU, dense.u(layer), dense.u_bytes(), cudaMemcpyHostToDevice), "U");
    for (int e = 0; e < E; ++e) {
        const uint8_t* p = src.blob(layer, e);
        if (!p) { std::fprintf(stderr, "null blob %d\n", e); return 2; }
        ck(cudaMemcpy(dA + (size_t) e * L.bytes, p, L.bytes, cudaMemcpyHostToDevice), "A");
    }
    std::vector<unsigned long long> ptr(E);
    for (int e = 0; e < E; ++e) ptr[e] = (unsigned long long) (dA + (size_t) e * L.bytes);
    int32_t ng = E;
    ck(cudaMemcpy(dPtr, ptr.data(), (size_t) E * 8, cudaMemcpyHostToDevice), "ptr");
    ck(cudaMemcpy(dStart, start.data(), (size_t) (E + 1) * 4, cudaMemcpyHostToDevice), "start");
    ck(cudaMemcpy(dN, &ng, 4, cudaMemcpyHostToDevice), "n");
    ck(cudaMemcpy(dDst, dst.data(), (size_t) cap * 4, cudaMemcpyHostToDevice), "dst");
    ck(cudaMemcpy(dTok, tok.data(), (size_t) cap * 4, cudaMemcpyHostToDevice), "tok");

    auto run_d = [&] { k::quantize_q8_1_rows(dx, T, H, dxq, nullptr); k::iq_mmvq(8, dD, dxq, dXl, H, G * ELL, T, nullptr); };
    auto run_a = [&] { k::native_expert_grouped(L, dPtr, dStart, dN, dDst, dTok, E, cap, dxlq, dscr, dYl, nullptr, E); };
    auto run_u = [&] {
        for (int g = 0; g < G; ++g) {
            int n = 0;
            for (int r = 0; r < cap; ++r) if (labels[topk[r]] == g) { ck(cudaMemcpy(dUIn + (size_t) n * ELL, dYl + (size_t) r * ELL, (size_t) ELL * 4, cudaMemcpyDeviceToDevice), "gather"); ++n; }
            if (!n) continue;
            k::quantize_q8_1_rows(dUIn, n, ELL, dUq, nullptr);
            k::iq_mmvq(8, (const uint8_t*) dU + (size_t) g * H * dense.u_row_bytes(), dUq, dUOut, ELL, H, n, nullptr);
            for (int i = 0; i < n; ++i) ck(cudaMemcpy(dY + (size_t) i * H, dUOut + (size_t) i * H, (size_t) H * 4, cudaMemcpyDeviceToDevice), "scatter");
        }
    };
    // warmup
    run_d(); k::quantize_q8_1_rows(dXl, T * G, ELL, dxlq, nullptr); run_a(); run_u(); ck(cudaDeviceSynchronize(), "warm");
    const int reps = 20;
    cudaEvent_t e0, e1; ck(cudaEventCreate(&e0), "ev"); ck(cudaEventCreate(&e1), "ev");
    auto timeit = [&](auto&& fn) {
        ck(cudaEventRecord(e0), "rec");
        for (int i = 0; i < reps; ++i) fn();
        ck(cudaEventRecord(e1), "rec"); ck(cudaEventSynchronize(e1), "sync");
        float ms = 0; ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
        return ms / reps;
    };
    const float dms = timeit(run_d);
    const float ams = timeit([&] { k::quantize_q8_1_rows(dXl, T * G, ELL, dxlq, nullptr); run_a(); });
    const float ums = timeit(run_u);
    std::printf("T=%d cap=%d D %.3f ms  A %.3f ms  U %.3f ms  total %.3f ms\n", T, cap, dms, ams, ums, dms + ams + ums);
    return 0;
}
