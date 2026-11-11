// tests/hip/prefill_gr_write_bench.cpp - the hyper-connection write-back (gr_write_norm_rs: R += bo * 2 sigmoid(inj / 4), then the next
// half's norm row scale and BF16 image) timed at the prompt path's shapes, with the bytes it moves, and a checksum of its
// outputs so a variant can be compared bit for bit (run twice with STRATA_GRW_V=0 / 1).
//   gr_write_bench [T=16384] [reps=20]
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {
constexpr int N = 2560, HC = 4, D = N * HC;
bool ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::printf("FAIL: %s: %s\n", what, cudaGetErrorString(e)); return false; }
    return true;
}
uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
}  // namespace

int main(int argc, char** argv) {
    const int64_t T = argc > 1 ? std::atoll(argv[1]) : 16384;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 20;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> R0((size_t) T * D), bo((size_t) T * N), inj((size_t) T * HC), w((size_t) HC * N);
    for (auto& v : R0) v = nd(rng);
    for (auto& v : bo) v = nd(rng);
    for (auto& v : inj) v = nd(rng);
    for (auto& v : w) v = 0.5f + 0.1f * nd(rng);
    float *dR = nullptr, *dR0 = nullptr, *dbo = nullptr, *dinj = nullptr, *dw = nullptr, *drs = nullptr;
    uint16_t* dx = nullptr;
    bool ok = ck(cudaMalloc(&dR, R0.size() * 4), "R") && ck(cudaMalloc(&dR0, R0.size() * 4), "R0") &&
              ck(cudaMalloc(&dbo, bo.size() * 4), "bo") && ck(cudaMalloc(&dinj, inj.size() * 4), "inj") &&
              ck(cudaMalloc(&dw, w.size() * 4), "w") && ck(cudaMalloc(&drs, (size_t) T * HC * 4), "rs") &&
              ck(cudaMalloc(&dx, (size_t) T * D * 2), "xn16");
    if (!ok) return 2;
    cudaMemcpy(dR0, R0.data(), R0.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dbo, bo.data(), bo.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dinj, inj.data(), inj.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice);
    cudaStream_t s;
    cudaStreamCreate(&s);
    // one reference pass for the checksum
    cudaMemcpy(dR, dR0, R0.size() * 4, cudaMemcpyDeviceToDevice);
    strata::prefill::gr_write_norm_rs(dR, dbo, dinj, HC, dw, 1e-6f, drs, dx, T, s, nullptr, D, false);
    cudaStreamSynchronize(s);
    std::vector<float> oR(R0.size()), ors((size_t) T * HC);
    std::vector<uint16_t> ox((size_t) T * D);
    cudaMemcpy(oR.data(), dR, oR.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(ors.data(), drs, ors.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(ox.data(), dx, ox.size() * 2, cudaMemcpyDeviceToHost);
    std::printf("checksum R %016llx rs %016llx xn16 %016llx\n", (unsigned long long) fnv(oR.data(), oR.size() * 4),
                (unsigned long long) fnv(ors.data(), ors.size() * 4), (unsigned long long) fnv(ox.data(), ox.size() * 2));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    float best = 1e30f, sum = 0;
    for (int r = 0; r < reps; ++r) {
        cudaMemcpy(dR, dR0, R0.size() * 4, cudaMemcpyDeviceToDevice);
        cudaStreamSynchronize(s);
        cudaEventRecord(e0, s);
        strata::prefill::gr_write_norm_rs(dR, dbo, dinj, HC, dw, 1e-6f, drs, dx, T, s, nullptr, D, false);
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        best = ms < best ? ms : best;
        sum += ms;
    }
    // bytes: R read + write, bo read (x4 rows, from cache after the first), inj, w, rs, xn16 write
    const double gb = ((double) T * D * 4 * 2 + (double) T * N * 4 + (double) T * D * 2) / 1e9;
    std::printf("gr_write_norm_rs T=%lld: best %.3f ms, mean %.3f ms, %.0f GB/s at best (%.2f GB moved, bo counted once)\n",
                (long long) T, best, sum / reps, gb / (best * 1e-3), gb);
    return 0;
}
