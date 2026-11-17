// src/prefill/gemm_dequant_side_parity.cu - STRATA_DEQUANT_SIDE: Gemm::native with the dense-weight dequant on a side
// stream into two scratches (Gemm::set_dequant_side) against the serial form, bitwise (GPU, synthetic, no model).
//
// A run of dense projections (Q4_K and Q6_K weights of the engine's shapes, one of them larger than the scratch so it takes
// the sliced path, which keeps the serial form) is executed twice from the same inputs: once serially, once with the side
// stream.  Every output must be identical bit for bit.  Also a run with a long queue of compute work ahead of each GEMM, so
// the side dequant really overlaps it.  Exit 77 without a CUDA device.
#include "strata/prefill/gemm.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
__global__ void fill_f16(__half* p, int64_t n, unsigned seed, float scale) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned x = (unsigned) i * 2654435761u + seed;
    x ^= x >> 16; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
    p[i] = __float2half(((float) (x & 0xffff) / 32768.0f - 1.0f) * scale);
}
// busy work on the compute stream ahead of a GEMM
__global__ void spin(float* p, int64_t n, int iters) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = p[i];
    for (int k = 0; k < iters; ++k) v = v * 1.0000001f + 1e-7f;
    p[i] = v;
}
bool ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) std::printf("FAIL: %s: %s\n", what, cudaGetErrorString(e));
    return e == cudaSuccess;
}
uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}
struct Weight { int type, bytes_per_block; int at[2]; int64_t N, K; void* dev = nullptr; };
}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev < 1) { std::printf("no CUDA device (skipped)\n"); return 77; }
    cudaStream_t st = nullptr;
    if (!ok(cudaStreamCreate(&st), "stream")) return 1;
    const int64_t T = 512;
    const int64_t scratch_elems = 2560LL * 4096;   // 6144 x 2560 does not fit: the sliced path
    strata::prefill::Gemm gemm;
    std::string err;
    if (!gemm.init(st, scratch_elems, err)) { std::printf("FAIL: %s\n", err.c_str()); return 1; }
    uint16_t* scratch2 = nullptr;
    if (!ok(cudaMalloc((void**) &scratch2, (size_t) scratch_elems * 2), "scratch2")) return 1;

    std::vector<Weight> ws = {
        {12, 144, {0, 2}, 10240, 2560}, {14, 210, {208, -1}, 2560, 6144}, {12, 144, {0, 2}, 6144, 2560},
        {14, 210, {208, -1}, 3072, 2560}, {12, 144, {0, 2}, 2560, 4096}, {13, 176, {0, 2}, 2560, 2560},
        {12, 144, {0, 2}, 512, 2560},    {14, 210, {208, -1}, 12288, 2560},
    };
    std::mt19937 rng(20261026u);
    for (Weight& w : ws) {
        const size_t blocks = (size_t) (w.N * w.K / 256), bytes = blocks * (size_t) w.bytes_per_block;
        std::vector<uint8_t> h(bytes);
        for (auto& b : h) b = (uint8_t) (rng() & 0xff);
        for (size_t k = 0; k < blocks; ++k)
            for (int at : w.at)
                if (at >= 0) { const uint16_t v = sane_half(rng); std::memcpy(&h[k * (size_t) w.bytes_per_block + (size_t) at], &v, 2); }
        if (!ok(cudaMalloc(&w.dev, bytes), "W") || !ok(cudaMemcpy(w.dev, h.data(), bytes, cudaMemcpyHostToDevice), "W copy")) return 1;
    }
    int64_t max_k = 0, max_n = 0;
    for (const Weight& w : ws) { max_k = std::max(max_k, w.K); max_n = std::max(max_n, w.N); }
    uint16_t* X = nullptr;
    float *busy = nullptr, *Y[2] = {nullptr, nullptr};
    const int64_t busy_n = 1 << 22;
    if (!ok(cudaMalloc((void**) &X, (size_t) T * max_k * 2), "X") || !ok(cudaMalloc((void**) &busy, (size_t) busy_n * 4), "busy") ||
        !ok(cudaMalloc((void**) &Y[0], (size_t) T * max_n * 4 * ws.size()), "Y0") ||
        !ok(cudaMalloc((void**) &Y[1], (size_t) T * max_n * 4 * ws.size()), "Y1"))
        return 1;
    fill_f16<<<(unsigned) ((T * max_k + 255) / 256), 256, 0, st>>>((__half*) X, T * max_k, 17u, 2.0f);
    cudaMemset(busy, 0, (size_t) busy_n * 4);
    if (!ok(cudaStreamSynchronize(st), "fill")) return 1;

    int bad = 0;
    for (int busy_iters : {0, 20000}) {
        for (int side = 0; side < 2; ++side) {
            std::string e2;
            if (!gemm.set_dequant_side(side ? scratch2 : nullptr, e2)) { std::printf("FAIL: set_dequant_side: %s\n", e2.c_str()); return 1; }
            cudaMemset(Y[side], 0xff, (size_t) T * max_n * 4 * ws.size());
            for (int round = 0; round < 3; ++round)   // the scratch parity of the second round differs from the first
                for (size_t i = 0; i < ws.size(); ++i) {
                    if (busy_iters) spin<<<(unsigned) ((busy_n + 255) / 256), 256, 0, st>>>(busy, busy_n, busy_iters);
                    gemm.native(X, ws[i].type, ws[i].dev, Y[side] + (size_t) i * T * max_n, T, ws[i].N, ws[i].K, max_n, 0.0f);
                }
            if (!ok(cudaStreamSynchronize(st), "run") || !ok(cudaDeviceSynchronize(), "device")) return 1;
        }
        std::vector<uint8_t> a((size_t) T * max_n * 4 * ws.size()), b(a.size());
        cudaMemcpy(a.data(), Y[0], a.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(), Y[1], b.size(), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < ws.size(); ++i) {
            // the written part of the output: T rows of N floats at stride max_n
            long long diff = 0;
            for (int64_t r = 0; r < T; ++r)
                if (std::memcmp(&a[((size_t) i * T * max_n + (size_t) r * max_n) * 4], &b[((size_t) i * T * max_n + (size_t) r * max_n) * 4],
                                (size_t) ws[i].N * 4) != 0)
                    ++diff;
            const bool sliced = ws[i].N * ws[i].K > scratch_elems;
            std::printf("  busy %-6d type %2d N %5lld K %4lld%s: %s\n", busy_iters, ws[i].type, (long long) ws[i].N, (long long) ws[i].K,
                        sliced ? " (sliced)" : "", diff ? "*** DIFFERS ***" : "bitwise equal");
            if (diff) ++bad;
        }
    }
    gemm.set_dequant_side(nullptr, err);
    std::printf("gemm_dequant_side_parity: %s\n", bad ? "FAILED" : "OK");
    return bad ? 1 : 0;
}
