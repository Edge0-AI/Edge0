// src/kernels/q8_gemv_bench.cpp - MODEL-UPDATE: the dense Q8_0 GEMVs of Unsloth's files (attention, SSM, shared expert,
// hyper-connections, head: all Q8_0 there) at the engine's shapes, 1..6 columns (a verify window's tokens), through
// native_mmvq - the call the engine makes - against the card's memory bandwidth.  Enough copies of each matrix are
// cycled to miss the GPU's last-level cache, as a layer's call does.
//
//     build/q8_gemv_bench [--cols 1,2,3,4,6] [--iters 50] [--bw GB/s]
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace K = strata::kernels;

namespace {
struct Shape { const char* what; int n_in, n_out; };
const Shape SHAPES[] = {
    {"GDN qkv 2560->10240", 2560, 10240},  {"GDN gate 2560->6144", 2560, 6144}, {"GDN/QSA out 6144->2560", 6144, 2560},
    {"QSA q 2560->12288", 2560, 12288},    {"QSA k/v 2560->512", 2560, 512},    {"HC down 10240->320", 10240, 320},
    {"HC up 320->10240", 320, 10240},      {"shared gate/up 2560->640", 2560, 640}, {"shared down 640->2560", 640, 2560},
    {"head 2560->248320", 2560, 248320},
};
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(1); }
}
std::vector<int> ints(const std::string& s) {
    std::vector<int> v;
    size_t from = 0;
    for (;;) {
        const size_t c = s.find(',', from);
        v.push_back(std::atoi(s.substr(from, c == std::string::npos ? std::string::npos : c - from).c_str()));
        if (c == std::string::npos) break;
        from = c + 1;
    }
    return v;
}
}  // namespace

int main(int argc, char** argv) {
    std::vector<int> cols = {1, 2, 3, 4, 6};
    int iters = 50;
    double bw = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--cols" && i + 1 < argc) cols = ints(argv[++i]);
        else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (a == "--bw" && i + 1 < argc) bw = std::atof(argv[++i]);
    }
    cudaDeviceProp prop;
    ck(cudaGetDeviceProperties(&prop, 0), "props");
    std::printf("q8_gemv_bench: %s\n", prop.name);
    std::mt19937 rng(7);
    for (const Shape& s : SHAPES) {
        const size_t bytes = (size_t) s.n_out * (s.n_in / 32) * 34;
        const int copies = (int) std::max<size_t>(1, std::min<size_t>(16, ((size_t) 512 << 20) / bytes));
        std::vector<uint8_t> host(bytes);
        for (size_t b = 0; b < bytes; b += 34) {   // fp16 scale 0.01 (0x211f), then random int8 weights
            host[b] = 0x1f;
            host[b + 1] = 0x21;
            for (int k = 0; k < 32; ++k) host[b + 2 + k] = (uint8_t) (rng() & 0xff);
        }
        std::vector<void*> w(copies);
        for (auto& p : w) {
            ck(cudaMalloc(&p, bytes), "malloc w");
            ck(cudaMemcpy(p, host.data(), bytes, cudaMemcpyHostToDevice), "copy w");
        }
        const int maxc = *std::max_element(cols.begin(), cols.end());
        std::vector<float> xh((size_t) s.n_in * maxc);
        for (auto& v : xh) v = std::normal_distribution<float>(0.f, 1.f)(rng);
        float *x = nullptr, *y = nullptr;
        void* xq = nullptr;
        ck(cudaMalloc(&x, xh.size() * 4), "x");
        ck(cudaMemcpy(x, xh.data(), xh.size() * 4, cudaMemcpyHostToDevice), "x copy");
        ck(cudaMalloc(&xq, K::native_q8_1_bytes(s.n_in, maxc)), "xq");
        ck(cudaMalloc(&y, (size_t) s.n_out * maxc * 4), "y");
        cudaStream_t st;
        ck(cudaStreamCreate(&st), "stream");
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        std::printf("\n%s: %.2f MB, %d copies\n", s.what, bytes / 1e6, copies);
        for (int c : cols) {
            K::native_quantize_q8_1(x, xq, s.n_in, c, st);
            for (int i = 0; i < 3; ++i) K::native_mmvq(8, w[i % copies], xq, y, s.n_in, s.n_out, c, st);
            cudaEventRecord(e0, st);
            for (int i = 0; i < iters; ++i) K::native_mmvq(8, w[i % copies], xq, y, s.n_in, s.n_out, c, st);
            cudaEventRecord(e1, st);
            ck(cudaEventSynchronize(e1), "sync");
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            const double us = 1000.0 * ms / iters, gbs = bytes / (us * 1e3);
            std::printf("  %d col%s: %8.1f us  %7.1f GB/s%s\n", c, c == 1 ? " " : "s", us, gbs,
                        bw > 0 ? (" = " + std::to_string((int) (100 * gbs / bw + 0.5)) + "% of bw").c_str() : "");
        }
        for (auto p : w) cudaFree(p);
        cudaFree(x);
        cudaFree(xq);
        cudaFree(y);
        cudaStreamDestroy(st);
    }
    return 0;
}
