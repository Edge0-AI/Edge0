// src/kernels/nvfp4_expert_gpu_parity.cpp - the decode path's GPU experts (native_expert_grouped) on real NVFP4
// blobs with their global scales, against a double-precision expert.
//
//     nvfp4_expert_gpu_parity pack/experts.bin [layer expert ...]
//
// Eight tokens of N(0,1) activations through one expert: q8_1 input, gate/up with s_gate / s_up, SwiGLU, q8_1 of the
// hidden, down with s_down on its output. Only the two activation roundings separate it from the reference (~1%).
#include "strata/kernels/iq_kernels.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: nvfp4_expert_gpu_parity pack/experts.bin [layer expert ...]\n"); return 2; }
    constexpr int64_t N = 2560, FF = 640;
    constexpr int T = 8;
    const k::NativeExpertLayout L = k::native_expert_layout(40, 40, N, FF);
    const size_t layer_bytes = 512 * L.bytes;          // L.bytes counts the 16-byte tail
    std::vector<int> picks;
    for (int i = 2; i + 1 < argc; i += 2) { picks.push_back(std::atoi(argv[i])); picks.push_back(std::atoi(argv[i + 1])); }
    if (picks.empty()) picks = {0, 0, 0, 7, 12, 100, 24, 300, 40, 5, 47, 511};
    std::ifstream f(argv[1], std::ios::binary);
    std::mt19937 rng(9);
    std::normal_distribution<float> nd(0.f, 1.f);
    const ggml_type_traits* tt = ggml_get_type_traits(GGML_TYPE_NVFP4);
    double worst = 0;
    for (size_t pi = 0; pi < picks.size(); pi += 2) {
        const int layer = picks[pi], e = picks[pi + 1];
        std::vector<uint8_t> blob(L.bytes);
        f.seekg((std::streamoff) (layer * layer_bytes + (size_t) e * L.bytes));
        if (!f.read((char*) blob.data(), (std::streamsize) blob.size())) { std::fprintf(stderr, "cannot read\n"); return 2; }
        float tail[4];
        std::memcpy(tail, blob.data() + L.tail_off, sizeof tail);
        std::vector<float> wg((size_t) FF * N), wu((size_t) FF * N), wd((size_t) N * FF), x((size_t) T * N);
        tt->to_float(blob.data(), wg.data(), FF * N);
        tt->to_float(blob.data() + L.up_off, wu.data(), FF * N);
        tt->to_float(blob.data() + L.down_off, wd.data(), N * FF);
        for (float& v : x) v = nd(rng);
        std::vector<double> ref((size_t) T * N);
        for (int t = 0; t < T; ++t) {
            std::vector<double> h(FF);
            for (int64_t r = 0; r < FF; ++r) {
                double g = 0, u = 0;
                for (int64_t c = 0; c < N; ++c) {
                    g += (double) wg[(size_t) r * N + c] * x[(size_t) t * N + c];
                    u += (double) wu[(size_t) r * N + c] * x[(size_t) t * N + c];
                }
                g *= tail[0];
                u *= tail[1];
                h[r] = g / (1 + std::exp(-g)) * u;
            }
            for (int64_t r = 0; r < N; ++r) {
                double s = 0;
                for (int64_t c = 0; c < FF; ++c) s += (double) wd[(size_t) r * FF + c] * h[c];
                ref[(size_t) t * N + r] = s * tail[2];
            }
        }
        uint8_t* db; float *dx, *dout; void *dxq, *scr; unsigned long long* dptr; int32_t *dstart, *dng, *ddst, *dtok;
        ck(cudaMalloc(&db, blob.size()), "blob");
        ck(cudaMemcpy(db, blob.data(), blob.size(), cudaMemcpyHostToDevice), "blob");
        ck(cudaMalloc(&dx, x.size() * 4), "x");
        ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
        ck(cudaMalloc(&dxq, (size_t) T * N / 32 * 36), "xq");
        ck(cudaMalloc(&dout, (size_t) T * N * 4), "out");
        ck(cudaMalloc(&scr, k::native_expert_scratch_bytes(T, FF)), "scratch");
        const unsigned long long ptr = (unsigned long long) db;
        const int32_t start[2] = {0, T}, ng = 1, ids[T] = {0, 1, 2, 3, 4, 5, 6, 7};
        ck(cudaMalloc(&dptr, 8), "p"); ck(cudaMemcpy(dptr, &ptr, 8, cudaMemcpyHostToDevice), "p");
        ck(cudaMalloc(&dstart, 8), "s"); ck(cudaMemcpy(dstart, start, 8, cudaMemcpyHostToDevice), "s");
        ck(cudaMalloc(&dng, 4), "n"); ck(cudaMemcpy(dng, &ng, 4, cudaMemcpyHostToDevice), "n");
        ck(cudaMalloc(&ddst, 4 * T), "d"); ck(cudaMemcpy(ddst, ids, 4 * T, cudaMemcpyHostToDevice), "d");
        ck(cudaMalloc(&dtok, 4 * T), "t"); ck(cudaMemcpy(dtok, ids, 4 * T, cudaMemcpyHostToDevice), "t");
        k::quantize_q8_1_rows(dx, T, N, dxq, nullptr);
        k::native_expert_grouped(L, dptr, dstart, dng, ddst, dtok, 1, T, dxq, scr, dout, nullptr);
        ck(cudaDeviceSynchronize(), "run");
        std::vector<float> got((size_t) T * N);
        ck(cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost), "out");
        for (void* p : {(void*) db, (void*) dx, dxq, (void*) dout, scr, (void*) dptr, (void*) dstart, (void*) dng,
                        (void*) ddst, (void*) dtok})
            cudaFree(p);
        double num = 0, den = 0;
        for (size_t i = 0; i < got.size(); ++i) { num += (got[i] - ref[i]) * (got[i] - ref[i]); den += ref[i] * ref[i]; }
        const double rel = std::sqrt(num / den);
        worst = std::fmax(worst, rel);
        std::printf("layer %2d expert %3d: s_gate %.3g s_up %.3g s_down %.3g -> expert output rel. error %.3f%%\n", layer,
                    e, tail[0], tail[1], tail[2], 100 * rel);
    }
    std::printf("RESULT: %s (worst %.3f%%)\n", worst < 0.03 ? "ok" : "TOO FAR", 100 * worst);
    return worst < 0.03 ? 0 : 1;
}
