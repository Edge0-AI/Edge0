// route_window_parity - the verify window's batched routing against the per-token calls, bit for bit.
//
// `moe_route_window` replaces n per-token (projection + top-k) pairs by one multi-column BF16 projection and one
// n-token top-k.  Speculative decoding is exact only if a token's route does not depend on how many tokens share
// its window, so both batched kernels must equal their one-token forms exactly - this checks that on random data
// shaped like the router (2560 -> 256, top-10) and like the shared-expert gate (2560 -> 1).
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/router_top10.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}
uint16_t bf16(float f) { uint32_t i; std::memcpy(&i, &f, 4); return (uint16_t) ((i + ((i >> 16) & 1u) + 0x7FFFu) >> 16); }
}  // namespace

int main() {
    const int N = 2560, E = 256, K = 10;
    int bad = 0;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int n_out : {E, 1}) {
        for (int T : {2, 3, 4, 6}) {
            std::vector<uint16_t> w((size_t) n_out * N);
            std::vector<float> x((size_t) T * N);
            for (auto& v : w) v = bf16(nd(rng) * 0.05f);
            for (auto& v : x) v = nd(rng);
            uint16_t* dw; float *dx, *dy1, *dyn;
            ck(cudaMalloc(&dw, w.size() * 2), "w"); ck(cudaMalloc(&dx, x.size() * 4), "x");
            ck(cudaMalloc(&dy1, (size_t) T * n_out * 4), "y1"); ck(cudaMalloc(&dyn, (size_t) T * n_out * 4), "yn");
            ck(cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice), "up w");
            ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "up x");
            for (int t = 0; t < T; ++t)
                strata::kernels::bf16_gemv_fp32_mmvf(dx + (size_t) t * N, dw, dy1 + (size_t) t * n_out, N, n_out, nullptr);
            strata::kernels::bf16_gemv_fp32_mmvf_cols(dx, dw, dyn, N, n_out, T, nullptr);
            ck(cudaDeviceSynchronize(), "sync");
            std::vector<float> y1((size_t) T * n_out), yn((size_t) T * n_out);
            ck(cudaMemcpy(y1.data(), dy1, y1.size() * 4, cudaMemcpyDeviceToHost), "down 1");
            ck(cudaMemcpy(yn.data(), dyn, yn.size() * 4, cudaMemcpyDeviceToHost), "down n");
            const bool same = std::memcmp(y1.data(), yn.data(), y1.size() * 4) == 0;
            std::printf("  mmvf cols  n_out %3d  T %d  %s\n", n_out, T, same ? "bit-identical" : "*** DIFFERS ***");
            if (!same) ++bad;
            if (n_out == E) {
                int *i1, *in_; float *w1, *wn;
                ck(cudaMalloc(&i1, (size_t) T * K * 4), "i1"); ck(cudaMalloc(&in_, (size_t) T * K * 4), "in");
                ck(cudaMalloc(&w1, (size_t) T * K * 4), "w1"); ck(cudaMalloc(&wn, (size_t) T * K * 4), "wn");
                for (int t = 0; t < T; ++t)
                    strata::kernels::router_top10(dy1 + (size_t) t * E, 1, E, K, i1 + t * K, w1 + t * K, nullptr);
                strata::kernels::router_top10(dy1, T, E, K, in_, wn, nullptr);
                ck(cudaDeviceSynchronize(), "sync top");
                std::vector<int> a((size_t) T * K), b((size_t) T * K);
                std::vector<float> fa((size_t) T * K), fb((size_t) T * K);
                ck(cudaMemcpy(a.data(), i1, a.size() * 4, cudaMemcpyDeviceToHost), "ids1");
                ck(cudaMemcpy(b.data(), in_, b.size() * 4, cudaMemcpyDeviceToHost), "idsn");
                ck(cudaMemcpy(fa.data(), w1, fa.size() * 4, cudaMemcpyDeviceToHost), "w1");
                ck(cudaMemcpy(fb.data(), wn, fb.size() * 4, cudaMemcpyDeviceToHost), "wn");
                const bool s2 = a == b && std::memcmp(fa.data(), fb.data(), fa.size() * 4) == 0;
                std::printf("  top10 n    T %d            %s\n", T, s2 ? "bit-identical" : "*** DIFFERS ***");
                if (!s2) ++bad;
                cudaFree(i1); cudaFree(in_); cudaFree(w1); cudaFree(wn);
            }
            cudaFree(dw); cudaFree(dx); cudaFree(dy1); cudaFree(dyn);
        }
    }
    std::printf(bad ? "route_window_parity: %d FAILURES\n" : "route_window_parity OK\n", bad);
    return bad ? 1 : 0;
}
