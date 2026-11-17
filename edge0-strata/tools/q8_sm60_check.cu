// tools/q8_sm60_check.cu - parity and speed of the STRATA_Q8_SM60=1 kernel (src/kernels/cuda/q8_sm60.cuh) on the
// Q8_0 shapes of Qwen3.8-Flash-Next (hidden 2560): the dense projections, the shared expert and the head.
//
// It includes the committed header, so the check covers the code the engine compiles, not a copy.  Parity: random
// Q8_0 weights in GGUF blocks -> q8sm60::pack_kernel -> the kernel, against a CPU double reference of
// sum over blocks of d_w * d_x * dot_int8, for 1..8 columns (max relative error < 1e-5).  Speed: a 1 GiB rotation so
// the weights come from DRAM, as in the engine.  No CMake target; build by hand:
//
//   nvcc -O3 -arch=sm_60 -std=c++17 -Iinclude -Isrc/kernels/cuda tools/q8_sm60_check.cu -o q8_sm60_check && ./q8_sm60_check
//
// Exit status 0 = every case within tolerance and no CUDA error.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s at %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)
struct Q81Block { half2 ds; int8_t qs[32]; };
struct Q80Block { half d; int8_t qs[32]; };
#include "strata/kernels/dp4a.hpp"
#include "q8_sm60.cuh"

static float h2f(half h) { return __half2float(h); }

static bool parity(int n_in, int n_out, int ncols) {
    const int nb = n_in / 32;
    std::vector<Q80Block> w((size_t) n_out * nb);
    std::vector<Q81Block> x((size_t) ncols * nb);
    srand(n_in * 7 + n_out + ncols);
    for (auto& b : w) { b.d = __float2half((rand() % 1000 + 1) * 1e-5f); for (auto& q : b.qs) q = (int8_t) (rand() % 255 - 127); }
    for (auto& b : x) { b.ds = __halves2half2(__float2half((rand() % 1000 + 1) * 1e-4f), __float2half(0.f)); for (auto& q : b.qs) q = (int8_t) (rand() % 255 - 127); }
    std::vector<double> ref((size_t) ncols * n_out);
    for (int j = 0; j < ncols; ++j) for (int r = 0; r < n_out; ++r) {
        double s = 0;
        for (int b = 0; b < nb; ++b) {
            const Q80Block& wb = w[(size_t) r * nb + b]; const Q81Block& xb = x[(size_t) j * nb + b];
            int dot = 0; for (int i = 0; i < 32; ++i) dot += wb.qs[i] * xb.qs[i];
            s += (double) h2f(wb.d) * __low2float(xb.ds) * dot;
        }
        ref[(size_t) j * n_out + r] = s;
    }
    Q80Block* dw; Q81Block* dx; float* dy; int8_t* qs;
    CK(cudaMalloc(&dw, w.size() * sizeof(Q80Block))); CK(cudaMemcpy(dw, w.data(), w.size() * sizeof(Q80Block), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dx, x.size() * sizeof(Q81Block))); CK(cudaMemcpy(dx, x.data(), x.size() * sizeof(Q81Block), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dy, ref.size() * 4)); CK(cudaMemset(dy, 0xff, ref.size() * 4));
    const size_t count = (size_t) n_out * nb;
    CK(cudaMalloc(&qs, count * 34));
    half* d = reinterpret_cast<half*>(qs + (size_t) n_in * n_out);
    q8sm60::pack_kernel<<<unsigned((count + 255) / 256), 256>>>(dw, qs, d, count);
    const bool took = q8sm60::launch(qs, d, dx, dy, n_in, n_out, ncols, 0);
    CK(cudaDeviceSynchronize());
    bool ok = true;
    if (took) {
        std::vector<float> y(ref.size()); CK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
        double m = 0, e = 0; for (size_t i = 0; i < y.size(); ++i) { m = std::max(m, std::fabs(ref[i])); e = std::max(e, std::fabs(y[i] - ref[i])); }
        ok = e / m < 1e-5 && std::isfinite(e);
        if (!ok) printf("  FAIL %d -> %d, %d cols: max rel err %.2e\n", n_in, n_out, ncols, e / m);
    }
    cudaFree(dw); cudaFree(dx); cudaFree(dy); cudaFree(qs);
    return ok;
}

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("%s, %d SMs, %d MHz\n", p.name, p.multiProcessorCount, p.clockRate / 1000);
    struct S { const char* name; int n_in, n_out; } shapes[] = {
        {"gdn attn_qkv", 2560, 10240}, {"gdn attn_gate", 2560, 6144}, {"ssm_out / attn_output", 6144, 2560},
        {"qsa attn_q", 2560, 12288}, {"qsa attn_k / v", 2560, 512}, {"shexp gate / up", 2560, 640},
        {"shexp down", 640, 2560}, {"head", 2560, 248320}};
    bool ok = true; int checked = 0, declined = 0;
    for (const S& s : shapes) for (int c = 1; c <= 8; ++c) {
        if (s.n_out > 20000 && c > 2) continue;   // the head's CPU reference is slow; 1-2 columns cover its path
        if (q8sm60::smem_bytes(c, s.n_in) > q8sm60::MAX_SMEM) { ++declined; continue; }
        ok = parity(s.n_in, s.n_out, c) && ok; ++checked;
    }
    printf("parity: %d shape x column cases %s (%d declined: shared memory)\n\n", checked, ok ? "PASS (max rel err < 1e-5)" : "FAIL", declined);
    if (!ok) return 1;

    const size_t POOL = 1ull << 30;
    int8_t* pool; CK(cudaMalloc(&pool, POOL)); CK(cudaMemset(pool, 1, POOL));
    Q81Block* dx; CK(cudaMalloc(&dx, 8 * (6144 / 32) * sizeof(Q81Block))); CK(cudaMemset(dx, 0, 8 * (6144 / 32) * sizeof(Q81Block)));
    float* dy; CK(cudaMalloc(&dy, (size_t) 8 * 248320 * 4));
    int8_t* head; const size_t head_bytes = (size_t) 248320 * 2560 / 32 * 34; CK(cudaMalloc(&head, head_bytes)); CK(cudaMemset(head, 0, head_bytes));
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    printf("%-22s %6s %7s |", "matrix", "n_in", "n_out"); for (int c = 1; c <= 6; ++c) printf("   T=%d us  GB/s", c); printf("\n");
    for (const S& s : shapes) {
        printf("%-22s %6d %7d |", s.name, s.n_in, s.n_out);
        const size_t bytes = (size_t) s.n_in / 32 * 34 * s.n_out;
        for (int c = 1; c <= 6; ++c) {
            if (q8sm60::smem_bytes(c, s.n_in) > q8sm60::MAX_SMEM) { printf("  %14s", "-"); continue; }
            const int iters = s.n_out > 20000 ? 10 : 100;
            auto at = [&](int i) -> int8_t* { if (s.n_out > 20000) return head; const size_t span = (POOL - bytes) / 256;
                                              return pool + ((size_t) i * ((bytes + 255) / 256) % span) * 256; };
            for (int i = 0; i < 3; ++i) { int8_t* q = at(i); q8sm60::launch(q, (half*) (q + (size_t) s.n_in * s.n_out), dx, dy, s.n_in, s.n_out, c, 0); }
            CK(cudaEventRecord(a));
            for (int i = 0; i < iters; ++i) { int8_t* q = at(i + 3); q8sm60::launch(q, (half*) (q + (size_t) s.n_in * s.n_out), dx, dy, s.n_in, s.n_out, c, 0); }
            CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
            float ms; CK(cudaEventElapsedTime(&ms, a, b)); const double us = ms * 1000.0 / iters;
            printf("  %7.1f %6.0f", us, bytes / (us * 1e3));
        }
        printf("\n");
    }
    return 0;
}
