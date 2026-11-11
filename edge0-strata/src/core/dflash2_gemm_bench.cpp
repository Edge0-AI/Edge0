// src/core/dflash2_gemm_bench.cpp - the drafter's Q8 GEMM (kernels/dflash2.hpp df2_gemm) on random planar weights: GB/s per shape and column count.
//
//     dflash2_gemm_bench [iters]
//
// Weight sets are cycled through 2 GiB of distinct matrices so the L2 cache cannot hold them (the drafter reads each matrix once per round).
#include "strata/kernels/dflash2.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 50;
    struct Shape { const char* name; int n_in, n_out; int mats; };
    const Shape shapes[] = {{"gate+up 2560->7680 x2", 2560, 7680, 2}, {"down 7680->2560", 7680, 2560, 1}, {"qkv 2560->(4096,1024,1024)", 2560, 4096, 3},
                            {"o 4096->2560", 4096, 2560, 1}, {"fc 12800->2560", 12800, 2560, 1}, {"kv 2560->1024 x2", 2560, 1024, 2}};
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (const Shape& sh : shapes) {
        const size_t wbytes = (size_t) sh.n_out * sh.n_in * 34 / 32;
        const size_t per_set = wbytes * sh.mats;
        const int sets = (int) std::max<size_t>(2, ((size_t) 1 << 30) / per_set);
        std::vector<uint8_t*> w((size_t) sets * sh.mats);
        for (auto& p : w) {
            if (cudaMalloc((void**) &p, wbytes) != cudaSuccess) { std::fprintf(stderr, "alloc\n"); return 1; }
            cudaMemset(p, 0x11, wbytes);
        }
        uint8_t* xq;
        float *xd, *y;
        cudaMalloc((void**) &xq, (size_t) sh.n_in * 8);
        cudaMalloc((void**) &xd, (size_t) sh.n_in / 32 * 8 * 4);
        cudaMalloc((void**) &y, (size_t) 8 * sh.n_out * 3 * 4);
        cudaMemset(xq, 3, (size_t) sh.n_in * 8);
        std::vector<float> ones((size_t) sh.n_in / 32 * 8, 0.01f);
        cudaMemcpy(xd, ones.data(), ones.size() * 4, cudaMemcpyHostToDevice);
        for (int var : {1, 2, 3, 4, 5, 6, 7, 8, 9}) {
            strata::kernels::df2_gemm_variant(var);
            const int ncols = 8;
            cudaDeviceSynchronize();
            cudaEventRecord(a);
            for (int it = 0; it < iters; ++it) {
                strata::kernels::Df2Mat m[3];
                for (int i = 0; i < sh.mats; ++i) m[i] = {w[(size_t) (it % sets) * sh.mats + i], y + (size_t) i * 8 * sh.n_out, sh.n_out};
                if (sh.mats == 3) { m[1].n_out = 1024; m[2].n_out = 1024; }
                strata::kernels::df2_gemm(m, sh.mats, xq, xd, sh.n_in, ncols, nullptr);
            }
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            float ms;
            cudaEventElapsedTime(&ms, a, b);
            const double bytes = sh.mats == 3 ? (double) (4096 + 1024 + 1024) * sh.n_in * 34 / 32 : (double) per_set;
            std::printf("%-30s variant %d: %7.1f us  %6.1f GB/s\n", sh.name, var, 1000.0 * ms / iters, bytes * iters / (ms / 1000.0) / 1e9);
        }
        for (auto p : w) cudaFree(p);
        cudaFree(xq); cudaFree(xd); cudaFree(y);
    }
    return 0;
}
