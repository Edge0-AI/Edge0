// src/kernels/cuda/s2_gemv.cu - P2.S2: the S2 GEMV, one thread per output row.
//
// Naive per the phase rule: dequantize on the fly, FP32 accumulation inside the row, no shared memory, no
// vector loads, no __ldg hints.  Phase 3 changes this file; the parity test is what says whether a change is
// still right.
#include "strata/kernels/s2_gemv.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int QK = 64;

__global__ void s2_gemv_kernel(const uint16_t* __restrict__ x, const uint8_t* __restrict__ codes,
                               const float* __restrict__ scales, float* __restrict__ y, long long n_in,
                               long long n_out) {
    const long long o = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n_out) return;

    const long long nb = n_in / QK;
    const uint8_t* c = codes + o * nb * (QK / 4);
    const float* s = scales + o * nb;

    float acc = 0.0f;
    for (long long b = 0; b < nb; ++b) {
        const float d = s[b];
        const uint8_t* cb = c + b * (QK / 4);
        const uint16_t* xb = x + b * QK;
        for (int j = 0; j < QK; ++j) {
            // (code - 1) in the INTEGER domain, then the group scale, then the activation; the CPU reference
            // below is written in the same order so the two differ only by floating-point contraction.
            const int code = (cb[j >> 2] >> ((j & 3) * 2)) & 0x03;
            acc += (float) (code - 1) * d * __half2float(__ushort_as_half(xb[j]));
        }
    }
    y[o] = acc;
}

}  // namespace

void s2_gemv(const uint16_t* x, const uint8_t* codes, const float* scales, float* y, int64_t n_in,
             int64_t n_out) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % QK != 0) {
        std::fprintf(stderr, "s2_gemv: n_in %lld is not a multiple of %d\n", (long long) n_in, QK);
        std::exit(1);
    }
    const int threads = 128;
    const long long blocks = (n_out + threads - 1) / threads;
    s2_gemv_kernel<<<(unsigned) blocks, threads>>>(x, codes, scales, y, n_in, n_out);
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "s2_gemv: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::kernels
