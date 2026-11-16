// src/kernels/cuda/dequant_s2.cu - P2.S2's first naive kernel.
//
// S2 is Q2_0's canonical form and 31.64 GiB of the artifact is Q2_0, so this is the single hottest decode in
// the engine.  The canonical plane layout is `docs/pack-format.md`'s: codes packed 4 per byte LSB-first
// (element i at byte i/4, bits (i%4)*2 - Q2_0's own convention, kept for every form) and one FP32 scale per
// group (64 elements for S2).  The decode is `(code + code_bias) * scale` with `code_bias = -1`, the
// subtraction inside the INTEGER domain, which is what makes it bit-exact (docs/pack-format.md §3.1).
//
// Naive on purpose: one thread per block, no vectors, no shared memory, no unrolling.  Phase 3 changes this
// file and the parity test in src/kernels/dequant_s2_parity.cpp is what says whether a change is still right.
#include "strata/kernels/dequant_s2.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int QK = 64;             // S2 group = Q2_0 block = 64 elements
constexpr int CODES_PER_BYTE = 4;

__global__ void dequant_s2_kernel(const uint8_t* __restrict__ codes, const float* __restrict__ scales,
                                  float* __restrict__ out, long long n_blocks) {
    const long long b = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const float d = scales[b];
    const uint8_t* c = codes + b * (QK / CODES_PER_BYTE);
    float* y = out + b * QK;
    for (int j = 0; j < QK; ++j) {
        const int code = (c[j / CODES_PER_BYTE] >> ((j % CODES_PER_BYTE) * 2)) & 0x03;
        y[j] = (float)(code - 1) * d;      // the -1 is applied to the CODE, then ONE multiply
    }
}

}  // namespace

void dequant_s2(const uint8_t* codes, const float* scales, float* out, int64_t n_blocks) {
    if (n_blocks <= 0) return;
    const int threads = 256;
    const long long blocks = (n_blocks + threads - 1) / threads;
    dequant_s2_kernel<<<(unsigned) blocks, threads>>>(codes, scales, out, n_blocks);
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "dequant_s2: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::kernels
