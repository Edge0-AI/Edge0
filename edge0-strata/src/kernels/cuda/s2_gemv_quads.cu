// src/kernels/cuda/s2_gemv_quads.cu - S2 with the code load amortised over four elements.
//
// THE HYPOTHESIS THIS TESTS.  L4 measured 66 G weights/s = 0.55 weights/cycle/SM, and at an estimated ~13
// instructions per weight that is ~7.2 instructions/cycle/SM against an SM's 4/cycle issue ceiling.  Compute
// (0.4% of FP32 peak) and bandwidth (2%) are excluded by measurement and the dependency chain was excluded by
// experiment in L4, so instruction ISSUE is what remains.  If that is right, cutting instructions per weight
// must move the rate; if the rate does not move, the hypothesis is wrong and that is worth knowing too.
//
// WHERE THE INSTRUCTIONS GO.  For S2 four consecutive elements share ONE byte (element i at byte i/4, bits
// (i%4)*2), and four halves of x are eight contiguous bytes.  The generic kernel walks with a thread stride
// and loads one byte and one half PER ELEMENT; a thread that takes a contiguous quad instead loads one byte
// and one 64-bit word for all four.  Memory instructions per element drop from 2 to 0.5, and the address
// arithmetic is computed once rather than four times.
//
// S2 IS SPECIALISED DELIBERATELY: it is 31.64 GiB of the 38 GiB pack, so it is the kernel that matters.  The
// generic `s_gemv_split` still serves S4 and S8.
#include "strata/kernels/s_gemv.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

// QK = 64 elements per group; a quad of 4 elements is 1/16 of a group, so
//     group = (quad * 4) >> 6 = quad >> 4
constexpr int QK_S2 = 64;
constexpr int CODES_PER_BYTE_S2 = 4;

__global__ void s2_gemv_quads_kernel(const uint16_t* __restrict__ x, const uint8_t* __restrict__ codes,
                                     const float* __restrict__ scales, float* __restrict__ y, long long n_in,
                                     long long n_out, int threads_per_row) {
    extern __shared__ float partial[];
    const long long o = blockIdx.x;
    if (o >= n_out) return;
    const int tid = threadIdx.x;

    const long long n_quads = n_in / 4;
    const uint8_t* c = codes + o * n_quads;             // exactly one code byte per quad
    const float* s = scales + o * (n_in / QK_S2);

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (long long q = tid; q < n_quads; q += threads_per_row) {
        const uint8_t byte = c[q];                      // ONE load for four codes
        const float d = s[q >> 4];                      // (q*4) >> 6, the group index as a shift
        // ONE 64-bit load for four halves.  `x` is 256-byte aligned and four halves are eight bytes, so
        // this is an aligned uint2 - if it were not, the misaligned access would fault rather than be slow.
        const uint2 xw = *reinterpret_cast<const uint2*>(x + q * 4);
        const __half2 h01 = *reinterpret_cast<const __half2*>(&xw.x);
        const __half2 h23 = *reinterpret_cast<const __half2*>(&xw.y);
        // the four codes, each carrying the -1 bias in the INTEGER domain; the scale is applied once per
        // element here rather than once per group, which is the same expression the generic kernel uses
        const float w0 = (float) ((int) (byte & 3) - 1) * d;
        const float w1 = (float) ((int) ((byte >> 2) & 3) - 1) * d;
        const float w2 = (float) ((int) ((byte >> 4) & 3) - 1) * d;
        const float w3 = (float) ((int) ((byte >> 6) & 3) - 1) * d;
        a0 += w0 * __low2float(h01);
        a1 += w1 * __high2float(h01);
        a2 += w2 * __low2float(h23);
        a3 += w3 * __high2float(h23);
    }
    partial[tid] = (a0 + a1) + (a2 + a3);
    __syncthreads();
    for (int step = threads_per_row / 2; step > 0; step >>= 1) {
        if (tid < step) partial[tid] += partial[tid + step];
        __syncthreads();
    }
    if (tid == 0) y[o] = partial[0];
}

}  // namespace

void s2_gemv_quads(const uint16_t* x, const uint8_t* codes, const float* scales, float* y, int64_t n_in,
                   int64_t n_out, int threads_per_row) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % 4 != 0) {
        std::fprintf(stderr, "s2_gemv_quads: n_in %lld is not a multiple of 4\n", (long long) n_in);
        std::exit(1);
    }
    const size_t smem = (size_t) threads_per_row * sizeof(float);
    s2_gemv_quads_kernel<<<(unsigned) n_out, threads_per_row, smem>>>(x, codes, scales, y, n_in, n_out,
                                                                     threads_per_row);
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "s2_gemv_quads: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::kernels
