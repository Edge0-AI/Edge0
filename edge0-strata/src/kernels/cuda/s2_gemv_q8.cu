// src/kernels/cuda/s2_gemv_q8.cu - the S2 GEMV that consumes Q8_0 ACTIVATIONS, as ggml does.
//
// THE GAP THIS CLOSES (round 186).  `ggml_mul_mat` converts src1 - the ACTIVATION - to the weight's
// `vec_dot_type` before the dot product, and for Q2_0 that is Q8_0.  The CUDA kernels built in rounds 166-177
// take FP16 activations instead, while the CPU expert path (`bench/micro/cpu_s2.cpp`) already quantizes to int8
// and uses `vpdpbusd`.  Left that way the two paths compute different numbers for different experts of the SAME
// token, and the measured size of the difference is the activation contract itself: 0.9498% relative, against
// P2.S2's 1e-3 tolerance for FP16 paths.
//
// This is the CORRECTNESS version, not the fast one.  It dequantizes each activation (`xq * dx`) inside the
// loop and does the same float arithmetic as `s2_gemv_quads`; the speed win available here is `__dp4a`, which
// turns four int8 MACs into one instruction, and that belongs to Phase 3 once the numerics are settled.  The
// phase's rule is the simplest correct kernel first, and a dp4a version whose numerics are wrong would be
// indistinguishable from a fast one.
//
// Layout: activations are (n_in/32) ggml Q8_0 blocks, 34 bytes each - `fp16 d` then `int8 qs[32]` - produced by
// `strata::kernels::quantize_q8_0`.  A quad of four elements lies inside one 32-element block, so the block
// scale is constant across it and is loaded once.
#include "strata/kernels/s2_gemv_q8.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int QK_S2 = 64;
constexpr int QK8_0 = 32;

__global__ void s2_gemv_q8_kernel(const uint8_t* __restrict__ act, const uint8_t* __restrict__ codes,
                                  const float* __restrict__ scales, float* __restrict__ y, long long n_in,
                                  long long n_out, int threads_per_row) {
    extern __shared__ float partial[];
    const long long o = blockIdx.x;
    if (o >= n_out) return;
    const int tid = threadIdx.x;

    const long long n_quads = n_in / 4;
    const uint8_t* c = codes + o * n_quads;             // one code byte per quad
    const float* s = scales + o * (n_in / QK_S2);

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (long long q = tid; q < n_quads; q += threads_per_row) {
        const uint8_t byte = c[q];
        const float d = s[q >> 4];                      // (q*4) >> 6, the S2 group index as a shift
        // the activation quad: four int8 in one 32-element block, so one block scale
        const long long ablk = (q * 4) / QK8_0;
        const uint8_t* blk = act + ablk * 34;
        const uint16_t dbits = (uint16_t) (blk[0] | (blk[1] << 8));
        const float dx = __half2float(__ushort_as_half(dbits));
        const int8_t* xq = reinterpret_cast<const int8_t*>(blk + 2);
        const int off = (int) ((q * 4) % QK8_0);

        const float w0 = (float) ((int) (byte & 3) - 1) * d;
        const float w1 = (float) ((int) ((byte >> 2) & 3) - 1) * d;
        const float w2 = (float) ((int) ((byte >> 4) & 3) - 1) * d;
        const float w3 = (float) ((int) ((byte >> 6) & 3) - 1) * d;
        a0 += w0 * ((float) xq[off + 0] * dx);
        a1 += w1 * ((float) xq[off + 1] * dx);
        a2 += w2 * ((float) xq[off + 2] * dx);
        a3 += w3 * ((float) xq[off + 3] * dx);
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

void s2_gemv_q8(const uint8_t* act, const uint8_t* codes, const float* scales, float* y, int64_t n_in,
                int64_t n_out, int threads_per_row, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % QK8_0 != 0 || n_in % QK_S2 != 0) {
        std::fprintf(stderr, "s2_gemv_q8: n_in %lld must be a multiple of %d\n", (long long) n_in, QK_S2);
        std::exit(1);
    }
    const size_t smem = (size_t) threads_per_row * sizeof(float);
    s2_gemv_q8_kernel<<<(unsigned) n_out, threads_per_row, smem, (cudaStream_t) stream>>>(act, codes, scales, y,
                                                                                         n_in, n_out,
                                                                                         threads_per_row);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "s2_gemv_q8 launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) cudaDeviceSynchronize();
}

}  // namespace strata::kernels
