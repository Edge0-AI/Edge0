// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{quantize.cu,vecdotq.cuh,mmvq.cu,common.cuh}
// and ggml/src/ggml-common.h. See docs/native-mmvq.md for exact scope.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/q8_1_finite.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/gfx_arch.hpp"
#include "strata/core/sm75_series.hpp"
#include "strata/core/sm120_series.hpp"
#include "strata/core/defaults_026.hpp"
#include "s26_tsum.cuh"
#if !defined(__HIPCC__)
#include "q8_1_il.cuh"
#endif
#include "strata/kernels/pdl.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <string>
#include <unordered_map>
#include <cstdio>
#include <vector>

namespace strata::kernels {
namespace {

constexpr int QK = 256;
constexpr int Q8K = 32;
constexpr int QI = 32;
constexpr int VDR = 2;
constexpr int WARPS = 4;
constexpr int WARP = 32;
constexpr int QUANT_THREADS = 256;

struct Q5KBlock {
    half2 dm;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};
struct Q81Block {
    half2 ds;
    int8_t qs[32];
};
struct Q20Block {
    half d;
    uint8_t qs[16];
};
struct Q3KBlock {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    half d;
};
struct IQ4XSBlock {
    half d;
    uint16_t scales_h;
    uint8_t scales_l[4];
    uint8_t qs[128];
};
struct Q4KBlock {
    half2 dm;
    uint8_t scales[12];
    uint8_t qs[128];
};
struct Q6KBlock {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half d;
};
struct Q40Block {
    half d;
    uint8_t qs[16];
};
struct Q50Block {
    half d;
    uint8_t qh[4];
    uint8_t qs[16];
};
struct Q80Block {
    half d;
    int8_t qs[32];
};
struct IQ4NLBlock {
    half d;
    uint8_t qs[16];
};
static_assert(sizeof(Q5KBlock) == 176 && alignof(Q5KBlock) == 4);
static_assert(sizeof(Q81Block) == 36 && alignof(Q81Block) == 4);
static_assert(sizeof(Q20Block) == 18 && alignof(Q20Block) == 2 && offsetof(Q20Block, qs) == 2);
static_assert(sizeof(Q3KBlock) == 110 && alignof(Q3KBlock) == 2 && offsetof(Q3KBlock, qs) == 32 &&
              offsetof(Q3KBlock, scales) == 96 && offsetof(Q3KBlock, d) == 108);
static_assert(sizeof(IQ4XSBlock) == 136 && alignof(IQ4XSBlock) == 2 &&
              offsetof(IQ4XSBlock, scales_h) == 2 && offsetof(IQ4XSBlock, scales_l) == 4 &&
              offsetof(IQ4XSBlock, qs) == 8);
static_assert(offsetof(Q5KBlock, scales) == 4 && offsetof(Q5KBlock, qh) == 16 &&
              offsetof(Q5KBlock, qs) == 48 && offsetof(Q81Block, qs) == 4);
static_assert(sizeof(Q4KBlock) == 144 && alignof(Q4KBlock) == 4 &&
              offsetof(Q4KBlock, scales) == 4 && offsetof(Q4KBlock, qs) == 16);
static_assert(sizeof(Q6KBlock) == 210 && alignof(Q6KBlock) == 2 &&
              offsetof(Q6KBlock, qh) == 128 && offsetof(Q6KBlock, scales) == 192 &&
              offsetof(Q6KBlock, d) == 208);
static_assert(sizeof(Q40Block) == 18 && alignof(Q40Block) == 2 && offsetof(Q40Block, qs) == 2);
static_assert(sizeof(Q50Block) == 22 && alignof(Q50Block) == 2 &&
              offsetof(Q50Block, qh) == 2 && offsetof(Q50Block, qs) == 6);
static_assert(sizeof(Q80Block) == 34 && alignof(Q80Block) == 2 && offsetof(Q80Block, qs) == 2);
static_assert(sizeof(IQ4NLBlock) == 18 && alignof(IQ4NLBlock) == 2 && offsetof(IQ4NLBlock, qs) == 2);

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, WARP);
    }
    return x;
}

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, WARP));
    }
    return x;
}

__launch_bounds__(QUANT_THREADS, 1)
__global__ void native_quantize_q8_1_kernel(const float* x_, Q81Block* y_, int n_in) {
    const float* STRATA_PDL_RESTRICT x = x_;   // __restrict__ below sm_70 only (pdl.hpp, #1469)
    Q81Block* STRATA_PDL_RESTRICT y = y_;
    pdl_trigger();   // PDL (pdl.hpp): the projection after this one may start loading its weights
    pdl_wait();
    const int i = int(blockIdx.x) * QUANT_THREADS + int(threadIdx.x);
    if (i >= n_in) return; // n_in is a multiple of 32: only whole warps return.
    const float xi = x[i];
    const float amax = warp_max(fabsf(xi));
    const float sum = warp_sum(xi);
    const float d = q8_1_finite(amax / 127.0f);   // #606: q8_1_finite.hpp - the same bits for every finite block
    const int8_t q = q8_1_quant(xi, d, amax);
    y[i / Q8K].qs[i % Q8K] = q;
    if (i % Q8K == 0) y[i / Q8K].ds = q8_1_ds(d, sum);
}

__launch_bounds__(QUANT_THREADS, 1)
__global__ void native_swiglu_quantize_q8_1_kernel(const float* __restrict__ gate,
                                                   const float* __restrict__ up,
                                                   Q81Block* __restrict__ y, int n_in) {
    const int i = int(blockIdx.x) * QUANT_THREADS + int(threadIdx.x);
    if (i >= n_in) return; // n_in is a multiple of 32: only whole warps return.
    const float gi = gate[i];
    const float xi = __fmul_rn(__fdividef(gi, __fadd_rn(1.0f, __expf(-gi))), up[i]);
    const float amax = warp_max(fabsf(xi));
    const float sum = warp_sum(xi);
    const float d = q8_1_finite(amax / 127.0f);   // #606, as native_quantize_q8_1_kernel: finite blocks bit for bit
    const int8_t q = q8_1_quant(xi, d, amax);
    y[i / Q8K].qs[i % Q8K] = q;
    if (i % Q8K == 0) y[i / Q8K].ds = q8_1_ds(d, sum);
}

// Exact pinned vec_dot_q5_K_q8_1_impl_vmmq expression and integer dot order.
__device__ __forceinline__ float q5_q8_dot_impl(
    const int* __restrict__ vl, const int* __restrict__ vh, const int* __restrict__ u,
    const uint8_t* __restrict__ sc, const uint8_t* __restrict__ m, const half2& dm5,
    const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0f0f0f0f;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0f0f0f0f;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = STRATA_DP4A(v0i, u[2 * i], STRATA_DP4A(v1i, u[2 * i + 1], 0));
        const int dot2 = STRATA_DP4A(0x01010101, u[2 * i], STRATA_DP4A(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm5f = __half22float2(dm5);
    return dm5f.x * sumf_d - dm5f.y * sumf_m;
}

__device__ __forceinline__ float q5_q8_dot(const Q5KBlock* __restrict__ bq5,
                                          const Q81Block* __restrict__ bq8, int iqs) {
    int vl[2];
    int vh[2];
    int u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* ql = reinterpret_cast<const int*>(bq5->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const int* qh = reinterpret_cast<const int*>(bq5->qh + 4 * ((iqs / 2) % 4));
    vl[0] = ql[0];
    vl[1] = ql[4];
    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq5->scales);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) |
                     ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) |
                     ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Q81Block* bq8i = bq8 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return q5_q8_dot_impl(vl, vh, u, sc, m, bq5->dm, d8);
}

// The generic ncols=1 oracle uses 4 warps and 1 row (or 4 rows for small K),
// eight weight blocks per K iteration, warp-ascending shared sum, then XOR tree.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q5_k_mmvq_kernel(const Q5KBlock* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = VDR * WARPS * WARP / QI;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / QK;
    float tmp[ROWS] = {};
    for (int kbx = tid / (QI / VDR); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * (QK / Q8K);
        const int kqs = VDR * (tid % (QI / VDR));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            // The source assumes allocator padding for partial row groups. This
            // guard preserves every valid row's math without an out-of-bounds read.
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q5_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// Exact pinned vec_dot_q2_0_q8_1: each thread handles one 32-element chunk.
// The weight block is only 2-byte aligned, so qs is intentionally loaded as
// int16_t, unlike the naturally 4-byte aligned activation codes.
__device__ __forceinline__ float q2_q8_dot(const Q20Block* __restrict__ w,
                                          const Q81Block* __restrict__ x, int iqs) {
    const float d2 = w->d;
    const int16_t* qs = reinterpret_cast<const int16_t*>(w->qs) + iqs * 4;
    const Q81Block* chunk = x + iqs;
    const int* q8 = reinterpret_cast<const int*>(chunk->qs);
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = qs[j];
        const int u = q8[j * 2];
        const int v = q8[j * 2 + 1];
        const int qe = __byte_perm(0x020100ff, 0x020100ff, q >> 0);
        const int qo = __byte_perm(0x020100ff, 0x020100ff, q >> 2);
        const int qx = __byte_perm(qe, qo, 0x5140);
        const int qy = __byte_perm(qe, qo, 0x7362);
        sumi = STRATA_DP4A(u, qx, sumi);
        sumi = STRATA_DP4A(v, qy, sumi);
    }
    const float d8 = __low2float(chunk->ds);
    return d2 * d8 * sumi;
}

// Q2_0 generic MMVQ: QK=64, QI=2, VDR=1, 64 blocks per iteration.
// Preserve the same outer accumulation and cross-warp reduction as the oracle.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q2_0_mmvq_kernel(const Q20Block* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = WARPS * WARP / 2;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 64;
    float tmp[ROWS] = {};
    for (int kbx = tid / 2; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 2;
        const int kqs = tid % 2;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q2_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// Q3_K's 110-byte stride gives alternate blocks only two-byte alignment.
// Preserve the pinned helper's pair of 16-bit loads and little-endian combine.
__device__ __forceinline__ int load_int_b2(const void* ptr, int i32) {
    const auto* x = static_cast<const uint16_t*>(ptr);
    int value = x[2 * i32] << 0;
    value |= x[2 * i32 + 1] << 16;
    return value;
}

__device__ __forceinline__ float q3_q8_dot_impl(int vl, int vh, const int* __restrict__ u,
                                              const uint8_t* __restrict__ scales,
                                              int scale_offset, float d3,
                                              const float* __restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int isc = scale_offset + 2 * i;
        const int isc_low = isc % 8;
        const int sc_shift_low = 4 * (isc / 8);
        const int sc_low = (scales[isc_low] >> sc_shift_low) & 0xf;
        const int isc_high = isc % 4;
        const int sc_shift_high = 2 * (isc / 4);
        const int sc_high = ((scales[8 + isc_high] >> sc_shift_high) & 3) << 4;
        const int sc = (sc_low | sc_high) - 32;
        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        const int vi = __vsubss4(vil, vih);
        sumf += d8[i] * (STRATA_DP4A(vi, u[i], 0) * sc);
    }
    return d3 * sumf;
}

__device__ __forceinline__ float q3_q8_dot(const Q3KBlock* __restrict__ w,
                                          const Q81Block* __restrict__ x, int iqs) {
    const int bq8_offset = 4 * (iqs / 8);
    const int scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
    const float d = w->d;
    const int vl = load_int_b2(w->qs, iqs);
    const int vh = ~load_int_b2(w->hmask, iqs % 8) >> bq8_offset;
    int u[4];
    float d8[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        u[i] = reinterpret_cast<const int*>(x[bq8_offset + i].qs)[iqs % 8];
        d8[i] = __low2float(x[bq8_offset + i].ds);
    }
    return q3_q8_dot_impl(vl, vh, u, w->scales, scale_offset, d, d8);
}

// Q3_K generic MMVQ: QK=256, QI=16, VDR=1, eight blocks per iteration.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q3_k_mmvq_kernel(const Q3KBlock* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = WARPS * WARP / 16;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 256;
    float tmp[ROWS] = {};
    for (int kbx = tid / 16; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = tid % 16;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q3_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// The pinned nonlinear IQ4 codebook and its CUDA two-stage byte lookup. The
// explicit alignment satisfies the four 32-bit table loads; values are unchanged.
__device__ __align__(4) int8_t iq4nl_values[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

__device__ __forceinline__ int2 iq4_table_lookup(int q4) {
#if defined(STRATA_HIP_GFX906)
    // AMD: llama.cpp's HIP lookup (see iq_kernels.cu get_int_from_table_16): 4 v_perm_b32 per 8 values
    const uint32_t* v32 = reinterpret_cast<const uint32_t*>(iq4nl_values);
    const uint32_t q_even = (uint32_t) q4, q_odd = (uint32_t) q4 >> 4;
    const uint32_t el = __builtin_amdgcn_perm(v32[1], v32[0], q_even & 0x07070707u);
    const uint32_t ol = __builtin_amdgcn_perm(v32[1], v32[0], q_odd & 0x07070707u);
    const uint32_t eh = __builtin_amdgcn_perm(v32[3], v32[2], q_even & 0x07070707u);
    const uint32_t oh = __builtin_amdgcn_perm(v32[3], v32[2], q_odd & 0x07070707u);
    return make_int2((int) __builtin_amdgcn_perm(eh, el, 0x03020100u | ((q_even & 0x08080808u) >> 1)),
                     (int) __builtin_amdgcn_perm(oh, ol, 0x03020100u | ((q_odd & 0x08080808u) >> 1)));
#else
    const uint32_t* table32 = reinterpret_cast<const uint32_t*>(iq4nl_values);
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = 0x32103210 | ((q4 & 0x88888888) >> 1);
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
#endif
}

// Exact pinned vec_dot_iq4_xs_q8_1: a lane consumes one 32-element subblock,
// computes integer dot products, applies signed scale in the integer domain,
// then multiplies the two half scales and integer sum in the original order.
__device__ __forceinline__ float iq4_xs_q8_dot(const IQ4XSBlock* __restrict__ w,
                                              const Q81Block* __restrict__ x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = reinterpret_cast<const int*>(w->qs)[iqs + j];
        const int2 v = iq4_table_lookup(aux_q4);
        const int u0 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j];
        const int u1 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j + 4];
        sumi = STRATA_DP4A(v.x, u0, sumi);
        sumi = STRATA_DP4A(v.y, u1, sumi);
    }
    const int ls = ((w->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) |
                   (((w->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    const float d = __half2float(w->d) * __low2float(x[iqs / 4].ds);
    return d * sumi;
}

// IQ4_XS generic MMVQ: QK=256, QI=32, VDR=4,16 blocks per iteration.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_iq4_xs_mmvq_kernel(const IQ4XSBlock* __restrict__ w,
                                         const Q81Block* __restrict__ x,
                                         float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = 4 * WARPS * WARP / 32;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 256;
    float tmp[ROWS] = {};
    for (int kbx = tid / 8; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = 4 * (tid % 8);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += iq4_xs_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// Exact pinned vec_dot_q4_K_q8_1_impl_vmmq expression and integer dot order.
__device__ __forceinline__ float q4_q8_dot_impl(
    const int* __restrict__ v, const int* __restrict__ u,
    const uint8_t* __restrict__ sc, const uint8_t* __restrict__ m, const half2& dm4,
    const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0f0f0f0f;
        const int v1i = (v[1] >> (4 * i)) & 0x0f0f0f0f;
        const int dot1 = STRATA_DP4A(v1i, u[2 * i + 1], STRATA_DP4A(v0i, u[2 * i], 0));
        const int dot2 = STRATA_DP4A(0x01010101, u[2 * i + 1], STRATA_DP4A(0x01010101, u[2 * i], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm4f = __half22float2(dm4);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

__device__ __forceinline__ float q4_q8_dot(const Q4KBlock* __restrict__ bq4,
                                          const Q81Block* __restrict__ bq8, int iqs) {
    int v[2];
    int u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* ql = reinterpret_cast<const int*>(bq4->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = ql[0];
    v[1] = ql[4];

    const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq4->scales);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) |
                     ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) |
                     ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Q81Block* bq8i = bq8 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return q4_q8_dot_impl(v, u, sc, m, bq4->dm, d8);
}

// The generic ncols=1 oracle uses 4 warps and 1 row (or 4 rows for small K),
// eight weight blocks per K iteration, warp-ascending shared sum, then XOR tree.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q4_k_mmvq_kernel(const Q4KBlock* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = VDR * WARPS * WARP / QI;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / QK;
    float tmp[ROWS] = {};
    for (int kbx = tid / (QI / VDR); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * (QK / Q8K);
        const int kqs = VDR * (tid % (QI / VDR));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            // The source assumes allocator padding for partial row groups. This
            // guard preserves every valid row's math without an out-of-bounds read.
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q4_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// Exact pinned vec_dot_q6_K_q8_1: keep signed per-16-element scales,
// signed-byte subtraction, DP4A order, and the float accumulation sequence.
__device__ __forceinline__ float q6_q8_dot_impl(int vl, int vh, const int* __restrict__ u,
                                              const int8_t* __restrict__ scales,
                                              float d, const float* __restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = __vsubss4(vil | vih, 0x20202020);
        sumf += d8[i] * (STRATA_DP4A(vi, u[i], 0) * sc);
    }
    return d * sumf;
}
#if !defined(__HIPCC__)   // CUDA only: the native_mmvq_il_pf_kernel family below (itself CUDA only) is the only user
// q6_q8_dot_impl with the two scales already in registers (the interleaved kernels' hoisted loads, P1): the same
// expression in the same order
__device__ __forceinline__ float q6_q8_dot_sc(int vl, int vh, const int* __restrict__ u, int sc0, int sc1, float d,
                                            const float* __restrict__ d8) {
    float sumf = 0.0f;
    const int scs[2] = {sc0, sc1};
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int sc = scs[i];
        const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = __vsubss4(vil | vih, 0x20202020);
        sumf += d8[i] * (STRATA_DP4A(vi, u[i], 0) * sc);
    }
    return d * sumf;
}

// P1: loads the compiler may not sink into a later branch (it moved the Q6_K scale and d loads of the hoisted group
// into the dot sections, where each one is a dependent L1 round trip); the same ld.global.nc the __restrict__ loads use
__device__ __forceinline__ int ldg_nc_u16(const void* p) { unsigned v; asm volatile("ld.global.nc.u16 %0, [%1];" : "=r"(v) : "l"(p)); return int(v); }
__device__ __forceinline__ int ldg_nc_s8(const void* p) { int v; asm volatile("ld.global.nc.s8 %0, [%1];" : "=r"(v) : "l"(p)); return v; }
__device__ __forceinline__ int ldg_nc_u8(const void* p) { unsigned v; asm volatile("ld.global.nc.u8 %0, [%1];" : "=r"(v) : "l"(p)); return int(v); }
__device__ __forceinline__ int ldg_nc_s32(const void* p) { int v; asm volatile("ld.global.nc.s32 %0, [%1];" : "=r"(v) : "l"(p)); return v; }
__device__ __forceinline__ int ldg_nc_b2(const void* p, int i32) {   // load_int_b2 pinned: two u16, little-endian
    const auto* x = static_cast<const uint16_t*>(p) + 2 * i32;
    return ldg_nc_u16(x) | (ldg_nc_u16(x + 1) << 16);
}
// P1: base + k * stride as one IMAD.WIDE.U32 (an asm so the front end does not share k * stride between pointers and
// turn each into a 64-bit add pair)
__device__ __forceinline__ const void* madw(const void* base, int k, int stride) {
    const void* r;
    asm("mad.wide.u32 %0, %1, %2, %3;" : "=l"(r) : "r"(k), "r"(stride), "l"(base));
    return r;
}
// P2: (float) p for |p| < 2^22 as an add on the FMA pipe instead of the quarter-rate I2F: p + 1.5 * 2^23
// as float bits is exactly 1.5 * 2^23 + p (unit ulp), and the subtraction is exact (Sterbenz), so the value is the
// conversion's, +0 included
__device__ __forceinline__ float i2f_small(int p_plus_magic) { return __int_as_float(p_plus_magic) - 12582912.0f; }
constexpr int kI2fMagic = 0x4B400000;
// P2: __vsubss4(x, 0x20202020) for bytes of x in [0, 63] (Q6_K's codes, so nothing saturates): x | 0x80 is x + 0x80 per
// byte, minus 0x20 never borrows (0x60..0x9f), and the xor takes the 0x80 back: the same int8 bytes, 2 instructions
// instead of ptxas's 8 for the saturating subtract sm_80 lacks
__device__ __forceinline__ int sub32_bytes(int x) { return (x + 0x60606060) ^ 0x80808080; }
// q6_q8_dot_sc with P2's integer forms: the dp4a inputs are the same 32-bit values, (dp * sc) the same int, and
// i2f_small its exact float (|dp * sc| <= 4 * 32 * 127 * 128 = 2.1M < 2^22), so every FFMA sees the bits q6_q8_dot_impl's does
__device__ __forceinline__ float q6_q8_dot_fast(int vl, int vh, const int* __restrict__ u, int sc0, int sc1, float d,
                                              const float* __restrict__ d8) {
    float sumf = 0.0f;
    const int scs[2] = {sc0, sc1};
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = sub32_bytes(vil | vih);
        sumf += d8[i] * i2f_small(STRATA_DP4A(vi, u[i], 0) * scs[i] + kI2fMagic);
    }
    return d * sumf;
}
#endif  // !__HIPCC__

__device__ __forceinline__ float q6_q8_dot(const Q6KBlock* __restrict__ w,
                                          const Q81Block* __restrict__ x, int iqs) {
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    const int vl = load_int_b2(w->ql, iqs);
    const int vh = load_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
    int u[2];
    float d8[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        u[i] = reinterpret_cast<const int*>(x[bq8_offset + 2 * i].qs)[iqs % 8];
        d8[i] = __low2float(x[bq8_offset + 2 * i].ds);
    }
    return q6_q8_dot_impl(vl, vh, u, w->scales + scale_offset, w->d, d8);
}

// Q6_K generic MMVQ: QK=256, QI=32, VDR=1, four blocks per iteration.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q6_k_mmvq_kernel(const Q6KBlock* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = WARPS * WARP / 32;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 256;
    float tmp[ROWS] = {};
    for (int kbx = tid / 32; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = tid % 32;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q6_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// The four 32-element formats use native two-byte loads and VDR=2. The affine
// Q4_0/Q5_0 correction consumes the original-input sum stored in Q8_1, exactly
// as the pinned CUDA dot does; a signed-integer code substitution would differ.
__device__ __forceinline__ float small_q8_dot(const Q40Block* __restrict__ w,
                                             const Q81Block* __restrict__ x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = load_int_b2(w->qs, iqs + i);
        const int vi0 = (v >> 0) & 0x0f0f0f0f;
        const int vi1 = (v >> 4) & 0x0f0f0f0f;
        sumi = STRATA_DP4A(vi0, reinterpret_cast<const int*>(x->qs)[iqs + i], sumi);
        sumi = STRATA_DP4A(vi1, reinterpret_cast<const int*>(x->qs)[iqs + i + 4], sumi);
    }
    const float2 ds = __half22float2(x->ds);
    const float d = w->d;
    return d * (sumi * ds.x - 4 * ds.y);
}

__device__ __forceinline__ float small_q8_dot(const Q50Block* __restrict__ w,
                                             const Q81Block* __restrict__ x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl = load_int_b2(w->qs, iqs + i);
        const int vh = load_int_b2(w->qh, 0) >> (4 * (iqs + i));
        int vi0 = (vl >> 0) & 0x0f0f0f0f;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = STRATA_DP4A(vi0, reinterpret_cast<const int*>(x->qs)[iqs + i], sumi);
        int vi1 = (vl >> 4) & 0x0f0f0f0f;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = STRATA_DP4A(vi1, reinterpret_cast<const int*>(x->qs)[iqs + i + 4], sumi);
    }
    const float2 ds = __half22float2(x->ds);
    const float d = w->d;
    return d * (sumi * ds.x - 8 * ds.y);
}

__device__ __forceinline__ float small_q8_dot(const Q80Block* __restrict__ w,
                                             const Q81Block* __restrict__ x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = load_int_b2(w->qs, iqs + i);
        const int u = reinterpret_cast<const int*>(x->qs)[iqs + i];
        sumi = STRATA_DP4A(v, u, sumi);
    }
    const float d0 = w->d;
    const float d1 = __low2float(x->ds);
    return d0 * d1 * float(sumi);
}

__device__ __forceinline__ float small_q8_dot(const IQ4NLBlock* __restrict__ w,
                                             const Q81Block* __restrict__ x, int iqs) {
    const int* q8 = reinterpret_cast<const int*>(x->qs) + iqs;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int2 v = iq4_table_lookup(load_int_b2(w->qs, iqs + i));
        sumi = STRATA_DP4A(v.x, q8[i], sumi);
        sumi = STRATA_DP4A(v.y, q8[i + 4], sumi);
    }
    const float d = __half2float(w->d) * __low2float(x->ds);
    return d * sumi;
}

// QI=4 for Q4_0/Q5_0/IQ4_NL and QI=8 for Q8_0. With VDR=2 this preserves
// the pinned 64/32-block iteration and 2048/1024-element small-K thresholds.
template<typename Weight, int Qi, bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_small_mmvq_kernel(const Weight* __restrict__ w,
                                         const Q81Block* __restrict__ x,
                                         float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = 2 * WARPS * WARP / Qi;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 32;
    float tmp[ROWS] = {};
    for (int kbx = tid / (Qi / 2); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kqs = 2 * (tid % (Qi / 2));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += small_q8_dot(w + block, x + kbx, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// ============================ plan v0.3 P3: ncols = 2..8 (speculative verify, small batches) ============================
//
// One generic kernel for every format, parameterized by the format's iteration traits below, which are
// transcribed from the ncols = 1 kernels above (same thread-to-block mapping, same blocks per iteration, same
// small-K rule). Column j reads activation blocks x + j * (n_in / 32) and writes y + j * n_out. Each (column,
// row) value is accumulated over kbx in the same order, summed across warps in the same order and reduced with
// the same warp tree as the ncols = 1 kernel, so every column is BITWISE equal to a single-column call on that
// column (checked by bench/micro/native_mmvq_multi.cpp). The ncols = 1 kernels are untouched.
// Columns one launch takes: 8, the 0.2.7 limit, or 16 where windows are wide (window_limits.hpp: the cases 9..16 are
// compiled only then, so an engine built with STRATA_WINDOW_MAX_T=8 has exactly the kernels it had).  A window has one
// column per row.  The wide launches read every weight once for all their columns; running 16 columns as two 8-column
// passes (native_mmvq below does it for more than MAX_NCOLS) reads the dense weights twice and measured slower than 8 rows.
constexpr int MAX_NCOLS = STRATA_VERIFY_MAX_T > 8 ? 16 : 8;

// Each format splits its dot product into `load` (everything that depends only on the weight block: codes,
// unpacked scales, block scale) and `apply` (the activation loads and the original *_impl expression). `load` runs
// once per (row, block) and `apply` once per column, so adding columns adds only activation work. `apply` calls
// the same impl functions, in the same order, with the same values as the ncols = 1 dot, which is what keeps
// every column bitwise equal to it.
struct Q5KTraits {
    using Block = Q5KBlock;
    [[maybe_unused]] static constexpr int TYPE = 13;   // ggml type
    static constexpr int DIV = QK, T = QI / VDR, KBY = QK / Q8K, BPI = VDR * WARPS * WARP / QI;
    __device__ static int kqs(int tid) { return VDR * (tid % (QI / VDR)); }
    struct W { int vl[2], vh[2]; uint16_t aux[2]; half2 dm; int bq8_offset; };
    __device__ static W load(const Block* __restrict__ bq5, int iqs) {
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        const int* ql = reinterpret_cast<const int*>(bq5->qs + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4));
        const int* qh = reinterpret_cast<const int*>(bq5->qh + 4 * ((iqs / 2) % 4));
        r.vl[0] = ql[0];
        r.vl[1] = ql[4];
        r.vh[0] = qh[0] >> r.bq8_offset;
        r.vh[1] = qh[4] >> r.bq8_offset;
        const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq5->scales);
        const int j = r.bq8_offset / 2;
        const int jm = j & 1;
        const uint32_t s0 = scales[jm];
        const uint32_t s2 = scales[jm + 2];
        const uint32_t s4 = scales[jm + 4];
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        r.aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        r.aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        r.dm = bq5->dm;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ bq8, int iqs) {
        int u[4];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const Q81Block* bq8i = bq8 + r.bq8_offset + i;
            d8[i] = __low2float(bq8i->ds);
            const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
            u[2 * i] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(r.aux);
        return q5_q8_dot_impl(r.vl, r.vh, u, sc, sc + 2, r.dm, d8);
    }
};
struct Q4KTraits {
    using Block = Q4KBlock;
    [[maybe_unused]] static constexpr int TYPE = 12;   // ggml type
    static constexpr int DIV = QK, T = QI / VDR, KBY = QK / Q8K, BPI = VDR * WARPS * WARP / QI;
    __device__ static int kqs(int tid) { return VDR * (tid % (QI / VDR)); }
    struct W { int v[2]; uint16_t aux[2]; half2 dm; int bq8_offset; };
    __device__ static W load(const Block* __restrict__ bq4, int iqs) {
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        const int* ql = reinterpret_cast<const int*>(bq4->qs + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4));
        r.v[0] = ql[0];
        r.v[1] = ql[4];
        const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq4->scales);
        const int j = r.bq8_offset / 2;
        const int jm = j & 1;
        const uint32_t s0 = scales[jm];
        const uint32_t s2 = scales[jm + 2];
        const uint32_t s4 = scales[jm + 4];
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        r.aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        r.aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        r.dm = bq4->dm;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ bq8, int iqs) {
        int u[4];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const Q81Block* bq8i = bq8 + r.bq8_offset + i;
            d8[i] = __low2float(bq8i->ds);
            const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
            u[2 * i] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(r.aux);
        return q4_q8_dot_impl(r.v, u, sc, sc + 2, r.dm, d8);
    }
};
struct Q20Traits {
    using Block = Q20Block;
    static constexpr int DIV = 64, T = 2, KBY = 2, BPI = WARPS * WARP / 2;
    __device__ static int kqs(int tid) { return tid % 2; }
    struct W { int qx[4], qy[4]; float d2; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.d2 = w->d;
        const int16_t* qs = reinterpret_cast<const int16_t*>(w->qs) + iqs * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = qs[j];
            const int qe = __byte_perm(0x020100ff, 0x020100ff, q >> 0);
            const int qo = __byte_perm(0x020100ff, 0x020100ff, q >> 2);
            r.qx[j] = __byte_perm(qe, qo, 0x5140);
            r.qy[j] = __byte_perm(qe, qo, 0x7362);
        }
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        const Q81Block* chunk = x + iqs;
        const int* q8 = reinterpret_cast<const int*>(chunk->qs);
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumi = STRATA_DP4A(q8[j * 2], r.qx[j], sumi);
            sumi = STRATA_DP4A(q8[j * 2 + 1], r.qy[j], sumi);
        }
        const float d8 = __low2float(chunk->ds);
        return r.d2 * d8 * sumi;
    }
};
struct Q3KTraits {
    using Block = Q3KBlock;
    [[maybe_unused]] static constexpr int TYPE = 11;   // ggml type
    static constexpr int DIV = 256, T = 16, KBY = 8, BPI = WARPS * WARP / 16;
    __device__ static int kqs(int tid) { return tid % 16; }
    struct W { int vl, vh; float d; const uint8_t* scales; int scale_offset, bq8_offset; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 8);
        r.scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
        r.d = w->d;
        r.vl = load_int_b2(w->qs, iqs);
        r.vh = ~load_int_b2(w->hmask, iqs % 8) >> r.bq8_offset;
        r.scales = w->scales;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int u[4];
        float d8[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            u[i] = reinterpret_cast<const int*>(x[r.bq8_offset + i].qs)[iqs % 8];
            d8[i] = __low2float(x[r.bq8_offset + i].ds);
        }
        return q3_q8_dot_impl(r.vl, r.vh, u, r.scales, r.scale_offset, r.d, d8);
    }
};
struct Q6KTraits {
    using Block = Q6KBlock;
    [[maybe_unused]] static constexpr int TYPE = 14;   // ggml type
    static constexpr int DIV = 256, T = 32, KBY = 8, BPI = WARPS * WARP / 32;
    __device__ static int kqs(int tid) { return tid % 32; }
    struct W { int vl, vh; float d; const int8_t* scales; int bq8_offset; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        const int vh_shift = 2 * ((iqs % 16) / 8);
        r.vl = load_int_b2(w->ql, iqs);
        r.vh = load_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
        r.scales = w->scales + scale_offset;
        r.d = w->d;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int u[2];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            u[i] = reinterpret_cast<const int*>(x[r.bq8_offset + 2 * i].qs)[iqs % 8];
            d8[i] = __low2float(x[r.bq8_offset + 2 * i].ds);
        }
        return q6_q8_dot_impl(r.vl, r.vh, u, r.scales, r.d, d8);
    }
};
struct IQ4XSTraits {
    using Block = IQ4XSBlock;
    [[maybe_unused]] static constexpr int TYPE = 23;   // ggml type
    static constexpr int DIV = 256, T = 8, KBY = 8, BPI = 4 * WARPS * WARP / 32;
    __device__ static int kqs(int tid) { return 4 * (tid % 8); }
    struct W { int2 v[4]; int ls; float dw; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
#pragma unroll
        for (int j = 0; j < 4; ++j) r.v[j] = iq4_table_lookup(reinterpret_cast<const int*>(w->qs)[iqs + j]);
        r.ls = ((w->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) | (((w->scales_h >> (iqs / 2)) & 0x03) << 4);
        r.dw = __half2float(w->d);
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int u0 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j];
            const int u1 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j + 4];
            sumi = STRATA_DP4A(r.v[j].x, u0, sumi);
            sumi = STRATA_DP4A(r.v[j].y, u1, sumi);
        }
        sumi *= r.ls - 32;
        const float d = r.dw * __low2float(x[iqs / 4].ds);
        return d * sumi;
    }
};
// The four 32-element formats: `load` decodes the two 32-bit weight chunks and block scale once per (row, block),
// and `apply` runs the exact same STRATA_DP4A and scale arithmetic per activation column as `small_q8_dot`.
template<typename Weight, int Qi>
struct SmallTraits;

template<>
struct SmallTraits<Q40Block, 4> {
    using Block = Q40Block;
    static constexpr int DIV = 32, T = 2, KBY = 1, BPI = 2 * WARPS * WARP / 4;
    __device__ static int kqs(int tid) { return 2 * (tid % 2); }
    struct W { int vi0[2], vi1[2]; float d; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int v = load_int_b2(w->qs, iqs + i);
            r.vi0[i] = (v >> 0) & 0x0f0f0f0f;
            r.vi1[i] = (v >> 4) & 0x0f0f0f0f;
        }
        r.d = w->d;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            sumi = STRATA_DP4A(r.vi0[i], reinterpret_cast<const int*>(x->qs)[iqs + i], sumi);
            sumi = STRATA_DP4A(r.vi1[i], reinterpret_cast<const int*>(x->qs)[iqs + i + 4], sumi);
        }
        const float2 ds = __half22float2(x->ds);
        return r.d * (sumi * ds.x - 4 * ds.y);
    }
};

template<>
struct SmallTraits<Q50Block, 4> {
    using Block = Q50Block;
    static constexpr int DIV = 32, T = 2, KBY = 1, BPI = 2 * WARPS * WARP / 4;
    __device__ static int kqs(int tid) { return 2 * (tid % 2); }
    struct W { int vi0[2], vi1[2]; float d; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        const int qh = load_int_b2(w->qh, 0);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int vl = load_int_b2(w->qs, iqs + i);
            const int vh = qh >> (4 * (iqs + i));
            int vi0 = (vl >> 0) & 0x0f0f0f0f;
            vi0 |= (vh << 4) & 0x00000010;
            vi0 |= (vh << 11) & 0x00001000;
            vi0 |= (vh << 18) & 0x00100000;
            vi0 |= (vh << 25) & 0x10000000;
            r.vi0[i] = vi0;
            int vi1 = (vl >> 4) & 0x0f0f0f0f;
            vi1 |= (vh >> 12) & 0x00000010;
            vi1 |= (vh >> 5) & 0x00001000;
            vi1 |= (vh << 2) & 0x00100000;
            vi1 |= (vh << 9) & 0x10000000;
            r.vi1[i] = vi1;
        }
        r.d = w->d;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            sumi = STRATA_DP4A(r.vi0[i], reinterpret_cast<const int*>(x->qs)[iqs + i], sumi);
            sumi = STRATA_DP4A(r.vi1[i], reinterpret_cast<const int*>(x->qs)[iqs + i + 4], sumi);
        }
        const float2 ds = __half22float2(x->ds);
        return r.d * (sumi * ds.x - 8 * ds.y);
    }
};

template<>
struct SmallTraits<Q80Block, 8> {
    using Block = Q80Block;
    static constexpr int DIV = 32, T = 4, KBY = 1, BPI = 2 * WARPS * WARP / 8;
    __device__ static int kqs(int tid) { return 2 * (tid % 4); }
    struct W { int v[2]; float d0; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
#pragma unroll
        for (int i = 0; i < 2; ++i) r.v[i] = load_int_b2(w->qs, iqs + i);
        r.d0 = w->d;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int u = reinterpret_cast<const int*>(x->qs)[iqs + i];
            sumi = STRATA_DP4A(r.v[i], u, sumi);
        }
        const float d1 = __low2float(x->ds);
        return r.d0 * d1 * float(sumi);
    }
};

template<>
struct SmallTraits<IQ4NLBlock, 4> {
    using Block = IQ4NLBlock;
    static constexpr int DIV = 32, T = 2, KBY = 1, BPI = 2 * WARPS * WARP / 4;
    __device__ static int kqs(int tid) { return 2 * (tid % 2); }
    struct W { int2 v[2]; float dw; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
#pragma unroll
        for (int i = 0; i < 2; ++i) r.v[i] = iq4_table_lookup(load_int_b2(w->qs, iqs + i));
        r.dw = __half2float(w->d);
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        const int* q8 = reinterpret_cast<const int*>(x->qs) + iqs;
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            sumi = STRATA_DP4A(r.v[i].x, q8[i], sumi);
            sumi = STRATA_DP4A(r.v[i].y, q8[i + 4], sumi);
        }
        const float d = r.dw * __low2float(x->ds);
        return d * sumi;
    }
};

// NW warps per block and ROWS rows per block. The EXACT layout (NW = 4, ROWS = 1, or 4 for small K) is the
// ncols = 1 layout and keeps every column bitwise equal to a single-column call. The UPSTREAM layout is
// llama.cpp's generic multi-column table (ncols 2-4: 4 warps; 5-8: 2 warps; always 2 rows per block): faster,
// equal to ncols = 1 only to float rounding (the cross-warp reduction groups partial sums differently).
bool g_multi_exact = true;   // until the upstream layout is timed on an idle GPU (plan rule: default only what is measured)
// B6 STRATA_B6_MMVQ_ROWS (lab, default 0 = off): see native_mmvq_set_b6_rows
// read at first use (not at load): a per-architecture default set at start (nv_defaults.hpp) must be seen
int g_b6_rows = -1;
int b6_rows_mode() { if (g_b6_rows < 0) { const char* v = std::getenv("STRATA_B6_MMVQ_ROWS"); g_b6_rows = v ? std::atoi(v) : 0; } return g_b6_rows; }

// S26 STRATA_TSUM=1 (TS): warp 0's NCOLS x ROWS sums as one transposed butterfly (s26_tsum.cuh), bitwise the same
static bool s26_tsum_on() {
    static const bool on = [] { const char* v = std::getenv("STRATA_TSUM"); return v && v[0] == '1'; }();
    return on;
}

// S26 STRATA_LFUSE (PAIR): blocks n_out.. compute the same rows of w2 into y2 (two matrices of one shape on one
// input in one launch; each output's code is the single matrix's)
template<typename F, int NCOLS, int NW, int ROWS, bool TS = false, bool PAIR = false>
__launch_bounds__(NW * WARP, (ROWS <= 2 ? 4 : 1))
__global__ void native_mmvq_multi_kernel(const typename F::Block* __restrict__ w, const Q81Block* x_, float* y_,
                                         int n_in, int n_out,
                                         const typename F::Block* __restrict__ w2 = nullptr, float* __restrict__ y2 = nullptr) {
    const Q81Block* STRATA_PDL_RESTRICT x = x_;   // __restrict__ below sm_70 only (pdl.hpp, #1469)
    float* STRATA_PDL_RESTRICT y = y_;
    constexpr int BPI = F::BPI * NW / WARPS;           // blocks per iteration scale with the warp count
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    int bxi = int(blockIdx.x);
    if constexpr (PAIR) {
        if (bxi >= n_out / ROWS) { bxi -= n_out / ROWS; w = w2; y = y2; }
    }
    const int row0 = ROWS * bxi;
    const int blocks_per_row = n_in / F::DIV;
    const int x_stride = n_in / Q8K;                   // Q8_1 blocks per activation column
    float tmp[NCOLS][ROWS] = {};
    int kbx = tid / F::T;
    if constexpr (kPdlPrefetch) {
        // PDL (pdl.hpp): the first block's weights are loaded before waiting for the activations, then that block
        // is applied exactly as the loop below would apply it (the same rows, columns and order)
        typename F::W w0[ROWS];
        const int kqs = F::kqs(tid);
        if (kbx < blocks_per_row) {
#pragma unroll
            for (int i = 0; i < ROWS; ++i)
                if (row0 + i < n_out) w0[i] = F::load(w + std::size_t(row0 + i) * blocks_per_row + kbx, kqs);
        }
        pdl_trigger();
        pdl_wait();
        if (kbx < blocks_per_row) {
            const int kby = kbx * F::KBY;
#pragma unroll
            for (int i = 0; i < ROWS; ++i) {
                if (row0 + i < n_out) {
#pragma unroll
                    for (int j = 0; j < NCOLS; ++j)
                        tmp[j][i] += F::apply(w0[i], x + std::size_t(j) * x_stride + kby, kqs);
                }
            }
        }
        kbx += BPI;
    }
    for (; kbx < blocks_per_row; kbx += BPI) {
        const int kby = kbx * F::KBY;
        const int kqs = F::kqs(tid);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                const typename F::W wv = F::load(w + block, kqs);      // once per (row, block)
#pragma unroll
                for (int j = 0; j < NCOLS; ++j)                         // then per column
                    tmp[j][i] += F::apply(wv, x + std::size_t(j) * x_stride + kby, kqs);
            }
        }
    }
    __shared__ float partial[NW - 1 > 0 ? NW - 1 : 1][NCOLS][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < NCOLS; ++j)
#pragma unroll
            for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
    if constexpr (TS) {
        constexpr int V = NCOLS * ROWS, P = s26ts::pow2_ceil(V);
        float v[P];
#pragma unroll
        for (int j = 0; j < NCOLS; ++j)
#pragma unroll
            for (int i = 0; i < ROWS; ++i) {
#pragma unroll
                for (int l = 0; l < NW - 1; ++l) tmp[j][i] += partial[l][j][i][threadIdx.x];
                v[j * ROWS + i] = tmp[j][i];
            }
#pragma unroll
        for (int k = V; k < P; ++k) v[k] = 0.0f;
        const float sum = s26ts::tsum<P>(v, int(threadIdx.x));
        const int k = s26ts::tsum_token<P>(int(threadIdx.x));
        const int j = k / ROWS, i = k % ROWS;
        if (int(threadIdx.x) == s26ts::tsum_lane<P>(k) && k < V && row0 + i < n_out) y[std::size_t(j) * n_out + row0 + i] = sum;
        return;
    }
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
#pragma unroll
            for (int l = 0; l < NW - 1; ++l) tmp[j][i] += partial[l][j][i][threadIdx.x];
            tmp[j][i] = warp_sum(tmp[j][i]);
            if (threadIdx.x == i && row0 + i < n_out) y[std::size_t(j) * n_out + row0 + i] = tmp[j][i];
        }
    }
}

// MODEL-UPDATE: the dense Q8_0 projections of Unsloth's files (all of attention, SSM and the shared expert are Q8_0
// there) take one row per block where that was measured faster - the same per-row sums and reduction as two rows per
// block (each row its own partials), so every output is the same bits.  q8_gemv_bench, 2-4 columns:
//   gfx1201 (R9700): n_in <= 4096 with n_out >= 4096 (qkv 2560->10240 73-79% -> 83-84% of the 644 GB/s, gate 2560->6144
//     51-68% -> 76-78%, q 2560->12288 74-80% -> 85-86%); 6144->2560 is slower there (79% -> 55-59%) and keeps two rows.
//   sm_86 (RTX A4000): every n_out >= 2048 (+1-4% at 2-4 columns, 6144->2560 at 6 columns 70% -> 76%).
// Other cards keep two rows until measured (STRATA_NO_MMVQ_ROWS2=1 forces one row everywhere; =0 keeps two everywhere).
int dev_class() {   // 1 = gfx12, 2 = sm_86, 0 = the rest; per device
    static int cls[16];   // 0 unknown, else class + 1
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) return 0;
    if (cls[dev] == 0) {
        int c = 0;
#if defined(__HIPCC__)
        cudaDeviceProp p;
        if (cudaGetDeviceProperties(&p, dev) == cudaSuccess && strata::kernels::gfx_arch_is_gfx12(p.gcnArchName)) c = 1;
#else
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess && major == 8 && minor == 6)
            c = 2;
#endif
        (void) cudaGetLastError();
        cls[dev] = c + 1;
    }
    return cls[dev] - 1;
}
template<typename F>
bool q8_0_rows1(int n_in, int n_out) {
    if constexpr (!std::is_same_v<F, SmallTraits<Q80Block, 8>>) {
        (void) n_in; (void) n_out;
        return false;
    } else {
        static const int env = [] {
            const char* v = std::getenv("STRATA_NO_MMVQ_ROWS2");
            return v == nullptr || v[0] == '\0' ? -1 : v[0] != '0';
        }();
        if (env == 0) return false;
        const int c = dev_class();
        return (c == 1 && n_in <= 4096 && n_out >= 4096) || (c == 2 && n_out >= 2048);
    }
}

// ---- B6 STRATA_B6_MMVQ_ROWS (lab, opt-in): the exact multi-column layout with each column's activation fragment
// loaded ONCE per weight block and reused across ROWS rows of the block (the default kernel reloads it for every row
// it applies it to, and the K-quant scales once per column).  A thread keeps the same blocks (kbx = tid / T, stride
// BPI with WARPS warps), each (row, column) sum adds the same per-block values in the same order, and the cross-warp
// and warp reductions are the default kernel's: every output is bitwise the default exact layout's, so bitwise a
// single-column call.  Only the formats with a B6X below; the others keep the default kernel.
template<typename F> struct B6X { static constexpr bool ok = false; };
template<> struct B6X<Q6KTraits> {
    static constexpr bool ok = true;
    struct X { int u[2]; float d8[2]; };
    struct W { int vl, vh, sc[2]; float d; };
    __device__ static X xload(const Q81Block* x, int iqs) {
        X r;
        const int off = 4 * (iqs / 16) + (iqs % 16) / 8;   // Q6KTraits::load's bq8_offset
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            r.u[i] = reinterpret_cast<const int*>(x[off + 2 * i].qs)[iqs % 8];
            r.d8[i] = __low2float(x[off + 2 * i].ds);
        }
        return r;
    }
    __device__ static W wload(const Q6KBlock* __restrict__ w, int iqs) {
        const Q6KTraits::W t = Q6KTraits::load(w, iqs);
        W r;
        r.vl = t.vl;
        r.vh = t.vh;
        r.sc[0] = t.scales[0];
        r.sc[1] = t.scales[4];
        r.d = t.d;
        return r;
    }
    __device__ static float dot(const W& w, const X& x) {   // q6_q8_dot_impl, the scales already in registers
        float sumf = 0.0f;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int vil = (w.vl >> (4 * i)) & 0x0f0f0f0f;
            const int vih = ((w.vh >> (4 * i)) << 4) & 0x30303030;
            const int vi = __vsubss4(vil | vih, 0x20202020);
            sumf += x.d8[i] * (STRATA_DP4A(vi, x.u[i], 0) * w.sc[i]);
        }
        return w.d * sumf;
    }
};
template<typename T, int KIND> struct B6XK {   // Q4_K (KIND 4) and Q5_K (KIND 5): the same activation fragment
    static constexpr bool ok = true;
    struct X { int u[4]; float d8[2]; };
    using W = typename T::W;
    __device__ static X xload(const Q81Block* bq8, int iqs) {
        X r;
        const int off = 2 * ((iqs / 2) / 4);   // the traits' bq8_offset
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const Q81Block* bq8i = bq8 + off + i;
            r.d8[i] = __low2float(bq8i->ds);
            const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
            r.u[2 * i] = q8[0];
            r.u[2 * i + 1] = q8[4];
        }
        return r;
    }
    __device__ static W wload(const typename T::Block* __restrict__ w, int iqs) { return T::load(w, iqs); }
    __device__ static float dot(const W& w, const X& x) {
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(w.aux);
        if constexpr (KIND == 4) return q4_q8_dot_impl(w.v, x.u, sc, sc + 2, w.dm, x.d8);
        else return q5_q8_dot_impl(w.vl, w.vh, x.u, sc, sc + 2, w.dm, x.d8);
    }
};
template<> struct B6X<Q4KTraits> : B6XK<Q4KTraits, 4> {};
template<> struct B6X<Q5KTraits> : B6XK<Q5KTraits, 5> {};
template<> struct B6X<IQ4XSTraits> {
    static constexpr bool ok = true;
    struct X { int u0[4], u1[4]; float dx; };
    using W = IQ4XSTraits::W;
    __device__ static X xload(const Q81Block* x, int iqs) {
        X r;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            r.u0[j] = reinterpret_cast<const int*>(x[iqs / 4].qs)[j];
            r.u1[j] = reinterpret_cast<const int*>(x[iqs / 4].qs)[j + 4];
        }
        r.dx = __low2float(x[iqs / 4].ds);
        return r;
    }
    __device__ static W wload(const IQ4XSBlock* __restrict__ w, int iqs) { return IQ4XSTraits::load(w, iqs); }
    __device__ static float dot(const W& w, const X& x) {   // IQ4XSTraits::apply
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumi = STRATA_DP4A(w.v[j].x, x.u0[j], sumi);
            sumi = STRATA_DP4A(w.v[j].y, x.u1[j], sumi);
        }
        sumi *= w.ls - 32;
        const float d = w.dw * x.dx;
        return d * sumi;
    }
};

template<typename F, int NCOLS, int ROWS>
__launch_bounds__(WARPS * WARP, 1)
__global__ void b6_mmvq_rows_kernel(const typename F::Block* __restrict__ w, const Q81Block* __restrict__ x,
                                    float* __restrict__ y, int n_in, int n_out) {
    using XB = B6X<F>;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / F::DIV;
    const int x_stride = n_in / Q8K;
    const int kqs = F::kqs(tid);
    float tmp[NCOLS][ROWS] = {};
    for (int kbx = tid / F::T; kbx < blocks_per_row; kbx += F::BPI) {
        const int kby = kbx * F::KBY;
        typename XB::X xv[NCOLS];
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) xv[j] = XB::xload(x + std::size_t(j) * x_stride + kby, kqs);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const typename XB::W wv = XB::wload(w + std::size_t(row0 + i) * blocks_per_row + kbx, kqs);
#pragma unroll
                for (int j = 0; j < NCOLS; ++j) tmp[j][i] += XB::dot(wv, xv[j]);
            }
        }
    }
    __shared__ float partial[WARPS - 1][NCOLS][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < NCOLS; ++j)
#pragma unroll
            for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
#pragma unroll
            for (int l = 0; l < WARPS - 1; ++l) tmp[j][i] += partial[l][j][i][threadIdx.x];
            tmp[j][i] = warp_sum(tmp[j][i]);
            if (threadIdx.x == i && row0 + i < n_out) y[std::size_t(j) * n_out + row0 + i] = tmp[j][i];
        }
    }
}

// the B6 launch for a format that has a B6X: true when it ran.  g_b6_rows: 1 = 2 rows per block (the fastest on both
// cards in b6_mmvq_bench), else 2 / 4 / 8 rows.
template<typename F, int NCOLS>
bool b6_launch(const typename F::Block* w, const Q81Block* x, float* y, int n_in, int n_out, cudaStream_t s) {
    if constexpr (!B6X<F>::ok) {
        return false;
    } else {
        if (b6_rows_mode() <= 0 || !g_multi_exact || s26_tsum_on() || n_in / F::DIV < 1) return false;
        const int rows = b6_rows_mode() == 1 ? 2 : b6_rows_mode();
        const dim3 threads(WARP, WARPS);
        if (rows == 2)
            b6_mmvq_rows_kernel<F, NCOLS, 2><<<unsigned((n_out + 1) / 2), threads, 0, s>>>(w, x, y, n_in, n_out);
        else if (rows == 8)
            b6_mmvq_rows_kernel<F, NCOLS, 8><<<unsigned((n_out + 7) / 8), threads, 0, s>>>(w, x, y, n_in, n_out);
        else
            b6_mmvq_rows_kernel<F, NCOLS, 4><<<unsigned((n_out + 3) / 4), threads, 0, s>>>(w, x, y, n_in, n_out);
        return true;
    }
}

template<typename F, int NCOLS>
void launch_multi_n(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, cudaStream_t s) {
    const auto* w = static_cast<const typename F::Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    if (b6_launch<F, NCOLS>(w, x, y, n_in, n_out, s)) return;   // STRATA_B6_MMVQ_ROWS
    if (!g_multi_exact) {
        constexpr int NW = NCOLS <= 4 ? 4 : 2;
        const unsigned blocks = unsigned((std::size_t(n_out) + 1) / 2);
        launch_pdl(native_mmvq_multi_kernel<F, NCOLS, NW, 2, false, false>, dim3(blocks), dim3(WARP, NW), 0, s, w, x, y, n_in, n_out,
                   (const typename F::Block*) nullptr, (float*) nullptr);
        return;
    }
    const dim3 threads(WARP, WARPS);
    // the PAIR arguments, unused here (launch_pdl passes every parameter)
    const typename F::Block* const no_w2 = nullptr;
    float* const no_y2 = nullptr;
    // #783 PR-h (stuchapin909): two rows per block for every K, not only small K - each row keeps its own partial sums and
    // reduction, so a row's result does not depend on its neighbour; STRATA_NO_MMVQ_ROWS2=1 keeps one row for large K
    static const bool rows1 = [] {
        const char* v = std::getenv("STRATA_NO_MMVQ_ROWS2");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    if ((rows1 || q8_0_rows1<F>(n_in, n_out)) && n_in / F::DIV >= F::BPI) {
        if (s26_tsum_on()) launch_pdl(native_mmvq_multi_kernel<F, NCOLS, WARPS, 1, true, false>, dim3(unsigned(n_out)), threads, 0, s, w, x, y, n_in, n_out, no_w2, no_y2);
        else launch_pdl(native_mmvq_multi_kernel<F, NCOLS, WARPS, 1, false, false>, dim3(unsigned(n_out)), threads, 0, s, w, x, y, n_in, n_out, no_w2, no_y2);
        return;
    }
    constexpr int ROWS = 2;
    const unsigned blocks = unsigned((std::size_t(n_out) + ROWS - 1) / ROWS);
    if (s26_tsum_on()) launch_pdl(native_mmvq_multi_kernel<F, NCOLS, WARPS, ROWS, true, false>, dim3(blocks), threads, 0, s, w, x, y, n_in, n_out, no_w2, no_y2);
    else launch_pdl(native_mmvq_multi_kernel<F, NCOLS, WARPS, ROWS, false, false>, dim3(blocks), threads, 0, s, w, x, y, n_in, n_out, no_w2, no_y2);
}

template<typename F>
void launch_multi(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                  void* stream) {
    const auto s = static_cast<cudaStream_t>(stream);
    switch (ncols) {
        case 2: launch_multi_n<F, 2>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 3: launch_multi_n<F, 3>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 4: launch_multi_n<F, 4>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 5: launch_multi_n<F, 5>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 6: launch_multi_n<F, 6>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 7: launch_multi_n<F, 7>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 8: launch_multi_n<F, 8>(weights, x_q8_1, y, n_in, n_out, s); break;
#if STRATA_VERIFY_MAX_T > 8
        case 9: launch_multi_n<F, 9>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 10: launch_multi_n<F, 10>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 11: launch_multi_n<F, 11>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 12: launch_multi_n<F, 12>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 13: launch_multi_n<F, 13>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 14: launch_multi_n<F, 14>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 15: launch_multi_n<F, 15>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 16: launch_multi_n<F, 16>(weights, x_q8_1, y, n_in, n_out, s); break;
#endif
        default: throw std::invalid_argument("native MMVQ multi-column launch requires 2 <= ncols <= MAX_NCOLS");
    }
}

#if !defined(__HIPCC__)   // CUDA only: the HIP paths keep native_mmvq's kernels
// ============================ ncols = 2..4 from interleaved activations (fork F4, Eddoursul) ============================
//
// `native_quantize_q8_1_il` also writes the columns interleaved (native_mmvq.hpp), so one load reads the same int of
// every column. A warp takes R rows and walks their blocks in the chunks of the exact layout above - its lane l's chunk
// m is what thread 32 (m % 4) + l does in iteration m / 4 - with a sum per virtual warp, added in warp order before the
// xor tree; each weight block is decoded once (the traits' `load`) and every column takes the single-column dot's own
// expression (the same *_q8_dot_impl calls), so each value is bitwise the multi-column kernel's. K-quants read the
// block-major copy; IQ4_XS, whose lanes read the same int of consecutive blocks, the position-major one.
struct IlIQ4XS {
    using F = IQ4XSTraits;
    static constexpr bool PM = true;
    template <int NC, class X>
    __device__ static void apply(const F::W& r, const X& x, int kby, int iqs, float (&out)[NC]) {
        const int b = kby + iqs / 4;
        int u0[4][NC], u1[4][NC];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            x.u(b, j, u0[j]);
            x.u(b, j + 4, u1[j]);
        }
        float d8[NC];
        x.scales(b, d8);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            int sumi = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                sumi = STRATA_DP4A(r.v[j].x, u0[j][c], sumi);
                sumi = STRATA_DP4A(r.v[j].y, u1[j][c], sumi);
            }
            sumi *= r.ls - 32;
            const float d = r.dw * d8[c];
            out[c] = d * sumi;
        }
    }
    // P1 (native_mmvq_il_pf_kernel): the block's bytes as loaded, the table lookup and the scale decode deferred to
    // applyr so the loads of a whole group issue before any of them is consumed; the same integers and floats as apply
    static constexpr bool HAS_PF = true;
    struct RP { const uint8_t *qs, *sl, *blk; };   // a row's blocks for this lane: its qs ints, its scales_l byte, the block
    __device__ static RP rowptrs(const F::Block* __restrict__ row, int iqs) {
        const auto* b = reinterpret_cast<const uint8_t*>(row);
        return {b + 8 + 4 * iqs, b + 4 + iqs / 8, b};
    }
    struct WR { int q[4]; int sl, sh, d; };
    // the pointers of block kbx (RP::at), and the loads of block kbx + dv from them (dv a compile-time offset, so the
    // group's loads share one pointer set and differ by an immediate)
    __device__ static RP at(const RP& rp, int kbx) {
        constexpr int S = int(sizeof(F::Block));
        return {static_cast<const uint8_t*>(madw(rp.qs, kbx, S)), static_cast<const uint8_t*>(madw(rp.sl, kbx, S)),
                static_cast<const uint8_t*>(madw(rp.blk, kbx, S))};
    }
    __device__ static WR loadr(const RP& rp, int dv) {
        constexpr int S = int(sizeof(F::Block));
        const auto* qs = reinterpret_cast<const int*>(rp.qs + dv * S);
        WR r;
#pragma unroll
        for (int j = 0; j < 4; ++j) r.q[j] = ldg_nc_s32(qs + j);
        r.sl = ldg_nc_u8(rp.sl + dv * S);
        r.sh = ldg_nc_u16(rp.blk + dv * S + 2);
        r.d = ldg_nc_u16(rp.blk + dv * S);
        return r;
    }
    // the activations: position-major (every int of the 8 at a stride of nb), so the kernel's block pointers are not used
    __device__ static int xb(int) { return 0; }
    __device__ static int xp(int) { return 0; }
    template <int NC, bool FAST, class X>
    __device__ static void applyr(const WR& rr, const X& x, const int*, const float*, int kby, int iqs, float (&out)[NC]) {
        F::W r;
#pragma unroll
        for (int j = 0; j < 4; ++j) r.v[j] = iq4_table_lookup(rr.q[j]);
        r.ls = ((rr.sl >> (iqs & 0x04)) & 0x0f) | (((rr.sh >> (iqs / 2)) & 0x03) << 4);
        r.dw = __half2float(__ushort_as_half((unsigned short) rr.d));
        if constexpr (!FAST) {
            apply<NC>(r, x, kby, iqs, out);
        } else {
            // P2: sumi (|sumi| <= 32 * 127 * 127 = 516K < 2^22) through i2f_small, then times (float)(ls - 32): the product
            // |sumi * (ls - 32)| < 2^24 is exact in fp32, so it is the float of the int apply() converts; the same d * sumi
            const int b = kby + iqs / 4;
            int u0[4][NC], u1[4][NC];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                x.u(b, j, u0[j]);
                x.u(b, j + 4, u1[j]);
            }
            float d8[NC];
            x.scales(b, d8);
            const float fls = float(r.ls - 32);
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                int sumi = 0;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    sumi = STRATA_DP4A(r.v[j].x, u0[j][c], sumi);
                    sumi = STRATA_DP4A(r.v[j].y, u1[j][c], sumi);
                }
                const float fsumi = i2f_small(sumi + kI2fMagic) * fls;
                const float d = r.dw * d8[c];
                out[c] = d * fsumi;
            }
        }
    }
};
struct IlQ4K {
    using F = Q4KTraits;
    static constexpr bool PM = false;
    static constexpr bool HAS_PF = true;   // P1: F::W holds every byte of the block already (no pointer into it)
    struct RP { const uint8_t *qs, *sc, *blk; int iqs; };   // this lane's ql ints, its scale words, the block (dm)
    __device__ static RP rowptrs(const F::Block* __restrict__ row, int iqs) {
        const auto* b = reinterpret_cast<const uint8_t*>(row);
        const int bq8_offset = 2 * ((iqs / 2) / 4), jm = (bq8_offset / 2) & 1;
        return {b + 16 + 16 * bq8_offset + 4 * ((iqs / 2) % 4), b + 4 + 2 * jm, b, iqs};
    }
    using WR = F::W;
    __device__ static RP at(const RP& rp, int kbx) {
        constexpr int S = int(sizeof(F::Block));
        return {static_cast<const uint8_t*>(madw(rp.qs, kbx, S)), static_cast<const uint8_t*>(madw(rp.sc, kbx, S)),
                static_cast<const uint8_t*>(madw(rp.blk, kbx, S)), rp.iqs};
    }
    __device__ static WR loadr(const RP& rp, int dv) {   // F::load's bytes and aux arithmetic, block dv past rp's
        constexpr int S = int(sizeof(F::Block));
        const auto* ql = reinterpret_cast<const int*>(rp.qs + dv * S);
        const auto* sc = rp.sc + dv * S;
        WR r;
        r.bq8_offset = 2 * ((rp.iqs / 2) / 4);
        r.v[0] = ldg_nc_s32(ql);
        r.v[1] = ldg_nc_s32(ql + 4);
        const uint32_t s0 = ldg_nc_u16(sc), s2 = ldg_nc_u16(sc + 4), s4 = ldg_nc_u16(sc + 8);
        const int dm = ldg_nc_s32(rp.blk + dv * S);
        const int j = r.bq8_offset / 2;
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        r.aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        r.aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        { half2 h; std::memcpy(&h, &dm, 4); r.dm = h; }
        return r;
    }
    __device__ static int xb(int iqs) { return 2 * ((iqs / 2) / 4); }   // the lane's q8_1 block and int within a 256-chunk
    __device__ static int xp(int iqs) { return (iqs / 2) % 4; }
    template <int NC, bool FAST, class X>
    __device__ static void applyr(const WR& r, const X& x, const int* qb, const float* db, int kby, int iqs, float (&out)[NC]) {
        if constexpr (!FAST) {
            apply<NC>(r, x, kby, iqs, out);
        } else {   // P2: q4_q8_dot_impl with (dot1 * sc), (dot2 * m) through i2f_small (|dot1 * sc| <= 15240 * 63, |dot2 * m| <= 1016 * 63)
            int u[4][NC];
            float d8[2][NC];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                x.qat(qb + i * X::QB, u[2 * i]);
                x.qat(qb + i * X::QB + x.qp(4), u[2 * i + 1]);
                x.dat(db + i * X::DB, d8[i]);
            }
            const uint8_t* sc = reinterpret_cast<const uint8_t*>(r.aux);
            const uint8_t* m = sc + 2;
            const float2 dm4f = __half22float2(r.dm);
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                float sumf_d = 0.0f;
                float sumf_m = 0.0f;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const int v0i = (r.v[0] >> (4 * i)) & 0x0f0f0f0f;
                    const int v1i = (r.v[1] >> (4 * i)) & 0x0f0f0f0f;
                    const int dot1 = STRATA_DP4A(v1i, u[2 * i + 1][c], STRATA_DP4A(v0i, u[2 * i][c], 0));
                    const int dot2 = STRATA_DP4A(0x01010101, u[2 * i + 1][c], STRATA_DP4A(0x01010101, u[2 * i][c], 0));
                    sumf_d += d8[i][c] * i2f_small(dot1 * sc[i] + kI2fMagic);
                    sumf_m += d8[i][c] * i2f_small(dot2 * m[i] + kI2fMagic);
                }
                out[c] = dm4f.x * sumf_d - dm4f.y * sumf_m;
            }
        }
    }
    template <int NC, class X>
    __device__ static void apply(const F::W& r, const X& x, int kby, int iqs, float (&out)[NC]) {
        int u[4][NC];
        float d8[2][NC];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int b = kby + r.bq8_offset + i;
            x.u(b, (iqs / 2) % 4, u[2 * i]);
            x.u(b, (iqs / 2) % 4 + 4, u[2 * i + 1]);
            x.scales(b, d8[i]);
        }
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(r.aux);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const int uc[4] = {u[0][c], u[1][c], u[2][c], u[3][c]};
            const float dc[2] = {d8[0][c], d8[1][c]};
            out[c] = q4_q8_dot_impl(r.v, uc, sc, sc + 2, r.dm, dc);
        }
    }
};
struct IlQ5K {
    using F = Q5KTraits;
    static constexpr bool PM = false;
    static constexpr bool HAS_PF = true;   // P1: as IlQ4K
    struct RP { const uint8_t *qs, *qh, *sc, *blk; int iqs; };
    __device__ static RP rowptrs(const F::Block* __restrict__ row, int iqs) {
        const auto* b = reinterpret_cast<const uint8_t*>(row);
        const int bq8_offset = 2 * ((iqs / 2) / 4), jm = (bq8_offset / 2) & 1;
        return {b + 48 + 16 * bq8_offset + 4 * ((iqs / 2) % 4), b + 16 + 4 * ((iqs / 2) % 4), b + 4 + 2 * jm, b, iqs};
    }
    using WR = F::W;
    __device__ static RP at(const RP& rp, int kbx) {
        constexpr int S = int(sizeof(F::Block));
        return {static_cast<const uint8_t*>(madw(rp.qs, kbx, S)), static_cast<const uint8_t*>(madw(rp.qh, kbx, S)),
                static_cast<const uint8_t*>(madw(rp.sc, kbx, S)), static_cast<const uint8_t*>(madw(rp.blk, kbx, S)), rp.iqs};
    }
    __device__ static WR loadr(const RP& rp, int dv) {   // F::load's bytes and aux arithmetic, block dv past rp's
        constexpr int S = int(sizeof(F::Block));
        const auto* ql = reinterpret_cast<const int*>(rp.qs + dv * S);
        const auto* qh = reinterpret_cast<const int*>(rp.qh + dv * S);
        const auto* sc = rp.sc + dv * S;
        WR r;
        r.bq8_offset = 2 * ((rp.iqs / 2) / 4);
        r.vl[0] = ldg_nc_s32(ql);
        r.vl[1] = ldg_nc_s32(ql + 4);
        r.vh[0] = ldg_nc_s32(qh) >> r.bq8_offset;
        r.vh[1] = ldg_nc_s32(qh + 4) >> r.bq8_offset;
        const uint32_t s0 = ldg_nc_u16(sc), s2 = ldg_nc_u16(sc + 4), s4 = ldg_nc_u16(sc + 8);
        const int dm = ldg_nc_s32(rp.blk + dv * S);
        const int j = r.bq8_offset / 2;
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        r.aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        r.aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        { half2 h; std::memcpy(&h, &dm, 4); r.dm = h; }
        return r;
    }
    __device__ static int xb(int iqs) { return 2 * ((iqs / 2) / 4); }
    __device__ static int xp(int iqs) { return (iqs / 2) % 4; }
    template <int NC, bool FAST, class X>
    __device__ static void applyr(const WR& r, const X& x, const int* qb, const float* db, int kby, int iqs, float (&out)[NC]) {
        if constexpr (!FAST) {
            apply<NC>(r, x, kby, iqs, out);
        } else {   // P2: q5_q8_dot_impl with (dot1 * sc), (dot2 * m) through i2f_small (|dot1 * sc| <= 31496 * 63 < 2^22)
            int u[4][NC];
            float d8[2][NC];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                x.qat(qb + i * X::QB, u[2 * i]);
                x.qat(qb + i * X::QB + x.qp(4), u[2 * i + 1]);
                x.dat(db + i * X::DB, d8[i]);
            }
            const uint8_t* sc = reinterpret_cast<const uint8_t*>(r.aux);
            const uint8_t* m = sc + 2;
            const float2 dm5f = __half22float2(r.dm);
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                float sumf_d = 0.0f;
                float sumf_m = 0.0f;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const int vl0i = (r.vl[0] >> (4 * i)) & 0x0f0f0f0f;
                    const int vl1i = (r.vl[1] >> (4 * i)) & 0x0f0f0f0f;
                    const int vh0i = ((r.vh[0] >> i) << 4) & 0x10101010;
                    const int vh1i = ((r.vh[1] >> i) << 4) & 0x10101010;
                    const int v0i = vl0i | vh0i;
                    const int v1i = vl1i | vh1i;
                    const int dot1 = STRATA_DP4A(v0i, u[2 * i][c], STRATA_DP4A(v1i, u[2 * i + 1][c], 0));
                    const int dot2 = STRATA_DP4A(0x01010101, u[2 * i][c], STRATA_DP4A(0x01010101, u[2 * i + 1][c], 0));
                    sumf_d += d8[i][c] * i2f_small(dot1 * sc[i] + kI2fMagic);
                    sumf_m += d8[i][c] * i2f_small(dot2 * m[i] + kI2fMagic);
                }
                out[c] = dm5f.x * sumf_d - dm5f.y * sumf_m;
            }
        }
    }
    template <int NC, class X>
    __device__ static void apply(const F::W& r, const X& x, int kby, int iqs, float (&out)[NC]) {
        int u[4][NC];
        float d8[2][NC];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int b = kby + r.bq8_offset + i;
            x.u(b, (iqs / 2) % 4, u[2 * i]);
            x.u(b, (iqs / 2) % 4 + 4, u[2 * i + 1]);
            x.scales(b, d8[i]);
        }
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(r.aux);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const int uc[4] = {u[0][c], u[1][c], u[2][c], u[3][c]};
            const float dc[2] = {d8[0][c], d8[1][c]};
            out[c] = q5_q8_dot_impl(r.vl, r.vh, uc, sc, sc + 2, r.dm, dc);
        }
    }
};
// Q3_K (143-dense): the Q2_0 pack's dense projections are mostly Q3_K, which only had the exact layout's kernel. The
// dot is q3_q8_dot_impl, the multi-column kernel's own call, on the same four blocks' ints and scales.
struct IlQ3K {
    using F = Q3KTraits;
    static constexpr bool PM = false;
    static constexpr bool HAS_PF = false;   // the Q2_0 pack's type: keeps native_mmvq_il_kernel (F::W points into the block)
    template <int NC, class X>
    __device__ static void apply(const F::W& r, const X& x, int kby, int iqs, float (&out)[NC]) {
        int u[4][NC];
        float d8[4][NC];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int b = kby + r.bq8_offset + i;
            x.u(b, iqs % 8, u[i]);
            x.scales(b, d8[i]);
        }
        // q3_q8_dot_impl's arithmetic with the column-independent part done once: the four scales, and the four sets of
        // signed values (vil - vih, vih = 4 where the high bit is set, is vil | 0xFC in that byte: vil is 0..3, so nothing
        // saturates and it is __vsubss4's result)
        int vi[4], sc[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int isc = r.scale_offset + 2 * i;
            const int isc_low = isc % 8;
            const int sc_shift_low = 4 * (isc / 8);
            const int sc_low = (r.scales[isc_low] >> sc_shift_low) & 0xf;
            const int isc_high = isc % 4;
            const int sc_shift_high = 2 * (isc / 4);
            const int sc_high = ((r.scales[8 + isc_high] >> sc_shift_high) & 3) << 4;
            sc[i] = (sc_low | sc_high) - 32;
            vi[i] = ((r.vl >> (2 * i)) & 0x03030303) | (((r.vh >> i) & 0x01010101) * 252);
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            float sumf = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) sumf += d8[i][c] * (STRATA_DP4A(vi[i], u[i][c], 0) * sc[i]);
            out[c] = r.d * sumf;
        }
    }
};
struct IlQ6K {
    using F = Q6KTraits;
    static constexpr bool PM = false;
    template <int NC, class X>
    __device__ static void apply(const F::W& r, const X& x, int kby, int iqs, float (&out)[NC]) {
        int u[2][NC];
        float d8[2][NC];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            x.u(kby + r.bq8_offset + 2 * i, iqs % 8, u[i]);
            x.scales(kby + r.bq8_offset + 2 * i, d8[i]);
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const int uc[2] = {u[0][c], u[1][c]};
            const float dc[2] = {d8[0][c], d8[1][c]};
            out[c] = q6_q8_dot_impl(r.vl, r.vh, uc, r.scales, r.d, dc);
        }
    }
    // P1: the two scale bytes loaded with the rest (F::W keeps a pointer into the block, which would put their loads
    // in the compute phase); q6_q8_dot_sc is q6_q8_dot_impl's expression
    static constexpr bool HAS_PF = true;
    // a row's four lane-adjusted byte pointers (ql, qh, scales, d): a block's loads are then one IMAD.WIDE each from kbx
    struct RP { const uint8_t *ql, *qh, *sc, *d; };
    __device__ static RP rowptrs(const F::Block* __restrict__ row, int iqs) {
        const auto* b = reinterpret_cast<const uint8_t*>(row);
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        return {b + 4 * iqs, b + 128 + 4 * (8 * (iqs / 16) + iqs % 8), b + 192 + scale_offset, b + 208};   // load_int_b2's int iqs
    }
    struct WR { int vl, vh, sc0, sc1, d; };
    __device__ static RP at(const RP& rp, int kbx) {
        constexpr int S = int(sizeof(F::Block));
        return {static_cast<const uint8_t*>(madw(rp.ql, kbx, S)), static_cast<const uint8_t*>(madw(rp.qh, kbx, S)),
                static_cast<const uint8_t*>(madw(rp.sc, kbx, S)), static_cast<const uint8_t*>(madw(rp.d, kbx, S))};
    }
    __device__ static WR loadr(const RP& rp, int dv) {   // block dv past rp's: immediates off its four pointers
        constexpr int S = int(sizeof(F::Block));
        WR r;
        r.vl = ldg_nc_b2(rp.ql + dv * S, 0);
        r.vh = ldg_nc_b2(rp.qh + dv * S, 0);
        r.sc0 = ldg_nc_s8(rp.sc + dv * S);
        r.sc1 = ldg_nc_s8(rp.sc + dv * S + 4);
        r.d = ldg_nc_u16(rp.d + dv * S);
        return r;
    }
    __device__ static int xb(int iqs) { return 4 * (iqs / 16) + (iqs % 16) / 8; }
    __device__ static int xp(int iqs) { return iqs % 8; }
    template <int NC, bool FAST, class X>
    __device__ static void applyr(const WR& r, const X& x, const int* qb, const float* db, int kby, int iqs, float (&out)[NC]) {
        const int vh = r.vh >> (2 * ((iqs % 16) / 8));
        int u[2][NC];
        float d8[2][NC];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            x.qat(qb + 2 * i * X::QB, u[i]);
            x.dat(db + 2 * i * X::DB, d8[i]);
        }
        const float d = __half2float(__ushort_as_half((unsigned short) r.d));
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const int uc[2] = {u[0][c], u[1][c]};
            const float dc[2] = {d8[0][c], d8[1][c]};
            out[c] = FAST ? q6_q8_dot_fast(r.vl, vh, uc, r.sc0, r.sc1, d, dc) : q6_q8_dot_sc(r.vl, vh, uc, r.sc0, r.sc1, d, dc);
        }
    }
};

// P1: native_mmvq_il_kernel with every weight load of a group (the 4 virtual warps x R rows) issued
// before any of the group's dots - the four `if (kbx < bpr)` sections of the kernel below each sit behind their own
// branch, so a warp had one 2-row x 1-block group in flight (ncu: 432 GB/s on the qkv Q6_K matrix).  The addresses are
// clamped to the row's last block (a tail re-reads it; its dot is skipped), and with DB the next group's loads are
// issued before this group's dots (two register sets, ping-pong, no copies).  The per-lane accumulation order is the
// kernel below's: acc[v] += block(4m + v) in ascending m, ((acc0 + acc1) + acc2) + acc3, warp_sum - the same bits.
// STRATA_MMVQ_IL_PF=1: hoisted group loads (4 blocks a SM); =3: + P2's exact integer forms (the traits' applyr<FAST>);
// =4: as 3 at 5 blocks a SM; =2 / =5: as 3 / 4 with the ring (DB 2: a section's registers reloaded with the next group's
// blocks right after its dots); unset / 0: the kernel below.
template <typename I, int NC, int R, int DB, int MINB, bool FAST>
__launch_bounds__(4 * WARP, MINB)
__global__ void native_mmvq_il_pf_kernel(const typename I::F::Block* __restrict__ w, const int* __restrict__ xq,
                                         const float* __restrict__ xd, float* __restrict__ y, int n_in, int n_out) {
    using F = typename I::F;
    using WR = typename I::WR;
    constexpr int SUB = WARP / F::T;   // weight blocks a chunk covers
    const int lane = int(threadIdx.x) & (WARP - 1);
    const auto x = q8_1_cols<NC, I::PM>(xq, xd, n_in / Q8K);
    using X = std::remove_const_t<decltype(x)>;
    const int bpr = n_in / F::DIV;
    const int kqs = F::kqs(lane);
    const int sub = lane / F::T;
    const int nm = (bpr + SUB - 1) / SUB;
    for (int grp = int(blockIdx.x) * 4 + (int(threadIdx.x) >> 5); grp * R < n_out; grp += int(gridDim.x) * 4) {
        const int row0 = grp * R;
        typename I::RP wr[R];
#pragma unroll
        for (int i = 0; i < R; ++i) wr[i] = I::rowptrs(w + std::size_t(min(row0 + i, n_out - 1)) * bpr, kqs);   // a partial last group recomputes its last row
        float acc[4][R][NC];
#pragma unroll
        for (int v = 0; v < 4; ++v)
#pragma unroll
            for (int i = 0; i < R; ++i)
#pragma unroll
                for (int c = 0; c < NC; ++c) acc[v][i][c] = 0.0f;
        // the groups whose four chunks are whole blocks for every lane run with no branch at all (ptxas sinks a load
        // whose only use sits behind a branch into it, which is what put the scale loads back into the dot sections);
        // the tail group clamps its loads and predicates its dots
        const int m_full = bpr / SUB;
        // a full group's 4 x R blocks: one pointer set per row at the group's first block, the other three at immediates
        // (v * SUB blocks on); a clamped (tail) group addresses each block on its own
        auto load_group = [&](int m0, bool clamp, WR (&wv)[4][R]) {
            if (!clamp) {
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    const typename I::RP p0 = I::at(wr[i], SUB * m0 + sub);
#pragma unroll
                    for (int v = 0; v < 4; ++v) wv[v][i] = I::loadr(p0, v * SUB);
                }
            } else {
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    const int kbx = min(SUB * (m0 + v) + sub, bpr - 1);
#pragma unroll
                    for (int i = 0; i < R; ++i) wv[v][i] = I::loadr(I::at(wr[i], kbx), 0);
                }
            }
        };
        // the lane's activation ints and scales of the group's first chunk; a section's are an immediate on
        const int* xq0 = x.qbase(I::xb(kqs) + sub * F::KBY, I::xp(kqs));
        const float* xd0 = x.dbase(I::xb(kqs) + sub * F::KBY);
        auto dot_group = [&](int m0, bool pred, const WR (&wv)[4][R]) {
            const int* qg = static_cast<const int*>(madw(xq0, SUB * m0 * F::KBY, X::QB * 4));
            const float* dg = static_cast<const float*>(madw(xd0, SUB * m0 * F::KBY, X::DB * 4));
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const int kbx = SUB * (m0 + v) + sub;
                if (!pred || kbx < bpr) {
#pragma unroll
                    for (int i = 0; i < R; ++i) {
                        float o[NC];
                        I::template applyr<NC, FAST>(wv[v][i], x, qg + v * SUB * F::KBY * X::QB, dg + v * SUB * F::KBY * X::DB,
                                                     kbx * F::KBY, kqs, o);
#pragma unroll
                        for (int c = 0; c < NC; ++c) acc[v][i][c] += o[c];
                    }
                }
            }
        };
        int m0 = 0;
        if constexpr (DB == 0) {
            for (; m0 + 4 <= m_full; m0 += 4) {
                WR wv[4][R];
                load_group(m0, false, wv);
                dot_group(m0, false, wv);
            }
        } else if constexpr (DB == 2) {
            // a ring: after a section's dots its registers take the next group's blocks of that section, so the warp
            // always has about a group's worth of loads in flight at the register cost of one group (the loads sit
            // behind a uniform branch whose consumers are in the next iteration, so nothing can sink)
            WR wv[4][R];
            if (m0 + 4 <= m_full) load_group(0, false, wv);
            for (; m0 + 4 <= m_full; m0 += 4) {
                const bool more = m0 + 8 <= m_full;
                typename I::RP pn[R];
                if (more) {
#pragma unroll
                    for (int i = 0; i < R; ++i) pn[i] = I::at(wr[i], SUB * (m0 + 4) + sub);
                }
                const int* qg = static_cast<const int*>(madw(xq0, SUB * m0 * F::KBY, X::QB * 4));
                const float* dg = static_cast<const float*>(madw(xd0, SUB * m0 * F::KBY, X::DB * 4));
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    const int kbx = SUB * (m0 + v) + sub;
#pragma unroll
                    for (int i = 0; i < R; ++i) {
                        float o[NC];
                        I::template applyr<NC, FAST>(wv[v][i], x, qg + v * SUB * F::KBY * X::QB, dg + v * SUB * F::KBY * X::DB,
                                                     kbx * F::KBY, kqs, o);
#pragma unroll
                        for (int c = 0; c < NC; ++c) acc[v][i][c] += o[c];
                    }
                    if (more) {
#pragma unroll
                        for (int i = 0; i < R; ++i) wv[v][i] = I::loadr(pn[i], v * SUB);
                    }
                }
            }
        } else {   // two register sets: a group's loads issue before the previous group's dots; no branch inside the loop
            WR wa[4][R], wb[4][R];
            if (m0 + 4 <= m_full) load_group(0, false, wa);
            for (; m0 + 12 <= m_full; m0 += 8) {
                load_group(m0 + 4, false, wb);
                dot_group(m0, false, wa);
                load_group(m0 + 8, false, wa);
                dot_group(m0 + 4, false, wb);
            }
            if (m0 + 8 <= m_full) {
                load_group(m0 + 4, false, wb);
                dot_group(m0, false, wa);
                dot_group(m0 + 4, false, wb);
                m0 += 8;
            } else if (m0 + 4 <= m_full) {
                dot_group(m0, false, wa);
                m0 += 4;
            }
        }
        if (m0 < nm) {
            WR wv[4][R];
            load_group(m0, true, wv);
            dot_group(m0, true, wv);
        }
#pragma unroll
        for (int i = 0; i < R; ++i)
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                float s = acc[0][i][c];
                s += acc[1][i][c];
                s += acc[2][i][c];
                s += acc[3][i][c];
                s = warp_sum(s);
                if (lane == 0 && row0 + i < n_out) y[std::size_t(c) * n_out + row0 + i] = s;
            }
    }
}

template <typename I, int NC, int R>
__launch_bounds__(4 * WARP)
__global__ void native_mmvq_il_kernel(const typename I::F::Block* __restrict__ w, const int* __restrict__ xq,
                                      const float* __restrict__ xd, float* __restrict__ y, int n_in, int n_out) {
    using F = typename I::F;
    constexpr int SUB = WARP / F::T;   // weight blocks a chunk covers
    const int lane = int(threadIdx.x) & (WARP - 1);
    const auto x = q8_1_cols<NC, I::PM>(xq, xd, n_in / Q8K);
    const int bpr = n_in / F::DIV;
    const int kqs = F::kqs(lane);
    const int nm = (bpr + SUB - 1) / SUB;
    // A grid-stride loop over the row groups, though the grid gives each warp one: without it ptxas gives several
    // instances fewer registers (the fork measured Q6_K's at 3 columns 10-50% slower).
    for (int grp = int(blockIdx.x) * 4 + (int(threadIdx.x) >> 5); grp * R < n_out; grp += int(gridDim.x) * 4) {
        const int row0 = grp * R;
        float acc[4][R][NC];
#pragma unroll
        for (int v = 0; v < 4; ++v)
#pragma unroll
            for (int i = 0; i < R; ++i)
#pragma unroll
                for (int c = 0; c < NC; ++c) acc[v][i][c] = 0.0f;
        for (int m0 = 0; m0 < nm; m0 += 4) {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const int kbx = SUB * (m0 + v) + lane / F::T;
                if (kbx < bpr) {
#pragma unroll
                    for (int i = 0; i < R; ++i) {
                        const int row = min(row0 + i, n_out - 1);   // a partial last group recomputes its last row
                        const typename F::W wv = F::load(w + std::size_t(row) * bpr + kbx, kqs);
                        float o[NC];
                        I::template apply<NC>(wv, x, kbx * F::KBY, kqs, o);
#pragma unroll
                        for (int c = 0; c < NC; ++c) acc[v][i][c] += o[c];
                    }
                }
            }
        }
#pragma unroll
        for (int i = 0; i < R; ++i)
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                float s = acc[0][i][c];
                s += acc[1][i][c];
                s += acc[2][i][c];
                s += acc[3][i][c];
                s = warp_sum(s);
                if (lane == 0 && row0 + i < n_out) y[std::size_t(c) * n_out + row0 + i] = s;
            }
    }
}

int g_il_pf = -1;   // native_mmvq_il_pf (tests, benchmarks): the variant for every launch; -1: STRATA_MMVQ_IL_PF
int il_pf_env() {
    static const int env = [] { const char* v = std::getenv("STRATA_MMVQ_IL_PF"); return v ? std::atoi(v) : 0; }();
    return env;
}
int il_pf_table(int type, int ncols, int n_out);   // the sm_80 table below: pf << 4 | rows, 0 = today's kernel and table
// the variant of one launch (g_il_pf, else STRATA_MMVQ_IL_PF: 0 / unset today's kernel, 1 the measured table, 2-6 that
// variant for every type and shape), and with the table its rows a warp (0: the rows table's)
int il_pf_for(int type, int ncols, int n_out, int* rows) {
    *rows = 0;
    if (g_il_pf >= 0) return g_il_pf;
    const int env = il_pf_env();
    if (env != 1) return env;
    const int t = il_pf_table(type, ncols, n_out);
    *rows = t & 15;
    return t >> 4;
}
// one (type, columns, rows) launch: today's kernel, or the variant il_pf_for names for a type that has them
template <typename I, int NC, int R>
void launch_il_one(unsigned blocks, const typename I::F::Block* w, const int* xq, const float* xd, float* y, int n_in, int n_out,
                   cudaStream_t s, int pf) {
    if constexpr (I::HAS_PF) {
        switch (pf) {
        case 6: native_mmvq_il_pf_kernel<I, NC, R, 0, 4, false><<<blocks, 4 * WARP, 0, s>>>(w, xq, xd, y, n_in, n_out); return;
        case 2: native_mmvq_il_pf_kernel<I, NC, R, 2, 4, true><<<blocks, 4 * WARP, 0, s>>>(w, xq, xd, y, n_in, n_out); return;
        case 3: native_mmvq_il_pf_kernel<I, NC, R, 0, 4, true><<<blocks, 4 * WARP, 0, s>>>(w, xq, xd, y, n_in, n_out); return;
        case 4: native_mmvq_il_pf_kernel<I, NC, R, 0, 5, true><<<blocks, 4 * WARP, 0, s>>>(w, xq, xd, y, n_in, n_out); return;
        case 5: native_mmvq_il_pf_kernel<I, NC, R, 2, 5, true><<<blocks, 4 * WARP, 0, s>>>(w, xq, xd, y, n_in, n_out); return;
        default: break;
        }
    }
    native_mmvq_il_kernel<I, NC, R><<<blocks, 4 * WARP, 0, s>>>(w, xq, xd, y, n_in, n_out);
}

template <typename I, int R>
void launch_il(const void* weights, const void* x_il, float* y, int n_in, int n_out, int ncols, cudaStream_t s, int pf) {
    const auto* w = static_cast<const typename I::F::Block*>(weights);
    const unsigned blocks = unsigned(((n_out + R - 1) / R + 3) / 4);
    if (ncols == 1) {   // 143-dense: one column reads the plain q8_1 blocks (x_il is then the caller's x_q8_1)
        launch_il_one<I, 1, R>(blocks, w, static_cast<const int*>(x_il), nullptr, y, n_in, n_out, s, pf);
        return;
    }
    const Q81IlParts parts = q8_1_il_parts(x_il, n_in, ncols);
    const int* xq = I::PM ? parts.pm : parts.bm;
    const float* xd = parts.d;
    switch (ncols) {
#define STRATA_IL_CASE(N) case N: launch_il_one<I, N, R>(blocks, w, xq, xd, y, n_in, n_out, s, pf); break;
        STRATA_IL_CASE(2) STRATA_IL_CASE(3) STRATA_IL_CASE(4)
#undef STRATA_IL_CASE
        // 147-verify: 5..8 columns (the cp = 8 copy) at 1 or 2 rows a warp; launch_il_rows maps 4 rows to 2 there
#define STRATA_IL_CASE(N) case N: if constexpr (R <= 2) launch_il_one<I, N, R>(blocks, w, xq, xd, y, n_in, n_out, s, pf); break;
        STRATA_IL_CASE(5) STRATA_IL_CASE(6) STRATA_IL_CASE(7) STRATA_IL_CASE(8)
#undef STRATA_IL_CASE
        default: throw std::invalid_argument("native_mmvq_il requires 1 <= ncols <= 8");
    }
}

template <typename I>
void launch_il_rows(int r, const void* weights, const void* x_il, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    int pf_rows = 0;
    const int pf = il_pf_for(I::F::TYPE, ncols, n_out, &pf_rows);
    if (pf_rows != 0) r = pf_rows;   // the pf table measured its rows with its kernel
    if (ncols > 4 && r == 4) r = 2;   // 5..8 columns: 1 or 2 rows a warp
    switch (r) {
    case 1: launch_il<I, 1>(weights, x_il, y, n_in, n_out, ncols, s, pf); break;
    case 4: launch_il<I, 4>(weights, x_il, y, n_in, n_out, ncols, s, pf); break;
    default: launch_il<I, 2>(weights, x_il, y, n_in, n_out, ncols, s, pf); break;
    }
}

// The interleaved copy (native_mmvq.hpp) of columns already quantized into plain q8_1 blocks, whoever wrote them (the
// quantizer, a fused norm): the same bytes, one int of a block a thread.
__global__ void native_q8_1_interleave_kernel(const Q81Block* __restrict__ y, int* __restrict__ bm, int* __restrict__ pm,
                                              float* __restrict__ dl, int nb, int n_total, int cp) {
    const int i = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (i >= n_total) return;
    const int c = i / (nb * 8), r = i - c * nb * 8, b = r / 8, p = r % 8;
    const Q81Block& blk = y[std::size_t(c) * nb + b];
    const int q = reinterpret_cast<const int*>(blk.qs)[p];
    bm[(std::size_t(b) * 8 + p) * cp + c] = q;
    pm[(std::size_t(p) * nb + b) * cp + c] = q;
    if (p == 0) dl[std::size_t(b) * cp + c] = __low2float(blk.ds);
}

// native_quantize_q8_1_kernel and native_q8_1_interleave_kernel in one launch: the same q8_1 blocks, and the
// interleaved copy's ints and scales (an int is four codes, gathered from lanes 4p..4p+3 with two shuffles).
__global__ void native_quantize_q8_1_il_kernel(const float* __restrict__ x, Q81Block* __restrict__ y,
                                               int* __restrict__ bm, int* __restrict__ pm, float* __restrict__ dl,
                                               int n_in, int n_total, int cp) {
    const int i = int(blockIdx.x) * QUANT_THREADS + int(threadIdx.x);
    if (i >= n_total) return;   // n_total is a multiple of 32: only whole warps return
    const float xi = x[i];
    const float amax = warp_max(fabsf(xi));
    const float sum = warp_sum(xi);
    const float d = q8_1_finite(amax / 127.0f);
    const int8_t q = q8_1_quant(xi, d, amax);
    y[i / Q8K].qs[i % Q8K] = q;
    const auto ds = q8_1_ds(d, sum);
    if (i % Q8K == 0) y[i / Q8K].ds = ds;
    const int lane = int(threadIdx.x) & 31;
    unsigned v = unsigned(uint8_t(q)) << (8 * (lane & 3));
    v |= __shfl_xor_sync(0xffffffffu, v, 1);
    v |= __shfl_xor_sync(0xffffffffu, v, 2);
    const int nb = n_in / Q8K, c = i / n_in, b = (i - c * n_in) / Q8K, pp = lane >> 2;
    if ((lane & 3) == 0) {
        bm[(std::size_t(b) * 8 + pp) * cp + c] = int(v);
        pm[(std::size_t(pp) * nb + b) * cp + c] = int(v);
    }
    if (lane == 0) dl[std::size_t(b) * cp + c] = __low2float(ds);
}

#endif  // !__HIPCC__

void validate_shape(int n_in, int ncols, int block_elems = Q8K) {
    if (n_in <= 0 || n_in % block_elems != 0) {
        throw std::invalid_argument("native MMVQ requires n_in > 0 and divisible by its block element count");
    }
    if (ncols < 1 || ncols > MAX_NCOLS) throw std::invalid_argument("native MMVQ requires 1 <= ncols <= MAX_NCOLS");
}
void validate_pointer(const void* p) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % 4 != 0) {
        throw std::invalid_argument("native MMVQ requires non-null 4-byte aligned device pointers");
    }
}
void validate_stream(void* stream) {
    if (!stream) throw std::invalid_argument("native MMVQ requires an explicit non-null CUDA stream");
}
void launch_check() {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("native MMVQ launch: ") + cudaGetErrorString(error));
    }
}


#if defined(STRATA_HIP_GFX906)
// ---- AMD (wave64) layout: one wavefront per R rows, four wavefronts per block, the row's blocks strided over the
// 64 lanes exactly as the CUDA kernels stride them over a block (kbx = lane / T, stride 64 / T), and a 64-lane
// butterfly instead of the LDS partials and __syncthreads of the one-block-per-row layout: on gfx906 a block per
// 2 KB row spent most of its time being launched and joined.  The SAME kernel serves every column count 1..8, so a
// column's sums do not depend on how many columns (verify tokens) ride along.  STRATA_MMVQ_WAVE=0: the CUDA layout.
bool g_wave_off = std::getenv("STRATA_MMVQ_WAVE") && std::string(std::getenv("STRATA_MMVQ_WAVE")) == "0";
template<typename F, int NCOLS, int R>
__launch_bounds__(256)
__global__ void native_mmvq_wave_kernel(const typename F::Block* __restrict__ w, const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int BPIW = 64 / F::T;
    static_assert(64 % F::T == 0, "a block's threads must tile the wavefront");
    const int lane = int(threadIdx.x) & 63;
    const int row0 = (int(blockIdx.x) * 4 + (int(threadIdx.x) >> 6)) * R;
    if (row0 >= n_out) return;
    const int blocks_per_row = n_in / F::DIV;
    const int x_stride = n_in / Q8K;
    const int kqs = F::kqs(lane);
    float tmp[NCOLS][R] = {};
    for (int kbx = lane / F::T; kbx < blocks_per_row; kbx += BPIW) {
        const int kby = kbx * F::KBY;
#pragma unroll
        for (int i = 0; i < R; ++i) {
            if (row0 + i < n_out) {
                const typename F::W wv = F::load(w + std::size_t(row0 + i) * blocks_per_row + kbx, kqs);
#pragma unroll
                for (int j = 0; j < NCOLS; ++j) tmp[j][i] += F::apply(wv, x + std::size_t(j) * x_stride + kby, kqs);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < NCOLS; ++j)
#pragma unroll
        for (int i = 0; i < R; ++i) {
            float v = tmp[j][i];
#pragma unroll
            for (int off = 32; off > 0; off >>= 1) v += __shfl_xor(v, off, 64);
            if (lane == 0 && row0 + i < n_out) y[std::size_t(j) * n_out + row0 + i] = v;
        }
}
template<typename F, int NCOLS>
void wave_launch_n(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, cudaStream_t s) {
    const auto* w = static_cast<const typename F::Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    constexpr int R = 1;
    const unsigned blocks = unsigned((std::size_t(n_out) + 4 * R - 1) / (4 * R));
    native_mmvq_wave_kernel<F, NCOLS, R><<<blocks, 256, 0, s>>>(w, x, y, n_in, n_out);
}
template<typename F>
void wave_launch(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    const auto s = static_cast<cudaStream_t>(stream);
    switch (ncols) {
        case 1: wave_launch_n<F, 1>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 2: wave_launch_n<F, 2>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 3: wave_launch_n<F, 3>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 4: wave_launch_n<F, 4>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 5: wave_launch_n<F, 5>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 6: wave_launch_n<F, 6>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 7: wave_launch_n<F, 7>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 8: wave_launch_n<F, 8>(weights, x_q8_1, y, n_in, n_out, s); break;
#if STRATA_VERIFY_MAX_T > 8
        case 9: wave_launch_n<F, 9>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 10: wave_launch_n<F, 10>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 11: wave_launch_n<F, 11>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 12: wave_launch_n<F, 12>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 13: wave_launch_n<F, 13>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 14: wave_launch_n<F, 14>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 15: wave_launch_n<F, 15>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 16: wave_launch_n<F, 16>(weights, x_q8_1, y, n_in, n_out, s); break;
#endif
        default: throw std::invalid_argument("native MMVQ (wave) requires 1 <= ncols <= MAX_NCOLS");
    }
}
#define STRATA_WAVE_MMVQ(...) \
    if (!g_wave_off) { wave_launch<__VA_ARGS__>(weights, x_q8_1, y, n_in, n_out, ncols, stream); launch_check(); return; }
#else
#define STRATA_WAVE_MMVQ(...)
#endif

template<typename Weight, int Qi>
void small_mmvq(const void* weights, const void* x_q8_1, float* y,
                int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 32);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    STRATA_WAVE_MMVQ(SmallTraits<Weight, Qi>)
    if (ncols > 1) {
        launch_multi<SmallTraits<Weight, Qi>>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Weight*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 32 < 2 * WARPS * WARP / Qi) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_small_mmvq_kernel<Weight, Qi, true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_small_mmvq_kernel<Weight, Qi, false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

template<typename Weight, int Qi>
void small_f32(const void* weights, const float* x, void* scratch_q8_1,
               float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 32);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    small_mmvq<Weight, Qi>(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

// STRATA_Q8_SM60=1 (opt-in, written for Pascal GP100): every Q8_0 dense matrix and the head get the packed planes
// below and run q8sm60::kernel on them (q8_sm60.cuh) instead of the exact kernels.
bool q8_sm60_on() {
#if defined(__HIPCC__)
    return false;   // a CUDA kernel for one NVIDIA chip: HIP builds never compile it
#else
    static const bool on = [] {
        const char* v = std::getenv("STRATA_Q8_SM60");
        const bool yes = v && v[0] == '1' && v[1] == '\0';
        if (yes) std::fprintf(stderr, "strata q8 sm60: Q8_0 decode GEMVs on the Pascal kernel (STRATA_Q8_SM60=1)\n");
        return yes;
    }();
    return on;
#endif
}
#if !defined(__HIPCC__)
#include "q8_sm60.cuh"
#endif

// ============================ STRATA_Q8_PACKED=1: the packed Q8_0 decode layout (opt-in) ============================
//
// A lossless load-time repack of a Q8_0 matrix: the SAME int8 values in a qs plane (row-major, n_in bytes per row)
// followed by the SAME fp16 scales in a d plane (row-major, n_in / 32 per row) - no requantization, same bytes.
// The GGUF's 34-byte blocks leave every int 2-byte aligned (load_int_b2 = two 16-bit loads); the planes give each
// thread one aligned 8-byte run, read with streaming (non-temporal) loads. Every thread owns exactly the bytes of
// the ncols = 1 / multi-column EXACT layout (block kbx = tid / 4 + 32 i, ints 2 (tid % 4) and +1), accumulates them
// in the same order with the same expression, and the cross-warp sum and XOR tree are the same: every output is
// BITWISE equal to native_small_mmvq_kernel / native_mmvq_multi_kernel<SmallTraits<Q80Block, 8>, NC, 4, 1>.
// Eligible: 1024 <= n_in <= 4096 and n_out >= 2048, run as 640 persistent workgroups (a grid-stride over rows, two
// partial buffers). Measured on gfx1151 (S26, 8 real UD-IQ4_XS layers per shape, paired, 1-5 columns): 2560 ->
// 6144/10240/12288 +10-12% per call (engine trace: -10.7% on those calls). 6144 -> 2560 (one workgroup per row)
// was +2-4% standalone but -3% in the engine, and rows < 2048 lose: neither is packed.
#if defined(__HIPCC__)
#define STRATA_Q8P_LOAD(p) __builtin_nontemporal_load(p)
#else
#define STRATA_Q8P_LOAD(p) __ldcs(p)
#endif
constexpr int Q8P_PERSIST_GRID = 640;
constexpr int Q8P_PERSIST_MAX_IN = 4096;

template<int NCOLS, bool TS = false>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q8_0_packed_kernel(const int8_t* __restrict__ qs, const half* __restrict__ dpl,
                                          const Q81Block* __restrict__ x, float* __restrict__ y,
                                          int n_in, int n_out, int row_step) {
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int nb = n_in / 32;
    const int kqs = 2 * (tid % 4);
    __shared__ float partial[2][WARPS - 1][NCOLS][WARP];
    int buf = 0;
    for (int row = int(blockIdx.x); row < n_out; row += row_step, buf ^= 1) {
        float tmp[NCOLS] = {};
        for (int kbx = tid / 4; kbx < nb; kbx += 2 * WARPS * WARP / 8) {
            const std::size_t block = std::size_t(row) * nb + kbx;
            const int* p = reinterpret_cast<const int*>(qs + block * 32) + kqs;
            const int v0 = STRATA_Q8P_LOAD(p);
            const int v1 = STRATA_Q8P_LOAD(p + 1);
            const float d0 = dpl[block];
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                const Q81Block* xb = x + std::size_t(j) * nb + kbx;
                const int* u = reinterpret_cast<const int*>(xb->qs);
                int sumi = 0;
                sumi = STRATA_DP4A(v0, u[kqs], sumi);
                sumi = STRATA_DP4A(v1, u[kqs + 1], sumi);
                const float d1 = __low2float(xb->ds);
                tmp[j] += d0 * d1 * float(sumi);
            }
        }
        if (threadIdx.y > 0) {
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) partial[buf][threadIdx.y - 1][j][threadIdx.x] = tmp[j];
        }
        __syncthreads();
        if (TS && threadIdx.y == 0) {   // S26 STRATA_TSUM=1: one transposed butterfly, bitwise the same sums
            constexpr int P = s26ts::pow2_ceil(NCOLS);
            float v[P];
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
                for (int l = 0; l < WARPS - 1; ++l) tmp[j] += partial[buf][l][j][threadIdx.x];
                v[j] = tmp[j];
            }
#pragma unroll
            for (int j = NCOLS; j < P; ++j) v[j] = 0.0f;
            const float sum = s26ts::tsum<P>(v, int(threadIdx.x));
            const int j = s26ts::tsum_token<P>(int(threadIdx.x));
            if (int(threadIdx.x) == s26ts::tsum_lane<P>(j) && j < NCOLS) y[std::size_t(j) * n_out + row] = sum;
        } else if (threadIdx.y == 0) {
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
                for (int l = 0; l < WARPS - 1; ++l) tmp[j] += partial[buf][l][j][threadIdx.x];
                tmp[j] = warp_sum(tmp[j]);
                if (threadIdx.x == 0) y[std::size_t(j) * n_out + row] = tmp[j];
            }
        }
    }
}

struct Q8Packed { const int8_t* qs; const half* d; int n_in, n_out; };
std::unordered_map<const void*, Q8Packed>& q8_packed_registry() {
    static std::unordered_map<const void*, Q8Packed> registry;
    return registry;
}

template<int NCOLS>
void q8_packed_launch(const Q8Packed& w, const Q81Block* x, float* y, cudaStream_t s) {
    const bool persist = w.n_in <= Q8P_PERSIST_MAX_IN;
    const int grid = persist ? (std::min)(w.n_out, Q8P_PERSIST_GRID) : w.n_out;
    if (s26_tsum_on())
        native_q8_0_packed_kernel<NCOLS, true><<<unsigned(grid), dim3(WARP, WARPS), 0, s>>>(w.qs, w.d, x, y, w.n_in, w.n_out,
                                                                                          persist ? grid : w.n_out);
    else
        native_q8_0_packed_kernel<NCOLS><<<unsigned(grid), dim3(WARP, WARPS), 0, s>>>(w.qs, w.d, x, y, w.n_in, w.n_out,
                                                                                    persist ? grid : w.n_out);
}

bool q8_packed_shape(int n_in, int n_out) {   // the shapes native_q8_0_packed_kernel was measured on
    return n_in > 0 && n_in % 32 == 0 && n_in / 32 >= 2 * WARPS * WARP / 8 && n_in <= Q8P_PERSIST_MAX_IN &&
           n_out >= 2048;
}

bool q8_packed_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    auto& registry = q8_packed_registry();
#if !defined(__HIPCC__)
    if (q8_sm60_on() && !registry.empty()) {
        const auto found = registry.find(weights);
        if (found != registry.end() && found->second.n_in == n_in && found->second.n_out == n_out) {
            validate_pointer(x_q8_1);
            validate_pointer(y);
            validate_stream(stream);
            // (the kernel takes at most q8sm60::MAX_COLS columns; a 9..16-row window is run in passes, each column bitwise
            // the one-column result, as native_mmvq does for more than MAX_NCOLS)
            bool all = true;
            for (int j = 0; all && j < ncols; j += q8sm60::MAX_COLS) {
                const int nj = ncols - j < q8sm60::MAX_COLS ? ncols - j : q8sm60::MAX_COLS;
                all = q8sm60::launch(found->second.qs, found->second.d,
                                     static_cast<const Q81Block*>(x_q8_1) + std::size_t(j) * std::size_t(n_in / Q8K),
                                     y + std::size_t(j) * std::size_t(n_out), n_in, n_out, nj, static_cast<cudaStream_t>(stream));
            }
            if (all) {
                launch_check();
                return true;
            }
            // a shape the Pascal kernel declines: the exact packed kernel only where it was measured, else the GGUF layout
            if (!q8_packed_shape(n_in, n_out)) return false;
        }
    }
#endif
    if (registry.empty() || (ncols > 1 && !g_multi_exact)) return false;   // packed = the EXACT layout only
    const auto found = registry.find(weights);
    if (found == registry.end() || found->second.n_in != n_in || found->second.n_out != n_out) return false;
    validate_shape(n_in, ncols, 32);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    switch (ncols) {
        case 1: q8_packed_launch<1>(found->second, x, y, s); break;
        case 2: q8_packed_launch<2>(found->second, x, y, s); break;
        case 3: q8_packed_launch<3>(found->second, x, y, s); break;
        case 4: q8_packed_launch<4>(found->second, x, y, s); break;
        case 5: q8_packed_launch<5>(found->second, x, y, s); break;
        case 6: q8_packed_launch<6>(found->second, x, y, s); break;
        case 7: q8_packed_launch<7>(found->second, x, y, s); break;
        case 8: q8_packed_launch<8>(found->second, x, y, s); break;
#if STRATA_VERIFY_MAX_T > 8
        case 9: q8_packed_launch<9>(found->second, x, y, s); break;
        case 10: q8_packed_launch<10>(found->second, x, y, s); break;
        case 11: q8_packed_launch<11>(found->second, x, y, s); break;
        case 12: q8_packed_launch<12>(found->second, x, y, s); break;
        case 13: q8_packed_launch<13>(found->second, x, y, s); break;
        case 14: q8_packed_launch<14>(found->second, x, y, s); break;
        case 15: q8_packed_launch<15>(found->second, x, y, s); break;
        case 16: q8_packed_launch<16>(found->second, x, y, s); break;
#endif
        default: throw std::invalid_argument("native MMVQ requires 1 <= ncols <= MAX_NCOLS");
    }
    launch_check();
    return true;
}


// ============================ STRATA_Q6_PACKED=1: the packed Q6_K head layout (opt-in) ============================
//
// The output heads (the main head, 248320 rows, and the MTP draft head's row subset) are Q6_K matrices of 2560-wide
// rows: 10 blocks of 210 bytes each, so every int load is 2-byte aligned and a row is 2100 bytes. The packed copy
// holds the SAME bytes as four planes - ql (128 B per block), qh (64 B), scales (16 B), d (2 B) - so a thread's
// two words are aligned dword loads (non-temporal). Thread ownership (block tid / 32 + 4 i, word tid % 32), the
// q6_q8_dot_impl expression, the per-thread order and the reduction are those of native_q6_k_mmvq_kernel<false> /
// native_mmvq_multi_kernel<Q6KTraits, NC, 4, 1>: every output is bitwise equal. Persistent workgroups.
int g_q6p_grid_override = 0;   // the self-test's grid sweep (STRATA_Q6P_SELFTEST=2)
int q6p_grid() {
    if (g_q6p_grid_override > 0) return g_q6p_grid_override;
    static const int grid = [] {
        const char* v = std::getenv("STRATA_Q6P_GRID");
        const int g = v ? std::atoi(v) : 0;
        return g > 0 ? g : 640;
    }();
    return grid;
}

template<int NCOLS, bool TS = false>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q6_k_packed_kernel(const int* __restrict__ ql, const int* __restrict__ qh,
                                          const int8_t* __restrict__ sc, const half* __restrict__ dpl,
                                          const Q81Block* __restrict__ x, float* __restrict__ y,
                                          int n_in, int n_out, int row_step) {
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int nb = n_in / 256;
    const int x_stride = n_in / Q8K;
    const int iqs = tid % 32;
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    const int qh_word = 8 * (iqs / 16) + iqs % 8;
    __shared__ float partial[2][WARPS - 1][NCOLS][WARP];
    int buf = 0;
    for (int row = int(blockIdx.x); row < n_out; row += row_step, buf ^= 1) {
        float tmp[NCOLS] = {};
        for (int kbx = tid / 32; kbx < nb; kbx += WARPS * WARP / 32) {
            const std::size_t block = std::size_t(row) * nb + kbx;
            const int vl = STRATA_Q8P_LOAD(ql + block * 32 + iqs);
            const int vh = STRATA_Q8P_LOAD(qh + block * 16 + qh_word) >> vh_shift;
            const int8_t* scales = sc + block * 16 + scale_offset;
            const float d = dpl[block];
            const int kby = kbx * 8;
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                const Q81Block* xb = x + std::size_t(j) * x_stride + kby;
                int u[2];
                float d8[2];
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    u[i] = reinterpret_cast<const int*>(xb[bq8_offset + 2 * i].qs)[iqs % 8];
                    d8[i] = __low2float(xb[bq8_offset + 2 * i].ds);
                }
                tmp[j] += q6_q8_dot_impl(vl, vh, u, scales, d, d8);
            }
        }
        if (threadIdx.y > 0) {
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) partial[buf][threadIdx.y - 1][j][threadIdx.x] = tmp[j];
        }
        __syncthreads();
        if (TS && threadIdx.y == 0) {   // S26 STRATA_TSUM=1: one transposed butterfly, bitwise the same sums
            constexpr int P = s26ts::pow2_ceil(NCOLS);
            float v[P];
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
                for (int l = 0; l < WARPS - 1; ++l) tmp[j] += partial[buf][l][j][threadIdx.x];
                v[j] = tmp[j];
            }
#pragma unroll
            for (int j = NCOLS; j < P; ++j) v[j] = 0.0f;
            const float sum = s26ts::tsum<P>(v, int(threadIdx.x));
            const int j = s26ts::tsum_token<P>(int(threadIdx.x));
            if (int(threadIdx.x) == s26ts::tsum_lane<P>(j) && j < NCOLS) y[std::size_t(j) * n_out + row] = sum;
        } else if (threadIdx.y == 0) {
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
                for (int l = 0; l < WARPS - 1; ++l) tmp[j] += partial[buf][l][j][threadIdx.x];
                tmp[j] = warp_sum(tmp[j]);
                if (threadIdx.x == 0) y[std::size_t(j) * n_out + row] = tmp[j];
            }
        }
    }
}

bool g_q6_bypass = false;   // the load-time self-test times the GGUF-layout kernels with the copy registered

// the packed planes of one Q6_K matrix (owned: allocated by native_q6_k_pack, freed by native_q6_k_unpack)
struct Q6Packed { void* base; const int* ql; const int* qh; const int8_t* sc; const half* d; int n_in, n_out; };
std::unordered_map<const void*, Q6Packed>& q6_packed_registry() {
    static std::unordered_map<const void*, Q6Packed> registry;
    return registry;
}

template<int NCOLS>
void q6_packed_launch(const Q6Packed& w, const Q81Block* x, float* y, cudaStream_t s) {
    const int grid = (std::min)(w.n_out, q6p_grid());
    if (s26_tsum_on())
        native_q6_k_packed_kernel<NCOLS, true><<<unsigned(grid), dim3(WARP, WARPS), 0, s>>>(w.ql, w.qh, w.sc, w.d, x, y, w.n_in,
                                                                                          w.n_out, grid);
    else
        native_q6_k_packed_kernel<NCOLS><<<unsigned(grid), dim3(WARP, WARPS), 0, s>>>(w.ql, w.qh, w.sc, w.d, x, y, w.n_in,
                                                                                    w.n_out, grid);
}

bool q6_packed_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    auto& registry = q6_packed_registry();
    if (registry.empty() || g_q6_bypass || (ncols > 1 && !g_multi_exact)) return false;
    const auto found = registry.find(weights);
    if (found == registry.end() || found->second.n_in != n_in || found->second.n_out != n_out) return false;
    validate_shape(n_in, ncols, 256);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    switch (ncols) {
        case 1: q6_packed_launch<1>(found->second, x, y, s); break;
        case 2: q6_packed_launch<2>(found->second, x, y, s); break;
        case 3: q6_packed_launch<3>(found->second, x, y, s); break;
        case 4: q6_packed_launch<4>(found->second, x, y, s); break;
        case 5: q6_packed_launch<5>(found->second, x, y, s); break;
        case 6: q6_packed_launch<6>(found->second, x, y, s); break;
        case 7: q6_packed_launch<7>(found->second, x, y, s); break;
        case 8: q6_packed_launch<8>(found->second, x, y, s); break;
#if STRATA_VERIFY_MAX_T > 8
        case 9: q6_packed_launch<9>(found->second, x, y, s); break;
        case 10: q6_packed_launch<10>(found->second, x, y, s); break;
        case 11: q6_packed_launch<11>(found->second, x, y, s); break;
        case 12: q6_packed_launch<12>(found->second, x, y, s); break;
        case 13: q6_packed_launch<13>(found->second, x, y, s); break;
        case 14: q6_packed_launch<14>(found->second, x, y, s); break;
        case 15: q6_packed_launch<15>(found->second, x, y, s); break;
        case 16: q6_packed_launch<16>(found->second, x, y, s); break;
#endif
        default: throw std::invalid_argument("native MMVQ requires 1 <= ncols <= MAX_NCOLS");
    }
    launch_check();
    return true;
}
} // namespace

bool native_q8_0_packed_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("STRATA_Q8_PACKED");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return enabled || q8_sm60_on();
}

bool native_q8_0_packed_eligible(int n_in, int n_out) {
    if (q8_sm60_on()) return n_in >= 128 && n_in % 32 == 0 && n_out > 0;   // the Pascal kernel takes any of them
    return q8_packed_shape(n_in, n_out);
}

namespace {
// packed copies made here (the head), freed by release; null: packed in place (release only unregisters it)
std::unordered_map<const void*, void*>& q8_sm60_owned() {
    static std::unordered_map<const void*, void*> owned;
    return owned;
}
bool q8_sm60_pack_impl(const void* weights, int n_in, int n_out, const char* what, bool quiet);
} // namespace

bool native_q8_0_sm60_pack(const void* weights, int n_in, int n_out, const char* what) {
    return q8_sm60_pack_impl(weights, n_in, n_out, what, false);
}
bool native_q8_0_sm60_pack_quiet(const void* weights, int n_in, int n_out, const char* what) {
    return q8_sm60_pack_impl(weights, n_in, n_out, what, true);
}

namespace {
bool q8_sm60_pack_impl(const void* weights, int n_in, int n_out, const char* what, bool quiet) {
#if defined(__HIPCC__)
    (void) weights; (void) n_in; (void) n_out; (void) what; (void) quiet;
    return false;
#else
    if (!q8_sm60_on() || !weights || !native_q8_0_packed_eligible(n_in, n_out)) return false;
    if (q8_packed_registry().count(weights)) return true;
    const std::size_t count = std::size_t(n_in / 32) * std::size_t(n_out);
    const std::size_t bytes = count * 34;
    void* packed = nullptr;
    if (cudaMalloc(&packed, bytes) != cudaSuccess) {
        cudaGetLastError();
        std::fprintf(stderr, "strata q8 sm60: no VRAM for the packed %s (%.0f MiB); it keeps the GGUF layout\n", what,
                     double(bytes) / 1048576.0);
        return false;
    }
    auto* qs = static_cast<int8_t*>(packed);
    auto* d = reinterpret_cast<half*>(qs + std::size_t(n_in) * n_out);
    q8sm60::pack_kernel<<<unsigned((count + 255) / 256), 256>>>(static_cast<const Q80Block*>(weights), qs, d, count);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        cudaGetLastError();
        cudaFree(packed);
        std::fprintf(stderr, "strata q8 sm60: packing the %s failed; it keeps the GGUF layout\n", what);
        return false;
    }
    q8_packed_registry()[weights] = Q8Packed{qs, d, n_in, n_out};
    q8_sm60_owned()[weights] = packed;
    if (!quiet)
        std::fprintf(stderr, "strata q8 sm60: packed %s (%d x %d, %.0f MiB more VRAM)\n", what, n_out, n_in,
                     double(bytes) / 1048576.0);
    return true;
#endif
}
} // namespace

void native_q8_0_sm60_release(const void* weights) {
    auto& owned = q8_sm60_owned();
    const auto it = owned.find(weights);
    if (it == owned.end()) return;
    q8_packed_registry().erase(weights);
    if (it->second) cudaFree(it->second);
    owned.erase(it);
}

void native_q8_0_pack_host(const void* gguf_blocks, void* out, int n_in, int n_out) {
    const auto* blocks = static_cast<const Q80Block*>(gguf_blocks);
    auto* qs = static_cast<int8_t*>(out);
    auto* d = reinterpret_cast<half*>(qs + std::size_t(n_in) * n_out);
    const std::size_t count = std::size_t(n_in / 32) * n_out;
    for (std::size_t b = 0; b < count; ++b) {
        std::memcpy(qs + b * 32, blocks[b].qs, 32);
        std::memcpy(d + b, &blocks[b].d, sizeof(half));
    }
}

void native_q8_0_packed_register(const void* gguf_weights, const void* packed, int n_in, int n_out) {
    validate_pointer(gguf_weights);
    validate_pointer(packed);
    if (!native_q8_0_packed_eligible(n_in, n_out)) throw std::invalid_argument("Q8_0 packed: ineligible shape");
    const auto* qs = static_cast<const int8_t*>(packed);
    q8_packed_registry()[gguf_weights] =
        Q8Packed{qs, reinterpret_cast<const half*>(qs + std::size_t(n_in) * n_out), n_in, n_out};
}

void native_q8_0_packed_unregister(const void* gguf_weights) { q8_packed_registry().erase(gguf_weights); }

bool native_q8_0_sm60_pack_in_place(void* weights, int n_in, int n_out, const char* what) {
#if defined(__HIPCC__)
    (void) weights; (void) n_in; (void) n_out; (void) what;
    return false;
#else
    if (!q8_sm60_on() || !weights || !native_q8_0_packed_eligible(n_in, n_out)) return false;
    if (q8_packed_registry().count(weights)) return true;
    // The GGUF blocks are gone afterwards, so only where q8_packed_mmvq can never fall back to the GGUF-layout kernels
    // (every column count fits the kernel's shared-memory planes) and its 16-byte weight loads stay aligned
    if (q8sm60::smem_bytes(q8sm60::MAX_COLS, n_in) > q8sm60::MAX_SMEM || reinterpret_cast<std::uintptr_t>(weights) % 16 != 0)
        return native_q8_0_sm60_pack(weights, n_in, n_out, what);
    const std::size_t bytes = std::size_t(n_in / 32) * std::size_t(n_out) * 34;   // both layouts: 34 bytes per 32
    std::vector<uint8_t> blocks(bytes), planes(bytes);
    if (cudaMemcpy(blocks.data(), weights, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaGetLastError();
        std::fprintf(stderr, "strata q8 sm60: reading the %s failed; it keeps the GGUF layout\n", what);
        return false;
    }
    native_q8_0_pack_host(blocks.data(), planes.data(), n_in, n_out);
    if (cudaMemcpy(weights, planes.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaGetLastError();
        // a partial write leaves neither layout: put the blocks back
        const bool restored = cudaMemcpy(weights, blocks.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess;
        cudaGetLastError();
        std::fprintf(stderr, "strata q8 sm60: writing the packed %s failed; %s\n", what,
                     restored ? "it keeps the GGUF layout" : "its GGUF blocks could not be restored either");
        return false;
    }
    native_q8_0_packed_register(weights, weights, n_in, n_out);
    q8_sm60_owned()[weights] = nullptr;
    return true;   // the caller says what it packed
#endif
}

bool native_q6_k_packed_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("STRATA_Q6_PACKED");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return enabled;
}

bool native_q6_k_pack(const void* weights, int n_in, int n_out, const char* what) {
    if (!weights || n_in <= 0 || n_in % 256 || n_in / 256 < WARPS * WARP / 32 || n_out < 2048) return false;
    if (q6_packed_registry().count(weights)) return true;
    const std::size_t nb = std::size_t(n_in / 256) * std::size_t(n_out);
    const std::size_t bytes = nb * sizeof(Q6KBlock);
    std::vector<Q6KBlock> host(nb);
    if (cudaMemcpy(host.data(), weights, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) { cudaGetLastError(); return false; }
    // planes: ql | qh | scales | d, each 256-byte aligned
    const std::size_t o_qh = (nb * 128 + 255) & ~std::size_t(255);
    const std::size_t o_sc = (o_qh + nb * 64 + 255) & ~std::size_t(255);
    const std::size_t o_d = (o_sc + nb * 16 + 255) & ~std::size_t(255);
    const std::size_t total = o_d + nb * 2;
    std::vector<uint8_t> packed(total, 0);
    for (std::size_t b = 0; b < nb; ++b) {
        std::memcpy(packed.data() + b * 128, host[b].ql, 128);
        std::memcpy(packed.data() + o_qh + b * 64, host[b].qh, 64);
        std::memcpy(packed.data() + o_sc + b * 16, host[b].scales, 16);
        std::memcpy(packed.data() + o_d + b * 2, &host[b].d, 2);
    }
    void* dev = nullptr;
    if (cudaMalloc(&dev, total) != cudaSuccess) { cudaGetLastError(); return false; }
    if (cudaMemcpy(dev, packed.data(), total, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(dev); cudaGetLastError(); return false;
    }
    auto* base = static_cast<uint8_t*>(dev);
    q6_packed_registry()[weights] = Q6Packed{dev, reinterpret_cast<const int*>(base),
                                             reinterpret_cast<const int*>(base + o_qh),
                                             reinterpret_cast<const int8_t*>(base + o_sc),
                                             reinterpret_cast<const half*>(base + o_d), n_in, n_out};
    std::fprintf(stderr, "strata: STRATA_Q6_PACKED=1 packed the %s (%d x %d, +%.1f MiB)\n", what, n_out, n_in,
                 total / 1048576.0);
    // STRATA_Q6P_SELFTEST=1: bitwise check against the GGUF-layout kernels (1..4 columns) and paired timings
    if (const char* v = std::getenv("STRATA_Q6P_SELFTEST"); v && (v[0] == '1' || v[0] == '2')) {
        cudaStream_t s = nullptr;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        const int nq = n_in / Q8K;
        std::vector<Q81Block> hx(std::size_t(4) * nq);
        uint32_t r = 12345u;
        for (auto& blk : hx) {
            for (auto& q : blk.qs) { r = r * 1664525u + 1013904223u; q = int8_t(int((r >> 24) & 255u) - 128); }
            r = r * 1664525u + 1013904223u;
            blk.ds = __floats2half2_rn(0.001f + float(r >> 8) * 1e-9f, 0.0f);
        }
        void *dx = nullptr, *ya = nullptr, *yb = nullptr;
        cudaMalloc(&dx, hx.size() * sizeof(Q81Block));
        cudaMalloc(&ya, std::size_t(4) * n_out * 4);
        cudaMalloc(&yb, std::size_t(4) * n_out * 4);
        cudaMemcpy(dx, hx.data(), hx.size() * sizeof(Q81Block), cudaMemcpyHostToDevice);
        std::vector<float> a(std::size_t(4) * n_out), c(std::size_t(4) * n_out);
        bool same = true;
        for (int nc = 1; nc <= 4; ++nc) {
            g_q6_bypass = true;
            native_q6_k_mmvq(weights, dx, (float*) ya, n_in, n_out, nc, s);
            g_q6_bypass = false;
            native_q6_k_mmvq(weights, dx, (float*) yb, n_in, n_out, nc, s);
            cudaStreamSynchronize(s);
            cudaMemcpy(a.data(), ya, std::size_t(nc) * n_out * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(c.data(), yb, std::size_t(nc) * n_out * 4, cudaMemcpyDeviceToHost);
            same = same && std::memcmp(a.data(), c.data(), std::size_t(nc) * n_out * 4) == 0;
        }
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        for (int nc : {1, 2, 4}) {
            std::vector<float> tb, tc;
            for (int k = 0; k < 11; ++k) {
                for (int pass = 0; pass < 2; ++pass) {
                    const bool cand = (pass == 0) == (k % 2 == 1);
                    g_q6_bypass = !cand;
                    cudaEventRecord(e0, s);
                    for (int it = 0; it < 5; ++it) native_q6_k_mmvq(weights, dx, (float*) (cand ? yb : ya), n_in, n_out, nc, s);
                    cudaEventRecord(e1, s);
                    cudaEventSynchronize(e1);
                    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
                    (cand ? tc : tb).push_back(ms * 1000.0f / 5);
                }
            }
            g_q6_bypass = false;
            std::sort(tb.begin(), tb.end()); std::sort(tc.begin(), tc.end());
            std::fprintf(stderr, "strata: Q6_K packed self-test %s ncols %d: GGUF %.1f us (%.1f-%.1f), packed %.1f us "
                         "(%.1f-%.1f), 11 paired samples\n", what, nc, tb[5], tb[0], tb[10], tc[5], tc[0], tc[10]);
        }
        std::fprintf(stderr, "strata: Q6_K packed self-test %s: outputs %s (1-4 columns)\n", what,
                     same ? "BITWISE EQUAL" : "DIFFER");
        if (v[0] == '2') {   // grid sweep of the packed kernel
            for (int grid : {160, 320, 640, 1280, 2560, 5120, 20480}) {
                g_q6p_grid_override = grid;
                for (int nc : {1, 2, 4}) {
                    std::vector<float> tc;
                    for (int k = 0; k < 7; ++k) {
                        cudaEventRecord(e0, s);
                        for (int it = 0; it < 5; ++it) native_q6_k_mmvq(weights, dx, (float*) yb, n_in, n_out, nc, s);
                        cudaEventRecord(e1, s);
                        cudaEventSynchronize(e1);
                        float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
                        tc.push_back(ms * 1000.0f / 5);
                    }
                    std::sort(tc.begin(), tc.end());
                    std::fprintf(stderr, "strata: Q6_K packed grid sweep %s grid %d ncols %d: %.1f us (%.1f-%.1f)\n",
                                 what, grid, nc, tc[3], tc[0], tc[6]);
                }
            }
            g_q6p_grid_override = 0;
        }
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        cudaFree(dx); cudaFree(ya); cudaFree(yb);
        cudaStreamDestroy(s);
        if (!same) { native_q6_k_unpack(weights); return false; }
    }
    return true;
}

void native_q6_k_unpack(const void* weights) {
    auto& registry = q6_packed_registry();
    const auto found = registry.find(weights);
    if (found == registry.end()) return;
    cudaFree(found->second.base);
    registry.erase(found);
}

void native_mmvq_set_multi_exact(bool exact) { g_multi_exact = exact; }
bool native_mmvq_multi_exact() { return g_multi_exact; }
void native_mmvq_set_b6_rows(int mode) { g_b6_rows = mode; }
int native_mmvq_b6_rows() { return b6_rows_mode(); }

std::size_t native_q8_1_bytes(int n_in, int ncols) {
    validate_shape(n_in, ncols);
    return std::size_t(ncols) * std::size_t(n_in / Q8K) * sizeof(Q81Block);
}

void native_quantize_q8_1(const float* x, void* x_q8_1, int n_in, int ncols, void* stream) {
    validate_shape(n_in, ncols);
    validate_pointer(x);
    validate_pointer(x_q8_1);
    validate_stream(stream);
    // Columns are contiguous and n_in is a multiple of 32, so ncols columns quantize as one vector of
    // ncols * n_in elements: every 32-element block stays inside one column.
    const int n_total = n_in * ncols;
    const unsigned blocks = unsigned((std::size_t(n_total) + QUANT_THREADS - 1) / QUANT_THREADS);
    launch_pdl(native_quantize_q8_1_kernel, dim3(blocks), dim3(QUANT_THREADS), 0, static_cast<cudaStream_t>(stream), x,
               static_cast<Q81Block*>(x_q8_1), n_total);
    launch_check();
}

void native_swiglu_quantize_q8_1(const float* gate, const float* up, void* x_q8_1,
                                 int n_in, int ncols, void* stream) {
    validate_shape(n_in, ncols);
    validate_pointer(gate);
    validate_pointer(up);
    validate_pointer(x_q8_1);
    validate_stream(stream);
    const int n_total = n_in * ncols;
    const unsigned blocks = unsigned((std::size_t(n_total) + QUANT_THREADS - 1) / QUANT_THREADS);
    native_swiglu_quantize_q8_1_kernel<<<blocks, QUANT_THREADS, 0,
                                         static_cast<cudaStream_t>(stream)>>>(
        gate, up, static_cast<Q81Block*>(x_q8_1), n_total);
    launch_check();
}

void native_q5_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, QK);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    STRATA_WAVE_MMVQ(Q5KTraits)
    if (ncols > 1) {
        launch_multi<Q5KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q5KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / QK < VDR * WARPS * WARP / QI) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q5_k_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_q5_k_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_q5_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    // Validate all outputs before enqueueing the first operation.
    validate_shape(n_in, ncols, QK);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q5_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

namespace {
// Turing: Q2_0 one to four columns with one warp per RPW rows.  The exact layout gives each row four warps, a chunk of
// a 64-weight block per thread and a cross-warp reduction; at the engine's 2560 inputs (40 blocks a row) 48 of the 128
// threads have no chunk and the reduction is most of the time (QSA attn_q 2560x12288 at 3 columns: 80 us, 110 GB/s on
// an RTX 2080 Ti).  Here lane l runs the chunks of the exact kernel's threads l, 32+l, 64+l and 96+l in their loop
// order, adds the four sums as warp 0 adds the other warps' partials, then the same warp_sum: every output is bit for
// bit native_q2_0_mmvq_kernel's (one column) and launch_multi's exact layout (two to four).  80 -> 31 us at three
// columns, 37 -> 25 us at one.  STRATA_Q2_WARP=0 keeps the old kernels; =1 takes this one on any CUDA card.
constexpr int kQ20WarpRows = 4;
template <int NCOLS>
__global__ void __launch_bounds__(WARPS * WARP) q2_0_warp_rows_kernel(const Q20Block* __restrict__ w,
                                                                      const Q81Block* __restrict__ x,
                                                                      float* __restrict__ y, int n_in, int n_out) {
    using F = Q20Traits;
    constexpr int RPW = kQ20WarpRows;
    static_assert(F::T == 2 && F::BPI == WARPS * WARP / 2, "the lanes play the exact layout's threads");
    const int lane = int(threadIdx.x) & (WARP - 1);
    const int row0 = (int(blockIdx.x) * WARPS + int(threadIdx.x) / WARP) * RPW;
    if (row0 >= n_out) return;
    const int bpr = n_in / F::DIV, x_stride = n_in / Q8K;
    float acc[NCOLS][RPW];
#pragma unroll
    for (int v = 0; v < WARPS; ++v) {   // the exact kernel's thread WARP * v + lane
        const int tid = WARP * v + lane;
        const int kqs = F::kqs(tid);
        float tmp[NCOLS][RPW] = {};
        for (int kbx = tid / F::T; kbx < bpr; kbx += F::BPI) {
            const int kby = kbx * F::KBY;
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                if (row0 + i < n_out) {
                    const F::W wv = F::load(w + std::size_t(row0 + i) * bpr + kbx, kqs);
#pragma unroll
                    for (int j = 0; j < NCOLS; ++j) tmp[j][i] += F::apply(wv, x + std::size_t(j) * x_stride + kby, kqs);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NCOLS; ++j)
#pragma unroll
            for (int i = 0; i < RPW; ++i) acc[j][i] = v == 0 ? tmp[j][i] : acc[j][i] + tmp[j][i];
    }
#pragma unroll
    for (int j = 0; j < NCOLS; ++j)
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            const float sum = warp_sum(acc[j][i]);
            if (lane == i && row0 + i < n_out) y[std::size_t(j) * n_out + row0 + i] = sum;
        }
}
bool q2_0_warp_rows_on() {
    static int on[16] = {};   // by device ordinal: 0 unknown, 1 yes, -1 no
    static const int env = [] { const char* v = std::getenv("STRATA_Q2_WARP"); return v ? (v[0] == '0' ? 0 : 1) : -1; }();
    if (env >= 0) return env == 1;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) return false;
    if (on[dev] == 0) {
        int maj = 0, mnr = 0;
        const bool ok = cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                        cudaDeviceGetAttribute(&mnr, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess;
        if (!ok) cudaGetLastError();
        on[dev] = ok && strata::sm75_default(maj, mnr) ? 1 : -1;   // Turing only, with STRATA_SM75_PERF=1
    }
    return on[dev] > 0;
}
}  // namespace

void native_q2_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 64);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    STRATA_WAVE_MMVQ(Q20Traits)
#if !defined(__HIPCC__)
    // one column always (bitwise the single-column kernel); two to four only where the exact layout is the one in use
    if (ncols <= 4 && (ncols == 1 || g_multi_exact) && q2_0_warp_rows_on()) {
        const auto* w = static_cast<const Q20Block*>(weights);
        const auto* x = static_cast<const Q81Block*>(x_q8_1);
        const auto s = static_cast<cudaStream_t>(stream);
        const int warps = (n_out + kQ20WarpRows - 1) / kQ20WarpRows;
        const unsigned blocks = unsigned((warps + WARPS - 1) / WARPS);
        switch (ncols) {
            case 1: q2_0_warp_rows_kernel<1><<<blocks, WARPS * WARP, 0, s>>>(w, x, y, n_in, n_out); break;
            case 2: q2_0_warp_rows_kernel<2><<<blocks, WARPS * WARP, 0, s>>>(w, x, y, n_in, n_out); break;
            case 3: q2_0_warp_rows_kernel<3><<<blocks, WARPS * WARP, 0, s>>>(w, x, y, n_in, n_out); break;
            default: q2_0_warp_rows_kernel<4><<<blocks, WARPS * WARP, 0, s>>>(w, x, y, n_in, n_out); break;
        }
        launch_check();
        return;
    }
#endif
    if (ncols > 1) {
        launch_multi<Q20Traits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q20Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 64 < WARPS * WARP / 2) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q2_0_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_q2_0_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_q2_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 64);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q2_0_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q3_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    STRATA_WAVE_MMVQ(Q3KTraits)
    if (ncols > 1) {
        launch_multi<Q3KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q3KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 256 < WARPS * WARP / 16) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q3_k_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_q3_k_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_q3_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q3_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_iq4_xs_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    STRATA_WAVE_MMVQ(IQ4XSTraits)
    if (ncols > 1) {
        launch_multi<IQ4XSTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const IQ4XSBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 256 < 4 * WARPS * WARP / 32) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_iq4_xs_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_iq4_xs_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_iq4_xs_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_iq4_xs_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q4_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    STRATA_WAVE_MMVQ(Q4KTraits)
    if (ncols > 1) {
        launch_multi<Q4KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q4KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 256 < WARPS * WARP / 16) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q4_k_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_q4_k_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_q4_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q4_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q6_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    if (q6_packed_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream)) return;   // STRATA_Q6_PACKED=1 heads
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    STRATA_WAVE_MMVQ(Q6KTraits)
    if (ncols > 1) {
        launch_multi<Q6KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q6KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 256 < WARPS * WARP / 32) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q6_k_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_q6_k_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_q6_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q6_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q4_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    small_mmvq<Q40Block, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q4_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    small_f32<Q40Block, 4>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q5_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    small_mmvq<Q50Block, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q5_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    small_f32<Q50Block, 4>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

template<int NCOLS>
void q8_0_pair_launch(const Q80Block* w1, const Q80Block* w2, const Q81Block* x, float* y1, float* y2, int n_in, int n_out,
                      cudaStream_t s) {
    using F = SmallTraits<Q80Block, 8>;
    const dim3 threads(WARP, WARPS);
    if (s26_tsum_on())
        native_mmvq_multi_kernel<F, NCOLS, WARPS, 1, true, true><<<unsigned(2 * n_out), threads, 0, s>>>(w1, x, y1, n_in, n_out, w2, y2);
    else
        native_mmvq_multi_kernel<F, NCOLS, WARPS, 1, false, true><<<unsigned(2 * n_out), threads, 0, s>>>(w1, x, y1, n_in, n_out, w2, y2);
}

bool native_mmvq_pair(int ggml_type, const void* w1, const void* w2, const void* x_q8_1, float* y1, float* y2,
                      int n_in, int n_out, int ncols, void* stream) {
    using F = SmallTraits<Q80Block, 8>;
    // only where both calls would run native_mmvq_multi_kernel<F, ncols, WARPS, 1> (exact layout, not packed)
    if (ggml_type != 8 || !g_multi_exact || ncols < 2 || ncols > 8 || n_out <= 0 || n_in <= 0 || n_in % 32 != 0 ||
        n_in / F::DIV < F::BPI || !w1 || !w2 || !x_q8_1 || !y1 || !y2 || !stream)
        return false;
    const auto& reg = q8_packed_registry();
    if (reg.count(w1) || reg.count(w2)) return false;
    const auto* a = static_cast<const Q80Block*>(w1);
    const auto* b = static_cast<const Q80Block*>(w2);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    switch (ncols) {
        case 2: q8_0_pair_launch<2>(a, b, x, y1, y2, n_in, n_out, s); break;
        case 3: q8_0_pair_launch<3>(a, b, x, y1, y2, n_in, n_out, s); break;
        case 4: q8_0_pair_launch<4>(a, b, x, y1, y2, n_in, n_out, s); break;
        case 5: q8_0_pair_launch<5>(a, b, x, y1, y2, n_in, n_out, s); break;
        case 6: q8_0_pair_launch<6>(a, b, x, y1, y2, n_in, n_out, s); break;
        case 7: q8_0_pair_launch<7>(a, b, x, y1, y2, n_in, n_out, s); break;
        default: q8_0_pair_launch<8>(a, b, x, y1, y2, n_in, n_out, s); break;
    }
    launch_check();
    return true;
}

void native_q8_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    if (q8_packed_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream)) return;   // STRATA_Q8_PACKED=1 tensors
    small_mmvq<Q80Block, 8>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q8_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    small_f32<Q80Block, 8>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_iq4_nl_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    small_mmvq<IQ4NLBlock, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_iq4_nl_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    small_f32<IQ4NLBlock, 4>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

bool native_mmvq_supported(int ggml_type) noexcept {
    return ggml_type == 2 || ggml_type == 6 || ggml_type == 7 || ggml_type == 8 || ggml_type == 11 ||
           ggml_type == 12 || ggml_type == 13 || ggml_type == 14 || ggml_type == 20 ||
           ggml_type == 23 || ggml_type == 42 || ggml_type == 16 || ggml_type == 17 || ggml_type == 18 ||
           ggml_type == 21 || ggml_type == 22 || ggml_type == 29;
}

std::size_t native_mmvq_weight_bytes(int ggml_type, int n_in, int n_out) {
    int block_elems, block_bytes;
    switch (ggml_type) {
    case 2: block_elems = 32; block_bytes = 18; break;
    case 6: block_elems = 32; block_bytes = 22; break;
    case 7: block_elems = 32; block_bytes = 24; break;
    case 8: block_elems = 32; block_bytes = 34; break;
    case 20: block_elems = 32; block_bytes = 18; break;
    case 11: block_elems = 256; block_bytes = 110; break;
    case 12: block_elems = 256; block_bytes = 144; break;
    case 13: block_elems = 256; block_bytes = 176; break;
    case 14: block_elems = 256; block_bytes = 210; break;
    case 23: block_elems = 256; block_bytes = 136; break;
    case 42: block_elems = 64; block_bytes = 18; break;
    case 16: case 17: case 18: case 21: case 22: case 29:
        block_elems = 256; block_bytes = (int) iq_row_bytes(ggml_type, 256); break;
    default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
    validate_shape(n_in, 1, block_elems);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    const std::size_t row_bytes = std::size_t(n_in / block_elems) * block_bytes;
    if (row_bytes > std::numeric_limits<std::size_t>::max() / std::size_t(n_out)) {
        throw std::length_error("native MMVQ weight byte count overflows size_t");
    }
    return row_bytes * std::size_t(n_out);
}

void native_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y,
                 int n_in, int n_out, int ncols, void* stream) {
    // more columns than one launch takes: ceil(ncols / MAX_NCOLS) passes over the same weights.  Every column of a
    // multi-column pass is bitwise the single-column result (bench/micro/native_mmvq_multi.cpp), so the split changes no
    // arithmetic.  Column j starts at activation block j * (n_in / Q8K) and writes y + j * n_out.  (Never reached by a
    // window of up to 16 rows in a wide build, nor of up to 8 in a narrow one: a guard for a caller that asks for more.)
    if (ncols > MAX_NCOLS) {
        const std::size_t col_blocks = std::size_t(n_in / Q8K);
        for (int j = 0; j < ncols; j += MAX_NCOLS) {
            const int nj = (ncols - j < MAX_NCOLS) ? (ncols - j) : MAX_NCOLS;
            native_mmvq(ggml_type, weights, static_cast<const Q81Block*>(x_q8_1) + std::size_t(j) * col_blocks,
                        y + std::size_t(j) * std::size_t(n_out), n_in, n_out, nj, stream);
        }
        return;
    }
    switch (ggml_type) {
    case 2: native_q4_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 6: native_q5_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 7: iq_mmvq(ggml_type, weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 8: native_q8_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 20: native_iq4_nl_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 11: native_q3_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 12: native_q4_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 13: native_q5_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 14: native_q6_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 23: native_iq4_xs_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 42: native_q2_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 16: case 17: case 18: case 21: case 22: case 29:
        iq_mmvq(ggml_type, weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
}

#if !defined(__HIPCC__)
std::size_t native_q8_1_il_bytes(int n_in, int ncols) {
    validate_shape(n_in, ncols);
    return std::size_t(n_in / Q8K) * std::size_t(native_q8_1_il_cp(ncols)) * (16 * sizeof(int) + sizeof(float));
}

void native_quantize_q8_1_il(const float* x, void* x_q8_1, void* x_il, int n_in, int ncols, void* stream) {
    validate_shape(n_in, ncols);
    if (ncols < 2 || ncols > 4) throw std::invalid_argument("native_quantize_q8_1_il requires 2 <= ncols <= 4");
    validate_pointer(x);
    validate_pointer(x_q8_1);
    validate_pointer(x_il);
    validate_stream(stream);
    const int cp = native_q8_1_il_cp(ncols);
    const Q81IlParts parts = q8_1_il_parts(x_il, n_in, ncols);
    const int n_total = n_in * ncols;
    native_quantize_q8_1_il_kernel<<<unsigned((n_total + QUANT_THREADS - 1) / QUANT_THREADS), QUANT_THREADS, 0,
                                     static_cast<cudaStream_t>(stream)>>>(
        x, static_cast<Q81Block*>(x_q8_1), const_cast<int*>(parts.bm), const_cast<int*>(parts.pm),
        const_cast<float*>(parts.d), n_in, n_total, cp);
    launch_check();
}

void native_q8_1_interleave(const void* x_q8_1, void* x_il, int n_in, int ncols, void* stream) {
    validate_shape(n_in, ncols);
    if (ncols < 2 || ncols > 8) throw std::invalid_argument("native_q8_1_interleave requires 2 <= ncols <= 8");
    validate_pointer(x_q8_1);
    validate_pointer(x_il);
    validate_stream(stream);
    const int nb = n_in / Q8K, cp = native_q8_1_il_cp(ncols);
    const Q81IlParts parts = q8_1_il_parts(x_il, n_in, ncols);
    const int n_total = nb * 8 * ncols;
    native_q8_1_interleave_kernel<<<unsigned((n_total + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const Q81Block*>(x_q8_1), const_cast<int*>(parts.bm), const_cast<int*>(parts.pm),
        const_cast<float*>(parts.d), nb, n_total, cp);
    launch_check();
}

namespace {
// Rows a warp at 2-4 columns ([ncols - 2]) by the matrix's row count, in classes: below 2048 (the keys' 512), 4096 (the
// output projections' 2560), 8192 (the gate projection's 6144), 12288 (qkv's 10240), and more (the queries' 12288, the
// head's 248K); 0: native_mmvq's kernels. The fork tuned its table on an RTX 3090. This one is read off
// mmvq_il_parity --bench on an RTX 3060 (sm_86) and an RTX 5070 (sm_120) together: per cell the rows count (1/2/4) whose
// worse card still takes at least 3% off native_mmvq's time, else 0; classes without a measured shape take their
// neighbour's value. Every choice is bitwise the same output, so the table is only speed.
struct IlRows { int type; uint8_t r[3][5]; };
constexpr IlRows kIlRowsShared[] = {   // sm_86 + sm_120, and the fallback for every other sm_80+ card
    {23, {{0, 0, 0, 0, 0}, {0, 0, 2, 4, 4}, {0, 1, 1, 1, 1}}},   // IQ4_XS
    {12, {{0, 0, 1, 1, 1}, {0, 1, 1, 1, 1}, {0, 1, 1, 1, 1}}},   // Q4_K
    {13, {{0, 0, 1, 1, 1}, {0, 4, 1, 1, 1}, {0, 1, 1, 1, 2}}},   // Q5_K
    {14, {{0, 0, 1, 1, 0}, {0, 0, 1, 1, 2}, {0, 0, 1, 1, 1}}},   // Q6_K
};
// Per-architecture tables (P10).  A card whose compute capability (major * 10 + minor) is listed here uses its own table;
// every other sm_80+ card uses kIlRowsShared.  sm_89 (Ada) has an opt-in table (kIlRows_89, below; STRATA_MMVQ_IL_89=1).  To add one, run
// `mmvq_il_parity --bench --emit-table` on the card (docs/MMVQ_IL_TABLE.md), paste the printed block as a
// `constexpr IlRows kIlRows_<cc>[]` and add {<cc>, kIlRows_<cc>, sizeof(kIlRows_<cc>) / sizeof(IlRows)} below.  Each choice
// is bitwise the same output, so a table is only speed.  Without a rebuild: STRATA_MMVQ_IL_ROWS (see il_env_rows).
// Volta (sm_70): read off mmvq_il_parity --bench on a V100-SXM2-32GB (two runs, mean), the same rule - per cell the
// rows count whose every measured shape takes at least 3% off native_mmvq's time (e.g. the Q5_K head at 3 columns
// 1094 -> 821 us, Q6_K 12288 rows 68.9 -> 47.9 us), else 0; unmeasured classes take a neighbour's value (PR 1401)
constexpr IlRows kIlRows_70[] = {
    {23, {{0, 2, 2, 2, 2}, {0, 2, 2, 2, 2}, {0, 4, 2, 2, 2}}},   // IQ4_XS
    {12, {{0, 2, 2, 4, 4}, {0, 2, 2, 2, 2}, {0, 2, 2, 2, 2}}},   // Q4_K
    {13, {{0, 2, 2, 2, 2}, {0, 2, 2, 2, 2}, {0, 1, 2, 2, 2}}},   // Q5_K
    {14, {{0, 1, 1, 1, 1}, {0, 2, 2, 2, 2}, {0, 2, 2, 4, 2}}},   // Q6_K
};
// Pascal (sm_60), read off mmvq_il_parity --bench --emit-table on a Tesla P100-PCIE-16GB (three runs, identical
// tables), the same 3% rule.  Pascal has no dp4a (strata_dp4a: four vmad), so the kernels are compute-bound and more rows a
// warp (weights and activations reused across rows) win 15-35% on the large shapes.  Bitwise equal to native_mmvq.
constexpr IlRows kIlRows_60[] = {
    {23, {{0, 4, 4, 4, 4}, {2, 4, 2, 4, 4}, {1, 1, 1, 1, 1}}},   // IQ4_XS
    {12, {{0, 4, 4, 4, 4}, {0, 2, 2, 2, 2}, {0, 0, 0, 4, 4}}},   // Q4_K
    {13, {{0, 2, 2, 2, 2}, {0, 4, 2, 2, 4}, {0, 2, 2, 2, 2}}},   // Q5_K
    {14, {{0, 2, 2, 2, 4}, {0, 0, 0, 4, 4}, {0, 0, 0, 4, 4}}},   // Q6_K
};
// Pascal with dp4a (sm_61), read off mmvq_il_parity --bench --emit-table on a Tesla P40 (three runs: a cell is non-zero
// only where every run picked a rows count that takes at least 3% off native_mmvq's time; where the runs disagreed on
// the count, the one that won twice).  E.g. IQ4_XS 2560x10240 at 2 columns 112 -> 65 us, Q6_K 12288x2560 at 4
// columns 189 -> 127 us, the Q5_K head at 3 columns 2545 -> 1766 us.  Bitwise equal to native_mmvq.
constexpr IlRows kIlRows_61[] = {
    {23, {{2, 4, 4, 4, 4}, {4, 4, 4, 4, 4}, {2, 2, 2, 2, 2}}},   // IQ4_XS
    {12, {{2, 0, 4, 4, 4}, {2, 2, 2, 2, 2}, {2, 2, 2, 2, 2}}},   // Q4_K
    {13, {{2, 0, 4, 4, 4}, {2, 2, 2, 2, 2}, {2, 2, 2, 2, 2}}},   // Q5_K
    {14, {{0, 0, 2, 2, 2}, {0, 4, 4, 4, 4}, {2, 4, 2, 2, 2}}},   // Q6_K
    {11, {{2, 2, 2, 2, 2}, {2, 2, 2, 2, 2}, {1, 1, 1, 1, 1}}},   // Q3_K
};
// RTX 30-series (sm_86), 143-dense: read off mmvq_il_parity --bench --emit-table on an RTX A4000 with the engine's real
// shapes (the output projections have 6144 inputs, not 4096), per cell the rows count whose worst measured shape is under
// 0.97 of native_mmvq's time. The four older rows are the shared table's; Q3_K (the Q2_0 pack's main dense type) is new.
constexpr IlRows kIlRows_86[] = {
    {23, {{0, 0, 0, 0, 0}, {0, 0, 2, 4, 4}, {0, 1, 1, 1, 1}}},   // IQ4_XS
    {12, {{0, 0, 1, 1, 1}, {0, 1, 1, 1, 1}, {0, 1, 1, 1, 1}}},   // Q4_K
    {13, {{0, 0, 1, 1, 1}, {0, 4, 1, 1, 1}, {0, 1, 1, 1, 2}}},   // Q5_K
    {14, {{0, 0, 1, 1, 0}, {0, 0, 1, 1, 2}, {0, 0, 1, 1, 1}}},   // Q6_K
    {11, {{1, 2, 2, 2, 2}, {1, 1, 1, 2, 2}, {1, 1, 1, 1, 1}}},   // Q3_K (two runs; 2560x10240 at 4 columns 120 -> 70 us)
};
// RTX PRO 5000 Blackwell (sm_120), 147-verify: read off mmvq_il_parity --bench --emit-table with the engine's shapes, the
// same 3% rule; it takes the interleaved kernels in more cells than the shared table (e.g. the Q5_K head at 2 columns
// 507 -> 367 us, IQ4_XS 2560x10240 at 3 columns 20.7 -> 14.9 us).  Bitwise equal to native_mmvq.
constexpr IlRows kIlRows_120[] = {
    {23, {{0, 1, 2, 2, 2}, {0, 4, 4, 4, 4}, {0, 2, 2, 4, 2}}},   // IQ4_XS
    {12, {{0, 0, 2, 1, 1}, {0, 2, 4, 2, 2}, {0, 2, 2, 2, 2}}},   // Q4_K
    {13, {{0, 1, 2, 1, 1}, {0, 1, 4, 2, 4}, {0, 1, 2, 1, 2}}},   // Q5_K
    {14, {{0, 0, 2, 1, 1}, {0, 0, 4, 2, 4}, {0, 2, 2, 2, 2}}},   // Q6_K
    {11, {{0, 1, 2, 2, 2}, {0, 2, 4, 2, 2}, {0, 2, 2, 2, 2}}},   // Q3_K
};
// Ada (sm_89): the cells where two cards agree.  mmvq_il_parity --bench --emit-table was run on an RTX 4070 (46 SMs,
// #3453) and on an RTX 4080 (48 SMs, same issue); a cell is kept only where both runs give the same rows count, else 0
// (the plain kernel), so a card that disagrees never gets a split it did not measure.  The 5-8 column table of that
// issue is not taken: the 4080 reproduced the 2-4 column table byte for byte in four runs, and the 5-8 column table
// changed in 10-14 of 20 entries between runs.  Bitwise equal to native_mmvq, so the table is only speed.
// mmvq_il_parity's check_tables rebuilds the table from the two source runs and compares.
constexpr IlRows kIlRows_89[] = {
    {23, {{0, 0, 0, 0, 0}, {0, 0, 1, 1, 1}, {0, 0, 1, 1, 1}}},   // IQ4_XS
    {12, {{0, 0, 1, 1, 1}, {0, 0, 1, 1, 1}, {0, 0, 1, 1, 1}}},   // Q4_K
    {13, {{0, 0, 1, 1, 0}, {0, 0, 1, 1, 0}, {0, 0, 1, 0, 0}}},   // Q5_K
    {14, {{0, 0, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 0, 0, 1, 1}}},   // Q6_K
    {11, {{0, 0, 1, 0, 1}, {0, 1, 1, 1, 1}, {0, 1, 1, 1, 1}}},   // Q3_K
};
// sm_80: read off mmvq_il_parity --bench --emit-table on a CMP 170HX (GA100, 70 SMs, HBM2e 1.5 TB/s), the same 3% rule;
// bitwise equal to native_mmvq (the tool checks every case).
constexpr IlRows kIlRows_80[] = {
    {23, {{0, 2, 2, 2, 2}, {0, 2, 2, 2, 2}, {0, 2, 1, 1, 1}}},   // IQ4_XS
    {12, {{0, 0, 4, 4, 4}, {0, 0, 2, 2, 2}, {0, 0, 2, 2, 2}}},   // Q4_K
    {13, {{0, 0, 2, 2, 4}, {0, 0, 2, 2, 2}, {0, 0, 2, 2, 2}}},   // Q5_K
    {14, {{0, 0, 2, 2, 2}, {0, 2, 2, 2, 2}, {0, 2, 4, 2, 2}}},   // Q6_K
    {11, {{1, 2, 2, 2, 2}, {1, 1, 2, 2, 2}, {1, 2, 2, 2, 2}}},   // Q3_K
};
// sm_80 P1/P2: the hoisted-load / exact-integer kernel variant and its rows a warp per cell, pf << 4 | rows
// (0x32 = variant 3 at 2 rows a warp; 0 = today's kernel with kIlRows_80), read off mmvq_il_parity --bench --emit-table
// --pf 0,3,4 on the CMP 170HX with the same 3% rule against native_mmvq; STRATA_MMVQ_IL_PF=1 takes it.  Every variant is
// bitwise native_mmvq (the tool checks every case, T 1-8, rows 1/2/4, each variant).
struct IlPf { int type; uint8_t v[3][5]; };
struct IlPf58 { int type; uint8_t v[4][5]; };
// Only Q6_K: the same variants are bitwise and measured for Q4_K / Q5_K / IQ4_XS too, but there they tie today's kernel
// within the run-to-run noise (their emitted cells flip between 0 and a variant), so those types keep kIlRows_80.
constexpr IlPf kIlPf_80[] = {
    {14, {{0x31, 0x42, 0x32, 0x42, 0x42}, {0x31, 0x41, 0x32, 0x32, 0x41}, {0x31, 0x41, 0x32, 0x32, 0x32}}},   // Q6_K
};
constexpr IlPf58 kIlPf58_80[] = {   // (rows 4 runs as 2 at 5+ columns, so the emitted 0x34 cells are written 0x32)
    {14, {{0x31, 0x42, 0x32, 0x32, 0x32}, {0x31, 0x31, 0x32, 0x32, 0x32}, {0x31, 0x31, 0x02, 0x32, 0x02}, {0x31, 0x31, 0x31, 0x31, 0x31}}},   // Q6_K
};
// Turing (sm_75): read off mmvq_il_parity --bench --emit-table on an RTX 2080 Ti 22 GB (SM 1650 MHz), the same 3% rule.
// The plain multi-column kernels are bound by dp4a throughput on Turing from 2 columns on (IQ4_XS 2560x10240 at 4
// columns 81.1 -> 37.4 us, the Q5_K head 1671 -> 1045 us); bitwise equal to native_mmvq.  No one-column table here:
// the engine's dedicated single-column kernels (native_q3_k_mmvq) sum in another order, and the gain was 0.5%.
constexpr IlRows kIlRows_75[] = {
    {23, {{2, 2, 2, 2, 2}, {2, 2, 4, 2, 4}, {2, 2, 2, 2, 2}}},   // IQ4_XS
    {12, {{0, 4, 2, 4, 4}, {1, 2, 2, 2, 2}, {1, 2, 2, 2, 2}}},   // Q4_K
    {13, {{0, 4, 2, 2, 4}, {1, 4, 2, 2, 2}, {1, 1, 2, 2, 2}}},   // Q5_K
    {14, {{0, 0, 1, 2, 1}, {0, 2, 2, 4, 4}, {0, 4, 4, 4, 4}}},   // Q6_K
    {11, {{1, 2, 4, 4, 4}, {1, 2, 4, 4, 4}, {1, 2, 2, 2, 2}}},   // Q3_K
};
// 147-verify: 5-8 columns (a verify window of 5+ rows) by n_out class, per architecture; 0 = native_mmvq's kernels.  Every
// choice is bitwise native_mmvq's output (mmvq_il_parity, T 1-8), so a table is only speed.  Measured on an RTX PRO 5000
// Blackwell (sm_120) with mmvq_il_parity --bench --emit-table; other cards have no table yet and keep native_mmvq's
// kernels at 5+ columns.  STRATA_MMVQ_IL_WIDE=0 turns the 5-8 column path off.
struct IlRows58 { int type; uint8_t r[4][5]; };
constexpr IlRows58 kIlRows58_120[] = {   // 4 = 2 rows a warp here (launch_il_rows)
    {23, {{0, 4, 4, 4, 4}, {0, 4, 2, 4, 2}, {0, 4, 2, 4, 2}, {0, 4, 2, 2, 2}}},   // IQ4_XS
    {12, {{0, 2, 4, 4, 4}, {0, 4, 4, 4, 4}, {0, 2, 2, 2, 2}, {0, 4, 2, 4, 4}}},   // Q4_K
    {13, {{0, 2, 2, 2, 4}, {0, 4, 2, 4, 2}, {0, 2, 4, 2, 2}, {0, 2, 4, 2, 2}}},   // Q5_K
    {14, {{0, 2, 4, 4, 4}, {0, 4, 2, 4, 2}, {0, 2, 4, 2, 2}, {0, 4, 4, 4, 2}}},   // Q6_K
    {11, {{0, 2, 4, 2, 4}, {0, 2, 2, 4, 4}, {0, 2, 2, 2, 4}, {0, 4, 2, 2, 2}}},   // Q3_K
};
constexpr IlRows58 kIlRows58_80[] = {   // CMP 170HX, as kIlRows_80
    {23, {{0, 4, 4, 2, 2}, {0, 2, 4, 4, 4}, {0, 2, 4, 1, 1}, {1, 4, 1, 1, 1}}},   // IQ4_XS
    {12, {{0, 0, 4, 2, 2}, {0, 2, 4, 2, 2}, {0, 2, 4, 4, 4}, {0, 2, 2, 4, 4}}},   // Q4_K
    {13, {{0, 0, 4, 2, 2}, {0, 4, 2, 2, 2}, {0, 4, 4, 2, 4}, {0, 2, 4, 4, 2}}},   // Q5_K
    {14, {{0, 0, 4, 2, 4}, {0, 0, 2, 2, 2}, {0, 2, 2, 2, 2}, {0, 0, 4, 2, 4}}},   // Q6_K
    {11, {{0, 0, 4, 2, 4}, {0, 0, 4, 1, 4}, {0, 0, 1, 1, 1}, {1, 1, 1, 1, 1}}},   // Q3_K
};
struct Il58Arch { int cc; const IlRows58* t; size_t n; };
constexpr Il58Arch kIl58Arch[] = {
    {80, kIlRows58_80, sizeof(kIlRows58_80) / sizeof(IlRows58)},
    {120, kIlRows58_120, sizeof(kIlRows58_120) / sizeof(IlRows58)},
    {1201, kIlRows58_120, sizeof(kIlRows58_120) / sizeof(IlRows58)},   // a 12.0 card with 80+ SMs (il_cc)
};
// One column: rows a warp by n_out class, 0 = native_mmvq's single-column kernels (STRATA_MMVQ_IL_T1=0 turns this off).
struct Il1Rows { int type; uint8_t r[5]; };
constexpr Il1Rows kIl1Rows_86[] = {
    {11, {0, 1, 1, 1, 1}},   // Q3_K: 2560x10240 60 -> 43 us, 6144x2560 30 -> 26 us on the A4000
};
// Ada (sm_89): the RTX 4070 and the RTX 4080 runs gave the same one-column row (only Q3_K takes the interleaved kernel)
constexpr Il1Rows kIl1Rows_89[] = {
    {11, {0, 0, 1, 1, 1}},   // Q3_K
};
// Tesla P40 (sm_61), the same three runs: IQ4_XS 2560x10240 102 -> 58 us, the Q5_K head 1754 -> 1507 us
constexpr Il1Rows kIl1Rows_61[] = {
    {23, {2, 2, 4, 4, 4}},   // IQ4_XS
    {13, {0, 0, 0, 2, 2}},   // Q5_K
    {11, {0, 0, 2, 2, 2}},   // Q3_K
};
constexpr Il1Rows kIl1Rows_80[] = {   // CMP 170HX, as kIlRows_80
    {23, {0, 0, 1, 4, 1}},   // IQ4_XS
    {12, {0, 0, 2, 2, 2}},   // Q4_K
    {13, {0, 0, 0, 4, 1}},   // Q5_K
    {11, {0, 0, 2, 2, 2}},   // Q3_K
};
struct Il1Arch { int cc; const Il1Rows* t; size_t n; };
constexpr Il1Arch kIl1Arch[] = {
    {80, kIl1Rows_80, sizeof(kIl1Rows_80) / sizeof(Il1Rows)},
    {61, kIl1Rows_61, sizeof(kIl1Rows_61) / sizeof(Il1Rows)},
    {86, kIl1Rows_86, sizeof(kIl1Rows_86) / sizeof(Il1Rows)},
    {89, kIl1Rows_89, sizeof(kIl1Rows_89) / sizeof(Il1Rows)},   // Ada: the RTX 4070 and 4080 agree
};
bool il_env_on(const char* name) {
    const char* v = std::getenv(name);
    return v == nullptr || v[0] != '0';
}
bool il_t1_on() { static const bool on = il_env_on("STRATA_MMVQ_IL_T1"); return on; }
bool il_q3k_on() { static const bool on = il_env_on("STRATA_MMVQ_IL_Q3K"); return on; }
// #3453: sm_89's own tables (kIlRows_89 / kIl1Rows_89) are ON by default on compute capability 8.9 only (owner decision, 0.2.7; no sm_89 card
// has run mmvq_il_parity or an end-to-end A/B for us, the reporter tests and reports back).  STRATA_MMVQ_IL_89=0 gives an Ada card what it
// answered before (the shared table, no one-column row) and so does STRATA_DEFAULTS=026; STRATA_MMVQ_IL_89=1 wins over that master switch
// (sm120_series.hpp mmvq_il_89_resolve).  Cached once; the lookup runs on the host once a launch.
bool il_89_opt_in() {
    static const bool on = strata::mmvq_il_89_resolve(std::getenv("STRATA_MMVQ_IL_89"), false, 89, strata::qwen_defaults_026());
    return on;
}
struct IlArch { int cc; const IlRows* t; size_t n; };
constexpr IlArch kIlArch[] = {
    {60, kIlRows_60, sizeof(kIlRows_60) / sizeof(IlRows)},   // Pascal GP100: measured on a Tesla P100
    {61, kIlRows_61, sizeof(kIlRows_61) / sizeof(IlRows)},   // Pascal with dp4a: measured on a Tesla P40
    {75, kIlRows_75, sizeof(kIlRows_75) / sizeof(IlRows)},   // Turing: measured on an RTX 2080 Ti
    {70, kIlRows_70, sizeof(kIlRows_70) / sizeof(IlRows)},   // Volta: PR 1401, measured on a V100-SXM2
    {80, kIlRows_80, sizeof(kIlRows_80) / sizeof(IlRows)},   // sm_80: measured on a CMP 170HX
    {86, kIlRows_86, sizeof(kIlRows_86) / sizeof(IlRows)},   // 143-dense: RTX A4000 with the real shapes (Q3_K added)
    {89, kIlRows_89, sizeof(kIlRows_89) / sizeof(IlRows)},   // Ada: the cells where an RTX 4070 and an RTX 4080 agree
    // 147-verify: RTX PRO 5000 Blackwell (110 SMs).  Only 12.0 cards with 80+ SMs (il_cc's 1201) take it: the shared table
    // was measured on an RTX 5070 (48 SMs) and stays that card's
    {1201, kIlRows_120, sizeof(kIlRows_120) / sizeof(IlRows)},
};
// STRATA_MMVQ_IL_ROWS="type:ncols:r0,r1,r2,r3,r4;..." overrides single rows of the table for every card (type = ggml
// type 23/12/13/14, ncols 2-4, r* = rows a warp 0/1/2/4 for the five n_out classes).  Parsed once.
struct IlEnvRow { int type, ncols; uint8_t r[5]; };
const std::vector<IlEnvRow>& il_env_rows() {
    static const std::vector<IlEnvRow> v = [] {
        std::vector<IlEnvRow> out;
        const char* e = std::getenv("STRATA_MMVQ_IL_ROWS");
        if (!e) return out;
        std::string str(e);
        size_t pos = 0;
        while (pos < str.size()) {
            size_t end = str.find(';', pos);
            if (end == std::string::npos) end = str.size();
            int t = 0, n = 0, r[5] = {0, 0, 0, 0, 0}, used = 0;
            if (std::sscanf(str.substr(pos, end - pos).c_str(), "%d:%d:%d,%d,%d,%d,%d%n", &t, &n, &r[0], &r[1], &r[2], &r[3],
                            &r[4], &used) >= 7 && n >= 1 && n <= 4) {
                IlEnvRow x{t, n, {}};
                bool ok = true;
                for (int i = 0; i < 5; ++i) { ok = ok && (r[i] == 0 || r[i] == 1 || r[i] == 2 || r[i] == 4); x.r[i] = (uint8_t) r[i]; }
                if (ok) out.push_back(x);
                else std::fprintf(stderr, "strata: STRATA_MMVQ_IL_ROWS: rows must be 0/1/2/4 - entry ignored\n");
            } else {
                std::fprintf(stderr, "strata: STRATA_MMVQ_IL_ROWS: cannot read '%s' (type:ncols:r0,r1,r2,r3,r4) - ignored\n",
                             str.substr(pos, end - pos).c_str());
            }
            pos = end + 1;
        }
        return out;
    }();
    return v;
}
}  // namespace
// The rows a warp for a card of compute capability `cc` (major * 10 + minor); 0: native_mmvq's kernels.  Pure, so a test
// can check every architecture's table without that card.
int native_mmvq_il_rows_for(int cc, int type, int ncols, int n_out) {
    if (ncols < 1 || ncols > 8) return 0;
    const int cls = n_out < 2048 ? 0 : n_out < 4096 ? 1 : n_out < 8192 ? 2 : n_out < 12288 ? 3 : 4;
    if (ncols > 4) {   // 147-verify: 5..8 columns, only where a card measured its table
        static const bool wide = il_env_on("STRATA_MMVQ_IL_WIDE");
        if (!wide) return 0;
        if (type == 11 && !il_q3k_on()) return 0;
        for (const Il58Arch& a : kIl58Arch)
            if (a.cc == cc)
                for (size_t i = 0; i < a.n; ++i)
                    if (a.t[i].type == type) return a.t[i].r[ncols - 5][cls];
        return 0;
    }
    for (const IlEnvRow& e : il_env_rows())
        if (e.type == type && e.ncols == ncols) return e.r[cls];
    if (ncols == 1) {   // 143-dense: one column (the plain q8_1 blocks, no interleaved copy); STRATA_MMVQ_IL_T1=0 turns it off
        if (!il_t1_on()) return 0;
        for (const Il1Arch& a : kIl1Arch)
            if (a.cc == cc && (cc != 89 || il_89_opt_in()))
                for (size_t i = 0; i < a.n; ++i)
                    if (a.t[i].type == type) return a.t[i].r[cls];
        return 0;
    }
    if (type == 11 && !il_q3k_on()) return 0;   // STRATA_MMVQ_IL_Q3K=0: Q3_K keeps native_mmvq's kernels
    const IlRows* t = kIlRowsShared;
    size_t n = sizeof(kIlRowsShared) / sizeof(IlRows);
    static const bool own120 = il_env_on("STRATA_MMVQ_IL_120");   // 147-verify: =0 gives a big 12.0 card the shared table again
    for (const IlArch& a : kIlArch)
        if (a.cc == cc && a.t != nullptr && (cc != 1201 || own120) && (cc != 89 || il_89_opt_in())) { t = a.t; n = a.n; }
    for (size_t i = 0; i < n; ++i)
        if (t[i].type == type) return t[i].r[ncols - 2][cls];
    return 0;
}
namespace {
int il_cc() {
    static int cc[16] = {};   // by device ordinal, 0 unknown
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) return 0;
    if (cc[dev] == 0) {
        int maj = 0, mnr = 0;
        if (cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&mnr, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) { cudaGetLastError(); return 0; }
        cc[dev] = maj * 10 + mnr;
        // 147-verify: a 12.0 card with 80+ SMs (RTX PRO 5000 / 6000, RTX 5080 / 5090) answers 1201, its own table's key;
        // a smaller one (RTX 5070: 48 SMs) keeps 120 and the shared table it was measured into
        int sms = 0;
        if (cc[dev] == 120 && cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess && sms >= 80)
            cc[dev] = 1201;
        // STRATA_MMVQ_IL_SM80=0: an sm_80 card takes the shared table as before kIlRows_80 (A/B)
        if (cc[dev] == 80 && !strata::sm80_default_on(std::getenv("STRATA_MMVQ_IL_SM80"), strata::qwen_defaults_026())) cc[dev] = 81;   // =0 or STRATA_DEFAULTS=026
        cudaGetLastError();
    }
    return cc[dev];
}
int il_rows(int type, int ncols, int n_out) { return native_mmvq_il_rows_for(il_cc(), type, ncols, n_out); }
int il_pf_table(int type, int ncols, int n_out) {
    if (il_cc() != 80 || ncols < 2 || ncols > 8) return 0;   // only the CMP 170HX measured one
    const int cls = n_out < 2048 ? 0 : n_out < 4096 ? 1 : n_out < 8192 ? 2 : n_out < 12288 ? 3 : 4;
    if (ncols <= 4) {
        for (const IlPf& t : kIlPf_80)
            if (t.type == type) return t.v[ncols - 2][cls];
    } else {
        for (const IlPf58& t : kIlPf58_80)
            if (t.type == type) return t.v[ncols - 5][cls];
    }
    return 0;
}
int g_tune_rows = 0;   // native_mmvq_il_tune (tests, benchmarks): rows a warp for every shape (0: the table)
}  // namespace
void native_mmvq_il_tune(int rows) { g_tune_rows = rows; }
void native_mmvq_il_pf(int pf) { g_il_pf = pf; }

namespace {
// sm_80 and newer (measured on sm_86 and sm_120), Volta (sm_70) and Pascal (sm_60, sm_61), each with its own table
// above; Turing keeps native_mmvq's kernels unless STRATA_SM75_PERF=1 (kIlRows_75)
// Volta (PR 1401): opt-in with STRATA_SM70_TABLE=1 until the author confirms on a V100 with the final code
bool sm70_opt_in() { const char* v = std::getenv("STRATA_SM70_TABLE"); return v != nullptr && std::atoi(v) != 0; }
// Pascal (sm_60, GP100): the same interleaved kernels (bitwise equal to native_mmvq's, mmvq_il_parity), its own table
// kIlRows_60.  STRATA_SM60_TABLE=0 keeps native_mmvq's kernels on a P100; any other value (or unset) uses the table.
// STRATA_SM75_IL=0 / 1 forces the table off / on; unset it follows STRATA_SM75_PERF=1 (Turing only)
bool sm75_il_on(int major, int minor) { return strata::sm75_switch("STRATA_SM75_IL", major, minor); }
bool sm60_table_on() { const char* v = std::getenv("STRATA_SM60_TABLE"); return v == nullptr || std::atoi(v) != 0; }
// Pascal (sm_61: Tesla P40 / P4, GTX 10 series; native dp4a): the same interleaved kernels (bitwise equal to
// native_mmvq's, mmvq_il_parity), its own table kIlRows_61.  STRATA_SM61_TABLE=0 keeps native_mmvq's kernels.
bool sm61_table_on() { const char* v = std::getenv("STRATA_SM61_TABLE"); return v == nullptr || std::atoi(v) != 0; }
bool il_arch_ok() {
    static int ok[16] = {};   // 0 unknown, 1 yes, -1 no, by device ordinal
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) return false;
    if (ok[dev] == 0) {
        int major = 0;
        int minor = 0;
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        ok[dev] = (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                   (major >= 8 || (major == 7 && minor == 0 && sm70_opt_in()) || (major == 7 && minor == 5 && sm75_il_on(major, minor)) ||
                    (major == 6 && minor == 0 && sm60_table_on()) || (major == 6 && minor == 1 && sm61_table_on()))) ? 1 : -1;
    }
    return ok[dev] > 0;
}
}  // namespace

bool native_mmvq_il_supported(int ggml_type, int ncols, int n_out) {
    if (ncols < 1 || ncols > 8 || !g_multi_exact || !il_arch_ok()) return false;
    return (g_tune_rows ? g_tune_rows : il_rows(ggml_type, ncols, n_out)) != 0 &&
           (ggml_type == 23 || ggml_type == 12 || ggml_type == 13 || ggml_type == 14 || ggml_type == 11);
}

void native_mmvq_il(int ggml_type, const void* weights, const void* x_q8_1, const void* x_il, float* y, int n_in,
                    int n_out, int ncols, void* stream) {
    if (!native_mmvq_il_supported(ggml_type, ncols, n_out)) {
        native_mmvq(ggml_type, weights, x_q8_1, y, n_in, n_out, ncols, stream);
        return;
    }
    const int r = g_tune_rows ? g_tune_rows : il_rows(ggml_type, ncols, n_out);
    validate_shape(n_in, ncols, QK);
    validate_pointer(weights);
    if (ncols == 1) x_il = x_q8_1;   // one column: the plain blocks, no interleaved copy
    validate_pointer(x_il);
    validate_pointer(y);
    validate_stream(stream);
    const auto s = static_cast<cudaStream_t>(stream);
    switch (ggml_type) {
    case 23: launch_il_rows<IlIQ4XS>(r, weights, x_il, y, n_in, n_out, ncols, s); break;
    case 12: launch_il_rows<IlQ4K>(r, weights, x_il, y, n_in, n_out, ncols, s); break;
    case 13: launch_il_rows<IlQ5K>(r, weights, x_il, y, n_in, n_out, ncols, s); break;
    case 14: launch_il_rows<IlQ6K>(r, weights, x_il, y, n_in, n_out, ncols, s); break;
    case 11: launch_il_rows<IlQ3K>(r, weights, x_il, y, n_in, n_out, ncols, s); break;
    }
    launch_check();
}

#else  // HIP: no interleaved path
std::size_t native_q8_1_il_bytes(int n_in, int ncols) { return native_q8_1_bytes(n_in, ncols) * 2; }
void native_quantize_q8_1_il(const float*, void*, void*, int, int, void*) {
    throw std::invalid_argument("native_quantize_q8_1_il is CUDA only");
}
void native_q8_1_interleave(const void*, void*, int, int, void*) {
    throw std::invalid_argument("native_q8_1_interleave is CUDA only");
}
bool native_mmvq_il_supported(int, int, int) { return false; }
int native_mmvq_il_rows_for(int, int, int, int) { return 0; }
void native_mmvq_il(int ggml_type, const void* weights, const void* x_q8_1, const void*, float* y, int n_in, int n_out,
                    int ncols, void* stream) {
    native_mmvq(ggml_type, weights, x_q8_1, y, n_in, n_out, ncols, stream);
}
void native_mmvq_il_tune(int) {}
void native_mmvq_il_pf(int) {}
#endif  // !__HIPCC__

} // namespace strata::kernels
