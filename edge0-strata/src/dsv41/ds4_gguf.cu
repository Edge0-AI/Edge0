// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_gguf.cu - a GGUF's BF16 matrices and MXFP4 experts on the device (see ds4_gguf.cuh).
#include "ds4_launch.h"   // strata_ds4_launch: <<<>>> on CUDA and HIP, a dispatch on Metal
#include "ds4_gguf.cuh"
#if defined(STRATA_METAL)
#include <ds4_metal_host.h>
#endif

#include "ds4_iq.cuh"
#if defined(STRATA_METAL)
#include "ds4_gguf_cpu.hpp"   // gg_row_to_bf16 (the DS4_DBG_IQ check)
#endif
#include "ds4_numerics.cuh"

#include <cuda_fp16.h>
#if !defined(STRATA_USE_HIP)
#include <mma.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace strata::ds4::gguf {
namespace {

std::atomic<int> g_errors{0};

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "ds4 gguf %s: %s\n", what, cudaGetErrorString(e));
        g_errors.fetch_add(1, std::memory_order_relaxed);
    }
}
void fail(const char* what) {
    std::fprintf(stderr, "ds4 gguf %s\n", what);
    g_errors.fetch_add(1, std::memory_order_relaxed);
}

constexpr int kWarps = 8, kThreads = kWarps * 32;
constexpr int kMaxEnt = 16;   // a group's entries in one pass (a window's tokens at most)

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// ---------------------------------------------------------------- BF16 matrices
struct BfJob {
    const void* w;
    size_t ld;          // values a row (BF16), bytes a row (MXFP4)
    const float* x;
    size_t xcs;
    float* y;
    size_t ycs;
    const float* bias;
    float alpha;
    int k, n, ncol;
    const int8_t* xq;   // (Q8_0: the activations rounded to int8, q8_quant's)
    const float* xd;
};
struct BfJobs {
    BfJob j[x3::kMaxMvJobs];
};

// grid (max rows / 8, jobs): a warp a row, each of its activations' dot in F32 - a lane 8 BF16 weights (16 bytes) and
// 8 activations a step (VEC), or one of each (rows or activations not 16-byte aligned)
template <int NC, bool VEC>
__global__ void __launch_bounds__(kThreads) bf16_mv_kernel(const BfJobs J) {
    const BfJob& b = J.j[blockIdx.y];
    const int row = blockIdx.x * kWarps + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= b.n) return;
    const uint16_t* w = (const uint16_t*) b.w + (size_t) row * b.ld;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
    if (VEC) {
        for (int i = lane * 8; i < b.k; i += 256) {
            const uint4 q = *(const uint4*) (w + i);
            const float wf[8] = {__uint_as_float(q.x << 16), __uint_as_float(q.x & 0xffff0000u),
                                 __uint_as_float(q.y << 16), __uint_as_float(q.y & 0xffff0000u),
                                 __uint_as_float(q.z << 16), __uint_as_float(q.z & 0xffff0000u),
                                 __uint_as_float(q.w << 16), __uint_as_float(q.w & 0xffff0000u)};
#pragma unroll
            for (int c = 0; c < NC; ++c)
                if (c < b.ncol) {
                    const float4* xp = (const float4*) (b.x + c * b.xcs + i);
                    const float4 x0 = __ldg(xp), x1 = __ldg(xp + 1);
                    float a = acc[c];
                    a = fmaf(wf[0], x0.x, a);
                    a = fmaf(wf[1], x0.y, a);
                    a = fmaf(wf[2], x0.z, a);
                    a = fmaf(wf[3], x0.w, a);
                    a = fmaf(wf[4], x1.x, a);
                    a = fmaf(wf[5], x1.y, a);
                    a = fmaf(wf[6], x1.z, a);
                    a = fmaf(wf[7], x1.w, a);
                    acc[c] = a;
                }
        }
    } else {
        for (int i = lane; i < b.k; i += 32) {
            const float wf = __uint_as_float((uint32_t) w[i] << 16);
#pragma unroll
            for (int c = 0; c < NC; ++c)
                if (c < b.ncol) acc[c] = fmaf(wf, b.x[c * b.xcs + i], acc[c]);
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c)
        if (c < b.ncol) {
            const float v = warp_sum(acc[c]);
            if (lane == 0) b.y[c * b.ycs + row] = b.alpha * v + (b.bias != nullptr ? b.bias[row] : 0.0f);
        }
}


__global__ void q8_rows_kernel(const uint8_t* __restrict__ w, size_t ld, int k, int n, uint16_t* __restrict__ dst,
                               bool bf16) {
    const size_t total = (size_t) n * (size_t) k;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t) gridDim.x * blockDim.x) {
        const size_t r = i / (size_t) k, c = i - r * (size_t) k;
        const uint8_t* p = w + r * ld + (c / 32) * kQ8Bytes;
        const float v = __half2float(__ushort_as_half(*(const uint16_t*) p)) * (float) (int8_t) p[2 + c % 32];
        dst[i] = bf16 ? strata::ds4::bf16_rn(v) : __half_as_ushort(__float2half_rn(v));
    }
}

__global__ void bf16_rows_kernel(const uint16_t* __restrict__ w, size_t ld, int k, int n, uint16_t* __restrict__ dst,
                                 bool bf16) {
    const size_t total = (size_t) n * (size_t) k;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t) gridDim.x * blockDim.x) {
        const size_t r = i / (size_t) k, c = i - r * (size_t) k;
        const uint16_t b = w[r * ld + c];
        dst[i] = bf16 ? b : __half_as_ushort(__float2half_rn(__uint_as_float((uint32_t) b << 16)));
    }
}

// ---------------------------------------------------------------- MXFP4 experts
__device__ __forceinline__ float e8m0_half(uint32_t e) {   // 2^(e - 128), ggml's GGML_E8M0_TO_FP32_HALF
    return __uint_as_float(e < 2 ? 0x00200000u << e : (e - 1) << 23);
}
__device__ __forceinline__ void lut_init(float* lut) {
    if (threadIdx.x < 16) {
        const float v[8] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f};
        lut[threadIdx.x] = threadIdx.x < 8 ? v[threadIdx.x] : -v[threadIdx.x - 8];
    }
}
// block p's 32 values, exactly (a code times a power of two)
__device__ __forceinline__ void mx_block(const uint8_t* __restrict__ p, const float* lut, float* w) {
    const float d = e8m0_half(p[0]);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const uint32_t q = p[1 + j];
        w[j] = lut[q & 15u] * d;
        w[j + 16] = lut[q >> 4] * d;
    }
}
__device__ __forceinline__ float swiglu(float gv, float uv, float limit) {
    if (limit > 0.0f) {
        gv = fminf(gv, limit);
        uv = fminf(fmaxf(uv, -limit), limit);
    }
    return __fmul_rn(__fdiv_rn(gv, __fadd_rn(1.0f, expf(-gv))), uv);
}
__device__ __forceinline__ float combine(const float* plan_w, const float* p, int k, const float* cpu_part, float sh) {
    float m = 0.0f;
    for (int i = 0; i < k; ++i) m = __fmaf_rn(plan_w[i], p[i], m);
    if (cpu_part != nullptr) m = __fadd_rn(m, *cpu_part);
    return __fadd_rn(m, sh);
}

// a Q8_0 block's 32 values, exactly (an FP16 scale times an int8; the block is only 2-byte aligned)
__device__ __forceinline__ void q8_block(const uint8_t* __restrict__ p, float* w) {
    const float d = __half2float(__ushort_as_half(*(const uint16_t*) p));
    const uint16_t* q = (const uint16_t*) (p + 2);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const uint16_t v = q[j];
        w[2 * j] = d * (float) (int8_t) (v & 0xff);
        w[2 * j + 1] = d * (float) (int8_t) (v >> 8);
    }
}

// ---- int8 activations (llama.cpp's MMVQ: its q8_1 blocks' rounding)
// rows of k F32 values (row r at x + r ldx; k % 32 == 0) -> q[r][k] int8 and d[r][k / 32] = max|x| / 127 per 32
__global__ void __launch_bounds__(kThreads) q8_quant_kernel(const float* __restrict__ x, size_t ldx, int k,
                                                            int8_t* __restrict__ q, float* __restrict__ d) {
    const int r = blockIdx.y, b = blockIdx.x * kWarps + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (b * 32 >= k) return;
    const float v = x[(size_t) r * ldx + (size_t) b * 32 + lane];
    float am = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    const float sc = am / 127.0f;
    q[(size_t) r * k + (size_t) b * 32 + lane] = (int8_t) (am > 0.0f ? __float2int_rn(v / sc) : 0);
    if (lane == 0) d[(size_t) r * (k / 32) + b] = sc;
}
void q8_quant(const float* x, size_t ldx, int rows, int k, int8_t* q, float* d, cudaStream_t s) {
    const dim3 grid((unsigned) ((k / 32 + kWarps - 1) / kWarps), (unsigned) rows);
#if defined(STRATA_METAL)
    if (::strata::dsv41::metal_host::q8_quant_rows(x, ldx, rows, k, (int8_t*) q, d)) return;
#endif
    strata_ds4_launch(q8_quant_kernel, grid, kThreads, 0, s, x, ldx, k, q, d);
}

// Several activations' rounding in one launch (a GEMV call's jobs: wo_a's eight groups, the shared expert's two, ...): item z is
// q8_quant's (x, ldx, rows, k, q, d), the same per-element arithmetic, so the bytes are the single launches'.  Identical items
// (the same x, stride, rows and k: two matrices reading one input) are rounded once and share the result.
constexpr int kQuantItems = x3::kMaxMvJobs;
struct QuantItems {
    const float* x[kQuantItems];
    size_t ldx[kQuantItems];
    int k[kQuantItems], rows[kQuantItems];
    int8_t* q[kQuantItems];
    float* d[kQuantItems];
    int n;
};
__global__ void __launch_bounds__(kThreads) q8_quant_multi_kernel(QuantItems it) {
    const int z = blockIdx.z;
    if (z >= it.n) return;
    const int r = blockIdx.y, b = blockIdx.x * kWarps + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int k = it.k[z];
    if (r >= it.rows[z] || b * 32 >= k) return;
    const float v = it.x[z][(size_t) r * it.ldx[z] + (size_t) b * 32 + lane];
    float am = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    const float sc = am / 127.0f;
    it.q[z][(size_t) r * k + (size_t) b * 32 + lane] = (int8_t) (am > 0.0f ? __float2int_rn(v / sc) : 0);
    if (lane == 0) it.d[z][(size_t) r * (k / 32) + b] = sc;
}
// adds an item unless an identical one is queued; returns the item whose result to read (its q and d)
int quant_add(QuantItems& it, const float* x, size_t ldx, int rows, int k, int8_t* q, float* d) {
    static const bool nodedup = std::getenv("DS4_QUANT_NODEDUP") != nullptr;
    for (int i = 0; i < it.n && !nodedup; ++i)
        if (it.x[i] == x && it.ldx[i] == ldx && it.rows[i] == rows && it.k[i] == k) return i;
    const int i = it.n++;
    it.x[i] = x;
    it.ldx[i] = ldx;
    it.rows[i] = rows;
    it.k[i] = k;
    it.q[i] = q;
    it.d[i] = d;
    return i;
}
void quant_flush(const QuantItems& it, cudaStream_t s) {
    if (it.n <= 0) return;
    static const bool single = std::getenv("DS4_QUANT_SINGLE") != nullptr;
#if defined(STRATA_METAL)
    if (!single && std::getenv("DS4_NO_Q8_BATCH") == nullptr) {   // the call's roundings in one dispatch (a dispatch a job was 45% of the decode token's)
        struct Spec { const float* x; uint64_t ldx; uint32_t rows, k; int8_t* q; float* d; } sp[kQuantItems];
        for (int i = 0; i < it.n; ++i) sp[i] = Spec{it.x[i], (uint64_t) it.ldx[i], (uint32_t) it.rows[i], (uint32_t) it.k[i], it.q[i], it.d[i]};
        if (::strata::dsv41::metal_host::q8_quant_multi(sp, it.n)) return;
    }
    for (int i = 0; i < it.n; ++i) q8_quant(it.x[i], it.ldx[i], it.rows[i], it.k[i], it.q[i], it.d[i], s);
    return;
#endif
    if (single) {
        for (int i = 0; i < it.n; ++i) q8_quant(it.x[i], it.ldx[i], it.rows[i], it.k[i], it.q[i], it.d[i], s);
        return;
    }
    int kmax = 0, rmax = 0;
    for (int i = 0; i < it.n; ++i) {
        kmax = std::max(kmax, it.k[i]);
        rmax = std::max(rmax, it.rows[i]);
    }
    const dim3 grid((unsigned) ((kmax / 32 + kWarps - 1) / kWarps), (unsigned) rmax, (unsigned) it.n);
    strata_ds4_launch(q8_quant_multi_kernel, grid, kThreads, 0, s, it);
}

// 8 E2M1 codes (a word of 4 code bytes) -> their values as int8: .x the low nibbles' (values 0..3 of the word's bytes),
// .y the high nibbles' - llama.cpp's get_int_from_table_16 (MIT), the table kvalues_mxfp4
__device__ __forceinline__ int2 e2m1_x8(uint32_t q4) {
    const uint32_t t0 = 0x03020100u, t1 = 0x0C080604u, t2 = 0xFDFEFF00u, t3 = 0xF4F8FAFCu;
    const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
    uint32_t tmp[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t idx = (q4 >> (16 * i)) & 0x7777u;
        const uint32_t lo = __byte_perm(t0, t1, idx), hi = __byte_perm(t2, t3, idx);
        tmp[i] = __byte_perm(lo, hi, sel >> (16 * i));
    }
    return make_int2((int) __byte_perm(tmp[0], tmp[1], 0x6420), (int) __byte_perm(tmp[0], tmp[1], 0x7531));
}
// an MXFP4 block's codes (4 words) against 32 int8 activations (xa: values 0..15, xb: 16..31)
__device__ __forceinline__ int mx_dot(const uint4& q, const int4& xa, const int4& xb) {
    int sum = 0;
    int2 v = e2m1_x8(q.x);
    sum = __dp4a(v.x, xa.x, sum);
    sum = __dp4a(v.y, xb.x, sum);
    v = e2m1_x8(q.y);
    sum = __dp4a(v.x, xa.y, sum);
    sum = __dp4a(v.y, xb.y, sum);
    v = e2m1_x8(q.z);
    sum = __dp4a(v.x, xa.z, sum);
    sum = __dp4a(v.y, xb.z, sum);
    v = e2m1_x8(q.w);
    sum = __dp4a(v.x, xa.w, sum);
    sum = __dp4a(v.y, xb.w, sum);
    return sum;
}
// the warp copies the blocks [b0, b0 + nbk) of a row (BB bytes each; the row 4-byte aligned, nbk BB % 4 == 0) into its
// stage in 4-byte words; every lane calls it
template <int BB>
__device__ __forceinline__ void stage_chunk(const uint8_t* row, int b0, int nbk, uint32_t* st) {
    const uint32_t* src = (const uint32_t*) (row + (size_t) b0 * BB);
    __syncwarp();   // (the lanes are done with the chunk before)
    for (int i = threadIdx.x & 31; i < nbk * BB / 4; i += 32) st[i] = src[i];
    __syncwarp();
}
// a row's next chunk of 32 blocks (BB bytes each) in the lanes' registers, then into the warp's stage
template <int BB>
struct Fetch {
    static constexpr int W = (32 * BB / 4 + 31) / 32;   // words a lane
    uint32_t r[W];
    int n = 0;
    __device__ __forceinline__ void load(const uint8_t* row, int b0, int nbk) {
        const uint32_t* src = (const uint32_t*) (row + (size_t) b0 * BB);
        n = nbk * BB / 4;
#pragma unroll
        for (int j = 0; j < W; ++j) {
            const int i = (int) (threadIdx.x & 31) + 32 * j;
            r[j] = i < n ? src[i] : 0u;
        }
    }
    __device__ __forceinline__ void store(uint32_t* st) const {
        __syncwarp();   // (the lanes are done with the chunk before)
#pragma unroll
        for (int j = 0; j < W; ++j) {
            const int i = (int) (threadIdx.x & 31) + 32 * j;
            if (i < n) st[i] = r[j];
        }
        __syncwarp();
    }
};

// MXFP4 block i of a stage (17-byte blocks from a 4-byte aligned start): its scale byte, its 16 code bytes as 4 words
__device__ __forceinline__ uint32_t mx_words_s(const uint32_t* st, int i, uint4& q) {
    const int o = 17 * i, w = o >> 2, sh = (o & 3) * 8;
    const uint32_t a0 = st[w], a1 = st[w + 1], a2 = st[w + 2], a3 = st[w + 3], a4 = st[w + 4];
    const uint32_t s0 = __funnelshift_r(a0, a1, sh), s1 = __funnelshift_r(a1, a2, sh), s2 = __funnelshift_r(a2, a3, sh),
                   s3 = __funnelshift_r(a3, a4, sh), s4 = a4 >> sh;
    q.x = __funnelshift_r(s0, s1, 8);
    q.y = __funnelshift_r(s1, s2, 8);
    q.z = __funnelshift_r(s2, s3, 8);
    q.w = __funnelshift_r(s3, s4, 8);
    return s0 & 0xffu;
}
// ... the same from global memory, byte by byte (rows not 4-byte aligned)
__device__ __forceinline__ uint32_t mx_words_g(const uint8_t* __restrict__ p, uint4& q) {
    uint32_t w[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
        w[j] = (uint32_t) p[1 + 4 * j] | (uint32_t) p[2 + 4 * j] << 8 | (uint32_t) p[3 + 4 * j] << 16 |
               (uint32_t) p[4 + 4 * j] << 24;
    q = make_uint4(w[0], w[1], w[2], w[3]);
    return p[0];
}
// Q8_0 block i of a stage (34-byte blocks): its scale, its 32 int8 as 8 words
__device__ __forceinline__ float q8_words_s(const uint32_t* st, int i, int* v) {
    const int o = 34 * i, w = o >> 2, sh = (o & 3) * 8;
    uint32_t a[9];
#pragma unroll
    for (int j = 0; j < 9; ++j) a[j] = st[w + j];
    uint32_t sw[9];
#pragma unroll
    for (int j = 0; j < 8; ++j) sw[j] = __funnelshift_r(a[j], a[j + 1], sh);
    sw[8] = a[8] >> sh;
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = (int) __funnelshift_r(sw[j], sw[j + 1], 16);
    return __half2float(__ushort_as_half((uint16_t) (sw[0] & 0xffffu)));
}
__device__ __forceinline__ float q8_words_g(const uint8_t* __restrict__ p, int* v) {
#pragma unroll
    for (int j = 0; j < 8; ++j)
        v[j] = (int) ((uint32_t) p[2 + 4 * j] | (uint32_t) p[3 + 4 * j] << 8 | (uint32_t) p[4 + 4 * j] << 16 |
                      (uint32_t) p[5 + 4 * j] << 24);
    return __half2float(__ushort_as_half((uint16_t) (p[0] | p[1] << 8)));
}

// grid (max rows / 8, jobs): a Q8_0 matrix's rows against the jobs' int8 activations (q8_quant's: activation c of job
// j at xq + c k, its scales at xd + c k / 32) - a warp a row, a lane a block a step, the blocks staged (ST) or read
template <int NC, bool ST>
__global__ void __launch_bounds__(kThreads) q8_mv_kernel(const BfJobs J) {
    __shared__ __align__(16) uint32_t stage[kWarps][32 * kQ8Bytes / 4];
    const BfJob& b = J.j[blockIdx.y];
    const int row = blockIdx.x * kWarps + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= b.n) return;
    const uint8_t* w = (const uint8_t*) b.w + (size_t) row * b.ld;
    const int nb = b.k / 32;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
    Fetch<kQ8Bytes> F;
    if (ST) F.load(w, 0, min(32, nb));
    for (int b0 = 0; b0 < nb; b0 += 32) {
        const int nbk = min(32, nb - b0), kb = b0 + lane;
        if (ST) {
            F.store(stage[threadIdx.x >> 5]);
            if (b0 + 32 < nb) F.load(w, b0 + 32, min(32, nb - b0 - 32));
        }
        if (lane >= nbk) continue;
        int v[8];
        const float dw = ST ? q8_words_s(stage[threadIdx.x >> 5], lane, v) : q8_words_g(w + (size_t) kb * kQ8Bytes, v);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (c < b.ncol) {
                const int4* xp = (const int4*) (b.xq + (size_t) c * b.k + (size_t) kb * 32);
                const int4 xa = xp[0], xb = xp[1];
                int sum = __dp4a(v[0], xa.x, 0);
                sum = __dp4a(v[1], xa.y, sum);
                sum = __dp4a(v[2], xa.z, sum);
                sum = __dp4a(v[3], xa.w, sum);
                sum = __dp4a(v[4], xb.x, sum);
                sum = __dp4a(v[5], xb.y, sum);
                sum = __dp4a(v[6], xb.z, sum);
                sum = __dp4a(v[7], xb.w, sum);
                acc[c] = fmaf(dw * b.xd[(size_t) c * nb + kb], (float) sum, acc[c]);
            }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c)
        if (c < b.ncol) {
            const float v = warp_sum(acc[c]);
            if (lane == 0) b.y[c * b.ycs + row] = b.alpha * v + (b.bias != nullptr ? b.bias[row] : 0.0f);
        }
}

// grid (max rows / 8, jobs): bf16_mv_kernel for MXFP4 (Q8 false) or Q8_0 (Q8 true) matrices - a lane a 32-value block
// a step, dequantised once for the job's activations
template <int NC, bool VEC, bool Q8>
__global__ void __launch_bounds__(kThreads) mx_mv_kernel(const BfJobs J) {
    __shared__ float lut[16];
    lut_init(lut);
    __syncthreads();
    const BfJob& b = J.j[blockIdx.y];
    const int row = blockIdx.x * kWarps + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= b.n) return;
    const uint8_t* w = (const uint8_t*) b.w + (size_t) row * b.ld;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
    for (int kb = lane; kb < b.k / kMxBlock; kb += 32) {
        float wv[32];
        if (Q8) q8_block(w + (size_t) kb * kQ8Bytes, wv);
        else mx_block(w + (size_t) kb * kMxBytes, lut, wv);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (c < b.ncol) {
                const float* x = b.x + c * b.xcs + kb * kMxBlock;
                float a = acc[c];
                if (VEC) {
#pragma unroll
                    for (int q = 0; q < 8; ++q) {
                        const float4 xv = __ldg((const float4*) x + q);
                        a = fmaf(wv[4 * q], xv.x, a);
                        a = fmaf(wv[4 * q + 1], xv.y, a);
                        a = fmaf(wv[4 * q + 2], xv.z, a);
                        a = fmaf(wv[4 * q + 3], xv.w, a);
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < 32; ++j) a = fmaf(wv[j], x[j], a);
                }
                acc[c] = a;
            }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c)
        if (c < b.ncol) {
            const float v = warp_sum(acc[c]);
            if (lane == 0) b.y[c * b.ycs + row] = b.alpha * v + (b.bias != nullptr ? b.bias[row] : 0.0f);
        }
}

// a token's experts (grp_ptr null: group i = expert i of the plan, plan_ptr[i] its blob or 0 off the device, its one
// entry i, every entry's input x) or a window's groups (group g: one expert, blob grp_ptr[g], entries e = t * 8 + i at
// grp_ent[grp_start[g] ..), entry e's input x + (e >> 3) x_stride)
struct MxArgs {
    const unsigned long long* plan_ptr;
    const unsigned long long* grp_ptr;
    const int *grp_start, *grp_ent, *n_groups;
    const float* x;
    size_t x_stride;
    float* h;      // [entries][n_ff]
    float* part;   // [entries][n_embd]
    size_t tr[3];
    int n_embd, n_ff;
    float limit;
    const int8_t* xq;   // x rounded to int8 (a token's: row 0; a window's: row t), q8_quant's rows of n_embd
    const float* xd;
    const int8_t* hq;   // h rounded to int8, row e = the entry, rows of n_ff
    const float* hd;
};

__device__ __forceinline__ bool mx_group(const MxArgs& a, int grp, const uint8_t*& blob, int& m0, int& m1) {
    if (a.grp_ptr == nullptr) {
        const unsigned long long p = a.plan_ptr[grp];
        if (p == 0ull) return false;
        blob = (const uint8_t*) p;
        m0 = grp;
        m1 = grp + 1;
        return true;
    }
    if (grp >= *a.n_groups) return false;
    blob = (const uint8_t*) a.grp_ptr[grp];
    m0 = a.grp_start[grp];
    m1 = a.grp_start[grp + 1];
    return true;
}

// grid (n_ff / 8, groups): a warp a row of the gate and of the up, for NE of the group's entries a pass (each weight
// block decoded once for them) against the entries' int8 x -> h = the clamped swiglu
template <int NE, bool ST>
__global__ void __launch_bounds__(kThreads) mx_gate_up_kernel(const MxArgs a) {
    __shared__ __align__(16) uint32_t stage[kWarps][2][32 * kMxBytes / 4];
    __shared__ int es[NE], xr[NE];
    const uint8_t* blob;
    int m0, m1;
    if (!mx_group(a, blockIdx.y, blob, m0, m1)) return;   // (uniform)
    const int row = blockIdx.x * kWarps + (threadIdx.x >> 5), lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const size_t rb = mx_row_bytes(a.n_embd);
    const uint8_t* g = blob + a.tr[0] + (size_t) row * rb;
    const uint8_t* u = blob + a.tr[1] + (size_t) row * rb;
    const int nb = a.n_embd / kMxBlock;
    for (int mm = m0; mm < m1; mm += NE) {
        const int ne = min(NE, m1 - mm);
        __syncthreads();
        if ((int) threadIdx.x < ne) {
            const int e = a.grp_ptr != nullptr ? a.grp_ent[mm + threadIdx.x] : mm + threadIdx.x;
            es[threadIdx.x] = e;
            xr[threadIdx.x] = a.grp_ptr != nullptr ? e >> 3 : 0;
        }
        __syncthreads();
        if (row >= a.n_ff) continue;
        float ag[NE], au[NE];
#pragma unroll
        for (int i = 0; i < NE; ++i) ag[i] = au[i] = 0.0f;
        Fetch<kMxBytes> FG, FU;
        if (ST) {
            FG.load(g, 0, min(32, nb));
            FU.load(u, 0, min(32, nb));
        }
        for (int b0 = 0; b0 < nb; b0 += 32) {
            const int nbk = min(32, nb - b0), b = b0 + lane;
            if (ST) {
                FG.store(stage[wp][0]);
                FU.store(stage[wp][1]);
                if (b0 + 32 < nb) {
                    FG.load(g, b0 + 32, min(32, nb - b0 - 32));
                    FU.load(u, b0 + 32, min(32, nb - b0 - 32));
                }
            }
            if (lane >= nbk) continue;
            uint4 qg, qu;
            const uint32_t eg = ST ? mx_words_s(stage[wp][0], lane, qg) : mx_words_g(g + (size_t) b * kMxBytes, qg);
            const uint32_t eu = ST ? mx_words_s(stage[wp][1], lane, qu) : mx_words_g(u + (size_t) b * kMxBytes, qu);
            const float dg = e8m0_half(eg), du = e8m0_half(eu);
#pragma unroll
            for (int i = 0; i < NE; ++i)
                if (i < ne) {
                    const int4* xp = (const int4*) (a.xq + (size_t) xr[i] * a.n_embd + (size_t) b * kMxBlock);
                    const int4 xa = xp[0], xb = xp[1];
                    const float dx = a.xd[(size_t) xr[i] * nb + b];
                    ag[i] = fmaf(dg * dx, (float) mx_dot(qg, xa, xb), ag[i]);
                    au[i] = fmaf(du * dx, (float) mx_dot(qu, xa, xb), au[i]);
                }
        }
#pragma unroll
        for (int i = 0; i < NE; ++i)
            if (i < ne) {
                const float gv = warp_sum(ag[i]), uv = warp_sum(au[i]);
                if (lane == 0) a.h[(size_t) es[i] * a.n_ff + row] = swiglu(gv, uv, a.limit);
            }
    }
}

// grid (n_embd / 8, groups): a warp a row of the down for NE entries a pass, against their int8 h -> part
template <int NE, bool ST>
__global__ void __launch_bounds__(kThreads) mx_down_kernel(const MxArgs a) {
    __shared__ __align__(16) uint32_t stage[kWarps][32 * kMxBytes / 4];
    __shared__ int es[NE];
    const uint8_t* blob;
    int m0, m1;
    if (!mx_group(a, blockIdx.y, blob, m0, m1)) return;
    const int row = blockIdx.x * kWarps + (threadIdx.x >> 5), lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const uint8_t* d = blob + a.tr[2] + (size_t) row * mx_row_bytes(a.n_ff);
    const int nb = a.n_ff / kMxBlock;
    for (int mm = m0; mm < m1; mm += NE) {
        const int ne = min(NE, m1 - mm);
        __syncthreads();
        if ((int) threadIdx.x < ne) es[threadIdx.x] = a.grp_ptr != nullptr ? a.grp_ent[mm + threadIdx.x] : mm + threadIdx.x;
        __syncthreads();
        if (row >= a.n_embd) continue;
        float ad[NE];
#pragma unroll
        for (int i = 0; i < NE; ++i) ad[i] = 0.0f;
        Fetch<kMxBytes> FD;
        if (ST) FD.load(d, 0, min(32, nb));
        for (int b0 = 0; b0 < nb; b0 += 32) {
            const int nbk = min(32, nb - b0), b = b0 + lane;
            if (ST) {
                FD.store(stage[wp]);
                if (b0 + 32 < nb) FD.load(d, b0 + 32, min(32, nb - b0 - 32));
            }
            if (lane >= nbk) continue;
            uint4 q;
            const float dd = e8m0_half(ST ? mx_words_s(stage[wp], lane, q) : mx_words_g(d + (size_t) b * kMxBytes, q));
#pragma unroll
            for (int i = 0; i < NE; ++i)
                if (i < ne) {
                    const int4* hp = (const int4*) (a.hq + (size_t) es[i] * a.n_ff + (size_t) b * kMxBlock);
                    const int4 ha = hp[0], hb = hp[1];
                    ad[i] = fmaf(dd * a.hd[(size_t) es[i] * nb + b], (float) mx_dot(q, ha, hb), ad[i]);
                }
        }
#pragma unroll
        for (int i = 0; i < NE; ++i)
            if (i < ne) {
                const float v = warp_sum(ad[i]);
                if (lane == 0) a.part[(size_t) es[i] * a.n_embd + row] = v;
            }
    }
}

struct CombArgs {
    const unsigned long long* plan_ptr;   // [T][8]
    const float* plan_w;                  // [T][8]
    const float* part;                    // [T * 8][n_embd]
    const float* cpu_part;
    size_t cpu_stride;
    const int* cpu_flag;
    const float* sh_out;                  // [T][n_embd]
    float* out;
    size_t out_stride;
    int k, n_embd;
};
// grid (n_embd / 128, T): each token's outputs - the experts in plan order (an expert off the device adds 0), then
// the CPU lane's part and the shared expert's (x3's combine)
__global__ void __launch_bounds__(128) mx_combine_kernel(const CombArgs a) {
    const int t = blockIdx.y, r = blockIdx.x * 128 + threadIdx.x;
    if (r >= a.n_embd) return;
    float p[8];
    for (int i = 0; i < a.k; ++i) {
        const int e = t * 8 + i;
        p[i] = a.plan_ptr[e] != 0ull ? a.part[(size_t) e * a.n_embd + r] : 0.0f;
    }
    a.out[(size_t) t * a.out_stride + r] =
        combine(a.plan_w + t * 8, p, a.k,
                a.cpu_flag != nullptr && a.cpu_flag[t] ? a.cpu_part + (size_t) t * a.cpu_stride + r : nullptr,
                a.sh_out != nullptr ? a.sh_out[(size_t) t * a.n_embd + r] : 0.0f);
}

__global__ void mx_wait_kernel(const unsigned* p, unsigned v) {
    if (threadIdx.x != 0) return;
    while (*(volatile const unsigned*) p != v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(200);
#endif
    }
    __threadfence_system();
}

// ---------------------------------------------------------------- the prompt path's expert sets
// a set's expert i (blob wbase + i stride) owns the sorted rows [bounds[i], bounds[i + 1]); local row = sorted row -
// row0.  A block: BM of its rows x BN outputs, the weights dequantised a 32-value block at a time into shared memory,
// F32 FMAs (each thread TM x TN outputs)
constexpr int BM = 64, BN = 64, TM = 4, TN = 4;
static_assert((BM / TM) * (BN / TN) == kThreads, "one output tile a thread");

struct SetK {
    const uint8_t* wbase;
    size_t stride;
    size_t tr[3];
    const int* bounds;
    int row0;
    int n_embd, n_ff;
    float limit;
    const float* xf;      // gate / up: F32 input rows x[idx[local]] (ldx)
    const int* idx;
    const __half* xh;     // down: F16 rows (ldx)
    int ldx;
    __half* h;            // gate / up: F16 out
    float* y;             // down: F32 out
    int ldo;
    int skip_le;          // the experts of at most this many rows left out (the light kernels took them; ds4_iq.cuh)
};

// one 32-value block of BN weight rows (rows c0 .., row bytes rb, block kb) -> ws[32][BN + 1]
__device__ __forceinline__ void mx_tile(const uint8_t* w, size_t rb, int c0, int kb, const float* lut,
                                        float (*ws)[BN + 1]) {
    const int r = threadIdx.x >> 2, q = threadIdx.x & 3;   // row r of the tile, values [8 q, 8 q + 8)
    const uint8_t* p = w + (size_t) (c0 + r) * rb + (size_t) kb * kMxBytes;
    const float d = e8m0_half(p[0]);
    const int b0 = (q & 1) * 8, hi = q >> 1;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t c = p[1 + b0 + j];
        ws[q * 8 + j][r] = lut[hi ? (c >> 4) : (c & 15u)] * d;
    }
}

template <bool GATE_UP>
__global__ void __launch_bounds__(kThreads) mx_set_kernel(const SetK a) {
    __shared__ float lut[16];
    __shared__ float wg[32][BN + 1];
    __shared__ float wu[GATE_UP ? 32 : 1][BN + 1];
    __shared__ float xs[BM][33];
    __shared__ int src[BM];
    lut_init(lut);
    const int e = blockIdx.z;
    const int lo = a.bounds[e] - a.row0, hi = a.bounds[e + 1] - a.row0;
    const int r0 = lo + blockIdx.x * BM;
    if (r0 >= hi || hi - lo <= a.skip_le) return;   // (uniform: before any barrier but lut_init's writes, which nobody
                                                    // reads then; skip_le: the light kernels' experts)
    const int nr = min(BM, hi - r0);
    const int c0 = blockIdx.y * BN;
    const uint8_t* blob = a.wbase + (size_t) e * a.stride;
    const int K = GATE_UP ? a.n_embd : a.n_ff;
    const size_t rb = mx_row_bytes(K);
    const int tid = (int) threadIdx.x;
    if (tid < BM) src[tid] = tid < nr ? (GATE_UP ? a.idx[r0 + tid] : r0 + tid) : -1;
    __syncthreads();
    const int ty = threadIdx.x / (BN / TN), tx = threadIdx.x % (BN / TN);
    float ag[TM][TN], au[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) ag[i][j] = au[i][j] = 0.0f;
    for (int kb = 0; kb < K / kMxBlock; ++kb) {
        mx_tile(blob + a.tr[GATE_UP ? 0 : 2], rb, c0, kb, lut, wg);
        if (GATE_UP) mx_tile(blob + a.tr[1], rb, c0, kb, lut, wu);
        for (int i = threadIdx.x; i < BM * kMxBlock; i += kThreads) {   // the rows' 32 inputs
            const int r = i / kMxBlock, c = i % kMxBlock, sr = src[r];
            float v = 0.0f;
            if (sr >= 0) {
                const size_t at = (size_t) sr * a.ldx + (size_t) kb * kMxBlock + c;
                v = GATE_UP ? a.xf[at] : __half2float(a.xh[at]);
            }
            xs[r][c] = v;
        }
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < kMxBlock; ++kk) {
            float xv[TM], gw[TN], uw[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) xv[i] = xs[ty * TM + i][kk];
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                gw[j] = wg[kk][tx * TN + j];
                if (GATE_UP) uw[j] = wu[kk][tx * TN + j];
            }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    ag[i][j] = fmaf(xv[i], gw[j], ag[i][j]);
                    if (GATE_UP) au[i][j] = fmaf(xv[i], uw[j], au[i][j]);
                }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int r = ty * TM + i;
        if (r >= nr) continue;
        const size_t lr = (size_t) (r0 + r);
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int c = c0 + tx * TN + j;
            if (GATE_UP) a.h[lr * a.ldo + c] = __float2half_rn(swiglu(ag[i][j], au[i][j], a.limit));
            else a.y[lr * a.ldo + c] = ag[i][j];
        }
    }
}

// the same on the tensor cores: a block BM rows x BN outputs, 8 warps each 16 rows x 32 outputs (2 fragments, gate and
// up: 4), a 32-value MXFP4 block of K a step (2 WMMA k-steps), the tiles in shared memory as F16
#if !defined(STRATA_USE_HIP)
namespace wm = nvcuda::wmma;
constexpr int kLdT = 40;   // halves a tile row (32 + 8: WMMA wants a multiple of 8; the pad spreads the banks)
template <bool GATE_UP>
__global__ void __launch_bounds__(kThreads) mx_set_tc_kernel(const SetK a) {
    struct Tiles {
        __half x[BM][kLdT], wg[BN][kLdT], wu[GATE_UP ? BN : 1][kLdT];
    };
    struct Out {
        float g[BM][BN + 4], u[GATE_UP ? BM : 1][BN + 4];
    };
    __shared__ __align__(32) unsigned char smem[sizeof(Tiles) > sizeof(Out) ? sizeof(Tiles) : sizeof(Out)];
    __shared__ float lut[16];
    __shared__ int src[BM];
    Tiles& T = *reinterpret_cast<Tiles*>(smem);
    Out& O = *reinterpret_cast<Out*>(smem);
    lut_init(lut);
    const int e = blockIdx.z;
    const int lo = a.bounds[e] - a.row0, hi = a.bounds[e + 1] - a.row0;
    const int r0 = lo + blockIdx.x * BM;
    if (r0 >= hi || hi - lo <= a.skip_le) return;   // (uniform; skip_le: the light kernels' experts)
    const int nr = min(BM, hi - r0);
    const int c0 = blockIdx.y * BN;
    const uint8_t* blob = a.wbase + (size_t) e * a.stride;
    const int K = GATE_UP ? a.n_embd : a.n_ff;
    const size_t rb = mx_row_bytes(K);
    const int tid = (int) threadIdx.x, warp = tid >> 5;
    if (tid < BM) src[tid] = tid < nr ? (GATE_UP ? a.idx[r0 + tid] : r0 + tid) : -1;
    __syncthreads();
    const int wm_ = warp >> 1, wn = (warp & 1) * 2;   // the warp's 16 rows, its 2 x 16 outputs
    wm::fragment<wm::accumulator, 16, 16, 16, float> fg[2], fu[2];
    for (int j = 0; j < 2; ++j) {
        wm::fill_fragment(fg[j], 0.0f);
        wm::fill_fragment(fu[j], 0.0f);
    }
    const int tr = tid >> 2, tq = tid & 3;   // the thread's tile row, its 8 values [8 tq, 8 tq + 8)
    for (int kb = 0; kb < K / kMxBlock; ++kb) {
        {   // the weights: row tr of the outputs, values 8 tq ..
            for (int m = 0; m < (GATE_UP ? 2 : 1); ++m) {
                const uint8_t* q = blob + a.tr[GATE_UP ? m : 2] + (size_t) (c0 + tr) * rb + (size_t) kb * kMxBytes;
                const float d = e8m0_half(q[0]);
                const int b0 = (tq & 1) * 8, hn = tq >> 1;
                __half* dst = m == 0 ? &T.wg[tr][tq * 8] : &T.wu[tr][tq * 8];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const uint32_t c = q[1 + b0 + j];
                    dst[j] = __float2half_rn(lut[hn ? (c >> 4) : (c & 15u)] * d);
                }
            }
            // the inputs: row tr, values 8 tq ..
            const int sr = src[tr];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                __half v = __float2half_rn(0.0f);
                if (sr >= 0) {
                    const size_t at = (size_t) sr * a.ldx + (size_t) kb * kMxBlock + tq * 8 + j;
                    v = GATE_UP ? __float2half_rn(a.xf[at]) : a.xh[at];
                }
                T.x[tr][tq * 8 + j] = v;
            }
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 32; kk += 16) {
            wm::fragment<wm::matrix_a, 16, 16, 16, __half, wm::row_major> fa;
            wm::load_matrix_sync(fa, &T.x[wm_ * 16][kk], kLdT);
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                wm::fragment<wm::matrix_b, 16, 16, 16, __half, wm::col_major> fb;
                wm::load_matrix_sync(fb, &T.wg[(wn + j) * 16][kk], kLdT);
                wm::mma_sync(fg[j], fa, fb, fg[j]);
                if (GATE_UP) {
                    wm::load_matrix_sync(fb, &T.wu[(wn + j) * 16][kk], kLdT);
                    wm::mma_sync(fu[j], fa, fb, fu[j]);
                }
            }
        }
        __syncthreads();
    }
    for (int j = 0; j < 2; ++j) {
        wm::store_matrix_sync(&O.g[wm_ * 16][(wn + j) * 16], fg[j], BN + 4, wm::mem_row_major);
        if (GATE_UP) wm::store_matrix_sync(&O.u[wm_ * 16][(wn + j) * 16], fu[j], BN + 4, wm::mem_row_major);
    }
    __syncthreads();
    for (int i = tid; i < BM * BN; i += kThreads) {
        const int r = i / BN, c = i % BN;
        if (r >= nr) continue;
        const size_t lr = (size_t) (r0 + r);
        if (GATE_UP) a.h[lr * a.ldo + c0 + c] = __float2half_rn(swiglu(O.g[r][c], O.u[r][c], a.limit));
        else a.y[lr * a.ldo + c0 + c] = O.g[r][c];
    }
}
#endif
#if defined(STRATA_USE_HIP)
// the same on an AMD wave32 GPU's WMMA (RDNA3 / RDNA3.5 gfx11, RDNA4 gfx12): 8 waves each 16 rows x 32 outputs, the
// tiles in shared memory as F16, F32 accumulation - each lane gives its row / column's 16 values of a k-step (gfx12:
// its half of them), and holds 8 of the 16 x 16 results (rows 2 i + hi on gfx11, 8 hi + i on gfx12; column lane % 16)
using h16v = _Float16 __attribute__((ext_vector_type(16)));
using f8v = float __attribute__((ext_vector_type(8)));
__device__ __forceinline__ f8v set_wmma(h16v a, h16v b, f8v c) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#elif defined(__gfx1200__) || defined(__gfx1201__)
    using h8 = _Float16 __attribute__((ext_vector_type(8)));
    const int k0 = (threadIdx.x & 31) / 16 * 8;
    h8 aa, bb;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        aa[i] = a[k0 + i];
        bb[i] = b[k0 + i];
    }
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(aa, bb, c);
#else
    __builtin_trap();
    return c;
#endif
}
__device__ __forceinline__ int set_wmma_row(int i, int hi) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    return 8 * hi + i;
#else
    return 2 * i + hi;
#endif
}
constexpr int kLdH = 40;   // halves a tile row (32 + 8: the pad spreads the banks)
template <bool GATE_UP>
__global__ void __launch_bounds__(kThreads) mx_set_wmma_kernel(const SetK a) {
    struct Tiles {
        _Float16 x[BM][kLdH], wg[BN][kLdH], wu[GATE_UP ? BN : 1][kLdH];
    };
    struct Out {
        float g[BM][BN + 4], u[GATE_UP ? BM : 1][BN + 4];
    };
    __shared__ __align__(16) unsigned char smem[sizeof(Tiles) > sizeof(Out) ? sizeof(Tiles) : sizeof(Out)];
    __shared__ float lut[16];
    __shared__ int src[BM];
    Tiles& T = *reinterpret_cast<Tiles*>(smem);
    Out& O = *reinterpret_cast<Out*>(smem);
    lut_init(lut);
    const int e = blockIdx.z;
    const int lo = a.bounds[e] - a.row0, hi_r = a.bounds[e + 1] - a.row0;
    const int r0 = lo + blockIdx.x * BM;
    if (r0 >= hi_r || hi_r - lo <= a.skip_le) return;   // (uniform; skip_le: the light kernels' experts)
    const int nr = min(BM, hi_r - r0);
    const int c0 = blockIdx.y * BN;
    const uint8_t* blob = a.wbase + (size_t) e * a.stride;
    const int K = GATE_UP ? a.n_embd : a.n_ff;
    const size_t rb = mx_row_bytes(K);
    const int tid = (int) threadIdx.x, warp = tid >> 5, lane = tid & 31, l16 = lane & 15, hi = lane >> 4;
    if (tid < BM) src[tid] = tid < nr ? (GATE_UP ? a.idx[r0 + tid] : r0 + tid) : -1;
    __syncthreads();
    const int wm_ = warp >> 1, wn = (warp & 1) * 2;   // the wave's 16 rows, its 2 x 16 outputs
    f8v fg[2], fu[2];
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i) fg[j][i] = fu[j][i] = 0.0f;
    const int tr = tid >> 2, tq = tid & 3;   // the thread's tile row, its 8 values [8 tq, 8 tq + 8)
    for (int kb = 0; kb < K / kMxBlock; ++kb) {
        for (int m = 0; m < (GATE_UP ? 2 : 1); ++m) {
            const uint8_t* q = blob + a.tr[GATE_UP ? m : 2] + (size_t) (c0 + tr) * rb + (size_t) kb * kMxBytes;
            const float d = e8m0_half(q[0]);
            const int b0 = (tq & 1) * 8, hn = tq >> 1;
            _Float16* dst = m == 0 ? &T.wg[tr][tq * 8] : &T.wu[tr][tq * 8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t cc = q[1 + b0 + j];
                dst[j] = (_Float16) (lut[hn ? (cc >> 4) : (cc & 15u)] * d);
            }
        }
        const int sr = src[tr];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float v = 0.0f;
            if (sr >= 0) {
                const size_t at = (size_t) sr * a.ldx + (size_t) kb * kMxBlock + tq * 8 + j;
                v = GATE_UP ? a.xf[at] : __half2float(a.xh[at]);
            }
            T.x[tr][tq * 8 + j] = (_Float16) v;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 32; kk += 16) {
            h16v av;
#pragma unroll
            for (int i = 0; i < 16; ++i) av[i] = T.x[wm_ * 16 + l16][kk + i];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                h16v bv;
#pragma unroll
                for (int i = 0; i < 16; ++i) bv[i] = T.wg[(wn + j) * 16 + l16][kk + i];
                fg[j] = set_wmma(av, bv, fg[j]);
                if (GATE_UP) {
#pragma unroll
                    for (int i = 0; i < 16; ++i) bv[i] = T.wu[(wn + j) * 16 + l16][kk + i];
                    fu[j] = set_wmma(av, bv, fu[j]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = wm_ * 16 + set_wmma_row(i, hi), cc = (wn + j) * 16 + l16;
            O.g[r][cc] = fg[j][i];
            if (GATE_UP) O.u[r][cc] = fu[j][i];
        }
    __syncthreads();
    for (int i = tid; i < BM * BN; i += kThreads) {
        const int r = i / BN, cc = i % BN;
        if (r >= nr) continue;
        const size_t lr = (size_t) (r0 + r);
        if (GATE_UP) a.h[lr * a.ldo + c0 + cc] = __float2half_rn(swiglu(O.g[r][cc], O.u[r][cc], a.limit));
        else a.y[lr * a.ldo + c0 + cc] = O.g[r][cc];
    }
}
#endif
bool tc_ok() {   // sm_70 and later (WMMA F16) / AMD gfx11, gfx12 (wave32 WMMA); STRATA_DS4_NO_TC=1 keeps the F32 kernel
#if defined(STRATA_USE_HIP)
    static const bool ok = [] {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return false;
        const char* v = std::getenv("STRATA_DS4_NO_TC");
        if (v != nullptr && v[0] == '1') return false;
        const char* arch = prop.gcnArchName;
        for (const char* p : {"gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx1200", "gfx1201"})
            if (std::strncmp(arch, p, 7) == 0) return true;
        return false;
    }();
    return ok;
#else
    static const bool ok = [] {
        int dev = 0, major = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        const char* v = std::getenv("STRATA_DS4_NO_TC");
        return major >= 7 && !(v != nullptr && v[0] == '1');
    }();
    return ok;
#endif
}

// ---------------------------------------------------------------- scratch
constexpr int kMaxEmbd = 8192;
constexpr int kMaxQK = 32768;   // a Q8_0 GEMV's inputs at most
struct Scratch {
    float* part = nullptr;   // [8][kMaxEmbd]: a token's experts' down outputs
    int8_t *xq = nullptr, *hq = nullptr, *mq = nullptr;   // int8 rows: x [kMaxEnt], h [kMaxEnt * 8], the GEMVs'
    float *xd = nullptr, *hd = nullptr, *md = nullptr;     // [kMaxMvJobs * kMaxMvCols] inputs; their scales
};
std::mutex g_mu;
std::unordered_map<cudaStream_t, Scratch> g_scratch;

Scratch* scratch(cudaStream_t s) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_scratch.find(s);
    if (it != g_scratch.end()) return &it->second;
    Scratch sc;
    const size_t nx = (size_t) kMaxEnt * kMaxEmbd, nh = (size_t) kMaxEnt * 8 * kMaxEmbd,
                 nm = (size_t) x3::kMaxMvJobs * x3::kMaxMvCols * kMaxQK;
    if (cudaMalloc(&sc.part, (size_t) 8 * kMaxEmbd * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&sc.xq, nx) != cudaSuccess || cudaMalloc(&sc.xd, nx / 32 * 4) != cudaSuccess ||
        cudaMalloc(&sc.hq, nh) != cudaSuccess || cudaMalloc(&sc.hd, nh / 32 * 4) != cudaSuccess ||
        cudaMalloc(&sc.mq, nm) != cudaSuccess || cudaMalloc(&sc.md, nm / 32 * 4) != cudaSuccess) {
        fail("scratch: cudaMalloc failed");
        return nullptr;
    }
    return &g_scratch.emplace(s, sc).first->second;
}

bool mx_geometry(const char* who, int k, int n_embd, int n_ff) {
    if (k < 1 || k > 8 || n_embd % BN != 0 || n_ff % BN != 0 || n_embd > kMaxEmbd || n_ff % kMxBlock != 0) {
        std::fprintf(stderr, "ds4 gguf %s: geometry not covered (k %d n_embd %d n_ff %d)\n", who, k, n_embd, n_ff);
        g_errors.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------- entry points
bool prepare(cudaStream_t s) { return scratch(s) != nullptr; }

int launch_errors() { return g_errors.load(std::memory_order_relaxed); }

size_t type_row_bytes(int type, int64_t k) {
    if (type == kTypeMXFP4) return k % kMxBlock == 0 ? mx_row_bytes(k) : 0;
    // known(), not supported(): a type with no device kernel still has a layout, and its rows are read by the CPU lane
    return iq::known(type) && k % iq::block_values(type) == 0 ? iq::row_bytes(type, k) : 0;
}
int type_block(int type) { return type == kTypeMXFP4 ? kMxBlock : iq::known(type) ? iq::block_values(type) : 0; }
const char* type_name(int type) { return type == kTypeMXFP4 ? "MXFP4" : iq::name(type); }
bool type_on_device(int type) { return type == kTypeMXFP4 || iq::supported(type); }
bool type_on_chunk(int type) {
#if defined(STRATA_METAL)
    // ds4_mm.metal instantiates the tile GEMM for IQ2_XXS / IQ2_XS / IQ2_S only; IQ1_M (the 2.00 bpw pack's type) has
    // the matvec kernel but no tile one, so a chunked prompt would stop in "kernel launch with no Metal dispatch".
    if (type == iq::kIQ4_NL) return false;   // (IQ1_M and IQ1_S: the tile kernels, DS4_MM_INST(29) / (19) and iq_bs)
    return type_on_device(type);
#else
    (void) type;
    return true;   // CUDA and HIP: the chunk path takes whatever the layer holds (types with no GPU kernel stay on the CPU lane)
#endif
}

namespace {
size_t row_stride(const x3::Mat& m) {   // values (BF16) or bytes (MXFP4, Q8_0, a low-bit type) a row
    if (m.ld > 0) return (size_t) m.ld;
    return m.fmt == kFmtMXFP4 ? mx_row_bytes(m.k)
           : m.fmt == kFmtQ8_0 ? q8_row_bytes(m.k)
           : m.fmt == kFmtGG   ? type_row_bytes(m.K, m.k)
                               : (size_t) m.k;
}
// a layer's low-bit experts as ds4_iq's arguments (false: its types not covered - printed)
bool iq_args(const x3::ExpertLayout& lay, int n_embd, int n_ff, float limit, iq::Args& ia, const char* who) {
    ia = iq::Args{};
    for (int r = 0; r < 3; ++r) ia.tr[r] = lay.trellis[r];
    ia.n_embd = n_embd;
    ia.n_ff = n_ff;
    ia.limit = limit;
    ia.tg = expert_type(lay, 0);
    ia.td = expert_type(lay, 2);
    if (ia.tg != expert_type(lay, 1) || !iq::supported(ia.tg) || !iq::supported(ia.td) || n_embd % iq::block_values(ia.tg) != 0 ||
        n_ff % iq::block_values(ia.td) != 0) {
        std::fprintf(stderr, "ds4 gguf %s: expert types %s / %s / %s (n_embd %d, n_ff %d) not covered\n", who,
                     type_name(expert_type(lay, 0)), type_name(expert_type(lay, 1)), type_name(expert_type(lay, 2)),
                     n_embd, n_ff);
        g_errors.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    return true;
}
template <int NC>
void mv_launch(int f, const BfJobs& J, int nj, int max_n, bool vec, cudaStream_t s) {
    const dim3 grid((unsigned) ((max_n + kWarps - 1) / kWarps), (unsigned) nj);
    if (f == 1) {
#if defined(STRATA_METAL)
        // the MXFP4 matvec: the engine's jobs go across as they are (same layout on both sides)
        if (max_n > 0 && nj > 0 &&
            ::strata::dsv41::metal_host::mx_mv((const void*) &J, nj, max_n, sizeof(BfJob), 1)) return;
#endif
        if (vec) strata_ds4_launch(mx_mv_kernel<NC, true, false>, grid, kThreads, 0, s, J);
        else strata_ds4_launch(mx_mv_kernel<NC, false, false>, grid, kThreads, 0, s, J);
    } else if (f == 2) {   // (vec here: every row 4-byte aligned, so staged)
#if defined(STRATA_METAL)
        if (max_n > 0 && nj > 0 &&
            ::strata::dsv41::metal_host::mx_mv((const void*) &J, nj, max_n, sizeof(BfJob), 2)) return;
#endif
        if (vec) strata_ds4_launch(q8_mv_kernel<NC, true>, grid, kThreads, 0, s, J);
        else strata_ds4_launch(q8_mv_kernel<NC, false>, grid, kThreads, 0, s, J);
    } else {
#if defined(STRATA_METAL)
        // the BF16 matvec (vec: the rows and activations 16-byte aligned, so the 8-value step is safe)
        if (vec && max_n > 0 && nj > 0 &&
            ::strata::dsv41::metal_host::mx_mv((const void*) &J, nj, max_n, sizeof(BfJob), 0)) return;
#endif
        if (vec) strata_ds4_launch(bf16_mv_kernel<NC, true>, grid, kThreads, 0, s, J);
        else strata_ds4_launch(bf16_mv_kernel<NC, false>, grid, kThreads, 0, s, J);
    }
}
void mv_set(int f, const BfJobs& J, int nj, int max_n, int max_c, bool vec, cudaStream_t s) {
    if (nj == 0) return;
    if (max_c <= 1) mv_launch<1>(f, J, nj, max_n, vec, s);
    else if (max_c <= 2) mv_launch<2>(f, J, nj, max_n, vec, s);
    else if (max_c <= 4) mv_launch<4>(f, J, nj, max_n, vec, s);
    else if (max_c <= 8) mv_launch<8>(f, J, nj, max_n, vec, s);
    else mv_launch<16>(f, J, nj, max_n, vec, s);
}
}  // namespace

x3::Mat view(const x3::Mat& m, int c0, int nc) {
    x3::Mat v = m;
    const size_t ld = row_stride(m);
    v.ld = (int) ld;
    v.raw = m.fmt != kFmtBF16 ? (const void*) ((const uint8_t*) m.raw + (size_t) c0 * ld)
                              : (const void*) ((const uint16_t*) m.raw + (size_t) c0 * ld);
    v.n = nc;
    return v;
}

bool mv(const x3::MvJob* jobs, int n, cudaStream_t s) {
    if (n <= 0 || n > x3::kMaxMvJobs) return false;
#if defined(STRATA_METAL)
    if (ds4_dbg("DS4_MV_LOG")) {   // how the engine batches its matvec jobs (one call = one quantiser dispatch)
        static std::atomic<unsigned long long> calls{0}, jobsum{0};
        static std::atomic<unsigned long long> hist[9];
        static std::atomic<int> seen{0};
        calls.fetch_add(1); jobsum.fetch_add((unsigned long long) n);
        hist[n < 8 ? n : 8].fetch_add(1);
        if (seen.fetch_add(1) % 256 == 255)
            std::fprintf(stderr, "mvlog calls=%llu jobs=%llu avg=%.2f hist1..4=%llu,%llu,%llu,%llu 5+=%llu,%llu,%llu,%llu\n",
                         (unsigned long long) calls.load(), (unsigned long long) jobsum.load(),
                         (double) jobsum.load() / (double) calls.load(),
                         (unsigned long long) hist[1].load(), (unsigned long long) hist[2].load(),
                         (unsigned long long) hist[3].load(), (unsigned long long) hist[4].load(),
                         (unsigned long long) hist[5].load(), (unsigned long long) hist[6].load(),
                         (unsigned long long) hist[7].load(), (unsigned long long) hist[8].load());
    }
#endif
    BfJobs J[3] = {};   // the BF16 jobs, the MXFP4 ones, the Q8_0 ones: a launch each
    int nj[3] = {0, 0, 0}, max_n[3] = {0, 0, 0}, max_c[3] = {1, 1, 1};
    bool vec[3] = {true, true, true};
    iq::MvJob IJ[x3::kMaxMvJobs];   // the low-bit ones (ds4_iq: a launch a type)
    int nij = 0;
    QuantItems QI;   // the int8 roundings of this call's jobs, one launch (identical inputs once)
    QI.n = 0;
    for (int i = 0; i < n; ++i) {
        const x3::MvJob& q = jobs[i];
        const x3::Mat& m = *q.m;
        if (m.fmt == kFmtGG) {   // the activations rounded to int8 (job i's rows of the scratch), then ds4_iq's GEMV
            Scratch* sc = scratch(s);
            if (sc == nullptr || m.raw == nullptr || q.xf == nullptr || q.y == nullptr || q.ncol < 1 ||
#if !defined(STRATA_METAL)
                q.ncol > x3::kMaxMvCols ||
#endif
                m.k > kMaxQK || type_row_bytes(m.K, m.k) == 0) {
                std::fprintf(stderr, "ds4 gguf mv: job %d unsupported (type %s k %d n %d)\n", i, type_name(m.K), m.k, m.n);
                g_errors.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            const size_t at = (size_t) i * x3::kMaxMvCols * kMaxQK;
            const int qi = quant_add(QI, q.xf, q.ncol > 1 ? q.xf_cs : (size_t) m.k, q.ncol, m.k, sc->mq + at, sc->md + at / 32);
            iq::MvJob& b = IJ[nij++];
            b.w = (const uint8_t*) m.raw;
            b.ld = row_stride(m);
            b.type = m.K;
            b.k = m.k;
            b.n = m.n;
            b.ncol = q.ncol;
            b.xq = QI.q[qi];
            b.xd = QI.d[qi];
            b.y = q.y;
            b.ycs = q.y_cs;
            b.bias = q.bias;
            b.alpha = q.alpha;
            continue;
        }
        const int f = m.fmt == kFmtMXFP4 ? 1 : m.fmt == kFmtQ8_0 ? 2 : 0;
        if ((m.fmt != kFmtBF16 && m.fmt != kFmtMXFP4 && m.fmt != kFmtQ8_0) || m.raw == nullptr || q.xf == nullptr ||
            q.y == nullptr || q.ncol < 1 || q.ncol > x3::kMaxMvCols || m.k <= 0 || m.n <= 0 ||
            (f != 0 && m.k % kMxBlock != 0)) {
            std::fprintf(stderr, "ds4 gguf mv: job %d unsupported (fmt %d k %d n %d, %d activations)\n", i, m.fmt, m.k,
                         m.n, q.ncol);
            g_errors.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        BfJob& b = J[f].j[nj[f]++];
        b.w = m.raw;
        b.ld = row_stride(m);
        b.x = q.xf;
        b.xcs = q.xf_cs;
        b.y = q.y;
        b.ycs = q.y_cs;
        b.bias = q.bias;
        b.alpha = q.alpha;
        b.k = m.k;
        b.n = m.n;
        b.ncol = q.ncol;
        max_n[f] = std::max(max_n[f], m.n);
        max_c[f] = std::max(max_c[f], q.ncol);
        const bool xa = ((uintptr_t) b.x & 15u) == 0 && (q.ncol == 1 || b.xcs % 4 == 0);
        if (f == 2) {   // the activations rounded to int8 first (each job its own rows of the scratch)
            Scratch* sc = scratch(s);
            if (sc == nullptr || m.k > kMaxQK) {
                fail("mv: Q8_0 inputs past the scratch");
                return false;
            }
            const size_t at = (size_t) i * x3::kMaxMvCols * kMaxQK;
            const int qi = quant_add(QI, q.xf, q.ncol > 1 ? q.xf_cs : (size_t) m.k, q.ncol, m.k, sc->mq + at, sc->md + at / 32);
            b.xq = QI.q[qi];
            b.xd = QI.d[qi];
            vec[f] = vec[f] && b.ld % 4 == 0 && ((uintptr_t) b.w & 3u) == 0 && (m.k / 32) % 2 == 0;
        } else {
            vec[f] = vec[f] && xa && (f != 0 || (m.k % 8 == 0 && b.ld % 8 == 0 && ((uintptr_t) b.w & 15u) == 0));
        }
    }
    quant_flush(QI, s);
    for (int f = 0; f < 3; ++f) mv_set(f, J[f], nj[f], max_n[f], max_c[f], vec[f], s);
#if defined(STRATA_METAL)
    if (ds4_dbg("DS4_DBG_Q8") && nj[2] > 0) {   // DBG: the Q8_0 matvec's first job against a host recompute
        ds4_metal_flush();
        const BfJob& b = J[2].j[0];
        const uint8_t* w = (const uint8_t*) strata::gpu::host_view(b.w);
        const int8_t* xq = (const int8_t*) strata::gpu::host_view(b.xq);
        const float* xd = (const float*) strata::gpu::host_view(b.xd);
        const float* y = (const float*) strata::gpu::host_view(b.y);
        const float* xf = (const float*) strata::gpu::host_view(b.x);
        for (int c = 0; c < std::min(b.ncol, 2); ++c)
            for (int r = 0; r < 2; ++r) {
                double acc = 0.0, accf = 0.0;
                for (int kb = 0; kb < b.k / 32; ++kb) {
                    const uint8_t* p = w + (size_t) r * b.ld + (size_t) kb * 34;
                    uint16_t dh;
                    std::memcpy(&dh, p, 2);
                    float dw = __half2float(*(const __half*) &dh);
                    int sum = 0;
                    for (int j = 0; j < 32; ++j) sum += (int) (int8_t) p[2 + j] * (int) xq[(size_t) c * b.k + kb * 32 + j];
                    acc += (double) dw * xd[(size_t) c * (b.k / 32) + kb] * sum;
                    for (int j = 0; j < 32; ++j) accf += (double) dw * (int) (int8_t) p[2 + j] * xf[(size_t) c * b.xcs + kb * 32 + j];
                }
                std::fprintf(stderr, "DBG q8 job0 k=%d n=%d ncol=%d ld=%zu xcs=%zu c=%d r=%d recompute=%.5g exactf=%.5g kernel_y=%.5g xd0=%g xq0=%d\n", b.k, b.n, b.ncol,
                             b.ld, b.xcs, c, r, acc, accf, y[(size_t) c * b.ycs + r], xd[(size_t) c * (b.k / 32)], (int) xq[(size_t) c * b.k]);
            }
    }
#endif
    if (nij > 0 && !iq::mv(IJ, nij, s)) {
        fail("mv: a low-bit job not covered");
        return false;
    }
#if defined(STRATA_METAL)
    if (ds4_dbg("DS4_DBG_IQ") && nij > 0) {   // DBG: the low-bit matvec's first job against a host decode of its rows
        ds4_metal_flush();
        for (int ji = 0; ji < std::min(nij, 2); ++ji) {
            const iq::MvJob& b = IJ[ji];
            const uint8_t* w = (const uint8_t*) strata::gpu::host_view(b.w);
            const int8_t* xq = (const int8_t*) strata::gpu::host_view(b.xq);
            const float* xd = (const float*) strata::gpu::host_view(b.xd);
            const float* y = (const float*) strata::gpu::host_view(b.y);
            std::vector<uint16_t> row((size_t) b.k);
            for (int r = 0; r < 3; ++r) {
                if (!gg_row_to_bf16(b.type, w + (size_t) r * b.ld, b.k, row.data())) { std::fprintf(stderr, "DBG iq: no host decode for type %d\n", b.type); break; }
                double acc = 0.0;
                for (int i = 0; i < b.k; ++i) {
                    uint32_t u = (uint32_t) row[(size_t) i] << 16; float wf; std::memcpy(&wf, &u, 4);
                    acc += (double) wf * xq[i] * xd[i / 32];
                }
                std::fprintf(stderr, "DBG iq job %d type %d k=%d n=%d ncol=%d ld=%zu r=%d host=%.5g kernel=%.5g w=%p\n", ji, b.type, b.k, b.n, b.ncol, b.ld, r, acc, y[r], (const void*) b.w);
            }
        }
    }
#endif
    check("mv");
    return true;
}

void rows(const x3::Mat& m, uint16_t* dst, bool bf16, cudaStream_t s) {
#if defined(STRATA_METAL)
    if (!bf16 && m.raw != nullptr) {   // the prompt chunk's dense matrices: F16 rows for the MPP GEMM
        const int kind = m.fmt == kFmtGG ? m.K : m.fmt == kFmtQ8_0 ? 8 : m.fmt == kFmtBF16 ? 0 : -1;
        if (kind >= 0 && ::strata::dsv41::metal_host::dequant_rows(kind, m.raw, row_stride(m), m.k, m.n, dst)) return;
    }
#endif
    if (m.fmt == kFmtGG && m.raw != nullptr) {   // a K-quant / i-quant dense matrix (the IST-DAS sets' Q5_K): its own dequantizer
        if (!iq::dequant_rows(m.K, m.raw, row_stride(m), m.k, m.n, dst, bf16, s)) fail("rows: a low-bit type not covered");
        return;
    }
    if ((m.fmt != kFmtBF16 && m.fmt != kFmtQ8_0) || m.raw == nullptr) {
        fail("rows: not a BF16 or Q8_0 matrix");
        return;
    }
    const size_t ld = row_stride(m);
    const size_t total = (size_t) m.n * (size_t) m.k;
    const unsigned blocks = (unsigned) std::min<size_t>((total + 255) / 256, 65535);
    if (m.fmt == kFmtQ8_0) strata_ds4_launch(q8_rows_kernel, blocks, 256, 0, s, (const uint8_t*) m.raw, ld, m.k, m.n, dst, bf16);
    else strata_ds4_launch(bf16_rows_kernel, blocks, 256, 0, s, (const uint16_t*) m.raw, ld, m.k, m.n, dst, bf16);
    check("rows");
}

namespace {
// (staged when the matrix's rows are 4-byte aligned: 17 nb % 4 == 0; the blobs' offsets are)
template <int NE>
struct GateUpK {
    static void run(const MxArgs& m, dim3 grid, cudaStream_t s) {
        if (mx_row_bytes(m.n_embd) % 4 == 0) strata_ds4_launch(mx_gate_up_kernel<NE, true>, grid, kThreads, 0, s, m);
        else strata_ds4_launch(mx_gate_up_kernel<NE, false>, grid, kThreads, 0, s, m);
    }
};
template <int NE>
struct DownK {
    static void run(const MxArgs& m, dim3 grid, cudaStream_t s) {
        if (mx_row_bytes(m.n_ff) % 4 == 0) strata_ds4_launch(mx_down_kernel<NE, true>, grid, kThreads, 0, s, m);
        else strata_ds4_launch(mx_down_kernel<NE, false>, grid, kThreads, 0, s, m);
    }
};
template <template <int> class Kern>
void launch_groups(const MxArgs& m, int rows, int groups, int ne_max, cudaStream_t s) {
    const dim3 grid((unsigned) ((rows + kWarps - 1) / kWarps), (unsigned) groups);
    if (ne_max <= 1) Kern<1>::run(m, grid, s);
    else if (ne_max <= 4) Kern<4>::run(m, grid, s);
    else if (ne_max <= 8) Kern<8>::run(m, grid, s);
    else Kern<16>::run(m, grid, s);
}
MxArgs mx_args(const x3::ExpertLayout& lay, int n_embd, int n_ff, float limit) {
    MxArgs m{};
    for (int r = 0; r < 3; ++r) m.tr[r] = lay.trellis[r];
    m.n_embd = n_embd;
    m.n_ff = n_ff;
    m.limit = limit;
    return m;
}
}  // namespace

void moe_gate_up(const x3::MoeArgs& a, const float* x, float* h, cudaStream_t s) {
    if (!mx_geometry("moe_gate_up", a.k, a.n_embd, a.n_ff) || h == nullptr) return;
    Scratch* sc = scratch(s);
    if (sc == nullptr) return;
    if (!mx_experts(a.lay)) {   // low-bit experts: ds4_iq's kernels
        iq::Args ia;
        if (!iq_args(a.lay, a.n_embd, a.n_ff, a.limit, ia, "moe_gate_up")) return;
        ia.plan_ptr = a.plan_ptr;
        ia.h = h;
        ia.xq = sc->xq;
        ia.xd = sc->xd;
        q8_quant(x, (size_t) a.n_embd, 1, a.n_embd, sc->xq, sc->xd, s);
        iq::gate_up(ia, a.k, 1, s);
        check("moe_gate_up");
        return;
    }
    MxArgs m = mx_args(a.lay, a.n_embd, a.n_ff, a.limit);
    m.plan_ptr = a.plan_ptr;
    m.x = x;
    m.h = h;
    m.xq = sc->xq;
    m.xd = sc->xd;
    q8_quant(x, (size_t) a.n_embd, 1, a.n_embd, sc->xq, sc->xd, s);
#if defined(STRATA_METAL)
    // mx_gate_up_kernel on the Metal dispatch (the MXFP4 expert GEMM); the launch below is the fallback
    if (::strata::dsv41::metal_host::mx_gu(a.plan_ptr, nullptr, nullptr, nullptr, nullptr, a.lay.trellis, a.n_embd,
                                           a.n_ff, a.limit, sc->xq, sc->xd, h, a.k)) {
        check("moe_gate_up");
        return;
    }
#endif
    launch_groups<GateUpK>(m, a.n_ff, a.k, 1, s);
    check("moe_gate_up");
}

void moe_down(const x3::MoeArgs& a, const float* h, const float* sh_out, float* out, cudaStream_t s) {
    if (!mx_geometry("moe_down", a.k, a.n_embd, a.n_ff) || h == nullptr) return;
    Scratch* sc = scratch(s);
    if (sc == nullptr) return;
    MxArgs m = mx_args(a.lay, a.n_embd, a.n_ff, a.limit);
    m.plan_ptr = a.plan_ptr;
    m.h = const_cast<float*>(h);
    m.part = sc->part;
    m.hq = sc->hq;
    m.hd = sc->hd;
    q8_quant(h, (size_t) a.n_ff, a.k, a.n_ff, sc->hq, sc->hd, s);
    if (!mx_experts(a.lay)) {   // low-bit experts: ds4_iq's down, then the same combine
        iq::Args ia;
        if (!iq_args(a.lay, a.n_embd, a.n_ff, a.limit, ia, "moe_down")) return;
        ia.plan_ptr = a.plan_ptr;
        ia.part = sc->part;
        ia.hq = sc->hq;
        ia.hd = sc->hd;
        iq::down(ia, a.k, 1, s);
    } else {
#if defined(STRATA_METAL)
        // mx_down_kernel on the Metal dispatch; the launch below is the fallback
        if (!::strata::dsv41::metal_host::mx_dn(a.plan_ptr, nullptr, nullptr, nullptr, nullptr, a.lay.trellis, a.n_embd,
                                                a.n_ff, sc->hq, sc->hd, sc->part, a.k))
#endif
            launch_groups<DownK>(m, a.n_embd, a.k, 1, s);
    }
    CombArgs c{};
    c.plan_ptr = a.plan_ptr;
    c.plan_w = a.plan_w;
    c.part = sc->part;
    c.cpu_part = a.cpu_part;
    c.cpu_stride = (size_t) a.n_embd;
    c.cpu_flag = a.cpu_flag;
    c.sh_out = sh_out;
    c.out = out;
    c.out_stride = (size_t) a.n_embd;
    c.k = a.k;
    c.n_embd = a.n_embd;
#if defined(STRATA_METAL)
    if (!::strata::dsv41::metal_host::mx_combine(c.plan_ptr, c.plan_w, c.part, c.cpu_part, c.cpu_stride,
                                                 c.cpu_flag, c.sh_out, c.out, c.out_stride, 1, c.k, c.n_embd))
#endif
    strata_ds4_launch(mx_combine_kernel, dim3((unsigned) ((a.n_embd + 127) / 128), 1), 128, 0, s, c);
    check("moe_down");
}

bool win_experts(const x3::WinArgs& a, cudaStream_t s) {
    if (a.T < 1 || a.T > kMaxEnt || !mx_geometry("win_experts", a.k, a.n_embd, a.n_ff) || a.h == nullptr ||
        a.part == nullptr || a.out == nullptr) {
        std::fprintf(stderr, "ds4 gguf win_experts: not covered (T %d)\n", a.T);
        return false;
    }
    MxArgs m = mx_args(a.lay, a.n_embd, a.n_ff, a.limit);
    m.grp_ptr = a.grp_ptr;
    m.grp_start = a.grp_start;
    m.grp_ent = a.grp_ent;
    m.n_groups = a.n_groups;
    Scratch* sc = scratch(s);
    if (sc == nullptr) return false;
    m.x = a.x;
    m.x_stride = a.x_stride;
    m.h = a.h;
    m.part = a.part;
    m.xq = sc->xq;
    m.xd = sc->xd;
    m.hq = sc->hq;
    m.hd = sc->hd;
    const int groups = a.T * a.k;   // at most (the blocks past the real count return at once)
    const bool mx = mx_experts(a.lay);
    iq::Args ia;
    if (!mx) {   // low-bit experts: ds4_iq's kernels on the same groups and scratch
        if (!iq_args(a.lay, a.n_embd, a.n_ff, a.limit, ia, "win_experts")) return false;
        ia.grp_ptr = a.grp_ptr;
        ia.grp_start = a.grp_start;
        ia.grp_ent = a.grp_ent;
        ia.n_groups = a.n_groups;
        ia.h = a.h;
        ia.part = a.part;
        ia.xq = sc->xq;
        ia.xd = sc->xd;
        ia.hq = sc->hq;
        ia.hd = sc->hd;
    }
    q8_quant(a.x, a.x_stride, a.T, a.n_embd, sc->xq, sc->xd, s);
    if (mx) {
#if defined(STRATA_METAL)
        // mx_gate_up_kernel on the Metal dispatch; the launch below is the fallback
        if (!::strata::dsv41::metal_host::mx_gu(a.plan_ptr, a.grp_ptr, a.grp_start, a.grp_ent, a.n_groups,
                                                a.lay.trellis, a.n_embd, a.n_ff, a.limit, sc->xq, sc->xd, a.h, groups))
#endif
            launch_groups<GateUpK>(m, a.n_ff, groups, a.T, s);
    } else {
        iq::gate_up(ia, groups, a.T, s);
    }
    q8_quant(a.h, (size_t) a.n_ff, a.T * 8, a.n_ff, sc->hq, sc->hd, s);   // (rows of picks not on the device: unused)
    if (mx) {
#if defined(STRATA_METAL)
        // mx_down_kernel on the Metal dispatch; the launch below is the fallback
        if (!::strata::dsv41::metal_host::mx_dn(a.plan_ptr, a.grp_ptr, a.grp_start, a.grp_ent, a.n_groups,
                                                a.lay.trellis, a.n_embd, a.n_ff, sc->hq, sc->hd, a.part, groups))
#endif
            launch_groups<DownK>(m, a.n_embd, groups, a.T, s);
    } else {
        iq::down(ia, groups, a.T, s);
    }
#if !defined(STRATA_METAL)
    // (this port's dispatches are synchronous and nothing is published to wait for: ds4_kernels.cu's wait_seq)
    if (a.wait_ptr != nullptr) strata_ds4_launch(mx_wait_kernel, 1, 32, 0, s, a.wait_ptr, a.wait_val);
#endif
    CombArgs c{};
    c.plan_ptr = a.plan_ptr;
    c.plan_w = a.plan_w;
    c.part = a.part;
    c.cpu_part = a.cpu_part;
    c.cpu_stride = a.cpu_stride != 0 ? a.cpu_stride : (size_t) a.n_embd;
    c.cpu_flag = a.cpu_flag;
    c.sh_out = a.sh_out;
    c.out = a.out;
    c.out_stride = a.out_stride;
    c.k = a.k;
    c.n_embd = a.n_embd;
#if defined(STRATA_METAL)
    // mx_combine_kernel on the Metal dispatch; the launch below is the fallback
    if (!::strata::dsv41::metal_host::mx_combine(c.plan_ptr, c.plan_w, c.part, c.cpu_part, c.cpu_stride,
                                                 c.cpu_flag, c.sh_out, c.out, c.out_stride, a.T, c.k, c.n_embd))
#endif
    strata_ds4_launch(mx_combine_kernel, dim3((unsigned) ((a.n_embd + 127) / 128), (unsigned) a.T), 128, 0, s, c);
    check("win_experts");
    return true;
}

namespace {
iq::SetArgs iq_set(const x3::SetArgs& sa) {   // a set as ds4_iq's arguments (the inputs / outputs and types to fill in)
    iq::SetArgs q;
    q.wbase = sa.wbase;
    q.stride = sa.stride;
    for (int r = 0; r < 3; ++r) q.tr[r] = sa.lay.trellis[r];
    q.bounds = sa.bounds;
    q.row0 = sa.row0;
    q.max_rows = sa.max_rows;
    q.n_exp = sa.n_exp;
    q.n_embd = sa.n_embd;
    q.n_ff = sa.n_ff;
    q.limit = sa.limit;
    q.xf16 = sa.x16;
    q.skip_ge = sa.skip_ge;
#if defined(STRATA_METAL)
    q.hbounds = sa.hbounds;
    q.total_rows = sa.total_rows;
#endif
    return q;
}
}  // namespace

void set_gate_up(const x3::SetArgs& sa, const float* x, int ldx, const int* row_idx, uint16_t* h16, int ldh,
                 cudaStream_t s) {
    if (sa.n_exp <= 0 || sa.max_rows <= 0) return;
    if (!mx_geometry("set_gate_up", 1, sa.n_embd, sa.n_ff)) return;
    if (!mx_experts(sa.lay)) {   // low-bit experts: ds4_iq's sets
        iq::Args ia;
        if (!iq_args(sa.lay, sa.n_embd, sa.n_ff, sa.limit, ia, "set_gate_up")) return;
        iq::SetArgs q = iq_set(sa);
        q.xf = x;
        q.idx = row_idx;
        q.ldx = ldx;
        q.h = h16;
        q.ldo = ldh;
        q.tg = ia.tg;
        q.td = ia.td;
        if (!iq::set_gate_up(q, s)) fail("set_gate_up: not covered");
        check("set_gate_up");
        return;
    }
    int skip = 0;
    {   // the experts of at most 16 rows: the light kernels (each weight read once; ds4_iq.cuh), the others the tiles
        iq::SetArgs q = iq_set(sa);
        q.xf = x;
        q.idx = row_idx;
        q.ldx = ldx;
        q.h = h16;
        q.ldo = ldh;
        q.tg = q.td = kTypeMXFP4;
        if (iq::light_gate_up(q, s)) {
            if (sa.max_rows <= 16) {
                check("set_gate_up");
                return;
            }
            skip = 16;
        }
    }
    SetK k{};
    k.skip_le = skip;
    k.wbase = sa.wbase;
    k.stride = sa.stride;
    for (int r = 0; r < 3; ++r) k.tr[r] = sa.lay.trellis[r];
    k.bounds = sa.bounds;
    k.row0 = sa.row0;
    k.n_embd = sa.n_embd;
    k.n_ff = sa.n_ff;
    k.limit = sa.limit;
    k.xf = x;
    k.idx = row_idx;
    k.ldx = ldx;
    k.h = (__half*) h16;
    k.ldo = ldh;
    const dim3 grid((unsigned) ((sa.max_rows + BM - 1) / BM), (unsigned) (sa.n_ff / BN), (unsigned) sa.n_exp);
#if !defined(STRATA_USE_HIP)
    if (tc_ok()) strata_ds4_launch(mx_set_tc_kernel<true>, grid, kThreads, 0, s, k);
#else
    if (tc_ok()) strata_ds4_launch(mx_set_wmma_kernel<true>, grid, kThreads, 0, s, k);
#endif
    else
        strata_ds4_launch(mx_set_kernel<true>, grid, kThreads, 0, s, k);
    check("set_gate_up");
}

void set_down(const x3::SetArgs& sa, const uint16_t* h16, int ldh, float* y, int ldy, cudaStream_t s) {
    if (sa.n_exp <= 0 || sa.max_rows <= 0) return;
    if (!mx_geometry("set_down", 1, sa.n_embd, sa.n_ff)) return;
    if (!mx_experts(sa.lay)) {   // low-bit experts: ds4_iq's sets
        iq::Args ia;
        if (!iq_args(sa.lay, sa.n_embd, sa.n_ff, sa.limit, ia, "set_down")) return;
        iq::SetArgs q = iq_set(sa);
        q.xh = h16;
        q.ldx = ldh;
        q.y = y;
        q.ldo = ldy;
        q.tg = ia.tg;
        q.td = ia.td;
        if (!iq::set_down(q, s)) fail("set_down: not covered");
        check("set_down");
        return;
    }
    int skip = 0;
    {   // (as set_gate_up's)
        iq::SetArgs q = iq_set(sa);
        q.xh = h16;
        q.ldx = ldh;
        q.y = y;
        q.ldo = ldy;
        q.tg = q.td = kTypeMXFP4;
        if (iq::light_down(q, s)) {
            if (sa.max_rows <= 16) {
                check("set_down");
                return;
            }
            skip = 16;
        }
    }
    SetK k{};
    k.skip_le = skip;
    k.wbase = sa.wbase;
    k.stride = sa.stride;
    for (int r = 0; r < 3; ++r) k.tr[r] = sa.lay.trellis[r];
    k.bounds = sa.bounds;
    k.row0 = sa.row0;
    k.n_embd = sa.n_embd;
    k.n_ff = sa.n_ff;
    k.xh = (const __half*) h16;
    k.ldx = ldh;
    k.y = y;
    k.ldo = ldy;
    const dim3 grid((unsigned) ((sa.max_rows + BM - 1) / BM), (unsigned) (sa.n_embd / BN), (unsigned) sa.n_exp);
#if !defined(STRATA_USE_HIP)
    if (tc_ok()) strata_ds4_launch(mx_set_tc_kernel<false>, grid, kThreads, 0, s, k);
#else
    if (tc_ok()) strata_ds4_launch(mx_set_wmma_kernel<false>, grid, kThreads, 0, s, k);
#endif
    else
        strata_ds4_launch(mx_set_kernel<false>, grid, kThreads, 0, s, k);
    check("set_down");
}

}  // namespace strata::ds4::gguf
