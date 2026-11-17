// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_kernels.cu - DeepSeek-V4.1's device kernels (see ds4_kernels.cuh).  No fast math here: the rotary
// angles, the softmaxes and the roundings follow the reference's F32 arithmetic.  A window's token is blockIdx.y.
#include "ds4_launch.h"   // strata_ds4_launch: <<<>>> on CUDA and HIP, a dispatch on Metal
#include "ds4_kernels.cuh"
#if defined(STRATA_METAL)
#include <ds4_metal_host.h>
#endif
#include "ds4_numerics.cuh"
#include "ds4_q6k.cuh"
#include "ds4_attn_split.cuh"
#include "ds4_wmma.cuh"
#include "ds4_experts_rule.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <cstdio>

namespace strata::ds4 {
namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float bf16f(uint16_t b) { return __uint_as_float((uint32_t) b << 16); }

__device__ __forceinline__ uint16_t f2bf16(float f) {   // round to nearest even (the values here are exact anyway)
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t) (u >> 16);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// block-wide sum for blockDim.x <= 1024 (a multiple of 32); every thread gets the result
__device__ float block_sum(float v) {
    __shared__ float red[32];
    __shared__ float total;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) red[w] = v;
    __syncthreads();
    if (w == 0) {
        float t = lane < (int) (blockDim.x >> 5) ? red[lane] : 0.0f;
        t = warp_sum(t);
        if (lane == 0) total = t;
    }
    __syncthreads();
    const float r = total;
    __syncthreads();
    return r;
}

// 2^ceil(log2(x)) for a positive normal x (the reference's fast_round_scale)
__device__ __forceinline__ float pow2_ceil(float x) {
    const uint32_t b = __float_as_uint(x);
    const int e = (int) ((b >> 23) & 0xffu) - 127 + ((b & 0x7fffffu) != 0u ? 1 : 0);
    return __uint_as_float((uint32_t) (e + 127) << 23);
}

// a load of memory another device or the host writes, never from a stale cache line (CUDA's __ldcv; HIP: volatile
// 32-bit loads)
template <class T>
__device__ __forceinline__ T ld_fresh(const T* p) {
#if defined(STRATA_USE_HIP)
    T v;
    const volatile uint32_t* s = (const volatile uint32_t*) p;
    uint32_t* d = (uint32_t*) &v;
#pragma unroll
    for (int i = 0; i < (int) (sizeof(T) / 4); ++i) d[i] = s[i];
    return v;
#else
    return __ldcv(p);
#endif
}

__device__ __forceinline__ float fp8_rt(float v) { return e4m3_round(v); }   // E4M3, nearest even (input clamped)

__device__ __forceinline__ float e2m1_rt(float v) {   // E2M1 magnitudes 0 .5 1 1.5 2 3 4 6, nearest, ties to even
    const float a = fabsf(v);
    float r;
    if (a <= 0.25f) r = 0.0f;
    else if (a < 0.75f) r = 0.5f;
    else if (a <= 1.25f) r = 1.0f;
    else if (a < 1.75f) r = 1.5f;
    else if (a <= 2.5f) r = 2.0f;
    else if (a < 3.5f) r = 3.0f;
    else if (a <= 5.0f) r = 4.0f;
    else r = 6.0f;
    return copysignf(r, v);
}

// ---------------------------------------------------------------- the caches' packed rows (kv_row_bytes)
// the E2M1 code of a value e2m1_rt returned (bit 3 the sign, -0 kept; magnitudes 0 .5 1 1.5 2 3 4 6 as 0 .. 7)
__device__ __forceinline__ uint32_t e2m1_code(float q) {
    const float a = fabsf(q);
    const uint32_t m = a <= 1.5f ? (uint32_t) (a * 2.0f) : a <= 2.0f ? 4u : a <= 3.0f ? 5u : a <= 4.0f ? 6u : 7u;
    return m | ((__float_as_uint(q) >> 31) << 3);
}
// the E4M3 code of v / 2^k for a v whose quotient is an E4M3 value (|v / 2^k| <= 448, at most 4 significant bits, a
// multiple of 2^-9; v 0 or a normal F32): integer arithmetic, exact, -0 kept
__device__ __forceinline__ uint32_t e4m3_code(float v, int k = 0) {
    const uint32_t b = __float_as_uint(v), sg = (b >> 24) & 0x80u, a = b & 0x7fffffffu;
    if (a == 0u) return sg;
    const int e = (int) (a >> 23) - 127 - k;   // the quotient's binade
    if (e >= -6) return sg | ((uint32_t) (e + 7) << 3) | ((a >> 20) & 7u);
    return sg | (((a & 0x7fffffu) | 0x800000u) >> (14 - e));   // a subnormal: m 2^-9
}
// an E2M1 code's value times 2^-14, by the conversion of an FP16 (the code's sign, exponent and mantissa bit are an
// FP16's at bits 15 and 11..9: exact, subnormals too) - the readers' decode: (value 2^-14) (scale 2^14) is value * scale
// exactly, both products exact
__device__ __forceinline__ float e2m1_val14(uint32_t c) {
    return __half2float(__ushort_as_half((unsigned short) (((c & 8u) << 12) | ((c & 7u) << 9))));
}
// a UE8M0 scale byte's value (the byte the F32 exponent: 2^(b - 127); 0 -> 0, an unwritten row's)
__device__ __forceinline__ float ue8m0_val(uint32_t b) { return __uint_as_float(b << 23); }

// ---------------------------------------------------------------- embed
__global__ void embed_kernel(const uint16_t* __restrict__ rows, float* __restrict__ h, int dim, int hc) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (e >= dim) return;
    const float v = bf16f(rows[(size_t) t * dim + e]);
    for (int s = 0; s < hc; ++s) h[((size_t) t * hc + s) * dim + e] = v;
}

// ---------------------------------------------------------------- finish
constexpr int kFinMax = 32;   // values a thread holds: len <= 32 * 256

// kPack: the rounded rows packed to f.pdst (f.quant's cache format) instead of written to x or f.dst
template <bool kPack>
__global__ void __launch_bounds__(kThreads) finish_kernel(float* x, int len, long long stride, Finish f) {
    float* v = x + (long long) blockIdx.x * stride;
    float r[kFinMax];
    const int nj = (len + kThreads - 1) / kThreads;
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < kFinMax; ++j) {
        if (j >= nj) break;
        const int e = threadIdx.x + j * kThreads;
        r[j] = e < len ? v[e] : 0.0f;
        ss += r[j] * r[j];
    }
    if (f.w != nullptr) {
        const float tot = block_sum(ss);
        const float inv = rsqrtf(tot / (float) len + f.eps);
#pragma unroll
        for (int j = 0; j < kFinMax; ++j) {
            if (j >= nj) break;
            const int e = threadIdx.x + j * kThreads;
            if (e < len) r[j] = r[j] * inv * f.w[e];
        }
    }
    if (f.rd > 0) {
        const int r0 = len - f.rd;
        const int pos = f.pos + (int) (blockIdx.x / f.vpp) * f.pos_step;
#pragma unroll
        for (int j = 0; j < kFinMax; ++j) {
            if (j >= nj) break;
            const int e = threadIdx.x + j * kThreads;
            const float other = __shfl_xor_sync(0xffffffffu, r[j], 1);
            if (e >= r0 && e < len) {
                const int i = (e - r0) >> 1;
                const float ang = (float) pos * f.freqs[i];
                const float c = cosf(ang);
                const float sn = f.inverse ? -sinf(ang) : sinf(ang);
                if (((e - r0) & 1) == 0) r[j] = r[j] * c - other * sn;   // re = a c - b s
                else r[j] = other * sn + r[j] * c;                          // im = a s + b c
            }
        }
    }
    if (f.quant != 0 && !kPack) {
        const int g = f.quant == 2 ? 16 : 32;
#pragma unroll
        for (int j = 0; j < kFinMax; ++j) {
            if (j >= nj) break;
            const int e = threadIdx.x + j * kThreads;
            float a = e < len ? fabsf(r[j]) : 0.0f;
            for (int o = g / 2; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
            if (e >= len) continue;
            if (f.quant == 1) {
                const float s = pow2_ceil(fmaxf(a, 1e-4f) * (1.0f / 448.0f));
                r[j] = fp8_rt(fminf(fmaxf(r[j] / s, -448.0f), 448.0f)) * s;
            } else if (f.quant == 2) {
                const float s = fp8_rt(fmaxf(a, 6.0f * 0.001953125f) / 6.0f);
                r[j] = e2m1_rt(fminf(fmaxf(r[j] / s, -6.0f), 6.0f)) * s;
            } else {
                const float s = pow2_ceil(fmaxf(a, 6.0f * 1.1754943508222875e-38f) * (1.0f / 6.0f));
                r[j] = e2m1_rt(fminf(fmaxf(r[j] / s, -6.0f), 6.0f)) * s;
            }
        }
    }
    if (kPack) {   // the same roundings, written as their codes and each group's scale (the readers' q * s: r exactly)
        const int g = f.quant == 2 ? 16 : 32;
        uint8_t* pr = f.pdst + (long long) blockIdx.x * f.pdst_stride;
#pragma unroll
        for (int j = 0; j < kFinMax; ++j) {
            if (j >= nj) break;
            const int e = threadIdx.x + j * kThreads;
            float a = e < len ? fabsf(r[j]) : 0.0f;
            for (int o = g / 2; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
            float q = 0.0f, s = 1.0f;
            if (e < len) {
                if (f.quant == 1) {
                    s = pow2_ceil(fmaxf(a, 1e-4f) * (1.0f / 448.0f));
                    q = fp8_rt(fminf(fmaxf(r[j] / s, -448.0f), 448.0f));
                } else if (f.quant == 2) {
                    s = fp8_rt(fmaxf(a, 6.0f * 0.001953125f) / 6.0f);
                    q = e2m1_rt(fminf(fmaxf(r[j] / s, -6.0f), 6.0f));
                } else {
                    s = pow2_ceil(fmaxf(a, 6.0f * 1.1754943508222875e-38f) * (1.0f / 6.0f));
                    q = e2m1_rt(fminf(fmaxf(r[j] / s, -6.0f), 6.0f));
                }
                r[j] = q * s;
            }
            if (f.quant == 1) {
                if (e < len) pr[e] = (uint8_t) e4m3_code(q);
            } else {
                const uint32_t c = e2m1_code(q);
                const uint32_t hi = __shfl_down_sync(0xffffffffu, c, 1);   // (value e + 1: the next lane's)
                if (e < len && (e & 1) == 0) pr[e >> 1] = (uint8_t) (c | (hi << 4));
            }
            if (e < len && e % g == 0)
                pr[(f.quant == 1 ? len : len / 2) + e / g] =
                    (uint8_t) (f.quant == 2 ? e4m3_code(s) : __float_as_uint(s) >> 23);
        }
    }
    uint16_t* d = f.dst != nullptr ? f.dst + (long long) blockIdx.x * f.dst_stride : nullptr;
#pragma unroll
    for (int j = 0; j < kFinMax; ++j) {
        if (j >= nj) break;
        const int e = threadIdx.x + j * kThreads;
        if (e >= len) continue;
        if (d != nullptr) d[e] = f2bf16(r[j]);
        else if (!kPack) v[e] = r[j];
    }
}

// ---------------------------------------------------------------- hyper-connections
__global__ void __launch_bounds__(kThreads) hc_mix_kernel(const float* __restrict__ h, const float* __restrict__ fn,
                                                          int hcdim, float eps, float* __restrict__ mixes) {
    const int m = blockIdx.x, t = blockIdx.y;
    const float* w = fn + (size_t) m * hcdim;
    const float* x = h + (size_t) t * hcdim;
    float acc = 0.0f, ss = 0.0f;
    for (int k = threadIdx.x * 4; k < hcdim; k += kThreads * 4) {
        const float4 hv = *(const float4*) (x + k);
        const float4 wv = *(const float4*) (w + k);
        acc += hv.x * wv.x + hv.y * wv.y + hv.z * wv.z + hv.w * wv.w;
        ss += hv.x * hv.x + hv.y * hv.y + hv.z * hv.z + hv.w * hv.w;
    }
    acc = block_sum(acc);
    ss = block_sum(ss);
    if (threadIdx.x == 0) mixes[(size_t) t * 32 + m] = acc * rsqrtf(ss / (float) hcdim + eps);
}

// a warp a token: lane j * hc + k holds comb[j][k] (hc <= 4: lanes 0..15; wider streams fall back to lane 0 alone).  The body is
// a device function (hc_mix_kernel's last block runs it too); mx is the token's 32 mixes
__device__ __forceinline__ void hc_split_body(const float* mx, int t, int lane, const float* __restrict__ scale,
                                              const float* __restrict__ base, int hc, int iters, float eps, float* pre,
                                              float* post, float* comb) {
    pre += (size_t) t * 8;
    post += (size_t) t * 8;
    comb += (size_t) t * 64;
    if (lane < hc) {
        pre[lane] = 1.0f / (1.0f + expf(-(mx[lane] * scale[0] + base[lane]))) + eps;
        post[lane] = 2.0f / (1.0f + expf(-(mx[hc + lane] * scale[1] + base[hc + lane])));
    }
    if (hc == 4) {
        // rows are the 4-lane groups (j = lane / 4), columns the lanes 4 apart (k = lane % 4)
        const bool on = lane < 16;
        float c = on ? mx[8 + lane] * scale[2] + base[8 + lane] : -FLT_MAX;
        auto row_sum = [&](float v) {
            v += __shfl_xor_sync(0xffffffffu, v, 1);
            return v + __shfl_xor_sync(0xffffffffu, v, 2);
        };
        auto col_sum = [&](float v) {
            v += __shfl_xor_sync(0xffffffffu, v, 4);
            return v + __shfl_xor_sync(0xffffffffu, v, 8);
        };
        float m = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 1));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
        c = on ? expf(c - m) : 0.0f;
        c = c / row_sum(c) + eps;                       // softmax over each row, + eps
        c = on ? c : 0.0f;
        c = c / (col_sum(c) + eps);                     // each destination column over its sources
        for (int it = 1; it < iters; ++it) {
            c = c / (row_sum(c) + eps);
            c = c / (col_sum(c) + eps);
        }
        if (on) comb[lane] = c;
        return;
    }
    if (lane != 0) return;
    float c[8][8];
    for (int j = 0; j < hc; ++j) {
        float m = -FLT_MAX;
        for (int k = 0; k < hc; ++k) {
            c[j][k] = mx[2 * hc + j * hc + k] * scale[2] + base[2 * hc + j * hc + k];
            m = fmaxf(m, c[j][k]);
        }
        float s = 0.0f;
        for (int k = 0; k < hc; ++k) {
            c[j][k] = expf(c[j][k] - m);
            s += c[j][k];
        }
        for (int k = 0; k < hc; ++k) c[j][k] = c[j][k] / s + eps;
    }
    for (int k = 0; k < hc; ++k) {
        float s = 0.0f;
        for (int j = 0; j < hc; ++j) s += c[j][k];
        for (int j = 0; j < hc; ++j) c[j][k] /= s + eps;
    }
    for (int it = 1; it < iters; ++it) {
        for (int j = 0; j < hc; ++j) {
            float s = 0.0f;
            for (int k = 0; k < hc; ++k) s += c[j][k];
            for (int k = 0; k < hc; ++k) c[j][k] /= s + eps;
        }
        for (int k = 0; k < hc; ++k) {
            float s = 0.0f;
            for (int j = 0; j < hc; ++j) s += c[j][k];
            for (int j = 0; j < hc; ++j) c[j][k] /= s + eps;
        }
    }
    for (int j = 0; j < hc; ++j)
        for (int k = 0; k < hc; ++k) comb[j * hc + k] = c[j][k];
}
__global__ void hc_split_kernel(const float* __restrict__ mixes, const float* __restrict__ scale,
                                const float* __restrict__ base, int hc, int iters, float eps, float* pre, float* post,
                                float* comb) {
    const int t = blockIdx.x;
    hc_split_body(mixes + (size_t) t * 32, t, threadIdx.x, scale, base, hc, iters, eps, pre, post, comb);
}

// hc_mix_kernel with hc_split_kernel behind it: the block that finishes the token's last mix (an atomic count, reset by it) runs
// the split on warp 0 - the same arithmetic as the two launches, one launch less and no round trip through the launch gap
__global__ void __launch_bounds__(kThreads) hc_mix_split_kernel(const float* __restrict__ h, const float* __restrict__ fn,
                                                                int hcdim, float eps, float* mixes, unsigned* cnt,
                                                                const float* __restrict__ scale,
                                                                const float* __restrict__ base, int hc, int iters,
                                                                float hc_eps, float* pre, float* post, float* comb) {
    const int m = blockIdx.x, t = blockIdx.y;
    const float* w = fn + (size_t) m * hcdim;
    const float* x = h + (size_t) t * hcdim;
    float acc = 0.0f, ss = 0.0f;
    for (int k = threadIdx.x * 4; k < hcdim; k += kThreads * 4) {
        const float4 hv = *(const float4*) (x + k);
        const float4 wv = *(const float4*) (w + k);
        acc += hv.x * wv.x + hv.y * wv.y + hv.z * wv.z + hv.w * wv.w;
        ss += hv.x * hv.x + hv.y * hv.y + hv.z * hv.z + hv.w * hv.w;
    }
    acc = block_sum(acc);
    ss = block_sum(ss);
    __shared__ unsigned last;
    __shared__ float smx[32];
    if (threadIdx.x == 0) {
        mixes[(size_t) t * 32 + m] = acc * rsqrtf(ss / (float) hcdim + eps);
        __threadfence();
        last = atomicAdd(cnt + t, 1u) == gridDim.x - 1 ? 1u : 0u;
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (threadIdx.x < 32) {
        const int n = gridDim.x;   // the mixes of the token (n_mix <= 32); read past the L1
        smx[threadIdx.x] = threadIdx.x < n ? *(const volatile float*) (mixes + (size_t) t * 32 + threadIdx.x) : 0.0f;
        __syncwarp();
        hc_split_body(smx, t, threadIdx.x, scale, base, hc, iters, hc_eps, pre, post, comb);
    }
    if (threadIdx.x == 0) cnt[t] = 0u;
}

__global__ void __launch_bounds__(1024) hc_pre_norm_kernel(const float* __restrict__ h, const float* __restrict__ mix,
                                                           const float* __restrict__ w, int dim, int hc, float eps,
                                                           float* __restrict__ x) {
    const int t = blockIdx.x;
    h += (size_t) t * hc * dim;
    mix += (size_t) t * 8;
    x += (size_t) t * dim;
    float r[8];
    float ss = 0.0f;
    int n = 0;
    for (int e = threadIdx.x; e < dim; e += blockDim.x, ++n) {
        float a = 0.0f;
        for (int s = 0; s < hc; ++s) a += mix[s] * h[(size_t) s * dim + e];
        r[n] = a;
        ss += a * a;
    }
    const float inv = rsqrtf(block_sum(ss) / (float) dim + eps);
    n = 0;
    for (int e = threadIdx.x; e < dim; e += blockDim.x, ++n) x[e] = w != nullptr ? r[n] * inv * w[e] : r[n];
}

__global__ void hc_post_kernel(const float* __restrict__ y, const float* __restrict__ res,
                               const float* __restrict__ post, const float* __restrict__ comb, int dim, int hc,
                               float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= dim * hc) return;
    const int k = i / dim, e = i - k * dim;
    const float* r = res + (size_t) t * hc * dim;
    float a = post[(size_t) t * 8 + k] * y[(size_t) t * dim + e];
    for (int j = 0; j < hc; ++j) a += comb[(size_t) t * 64 + j * hc + k] * r[(size_t) j * dim + e];
    out[(size_t) t * hc * dim + i] = a;
}

// ---------------------------------------------------------------- gemv over F16 weights, a warp a row
__global__ void __launch_bounds__(kThreads) gemv_f16_kernel(const uint16_t* __restrict__ W, const float* __restrict__ x,
                                                            int n, int k, float alpha, float* __restrict__ y) {
    const int row = blockIdx.x * (kThreads / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31, t = blockIdx.y;
    if (row >= n) return;
    const uint16_t* w = W + (size_t) row * k;
    const float* xv = x + (size_t) t * k;
    float acc = 0.0f;
    for (int c = lane * 8; c < k; c += 32 * 8) {
        const uint4 q = *(const uint4*) (w + c);
        const __half2* hp = (const __half2*) &q;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __half22float2(hp[i]);
            acc += f.x * xv[c + 2 * i] + f.y * xv[c + 2 * i + 1];
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) y[(size_t) t * n + row] = alpha * acc;
}
// ... for T activations (rows ldx apart, 16-byte aligned; outputs ldy apart), kRowsT of them a block (grid y): each
// weight row read once for the block's activations, theirs in float4s - each activation's sum gemv_f16_kernel's (the
// same products in the same order), so its bits too.  (A block for all of them, 16 at most, was slower than a launch
// a token: its accumulators and guards kept the loop from unrolling - 66 against 29 us at 5 tokens on an RTX 5060 Ti,
// this 9)
constexpr int kRowsT = 2;
__global__ void __launch_bounds__(kThreads) gemv_f16_rows_kernel(const uint16_t* __restrict__ W,
                                                                 const float* __restrict__ x, size_t ldx, int n, int k,
                                                                 int T, float alpha, float* __restrict__ y, size_t ldy) {
    const int row = blockIdx.x * (kThreads / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int t0 = blockIdx.y * kRowsT, nt = min(kRowsT, T - t0);
    if (row >= n) return;
    const uint16_t* w = W + (size_t) row * k;
    float acc[kRowsT];
#pragma unroll
    for (int t = 0; t < kRowsT; ++t) acc[t] = 0.0f;
    for (int c = lane * 8; c < k; c += 32 * 8) {
        const uint4 q = *(const uint4*) (w + c);
        const __half2* hp = (const __half2*) &q;
        float xv[kRowsT][8];
#pragma unroll
        for (int t = 0; t < kRowsT; ++t)
            if (t < nt) {
                const float* xp = x + (size_t) (t0 + t) * ldx + c;
                const float4 a = *(const float4*) xp, b = *(const float4*) (xp + 4);
                xv[t][0] = a.x, xv[t][1] = a.y, xv[t][2] = a.z, xv[t][3] = a.w;
                xv[t][4] = b.x, xv[t][5] = b.y, xv[t][6] = b.z, xv[t][7] = b.w;
            }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __half22float2(hp[i]);
#pragma unroll
            for (int t = 0; t < kRowsT; ++t)
                if (t < nt) acc[t] += f.x * xv[t][2 * i] + f.y * xv[t][2 * i + 1];
        }
    }
#pragma unroll
    for (int t = 0; t < kRowsT; ++t)
        if (t < nt) {
            const float a = warp_sum(acc[t]);
            if (lane == 0) y[(size_t) (t0 + t) * ldy + row] = alpha * a;
        }
}

// ---------------------------------------------------------------- gemv over F32 weights, a warp a row (the router)
__global__ void __launch_bounds__(kThreads) gemv_f32_kernel(const float* __restrict__ W, const float* __restrict__ x,
                                                            int n, int k, float* __restrict__ y, int ldy) {
    const int row = blockIdx.x * (kThreads / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31, t = blockIdx.y;
    if (row >= n) return;
    const float4* w = (const float4*) (W + (size_t) row * k);
    const float4* xv = (const float4*) (x + (size_t) t * k);
    float acc = 0.0f;
    for (int c = lane; c < k / 4; c += 32) {
        const float4 a = w[c], b = xv[c];
        acc = fmaf(a.x, b.x, acc);
        acc = fmaf(a.y, b.y, acc);
        acc = fmaf(a.z, b.z, acc);
        acc = fmaf(a.w, b.w, acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) y[(size_t) t * ldy + row] = acc;
}

// ---------------------------------------------------------------- the router
__global__ void __launch_bounds__(512) route_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                                                    int n, int k, float route_scale, int* idx, float* w,
                                                    const float* __restrict__ bias_vl, const VlBits vl) {
    __shared__ float sc[1024], bs[1024];
    __shared__ float bv[32];
    __shared__ int bi[32];
    __shared__ int pick[16];
    const int t = blockIdx.x;
    logits += (size_t) t * 1024;
    if (bias_vl != nullptr && ((vl.w[t >> 5] >> (t & 31)) & 1u)) bias = bias_vl;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float l = logits[i];
        const float sp = l > 20.0f ? l : log1pf(expf(l));
        sc[i] = sqrtf(sp);
        bs[i] = sc[i] + bias[i];
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    for (int r = 0; r < k; ++r) {
        float best = -FLT_MAX;
        int bix = 0x7fffffff;
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            if (bs[i] > best || (bs[i] == best && i < bix)) best = bs[i], bix = i;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bix, o);
            if (ov > best || (ov == best && oi < bix)) best = ov, bix = oi;
        }
        if (lane == 0) bv[wp] = best, bi[wp] = bix;
        __syncthreads();
        if (threadIdx.x == 0) {
            float b = -FLT_MAX;
            int ix = 0x7fffffff;
            for (int q = 0; q < (int) (blockDim.x >> 5); ++q)
                if (bv[q] > b || (bv[q] == b && bi[q] < ix)) b = bv[q], ix = bi[q];
            pick[r] = ix;
            bs[ix] = -FLT_MAX;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int r = 0; r < k; ++r) s += sc[pick[r]];
        for (int r = 0; r < k; ++r) {
            idx[(size_t) t * kPicks + r] = pick[r];
            w[(size_t) t * kPicks + r] = sc[pick[r]] / (s + 1e-20f) * route_scale;
        }
    }
}

__global__ void swiglu_kernel(const float* __restrict__ g, const float* __restrict__ u, int n, float lim,
                              float* __restrict__ h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gv = g[i], uv = u[i];
    if (lim > 0.0f) {
        gv = fminf(gv, lim);
        uv = fminf(fmaxf(uv, -lim), lim);
    }
    h[i] = gv / (1.0f + expf(-gv)) * uv;
}

// ---------------------------------------------------------------- the compressor (ratio > 1), a window in order
__global__ void __launch_bounds__(512) compress_kernel(const float* __restrict__ kv, const float* __restrict__ score,
                                                       float* kv_state, float* score_state, int ratio, int p0, int T,
                                                       int d, const float* __restrict__ w, float eps,
                                                       float* __restrict__ latent) {
    int g = 0;
    for (int t = 0; t < T; ++t) {
        const int slot = (p0 + t) % ratio;
        const bool pool = (p0 + t + 1) % ratio == 0;
        for (int e = threadIdx.x; e < d; e += blockDim.x) {
            kv_state[(size_t) slot * d + e] = kv[(size_t) t * d + e];
            score_state[(size_t) slot * d + e] = score[(size_t) t * d + e];
        }
        __syncthreads();
        if (!pool) continue;
        // (kv * softmax(score, over the slots)).sum: each slot's probability first, then the sum, as torch does
        float val[2] = {0.0f, 0.0f};   // d <= 2 * blockDim
        float ss = 0.0f;
        int n = 0;
        for (int e = threadIdx.x; e < d; e += blockDim.x, ++n) {
            float m = -FLT_MAX;
            for (int s = 0; s < ratio; ++s) m = fmaxf(m, score_state[(size_t) s * d + e]);
            float den = 0.0f;
            for (int s = 0; s < ratio; ++s) den += expf(score_state[(size_t) s * d + e] - m);
            float acc = 0.0f;
            for (int s = 0; s < ratio; ++s)
                acc += kv_state[(size_t) s * d + e] * (expf(score_state[(size_t) s * d + e] - m) / den);
            val[n] = acc;
            ss += acc * acc;
        }
        const float inv = rsqrtf(block_sum(ss) / (float) d + eps);
        n = 0;
        for (int e = threadIdx.x; e < d; e += blockDim.x, ++n) latent[(size_t) g * d + e] = val[n] * inv * w[e];
        ++g;
        __syncthreads();
    }
}

// ---------------------------------------------------------------- per-query sizes
__device__ __forceinline__ void rows_at(const Rows& r, int t, int& n, int& k) {
    if (r.gen == 0) {
        n = r.n[t];
        k = r.k[t];
        return;
    }
    const int nn = (int) ((r.p0 + t + 1) / r.R);
    const int cb = min(r.ccap, (nn + r.blk - 1) / r.blk);
    switch (r.gen) {
    case 1: n = nn, k = min(r.kcap, nn); break;
    case 2: n = nn, k = cb; break;
    case 3: n = cb * r.blk, k = min(r.kcap, nn); break;
    default: n = (nn + r.blk - 1) / r.blk, k = r.kcap; break;
    }
}

// ---------------------------------------------------------------- the indexer
// a thread a position: its key row (KF: kKvFp4p packed, or BF16) against every head's query (shared memory), relu,
// weighted sum
template <int HD, int KF>
__global__ void __launch_bounds__(kThreads) index_score_kernel(const float* __restrict__ q, const float* __restrict__ w,
                                                               const uint8_t* __restrict__ kc, int heads, int n,
                                                               const int* __restrict__ cand, int cand_stride, int blk,
                                                               Rows lim, float* __restrict__ out,
                                                               long long out_stride) {
    extern __shared__ float qs[];   // heads * HD, then heads weights
    const int t = blockIdx.y;
    for (int i = threadIdx.x; i < heads * HD; i += blockDim.x) qs[i] = q[(size_t) t * heads * HD + i];
    for (int i = threadIdx.x; i < heads; i += blockDim.x) qs[heads * HD + i] = w[(size_t) t * 64 + i];
    __syncthreads();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float* o = out + (long long) t * out_stride;
    int ln, lk;
    rows_at(lim, t, ln, lk);
    int p = i;
    if (cand != nullptr) {
        if (i >= lk * blk) {
            o[i] = -INFINITY;
            return;
        }
        p = cand[(size_t) t * cand_stride + i / blk] * blk + i % blk;
    }
    if (p >= ln) {
        o[i] = -INFINITY;
        return;
    }
    const uint8_t* kr = kc + (size_t) p * kv_row_bytes(KF, HD);
    float acc[32];
#pragma unroll
    for (int h = 0; h < 32; ++h) acc[h] = 0.0f;
#pragma unroll 2
    for (int c = 0; c < HD; c += 8) {
        float kv[8];
        if (KF == kKvFp4p) {   // 8 codes in 4 bytes, the group's power-of-two scale: the values the finish rounded to
            const uint32_t cw = *(const uint32_t*) (kr + c / 2);
            const float ks = ue8m0_val(kr[HD / 2 + c / 32]);
#pragma unroll
            for (int x = 0; x < 8; ++x) kv[x] = e2m1_val14(cw >> (4 * x)) * 16384.0f * ks;   // (any 2^k: exact)
        } else {
            const uint4 raw = *(const uint4*) (kr + c * 2);
            const uint16_t* b = (const uint16_t*) &raw;
#pragma unroll
            for (int x = 0; x < 8; ++x) kv[x] = bf16f(b[x]);
        }
#pragma unroll
        for (int h = 0; h < 32; ++h) {
            if (h >= heads) break;
            const float* qh = qs + h * HD + c;
#pragma unroll
            for (int x = 0; x < 8; ++x) acc[h] += qh[x] * kv[x];
        }
    }
    float s = 0.0f;
#pragma unroll
    for (int h = 0; h < 32; ++h) {
        if (h >= heads) break;
        s += fmaxf(acc[h], 0.0f) * qs[heads * HD + h];
    }
    o[i] = s;
}

__device__ __forceinline__ uint32_t ord_key(float f) {
    const uint32_t b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// a block a row: the k-th largest key by four 8-bit radix passes, then everything above it and enough of its ties
__global__ void __launch_bounds__(1024) topk_kernel(const float* __restrict__ vals, long long stride, Rows r,
                                                    const int* __restrict__ cand, int cand_stride, int blk,
                                                    int* __restrict__ out, int out_stride) {
    __shared__ unsigned hist[256];
    __shared__ uint32_t prefix_s, mask_s;
    __shared__ int remaining_s;
    __shared__ int cnt_gt, cnt_eq;
    const int t = blockIdx.x;
    int n, k;
    rows_at(r, t, n, k);
    vals += (long long) t * stride;
    out += (size_t) t * out_stride;
    const int* cd = cand != nullptr ? cand + (size_t) t * cand_stride : nullptr;
    if (n <= 0) return;
    if (k >= n) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = cd != nullptr ? cd[i / blk] * blk + i % blk : i;
        return;
    }
    if (threadIdx.x == 0) {
        prefix_s = 0u;
        mask_s = 0u;
        remaining_s = k;
        cnt_gt = 0;
        cnt_eq = 0;
    }
    __syncthreads();
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0u;
        __syncthreads();
        const uint32_t prefix = prefix_s, mask = mask_s;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const uint32_t u = ord_key(vals[i]);
            if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & 0xffu], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            const int rem = remaining_s;
            int cum = 0;
            for (int d = 255; d >= 0; --d) {
                const int h = (int) hist[d];
                if (cum + h >= rem) {
                    prefix_s = prefix | ((uint32_t) d << shift);
                    mask_s = mask | (0xffu << shift);
                    remaining_s = rem - cum;
                    break;
                }
                cum += h;
            }
        }
        __syncthreads();
    }
    const uint32_t thr = prefix_s;
    const int remaining = remaining_s;
    const int n_gt = k - remaining;
    // the picks in index order (a scan a pass of 1024 values), so the attention sums them in the same order every run
    // and the ties kept are the lowest positions
    __shared__ int wgt[32], weq[32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const unsigned below = (1u << lane) - 1u;
    for (int i0 = 0; i0 < n; i0 += blockDim.x) {
        const int i = i0 + threadIdx.x;
        const uint32_t u = i < n ? ord_key(vals[i]) : 0u;
        const bool gt = i < n && u > thr, eq = i < n && u == thr;
        const unsigned bg = __ballot_sync(0xffffffffu, gt), be = __ballot_sync(0xffffffffu, eq);
        if (lane == 0) {
            wgt[wid] = __popc(bg);
            weq[wid] = __popc(be);
        }
        __syncthreads();
        if (wid == 0) {
            const int g0 = lane < nw ? wgt[lane] : 0, e0 = lane < nw ? weq[lane] : 0;
            int gs = g0, es = e0;
            for (int o = 1; o < 32; o <<= 1) {
                const int a = __shfl_up_sync(0xffffffffu, gs, o), b = __shfl_up_sync(0xffffffffu, es, o);
                if (lane >= o) {
                    gs += a;
                    es += b;
                }
            }
            if (lane < nw) {
                wgt[lane] = cnt_gt + gs - g0;
                weq[lane] = cnt_eq + es - e0;
            }
            __syncwarp();
            if (lane == 31) {
                cnt_gt += gs;
                cnt_eq += es;
            }
        }
        __syncthreads();
        int slot = -1;
        if (gt) slot = wgt[wid] + __popc(bg & below);
        else if (eq) {
            const int e = weq[wid] + __popc(be & below);
            if (e < remaining) slot = n_gt + e;
        }
        if (slot >= 0) out[slot] = cd != nullptr ? cd[i / blk] * blk + i % blk : i;
        __syncthreads();
    }
}

__global__ void block_max_kernel(const float* __restrict__ score, long long stride, Rows r, int blk,
                                 float* __restrict__ bm, long long bm_stride) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    int n, k_;
    rows_at(r, t, n, k_);
    const int nb = (n + blk - 1) / blk;
    if (b >= nb) return;
    float m = -INFINITY;
    for (int j = 0; j < blk; ++j) {
        const int p = b * blk + j;
        if (p < n) m = fmaxf(m, score[(long long) t * stride + p]);
    }
    if (b == (n - 1) / blk) m = INFINITY;   // the newest position's block is pinned in
    bm[(long long) t * bm_stride + b] = m;
}

// ---------------------------------------------------------------- sparse attention, a block a (head, query)
constexpr int kAttTile = 64;

// kContig: query t's selected rows are ckv's rows t * sel_stride .. (the scratch of rows decoded once a query) - read
// without sel, a warp's rows of a tile loaded kPre at a time before their dots (0: as the BF16 kernel); the
// arithmetic the same
template <bool kContig, int kPre = 0>
__global__ void __launch_bounds__(kThreads) sparse_attn_kernel(const float* __restrict__ q,
                                                               const uint16_t* __restrict__ win,
                                                               const uint16_t* __restrict__ ckv,
                                                               const int* __restrict__ sel, int sel_stride,
                                                               AttnRows ar, const float* __restrict__ sink, int heads,
                                                               int d, float scale, float* __restrict__ out) {
    // d == 512: a thread owns 2 output values, a lane 16 of a row's dot
    const int h = blockIdx.x, t = blockIdx.y;
    int n_win, n_sel, w0;
    if (ar.gen) {
        const long long ta = ar.t0 + t, pa = ar.p0 + ta;
        const long long first = max((long long) ar.win - ar.p0, ta + 1);   // the window buffer row of max(0, pa - win + 1)
        w0 = (int) first;
        n_win = (int) (ar.win + ta - first + 1);
        n_sel = ar.R > 0 ? (int) min((long long) ar.kcap, (pa + 1) / ar.R) : 0;
    } else {
        n_win = ar.nwin[t], n_sel = ar.nsel[t], w0 = ar.win0[t];
    }
    const uint16_t* wbase = win + (size_t) w0 * d;
    const int* sl = sel + (size_t) t * sel_stride;
    __shared__ float qs[512];
    __shared__ float sc[kAttTile];
    __shared__ const uint16_t* rows[kAttTile];
    __shared__ float m_run_s, tile_scale_s, den_s;
    for (int i = threadIdx.x; i < d; i += blockDim.x) qs[i] = q[((size_t) t * heads + h) * d + i];
    if (threadIdx.x == 0) {
        m_run_s = -1e30f;
        den_s = 0.0f;
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    float o0 = 0.0f, o1 = 0.0f;
    const int total = n_win + n_sel;
    for (int t0 = 0; t0 < total; t0 += kAttTile) {
        const int nt = min(kAttTile, total - t0);
        if (threadIdx.x < nt) {
            const int j = t0 + threadIdx.x;
            rows[threadIdx.x] = j < n_win ? wbase + (size_t) j * d
                                          : ckv + (kContig ? (size_t) t * sel_stride + (j - n_win) : (size_t) sl[j - n_win]) * d;
        }
        __syncthreads();
        if (kContig && kPre > 0) {   // the warp's rows of the tile (wp, wp + 8, ..), kPre loads at a time, then dots
#pragma unroll
            for (int k0 = 0; k0 < kAttTile / 8; k0 += (kPre > 0 ? kPre : 1)) {
                uint4 ra[kPre > 0 ? kPre : 1], rb[kPre > 0 ? kPre : 1];
#pragma unroll
                for (int k = 0; k < kPre; ++k) {
                    const int jj = wp + 8 * (k0 + k);
                    if (jj < nt) {
                        ra[k] = *(const uint4*) (rows[jj] + lane * 16);
                        rb[k] = *(const uint4*) (rows[jj] + lane * 16 + 8);
                    }
                }
#pragma unroll
                for (int k = 0; k < kPre; ++k) {
                    const int jj = wp + 8 * (k0 + k);
                    if (jj < nt) {
                        const uint16_t* pa = (const uint16_t*) &ra[k];
                        const uint16_t* pb = (const uint16_t*) &rb[k];
                        float s = 0.0f;
#pragma unroll
                        for (int x = 0; x < 8; ++x) s += qs[lane * 16 + x] * bf16f(pa[x]);
#pragma unroll
                        for (int x = 0; x < 8; ++x) s += qs[lane * 16 + 8 + x] * bf16f(pb[x]);
                        s = warp_sum(s);
                        if (lane == 0) sc[jj] = s * scale;
                    }
                }
            }
        } else
            for (int jj = wp; jj < nt; jj += kThreads / 32) {
                const uint16_t* rr = rows[jj];
                const uint4 a = *(const uint4*) (rr + lane * 16);
                const uint4 b = *(const uint4*) (rr + lane * 16 + 8);
                const uint16_t* pa = (const uint16_t*) &a;
                const uint16_t* pb = (const uint16_t*) &b;
                float s = 0.0f;
#pragma unroll
                for (int x = 0; x < 8; ++x) s += qs[lane * 16 + x] * bf16f(pa[x]);
#pragma unroll
                for (int x = 0; x < 8; ++x) s += qs[lane * 16 + 8 + x] * bf16f(pb[x]);
                s = warp_sum(s);
                if (lane == 0) sc[jj] = s * scale;
            }
        __syncthreads();
        if (wp == 0) {
            float m = -INFINITY;
            for (int j = lane; j < nt; j += 32) m = fmaxf(m, sc[j]);
            m = warp_max(m);
            const float m_old = m_run_s;
            const float m_new = fmaxf(m_old, m);
            float s = 0.0f;
            for (int j = lane; j < nt; j += 32) {
                const float p = expf(sc[j] - m_new);
                sc[j] = p;
                s += p;
            }
            s = warp_sum(s);
            __syncwarp();
            if (lane == 0) {
                tile_scale_s = expf(m_old - m_new);
                den_s = den_s * tile_scale_s + s;
                m_run_s = m_new;
            }
        }
        __syncthreads();
        const float cs = tile_scale_s;
        o0 *= cs;
        o1 *= cs;
        const int c = threadIdx.x * 2;
        if (kContig && kPre > 0) {   // (eight rows' loads at once)
#pragma unroll 8
            for (int j = 0; j < nt; ++j) {
                const uint32_t pr = *(const uint32_t*) (rows[j] + c);
                o0 += sc[j] * bf16f((uint16_t) (pr & 0xffffu));
                o1 += sc[j] * bf16f((uint16_t) (pr >> 16));
            }
        } else
            for (int j = 0; j < nt; ++j) {
                const uint32_t pr = *(const uint32_t*) (rows[j] + c);
                o0 += sc[j] * bf16f((uint16_t) (pr & 0xffffu));
                o1 += sc[j] * bf16f((uint16_t) (pr >> 16));
            }
        __syncthreads();
    }
    const float den = den_s + expf(sink[h] - m_run_s);
    const int c = threadIdx.x * 2;
    float* ov = out + ((size_t) t * heads + h) * d;
    ov[c] = o0 / den;
    ov[c + 1] = o1 / den;
}

// ---------------------------------------------------------------- sparse attention over the packed compressed rows
// G heads of one query a block (8 / G warps a head): each pass stages 16 of a tile's rows in shared memory as BF16
// (the window's rows as they are; the packed rows decoded - exact in BF16), once for all G heads, through one 16-byte
// load a thread.  Each head's arithmetic is sparse_attn_kernel's, operation for operation: its tiles of 64 rows, its
// dots by a warp's lanes (16 values each, in order) and the same butterfly, its softmax by one warp the same way, each
// output value's weighted sum by the same thread in row order - the same bits for any G.
constexpr int kStage = 16;   // rows a pass stages (a thread: 32 values of one)

// values part * 32 .. + 32 of a packed compressed row (kKvFp4, 512 values) as BF16 into dst[0 .. 4) (exact: the
// values are E2M1 times an E4M3 scale)
__device__ __forceinline__ void unpack32(const uint8_t* rr, int part, uint4* dst) {
    const uint4 cw = *(const uint4*) (rr + part * 16);   // 32 codes, two scale groups
    const float k0 = e4m3_value(rr[256 + part * 2]) * 16384.0f, k1 = e4m3_value(rr[256 + part * 2 + 1]) * 16384.0f;
    const uint32_t wv[4] = {cw.x, cw.y, cw.z, cw.w};
    uint32_t o[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {   // values 2i, 2i + 1 (the low and the high nibble of byte i)
        const uint32_t b = wv[i >> 2] >> (8 * (i & 3));
        const float ks = i < 8 ? k0 : k1;
        const float lo = e2m1_val14(b) * ks, hi = e2m1_val14(b >> 4) * ks;   // (exact: their BF16 bits)
        o[i] = (__float_as_uint(lo) >> 16) | (__float_as_uint(hi) & 0xffff0000u);
    }
    dst[0] = make_uint4(o[0], o[1], o[2], o[3]);
    dst[1] = make_uint4(o[4], o[5], o[6], o[7]);
    dst[2] = make_uint4(o[8], o[9], o[10], o[11]);
    dst[3] = make_uint4(o[12], o[13], o[14], o[15]);
}

// a few queries' attention (a decode step, a short window): the rows each query selects decoded once to BF16 rows -
// out[t * cap + i] = ckv[sel[t][i]] for i < the query's n_sel - for the BF16 kernel reading them in place (once a
// query, not once a head)
__global__ void __launch_bounds__(kThreads) unpack_sel_kernel(const uint8_t* __restrict__ ckv,
                                                              const int* __restrict__ sel, int sel_stride, AttnRows ar,
                                                              int cap, uint16_t* __restrict__ out) {
    const int t = blockIdx.y, i = blockIdx.x * kStage + (int) (threadIdx.x >> 4), part = threadIdx.x & 15;
    int n_sel;
    if (ar.gen) {
        const long long pa = ar.p0 + ar.t0 + t;
        n_sel = ar.R > 0 ? (int) min((long long) ar.kcap, (pa + 1) / ar.R) : 0;
    } else {
        n_sel = ar.nsel[t];
    }
    if (i >= n_sel) return;
    const size_t r = (size_t) t * cap + i;
    unpack32(ckv + (size_t) sel[(size_t) t * sel_stride + i] * kv_row_bytes(kKvFp4, 512), part, (uint4*) (out + r * 512 + part * 32));
}

template <int G>
__global__ void __launch_bounds__(kThreads) sparse_attn_packed_kernel(const float* __restrict__ q,
                                                                      const uint16_t* __restrict__ win,
                                                                      const uint8_t* __restrict__ ckv,
                                                                      const int* __restrict__ sel, int sel_stride,
                                                                      AttnRows ar, const float* __restrict__ sink,
                                                                      int heads, float scale, float* __restrict__ out) {
    constexpr int d = 512, W = 8 / G;   // (kThreads / 32 warps)
    const int hb = blockIdx.x * G, t = blockIdx.y;
    int n_win, n_sel, w0;
    if (ar.gen) {
        const long long ta = ar.t0 + t, pa = ar.p0 + ta;
        const long long first = max((long long) ar.win - ar.p0, ta + 1);
        w0 = (int) first;
        n_win = (int) (ar.win + ta - first + 1);
        n_sel = ar.R > 0 ? (int) min((long long) ar.kcap, (pa + 1) / ar.R) : 0;
    } else {
        n_win = ar.nwin[t], n_sel = ar.nsel[t], w0 = ar.win0[t];
    }
    const uint16_t* wbase = win + (size_t) w0 * d;
    const int* sl = sel + (size_t) t * sel_stride;
    const size_t crow = kv_row_bytes(kKvFp4, d);
    __shared__ __align__(16) uint16_t tile[kStage * d];
    __shared__ float sc[G][kAttTile];
    __shared__ const uint8_t* rows[kAttTile];
    __shared__ float m_run_s[G], tile_scale_s[G], den_s[G];
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5, gw = wp / W, wr = wp % W;
    float qr[16];   // this lane's 16 values of its warp's head
#pragma unroll
    for (int x = 0; x < 16; ++x) qr[x] = q[((size_t) t * heads + hb + gw) * d + lane * 16 + x];
    if (threadIdx.x < G) {
        m_run_s[threadIdx.x] = -1e30f;
        den_s[threadIdx.x] = 0.0f;
    }
    float o0[G], o1[G];
#pragma unroll
    for (int g = 0; g < G; ++g) o0[g] = o1[g] = 0.0f;
    const int c = threadIdx.x * 2;
    const int sr = threadIdx.x >> 4, part = threadIdx.x & 15;   // the staging: a thread's row and 32 values
    // rows [s0, s0 + n) of the tile at t0 into tile[] as BF16
    auto stage = [&](int t0, int s0, int n) {
        if (sr >= n) return;
        const int j = s0 + sr;
        uint4* dst = (uint4*) (tile + sr * d + part * 32);
        if (t0 + j < n_win) {   // a window row: BF16 already
            const uint4* src = (const uint4*) (rows[j] + part * 64);
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
            dst[3] = src[3];
            return;
        }
        unpack32(rows[j], part, dst);
    };
    const int total = n_win + n_sel;
    for (int t0 = 0; t0 < total; t0 += kAttTile) {
        const int nt = min(kAttTile, total - t0);
        if (threadIdx.x < nt) {
            const int j = t0 + threadIdx.x;
            rows[threadIdx.x] = j < n_win ? (const uint8_t*) (wbase + (size_t) j * d) : ckv + (size_t) sl[j - n_win] * crow;
        }
        __syncthreads();
        for (int s0 = 0; s0 < nt; s0 += kStage) {   // the dots
            const int n = min(kStage, nt - s0);
            stage(t0, s0, n);
            __syncthreads();
#if !defined(STRATA_USE_HIP)   // (HIP leaves a loop with shuffles and a remainder rolled anyway)
#pragma unroll 2
#endif
            for (int r = wr; r < n; r += W) {   // (two rows' chains at once)
                const uint16_t* vr = tile + r * d + lane * 16;
                const uint4 a = *(const uint4*) vr;
                const uint4 b = *(const uint4*) (vr + 8);
                const uint16_t* pa = (const uint16_t*) &a;
                const uint16_t* pb = (const uint16_t*) &b;
                float s = 0.0f;
#pragma unroll
                for (int x = 0; x < 8; ++x) s += qr[x] * bf16f(pa[x]);
#pragma unroll
                for (int x = 0; x < 8; ++x) s += qr[8 + x] * bf16f(pb[x]);
                s = warp_sum(s);
                if (lane == 0) sc[gw][s0 + r] = s * scale;
            }
            __syncthreads();
        }
        if (wp < G) {   // head wp's softmax over the tile
            float* scg = sc[wp];
            float m = -INFINITY;
            for (int j = lane; j < nt; j += 32) m = fmaxf(m, scg[j]);
            m = warp_max(m);
            const float m_old = m_run_s[wp];
            const float m_new = fmaxf(m_old, m);
            float s = 0.0f;
            for (int j = lane; j < nt; j += 32) {
                const float p = expf(scg[j] - m_new);
                scg[j] = p;
                s += p;
            }
            s = warp_sum(s);
            __syncwarp();
            if (lane == 0) {
                tile_scale_s[wp] = expf(m_old - m_new);
                den_s[wp] = den_s[wp] * tile_scale_s[wp] + s;
                m_run_s[wp] = m_new;
            }
        }
        __syncthreads();
#pragma unroll
        for (int g = 0; g < G; ++g) {
            const float cs = tile_scale_s[g];
            o0[g] *= cs;
            o1[g] *= cs;
        }
        for (int s0 = 0; s0 < nt; s0 += kStage) {   // the weighted sums
            const int n = min(kStage, nt - s0);
            stage(t0, s0, n);
            __syncthreads();
            for (int r = 0; r < n; ++r) {
                const uint32_t pr = *(const uint32_t*) (tile + r * d + c);
                const float v0 = bf16f((uint16_t) (pr & 0xffffu)), v1 = bf16f((uint16_t) (pr >> 16));
#pragma unroll
                for (int g = 0; g < G; ++g) {
                    o0[g] += sc[g][s0 + r] * v0;
                    o1[g] += sc[g][s0 + r] * v1;
                }
            }
            __syncthreads();
        }
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const float den = den_s[g] + expf(sink[hb + g] - m_run_s[g]);
        float* ov = out + ((size_t) t * heads + hb + g) * d;
        ov[c] = o0[g] / den;
        ov[c + 1] = o1[g] / den;
    }
}

// DS4_FAST_ATTN=1: the chunked prompt's packed-KV attention on the matrix cores (gfx11/gfx12 WMMA; the R9700's gfx12).
// The matrix's rows are 16 heads of ONE query, so a block stages a 16-key tile once for all of them: Q (16 heads x 512
// dims) is the A operand of the scores' product, the key tile is B, and the same tile is B of the values' product (the
// heads' softmaxes and their 512-dim outputs are its rows).  The QK and PV sums are the matrix cores' order, not
// sparse_attn_packed_kernel's scalar lanes' - hence opt-in (DS4_FAST_ATTN).
// (HIP only: the CUDA build keeps the scalar kernels, so this and the dispatch below compile out there.)
#if defined(STRATA_USE_HIP)
constexpr int kFaHeads = 16;   // heads a tile (the matrix's rows)
constexpr int kFaKeys = 16;    // keys a tile
constexpr int kFaD = 512;      // the head dim
constexpr int kFaLd = 520;     // halves a tile row (512 + 8: the pad keeps the 16 rows off one bank)
__global__ void __launch_bounds__(kThreads) sparse_attn_wmma_kernel(const float* __restrict__ q,
                                                                    const uint16_t* __restrict__ win,
                                                                    const uint8_t* __restrict__ ckv,
                                                                    const int* __restrict__ sel, int sel_stride,
                                                                    AttnRows ar, const float* __restrict__ sink,
                                                                    int heads, float scale, float* __restrict__ out) {
    __shared__ __align__(16) uint16_t qs[kFaHeads * kFaLd];
    __shared__ __align__(16) uint16_t ks[kFaKeys * kFaLd];
    __shared__ float sc[kFaHeads][kFaKeys];
    __shared__ float part[8][kFaHeads][kFaKeys];   // (the 8 warps' 64-dim slices of the scores)
    __shared__ uint16_t ps[kFaHeads][kFaKeys];
    __shared__ const uint8_t* rows[kFaKeys];
    __shared__ float m_run[kFaHeads], den[kFaHeads], ts[kFaHeads];
    const int tid = (int) threadIdx.x, wp = tid >> 5, lane = tid & 31, l16 = lane & 15, hi = lane >> 4;
    const int t = blockIdx.y, hb = blockIdx.x * kFaHeads;
    int n_win, n_sel, w0;
    if (ar.gen) {
        const long long ta = ar.t0 + t, pa = ar.p0 + ta;
        const long long first = max((long long) ar.win - ar.p0, ta + 1);
        w0 = (int) first;
        n_win = (int) (ar.win + ta - first + 1);
        n_sel = ar.R > 0 ? (int) min((long long) ar.kcap, (pa + 1) / ar.R) : 0;
    } else {
        n_win = ar.nwin[t], n_sel = ar.nsel[t], w0 = ar.win0[t];
    }
    const uint16_t* wbase = win + (size_t) w0 * kFaD;
    const int* sl = sel + (size_t) t * sel_stride;
    const size_t crow = kv_row_bytes(kKvFp4, kFaD);
    // the block's 16 heads' queries as FP16 (the matrix's A operand), and their running softmax state
    for (int i = tid; i < kFaHeads * kFaD; i += kThreads) {
        const int h = i / kFaD, c = i % kFaD;
        qs[h * kFaLd + c] = __builtin_bit_cast(uint16_t, (_Float16) q[((size_t) t * heads + hb + h) * kFaD + c]);
    }
    if (tid < kFaHeads) {
        m_run[tid] = -1e30f;
        den[tid] = 0.0f;
    }
    f8v o[4];
#pragma unroll
    for (int f = 0; f < 4; ++f)
#pragma unroll
        for (int i = 0; i < 8; ++i) o[f][i] = 0.0f;
    __syncthreads();
    auto stage = [&](int t0, int n) {   // the tile's n rows into ks (BF16 rows -> FP16)
        const int sr = tid >> 4, part = tid & 15;
        if (sr >= n) return;
        const uint8_t* row = rows[sr];
        uint16_t tmp[32];
        if (t0 + sr < n_win) {
            const uint4* src = (const uint4*) (row + part * 64);
            const uint4 a = src[0], b = src[1], c = src[2], d = src[3];
            ((uint4*) tmp)[0] = a;
            ((uint4*) tmp)[1] = b;
            ((uint4*) tmp)[2] = c;
            ((uint4*) tmp)[3] = d;
        } else {
            unpack32(row, part, (uint4*) tmp);
        }
        uint16_t* dst = ks + sr * kFaLd + part * 32;
#pragma unroll
        for (int i = 0; i < 32; ++i) dst[i] = __builtin_bit_cast(uint16_t, (_Float16) bf16f(tmp[i]));
    };
    const int total = n_win + n_sel;
    for (int t0 = 0; t0 < total; t0 += kFaKeys) {
        const int nt = min(kFaKeys, total - t0);
        if (tid < nt) {
            const int j = t0 + tid;
            rows[tid] = j < n_win ? (const uint8_t*) (wbase + (size_t) j * kFaD) : ckv + (size_t) sl[j - n_win] * crow;
        }
        __syncthreads();
        stage(t0, nt);
        __syncthreads();
        {   // the scores in 8 slices of 64 dims, a warp a slice, summed in shared
            f8v acc = {0, 0, 0, 0, 0, 0, 0, 0};
            const int k0 = wp * 64;
            for (int kk = k0; kk < k0 + 64; kk += 16) {
                h16v a, b;
                const uint32_t* qa = (const uint32_t*) (qs + l16 * kFaLd + kk);
                const uint32_t* kb = (const uint32_t*) (ks + l16 * kFaLd + kk);
#pragma unroll
                for (int i = 0; i < 8; ++i) {   // (two halves a load)
                    a[2 * i] = __builtin_bit_cast(_Float16, (uint16_t) (qa[i] & 0xffffu));
                    a[2 * i + 1] = __builtin_bit_cast(_Float16, (uint16_t) (qa[i] >> 16));
                    b[2 * i] = __builtin_bit_cast(_Float16, (uint16_t) (kb[i] & 0xffffu));
                    b[2 * i + 1] = __builtin_bit_cast(_Float16, (uint16_t) (kb[i] >> 16));
                }
                acc = set_wmma(a, b, acc);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) part[wp][set_wmma_row(i, hi)][l16] = acc[i];
            __syncthreads();
            const int h = tid >> 4, k = tid & 15;
            float a = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; ++w) a += part[w][h][k];
            sc[h][k] = t0 + k < total ? a * scale : -INFINITY;
        }
        __syncthreads();
        {   // the online softmax: a warp per 2 heads, a lane a key (16 lanes a head)
            const int h = 2 * wp + (lane >> 4), k = lane & 15;
            float m = sc[h][k];
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
            const float mo = m_run[h], mn = fmaxf(mo, m);
            const float p = expf(sc[h][k] - mn);
            ps[h][k] = __builtin_bit_cast(uint16_t, (_Float16) p);
            float s = p;
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
            if (k == 0) {
                ts[h] = expf(mo - mn);
                den[h] = den[h] * ts[h] + s;
                m_run[h] = mn;
            }
        }
        __syncthreads();
        const int dim0 = wp * 64;
#pragma unroll
        for (int f = 0; f < 4; ++f)
#pragma unroll
            for (int i = 0; i < 8; ++i) o[f][i] *= ts[set_wmma_row(i, hi)];
#pragma unroll
        for (int f = 0; f < 4; ++f) {   // the values: P (16 heads x 16 keys) x V (16 keys x 64 dims this warp)
            h16v a, b;
            const uint32_t* pp = (const uint32_t*) ps[l16];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                a[2 * i] = __builtin_bit_cast(_Float16, (uint16_t) (pp[i] & 0xffffu));
                a[2 * i + 1] = __builtin_bit_cast(_Float16, (uint16_t) (pp[i] >> 16));
                b[2 * i] = __builtin_bit_cast(_Float16, ks[(2 * i) * kFaLd + dim0 + f * 16 + l16]);
                b[2 * i + 1] = __builtin_bit_cast(_Float16, ks[(2 * i + 1) * kFaLd + dim0 + f * 16 + l16]);
            }
            o[f] = set_wmma(a, b, o[f]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int f = 0; f < 4; ++f)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int h = set_wmma_row(i, hi), dm = wp * 64 + f * 16 + l16;
            const float dv = den[h] + expf(sink[hb + h] - m_run[h]);
            out[((size_t) t * heads + hb + h) * kFaD + dm] = o[f][i] / dv;
        }
}
#endif   // STRATA_USE_HIP

// the ring's rows are kKvFp8: E4M3 codes, a UE8M0 scale per 32 (decoded: exactly the BF16 values they were made from)
__global__ void ring_to_ext_kernel(const uint8_t* __restrict__ ring, uint16_t* __restrict__ ext, long long p0, int n,
                                   int d) {
    const int r = blockIdx.x;
    const long long p = p0 - n + r;
    const uint8_t* row = ring + (size_t) (p >= 0 ? p % n : 0) * kv_row_bytes(kKvFp8, d);
    for (int e = threadIdx.x; e < d; e += blockDim.x)
        ext[(size_t) r * d + e] = p >= 0 ? f2bf16(e4m3_value(row[e]) * ue8m0_val(row[d + e / 32])) : (uint16_t) 0;
}

// a window row's BF16 values (FP8 roundings: each an E4M3 value times its group's power of two) as E4M3 codes at the
// smallest power-of-two scale 2^k their group fits under (max |v| <= 448 * 2^k = 1.75 * 2^(8 + k)): never above the
// rounding's own scale, so each v / 2^k is that E4M3 value times a power of two >= 1, at most 448 - an E4M3 value again
__global__ void __launch_bounds__(kThreads) ext_to_ring_kernel(const uint16_t* __restrict__ ext,
                                                               uint8_t* __restrict__ ring, long long p0, int n, int T,
                                                               int d, uint8_t* __restrict__ backup) {
    const int t = blockIdx.x;
    if (T - t > n) return;   // overwritten by a later row of the window anyway
    const size_t rb = kv_row_bytes(kKvFp8, d);
    uint8_t* row = ring + (size_t) ((p0 + t) % n) * rb;
    if (backup != nullptr) {
        for (size_t i = threadIdx.x; i < rb; i += blockDim.x) backup[(size_t) t * rb + i] = row[i];
        __syncthreads();
    }
    const uint16_t* src = ext + (size_t) (n + t) * d;
    for (int e0 = 0; e0 < d; e0 += blockDim.x) {   // (d a multiple of 32: a warp's 32 values one group)
        const int e = e0 + (int) threadIdx.x;
        const float v = e < d ? bf16f(src[e]) : 0.0f;
        uint32_t a = __float_as_uint(v) & 0x7fffffffu;   // (the magnitudes' order as bits)
        for (int o = 16; o > 0; o >>= 1) a = max(a, (uint32_t) __shfl_xor_sync(0xffffffffu, a, o));
        const int k = a == 0u ? 0 : (int) (a >> 23) - 135 + ((a & 0x7fffffu) > 0x600000u ? 1 : 0);
        if (e >= d) continue;
        row[e] = (uint8_t) e4m3_code(v, k);
        if ((e & 31) == 0) row[d + e / 32] = (uint8_t) (k + 127);
    }
}

__global__ void ring_restore_kernel(uint8_t* __restrict__ ring, const uint8_t* __restrict__ backup, long long p0, int n,
                                    int keep, size_t rb) {
    const int t = keep + blockIdx.x;
    uint8_t* row = ring + (size_t) ((p0 + t) % n) * rb;
    for (size_t i = threadIdx.x; i < rb; i += blockDim.x) row[i] = backup[(size_t) t * rb + i];
}
__global__ void ring_restore_batch_kernel(const RingBatch b, long long p0, int n, int keep, size_t rb) {
    const int t = keep + blockIdx.x, li = blockIdx.y;
    uint8_t* row = b.ring[li] + (size_t) ((p0 + t) % n) * rb;
    const uint8_t* src = b.bak[li] + (size_t) t * rb;
    for (size_t i = threadIdx.x; i < rb; i += blockDim.x) row[i] = src[i];
}

__global__ void hc_mean_kernel(const float* __restrict__ h, int hc, int dim, float* __restrict__ out,
                               long long out_stride) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (e >= dim) return;
    float a = 0.0f;
    for (int s = 0; s < hc; ++s) a += h[((size_t) t * hc + s) * dim + e];
    out[(long long) t * out_stride + e] = a / (float) hc;
}

__global__ void iota_rows_kernel(int* out, int n, int stride) {
    const int t = blockIdx.x;
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[(size_t) t * stride + i] = i;
}

// a warp a vocabulary row: E[*tok] (shared memory) . H[v]
__global__ void __launch_bounds__(kThreads) markov_kernel(float* __restrict__ logits, const uint16_t* __restrict__ E,
                                                          const uint16_t* __restrict__ H, const int* __restrict__ tok,
                                                          int V, int R) {
    extern __shared__ float er[];
    const uint16_t* erow = E + (size_t) (*tok) * R;
    for (int i = threadIdx.x; i < R; i += blockDim.x) er[i] = bf16f(erow[i]);
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int v = blockIdx.x * (kThreads / 32) + (threadIdx.x >> 5);
    if (v >= V) return;
    const uint16_t* hr = H + (size_t) v * R;
    float a = 0.0f;
    for (int c = lane * 8; c < R; c += 32 * 8) {
        const uint4 q = *(const uint4*) (hr + c);
        const __half2* hp = (const __half2*) &q;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __half22float2(hp[i]);
            a += f.x * er[c + 2 * i] + f.y * er[c + 2 * i + 1];
        }
    }
    a = warp_sum(a);
    if (lane == 0) logits[v] += a;
}

// the same bias with the head quantised to Q8_0 blocks of 32 (34 B: an F16 d and 32 int8): half the bytes read for the same sums.
// A lane's 8 values sit inside one block (8 divides 32), so a lane reads one d and 8 codes; the sums are the same terms, in the
// block's own order, with the quantisation's error the only difference.
__global__ void __launch_bounds__(kThreads) markov_kernel_q8(float* __restrict__ logits, const uint16_t* __restrict__ E,
                                                            const uint8_t* __restrict__ Hq, const int* __restrict__ tok,
                                                            int V, int R) {
    extern __shared__ float er[];
    const uint16_t* erow = E + (size_t) (*tok) * R;
    for (int i = threadIdx.x; i < R; i += blockDim.x) er[i] = bf16f(erow[i]);
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int v = blockIdx.x * (kThreads / 32) + (threadIdx.x >> 5);
    if (v >= V) return;
    const uint8_t* hr = Hq + (size_t) v * (size_t) (R / 32) * 34;
    float a = 0.0f;
    for (int c = lane * 8; c < R; c += 32 * 8) {
        const uint8_t* blk = hr + (size_t) (c >> 5) * 34;
        const float d = __half2float(*(const __half*) blk);
        const int8_t* q = (const int8_t*) (blk + 2 + (c & 31));
        float t = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) t += (float) q[i] * er[c + i];
        a += d * t;
    }
    a = warp_sum(a);
    if (lane == 0) logits[v] += a;
}

// a member's rows [v0, v0 + vn) of the bias (markov_kernel's sums, the prev token's E row already on the device, BF16)
__global__ void __launch_bounds__(kThreads) markov_share_kernel(float* __restrict__ logits,
                                                                const uint16_t* __restrict__ H,
                                                                const uint16_t* __restrict__ erow, int vn, int R) {
    extern __shared__ float er[];
    for (int i = threadIdx.x; i < R; i += blockDim.x) er[i] = bf16f(erow[i]);
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int v = blockIdx.x * (kThreads / 32) + (threadIdx.x >> 5);
    if (v >= vn) return;
    const uint16_t* hr = H + (size_t) v * R;
    float a = 0.0f;
    for (int c = lane * 8; c < R; c += 32 * 8) {
        const uint4 q = *(const uint4*) (hr + c);
        const __half2* hp = (const __half2*) &q;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __half22float2(hp[i]);
            a += f.x * er[c + 2 * i] + f.y * er[c + 2 * i + 1];
        }
    }
    a = warp_sum(a);
    if (lane == 0) logits[v] += a;
}
// the share's argmax (the greedy sampler's rule: the larger value, on a tie the lower index) -> slot {value, v0 + i,
// seq}, the seq last
__global__ void __launch_bounds__(1024) argmax_share_kernel(const float* __restrict__ row, int vn, int v0,
                                                           unsigned* slot, unsigned seq) {
    __shared__ float sv[32];
    __shared__ int si[32];
    float bv = __int_as_float(0xff800000);
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < vn; i += blockDim.x) {
        const float x = row[i];
        if (x > bv) bv = x, bi = i;   // (ascending: the lowest index of a tie stays)
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
    }
    if ((threadIdx.x & 31) == 0) sv[threadIdx.x >> 5] = bv, si[threadIdx.x >> 5] = bi;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < (int) (blockDim.x >> 5); ++w)
            if (sv[w] > bv || (sv[w] == bv && si[w] < bi)) bv = sv[w], bi = si[w];
        slot[0] = __float_as_uint(bv);
        slot[1] = (unsigned) (v0 + (bi == 0x7fffffff ? 0 : bi));
        __threadfence_system();
        *(volatile unsigned*) (slot + 2) = seq;
    }
}
// every member's share argmax of this step (slots nm apart) -> the global one into *tok, and its E row (host-mapped
// BF16 [V][R]) into erow for the next step's bias
__global__ void markov_pick_kernel(const unsigned* slots, int nm, unsigned seq, int* tok, const uint16_t* E,
                                   uint16_t* erow, int R) {
    __shared__ int pick;
    if ((int) threadIdx.x < nm)
        while ((int) (*(volatile const unsigned*) (slots + (size_t) threadIdx.x * 4 + 2) - seq) < 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
            __nanosleep(32);
#endif
        }
    __syncthreads();
    if (threadIdx.x == 0) {
        float bv = __int_as_float(0xff800000);
        int bi = 0x7fffffff;
        for (int m = 0; m < nm; ++m) {   // (members in vocabulary order: a tie keeps the lower index)
            const float v = __uint_as_float(*(volatile const unsigned*) (slots + (size_t) m * 4));
            const int i = (int) *(volatile const unsigned*) (slots + (size_t) m * 4 + 1);
            if (v > bv || (v == bv && i < bi)) bv = v, bi = i;
        }
        pick = bi;
        *tok = bi;
    }
    __syncthreads();
#if defined(STRATA_USE_HIP)   // (E is fixed once loaded: a plain load is as fresh)
    for (int i = threadIdx.x; i < R; i += blockDim.x) erow[i] = E[(size_t) pick * R + i];
#else
    for (int i = threadIdx.x; i < R; i += blockDim.x) erow[i] = __ldcv(E + (size_t) pick * R + i);
#endif
}

__global__ void __launch_bounds__(1024) confidence_kernel(const float* __restrict__ x, const uint16_t* __restrict__ E,
                                                          const int* __restrict__ tok, const float* __restrict__ w,
                                                          int dim, int R, float* __restrict__ out) {
    const int t = blockIdx.x;
    const float* xt = x + (size_t) t * dim;
    const uint16_t* er = E + (size_t) tok[t] * R;
    float a = 0.0f;
    for (int i = threadIdx.x; i < dim + R; i += blockDim.x) a += w[i] * (i < dim ? xt[i] : bf16f(er[i - dim]));
    a = block_sum(a);
    if (threadIdx.x == 0) out[t] = 1.0f / (1.0f + expf(-a));
}

// ---------------------------------------------------------------- engram
// ggml's IQ4_NL: a block of 32 values is 18 bytes, an fp16 d then 16 packed nibble bytes; the low nibble of qs[j] is
// value j and its high nibble value j + 16, each an index into this codebook (ggml-common.h's kvalues_iq4nl,
// dequantize_row_iq4_nl).  A row is hd / 32 of these blocks, the d of every one inside it - no separate scales.
__device__ __constant__ int8_t kIq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                             1,    13,   25,   38,   53,   69,   89,   113};
__device__ __forceinline__ float iq4nl_value(const uint8_t* b, int e) {
    const uint8_t q = b[2 + (e & 15)];
    const int v = (e & 16) ? (q >> 4) : (q & 15);
    return __half2float(__ushort_as_half(*(const uint16_t*) b)) * (float) kIq4nl[v];
}

__global__ void engram_dequant_kernel(const uint8_t* __restrict__ rows, const uint8_t* __restrict__ scales, int hd,
                                      float* __restrict__ out, int fmt) {
    const int r = blockIdx.x;
    if (fmt == 1) {   // (ggml Q6_K superblocks at the row's start)
        for (int e = threadIdx.x; e < hd; e += blockDim.x)
            out[(size_t) r * hd + e] = q6k_value(rows + (size_t) r * hd + (size_t) (e / 256) * kQ6kBytes, e % 256);
        return;
    }
    if (fmt == 2) {   // (ggml IQ4_NL blocks at the row's start, their scales inside)
        for (int e = threadIdx.x; e < hd; e += blockDim.x)
            out[(size_t) r * hd + e] = iq4nl_value(rows + (size_t) r * hd + (size_t) (e / 32) * 18, e % 32);
        return;
    }
    for (int e = threadIdx.x; e < hd; e += blockDim.x) {
        const float hv = e4m3_value(rows[(size_t) r * hd + e]);
        const uint8_t sb = scales[(size_t) r * (hd / 32) + e / 32];
        const float s = sb == 0xffu ? __int_as_float(0x7fc00000) : (sb == 0 ? 5.877471754111438e-39f
                                                                            : __uint_as_float((uint32_t) sb << 23));
        out[(size_t) r * hd + e] = hv * s;
    }
}

__global__ void __launch_bounds__(1024) engram_gate_kernel(float* __restrict__ h, const float* __restrict__ kv,
                                                           const float* __restrict__ w, int hc, int dim, float eps) {
    const int k = blockIdx.x, t = blockIdx.y;
    float* hk = h + ((size_t) t * hc + k) * dim;
    const float* kvt = kv + (size_t) t * (hc + 1) * dim;
    const float* key = kvt + (size_t) k * dim;
    const float* val = kvt + (size_t) hc * dim;
    const float* wk = w + (size_t) k * dim;
    float sh = 0.0f, sk = 0.0f, sd = 0.0f;
    for (int e = threadIdx.x; e < dim; e += blockDim.x) {
        const float hv = hk[e], kv_ = key[e];
        sh += hv * hv;
        sk += kv_ * kv_;
        sd += hv * wk[e] * kv_;
    }
    sh = block_sum(sh);
    sk = block_sum(sk);
    sd = block_sum(sd);
    const float rstd = rsqrtf(sh / (float) dim + eps) * rsqrtf(sk / (float) dim + eps);
    const float dot = sd * rstd * rsqrtf((float) dim);
    const float sq = copysignf(sqrtf(fmaxf(fabsf(dot), 1e-6f)), dot);
    const float gate = 1.0f / (1.0f + expf(-sq));
    for (int e = threadIdx.x; e < dim; e += blockDim.x) hk[e] = hk[e] + gate * val[e];
}

__global__ void gather_kernel(const float* __restrict__ x, Gather g, int cols, float* __restrict__ out) {
    const int r = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < cols) out[(size_t) r * cols + c] = x[(size_t) g.src[r] * cols + c];
}

__global__ void combine_kernel(const float* __restrict__ ed, const float* __restrict__ cpu,
                               const float* __restrict__ sh, Combine cm, int dim, float* __restrict__ out) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (e >= dim) return;
    float a = 0.0f;
    for (int j = 0; j < cm.n[t]; ++j) a += cm.w[t][j] * ed[(size_t) cm.row[t][j] * dim + e];
    if (cpu != nullptr && cm.cpu[t]) a += cpu[(size_t) t * dim + e];
    if (sh != nullptr) a += sh[(size_t) t * dim + e];
    out[(size_t) t * dim + e] = a;
}

__global__ void __launch_bounds__(512) route_publish_kernel(const float* __restrict__ logits,
                                                            const float* __restrict__ bias, int n, int k,
                                                            float route_scale, const unsigned long long* __restrict__ vram,
                                                            const float* __restrict__ x, int dim, int* idx, float* w,
                                                            unsigned long long* plan_ptr, Req* req,
                                                            const float* __restrict__ bias_vl, unsigned vl) {
    __shared__ float sc[1024], bs[1024];
    __shared__ float bv[32];
    __shared__ int bi[32];
    __shared__ int pick[16];
    __shared__ int on_cpu;
    const int t = blockIdx.x;
    logits += (size_t) t * 1024;
    if (bias_vl != nullptr && ((vl >> t) & 1u)) bias = bias_vl;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float l = logits[i];
        const float sp = l > 20.0f ? l : log1pf(expf(l));
        sc[i] = sqrtf(sp);
        bs[i] = sc[i] + bias[i];
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    for (int r = 0; r < k; ++r) {
        float best = -FLT_MAX;
        int bix = 0x7fffffff;
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            if (bs[i] > best || (bs[i] == best && i < bix)) best = bs[i], bix = i;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bix, o);
            if (ov > best || (ov == best && oi < bix)) best = ov, bix = oi;
        }
        if (lane == 0) bv[wp] = best, bi[wp] = bix;
        __syncthreads();
        if (threadIdx.x == 0) {
            float b = -FLT_MAX;
            int ix = 0x7fffffff;
            for (int q = 0; q < (int) (blockDim.x >> 5); ++q)
                if (bv[q] > b || (bv[q] == b && bi[q] < ix)) b = bv[q], ix = bi[q];
            pick[r] = ix;
            bs[ix] = -FLT_MAX;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int r = 0; r < k; ++r) s += sc[pick[r]];
        int cpu = 0;
        for (int r = 0; r < k; ++r) {
            const float wv = sc[pick[r]] / (s + 1e-20f) * route_scale;
            idx[(size_t) t * kPicks + r] = pick[r];
            w[(size_t) t * kPicks + r] = wv;
            plan_ptr[(size_t) t * kPicks + r] = vram[pick[r]];
            cpu |= vram[pick[r]] == 0ull;
            if (req != nullptr) {
                req->ids[t][r] = pick[r];
                req->w[t][r] = wv;
            }
        }
        on_cpu = cpu;
    }
    // the token's FFN input to the host only where the CPU lane reads it (a pick of its on the CPU lane): each row
    // 20 KB over PCIe, where most windows have none
    __syncthreads();
    if (req != nullptr && on_cpu)
        for (int i = threadIdx.x; i < dim; i += blockDim.x) req->x[t][i] = x[(size_t) t * dim + i];
}

// a later layer's m likely picks (route_kernel's choice from its logits) into req->ahead[t][off ..)
__global__ void __launch_bounds__(512) route_ahead_kernel(const float* __restrict__ logits,
                                                          const float* __restrict__ bias, int n, int m, Req* req,
                                                          int off) {
    __shared__ float bs[1024];
    __shared__ float bv[32];
    __shared__ int bi[32];
    const int t = blockIdx.x;
    logits += (size_t) t * 1024;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float l = logits[i];
        const float sp = l > 20.0f ? l : log1pf(expf(l));
        bs[i] = sqrtf(sp) + bias[i];
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    for (int r = 0; r < m; ++r) {
        float best = -FLT_MAX;
        int bix = 0x7fffffff;
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            if (bs[i] > best || (bs[i] == best && i < bix)) best = bs[i], bix = i;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bix, o);
            if (ov > best || (ov == best && oi < bix)) best = ov, bix = oi;
        }
        if (lane == 0) bv[wp] = best, bi[wp] = bix;
        __syncthreads();
        if (threadIdx.x == 0) {
            float b = -FLT_MAX;
            int ix = 0x7fffffff;
            for (int q = 0; q < (int) (blockDim.x >> 5); ++q)
                if (bv[q] > b || (bv[q] == b && bi[q] < ix)) b = bv[q], ix = bi[q];
            req->ahead[t][off + r] = ix < n ? ix : -1;
            if (ix < n) bs[ix] = -FLT_MAX;
        }
        __syncthreads();
    }
}

__global__ void publish_kernel(Req* req, int T, unsigned seq, int na0, int na1) {
    __threadfence_system();
    req->na[0] = na0;
    req->na[1] = na1;
    req->T = T;
    __threadfence_system();
    *(volatile unsigned*) &req->seq = seq;
}

__global__ void wait_seq_kernel(const unsigned* seq_ptr, unsigned seq) {
    if (threadIdx.x != 0) return;
    while (*(volatile const unsigned*) seq_ptr != seq) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(200);
#endif
    }
    __threadfence_system();
}

__global__ void init_pre_kernel(float* pre, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < T * 8) pre[i] = (i % 8) == 0 ? 1.0f : 0.0f;
}


// the hand-offs through host-mapped memory: a grid-stride copy of the segments (16 bytes a thread), then (put, flag
// given) the last block to finish sets the flag after a system fence; a get first waits for the flag
__device__ __forceinline__ void xfer_copy(const Xfer& x, bool from_host) {
    size_t total = 0;
    for (int i = 0; i < x.n; ++i) total += x.bytes[i] / 16;
    for (size_t q = (size_t) blockIdx.x * blockDim.x + threadIdx.x; q < total; q += (size_t) gridDim.x * blockDim.x) {
        size_t r = q;
        int i = 0;
        while (r >= x.bytes[i] / 16) r -= x.bytes[i++] / 16;
        const uint4* src = (const uint4*) x.src[i] + r;
        ((uint4*) x.dst[i])[r] = from_host ? ld_fresh(src) : *src;
    }
}

__global__ void __launch_bounds__(kThreads) xfer_put_kernel(const Xfer x, unsigned* flag, unsigned seq, unsigned* cnt) {
    xfer_copy(x, false);
    if (flag == nullptr) return;
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0 && atomicAdd(cnt, 1u) == gridDim.x - 1) {
        *cnt = 0u;
        __threadfence_system();
        *(volatile unsigned*) flag = seq;
    }
}

__device__ __forceinline__ void wait_at_least(const unsigned* flag, unsigned seq) {
    while ((int) (*(volatile const unsigned*) flag - seq) < 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(100);
#endif
    }
}

__global__ void __launch_bounds__(kThreads) xfer_get_kernel(const Xfer x, const unsigned* flag, unsigned seq) {
    if (threadIdx.x == 0) wait_at_least(flag, seq);
    __syncthreads();
    xfer_copy(x, true);
}

__global__ void wait_flags_kernel(const unsigned* flags, int n, int stride, unsigned seq) {
    if (threadIdx.x != 0) return;
    for (int i = 0; i < n; ++i) wait_at_least(flags + (size_t) i * stride, seq);
    __threadfence_system();
}
__global__ void set_flag_kernel(unsigned* flag, unsigned v) {
    __threadfence_system();
    *(volatile unsigned*) flag = v;
}

// three members or more: a reduce-scatter and an all-gather in one launch, a block a contiguous range of the row cut
// in nm chunks.  The block puts its range of this member's partial out and flags it; once every member's flag of the
// range is up it sums chunk mi of the range in member order (ar_sum_kernel's sums, bit for bit), puts that out and
// flags it; once every member's sum flag is up it reads the other chunks' sums.  Each member reads about two rows over
// PCIe whatever nm is, where every member reading every other row grows with nm
__global__ void __launch_bounds__(kThreads) ar_rs_kernel(float* x, int n4, float* rows, float* sums, size_t stride,
                                                        unsigned* bflags, int nm, int mi, int slot, int slots,
                                                        unsigned seq) {
    float4* x4 = (float4*) x;
    const int per = (n4 + (int) gridDim.x - 1) / (int) gridDim.x;
    const int lo = blockIdx.x * per, hi = min(n4, lo + per), cs = (hi - lo + nm - 1) / nm;
    auto rflag = [&](int m) { return bflags + (((size_t) m * slots + slot) * kArBlocks + blockIdx.x) * 8; };
    auto sflag = [&](int m) {
        return bflags + ((((size_t) nm + m) * slots + slot) * kArBlocks + blockIdx.x) * 8;
    };
    auto row = [&](int m) { return (float4*) (rows + ((size_t) m * slots + slot) * stride); };
    float4* sum = (float4*) (sums + (size_t) slot * stride);
    if (lo < hi) {
        float4* mine = row(mi);
        for (int q = lo + threadIdx.x; q < hi; q += blockDim.x) mine[q] = x4[q];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) *(volatile unsigned*) rflag(mi) = seq;
    if ((int) threadIdx.x < nm && (int) threadIdx.x != mi) wait_at_least(rflag(threadIdx.x), seq);
    __syncthreads();
    const int c0 = min(hi, lo + mi * cs), c1 = min(hi, c0 + cs);
    for (int q = c0 + threadIdx.x; q < c1; q += blockDim.x) {
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int m = 0; m < nm; ++m) {
            const float4 p = m == mi ? x4[q] : ld_fresh(row(m) + q);
            if (m == 0) v = p;
            else {
                v.x = __fadd_rn(v.x, p.x);
                v.y = __fadd_rn(v.y, p.y);
                v.z = __fadd_rn(v.z, p.z);
                v.w = __fadd_rn(v.w, p.w);
            }
        }
        sum[q] = v;
        x4[q] = v;
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) *(volatile unsigned*) sflag(mi) = seq;
    if ((int) threadIdx.x < nm && (int) threadIdx.x != mi) wait_at_least(sflag(threadIdx.x), seq);
    __syncthreads();
    for (int q = lo + threadIdx.x; q < hi; q += blockDim.x)
        if (q < c0 || q >= c1) x4[q] = ld_fresh(sum + q);
}

// ar_rs_kernel with the rows and the sums in FP16 (DS4_AR_F16): half the bytes over PCIe.  Every member's partial is
// rounded to FP16 (the owner's too), summed in F32 in member order, the sum rounded to FP16 - and every member, the
// owner too, takes that rounded sum, so the members' copies of the stream stay the same
__device__ __forceinline__ uint2 to_h4(float4 v) {
    const __half2 a = __floats2half2_rn(v.x, v.y), b = __floats2half2_rn(v.z, v.w);
    return make_uint2(*(const unsigned*) &a, *(const unsigned*) &b);
}
__device__ __forceinline__ float4 from_h4(uint2 u) {
    const float2 a = __half22float2(*(const __half2*) &u.x), b = __half22float2(*(const __half2*) &u.y);
    return make_float4(a.x, a.y, b.x, b.y);
}
__device__ __forceinline__ float4 sat4(float4 v) {   // (FP16's range: a partial past it saturates instead of inf)
    const float m = 65504.0f;
    return make_float4(fminf(fmaxf(v.x, -m), m), fminf(fmaxf(v.y, -m), m), fminf(fmaxf(v.z, -m), m), fminf(fmaxf(v.w, -m), m));
}
__global__ void __launch_bounds__(kThreads) ar_rs16_kernel(float* x, int n4, float* rows, float* sums, size_t stride,
                                                          unsigned* bflags, int nm, int mi, int slot, int slots,
                                                          unsigned seq) {
    float4* x4 = (float4*) x;
    const int per = (n4 + (int) gridDim.x - 1) / (int) gridDim.x;
    const int lo = blockIdx.x * per, hi = min(n4, lo + per), cs = (hi - lo + nm - 1) / nm;
    auto rflag = [&](int m) { return bflags + (((size_t) m * slots + slot) * kArBlocks + blockIdx.x) * 8; };
    auto sflag = [&](int m) {
        return bflags + ((((size_t) nm + m) * slots + slot) * kArBlocks + blockIdx.x) * 8;
    };
    auto row = [&](int m) { return (uint2*) (rows + ((size_t) m * slots + slot) * stride); };   // (FP16: half a row)
    uint2* sum = (uint2*) (sums + (size_t) slot * stride);
    if (lo < hi) {
        uint2* mine = row(mi);
        for (int q = lo + threadIdx.x; q < hi; q += blockDim.x) mine[q] = to_h4(sat4(x4[q]));
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) *(volatile unsigned*) rflag(mi) = seq;
    if ((int) threadIdx.x < nm && (int) threadIdx.x != mi) wait_at_least(rflag(threadIdx.x), seq);
    __syncthreads();
    const int c0 = min(hi, lo + mi * cs), c1 = min(hi, c0 + cs);
    for (int q = c0 + threadIdx.x; q < c1; q += blockDim.x) {
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int m = 0; m < nm; ++m) {
            const uint2 u = m == mi ? to_h4(sat4(x4[q])) : ld_fresh(row(m) + q);
            const float4 p = from_h4(u);
            if (m == 0) v = p;
            else {
                v.x = __fadd_rn(v.x, p.x);
                v.y = __fadd_rn(v.y, p.y);
                v.z = __fadd_rn(v.z, p.z);
                v.w = __fadd_rn(v.w, p.w);
            }
        }
        const uint2 s16 = to_h4(sat4(v));
        sum[q] = s16;
        x4[q] = from_h4(s16);
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) *(volatile unsigned*) sflag(mi) = seq;
    if ((int) threadIdx.x < nm && (int) threadIdx.x != mi) wait_at_least(sflag(threadIdx.x), seq);
    __syncthreads();
    for (int q = lo + threadIdx.x; q < hi; q += blockDim.x)
        if (q < c0 || q >= c1) x4[q] = from_h4(ld_fresh(sum + q));
}

// x[i] = the members' partials summed in member order (this member's from x, the others' from their host rows)
__global__ void __launch_bounds__(kThreads) ar_sum_kernel(float* x, int n4, const float* peers, size_t stride,
                                                         const unsigned* flags, int nm, int mi, int slot, int slots,
                                                         unsigned seq) {
    if (threadIdx.x == 0)
        for (int m = 0; m < nm; ++m)
            if (m != mi) wait_at_least(flags + ((size_t) m * slots + slot) * 32, seq);
    __syncthreads();
    for (int q = blockIdx.x * blockDim.x + threadIdx.x; q < n4; q += gridDim.x * blockDim.x) {
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int m = 0; m < nm; ++m) {
            const float4 p = m == mi ? ((const float4*) x)[q]
                                     : ld_fresh((const float4*) (peers + ((size_t) m * slots + slot) * stride) + q);
            if (m == 0) v = p;
            else {
                v.x = __fadd_rn(v.x, p.x);
                v.y = __fadd_rn(v.y, p.y);
                v.z = __fadd_rn(v.z, p.z);
                v.w = __fadd_rn(v.w, p.w);
            }
        }
        ((float4*) x)[q] = v;
    }
}


// a thread a (token, value): every stream's value read before any is written
__global__ void hc_post_inplace_kernel(const float* __restrict__ y, float* h, const float* __restrict__ post,
                                       const float* __restrict__ comb, int dim, int hc) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (e >= dim) return;
    float* r = h + (size_t) t * hc * dim;
    float rv[8];
    for (int j = 0; j < hc; ++j) rv[j] = r[(size_t) j * dim + e];
    const float yv = y[(size_t) t * dim + e];
    for (int k = 0; k < hc; ++k) {
        float a = post[(size_t) t * 8 + k] * yv;
        for (int j = 0; j < hc; ++j) a += comb[(size_t) t * 64 + j * hc + k] * rv[j];
        r[(size_t) k * dim + e] = a;
    }
}

__global__ void __launch_bounds__(kThreads) hc_mix_scale_kernel(const float* __restrict__ h, int hcdim, int n_mix,
                                                                float eps, float* mixes) {
    const int t = blockIdx.x;
    const float* x = h + (size_t) t * hcdim;
    float ss = 0.0f;
    for (int k = threadIdx.x * 4; k < hcdim; k += kThreads * 4) {
        const float4 hv = *(const float4*) (x + k);
        ss += hv.x * hv.x + hv.y * hv.y + hv.z * hv.z + hv.w * hv.w;
    }
    ss = block_sum(ss);
    const float inv = rsqrtf(ss / (float) hcdim + eps);
    if (threadIdx.x < n_mix) mixes[(size_t) t * 32 + threadIdx.x] *= inv;
}

__global__ void engram_dequant_tok_kernel(const uint8_t* __restrict__ src, size_t per, int cols, int hd,
                                          float* __restrict__ out, int fmt) {
    const int r = blockIdx.x, t = r / cols, c = r - t * cols;
    const uint8_t* rows = src + (size_t) t * per + (size_t) c * hd;
    const uint8_t* scales = src + (size_t) t * per + (size_t) cols * hd + (size_t) c * (hd / 32);
    if (fmt == 1) {   // (ggml Q6_K superblocks at the row's start)
        for (int e = threadIdx.x; e < hd; e += blockDim.x)
            out[(size_t) r * hd + e] = q6k_value(rows + (size_t) (e / 256) * kQ6kBytes, e % 256);
        return;
    }
    if (fmt == 2) {   // (ggml IQ4_NL blocks at the row's start, their scales inside)
        for (int e = threadIdx.x; e < hd; e += blockDim.x)
            out[(size_t) r * hd + e] = iq4nl_value(rows + (size_t) (e / 32) * 18, e % 32);
        return;
    }
    for (int e = threadIdx.x; e < hd; e += blockDim.x) {
        const float hv = e4m3_value(rows[e]);
        const uint8_t sb = scales[e / 32];
        const float s = sb == 0xffu ? __int_as_float(0x7fc00000) : (sb == 0 ? 5.877471754111438e-39f
                                                                            : __uint_as_float((uint32_t) sb << 23));
        out[(size_t) r * hd + e] = hv * s;
    }
}

// a block a virtual slot: its picks counted (sort pass 1), or (fill) placed in token order by a block-wide scan
template <bool FILL>
__global__ void __launch_bounds__(kThreads) moe_sort_kernel(const int* __restrict__ idx, int T, int k,
                                                            const int* __restrict__ vs, int* bounds, int* row_tok,
                                                            int* pos) {
    const int slot = blockIdx.x;
    __shared__ int warp_n[kThreads / 32];
    __shared__ int base_s;
    if (threadIdx.x == 0) base_s = FILL ? bounds[slot] : 0;
    __syncthreads();
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const int n = T * k;
    for (int q0 = 0; q0 < n; q0 += kThreads) {
        const int q = q0 + threadIdx.x;
        const int t = q / k, i = q - t * k;
        const bool hit = q < n && vs[idx[(size_t) t * kPicks + i]] == slot;
        const unsigned m = __ballot_sync(0xffffffffu, hit);
        if (lane == 0) warp_n[wp] = __popc(m);
        __syncthreads();
        int before = 0, total = 0;
        for (int w = 0; w < kThreads / 32; ++w) {
            if (w < wp) before += warp_n[w];
            total += warp_n[w];
        }
        if (FILL && hit) {
            const int r = base_s + before + __popc(m & ((1u << lane) - 1u));
            row_tok[r] = t;
            pos[(size_t) t * kPicks + i] = r;
        }
        __syncthreads();
        if (threadIdx.x == 0) base_s += total;
        __syncthreads();
    }
    if (!FILL && threadIdx.x == 0) bounds[slot + 1] = base_s;   // (the counts; the scan makes them bounds)
}

__global__ void bounds_scan_kernel(int* bounds, int n_vs) {   // one thread: bounds[0] = 0, bounds[s + 1] += bounds[s]
    if (threadIdx.x != 0) return;
    bounds[0] = 0;
    for (int s = 0; s < n_vs; ++s) bounds[s + 1] += bounds[s];
}

__global__ void moe_combine_kernel(float* acc, const float* __restrict__ y, const int* __restrict__ pos,
                                   const float* __restrict__ w, int k, int r0, int r1, int dim) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (e >= dim) return;
    float a = acc[(size_t) t * dim + e];
    bool any = false;
    for (int i = 0; i < k; ++i) {
        const int r = pos[(size_t) t * kPicks + i];
        if (r < r0 || r >= r1) continue;
        a = __fmaf_rn(w[(size_t) t * kPicks + i], y[(size_t) (r - r0) * dim + e], a);
        any = true;
    }
    if (any) acc[(size_t) t * dim + e] = a;
}

int cdiv(int a, int b) { return (a + b - 1) / b; }

}  // namespace

namespace {

// ---- the split-KV attention's scratch: a device allocation a (device, stream), made on first use and kept
constexpr int kSplitG = 4, kSplitR = 40, kSplitCap = 24, kSplitMaxT = 5;
bool split_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("DS4_ATTN_SPLIT");
        return e != nullptr && std::atoi(e) != 0;   // opt-in: no gain measured on 4x R9700 (21.8 vs 21.9 tok/s)
    }();
    return on;
}
// DS4_FAST_ATTN: the chunked prompt's packed-KV attention on the matrix cores (sparse_attn_wmma_kernel).  The
// QK/PV sums are the matrix cores' order, so the outputs differ from the scalar kernel's in the last places (the owner
// approved the trade for the prefill; the decode path stays on the scalar kernels).
// 0.2.7: ON by default on gfx1200 / gfx1201 (RDNA4: the R9700, RX 9000), where it was measured and gated (KL 0.0003 against the
// 0.006 gate, ids identical); gfx11 and gfx1151 (Strix Halo) are untested and keep the scalar kernels.  DS4_FAST_ATTN=0 is the
// off switch (exactly the 0.2.6 arithmetic), DS4_FAST_ATTN=1 turns it on elsewhere.  Decided per device (the first call there).
// The Marlin stream measured it ON on 4x R9700 (gfx1201), 2.25 bpw, 8K code prompts: 1180 -> 1354 ppts (+14.7%), the same ids,
// KL 0.10833 against 0.10925 for the scalar path.  gfx1151 on Windows was the untested part; measured on Strix Halo 2026-10-28
// (32K prompt, 2.00 bpw): 101.6 -> 70.8 s (+43.6%), the sparse-attention stage 38.3 -> 7.0 s, the same greedy first token
// (27408,), so it joins the default there and stays opt-in on every other gfx11 part.
#if defined(STRATA_USE_HIP)
bool device_is_gfx12() {
    static std::atomic<int> cache[64];   // 0 unknown, 1 no, 2 yes
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) return false;
    int c = cache[dev].load();
    if (c == 0) {
        cudaDeviceProp prop{};
        const bool yes = cudaGetDeviceProperties(&prop, dev) == cudaSuccess &&
                         (std::strncmp(prop.gcnArchName, "gfx1200", 7) == 0 || std::strncmp(prop.gcnArchName, "gfx1201", 7) == 0);
        c = yes ? 2 : 1;
        cache[dev].store(c);
    }
    return c == 2;
}
#if defined(_WIN32)
bool device_is_gfx1151win() {
    static std::atomic<int> cache[64];   // 0 unknown, 1 no, 2 yes
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) return false;
    int c = cache[dev].load();
    if (c == 0) {
        cudaDeviceProp prop{};
        const bool yes = cudaGetDeviceProperties(&prop, dev) == cudaSuccess && std::strncmp(prop.gcnArchName, "gfx1151", 7) == 0;
        c = yes ? 2 : 1;
        cache[dev].store(c);
    }
    return c == 2;
}
#endif
#endif
bool fast_attn_enabled() {
    static const int env = [] {
        const char* e = std::getenv("DS4_FAST_ATTN");
        return e == nullptr ? -1 : (std::atoi(e) != 0 ? 1 : 0);
    }();
#if defined(STRATA_USE_HIP)
#if defined(_WIN32)
    return strata::ds4::fast_attn_default(env, device_is_gfx12(), device_is_gfx1151win(), strata::ds4::defaults_026());
#else
    return strata::ds4::fast_attn_default(env, device_is_gfx12(), false, strata::ds4::defaults_026());
#endif
#else
    return strata::ds4::fast_attn_default(env, false, false, strata::ds4::defaults_026());
#endif
}
attnsplit::AttnSplitScratch split_scratch(cudaStream_t s, int heads) {
    static std::mutex mu;
    static std::map<std::pair<int, cudaStream_t>, attnsplit::AttnSplitScratch> m;
    int dev = 0;
    cudaGetDevice(&dev);
    std::lock_guard<std::mutex> lk(mu);
    auto it = m.find({dev, s});
    if (it != m.end()) return it->second;
    attnsplit::AttnSplitScratch sc;
    sc.nsplit = kSplitCap;
    sc.max_t = kSplitMaxT;
    const size_t np = (size_t) kSplitMaxT * heads * kSplitCap;
    if (cudaMalloc((void**) &sc.part, np * 512 * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void**) &sc.ml, np * 2 * sizeof(float)) != cudaSuccess) {
        cudaGetLastError();
        if (sc.part != nullptr) cudaFree(sc.part);
        sc = attnsplit::AttnSplitScratch{};
    }
    m[{dev, s}] = sc;
    return sc;
}
}  // namespace

void route_publish(const float* logits, const float* bias, int n, int k, int T, float route_scale,
                   const unsigned long long* vram, const float* x, int dim, int* idx, float* w,
                   unsigned long long* plan_ptr, Req* req, unsigned seq, cudaStream_t s, const float* bias_vl,
                   unsigned vl, int na0, int na1) {
#if defined(STRATA_METAL)
    // selection only: the prefetch schedule (req) is published by the CUDA kernel, and is turned on separately here
    if (strata::dsv41::metal_host::route_publish(logits, bias, n, k, T, route_scale, vram, idx, w, plan_ptr,
            vl != 0 ? bias_vl : nullptr, vl)) {
        // no prefetcher on the Mac path, so publish_kernel has no consumer: the selection is the whole job here
        (void) req; (void) seq; (void) na0; (void) na1; (void) x; (void) dim;
        return;
    }
#endif
    strata_ds4_launch(route_publish_kernel, T, 512, 0, s, logits, bias, n, k, route_scale, vram, x, dim, idx, w, plan_ptr, req,
                                           vl != 0 ? bias_vl : nullptr, vl);
    if (req != nullptr) strata_ds4_launch(publish_kernel, 1, 1, 0, s, req, T, seq, na0, na1);
}
void route_ahead(const float* logits, const float* bias, int n, int m, int T, Req* req, int off, cudaStream_t s) {
#if defined(STRATA_METAL)
    // no prefetch schedule on the Mac path yet: the router publishes no requests, so there is nothing to look ahead for
    (void) logits; (void) bias; (void) n; (void) m; (void) T; (void) req; (void) off; (void) s;
    return;
#endif
    if (req != nullptr && m > 0 && T > 0 && n <= 1024 && off + m <= kAhead)
        strata_ds4_launch(route_ahead_kernel, T, 512, 0, s, logits, bias, n, m, req, off);
}
void wait_seq(const unsigned* seq_ptr, unsigned seq, cudaStream_t s) {
#if defined(STRATA_METAL)
    // nothing is published, so nothing is waited for: the CUDA path spins until the host's prefetcher acknowledges
    (void) seq_ptr; (void) seq; (void) s;
    return;
#endif
    strata_ds4_launch(wait_seq_kernel, 1, 32, 0, s, seq_ptr, seq);
}

// the RAM store's experts for the device (a decode token, window of one): a pick the VRAM table left at 0 whose expert the store holds
// (zc_tab[expert] = the slot's device address over PCIe, 0: not) is copied to staging slot i with wide coalesced loads (the PCIe
// rate; the expert kernels' own small loads over PCIe ran at a tenth of it), then the plan points at the copy
__global__ void zc_copy_kernel(const unsigned long long* __restrict__ plan, const int* __restrict__ idx,
                               const unsigned long long* __restrict__ zc_tab, uint8_t* __restrict__ stage, size_t stride, size_t bytes) {
    const int i = blockIdx.y;
    if (plan[i] != 0) return;
    const unsigned long long p = zc_tab[idx[i]];
    if (p == 0) return;
    const uint4* src = (const uint4*) p;
    uint4* dst = (uint4*) (stage + (size_t) i * stride);
    const size_t n = bytes / 16;
    for (size_t o = (size_t) blockIdx.x * blockDim.x + threadIdx.x; o < n; o += (size_t) gridDim.x * blockDim.x) dst[o] = src[o];
}
__global__ void zc_fix_kernel(unsigned long long* __restrict__ plan, const int* __restrict__ idx, const unsigned long long* __restrict__ zc_tab,
                              int k, uint8_t* __restrict__ stage, size_t stride) {
    const int i = threadIdx.x;
    if (i >= k || plan[i] != 0) return;
    if (zc_tab[idx[i]] != 0) plan[i] = (unsigned long long) (stage + (size_t) i * stride);
}
void zc_stage(unsigned long long* plan, const int* idx, const unsigned long long* zc_tab, int k, uint8_t* stage, size_t stride, size_t bytes,
              cudaStream_t s) {
    strata_ds4_launch(zc_copy_kernel, dim3(64, k), 256, 0, s, plan, idx, zc_tab, stage, stride, bytes);
    strata_ds4_launch(zc_fix_kernel, 1, 32, 0, s, plan, idx, zc_tab, k, stage, stride);
}

void init_pre(float* pre, int T, cudaStream_t s) {
#if defined(STRATA_METAL)
    if (ds4_dbg("DS4_SKIP_INITPRE")) return;
#endif
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::init_pre(pre, T, s)) return;   // the ported MSL kernel
#endif
    strata_ds4_launch(init_pre_kernel, cdiv(T * 8, 128), 128, 0, s, pre, T);
}

void embed(const uint16_t* rows, int T, float* h, int dim, int hc, cudaStream_t s) {
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::embed_addr(rows, T, h, dim, hc)) return;
#endif
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::embed(rows, T, h, dim, hc, s)) return;
#endif
    strata_ds4_launch(embed_kernel, dim3(cdiv(dim, kThreads), T), kThreads, 0, s, rows, h, dim, hc);
}

void finish(float* x, int n_vec, int len, long long stride, const Finish& f, cudaStream_t s) {
    if (len > kFinMax * kThreads) {
        std::fprintf(stderr, "ds4 finish: a vector of %d values is more than %d\n", len, kFinMax * kThreads);
        return;
    }
    if (n_vec <= 0) return;
    if (f.pdst != nullptr) {
        if (f.quant < kKvFp8 || f.quant > kKvFp4p || len % 32 != 0) {
            std::fprintf(stderr, "ds4 finish: packed rows of quant %d, %d values not covered\n", f.quant, len);
            return;
        }
#if defined(STRATA_METAL)
        const auto& mp = *reinterpret_cast<const strata::dsv41::metal_host::FinishIn*>(&f);
        if (strata::dsv41::metal_host::finish(x, n_vec, len, stride, mp)) return;   // finish_kernel<true> (the KV pack)
#endif
        strata_ds4_launch(finish_kernel<true>, n_vec, kThreads, 0, s, x, len, stride, f);
    } else {
#if defined(STRATA_METAL)
        const auto& mf = *reinterpret_cast<const strata::dsv41::metal_host::FinishIn*>(&f);
        if (strata::dsv41::metal_host::finish(x, n_vec, len, stride, mf)) return;   // the ported MSL kernel
#endif
        strata_ds4_launch(finish_kernel<false>, n_vec, kThreads, 0, s, x, len, stride, f);
    }
}

void hc_mix(const float* h, const float* fn, int hcdim, int n_mix, int T, float eps, float* mixes, cudaStream_t s) {
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::hc_mix(h, fn, hcdim, n_mix, T, eps, mixes)) return;
#endif
    strata_ds4_launch(hc_mix_kernel, dim3(n_mix, T), kThreads, 0, s, h, fn, hcdim, eps, mixes);
}
void hc_mix_split(const float* h, const float* fn, int hcdim, int n_mix, int T, float eps, float* mixes, unsigned* cnt,
                  const float* scale, const float* base, int hc, int iters, float hc_eps, float* pre, float* post,
                  float* comb, cudaStream_t s) {
    strata_ds4_launch(hc_mix_split_kernel, dim3(n_mix, T), kThreads, 0, s, h, fn, hcdim, eps, mixes, cnt, scale, base, hc, iters, hc_eps, pre,
                                                            post, comb);
}
void hc_split(const float* mixes, const float* scale, const float* base, int hc, int iters, float hc_eps, int T,
              float* pre, float* post, float* comb, cudaStream_t s) {
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::hc_split(mixes, scale, base, hc, iters, hc_eps, T, pre, post, comb)) return;
#endif
    strata_ds4_launch(hc_split_kernel, T, 32, 0, s, mixes, scale, base, hc, iters, hc_eps, pre, post, comb);
}
void hc_mix_split(const float* h, const float* fn, int hcdim, int n_mix, int T, float eps, float* mixes,
                  const float* scale, const float* base, int hc, int iters, float hc_eps, float* pre, float* post,
                  float* comb, cudaStream_t s) {
#if defined(STRATA_METAL)
    // one dispatch: the mix and the sinkhorn split of its result (bit-identical to the two, see ds4_init.metal)
    if (strata::dsv41::metal_host::hc_mix_split(h, fn, hcdim, n_mix, T, eps, mixes, scale, base, hc, iters, hc_eps,
                                                pre, post, comb))
        return;
#endif
    hc_mix(h, fn, hcdim, n_mix, T, eps, mixes, s);
    hc_split(mixes, scale, base, hc, iters, hc_eps, T, pre, post, comb, s);
}

void hc_pre_norm(const float* h, const float* mix, const float* w, int dim, int hc, int T, float eps, float* x,
                 cudaStream_t s) {
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::hc_pre_norm(h, mix, w, dim, hc, T, eps, x)) return;
#endif
    strata_ds4_launch(hc_pre_norm_kernel, T, 1024, 0, s, h, mix, w, dim, hc, eps, x);
}
void hc_post(const float* y, const float* res, const float* post, const float* comb, int dim, int hc, int T,
             float* out, cudaStream_t s) {
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::hc_post_v(y, res, post, comb, dim, hc, T, out)) return;
#endif
    strata_ds4_launch(hc_post_kernel, dim3(cdiv(dim * hc, kThreads), T), kThreads, 0, s, y, res, post, comb, dim, hc, out);
}

void hc_post_inplace(const float* y, float* h, const float* post, const float* comb, int dim, int hc, int T,
                     cudaStream_t s) {
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::hc_post(y, h, post, comb, dim, hc, T)) return;
#endif
    strata_ds4_launch(hc_post_inplace_kernel, dim3(cdiv(dim, kThreads), T), kThreads, 0, s, y, h, post, comb, dim, hc);
}
void hc_mix_scale(const float* h, int hcdim, int n_mix, int T, float eps, float* mixes, cudaStream_t s) {
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::hc_mix_scale(h, hcdim, n_mix, T, eps, mixes)) return;
#endif
    strata_ds4_launch(hc_mix_scale_kernel, T, kThreads, 0, s, h, hcdim, n_mix, eps, mixes);
}
void engram_dequant_tok(const uint8_t* src, size_t per, int cols, int T, int hd, float* out, cudaStream_t s, int fmt) {
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::engram_dequant_tok(src, per, cols, T, hd, out, fmt)) return;
#endif
    if (T > 0) strata_ds4_launch(engram_dequant_tok_kernel, T * cols, kThreads, 0, s, src, per, cols, hd, out, fmt);
}
void moe_sort(const int* idx, int T, int k, const int* vs, int n_vs, int* bounds, int* row_tok, int* pos,
              cudaStream_t s) {
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::moe_sort(idx, T, k, vs, n_vs, bounds, row_tok, pos)) return;
#endif
    strata_ds4_launch(moe_sort_kernel<false>, n_vs, kThreads, 0, s, idx, T, k, vs, bounds, row_tok, pos);
    strata_ds4_launch(bounds_scan_kernel, 1, 32, 0, s, bounds, n_vs);
    strata_ds4_launch(moe_sort_kernel<true>, n_vs, kThreads, 0, s, idx, T, k, vs, bounds, row_tok, pos);
}
void moe_combine(float* acc, const float* y, const int* pos, const float* w, int T, int k, int r0, int r1, int dim,
                 cudaStream_t s) {
    if (r1 > r0 && T > 0) {
#if defined(STRATA_METAL)
        if (strata::dsv41::metal_host::moe_combine(acc, y, pos, w, T, k, r0, r1, dim)) return;
#endif
        strata_ds4_launch(moe_combine_kernel, dim3(cdiv(dim, kThreads), T), kThreads, 0, s, acc, y, pos, w, k, r0, r1, dim);
    }
}

void gemv_f16(const uint16_t* W, const float* x, int n, int k, int T, float alpha, float* y, cudaStream_t s) {
    strata_ds4_launch(gemv_f16_kernel, dim3(cdiv(n, kThreads / 32), T), kThreads, 0, s, W, x, n, k, alpha, y);
}
void gemv_f16_rows(const uint16_t* W, const float* x, size_t ldx, int n, int k, int T, float alpha, float* y, size_t ldy,
                   cudaStream_t s) {
#if defined(STRATA_METAL)
    // gemv_f16_rows_kernel on the Metal dispatch; the launch below is the fallback
    if (strata::dsv41::metal_host::gemv_f16_rows(W, x, ldx, n, k, T, alpha, y, ldy)) return;
#endif
    if (T > 0)
        strata_ds4_launch(gemv_f16_rows_kernel, dim3(cdiv(n, kThreads / 32), cdiv(T, kRowsT)), kThreads, 0, s, W, x, ldx, n, k, T, alpha,
                                                                                               y, ldy);
}

void gemv_f32(const float* W, const float* x, int n, int k, int T, float* y, int ldy, cudaStream_t s) {
#if defined(STRATA_METAL)
    if (T > 0 && ::strata::dsv41::metal_host::gemv_f32(W, x, n, k, T, y, ldy)) return;
#endif
    if (T > 0) strata_ds4_launch(gemv_f32_kernel, dim3(cdiv(n, kThreads / 32), T), kThreads, 0, s, W, x, n, k, y, ldy);
}

void route(const float* logits, const float* bias, int n, int k, int T, float route_scale, int* idx, float* w,
           cudaStream_t s, const float* bias_vl, const VlBits* vl) {
    static const VlBits none = {};
#if defined(STRATA_METAL)
    // route_kernel on the Metal dispatch; the launch below is the fallback
    if (strata::dsv41::metal_host::route(logits, bias, n, k, T, route_scale, idx, w, bias_vl, vl)) return;
#endif
    strata_ds4_launch(route_kernel, T, 512, 0, s, logits, bias, n, k, route_scale, idx, w, vl != nullptr ? bias_vl : nullptr,
                                   vl != nullptr ? *vl : none);
}

void swiglu(const float* g, const float* u, int n, float lim, float* h, cudaStream_t s) {
#if defined(STRATA_METAL)
    // declared here rather than in the header: the header carries the other entry points, this one is only used here
    namespace mh = strata::dsv41::metal_host;
    if (n > 0 && mh::swiglu(g, u, n, lim, h)) return;
#endif
    if (n > 0) strata_ds4_launch(swiglu_kernel, cdiv(n, kThreads), kThreads, 0, s, g, u, n, lim, h);
}

void compress_win(const float* kv, const float* score, float* kv_state, float* score_state, int ratio, int p0, int T,
                  int d, const float* w, float eps, float* latent, cudaStream_t s) {
#if defined(STRATA_METAL)
    // compress_kernel on the Metal dispatch; the launch below is the fallback
    if (strata::dsv41::metal_host::compress_win(kv, score, kv_state, score_state, ratio, p0, T, d, w, eps, latent)) return;
#endif
    strata_ds4_launch(compress_kernel, 1, 512, 0, s, kv, score, kv_state, score_state, ratio, p0, T, d, w, eps, latent);
}

void index_score(const float* q, const float* w, const void* k, int heads, int hd, int T, int n, const int* cand,
                 int cand_stride, int blk, const Rows& lim, float* out, long long out_stride, cudaStream_t s,
                 int kfmt) {
    if (hd != 128 || heads > 32 || (kfmt != kKvFp4p && kfmt != kKvBf16)) {
        std::fprintf(stderr, "ds4 index_score: %d heads of %d (keys %d) not covered\n", heads, hd, kfmt);
        return;
    }
    if (n <= 0) return;
#if defined(STRATA_METAL)
    // index_score_kernel<128, kfmt> on the Metal dispatch; the launches below are the fallback
    if (strata::dsv41::metal_host::index_score(q, w, (const uint8_t*) k, heads, hd, T, n, cand, cand_stride, blk, &lim,
                                               out, out_stride, kfmt))
        return;
#endif
    const size_t smem = (size_t) (heads * hd + heads) * sizeof(float);
    const dim3 grid(cdiv(n, kThreads), T);
    if (kfmt == kKvFp4p)
        strata_ds4_launch(index_score_kernel<128, kKvFp4p>, grid, kThreads, smem, s, q, w, (const uint8_t*) k, heads, n, cand,
                                                                       cand_stride, blk, lim, out, out_stride);
    else
        strata_ds4_launch(index_score_kernel<128, kKvBf16>, grid, kThreads, smem, s, q, w, (const uint8_t*) k, heads, n, cand,
                                                                       cand_stride, blk, lim, out, out_stride);
}

void topk(const float* vals, long long stride, int T, const Rows& r, const int* cand, int cand_stride, int blk,
          int* out, int out_stride, cudaStream_t s) {
#if defined(STRATA_METAL)
    // topk_kernel on the Metal dispatch; the launch below is the fallback
    if (strata::dsv41::metal_host::topk(vals, stride, T, &r, cand, cand_stride, blk, out, out_stride)) return;
#endif
    strata_ds4_launch(topk_kernel, T, 1024, 0, s, vals, stride, r, cand, cand_stride, blk, out, out_stride);
}

void cand_blocks(const float* score, long long stride, int T, const Rows& r, int blk, float* bm, long long bm_stride,
                 int* out, int out_stride, cudaStream_t s) {
    int nb_max = 1;
    Rows rb = r;
    if (r.gen != 0) {   // (a chunk: the widest query's blocks, made on the device per query)
        rb.gen = 4;
        rb.blk = blk;
        nb_max = cdiv((int) ((r.p0 + T) / r.R), blk);
    } else
        for (int t = 0; t < T; ++t) {
            rb.n[t] = cdiv(r.n[t], blk);
            rb.k[t] = r.k[t];
            nb_max = rb.n[t] > nb_max ? rb.n[t] : nb_max;
        }
#if defined(STRATA_METAL)
    // block_max_kernel on the Metal dispatch; the launch below is the fallback.  The selection of the best blocks (topk) follows either way:
    // returning here skipped it, the candidate list stayed unwritten and every context past 16K tokens (4,096 compressed rows) read garbage.
    if (!strata::dsv41::metal_host::block_max(score, stride, T, &r, blk, bm, bm_stride, nb_max))
#endif
    strata_ds4_launch(block_max_kernel, dim3(cdiv(nb_max, kThreads), T), kThreads, 0, s, score, stride, r, blk, bm, bm_stride);
    topk(bm, bm_stride, T, rb, nullptr, 0, 1, out, out_stride, s);
}

// the multiprocessors of the current device (cached a device)
static int sm_count() {
    static std::atomic<int> cached[64];
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) dev = 0;
    int v = cached[dev].load(std::memory_order_relaxed);
    if (v <= 0) {
        if (cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || v <= 0) v = 1;
        cached[dev].store(v, std::memory_order_relaxed);
    }
    return v;
}

void sparse_attn(const float* q, const uint16_t* win, const void* ckv, const int* sel, int sel_stride,
                 const AttnRows& a, const float* sink, int heads, int d, int T, float scale, float* out, cudaStream_t s,
                 int ckv_fmt, int heads_per_block, const AttnScratch* scr) {
    if (d != 512 || (ckv_fmt != kKvFp4 && ckv_fmt != kKvBf16)) {
        std::fprintf(stderr, "ds4 sparse_attn: head dim %d (rows %d) not covered\n", d, ckv_fmt);
        return;
    }
#if defined(STRATA_METAL)
    // sparse_attn_packed_kernel<G> / sparse_attn_kernel<false> on the Metal dispatch; the launches below are the fallback
    if (strata::dsv41::metal_host::sparse_attn(q, win, ckv, sel, sel_stride, &a, sink, heads, d, T, scale, out, ckv_fmt))
        return;
#endif
    if (ckv_fmt == kKvBf16) {
        strata_ds4_launch(sparse_attn_kernel<false>, dim3(heads, T), kThreads, 0, s, q, win, (const uint16_t*) ckv, sel, sel_stride, a,
                                                                       sink, heads, d, scale, out);
        return;
    }
    // a decode step / short verify window of packed rows: the split-KV kernel (Strata's CUDA lab, ds4_attn_split.cuh; 2.9x at
    // one query on an A4000).  The same inputs; the merge sums the chunks in another order, so the bits differ in the last
    // place (4e-7 against a double reference).  DS4_ATTN_SPLIT=1 turns it on (off by default: no measurable decode gain on the R9700 box, the decode is bound elsewhere).
    if (ckv_fmt == kKvFp4 && a.gen == 0 && T >= 1 && T <= kSplitMaxT && heads % kSplitG == 0 && split_enabled()) {
        int total_max = 0;
        for (int t = 0; t < T; ++t) total_max = std::max(total_max, a.nwin[t] + a.nsel[t]);
        if (total_max > 0 && total_max <= kSplitCap * kSplitR) {
            const attnsplit::AttnSplitScratch sc = split_scratch(s, heads);
            if (sc.part != nullptr) {
                attnsplit::sparse_attn_split<kSplitG, kSplitR>(q, win, (const uint8_t*) ckv, kv_row_bytes(kKvFp4, d), sel,
                                                               sel_stride, a, sink, heads, T, scale, out, sc, total_max, s);
                return;
            }
        }
    }
    // a few queries (a decode step, a short window): their selected rows decoded once a query into the scratch, then
    // the BF16 rows' kernel itself (heads_per_block -1: this way at any T)
    const bool few = heads_per_block == -1 || (heads_per_block == 0 && T <= kAttnScratchT);
    if (few && scr != nullptr && scr->rows != nullptr && T <= scr->max_t && sel_stride <= scr->cap) {
#if defined(STRATA_METAL)
        if (strata::dsv41::metal_host::unpack_sel(ckv, sel, sel_stride, a.gen, a.p0, a.t0, a.R, a.kcap, a.nsel,
                scr->cap, scr->rows, T, a.gen ? 1 : (T < kWin ? T : kWin), kv_row_bytes(kKvFp4, 512))) return;
#endif
        strata_ds4_launch(unpack_sel_kernel, dim3(cdiv(sel_stride, kStage), T), kThreads, 0, s, (const uint8_t*) ckv, sel, sel_stride, a,
                                                                               scr->cap, scr->rows);
#if !defined(STRATA_USE_HIP)
        if (T == 1)   // (V100: the loads ahead pay for the decode's kernel at one query, not at more)
            strata_ds4_launch(sparse_attn_kernel<true, 8>, dim3(heads, T), kThreads, 0, s, q, win, scr->rows, nullptr, scr->cap, a, sink,
                                                                             heads, d, scale, out);
        else
#endif
            strata_ds4_launch(sparse_attn_kernel<true, 0>, dim3(heads, T), kThreads, 0, s, q, win, scr->rows, nullptr, scr->cap, a, sink,
                                                                             heads, d, scale, out);
        return;
    }
    // DS4_FAST_ATTN=1: the chunked prompt's matrix-core path (16 heads a block, a 16-key tile).  The decode and the
    // verify windows keep the scalar kernels: one query's heads are not the matrix's rows there.
#if defined(STRATA_USE_HIP)
    if (fast_attn_enabled() && a.gen != 0 && T >= kFaKeys && heads % kFaHeads == 0) {
        strata_ds4_launch(sparse_attn_wmma_kernel, dim3(heads / kFaHeads, T), kThreads, 0, s, q, win, (const uint8_t*) ckv, sel, sel_stride,
                                                                              a, sink, heads, scale, out);
        return;
    }
#endif
    // packed rows: as many heads a block as still leaves two blocks a multiprocessor (a decoded row serves them all)
    int G = heads_per_block;    if (G != 1 && G != 2 && G != 4 && G != 8) {
        G = 8;
        while (G > 1 && (heads % G != 0 || (heads / G) * T < 2 * sm_count())) G /= 2;
    }
    if (heads % G != 0) G = 1;
    const dim3 grid(heads / G, T);
    const uint8_t* c = (const uint8_t*) ckv;
    if (G == 8) strata_ds4_launch(sparse_attn_packed_kernel<8>, grid, kThreads, 0, s, q, win, c, sel, sel_stride, a, sink, heads, scale, out);
    else if (G == 4) strata_ds4_launch(sparse_attn_packed_kernel<4>, grid, kThreads, 0, s, q, win, c, sel, sel_stride, a, sink, heads, scale, out);
    else if (G == 2) strata_ds4_launch(sparse_attn_packed_kernel<2>, grid, kThreads, 0, s, q, win, c, sel, sel_stride, a, sink, heads, scale, out);
    else strata_ds4_launch(sparse_attn_packed_kernel<1>, grid, kThreads, 0, s, q, win, c, sel, sel_stride, a, sink, heads, scale, out);
}

void ring_to_ext(const uint8_t* ring, uint16_t* ext, int64_t p0, int n, int d, cudaStream_t s) {
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::ring_to_ext(ring, ext, (long long) p0, n, d, kv_row_bytes(kKvFp8, d))) return;
#endif
    strata_ds4_launch(ring_to_ext_kernel, n, kThreads, 0, s, ring, ext, (long long) p0, n, d);
}
void ext_to_ring(const uint16_t* ext, uint8_t* ring, int64_t p0, int n, int T, int d, cudaStream_t s, uint8_t* backup) {
    if (d % 32 != 0) {
        std::fprintf(stderr, "ds4 ext_to_ring: row of %d not covered\n", d);
        return;
    }
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::ext_to_ring(ext, ring, (long long) p0, n, T, d, kv_row_bytes(kKvFp8, d), backup)) return;
#endif
    strata_ds4_launch(ext_to_ring_kernel, T, kThreads, 0, s, ext, ring, (long long) p0, n, T, d, backup);
}
void ring_restore(uint8_t* ring, const uint8_t* backup, int64_t p0, int n, int keep, int T, int d, cudaStream_t s) {
#if defined(STRATA_METAL)
    {   // (row copies on the context queue: the drafter's rollback is a few rows)
        const size_t rb = kv_row_bytes(kKvFp8, d);
        for (int t = keep; t < T; ++t) cudaMemcpyAsync(ring + (size_t) ((p0 + t) % n) * rb, backup + (size_t) t * rb, rb, cudaMemcpyDeviceToDevice, s);
        return;
    }
#endif
    if (T > keep)
        strata_ds4_launch(ring_restore_kernel, T - keep, kThreads, 0, s, ring, backup, (long long) p0, n, keep, kv_row_bytes(kKvFp8, d));
}
void ring_restore_batch(const RingBatch& b, int64_t p0, int n, int keep, int T, int d, cudaStream_t s) {
#if defined(STRATA_METAL)
    {
        const size_t rb = kv_row_bytes(kKvFp8, d);
        for (int li = 0; li < b.n; ++li)
            for (int t = keep; t < T; ++t)
                cudaMemcpyAsync(b.ring[li] + (size_t) ((p0 + t) % n) * rb, b.bak[li] + (size_t) t * rb, rb, cudaMemcpyDeviceToDevice, s);
        return;
    }
#endif
    if (T > keep && b.n > 0)
        strata_ds4_launch(ring_restore_batch_kernel, dim3(T - keep, b.n), kThreads, 0, s, b, (long long) p0, n, keep,
                                                                             kv_row_bytes(kKvFp8, d));
}
void hc_mean(const float* h, int T, int hc, int dim, float* out, long long out_stride, cudaStream_t s) {
#if defined(STRATA_METAL)
    if (ds4_dbg("DS4_SKIP_HCMEAN")) return;
#endif
#if defined(STRATA_METAL)
    // hc_mean_kernel on the Metal dispatch; the launch below is the fallback
    if (strata::dsv41::metal_host::hc_mean(h, T, hc, dim, out, out_stride)) return;
#endif
    strata_ds4_launch(hc_mean_kernel, dim3(cdiv(dim, kThreads), T), kThreads, 0, s, h, hc, dim, out, out_stride);
}
void iota_rows(int* out, int T, int n, int stride, cudaStream_t s) {
#if defined(STRATA_METAL)
    // iota_rows_kernel on the Metal dispatch; the launch below is the fallback
    if (strata::dsv41::metal_host::iota_rows(out, T, n, stride)) return;
#endif
    strata_ds4_launch(iota_rows_kernel, T, 64, 0, s, out, n, stride);
}
void markov_add(float* logits, const uint16_t* E, const uint16_t* H, const int* tok, int V, int R, cudaStream_t s) {
#if defined(STRATA_METAL)
    // markov_kernel on the Metal dispatch (the drafter's Markov bias); the launch below is the fallback
    if (strata::dsv41::metal_host::markov_add(logits, E, H, tok, V, R)) return;
#endif
    strata_ds4_launch(markov_kernel, cdiv(V, kThreads / 32), kThreads, (size_t) R * sizeof(float), s, logits, E, H, tok, V, R);
}
void markov_add_q8(float* logits, const uint16_t* E, const uint8_t* Hq, const int* tok, int V, int R, cudaStream_t s) {
    strata_ds4_launch(markov_kernel_q8, cdiv(V, kThreads / 32), kThreads, (size_t) R * sizeof(float), s, logits, E, Hq, tok, V, R);
}
void markov_share(float* logits, const uint16_t* H, const uint16_t* erow, int vn, int R, cudaStream_t s) {
    strata_ds4_launch(markov_share_kernel, cdiv(vn, kThreads / 32), kThreads, (size_t) R * sizeof(float), s, logits, H, erow, vn, R);
}
void argmax_share(const float* row, int vn, int v0, unsigned* slot, unsigned seq, cudaStream_t s) {
    strata_ds4_launch(argmax_share_kernel, 1, 1024, 0, s, row, vn, v0, slot, seq);
}
void markov_pick(const unsigned* slots, int nm, unsigned seq, int* tok, const uint16_t* E, uint16_t* erow, int R,
                 cudaStream_t s) {
    strata_ds4_launch(markov_pick_kernel, 1, 256, 0, s, slots, nm, seq, tok, E, erow, R);
}
void confidence(const float* x, const uint16_t* E, const int* tok, const float* w, int T, int dim, int R, float* out,
                cudaStream_t s) {
#if defined(STRATA_METAL)
    // confidence_kernel on the Metal dispatch; the launch below is the fallback
    if (strata::dsv41::metal_host::confidence(x, E, tok, w, T, dim, R, out)) return;
#endif
    strata_ds4_launch(confidence_kernel, T, 1024, 0, s, x, E, tok, w, dim, R, out);
}

void engram_dequant(const uint8_t* rows, const uint8_t* scales, int n, int hd, float* out, cudaStream_t s, int fmt) {
#if defined(STRATA_METAL)
    // engram_dequant_kernel on the Metal dispatch (fmt 2, this model's IQ4_NL tables); the launch below is the fallback
    if (strata::dsv41::metal_host::engram_dequant(rows, scales, n, hd, out, fmt)) return;
#endif
    strata_ds4_launch(engram_dequant_kernel, n, kThreads, 0, s, rows, scales, hd, out, fmt);
}

void engram_gate(float* h, const float* kv, const float* w, int hc, int dim, int T, float eps, cudaStream_t s) {
#if defined(STRATA_METAL)
    // engram_gate_kernel on the Metal dispatch; the launch below is the fallback
    if (strata::dsv41::metal_host::engram_gate(h, kv, w, hc, dim, T, eps)) return;
#endif
    strata_ds4_launch(engram_gate_kernel, dim3(hc, T), 1024, 0, s, h, kv, w, hc, dim, eps);
}

void gather_rows(const float* x, const Gather& g, int n, int cols, float* out, cudaStream_t s) {
#if defined(STRATA_METAL)
    // gather_kernel on the Metal dispatch; the launch below is the fallback
    if (strata::dsv41::metal_host::gather_rows(x, &g, n, cols, out)) return;
#endif
    if (n > 0) strata_ds4_launch(gather_kernel, dim3(cdiv(cols, kThreads), n), kThreads, 0, s, x, g, cols, out);
}

void combine(const float* ed, const float* cpu, const float* sh, const Combine& c, int T, int dim, float* out,
             cudaStream_t s) {
#if defined(STRATA_METAL)
    // combine_kernel on the Metal dispatch; the launch below is the fallback
    if (strata::dsv41::metal_host::combine(ed, cpu, sh, &c, T, dim, out)) return;
#endif
    strata_ds4_launch(combine_kernel, dim3(cdiv(dim, kThreads), T), kThreads, 0, s, ed, cpu, sh, c, dim, out);
}

void xfer_put(const Xfer& x, unsigned* flag, unsigned seq, unsigned* cnt, cudaStream_t s) {
    size_t total = 0;
    for (int i = 0; i < x.n; ++i) total += x.bytes[i] / 16;
    const int blocks = (int) std::max<size_t>(1, std::min<size_t>(64, (total + kThreads - 1) / kThreads));
    strata_ds4_launch(xfer_put_kernel, blocks, kThreads, 0, s, x, flag, seq, cnt);
    // (another GPU, or the host, waits for this flag: on Windows the put must not sit in the queue - #303: two RTX 3080s
    //  hung at the first hand-off, the receiver spinning on a flag whose writer had never been submitted)
    win_submit(s);
}
void win_submit(cudaStream_t s) {
#if defined(_WIN32)
    static const bool on = [] {
        const char* v = std::getenv("DS4_WIN_SUBMIT");
        return v == nullptr || std::atoi(v) != 0;
    }();
    if (on) {
        (void) cudaStreamQuery(s);
        (void) cudaGetLastError();
    }
#else
    (void) s;
#endif
}
void xfer_get(const Xfer& x, const unsigned* flag, unsigned seq, cudaStream_t s) {
    size_t total = 0;
    for (int i = 0; i < x.n; ++i) total += x.bytes[i] / 16;
    const int blocks = (int) std::max<size_t>(1, std::min<size_t>(64, (total + kThreads - 1) / kThreads));
    strata_ds4_launch(xfer_get_kernel, blocks, kThreads, 0, s, x, flag, seq);
}
void wait_flags(const unsigned* flags, int n, int stride, unsigned seq, cudaStream_t s) {
    if (n > 0) strata_ds4_launch(wait_flags_kernel, 1, 32, 0, s, flags, n, stride, seq);
}
void set_flag(unsigned* flag, unsigned v, cudaStream_t s) {
    strata_ds4_launch(set_flag_kernel, 1, 1, 0, s, flag, v);
    win_submit(s);   // (the writer on another GPU waits for this acknowledgement before its next put - #384)
}
void allreduce(float* x, int n, const ArGroup& a, int mi, unsigned seq, unsigned* cnt, cudaStream_t s) {
    const int slot = (int) (seq % (unsigned) a.slots);
    if (a.nm > 2 && a.sums != nullptr && a.bflags != nullptr) {   // (the reduce-scatter: ar_rs_kernel)
        static const bool f16 = [] {
            const char* e = std::getenv("DS4_AR_F16");
            return e != nullptr && std::atoi(e) != 0;   // opt-in: no gain measured on 4x R9700 (21.8 vs 21.9 tok/s)   // (on unless DS4_AR_F16=0)
        }();
        const int n4 = n / 4;
        const int blocks = std::max(1, std::min(kArBlocks, cdiv(n4, kThreads)));
        if (f16)
            strata_ds4_launch(ar_rs16_kernel, blocks, kThreads, 0, s, x, n4, a.data, a.sums, a.stride, a.bflags, a.nm, mi, slot, a.slots, seq);
        else
            strata_ds4_launch(ar_rs_kernel, blocks, kThreads, 0, s, x, n4, a.data, a.sums, a.stride, a.bflags, a.nm, mi, slot, a.slots, seq);
        return;
    }
    Xfer xp;
    xp.n = 1;
    xp.src[0] = x;
    xp.dst[0] = a.data + ((size_t) mi * a.slots + slot) * a.stride;
    xp.bytes[0] = (size_t) n * 4;
    xfer_put(xp, a.flags + ((size_t) mi * a.slots + slot) * 32, seq, cnt, s);
    const int n4 = n / 4;
    const int blocks = std::max(1, std::min(64, cdiv(n4, kThreads)));
    strata_ds4_launch(ar_sum_kernel, blocks, kThreads, 0, s, x, n4, a.data, a.stride, a.flags, a.nm, mi, slot, a.slots, seq);
}

}  // namespace strata::ds4
