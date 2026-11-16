// src/prefill/moe_f16_common.cuh - the parts the gfx11 F16 prompt-expert kernels share (STRATA_PF_F16=1): the
// switch, the work item's width, the int8 activations staged as F16, the WMMA over the token tiles and the two
// epilogues.  moe_f16_kq.cu (the K-quant kernels) and moe_fused_iq.cu (the other native formats) include it.
//
// A work item is 128 weight rows (gate/up: the 64 gate and 64 up rows of 64 features; down: 128 output rows) against
// one 64-row tile of fused::group or two consecutive tiles of one expert; each of the eight waves owns 16 of the
// weight rows.  The F16 WMMA accumulates all of K in F32.  The fragment loop and the epilogues' layouts follow gufo's
// RoutedF16GEMMKernel.
//
// Adapted from gufo 97b8645f: src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp (RoutedF16GEMMKernel).
// MIT License
// Copyright (c) 2026 gufo contributors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include <cstdint>
#include <cstdlib>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "strata/prefill/moe_fused.hpp"

#if defined(__HIPCC__) && (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
                           defined(__gfx1150__) || defined(__gfx1151__))
#define STRATA_F16_DEVICE 1
#else
#define STRATA_F16_DEVICE 0
#endif

namespace strata::prefill::fused::f16 {

constexpr int AB = 80;           // activation / H bytes per 64 values: 64 codes, {d0, d1}, {sum0, sum1}
constexpr int THREADS = 256;
constexpr int ROWS = 128;        // weight rows per work item
constexpr int GU_NFB = 640 / 64, D_NFB = 2560 / ROWS;   // work items per tile: 64 features, 128 output rows
constexpr int GU_ACT_LD = 2560 / 64 * AB, H_LD = 640 / 64 * AB;   // bytes per activation / H row

/// STRATA_PF_F16=1 (on top of STRATA_PF_FUSED=1): the covered formats take the F16 kernels on gfx11.
inline bool requested() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_PF_F16");
        return v != nullptr && v[0] == '1';
    }();
    return on;
}

/// Two tiles a work item when an expert has a tile's rows or more on average.  gfx1151, Q4_K, top 10 over 512
/// experts, one 128-expert launch, ms (64 / 128): 1024 tokens 1.22 / 1.28, 2048 1.51 / 1.64, 4096 2.73 / 2.48, 8192
/// 4.76 / 4.59, 16384 9.19 / 8.46; Q5_1 down 2048 1.02 / 1.04, 4096 1.61 / 1.48, 16384 5.29 / 4.83.
/// STRATA_PF_F16_BN=64|128 forces one.
inline int pick_bn(int64_t n, int n_expert) {
    static const int forced = [] {
        const char* v = std::getenv("STRATA_PF_F16_BN");
        const int t = v ? std::atoi(v) : 0;
        return t == kTileRows || t == 2 * kTileRows ? t : 0;
    }();
    if (forced) return forced;
    return (double) n / (n_expert > 0 ? n_expert : 1) >= kTileRows ? 2 * kTileRows : kTileRows;
}

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f wmma_f16(v16h a, v16h b, v8f c) {
#if STRATA_F16_DEVICE
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
    __builtin_trap();
    return c;
#endif
}

/// The work item of block `w`: its tile (fused::group's tables), whether it is the second tile of a pair (the first
/// takes it), and its rows.  Returns false when the block has no work.
template <int BN, int NFB>
__device__ __forceinline__ bool work_item(const Batch& b, const int32_t* ts, const int2* tiles, const int32_t* off, int w,
                                          int& fb, int& e, int& row0, int& nrows) {
    static_assert(BN == kTileRows || BN == 2 * kTileRows, "one tile or a pair");
    const int t0 = ts[b.e0];
    if (w >= (ts[b.e1] - t0) * NFB) return false;
    const int2 tl = tiles[t0 + w / NFB];
    fb = w % NFB;
    e = tl.x;
    row0 = tl.y;
    if (BN > kTileRows && ((t0 + w / NFB - ts[e]) & 1)) return false;
    nrows = min(BN, off[e + 1] - row0);
    return true;
}

/// The int8 activation (or H) rows of a work item, staged as F16 per stage: BN rows x 8 chunks of 8 values, chunk
/// sub = kb * 4 + quarter, into LDS [sub][row] (stride BN + 1) - the B fragments' layout of the F16 WMMA.  A row past
/// the item's end is zero.
template <int BN> struct ActStage {
    static constexpr int FETCH = BN * 8 / THREADS, STRIDE = BN + 1, BYTES = 2 * 4 * STRIDE * 16;
    const uint8_t* src[FETCH];
    int slot[FETCH], sub[FETCH];
    uint2 codes[FETCH];
    float d[FETCH];
    // rows[t]: the row of item row t in `base` (rows of `ld` bytes), or -1
    __device__ __forceinline__ void init(const uint8_t* base, size_t ld, const int* rows, int tid) {
#pragma unroll
        for (int i = 0; i < FETCH; ++i) {
            const int chunk = tid + i * THREADS, t = chunk >> 3, c = chunk & 7;
            src[i] = rows[t] >= 0 ? base + (size_t) rows[t] * ld : nullptr;
            slot[i] = c * STRIDE + t;
            sub[i] = c;
        }
    }
    __device__ __forceinline__ void fetch(int s) {
#pragma unroll
        for (int i = 0; i < FETCH; ++i) {
            if (src[i] != nullptr) {
                const uint8_t* p = src[i] + (size_t) s * AB;
                codes[i] = *reinterpret_cast<const uint2*>(p + 8 * sub[i]);
                d[i] = *reinterpret_cast<const float*>(p + 64 + 4 * (sub[i] >> 2));
            } else {
                codes[i] = make_uint2(0u, 0u);
                d[i] = 0.0f;
            }
        }
    }
    __device__ __forceinline__ void commit(uint4* s_act) const {
#pragma unroll
        for (int i = 0; i < FETCH; ++i) {
            const uint32_t ws[2] = {codes[i].x, codes[i].y};
            __half2 h[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t wd = ws[j >> 1] >> (16 * (j & 1));
                const float lo = (float) (int8_t) (wd & 0xFFu), hi = (float) (int8_t) ((wd >> 8) & 0xFFu);
                h[j] = __floats2half2_rn(lo * d[i], hi * d[i]);
            }
            s_act[slot[i]] = make_uint4(__builtin_bit_cast(uint32_t, h[0]), __builtin_bit_cast(uint32_t, h[1]),
                                        __builtin_bit_cast(uint32_t, h[2]), __builtin_bit_cast(uint32_t, h[3]));
        }
    }
};

/// Two F16 WMMAs (K 32) of a wave's 16 weight rows against each live 16-row tile of the staged activations.
template <int BN>
__device__ __forceinline__ void mma_tiles(v8f (&acc)[BN / 16], v16h a_lo, v16h a_hi, const uint4* s_act, int kb, int sub_lane,
                                          int live_tiles) {
    constexpr int STRIDE = ActStage<BN>::STRIDE;
#pragma unroll
    for (int j = 0; j < BN / 16; ++j) {
        asm volatile("" ::: "memory");   // one token tile's fragments live at a time (gufo: hoisting spills)
        if (j >= live_tiles) continue;
        const uint4* frag = s_act + (kb * 4) * STRIDE + j * 16 + sub_lane;
        uint4 bq[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) bq[q] = frag[q * STRIDE];
        v16h b_lo, b_hi;
        __builtin_memcpy(&b_lo, &bq[0], 32);
        __builtin_memcpy(&b_hi, &bq[2], 32);
        acc[j] = wmma_f16(a_lo, b_lo, acc[j]);
        acc[j] = wmma_f16(a_hi, b_hi, acc[j]);
    }
}

constexpr int GU_EPI_STRIDE = 18, GU_EPI_PLANE = 16 * GU_EPI_STRIDE;
constexpr int GU_EPI_BYTES = 8 * GU_EPI_PLANE * 4, D_EPI_BYTES = 8 * 256 * 4;

/// Gate/up epilogue (every wave; `lds` at least GU_EPI_BYTES, free): per token tile, each wave's 16 x 16
/// accumulators as plane [token][row] (waves 0-3 the gate rows, 4-7 the up rows), then a thread per (token, half of
/// the 64 features): SwiGLU, H as int8 per 32 features with its scale and the codes' sum (native_w11_kernel's H).
template <int BN>
__device__ __forceinline__ void gu_epilogue(const v8f (&acc)[BN / 16], uint8_t* lds, int tid, int live_tiles, int nrows, int row0,
                                            int fb, uint8_t* __restrict__ out) {
    const int wave = tid >> 5, lane = tid & 31, sub_lane = lane & 15, half_id = lane >> 4;
    float* base = reinterpret_cast<float*>(lds);
    float* scratch = base + wave * GU_EPI_PLANE;
#pragma unroll
    for (int j = 0; j < BN / 16; ++j) {
        if (j >= live_tiles) break;   // uniform
#pragma unroll
        for (int l = 0; l < 8; ++l) scratch[sub_lane * GU_EPI_STRIDE + 2 * l + half_id] = acc[j][l];
        __syncthreads();
        if (tid < 32) {
            const int t = tid >> 1, hh = tid & 1, r = j * 16 + t;
            if (r < nrows) {
                float hv[32];
                float am = 0.0f;
#pragma unroll
                for (int f = 0; f < 32; ++f) {
                    const int ft = 32 * hh + f;
                    const int idx = (ft >> 4) * GU_EPI_PLANE + t * GU_EPI_STRIDE + (ft & 15);
                    const float gt = base[idx], up = base[idx + 4 * GU_EPI_PLANE];
                    hv[f] = gt / (1.0f + __expf(-gt)) * up;
                    am = fmaxf(am, fabsf(hv[f]));
                }
                const float inv = am > 0.0f ? 127.0f / am : 0.0f;
                uint8_t* o = out + (size_t) (row0 + r) * H_LD + (size_t) fb * AB;
                uint32_t wd[8] = {};
                int csum = 0;
#pragma unroll
                for (int f = 0; f < 32; ++f) {
                    const int c = __float2int_rn(hv[f] * inv);
                    csum += c;
                    wd[f >> 2] |= (uint32_t) (uint8_t) (int8_t) c << (8 * (f & 3));
                }
                uint4* o4 = reinterpret_cast<uint4*>(o + 32 * hh);
                o4[0] = make_uint4(wd[0], wd[1], wd[2], wd[3]);
                o4[1] = make_uint4(wd[4], wd[5], wd[6], wd[7]);
                *reinterpret_cast<float*>(o + 64 + 4 * hh) = am / 127.0f;
                *reinterpret_cast<float*>(o + 72 + 4 * hh) = (float) csum;
            }
        }
        __syncthreads();
    }
}

/// Down epilogue (`lds` at least D_EPI_BYTES, free): each wave's 16 x 16 tile through LDS ([token][row]), then 16
/// consecutive FP32 rows per routed row into dm [routed row][2560].
template <int BN>
__device__ __forceinline__ void down_epilogue(const v8f (&acc)[BN / 16], uint8_t* lds, int tid, int live_tiles, int nrows, int row0,
                                              int fb, float* __restrict__ dm) {
    const int wave = tid >> 5, lane = tid & 31, sub_lane = lane & 15, half_id = lane >> 4;
    float* tile = reinterpret_cast<float*>(lds) + wave * 256;
    const int r0 = fb * ROWS + wave * 16;
#pragma unroll
    for (int j = 0; j < BN / 16; ++j) {
        if (j >= live_tiles) break;   // uniform
#pragma unroll
        for (int l = 0; l < 8; ++l) tile[sub_lane * 16 + 2 * l + half_id] = acc[j][l];
        __builtin_amdgcn_wave_barrier();
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            const int flat = q * 32 + lane, t = j * 16 + (flat >> 4);
            if (t < nrows) dm[(size_t) (row0 + t) * 2560 + r0 + (flat & 15)] = tile[flat];
        }
        __builtin_amdgcn_wave_barrier();
    }
}

}  // namespace strata::prefill::fused::f16
