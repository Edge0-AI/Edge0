// src/prefill/moe_f16_kq.cu - see moe_f16_kq.hpp.  gfx11 only (opt-in STRATA_PF_F16=1 with STRATA_PF_FUSED_KQ=1).
//
// gufo's routed-expert kernels for the K-quants, on Strata's tables, int8 activations and H (the work item, the
// activation staging and the epilogues are moe_f16_common.cuh's):
//   - gate/up (Q4_K, Q5_K): the weight codes stay packed in LDS (one 32-byte nibble group per row and stage) and
//     each wave turns its own 16 rows into F16 right after the LDS read: w = q * (d sc) - dmin m as one packed FMA;
//   - down (Q5_1): a K block's 16 code bytes and its high-bit word in LDS, w = q d + m per wave.
// The output differs from native_w11_kernel's (F16 weights instead of int8 sub-blocks), like STRATA_PF_FUSED_KQ's
// does from the MMQ path.
//
// Adapted from gufo 97b8645f: src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp
// (RoutedF16GEMMKernel and its helpers).
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
#include "moe_f16_kq.hpp"

#include "moe_f16_common.cuh"

#include "ggml.h"

#include <algorithm>

namespace strata::prefill::fused::f16kq {
namespace {

using namespace f16;

constexpr int T_Q4_K = GGML_TYPE_Q4_K, T_Q5_K = GGML_TYPE_Q5_K;
constexpr int GU_NS = 2560 / 64, D_NS = 640 / 64;   // stages of 64 values
constexpr int FEAT = 64;                             // features (gate rows = up rows) per gate/up item

// Byte i of the 16-byte K-quant block header (d, dmin, scales[12]).
__device__ __forceinline__ uint32_t header_byte(const uint4& h, uint32_t i) {
    const uint32_t word = i < 4 ? h.x : i < 8 ? h.y : i < 12 ? h.z : h.w;
    return (word >> (8U * (i & 3U))) & 0xFFU;
}
// Bit s of each of the four bytes of w, packed into bits 0-3.
__device__ __forceinline__ uint32_t gather_bit(uint32_t w, int s) { return (((w >> s) & 0x01010101U) * 0x01020408U) >> 24U; }
// Bits 0-3 of `bits` to bit 4 of each byte.
__device__ __forceinline__ uint32_t spread_high_bits(uint32_t bits) { return (__umul24(bits, 0x00204081U) & 0x01010101U) << 4U; }

constexpr uint32_t HALF_MAGIC = 0x64646464U;   // the high bytes of 1024.0 as F16
// Four code bytes (< 32) to four halves: 1024 + q exactly, minus 1024, then q * scale + bias.
__device__ __forceinline__ void codes_to_halves(uint32_t codes, __half2 magic, __half2 scale2, __half2 bias2, __half2& lo,
                                                __half2& hi) {
#if STRATA_F16_DEVICE
    const uint32_t p0 = __builtin_amdgcn_perm(codes, HALF_MAGIC, 0x01050004U);
    const uint32_t p1 = __builtin_amdgcn_perm(codes, HALF_MAGIC, 0x03070206U);
    lo = __hfma2(__hadd2(__builtin_bit_cast(__half2, p0), magic), scale2, bias2);
    hi = __hfma2(__hadd2(__builtin_bit_cast(__half2, p1), magic), scale2, bias2);
#else
    (void) codes; (void) magic; (void) scale2; (void) bias2; lo = hi = __half2{};
#endif
}

// Gate/up (Q4_K / Q5_K): code rows 0..63 are the item's gate rows, 64..127 its up rows; a thread pair per row.
template <int WT, int BN>
__global__ void __launch_bounds__(THREADS) gate_up_kernel(const Batch b, const NativeGeom geo, const int32_t* __restrict__ ts,
                                                          const int2* __restrict__ tiles, const int32_t* __restrict__ off,
                                                          const uint8_t* __restrict__ act, const int32_t* __restrict__ src,
                                                          uint8_t* __restrict__ out) {
#if STRATA_F16_DEVICE
    constexpr bool Q5K = WT == T_Q5_K;
    constexpr int BLOCK_CHUNKS = Q5K ? 11 : 9;   // 16-byte chunks per 256-value block
    constexpr int CODE_CHUNK = Q5K ? 3 : 1;      // the first chunk of qs
    // LDS (bytes): codes ROWS x 2 chunks (swizzled); Q5_K's high bits and the (scale, bias) half2 per row and K
    // block; the activations.  The epilogue reuses the front.
    constexpr int CODE_BYTES = ROWS * 2 * 16;
    constexpr int HIGH_BYTES = Q5K ? 2 * ROWS * 4 : 0;
    constexpr int SCALE_BYTES = 2 * ROWS * 4;
    constexpr int STAGE_BYTES = CODE_BYTES + HIGH_BYTES + SCALE_BYTES + ActStage<BN>::BYTES;
    static_assert(GU_EPI_BYTES <= STAGE_BYTES, "the epilogue fits the stage");
    __shared__ __attribute__((aligned(16))) uint8_t lds[STAGE_BYTES];
    __shared__ int srow[BN];
    auto* s_codes = reinterpret_cast<uint4*>(lds);
    auto* s_high = reinterpret_cast<uint32_t*>(lds + CODE_BYTES);
    auto* s_scale = reinterpret_cast<uint32_t*>(lds + CODE_BYTES + HIGH_BYTES);
    auto* s_act = reinterpret_cast<uint4*>(lds + CODE_BYTES + HIGH_BYTES + SCALE_BYTES);

    const int tid = threadIdx.x, wave = tid >> 5, sub_lane = tid & 15;
    const int f_c = tid & 1;   // this thread's chunk (and K block) of its weight row
    const auto swizzle = [](int row, int c) { return row * 2 + (c ^ ((row >> 2) & 1)); };
    const __half2 magic = __floats2half2_rn(-1024.0f, -1024.0f);

    int fb, e, row0, nrows;
    if (!work_item<BN, GU_NFB>(b, ts, tiles, off, blockIdx.x, fb, e, row0, nrows)) return;
    const int live_tiles = (nrows + 15) / 16;
    const uint8_t* blob = b.blob[e - b.e0];
    if (tid < BN) srow[tid] = tid < nrows ? src[row0 + tid] : -1;
    __syncthreads();

    const uint8_t* f_ptr = blob + ((tid >> 1) >= FEAT ? geo.up_off : 0) + (size_t) (fb * FEAT + ((tid >> 1) & (FEAT - 1))) * geo.gu_row;
    uint4 f_header = make_uint4(0u, 0u, 0u, 0u), f_codes, f_qh[2] = {};
    uint4 code_cache[4];   // the block's four nibble groups, this thread's chunk of each
    uint32_t f_high = 0;
    int f_sb32 = 0;
    ActStage<BN> as;
    as.init(act, GU_ACT_LD, srow, tid);   // the tokens' activation rows

    const auto fetch_stage = [&](int s) {
        const int kb0 = 2 * s;
        const uint4* blk = reinterpret_cast<const uint4*>(f_ptr) + (kb0 / 8) * BLOCK_CHUNKS;
        if (kb0 % 8 == 0) {   // a new 256-value block every four stages
            f_header = blk[0];
#pragma unroll
            for (int group = 0; group < 4; ++group) code_cache[group] = blk[CODE_CHUNK + group * 2 + f_c];
            if constexpr (Q5K) {
                f_qh[0] = blk[1];
                f_qh[1] = blk[2];
            }
        }
        const int sb32 = (kb0 % 8) + f_c;
        {
            const int group = sb32 / 2;   // uniform across the wave
            const uint32_t low = 0u - (group & 1u), high = 0u - ((group >> 1u) & 1u);
            const auto select = [](uint32_t a, uint32_t bb, uint32_t mask) {
                uint32_t value;
                asm("v_cndmask_b32_e64 %0, %1, %2, %3" : "=v"(value) : "v"(a), "v"(bb), "s"(mask));
                return value;
            };
            const uint4 &c0 = code_cache[0], &c1 = code_cache[1], &c2 = code_cache[2], &c3 = code_cache[3];
            f_codes = make_uint4(select(select(c0.x, c1.x, low), select(c2.x, c3.x, low), high),
                                 select(select(c0.y, c1.y, low), select(c2.y, c3.y, low), high),
                                 select(select(c0.z, c1.z, low), select(c2.z, c3.z, low), high),
                                 select(select(c0.w, c1.w, low), select(c2.w, c3.w, low), high));
        }
        f_sb32 = sb32;
        if constexpr (Q5K) {
            const uint32_t qh[8] = {f_qh[0].x, f_qh[0].y, f_qh[0].z, f_qh[0].w, f_qh[1].x, f_qh[1].y, f_qh[1].z, f_qh[1].w};
            uint32_t hb = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) hb |= gather_bit(qh[i], sb32) << (4 * i);
            f_high = hb;
        }
        as.fetch(s);
    };

    const auto commit_stage = [&]() {
        const int row = tid >> 1;
        s_codes[swizzle(row, f_c)] = f_codes;
        if constexpr (Q5K) s_high[f_c * ROWS + row] = f_high;
        const int sb32 = f_sb32;
        uint32_t sc, mn;
        if (sb32 < 4) {
            sc = header_byte(f_header, 4 + sb32) & 0x3FU;
            mn = header_byte(f_header, 8 + sb32) & 0x3FU;
        } else {
            sc = (header_byte(f_header, 8 + sb32) & 0x0FU) | ((header_byte(f_header, sb32) >> 6U) << 4U);
            mn = (header_byte(f_header, 8 + sb32) >> 4U) | ((header_byte(f_header, 4 + sb32) >> 6U) << 4U);
        }
        const __half2 dm = __builtin_bit_cast(__half2, f_header.x);
        const float scale = __low2float(dm) * (float) sc, offset = __high2float(dm) * (float) mn;
        s_scale[f_c * ROWS + row] = __builtin_bit_cast(uint32_t, __floats2half2_rn(scale, -offset));
        as.commit(s_act);
    };

    v8f acc[BN / 16];
#pragma unroll
    for (int j = 0; j < BN / 16; ++j) acc[j] = v8f{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    const auto compute_stage = [&]() {
        const int row = wave * 16 + sub_lane;   // waves 0-3: gate rows, 4-7: the matching up rows
        const uint4 raw0 = s_codes[swizzle(row, 0)], raw1 = s_codes[swizzle(row, 1)];
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
            const __half2 sb = __builtin_bit_cast(__half2, s_scale[kb * ROWS + row]);
            const __half2 scale2 = __low2half2(sb), bias2 = __high2half2(sb);
            // elements 0-15 of K block 2s + kb are the low (kb = 0) or high (kb = 1) nibbles of chunk 0, 16-31 of chunk 1
            const unsigned shift = 4U * (unsigned) kb;
            const uint32_t words[8] = {raw0.x, raw0.y, raw0.z, raw0.w, raw1.x, raw1.y, raw1.z, raw1.w};
            const uint32_t hb = Q5K ? s_high[kb * ROWS + row] : 0U;
            __half2 h[16];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                uint32_t nib = (words[i] >> shift) & 0x0F0F0F0FU;
                if constexpr (Q5K) nib |= spread_high_bits((hb >> (4 * i)) & 0xFU);
                codes_to_halves(nib, magic, scale2, bias2, h[2 * i], h[2 * i + 1]);
            }
            v16h a_lo, a_hi;
            __builtin_memcpy(&a_lo, &h[0], 32);
            __builtin_memcpy(&a_hi, &h[8], 32);
            mma_tiles<BN>(acc, a_lo, a_hi, s_act, kb, sub_lane, live_tiles);
        }
    };

    fetch_stage(0);
    for (int s = 0; s < GU_NS; ++s) {
        commit_stage();
        __syncthreads();
        if (s + 1 < GU_NS) fetch_stage(s + 1);
        compute_stage();
        __syncthreads();
    }
    gu_epilogue<BN>(acc, lds, tid, live_tiles, nrows, row0, fb, out);
#else
    (void) b; (void) geo; (void) ts; (void) tiles; (void) off; (void) act; (void) src; (void) out;
    __builtin_trap();
#endif
}

// Q5_1 down: 128 of the 2560 output rows against a tile's (or pair's) H rows, K = 640 in 10 stages; the routed rows'
// FP32 outputs in `dm` (native_w11_kernel<T_Q5_1, false>'s).
template <int BN>
__global__ void __launch_bounds__(THREADS) down_kernel(const Batch b, const NativeGeom geo, const int32_t* __restrict__ ts,
                                                       const int2* __restrict__ tiles, const int32_t* __restrict__ off,
                                                       const uint8_t* __restrict__ h, float* __restrict__ dm) {
#if STRATA_F16_DEVICE
    constexpr int CODE_BYTES = ROWS * 2 * 16, HIGH_BYTES = 2 * ROWS * 4, SCALE_BYTES = 2 * ROWS * 4;
    constexpr int STAGE_BYTES = CODE_BYTES + HIGH_BYTES + SCALE_BYTES + ActStage<BN>::BYTES;
    static_assert(D_EPI_BYTES <= STAGE_BYTES, "the epilogue fits the stage");
    __shared__ __attribute__((aligned(16))) uint8_t lds[STAGE_BYTES];
    __shared__ int srow[BN];
    auto* s_codes = reinterpret_cast<uint4*>(lds);
    auto* s_high = reinterpret_cast<uint32_t*>(lds + CODE_BYTES);
    auto* s_scale = reinterpret_cast<uint32_t*>(lds + CODE_BYTES + HIGH_BYTES);
    auto* s_act = reinterpret_cast<uint4*>(lds + CODE_BYTES + HIGH_BYTES + SCALE_BYTES);

    const int tid = threadIdx.x, wave = tid >> 5, sub_lane = tid & 15;
    const int f_c = tid & 1;   // this thread's K block of its weight row
    const auto swizzle = [](int row, int c) { return row * 2 + (c ^ ((row >> 2) & 1)); };
    const __half2 magic = __floats2half2_rn(-1024.0f, -1024.0f);

    int fb, e, row0, nrows;
    if (!work_item<BN, D_NFB>(b, ts, tiles, off, blockIdx.x, fb, e, row0, nrows)) return;
    const int live_tiles = (nrows + 15) / 16;
    if (tid < BN) srow[tid] = tid < nrows ? row0 + tid : -1;   // H is indexed by the routed row
    __syncthreads();

    const uint2* f_ptr = reinterpret_cast<const uint2*>(b.blob[e - b.e0] + geo.down_off + (size_t) (fb * ROWS + (tid >> 1)) * geo.d_row);
    uint4 f_codes;
    uint32_t f_high = 0, f_dm = 0;
    ActStage<BN> as;
    as.init(h, H_LD, srow, tid);
    const auto fetch_stage = [&](int s) {
        const uint2* words = f_ptr + (2 * s + f_c) * 3;   // block_q5_1: {d, m}, qh, qs[16]
        const uint2 w0 = words[0], w1 = words[1], w2 = words[2];
        f_codes = make_uint4(w1.x, w1.y, w2.x, w2.y);
        f_high = w0.y;
        f_dm = w0.x;
        as.fetch(s);
    };
    const auto commit_stage = [&]() {
        const int row = tid >> 1;
        s_codes[swizzle(row, f_c)] = f_codes;
        s_high[f_c * ROWS + row] = f_high;
        s_scale[f_c * ROWS + row] = f_dm;   // the half2 (d, m): w = q d + m
        as.commit(s_act);
    };
    v8f acc[BN / 16];
#pragma unroll
    for (int j = 0; j < BN / 16; ++j) acc[j] = v8f{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const auto compute_stage = [&]() {
        const int row = wave * 16 + sub_lane;
        const uint4 raw[2] = {s_codes[swizzle(row, 0)], s_codes[swizzle(row, 1)]};
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
            const __half2 sb = __builtin_bit_cast(__half2, s_scale[kb * ROWS + row]);
            const __half2 scale2 = __low2half2(sb), bias2 = __high2half2(sb);
            // elements 0-15 are the low nibbles of the 16 code bytes, 16-31 the high, plus bit j of the high-bit word
            const uint32_t words[4] = {raw[kb].x, raw[kb].y, raw[kb].z, raw[kb].w};
            const uint32_t hb = s_high[kb * ROWS + row];
            __half2 hh[16];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t lo = (words[i] & 0x0F0F0F0FU) | spread_high_bits((hb >> (4 * i)) & 0xFU);
                const uint32_t hi = ((words[i] >> 4U) & 0x0F0F0F0FU) | spread_high_bits((hb >> (16 + 4 * i)) & 0xFU);
                codes_to_halves(lo, magic, scale2, bias2, hh[2 * i], hh[2 * i + 1]);
                codes_to_halves(hi, magic, scale2, bias2, hh[8 + 2 * i], hh[8 + 2 * i + 1]);
            }
            v16h a_lo, a_hi;
            __builtin_memcpy(&a_lo, &hh[0], 32);
            __builtin_memcpy(&a_hi, &hh[8], 32);
            mma_tiles<BN>(acc, a_lo, a_hi, s_act, kb, sub_lane, live_tiles);
        }
    };
    fetch_stage(0);
    for (int s = 0; s < D_NS; ++s) {
        commit_stage();
        __syncthreads();
        if (s + 1 < D_NS) fetch_stage(s + 1);
        compute_stage();
        __syncthreads();
    }
    down_epilogue<BN>(acc, lds, tid, live_tiles, nrows, row0, fb, dm);
#else
    (void) b; (void) geo; (void) ts; (void) tiles; (void) off; (void) h; (void) dm;
    __builtin_trap();
#endif
}

}  // namespace

bool covers(int gu_type) { return gu_type == T_Q4_K || gu_type == T_Q5_K; }

bool covers_down(int d_type) { return d_type == GGML_TYPE_Q5_1; }

void gate_up(const Batch& b, const NativeGeom& g, const int32_t* ts, const int2* tiles, const int32_t* off, int64_t n,
             int n_expert, int64_t work_bound, const uint8_t* act, const int32_t* src, uint8_t* out, cudaStream_t s) {
    // one block per work item, the grid the batch's bound (gufo's launch: the hardware balances the blocks; the items
    // of a tile are consecutive, so they meet its activations in L2)
    const unsigned grid = (unsigned) std::max<int64_t>(1, work_bound * GU_NFB);
    const bool q5k = g.gu_type == T_Q5_K;
    if (pick_bn(n, n_expert) == 2 * kTileRows) {
        if (q5k) gate_up_kernel<T_Q5_K, 2 * kTileRows><<<grid, THREADS, 0, s>>>(b, g, ts, tiles, off, act, src, out);
        else gate_up_kernel<T_Q4_K, 2 * kTileRows><<<grid, THREADS, 0, s>>>(b, g, ts, tiles, off, act, src, out);
    } else {
        if (q5k) gate_up_kernel<T_Q5_K, kTileRows><<<grid, THREADS, 0, s>>>(b, g, ts, tiles, off, act, src, out);
        else gate_up_kernel<T_Q4_K, kTileRows><<<grid, THREADS, 0, s>>>(b, g, ts, tiles, off, act, src, out);
    }
}

void down(const Batch& b, const NativeGeom& g, const int32_t* ts, const int2* tiles, const int32_t* off, int64_t n, int n_expert,
          int64_t work_bound, const uint8_t* h, float* dm, cudaStream_t s) {
    const unsigned grid = (unsigned) std::max<int64_t>(1, work_bound * D_NFB);
    if (pick_bn(n, n_expert) == 2 * kTileRows) down_kernel<2 * kTileRows><<<grid, THREADS, 0, s>>>(b, g, ts, tiles, off, h, dm);
    else down_kernel<kTileRows><<<grid, THREADS, 0, s>>>(b, g, ts, tiles, off, h, dm);
}

}  // namespace strata::prefill::fused::f16kq
