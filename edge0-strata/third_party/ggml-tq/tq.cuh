// third_party/ggml-tq/tq.cuh - the trellis expert types (TQ2_T, TQK6, TQK7) for ggml-cuda's MMQ tile loader.
// Derived from agentionai/llama.cpp ggml/src/ggml-cuda/tq.cuh (MIT, the ggml authors; Laurent Zuijdwijk): only what the
// MMQ path needs (no mat-vec kernels), reading Strata's copies of the codebook tables (third_party/ggml/ggml-common-tq.h,
// src/kernels/cuda/tq_lut_i8.cuh) instead of the fork's ggml-common.h.
//
// A 128-weight block is 32 trellis steps of 4 weights.  Step t has a 16-bit state s (the codebook index uses its low
// TQ_STATE_BITS bits) and its weights are d * lut[x >> 21].{lo,hi}, d * lut[(x >> 10) & 2047].{lo,hi},
// x = (s & TQ_STATE_MASK) * 0x9e3779b1.  A block is split into 8 lanes of 4 consecutive steps (16 weights).
#pragma once

#include "common.cuh"
#include "ggml-common-tq.h"
#include "tq_lut_i8.cuh"

#define TQ_LUT_POINTS   2048
#define TQ_LANES        8   // lanes per 128-weight block
#define TQ_STEPS_LANE   4   // trellis steps per lane
#define TQ_PAIRS_LANE   8   // codebook pairs per lane (the TQP layout: 2 weights per step)

#define QR_TQ 4
#define QI_TQ (QK_TQK/QR_TQ)
static_assert(QK_TQ2_T == QK_TQK, "trellis block sizes differ");

template<>
struct ggml_cuda_type_traits<GGML_TYPE_TQ2_T> {
    static constexpr int qk = QK_TQ2_T;
    static constexpr int qr = QR_TQ;
    static constexpr int qi = QI_TQ;
    static constexpr int bs = sizeof(block_tq2_t);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_TQK6> {
    static constexpr int qk = QK_TQK;
    static constexpr int qr = QR_TQ;
    static constexpr int qi = QI_TQ;
    static constexpr int bs = sizeof(block_tqk6);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_TQK7> {
    static constexpr int qk = QK_TQK;
    static constexpr int qr = QR_TQ;
    static constexpr int qi = QI_TQ;
    static constexpr int bs = sizeof(block_tqk7);
};

// Gyro-L pair trellis (types 147-149): the same lane packing (16 values = 4 ints of 4 int8), 64 steps of
// 2 weights per 128-value block.
template<> struct ggml_cuda_type_traits<GGML_TYPE_TQP5> {
    static constexpr int qk = QK_TQP; static constexpr int qr = QR_TQ; static constexpr int qi = QI_TQ;
    static constexpr int bs = sizeof(block_tqp5); };
template<> struct ggml_cuda_type_traits<GGML_TYPE_TQP6> {
    static constexpr int qk = QK_TQP; static constexpr int qr = QR_TQ; static constexpr int qi = QI_TQ;
    static constexpr int bs = sizeof(block_tqp6); };
template<> struct ggml_cuda_type_traits<GGML_TYPE_TQP7> {
    static constexpr int qk = QK_TQP; static constexpr int qr = QR_TQ; static constexpr int qi = QI_TQ;
    static constexpr int bs = sizeof(block_tqp7); };

static constexpr __host__ __device__ bool ggml_cuda_type_is_tqp(ggml_type type) {
    return type == GGML_TYPE_TQP5 || type == GGML_TYPE_TQP6 || type == GGML_TYPE_TQP7;
}

static constexpr __host__ __device__ bool ggml_cuda_type_is_tq(ggml_type type) {
    return type == GGML_TYPE_TQ2_T || type == GGML_TYPE_TQK6 || type == GGML_TYPE_TQK7 || ggml_cuda_type_is_tqp(type);
}

template <ggml_type type> struct tq_block_info;
template <> struct tq_block_info<GGML_TYPE_TQ2_T> { typedef block_tq2_t block_t; static constexpr int K = 8; };
template <> struct tq_block_info<GGML_TYPE_TQK6>  { typedef block_tqk6  block_t; static constexpr int K = 6; };
template <> struct tq_block_info<GGML_TYPE_TQK7>  { typedef block_tqk7  block_t; static constexpr int K = 7; };
template <> struct tq_block_info<GGML_TYPE_TQP5>  { typedef block_tqp5  block_t; static constexpr int K = 5; };
template <> struct tq_block_info<GGML_TYPE_TQP6>  { typedef block_tqp6  block_t; static constexpr int K = 6; };
template <> struct tq_block_info<GGML_TYPE_TQP7>  { typedef block_tqp7  block_t; static constexpr int K = 7; };
static_assert(QK_TQP == QK_TQK, "trellis block sizes differ");

// States of steps 4l..4l+3 of one block. qs is the block's qs array, 2-byte aligned
// (every trellis block has an even size and qs sits right after the fp16 scale).
template <ggml_type type>
static __device__ __forceinline__ void tq_states4(const uint8_t * qs, const int l, uint32_t s[TQ_STEPS_LANE]) {
    const uint16_t * qs16 = (const uint16_t *) qs;
    if constexpr (type == GGML_TYPE_TQ2_T) {
        // s_t = qs[(t + 31) % 32] << 8 | qs[t]
        const uint32_t w01  = qs16[2*l + 0];
        const uint32_t w23  = qs16[2*l + 1];
        const uint32_t prev = qs[(4*l + 31) & 31];
        const uint32_t b0 = w01 & 0xFF, b1 = w01 >> 8, b2 = w23 & 0xFF, b3 = w23 >> 8;
        s[0] = (prev << 8) | b0;
        s[1] = (b0   << 8) | b1;
        s[2] = (b1   << 8) | b2;
        s[3] = (b2   << 8) | b3;
    } else {
        // Step t's state is the 16 stream bits from bit (31 - t)*K (circular over 32*K bits, i.e. 2*K u16 words).
        // Steps 4l..4l+3 all lie in the four words from wb = off3 >> 4, off3 = (28 - 4l)*K.
        constexpr int K      = tq_block_info<type>::K;
        constexpr int nwords = 2*K;
        const int off3 = (28 - 4*l)*K;
        const int wb   = off3 >> 4;
        const int r    = off3 & 15;
        const int wi1  = wb + 1 >= nwords ? wb + 1 - nwords : wb + 1;
        const int wi2  = wb + 2 >= nwords ? wb + 2 - nwords : wb + 2;
        const int wi3  = wb + 3 >= nwords ? wb + 3 - nwords : wb + 3;
        const uint64_t v = (uint64_t) qs16[wb] | ((uint64_t) qs16[wi1] << 16) |
                           ((uint64_t) qs16[wi2] << 32) | ((uint64_t) qs16[wi3] << 48);
#pragma unroll
        for (int i = 0; i < TQ_STEPS_LANE; ++i) {
            s[i] = (uint32_t) (v >> (r + (3 - i)*K)) & 0xFFFFu;
        }
    }
}

// TQP: the 15-bit states of steps 8l..8l+7 (s[j] = step 8l + j).  Step t's window starts at stream bit
// (63 - t)*K, so the lane's steps lie in the bits from S = (56 - 8l)*K: step 8l + j at S + (7 - j)*K.
// S = 8*(7 - l)*K, so r = S & 15 is 0 or 8 and r + 7*K + 15 <= 72: five u16 words (wrapped modulo the
// 4*K words of the stream) cover it, the fifth only when r + 7*K + 15 > 64 (never for K = 5).
template <ggml_type type>
static __device__ __forceinline__ void tqp_states8(const uint8_t * qs, const int l, uint32_t s[TQ_PAIRS_LANE]) {
    constexpr int K      = tq_block_info<type>::K;
    constexpr int nwords = 4*K;
    const uint16_t * qs16 = (const uint16_t *) qs;
    const int S   = (56 - 8*l)*K;
    const int wb  = S >> 4;
    const int r   = S & 15;
    const int wi1 = wb + 1 >= nwords ? wb + 1 - nwords : wb + 1;
    const int wi2 = wb + 2 >= nwords ? wb + 2 - nwords : wb + 2;
    const int wi3 = wb + 3 >= nwords ? wb + 3 - nwords : wb + 3;
    uint64_t v = (uint64_t) qs16[wb] | ((uint64_t) qs16[wi1] << 16) |
                 ((uint64_t) qs16[wi2] << 32) | ((uint64_t) qs16[wi3] << 48);
    v >>= r;
    if constexpr (8 + 7*K + TQ_STATE_BITS > 64) {
        const int wi4 = wb + 4 >= nwords ? wb + 4 - nwords : wb + 4;
        v |= r ? (uint64_t) qs16[wi4] << (64 - r) : 0;
    }
#pragma unroll
    for (int j = 0; j < TQ_PAIRS_LANE; ++j) {
        s[j] = (uint32_t) (v >> ((7 - j)*K)) & TQ_STATE_MASK;
    }
}

// MMQ (batched) path: lane l of block x as 16 int8 values (4 packed ints, weight order) with one float scale per
// lane, i.e. weights ~= scale * q.  The int8 codebook (one global scale) is already int8: no decode to float.
template <ggml_type type>
static __device__ __forceinline__ float tq_lane_to_q8(const void * x, const int l, int q[TQ_STEPS_LANE]) {
    typedef typename tq_block_info<type>::block_t block_t;
    const block_t * b = (const block_t *) x;

    if constexpr (ggml_cuda_type_is_tqp(type)) {
        // 8 pairs of 2 weights, in weight order: two pairs per int, as the TQ path packs two steps
        uint32_t s[TQ_PAIRS_LANE];
        tqp_states8<type>(b->qs, l, s);
#pragma unroll
        for (int i = 0; i < TQ_STEPS_LANE; ++i) {
            const uint32_t p0 = tq_lut_i8[((s[2 * i] & TQ_STATE_MASK) * 0x9e3779b1u) >> 21];
            const uint32_t p1 = tq_lut_i8[((s[2 * i + 1] & TQ_STATE_MASK) * 0x9e3779b1u) >> 21];
            q[i] = (int) (p0 | (p1 << 16));
        }
    } else {
        uint32_t s[TQ_STEPS_LANE];
        tq_states4<type>(b->qs, l, s);
#pragma unroll
        for (int i = 0; i < TQ_STEPS_LANE; ++i) {
            const uint32_t xh = (s[i] & TQ_STATE_MASK) * 0x9e3779b1u;
            q[i] = (int) ((uint32_t) tq_lut_i8[xh >> 21] | ((uint32_t) tq_lut_i8[(xh >> 10) & 2047u] << 16));
        }
    }
    // Strata's block structs hold the fp16 scale as uint16_t (the fork's as ggml_half): reinterpret, do not convert
    return __half2float(__ushort_as_half(b->d)) * TQ_LUT_I8_SCALE;
}
