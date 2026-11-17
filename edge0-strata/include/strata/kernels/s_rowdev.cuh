// include/strata/kernels/s_rowdev.cuh - ONE output row of an S-family GEMV, warp-collective.
//
// The V100 fusion work computes a row of `gate` and a row of `up` inside ONE kernel (llama.cpp's
// `mul_mat_gated` shape: both projections from one read of x, SwiGLU in the epilogue).  For the fused and
// unfused pipelines to agree BITWISE each row must be accumulated exactly as the standalone kernel that
// would have produced it - same iteration space, same accumulator layout, same reduction tree, same
// expressions.  These functions are those accumulation loops; `s_gemv_pair.cu` calls each one and
// `fusions_parity` checks the pair against the two standalone calls BITWISE, which is what keeps these
// copies honest (the standalone kernels are untouched and remain the reference).
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace strata::kernels {

/// ggml's `kvalues_iq4nl`, the table `s_gemv.cu` holds in `__constant__`.
__device__ __forceinline__ float iq4nl_value(int code) {
    const signed char tbl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                 1,   13,  25,  38,  53,  69,  89,  113};
    return (float) tbl[code & 0x0F];
}

template <int CODE_BITS>
__device__ __forceinline__ float decode_row(int code, int bias, int codebook) {
    if (codebook == 1) return iq4nl_value(code);
    return (float) (code + bias);
}

/// the Q8_K activation element at `i` - `s_gemv.cu`'s `q8k_at`, verbatim
__device__ __forceinline__ float q8k_row_at(const uint8_t* __restrict__ x, long long i) {
    const uint8_t* blk = x + (i / 256) * 292;
    const float d = __ldg((const float*) blk);
    const int8_t q = ((const int8_t*) (blk + 4))[i % 256];
    return d * (float) q;
}

/// the Q8_0 activation element at `i` - `s_gemv.cu`'s `q8_0_at`, verbatim
__device__ __forceinline__ float q8_0_row_at(const uint8_t* __restrict__ x, long long i) {
    const uint8_t* blk = x + (i / 32) * 34;
    const float d = __half2float(__ushort_as_half(*(const uint16_t*) blk));
    const int8_t q = ((const int8_t*) (blk + 2))[i % 32];
    return d * (float) q;
}

/// `s_gemv_q8_split_kernel`'s row (`s_gemv.cu`), verbatim: 16 accumulators over `i = lane*16` stepping
/// `32*16`, the `(((acc0+acc1)+(acc2+acc3))+...)` tree, then the `__shfl_down_sync` tree.  The reduced
/// value is valid in lane 0.
template <int CODE_BITS, bool Q8K>
__device__ __forceinline__ float row_s_q8(const uint8_t* __restrict__ x, const uint8_t* __restrict__ codes,
                                          const float* __restrict__ scales, const float* __restrict__ offset,
                                          long long n_in, long long o, int lane, int bias, int codebook,
                                          int group_shift, int has_offset) {
    constexpr int PER_BYTE = 8 / CODE_BITS;
    const long long n_groups = n_in >> group_shift;
    const long long codes_per_row = n_in / PER_BYTE;
    const uint8_t* c = codes + o * codes_per_row;
    const float* s = scales + o * n_groups;
    const float* off = has_offset ? offset + o * n_groups : nullptr;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    float acc8 = 0.0f, acc9 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;
    float acc12 = 0.0f, acc13 = 0.0f, acc14 = 0.0f, acc15 = 0.0f;
    constexpr int QE = 16;
    constexpr int QW = 4 * CODE_BITS / 8;
    constexpr unsigned MASK = (1u << CODE_BITS) - 1u;
    long long i = (long long) lane * QE;
    for (; i + QE <= n_in; i += 32 * QE) {
        const long long g = i >> group_shift;
        const float d = s[g];
        const float b = off ? off[g] : 0.0f;
        const uint8_t* cp = c + i / PER_BYTE;
        unsigned v, v2, v3, v4;
        if (QW == 1) { v = cp[0]; v2 = cp[1]; v3 = cp[2]; v4 = cp[3]; }
        else if (QW == 2) { v = *(const uint16_t*) cp; v2 = *(const uint16_t*) (cp + 2);
                            v3 = *(const uint16_t*) (cp + 4); v4 = *(const uint16_t*) (cp + 6); }
        else { v = *(const uint32_t*) cp; v2 = *(const uint32_t*) (cp + 4);
               v3 = *(const uint32_t*) (cp + 8); v4 = *(const uint32_t*) (cp + 12); }
        const int blk_elems = Q8K ? 256 : 32;
        const uint8_t* xb = x + (i / blk_elems) * (Q8K ? 292 : 34);
        const int xi = (int) (i % blk_elems);
        const float xd = Q8K ? __ldg((const float*) xb)
                             : __half2float(__ushort_as_half(*(const uint16_t*) xb));
        const int8_t* xq = (const int8_t*) (xb + (Q8K ? 4 : 2)) + xi;

        const float w0 = decode_row<CODE_BITS>((int) (v & MASK), bias, codebook) * d + b;
        const float w1 = decode_row<CODE_BITS>((int) ((v >> CODE_BITS) & MASK), bias, codebook) * d + b;
        const float w2 = decode_row<CODE_BITS>((int) ((v >> (2 * CODE_BITS)) & MASK), bias, codebook) * d + b;
        const float w3 = decode_row<CODE_BITS>((int) ((v >> (3 * CODE_BITS)) & MASK), bias, codebook) * d + b;
        const float w4 = decode_row<CODE_BITS>((int) (v2 & MASK), bias, codebook) * d + b;
        const float w5 = decode_row<CODE_BITS>((int) ((v2 >> CODE_BITS) & MASK), bias, codebook) * d + b;
        const float w6 = decode_row<CODE_BITS>((int) ((v2 >> (2 * CODE_BITS)) & MASK), bias, codebook) * d + b;
        const float w7 = decode_row<CODE_BITS>((int) ((v2 >> (3 * CODE_BITS)) & MASK), bias, codebook) * d + b;
        acc0 += w0 * (xd * (float) xq[0]);
        acc1 += w1 * (xd * (float) xq[1]);
        acc2 += w2 * (xd * (float) xq[2]);
        acc3 += w3 * (xd * (float) xq[3]);
        acc4 += w4 * (xd * (float) xq[4]);
        acc5 += w5 * (xd * (float) xq[5]);
        acc6 += w6 * (xd * (float) xq[6]);
        acc7 += w7 * (xd * (float) xq[7]);
        const float w8  = decode_row<CODE_BITS>((int) (v3 & MASK), bias, codebook) * d + b;
        const float w9  = decode_row<CODE_BITS>((int) ((v3 >> CODE_BITS) & MASK), bias, codebook) * d + b;
        const float w10 = decode_row<CODE_BITS>((int) ((v3 >> (2 * CODE_BITS)) & MASK), bias, codebook) * d + b;
        const float w11 = decode_row<CODE_BITS>((int) ((v3 >> (3 * CODE_BITS)) & MASK), bias, codebook) * d + b;
        const float w12 = decode_row<CODE_BITS>((int) (v4 & MASK), bias, codebook) * d + b;
        const float w13 = decode_row<CODE_BITS>((int) ((v4 >> CODE_BITS) & MASK), bias, codebook) * d + b;
        const float w14 = decode_row<CODE_BITS>((int) ((v4 >> (2 * CODE_BITS)) & MASK), bias, codebook) * d + b;
        const float w15 = decode_row<CODE_BITS>((int) ((v4 >> (3 * CODE_BITS)) & MASK), bias, codebook) * d + b;
        acc8  += w8  * (xd * (float) xq[8]);
        acc9  += w9  * (xd * (float) xq[9]);
        acc10 += w10 * (xd * (float) xq[10]);
        acc11 += w11 * (xd * (float) xq[11]);
        acc12 += w12 * (xd * (float) xq[12]);
        acc13 += w13 * (xd * (float) xq[13]);
        acc14 += w14 * (xd * (float) xq[14]);
        acc15 += w15 * (xd * (float) xq[15]);
    }
    for (; i < n_in; i += 32 * QE) {
        for (int k = 0; k < QE && i + k < n_in; ++k) {
            const long long e = i + k;
            const long long g = e >> group_shift;
            const int code = (c[e / PER_BYTE] >> ((int) (e % PER_BYTE) * CODE_BITS)) & MASK;
            acc0 += (decode_row<CODE_BITS>(code, bias, codebook) * s[g] + (off ? off[g] : 0.0f)) *
                    (Q8K ? q8k_row_at(x, e) : q8_0_row_at(x, e));
        }
    }
    float acc = (((acc0 + acc1) + (acc2 + acc3)) + ((acc4 + acc5) + (acc6 + acc7))) +
                (((acc8 + acc9) + (acc10 + acc11)) + ((acc12 + acc13) + (acc14 + acc15)));
    for (int step = 16; step > 0; step >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, step);
    return acc;
}

/// `s2_gemv_q8_kernel`'s row (`s2_gemv_q8.cu`), verbatim: quads of four elements strided over
/// `threads_per_row` threads, four accumulators, then the shared-memory tree
/// `partial[tid] += partial[tid + step]`.  `partial` must hold `threads_per_row` floats; the row value
/// lands in `partial[0]`.
__device__ __forceinline__ void row_s2_q8(const uint8_t* __restrict__ act, const uint8_t* __restrict__ codes,
                                          const float* __restrict__ scales, long long n_in, long long o,
                                          int tid, int threads_per_row, float* __restrict__ partial) {
    const long long n_quads = n_in / 4;
    const uint8_t* c = codes + o * n_quads;
    const float* s = scales + o * (n_in / 64);

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (long long q = tid; q < n_quads; q += threads_per_row) {
        const uint8_t byte = c[q];
        const float d = s[q >> 4];
        const long long ablk = (q * 4) / 32;
        const uint8_t* blk = act + ablk * 34;
        const uint16_t dbits = (uint16_t) (blk[0] | (blk[1] << 8));
        const float dx = __half2float(__ushort_as_half(dbits));
        const int8_t* xq = (const int8_t*) (blk + 2);
        const int off = (int) ((q * 4) % 32);

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
    __syncwarp();
    for (int step = threads_per_row / 2; step > 0; step >>= 1) {
        if (tid < step) partial[tid] += partial[tid + step];
        __syncwarp();
    }
}

}  // namespace strata::kernels
