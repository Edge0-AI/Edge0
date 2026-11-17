// include/strata/kernels/quantize_act_dev.cuh - the per-block activation quantizers, ONE copy each.
//
// The fusion work (V100) folds `silu(gate) * up` and the activation quantizer that follows it into one
// kernel.  For that to be bit-exact the fused kernel must quantize EXACTLY what `quantize_act.cu`'s
// kernels quantize - same amax order, same double `rint`, same clamps.  A second copy of the math is a
// second thing to drift, so `quantize_act.cu` and the fused kernels both include this file and call the
// same functions.  The standalone kernels are thin wrappers around these; there is no other difference.
#pragma once

#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>

namespace strata::kernels {

constexpr int Q8_0_BLOCK = 32;
constexpr int Q8_0_BLOCK_BYTES = 34;
constexpr int Q8_K_BLOCK = 256;
constexpr int Q8_K_BLOCK_BYTES = 292;

/// ggml's `nearest_int` (ggml-quants.c L621).  The magic constant is 1.5 * 2^23; it is transcribed rather
/// than rewritten as `rintf` because the two differ on exact ties - which is the whole point of it.
__device__ __forceinline__ int nearest_int_dev(float fval) {
    const float val = fval + 12582912.0f;
    int i;
    memcpy(&i, &val, 4);
    return (i & 0x007fffff) - 0x00400000;
}

/// `block_q8_0` for one 32-element block, ggml's bytes: `d32 = amax/127` in fp32, `d` stored fp16, the
/// QUANTIZED INTEGERS divided by `d32` (not by the stored fp16), and `rint` in DOUBLE.  See the three
/// subtleties in `quantize_act.cu`'s header comment - every one of them is load-bearing.
__device__ __forceinline__ void quantize_q8_0_block(const float* __restrict__ xb, uint8_t* __restrict__ out) {
    float amax = 0.0f;
    for (int i = 0; i < Q8_0_BLOCK; ++i) amax = fmaxf(amax, fabsf(xb[i]));
    if (amax == 0.0f) {
        const uint16_t zb = f16_from_f32(0.0f);
        out[0] = (uint8_t) (zb & 0xFF);
        out[1] = (uint8_t) (zb >> 8);
        for (int i = 0; i < Q8_0_BLOCK; ++i) out[2 + i] = 0;
        return;
    }
    const float d32 = amax / 127.0f;
    const uint16_t d16bits = f16_from_f32(d32);
    out[0] = (uint8_t) (d16bits & 0xFF);
    out[1] = (uint8_t) (d16bits >> 8);
    for (int i = 0; i < Q8_0_BLOCK; ++i) {
        double q = rint((double) xb[i] / (double) d32);
        if (q > 127.0) q = 127.0;
        if (q < -128.0) q = -128.0;
        out[2 + i] = (uint8_t) (int8_t) q;
    }
}

/// `block_q8_0` bytes PLUS the fp32 scale and the CPU's rounding rule (round half away from zero) - the
/// `quantize_q8_0_scaled_kernel` body, verbatim, so a hit and a CPU miss agree.
__device__ __forceinline__ void quantize_q8_0_scaled_block(const float* __restrict__ xb,
                                                           uint8_t* __restrict__ out, float* __restrict__ scale_out) {
    float amax = 0.0f;
    for (int i = 0; i < Q8_0_BLOCK; ++i) amax = fmaxf(amax, fabsf(xb[i]));
    const float s = amax > 0.f ? amax / 127.f : 0.f;
    const float inv = s > 0.f ? 1.f / s : 0.f;
    *scale_out = s;
    const uint16_t d16bits = f16_from_f32(s);
    out[0] = (uint8_t) (d16bits & 0xFF);
    out[1] = (uint8_t) (d16bits >> 8);
    for (int i = 0; i < Q8_0_BLOCK; ++i) {
        const float t = xb[i] * inv;
        const float r = t + (t >= 0.f ? 0.5f : -0.5f);
        int v = (int) r;
        v = v < -127 ? -127 : (v > 127 ? 127 : v);
        out[2 + i] = (uint8_t) (int8_t) v;
    }
}

/// `block_q8_K` for one 256-element block - the `quantize_q8_K_kernel` body, verbatim (including the
/// zero-block divergence note: bsums are zeroed where ggml leaves them unwritten).
__device__ __forceinline__ void quantize_q8_K_block(const float* __restrict__ xb, uint8_t* __restrict__ out) {
    float* d = (float*) out;
    int8_t* qs = (int8_t*) (out + 4);
    int16_t* bsums = (int16_t*) (out + 4 + Q8_K_BLOCK);
    float max = 0.0f, amax = 0.0f;
    for (int j = 0; j < Q8_K_BLOCK; ++j) {
        const float ax = fabsf(xb[j]);
        if (ax > amax) {
            amax = ax;
            max = xb[j];
        }
    }
    if (amax == 0.0f) {
        *d = 0.0f;
        for (int j = 0; j < Q8_K_BLOCK; ++j) qs[j] = 0;
        for (int j = 0; j < Q8_K_BLOCK / 16; ++j) bsums[j] = 0;
        return;
    }
    const float iscale = -127.0f / max;
    for (int j = 0; j < Q8_K_BLOCK; ++j) {
        const int v = nearest_int_dev(__fmul_rn(iscale, xb[j]));
        qs[j] = (int8_t) min(127, v);
    }
    for (int j = 0; j < Q8_K_BLOCK / 16; ++j) {
        int sum = 0;
        for (int ii = 0; ii < 16; ++ii) sum += qs[j * 16 + ii];
        bsums[j] = (int16_t) sum;
    }
    *d = 1.0f / iscale;
}

}  // namespace strata::kernels
