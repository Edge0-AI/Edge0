// include/strata/kernels/q8_1_finite.hpp - #606: the q8_1 activation block's scale and sum, kept finite.
//
// A q8_1 block stores its scale d = amax / 127 and the sum of its 32 raw activations as fp16. A sum past 65504 (one
// massive-activation dimension among 32 values is enough) rounds to inf, and d does past amax = 8.3M; the dot
// products then read inf * 0 = NaN, and a NaN in the residual ends as one token repeated (ggml-org/llama.cpp#23606
// is the same defect). Clamped to the largest finite half, such a block gives an approximate product instead of NaN,
// and the values are held to the int8 range they then exceed. Every block that was finite before is stored bit for
// bit as before: values in (65504, 65520) round to 65504 anyway, and a finite block's |x / d| is at most 127. A NaN
// stays NaN (the comparisons are false for it), so STRATA_DBG_NAN still sees one.
#pragma once

#include <cuda_fp16.h>

namespace strata::kernels {

constexpr float kHalfMax = 65504.0f;

__device__ __forceinline__ float q8_1_finite(const float v) {
    return fabsf(v) > kHalfMax ? copysignf(kHalfMax, v) : v;
}

/// #3948: native_mmvq.cu (native_quantize_q8_1, the separate quantiser every consumer of a q8_1 image was written for)
/// is compiled with --use_fast_math, which makes `a / b` the fast full-range division (div.full.f32, up to 2 ulp off). The
/// fused producers (fused_gr.cu's hyper-connection read, verify_kernels.cu's GDN output norm: STRATA_QFUSE) are not, so
/// their IEEE division gave a different code at the rare element whose x / d lies next to a rounding boundary (found by
/// gr_parity: one code in about 10^6, off by one). A translation unit that defines STRATA_Q81_FASTMATH_DIV before including
/// this header divides the way the fast-math one does; every other one, and every HIP build, keeps `a / b` (so nothing
/// that compiled before this changes).
__device__ __forceinline__ float q81_div(const float a, const float b) {
#if defined(STRATA_Q81_FASTMATH_DIV) && defined(__CUDA_ARCH__) && !defined(__HIPCC__)
    float r;
    asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
#else
    return a / b;
#endif
}

/// The block's quantized value of `xi` for the (finite) scale `d`; `amax` = 0 is an all-zero block.
__device__ __forceinline__ int8_t q8_1_quant(const float xi, const float d, const float amax) {
    if (amax == 0.0f) return 0;
    const float q = roundf(q81_div(xi, d));
    return (int8_t) (q > 127.0f ? 127.0f : q < -127.0f ? -127.0f : q);
}

__device__ __forceinline__ half2 q8_1_ds(const float d, const float sum) {
    return make_half2(__float2half(q8_1_finite(d)), __float2half(q8_1_finite(sum)));
}

}  // namespace strata::kernels
