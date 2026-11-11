// include/strata/kernels/swiglu.cuh - the SwiGLU expressions, ONE copy each.
//
// The fusion work (V100) moves `silu(gate) * up` into the epilogue of the projection that computes `up`.
// For that to be bit-exact the fused epilogue must compute EXACTLY the expression the standalone kernel
// computed - same operations, same precision, same order.  A second copy of the expression is a second
// thing to drift, so every caller (the standalone kernels and the fused epilogues) includes this file.
//
// THREE VARIANTS, because the engine really has three:
//
//   `swilu_legacy`  the `ref/moe.py` transcription: silu in DOUBLE then cast, multiply in float
//                   (`shared_expert.cu`'s `swiglu_kernel`, and the S2 expert path used to use this shape).
//   `swilu_fast`    the S2 expert path's float expression with `__expf`
//                   (`s2_expert_grouped.cu`'s `swiglu_kernel`).
//   `swilu_native`  the pinned CUDA contract: `__fdividef` and `__expf`
//                   (`shared_expert.cu`'s `native_swiglu_kernel`).
//
// Bit-exactness across translation units: both call sites compile this with the same CUDA flags (no
// `--use_fast_math` on the files that use the legacy and fast variants), the expressions are straight-line
// IEEE ops around one libdevice call, and the parity tests compare the fused and unfused pipelines bitwise.
#pragma once

#include <cuda_runtime.h>

namespace strata::kernels {

/// `out = (float)((double)g / (1.0 + exp(-(double)g))) * u` - `shared_expert.cu`'s `swiglu_kernel`.
__device__ __forceinline__ float swilu_legacy(float g, float u) {
    const double x = (double) g;
    return (float) (x / (1.0 + exp(-x))) * u;
}

/// `out = (g / (1.0f + __expf(-g))) * u` - `s2_expert_grouped.cu`'s `swiglu_kernel`.
__device__ __forceinline__ float swilu_fast(float g, float u) {
    return (g / (1.0f + __expf(-g))) * u;
}

/// `out = __fdividef(g, 1.0f + __expf(-g)) * u` - `shared_expert.cu`'s `native_swiglu_kernel`.
__device__ __forceinline__ float swilu_native(float g, float u) {
    return __fdividef(g, 1.0f + __expf(-g)) * u;
}

}  // namespace strata::kernels
