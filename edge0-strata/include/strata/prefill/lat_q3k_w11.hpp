// include/strata/prefill/lat_q3k_w11.hpp - the q3k pack's latent A stage (the routed experts of a prompt chunk) on
// the gfx11 matrix cores (RDNA3 / RDNA3.5, v_wmma_i32_16x16x16_iu8).  It does the work of moe_fused_q3k.cu's
// latent_kernel_q3k, which needs mma.sync and is CUDA-only: gate/up (Q3_K) with SwiGLU and the q8_1 rounding of H
// in its epilogue, then down (IQ4_NL), each expert read where it already is.  HIP builds only.
#pragma once

#include "strata/prefill/moe_fused.hpp"

#include <cstddef>
#include <cstdint>

namespace strata::prefill::q3k_w11 {

/// The current device runs these kernels (a gfx11 device; read once per device).
bool available();

/// The batch's products, as fused_q3k::experts_latent_q3k: `cnt`, `off`, `ts` and `tiles` are the device tables of
/// fused_q3k::group_latent (rows per expert, first row, first tile, and the {expert, first row} tile list).
/// `xa`: the xl rows' q8_1 blocks.  `src`: entry -> xl row.  `hbuf`: n rows of (iff / 64) * 80 bytes.
/// `dm`: [n][ell] FP32.  `n` bounds the launch grid.
void experts(const fused::WideBatch& b, const fused::LatentGeom& g, const int32_t* cnt, const int32_t* off,
             const int32_t* ts, const void* tiles, int64_t n, const void* xa, const int32_t* src, void* hbuf, float* dm,
             void* stream);

}  // namespace strata::prefill::q3k_w11
