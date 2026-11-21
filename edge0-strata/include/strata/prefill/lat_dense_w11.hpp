// include/strata/prefill/lat_dense_w11.hpp - the latent (edge0) D and U dense projections of the prompt path on the
// gfx11 (RDNA3 / RDNA3.5) matrix cores.  HIP only: the CUDA build has latent_dense_kernel (moe_fused.cu).
//
// Both projections are Q8_0 weight rows times q8_1 activation rows.  A 32-value block of each side is one exact
// s8 x s8 integer dot on v_wmma_i32_16x16x16_iu8 (two k-steps of 16), and the two block scales are applied per
// block in FP32: the same terms as iq_mmvq(Q8_0) and latent_u_grouped, in another summation order.
#pragma once

#include <cstdint>

namespace strata::prefill::dense_w11 {

/// This build has the gfx11 kernels, the current device is one of the gfx11 targets, and `fmt` is 8 (Q8_0
/// projections; the Q4_0 file keeps iq_mmvq / latent_u_grouped).
bool available(int fmt);

/// D: the layer input's q8_1 rows `xq` [M, K] against the layer's G * ell Q8_0 weight rows `w`.  Row t's products
/// for group g become the q8_1 row t * G + g of `outq` (ell / 32 blocks of 36 bytes), quantized as q8_1_store
/// does.  `inv` (ell floats, or nullptr): the Q3_K pack's 1 / s column scales, applied before the quantization.
/// `xl_f32` (or nullptr): the same scaled products in FP32, row t * G + g of ell values, for a consumer that reads
/// the FP32 rows (STRATA_LATENT_MMQ).
void project_d(const void* w, const void* xq, int64_t M, int K, int G, int64_t ell, void* outq, float* xl_f32,
               const float* inv, void* stream);

/// U: each entry's q8_1 row of `yq` [M, K] against its label's H Q8_0 weight rows (`w`: n_groups groups of H
/// rows).  `rows` [M] lists the entries bucketed by label and `goff` [n_groups + 1] the run of each label, both on
/// the device (fused::latent_dense_u_tables).  `tiles`: an upper bound of the 64-entry tiles of all runs.  FP32 into
/// `out` [M, H].
void project_u(const void* w, const void* yq, const int32_t* rows, const int32_t* goff, int n_groups, int64_t tiles,
               int64_t M, int K, int H, float* out, void* stream);

}  // namespace strata::prefill::dense_w11
