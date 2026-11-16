// include/strata/prefill/moe_fused_q3k.hpp - moe_fused.cu's latent A stage for the q3k pack.
#pragma once

#include "strata/prefill/moe_fused.hpp"

namespace strata::prefill::fused_q3k {

/// moe_fused.cu's experts_latent, verbatim except the weight decode: Q3_K gate/up and IQ4_NL down in place of the
/// NVFP4 blocks, decoded by the quantizers lat_q3k_gu.cu / lat_q3k_down.cu carry.  Same arguments, same tables
/// (group_latent), same staged rows, same mma, same epilogue, same two launches.
void experts_latent_q3k(const fused::WideBatch& b, const fused::LatentGeom& g, int n_expert, int64_t n,
                        const void* scratch, const void* xa, const int32_t* src, void* hbuf, float* dm, void* stream);


// The latent D projection with the pack's shared column scale (inv = 1/s, one float an ell column; nullptr: none).
void latent_dense_d(const void* w, const void* xq, int64_t M, int K, int G, int64_t ell, void* outq,
                    void* stream, int fmt, const float* inv);
}  // namespace strata::prefill::fused_q3k