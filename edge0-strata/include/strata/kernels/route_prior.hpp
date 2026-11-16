// include/strata/kernels/route_prior.hpp - STRATA_ROUTE_PRIOR=lambda (opt-in, changes the output): cache-conditioned
// routing in verify windows.  See src/kernels/cuda/route_prior.cu.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// Top-10 per token over `logits` (n_tok x 512) where a resident expert (`res_layer[e] >= 0`, the layer's residency
/// table) gets `lambda` added to its logit for the SELECTION; the weights are the router's own probabilities
/// (softmax over the 512 unbiased logits) renormalised over the 10 picked, as the native router does.
void route_prior_top10(const float* logits, const int32_t* res_layer, float lambda, int32_t* ids, float* weights,
                       int n_tok, void* stream);

}  // namespace strata::kernels
