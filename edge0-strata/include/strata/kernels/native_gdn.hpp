#pragma once
#include "strata/kernels/gdn.hpp"

namespace strata::kernels {

// Configure before creating/capturing a session. Existing captures retain their
// selected kernel, so changing the flag does not change an existing graph.
void native_gdn_set_enabled(bool enabled);
bool native_gdn_enabled();

// Pinned CUDA gated_delta_net recurrence for one token, S=128 and h_v % h_k=0.
// q/k are normalized F32 inputs; q MUST NOT be pre-scaled by 1/sqrt(S).
// Scaling is applied to the completed readout, as in the pinned fused operator.
// State retains Strata's layout: state[(i*h_v+h)*S+j]. The caller owns it and
// output; both must be disjoint from each other and from every read-only input.
// All spans must be aligned to four bytes. A nonnull ordered stream is required.
// No allocation, wait, or change to normalization/gate preprocessing occurs.
void native_gdn_step(float* state, const float* q, const float* k, const float* v,
                     const float* gate, const float* beta, float* output,
                     const GdnShapes& shapes, void* stream);

} // namespace strata::kernels
