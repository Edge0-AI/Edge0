// include/strata/kernels/router_top10.hpp - the MoE router, host-callable (P2.S2).
//
// logits (n_tokens, n_expert) -> the top-k experts per token with renormalised weights.  Semantics are
// `ref/moe.py::router`'s: softmax over ALL experts, stable descending argsort (ties break by index), gather,
// then renormalise with ggml's 2**-14 lower clamp.
//
// The BF16 GEMV that PRODUCES the logits is a separate kernel and is not here; this handles the routing only,
// which is why `logits` is taken as an f32 buffer the caller has already filled.
#pragma once

#include <cstdint>

namespace strata::kernels {

// `stream` may be null, in which case the call synchronises before returning.  `ids` is (n_tokens, k) int32 and
// `weights` is (n_tokens, k) f32, both device pointers.  k must be <= 64.
void router_top10(const float* logits, int n_tokens, int n_expert, int k, int* ids, float* weights,
                  void* stream);

/// Tests (S6, HIP only): one router kernel by name - 0 the portable kernel, 1 the AMD fast kernel (the default there
/// for 64 < n_expert <= 512, k <= 32; bitwise the same results), 2 the fast kernel forced onto its serial-sum path.
/// False (nothing launched) on CUDA builds or outside that geometry.
bool router_top10_variant(const float* logits, int n_tokens, int n_expert, int k, int* ids, float* weights,
                          void* stream, int variant);

}  // namespace strata::kernels
