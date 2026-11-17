#pragma once
#include <cstdint>

namespace strata::kernels {

// Optional pinned CUDA GDN preprocessing. The layer selects these functions with
// native_gdn_enabled(); direct callers explicitly select them by calling them.
// Require a nonnull ordered stream, finite inputs, and four-byte aligned spans.
// No function allocates or synchronizes. All pointers are caller-owned.

// Four-tap convolution, then FP32 fast-math SiLU. History is [channels,3], oldest
// first; weights are [channels,4]. Both outputs are required, mutually disjoint,
// and disjoint from input/weights/history. History must not alias other spans.
void native_gdn_conv_silu(float* history, const float* input, const float* weights,
                          float* raw_output, float* silu_output, int64_t channels,
                          int64_t d_conv, void* stream);

// In-place [rows,128]: scale(rms_norm(x,epsilon/128),1/sqrt(128)). This includes
// the L2 normalization scale, but NOT the recurrence's readout scale.
// Epsilon is finite/nonnegative; epsilon=0 additionally requires nonzero rows.
void native_gdn_l2_norm(float* input, int64_t rows, int64_t cols, float epsilon, void* stream);

// In-place sigmoid over per-head beta; one token.
void native_gdn_beta_gate(float* beta, int64_t heads, void* stream);

// One token: gate[h] = softplus(alpha[h]+dt[h])*ssm_a[h], using the pinned CUDA
// log(1+exp(x)) expression and threshold 20. Output is disjoint from inputs.
void native_gdn_gate(const float* alpha, const float* dt, const float* ssm_a,
                     float* gate, int64_t heads, void* stream);

// [heads,128]: rms_norm(output,epsilon)*gamma*sigmoid(z). Gamma contains only
// 128 floats and broadcasts across heads. Destination is disjoint from inputs.
void native_gdn_out_norm(const float* output, const float* z, const float* gamma,
                         float* destination, int64_t heads, int64_t cols,
                         float epsilon, void* stream);

} // namespace strata::kernels
