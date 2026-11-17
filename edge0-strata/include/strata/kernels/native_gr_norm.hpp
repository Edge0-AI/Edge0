#pragma once

namespace strata::kernels {

// Pinned llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d CUDA weighted F32
// RMSNorm for contiguous [n_cols, n_rows] data and equally shaped F32 gamma.
// Every row has independent norm statistics. Output is (scale * input) * gamma.
// Uses 256 threads below n_cols=1024, otherwise 1024, with the pinned XOR tree.
// Compile the implementation with --use_fast_math; no FP64, allocation, or wait.
// All buffers are caller-owned, nonoverlapping, and at least four-byte aligned.
// Require positive dimensions and finite nonnegative epsilon. Inputs/gamma and
// intermediate sums must be finite; epsilon=0 additionally requires nonzero rows.
// Enqueues on the supplied CUDA stream. A null handle explicitly selects CUDA's
// default stream, preserving gr_read's existing API; this function never syncs.
void native_gr_rms_norm_weighted(const float* input, const float* gamma, float* output,
                                 int n_cols, int n_rows, float epsilon, void* stream);

void native_gr_rms_norm_weighted_multi(const float* input, const float* gamma, float* output,
                                       int n_cols, int n_rows, int n_tok, float epsilon, void* stream);

} // namespace strata::kernels
