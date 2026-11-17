// include/strata/kernels/fused_gdn.hpp - plan v0.3 P3: the gated delta-net step and its output norm in ONE kernel.
//
// The native step (after llama.cpp's gated_delta_net.cu) gives each warp one (head, column) and strides its 32
// lanes over the 128 ROWS of the state, which is stored `[row][head][col]`: every lane's load is 24.5 KB from its
// neighbour's, so each 32 B sector delivers 4 useful bytes - 47.6 us per layer for 6.3 MB of state traffic.  Here a
// block owns one whole head, its 512 threads are (row group of 32, column), and a row's 128 columns are one
// contiguous 512 B load.  With the head in one block, the per-head RMS norm, the norm weight and the sigmoid(z)
// gate of `gdn_out_norm` run in the same kernel.
//
//   S <- g S + k (beta (v - g S^T k))^T      (g = exp(gate[head]), per value head; q/k shared by h_v / h_k heads)
//   o  = (S^T q) / sqrt(128)
//   y  = rmsnorm(o) * gamma * sigmoid(z)
#pragma once

#include <cstdint>

namespace strata::kernels {

/// The 4-tap causal conv + SiLU over all `channels` (history is `[channel][3]`, updated), then the L2 norm of
/// each of the first `qk_heads` 128-channel heads (q then k), eps as the native `l2_norm` (x / sqrt(sum + eps)).
/// One block per 128-channel head; replaces conv_silu + two l2_norm launches.
void fused_gdn_conv_l2(float* history, const float* qkv, const float* conv_w, float* h, int channels, int qk_heads,
                       float eps, void* stream);

/// alpha and beta, both `(h_v, n_embd)` BF16 against the FP32 activation, with their epilogues:
/// gate = softplus(alpha + dt) * ssm_a,  beta = sigmoid(beta).  One warp per row; replaces two MMVF launches and
/// two elementwise kernels.
void fused_gdn_ab(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, void* stream);

void fused_gdn_step_norm(float* state, const float* q, const float* k, const float* v, const float* gate,
                         const float* beta, const float* z, const float* gamma, float eps, float* y, int h_k, int h_v,
                         void* stream);

/// The same step with the SiLU output gate, y = rmsnorm(o) * gamma * silu(z): Qwen3.6-35B-A3B's (qwen35moe) GDN
/// (CUDA / HIP; the SYCL port has no qwen35moe path).
void fused_gdn_step_norm_silu(float* state, const float* q, const float* k, const float* v, const float* gate,
                              const float* beta, const float* z, const float* gamma, float eps, float* y, int h_k,
                              int h_v, void* stream);

}  // namespace strata::kernels
