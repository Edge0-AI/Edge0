// include/strata/kernels/gdn.hpp - the gated delta-net (GDN) layer's non-projection parts, P2.S2.
//
// Transcribed from `ref/gdn.py`, which is itself transcribed from `build_layer_attn_linear` and
// `build_delta_net_autoregressive`.  The projections (`wqkv`, `wqkv_gate`, `ssm_out`) are the existing
// `s_gemv`/`s2_gemv` kernels; what lives here is everything BETWEEN them.
//
// Geometry of the real artifact (48 layers are GDN, 12 are QSA):
//
//     S = state_size = 128          head_k_dim == head_v_dim
//     h_k = 16  k/q heads           h_v = 48  v heads        (idx[h] = h % h_k)
//     C   = 2*128*16 + 128*48 = 10240 channels               d_conv = 4
//
// STATE LAYOUT.  This is the one place the kernels deliberately differ from the reference, and the reason is
// coalescing.  `ref/gdn.py` carries the state as (S, S, H_v) = (i, j, h); here it is **(S, H_v, S)** =
// (i, h, j), so that for a fixed i a warp walking consecutive j reads CONTIGUOUS memory.  The state is a
// runtime buffer, not a weight, so this costs nothing to choose - but it must be said, because the two
// layouts index the same three numbers and a mismatch is silent.
//
// WHAT MAKES THE RECURRENCE PARALLEL, and it is worth stating because it is not obvious:
//
//     dec      = exp(gate)                          (h_v,)
//     st       = state * dec
//     sk[h,j]  = sum_i st[i,j,h] * k[idx[h]][i]
//     d[h,j]   = (v[h,j] - sk[h,j]) * beta[h]
//     st[i,j,h] += k[idx[h]][i] * d[h,j]
//     o[h,j]   = sum_i st[i,j,h] * q[idx[h]][i]
//
// Every line touches ONLY the (j, h) column.  Nothing couples one column to another, so one thread can own a
// column for the whole step and no barrier is needed anywhere in the recurrence.  (The decay must still be
// applied BEFORE the rank-1 update - `ref/gdn.py` PROPERTY 5 pins that, and the test re-pins it here.)
#pragma once

#include <cstdint>

namespace strata::kernels {

/// Geometry of one GDN layer.  The real artifact: S = 128, h_k = 16, h_v = 48.
struct GdnShapes {
    int64_t S = 0;    ///< head dim, and the state's contracted axis
    int64_t h_k = 0;  ///< key/query heads (q and k are (h_k, S))
    int64_t h_v = 0;  ///< value heads (v, z, o are (h_v, S)); h_v must be a multiple of h_k
};

/// One step of the delta-rule recurrence, IN PLACE on `state`.
///
///   state (S, h_v, S) f32, j fastest, updated in place
///   q, k  (h_k, S)    already L2-normalised, and q ALREADY SCALED by 1/sqrt(S) - the two llama.cpp paths
///                     apply that scale in different places, so the caller owns it
///   v     (h_v, S)
///   gate  (h_v,)      the PRE-exp value; `dec = exp(gate)` is formed here
///   beta  (h_v,)
///   o     (h_v, S)
///
/// The head pairing is `h % h_k` (MODULO), not `h / (h_v/h_k)` (INTERLEAVE).  `ref/gdn.py` PROPERTY 8 and 11
/// pin this and the test re-pins it, because both conventions produce well-formed output.
void gdn_step(float* state, const float* q, const float* k, const float* v, const float* gate,
              const float* beta, float* o, const GdnShapes& s, void* stream);

/// `ggml_ssm_conv` with the carried state prepended.  Updates `conv_state` in place.
///
///   conv_state (d_conv-1, channels)  the previous d_conv-1 input columns, OLDEST FIRST
///   x          (channels,)           the new column
///   kW         (channels, d_conv)    the weight in GGML-NATIVE layout: `kW[c*d_conv + i]` is tap i of
///                                    channel c.  The manifest gives `ssm_conv1d.weight` ne = [4, 10240],
///                                    i.e. ne0 = d_conv is the FAST axis, which is this and NOT the
///                                    reference's row-major (d_conv, channels).
///   out        (channels,)
///
///   out[c] = sum_{i<d_conv} inp[i][c] * kW[c*d_conv + i],  inp = [conv_state | x]
///
/// `kernel[0]` reads the OLDEST state row and the new input lands in the LAST state row.
void gdn_conv_step(float* conv_state, const float* x, const float* kW, float* out, int64_t channels,
                   int64_t d_conv, void* stream);

/// `build_gdn_l2_norm`: `x / sqrt(sum(x^2) + eps)`, over the LAST axis.
///
/// The `+ eps` is an absolute floor on the SQUARED NORM, not on the mean - so the sum is formed first and
/// there is NO division by the width.  The plausible wrong reading (`x / sqrt(mean(x^2) + eps)`) differs by
/// sqrt(S) = 11.3x here, which is why the test asserts the two are distinguishable.
void gdn_l2_norm(float* x, int64_t rows, int64_t cols, float eps, void* stream);

/// `beta = sigmoid(beta)`, in place over `h_v` values.
///
/// **`gdn_step` DOES NOT DO THIS AND THE LAYER MUST.**  `build_layer_attn_linear` L889 is
/// `beta = ggml_sigmoid(beta)` and the step kernel's contract is `d = (v - sk) * beta`, so what it receives has
/// to be a fraction.  Passing the raw `ssm_beta @ cur` leaves the delta-net write strength unbounded and signed.
/// `ref/gdn.py` and `ref/model.py` both sigmoid it; this call is what makes the C++ agree with them.
void gdn_beta_gate(float* beta, int64_t h_v, void* stream);

/// The layer's closing norm: `y = rms_norm(o, eps) * ssm_norm * sigmoid(z)`.
///
/// Note SIGMOID, not SiLU - qwen3.5 used SiLU and this artifact does not, and `ref/gdn.py` says so outright.
/// The norm is over the LAST axis: one RMS per head, so it is per-ROW here and not over the whole (h_v, S).
void gdn_out_norm(const float* o, const float* z, const float* ssm_norm, float* y, int64_t h_v, int64_t S,
                  float eps, void* stream);

}  // namespace strata::kernels
