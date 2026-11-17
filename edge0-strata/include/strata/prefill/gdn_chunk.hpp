// include/strata/prefill/gdn_chunk.hpp - the chunked GDN prompt recurrence for Volta (sm_70), v100/gdn-chunk.
//
// The prompt recurrence (`gdn_recurrence`, src/prefill/kernels.cu) walks the chunk's tokens one by one; each
// token depends on the state left by the previous one, so the whole launch is a serial chain of T steps per
// (head, column-block).  On a V100 that chain, not the flops, is what the GDN layers cost: the arithmetic is
// O(S^2) per token and S = 128.
//
// The chunked form (the FlashQLA "gdn-chunk" structure of 1Cat-vLLM, as ported for sm_70 in llama.cpp-v100's
// ggml/src/ggml-cuda/gdn-chunk-sm70*: cumsum -> kkt -> fwd, 64-token chunks) trades the token chain for a
// CHUNK chain: inside one 64-token chunk everything runs as dense matmuls and one triangular solve, and only
// the chunks stay in order (each needs the previous chunk's state).  Same recurrence, different summation
// order - so FP32-level, NOT bit-exact against `gdn_recurrence` (the precedent: qsa_prompt_attn.hpp).
//
// THE MATH, derived from the recurrence in `strata/kernels/gdn.hpp` (per head; the state columns j are
// independent, so everything below is per column and can be written in matrix form over the i axis):
//
//     a_t = exp(gate_t)                              (decay)
//     W_t = a_t W_{t-1} + k_t delta_t^T              (state, i x j)
//     delta_t = beta_t (v_t - a_t k_t^T W_{t-1})
//     o_t = q_t^T W_t                                (then * rsqrtf(S) by the caller-side convention)
//
// Unrolled over a chunk (c_t = inclusive cumsum of gate inside the chunk, sigma = c_63, W_in = state at the
// chunk's first token - one step per token, no approximations):
//
//     W_t   = exp(c_t) W_in + sum_{s<=t} exp(c_t-c_s) k_s delta_s^T
//     x_t   := a_t k_t^T W_{t-1}
//           = exp(c_t) k_t^T W_in + sum_{s<t} exp(c_t-c_s) (k_t.k_s) delta_s
//     delta_t = beta_t (v_t - x_t)
//
// which is one lower-triangular solve per chunk and column:
//
//     (I + strict_lower(A)) delta = b        A[t,s] = beta_t exp(c_t-c_s) (k_t.k_s)   (s < t)
//     b_t = beta_t ( v_t - exp(c_t) k_t^T W_in )
//     o_t = exp(c_t) q_t^T W_in + sum_{s<=t} P[t,s] delta_s     P[t,s] = exp(c_t-c_s) (q_t.k_s)
//     W_out = exp(sigma) W_in + sum_s k_s exp(sigma-c_s) delta_s^T
//
// Every decay factor goes from a LATER token back to an EARLIER one - <= 1 for a real GDN gate (log-decay,
// negative) - so no exp of a positive argument is ever formed, delta stays O(beta v), and the solve is
// well-scaled in f32.  (The reference's equivalent move is its cumsum/g_rev_exp pair; its kkt kernel keeps
// the matrix decay-free and lets fwd multiply the mask in - this implementation bakes the decay into A and P
// in the kkt stage, which is the same numbers with one less pass.)
//
// The three kernels here are the same three stages as the reference:
//
//   gdn_chunk_cumsum_kernel  g_cumsum[t][h] = cumsum of gate over the chunk's 64 tokens
//   gdn_chunk_kkt_kernel     A and P = (q_t.k_s) per (chunk, head), one block each
//   gdn_chunk_fwd_kernel     one block per (head, 32 value-columns) over ALL chunks: the state slice lives in
//                            registers across chunks (chunk-serial scan), each chunk is a batch of matmuls
//
// Kernels are hand-written CUDA, FP32 throughout (the q/k staging through FP16 and the m8n8k4/wmma of the
// reference's TileLang output are a later step; nothing here depends on TileLang).  Dispatch is a runtime
// compute-capability check in `gdn_recurrence` - sm_80 and up keep the old recurrence unless STRATA_GDN_CHUNK
// overrides it.
#pragma once

#include <cstdint>

namespace strata::prefill {

/// Is the chunked recurrence worth dispatching to on this device?  Volta (cc 7.0) only - that is where the
/// token chain hurts and where this was measured; every other architecture keeps the existing recurrence.
bool gdn_chunk_available();

/// The chunked recurrence over `T` tokens (T a multiple of 64 - the caller keeps `T % 64` for the recurrent
/// kernel; `gdn_recurrence` does this itself).  Same buffers and conventions as `gdn_recurrence`, minus the
/// output norm, which the caller runs over the whole chunk as before:
///
///   state (S, h_v, S) f32, j fastest, updated in place
///   h     [T, C]      post-conv, q/k L2-normalised rows (q NOT scaled by 1/sqrt(S); this writes o/sqrt(S))
///   gate  [T, h_v]    the PRE-exp gate; beta [T, h_v] already sigmoided
///   oc    [T, h_v, S] the un-normed o * rsqrtf(S) - what `gdn_rec_cols_*` writes as `y` before the norm
///
/// FP32-level, not bit-exact against the recurrence: the same recurrence in a different summation order.
/// Scratch (A, P and the cumsum) is a grow-only static workspace, segmented so a 32k prompt does not size it.
void gdn_chunk_recurrence(float* state, const float* h, const float* gate, const float* beta, float* oc,
                          int64_t T, void* stream);

}  // namespace strata::prefill
