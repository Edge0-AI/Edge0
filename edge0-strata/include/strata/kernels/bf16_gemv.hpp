// include/strata/kernels/bf16_gemv.hpp - the GEMV for a BF16 weight, P2.S5.
//
// WHY THIS EXISTS WHEN `s_gemv` ALREADY DOES.  `s_gemv` decodes the S-family canonical forms (S2/S4/S8) from
// `codes` + `scales`, and a BF16 weight is not one of them - it has no codes and no scales, it is a plain
// 16-bit matrix.  Six tensors per layer are in that state:
//
//     ssm_alpha.weight, ssm_beta.weight                 GDN,  [n_embd, 48]
//     indexer.q_proj.weight, indexer.k_proj.weight      QSA,  [n_embd, 512] and [n_embd, 128]
//     ple_value.weight                                  layer 1, [n_embd, n_embd]
//     ffn_gate_inp.weight                               handled inside `router_top10`
//
// The original entry points take BF16 activation bits for the historical CPU-derived contract. The explicit
// FP32 entry point below follows the pinned CUDA single-token MMVF contract. See docs/activation-contract.md
// for why activation precision depends on backend and batch geometry rather than weight type alone.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// `y[o] = sum_i f32(x[i]) * f32(w[o*n_in + i])`, with both sides bf16-valued so every product is exact in
/// f32 and only the summation order differs.
///
/// THE WEIGHT LAYOUT IS THE MANIFEST'S: shape [n_in, n_out] with ne0 = n_in CONTIGUOUS, so row `o` is a
/// contiguous run of `n_in` bf16 values.  That is the same convention `s_gemv` uses, and it is what
/// `strata::core::WeightRef::ne0/ne1` already report.
void bf16_gemv(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, void* stream);

/// TWO weight matrices from ONE kernel (llama.cpp's `*_mul_mat_multi` shape): `y1 = w1 @ x`, `y2 = w2 @ x`.
/// One warp computes row o of both matrices with two accumulators over one lane-strided pass, so each output
/// is bitwise what `bf16_gemv` produced for that matrix wherever `bf16_gemv` used its warp kernel
/// (`n_out >= 64`); a matrix below that threshold falls back to the plain call to keep the naive kernel's
/// summation order.  `n_out1` or `n_out2` may be 0 for a one-sided call.
void bf16_gemv_pair(const uint16_t* x, const uint16_t* w1, float* y1, int64_t n_out1, const uint16_t* w2,
                    float* y2, int64_t n_out2, int64_t n_in, void* stream);

/// The row-split variant, for the same reason `s_gemv_split` exists: with one thread per output row the
/// parallelism IS the output width, and `ssm_alpha` has 48 of them over 48 SMs.  One warp per row, lanes
/// striding the reduction axis - which is also what makes the access pattern coalesced in this layout.
///
/// NOT bit-identical to `bf16_gemv`: the partial sums are added in a different order.  Both are kept and both
/// are tested, because the naive one is the reference the split one is checked against.
void bf16_gemv_split(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out,
                     int threads_per_row, void* stream);

/// Single-token, contiguous 2D BF16-weight/F32-activation MMVF from the pinned llama.cpp CUDA path.
/// Preserves pair accumulation, adaptive block size and warp-XOR reduction order. This is a separate
/// opt-in entry point; the legacy BF16-activation kernels above retain their behavior.
/// `w[o*n_in+i]` is BF16 bits; x/y are F32. n_in must be positive and even, n_out positive, both <= INT_MAX.
/// x must be 8-byte aligned, w 4-byte aligned, and y 4-byte aligned. Throws on invalid geometry/pointers.
/// The device buffers must not overlap; allocation bounds remain the caller's responsibility.
/// No allocations or synchronization, including when stream is null (the CUDA default stream).
void bf16_gemv_fp32_mmvf(const float* x, const uint16_t* w, float* y,
                         int64_t n_in, int64_t n_out, void* stream);
/// `ncols` columns at once: x is [ncols][n_in], y is [ncols][n_out]; each column bitwise equal to a
/// `bf16_gemv_fp32_mmvf` call on it.
void bf16_gemv_fp32_mmvf_cols(const float* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, int ncols,
                              void* stream);
/// bf16_gemv_fp32_mmvf for n_tok (1..8) activation rows x[t * ldx], outputs y[t * ldy + j]; one launch,
/// the weight read once, every output bit-identical to its own single-row call.
void bf16_gemv_fp32_mmvf_multi(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                               int64_t n_in, int64_t n_out, int n_tok, void* stream);
/// S26 STRATA_LFUSE: bf16_gemv_fp32_mmvf_multi(x, ldx, w, y, ldy, n_in, n_out, n_tok) plus the 1-row
/// bf16_gemv_fp32_mmvf_multi(x, ldx, w_aux, y_aux, ldy_aux, n_in, 1, n_tok) in ONE launch, every output bitwise the
/// two calls'. Returns false (nothing launched) unless the rows path applies (STRATA_MMVF_ROWS=1, n_out >= 64, 2-8 rows).
bool bf16_gemv_fp32_mmvf_multi_aux(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                                   int64_t n_in, int64_t n_out, int n_tok, const uint16_t* w_aux, float* y_aux,
                                   int64_t ldy_aux, void* stream);

}  // namespace strata::kernels
