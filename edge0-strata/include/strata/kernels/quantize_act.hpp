// include/strata/kernels/quantize_act.hpp - activation quantization, host-callable (P2.S2).
//
// `ggml_mul_mat` converts src1 (the ACTIVATION) to the weight's `vec_dot_type` before the dot product.  The
// CPU expert path already does this (`bench/micro/cpu_s2.cpp`, int8 + vpdpbusd, parity 1.461e-06); the GPU
// kernels take FP16 activations.  Leaving the two inconsistent means the same token computes different
// numbers for different experts, so this is the GPU half of the same rule.
//
// Q2_0 -> Q8_0 is the case that matters most: the routed experts are all Q2_0, which is 31.64 GiB of the 38
// GiB pack.  Q8_K is the OTHER HALF of `VEC_DOT_TYPE` (Q3_K/Q4_K/Q5_K/Q6_K/IQ4_XS) and covers 2.89 GiB of
// dense weights - the attention projections and `ssm_out`, i.e. the numerically sensitive ones.
//
// `docs/activation-contract.md` records the decision that BOTH must exist and be used: an FP16 activation
// against a quantized weight disagrees with the oracle by 0.6-0.9%, which is 6-9x P2.S2's 1e-3 tolerance.
#pragma once

#include <cstdint>

namespace strata::kernels {

// x (n floats) -> ggml's block_q8_0 layout: 34 bytes per 32 elements, { fp16 d ; int8 qs[32] }.
// n must be a multiple of 32.  `stream` may be null, in which case the call synchronises.
void quantize_q8_0(const float* x, uint8_t* blocks, int64_t n, void* stream);

// **R4.2h: THE SAME BLOCKS, PLUS THE CPU's fp32 SCALE - and the CPU's rounding rule with it.**  The engine
// computes each routed expert twice (misses on the CPU through `act_quant_q8_1`, hits on the GPU through
// `quantize_q8_0`) and the two disagreed: `ActQ::scale` is fp32 while a `block_q8_0` stores fp16 `d`, which is
// **4.761e-04 relative on 80 of 80 chunks** (`bench/micro/act_quant_parity.cu`).  This variant exists so the
// hit path can use the CPU's multiplier.  Writes `blocks[n/32 * 34]` and `scales[n/32]`; `scales` must not be
// null.  `quantize_q8_0` is unchanged and still matches ggml's bytes, which is what `moe_hit_parity` checks.
void quantize_q8_0_scaled(const float* x, uint8_t* blocks, float* scales, int64_t n, void* stream);

// ===================== fused silu(gate) * up + quantize =====================
//
// One kernel where two ran: the standalone SwiGLU pass wrote the products and the quantizer read them back.
// The products are computed in registers, stored to `gate_out` exactly as the standalone kernels stored them,
// and quantized from those same values - the bytes out are what the two-kernel pipeline produced, with one
// launch and one round trip over the intermediate gone.  `gate_out` may alias `gate`.
//
// `swilu_kind` selects the SwiGLU expression this replaces: 0 = the double-precision `ref/moe.py` form
// (`shared_expert`'s legacy kernel), 1 = the S2 expert path's float `__expf` form, 2 = the pinned CUDA
// native form.  The expressions live in `strata/kernels/swiglu.cuh`, one copy each.

// n pairs -> ggml block_q8_0 for the products.
void swilu_quantize_q8_0(const float* gate, const float* up, float* gate_out, int64_t n, int swilu_kind,
                         uint8_t* blocks, void* stream);

// The same blocks plus the CPU contract's fp32 scales (`quantize_q8_0_scaled`).
void swilu_quantize_q8_0_scaled(const float* gate, const float* up, float* gate_out, int64_t n, int swilu_kind,
                                uint8_t* blocks, float* scales, void* stream);

// The same for the Q8_K contract (n a multiple of 256).
void swilu_quantize_q8_K(const float* gate, const float* up, float* gate_out, int64_t n, int swilu_kind,
                         uint8_t* blocks, void* stream);

// ===================== every activation image of x in ONE pass =====================
//
// The layer start wants 2-3 images of the SAME x (ggml block_q8_0, block_q8_K, bf16) and the engine ran one
// kernel per image - three reads and three launches.  This produces whichever images the caller asks for in
// one pass; a null pointer skips that image.  The bytes of each image are what its standalone kernel wrote
// (the per-block math is shared with those kernels).  n must be a multiple of 256 (the Q8_K block).
void quantize_act_images(const float* x, int64_t n, uint8_t* q8_0_blocks, uint8_t* q8_K_blocks,
                         uint16_t* bf16, void* stream);

// The inverse, for round-trip checks: each element becomes `q * d16`.
void dequant_q8_0(const uint8_t* blocks, float* x, int64_t n, void* stream);

// x (n floats) -> ggml's block_q8_K layout: 292 bytes per 256 elements
//   { float d ; int8_t qs[256] ; int16_t bsums[16] }
// n must be a multiple of 256.  `stream` may be null, in which case the call synchronises.
//
// This is `quantize_row_q8_K_ref` (ggml-quants.c L2768) transcribed, including the two things that look like
// mistakes and are not:
//
//   * the scale is `-127/max`, NOT `-128/max`.  The source carries the `-128` version COMMENTED OUT with the
//     note that IQ2_XXS needs the change for an awkward AVX implementation.  Getting this wrong is a 0.79%
//     scaling error - the same order as the quantisation step, so it yields a working-looking activation that
//     ggml never sees.  It cost round 120.
//   * `max` is the SIGNED value at the position of the largest magnitude, so a POSITIVE max gives a NEGATIVE
//     iscale and the `+max` element maps to -127 while the `-max` element maps to +127.
//
// The rounding is `nearest_int`, which is round-half-to-EVEN via the 12582912.0f magic number - not
// `roundf`'s half-away-from-zero.  They differ only on exact ties, i.e. almost never on real data, which is
// precisely why the wrong one would survive any amount of end-to-end testing; the parity test constructs
// exact ties on purpose.
void quantize_q8_K(const float* x, uint8_t* blocks, int64_t n, void* stream);

// The inverse: each element becomes `d * qs[j]`, with `d = 1/iscale` stored as f32.
void dequant_q8_K(const uint8_t* blocks, float* x, int64_t n, void* stream);

}  // namespace strata::kernels
