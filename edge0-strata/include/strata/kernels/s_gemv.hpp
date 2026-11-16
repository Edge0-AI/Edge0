// include/strata/kernels/s_gemv.hpp - the S-family GEMV, host-callable (P2.S2).
//
// ONE kernel for S2, S4 and S8, because `docs/pack-format.md` gives them ONE decode:
//
//     value = cb[code] * scale + offset
//
// with a per-TYPE codebook, code bias, group size and offset presence.  That uniformity is the point of the
// canonical form - 13 source types collapse to three code widths and a handful of attributes - so the kernel
// takes those attributes as ARGUMENTS rather than being specialised per source type.  The manifest already
// carries them per tensor (round 163 added the tests that keep it honest).
#pragma once

#include <cstdint>

namespace strata::kernels {

// Which table turns a code into a number.  `Affine` is `code + bias`; `Iq4Nl` is the non-linear
// `kvalues_iq4nl`, which no bias can express - the reason the canonical form carries a codebook at all.
enum class Codebook : int { Affine = 0, Iq4Nl = 1 };

struct SForm {
    int code_bits = 2;              // 2, 4 or 8
    int code_bias = -1;             // subtracted from the code IN THE INTEGER DOMAIN before the multiply
    int group_elems = 64;           // elements per scale (and per offset, when present)
    Codebook codebook = Codebook::Affine;
    bool has_offset = false;        // only the K-quants with a min (Q4_K, Q5_K) have one

    /// WHICH ACTIVATION THIS WEIGHT'S `vec_dot_type` IS: 0 = Q8_0, 1 = Q8_K.
    ///
    /// **IT IS AN ATTRIBUTE OF THE CANONICAL FORM AND IT CANNOT BE DERIVED FROM THE OTHERS.**  Q5_0 and Q5_K
    /// are both 8-bit with bias -16 (only `has_offset` differs); IQ4_NL and IQ4_XS are both 4-bit with the
    /// IQ4NL codebook and no offset, and differ in NOTHING this struct otherwise holds.  So it is CARRIED from
    /// the manifest's `source_type` rather than reconstructed - see `docs/activation-contract.md` and the
    /// `act_kind` column `tools/pack_index.py` writes.
    int act_kind = 0;
};

// y[o] = sum_i fp16(x[i]) * W[i][o].  GGUF's ne[0] varies fastest, so a tensor is [n_in, n_out] with n_in
// CONTIGUOUS: row o is a contiguous run of n_in/group_elems groups.
//
// `codes`  n_out * (n_in / (8/code_bits)) bytes
// `scales` n_out * (n_in / group_elems) floats
// `offset` the same count as scales, or nullptr when `has_offset` is false
// `y`      n_out floats
void s_gemv(const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset, float* y,
            int64_t n_in, int64_t n_out, const SForm& form);

// The same computation with the ROW SPLIT ACROSS THREADS: one block of `threads_per_row` threads per output
// row, each accumulating a strided subset, reduced through shared memory.
//
// WHY BOTH EXIST.  `s_gemv` gives one thread per output row, so its parallelism is the OUTPUT WIDTH - and the
// expert gate/up shape is [2560 x 640], which is 640 threads over 48 SMs, 13 per SM.  Measured, that
// orientation ran 4.6x slower than [640 x 2560] with identical weights (Memory/LEDGER.md L1).  This variant
// makes the parallelism independent of the output width.
//
// It is NOT bit-identical to `s_gemv`: the partial sums are added in a different order, so the parity test
// carries a tolerance for exactly this reason.  Both are kept and both are tested, because the naive one is
// the reference the split one is checked against.
void s_gemv_split(const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset,
                  float* y, int64_t n_in, int64_t n_out, const SForm& form, int threads_per_row);

// S2 ONLY, with the code load amortised over four consecutive elements (they share one byte) and the
// activation load over four halves (one 64-bit word).  S2 is 31.64 GiB of the 38 GiB pack, so it is the kernel
// whose instruction count matters.  No `offset` - S2 has none, and a parameter that is always null is a
// parameter that will eventually be passed by mistake.
void s2_gemv_quads(const uint16_t* x, const uint8_t* codes, const float* scales, float* y, int64_t n_in,
                   int64_t n_out, int threads_per_row);

// S2 with two further instruction reductions: `x` staged in SHARED memory once per block instead of re-read
// from global by every output row, and the code unpack replaced by a 256-entry constant-memory table of
// float4 (one broadcast load per four elements instead of shifts, masks and an int-to-float convert).
// `stage_x` selects the staging so the two can be measured against each other rather than assumed.
void s2_gemv_fast(const uint16_t* x, const uint8_t* codes, const float* scales, float* y, int64_t n_in,
                  int64_t n_out, int threads_per_row, bool stage_x);

// The same computation as `s_gemv_split` but launched on `stream` (a cudaStream_t passed as void*, so this
// header stays free of CUDA types) and WITHOUT synchronising.  It exists because the architecture's premise -
// that the expert stream overlaps with the compute that consumes it - cannot be measured through an entry
// point that waits for every stream.  The synchronous ones remain the default; this is for the loop that has
// to overlap, and for the test that establishes whether overlapping is worth building.
void s_gemv_split_async(const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset,
                        float* y, int64_t n_in, int64_t n_out, const SForm& form, int threads_per_row,
                        void* stream);

/// THE SAME GEMV OVER **Q8_K** ACTIVATIONS, which is the contract for every K-quant and IQ-quant weight.
///
/// `docs/activation-contract.md` settles this: `ggml_mul_mat` converts the activation to the weight's
/// `vec_dot_type`, and for Q3_K/Q4_K/Q5_K/Q6_K/IQ4_XS that type is Q8_K - NOT fp16.  The two differ by
/// 0.6-0.9% per GEMV against a 1e-3 tolerance, and the K-quants are 2.89 GiB of the dense weights: the
/// attention projections and `ssm_out`, i.e. the numerically sensitive ones.
///
/// `x` is the block_q8_K layout, 292 bytes per 256 elements: `{ f32 d ; int8_t qs[256] ; int16_t bsums[16] }`.
/// `bsums` is NOT read here - it exists for ggml's AVX2 path - but the STRIDE includes it, so a buffer
/// produced by `quantize_q8_K` can be passed straight in.
///
/// The activation is dequantized on the fly (`d * qs[i]`), one multiply per element, exactly as the fp16 path
/// does one `__half2float` per element.  The weight decode is the SAME code as `s_gemv` - one implementation,
/// because a second decode of the canonical form is a second thing to get wrong.
void s_gemv_q8k(const uint8_t* x_q8k, const uint8_t* codes, const float* scales, const float* offset,
                float* y, int64_t n_in, int64_t n_out, const SForm& form, void* stream);

/// The row-split form, for output widths too small to fill the machine.
///
/// **THERE IS NO `threads_per_row` HERE, AND THAT IS A FIX RATHER THAN AN OVERSIGHT.**  The parameter used to be
/// in this signature and the implementation threw it away - `(void) threads_per_row;  // informational` - so a
/// caller reading the declaration reasonably believed it was tunable, and round 224 spent a whole sweep
/// measuring it and concluding from the flat result that the kernel was instruction-bound.  The sweep could not
/// have shown anything else: **it was measuring a knob that is not connected.**
///
/// This form is WARP per output row, which is what the shape wants, and the parameter is gone from the
/// interface so that no one can believe otherwise.  `s_gemv_split`/`s_gemv_split_async` DO take threads_per_row
/// and DO use it, if a caller wants to tune that shape.
void s_gemv_q8k_split(const uint8_t* x_q8k, const uint8_t* codes, const float* scales, const float* offset,
                      float* y, int64_t n_in, int64_t n_out, const SForm& form, void* stream);

/// THE SAME, OVER A **Q8_0** ACTIVATION - `block_q8_0`, 34 bytes per 32 elements, from `quantize_q8_0`.
///
/// **THIS IS THE KERNEL THAT WAS MISSING, AND ITS ABSENCE WAS DESCRIBED INCORRECTLY FOR SEVERAL ROUNDS.**  The
/// legacy block formats (Q2_0, Q4_0, Q5_0, Q8_0, IQ4_NL) have `vec_dot_type` Q8_0, not Q8_K.  Every one of them
/// in this pack except `ffn_down_shexp` is `Q2_0`, whose attributes happen to BE S2's (2 bits, group 64,
/// bias -1) - so `s2_gemv_q8`, which hardcodes those, covered them by coincidence.  `ffn_down_shexp` is
/// IQ4_NL/Q4_0/Q5_0/Q8_0 in every layer with **group 32 and bias -16/0/-8**, and its `n_in` is **640**.
///
/// 640 is a multiple of 32 and NOT of 256, so **Q8_K is structurally impossible for it rather than merely
/// unimplemented** - which is what the "Q8_K is not implemented" note in `shared_expert.hpp` was really
/// pointing at.
///
/// `n_in` must be a multiple of 32.  `group_elems` must be a power of two and a multiple of four, the same two
/// requirements `s_gemv_q8k_split` has and for the same reasons.
void s_gemv_q8_0_split(const uint8_t* x_q8_0, const uint8_t* codes, const float* scales, const float* offset,
                       float* y, int64_t n_in, int64_t n_out, const SForm& form, void* stream);

/// GATE/UP DUAL GEMV + SwiGLU IN ONE KERNEL (llama.cpp's `mul_mat_gated`): `out[o] = silu(gate_row_o) *
/// up_row_o`, one read of the activation for both projections, the `up` intermediate never materialised.
///
/// Each row is accumulated by the same row function its standalone kernel uses
/// (`strata/kernels/s_rowdev.cuh`) and the SwiGLU is the shared expression (`strata/kernels/swiglu.cuh`),
/// so the output is BITWISE what `gemv(gate)` + `gemv(up)` + the standalone SwiGLU produced - checked by
/// `fusions_parity`.  Both forms may be any S-family form (the two sides are dispatched independently;
/// a Q8_0 side and a Q8_K side in one kernel is fine).
///
/// `swilu_kind`: 0 = the double-precision `ref/moe.py` form, 1 = the float `__expf` form, 2 = the pinned
/// CUDA native form.  Returns false when a side's form is outside the warp-per-row kernel's contract
/// (the caller keeps its unfused sequence).
bool s_gemv_pair_silu(const uint8_t* x_q8_0, const uint8_t* x_q8k, const SForm& form_g, const uint8_t* codes_g,
                      const float* scales_g, const float* off_g, const SForm& form_u, const uint8_t* codes_u,
                      const float* scales_u, const float* off_u, float* out, int64_t n_in, int64_t n_out,
                      int threads_per_row, int swilu_kind, void* stream);

}  // namespace strata::kernels
