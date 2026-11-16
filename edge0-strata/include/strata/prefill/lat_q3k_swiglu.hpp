// include/strata/prefill/lat_q3k_swiglu.hpp - M2b of the q3k latent A stage: SwiGLU and the q8_1 of H.
#pragma once

#include <cstdint>

namespace strata::prefill {

/// Features per entry (n_ff): the gate/up row width and H's width.
constexpr int64_t lat_q3k_swiglu_ff = 640;
/// An entry's staged H row: n_ff / 64 = 10 chunks of 80 B (64 int8 codes, then the two 32-value blocks' fp16 d's).
constexpr int64_t lat_q3k_swiglu_hq_row = 800;

/// H = silu(gate) * up, quantized to q8_1 exactly as the engine's q8_1_store does (the block's amax over its 32
/// values, d = amax / 127 through q8_1_finite, code = round(x / d) held to +-127, amax == 0 -> all-zero block), and
/// written in the staged row layout the down half reads: a 32-value block b of an entry is chunk b / 2, sub-blocks
/// 2 * (b & 1) + (v / 16), its codes at q8_1_store's permuted position, its d at 64 + 2 * (b & 1).
///   gate/up: [rows][lat_q3k_swiglu_ff] fp32, entry-major (the layout lat_q3k_gu writes)
///   hq     : rows x lat_q3k_swiglu_hq_row bytes
///   hout   : optional (nullptr to skip): the float H it computed, [rows][lat_q3k_swiglu_ff] - the tests use it to
///            separate "SwiGLU right" from "quantizer right"
void lat_q3k_swiglu_q8(const float* gate, const float* up, int64_t rows, void* hq, float* hout, void* stream);

}  // namespace strata::prefill
