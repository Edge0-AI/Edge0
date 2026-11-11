// include/strata/kernels/s2_gemv_q8.hpp - the S2 GEMV over Q8_0-QUANTIZED activations.
//
// `ggml_mul_mat` converts src1 to the weight's `vec_dot_type`, which for Q2_0 is Q8_0.  The other CUDA kernels
// here take FP16 activations, and the CPU expert path already quantizes to int8 - so this is the missing GPU
// half of the same rule, and without it the two paths disagree on the same token.
//
// CORRECTNESS VERSION: the activation is dequantized inside the loop.  The speed win here is `__dp4a` (four
// int8 MACs per instruction) and belongs to Phase 3, once the numerics are settled.
#pragma once

#include <cstdint>

namespace strata::kernels {

// `act`   (n_in/32) ggml Q8_0 blocks, 34 bytes each, as `quantize_q8_0` produces
// `codes` n_out * (n_in/4) bytes, one 2-bit code byte per quad
// `scales` n_out * (n_in/64)
// `y`     n_out floats
void s2_gemv_q8(const uint8_t* act, const uint8_t* codes, const float* scales, float* y, int64_t n_in,
                int64_t n_out, int threads_per_row, void* stream);

}  // namespace strata::kernels
