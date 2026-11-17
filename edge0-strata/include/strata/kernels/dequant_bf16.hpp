// include/strata/kernels/dequant_bf16.hpp - plan v0.3 P5: native GGUF blocks -> BF16 on the device, for the
// tensor-core GEMMs of batched prompt processing.
//
// One thread per 32-element group, transcribed from the reference `dequantize_row_*` functions of the pinned
// llama.cpp (`ggml/src/ggml-quants.c`, MIT): Q4_0 (2), Q5_0 (6), Q8_0 (8), Q3_K (11), Q4_K (12), Q5_K (13), Q6_K (14),
// IQ4_NL (20), IQ4_XS (23) and Q2_0 (42).  A row-major (n_rows, n_cols) tensor of blocks becomes a row-major BF16
// matrix; `n_cols` must be a multiple of the type's block size.
#pragma once

#include <cstdint>

namespace strata::kernels {

bool dequant_bf16_supported(int ggml_type) noexcept;

/// `rows` rows of `cols` values starting at block row `row0` (so a caller can dequantize a slice of a tensor).
void dequant_bf16(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, uint16_t* out,
                  void* stream);

/// The same into FP16 bits (the prompt path's quantized-weight GEMMs: Q2_0 values are exact in FP16).
void dequant_f16(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, uint16_t* out,
                 void* stream);

/// dequant_f16 with output row stride `ld` >= cols (a multiple of 8): S23's padded weight scratch.  False (nothing
/// launched) for the i-quant formats and unsupported shapes.
bool dequant_f16_ld(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, int64_t ld,
                    uint16_t* out, void* stream);

/// The same into FP32 (tests and small tensors).
void dequant_f32(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, float* out, void* stream);

/// The wide kernels (a thread per 8 values, bitwise the same output; Q4_K, Q5_K, Q6_K, IQ4_XS): 0 off, 1 on, -1 the
/// default (STRATA_DEQUANT_WIDE, else on for compute capability 6.1).  For dequant_wide_parity.
void dequant_set_wide(int mode);

}  // namespace strata::kernels
