// include/strata/kernels/cpu/nvfp4_avx512.hpp - NVFP4 expert rows in 512-bit lanes, several tokens at once.
//
// ggml-cpu's NVFP4 dot product (ggml_vec_dot_nvfp4_q8_0) is AVX2 and single-token: every token of a verify window
// re-decodes the same weights. These decode a 64-value block once and apply it to every token. The arithmetic is
// ggml's: an exact integer sum per 16-value sub-block, times the sub-block's UE4M3 scale and the Q8_0 block's scale;
// only the order of the float additions differs. Activations are Q8_0, NVFP4's vec_dot_type in ggml-cpu.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// Whether rows of `n` values fit the kernels (whole 64-value blocks, at most 64 of them).
bool nvfp4_512_fits(int n);

/// Gate/up rows [r0, r1) of an NVFP4 expert blob for `nt` tokens: ff[t][r] = silu(s_gate * g) * (s_up * u), where g
/// and u are the raw NVFP4 dot products of gate row r and up row r (at `up_off`) with activation t (Q8_0, `n` values).
/// `s_gate` / `s_up` are the expert's global scales from its blob tail.
void nvfp4_512_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                       float* const* ff, int r0, int r1, float s_gate, float s_up);

/// Plain rows [r0, r1) of an NVFP4 matrix (`row_bytes` apart) against `nt` Q8_0 activations of `n` values, times
/// `scale` (the down matrix's s_down).
void nvfp4_512_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                    int r0, int r1, float scale = 1.f);

}  // namespace strata::kernels::cpu
