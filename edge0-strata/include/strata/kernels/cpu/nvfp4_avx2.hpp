// include/strata/kernels/cpu/nvfp4_avx2.hpp - NVFP4 expert rows in 256-bit lanes, several tokens at once, for CPUs
// without AVX-512 (Zen 2/3, Intel 12th-14th gen).
//
// ggml-cpu's NVFP4 dot product (ggml_vec_dot_nvfp4_q8_0) re-decodes a row's blocks for every token of a window;
// these decode a 64-value block once and apply it to each token. Each token's arithmetic is that dot product's AVX2
// path step for step (the same integer sums, the same FMA order, the same horizontal sum), so a row equals
// ggml-cpu's bit for bit - what an AVX2 CPU computed before these existed. Activations are Q8_0.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// Whether rows of `n` values fit the kernels (whole 64-value blocks, at most 64 of them).
bool nvfp4_256_fits(int n);

/// Gate/up rows [r0, r1) for `nt` tokens: ff[t][r] = silu(s_gate * g) * (s_up * u), as nvfp4_512_gu_rows.
void nvfp4_256_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                       float* const* ff, int r0, int r1, float s_gate, float s_up);

/// Plain rows [r0, r1) against `nt` Q8_0 activations of `n` values, times `scale`, as nvfp4_512_rows.
void nvfp4_256_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                    int r0, int r1, float scale = 1.f);

}  // namespace strata::kernels::cpu
