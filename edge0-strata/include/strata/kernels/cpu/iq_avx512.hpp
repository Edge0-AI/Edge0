// include/strata/kernels/cpu/iq_avx512.hpp - plan v0.3 P6: AVX-512 multi-token dot products for the i-quant
// expert formats (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S) against Q8_K activations (ggml's block_q8_K).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

bool iq512_supported(int ggml_type) noexcept;
/// Whether the grid lookups of these rows are gathers (the default where cpu_gather_fast(), STRATA_IQ_GATHER=1 / 0 force it): the
/// same words, so the same bits as the scalar lookups, and 1.3-2x faster where a gather is cheap (Sapphire Rapids).
bool iq512_gather_on() noexcept;
/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]), rows [r0, r1); gate rows at blob, up rows at blob + up_off.
void iq512_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                   int nt, float* const* ff, int r0, int r1);
/// out[t][r] = w_r . a[t], rows [r0, r1).
void iq512_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                float* const* out, int r0, int r1);

/// The DeepSeek-V4.1 CPU lane's rows (src/dsv41/ds4_gg_cpu.cpp): the same 64-value decode, summed in ggml's order, so a token's
/// dot is ggml-cpu's AVX2 vec_dot bit for bit at any number of tokens (1-6 in one pass, 7 and more as a prompt chunk's tiles).
/// The caller checks cpu_avx512_ok() (AVX-512 F/BW/VL/DQ/VNNI/VBMI).
bool iq512x_supported(int ggml_type) noexcept;
void iq512x_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                 float* const* out, int r0, int r1);

/// Singleton-equivalent accumulation for types 18, 21 and 22 (the linked ggml AVX2 order, one token or many); the caller
/// checks the linked ggml ISA.  Used only for window-invariant native arithmetic.
void iq512_reference_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                             const void* const* act, int nt, float* const* ff, int r0, int r1);

}  // namespace strata::kernels::cpu
