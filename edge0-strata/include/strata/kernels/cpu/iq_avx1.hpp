// include/strata/kernels/cpu/iq_avx1.hpp - AVX1 (SSE4.2 + AVX, no AVX2/FMA) multi-token expert rows
// for the older-CPU builds (STRATA_ISA_FLOOR=avx: Sandy/Ivy Bridge, Xeon E5 v1/v2, Bulldozer).
//
// The sibling of kq_avx1.cpp for the expert kernels: without these, every CPU expert on such a CPU runs on
// ggml-cpu's scalar `_generic` dots (every x86 i-quant dot is #if defined(__AVX2__)).  Everything these
// kernels need exists in 128-bit on that ISA: pshufb, sign_epi8 and maddubs are SSSE3, madd_epi16 is SSE2;
// the float accumulation is 128-bit SSE2 (no FMA).  No gathered variants: the non-gather Fmt32 decode of
// iq_avx2.cpp is the template and it is 128-bit-portable.
//
// Compiled for AVX (-mavx / /arch:AVX, see CMakeLists.txt) and called only behind cpu_avx1_ok(), so a CPU
// without AVX never reaches it.  The arithmetic is ggml's (the `_generic` references): a multi-token row is
// bit for bit the same kernel's single-token row for that token (#152), and rows match ggml-cpu's dot to
// rel <= 1e-5 (iq_avx1_parity checks both).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// gate/up formats with an AVX1 kernel: IQ3_XXS (18), IQ3_S (21), IQ2_S (22), IQ4_XS (23).
bool iq128_supported(int ggml_type) noexcept;
/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]), rows [r0, r1); gate rows at blob, up rows at blob + up_off.
/// Activations are ggml's quantized form for the format (Q8_K for every format here), as in native_gu_rows.
void iq128_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                   int nt, float* const* ff, int r0, int r1);

/// down formats with an AVX1 kernel: IQ4_NL (20), Q2_0 (42, the fork's type).  Activations: Q8_0 for both.
bool iq128_down_supported(int ggml_type) noexcept;
void iq128_down_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt,
                     float* const* out, int r0, int r1);

/// ggml's quantize_row_q8_K (x86 runs the scalar reference), byte-identical, in SSE4.2/AVX1: the older CPUs'
/// copy of q8k_quant_avx2 (no 256-bit integer ops, no FMA).  q8k_quant_parity checks the bytes.
void q8k_quant_avx1(const float* x, void* y, int64_t n);

}  // namespace strata::kernels::cpu
