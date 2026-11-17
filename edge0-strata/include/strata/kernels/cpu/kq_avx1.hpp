// include/strata/kernels/cpu/kq_avx1.hpp - the AVX1 router dot for older CPUs (from the Strata_Dirigo fork).
//
// bf16_rows_dot_multi (kq_avx2.cpp) is compiled with -mavx2 and uses instructions an AVX-only CPU does
// not have (`vpmovzxwd`, `vfmadd`), so an AVX-only host cannot call it at all.  This is the same
// arithmetic on AVX1 instructions: 8 float lanes per token, one weight stream per row shared by all
// `nt` tokens, one horizontal sum per (row, token).  AVX1 has no FMA, so the multiply and the add stay
// separate; it has no 256-bit integer ops, so the bf16 -> f32 widening runs in 128-bit halves.
#pragma once

#include <cstdint>

namespace strata::kernels::cpu {

/// Mirrors bf16_rows_dot_multi's arithmetic (same accumulation shape, mul+add instead of FMA).
/// Requires cols % 8 == 0 and nt <= 8, exactly like the AVX2 kernel it stands in for.
void bf16_rows_dot_multi_avx1(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out);

}  // namespace strata::kernels::cpu
