// include/strata/kernels/cpu/kq_avx2.hpp - AVX-2 multi-token dot products for Unsloth UD-Q4_K_XL's expert formats
// (Q4_K gate/up against Q8_K activations; Q5_1 down against Q8_1, Q8_0 down against Q8_0), bit-exact against
// ggml-cpu's AVX2 vec_dot for every token.  See src/kernels/cpu/kq_avx2.cpp.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// Q4_K (12), Q5_1 (7), Q8_0 (8).
bool kq256_supported(int ggml_type) noexcept;

/// edge0 latent: the qs loop of the q8_1 quant (per 32 values: amax/sum stay CALLER-side scalar,
/// this does only the roundf/clamp/store part with AVX2 - _mm256_cvtps_epi32 is round-to-nearest-
/// even, roundf's mode, so every byte matches the scalar loop bit for bit).
void lat_quant_q8_1_qs_avx2(const float* x, const float* d_ptr, int8_t* q);

/// edge0 latent: the qs loop of the q8_1 quant (per 32 values: amax/sum stay CALLER-side scalar,
/// this does only the roundf/clamp/store part with AVX2 - _mm256_cvtps_epi32 is round-to-nearest-
/// even, roundf's mode, so every byte matches the scalar loop bit for bit).
void lat_quant_q8_1_qs_avx2(const float* x, const float* d_ptr, int8_t* q);

/// edge0 latent: the qs loop of the q8_1 quant (per 32 values: amax/sum stay CALLER-side scalar,
/// this does only the roundf/clamp/store part with AVX2 - _mm256_cvtps_epi32 is round-to-nearest-
/// even, roundf's mode, so every byte matches the scalar loop bit for bit).
void lat_quant_q8_1_qs_avx2(const float* x, const float* d_ptr, int8_t* q);
/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]), rows [r0, r1); gate rows at blob, up rows at blob + up_off.
void kq256_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                   int nt, float* const* ff, int r0, int r1);
/// out[t][r] = w_r . a[t], rows [r0, r1).
void kq256_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                float* const* out, int r0, int r1);

/// out[r] = w_r . x for BF16 rows (bits), fp32 x: the routing-aware prefetch's router (an estimate only).
void bf16_rows_dot(const uint16_t* w, int rows, int cols, const float* x, float* out);
/// The same for `nt` <= 8 tokens (x: nt rows of cols, cols % 8 == 0), out[t * rows + r].
void bf16_rows_dot_multi(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out);

/// edge0 latent MoE: out[t][r] = w_r . act[t], w is Q8_0 rows (row_bytes), act is Q8_1 rows of n values.
void lat_q8_0_q8_1_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                         float* const* out, int r0, int r1);
/// Q4DU: the same with Q4_0 weight rows (18 B/block: {f16 d; uint8 nib[16]}, element j in byte j's low
/// nibble, j+16 in its high one, value (code - 8) * d).  Integer chain as the AVX2 Q8_0 dot: maddubs of
/// the unsigned nibbles against the q8_1 codes then the per-byte +/- split - ggml's own q4_0 x q8_0
/// AVX2 form - with dx * dw applied per block in the same float order.
void lat_q4_0_q8_1_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                         float* const* out, int r0, int r1);

}  // namespace strata::kernels::cpu
