// include/strata/kernels/cpu/iq_avx2.hpp - AVX-2 multi-token dot products for the i-quant expert
// formats (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS) against Q8_K activations (ggml's block_q8_K).
//
// The AVX-512 kernel's scheme on the CPUs without it (AMD Zen 2/3, Intel Core 12th-14th gen): the weights of a
// 32-value chunk are decoded once per verify window and every token applies them with five instructions.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

bool iq256_supported(int ggml_type) noexcept;
/// Bit-identical to ggml's singleton gate/up for IQ2_XXS, IQ2_XS, IQ3_XXS and IQ3_S.
/// The caller checks AVX2 support and passes one Q8_K activation.
void iq256_gu_rows_exact_one(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                             const void* act, float* ff, int r0, int r1);

/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]), rows [r0, r1); gate rows at blob, up rows at blob + up_off.
void iq256_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                   int nt, float* const* ff, int r0, int r1);
/// out[t][r] = w_r . a[t], rows [r0, r1).
void iq256_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                float* const* out, int r0, int r1);

/// The DeepSeek-V4.1 CPU lane's rows (src/dsv41/ds4_gg_cpu.cpp).  The same loops as iq256_rows / iq256_gu_rows with ggml's float
/// order, so a token's dot is ggml-cpu's vec_dot bit for bit at any number of tokens (the main engine's iq256_* are unchanged),
/// and IQ1_M (29), whose ggml dot is single-token.  The caller checks AVX2 support.
bool iq256x_supported(int ggml_type) noexcept;
void iq256x_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                 float* const* out, int r0, int r1);

/// The kernels' variants, a mask.  Every variant computes the same bits; they differ in speed from core to core.
/// kIq256Gather: IQ3_XXS, IQ3_S and IQ2_S read their grid entries with one gather (STRATA_IQ256_GATHER).
/// kIq256Vnni: AVX-VNNI's vpdpwssd / vpdpbusd, every format and iq4nl256_down_rows (cpu_avxvnni_ok).
inline constexpr int kIq256Gather = 1;
inline constexpr int kIq256Vnni = 2;
/// The variant iq256_gu_rows / iq256_rows take on the calling thread.
int iq256_variant() noexcept;
/// Every variant bit this build and this CPU can run (tests and benches).
int iq256_variants() noexcept;
/// iq256_gu_rows / iq256_rows in a given variant (tests and benches; only bits of iq256_variants()).
void iq256_gu_rows_v(int variant, int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                     const void* const* act, int nt, float* const* ff, int r0, int r1);
void iq256_rows_v(int variant, int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act,
                  int nt, float* const* out, int r0, int r1);

/// ggml's quantize_row_q8_K (x86 runs the scalar reference), byte-identical, in AVX-2: n values -> n/256 block_q8_K.
void q8k_quant_avx2(const float* x, void* y, int64_t n);

/// out[t][r] = w_r . h[t] for IQ4_NL (type 20) rows against Q8_0 activations (ggml's block_q8_0).
/// IQ4_NL is a 32-value-block format, so this does not go through iq256_rows (QK_K blocks, Q8_K acts).
void iq4nl256_down_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt,
                        float* const* out, int r0, int r1);
/// The same in a given variant (only the kIq256Vnni bit matters here).
void iq4nl256_down_rows_v(int variant, const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt,
                          float* const* out, int r0, int r1);

// Singleton-equivalent IQ4_NL down rows; caller checks the linked ggml AVX2/FMA path.
void iq4nl_reference_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* hq,
                          int nt, float* const* out, int r0, int r1);

}  // namespace strata::kernels::cpu
