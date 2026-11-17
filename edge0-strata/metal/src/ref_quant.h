// metal/src/ref_quant.h - CPU reference dequantizers for the parity programs, transcribed from ggml's
// dequantize_row_* (ggml-quants.c at the pinned 3cf03257) and Strata's src/artifact/dequant.hpp (Q2_0), plus a
// deterministic generator of random-but-valid blocks for synthetic weights.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ref {

// GGML type ids (stable, as in native_mmvq.hpp)
enum : int {
    F32 = 0, F16 = 1, Q4_0 = 2, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14,
    IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18, IQ1_S = 19, IQ1_M = 29, IQ3_S = 21, IQ2_S = 22, IQ4_NL = 20, IQ4_XS = 23, BF16 = 30, Q2_0 = 42,
    TQ2_T = 144, TQK6 = 145, TQK7 = 146,   // agentionai's trellis types (Gyro): 128 values a block, 34 / 26 / 30 bytes
};

const char* type_name(int t);
int block_elems(int t);
int block_bytes(int t);
inline size_t row_bytes(int t, int n) { return (size_t)n / block_elems(t) * block_bytes(t); }

float f16_to_f32(uint16_t h);
uint16_t f32_to_f16(float f);
float bf16_to_f32(uint16_t h);
uint16_t f32_to_bf16(float f);   // round to nearest even

// Dequantizes `n` elements (a whole number of blocks) of type t.
void dequant(int t, const void* src, float* dst, size_t n);

// ggml's quantize_row_q8_0_ref: per 32 values d = max|x| / 127 (stored fp16), q = roundf(x * (1 / d)).
void quantize_q8_0(const float* x, size_t n, uint8_t* out);

// Fills `bytes` with random valid blocks of type t for a matrix of rows x cols: random codes, block scales drawn so
// that the weights have roughly unit-free magnitude `scale` (the d fields are finite FP16).
void random_blocks(int t, uint8_t* dst, int rows, int cols, uint64_t seed, float scale = 0.02f);

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    float uni() { return (next() >> 40) * (1.0f / 16777216.0f); }            // [0,1)
    float sym() { return uni() * 2.0f - 1.0f; }                             // [-1,1)
};

}  // namespace ref
