// Whole-tensor load of non-expert weights and fp32 dequant (q*scale + bias, little-endian).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "safio.hpp"

namespace e0n {

struct DenseF32 {
  int64_t rows = 0, cols = 0;
  std::vector<float> v;  // row-major [rows, cols]
};

bool read_dense_f32(const ModelFiles& files, const std::string& tensor, DenseF32& out,
                    std::string& err);

// Quantized tensor (last dim is packed: bits=4 → ×8, bits=8 → ×4).
// weight name like "...gate.weight"; scales/biases take the same suffix. Output [R, C] fp32.
bool read_dense_dequant(const ModelFiles& files, const std::string& tensor, int64_t bits,
                        int64_t group_size, DenseF32& out, std::string& err);

inline float bf16_bits_to_f(uint16_t b) {
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  __builtin_memcpy(&f, &u, 4);
  return f;
}
float f16_bits_to_f(uint16_t b);

void dequant_row(const uint32_t* packed, const uint16_t* scales_row,
                 const uint16_t* biases_row, int64_t cols_packed, int bits,
                 int64_t group_size, std::vector<float>& dst_row);

}  // namespace e0n
