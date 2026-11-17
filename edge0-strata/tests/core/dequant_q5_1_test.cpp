// dequantize_q5_1 (strata/artifact/dequant.hpp: the PLE table of a Q5_1 GGUF) against ggml's dequantize_row_q5_1,
// on synthetic blocks: random codes, high bits and scales, plus the edge codes 0 and 31 and a negative minimum.
// No model, no GPU.
#include "strata/artifact/dequant.hpp"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main() {
    constexpr int kBlocks = 4096;
    constexpr int kBlockBytes = 24;
    std::mt19937 rng(20261002);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(-0.05f, 0.05f);
    std::vector<uint8_t> blocks((size_t) kBlocks * kBlockBytes);
    for (int b = 0; b < kBlocks; ++b) {
        uint8_t* blk = blocks.data() + (size_t) b * kBlockBytes;
        const ggml_fp16_t d = ggml_fp32_to_fp16(scale(rng));
        const ggml_fp16_t m = ggml_fp32_to_fp16(b == 1 ? -1.5f : scale(rng) * 20.0f);
        std::memcpy(blk, &d, 2);
        std::memcpy(blk + 2, &m, 2);
        for (int i = 4; i < kBlockBytes; ++i) blk[i] = (uint8_t) byte(rng);
        if (b == 0) std::memset(blk + 4, 0x00, 20);   // every code 0: the value is m
        if (b == 1) std::memset(blk + 4, 0xFF, 20);   // every code 31 (all high bits set)
    }
    const auto* traits = ggml_get_type_traits(GGML_TYPE_Q5_1);
    std::vector<float> want((size_t) kBlocks * 32), got((size_t) kBlocks * 32);
    traits->to_float(blocks.data(), want.data(), (int64_t) want.size());
    for (int b = 0; b < kBlocks; ++b) strata::dequantize_q5_1(blocks.data() + (size_t) b * kBlockBytes, got.data() + b * 32);
    double max_abs = 0.0;
    for (size_t i = 0; i < want.size(); ++i) max_abs = std::fmax(max_abs, std::fabs((double) got[i] - want[i]));
    const bool ok = max_abs <= 1e-6;
    std::printf("Q5_1 dequant: %d blocks, max_abs %.3e %s\n", kBlocks, max_abs, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
