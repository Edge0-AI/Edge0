// Q5_1 shared-expert projection: Strata CUDA dequantization vs ggml.
#define NOMINMAX
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "ggml.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: q5_projection_parity <shard1.gguf>\n");
        return 2;
    }
    strata::GgufFile gguf(argv[1]);
    const auto* t = gguf.find("blk.0.ffn_down_shexp.weight");
    if (!t || t->type != 7 || t->shape.size() != 2) {
        std::fprintf(stderr, "expected blk.0.ffn_down_shexp.weight Q5_1\n");
        return 2;
    }
    const int64_t cols = (int64_t) t->shape[0], rows = std::min<int64_t>(4, (int64_t) t->shape[1]);
    const size_t weight_bytes = (size_t) rows * (size_t) (cols / 32) * 24;
    const size_t elements = (size_t) rows * (size_t) cols;
    const uint8_t* source = gguf.tensor_data(*t);
    void *dweight = nullptr, *dfloat = nullptr, *dbf16 = nullptr;
    if (cudaMalloc(&dweight, weight_bytes) != cudaSuccess ||
        cudaMalloc(&dfloat, elements * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dbf16, elements * sizeof(uint16_t)) != cudaSuccess) {
        std::fprintf(stderr, "CUDA allocation failed\n");
        return 1;
    }
    cudaMemcpy(dweight, source, weight_bytes, cudaMemcpyHostToDevice);
    strata::kernels::dequant_f32(7, dweight, 0, rows, cols, (float*) dfloat, nullptr);
    strata::kernels::dequant_bf16(7, dweight, 0, rows, cols, (uint16_t*) dbf16, nullptr);
    std::vector<float> got(elements), ref(elements);
    std::vector<uint16_t> got_bf16(elements);
    cudaMemcpy(got.data(), dfloat, elements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(got_bf16.data(), dbf16, elements * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaFree(dweight); cudaFree(dfloat); cudaFree(dbf16);
    const auto* traits = ggml_get_type_traits(GGML_TYPE_Q5_1);
    for (int64_t r = 0; r < rows; ++r)
        traits->to_float(source + (size_t) r * (size_t) (cols / 32) * 24,
                         ref.data() + (size_t) r * (size_t) cols, cols);
    double max_abs = 0;
    size_t bf16_bad = 0;
    for (size_t i = 0; i < elements; ++i) {
        max_abs = std::max(max_abs, (double) std::fabs(got[i] - ref[i]));
        uint32_t bits;
        std::memcpy(&bits, &ref[i], sizeof(bits));
        const uint16_t expected = (uint16_t) ((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
        if (got_bf16[i] != expected) ++bf16_bad;
    }
    const bool ok = max_abs <= 1e-6 && bf16_bad == 0;
    std::printf("Q5_1 projection: %zu elements, max_abs %.3e, bf16 differences %zu %s\n",
                elements, max_abs, bf16_bad, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
