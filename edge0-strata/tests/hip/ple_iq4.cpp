// Real PLE IQ3_XXS/IQ4_XS matrix parity: ggml CPU dequantization versus HIP native MMVQ.
// Usage: hip_ple_iq4 <model-shard-1.gguf> [expected-type-id]
#include <hip/hip_runtime.h>

#include "ggml.h"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kNIn = 2560;
constexpr int kNOut = 10240;
constexpr int kTypeIq3Xxs = 18;
constexpr int kTypeIq4Xs = 23;
constexpr int kReplayCount = 3;

#define CHECK(call)                                                                                                  \
    do {                                                                                                             \
        const hipError_t error = (call);                                                                             \
        if (error != hipSuccess) {                                                                                   \
            std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(error));                                    \
            return 2;                                                                                                \
        }                                                                                                            \
    } while (0)

struct Q8Reference {
    std::vector<float> dequantized;
};

Q8Reference quantize_q8_1_reference(const std::vector<float>& x) {
    Q8Reference result;
    result.dequantized.resize(x.size());
    for (size_t offset = 0; offset < x.size(); offset += 32) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(x[offset + i]));
        if (amax == 0.0f) {
            std::fill(result.dequantized.begin() + offset, result.dequantized.begin() + offset + 32, 0.0f);
            continue;
        }
        const float scale_f32 = amax / 127.0f;
        const float scale_f16 = strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(scale_f32));
        for (int i = 0; i < 32; ++i) {
            const int q = static_cast<int>(std::round(x[offset + i] / scale_f32));
            result.dequantized[offset + i] = static_cast<float>(static_cast<int8_t>(q)) * scale_f16;
        }
    }
    return result;
}

struct Parity {
    double rel_l1_q8 = 0.0;
    double worst_terms_q8 = 0.0;
    double rel_l1_f32 = 0.0;
    double max_abs_q8 = 0.0;
    bool finite = true;
};

Parity compare(const uint8_t* weights, size_t row_bytes, const ggml_type_traits* traits,
               const std::vector<float>& x, const std::vector<float>& x_q8,
               const std::vector<float>& actual) {
    std::vector<float> row(kNIn);
    double q8_error = 0.0, q8_magnitude = 0.0;
    double f32_error = 0.0, f32_magnitude = 0.0;
    Parity result;
    for (int r = 0; r < kNOut; ++r) {
        traits->to_float(weights + static_cast<size_t>(r) * row_bytes, row.data(), kNIn);
        double expected_q8 = 0.0, terms_q8 = 0.0, expected_f32 = 0.0;
        for (int i = 0; i < kNIn; ++i) {
            const double term_q8 = static_cast<double>(row[i]) * x_q8[i];
            const double term_f32 = static_cast<double>(row[i]) * x[i];
            expected_q8 += term_q8;
            terms_q8 += std::fabs(term_q8);
            expected_f32 += term_f32;
        }
        const double got = actual[r];
        if (!std::isfinite(got) || !std::isfinite(expected_q8) || !std::isfinite(expected_f32)) result.finite = false;
        q8_error += std::fabs(got - expected_q8);
        q8_magnitude += std::fabs(expected_q8);
        f32_error += std::fabs(got - expected_f32);
        f32_magnitude += std::fabs(expected_f32);
        result.max_abs_q8 = std::fmax(result.max_abs_q8, std::fabs(got - expected_q8));
        result.worst_terms_q8 = std::fmax(result.worst_terms_q8,
                                         std::fabs(got - expected_q8) / std::fmax(terms_q8, 1e-30));
    }
    result.rel_l1_q8 = q8_error / std::fmax(q8_magnitude, 1e-30);
    result.rel_l1_f32 = f32_error / std::fmax(f32_magnitude, 1e-30);
    return result;
}

std::vector<float> make_input(uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 0.6f);
    std::vector<float> x(kNIn);
    for (int i = 0; i < kNIn; ++i)
        x[i] = normal(rng) + 0.1f * std::sin(static_cast<float>(i) * 0.013f);
    return x;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        std::fprintf(stderr, "usage: hip_ple_iq4 <model-shard-1.gguf> [expected-type-id]\n");
        return 2;
    }

    try {
        strata::GgufFile gguf(argv[1]);
        const strata::TensorInfo* tensor = gguf.find("blk.1.ple_key.weight");
        if (!tensor || (tensor->type != kTypeIq3Xxs && tensor->type != kTypeIq4Xs) || tensor->shape.size() != 2 ||
            tensor->shape[0] != kNIn || tensor->shape[1] != kNOut) {
            std::fprintf(stderr, "expected blk.1.ple_key.weight IQ3_XXS or IQ4_XS [2560,10240] in %s\n", argv[1]);
            return 1;
        }
        if (argc == 3 && std::atoi(argv[2]) != static_cast<int>(tensor->type)) {
            std::fprintf(stderr, "requested GGML type %s but blk.1.ple_key.weight is type %u (%s)\n",
                         argv[2], tensor->type, tensor->type_name());
            return 1;
        }
        const auto* traits = ggml_get_type_traits(static_cast<ggml_type>(tensor->type));
        if (!traits || !traits->to_float) {
            std::fprintf(stderr, "ggml has no %s CPU dequantizer\n", tensor->type_name());
            return 1;
        }
        const uint8_t* host_weights = gguf.tensor_data(*tensor);
        const size_t weight_bytes = strata::kernels::native_mmvq_weight_bytes(static_cast<int>(tensor->type), kNIn, kNOut);
        const size_t row_bytes = weight_bytes / kNOut;
        const size_t expected_bytes_per_block = tensor->type == kTypeIq3Xxs ? 98 : 136;
        if (weight_bytes != static_cast<size_t>(tensor->elements()) / 256 * expected_bytes_per_block ||
            row_bytes != static_cast<size_t>(kNIn / 256) * expected_bytes_per_block) {
            std::fprintf(stderr, "unexpected %s storage geometry: %zu bytes, row %zu\n",
                         tensor->type_name(), weight_bytes, row_bytes);
            return 1;
        }

        uint8_t* device_weights = nullptr;
        float* device_x = nullptr;
        float* device_y = nullptr;
        void* device_q8_1 = nullptr;
        CHECK(hipMalloc(reinterpret_cast<void**>(&device_weights), weight_bytes));
        CHECK(hipMalloc(reinterpret_cast<void**>(&device_x), kNIn * sizeof(float)));
        CHECK(hipMalloc(reinterpret_cast<void**>(&device_y), kNOut * sizeof(float)));
        CHECK(hipMalloc(&device_q8_1, strata::kernels::native_q8_1_bytes(kNIn)));
        CHECK(hipMemcpy(device_weights, host_weights, weight_bytes, hipMemcpyHostToDevice));

        hipStream_t stream = nullptr;
        CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
        strata::kernels::native_quantize_q8_1(device_x, device_q8_1, kNIn, 1, static_cast<void*>(stream));
        strata::kernels::native_mmvq(static_cast<int>(tensor->type), device_weights, device_q8_1, device_y,
                                     kNIn, kNOut, 1, static_cast<void*>(stream));
        hipGraph_t graph = nullptr;
        CHECK(hipStreamEndCapture(stream, &graph));
        hipGraphExec_t graph_exec = nullptr;
        CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

        std::vector<float> actual(kNOut);
        bool ok = true;
        for (int replay = 0; replay < kReplayCount; ++replay) {
            const std::vector<float> x = make_input(0x1572u + static_cast<uint32_t>(replay));
            const Q8Reference q8 = quantize_q8_1_reference(x);
            CHECK(hipMemcpyAsync(device_x, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice, stream));
            CHECK(hipGraphLaunch(graph_exec, stream));
            CHECK(hipMemcpyAsync(actual.data(), device_y, actual.size() * sizeof(float), hipMemcpyDeviceToHost, stream));
            CHECK(hipStreamSynchronize(stream));

            const Parity parity = compare(host_weights, row_bytes, traits, x, q8.dequantized, actual);
            const bool pass = parity.finite && parity.rel_l1_q8 <= 5e-5 && parity.worst_terms_q8 <= 1e-4 &&
                              parity.rel_l1_f32 <= 3e-2;
            std::printf("%s PLE graph replay %d/%d: q8-ref rel-L1 %.3e, worst terms %.3e, "
                        "original-f32 rel-L1 %.3e, max q8 abs %.3e %s\n",
                        tensor->type_name(), replay + 1, kReplayCount, parity.rel_l1_q8, parity.worst_terms_q8,
                        parity.rel_l1_f32, parity.max_abs_q8, pass ? "PASS" : "FAIL");
            ok = ok && pass;
        }

        CHECK(hipGraphExecDestroy(graph_exec));
        CHECK(hipGraphDestroy(graph));
        CHECK(hipStreamDestroy(stream));
        CHECK(hipFree(device_q8_1));
        CHECK(hipFree(device_y));
        CHECK(hipFree(device_x));
        CHECK(hipFree(device_weights));
        if (!ok) return 1;
        std::printf("PLE %s native MMVQ parity OK: real GGUF matrix, CPU dequant reference, %d graph replays\n",
                    tensor->type_name(), kReplayCount);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "hip_ple_iq4: %s\n", error.what());
        return 1;
    }
}
