// Device parity for the scalar HIP implementation of the optional native QSA scorer.
#include <hip/hip_runtime.h>
#include "strata/kernels/native_qsa_score.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

#define CHECK(call)                                                                                                  \
    do {                                                                                                             \
        const hipError_t error = (call);                                                                             \
        if (error != hipSuccess) {                                                                                   \
            std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(error));                                     \
            return 2;                                                                                                \
        }                                                                                                            \
    } while (0)

int main() {
    using namespace strata::kernels;
    constexpr int kDim = 128, kHeads = 4, kBlock = 4;
    constexpr int kMaxCells = 8, kMaxBlocks = 3, kCells = 7;
    const QsaShapes shapes = qsa_real_shapes();

    std::vector<float> pooled(kMaxBlocks * kDim), query(kHeads * kDim), bias(kMaxBlocks),
        scores(kMaxCells, -123.0f), expected(kMaxCells, -123.0f);
    const float head_sign[kHeads] = {1.0f, -1.0f, 1.0f, -1.0f};
    for (int row = 0; row < kMaxBlocks; ++row)
        for (int d = 0; d < kDim; ++d)
            pooled[row * kDim + d] = static_cast<float>(row + 1) * (1.0f + static_cast<float>(d % 7) / 8.0f) / 128.0f;
    for (int head = 0; head < kHeads; ++head)
        for (int d = 0; d < kDim; ++d)
            query[head * kDim + d] = head_sign[head] * (1.0f + static_cast<float>((head + d) % 5) / 16.0f) / 16.0f;
    for (int row = 0; row < kMaxBlocks; ++row) bias[row] = (static_cast<float>(row) - 1.0f) / 8.0f;

    int32_t step[kStepCount] = {kCells - 1, kCells, kCells / kBlock, static_cast<int32_t>(kCells)};
    bool saw_positive_head = false, saw_negative_head = false;
    for (int row = 0; row <= step[kStepNBid]; ++row) {
        float sum = 0.0f;
        for (int head = 0; head < kHeads; ++head) {
            float dot = 0.0f;
            for (int d = 0; d < kDim; ++d)
                dot = std::fma(pooled[row * kDim + d], query[head * kDim + d], dot);
            saw_positive_head = saw_positive_head || dot > 0.0f;
            saw_negative_head = saw_negative_head || dot < 0.0f;
            sum = sum + (dot > 0.0f ? dot : 0.0f);
        }
        sum = sum + bias[row];
        if (row == step[kStepNBid] && kCells % kBlock != 0) sum = sum + 1.0e9f;
        const int lo = row * kBlock;
        const int hi = (row + 1) * kBlock < kCells ? (row + 1) * kBlock : kCells;
        for (int cell = lo; cell < hi; ++cell) expected[cell] = sum;
    }
    if (!saw_positive_head || !saw_negative_head) {
        std::fprintf(stderr, "fixture did not exercise both positive and negative head scores\n");
        return 1;
    }

    float *device_pooled = nullptr, *device_query = nullptr, *device_bias = nullptr, *device_scores = nullptr;
    int32_t* device_step = nullptr;
    CHECK(hipMalloc(reinterpret_cast<void**>(&device_pooled), pooled.size() * sizeof(float)));
    CHECK(hipMalloc(reinterpret_cast<void**>(&device_query), query.size() * sizeof(float)));
    CHECK(hipMalloc(reinterpret_cast<void**>(&device_bias), bias.size() * sizeof(float)));
    CHECK(hipMalloc(reinterpret_cast<void**>(&device_scores), scores.size() * sizeof(float)));
    CHECK(hipMalloc(reinterpret_cast<void**>(&device_step), sizeof(step)));
    CHECK(hipMemcpy(device_pooled, pooled.data(), pooled.size() * sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(device_query, query.data(), query.size() * sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(device_bias, bias.data(), bias.size() * sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(device_scores, scores.data(), scores.size() * sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(device_step, step, sizeof(step), hipMemcpyHostToDevice));

    hipStream_t stream;
    CHECK(hipStreamCreate(&stream));
    native_qsa_score(device_pooled, device_query, device_bias, shapes, device_step, kMaxBlocks, kMaxCells,
                     device_scores, static_cast<void*>(stream));
    CHECK(hipStreamSynchronize(stream));
    CHECK(hipMemcpy(scores.data(), device_scores, scores.size() * sizeof(float), hipMemcpyDeviceToHost));

    bool ok = true;
    for (int cell = 0; cell < kMaxCells; ++cell) {
        if (scores[cell] != expected[cell]) {
            std::fprintf(stderr, "score[%d] got %.9g, expected %.9g\n", cell, scores[cell], expected[cell]);
            ok = false;
        }
    }

    CHECK(hipStreamDestroy(stream));
    CHECK(hipFree(device_step));
    CHECK(hipFree(device_scores));
    CHECK(hipFree(device_bias));
    CHECK(hipFree(device_query));
    CHECK(hipFree(device_pooled));
    if (!ok) return 1;
    std::puts("native QSA HIP scalar parity OK: FMA order, per-head ReLU, bias, partial tail and padding");
    return 0;
}
