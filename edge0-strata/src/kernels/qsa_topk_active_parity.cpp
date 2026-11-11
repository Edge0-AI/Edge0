// Capacity 262144, live prefill up to the register boundary, and the legacy fallbacks.
// Compare selected IDs only: unused output padding is not part of the API contract.
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace k = strata::kernels;
namespace {
void ck(cudaError_t e) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e));
        std::exit(2);
    }
}
struct Case { int64_t ctx, queries; int mode; };
bool check(Case c) {
    constexpr int64_t stride = 262144 / 4 + 2;
    const auto s = k::qsa_real_shapes();
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    std::vector<int32_t> steps((size_t)(c.queries * k::kStepCount));
    std::vector<float> scores((size_t)(c.queries * stride), -999.f);
    std::mt19937 rng(742913);
    std::uniform_real_distribution<float> random(0.f, 200.f);
    for (int64_t i = 0; i < c.queries; ++i) {
        auto* st = steps.data() + i * k::kStepCount;
        const int64_t pos = c.ctx - c.queries + i;
        st[k::kStepPos] = (int32_t)pos;
        st[k::kStepNKv] = (int32_t)(pos + 1);
        st[k::kStepNBid] = (int32_t)((pos + 1) / s.idx_block);
        st[k::kStepWidth] = (int32_t)k::qsa_selection_width(pos + 1, s);
        for (int64_t j = 0; j <= st[k::kStepNBid]; ++j)
            scores[(size_t)(i * stride + j)] = c.mode == 1 ? 0.f : c.mode == 2 ? float(j % 7) : random(rng);
        if ((pos + 1) % s.idx_block)
            scores[(size_t)(i * stride + st[k::kStepNBid])] += 1e9f;
    }
    int64_t active = steps[(size_t)((c.queries - 1) * k::kStepCount + k::kStepNBid)] + 1;
    if (c.mode == 3) active = 0;
    if (c.mode == 4) active = stride + 1;
    float* ds = nullptr;
    int32_t *dt = nullptr, *da = nullptr, *db = nullptr;
    ck(cudaMalloc(&ds, scores.size() * sizeof(float)));
    ck(cudaMalloc(&dt, steps.size() * sizeof(int32_t)));
    ck(cudaMalloc(&da, (size_t)(c.queries * cap) * sizeof(int32_t)));
    ck(cudaMalloc(&db, (size_t)(c.queries * cap) * sizeof(int32_t)));
    ck(cudaMemcpy(ds, scores.data(), scores.size() * sizeof(float), cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dt, steps.data(), steps.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    // No bound: the exact capacity-based dispatch used by captured decode.
    k::qsa_block_topk(ds, dt, c.queries, stride, cap, s, da, nullptr);
    k::qsa_block_topk(ds, dt, c.queries, stride, cap, s, db, nullptr, active);
    ck(cudaDeviceSynchronize());
    std::vector<int32_t> a((size_t)(c.queries * cap)), b(a.size());
    ck(cudaMemcpy(a.data(), da, a.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(b.data(), db, b.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int64_t i = 0; i < c.queries; ++i) {
        const int64_t w = steps[(size_t)(i * k::kStepCount + k::kStepWidth)];
        if (!std::equal(a.begin() + i * cap, a.begin() + i * cap + w, b.begin() + i * cap)) {
            std::fprintf(stderr, "FAIL ctx=%lld queries=%lld mode=%d row=%lld\n",
                         (long long)c.ctx, (long long)c.queries, c.mode, (long long)i);
            ok = false;
            break;
        }
    }
    ck(cudaFree(ds)); ck(cudaFree(dt)); ck(cudaFree(da)); ck(cudaFree(db));
    return ok;
}
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: no CUDA device");
        return 77;
    }
    // the active-bound dispatch is Turing's in the engine; forced here so every CUDA card checks it
#if defined(_WIN32)
    _putenv_s("STRATA_TOPK_ACTIVE_ANY", "1");
#else
    setenv("STRATA_TOPK_ACTIVE_ANY", "1", 1);
#endif
    const Case cases[] = {
        {1111, 256, 0}, {8192, 256, 0}, {32768, 256, 0}, {131072, 256, 0},
        {131075, 255, 0}, {135164, 256, 0}, {135168, 256, 0}, {250022, 256, 0},
        {262144, 256, 0}, {131072, 1, 0}, {131072, 8, 0}, {131072, 257, 0},
        {131072, 256, 1}, {131072, 256, 2}, {131072, 256, 3}, {131072, 256, 4},
        // either side of the Turing wide-kernel bound (22,528 blocks), and ties above the register limit
        {90108, 256, 0}, {90112, 256, 0}, {200000, 256, 2}, {262144, 256, 1},
    };
    for (const auto c : cases) if (!check(c)) return 1;
    std::puts("PASS: 20 top-k active-bound cases, selected IDs bitwise identical");
    return 0;
}
