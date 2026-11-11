// The prompt path's block scores on the thread-per-head kernel (an active-block count) against block_scores_kernel
// (no count, more than 8 queries), every score of blocks 0..n_bid bitwise; then the time of each.  GPU, synthetic.
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
// ctx: the last query's position + 1; queries: a prompt chunk ending there; mode 0 normal values, 1 wide magnitudes
// and exact zeros, 2 small integers (many ties and cancellations)
struct Case { int64_t ctx, queries; int mode; bool time; };
bool check(Case c) {
    const auto s = k::qsa_real_shapes();
    const int64_t D = s.idx_dim, H = s.idx_n_head, max_blocks = c.ctx / s.idx_block + 2;
    std::mt19937 rng(91731 + (unsigned) c.ctx + 7 * (unsigned) c.queries + 101 * (unsigned) c.mode);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> small(-3, 3), expo(-20, 20), coin(0, 9);
    auto value = [&] {
        if (c.mode == 1) return coin(rng) == 0 ? 0.f : std::ldexp(nd(rng), expo(rng));
        if (c.mode == 2) return (float) small(rng);
        return nd(rng);
    };
    std::vector<float> pooled((size_t) (max_blocks * D)), dead((size_t) D), q((size_t) (c.queries * H * D));
    for (auto& v : pooled) v = value();
    for (auto& v : dead) v = value();
    for (auto& v : q) v = value();
    std::vector<int32_t> steps((size_t) (c.queries * k::kStepCount));
    for (int64_t i = 0; i < c.queries; ++i) {
        int32_t* st = steps.data() + i * k::kStepCount;
        const int64_t pos = c.ctx - c.queries + i;
        st[k::kStepPos] = (int32_t) pos;
        st[k::kStepNKv] = (int32_t) (pos + 1);
        st[k::kStepNBid] = (int32_t) ((pos + 1) / s.idx_block);
        st[k::kStepWidth] = (int32_t) k::qsa_selection_width(pos + 1, s);
    }
    const int64_t active = steps[(size_t) ((c.queries - 1) * k::kStepCount + k::kStepNBid)] + 1;
    float *dp = nullptr, *dd = nullptr, *dq = nullptr, *da = nullptr, *db = nullptr;
    int32_t* dt = nullptr;
    const size_t n_out = (size_t) (c.queries * max_blocks);
    ck(cudaMalloc(&dp, pooled.size() * 4)); ck(cudaMalloc(&dd, dead.size() * 4)); ck(cudaMalloc(&dq, q.size() * 4));
    ck(cudaMalloc(&dt, steps.size() * 4)); ck(cudaMalloc(&da, n_out * 4)); ck(cudaMalloc(&db, n_out * 4));
    ck(cudaMemcpy(dp, pooled.data(), pooled.size() * 4, cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dd, dead.data(), dead.size() * 4, cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dt, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice));
    ck(cudaMemset(da, 0x7f, n_out * 4));   // different fill in each: an unwritten score cannot match by chance
    ck(cudaMemset(db, 0x3e, n_out * 4));
    k::qsa_block_scores(dp, dd, dq, dt, c.queries, max_blocks, s, da, nullptr);           // block_scores_kernel
    k::qsa_block_scores(dp, dd, dq, dt, c.queries, max_blocks, s, db, nullptr, active);   // the thread kernel
    ck(cudaDeviceSynchronize());
    std::vector<uint32_t> a(n_out), b(n_out);
    ck(cudaMemcpy(a.data(), da, n_out * 4, cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(b.data(), db, n_out * 4, cudaMemcpyDeviceToHost));
    bool ok = true;
    int64_t checked = 0;
    for (int64_t i = 0; i < c.queries && ok; ++i) {
        const int64_t n_bid = steps[(size_t) (i * k::kStepCount + k::kStepNBid)];
        for (int64_t j = 0; j <= n_bid; ++j, ++checked) {
            const size_t at = (size_t) (i * max_blocks + j);
            if (a[at] != b[at]) {
                float fa, fb;
                std::memcpy(&fa, &a[at], 4); std::memcpy(&fb, &b[at], 4);
                std::fprintf(stderr, "FAIL ctx=%lld queries=%lld mode=%d query=%lld block=%lld: %.9g vs %.9g\n",
                             (long long) c.ctx, (long long) c.queries, c.mode, (long long) i, (long long) j, fa, fb);
                ok = false;
                break;
            }
        }
    }
    if (ok) std::printf("ok   ctx=%7lld queries=%5lld mode=%d: %lld scores identical\n", (long long) c.ctx,
                        (long long) c.queries, c.mode, (long long) checked);
    if (ok && c.time) {
        cudaEvent_t e0, e1, e2;
        ck(cudaEventCreate(&e0)); ck(cudaEventCreate(&e1)); ck(cudaEventCreate(&e2));
        constexpr int N = 5;
        ck(cudaEventRecord(e0));
        for (int r = 0; r < N; ++r) k::qsa_block_scores(dp, dd, dq, dt, c.queries, max_blocks, s, da, nullptr);
        ck(cudaEventRecord(e1));
        for (int r = 0; r < N; ++r) k::qsa_block_scores(dp, dd, dq, dt, c.queries, max_blocks, s, db, nullptr, active);
        ck(cudaEventRecord(e2));
        ck(cudaEventSynchronize(e2));
        float t_old = 0.f, t_new = 0.f;
        ck(cudaEventElapsedTime(&t_old, e0, e1)); ck(cudaEventElapsedTime(&t_new, e1, e2));
        std::printf("     time per call: warp kernel %.2f ms, thread kernel %.2f ms (%.2fx)\n", t_old / N, t_new / N,
                    t_old / t_new);
        cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2);
    }
    ck(cudaFree(dp)); ck(cudaFree(dd)); ck(cudaFree(dq)); ck(cudaFree(dt)); ck(cudaFree(da)); ck(cudaFree(db));
    return ok;
}
}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: no CUDA device");
        return 77;
    }
    const Case cases[] = {
        {1111, 9, 0, false},      {1111, 64, 1, false},     {8192, 65, 0, false},     {8195, 200, 2, false},
        {32768, 256, 0, false},   {32771, 1000, 1, false},  {131072, 64, 2, false},   {131075, 333, 0, false},
        {262144, 129, 1, false},  {40000, 2048, 0, true},   {140000, 1024, 0, true},  {200000, 1024, 0, true},
    };
    for (const auto c : cases) if (!check(c)) return 1;
    std::puts("PASS: block scores of the thread kernel bitwise identical to block_scores_kernel");
    return 0;
}
