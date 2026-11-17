// prefill_mmq_scatter_test - the prompt path's MoE input quantized per token and scattered to its k rows
// (mmq::quantize_scatter) against the per-row gather (mmq::quantize over the expert-order source rows): the same
// q8_1 bytes, the whole buffer, for every q8_1 layout MMQ uses (D4: Q2_0 Q8_0 IQ2_XS, DS4: Q4_1, D2S6: Q2_K), on a
// routing built the way Prefill builds it (10 of 512 experts a token, rows grouped by expert), with an all-zero
// token and token counts that are not multiples of anything.  --bench: both at 2K, 8K and 32K tokens.
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace mmq = strata::prefill::mmq;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n) { ck(cudaMalloc(&p, n), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
};

constexpr int kExperts = 512, kUsed = 10;
constexpr int64_t kCols = 2560;

// Prefill's maps: slot[t * k + j] = the token's row in expert order, src[row] = the token
void routing(int64_t T, std::mt19937& rng, std::vector<int32_t>& slot, std::vector<int32_t>& src) {
    std::vector<int32_t> ids((size_t) (T * kUsed));
    std::vector<int32_t> pick(kExperts);
    for (int e = 0; e < kExperts; ++e) pick[(size_t) e] = e;
    for (int64_t t = 0; t < T; ++t) {
        for (int j = 0; j < kUsed; ++j) std::swap(pick[(size_t) j], pick[(size_t) (j + rng() % (kExperts - j))]);
        for (int j = 0; j < kUsed; ++j) ids[(size_t) (t * kUsed + j)] = pick[(size_t) j];
    }
    std::vector<int32_t> cnt(kExperts, 0), fill(kExperts, 0);
    for (int32_t e : ids) ++cnt[(size_t) e];
    for (int e = 1; e < kExperts; ++e) fill[(size_t) e] = fill[(size_t) e - 1] + cnt[(size_t) e - 1];
    slot.assign(ids.size(), 0);
    src.assign(ids.size(), 0);
    for (size_t i = 0; i < ids.size(); ++i) {
        const int32_t p = fill[(size_t) ids[i]]++;
        slot[i] = p;
        src[(size_t) p] = (int32_t) (i / kUsed);
    }
}

int check(ggml_type t, int64_t T, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<int32_t> slot, src;
    routing(T, rng, slot, src);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t) (T * kCols));
    for (float& v : x) v = nd(rng) * 0.7f;
    if (T > 3) std::fill(x.begin() + 2 * kCols, x.begin() + 3 * kCols, 0.0f);   // an all-zero token
    const int64_t rows = T * kUsed;
    const size_t qb = mmq::q8_bytes(rows, kCols);
    Dev dx(x.size() * 4), dslot(slot.size() * 4), dsrc(src.size() * 4), qa(qb), qs(qb);
    ck(cudaMemcpy(dx.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(dslot.p, slot.data(), slot.size() * 4, cudaMemcpyHostToDevice), "slot");
    ck(cudaMemcpy(dsrc.p, src.data(), src.size() * 4, cudaMemcpyHostToDevice), "src");
    ck(cudaMemset(qa.p, 0xA5, qb), "memset");
    ck(cudaMemset(qs.p, 0xA5, qb), "memset");
    mmq::quantize((const float*) dx.p, (const int32_t*) dsrc.p, qa.p, (int) t, kCols, kCols, rows, nullptr);
    mmq::quantize_scatter((const float*) dx.p, (const int32_t*) dslot.p, (const int32_t*) dsrc.p, qs.p, (int) t, kCols,
                          kCols, T, kUsed, nullptr);
    ck(cudaDeviceSynchronize(), "quantize");
    std::vector<uint8_t> a(qb), b(qb);
    ck(cudaMemcpy(a.data(), qa.p, qb, cudaMemcpyDeviceToHost), "back");
    ck(cudaMemcpy(b.data(), qs.p, qb, cudaMemcpyDeviceToHost), "back");
    size_t diff = 0;
    for (size_t i = 0; i < qb; ++i) diff += a[i] != b[i];
    std::printf("%s %-7s T=%-5lld %zu of %zu bytes differ\n", diff ? "FAIL" : "ok  ", ggml_type_name(t), (long long) T,
                diff, qb);
    return diff ? 1 : 0;
}

void bench(ggml_type t, int64_t T) {
    std::mt19937 rng(7);
    std::vector<int32_t> slot, src;
    routing(T, rng, slot, src);
    std::vector<float> x((size_t) (T * kCols), 0.25f);
    const int64_t rows = T * kUsed;
    const size_t qb = mmq::q8_bytes(rows, kCols);
    Dev dx(x.size() * 4), dslot(slot.size() * 4), dsrc(src.size() * 4), q(qb);
    ck(cudaMemcpy(dx.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(dslot.p, slot.data(), slot.size() * 4, cudaMemcpyHostToDevice), "slot");
    ck(cudaMemcpy(dsrc.p, src.data(), src.size() * 4, cudaMemcpyHostToDevice), "src");
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "event");
    ck(cudaEventCreate(&e1), "event");
    std::vector<float> tg, ts;
    for (int i = 0; i < 23; ++i)
        for (int v = 0; v < 2; ++v) {
            cudaEventRecord(e0);
            if (v == 0) mmq::quantize((const float*) dx.p, (const int32_t*) dsrc.p, q.p, (int) t, kCols, kCols, rows, nullptr);
            else mmq::quantize_scatter((const float*) dx.p, (const int32_t*) dslot.p, (const int32_t*) dsrc.p, q.p, (int) t,
                                       kCols, kCols, T, kUsed, nullptr);
            cudaEventRecord(e1);
            ck(cudaEventSynchronize(e1), "bench");
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, e0, e1);
            if (i >= 3) (v == 0 ? tg : ts).push_back(ms);
        }
    std::sort(tg.begin(), tg.end());
    std::sort(ts.begin(), ts.end());
    const float g = tg[tg.size() / 2], s = ts[ts.size() / 2];
    std::printf("  %-7s T=%lld x %d rows: gather %.3f ms, scatter %.3f ms (%.2fx)\n", ggml_type_name(t), (long long) T,
                kUsed, g, s, g / s);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
}
}  // namespace

int main(int argc, char** argv) {
    const bool do_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    int fails = 0;
    try {
        const ggml_type types[] = {GGML_TYPE_Q2_0, GGML_TYPE_Q8_0, GGML_TYPE_IQ2_XS, GGML_TYPE_Q4_1, GGML_TYPE_Q2_K};
        uint32_t seed = 11;
        for (ggml_type t : types)
            for (int64_t T : {1, 2, 7, 600, 2099}) fails += check(t, T, seed++);
        if (do_bench) {
            std::printf("--bench (median of 20):\n");
            for (ggml_type t : {GGML_TYPE_IQ2_XS, GGML_TYPE_Q2_0}) {
                bench(t, 2048);
                bench(t, 8192);
                bench(t, 32768);
            }
        }
    } catch (const std::exception& e) {
        std::printf("FAIL %s\n", e.what());
        return 1;
    }
    std::printf("prefill_mmq_scatter_test: %d failures\n", fails);
    return fails ? 1 : 0;
}
