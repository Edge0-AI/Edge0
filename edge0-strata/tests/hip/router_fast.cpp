// S6: the AMD fast router (router_top10's default on HIP for 64 < n_expert <= 512, k <= 32) against the portable
// kernel, BITWISE: ids and weights of 65,536 rows of 512 logits - normal rows, rows with many exact ties, near-ties,
// a wide spread (tiny probabilities), all-equal rows and rows with a NaN - plus the fast kernel forced onto its
// serial-sum path, and a 256-expert / k=8 geometry. Then the time per single-token call of each.
#include <hip/hip_runtime.h>
#include "strata/kernels/router_top10.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(call)                                                                                                  \
    do {                                                                                                             \
        const hipError_t error = (call);                                                                             \
        if (error != hipSuccess) {                                                                                   \
            std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(error));                                     \
            return 2;                                                                                                \
        }                                                                                                            \
    } while (0)

namespace k = strata::kernels;

static int compare(int n_expert, int topk, int rows, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 2.f);
    std::vector<float> lg((size_t) rows * n_expert);
    for (size_t i = 0; i < lg.size(); ++i) {
        const size_t r = i / n_expert;
        float x = nd(rng) * (r % 7 == 5 ? 40.0f : 1.0f);
        if (r % 3 == 1) x = std::round(x * 4.0f) / 4.0f;
        if (r % 3 == 2) x = std::round(x * 64.0f) / 64.0f;
        if (r % 1000 == 999 && (i % n_expert) == 77) x = NAN;
        if (r % 1000 == 998) x = 1.0f;
        lg[i] = x;
    }
    const size_t out = (size_t) rows * topk;
    float *d_l = nullptr, *d_w[3] = {};
    int* d_i[3] = {};
    CHECK(hipMalloc(&d_l, lg.size() * 4));
    CHECK(hipMemcpy(d_l, lg.data(), lg.size() * 4, hipMemcpyHostToDevice));
    std::vector<int> ids[3];
    std::vector<float> w[3];
    for (int v = 0; v < 3; ++v) {
        CHECK(hipMalloc(&d_w[v], out * 4));
        CHECK(hipMalloc(&d_i[v], out * 4));
        CHECK(hipMemset(d_w[v], 0x3c, out * 4));   // what an unwritten rank keeps (NaN rows)
        CHECK(hipMemset(d_i[v], 0x11, out * 4));
        if (!k::router_top10_variant(d_l, rows, n_expert, topk, d_i[v], d_w[v], nullptr, v)) {
            std::fprintf(stderr, "router_top10_variant %d refused\n", v);
            return 2;
        }
        CHECK(hipDeviceSynchronize());
        ids[v].resize(out);
        w[v].resize(out);
        CHECK(hipMemcpy(ids[v].data(), d_i[v], out * 4, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(w[v].data(), d_w[v], out * 4, hipMemcpyDeviceToHost));
    }
    int fails = 0;
    for (int v = 1; v < 3; ++v) {
        int bad = 0;
        for (int r = 0; r < rows; ++r)
            if (std::memcmp(&ids[0][(size_t) r * topk], &ids[v][(size_t) r * topk], topk * 4) ||
                std::memcmp(&w[0][(size_t) r * topk], &w[v][(size_t) r * topk], topk * 4)) {
                if (bad < 3)
                    std::printf("  row %d differs: id0 %d/%d w0 %.9g/%.9g\n", r, ids[0][(size_t) r * topk],
                                ids[v][(size_t) r * topk], w[0][(size_t) r * topk], w[v][(size_t) r * topk]);
                ++bad;
            }
        std::printf("%s %d experts, k %d, %d rows: %s vs the portable kernel: %d rows differ\n", bad ? "FAIL" : "PASS",
                    n_expert, topk, rows, v == 1 ? "fast kernel" : "fast kernel, serial path", bad);
        fails += bad != 0;
    }
    // time: single-token calls back to back
    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0));
    CHECK(hipEventCreate(&e1));
    for (int v = 0; v < 2; ++v) {
        const int reps = 2000;
        CHECK(hipEventRecord(e0));
        for (int i = 0; i < reps; ++i)
            k::router_top10_variant(d_l + (size_t) (i % 64) * n_expert, 1, n_expert, topk, d_i[v], d_w[v], nullptr, v);
        CHECK(hipEventRecord(e1));
        CHECK(hipEventSynchronize(e1));
        float ms = 0;
        CHECK(hipEventElapsedTime(&ms, e0, e1));
        std::printf("  %s: %.2f us per single-token call\n", v ? "fast" : "portable", 1000.0f * ms / reps);
    }
    for (int v = 0; v < 3; ++v) { (void) hipFree(d_w[v]); (void) hipFree(d_i[v]); }
    (void) hipFree(d_l);
    return fails;
}

int main() {
    int fails = 0;
    fails += compare(512, 10, 65536, 17);
    fails += compare(256, 8, 8192, 5);
    std::printf("FAILURES: %d\n", fails);
    return fails ? 1 : 0;
}
