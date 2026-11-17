// src/kernels/native_multi_parity.cpp - the multi-token router and the k = 10 combine, bitwise against their
// single-token contracts (#783 PR-c: route_multi, combine_k10_vec4).
//
//   * native_router_top10_multi (one warp per token, 8 tokens per CTA) against native_router_top10 token by token,
//     for n = 1..19 (a partial last CTA, and more than one CTA), on plain logits, on logits with exact ties and on
//     logits with NaN (the router maps NaN to -FLT_MAX): ids and weights are compared with memcmp;
//   * native_moe_combine / native_moe_combine_multi (k = 10 with a shared row takes the float4 kernel; a misaligned
//     output, k = 9 and a null shared row take the scalar one) against a host replay of the documented contract:
//     the first product rounds to F32, the next ones accumulate with FMA in expert order, the shared row is added
//     once: memcmp.
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/router_top10.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

template <typename T>
T* up(const std::vector<T>& h) {
    void* p = nullptr;
    check(cudaMalloc(&p, h.size() * sizeof(T) + 64), "malloc");
    check(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "h2d");
    return (T*) p;
}
template <typename T>
std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    check(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "d2h");
    return h;
}

}  // namespace

int main(int argc, char**) {
    (void) argc;
    using namespace strata::kernels;
    std::mt19937 rng(783);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    int bad = 0;
    cudaStream_t st = nullptr;
    check(cudaStreamCreate(&st), "stream");

    // ---- the multi-token router
    for (int mode = 0; mode < 3; ++mode) {
        const char* names[] = {"random logits", "exact ties", "NaN logits"};
        int mode_bad = 0, cases = 0;
        for (int n = 1; n <= 19; ++n) {
            std::vector<float> lg((size_t) n * 512);
            for (auto& v : lg) v = gauss(rng) * 2.0f;
            if (mode == 1)
                for (size_t i = 0; i < lg.size(); ++i) lg[i] = (float) (int) (lg[i] * 2.0f) * 0.5f;   // coarse: many exact ties
            if (mode == 2)
                for (int t = 0; t < n; ++t) lg[(size_t) t * 512 + (size_t) (rng() % 512)] = std::nanf("");
            float* d_l = up(lg);
            int32_t *d_i, *d_ir;
            float *d_w, *d_wr;
            check(cudaMalloc(&d_i, (size_t) n * 10 * 4), "i");
            check(cudaMalloc(&d_ir, (size_t) n * 10 * 4), "ir");
            check(cudaMalloc(&d_w, (size_t) n * 10 * 4), "w");
            check(cudaMalloc(&d_wr, (size_t) n * 10 * 4), "wr");
            native_router_top10_multi(d_l, d_i, d_w, n, st);
            for (int t = 0; t < n; ++t) native_router_top10(d_l + (size_t) t * 512, d_ir + t * 10, d_wr + t * 10, st);
            check(cudaDeviceSynchronize(), "sync");
            const auto i1 = down(d_i, (size_t) n * 10), i2 = down(d_ir, (size_t) n * 10);
            const auto w1 = down(d_w, (size_t) n * 10), w2 = down(d_wr, (size_t) n * 10);
            ++cases;
            if (std::memcmp(i1.data(), i2.data(), i1.size() * 4) != 0 || std::memcmp(w1.data(), w2.data(), w1.size() * 4) != 0) {
                std::printf("    *** native_router_top10_multi n=%d (%s) differs from native_router_top10 ***\n", n, names[mode]);
                ++mode_bad;
            }
            cudaFree(d_l); cudaFree(d_i); cudaFree(d_ir); cudaFree(d_w); cudaFree(d_wr);
        }
        std::printf("  %-52s %s (%d cases)\n", (std::string("router multi == single, ") + names[mode]).c_str(),
                    mode_bad ? "*** NO ***" : "bitwise", cases);
        bad += mode_bad;
    }

    // ---- #1357: a model that does not have 512 experts takes the generic router (the native one reads 512 floats a
    // row, so mtp.cpp's per-token call is guarded by n_expert == 512 like its siblings): router_top10 on 256-wide
    // rows picks the host reference's ids in order, with its renormalised softmax weights
    {
        const int n = 3, NE = 256, K = 10;
        std::vector<float> lg((size_t) n * NE);
        for (auto& v : lg) v = gauss(rng) * 2.0f;
        float* d_l = up(lg);
        int32_t* d_i = nullptr;
        float* d_w = nullptr;
        check(cudaMalloc(&d_i, (size_t) n * K * 4), "i");
        check(cudaMalloc(&d_w, (size_t) n * K * 4), "w");
        router_top10(d_l, n, NE, K, d_i, d_w, st);
        check(cudaDeviceSynchronize(), "sync");
        const auto gi = down(d_i, (size_t) n * K);
        const auto gw = down(d_w, (size_t) n * K);
        int rbad = 0;
        for (int t = 0; t < n; ++t) {
            std::vector<int> ord(NE);
            for (int i = 0; i < NE; ++i) ord[i] = i;
            const float* row = lg.data() + (size_t) t * NE;
            std::sort(ord.begin(), ord.end(), [&](int a, int b) { return row[a] != row[b] ? row[a] > row[b] : a < b; });
            double sum = 0;
            for (int j = 0; j < K; ++j) sum += std::exp((double) row[ord[j]] - row[ord[0]]);
            for (int j = 0; j < K; ++j) {
                const double want = std::exp((double) row[ord[j]] - row[ord[0]]) / sum;
                if (gi[(size_t) t * K + j] != ord[j] || std::fabs(gw[(size_t) t * K + j] - want) > 1e-5) ++rbad;
            }
        }
        std::printf("  %-52s %s\n", "router_top10 on 256 experts (non-512 fallback)", rbad ? "*** NO ***" : "matches the host reference");
        bad += rbad;
        cudaFree(d_l); cudaFree(d_i); cudaFree(d_w);
    }

    // ---- the combine
    struct Cfg { const char* name; int k; bool shared; int out_off; };
    const Cfg cfgs[] = {{"k=10 + shared (float4 kernel)", 10, true, 0}, {"k=10 + shared, output misaligned", 10, true, 1},
                        {"k=10 without shared", 10, false, 0}, {"k=9 + shared", 9, true, 0}};
    const int N = 2048;
    for (const Cfg& c : cfgs) {
        int cfg_bad = 0, cases = 0;
        for (int n = 1; n <= 8; ++n) {
            std::vector<float> parts((size_t) n * c.k * N), w((size_t) n * c.k), sh((size_t) n * N);
            for (auto& v : parts) v = gauss(rng);
            for (auto& v : w) v = (float) (rng() % 1000) / 1000.0f;
            for (auto& v : sh) v = gauss(rng);
            float* d_p = up(parts);
            float* d_w = up(w);
            float* d_s = up(sh);
            float* d_o = nullptr;
            check(cudaMalloc(&d_o, (size_t) n * N * 4 + 64), "o");
            float* out = d_o + c.out_off;
            if (c.out_off) {
                // a misaligned output cannot take a multi launch of rows with the 4-float stride contract either way:
                // the rows are still n*N floats from `out`
            }
            native_moe_combine_multi(d_p, d_w, c.shared ? d_s : nullptr, out, N, c.k, n, st);
            check(cudaDeviceSynchronize(), "sync");
            const auto got = down(out, (size_t) n * N);
            std::vector<float> want((size_t) n * N);
            for (int t = 0; t < n; ++t)
                for (int col = 0; col < N; ++col) {
                    float s = parts[((size_t) t * c.k) * N + col] * w[(size_t) t * c.k];
                    for (int e = 1; e < c.k; ++e)
                        s = std::fmaf(parts[((size_t) t * c.k + e) * N + col], w[(size_t) t * c.k + e], s);
                    if (c.shared) s += sh[(size_t) t * N + col];
                    want[(size_t) t * N + col] = s;
                }
            ++cases;
            if (std::memcmp(got.data(), want.data(), got.size() * 4) != 0) {
                std::printf("    *** native_moe_combine_multi n=%d (%s) differs from the contract's host replay ***\n", n, c.name);
                ++cfg_bad;
            }
            // the single-token call is the same arithmetic
            if (n == 1) {
                check(cudaMemset(d_o, 0, (size_t) N * 4 + 64), "z");
                native_moe_combine(d_p, d_w, c.shared ? d_s : nullptr, out, N, c.k, st);
                check(cudaDeviceSynchronize(), "sync");
                const auto g1 = down(out, (size_t) N);
                ++cases;
                if (std::memcmp(g1.data(), want.data(), g1.size() * 4) != 0) {
                    std::printf("    *** native_moe_combine (%s) differs from the contract's host replay ***\n", c.name);
                    ++cfg_bad;
                }
            }
            cudaFree(d_p); cudaFree(d_w); cudaFree(d_s); cudaFree(d_o);
        }
        std::printf("  %-52s %s (%d cases)\n", (std::string("combine ") + c.name).c_str(), cfg_bad ? "*** NO ***" : "bitwise", cases);
        bad += cfg_bad;
    }

    std::printf("\nnative_multi: %d failures\n", bad);
    if (bad) return 1;
    std::printf("native_multi_parity OK\n");
    return 0;
}
