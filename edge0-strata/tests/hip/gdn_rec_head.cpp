// tests/hip/gdn_rec_head.cpp - the prompt GDN recurrence's four-lanes-per-column kernel + norm (gdn_recurrence_variant 1)
// against the column-split kernels + the norm kernel (variant 0, the default before it):
// y, its FP16 bits and the final state must be BITWISE equal, over chunk lengths around the 8-token staging and a
// long one, from a nonzero state.  Synthetic inputs in the model's ranges (unit q/k rows, log-decay gates < 0).
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {
constexpr int S = 128, HK = 16, HV = 48, C = 10240;
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* dev(const std::vector<T>& h) {
    T* d = nullptr;
    ck(cudaMalloc(&d, h.size() * sizeof(T)), "malloc");
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
template <typename T> std::vector<T> host(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}
}  // namespace

int main() {
    // variants 0 and 1 are the references here; STRATA_GDN_CHUNKED=1 would turn them into the opt-in chunked recurrence
    // (tests/hip/gdn_chunked.cpp), whose bits differ
#ifdef _WIN32
    _putenv_s("STRATA_GDN_CHUNKED", "");
#else
    unsetenv("STRATA_GDN_CHUNKED");
#endif
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.f, 1.f);
    int failures = 0;
    const int64_t lens[] = {1, 7, 8, 9, 17, 100, 2048, 2051};
    for (const int64_t T : lens) {
        std::vector<float> h((size_t) T * C), gate((size_t) T * HV), beta((size_t) T * HV), z((size_t) T * HV * S),
            gamma(S), state((size_t) S * HV * S);
        for (int64_t t = 0; t < T; ++t) {
            float* ht = h.data() + t * C;
            for (int i = 0; i < C; ++i) ht[i] = nd(rng);
            for (int hh = 0; hh < 2 * HK; ++hh) {   // q and k heads: unit rows, as after gdn_conv's L2 norm
                double ss = 0;
                for (int d = 0; d < S; ++d) ss += (double) ht[hh * S + d] * ht[hh * S + d];
                for (int d = 0; d < S; ++d) ht[hh * S + d] = (float) (ht[hh * S + d] / std::sqrt(ss + 1e-6));
            }
        }
        for (auto& x : gate) x = -2.0f * ud(rng);
        for (auto& x : beta) x = ud(rng);
        for (auto& x : z) x = nd(rng);
        for (auto& x : gamma) x = 0.5f + ud(rng);
        for (auto& x : state) x = 0.1f * nd(rng);
        const float* dh = dev(h);
        const float* dg = dev(gate);
        const float* db = dev(beta);
        const float* dz = dev(z);
        const float* dgm = dev(gamma);
        std::vector<float> y_ref, st_ref;
        std::vector<uint16_t> y16_ref;
        for (int variant : {0, 3, 1, 4, 5}) {
            float* dst = dev(state);
            float* dy = nullptr;
            uint16_t* dy16 = nullptr;
            ck(cudaMalloc(&dy, (size_t) T * HV * S * 4), "malloc");
            ck(cudaMalloc(&dy16, (size_t) T * HV * S * 2), "malloc");
            strata::prefill::gdn_recurrence_variant(variant, dst, dh, dg, db, dz, dgm, 1e-6f, dy, dy16, T, nullptr);
            ck(cudaDeviceSynchronize(), "run");
            auto y = host(dy, (size_t) T * HV * S);
            auto y16 = host(dy16, (size_t) T * HV * S);
            auto st = host(dst, state.size());
            if (variant == 0) {
                y_ref = y; y16_ref = y16; st_ref = st;
            } else {
                size_t dy_n = 0, d16 = 0, dsn = 0;
                for (size_t i = 0; i < y.size(); ++i) dy_n += std::memcmp(&y[i], &y_ref[i], 4) != 0;
                for (size_t i = 0; i < y16.size(); ++i) d16 += y16[i] != y16_ref[i];
                for (size_t i = 0; i < st.size(); ++i) dsn += std::memcmp(&st[i], &st_ref[i], 4) != 0;
                // (y, the FP32 scratch, is not part of the contract: 0.1.39's norm kernel no longer stores the normalized value there,
                // while the head kernels store it unless STRATA_GDN_NOY; what the out projection reads is y16, and the state)
                const bool ok = d16 == 0 && dsn == 0;
                failures += !ok;
                std::printf("%s T %lld variant %d vs 0: y %zu, y16 %zu, state %zu values differ\n", ok ? "PASS" : "FAIL",
                            (long long) T, variant, dy_n, d16, dsn);
            }
            cudaFree(dst); cudaFree(dy); cudaFree(dy16);
        }
        // timing at the long lengths: variant 0 vs 1
        if (T >= 2048) {
            float* dst = dev(state);
            float* dy = nullptr;
            uint16_t* dy16 = nullptr;
            ck(cudaMalloc(&dy, (size_t) T * HV * S * 4), "malloc");
            ck(cudaMalloc(&dy16, (size_t) T * HV * S * 2), "malloc");
            for (int variant : {0, 1}) {
                cudaEvent_t a, b;
                cudaEventCreate(&a); cudaEventCreate(&b);
                strata::prefill::gdn_recurrence_variant(variant, dst, dh, dg, db, dz, dgm, 1e-6f, dy, dy16, T, nullptr);
                cudaEventRecord(a);
                for (int r = 0; r < 3; ++r)
                    strata::prefill::gdn_recurrence_variant(variant, dst, dh, dg, db, dz, dgm, 1e-6f, dy, dy16, T, nullptr);
                cudaEventRecord(b);
                ck(cudaEventSynchronize(b), "time");
                float ms = 0;
                cudaEventElapsedTime(&ms, a, b);
                std::printf("  T %lld variant %d: %.3f ms per call (%.2f us per token)\n", (long long) T, variant, ms / 3,
                            ms / 3 * 1000 / (double) T);
            }
            cudaFree(dst); cudaFree(dy); cudaFree(dy16);
        }
        cudaFree((void*) dh); cudaFree((void*) dg); cudaFree((void*) db); cudaFree((void*) dz); cudaFree((void*) dgm);
    }
    std::printf("FAILURES: %d\n", failures);
    return failures ? 1 : 0;
}
