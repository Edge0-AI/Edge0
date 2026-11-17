// bench/gdn_chunk_bench.cpp - v100/gdn-chunk: the chunked GDN prompt recurrence against the token recurrence,
// real GDN geometry (S=128, 16 q/k heads, 48 v heads, C=10240), CUDA events.
//
// One call of `gdn_recurrence` per side (the whole prompt chunk for one GDN layer: conv output -> out norm),
// A/B'd with STRATA_GDN_CHUNK=0 / =1 so the dispatch, the tail and the norm are timed exactly as the engine
// runs them.  The two paths are not bit-exact (the chunked one reorders the sums); the bench reports the max
// relative difference it saw so a silent divergence cannot hide behind a speedup.
//
//   usage: gdn_chunk_bench [T ...]        (default 1024 4096 8192)
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int S = 128, HV = 48, C = 10240;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

struct Rng {
    uint64_t s = 12345;
    float uni() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return (float) ((s >> 40) * (1.0 / 16777216.0));
    }
};

float timed(const char* mode, const char* warp, float* st, float* st0, float* h, float* g, float* b, float* z,
            float* ga, float* y, uint16_t* y16, int64_t T, int iters) {
    setenv("STRATA_GDN_CHUNK", mode, 1);
    setenv("STRATA_GDN_REC_WARP", warp, 1);
    cudaEvent_t e0, e1;
    check(cudaEventCreate(&e0), "event");
    check(cudaEventCreate(&e1), "event");
    check(cudaMemcpy(st, st0, (size_t) S * HV * S * 4, cudaMemcpyDeviceToDevice), "reset");
    strata::prefill::gdn_recurrence(st, h, g, b, z, ga, 1e-5f, y, y16, T, nullptr);
    check(cudaDeviceSynchronize(), "warmup");
    check(cudaEventRecord(e0), "record");
    for (int i = 0; i < iters; ++i) {
        check(cudaMemcpy(st, st0, (size_t) S * HV * S * 4, cudaMemcpyDeviceToDevice), "reset");
        strata::prefill::gdn_recurrence(st, h, g, b, z, ga, 1e-5f, y, y16, T, nullptr);
    }
    check(cudaEventRecord(e1), "record");
    check(cudaEventSynchronize(e1), "sync");
    float ms = 0;
    check(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ms / iters;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<int64_t> ts;
    for (int i = 1; i < argc; ++i) ts.push_back(std::atoll(argv[i]));
    if (ts.empty()) ts = {1024, 4096, 8192};
    int64_t tmax = 0;
    for (int64_t t : ts) tmax = t > tmax ? t : tmax;

    int dev = 0;
    cudaDeviceProp p{};
    check(cudaGetDevice(&dev), "device");
    check(cudaGetDeviceProperties(&p, dev), "props");
    const int cc = p.major * 10 + p.minor;
    std::printf("%s (cc %d)  GDN prompt recurrence: chunked vs token-by-token\n", p.name, cc);

    Rng rng;
    const size_t n_h = (size_t) tmax * C, n_y = (size_t) tmax * HV * S, n_g = (size_t) tmax * HV;
    float *d_h, *d_y, *d_z, *d_g, *d_b, *d_ga, *d_st, *d_st0;
    uint16_t* d_y16;
    check(cudaMalloc(&d_h, n_h * 4), "malloc");
    check(cudaMalloc(&d_y, n_y * 4), "malloc");
    check(cudaMalloc(&d_z, n_y * 4), "malloc");
    check(cudaMalloc(&d_g, n_g * 4), "malloc");
    check(cudaMalloc(&d_b, n_g * 4), "malloc");
    check(cudaMalloc(&d_ga, S * 4), "malloc");
    check(cudaMalloc(&d_st, (size_t) S * HV * S * 4), "malloc");
    check(cudaMalloc(&d_st0, (size_t) S * HV * S * 4), "malloc");
    check(cudaMalloc(&d_y16, n_y * 2), "malloc");

    std::vector<float> tmp((size_t) tmax * C);
    for (size_t i = 0; i < tmp.size(); ++i) tmp[i] = rng.uni() - 0.5f;
    // q/k rows are L2-normalised rows (as gdn_conv leaves them); v rows stay O(1).  A degenerate near-zero
    // v would make the out-norm amplify rounding noise by 1/sqrt(eps) and the diff column meaningless.
    for (int64_t t = 0; t < tmax; ++t)
        for (int row = 0; row < 2 * 16; ++row) {
            float* r = tmp.data() + (size_t) t * C + row * S;
            double ss = 0;
            for (int i = 0; i < S; ++i) ss += (double) r[i] * r[i];
            const float nrm = (float) (1.0 / std::sqrt(ss + 1e-5));
            for (int i = 0; i < S; ++i) r[i] *= nrm;
        }
    check(cudaMemcpy(d_h, tmp.data(), n_h * 4, cudaMemcpyHostToDevice), "H2D");
    tmp.resize(n_g);
    for (size_t i = 0; i < n_g; ++i) tmp[i] = -0.02f - 0.3f * rng.uni();
    check(cudaMemcpy(d_g, tmp.data(), n_g * 4, cudaMemcpyHostToDevice), "H2D");
    for (size_t i = 0; i < n_g; ++i) tmp[i] = 1.0f / (1.0f + std::exp(-(4.0f * rng.uni() - 2.0f)));
    check(cudaMemcpy(d_b, tmp.data(), n_g * 4, cudaMemcpyHostToDevice), "H2D");
    tmp.resize(n_y);
    for (size_t i = 0; i < n_y; ++i) tmp[i] = 2.0f * rng.uni() - 1.0f;
    check(cudaMemcpy(d_z, tmp.data(), n_y * 4, cudaMemcpyHostToDevice), "H2D");
    tmp.resize(S);
    for (int i = 0; i < S; ++i) tmp[i] = 0.5f + 2.0f * rng.uni();
    check(cudaMemcpy(d_ga, tmp.data(), S * 4, cudaMemcpyHostToDevice), "H2D");
    tmp.resize((size_t) S * HV * S);
    for (size_t i = 0; i < tmp.size(); ++i) tmp[i] = 0.1f * (rng.uni() - 0.5f);
    check(cudaMemcpy(d_st0, tmp.data(), tmp.size() * 4, cudaMemcpyHostToDevice), "H2D");

    std::printf("%8s %12s %12s %12s %12s %12s   %s\n", "T", "pipe(rec)", "warp(rec)", "fast(rec)", "chunk FMA",
                "chunk wmma", "max abs diff vs pipe");
    for (int64_t T : ts) {
        const int iters = T >= 4096 ? 5 : 20;
        const float rec = timed("0", "0", d_st, d_st0, d_h, d_g, d_b, d_z, d_ga, d_y, d_y16, T, iters);
        std::vector<float> y0((size_t) T * HV * S);
        check(cudaMemcpy(y0.data(), d_y, y0.size() * 4, cudaMemcpyDeviceToHost), "D2H");
        const float wrp = timed("0", "1", d_st, d_st0, d_h, d_g, d_b, d_z, d_ga, d_y, d_y16, T, iters);
        const float fst = timed("0", "2", d_st, d_st0, d_h, d_g, d_b, d_z, d_ga, d_y, d_y16, T, iters);
        const float fma = timed("2", "0", d_st, d_st0, d_h, d_g, d_b, d_z, d_ga, d_y, d_y16, T, iters);
        const float ch = timed("1", "0", d_st, d_st0, d_h, d_g, d_b, d_z, d_ga, d_y, d_y16, T, iters);
        std::vector<float> y1((size_t) T * HV * S);
        check(cudaMemcpy(y1.data(), d_y, y1.size() * 4, cudaMemcpyDeviceToHost), "D2H");
        double absd = 0;
        for (size_t i = 0; i < y0.size(); ++i) {
            const double d = std::fabs((double) y1[i] - (double) y0[i]);
            if (d > absd) absd = d;
        }
        std::printf("%8lld %11.3fms %11.3fms %11.3fms %11.3fms %11.3fms   %.2e (wmma vs pipe)\n", (long long) T,
                    rec, wrp, fst, fma, ch, absd);
    }
    return 0;
}
