// src/prefill/gr_upmix_parity.cu - the hyper-connection read's fused up projection + mix (gr_upmix, STRATA_HC_UPMIX=1)
// against the default pair (Gemm::bf16 - cuBLAS - writing `gated`, then gr_mix_r), on synthetic inputs of the model's
// shapes (N = 2560, 4 streams, K = 320).  Where cuBLAS's kernel sums K in the fused kernel's order the outputs are the
// same bits (CUDA 13.3 on an RTX 5090 from 33 tokens - the prompt path's default there); elsewhere FP32 rounding: the
// check bounds max |diff| / max |value| of `mixed`, counts the BF16 / FP16 images that differ, and measures both
// against an FP64 reference on sampled tokens (the fused one must be as close to it as cuBLAS).
//   gr_upmix_parity            the checks (T = 1 .. 32768)
//   gr_upmix_parity --bench    and the timings of the pair and the fused kernel at T = 2048 / 8192 / 32768
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;

uint16_t to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
float from_bf16(uint16_t h) {
    const uint32_t u = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
bool ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) std::printf("FAIL: %s: %s\n", what, cudaGetErrorString(e));
    return e == cudaSuccess;
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    if (!ck(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc")) std::exit(2);
    return p;
}

struct Inputs {
    int64_t T;
    std::vector<uint16_t> lo16, wu;
    std::vector<float> R, rs, w;
    uint16_t *d_lo16, *d_wu;
    float *d_R, *d_rs, *d_w;
    Inputs(int64_t t, uint32_t seed) : T(t) {
        std::mt19937 g(seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        lo16.resize((size_t) T * LR);
        for (auto& x : lo16) x = to_bf16(nd(g) * 0.5f);            // silu outputs: O(1) and below
        wu.resize((size_t) D * LR);
        for (auto& x : wu) x = to_bf16(nd(g) * 0.12f);              // gated ~ N(0, ~1): sigmoid in its live range
        R.resize((size_t) T * D);
        for (auto& x : R) x = nd(g) * 3.0f;
        rs.resize((size_t) T * HC);
        for (auto& x : rs) x = 0.2f + 0.3f * std::fabs(nd(g));
        w.resize(D);
        for (auto& x : w) x = 1.0f + 0.1f * nd(g);
        d_lo16 = dalloc<uint16_t>(lo16.size());
        d_wu = dalloc<uint16_t>(wu.size());
        d_R = dalloc<float>(R.size());
        d_rs = dalloc<float>(rs.size());
        d_w = dalloc<float>(w.size());
        cudaMemcpy(d_lo16, lo16.data(), lo16.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_wu, wu.data(), wu.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_R, R.data(), R.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_rs, rs.data(), rs.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_w, w.data(), w.size() * 4, cudaMemcpyHostToDevice);
    }
    ~Inputs() { cudaFree(d_lo16); cudaFree(d_wu); cudaFree(d_R); cudaFree(d_rs); cudaFree(d_w); }
};

struct Outputs {
    float* mixed;
    uint16_t *bf, *h;
    explicit Outputs(int64_t T) {
        mixed = dalloc<float>((size_t) T * N);
        bf = dalloc<uint16_t>((size_t) T * N);
        h = dalloc<uint16_t>((size_t) T * N);
    }
    ~Outputs() { cudaFree(mixed); cudaFree(bf); cudaFree(h); }
};

// mixed[t][d] in FP64 from the same BF16 inputs (the exact result both GPU paths round)
double exact_mixed(const Inputs& in, int64_t t, int d) {
    double s = 0.0;
    for (int c = 0; c < HC; ++c) {
        double g = 0.0;
        const int j = c * N + d;
        for (int k = 0; k < LR; ++k)
            g += (double) from_bf16(in.lo16[(size_t) t * LR + k]) * (double) from_bf16(in.wu[(size_t) j * LR + k]);
        const double x = (double) in.R[(size_t) t * D + j] * in.rs[(size_t) t * HC + c] * in.w[j];
        s += x / (1.0 + std::exp(-g));
    }
    return s / HC;
}

int check(strata::prefill::Gemm& gm, cudaStream_t st, float* gated, int64_t T, uint32_t seed) {
    Inputs in(T, seed);
    Outputs ref(T), got(T);
    gm.bf16(in.d_lo16, in.d_wu, gated, T, D, LR);
    strata::prefill::gr_mix_r(in.d_R, in.d_rs, in.d_w, gated, ref.mixed, ref.bf, T, st, ref.h);
    if (!strata::prefill::gr_upmix(in.d_lo16, in.d_wu, in.d_R, in.d_rs, in.d_w, got.mixed, got.bf, got.h, T, st)) {
        std::printf("FAIL T=%lld: gr_upmix refused (needs CUDA sm_80+)\n", (long long) T);
        return 1;
    }
    if (!ck(cudaStreamSynchronize(st), "run")) return 1;
    const size_t n = (size_t) T * N;
    std::vector<float> a(n), b(n);
    std::vector<uint16_t> abf(n), bbf(n), ah(n), bh(n);
    cudaMemcpy(a.data(), ref.mixed, n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), got.mixed, n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(abf.data(), ref.bf, n * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(bbf.data(), got.bf, n * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(ah.data(), ref.h, n * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(bh.data(), got.h, n * 2, cudaMemcpyDeviceToHost);
    double md = 0, mv = 0;
    size_t bf_diff = 0, h_diff = 0, bit_diff = 0, nonfinite = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(b[i])) ++nonfinite;
        md = std::max(md, (double) std::fabs(a[i] - b[i]));
        mv = std::max(mv, (double) std::fabs(a[i]));
        bf_diff += abf[i] != bbf[i];
        h_diff += ah[i] != bh[i];
        bit_diff += std::memcmp(&a[i], &b[i], 4) != 0;
    }
    // FP64 on sampled tokens: each path's max |error| / max |value|
    std::mt19937 g(seed ^ 0x5a5a);
    const int samples = (int) std::min<int64_t>(T, 24);
    double ea = 0, eb = 0, ev = 0;
    for (int i = 0; i < samples; ++i) {
        const int64_t t = T <= 24 ? i : (int64_t) (g() % (uint32_t) T);
        for (int d = 0; d < N; d += 37) {
            const double x = exact_mixed(in, t, d);
            ea = std::max(ea, std::fabs(a[(size_t) t * N + d] - x));
            eb = std::max(eb, std::fabs(b[(size_t) t * N + d] - x));
            ev = std::max(ev, std::fabs(x));
        }
    }
    const double rel = md / std::max(mv, 1e-30);
    const bool bad = nonfinite != 0 || rel > 1e-5 || eb > 2.0 * ea + 1e-6 * ev;
    std::printf("%s T=%-6lld %s mixed max |diff| / max |value| %.2e, BF16 / FP16 images differing %.4f%% / %.4f%%; "
                "vs FP64 (sampled): cuBLAS + mix %.2e, fused %.2e\n", bad ? "FAIL" : "ok  ", (long long) T,
                bit_diff == 0 && bf_diff == 0 && h_diff == 0 ? "bitwise " : "        ", rel,
                100.0 * (double) bf_diff / (double) n, 100.0 * (double) h_diff / (double) n, ea / ev, eb / ev);
    return bad ? 1 : 0;
}

float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

void bench(strata::prefill::Gemm& gm, cudaStream_t st, float* gated, int64_t T) {
    Inputs in(T, 7);
    Outputs o(T);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    std::vector<float> tp, tg, tm, tf;
    for (int i = 0; i < 23; ++i) {
        const bool fused_first = i & 1;   // alternate which runs first
        for (int k = 0; k < 2; ++k) {
            if ((k == 0) == fused_first) {
                cudaEventRecord(e0, st);
                strata::prefill::gr_upmix(in.d_lo16, in.d_wu, in.d_R, in.d_rs, in.d_w, o.mixed, o.bf, o.h, T, st);
                cudaEventRecord(e1, st);
                cudaEventSynchronize(e1);
                float ms;
                cudaEventElapsedTime(&ms, e0, e1);
                if (i >= 3) tf.push_back(ms);
            } else {
                cudaEvent_t em;
                cudaEventCreate(&em);
                cudaEventRecord(e0, st);
                gm.bf16(in.d_lo16, in.d_wu, gated, T, D, LR);
                cudaEventRecord(em, st);
                strata::prefill::gr_mix_r(in.d_R, in.d_rs, in.d_w, gated, o.mixed, o.bf, T, st, o.h);
                cudaEventRecord(e1, st);
                cudaEventSynchronize(e1);
                float ms, msg;
                cudaEventElapsedTime(&ms, e0, e1);
                cudaEventElapsedTime(&msg, e0, em);
                if (i >= 3) { tp.push_back(ms); tg.push_back(msg); tm.push_back(ms - msg); }
                cudaEventDestroy(em);
            }
        }
    }
    const float p = median(tp), f = median(tf);
    // the fused kernel's bytes: R and the outputs (FP32 + 3 x 16-bit), lo16 and w_up once
    const double bytes = (double) T * D * 4 + (double) T * N * (4 + 6) + (double) T * LR * 2 + (double) D * LR * 2;
    std::printf("  T=%-6lld default pair %.3f ms (up GEMM %.3f + gr_mix_r %.3f) | fused %.3f ms: %.2fx  "
                "(fused: %.0f GB/s of its bytes, %.0f TFLOPS of its GEMM)\n", (long long) T, p, median(tg), median(tm),
                f, p / f, bytes / f / 1e6, 2.0 * T * D * LR / f / 1e9);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
}

}  // namespace

int main(int argc, char** argv) {
    {   // the fused read is sm_80+ (cp.async, BF16 mma): an older card has nothing to compare
        int dev = 0, major = 0;
        if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) {
            std::printf("gr_upmix_parity: no CUDA device, skipped\n");
            return 77;
        }
        if (major < 8) { std::printf("gr_upmix_parity: sm_%d0, the fused read needs sm_80+: skipped\n", major); return 77; }
    }
    const bool do_bench = argc > 1 && std::string(argv[1]) == "--bench";
    cudaStream_t st = nullptr;
    if (!ck(cudaStreamCreate(&st), "stream")) return 1;
    strata::prefill::Gemm gm;
    std::string err;
    if (!gm.init(st, 0, err)) { std::printf("FAIL: %s\n", err.c_str()); return 1; }
    const int64_t Tmax = 32768;
    float* gated = dalloc<float>((size_t) Tmax * D);
    int fails = 0;
    uint32_t seed = 101;
    for (const int64_t T : {1, 7, 16, 31, 32, 33, 48, 63, 64, 65, 96, 126, 127, 128, 129, 600, 1000, 2048, 4099, 8192, 32768}) fails += check(gm, st, gated, T, seed++);
    if (do_bench) {
        std::printf("--bench: the hyper-connection read's up projection + mix (medians of 20 alternating runs)\n");
        for (const int64_t T : {2048, 8192, 32768}) bench(gm, st, gated, T);
    }
    cudaFree(gated);
    std::printf("gr_upmix_parity: %d failures\n", fails);
    return fails ? 1 : 0;
}
