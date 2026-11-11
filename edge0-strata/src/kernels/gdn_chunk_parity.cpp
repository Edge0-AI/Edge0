// src/kernels/gdn_chunk_parity.cpp - v100/gdn-chunk: the chunked GDN prompt recurrence vs the token recurrence
// and a host reference (GPU, synthetic, no model).
//
// The chunked path (src/prefill/gdn_chunk.cu) computes the SAME recurrence in a different summation order:
// a 64-token chunk becomes one triangular solve and a batch of matmuls instead of 64 sequential rank-1 steps.
// That is FP32-level, NOT bit-exact - the precedent is qsa_prompt_attn.hpp.  So this test does not ask for
// equality; it measures the error of both GPU paths against a double-precision transcription of the same
// recurrence (`ref/gdn.py`'s, as re-pinned by gdn_parity) and asserts bounds:
//
//   1. the recurrence vs the host reference: the FP32 floor of the old path,
//   2. the chunked path vs the host reference: the number that matters for the model,
//   3. chunked vs recurrence directly: the observable switch STRATA_GDN_CHUNK flips.
//
// Both GPU paths go through the REAL dispatch (`gdn_recurrence`), A/B'd with the env switch, so the tail
// (< 64 tokens) and the output norm are part of what is compared - the recurrence kernels' raw FP32 output for
// the kernel comparisons (0.1.38's gdn_out_norm_kernel leaves it in place and writes only the FP16 copy), and
// the norm's FP16 output once against the host reference's normalized values.
#include "strata/prefill/gdn_chunk.hpp"
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr int S = 128, HK = 16, HV = 48, C = 10240;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// the FP16 store's value back as FP32, for the one normalized-output comparison
float half_to_float(uint16_t u) {
    __half h;
    std::memcpy(&h, &u, 2);
    return __half2float(h);
}

// deterministic LCG - the same inputs on every machine and every run
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    float uni() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return (float) ((s >> 40) * (1.0 / 16777216.0));
    }
};

/// The GDN chunk as a double-precision transcription of the recurrence in strata/kernels/gdn.hpp: the state
/// (S, h_v, S) f32 layout, decay BEFORE the rank-1 update, modulo head pairing.  `yraw_out` is the raw output
/// the recurrence kernels leave for gdn_out_norm_kernel; `y_out` is the norm's value, formed exactly as
/// gdn_out_norm_kernel forms it (sqrt(S) in the oc scale, +eps on the SCALED squared norm).
void ref_full(const std::vector<float>& st0, const std::vector<float>& h, const std::vector<float>& gate,
              const std::vector<float>& beta, const std::vector<float>& z, const std::vector<float>& gamma,
              float eps, int64_t T, std::vector<float>& yraw_out, std::vector<float>& y_out,
              std::vector<float>& st_out) {
    std::vector<double> st((size_t) S * HV * S);
    for (size_t i = 0; i < st.size(); ++i) st[i] = (double) st0[i];
    y_out.assign((size_t) T * HV * S, 0.0f);
    yraw_out.assign((size_t) T * HV * S, 0.0f);
    std::vector<double> o(S);
    for (int64_t t = 0; t < T; ++t) {
        for (int hh = 0; hh < HV; ++hh) {
            const int qh = hh % HK;
            const double dec = std::exp((double) gate[(size_t) t * HV + hh]);
            const double b = (double) beta[(size_t) t * HV + hh];
            for (int j = 0; j < S; ++j) {
                double sk = 0.0;
                for (int i = 0; i < S; ++i) sk += st[((size_t) i * HV + hh) * S + j] * (double) h[(size_t) t * C + HK * S + qh * S + i];
                const double d = ((double) h[(size_t) t * C + 2 * HK * S + hh * S + j] - dec * sk) * b;
                for (int i = 0; i < S; ++i) {
                    double& w = st[((size_t) i * HV + hh) * S + j];
                    w = dec * w + (double) h[(size_t) t * C + HK * S + qh * S + i] * d;
                }
                double dot = 0.0;
                for (int i = 0; i < S; ++i) dot += st[((size_t) i * HV + hh) * S + j] * (double) h[(size_t) t * C + qh * S + i];
                o[j] = dot / std::sqrt((double) S);  // the oc scale of gdn_rec_cols_kernel
            }
            double ss = 0.0;
            for (int j = 0; j < S; ++j) ss += o[j] * o[j];
            const double rs = 1.0 / std::sqrt(ss / (double) S + (double) eps);
            for (int j = 0; j < S; ++j) {
                yraw_out[((size_t) t * HV + hh) * S + j] = (float) o[j];
                const double sig = 1.0 / (1.0 + std::exp(-(double) z[(size_t) t * HV * S + hh * S + j]));
                y_out[((size_t) t * HV + hh) * S + j] = (float) (o[j] * rs * (double) gamma[j] * sig);
            }
        }
    }
    st_out.resize(st.size());
    for (size_t i = 0; i < st.size(); ++i) st_out[i] = (float) st[i];
}

struct Err {
    float abs_ = 0, rel_ = 0;
    void see(float a, float b) {
        const float d = std::fabs(a - b);
        const float m = std::fabs(b);
        // a NaN or inf anywhere is a hard error: an unordered comparison would otherwise leave the max at 0
        if (!std::isfinite(a) || !std::isfinite(b)) {
            abs_ = rel_ = std::numeric_limits<float>::infinity();
            return;
        }
        if (d > abs_) abs_ = d;
        const float r = d / (m + 1e-4f);
        if (r > rel_) rel_ = r;
    }
};

int failures = 0;

void expect(const char* what, const Err& e, float bound_abs, float bound_rel) {
    const bool ok = e.rel_ <= bound_rel && e.abs_ <= bound_abs;
    std::printf("  %-38s max abs %.3e  max rel %.3e  (bounds abs %.1e rel %.1e) %s\n", what, e.abs_, e.rel_,
                bound_abs, bound_rel, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

void run_case(int64_t T, uint64_t seed) {
    std::printf("T = %lld%s\n", (long long) T, (T % 64) ? "  (tail: T % 64 != 0)" : "");
    Rng rng(seed);
    std::vector<float> h((size_t) T * C), gate((size_t) T * HV), beta((size_t) T * HV), z((size_t) T * HV * S),
        gamma(S), st0((size_t) S * HV * S);
    for (int64_t t = 0; t < T; ++t) {
        for (int qh = 0; qh < HK; ++qh) {  // q and k: L2-normalised rows, as gdn_conv leaves them
            for (int part = 0; part < 2; ++part) {
                double ss = 0;
                float row[S];
                for (int i = 0; i < S; ++i) {
                    row[i] = rng.uni() - 0.5f;
                    ss += (double) row[i] * row[i];
                }
                const float nrm = (float) (1.0 / std::sqrt(ss + 1e-5));
                for (int i = 0; i < S; ++i) h[(size_t) t * C + (part ? HK * S : 0) + qh * S + i] = row[i] * nrm;
            }
        }
        for (int hh = 0; hh < HV; ++hh) {
            for (int j = 0; j < S; ++j)  // v: O(1) rows
                h[(size_t) t * C + 2 * HK * S + hh * S + j] = rng.uni() - 0.5f;
            gate[(size_t) t * HV + hh] = -0.02f - 0.3f * rng.uni();
            beta[(size_t) t * HV + hh] = 1.0f / (1.0f + std::exp(-(4.0f * rng.uni() - 2.0f)));
            for (int j = 0; j < S; ++j) z[((size_t) t * HV + hh) * S + j] = 2.0f * rng.uni() - 1.0f;
        }
    }
    for (int j = 0; j < S; ++j) gamma[j] = 0.5f + 2.0f * rng.uni();
    for (size_t i = 0; i < st0.size(); ++i) st0[i] = 0.1f * (rng.uni() - 0.5f);
    const float eps = 1e-5f;

    std::vector<float> y_ref, yraw_ref, st_ref;
    ref_full(st0, h, gate, beta, z, gamma, eps, T, yraw_ref, y_ref, st_ref);

    float *d_st = nullptr, *d_h = nullptr, *d_g = nullptr, *d_b = nullptr, *d_z = nullptr, *d_ga = nullptr,
          *d_y = nullptr, *d_st0 = nullptr;
    uint16_t* d_y16 = nullptr;
    check(cudaMalloc(&d_st, st0.size() * 4), "malloc");
    check(cudaMalloc(&d_st0, st0.size() * 4), "malloc");
    check(cudaMalloc(&d_h, h.size() * 4), "malloc");
    check(cudaMalloc(&d_g, gate.size() * 4), "malloc");
    check(cudaMalloc(&d_b, beta.size() * 4), "malloc");
    check(cudaMalloc(&d_z, z.size() * 4), "malloc");
    check(cudaMalloc(&d_ga, gamma.size() * 4), "malloc");
    check(cudaMalloc(&d_y, y_ref.size() * 4), "malloc");
    check(cudaMalloc(&d_y16, y_ref.size() * 2), "malloc");
    check(cudaMemcpy(d_st0, st0.data(), st0.size() * 4, cudaMemcpyHostToDevice), "H2D");
    check(cudaMemcpy(d_h, h.data(), h.size() * 4, cudaMemcpyHostToDevice), "H2D");
    check(cudaMemcpy(d_g, gate.data(), gate.size() * 4, cudaMemcpyHostToDevice), "H2D");
    check(cudaMemcpy(d_b, beta.data(), beta.size() * 4, cudaMemcpyHostToDevice), "H2D");
    check(cudaMemcpy(d_z, z.data(), z.size() * 4, cudaMemcpyHostToDevice), "H2D");
    check(cudaMemcpy(d_ga, gamma.data(), gamma.size() * 4, cudaMemcpyHostToDevice), "H2D");

    std::vector<float> y_gpu(y_ref.size()), st_gpu(st0.size());
    std::vector<float> y_rec(y_ref.size()), st_rec(st0.size()), y_warp(y_ref.size()), st_warp(st0.size()),
        y_fast(y_ref.size()), st_fast(st0.size()), y_fma(y_ref.size()), st_fma(st0.size());
    auto run = [&](const char* chunk, const char* warp, std::vector<float>& y, std::vector<float>& st,
                   std::vector<float>* y16f = nullptr) {
        setenv("STRATA_GDN_CHUNK", chunk, 1);
        setenv("STRATA_GDN_REC_WARP", warp, 1);
        check(cudaMemcpy(d_st, d_st0, st0.size() * 4, cudaMemcpyDeviceToDevice), "D2D");
        strata::prefill::gdn_recurrence(d_st, d_h, d_g, d_b, d_z, d_ga, eps, d_y, d_y16, T, nullptr);
        check(cudaDeviceSynchronize(), "sync");
        check(cudaMemcpy(y.data(), d_y, y.size() * 4, cudaMemcpyDeviceToHost), "D2H");
        check(cudaMemcpy(st.data(), d_st, st.size() * 4, cudaMemcpyDeviceToHost), "D2H");
        if (y16f != nullptr) {
            std::vector<uint16_t> h16(y.size());
            check(cudaMemcpy(h16.data(), d_y16, h16.size() * 2, cudaMemcpyDeviceToHost), "D2H");
            y16f->resize(y.size());
            for (size_t i = 0; i < y.size(); ++i) (*y16f)[i] = half_to_float(h16[i]);
        }
    };
    std::vector<float> y16_rec;
    run("0", "0", y_rec, st_rec, &y16_rec);  // the old pipelined recurrence (+ the norm's FP16 output)
    run("0", "1", y_warp, st_warp);  // the sync-free warp recurrence - must be BIT-EXACT vs the above
    run("0", "2", y_fast, st_fast);  // the chain-split recurrence (the V100 default)
    run("1", "0", y_gpu, st_gpu);    // the chunked path's wmma variant
    run("2", "0", y_fma, st_fma);    // the chunked path's FP32 FMA variant

    size_t bits = 0;
    for (size_t i = 0; i < y_ref.size(); ++i)
        if (std::memcmp(&y_rec[i], &y_warp[i], 4) != 0) ++bits;
    for (size_t i = 0; i < st_ref.size(); ++i)
        if (std::memcmp(&st_rec[i], &st_warp[i], 4) != 0) ++bits;
    std::printf("  %-38s %s\n", "warp recurrence vs pipelined (bits)", bits ? "FAIL" : "bit-exact");
    if (bits) ++failures;

    Err e_yr, e_y, e_yf, e_ym, e_yc, e_yn, e_sr, e_s, e_sf, e_sm, e_sc;
    for (size_t i = 0; i < y_ref.size(); ++i) {
        e_yr.see(y_rec[i], yraw_ref[i]);
        e_yf.see(y_fast[i], yraw_ref[i]);
        e_ym.see(y_fma[i], yraw_ref[i]);
        e_y.see(y_gpu[i], yraw_ref[i]);
        e_yc.see(y_gpu[i], y_rec[i]);
        e_yn.see(y16_rec[i], y_ref[i]);
    }
    for (size_t i = 0; i < st_ref.size(); ++i) {
        e_sr.see(st_rec[i], st_ref[i]);
        e_sf.see(st_fast[i], st_ref[i]);
        e_sm.see(st_fma[i], st_ref[i]);
        e_s.see(st_gpu[i], st_ref[i]);
        e_sc.see(st_gpu[i], st_rec[i]);
    }
    // The bounds are the measured errors of this build on the V100 with ~3x margin (rel uses a 1e-4 floor on
    // the reference magnitude).  The recurrence is the FP32 floor; the FMA chunked path reorders the sums
    // (FP32-level, not bit-exact - qsa_prompt_attn.hpp's precedent); the wmma chunked path additionally
    // stages q/k/v and the state through FP16 (the reference's "f32 -> f16 暂存"), which is the error level
    // llama.cpp-v100's FlashQLA sm70 path ships with.  The y rows compared here are the recurrence kernels'
    // raw output (0.1.38's norm keeps its FP32 store out); the last line reads the norm's FP16 copy instead.
    expect("recurrence vs host ref (y)", e_yr, 5e-6f, 1e-2f);
    expect("chain-split recurrence (y)", e_yf, 1e-5f, 2e-2f);
    expect("chunked FMA vs host ref (y)", e_ym, 1e-5f, 2e-2f);
    expect("chunked wmma vs host ref (y)", e_y, 6e-3f, 5e1f);
    expect("chunked wmma vs recurrence (y)", e_yc, 6e-3f, 5e1f);
    expect("output norm (FP16) vs host ref (y)", e_yn, 8e-3f, 5e-2f);
    expect("recurrence vs host ref (state)", e_sr, 1e-7f, 5e-4f);
    expect("chain-split recurrence (state)", e_sf, 5e-7f, 2e-3f);
    expect("chunked FMA vs host ref (state)", e_sm, 5e-7f, 2e-3f);
    expect("chunked wmma vs host ref (state)", e_s, 3e-4f, 2e0f);
    expect("chunked wmma vs recurrence (state)", e_sc, 3e-4f, 2e0f);

    cudaFree(d_st); cudaFree(d_st0); cudaFree(d_h); cudaFree(d_g); cudaFree(d_b); cudaFree(d_z);
    cudaFree(d_ga); cudaFree(d_y); cudaFree(d_y16);
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--selftest") == 0) selftest = true;
        else { std::fprintf(stderr, "usage: gdn_chunk_parity [--selftest]\n"); return 2; }
    }
    {
        // the kkt stage needs 2*64*129 floats (66 KB) of dynamic shared memory: a card whose opt-in limit is lower
        // (Pascal: 48 KB) cannot launch it, and the engine dispatch never selects the chunked path there
        int dev = 0, smem = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        if (smem < 2 * 64 * 129 * 4) {
            std::printf("gdn_chunk_parity: the card's shared-memory opt-in (%d B) is below the kkt stage's need, skipped\n", smem);
            return 77;
        }
    }
    if (!strata::prefill::gdn_chunk_available())
        std::printf("note: not a V100 (cc 7.0) - the parity still runs; the engine dispatch keeps the recurrence\n");
    run_case(64, 1);
    run_case(128, 2);
    run_case(200, 3);  // tail: T % 64 = 8
    run_case(320, 4);
    std::printf("\ngdn_chunk: %d failures\n", failures);
    if (selftest && failures == 0) std::printf("gdn_chunk_parity OK\n");
    return failures ? 1 : 0;
}
