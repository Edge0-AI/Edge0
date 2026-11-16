// src/kernels/bf16_gemv_parity.cpp - P2.S5's test for the BF16 GEMV.
//
// THREE THINGS ARE CHECKED, and the third is the one the engine's correctness rests on:
//
//   1. both kernels against a host reference that reads the SAME bf16 bits - so what is under test is the
//      reduction, not the conversion, and the tolerance is set by summation order and nothing else;
//   2. the naive and the split kernel agree with each other, because the naive one is the reference the
//      split one exists to be checked against and a disagreement there means one of them is wrong;
//   3. **THE ACTIVATION CONTRACT IS OBSERVABLE.**  A BF16 weight sees a BF16 activation; feeding it fp16 or
//      f32 instead is a 2^-9 = 1.95e-03 relative change, which is above any 1e-3 tolerance.  An fp16
//      activation is the rival reading, computed and required to differ, because it is what the engine did
//      before `docs/activation-contract.md` settled the question and it produces perfectly plausible output.
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

using strata::kernels::bf16_from_f32;
using strata::kernels::f32_from_bf16;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

double rel_l1(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0, m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += std::fabs((double) a[i] - (double) b[i]);
        m += std::fabs((double) a[i]);
    }
    return d / (m > 1e-30 ? m : 1e-30);
}

/// The host reference, in DOUBLE, over the same bf16 bits.
void reference(const std::vector<uint16_t>& x, const std::vector<uint16_t>& w, long long n_in, long long n_out,
               std::vector<float>& y) {
    y.assign((size_t) n_out, 0.0f);
    for (long long o = 0; o < n_out; ++o) {
        double acc = 0;
        for (long long i = 0; i < n_in; ++i)
            acc += (double) f32_from_bf16(x[(size_t) i]) * (double) f32_from_bf16(w[(size_t) (o * n_in + i)]);
        y[(size_t) o] = (float) acc;
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: bf16_gemv_parity [--selftest]\n"); return 2; }
    }
    int bad = 0;

    // THE REAL SHAPES, because a kernel that only works on a square toy is not a kernel this engine can use.
    // `ssm_alpha` is [2560, 48] - a wide reduction and 48 outputs - and `indexer.k_proj` is [2560, 128].
    struct Shape { long long n_in, n_out; const char* what; };
    const Shape shapes[] = {
        {2560, 48, "ssm_alpha.weight  [2560, 48]"},
        {2560, 128, "indexer.k_proj   [2560, 128]"},
        {2560, 512, "indexer.q_proj   [2560, 512]"},
        {2560, 2560, "ple_value        [2560, 2560]"},
    };

    for (const Shape& s : shapes) {
        std::mt19937 rng(4242);
        std::normal_distribution<float> g(0.0f, 1.0f);
        std::vector<float> fx((size_t) s.n_in), fw((size_t) (s.n_in * s.n_out));
        for (auto& v : fx) v = g(rng);
        // a realistic weight scale: these are projections, not embeddings
        for (auto& v : fw) v = g(rng) * 0.05f;

        // THE ACTIVATION IS ROUNDED TO BF16 FIRST, which is the contract, and the WEIGHT is already bf16 in
        // the pack - the loaded bytes ARE the high 16 bits of an f32.
        std::vector<uint16_t> x((size_t) s.n_in), w((size_t) (s.n_in * s.n_out));
        for (size_t i = 0; i < fx.size(); ++i) x[i] = bf16_from_f32(fx[i]);
        for (size_t i = 0; i < fw.size(); ++i) w[i] = bf16_from_f32(fw[i]);

        std::vector<float> want;
        reference(x, w, s.n_in, s.n_out, want);

        uint16_t *d_x = nullptr, *d_w = nullptr;
        float *d_y = nullptr;
        check(cudaMalloc(&d_x, x.size() * 2), "x");
        check(cudaMalloc(&d_w, w.size() * 2), "w");
        check(cudaMalloc(&d_y, (size_t) s.n_out * 4), "y");
        check(cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice), "cx");
        check(cudaMemcpy(d_w, w.data(), w.size() * 2, cudaMemcpyHostToDevice), "cw");

        std::vector<float> naive((size_t) s.n_out), warp((size_t) s.n_out), split((size_t) s.n_out);
        strata::kernels::bf16_gemv(d_x, d_w, d_y, s.n_in, s.n_out, nullptr);
        check(cudaMemcpy(naive.data(), d_y, naive.size() * 4, cudaMemcpyDeviceToHost), "cy1");
        strata::kernels::bf16_gemv_split(d_x, d_w, d_y, s.n_in, s.n_out, 32, nullptr);
        check(cudaMemcpy(warp.data(), d_y, warp.size() * 4, cudaMemcpyDeviceToHost), "cy2");
        strata::kernels::bf16_gemv_split(d_x, d_w, d_y, s.n_in, s.n_out, 256, nullptr);
        check(cudaMemcpy(split.data(), d_y, split.size() * 4, cudaMemcpyDeviceToHost), "cy3");

        const double rn = rel_l1(want, naive), rw = rel_l1(want, warp), rs = rel_l1(want, split);
        std::printf("  %-34s naive %.2e  warp %.2e  split256 %.2e\n", s.what, rn, rw, rs);
        // The products are EXACT - both sides are bf16-valued, so f32 represents each product exactly - and
        // the only difference is the order the 2560 terms are added.  A tight bound is therefore meaningful.
        if (!(rn <= 1e-5 && rw <= 1e-5 && rs <= 1e-5)) { std::printf("    *** over 1e-5 ***\n"); ++bad; }

        // ---- 3. THE ACTIVATION CONTRACT.  The rival: an fp16 activation instead of bf16, which is what the
        // engine did before the contract was settled.  It must be observable, or check 1 above would pass
        // against either and the contract would be untested.
        {
            std::vector<float> fp16_as_f32((size_t) s.n_in);
            std::vector<uint16_t> x16((size_t) s.n_in);
            for (size_t i = 0; i < fx.size(); ++i) {
                x16[i] = strata::kernels::f16_from_f32(fx[i]);
                fp16_as_f32[i] = strata::kernels::f32_from_f16(x16[i]);
            }
            // Round the fp16 value to bf16 - i.e. give the kernel a bf16 activation whose VALUES came through
            // fp16.  That is the whole difference between the two contracts at this call site.
            for (size_t i = 0; i < fp16_as_f32.size(); ++i) x[i] = bf16_from_f32(fp16_as_f32[i]);
            check(cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice), "cx2");
            std::vector<float> rival((size_t) s.n_out);
            strata::kernels::bf16_gemv(d_x, d_w, d_y, s.n_in, s.n_out, nullptr);
            check(cudaMemcpy(rival.data(), d_y, rival.size() * 4, cudaMemcpyDeviceToHost), "cy4");
            const double r = rel_l1(want, rival);
            const bool visible = r > 1e-4;
            std::printf("      %-30s %s (%.4f%% apart)\n", "bf16 vs fp16 activation",
                        visible ? "yes" : "*** NO ***", r * 100);
            if (!visible) ++bad;
        }
        cudaFree(d_x); cudaFree(d_w); cudaFree(d_y);
    }

    std::printf("\nbf16_gemv: %d failures\n", bad);
    if (bad) return 1;
    if (selftest) std::printf("bf16_gemv_parity OK\n");
    return 0;
}
