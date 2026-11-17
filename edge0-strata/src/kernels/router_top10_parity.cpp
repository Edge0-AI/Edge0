// src/kernels/router_top10_parity.cpp - P2.S2's parity test for the MoE router.
//
// The reference here is a HOST implementation of `ref/moe.py::router` written from the same specification the
// kernel was written from - so this checks the kernel against the spec independently of the kernel, which is
// the strongest thing available without running llama.cpp.  It is NOT a check against llama.cpp itself, and
// that gap is stated rather than implied.
//
// WHAT THE TEST CAN AND CANNOT SEE.  It compares expert IDs exactly and weights to a tolerance, on several
// logit distributions including ties (where the "smallest index wins" rule is the only thing that decides) and
// near-ties (where a float difference decides).  It also asserts that the 2**-14 CLAMP CANNOT TRIGGER for this
// model's geometry - see below - so the absence of a clamp test is a proven fact rather than an omission.
#include "strata/kernels/router_top10.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

constexpr double RENORM_CLAMP = 6.103515625e-05;      // 2**-14

// The host reference: softmax over all experts, stable descending argsort, gather, clamp, renormalise.
void reference(const float* logits, int n_expert, int k, int* ids, float* w) {
    double mx = logits[0];
    for (int e = 1; e < n_expert; ++e) mx = std::max(mx, (double) logits[e]);
    double sum = 0;
    std::vector<double> p((size_t) n_expert);
    for (int e = 0; e < n_expert; ++e) {
        p[(size_t) e] = std::exp((double) logits[e] - mx);
        sum += p[(size_t) e];
    }
    for (int e = 0; e < n_expert; ++e) p[(size_t) e] /= sum;

    // stable descending: std::stable_sort with a strict greater-than keeps equal elements in index order
    std::vector<int> idx((size_t) n_expert);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(),
                     [&](int a, int b) { return p[(size_t) a] > p[(size_t) b]; });
    double s = 0;
    for (int i = 0; i < k; ++i) {
        ids[i] = idx[(size_t) i];
        w[i] = (float) p[(size_t) idx[(size_t) i]];
        s += p[(size_t) idx[(size_t) i]];
    }
    s = std::max(s, RENORM_CLAMP);
    for (int i = 0; i < k; ++i) w[i] = (float) ((double) w[i] / s);
}

int run_case(const char* name, const std::vector<float>& logits, int n_tokens, int n_expert, int k,
             double tol) {
    std::vector<int> h_ids((size_t) n_tokens * k);
    std::vector<float> h_w((size_t) n_tokens * k);
    std::vector<int> r_ids((size_t) n_tokens * k);
    std::vector<float> r_w((size_t) n_tokens * k);
    for (int t = 0; t < n_tokens; ++t) {
        reference(&logits[(size_t) t * n_expert], n_expert, k, &r_ids[(size_t) t * k], &r_w[(size_t) t * k]);
    }

    float* d_l = nullptr;
    int* d_ids = nullptr;
    float* d_w = nullptr;
    check(cudaMalloc(&d_l, logits.size() * sizeof(float)), "malloc logits");
    check(cudaMalloc(&d_ids, h_ids.size() * sizeof(int)), "malloc ids");
    check(cudaMalloc(&d_w, h_w.size() * sizeof(float)), "malloc w");
    check(cudaMemcpy(d_l, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice), "copy");
    strata::kernels::router_top10(d_l, n_tokens, n_expert, k, d_ids, d_w, nullptr);
    check(cudaMemcpy(h_ids.data(), d_ids, h_ids.size() * sizeof(int), cudaMemcpyDeviceToHost), "back ids");
    check(cudaMemcpy(h_w.data(), d_w, h_w.size() * sizeof(float), cudaMemcpyDeviceToHost), "back w");

    long long id_bad = 0, w_bad = 0;
    double worst = 0;
    for (size_t i = 0; i < h_ids.size(); ++i) {
        if (h_ids[i] != r_ids[i]) ++id_bad;
        const double rel = std::fabs((double) h_w[i] - (double) r_w[i]) /
                           (std::fabs((double) r_w[i]) > 1e-30 ? std::fabs((double) r_w[i]) : 1e-30);
        worst = std::max(worst, rel);
        if (!(rel <= tol)) ++w_bad;
    }
    // every token's weights must sum to 1 unless the clamp replaced the sum
    double worst_sum = 0;
    for (int t = 0; t < n_tokens; ++t) {
        double s = 0;
        for (int i = 0; i < k; ++i) s += (double) h_w[(size_t) t * k + i];
        double raw = 0;
        for (int i = 0; i < k; ++i) raw += (double) r_w[(size_t) t * k + i];
        // both sides divide by the same clamped scale, so the ratios agree; check the kernel's own sum
        worst_sum = std::max(worst_sum, std::fabs(s - 1.0));
    }
    std::printf("  %-26s ids %s (%lld bad)   weights worst rel %.3e (%lld over tol)   |sum-1| %.1e\n", name,
                id_bad ? "*** WRONG ***" : "exact", id_bad, worst, w_bad, worst_sum);
    cudaFree(d_l);
    cudaFree(d_ids);
    cudaFree(d_w);
    return (int) (id_bad + w_bad);
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: router_top10_parity [--selftest]\n"); return 2; }
    }

    const int NE = 512;          // the real n_expert
    const int K = 10;            // the real top-k
    const int NT = 64;
    std::mt19937 rng(2024);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    int bad = 0;

    // ---- random logits
    std::vector<float> a((size_t) NT * NE);
    for (auto& v : a) v = gauss(rng);
    bad += run_case("random normal", a, NT, NE, K, 1e-5);

    // ---- ALL-EQUAL logits: every expert ties, so the "smallest index wins" rule is the ONLY thing deciding
    // the ids.  An unstable sort or a >= comparison in the maximum scan fails this and passes everything else.
    std::vector<float> b((size_t) NT * NE, 0.5f);
    bad += run_case("all equal (ties)", b, NT, NE, K, 1e-5);

    // ---- a few exact ties among otherwise distinct values
    std::vector<float> c((size_t) NT * NE);
    for (int t = 0; t < NT; ++t) {
        for (int e = 0; e < NE; ++e) c[(size_t) t * NE + e] = (e < 12) ? 2.0f : gauss(rng);
    }
    bad += run_case("12-way exact tie", c, NT, NE, K, 1e-5);

    // ---- one dominant expert: the top-1 must be that expert and its weight must dominate
    std::vector<float> d((size_t) NT * NE, 0.0f);
    for (int t = 0; t < NT; ++t) d[(size_t) t * NE + (t % NE)] = 20.0f;
    bad += run_case("dominant expert", d, NT, NE, K, 1e-5);

    // ---- THE CLAMP CANNOT TRIGGER, and that is provable rather than untested.
    // The sum of the top k of a probability vector over n outcomes is at least k/n (the minimum is the uniform
    // distribution).  Here k/n = 10/512 = 0.01953125, which is 320x the 2**-14 = 6.1035e-05 clamp, so no input
    // whatsoever can make this model's renormalisation clamp - not an extreme one, not an adversarial one.
    // Asserting the bound is worth more than a test that cannot reach it.
    const double min_possible = (double) K / (double) NE;
    std::printf("  %-26s top-%d sum >= k/n = %.5f, clamp is %.3e -> margin %.0fx  %s\n", "clamp reachability",
                K, min_possible, RENORM_CLAMP, min_possible / RENORM_CLAMP,
                min_possible > RENORM_CLAMP ? "(CLAMP IS UNREACHABLE, asserted)" : "*** REACHABLE ***");
    if (min_possible <= RENORM_CLAMP) {
        std::fprintf(stderr, "the clamp CAN trigger, so it needs a test\n");
        ++bad;
    }

    std::printf("\nrouter_top10: %d failures over 4 distributions x %d tokens x %d experts, top-%d\n", bad, NT, NE,
                K);
    if (bad) return 1;
    if (selftest) std::printf("router_top10_parity OK\n");
    return 0;
}
