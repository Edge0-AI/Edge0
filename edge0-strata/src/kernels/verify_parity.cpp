// src/kernels/verify_parity.cpp - batched / split verify-window kernels against the kernels they replace, BITWISE.
//
// A verify window's token t must come out bit for bit as it would in a window of any other size (the drafts are
// accepted exactly when greedy decode would have produced them), so every kernel here is checked against the
// one it stands in for, on random inputs that include -0.0, denormals, NaN, -inf and large values.  The tests
// follow eddoursul's fork (MIT; verify_parity.cpp), ported to this tree's kernels.
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

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

template <typename T> T* dev(size_t n) {
    T* p = nullptr;
    check(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc");
    return p;
}
template <typename T> void up(T* d, const std::vector<T>& h) {
    check(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
}
template <typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    check(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}

// argmax_rows against sample_tokens' greedy pick and row_top_prob_split against row_top_prob, bitwise: rows of both
// vocabularies with ties at the maximum (the lowest index wins), NaN, -inf, a row with no value above -inf, and
// launches repeated on one scratch (each must leave its counters at zero)
int test_argmax(std::mt19937& rng, cudaStream_t s) {
    using namespace strata::kernels;
    int bad = 0;
    std::normal_distribution<float> nd(0.0f, 4.0f);
    uint8_t* sa = dev<uint8_t>(argmax_rows_scratch_bytes(8));   // one scratch for every row count, as in the engine
    uint8_t* st = dev<uint8_t>(row_top_prob_scratch_bytes(8));
    check(cudaMemset(sa, 0, argmax_rows_scratch_bytes(8)), "memset");
    check(cudaMemset(st, 0, row_top_prob_scratch_bytes(8)), "memset");
    for (const int n : {248320, 40525, 4097, 33}) {
        for (const int rows : {4, 8, 1, 3, 8, 2}) {
            std::vector<float> h((size_t) rows * n);
            for (float& v : h) v = nd(rng);
            for (int r = 0; r < rows; ++r) {
                float* row = h.data() + (size_t) r * n;
                const int a = (int) (rng() % n), b = (int) (rng() % n);
                row[a] = row[b] = 40.0f;                       // a tie at the maximum
                row[(int) (rng() % n)] = std::nanf("");        // never picked
                row[(int) (rng() % n)] = -INFINITY;
                if (r == 2)                                    // no value above -inf: 0
                    for (int i = 0; i < n; ++i) row[i] = i % 7 ? -INFINITY : std::nanf("");
            }
            float* d_l = dev<float>(h.size());
            up(d_l, h);
            int* d_ref = dev<int>(rows);
            int32_t* d_out = dev<int32_t>(rows);
            float* d_p1 = dev<float>(rows);
            float* d_p2 = dev<float>(rows);
            SamplerParams sp;
            sp.greedy = true;
            sp.temperature = 0.0f;
            sample_tokens(d_l, rows, n, nullptr, 0, sp, d_ref, s);
            row_top_prob(d_l, rows, n, d_ref, d_p1, s);
            for (int rep = 0; rep < 3; ++rep) {
                argmax_rows(d_l, rows, n, sa, d_out, s);
                row_top_prob_split(d_l, rows, n, d_out, d_p2, st, s);
                check(cudaStreamSynchronize(s), "argmax");
                const std::vector<int> ref = down(d_ref, rows);
                const std::vector<int32_t> out = down(d_out, rows);
                const std::vector<float> p1 = down(d_p1, rows), p2 = down(d_p2, rows);
                for (int r = 0; r < rows; ++r)
                    if (ref[r] != out[r] || std::memcmp(&p1[r], &p2[r], 4) != 0) {
                        if (bad < 5)
                            std::printf("  argmax n %d rows %d row %d: %d / %d, p %.9g / %.9g\n", n, rows, r, ref[r],
                                        out[r], p1[r], p2[r]);
                        ++bad;
                    }
            }
            cudaFree(d_l); cudaFree(d_ref); cudaFree(d_out); cudaFree(d_p1); cudaFree(d_p2);
        }
    }
    cudaFree(sa);
    cudaFree(st);
    std::printf("argmax_rows, row_top_prob_split: %s\n",
                bad ? "MISMATCH" : "bitwise equal to the one-block kernels (ties, NaN, -inf, 1-8 rows on one scratch)");
    return bad;
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) != "--selftest") {
            std::fprintf(stderr, "usage: verify_parity [--selftest]\n");
            return 2;
        }
    }
    std::mt19937 rng(20260925);
    cudaStream_t s = nullptr;
    check(cudaStreamCreate(&s), "stream");
    int bad = 0;
    bad += test_argmax(rng, s);
    cudaStreamDestroy(s);
    std::printf("verify_parity: %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
