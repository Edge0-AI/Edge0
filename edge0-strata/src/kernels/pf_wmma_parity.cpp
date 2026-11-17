// src/kernels/pf_wmma_parity.cpp - gemm_iq_f16_grouped (the Volta prompt experts, STRATA_PF_WMMA) against a double
// reference over the same FP16 weights (iq_dequant_f16, the FP16 path's dequantization) and FP16 activations.
//
//     build/pf_wmma_parity            (needs a CUDA GPU with FP16 tensor cores: sm_70+; synthetic blobs, no model)
//
// Every gate/up format the prompt path gives it (IQ2_XXS, IQ2_S, IQ1_M, and this pack's IQ4_NL / IQ3_S / IQ4_XS: 1280
// rows of 2560) and the down products (Q2_0, and IQ4_NL: 2560 rows of 640); a group of experts with 0, 1, 63, 64, 65,
// 200 and 300 rows (one past the 256-row chunk); the rows placed past a nonzero first bound.  The sums are FP32 on
// tensor cores in another order than the reference, so the check is a tolerance: max |got - ref| <= 2e-3 * (max |ref|
// of the matrix) + 1e-6.
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

const char* name_of(int t) {
    switch (t) {
        case 16: return "IQ2_XXS";
        case 17: return "IQ2_XS";
        case 18: return "IQ3_XXS";
        case 20: return "IQ4_NL";
        case 21: return "IQ3_S";
        case 22: return "IQ2_S";
        case 23: return "IQ4_XS";
        case 29: return "IQ1_M";
        case 42: return "Q2_0";
        case 12: return "Q4_K";
        case 13: return "Q5_K";
        case 14: return "Q6_K";
        case 7: return "Q5_1";
        case 8: return "Q8_0";
        default: return "?";
    }
}
int block_values(int t) { return t == 20 || t == 7 || t == 8 ? 32 : t == 42 ? 64 : 256; }

// `rows` rows of `n` values of format t: random bytes, then a finite fp16 scale in every block
std::vector<uint8_t> random_rows(int t, int64_t rows, int64_t n, std::mt19937& rng) {
    const size_t rb = k::iq_row_bytes(t, n), bs = k::iq_row_bytes(t, block_values(t));
    std::vector<uint8_t> w((size_t) rows * rb);
    std::uniform_int_distribution<int> byte(0, 255), ex(2, 8), man(0, 1023), sgn(0, 1);
    for (auto& b : w) b = (uint8_t) byte(rng);
    for (size_t o = 0; o < w.size(); o += bs) {
        if (t == 29) {
            // IQ1_M: the fp16 scale is the top nibbles of its four scale words (bytes 48..55), the sign and the
            // exponent's top bits in the last: 0x1 / 0x2 there (0x9 / 0xA negative) keeps it in 2^-11 .. 2^-3
            const uint8_t nib = (uint8_t) ((sgn(rng) ? 0x8 : 0x0) | (1 + (byte(rng) & 1)));
            w[o + 55] = (uint8_t) ((w[o + 55] & 0x0F) | (nib << 4));
        } else {   // the other formats start their block with the fp16 scale: 2^-13 .. 2^-6
            // (Q4_K / Q5_K: 2^-14 .. 2^-11, their 6-bit sub-scales multiply it by up to 63)
            const int e = (t == 12 || t == 13) ? 1 + ex(rng) % 4 : ex(rng);
            const uint16_t h = (uint16_t) ((sgn(rng) ? 0x8000 : 0) | (e << 10) | man(rng));
            std::memcpy(&w[o], &h, 2);
            if (t == 12 || t == 13 || t == 7) {   // Q4_K / Q5_K: dmin, Q5_1: m - the second fp16, as small
                const int em = (t == 7) ? ex(rng) : 1 + ex(rng) % 4;
                const uint16_t m = (uint16_t) ((sgn(rng) ? 0x8000 : 0) | (em << 10) | man(rng));
                std::memcpy(&w[o + 2], &m, 2);
            }
        }
    }
    return w;
}


int run(int ty, int n_out, int K, std::mt19937& rng) {
    const std::vector<int> rows = {0, 1, 63, 64, 65, 200, 300};
    const int E = (int) rows.size(), base = 5;
    std::vector<int32_t> bounds(E + 1);
    bounds[0] = base;
    for (int e = 0; e < E; ++e) bounds[e + 1] = bounds[e] + rows[e];
    const int total = bounds[E];
    const size_t eb = k::iq_row_bytes(ty, K) * (size_t) n_out;
    std::vector<uint8_t> w;
    for (int e = 0; e < E; ++e) { auto r = random_rows(ty, n_out, K, rng); w.insert(w.end(), r.begin(), r.end()); }
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<__half> x((size_t) total * K);
    for (auto& v : x) v = __float2half(nd(rng));
    uint8_t* dw; __half* dx; int32_t* db; float* dy; uint16_t* dq;
    ck(cudaMalloc(&dw, w.size() + 4096), "w");
    ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice), "w");
    ck(cudaMalloc(&dx, x.size() * 2), "x");
    ck(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice), "x");
    ck(cudaMalloc(&db, bounds.size() * 4), "b");
    ck(cudaMemcpy(db, bounds.data(), bounds.size() * 4, cudaMemcpyHostToDevice), "b");
    ck(cudaMalloc(&dy, (size_t) total * n_out * 4), "y");
    ck(cudaMemset(dy, 0xff, (size_t) total * n_out * 4), "y");
    ck(cudaMalloc(&dq, (size_t) E * n_out * K * 2), "dq");
    const int maxr = *std::max_element(rows.begin(), rows.end());
    if (!k::gemm_iq_f16_grouped(ty, dw, eb, n_out, K, dx, K, db, E, maxr, dy, n_out, nullptr)) {
        std::printf("%-8s %5d x %4d: not launched\n", name_of(ty), n_out, K);
        return 1;
    }
    for (int e = 0; e < E; ++e) k::iq_dequant_f16(ty, dw + e * eb, (int64_t) n_out * K, dq + (size_t) e * n_out * K, nullptr);
    ck(cudaDeviceSynchronize(), "run");
    std::vector<float> y((size_t) total * n_out);
    std::vector<uint16_t> q((size_t) E * n_out * K);
    ck(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost), "y");
    ck(cudaMemcpy(q.data(), dq, q.size() * 2, cudaMemcpyDeviceToHost), "dq");
    double worst = 0, scale = 0;
    for (int e = 0; e < E; ++e)
        for (int r = bounds[e]; r < bounds[e + 1]; ++r)
            for (int n = 0; n < n_out; ++n) {
                double s = 0;
                const uint16_t* wr = &q[((size_t) e * n_out + n) * K];
                for (int kk = 0; kk < K; ++kk) {
                    __half h; std::memcpy(&h, &wr[kk], 2);
                    s += (double) __half2float(h) * (double) __half2float(x[(size_t) r * K + kk]);
                }
                scale = std::max(scale, std::fabs(s));
                worst = std::max(worst, std::fabs(s - (double) y[(size_t) r * n_out + n]));
            }
    const bool ok = worst <= 2e-3 * scale + 1e-6;
    std::printf("%-8s %5d x %4d, %d rows in %d experts: max |err| %.3g of max |ref| %.3g  %s\n", name_of(ty), n_out, K,
                total - base, E, worst, scale, ok ? "ok" : "FAIL");
    cudaFree(dw); cudaFree(dx); cudaFree(db); cudaFree(dy); cudaFree(dq);
    return ok ? 0 : 1;
}

}  // namespace

int main() {
    // every sm_70+ card in one process, in turn: a layer split launches the tiles on its second card after the first
    int n_dev = 0, ran = 0, fails = 0;
    cudaGetDeviceCount(&n_dev);
    for (int dev = 0; dev < n_dev; ++dev) {
        int major = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        if (major < 7) continue;
        cudaSetDevice(dev);
        std::printf("device %d\n", dev);
        ++ran;
        std::mt19937 rng(70);
        for (int ty : {16, 22, 29}) fails += run(ty, 1280, 2560, rng);
        fails += run(42, 2560, 640, rng);
        for (int ty : {20, 21, 23}) fails += run(ty, 1280, 2560, rng);   // this pack's gate/up formats, generic WK=256 path
        fails += run(20, 2560, 640, rng);                                // the IQ4_NL down, the new WK=128 lane
    }
    if (!ran) { std::printf("pf_wmma_parity: needs sm_70+, skipped\n"); return 77; }
    std::printf("pf_wmma_parity: %d failures on %d devices\n", fails, ran);
    return fails ? 1 : 0;
}
