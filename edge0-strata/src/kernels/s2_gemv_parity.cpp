// src/kernels/s2_gemv_parity.cpp - P2.S2's parity test for the S2 GEMV.
//
// The reference is the same CHAIN as the decode test: `strata::dequantize_q2_0` is a scalar transcription that
// `bench/micro/dequant_xcheck` checks against **ggml's own** dequantizer, so comparing the GPU GEMV against a
// dot product built on that scalar decode compares against ggml through one proven hop.
//
// WHY A TOLERANCE AND NOT BIT EQUALITY.  The decode test is bit-exact because S2's decode is exact
// integer-to-float arithmetic.  A dot product is not: the compiler is free to contract `a*b + acc` into an
// FMA, which changes the result in the last bits, and nvcc does.  So this asserts a RELATIVE error, measures
// what it actually is, and prints the number - the phase spec allows 1e-3 for FP16 paths, and if the measured
// error were near that the tolerance would be hiding something rather than bounding rounding.
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/s2_gemv.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int QK = 64;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// FP16 patterns, and they are deliberately NOT all powers of two.
//
// The first version of this table was {1.0, 0.5, 0.25, ...} - every value a power of two.  The test then
// reported a worst relative error of EXACTLY zero, which is not a property of the kernel: (code-1) is in
// {-1,0,1,2} and a power-of-two times an fp16 is exact, so every product was exact and no rounding could
// occur in either implementation.  A test whose arithmetic cannot round cannot exercise its own tolerance, and
// it would have hidden an accumulation-order or contraction bug for ever.
//
// The added patterns have full 11-bit mantissas, so products round and the two implementations'
// rounding/contraction behaviour actually matters.
const uint16_t kScales[] = {
    0x3C00, 0x3800, 0x3400, 0x3000, 0x2C00, 0x2800, 0x2000,          // 1, 0.5, 0.25, ... powers of two
    0xBC00, 0xB800, 0xB400, 0xB000, 0xAC00, 0x1800, 0x1400,
    0x3E00, 0x3555, 0x3C01, 0x4248, 0x4123, 0x2AAA, 0x4A2B,          // 1.75, 0.333.., 1.0009765625, ...
    0xBE00, 0xB555, 0xC248, 0x2AAB, 0x4A2C, 0x2AAB};

}  // namespace

int main(int argc, char** argv) {
    long long n_in = 2560;          // the real n_embd, so the row length is the engine's
    long long n_out = 640;          // the real expert intermediate width
    double tol = 1e-5;
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--selftest") == 0) selftest = true;
        else if (std::strcmp(argv[i], "--n-in") == 0 && i + 1 < argc) n_in = std::atoll(argv[++i]);
        else if (std::strcmp(argv[i], "--n-out") == 0 && i + 1 < argc) n_out = std::atoll(argv[++i]);
        else if (std::strcmp(argv[i], "--tol") == 0 && i + 1 < argc) tol = std::atof(argv[++i]);
        else {
            std::fprintf(stderr, "usage: s2_gemv_parity [--selftest] [--n-in N] [--n-out N] [--tol T]\n");
            return 2;
        }
    }
    if (n_in % QK != 0) { std::fprintf(stderr, "--n-in must be a multiple of %d\n", QK); return 2; }
    const long long nb = n_in / QK;

    std::mt19937 rng(999);
    const int n_scales = (int) (sizeof(kScales) / sizeof(kScales[0]));

    // FP16 activations.  Patterns are drawn from the same finite/exact table as the decode test, because a
    // random 16-bit pattern can be NaN and a NaN makes the comparison fail for the wrong reason.
    std::vector<uint16_t> x((size_t) n_in);
    for (long long i = 0; i < n_in; ++i) x[(size_t) i] = kScales[rng() % n_scales];

    // S2 weight planes for an [n_in, n_out] matrix: row o occupies a contiguous run of nb blocks.
    std::vector<uint8_t> codes((size_t) n_out * nb * 16);
    std::vector<float> scales((size_t) n_out * nb);
    for (size_t i = 0; i < codes.size(); ++i) codes[i] = (uint8_t) (rng() & 0xFF);
    for (size_t i = 0; i < scales.size(); ++i) {
        scales[i] = strata::fp16_to_fp32(kScales[rng() % n_scales]);
    }

    // CPU reference: rebuild each row's RAW Q2_0 blocks from the planes, decode with the scalar path that
    // dequant_xcheck proved equal to ggml, and accumulate in the same order the kernel does.
    std::vector<float> ref((size_t) n_out, 0.0f);
    std::vector<uint8_t> raw(18);
    std::vector<float> dec(QK);
    for (long long o = 0; o < n_out; ++o) {
        float acc = 0.0f;
        for (long long b = 0; b < nb; ++b) {
            const float d = scales[(size_t) (o * nb + b)];
            // write `d` back as the fp16 pattern the plane came from: the plane stores a WIDENED fp16, so
            // re-narrowing is exact and the scalar decoder sees the same value the kernel does
            const uint16_t d_bits = (uint16_t) [&] {
                for (uint16_t c : kScales) {
                    if (strata::fp16_to_fp32(c) == d) return c;
                }
                return (uint16_t) 0x3C00;
            }();
            raw[0] = (uint8_t) (d_bits & 0xFF);
            raw[1] = (uint8_t) (d_bits >> 8);
            std::memcpy(&raw[2], &codes[(size_t) ((o * nb + b) * 16)], 16);
            strata::dequantize_q2_0(raw.data(), dec.data());
            for (int j = 0; j < QK; ++j) {
                acc += dec[(size_t) j] * strata::fp16_to_fp32(x[(size_t) (b * QK + j)]);
            }
        }
        ref[(size_t) o] = acc;
    }

    uint16_t* d_x = nullptr;
    uint8_t* d_codes = nullptr;
    float *d_scales = nullptr, *d_y = nullptr;
    check(cudaMalloc(&d_x, x.size() * sizeof(uint16_t)), "cudaMalloc x");
    check(cudaMalloc(&d_codes, codes.size()), "cudaMalloc codes");
    check(cudaMalloc(&d_scales, scales.size() * sizeof(float)), "cudaMalloc scales");
    check(cudaMalloc(&d_y, (size_t) n_out * sizeof(float)), "cudaMalloc y");
    check(cudaMemcpy(d_x, x.data(), x.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "copy x");
    check(cudaMemcpy(d_codes, codes.data(), codes.size(), cudaMemcpyHostToDevice), "copy codes");
    check(cudaMemcpy(d_scales, scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice),
          "copy scales");

    strata::kernels::s2_gemv(d_x, d_codes, d_scales, d_y, n_in, n_out);

    std::vector<float> got((size_t) n_out);
    check(cudaMemcpy(got.data(), d_y, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy back");

    long long bad = 0, first_bad = -1;
    double worst = 0.0;
    for (long long o = 0; o < n_out; ++o) {
        const double a = ref[(size_t) o], b = got[(size_t) o];
        const double denom = std::fabs(a) > 1e-30 ? std::fabs(a) : 1e-30;
        const double rel = std::fabs(a - b) / denom;
        if (!(rel <= worst)) worst = rel;
        if (!(rel <= tol)) {
            if (first_bad < 0) first_bad = o;
            ++bad;
        }
    }

    // NON-VACUITY: a zero matrix and a zero activation vector agree at zero.  The reference must actually vary
    // and its magnitudes must be non-trivial, or the comparison cannot distinguish anything.
    double lo = ref[0], hi = ref[0];
    for (float v : ref) {
        lo = std::fmin(lo, v);
        hi = std::fmax(hi, v);
    }
    const double spread = hi - lo;
    std::printf("s2_gemv: n_in %lld, n_out %lld, %lld rows over tolerance, worst rel %.3e (tol %.1e)\n", n_in,
                n_out, bad, worst, tol);
    std::printf("  reference spread [%.4g, %.4g]\n", lo, hi);
    if (spread <= 0.0) {
        std::fprintf(stderr, "VACUOUS: the reference is constant, so agreement proves nothing\n");
        return 1;
    }
    if (bad) {
        std::printf("  first bad row %lld: ref %.9g got %.9g\n", first_bad, (double) ref[(size_t) first_bad],
                    (double) got[(size_t) first_bad]);
        return 1;
    }
    std::printf("  agrees with the scalar decode that dequant_xcheck proved equal to ggml\n");
    if (selftest) std::printf("s2_gemv_parity OK\n");

    cudaFree(d_x);
    cudaFree(d_codes);
    cudaFree(d_scales);
    cudaFree(d_y);
    return 0;
}
