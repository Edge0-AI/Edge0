// src/kernels/dequant_s2_parity.cpp - P2.S2's parity test for the S2 decode.
//
// THE REFERENCE IS THE CHAIN, NOT A SECOND OPINION.  `strata::dequantize_q2_0` (include/strata/artifact/
// dequant.hpp) is a scalar transcription that `bench/micro/dequant_xcheck` checks against **ggml's own**
// dequantizer on real block bytes; and `tools/canonical_xcheck.py --full` checks that the canonical form
// decodes to the same values on every block of the artifact.  So this compares "GPU canonical decode" against
// "CPU decode already proven equal to ggml", and a fault in either the canonicaliser or the kernel shows up
// here.  A parity test whose two sides were written together would show neither.
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/dequant_s2.hpp"

#include <cuda_runtime.h>

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

// fp16 patterns that are finite, normal and exactly representable.  Random 16-bit patterns would include NaN
// and infinity, and a NaN makes `!=` true for a reason that has nothing to do with the kernel - the test
// would fail for the wrong cause, or worse, a real fault would hide behind a NaN it caused itself.
const uint16_t kScales[] = {0x3C00, 0x3800, 0x3400, 0x3000, 0x2C00, 0x2800, 0x2000,
                            0xBC00, 0xB800, 0xB400, 0xB000, 0xAC00, 0x1800, 0x1400};

}  // namespace

int main(int argc, char** argv) {
    long long n_blocks = 200000;
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--selftest") == 0) selftest = true;
        else if (std::strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) n_blocks = std::atoll(argv[++i]);
        else {
            std::fprintf(stderr, "usage: dequant_s2_parity [--selftest] [--blocks N]\n");
            return 2;
        }
    }
    if (n_blocks <= 0) { std::fprintf(stderr, "--blocks must be positive\n"); return 2; }

    std::mt19937 rng(12345);
    const int n_scales = (int) (sizeof(kScales) / sizeof(kScales[0]));

    // raw Q2_0 blocks: { fp16 d ; uint8 qs[16] } - what ggml produced and what the artifact stores
    std::vector<uint8_t> raw((size_t) n_blocks * 18);
    for (long long b = 0; b < n_blocks; ++b) {
        const uint16_t d = kScales[rng() % n_scales];
        raw[(size_t) b * 18 + 0] = (uint8_t) (d & 0xFF);
        raw[(size_t) b * 18 + 1] = (uint8_t) (d >> 8);
        for (int t = 0; t < 16; ++t) raw[(size_t) b * 18 + 2 + t] = (uint8_t) (rng() & 0xFF);
    }

    // the CPU reference, over the RAW blocks
    std::vector<float> cpu((size_t) n_blocks * QK);
    for (long long b = 0; b < n_blocks; ++b) {
        strata::dequantize_q2_0(&raw[(size_t) b * 18], &cpu[(size_t) b * QK]);
    }

    // canonical planes: S2 packs 4 codes per byte exactly as Q2_0 stores them, so the codes plane is the raw
    // qs bytes; the scales plane is the fp16 `d` widened.  Copied into planes of their own rather than
    // aliased, because that is what the kernel will read in the engine.
    std::vector<uint8_t> codes((size_t) n_blocks * 16);
    std::vector<float> scales((size_t) n_blocks);
    for (long long b = 0; b < n_blocks; ++b) {
        std::memcpy(&codes[(size_t) b * 16], &raw[(size_t) b * 18 + 2], 16);
        const uint16_t d = (uint16_t) (raw[(size_t) b * 18] | (raw[(size_t) b * 18 + 1] << 8));
        scales[(size_t) b] = strata::fp16_to_fp32(d);
    }

    uint8_t* d_codes = nullptr;
    float *d_scales = nullptr, *d_out = nullptr;
    check(cudaMalloc(&d_codes, codes.size()), "cudaMalloc codes");
    check(cudaMalloc(&d_scales, scales.size() * sizeof(float)), "cudaMalloc scales");
    check(cudaMalloc(&d_out, cpu.size() * sizeof(float)), "cudaMalloc out");
    check(cudaMemcpy(d_codes, codes.data(), codes.size(), cudaMemcpyHostToDevice), "copy codes");
    check(cudaMemcpy(d_scales, scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice),
          "copy scales");

    strata::kernels::dequant_s2(d_codes, d_scales, d_out, n_blocks);

    std::vector<float> gpu(cpu.size());
    check(cudaMemcpy(gpu.data(), d_out, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy back");

    long long bad = 0, first_bad = -1;
    for (size_t i = 0; i < cpu.size(); ++i) {
        // bit comparison, not a tolerance: S2's decode is exact integer-to-float arithmetic, so ANY
        // difference is a bug and a tolerance would only hide which one
        if (std::memcmp(&cpu[i], &gpu[i], sizeof(float)) != 0) {
            if (first_bad < 0) first_bad = (long long) i;
            ++bad;
        }
    }

    // NON-VACUITY.  Two buffers of zeros compare equal, and so do two buffers a broken test filled with the
    // same constant - an agreement check that cannot distinguish "both correct" from "both empty" is the
    // failure this project keeps finding.  So the data must actually vary, and the four codes must all occur:
    // an S2 decode that only ever saw one code would agree while being unable to tell -1 from +2.
    bool seen[4] = {false, false, false, false};
    for (size_t i = 0; i < codes.size(); ++i) {
        for (int t = 0; t < 4; ++t) seen[(codes[i] >> (2 * t)) & 3] = true;
    }
    int distinct = 0;
    for (bool s : seen) distinct += s ? 1 : 0;
    long long distinct_vals = 0;
    for (size_t i = 0; i < cpu.size() && distinct_vals < 8; ++i) {
        bool dup = false;
        for (size_t j = 0; j < i && !dup; ++j) dup = (cpu[j] == cpu[i]);
        if (!dup) ++distinct_vals;
    }
    std::printf("  non-vacuity: %d of 4 codes present, %lld distinct values in the first elements\n", distinct,
                distinct_vals);
    if (distinct != 4 || distinct_vals < 4) {
        std::fprintf(stderr, "VACUOUS: the comparison could not have distinguished a wrong decode\n");
        return 1;
    }

    std::printf("dequant_s2: %lld blocks, %lld elements, %lld mismatched\n", n_blocks, (long long) cpu.size(),
                bad);
    if (bad) {
        std::printf("  first mismatch at element %lld (block %lld, offset %d): cpu %.9g gpu %.9g\n",
                    (long long) first_bad, (long long) (first_bad / QK), (int) (first_bad % QK),
                    (double) cpu[(size_t) first_bad], (double) gpu[(size_t) first_bad]);
        return 1;
    }
    std::printf("  bit-exact against the scalar dequantizer that dequant_xcheck proved equal to ggml\n");
    if (selftest) std::printf("dequant_s2_parity OK\n");

    cudaFree(d_codes);
    cudaFree(d_scales);
    cudaFree(d_out);
    return 0;
}
