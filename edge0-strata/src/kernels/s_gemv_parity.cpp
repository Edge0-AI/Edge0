// src/kernels/s_gemv_parity.cpp - P2.S2's parity test for the S-family GEMV.
//
// Four source types, chosen to cover every branch the kernel has:
//
//   S2  Q2_0   2-bit, bias  -1, group 64   affine
//   S4  Q4_0   4-bit, bias  -8, group 32   affine
//   S4  IQ4_NL 4-bit, bias   0, group 32   CODEBOOK  (non-linear, no bias can express it)
//   S8  Q8_0   8-bit, bias -128, group 32  affine
//
// That is all three template instantiations, both codebooks, and four different biases.  The offset branch
// (`has_offset`, only Q4_K and Q5_K) is NOT exercised here and the test prints that fact rather than leaving a
// green result to imply otherwise - Q4_K's canonicalisation needs the packed 6-bit scale/min pairs, and the
// honest move is to say so than to half-build it.
//
// THE REFERENCE IS THE CHAIN: for each type the CPU side reconstructs the RAW GGUF block from the canonical
// planes and decodes it with that type's scalar dequantizer from include/strata/artifact/dequant.hpp - the
// ones `bench/micro/dequant_xcheck` checks against **ggml's own** dequantizers.  Reconstructing raw from
// canonical is the inverse of what the packer does, and it is written here independently of the packer so the
// two are not the same code.
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/s_gemv.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <chrono>
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

// Finite, exactly representable fp16 patterns WITH full mantissas.  Not powers of two: with only powers of
// two every product is exact and the comparison cannot round, which is how the first version of s2_gemv's test
// managed a worst error of exactly zero without proving anything.
const uint16_t kScales[] = {0x3E00, 0x3555, 0x3C01, 0x4248, 0x4123, 0x2AAA, 0x4A2B, 0x3800,
                            0xBE00, 0xB555, 0xC248, 0x2AAB, 0x4A2C, 0x2AAB, 0xB800};

struct Case {
    const char* name;
    strata::kernels::SForm form;
    int raw_bytes;        // one block
    // Reconstruct the raw GGUF block for one group from the canonical codes/scales, and return the number of
    // elements the block decodes to (which may exceed group_elems: Q8_0's block IS one group, Q2_0's too).
    int (*to_raw)(const uint8_t* codes, size_t code_off, const uint16_t* scale_bits, size_t scale_off, uint8_t* raw);
    double (*dequant)(const uint8_t* raw, float* out);
};

// ---- Q2_0: raw = { fp16 d ; uint8 qs[16] }, 64 elements, code i at byte i/4 bits (i%4)*2 - identical to the
// canonical plane, so this is a straight copy plus the scale.
int q2_0_to_raw(const uint8_t* codes, size_t co, const uint16_t* scale_bits, size_t so, uint8_t* raw) {
    const uint16_t d = scale_bits[so];
    raw[0] = (uint8_t) (d & 0xFF);
    raw[1] = (uint8_t) (d >> 8);
    // `codes` is one code PER ELEMENT (64 of them); a Q2_0 block stores them PACKED, 4 per byte, element i at
    // byte i/4 bits (i%4)*2.  The two coincide as a byte layout only if you confuse them, which is exactly the
    // mistake the first version of this function made by copying 16 bytes verbatim.
    for (int b = 0; b < 16; ++b) {
        raw[2 + b] = (uint8_t) ((codes[co + 4 * b + 0] & 3) | ((codes[co + 4 * b + 1] & 3) << 2) |
                                ((codes[co + 4 * b + 2] & 3) << 4) | ((codes[co + 4 * b + 3] & 3) << 6));
    }
    return 64;
}
double q2_0_deq(const uint8_t* raw, float* out) {
    strata::dequantize_q2_0(raw, out);
    return 64;
}

// ---- Q4_0: raw = { fp16 d ; uint8 qs[16] }, 32 elements; element j is the LOW nibble of byte j and element
// j+16 the HIGH nibble.  The canonical plane stores one code per element, so this re-packs them.
int q4_0_to_raw(const uint8_t* codes, size_t co, const uint16_t* scale_bits, size_t so, uint8_t* raw) {
    const uint16_t d = scale_bits[so];
    raw[0] = (uint8_t) (d & 0xFF);
    raw[1] = (uint8_t) (d >> 8);
    for (int j = 0; j < 16; ++j) raw[2 + j] = (uint8_t) ((codes[co + j] & 0x0F) | (codes[co + 16 + j] << 4));
    return 32;
}
double q4_0_deq(const uint8_t* raw, float* out) {
    strata::dequantize_q4_0(raw, out);
    return 32;
}

// ---- IQ4_NL: identical packing to Q4_0, different decode (a table, not a bias).
int iq4_to_raw(const uint8_t* codes, size_t co, const uint16_t* scale_bits, size_t so, uint8_t* raw) {
    return q4_0_to_raw(codes, co, scale_bits, so, raw);
}
double iq4_deq(const uint8_t* raw, float* out) {
    strata::dequantize_iq4_nl(raw, out);
    return 32;
}

// ---- Q8_0: raw = { fp16 d ; int8 qs[32] }, 32 elements.  The canonical code plane is UNSIGNED with bias
// -128 (round 154's fix), so the raw int8 byte is exactly the canonical code reinterpreted - a straight copy,
// which is why this type is the cheapest one to check and the one whose sign bug was found by the pack.
int q8_0_to_raw(const uint8_t* codes, size_t co, const uint16_t* scale_bits, size_t so, uint8_t* raw) {
    const uint16_t d = scale_bits[so];
    raw[0] = (uint8_t) (d & 0xFF);
    raw[1] = (uint8_t) (d >> 8);
    // NOT a straight copy.  The canonical code plane is UNSIGNED with bias -128, so canonical code c means the
    // value (c - 128); the raw block stores that as an `int8_t`, whose reinterpretation as an unsigned byte is
    // `c ^ 0x80` - a SIGN-BIT FLIP, not the identity.  Assuming identity is the same family of mistake as the
    // round-154 Q8_0 bug that the pack found, and it is why this reconstruction is written out rather than
    // memcpy'd.
    for (int j = 0; j < 32; ++j) raw[2 + j] = (uint8_t) (codes[co + j] ^ 0x80);
    return 32;
}
double q8_0_deq(const uint8_t* raw, float* out) {
    strata::dequantize_q8_0(raw, out);
    return 32;
}

// ---- S4/Q4_K: the ONLY canonical form with an offset, and the only one whose scale is an FP32 PRODUCT
// (`fl(d*sc)`) rather than a widened fp16.  Both facts make it the case worth having.
//
// There is no `to_raw` for it in the table above because its canonical planes are DERIVED from the raw
// parameters, not the other way round: the packer computes `scale = d*sc` and `offset = -(dmin*m)`, so this
// generates d, dmin and the eight (sc, m) pairs, derives the planes exactly as the packer does, and writes a
// raw block from the SAME parameters.  Inverting those products instead would be ambiguous.
void test_q4k(long long n_in, long long n_out, double tol, int* total_bad) {
    using strata::kernels::Codebook;
    using strata::kernels::SForm;
    const SForm form{4, 0, 32, Codebook::Affine, true};

    std::mt19937 rng(777);
    const int n_scales = (int) (sizeof(kScales) / sizeof(kScales[0]));
    const long long n_groups = n_in / 32;
    const long long codes_per_row = n_in / 2;

    std::vector<uint16_t> x((size_t) n_in);
    for (long long i = 0; i < n_in; ++i) x[(size_t) i] = kScales[rng() % n_scales];

    std::vector<uint8_t> codes_el((size_t) n_out * n_in);
    for (size_t i = 0; i < codes_el.size(); ++i) codes_el[i] = (uint8_t) (rng() & 0x0F);
    std::vector<uint8_t> codes((size_t) n_out * codes_per_row, 0);
    for (size_t i = 0; i < codes_el.size(); ++i) {
        const size_t o = i / (size_t) n_in, k = i % (size_t) n_in;
        codes[o * (size_t) codes_per_row + k / 2] |= (uint8_t) (codes_el[i] << ((k % 2) * 4));
    }
    std::vector<float> scales((size_t) n_out * n_groups), offsets((size_t) n_out * n_groups);

    struct Row {
        uint16_t d, dmin;
        uint8_t sc[8], m[8];
    };
    std::vector<Row> rows((size_t) n_out);
    for (long long o = 0; o < n_out; ++o) {
        Row& r = rows[(size_t) o];
        r.d = kScales[rng() % n_scales];
        r.dmin = kScales[rng() % n_scales];
        for (int g = 0; g < 8; ++g) {
            r.sc[g] = (uint8_t) (rng() & 0x3F);
            r.m[g] = (uint8_t) (rng() & 0x3F);
            for (long long sb = 0; sb < n_in / 256; ++sb) {
                scales[(size_t) (o * n_groups + sb * 8 + g)] = strata::fp16_to_fp32(r.d) * (float) r.sc[g];
                offsets[(size_t) (o * n_groups + sb * 8 + g)] = -(strata::fp16_to_fp32(r.dmin) * (float) r.m[g]);
            }
        }
    }

    std::vector<float> ref((size_t) n_out, 0.0f);
    std::vector<double> cond((size_t) n_out, 0.0);
    double worst_plane_ref = 0.0;   // max |planes - raw block| over every element, on the CPU alone
    std::vector<uint8_t> blk(144);
    std::vector<float> dec(256);
    // A Q4_K raw block covers 256 elements, so a 2560-element row is TEN superblocks, each with its own
    // 144-byte block.  The first version of this loop ran `step < 4` - one superblock, 256 of the row's 2560
    // elements - while the kernel accumulated all of them, and the resulting 29x relative error was the
    // reference being short, not the kernel being wrong.
    for (long long o = 0; o < n_out; ++o) {
        const Row& r = rows[(size_t) o];
        float acc = 0.0f;
        double sum_abs = 0.0;      // the natural scale of this row's dot product (see the metric below)
        double plane_ref_diff = 0.0;   // (a) vs (b): planes against the raw block, both on the CPU
        for (long long sb = 0; sb < n_in / 256; ++sb) {
            // ONE 144-byte block per 256-element SUPERBLOCK, with all four steps' qs bytes filled BEFORE the
            // decode.  The previous version decoded a fresh block per STEP and then read all 256 elements
            // from each, so three quarters of every decode came from zeroed qs - four decodes of a
            // quarter-filled block per superblock, which is the whole of the 1.17e4 disagreement.
            // `scripts/probe_q4k_groups.cpp` builds it this way and agrees exactly, group by group.
            blk.assign(144, 0);
            blk[0] = (uint8_t) (r.d & 0xFF);
            blk[1] = (uint8_t) (r.d >> 8);
            blk[2] = (uint8_t) (r.dmin & 0xFF);
            blk[3] = (uint8_t) (r.dmin >> 8);
            for (int j = 0; j < 4; ++j) {          // the exact inverse of get_scale_min_k4
                blk[4 + j] = (uint8_t) ((r.sc[j] & 0x3F) | ((r.sc[j + 4] >> 4) << 6));
                blk[8 + j] = (uint8_t) ((r.m[j] & 0x3F) | ((r.m[j + 4] >> 4) << 6));
                blk[12 + j] = (uint8_t) ((r.sc[j + 4] & 0x0F) | ((r.m[j + 4] & 0x0F) << 4));
            }
            for (int step = 0; step < 4; ++step) {
                for (int l = 0; l < 32; ++l) {     // elements 0..31 LOW nibbles, 32..63 HIGH
                    const long long e0 = o * n_in + sb * 256 + step * 64 + l;
                    const long long e1 = e0 + 32;
                    blk[16 + step * 32 + l] =
                        (uint8_t) ((codes_el[(size_t) e0] & 0x0F) | (codes_el[(size_t) e1] << 4));
                }
            }
            strata::dequantize_q4_K(blk.data(), dec.data());
            for (int j = 0; j < 256; ++j) {
                const float term = dec[(size_t) j] * strata::fp16_to_fp32(x[(size_t) (sb * 256 + j)]);
                acc += term;
                sum_abs += std::fabs((double) term);      // the CONDITION of this dot product

                // THE THIRD DECODE.  (a) the raw block through the scalar dequantizer, (b) the canonical
                // PLANES through `code*scale + offset`, (c) the GPU.  If (a) != (b) then this test built
                // planes that disagree with its own raw block and the kernel is innocent; if (a) == (b) and
                // (c) differs, the kernel is wrong.  Without (b) there is no way to tell which.
                const long long elem = sb * 256 + j;
                const long long g = elem / 32;
                const int code = (int) codes_el[(size_t) (o * n_in + elem)];
                const float b_dec = (float) code * scales[(size_t) (o * n_groups + g)] +
                                    offsets[(size_t) (o * n_groups + g)];
                if (std::fabs((double) (b_dec - dec[(size_t) j])) > plane_ref_diff) {
                    plane_ref_diff = std::fabs((double) (b_dec - dec[(size_t) j]));
                }
            }
        }
        ref[(size_t) o] = acc;
        cond[(size_t) o] = sum_abs;
        if (!(plane_ref_diff <= worst_plane_ref)) worst_plane_ref = plane_ref_diff;
    }

    uint16_t* d_x = nullptr;
    uint8_t* d_codes = nullptr;
    float *d_scales = nullptr, *d_offsets = nullptr, *d_y = nullptr;
    check(cudaMalloc(&d_x, x.size() * sizeof(uint16_t)), "cudaMalloc x");
    check(cudaMalloc(&d_codes, codes.size()), "cudaMalloc codes");
    check(cudaMalloc(&d_scales, scales.size() * sizeof(float)), "cudaMalloc scales");
    check(cudaMalloc(&d_offsets, offsets.size() * sizeof(float)), "cudaMalloc offsets");
    check(cudaMalloc(&d_y, (size_t) n_out * sizeof(float)), "cudaMalloc y");
    check(cudaMemcpy(d_x, x.data(), x.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "copy x");
    check(cudaMemcpy(d_codes, codes.data(), codes.size(), cudaMemcpyHostToDevice), "copy codes");
    check(cudaMemcpy(d_scales, scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice),
          "copy scales");
    check(cudaMemcpy(d_offsets, offsets.data(), offsets.size() * sizeof(float), cudaMemcpyHostToDevice),
          "copy offsets");
    strata::kernels::s_gemv(d_x, d_codes, d_scales, d_offsets, d_y, n_in, n_out, form);
    std::vector<float> got((size_t) n_out);
    check(cudaMemcpy(got.data(), d_y, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy back");

    // THE ERROR IS MEASURED AGAINST sum|term|, NOT AGAINST THE RESULT.  A dot product of 2560 signed terms
    // cancels, so `|ref-got| / |ref|` is a statement about the condition number, not about the kernel: a row
    // whose terms are ~1e7 and whose result is ~1e3 reports a relative error of 50 for an absolute error of
    // 1e-6 of the terms.  `sum_abs` is the natural scale of the computation, and comparing to it is what
    // makes a tolerance mean "agrees to within rounding" instead of "did not cancel".
    long long bad = 0;
    double worst = 0.0, worst_res = 0.0;
    for (long long o = 0; o < n_out; ++o) {
        const double a = ref[(size_t) o], b = got[(size_t) o];
        const double scale = cond[(size_t) o] > 1e-30 ? cond[(size_t) o] : 1e-30;
        const double rel = std::fabs(a - b) / scale;
        const double rel_res = std::fabs(a - b) / (std::fabs(a) > 1e-30 ? std::fabs(a) : 1e-30);
        if (!(rel <= worst)) worst = rel;
        if (!(rel_res <= worst_res)) worst_res = rel_res;
        if (!(rel <= tol)) ++bad;
    }
    double lo = ref[0], hi = ref[0];
    for (float v : ref) {
        lo = std::fmin(lo, v);
        hi = std::fmax(hi, v);
    }
    std::printf("  %-10s %lld rows, %lld over tol, worst rel-to-terms %.3e (rel-to-result %.3e)\n",
                "S4/Q4_K", n_out, bad, worst, worst_res);
    std::printf("             ref spread [%.4g, %.4g], condition (sum|term|) up to %.4g\n", lo, hi,
                *std::max_element(cond.begin(), cond.end()));
    std::printf("             CPU planes vs CPU raw block: max |diff| = %.3e -> %s\n", worst_plane_ref,
                worst_plane_ref == 0.0
                    ? "the test is SELF-CONSISTENT"
                    : "*** THE TEST DISAGREES WITH ITSELF - fix the reference before believing any GPU result ***");
    if (hi - lo <= 0.0) {
        std::fprintf(stderr, "VACUOUS: S4/Q4_K reference is constant\n");
        std::exit(1);
    }
    *total_bad += (int) bad;
    cudaFree(d_x);
    cudaFree(d_codes);
    cudaFree(d_scales);
    cudaFree(d_offsets);
    cudaFree(d_y);
}

}  // namespace


// ------------------------------------------------------------------ benchmark
//
// P2.S10 wants a first measurement and a ledger entry, and the objective is stated in tokens per second, so
// the useful output is not GiB/s but "milliseconds of expert matvec per token".  Warm-up runs are discarded
// (the first launch pays module load and page faults) and the MINIMUM is reported alongside the mean, because
// the minimum is the one not polluted by another process on the machine.
void bench_s2_gemv(long long n_in, long long n_out, int iters, int* split_bad) {
    using strata::kernels::Codebook;
    using strata::kernels::SForm;
    const SForm form{2, -1, 64, Codebook::Affine, false};
    const long long n_groups = n_in / 64;
    const long long codes_per_row = n_in / 4;

    std::vector<uint8_t> codes((size_t) n_out * codes_per_row, 0xA5);
    std::vector<float> scales((size_t) n_out * n_groups, 0.001f);
    std::vector<uint16_t> x((size_t) n_in, 0x3C00);            // 1.0 in fp16

    uint16_t* d_x = nullptr;
    uint8_t* d_codes = nullptr;
    float *d_scales = nullptr, *d_y = nullptr;
    check(cudaMalloc(&d_x, x.size() * sizeof(uint16_t)), "bench x");
    check(cudaMalloc(&d_codes, codes.size()), "bench codes");
    check(cudaMalloc(&d_scales, scales.size() * sizeof(float)), "bench scales");
    check(cudaMalloc(&d_y, (size_t) n_out * sizeof(float)), "bench y");
    check(cudaMemcpy(d_x, x.data(), x.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "bench copy x");
    check(cudaMemcpy(d_codes, codes.data(), codes.size(), cudaMemcpyHostToDevice), "bench copy codes");
    check(cudaMemcpy(d_scales, scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice),
          "bench copy scales");

    for (int i = 0; i < 3; ++i) strata::kernels::s_gemv(d_x, d_codes, d_scales, nullptr, d_y, n_in, n_out, form);

    std::vector<double> ms;
    for (int i = 0; i < iters; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        strata::kernels::s_gemv(d_x, d_codes, d_scales, nullptr, d_y, n_in, n_out, form);
        const auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    double mean = 0;
    for (double v : ms) mean += v;
    mean /= (double) ms.size();

    const double weights = (double) n_in * (double) n_out;
    const double bytes = (double) n_out * (double) codes_per_row + (double) n_out * n_groups * 4.0;
    std::printf("s2_gemv bench  [%lld x %lld]  %d iters\n", n_in, n_out, iters);
    std::printf("  ms per gemv: min %.4f  median %.4f  mean %.4f\n", ms.front(), ms[ms.size() / 2], mean);
    std::printf("  min -> %.1f G weights/s, %.1f GiB/s of S2 streamed\n", weights / ms.front() / 1e6,
                bytes / (1024.0 * 1024 * 1024) / (ms.front() / 1000.0));

    // The projection the objective is stated against: 48 layers x 10 experts x 3 roles, all of one token's
    // experts, at the measured per-role rate.  This is the VRAM-resident path only - the CPU expert path and
    // the KV/attention work are separate and are NOT included.
    const double roles_per_token = 48.0 * 10.0 * 3.0;
    const double tok_ms = roles_per_token * ms.front();
    std::printf("  per token: %.0f roles x %.4f ms = %.1f ms  ->  %.1f tok/s (expert matvec only)\n",
                roles_per_token, ms.front(), tok_ms, tok_ms > 0 ? 1000.0 / tok_ms : 0.0);
    std::printf("  NOTE: expert matvec ONLY. No attention, no KV, no norm, no router, no sampling, and it\n"
                "        assumes every expert is already resident in VRAM at zero cost.\n");

    // The NAIVE kernel's output is the reference for the split one.  `s_gemv` is checked against the scalar
    // dequantizer by this file's parity cases, so its result here is a validated value, and the split kernel
    // sums in a different order - so this is a relative comparison, not bit equality.
    strata::kernels::s_gemv(d_x, d_codes, d_scales, nullptr, d_y, n_in, n_out, form);
    std::vector<float> ref_naive((size_t) n_out);
    check(cudaMemcpy(ref_naive.data(), d_y, ref_naive.size() * sizeof(float), cudaMemcpyDeviceToHost),
          "copy naive reference");

    // ---- the row-split variant, same planes, same process -------------------
    // Swept rather than guessed: more threads per row buys parallelism and costs a deeper reduction.
    for (int tpr : {32, 64, 128, 256}) {
        if (tpr > n_in) continue;
        for (int i = 0; i < 3; ++i) {
            strata::kernels::s_gemv_split(d_x, d_codes, d_scales, nullptr, d_y, n_in, n_out, form, tpr);
        }
        std::vector<double> sms;
        for (int i = 0; i < iters; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            strata::kernels::s_gemv_split(d_x, d_codes, d_scales, nullptr, d_y, n_in, n_out, form, tpr);
            sms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                              .count());
        }
        std::sort(sms.begin(), sms.end());

        // correctness of the thing being timed
        std::vector<float> got((size_t) n_out);
        check(cudaMemcpy(got.data(), d_y, got.size() * sizeof(float), cudaMemcpyDeviceToHost),
              "copy split result");
        long long bad = 0;
        double worst = 0.0;
        for (long long o = 0; o < n_out; ++o) {
            const double a = ref_naive[(size_t) o], b = got[(size_t) o];
            const double rel = std::fabs(a - b) / (std::fabs(a) > 1e-30 ? std::fabs(a) : 1e-30);
            if (!(rel <= worst)) worst = rel;
            if (!(rel <= 1e-4)) ++bad;
        }
        if (bad) {
            std::printf("  split tpr=%-4d *** WRONG *** %lld of %lld rows differ, worst rel %.3e\n", tpr,
                        bad, n_out, worst);
            *split_bad += (int) bad;
        }

        // ---- the quad-amortised S2 kernel, timed AND checked against the naive reference ----
        // Only meaningful for S2: it assumes four elements share one code byte, which is true of 2-bit codes.
        if (form.code_bits == 2 && (n_in % 64) == 0) {
            for (int t2 : {32, 64, 128}) {
                for (int i = 0; i < 3; ++i) {
                    strata::kernels::s2_gemv_quads(d_x, d_codes, d_scales, d_y, n_in, n_out, t2);
                }
                std::vector<double> qms;
                for (int i = 0; i < iters; ++i) {
                    const auto q0 = std::chrono::steady_clock::now();
                    strata::kernels::s2_gemv_quads(d_x, d_codes, d_scales, d_y, n_in, n_out, t2);
                    qms.push_back(std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - q0).count());
                }
                std::sort(qms.begin(), qms.end());
                check(cudaMemcpy(got.data(), d_y, got.size() * sizeof(float), cudaMemcpyDeviceToHost),
                      "copy quads result");
                long long qbad = 0;
                double qworst = 0.0;
                for (long long o = 0; o < n_out; ++o) {
                    const double a = ref_naive[(size_t) o], b = got[(size_t) o];
                    const double rel = std::fabs(a - b) / (std::fabs(a) > 1e-30 ? std::fabs(a) : 1e-30);
                    if (rel > qworst) qworst = rel;
                    if (!(rel <= 1e-4)) ++qbad;
                }
                if (qbad) *split_bad += (int) qbad;
                std::printf("  quads tpr=%-4d min %.4f ms  -> %6.1f G weights/s   vs naive %.2fx   %s\n", t2,
                            qms.front(), weights / qms.front() / 1e6, ms.front() / qms.front(),
                            qbad ? "*** WRONG ***" : "checked ok");
            }
        }

        // ---- the two-lever kernel, both configurations, timed AND checked ----
        if (form.code_bits == 2 && (n_in % 64) == 0) {
            for (int staged = 0; staged < 2; ++staged) {
                for (int t3 : {32, 64, 128}) {
                    for (int i = 0; i < 3; ++i) {
                        strata::kernels::s2_gemv_fast(d_x, d_codes, d_scales, d_y, n_in, n_out, t3, staged != 0);
                    }
                    std::vector<double> fms;
                    for (int i = 0; i < iters; ++i) {
                        const auto q0 = std::chrono::steady_clock::now();
                        strata::kernels::s2_gemv_fast(d_x, d_codes, d_scales, d_y, n_in, n_out, t3, staged != 0);
                        fms.push_back(std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() - q0).count());
                    }
                    std::sort(fms.begin(), fms.end());
                    check(cudaMemcpy(got.data(), d_y, got.size() * sizeof(float), cudaMemcpyDeviceToHost),
                          "copy fast result");
                    long long fbad = 0;
                    for (long long o = 0; o < n_out; ++o) {
                        const double a = ref_naive[(size_t) o], b = got[(size_t) o];
                        const double rel = std::fabs(a - b) / (std::fabs(a) > 1e-30 ? std::fabs(a) : 1e-30);
                        if (!(rel <= 1e-4)) ++fbad;
                    }
                    if (fbad) *split_bad += (int) fbad;
                    std::printf("  fast %s tpr=%-4d min %.4f ms  -> %6.1f G weights/s   vs naive %.2fx   %s\n",
                                staged ? "staged" : "global", t3, fms.front(),
                                weights / fms.front() / 1e6, ms.front() / fms.front(),
                                fbad ? "*** WRONG ***" : "checked ok");
                }
            }
        }

        const double sp = ms.front() / sms.front();
        std::printf("  split tpr=%-4d min %.4f ms  -> %6.1f G weights/s   speedup vs naive %.2fx\n", tpr,
                    sms.front(), weights / sms.front() / 1e6, sp);
    }

    cudaFree(d_x);
    cudaFree(d_codes);
    cudaFree(d_scales);
    cudaFree(d_y);
}

int main(int argc, char** argv) {
    long long n_in = 2560;
    long long n_out = 128;
    // 1e-4, not 1e-5: the measured worst is ~1e-5 (FMA contraction of a 2560-term f32 accumulation), so a
    // 1e-5 threshold would pass by 4% and fail on a different n_in.  The phase spec allows 1e-3 for FP16
    // paths; this is 10x tighter than the spec and 10x looser than the measurement.
    double tol = 1e-4;
    bool selftest = false;
    bool bench = false;
    int iters = 200;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") selftest = true;
        else if (a == "--bench") bench = true;
        else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (a == "--n-in" && i + 1 < argc) n_in = std::atoll(argv[++i]);
        else if (a == "--n-out" && i + 1 < argc) n_out = std::atoll(argv[++i]);
        else {
            std::fprintf(stderr, "usage: s_gemv_parity [--selftest] [--n-in N] [--n-out N]\n");
            return 2;
        }
    }

    using strata::kernels::Codebook;
    using strata::kernels::SForm;
    std::vector<Case> cases;
    cases.push_back({"S2/Q2_0", SForm{2, -1, 64, Codebook::Affine, false}, 18, q2_0_to_raw, q2_0_deq});
    cases.push_back({"S4/Q4_0", SForm{4, -8, 32, Codebook::Affine, false}, 18, q4_0_to_raw, q4_0_deq});
    cases.push_back({"S4/IQ4_NL", SForm{4, 0, 32, Codebook::Iq4Nl, false}, 18, iq4_to_raw, iq4_deq});
    cases.push_back({"S8/Q8_0", SForm{8, -128, 32, Codebook::Affine, false}, 34, q8_0_to_raw, q8_0_deq});

    std::mt19937 rng(4242);
    const int n_scales = (int) (sizeof(kScales) / sizeof(kScales[0]));
    int total_bad = 0;

    for (const Case& tc : cases) {
        const int bits = tc.form.code_bits;
        const int per_byte = 8 / bits;
        const int G = tc.form.group_elems;
        const long long n_groups = n_in / G;
        const long long codes_per_row = n_in / per_byte;

        // activations
        std::vector<uint16_t> x((size_t) n_in);
        for (long long i = 0; i < n_in; ++i) x[(size_t) i] = kScales[rng() % n_scales];

        // Codes are generated ONE PER ELEMENT and then packed into the plane the kernel reads.
        //
        // The first version of this test generated a random BYTE ARRAY and used it as both, which is the bug
        // that made S2 pass and the other three fail by 50x-500x.  The two are not the same thing: the kernel
        // reads element i from byte i/per_byte at bit (i%per_byte)*bits, while the reference needs one code per
        // element to rebuild a raw GGUF block.  For S2 (per_byte = 4, which IS Q2_0's own raw layout) treating
        // one random array as both happened to agree - and that coincidence is exactly what a test must not
        // depend on.
        std::vector<uint8_t> codes_el((size_t) n_out * n_in);
        for (size_t i = 0; i < codes_el.size(); ++i) codes_el[i] = (uint8_t) (rng() & ((1u << bits) - 1));
        std::vector<uint8_t> codes((size_t) n_out * codes_per_row, 0);
        for (long long o = 0; o < n_out; ++o) {
            for (long long i = 0; i < n_in; ++i) {
                const long long byi = o * codes_per_row + i / per_byte;
                codes[(size_t) byi] |= (uint8_t) (codes_el[(size_t) (o * n_in + i)]
                                                  << ((i % per_byte) * bits));
            }
        }
        std::vector<float> scales((size_t) n_out * n_groups);
        std::vector<uint16_t> scale_bits((size_t) n_out * n_groups);
        for (size_t i = 0; i < scales.size(); ++i) {
            scale_bits[i] = kScales[rng() % n_scales];
            scales[i] = strata::fp16_to_fp32(scale_bits[i]);
        }

        // CPU reference
        std::vector<float> ref((size_t) n_out, 0.0f);
        std::vector<uint8_t> raw((size_t) tc.raw_bytes);
        std::vector<float> dec((size_t) G);
        for (long long o = 0; o < n_out; ++o) {
            float acc = 0.0f;
            for (long long g = 0; g < n_groups; ++g) {
                const size_t co = (size_t) o * n_in + (size_t) g * G;   // the UNPACKED code array
                const size_t so = (size_t) o * n_groups + (size_t) g;
                // The raw block carries the scale as fp16 and the canonical plane carries it widened, so the
                // block is rebuilt from the PATTERN the plane came from.  No float-to-fp16 conversion is
                // needed anywhere, and the decoder therefore sees bit-identically what the kernel reads.
                const int n = tc.to_raw(codes_el.data(), co, scale_bits.data(), so, raw.data());
                tc.dequant(raw.data(), dec.data());
                for (long long j = 0; j < (long long) n; ++j) {
                    acc += dec[(size_t) j] * strata::fp16_to_fp32(x[(size_t) (g * G + j)]);
                }
            }
            ref[(size_t) o] = acc;
        }

        // GPU
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
        strata::kernels::s_gemv(d_x, d_codes, d_scales, nullptr, d_y, n_in, n_out, tc.form);
        std::vector<float> got((size_t) n_out);
        check(cudaMemcpy(got.data(), d_y, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy back");

        long long bad = 0;
        double worst = 0.0;
        for (long long o = 0; o < n_out; ++o) {
            const double a = ref[(size_t) o], b = got[(size_t) o];
            const double rel = std::fabs(a - b) / (std::fabs(a) > 1e-30 ? std::fabs(a) : 1e-30);
            if (!(rel <= worst)) worst = rel;
            if (!(rel <= tol)) ++bad;
        }
        double lo = ref[0], hi = ref[0];
        for (float v : ref) {
            lo = std::fmin(lo, v);
            hi = std::fmax(hi, v);
        }
        std::printf("  %-10s %lld rows, %lld over tol, worst rel %.3e   ref spread [%.4g, %.4g]\n", tc.name,
                    n_out, bad, worst, lo, hi);
        if (hi - lo <= 0.0) {
            std::fprintf(stderr, "VACUOUS: %s reference is constant\n", tc.name);
            return 1;
        }
        total_bad += (int) bad;

        cudaFree(d_x);
        cudaFree(d_codes);
        cudaFree(d_scales);
        cudaFree(d_y);
    }

    std::printf("s_gemv: %zu type cases, %d rows over tolerance (tol %.1e)\n", cases.size(), total_bad, tol);
    // Q4_K IS NOT PART OF THE DEFAULT RUN.  It FAILS: 128 of 128 rows, worst error 0.586 of sum|term| - a
    // real difference, not cancellation (the condition number is 2.8e7 and the error is 1.6e7).  Either the
    // kernel's offset handling or this test's reconstruction of a Q4_K block is wrong and the two have not
    // been separated yet.  Running it by default would turn ctest red and hide regressions in the four cases
    // that ARE verified; leaving it out silently would be worse.  It is one flag away, and it is recorded as
    // OPEN in the project state.
    // Q4_K is IN the default run as of round 171.  Rounds 169-170 gated it because it FAILED, and the
    // failure was this test's own reconstruction of a Q4_K block - four decodes of a quarter-filled block per
    // superblock - not the kernel.  It now agrees to 2.8e-07 of sum|term|, so the gate goes rather than
    // staying as a habit: a gate that outlives its reason is a check that quietly stopped running.
    test_q4k(n_in, n_out, tol, &total_bad);
    std::printf("s_gemv total: %d rows over tolerance\n", total_bad);
    if (bench) {
        // the real expert shapes: gate/up are [2560 x 640], down is [640 x 2560]
        int split_bad = 0;
        bench_s2_gemv(2560, 640, iters, &split_bad);
        bench_s2_gemv(640, 2560, iters, &split_bad);
        std::printf("split kernel rows disagreeing with the naive reference: %d\n", split_bad);
        // accumulated HERE, inside the block that declares it, and it makes the binary FAIL: a kernel that is
        // benchmarked but never compared is an unverified kernel being quoted for performance.
        total_bad += split_bad;
    }
    if (total_bad) return 1;
    if (selftest) std::printf("s_gemv_parity OK\n");
    return 0;
}
