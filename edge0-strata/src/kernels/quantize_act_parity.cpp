// src/kernels/quantize_act_parity.cpp - P2.S2's parity test for the Q8_0 activation quantizer.
//
// THE CHECK IS ON THE BYTES, not on the round-tripped values.  `ref/quant.py::q8_0`'s central finding is that
// dividing by the fp16-rounded scale instead of the fp32 one changed 18 of 80 blocks' quantized integers
// while leaving the dequantized values within a rounding step - so a comparison of VALUES would have passed
// and the engine would still have disagreed with ggml.  Comparing the 34-byte block catches that directly.
//
// The reference is a host transcription of `ref/quant.py::q8_0`:
//
//     d32 = amax / 127.0f                       (f32)
//     d16 = fp16(d32)                           (stored, and used to DEQUANTIZE)
//     q   = clip(rint(x / (double)d32), -128, 127)
//     round trip = q * d16
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// THE CONVERSIONS COME FROM `f16_bits.hpp`, and this test used to carry its own.  The private `f16_bits_to_f32`
// below flushed fp16 SUBNORMALS to zero, and the private encoder returned NaN for finite fp16 overflow - both
// invisible with the O(1) fixture this file has.  Round 198 validated the shared pair against 395 numpy
// float16 vectors and all 65,536 reverse patterns, so the test now shares the implementation it is checking
// instead of guessing at it independently.  Thin aliases keep the body below unchanged.
using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_f16;
inline uint16_t f32_to_f16_bits(float f) { return f16_from_f32(f); }
inline float f16_bits_to_f32(uint16_t h) { return f32_from_f16(h); }

void reference_q8_0(const float* x, long long n, std::vector<uint8_t>& blocks, std::vector<float>& back) {
    blocks.assign((size_t) (n / 32) * 34, 0);
    back.assign((size_t) n, 0.0f);
    for (long long b = 0; b < n / 32; ++b) {
        const float* xb = x + b * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(xb[i]));
        uint8_t* out = &blocks[(size_t) b * 34];
        if (amax == 0.0f) continue;
        const float d32 = amax / 127.0f;
        const uint16_t dbits = f32_to_f16_bits(d32);
        out[0] = (uint8_t) (dbits & 0xFF);
        out[1] = (uint8_t) (dbits >> 8);
        const float d16 = f16_bits_to_f32(dbits);
        for (int i = 0; i < 32; ++i) {
            double q = std::rint((double) xb[i] / (double) d32);
            if (q > 127.0) q = 127.0;
            if (q < -128.0) q = -128.0;
            out[2 + i] = (uint8_t) (int8_t) q;
            back[(size_t) (b * 32 + i)] = (float) (int8_t) out[2 + i] * d16;
        }
    }
}

int run_case(const char* name, const std::vector<float>& x, bool check_bytes) {
    const long long n = (long long) x.size();
    std::vector<uint8_t> r_blocks, g_blocks((size_t) (n / 32) * 34);
    std::vector<float> r_back, g_back((size_t) n);
    reference_q8_0(x.data(), n, r_blocks, r_back);

    float* d_x = nullptr;
    uint8_t* d_b = nullptr;
    float* d_back = nullptr;
    check(cudaMalloc(&d_x, (size_t) n * sizeof(float)), "malloc x");
    check(cudaMalloc(&d_b, g_blocks.size()), "malloc blocks");
    check(cudaMalloc(&d_back, (size_t) n * sizeof(float)), "malloc back");
    check(cudaMemcpy(d_x, x.data(), (size_t) n * sizeof(float), cudaMemcpyHostToDevice), "copy x");
    // PROBE: does the DEVICE hold what the host thinks it sent?  The scale byte differs with the SAME
    // exponent and a zeroed mantissa, which says the kernel's amax differs from the host's on identical
    // input - so the first thing to rule out is the copy itself.
    {
        std::vector<float> rt((size_t) n);
        check(cudaMemcpy(rt.data(), d_x, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost), "probe copy back");
        long long diff = 0;
        for (long long i = 0; i < n; ++i) if (std::memcmp(&rt[(size_t) i], &x[(size_t) i], 4) != 0) ++diff;
        if (diff) std::printf("    PROBE: device x differs from host x in %lld of %lld elements\n", diff, n);
    }
    strata::kernels::quantize_q8_0(d_x, d_b, n, nullptr);
    strata::kernels::dequant_q8_0(d_b, d_back, n, nullptr);
    check(cudaMemcpy(g_blocks.data(), d_b, g_blocks.size(), cudaMemcpyDeviceToHost), "back blocks");
    check(cudaMemcpy(g_back.data(), d_back, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost), "back vals");

    long long byte_bad = 0, val_bad = 0;
    if (check_bytes) {
        for (size_t i = 0; i < g_blocks.size(); ++i) if (g_blocks[i] != r_blocks[i]) ++byte_bad;
    }
    for (long long i = 0; i < n; ++i) {
        const float a = r_back[(size_t) i], b = g_back[(size_t) i];
        // the round trip is `q * d16`: both sides use the SAME fp16 d, so this is exact when q agrees
        if (std::memcmp(&a, &b, sizeof(float)) != 0) ++val_bad;
    }
    if (byte_bad && check_bytes) {
        for (size_t i = 0; i < g_blocks.size(); ++i) {
            if (g_blocks[i] != r_blocks[i]) {
                const size_t blk = i / 34;
                const float amax = [&] { float m = 0; for (int k = 0; k < 32; ++k) m = std::fmax(m, std::fabs(x[(size_t)(blk*32+k)])); return m; }();
                std::printf("    first bad byte: block %zu offset %zu  amax %.9g  d32 %.9g\n", blk, i % 34, amax, amax/127.0f);
                std::printf("      ref scale bytes %02x %02x   got %02x %02x\n", r_blocks[blk*34], r_blocks[blk*34+1], g_blocks[blk*34], g_blocks[blk*34+1]);
                std::printf("      ref[0..9]:"); for (int k = 0; k < 10; ++k) std::printf(" %02x", r_blocks[blk*34+k]);
                std::printf("\n      got[0..9]:"); for (int k = 0; k < 10; ++k) std::printf(" %02x", g_blocks[blk*34+k]);
                std::printf("\n      ref[10..17]:"); for (int k = 10; k < 18; ++k) std::printf(" %02x", r_blocks[blk*34+k]);
                std::printf("\n      got[10..17]:"); for (int k = 10; k < 18; ++k) std::printf(" %02x", g_blocks[blk*34+k]);
                std::printf("\n      x[0..3] = %.9g %.9g %.9g %.9g\n", (double)x[(size_t)(blk*32)], (double)x[(size_t)(blk*32+1)], (double)x[(size_t)(blk*32+2)], (double)x[(size_t)(blk*32+3)]);
                break;
            }
        }
    }
    std::printf("  %-26s blocks %s (%lld bad bytes)   round trip %s (%lld differ)\n", name,
                !check_bytes ? "not compared" : (byte_bad ? "*** WRONG ***" : "byte-exact"), byte_bad,
                val_bad ? "*** WRONG ***" : "bit-exact", val_bad);
    cudaFree(d_x);
    cudaFree(d_b);
    cudaFree(d_back);
    return (int) (byte_bad + val_bad);
}

// A host transcription of ggml's `quantize_row_q8_K_ref` (ggml-quants.c L2768), including `nearest_int`.
//
// NOTE WHICH REFERENCE THIS IS.  `ref/quant.py::q8_K` is the numpy transcription and it multiplies in
// FLOAT64 (`iscale * blk.astype(np.float64)`), while ggml multiplies in FLOAT32 before `nearest_int`.  The
// oracle is ggml, so this follows ggml.  The two agree except where an f32 product lands exactly on a tie or
// a rounding boundary, which is exactly the regime the "exact ties" case below constructs.
int nearest_int_host(float fval) {
    const float val = fval + 12582912.0f;
    int i;
    std::memcpy(&i, &val, 4);
    return (i & 0x007fffff) - 0x00400000;
}

void reference_q8_K(const float* x, long long n, std::vector<uint8_t>& blocks, std::vector<float>& back,
                    float scale_num) {
    const int QK = 256;
    blocks.assign((size_t) (n / QK) * 292, 0);
    back.assign((size_t) n, 0.0f);
    for (long long b = 0; b < n / QK; ++b) {
        const float* xb = x + b * QK;
        uint8_t* out = &blocks[(size_t) b * 292];
        float max = 0.0f, amax = 0.0f;
        for (int j = 0; j < QK; ++j) {
            const float ax = std::fabs(xb[j]);
            if (ax > amax) { amax = ax; max = xb[j]; }     // STRICTLY greater: ties keep the FIRST maximum
        }
        if (amax == 0.0f) continue;                        // d = 0, qs = 0, bsums = 0 (see the kernel note)
        const float iscale = scale_num / max;              // -127 by default; -128 is the rival reading
        int8_t* qs = (int8_t*) (out + 4);
        for (int j = 0; j < QK; ++j) {
            const int v = nearest_int_host(iscale * xb[j]);
            qs[j] = (int8_t) std::min(127, v);
        }
        int16_t* bsums = (int16_t*) (out + 4 + QK);
        for (int j = 0; j < QK / 16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) sum += qs[j * 16 + ii];
            bsums[j] = (int16_t) sum;
        }
        const float d = 1.0f / iscale;
        std::memcpy(out, &d, 4);
        for (int j = 0; j < QK; ++j) back[(size_t) (b * QK + j)] = (float) qs[j] * d;
    }
}

int run_case_k(const char* name, const std::vector<float>& x, bool check_bytes, float scale_num = -127.0f) {
    const long long n = (long long) x.size();
    std::vector<uint8_t> r_blocks, g_blocks((size_t) (n / 256) * 292);
    std::vector<float> r_back, g_back((size_t) n);
    reference_q8_K(x.data(), n, r_blocks, r_back, scale_num);

    float* d_x = nullptr;
    uint8_t* d_b = nullptr;
    float* d_back = nullptr;
    check(cudaMalloc(&d_x, (size_t) n * sizeof(float)), "malloc x");
    check(cudaMalloc(&d_b, g_blocks.size()), "malloc blocks");
    check(cudaMalloc(&d_back, (size_t) n * sizeof(float)), "malloc back");
    check(cudaMemcpy(d_x, x.data(), (size_t) n * sizeof(float), cudaMemcpyHostToDevice), "copy x");
    strata::kernels::quantize_q8_K(d_x, d_b, n, nullptr);
    strata::kernels::dequant_q8_K(d_b, d_back, n, nullptr);
    check(cudaMemcpy(g_blocks.data(), d_b, g_blocks.size(), cudaMemcpyDeviceToHost), "back blocks");
    check(cudaMemcpy(g_back.data(), d_back, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost), "back vals");

    long long byte_bad = 0, val_bad = 0, first_bad = -1;
    if (check_bytes)
        for (size_t i = 0; i < g_blocks.size(); ++i)
            if (g_blocks[i] != r_blocks[i]) { if (first_bad < 0) first_bad = (long long) i; ++byte_bad; }
    for (long long i = 0; i < n; ++i)
        if (std::memcmp(&r_back[(size_t) i], &g_back[(size_t) i], 4) != 0) ++val_bad;
    std::printf("  %-26s blocks %s (%lld bad bytes)   round trip %s (%lld differ)\n", name,
                !check_bytes ? "not compared" : (byte_bad ? "*** WRONG ***" : "byte-exact"), byte_bad,
                val_bad ? "*** WRONG ***" : "bit-exact", val_bad);
    if (first_bad >= 0) {
        // Say WHERE and WHAT, because "1 element in 524288" is the signature of a rounding boundary and not
        // of a wrong formula - and the two need completely different responses.
        const size_t blk = (size_t) first_bad / 292, off = (size_t) first_bad % 292;
        const char* where = off < 4 ? "d (the f32 scale)" : (off < 260 ? "qs (the quants)" : "bsums");
        std::printf("      first bad byte: block %zu offset %zu = %s   ref %02x  got %02x\n", blk, off,
                    where, r_blocks[(size_t) first_bad], g_blocks[(size_t) first_bad]);
        if (off >= 4 && off < 260) {
            const int j = (int) off - 4;
            const float xv = x[(size_t) (blk * 256 + j)];
            float mx = 0.0f;
            for (int k = 0; k < 256; ++k) { const float ax = std::fabs(x[(size_t) (blk * 256 + k)]); if (ax > std::fabs(mx)) mx = x[(size_t) (blk * 256 + k)]; }
            const float isc = -127.0f / mx;
            const float prod = isc * xv;
            std::printf("      element %d: x %.9g  max %.9g  iscale %.9g  iscale*x %.9g (frac %.6f)\n",
                        j, (double) xv, (double) mx, (double) isc, (double) prod, (double) (prod - std::floor(prod)));
        } else if (off < 4) {
            float rd, gd;
            std::memcpy(&rd, &r_blocks[(size_t) (blk * 292)], 4);
            std::memcpy(&gd, &g_blocks[(size_t) (blk * 292)], 4);
            std::printf("      d: ref %.9g  got %.9g\n", (double) rd, (double) gd);
        }
    }
    cudaFree(d_x);
    cudaFree(d_b);
    cudaFree(d_back);
    return (int) (byte_bad + val_bad);
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: quantize_act_parity [--selftest]\n"); return 2; }
    }
    std::mt19937 rng(7);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    int bad = 0;
    const long long N = 32 * 4096;

    // ---- ordinary activations
    std::vector<float> a((size_t) N);
    for (auto& v : a) v = gauss(rng);
    bad += run_case("random normal", a, true);

    // ---- an activation with a very small dynamic range, where d32 is tiny and the fp16 rounding of d is a
    // LARGE relative change - this is where dividing by d16 instead of d32 would show up in the bytes
    std::vector<float> b((size_t) N);
    for (auto& v : b) v = gauss(rng) * 1e-6f;
    bad += run_case("small magnitude (1e-6)", b, true);

    // ---- an all-zero block must produce a zeroed block, not a division by zero
    std::vector<float> c((size_t) N, 0.0f);
    bad += run_case("all zero", c, true);

    // ---- values that land exactly on .5 boundaries, where rint's half-to-even and a float division's
    // rounding both decide the result
    std::vector<float> d((size_t) N);
    for (long long i = 0; i < N; ++i) {
        // construct so that x / d32 is very close to a .5 boundary: d32 = amax/127, so pick amax = 127.0 and
        // x = k + 0.5 for integer k
        d[(size_t) i] = ((i % 200) - 100) + 0.5f;
    }
    d[0] = 127.0f;
    bad += run_case("exact .5 boundaries", d, true);

    // ---- a single extreme value in each block, so d32 is set by it and every other quant is small
    std::vector<float> e((size_t) N, 0.0f);
    for (long long blk = 0; blk < N / 32; ++blk) e[(size_t) (blk * 32)] = 1000.0f + (float) (blk % 7);
    bad += run_case("one extreme per block", e, true);

    std::printf("\nquantize_q8_0: %d failures over 5 distributions x %lld elements\n", bad, N);

    // ============================== Q8_K ==============================
    //
    // The K-quants' activation: 2.89 GiB of dense weights - the attention projections and `ssm_out`.  The
    // oracle here is GGML's C reference, and the traps are the two the source comments warn about.
    // ============================== the fp16 conversion itself ==============================
    //
    // REGRESSION CHECK FOR A BUG THAT SHIPPED.  The private encoder in `quantize_act.cu` - and the identical
    // one in `shared_expert.cu` - tested `if (exp >= 31)` to catch an out-of-range exponent, which conflates
    // an f32 INF/NAN (raw exponent 255) with a FINITE value too large for fp16.  Every finite overflow came
    // back as a NaN instead of saturating to inf.  **Neither file's fixture could see it, because every value
    // in both is O(1)** - the Q8_0 case above scales by a max that is never near 65504.  Round 198's
    // numpy-generated oracle caught it on the first run.
    //
    // The decoder in this same test had the matching defect in the other direction: it flushed fp16
    // SUBNORMALS to zero, so 0x0001 decoded as 0.0 instead of 2^-24.
    //
    // These are the exact values, asserted against what numpy's float16 says.  They are here rather than
    // only in `qsa_parity` because this is the file that OWNS the Q8_0/Q8_K scales, and a scale that is NaN
    // poisons a whole block silently.
    {
        struct F16Case { float in; uint16_t want; const char* what; };
        const F16Case cases[] = {
            {1e30f, 0x7C00u, "1e30 -> +inf (finite overflow)"},
            {-1e30f, 0xFC00u, "-1e30 -> -inf"},
            {65536.0f, 0x7C00u, "65536 -> +inf (just past fp16 max)"},
            // FLT_MAX is 3.4028235e38, so `1e45f` is not even a legal literal; 3e38 is the largest finite
            // f32 that still overflows fp16, which is what this case is for.
            {3.0e38f, 0x7C00u, "3e38 (near FLT_MAX) -> +inf"},
            {65504.0f, 0x7BFFu, "65504 -> the largest finite fp16, NOT inf"},
            // The boundary, checked against numpy: 65520.0 is the exact tie between 65504 and the
            // "next" fp16 (65536 = inf) and rounds to EVEN, i.e. to inf; 65519.996 is below it and stays
            // finite.  The first version of this table asserted inf for 65519.996 - the FIXTURE was wrong and
            // the kernel was right, which is why the expected bits are quoted from numpy rather than reasoned.
            {65519.996f, 0x7BFFu, "65519.996 -> just below the boundary, still finite"},
            {65520.0f, 0x7C00u, "65520.0 -> the exact tie, rounds to even = inf"},
            {65536.0f, 0x7C00u, "65536 -> past fp16 max"},
            {5.9604645e-8f, 0x0001u, "2^-24 -> the smallest subnormal, NOT zero"},
            {1.0f, 0x3C00u, "1.0"},
            {0.0f, 0x0000u, "0.0"},
        };
        int fp16_bad = 0;
        for (const F16Case& c : cases) {
            const uint16_t got = strata::kernels::f16_from_f32(c.in);
            if (got != c.want) {
                std::printf("    *** %s: want 0x%04X got 0x%04X\n", c.what, c.want, got);
                ++fp16_bad;
            }
        }
        // and the reverse direction on the pattern that was flushed
        const float sub = strata::kernels::f32_from_f16(0x0001u);
        const bool sub_ok = (sub == 5.9604645e-8f);
        std::printf("  %-44s %s (%d of %d wrong)\n", "f32->f16 overflow saturates, not NaN",
                    fp16_bad ? "*** NO ***" : "yes", fp16_bad, (int) (sizeof cases / sizeof cases[0]));
        std::printf("  %-44s %s (0x0001 -> %.9g, want 2^-24)\n", "f16->f32 keeps subnormals",
                    sub_ok ? "yes" : "*** NO ***", (double) sub);
        if (fp16_bad) bad += fp16_bad;
        if (!sub_ok) ++bad;
    }

    std::printf("\nQ8_K  (ggml block_q8_K: 292 bytes / 256 elements, {f32 d; i8 qs[256]; i16 bsums[16]})\n");
    int bad_k = 0;
    const long long NK = 256 * 2048;

    std::vector<float> ka((size_t) NK);
    for (auto& v : ka) v = gauss(rng);
    bad_k += run_case_k("random normal", ka, true);

    std::vector<float> kb((size_t) NK);
    for (auto& v : kb) v = gauss(rng) * 1e-7f;      // tiny: the scale is far from fp16-representable
    bad_k += run_case_k("small magnitude (1e-7)", kb, true);

    std::vector<float> kc((size_t) NK, 0.0f);
    bad_k += run_case_k("all zero", kc, true);

    // EXACT TIES.  `nearest_int` is round-half-to-EVEN (the 12582912.0f magic number); `roundf` is
    // half-AWAY-FROM-ZERO.  They differ only when `iscale*x` is exactly k+0.5, which never happens on real
    // data - so this fixture MANUFACTURES it: with max = 127.0 and iscale = -1.0, x = -(k+0.5) gives exactly
    // k+0.5, and the two rules disagree for every EVEN k.
    std::vector<float> kd((size_t) NK, 0.0f);
    for (long long blk = 0; blk < NK / 256; ++blk)
        for (int j = 0; j < 256; ++j) kd[(size_t) (blk * 256 + j)] = -(float) ((j % 100) + 0.5f);
    for (long long blk = 0; blk < NK / 256; ++blk) kd[(size_t) (blk * 256)] = 127.0f;   // fixes max = 127
    bad_k += run_case_k("exact .5 ties (half-to-even)", kd, true);
    {
        // assert the tie fixture can SEE the difference before trusting the comparison above
        std::vector<uint8_t> rb;
        std::vector<float> rback;
        reference_q8_K(kd.data(), NK, rb, rback, -127.0f);
        long long ties = 0, differ = 0;
        for (long long i = 0; i < NK; ++i) {
            const float t = -kd[(size_t) i];
            if (t - std::floor(t) == 0.5f) {
                ++ties;
                const int even = nearest_int_host(t);
                const int away = (int) std::round(t);
                if (even != away) ++differ;
            }
        }
        std::printf("  %-26s %lld ties, %lld where half-to-even != half-away (%.1f%%)\n",
                    "  tie fixture discriminates", ties, differ, ties ? 100.0 * (double) differ / ties : 0.0);
        if (differ < ties / 3) { std::printf("    *** the tie fixture cannot see the rounding rule ***\n"); ++bad_k; }
    }

    // THE -127 TRAP.  ggml uses `-127/max`; the `-128` version sits COMMENTED OUT in the source, and using it
    // is a 0.79% scaling error that leaves a working-looking activation.  Computed with the rival constant
    // and required to differ.
    {
        std::vector<uint8_t> r127, r128;
        std::vector<float> b127, b128;
        reference_q8_K(ka.data(), NK, r127, b127, -127.0f);
        reference_q8_K(ka.data(), NK, r128, b128, -128.0f);
        double d = 0, m = 0;
        long long qdiffer = 0;
        for (long long i = 0; i < NK; ++i) { d += std::fabs(b127[(size_t) i] - b128[(size_t) i]); m += std::fabs(b127[(size_t) i]); }
        for (size_t i = 4; i < r127.size(); ++i) if (r127[i] != r128[i]) ++qdiffer;
        // The expected size of the effect is exactly (1 - 127/128) = 0.78% of the magnitude, so the floor is
        // 0.5% - below what the rival constant must produce and far above any noise.  The byte count is the
        // blunter evidence and is reported alongside it: 172,017 bytes differ, i.e. a third of the quants.
        std::printf("  %-26s %-4s (%.3f%% of magnitude, %lld bytes differ)\n", "-127 vs -128 scale observable",
                    d / m > 0.005 ? "yes" : "*** NO ***", 100.0 * d / m, qdiffer);
        if (!(d / m > 0.005)) ++bad_k;
    }

    std::printf("\nquantize_act: %d failures over 5 Q8_0 + 4 Q8_K distributions\n", bad + bad_k);
    if (bad + bad_k) return 1;
    if (selftest) std::printf("quantize_act_parity OK\n");
    return 0;
}
