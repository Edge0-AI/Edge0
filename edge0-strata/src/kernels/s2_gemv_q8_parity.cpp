// src/kernels/s2_gemv_q8_parity.cpp - the Q8_0-activation GEMV, and the size of the gap it closes.
//
// THREE CHECKS:
//   1. The quantized-activation GEMV matches a host reference that quantizes the activation exactly as
//      `ref/quant.py::q8_0` does and then dots it against the same weights.
//   2. **The gap is REAL and MEASURABLE in this project's own kernels**: the same weights driven by an FP16
//      activation and by a Q8_0 activation must DIFFER, by roughly the activation contract's ~1%.  Without this
//      the first check would pass on a kernel that quietly used fp16 anyway, and the round-186 gap would look
//      fixed without being fixed.
//   3. The measured size of that difference is reported, so the number in the state file is one this test
//      produces rather than one quoted from elsewhere.
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s_gemv.hpp"
#include "strata/kernels/s2_gemv_q8.hpp"

#include <cuda_runtime.h>

#include <cmath>
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

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: s2_gemv_q8_parity [--selftest]\n"); return 2; }
    }

    const long long n_in = 2560, n_out = 64;
    const int tpr = 32;
    std::mt19937 rng(31);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    std::vector<float> x((size_t) n_in);
    for (auto& v : x) v = gauss(rng);
    std::vector<uint8_t> codes((size_t) n_out * n_in / 4);
    for (auto& v : codes) v = (uint8_t) (rng() & 0xFF);
    std::vector<float> scales((size_t) n_out * (n_in / 64));
    for (auto& v : scales) v = 0.001f + 0.0001f * (float) (rng() % 100);

    // ---- device buffers
    float* d_x = nullptr;
    uint8_t* d_act = nullptr;
    uint8_t* d_codes = nullptr;
    float *d_scales = nullptr, *d_y_fp16 = nullptr, *d_y_q8 = nullptr;
    check(cudaMalloc(&d_x, (size_t) n_in * sizeof(float)), "m x");
    check(cudaMalloc(&d_act, (size_t) (n_in / 32) * 34), "m act");
    check(cudaMalloc(&d_codes, codes.size()), "m codes");
    check(cudaMalloc(&d_scales, scales.size() * sizeof(float)), "m scales");
    check(cudaMalloc(&d_y_fp16, (size_t) n_out * sizeof(float)), "m y1");
    check(cudaMalloc(&d_y_q8, (size_t) n_out * sizeof(float)), "m y2");
    check(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "c x");
    check(cudaMemcpy(d_codes, codes.data(), codes.size(), cudaMemcpyHostToDevice), "c codes");
    check(cudaMemcpy(d_scales, scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice), "c scales");

    strata::kernels::quantize_q8_0(d_x, d_act, n_in, nullptr);

    // ---- host reference: quantize the activation with the project's own rule, then dot
    std::vector<uint8_t> h_act((size_t) (n_in / 32) * 34);
    check(cudaMemcpy(h_act.data(), d_act, h_act.size(), cudaMemcpyDeviceToHost), "c act back");
    auto act_val = [&](long long i) {
        const uint8_t* blk = &h_act[(size_t) (i / 32) * 34];
        const uint16_t dbits = (uint16_t) (blk[0] | (blk[1] << 8));
        // fp16 -> f32 without a CUDA header: 0x3C00 = 1.0, and the block scales here are normal
        float d;
        const uint32_t sign = (uint32_t) (dbits & 0x8000) << 16;
        uint32_t ex = (dbits >> 10) & 0x1F, man = dbits & 0x3FF;
        uint32_t out;
        if (ex == 0) { out = sign; }
        else { out = sign | ((ex - 15 + 127) << 23) | (man << 13); }
        std::memcpy(&d, &out, 4);
        return (float) ((int8_t) blk[2 + (i % 32)]) * d;
    };
    std::vector<float> ref((size_t) n_out, 0.0f);
    for (long long o = 0; o < n_out; ++o) {
        float acc = 0.0f;
        for (long long g = 0; g < n_in / 64; ++g) {
            const float d = scales[(size_t) (o * (n_in / 64) + g)];
            for (int j = 0; j < 64; ++j) {
                const long long i = g * 64 + j;
                const uint8_t byte = codes[(size_t) (o * (n_in / 4) + i / 4)];
                const int code = (byte >> ((i % 4) * 2)) & 3;
                acc += (float) (code - 1) * d * act_val(i);
            }
        }
        ref[(size_t) o] = acc;
    }
    strata::kernels::s2_gemv_q8(d_act, d_codes, d_scales, d_y_q8, n_in, n_out, tpr, nullptr);
    std::vector<float> got((size_t) n_out);
    check(cudaMemcpy(got.data(), d_y_q8, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "c y2");

    int bad = 0;
    double worst = 0.0, sum_abs = 0.0;
    for (long long o = 0; o < n_out; ++o) {
        const double a = ref[(size_t) o], b = got[(size_t) o];
        sum_abs += std::fabs(a);
        // relative to the row magnitude is not needed here - these rows are not cancelling - but the metric is
        // stated because round 169 and round 189 both showed the wrong denominator reads as a wrong kernel
        const double rel = std::fabs(a - b) / (std::fabs(a) > 1e-30 ? std::fabs(a) : 1e-30);
        if (!(rel <= worst)) worst = rel;
        // 1e-4, not 1e-5: the measured worst is 1.479e-05, which is FMA contraction over 2560 terms rather
        // than a structural error, and a threshold that passes 62 of 64 by luck is not a threshold.  The phase
        // allows 1e-3 for FP16 paths, so this is ten times tighter than the spec and ten times looser than the
        // measurement.
        if (!(rel <= 1e-4)) ++bad;
    }
    std::printf("  %-38s %s (%d of %lld over 1e-4, worst rel %.3e, mean |ref| %.3f)\n",
                "q8-activation GEMV vs host reference", bad ? "*** WRONG ***" : "matches", bad, n_out, worst,
                sum_abs / (double) n_out);

    // ---- CHECK 2: the gap is real.  The SAME weights driven by an fp16 activation must differ.
    // The fp16 path wants fp16 activations, and `x` is f32, so it is converted exactly (the values are normal
    // and small), which keeps the comparison about the ACTIVATION FORMAT and nothing else.
    // A REAL fp32 -> fp16 conversion.  The first version wrote `0x3C00 + (int)(x*64)`, which adds to the raw
    // BITS and therefore changes the EXPONENT - x = 2.0 became 0x3C80 = 1.25 - so the two paths were driven by
    // unrelated activations and the comparison reported 2770%.  The number was a symptom of the fixture, and
    // the fixture is what the check exists to be able to trust.
    auto f32_to_f16 = [](float f) -> uint16_t {
        uint32_t b;
        std::memcpy(&b, &f, 4);
        const uint32_t sign = (b >> 16) & 0x8000u;
        int exp = (int) ((b >> 23) & 0xFF) - 127 + 15;
        uint32_t man = b & 0x7FFFFFu;
        if (exp >= 31) return (uint16_t) (sign | 0x7C00u | (man ? 0x200u : 0u));
        if (exp <= 0) {
            if (exp < -10) return (uint16_t) sign;
            man |= 0x800000u;
            const uint32_t sh = (uint32_t) (14 - exp);
            uint32_t h = (man >> sh) & 0x3FFu;
            const uint32_t rem = man & ((1u << sh) - 1u);
            if (rem > (1u << (sh - 1)) || (rem == (1u << (sh - 1)) && (h & 1u))) ++h;
            return (uint16_t) (sign | h);
        }
        uint16_t h = (uint16_t) (sign | ((uint32_t) exp << 10) | (man >> 13));
        const uint32_t rem = man & 0x1FFFu;
        if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
        return h;
    };
    std::vector<uint16_t> hx((size_t) n_in);
    for (long long i = 0; i < n_in; ++i) hx[(size_t) i] = f32_to_f16(x[(size_t) i]);
    uint16_t* d_hx = nullptr;
    check(cudaMalloc(&d_hx, hx.size() * sizeof(uint16_t)), "m hx");
    check(cudaMemcpy(d_hx, hx.data(), hx.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "c hx");
    strata::kernels::SForm form{2, -1, 64, strata::kernels::Codebook::Affine, false};
    strata::kernels::s_gemv_split(d_hx, d_codes, d_scales, nullptr, d_y_fp16, n_in, n_out, form, tpr);
    std::vector<float> y16((size_t) n_out);
    check(cudaMemcpy(y16.data(), d_y_fp16, y16.size() * sizeof(float), cudaMemcpyDeviceToHost), "c y1");

    double diff = 0.0;
    for (long long o = 0; o < n_out; ++o) {
        diff += std::fabs((double) y16[(size_t) o] - (double) got[(size_t) o]);
    }
    const double rel_gap = diff / (sum_abs > 1e-30 ? sum_abs : 1e-30);
    std::printf("  %-38s %.4f%% of the reference magnitude\n", "fp16-activation vs q8-activation", rel_gap * 100);
    if (rel_gap < 1e-4) {
        std::fprintf(stderr,
                     "the two activation paths agree to %.2e, so this test CANNOT see the difference it exists "
                     "to measure - either the q8 path silently used fp16, or the fixture is degenerate\n",
                     rel_gap);
        ++bad;
    }

    std::printf("\ns2_gemv_q8: %d failures\n", bad);
    if (bad) return 1;
    if (selftest) std::printf("s2_gemv_q8_parity OK\n");
    return 0;
}
