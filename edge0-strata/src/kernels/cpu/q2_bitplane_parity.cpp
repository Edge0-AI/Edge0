// src/kernels/cpu/q2_bitplane_parity.cpp - the opt-in Q2_0 bit-plane kernel (STRATA_Q2_BITPLANE=1, PR #706),
// on the AVX-2 pair where the CPU has AVX2 and on the AVX1 pair below it (issue #1699) - the bit-plane is an
// AVX2/AVX1 feature, the AVX-512 kernel does not read the image.
//
//   1. against the legacy kernel of the same rung: the integer part is exact, only the float summation order
//      differs, so the two agree to a few ulps of the largest term (a relative bound on the row's magnitude);
//   2. bitwise: a token's rows are the same alone, in a window of any width, and cut into any row ranges
//      (the engine's pool splits rows between workers, a verify window changes the width);
//   3. the legacy path is untouched when the switch is off (this test only runs with the switch on: ctest sets it).
// Synthetic data, CPU only; skipped (pass) on a CPU without AVX.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace c = strata::kernels::cpu;

int main() {
    if (!c::q2_bitplane_enabled()) {
        std::printf("q2_bitplane_parity: STRATA_Q2_BITPLANE=1 is not set\n");
        return 1;
    }
    if (!c::cpu_avx1_ok()) { std::printf("q2_bitplane_parity: no AVX on this CPU, skipped\n"); return 0; }
    const bool avx2 = c::cpu_avx2_ok();   // the pair under test: AVX-2 where present, AVX1 below it
    auto quant = [&](const float* x, int n, c::ActQ& q) { avx2 ? c::act_quant_q8_1_avx2(x, n, q)
                                                              : c::act_quant_q8_1_avx1(x, n, q); };
    auto rows_bp_k = [&](const uint8_t* w, size_t rb, int nb, const c::ActQ* const* a, int nt, float* const* o,
                         int r0, int r1) { avx2 ? c::q2_0_gguf_rows_multi_avx2(w, rb, nb, a, nt, o, r0, r1)
                                               : c::q2_0_gguf_rows_multi_avx1(w, rb, nb, a, nt, o, r0, r1); };
    auto rows_lg = [&](const uint8_t* w, size_t rb, int nb, const c::ActQ* const* a, int nt, float* const* o,
                       int r0, int r1) { avx2 ? c::q2_0_gguf_rows_multi_avx2_legacy(w, rb, nb, a, nt, o, r0, r1)
                                             : c::q2_0_gguf_rows_multi_avx1_legacy(w, rb, nb, a, nt, o, r0, r1); };
    std::mt19937 rng(706);
    std::normal_distribution<float> nd(0.f, 1.f);
    int fail = 0;
    for (const int n : {128, 256, 1024, 2560}) {   // 2560 = 40 blocks: an even count of pairs; 128 = one pair
        const int nblocks = n / 64;
        const size_t row_bytes = (size_t) nblocks * 18;
        const int rows = 37;   // not a multiple of 4: the 4-row path and the tail
        std::vector<uint8_t> w((size_t) rows * row_bytes);
        for (int r = 0; r < rows; ++r)
            for (int b = 0; b < nblocks; ++b) {
                uint8_t* blk = &w[(size_t) r * row_bytes + (size_t) b * 18];
                const uint16_t d = (uint16_t) (0x1C00 + rng() % 0x0800);   // fp16 ~0.004-0.016
                std::memcpy(blk, &d, 2);
                for (int i = 0; i < 16; ++i) blk[2 + i] = (uint8_t) rng();
            }
        const int NT = 8;
        static c::ActQ acts[8];
        std::vector<std::vector<float>> xs(NT, std::vector<float>((size_t) n));
        for (int t = 0; t < NT; ++t) {
            for (float& v : xs[(size_t) t]) v = nd(rng) * (t % 3 == 0 ? 8.f : 1.f);
            quant(xs[(size_t) t].data(), n, acts[t]);
            if (acts[t].bp_pairs != nblocks / 2) { std::printf("FAIL n=%d: no bit-plane image\n", n); ++fail; }
        }
        const c::ActQ* ap[8];
        for (int t = 0; t < NT; ++t) ap[t] = &acts[t];
        // the whole window at once, with the bit-plane kernel
        std::vector<std::vector<float>> win(NT, std::vector<float>((size_t) rows)), ref = win, alone = win, cut = win;
        float* wo[8]; float* ro[8]; float* ao[8]; float* co[8];
        for (int t = 0; t < NT; ++t) { wo[t] = win[(size_t) t].data(); ro[t] = ref[(size_t) t].data();
                                       ao[t] = alone[(size_t) t].data(); co[t] = cut[(size_t) t].data(); }
        rows_bp_k(w.data(), row_bytes, nblocks, ap, NT, wo, 0, rows);
        rows_lg(w.data(), row_bytes, nblocks, ap, NT, ro, 0, rows);
        for (int t = 0; t < NT; ++t) {   // one token at a time
            rows_bp_k(w.data(), row_bytes, nblocks, ap + t, 1, ao + t, 0, rows);
        }
        for (int r0 = 0; r0 < rows; r0 += 5)   // rows cut into ranges of five
            rows_bp_k(w.data(), row_bytes, nblocks, ap, 3, co, r0, r0 + 5 < rows ? r0 + 5 : rows);
        for (int t = 0; t < NT; ++t)
            for (int r = 0; r < rows; ++r) {
                const float a = win[(size_t) t][(size_t) r], b = ref[(size_t) t][(size_t) r];
                if (std::memcmp(&a, &alone[(size_t) t][(size_t) r], 4) != 0) {
                    std::printf("FAIL n=%d t=%d r=%d: a token alone differs from the window\n", n, t, r); ++fail;
                }
                if (t < 3 && std::memcmp(&a, &cut[(size_t) t][(size_t) r], 4) != 0) {
                    std::printf("FAIL n=%d t=%d r=%d: a row range differs from the whole\n", n, t, r); ++fail;
                }
                // a few ulps of the row's magnitude (|d| * |q| * n bounds the sum's terms)
                const float tol = 1e-4f * (std::fabs(b) + 1.f);
                if (!(std::fabs(a - b) <= tol)) {
                    std::printf("FAIL n=%d t=%d r=%d: bit-plane %.7g vs legacy %.7g\n", n, t, r, a, b); ++fail;
                }
            }
    }
    // a width that is not a multiple of 128 (an odd number of blocks): no image, the legacy kernel answers
    {
        const int n = 192, nblocks = 3;
        static c::ActQ a;
        std::vector<float> x((size_t) n);
        for (float& v : x) v = nd(rng);
        quant(x.data(), n, a);
        if (a.bp_pairs != 0) { std::printf("FAIL: an odd block count got an image\n"); ++fail; }
    }
    std::printf(fail ? "FAIL (%d)\n" : "PASS\n", fail);
    return fail ? 1 : 0;
}
