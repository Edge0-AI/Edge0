// src/kernels/cpu/q8k_quant_parity.cpp - the AVX-2 Q8_K activation quantizer (q8k_quant_avx2, iq_avx2.cpp) against
// ggml's, byte for byte: the reference (ggml-base's from_float_ref) and what ggml-cpu runs (its from_float), over
// rows of random, scaled, sparse and degenerate values; then the same through native_quant_act / native_quant_h with
// an IQ3_S expert's format, which is what the engine calls.  No GPU, no model.
//
//     q8k_quant_parity            (STRATA_FORCE_ISA=avx: the AVX-2 copy is skipped, native_quant_* must equal ggml's)
//     q8k_quant_parity --bench    ns per 2560-value row, ggml-cpu's and the AVX-2 copy
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx1.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

// ggml-quants.h (ggml-base): the scalar reference, which ggml-cpu's x86 quantize_row_q8_K calls.  Q8_K has no
// from_float_ref in ggml's type traits, so it is declared here (block_q8_K* in ggml; the bytes are all this reads).
extern "C" void quantize_row_q8_K_ref(const float* x, void* y, int64_t k);

namespace {

constexpr int kQK = 256;

std::vector<std::vector<float>> cases(int n) {
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(-100.f, 100.f);
    std::vector<std::vector<float>> c;
    auto add = [&](auto fill) {
        std::vector<float> x((size_t) n);
        for (int j = 0; j < n; ++j) x[(size_t) j] = fill(j);
        c.push_back(std::move(x));
    };
    add([&](int) { return nd(rng); });
    add([&](int) { return nd(rng) * 1e-3f; });
    add([&](int) { return nd(rng) * 1e4f; });
    add([&](int) { return ud(rng); });
    add([&](int) { return 0.f; });                                         // every block zero
    add([&](int j) { return (j / kQK) % 2 ? nd(rng) : 0.f; });             // zero blocks between random ones
    add([&](int j) { return j % kQK == 77 ? nd(rng) : 0.f; });             // one value per block
    add([&](int j) {                                                       // the largest magnitude twice, the
        const int k = j % kQK;                                             // FIRST one's sign wins
        return k == 3 ? -5.f : k == 100 ? 5.f : nd(rng);
    });
    add([&](int j) {
        const int k = j % kQK;
        return k == 3 ? 5.f : k == 100 ? -5.f : nd(rng);
    });
    add([&](int j) { return j % kQK == 9 ? 3e38f : nd(rng); });            // extremes beside ordinary values
    add([&](int j) { return j % kQK == 9 ? -3e38f : nd(rng) * 1e30f; });
    add([&](int) { return nd(rng) * 1e-40f; });                            // denormals
    add([&](int j) { return (float) ((j % 255) - 127) / 127.f; });         // the grid itself, and its ties
    add([&](int j) { return ((float) ((j * 37) % 509) - 254.f) / 508.f; });
    add([&](int j) { return j % kQK == 5 ? std::numeric_limits<float>::quiet_NaN() : nd(rng); });
    add([&](int j) { return j % kQK == 5 ? std::numeric_limits<float>::infinity() : nd(rng); });
    return c;
}

// both outputs prefilled alike: a zero block leaves bsums unwritten in the reference, and so must the copy
bool same(const char* tag, int n, int ci, ggml_from_float_t a, void (*b)(const float*, void*, int64_t),
          const std::vector<float>& x, size_t bytes) {
    std::vector<uint8_t> ya(bytes, 0xAB), yb(bytes, 0xAB);
    a(x.data(), ya.data(), n);
    b(x.data(), yb.data(), n);
    if (std::memcmp(ya.data(), yb.data(), bytes) == 0) return true;
    size_t at = 0;
    while (ya[at] == yb[at]) ++at;
    std::printf("  %s n=%d case %d: MISMATCH at byte %zu of %zu\n", tag, n, ci, at, bytes);
    return false;
}

const cpu::NativeFmt* g_fmt = nullptr;
void via_act(const float* x, void* y, int64_t) { cpu::native_quant_act(*g_fmt, x, y); }
void via_h(const float* x, void* y, int64_t) { cpu::native_quant_h(*g_fmt, x, y); }

}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    ggml_cpu_init();
    const ggml_from_float_t ref = quantize_row_q8_K_ref;
    const ggml_from_float_t gcpu = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float;
    const bool avx2 = cpu::cpu_avx2_ok();
    const bool avx1 = !avx2 && cpu::cpu_avx1_ok();   // the older CPUs' copy (checked wherever AVX exists)
    int failures = 0, checked = 0;
    for (const int n : {256, 512, 2560, 6144}) {
        const size_t bytes = ggml_row_size(GGML_TYPE_Q8_K, n);
        const auto cs = cases(n);
        for (int ci = 0; ci < (int) cs.size(); ++ci) {
            failures += !same("ggml-cpu vs reference", n, ci, ref, gcpu, cs[(size_t) ci], bytes);
            if (avx2) failures += !same("AVX-2 vs reference", n, ci, ref, cpu::q8k_quant_avx2, cs[(size_t) ci], bytes);
            if (avx1) failures += !same("AVX1 vs reference", n, ci, ref, cpu::q8k_quant_avx1, cs[(size_t) ci], bytes);
            ++checked;
        }
    }
    // the engine's entry points: an IQ3_S expert (type 21) quantizes both of its activations to Q8_K
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(21, 21, 2560, 512, f, err)) {
        std::printf("native_fmt: %s\n", err.c_str());
        return 1;
    }
    g_fmt = &f;
    for (const auto& x : cases(2560))
        failures += !same("native_quant_act", 2560, 0, ref, via_act, x, ggml_row_size(GGML_TYPE_Q8_K, 2560));
    for (const auto& x : cases(512))
        failures += !same("native_quant_h", 512, 0, ref, via_h, x, ggml_row_size(GGML_TYPE_Q8_K, 512));
    std::printf("q8k_quant_parity: %d rows x 4 sizes, AVX-2 copy %s, AVX1 copy %s: %s\n", checked / 4,
                avx2 ? "checked" : "skipped (no AVX2, or STRATA_FORCE_ISA)",
                avx1 ? "checked" : "skipped (AVX2 present, no AVX, or STRATA_FORCE_ISA)", failures ? "FAILED" : "identical");
    if (bench && (avx2 || avx1)) {
        std::mt19937 rng(3);
        std::normal_distribution<float> nd(0.f, 1.f);
        const int n = 2560, rows = 4096;
        std::vector<float> x((size_t) n * rows);
        for (auto& v : x) v = nd(rng);
        const size_t bytes = ggml_row_size(GGML_TYPE_Q8_K, n);
        std::vector<uint8_t> y(bytes * rows);
        auto time = [&](const char* tag, auto fn) {
            double best = 1e30;
            for (int rep = 0; rep < 5; ++rep) {
                const auto t0 = std::chrono::steady_clock::now();
                for (int r = 0; r < rows; ++r) fn(x.data() + (size_t) r * n, y.data() + bytes * r, n);
                const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
                best = std::min(best, ns / rows);
            }
            std::printf("  %-10s %7.0f ns per %d-value row\n", tag, best, n);
            return best;
        };
        const double a = time("ggml-cpu", gcpu);
        const double b = avx2 ? time("AVX-2", cpu::q8k_quant_avx2) : time("AVX1", cpu::q8k_quant_avx1);
        std::printf("  %.1fx\n", a / b);
    }
    return failures ? 1 : 0;
}
