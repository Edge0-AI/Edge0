// src/kernels/bf16_bits_test.cpp - `bf16_bits.hpp` against ggml's rule, over EVERY f32 bit pattern (CPU only).
//
// the rounding add `i + 0x7FFF + bit16` carries out of the mantissa for a NaN, so 0x7FFFFFFF came back as
// -0 and 0x7F800001 as +inf - an upstream NaN was masked instead of propagated.  `ggml_compute_fp32_to_bf16`
// (ggml-impl.h at the pinned llama.cpp) tests the NaN first and returns `(i >> 16) | 64`.  Three checks, all
// exhaustive (2^32 inputs, a few seconds):
//
//   1. the header equals ggml's function, transcribed below, on every input;
//   2. on every NON-NaN input it equals the previous rounding, so the change cannot move a finite value or an
//      infinity by a single bit;
//   3. every NaN input gives a quiet NaN of the same sign.
//
// And the round trip over all 65,536 bf16 patterns: a non-NaN bf16 survives f32 and back unchanged.
#include "strata/kernels/bf16_bits.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

/// ggml_compute_fp32_to_bf16, ggml/src/ggml-impl.h (MIT, the ggml authors), on raw bits.
uint16_t ggml_bf16(uint32_t u) {
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 64);
    return (uint16_t) ((u + (0x7fffu + ((u >> 16) & 1u))) >> 16);
}

/// The rule before this fix, kept only to prove the change is confined to NaN inputs.
uint16_t previous_bf16(uint32_t u) {
    u = (u + ((u >> 16) & 1u) + 0x7FFFu) & 0xFFFF0000u;
    return (uint16_t) (u >> 16);
}

bool is_nan_bits(uint32_t u) { return (u & 0x7fffffffu) > 0x7f800000u; }

}  // namespace

int main() {
    unsigned long long vs_ggml = 0, vs_previous = 0, nan_lost = 0, nan_inputs = 0, previous_masked = 0;
    uint32_t first_bad = 0;
    bool have_bad = false;
    for (uint64_t k = 0; k <= 0xFFFFFFFFull; ++k) {
        const uint32_t u = (uint32_t) k;
        float f;
        std::memcpy(&f, &u, 4);
        const uint16_t got = strata::kernels::bf16_from_f32(f);
        if (got != ggml_bf16(u)) {
            ++vs_ggml;
            if (!have_bad) { first_bad = u; have_bad = true; }
        }
        if (is_nan_bits(u)) {
            ++nan_inputs;
            const uint32_t back = (uint32_t) got << 16;
            // a quiet NaN (bit 22 of the f32) with the input's sign
            if (!is_nan_bits(back) || (back & 0x00400000u) == 0 || (back >> 31) != (u >> 31)) ++nan_lost;
            if (!is_nan_bits((uint32_t) previous_bf16(u) << 16)) ++previous_masked;
        } else if (got != previous_bf16(u)) {
            ++vs_previous;
            if (!have_bad) { first_bad = u; have_bad = true; }
        }
    }
    std::printf("  f32 -> bf16 over 2^32 inputs: %llu differ from ggml, %llu non-NaN differ from the previous rule\n",
                vs_ggml, vs_previous);
    std::printf("  NaN inputs: %llu, not a quiet same-sign NaN now: %llu (the previous rule masked %llu of them)\n",
                nan_inputs, nan_lost, previous_masked);
    if (have_bad) std::printf("    first mismatch at input 0x%08x\n", first_bad);

    unsigned long long trip_bad = 0;
    for (uint32_t h = 0; h <= 0xFFFFu; ++h) {
        const float f = strata::kernels::f32_from_bf16((uint16_t) h);
        const uint16_t back = strata::kernels::bf16_from_f32(f);
        const bool nan = is_nan_bits(h << 16);
        if (back != (nan ? (uint16_t) (h | 64u) : (uint16_t) h)) ++trip_bad;
    }
    std::printf("  bf16 -> f32 -> bf16 over 65536 patterns: %llu wrong\n", trip_bad);

    const bool ok = vs_ggml == 0 && vs_previous == 0 && nan_lost == 0 && trip_bad == 0 && previous_masked > 0;
    std::printf("\nbf16_bits_test %s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
