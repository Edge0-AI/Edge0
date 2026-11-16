// include/strata/kernels/bf16_bits.hpp - the bf16 conversions, written out, as bits.
//
// bf16 is the TOP HALF of an f32, so `f32_from_bf16` is a shift and needs no rounding at all.  The forward
// direction does: ggml's `ggml_compute_fp32_to_bf16` adds `((i >> 16) & 1) + 0x7FFF` and truncates 16 bits.
//
// **THE `& 1` IS ON BIT 16**, the lowest KEPT bit, which is what makes an exact tie round to EVEN.  Using bit
// 15 there is a plausible mistake: it differs only on exact ties, i.e. almost never on real data, which is
// exactly why it would survive any amount of end-to-end testing.  `ref/quant.py` documents the same rule for
// the same reason.
//
// WHY THIS IS SEPARATE FROM `f16_bits.hpp`: fp16 and bf16 are not two spellings of one idea.  fp16 has 5
// exponent bits and 10 mantissa bits; bf16 has 8 and 7.  A conversion routine written for one and reused for
// the other is wrong in a way that produces plausible numbers - bf16's 2^-9 relative precision looks like a
// small fp16 error and is not small at all next to fp16's 2^-11.
//
// `gr.cu` carried a private copy of `f32_to_bf16_bits` until this header existed.  It now includes this.
#pragma once

#include <cstdint>
#include <cstring>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_BF16_HD __host__ __device__
#else
#define STRATA_BF16_HD
#endif

namespace strata::kernels {

/// Round-to-nearest-even f32 -> bf16, returned as raw bits in the LOW half of the uint16.
///
/// **A NaN IS PASSED THROUGH, QUIETED, AS `ggml_compute_fp32_to_bf16` DOES.**  The rounding add alone
/// carries out of the mantissa for a NaN: 0x7FFFFFFF became 0x8000 (-0) and 0x7F800001 became 0x7F80 (+inf), so
/// a NaN upstream was masked instead of propagated.  The test is on the magnitude bits, the result keeps the
/// sign and the top mantissa bits and sets the quiet bit (64 = bit 6 of the bf16).  Every non-NaN input,
/// infinities included, rounds exactly as before.
STRATA_BF16_HD inline uint16_t bf16_from_f32(float f) {
    uint32_t i;
#if defined(__HIP_DEVICE_COMPILE__)
    __builtin_memcpy(&i, &f, 4);
#else
    std::memcpy(&i, &f, 4);
#endif
    if ((i & 0x7FFFFFFFu) > 0x7F800000u) return (uint16_t) ((i >> 16) | 64u);
    i = (i + ((i >> 16) & 1u) + 0x7FFFu) & 0xFFFF0000u;
    return (uint16_t) (i >> 16);
}

/// bf16 bits -> f32.  Exact, and no rounding: the low 16 bits of an f32 are simply zero.
STRATA_BF16_HD inline float f32_from_bf16(uint16_t h) {
    const uint32_t i = (uint32_t) h << 16;
    float f;
#if defined(__HIP_DEVICE_COMPILE__)
    __builtin_memcpy(&f, &i, 4);
#else
    std::memcpy(&f, &i, 4);
#endif
    return f;
}

}  // namespace strata::kernels
