// include/strata/kernels/dequant_s2.hpp - the S2 decode, host-callable.
//
// S2 is Q2_0's canonical form and 31.64 GiB of the artifact is Q2_0, so this is the hottest decode in the
// engine.  Plane layout is docs/pack-format.md's: codes packed 4 per byte LSB-first (element i at byte i/4,
// bits (i%4)*2) and one FP32 scale per group of 64.  Decode is `(code - 1) * scale` with the subtraction in
// the INTEGER domain, which is what makes it bit-exact (docs/pack-format.md §3.1).
#pragma once

#include <cstdint>

namespace strata::kernels {

// Decode `n_blocks` S2 blocks.  `codes` is n_blocks*16 bytes, `scales` n_blocks floats, `out` n_blocks*64
// floats.  All device pointers.  Synchronises before returning, so a caller can read `out` immediately -
// correctness first; overlap is a Phase 3 concern.
void dequant_s2(const uint8_t* codes, const float* scales, float* out, int64_t n_blocks);

}  // namespace strata::kernels
