// include/strata/kernels/s2_gemv.hpp - the S2 GEMV, host-callable (P2.S2).
//
// The expert matvec is the hottest thing in the engine: every token runs 48 layers x 10 experts x 3 roles of
// it, against weights that live in DRAM at 31.64 GiB.  Layout is `docs/pack-format.md`'s S2: codes packed 4
// per byte LSB-first, one FP32 scale per 64 elements.  GGUF's ne[0] varies fastest, so a tensor is [n_in,
// n_out] with n_in CONTIGUOUS - which is why one thread per output row reads a contiguous run of blocks.
//
// Activations are FP16 per the phase spec ("FP16 activations in").  The OUTPUT here is FP32 rather than FP16:
// the intended FP16 out needs an f32->fp16 conversion, and rather than write one under a parity test that
// would then be testing two new things at once, this step keeps the output width the one that can be checked
// exactly.  The FP16 output and its 1e-3 tolerance is the next increment, and this file says so rather than
// leaving a reader to wonder.
#pragma once

#include <cstdint>

namespace strata::kernels {

// y[o] = sum_i fp16(x[i]) * W[i][o], with W in S2 canonical planes.
// `codes` is n_out * (n_in/64) * 16 bytes, `scales` n_out * (n_in/64) floats, `x` n_in uint16 fp16 patterns,
// `y` n_out floats.  All device pointers.  n_in must be a multiple of 64.
void s2_gemv(const uint16_t* x, const uint8_t* codes, const float* scales, float* y, int64_t n_in,
             int64_t n_out);

}  // namespace strata::kernels
