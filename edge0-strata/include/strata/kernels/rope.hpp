// include/strata/kernels/rope.hpp - NEOX partial RoPE, host-callable (P2.S2).
//
// Pairing is NEOX: (i, i + n_rot/2), NOT the adjacent pair (2i, 2i+1).  The rotation is PARTIAL - only the
// first n_rot of head_dim are touched, 64 of 256 in this artifact - and the tail passes through.
//
// The cos/sin table is built on the HOST in float64 (see src/kernels/cuda/rope.cu for why) and the kernel is
// a pure rotation over it.  The table depends on (n_rot, theta, position) and not on the token, so it is worth
// caching; `build_rope_table` fills `max_pos` positions of `n_rot/2` pairs each.
//
// ROPE SCALING RIDES IN THE TABLE, NOT IN THE KERNEL.  The kernels below read cos/sin values and
// cannot tell a scaled table from an unscaled one - with scaling on, the scaled angles (and YaRN's
// mscale magnitude correction, folded into the same values) simply ARE the table.  That is why
// neither `rope_neox_apply` nor the indexer's pooling kernel takes any scaling argument: the config
// enters once, here, at build time (`rope_scaling.hpp`).
#pragma once

#include <cstdint>

#include "strata/kernels/rope_scaling.hpp"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_ROPE_HD __host__ __device__
#else
#define STRATA_ROPE_HD
#endif

namespace strata::kernels {

/// ONE NEOX PAIR: `(a, b) -> (a*c - b*s, a*s + b*c)`.
///
/// This exists so the pairing and the sign convention live in ONE place.  `rope_neox_apply` walks a whole row
/// with it, and `indexer_key_append`'s pooling kernel uses it to rotate the single pooled row it has just
/// written - two call sites that MUST agree, because the indexer's key is compared against query scores
/// downstream and a mismatched rotation produces a well-formed key pointing the wrong way.
///
/// It is `__host__ __device__` (the `f16_bits.hpp` pattern) rather than device-only so a host-side reference in
/// a test can call THE SAME function instead of transcribing it.  A transcription is exactly how the pairing
/// convention gets written down wrong.
///
/// In-place is safe: the caller writes both outputs from the two inputs it read, and under NEOX no two pairs
/// share an element (`i` pairs with `i + n_rot/2`, so the halves are disjoint).
STRATA_ROPE_HD inline void rope_neox_pair(float a, float b, float c, float s, float& oa, float& ob) {
    oa = a * c - b * s;
    ob = a * s + b * c;
}

void build_rope_table(int n_rot, double theta, int max_pos, float* cos_tab, float* sin_tab);

/// The same table under the process's rope scaling (rope_scaling.hpp).  For `RopeScalingType::None`
/// this IS the five-argument builder above - the parity test holds the two to be bit-identical - and
/// for linear/YaRN the scaled angles (YaRN's mscale included) replace them.  The math is ggml's,
/// transcribed; see the header of `rope_scaling.hpp` for what arrives from where.
void build_rope_table(int n_rot, const RopeScaling& scaling, int max_pos, float* cos_tab, float* sin_tab);

// `x` and `out` are (rows, head_dim) and `pos` is (rows,) - one position per row, so a batch of heads at
// different sequence positions is one call.  `cos_tab`/`sin_tab` are (max_pos, n_rot/2).  `out` may equal `x`.
//
// **`pos` IS A DEVICE ARRAY, NOT A HOST SCALAR**, and that is a graph-capture requirement rather than a style
// choice: a CUDA graph bakes kernel arguments in at capture time, so a host scalar position would replay the
// first token's value forever.  See the same note on `indexer_key_append` in `qsa.hpp`.
void rope_neox_apply(const float* x, float* out, int64_t rows, int head_dim, int n_rot, const float* cos_tab,
                     const float* sin_tab, const int* pos, void* stream);

}  // namespace strata::kernels
