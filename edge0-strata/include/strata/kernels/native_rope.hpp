#pragma once

#include "strata/kernels/rope_scaling.hpp"

namespace strata::kernels {
void native_rope_set_enabled(bool enabled);
bool native_rope_enabled();

// Pinned CUDA text-only IMRoPE: F32 rows, 64 rotated channels, equal text positions in all four
// IMRoPE sections. Each device position must be nonnegative. The position buffer remains live
// through graph replay. Supports head_dim 128/256 and exact x==out; partial overlap is rejected.
// Explicit stream required. No allocation or synchronization.
//
// The scaling (rope_scaling.hpp) rides in as the resolved process config: none reproduces the
// original contract exactly, linear/YaRN apply ggml's rope_yarn - the interpolation mix, the
// corr_dims ramp along the pairs, and the mscale magnitude correction folded into cos and sin.
// The struct's knobs are process constants, so kernel arguments baked at graph capture stay valid.
void native_rope_apply(const float* x, float* out, int rows, int head_dim,
                       int n_rot, const RopeScaling& scaling, const int* positions, void* stream);

/// (#783 PR-f, stuchapin909) native_qsa_rms_norm_weighted + native_rope_apply in one launch, bit-identical to the pair
/// (rope_parity check 6): x rows are `in_stride` apart (2 * head_dim reads a q row out of a q|gate row), out rows are
/// head_dim apart, x may equal out when in_stride == head_dim. head_dim 128 or 256 and n_rot 64 only.
void native_qsa_rms_norm_rope(const float* x, int in_stride, const float* gamma, float* out,
                              int rows, int head_dim, int n_rot, float epsilon,
                              const RopeScaling& scaling, const int* positions, void* stream);
/// Whether callers should take the fused kernel (and the in-place rope's 32-thread launch) for this geometry. On the
/// CUDA build it is on unless STRATA_NO_NORM_ROPE=1; on the HIP build it is off until the AMD parity check passes there
/// (rope_parity check 6) and only STRATA_NORM_ROPE=1 turns it on.
bool native_norm_rope_usable(int head_dim, int n_rot);
}
