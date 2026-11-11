// include/strata/kernels/qsa_prompt_attn.hpp - perf-review D-1: the prompt path's QSA attention on tensor cores.
//
// `qsa_decode_attn_batch` serves a prompt one query at a time with the decode kernel: FP32 dot products with a warp
// reduction per head and cell, 64-cell chunks whose partial sums go through global scratch to a second (merge)
// kernel. At 32K that is 21% of the prompt (5.6 s of 26 s on a 5070, Q2_0, int8 KV), and it is limited by
// instruction issue, not memory (~150 GB/s of logical reads, 3.4 TFLOP/s).
//
// This kernel keeps the same per-query selection (no masking, no union) and reads the same pools, but:
//   * one block per (query, KV head) walks all its selected cells in chunks of 32 with an online softmax: no
//     split-K scratch and no merge kernel;
//   * q.k and p.v are m16n8k16 FP16 MMAs with FP32 accumulation over the 12 query heads of the KV head (+4 pad rows);
//   * the stored values enter exactly: int8 codes are exact in FP16 and their per-64 scales are applied in FP32
//     (to the q.k partial of each 64-dim group, and folded into p for p.v); FP16 KV is used as is;
//   * q and p are split into FP16 hi + lo parts (two MMAs each), so they keep ~22 bits: the result differs from
//     the FP32 kernel by summation order and the exp2 rounding, not by an FP16 cast.
// Not bitwise equal to `qsa_decode_attn_batch`; `qsa_prompt_attn_parity` bounds the difference and the prompt
// quality gate (needles, teacher-forced top-1) checks it end to end. Q4_0 KV (mode 4): each block's codes enter
// exactly as int8 and its scale (one per 32 values) in FP32, as int8 KV's do (STRATA_PROMPT_ATTN_Q4=0: the old kernel).
#pragma once

#include "strata/kernels/qsa_decode_attn.hpp"

#include <cstdint>

namespace strata::kernels {

/// Same arguments and output as `qsa_decode_attn_batch` minus the scratch. Returns false (nothing launched) when the
/// geometry is not 24 heads / 2 KV heads / 256: the caller then uses the old kernel.
bool qsa_prompt_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* attn, int64_t n_q, void* stream);

/// HIP only (false in every other build): the prompt attention runs on the RDNA4 matrix cores on the current device -
/// the default on gfx12 (STRATA_HIP_WMMA=0 turns it off), STRATA_HIP_WMMA=1 on gfx11 as well.  The prompt path sizes its
/// expert ring for it (prefill.cpp).
bool qsa_hip_wmma_active();

/// STRATA_QSA_SPLIT_ATTN=1 (the idea of fork Eddoursul fd23d91): `qsa_decode_attn_batch`'s decode
/// attention through the int8 tensor-core kernel above with each query's cells split among up to 20 blocks, merged with
/// the log-sum-exp rescale, its partials in `scratch` (`scratch_floats` available).  False (nothing
/// launched) unless int8 KV on sm_80+: the caller keeps qsa_decode_attn_batch.  Not bitwise the decode kernel: (b).
bool qsa_decode_attn_split(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, uint64_t scratch_floats, float* attn, int64_t n_q,
                           void* stream);
/// Tests: the int8-KV kernel on INT8 tensor cores (1), v2 (0), or as STRATA_PROMPT_ATTN_IMMA says (-1).
void qsa_prompt_attn_set_imma(int on);
/// HIP: STRATA_HIP_WMMA asks for the matrix-core prompt attention (=1 or =auto) and this device is a gfx11 / gfx12 matrix-core part
/// that has not switched it off after a failed launch.  The prompt path's 96-slot expert ring follows it.  False off HIP.
bool qsa_prompt_attn_matrix_cores_requested();

}  // namespace strata::kernels
