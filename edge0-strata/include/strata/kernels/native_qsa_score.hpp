#pragma once
#include "strata/kernels/qsa.hpp"

namespace strata::kernels {

// Configure before capture; independent of native key pooling and norm/gate.
void native_qsa_score_set_enabled(bool enabled);
bool native_qsa_score_enabled();

// SM120, single contiguous text sequence: F32 pooled[128,max_blocks] and
// query[128,4], four-cell blocks, top-k budget2048. Matches the pinned CUDA MMF
// dispatch with the model's n_kv padded to256 (n_blocks a multiple of64), even
// when only a partial32-row tile is live. This is not the unpadded odd-row
// cuBLAS contract. MMA consumes the same raw F32 register bits as the oracle.
//
// step[kStepCount] is device memory, refreshed before each replay. max_blocks
// must equal max_cells/4+1. Completed pooled rows[0,n_bid) and the spare row
// [n_bid] must be initialized. Writes cell_scores[0,n_kv) only. Applies ReLU
// separately per head, ordered F32 head addition, optional F32 block bias,
// then finite1e9 F32 bias for the incomplete tail; full blocks also receive
// the reference's +0 bias. Does not alter selection, ties, or selected order.
//
// All spans are aligned, disjoint, persistent through replay; inputs finite.
// Explicit nonnull stream. No allocation, host count reads, or synchronization.
// Invalid device counts suppress writes defensively; callers own count validity.
void native_qsa_score(const float* pooled, const float* query, const float* block_bias,
                      const QsaShapes& shapes, const int32_t* step,
                      int64_t max_blocks, int64_t max_cells,
                      float* cell_scores, void* stream);

} // namespace strata::kernels
