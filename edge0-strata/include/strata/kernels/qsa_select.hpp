// include/strata/kernels/qsa_select.hpp - plan v0.3 P5/P7: the QSA indexer's block scores and top-k selection,
// for many queries at once and at long context.
//
// The per-token path scores every pooled block with one FP64 block per row (qsa_index_kernel) and selects with a
// single-block kernel that makes 32 bit-serial passes over every CELL (topk_kernel): ~0.1-0.2 ms per query at 32K,
// once per QSA layer and token - 1-2 ms of a decode token and most of a 32K prompt's time.  Here:
//
//   qsa_block_scores : one warp per (query, block), FP32; relu per indexer head, summed; block n_bid (the
//                      incomplete tail) scores the `dead` key and gets +1e9 when it has cells - which is exactly
//                      what the per-token path reads there, because pooled[n_bid] holds `dead` at that time (in
//                      a chunk it may already hold a block completed later, so it is not read).
//   qsa_block_topk   : one block per query; a 4-pass radix select over the query's n_bid + 1 blocks, each
//                      weighted by its cell count, then the cells emitted in ascending order with ties to the
//                      lowest index - the same selection as topk_kernel, over a quarter of the elements.  On sm_90+
//                      a call of up to 16 queries runs it on a cluster of 8 CTAs per query: the same ids.
//
// Queries carry their own step record (pos, n_kv, n_bid, width) as everywhere else in QSA.
#pragma once

#include "strata/kernels/qsa.hpp"

#include <cstdint>

namespace strata::kernels {

/// scores [nq, max_blocks]; q_idx [nq, idx_n_head, idx_dim] (normed and rotated); steps [nq, kStepCount].
/// Batch windows: query q's own indexer keys - pooled[q], dead[q] (q < n <= 8).
struct QsaScoreRows {
    const float* pooled[8] = {};
    const float* dead[8] = {};
    int n = 0;
};
/// qsa_block_scores for up to 8 queries of DIFFERENT sequences in one launch (the multi kernel, per query its own
/// keys; the same arithmetic in the same order: bit-identical scores).
void qsa_block_scores_rows(const QsaScoreRows& rows, const float* q_idx, const int32_t* steps, int64_t nq,
                           int64_t max_blocks, const QsaShapes& s, float* scores, void* stream);
void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                      int64_t max_blocks, const QsaShapes& s, float* scores, void* stream,
                      int64_t active_blocks = -1);   ///< perf-review C-1: > 0 launches only this many blocks (the
                                                     ///< batch's largest n_bid + 1; not for a captured graph)

/// qsa_block_scores for `nq` queries of different slots, query i scoring its own slot's keys (pooled[i], dead[i]; host
/// arrays of device pointers): one launch whose blocks each compute exactly what the single-query call computes.
/// False (nothing launched) where that call would run another kernel (nq > 8, STRATA_SCORES_MULTI=0): loop instead.
bool qsa_block_scores_slots(const float* const* pooled, const float* const* dead, const float* q_idx,
                            const int32_t* steps, int64_t nq, int64_t max_blocks, const QsaShapes& s, float* scores,
                            void* stream);
/// QSAROWS (docs/QSAROWS.md): block_scores_multi_kernel's scores for a call of up to 4 queries, bit for bit, with the
/// queries in registers and the four heads reduce-scattered (six shuffles instead of twenty a block and query).
/// qsa_block_scores takes it with STRATA_QSA_ROWS=1; false (nothing launched) above 4 queries or on another geometry.
/// For qsa_rows_bench's parity and timing it is callable on its own.
bool qsa_block_scores_rows(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                           int64_t max_blocks, const QsaShapes& s, float* scores, void* stream);

/// The same scores on tensor cores (3xTF32, FP32-level accuracy but another summation order: not bitwise; the tail
/// block n_bid is the warp kernel's arithmetic). For the prompt path; false (nothing launched) on another geometry.
/// On AMD it is the gfx12 (RDNA4) WMMA kernel (a three-way bf16 split, six products); false on any other AMD target.
/// On CUDA below sm_80 (no TF32 tensor cores) it is an FP32 tiled kernel, also not bitwise, and only with STRATA_SELECT_SIMT=1 (otherwise false).
bool qsa_block_scores_tc(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                         int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t active_blocks);

/// ids [nq, cap] (cells, ascending); `cap` >= the largest selection width.
void qsa_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                    const QsaShapes& s, int32_t* ids, void* stream,
                    int64_t active_blocks = -1);   ///< > 0: no query of the call has more than this many blocks (n_bid + 1;
                                                   ///< the same contract as qsa_block_scores's). The register kernel is
                                                   ///< chosen by this, not by the capacity max_blocks (a long
                                                   ///< --max-context otherwise sends every short prompt to the slow one).
                                                   ///< CUDA uses the bound only on sm_75; HIP keeps its existing policy.
                                                   ///< Omit it for captured graphs whose context can grow after capture.
/// The same ids on a thread-block cluster of 8 CTAs per query (sm_90+, CUDA; S19). qsa_block_topk takes it for calls
/// of up to 16 queries unless STRATA_QSA_CLUSTER=0. False (nothing launched) where it cannot run: HIP, a card or a
/// build below sm_90, or a capacity whose keys do not fit one cluster's shared memory. Capturable.
bool qsa_block_topk_cluster(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                            const QsaShapes& s, int32_t* ids, void* stream);
/// The original kernel (256 threads, one shared histogram, keys read from memory on every radix pass), for tests and
/// STRATA_TOPK_OLD=1: the same ids.
void qsa_block_topk_ref(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                        const QsaShapes& s, int32_t* ids, void* stream);

/// The long-context decode select (the verifier's STRATA_SELECT_LONG=1 windows; CUDA builds with
/// STRATA_EXPERIMENTAL_SM60, elsewhere the plain calls).  qsa_block_topk_long: qsa_block_topk's ids; a call with
/// `multi_scratch` (qsa_topk_multi_scratch_ints(nq) device ints, one call at a time per scratch, no initial value), no
/// active count and <= 16 queries runs several CTAs per query (block_topk_multi_*: STRATA_TOPK_MULTI=0 off, =1 on any
/// card, default compute capability 6.x past the register kernel's reach); otherwise it is qsa_block_topk.
/// qsa_block_scores_long: qsa_block_scores's bits; a call of <= 8 queries with no active count runs
/// block_scores_dx_kernel (STRATA_SCORES_DX=0 off, =1 on any card, default compute capability 6.x); otherwise it is
/// qsa_block_scores.
void qsa_block_topk_long(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                         const QsaShapes& s, int32_t* ids, void* stream, int64_t active_blocks, int32_t* multi_scratch);
void qsa_block_scores_long(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                           int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t active_blocks);
/// Device ints a qsa_block_topk_long call of `nq` queries may use as multi_scratch (0 on a build without the multi-CTA
/// path: pass null).
int64_t qsa_topk_multi_scratch_ints(int64_t nq);

}  // namespace strata::kernels
