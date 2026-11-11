// include/strata/kernels/qsa.hpp - the QSA (full-attention) layer's cache, indexer and attention, P2.S2.
//
// Spec of record: `ref/qsa.py` (whose own selftest carries eleven PROPERTY checks) and `ref/model.py::_qsa` /
// `_indexer` / `_attend`, transcribed from `build_layer_attn` / `build_attn_qsa` in the source.  The
// projections are NOT here: `attn_q`, `attn_k`, `attn_v`, `attn_output`, `indexer.q_proj` and
// `indexer.k_proj` are the existing `s_gemv`/`s2_gemv` kernels, exactly as `gdn_step` leaves them.  What lives
// here is everything BETWEEN the projections:
//
//     kv_append            the new token's K and V into the paged cache
//     indexer_key_append   the raw indexer tail, and the pooling that fires on block completion
//     qsa_index            per-CELL indexer scores: Relu per head, sum over heads, + bias, mapped by cell_block
//     topk_512             the selection: the `width` largest cells, ties by ascending index, returned ascending
//     kv_gather            selected cells through the PAGE TABLE into a contiguous scratch
//     qsa_attend           softmax GQA over the scratch
//     qsa_gate_apply       `attn * sigmoid(gate)` and the round to the fp16 activation `attn_output` consumes
//
// DECODE ONLY, and that is a deliberate scope statement rather than an omission.  `ref/qsa.py::indexer_scores`
// has no mask parameter and says why: "The causal part of the mask is already satisfied: every cached cell
// precedes the query at decode."  The plan's `qsa_index` bullet asks for a "block-causal mask", and at decode
// its entire content is the TAIL -> SPARE SLOT mapping (`cell_block`), which is implemented here and tested.
// The per-query form - a query at position i may only see blocks with `block_start + r - 1 <= i`, plus the tail
// tokens - is a PREFILL property: `phases/phase-5-prefill-and-context-reuse.md` L73 builds it as a
// [chunk x blocks] GEMM with per-row top-k.  Adding a mask flag here that decode cannot exercise would be a
// parameter with no observable behaviour, and this project has already been bitten by checks that cannot see
// the thing they claim to check.
//
// ============================================================================================
// **THIS WHOLE API IS NOT YET CAPTURABLE, AND IT IS NOT OBVIOUS WHICH PARTS ARE NOT.**
//
// A CUDA graph bakes kernel ARGUMENTS in at capture time.  Any argument that CHANGES PER TOKEN is therefore
// frozen at its first-token value on replay - silently, because the graph still runs, still produces finite
// output, and still looks correct.  Auditing this header for that property (round 211-212) found FIVE entry
// points, not one:
//
//     kv_append           int64_t pos        every token      the new cell's position
//     qsa_index           int64_t n_bid      every token      completed blocks
//     qsa_index           int64_t n_kv       every token      cached cells
//     topk_512            int64_t n_kv       every token      cached cells
//     kv_gather           int64_t n_ids      every token      = qsa_selection_width(n_kv, s)
//     qsa_attend          int64_t n_ids      every token      as above
//     indexer_key_append  int64_t pos        FIXED            now takes `const int32_t* pos_dev`
//
// WHAT A CAPTURED LAYER WOULD ACTUALLY DO, and it is worse than a crash: `kv_append` would write EVERY token to
// position 0, `qsa_index` and `topk_512` would score exactly one cell, `kv_gather` would gather one cell, and
// attention would be over a single cell for the whole sequence.  Finite, plausible, and indistinguishable from
// "the model has no context" - a symptom that would be blamed on the KV cache, the selection, or the weights.
//
// THE FIX IS ONE DEVICE BUFFER, NOT SIX SIGNATURES.  Every one of these values is a pure function of the
// current position and the geometry, so they belong together in a small device-resident step state that is
// written once per token:
//
//     step[0] = pos       step[1] = n_kv = pos+1
//     step[2] = n_bid     step[3] = width = qsa_selection_width(n_kv, s)
//
// Each kernel reads the entries it needs, every argument list becomes constant across tokens, and the whole
// QSA path captures.  Writing it as six separate pointer parameters would leave six places to forget and no
// single thing to check.
// ============================================================================================
//
// ---- LAYOUTS.  Every one of these is a runtime buffer and not a weight, so the layout is a free choice - but
// it must be WRITTEN DOWN, because the alternatives index the same numbers and a mix-up is silent.
//
// K/V CACHE: FP16, paged, `[page][kv_head][page_size][head_dim]` with head_dim fastest.  The plan says "FP16
// first" and fp16 is also what makes 32K of context fit: 12 layers x 2 heads x 256 dims x 2 (K and V) x 2 B x
// 32768 = 805 MiB, against 1.6 GiB in f32.  The page granule is coarse (the shape's `page_size`) for the
// reason `Memory/Qwen-5070-engine-campaign/src/plans/paged-kv-cache.md` L146-170 gives: this engine serves a
// handful of sequences and not hundreds, so fine granularity buys nothing and costs per-token indirection.  The
// physical row for logical cell `t` and head `h` is
//
//     page = table[t / page_size];  (page * n_head_kv + h) * page_size + (t % page_size)
//
// i.e. ONE HEAD's worth of consecutive cells is contiguous within a page.  That is the property attention
// wants; the alternative (cell-major inside the page) makes a cell's two heads contiguous instead, which is
// the property `kv_append` wants and nowhere near as valuable.  **Nothing outside this header may assume the
// mapping**: `qsa_attend` reads a gathered scratch and never a pool, and the test runs the same fixture at
// page_size 1, 4 and 512 with a permuted page table and requires the attention output to be IDENTICAL, so the
// granule cannot leak into the result.
//
// INDEXER KEYS: FP32, not fp16 and not the plan's INT8.  These carry no tolerance anywhere in the contract:
// `docs/capture-format.md` L127 makes `indexer_ids` an EXACT key, because a different selected cell is a
// different attention input.  The key is a pooled MEAN of four raw keys followed by a norm and a rotation, so
// an fp16 store perturbs every score by ~2^-11 relative, and the selection boundary is exactly where that
// matters.  The cost of fp32 is 512 B per block per layer = 128 B/token/layer = 1.5 KiB/token over the 12 QSA
// layers, i.e. 50 MiB at 32K: cheap next to a correctness contract.  The plan's INT8 indexer key is a Phase 3
// optimisation and it has to be justified by a measured selection-flip rate against this baseline.
//
// The RAW tail is fp32 for the same reason and because it is only `idx_block - 1` rows.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// Geometry of one QSA layer.  The real artifact: n_head 24, n_head_kv 2, head_dim 256, n_rot 64,
/// idx_n_head 4, idx_dim 128, idx_block 4, idx_top_k 2048, freq_base 1e7, rms_eps 1e-6.
struct QsaShapes {
    int64_t n_head = 0;     ///< query heads (24).  n_head must be a multiple of n_head_kv
    int64_t n_head_kv = 0;  ///< key/value heads (2).  The q -> kv map is `q / (n_head/n_head_kv)`, NOT `q % kv`
    int64_t head_dim = 0;   ///< 256, and must be a multiple of 4
    int64_t n_rot = 0;      ///< 64 rotated dims of head_dim (and of idx_dim), NEOX pairs
    int64_t idx_n_head = 0; ///< indexer query heads (4)
    int64_t idx_dim = 0;    ///< indexer key width (128)
    int64_t idx_block = 0;  ///< r = 4: the indexer pools one key per 4 cells
    int64_t idx_top_k = 0;  ///< 2048 cells of selection budget
    int64_t page_size = 0;  ///< KV page granule in CELLS; 1 is a legal unpaged configuration
};

/// The real artifact's geometry, so no caller transcribes it by hand.
inline QsaShapes qsa_real_shapes() {
    QsaShapes s;
    s.n_head = 24; s.n_head_kv = 2; s.head_dim = 256; s.n_rot = 64;
    s.idx_n_head = 4; s.idx_dim = 128; s.idx_block = 4; s.idx_top_k = 2048;
    // one page = one indexer block (4 cells): the granule KV streaming keeps resident (kv_stream.hpp). The readers
    // resolve a row per cell anyway, so the page size costs them nothing (measured: identical output and speed).
    s.page_size = 4;
    return s;
}

/// The one legal RMSNorm epsilon for this artifact (`attention.layer_norm_rms_epsilon`).
inline float qsa_rms_eps() { return 1e-6f; }
/// `rope.freq_base`, the DEFAULT frequency base.  The artifact ships no `rope.scaling` keys, so the
/// process's rope scaling starts at none - the runtime configuration lives in `rope_scaling.hpp`
/// (`rope_scaling_set`, once at startup, before the table is built and any graph captured), and this
/// constant is what that config's `freq_base` defaults to and what CLI `--rope-freq-base` overrides.
inline double qsa_freq_base() { return 1e7; }

/// The selection width: `min(n_kv, idx_top_k + idx_block - 1)`.
///
/// The `r - 1 = 3` is not slack, it is the incomplete tail: the budget is whole blocks PLUS whatever cells do
/// not fill one.  So the plan's "selection skipped when context <= 2,048" is conservative by two - the source's
/// own bound is 2,051, and `ref/qsa.py` L292-294 records the disagreement.  Below the bound the selection is
/// the IDENTITY, which `topk_512` is required to reproduce exactly rather than approximately.
inline int64_t qsa_selection_width(int64_t n_kv, const QsaShapes& s) {
    const int64_t w = s.idx_top_k + s.idx_block - 1;
    return n_kv < w ? n_kv : w;
}

/// THE PER-TOKEN VALUES EVERY QSA KERNEL NEEDS, IN DEVICE MEMORY.
///
/// This exists so the QSA layer can be a CUDA graph.  A graph bakes kernel ARGUMENTS in at capture time, so any
/// per-token argument is frozen at its first-token value on replay - silently, because the graph still runs and
/// still produces finite output.  Six entry points used to take one of these as a host scalar; every one of
/// them now reads it from here instead, which makes every argument list constant across tokens.
///
/// WHAT IT PREVENTS, concretely: with `pos` frozen, `kv_append` writes every token to cell 0, `qsa_index` and
/// `topk_512` score one cell, `kv_gather` gathers one cell, and attention runs over a single cell for the whole
/// sequence.  Finite, plausible, and indistinguishable from "the model does not use its context".
///
/// The four entries are NOT independent - `n_kv = pos+1` and `width = qsa_selection_width(n_kv, s)` - which is
/// exactly why they live in one buffer with one writer: `qsa_step_write` computes all four from `pos`, so a
/// caller cannot update one and forget another.
///
/// It is written by a plain H2D copy of four int32, which IS capturable and re-reads its source at replay
/// (`bench/micro/pinned_capture.cu` case B) - so the same buffer works in a captured graph and in a direct
/// launch.
enum QsaStep : int {
    kStepPos = 0,     ///< the new cell's sequence position
    kStepNKv = 1,     ///< pos + 1: how many cells are in the cache
    kStepNBid = 2,    ///< n_kv / idx_block: how many blocks have COMPLETED
    kStepWidth = 3,   ///< qsa_selection_width(n_kv, s): how many cells the selection returns
    kStepCount = 4,
};

/// `step` is (kStepCount,) int32 DEVICE memory.  Pointers to it are passed to the entry points below; it is
/// allocated by the caller (it is per-sequence state, not scratch).
///
/// Returns the bytes one step buffer needs, so a caller sizing an arena does not transcribe `kStepCount`.
inline uint64_t qsa_step_bytes() { return sizeof(int32_t) * (uint64_t) kStepCount; }

/// Fills a HOST array of `kStepCount` int32 from `pos`, so a caller can upload it in one copy.  Keeping the
/// arithmetic here rather than at the call site is the point of the buffer: `n_kv` and `width` are derived,
/// and a caller computing them by hand is a caller who can compute them differently.
void qsa_step_fill(int32_t* host_step, int64_t pos, const QsaShapes& s);


/// One token's K and V into the paged FP16 cache, at logical cell `pos`.
///
///   k_pool, v_pool  `[page][kv_head][page_size][head_dim]` fp16, page_size*n_pages cells each
///   page_table      (n_pages,) i32: logical page -> physical page.  NOT assumed to be the identity
///   kcur, vcur      (n_head_kv, head_dim) f32.  `kcur` must ALREADY be normed and rotated - V is never rotated
///
/// The fp32 -> fp16 conversion is `f16_from_f32` (see f16_bits.hpp) and not `__float2half`.
void kv_append(uint16_t* k_pool, uint16_t* v_pool, const int32_t* page_table, int64_t pos,
               const float* kcur, const float* vcur, const QsaShapes& s, void* stream);

/// The selected cells through the page table into a contiguous scratch.
///
///   ids            (n_ids,) i32, logical cells, ASCENDING and unique as `topk_512` returns them
///   k_scratch etc  (n_ids, n_head_kv, head_dim) fp16, written in full
///
/// A cell that is not gathered does not exist for `qsa_attend`, which is how the top-k mask is realised: the
/// source fills the kq mask with -inf and leaves the selected cells at zero, so an unselected cell has weight
/// exactly zero even when it would score highest (`ref/qsa.py` PROPERTY 8 tests precisely that).
void kv_gather(const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* page_table, const int32_t* ids,
               int64_t n_ids, const QsaShapes& s, uint16_t* k_scratch, uint16_t* v_scratch, void* stream);

/// WHERE THE INDEXER'S PER-SEQUENCE STATE LIVES.  One set per QSA layer per sequence.
///
///   tail      (idx_block - 1, idx_dim) f32  the RAW tail: cells of the block being filled, slot = pos % idx_block
///   dead      (idx_dim,) f32                the spare slot's key, = rms_norm(raw key of cell 0), CONSTANT
///   pooled    (max_cells/idx_block + 1, idx_dim) f32  block b at row b, the spare slot at row n_bid
///   block_pos (1,) i32                      scratch: the kernel writes the completed block's first-cell
///                                           position here so the rotation can read it ON THE DEVICE
///
/// **Every pointer in this struct is a DEVICE pointer**, and so are every other pointer this header's entry
/// points take: `rope_neox_apply` indexes its cos/sin table and its position array on the device, so a host
/// scalar passed as `pos` is an illegal access rather than a convenience.  `block_pos` exists for exactly that
/// reason - the rotation position is derived from the host's `pos`, so it has to be handed to the device
/// through memory, which is also the only form a CUDA graph can replay.
///
/// The spare slot MOVES: it is at row `n_bid`, and `n_bid` grows by one on every block completion.  It is also
/// constant in value, because it is derived from cell 0's raw key and `rope_neox(., 0)` is the identity - which
/// is what makes a tail of three keys sufficient.  A design that pooled only from the tail could not produce
/// this key at all after the first block.
struct QsaIndexerBuffers {
    float* tail = nullptr;
    float* dead = nullptr;
    float* pooled = nullptr;
    int32_t* block_pos = nullptr;
};

/// Append one raw indexer key (IDX_DIM wide, fp32, NEVER normed and NEVER rotated - `ref/qsa.py` L201-210) and
/// run the pooling if this cell COMPLETES a block.
///
/// `pooled[b] = rope_neox(rms_norm(mean(raw[b*r .. b*r+r-1]), w_k_norm), pos of cell b*r)` for b = pos/r on a
/// completion, and the spare slot `pooled[n_bid] = rope_neox(rms_norm(raw[0], w_k_norm), 0)`.  The rotation
/// position is the block's FIRST cell, which `ref/qsa.py` PROPERTY 4 pins and the test re-pins: rotating at the
/// block's LAST cell keeps every shape and every magnitude.
///
/// **`pos_dev` IS A DEVICE POINTER AND THIS FUNCTION IS CAPTURABLE.**  It used to take `int64_t pos` as a host
/// scalar, and a CUDA graph bakes kernel ARGUMENTS in at capture time - so a captured call would have replayed
/// the FIRST token's position forever: `slot = pos % idx_block` would never advance, no block would ever
/// complete, the indexer would keep pooling cells 0-3, and the selection would stay pinned to the first block.
/// Not a crash and not a small error - a plausible attention output over the wrong cells.  Worse than a stale
/// POINTER, which this project's pitfalls list already warns about, because a stale pointer tends to fault while
/// a stale VALUE looks like a working graph.
///
/// It could not be fixed by swapping the scalar for a pointer alone: `pos` was ALSO read on the host, to decide
/// whether to run the completion rotation at all and which pooled row to rotate.  So the rotation moved INTO
/// the pooling kernel, which is why this now takes `cos_tab`/`sin_tab` and `n_rot`, and why `block_pos` is kept
/// only as a readable record of the rotation position rather than as the input to a second launch.
///
/// `pos_base` IS THE SEQUENCE'S FIRST CELL'S POSITION, so cell j sits at position `pos_base + j`.  The kernel
/// cannot derive it: `QsaCache.pos` is what the reference rotates by, and equating a cell's position with its
/// index is only right for a sequence that starts at 0.  The test runs the fixture at `pos_base = 100`, where
/// the two readings are 514% apart.  `pos_base` is CONSTANT for a sequence, so unlike `pos_dev` it is safe to
/// bake into a graph.
void indexer_key_append(const float* raw, const int32_t* pos_dev, int32_t pos_base, const float* w_k_norm,
                        float eps, const QsaIndexerBuffers& b, const QsaShapes& s, const float* cos_tab,
                        const float* sin_tab, void* stream);

/// Per-CELL indexer scores, `n_kv = pos + 1` of them, fp32.
///
///   pooled  (n_bid + 1, idx_dim)   block keys, the spare slot last, as `indexer_key_append` maintains them
///   q_idx   (idx_n_head, idx_dim)  ALREADY normed and rotated by the caller
///   bias    (n_bid + 1,) or null   optional additional per-block bias
///
///   score[j] = bias[cb] + sum_h Relu(dot(pooled[cb], q_idx[h])),   cb = min(j / r, n_bid)
///   score[j] += 1e9f for incomplete-tail cells (j >= (n_kv / r) * r).
/// The tail bias matches llama.cpp's causal decode input and applies even when bias is null.
/// Completed blocks receive no tail bias; the step API derives this from each replay's n_kv.
///
/// The Relu is PER HEAD and the sum is over heads AFTERWARDS: `relu(sum_h dot_h)` is the rival reading, and it
/// is a different number whenever any head's dot is negative.  Flipping one query head's sign can never lower
/// the score, which is the property the test asserts.  There is NO softmax and NO 1/sqrt(d) here.
void qsa_index(const float* pooled, int64_t n_bid, const float* q_idx, const float* bias, const QsaShapes& s,
               int64_t n_kv, float* cell_scores, void* stream);

/// The largest `qsa_selection_width(n_kv)` cells of `cell_scores`, ascending, into `ids`.
///
/// Ties are broken by ASCENDING INDEX.  ggml does not specify that, and `ref/qsa.py` is explicit about it so a
/// mismatch is at least diagnosable; the host test compares against `np.lexsort((arange, -scores))`, which is
/// the same rule, implemented independently of the kernel's threshold search.
///
/// `cap` is the caller's buffer size in ids and must be >= the width; the kernel writes EXACTLY `width` and
/// refuses rather than truncating.  The `-1` padding that `docs/capture-format.md` L66-73 describes is the
/// capture writer's job, not this kernel's: the padded width is a property of the run, not of one token.
///
/// `n_kv` must be <= 32768 (`kTopkMaxCells`): the selection runs in ONE block, which is what makes the
/// ascending tie rule deterministic without a second pass over global memory.  Phase 3 replaces this with a
/// radix select and Phase 7 raises the cap; until then a longer context is refused loudly.
inline constexpr int64_t kTopkMaxCells = 32768;

void topk_512(const float* cell_scores, int64_t n_kv, const QsaShapes& s, int64_t cap, int32_t* ids,
              void* stream);

/// Softmax GQA over ONLY the gathered cells.  `q` is (n_head, head_dim) f32, already normed and rotated.
///
///   scores[j] = dot(k[j][q / (n_head/n_head_kv)], q[h]) / sqrt(head_dim)
///   w         = softmax_j(scores)
///   attn[h]   = sum_j w[j] * v[j][q / (n_head/n_head_kv)]
///
/// The q -> kv head map is INTEGER DIVISION (`ggml-cpu/ops.cpp` L8729: `iv2 = iq2 / rv2`), not `q % n_head_kv`
/// - with 24 and 2 the two give [0 x12, 1 x12] and [0,1,0,1,...], and both produce a well-formed result.
///
/// `attn` is (n_head, head_dim) f32 and NOT yet gated; `weights`, when non-null, is (n_head, n_ids) f32, which
/// is what `docs/capture-format.md`'s `L{n}.qsa_scores` records.  Produced here rather than recomputed by the
/// caller so the capture cannot drift from the attention it claims to describe.  An EMPTY `ids` is legal and
/// gives a zero `attn` (no selected cell has no softmax to take).
void qsa_attend(const float* q, const uint16_t* k_scratch, const uint16_t* v_scratch, int64_t n_ids,
                const QsaShapes& s, float* attn, float* weights, void* stream);

/// `attn *= sigmoid(gate)` in f32, then ONE round to fp16 for the `attn_output` GEMV.
///
///   q_full  (n_head, 2*head_dim) f32  the raw `attn_q` projection, BEFORE any norm
///   gate    = q_full[h][head_dim + d]  - the SECOND half of each head's 2*head_dim block
///   out     (n_head, head_dim) fp16
///
/// The split is NOT element-interleaved (`ref/qsa.py` PROPERTY 10 pins that: `q = per_head[:, :head_dim]`,
/// `gate = per_head[:, head_dim:]`), and the rounding happens AFTER the multiply so there is one rounding
/// rather than two.  SIGMOID, not SiLU: this artifact's QSA gate is a sigmoid.
void qsa_gate_apply(const float* attn, const float* q_full, const QsaShapes& s, uint16_t* out, void* stream);

/// THE SAME MULTIPLY WITHOUT THE fp16 ROUND, for the caller whose weight wants Q8_K.
///
/// `attn_output` is a K-quant (Q4_K/Q5_K/Q6_K in every QSA layer of this artifact) and its `vec_dot_type` is
/// **Q8_K, not fp16** - `docs/activation-contract.md` is explicit, and the measured cost of getting it wrong is
/// 0.66-1.41% per GEMV against a 1e-3 tolerance, which is above what Gate C1 allows.  The fp16 form exists
/// because that is what the engine did first; this one exists because that was a gap, and the caller quantizes
/// to Q8_K between the two calls.
///
/// The arithmetic is IDENTICAL to `qsa_gate_apply`'s up to the last step, and the multiply is in double in both
/// - so switching between them changes exactly one rounding, which is the point.
void qsa_gate_apply_f32(const float* attn, const float* q_full, const QsaShapes& s, float* out, void* stream);

// ================= THE CAPTURABLE ENTRY POINTS =================
//
// These take the per-token counts from `step` AND size every launch from a CONSTANT capacity, which is the two
// things a CUDA graph requires.  The host-scalar entry points above are thin wrappers over these: they fill a
// module-level step buffer and pass this token's counts as the capacities, so they behave exactly as before and
// stay convenient for a direct launch or a test.
//
// **THE LAYER MUST USE THESE.**  The wrappers are correct for a direct launch and WRONG inside a graph: their
// grid and their shared size change per token, and a graph bakes both in.  A captured `qsa_index` launched for
// one block leaves every other pooled row unscored; a captured `qsa_attend` sized for one cell reads past its
// shared buffer once the sequence grows.  Both failures are silent.
//
// The kernels already guard for the surplus: `qsa_index_kernel` returns for `b > n_bid`, `kv_gather_kernel`
// returns past its device `n_ids`, and `qsa_attend_kernel` handles an empty selection itself.  So passing a
// capacity LARGER than this token needs is correct, not merely safe.

/// `max_blocks` must be at least `(max_cells / idx_block) + 2`.
void qsa_index_step(const float* pooled, const float* q_idx, const float* bias, const QsaShapes& s,
                    const int32_t* step, int64_t max_blocks, float* cell_scores, void* stream);

/// `cap` must be `qsa_selection_width(kTopkMaxCells, s)` - the largest width the kernel can ever write.
void topk_512_step(const float* cell_scores, const QsaShapes& s, int64_t cap, const int32_t* step,
                   int32_t* ids, void* stream);

/// `max_ids` must be the CAPACITY of `k_scratch`/`v_scratch` in ids.
void kv_gather_step(const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* page_table,
                    const int32_t* ids, const int32_t* step, int64_t max_ids, const QsaShapes& s,
                    uint16_t* k_scratch, uint16_t* v_scratch, void* stream);

/// `max_ids` must be at least the largest selection the geometry can produce, because it sizes the SHARED
/// memory the softmax works in.
void qsa_attend_step(const float* q, const uint16_t* k_scratch, const uint16_t* v_scratch,
                     const int32_t* step, int64_t max_ids, const QsaShapes& s, float* attn, float* weights,
                     void* stream);

struct KvHostPools;   // kv_stream.hpp

/// The cell's position comes from `step`, so this is the form a graph may contain. With a host copy (KV streaming)
/// the cell is written there too, and to VRAM only if its block is resident.
void kv_append_step(uint16_t* k_pool, uint16_t* v_pool, const int32_t* page_table, const int32_t* step,
                    const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                    const KvHostPools* host = nullptr);

}  // namespace strata::kernels
