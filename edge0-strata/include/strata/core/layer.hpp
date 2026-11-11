// include/strata/core/layer.hpp - one GDN LAYER, composed.  P2.S5's mixer.
//
// This is the first place in the engine where the kernels are COMPOSED rather than tested individually, so
// the thing it has to get right is the ORDER and the buffers, not the arithmetic.  Every kernel it calls is
// parity-tested on its own; what can go wrong here is a buffer sized for the wrong shape, a step done in the
// wrong order, or a value that should have been converted and was not.
//
// `ref/gdn.py::gdn_decode` is the specification, and the order below is its docstring verbatim:
//
//     qkv  = wqkv @ x                     (10240 = q|k|v, each a contiguous run of 128-wide heads)
//     conv -> SiLU                        (SiLU is applied to the WHOLE conv output before the split)
//     q,k  = l2_norm                      (v is NOT normalised)
//     beta = sigmoid(wbeta @ x)
//     gate = softplus(walpha @ x + dt) * ssm_a
//     recurrent step
//     y    = rms_norm(o, ssm_norm) * sigmoid(z)      <- SIGMOID for qwen4exp; qwen3.5 used SiLU
//     out  = ssm_out @ y
//
// FOUR THINGS IN THAT ORDER ARE EASY TO READ PAST, and each produces a plausible number:
//
//   * **SiLU is applied BEFORE the split**, to the whole 10240-wide conv output.  Applying it only to q and
//     k, or after the reshape, is the same arithmetic on different data.
//   * **v is NOT normalised.** `l2_norm` applies to q and k only - two rows of 16 heads, not the 48 of v.
//   * **the `/hc`-style scale on q is `1/sqrt(S)`**, applied by the CALLER because `ref/gdn.py` says the two
//     llama.cpp paths apply it in different places.  It goes on AFTER the norm.
//   * **the state is transposed** relative to the reference: (S, h_v, S) here, (S, S, h_v) there.  That is
//     `gdn.hpp`'s choice and it is the reason `gdn_step`'s arguments are laid out as they are.
//
// THE ACTIVATIONS FOLLOW THE CONTRACT, which is why three different GEMVs appear below: `attn_qkv`,
// `attn_gate` and `ssm_out` are K-quants (Q8_K), while `ssm_alpha` and `ssm_beta` are BF16 (bf16).  Using one
// activation format for all five would be a 0.66-1.41% error on the three that matter - measured, in
// `s_gemv_q8k_parity` and `bf16_gemv_parity`.
#pragma once

#include "strata/core/layout.hpp"
#include "strata/core/weights.hpp"

#include "strata/kernels/gr.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"

#include <cstdio>
#include <functional>

#include "strata/core/vmm.hpp"
#include <cstdlib>
#include <cstdint>
#include <string>

namespace strata::core {

/// Select pinned CUDA BF16/F32 projections for SSM gates, routing and the sparse indexer.
/// Configure before session capture; captured graphs retain the selected implementation.
/// Does not change GR, PLE or the shared expert's scalar gate.
void layer_set_native_bf16(bool enabled);
// Diagnostic vector-attention adapter; configure before capture, context <=256.
void layer_set_native_flash_attn_short(bool enabled);

/// Every buffer one GDN layer needs, carved from an arena ONCE.  Sizes are functions of `ModelGeometry`, and
/// `gdn_buffers_bytes` returns the total so a caller can size the arena before committing.
struct GdnBuffers {
    // the activation, in the formats the contract asks for.  BOTH quantized images are produced for every
    // layer, because WHICH ONE A WEIGHT WANTS IS A PROPERTY OF THE WEIGHT and the pack mixes them by layer -
    // `attn_qkv` alone is IQ4_XS on 13 layers, Q3_K on 18, Q4_K on 4 and S2 on 1.
    uint8_t* x_q8k = nullptr;      ///< n_embd, block_q8_K
    uint8_t* x_q8_0 = nullptr;     ///< n_embd, block_q8_0
    uint16_t* x_bf16 = nullptr;    ///< n_embd, bf16 bits

    // the qkv path
    float* qkv = nullptr;          ///< conv_channels
    float* conv_out = nullptr;     ///< conv_channels
    float* h = nullptr;            ///< conv_channels = silu(conv_out), and it IS q|k|v in place

    // the per-head scalars and the recurrence
    float* alpha = nullptr;        ///< h_v
    float* beta = nullptr;         ///< h_v
    float* gate = nullptr;         ///< h_v
    float* o = nullptr;            ///< h_v * S
    float* z = nullptr;            ///< value_dim
    float* y = nullptr;            ///< value_dim
    uint8_t* y_q8k = nullptr;      ///< value_dim, block_q8_K
    uint8_t* y_q8_0 = nullptr;     ///< value_dim, block_q8_0

    // the recurrent state, which outlives the call
    float* state = nullptr;        ///< S * h_v * S
    float* conv_state = nullptr;   ///< conv_channels * (d_conv-1)
};

/// Bytes needed for `b`, 16-byte aligned between regions.
uint64_t gdn_buffers_bytes(const ModelGeometry& g);
/// Carves `base` (which must be DEVICE memory) into `b`.  Returns the bytes used.
uint64_t gdn_buffers_init(const ModelGeometry& g, void* base, GdnBuffers& b);
/// Zeroes the recurrent and conv state, so a fresh sequence starts from the reference's own `zeros()`.
void gdn_buffers_zero_state(const GdnBuffers& b, const ModelGeometry& g, void* stream);

/// `Mixed` (n_embd f32) through one GDN layer to `out` (n_embd f32).  `state` and `conv_state` are updated.
///
/// `layer` selects the weights; `tables` must outlive the call.  Returns false and fills `err` if a tensor is
/// missing - which is a wiring mistake here rather than a shape problem, because `check_layer` already
/// asserted the shapes at load time.
bool gdn_layer(const WeightTable& tables, const ModelGeometry& g, int64_t layer, const GdnBuffers& b,
               const float* mixed, float* out, void* stream, std::string& err);

// ================================ the MoE block ================================

struct Doorbell;   // defined with the rest of the handoff, below; only a pointer is needed here

/// Buffers for the MoE block, on EVERY layer.
struct MoEBuffers {
    uint16_t* x_bf16 = nullptr;   ///< n_embd, for the ROUTER's weight, which is BF16
    uint16_t* x_f16 = nullptr;    ///< n_embd, for the SHARED EXPERT's weights (see the note in `moe_layer`)
    float* logits = nullptr;      ///< n_expert
    int* ids = nullptr;           ///< k
    float* weights = nullptr;     ///< k
    float* shared = nullptr;      ///< n_embd
    float* sh_scratch = nullptr;  ///< shared_expert_scratch_bytes(n_ff): the shared expert's own scratch
    uint8_t* x_q8_0 = nullptr;    ///< n_embd, block_q8_0 - the shared expert's weights want Q8_0 on the layers
                                  ///< whose types are Q2_0/Q4_0/Q5_0/Q8_0/IQ4_NL, which is most of them
    uint8_t* x_q8k = nullptr;     ///< n_embd, block_q8_K - and Q8_K on the K-quant ones
};

uint64_t moe_buffers_bytes(const ModelGeometry& g, int64_t k);
uint64_t moe_buffers_init(const ModelGeometry& g, int64_t k, void* base, MoEBuffers& b);

/// The MoE block: router -> (the caller's expert outputs) -> combine.
///
/// `parts` is (k, n_embd) device f32 and is **filled by the CALLER** - by the CPU pool and the VRAM-resident
/// experts.  That split is the architecture's, not a shortcut: the routed experts are the only part of the
/// model that does not fit in VRAM, and every design decision in this engine exists to keep them out of the
/// graph.  This function therefore owns the two ends - which experts, and how their outputs are combined -
/// and deliberately not the middle.
///
/// TWO ACTIVATION FORMATS AGAIN, and the router's is the one that matters most:
///
///   * `ffn_gate_inp.weight` is BF16, so the router sees a BF16 activation.  `ref/moe.py` L88-92 calls this
///     "the highest-consequence rounding in the model": the logits were measured **8.100e-03** off without it,
///     which is enough to FLIP a top-10 selection - and a flipped selection is not a small numeric error, it
///     is a different expert reading a different weight.
///   * the shared expert's weights are Q3_K/IQ4_XS/Q5_0 IQ-quants whose contract is Q8_K, which
///     `shared_expert` does not implement; it takes fp16.  That is the known gap its own header documents and
///     `docs/activation-contract.md` names as outstanding.
bool moe_layer(const WeightTable& tables, const ModelGeometry& g, int64_t layer, int64_t k, const MoEBuffers& b,
               const float* x, const float* parts, float* out, void* stream, std::string& err,
               const Doorbell* db = nullptr);

// ================================ THE MOE, SPLIT AT THE ROUTER ================================
//
// **THE SPLIT EXISTS BECAUSE THE ONE-LAYER DELAY WAS NOT A TIMING CHANGE.**  `moe_layer` is
// `route -> publish -> shared expert -> combine`, and `combine` multiplies `parts` by THIS layer's
// `b.weights`.  So `parts` must be THIS layer's expert outputs.  A host loop that hands layer `l`'s pool
// result to layer `l+1` computes `sum_j w_{l+1}[j] * expert_{ids_l,j}(x_l)` while the model is
// `sum_j w_l[j] * expert_{ids_l,j}(x_l)` - **both the expert selection and the input one layer stale while the
// weights are current** - and `session.hpp` states the contract it violates: the doorbell is "a change to WHEN
// the host runs, not to what this function computes".
//
// The one-layer delay was introduced for a real reason - it is the only way to overlap a CPU pool with a GPU
// that has already queued the combine - and the fix is to split the GRAPH, not to delay the answer:
//
//     block_layer_pre[l]   attn -> mixer -> gr_write -> ffn gr_read -> moe_route   [rings the doorbell]
//     ... the host runs the pool here, on the ids `moe_route` just published ...
//     block_layer_post[l]  moe_finish -> gr_write
//
// `post[l]` is launched AFTER the pool returns, so it sees layer `l`'s experts with layer `l`'s weights, and it
// is still a captured graph.  What it costs is the overlap: the window is now whatever GPU work follows the
// ring inside `pre[l]` - the shared expert and nothing else, because everything after the combine depends on
// `parts`.  **A per-layer CPU pool cannot be hidden behind a strictly serial residual chain**, which is why
// Phase 3's VRAM expert cache is the answer to the CPU term and the doorbell is not.

/// The routing half: bf16 activation, the router GEMV, top-k, and the doorbell publish.  Ends with the ring
/// rung, so a host that sees it knows `h_x_f`/`h_ids`/`h_weights` are in place.
bool moe_route(const WeightTable& tables, const ModelGeometry& g, int64_t layer, int64_t k, const MoEBuffers& b,
               const float* x, void* stream, std::string& err, const Doorbell* db = nullptr);
/// The verify window's n tokens routed at once (see layer.cpp); bitwise per token what `moe_route` gives.
bool moe_route_window(const WeightTable& tables, const ModelGeometry& g, int64_t layer, int64_t k, const MoEBuffers& b,
                      const float* x, float* logits, int32_t* ids, float* weights, int n, void* stream, std::string& err);

/// The finishing half: the shared expert and the combination.
///
/// **`parts` MUST BE THE EXPERTS THIS LAYER'S OWN ROUTER SELECTED.**  It is combined with `b.weights`, which
/// `moe_route` wrote for THIS layer.  Passing the previous layer's vectors is the L100 bug: it produces a
/// finite, fluent, deterministic token that is not the model's.
bool moe_finish(const WeightTable& tables, const ModelGeometry& g, int64_t layer, int64_t k, const MoEBuffers& b,
                const float* x, const float* parts, float* out, void* stream, std::string& err);

/// Plan v0.3 P3: the two parts of `moe_finish` separately.  `moe_shared` computes the shared expert into
/// `b.shared` and needs only `x`; `moe_combine_parts` combines `parts` with `b.weights` and `b.shared`.
/// `moe_finish` is exactly `moe_shared` followed by `moe_combine_parts`.
bool moe_shared(const WeightTable& tables, const ModelGeometry& g, int64_t layer, const MoEBuffers& b,
                const float* x, void* stream, std::string& err);
bool moe_combine_parts(const ModelGeometry& g, int64_t layer, int64_t k, const MoEBuffers& b, const float* parts,
                       float* out, void* stream, std::string& err);

/// Plan v0.3 P3 (default ON): the shared expert runs at the END OF `pre[l]`, after the doorbell has rung, so
/// the GPU computes it while the host runs the CPU pool; `post[l]` then only combines.  Same kernels on the
/// same inputs in the same stream order relative to their consumers, so the result is bitwise unchanged.
/// Must be set before capture.  `false` restores the shared expert inside `post[l]` (A/B arm).
void layer_set_shared_early(bool enabled);
bool layer_shared_early();

// ================================ the QSA mixer ================================

/// THE PERSISTENT STATE OF ONE QSA LAYER FOR ONE SEQUENCE: the KV cache, the indexer's pooled keys, the rope
/// tables, and the per-token step buffer.  None of it is scratch - it survives every token.
///
/// `max_cells` is the sequence capacity the arena was sized for, and it fixes the rope table's length: a table
/// shorter than the sequence would have `rope_neox_apply` read past it, which is a wrong rotation rather than a
/// fault.
struct QsaState {
    uint16_t* k_pool = nullptr;      ///< [page][kv_head][page_size][head_dim] fp16 (null in INT8 mode)
    uint16_t* v_pool = nullptr;
    /// Plan v0.3 P7: INT8 KV (qsa_set_kv_int8). Codes [page][kv_head][page_size][head_dim], one FP16 scale per 64.
    bool kv_int8 = false;
    int8_t* k_q = nullptr;
    int8_t* v_q = nullptr;
    uint16_t* k_scale = nullptr;
    uint16_t* v_scale = nullptr;
    /// PR #21: Q4_0 KV with Walsh-Hadamard rotation (qsa_set_kv_q4, kv_q4.hpp): 144 B per cell and head
    bool kv_q4 = false;
    uint8_t* k_q4 = nullptr;
    uint8_t* v_q4 = nullptr;
    /// Hybrid K8V4: K in INT8 (unrotated - the scores stay exact), V in rotated Q4_0 (kv_q4.hpp): 816 B per cell.
    /// Uses k_q/k_scale + v_q4. Mode 0 only (no KV streaming, no ring); under this setting the MTP drafter's
    /// ring state stays plain INT8, so the block movers never see the hybrid layout.
    bool kv_hybrid = false;
    /// K and V go through kv_q4.hpp's Walsh-Hadamard rotation before they are stored, the queries too, the output
    /// back: always for Q4_0, for INT8 by qsa_set_kv_int8_rotate (spreads outlier channels over the scale groups)
    bool kv_rot = false;
    int32_t* page_table = nullptr;   ///< (n_pages,) logical page -> physical page (-1: not resident, streamed)
    int64_t n_pages = 0;
    int64_t max_cells = 0;

    /// KV STREAMING (docs/kv-streaming-design.md, `kv_stream.hpp`). `kv_mode` 0: every page in VRAM, identity
    /// table (n_slots == n_pages, no host copy). 1: streamed - the authoritative K/V in `host`, `n_slots` pages
    /// in VRAM, `map` resolves the selection's blocks on device. 2: a ring (the MTP drafter's window) - `n_slots`
    /// pages at `block % n_slots`, with the host copy for a resume. The pool pointers above point at the slots.
    int kv_mode = 0;
    int64_t n_slots = 0;
    /// The elastic K/V (qsa_set_kv_elastic): this state's pools are in a VMM range mapped only as far as the
    /// context needs - its index in layer.cpp's registry; -1: carved from the arena as before.
    int32_t kv_elastic = -1;
    strata::kernels::KvHostPools host;
    strata::kernels::KvStreamMap map;
    int64_t idx_pooled_rows = 0;     ///< rows of `idx_pooled` (a ring, which has no indexer, keeps 2)

    float* idx_tail = nullptr;       ///< (idx_block - 1, idx_dim): the raw tail of the block being filled
    float* idx_dead = nullptr;       ///< (idx_dim,): the spare slot's key, CONSTANT for the sequence
    float* idx_pooled = nullptr;     ///< (max_cells/idx_block + 2, idx_dim)
    int32_t* idx_block_pos = nullptr;

    float* cos_tab = nullptr;        ///< (max_cells, n_rot/2), built on the HOST in float64
    bool owns_rope = false;          ///< built the table above (not borrowed with share_rope): it releases it
    float* sin_tab = nullptr;

    /// THE PER-TOKEN COUNTS, IN DEVICE MEMORY - the whole reason this layer can be a graph.  `qsa_step_fill`
    /// computes all four from the position, so one H2D per token updates every count every kernel needs.
    int32_t* step = nullptr;
    int32_t* attention_status = nullptr; ///< per-layer device status, captured D2H to host_step[kStepCount]
    /// The rope positions, (n_head,) int32, all equal to the current position.  `rope_neox_apply` reads its
    /// positions from the DEVICE, so a host scalar here would be an illegal access (and uncapturable).
    int32_t* pos_dev = nullptr;

    /// PINNED HOST STAGING AT FIXED ADDRESSES, and they are not an optimisation.  The uploads in `qsa_layer`
    /// copy FROM these, so a CUDA graph captures the SOURCE POINTER - and the first version of the layer filled
    /// a stack array and a local `std::vector` instead.  Those addresses are dead by the time the graph is
    /// replayed, which is this project's own recorded pitfall ("graph inputs live at fixed addresses") walked
    /// straight into: the replay would read whatever the stack held, and the step counts would be wrong in a way
    /// that looks like a working token.
    ///
    /// Pinned as well as fixed, because a pageable source would make `cudaMemcpyAsync` a synchronous staged
    /// copy and give back the overlap the doorbell protocol exists for.
    int32_t* host_step = nullptr;    ///< (kStepCount + 1,) pinned; final element is attention status
    int32_t* host_pos = nullptr;     ///< (n_head,) pinned
};

/// Plan v0.3 P7: the RoPE cos/sin table (max_cells x n_rot/2 x 2 floats, 64 MiB at 262K) is identical in every
/// QSA layer. `with_rope = false` sizes a state that borrows it; `share_rope` points `st` at another state's table
/// instead of building a copy (the session builds it once, in the first QSA layer).
uint64_t qsa_state_bytes(const ModelGeometry& g, int64_t max_cells, bool with_rope = true, int64_t ring_cells = 0);
/// KV streaming: keep `cells` cells of each QSA layer in VRAM and the rest in pinned host memory (0: all in VRAM,
/// the default). Set before sizing and initializing the session; a context that fits in `cells` is not streamed.
/// Also puts the MTP drafter's K/V in a ring of its window (`ring_cells` of qsa_state_bytes/init; -1 forces a fully
/// resident state).
void qsa_set_kv_resident(int64_t cells);
/// THE ELASTIC K/V (--kv-grow; vmm.hpp).  A mode-0 state's K/V pools get addresses for every cell of the context but
/// physical memory only for the first `init_cells`; `qsa_kv_elastic_grow` maps more as the context grows (from
/// chunks the expert cache gives up) and `qsa_kv_elastic_shrink` hands them back.  The addresses never move, so the
/// captured graphs stay valid.  Set before sizing and initializing the session and the drafter.
void qsa_set_kv_elastic(bool enabled, int64_t init_cells);
bool qsa_kv_elastic();
/// Cells every elastic state can hold now (INT64_MAX when none is elastic).
int64_t qsa_kv_elastic_cells();
/// Chunks still to map for every elastic state to hold `cells` cells.
int64_t qsa_kv_elastic_need(int64_t cells);
/// Maps them, each from `take()` (0: a new chunk), the new memory zeroed.  Synchronous; nothing may be running on
/// the device.  false: out of memory.
bool qsa_kv_elastic_grow(int64_t cells, const std::function<VmmChunk()>& take);
/// Unmaps the chunks past `cells` cells, each handed to `give`.  Returns how many.
int64_t qsa_kv_elastic_shrink(int64_t cells, const std::function<void(VmmChunk)>& give);
/// Physical bytes the elastic pools hold, and what all of them would at the full context.
uint64_t qsa_kv_elastic_mapped_bytes();
uint64_t qsa_kv_elastic_full_bytes();
int64_t qsa_kv_resident();
/// The fewest resident cells a streamed layer may have: one verify window's selections (8 queries x 2,051 cells
/// in whole blocks) must fit at once, with room to spare.
int64_t qsa_kv_resident_min();
/// Pinned host bytes the streamed states hold (their host copies).
uint64_t qsa_kv_host_bytes();
/// Plan v0.3 P7: store K/V as INT8 with FP16 scales per 64 values (half the VRAM of FP16). Set before sizing and
/// initializing the session; default off until gate G-C accepts it.
void qsa_set_kv_int8(bool enabled);
/// INT8 K/V through the Hadamard rotation (off by default: STRATA_KV_ROT=1)
void qsa_set_kv_int8_rotate(bool enabled);
bool qsa_kv_int8();
/// PR #21: store K/V as Q4_0 after a Hadamard rotation (`--kv q4_0`): 576 B per cell, vs 1,056 in INT8.
void qsa_set_kv_q4(bool enabled);
bool qsa_kv_q4();
/// Hybrid K8V4 (`--kv k8v4`): K in INT8, V in rotated Q4_0 - 816 B per cell; streams with --kv-resident too.
void qsa_set_kv_hybrid(bool enabled);
bool qsa_kv_hybrid();
/// The state's KV format for the block-moving functions of kv_stream.hpp (kKvF16 / kKvInt8 / kKvQ4 / kKvHybrid).
inline int qsa_kv_format(const QsaState& st) {
    if (st.kv_hybrid) return strata::kernels::kKvHybrid;   // K8V4: its own three runs (kv_stream.cu)
    return st.kv_q4 ? strata::kernels::kKvQ4 : st.kv_int8 ? strata::kernels::kKvInt8 : strata::kernels::kKvF16;
}
uint64_t qsa_state_init(const ModelGeometry& g, int64_t max_cells, void* base, QsaState& st,
                        const QsaState* share_rope = nullptr, int64_t ring_cells = 0);
/// KV streaming: the pools a reader sees (the VRAM slots) and, when streamed, make the selection's blocks resident.
strata::kernels::QsaAttnPools qsa_attn_pools(const QsaState& st);
void qsa_kv_resolve(const QsaState& st, const ModelGeometry& g, const int32_t* ids, const int32_t* steps, int64_t n_q,
                    int64_t cap, void* stream);
/// Zeroes the pools AND the indexer, so a fresh sequence matches the reference's own `zeros()`.  The KV pool
/// matters even for cells that are never attended, because `kv_gather` reads whatever the selection names.
void qsa_state_zero(const QsaState& st, const ModelGeometry& g, void* stream);

// ================================ PER-STAGE TIMING, DEBUG ONLY ================================
//
// Event pairs around the pieces of `block_layer_pre`/`_post`, recorded per (layer, stage) and read once at the
// end so that **NOTHING IS SYNCHRONISED DURING THE RUN** - a per-stage sync would serialise the pipeline and
// report the serialised time.  **CAPTURED GRAPHS CANNOT USE IT**: an event record inside a stream capture is
// silently dropped, so this is for the `--no-capture` path only.
//
// It exists because the engine spends ~1.047 ms per layer with the experts off while every cost model in
// `bench/` predicts less than half that, and the gap has to be attributed before it can be attacked.
//   slot 0 gr_read(attn)  1 the attention block  2 gr_write(attn)  3 gr_read(ffn)  4 router
//   slot 5 moe_finish     6 gr_write(ffn)
bool stage_timing_enable();
void stage_timing_name(int slot, const char* name);
void stage_timing_report(int64_t n_layers);
/// The same event pairs, reachable from `session.cpp`, for marks that sit OUTSIDE the captured graphs - the
/// two `cudaGraphLaunch` calls per layer.  Those are ordinary stream operations, so an event around them is
/// legal and is not captured, which is what lets the captured path's 1.047 ms/layer be split into the two
/// graphs' execution and the gap the host spends between them.  Slots 16 and 17 are reserved for it; slots
/// 0-15 are the in-layer marks and are only meaningful with `--no-capture`.
void stage_mark_begin(int64_t layer, int slot, void* stream);
void stage_mark_end(int64_t layer, int slot, void* stream);

/// Scratch for one QSA layer.  Every buffer here is overwritten by every token.
struct QsaBuffers {
    uint8_t* x_q8_0 = nullptr;     ///< n_embd, block_q8_0 - `attn_q` is Q2_0, whose vec_dot_type is Q8_0
    uint8_t* x_q8k = nullptr;      ///< n_embd, block_q8_K - the three K-quant projections
    uint16_t* x_bf16 = nullptr;    ///< n_embd - the two BF16 indexer projections
    float* q_full = nullptr;       ///< n_head * 2 * head_dim: the raw `attn_q` output, q | gate per head
    float* qcur = nullptr;         ///< n_head * head_dim: the FIRST half of each head, contiguously
    float* kcur = nullptr;         ///< n_head_kv * head_dim
    float* vcur = nullptr;         ///< n_head_kv * head_dim
    float* idx_raw = nullptr;      ///< idx_key_dim: NEVER normed and NEVER rotated
    float* q_idx = nullptr;        ///< idx_q_heads * idx_key_dim
    float* cell_scores = nullptr;  ///< max_cells
    int32_t* ids = nullptr;        ///< cap = qsa_selection_width(kTopkMaxCells, s)
    uint16_t* k_scratch = nullptr; ///< cap * n_head_kv * head_dim
    uint16_t* v_scratch = nullptr;
    float* attn = nullptr;         ///< n_head * head_dim
    uint16_t* attn16 = nullptr;    ///< n_head * head_dim, for a fp16-activation consumer
    /// `attn * sigmoid(gate)` in f32 and then in Q8_K - what `attn_output` ACTUALLY wants.  Its weights are
    /// Q4_K/Q5_K/Q6_K, whose `vec_dot_type` is Q8_K, and the fp16 path is worth 0.66-1.41% against the
    /// reference.  `docs/activation-contract.md`.
    float* attn32 = nullptr;
    uint8_t* attn_q8k = nullptr;
    float* attn_scratch = nullptr;   ///< plan v0.3 P3: split-K decode attention partials
};

uint64_t qsa_buffers_bytes(const ModelGeometry& g, int64_t max_cells);
uint64_t qsa_buffers_init(const ModelGeometry& g, int64_t max_cells, void* base, QsaBuffers& b);

/// `x` (n_embd f32) through one QSA layer to `out` (n_embd f32) at sequence position `pos`.
///
/// `pos_base` is the sequence's FIRST cell's position, which the indexer needs and cannot derive - equating a
/// cell's position with its index is only right for a sequence starting at 0.
///
/// THE OP ORDER IS `ref/model.py::_qsa`, and two things in it are easy to read past:
///
///   * the indexer's raw key is appended BEFORE any norm or rotation, and it is the raw `indexer.k_proj`
///     output that is pooled and rotated LATER, once per block and at the block's FIRST cell;
///   * `q_full` is split into q and gate by HALVES of each head's 2*head_dim block, contiguous, not
///     interleaved - `qsa_gate_apply` reads the second half and `qcur` is copied out of the first.
///
/// THE THREE ACTIVATION FORMATS FOLLOW `docs/activation-contract.md`, and they are not interchangeable:
/// `attn_q` is Q2_0 (Q8_0 activation), `attn_k`/`attn_v`/`attn_output` are K-quants (Q8_K), and the two
/// indexer projections are BF16 (bf16).
bool qsa_layer(const WeightTable& tables, const ModelGeometry& g, int64_t layer, int64_t pos, int32_t pos_base,
               const QsaState& st, const QsaBuffers& b, const float* x, float* out, void* stream,
               std::string& err, float* dump = nullptr);

// ================================ THE DOORBELL ================================

/// THE HANDOFF: `x_f` and the miss list reach the CPU WHILE THE LAYER IS STILL RUNNING.
///
/// Everything about the shape of this struct comes from two measurements, and neither was a guess:
///
///   * **THE WAIT MUST POLL THE DRIVER.**  Round 195: a host spin that only reads a memory location never runs
///     the kernel - 5,907,703 spins over 500 ms and the datum flips the instant `cudaStreamSynchronize` is
///     called and never before, because on Windows the driver BATCHES command submission and a memory-only
///     spin gives it no reason to flush.  So `h_seq` is read AND `cudaEventQuery` is called, which is a query
///     and not a blocking sync - P2.X3's "zero synchronization calls in the layer loop" still holds.
///   * **THE WRITE MUST BE CAPTURABLE, AND A KERNEL IS.**  Round 199: `cudaEventRecord` inside a capture is
///     SILENTLY DROPPED - 2 nodes for two kernels plus an event record, and the event never completed across a
///     41 ms graph.  A mid-graph doorbell KERNEL is captured normally, and round 209 measured the host seeing a
///     pinned handoff **0.050 ms** into a 39.8 ms graph.
///
/// The memory is MAPPED PINNED, so it has one address for the device and another for the host and both refer
/// to the same bytes.  Pinned as well as mapped, because a pageable destination would turn the device's write
/// into a staged copy and give back the overlap this exists for.
struct Doorbell {
    int64_t n_embd = 0;
    int64_t k = 0;
    // the HOST's view
    float* h_x_f = nullptr;        ///< n_embd
    int32_t* h_ids = nullptr;      ///< k
    float* h_weights = nullptr;    ///< k
    uint32_t* h_seq = nullptr;     ///< 1
    /// Plan v0.3 P3 token graph: the HOST's answer.  The host writes the ring value it has served (after the
    /// pool's output is in place); the device's `doorbell_wait` spins until the flag equals the ring.
    uint32_t* h_flag = nullptr;    ///< 1
    // the DEVICE's view of the same bytes
    float* d_x_f = nullptr;
    int32_t* d_ids = nullptr;
    float* d_weights = nullptr;
    uint32_t* d_seq = nullptr;
    uint32_t* d_flag = nullptr;
};

/// Allocates the four mapped-pinned regions.  Returns the bytes, or 0 on failure.
uint64_t doorbell_init(const ModelGeometry& g, int64_t k, Doorbell& db);
void doorbell_free(Doorbell& db);
/// Zeroes `h_seq` (and `h_flag`), so a caller can wait for it to become `expected` from a known state.
void doorbell_reset(const Doorbell& db);

// ================================ ONE WHOLE BLOCK ================================

/// The gated-residual stack plus the two buffers the halves hand to each other.
///
/// `R` is (hc, n_embd) and is BOTH the input and the output: `hc` residual streams, each n_embd wide, which is
/// what `gr_read`/`gr_write` exist for.  It is not one vector with a gate.
struct BlockBuffers {
    float* R = nullptr;                       ///< (hc, n_embd)
    float* mixed = nullptr;                   ///< n_embd: what the mixer consumes
    float* block_out = nullptr;               ///< n_embd: what a half produces
    float* inject = nullptr;                  ///< hc: `gr_read`'s per-stream injection, reused per half
    /// Plan v0.3 P3 fused GR: the FFN half's injection lives here (the attention half's stays in `inject`),
    /// because the fused read of one half reads the other half's injection while writing its own.
    float* inject2 = nullptr;                 ///< hc
    float* gr_rs = nullptr;                   ///< hc: the fused read's per-stream 1/rms
    uint8_t* head_q8k = nullptr;              ///< `q8k_bytes(n_embd)`: the head's Q8_K activation image
    strata::kernels::GrWorkspace gr;          ///< `gr_read`'s scratch

    /// THE HALF-LEVEL C1 ORACLE, and it is a HOST buffer of `(2 * n_embd + 2 * hc + n_head * head_dim)` floats
    /// PER LAYER: `[0, n_embd)` the attention half's `block_out`, `[n_embd, 2*n_embd)` the MoE half's, the two
    /// `hc`-vectors of `inject`, and finally the QSA layers' PRE-GATE attention (`n_head * head_dim`, unused by
    /// GDN).  The residual ladder (`session_loop`'s `dump_layers`) says WHICH LAYER diverges; this says which
    /// stage, and it is the only way to separate the reference's `linear_attn_out-<l>` from its `ffn_out-<l>`
    /// without re-deriving one of them.  The copies are issued from inside `block_layer_pre`/`_post`, so they
    /// are recorded as nodes of the per-layer captured graph and replay with the layer they belong to.  Null
    /// means no dumping, which is the default and costs nothing.
    float* dump = nullptr;
};

uint64_t block_buffers_bytes(const ModelGeometry& g);

/// Plan v0.3 P3: the hyper-connection read as two fused kernels with the previous half's write folded in
/// (`strata/kernels/fused_gr.hpp`).  Requires the native MMVF contract (FP32 activations).  The residual `R`
/// then lags by one pending write between halves; it is materialised before the layer-1 PLE and after the last
/// layer.  Set before capture; not for the per-stage or dump measurements.
void layer_set_fused_gr(bool enabled);
bool layer_fused_gr();

/// Plan v0.3 P3/P7: QSA attention by the split-K kernel reading the KV pools directly (default ON; `false` keeps
/// the gather + one-block-per-head kernel).  Set before capture.
void layer_set_fast_attn(bool enabled);
/// Plan v0.3 P3: doorbell payload + ring as one kernel, QSA step staging read by a kernel from mapped memory
/// (default ON; `false` keeps the memcpy nodes).  Set before capture.
void layer_set_publish_kernel(bool enabled);
/// Plan v0.3 P3: GDN step + output norm as one coalesced kernel under the native contract (default ON).
void layer_set_fused_gdn(bool enabled);
/// Plan v0.3 P7: QSA block scores in FP32 and a radix top-k over blocks (default ON; `false` = the FP64 row scores
/// and the bit-serial cell top-k).  Set before capture.
void layer_set_fast_select(bool enabled);
/// Plan v0.3 P6: whether the current decode configuration is the one the speculative verify window reproduces
/// bit for bit (native projections, fused GR and GDN, split-K attention, block selection, native indexer).
bool layer_verify_compatible(std::string& why);
uint64_t block_buffers_init(const ModelGeometry& g, void* base, BlockBuffers& b);

// ================================ THE TWO ENDS OF A TOKEN ================================

/// THE EMBEDDING LOOKUP, done as a COLUMN DEQUANT rather than a GEMV.
///
/// `token_embd.weight` is `[n_embd, n_vocab]`, so `out[i] = W[i][token]` - one column, `n_embd` values.  Running
/// `output`-style arithmetic on a one-hot activation would produce the same numbers with 248320x the work.
///
/// **THE DECODE IS TRANSCRIBED FROM `s_gemv_q8k_split` AND MUST STAY BIT-IDENTICAL TO IT**: the code is the
/// `CODE_BITS`-wide field, least-significant first, at bit `(i % PER_BYTE) * CODE_BITS` of byte `i / PER_BYTE`;
/// the value is `(code + code_bias) * scales[i / group_elems] + offsets[i / group_elems]`; and the column's
/// planes start at `token * n_in / PER_BYTE` and `token * n_in / group_elems`.  A second decoder that drifts
/// from the kernel's would make the prompt embed differently from the tokens generated after it.
bool embed_row(const WeightTable& tables, const ModelGeometry& g, int64_t token, float* out_dev, void* stream,
               std::string& err);

/// `ref/model.py` L520: `fin = GR.gr_read(R, output_hc_*..., inject=None, rms_eps); logits = output @ fin`.
///
/// `R` is read from `bb.R` and the result goes to `logits`, which the caller sizes at `output.weight`'s `ne1`
/// (the vocabulary).  `bb.head_q8k` and `bb.mixed` are scratch.
bool lm_head(const WeightTable& tables, const ModelGeometry& g, const BlockBuffers& bb, float* logits,
             void* stream, std::string& err);

/// Compute the final residual mix in bb.mixed, before the output projection. This is shared by the
/// canonical and experimental native head so only the projection's arithmetic changes in their A/B test.
bool lm_head_mix(const WeightTable& tables, const ModelGeometry& g, const BlockBuffers& bb,
                 void* stream, std::string& err);

/// THE PER-LAYER OP ORDER of `phases/phase-2-correct-engine.md` P2.S5, architecture Ă„â€šĂ˘â‚¬ĹľÄ‚ËĂ˘â€šÂ¬ÄąË‡Ă„â€šĂ‹ÂÄ‚ËĂ˘â‚¬ĹˇĂ‚Â¬Ă„Ä…Ă‹â€ˇÄ‚â€žĂ˘â‚¬ĹˇÄ‚ËĂ˘â€šÂ¬ÄąË‡Ă„â€šĂ˘â‚¬ĹˇÄ‚â€šĂ‚Â§6.1, for ONE layer:
///
///     gr_read (attn)  ->  mixer  ->  gr_write (attn)
///     gr_read (ffn)   ->  MoE    ->  gr_write (ffn)
///
/// `layer` picks the mixer: `is_qsa_layer` selects `qsa_layer`, otherwise `gdn_layer`.  `pos`/`pos_base` reach
/// the QSA layer and are ignored by GDN, which has no position argument at all - its whole per-token input is
/// `x` and what advances between tokens is its recurrent state.
///
/// **`parts` IS THE CALLER'S**, and that is the architecture's split rather than an omission: the routed experts
/// are the only part of the model that does not fit in VRAM, so the CPU pool and the VRAM-resident experts fill
/// `parts` and this function owns the two ends.  In P2.S5 everything is a miss, so a caller passes whatever the
/// pool produced; the doorbell protocol is what lets it be produced DURING this layer rather than before it.
// ================================ THE PLE (LAYER 1'S PER-LAYER EMBEDDING) ================================
//
// **THIS MODULE EXISTED, WAS PARITY-TESTED, AND WAS NEVER CALLED - WHICH IS WHY GATE C1 FAILED.**  Layer 1 is
// the only layer with `blk.1.ple_*` tensors (six of them) and `block_layer` ran GDN/QSA + GR + MoE without it,
// so the engine computed a different model.  `ple_parity` verifies all three pieces against things that are
// not this project's code - the hash against `ref/ngram.py` with the artifact's own constants, the IQ4_NL row
// read against numpy over the original GGUF, and the block against ggml's own graph - and none of that helps
// if nothing calls it.
//
// THE TABLE IS NOT IN THE PACK.  `per_layer_token_embd.weight` is 51.2e9 elements (28.8 GB) of IQ4_NL in the
// ORIGINAL second GGUF shard, and it is the only tensor this engine reads from the GGUF rather than from the
// canonical pack.  It is also why the PLE is cheap: sixteen 90-byte rows per token is 1440 bytes of traffic,
// against the experts' 664 MB.
struct PleRun {
    /// Null when the PLE is not wired (no table, or a pack without `blk.1.ple_*`).  Then `block_layer_pre`
    /// SKIPS the module entirely, which is what every measurement before this change did.
    strata::kernels::PleTable* table = nullptr;
    strata::kernels::PleWeights w;
    /// The session's PLE state, by pointer so `block_layer_pre` needs no `SessionState`.
    const int32_t* token = nullptr;    ///< the token being decoded
    const int32_t* prev = nullptr;     ///< (2,) predecessors, OLDEST FIRST, `-1` where there is none
    float* hist = nullptr;             ///< NG_HIST x hc_dim NORMALIZED history, row-fastest
    /// The gathered 2560-wide row vector on the DEVICE, and the host staging it is copied from: `gather`
    /// writes HOST memory (the table is a host mapping) and the block wants the device.
    float* emb_dev = nullptr;
    float* emb_host = nullptr;
    /// Internal workspace plus a disjoint normalized export for the next history row.
    /// Allocate ple_run_scratch_bytes(), not merely the lower-level ple_block_scratch_bytes().
    float* scratch = nullptr;
    strata::kernels::PleConsts consts;
    bool ready() const {
        return table != nullptr && table->is_open() && hist != nullptr && emb_dev != nullptr &&
               emb_host != nullptr && scratch != nullptr && token != nullptr && prev != nullptr &&
               ((w.key_codes != nullptr && w.key_scales != nullptr) || w.key_bf16 != nullptr ||
                w.key_native_data != nullptr) &&
               w.value_bf16 != nullptr &&
               w.norm_key != nullptr && w.norm_query != nullptr && w.norm_conv != nullptr &&
               w.conv1d_f16 != nullptr;
    }
};

/// **CALLED ONCE PER TOKEN, FROM THE DRIVER, *OUTSIDE* ANY CAPTURE.**  The n-gram hash and the table gather
/// are host operations, and `block_layer_pre` is captured - so doing them there would run them once, at capture
/// time, and replay the same rows for every token forever.  This is the doorbell's cloned-literal bug in a
/// different coat: silent, finite, and stable.
bool ple_stage_token(const PleRun& p, void* stream, std::string& err);

/// Plan v0.3 P2: `ple_stage_token` in two halves, so the SSD reads overlap other host work.
/// `ple_issue_token` hashes the token and starts its 16 row reads; `ple_finish_token` waits for them,
/// dequantizes and queues the upload. Both are driver-side, outside any capture, like `ple_stage_token`.
bool ple_issue_token(const PleRun& p, std::string& err);
bool ple_finish_token(const PleRun& p, void* stream, std::string& err);

/// Internal PLE workspace plus one separate hc_dim normalized row for advancing history.
uint64_t ple_run_scratch_bytes();

/// `ple` is the layer-1 PLE, or null to skip it (which is what every measurement before LEDGER L123 did).
/// It is applied INSIDE `pre[1]`, before layer 1's own `gr_read`, because the block's output is
/// `hidden + gated + silu(conv)` on the residual itself - it is a residual UPDATE, and the architecture puts
/// the gathered rows in place "before layer 2".
bool block_layer(const WeightTable& tables, const ModelGeometry& g, int64_t layer, int64_t pos, int32_t pos_base,
                 const GdnBuffers& gb, const QsaState& qst, const QsaBuffers& qb, const MoEBuffers& mb,
                 int64_t k, const BlockBuffers& bb, const float* parts, void* stream, std::string& err,
                 const Doorbell* db = nullptr, const PleRun* ple = nullptr);

/// **THE BLOCK, SPLIT AT THE ROUTER, SO A HOST LOOP CAN PUT THE CPU POOL IN THE MIDDLE.**
///
///     pre[l]   gr_read(attn) -> mixer -> gr_write(attn) -> gr_read(ffn) -> moe_route   [rings the doorbell]
///     post[l]  moe_finish -> gr_write(ffn)
///
/// `parts` reaches only `post`, and it must be THIS layer's experts.  The two calls are NOT independent: `post`
/// reads `bb.mixed` and `bb.inject` exactly as `pre` left them, so nothing may run between them that touches the
/// block buffers - and in particular `pre[l+1]` may NOT be launched between them, because its own `gr_read`
/// overwrites both.  That constraint is why the doorbell's window is the shared expert and nothing more, and
/// why the CPU expert term is answered by Phase 3's VRAM cache rather than by this pipeline.
///
/// **`half` EXISTS SO THE CAPTURED GRAPH CAN BE SPLIT FOR MEASUREMENT, AND IT IS THE ONLY REASON.**
///
/// The plan's R3 items - GR, GDN, the router - are each worth several milliseconds and there has never been a
/// per-stage breakdown of the CAPTURED graph to choose between them.  The `--no-capture` table cannot supply
/// one: round 312 showed its `moe_finish` figure was a host-gap artefact, four times the captured cost of the
/// same work.  An event recorded BETWEEN two `cudaGraphLaunch` calls IS valid, so the answer is to capture the
/// layer as more than one graph and time them from outside.
///
/// `half` selects a prefix of the stages, and the seam is the one this file already documents:
///
///     1 = stages 0..2   gr_read -> attention -> gr_write        the MIXER
///     2 = stages 3..4   gr_read -> moe_route                    the FFN front and the ROUTER
///     0 = both, which is what every caller before this used and what `session_loop` still uses
///
/// **A half is not a free-standing call.**  `2` reads what `1` left in `bb.mixed`/`bb.inject`, exactly as
/// `block_layer_post` does; the two are only meaningful replayed in order on one stream, which is what
/// `session_replay_stages` does.
bool block_layer_pre(const WeightTable& tables, const ModelGeometry& g, int64_t layer, int64_t pos,
                     int32_t pos_base, const GdnBuffers& gb, const QsaState& qst, const QsaBuffers& qb,
                     const MoEBuffers& mb, int64_t k, const BlockBuffers& bb, void* stream, std::string& err,
                     const Doorbell* db = nullptr, const PleRun* ple = nullptr, int half = 0,
                     int stage_prefix = 0);

bool block_layer_post(const WeightTable& tables, const ModelGeometry& g, int64_t layer, int64_t k,
                      const MoEBuffers& mb, const BlockBuffers& bb, const float* parts, void* stream,
                      std::string& err);

}  // namespace strata::core
