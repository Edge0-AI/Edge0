// include/strata/core/session.hpp - ONE TOKEN through all 48 layers, and the doorbell protocol between them.
//
// P2.S5's host loop, and the first place the engine runs a whole forward pass rather than one block.  What it
// owns is the ORDER across layers and the state that outlives a layer:
//
//   * the gated residual stack `R` - (hc, n_embd), which every block reads and writes;
//   * 12 QSA states (KV cache + indexer) and 36 GDN states (recurrence + conv history), because `is_qsa_layer`
//     splits the 48 layers 12/36 and the two kinds of state have nothing in common;
//   * the per-token expert handoff, which is where the CPU pool meets the GPU.
//
// WHY THE LAYER GRAPHS ARE CAPTURED AND THE LOOP IS NOT.  A graph bakes in its arguments, so the graph unit is
// ONE LAYER: everything per-token (the position, the step counts) arrives through fixed-address device buffers
// (rounds 211-214), and the HOST decides when each layer runs.  That decision is the whole point - it is what
// lets the CPU expert pool start while the GPU is still working on the next layer.
#pragma once

#include "strata/core/hit_hook.hpp"
#include "strata/core/layer.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// Everything a sequence needs that is NOT a weight: the per-layer state, the block scratch, and the pinned
/// handoff between the GPU and the CPU expert pool.
struct SessionState {
    int64_t max_cells = 0;

    GdnBuffers gdn;                 ///< the 36 GDN layers share one set of scratch; their STATE is per layer
    float* gdn_state = nullptr;     ///< (n_gdn_layers, gdn_state_floats)

    QsaState* qsa_states = nullptr;      ///< one per QSA layer
    void* qsa_state_arena = nullptr;
    QsaBuffers qsa_bufs;                 ///< scratch, shared across the 12 (they never run concurrently)
    void* qsa_buf_arena = nullptr;

    MoEBuffers moe;
    void* moe_arena = nullptr;
    BlockBuffers block;
    void* block_arena = nullptr;

    float* R = nullptr;             ///< alias of `block.R`, named for what it means at this level
    int64_t k = 10;                 ///< experts per token

    // **THE LAYER-RANGE CARVE (multi-GPU).**  A split stage's session used to carve state for ALL 48 layers
    // (every QSA KV pool at max_cells, every GDN recurrence) while running only its slice - the same
    // whole-model-sized-buffer disease the QSA chunked-prefill PR cured upstream.  A session now owns
    // [layer_lo, layer_hi): `qsa_states`/`gdn_state` keep GLOBAL ordinal indexing, the arrays simply hold
    // fewer rows, so per-layer consumers subtract `gdn_ord0`, and every old model-level `qsa_states[0]` use
    // (the shared RoPE table, the prefill staging identity, the MTP rope borrow) reads `qsa_states[primary]`.
    // A range with no QSA layer still carves ONE primary state so those model-level uses stay valid.
    int64_t layer_lo = 0, layer_hi = 0;   ///< resolved by `session_init` (hi = n_layers for the full range)
    int64_t qsa_ord0 = 0;                 ///< global QSA ordinal of the first allocated state
    int64_t qsa_alloc = 0;                ///< allocated QSA states (>= 1 whenever the model has any)
    int64_t gdn_ord0 = 0;                 ///< global GDN ordinal of `gdn_state` row 0
    int64_t gdn_alloc = 0;                ///< allocated GDN rows
    int qsa_primary() const { return (int) qsa_ord0; }
    /// The doorbell the host loop polls.  Null until a caller provides one - the engine runs without it, and
    /// neither `session_token` nor `session_replay` looks at it.
    const Doorbell* db = nullptr;

    // ================================ THE PLE, AND IT IS SEQUENCE STATE ================================
    //
    // **THE PLE IS A LAYER-1 MODULE AND THIS ENGINE DID NOT RUN IT, WHICH IS WHY C1 FAILED** (LEDGER L123).
    // Every part of it was already built and parity-tested - the hash against `ref/ngram.py` with the
    // artifact's own constants, the IQ4_NL row read against numpy over the original GGUF, and the block
    // against ggml's own graph - and none of it was ever called from the model path.  These fields are what
    // was missing on the state side.
    //
    // `ple_prev` is the last two tokens OLDEST FIRST with `-1` where there is none, which is exactly
    // `ngram_rows`' `prev` argument; `ple_token` is the token being decoded.  **They are advanced by the
    // DRIVER, once per token**, because the hash is a function of the sequence and not of a layer.
    int32_t ple_prev[2] = {-1, -1};
    int32_t ple_token = -1;
    /// `NG_HIST` rows of `hc_dim` NORMALIZED history, **ROW-FASTEST**: `ple_hist[row + NG_HIST*channel]`.
    /// That is `ggml_reshape_3d(state, d_conv-1, conv_channels, n_seqs)`, not a layout choice - the transposed
    /// reading fails ggml's own assert (`ple_layer_xcheck.cpp` L79-83).
    float* ple_hist = nullptr;
    /// **THE WHOLE PLE RUN LIVES HERE, AND THAT IS NOT A CONVENIENCE.**  Its history and its token window are
    /// sequence state, so the session is where they belong - and putting the rest of it here too means
    /// `session_capture`/`session_loop`/`session_token` need no new parameter to reach it.  A `PleRun` whose
    /// `ready()` is false is skipped, which is what every caller had before LEDGER L123.
    PleRun ple;
};

/// Bytes for a whole session at `max_cells` of context.  Every layer's state is sized at once, because P2.T10
/// requires ZERO token-path allocations - a `cudaMalloc` that happened on the first token of a longer sequence
/// would satisfy every test here and fail that one.
///
/// [layer_lo, layer_hi) carves only that range's per-layer state (a split stage runs a slice of the model);
/// the default full range is byte-identical to the old whole-model carve.  Pure arithmetic - safe to call for
/// a candidate range before anything is allocated, which is how the layer-split search prices a placement.
uint64_t session_bytes(const ModelGeometry& g, int64_t max_cells, int64_t k, int64_t layer_lo = 0,
                       int64_t layer_hi = -1);
/// Carves `base` (DEVICE memory) into `s`.  Returns the bytes used.  Same range convention as `session_bytes`.
uint64_t session_init(const ModelGeometry& g, int64_t max_cells, int64_t k, void* base, SessionState& s,
                      int64_t layer_lo = 0, int64_t layer_hi = -1);
/// Before the memory `session_init` carved is freed: forgets what points into it from outside the session (the
/// rope kernels' registered angle table, #280), so a later session never rotates by freed memory.
void session_release(SessionState& s);
/// Zeroes every layer's state - the residual to `R_init`, everything else to zero, so a fresh sequence starts
/// from the reference's own `zeros()`.
void session_zero(SessionState& s, const ModelGeometry& g, const float* R_init, void* stream);

/// One token: layers 0..47 in order, each a `block_layer`, and the residual is updated in place.
///
/// `parts` is (k, n_embd) DEVICE memory, filled by the caller - by the CPU expert pool and the VRAM-resident
/// experts.  In this phase `moe_layer` consumes it directly; the doorbell protocol's job is to let it be
/// produced DURING the layer rather than before it, which is a change to *when* the host runs, not to what this
/// function computes.
///
/// `sync_every_layer` inserts a device synchronisation and an error check after EVERY layer.  That is the
/// `--sync-every-layer` debug mode, and it is deliberately a parameter rather than a compile-time choice: the
/// errors it catches (a kernel reading a buffer that a later layer writes) are invisible otherwise, and the
/// synchronisation it costs is exactly what P2.X3 forbids in the fast path.
bool session_token(const WeightTable& tables, const ModelGeometry& g, int64_t pos, int32_t pos_base,
                   SessionState& s, const float* parts, void* stream, bool sync_every_layer,
                   std::string& err);

// ================================ the captured form ================================

/// ONE GRAPH PER LAYER, because a graph bakes in its POINTERS and every layer has different weights AND
/// different state.  That is not a limitation to work around - it is what makes the per-layer graph the right
/// unit: the host decides when each layer runs, which is what lets the CPU expert pool start while the GPU is
/// still working on the next one.
///
/// WHY THIS IS WORTH DOING, MEASURED: **the engine is launch-bound, not bandwidth-bound.**  Every kernel in
/// `gdn_layer` measured ~0.028 ms - including `gdn_gate`, which processes FORTY-EIGHT elements, and a kernel
/// cannot execute for 28 us.  That is what a null-stream launch costs on this driver, and `gdn_layer` makes ~19
/// of them while a block makes ~45.  Replaying the 43-node block graph measured **1.585 ms against 2.393 ms**
/// for the same work as direct launches - a 1.51x that is pure launch overhead.
struct SessionGraphs {
    cudaGraphExec_t* execs = nullptr;   ///< one per layer, in layer order
    int64_t n = 0;
    /// **THE SECOND GRAPH PER LAYER: `moe_finish` + `gr_write`, launched AFTER the host has run the pool.**
    ///
    /// Without it the loop has to hand layer `l`'s expert vectors to layer `l+1`, which multiplies them by
    /// layer `l+1`'s router weights - see `layer.hpp`'s note on `moe_route`.  With it, `post[l]` is launched
    /// after the pool returns, so layer `l`'s experts are combined with layer `l`'s weights.
    ///
    /// The cost is the overlap, and it is worth stating: the window is now whatever GPU work follows the ring
    /// inside `pre[l]` - the shared expert, and nothing else, because everything after the combine depends on
    /// `parts`.  **A per-layer CPU pool cannot be hidden behind a strictly serial residual chain**, which is
    /// why the CPU term is answered by Phase 3's VRAM cache and not by this pipeline.
    cudaGraphExec_t* posts = nullptr;
    bool captured = false;
    /// THE DEVICE ADDRESS THE GRAPHS WERE CAPTURED WITH.  The host loop copies each layer's expert outputs
    /// here before launching the next layer, and a graph bakes the POINTER, so it has to be this one.
    float* parts_dev = nullptr;

    /// **DIAGNOSTIC, AND IT IS THE ONE THAT SAYS WHETHER THE DOORBELL IS DECORATIVE.**
    ///
    /// Counts the layers where the ring was observed while the graph was STILL RUNNING - i.e. the layers where
    /// the pool actually had a window to work in.  A nonzero `h_seq` read at the same moment
    /// `cudaEventQuery` returns `cudaSuccess` means the layer was already over, and the pool then runs with the
    /// GPU IDLE, which makes the whole pipeline equivalent to a serialised one.
    ///
    /// That distinction is invisible in the loop's RETURN VALUE - both paths produce a correct token - and it
    /// is invisible in a per-layer timing, because the layer still takes the same time either way.  It shows up
    /// only as "the overlap hides nothing", which is why it is counted here.
    int64_t rings_mid_graph = 0;

    /// **THE WINDOW, MEASURED RATHER THAN ASSUMED.**  Milliseconds from `cudaGraphLaunch` to the moment the
    /// ring was observed, summed over the loop.  The pool's window is `layer period - this`, and that is the
    /// entire budget the overlap has: if the CPU's per-layer work exceeds it, the GPU goes idle for the
    /// difference on every layer and a "pipelined" loop measures the same as a serialised one.
    ///
    /// It is NOT the same as the doorbell's position inside the graph.  It includes the launch itself, the
    /// driver's submission, and however long the host took to notice - all of which are real and none of which
    /// the pool gets to use.
    double ms_to_ring = 0;

    // ================================ R0.9: THE LAYER AS THREE GRAPHS ================================
    //
    // **THE CAPTURED PER-STAGE TABLE THE PROTOCOL HAS DEMANDED SINCE R0 AND NOBODY HAS HAD.**  Round 309's
    // came from `--no-capture`, and round 312 showed its `moe_finish` figure was a host-gap artefact four
    // times the captured cost of the same work - so the plan has been choosing between R3.3, R3.4 and R3.5
    // without knowing what any of them is worth.
    //
    // An event recorded BETWEEN two `cudaGraphLaunch` calls is not inside a capture and is therefore valid,
    // so the layer is captured at its own documented seam (`block_layer_pre`'s `half`) and timed from outside.
    /// **R0.11: FIVE PREFIX GRAPHS, ADDED BESIDE `preA`/`preB` RATHER THAN REPLACING THEM.**
    /// `preP[k]` (k = 1..5) runs stages `0..k-1`. Consecutive differences are the per-stage times, which is
    /// what prices the two `gr_read`s and the `gr_write` (R3.3) against GDN's interior (R3.4). A prefix and not
    /// a range, because the stages are a chain: stage 1 reads what stage 0 wrote and stage 2 mutates the
    /// residual in place.
    ///
    /// **THE FIFTH PREFIX USED TO BE `execs[l]`, AND THAT WAS THE BUG ROUND 324 COULD NOT FIND.**  Prefix 5
    /// has ~7 more nodes than prefix 4, so the two differ in launch setup as well as in execution - and the
    /// difference attributed ALL of it to stage 4, the router. Capturing prefix 5 the same way as the others
    /// makes the two comparable and the difference purely stage 4's.
    cudaGraphExec_t* preP[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    cudaGraphExec_t* preA = nullptr;   ///< `half == 1`: gr_read -> attention -> gr_write   (the MIXER)
    cudaGraphExec_t* preB = nullptr;   ///< `half == 2`: gr_read -> moe_route              (FFN front + ROUTER)
    bool split_captured = false;

    /// How many times `session_loop` has run.  `rings_mid_graph` and `ms_to_ring` are CUMULATIVE and the loop
    /// is called once per token, so anything reporting a ratio needs this - and prefill runs the loop too, so
    /// "per generated token" is the wrong denominator.
    int64_t calls_total = 0;

    /// **WHERE THE PER-LAYER ROUND TRIP GOES, AND `ms_to_ring` ALONE CANNOT SAY.**
    ///
    /// `--no-pool` measures **38.73 ms/token against a 26.32 ms pure-GPU floor**, so ~12.4 ms/token - a quarter
    /// of the token - is spent in the loop with no expert work at all: 0.258 ms per layer. `ms_to_ring` covers
    /// only the first half of that; it stops when the ring is seen, and everything the host then does before
    /// the next `pre` is launched is invisible to it.
    ///
    /// So the round trip is split at the ring, and **`ms_to_ring` above IS the first half** - do not add a
    /// second counter for it:
    ///
    ///   * `ms_to_ring` - launch -> ring observed. The GPU is working (or already finished) here.
    ///   * `ms_host`    - ring observed -> `pre[l+1]` launched. **The GPU has nothing queued and is IDLE for
    ///                    all of this unless `pre[l]`'s tail still covers it.** This is
    ///                    `cudaMemcpyAsync(parts)` + `cudaGraphLaunch(post[l])` + `cudaGraphLaunch(pre[l+1])` +
    ///                    `cudaEventRecord`, and if it is the large half then the fix is fewer driver calls
    ///                    (R2.4), not a faster kernel.
    ///
    /// Cumulative, like `ms_to_ring`; divide by `calls_total * n_layers`.
    double ms_host = 0;
};

/// Captures all 48 layer graphs.  Idempotent: a second call is a no-op.
///
/// `pos`/`pos_base` are baked in ONLY through the fixed-address pinned staging in each `QsaState`, which the
/// replay re-reads every token - so the position is data, not an argument.  That is the whole reason rounds
/// 211-214 moved the per-token counts into device buffers.
bool session_capture(const WeightTable& tables, const ModelGeometry& g, SessionState& s, const float* parts,
                     SessionGraphs& gr, std::string& err, bool split = false, int64_t layer_lo = 0,
                     int64_t layer_hi = -1);

/// One token by REPLAYING the captured graphs.  Identical arithmetic to `session_token`; the only difference is
/// that ~2,000 kernel launches become 48 graph launches.
bool session_replay(const ModelGeometry& g, int64_t pos, int32_t pos_base, SessionState& s, SessionGraphs& gr,
                    void* stream, std::string& err);

/// One token by replaying BOTH halves of every layer - `pre[l]` and `post[l]` - with no pool running.
///
/// **THIS IS THE TRUE PER-TOKEN GPU FLOOR, AND `session_replay` IS NOT.**  `session_replay` launches only
/// `gr.execs[l]`, the `pre` graphs, so it omits the 48 `post` graphs (shared expert, combine, second `gr_write`)
/// entirely - and everything published as "39.8 ms pure GPU" came from it while being described as the whole
/// GPU.  See `Memory/ERRORS.md` A4.
///
/// `parts` is left at whatever the buffer holds, because no pool writes it; that is deliberate and makes the
/// measurement pure GPU work.  The LM head is NOT included here - the caller times it separately, since it runs
/// on the host-side path rather than inside a captured graph.
bool session_replay_full(const ModelGeometry& g, int64_t pos, int32_t pos_base, SessionState& s,
                         SessionGraphs& gr, void* stream, std::string& err);

/// **ONE TOKEN THROUGH THE SPLIT GRAPHS, WITH EVERY STAGE TIMED FROM OUTSIDE THE CAPTURE.**  Requires
/// `session_capture(..., split = true)`.  Replays `preA[l]`, `preB[l]`, `post[l]` in order on one stream -
/// the same order `session_loop` uses - and reports the three totals in milliseconds for the whole token.
///
/// The stages are ordered by the stream, so an event pair around each launch measures that launch's GPU work
/// plus the gap the driver left before it.  That gap is real and is why the three totals sum slightly above
/// `session_replay_full`; the comparison is printed by the caller rather than hidden here.
/// **R0.11: THE FIVE STAGES SEPARATELY, BY DIFFERENCING PREFIXES.**  `stage_ms` is resized to 5 and holds
/// stages 0..4 in milliseconds for a whole token; `mixer_per_layer` holds prefix 3's time for each layer, which
/// is what R0.10's GDN-versus-QSA split needs.  Requires `session_capture(..., split = true)`.
///
/// **THE RESIDUAL IS SAVED AND RESTORED BETWEEN PREFIXES.**  Stage 2 is `gr_write`, which mutates `R` in
/// place, so without the restore every prefix after the first reads a residual the earlier ones already
/// advanced - and every number after the first would be measuring a different tensor while looking correct.
bool session_replay_stage_prefixes(const ModelGeometry& g, int64_t pos, int32_t pos_base, SessionState& s,
                                   SessionGraphs& gr, void* stream, std::vector<double>& stage_ms,
                                   std::vector<double>& mixer_per_layer, std::string& err);

/// **R3.5c: ONE PREFIX, ALL 48 LAYERS, BACK TO BACK.**  `k` in 1..5 replays `preP[k-1][l]` (or `execs[l]` for
/// k = 5) for every layer with no other graph in between, and reports the whole sweep in milliseconds.
///
/// **WHY THIS EXISTS: TO RULE OUT THE PREFIX TABLE'S OWN BIAS.**  The five-stage table differences prefix
/// times measured with FIVE DIFFERENT GRAPHS LAUNCHED PER LAYER.  Stage 4's interval therefore contains the
/// launch of `execs[l]` - the full five-stage graph - while stage 3's contains `preP[3][l]`, a four-stage one.
/// A launch cost that scales with graph size would inflate stage 4 specifically, and stage 4 is the router,
/// which is the thing under investigation.  This sweep launches ONE graph type per layer, so there is nothing
/// to switch between and nothing to attribute wrongly.
///
/// Five separate sweeps give five totals; their consecutive differences are the stage times on the same
/// arithmetic as before, and **if the two methods disagree the difference is the bias and the sweep is right.**
bool session_replay_stage_sweep(const ModelGeometry& g, int64_t pos, int32_t pos_base, SessionState& s,
                                SessionGraphs& gr, void* stream, int k, double& ms, std::string& err);

bool session_replay_stages(const ModelGeometry& g, int64_t pos, int32_t pos_base, SessionState& s,
                           SessionGraphs& gr, void* stream, double& ms_mixer, double& ms_ffn, double& ms_post,
                           std::string& err);

/// **THE SAME REPLAY, KEEPING EVERY LAYER INSTEAD OF SUMMING THEM.**
///
/// The 48 layers are not 48 of the same thing: 36 are GDN and 12 are QSA, and the mixer - 55.8% of the GPU
/// floor - contains both. A total cannot separate them; a per-layer vector can, and it is already being
/// measured. `out` is resized to `n_layers` and holds each layer's mixer time in milliseconds.
///
/// **THIS IS THE CHEAPEST HALF OF R0.10 AND IT NEEDS NO NEW SPLIT.** Round 309's uncaptured table put GDN at
/// 10.88 ms for 36 layers and QSA at 4.56 for 12, which would make the recurrence the largest single R3
/// target; that table also had `moe_finish` wrong by 4x, so the ratio is worth re-deriving from the captured
/// graph rather than inherited.
bool session_replay_stages_per_layer(const ModelGeometry& g, int64_t pos, int32_t pos_base, SessionState& s,
                                     SessionGraphs& gr, void* stream, std::vector<double>& mixer_per_layer,
                                     double& ms_ffn, double& ms_post, std::string& err);

/// Releases the graph execs.  The `SessionGraphs` struct holds no device memory of its own.
void session_graphs_free(SessionGraphs& gr);

// ================================ THE HOST LOOP ================================

/// WHAT THE POOL HAS TO DO, as the loop sees it: given the `x_f` a layer published and the miss list it
/// selected, produce the `k` expert outputs into `out` (k x n_embd, HOST memory).  A function pointer rather
/// than a virtual class because the loop is the thing under test and the pool is the thing being faked.
using PoolFn = void (*)(void* user, const float* x_f, const int32_t* ids, const float* weights, int64_t n_embd,
                        int64_t k, float* out);

/// **R4.2c: THE VRAM-RESIDENT HALF OF THE LAYER'S EXPERTS.**  Called by the loop after the misses have been
/// staged into `parts` and before `post[l]` is launched, so its kernels are stream-ordered between them: the
/// misses are already in `parts`, the hits overwrite the rows the CPU zeroed, and `post[l]`'s `moe_combine`
/// reads the sum and cannot tell which engine produced which row.
///
/// **WHY IT IS A CALLBACK RATHER THAN MORE PARAMETERS.**  `session_loop` owns the ORDER across layers and has
/// no business knowing what an expert cache is.  Everything the hit path needs lives on the caller's `user`,
/// and this has the same shape as `PoolFn` for the same reason.
///
/// A null pointer means "no VRAM tier" - the default, and what every caller had before R4.
// The hit hook's type lives in its own header - see `hit_hook.hpp` for why and for the ordering it wants.

/// `phases/phase-2-correct-engine.md` P2.S5's host loop, which is the point of everything above it:
///
///     launch G[l] -> poll the doorbell -> read the miss list -> dispatch the pool -> wait for it ->
///     write y_miss -> launch G[l+1]
///
/// **NO `cudaStreamSynchronize` IN THE LOOP**, which is P2.X3.  Waiting is `cudaEventQuery` on an event recorded
/// right after the launch - a QUERY, and it is not optional: round 195 measured a host spin that only read
/// memory never running the kernel at all, 5,907,703 spins over 500 ms with the datum flipping only when
/// `cudaStreamSynchronize` was called.
///
/// **THE OVERLAP IS THE ONE-LAYER DELAY.**  A layer publishes `x_f` at ~71% of its own duration, so the pool has
/// the remaining 29% of that layer to work in; its result is consumed by the NEXT layer, which is why the
/// combine is not waiting on it.  `overlap = false` runs the same sequence with a synchronisation after every
/// layer, which is the comparison that says whether the pipeline is worth its complexity - and it is a
/// parameter rather than a second function so that the two cannot drift.
///
/// **`dump_layers`, WHEN NOT NULL, IS THE C1 ORACLE AND IT IS A HOST BUFFER OF `(n_layers + 1) * hc * n_embd`
/// FLOATS.**  Slot 0 is the residual the function was ENTERED with - the token's embedding broadcast to every
/// stream, which is the reference's `hc_init` node - and slot `l + 1` is `s.R` after layer `l`, which is the
/// reference's `l_last-<l>`.  Each is enqueued as a device-to-host copy, and the final `cudaStreamSynchronize`
/// at the end of the loop is what makes every one of them complete.  **NO EXTRA SYNCHRONISATION IS INSERTED**,
/// which is the whole reason this is written as an enqueue rather than a read: `pre[l+1]` does not write `R` and
/// `post[l+1]` is launched after the copy on the same stream, so FIFO order alone makes slot `l + 1` a true
/// snapshot.  It costs 40 KB per layer per token, and it is the only way to bisect the engine against the
/// reference layer by layer instead of against one end-to-end perplexity.
/// Per-token scratch that MUST NOT be reallocated per token (P2.T10, review finding H3).
///
/// **`session_loop` IS CALLED ONCE PER TOKEN, SO ANYTHING IT ALLOCATES IS A PER-TOKEN ALLOCATION.**  It used to
/// allocate three things on every token of every run:
///
///   * `cudaHostAlloc` for the pool's pinned staging, freed by an RAII destructor at the end of the call - and
///     `cudaFreeHost` IMPLICITLY SYNCHRONISES THE DEVICE, so every token ended with a device sync;
///   * `cudaEventCreate` / `cudaEventDestroy` for the doorbell probe;
///   * a host-affinity change (`physical_cores` + pin + restore).
///
/// A comment in the old code said "allocated once", but once per token is not once.
///
/// Own one of these at session setup and pass it in.  **Passing `nullptr` keeps the old behaviour** - the loop
/// allocates and frees locally - so every existing caller and test is unaffected, but the allocation is then
/// visible at the call site instead of hidden inside a hot function.
struct SessionLoopScratch {
    float* y_miss = nullptr;        ///< pinned host staging for the pool's answer, `parts_bytes` long
    size_t parts_bytes = 0;
    cudaEvent_t probe = nullptr;
    strata::kernels::cpu::ThreadAffinity pinned_core{};  ///< affinity to restore, or invalid if the host was not pinned
    bool pinned = false;

    /// Allocates the buffers and pins the host thread.  Call ONCE, at session setup.
    bool init(size_t parts_bytes_in, std::string& err);
    /// Frees everything and restores the affinity.  Safe to call twice, or on a default-constructed object.
    void free();
};

bool session_loop(const ModelGeometry& g, int64_t pos, int32_t pos_base, SessionState& s, SessionGraphs& gr,
                  PoolFn pool, HitFn hits, void* user, bool overlap, void* stream, std::string& err,
                  float* dump_layers = nullptr, SessionLoopScratch* scratch = nullptr);

// ================================ plan v0.3 P3: THE WHOLE TOKEN AS ONE GRAPH ================================
//
// Under WDDM every graph launch costs ~0.3-0.4 ms of submission latency, and the per-layer loop above makes 96
// of them per token: measured 48.6 ms/token for 12 ms of GPU kernels and 27 ms of pool.  The token graph holds
// all 48 layers.  Between `pre[l]` (which rings the doorbell) and the parts copy, a one-thread kernel waits ON
// THE DEVICE until the host writes the ring value into `db.h_flag`; the host only polls rings, runs the pool
// and writes flags, with no driver call in the loop (bench/micro/device_wait.cu: 5.2 us per handoff).
//
//     for l:  pre[l] (.. router, doorbell ring, shared expert)  ->  doorbell_wait  ->  parts <- y_miss (H2D)
//             ->  post[l] (combine, gr_write)
//
// Same kernels, same order per layer as `session_loop`, so the token is bitwise the same.  No VRAM expert tier
// yet (the hit decision is a host step between ring and post); the caller falls back to `session_loop` then.
struct TokenGraph {
    cudaGraphExec_t exec = nullptr;
    bool captured = false;
    int64_t n_layers = 0;
    const float* y_src = nullptr;    ///< the pinned host staging the graph's H2D copies read (baked in)
    size_t parts_bytes = 0;
    int64_t calls = 0;
    double ms_wait = 0;              ///< host waiting for rings (the GPU is running)
    double ms_pool = 0;              ///< host inside the pool
    int64_t flushes = 0;             ///< cudaStreamQuery calls made because a ring was slow to appear
};

/// Plan v0.3 P4: the VRAM expert tier inside the token graph.  Residency is STATIC during a token (a
/// `n_layers x n_expert` table, slot or -1, on the device; the host pool reads its own copy), so the hit list
/// is built on the device right after the ring and the hits run while the CPU computes the misses.
struct TokenHits {
    const int32_t* d_res = nullptr;      ///< device [n_layers * n_expert]
    int64_t n_expert = 0;
    const uint8_t* cache_base = nullptr; ///< slot 0 of the VRAM expert arena
    int64_t blob = 0;                    ///< bytes per slot
    int32_t* d_slot = nullptr;           ///< device, k entries
    int32_t* d_dst = nullptr;            ///< device, k entries
    int32_t* d_count = nullptr;          ///< device, 1
    uint8_t* x_q8 = nullptr;             ///< the activation's q8_0 blocks, (n_embd/32)*34 B
    float* x_scale = nullptr;            ///< and its fp32 scales (the CPU pool's contract)
    void* scratch = nullptr;             ///< moe_hit_grouped_scratch_bytes(k, ...)
    float* hit_out = nullptr;            ///< device, k * n_embd
    bool on() const { return d_res && cache_base && d_slot && d_dst && d_count && x_q8 && x_scale && scratch && hit_out; }
};

bool session_capture_token(const WeightTable& tables, const ModelGeometry& g, SessionState& s, float* parts_dev,
                           const float* y_miss_host, size_t parts_bytes, TokenGraph& tg, std::string& err,
                           const TokenHits* hits = nullptr);
bool session_run_token(const ModelGeometry& g, int64_t pos, int32_t pos_base, SessionState& s, TokenGraph& tg,
                       PoolFn pool, void* user, float* y_miss_host, void* stream, std::string& err);
void token_graph_free(TokenGraph& tg);

}  // namespace strata::core
