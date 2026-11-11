// include/strata/core/verify.hpp - plan v0.3 P6: the speculative VERIFY window.
//
// T tokens at consecutive positions p0 .. p0+T-1 - the last accepted token and T-1 drafts - go through all 48
// layers in ONE captured graph, and the head's argmax is produced for every one of them.  Token t's argmax is
// what plain greedy decode would produce after token t, BIT FOR BIT: every kernel here is either the single-token
// kernel applied per token, or a multi-token kernel whose per-token arithmetic is the single-token kernel's
// (multi-column MMVQ in exact mode, the T-token GDN kernels, the per-token hit activation, the multi-token CPU
// expert rows).  So a draft is accepted exactly when greedy decode would have produced it.
//
// What the window costs is the dense weights read ONCE for T tokens and the union of the T tokens' missed
// experts on the CPU (measured on decode traces: 1.75x one token's misses for T=2, 2.4x for 3, 3.05x for 4).
//
// STATE.  The window appends K/V and indexer keys for all T positions and leaves the GDN state untouched.
// `commit(n_keep)` then makes the first `n_keep` tokens permanent: the GDN conv history and recurrent state are
// advanced by replaying those tokens from inputs the window stored, the indexer's key tail is restored from a
// snapshot and the accepted keys re-appended (a rejected key can land in a slot the current block still needs),
// and the PLE history is set to its snapshot after token n_keep-1.  K/V cells past the accepted prefix are simply
// overwritten when those positions are processed again, before any query can read them.
//
// Requires the default native decode configuration (native projections, fused GR, fused GDN, fast attention and
// selection, native indexer) and a profile-filled VRAM expert tier with its residency table on the device.
#pragma once

#include <cstdio>

#include "strata/core/expert_source.hpp"
#include "strata/core/latent_dense.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/window_limits.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <map>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// 0.1.43 (STRATA_ADAPT_OVERLAP): the adaptive tier's copies still landing while a decode window runs.  The window's
/// residency tables are the baseline's (the swapped-in experts are resident from the window's first layer), but their
/// bytes arrive layer by layer: `run` holds a layer's plan (the first thing that lets the GPU read the layer's expert
/// slots) back until `landed >= layer` for a layer with a swap (`has[layer]`).  Set `active` only around one window.
struct AdaptGate {
    struct Tick { AdaptGate* gate; int32_t layer; };
    std::atomic<int32_t> landed{-1};      ///< the highest layer whose swap copies have landed (n_layers: all of them)
    std::atomic<bool> active{false};
    std::vector<uint8_t> has;             ///< per layer: a swap copies into this layer's slots
    std::vector<Tick> ticks;              ///< host-function contexts, one per layer and the final one (stable storage)
    const int32_t* dvals = nullptr;       ///< device [0..n_layers]: the source of those copies
    volatile int32_t* hland = nullptr;   ///< pinned host int the copy stream writes (a D2H copy of the layer number), instead of a host function; null: host functions
    int32_t reached() const {
        const int32_t a = landed.load(std::memory_order_acquire);
        if (hland == nullptr) return a;
        const int32_t b = *hland;
        return a > b ? a : b;
    }
};

class NativeHead;
class RemoteExpertOpt;

/// The CPU pool for a window: x_f (n_tok, n_embd), ids (n_tok, k) -> out (n_tok * k, n_embd), hit rows zeroed.
using PoolMultiFn = void (*)(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                             int64_t layer);

struct VerifyHits {
    const int32_t* d_res = nullptr;      ///< device [n_layers * n_expert] slot or -1
    const int32_t* h_res = nullptr;      ///< host [n_layers * n_expert] slot or -1 (when all >= 0 in stage: zero-doorbell)
    const uint8_t* cache_base = nullptr; ///< slot 0 of the VRAM expert arena
    const uint64_t* slot_off = nullptr;   ///< E-6: host per-slot offsets when slots differ in size (null: slot * blob)
    int64_t n_slots = 0;                  ///< E-6: how many (for the device copy)
    int64_t blob = 0;
};

class Verifier {
public:
    Verifier() = default;
    ~Verifier();
    Verifier(const Verifier&) = delete;
    Verifier& operator=(const Verifier&) = delete;
    void set_remote_expert_opt(RemoteExpertOpt* opt) { remote_opt_ = opt; } // before init/capture
    /// A session-level shared latent D/U pair (allocated and uploaded elsewhere, e.g. the engine): the init
    /// keeps no arena copy of its own and skips its upload; the verifiers only read the pair.  Set before `init`.
    void set_shared_latent(const uint8_t* d, const uint8_t* u) { lat_d_shared_ = d; lat_u_shared_ = u; }

    /// The watchdog's view of the window in flight (issue #31): the layer, the GPU's sequence, the flags.
    void diag(std::FILE* f) const;
    /// #649 (STRATA_VERIFY_TRACE=1, opt-in): the last windows' host-side handshake events and the GPU's breadcrumbs
    /// (in mapped memory, so they read while the GPU hangs).  Nothing when the trace is off.
    void trace_dump(std::FILE* f) const;

    /// #267: raise every flag the window's spin kernels wait on past any ring (UINT32_MAX), so a window the GPU
    /// cannot finish drains instead of staying resident, then wait up to `timeout_ms` for its streams.  For the
    /// paths that give up on the engine (a timed-out window, the serve watchdog): the window then ran on whatever
    /// the flags guarded, so this verifier refuses every later window.  True when the streams finished.
    bool release_gpu_waits(int timeout_ms);

    /// `max_t` <= kVerifyMaxT.  `head` may be null (the canonical head is then run per token).
    bool init(const WeightTable& wt, const ModelGeometry& g, SessionState& ss, const VerifyHits& hits,
              const NativeHead* head, int max_t, std::string& err);

    /// One window: `tokens[0..T)` at positions pos0.., the pool served per layer; `out[t]` = argmax after token t.
    /// The PLE rows are gathered here from `ss.ple_prev` and the tokens.  Captures the T-token graph on first use.
    bool run(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool, void* user, int32_t* out, std::string& err);
    /// Diagnostics: row `t` of the last window's head logits (n_vocab floats) to the host. Valid after run().
    bool copy_logits(int t, float* host) const;
    int64_t vocab() const { return next_ ? next_->vocab() : n_vocab_; }
    /// #879 GUARD: the first row of the last finished window (run, run_slot_rows, batch_poll, pl_finish) whose head
    /// logits held a NaN or an infinity, else -1; nonfinite_pos() is that row's position.  Every window graph
    /// checks its T rows on the device beside the greedy pick (logits_nonfinite_rows into mapped flags, read after
    /// the sync the window already does: no extra sync).  On the stage with the head (a layer split asks the last
    /// stage).  STRATA_NAN_GUARD=0: off (always -1).  STRATA_NAN_INJECT=<n>[,<n>...] (debug): the n-th window the
    /// guard checks in this process (counted from 1 over every stage-with-head verifier: prompt-tail windows, decode
    /// windows and batch windows alike) reports row 0 as non-finite, so the whole response can be tested.
    int nonfinite_row() const { return next_ != nullptr && !last_stage() ? next_->nonfinite_row() : nf_row_; }
    int64_t nonfinite_pos() const { return next_ != nullptr && !last_stage() ? next_->nonfinite_pos() : nf_pos_; }
    /// Row t of the last window (a batch window: each slot's own rows) held a non-finite logit.
    bool nonfinite_in(int t) const {
        if (next_ != nullptr && !last_stage()) return next_->nonfinite_in(t);
        return t >= 0 && t < 32 && ((nf_mask_ >> t) & 1u) != 0;
    }
    static bool nan_guard_on();
    /// The most tokens one window holds (`init`'s `max_t`; 0 before it).
    int max_t() const { return max_t_; }
    /// The sampling the verify window's head applies (temperature / top_p / top_k / seed).  Set per
    /// request; greedy by default.  The sampling itself runs OUTSIDE the captured graph - its
    /// parameters would otherwise be baked forever - so this can change between requests freely.
    void set_sampling(const strata::kernels::SamplerParams& sp) {
        sampling_ = sp;   // row t of a window at pos0 draws Philox(seed, pos0 + t): see run()
        sampling_.logit_bias = logit_bias_host_.empty() ? nullptr : logit_bias_device_;
        if (next_) next_->set_sampling(sp);
    }
    // One request's fixed vocabulary bias. Empty clears it; copied only when changed, on the final stage's device.
    bool set_logit_bias(const std::vector<float>& bias, std::string& err);

    /// The penalty histories for `sampling_.penalty_last_n`: ONE ROW PER WINDOW ROW, T rows of `history_len`
    /// int32 slots at that stride (`strata::kernels::penalty_rows` builds them), most recent token LAST, unused
    /// front slots -1 (the kernel reads only the tail window).  Row t follows the window's drafts 1..t - staging
    /// row 0 alone (before 0.1.19) left the drafted rows with unwritten histories.  Null disables the penalties
    /// entirely - the neutral run's sampling call is byte-for-byte what it was.  The engine re-uploads the rows
    /// before every window; the buffer must hold kVerifyMaxT rows and stay alive across the request.
    void set_history(const int32_t* history, int history_len) {
        hist_d_ = history;
        hist_len_ = history_len;
        if (next_) next_->set_history(history, history_len);
    }
    /// PROBABILISTIC DRAFT ACCEPTANCE (core/spec_prob.hpp, STRATA_SPEC_PROB=1): for the NEXT run() only, judge the
    /// window's first `n_q` drafts against the drafter's distributions `q` (host memory, n_q rows of kSpecQStride
    /// int32: ids, -1 terminated, then probabilities as float bits) with rejection sampling instead of exact match; the
    /// drafts past `n_q` (a lookup chain's tail) are point masses.  Has no effect on a greedy request, a window of one
    /// token, or where the split sampler cannot run - those take the exact-match path.  Cleared by run().
    void set_spec_q(const int32_t* q, int n_q) {
        spec_q_ = q;
        spec_nq_ = n_q;
        spec_measure_ = false;
        if (next_) next_->set_spec_q(q, n_q);
    }
    /// Borrow immutable gated measure rows for the next run only. nq==T denotes
    /// a NULL tail, nq<T a plain bonus. Errors never fall back to ordinary p.
    /// nq == 0 accepts a null pointer: point-only rows and a plain bonus (T1: plain p).
    /// Keep the host rows alive and unchanged until run returns.
    void set_spec_proposal(const int32_t* q, int n_q) {
        spec_q_ = q;
        spec_nq_ = n_q;
        spec_measure_ = true;
        if (next_) next_->set_spec_proposal(q, n_q);
    }

    /// Counters of the rejection path (windows judged, drafts kept / offered over its rows with a q list).
    int64_t spec_windows = 0;

    /// Off: `run` skips the request's head sampling and `out` is the recorded greedy pick.  For windows whose
    /// picks are discarded - a prompt read through windows commits every token - so they cost no sampler launch
    /// or sync and never read a history staged for another position.
    void set_head_sampling(bool on) { head_sampling_ = on; if (next_) next_->set_head_sampling(on); }

    /// LAYER SPLIT (multi-GPU): this verifier runs layers [layer_begin, layer_end) of every window.  A stage that
    /// does not start at layer 0 takes its residual from `handoff_in` instead of embedding the tokens; a stage that
    /// does not end at the last layer writes its residual to `handoff_out` and has no head.  The hand-off holds,
    /// per token, the residual R (hc x n_embd), the last layer's pending write bo (n_embd) and inject (hc): the next
    /// stage folds that write into its first read exactly as the unsplit window does, so the split is bit-exact.
    /// Both pointers must be device-visible (mapped pinned memory, portable when the stages are on two devices).
    /// Set before `init`.  Default: the whole model, no hand-off.
    void set_stage(int64_t layer_begin, int64_t layer_end, const float* handoff_in, float* handoff_out) {
        lb_ = layer_begin; le_ = layer_end; hand_in_ = handoff_in; hand_out_ = handoff_out;
    }
    /// The rows the hand-off buffers of `set_stage` hold (0: kVerifyMaxT, the default).  More than --batch's 8 slots on a
    /// layer split: the pipelined slot groups write their rows at hbase = group * rows-per-group, one row per slot.
    void set_handoff_rows(int rows) { handoff_rows_ = rows; }
    /// Is the zero-doorbell graph the one a batch window would run now (every expert of the stage in VRAM by the host
    /// table)?  Looks at the table, as each window does.  (STRATA_ADMIT_MERGE_FIRST's gate.)
    bool ar_live() { refresh_ar(); return ar_on(); }
    /// STRATA_SPLIT_OVERLAP=1 (opt-in; stages on DIFFERENT devices only): the hand-off carries a ready flag (one
    /// mapped word per hand-off buffer: host and device addresses).  The writing stage raises it from the GPU when
    /// its hand-off is out (handoff_publish); the reading stage's window graph waits on it (wait_flag_ge) - so the
    /// next stage's graph is launched while this stage is still running, and the host no longer syncs between the
    /// stages.  Null (the default): the old order - sync, then launch the next stage.  Set before `init`.
    void set_handoff_flags(uint32_t* in_host, uint32_t* in_dev, uint32_t* out_host, uint32_t* out_dev) {
        hflag_in_h_ = in_host; hflag_in_d_ = in_dev; hflag_out_h_ = out_host; hflag_out_d_ = out_dev;
    }
    /// The next stage: `run` and `commit` continue into it (its pool calls get `next_user`); sampling settings
    /// and `final_R` are the last stage's.
    void set_next(Verifier* next, void* next_user) { next_ = next; next_user_ = next_user; }
    /// floats per token in a hand-off buffer
    static int64_t handoff_floats(const ModelGeometry& g) { return (int64_t) g.hc * g.n_embd + g.n_embd + g.hc; }

    /// Keep the first `n_keep` (1..T) tokens of the last window; advances `ss.ple_prev` by them.
    bool commit(int n_keep, std::string& err);

    // ================================ SEVERAL SEQUENCES IN ONE WINDOW ================================
    //
    // A batch window holds S INDEPENDENT sequences, one token each: row s is slot s, at slot s's own position,
    // reading and writing slot s's own state (GDN recurrence and conv history, QSA K/V and indexer, PLE history),
    // which lives in `slots[s]` - a session carved like this verifier's own (same layer range, same max_cells).
    // Everything that is per row already (hyper-connections, dense projections, router, shared expert, the
    // routed experts on the GPU and the CPU, the head) runs once over the S rows, so the weights are read once
    // per window for all the sequences.  Row s's arithmetic is the single-token window's, so a slot's greedy
    // tokens are its solo greedy tokens (modulo the multi-token CPU kernel choice: STRATA_IQ_MT_MIN=1).
    //
    // Greedy only, no drafts (MTP) in a batch window.  `init_slots` once after `init` (S <= max_t); a layer
    // split's stages each get their own sessions, and run_slots/commit_slots continue into the next stage.
    bool init_slots(const std::vector<SessionState*>& slots, std::string& err);
    int n_slots() const { return (int) slots_.size(); }
    /// One batch window over slots [0, S): tokens[s] at positions pos[s]; out[s] = the greedy pick after it.
    bool run_slots(int S, const int32_t* tokens, const int64_t* pos, PoolMultiFn pool, void* user, int32_t* out,
                   std::string& err);
    /// The same over the S slots `rows` (row t is slot rows[t], any distinct slots in any order): the slots not
    /// listed are not touched, so an idle slot keeps its state (a finished conversation it may continue later).
    bool run_slot_rows(const int* rows, int S, const int32_t* tokens, const int64_t* pos, PoolMultiFn pool, void* user,
                       int32_t* out, std::string& err);
    /// Keep every row of the last batch window.
    bool commit_slots(std::string& err);
    /// Commit an accepted prefix in each contiguous slot group of the last window. `keep` has
    /// one entry per slot (indexed by slot ID), each in 1..that slot's group length.
    bool commit_slot_prefixes(const int* keep, std::string& err);
    /// --batch-mtp: slot rotation makes many row layouts, so bound the captured batch graph pairs (LRU). 0 = unbounded.
    void set_batch_graph_limit(size_t n) { batch_graph_limit_ = n; }

    // ---- The stages of a layer split as a PIPELINE.  A batch window over the slot GROUP
    // [base, base + S) is launched on ONE stage with its commit right behind it on the stage's stream (a batch window
    // keeps every row, so the commit needs no host decision), and the host serves the rings of every stage that has
    // a window in flight from one thread (batch_poll does not block).  Stage k can then run group g while stage k+1
    // runs group g-1.  Rows of group `base` use hand-off rows [base, base + S), so groups never share a hand-off row.
    bool batch_launch(int base, int S, const int32_t* tokens, const int64_t* pos, std::string& err);
    /// 1 = this stage's window and commit are done (the last stage's picks are in batch_out), 0 = still running,
    /// -1 = an error (err).  Serves every layer that has rung so far.
    int batch_poll(PoolMultiFn pool, void* user, std::string& err);
    bool batch_busy() const { return b_running_.load(std::memory_order_acquire); }
    // ---- STRATA_PIPED_MTP: the pipelined groups with --batch-mtp's drafts.  A group's window holds its slot's token
    // and its drafts, so how many rows to keep is known only once the LAST stage has picked - and by then this stage
    // runs another group's window, which reuses the scratch the commit replays from.  So right behind the window a
    // save graph copies those rows (each GDN layer's conv and recurrence inputs, each QSA layer's raw indexer keys, the
    // PLE history snapshot) to the group's own rows [hbase, hbase + S) of the saved buffers, and the commit, queued
    // after the verdict, replays from there.  Everything else the commit reads is per slot already.
    /// The saved buffers for `rows` hand-off rows (after init_slots).
    bool init_deferred_commit(int rows, std::string& err);
    bool deferred_commit_on() const { return dc_rows_ > 0; }
    /// The window over the slots `rows` (a slot's rows contiguous) at hand-off rows [hbase, hbase + S), its save right
    /// behind it, and no commit (batch_commit_rows, after the verdict).  The PLE history advances at the commit.
    bool batch_launch_rows(const int* rows, int S, int hbase, const int32_t* tokens, const int64_t* pos,
                           std::string& err);
    /// Queue the commit of that window's accepted prefixes on this stage's stream (no host wait): `keep` per slot ID,
    /// 1..the slot's rows.  `rows`, `S`, `hbase`, `tokens` and `pos` are the window's, as batch_launch_rows got them.
    bool batch_commit_rows(const int* rows, int S, int hbase, const int* keep, const int32_t* tokens,
                           const int64_t* pos, std::string& err);
    /// Serves the layers of this stage's window that have rung (the CPU pool and the flags), with no CUDA call: 1 once
    /// every layer is served (or the window raises no host doorbells), 0 while one is still to ring.  batch_poll
    /// calls it, unless set_external_serve(true) moved it to a thread of its own (STRATA_PIPED_CONCURRENT: a CUDA
    /// call can wait for the driver's lock while another thread holds it waiting for the GPU, whose window waits
    /// for its doorbell, #31); batch_poll then makes no CUDA call before every layer is served.
    int batch_serve(PoolMultiFn pool, void* user);
    void set_external_serve(bool on) { ext_serve_.store(on, std::memory_order_release); }
    /// A slot's sampling (temperature / top_p / top_k / min_p / seed; penalties are not applied in batch windows):
    /// its row is drawn again on the last stage with Philox(seed, position), as a solo window draws it.  Greedy by
    /// default.  Set on the first stage, it reaches the last.
    void set_slot_sampling(int slot, const strata::kernels::SamplerParams& sp) {
        if (slot >= 0 && slot < (int) slot_sp_.size()) slot_sp_[(size_t) slot] = sp;
        if (next_) next_->set_slot_sampling(slot, sp);
    }
    const int32_t* batch_out() const { return b_out_; }
    bool last_stage() const { return g_ != nullptr && le_ == g_->n_layers; }
    /// commit() returns without waiting for its graph (a single-GPU session sets it): the next window follows it on
    /// the same stream and the drafter reads nothing it writes, so it overlaps the draft. Whoever reads or writes
    /// the session from another stream or the host afterwards (a new request, a checkpoint, a snapshot, the prompt
    /// path, the end of a run) calls wait_commit() first.  STRATA_COMMIT_SYNC=1 keeps the wait.
    static void set_commit_async(bool on);
    /// Waits for the last commit graph when commit() did not (an event recorded after it, not the whole device);
    /// false with `err` when it failed.  Free when nothing is pending.
    bool wait_commit(std::string& err);

    // ---- PIPELINED WINDOWS (--pipeline-windows, a layer split on two GPUs).  One conversation's windows with the
    // stages overlapped: stage 0 runs window K+1 while stage 1 still runs window K.  The same window as `run`, driven
    // without blocking the host, so one host thread keeps a window in flight on each stage (the batch pipeline's
    // pattern, batch_launch / batch_poll, for one sequence with drafts).  Two verifiers per stage (one per window
    // parity) share the stage's stream and each has its own hand-off.  `pl_launch` stages and launches (it never
    // captures: `capture_all` first, with nothing in flight), `service` serves the layers whose doorbells have rung
    // and returns at once, `done` polls the window's completion, `pl_finish` reads the picks, `pl_commit_async`
    // queues the commit.  Nothing chains to `next_`: the caller drives every stage.
    /// The compute stream to use instead of a private one (the two verifiers of one stage share it).  Before `init`.
    void set_stream(cudaStream_t s) { ext_stream_ = s; }
    /// Every layer copies the token rows to the host (doorbell_publish), not only the layers with a routed expert
    /// outside the VRAM tier by the device's residency table: with windows in flight the adaptive tier marks an
    /// expert evicted on the host (the pool then computes it on the CPU) before the device table follows.  Also turns
    /// the device-planned layers (E-6) off, which read the device table this mode lets lag.  Before the first window
    /// (its graphs are captured there).
    void set_always_publish(bool on) {
        always_publish_ = on;
        if (on) device_plan_ = false;
    }
    /// 0.1.43: the adaptive tier's overlap gate (see AdaptGate).  Refused (false) with the device-planned layers (E-6),
    /// which never ask the host.  The pointer must outlive the verifier's windows.
    bool set_adapt_gate(AdaptGate* g) {
        if (g != nullptr && device_plan_) return false;
        adapt_gate_ = g;
        return true;
    }
    double ms_gate = 0;   ///< host time spent holding a layer's plan for the adaptive copies (decode timing prints it)
    cudaStream_t stream() const { return cs_; }
    int device() const { return device_; }
    /// Capture every window size and the commit graph now (a capture syncs the stream: never with a window in flight).
    bool capture_all(std::string& err);
    bool pl_launch(int T, const int32_t* tokens, int64_t pos0, std::string& err);
    /// Stage a window ahead of its launch (positions, and the PLE rows from `ple_prev` = the two tokens before the
    /// window as they WILL be, their pages prefetched).  A later `pl_launch` of the same window (T, pos0, tokens, and
    /// `ss.ple_prev` equal to `ple_prev` by then) skips the staging; anything else stages again.
    bool prestage(int T, const int32_t* tokens, int64_t pos0, const int32_t ple_prev[2], std::string& err);
    /// 1: every layer served; 0: the GPU has not reached the next layer yet; -1: an error (`err`).
    int service(PoolMultiFn pool, void* user, std::string& err);
    bool in_flight() const { return fl_active_; }
    /// The window's graph (and its profile copy) completed; false while it runs.  An error sets `err`.
    bool done(std::string& err);
    /// After `done`: the profile, the last stage's host sampling and picks (`out` may be null on an earlier stage).
    bool pl_finish(int32_t* out, std::string& err);
    /// The commit without a host sync; `ss.ple_prev` advances now (host side).  A second call for the same window
    /// (after its state was restored) replays it with another count.
    bool pl_commit_async(int n_keep, std::string& err);
    /// Fold another verifier's counters and GPU profile into this one's (the two verifiers of one stage report once).
    void absorb_stats(Verifier& o);
    /// The watchdog's line for a pipelined verifier: in flight, layers served, the GPU's ring and flags, its events.
    void diag_pipelined(std::FILE* f, const char* name) const;

    /// Measurement hook (STRATA_LOGPOS): after run(), write one line per row t of the last window's head -
    /// "pos target logprob top top_logprob hit extra_logprob target_logprob_without_extra" - where row t is the
    /// distribution at pos0 + t, targets[t] is the token at pos0 + t + 1, extra_logprob is the log-probability of
    /// `extra_id` in the same row, and the last column is the target's log-probability in that row renormalized
    /// over every token but `extra_id` (both nan when they do not apply).  A layer split's earlier stage forwards
    /// to the stage that holds the head.
    bool window_logprobs(const int32_t* targets, int T, int64_t pos0, int32_t extra_id, std::FILE* out,
                         std::string& err);

    /// Token t's residual after the last layer, (hc, n_embd) on the device, valid until the next `run`.
    const float* final_R(int t) const;
    const float* final_R_all() const { return next_ ? next_->final_R_all() : R_; }
    /// DFlash2 (before init): keep the completed outputs of these layers (zero-based; the mean of the hc streams) for every row of
    /// each window in a device buffer, tap_dev() = [max_t][n_taps][n_embd] floats, valid until the next window.  Single-stage
    /// verifiers only; STRATA_TAP_FILE (research dump) turns the same buffer on without this call.
    void set_taps(const std::vector<int>& layers) { tap_req_ = layers; }
    const float* tap_dev() const { return tap_dev_; }
    bool tapping() const { return tap_on_; }

    /// The GPU plan the pool writes each layer (VRAM hits + the PCIe share of the misses); give it to the
    /// dispatch (`ExpertDispatch::plan`) before the first `run`.
    GpuPlanSink* plan_sink() { return &sink_; }
    /// Plan v0.3 P6: split the window into two token groups and pipeline the CPU experts of one with the GPU work
    /// of the other (default on).  Set before the first `run`.
    void set_split(bool on) { split_ = on; }
    /// Plan v0.3 P6: how the PCIe share of the misses reaches the GPU: 0 = DMA into staging (the copy engine works
    /// beside the CPU; best when the CPU is compute-bound, the i-quants), 1 = the grouped kernel reads the mapped
    /// arena directly, 2 = a copy kernel stages it inside the graph (no API calls on the pool's thread; best when
    /// the CPU is RAM-bound, Q2_0).  Set before the first `run`.
    void set_pcie_mode(int mode) { sink_.pcie_mode = mode; }
    /// Called before every window this verifier launches (run, the batch and pipelined launches): the device's
    /// residency table catches up with the host's (the adaptive tier's evictions while its copies are in flight).
    /// It gets this verifier's stream; false (with the message in `err`) fails the launch.
    void set_pre_launch(std::function<bool(cudaStream_t cs, std::string& err)> f) { pre_launch_ = std::move(f); }
    /// the pool never plans a PCIe share (--pcie-frac 0): the window skips that path.  Before the first run.

    /// The PCIe count of each layer from measured costs (PcieModel) instead of a fixed share: the spin kernels stamp
    /// each step's GPU work (the plan's arrival, the PCIe part's start, the wait for the CPU's rows; GPU clock), the host
    /// times its pool, and after each window *model (the dispatch's) takes the costs. One GPU, no batch slots. Set
    /// before `init` (the stamps are captured). `paused`: a request's own --pcie-frac holds meanwhile.
    void set_pcie_balance(PcieModel* model) { bal_ = model; }
    void pause_pcie_balance(bool paused) {
        bal_paused_ = paused;
        if (bal_ != nullptr && m_bal_ != nullptr) bal_->on = !paused;
    }
    bool pcie_balance_on() const { return bal_ != nullptr && m_bal_ != nullptr; }

    double ms_wait = 0, ms_pool = 0, ms_host = 0, ms_commit = 0;
    int64_t windows = 0;
    /// STRATA_VERIFY_PROFILE=1 - GPU stage times of the windows since the last call (ms per
    /// window), as one line; empty when off.
    std::string profile_report();
    /// STRATA_VERIFY_PROFILE: the same stage split, one line per window size T (bench: with STRATA_SPEC_T_CYCLE). Resets.
    std::string profile_by_t_report();
    /// STRATA_ROUTE_STATS=1: per window size, the routed experts per layer: distinct, and new per row (not in the rows before it). Resets.
    std::string route_stats_report();

private:
    RemoteExpertOpt* remote_opt_ = nullptr;
    bool capture(int T, std::string& err);
    // batch windows (see init_slots)
    std::vector<SessionState*> slots_;
    bool batch_rec_ = false;               ///< record_window is capturing a batch window
    int row_base_ = 0;                     ///< ... its hand-off rows start here (a pipeline group's own rows)
    int brow_[strata::kernels::kWindowMaxT] = {};                     ///< ... and row t is slot brow_[t]
    bool last_batch_ = false;              ///< the last run was a batch window (set_plan_slot: one group)
    std::map<std::vector<int>, cudaGraphExec_t> exec_bm_, commit_bm_;   ///< full row layout avoids slot-ID collisions
    std::map<std::vector<int>, uint64_t> bm_used_;   ///< last use of each captured layout (LRU)
    uint64_t bm_tick_ = 0;
    size_t batch_graph_limit_ = 0;         ///< 0: keep every captured batch graph (0.1.39); N: LRU-evict beyond N layouts
    int last_rows_[strata::kernels::kWindowMaxT] = {};                ///< the slots of the last batch window's rows
    std::vector<int> bkey(const int* rows, int S, int hbase) const {   // #871: the doorbell variant has its own graphs
        std::vector<int> k = batch_key(rows, S, hbase);
        k.push_back(ar_off_ ? 1 : 0);
        return k;
    }
    static std::vector<int> batch_key(const int* rows, int S, int hbase) {
        std::vector<int> k;
        k.reserve((size_t) S + 1);
        k.push_back(hbase);
        for (int t = 0; t < S; ++t) k.push_back(rows[t]);
        return k;
    }
    // batch_launch / batch_poll (with external serving, batch_serve runs on another thread: running and the served
    // layer count are atomics, set by batch_launch before running)
    std::atomic<bool> b_running_{false};
    std::atomic<int64_t> b_k_{0};
    int64_t b_steps_ = 0;
    std::atomic<bool> ext_serve_{false};
    // when a layer was last served (steady clock ticks): the serving thread writes it, batch_poll's checks read it
    std::atomic<int64_t> b_last_{0};
    void ring_now() {
        b_last_.store((int64_t) std::chrono::steady_clock::now().time_since_epoch().count(), std::memory_order_release);
    }
    std::chrono::steady_clock::time_point last_ring() const {
        return std::chrono::steady_clock::time_point(
            std::chrono::steady_clock::duration(b_last_.load(std::memory_order_acquire)));
    }
    std::chrono::steady_clock::time_point ext_checked_{};   ///< batch_poll's last check with external serving
    int32_t b_out_[strata::kernels::kWindowMaxT] = {};
    std::vector<strata::kernels::SamplerParams> slot_sp_;
    bool sample_rows(int S, std::string& err);   ///< the sampled slots' rows of the last batch window
    int32_t* h_commitb_ = nullptr; int32_t* m_commitb_ = nullptr;   // per slot [1, 0, pos, -1 ..], stride 2 + max_t
    int32_t* commitb_ = nullptr;
    float* tail_snap_b_ = nullptr;         ///< per (slot, QSA layer) indexer tail snapshot
    void* arena_b_ = nullptr;
    int64_t last_pos_b_[strata::kernels::kWindowMaxT] = {};
    bool capture_batch(const int* rows, int S, int hbase, std::string& err);
    // STRATA_PIPED_MTP (init_deferred_commit): the saved rows, laid out as the window scratch (global layer index) with
    // dc_rows_ rows per layer instead of max_t_; the slots' accepted prefixes (mapped and device, 2 + max_t_ per slot)
    int dc_rows_ = 0;
    bool staging_deferred_ = false;   ///< stage_batch for batch_launch_rows: no capture_commit_batch
    float *dc_qkv_ = nullptr, *dc_h_ = nullptr, *dc_gate_ = nullptr, *dc_beta_ = nullptr, *dc_idx_ = nullptr;
    float* dc_hist_ = nullptr;
    void* dc_arena_ = nullptr;
    int32_t *h_commitd_ = nullptr, *m_commitd_ = nullptr, *commitd_ = nullptr;
    std::map<std::vector<int>, cudaGraphExec_t> save_bm_, dcommit_bm_;
    bool capture_deferred(const int* rows, int S, int hbase, std::string& err);
    /// Free the graph pair of the least recently used batch layout other than `keep` (`evicted`: there was one).
    bool evict_batch_graph(const std::vector<int>& keep, bool& evicted, std::string& err);
    /// cudaGraphInstantiate; out of VRAM, batch layouts other than `key` are evicted (LRU) until it fits, and then (a
    /// single-request window, `keep_t` > 0) the graphs of the other window sizes, largest first (#3702): a window size
    /// whose graph does not fit used to end the engine ("verify: instantiate: out of memory"); a freed size is captured
    /// again when it is next used.
    bool instantiate_evicting(cudaGraphExec_t& ex, cudaGraph_t graph, const std::vector<int>& key, const char* what,
                              std::string& err, int keep_t = -1, unsigned long long flags = 0);
    /// Free the graph of the largest window size other than `keep_t` (and the prelaunched one); `evicted`: there was one.
    bool evict_window_graph(int keep_t, bool& evicted, std::string& err);
    bool no_window_eviction_ = false;   ///< capture_all wants every size at once: it must not free one to capture another
    // STRATA_GRAPH_PRIO (record_window / capture): the captured tails of the priority streams at their joins and the
    // main stream's tails at their forks, so that a capture that dropped the stream priorities can be repaired per
    // node (apply_node_priorities walks each tail back to its fork)
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)   // (the three methods are no-ops there)
    std::vector<cudaGraphNode_t> prio_forks_;
    std::vector<std::pair<cudaGraphNode_t, int>> prio_tails_;
#endif
    void note_prio_fork(cudaStream_t from);
    void note_prio_tail(cudaStream_t on, int prio);
    void apply_node_priorities(cudaGraph_t graph, int T);
    void collect_profile();   ///< STRATA_VERIFY_PROFILE: add the last window's stamps to prof_sum_
    void accumulate_profile(const unsigned long long* stamps);
    void accumulate_profile_inner(const unsigned long long* stamps);   ///< one window's stamps (host copy) into prof_sum_
    // pipelined windows (pl_launch ...)
    cudaStream_t ext_stream_ = nullptr;   ///< set_stream: the stage's shared stream (not destroyed here)
    bool always_publish_ = false;
    bool adapt_gate_wait(int32_t layer, std::string& err);
    AdaptGate* adapt_gate_ = nullptr;     ///< 0.1.43: STRATA_ADAPT_OVERLAP's gate (null: off)
    void pl_stage(int T, const int32_t* tokens, int64_t pos0, const int32_t ple_prev[2]);
    cudaEvent_t ev_done_ = nullptr, ev_commit_ = nullptr;
    unsigned long long* prof_pin_ = nullptr;   ///< pinned host copy of the stamps (pipelined windows)
    bool fl_active_ = false, fl_prof_ = false, fl_ple_ = false, commit_live_ = false, pl_prestaged_ = false;
    int fl_T_ = 0;
    int64_t fl_k_ = 0, fl_total_ = 0;
    double fl_since_ms_ = 0, fl_flush_ms_ = 0, fl_launch_ms_ = 0;
    int32_t pl_prev_[2] = {-1, -1};
    std::vector<uint32_t> pl_ple_rows_;   ///< the window's PLE rows (T x PLE_N_HEADS), gathered when layer 0 is served
    bool capture_commit_batch(const int* rows, int S, int hbase, std::string& err);
    bool stage_batch(const int* rows, int S, int hbase, const int32_t* tokens, const int64_t* pos, std::string& err);
    strata::kernels::SamplerParams sampling_ = [] {
        strata::kernels::SamplerParams s;
        s.greedy = true;
        s.temperature = 0.0f;
        return s;
    }();   ///< greedy by default; per-request via set_sampling
    const int32_t* hist_d_ = nullptr;   ///< penalty-history row (set_history); null = no penalties apply
    const int32_t* spec_q_ = nullptr;   ///< set_spec_q: the drafter's q rows for the next window (host)
    int spec_nq_ = 0;
    bool spec_measure_ = false;   ///< explicit per-call measure borrow
    int32_t* d_spec_ = nullptr;          ///< device: kVerifyMaxT draft ids, then kVerifyMaxT q rows
    int hist_len_ = 0;
    bool head_sampling_ = true;          ///< set_head_sampling
#if defined(STRATA_GPU_METAL)
    void* metal_ = nullptr;              ///< the Metal decode window (metal/src/engine/window.cpp): init/run/commit go there
#endif
    std::vector<float> logit_bias_host_;
    float* logit_bias_device_ = nullptr;
    int device_ = -1;                    ///< the device `init` ran on: run/commit switch to it (layer split)
    unsigned long long* rr_stats_ = nullptr;   ///< STRATA_ROUTE_RESIDENT's counters on device_ (#1578); null: none
    std::atomic<bool> released_{false};  ///< #267: release_gpu_waits ran (maybe on the watchdog thread): no more windows
    bool all_resident_ = false;           ///< 100% of experts in [lb_, le_) resident in VRAM: zero-doorbell graph
    /// #871: the zero-doorbell graph plans from the device residency table alone, so it is only right while every
    /// expert of the stage is in VRAM.  A prompt loan, a VRAM shrink or an adaptive swap marks some -1 for a while:
    /// every window then runs the doorbell graph (exec_nr_ / the batch key's top bit), the same as a stage that never
    /// was all-resident.  refresh_ar() looks at the host table before each window.
    bool ar_off_ = false;
    bool ar_on() const { return all_resident_ && !ar_off_; }
    void refresh_ar();
    uint32_t* h_plan_err_ = nullptr; uint32_t* m_plan_err_ = nullptr;   // set by resident_plan on a -1 (all-resident graph)
    const int32_t* h_res_ = nullptr;      ///< the host residency table (VerifyHits::h_res)
    bool device_plan_ = false;            ///< E-6: resident-only layers planned on the device (STRATA_VERIFY_DEVICE_PLAN)
    uint32_t* skip_ = nullptr;            ///< E-6: per group, the ring whose plan the device built (0: the host's)
    unsigned* qcnt_ = nullptr;             ///< S26 STRATA_QFUSE: the HC read's q8_1 group counters (n_embd / 32)
    unsigned long long* slot_off_d_ = nullptr;   ///< E-6: the slot offsets on the device
    int64_t lb_ = 0, le_ = -1;           ///< set_stage: the layers this verifier runs (-1: to the last)
    int handoff_rows_ = 0;               ///< set_handoff_rows (0: kVerifyNarrowT)
    const float* hand_in_ = nullptr;
    float* hand_out_ = nullptr;
    uint32_t* hflag_in_h_ = nullptr;     ///< set_handoff_flags: the incoming hand-off's ready word (host / device)
    uint32_t* hflag_in_d_ = nullptr;
    uint32_t* hflag_out_h_ = nullptr;    ///< ... and the outgoing one's
    uint32_t* hflag_out_d_ = nullptr;
    uint32_t* hcount_d_ = nullptr;       ///< handoff_publish's block counter (device memory)
    uint32_t* herr_h_ = nullptr;         ///< the bounded hand-off wait gave up (mapped; the reading stage's)
    uint32_t* herr_d_ = nullptr;
    uint32_t* hdrop_h_ = nullptr;        ///< STRATA_TEST_HANDOFF_DROP: the writing stage withholds its flag (mapped)
    uint32_t* hdrop_d_ = nullptr;
    bool handoff_failed(std::string& err) const;   ///< the bounded hand-off wait timed out in the last window
    int prelaunched_ = 0;                ///< prelaunch ran for a window of this size: run skips staging + launch
    /// The overlapped split: stage this window's inputs and launch its graph now (the graph waits on the hand-off
    /// flag); `run` then only serves the host side.  Called on a helper thread while the previous stage runs.
    bool prelaunch(int T, const int32_t* tokens, int64_t pos0, std::string& err);
    bool chain_overlap() const;          ///< this stage hands off through a flag to a next stage that waits on it
    Verifier* next_ = nullptr;
    void* next_user_ = nullptr;
    bool ple_stage() const { return lb_ <= 1 && 1 < le_; }   ///< holds layer 1, where the PLE block runs
    void stage_inputs(int T, const int32_t* tokens, int64_t pos0);
    bool staged_ = false;
    bool copy_used_ = false;
    bool capture_commit(std::string& err);
    bool record_window(int T, cudaStream_t cs, std::string& err);
    // #649: STRATA_VERIFY_TRACE=1 - a host event ring (trace_ev) and GPU breadcrumbs: the profiler's stamp points,
    // per layer and token group, written to mapped memory (null when the trace is off)
    void trace_ev(const char* what, int64_t step, int64_t layer, int64_t aux) const;
    unsigned long long* trace_h_ = nullptr;
    unsigned long long* trace_m_ = nullptr;
    size_t trace_n_ = 0;
    static constexpr int kProfPer = 36;              // stamps per layer (32+3 ym split, hc-read second
                                      // half's up-stamp at slot 32 = the next layer's slot 0: D8)
    bool prof_on_ = false;
    unsigned long long* prof_ = nullptr;              // device: n_layers * kProfPer + 4 stamps
    std::vector<unsigned long long> prof_h_;
    double prof_sum_[2][kProfPer] = {};   // [GDN / QSA layers][stage]
    int64_t prof_windows_ = 0;
    double prof_t_sum_[9][2][kProfPer] = {};   // the same per window size T (1..8: wider windows are not profiled), for profile_by_t_report
    int64_t prof_t_win_[9] = {};
    bool rs_on_ = false;                      // STRATA_ROUTE_STATS
    unsigned long long* rs_dev_ = nullptr;    // [9 T][n_layers][8 rows]: experts of row r that no earlier row of the window routed to
    int64_t rs_win_[9] = {};

    const WeightTable* wt_ = nullptr;
    const ModelGeometry* g_ = nullptr;
    SessionState* ss_ = nullptr;
    VerifyHits hits_;
    const NativeHead* head_ = nullptr;
    int max_t_ = 0;
    float* ple_key_ = nullptr;   ///< STRATA_PLE_BATCH: the window rows' PLE key / value projections
    float* ple_val_ = nullptr;
    int last_t_ = 0;
    int64_t last_pos0_ = 0;
    int32_t last_tokens_[strata::kernels::kWindowMaxT] = {};
    int64_t n_vocab_ = 0;
    cudaStream_t cs_ = nullptr;
    cudaStream_t sh_cs_ = nullptr;
    cudaEvent_t ev_fork_ = nullptr, ev_join_ = nullptr, ev_hq_ = nullptr;
    // STRATA_DF_BRANCH: a layer's mixer work that reads only the layer's input, captured as parallel graph branches
    // on these side streams (record_window); the same kernels on the same inputs, only their order is freer
    bool df_branch_ = false;
    cudaStream_t df_side_[2] = {};
    cudaEvent_t df_fork_ = nullptr;
    cudaEvent_t df_join_[2] = {};
    // [sel_long_][T]: 1 is the long-context QSA select (STRATA_SELECT_LONG=1: multi-CTA top-k, shuffle-free decode
    // scores; see run()); without it only [0] is ever captured
    // (per window size T = 1..kWindowMaxT: a solo window is at most 8 rows, but a merge window of a wide batch is not)
    cudaGraphExec_t exec_[2][strata::kernels::kWindowMaxT + 1] = {};
    cudaGraphExec_t exec_nr_[2][strata::kernels::kWindowMaxT + 1] = {};   // #871: the doorbell variant of a stage that is all-resident otherwise
    bool launched_[2][strata::kernels::kWindowMaxT + 1] = {}, launched_nr_[2][strata::kernels::kWindowMaxT + 1] = {};   ///< this graph has been launched once (early_ple_flag: a cold first launch raises the PLE flag first)
    int sel_long_ = 0;
    // fork 34a35db (STRATA_SIDE_MIXER=1): the GDN gates and the QSA indexer on a branch of the window's graph beside the
    // mixer's projections, joined before the recurrence / the attention
    cudaStream_t side_ = nullptr;
    cudaEvent_t mfork_ = nullptr, mjoin_ = nullptr;
    // fork b063c13 (STRATA_SIDE_MIXER=2): a second branch for the GDN gate projection and the QSA queries' chain
    cudaStream_t side2_ = nullptr;
    cudaEvent_t pfork_ = nullptr, pjoin_ = nullptr;
    cudaGraphExec_t commit_exec_ = nullptr;

    // mapped staging (host pointer, device alias)
    int32_t* h_tok_ = nullptr;   int32_t* m_tok_ = nullptr;     // T
    int32_t* h_step_ = nullptr;  int32_t* m_step_ = nullptr;    // T * kStepCount
    int32_t* h_pos_ = nullptr;   int32_t* m_pos_ = nullptr;     // T * n_head
    int32_t* h_commit_ = nullptr; int32_t* m_commit_ = nullptr; // [n_keep, n_keep-1, pos_0 .. pos_{T-1}]
    float* h_ple_ = nullptr;     float* m_ple_ = nullptr;       // T * n_embd
    int32_t* h_out_ = nullptr;   int32_t* m_out_ = nullptr;     // T argmax ids
    // STRATA_TAP_FILE (off by default; nothing is allocated or captured without it): the DFlash2 experiment's hook.
    // After each layer in STRATA_TAP_LAYERS (default 5,15,25,35,45) the window copies the layer's completed output
    // (the residual with its pending FFN write applied, the mean of the hc streams) per row; run() appends
    // (pos, token, taps) records to the file.
    bool tap_on_ = false;
    std::vector<int> tap_req_;                    ///< set_taps
    std::vector<int> tap_layers_;                 ///< the tapped layer ids (zero-based, completed outputs)
    std::vector<int> tap_slot_;                   ///< layer -> slot in tap_layers_, or -1
    float* tap_dev_ = nullptr;                    ///< [max_t][n_taps][n_embd]
    float* tap_scratch_ = nullptr;                ///< one row's streams, [hc][n_embd]
    float* h_tap_ = nullptr;     float* m_tap_ = nullptr;       ///< the mapped copy of tap_dev_
    std::FILE* tap_file_ = nullptr;
    void tap_write(int T, const int32_t* tokens, int64_t pos0);
    uint32_t* h_nf_ = nullptr;   uint32_t* m_nf_ = nullptr;     // #879: kVerifyMaxT * kNonfiniteBlocks flags
    int nf_row_ = -1;                     ///< nonfinite_row()
    int64_t nf_pos_ = -1;                 ///< nonfinite_pos()
    uint32_t nf_mask_ = 0;                ///< nonfinite_in(): bit t = row t
    void note_nonfinite(int T, const int64_t* pos_b);   ///< the window's flags (+ STRATA_NAN_INJECT) -> nf_row_
    float* h_x_ = nullptr;       float* m_x_ = nullptr;         // doorbell payload: T * n_embd
    int32_t* h_ids_ = nullptr;   int32_t* m_ids_ = nullptr;     // T * k
    float* h_w_ = nullptr;       float* m_w_ = nullptr;         // T * k
    uint32_t* h_seq_ = nullptr;  uint32_t* m_seq_ = nullptr;
    uint32_t* h_flag_ = nullptr; uint32_t* m_flag_ = nullptr;
    uint32_t* h_flagA_ = nullptr; uint32_t* m_flagA_ = nullptr;  // the GPU plan is in place
    uint32_t* h_flagB_ = nullptr; uint32_t* m_flagB_ = nullptr;  // the PCIe share's DMA copies have landed
    // set_pcie_balance: per step (layer x group, stride 2 groups) the GPU's clock as the plan arrives, as the PCIe
    // part starts and as it waits for the CPU rows (4 words a step); the pool's ms, its experts, the PCIe ones, and
    // whether the step had a plan
    unsigned long long* h_bal_ = nullptr; unsigned long long* m_bal_ = nullptr;
    std::vector<double> bal_p_;
    std::vector<int32_t> bal_nc_, bal_np_;
    std::vector<char> bal_ok_;
    PcieModel* bal_ = nullptr;
    bool bal_paused_ = false;
    void pcie_balance_window(int64_t steps, int G);
    cudaEvent_t commit_done_ = nullptr;   // recorded after an async commit (set_commit_async); see wait_commit
    bool commit_pending_ = false;
    cudaStream_t copy_ = nullptr;                                 // the copy engine's stream (DMA of missed experts)
    struct FlagSet { uint32_t* flag; uint32_t value; };
    FlagSet flag_sets_[2 * 64 * 2] = {};                          // host-function arguments, one per (layer, group)
    static void fetch_dma(void* ctx, const uint8_t* const* src, int n, size_t bytes);
    static void raise_flag(uint32_t* flag, uint32_t value);
    int32_t* h_plan_ = nullptr;  int32_t* m_plan_ = nullptr;     // counts | start | dst | tok | ptr (as int32 pairs)
    int64_t plan_i32_ = 0;                                        // int32 words in the plan block
    GpuPlanSink sink_;
    uint32_t cur_layer_ = 0;
    static void publish_plan(void* ctx);
    void set_plan_slot(int grp);
    bool split_ = false;   // opt-in (--spec-split): exact but slower, see the overlap study
    std::function<bool(cudaStream_t, std::string&)> pre_launch_;   // set_pre_launch
    int groups_[strata::kernels::kWindowMaxT + 1] = {};
    float* h_ymiss_ = nullptr;   float* m_ymiss_ = nullptr;     // T * k * n_embd

    // device
    void* arena_ = nullptr;
    int32_t *tok_ = nullptr, *step_ = nullptr, *pos_ = nullptr, *commit_ = nullptr;
    float *ple_ = nullptr, *emb_ = nullptr, *R_ = nullptr, *mixed_ = nullptr, *bo_ = nullptr;
    float *inj_ = nullptr, *inj2_ = nullptr, *lo_ = nullptr, *rs_ = nullptr, *xn_ = nullptr;
    uint8_t* xq_ = nullptr;                                   // T columns of q8_1
    uint8_t* xil_ = nullptr;                                  // fork F4: the interleaved copy of xq_'s 2-4 columns
    uint8_t* sh_xq_ = nullptr;                                // T columns of q8_1 for shared expert branch
    float *qkv_L_ = nullptr, *h_L_ = nullptr, *gate_L_ = nullptr, *beta_L_ = nullptr;   // per GDN layer
    float *z_ = nullptr, *y_ = nullptr, *y_dummy_ = nullptr;
    float *qfull_ = nullptr, *qcur_ = nullptr, *kcur_ = nullptr, *vcur_ = nullptr, *idx_raw_L_ = nullptr;
    float *qidx_ = nullptr, *scores_ = nullptr, *attn_ = nullptr, *attn32_ = nullptr, *attn_scratch_ = nullptr;
    float* tail_snap_ = nullptr;                              // per QSA layer
    int32_t* sel_ = nullptr;
    int32_t* topk_scratch_ = nullptr;   // qsa_block_topk_long's multi-CTA exchange (STRATA_SELECT_LONG=1 only)
    float *logits_ = nullptr, *w_ = nullptr, *shared_ = nullptr, *parts_ = nullptr, *hit_out_ = nullptr;
    int32_t *ids_ = nullptr, *hit_slot_ = nullptr, *hit_dst_ = nullptr, *hit_count_ = nullptr;
    int32_t* plan_ = nullptr;                                     // device copy of the plan block
    uint8_t* staging_ = nullptr;                                  // VRAM slots for the PCIe share of the misses
    static constexpr int64_t kStagingBlobs = 16;
    static constexpr int64_t kPcieGroupRows = 4;                  // the PCIe call's groups side by side (of <= 16)
    uint8_t* hit_xq_ = nullptr;
    uint8_t* nat_xq_ = nullptr;   // plan v0.3 P6: q8_1 activations for a native pack's grouped experts
    float* nat_xr_ = nullptr;     // APR: the Hadamard-rotated expert input (the shared expert and the CPU keep xm); nullptr
                                  // unless some layer's gate/up is folded
    const float* had_sx_ = nullptr;   // APR: device sign vectors for x (n_embd) and h (n_ff); nullptr = identity
    const float* had_sh_ = nullptr;
    float* hit_xs_ = nullptr;
    void* hit_scratch_ = nullptr;
    float *head_mixed_ = nullptr, *head_inj_ = nullptr, *head_logits_ = nullptr;
    uint16_t* sh_bf16_ = nullptr;
    uint8_t* arg_scratch_ = nullptr;   ///< argmax_rows' partials and counters
    int32_t* one_ = nullptr;           ///< device {1}: the n_keep of a one-token window, which commits itself
    float *sh_gate_ = nullptr, *sh_up_ = nullptr, *sh_g_ = nullptr;
    float* hist_snap_ = nullptr;                              // T * NG_HIST * NG_HC_DIM
    // latent MoE (new files; non-latent paths leave every pointer null)
    bool latent_ = false;
    strata::core::LatentDense lat_dense_;
    int lat_fmt_ = 8;   // Q4DU: the dense pack's format (8 = Q8_0, 4 = Q4_0), set at open
    strata::core::LatentA lat_a_;
    std::vector<int32_t> lat_labels_;                     ///< [layers*E] per-layer group labels, from labels.bin
    int32_t* lat_labels_dev_ = nullptr;                   ///< device copy, uploaded once at init
    uint8_t *lat_d_ = nullptr, *lat_u_ = nullptr;         ///< every layer's D/U Q8_0, resident (uploaded once at init)
    const uint8_t *lat_d_shared_ = nullptr, *lat_u_shared_ = nullptr;   ///< the pair from set_shared_latent
    float *lat_xl_ = nullptr, *lat_yl_ = nullptr;         ///< [T*G, ELL] and [T*K, ELL]
    void *lat_xq_ = nullptr, *lat_xlq_ = nullptr, *lat_uq_ = nullptr, *hq_dev_ = nullptr;
    uint8_t* h_latxlq_ = nullptr;   ///< mapped host mirror of the D stage's f32 lat_xl_ rows (the CPU share's A input; row t*G + g)
    uint8_t* m_latxlq_ = nullptr;
    uint8_t* m_latact0_ = nullptr;   ///< DESIGN-4: the D rows quantized Q8_0 (mapped host, the jobs read it)
    uint8_t* lat_act0_dev_ = nullptr;   ///< the write-path cut: the device scratch the kernel fills (fixed ~0.6MB)
    uint32_t* h_xlseq_ = nullptr;   ///< edge0: the D stage's doorbell (STORE of l+1 after the D2H copy node)
    void* m_xlseq_ = nullptr;
    int64_t lat_blob_ = 0;
    int64_t lat_H_ = 0, lat_ELL_ = 0, lat_I_ = 0; int lat_G_ = 0, lat_E_ = 0, lat_K_ = 0;
    int64_t cap_ = 0, max_blocks_ = 0, attn_scratch_floats_ = 0;
};

}  // namespace strata::core
