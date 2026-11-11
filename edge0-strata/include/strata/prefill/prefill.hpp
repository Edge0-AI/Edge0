// include/strata/prefill/prefill.hpp - plan v0.3 P5: batched prompt processing.
//
// The prompt's positions [pos0, pos0 + n) are processed in chunks of `chunk` tokens through all 48 layers, leaving
// the session state (GDN recurrence and conv state, QSA KV pools and indexer, PLE history) where the token path
// would have left it; the decode loop then continues with the next token.  Per layer: the projections are
// tensor-core GEMMs (quantized weights dequantized to FP16 on the fly, BF16 weights as they are), the recurrences
// walk the chunk inside one kernel, and the routed experts are grouped by expert: resident ones are read from the
// VRAM tier, the others streamed from the host arena through a pinned ring on a copy stream.
//
// Requires the native weights (`--native`): every quantized projection must carry its GGUF blocks.
#pragma once

#include <array>

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"
#include "strata/core/latent_share.hpp"
#include "strata/core/weights.hpp"

#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <string>

namespace strata::core { class PeerExperts; }
namespace strata::kernels::cpu { class ExpertPool; }

namespace strata::prefill {

struct PrefillStats {
    int64_t tokens = 0;
    int64_t chunks = 0;
    double ms_total = 0;
    double ms_experts_host = 0;     ///< host time staging non-resident experts
    int64_t experts_streamed = 0;   ///< expert blobs copied host -> device
    int64_t experts_dma = 0;        ///< ...of which straight from the pinned arena (no CPU copy)
    int64_t experts_resident = 0;   ///< expert-layer groups served from the VRAM tier
    int64_t experts_direct = 0;     ///< ...of which MMQ read in their cache slots, not gathered (STRATA_MMQ_DIRECT=1)
    int64_t experts_cpu = 0;        ///< ...computed on the CPU pool instead of streamed (STRATA_PREFILL_CPU_SHARE)
    double cpu_share = 0;           ///< ...the share of the streamed ones it took last (measured by default)
    double ms_ple = 0;
};

}  // namespace strata::prefill
namespace strata::core { class MtpDrafter; }
namespace strata::prefill {

/// HIP: one tiny BF16 and one tiny FP16 product on a private handle, so the first prompt does not pay hipBLAS's one-time
/// library loads (about 2.5 s on an R9700); once per device, later calls wait for the first and return.  Safe to call from a
/// thread before the expert arena loads.  STRATA_PREFILL_WARM=0 makes it a no-op.  Nothing on CUDA / SYCL.
void warm_gemm_libraries(int device);


class Prefill {
public:
    /// E-9: the draft layer's K/V for prompt cells [cell0, cell0 + n) from their final residual rows `R_rows`
    /// (device) and `next_tokens` (host: the token at cell+1), in batches through this path's GEMMs and its idle
    /// scratch - call it from on_chunk.  false with `err` empty: not applicable here (a ring or hybrid K/V, another
    /// device, too little scratch; STRATA_MTP_BATCH=0), the caller runs the drafter's own pass.  Not bit-identical to
    /// that pass (FP16 GEMMs instead of Q8_1 activations): the drafts may differ, never the target's tokens' logits.
    /// `why` (optional): on a decline (false, `err` empty) the reason, a static string.
    bool draft_kv(core::MtpDrafter& mtp, const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0,
                  std::string& err, const char** why = nullptr);
    Prefill();
    ~Prefill();
    Prefill(const Prefill&) = delete;
    Prefill& operator=(const Prefill&) = delete;

    /// Frees every buffer, stream and event `init` made (as the destructor does) and starts over empty, so `init` can
    /// run again - with a smaller chunk when the first one did not fit.  The stage range (`set_stage`) and the
    /// callbacks stay.  The device `init` ran on must be current.
    void reset();

    /// `host_res`: the static residency table (n_layers x n_expert, slot or -1) or null; `cache` its slots.
    /// `borrow`/`borrow_bytes`: device memory to carve every buffer from (the top slots of the expert cache,
    /// lent for the prompt and refilled after it); null = allocate normally.
    bool init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
              core::ExpertSource* src, const core::ExpertCache* cache, const int32_t* host_res, int64_t chunk,
              void* stream, std::string& err, void* borrow = nullptr, uint64_t borrow_bytes = 0,
              bool borrow_spill = false);

    /// With borrowed buffers: lay them out again for chunks of `chunk` tokens (at most `init`'s) in `borrow` - a
    /// request lends only the slots its prompt needs.  The stream must be idle (between prompts).
    bool relayout(int64_t chunk, void* borrow, uint64_t borrow_bytes, std::string& err);
    int64_t chunk() const;

    /// The share of the streamed experts' bytes DMA-able straight from pinned RAM (1 = all).  Sizes the streamed
    /// ring (a big one only pays when the copy engine, not the host copies, is the limit); set before bytes_needed.
    static void set_pinned_share(double share);
    static double pinned_share();
    /// The chunk size from which a chunk streams every expert the GPU does not hold, routed or not; a smaller one
    /// stages only the experts it routes to.  1024, or STRATA_PREFILL_STREAM_MIN, or the CPU share's limit once that
    /// is armed (3072 when STRATA_PREFILL_CPU_SHARE is set).
    static int64_t stream_all_min_tokens();
    /// --prefill-pipe (opt-in): the chunk a layer split's first stage reads an `n`-token prompt in, for `stages`
    /// stages, chunks of at most `cap` (the buffers' size) and a stage-chunk cost of b + a C with b/a = `ratio` tokens
    /// (0: unknown - STRATA_PREFILL_EQUAL's rule: the same chunk count as `cap` gives, evened out when its last chunk
    /// has stream_all_min tokens or more): the chunk count with the pipeline's least time, see the definition. `cap`
    /// when that is today's split or not 3% better than it. A function of its arguments only.
    static int64_t pipeline_chunk(int64_t n, int64_t cap, int stages, double ratio);
    /// --prefill-pipe for this run: 0 off (the buffers' chunks), 1 today's chunk count evened out (STRATA_PREFILL_EQUAL's
    /// rule), a number > 1 the rig's b/a in tokens (setup's --calibrate measures it). Default: STRATA_PREFILL_PIPE, else 0.
    static void set_pipe(double value);
    /// The same for the next prompts only (the `pipe=` request key, setup's calibration); < 0 = the run's own value.
    static void set_pipe_request(double value);
    /// The --prefill-pipe value in force (the request's, else the run's).
    static double pipe_setting();
    /// What a --prefill-pipe value plans by: -1 off, 0 today's chunk count evened out, else b/a in whole tokens (two
    /// spellings of one value, e.g. a request key and the flag, compare equal).
    static int64_t pipe_key(double value);
    /// The stages of the prompt path from this one on (1 without a layer split).
    int stage_count() const;
    /// --prefill-pipe on a layer split's first stage: the chunk an `n`-token segment is read in when it is read whole
    /// (pipeline_chunk over chunk(), the stage count and pipe_setting()), or 0 when the pipe is off here (one stage,
    /// value 0, or a later stage) and the buffers' chunk applies.  The engine plans a segment once, on its whole
    /// length, and reads every part of it with force_chunk, so its chunks do not depend on how it is cut into runs.
    int64_t plan_chunk(int64_t n) const;
    /// The next run on this (first) stage only: chunks of `chunk` tokens (at most chunk()) instead of a plan, from
    /// its first token; `single`: the segment is one chunk (the stage helper may bind); `planned`: the chunks came
    /// from a --prefill-pipe plan (such a read takes no CPU share: its chunks overlap on the stages by design).
    void force_chunk(int64_t chunk, bool single, bool planned);
    /// One log line for a planned segment (the same as a whole read logs when its plan differs from chunk()).
    static void log_plan(int64_t n, int64_t chunk, int stages);
    /// The expert blobs this stage and the later ones copied host -> device so far (PrefillStats::experts_streamed).
    int64_t chain_experts_streamed() const;
    /// #340: the streamed ring's slot count for chunks that stream every expert, instead of the pinned-share rule
    /// (0 = that rule). Set before any `bytes_needed`/`init` (both count the ring); STRATA_PREFILL_RING still wins.
    static void set_ring_override(int slots);
    /// #583: the auto chunk scan's byte-budget ring, for chunks above `small_max` (0.1.39's auto chunk: a prompt that
    /// fits it keeps 0.1.39's ring).  0 slots = none.  A layer split's set_ring_override and STRATA_PREFILL_RING win.
    static void set_ring_budget(int slots, int64_t small_max);

    /// Device bytes `init` needs for a chunk of `chunk` tokens (what a borrowed region must hold).  `src`: init gets an
    /// ExpertSource (carve picks the fused MoE layout only then).
    static uint64_t bytes_needed(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk,
                                 bool src = true);

    /// The same without the streamed ring: what the chunk's own buffers cost.  The auto chunk scan sizes the chunk
    /// first and hands the ring what the chunk leaves over, so it needs the chunk priced on its own.
    /// What `init` allocates when the prompt path OWNS its buffers: each cudaMalloc rounded up to a 2 MiB page and the
    /// ring as one allocation (the startup sizing of a cache without a loan).
    static uint64_t bytes_needed_owned(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk,
                                       bool src = true);
    static uint64_t bytes_needed_no_ring(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk,
                                         bool src = true);

    /// The streamed ring's byte budget as a slot count for this pack (the measured slot count x Q2_0's blob, over
    /// max_blob, never past ring_cap()):
    /// what the auto chunk scan treats as a full ring.  A slot is one whole blob, so a pack with bigger blobs than
    /// Q2_0's gets fewer of them for the same bytes - 384 on Q2_0, 199 on a 2.54 MiB-blob IQ3_S pack.
    static int64_t ring_max_slots();
    /// 0.1.39's ring for this PC (1024 fused / 384 pinned / 96), within ring_cap().
    static int64_t ring_default_slots();
    /// #583: the ring the auto scan keeps full, given the chunk 0.1.39's rule picked: its byte budget where that rule's
    /// chunk was small (< 6144), else 0.1.39's ring (measured: shrinking it for a bigger chunk lost there).
    static int64_t ring_cap_for(int64_t old_chunk);

    /// What the ring actually resolves to for a chunk of `chunk` tokens, after the override, STRATA_PREFILL_RING
    /// and the pinned-share rule - the slot count `init` lays out.  The engine reports it on its INFO line so the
    /// Monitor tab shows the pair the run really got, not what it asked for.
    static int64_t ring_slots_for(int64_t chunk);

    /// 0.1.39b (#583, the default): the ring as a byte budget, the loan's corrected count and the auto chunk scan that
    /// keeps the ring full.  STRATA_RING_BYTES=0: 0.1.39's ring, loan and chunk list.
    static bool ring_bytes_enabled();

    /// STRATA_MMQ_DIRECT: the guard each sized expert-cache slot keeps after its blob (ExpertCache::set_slot_guard),
    /// so the prompt MMQ can read a resident expert in its slot.  0 unless STRATA_MMQ_DIRECT is 1 or 2 on a CUDA
    /// build with the pointer-table MMQ (STRATA_MMQ_TAB).
    static size_t slot_guard_bytes();

    /// STRATA_EXPERT_NIBBLE=1: whether every converted layer's experts reach the prompt path's MMQ gather only (the
    /// GGUF blocks rebuilt from the 4-bit slots); false with the first layer and the reason in `why`.  Pure: the
    /// layers' formats, the environment and the visible GPUs (generate.cpp asks once, before anything is allocated).
    static bool nibble_prompt_check(std::string& why);

    /// Positions [pos0, pos0 + n) holding `tokens`; `ss.ple_prev` must be the two tokens before pos0 (oldest
    /// first, -1 for none) and is advanced to the last two of these.
    bool run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err);

    const PrefillStats& stats() const { return stats_; }

    /// multi-GPU: the experts the peer GPU holds are computed THERE for every prompt chunk (up to `cap_rows` routed
    /// rows per layer; the rest of the peer's experts are read by this GPU over P2P).  Allocates the peer's buffers for
    /// chunks of up to init's `chunk` tokens.  Needs P2P between the two cards.
    bool set_peer(core::PeerExperts* peer, int64_t cap_rows, std::string& err);
    /// A session-level shared latent D/U pair (allocated and uploaded elsewhere, e.g. the engine): the init
    /// keeps no copy of its own and skips its upload; the prompt path only reads the pair.  Set before the first `run`.
    void set_shared_latent(uint8_t* d, uint8_t* u) { strata::core::latent_shared_d() = d; strata::core::latent_shared_u() = u; }

    /// Plan v0.3 P6: called after every chunk with the chunk's final multi-stream residual rows (device,
    /// T x hc*n_embd, valid until the next chunk) and the chunk's first position; the MTP draft layer builds its
    /// K/V from them.  The prefill stream is synchronized before the call.
    std::function<bool(const float* R_rows, int64_t T, int64_t pos0, std::string& err)> on_chunk;
    /// DFlash2 prompt taps (CUDA / HIP, single stage): before `run`, ask for the completed output of each layer in `layers` (zero-based; the
    /// mean of the hc streams) for the prompt rows at positions >= `first_pos`, at most `cap_rows` rows per chunk (a staging buffer of
    /// cap_rows x layers x n_embd floats is allocated at the first chunk).  After each chunk that has such rows, on_taps(taps
    /// [rows][layers][n_embd] device, rows, first position, err) runs with the stream synchronized.  clear_taps frees the staging.
    void set_taps(const std::vector<int>& layers, int64_t first_pos, int64_t cap_rows) { tap_layers_ = layers; tap_first_ = first_pos; tap_cap_ = cap_rows; }
    void clear_taps();
    std::function<bool(const float* taps, int64_t rows, int64_t pos0, std::string& err)> on_taps;

    /// Layer split: called by every stage when it has read a chunk, with the position reached, while its own state
    /// is still at that chunk's end (its stream synchronized; the last stage calls it just before `on_chunk`).  An
    /// earlier stage is a chunk or more ahead of the last one by the time `on_chunk` runs, so this is where a
    /// mid-prompt checkpoint takes each stage's part.  Runs on that stage's thread, with its device current.
    std::function<bool(int64_t done, std::string& err)> on_stage_chunk;

    /// Checked before every chunk: true stops the prompt early (`run` returns false with err "cancelled").
    std::function<bool()> should_stop;
    /// The first stage only, checked before every chunk but the first: true ends the run there as if the prompt
    /// ended there.  `run` returns true once every stage has read the tokens before that chunk, and `stopped_at()`
    /// says how many of the run's tokens that is (-1 when the run read them all).  A read beside the batch slots
    /// (STRATA_PIPED_CONCURRENT) gives way to a waiting shorter request there.
    std::function<bool()> stop_at_chunk;
    /// 149-prefetch (STRATA_RAM_ADAPT2): a layer's routed expert ids (T x k, -1 = none) after the prompt path downloaded them
    /// for its grouping on the host; a policy may count them. Not called on the paths that keep the ids on the GPU.
    std::function<void(int64_t layer, const int32_t* ids, int64_t n)> on_routes;
    int64_t stopped_at() const { return stopped_at_; }
    /// KVA: the end of the positions the last run approximated (they are a prefix: chunks from the run's first position
    /// on), i.e. the first position whose final rows are real; 0 when that run approximated nothing.  Raised chunk by
    /// chunk, before on_chunk reports the chunk, so the draft layer can skip the chunk's cells.  Reset by every run.
    int64_t kva_floor() const { return kva_floor_; }
    /// Grows this stage's (and the later stages') pinned upload slots to what a text chunk of `T` tokens uploads, so
    /// a run of such chunks allocates no pinned memory (STRATA_PIPED_CONCURRENT: an allocation or free on the read's
    /// thread takes the driver's lock).  Before the run, with nothing of this path in flight.  False: a slot could
    /// not be had (it then grows at first use, as before).
    bool reserve_uploads(int64_t T);

    /// The vision path: HOST rows (n_embd floats) indexed by absolute position, read in place of the token
    /// embedding where non-null (an image's <|image_pad|> cells).  Null (default): every position embeds its token.
    const float* const* embd_rows = nullptr;

    /// LAYER SPLIT (multi-GPU): this prompt path runs layers [layer_begin, layer_end) (-1: to the last) on the
    /// device `init` runs on.  A stage that does not start at layer 0 reads each chunk's residual rows from the
    /// previous stage instead of embedding the tokens; a stage that does not end at the last layer copies its rows
    /// to pinned host buffers (two, allocated by `init`) and runs `next` on them - on a thread, so the next stage
    /// reads chunk c while this one reads chunk c + 1.  `on_chunk` belongs on the last stage.  Set before `init`.
    void set_stage(int64_t layer_begin, int64_t layer_end, Prefill* next) {
        stage_lb_ = layer_begin; stage_le_ = layer_end; next_ = next;
    }

    /// LAYER SPLIT: `helper` is another stage's prompt path (another GPU).  A prompt of one chunk runs the stages one
    /// after the other, so while this stage reads it the helper's GPU idles: it then streams a share of this stage's
    /// non-resident experts over its own PCIe link into its own (lent) prompt buffers, computes their rows, and sends
    /// them back - the --peer-device peer's streaming, without P2P (activations and rows through mapped host memory,
    /// read by copy kernels).  Only on the MMQ prompt path, only for chunks of stream_all_min() tokens and more, up to
    /// the size where the measured share stops paying.  Opt-in: STRATA_PREFILL_HELP=1 (not bit-identical to the default).  Both stages must have
    /// run `init`.
    bool set_stage_helper(Prefill* helper, std::string& err);

    /// The CPU expert pool (decode's, idle while a prompt is read). With STRATA_PREFILL_CPU_SHARE set, a chunk below
    /// stream_all_min() tokens - an agent's tool output - hands it the non-resident experts routed by at most MAXT of
    /// its tokens, fewest first, up to a share of the experts it would stream (`auto`: measured, where both sides end
    /// together). The CPU reads them from RAM (the arena, the page-locked copy, or the mapped experts.bin's pages)
    /// while the rest come over PCIe; their rows go to Dm's tail, as a peer's do. Not bit-identical to the GPU's rows
    /// (the CPU's own activation format). Unset (default) or null: every expert on the GPU. Only for a pool no other
    /// thread runs meanwhile (no batch slots); on a layer split every stage may have it - one stage at a time takes it
    /// for a chunk. Set before `init`.
    void set_cpu_pool(kernels::cpu::ExpertPool* pool);
    /// With STRATA_PREFILL_CPU_SHARE set and `applies` (a pool will be set): the chunks staged after their routing - the
    /// only ones the share applies to - go up to STRATA_PREFILL_CPU_SHARE_MAX tokens (default 3072) instead of 1024.
    /// Before any `bytes_needed` (the ring and the small-chunk buffers follow the limit); `init` also sets it.
    /// `by_default`: this path is one the share was measured on (CUDA, one GPU, no batch slots), so with the variable unset
    /// the share is on for chunks below 1024 tokens (STRATA_PREFILL_CPU_SHARE=0 turns it off).
    static void arm_cpu_share(bool applies, bool by_default = false);

    /// Layer-major prefill (STRATA_PREFILL_LAYER_MAJOR=1): true while `on_chunk` reports a sub-chunk before the end of
    /// its super-chunk.  The session state is already at the super-chunk's end then, so no checkpoint may be taken.
    /// On a layer split it is the last stage's (the one that reports the rows), asked through the first.
    /// One prompt of a multi-prompt read (run_multi): its tokens, the position it starts at and, per stage of a layer split
    /// (index 0 = the first stage), the session it reads into (nullptr / missing: that stage's own main session).
    struct Seg {
        const int64_t* tokens = nullptr;
        int64_t n = 0;
        int64_t pos0 = 0;
        std::vector<core::SessionState*> stage_ss;
    };
    /// Several prompts in ONE layer-major pass (the streamed experts of a layer are read once for all of them).  Each
    /// prompt fills its own sessions; `on_chunk` / `on_stage_chunk` report the FIRST prompt only.  Returns false with `err`
    /// when the layer-major walk does not apply (then nothing was read; the caller reads the prompts one by one).
    bool run_multi(const std::vector<Seg>& segs, std::string& err);

    bool layer_major_mid() const {
        const Prefill* p = this;
        while (p->next_ != nullptr) p = p->next_;
        return p->lm_mid_;
    }

private:
    static uint64_t bytes_needed_impl(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk,
                                      bool owned_pages, bool src);
    kernels::cpu::ExpertPool* cpu_pool_ = nullptr;   ///< set_cpu_pool
    // Layer-major prefill: the super-chunk's buffers for an n-token prompt (its size, 0 = not this prompt: `why`),
    // the run, and the buffers given back after it.  `cap` (> 0): at most that many tokens (a layer split's stages
    // share one size); lm_prepare_chain sizes every stage of a split.
    int64_t lm_prepare(int64_t n, std::string& why, int64_t cap = 0);
    int64_t lm_prepare_chain(int64_t n, std::string& why);
    void lm_release_chain();
    bool run_layer_major(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err);
    struct LmSub { int seg = 0; int64_t c0 = 0, T = 0, t0 = 0; };   ///< a sub-chunk: prompt, first token, tokens, row in the super-chunk
    bool lm_run(const std::vector<Seg>& segs, int depth, const std::vector<LmSub>* given_super, std::string& err);
    void lm_release();
    bool lm_mid_ = false;
    bool lm_force_ = false;        ///< run_multi: the walk even for a short total
    bool lm_by_default_ = false;   ///< the walk is on because nothing was set (a wider VRAM margin, the measured super-chunk size)
    // Stage-1 pipeline: intermediate stages return after handing their chunk to
    // the direct successor. The public run() drains the chain once at prompt end.
    bool run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err);
    bool drain_pipeline(std::string& err);

    int64_t stage_lb_ = 0, stage_le_ = -1;
    std::vector<int> tap_layers_;
    int64_t tap_first_ = 0, tap_cap_ = 0;
    float* tap_dev_ = nullptr;
    Prefill* next_ = nullptr;
    Prefill* helper_ = nullptr;         ///< set_stage_helper
    bool single_chunk_ = false;         ///< a later stage: the prompt is one chunk (set by the stage before)
    int64_t stopped_at_ = -1;           ///< stop_at_chunk ended the last run after this many tokens (-1: it read all)
    int64_t kva_floor_ = 0;             ///< kva_floor()
    bool pipe_chunked_ = false;         ///< a later stage: the chunks came from a --prefill-pipe plan (set likewise)
    int64_t force_chunk_ = 0;           ///< force_chunk, for the next run (0: none)
    bool force_single_ = false, force_planned_ = false;
    bool bind_stage_helper(int64_t T);  // binds the helper's buffers for a one-chunk prompt of T tokens
    const float* hand_in_ = nullptr;    ///< the previous stage's rows of the chunk being read (host, pinned)
    /// STRATA_HANDOFF_PIPE: the previous stage's per-piece "landed in hand_in_" events (cudaEvent_t[hand_pieces_],
    /// on its device) when it copies the chunk in pieces without a host wait; null with the one-copy hand-off
    const void* hand_ev_ = nullptr;
    int hand_pieces_ = 0;

    std::string next_err_;
    std::future<bool> next_run_;
    int hand_buf_ = 0;

    bool carve(std::size_t T, void* alloc);   // the device buffers of a chunk (prefill.cpp's Alloc)
    void release();                          // the destructor's cleanup (also `reset`'s)
    struct Impl;
    std::unique_ptr<Impl> impl_;
    PrefillStats stats_;
};

}  // namespace strata::prefill
