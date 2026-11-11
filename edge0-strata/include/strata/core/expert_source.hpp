// include/strata/core/expert_source.hpp - P2.S3/P2.S5: the expert pool's adapter to the host loop.
//
// `session_loop` publishes, per layer, the NORMED activation `x_f`, the ten routed expert ids and their router
// weights, and expects `k x n_embd` floats back.  This is the piece that turns those four things into an
// `ExpertPool::run` call.  Nothing here is clever, and that is the point: the pool, the kernel and the loop are
// each already verified, so this file has exactly one job - get the CONTRACT between them right.
//
// THE CONTRACT, and each clause is a way to be wrong:
//
//   1. **`x_f` IS THE SAME PINNED BUFFER EVERY LAYER.**  Its ADDRESS never changes, so the `ActQ` cannot be
//      cached by pointer - a cache keyed on `x_f` would quantize layer 0 and reuse it for all 47 remaining
//      layers, which is a finite, plausible, completely wrong token.  There is no cache here at all: the
//      conversion is one 2560-element pass against a 0.3 ms/layer budget, and a correct answer is worth more
//      than the microseconds.
//   2. **THE POOL DOES NOT APPLY THE ROUTER WEIGHT.**  `moe_combine` (`src/core/layer.cpp:479`) sums
//      `w[i] * parts[i]` on the device.  `ExpertJob::weight` is a diagnostic field; setting it here would
//      apply the weight twice, which is invisible in a single layer and compounds over 48.
//   3. **THE KERNEL'S GEOMETRY IS FIXED BY THE ARTIFACT** (`H = 2560`, `FF = 640`, `BLOB = 1,382,400`).  A
//      different `n_embd` or a different expert width is REFUSED rather than mis-indexed, because the blob's
//      internal offsets are compile-time constants and reading a 640-wide expert as a 2560-wide one walks off
//      the end into the next expert's bytes without faulting.
#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/pinned.hpp"
#include "strata/core/exchange_storage.hpp"
#include "strata/core/hit_hook.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/core/latent_dense.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstddef>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <string>
#include <vector>

namespace strata::kernels::cpu {
struct ExpertLayout;
}

namespace strata::core {

class PeerExperts;   // multi-GPU: the second GPU's expert tier (peer_experts.hpp)
class RemoteExperts;
struct LoadStats;

namespace detail {

/// Sentinel used by the pure complement planner for a blob that remains in the mmap fallback.
inline constexpr uint64_t kNoCacheComplement = ~uint64_t{0};

/// Required cgroup-v2 usage counters for the conservative cache-reclaim allowance.
struct CgroupMemoryStat {
    uint64_t current = 0;
    uint64_t inactive_file = 0;
    uint64_t file_dirty = 0;
    uint64_t file_writeback = 0;
    bool valid = false;
    uint64_t file = 0;   ///< 149: memory.stat "file" (active + inactive page cache); 0 when not reported
};

/// Calculate additional bytes under a finite cgroup limit after reclaiming the group's clean page cache, as
/// MemAvailable counts it on a bare host: every clean file page (active and inactive: memory.stat "file", else
/// inactive_file), less a low-watermark share the kernel keeps (min(half the clean cache, limit / 64)).  A process
/// whose own reads (a GPU cache fill, a read-ahead) left gigabytes of active cache in its group was judged short of
/// RAM before (149: 1.2 GiB "available" of a 32 GiB group, the file tier then switched to unbuffered reads).
/// Returns false when the required memory.stat counters were unavailable.
bool cgroup_available_bytes(uint64_t limit, const CgroupMemoryStat& stat, uint64_t& bytes);

/// #633: the RAM this process can get.  `available`: MemAvailable (Windows: the available physical memory), lowered
/// to the room under the tightest cgroup limit; `cgroup_limit`: that tightest limit itself (v2 memory.max of the group
/// and its ancestors, or v1 memory.limit_in_bytes), ~0 when there is none - what a container can never exceed.
struct HostMemory {
    uint64_t available = 0;
    uint64_t cgroup_limit = ~uint64_t{0};
    uint64_t commit = ~uint64_t{0};   ///< available Windows commit capacity; ~0 when not reported
};

/// Linux reads `meminfo`, `self_cgroup` and the cgroup tree under `cgroup_root` (the parameters are for tests; the
/// defaults are the real files).  cgroup v2 as before (an unreadable limit of a group that has one fails); cgroup v1's
/// memory controller (`<root>/memory/<path>`: memory.limit_in_bytes - memory.usage_in_bytes); no cgroup line at all is
/// MemAvailable alone.  False when the RAM cannot be determined.
bool host_available_memory(HostMemory& m, const std::string& meminfo = "/proc/meminfo",
                           const std::string& self_cgroup = "/proc/self/cgroup",
                           const std::string& cgroup_root = "/sys/fs/cgroup");

/// #4051: wait for the RAM an arena needs to come back.  An engine that was still exiting (or a program that was ending)
/// holds memory a few seconds longer than its process; loading into that shortfall sent a 60 GiB host into reclaim for
/// good.  Calls `sample(available)` (false: the RAM cannot be read; the wait ends) every `step_s` seconds, through
/// `sleep(step_s)`, until `available >= need` or `max_wait_s` seconds have gone.  `waited` gets the seconds waited,
/// `available` the last reading.  True when the RAM is there.  `progress(waited, available)` is called about every
/// `note_s` seconds while waiting.  `max_wait_s <= 0` is no wait at all.
bool wait_for_ram(uint64_t need, double max_wait_s, double step_s, double note_s,
                  const std::function<bool(uint64_t&)>& sample, const std::function<void(double)>& sleep,
                  const std::function<void(double, uint64_t)>& progress, double& waited, uint64_t& available);

/// Bound a resident budget by RAM and commit capacity after headroom; leave 256 MiB more when clamping.
/// Pass UINT64_MAX for commit when the platform does not report it.
uint64_t clamp_resident_budget(uint64_t requested, uint64_t physical, uint64_t commit, uint64_t headroom);

/// #1250: one cudaHostAlloc of the whole page-locked complement has no way back when the kernel does not give the
/// pages as fast as the driver takes them (the process is OOM-killed, even with MemAvailable high: clean file cache
/// that cannot be reclaimed at that moment).  `free_now` is MemFree (pages that are really free), `bytes` the
/// complement, `reserve` what must stay.  True when the pages would have to come from reclaimed cache.
bool pin_depends_on_reclaim(uint64_t free_now, uint64_t bytes, uint64_t reserve);
/// The paced registration takes the next `step` bytes only while the RAM available after it (cgroup limits included)
/// is still above `reserve`.
bool pin_step_fits(uint64_t available, uint64_t step, uint64_t reserve);
/// Parses `key` ("MemFree") of a /proc/meminfo text; 0 when absent.
uint64_t meminfo_bytes(const std::string& meminfo_text, const std::string& key);

/// 0.1.42 (NUMA report): the MemFree to judge the page-locked complement by.  `global_free` is the whole host's MemFree;
/// `node_free` each NUMA node's own (/sys/devices/system/node/node*/meminfo, empty or one entry on a single-node host).
/// A dual-socket host whose second node is full of another model's file cache has plenty free in total but none where
/// the driver may allocate: with two or more nodes the figure is the smaller of the total and (node count x the
/// SMALLEST node's free), i.e. what an even spread would find.  Only lowers the figure, so the pages are then taken in
/// steps (which reclaim cache as they fault) instead of in one driver call; it never refuses and never caps.
uint64_t numa_pin_free(uint64_t global_free, const std::vector<uint64_t>& node_free);
/// Byte ranges [first, second) of slice `slice` of 2 inside one complement slot at arena offset `slot_off`, whose roles
/// (gate, up, down) are `role_bytes[0..2]` long and back to back. Role r's slice-0 bytes are [start_r, start_r + role_r/2),
/// its slice-1 bytes the rest. Page rule (absolute arena offsets): any page holding a slice-0 byte belongs to slice 0,
/// so slice-0 ranges round OUTWARDS (start down, end up) and slice-1 ranges round INWARDS (start up, end down) - this
/// also settles pages shared across role and slot boundaries. Returns three ranges in arena offsets, possibly empty
/// (first == second). Adjacent roles smaller than a page may give overlapping slice-0 ranges: harmless (same node, the
/// caller coalesces them).
std::array<std::pair<uint64_t, uint64_t>, 3> numa_slice_ranges(uint64_t slot_off, const uint64_t role_bytes[3],
                                                               int slice, uint64_t page = 4096);
/// The sentence appended to a "not enough RAM" start failure: how much of the RAM is file cache (Cached + Buffers of
/// `meminfo_text`, mapped pages included) and what to do about it; "" when under 1 GiB of cache (nothing to say).
std::string page_cache_hint(const std::string& meminfo_text);

/// Build compact offsets for experts absent from both the primary GPU cache and an optional second GPU tier.
/// Kept CPU-only so selection and byte accounting can be tested without initializing a GPU.
bool make_cache_complement_plan(
    int64_t n_layers, int64_t n_expert, const std::vector<uint64_t>& layer_blob_bytes,
    const std::vector<std::pair<int32_t, int32_t>>& primary_gpu_pairs,
    const std::vector<std::pair<int32_t, int32_t>>& additional_gpu_pairs,
    std::vector<uint64_t>& offsets, uint64_t& bytes, std::string& err);

/// Resolve one blob through the compact copy when present, otherwise preserve its exact mapped-file fallback.
const uint8_t* cache_complement_blob_or_fallback(
    size_t index, const std::vector<uint64_t>& offsets, const uint8_t* complement_host,
    const uint8_t* mapped_fallback);

/// The resident RAM mode: which GPU-cache slots' experts are kept in RAM too.  The prompt path lends the cache's
/// LAST slots (from `lend_from` on; a short prompt lends only the last few), and a lent slot's expert is streamed
/// from RAM during the prompt and copied back into its slot after it.  `base_bytes` (every expert no slot holds)
/// must fit `budget`; slots are then added from the end down to `lend_from` while they still fit.  Returns the
/// first slot kept in RAM (`slot_bytes.size()` = none), or -1 when `base_bytes` alone exceeds `budget`.
int64_t choose_resident_keep_from(const std::vector<uint64_t>& slot_bytes, uint64_t base_bytes, uint64_t budget,
                                  int64_t lend_from);

/// The adaptive tier swapped `in` into a GPU slot and `out` out of it: `out` takes `in`'s place in the compact copy
/// (the caller copies out's bytes there).  False, and nothing changed, unless `in` is in the copy and `out` is not.
bool exchange_cache_complement(std::vector<uint64_t>& offsets, size_t in, size_t out);

/// A layer split: add each stage's lend region to a plan built from the complement plus every stage's
/// cache (the `additional_gpu_pairs` shape).  Each region lists one stage's (layer, expert) pairs, the
/// pairs in its cache's highest slots first - the slots the prompt path borrows, from the tail in.  A pair
/// already in the plan is passed over; a stage's walk stops at the first pair that does not fit `cap_bytes`
/// and the NEXT stage's walk still runs (the pairs past the cap keep the mapped-file fallback they had).
/// Returns the pairs added, or -1 - leaving the plan untouched - when a pair is outside the geometry or a
/// region repeats one.
int64_t append_stage_lend_regions(int64_t n_layers, int64_t n_expert, const std::vector<uint64_t>& layer_blob_bytes,
                                  std::vector<uint64_t>& offsets, uint64_t& bytes, uint64_t cap_bytes,
                                  const std::vector<std::vector<std::pair<int32_t, int32_t>>>& regions,
                                  std::string& err);

}  // namespace detail



/// Where one routed expert's bytes come from.
///
/// Phase 2 has NO cache (`phase-2-correct-engine.md`: hit rate `h = 0`), so the only implementation is a
/// file-backed reader.  The interface exists anyway because Phase 3 replaces exactly this object with the VRAM
/// cache, and because a test can supply an in-memory source without a 34 GB artifact.
class ExpertSource {
public:
    virtual ~ExpertSource() = default;

    /// The expert-layout blob for `(layer, expert)`, or nullptr if it cannot be produced.
    ///
    /// The pointer only has to stay valid until the next `blob()` call: with `h = 0` every expert is computed
    /// immediately and nothing is retained.  A CACHING source must return pointers into the cache, not into a
    /// reused staging buffer - otherwise the pool would read the next expert's bytes while computing this one.
    virtual const uint8_t* blob(int64_t layer, int64_t expert) = 0;
    /// A blob pointer that stays valid (not recycled) for as long as the source is open: for a copy that runs
    /// asynchronously long after the call (the adaptive tier's swap-ins).  Defaults to `blob`.
    virtual const uint8_t* blob_stable(int64_t layer, int64_t expert) { return blob(layer, expert); }
    /// The pieces (up to 3) a blob consists of when the source can hand them out where they already are - the GGUF
    /// mapped in place keeps an expert's gate, up and down in three tensors - so the caller can copy each straight to
    /// its destination instead of `blob_stable` assembling a staging copy first.  0: not available for this
    /// (layer, expert) - take `blob_stable`.  The pointers stay valid as long as the source is open.
    virtual int blob_parts(int64_t layer, int64_t expert, BlobPart* out) { (void) layer; (void) expert; (void) out; return 0; }
    /// The caller has queued an asynchronous copy out of `blob` (a pointer this source handed out) on `stream` (a
    /// cudaStream_t): the source must not overwrite or recycle that memory until the copy has finished.  A source whose
    /// pointers are never recycled ignores it.  (#1237: a page-locked stage buffer is read by the DMA after the copy call
    /// returned, where a pageable one is staged by the driver before it returns.)
    virtual void note_async_read(const uint8_t* blob, void* stream) { (void) blob; (void) stream; }

    /// Blobs touched, for the driver to report.  A source that does not count returns 0.
    virtual int64_t reads() const { return 0; }

    /// Plan v0.3 P5: whether `blob(layer, expert)` lies in page-locked, CUDA-registered memory, so an
    /// asynchronous host-to-device copy can DMA it directly (no staging copy on the CPU).
    virtual bool pinned(int64_t layer, int64_t expert) const { (void) layer; (void) expert; return false; }

    /// Called once before the first expert of a layer.  A source that reads from disk wants to start the read
    /// here so it overlaps the quantisation, and a prefetching source in Phase 3 wants the ids.
    virtual void begin_layer(int64_t layer, const int32_t* ids, int64_t k) { (void) layer; (void) ids; (void) k; }
    /// Plan v0.3 P6: the DEVICE address of a pinned, mapped blob (the GPU can read it over PCIe), or null.
    virtual const uint8_t* device_alias(int64_t layer, int64_t expert) const { (void) layer; (void) expert; return nullptr; }
    /// A file-backed source: start reading this expert's pages now (it will be needed by the CPU); no-op elsewhere.
    virtual void prefetch(int64_t layer, int64_t expert) { (void) layer; (void) expert; }
    /// A file-backed source: this expert lives in VRAM, so its pages need not stay in RAM - hand them back to the
    /// kernel (a later read re-reads the file; no result depends on it).  Returns the bytes released; 0 elsewhere.
    virtual uint64_t release(int64_t layer, int64_t expert) { (void) layer; (void) expert; return 0; }
    /// Whether the verify window may give the GPU a PCIe share of this layer's misses at all (each expert is still
    /// checked with `pinned`).  The arena answers per layer through its expert 0; the resident RAM mode's compact
    /// copy has no expert 0 when the GPU cache holds it, so it answers for the whole copy.
    virtual bool pcie_layer(int64_t layer) const { return device_alias(layer, 0) != nullptr; }

    /// CS-T: whether `blob(layer, expert)` would be assembled into a short-lived buffer (a native pack read from its
    /// GGUF shards in place, where an expert's gate, up and down rows are three separate slices).  Such a pointer
    /// stays valid for the layer it was asked in and the next one or two; a consumer that keeps a blob longer (the
    /// prompt path's stager queues a whole chunk) copies it with `copy_blob` instead.
    virtual bool transient(int64_t layer, int64_t expert) const { (void) layer; (void) expert; return false; }
    /// #1353: of the bytes `transient` experts would be copied from, the share whose pages are in RAM right now (a
    /// GGUF read in place whose page cache is warm), measured on a sample of up to `samples` experts; -1 when this
    /// source cannot tell (not a file source, or no residency query on this OS).  The prompt path's stager uses it to
    /// pick the SSD profile (many threads, a deep ring) only for blobs that really are page faults on a disk.
    virtual double cached_share(int64_t samples) const { (void) samples; return -1.0; }
    /// Disk sessions: the files this source read its experts from, as (role, path), resolved by the loader itself
    /// - the pack's experts.bin, or every GGUF tensor native_experts.txt named, per layer and role
    /// ("expert blk.L.ffn_up").  Filled by open(); empty before.
    const std::vector<std::pair<std::string, std::string>>& model_inputs() const { return inputs_; }
    /// The blob's bytes into `dst` (blob_bytes(layer) of them).  Safe from several threads for a source whose
    /// `transient` can be true.
    virtual bool copy_blob(int64_t layer, int64_t expert, uint8_t* dst);
    /// Asks the OS for these pairs' bytes ahead of the `blob` calls that read them.  False (nothing asked) by default.
    virtual bool advise_pairs(const std::pair<int32_t, int32_t>* pairs, int64_t n) const {
        (void) pairs; (void) n; return false;
    }
    /// pp-opt: `n` blobs at once (the prompt path's stager claims runs of consecutive experts): `dst[i]` gets the
    /// blob of (layers[i], experts[i]).  A source that reads a drive may merge neighbouring blobs into one request.
    /// Default: copy_blob one by one.
    virtual bool copy_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n);
    /// The `n` experts of `layer` the CPU is about to ask `blob` for, all at once: a source that reads a file may
    /// fetch them in parallel.  The bytes `blob` then returns are the same.  Default: nothing.
    virtual void prefetch(int64_t layer, const int64_t* experts, int64_t n) { (void) layer; (void) experts; (void) n; }
    /// 149-cold (STRATA_COLD_OVERLAP): `prefetch`, but the blobs that must be read are read in the background.
    /// `pending[i]` is set to 1 for each expert whose blob is not ready yet (0: `blob` returns it at once); returns how
    /// many.  `wait_landed` then waits until at least one pending expert's blob is ready (its state goes 1 -> 2) and
    /// returns how many landed; `prefetch_end` joins the reads (every claimed blob is published by then).  `blob` on
    /// a pending expert still works: it waits for that blob.  Default: the synchronous `prefetch`, nothing pending.
    virtual int64_t prefetch_begin(int64_t layer, const int64_t* experts, int64_t n, uint8_t* pending) {
        prefetch(layer, experts, n);
        for (int64_t i = 0; i < n; ++i) pending[i] = 0;
        return 0;
    }
    virtual int64_t wait_landed(int64_t layer, const int64_t* experts, int64_t n, uint8_t* state) {
        (void) layer; (void) experts;
        int64_t got = 0;
        for (int64_t i = 0; i < n; ++i)
            if (state[i] == 1) { state[i] = 2; ++got; }
        return got;
    }
    virtual void prefetch_end() {}
    /// 149 STRATA_IO_PF_GATE: whether the decode's read-ahead runs now (the look-ahead then predicts one layer, not two)
    virtual bool read_ahead_open() const { return true; }
    /// 0.2.0: the source the Linux file-tier defaults apply to (the GGUF read in place); STRATA_COLD_OVERLAP's default
    virtual bool linux_file_defaults() const { return false; }
    /// 149-cold STRATA_COLD_OVERLAP=3: the last prefetch_begin left this expert unstaged because the page cache holds
    /// all of it (the GGUF read in place): its gate, up and down rows in the mapping.  False: ask `blob`.
    virtual bool zc_roles(int64_t layer, int64_t expert, const uint8_t** p) { (void) layer; (void) expert; (void) p; return false; }
    /// CS-T: the experts of `layer` a predictor expects next - a source that reads a file may start reading their
    /// pages now, in the background.  Only warms: what `blob` returns is unchanged.  Default: nothing.
    virtual void warm(int64_t layer, const int64_t* experts, int64_t n) { (void) layer; (void) experts; (void) n; }
    /// 149-prefetch (STRATA_IO_PF_SCHED=1): `warm` with each expert's confidence (0..1) and how many layers ahead of the
    /// decode its layer is (the deadline: past it the read is dropped).  Default: plain `warm`.
    virtual void warm_scored(int64_t layer, const int64_t* experts, const float* conf, int64_t n, int ahead) {
        (void) conf; (void) ahead; warm(layer, experts, n);
    }
    /// Whether the predictor should compute the confidences for warm_scored (else it calls plain warm).
    virtual bool wants_scores() const { return false; }
    /// Whether `warm` does anything (the predictor is not run otherwise).
    virtual bool warms() const { return false; }
protected:
    /// What model_inputs() returns; set by a source's open().
    void record_inputs(const std::string& pack_experts, const std::string& gguf,
                       const strata::kernels::cpu::ExpertLayout& lay, bool from_gguf);
    std::vector<std::pair<std::string, std::string>> inputs_;
};

/// CS-T, routing-aware prefetch of the file tier: when the CPU pool starts layer `l`, a worker thread applies layer
/// l+1's router (BF16, host copy) to layer l's MoE input - the residual stream changes little from one layer to the
/// next - takes each token's top `k` experts, drops the ones the GPU cache or the RAM copy holds, and asks the
/// source to `warm` the rest, so their pages are on the way while layer l computes.  A prediction only warms pages:
/// it never changes which experts are computed or how.
struct ForesightSwap;
class RouterLookahead {
public:
    RouterLookahead() = default;
    ~RouterLookahead();
    RouterLookahead(const RouterLookahead&) = delete;
    RouterLookahead& operator=(const RouterLookahead&) = delete;
    /// `routers[l]`: layer l's ffn_gate_inp as BF16 bits, n_expert rows of n_embd.
    bool start(std::vector<std::vector<uint16_t>> routers, int64_t n_embd, int64_t n_expert, int k, ExpertSource* src,
               std::string& err, bool allow_cold_source = false);
    /// One prefetch design (#1348): the predicted experts that no cache holds also go to the Foresight swap space
    /// (ForesightSwap::predict), which copies them to VRAM ahead of the layer.  nullptr detaches it (waits for a
    /// prediction in progress), so the swap space can be destroyed first.
    void set_foresight(ForesightSwap* fs);
    /// Layer `layer`'s MoE input for `n_tok` tokens (host floats): predict and warm layer + 1.  Never waits: a
    /// prediction still running for an earlier layer makes this one skip.
    void submit(int64_t layer, const float* x, int64_t n_tok, const int32_t* host_res);
    /// STRATA_IO_PREFETCH_DEPTH: how many layers ahead of `layer` are predicted and warmed (default 1, the next layer).
    /// A deeper one predicts layer + j from layer's input, which is less exact the further it looks.
    void set_depth(int d) { depth_ = d < 1 ? 1 : d > 4 ? 4 : d; }
    int depth() const { return depth_; }
    /// STRATA_LOOKAHEAD_STATS=1 (measurement only): remember each prediction (top k, k+4 and k+10 per token) and let the
    /// dispatch score it against the experts the next layer really routes outside the GPU cache.
    void enable_stats(int64_t n_layers);
    bool stats_on() const { return stats_on_; }
    void stats_window_start() { gen_.fetch_add(1, std::memory_order_relaxed); }
    void stats_score(int64_t layer, const int32_t* ids, int64_t n, const int32_t* host_res, int pcie_num = 0);
    int64_t predicted() const { return predicted_.load(std::memory_order_relaxed); }
    int64_t skipped() const { return skipped_.load(std::memory_order_relaxed); }
    double busy_ms() const { return (double) busy_us_.load(std::memory_order_relaxed) / 1000.0; }

private:
    void run();
    std::vector<std::vector<uint16_t>> routers_;
    int64_t n_embd_ = 0, n_expert_ = 0;
    int k_ = 10;
    int depth_ = 1;
    ExpertSource* src_ = nullptr;
    std::thread thread_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool quit_ = false, pending_ = false, busy_ = false;
    int64_t layer_ = -1, n_tok_ = 0;
    const int32_t* host_res_ = nullptr;
    uint64_t sub_gen_ = 0;
    ForesightSwap* fs_ = nullptr;                   ///< (under mu_)
    std::vector<float> x_;
    std::atomic<int64_t> predicted_{0}, skipped_{0};
    std::atomic<uint64_t> busy_us_{0};
    // STRATA_LOOKAHEAD_STATS
    bool stats_on_ = false;
    int64_t words_ = 0;
    std::unique_ptr<std::atomic<uint64_t>[]> pmask_;   // [layer][3 sizes][words]
    std::unique_ptr<std::atomic<uint64_t>[]> pstamp_;  // [layer]: gen + 1 of the prediction (0 = none)
    std::unique_ptr<std::atomic<int32_t>[]> pvote_;    // [layer][kVoteN]: experts by votes, expert + 1 (0 = none)
    std::atomic<uint64_t> gen_{0};
    uint64_t st_layers_ = 0, st_nopred_ = 0, st_actual_ = 0, st_hit_[3] = {0, 0, 0}, st_pred_[3] = {0, 0, 0},
             st_predhit_[3] = {0, 0, 0};
    uint64_t st_windows_ = 0;
    uint64_t st_vhit_[3] = {0, 0, 0}, st_vpred_[3] = {0, 0, 0}, st_pch_ = 0, st_pcov_[3] = {0, 0, 0};
    static constexpr int kVoteN = 16;
};

/// Plan v0.3 P6: what the GPU computes in a verify window's layer, written by the pool (mapped host memory) right
/// after the ring and published before the CPU starts its own share.  Groups of entries that share a blob; the
/// blob is a VRAM slot or a pinned host blob read over PCIe.
struct GpuPlanSink {
    int32_t* counts = nullptr;             ///< [0] groups, [1] entries
    int32_t* start = nullptr;              ///< cap + 1
    int32_t* dst = nullptr;                ///< cap: the entry's row of parts (token * k + j)
    int32_t* tok = nullptr;                ///< cap: the entry's token
    unsigned long long* ptr = nullptr;     ///< cap: the group's blob, device address
    /// Plan v0.3 P6 (DMA): the PCIe share is a second list of groups - `ptr2[q]` is staging slot q, `start2` indexes
    /// the same dst/tok entries - computed by the GPU after `fetch` has copied their blobs into staging with the
    /// copy engine.  counts[0] = VRAM groups, counts[1] = all entries, counts[2] = PCIe groups.
    unsigned long long* ptr2 = nullptr;
    int32_t* start2 = nullptr;
    unsigned long long staging = 0;
    int64_t staging_cap = 0;
    int64_t cap = 0;
    void (*publish)(void* ctx) = nullptr;
    /// Starts the DMA copies of `n` host blobs (pinned) into staging slots 0..n-1 and signals the GPU when they land.
    void (*fetch)(void* ctx, const uint8_t* const* src, int n, size_t bytes) = nullptr;
    void* ctx = nullptr;
    /// Plan v0.3 P6: how a PCIe group reaches the GPU.  0 = the copy engine stages it (DMA, `fetch`); 1 = the grouped
    /// kernel reads the mapped arena directly; 2 = a copy kernel stages it inside the graph.  For 1 and 2 `ptr2`
    /// holds the arena's device alias.
    int pcie_mode = 0;
    int32_t cpu_jobs = 0;   ///< host only: the distinct experts the pool computes for the layer just published
};

/// Verifier::set_pcie_balance: a verify window's costs, fitted to each layer's measured times (ms, least squares with
/// forgetting): the CPU pool's time with n_c CPU and n_p PCIe experts, a + c n_c + d n_p (none: p0) - d is the PCIe
/// copies' DRAM reads slowing the pool, as the pool and the link read the same RAM - and the GPU's, g0 + g n_p.  A
/// layer then reads over PCIe the m of its nmiss misses that minimizes max(GPU, CPU). Measured: an RTX 5090 + 9950X3D
/// pool is DRAM-bound (moving experts to PCIe left the pool's time where it was, d ~ c: few move), one worker on
/// ggml-cpu's dot is not (d ~ 0: most move). Until both fits have data, and while `on` is off, pcie_num's share.
struct PcieModel {
    bool on = false;
    double a = 0, c = 0, d = 0, p0 = 0, g0 = 0, g = 0;   ///< the fits (read by `pick`)
    int np0 = 0;
    double A[3][3] = {}, B[3] = {};   ///< the CPU fit's normal equations, [1, n_c, n_p]
    double C[2][2] = {}, D[2] = {};   ///< the GPU fit's, [1, n_p]
    int ncpu = 0, ngpu = 0, ngpu_p = 0;   ///< samples in each (ngpu_p: with a PCIe expert)
    uint32_t explore = 0;
    static void mean(double& v, int& n, double x) {   // the first 16 evenly, then 1/20 each
        if (n < 16) v += (x - v) / ++n;
        else v += 0.05 * (x - v);
    }
    /// One layer's times: the pool's `p` with `n_c` CPU and `n_p` PCIe experts, the GPU's `gpu` (ms).
    void sample(int n_c, int n_p, double p, double gpu) {
        constexpr double lam = 0.995;   // ~200 layers' memory: a few windows
        if (n_c == 0) mean(p0, np0, p);
        else {
            const double x[3] = {1.0, (double) n_c, (double) n_p};
            for (int i = 0; i < 3; ++i) {
                B[i] = lam * B[i] + x[i] * p;
                for (int j = 0; j < 3; ++j) A[i][j] = lam * A[i][j] + x[i] * x[j];
            }
            ++ncpu;
        }
        const double y[2] = {1.0, (double) n_p};
        for (int i = 0; i < 2; ++i) {
            D[i] = lam * D[i] + y[i] * gpu;
            for (int j = 0; j < 2; ++j) C[i][j] = lam * C[i][j] + y[i] * y[j];
        }
        ++ngpu;
        ngpu_p += n_p > 0;
    }
    /// Solve both fits (after a window). A ridge keeps a fit whose counts barely vary near its mean ratio.
    void solve() {
        if (ncpu >= 8) {
            double M[3][4];
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) M[i][j] = A[i][j] + (i == j && i > 0 ? 1e-3 * A[0][0] : 0.0);
                M[i][3] = B[i];
            }
            bool ok = true;
            for (int k = 0; k < 3 && ok; ++k) {   // Gauss-Jordan, partial pivot
                int piv = k;
                for (int i = k + 1; i < 3; ++i) if (std::abs(M[i][k]) > std::abs(M[piv][k])) piv = i;
                if (std::abs(M[piv][k]) < 1e-12) { ok = false; break; }
                for (int j = 0; j < 4; ++j) std::swap(M[k][j], M[piv][j]);
                for (int i = 0; i < 3; ++i) {
                    if (i == k) continue;
                    const double f = M[i][k] / M[k][k];
                    for (int j = k; j < 4; ++j) M[i][j] -= f * M[k][j];
                }
            }
            if (ok) {
                a = std::max(M[0][3] / M[0][0], 0.0);
                c = std::max(M[1][3] / M[1][1], 0.0);
                d = std::clamp(M[2][3] / M[2][2], 0.0, c);   // a PCIe expert slows the pool at most as one of its own
            }
        }
        if (ngpu >= 8 && ngpu_p >= 4) {
            const double det = C[0][0] * C[1][1] - C[0][1] * C[1][0];
            if (std::abs(det) > 1e-12) {
                g0 = std::max((D[0] * C[1][1] - C[0][1] * D[1]) / det, 0.0);
                g = std::max((C[0][0] * D[1] - C[1][0] * D[0]) / det, 0.0);
            }
        }
    }
    bool ready() const { return on && ncpu >= 8 && ngpu_p >= 4; }
    /// The PCIe count for a layer of `nmiss` misses (`fixed`: pcie_num's).  Every 64th layer with a miss that would
    /// keep them all on one side moves one to the other, so both fits keep seeing both.
    int pick(int nmiss, int fixed) {
        int m = fixed;
        if (ready()) {
            double best = 1e300;
            for (int k = 0; k <= nmiss; ++k) {
                const double cpu = nmiss - k > 0 ? a + c * (nmiss - k) + d * k : p0;
                const double t = std::max(g0 + g * k, cpu);
                if (t < best - 1e-6) { best = t; m = k; }
            }
        }
        if (on && nmiss >= 1 && (++explore & 63u) == 0) m = m == 0 ? 1 : m == nmiss ? nmiss - 1 : m;
        return m;
    }
};

/// The adapter's own state.  One per session, reused every layer so the token path allocates nothing (P2.T10).
struct ForesightSwap;

struct ExpertDispatch {
    ForesightSwap* fs = nullptr;   ///< Foresight swap space (STRATA_FS_SLOTS; null = off)
    strata::kernels::cpu::ExpertPool* pool = nullptr;
    ExpertSource* src = nullptr;
    RouterLookahead* lookahead = nullptr;   ///< CS-T: warms the next layer's predicted file-tier experts
    RemoteExperts* remote[3] = {}; ///< optional CUDA1..3 tiers for otherwise CPU-served rows
    int remote_count = 0;
    int64_t n_expert = strata::kernels::cpu::NE;

    /// Counters, for the driver to report rather than for control flow.
    int64_t layers = 0;
    int64_t experts = 0;
    int64_t missing = 0;

    // ================================ R4: THE VRAM TIER, MEASURED BEFORE IT IS USED ================================
    //
    // **THE CACHE IS CONSULTED AND FILLED HERE, AND NOTHING IS COMPUTED FROM IT YET.**  That is deliberate and
    // it is `R4-design-note.md` §7 step 2: the *dispatch* is measured on its own before any kernel is written,
    // because a hit rate measured after the kernel exists cannot say whether a disappointing result was the
    // policy, the split or the kernel.
    //
    // So every expert is still computed by the CPU, exactly as before, and the run is numerically identical
    // with the cache on or off.  What changes is that the engine now reports **h on its own routing, on a real
    // workload** - which is the number `R4.1` asks for and which until now existed only from offline traces.
    //
    // The policy is compulsory-miss: the first time `(layer, expert)` is routed, if a slot is free it is
    // admitted and filled from the arena.  No eviction, because eviction policy is the measured question
    // (`R4.1`'s LFU-decay vs LRU sweep) and a placeholder would set the hit rate everything is sized against.
    ExpertCache* cache = nullptr;
    int64_t cache_hits = 0;      ///< lookups already resident
    int64_t cache_admitted = 0;  ///< lookups that took a slot
    int64_t cache_refused = 0;   ///< lookups with no slot free (the cache is full)
    void* cache_stream = nullptr;
    const char* cache_fail = nullptr;

    // ================================ R4.2c: THE HITS GO TO THE GPU ================================
    //
    // **THE SPLIT IS BY ROUTER INDEX, AND THAT IS THE WHOLE TRICK.**  A layer routes ten experts; the resident
    // ones are computed on the GPU and the rest on the CPU, and both answers have to end up in `parts` at the
    // index the ROUTER gave them, because that is the order `moe_combine` weights against.  So the CPU's `out`
    // rows are zeroed for the hits, and each hit carries its routed index to the kernel as `dst`.
    //
    // Everything below is per-session state rather than per-call, so the token path allocates nothing (P2.T10).
    const uint8_t* cache_base = nullptr;   ///< the slot arena on the DEVICE
    int64_t cache_blob = 0;                ///< bytes per slot
    const uint64_t* cache_slot_off = nullptr;   ///< plan v0.3 P6: per-slot offsets when the slots differ in size
    void* hit_scratch = nullptr;           ///< `moe_hit_grouped_scratch_bytes(K, ...)`
    float* parts_out = nullptr;            ///< the graph's `parts` buffer, on the device
    /// Where the GPU's hits land, `K x n_embd`, DEVICE and separate from `parts_out` on purpose: see
    /// `HitPhase`.  Zeroed by the hit path each layer before the kernel writes it.
    float* hit_out = nullptr;
    int64_t parts_elems = 0;               ///< `K * n_embd`, the length of both buffers
    const float* mixed = nullptr;          ///< the layer's normed activation, for the hit kernel's Q8_0
    uint8_t* x_q8_0_hit = nullptr;         ///< `(n_embd/32) * 34` bytes, its own buffer
    /// **R4.2h: THE CPU's fp32 ACTIVATION SCALES, `(n_embd/32)` FLOATS.**  Without them the GPU's hits are
    /// computed with the `block_q8_0`'s fp16 `d` while the CPU's misses use the fp32 `ActQ::scale`
    /// (`cpu/expert.cpp:92`) - **4.761e-04 relative on 80 of 80 chunks**, measured with both real
    /// implementations linked in `bench/micro/act_quant_parity.cu`.  That disagreement is why turning the
    /// cache on changed the generated tokens.  Required whenever the hit path runs.
    float* x_q8_0_hit_scale = nullptr;
    int32_t* d_slot = nullptr;             ///< device, K entries
    int32_t* d_dst = nullptr;              ///< device, K entries
    std::vector<int32_t> h_slot, h_dst;    ///< host staging, sized at session setup
    /// **PER ROUTER INDEX, DECIDED IN `Launch` AND CONSUMED BY THE POOL.**  The two callbacks share it
    /// so the decision is made exactly once, on this layer's ids, and neither side can re-decide it.
    std::vector<uint8_t> is_hit;
    bool decided = false;
    /// Plan v0.3 P4 token graph: the STATIC residency table (`n_layers x n_expert`, slot or -1), the host's copy
    /// of what the device hit path reads.  When set, the pool leaves a resident expert's row at zero (the GPU
    /// computes it) without any `Launch` callback.
    const int32_t* host_res = nullptr;
    /// Plan v0.3 P4: split every expert by rows across the pool's threads (default on; A/B `--no-split-rows`).
    bool split_rows = true;

    // ================================ R4.2d: DID THE GPU ACTUALLY START? ================================
    //
    // **THE HIT PATH IS 0.91 ms SLOWER AND THE ONLY EXPLANATION LEFT IS THAT IT NEVER OVERLAPS.**  The kernel
    // is 159 GB/s against the CPU's 35, the CPU's half of the drain falls 6 ms, and none of it reaches the
    // token - which is what it would look like if the enqueued hit kernels did not BEGIN until the host next
    // entered the driver, i.e. after `pool()` returned.  That is the third sighting of this driver behaving
    // that way (rounds 195/287 on the doorbell, round 36 on the head and sampler) and it decides whether R4 can
    // pay at all, so it gets measured rather than argued.
    //
    // `hit_done` is recorded on the stream straight after the hit kernel.  `Combine` - which runs after the
    // pool - queries it: SUCCESS means the GPU finished while the CPU was working, NOT-READY means it had not.
    // Same shape as the doorbell's `rings_mid_graph`, and for the same reason.
    void* hit_done = nullptr;      ///< cudaEvent_t, created at session setup
    int64_t hit_ready = 0;         ///< layers where the hit work was DONE by the time the pool returned
    int64_t hit_late = 0;          ///< layers where it was not
    /// The candidate fix, as an A/B arm: enter the driver once right after the hit launch.  If submission is
    /// lazy, this starts the GPU work before the pool instead of after it.
    bool hit_poke = false;
    bool hit_cpu_order = false;    ///< experimental CPU-order GPU expert arithmetic; opt-in only
    int64_t n_hits = 0;                    ///< this layer's hits
    /// Set by `Launch` and consumed by `Combine`, so a `Combine` with no `Launch` in front of it cannot
    /// add a stale buffer into `parts`.
    bool hit_pending = false;
    const char* hit_fail = nullptr;

    /// Whether the hit path is wired up.  All of it or none of it: a half-configured hit path would compute
    /// some experts twice and others not at all, which is a wrong token rather than an error.
    bool hits_ready() const {
        return cache != nullptr && cache_base != nullptr && parts_out != nullptr && hit_out != nullptr &&
               mixed != nullptr &&
               hit_scratch != nullptr && x_q8_0_hit != nullptr && x_q8_0_hit_scale != nullptr && d_slot != nullptr && d_dst != nullptr;
    }

    std::vector<strata::kernels::cpu::ExpertJob> jobs;
    strata::kernels::cpu::ActQ act;
    /// Plan v0.3 P6 verify window: one quantized activation per token, the multi-token jobs, and the expert ->
    /// job map (reset after every layer).
    std::vector<strata::kernels::cpu::ActQ> act_multi;
    /// Plan v0.3 P6: a native pack's per-token activations (MAXT x kNativeActBytes).
    std::vector<uint8_t> nact_multi;
    std::vector<strata::kernels::cpu::ExpertJobMulti> jobs_multi;
    std::vector<int16_t> job_of;
    /// Plan v0.3 P6: the verify window's GPU plan (VRAM hits + the PCIe share of the misses); `pcie_num`/256 of
    /// each layer's distinct missed experts (the last ones in routing order) are read by the GPU over PCIe.
    GpuPlanSink* plan = nullptr;
    int pcie_num = 0;
    PcieModel pcie_model;          ///< Verifier::set_pcie_balance: the PCIe count from measured costs
    int64_t pcie_experts = 0;      ///< distinct experts the GPU read over PCIe in verify windows
    /// #588: routed (token, expert) entries the GPU computed from outside its cache in verify windows: read over PCIe
    /// (--pcie-frac, kind 1) or on another GPU (kind 2).  In neither cache_hits nor cache_refused.
    int64_t offload_entries = 0;
    /// STRATA_ROUTE_TAIL_SKIP=R: entries skipped (a missed expert every token of the window routes only at rank >= R)
    /// and the distinct experts behind them (neither copied nor computed)
    int64_t tail_skipped = 0;
    int64_t tail_experts = 0;
    double ms_plan = 0, ms_actq = 0, ms_jobs = 0, ms_run = 0;   ///< verify-window dispatch sections
    /// 149-cold STRATA_COLD_OVERLAP: 0 off (the CPU starts once every blob of the layer is read), 1 the blobs that
    /// must be read are read in the background while the pool computes the ready experts, each landed one computed
    /// as it lands; 2 measure only (read everything first, then the ready experts, then the read ones, timed apart).
    /// The per-layer counters: layers with a blob to read, those experts, the ready ones beside them; R = the reads'
    /// wall time, C_r / C_c = the pool on the ready / the read experts, wait = what the decode thread waited for
    /// reads, overlap_max = the sum over layers of min(R, C_r) (mode 2: the stall an overlap could hide).
    /// 3 = 1 + zero-copy: an expert the page cache holds whole (the GGUF in place, Linux mincore) is computed from the
    /// mapping instead of a stage buffer; 4 = zero-copy with the reads as in mode 0.  Same bytes, same bits.
    int cold_mode = 0;
    std::vector<strata::kernels::cpu::NativeFmt> zc_fmt;   ///< mode 3: the per-job layouts of the zero-copy experts
    int64_t cold_zc = 0;                                    ///< mode 3: experts computed from the mapping directly
    int64_t cold_layers = 0, cold_pending = 0, cold_ready = 0, cold_batches = 0;
    double ms_cold_r = 0, ms_cold_cr = 0, ms_cold_cc = 0, ms_cold_wait = 0, ms_cold_hide = 0;
    /// Plan v0.3 P6: decayed routing counts per (layer, expert) during decode (sized by the caller; empty = off),
    /// which the driver uses to swap the most-routed missing experts into the VRAM tier between rounds.
    std::vector<float> usage;
    /// The adaptive tier's evictions reach the device's table before the next window, not when the round's copies
    /// land: until then doorbell_publish_res can skip a layer's activation while the pool computes an evicted expert
    /// from the one published before.  The driver sets it where the tier does not wait for its copies
    /// (STRATA_ADAPT_NOWAIT=1, STRATA_ADAPT_LAG > 1; STRATA_ADAPT_EVICT_SYNC=0: off, for an A/B).  `res_dirty` asks the
    /// driver to patch the device's copy before the next window, `res_dirty_idx` lists the entries changed since.
    bool evict_sync = false;
    bool res_dirty = false;
    std::vector<int32_t> res_dirty_idx;
    /// STRATA_ADAPT_CHECKRES: the device's residency table as read back before the window; a layer with CPU experts
    /// whose routed ids all look resident there publishes no activation (doorbell_publish_res) - counted
    const int32_t* dres_check = nullptr;
    int64_t stale_x = 0;
    int64_t multi_misses = 0;      ///< distinct (layer, expert) pairs the CPU computed in verify windows
    int64_t multi_entries = 0;     ///< routed (token, expert) entries the CPU served in verify windows
    /// Multi-GPU: the second GPU's tier.  Its experts are computed there instead of on the CPU (kind 2).
    PeerExperts* peer = nullptr;
    int64_t peer_entries = 0;      ///< routed entries the peer served in verify windows
    /// Latent MoE (edge0): the CPU pool computes the A-blob experts on the host.  The A blobs are native
    /// NVFP4 with n_embd = ELL and n_ff = I; the shared D/U Q8_0 tables and the per-expert group labels live
    /// in host RAM.  Set by the driver when the pack is latent (fmt[0].n_embd != n_embd).
    bool latent = false;
    int64_t lat_H = 0, lat_ELL = 0, lat_I = 0;   ///< 2560 / 1792 / 640
    int lat_G = 0, lat_E = 0;                    ///< groups (8) / experts per layer (512)
    const LatentDense* lat_dense = nullptr;      ///< host D/U (mmap), read-only
    const int32_t* lat_labels = nullptr;         ///< host group labels, (n_layers * lat_E), read-only
    const uint8_t* lat_xlq = nullptr;            ///< this window's D-stage f32 xl rows (mapped, row t*G + g), read-only
    const uint32_t* lat_xlseq = nullptr;         ///< the D stage's doorbell word; >= layer+1 means the rows are in
    const uint8_t* lat_act0 = nullptr;          ///< DESIGN-4: the D rows as Q8_0, quantized on the GPU (mapped)
    /// Set when `dispatch` could not produce an answer.  The loop itself has no error channel, so this is
    /// where a source failure surfaces: the driver checks it after `session_loop` returns rather than the
    /// engine computing from a half-filled `parts`.
    ///
    /// **`failed` LATCHES AND `fail` IS ONLY THE MESSAGE.**  Clearing the message must not re-arm the adapter:
    /// a session that failed at layer 5 has already fed `moe_combine` whatever `parts` held, so every layer
    /// after it is built on a hole - and resuming into a plausible-looking token is the exact outcome this
    /// whole mechanism exists to prevent.  `failed` is what the short-circuit reads.
    bool failed = false;
    const char* fail = nullptr;
    int64_t fail_layer = -1;
    int64_t fail_expert = -1;
};

/// `strata::core::PoolFn`, exactly.  Silent on failure BY SIGNATURE - see `ExpertDispatch::fail`.
void expert_pool_dispatch(void* user, const float* x_f, const int32_t* ids, const float* weights, int64_t n_embd,
                          int64_t k, float* out);

/// Plan v0.3 P6: the pool for a verify window of `n_tok` tokens.  `x_f` is (n_tok, n_embd), `ids` (n_tok, k) and
/// `out` (n_tok * k, n_embd).  Each distinct missed expert is computed once for all the tokens routed to it;
/// resident experts' rows are zeroed (the GPU adds them).  Requires `host_res` (the token-graph residency).
/// STRATA_ROUTE_TAIL_SKIP rank (0: off).  Unset in the environment: 0 until the serve start-up sets the default.
int tail_skip_rank();
void set_tail_skip_rank(int r);

void expert_pool_dispatch_multi(ExpertDispatch& d, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k,
                                float* out, strata::kernels::cpu::NativeArithmetic arithmetic = strata::kernels::cpu::NativeArithmetic::Default);

/// **THE HITS, LAUNCHED AFTER THE MISSES ARE STAGED AND BEFORE `post[l]`.**  Same shape as `PoolFn` and for the
/// same reason: `session_loop` owns the ORDER and this owns the work, so the loop needs to know nothing about
/// expert caches.  Returns immediately when there are no hits, which is every layer until the cache is warm.
///
/// It must run AFTER the host has copied the misses into `parts_dev` (it writes into the same buffer, on rows
/// the CPU zeroed) and BEFORE `post[l]` (which reads it).  Both are stream-ordered on the loop's own stream.
void expert_hit_run(void* user, void* stream, HitPhase phase, const int32_t* ids, int64_t k);
/// The pool half of the same decision; see `ExpertDispatch::is_hit`.

/// **PHASE 2'S ONLY SOURCE: `experts.bin`, memory-mapped, no cache.**
///
/// `experts.bin` is 33,973,862,400 B and `BLOB` is 1,382,400, so it holds exactly `48 x 512 = 24,576` blobs and
/// the index is `layer * 512 + expert` with NO padding.  A mapping is therefore the whole implementation: the
/// blob pointer is base plus a multiply, and the page fault that follows is the read.
///
/// **AND THAT IS NOT AS SLOW AS IT SOUNDS, WHICH IS THE POINT.**  The expert set is 34 GB and this machine has
/// 64 GB of DDR5, so a warm OS page cache holds ALL of it - the second token onward is a DRAM read at the
/// measured 44.14 GB/s, not a disk read.  The first token's 663.6 MB comes off the disk and is the cold-path
/// number; `benches` should report the two separately rather than blending them.
///
/// IT IS NOT AN LRU, AN LFU OR A PREFETCHER.  Phase 3 replaces this object with those.  Phase 2 is hit rate
/// `h = 0` on purpose, so that the CPU path and the host round trip are exercised on every layer of every
/// token - which is the only way they get debugged before speed matters.
class FileExpertSource : public ExpertSource {
public:
    FileExpertSource() = default;
    ~FileExpertSource() override;
    FileExpertSource(const FileExpertSource&) = delete;
    FileExpertSource& operator=(const FileExpertSource&) = delete;

    /// Maps `<pack_dir>/experts.bin` and checks its size against the loaded expert layout.  Canonical packs use
    /// `n_layers * n_expert * BLOB`; native packs use their variable per-layer blob sizes and offsets.
    ///
    /// The size check is not a formality: a short file would fault at the END of a long sequence, and an
    /// over-long one means the pack is not the one the geometry came from.  Refuses with the two numbers.
    ///
    /// CS-T: a native pack WITHOUT experts.bin, after `set_gguf`, maps the model's GGUF shards instead (every file
    /// native_experts.txt names, after `check_experts_gguf`), and assembles a blob from its three role slices when
    /// it is asked for: the SSD tier, read in place through the OS file cache, with no 30-77 GB experts.bin copy.
    /// With experts.bin present nothing changes.
    bool open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err);
    /// CS-T: the --native shard (native_experts.txt names the other files beside it); see `open`.
    void set_gguf(const std::string& native) { gguf_ = native; }
    /// STRATA_NUMA_SPLIT: files the engine read at start outside this source (the pack's dense.bin, the MTP head),
    /// whose cached pages the placement's drop gives back too.  Set BEFORE `pin_cache_complement`.
    void set_extra_drop_files(std::vector<std::string> paths) { extra_drop_files_ = std::move(paths); }
    /// Whether the experts are read from the GGUF shards in place (no experts.bin).
    bool gguf_mode() const { return !role_ptr_.empty(); }
    /// Pin a compact host mirror of experts absent from a fully filled static GPU cache. The mmap remains open
    /// as a fallback for later cache reloads. This is opt-in because the complement may still be a large allocation.
    ///
    /// The resident RAM mode (`--resident-experts`, `--resident-cpu-experts`):
    ///   - `pin`: page-locked and mapped (`cudaHostAlloc`), so the prompt path copies it by DMA and the verify
    ///     window may read a share of the misses over PCIe; when the driver refuses, ordinary memory locked in the
    ///     working set instead.  `pin = false` is ordinary pageable memory (the ROCm arm: large pinned allocations
    ///     can fail there, and it is what the HIP measurements used).
    ///   - `lend_from_slot` >= 0: the GPU-cache slots from there to the end are the prompt path's lend region; their
    ///     experts are kept in RAM too, from the last slot down, as far as available memory minus headroom allows
    ///     (a lent slot's expert is streamed during the prompt and copied back after it).
    ///   - the rest (the experts no slot holds) must fit that budget, or nothing is allocated and this returns false.
    ///
    /// CS-T, `budget_bytes` > 0 (`--resident-budget-gib`): only as many of those experts as fit `budget_bytes`, taken
    /// in `rank` order (the expert profile: the hottest after the GPU cache's), are copied; the rest stay on the
    /// mapped files (the SSD tier).  No lend region then (a lent slot's expert is read from the files).
    /// Available memory is limited by both RAM and commit capacity on Windows.
    /// #467: `budget_bytes` = `kResidentWhatFits` sizes that path from available memory minus headroom and the #403
    /// margin - the soft --resident-experts mode's second try when the whole complement does not fit;
    /// false when not even one expert fits.  On Windows the mapped experts leave the working set before any reading.
    static constexpr uint64_t kResidentWhatFits = ~0ull;
    bool pin_cache_complement(
        const ExpertCache& cache, std::string& err, bool pin = true,
        const std::vector<std::pair<int32_t, int32_t>>& additional_gpu_pairs = {}, int64_t lend_from_slot = -1,
        uint64_t headroom_bytes = 8ull << 30, uint64_t budget_bytes = 0,
        const std::vector<std::pair<int32_t, int32_t>>* rank = nullptr);
    /// A layer split: the prompt path borrows the TAIL SLOTS of every stage's cache - CUDA0's own part
    /// first, then every stage in order.  Set these BEFORE `pin_cache_complement` to keep those slots' experts
    /// in the RAM copy too: each region lists its stage's (layer, expert) pairs, the highest cache slot first,
    /// as the borrowing takes them.  Pairs past the budget keep the mapped-file fallback.  Empty (the
    /// default) keeps #848's shape: with `additional_gpu_pairs` the single lend region is off and the copy
    /// holds only the complement.
    void stage_lend_regions(std::vector<std::vector<std::pair<int32_t, int32_t>>> regions) {
        stage_lend_regions_ = std::move(regions);
    }
    void close();

    bool mapped() const { return base_ != nullptr; }
    int64_t blobs() const { return blobs_; }
    uint64_t pinned_bytes() const { return complement_pinned_ ? complement_pin_limit_ : 0; }
    uint64_t resident_bytes() const { return complement_bytes_; }
    bool complement_pinned() const { return complement_pinned_; }
    bool complement_ready() const { return complement_ready_; }
    uint64_t locked_bytes() const { return complement_locked_; }
    /// Lend-region slots whose experts the compact copy holds (the last ones of the cache).
    int64_t resident_lent_slots() const { return complement_lent_slots_; }

    // ---- the resident RAM mode and the adaptive tier.  A swap puts `in` (held here) into a GPU slot and evicts
    // `out` (held only by that slot).  Before the slot is overwritten the caller copies it back into an exchange
    // buffer and calls `stage_exchange`: `out` is then read from that buffer, and `in` still from here (the CPU
    // computes both until the swap lands).  Once the slot copy has landed, `commit_exchanges` moves `out` into
    // `in`'s place, so the copy keeps holding exactly the experts the GPU does not - with no read of the file.
    // STRATA_EXCHANGE_ROTATE=1 can instead transfer buffer ownership (fully pinned/mapped slots; a pack whose layers
    // differ in block size keeps spares per size).
    /// Whether the compact copy holds `(layer, expert)`.
    bool has_resident(int64_t layer, int64_t expert) const;
    /// Host room for `n` evicted blobs (page-locked when possible).  Idempotent for the same or a smaller `n`.
    /// `bounce`: how many of the last buffers are only staged through and never hold an exchange (STRATA_EXCHANGE_ROTATE
    /// keeps those in the largest block size alone).
    bool reserve_exchanges(int64_t n, std::string& err, int64_t bounce = 0);
    int64_t exchange_capacity() const { return xstage_cap_; }
    /// The exchange buffers are page-locked (cudaHostAlloc): copies to and from them are asynchronous.
    bool exchange_pinned() const { return xstage_pinned_; }
    /// Buffer `q`.  `layer` >= 0: the buffer an exchange of that layer's experts stages into (STRATA_EXCHANGE_ROTATE with
    /// several block sizes keeps one per size); without it, one that holds any layer's block.
    uint8_t* exchange_buffer(int64_t q, int64_t layer = -1) const;
    /// Requires `has_resident(layer, in)`, `!has_resident(layer, out)` and `exchange_buffer(q)` holding out's blob.
    bool stage_exchange(int64_t layer, int64_t in, int64_t out, int64_t q);
    /// --pipeline-windows: size the exchange table now, so a `stage_exchange` on the adaptive tier's thread never
    /// reallocates it under a concurrent `blob` (the pool reads it while windows are in flight).
    void prepare_overrides() { if (override_.empty()) override_.assign((size_t) blobs_, nullptr); }
    /// After the GPU copies of every staged swap have landed.  Returns how many exchanges were applied.
    int64_t commit_exchanges();
    /// 0.1.43 (STRATA_ADAPT_OVERLAP): BEFORE the GPU copies of the staged swaps have landed, make each staged `out` look to the
    /// planner as it will after `commit_exchanges` (pinned, with a device alias into its exchange buffer, so the PCIe share
    /// treats it exactly as a committed one) without touching `in`'s place, which the copy still reads.  `commit_exchanges`
    /// (after the copies have landed) ends it.  False: not possible here (rotation, exchange buffers not page-locked): the
    /// caller waits for the copies instead.
    bool soft_commit_exchanges();
    /// `commit_exchanges` in two halves, for the asynchronous adaptive tier (--adapt-async): `commit_copies` moves
    /// every staged evicted blob into its `in`'s place in the copy (on another thread: safe once nothing computes `in`
    /// from RAM - it is resident on the GPU - while `out` is still read from its exchange buffer), then `commit_flip`
    /// (the caller's thread, between windows) points `out` there and drops the staging.  Returns how many were applied.
    void commit_copies();
    int64_t commit_flip();
    /// #2134: the Foresight swap space (STRATA_FS_SLOTS), null = none.  Each commit above holds the staged experts in it
    /// (ForesightSwap::hold: their slots are dropped and not refilled) while their bytes move, so no slot is served
    /// with bytes copied from a place an exchange rewrote.
    void set_foresight_swap(ForesightSwap* fs) { fs_swap_.store(fs, std::memory_order_release); }
    /// The compact copy's blob of `(layer, expert)`, or null; not counted as a read (any thread).
    const uint8_t* resident_blob(int64_t layer, int64_t expert) const;
    int64_t exchanges() const { return exchanges_; }
    /// STRATA_RAM_ADAPT: the compact copy gives `out`'s place to `in`, read from the file, so the RAM tier can follow
    /// the conversation as the VRAM tier does.  Both are experts of `layer` (one blob size), `out` is held by the copy
    /// and `in` is not, no exchange is staged, and nothing may be reading `out`'s blob: the caller's thread, between
    /// windows, with no adaptive round in flight.  A failed read leaves the copy as it was.  Not with
    /// STRATA_EXCHANGE_ROTATE (false, nothing changes).
    bool replace_resident(int64_t layer, int64_t in, int64_t out);
    /// STRATA_RAM_ADAPT: the same move as replace_resident, but `in` is copied out of the finished stage buffer
    /// a demand read left it in (no file read).  False, nothing changed, when `in` is not staged or still filling.
    bool replace_resident_staged(int64_t layer, int64_t in, int64_t out);
    int64_t ram_adapts() const { return ram_adapts_; }
    int64_t ram_staged() const { return ram_staged_; }   ///< of ram_adapts(): copied out of a stage buffer
    // ---- 149-prefetch (STRATA_RAM_ADAPT2): the same promotion, staged.  `ram_stage` queues the read of `in` on an I/O thread
    // (false: already queued, the queue is full, or no compact copy); `ram_install` - called by the decode thread at a window
    // boundary, when no window reads the RAM copy - puts every read that has landed into `out`'s place, unless the GPU tier
    // took `in` or `out` meanwhile (`host_res`: the residency table, n_expert per layer) or the copy no longer fits.
    bool ram_stage(int64_t layer, int64_t in, int64_t out);
    int64_t ram_install(const int32_t* host_res, int64_t n_expert);
    bool ram_queued(int64_t layer, int64_t expert) const;   ///< `expert` is the `in` or `out` of a staged promotion
    int64_t ram_staged_reads() const { return ram_staged_reads_.load(std::memory_order_relaxed); }
    int64_t ram_install_skips() const { return ram_install_skips_.load(std::memory_order_relaxed); }
    int64_t ram_install_gpu() const { return ram_install_gpu_.load(std::memory_order_relaxed); }   ///< skipped: the GPU tier took in or out
    /// Where an expert outside the GPU is now: 0 the RAM copy, 1 the files with its pages in the page cache, 2 the files, cold
    /// (Linux: four sampled pages; elsewhere 2).
    int tier_of(int64_t layer, int64_t expert) const;
    uint64_t blob_bytes(int64_t layer) const {
        return layer >= 0 && (size_t) layer < layer_blob_bytes_.size() ? layer_blob_bytes_[(size_t) layer] : 0;
    }
    bool exchange_rotation() const { return exchange_storage_.active(); }
    uint64_t rotated_exchanges() const { return exchange_storage_.exchanges(); }
    uint64_t avoided_exchange_copy_bytes() const { return exchange_storage_.avoided_bytes(); }
    /// With the compact copy ready: blobs read from the mapped file since (what the plain mmap mode may read from
    /// the SSD).  0 in a steady resident mode; lend-region experts that did not fit the RAM count here.
    int64_t file_reads() const { return file_reads_.load(std::memory_order_relaxed); }
    /// CS-T per-tier counters: blobs served from the RAM copy, and the bytes read from the mapped files (the blobs
    /// `file_reads` counts, plus the prompt path's copies of them).
    int64_t ram_reads() const { return ram_reads_.load(std::memory_order_relaxed); }
    uint64_t file_read_bytes() const { return file_read_bytes_.load(std::memory_order_relaxed); }
    /// Of those, the bytes `blob` read (decode windows, adaptive swaps; the rest are the prompt path's copies), and
    /// the time spent reading the files, summed over threads.
    uint64_t file_blob_bytes() const { return file_blob_bytes_.load(std::memory_order_relaxed); }
    double file_ms() const { return (double) file_us_.load(std::memory_order_relaxed) / 1000.0; }
    /// Threads `prefetch` reads the GGUF with (STRATA_FETCH_THREADS, default 8).
    void set_fetch_threads(int n) { fetch_threads_ = n < 1 ? 1 : n; }
    /// #286: read the experts straight from the drive (Windows: FILE_FLAG_NO_BUFFERING, overlapped; Linux: O_DIRECT and
    /// the kernel's asynchronous reads) instead of
    /// through the mapped files - the GGUF in place, or a pack's experts.bin - when the file cache could not keep them
    /// beside `ram_bytes` (the RAM budget), cached now or not (their mapped pages would land in the working set); see
    /// experts_unbuffered.  The mapped reads' page faults are one small request each, and the pages they bring in
    /// take the RAM the budget was sized for.  `why` says what decided.
    /// #577: `ram_bytes` counts up to every expert, and only the experts outside it are what the cache must keep.
    bool set_unbuffered(uint64_t ram_bytes, std::string& why);
    /// #577: the same decision once the RAM copy is built (pin_cache_complement), from the RAM it really holds and the
    /// expert bytes outside it; switches either way (startup only, nothing reading).  Returns whether unbuffered.
    bool recheck_unbuffered(std::string& why);
    bool unbuffered() const { return !direct_.empty(); }
    /// MODEL-UPDATE round 2 (Linux, opt-in STRATA_PROMPT_DIRECT=1): with the file tier read through the file cache (the
    /// decode's choice: re-routed experts come back from it), the prompt path's batched stager reads (copy_blobs) go
    /// unbuffered instead - the same bytes, read in merged O_DIRECT requests.  False when not opened (and why).
    bool open_prompt_direct(std::string& why);
    /// pp-opt (Windows, experts.bin, unbuffered): close the mapped view once startup no longer needs it.  NTFS runs
    /// the unbuffered reads of a mapped file one at a time - on a PCIe 4 NVMe (WD SN580) 2 MiB reads at queue depth
    /// 32 measured 2.35 GB/s with experts.bin mapped and 3.57 GB/s without.  After this every file read goes through
    /// read_direct (mapped_blob answers nullptr).  Returns whether the view was closed; `why` says why not.
    bool drop_mapping(std::string& why);
    /// Windows, experts.bin, a file tier that reads through the file cache (--resident-experts): the startup reads -
    /// the GPU cache's fill from the profile and the RAM copy (pin_cache_complement) - go unbuffered (read_direct)
    /// with the view closed meanwhile.  Through the mapping they put the file's pages in the working set beside the
    /// RAM copy, and on a PC short of RAM that pages other memory out.  The bytes and their places are the same.
    /// end_startup_unbuffered maps the view again and closes the unbuffered handles and the stage buffers, as before
    /// the startup.  STRATA_STARTUP_UNBUFFERED=0: the mapped copy (A/B).  False (and `why`) when it does not apply.
    bool begin_startup_unbuffered(std::string& why);
    /// False when the view could not be mapped again: then the file tier stays unbuffered, `why` says so.
    bool end_startup_unbuffered(std::string& why);
    bool startup_unbuffered() const { return startup_ub_; }
    /// Every expert's bytes (n_layers x n_expert blobs).
    uint64_t expert_bytes() const;
    /// #286, unbuffered: assembles the blobs of these pairs ahead of the `blob` calls that will ask for them (the
    /// GPU cache's fill from the profile) - one batch of reads instead of one blob at a time.  At most 64 pairs.
    void prefetch_pairs(const std::pair<int32_t, int32_t>* pairs, int64_t n);
    /// Mapped reads: asks the OS for these pairs' blobs ahead (platform::advise_willneed).  False when unbuffered,
    /// when read-ahead is off, or before `open`.
    bool advise_pairs(const std::pair<int32_t, int32_t>* pairs, int64_t n) const override;

    const uint8_t* blob(int64_t layer, int64_t expert) override;
    /// With io prefetch in the experts.bin tier, the mapping itself (blob() may hand out a recycled staging buffer).
    const uint8_t* blob_stable(int64_t layer, int64_t expert) override;
    int blob_parts(int64_t layer, int64_t expert, BlobPart* out) override;
    void note_async_read(const uint8_t* blob, void* stream) override;
    /// Windows, opt-in with STRATA_FILE_RELEASE=1: trim the file mapping's full pages after their expert reaches
    /// VRAM. Never touches the resident complement or staging buffers. Disabled by default.
    uint64_t release(int64_t layer, int64_t expert) override;
    bool pinned(int64_t layer, int64_t expert) const override;
    const uint8_t* device_alias(int64_t layer, int64_t expert) const override;
    bool pcie_layer(int64_t layer) const override;
    bool transient(int64_t layer, int64_t expert) const override;
    double cached_share(int64_t samples) const override;
    bool copy_blob(int64_t layer, int64_t expert, uint8_t* dst) override;
    /// pp-opt, unbuffered: the file-tier blobs of the run in one read_direct batch (neighbours merged into one request)
    bool copy_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n) override;
    /// CS-T: advances the assembled blobs' age (see staged_blob).
    void begin_layer(int64_t layer, const int32_t* ids, int64_t k) override;
    /// CS-T: the GGUF in place assembles the missed experts on `fetch_threads_` threads.
    void prefetch(int64_t layer, const int64_t* experts, int64_t n) override;
    int64_t prefetch_begin(int64_t layer, const int64_t* experts, int64_t n, uint8_t* pending) override;
    int64_t wait_landed(int64_t layer, const int64_t* experts, int64_t n, uint8_t* state) override;
    void prefetch_end() override;
    bool zc_roles(int64_t layer, int64_t expert, const uint8_t** p) override;
    /// 149-cold: the wall time of the last background read batch (prefetch_begin -> its last blob published), us
    double async_fill_us() const { return async_fill_us_; }
    /// CS-T: the GGUF in place asks the OS for the predicted experts' pages (PrefetchVirtualMemory on Windows,
    /// madvise(WILLNEED) elsewhere), skipping the RAM copy's.
    void warm(int64_t layer, const int64_t* experts, int64_t n) override;
    /// 149-prefetch: the scheduled read-ahead (STRATA_IO_PF_SCHED=1 with STRATA_IO_PREFETCH=1): admission by confidence
    /// against the queue's pressure, a speculative byte budget per decode window, a deadline per read.
    void warm_scored(int64_t layer, const int64_t* experts, const float* conf, int64_t n, int ahead) override;
    bool io_sched() const { return io_sched_; }
    bool wants_scores() const override { return io_sched_ && io_pf_; }
    /// Not when the reads are unbuffered: the warmed pages would be read through the file cache, a second time.
    bool warms() const override { return (!role_ptr_.empty() || io_pf_) && direct_.empty(); }
    /// Of the blobs the file tier read for the decode, how many had been warmed for their layer beforehand.
    int64_t warmed_hits() const { return warm_hits_.load(std::memory_order_relaxed); }
    int64_t warmed() const { return warm_count_.load(std::memory_order_relaxed); }
    /// #286: blobs an unbuffered read could not deliver, read through the mapping instead (0 when all went direct).
    int64_t direct_fallbacks() const { return direct_fallbacks_.load(std::memory_order_relaxed); }

    /// Blobs touched, for the driver to report.  With `h = 0` this is `48 * k` per token and the number is only
    /// interesting once Phase 3 makes it not so.
    int64_t reads() const override { return reads_.load(std::memory_order_relaxed); }

    // ---- the Linux I/O path of the mapped file tier (STRATA_IO_PREFETCH=1, opt in; no effect on Windows)
    //
    // The file tier's reads are page faults: the CPU kernels (experts.bin) or the fetch threads (the GGUF in place)
    // touch the mapping, and each fault is a read of a few pages that blocks the layer.  With this on:
    //   - an expert whose pages are not in the page cache is read with ONE pread of its whole blob (large requests,
    //     several at once) into a staging buffer instead of being faulted in;
    //   - the experts the router lookahead predicts for the next layers are read the same way on a pool of I/O
    //     threads, while the current layer computes (warm() queues them; the kernels then find them staged);
    //   - an expert whose pages are all cached is still handed out as the mapped pointer (no copy).
    // The bytes are the same bytes, so the output is the same.  `threads` = I/O threads (0: default 4).
    void set_io_prefetch(bool on, int threads);
    bool io_prefetch() const { return io_pf_; }
    /// blob() hands out stage buffers it recycles (the GGUF in place, an unbuffered source, the I/O read-ahead): one
    /// thread at a time (STRATA_PIPED_CONCURRENT is off with such a source)
    bool stages_blobs() const { return staged(); }
    /// Counts the cached / uncached bytes of every blob read from the files (mincore, page-granular).  Free with
    /// io_prefetch; otherwise opt in with STRATA_IO_STATS=1 (the check costs a few microseconds per blob).
    void set_io_stats(bool on) { io_stats_ = on || io_pf_; }
    bool io_stats() const { return io_stats_; }
    struct IoCounters {
        uint64_t cached_bytes = 0, uncached_bytes = 0;       ///< file reads at hand-out, by page-cache residency
        uint64_t pread_bytes = 0, pread_us = 0, pread_n = 0; ///< bytes read with pread, thread time, calls
        uint64_t pf_demand = 0;                              ///< blobs the layer asked for that the workers were handed
        uint64_t pf_jobs = 0, pf_read_blobs = 0, pf_resident_skips = 0, pf_dropped = 0;   ///< prefetch workers
        uint64_t pf_used = 0, pf_unused = 0;                 ///< prefetched blobs a layer then used / never used
        uint64_t crit_us = 0;                                ///< wall time the decode waited on file reads (prefetch + staged_blob)
        uint64_t crit_n = 0;
        // 149-prefetch: the read-ahead's bytes and what came of them
        uint64_t pf_bytes = 0;                               ///< bytes the read-ahead workers read (all modes)
        uint64_t pf_used_bytes = 0;                          ///< of them, blobs a layer then asked for (arrived before demand)
        uint64_t pf_late = 0;                                ///< reads dropped because the decode had reached their layer
        uint64_t pf_rej_conf = 0, pf_rej_budget = 0;         ///< predictions not admitted (STRATA_IO_PF_SCHED)
        uint64_t pf_budget_bytes = 0;                        ///< the current per-window speculative budget (scheduler)
    };
    IoCounters io_counters() const;

private:
    const uint8_t* resident_blob(size_t index) const;
    const uint8_t* mapped_blob(int64_t layer, int64_t expert) const;
    /// The blob's bytes from the mapped file(s) - experts.bin, or the three GGUF role slices - into `dst`.
    bool copy_from_files(int64_t layer, int64_t expert, uint8_t* dst) const;
    bool open_gguf(std::string& err);
    const uint8_t* staged_blob(int64_t layer, int64_t expert);
    bool claim_stage(int64_t key, size_t& v, bool& fill, bool ahead = false);
    bool fill_stage(size_t v, int64_t layer, int64_t expert, uint8_t* dst);
    void publish_stage(size_t v, int64_t layer, bool ok, double us);
    struct Fill { size_t v; int64_t layer, e; uint8_t* dst; };
    /// The claimed buffers' blobs: one overlapped batch when unbuffered, else the fetch threads' mapped copies.
    void fill_many(const std::vector<Fill>& todo);
    /// 149-cold: the claim half of `prefetch` (the blobs this call must fill; pending[i] = 1 for those, if given)
    void claim_todo(int64_t layer, const int64_t* experts, int64_t n, std::vector<Fill>& todo, uint8_t* pending,
                    bool zc = false);
    std::vector<int64_t> zc_keys_;   ///< 149-cold mode 3: the keys the last prefetch_begin left in the mapping
public:
    /// 149 STRATA_IO_PF_GATE=1 (with STRATA_IO_PREFETCH=1): the decode's read-ahead and its pread demand reads run only
    /// while the page cache can hold enough of the file tier: (the memory available, cgroup limits included, + the
    /// file-tier bytes cached now) / the experts' bytes, checked every 16 decode windows; it closes below
    /// STRATA_IO_PF_GATE_LO 0.55 and opens above STRATA_IO_PF_GATE_HI 0.65.  Closed, the decode reads as without io
    /// prefetch; the prompt path is not gated.  (A held share at demand would not do: the read-ahead itself raises it.)
    /// `DecodeScope` marks the decode thread (and the fetch threads it starts).
    double gate_ratio() const { return gate_ratio_; }
    struct DecodeScope {
        bool prev;
        DecodeScope();
        ~DecodeScope();
    };
    static bool in_decode();
    bool gate_open() const { return !gate_on_ || gate_open_.load(std::memory_order_relaxed); }
    bool read_ahead_open() const override { return gate_open(); }
#if defined(__linux__)
    bool linux_file_defaults() const override { return !role_ptr_.empty(); }   // the GGUF in place only (measured)
#endif
    /// 0.2.1: io prefetch + the decode gate by default also for an experts.bin pack (zero-copy is moot there: its blob is one
    /// contiguous range the kernels already read from the mapping; the COLD_OVERLAP default stays with the GGUF in place).
    /// Measured (RTX 3060, 43 GB pack): 32 GB of RAM decode +20..+34%, 16 GB decode +3..+8% but the cold 4K prompt -4..-6%, so
    /// below the gate's room test the pack keeps reading as before
    bool linux_io_defaults() const {
#if defined(__linux__)
        return role_ptr_.empty() ? cache_room_ratio() >= 0.55 : true;   // experts.bin: only while the page cache has room for it
#else
        return false;
#endif
    }
    /// the memory the page cache can use (the tightest cgroup limit less what is not page cache, else MemAvailable) over the
    /// experts' bytes; 0 when unknown.  The decode gate's room test (STRATA_IO_PF_GATE_LO), asked once at start.
    double cache_room_ratio() const;
    bool gate_on() const { return gate_on_; }
    double gate_held() const { return gate_ewma_; }
    int64_t gate_closed_windows() const { return gate_closed_w_; }
    int64_t gate_flips() const { return gate_flips_; }
private:
    void gate_tick();
    void gate_capacity();
    bool gate_on_ = false;
    double gate_lo_ = 0.55, gate_hi_ = 0.65, gate_ewma_ = -1.0, gate_ratio_ = -1.0;
    int64_t gate_win_ = 0, gate_useless_since_ = 0;
    uint64_t gate_prev_pf_ = 0, gate_prev_used_ = 0;
    double gate_useful_ = 1.0, gate_min_useful_ = 0.10;
    bool gate_useless_ = false;
public:
    double gate_useful() const { return gate_useful_; }
private:
    std::atomic<bool> gate_open_{true};
    mutable std::atomic<uint64_t> gate_c_{0}, gate_u_{0};
    uint64_t gate_prev_c_ = 0, gate_prev_u_ = 0;
    int64_t gate_closed_w_ = 0, gate_flips_ = 0;
    std::vector<Fill> async_todo_;
    std::vector<std::thread> async_th_;
    std::atomic<size_t> async_next_{0};
    std::atomic<int64_t> async_end_ns_{0};
    std::chrono::steady_clock::time_point async_t0_{};
    double async_fill_us_ = 0;
    /// #286: the blobs from the drive, unbuffered: every role's 4 KiB-aligned window is read at once (overlapped on
    /// Windows, io_submit on Linux) into this thread's aligned buffer, then copied into place.  False when a read fails.
    bool read_direct(const Fill* fills, size_t n) const;
    bool open_direct(std::string& why);
    /// Windows: experts.bin mapped again as `open` maps it (begin/end_startup_unbuffered).
    bool map_view(std::string& why);
    std::vector<std::string> paths_;          ///< the mapped files, as maps_
    std::vector<void*> pdirect_;              ///< open_prompt_direct: O_DIRECT fds for copy_blobs only
    std::vector<void*> direct_;               ///< #286: per file, an unbuffered handle (Windows) or O_DIRECT fd (Linux)
    std::vector<int> role_file_;              ///< 3 x n_layers: index into maps_ / direct_
    /// blobs assembled in the stage buffers (`blob` hands those out): the GGUF in place, or any unbuffered source
    bool staged() const { return !role_ptr_.empty() || !direct_.empty() || io_pf_; }
    static constexpr uint64_t kNoComplement = detail::kNoCacheComplement;
    // ---- CS-T: the GGUF shards in place
    std::string gguf_;
    std::vector<std::string> extra_drop_files_;   ///< set_extra_drop_files
    struct Map {
        const uint8_t* base = nullptr;
        uint64_t bytes = 0;
        bool closed = false;   ///< drop_mapping unmapped the view (base stays for offsets)
#if defined(_WIN32)
        void* file = nullptr;
        void* mapping = nullptr;
#else
        int fd = -1;
#endif
    };
    std::vector<Map> maps_;
    std::vector<const uint8_t*> role_ptr_;    ///< 3 x n_layers: gate / up / down of the layer's expert 0
    std::vector<uint64_t> role_bytes_;        ///< 3 x n_layers: bytes per expert of that role
    // the blobs assembled for `blob()`: a small pool of buffers, one per recent (layer, expert).  A buffer is
    // reused only once `kStageAge` layer changes have passed since its blob was last asked for, so a pointer holds
    // through the layer it was asked in and the next ones (the pool computes a layer's misses before the next).
    static constexpr uint64_t kStageAge = 3;
    // A stage buffer feeds cudaMemcpyAsync (the cache fill), so it is cudaHostAlloc'd where the driver allows:
    // a pageable source would be staged through the driver's bounce buffer - an extra copy at a fraction of
    // the transfer rate, with the calling thread doing it.  `pinned` picks the free (defined in the .cpp: the
    // header has no cuda_runtime.h).
    struct StageBufFree {
        bool pinned = false;
        void operator()(uint8_t* p) const noexcept;
    };
    std::mutex stage_mu_;
    std::vector<std::unique_ptr<uint8_t[], StageBufFree>> stage_buf_;
    std::vector<int64_t> stage_key_;
    std::vector<uint64_t> stage_epoch_, stage_used_;
    std::vector<char> stage_busy_;            ///< being filled (outside stage_mu_): never a victim
    std::condition_variable stage_cv_;
    int fetch_threads_ = 8;
    std::atomic<uint64_t> file_blob_bytes_{0}, file_us_{0};
    std::unique_ptr<std::atomic<uint32_t>[]> warm_stamp_;   ///< per (layer, expert): epoch_ + 1 when warmed
    std::atomic<int64_t> warm_hits_{0}, warm_count_{0};
    mutable std::atomic<int64_t> direct_fallbacks_{0};
    std::unordered_map<int64_t, size_t> stage_of_;
    uint64_t stage_blob_ = 0;
    uint64_t stage_seq_ = 0;
    int64_t ram_adapts_ = 0;                  ///< replace_resident(_staged): experts moved into the compact copy
    int64_t ram_staged_ = 0;                  ///< replace_resident_staged: of those, copied out of a stage buffer
    std::vector<uint8_t> ram_adapt_buf_;      ///< ...one blob, read here before it replaces another
    // 149-prefetch: the staged promotions (ram_stage / ram_install)
    struct RamJob { int64_t layer, in, out; std::vector<uint8_t> buf; int state; };   ///< state 0 queued, 1 reading, 2 landed, 3 failed
    mutable std::mutex ram_mu_;
    std::condition_variable ram_cv_;
    std::deque<std::unique_ptr<RamJob>> ram_jobs_;
    std::thread ram_thread_;
    bool ram_quit_ = false;
    std::atomic<int64_t> ram_staged_reads_{0}, ram_install_skips_{0}, ram_install_gpu_{0};
    bool install_resident(int64_t layer, int64_t in, int64_t out, const uint8_t* src);   ///< the second half of replace_resident
    void ram_worker();
    void ram_stop();
    uint64_t epoch_ = 0;
    int64_t last_layer_ = -1;
    bool stage_grew_ = false;
    bool stage_pin_said_ = false, stage_pin_failed_said_ = false;   ///< the one-time notes in claim_stage
    // #1237: per stage buffer, the event recorded after the last asynchronous copy out of it (note_async_read); a
    // buffer is refilled only once that event has completed.  cudaEvent_t, stored untyped (no cuda_runtime.h here).
    std::vector<void*> stage_ev_;
    std::vector<char> stage_ev_live_;                 ///< a copy was queued since the buffer was last waited for
    std::unordered_map<const uint8_t*, size_t> stage_idx_;   ///< buffer address -> index (the buffers never move)
    void stage_free_events();
    std::atomic<uint64_t> stage_fence_waits_{0};      ///< refills that had to wait for a queued copy
    std::atomic<int64_t> ram_reads_{0};
    std::atomic<uint64_t> file_read_bytes_{0};
    // ---- the Linux I/O path (set_io_prefetch)
    bool io_pf_ = false;
    bool io_fill_ = false;                            ///< experts.bin: the workers only fill the page cache (large preads), no staging, nothing waits
    bool io_ahead_ = true;                            ///< STRATA_IO_PF_AHEAD=0: no read-ahead workers' queue, demand preads only
    bool io_stats_ = false;
    int io_threads_ = 4;
    std::vector<std::thread> io_workers_;
    std::mutex io_mu_;
    std::condition_variable io_cv_;
    std::deque<std::pair<int32_t, int32_t>> io_q_;   ///< (layer, expert) to read ahead
    bool io_quit_ = false;
    // ---- 149-prefetch: the scheduled read-ahead (STRATA_IO_PF_SCHED=1)
    struct IoJob { int32_t layer, expert; float conf; uint64_t deadline; uint64_t bytes; };
    bool io_sched_ = false;
    float io_minconf_ = 0.5f;                          ///< STRATA_IO_PF_MINCONF: the bar with an empty queue
    uint64_t io_budget_fixed_ = 0;                     ///< STRATA_IO_PF_BUDGET_MB (0: auto from the measured rates)
    double io_budget_frac_ = 0.5;                      ///< STRATA_IO_PF_BUDGET_FRAC: share of what the drive can read in a window
    std::vector<IoJob> io_jobs_;                       ///< (under io_mu_) admitted, not yet started
    uint64_t io_q_bytes_ = 0;                          ///< (under io_mu_) their bytes
    uint64_t io_win_bytes_ = 0;                        ///< (under io_mu_) admitted in the current window
    double io_bw_ = 0.0;                               ///< (under io_mu_) EWMA bytes/us of one worker's reads
    double io_win_us_ = 0.0;                           ///< (under io_mu_) EWMA of a decode window's wall time
    std::chrono::steady_clock::time_point io_win_t0_{};
    double io_scale_ = 1.0;                           ///< (under io_mu_) the budget's feedback factor: shrinks while read-ahead goes unused
    double io_useful_ = -1.0;                         ///< (under io_mu_) EWMA of used / read-ahead bytes
    uint64_t io_prev_pf_ = 0, io_prev_used_ = 0;       ///< (under io_mu_) the counters at the last window tick
    double io_target_ = 0.5;                          ///< STRATA_IO_PF_TARGET: the used share the feedback aims at (0: no feedback)
    std::atomic<uint64_t> io_epoch_{0};                ///< the decode's layer clock (begin_layer), read by the workers
    std::unique_ptr<std::atomic<uint8_t>[]> pf_mark_;  ///< per (layer, expert): read ahead, not yet asked for (fill mode)
    std::atomic<uint64_t> io_pf_bytes_{0}, io_pf_used_bytes_{0}, io_pf_late_{0}, io_pf_rej_conf_{0}, io_pf_rej_budget_{0};
    uint64_t io_budget_now() const;                    ///< (io_mu_ held) the per-window budget in bytes
    void io_window_tick();                             ///< begin_layer: a new decode window starts at layer 0
    std::vector<char> stage_pf_;                      ///< per stage buffer: filled ahead, not yet used by a layer
    std::vector<int> io_fds_;                         ///< per mapped file: a plain descriptor (POSIX_FADV_RANDOM) for the preads
    mutable std::atomic<uint64_t> io_cached_{0}, io_uncached_{0}, io_pread_bytes_{0}, io_pread_us_{0}, io_pread_n_{0};
    std::atomic<uint64_t> io_pf_jobs_{0}, io_pf_demand_{0}, io_pf_blobs_{0}, io_pf_skips_{0}, io_pf_dropped_{0}, io_pf_used_{0}, io_pf_unused_{0},
        io_crit_us_{0}, io_crit_n_{0};
    struct Span { const uint8_t* p; uint64_t n; int fd; uint64_t off; uint64_t at; };
    /// The (up to 3) file ranges of a blob: pointer in the mapping, size, descriptor and file offset, offset in the blob.
    int spans(int64_t layer, int64_t expert, Span* out) const;
    /// Bytes of the blob's pages in the page cache (mincore); `total` the blob's bytes.
    /// `exact` false: four sampled pages (a blob is cached when they are; a handful of syscalls, not hundreds of pages' walk).
    uint64_t cached_bytes(int64_t layer, int64_t expert, uint64_t& total, bool exact = true) const;
    void io_worker();
    void io_enqueue(int64_t layer, int64_t expert, bool ahead = false);   ///< ahead: a prediction (warm), not a layer's demand
    static constexpr int32_t kIoAhead = 1 << 30;      ///< io_q_: the expert field's flag for a prediction
    void io_stop();
    const uint8_t* io_blob(int64_t layer, int64_t expert);
    const uint8_t* base_ = nullptr;
    int64_t blobs_ = 0;
    int64_t n_layers_ = 0;
    int64_t n_expert_ = 0;
    uint64_t mapped_bytes_ = 0;
    std::vector<uint64_t> layer_offsets_, layer_blob_bytes_;
    void* complement_arena_ = nullptr;
    const uint8_t* complement_host_ = nullptr;
    const uint8_t* complement_device_ = nullptr;
    uint64_t complement_bytes_ = 0;
    std::vector<uint64_t> complement_offsets_;
    detail::ExchangeStorage exchange_storage_; // authoritative when active; original arenas still own memory
    bool complement_pinned_ = false;
    bool complement_partial_ = false;         ///< CS-T: only the first complement_pin_limit_ bytes are registered
    bool complement_registered_ = false;      ///< the arena is ordinary memory registered with cudaHostRegister (not cudaHostAlloc)
    uint64_t complement_pin_limit_ = 0;
    uint64_t complement_lock_off_ = 0;        ///< the working-set lock covers [lock_off, lock_off + locked)
    bool complement_ready_ = false;
    uint64_t complement_locked_ = 0;          ///< bytes held in the working set (pin refused)
    int64_t complement_lent_slots_ = 0;
    std::vector<std::vector<std::pair<int32_t, int32_t>>> stage_lend_regions_;  ///< a split: each stage's lend region
    std::vector<const uint8_t*> override_;    ///< staged exchanges: an evicted expert read from its exchange buffer
    struct Exchange { size_t in, out; int64_t q; uint64_t bytes; };
    std::vector<Exchange> staged_;
    std::atomic<ForesightSwap*> fs_swap_{nullptr};   ///< set_foresight_swap
    void fs_hold_staged(bool hold);
    std::vector<size_t> soft_in_;             ///< soft_commit_exchanges: per `out` index, the `in` whose place it will take (SIZE_MAX: none)
    const uint8_t* xstage_dev_ = nullptr;     ///< the exchange buffers' device alias (soft_commit_exchanges)
    uint8_t* xstage_ = nullptr;               ///< exchange buffers, `xstage_cap_ x xstage_blob_`
    bool xstage_pinned_ = false;
    int64_t xstage_cap_ = 0;
    uint64_t xstage_blob_ = 0;
    int64_t exchanges_ = 0;
    std::atomic<int64_t> file_reads_{0};
    std::atomic<int64_t> reads_{0};   ///< (a counter only; two threads can read blobs at once)
#if defined(_WIN32)
    void* file_ = nullptr;
    void* mapping_ = nullptr;
#else
    int fd_ = -1;
#endif
    bool unmapped_ = false;   ///< pp-opt: drop_mapping closed the view (base_ stays as the "opened" mark only)
    bool startup_ub_ = false;                 ///< begin_startup_unbuffered: until end_startup_unbuffered
    uint64_t startup_blob_bytes_ = 0, startup_us_ = 0;   ///< the staged reads' counters at its begin (put back at the end)
};

// ================================ THE RESIDENT ARENA (R2.1) ================================
//
// **THE MMAP ABOVE IS THE REVIEW'S FINDING C1 AND IT IS WORTH 2.2x.**  Read the comment on `FileExpertSource`
// about the page cache holding all 34 GB: that is true of the expert file ALONE, and it is not what this engine
// does.  The PLE/n-gram shard is a 26.8 GB file that the engine also maps, so 34 GB of experts plus 26.8 GB of
// n-gram is 60.8 GB of mapped, file-backed pages on a 63 GB machine - and file-backed pages are exactly the ones
// the OS drops from the standby list when it wants memory.  The expert stream then re-faults from disk.
//
// Measured, this round: the pool runs at **~19 GB/s in the engine against 42.8 GB/s in `bench/micro/cpu_s2.cpp`
// on the same machine, reading the same 34 GB**.  The micro does `std::fread` into a heap arena and reads
// ordinary (anonymous, resident) memory; the engine reads `MapViewOfFile`.  That is the whole difference, and it
// is why this class exists.
//
// It reads `experts.bin` into a `PinnedArena` once at startup, so the expert stream comes from anonymous memory
// the OS has no cheaper reason to evict.  `PinnedArena` also tries `cudaHostRegister`, which the GPU needs for
// Phase 3's cache fills and the CPU/PCIe miss split - but registration is best-effort and reported, not assumed.
class ArenaExpertSource : public ExpertSource {
public:
    ArenaExpertSource() = default;
    ~ArenaExpertSource() override;
    ArenaExpertSource(const ArenaExpertSource&) = delete;
    ArenaExpertSource& operator=(const ArenaExpertSource&) = delete;

    /// Allocates and loads `<pack_dir>/experts.bin`.  Prints nothing; the caller reports `note()` and the load
    /// rate, because those are the two numbers that say whether the arena is the one that was asked for.
    bool open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, int threads, std::string& err,
              uint64_t max_pinned_bytes = 0, const std::string& shared_arena_file = {});
    /// Plan v0.3 P6: a native pack without experts.bin takes its experts from the model's GGUF: `native` is the
    /// --native shard, and native_experts.txt names the other shards beside it (per layer, or per role in v4).
    void set_gguf(const std::string& native) { gguf_ = native; }
    void close();

    bool mapped() const { return base_ != nullptr; }
    int64_t blobs() const { return blobs_; }
    const uint8_t* blob(int64_t layer, int64_t expert) override;
    /// `blob` without the read counter: a pure pointer into the arena, safe from another thread (the async adaptive
    /// tier's job thread)
    const uint8_t* blob_at(int64_t layer, int64_t expert) const;
    int64_t reads() const { return reads_.load(std::memory_order_relaxed); }
    bool pinned(int64_t layer, int64_t expert) const override;
    const uint8_t* device_alias(int64_t layer, int64_t expert) const override;
    /// Prefix pinning: the old rule (device_alias(layer, 0)); pin by need: at least one pinned range in the layer.
    bool pcie_layer(int64_t layer) const override;
    void prefetch(int64_t layer, int64_t expert) override;
    uint64_t release(int64_t layer, int64_t expert) override;

    /// Pin by need (STRATA_PIN_BY_NEED=1): BEFORE open(), the (layer, expert) pairs worth pinning, by decreasing
    /// priority - the profile's ranked pairs the GPU cache will NOT hold, then the unranked ones.  open() merges
    /// neighbours into byte ranges and registers only those (PinnedArena::ByNeed); pinned()/device_alias()/pcie_layer()
    /// then follow the registered ranges instead of a prefix.  Never called: the old prefix pinning, unchanged.
    /// `n_first`: the first n_first pairs are the non-resident estimate; the rest (the pairs taken as GPU-resident, which
    /// only get what the cap leaves over) never merge into a run with them, so their ranges come after all of the first.
    void set_pin_priority(std::vector<std::pair<int32_t, int32_t>> ranked, size_t n_first = (size_t) -1) {
        pin_pairs_ = std::move(ranked);
        pin_first_ = n_first;
        pin_by_need_ = true;
    }
    /// Pin by need, after open(): the registered ranges (0 when the prefix rule, or nothing, is in force)
    size_t pin_ranges() const { return pin_start_.size(); }

    /// What backing was obtained and why, for the startup print.  "The engine adapts to the machine it is on" is
    /// only true if the engine says what it got.
    const std::string& note() const { return note_; }
    /// #633: set by `open` when the RAM available is less than the arena needs (a recommendation; it still loads)
    const std::string& ram_warning() const { return ram_warning_; }
    uint64_t pinned_arena_bytes() const { return pinned_bytes_; }   ///< what the driver registered (the pinned prefix, or the total by need)
    double load_gib_per_second() const { return gib_per_s_; }
    // Loader fix: the load, split.  `load_seconds()` is the wall clock of the load loop; the other two are
    // sums over the reader threads (see LoadStats), so on their own they say how much of that wall was spent
    // waiting for the disk and how much in memcpy + FNV-1a.
    double load_seconds() const { return load_seconds_; }
    double load_read_seconds() const { return load_read_s_; }
    double load_copy_seconds() const { return load_copy_s_; }

private:
    void* arena_ = nullptr;          ///< the PinnedArena, owned
    void* map_ = nullptr;            ///< STRATA_ARENA_MMAP: the arena file, mapped read-only (not the PinnedArena)
    uint64_t map_bytes_ = 0;
    std::vector<const uint8_t*> dev_slice_;   ///< device alias of each registered slice (or of the whole range)
    uint64_t slice_bytes_ = 0;
    const uint8_t* base_ = nullptr;
    int64_t blobs_ = 0;
    int64_t n_expert_ = 0;
    std::atomic<int64_t> reads_{0};   ///< (a counter only; two threads can read blobs at once)
    std::string note_;
    std::string ram_warning_;
    double gib_per_s_ = 0.0;
    double load_seconds_ = 0.0;
    double load_read_s_ = 0.0;
    double load_copy_s_ = 0.0;
    uint64_t pinned_bytes_ = 0;
    // pin by need: set_pin_priority's pairs, and (after open) the registered ranges - sorted starts, ends and the
    // device alias of each start - plus pcie_layer's answer per layer.  `ranges_` false = the prefix rule above.
    bool pin_by_need_ = false;
    bool ranges_ = false;
    // pin by need: the arena's page-aligned blob stride (see PadMap); `padded_` false = the packed experts.bin layout
    bool padded_ = false;
    PadMap pad_;
    /// where blob (layer, expert) starts in the arena: the ONE address function of this class
    uint64_t arena_off(int64_t layer, int64_t expert) const;
    std::vector<std::pair<int32_t, int32_t>> pin_pairs_;
    size_t pin_first_ = (size_t) -1;   // pin_pairs_[0, pin_first_) are the first tier (see set_pin_priority)
    std::vector<uint64_t> pin_start_, pin_end_;
    std::vector<const uint8_t*> pin_dev_;
    std::vector<uint8_t> layer_pcie_;
    /// index of the registered range holding [off, off + len), or -1
    int64_t pin_range_of(uint64_t off, uint64_t len) const;
    std::string gguf_;
};

/// Plan v0.3 P6: checks native_experts.txt's GGUF spans against the files, before anything is read: each layer's
/// gate/up/down at its recorded (file, offset) must be that tensor (`blk.L.ffn_<role>_exps.weight`), of the
/// layout's type and dimensions, and inside the file.  `native` is the --native shard (see set_gguf).
bool check_experts_gguf(const std::string& native, const strata::kernels::cpu::ExpertLayout& lay, std::string& err);
/// Fills `dst` (lay.total bytes, the experts.bin layout) from the GGUF files, one role at a time.
/// `unbuffered`: each chunk read past the file cache (Windows); `ready`: layer l is written only once
/// *ready > l + 1 (an arena that is still being registered).
/// `pad` (optional): the destination is the page-aligned padded layout (pin by need), see PadMap.
LoadStats load_experts_gguf(const std::string& native, uint8_t* dst, const strata::kernels::cpu::ExpertLayout& lay,
                            int threads, bool unbuffered = false, const std::atomic<int>* ready = nullptr,
                            const PadMap* pad = nullptr);

}  // namespace strata::core
