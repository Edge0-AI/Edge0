// include/strata/kernels/cpu/pool.hpp - P2.S3: the CPU expert pool.
//
// A layer runs TEN experts against ONE activation, and the expert kernel is DRAM-bound (Memory/LEDGER.md L9:
// 42.55 GB/s on 6 cores, tracking core count almost exactly).  So the pool's job is not to be clever - it is
// to keep every physical core reading expert bytes for the whole layer, and to be cheap enough that ten
// dispatches per layer cost less than one expert.
//
// WHY A FLAT BATCH AND NOT A RING.  P2.S3 describes a "lock-free SPMC ring", which is what you need when
// jobs arrive while others are still being computed.  Here they cannot: the host must SUM all ten outputs
// before the next layer starts, so a layer is a barrier by construction and the queue never holds more than
// one batch.  A ring would add a wrap-around to get wrong and buy nothing.  What is kept from the phase is
// the part that matters: `head`/`done` are single fetch_add counters, one claim per worker, no lock.
//
// THE COMPLETION PROTOCOL, because this is where a pool usually goes wrong.  `run()` waits for `done == n`
// AND for every worker to PARK.  Waiting only for `done` is not enough: a worker can still be inside the
// drain loop after its last `done` increment, and the host resetting `head` underneath it would let that
// worker claim a job from the NEXT batch before the next batch has been published.
//
// **AND `parked` IS NOT ENOUGH EITHER (issue #29).**  A worker that went to sleep (after `kSpinBeforeSleep`) is
// still counted as parked when it wakes, so for a moment after it has seen a new epoch the host believes it is
// idle.  On a card with most experts in VRAM the workers sleep in the middle of a request, tiny batches finish
// before a sleeper is awake, and that moment comes round constantly: a late worker could claim from the NEXT
// batch while the host was still writing it, run a job twice, or add to `done` after the host had reset it -
// and `done != n` then never ended, with the GPU waiting on the pool forever.  So every claim carries its
// batch: `head` is one 64-bit word `epoch | njobs | index`, a claim is a CAS that only succeeds for the epoch
// the worker woke for, and a late worker's claim simply fails.  A claim that succeeds belongs to the current
// batch, whose description the host cannot change until that job's `done` has landed.
#pragma once

#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool_plan.hpp"

#include <atomic>
#include <cstdio>
#include <memory>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::kernels::cpu {

/// One expert evaluation.  `act` is SHARED and read-only across the whole batch - that sharing is the point
/// of `s2_expert_vnni_q` and it is what saves 480 redundant activation conversions per token.
struct ExpertJob {
    const uint8_t* blob = nullptr;   ///< one 1,382,400-byte expert
    const ActQ* act = nullptr;       ///< the layer's quantized activation, shared
    float* out = nullptr;            ///< H floats, written by exactly one worker
    float weight = 1.0f;             ///< the router weight; applied by the HOST, not here
    int slot = -1;                   ///< the job's index, for diagnostics
};

/// Plan v0.3 P6: one expert for the `nt` tokens of a verify window that were routed to it.  Every token's output
/// is bitwise the single-token job's.
struct ExpertJobMulti {
    const uint8_t* blob = nullptr;
    int nt = 0;
    const ActQ* act[MAXT] = {};
    float* out[MAXT] = {};
    /// Plan v0.3 P6: a native pack's activations (the layer's `vec_dot_type`), one per token.
    const void* nact[MAXT] = {};
    /// 149-cold: this job's own layout (null: the layer's) - the GGUF read in place, where gate / up / down are three
    /// tensors: `blob` is the gate rows and the offsets reach the expert's up and down rows in the mapping.
    const NativeFmt* fmt = nullptr;
};

/// How worker threads are allocated across physical/logical CPU cores (#272).  `All` is the layout the pool has
/// always used and the default; the hybrid-aware ones are opt-in (--pool-affinity auto|p-cores).
enum class PoolAffinity {
    Auto,      ///< Hybrid: prioritize physical P-cores, then SMT, then E-cores (defaults to P-core count)
    PCores,    ///< Restrict workers strictly to Performance cores and their SMT siblings
    All,       ///< The default: one worker per physical core in the OS's order, without hybrid distinction
};

struct CpuTopology {
    bool is_hybrid = false;
    int p_cores = 0;                ///< Physical performance cores
    int p_threads = 0;              ///< Total logical threads on performance cores
    int e_cores = 0;                ///< Efficient cores
    std::vector<int> worker_cores;  ///< Ordered CPU IDs; on Windows, group * 64 + processor within the group
    int host_core = -1;             ///< Logical core reserved for host thread
    int host_sibling = -1;          ///< The host core's SMT sibling (-1: none), the logical processor `Sibling` takes
    /// For the default worker count (pool_plan.hpp): the CPU package (socket) of each worker core (parallel to
    /// `worker_cores`, -1: not known) and of the host core, the logical processors the process may use, whether the
    /// OS told physical cores from SMT siblings, and the container's CPU quota (0: none).  On a multi-socket CPU the
    /// host's package comes first in `worker_cores`.
    std::vector<int> worker_package;
    int host_package = -1;
    int logical = 0;
    bool layout_known = true;
    double cpu_quota = 0;
};

/// The description plan_pool_workers() takes, from a detected topology.
PoolPlanInput pool_plan_input(const CpuTopology& topo);
/// The CPUs a container quota allows (cgroup v2 cpu.max / v1 cfs quota), 0 when none or not Linux.
double container_cpu_quota();

/// Which core the host thread - the layer loop that spins on the GPU's flags - takes when `skip_first` reserves one
/// (--host-core first|last|sibling, STRATA_HOST_CORE).  `First` is the layout the pool has always used.  `Last`
/// (eddoursul's fork, F12) puts the host on the last physical core and lets the workers have the first: Windows sends
/// a GPU's interrupts to one logical processor, usually the first, and every copy that lands raises one (~13 us in the
/// ISR and a DPC), which delays a spinning host there.  The placement changes no result, only where threads run.  A
/// hybrid CPU keeps `First` under `Last` (its first core is the best P-core).  `Sibling` keeps the host on that first
/// core but on its other hardware thread: the interrupts stay on the first logical processor, no worker shares the
/// core (the workers take one logical processor per core), and it applies to hybrid CPUs too.  Without an SMT sibling
/// it is `First`.
enum class HostCore { First, Last, Sibling };
void set_host_core(HostCore where);
HostCore host_core_setting();

/// `--pool-cores LIST` / STRATA_POOL_CORES / the config's "pool_cores" (experimental, off by default): pin the pool's
/// workers to exactly these logical processors, in this order, instead of one per physical core.  An empty list (the
/// default) leaves the placement as it has always been.  Set before the pool is built.  Processors the process may not
/// run on are left out with a warning; SMT siblings of one core and the host's processor are kept with a warning (a
/// person who asks for them gets them).  The worker count is the list's length unless --pool-workers says otherwise.
void set_pool_cores(const std::vector<int>& cpus);
const std::vector<int>& pool_cores_setting();

/// Every logical processor this process may run on and the physical core it belongs to (see LogicalCpuInfo).  Empty when
/// the system does not say (macOS, a sandbox without the topology).
std::vector<LogicalCpuInfo> logical_cpu_layout();

CpuTopology detect_cpu_topology(bool skip_first, PoolAffinity affinity = PoolAffinity::All);

/// The logical processor the host thread is pinned to under the current setting (-1: no topology).
int planned_host_core(PoolAffinity affinity = PoolAffinity::All);

/// One logical processor per PHYSICAL core, so a worker is never scheduled onto an SMT sibling of another
/// worker.  On the 6-core/12-thread machine this project measures on, `hardware_concurrency()/2` workers on
/// logical processors 0..5 would put every worker on a sibling pair and halve the useful bandwidth - which is
/// exactly the kind of error that shows up as "the CPU path is slower than the model says" with no clue why.
///
/// `skip_first` drops the first core, which P2.S3 reserves for the host loop.
std::vector<int> physical_cores(bool skip_first, PoolAffinity affinity = PoolAffinity::All);

/// **THE RESERVATION IS A FICTION UNLESS THE HOST IS ACTUALLY PUT THERE.**
///
/// `physical_cores(true)` keeps the workers off the first physical core so that the host loop can spin on
/// `cudaEventQuery` without stealing a worker's cycles.  Nothing in the pool can enforce the other half of
/// that, so this is it: the host loop calls this on entry and restores on exit.
///
/// MEASURED, and this is why it exists: the pool runs at **36.32 GB/s on 5 workers with nothing else running**
/// - exactly 5/6 of L9's 44.14 on 6 - and at **26.9 GB/s inside the host loop**, where the unpinned spinning
/// host is free to land on a worker's core or its SMT sibling.  That 1.35x is not the kernel.
///
/// The previous host placement. `valid` is false when querying or setting placement failed.
/// Windows uses a reversible CPU Set selection, leaving hard affinity (including Windows 11's implicit
/// all-group eligibility) untouched. An empty selection restores inheritance from the process defaults.
/// Linux stores the complete, dynamically sized native CPU mask, including CPU IDs above 63.
struct ThreadAffinity {
#if defined(_WIN32)
    std::vector<unsigned long> cpu_sets;
#else
    std::vector<unsigned long> mask;
#endif
    bool valid = false;
};

/// Selects one encoded processor for the caller; Windows CPU Sets respect existing hard-affinity limits.
/// Restore on the same calling thread. Pool-owned workers use hard group affinity instead.
ThreadAffinity pin_current_thread(int core);
void restore_thread_affinity(const ThreadAffinity& previous);

/// STRATA_NUMA_SPLIT's decision for a pool (see `ExpertPool`'s NodeSync).  `1` splits on the real NUMA nodes of
/// the pinned threads (Linux, exactly 2 nodes); `test` splits by thread index (thread j -> slice j % 2) with no real
/// node, for the parity test.
struct NumaSplitPlan {
    bool on = false;                  ///< split enabled
    bool test = false;                ///< STRATA_NUMA_SPLIT=test: no real nodes, never places memory
    int nodes[2] = {-1, -1};          ///< real node id of slice 0 and 1 (ascending); -1 in test mode
    std::vector<int> cores[2];        ///< the cores the pool pinned for each slice (workers + host)
    std::string why;                  ///< reason when off ("" when on, or when the env var is unset)
};
/// The live pool's decision (set by the ExpertPool constructor, reset by its destructor); `on == false` when no
/// pool exists or the split is off.
const NumaSplitPlan& active_numa_split();

class ExpertPool {
public:
    /// `n_workers <= 0` means "every physical core except the first".  Workers are pinned to physical cores
    /// (minus core 0 by default) and each owns one `ExpertScratch`, so nothing in the token path allocates.
    ///
    /// **`host_works` PUTS THE HOST THREAD INTO THE DRAIN (R2.2's FIRST HALF).**
    ///
    /// The pool reserves core 0 for the host loop so the doorbell spin cannot steal a worker's cycles - but
    /// during `run()` the host does not spin, it waits, so core 0 is idle for the whole drain. Measured on the
    /// 6-core machine this project targets: the engine's pool drains at **33.7 GB/s** (663.6 MB of expert
    /// blobs in 19.71 ms/token) where the same kernel on 5 workers should reach 5/6 x 44.14 = 36.8 and the
    /// machine measures 44.14 GB/s on all six. So the sixth core is being paid for and not used.
    ///
    /// With `host_works`, `run()` claims jobs itself instead of spinning on `done_`, and the pool is six
    /// threads on six cores. `false` is the A/B arm and exists so the change is measurable rather than
    /// asserted - the counter it moves is `pool phases ... drain`, which is host-side and needs no profiler.
    explicit ExpertPool(int n_workers = 0, bool pin = true, bool host_works = true,
                        PoolAffinity affinity = PoolAffinity::All, int tasks = 0);
    /// Batched row phases: 0 keeps three tasks per participating thread; explicit counts are capped by rows.
    static constexpr int kMaxTasks = 4096;
    /// The watchdog's view of the pool (issue #31): the batch, the counters, every thread's state.
    void diag(std::FILE* f) const;
    ~ExpertPool();
    ExpertPool(const ExpertPool&) = delete;
    ExpertPool& operator=(const ExpertPool&) = delete;

    int workers() const { return n_; }
    /// What bounded the default worker count (pool_plan.hpp; "" when a count was configured), the physical cores
    /// and logical processors counted - for the start-up line.
    const char* plan_limit() const { return plan_limit_; }
    /// True when the workers sit where `--pool-cores` said (the list, minus what could not be used).
    bool explicit_cores() const { return explicit_cores_; }
    /// The start-up line about where the workers and the host run ("CPU workers on logical processors 2,4,...,30 (15
    /// workers, one per physical core; host on 0)"); "" without workers.
    std::string placement_line() const;
    int plan_cores() const { return plan_cores_; }
    int plan_logical() const { return plan_logical_; }
    /// STRATA_NUMA_SPLIT: 2 when the native row phases are split per NUMA node, 0 when off.
    int numa_slices() const { return numa_n_; }
    /// Accumulated drain time of slice `s` over the mode-5/6 phases, ms: publish to the slice's last `done`.
    double numa_slice_ms(int s) const { return s >= 0 && s < numa_n_ ? slice_ms_[s] : 0.0; }
    /// Whether the host thread also drains.  Reported at startup, because "the engine adapts to the machine it
    /// is on" is only true if the engine says which adaptation it took.
    bool host_works() const { return host_works_; }

    bool is_hybrid() const { return topo_.is_hybrid; }
    /// The CPUs worker i is pinned to, in worker order (STRATA_NUMA_SPLIT reorders them: interleave_packages).
    const std::vector<int>& worker_cores() const { return topo_.worker_cores; }
    int p_cores() const { return topo_.p_cores; }
    int p_threads() const { return topo_.p_threads; }
    int e_cores() const { return topo_.e_cores; }
    PoolAffinity affinity() const { return affinity_; }

    /// Publish `n` jobs, then block until every one has been claimed AND every worker has parked.
    /// `jobs` must outlive the call (it does, and the workers never touch it afterwards).
    void run(ExpertJob* jobs, int n);

    /// Plan v0.3 P4: the same outputs as `run`, bitwise, with every expert split by rows across all threads
    /// (gate/up rows, then the intermediate's quantization, then down rows).  With fewer experts than threads -
    /// the case once the VRAM tier takes half of them - `run` leaves cores idle and each expert streams at one
    /// core's bandwidth; this streams every expert at all of them.  At most `kMaxSplit` experts.
    void run_split(ExpertJob* jobs, int n);
    static constexpr int kMaxSplit = 16;
    /// Plan v0.3 P6: `run_split` for multi-token jobs (at most `kMaxSplitMulti`); the rows of each expert are
    /// read once for all of its tokens.
    void run_split_multi(ExpertJobMulti* jobs, int n);
    /// Plan v0.3 P6: the same for a native pack's layer (ggml-cpu arithmetic, `nact` activations).
    void run_split_multi_native(const NativeFmt& f, ExpertJobMulti* jobs, int n,
                                NativeArithmetic arithmetic = NativeArithmetic::Default);
    static constexpr int kMaxSplitMulti = 96;
    /// Latent MoE (edge0): one Q8_0 matvec of the CPU share's D or U stage - `w` is nrows rows of row_bytes,
    /// `act`/`out` are its nt activation rows (Q8_0 blocks, ncols wide) and output rows (nrows floats).
    struct LatentMatvec {
        const uint8_t* w = nullptr;
        size_t row_bytes = 0;
        int nrows = 0, ncols = 0, nt = 0;
        const void* const* act = nullptr;
        float* const* out = nullptr;
    };
    /// Latent MoE (edge0): every matvec's rows over the workers (mode 7: kq256_rows type 8); one call, one barrier.
    void run_latent_rows(LatentMatvec* mv, int n);
    /// run_split_multi's phases, accumulated ms: gate/up rows, the intermediate quantization, down rows.
    double ms_multi_gu = 0, ms_multi_q = 0, ms_multi_down = 0;
    int64_t multi_bytes = 0;

    /// Total `_mm_pause` iterations spent waiting, over all workers, is no longer counted - see the note on the
    /// atomics below.  It was a LOCKED read-modify-write in the park loop, so measuring the contention added to
    /// it.
    long long pauses() const { return 0; }

    /// **WHERE `run()` SPENDS ITS TIME, in milliseconds accumulated over its lifetime.**  Three phases per
    /// layer - wait for every worker to be parked, wait for the drain, wait for them to re-park - and until now
    /// all three were reported as one number.  Without the split there is no way to tell a pool that is slow at
    /// the WORK from one that is slow at the SYNCHRONISATION, and those need opposite fixes: the first is a
    /// kernel problem and the second is a barrier problem.
    ///
    /// Only the host thread touches these, in `run()`, so they need no atomics.
    void phase_ms(double& wait_park, double& drain, double& repark) const {
        wait_park = ms_wait_park_;
        drain = ms_drain_;
        repark = ms_repark_;
    }

    /// **A PARKED WORKER SPINS FOR THIS LONG, THEN SLEEPS.**  The park is a `_mm_pause` spin because a layer's
    /// batches are microseconds apart and a wake-up from the OS costs more than that.  But a spin that never ends
    /// keeps every worker's core at 100% while the engine waits for a request - issue #4, "CPU 50% even when
    /// doing nothing" (7 of the 5700X's 16 threads).  Between requests the workers block on `sleep_cv_`.  NOT only
    /// between requests: with most experts in VRAM (a 24 GB card) many layers have no CPU work, so the workers
    /// also sleep mid-request - which the claim protocol above must survive (issue #29).
    /// `STRATA_POOL_SPIN_US` overrides it (a test knob: a short spin makes the workers sleep constantly).
    static constexpr std::chrono::milliseconds kSpinBeforeSleep{20};
    /// A pool wait that sees no completion for this long is a bug; the engine stops with a message instead of
    /// spinning forever, and the server starts it again (issue #29).
    static constexpr std::chrono::seconds kStall{60};

private:
    int phase_tasks(int64_t rows) const;
    int tasks_ = 0;
    void worker(int id);
    void drain(int id, ExpertScratch& scratch, uint32_t epoch);
    void run_phase(int mode, int n_tasks);
    /// Claim the next job of batch `epoch`, or -1 (that batch is exhausted, or it is not the current one).
    int claim(uint32_t epoch);
    /// Publish the batch whose description the caller has just written: reset `done`, then `head`, then the epoch.
    uint32_t begin_batch(int n);
    /// The host's waits, bounded by `kStall`.
    void wait_parked(const char* what);
    double tail_fin_[64] = {};          ///< PROBE-TAIL: worker id -> its last-task finish (ms, epoch-tagged below)
    uint64_t tail_ep_[64] = {};         ///< PROBE-TAIL: worker id -> epoch the stamp belongs to
    void wait_done(int n);
    /// Bump `epoch_`, and wake the workers that went to sleep.  Every publish goes through here.
    void publish();

    int n_ = 0;
    bool explicit_cores_ = false;
    bool pinned_ = true;
    std::vector<LogicalCpuInfo> layout_;   // the logical processors and their cores, read when the pool was built (the caller may be pinned later)
    const char* plan_limit_ = "";
    int plan_cores_ = 0;
    int plan_logical_ = 0;
    bool host_works_ = true;
    ExpertJob* jobs_ = nullptr;
    int njobs_ = 0;
    /// The host's own scratch when `host_works_`.  A separate object rather than a share of `scratch_[i]`,
    /// because a worker may own any index and the two must not be able to collide.
    ExpertScratch host_scratch_;
    // `run()`'s three phases, accumulated.  Host-thread only; see `phase_ms`.
    double ms_wait_park_ = 0.0;
    double ms_drain_ = 0.0;
    double ms_repark_ = 0.0;
    // ---- EACH ATOMIC GETS ITS OWN CACHE LINE, AND THE SPIN COUNTER IS GONE.  (Review finding C3.)
    //
    // These were six adjacent atomics, which put `head_`, `done_`, `parked_` and `epoch_` on ONE cache line -
    // the four that workers and the host actually contend on, invalidating each other on every access.
    //
    // Worse, the parked spin did `pauses_.fetch_add(1)` on EVERY iteration: a LOCKED read-modify-write, five
    // workers against one line, at roughly one iteration per `_mm_pause`.  So the line the host must WRITE to
    // publish work (`epoch_`) and READ to confirm the workers are parked (`parked_`) was being hammered by the
    // very threads waiting for it.  That is contention the pool imposes on itself.
    //
    // The counter was diagnostic only - `pauses()` was read in one place, to print a number nothing branched on
    // - so it is deleted rather than amortised.  `alignas(64)` then stops the remaining four sharing.
    // issue #31 diagnostics: each worker's state (kParked, kSleeping, kBetween, or the job it runs) and the host's
    // (kIdle, kWaitParked, kWaitDone, or its job), printed by the serve watchdog through `diag`
    static constexpr int32_t kParked = -1, kSleeping = -2, kBetween = -3, kIdle = -10, kWaitParked = -11,
                             kWaitDone = -12;
    std::unique_ptr<std::atomic<int32_t>[]> wstate_;
    std::atomic<int32_t> hstate_{kIdle};
    std::atomic<int64_t> hstate_ms_{0};
    alignas(64) std::atomic<uint64_t> head_{0};   // epoch << 32 | njobs << 16 | next index (issue #29)
    alignas(64) std::atomic<uint32_t> done_{0};
    alignas(64) std::atomic<uint32_t> parked_{0};
    alignas(64) std::atomic<uint32_t> epoch_{0};
    alignas(64) std::atomic<bool> stop_{false};
    // The sleep after `kSpinBeforeSleep`.  `sleepers_` is how `publish` knows whether anyone needs waking, so the
    // token path pays one uncontended load per publish and never takes the mutex while the workers spin.
    alignas(64) std::atomic<uint32_t> sleepers_{0};
    std::mutex sleep_mu_;
    std::condition_variable sleep_cv_;
    std::chrono::microseconds spin_before_sleep_{kSpinBeforeSleep};
    std::vector<std::thread> threads_;
    std::vector<ExpertScratch> scratch_;   // one per worker: no allocation, no false sharing of the hot data
    // run_split state: mode 0 = whole experts, 1 = gate/up row parts, 2 = down row parts
    int mode_ = 0;
    int parts_a_ = 1, parts_b_ = 1;
    struct SplitBuf {
        alignas(64) float ff[FF];
        ActQ a2;
    };
    std::vector<SplitBuf> split_;
    // run_split_multi state: mode 3 = gate/up row parts, 4 = down row parts
    ExpertJobMulti* mjobs_ = nullptr;
    int64_t mrows_ = 0;     // rows of the current multi phase across all its experts (n * FF, then n * H)
    int mtasks_ = 1;        // equal row ranges the phase is cut into
    struct SplitBufMulti {
        alignas(64) float ff[MAXT][FF];
        ActQ a2[MAXT];
        alignas(64) uint8_t hq[MAXT][kNativeHBytes];   // plan v0.3 P6: native down activations
    };
    const NativeFmt* nfmt_ = nullptr;
    NativeArithmetic narithmetic_ = NativeArithmetic::Default;
    std::vector<SplitBufMulti> split_multi_;
    LatentMatvec* lmv_ = nullptr;   ///< mode 7: the latent D/U matvecs (edge0)
    int lmv_n_ = 0;
    // STRATA_NUMA_SPLIT=1 (opt-in, Linux, exactly 2 nodes): in the native row phases (modes 5/6) every expert's rows
    // are cut into one slice per NUMA node (slice k = rows [per*k/2, per*(k+1)/2) of the gate/up or down matrix, the
    // part the resident complement places on node k in step B), and each node's threads claim and count on their own
    // slice's counters; in every mode they park on their own `parked`.  Off (`numa_n_ == 0`, the decision is final in
    // the constructor): none of this is touched and the pool is as before.
    //
    // THE CLAIM PROTOCOL IS issue #29's, ONCE PER SLICE.  Each slice head packs `epoch | njobs | index` and is claimed
    // with the same CAS as `head_`.  A mode-5/6 batch stores every slice head with its epoch and leaves `head_` alone;
    // any other batch stores `head_` and leaves the slice heads alone.  So for batch `e` exactly one family of heads
    // carries `e`, and a stale head is always exhausted (`wait_done` saw every task of it done).  A worker therefore
    // never reads `mode_` to pick a counter: it tries its slice head, then `head_`, and the epoch tag makes the wrong
    // one fail - as it makes a late worker's claim fail.
    struct NodeSync {
        alignas(64) std::atomic<uint64_t> head{0};       // as `head_`, for this slice of the phase
        alignas(64) std::atomic<uint32_t> done{0};       // as `done_`
        alignas(64) std::atomic<uint32_t> parked{0};     // as `parked_`, this slice's workers only
        alignas(64) std::atomic<int64_t> drained_ns{0};  // steady_clock ns when `done` reached `tasks` (0: not yet)
        // the batch's description, written by the host before the head's release store, read after a claim
        alignas(64) int base = 0;   // the slice's first task index in the phase (task ids are global for `diag`)
        int tasks = 0;              // T: the slice's task count
        int h = 0;                  // rows of each expert in the slice
        int row0 = 0;               // the slice's first matrix row (per*k/2)
        int64_t rows = 0;           // R = experts * h
    };
    // CBA: the NodeSync blocks live wherever `new` puts them (the host's node); per-node pages if the remote home
    // of a line ever shows up in the counters.
    std::unique_ptr<NodeSync[]> nsync_;
    int numa_n_ = 0;                  // slices (2 when on), 0 = off
    std::vector<int> wslice_;         // each worker's slice
    int hslice_ = 0;                  // the host's slice (when it drains)
    std::vector<int> slice_threads_;  // threads draining each slice (the host counted in its own)
    int64_t pub_ns_ = 0;              // the current mode-5/6 phase's publish time (host only)
    double slice_ms_[2] = {0.0, 0.0}; // accumulated per-slice drain time (host only)
    bool numa_phase() const { return numa_n_ > 0 && mode_ >= 5; }
    /// The claim of `claim()` on slice `slice`'s head: `base + i`, or -1.
    int claim_slice(uint32_t epoch, int slice);
    std::atomic<uint32_t>& parked_of(int id) { return numa_n_ > 0 ? nsync_[(size_t) wslice_[(size_t) id]].parked : parked_; }
    uint32_t parked_count() const;
    /// Fill the slices' descriptions for a mode-5/6 phase of `nb` experts of `per` rows; returns the total tasks.
    int numa_plan_phase(int per, int nb);
    /// Per-slice counters for the stall messages and `diag` ("" when off).
    std::string numa_counters() const;
    PoolAffinity affinity_ = PoolAffinity::All;
    CpuTopology topo_;
};

}  // namespace strata::kernels::cpu
