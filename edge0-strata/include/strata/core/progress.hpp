// include/strata/core/progress.hpp - issue #29: whether a request is still moving, and where it is.
//
// The engine's host loop and the GPU wait on each other through flags; a protocol bug there does not crash, it
// spins forever (the GPU "100%", one CPU core busy, no tokens).  `--serve` runs a watchdog thread over this: a
// request whose heartbeat stops for `STRATA_WATCHDOG_S` seconds (default 60; 0 = off) ends the engine with the
// stage it was stuck in, and the server starts it again instead of hanging.  Tokens, prompt chunks and verify
// windows beat; the stage is two relaxed stores per layer, which the token path does not notice.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>

namespace strata::core {

struct Progress {
    std::atomic<uint64_t> beats{0};
    std::atomic<bool> busy{false};
    std::atomic<const char*> where{"idle"};
    std::atomic<int64_t> detail{-1};
    std::atomic<int64_t> chunk{-1};      ///< #251: the prompt chunk (its first position) a batched-read stage is in
    std::atomic<int64_t> since_ms{0};    ///< when `where` was set (steady clock): how long a stage has lasted
    std::atomic<uint64_t> ticks{0};      ///< layers served: a window that still moves, slowly, against one that stopped
    /// a step that blocks in one call (a file flush, a host->device transfer of a whole session) is given until this
    /// time (steady ms; 0 = none) instead of the watchdog's limit - an explicit, bounded allowance, not a beat
    std::atomic<int64_t> allow_until_ms{0};
};

inline int64_t progress_now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

/// What the watchdog prints about a part of the engine when it fires (the expert pool, the verify window).
using DiagFn = void (*)(std::FILE*);
inline std::atomic<DiagFn>& diag_pool_fn() { static std::atomic<DiagFn> f{nullptr}; return f; }
inline std::atomic<DiagFn>& diag_verify_fn() { static std::atomic<DiagFn> f{nullptr}; return f; }
/// #267: what a path that ends the engine runs first - it releases the GPU's spin waits on host flags (the verify
/// window's), so no kernel stays resident while the process goes away (on Windows that left the GPU "lost").
inline std::atomic<DiagFn>& release_gpu_fn() { static std::atomic<DiagFn> f{nullptr}; return f; }
inline void release_gpu_waits(std::FILE* f) {
    if (auto fn = release_gpu_fn().load()) fn(f);
}
/// --pipeline-windows: the pipelined loop's windows and the drafter's chain (set while that loop runs)
inline std::atomic<DiagFn>& diag_pipeline_fn() { static std::atomic<DiagFn> f{nullptr}; return f; }

inline Progress& progress() {
    static Progress p;
    return p;
}

inline void progress_at(const char* where, int64_t detail = -1, int64_t chunk = -1) {
    Progress& p = progress();
    p.where.store(where, std::memory_order_relaxed);
    p.detail.store(detail, std::memory_order_relaxed);
    p.chunk.store(chunk, std::memory_order_relaxed);
    p.since_ms.store(progress_now_ms(), std::memory_order_relaxed);
}

inline void progress_beat() { progress().beats.fetch_add(1, std::memory_order_relaxed); }
inline void progress_tick() { progress().ticks.fetch_add(1, std::memory_order_relaxed); }
/// A beat that also sets (seconds > 0) or clears (0) the allowance of a blocking step: the watchdog does not fire
/// before the allowance ends, and fires at its end if nothing beat since.
inline void progress_allow(int64_t seconds) {
    progress().allow_until_ms.store(seconds > 0 ? progress_now_ms() + seconds * 1000 : 0, std::memory_order_relaxed);
    progress_beat();
}

}  // namespace strata::core
