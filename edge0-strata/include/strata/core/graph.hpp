// include/strata/core/graph.hpp - P2.S5: the GraphRegistry.
//
// `GraphRegistry`: capture/replay wrappers keyed by `(layer_type, n_tokens)`, with static input/output
// buffers at fixed addresses.
//
// THREE THINGS ARE BAKED INTO THIS INTERFACE, and each one is a measurement rather than a preference.  All
// three come from `bench/micro/graph_capture.cu`, which was written to answer the phase's own instruction -
// "verify on the installed CUDA version with a minimal test before building on it".
//
// 1. **THE WAIT MUST POLL THE DRIVER, NOT MEMORY.**  The phase's host loop says "spin on
//    doorbell_seq == expected ... no cudaStreamSynchronize anywhere".  A spin that only reads memory NEVER
//    RUNS THE KERNEL: 5,907,703 spins over 500 ms, and the datum flips to 1 the instant anything calls into
//    the driver and never before.  On Windows the display driver model BATCHES command submission, and a
//    thread that only reads memory gives it no reason to flush.  Hence `wait_ms`, which polls an event -
//    a QUERY and not a blocking sync, so P2.X3's "zero synchronization calls in the layer loop" still holds,
//    and it costs 0.021 ms.
//
// 2. **A GRAPH RE-READS ITS INPUT BUFFERS AT REPLAY, BUT NOT ITS KERNEL ARGUMENTS.**  Verified: eight
//    replays with a different pinned input each time all tracked correctly, so the data path is live; but a
//    kernel argument is copied into the node at capture and is never re-read.  **Anything that changes per
//    step - a position, a page-table base, a token id - must be DATA in a device buffer, never an argument.**
//    This is the phase's "graph capture with pointers that change between steps silently replays stale
//    addresses" hazard, and it now has a measurement behind it.
//
// 3. **FIXED ADDRESSES ARE THE CALLER'S OBLIGATION.**  The registry cannot check them.  Every buffer a
//    captured body touches must be allocated before `record` and never reallocated; the registry holds the
//    graph, not the memory.
//
// What the registry is FOR: a layer's ops are the same every token, so recording them once and replaying
// removes the per-call host cost of submitting them individually.  Measured, that cost is 10-22 us PER CALL
// into the driver on this machine, and a layer with 8 submissions is ~0.1 ms of round trip.
#pragma once

#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace strata::core {

/// The two attention families.  `docs/semantics.md`: 36 GDN layers and 12 QSA, at indices 3, 7, ... 47.
enum class LayerType : int { GDN = 0, QSA = 1 };
inline const char* to_string(LayerType t) { return t == LayerType::GDN ? "GDN" : "QSA"; }

/// One recorded graph.  Move-only: it owns a `cudaGraphExec_t` and an event, and two owners would
/// double-destroy.
class CapturedGraph {
public:
    CapturedGraph() = default;
    ~CapturedGraph() { reset(); }
    CapturedGraph(const CapturedGraph&) = delete;
    CapturedGraph& operator=(const CapturedGraph&) = delete;
    CapturedGraph(CapturedGraph&& o) noexcept { *this = std::move(o); }
    CapturedGraph& operator=(CapturedGraph&& o) noexcept;

    /// Start recording everything the caller launches into `stream`.
    bool begin(void* stream, std::string& err);

    /// Stop recording, instantiate, and record the completion event.  A capture that produced ZERO nodes is
    /// reported as an error: it means the body did nothing, which is a wiring mistake and NOT an empty graph
    /// to be replayed happily forever.
    bool end(void* stream, std::string& err);

    /// Replay.  Does not synchronise.
    bool launch(void* stream, std::string& err) const;

    /// The host loop's wait: poll `done_` until it completes or `timeout_ms` elapses.  Returns false on
    /// timeout, which the caller must treat as a hang and not as "probably fine".  See NOTE 1 above for why
    /// this exists instead of a memory spin.
    bool wait_ms(int timeout_ms) const;

    size_t nodes() const { return nodes_; }
    bool valid() const { return exec_ != nullptr; }

private:
    void reset();
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t exec_ = nullptr;
    cudaEvent_t done_ = nullptr;
    size_t nodes_ = 0;
};

/// Capture-once, replay-many, keyed by `(layer_type, n_tokens)`.
///
/// `record` is given a CALLABLE that launches the layer's ops.  On the first call for a key it captures;
/// afterwards it returns the cached graph and does NOT run the callable - so a caller that mutates state
/// inside the body will be surprised, which is why the body must only launch.
class GraphRegistry {
public:
    explicit GraphRegistry(void* stream) : stream_(stream) {}

    /// Capture `body` for `key` if it is not already recorded.  Returns false and fills `err` on failure.
    /// Recording the same key twice is a no-op and is NOT an error - that is the point of a registry, and a
    /// regression test asserts that the body runs exactly once across many calls.
    bool record(LayerType type, int n_tokens, const std::function<void()>& body, std::string& err);

    const CapturedGraph* find(LayerType type, int n_tokens) const;

    /// Launch and wait, in that order, the two things the host loop does per layer.
    bool launch(LayerType type, int n_tokens, int timeout_ms, std::string& err) const;

    size_t size() const { return graphs_.size(); }
    /// How many times a BODY was actually captured.  With a working cache this equals `size()`.
    size_t captures() const { return captures_; }

private:
    using Key = std::pair<int, int>;
    void* stream_ = nullptr;
    std::map<Key, CapturedGraph> graphs_;
    size_t captures_ = 0;
};

}  // namespace strata::core
