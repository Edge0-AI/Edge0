#pragma once
// Foresight swap space (opt-in: STRATA_FS_SLOTS > 0; unset = nothing here exists and the engine is unchanged).
//
// A few VRAM slots per layer on each card, apart from the expert cache, refilled on their own copy stream with experts
// that just missed.  The verify pool asks, for every expert it would otherwise give the CPU, whether that expert sits
// in a slot whose copy has landed; if so the expert becomes a plain GPU group (kind 0) pointing at the slot, exactly
// like a cache hit - the grouped kernel reads whatever address the plan carries, so no kernel changes.  The host pool
// decides and the GPU follows the plan, so both always agree on who computes an expert.
//
// Safety: a slot is refilled only when every window that may have read it has finished: each plan reference stamps
// the slot with the count of finished verify windows, and a slot is reusable once `depth` more windows finished
// (STRATA_FS_DEPTH, default 4; --pipeline-windows 2 keeps at most two windows in flight).  A copy that has not landed
// is simply not used (the CPU computes the expert as before).  Not with STRATA_VERIFY_DEVICE_PLAN (the device would
// plan resident layers without the host).
//
// With STRATA_FS_AHEAD=1 (default 1 when the swap space is on) the router look-ahead (RouterLookahead, one layer ahead on the
// previous layer's MoE input) also feeds the slots, not only the page warming of the file tier.
//
// Switches: STRATA_FS_SLOTS (per layer and card), STRATA_FS_BUDGET (copies per card per finished window, default 32),
// STRATA_FS_ADMIT (misses within the last 4 windows before an expert is copied, default 1), STRATA_FS_DEPTH,
// STRATA_FS_VERIFY=1 (test mode: every slot is read back and compared with its source the first time it is used).
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::core {

class ExpertSource;

struct ForesightSwap {
    struct Slot {
        int32_t expert = -1;
        bool ready = false;                  ///< the copy has landed (dispatch thread only)
        bool loading = false;                ///< a copy is assigned and not yet seen landed (dispatch thread only)
        std::atomic<bool> issued{false};     ///< the filler has queued the copy and recorded `ev`
        uint64_t last_ref = 0;               ///< finished-window count when a plan last pointed here
        unsigned long long dptr = 0;         ///< the slot's device address
        const uint8_t* src = nullptr;        ///< the host blob being copied in
        uint64_t bytes = 0;
        void* ev = nullptr;                  ///< cudaEvent_t on the layer's card
    };

    int slots = 0, budget = 32, admit = 1, depth = 4;
    int64_t n_layers = 0, n_expert = 0;
    std::unique_ptr<Slot[]> slot;            ///< n_layers x slots
    std::vector<int16_t> where;              ///< (layer, expert) -> slot in its layer, -1
    std::vector<uint8_t> miss_cnt;           ///< STRATA_FS_ADMIT > 1: recent misses per (layer, expert)
    std::vector<uint64_t> miss_at;
    std::vector<int> layer_card, layer_dev;
    std::vector<void*> card_stream;          ///< cudaStream_t per card
    std::vector<void*> card_mem;             ///< the slots' allocation per card
    std::vector<int> card_dev;
    std::vector<uint64_t> card_round;
    std::vector<int> card_used;
    std::atomic<uint64_t> completed{0};      ///< finished verify windows (the serve loops count them)
    std::atomic<bool> failed{false};
    // counters (dispatch thread)
    uint64_t hits = 0, copies = 0, busy = 0, over_budget = 0, pending = 0, predicted = 0;   // (under smu_)
    bool verify = false;                     ///< STRATA_FS_VERIFY=1: read every landed slot back and compare (slow, tests)
    uint64_t verify_ok = 0, verify_bad = 0;
    uint64_t dropped = 0;                    ///< slots dropped because an exchange moved their expert's bytes (under smu_)
    static constexpr int16_t kHeld = -2;     ///< `where`: an exchange is moving this expert's bytes, no copy now (#2134)
    static constexpr int32_t kDropped = -2;  ///< Slot::expert: dropped by `hold`, reusable under the same depth rule

    /// Allocates the slots on each card and starts the filler.  `card_of_layer[l]` indexes `devices` (CUDA ids).
    bool init(int slots_per_layer, int64_t layers, int64_t experts, const std::vector<int>& card_of_layer,
              const std::vector<int>& devices, std::string& err);
    ~ForesightSwap();

    /// The pool, for a routed expert that is not in the cache: true and its device address when a landed slot holds it.
    bool take(int64_t layer, int32_t expert, unsigned long long& ptr);
    /// The pool, for an expert the CPU computes now: maybe copy it into a slot of its layer for the next windows.
    void note_miss(ExpertSource* src, int64_t layer, int32_t expert);
    /// The router look-ahead (any thread), for an expert the next layer's router predicts and no cache holds: maybe copy
    /// it into a slot of that layer now, so that the copy has landed when the layer's plan is made.  One prefetch design
    /// with note_miss (the same slots, budget and reuse rule); a prediction only ever moves bytes, the plan is still made
    /// by the host pool and the model's own router stays authoritative.
    void predict(ExpertSource* src, int64_t layer, int32_t expert);
    /// #2134: the adaptive tier moves this expert's bytes in the RAM copy (an exchange, its two halves or a rotation),
    /// so a copy issued from its old place may land holding another expert's bytes.  Its slot is dropped (never served;
    /// refilled only once `depth` windows finished, like any slot a window may still read; a copy in flight finishes
    /// first) and no copy of it is issued until `unhold`, when its bytes sit at their new place.  Any thread.
    void hold(int64_t layer, int32_t expert);
    void unhold(int64_t layer, int32_t expert);
    std::string report() const;

private:
    void request_locked(ExpertSource* src, int64_t layer, int32_t expert, bool predicted);
    std::mutex smu_;                         ///< slot, where, card_* and the counters: the pool thread and the look-ahead thread
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<int64_t> queue_;              ///< slot indices whose copy the filler issues
    bool stop_ = false;
    std::thread filler_;
    void fill_loop();
};

ForesightSwap* foresight_swap_from_env(int64_t layers, int64_t experts, const std::vector<int>& card_of_layer,
                                       const std::vector<int>& devices);

}  // namespace strata::core
