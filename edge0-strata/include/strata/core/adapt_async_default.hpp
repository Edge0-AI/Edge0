// include/strata/core/adapt_async_default.hpp - the asynchronous adaptive tier (--adapt-async 1) as a measured, per-session choice.
//
// Measured: on an RTX PRO 5000 box (64 cores) with every expert in the RAM arena, `--adapt-async 1` gave decode story +5.6%,
// code +5.1%, doc8k +10.9% (5/5 pairs faster in every class; the tier's wait per verify window fell from 1.4 ms to 0.14 ms).
// On a CPU-bound box (RTX A4000, Xeon E5-2643 v3, 12 threads, CPU experts 35 of 55 ms per window) the same switch gave
// story 0.99, code 1.00, doc8k 1.04 and a 4K-prompt decode 0.945: the helper thread competes with the expert pool.  So it
// is neither always on nor always off: where the engine could run it (CUDA, `--serve`, one GPU, the expert arena, no batch
// slots, no layer split) it starts on the blocking tier, measures ~64 windows, and only when the tier's wait is a large part
// of a window does it try the asynchronous tier for ~64 windows and go back to the blocking tier if the decode got slower by more than 2%.
// Off switches: `--adapt-async 0`, `STRATA_ADAPT_ASYNC=0`.  `--adapt-async 1` keeps its old meaning (always on).
// Not bit-exact from run to run, like the adaptive tier itself.  The identity gates run with `--adapt-every 100000` (no
// adaptive tier), so they never reach this.
#pragma once

#include <cstdlib>

namespace strata::core {

struct AdaptAsyncAuto {
    int flag = -1;               ///< --adapt-async on the command line: -1 not given, else 0 / 1
    int env = -1;                ///< STRATA_ADAPT_ASYNC: -1 unset, else its value (0 = off)
    bool cuda = false;           ///< a CUDA build (the HIP builds keep it opt-in)
    bool serve = false;
    bool adaptive = false;       ///< --adapt-every and --adapt-swaps both above 0
    bool arena = false;          ///< the experts come from the RAM arena (every expert in RAM)
    bool multi_gpu = false;      ///< a layer split
    bool pipeline_windows = false;
    bool batch = false;          ///< --batch slots
    bool peer = false;           ///< --peer-device
    bool remote_opt = false;     ///< --remote-expert-opt
    bool adapt_overlap = false;  ///< the opt-in STRATA_ADAPT_OVERLAP tier, which excludes the asynchronous one
};

/// STRATA_ADAPT_ASYNC as the process sees it (-1 unset).
inline int adapt_async_env() {
    const char* v = std::getenv("STRATA_ADAPT_ASYNC");
    return v == nullptr || v[0] == '\0' ? -1 : std::atoi(v);
}

/// true: the engine may choose the asynchronous tier by measurement.  `why_not` (may be null) gets a short reason otherwise.
inline bool adapt_async_self_decides(const AdaptAsyncAuto& a, const char** why_not = nullptr) {
    auto no = [&](const char* why) { if (why_not != nullptr) *why_not = why; return false; };
    if (a.flag >= 0) return no("--adapt-async was given");          // the user's choice, 0 or 1, is applied as before
    if (a.env == 0) return no("STRATA_ADAPT_ASYNC=0");
    if (!a.cuda) return no("not a CUDA build");
    if (!a.serve) return no("it needs --serve");
    if (!a.adaptive) return no("the adaptive tier is off");
    if (!a.arena) return no("not the expert arena");
    if (a.multi_gpu) return no("a layer split");
    if (a.pipeline_windows) return no("--pipeline-windows");
    if (a.batch) return no("--batch slots");
    if (a.peer) return no("--peer-device");
    if (a.remote_opt) return no("--remote-expert-opt");
    if (a.adapt_overlap) return no("STRATA_ADAPT_OVERLAP");
    return true;
}

/// The measure -> trial -> keep or go back decision.  Feed it one finished decode window at a time (wall ms of the window,
/// ms of it the adaptive tier held it up, tokens it produced); it answers what to do before the next window.
class AdaptAsyncSelf {
public:
    enum class Phase { Off, Measure, Trial, Done };
    enum class Action { None, ToAsync, ToBlocking };
    struct Config {
        int warmup_windows = 64;       ///< steady windows ignored first: the swap tier, the cache and the drafter are still settling after a start
        int measure_windows = 64;      ///< windows measured on the blocking tier
        int trial_windows = 64;        ///< windows measured on the asynchronous tier (after trial_skip warm-up windows)
        int trial_skip = 4;
        double wait_frac_min = 0.05;   ///< try the asynchronous tier when the wait is at least this part of a window
        double gain_min = 0.98;        ///< keep it unless ms per token got worse by more than 1/gain_min (a guard against harm: 64 windows of text cannot resolve a 3-5% gain)
    };
    AdaptAsyncSelf() = default;
    explicit AdaptAsyncSelf(const Config& c) : c_(c) {}

    void arm() { phase_ = Phase::Measure; reset_sums(); }
    Phase phase() const { return phase_; }
    bool trialing() const { return phase_ == Phase::Trial; }
    bool kept() const { return phase_ == Phase::Done && kept_; }

    /// One decode window.  Windows with no token (or a non-positive time) are ignored.
    Action window(double window_ms, double wait_ms, int tokens) {
        if (tokens <= 0 || !(window_ms > 0.0)) return Action::None;
        if (phase_ == Phase::Measure) {
            if (warmed_ < c_.warmup_windows) { ++warmed_; return Action::None; }
            ms_ += window_ms; wait_ += wait_ms; tok_ += tokens; ++n_;
            if (n_ < c_.measure_windows) return Action::None;
            measure_ms_per_token_ = ms_ / (double) tok_;
            wait_frac_ = ms_ > 0.0 ? wait_ / ms_ : 0.0;
            reset_sums();
            if (wait_frac_ >= c_.wait_frac_min) { phase_ = Phase::Trial; skipped_ = 0; return Action::ToAsync; }
            phase_ = Phase::Done; kept_ = false;                    // the wait is small: nothing to win
            return Action::None;
        }
        if (phase_ == Phase::Trial) {
            if (skipped_ < c_.trial_skip) { ++skipped_; return Action::None; }
            ms_ += window_ms; tok_ += tokens; ++n_;
            if (n_ < c_.trial_windows) return Action::None;
            trial_ms_per_token_ = ms_ / (double) tok_;
            phase_ = Phase::Done;
            kept_ = trial_ms_per_token_ * c_.gain_min <= measure_ms_per_token_;
            return kept_ ? Action::None : Action::ToBlocking;
        }
        return Action::None;
    }

    double wait_fraction() const { return wait_frac_; }
    double measure_ms_per_token() const { return measure_ms_per_token_; }
    double trial_ms_per_token() const { return trial_ms_per_token_; }

private:
    void reset_sums() { ms_ = wait_ = 0.0; tok_ = 0; n_ = 0; }
    Config c_{};
    Phase phase_ = Phase::Off;
    double ms_ = 0.0, wait_ = 0.0;
    long long tok_ = 0;
    int n_ = 0, skipped_ = 0, warmed_ = 0;
    double wait_frac_ = 0.0, measure_ms_per_token_ = 0.0, trial_ms_per_token_ = 0.0;
    bool kept_ = false;
};

}  // namespace strata::core
