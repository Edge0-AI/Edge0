// include/strata/program/pipe_probe.hpp - STRATA_P4_TIMING / STRATA_P4_TRACE: where the wall time of a pipelined batch window goes.
//
// Opt-in measurement for the layer-split batch loop (generate.cpp, `pump`).  Off, every hook is one predictable branch and
// reads no clock.  On:
//   STRATA_P4_TIMING=1   per stage: the window's time from launch to done, the cycle between launches, the gap between a
//                        window done and the next launch, the stage's busy share, the host launch cost; the pump's cost
//                        per tick, the verdict work per window, how many groups were in flight, the ticks a free stage had
//                        a group waiting or was starved.  Printed to stderr with the batch summary line.
//   STRATA_P4_TRACE=PATH one line per window event ("L stage time_us group" at a launch, "D stage time_us 0" when the stage
//                        is done), flushed on every line so a killed run keeps its trace.  "1" or an empty value writes
//                        ./strata-p4-trace.csv.
// Written for the four-card R9700 investigation (docs/BATCHING.md): it found that the stages run 4 of 4 concurrently in the
// decode phase and that the serial admissions were the idle time.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::program {

class PipeProbe {
public:
    using Clock = std::chrono::steady_clock;
    using Time = Clock::time_point;
    static constexpr int kMaxStages = 16;
    static constexpr int kMaxGroups = 64;

    PipeProbe(int n_stages, size_t n_groups)
        : n_(std::min(n_stages, kMaxStages)), n_groups_(n_groups) {
        timing_ = std::getenv("STRATA_P4_TIMING") != nullptr;
        const char* tr = std::getenv("STRATA_P4_TRACE");
        if (tr != nullptr) {
            const char* path = (tr[0] == '\0' || std::strcmp(tr, "1") == 0) ? "strata-p4-trace.csv" : tr;
            trace_ = std::fopen(path, "w");
            if (trace_ != nullptr) std::setvbuf(trace_, nullptr, _IONBF, 0);
            else std::fprintf(stderr, "strata p4: cannot write the trace %s\n", path);
        }
    }
    ~PipeProbe() { if (trace_ != nullptr) std::fclose(trace_); }
    PipeProbe(const PipeProbe&) = delete;
    PipeProbe& operator=(const PipeProbe&) = delete;

    bool on() const { return timing_ || trace_ != nullptr; }
    bool timing() const { return timing_; }
    Time now() const { return on() ? Clock::now() : Time{}; }

    /// The pump's cost: constructed at the top of `pump`, adds its life to the per-tick figure.
    struct PumpScope {
        PipeProbe* p = nullptr;
        Time t0{};
        explicit PumpScope(PipeProbe& probe) {
            ++probe.ticks_;
            if (probe.on()) { p = &probe; t0 = Clock::now(); }
        }
        ~PumpScope() { if (p != nullptr) p->pump_ms_ += ms(Clock::now() - t0); }
    };
    /// The verdict work of a window that reached the last stage.
    struct VerdictScope {
        PipeProbe* p = nullptr;
        Time t0{};
        explicit VerdictScope(PipeProbe& probe) {
            if (probe.timing_) { p = &probe; t0 = Clock::now(); ++probe.verdict_n_; }
        }
        ~VerdictScope() { if (p != nullptr) p->verdict_ms_ += ms(Clock::now() - t0); }
    };

    /// Per tick: stages with a window in flight, groups in flight.
    void busy(int stages, int groups) { busy_ += stages; inflight_ += groups; }
    /// A stage k >= 1 that is free: whether a group is waiting at it (else it is starved by the stage before).
    void free_stage(int k, bool group_waiting) {
        if (k < 0 || k >= n_) return;
        if (group_waiting) ++starve_idle_[k]; else ++starve_none_[k];
    }
    void span_start(Time t) { span_start_ = t; span_ms_ = 0; }
    void span_end(double wall_ms) { span_ms_ = wall_ms; }
    void polled(Time before) { if (on()) poll_ms_ += ms(Clock::now() - before); }
    /// Stage k reported its window done.
    void done(int k, int group) {
        if (!on() || k < 0 || k >= n_) return;
        const Time t = Clock::now();
        if (timing_ && launched_[k]) { win_ms_[k] += ms(t - at_[k]); ++win_n_[k]; }
        if (trace_ != nullptr) std::fprintf(trace_, "D %d %.3f 0\n", k, us(t));
        done_at_[k] = t;
        if (timing_ && group >= 0 && group < kMaxGroups && left_[group] != Time{}) { q_wait_ += ms(t - left_[group]); ++q_n_; }
    }
    /// Stage k launched `group`'s window; `before` is now() taken before the launch call.
    void launched(int k, int group, Time before) {
        if (!on() || k < 0 || k >= n_) return;
        const Time t = Clock::now();
        launch_ms_ += ms(t - before);
        at_[k] = t;
        launched_[k] = true;
        if (group >= 0 && group < kMaxGroups) left_[group] = t;
        if (trace_ != nullptr) std::fprintf(trace_, "L %d %.3f %d\n", k, us(t), group);
        if (timing_) {
            if (last_launch_[k] != Time{}) cycle_ms_[k] += ms(t - last_launch_[k]);
            if (done_at_[k] != Time{}) gap_ms_[k] += ms(t - done_at_[k]);
            last_launch_[k] = t;
        }
    }
    /// Print and reset (the batch summary line calls it when the loop goes idle).
    void flush() {
        if (!timing_) return;
        const double ticks = (double) std::max(1LL, ticks_);
        std::fprintf(stderr, "strata p4: pump %.3f ms/tick over %lld ticks; stage-busy %.2f of %d; group wait %.1f ms;\n",
                     pump_ms_ / ticks, ticks_, busy_ / ticks, n_, q_n_ ? q_wait_ / (double) q_n_ : 0.0);
        for (int k = 0; k < n_; ++k) {
            const double n = win_n_[k] ? (double) win_n_[k] : 1.0;
            std::fprintf(stderr, "strata p4:   s%d windows %lld  win %.2f ms  cycle %.2f ms  gap %.2f ms  busy %.0f%%  launch %.2f ms\n",
                         k, win_n_[k], win_ms_[k] / n, cycle_ms_[k] / n, gap_ms_[k] / n,
                         100.0 * win_ms_[k] / std::max(1e-9, span_ms_), launch_ms_ / n);
        }
        std::fprintf(stderr, "strata p4:   verdict work %.2f ms/window; inflight groups avg %.2f of %zu\n",
                     verdict_ms_ / std::max(1.0, (double) verdict_n_), inflight_ / ticks, n_groups_);
        for (int k = 1; k < n_; ++k)
            std::fprintf(stderr, "strata p4:   s%d free ticks: with a group waiting %lld, starved %lld\n", k, starve_idle_[k],
                         starve_none_[k]);
        pump_ms_ = poll_ms_ = launch_ms_ = verdict_ms_ = busy_ = inflight_ = q_wait_ = 0;
        ticks_ = verdict_n_ = q_n_ = 0;
        for (int k = 0; k < kMaxStages; ++k) {
            win_ms_[k] = cycle_ms_[k] = gap_ms_[k] = 0;
            win_n_[k] = starve_idle_[k] = starve_none_[k] = 0;
        }
        span_ms_ = 0;
    }

private:
    static double ms(Clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }
    static double us(Time t) { return std::chrono::duration<double, std::micro>(t.time_since_epoch()).count(); }

    int n_ = 0;
    size_t n_groups_ = 0;
    bool timing_ = false;
    std::FILE* trace_ = nullptr;
    double pump_ms_ = 0, poll_ms_ = 0, launch_ms_ = 0, verdict_ms_ = 0, busy_ = 0, inflight_ = 0, q_wait_ = 0, span_ms_ = 0;
    long long ticks_ = 0, verdict_n_ = 0, q_n_ = 0;
    double win_ms_[kMaxStages] = {}, cycle_ms_[kMaxStages] = {}, gap_ms_[kMaxStages] = {};
    long long win_n_[kMaxStages] = {}, starve_idle_[kMaxStages] = {}, starve_none_[kMaxStages] = {};
    Time at_[kMaxStages] = {}, last_launch_[kMaxStages] = {}, done_at_[kMaxStages] = {};
    bool launched_[kMaxStages] = {};
    Time left_[kMaxGroups] = {};
    Time span_start_{};
};

}  // namespace strata::program
