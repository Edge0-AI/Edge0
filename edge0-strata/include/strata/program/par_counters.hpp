// include/strata/program/par_counters.hpp - STRATA_PAR_COUNTERS=1: counters-only parallelism accounting (research, opt-in).
//
// Off (the default), every hook is one branch on a const bool and reads no clock.  On, the batch loop (one host thread)
// adds plain integers and, at the places that already wait (a drain, an admission), two clock reads.  Nothing here
// synchronizes a device.  Printed to stderr as "strata par: ..." with the batch summary line (when the loop goes idle):
//   windows      group windows launched, rows carried, rows used by live slots, padded rows, histogram of live slots per window
//   drains       pipeline drains before a control line, by line kind (BGEN = an admission), groups in flight, ms waited
//   admissions   prompts admitted into a slot: prompt tokens, ms in the prompt phase, ms of the drain before it
//   idle         ticks with stage 0 free and no launchable group (no live slot / every group in flight)
// Stage busy/idle with reason stays in STRATA_P4_TIMING (pipe_probe.hpp); this adds what it lacked.
#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace strata::program {

/// Expert-dispatch totals (cumulative in the engine); a span's delta is printed at flush.
struct ParEx { long long misses = 0, entries = 0, hits = 0, offload = 0, pcie = 0; double run_ms = 0, jobs_ms = 0, vwait_ms = 0, vpool_ms = 0; };

class ParCounters {
public:
    using Ex = ParEx;
    using Clock = std::chrono::steady_clock;
    ParCounters() { const char* e = std::getenv("STRATA_PAR_COUNTERS"); on_ = e != nullptr && e[0] == '1' && e[1] == '\0'; }
    bool on() const { return on_; }

    void window(int rows, int live) {
        if (!on_) return;
        ++windows_; rows_ += rows; used_ += live;
        ++hist_[live < 0 ? 0 : (live > kHist - 1 ? kHist - 1 : live)];
    }
    void drain(const std::string& line, int groups_in_flight, double ms) {
        if (!on_) return;
        const bool adm = line.rfind("BGEN", 0) == 0;
        if (adm) { ++adm_drains_; adm_drain_ms_ += ms; } else { ++other_drains_; other_drain_ms_ += ms; }
        if (groups_in_flight > 0) ++drains_busy_;
        groups_at_drain_ += groups_in_flight;
    }
    void admission(long long prompt_tokens, double prompt_ms) {
        if (!on_) return;
        ++adm_; adm_tokens_ += prompt_tokens; adm_ms_ += prompt_ms;
    }
    void begin(const Ex& e, int layers) { if (on_) { ex0_ = e; layers_ = layers > 0 ? layers : 1; } }
    void adapt(double ms) { if (on_) { ++adapt_; adapt_ms_ += ms; } }
    void no_launch(bool any_live) { if (on_) { if (any_live) ++idle_all_inflight_; else ++idle_no_live_; } }

    void flush(double wall_ms, const ParEx& e = ParEx{}) {
        if (!on_ || (windows_ == 0 && adm_ == 0 && adm_drains_ + other_drains_ == 0)) return;
        std::fprintf(stderr, "strata par: wall %.0f ms; windows %lld rows %lld used %lld padded %lld (%.1f%% used); live slots/window:",
                     wall_ms, windows_, rows_, used_, rows_ - used_, rows_ ? 100.0 * used_ / rows_ : 0.0);
        for (int i = 0; i < kHist; ++i) if (hist_[i]) std::fprintf(stderr, " %d:%lld", i, hist_[i]);
        std::fprintf(stderr, "; drains: admission %lld (%.0f ms) other %lld (%.0f ms), %lld with groups in flight, avg %.2f groups; "
                             "admissions %lld, %lld prompt tokens in %.0f ms; adapt rounds %lld %.0f ms; "
                             "ticks without a launch: no live slot %lld, all groups in flight %lld\n",
                     adm_drains_, adm_drain_ms_, other_drains_, other_drain_ms_, drains_busy_,
                     (adm_drains_ + other_drains_) ? (double) groups_at_drain_ / (adm_drains_ + other_drains_) : 0.0, adm_, adm_tokens_, adm_ms_,
                     adapt_, adapt_ms_, idle_no_live_, idle_all_inflight_);
        if (windows_ > 0 && (e.misses | e.entries | e.hits | e.offload) != 0) {
            const double wl = (double) windows_ * layers_;
            std::fprintf(stderr, "strata par experts: per window-layer: routed entries (CPU) %.2f, distinct CPU-computed experts %.2f, "
                                 "GPU cache hits %.2f, PCIe/offload entries %.2f, distinct PCIe experts %.2f; ms per window: CPU pool run %.2f, "
                                 "job build %.2f, verifier wait %.2f, verifier pool %.2f (of %.2f wall)\n",
                         (e.entries - ex0_.entries) / wl, (e.misses - ex0_.misses) / wl, (e.hits - ex0_.hits) / wl,
                         (e.offload - ex0_.offload) / wl, (e.pcie - ex0_.pcie) / wl, (e.run_ms - ex0_.run_ms) / windows_,
                         (e.jobs_ms - ex0_.jobs_ms) / windows_, (e.vwait_ms - ex0_.vwait_ms) / windows_,
                         (e.vpool_ms - ex0_.vpool_ms) / windows_, wall_ms / windows_);
        }
        std::fflush(stderr);
        reset();
    }

private:
    static constexpr int kHist = 18;
    void reset() {
        windows_ = rows_ = used_ = adm_drains_ = other_drains_ = drains_busy_ = groups_at_drain_ = adm_ = adm_tokens_ = adapt_ = 0;
        idle_no_live_ = idle_all_inflight_ = 0;
        adm_drain_ms_ = other_drain_ms_ = adm_ms_ = adapt_ms_ = 0;
        std::memset(hist_, 0, sizeof hist_);
    }
    bool on_ = false;
    Ex ex0_{};
    int layers_ = 1;
    long long windows_ = 0, rows_ = 0, used_ = 0, adm_drains_ = 0, other_drains_ = 0, drains_busy_ = 0, groups_at_drain_ = 0;
    long long adm_ = 0, adm_tokens_ = 0, adapt_ = 0, idle_no_live_ = 0, idle_all_inflight_ = 0, hist_[kHist] = {};
    double adm_drain_ms_ = 0, other_drain_ms_ = 0, adm_ms_ = 0, adapt_ms_ = 0;
};

}  // namespace strata::program
