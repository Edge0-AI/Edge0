// include/strata/spec/controller.hpp - plan v0.3 P6: which drafter, and how many draft tokens, this step.
//
// Each step the controller maximizes  E[tokens committed] / T(step)  over the choices
//     none (k = 0) | suffix lookup with k <= its proposal | MTP with k <= K_MAX
// using running estimates of per-position acceptance and a cost model of a (k+1)-token verify pass. It keeps
// k = 0 unless the best choice beats plain decoding by `min_gain` (plan: 5%).
//
// Acceptance: for MTP, position i's conditional acceptance p_i (EMA). For lookup, one conditional acceptance per
// match-length bucket (a long match is far more reliable than a trigram), also an EMA. `observe` updates them
// from how many drafted tokens the verify pass accepted.
//
// Cost: T(n tokens) = dense(n) + experts(n) + sync + draft. The defaults are the 23 Sep measurements:
// dense ratio per token count from native MMVQ (bench/results/2026-09-23-mmvq-multi), distinct experts per window
// from the routing trace (2026-09-23-spec-economics), extra-token CPU cost 0.2-0.35 (2026-09-23-cpu-expert-multi).
// Replace them with in-engine measurements through `CostModel`.
#pragma once

#include <array>
#include <cstdint>

namespace strata::spec {

inline constexpr int K_MAX = 8;

struct CostModel {
    double dense_ms = 11.0;                                  // one-token dense pass (plan P3 target)
    std::array<double, K_MAX + 1> dense_ratio{1.0, 1.05, 1.3, 1.45, 1.85, 2.2, 2.6, 3.0, 3.3};   // by n = k+1 (index n-1)
    double cpu_all_miss_ms = 15.8;                           // 480 experts on the CPU
    double hit_rate = 0.55;                                  // share of distinct experts served from VRAM
    std::array<double, K_MAX + 1> distinct_ratio{1.0, 1.70, 2.31, 2.88, 3.40, 3.89, 4.35, 4.80, 5.2};  // U(n)/U(1)
    double extra_use_cost = 0.25;                            // each extra token on an expert, fraction of a read
    double sync_ms = 2.4;
    double mtp_draft_ms = 1.2;                               // per MTP draft token
    double lookup_draft_ms = 0.01;

    /// Step time for verifying n = k + 1 tokens, plus drafting k tokens with the given source.
    double step_ms(int k, bool mtp) const;
};

enum class Source { None, Lookup, Mtp };

struct Choice {
    Source source = Source::None;
    int k = 0;
    double expected_tokens = 1.0;
    double tokens_per_ms = 0.0;
};

class Controller {
public:
    explicit Controller(CostModel cost = {}, double min_gain = 0.05, double ema = 0.05);

    /// `lookup_available` tokens the suffix drafter can propose now, from a match of `lookup_match` tokens.
    Choice choose(int lookup_available, int lookup_match, bool mtp_ready) const;

    /// After verification: `accepted` of the `k` drafted tokens matched (a prefix).
    void observe(const Choice& c, int accepted, int lookup_match);

    double mtp_accept(int position) const { return mtp_p_[position]; }     // position 0 = first draft token
    double lookup_accept(int match) const { return lookup_q_[bucket(match)]; }
    const CostModel& cost() const { return cost_; }

private:
    static int bucket(int match) { return match >= 16 ? 3 : match >= 8 ? 2 : match >= 5 ? 1 : 0; }
    static double expected(const double* p, int k, bool conditional_same);

    CostModel cost_;
    double min_gain_, ema_;
    std::array<double, K_MAX> mtp_p_{};
    std::array<double, 4> lookup_q_{};
};

}  // namespace strata::spec
