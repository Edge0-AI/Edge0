// src/spec/controller.cpp - see include/strata/spec/controller.hpp.
#include "strata/spec/controller.hpp"

#include <algorithm>

namespace strata::spec {

double CostModel::step_ms(int k, bool mtp) const {
    const int n = std::clamp(k + 1, 1, K_MAX + 1);
    const double dense = dense_ms * dense_ratio[n - 1];
    // CPU misses: distinct experts grow as U(n); each extra token routed to a missed expert adds a fraction of a
    // read. Tokens per distinct expert = n / U(n)-ratio (U measured in units of one token's 10 experts).
    const double u = distinct_ratio[n - 1];
    const double uses_per_expert = (double) n / u;
    const double cpu = (1.0 - hit_rate) * cpu_all_miss_ms * u * (1.0 + extra_use_cost * (uses_per_expert - 1.0));
    const double draft = k * (mtp ? mtp_draft_ms : lookup_draft_ms);
    return dense + cpu + sync_ms + draft;
}

Controller::Controller(CostModel cost, double min_gain, double ema) : cost_(cost), min_gain_(min_gain), ema_(ema) {
    // Priors: MTP per-position acceptance from flyweight's measurement on this model (0.86 per draft); lookup by
    // match length from the offline replay (2026-09-23-suffix-lookup). Both are then learned per session.
    mtp_p_.fill(0.86);
    lookup_q_ = {0.35, 0.6, 0.8, 0.92};
}

double Controller::expected(const double* p, int k, bool same) {
    double e = 1.0, run = 1.0;                                // the verify pass always commits one token
    for (int i = 0; i < k; ++i) {
        run *= same ? p[0] : p[i];
        e += run;
    }
    return e;
}

Choice Controller::choose(int lookup_available, int lookup_match, bool mtp_ready) const {
    Choice best;
    best.tokens_per_ms = 1.0 / cost_.step_ms(0, false);
    const double baseline = best.tokens_per_ms;
    auto consider = [&](Source s, int k, double e) {
        const double rate = e / cost_.step_ms(k, s == Source::Mtp);
        if (rate > best.tokens_per_ms) best = Choice{s, k, e, rate};
    };
    const int lk = std::min(lookup_available, K_MAX);
    const double q = lookup_q_[bucket(lookup_match)];
    for (int k = 1; k <= lk; ++k) consider(Source::Lookup, k, expected(&q, k, true));
    if (mtp_ready)
        for (int k = 1; k <= K_MAX; ++k) consider(Source::Mtp, k, expected(mtp_p_.data(), k, false));
    if (best.source != Source::None && best.tokens_per_ms < baseline * (1.0 + min_gain_)) {
        Choice none;
        none.tokens_per_ms = baseline;
        return none;
    }
    return best;
}

void Controller::observe(const Choice& c, int accepted, int lookup_match) {
    if (c.source == Source::None || c.k <= 0) return;
    accepted = std::clamp(accepted, 0, c.k);
    // Positions 0..accepted-1 were accepted given their prefix; position `accepted` (if drafted) was rejected.
    const int seen = std::min(c.k, accepted + 1);
    for (int i = 0; i < seen; ++i) {
        const double hit = i < accepted ? 1.0 : 0.0;
        if (c.source == Source::Mtp) mtp_p_[i] += ema_ * (hit - mtp_p_[i]);
        else lookup_q_[bucket(lookup_match)] += ema_ * (hit - lookup_q_[bucket(lookup_match)]);
    }
    // Everything drafted was accepted: the positions beyond the window were never tested, so without this they
    // keep their prior forever and the window can never grow. Pull them toward the deepest observed rate.
    if (c.source == Source::Mtp && accepted == c.k && c.k < K_MAX)
        for (int i = c.k; i < K_MAX; ++i) mtp_p_[i] += ema_ * (mtp_p_[c.k - 1] - mtp_p_[i]);
}

}  // namespace strata::spec
