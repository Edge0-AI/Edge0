// src/spec/draft_policy.cpp - see include/strata/spec/draft_policy.hpp.
#include "strata/spec/draft_policy.hpp"

#include <algorithm>
#include <cstdio>

namespace strata::spec {
namespace {

// The shape of a round's cost by window size, relative to one token, used only for sizes not measured yet (the
// measured round times replace it). Between the RTX 5070's measured curves: +10 ms per token with every missed expert
// on the CPU (bench/results/2026-09-27-spec/window-cost), flatter with the default CPU/DMA split.
constexpr double kShape[DraftPolicy::kMaxT + 1] = {0.0, 1.0, 1.35, 1.7, 2.05, 2.45, 2.85, 3.25, 3.6};
constexpr double kCostAlpha = 0.1;    // EMA weight of a new round time
constexpr double kTokAlpha = 0.05;    // EMA weight of a new MTP window outcome
constexpr double kCalDecay = 0.995;   // the MTP probability -> acceptance table fades slowly
constexpr double kDecay = 0.97;       // lookup counts: older windows fade
// Before a bucket has data: the longer the match, the likelier its continuation (llama.cpp's lookup decoding gates
// on the same thing); worth 4 observations, so a few real windows override it
constexpr double kPriorQ[DraftPolicy::kBuckets] = {0.75, 0.88, 0.93, 0.96};
constexpr double kPriorN = 4.0;
constexpr int kProbes = 3;       // a lookup window size is tried this often before its guessed cost can veto it
// A size not measured for this many rounds is tried again when a confident lookup could use it: its cost was set in
// one stretch (a prompt's cache refill, say) and, priced high, it would never be measured again (#1252)
constexpr double kStaleRounds = 300.0;

}  // namespace

DraftPolicy::DraftPolicy(int max_t, double margin)
    : max_t_(std::clamp(max_t, 1, kMaxT)), margin_(margin) {}

int DraftPolicy::bucket(int match) {
    return match < 6 ? 0 : match < 12 ? 1 : match < 24 ? 2 : 3;
}

double DraftPolicy::lookup_rate(int match) const {
    const int b = bucket(match);
    return (ok_[b] + kPriorN * kPriorQ[b]) / (ok_[b] + bad_[b] + kPriorN);
}

double DraftPolicy::cost_ms(int t) const {
    t = std::clamp(t, 1, kMaxT);
    if (cost_n_[bk_][t] > 0) return cost_[bk_][t];
    // scale from the measured sizes, weighting each by how often it was seen
    double num = 0.0, den = 0.0;
    for (int u = 1; u <= kMaxT; ++u)
        if (cost_n_[bk_][u] > 0) {
            const double w = std::min(cost_n_[bk_][u], 20.0);
            num += w * cost_[bk_][u] * kShape[t] / kShape[u];
            den += w;
        }
    return den > 0 ? num / den : kShape[t];
}

double DraftPolicy::clamp_sample(int t, double ms) const {
    if (!robust_) return ms;
    const double known = cost_ms(t);           // the measured value, or the scaled guess from the other widths
    if (cost_n_[bk_][t] == 0) {
        // the first use of a width captures its graph: that round says nothing about its cost, the guess from the other widths stands in
        for (int u = 1; u <= kMaxT; ++u)
            if (cost_n_[bk_][u] > 0) return known;
        return ms;
    }
    return std::min(ms, 2.0 * known);
}

void DraftPolicy::seed_costs(const double* ms, int n, double weight, int bucket) {
    bucket = std::clamp(bucket, 0, kCtxBuckets - 1);
    for (int t = 1; t <= std::min(n, kMaxT); ++t)
        if (ms[t - 1] > 0.0 && cost_n_[bucket][t] == 0) {
            cost_[bucket][t] = ms[t - 1];
            cost_n_[bucket][t] = weight;
        }
}

void DraftPolicy::set_context(long long pos) {
    if (!robust_) return;
    const int nb = pos < 6000 ? 0 : pos < 24000 ? 1 : 2;
    if (nb == bk_) return;
    bool has = false;
    for (int t = 1; t <= kMaxT; ++t) has = has || cost_n_[nb][t] > 0;
    if (!has) {   // nothing measured at this length yet: the nearest bucket that has data is the prior (one sample's worth)
        for (int d = 1; d < kCtxBuckets && !has; ++d)
            for (const int src : {nb - d, nb + d}) {
                if (src < 0 || src >= kCtxBuckets || has) continue;
                for (int t = 1; t <= kMaxT; ++t)
                    if (cost_n_[src][t] > 0) {
                        cost_[nb][t] = cost_[src][t];
                        cost_n_[nb][t] = 1.0;
                        last_[nb][t] = rounds_;
                        has = true;
                    }
            }
    }
    bk_ = nb;
}

std::string DraftPolicy::cost_table_text() const {
    std::string out;
    char b[64];
    for (int k = 0; k < kCtxBuckets; ++k) {
        bool any = false;
        for (int t = 1; t <= kMaxT; ++t) any = any || cost_n_[k][t] > 0;
        if (!any) continue;
        out += (out.empty() ? "" : " | ");
        out += k == 0 ? "ctx<6K:" : k == 1 ? "ctx<24K:" : "ctx>=24K:";
        for (int t = 1; t <= kMaxT; ++t)
            if (cost_n_[k][t] > 0) {
                std::snprintf(b, sizeof b, " T%d:%.2f(n=%.0f)", t, cost_[k][t], cost_n_[k][t]);
                out += b;
            }
    }
    return out;
}

bool DraftPolicy::stale(int t) const { return cost_n_[bk_][t] > 0 && rounds_ - last_[bk_][t] > kStaleRounds; }

double DraftPolicy::mtp_tokens(int t) const {
    if (mtp_n_[t] > 0) return mtp_tok_[t];
    return 1.0 + 0.7 * (t - 1);       // before any MTP window of this size: a typical acceptance
}

DraftPolicy::Pick DraftPolicy::choose(int t_mtp, int lookup_k, int match) const {
    Pick p;
    p.t = std::clamp(t_mtp, 1, max_t_);
    if (lookup_k <= 0) return p;
    const double base = mtp_tokens(p.t) / cost_ms(p.t);
    const double q = lookup_rate(match);
    double e = 1.0, qi = 1.0, best = 0.0;
    int best_t = 0;
    for (int k = 1; k <= std::min(lookup_k, max_t_ - 1); ++k) {
        qi *= q;
        e += qi;
        const double r = e / cost_ms(k + 1);
        if (r > best) { best = r; best_t = k + 1; }
    }
    const int t_full = std::min(lookup_k, max_t_ - 1) + 1;
    if (best_t > 0 && best > base * (1.0 + margin_)) {
        p.lookup = true;
        p.t = best_t;
        // a wider size whose cost has gone stale is re-measured, so one slow stretch cannot price it out for good
        if (best_t < t_full && q >= 0.85 && stale(t_full)) p.t = t_full;
        return p;
    }
    // a guessed cost can keep the policy from ever measuring a size: the first few times a confident lookup would
    // need a size not measured yet, it is tried (verification keeps the output; only the one round's speed is at stake)
    if (t_full > p.t && (cost_n_[bk_][t_full] - delayed_n_[bk_][t_full] + tried_[bk_][t_full] < kProbes || stale(t_full)) && q >= 0.85) {
        p.lookup = true;
        p.t = t_full;
    }
    return p;
}

void DraftPolicy::observe(bool lookup, int t, int accepted, int match, double round_ms) {
    t = std::clamp(t, 1, kMaxT);
    if (round_ms > 0) {
        round_ms = clamp_sample(t, round_ms);
        const double alpha = robust_ ? std::max(kCostAlpha, 1.0 / (cost_n_[bk_][t] + 1.0)) : kCostAlpha;   // a running mean for the first samples
        cost_[bk_][t] = cost_n_[bk_][t] > 0 ? (1.0 - alpha) * cost_[bk_][t] + alpha * round_ms : round_ms;
        cost_n_[bk_][t] += 1.0;
        last_[bk_][t] = rounds_;
    }
    rounds_ += 1.0;
    if (lookup) {
        const int b = bucket(match);
        ok_[b] = kDecay * ok_[b] + accepted;
        bad_[b] = kDecay * bad_[b] + (accepted < t - 1 ? 1.0 : 0.0);
    } else {
        const double got = accepted + 1.0;
        mtp_tok_[t] = mtp_n_[t] > 0 ? (1.0 - kTokAlpha) * mtp_tok_[t] + kTokAlpha * got : got;
        mtp_n_[t] += 1.0;
    }
}

void DraftPolicy::observe_committed(bool lookup, int t, int accepted, int match) {
    t = std::clamp(t, 1, kMaxT);
    rounds_ += 1.0;
    if (lookup) {
        tried_[bk_][t] += 1.0;
        last_[bk_][t] = rounds_;   // tried now: not stale within this generation
        const int b = bucket(match);
        ok_[b] = kDecay * ok_[b] + accepted;
        bad_[b] = kDecay * bad_[b] + (accepted < t - 1 ? 1.0 : 0.0);
    } else {
        const double got = accepted + 1.0;
        mtp_tok_[t] = mtp_n_[t] > 0 ? (1.0 - kTokAlpha) * mtp_tok_[t] + kTokAlpha * got : got;
        mtp_n_[t] += 1.0;
    }
}

void DraftPolicy::defer_cost(int64_t end, int t, double round_ms) {
    if (round_ms > 0) held_.push_back({end, bk_, std::clamp(t, 1, kMaxT), round_ms});
}

void DraftPolicy::apply_costs(int64_t committed) {
    size_t n = 0;
    const int bucket_now = bk_;
    for (; n < held_.size() && held_[n].end < committed; ++n) {
        const HeldCost& h = held_[n];
        bk_ = h.bucket;   // observe()'s update, in the bucket the round ran in
        const double ms = clamp_sample(h.t, h.ms);
        const double alpha = robust_ ? std::max(kCostAlpha, 1.0 / (cost_n_[bk_][h.t] + 1.0)) : kCostAlpha;
        cost_[bk_][h.t] = cost_n_[bk_][h.t] > 0 ? (1.0 - alpha) * cost_[bk_][h.t] + alpha * ms : ms;
        cost_n_[bk_][h.t] += 1.0;
        delayed_n_[bk_][h.t] += 1.0;   // a round already counted by observe_committed
        last_[bk_][h.t] = rounds_;
    }
    bk_ = bucket_now;
    held_.erase(held_.begin(), held_.begin() + (std::ptrdiff_t) n);
}

namespace {
int p_bucket(double p) {
    const int b = (int) (p * 8.0);
    return b < 0 ? 0 : b > 7 ? 7 : b;
}
}  // namespace

// Until a bucket has data a draft is as likely to hold as the MTP says (worth kPriorN observations); measured
// acceptance of the MTP chain is not the same as its own softmax (code and thinking text are over-confident).
double DraftPolicy::draft_rate(double p) const {
    const int b = p_bucket(p);
    const double centre = std::clamp(p, 0.0, 1.0);
    return (pa_ok_[b] + kPriorN * centre) / (pa_n_[b] + kPriorN);
}

void DraftPolicy::observe_probs(const float* p, int n, int accepted) {
    // draft j was reached when the drafts before it held; it held when j < accepted
    for (int j = 0; j < n && j <= accepted; ++j) {
        const int b = p_bucket(p[j]);
        pa_ok_[b] = kCalDecay * pa_ok_[b] + (j < accepted ? 1.0 : 0.0);
        pa_n_[b] = kCalDecay * pa_n_[b] + 1.0;
    }
}

bool DraftPolicy::keep_draft(ChainState& st, int idx, float p, int cap) const {
    cap = std::clamp(cap, 1, max_t_);
    if (idx == 0) {
        st = ChainState{};
        st.best_v = 1.0 / cost_ms(1);
    }
    const double a = draft_rate(p);
    st.q *= a;
    st.e += st.q;
    const int rows = idx + 2;
    if (rows <= cap) {
        const double v = st.e / cost_ms(rows);
        if (v > st.best_v * (1.0 + margin_ / 3.0)) {
            st.best_v = v;
            st.best_rows = rows;
        }
        // a width not measured yet is tried a few times when the chain is likely to hold that far
        if (cost_n_[bk_][rows] < kProbes && st.q >= 0.15) st.best_rows = rows;
    }
    st.kept = st.best_rows - 1;
    if (rows + 1 > cap) return false;                       // no row left for another draft
    // one or two more drafts, each as likely to hold as this one did, could they beat the best width so far?
    const double v1 = (st.e + st.q * a) / cost_ms(rows + 1);
    const double v2 = rows + 2 <= cap ? (st.e + st.q * a + st.q * a * a) / cost_ms(rows + 2) : 0.0;
    return std::max(v1, v2) > st.best_v * (1.0 + margin_ / 3.0) || (cost_n_[bk_][rows + 1] < kProbes && st.q * a >= 0.15);
}

double DraftPolicy::chain_rate(int match) const {
    // prior: a chained draft is a lookup continuing the MTP's drafts - weaker evidence than a plain match of that length
    constexpr double kChainPrior[kBuckets] = {0.45, 0.65, 0.8, 0.9};
    const int b = bucket(match);
    return (cok_[b] + kPriorN * kChainPrior[b]) / (cok_[b] + cbad_[b] + kPriorN);
}

int DraftPolicy::chain(int t_mtp, double p_mtp, int k_avail, int match) const {
    t_mtp = std::clamp(t_mtp, 1, max_t_);
    const int kmax = std::min(k_avail, max_t_ - t_mtp);
    if (kmax <= 0) return 0;
    const double e0 = mtp_tokens(t_mtp), base = e0 / cost_ms(t_mtp), c = chain_rate(match);
    p_mtp = std::clamp(p_mtp, 0.0, 1.0);
    double gain = 0.0, ci = 1.0, best = 0.0;
    int best_k = 0;
    for (int k = 1; k <= kmax; ++k) {
        ci *= c;
        gain += ci;
        const double r = (e0 + p_mtp * gain) / cost_ms(t_mtp + k);
        if (r > best) { best = r; best_k = k; }
    }
    const int t_full = t_mtp + kmax;
    if (best_k > 0 && best > base * (1.0 + margin_)) {
        // As choose(): a profitable shorter chain must not keep a stale full size from being measured again.
        if (best_k < kmax && p_mtp * c >= 0.6 && stale(t_full)) return kmax;
        return best_k;
    }
    // as choose(): a size whose cost is only guessed is tried a few times when the continuation looks likely
    if ((cost_n_[bk_][t_full] < kProbes || stale(t_full)) && p_mtp * c >= 0.6) return kmax;
    return 0;
}

void DraftPolicy::observe_chain(int t_mtp, int k, int accepted, int match, double round_ms) {
    const int t = std::clamp(t_mtp + k, 1, kMaxT);
    if (round_ms > 0) {
        round_ms = clamp_sample(t, round_ms);
        const double alpha = robust_ ? std::max(kCostAlpha, 1.0 / (cost_n_[bk_][t] + 1.0)) : kCostAlpha;   // a running mean for the first samples
        cost_[bk_][t] = cost_n_[bk_][t] > 0 ? (1.0 - alpha) * cost_[bk_][t] + alpha * round_ms : round_ms;
        cost_n_[bk_][t] += 1.0;
        last_[bk_][t] = rounds_;
    }
    rounds_ += 1.0;
    // the MTP part is an ordinary MTP window as far as its own drafts go
    const int mtp_acc = std::min(accepted, t_mtp - 1);
    const double got = mtp_acc + 1.0;
    mtp_tok_[t_mtp] = mtp_n_[t_mtp] > 0 ? (1.0 - kTokAlpha) * mtp_tok_[t_mtp] + kTokAlpha * got : got;
    mtp_n_[t_mtp] += 1.0;
    if (accepted < t_mtp - 1) return;   // the chain was never reached: it says nothing about the lookup
    const int b = bucket(match), ok = accepted - (t_mtp - 1);
    cok_[b] = kDecay * cok_[b] + ok;
    cbad_[b] = kDecay * cbad_[b] + (ok < k ? 1.0 : 0.0);
}

}  // namespace strata::spec
