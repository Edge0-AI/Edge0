// include/strata/spec/draft_policy.hpp - per verify round: the MTP's window, or a lookup (suffix) window?
//
// The suffix drafter (prompt lookup) proposes the tokens that followed an earlier repeat of the context. Taken
// whenever it proposes more than the MTP, it lost 2-8% on ordinary text: a long lookup window costs much more to
// verify than the MTP's usual 3-4 tokens, and it is only worth that when enough of it is accepted. llama.cpp's
// lookup decoding answers the same problem with confidence thresholds on its n-gram statistics; here the policy
// learns both sides online and compares expected committed tokens per millisecond:
//
//   MTP     E = the mean tokens a window of that size has committed (EMA), at the measured cost of that size
//   lookup  E(k) = 1 + q + q^2 + ... + q^k for k <= the proposal, q = the acceptance rate of lookup drafts whose
//           match was about as long (4 buckets of match length, decayed counts), at the measured cost of k + 1
//
// and takes the lookup window only when its best E/cost beats the MTP's by `margin`. Costs are the measured round
// times per window size (EMA; sizes not seen yet are scaled from seen ones by a prior shape), so the policy adapts
// to the machine and the context length. It only chooses which drafts to verify: the output is unchanged.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::spec {

class DraftPolicy {
public:
    static constexpr int kMaxT = 8;
    static constexpr int kBuckets = 4;
    static constexpr int kCtxBuckets = 3;   // STRATA_SPEC_COST: row costs per context length (< 6K, < 24K, longer)

    explicit DraftPolicy(int max_t, double margin = 0.03);

    struct Pick {
        bool lookup = false;
        int t = 1;                      // window size (1 + drafts)
    };
    /// `t_mtp`: the MTP's window; `lookup_k`: the lookup proposal's length (0 = none); `match`: its match length.
    Pick choose(int t_mtp, int lookup_k, int match) const;
    /// After the round: the window it used, the drafts accepted, and the round's time (verify + commit + draft).
    void observe(bool lookup, int t, int accepted, int match, double round_ms);
    /// Gated proposals, inside one generation: a round's committed outcome only, never its time. `t` is the
    /// lookup window chosen before the draws, or the configured MTP window; both are fixed by the history. A
    /// lookup window counts as a probe of its size, so a size whose cost is still guessed is tried kProbes times,
    /// not every round. Round times include the next window's drafts and the rows past the accepted ones, whose
    /// Philox counters the next rounds draw again, so they wait (defer_cost) or go to the live policy (observe).
    void observe_committed(bool lookup, int t, int accepted, int match);
    /// Gated proposals, inside one generation: a round's time, held until the committed position passes `end`,
    /// the last Philox counter of any draw that time depends on (the window's drafts and rows, the next window's
    /// drafts when the time includes them). Until then those counters can be drawn again by the next rounds and the
    /// time (rejected rows' routing, the next draft's gated length) would steer a window by its own draws.
    void defer_cost(int64_t end, int t, double round_ms);
    /// Apply, in order, the held times whose `end` is below `committed` (the next window's first counter), exactly
    /// as observe() prices a round, in the context bucket the round ran in.
    void apply_costs(int64_t committed);

    /// --lookup-chain: how many of `k_avail` lookup tokens to append after an MTP window of `t_mtp` tokens whose
    /// drafts all hold with probability `p_mtp` (estimated from the draft layer's own probabilities): the k with the
    /// best expected tokens per ms, E(t_mtp) + p_mtp (c + c^2 + .. + c^k) at the cost of t_mtp + k, if it beats the MTP
    /// window alone by `margin`; c = the acceptance of chained drafts by match length (their own counts). 0 = none.
    int chain(int t_mtp, double p_mtp, int k_avail, int match) const;
    /// After a chained round: the MTP window's size, the chained tokens, the window's accepted drafts, its time.
    void observe_chain(int t_mtp, int k, int accepted, int match, double round_ms);
    double chain_rate(int match) const;
    double lookup_rate(int match) const;   // current q for a match length
    double cost_ms(int t) const;           // measured or scaled round time of a window of t tokens
    /// A per-card row-cost table (round ms of the windows of 1..n rows, the draft included) counted as `weight` earlier rounds each:
    /// the start of a process that already knows the card (STRATA_SPEC_COST_TABLE; tuning time), replaced by what it then measures.
    void seed_costs(const double* ms, int n, double weight, int bucket = 0);
    /// STRATA_SPEC_COST: the context position of the next round; selects the row-cost table of its length (a row of attention costs
    /// more at 16K than at 1K).  A bucket with no measurement yet starts from the nearest one that has.  A no-op without set_robust(true).
    void set_context(long long pos);
    /// The measured round cost of every window size seen: "T1:39.8(n=120) T2:..." for the log / the next process's table.
    std::string cost_table_text() const;
    /// STRATA_SPEC_COST: a round time far above what is known for that width (the first use of a width captures its graph; a stalled
    /// round) counts as a tenth of a sample, so one capture cannot price a width out of use.  Off: the cost EMA as it always was.
    void set_robust(bool on) { robust_ = on; }

    /// COST-AWARE MTP WINDOW (STRATA_SPEC_COST=1): the MTP chain is drafted one step at a time; after draft j (probability p_j)
    ///   E_j = 1 + q_0 + .. + q_j,  q_i = prod_{k<=i} a(p_k)   a() = the acceptance measured for MTP drafts of that probability
    /// is the expected tokens of the window of j + 2 rows, worth E_j / cost(j + 2) tokens per ms (cost() = the measured round time of
    /// that window size, which includes the draft).  The window is the best of the widths seen; the chain goes on only while one or two
    /// more drafts could still beat it (the cost is not convex: on a card with experts in RAM a window of 3 costs about a window of 2).
    /// It only chooses which rows are verified; the output is unchanged.
    struct ChainState {
        double q = 1.0, e = 1.0;
        double best_v = 0.0;
        int best_rows = 1;                ///< the window to verify: 1 + the drafts kept
        int kept = 0;                     ///< best_rows - 1
    };
    /// Draft `idx` (0-based, probability p) was just computed: update `st` and say whether to compute draft idx + 1.
    /// idx == 0 starts a new chain.  `cap` = the widest window (rows).
    bool keep_draft(ChainState& st, int idx, float p, int cap) const;
    /// After a round of the MTP window T whose drafts were p[0..T-1): `accepted` of them held.  Teaches draft_rate().
    void observe_probs(const float* p, int n_window_drafts, int accepted);
    double draft_rate(double p) const;     // acceptance of an MTP draft that came with probability p

private:
    static int bucket(int match);
    double mtp_tokens(int t) const;
    bool stale(int t) const;               // a size not measured for a long stretch of rounds

    int max_t_;
    double margin_;
    std::array<std::array<double, kMaxT + 1>, kCtxBuckets> cost_{}, cost_n_{};   // round ms by context bucket and window size
    int bk_ = 0;                                           // the current context bucket
    std::array<double, kMaxT + 1> mtp_tok_{}, mtp_n_{};    // tokens committed by MTP windows of that size
    std::array<double, kBuckets> ok_{}, bad_{};            // lookup drafts accepted / windows cut short, decayed
    std::array<double, kBuckets> cok_{}, cbad_{};          // chained lookup drafts, the same (reached rounds only)
    double rounds_ = 0;                                    // rounds observed so far
    bool robust_ = false;
    double clamp_sample(int t, double ms) const;
    static constexpr int kPBuckets = 8;
    std::array<double, kPBuckets> pa_ok_{}, pa_n_{};       // MTP drafts reached / accepted by probability bucket (decayed)
    std::array<std::array<double, kMaxT + 1>, kCtxBuckets> last_{};   // the round at which each size was last measured
    // Gated proposals (observe_committed / defer_cost / apply_costs); all zero or empty on any other path.
    std::array<std::array<double, kMaxT + 1>, kCtxBuckets> tried_{};      // lookup windows of that size run with no time yet
    std::array<std::array<double, kMaxT + 1>, kCtxBuckets> delayed_n_{};  // of cost_n_: times applied by apply_costs (not new probes)
    struct HeldCost { int64_t end; int bucket; int t; double ms; };
    std::vector<HeldCost> held_;                                          // defer_cost, oldest first
};

}  // namespace strata::spec
