// include/strata/core/coupled_draft.hpp - COUPLED DRAFT SAMPLING (STRATA_SPEC_COUPLED=1, off by default).
//
// Under sampling the verify window samples row t of a window at position pos0 as Philox(seed, pos0 + t) over the
// sampler chain (penalties -> top_k -> top_p -> min_p -> temperature -> inverse-CDF pick; verify.cpp run(),
// sampler.cu), and a draft is kept only when it EQUALS that row's pick.  The output is therefore a function of the
// seed alone, whatever the drafts are: drafts only decide how many tokens a window yields.
//
// The MTP draft layer used to propose its ARGMAX.  In coupled mode it SAMPLES instead, with the target's own chain
// (the request's top_k / top_p / min_p / temperature and penalties over the same history) and the SAME uniform the
// target will draw for the row that verifies the draft.  When the draft's distribution is close to the target's,
// the two inverse-CDF picks from one uniform agree far more often than an argmax agrees with a sample; the target's
// pick - and so the text - does not change, because verification is still exact-match against the target's sample.
//
// STRATA_SPEC_GUMBEL=1 (opt-in) replaces the inverse-CDF pick on BOTH sides with a Gumbel-max pick: argmax p_i / E_i,
// E_i ~ Exp(1) keyed by (seed, counter, token id) (sampler.cu `gumbel_exp`).  A token then gets the same noise in the
// draft's and the target's draw whichever other tokens survived either cut, so they agree on the tokens they share;
// with one uniform over two probability-sorted lists, a one-token difference in the lists shifts every later pick.
//
// THE COUNTER ARITHMETIC (all positions are sequence indices; a window's row t holds the token at pos0 + t and its
// pick is the token at pos0 + t + 1, drawn with counter pos0 + t):
//   * MTP cell c pairs the main model's residual at position c with the token at c + 1 and predicts the token at
//     c + 2 (mtp.hpp).
//   * After a window at p with a drafts accepted, the next window starts at p' = p + a + 1 (its row 0 is the token
//     the window picked at row a).  Chain step j (j = 0 ..) runs at cell p + a + j and predicts the token at
//     p + a + j + 2 = p' + j + 1: row j + 1 of the next window, VERIFIED BY ROW j's pick, counter p' + j.
//   * So a draft made at cell c is verified with counter c + 1 - whatever the window size T, the accepted count, a
//     suffix-drafter window in between (its drafts are then simply not used), or `draft_first` (cell p - 1, a = 0 for
//     a window at p).  The kernel reads c from the step record the host already stages for the cell (device memory,
//     so the captured graphs replay with each round's own counter), and the seed and chain from a device copy of the
//     request's SamplerParams, refreshed at the start of every round graph.
//
// THE PENALTY HISTORY.  The verify row j of the next window counts penalties over penalty_rows(consumed', window')
// row j: the last h of (consumed', x', drafts 0 .. j-1), where consumed' is what the session holds after the commit
// and x' the window's pick at row a.  The drafter keeps a device RING of kCoupledHistCap + max_t slots: the host
// stages the base (the last h of consumed' + x', -1 padded in front) in [cap - h, cap) before the round, and draft j
// writes its token to cap + j - so draft j's window is [cap + j - h, cap + j), exactly the row the target will use.
#pragma once

#include "strata/kernels/sampler.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_COUPLED_HD __host__ __device__
#else
#define STRATA_COUPLED_HD
#endif

namespace strata::core {

/// The longest penalty window the coupled drafter mirrors (serve's kPenaltyWindowCap).
inline constexpr int kCoupledHistCap = 4096;

inline bool& coupled_draft_state() {
    static bool on = [] {
        const char* e = std::getenv("STRATA_SPEC_COUPLED");
        return e != nullptr && *e != '\0' && std::strcmp(e, "0") != 0;
    }();
    return on;
}

/// STRATA_SPEC_PROB=1: probabilistic draft acceptance (core/spec_prob.hpp).  It runs on the coupled drafter's
/// machinery (the chain's sampling over the draft head, the penalty ring), so it switches that setup on too.
inline bool spec_prob_env() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_SPEC_PROB");
        return e != nullptr && *e != '\0' && std::strcmp(e, "0") != 0;
    }();
    return on;
}

/// Whether coupled draft sampling is enabled (via STRATA_SPEC_COUPLED=1 or --coupled-draft CLI flag), or the
/// probabilistic mode that needs the same buffers and graphs.
inline bool coupled_draft_env() {
    return coupled_draft_state() || spec_prob_env();
}

inline void set_coupled_draft(bool on) {
    coupled_draft_state() = on;
}

/// The MTP cell of chain step j after a window at `p` with `a` drafts accepted (draft() runs step j there).
STRATA_COUPLED_HD inline int64_t coupled_draft_cell(int64_t p, int a, int j) { return p + a + j; }

/// The Philox counter the verify window will draw for the row that checks the draft made at MTP cell `cell`.
STRATA_COUPLED_HD inline uint64_t coupled_draft_counter(int64_t cell) { return (uint64_t) (cell + 1); }

/// The penalty window the draft layer and the target use: min(penalty_last_n, cap), 0 when penalties are off.
STRATA_COUPLED_HD inline int coupled_hist_len(int penalty_last_n, int cap) {
    return penalty_last_n <= 0 ? 0 : (penalty_last_n < cap ? penalty_last_n : cap);
}

/// Where draft j's penalty window starts in the ring (base in [cap - h, cap), draft i at cap + i).
STRATA_COUPLED_HD inline int coupled_hist_start(int cap, int j, int h) { return cap + j - h; }

/// The ring's base on the host: the last `h` tokens of (tail[0..n_tail), next), -1 padded in front, into
/// out[0..h) - row 0 of `penalty_rows` for the one-token window [next].
inline void coupled_hist_base(const int32_t* tail, int64_t n_tail, int32_t next, int h, int32_t* out) {
    if (h <= 0) return;
    strata::kernels::penalty_rows(tail, n_tail, &next, 1, h, out);
}

}  // namespace strata::core
