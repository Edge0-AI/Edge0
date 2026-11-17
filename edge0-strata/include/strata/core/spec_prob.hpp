// include/strata/core/spec_prob.hpp - PROBABILISTIC DRAFT ACCEPTANCE (STRATA_SPEC_PROB=1, off by default).
//
// Speculative rejection sampling (Leviathan et al. / Chen et al.) for the MTP drafts of a SAMPLED request.  Under
// sampling the verify window used to accept a draft only when it EQUALS the token the target sampled for that row
// (exact match).  For a draft that is the head's argmax - a point mass - that is already the standard rule, but it
// throws away what the draft head knows: its whole distribution q.  Here the draft layer SAMPLES its draft d from q
// (its own Philox stream) and the verifier accepts d with probability min(1, p(d) / q(d)); on a reject it samples the
// replacement from the residual norm(max(0, p - q)).  The output token of every row is then distributed exactly as
// p, the target's chain (penalties -> top_k -> top_p -> min_p -> temperature), whatever q is.
//
//   p   the TARGET'S distribution at the row: the list the sampler really samples from, after penalties, top_k,
//       top_p, min_p and temperature, renormalised.  At most 64 entries (kSelMax).  NOT the raw softmax: using it
//       would accept tokens the chain never emits (the llama.cpp #29924 class of bug).
//   q   the DRAFTER'S distribution: the same chain over the draft head's logits, with the drafter's own top_k /
//       top_p cut (each list is cut by its own rule - q never borrows p's cut).  A sparse list of at most 64
//       (id, probability) pairs, probabilities summing to 1.
//
// "NO q MEANS POINT MASS, NEVER q = 0."  A drafter without a candidate list (prompt lookup, suffix drafts, the
// lookup chain) proposes one token for sure: q is a delta at d, the accept probability is p(d) and the residual is p
// with d removed.  Treating "no list" as q = 0 would reject every such draft (llama.cpp #29924).  The engine only
// builds q lists for the MTP drafts; every other row of a window is a point-mass row.
//
// THE VERIFY ROWS ARE INDEPENDENT.  Row t of a window at pos0 judges draft t (token t + 1 of the window) with its
// own uniforms, three disjoint Philox counter spaces keyed by the row's position pos0 + t:
//     u_pick   counter = pos          the plain sample (the window's last row: the bonus token, and every row of a
//                                     window with no draft) - the same draw the exact-match sampler uses
//     u_accept counter = pos | 1<<62  accept / reject
//     u_resid  counter = pos | 1<<63  the residual sample after a reject
//     (the drafter's own draw: counter = pos | 1<<61)
// Row t writes out[t] = d when it accepts, and the residual sample (never d) when it rejects, so the host's
// unchanged loop `while (a < T-1 && window[a+1] == out[a]) ++a` takes exactly the accepted prefix, and out[a] is
// the bonus token.  Rows after the first reject are computed and ignored.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace strata::core {

/// The default confidence gate tests the drawn token. gated_proposal.hpp carries its admitted
/// mass and a NULL outcome to the verifier; a failed gate is still a proposal attempt.
/// Explicit `top` tests q's largest probability before the draw and uses the normalized-q path.
inline bool spec_gate_pick() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_SPEC_PROB_GATE");
        return e == nullptr || std::strcmp(e, "top") != 0;
    }();
    return on;
}

/// STRATA_SPEC_PROB_DT=x: the drafter's chain runs at x * the request's temperature (default 1).  Any q is valid for
/// rejection sampling; the acceptance sum min(p, q) is what it changes.
inline float spec_draft_temp_scale() {
    static const float s = [] {
        const char* e = std::getenv("STRATA_SPEC_PROB_DT");
        const float v = e != nullptr ? (float) std::atof(e) : 1.0f;
        return v > 0.0f ? v : 1.0f;
    }();
    return s;
}

/// STRATA_SPEC_MIN_TEMP=x: coupled / probabilistic drafting only for requests with temperature >= x (default 0: every
/// sampled request).  Measured on the RTX 3060 and 5070: sampled drafting gains at temperature 1.0 and loses below
/// about 0.7, where the draft head's argmax is the better guess; below x the drafter keeps its argmax drafts.
inline float spec_min_temp() {
    static const float s = [] {
        const char* e = std::getenv("STRATA_SPEC_MIN_TEMP");
        return e != nullptr ? (float) std::atof(e) : 0.0f;
    }();
    return s;
}

/// One q row = 128 int32 slots: 64 token ids (-1 ends the list), then the 64 probabilities as float bits.
inline constexpr int kSpecQEntries = 64;
inline constexpr int kSpecQStride = 2 * kSpecQEntries;

inline constexpr uint64_t kSpecCtrAccept = 1ull << 62;
inline constexpr uint64_t kSpecCtrResid = 1ull << 63;
inline constexpr uint64_t kSpecCtrDraft = 1ull << 61;

/// The post-chain distribution of a row: `n` ids and their probabilities (double, the sampler's own `ex`).
struct SpecP {
    const int32_t* id;
    const double* p;
    int n;
};

/// The drafter's list for the row (n < 0: no list, a point mass at the draft).
struct SpecQ {
    const int32_t* id;
    const float* q;
    int n;
};

struct SpecResult {
    int32_t token;
    int accepted;   // 1: the draft was kept
};

/// The inverse-CDF pick of the sampler's tail, over weights w[i] / sum.
inline int32_t spec_pick(const SpecP& P, const double* w, double sum, float u) {
    double cum = 0.0;
    int last = -1;
    for (int i = 0; i < P.n; ++i) {
        if (!(w[i] > 0.0)) continue;
        last = i;
        cum += w[i] / sum;
        if ((double) u < cum) return P.id[i];
    }
    return last >= 0 ? P.id[last] : P.id[P.n > 0 ? P.n - 1 : 0];
}

/// A plain sample from p (no draft on this row: the last row of a window).  Matches the sampler's tail.
inline int32_t spec_sample(const SpecP& P, float u_pick) {
    double cum = 0.0;
    int32_t pick = P.id[P.n > 0 ? P.n - 1 : 0];
    for (int i = 0; i < P.n; ++i) {
        cum += P.p[i];
        if ((double) u_pick < cum) { pick = P.id[i]; break; }
    }
    return pick;
}

/// One verify row with a draft `d`: accept with min(1, p(d)/q(d)), else the residual sample.
inline SpecResult spec_verify_row(const SpecP& P, int32_t d, const SpecQ& Q, float u_accept, float u_resid) {
    double pd = 0.0;
    for (int i = 0; i < P.n; ++i)
        if (P.id[i] == d) { pd = P.p[i]; break; }
    double qd = -1.0;   // < 0: point mass
    if (Q.n >= 0)
        for (int i = 0; i < Q.n; ++i)
            if (Q.id[i] == d) { qd = (double) Q.q[i]; break; }
    const bool point = !(qd > 0.0);   // no list, or d is not in it (it always is for a drafted token): a point mass
    const bool accept = pd > 0.0 && (point ? (double) u_accept < pd : (double) u_accept * qd < pd);
    if (accept) return {d, 1};
    // the residual over p's support: max(0, p - q) (q = delta at d for a point mass: p without d)
    double w[64];
    double sum = 0.0;
    const int n = P.n < 64 ? P.n : 64;
    for (int i = 0; i < n; ++i) {
        double wi;
        if (point) {
            wi = P.id[i] == d ? 0.0 : P.p[i];
        } else {
            double qi = 0.0;
            for (int j = 0; j < Q.n; ++j)
                if (Q.id[j] == P.id[i]) { qi = (double) Q.q[j]; break; }
            wi = P.p[i] - qi;
            if (wi < 0.0) wi = 0.0;
        }
        w[i] = wi;
        sum += wi;
    }
    if (!(sum > 0.0)) return {d, 1};   // p and q agree everywhere (rounding): the draft is as likely as it can be
    return {spec_pick(SpecP{P.id, P.p, n}, w, sum, u_resid), 0};
}

}  // namespace strata::core
