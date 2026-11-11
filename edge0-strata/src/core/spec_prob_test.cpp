// src/core/spec_prob_test.cpp - probabilistic draft acceptance (include/strata/core/spec_prob.hpp), no GPU.
//
// A reference rejection sampler on synthetic target distributions p and drafter distributions q over a 16-token
// vocabulary; 2,000,000 trials per case.  Each trial draws a draft d from q, judges it with spec_verify_row and
// records the token the row emits; a chi-square test (99.9% bound) compares the emitted histogram with p.  If
// min(1, p/q) plus the residual preserved p only approximately the test would fail; two controls with a deliberately
// wrong rule must fail it, which shows the test can tell.
//
// The cases: q = p; q = a point mass (the argmax draft, no list); q disjoint from p; q's support wider than p's; the
// draft head's argmax outside p's support (p(d) = 0); a perturbed q.  The chains: top_k 1, 5, 64; top_p 0.5, 0.95;
// min_p 0.1; temperature 0.3, 1.0, 1.5 - p and q each cut by their OWN chain (q never borrows p's cut), the sampler's
// own arithmetic (top_p's cut in double over the top_k list, min_p in logit space, temperature after).
// Then a two-token WINDOW over a tiny Markov model checks the JOINT of the first two tokens (a wrong residual shows
// up in the pairs first), with a draft of depth 2 whose second row is conditioned on the first draft.
// Truncation regressions: a point-mass row inside a window that also has q rows (a lookup chain's tail) behaves as
// exact match (accept probability p(d)), and "no list" is never "q = 0".
#include "strata/core/spec_prob.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace strata::core;

namespace {

constexpr int V = 16;
int g_fail = 0;
int g_cases = 0;

struct Chain {
    int top_k;
    float top_p;
    float min_p;
    float temp;
};

struct List {
    std::vector<int32_t> id;
    std::vector<double> p;
    std::vector<float> q;   // the same probabilities as float, a drafter's list
};

// the sampler's chain over logits (no penalties): top_k list in selection order, top_p cut in double on the raw
// logits, min_p in logit space, temperature, normalise (sampled_probs_warp)
List chain_list(const std::vector<float>& logit, const Chain& c) {
    std::vector<int> ord(V);
    for (int i = 0; i < V; ++i) ord[i] = i;
    std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return logit[a] > logit[b]; });
    const int k = std::min(V, (c.top_k > 0 && c.top_k < 64) ? c.top_k : 64);
    int n_keep = k;
    const float mx = logit[ord[0]];
    if (c.top_p < 1.0f) {
        double sum = 0.0;
        std::vector<double> e(k);
        for (int i = 0; i < k; ++i) { e[i] = std::exp((double) logit[ord[i]] - (double) mx); sum += e[i]; }
        double cum = 0.0;
        int cut = k;
        for (int i = 0; i < k; ++i) {
            cum += e[i] / sum;
            if (cum >= (double) c.top_p) { cut = i + 1; break; }
        }
        n_keep = std::max(cut, 1);
    }
    if (c.min_p > 0.0f) {
        const float thresh = logit[ord[0]] + std::log(c.min_p);
        for (int i = 0; i < n_keep; ++i)
            if (logit[ord[i]] < thresh) { n_keep = i; break; }
    }
    const float inv_t = 1.0f / c.temp;
    float smx = logit[ord[0]] * inv_t;
    for (int i = 1; i < n_keep; ++i) smx = std::max(smx, logit[ord[i]] * inv_t);
    List L;
    double sum = 0.0;
    for (int i = 0; i < n_keep; ++i) {
        L.id.push_back(ord[i]);
        L.p.push_back(std::exp((double) (logit[ord[i]] * inv_t) - (double) smx));
        sum += L.p.back();
    }
    for (double& x : L.p) x /= sum;
    for (double x : L.p) L.q.push_back((float) x);
    return L;
}

float uni(std::mt19937_64& g) { return (float) (g() >> 40) * (1.0f / 16777216.0f); }

int32_t draw(const List& L, float u) {   // the sampler's inverse-CDF pick
    double cum = 0.0;
    int32_t pick = L.id.back();
    for (size_t i = 0; i < L.id.size(); ++i) {
        cum += L.p[i];
        if ((double) u < cum) { pick = L.id[i]; break; }
    }
    return pick;
}

// chi-square statistic of `count` against probabilities `prob` (cells with expectation < 5 are pooled), and the 99.9%
// critical value for its degrees of freedom (Wilson-Hilferty)
struct Chi { double stat; int df; double crit; };
Chi chi_square(const std::vector<double>& count, const std::vector<double>& prob, double n) {
    double stat = 0.0, pool_o = 0.0, pool_e = 0.0;
    int cells = 0;
    for (size_t i = 0; i < count.size(); ++i) {
        const double e = prob[i] * n;
        if (e < 5.0) { pool_o += count[i]; pool_e += e; continue; }
        stat += (count[i] - e) * (count[i] - e) / e;
        ++cells;
    }
    if (pool_e >= 5.0) { stat += (pool_o - pool_e) * (pool_o - pool_e) / pool_e; ++cells; }
    else if (pool_o > pool_e + 6.0 * std::sqrt(pool_e) + 10.0) stat += 1e9;   // far more hits than the thin tail's expectation
    const int df = std::max(1, cells - 1);
    const double z = 3.0902;   // 99.9%
    const double a = 2.0 / (9.0 * df);
    const double crit = df * std::pow(1.0 - a + z * std::sqrt(a), 3.0);
    return {stat, df, crit};
}

void report(const char* name, const Chi& c, bool expect_fail = false) {
    ++g_cases;
    const bool pass = c.stat <= c.crit;
    const bool ok = expect_fail ? !pass : pass;
    if (!ok) ++g_fail;
    std::printf("  %-4s %-58s chi2 %9.1f  df %2d  bound %6.1f%s\n", ok ? "ok" : "FAIL", name, c.stat, c.df, c.crit,
                expect_fail ? "  (control: must fail)" : "");
}

enum class Rule { Standard, AlwaysAccept, NoListMeansZeroQ };

// one row: draw d from q, judge it
int32_t one_trial(std::mt19937_64& g, const List& P, const List* Q, int32_t point_d, Rule rule, bool* accepted) {
    const SpecP sp{P.id.data(), P.p.data(), (int) P.id.size()};
    int32_t d;
    SpecQ sq{nullptr, nullptr, -1};
    if (Q != nullptr) {
        d = draw(*Q, uni(g));
        sq = SpecQ{Q->id.data(), Q->q.data(), (int) Q->id.size()};
    } else {
        d = point_d;
    }
    const float ua = uni(g), ur = uni(g);
    SpecResult r = spec_verify_row(sp, d, sq, ua, ur);
    if (rule == Rule::AlwaysAccept) r = {d, 1};
    if (rule == Rule::NoListMeansZeroQ && Q == nullptr) r = {d, 0};   // "no q: q = 0": reject; (the residual is then p)
    if (rule == Rule::NoListMeansZeroQ && Q == nullptr) r.token = draw(P, ur);
    *accepted = r.accepted != 0;
    return r.token;
}

struct Setup {
    const char* name;
    List P;
    List Q;
    bool point;     // q is a point mass at its argmax: no list
    int32_t d_point;
};

void run_case(const Setup& S, int trials, std::mt19937_64& g, Rule rule, bool expect_fail, double* acc_rate) {
    std::vector<double> cnt(V, 0.0), prob(V, 0.0);
    for (size_t i = 0; i < S.P.id.size(); ++i) prob[(size_t) S.P.id[i]] = S.P.p[i];
    long long acc = 0;
    for (int t = 0; t < trials; ++t) {
        bool a = false;
        const int32_t tok = one_trial(g, S.P, S.point ? nullptr : &S.Q, S.d_point, rule, &a);
        cnt[(size_t) tok] += 1.0;
        acc += a;
    }
    // a token outside p's support must never be emitted
    double outside = 0.0;
    for (int i = 0; i < V; ++i)
        if (prob[(size_t) i] == 0.0) outside += cnt[(size_t) i];
    Chi c = chi_square(cnt, prob, (double) trials);
    if (outside > 0.0) c.stat += 1e9;
    if (acc_rate) *acc_rate = (double) acc / trials;
    char nm[160];
    std::snprintf(nm, sizeof nm, "%s (accept %.3f)", S.name, (double) acc / trials);
    report(nm, c, expect_fail);
}

std::vector<float> rand_logits(std::mt19937_64& g, float spread) {
    std::normal_distribution<float> nd(0.0f, spread);
    std::vector<float> l(V);
    for (float& x : l) x = nd(g);
    return l;
}

}  // namespace

int main() {
    std::mt19937_64 g(20261006);
    const int N = 2000000;
    const std::vector<Chain> chains = {{64, 1.0f, 0.0f, 1.0f},  {1, 1.0f, 0.0f, 1.0f},   {5, 1.0f, 0.0f, 1.0f},
                                       {64, 0.5f, 0.0f, 1.0f},  {64, 0.95f, 0.0f, 1.0f}, {20, 0.95f, 0.05f, 0.7f},
                                       {64, 1.0f, 0.1f, 1.0f},  {64, 1.0f, 0.0f, 0.3f},  {64, 1.0f, 0.0f, 1.5f},
                                       {8, 0.9f, 0.02f, 1.0f}};
    std::printf("spec_prob_test: %d trials per case, 16-token vocabulary\n", N);
    for (const Chain& c : chains) {
        std::printf("chain top_k %d top_p %.2f min_p %.2f temp %.1f\n", c.top_k, c.top_p, c.min_p, c.temp);
        const std::vector<float> lp = rand_logits(g, 2.0f);
        const List P = chain_list(lp, c);
        // q = p
        run_case({"q = p", P, P, false, 0}, N, g, Rule::Standard, false, nullptr);
        // a perturbed q (the typical drafter), cut by its own chain
        std::vector<float> lq = lp;
        std::normal_distribution<float> nd(0.0f, 0.8f);
        for (float& x : lq) x += nd(g);
        run_case({"perturbed q", P, chain_list(lq, c), false, 0}, N, g, Rule::Standard, false, nullptr);
        // q a point mass at its argmax (no list: the argmax draft)
        {
            const List Qp = chain_list(lq, Chain{1, 1.0f, 0.0f, 1.0f});
            run_case({"q = point mass (argmax draft, no list)", P, Qp, true, Qp.id[0]}, N, g, Rule::Standard, false,
                     nullptr);
        }
        // q disjoint from p: the drafter's top tokens are the target's tail
        {
            std::vector<float> ld(V);
            for (int i = 0; i < V; ++i) ld[i] = -lp[i];
            run_case({"q disjoint from p (inverted logits)", P, chain_list(ld, c), false, 0}, N, g, Rule::Standard,
                     false, nullptr);
        }
        // q's support wider than p's: p cut hard, q uncut
        run_case({"q support wider than p's", P, chain_list(lq, Chain{64, 1.0f, 0.0f, 1.0f}), false, 0}, N, g,
                 Rule::Standard, false, nullptr);
        // the drafter's argmax is outside p's support: p(d) = 0 for the point-mass draft
        {
            const List Qp = chain_list(lq, Chain{1, 1.0f, 0.0f, 1.0f});
            std::vector<float> lo = lp;
            lo[(size_t) Qp.id[0]] = -50.0f;   // the target gives the drafter's favourite no mass
            const List Po = chain_list(lo, c);
            run_case({"p(argmax draft) = 0, point mass", Po, Qp, true, Qp.id[0]}, N, g, Rule::Standard, false, nullptr);
        }
    }
    // controls: wrong rules must fail the test
    {
        std::printf("controls\n");
        const Chain c{64, 0.95f, 0.0f, 1.0f};
        const std::vector<float> lp = rand_logits(g, 2.0f);
        std::vector<float> lq = lp;
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (float& x : lq) x += nd(g);
        const List P = chain_list(lp, c);
        run_case({"always accept the draft", P, chain_list(lq, c), false, 0}, 200000, g, Rule::AlwaysAccept, true,
                 nullptr);
        const List Qp = chain_list(lq, Chain{1, 1.0f, 0.0f, 1.0f});
        const double pd = [&] { double v = 0; for (size_t i = 0; i < P.id.size(); ++i) if (P.id[i] == Qp.id[0]) v = P.p[i]; return v; }();
        double acc_std = 0, acc_zero = 0;
        run_case({"point mass, standard rule", P, Qp, true, Qp.id[0]}, 200000, g, Rule::Standard, false, &acc_std);
        run_case({"point mass, 'no list = q 0' (rejects all: #29924)", P, Qp, true, Qp.id[0]}, 200000, g,
                 Rule::NoListMeansZeroQ, false, &acc_zero);
        ++g_cases;
        const bool ok = std::fabs(acc_std - pd) < 0.01 && acc_zero == 0.0;
        if (!ok) ++g_fail;
        std::printf("  %-4s point-mass accept rate %.4f = p(d) %.4f; the q = 0 rule accepts %.4f\n", ok ? "ok" : "FAIL",
                    acc_std, pd, acc_zero);
    }

    // ---- the JOINT of the first two tokens over a Markov model, drafts of depth 2.
    // p0 over x0, p1(. | x0) over x1; the drafter q0, q1(. | d0) are perturbations; window = [d0, d1] verified by two
    // rows, then the continuation: after an accepted row the next row's pick; after a reject the row's own residual
    // token is x0 and the NEXT window starts from it (a plain sample of x1 from p1(. | x0)).
    for (const Chain& c : std::vector<Chain>{{64, 1.0f, 0.0f, 1.0f}, {20, 0.95f, 0.05f, 0.7f}, {5, 0.5f, 0.0f, 1.5f}}) {
        std::printf("joint (x0, x1) over a Markov model: top_k %d top_p %.2f min_p %.2f temp %.1f\n", c.top_k, c.top_p,
                    c.min_p, c.temp);
        std::vector<List> P1(V), Q1(V);
        const std::vector<float> l0 = rand_logits(g, 1.5f);
        std::vector<float> m0 = l0;
        std::normal_distribution<float> nd(0.0f, 0.9f);
        for (float& x : m0) x += nd(g);
        const List P0 = chain_list(l0, c), Q0 = chain_list(m0, c);
        for (int a = 0; a < V; ++a) {
            const std::vector<float> l1 = rand_logits(g, 1.5f);
            std::vector<float> m1 = l1;
            for (float& x : m1) x += nd(g);
            P1[(size_t) a] = chain_list(l1, c);
            Q1[(size_t) a] = chain_list(m1, c);
        }
        std::vector<double> cnt((size_t) V * V, 0.0), prob((size_t) V * V, 0.0);
        for (size_t i = 0; i < P0.id.size(); ++i)
            for (size_t j = 0; j < P1[(size_t) P0.id[i]].id.size(); ++j)
                prob[(size_t) P0.id[i] * V + (size_t) P1[(size_t) P0.id[i]].id[j]] = P0.p[i] * P1[(size_t) P0.id[i]].p[j];
        long long acc_tokens = 0;
        for (int t = 0; t < N; ++t) {
            const int32_t d0 = draw(Q0, uni(g));
            const int32_t d1 = draw(Q1[(size_t) d0], uni(g));
            // row 0 judges d0 against p0
            const SpecP s0{P0.id.data(), P0.p.data(), (int) P0.id.size()};
            const SpecQ q0{Q0.id.data(), Q0.q.data(), (int) Q0.id.size()};
            const float a0 = uni(g), r0 = uni(g);
            const SpecResult r = spec_verify_row(s0, d0, q0, a0, r0);
            const int32_t x0 = r.token;
            int32_t x1;
            if (r.accepted) {
                // row 1 (conditioned on the window's d0 == x0) judges d1 against p1(. | d0)
                const List& P = P1[(size_t) x0];
                const SpecP s1{P.id.data(), P.p.data(), (int) P.id.size()};
                const List& Q = Q1[(size_t) d0];
                const SpecQ q1{Q.id.data(), Q.q.data(), (int) Q.id.size()};
                const float a1 = uni(g), r1 = uni(g);
                x1 = spec_verify_row(s1, d1, q1, a1, r1).token;
                ++acc_tokens;
            } else {
                x1 = draw(P1[(size_t) x0], uni(g));   // the next window: row 0 of a fresh window, plain
            }
            cnt[(size_t) x0 * V + (size_t) x1] += 1.0;
        }
        char nm[96];
        std::snprintf(nm, sizeof nm, "joint of 2 tokens (row-0 kept %.3f)", (double) acc_tokens / N);
        report(nm, chi_square(cnt, prob, (double) N));
    }

    // ---- truncation regressions
    {
        std::printf("truncation regressions\n");
        // a window of 3 drafts: rows 0 and 1 carry q lists, row 2 is a lookup chain's point-mass tail.  Row 2 must
        // accept with probability p(d) (exact match), whatever the lists are, and a shortened draft (n_q smaller than
        // the number of drafts: the tail now starts earlier) must not inherit an empty list as q = 0.
        const Chain c{20, 0.95f, 0.0f, 0.7f};
        const List P = chain_list(rand_logits(g, 2.0f), c);
        const SpecP sp{P.id.data(), P.p.data(), (int) P.id.size()};
        for (size_t pick : {0u, 1u, 3u}) {
            const int32_t d = P.id[std::min(pick, P.id.size() - 1)];
            long long acc = 0;
            const int n = 400000;
            for (int t = 0; t < n; ++t) acc += spec_verify_row(sp, d, SpecQ{nullptr, nullptr, -1}, uni(g), uni(g)).accepted;
            double pd = 0;
            for (size_t i = 0; i < P.id.size(); ++i) if (P.id[i] == d) pd = P.p[i];
            ++g_cases;
            const double rate = (double) acc / n, se = std::sqrt(pd * (1 - pd) / n);
            const bool ok = std::fabs(rate - pd) < 4.5 * se + 1e-9;
            if (!ok) ++g_fail;
            std::printf("  %-4s point-mass row: accept %.4f vs p(d) %.4f (4.5 se %.4f)\n", ok ? "ok" : "FAIL", rate, pd,
                        4.5 * se);
        }
        // a draft outside p's support (cut by top_p / top_k / min_p) is rejected and the replacement is in p's support
        const int32_t outside = [&] {
            for (int v = 0; v < V; ++v) {
                bool in = false;
                for (int32_t id : P.id) in = in || id == v;
                if (!in) return v;
            }
            return -1;
        }();
        long long bad = 0;
        for (int t = 0; t < 100000 && outside >= 0; ++t) {
            const SpecResult r = spec_verify_row(sp, outside, SpecQ{nullptr, nullptr, -1}, uni(g), uni(g));
            bad += r.accepted || r.token == outside;
        }
        ++g_cases;
        if (bad != 0 || outside < 0) ++g_fail;
        std::printf("  %-4s a draft outside top_k/top_p/min_p support is always rejected, replaced inside p (%lld bad)\n",
                    (bad == 0 && outside >= 0) ? "ok" : "FAIL", bad);
        // p == q everywhere: always accepted (min(1, p/q) = 1), never a 0 / 0
        long long rej = 0;
        const List Q = P;
        const SpecQ sq{Q.id.data(), Q.q.data(), (int) Q.id.size()};
        for (int t = 0; t < 200000; ++t) {
            const int32_t d = draw(Q, uni(g));
            rej += !spec_verify_row(sp, d, sq, uni(g), uni(g)).accepted;
        }
        ++g_cases;
        if (rej != 0) ++g_fail;
        std::printf("  %-4s p == q: every draft is accepted (%lld rejected)\n", rej == 0 ? "ok" : "FAIL", rej);
    }

    std::printf("spec_prob_test: %d cases, %d failed\n", g_cases, g_fail);
    return g_fail == 0 ? 0 : 1;
}
