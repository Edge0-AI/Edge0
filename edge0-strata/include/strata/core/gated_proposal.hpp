#pragma once
// Host packet builder for gated speculative proposals.
// Draft tokens, scores and raw q remain read-only. Retained float probabilities
// keep their original bits; masked probabilities become +0 without renormalization.
#include "strata/core/spec_prob.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace strata::core::gated {
constexpr int kRows = 8;
constexpr int kEntries = strata::core::kSpecQEntries;
constexpr int kStride = strata::core::kSpecQStride;
struct RootChoice {
    bool eligible = false;
    int rescue_id = -1;
    int window = 1;
};
struct Packet {
    int T = 1, n_q = 0, offers = 0, attempts = 0, failed_index = -1;
    bool from_sfx = false, null_tail = false;
    std::array<int32_t, kRows * kStride> mu{};
    const char* error = nullptr;
};

inline float probability(const int32_t* row, int entry) {
    float p;
    std::memcpy(&p, row + kEntries + entry, sizeof(p));
    return p;
}

inline float mass_at(const int32_t* row, int id) {
    for (int i = 0; i < kEntries && row[i] >= 0; ++i)
        if (row[i] == id) return probability(row, i);
    return 0.0f;
}

// Preserve IDs, padding and retained probability bits. A negative rescue_id keeps
// only entries at or above theta; otherwise retain the rescue token as well.
inline bool mask_row(const int32_t* raw, int32_t* masked, float theta, int rescue_id, int vocab) {
    if (!raw || !masked || !std::isfinite(theta) || theta < 0.0f || vocab <= 0) return false;
    std::memcpy(masked, raw, kStride * sizeof(int32_t));
    int n = 0;
    for (; n < kEntries && raw[n] >= 0; ++n) {
        const float q = probability(raw, n);
        if (raw[n] >= vocab || !std::isfinite(q) || q < 0.0f || q > 1.0f) return false;
        for (int j = 0; j < n; ++j)
            if (raw[j] == raw[n]) return false;
        if (!(q >= theta || raw[n] == rescue_id)) {
            const float zero = 0.0f;
            std::memcpy(masked + kEntries + n, &zero, sizeof(zero));
        }
    }
    return n > 0;
}

inline bool draw_matches(const int32_t* raw, int d, float score, int vocab) {
    if (d < 0 || d >= vocab || !std::isfinite(score)) return false;
    const float q = mass_at(raw, d);
    return q > 0.0f && std::memcmp(&q, &score, sizeof(q)) == 0;
}

// budget is the actual producer max_steps for the latest successful draft call,
// not guessed from generated count. attempts counts rows inspected for this
// packet; n_generated separately counts all materialized rows, including discarded
// MTP lookahead when suffix wins. RootChoice must be computed without D/Tactual.
inline bool make_packet(Packet& out, const int32_t* raw_q, const int32_t* drafts, const float* scores,
                        int n_generated, int budget, int capacity, int vocab, float theta,
                        bool initial, const RootChoice& root) {
    out = Packet{};
    auto fail = [&](const char* message) {
        out.error = message;
        return false;
    };
    if (capacity < 1 || capacity > kRows || budget < 0 || budget >= capacity || n_generated < 0 ||
        n_generated > budget || !std::isfinite(theta) || theta < 0.0f) return fail("invalid attempt budget/count");
    if (initial) {
        if (n_generated != 0) return fail("initial window has stale attempts");
        return true; // No proposal attempt: ordinary single-row sampling.
    }
    if (n_generated == 0) {
        if (budget != 0) return fail("missing materialized attempt");
        return true; // Zero proposal budget: ordinary single-row sampling.
    }
    if (!raw_q || !drafts || !scores) return fail("missing raw proposal buffers");
    if (root.eligible && (root.rescue_id < 0 || root.rescue_id >= vocab || root.window < 2 || root.window > capacity))
        return fail("invalid prefix-only root choice");
    // Build the common root measure before consulting D0: every root branch sees
    // H union {s0} when E, including MTP and NULL branches where D0 != s0.
    if (!mask_row(raw_q, out.mu.data(), theta, root.eligible ? root.rescue_id : -1, vocab)) return fail("invalid root q");
    if (!draw_matches(raw_q, drafts[0], scores[0], vocab)) return fail("raw q(D0) differs from own-score dprob");
    if (root.eligible && drafts[0] == root.rescue_id) {
        if (!(mass_at(out.mu.data(), drafts[0]) > 0.0f)) return fail("rescued offer outside root mu");
        out.T = root.window;
        out.n_q = 1;
        out.offers = out.T - 1;
        out.attempts = 1;
        out.from_sfx = true;
        return true; // real root mu, then suffix deltas, last row plain bonus
    }
    for (int j = 0; j < n_generated; ++j) {
        const int32_t* raw = raw_q + (size_t) j * kStride;
        int32_t* mu = out.mu.data() + (size_t) j * kStride;
        if (j > 0 && !mask_row(raw, mu, theta, -1, vocab)) return fail("invalid later q");
        if (!draw_matches(raw, drafts[j], scores[j], vocab)) return fail("raw q(D) differs from own-score dprob");
        out.attempts = j + 1;
        const bool offer = scores[j] >= theta;
        if (!offer) {
            if (mass_at(mu, drafts[j]) != 0.0f) return fail("failed draw retained outside rescue branch");
            out.offers = j;
            out.T = j + 1;
            out.n_q = out.T;
            out.null_tail = true;
            out.failed_index = j;
            return true; // The failed draft is not a window input.
        }
        if (!(mass_at(mu, drafts[j]) > 0.0f)) return fail("offered draw outside positive mu support");
    }
    if (n_generated != budget) return fail("incomplete attempts are not a budget bonus");
    out.offers = n_generated;
    out.T = n_generated + 1;
    out.n_q = n_generated;
    return true;
}
}  // namespace strata::core::gated
