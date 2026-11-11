// include/strata/spec/draft_source.hpp - a source of draft tokens beside the MTP layer (--lookup-chain).
//
// A source proposes how the context continues; the verify window checks the proposal, so a source can only change
// the speed, never the output. The engine asks its sources after the MTP has drafted: the context it passes ends
// with those MTP drafts (not yet verified), and whatever a source returns is appended to the same window.
//
// The built-in source is the request's own text (PromptLookupSource: the suffix drafter over prompt + output).
// Other sources (a retrieval store over a codebase or earlier conversations) register with
// extra_draft_sources() before the first request; the engine takes the proposal with the longest match.
#pragma once

#include "strata/spec/suffix_drafter.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace strata::spec {

class DraftSource {
public:
    virtual ~DraftSource() = default;
    virtual const char* name() const = 0;
    /// A new request starts (its prompt follows through append()).
    virtual void reset() {}
    /// Committed context, in order: the request's prompt, then every token the verify windows accept.
    virtual void append(const int32_t* tokens, size_t n) { (void) tokens; (void) n; }
    /// Propose up to `max_k` tokens that follow the context. `tail` holds the context's last `n_tail` tokens (at most
    /// 64); its last `n_pending` are drafts not yet verified (never passed to append()). Writes `out`, returns how
    /// many (0 = nothing to propose) and sets `*match` to the number of trailing context tokens the proposal is based
    /// on (its confidence: the engine requires --lookup-chain-min and prefers the longest).
    virtual int propose(const int32_t* tail, int n_tail, int n_pending, int max_k, int32_t* out, int* match) = 0;
};

/// The request's own text: the suffix drafter (prompt lookup) behind the DraftSource interface.
class PromptLookupSource final : public DraftSource {
public:
    explicit PromptLookupSource(SuffixDrafter& d) : d_(d) {}
    const char* name() const override { return "prompt-lookup"; }
    void reset() override { d_.reset(); }
    void append(const int32_t* tokens, size_t n) override { d_.append(tokens, n); }
    int propose(const int32_t* tail, int n_tail, int n_pending, int max_k, int32_t* out, int* match) override {
        n_pending = n_pending < n_tail ? n_pending : n_tail;
        const int k = d_.propose_after(tail + (n_tail - n_pending), n_pending, max_k, out);
        *match = k > 0 ? d_.last_match() : 0;
        return k;
    }

private:
    SuffixDrafter& d_;
};

/// Sources besides the request's own text, asked after it (empty unless something registered one at startup).
std::vector<std::unique_ptr<DraftSource>>& extra_draft_sources();

/// Ask `own` and then every extra source; the proposal with the longest match of at least `min_match` wins
/// (ties keep the earlier source). Returns its length (0 = none); `*match` and `*source` (index: 0 = own) describe it.
int propose_from_sources(DraftSource& own, const int32_t* tail, int n_tail, int n_pending, int max_k, int min_match,
                         int32_t* out, int* match, int* source);

}  // namespace strata::spec
