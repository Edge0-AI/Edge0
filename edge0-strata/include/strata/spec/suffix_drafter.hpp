// include/strata/spec/suffix_drafter.hpp - plan v0.3 P6: the suffix-lookup drafter (no weights, no GPU).
//
// Proposes the tokens that followed the longest earlier occurrence of the sequence's current suffix: when the
// output quotes its input (edits, refactors, repeated code) the continuation is usually exact, and a verify pass
// accepts many tokens at once. It needs no model and costs microseconds.
//
// Index: every trigram of the history (prompt + accepted output) maps to its WAYS most recent end positions in a
// fixed-size open-addressing table, so memory is bounded (~20 bytes per history token) and appends are O(1).
// A proposal checks those candidates, extends each match backwards up to `max_match`, and takes the longest
// (most recent on ties). Matches shorter than `min_match` propose nothing.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::spec {

class SuffixDrafter {
public:
    static constexpr int WAYS = 4;

    explicit SuffixDrafter(int min_match = 3, int max_match = 32, size_t capacity_tokens = 1u << 19);

    void reset();
    /// Add tokens to the history (the prompt, then every accepted token).
    void append(const int32_t* tokens, size_t n);
    void append(int32_t token) { append(&token, 1); }

    /// Write up to `max_k` proposed next tokens to `out`; returns how many (0 = no match of at least min_match).
    int propose(int max_k, int32_t* out);

    /// --lookup-chain: as propose(), for the history followed by `extra` (the draft head's proposal for the next
    /// tokens), so the lookup continues where those drafts end. The history itself is unchanged; the extra tokens are
    /// only part of the suffix being matched and may be read back as part of a periodic continuation.
    int propose_after(const int32_t* extra, int n_extra, int max_k, int32_t* out);

    /// Length of the match behind the last proposal (0 if none).
    int last_match() const { return last_match_; }
    size_t size() const { return hist_.size(); }
    const std::vector<int32_t>& history() const { return hist_; }

private:
    struct Slot {
        uint64_t key = 0;             // trigram hash + 1 (0 = empty)
        uint32_t pos[WAYS] = {};      // end positions, most recent first
        uint8_t n = 0;
    };
    Slot* find_slot(uint64_t key, bool insert);
    uint64_t key_at(size_t end) const;

    int min_match_, max_match_;
    std::vector<int32_t> hist_;
    std::vector<Slot> table_;
    size_t mask_ = 0;
    int last_match_ = 0;
};

}  // namespace strata::spec
