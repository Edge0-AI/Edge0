// include/strata/prefill/kva_floor.hpp - which cells of a session were built from approximated rows (KVA).
//
// A KVA read leaves the draft layer's K/V with cells [0, floor) built from rows the late layers never computed.  The
// draft layer must not attend to them (it would only cost draft acceptance, but that is the decode speed), and a
// session that KVA did not approximate must not lose any draft context at all.  The floor therefore follows the
// SESSION'S TOKENS, not the process: after a KVA read the registry remembers the prefix (its token ids) that was
// approximated; whatever state a later request continues from (the live session, a conversation checkpoint, a parked
// conversation), the floor of that state is how much of its tokens agree with a remembered prefix, capped by the
// prefix's floor.  A state no KVA read produced has no common prefix, so its floor is 0 and its draft layer attends
// to the whole window, as with STRATA_KVA=0.
//
// A mislabel can only cost draft cells (an exact state that shares a prefix with an approximated one reports the
// floor too high); it can never make the draft attend to approximated cells that were registered, and the target
// model verifies every token whatever the draft read.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace strata::prefill {

class KvaFloorRegistry {
public:
    explicit KvaFloorRegistry(size_t cap = 8) : cap_(cap) {}

    /// A KVA read approximated the cells [0, floor) of the prompt `ids` (at least `floor` tokens).
    template <class Token>
    void note(const Token* ids, int64_t floor) {
        if (floor <= 0) return;
        ++clock_;
        for (Entry& e : v_)
            if ((int64_t) e.ids.size() == floor && std::equal(e.ids.begin(), e.ids.end(), ids)) { e.used = clock_; return; }
        Entry e;
        e.ids.assign(ids, ids + floor);
        e.used = clock_;
        if (v_.size() >= cap_) {
            size_t lru = 0;
            for (size_t i = 1; i < v_.size(); ++i) if (v_[i].used < v_[lru].used) lru = i;
            v_.erase(v_.begin() + (std::ptrdiff_t) lru);
        }
        v_.push_back(std::move(e));
    }

    /// The first cell of a state holding the tokens ids[0, n) that is NOT an approximated one: the longest common
    /// prefix with a remembered read, over the remembered reads.  0 when the tokens match none.
    template <class Token>
    int64_t floor_for(const Token* ids, int64_t n) const {
        int64_t best = 0;
        for (const Entry& e : v_) {
            const int64_t m = std::min<int64_t>(n, (int64_t) e.ids.size());
            int64_t i = 0;
            while (i < m && (int64_t) ids[i] == e.ids[(size_t) i]) ++i;
            best = std::max(best, i);
        }
        return best;
    }

    size_t size() const { return v_.size(); }
    void clear() { v_.clear(); }

private:
    struct Entry {
        std::vector<int64_t> ids;
        uint64_t used = 0;
    };
    std::vector<Entry> v_;
    size_t cap_;
    uint64_t clock_ = 0;
};

}  // namespace strata::prefill
