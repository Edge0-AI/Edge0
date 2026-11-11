// src/spec/draft_source.cpp - see include/strata/spec/draft_source.hpp.
#include "strata/spec/draft_source.hpp"

#include <vector>

namespace strata::spec {

std::vector<std::unique_ptr<DraftSource>>& extra_draft_sources() {
    static std::vector<std::unique_ptr<DraftSource>> sources;
    return sources;
}

int propose_from_sources(DraftSource& own, const int32_t* tail, int n_tail, int n_pending, int max_k, int min_match,
                         int32_t* out, int* match, int* source) {
    *match = 0;
    *source = -1;
    if (max_k <= 0) return 0;
    int best = 0;
    std::vector<int32_t> buf((size_t) max_k);
    auto ask = [&](DraftSource& s, int index) {
        int m = 0;
        const int k = s.propose(tail, n_tail, n_pending, max_k, buf.data(), &m);
        if (k <= 0 || m < min_match || m <= *match) return;
        for (int i = 0; i < k && i < max_k; ++i) out[i] = buf[(size_t) i];
        best = k < max_k ? k : max_k;
        *match = m;
        *source = index;
    };
    ask(own, 0);
    int index = 1;
    for (auto& s : extra_draft_sources())
        if (s) ask(*s, index++);
    return best;
}

}  // namespace strata::spec
