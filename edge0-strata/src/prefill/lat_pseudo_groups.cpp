#include "strata/prefill/lat_pseudo_groups.hpp"

namespace strata::prefill {

std::vector<LatPseudoGroup> flatten_groups(const unsigned long long* blobs, const int32_t* starts,
                                           const int32_t* counts, int n_groups, int tile) {
    std::vector<LatPseudoGroup> out;
    if (blobs == nullptr || starts == nullptr || counts == nullptr || n_groups <= 0 || tile <= 0) return out;
    for (int g = 0; g < n_groups; ++g) {
        const int32_t c = counts[g];
        for (int32_t off = 0; off < c; off += tile) {
            out.push_back({blobs[g], (int32_t) (starts[g] + off), (int32_t) (c - off < tile ? c - off : tile)});
        }
    }
    return out;
}

}  // namespace strata::prefill
