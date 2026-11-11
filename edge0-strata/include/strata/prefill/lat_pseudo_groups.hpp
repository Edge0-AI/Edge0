// The latent A stage's tile flatten: one <= 64-row pseudo-group per tile of each group, so the fused kernel's
// block-per-group mapping cannot be pinned by a long group (measured 0.4419 vs 0.0652 us/entry on the engine's own
// group lengths).  Pure host logic, no CUDA: the kernel reads (blob, start, count) triples and the result is the same
// set of rows as the unflattened groups.
#pragma once

#include <cstdint>
#include <vector>

namespace strata::prefill {

struct LatPseudoGroup {
    unsigned long long blob;   // the group's expert blob
    int32_t start;             // the absolute entry row of this tile
    int32_t count;             // 1 .. 64
};

/// One pseudo-group per <= `tile` rows of each group.  `starts`/`counts` are the caller's groups (counts[g] rows from
/// starts[g]); `blobs` is their blob pointers.  The output covers exactly the same rows, in the same order, no
/// duplicates: every (start..start+count) of the input appears as a series of `tile`-sized pieces.
std::vector<LatPseudoGroup> flatten_groups(const unsigned long long* blobs, const int32_t* starts,
                                           const int32_t* counts, int n_groups, int tile = 64);

}  // namespace strata::prefill
