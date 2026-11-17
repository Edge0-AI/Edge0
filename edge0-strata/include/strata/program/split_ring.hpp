#pragma once

#include <cstdint>
#include <cstdlib>

// A layer split's prompt-path ring and the lend slots sized from it, kept apart from the engine so they can be tested
// on the CPU (header-only, like pipeline_gate.hpp).
//
//  1. split_ring_slots: the streamed-ring override a split chooses (STRATA_SPLIT_RING, else 96 when most experts are
//     resident).  The choice must be made BEFORE anything sizes the prompt path with Prefill::bytes_needed (which counts
//     ring_slots * max_blob): the split's lend regions (the RAM copies of the experts the prompt path borrows) used to
//     be sized first, with the default ring, so a larger ring borrowed more slots than the copy kept.
//  2. split_lend_slots: how many tail slots of a cache `need` bytes take - the scan's own arithmetic, used for the
//     RAM-copy region and for the borrow itself so the two cannot disagree.
namespace strata::program {

/// The ring override (slots) for a layer split, 0 = none (the pinned-share rule stays).
///   env        STRATA_SPLIT_RING as read from the environment (nullptr = unset); 0 / non-numeric = the pinned-share rule
///   res_share  the share of (layer, expert) pairs resident in some stage cache, 0..1
inline int split_ring_slots(const char* env, double res_share) {
    const int ring = env != nullptr ? std::atoi(env) : (res_share >= 0.75 ? 96 : 0);
    return ring > 0 ? ring : 0;
}

/// The number of tail slots of a cache of `slots` slots whose bytes cover `need`.  `offsets` is the cache's per-slot
/// byte-offset prefix sum (slots + 1 entries, `bytes` = offsets[slots]) or nullptr when every slot is `max_blob` bytes.
inline int64_t split_lend_slots(int64_t slots, int64_t bytes, const uint64_t* offsets, uint64_t max_blob, uint64_t need) {
    if (offsets != nullptr) {
        int64_t k = 0;
        while (k < slots && (uint64_t) (bytes - (int64_t) offsets[slots - k]) < need) ++k;
        return k;
    }
    return max_blob == 0 ? 0 : (int64_t) ((need + max_blob - 1) / max_blob);
}

}  // namespace strata::program
