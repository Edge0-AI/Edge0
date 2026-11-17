// Host-only opt-in grouping permutation. No HIP calls, device math, or precision changes.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace strata::prefill::detail {

inline bool mmq_resident_sort_flag(const char* value) {
    return value != nullptr && std::strcmp(value, "1") == 0;
}

inline bool mmq_resident_sort_requested() {
    static const bool on = mmq_resident_sort_flag(std::getenv("STRATA_MMQ_RESIDENT_SORT_NE"));
    return on;
}

// STRATA_MMQ_GROUP_SORT=1: the same row-count order for the full 16-expert groups only; the last, partial group keeps
// its experts in ID order. A full group's launch has 160 * ntx (gate/up) or 320 * ntx (down) tiles, at or above the
// 90 % tile efficiency on the cards' SM counts, so it runs one block per tile with no stream-k K split whatever its tile
// width: each expert's rows get the same bits in any full group. The partial group can take the stream-k split, whose
// K boundaries depend on the group's tile count - leaving it unchanged keeps the prompt's bits.
inline bool mmq_group_sort_requested() {
    static const bool on = mmq_resident_sort_flag(std::getenv("STRATA_MMQ_GROUP_SORT"));
    return on;
}

// How many of `n` ordered experts the sort may move: all of them (STRATA_MMQ_RESIDENT_SORT_NE), or the full groups
// of `group` experts (STRATA_MMQ_GROUP_SORT).
inline size_t mmq_resident_sort_span(size_t n, size_t group, bool keep_tail) {
    return keep_tail && group > 0 ? n - n % group : n;
}

inline bool mmq_resident_sort_eligible(bool requested, bool use_mmq, bool native_layout, bool has_cache,
                                      const int32_t* layer_resident, int64_t n_expert,
                                      bool stream_all, bool layer_stream_empty) {
    // native-layout packs too: slot/src/off are built the same way for both layouts, and nothing reads off[e + 1] as a
    // count (4x GP100, sm_60: 1.2K prompt 7.27 -> 5.32 s, 35K 63.3 -> 54.1 s, the same bits)
    (void) native_layout;
    if (!requested || !use_mmq || !has_cache || !layer_resident || n_expert <= 0) return false;
    // All experts, including unselected ones: leave every partial/streamed layer in ID order.
    if (stream_all && !layer_stream_empty) return false;
    for (int64_t e = 0; e < n_expert; ++e) if (layer_resident[e] < 0) return false;
    return true;
}

// Preconditions are established by the existing caller: IDs were range-checked and counted;
// order contains each positive-count expert once in ID order; off has count.size()+1 entries;
// routes fits its original int32 offsets, and slot/src each have routes writable entries.
// After sorting, off[e] is an expert-indexed START, not an ID-prefix end at off[e+1].
// Only order[0, span) is sorted (span >= order.size(): all of it); the entries after it keep their positions.
inline void mmq_resident_sort_rows(const int32_t* ids, int64_t routes, int32_t k,
                                   const std::vector<int32_t>& count, std::vector<int32_t>& off,
                                   std::vector<int32_t>& order, int32_t* slot, int32_t* src,
                                   size_t span = SIZE_MAX) {
    std::stable_sort(order.begin(), order.begin() + (ptrdiff_t) std::min(span, order.size()), [&](int32_t a, int32_t b) {
        return count[(size_t) a] != count[(size_t) b]
            ? count[(size_t) a] < count[(size_t) b] : a < b;
    });
    std::fill(off.begin(), off.end(), 0);
    int32_t cursor = 0;
    for (int32_t e : order) {
        off[(size_t) e] = cursor;
        cursor += count[(size_t) e];
    }
    off.back() = cursor;
    std::vector<int32_t> fill(off.begin(), off.end() - 1);
    // Preserve token-major/rank-major traversal, hence the original within-expert row order.
    for (int64_t i = 0; i < routes; ++i) {
        const int32_t e = ids[(size_t) i];
        const int32_t p = fill[(size_t) e]++;
        slot[(size_t) i] = p;
        src[(size_t) p] = (int32_t) (i / k);
    }
}

}  // namespace strata::prefill::detail
