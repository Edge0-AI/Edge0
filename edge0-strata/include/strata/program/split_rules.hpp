// include/strata/program/split_rules.hpp - rules for the automatic layer split that do not need a GPU to check.
//
// A stage that owns one or two layers holds a handful of expert slots (a 12 GB 3060 with one layer held 512), and
// the prompt chunk of the whole pipeline is clamped to what the smallest cache can lend (#448): 576 -> 347 tok/s in
// #1616.  So the search first looks for a placement in which every stage owns at least kMinStageLayers layers, and
// only falls back to the unrestricted best when no such placement can start the prompt path.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace strata::program::split_rules {

constexpr int64_t kMinStageLayers = 3;

/// The restriction only makes sense when the model has the layers for it.
inline bool restriction_possible(int64_t n_layers, int stages) {
    return stages >= 2 && n_layers >= kMinStageLayers * stages;
}

/// cuts: the first layer of each later stage (stages - 1 of them); n_layers: the model's layers.
/// Returns the size of the smallest stage.
inline int64_t smallest_stage(const std::vector<int64_t>& cuts, int64_t n_layers) {
    int64_t lo = n_layers, prev = 0;
    for (const int64_t c : cuts) {
        if (c - prev < lo) lo = c - prev;
        prev = c;
    }
    if (n_layers - prev < lo) lo = n_layers - prev;
    return lo;
}

/// True when some stage owns fewer than kMinStageLayers layers.
inline bool has_tiny_stage(const std::vector<int64_t>& cuts, int64_t n_layers) {
    return smallest_stage(cuts, n_layers) < kMinStageLayers;
}

// ---- #2644: the #1616 floor must not starve a card.  On an asymmetric pair (RTX 4060 Ti 16 GB + RTX 3060 12 GB, 48
// layers, IQ3_S) the predicted decode window prefers almost every layer on the faster card, the first pick was K=2, and the
// #1616 retry (every stage at least kMinStageLayers) landed on the edge of that restriction: K=3, the 3060 with 3 layers,
// 12 GB idle, 452 tok/s of prefill against 802 at 28 layers (the prompt flows through the stages as a pipeline, so one
// busy card is the whole rig).  So before that retry, a two-stage rig is asked for a placement in which each stage owns at
// least its share of the layers by its room for experts; the three-layer rule stays the second try, and the first pick
// the last.

/// Per stage, the fewest layers the share-of-room retry accepts: floor(n_layers x room_i / sum of rooms), at least
/// kMinStageLayers, and (when the floors would not fit) trimmed from the largest back toward kMinStageLayers until
/// they do.  Rooms <= 0 count as 0; no room anywhere gives every stage kMinStageLayers.
inline std::vector<int64_t> share_floors(const std::vector<int64_t>& room, int64_t n_layers) {
    std::vector<int64_t> floors(room.size(), kMinStageLayers);
    double total = 0.0;
    for (const int64_t r : room) total += r > 0 ? (double) r : 0.0;
    if (total <= 0.0) return floors;
    int64_t sum = 0;
    for (size_t i = 0; i < room.size(); ++i) {
        const double share = (room[i] > 0 ? (double) room[i] : 0.0) / total * (double) n_layers;
        floors[i] = std::max<int64_t>(kMinStageLayers, (int64_t) share);   // truncated: the floors sum to <= n_layers
        sum += floors[i];
    }
    while (sum > n_layers) {   // only through the kMinStageLayers lift of a small share
        size_t big = 0;
        for (size_t i = 1; i < floors.size(); ++i)
            if (floors[i] > floors[big]) big = i;
        if (floors[big] <= kMinStageLayers) break;
        --floors[big];
        --sum;
    }
    return floors;
}

/// True when some stage of `cuts` owns fewer layers than its floor.
inline bool below_floors(const std::vector<int64_t>& cuts, int64_t n_layers, const std::vector<int64_t>& floors) {
    int64_t prev = 0;
    for (size_t i = 0; i <= cuts.size() && i < floors.size(); ++i) {
        const int64_t end = i < cuts.size() ? cuts[i] : n_layers;
        if (end - prev < floors[i]) return true;
        prev = end;
    }
    return false;
}

/// The share-of-room retry is worth running when some floor is above the plain three-layer rule's.
inline bool floors_exceed_minimum(const std::vector<int64_t>& floors) {
    for (const int64_t f : floors)
        if (f > kMinStageLayers) return true;
    return false;
}

// ---- #1440: headroom among (near-)ties.  The search scores a placement by its predicted window time alone, so on two
// equal cards every K whose caches hold all the profiled pairs ties, and the first one tried wins: the smallest K, which
// gives the LAST stage (the one that also carries the MTP drafter) the most layers and lets its cache fill the card to
// the reserve.  On Linux xe a card that runs out of VRAM does not fail an allocation, the driver moves buffers to system
// memory: 2x Arc Pro B70, IQ3_S, auto picked K=24 with CUDA1 at 31.8 of 31.9 GiB and decoded ~11 tok/s, against ~54 with
// --layer-split 30.  So among the placements within `tol` of the best predicted time, the one whose tightest stage
// keeps the most VRAM beyond all of its layers' experts wins (ties on that: the faster, then the first).

/// One scored placement: its cuts, its predicted window time and the smallest spare room over its stages (bytes; a
/// stage's room after its session minus every expert of its layers; negative when its cache cannot hold them all).
struct Placement {
    std::vector<int64_t> cuts;
    double ms = 0.0;
    int64_t min_spare = 0;
};

/// The index of the plain pick (the first with the lowest time) in `c`, or 0 when `c` is empty.
inline size_t fastest(const std::vector<Placement>& c) {
    size_t best = 0;
    for (size_t i = 1; i < c.size(); ++i)
        if (c[i].ms < c[best].ms) best = i;
    return best;
}

/// The index of the pick with headroom: of the placements within `tol` (a fraction) of the fastest, the one with the
/// largest `min_spare`.  tol <= 0 still breaks exact ties.  0 when `c` is empty.
inline size_t pick_with_headroom(const std::vector<Placement>& c, double tol) {
    if (c.empty()) return 0;
    const size_t best = fastest(c);
    const double limit = c[best].ms + c[best].ms * (tol > 0.0 ? tol : 0.0);
    size_t pick = best;
    for (size_t i = 0; i < c.size(); ++i) {
        if (c[i].ms > limit) continue;
        if (c[i].min_spare > c[pick].min_spare || (c[i].min_spare == c[pick].min_spare && c[i].ms < c[pick].ms))
            pick = i;
    }
    return pick;
}

// ---- #2007: the miss cost of a stage on a narrower link.  0.1.41 scaled each stage's miss term by 20 GB/s / its probed
// link (clamped to 1..64), for 2x 4090 D with CUDA1 on a physical x1 link (1.6 GB/s: x12, auto K 22 -> 38, prompt
// ~350 -> ~1400 tok/s).  The same scale moved 2x Tesla T10 from K=23 to K=14: prompt -23%, decode -15% (xiake-1).
// So the scale is now relative to the stage with the fastest link (a symmetric rig keeps the fitted formula, whatever
// its link speed) and applies only from kLinkScaleMin on (a narrow riser or a chipset x1/x4 slot against an x16 one);
// below that the probe's noise and the formula's fit are the larger effects.

constexpr double kLinkScaleMin = 4.0;

/// Per stage: the factor on its miss term, from the probed H2D GB/s of each stage (<= 0: not probed -> 1.0).
inline std::vector<double> link_miss_scale(const std::vector<double>& gbps, double min_ratio = kLinkScaleMin) {
    double fastest = 0.0;
    for (const double b : gbps) fastest = b > fastest ? b : fastest;
    std::vector<double> s(gbps.size(), 1.0);
    if (fastest <= 0.0) return s;
    for (size_t i = 0; i < gbps.size(); ++i) {
        if (gbps[i] <= 0.0) continue;
        const double r = fastest / gbps[i];
        s[i] = r >= min_ratio ? (r > 64.0 ? 64.0 : r) : 1.0;
    }
    return s;
}

/// The smallest spare room over the stages of `cuts`: room[i] - session(lb, le) - need(lb, le), where `need_prefix[l]`
/// is the bytes of every expert of layers [0, l) (n_layers + 1 entries).
template <class Session>
inline int64_t min_spare(const std::vector<int64_t>& cuts, int64_t n_layers, const std::vector<int64_t>& room,
                         const std::vector<int64_t>& need_prefix, Session&& session) {
    int64_t lo = INT64_MAX;
    for (size_t i = 0; i <= cuts.size(); ++i) {
        const int64_t lb = i == 0 ? 0 : cuts[i - 1];
        const int64_t le = i < cuts.size() ? cuts[i] : n_layers;
        const int64_t spare = room[i] - session(lb, le) - (need_prefix[(size_t) le] - need_prefix[(size_t) lb]);
        if (spare < lo) lo = spare;
    }
    return lo;
}

}  // namespace strata::program::split_rules
