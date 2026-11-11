#pragma once
// Which single-request window graph to free when the next one cannot be instantiated for lack of VRAM (#3702).
// Header-only so a CPU test can run it (src/core/window_evict_test.cpp).

namespace strata::core {

/// The size T (1..max_t) whose captured graph is freed first: the largest size other than `keep` (the one being
/// captured) and `pinned` (a window of that size was prelaunched and must stay), among those with a graph (`have[t]`).
/// The larger sizes cost the most VRAM and are asked for least often, and a freed size is captured again when it is
/// next needed.  0: nothing to free.
inline int pick_window_victim(const bool* have, int max_t, int keep, int pinned) {
    for (int t = max_t; t >= 1; --t) {
        if (t == keep || t == pinned || !have[t]) continue;
        return t;
    }
    return 0;
}

}  // namespace strata::core
