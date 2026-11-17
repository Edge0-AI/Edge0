// include/strata/program/stage_reserve.hpp - what the layer split keeps free on a card when it sizes the expert cache.
//
// This is the arithmetic of `stage_room` in src/program/generate.cpp, moved here unchanged so a host test can pin it
// (tests/program/stage_reserve_test.cpp). `reserve_mib` is exactly the sum that function subtracted from the card's
// free memory before; the default placement does not change.
//
// #3702: on an unequal pair (RTX 3060 + RTX 5060 Ti) the last card was left with 68 MiB after loading although
// 700 + 96 MiB were reserved for it, and the verify windows' graphs (captured on first use, 4-6 MiB each and once 42 MiB)
// then ran it out of memory. `proposed_window_mib` below is a PROPOSAL for a larger window term; nothing calls it
// (it needs the model gate in phaseA-tests/gh027/gates/3702-later-reserve.sh before it can become the default).
#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::program::stage_reserve {

constexpr int64_t kWindowMib = 96;       ///< the verify windows on a later stage; 75 MiB measured, rounded up
constexpr int64_t kDrafterMib = 1000;    ///< the MTP drafter (839 MiB) + the head, on the last stage only

struct Inputs {
    int64_t base_reserve_mib = 0;   ///< --vram-reserve-later-mib on a later stage when given, else --vram-reserve-mib
    int64_t prompt_buffers_mib = 0; ///< the prompt path's own buffers (0 when they are borrowed from the cache)
    bool later = false;             ///< a stage after the first
    bool drafter = false;           ///< the stage that carries the MTP drafter and the head
    int64_t pipe_window_mib = 0;    ///< --pipeline-windows: a second verifier on every stage
    int64_t pipe_snap_mib = 0;      ///< --pipeline-windows 2 or more: the first stage's second GDN state
};

/// What the plan keeps free on the card, in MiB (generate.cpp's `reserve` before the shift).
inline int64_t reserve_mib(const Inputs& in) {
    return in.base_reserve_mib + in.prompt_buffers_mib + (in.later ? kWindowMib : in.pipe_snap_mib) + in.pipe_window_mib +
           (in.drafter ? kDrafterMib : 0);
}

/// The room the cache may use: the free bytes less the reserve, never below 0.
inline int64_t room_bytes(int64_t free_bytes, const Inputs& in) {
    return std::max<int64_t>(free_bytes - (reserve_mib(in) << 20), 0);
}

// ---------------------------------------------------------------------------------------------------- the proposal
constexpr int64_t kGraphMib = 6;            ///< one window graph, captured on first use (log of #3702: 4, 6, 6)
constexpr int64_t kInstantiateBurstMib = 48; ///< the largest single step seen (42 MiB in #3702): room kept for one more

/// PROPOSAL, unused: the window term of a later stage when the windows of every size 1..max_window_t can still be
/// captured after loading: the first window's arena (kWindowMib already holds it), a graph for each further size, and
/// one burst.
inline int64_t proposed_window_mib(int max_window_t) {
    const int64_t sizes = std::max<int64_t>(max_window_t, 1);
    return kWindowMib + (sizes - 1) * kGraphMib + kInstantiateBurstMib;
}

/// PROPOSAL, unused: reserve_mib with the proposed window term on a later stage.
inline int64_t proposed_reserve_mib(const Inputs& in, int max_window_t) {
    Inputs p = in;
    int64_t r = reserve_mib(p);
    if (in.later) r += proposed_window_mib(max_window_t) - kWindowMib;
    return r;
}

}  // namespace strata::program::stage_reserve
