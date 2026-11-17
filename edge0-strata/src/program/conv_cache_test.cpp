// src/program/conv_cache_test.cpp - the conversation cache's retention policy: the shared root survives,
// the leaves rotate by least recent use.
//
// The scenarios, in the serve loop's terms (generate.cpp): checkpoints are a prefix chain - stamps[0] is
// the root, the deepest point every request so far has shared (the end of the system prompt, in practice).
// A stamp advances on creation, when a new checkpoint lands on an existing length, and when a request
// mounts through the checkpoint.
//
//   1. one conversation over its budget: the oldest leaf leaves - with fresh stamps in creation order this
//      is exactly the previous first-in-first-out behaviour;
//   2. the root survives arbitrarily many newer checkpoints however old its stamp is;
//   3. a mount advances a checkpoint's stamp and the rotation then takes a different leaf;
//   4. a one-slot budget has no room for root and leaf, so the pin is off and the oldest leaves (the
//      newest point stays, as before);
//   5. the policy is pure: the same stamps, the same victim.
#include "strata/program/conv_cache.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

using strata::program::conv_cache::eviction_victim;
using strata::program::conv_cache::restore_fits;
using strata::program::conv_cache::spread_victim;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
}  // namespace

// a conversation of `turns` turns of `turn` tokens each after a `root`-token system prompt, a checkpoint at every turn's
// end and at most `cap` kept; returns how many tokens a request whose prompt differs `back` tokens before the end reuses
int64_t reuse_after_edit(bool spread, int64_t root, int64_t turn, int turns, int64_t cap, int64_t back) {
    std::vector<int64_t> lens = {root};
    std::vector<uint64_t> stamps = {1};
    for (int t = 1; t <= turns; ++t) {
        lens.push_back(root + t * turn);
        stamps.push_back(uint64_t(t + 1));
        while ((int64_t) lens.size() > cap) {
            const size_t v = spread ? spread_victim(lens.data(), stamps.data(), lens.size(), cap, nullptr, nullptr)
                                    : eviction_victim(stamps.data(), stamps.size(), cap);
            lens.erase(lens.begin() + (std::ptrdiff_t) v);
            stamps.erase(stamps.begin() + (std::ptrdiff_t) v);
        }
    }
    const int64_t edit = lens.back() - back;
    int64_t reuse = 0;
    for (int64_t l : lens)
        if (l <= edit) reuse = l;
    return reuse;
}

int main() {
    std::printf("conv_cache_test\n");
    {   // --prompt-cache-spread: 200 turns of 1,000 tokens after a 9,525-token system prompt, 6 checkpoints
        const int64_t end = 9525 + 200 * 1000;
        const int64_t lru6 = reuse_after_edit(false, 9525, 1000, 200, 6, 6000);
        const int64_t spread6 = reuse_after_edit(true, 9525, 1000, 200, 6, 6000);
        std::printf("  an edit 6K before the end: least recent use reuses %lld, spread %lld of %lld\n", (long long) lru6,
                    (long long) spread6, (long long) end);
        check(lru6 == 9525, "least recent use: an edit 6K back falls through to the root");
        check(end - spread6 < 4 * 6000, "spread: an edit 6K back reads less than 24K again");
        const int64_t spread42 = reuse_after_edit(true, 9525, 1000, 200, 6, 42000);
        check(end - spread42 < 4 * 42000, "spread: an edit 42K back reads less than 168K again");
        check(reuse_after_edit(true, 9525, 1000, 200, 6, 500) == end - 1000, "spread: the turn before is still kept");
    }
    {
        const std::vector<int64_t> lens = {100, 200, 300, 400};
        const std::vector<uint64_t> stamps = {1, 2, 3, 4};
        const bool pinned[] = {false, true, true, false};
        check(spread_victim(lens.data(), stamps.data(), 4, 3, nullptr, pinned) ==
                  eviction_victim(stamps.data(), 4, 3, nullptr, pinned),
              "spread: every candidate pinned - the plain policy");
        check(spread_victim(lens.data(), stamps.data(), 2, 1, nullptr, nullptr) == 0,
              "spread: two items, one slot - the plain policy");
    }
    {
        const std::vector<uint64_t> stamps = {1, 2, 3, 4, 5, 6, 7};   // chat A: root + six turns, one over
        check(eviction_victim(stamps.data(), stamps.size(), 6) == 1,
              "over budget with fresh stamps: the oldest leaf leaves (the FIFO it was)");
    }
    {
        // chat B mounts through the root long after it was created; its stamp stays ancient
        const std::vector<uint64_t> stamps = {1, 100, 101, 102, 103, 104, 105};
        const size_t v = eviction_victim(stamps.data(), stamps.size(), 6);
        check(v == 1, "the root's stale stamp never makes it the victim");
        check(v != 0, "the root - the shared prefix - is never the victim");
    }
    {
        // the same chain after mounting checkpoint 1: its stamp jumps, the rotation moves to the next leaf
        const std::vector<uint64_t> stamped = {1, 8, 3, 4, 5, 6, 7};
        check(eviction_victim(stamped.data(), stamped.size(), 6) == 2,
              "after a mount, the rotation takes the next least recently used leaf");
    }
    {
        const std::vector<uint64_t> stamps = {1, 90, 91, 92};
        check(eviction_victim(stamps.data(), stamps.size(), 2) == 1,
              "a two-slot budget keeps root + newest and drops the leaf in between");
        check(eviction_victim(stamps.data(), stamps.size(), 1) == 0,
              "a one-slot budget has no pin: the oldest leaves, the newest point stays");
    }
    {
        const std::vector<uint64_t> stamps = {7, 3, 3, 5};
        const size_t a = eviction_victim(stamps.data(), stamps.size(), 4);
        const size_t b = eviction_victim(stamps.data(), stamps.size(), 4);
        check(a == 1 && b == 1, "ties among equally stale leaves: the earlier index, deterministically");
    }
    {   // --prompt-cache-tail: the tail checkpoint goes first; never the root, never the newest item
        const std::vector<uint64_t> st = {1, 2, 3, 4, 5, 6, 7};
        const bool t2[7] = {false, false, false, true, false, false, false};
        check(eviction_victim(st.data(), st.size(), 6, t2) == 3, "a stale tail checkpoint leaves before the oldest leaf");
        const bool t3[7] = {false, false, false, false, false, false, true};
        check(eviction_victim(st.data(), st.size(), 6, t3) == 1, "the newest item, even a tail, is not the victim");
        const bool t4[7] = {true, false, false, false, false, false, false};
        check(eviction_victim(st.data(), st.size(), 6, t4) == 1, "the root is never the victim, tail flag or not");
        check(eviction_victim(st.data(), st.size(), 6, nullptr) == 1, "no tail flags: the plain policy");
    }
    {   // pin=N: a pinned shared prefix is never the victim, however stale its stamp
        const std::vector<uint64_t> st = {1, 2, 30, 4, 5, 6, 7};
        const bool pin1[7] = {false, true, false, false, false, false, false};
        check(eviction_victim(st.data(), st.size(), 6, nullptr, pin1) == 3,
              "the oldest checkpoint is pinned: the next least recently used leaves");
        check(eviction_victim(st.data(), st.size(), 6, nullptr, nullptr) == 1, "no pin flags: the plain policy");
        const bool pin2[7] = {false, true, false, true, false, false, false};
        check(eviction_victim(st.data(), st.size(), 6, nullptr, pin2) == 4, "two pins: the third oldest leaves");
        const bool tail3[7] = {false, false, false, true, false, false, false};
        check(eviction_victim(st.data(), st.size(), 6, tail3, pin2) == 4, "a pinned item is not taken as the tail either");
        const bool all[4] = {false, true, true, true};
        const std::vector<uint64_t> s4 = {1, 9, 3, 5};
        check(eviction_victim(s4.data(), s4.size(), 4, nullptr, all) == 2, "everything pinned: the oldest pin leaves, never the root");
        check(eviction_victim(s4.data(), s4.size(), 1, nullptr, all) == 0, "a one-slot budget keeps no pin: the oldest leaves");
    }
    {   // #2409: a restore is refused only for a card that must create its copy stream and has no room for it
        constexpr uint64_t need = strata::program::conv_cache::kRestoreFirstUseMib << 20;
        const uint64_t loaded[] = {300ull << 20, 104ull << 20};         // a loaded card: far under --vram-reserve-mib
        const uint8_t ready[] = {1, 1};
        check(restore_fits(loaded, ready, 2, need), "cards with their stream already: any amount free admits (the #2409 review's case)");
        const uint8_t later_new[] = {1, 0};
        check(restore_fits(loaded, later_new, 2, need), "a card that must create its stream, with 104 MiB free: admitted");
        const uint64_t full[] = {300ull << 20, 20ull << 20};
        check(!restore_fits(full, later_new, 2, need), "a card that must create its stream, with 20 MiB free: reread instead");
        check(restore_fits(full, ready, 2, need), "the same full card with its stream: admitted");
        const uint64_t edge[] = {need, need - 1};
        const uint8_t none[] = {0, 0};
        check(!restore_fits(edge, none, 2, need), "one byte under the need refuses");
        check(restore_fits(edge, none, 1, need), "exactly the need admits");
        check(restore_fits(nullptr, none, 2, need) && restore_fits(full, nullptr, 2, need) && restore_fits(full, none, 0, need) &&
                  restore_fits(full, none, 2, 0),
              "no sample, no cards, no need: do not refuse");
    }
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
