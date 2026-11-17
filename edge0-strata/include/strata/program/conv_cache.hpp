// include/strata/program/conv_cache.hpp - the conversation cache's retention policy: which checkpoint
// leaves when the cache is over its slot budget.
//
// The cache holds conversation checkpoints: snapshots of the running state (the GDN recurrence, the PLE
// history, the QSA indexer tails - `ConvCheckpoint` in generate.cpp) that a request resumes its prompt
// from instead of reading those tokens again.  The request path keeps only the checkpoints whose tokens
// are a prefix of the current prompt - any other checkpoint's positional cells have been overwritten,
// because the KV cache is one arena that holds one branch of history at a time - so at any moment the
// retained checkpoints are a CHAIN: sorted by length, each a prefix of the next.  That is a radix cache's
// tree collapsed onto the one branch of history the session can hold.
//
// The chain's root is therefore the deepest point every request so far has shared - in practice the end
// of the system prompt, which every new chat of the same client mounts through - and it is exactly the
// node a radix cache keeps alive while its leaves rotate.  First-in-first-out kept it only by accident of
// being first: after `prompt_cache` newer checkpoints it was gone, and the next chat read the whole
// prefix again at prefill speed.  So:
//
//   * the root is PINNED: dropping it frees one slot's ~118 MB and costs every future conversation that
//     shares the prefix a full re-read of it (a 30K system prompt is ~30 s at ~1,000 tok/s);
//   * every other slot rotates by least recent use.  A checkpoint's stamp advances when it is created,
//     when a new checkpoint lands on its length, and when a request mounts through it.
//
// With a budget below two slots there is no room to keep root and leaf apart, so the pin switches off and
// the oldest checkpoint leaves - the previous behaviour, one slot = the newest point only.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::program::conv_cache {

/// The index in `stamps` of the chain item to drop once the chain holds more than `cap` items.  `stamps`
/// are the items' last-use stamps; the caller owns the chain and erases the returned index.  Pure and
/// deterministic so conv_cache_test.cpp can walk the scenarios by hand.
inline size_t eviction_victim(const uint64_t* stamps, size_t n, int64_t cap) {
    if (cap < 2 || n < 2) return 0;   // no room for root and leaf: the pin is off, the oldest leaves
    size_t v = 1;                     // the root (0) is pinned; least recent use among the rest
    for (size_t i = 2; i < n; ++i)
        if (stamps[i] < stamps[v]) v = i;
    return v;
}

/// --prompt-cache-tail: the extra checkpoint near the prompt's end (`tail[i]`) is the first to go - it serves
/// only a branch of the last request - unless it is the newest item (the one just saved: `n > 1` and the newest
/// stamp).  With no such item this is the plain policy.  Never the root.
inline size_t eviction_victim(const uint64_t* stamps, size_t n, int64_t cap, const bool* tail) {
    if (tail != nullptr && n >= 2 && cap >= 2) {
        size_t newest = 0;
        for (size_t i = 1; i < n; ++i)
            if (stamps[i] > stamps[newest]) newest = i;
        for (size_t i = 1; i < n; ++i)
            if (tail[i] && i != newest) return i;
    }
    return eviction_victim(stamps, n, cap);
}

/// The pin=N shared prefixes (`pinned[i]`): they are never the victim, whatever their stamp - the queries that branch
/// from one keep coming back to it, and a least-recently-used rule would drop it while they still do.  The root and
/// the tail rule are as above.  When every other item is pinned too, the policy falls back to the oldest pinned one:
/// the cache still keeps to its `cap` (the caller limits how many pins it takes, so this is a backstop).
inline size_t eviction_victim(const uint64_t* stamps, size_t n, int64_t cap, const bool* tail, const bool* pinned) {
    if (pinned == nullptr) return eviction_victim(stamps, n, cap, tail);
    if (cap < 2 || n < 2) return 0;
    size_t newest = 0;
    for (size_t i = 1; i < n; ++i)
        if (stamps[i] > stamps[newest]) newest = i;
    if (tail != nullptr)
        for (size_t i = 1; i < n; ++i)
            if (tail[i] && !pinned[i] && i != newest) return i;
    size_t v = n;                    // the least recently used among the unpinned non-root items
    for (size_t i = 1; i < n; ++i)
        if (!pinned[i] && (v == n || stamps[i] < stamps[v])) v = i;
    if (v != n) return v;
    v = 1;                           // all pinned: the oldest of them (never the root, index 0)
    for (size_t i = 2; i < n; ++i)
        if (stamps[i] < stamps[v]) v = i;
    return v;
}

/// #2409: what a conversation restore can allocate on a card. It writes into session buffers that already exist; the one
/// thing it may create is the calling thread's private copy stream for that card (conversation_copy.hpp), which it takes
/// from the pool of finished threads when there is one, and the first launch of the two small residency kernels loads
/// their code. A stream and an event cost well under 2 MiB where it was measured (conversation_copy_streams_test prints
/// it); 64 MiB covers both many times over and is far below the reserves (--vram-reserve-mib 700), so the check can only
/// fire on a card that is really full.
constexpr uint64_t kRestoreFirstUseMib = 64;

/// Whether a parked snapshot can be restored without a first-use allocation failing: every card for which the restore
/// would have to create its copy stream (`stream_ready[i]` false) must have `need_bytes` free. A card that already has a
/// stream needs nothing, so it never refuses - unlike a floor on `--vram-reserve-mib`, which a loaded card is always under
/// (#2409's review). A missing sample (`free_bytes` null) does not refuse.
inline bool restore_fits(const uint64_t* free_bytes, const uint8_t* stream_ready, size_t n, uint64_t need_bytes) {
    if (free_bytes == nullptr || stream_ready == nullptr || need_bytes == 0) return true;
    for (size_t i = 0; i < n; ++i)
        if (!stream_ready[i] && free_bytes[i] < need_bytes) return false;
    return true;
}

/// --prompt-cache-spread: the item whose removal leaves the smallest gap for its distance from the newest, so the kept
/// points thin out geometrically towards the root and an edit d tokens before the end reads about d again.  `lens`:
/// the chain's token counts.  Never the root, the newest or a pinned item; the tail rule first, as above.
inline size_t spread_victim(const int64_t* lens, const uint64_t* stamps, size_t n, int64_t cap, const bool* tail,
                            const bool* pinned) {
    if (n >= 3 && cap >= 2) {
        auto candidate = [&](size_t i) { return !(pinned && pinned[i]); };
        if (tail != nullptr)
            for (size_t i = 1; i + 1 < n; ++i)
                if (tail[i] && candidate(i)) return i;
        size_t v = n;
        double best = 0;
        for (size_t i = 1; i + 1 < n; ++i) {
            if (!candidate(i)) continue;
            const double gap = double(lens[i + 1] - lens[i - 1]) / double(lens[n - 1] - lens[i - 1]);
            if (v == n || gap < best) { v = i; best = gap; }
        }
        if (v != n) return v;
    }
    return eviction_victim(stamps, n, cap, tail, pinned);
}

}  // namespace strata::program::conv_cache
