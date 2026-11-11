// CPU only: pressure is injected through telemetry, not by exhausting the host.
#include "strata/core/session_save_reclaim.hpp"
#include <cstdio>
#include <cstdlib>
#include <new>

static bool deny_alloc = false;
void* operator new(std::size_t n) {
    if (deny_alloc) throw std::bad_alloc();
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

using namespace strata::core;
static int checks_run = 0;
static void check(bool ok, const char* label) {
    ++checks_run;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
static ConversationCheckpoint checkpoint(size_t n, bool pinned = false) {
    ConversationCheckpoint c;
    c.ids.assign(n, 42); c.imgs = {{1, 99}};
    c.gdn.assign(4096, 1); c.ple = {2}; c.tails = {3}; c.dead = {4}; c.block_pos = {5};
    c.used = n; c.pinned = pinned;
    return c;
}
static bool same(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    return a.ids == b.ids && a.imgs == b.imgs && a.gdn == b.gdn && a.ple == b.ple &&
           a.tails == b.tails && a.dead == b.dead && a.block_pos == b.block_pos &&
           a.used == b.used && a.pinned == b.pinned && a.stage_parts.size() == b.stage_parts.size();
}
static void park(ConversationCache& cache, int tokens, bool pin = false) {
    SavedConversation c; c.live = checkpoint(tokens);
    c.checkpoints.push_back(checkpoint(tokens - 1, pin));
    // Distinct histories, so put() does not remove superseded entries.
    c.live.ids[0] = tokens; c.checkpoints[0].ids[0] = tokens;
    check(cache.put(std::move(c)), "fixture parked");
}
int main() {
    SessionSaveLive live; live.state_bytes = 1000; live.tokens = 100; live.images = 1; live.kv_layers = 2;
    const uint64_t floor = 1u << 20;
    // Every selected position, first-tie preference, and explicit pins (a pin can be shallower).
    for (size_t selected = 0; selected < 4; ++selected) for (bool pin : {false, true}) {
        std::vector<ConversationCheckpoint> chain;
        for (size_t i = 0; i < 4; ++i) chain.push_back(checkpoint(i == selected ? 300 : 100, pin && i == selected));
        if (pin) chain[(selected + 1) % 4].ids.resize(500); // deeper, unpinned
        auto before = *session_deepest_checkpoint(chain);
        const auto* storage = session_deepest_checkpoint(chain)->gdn.data();
        ConversationCache cache(1u << 20, 10);
        park(cache, 10); park(cache, 20, true); park(cache, 30);
        std::vector<ConversationKv> kv(1); kv[0].k.resize(16384); cache.retain(std::move(kv), 100);
        const size_t bytes_before = cache.bytes();
        size_t chain_bytes = 0; for (const auto& c : chain) chain_bytes += c.bytes();
        deny_alloc = true;
        auto freed = session_save_reclaim(chain, cache, live, floor, [] { return std::optional<uint64_t>(0); });
        deny_alloc = false;
        check(chain.size() == 1 && same(chain[0], before), "selected checkpoint unchanged under pressure");
        check(chain[0].gdn.data() == storage, "checkpoint moved without copying storage");
        check(freed.checkpoints == 3 && freed.parked == 2 && cache.size() == 1, "pins survive, other entries reclaimed");
        check(cache.evictions() == 2 && cache.retained_bytes() == 0, "cache accounting and retained buffers");
        check(freed.bytes == bytes_before - cache.bytes() + chain_bytes - chain[0].bytes(), "released byte accounting");
        check(session_deepest_checkpoint(chain)->ids == before.ids, "same file selection after reclaim");
        auto again = session_save_reclaim(chain, cache, live, floor, [] { return std::optional<uint64_t>(0); });
        check(again.bytes == 0 && again.checkpoints == 0 && again.parked == 0, "exhausted reclaim is idempotent");
    }
    // No-pressure path, unknown telemetry, overflowed estimate: no cache mutation.
    for (int mode = 0; mode < 4; ++mode) {
        std::vector<ConversationCheckpoint> chain{checkpoint(100), checkpoint(300)};
        ConversationCache cache(1u << 20, 10); park(cache, 10);
        const auto before = chain;
        const auto bytes = cache.bytes(); auto sl = live;
        if (mode == 2) sl.state_bytes = UINT64_MAX;
        const uint64_t need = session_save_peak_bytes(session_deepest_checkpoint(chain), sl);
        auto r = session_save_reclaim(chain, cache, sl, floor, [&]() -> std::optional<uint64_t> {
            if (mode == 1) return std::nullopt;
            return mode == 2 ? 0 : need + floor + (mode == 3 ? 1 : 0);
        });
        check(!r.bytes && !r.parked && !r.checkpoints && cache.bytes() == bytes, "no eviction without known pressure");
        check(chain.size() == 2 && same(chain[0], before[0]) && same(chain[1], before[1]), "unchanged order/content");
    }
    // Admit after any individual step; never clear the whole cache when one release suffices.
    for (int sufficient_after = 1; sufficient_after <= 4; ++sufficient_after) {
        std::vector<ConversationCheckpoint> chain{checkpoint(100), checkpoint(300), checkpoint(300)};
        auto* selected_storage = chain[1].gdn.data();
        ConversationCache cache(1u << 20, 10); park(cache, 10); park(cache, 20);
        std::vector<ConversationKv> kv(1); kv[0].v.resize(100); cache.retain(std::move(kv), 100);
        const auto need = session_save_peak_bytes(session_deepest_checkpoint(chain), live);
        int probes = 0;
        auto r = session_save_reclaim(chain, cache, live, floor, [&]() -> std::optional<uint64_t> {
            return probes++ >= sufficient_after ? need + floor : 0;
        });
        check(probes == sufficient_after + 1, "stop immediately once RAM meets unchanged floor");
        check(r.parked == size_t(std::min(sufficient_after - 1, 2)), "one-at-a-time parked eviction");
        check(r.checkpoints == size_t(sufficient_after == 4), "checkpoint eviction only after parked cache");
        check(session_deepest_checkpoint(chain)->gdn.data() == selected_storage, "first deepest tie retained");
    }
    {
        std::vector<ConversationCheckpoint> chain{checkpoint(100, true), checkpoint(300, true), checkpoint(500)};
        ConversationCache cache(0, 0);
        auto r = session_save_reclaim(chain, cache, live, floor, [] { return std::optional<uint64_t>(0); });
        check(r.checkpoints == 1 && chain.size() == 2 && chain[0].pinned && chain[1].pinned, "all pins survive in original order");
        std::vector<ConversationCheckpoint> out; std::string why;
        check(!session_save_checkpoints(chain, live, [](uint64_t, std::string& w) { w = "low RAM"; return false; }, out, why)
              && out.empty() && why == "low RAM", "still refused before copying when reclamation is insufficient");
    }
    {
        std::vector<ConversationCheckpoint> chain;
        ConversationCache cache(0, 0);
        auto r = session_save_reclaim(chain, cache, live, floor, [] { return std::optional<uint64_t>(0); });
        check(!r.bytes && !r.checkpoints && !r.parked, "empty chain and disabled cache");
    }
    std::printf("session_save_reclaim: %d checks passed\n", checks_run);
}
