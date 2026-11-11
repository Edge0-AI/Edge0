// Optional SAVE admission: free reconstructible host caches, never the live device state.
#pragma once

#include "strata/core/conversation_file.hpp"
#include "strata/core/conversation_memory.hpp"

namespace strata::core {

struct SessionSaveReclaimed {
    size_t checkpoints = 0, parked = 0, bytes = 0;
};

// Called BEFORE session_save_checkpoints selects a pointer into chain. Probe after each
// eviction: freed capacity is not necessarily physical RAM returned by the allocator.
// No allocation is needed here. Unknown telemetry / sizes cannot justify cache eviction.
// Preserve every explicit pin and exactly the checkpoint the file would have selected.
template<class Probe>
SessionSaveReclaimed session_save_reclaim(std::vector<ConversationCheckpoint>& chain,
                                        ConversationCache& cache, const SessionSaveLive& live,
                                        uint64_t floor, Probe available) {
    SessionSaveReclaimed out;
    const auto* selected = session_deepest_checkpoint(chain);
    const uint64_t need = session_save_peak_bytes(selected, live);
    auto room = available();
    auto done = [&] { return !room || need == UINT64_MAX || conversation_memory_admit(room, need, floor); };
    if (done()) return out;
    if (cache.retained_bytes()) {
        out.bytes += cache.retained_bytes();
        { auto unused = cache.take_reuse(); }
        room = available();
        if (done()) return out;
    }
    while (true) {
        const size_t before = cache.bytes();
        if (!cache.evict_oldest()) break; // pinned conversations stay parked
        ++out.parked;
        out.bytes += before - cache.bytes();
        room = available();
        if (done()) return out;
    }
    size_t keep = selected ? size_t(selected - chain.data()) : chain.size();
    // Erase from the back to preserve order, ties and the selected checkpoint's storage.
    for (size_t i = chain.size(); i-- > 0;) {
        if (i == keep || chain[i].pinned) continue;
        out.bytes += chain[i].bytes();
        chain.erase(chain.begin() + static_cast<std::ptrdiff_t>(i));
        if (i < keep) --keep;
        ++out.checkpoints;
        room = available();
        if (done()) break;
    }
    return out;
}

} // namespace strata::core
