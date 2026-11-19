#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>

namespace strata::core::detail {

struct HostMemory {
    // Physical RAM available, bounded by the remaining capacity of every cgroup.
    uint64_t available = 0;
    // Tightest absolute cgroup limit; UINT64_MAX when none is reported.
    uint64_t cgroup_limit = ~uint64_t{0};
    // Windows commit capacity; UINT64_MAX when the platform does not report it.
    uint64_t commit = ~uint64_t{0};
    // Remaining capacity under the cgroup limits, before bounding by physical
    // MemAvailable. Kept separately so host pressure is still only a warning.
    uint64_t cgroup_available = ~uint64_t{0};
};

// The serving headroom an arena must leave free, matching the default conversation-cache
// floor. It is not a mandatory allocation or a prediction of peak memory: on a bare host it is
// only a warning. STRATA_ARENA_HEADROOM_MIB overrides it in MiB; 0 turns the check off for a
// caller that never opens a conversation (a batch job, a benchmark run).
inline constexpr uint64_t kArenaServingHeadroom = 2560ull << 20;

inline uint64_t arena_serving_headroom() {
    const char* v = std::getenv("STRATA_ARENA_HEADROOM_MIB");
    if (v == nullptr || v[0] == '\0') return kArenaServingHeadroom;
    char* end = nullptr;
    const unsigned long long mib = std::strtoull(v, &end, 10);
    if (end == v || *end != '\0' || mib > 65536ull) return kArenaServingHeadroom;   // unreadable: the default
    return (uint64_t) mib << 20;
}

inline std::optional<uint64_t> arena_headroom_target_bytes(uint64_t experts, uint64_t blob) {
    constexpr auto max = std::numeric_limits<uint64_t>::max();
    const uint64_t headroom = arena_serving_headroom();
    if (experts > max - blob || experts + blob > max - headroom) return std::nullopt;
    return experts + blob + headroom;
}

inline bool arena_cgroup_shortfall(const HostMemory& memory, uint64_t required) {
    return memory.cgroup_limit < required || memory.cgroup_available < required;
}

// #301: a cgroup limit is hard - the kernel stops the group instead of handing the pages back - so an arena
// that leaves the group without serving headroom is refused, where the same shape on a bare host only warns
// (its pages can be reclaimed or other programs closed: recommend, do not force).
inline bool arena_cgroup_headroom_refusal(const HostMemory& memory, uint64_t advised) {
    if (memory.cgroup_limit == std::numeric_limits<uint64_t>::max()) return false;   // no limit in force
    return std::min(memory.cgroup_limit, memory.cgroup_available) < advised;
}

}  // namespace strata::core::detail
