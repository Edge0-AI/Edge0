#pragma once

#include <cstdint>
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

// An advisory serving-headroom target, matching the default conversation-cache
// floor. It is not a mandatory allocation or a prediction of peak memory.
inline constexpr uint64_t kArenaServingHeadroom = 2560ull << 20;

inline std::optional<uint64_t> arena_headroom_target_bytes(uint64_t experts, uint64_t blob) {
    constexpr auto max = std::numeric_limits<uint64_t>::max();
    if (experts > max - blob || experts + blob > max - kArenaServingHeadroom) return std::nullopt;
    return experts + blob + kArenaServingHeadroom;
}

inline bool arena_cgroup_shortfall(const HostMemory& memory, uint64_t required) {
    return memory.cgroup_limit < required || memory.cgroup_available < required;
}

}  // namespace strata::core::detail
