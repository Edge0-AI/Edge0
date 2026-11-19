#include "strata/core/arena_memory.hpp"

#include <cstdio>
#include <cstdlib>
#include <limits>

using namespace strata::core::detail;
constexpr uint64_t MiB = 1ull << 20, GiB = 1ull << 30;
int checks = 0;
void check(bool condition, const char* message) {
    ++checks;
    if (!condition) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}

int main() {
    // #301: approximate the reported arena with integer MiB, plus its padded blob.
    constexpr uint64_t experts = 27300 * MiB, blob = 2 * MiB;
    constexpr uint64_t allocation = experts + blob;
    const auto target = arena_headroom_target_bytes(experts, blob);
    check(target && *target == allocation + kArenaServingHeadroom, "include advisory serving headroom");
    HostMemory m;
    m.available = 28500 * MiB;
    m.cgroup_limit = 29 * GiB;
    m.cgroup_available = m.cgroup_limit - 512 * MiB;
    check(experts + blob < m.cgroup_limit && experts + blob < m.available, "the old arena-only checks pass");
    check(!arena_cgroup_shortfall(m, allocation), "a fitting 29 GiB allocation is not refused for advisory headroom");
    check(m.available < *target, "a fitting allocation with little serving headroom warrants a warning");

    m.cgroup_limit = 32 * GiB;
    m.cgroup_available = 26 * GiB;
    check(arena_cgroup_shortfall(m, allocation), "already charged memory matters even when the total cap fits");
    m.cgroup_available = allocation;
    check(!arena_cgroup_shortfall(m, allocation), "exact remaining-capacity boundary fits");
    --m.cgroup_available;
    check(arena_cgroup_shortfall(m, allocation), "one byte below boundary does not fit");
    m.cgroup_available = 0;
    check(arena_cgroup_shortfall(m, allocation), "exhausted group is a known shortfall");

    m = HostMemory{};
    m.available = 1 * GiB;
    check(!arena_cgroup_shortfall(m, allocation), "bare-host pressure does not become a hard refusal");
    check(m.available < *target, "bare-host pressure still needs a warning/wait");
    m.cgroup_limit = 64 * GiB;
    m.cgroup_available = 40 * GiB;
    check(!arena_cgroup_shortfall(m, allocation), "physical pressure with a roomy group is warning-only");
    m.available = 40 * GiB;
    check(!arena_cgroup_shortfall(m, allocation), "roomy group and host admit the allocation");

    constexpr uint64_t max = std::numeric_limits<uint64_t>::max();
    check(!arena_headroom_target_bytes(max, 1), "arena plus blob overflow is rejected");
    check(!arena_headroom_target_bytes(max - kArenaServingHeadroom, 1), "headroom target overflow is rejected");
    check(arena_headroom_target_bytes(max - kArenaServingHeadroom, 0) == max, "largest representable target");
    check(arena_headroom_target_bytes(0, 0) == kArenaServingHeadroom, "zero-sized arena retains advisory target");
    std::printf("arena_memory_test: %d checks passed\n", checks);
}
