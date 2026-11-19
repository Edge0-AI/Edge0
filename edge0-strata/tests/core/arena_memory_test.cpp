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

    // #301: the reported shape - a 29 GiB group that holds the 26.66 GiB arena but leaves no serving
    // headroom - is refused under the group's hard limit, where a bare host only warns.
    HostMemory split;
    split.available = 28500 * MiB;
    split.cgroup_limit = 29 * GiB;
    split.cgroup_available = split.cgroup_limit - 512 * MiB;
    check(!arena_cgroup_shortfall(split, allocation), "the arena itself still fits the 29 GiB group");
    check(arena_cgroup_headroom_refusal(split, *target), "the same load is refused for the missing headroom");

    HostMemory boundary;
    boundary.available = 32 * GiB;
    boundary.cgroup_limit = 32 * GiB;
    boundary.cgroup_available = *target;
    check(!arena_cgroup_headroom_refusal(boundary, *target), "exactly the advised target fits");
    --boundary.cgroup_available;
    check(arena_cgroup_headroom_refusal(boundary, *target), "one byte below the advised target is refused");

    HostMemory host;
    host.available = 28500 * MiB;
    check(!arena_cgroup_headroom_refusal(host, *target), "a bare host keeps the warning, not a refusal");

    HostMemory roomy;
    roomy.available = 40 * GiB;
    roomy.cgroup_limit = 64 * GiB;
    roomy.cgroup_available = 40 * GiB;
    check(!arena_cgroup_headroom_refusal(roomy, *target), "a roomy group admits the arena and its headroom");

    // A batch job that never opens a conversation opts out: the target becomes the allocation itself.
#if defined(_WIN32)
    check(_putenv_s("STRATA_ARENA_HEADROOM_MIB", "0") == 0, "set STRATA_ARENA_HEADROOM_MIB");
#else
    check(setenv("STRATA_ARENA_HEADROOM_MIB", "0", 1) == 0, "set STRATA_ARENA_HEADROOM_MIB");
#endif
    check(arena_headroom_target_bytes(experts, blob) == allocation, "headroom 0 keeps the allocation as the target");
    check(!arena_cgroup_headroom_refusal(split, allocation), "headroom 0 admits a bare allocation");
    check(arena_serving_headroom() == 0, "headroom 0 is reported");
#if defined(_WIN32)
    check(_putenv_s("STRATA_ARENA_HEADROOM_MIB", "") == 0, "unset STRATA_ARENA_HEADROOM_MIB");
#else
    check(unsetenv("STRATA_ARENA_HEADROOM_MIB") == 0, "unset STRATA_ARENA_HEADROOM_MIB");
#endif
    check(arena_headroom_target_bytes(experts, blob) == *target, "the default headroom returns");

    constexpr uint64_t max = std::numeric_limits<uint64_t>::max();
    check(!arena_headroom_target_bytes(max, 1), "arena plus blob overflow is rejected");
    check(!arena_headroom_target_bytes(max - kArenaServingHeadroom, 1), "headroom target overflow is rejected");
    check(arena_headroom_target_bytes(max - kArenaServingHeadroom, 0) == max, "largest representable target");
    check(arena_headroom_target_bytes(0, 0) == kArenaServingHeadroom, "zero-sized arena retains advisory target");
    std::printf("arena_memory_test: %d checks passed\n", checks);
}
