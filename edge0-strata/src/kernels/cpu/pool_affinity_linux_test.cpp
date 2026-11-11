#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "strata/kernels/cpu/pool.hpp"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include "pool_affinity_linux.hpp"

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

namespace cpu = strata::kernels::cpu;

class DynamicCpuSet {
public:
    explicit DynamicCpuSet(size_t count)
        : count_(count), bytes_(CPU_ALLOC_SIZE(count)), set_(CPU_ALLOC(count)) {
        if (set_) CPU_ZERO_S(bytes_, set_);
    }

    ~DynamicCpuSet() {
        if (set_) CPU_FREE(set_);
    }

    DynamicCpuSet(const DynamicCpuSet&) = delete;
    DynamicCpuSet& operator=(const DynamicCpuSet&) = delete;

    cpu_set_t* get() { return set_; }
    const cpu_set_t* get() const { return set_; }
    size_t bytes() const { return bytes_; }
    size_t count() const { return count_; }
    explicit operator bool() const { return set_ != nullptr; }

private:
    size_t count_;
    size_t bytes_;
    cpu_set_t* set_;
};

static size_t initial_cpu_count() {
    const long configured = sysconf(_SC_NPROCESSORS_CONF);
    return (std::max)(size_t(128), configured > 0 ? (size_t) configured : size_t(0));
}

// Query with CPU_ALLOC rather than cpu_set_t: this must observe CPUs beyond CPU_SETSIZE too.
static bool get_affinity(std::vector<int>& ids) {
    size_t count = initial_cpu_count();
    for (int attempt = 0; attempt < 8; ++attempt) {
        DynamicCpuSet set(count);
        if (!set) return false;
        const int error = pthread_getaffinity_np(pthread_self(), set.bytes(), set.get());
        if (error == 0) {
            ids.clear();
            for (size_t i = 0; i < set.count(); ++i)
                if (CPU_ISSET_S(i, set.bytes(), set.get())) ids.push_back((int) i);
            return !ids.empty();
        }
        if (error != EINVAL || count > (size_t(1) << 20)) {
            std::fprintf(stderr, "pthread_getaffinity_np failed: %s\n", std::strerror(error));
            return false;
        }
        count *= 2;
    }
    std::fprintf(stderr, "pthread_getaffinity_np still needs a larger CPU set\n");
    return false;
}

static bool set_affinity(const std::vector<int>& ids) {
    if (ids.empty()) return false;
    const int highest = *std::max_element(ids.begin(), ids.end());
    if (highest < 0) return false;
    const size_t count = (std::max)(initial_cpu_count(), (size_t) highest + 1);
    DynamicCpuSet set(count);
    if (!set) return false;
    for (int id : ids) {
        if (id < 0 || (size_t) id >= count) return false;
        CPU_SET_S((size_t) id, set.bytes(), set.get());
    }
    const int error = pthread_setaffinity_np(pthread_self(), set.bytes(), set.get());
    if (error != 0) {
        std::fprintf(stderr, "pthread_setaffinity_np failed: %s\n", std::strerror(error));
        return false;
    }
    return true;
}

class OriginalAffinity {
public:
    bool capture() { return get_affinity(ids_); }
    const std::vector<int>& ids() const { return ids_; }
    bool restore() { return ids_.empty() || set_affinity(ids_); }
    ~OriginalAffinity() { if (!ids_.empty()) set_affinity(ids_); }

private:
    std::vector<int> ids_;
};

static bool expect_affinity(const std::vector<int>& expected, const char* label) {
    std::vector<int> actual;
    if (!get_affinity(actual)) {
        std::fprintf(stderr, "Linux affinity test: %s: cannot query current affinity\n", label);
        return false;
    }
    if (actual != expected) {
        std::fprintf(stderr, "Linux affinity test: %s: expected %zu CPUs, observed %zu\n",
                     label, expected.size(), actual.size());
        return false;
    }
    return true;
}

static bool round_trip_full_mask(const std::vector<int>& allowed) {
    const int target = allowed.front();
    const auto previous = cpu::pin_current_thread(target);
    const bool pinned = previous.valid && expect_affinity({target}, "pin from full allowed mask");
    cpu::restore_thread_affinity(previous);
    return pinned && expect_affinity(allowed, "restore full allowed mask");
}

static bool topology_respects_current_mask() {
    std::vector<int> allowed;
    if (!get_affinity(allowed)) return false;
    const auto topology = cpu::detect_cpu_topology(false);
    if (topology.worker_cores.empty()) return false;
    for (int core : topology.worker_cores) {
        if (!std::binary_search(allowed.begin(), allowed.end(), core)) {
            std::fprintf(stderr, "Linux affinity test: topology selected disallowed CPU %d\n", core);
            return false;
        }
    }
    return true;
}

static bool round_trip_sparse_nested(const std::vector<int>& allowed) {
    if (allowed.size() < 2) {
        std::puts("SKIP sparse/nested affinity: fewer than two allowed CPUs");
        return true;
    }
    const std::vector<int> sparse{allowed.front(), allowed.back()};
    if (!set_affinity(sparse) || !expect_affinity(sparse, "establish sparse mask") ||
        !topology_respects_current_mask()) return false;

    const auto outer = cpu::pin_current_thread(sparse.front());
    const bool outer_pinned = outer.valid && expect_affinity({sparse.front()}, "outer pin");
    if (!outer_pinned) {
        cpu::restore_thread_affinity(outer);
        return false;
    }

    const auto inner = cpu::pin_current_thread(sparse.back());
    const bool inner_pinned = inner.valid && expect_affinity({sparse.back()}, "inner pin");
    cpu::restore_thread_affinity(inner);
    const bool outer_restored = expect_affinity({sparse.front()}, "restore inner pin to outer pin");
    cpu::restore_thread_affinity(outer);
    const bool sparse_restored = expect_affinity(sparse, "restore outer pin to sparse mask");
    const bool full_restored = set_affinity(allowed) && expect_affinity(allowed, "restore full mask after sparse test");
    return inner_pinned && outer_restored && sparse_restored && full_restored;
}

static bool round_trip_high_cpu(const std::vector<int>& allowed, bool require_high) {
    const auto it = std::find_if(allowed.begin(), allowed.end(), [](int id) { return id >= 64; });
    if (it == allowed.end()) {
        if (require_high) {
            std::fprintf(stderr, "Linux affinity test: --require-high-cpu requested, but no allowed CPU ID is >= 64\n");
            return false;
        }
        std::puts("SKIP high-CPU affinity path: no allowed CPU ID is >= 64 (use --require-high-cpu to require it)");
        return true;
    }
    const int target = allowed.front() == *it ? (allowed.size() > 1 ? allowed.back() : *it) : allowed.front();
    if (target == *it) {
        if (require_high) {
            std::fprintf(stderr, "Linux affinity test: --require-high-cpu needs at least two allowed CPUs to test restoration\n");
            return false;
        }
        std::puts("SKIP high-CPU affinity round trip: only one CPU is allowed, so no meaningful pin transition is possible");
        return true;
    }

    // Restrict the caller to one high-numbered CPU, then pin to a different originally allowed CPU.
    // Restoring the high-only mask catches truncation even when the original process mask is sparse.
    const std::vector<int> high_only{*it};
    if (!set_affinity(high_only) || !expect_affinity(high_only, "establish high-only mask") ||
        !topology_respects_current_mask()) return false;
    const auto previous = cpu::pin_current_thread(target);
    const bool pinned = previous.valid && expect_affinity({target}, "pin away from high-only mask");
    cpu::restore_thread_affinity(previous);
    const bool restored = pinned && expect_affinity(high_only, "restore high-only mask");
    if (restored) std::printf("High-CPU affinity round trip: saved CPU %d, pin target %d\n", *it, target);
    return restored;
}

static bool negative_cpu_is_noop() {
    std::vector<int> before;
    if (!get_affinity(before)) return false;
    const auto invalid = cpu::pin_current_thread(-1);
    std::vector<int> after;
    const bool unchanged = get_affinity(after) && before == after;
    cpu::restore_thread_affinity(invalid);
    return !invalid.valid && unchanged && expect_affinity(before, "negative CPU no-op");
}

static bool worker_thread_can_pin(int target) {
    bool ok = false;
    std::thread worker([&] {
        std::vector<int> before;
        if (!get_affinity(before)) return;
        std::vector<unsigned long> saved;
        if (cpu::detail::get_thread_affinity(saved) != 0) return;
        const int pin_error = cpu::detail::pin_thread_to_cpu(target);
        const bool pinned = pin_error == 0 && expect_affinity({target}, "worker thread helper pin") &&
                            sched_getcpu() == target;
        const int restore_error = cpu::detail::set_thread_affinity(saved);
        ok = pinned && restore_error == 0 && expect_affinity(before, "worker thread helper restore");
    });
    worker.join();
    return ok;
}

static bool failed_pin_preserves_affinity() {
    std::vector<int> before;
    if (!get_affinity(before)) return false;
    // An ID outside the caller's current mask may still be online and legal, so it is not a
    // reliable failure case. INT_MAX is rejected before allocating or changing any CPU set.
    const int error = cpu::detail::pin_thread_to_cpu(INT_MAX);
    std::vector<int> after;
    const bool helper_unchanged = get_affinity(after) && before == after;
    if (error == 0) {
        std::fprintf(stderr, "Linux affinity test: pin_thread_to_cpu(INT_MAX) unexpectedly succeeded\n");
        return false;
    }
    const auto invalid = cpu::pin_current_thread(INT_MAX);
    cpu::restore_thread_affinity(invalid);
    std::vector<int> after_public;
    const bool public_unchanged = get_affinity(after_public) && before == after_public;
    if (!helper_unchanged || invalid.valid || !public_unchanged)
        std::fprintf(stderr, "Linux affinity test: failed pin changed the observed affinity or returned a valid snapshot\n");
    return helper_unchanged && !invalid.valid && public_unchanged;
}

int main(int argc, char** argv) {
    bool require_high = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--require-high-cpu") == 0) require_high = true;
        else {
            std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return 2;
        }
    }

    OriginalAffinity original;
    if (!original.capture()) {
        std::fprintf(stderr, "Linux affinity test: cannot capture original thread affinity\n");
        return 1;
    }
    const auto allowed = original.ids();

    bool ok = round_trip_full_mask(allowed) && worker_thread_can_pin(allowed.back());
    if (ok) ok = round_trip_sparse_nested(allowed);
    if (ok) ok = round_trip_high_cpu(allowed, require_high);
    if (ok) ok = negative_cpu_is_noop();
    if (ok) ok = failed_pin_preserves_affinity();
    const bool restored = original.restore() && expect_affinity(allowed, "final original affinity cleanup");
    if (!ok || !restored) return 1;

    std::printf("Linux affinity test: full, sparse/nested, worker helper, invalid-target, failed-pin, and cleanup checks passed (%zu allowed CPUs)\n",
                allowed.size());
    return 0;
}

#else

int main() {
    std::puts("SKIP Linux affinity test: Linux is required");
    return 0;
}

#endif
