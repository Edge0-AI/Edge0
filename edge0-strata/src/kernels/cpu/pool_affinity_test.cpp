#include "strata/kernels/cpu/pool.hpp"
#include "pool_affinity_win.hpp"

#include <algorithm>
#include <cstdio>
#include <thread>
#include <vector>

namespace cpu = strata::kernels::cpu;

static bool check(bool ok, const char* message) {
    if (!ok) std::fprintf(stderr, "Windows affinity test: %s (error %lu)\n", message, (unsigned long) GetLastError());
    return ok;
}

static bool select(const std::vector<ULONG>& ids) {
    return SetThreadSelectedCpuSets(GetCurrentThread(), ids.empty() ? nullptr : ids.data(), (ULONG) ids.size()) != 0;
}

static bool selected_is(std::vector<ULONG> expected) {
    std::vector<ULONG> actual;
    if (!cpu::detail::get_thread_cpu_sets(actual)) return false;
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());
    return actual == expected;
}

// Establish reachability before the round trip: CPU Sets are soft affinity and cannot override an
// existing job/hard-affinity restriction. Afterwards, require the same observable reachability.
static bool observe(int core) {
    for (int i = 0; i < 200; ++i) {
        Sleep(1);
        PROCESSOR_NUMBER current{};
        GetCurrentProcessorNumberEx(&current);
        if ((int) current.Group * 64 + current.Number == core) return true;
    }
    return false;
}

static bool host_round_trips(const std::vector<int>& cores, const std::vector<ULONG>& ids) {
    if (!check(select({}), "clear initial thread CPU Sets")) return false;
    std::vector<bool> reachable;
    for (size_t i = 0; i < cores.size(); ++i) {
        if (!check(select({ids[i]}), "select baseline CPU Set")) return false;
        reachable.push_back(observe(cores[i]));
    }
    if (!check(select({}), "clear baseline CPU Sets")) return false;

    for (size_t i = 0; i < cores.size(); ++i) {
        const auto previous = cpu::pin_current_thread(cores[i]);
        const bool pinned = previous.valid && previous.cpu_sets.empty() && selected_is({ids[i]});
        cpu::restore_thread_affinity(previous);
        if (!check(pinned && selected_is({}), "restore empty thread assignment")) return false;
    }

    // GetThreadGroupAffinity alone misses a lost implicit all-group state. Selecting each CPU Set
    // again must still permit execution there, without ever setting a hard group affinity.
    for (size_t i = 0; i < cores.size(); ++i) {
        if (!check(select({ids[i]}), "select CPU Set after restore")) return false;
        if (reachable[i]) {
            if (!check(observe(cores[i]), "lost CPU/group eligibility after host restore")) return false;
            std::printf("Host restore: still executes on group %d processor %d\n", cores[i] / 64, cores[i] & 63);
        } else {
            std::printf("SKIP scheduling observation for group %d: not reachable before test\n", cores[i] / 64);
        }
    }

    // A nonempty selection may span groups. Nested host selections must restore the entire set.
    if (!check(select(ids), "set existing multi-CPU selection")) return false;
    const auto outer = cpu::pin_current_thread(cores.front());
    const auto inner = cpu::pin_current_thread(cores.back());
    const bool inner_ok = inner.valid && selected_is({ids.back()});
    cpu::restore_thread_affinity(inner);
    const bool outer_ok = outer.valid && selected_is({ids.front()});
    cpu::restore_thread_affinity(outer);
    if (!check(inner_ok && outer_ok && selected_is(ids), "restore nested / existing CPU Set selections")) return false;
    const auto invalid = cpu::pin_current_thread(-1);
    cpu::restore_thread_affinity(invalid);
    if (!check(!invalid.valid && selected_is(ids), "invalid target must not mutate selection")) return false;
    return check(select({}), "clear final host selection");
}

static bool inherited_default(int core, ULONG id, int temporary_core, ULONG temporary_id) {
    ULONG count = 0;
    if (!GetProcessDefaultCpuSets(GetCurrentProcess(), nullptr, 0, &count) &&
        GetLastError() != ERROR_INSUFFICIENT_BUFFER) return false;
    std::vector<ULONG> saved(count);
    if (count && !GetProcessDefaultCpuSets(GetCurrentProcess(), saved.data(), count, &count)) return false;
    saved.resize(count);
    if (!check(SetProcessDefaultCpuSets(GetCurrentProcess(), &id, 1) != 0, "set process default")) return false;
    bool ok = false;
    std::thread host([&] {
        if (!select({})) return;
        const bool reachable = observe(core);
        const auto previous = cpu::pin_current_thread(temporary_core);
        const bool overridden = previous.valid && selected_is({temporary_id});
        cpu::restore_thread_affinity(previous);
        ULONG inherited = 0, required = 0;
        ok = overridden && previous.cpu_sets.empty() && selected_is({}) &&
             GetProcessDefaultCpuSets(GetCurrentProcess(), &inherited, 1, &required) &&
             required == 1 && inherited == id && (!reachable || observe(core));
    });
    host.join();
    const bool restored = SetProcessDefaultCpuSets(GetCurrentProcess(), saved.empty() ? nullptr : saved.data(),
                                                  (ULONG) saved.size()) != 0;
    return check(ok && restored, "restore inherited process default");
}

// --host-core sibling: the host on the first layout's host core's SMT sibling, which no worker takes, and the first
// logical processor left to nothing; `first` on a core without SMT.  Every affinity mode.
static bool host_sibling_layout() {
    bool ok = true;
    for (const auto affinity : {cpu::PoolAffinity::All, cpu::PoolAffinity::Auto, cpu::PoolAffinity::PCores}) {
        cpu::set_host_core(cpu::HostCore::First);
        const cpu::CpuTopology first = cpu::detect_cpu_topology(true, affinity);
        cpu::set_host_core(cpu::HostCore::Sibling);
        const cpu::CpuTopology sib = cpu::detect_cpu_topology(true, affinity);
        cpu::set_host_core(cpu::HostCore::First);
        const auto has = [&](int core) {
            return std::find(sib.worker_cores.begin(), sib.worker_cores.end(), core) != sib.worker_cores.end();
        };
        if (first.host_sibling < 0)
            ok = check(sib.host_core == first.host_core && sib.worker_cores == first.worker_cores,
                       "--host-core sibling without SMT must be first") && ok;
        else
            ok = check(sib.host_core == first.host_sibling && sib.host_core != first.host_core && !has(sib.host_core) &&
                           !has(first.host_core),
                       "--host-core sibling: the host on the sibling, no worker on its core") && ok;
    }
    return ok;
}

int main() {
    if (!host_sibling_layout()) return 1;
    const auto all = cpu::physical_cores(false);
    std::vector<int> cores;
    std::vector<ULONG> ids;
    for (int core : all) {
        if (std::any_of(cores.begin(), cores.end(), [core](int other) { return other / 64 == core / 64; })) continue;
        ULONG id = 0;
        if (!cpu::detail::cpu_set_for_core(core, id)) continue;
        cores.push_back(core);
        ids.push_back(id);
    }
    if (!check(!cores.empty(), "no usable CPU Sets")) return 1;
    if (!check(cores.size() == GetActiveProcessorGroupCount(), "CPU Sets do not cover all active groups")) return 1;

    // Hard pins belong on pool-owned threads; never contaminate the caller under test with one.
    for (size_t i = 0; i < cores.size(); ++i) {
        bool ok = false;
        std::thread worker([&] {
            if (!cpu::detail::set_thread_group_affinity(cores[i], (int) i)) return;
            GROUP_AFFINITY pinned{};
            if (!GetThreadGroupAffinity(GetCurrentThread(), &pinned)) return;
            ok = pinned.Group == cores[i] / 64 && pinned.Mask == (KAFFINITY(1) << (cores[i] & 63));
            // Host placement must also preserve an existing restrictive hard affinity.
            const auto previous = cpu::pin_current_thread(cores.back());
            cpu::restore_thread_affinity(previous);
            GROUP_AFFINITY after{};
            ok = ok && previous.valid && GetThreadGroupAffinity(GetCurrentThread(), &after) &&
                 after.Group == pinned.Group && after.Mask == pinned.Mask;
        });
        worker.join();
        if (!check(ok, "worker hard pin / caller hard-affinity preservation")) return 1;
    }

    bool host_ok = false;
    std::thread host([&] { host_ok = host_round_trips(cores, ids); });
    host.join();
    if (!host_ok || !inherited_default(cores.back(), ids.back(), cores.front(), ids.front())) return 1;
    std::printf("Windows affinity test: worker pins, host CPU Sets and process-default restoration passed (%zu groups)\n", cores.size());
    return 0;
}
