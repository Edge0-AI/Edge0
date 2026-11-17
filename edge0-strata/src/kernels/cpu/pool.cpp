// src/kernels/cpu/pool.cpp - P2.S3: the CPU expert pool.  Read pool.hpp first; it explains the protocol.
#include "strata/kernels/cpu/pool.hpp"
#include "strata/core/progress.hpp"
#include "strata/platform/aux_cpus.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"
namespace strata { namespace kernels {
static float lat_q8_0_q8_1_row(const uint8_t* row, const uint8_t* xq, int ncols) {
    const int nb = ncols / 32;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* wb = row + (size_t) b * 34;
        const uint8_t* xb = xq + (size_t) b * 36;
        const float dw = strata::kernels::f32_from_f16((uint16_t)(wb[0] | (wb[1] << 8)));
        const float dx = strata::kernels::f32_from_f16((uint16_t)(xb[0] | (xb[1] << 8)));
        const int8_t* ws = (const int8_t*)(wb + 2);
        const int8_t* xs = (const int8_t*)(xb + 4);
        int s = 0;
        for (int j = 0; j < 32; ++j) s += (int)xs[j] * (int)ws[j];
        acc += dw * dx * (float) s;
    }
    return acc;
}
// Q4DU: the Q4_0 form of the same row - {f16 d; uint8 nib[16]} blocks, element j in byte j's low
// nibble, j+16 in its high one, value (code - 8) * d; the reference for the AVX2 path's numbers.
static float lat_q4_0_q8_1_row(const uint8_t* row, const uint8_t* xq, int ncols) {
    const int nb = ncols / 32;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* wb = row + (size_t) b * 18;
        const uint8_t* xb = xq + (size_t) b * 36;
        const float dw = strata::kernels::f32_from_f16((uint16_t)(wb[0] | (wb[1] << 8)));
        const float dx = strata::kernels::f32_from_f16((uint16_t)(xb[0] | (xb[1] << 8)));
        const int8_t* xs = (const int8_t*)(xb + 4);
        int s = 0;
        for (int j = 0; j < 16; ++j) {
            const int byte_ = wb[2 + j];
            s += (int)xs[j] * ((int)(byte_ & 0xF) - 8);
            s += (int)xs[j + 16] * ((int)(byte_ >> 4) - 8);
        }
        acc += dw * dx * (float) s;
    }
    return acc;
}
} }
#include "strata/kernels/cpu/numa_pool.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include "strata/platform/cpu_relax.hpp"
#if defined(_MSC_VER)
#include <intrin.h>   // __cpuidex, __rdtsc (STRATA_POOL_UMWAIT)
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "pool_affinity_win.hpp"
#else
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include "pool_affinity_linux.hpp"
#endif

namespace strata::kernels::cpu {

namespace {
// STRATA_POOL_UMWAIT=<TSC cycles> (experimental, x86 WAITPKG only): a parked worker waits with UMONITOR/UMWAIT on the
// epoch's cache line instead of spinning on _mm_pause.  It wakes when the host publishes (a write to that line) or
// after the cycles, in the light C0.1 state; on a laptop the CPU then gives back the power the spin took, which the
// GPU's share of the budget gets.  0 or unset (or no WAITPKG): the spin, unchanged.  STRATA_POOL_UMWAIT_C02=1: the
// deeper C0.2 state (slower wake).  MSVC x64 only for now (the intrinsics need no target flag there); other builds spin.
uint64_t umwait_cycles() {
    static const uint64_t c = [] {
        const char* e = std::getenv("STRATA_POOL_UMWAIT");
        if (e == nullptr || std::atoll(e) <= 0) return (uint64_t) 0;
#if defined(_MSC_VER) && defined(_M_X64) && !defined(__clang__)   // MSVC proper: clang-cl (the Windows HIP build) has no WAITPKG intrinsics by default
        int r[4] = {0, 0, 0, 0};
        __cpuidex(r, 7, 0);
        if (((r[2] >> 5) & 1) == 0) {
            std::fprintf(stderr, "strata: STRATA_POOL_UMWAIT: this CPU has no WAITPKG, the workers spin\n");
            return (uint64_t) 0;
        }
        std::fprintf(stderr, "strata: STRATA_POOL_UMWAIT: parked workers wait with UMWAIT (%lld cycles)\n", std::atoll(e));
        return (uint64_t) std::atoll(e);
#else
        return (uint64_t) 0;
#endif
    }();
    return c;
}
unsigned umwait_state() {
    static const unsigned s = [] { const char* e = std::getenv("STRATA_POOL_UMWAIT_C02"); return e && *e == '1' ? 0u : 1u; }();
    return s;   // 1 = C0.1 (light), 0 = C0.2
}
// STRATA_POOL_UMWAIT_ESC=<TSC cycles> (experimental): an escalating wait - C0.1 for the first cycles of a park, C0.2 after
// (a wait that has lasted that long is a GPU phase, a draft or a prompt read, where a slower wake costs little).
uint64_t umwait_escalate() {
    static const uint64_t c = [] { const char* e = std::getenv("STRATA_POOL_UMWAIT_ESC"); return e ? (uint64_t) (std::max)(0LL, std::atoll(e)) : (uint64_t) 0; }();
    return c;
}

constexpr uint64_t pack_head(uint32_t epoch, uint32_t n, uint32_t i) {
    return ((uint64_t) epoch << 32) | ((uint64_t) n << 16) | (uint64_t) i;
}
}  // namespace

namespace {
std::atomic<int> g_host_core{(int) HostCore::First};
}  // namespace

void set_host_core(HostCore where) { g_host_core.store((int) where); }
HostCore host_core_setting() { return (HostCore) g_host_core.load(); }

namespace {
std::vector<int> g_pool_cores;   // --pool-cores: set once at start-up, before the pool exists
}  // namespace
void set_pool_cores(const std::vector<int>& cpus) { g_pool_cores = cpus; }
const std::vector<int>& pool_cores_setting() { return g_pool_cores; }

std::vector<LogicalCpuInfo> logical_cpu_layout() {
    std::vector<LogicalCpuInfo> out;
#if defined(_WIN32)
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) return out;
    std::vector<char> buf(len);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(), &len))
        return out;
    // a `start /affinity` or a job limit on a PC with one processor group: only those processors are the process's
    DWORD_PTR proc_mask = 0, sys_mask = 0;
    const bool limited = GetActiveProcessorGroupCount() == 1 && GetProcessAffinityMask(GetCurrentProcess(), &proc_mask, &sys_mask);
    long core = 0;
    for (const char* p = buf.data(); p < buf.data() + len; p += ((const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) p)->Size) {
        const auto* e = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) p;
        if (e->Relationship != RelationProcessorCore) continue;
        for (WORD group = 0; group < e->Processor.GroupCount; ++group) {
            const GROUP_AFFINITY& g = e->Processor.GroupMask[group];
            for (int bit = 0; bit < 64; ++bit)
                if ((g.Mask & (KAFFINITY(1) << bit)) && (!limited || (proc_mask & (DWORD_PTR(1) << bit))))
                    out.push_back({(int) (g.Group * 64 + bit), core});
        }
        ++core;
    }
#elif defined(__linux__)
    std::vector<unsigned long> mask;
    std::vector<int> allowed;
    if (detail::get_thread_affinity(mask, &allowed) != 0) return out;
    auto read = [](int cpu, const char* what) -> long {
        char path[96];
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/%s", cpu, what);
        long v = -1;
        if (std::FILE* f = std::fopen(path, "r")) {
            if (std::fscanf(f, "%ld", &v) != 1) v = -1;
            std::fclose(f);
        }
        return v;
    };
    for (int cpu : allowed) {
        const long pkg = read(cpu, "physical_package_id"), core = read(cpu, "core_id");
        out.push_back({cpu, pkg >= 0 && core >= 0 ? pkg * 1000000 + core : -1});
    }
#else
    for (unsigned i = 0; i < std::thread::hardware_concurrency(); ++i) out.push_back({(int) i, -1});
#endif
    return out;
}

static CpuTopology detect_cpu_topology_impl(bool skip_first, PoolAffinity affinity);

// The fields the default worker count needs (pool_plan.hpp): the logical processors the process may use, the CPU
// package of the host core and of every worker core, whether the OS told physical cores from SMT siblings, the
// container CPU quota.  On a multi-socket CPU (not hybrid) the host package comes first in `worker_cores`.
static void fill_plan_fields(CpuTopology& topo, bool skip_first) {
    topo.cpu_quota = container_cpu_quota();
    topo.worker_package.assign(topo.worker_cores.size(), -1);
#if defined(_WIN32)
    (void) skip_first;
    topo.logical = (int) GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    return;   // (the sockets of a Windows box are not told apart: one package)
#elif defined(__APPLE__)
    (void) skip_first;
    topo.logical = (int) std::thread::hardware_concurrency();
    return;   // no SMT on Apple silicon and no sysfs: the cores are the processors
#else
    std::vector<unsigned long> mask;
    std::vector<int> allowed;
    topo.logical = detail::get_thread_affinity(mask, &allowed) == 0 ? (int) allowed.size()
                                                                     : (int) std::thread::hardware_concurrency();
    auto read = [](int cpu, const char* what) -> long {
        char path[96];
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/%s", cpu, what);
        long v = -1;
        if (std::FILE* f = std::fopen(path, "r")) {
            if (std::fscanf(f, "%ld", &v) != 1) v = -1;
            std::fclose(f);
        }
        return v;
    };
    bool known = !topo.worker_cores.empty() || topo.host_core >= 0;
    auto pkg_of = [&](int cpu) {
        const long pkg = read(cpu, "physical_package_id"), core = read(cpu, "core_id");
        if (pkg < 0 || core < 0) known = false;
        return (int) pkg;
    };
    if (topo.host_core >= 0) topo.host_package = pkg_of(topo.host_core);
    for (size_t i = 0; i < topo.worker_cores.size(); ++i) topo.worker_package[i] = pkg_of(topo.worker_cores[i]);
    topo.layout_known = known;
    if (!known) return;
    if (skip_first && !topo.is_hybrid && topo.host_core >= 0 && topo.host_package >= 0) {
        // the host socket first (stable inside each socket): the default workers are the local ones
        std::vector<size_t> order(topo.worker_cores.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::stable_partition(order.begin(), order.end(),
                              [&](size_t i) { return topo.worker_package[i] == topo.host_package; });
        std::vector<int> cores, pkgs;
        for (size_t i : order) { cores.push_back(topo.worker_cores[i]); pkgs.push_back(topo.worker_package[i]); }
        topo.worker_cores = std::move(cores);
        topo.worker_package = std::move(pkgs);
    }
#endif
}

CpuTopology detect_cpu_topology(bool skip_first, PoolAffinity affinity) {
    CpuTopology topo = detect_cpu_topology_impl(skip_first, affinity);
    // --host-core last: the host takes the last core, the workers the others (the first included)
    if (skip_first && host_core_setting() == HostCore::Last && !topo.is_hybrid && topo.host_core >= 0 &&
        !topo.worker_cores.empty()) {
        const int last = topo.worker_cores.back();
        topo.worker_cores.pop_back();
        topo.worker_cores.insert(topo.worker_cores.begin(), topo.host_core);
        topo.host_core = last;
    }
    // --host-core sibling: the host on its core's other hardware thread; the first logical processor stays to the
    // interrupts and to nothing else (a worker on it would share the host's core)
    if (skip_first && host_core_setting() == HostCore::Sibling && topo.host_sibling >= 0) {
        topo.worker_cores.erase(std::remove(topo.worker_cores.begin(), topo.worker_cores.end(), topo.host_sibling),
                                topo.worker_cores.end());
        topo.host_core = topo.host_sibling;
    }
    fill_plan_fields(topo, skip_first);
    return topo;
}

double container_cpu_quota() {
#if defined(__linux__)
    // cgroup v2: "max 100000" (no limit) or "<quota> <period>"; v1: cpu.cfs_quota_us (-1: none) / cpu.cfs_period_us
    if (std::FILE* f = std::fopen("/sys/fs/cgroup/cpu.max", "r")) {
        char a[32] = {0};
        long period = 0;
        double q = 0;
        if (std::fscanf(f, "%31s %ld", a, &period) == 2 && std::strcmp(a, "max") != 0 && period > 0)
            q = std::atof(a) / (double) period;
        std::fclose(f);
        return q > 0 ? q : 0.0;
    }
    long quota = -1, period = 0;
    if (std::FILE* f = std::fopen("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r")) {
        if (std::fscanf(f, "%ld", &quota) != 1) quota = -1;
        std::fclose(f);
    }
    if (std::FILE* f = std::fopen("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r")) {
        if (std::fscanf(f, "%ld", &period) != 1) period = 0;
        std::fclose(f);
    }
    return quota > 0 && period > 0 ? (double) quota / (double) period : 0.0;
#else
    return 0.0;
#endif
}

PoolPlanInput pool_plan_input(const CpuTopology& topo) {
    PoolPlanInput in;
    in.logical = topo.logical > 0 ? topo.logical : (int) std::thread::hardware_concurrency();
    in.cpu_quota = topo.cpu_quota;
    if (topo.is_hybrid) { in.p_cores = topo.p_cores; in.e_cores = topo.e_cores; }
    if (!topo.layout_known) return in;   // cores_per_package stays empty
    // the cores per package, the host package first; the host core counts in its package
    std::vector<std::pair<int, int>> counts;   // (package, cores)
    auto add = [&](int pkg) {
        for (auto& c : counts)
            if (c.first == pkg) { ++c.second; return; }
        counts.emplace_back(pkg, 1);
    };
    if (topo.host_core >= 0) add(topo.host_package);
    for (size_t i = 0; i < topo.worker_cores.size(); ++i)
        add(i < topo.worker_package.size() ? topo.worker_package[i] : -1);
    if (topo.host_core >= 0) {   // (the host package first even when it has no worker core listed)
        auto it = std::find_if(counts.begin(), counts.end(), [&](const auto& c) { return c.first == topo.host_package; });
        if (it != counts.begin()) std::iter_swap(counts.begin(), it);
    }
    for (const auto& c : counts) in.cores_per_package.push_back(c.second);
    return in;
}

int planned_host_core(PoolAffinity affinity) { return detect_cpu_topology(true, affinity).host_core; }

static CpuTopology detect_cpu_topology_impl(bool skip_first, PoolAffinity affinity) {
    CpuTopology topo;
#if defined(_WIN32)
    // Ask the OS rather than assuming a layout.  `hardware_concurrency()` returns LOGICAL processors, and on
    // every SMT machine half of them are siblings - pinning one worker to each of the first N would put two
    // workers on each physical core and halve the bandwidth the expert kernel is bound by.
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) {
        // Preserve group identity even when detailed core topology is unavailable.
        const WORD groups = GetActiveProcessorGroupCount();
        for (WORD group = 0; group < groups; ++group) {
            const DWORD count = GetActiveProcessorCount(group);
            if (count == 0 || count == (DWORD) -1 || count > 64) continue;
            for (DWORD i = 0; i < count; ++i) topo.worker_cores.push_back((int) group * 64 + (int) i);
        }
        if (skip_first && !topo.worker_cores.empty()) {
            topo.host_core = topo.worker_cores.front();
            topo.worker_cores.erase(topo.worker_cores.begin());
        }
        // The OS did not say which logical processors share a core, and every one is listed as a core above: the default
        // worker count must not count them as cores (30 workers on a 16c/32t PC would sit two to a core).  With the layout
        // unknown the plan takes half the logical processors (pool_plan.hpp), as on Linux without sysfs.
        topo.layout_known = false;
        return topo;
    }
    std::vector<char> buf(len);
    if (GetLogicalProcessorInformationEx(RelationProcessorCore,
                                         (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(), &len)) {
        const char* p = buf.data();
        const char* end = p + len;
        struct CoreDesc {
            uint8_t efficiency = 0;
            bool has_smt = false;
            std::vector<int> lps;
        };
        std::vector<CoreDesc> descs;
        while (p < end) {
            const auto* e = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) p;
            if (e->Relationship == RelationProcessorCore) {
                CoreDesc cd;
                cd.efficiency = e->Processor.EfficiencyClass;
                cd.has_smt = (e->Processor.Flags & LTP_PC_SMT) != 0;
                for (WORD group = 0; group < e->Processor.GroupCount; ++group) {
                    const GROUP_AFFINITY& g = e->Processor.GroupMask[group];
                    for (int bit = 0; bit < 64; ++bit) {
                        if (g.Mask & (KAFFINITY(1) << bit)) {
                            cd.lps.push_back((int) (g.Group * 64 + bit));
                        }
                    }
                }
                if (!cd.lps.empty()) {
                    descs.push_back(std::move(cd));
                }
            }
            p += e->Size;
        }

        uint8_t min_eff = 255, max_eff = 0;
        for (const auto& c : descs) {
            min_eff = (std::min)(min_eff, c.efficiency);
            max_eff = (std::max)(max_eff, c.efficiency);
        }

        topo.is_hybrid = (max_eff > min_eff);
        if (topo.is_hybrid) {
            for (const auto& c : descs) {
                if (c.efficiency == max_eff) {
                    topo.p_cores++;
                    topo.p_threads += (int) c.lps.size();
                } else {
                    topo.e_cores++;
                }
            }
        } else {
            topo.p_cores = (int) descs.size();
            for (const auto& c : descs) topo.p_threads += (int) c.lps.size();
        }

        if (affinity == PoolAffinity::All || !topo.is_hybrid) {
            // #642 (from Hardin22's fork): on a hybrid CPU the P-cores first (the host takes the first of them), so a
            // pool smaller than the core count (setup's --pool-workers for a hybrid CPU) runs on the P-cores and the
            // first E-cores rather than on whatever the OS numbered first.  All cores alike: the order is unchanged.
            if (topo.is_hybrid)
                std::stable_sort(descs.begin(), descs.end(),
                                 [](const CoreDesc& x, const CoreDesc& y) { return x.efficiency > y.efficiency; });
            for (const auto& c : descs) topo.worker_cores.push_back(c.lps[0]);
            if (skip_first && !topo.worker_cores.empty()) {
                topo.host_core = topo.worker_cores.front();
                topo.worker_cores.erase(topo.worker_cores.begin());
                if (descs.front().lps.size() > 1) topo.host_sibling = descs.front().lps[1];
            }
            return topo;
        }

        // Hybrid CPU with Auto or PCores affinity:
        // Prioritize Performance cores:
        // 1. Primary logical processor of each P-core (avoids SMT resource contention)
        // 2. SMT sibling logical processors of P-cores
        // 3. E-cores (only as overflow in Auto mode)
        std::vector<int> p_primaries;
        std::vector<int> p_siblings;
        std::vector<int> e_cores;

        for (const auto& c : descs) {
            if (c.efficiency == max_eff) {
                p_primaries.push_back(c.lps[0]);
                for (size_t s = 1; s < c.lps.size(); ++s) {
                    p_siblings.push_back(c.lps[s]);
                }
            } else {
                for (int lp : c.lps) e_cores.push_back(lp);
            }
        }

        if (skip_first && !p_primaries.empty()) {
            topo.host_core = p_primaries.front();
            p_primaries.erase(p_primaries.begin());
            for (const auto& c : descs)
                if (c.lps[0] == topo.host_core) {
                    if (c.lps.size() > 1) topo.host_sibling = c.lps[1];
                    break;
                }
        }

        for (int cpu : p_primaries) topo.worker_cores.push_back(cpu);
        for (int cpu : p_siblings) topo.worker_cores.push_back(cpu);
        if (affinity != PoolAffinity::PCores) {
            for (int cpu : e_cores) topo.worker_cores.push_back(cpu);
        }
        return topo;
    }
#else
    // The logical CPUs this process may run on, ONE PER PHYSICAL CORE (issue #40): SMT siblings share a core's
    // load/store bandwidth, so a worker on each would put two workers on one core, as the Windows branch above
    // explains.  sysfs names each CPU's (package, core); the first allowed CPU of each pair is kept, so a taskset
    // that leaves out the first sibling still gets its core.  Without sysfs every allowed CPU counts, as before.
    auto topo_read = [](int cpu, const char* what) -> long {
        char path[96];
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/%s", cpu, what);
        long v = -1;
        if (std::FILE* f = std::fopen(path, "r")) {
            if (std::fscanf(f, "%ld", &v) != 1) v = -1;
            std::fclose(f);
        }
        return v;
    };
    auto cap_read = [](int cpu) -> long {
        char path[96];
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpu_capacity", cpu);
        long v = -1;
        if (std::FILE* f = std::fopen(path, "r")) {
            if (std::fscanf(f, "%ld", &v) != 1) v = -1;
            std::fclose(f);
        }
        return v;
    };

    // #798: a cpulist ("0-7,16") of one of the hybrid PMU's CPU sets: /sys/devices/cpu_core/cpus and cpu_atom/cpus
    auto cpulist_read = [](const char* path, std::vector<int>& out) -> bool {
        std::FILE* f = std::fopen(path, "r");
        if (!f) return false;
        char text[1024] = {0};
        const size_t n = std::fread(text, 1, sizeof text - 1, f);
        std::fclose(f);
        text[n] = '\0';
        for (const char* p = text; *p;) {
            char* end = nullptr;
            const long lo = std::strtol(p, &end, 10);
            if (end == p) break;
            long hi = lo;
            if (*end == '-') {
                p = end + 1;
                hi = std::strtol(p, &end, 10);
            }
            for (long c = lo; c <= hi && c < 4096; ++c) out.push_back((int) c);
            p = end;
            while (*p == ',' || *p == ' ' || *p == '\n')  ++p;
        }
        return !out.empty();
    };

    std::vector<int> allowed;
    std::vector<unsigned long> allowed_mask;
    if (detail::get_thread_affinity(allowed_mask, &allowed) != 0) {
        for (unsigned i = 0; i < std::thread::hardware_concurrency(); ++i) allowed.push_back((int) i);
    }

    struct CoreLinux {
        int cpu = -1;
        long pkg = -1;
        long core = -1;
        long cap = -1;
        bool is_sibling = false;
    };
    std::vector<CoreLinux> all_cpus;
    std::vector<std::pair<long, long>> seen_phys;
    long max_cap = 0, min_cap = 1000000;

    for (int cpu : allowed) {
        CoreLinux cl;
        cl.cpu = cpu;
        cl.pkg = topo_read(cpu, "physical_package_id");
        cl.core = topo_read(cpu, "core_id");
        cl.cap = cap_read(cpu);
        if (cl.cap > 0) {
            max_cap = (std::max)(max_cap, cl.cap);
            min_cap = (std::min)(min_cap, cl.cap);
        }
        if (cl.pkg >= 0 && cl.core >= 0) {
            const std::pair<long, long> key{cl.pkg, cl.core};
            if (std::find(seen_phys.begin(), seen_phys.end(), key) != seen_phys.end()) {
                cl.is_sibling = true;
            } else {
                seen_phys.push_back(key);
            }
        }
        all_cpus.push_back(cl);
    }

    // #798: which CPUs are E-cores.  Intel's hybrid PMU lists them (cpu_atom/cpus beside cpu_core/cpus); without it a
    // CPU is an E-core when its capacity is under 90% of the largest.  Turbo Boost Max 3.0 gives the "favored"
    // P-cores a slightly higher capacity (1024 against 1012 on a Core Ultra 7 270K Plus): "capacity == the maximum"
    // counted 2 of its 8 P-cores and started 1 pool worker.
    // #3262: Meteor Lake and Arrow Lake-H list their LP-E cores (SoC tile, no L3, ~2.5 GHz) in a third set,
    // /sys/devices/cpu_lowpower/cpus, that is in neither of the other two; they are E-cores too (and sort last).
    std::vector<int> pmu_core, pmu_atom, pmu_lowpower;
    const bool pmu = cpulist_read("/sys/devices/cpu_core/cpus", pmu_core) &&
                     cpulist_read("/sys/devices/cpu_atom/cpus", pmu_atom);
    cpulist_read("/sys/devices/cpu_lowpower/cpus", pmu_lowpower);
    auto is_e = [&](const CoreLinux& cl) {
        return hybrid_cpu_is_efficiency(cl.cpu, cl.cap, max_cap, pmu, pmu_core, pmu_atom, pmu_lowpower);
    };
    bool any_e = false, any_p = false;
    for (const auto& cl : all_cpus) (is_e(cl) ? any_e : any_p) = true;
    topo.is_hybrid = any_e && any_p;
    if (topo.is_hybrid) {
        for (const auto& cl : all_cpus) {
            if (!is_e(cl)) {
                if (!cl.is_sibling) topo.p_cores++;
                topo.p_threads++;
            } else {
                if (!cl.is_sibling) topo.e_cores++;
            }
        }
    } else {
        topo.p_cores = (int) seen_phys.size();
        topo.p_threads = (int) all_cpus.size();
    }

    // the host core's SMT sibling, an allowed CPU of the same (package, core) (--host-core sibling)
    auto sibling_of = [&](int cpu) {
        for (const auto& h : all_cpus)
            if (h.cpu == cpu && h.pkg >= 0 && h.core >= 0)
                for (const auto& cl : all_cpus)
                    if (cl.cpu != cpu && cl.pkg == h.pkg && cl.core == h.core) return cl.cpu;
        return -1;
    };

    if (affinity == PoolAffinity::All || !topo.is_hybrid) {
        if (topo.is_hybrid)   // #642: the P-cores first (see the Windows branch)
            std::stable_sort(all_cpus.begin(), all_cpus.end(),
                             [&](const CoreLinux& x, const CoreLinux& y) {
                                 return is_e(x) != is_e(y) ? !is_e(x) : x.cap > y.cap;
                             });
        for (const auto& cl : all_cpus) {
            if (!cl.is_sibling) topo.worker_cores.push_back(cl.cpu);
        }
        if (skip_first && !topo.worker_cores.empty()) {
            topo.host_core = topo.worker_cores.front();
            topo.worker_cores.erase(topo.worker_cores.begin());
            topo.host_sibling = sibling_of(topo.host_core);
        }
        return topo;
    }

    // Hybrid CPU on Linux:
    std::vector<int> p_primaries;
    std::vector<int> p_siblings;
    std::vector<int> e_cores;

    for (const auto& cl : all_cpus) {
        if (!is_e(cl)) {
            if (!cl.is_sibling) p_primaries.push_back(cl.cpu);
            else p_siblings.push_back(cl.cpu);
        } else {
            e_cores.push_back(cl.cpu);
        }
    }

    if (skip_first && !p_primaries.empty()) {
        topo.host_core = p_primaries.front();
        p_primaries.erase(p_primaries.begin());
        topo.host_sibling = sibling_of(topo.host_core);
    }

    for (int cpu : p_primaries) topo.worker_cores.push_back(cpu);
    for (int cpu : p_siblings) topo.worker_cores.push_back(cpu);
    if (affinity != PoolAffinity::PCores) {
        for (int cpu : e_cores) topo.worker_cores.push_back(cpu);
    }
    return topo;
#endif
    return topo;
}

std::vector<int> physical_cores(bool skip_first, PoolAffinity affinity) {
    return detect_cpu_topology(skip_first, affinity).worker_cores;
}

namespace {

bool pin_this_thread(int core, [[maybe_unused]] int worker = -1) {
    if (core < 0) return false;
#if defined(_WIN32)
    return detail::set_thread_group_affinity(core, worker);
#else
    const int error = detail::pin_thread_to_cpu(core);
    if (error != 0)
        std::fprintf(stderr, "strata cpu pool: affinity for worker %d (CPU %d) failed: %d; previous affinity kept\n",
                     worker, core, error);
    return error == 0;
#endif
}

#if defined(__linux__)
// Where the NUMA topology is read from: /sys. STRATA_TEST_SYSFS_ROOT=<dir> reads <dir>/devices/system/{node,cpu} instead,
// so a test can give the pool a made-up two-node machine (pool_tasks_test); nobody sets it for real.
std::string sysfs_root() {
    const char* e = std::getenv("STRATA_TEST_SYSFS_ROOT");
    return e != nullptr && *e != '\0' ? std::string(e) : std::string("/sys");
}

// The NUMA node of a logical CPU (sysfs), or -1.
int cpu_numa_node(int cpu) {
    if (cpu < 0) return -1;
    const std::string root = sysfs_root();
    char path[512];
    for (int node = 0; node < 64; ++node) {
        std::snprintf(path, sizeof path, "%s/devices/system/cpu/cpu%d/node%d", root.c_str(), cpu, node);
        if (access(path, F_OK) == 0) return node;
    }
    return -1;
}
#endif

// The NUMA nodes with CPUs this machine has (0.2.2: the split is tried by itself from 2 on).  Linux: the sysfs node folder
// (numa_pool.hpp); Windows: GetNumaHighestNodeNumber; any other system, or a sandbox that hides the folder: 1.
int numa_node_count() {
#if defined(__linux__)
    return (std::max)(1, numa_cpu_node_count_at(sysfs_root() + "/devices/system/node"));
#elif defined(_WIN32)
    ULONG highest = 0;
    return GetNumaHighestNodeNumber(&highest) ? (int) highest + 1 : 1;
#else
    return 1;
#endif
}

// The real-node split for threads on `thread_core` (numa_pool.hpp).  Linux only: the placement of the experts' memory
// (first touch by threads pinned to a node, move_pages to check) has no Windows or macOS version.
NumaSplitDecision decide_real_numa_split([[maybe_unused]] const std::vector<int>& thread_core, [[maybe_unused]] int n_workers,
                                         [[maybe_unused]] bool pinned) {
#if defined(__linux__)
    // the expert source's own reading of it (expert_source.cpp): exactly "1"
    const char* rotate = std::getenv("STRATA_EXCHANGE_ROTATE");
    return decide_numa_split(thread_core, n_workers, pinned, rotate != nullptr && std::strcmp(rotate, "1") == 0, cpu_numa_node);
#else
    NumaSplitDecision d;
    d.why = "the split needs Linux";
    return d;
#endif
}

}  // namespace

ThreadAffinity pin_current_thread(int core) {
    if (core < 0) return {};
#if defined(_WIN32)
    ThreadAffinity previous;
    ULONG target = 0;
    if (!detail::get_thread_cpu_sets(previous.cpu_sets) || !detail::cpu_set_for_core(core, target) ||
        !SetThreadSelectedCpuSets(GetCurrentThread(), &target, 1)) {
        std::fprintf(stderr, "strata cpu pool: host CPU Set selection for processor %d failed: %lu; previous placement kept\n",
                     core, (unsigned long) GetLastError());
        return {};
    }
    previous.valid = true;
    return previous;
#else
    ThreadAffinity previous;
    int error = detail::get_thread_affinity(previous.mask);
    if (error == 0) error = detail::pin_thread_to_cpu(core);
    if (error != 0) {
        std::fprintf(stderr, "strata cpu pool: host affinity for CPU %d failed: %d; previous affinity kept\n", core, error);
        return {};
    }
    previous.valid = true;
    return previous;
#endif
}

void restore_thread_affinity(const ThreadAffinity& previous) {
    if (!previous.valid) return;
#if defined(_WIN32)
    // Clearing an originally empty selection restores process-default/all-group eligibility without
    // turning the caller's implicit Windows 11 affinity into an explicit single-group hard mask.
    if (!SetThreadSelectedCpuSets(GetCurrentThread(), previous.cpu_sets.empty() ? nullptr : previous.cpu_sets.data(),
                                 (ULONG) previous.cpu_sets.size()))
        std::fprintf(stderr, "strata cpu pool: host CPU Set restoration failed: %lu\n",
                     (unsigned long) GetLastError());
#else
    const int error = detail::set_thread_affinity(previous.mask);
    if (error != 0)
        std::fprintf(stderr, "strata cpu pool: host affinity restoration failed: %d\n", error);
#endif
}

namespace {
std::atomic<const ExpertPool*> g_diag_pool{nullptr};
void diag_active_pool(std::FILE* f) {
    if (const ExpertPool* p = g_diag_pool.load()) p->diag(f);
}
int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
// CBA: one plain global, written by the constructor and destructor of the pool that owns it; fine while pools are
// built and destroyed on one thread (generate's measure pools, then the main one), an atomic owner if that changes.
NumaSplitPlan g_numa_plan;
const ExpertPool* g_numa_owner = nullptr;
}  // namespace

const NumaSplitPlan& active_numa_split() { return g_numa_plan; }

void ExpertPool::diag(std::FILE* f) const {
    const uint64_t h = head_.load();
    uint32_t batch = (uint32_t) (h >> 32), claimed = (uint32_t) h & 0xffffu, total = (uint32_t) (h >> 16) & 0xffffu,
             done = done_.load();
    // CBA: this summed-slices branch is never run by a test (it needs a diag call mid-phase), it is a trivial sum, a mid-phase
    // diag test if it ever misreports
    if (numa_n_ > 0 && mode_ >= 5) {   // a split native batch never writes head_/done_: the slices' sums instead
        claimed = total = done = 0;
        for (int k = 0; k < numa_n_; ++k) {
            const uint64_t sh = nsync_[(size_t) k].head.load();
            batch = (uint32_t) (sh >> 32);   // every slice head carries the batch's epoch
            claimed += (uint32_t) sh & 0xffffu;
            total += (uint32_t) (sh >> 16) & 0xffffu;
            done += nsync_[(size_t) k].done.load();
        }
    }
    std::fprintf(f, "  expert pool: epoch %u, batch epoch %u: %u of %u jobs claimed, %u done; %u of %d workers parked, "
                    "%u sleeping; mode %d\n", epoch_.load(), batch, claimed, total, done, parked_count(), n_,
                 sleepers_.load(), mode_);
    if (numa_n_ > 0) std::fprintf(f, "  expert pool NUMA slices:%s\n", numa_counters().c_str());
    std::fprintf(f, "  expert pool threads:");
    for (int i = 0; i < n_; ++i) {
        const int32_t s = wstate_[(size_t) i].load();
        if (s == kParked) std::fprintf(f, " w%d=parked", i);
        else if (s == kSleeping) std::fprintf(f, " w%d=sleeping", i);
        else if (s == kBetween) std::fprintf(f, " w%d=draining", i);
        else std::fprintf(f, " w%d=job%d", i, s);
    }
    const int32_t hs = hstate_.load();
    const char* hn = hs == kIdle ? "idle" : hs == kWaitParked ? "waiting for the workers to park"
                   : hs == kWaitDone ? "waiting for the jobs to finish" : "running a job";
    std::fprintf(f, "; host %s", hn);
    if (hs >= 0) std::fprintf(f, " %d", hs);
    std::fprintf(f, " for %lld ms\n", (long long) (now_ms() - hstate_ms_.load()));
}

ExpertPool::ExpertPool(int n_workers, bool pin, bool host_works, PoolAffinity affinity, int tasks)
    : tasks_(tasks), host_works_(host_works), affinity_(affinity), topo_(detect_cpu_topology(true, affinity)) {
    if (tasks < 0 || tasks > kMaxTasks) throw std::invalid_argument("expert pool tasks must be in 0..4096");
    pinned_ = pin;
    layout_ = logical_cpu_layout();   // now, before anything pins the caller (Linux reads the calling thread's allowed CPUs)
    if (const char* e = std::getenv("STRATA_POOL_SPIN_US"))   // a test knob; see kSpinBeforeSleep
        spin_before_sleep_ = std::chrono::microseconds((std::max)(0, std::atoi(e)));
    // STRATA_NUMA_SPLIT, read here for the worker count (the split itself is decided below, once the count is known).
    // 0.2.2: with 2 or more NUMA nodes and neither variable set the split is tried by itself, and STRATA_NUMA_POOL=0 turns that
    // off (numa_pool.hpp).  One node: pick_numa_pool says Unchanged and nothing below differs from the pool before the split.
    const char* env = std::getenv("STRATA_NUMA_SPLIT");
    const int numa_nodes = numa_node_count();
    const NumaPoolPick pool_pick = pick_numa_pool(std::getenv("STRATA_NUMA_POOL"), env, numa_nodes);
    if (!pool_pick.log.empty()) std::fprintf(stderr, "%s\n", pool_pick.log.c_str());
    if (pool_pick.choice == NumaPoolChoice::Off) env = nullptr;
    bool auto_pool = pool_pick.choice == NumaPoolChoice::Auto;   // the split without STRATA_NUMA_SPLIT: only if it can run
    bool env_off = !auto_pool && (env == nullptr || *env == '\0' || std::strcmp(env, "0") == 0);
    bool env_known = auto_pool || (!env_off && (std::strcmp(env, "1") == 0 || std::strcmp(env, "test") == 0));
    const PoolPlanInput plan_in = pool_plan_input(topo_);
    const PoolPlan wplan = plan_pool_workers(plan_in);
    // --pool-cores (experimental, off by default): the person's own list of logical processors for the workers.  The plan above
    // was made from the real topology; only the cores the workers are pinned to (and, by default, their count) come from the list.
    if (!pool_cores_setting().empty()) {
        const PoolCoresCheck chk = check_pool_cores(pool_cores_setting(), layout_, topo_.host_core);
        static std::atomic<bool> told{false};   // (a throw-away pool of --pool-affinity measure builds the same list again)
        const bool say = !told.exchange(true);
        if (say)
            for (const std::string& w : chk.warnings) std::fprintf(stderr, "strata cpu pool: --pool-cores: %s\n", w.c_str());
        if (chk.cpus.empty()) {
            if (say)
                std::fprintf(stderr, "strata cpu pool: --pool-cores: no usable logical processor in the list; the pool uses its "
                                     "default placement\n");
        } else {
            topo_.worker_cores = chk.cpus;
            topo_.worker_package.assign(chk.cpus.size(), -1);
            explicit_cores_ = true;
        }
    }
    // the split needs workers on both nodes: its cores are taken round-robin over the nodes (the packages when a core's node
    // is unknown), so any count under the core count (the default or a configured one) spans them evenly (one node: the order
    // is kept)
    auto interleave_cores = [&] {
        if (explicit_cores_) return;   // --pool-cores: the person's order stays
        if (topo_.worker_package.size() != topo_.worker_cores.size()) return;
        std::vector<int> key = topo_.worker_package;
#if defined(__linux__)
        std::vector<int> by_node;
        for (int c : topo_.worker_cores) by_node.push_back(cpu_numa_node(c));
        if (std::none_of(by_node.begin(), by_node.end(), [](int n) { return n < 0; })) key = by_node;
#endif
        std::vector<int> cores, pkgs;
        for (size_t i : interleave_packages(key)) {
            cores.push_back(topo_.worker_cores[i]);
            pkgs.push_back(topo_.worker_package[i]);
        }
        topo_.worker_cores = std::move(cores);
        topo_.worker_package = std::move(pkgs);
    };
    // the worker count: configured, else the hybrid P cores, else the plan (`span`: every socket counts, the split's default)
    auto pick_workers = [&](bool span, const char** limit) -> int {
        if (n_workers > 0) return n_workers;
        if (explicit_cores_) {
            *limit = "your --pool-cores list";
            return (int) topo_.worker_cores.size();
        }
        if (topo_.is_hybrid && affinity_ != PoolAffinity::All) return (std::max)(1, topo_.p_cores - 1);
        if (span) {
            PoolPlanInput span_in = plan_in;
            span_in.span_packages = true;
            const PoolPlan sp = plan_pool_workers(span_in);
            *limit = sp.limit;
            return (std::min)(sp.workers, (std::max)(1, (int) topo_.worker_cores.size()));
        }
        *limit = wplan.limit;
        return (std::min)(wplan.workers, (std::max)(1, (int) topo_.worker_cores.size()));
    };
    if (auto_pool) {
        // try the split on a copy: when it cannot run the cores, the count and the plan stay exactly what they were
        const CpuTopology plain_topo = topo_;
        interleave_cores();
        const char* try_limit = nullptr;
        const int try_n = pick_workers(true, &try_limit);
        std::vector<int> thread_core;
        for (int j = 0; j < try_n; ++j)
            thread_core.push_back(pin && j < (int) topo_.worker_cores.size() ? topo_.worker_cores[(size_t) j] : -1);
        if (host_works_) thread_core.push_back(topo_.host_core);
        const NumaSplitDecision tried = decide_real_numa_split(thread_core, try_n, pin);
        if (!tried.on) {
            topo_ = plain_topo;
            auto_pool = env_known = false;
            env_off = true;
            std::fprintf(stderr, "strata cpu pool: NUMA pool off: %s (%d NUMA nodes found), the pool runs as before\n",
                         tried.why.c_str(), numa_nodes);
        }
    } else if (env_known) {
        interleave_cores();
    }
    {
        const char* limit = nullptr;
        n_ = pick_workers(env_known, &limit);
        if (limit != nullptr) plan_limit_ = limit;
    }
    plan_cores_ = wplan.cores;
    plan_logical_ = plan_in.logical;
    if (n_ < 1) n_ = 1;
    scratch_.resize((size_t) n_);
    wstate_.reset(new std::atomic<int32_t>[(size_t) n_]);
    for (int i = 0; i < n_; ++i) wstate_[(size_t) i].store(kParked);
    hstate_ms_.store(now_ms());
    g_diag_pool.store(this);
    strata::core::diag_pool_fn().store(&diag_active_pool);
    split_.resize((size_t) kMaxSplit);
    split_multi_.resize((size_t) kMaxSplitMulti);
    // STRATA_NUMA_SPLIT: decided here, before any worker starts (a worker counts into its slice's `parked` from its
    // first instruction), and never revised.  Unset: the plan stays empty and nothing below runs.
    NumaSplitPlan plan;
    if (!env_off && !env_known) {
        plan.why = std::string("unknown STRATA_NUMA_SPLIT value '") + env + "' (use 1 or test)";
        std::fprintf(stderr, "strata cpu pool: NUMA split off: %s\n", plan.why.c_str());
    } else if (env_known) {
        plan.test = !auto_pool && std::strcmp(env, "test") == 0;
        const int threads = n_ + (host_works_ ? 1 : 0);   // thread j < n_: worker j; j == n_: the host
        std::vector<int> slice((size_t) threads, 0);
        std::string why;
        if (threads < 2) {
            why = "the pool has " + std::to_string(threads) + " thread, the split needs 2 or more";
        } else if (plan.test) {
            for (int j = 0; j < threads; ++j) slice[(size_t) j] = j % 2;
        } else {
            std::vector<int> thread_core;
            for (int j = 0; j < threads; ++j)
                thread_core.push_back(j == n_ ? topo_.host_core
                                              : pin && j < (int) topo_.worker_cores.size() ? topo_.worker_cores[(size_t) j] : -1);
            const NumaSplitDecision d = decide_real_numa_split(thread_core, n_, pin);
            why = d.why;
            if (d.on) {
                slice = d.slice;
                plan.nodes[0] = d.nodes[0];
                plan.nodes[1] = d.nodes[1];
                for (int j = 0; j < threads; ++j) plan.cores[slice[(size_t) j]].push_back(thread_core[(size_t) j]);
            }
        }
        plan.why = why;
        plan.on = plan.why.empty();
        if (!plan.on) {   // a refusal after the slices were planned (unbalanced) leaves no half-filled plan behind
            plan.nodes[0] = plan.nodes[1] = -1;
            plan.cores[0].clear();
            plan.cores[1].clear();
        }
        if (plan.on) {
            numa_n_ = 2;
            nsync_.reset(new NodeSync[2]);
            wslice_.assign(slice.begin(), slice.begin() + n_);
            hslice_ = host_works_ ? slice[(size_t) n_] : 0;
            slice_threads_.assign(2, 0);
            for (int k : slice) ++slice_threads_[(size_t) k];
            if (auto_pool)
                std::fprintf(stderr, "strata cpu pool: NUMA pool on: %d NUMA nodes found, one slice of the pool per node "
                                     "(STRATA_NUMA_POOL=0 turns it off)\n", numa_nodes);
            if (plan.test)
                std::fprintf(stderr, "strata cpu pool: NUMA split on (test, no real nodes): slice 0 = %d threads, "
                                     "slice 1 = %d threads\n", slice_threads_[0], slice_threads_[1]);
            else
                std::fprintf(stderr, "strata cpu pool: NUMA split on: slice 0 = node %d (%d threads), slice 1 = node %d "
                                     "(%d threads)\n", plan.nodes[0], slice_threads_[0], plan.nodes[1], slice_threads_[1]);
        } else {
            std::fprintf(stderr, "strata cpu pool: NUMA split off: %s\n", plan.why.c_str());
        }
    }
    if (n_workers > 0) {
        // recommend, never force: a count above what this CPU uses well runs as configured, with a plain warning
        // (STRATA_NUMA_SPLIT on: workers on the other socket are the point, that case is not warned about)
        PoolPlanInput warn_in = plan_in;
        warn_in.span_packages = plan.on;
        static std::atomic<bool> warned{false};
        const std::string why = pool_workers_warning(n_workers, warn_in, plan_pool_workers(warn_in));
        if (!why.empty() && !warned.exchange(true)) std::fprintf(stderr, "strata cpu pool: warning: %s\n", why.c_str());
    }
    g_numa_plan = std::move(plan);
    g_numa_owner = this;
    threads_.reserve((size_t) n_);
    for (int i = 0; i < n_; ++i) {
        const int core = pin ? (i < (int) topo_.worker_cores.size() ? topo_.worker_cores[(size_t) i] : -1) : -1;
        threads_.emplace_back([this, i, core] {
            strata::platform::name_this_thread("strata-pool", i);
            strata::aux_cpus::note_owned_thread();   // --aux-cpus never moves a worker
            // a failed pin only costs locality under the split: the worker keeps the slice it was given
            if (!pin_this_thread(core, i) && core >= 0 && numa_n_ > 0)
                std::fprintf(stderr, "strata cpu pool: NUMA split: worker %d not pinned to CPU %d, its slice's rows may "
                                     "be read remotely\n", i, core);
            worker(i);
        });
    }
}

std::string ExpertPool::placement_line() const {
    if (n_ <= 0) return "";
    if (!pinned_) return "CPU workers: " + std::to_string(n_) + " workers, not pinned to logical processors (the OS places them)";
    // a worker beyond the cores listed runs unpinned (more --pool-workers than --pool-cores entries)
    std::vector<int> on(topo_.worker_cores.begin(), topo_.worker_cores.begin() + (std::min)((size_t) n_, topo_.worker_cores.size()));
    std::string s = describe_worker_placement(on, topo_.host_core, layout_, explicit_cores_);
    if ((int) on.size() < n_) s += " (" + std::to_string(n_ - (int) on.size()) + " more workers run unpinned)";
    return s;
}

ExpertPool::~ExpertPool() {
    const ExpertPool* self = this;
    g_diag_pool.compare_exchange_strong(self, nullptr);
    if (g_numa_owner == this) {
        g_numa_plan = NumaSplitPlan{};
        g_numa_owner = nullptr;
    }
    stop_.store(true, std::memory_order_release);
    // Bump the epoch so a PARKED worker notices the stop flag rather than sleeping through it.
    publish();
    for (auto& t : threads_) t.join();
}

void ExpertPool::publish() {
    // Both sides are seq_cst, and that is the whole lost-wakeup argument: a worker going to sleep does
    // `sleepers_++` and then reads `epoch_`, the host does `epoch_++` and then reads `sleepers_`.  In one total
    // order at least one of them sees the other's write - the worker sees the new epoch and does not sleep, or
    // the host sees the sleeper and notifies under the mutex the worker holds until it is inside `wait`.
    // On x86 the fetch_add is a locked xadd either way, so this costs the token path nothing.
    epoch_.fetch_add(1, std::memory_order_seq_cst);
    if (sleepers_.load(std::memory_order_seq_cst) != 0) {
        std::lock_guard<std::mutex> lk(sleep_mu_);
        sleep_cv_.notify_all();
    }
}

void ExpertPool::worker(int id) {
    uint32_t seen = 0;
    // ARRIVE at the park before the first wait, so `parked_ == n_` is true from construction.  Counting only
    // on the RETURN from a drain leaves `parked_` at 0 until each worker has finished one batch, and the first
    // `run()` - which waits for `parked_ == n_` before publishing - then deadlocks.  It deadlocks on the very
    // first call, which is the good case; a version that deadlocked on the second would be far worse.
    parked_of(id).fetch_add(1, std::memory_order_acq_rel);
    for (;;) {
        // Park: wait for work.  `_mm_pause` rather than a bare spin because it yields the pipeline to the
        // sibling hyperthread; `epoch_` is bumped once per LAYER, not once per expert, so most of these
        // iterations are spent here with nothing to do.
        //
        // **AND NOTHING ELSE HAPPENS IN HERE.**  This loop used to do `pauses_.fetch_add(1)` on every iteration
        // - a locked read-modify-write, five workers against one cache line - so the workers spent their wait
        // invalidating each other's caches and the very line the host writes to publish work.  The counter was
        // diagnostic and nothing branched on it.  See the note on the atomics in pool.hpp.
        //
        // After `kSpinBeforeSleep` with no work the worker sleeps instead (issue #4).  The clock is read once
        // every 1024 pauses, so the spin itself is unchanged.
        const auto parked_at = std::chrono::steady_clock::now();
        uint32_t spins = 0;
        const uint64_t um = umwait_cycles();
#if defined(_MSC_VER) && defined(_M_X64) && !defined(__clang__)   // MSVC proper: clang-cl (the Windows HIP build) has no WAITPKG intrinsics by default
        const uint64_t um_esc = umwait_escalate();
        const uint64_t um_t0 = um != 0 && um_esc != 0 ? __rdtsc() : 0;
#endif
        while (epoch_.load(std::memory_order_acquire) == seen) {
            if (stop_.load(std::memory_order_relaxed)) return;
#if defined(_MSC_VER) && defined(_M_X64) && !defined(__clang__)   // MSVC proper: clang-cl (the Windows HIP build) has no WAITPKG intrinsics by default
            if (um != 0) {   // STRATA_POOL_UMWAIT: wake on the publish's write to epoch_, or after `um` cycles
                _umonitor((void*) &epoch_);
                if (epoch_.load(std::memory_order_acquire) != seen) break;
                const uint64_t now = __rdtsc();
                const unsigned st = um_esc != 0 ? (now - um_t0 < um_esc ? 1u : 0u) : umwait_state();
                _umwait(st, now + um);
                if ((++spins & 15u) != 0) continue;   // each wait is ~um cycles: look at the clock more often
            } else
#endif
            {
                strata_cpu_pause();
                if ((++spins & 1023u) != 0) continue;
            }
            if (std::chrono::steady_clock::now() - parked_at < spin_before_sleep_) continue;
            std::unique_lock<std::mutex> lk(sleep_mu_);
            wstate_[(size_t) id].store(kSleeping, std::memory_order_relaxed);
            sleepers_.fetch_add(1, std::memory_order_seq_cst);
            sleep_cv_.wait(lk, [&] {
                return epoch_.load(std::memory_order_seq_cst) != seen || stop_.load(std::memory_order_relaxed);
            });
            sleepers_.fetch_sub(1, std::memory_order_relaxed);
            wstate_[(size_t) id].store(kParked, std::memory_order_relaxed);
        }
        if (stop_.load(std::memory_order_acquire)) return;
        // acquire: the batch this epoch published (`head`, and the description before it) is visible from here
        seen = epoch_.load(std::memory_order_acquire);
        parked_of(id).fetch_sub(1, std::memory_order_acq_rel);   // leaving the park

        // Drain: one claim per iteration, so a slow worker takes fewer experts and a fast one takes more.
        // Every job is the same size (all experts are 1,382,400 bytes), so there is nothing to schedule.  Only
        // this epoch's jobs: if the host has already moved on, the claims fail and the worker parks again.
        wstate_[(size_t) id].store(kBetween, std::memory_order_relaxed);
        drain(id, scratch_[(size_t) id], seen);
        wstate_[(size_t) id].store(kParked, std::memory_order_relaxed);
        parked_of(id).fetch_add(1, std::memory_order_acq_rel);   // back at the park
    }
}

int ExpertPool::claim(uint32_t epoch) {
    uint64_t h = head_.load(std::memory_order_acquire);
    for (;;) {
        if ((uint32_t) (h >> 32) != epoch) return -1;               // not the batch this thread woke for
        const uint32_t n = (uint32_t) (h >> 16) & 0xffffu, i = (uint32_t) h & 0xffffu;
        if (i >= n) return -1;                                       // exhausted
        if (head_.compare_exchange_weak(h, h + 1, std::memory_order_acq_rel, std::memory_order_acquire))
            return (int) i;
    }
}

int ExpertPool::claim_slice(uint32_t epoch, int slice) {
    NodeSync& ns = nsync_[(size_t) slice];
    uint64_t h = ns.head.load(std::memory_order_acquire);
    for (;;) {
        if ((uint32_t) (h >> 32) != epoch) return -1;               // not this batch, or not a mode-5/6 batch
        const uint32_t n = (uint32_t) (h >> 16) & 0xffffu, i = (uint32_t) h & 0xffffu;
        if (i >= n) return -1;                                       // the slice is exhausted
        if (ns.head.compare_exchange_weak(h, h + 1, std::memory_order_acq_rel, std::memory_order_acquire))
            return ns.base + (int) i;                                // the description is stable from here
    }
}

uint32_t ExpertPool::parked_count() const {
    if (numa_n_ == 0) return parked_.load(std::memory_order_acquire);
    uint32_t p = 0;   // each slice's counter is bounded by its own workers, so the sum is n_ only when each is full
    for (int k = 0; k < numa_n_; ++k) p += nsync_[(size_t) k].parked.load(std::memory_order_acquire);
    return p;
}

std::string ExpertPool::numa_counters() const {
    std::string s;
    for (int k = 0; k < numa_n_; ++k) {
        const NodeSync& ns = nsync_[(size_t) k];
        const uint64_t h = ns.head.load();
        char buf[200];
        std::snprintf(buf, sizeof buf, "%s slice %d: batch epoch %u, %u of %u claimed, %u done, %u parked, %d threads",
                      k == 0 ? " [" : ";", k, (uint32_t) (h >> 32), (uint32_t) h & 0xffffu, (uint32_t) (h >> 16) & 0xffffu,
                      ns.done.load(), ns.parked.load(), slice_threads_[(size_t) k]);
        s += buf;
    }
    if (numa_n_ > 0) s += "]";
    return s;
}

int ExpertPool::numa_plan_phase(int per, int nb) {
    int base = 0;
    for (int k = 0; k < numa_n_; ++k) {
        NodeSync& ns = nsync_[(size_t) k];
        ns.row0 = per * k / numa_n_;
        ns.h = per * (k + 1) / numa_n_ - ns.row0;
        ns.rows = (int64_t) nb * ns.h;
        // about three tasks per thread of the slice's node; an explicit --pool-tasks is shared between the slices
        const int64_t want = tasks_ == 0 ? 3 * (int64_t) slice_threads_[(size_t) k] : (int64_t) (std::max)(1, tasks_ / 2);
        ns.tasks = (int) (std::min)(want, ns.rows);
        ns.base = base;
        base += ns.tasks;
    }
    return base;
}

uint32_t ExpertPool::begin_batch(int n) {
    if (n < 0 || n > 0xffff) {
        std::fprintf(stderr, "strata: expert pool batch of %d jobs is out of range\n", n);
        std::abort();
    }
    // Every job of the previous batch has completed (`wait_done`), and a claim of it can no longer succeed, so
    // nothing adds to `done` until this batch's first claim - which the release below orders after the reset.
    const uint32_t e = epoch_.load(std::memory_order_relaxed) + 1;   // only the host bumps the epoch
    if (numa_phase()) {
        // every slice's head, even a slice with no task, and never `head_`: for this epoch only the slice heads
        // carry `e`, so a claim on `head_` fails, and every head left from an earlier batch is exhausted (pool.hpp)
        for (int k = 0; k < numa_n_; ++k) {
            NodeSync& ns = nsync_[(size_t) k];
            ns.done.store(0, std::memory_order_relaxed);
            ns.drained_ns.store(0, std::memory_order_relaxed);
            ns.head.store(pack_head(e, (uint32_t) ns.tasks, 0), std::memory_order_release);
        }
        pub_ns_ = now_ns();
    } else {
        done_.store(0, std::memory_order_relaxed);
        head_.store(pack_head(e, (uint32_t) n, 0), std::memory_order_release);
    }
    publish();
    return e;
}

void ExpertPool::wait_parked(const char* what) {
    hstate_.store(kWaitParked, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
    uint32_t spins = 0;
    std::chrono::steady_clock::time_point t0{};
    while (parked_count() != (uint32_t) n_) {
        strata_cpu_pause();
        if ((++spins & 1023u) != 0) continue;
        const auto now = std::chrono::steady_clock::now();
        if (spins == 1024u) t0 = now;
        else if (now - t0 > kStall) {
            std::fprintf(stderr, "strata: the CPU expert pool stalled %s (%u of %d workers parked)%s - stopping the engine "
                                 "so the server can start it again (issue #29)\n",
                         what, parked_count(), n_, numa_counters().c_str());
            strata::core::release_gpu_waits(stderr);   // #267: the GPU may be spinning on this layer's flag
            std::fflush(stderr);
            std::abort();
        }
    }
}

void ExpertPool::wait_done(int n) {
    hstate_.store(kWaitDone, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
    uint32_t spins = 0, seen = 0;
    std::chrono::steady_clock::time_point t0{};
    for (;;) {
        uint32_t d = 0;
        if (numa_phase())   // the slices' counts, each bounded by its tasks: the sum is n only when every slice is done
            for (int k = 0; k < numa_n_; ++k) d += nsync_[(size_t) k].done.load(std::memory_order_acquire);
        else
            d = done_.load(std::memory_order_acquire);
        if (d >= (uint32_t) n) {
            static std::atomic<int64_t> s_tw{0};
            const int64_t tw = s_tw.fetch_add(1, std::memory_order_relaxed);
            if (std::getenv("STRATA_DIAG_A") != nullptr && tw % 48 == 0) {
                double lo = 1e18, hi = -1e18; int lo_id = -1, hi_id = -1, cnt = 0;
                for (int k = 0; k < n_ && k < 64; ++k) {
                    if (tail_fin_[k] <= 0) continue;
                    if (tail_fin_[k] < lo) { lo = tail_fin_[k]; lo_id = k; }
                    if (tail_fin_[k] > hi) { hi = tail_fin_[k]; hi_id = k; }
                    ++cnt;
                }
                if (cnt > 1)
                    std::fprintf(stderr, "PROBETAIL mode=%d n=%d last=w%d spread=%.1f us (first w%d of %d)\n",
                                 mode_, n, hi_id, (hi - lo) / 1000.0, lo_id, cnt);
            }
            return;
        }                                // `>=`: never a wait that an overshoot outlives
        strata_cpu_pause();
        if ((++spins & 1023u) != 0) continue;
        const auto now = std::chrono::steady_clock::now();
        if (spins == 1024u || d != seen) { t0 = now; seen = d; }     // progress restarts the clock
        else if (now - t0 > kStall) {
            std::fprintf(stderr, "strata: the CPU expert pool stalled: %u of %d jobs done, %u of %d workers parked%s - "
                                 "stopping the engine so the server can start it again (issue #29)\n",
                         d, n, parked_count(), n_, numa_counters().c_str());
            strata::core::release_gpu_waits(stderr);   // #267
            std::fflush(stderr);
            std::abort();
        }
    }
}

void ExpertPool::drain(int id, ExpertScratch& scratch, uint32_t epoch) {
    const int s = numa_n_ == 0 ? 0 : id >= 0 ? wslice_[(size_t) id] : hslice_;
    for (;;) {
        // STRATA_NUMA_SPLIT: a mode-5/6 batch lives on the slice heads, any other on `head_`; the epoch tag makes the
        // other family's claim fail, so `mode_` is never read before a claim has succeeded (pool.hpp)
        int ci = numa_n_ > 0 ? claim_slice(epoch, s) : -1;
        const bool sliced = ci >= 0;
        if (!sliced) ci = claim(epoch);
        if (ci < 0) {
            if (id >= 0 && id < 64) {   // PROBE-TAIL: this worker's last task of the phase
                if (tail_ep_[id] != (uint64_t) epoch) {
                    tail_ep_[id] = (uint64_t) epoch;
                    tail_fin_[id] = (double) now_ns();
                }
            }
            break;
        }
        const uint32_t i = (uint32_t) ci;
        if (id >= 0) wstate_[(size_t) id].store(ci, std::memory_order_relaxed);
        else { hstate_.store(ci, std::memory_order_relaxed); hstate_ms_.store(now_ms(), std::memory_order_relaxed); }
        if (mode_ == 0) {
            const ExpertJob& j = jobs_[i];
            s2_expert_vnni_q(j.blob, *j.act, j.out, scratch);
        } else if (mode_ == 1) {
            const int e = (int) i / parts_a_, part = (int) i % parts_a_;
            const int r0 = FF * part / parts_a_, r1 = FF * (part + 1) / parts_a_;
            s2_expert_gu_rows(jobs_[e].blob, *jobs_[e].act, split_[(size_t) e].ff, r0, r1);
        } else if (mode_ == 2) {
            const int e = (int) i / parts_b_, part = (int) i % parts_b_;
            const int r0 = H * part / parts_b_, r1 = H * (part + 1) / parts_b_;
            s2_expert_down_rows(jobs_[e].blob, split_[(size_t) e].a2, jobs_[e].out, r0, r1);
        } else if (mode_ == 7) {
            // latent MoE (edge0): a row range of one of the D/U matvecs (Q8_0 weights x Q8_1 acts).
            // AVX-2 when the host has it (one loaded weight row serves all the tokens; the scalar
            // lat_q8_0_q8_1_row below is the fallback and the reference for the numbers).
            const int64_t R = mrows_, T = mtasks_;
            const int64_t g0 = R * (int64_t) i / T, g1 = R * (int64_t) (i + 1) / T;
            int64_t base = 0;
            for (int m = 0; m < lmv_n_ && base < g1; ++m) {
                const LatentMatvec& mv = lmv_[m];
                if (base + mv.nrows > g0) {
                    const int r0 = (int) std::max<int64_t>(g0 - base, 0);
                    const int r1 = (int) std::min<int64_t>(g1 - base, mv.nrows);
                    // Q4DU: the matvec's row bytes decide the format (18 vs 34 per 32-value
                    // block - set by the caller from LatentDense::row_b()); no other plumbing.
                    const bool lat_q4 = mv.row_bytes == (size_t) (mv.ncols / 32) * 18;
                    if (lat_q4) {
                        if (strata::kernels::cpu::cpu_avx2_ok())
                            strata::kernels::cpu::lat_q4_0_q8_1_rows(mv.w, mv.row_bytes, mv.ncols, mv.act, mv.nt,
                                                                     mv.out, r0, r1);
                        else
                            for (int r = r0; r < r1; ++r) {
                                const uint8_t* row = mv.w + (size_t) r * mv.row_bytes;
                                for (int t = 0; t < mv.nt; ++t)
                                    mv.out[t][r] = lat_q4_0_q8_1_row(row, (const uint8_t*) mv.act[t], mv.ncols);
                            }
                    } else
                    if (strata::kernels::cpu::cpu_avx2_ok())
                        strata::kernels::cpu::lat_q8_0_q8_1_rows(mv.w, mv.row_bytes, mv.ncols, mv.act, mv.nt,
                                                                 mv.out, r0, r1);
                    else
                        for (int r = r0; r < r1; ++r) {
                            const uint8_t* row = mv.w + (size_t) r * mv.row_bytes;
                            for (int t = 0; t < mv.nt; ++t)
                                mv.out[t][r] = lat_q8_0_q8_1_row(row, (const uint8_t*) mv.act[t], mv.ncols);
                        }
                }
                base += mv.nrows;
            }
        } else if (mode_ >= 5) {
            // plan v0.3 P6: native layers, 5 = gate/up rows, 6 = down rows (the layer's own n_ff / n_embd: Flash-Next's
            // FF / H, or another family's smaller expert)
            // Task j of T over R rows of `h` per expert from matrix row `row0`: the whole phase (R = mrows_, h = per,
            // row0 = 0) or, under the NUMA split, this thread's slice (slice-local row g -> expert g / h, matrix row
            // row0 + g % h)
            const int per = mode_ == 5 ? (int) nfmt_->n_ff : (int) nfmt_->n_embd;
            int64_t R = mrows_, T = mtasks_, j = i;
            int h = per, row0 = 0;
            if (sliced) {
                const NodeSync& ns = nsync_[(size_t) s];
                R = ns.rows; T = ns.tasks; h = ns.h; row0 = ns.row0; j = ci - ns.base;
            }
            const int64_t g0 = R * j / T, g1 = R * (j + 1) / T;
            for (int64_t g = g0; g < g1;) {
                const int e = (int) (g / h), l0 = (int) (g % h);
                const int l1 = (int) std::min<int64_t>(h, l0 + (g1 - g));
                const int r0 = row0 + l0, r1 = row0 + l1;
                SplitBufMulti& sb = split_multi_[(size_t) e];
                const NativeFmt* const nfmt = mjobs_[e].fmt != nullptr ? mjobs_[e].fmt : nfmt_;   // 149-cold: same types and rows
                if (mode_ == 5 && q2_native_kernels(nfmt->gu_type)) {
                    // a native Q2_0 pack: gate and up rows on the Q2_0 kernels, then SwiGLU
                    thread_local float gbuf[MAXT][FF], ubuf[MAXT][FF];
                    float* gp[MAXT];
                    float* up[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) { gp[t] = gbuf[t]; up[t] = ubuf[t]; }
                    const int nbk = (int) (nfmt->n_embd / 64);
                    q2_rows_any(mjobs_[e].blob, nfmt->gu_row, nbk, mjobs_[e].act, mjobs_[e].nt, gp, r0, r1);
                    q2_rows_any(mjobs_[e].blob + nfmt->up_off, nfmt->gu_row, nbk, mjobs_[e].act, mjobs_[e].nt, up, r0, r1);
                    for (int t = 0; t < mjobs_[e].nt; ++t)
                        for (int r = r0; r < r1; ++r)
                            sb.ff[t][r] = (gbuf[t][r] / (1.f + std::exp(-gbuf[t][r]))) * ubuf[t][r];
                } else if (mode_ == 5) {
                    float* ff[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) ff[t] = sb.ff[t];
                    if (narithmetic_ == NativeArithmetic::WindowInvariant)
                        native_gu_rows_invariant(*nfmt, mjobs_[e].blob, mjobs_[e].nact, mjobs_[e].nt, ff, r0, r1);
                    else
                        native_gu_rows(*nfmt, mjobs_[e].blob, mjobs_[e].nact, mjobs_[e].nt, ff, r0, r1);
                } else if (q2_native_kernels(nfmt->d_type)) {
                    // Q2_0 down (most IQ layers): the AVX-512 kernel, ggml-cpu has only a scalar one on x86
                    const ActQ* a2[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) a2[t] = &sb.a2[t];
                    q2_rows_any(mjobs_[e].blob + nfmt->down_off, nfmt->d_row, (int) (nfmt->n_ff / 64), a2,
                                mjobs_[e].nt, mjobs_[e].out, r0, r1);
                } else {
                    const void* hq[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) hq[t] = sb.hq[t];
                    if (narithmetic_ == NativeArithmetic::WindowInvariant)
                        native_down_rows_invariant(*nfmt, mjobs_[e].blob, hq, mjobs_[e].nt, mjobs_[e].out, r0, r1);
                    else
                        native_down_rows(*nfmt, mjobs_[e].blob, hq, mjobs_[e].nt, mjobs_[e].out, r0, r1);
                }
                g += l1 - l0;
            }
        } else {
            // plan v0.3 P6: an equal range of the phase's rows across ALL its experts (a range may span two)
            const int per = mode_ == 3 ? FF : H;
            const int64_t g0 = mrows_ * (int64_t) i / mtasks_, g1 = mrows_ * (int64_t) (i + 1) / mtasks_;
            for (int64_t r = g0; r < g1;) {
                const int e = (int) (r / per), r0 = (int) (r % per);
                const int r1 = (int) std::min<int64_t>(per, r0 + (g1 - r));
                SplitBufMulti& sb = split_multi_[(size_t) e];
                if (mode_ == 3) {
                    float* ff[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) ff[t] = sb.ff[t];
                    s2_expert_gu_rows_multi(mjobs_[e].blob, mjobs_[e].act, mjobs_[e].nt, ff, r0, r1);
                } else {
                    const ActQ* a2[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) a2[t] = &sb.a2[t];
                    s2_expert_down_rows_multi(mjobs_[e].blob, a2, mjobs_[e].nt, mjobs_[e].out, r0, r1);
                }
                r += r1 - r0;
            }
        }
        if (sliced) {
            NodeSync& ns = nsync_[(size_t) s];
            const uint32_t t = (uint32_t) ns.tasks;
            // the slice's last task stamps its drain time; read by the host once this thread has parked again
            if (ns.done.fetch_add(1, std::memory_order_release) + 1 == t)
                ns.drained_ns.store(now_ns(), std::memory_order_relaxed);
        } else {
            done_.fetch_add(1, std::memory_order_release);
        }
    }
}

void ExpertPool::run_phase(int mode, int n_tasks) {
    // PROBE-PHASE: the five walls of one phase dispatch, per mode (5=gu 6=down 7=latent-U), DIAGP-style
    static std::atomic<double> s_pb[8]{}, s_bb[8]{}, s_hd[8]{}, s_wd[8]{}, s_pa[8]{};
    static std::atomic<int64_t> s_pn[8]{};
    const auto _p0 = std::chrono::steady_clock::now();
    wait_parked("before a phase");
    const auto _p1 = std::chrono::steady_clock::now();
    mode_ = mode;
    njobs_ = n_tasks;
    const uint32_t e = begin_batch(n_tasks);
    const auto _p2 = std::chrono::steady_clock::now();
    if (host_works_) drain(-1, host_scratch_, e);
    const auto _p3 = std::chrono::steady_clock::now();
    wait_done(n_tasks);
    const auto _p4 = std::chrono::steady_clock::now();
    wait_parked("after a phase");
    const auto _p5 = std::chrono::steady_clock::now();
    const int mi = mode;
    s_pb[mi].fetch_add(std::chrono::duration<double, std::milli>(_p1 - _p0).count(), std::memory_order_relaxed);
    s_bb[mi].fetch_add(std::chrono::duration<double, std::milli>(_p2 - _p1).count(), std::memory_order_relaxed);
    s_hd[mi].fetch_add(std::chrono::duration<double, std::milli>(_p3 - _p2).count(), std::memory_order_relaxed);
    s_wd[mi].fetch_add(std::chrono::duration<double, std::milli>(_p4 - _p3).count(), std::memory_order_relaxed);
    s_pa[mi].fetch_add(std::chrono::duration<double, std::milli>(_p5 - _p4).count(), std::memory_order_relaxed);
    s_pn[mi].fetch_add(1, std::memory_order_relaxed);
    static std::atomic<int64_t> s_total{0};
    const int64_t tot = s_total.fetch_add(1, std::memory_order_relaxed);
    if (std::getenv("STRATA_DIAG_A") != nullptr && tot % 288 == 0) {
        std::fprintf(stderr, "PROBEPH per-mode ms/phase (parkB begin hostDrain waitDone parkA):\n");
        for (int m = 0; m < 8; ++m) {
            const int64_t c = s_pn[m].load();
            if (c == 0) continue;
            std::fprintf(stderr, "  mode %d: parkB %.3f begin %.3f hostDrain %.3f waitDone %.3f parkA %.3f (n %lld)\n",
                         m, s_pb[m].load() / (double) c, s_bb[m].load() / (double) c, s_hd[m].load() / (double) c,
                         s_wd[m].load() / (double) c, s_pa[m].load() / (double) c, (long long) c);
        }
    }
    // Each slice's stamp is stored after its last `done` increment, so `wait_done` can return before it lands; the
    // thread that stored it has parked again since (it left the park to claim), and that re-park is what the wait
    // above acquired.  So the stamps are read here, not in `wait_done`.
    if (numa_phase())
        for (int k = 0; k < numa_n_; ++k) {
            const int64_t t = nsync_[(size_t) k].drained_ns.load(std::memory_order_acquire);
            if (t != 0) slice_ms_[k] += (double) (t - pub_ns_) / 1e6;
        }
    hstate_.store(kIdle, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
}

void ExpertPool::run_split(ExpertJob* jobs, int n) {
    if (n <= 0) return;
    if (n > kMaxSplit || n_ == 1 || expert_oracle_q8_0_enabled()) { run(jobs, n); return; }
    const auto t0 = std::chrono::steady_clock::now();
    jobs_ = jobs;
    const int threads = n_ + (host_works_ ? 1 : 0);
    // about three tasks per thread in each phase, so the tail is short
    parts_a_ = (std::max)(1, (3 * threads + n - 1) / n);
    parts_b_ = parts_a_;
    run_phase(1, n * parts_a_);
    for (int e = 0; e < n; ++e) act_quant_q8_1(split_[(size_t) e].ff, FF, split_[(size_t) e].a2);
    run_phase(2, n * parts_b_);
    mode_ = 0;
    ms_drain_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int ExpertPool::phase_tasks(int64_t rows) const {
    return tasks_ == 0 ? 3 * (n_ + (host_works_ ? 1 : 0)) : (int) (std::min)(rows, (int64_t) tasks_);
}

void ExpertPool::run_split_multi(ExpertJobMulti* jobs, int n) {
    if (n <= 0) return;
    if (n > kMaxSplitMulti || expert_oracle_q8_0_enabled()) {
        // one token at a time through the single-token path (the oracle contract has no multi kernel)
        std::vector<ExpertJob> single;
        for (int e = 0; e < n; ++e)
            for (int t = 0; t < jobs[e].nt; ++t) {
                ExpertJob j;
                j.blob = jobs[e].blob;
                j.act = jobs[e].act[t];
                j.out = jobs[e].out[t];
                single.push_back(j);
            }
        for (size_t i = 0; i < single.size(); i += kMaxSplit)
            run_split(single.data() + i, (int) (std::min)((size_t) kMaxSplit, single.size() - i));
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    mjobs_ = jobs;
    mrows_ = (int64_t) n * FF;
    mtasks_ = phase_tasks(mrows_);
    run_phase(3, mtasks_);
    const auto t1 = std::chrono::steady_clock::now();
    for (int e = 0; e < n; ++e)
        for (int t = 0; t < jobs[e].nt; ++t)
            act_quant_q8_1(split_multi_[(size_t) e].ff[t], FF, split_multi_[(size_t) e].a2[t]);
    const auto t2 = std::chrono::steady_clock::now();
    mrows_ = (int64_t) n * H;
    mtasks_ = phase_tasks(mrows_);
    run_phase(4, mtasks_);
    const auto t3 = std::chrono::steady_clock::now();
    ms_multi_gu += std::chrono::duration<double, std::milli>(t1 - t0).count();
    ms_multi_q += std::chrono::duration<double, std::milli>(t2 - t1).count();
    ms_multi_down += std::chrono::duration<double, std::milli>(t3 - t2).count();
    multi_bytes += (int64_t) n * (int64_t) BLOB;
    mode_ = 0;
    ms_drain_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void ExpertPool::run_split_multi_native(const NativeFmt& f, ExpertJobMulti* jobs, int n, NativeArithmetic arithmetic) {
    if (n <= 0) return;
    const auto t0 = std::chrono::steady_clock::now();
    const bool diag_a = std::getenv("STRATA_DIAG_A") != nullptr;
    // more distinct experts than buffers: run them in batches
    for (int b0 = 0; b0 < n; b0 += kMaxSplitMulti) {
        const int nb = (std::min)(kMaxSplitMulti, n - b0);
        mjobs_ = jobs + b0;
        nfmt_ = &f;
        narithmetic_ = arithmetic;
        mrows_ = (int64_t) nb * f.n_ff;
        mtasks_ = numa_n_ > 0 ? numa_plan_phase((int) f.n_ff, nb) : phase_tasks(mrows_);
        const int diag_rows = (int) mrows_, diag_tasks = mtasks_;   // DIAGA: the split before run
        const auto a = std::chrono::steady_clock::now();
        run_phase(5, mtasks_);
        const auto b = std::chrono::steady_clock::now();
        if (diag_a)
            std::fprintf(stderr, "DIAGA phase5: jobs=%d n_ff=%lld rows=%d tasks=%d wall=%.3f ms\n",
                         nb, (long long) f.n_ff, diag_rows, diag_tasks,
                         std::chrono::duration<double, std::milli>(b - a).count());
        for (int e = 0; e < nb; ++e)
            for (int t = 0; t < mjobs_[e].nt; ++t)
                if (q2_native_kernels(f.d_type)) act_quant_any(split_multi_[(size_t) e].ff[t], FF, split_multi_[(size_t) e].a2[t]);
                else native_quant_h(f, split_multi_[(size_t) e].ff[t], split_multi_[(size_t) e].hq[t]);
        const auto c = std::chrono::steady_clock::now();
        if (diag_a)
            std::fprintf(stderr, "DIAGA quant: jobs=%d entries=%d wall=%.3f ms\n",
                         nb, (int) [&]{ int n = 0; for (int e = 0; e < nb; ++e) n += mjobs_[e].nt; return n; }(),
                         std::chrono::duration<double, std::milli>(c - b).count());
        mrows_ = (int64_t) nb * f.n_embd;
        mtasks_ = numa_n_ > 0 ? numa_plan_phase((int) f.n_embd, nb) : phase_tasks(mrows_);
        run_phase(6, mtasks_);
        const auto d = std::chrono::steady_clock::now();
        if (diag_a)
            std::fprintf(stderr, "DIAGA phase6: jobs=%d n_embd=%lld rows=%d tasks=%d wall=%.3f ms\n",
                         nb, (long long) f.n_embd, (int) mrows_, (int) mtasks_,
                         std::chrono::duration<double, std::milli>(d - c).count());
        ms_multi_gu += std::chrono::duration<double, std::milli>(b - a).count();
        ms_multi_q += std::chrono::duration<double, std::milli>(c - b).count();
        ms_multi_down += std::chrono::duration<double, std::milli>(d - c).count();
    }
    multi_bytes += (int64_t) n * (int64_t) f.bytes;
    mode_ = 0;
    ms_drain_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void ExpertPool::run_latent_rows(LatentMatvec* mv, int n) {
    if (n <= 0) return;
    const auto t0 = std::chrono::steady_clock::now();
    const bool diag_a = std::getenv("STRATA_DIAG_A") != nullptr;
    int64_t rows = 0, bytes = 0;
    for (int m = 0; m < n; ++m) { rows += mv[m].nrows; bytes += (int64_t) mv[m].row_bytes * mv[m].nrows; }
    lmv_ = mv;
    lmv_n_ = n;
    mrows_ = rows;
    mtasks_ = phase_tasks(rows);
    run_phase(7, mtasks_);
    mode_ = 0;
    const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (diag_a)
        std::fprintf(stderr, "DIAGU groups=%d rows=%lld tasks=%d bytes=%lld wall=%.3f ms = %.1f GB/s\n",
                     n, (long long) rows, mtasks_, (long long) bytes, wall, (double) bytes / wall / 1e6);
    ms_drain_ += wall;
}

void ExpertPool::run(ExpertJob* jobs, int n) {
    if (n <= 0) return;
    if (n_ == 1) {   // no workers: run inline, so a single-core machine still produces a token
        for (int i = 0; i < n; ++i) s2_expert_vnni_q(jobs[i].blob, *jobs[i].act, jobs[i].out, scratch_[0]);
        return;
    }
    // Wait for every worker to be parked BEFORE touching the batch, so the publish below is the only thing
    // that can move a worker into the drain loop.
    //
    // THE THREE PHASES ARE TIMED SEPARATELY.  They were one number, which cannot distinguish a pool that is
    // slow at the WORK from one that is slow at the SYNCHRONISATION - and those need opposite fixes.
    const auto t_a = std::chrono::steady_clock::now();
    wait_parked("before a batch");
    const auto t_b = std::chrono::steady_clock::now();
    jobs_ = jobs;
    njobs_ = n;
    mode_ = 0;
    const uint32_t e = begin_batch(n);   // done, then head (release), then the epoch: the batch is described first

    // ---- **THE HOST DRAINS TOO (R2.2), INSTEAD OF SPINNING ON `done_`.**
    //
    // The loop below used to be `while (done_ != n) strata_cpu_pause();`.  The host is pinned to core 0 - the core
    // `physical_cores(true)` deliberately keeps the five workers off - so for the whole drain that core was
    // idle while five cores did six cores' worth of work.  Measured before the change: 33.7 GB/s against
    // 5/6 x 44.14 = 36.8 for five workers and 44.14 for six.
    //
    // The host claims through the SAME `head_` counter, so this is not a second scheduler and nothing about
    // the ordering changes: `head_` is a single `fetch_add`, every job is the same size, and a thread that
    // arrives late simply claims nothing.  `done_` is still the completion signal and the host still waits for
    // it - what changed is only that the host arrives at that wait having done a share of the work.
    //
    // The host's `done_.fetch_add` is a release for the same reason a worker's is: `j.out` is read by the
    // device after `run()` returns, so the write must be published, not merely performed.
    if (host_works_) {
        for (;;) {
            const int ci = claim(e);
            if (ci < 0) break;
            hstate_.store(ci, std::memory_order_relaxed);
            hstate_ms_.store(now_ms(), std::memory_order_relaxed);
            const ExpertJob& j = jobs_[ci];
            s2_expert_vnni_q(j.blob, *j.act, j.out, host_scratch_);
            done_.fetch_add(1, std::memory_order_release);
        }
    }

    wait_done(n);
    // And park again, so the next `run` starts from a known state.  See the header for why `done` alone is
    // not enough.
    const auto t_c = std::chrono::steady_clock::now();
    wait_parked("after a batch");
    hstate_.store(kIdle, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
    const auto t_d = std::chrono::steady_clock::now();

    ms_wait_park_ += std::chrono::duration<double, std::milli>(t_b - t_a).count();
    ms_drain_ += std::chrono::duration<double, std::milli>(t_c - t_b).count();
    ms_repark_ += std::chrono::duration<double, std::milli>(t_d - t_c).count();
}

}  // namespace strata::kernels::cpu

#if defined(_WIN32)
// strata/platform/thread_name.hpp: the Windows half lives here, where <windows.h> is already included.  SetThreadDescription
// (Windows 10 1607 and later) is looked up, not linked: older SDKs and toolchains lack the import.
namespace strata::platform {
void name_this_thread(const char* base, int index) {
    char name[64];
    if (index >= 0) std::snprintf(name, sizeof name, "%s-%d", base, index);
    else std::snprintf(name, sizeof name, "%s", base);
    using SetDesc = HRESULT(WINAPI*)(HANDLE, PCWSTR);
    static const SetDesc fn = (SetDesc) GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "SetThreadDescription");
    if (fn == nullptr) return;
    wchar_t wide[64];
    int i = 0;
    for (; name[i] != '\0' && i < 63; ++i) wide[i] = (wchar_t) (unsigned char) name[i];
    wide[i] = L'\0';
    (void) fn(GetCurrentThread(), wide);
}
}  // namespace strata::platform
#endif
