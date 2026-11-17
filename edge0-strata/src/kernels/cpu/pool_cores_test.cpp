// pool_cores_test: where the CPU pool's workers are placed, and --pool-cores (experimental, off by default).
// Builds real pools (no model, no GPU).  The default placement must be what detect_cpu_topology() says (one logical
// processor per physical core, the host's core left out); a list given to set_pool_cores() must be taken as it is.
#include "strata/kernels/cpu/pool.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

static int g_fail = 0;

static void expect(bool ok, const char* what) {
    if (!ok) {
        ++g_fail;
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
}

static std::string join(const std::vector<int>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
    return s;
}

int main() {
    const std::vector<cpu::LogicalCpuInfo> layout = cpu::logical_cpu_layout();
    std::printf("pool_cores_test: %zu logical processors\n", layout.size());
    expect(!layout.empty(), "the system names its logical processors");
    const cpu::CpuTopology topo = cpu::detect_cpu_topology(true);
    expect(!topo.worker_cores.empty() || topo.host_core >= 0, "the topology has cores");

    // 1. the default: nothing set, the placement is the topology's, one worker per physical core
    {
        cpu::set_pool_cores({});
        const int n = (int) std::min<size_t>(2, topo.worker_cores.size());
        if (n >= 1) {
            cpu::ExpertPool pool(n, true, true);
            expect(!pool.explicit_cores(), "default: not an explicit list");
            expect(pool.workers() == n, "default: the worker count asked for");
            expect(std::equal(topo.worker_cores.begin(), topo.worker_cores.begin() + n, pool.worker_cores().begin()),
                   "default: the workers sit on the topology's cores, in its order");
            const std::string line = pool.placement_line();
            std::printf("  default: %s\n", line.c_str());
            expect(line.rfind("CPU workers on logical processors ", 0) == 0, "the start-up line starts as documented");
            std::vector<long> cores;
            bool known = true;
            for (int i = 0; i < n; ++i) {
                long c = -1;
                for (const auto& l : layout) if (l.cpu == pool.worker_cores()[(size_t) i]) c = l.core;
                known = known && c >= 0;
                cores.push_back(c);
            }
            if (known) {
                std::sort(cores.begin(), cores.end());
                expect(std::unique(cores.begin(), cores.end()) == cores.end(), "default: no two workers on one physical core");
                expect(line.find("one per physical core") != std::string::npos, "the line says one per physical core");
            } else {
                expect(line.find("one per physical core") == std::string::npos, "unknown layout: the line does not claim it");
            }
            for (const auto& l : layout) if (l.cpu == topo.host_core)
                for (int i = 0; i < n; ++i)
                    for (const auto& m : layout)
                        if (m.cpu == pool.worker_cores()[(size_t) i] && l.core >= 0)
                            expect(m.core != l.core, "default: no worker on the host's core");
        }
    }

    // 2. --pool-cores: the list as given, in its order; the count is the list's length
    if (topo.worker_cores.size() >= 3) {
        const std::vector<int> want{topo.worker_cores[2], topo.worker_cores[0]};
        cpu::set_pool_cores(want);
        cpu::ExpertPool pool(0, true, true);
        expect(pool.explicit_cores(), "list: explicit");
        expect(pool.workers() == 2, "list: the count is the list's length");
        expect(std::equal(want.begin(), want.end(), pool.worker_cores().begin()), "list: the workers sit on the listed processors, in order");
        expect(std::string(pool.plan_limit()).find("--pool-cores") != std::string::npos, "list: the start-up line names the list");
        const std::string line = pool.placement_line();
        std::printf("  list %s: %s\n", join(want).c_str(), line.c_str());
        expect(line.find("your --pool-cores list") != std::string::npos, "list: the line says it is the person's list");

        // --pool-workers under the list: the first N entries
        cpu::ExpertPool one(1, true, true);
        expect(one.workers() == 1 && one.worker_cores()[0] == want[0], "list + --pool-workers 1: the first entry");
    }

    // 3. a processor that does not exist is left out (warning), the rest stays; nothing usable falls back to the default
    if (topo.worker_cores.size() >= 2) {
        cpu::set_pool_cores({99999, topo.worker_cores[1]});
        cpu::ExpertPool pool(0, true, true);
        expect(pool.explicit_cores() && pool.workers() == 1 && pool.worker_cores()[0] == topo.worker_cores[1],
               "an unusable entry is left out, the usable one kept");
        cpu::set_pool_cores({99999});
        cpu::ExpertPool fallback(1, true, true);
        expect(!fallback.explicit_cores() && fallback.worker_cores()[0] == topo.worker_cores[0],
               "nothing usable: the default placement");
    }

    // 4. two SMT siblings of one core are kept (and warned about by check_pool_cores, tested in pool_plan_test)
    for (const auto& a : layout) {
        const cpu::LogicalCpuInfo* b = nullptr;
        for (const auto& c : layout) if (c.cpu != a.cpu && a.core >= 0 && c.core == a.core) { b = &c; break; }
        if (b == nullptr) continue;
        cpu::set_pool_cores({a.cpu, b->cpu});
        cpu::ExpertPool pool(0, true, true);
        expect(pool.explicit_cores() && pool.workers() == 2 && pool.worker_cores()[0] == a.cpu && pool.worker_cores()[1] == b->cpu,
               "SMT siblings in the list are kept as given");
        const std::string line = pool.placement_line();
        std::printf("  siblings %d,%d: %s\n", a.cpu, b->cpu, line.c_str());
        expect(line.find("more than one worker") != std::string::npos, "the line says two workers share a core");
        break;
    }
    cpu::set_pool_cores({});

    if (g_fail) {
        std::fprintf(stderr, "pool_cores_test: %d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("pool_cores_test: OK\n");
    return 0;
}
