// aux_cpus_test: the CPU-set derivation of --aux-cpus (pure functions, no GPU) and, on Linux, the placement of live
// threads (a thread of the process moved to the aux set by the sweep, one that pins itself at creation, the owned ones
// left alone, and the original placement put back).
#include "strata/platform/aux_cpus.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

using namespace strata::aux_cpus;

namespace {
int failures = 0;
#define CHECK(c)                                                                                                  \
    do {                                                                                                          \
        if (!(c)) {                                                                                               \
            std::fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #c);                                   \
            ++failures;                                                                                           \
        }                                                                                                         \
    } while (0)

using V = std::vector<int>;

std::vector<LogicalCpu> cpus(std::initializer_list<std::array<long, 3>> rows) {
    std::vector<LogicalCpu> out;
    for (const auto& r : rows) out.push_back({(int) r[0], r[1], r[2]});
    return out;
}

// SMT machine: CPU i and i + n are the two threads of core i
std::vector<LogicalCpu> smt(int n_cores, int n_threads_total) {
    std::vector<LogicalCpu> out;
    for (int c = 0; c < n_threads_total; ++c) out.push_back({c, 0, c % n_cores});
    return out;
}

void test_parse() {
    V v;
    CHECK(parse_cpu_list("24", v) && v == V{24});
    CHECK(parse_cpu_list("26,24", v) && v == (V{24, 26}));
    CHECK(parse_cpu_list("24-27,30,25", v) && v == (V{24, 25, 26, 27, 30}));
    CHECK(parse_cpu_list("0", v) && v == V{0});
    for (const char* bad : {"", ",", "1,", ",1", "a", "1-", "-1", "3-1", "1,,2", "1 2", "1;2", "99999999"}) {
        CHECK(!parse_cpu_list(bad, v));
        CHECK(v.empty());
    }
    Config c;
    std::string err;
    CHECK(parse_spec("", c, err) && !c.on);
    CHECK(parse_spec("off", c, err) && !c.on);
    CHECK(parse_spec("auto", c, err) && c.on && !c.use_list);
    CHECK(parse_spec("24,26", c, err) && c.on && c.use_list && c.list == (V{24, 26}));
    CHECK(!parse_spec("fast", c, err) && !err.empty());
}

void test_derive() {
    // the production box's shape: 16 workers on 16 physical cores, the host on core 2, 24 allowed CPUs - CPUs 0-15
    // are cores 0-15 and 22-29 the second threads of cores 0-7: the host's sibling (CPU 24) is the only spare one
    {
        std::vector<LogicalCpu> a;
        for (int c = 0; c < 16; ++c) a.push_back({c, 0, c});
        for (int c = 22; c < 30; ++c) a.push_back({c, 0, c - 22});
        V workers;
        for (int c = 0; c < 16; ++c)
            if (c != 2) workers.push_back(c);
        CHECK(derive(a, 2, workers) == V{24});
        // an explicit list is taken as given, minus the host's CPU and the workers' CPUs
        CHECK(restrict_list({24, 26, 2, 5, 99}, a, 2, workers) == (V{24, 26}));
    }
    // 12 cores x 2 threads, host on CPU 0, workers on CPUs 1-7: the host's sibling first (12), then both threads of
    // the four worker-free cores 8-11; the siblings of the workers (13-19) are never offered
    {
        const auto a = smt(12, 24);
        CHECK(derive(a, 0, V{1, 2, 3, 4, 5, 6, 7}) == (V{12, 8, 9, 10, 11, 20, 21, 22, 23}));
    }
    // no SMT, every other core a worker: nothing spare
    {
        const auto a = cpus({{0, 0, 0}, {1, 0, 1}, {2, 0, 2}, {3, 0, 3}});
        CHECK(derive(a, 0, V{1, 2, 3}).empty());
    }
    // no SMT, a core without a worker: that core
    {
        const auto a = cpus({{0, 0, 0}, {1, 0, 1}, {2, 0, 2}, {3, 0, 3}});
        CHECK(derive(a, 0, V{1, 2}) == V{3});
    }
    // every CPU is a worker or the host's: nothing, with SMT too
    {
        const auto a = smt(4, 8);
        CHECK(derive(a, 0, V{1, 2, 3}) == V{4});          // only the host's sibling
        CHECK(derive(a, 0, V{1, 2, 3, 4}).empty());       // CPU 4 is a worker: the host core's sibling is taken
    }
    // two packages: core 1 of package 1 is not core 1 of package 0
    {
        const auto a = cpus({{0, 0, 0}, {1, 0, 1}, {2, 1, 0}, {3, 1, 1}});
        CHECK(derive(a, 0, V{1, 2}) == V{3});
    }
    // unknown cores (no sysfs): a CPU of unknown core could be a worker's sibling - none offered
    {
        const auto a = cpus({{0, -1, -1}, {1, -1, -1}, {2, -1, -1}, {3, -1, -1}});
        CHECK(derive(a, 0, V{1}).empty());
    }
    // the host's CPU unknown: the free cores only
    {
        const auto a = smt(4, 8);
        CHECK(derive(a, -1, V{0, 1}) == (V{2, 3, 6, 7}));
    }
    // CPUs the process may not use are not offered (a taskset): 5 and 6 are outside `a`
    {
        const auto a = cpus({{0, 0, 0}, {1, 0, 1}, {4, 0, 0}});
        CHECK(derive(a, 0, V{1}) == V{4});
    }
    CHECK(derive(std::vector<LogicalCpu>{}, 0, V{1}).empty());
}

#if defined(__linux__)
int tid_of_self() { return (int) syscall(SYS_gettid); }

std::vector<int> mask_of(int tid) {
    cpu_set_t m;
    CPU_ZERO(&m);
    std::vector<int> out;
    if (sched_getaffinity(tid, sizeof m, &m) != 0) return out;
    for (int c = 0; c < CPU_SETSIZE; ++c)
        if (CPU_ISSET(c, &m)) out.push_back(c);
    return out;
}

void test_live() {
    const std::vector<int> allowed = mask_of(0);
    if (allowed.size() < 4) {
        std::printf("aux_cpus_test: live placement skipped (needs 4 allowed CPUs, have %zu)\n", allowed.size());
        return;
    }
    const int host = allowed[0], worker = allowed[1], aux_a = allowed[2], aux_b = allowed[3];
    std::string err;
    CHECK(configure(std::to_string(aux_a) + "," + std::to_string(aux_b), err));
    // before the plan exists nothing is moved
    CHECK(sweep() == 0);
    const std::string line = set_topology(host, V{worker});
    std::printf("%s\n", line.c_str());
    CHECK(plan() == (V{aux_a, aux_b}));
    CHECK(enabled());

    // threads that stay alive until told: `floater` is created before the sweep (a CUDA driver thread's case), `owned`
    // is a pool worker (registered), `late` pins itself at creation
    std::atomic<int> tid_floater{0}, tid_owned{0}, tid_late{0};
    std::atomic<bool> stop{false};
    auto body = [&](std::atomic<int>& tid, bool owned, bool pin) {
        if (owned) note_owned_thread();
        if (pin) pin_current_thread();
        tid = tid_of_self();
        while (!stop.load()) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    };
    set_enabled(false);                                   // earlier threads are born unmoved
    CHECK(!enabled());
    std::thread floater([&] { body(tid_floater, false, false); });
    std::thread owned([&] { body(tid_owned, true, false); });
    for (int i = 0; i < 500 && (!tid_floater || !tid_owned); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    CHECK(mask_of(tid_floater) == allowed);

    set_enabled(true);                                    // the sweep
    const V aux{aux_a, aux_b};
    CHECK(mask_of(tid_floater) == aux);
    CHECK(mask_of(tid_owned) == allowed);                 // owned threads are left alone
    CHECK(mask_of(tid_of_self()) == allowed);             // so is the main thread (configure registered it)

    std::thread late([&] { body(tid_late, false, true); });
    for (int i = 0; i < 500 && !tid_late; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    CHECK(mask_of(tid_late) == aux);                      // pinned at creation

    set_enabled(false);                                   // back to the placements they had
    CHECK(mask_of(tid_floater) == allowed);
    CHECK(mask_of(tid_late) == allowed);
    CHECK(mask_of(tid_owned) == allowed);
    CHECK(sweep() == 0);                                  // off: nothing moves

    // a thread pinned to ONE worker CPU by someone else is not ours even when unregistered
    std::atomic<int> tid_pinned{0};
    std::thread pinned([&] {
        cpu_set_t m;
        CPU_ZERO(&m);
        CPU_SET(worker, &m);
        sched_setaffinity(0, sizeof m, &m);
        tid_pinned = tid_of_self();
        while (!stop.load()) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    });
    for (int i = 0; i < 500 && !tid_pinned; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    set_enabled(true);
    CHECK(mask_of(tid_pinned) == V{worker});
    CHECK(mask_of(tid_floater) == aux);
    set_enabled(false);

    stop = true;
    floater.join();
    owned.join();
    late.join();
    pinned.join();
}
#else
void test_live() { std::printf("aux_cpus_test: live placement is Linux only\n"); }
#endif

}  // namespace

int main() {
    test_parse();
    test_derive();
    test_live();
    if (failures) {
        std::fprintf(stderr, "aux_cpus_test: %d check(s) failed\n", failures);
        return 1;
    }
    std::printf("aux_cpus_test: ok\n");
    return 0;
}
