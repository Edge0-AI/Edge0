// pool_plan_test: the default CPU expert worker count on mock CPUs (include/strata/kernels/cpu/pool_plan.hpp).
// Pure arithmetic, no threads and no OS calls: it runs on any machine.
#include "strata/kernels/cpu/pool_plan.hpp"

#include <cstdio>
#include <string>

namespace cpu = strata::kernels::cpu;

static int g_fail = 0;

static void expect(bool ok, const char* what) {
    if (!ok) {
        ++g_fail;
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
}

static cpu::PoolPlanInput cpu_of(int logical, std::vector<int> per_package, double quota = 0) {
    cpu::PoolPlanInput in;
    in.logical = logical;
    in.cores_per_package = std::move(per_package);
    in.cpu_quota = quota;
    return in;
}

static void line(const char* name, const cpu::PoolPlanInput& in) {
    const cpu::PoolPlan p = cpu::plan_pool_workers(in);
    std::printf("  %-34s logical %3d  cores %3d (host socket %3d, %d socket%s)  -> %2d workers  [%s]\n", name, in.logical,
                p.cores, p.cores_local, p.packages, p.packages == 1 ? "" : "s", p.workers, p.limit);
}

int main() {
    std::printf("pool_plan_test: cap %d, reserve %d from %d workers\n", cpu::kPoolWorkersCap, cpu::kPoolReserve,
                cpu::kPoolReserveFrom);

    // 8 cores / 16 threads: every physical core but the host's, never the SMT siblings
    const auto c8 = cpu_of(16, {8});
    line("8c/16t", c8);
    expect(cpu::plan_pool_workers(c8).workers == 7, "8c/16t -> 7 workers");

    // 6 cores / 12 threads (the small test PCs) and a CPU without SMT
    expect(cpu::plan_pool_workers(cpu_of(12, {6})).workers == 5, "6c/12t -> 5 workers");
    expect(cpu::plan_pool_workers(cpu_of(4, {4})).workers == 3, "4c/4t (no SMT) -> 3 workers");
    expect(cpu::plan_pool_workers(cpu_of(1, {1})).workers == 1, "1 core -> 1 worker (never 0)");
    expect(cpu::plan_pool_workers(cpu_of(2, {2})).workers == 1, "2 cores -> 1 worker");

    // 16 cores / 32 threads
    const auto c16 = cpu_of(32, {16});
    line("16c/32t", c16);
    {
        const int w = cpu::plan_pool_workers(c16).workers;
        expect(w == 15 - (15 >= cpu::kPoolReserveFrom ? cpu::kPoolReserve : 0) || w == cpu::kPoolWorkersCap,
               "16c/32t -> physical cores minus the host (and the reserve)");
        expect(w <= 15, "16c/32t never above the physical cores minus the host");
    }

    // 64 cores / 128 threads: the cap, far below the logical count
    const auto c64 = cpu_of(128, {64});
    line("64c/128t", c64);
    expect(cpu::plan_pool_workers(c64).workers == cpu::kPoolWorkersCap, "64c/128t -> the cap");
    expect(cpu::plan_pool_workers(c64).workers < 64, "64c/128t is not near the logical count");

    // 2 sockets x 64 cores / 256 threads: the user report - never 255, one socket's cores at most, under the cap
    const auto c2s = cpu_of(256, {64, 64});
    line("2-socket 128c/256t", c2s);
    expect(cpu::plan_pool_workers(c2s).workers == cpu::kPoolWorkersCap, "2-socket 128c/256t -> the cap");
    expect(cpu::plan_pool_workers(c2s).packages == 2, "2-socket: two packages");
    expect(cpu::plan_pool_workers(c2s).cores == 128, "2-socket: 128 physical cores counted");

    // 2 sockets x 20 cores (the Xeons of bench #674 #707: 17-18 workers beat 35): the host's socket only
    const auto xeon = cpu_of(80, {20, 20});
    line("2-socket 40c/80t (Xeon)", xeon);
    expect(cpu::plan_pool_workers(xeon).workers == 18, "2x20c: 18 workers (bench #674 #707: 17-18 beat 35): the host socket only, minus the host and the reserve");
    // ... and a smaller second socket does not matter; the host's socket comes first in the input
    expect(cpu::plan_pool_workers(cpu_of(48, {6, 18})).workers == 5, "host socket of 6 cores: 5 workers");

    // an unknown core layout (no sysfs, a container): half the logical processors for a many-thread CPU, never all
    const auto unk = cpu_of(256, {});
    line("256 logical, layout unknown", unk);
    expect(!cpu::plan_pool_workers(unk).topology_known, "unknown layout is flagged");
    expect(cpu::plan_pool_workers(unk).workers <= cpu::kPoolWorkersCap, "unknown 256 -> at most the cap");
    expect(cpu::plan_pool_workers(unk).workers < 255, "unknown 256 -> never 255");
    expect(cpu::plan_pool_workers(cpu_of(8, {})).workers == 7, "unknown, 8 logical -> 7 workers");

    // hybrid CPUs: with more E-cores than P-cores, the P-cores but the host's plus half of the E-cores
    {
        auto h = cpu_of(20, {20});
        h.p_cores = 8;
        h.e_cores = 12;
        line("hybrid 8P+12E (Core Ultra 7 265)", h);
        expect(cpu::plan_pool_workers(h).workers == 13, "8P+12E -> 13 workers (measured: 12-16 beat the 19 of one per core)");
        auto h2 = cpu_of(32, {24});
        h2.p_cores = 8;
        h2.e_cores = 16;
        expect(cpu::plan_pool_workers(h2).workers == 15, "8P+16E -> 15 workers (#642)");
        auto h3 = cpu_of(24, {16});
        h3.p_cores = 8;
        h3.e_cores = 8;
        expect(cpu::plan_pool_workers(h3).workers == 15, "8P+8E: not more E-cores than P-cores, every core but the host's");
    }

    // #3262: Core Ultra 7 255H (6P + 8E + 2 LP-E, no SMT).  The LP-E cores (CPUs 14-15) are in cpu_lowpower/cpus, in
    // neither the core nor the atom list; they are efficiency cores.
    {
        const std::vector<int> core = {0, 1, 2, 3, 4, 5}, atom = {6, 7, 8, 9, 10, 11, 12, 13}, lp = {14, 15};
        int e = 0, p = 0;
        for (int c = 0; c < 16; ++c) {
            const long cap = c < 6 ? 1024 : c < 14 ? 774 : 312;
            (cpu::hybrid_cpu_is_efficiency(c, cap, 1024, true, core, atom, lp) ? e : p)++;
        }
        expect(p == 6 && e == 10, "255H: 6 P-cores, 8 E + 2 LP-E counted as efficiency cores");
        expect(cpu::hybrid_cpu_is_efficiency(14, 312, 1024, true, core, atom, lp), "LP-E core 14 is an E-core");
        expect(!cpu::hybrid_cpu_is_efficiency(5, 1024, 1024, true, core, atom, lp), "P-core 5 stays a P-core");
        // a CPU in the lists of no kind falls back to its capacity (312 < 90% of 1024: an E-core)
        expect(cpu::hybrid_cpu_is_efficiency(14, 312, 1024, true, core, atom, {}), "LP-E core with no third list: capacity says E");
        // a CPU in no list with a favoured-P capacity is a P-core
        expect(!cpu::hybrid_cpu_is_efficiency(20, 1012, 1024, true, core, atom, lp), "unlisted CPU, capacity 1012/1024: P");
        // the plan with the right counts: 6P + 10E -> 5 + 5 workers, not the 15 of "every core but the host's"
        auto h = cpu_of(16, {16});
        h.p_cores = 6;
        h.e_cores = 10;
        expect(cpu::plan_pool_workers(h).workers == 10, "255H -> 10 workers (the measured 9-13 are flat; 15 is -35%)");
    }

    // a container quota
    const auto q = cpu_of(128, {64}, 4.0);
    line("64c, container quota 4 CPUs", q);
    expect(cpu::plan_pool_workers(q).workers == 3, "quota 4 CPUs -> 3 workers");
    expect(cpu::plan_pool_workers(cpu_of(128, {64}, 200.0)).workers == cpu::kPoolWorkersCap, "a big quota changes nothing");

    // warnings: only above the plan, and each says what to do
    const cpu::PoolPlan p2s = cpu::plan_pool_workers(c2s);
    expect(cpu::pool_workers_warning(0, c2s, p2s).empty(), "0 (the default) never warns");
    expect(cpu::pool_workers_warning(p2s.workers, c2s, p2s).empty(), "the plan itself never warns");
    expect(cpu::pool_workers_warning(4, c2s, p2s).empty(), "fewer than the plan never warns");
    {
        const std::string w = cpu::pool_workers_warning(256, c2s, p2s);
        expect(!w.empty(), "256 workers on 256 threads warns");
        expect(w.find("--pool-workers " + std::to_string(p2s.workers)) != std::string::npos, "the warning names the recommendation");
        expect(w.find("still runs") != std::string::npos, "the warning says it still runs");
    }
    expect(cpu::pool_workers_warning(200, c2s, p2s).find("physical cores") != std::string::npos,
           "200 workers on 128 cores: more than the physical cores");
    expect(cpu::pool_workers_warning(300, c2s, p2s).find("256 CPU threads") != std::string::npos,
           "300 workers on 256 threads: more than the CPU threads");
    expect(cpu::pool_workers_warning(30, xeon, cpu::plan_pool_workers(xeon)).find("socket") != std::string::npos,
           "30 workers on 2x20c: crosses the socket");
    expect(!cpu::pool_workers_warning(40, c64, cpu::plan_pool_workers(c64)).empty(), "40 workers on 64c is above the cap: warns");
    expect(cpu::pool_workers_warning(8, q, cpu::plan_pool_workers(q)).find("container") != std::string::npos,
           "8 workers under a 4-CPU quota: names the container limit");

    // STRATA_NUMA_SPLIT (span_packages): the default spans every socket, the other bounds stay; off = the plan above
    {
        auto x = xeon;   // 2 x 20 cores: 39 but the host's, one kept free -> 38, then the cap
        x.span_packages = true;
        line("2-socket 40c/80t, NUMA split", x);
        const cpu::PoolPlan px = cpu::plan_pool_workers(x);
        expect(px.workers == cpu::kPoolWorkersCap, "2x20c split: both sockets, then the cap");
        expect(px.cores_local == 40, "2x20c split: every socket counts as local");
        auto b = cpu_of(48, {12, 12});   // the 2 x Xeon Gold 6226 box: 23 but the host's, one kept free
        expect(cpu::plan_pool_workers(b).workers == 11, "2x12c no split: the host socket's 11 (unchanged)");
        b.span_packages = true;
        expect(cpu::plan_pool_workers(b).workers == 22, "2x12c split: 22 workers across both sockets");
        auto one = cpu_of(16, {8});
        one.span_packages = true;
        expect(cpu::plan_pool_workers(one).workers == 7, "one socket: the split changes nothing");
        // the socket warning is silenced while the split is on; the others stay
        expect(cpu::pool_workers_warning(30, x, px).empty(), "30 workers on 2x20c under the split: no socket warning");
        auto bx = cpu_of(80, {20, 20});
        bx.span_packages = true;
        const std::string w = cpu::pool_workers_warning(36, bx, cpu::plan_pool_workers(bx));
        expect(!w.empty() && w.find("socket") == std::string::npos, "36 on 2x20c under the split: the cap warning, not the socket one");
        expect(cpu::pool_workers_warning(60, bx, cpu::plan_pool_workers(bx)).find("physical cores") != std::string::npos,
               "60 on 2x20c under the split: still more than the physical cores");
    }
    // interleave_packages: round-robin over the packages (host's first), stable inside each
    {
        const std::vector<size_t> o = cpu::interleave_packages({0, 0, 0, 1, 1, 1, 1});
        const std::vector<size_t> want = {0, 3, 1, 4, 2, 5, 6};
        expect(o == want, "interleave 3+4: 0 3 1 4 2 5 6");
        expect(cpu::interleave_packages({}).empty(), "interleave of nothing");
        const std::vector<size_t> same = cpu::interleave_packages({1, 1, 1});
        expect(same == std::vector<size_t>({0, 1, 2}), "one package: the order is kept");
    }
    // a configured count under the split, on the 2 x 12-core box: the worker cores as fill_plan_fields orders them
    // (the host's socket first: its 11 cores but the host's, then the other socket's 12), taken in interleave order;
    // the host thread drains on socket 0
    {
        std::vector<int> pkg(11, 0);
        pkg.insert(pkg.end(), 12, 1);
        const std::vector<size_t> o = cpu::interleave_packages(pkg);
        auto per = [&](int n, bool interleaved, int s) {
            int c = 0;
            for (int i = 0; i < n; ++i) c += pkg[interleaved ? o[(size_t) i] : (size_t) i] == s;
            return c;
        };
        expect(per(12, true, 0) == 6 && per(12, true, 1) == 6, "--pool-workers 12 under the split: 6 + 6 workers");
        expect(cpu::numa_slices_balanced(per(12, true, 0) + 1, per(12, true, 1)), "12: 7 / 6 threads, balanced");
        expect(per(13, true, 0) == 7 && per(13, true, 1) == 6, "--pool-workers 13 under the split: 7 + 6 workers");
        expect(cpu::numa_slices_balanced(per(13, true, 0) + 1, per(13, true, 1)), "13: 8 / 6 threads, balanced");
        expect(!cpu::numa_slices_balanced(per(12, false, 0) + 1, per(12, false, 1)),
               "12 in the host-first order would be 12 / 1 threads: refused (what the interleave avoids)");
        expect(per(22, true, 0) == 11 && per(22, true, 1) == 11, "22 (the split default here): 11 + 11 workers");
    }

    // ---- --pool-cores and the start-up line (pure helpers) ----
    {
        std::vector<int> v;
        std::string err;
        expect(cpu::parse_pool_cores("0,2,4,6", v, err) && v == std::vector<int>({0, 2, 4, 6}), "list: 0,2,4,6");
        expect(cpu::parse_pool_cores("2-14:2", v, err) && v == std::vector<int>({2, 4, 6, 8, 10, 12, 14}), "list: 2-14:2");
        expect(cpu::parse_pool_cores("4-6", v, err) && v == std::vector<int>({4, 5, 6}), "list: 4-6");
        expect(cpu::parse_pool_cores(" 6 , 2 ,2, 0-1 ", v, err) && v == std::vector<int>({6, 2, 0, 1}),
               "list: spaces ok, the person's order kept, a repeated number once");
        expect(!cpu::parse_pool_cores("", v, err) && v.empty() && !err.empty(), "list: empty is an error with a message");
        expect(!cpu::parse_pool_cores("0,,2", v, err) && v.empty(), "list: an empty item");
        expect(!cpu::parse_pool_cores("0,a", v, err), "list: letters");
        expect(!cpu::parse_pool_cores("6-2", v, err), "list: a range that goes down");
        expect(!cpu::parse_pool_cores("0-8:0", v, err), "list: a step of 0");
        expect(!cpu::parse_pool_cores("-3", v, err), "list: a negative number");
        expect(!cpu::parse_pool_cores("0-99999", v, err) && v.empty(), "list: a silly range is refused with a message, not expanded");
        expect(cpu::parse_pool_cores("0-4095", v, err) && v.size() == 4096, "list: 4096 entries are fine");
        expect(cpu::format_cpu_list({2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30}) == "2,4,...,30", "format: a stride");
        expect(cpu::format_cpu_list({0, 1, 2, 3, 4}) == "0-4", "format: a run");
        expect(cpu::format_cpu_list({2, 4, 7}) == "2,4,7", "format: short and irregular lists in full");
        expect(cpu::format_cpu_list({1, 2, 4, 5, 9}) == "1,2,4,5,9", "format: irregular");
        expect(cpu::format_cpu_list({}) == "none", "format: none");

        // Windows-style numbering of 16 cores / 32 threads: core k = logical 2k and 2k+1
        std::vector<cpu::LogicalCpuInfo> win;
        for (int i = 0; i < 32; ++i) win.push_back({i, i / 2});
        std::vector<int> w;
        for (int c = 1; c < 16; ++c) w.push_back(2 * c);
        const std::string line = cpu::describe_worker_placement(w, 0, win, false);
        std::printf("  %s\n", line.c_str());
        expect(line == "CPU workers on logical processors 2,4,...,30 (15 workers, one per physical core; host on 0)",
               "the default line on 16c/32t");
        // the mistake the person worried about: workers on logical 0..7 of an adjacent-numbering SMT PC
        const std::string bad = cpu::describe_worker_placement({1, 2, 3, 4, 5, 6, 7}, 0, win, false);
        std::printf("  %s\n", bad.c_str());
        expect(bad.find("one per physical core") == std::string::npos && bad.find("carry more than one worker") != std::string::npos,
               "workers on logical 1..7 of 16c/32t are reported as sharing cores");
        expect(bad.find("shares the host's core") != std::string::npos, "... and the worker on the host's core is named");
        const std::string unk = cpu::describe_worker_placement({1, 2, 3}, 0, {}, false);
        expect(unk.find("did not say") != std::string::npos && unk.find("one per physical core") == std::string::npos,
               "unknown layout: no claim");
        expect(cpu::describe_worker_placement({2, 4}, -1, win, true).find("your --pool-cores list") != std::string::npos &&
               cpu::describe_worker_placement({2, 4}, -1, win, true).find("host not pinned") != std::string::npos,
               "a given list and an unpinned host are said");

        // check_pool_cores: drop what cannot be used, warn about siblings and the host's processor, never refuse
        cpu::PoolCoresCheck c = cpu::check_pool_cores({2, 4, 6, 8}, win, 0);
        expect(c.cpus == std::vector<int>({2, 4, 6, 8}) && c.warnings.empty(), "check: one per core, no warning");
        c = cpu::check_pool_cores({2, 3, 6}, win, 0);
        expect(c.cpus == std::vector<int>({2, 3, 6}) && c.warnings.size() == 1 && c.warnings[0].find("two hardware threads") != std::string::npos,
               "check: SMT siblings are kept with one warning");
        c = cpu::check_pool_cores({0, 2}, win, 0);
        expect(c.cpus == std::vector<int>({0, 2}) && c.warnings.size() == 1 && c.warnings[0].find("host thread's") != std::string::npos,
               "check: the host's processor is kept with a warning");
        c = cpu::check_pool_cores({1, 2}, win, 0);
        expect(c.cpus == std::vector<int>({1, 2}) && c.warnings.size() == 1 && c.warnings[0].find("shares a core with the host") != std::string::npos,
               "check: the host's SMT sibling is kept with a warning");
        c = cpu::check_pool_cores({2, 40, 99}, win, 0);
        expect(c.cpus == std::vector<int>({2}) && c.warnings.size() == 1 && c.warnings[0].find("40,99") != std::string::npos,
               "check: processors this process may not use are left out and named");
        c = cpu::check_pool_cores({40}, win, 0);
        expect(c.cpus.empty() && c.warnings.size() == 1, "check: nothing usable");
        c = cpu::check_pool_cores({2, 3}, {{2, -1}, {3, -1}}, 0);
        expect(c.cpus.size() == 2 && c.warnings.empty(), "check: an unknown layout cannot tell siblings: no warning");
    }

    if (g_fail) {
        std::fprintf(stderr, "pool_plan_test: %d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("pool_plan_test: OK\n");
    return 0;
}
