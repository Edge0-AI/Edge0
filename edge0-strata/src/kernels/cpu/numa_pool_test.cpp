// numa_pool_test: when the CPU expert pool splits per NUMA node (include/strata/kernels/cpu/numa_pool.hpp), on mock machines.
// No threads, no OS topology: a fake sysfs node folder in a temp directory and a made-up core -> node map. It runs anywhere.
#include "strata/kernels/cpu/numa_pool.hpp"

#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;
namespace fs = std::filesystem;

static int g_fail = 0;

static void expect(bool ok, const std::string& what) {
    if (!ok) {
        ++g_fail;
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    }
}

// /sys/devices/system/node with `node<i>/cpulist` files; an empty string is a memory-only node
static fs::path make_nodes(const fs::path& root, const std::vector<std::string>& cpulists, bool noise = true) {
    fs::remove_all(root);
    fs::create_directories(root);
    for (size_t i = 0; i < cpulists.size(); ++i) {
        fs::create_directories(root / ("node" + std::to_string(i)));
        std::ofstream(root / ("node" + std::to_string(i)) / "cpulist") << cpulists[i] << "\n";
    }
    if (noise) {   // files and folders in that directory that are not nodes
        std::ofstream(root / "online") << "0-1\n";
        std::ofstream(root / "possible") << "0-1\n";
        fs::create_directories(root / "nodes");
        fs::create_directories(root / "node");
        fs::create_directories(root / "nodeX");
    }
    return root;
}

int main() {
    const fs::path tmp = fs::temp_directory_path() / ("strata_numa_pool_test_" + std::to_string((long long) std::time(nullptr)));

    // ---- the node count of a sysfs node folder
    expect(cpu::numa_cpu_node_count_at((tmp / "missing").string()) == 0, "no folder: 0 nodes");
    expect(cpu::numa_cpu_node_count_at(make_nodes(tmp, {"0-23"}).string()) == 1, "one node");
    expect(cpu::numa_cpu_node_count_at(make_nodes(tmp, {"0-11,24-35", "12-23,36-47"}).string()) == 2, "two sockets: 2 nodes");
    expect(cpu::numa_cpu_node_count_at(make_nodes(tmp, {"0-7", "8-15", "16-23", "24-31"}).string()) == 4, "four nodes");
    expect(cpu::numa_cpu_node_count_at(make_nodes(tmp, {"0-23", ""}).string()) == 1, "a memory-only node (CXL) is not a node of CPUs");
    expect(cpu::numa_cpu_node_count_at(make_nodes(tmp, {"0-11", "12-23", ""}).string()) == 2, "two nodes of CPUs and a memory-only one");

    // ---- pick_numa_pool: one node is today's path
    using C = cpu::NumaPoolChoice;
    auto pick = [](const char* pool, const char* split, int nodes) { return cpu::pick_numa_pool(pool, split, nodes); };
    for (int nodes : {0, 1}) {
        for (const char* pool : {(const char*) nullptr, "", "1", "garbage"}) {
            for (const char* split : {(const char*) nullptr, ""}) {
                const auto p = pick(pool, split, nodes);
                expect(p.choice == C::Unchanged && p.log.empty(), "one node, nothing set: Unchanged and silent");
            }
        }
        const auto zero = pick("0", nullptr, nodes);
        expect(zero.choice == C::Unchanged && zero.log.empty(), "one node, STRATA_NUMA_POOL=0: Unchanged and silent");
        expect(pick(nullptr, "1", nodes).choice == C::Unchanged, "one node, STRATA_NUMA_SPLIT=1: the old variable decides");
        expect(pick(nullptr, "test", nodes).choice == C::Unchanged, "one node, STRATA_NUMA_SPLIT=test: the old variable decides");
    }
    {
        const auto p = pick(nullptr, nullptr, 2);
        expect(p.choice == C::Auto && p.log.empty(), "two nodes, nothing set: Auto");
        expect(pick("1", nullptr, 2).choice == C::Auto, "two nodes, STRATA_NUMA_POOL=1: Auto");
        expect(pick("", "", 2).choice == C::Auto, "two nodes, empty variables: Auto");
        expect(pick(nullptr, nullptr, 4).choice == C::Auto, "four nodes: Auto (the decision refuses it, with a line)");
        for (const char* off : {"0", "off", "OFF", "false", "No"}) {
            const auto o = pick(off, nullptr, 2);
            expect(o.choice == C::Off, std::string("two nodes, STRATA_NUMA_POOL=") + off + ": Off");
            expect(o.log.find("STRATA_NUMA_POOL=0") != std::string::npos && o.log.find("2 NUMA nodes") != std::string::npos,
                   "the off line names the switch and the node count");
        }
        expect(pick("0", "1", 2).choice == C::Off, "STRATA_NUMA_POOL=0 beats STRATA_NUMA_SPLIT=1");
        expect(pick("0", "test", 1).choice == C::Off, "STRATA_NUMA_POOL=0 beats STRATA_NUMA_SPLIT=test, even on one node");
        expect(pick(nullptr, "1", 2).choice == C::Unchanged, "two nodes, STRATA_NUMA_SPLIT=1: the old variable decides");
        expect(pick(nullptr, "0", 2).choice == C::Unchanged, "two nodes, STRATA_NUMA_SPLIT=0: the old variable decides (off)");
    }

    // ---- decide_numa_split on a made-up 2 x 12-core box (host core 0; cores 0-11 node 0, 12-23 node 1)
    auto node_of = [](int core) { return core < 0 ? -1 : core < 12 ? 0 : 1; };
    auto cores = [](std::initializer_list<int> v) { return std::vector<int>(v); };
    {   // 22 workers interleaved over the nodes + the host
        std::vector<int> c;
        for (int i = 0; i < 11; ++i) { c.push_back(1 + i); c.push_back(12 + i); }
        c.push_back(0);   // the host
        const auto d = cpu::decide_numa_split(c, 22, true, false, node_of);
        expect(d.on, "22 workers + host on 2 nodes: on");
        expect(d.nodes[0] == 0 && d.nodes[1] == 1, "the nodes are 0 and 1");
        expect(d.per[0] == 12 && d.per[1] == 11, "12 threads on node 0 (the host counted), 11 on node 1");
        expect(d.slice.size() == c.size() && d.slice.back() == 0, "the host drains in slice 0");
        expect(d.slice[0] == 0 && d.slice[1] == 1, "workers alternate between the slices");
    }
    {   // all workers on one node: refused
        const auto d = cpu::decide_numa_split(cores({1, 2, 3, 4, 0}), 4, true, false, node_of);
        expect(!d.on && d.why.find("span 1 NUMA node(s)") != std::string::npos, "one node's threads only: refused with the count");
    }
    {   // a node with under half of the other's threads: unbalanced
        std::vector<int> c = {1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 0};
        const auto d = cpu::decide_numa_split(c, 10, true, false, node_of);
        expect(!d.on && d.why.find("unbalanced") != std::string::npos, "10 + 1 threads: refused as unbalanced");
        expect(d.nodes[0] == -1 && d.slice.empty() && d.per[0] == 0, "a refused split leaves no plan behind");
    }
    {   // three nodes
        auto three = [](int core) { return core < 8 ? 0 : core < 16 ? 1 : 2; };
        const auto d = cpu::decide_numa_split(cores({1, 9, 17, 2, 10, 18, 0}), 6, true, false, three);
        expect(!d.on && d.why.find("span 3 NUMA node(s)") != std::string::npos && d.why.find("exactly 2") != std::string::npos,
               "three nodes: refused, exactly 2 are supported");
    }
    {
        const auto base = cores({1, 12, 2, 13, 0});
        expect(cpu::decide_numa_split(base, 4, true, false, node_of).on, "4 workers + host, balanced: on");
        const auto rot = cpu::decide_numa_split(base, 4, true, true, node_of);
        expect(!rot.on && rot.why.find("STRATA_EXCHANGE_ROTATE=1") != std::string::npos, "rotating slots: refused");
        const auto nopin = cpu::decide_numa_split(base, 4, false, false, node_of);
        expect(!nopin.on && nopin.why.find("not pinned") != std::string::npos, "unpinned threads: refused");
        const auto nocore = cpu::decide_numa_split(cores({1, -1, 2, 13}), 4, true, false, node_of);
        expect(!nocore.on && nocore.why.find("worker 1 has no core") != std::string::npos, "a worker without a core: refused");
        const auto nohost = cpu::decide_numa_split(cores({1, 12, -1}), 2, true, false, node_of);
        expect(!nohost.on && nohost.why.find("host thread has no core") != std::string::npos, "a host without a core: refused");
        auto unknown = [](int core) { return core == 12 ? -1 : 0; };
        const auto un = cpu::decide_numa_split(base, 4, true, false, unknown);
        expect(!un.on && un.why.find("CPU 12 has no known NUMA node") != std::string::npos, "a core with no node: refused");
        const auto one = cpu::decide_numa_split(cores({1}), 1, true, false, node_of);
        expect(!one.on && one.why.find("needs 2 or more") != std::string::npos, "one thread: refused");
    }
    {   // a single-node machine fed to the decision (STRATA_NUMA_SPLIT=1 on one node): refused, never half-on
        auto flat = [](int core) { return core < 0 ? -1 : 0; };
        const auto d = cpu::decide_numa_split(cores({1, 2, 3, 4, 0}), 4, true, false, flat);
        expect(!d.on, "every core on node 0: off");
    }

    fs::remove_all(tmp);
    if (g_fail) {
        std::fprintf(stderr, "numa_pool_test: %d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("numa_pool_test: OK\n");
    return 0;
}
