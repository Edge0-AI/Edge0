// include/strata/kernels/cpu/numa_pool.hpp - when the CPU expert pool splits per NUMA node (0.2.2).
//
// Pure decisions, no threads and no globals, so a test can feed them a made-up machine:
//   * numa_cpu_node_count_at(dir)  how many NUMA nodes with CPUs a sysfs node folder lists (Linux: /sys/devices/system/node;
//                                  Windows reads GetNumaHighestNodeNumber in pool.cpp; every other system: 1)
//   * pick_numa_pool(...)          whether the split is tried without STRATA_NUMA_SPLIT, and the line that says so
//   * decide_numa_split(...)       the split's plan for a pool whose threads sit on given cores
//
// One node is exactly the code path before the split existed: pick_numa_pool answers `Unchanged` with no log line and the pool
// never reads anything else from here.  STRATA_NUMA_POOL=0 is the off switch of the default; STRATA_NUMA_SPLIT=1 / test keep
// their meaning from the contributor's branch (#3201).
#pragma once

#include "strata/kernels/cpu/pool_plan.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <system_error>
#include <vector>

namespace strata::kernels::cpu {

/// Whether text is "0", "off", "false" or "no" (any case): the spelling of every off switch of the pool.
inline bool numa_switch_off(const char* v) {
    if (v == nullptr) return false;
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s == "0" || s == "off" || s == "false" || s == "no";
}

/// The NUMA nodes with CPUs under a sysfs node folder (`node0`, `node1`, ... each holding a `cpulist`). A node without CPUs
/// (CXL or other memory-only nodes) is not counted: the pool splits threads, and a node with none has none to split.
/// A folder that is missing or has no node entries counts 0 (the caller treats 0 and 1 alike: no split).
inline int numa_cpu_node_count_at(const std::string& node_dir) {
    std::error_code ec;
    int nodes = 0;
    for (std::filesystem::directory_iterator it(node_dir, ec), end; !ec && it != end; it.increment(ec)) {
        const std::string name = it->path().filename().string();
        if (name.size() < 5 || name.compare(0, 4, "node") != 0) continue;
        if (!std::all_of(name.begin() + 4, name.end(), [](unsigned char c) { return std::isdigit(c) != 0; })) continue;
        std::ifstream f(it->path() / "cpulist");
        std::string list;
        std::getline(f, list);
        if (list.find_first_of("0123456789") != std::string::npos) ++nodes;
    }
    return nodes;
}

enum class NumaPoolChoice {
    Unchanged,   ///< nothing new: STRATA_NUMA_SPLIT (set or not) decides as it did before 0.2.2
    Off,         ///< STRATA_NUMA_POOL=0 on a machine that matters: the split is off, whatever STRATA_NUMA_SPLIT says
    Auto,        ///< two or more NUMA nodes and nothing set: try the split by itself; if it cannot run the pool stays as it was
};
struct NumaPoolPick {
    NumaPoolChoice choice = NumaPoolChoice::Unchanged;
    std::string log;   ///< the line to print ("" = print nothing)
};

/// `pool_env` is STRATA_NUMA_POOL, `split_env` STRATA_NUMA_SPLIT (nullptr or "" = unset), `nodes` the NUMA nodes with CPUs.
///   one node, nothing set                       -> Unchanged, silent (today's code path to the instruction)
///   STRATA_NUMA_POOL=0, two or more nodes       -> Off, with a line that says so
///   STRATA_NUMA_POOL=0, STRATA_NUMA_SPLIT set   -> Off (the off switch wins), with a line
///   STRATA_NUMA_SPLIT set (1, test, 0, ...)     -> Unchanged: the contributor's variable keeps its old meaning
///   two or more nodes, nothing set              -> Auto
inline NumaPoolPick pick_numa_pool(const char* pool_env, const char* split_env, int nodes) {
    const bool split_set = split_env != nullptr && *split_env != '\0';
    NumaPoolPick pick;
    if (numa_switch_off(pool_env) && (nodes >= 2 || split_set)) {
        pick.choice = NumaPoolChoice::Off;
        pick.log = "strata cpu pool: NUMA pool off (STRATA_NUMA_POOL=0): " +
                   (nodes >= 2 ? std::to_string(nodes) + " NUMA nodes found" : std::string("one NUMA node")) +
                   ", the pool runs as before";
        return pick;
    }
    if (split_set || nodes < 2) return pick;
    pick.choice = NumaPoolChoice::Auto;
    return pick;
}

struct NumaSplitDecision {
    bool on = false;
    std::string why;               ///< the reason when off
    int nodes[2] = {-1, -1};       ///< the two node ids, ascending
    std::vector<int> slice;        ///< each thread's slice (0 or 1)
    int per[2] = {0, 0};           ///< threads per slice
};

/// The real-node split of a pool whose threads sit on `thread_core` (workers first, then the host when it drains; -1 = no core).
/// `n_workers` tells a worker from the host in the messages; `pinned` is whether the pool pins its threads; `rotate` is
/// STRATA_EXCHANGE_ROTATE=1 (experts move between slots, the split needs fixed ones); `node_of(core)` is the core's NUMA node or -1.
inline NumaSplitDecision decide_numa_split(const std::vector<int>& thread_core, int n_workers, bool pinned, bool rotate,
                                           const std::function<int(int)>& node_of) {
    NumaSplitDecision d;
    const int threads = (int) thread_core.size();
    std::vector<int> node((size_t) threads, -1);
    if (threads < 2) {
        d.why = "the pool has " + std::to_string(threads) + " thread, the split needs 2 or more";
        return d;
    }
    if (rotate) {
        d.why = "STRATA_EXCHANGE_ROTATE=1 moves experts between slots, the split needs fixed slots";
        return d;
    }
    if (!pinned) {
        d.why = "the pool's threads are not pinned";
        return d;
    }
    for (int j = 0; j < threads; ++j) {
        const int c = thread_core[(size_t) j];
        if (c < 0) {
            d.why = j < n_workers ? "worker " + std::to_string(j) + " has no core to be pinned to" : "the host thread has no core";
            return d;
        }
        if ((node[(size_t) j] = node_of(c)) < 0) {
            d.why = "CPU " + std::to_string(c) + " has no known NUMA node";
            return d;
        }
    }
    std::vector<int> nodes = node;
    std::sort(nodes.begin(), nodes.end());
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
    if (nodes.size() != 2) {
        d.why = "the pool's threads span " + std::to_string(nodes.size()) + " NUMA node(s), the NUMA split supports exactly 2";
        return d;
    }
    d.nodes[0] = nodes[0];
    d.nodes[1] = nodes[1];
    d.slice.assign((size_t) threads, 0);
    for (int j = 0; j < threads; ++j) {
        d.slice[(size_t) j] = node[(size_t) j] == nodes[1] ? 1 : 0;
        ++d.per[d.slice[(size_t) j]];
    }
    if (!numa_slices_balanced(d.per[0], d.per[1])) {
        d.why = "slices too unbalanced (node " + std::to_string(nodes[0]) + ": " + std::to_string(d.per[0]) + " thread" +
                (d.per[0] == 1 ? "" : "s") + ", node " + std::to_string(nodes[1]) + ": " + std::to_string(d.per[1]) + " thread" +
                (d.per[1] == 1 ? "" : "s") + ")";
        d.nodes[0] = d.nodes[1] = -1;
        d.slice.clear();
        d.per[0] = d.per[1] = 0;
        return d;
    }
    d.on = true;
    return d;
}

}  // namespace strata::kernels::cpu
