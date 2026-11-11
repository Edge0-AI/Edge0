// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// tests/ds4_cpu_bench.cpp - how fast does this CPU run DeepSeek-V4.1-Flash's EXL3 experts (5120 -> 2304 -> 5120, K 3)?
//
//   ds4_cpu_bench [--threads N] [--node 0|1|-1] [--mode band|rows] [--experts N] [--per-call N] [--iters N]
//
// Random trellis words in memory bound to --node (-1: wherever the first touch puts them), --threads workers pinned to
// that node's physical cores (CPU i is on node i % 2 here, CPUs 0..43 the physical cores), each call computing
// --per-call experts' gate and up (all columns) from one activation: the time of a matrix pass and its GB/s.
#include "strata/dsv41/exl3_cpu.hpp"
#include "../../src/dsv41/ds4_pool.hpp"

#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace kc = strata::dsv41k::cpu;

static void* node_alloc(size_t bytes, int node) {
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    madvise(p, bytes, MADV_HUGEPAGE);
    if (node >= 0) {
        unsigned long mask = 1ul << node;
        if (syscall(SYS_mbind, p, bytes, 2 /*MPOL_BIND*/, &mask, 64, 0) != 0) std::perror("mbind");
    }
    return p;
}

int main(int argc, char** argv) {
    int threads = 22, node = 0, n_exp = 64, per_call = 2, iters = 40;
    std::string mode = "rows";
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--threads") threads = std::atoi(argv[i + 1]);
        else if (a == "--node") node = std::atoi(argv[i + 1]);
        else if (a == "--mode") mode = argv[i + 1];
        else if (a == "--experts") n_exp = std::atoi(argv[i + 1]);
        else if (a == "--per-call") per_call = std::atoi(argv[i + 1]);
        else if (a == "--iters") iters = std::atoi(argv[i + 1]);
    }
    const int E = 5120, FF = 2304, K = 3;
    const size_t tb = (size_t) E / 16 * FF / 16 * 32 * K;   // one matrix's trellis
    std::vector<int> cpus;
    for (int c = 0; c < 44; ++c)
        if (node < 0 || c % 2 == node) cpus.push_back(c);
    strata::ds4::Pool pool(threads - 1, std::vector<int>(cpus.begin() + 1, cpus.end()));
    {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpus[0], &set);
        sched_setaffinity(0, sizeof set, &set);
    }
    uint8_t* mem = (uint8_t*) node_alloc(tb * n_exp, node);
    std::vector<uint16_t> suh(E), svh(FF);
    for (auto& v : suh) v = 0x3c00;   // 1.0
    for (auto& v : svh) v = 0x3c00;
    pool.run(n_exp, [&](int e) {
        std::mt19937 r(e);
        uint32_t* w = (uint32_t*) (mem + tb * e);
        for (size_t i = 0; i < tb / 4; ++i) w[i] = r();
    });
    std::vector<float> x(E, 0.5f), xh(E), acc((size_t) 64 * FF), y((size_t) per_call * FF);
    const float xs = 0.0f;
    kc::Exl3Part p0;
    p0.suh = suh.data();
    p0.svh = svh.data();
    p0.k = E;
    p0.n = FF;
    p0.K = K;
    kc::exl3_cpu_prepare(p0, x.data(), xh.data());
    const int GQ = E / 256;   // rows mode: pieces of 256 inputs over every column
    std::vector<float> accp((size_t) per_call * GQ * FF);
    auto t0 = std::chrono::steady_clock::now();
    long long bytes = 0;
    for (int it = 0; it < iters; ++it) {
        const int base = (it * per_call) % (n_exp - per_call + 1);
        if (mode == "band") {
            pool.run(per_call * (FF / 128), [&](int j) {
                kc::Exl3Part p = p0;
                p.trellis = (const uint32_t*) (mem + tb * (base + j / (FF / 128)));
                kc::exl3_cpu_band(p, xh.data(), xs, (j % (FF / 128)) * 128,
                                  y.data() + (size_t) (j / (FF / 128)) * FF + (j % (FF / 128)) * 128);
            });
        } else {
            pool.run(per_call * GQ, [&](int j) {
                kc::Exl3Part p = p0;
                p.trellis = (const uint32_t*) (mem + tb * (base + j / GQ));
                const int q = j % GQ;
                kc::exl3_cpu_rows_acc(p, xh.data(), q * 16, (q + 1) * 16, accp.data() + (size_t) j * FF);
            });
        }
        bytes += (long long) tb * per_call;
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%s node %d threads %d per-call %d: %.3f ms per matrix, %.1f GB/s\n", mode.c_str(), node, threads,
                per_call, ms / (iters * per_call), bytes / 1e6 / ms);
    return 0;
}
