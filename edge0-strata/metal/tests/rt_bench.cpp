// metal/tests/rt_bench.cpp - the M-series GPU's achievable read bandwidth and the cost of a dispatch.
//
//   rt_bench bw [MiB]       streaming read kernels over a buffer (default 2048 MiB, far beyond the SLC), per
//                           threadgroup size and grid size; read+write (copy); small sizes (cache-resident)
//   rt_bench dispatch       empty / dependent / small kernels: N dispatches in one serial or concurrent encoder,
//                           with or without barriers, through an indirect command buffer, and one command buffer
//                           per dispatch (commit + wait round trip)
#include "mtl_rt.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

template <class F> static void timed(int warm, int reps, F f, double& med, double& best) {
    for (int i = 0; i < warm; ++i) f();
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) t.push_back(f());
    med = median(t);
    best = *std::min_element(t.begin(), t.end());
}

static void bw(mtl::Context& cx, int lib, size_t mib) {
    const size_t bytes = mib << 20;
    mtl::Buffer src = cx.alloc(bytes);
    for (size_t i = 0; i < bytes / 4; ++i) src.as<uint32_t>()[i] = (uint32_t)(i * 2654435761u);
    mtl::Buffer out = cx.alloc(64u << 20);
    const uint32_t n = (uint32_t)(bytes / 16);
    mtl::Pipeline pr = cx.pipeline(lib, "k_read"), pu = cx.pipeline(lib, "k_read_u4"),
                  pc = cx.pipeline(lib, "k_read_chunk");
    std::printf("buffer %zu MiB; k_read max threads %d, simd width %d\n", mib, pr.max_threads, pr.simd_width);
    std::printf("%-12s %6s %8s %10s %10s\n", "kernel", "tg", "groups", "GB/s med", "GB/s best");
    double best_all = 0;
    std::string best_cfg;
    for (const mtl::Pipeline* p : {&pr, &pu}) {
        for (uint32_t tg : {64u, 128u, 256u, 512u, 1024u}) {
            if ((int)tg > p->max_threads) continue;
            for (uint32_t groups : {80u, 160u, 320u, 640u, 1280u, 2560u, 5120u}) {
                if ((size_t)groups * tg * 4 > out.bytes) continue;
                mtl::Dispatch d;
                d.p = p;
                d.args = {mtl::Arg::B(src), mtl::Arg::B(out), mtl::Arg::V(n)};
                d.grid[0] = groups;
                d.tg[0] = tg;
                double med, best;
                timed(2, 7, [&] { return cx.run({d}); }, med, best);
                const double gm = bytes / med / 1e9, gb = bytes / best / 1e9;
                std::printf("%-12s %6u %8u %10.1f %10.1f\n", p == &pr ? "read" : "read_u4", tg, groups, gm, gb);
                if (gm > best_all) {
                    best_all = gm;
                    best_cfg = std::string(p == &pr ? "read" : "read_u4") + " tg " + std::to_string(tg) + " groups " +
                               std::to_string(groups);
                }
            }
        }
    }
    // contiguous chunk per threadgroup
    for (uint32_t tg : {256u, 512u, 1024u}) {
        for (uint32_t chunk_kib : {16u, 64u, 256u, 1024u}) {
            const uint32_t chunk = chunk_kib * 1024 / 16;
            const uint32_t groups = (n + chunk - 1) / chunk;
            if ((size_t)groups * tg * 4 > out.bytes) continue;
            mtl::Dispatch d;
            d.p = &pc;
            d.args = {mtl::Arg::B(src), mtl::Arg::B(out), mtl::Arg::V(n), mtl::Arg::V(chunk)};
            d.grid[0] = groups;
            d.tg[0] = tg;
            double med, best;
            timed(2, 7, [&] { return cx.run({d}); }, med, best);
            const double gm = bytes / med / 1e9;
            std::printf("chunk %4uK   %6u %8u %10.1f %10.1f\n", chunk_kib, tg, groups, gm, bytes / best / 1e9);
            if (gm > best_all) {
                best_all = gm;
                best_cfg = "chunk " + std::to_string(chunk_kib) + "K tg " + std::to_string(tg);
            }
        }
    }
    std::printf("BEST read: %.1f GB/s (%s)\n", best_all, best_cfg.c_str());

    // read + write
    {
        mtl::Buffer dst = cx.alloc(bytes / 2);
        mtl::Pipeline pk = cx.pipeline(lib, "k_copy");
        const uint32_t nn = (uint32_t)(bytes / 2 / 16);
        double bestc = 0;
        for (uint32_t tg : {256u, 1024u})
            for (uint32_t groups : {640u, 2560u}) {
                mtl::Dispatch d;
                d.p = &pk;
                d.args = {mtl::Arg::B(src), mtl::Arg::B(dst), mtl::Arg::V(nn)};
                d.grid[0] = groups;
                d.tg[0] = tg;
                double med, best;
                timed(2, 7, [&] { return cx.run({d}); }, med, best);
                bestc = std::max(bestc, (double)bytes / med / 1e9);
            }
        std::printf("BEST copy (read+write, both counted): %.1f GB/s\n", bestc);
        cx.free(dst);
    }
    // cache-resident sizes: the same kernel on a small prefix
    for (size_t kib : {1024ul, 4096ul, 16384ul, 32768ul, 65536ul, 131072ul, 262144ul}) {
        if (kib * 1024 > bytes) break;
        const uint32_t nn = (uint32_t)(kib * 1024 / 16);
        double bestg = 0;
        for (uint32_t groups : {160u, 640u, 2560u}) {
            mtl::Dispatch d;
            d.p = &pu;
            d.args = {mtl::Arg::B(src), mtl::Arg::B(out), mtl::Arg::V(nn)};
            d.grid[0] = groups;
            d.tg[0] = 256;
            // many repetitions in one command buffer, so the launch is amortised
            const int reps = (int)std::max<size_t>(1, (256ul << 10) / kib * 4);
            double med, best;
            timed(1, 5, [&] { return cx.run({d}, reps) / reps; }, med, best);
            bestg = std::max(bestg, kib * 1024.0 / med / 1e9);
        }
        std::printf("resident %7zu KiB: %.1f GB/s\n", kib, bestg);
    }
    cx.free(src);
    cx.free(out);
}

// A read of `kib` from DRAM per dispatch (each dispatch a different, uncached chunk of a 4 GiB buffer), dependent
// dispatches in one serial encoder: the bound for a kernel that streams that much once.
static void bwsize(mtl::Context& cx, int lib) {
    const size_t big = 4ull << 30;
    mtl::Buffer src = cx.alloc(big), out = cx.alloc(64u << 20);
    mtl::Pipeline pr = cx.pipeline(lib, "k_read");
    std::printf("%10s %8s %10s %10s\n", "chunk KiB", "tg x grp", "us/disp", "GB/s");
    for (size_t kib : {1152ul, 2304ul, 4500ul, 9000ul, 13500ul, 18000ul, 36000ul, 72000ul}) {
        const size_t bytes = kib * 1024;
        const int R = (int)std::min<size_t>(400, big / bytes);
        double bestt = 1e9;
        uint32_t bg = 0, btg = 0;
        for (uint32_t tg : {256u, 1024u})
            for (uint32_t groups : {80u, 160u, 320u, 640u, 1280u}) {
                std::vector<mtl::Dispatch> ds;
                const uint32_t n = (uint32_t)(bytes / 16);
                for (int r = 0; r < R; ++r) {
                    mtl::Dispatch d;
                    d.p = &pr;
                    d.args = {mtl::Arg::B(src, (size_t)r * ((big / R) & ~4095ull)), mtl::Arg::B(out), mtl::Arg::V(n)};
                    d.grid[0] = groups;
                    d.tg[0] = tg;
                    ds.push_back(d);
                }
                double med, best;
                timed(1, 5, [&] { return cx.run(ds) / R; }, med, best);
                if (med < bestt) { bestt = med; bg = groups; btg = tg; }
            }
        std::printf("%10zu %4ux%-4u %10.2f %10.1f\n", kib, btg, bg, bestt * 1e6, bytes / bestt / 1e9);
    }
    cx.free(src);
    cx.free(out);
}

static void dispatch(mtl::Context& cx, int lib) {
    mtl::Buffer buf = cx.alloc(1 << 20), x = cx.alloc(4 * 2560), y = cx.alloc(4 * 2560);
    mtl::Pipeline pe = cx.pipeline(lib, "k_empty"), pch = cx.pipeline(lib, "k_chain"), ps = cx.pipeline(lib, "k_small");
    auto mk = [&](const mtl::Pipeline& p, int kind) {
        mtl::Dispatch d;
        d.p = &p;
        if (kind == 2) {
            d.args = {mtl::Arg::B(x), mtl::Arg::B(y)};
            d.grid[0] = 10;
            d.tg[0] = 256;
        } else {
            d.args = {mtl::Arg::B(buf)};
            d.grid[0] = 1;
            d.tg[0] = 32;
        }
        return d;
    };
    const int N = 2000;
    struct K { const char* name; const mtl::Pipeline* p; int kind; } ks[] = {
        {"empty", &pe, 0}, {"chain", &pch, 1}, {"small", &ps, 2}};
    std::printf("%-8s %-36s %10s\n", "kernel", "mode", "us/dispatch");
    for (const K& k : ks) {
        std::vector<mtl::Dispatch> ds(N, mk(*k.p, k.kind));
        double med, best;
        timed(2, 7, [&] { return cx.run(ds, 1, false, false); }, med, best);
        std::printf("%-8s %-36s %10.2f\n", k.name, "serial encoder", med / N * 1e6);
        timed(2, 7, [&] { return cx.run(ds, 1, true, false); }, med, best);
        std::printf("%-8s %-36s %10.2f\n", k.name, "concurrent encoder, no barrier", med / N * 1e6);
        timed(2, 7, [&] { return cx.run(ds, 1, true, true); }, med, best);
        std::printf("%-8s %-36s %10.2f\n", k.name, "concurrent encoder + barrier each", med / N * 1e6);
        timed(2, 7, [&] { return cx.run(ds, 1, false, true); }, med, best);
        std::printf("%-8s %-36s %10.2f\n", k.name, "serial encoder + barrier each", med / N * 1e6);
        timed(2, 7, [&] { return cx.run_icb(ds, 1, false); }, med, best);
        std::printf("%-8s %-36s %10.2f\n", k.name, "ICB, no barrier", med / N * 1e6);
        timed(2, 7, [&] { return cx.run_icb(ds, 1, true); }, med, best);
        std::printf("%-8s %-36s %10.2f\n", k.name, "ICB + barrier each", med / N * 1e6);
    }
    // command buffer round trips
    {
        std::vector<mtl::Dispatch> one(1, mk(pe, 0));
        double gpu = 0;
        const int R = 300;
        cx.run_wall(one, 20);
        const double wall = cx.run_wall(one, R, &gpu);
        std::printf("%-8s %-36s %10.2f  (GPU %.2f us)\n", "empty", "1 cmdbuf/dispatch, commit+wait wall", wall / R * 1e6,
                    gpu / R * 1e6);
        std::vector<mtl::Dispatch> d100(100, mk(pe, 0));
        const double wall2 = cx.run_wall(d100, 100, &gpu);
        std::printf("%-8s %-36s %10.2f  (GPU %.2f us)\n", "empty", "100 dispatches/cmdbuf, wall per cb", wall2 / 100 * 1e6,
                    gpu / 100 * 1e6);
    }
    // the fixed cost of one command buffer as seen by the GPU timer (1 dispatch)
    {
        std::vector<mtl::Dispatch> one(1, mk(pe, 0));
        double med, best;
        timed(5, 21, [&] { return cx.run(one); }, med, best);
        std::printf("%-8s %-36s %10.2f\n", "empty", "single dispatch cmdbuf GPU time", med * 1e6);
    }
    cx.free(buf);
    cx.free(x);
    cx.free(y);
}

int main(int argc, char** argv) {
    mtl::Context cx;
    std::printf("device: %s  apple9 %d apple10 %d metal4 %d  working set %.1f GiB\n", cx.device_name().c_str(),
                cx.has_family_apple(9), cx.has_family_apple(10), cx.supports_metal4(),
                cx.recommended_working_set() / 1073741824.0);
    const int lib = cx.compile({"rt_bench.metal"}, "#define NO_GGML_COMMON 1\n");
    const std::string mode = argc > 1 ? argv[1] : "all";
    if (mode == "bw" || mode == "all") bw(cx, lib, argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 2048);
    if (mode == "dispatch" || mode == "all") dispatch(cx, lib);
    if (mode == "bwsize" || mode == "all") bwsize(cx, lib);
    return 0;
}
