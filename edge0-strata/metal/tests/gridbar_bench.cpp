// metal/tests/gridbar_bench.cpp - phases of a streaming read separated by dispatch boundaries vs by an in-kernel
// grid barrier (gridbar.metal), and whether a phase sees the previous phase's writes of the other threadgroups.
//   gridbar_bench [MiB per phase] [phases]
#include "mtl_rt.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    const double mib = argc > 1 ? std::atof(argv[1]) : 4.5;
    const int phases = argc > 2 ? std::atoi(argv[2]) : 96;
    mtl::Context cx;
    const int lib = cx.compile({"gridbar.metal"}, std::getenv("GB_FENCE") ? "#define NO_GGML_COMMON 1\n#define GB_FENCE 1\n"
                                                                        : "#define NO_GGML_COMMON 1\n");
    mtl::Pipeline pp = cx.pipeline(lib, "k_phases"), p1 = cx.pipeline(lib, "k_phase1");
    const uint32_t n = (uint32_t)(mib * 1048576 / 16);
    mtl::Buffer src = cx.alloc((size_t)n * 16 * phases), out = cx.alloc(4u << 20), bar = cx.alloc(4),
                flag = cx.alloc(4 * 64 * (size_t)phases), bad = cx.alloc(4);
    for (size_t i = 0; i < src.bytes / 4; i += 1024) src.as<uint32_t>()[i] = (uint32_t)i;
    std::printf("device %s: %d phases of %.1f MiB\n", cx.device_name().c_str(), phases, mib);
    // dispatch boundaries
    for (uint32_t groups : {40u, 160u, 640u}) {
        std::vector<mtl::Dispatch> ds;
        for (int p = 0; p < phases; ++p) {
            mtl::Dispatch d;
            d.p = &p1;
            const uint32_t pu = p;
            d.args = {mtl::Arg::B(src), mtl::Arg::B(out), mtl::Arg::V(n), mtl::Arg::V(pu)};
            d.grid[0] = groups;
            d.tg[0] = 1024;
            ds.push_back(d);
        }
        cx.warm(ds, 0.2);
        std::vector<double> t;
        for (int i = 0; i < 5; ++i) t.push_back(cx.run(ds));
        std::sort(t.begin(), t.end());
        std::printf("dispatch per phase, %4u x 1024 threads: %7.2f us a phase  %6.1f GB/s\n", groups,
                    t[2] / phases * 1e6, n * 16.0 * phases / t[2] / 1e9);
    }
    // grid barrier, at most one threadgroup a core
    for (uint32_t groups : {10u, 20u, 40u}) {
        mtl::Dispatch d;
        d.p = &pp;
        const uint32_t ph = phases;
        d.args = {mtl::Arg::B(src), mtl::Arg::B(out), mtl::Arg::B(bar), mtl::Arg::B(flag), mtl::Arg::B(bad),
                  mtl::Arg::V(n), mtl::Arg::V(ph)};
        d.grid[0] = groups;
        d.tg[0] = 1024;
        std::vector<double> t;
        uint32_t badc = 0;
        for (int i = 0; i < 6; ++i) {
            bar.as<uint32_t>()[0] = 0;
            bad.as<uint32_t>()[0] = 0;
            for (size_t k = 0; k < flag.bytes / 4; ++k) flag.as<uint32_t>()[k] = 0xFFFFFFFFu;
            const double s = cx.run({d});
            if (i) t.push_back(s);
            badc += bad.as<uint32_t>()[0];
        }
        std::sort(t.begin(), t.end());
        std::printf("grid barrier,       %4u x 1024 threads: %7.2f us a phase  %6.1f GB/s  stale flags seen: %u\n",
                    groups, t[2] / phases * 1e6, n * 16.0 * phases / t[2] / 1e9, badc);
    }
    return 0;
}
