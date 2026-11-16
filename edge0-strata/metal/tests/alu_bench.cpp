// metal/tests/alu_bench.cpp - per-operation instruction rates of the GPU (Gops/s, one op = one lane-operation).
#include "mtl_rt.h"

#include <algorithm>
#include <cstdio>
#include <vector>

int main() {
    mtl::Context cx;
    const int lib = cx.compile({"alu_bench.metal"}, "#define NO_GGML_COMMON 1\n");
    const uint32_t groups = 2560, tg = 256, n = groups * tg;
    mtl::Buffer out = cx.alloc(4 * n), tab = cx.alloc(4096);
    for (int i = 0; i < 1024; ++i) tab.as<uint32_t>()[i] = i * 7;
    struct K { const char* name; double ops_per_iter; } ks[] = {
        {"a_fma32", 8}, {"a_fma16", 8}, {"a_fmamix", 8}, {"a_iand", 16}, {"a_shift", 16}, {"a_cvt", 16},
        {"a_tgload", 4}, {"a_sel", 8}, {"a_dot32", 8}, {"a_dot16", 8}, {"a_dotmix", 8}, {"a_fma32x16", 8}, {"a_devload", 4}, {"a_cstload", 4}, {"a_tgload4", 4}, {"a_popc", 4}};
    std::printf("device %s\n", cx.device_name().c_str());
    for (const K& k : ks) {
        mtl::Pipeline p = cx.pipeline(lib, k.name);
        mtl::Dispatch d;
        d.p = &p;
        const float kf = 0.999f;
        const uint32_t ku = 3;
        d.args = {mtl::Arg::B(out), std::string(k.name) == "a_fma32" || std::string(k.name) == "a_fma16" ||
                                            std::string(k.name) == "a_fmamix" || std::string(k.name).rfind("a_dot", 0) == 0 || std::string(k.name) == "a_fma32x16"
                                        ? mtl::Arg::V(kf)
                                        : mtl::Arg::V(ku)};
        if (std::string(k.name) == "a_devload" || std::string(k.name) == "a_cstload") d.args.push_back(mtl::Arg::B(tab));
        d.grid[0] = groups;
        d.tg[0] = tg;
        cx.run({d});
        std::vector<double> t;
        for (int i = 0; i < 7; ++i) t.push_back(cx.run({d}));
        std::sort(t.begin(), t.end());
        const double ops = (double)n * 1024 * k.ops_per_iter;
        std::printf("%-10s %8.1f Gops/s  (%.3f ms)\n", k.name, ops / t[3] / 1e9, t[3] * 1e3);
    }
    return 0;
}
