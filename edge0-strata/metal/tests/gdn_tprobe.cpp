// metal/tests/gdn_tprobe.cpp - times kernels/gdn_tprobe.metal: the five MPP matmul2d products of a GDN chunk step, NC chunk steps a
// threadgroup, 192 threadgroups (h_v * 4).  Compare with gdn_scanr_4's stage (chunks of 32 -> T = 32 * NC).
//   gdn_tprobe [NC]
#include "mtl_rt.h"
#include "ref_quant.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    const int NC = argc > 1 ? std::atoi(argv[1]) : 64;
    mtl::Context cx;
    const int lib = cx.compile({"gdn_tprobe.metal"}, "#define NO_GGML_COMMON 1\n");
    std::printf("device %s, NC %d (T %d)\n", cx.device_name().c_str(), NC, NC * 32);
    const size_t NB = (size_t)48 * NC;
    mtl::Buffer out = cx.alloc(4 * 192 * 4096), W = cx.alloc(2 * NB * 32 * 128), Q = cx.alloc(2 * NB * 32 * 128),
                K = cx.alloc(2 * NB * 32 * 128), M = cx.alloc(2 * NB * 32 * 32);
    ref::Rng r(3);
    for (auto* b : {&W, &Q, &K}) for (size_t i = 0; i < NB * 32 * 128; ++i) b->as<uint16_t>()[i] = ref::f32_to_f16(r.sym() * 0.1f);
    for (size_t i = 0; i < NB * 32 * 32; ++i) M.as<uint16_t>()[i] = ref::f32_to_f16(r.sym() * 0.1f);
    for (int pr : {0, 1}) {
        mtl::Pipeline p = cx.pipeline(lib, "gdn_tprobe", pr ? mtl::Constants().i(0, pr) : mtl::Constants());
        mtl::Dispatch d; d.p = &p;
        d.args = {mtl::Arg::B(out), mtl::Arg::B(W), mtl::Arg::B(Q), mtl::Arg::B(K), mtl::Arg::B(M), mtl::Arg::V(NC)};
        d.grid[0] = 192; d.tg[0] = 128;
        cx.warm({d}, 0.3);
        std::vector<double> t;
        for (int i = 0; i < 7; ++i) t.push_back(cx.run({d}) * 1e3);
        std::sort(t.begin(), t.end());
        std::printf("probe %d (%s): %.3f ms for %d chunk steps x 192 threadgroups  (max threads %d, tg mem %d)\n", pr,
                    pr ? "SIMT parts only" : "with the five products", t[3], NC, p.max_threads, p.static_tg_mem);
    }
    return 0;
}
