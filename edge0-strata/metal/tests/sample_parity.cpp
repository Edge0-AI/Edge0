// metal/tests/sample_parity.cpp - greedy argmax (sample.metal) vs the CPU over 248,320 logits with planted ties,
// and its time.
#include "mtl_rt.h"
#include "ref_quant.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    constexpr uint32_t V = 248320, PARTS = 120;
    mtl::Context cx;
    const int lib = cx.compile({"sample.metal"}, "#define NO_GGML_COMMON 1\n");
    mtl::Pipeline pp = cx.pipeline(lib, "argmax_part"), pf = cx.pipeline(lib, "argmax_final");
    const int MAXT = 8;
    mtl::Buffer lg = cx.alloc(4 * (size_t)V * MAXT), pv = cx.alloc(4 * PARTS * MAXT), pi = cx.alloc(4 * PARTS * MAXT),
                out = cx.alloc(4 * MAXT);
    ref::Rng r(5);
    int fails = 0;
    struct A { uint32_t n_vocab, parts; } a{V, PARTS};
    for (int trial = 0; trial < 20; ++trial) {
        const int nt = 1 + trial % MAXT;
        for (size_t i = 0; i < (size_t)V * nt; ++i) lg.as<float>()[i] = std::round(r.sym() * 1000.0f) / 100.0f;
        for (int t = 0; t < nt; ++t) {
            // plant a tie at the maximum
            const size_t i1 = r.next() % V, i2 = r.next() % V;
            lg.as<float>()[t * V + i1] = 20.0f;
            lg.as<float>()[t * V + i2] = 20.0f;
        }
        mtl::Dispatch d1, d2;
        d1.p = &pp;
        d1.args = {mtl::Arg::B(lg), mtl::Arg::B(pv), mtl::Arg::B(pi), mtl::Arg::V(a)};
        d1.grid[0] = PARTS;
        d1.grid[1] = nt;
        d1.tg[0] = 256;
        d2.p = &pf;
        d2.args = {mtl::Arg::B(pv), mtl::Arg::B(pi), mtl::Arg::B(out), mtl::Arg::V(a)};
        d2.grid[0] = nt;
        d2.tg[0] = 256;
        cx.run({d1, d2});
        for (int t = 0; t < nt; ++t) {
            const float* l = lg.as<float>() + (size_t)t * V;
            int best = 0;
            for (uint32_t i = 1; i < V; ++i)
                if (l[i] > l[best]) best = (int)i;
            fails += out.as<int>()[t] != best;
        }
        if (trial == 19) {
            std::vector<mtl::Dispatch> ds;
            for (int k = 0; k < 64; ++k) { ds.push_back(d1); ds.push_back(d2); }
            cx.warm(ds, 0.05);
            std::vector<double> ts;
            for (int k = 0; k < 7; ++k) ts.push_back(cx.run(ds) / 64);
            std::sort(ts.begin(), ts.end());
            std::printf("argmax %d token(s) over %u logits: %.2f us (2 dispatches)\n", nt, V, ts[3] * 1e6);
        }
    }
    std::printf("argmax parity, 20 windows with planted ties: %d wrong  %s\n", fails, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
