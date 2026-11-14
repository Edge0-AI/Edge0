// metal/tests/pair_parity.cpp - gemv_pair (the shared expert's gate + up + SwiGLU in one dispatch) vs double, for
// format pairs of the packs, 1 / 3 / 8 tokens; and the raw-activation Q2_0 GEMV (type 142) the down then uses.
#include "mtl_rt.h"
#include "ref_quant.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace {
struct GemvArgs { uint32_t n_in, n_out, n_tok, row_bytes, x_stride, y_stride; };
int xkind(int t) {
    switch (t) {
        case ref::Q2_0: return 1;
        case ref::Q4_K: case ref::Q5_K: return 2;
        case ref::Q6_K: case ref::Q3_K: return 3;
        default: return 0;
    }
}
}  // namespace

int main() {
    mtl::Context cx;
    const int lib = cx.compile({"quant.metal", "units.metal", "gemv.metal"});
    mtl::Pipeline pprep = cx.pipeline(lib, "xprep");
    const int NIN = 2560, NOUT = 640, MAXT = 8;
    mtl::Buffer x = cx.alloc(4 * NIN * MAXT), xp = cx.alloc(4 * (4 * NIN * MAXT + NIN / 16 * MAXT)), h = cx.alloc(4 * NOUT * MAXT);
    ref::Rng r(4);
    for (int i = 0; i < NIN * MAXT; ++i) x.as<float>()[i] = r.sym();
    {
        mtl::Dispatch d;
        d.p = &pprep;
        const uint32_t n = NIN, nt = MAXT;
        d.args = {mtl::Arg::B(x), mtl::Arg::B(xp), mtl::Arg::V(n), mtl::Arg::V(nt)};
        d.tg[0] = 256;
        d.grid[0] = (NIN / 16 * MAXT + 255) / 256;
        cx.run({d});
    }
    const size_t plane = (size_t)4 * NIN * MAXT;
    const std::vector<std::pair<int, int>> pairs = {{ref::Q2_0, ref::Q3_K}, {ref::Q3_K, ref::Q2_0}, {ref::IQ4_XS, ref::IQ3_S},
                                                    {ref::Q4_K, ref::Q6_K}, {ref::Q5_K, ref::Q4_K}, {ref::Q6_K, ref::IQ3_S},
                                                    {ref::Q2_0, ref::Q2_0}, {ref::IQ3_S, ref::Q4_K}};
    int fails = 0;
    for (auto [fg, fu] : pairs) {
        const size_t rg = ref::row_bytes(fg, NIN), ru = ref::row_bytes(fu, NIN);
        mtl::Buffer Wg = cx.alloc(rg * NOUT), Wu = cx.alloc(ru * NOUT);
        ref::random_blocks(fg, Wg.as<uint8_t>(), NOUT, NIN, 7 + fg, 0.02f);
        ref::random_blocks(fu, Wu.as<uint8_t>(), NOUT, NIN, 9 + fu, 0.02f);
        std::vector<float> wg((size_t)NIN * NOUT), wu((size_t)NIN * NOUT);
        ref::dequant(fg, Wg.ptr, wg.data(), wg.size());
        ref::dequant(fu, Wu.ptr, wu.data(), wu.size());
        for (int nt : {1, 3, 8}) {
            mtl::Pipeline p = cx.pipeline(lib, "gemv_pair_" + std::to_string(fg) + "_" + std::to_string(fu),
                                          mtl::Constants().i(0, nt));
            GemvArgs a{(uint32_t)NIN, (uint32_t)NOUT, (uint32_t)nt, (uint32_t)rg, (uint32_t)NIN, (uint32_t)NOUT};
            const uint32_t rbu = (uint32_t)ru;
            mtl::Dispatch d;
            d.p = &p;
            d.args = {mtl::Arg::B(Wg), mtl::Arg::B(xp, plane * xkind(fg)), mtl::Arg::B(Wu), mtl::Arg::B(xp, plane * xkind(fu)),
                      mtl::Arg::B(xp, plane * 4), mtl::Arg::B(h), mtl::Arg::V(a), mtl::Arg::V(rbu)};
            d.grid[0] = NOUT / 4;
            d.tg[0] = 128;
            cx.run({d});
            double se = 0, sr = 0;
            for (int t = 0; t < nt; ++t)
                for (int o = 0; o < NOUT; ++o) {
                    double g = 0, u = 0;
                    for (int k = 0; k < NIN; ++k) {
                        g += (double)wg[(size_t)o * NIN + k] * x.as<float>()[t * NIN + k];
                        u += (double)wu[(size_t)o * NIN + k] * x.as<float>()[t * NIN + k];
                    }
                    const double ref = g / (1.0 + std::exp(-g)) * u;
                    const double e = h.as<float>()[t * NOUT + o] - ref;
                    se += e * e;
                    sr += ref * ref;
                }
            const double rel = std::sqrt(se / sr);
            fails += !(rel < 1e-5);
            std::printf("pair %-7s + %-7s ntok %d: rel L2 %.2e  %s\n", ref::type_name(fg), ref::type_name(fu), nt, rel,
                        rel < 1e-5 ? "PASS" : "FAIL");
        }
        cx.free(Wg);
        cx.free(Wu);
    }
    // the raw-activation Q2_0 GEMV
    {
        const int ni = 640, no = 2560;
        const size_t rb = ref::row_bytes(ref::Q2_0, ni);
        mtl::Buffer W = cx.alloc(rb * no), y = cx.alloc(4 * no * MAXT);
        ref::random_blocks(ref::Q2_0, W.as<uint8_t>(), no, ni, 3, 0.02f);
        std::vector<float> wf((size_t)ni * no);
        ref::dequant(ref::Q2_0, W.ptr, wf.data(), wf.size());
        mtl::Pipeline p = cx.pipeline(lib, "gemv_142_8_1_4_1", mtl::Constants().i(0, 3));
        GemvArgs a{(uint32_t)ni, (uint32_t)no, 3, (uint32_t)rb, (uint32_t)ni, (uint32_t)no};
        mtl::Dispatch d;
        d.p = &p;
        d.args = {mtl::Arg::B(W), mtl::Arg::B(x), mtl::Arg::B(x), mtl::Arg::B(y), mtl::Arg::V(a)};
        d.grid[0] = no / 16;
        d.tg[0] = 128;
        cx.run({d});
        double se = 0, sr = 0;
        for (int t = 0; t < 3; ++t)
            for (int o = 0; o < no; ++o) {
                double ref = 0;
                for (int k = 0; k < ni; ++k) ref += (double)wf[(size_t)o * ni + k] * x.as<float>()[t * ni + k];
                const double e = y.as<float>()[t * no + o] - ref;
                se += e * e;
                sr += ref * ref;
            }
        const double rel = std::sqrt(se / sr);
        fails += !(rel < 1e-5);
        std::printf("Q2_0 raw-x GEMV 640 -> 2560, 3 tokens: rel L2 %.2e  %s\n", rel, rel < 1e-5 ? "PASS" : "FAIL");
    }
    std::printf("%s\n", fails ? "SOME FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
