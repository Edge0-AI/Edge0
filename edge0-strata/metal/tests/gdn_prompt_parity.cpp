// metal/tests/gdn_prompt_parity.cpp - the prompt GDN passes (gdn_prompt.metal: gdn_pre -> gdn_scan -> gdn_post)
// against the decode step run over the same T tokens in order (gdn.metal's gdn_step, GDN_NTOK = T, itself checked
// against the CPU reference by gdn_parity): y, the final state and the conv history.  The scan reassociates
// o = S_new^T q as g S^T q + delta (k . q), so the match is to rounding, not bitwise.  Then the speed for T = 2048.
//   gdn_prompt_parity [--bench]
#include "mtl_rt.h"
#include "ref_quant.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
constexpr int HK = 16, HV = 48, CH = 10240, S = 128;
struct GdnArgs { uint32_t h_k, h_v, channels, commit; float eps; uint32_t plane; };
struct GdnPArgs { uint32_t n_tok, h_k, h_v, channels; float eps; uint32_t f32; };
double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    mtl::Context cx;
    const int lib = cx.compile({"gdn.metal", "gdn_prompt.metal"}, "#define NO_GGML_COMMON 1\n");
    std::printf("device %s\n", cx.device_name().c_str());
    const int TMAX = 2048;
    auto F = [&](size_t n) { return cx.alloc(4 * n); };
    mtl::Buffer qkv = F((size_t)TMAX * CH), al = F(TMAX * HV), be = F(TMAX * HV), z = F((size_t)TMAX * HV * S), cw = F(CH * 4),
                dt = F(HV), sa = F(HV), gam = F(S), st0 = F(S * HV * S), stA = F(S * HV * S), stB = F(S * HV * S),
                h0 = F(CH * 3), hA = F(CH * 3), hA2 = F(CH * 3), hB = F(CH * 3), yA = F((size_t)TMAX * HV * S),
                yB = F((size_t)TMAX * HV * S), yh = cx.alloc(2 * (size_t)TMAX * HV * S), qn = F((size_t)TMAX * HK * S),
                kn = F((size_t)TMAX * HK * S), vc = F((size_t)TMAX * HV * S), gb = F(2 * TMAX * HV), kq = F(TMAX * HK),
                o = F((size_t)TMAX * HV * S);
    const size_t NCM = (TMAX + 31) / 32;
    mtl::Buffer cU = F((size_t)HV * NCM * 32 * S), cW = F((size_t)HV * NCM * 32 * S), clg = F((size_t)HV * NCM * 32),
                cD = F((size_t)HV * NCM * 32 * S);
    ref::Rng r(21);
    for (size_t i = 0; i < (size_t)TMAX * CH; ++i) qkv.as<float>()[i] = r.sym() * 2.0f;
    for (int i = 0; i < TMAX * HV; ++i) { al.as<float>()[i] = r.sym() * 3.0f; be.as<float>()[i] = r.sym() * 2.0f; }
    for (size_t i = 0; i < (size_t)TMAX * HV * S; ++i) z.as<float>()[i] = r.sym() * 2.0f;
    for (int i = 0; i < CH * 4; ++i) cw.as<float>()[i] = r.sym() * 0.6f;
    for (int i = 0; i < HV; ++i) { dt.as<float>()[i] = r.sym(); sa.as<float>()[i] = -0.2f - r.uni() * 2.0f; }
    for (int i = 0; i < S; ++i) gam.as<float>()[i] = 0.5f + r.uni();
    for (int i = 0; i < S * HV * S; ++i) st0.as<float>()[i] = r.sym() * 0.1f;
    for (int i = 0; i < CH * 3; ++i) h0.as<float>()[i] = r.sym();
    auto prompt_ds = [&](int T, const mtl::Pipeline& pre, const mtl::Pipeline& scan, const mtl::Pipeline& post, int cb,
                         std::vector<mtl::Dispatch>& ds) {
        GdnPArgs a{(uint32_t)T, HK, HV, CH, 1e-6f, 1u};
        mtl::Dispatch d1;
        d1.p = &pre;
        d1.args = {mtl::Arg::B(qkv), mtl::Arg::B(hB), mtl::Arg::B(cw), mtl::Arg::B(al), mtl::Arg::B(be), mtl::Arg::B(dt),
                   mtl::Arg::B(sa), mtl::Arg::B(qn), mtl::Arg::B(kn), mtl::Arg::B(vc), mtl::Arg::B(gb), mtl::Arg::B(kq),
                   mtl::Arg::B(hA2), mtl::Arg::V(a)};
        d1.grid[0] = HK + HV / 2;
        d1.grid[1] = T;
        d1.tg[0] = 256;
        ds.push_back(d1);
        if (cb == -100) {   // the chunked form: gdn_cprep, gdn_cstate, gdn_cout
            const uint32_t nc = (T + 31) / 32;
            static mtl::Pipeline pcp = cx.pipeline(lib, "gdn_cprep"), pcs = cx.pipeline(lib, "gdn_cstate"),
                                 pco = cx.pipeline(lib, "gdn_cout");
            mtl::Dispatch e1; e1.p = &pcp;
            e1.args = {mtl::Arg::B(qn), mtl::Arg::B(kn), mtl::Arg::B(vc), mtl::Arg::B(gb), mtl::Arg::B(cU), mtl::Arg::B(cW),
                       mtl::Arg::B(clg), mtl::Arg::V(a)};
            e1.grid[0] = HV; e1.grid[1] = nc; e1.tg[0] = 256; ds.push_back(e1);
            mtl::Dispatch e2; e2.p = &pcs;
            e2.args = {mtl::Arg::B(stB), mtl::Arg::B(qn), mtl::Arg::B(kn), mtl::Arg::B(cU), mtl::Arg::B(cW), mtl::Arg::B(clg),
                       mtl::Arg::B(cD), mtl::Arg::B(o), mtl::Arg::V(a)};
            e2.grid[0] = HV * 4; e2.tg[0] = 256; ds.push_back(e2);
            mtl::Dispatch e3; e3.p = &pco;
            e3.args = {mtl::Arg::B(qn), mtl::Arg::B(kn), mtl::Arg::B(cD), mtl::Arg::B(clg), mtl::Arg::B(o), mtl::Arg::V(a)};
            e3.grid[0] = HV; e3.grid[1] = nc; e3.tg[0] = 128; ds.push_back(e3);
        } else {
        mtl::Dispatch d2;
        d2.p = &scan;
        d2.args = {mtl::Arg::B(stB), mtl::Arg::B(qn), mtl::Arg::B(kn), mtl::Arg::B(vc), mtl::Arg::B(gb), mtl::Arg::B(kq),
                   mtl::Arg::B(o), mtl::Arg::V(a)};
        // cb > 0: gdn_scan_<cb> (cb column blocks of 32 a threadgroup); cb < 0: gdn_scanr_<-cb> (-cb row blocks)
        d2.grid[0] = cb > 0 ? HV * 4 / cb : HV * 4;
        d2.tg[0] = 32 * (cb > 0 ? cb : -cb);
        ds.push_back(d2);
        }
        mtl::Dispatch d3;
        d3.p = &post;
        d3.args = {mtl::Arg::B(o), mtl::Arg::B(z), mtl::Arg::B(gam), mtl::Arg::B(yh), mtl::Arg::B(yB), mtl::Arg::V(a)};
        d3.grid[0] = HV;
        d3.grid[1] = T;
        d3.tg[0] = 128;
        ds.push_back(d3);
    };
    mtl::Pipeline pre = cx.pipeline(lib, "gdn_pre"), post = cx.pipeline(lib, "gdn_post");
    int fails = 0;
    if (!bench)
        for (int T : {1, 3, 37, 300})
            for (int cb : {1, 4, -2, -4, -8, -9, -100}) {
                mtl::Constants pc;
                if (cb < 0 && cb != -9) pc.i(6, 0);
                mtl::Pipeline scan = cx.pipeline(lib, cb == -100 ? std::string("gdn_scanr_4") : cb == -9 ? std::string("gdn_scan_v") : cb > 0 ? "gdn_scan_" + std::to_string(cb) : "gdn_scanr_" + std::to_string(-cb), cb == -100 ? mtl::Constants().i(6, 0) : pc);
                if (cb == -9) cb = -8;
                mtl::Pipeline step = cx.pipeline(lib, "gdn_step", mtl::Constants().i(2, T));
                std::copy(st0.as<float>(), st0.as<float>() + S * HV * S, stA.as<float>());
                std::copy(st0.as<float>(), st0.as<float>() + S * HV * S, stB.as<float>());
                std::copy(h0.as<float>(), h0.as<float>() + CH * 3, hA.as<float>());
                std::copy(h0.as<float>(), h0.as<float>() + CH * 3, hB.as<float>());
                GdnArgs da{HK, HV, CH, 1, 1e-6f, 0};
                mtl::Dispatch d;
                d.p = &step;
                d.args = {mtl::Arg::B(stA), mtl::Arg::B(hA), mtl::Arg::B(qkv), mtl::Arg::B(al), mtl::Arg::B(be), mtl::Arg::B(z),
                          mtl::Arg::B(cw), mtl::Arg::B(dt), mtl::Arg::B(sa), mtl::Arg::B(gam), mtl::Arg::B(yA), mtl::Arg::V(da),
                          mtl::Arg::B(hA2)};
                d.grid[0] = HV;
                d.tg[0] = 512;
                cx.run({d});
                std::vector<float> hdec(hA2.as<float>(), hA2.as<float>() + CH * 3);
                std::vector<mtl::Dispatch> ds;
                prompt_ds(T, pre, scan, post, cb, ds);
                cx.run(ds);
                auto rel = [](const float* a, const float* b, size_t n) {
                    double se = 0, sr = 0;
                    for (size_t i = 0; i < n; ++i) { se += (double)(a[i] - b[i]) * (a[i] - b[i]); sr += (double)b[i] * b[i]; }
                    return std::sqrt(se / std::max(sr, 1e-30));
                };
                const double ry = rel(yB.as<float>(), yA.as<float>(), (size_t)T * HV * S),
                             rs = rel(stB.as<float>(), stA.as<float>(), S * HV * S), rh = rel(hA2.as<float>(), hdec.data(), CH * 3);
                double ryh = 0, syh = 0;
                for (size_t i = 0; i < (size_t)T * HV * S; ++i) {
                    const double e = ref::f16_to_f32(yh.as<uint16_t>()[i]) - yA.as<float>()[i];
                    ryh += e * e;
                    syh += (double)yA.as<float>()[i] * yA.as<float>()[i];
                }
                const bool ok = ry < 1e-5 && rs < 1e-5 && rh == 0 && std::sqrt(ryh / syh) < 1e-3;
                fails += !ok;
                std::printf("T %3d, scan CB %d: y rel %.2e (fp16 %.2e), state rel %.2e, history %s  %s\n", T, cb, ry,
                            std::sqrt(ryh / syh), rs, rh == 0 ? "exact" : "DIFFERS", ok ? "PASS" : "FAIL");
            }
    if (fails || !bench) {
        std::printf("%s\n", fails ? "SOME FAILED" : "ALL PASS");
        return fails ? 1 : 0;
    }
    // speed, T = 2048: the decode step over the chunk vs the three passes (each scan width)
    const int T = 2048;
    mtl::Pipeline step = cx.pipeline(lib, "gdn_step", mtl::Constants().i(2, T));
    GdnArgs da{HK, HV, CH, 1, 1e-6f, 0};
    mtl::Dispatch d;
    d.p = &step;
    d.args = {mtl::Arg::B(stA), mtl::Arg::B(hA), mtl::Arg::B(qkv), mtl::Arg::B(al), mtl::Arg::B(be), mtl::Arg::B(z),
              mtl::Arg::B(cw), mtl::Arg::B(dt), mtl::Arg::B(sa), mtl::Arg::B(gam), mtl::Arg::B(yA), mtl::Arg::V(da), mtl::Arg::B(hA2)};
    d.grid[0] = HV;
    d.tg[0] = 512;
    auto timeit = [&](const std::vector<mtl::Dispatch>& ds) {
        cx.warm(ds, 0.3);
        std::vector<double> t;
        for (int i = 0; i < 5; ++i) t.push_back(cx.run(ds));
        return median(t);
    };
    std::printf("T %d: decode step over the chunk %.2f ms\n", T, timeit({d}) * 1e3);
    for (const char* nm : {"gdn_scanr_4", "chunked", "gdn_scanr_4", "chunked"}) {
        const bool ch = std::string(nm) == "chunked";
        const int cb = ch ? -100 : nm[9] == 'v' ? -8 : nm[8] == 'r' ? -(nm[10] - '0') : nm[9] - '0';
        mtl::Constants pc;
        if (cb < 0 && nm[9] != 'v') pc.i(6, std::getenv("GDN_PROBE") ? std::atoi(std::getenv("GDN_PROBE")) : 0);
        mtl::Pipeline scan = cx.pipeline(lib, ch ? "gdn_scanr_4" : nm, pc);
        std::vector<mtl::Dispatch> ds;
        prompt_ds(T, pre, scan, post, cb, ds);
        const double all = timeit(ds), tp = timeit({ds[0]}),
                     ts = timeit(std::vector<mtl::Dispatch>(ds.begin() + 1, ds.end() - 1)), to = timeit({ds.back()});
        if (ch) std::printf("  chunked parts: cprep %.3f, cstate %.3f, cout %.3f ms\n", timeit({ds[1]}) * 1e3, timeit({ds[2]}) * 1e3,
                            timeit({ds[3]}) * 1e3);
        std::printf("T %d, %-13s: pre %.3f + scan %.3f + post %.3f ms = %.3f ms together\n", T, nm, tp * 1e3, ts * 1e3,
                    to * 1e3, all * 1e3);
    }
    return 0;
}
