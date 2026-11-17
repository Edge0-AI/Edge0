// metal/tests/gr2_parity.cpp - the two-dispatch hyper-connection read (gr2.metal: gr_down3 + gr_up3), BF16 and Q8_0
// weights, against a double-precision reference of fused_gr.hpp's formulas on the SAME weights (the Q8_0 copy
// dequantized for its own check), plus how far the Q8_0 read lands from the BF16 one; and the speed.
//
//   gr2_parity            parity: 1, 3 and 8 tokens, with and without the folded write, BF16 and Q8_0
//   gr2_parity --bench    GPU time of the 2 dispatches (48 weight sets rotated) per tile, vs the 3-dispatch read
#include "mtl_rt.h"
#include "ref_quant.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
constexpr int N = 2560, HC = 4, LR = 320, NIN = HC * N, NR = LR + HC, MAXT = 8;
struct GrArgs { uint32_t n_embd, hc, hc_lr, n_tok, apply; float eps; };
struct GrArgs3 { uint32_t n_embd, hc, hc_lr, n_tok, apply; float eps; uint32_t rb_down, rb_up, comb, topk; };
double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
struct Err {
    double se = 0, sr = 0;
    void add(double g, double r) { se += (g - r) * (g - r); sr += r * r; }
    double l2() const { return sr > 0 ? std::sqrt(se / sr) : std::sqrt(se); }
};
}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    mtl::Context cx;
    const int lib = cx.compile({"quant.metal", "units.metal", "gr.metal", "gr2.metal"});
    std::printf("device %s\n", cx.device_name().c_str());
    const int SETS = bench ? 48 : 1;
    const size_t bd = (size_t)LR * NIN * 2, bi = (size_t)HC * NIN * 2, bu = (size_t)NIN * LR * 2;
    const size_t qd = (size_t)LR * NIN / 32 * 34, qi = (size_t)HC * NIN / 32 * 34, qu = (size_t)NIN * LR / 32 * 34;
    const size_t set = bd + bi + bu + qd + qi + qu + NIN * 4;
    mtl::Buffer W = cx.alloc(set * SETS);
    ref::Rng rng(3);
    {
        uint8_t* b = W.as<uint8_t>();
        ref::random_blocks(ref::BF16, b, LR, NIN, 11, 0.02f);
        ref::random_blocks(ref::BF16, b + bd, HC, NIN, 22, 0.02f);
        ref::random_blocks(ref::BF16, b + bd + bi, NIN, LR, 33, 0.1f);
        std::vector<float> f((size_t)LR * NIN);
        ref::dequant(ref::BF16, b, f.data(), (size_t)LR * NIN);
        ref::quantize_q8_0(f.data(), (size_t)LR * NIN, b + bd + bi + bu);
        ref::dequant(ref::BF16, b + bd, f.data(), (size_t)HC * NIN);
        ref::quantize_q8_0(f.data(), (size_t)HC * NIN, b + bd + bi + bu + qd);
        ref::dequant(ref::BF16, b + bd + bi, f.data(), (size_t)NIN * LR);
        ref::quantize_q8_0(f.data(), (size_t)NIN * LR, b + bd + bi + bu + qd + qi);
        float* wn = (float*)(b + bd + bi + bu + qd + qi + qu);
        for (int i = 0; i < NIN; ++i) wn[i] = 1.0f + 0.1f * rng.sym();
        for (int s = 1; s < SETS; ++s) std::memcpy(b + s * set, b, set);
    }
    mtl::Buffer R0 = cx.alloc(4 * MAXT * NIN), Rout = cx.alloc(4 * MAXT * NIN), bo = cx.alloc(4 * MAXT * N),
                inj = cx.alloc(4 * MAXT * HC), Dprev = cx.alloc(4 * MAXT * NR * HC), D = cx.alloc(4 * MAXT * NR * HC),
                rs = cx.alloc(4 * MAXT * HC), xp = cx.alloc(4 * (4 * MAXT * N + MAXT * N / 16)),
                xr = cx.alloc(4 * (4 * MAXT * N + MAXT * N / 16)), xpb = cx.alloc(4 * 2 * MAXT * N),
                R3 = cx.alloc(4 * MAXT * NIN), xn = cx.alloc(4 * MAXT * NIN), lo = cx.alloc(4 * MAXT * LR),
                injo = cx.alloc(4 * MAXT * HC);
    for (int i = 0; i < MAXT * NIN; ++i) R0.as<float>()[i] = rng.sym() * 2.0f;
    for (int i = 0; i < MAXT * N; ++i) bo.as<float>()[i] = rng.sym();
    for (int i = 0; i < MAXT * HC; ++i) inj.as<float>()[i] = rng.sym() * 3.0f;
    // the previous read's inject partials: stream 0 carries the whole value
    for (int t = 0; t < MAXT; ++t)
        for (int c = 0; c < HC; ++c) Dprev.as<float>()[(t * NR + LR + c) * HC] = inj.as<float>()[t * HC + c];
    mtl::Pipeline pprep = cx.pipeline(lib, "xprep");
    int fails = 0;
    for (int nt : bench ? std::vector<int>{1, 3} : std::vector<int>{1, 3, 8}) {
        for (int qq = 0; qq < 2; ++qq) {
            const int q8 = std::getenv("GR_Q8FIRST") ? 1 - qq : qq;
            const int F = q8 ? 8 : 30;
            for (int tt : bench ? std::vector<int>{1, 2, 3} : std::vector<int>{3}) {
                for (int rps : bench ? std::vector<int>{1, 2} : std::vector<int>{1}) {
                    if (bench && tt > nt) continue;
                    const char* pre = (!q8 && std::getenv("GR_PRELOAD")) ? "gr_down3b_" : "gr_down3_";
                    const std::string dn = !q8 && std::getenv("GR_PRELOAD")
                                               ? std::string(pre) + std::to_string(tt) + "_" + std::to_string(rps)
                                               : std::string(pre) + std::to_string(F) + "_" + std::to_string(tt) + "_" + std::to_string(rps);
                    mtl::Pipeline pd = cx.pipeline(lib, dn, mtl::Constants().i(1, nt)),
                                  pu = cx.pipeline(lib, "gr_up3_" + std::to_string(F), mtl::Constants().i(1, nt));
                    for (int apply = 0; apply < 2; ++apply) {
                        if (bench && !apply) continue;
                        GrArgs3 a{N, HC, LR, MAXT, (uint32_t)apply, 1e-6f, (uint32_t)(q8 ? NIN / 32 * 34 : NIN * 2),
                                  (uint32_t)(q8 ? LR / 32 * 34 : LR * 2)};
                        auto enc = [&](int s, std::vector<mtl::Dispatch>& ds) {
                            const size_t o = (size_t)s * set;
                            const size_t od = q8 ? o + bd + bi + bu : o, oi = q8 ? o + bd + bi + bu + qd : o + bd,
                                         ou = q8 ? o + bd + bi + bu + qd + qi : o + bd + bi,
                                         ow = o + bd + bi + bu + qd + qi + qu;
                            mtl::Dispatch d;
                            d.p = &pd;
                            d.args = {mtl::Arg::B(W, od), mtl::Arg::B(W, oi), mtl::Arg::B(R0), mtl::Arg::B(Rout), mtl::Arg::B(bo),
                                      mtl::Arg::B(Dprev), mtl::Arg::B(W, ow), mtl::Arg::B(D), mtl::Arg::B(rs), mtl::Arg::V(a)};
                            d.grid[0] = (NR + 8 * rps - 1) / (8 * rps);
                            d.grid[1] = HC;
                            d.tg[0] = 256;
                            ds.push_back(d);
                            mtl::Dispatch u;
                            u.p = &pu;
                            u.args = {mtl::Arg::B(W, ou), mtl::Arg::B(D), mtl::Arg::B(Rout), mtl::Arg::B(xp), mtl::Arg::V(a),
                                      mtl::Arg::B(W, ow), mtl::Arg::B(rs)};
                            u.grid[0] = N / 16;
                            u.tg[0] = 256;
                            ds.push_back(u);
                        };
                        if (bench) {
                            std::vector<mtl::Dispatch> ds;
                            for (int k = 0; k < 96; ++k) enc(k % SETS, ds);
                            if (const char* only = std::getenv("GR_ONLY")) {
                                std::vector<mtl::Dispatch> o;
                                for (size_t i = (std::string(only) == "up"); i < ds.size(); i += 2) o.push_back(ds[i]);
                                ds = o;
                            }
                            cx.warm(ds, 0.1);
                            std::vector<double> t;
                            for (int i = 0; i < 7; ++i) t.push_back(cx.run(ds) / 96);
                            const double bytes = q8 ? qd + qi + qu : bd + bi + bu;
                            std::printf("bench ntok %d %-4s TT %d RPS %d: 2 dispatches %7.2f us  %6.1f GB/s (%.2f MB)\n", nt,
                                        q8 ? "Q8_0" : "BF16", tt, rps, median(t) * 1e6, bytes / median(t) / 1e9, bytes / 1e6);
                            continue;
                        }
                        std::vector<mtl::Dispatch> ds;
                        enc(0, ds);
                        cx.run(ds);
                        const uint8_t* b = W.as<uint8_t>();
                        std::vector<float> wd((size_t)LR * NIN), wi((size_t)HC * NIN), wu((size_t)NIN * LR);
                        const int ty = q8 ? ref::Q8_0 : ref::BF16;
                        ref::dequant(ty, b + (q8 ? bd + bi + bu : 0), wd.data(), wd.size());
                        ref::dequant(ty, b + (q8 ? bd + bi + bu + qd : bd), wi.data(), wi.size());
                        ref::dequant(ty, b + (q8 ? bd + bi + bu + qd + qi : bd + bi), wu.data(), wu.size());
                        const float* wn = (const float*)(b + bd + bi + bu + qd + qi + qu);
                        Err eR, elo, einj, emix;
                        for (int t = 0; t < nt; ++t) {
                            std::vector<double> r(NIN), x(NIN);
                            for (int c = 0; c < HC; ++c) {
                                const double g = apply ? 2.0 / (1.0 + std::exp(-(double)inj.as<float>()[t * HC + c] / HC)) : 0.0;
                                double ss = 0;
                                for (int d = 0; d < N; ++d) {
                                    double v = R0.as<float>()[t * NIN + c * N + d];
                                    if (apply) v += bo.as<float>()[t * N + d] * g;
                                    r[c * N + d] = v;
                                    ss += v * v;
                                }
                                const double rsv = 1.0 / std::sqrt(ss / N + 1e-6);
                                for (int d = 0; d < N; ++d) x[c * N + d] = r[c * N + d] * wn[c * N + d] * rsv;
                            }
                            for (int i = 0; i < NIN; ++i) eR.add(Rout.as<float>()[t * NIN + i], r[i]);
                            std::vector<float> lg(LR);
                            for (int k = 0; k < LR; ++k) {
                                double s = 0;
                                for (int i = 0; i < NIN; ++i) s += (double)wd[(size_t)k * NIN + i] * x[i];
                                const double z = s / HC;
                                const float* dp = D.as<float>() + (t * NR + k) * HC;
                                const float zg = (((dp[0] + dp[1]) + dp[2]) + dp[3]) / (float)HC;
                                lg[k] = zg / (1.0f + std::exp(-zg));
                                elo.add(lg[k], z / (1.0 + std::exp(-z)));
                            }
                            for (int c = 0; c < HC; ++c) {
                                double s = 0;
                                for (int i = 0; i < NIN; ++i) s += (double)wi[(size_t)c * NIN + i] * x[i];
                                const float* dp = D.as<float>() + (t * NR + LR + c) * HC;
                                einj.add(((dp[0] + dp[1]) + dp[2]) + dp[3], s);
                            }
                            for (int d = 0; d < N; ++d) {
                                double m = 0;
                                for (int c = 0; c < HC; ++c) {
                                    double s = 0;
                                    for (int k = 0; k < LR; ++k) s += (double)wu[(size_t)(c * N + d) * LR + k] * lg[k];
                                    m += x[c * N + d] / (1.0 + std::exp(-s));
                                }
                                emix.add(xp.as<float>()[t * N + d], m / HC);
                            }
                        }
                        {
                            mtl::Dispatch d;
                            d.p = &pprep;
                            const uint32_t n = N, ntt = MAXT;
                            d.args = {mtl::Arg::B(xp), mtl::Arg::B(xr), mtl::Arg::V(n), mtl::Arg::V(ntt)};
                            d.tg[0] = 256;
                            d.grid[0] = (N / 16 * MAXT + 255) / 256;
                            cx.run({d});
                        }
                        bool planes = true;
                        for (int k = 0; k < 4; ++k)
                            planes = planes && std::memcmp(xp.as<float>() + (size_t)k * MAXT * N,
                                                           xr.as<float>() + (size_t)k * MAXT * N, 4 * (size_t)nt * N) == 0;
                        planes = planes && std::memcmp(xp.as<float>() + 4 * MAXT * N, xr.as<float>() + 4 * MAXT * N,
                                                       4 * nt * N / 16) == 0;
                        double qdiff = -1;
                        if (q8) {
                            Err e;
                            for (int i = 0; i < nt * N; ++i) e.add(xp.as<float>()[i], xpb.as<float>()[apply * MAXT * N + i]);
                            qdiff = e.l2();
                        } else {
                            std::memcpy(xpb.as<float>() + apply * MAXT * N, xp.ptr, 4 * (size_t)MAXT * N);
                        }
                        const bool ok = eR.l2() < 1e-6 && elo.l2() < 1e-5 && einj.l2() < 1e-5 && emix.l2() < 1e-5 && planes;
                        fails += !ok;
                        char extra[96] = "";
                        if (qdiff >= 0) std::snprintf(extra, sizeof extra, ", mixed vs the BF16 read %.2e", qdiff);
                        std::printf("parity ntok %d %-4s apply %d: R' %.2e, lo %.2e, inject %.2e, mixed %.2e, planes %s%s  %s\n",
                                    nt, q8 ? "Q8_0" : "BF16", apply, eR.l2(), elo.l2(), einj.l2(), emix.l2(),
                                    planes ? "bitwise" : "NO", extra, ok ? "PASS" : "FAIL");
                    }
                }
            }
        }
    }
    if (bench) {
        mtl::Pipeline pn = cx.pipeline(lib, "gr_norm"), pd = cx.pipeline(lib, "gr_down_8_8", mtl::Constants().i(1, 1)),
                      pu = cx.pipeline(lib, "gr_up", mtl::Constants().i(1, 1));
        GrArgs a{N, HC, LR, 1, 1, 1e-6f};
        std::vector<mtl::Dispatch> ds;
        for (int k = 0; k < 96; ++k) {
            const size_t o = (size_t)(k % SETS) * set;
            mtl::Dispatch d1, d2, d3;
            d1.p = &pn;
            d1.args = {mtl::Arg::B(R3), mtl::Arg::B(bo), mtl::Arg::B(inj), mtl::Arg::B(W, o + bd + bi + bu + qd + qi + qu),
                       mtl::Arg::B(xn), mtl::Arg::V(a)};
            d1.grid[0] = HC;
            d1.tg[0] = 256;
            d2.p = &pd;
            d2.args = {mtl::Arg::B(W, o), mtl::Arg::B(W, o + bd), mtl::Arg::B(xn), mtl::Arg::B(lo), mtl::Arg::B(injo),
                       mtl::Arg::V(a)};
            d2.grid[0] = LR + HC;
            d2.tg[0] = 256;
            d3.p = &pu;
            d3.args = {mtl::Arg::B(W, o + bd + bi), mtl::Arg::B(lo), mtl::Arg::B(xn), mtl::Arg::B(xp), mtl::Arg::V(a)};
            d3.grid[0] = N / 16;
            d3.tg[0] = 256;
            ds.push_back(d1);
            ds.push_back(d2);
            ds.push_back(d3);
        }
        cx.warm(ds, 0.1);
        std::vector<double> t;
        for (int i = 0; i < 7; ++i) t.push_back(cx.run(ds) / 96);
        std::printf("bench ntok 1 BF16 3-dispatch read (gr.metal): %.2f us\n", median(t) * 1e6);
        return 0;
    }
    std::printf("%s\n", fails ? "SOME FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
