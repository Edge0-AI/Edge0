// metal/tests/gr_parity.cpp - the hyper-connection read (gr.metal) against a double-precision CPU reference
// (fused_gr.hpp's formulas), and its speed.
//
//   gr_parity [--ntok N]             parity for N = 1, 3, 8 (or N), with and without the folded write
//   gr_parity --bench [--ntok N]     per-kernel GPU time, 8 weight sets rotated (no cache reuse across calls)
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
constexpr int N = 2560, HC = 4, LR = 320, NIN = HC * N;
struct GrArgs { uint32_t n_embd, hc, hc_lr, n_tok, apply; float eps; };
double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
struct Err {
    double se = 0, sr = 0, mx = 0, mr = 0;
    void add(double g, double r) { se += (g - r) * (g - r); sr += r * r; mx = std::max(mx, std::fabs(g - r)); mr = std::max(mr, std::fabs(r)); }
    double l2() const { return sr > 0 ? std::sqrt(se / sr) : std::sqrt(se); }
};
}  // namespace

int main(int argc, char** argv) {
    bool bench = false;
    std::vector<int> ntoks = {1, 3, 8};
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bench") bench = true;
        else if (a == "--ntok") ntoks = {std::atoi(argv[++i])};
    }
    mtl::Context cx;
    const int lib = cx.compile({"quant.metal", "units.metal", "gr.metal"});
    std::printf("device %s\n", cx.device_name().c_str());
    const int SETS = bench ? 48 : 1;
    const bool q8 = std::getenv("GR_Q8") != nullptr;   // the Q8_0 form of the HC matrices (gr_down_q8_*, gr_up_q8)
    const int wty = q8 ? ref::Q8_0 : ref::BF16;
    // GR_INJ_F32 (with GR_Q8): the inject rows as F32 (the Unsloth files), read by gr_down_q8f_*
    const bool injf = q8 && std::getenv("GR_INJ_F32") != nullptr;
    const size_t wd_bytes = (size_t)LR * NIN * 2, wi_bytes = (size_t)HC * NIN * 4, wu_bytes = (size_t)NIN * LR * 2;
    const size_t set_bytes = wd_bytes + wi_bytes + wu_bytes + NIN * 4;
    mtl::Buffer W = cx.alloc(set_bytes * SETS);
    ref::Rng rng(3);
    for (int s = 0; s < SETS; ++s) {
        uint8_t* b = W.as<uint8_t>() + s * set_bytes;
        ref::random_blocks(ref::BF16, b, LR, NIN, 11 + s, 0.02f);
        ref::random_blocks(ref::BF16, b + wd_bytes, HC, NIN, 22 + s, 0.02f);
        ref::random_blocks(ref::BF16, b + wd_bytes + wi_bytes, NIN, LR, 33 + s, 0.1f);
        float* wn = (float*)(b + wd_bytes + wi_bytes + wu_bytes);
        for (int i = 0; i < NIN; ++i) wn[i] = 1.0f + 0.1f * rng.sym();
        if (q8) {
            // GR_Q8: the three matrices as Q8_0 (quantized from the BF16 values), each at its BF16 offset (it fits)
            const size_t offs[3] = {0, wd_bytes, wd_bytes + wi_bytes};
            const size_t ns[3] = {(size_t)LR * NIN, (size_t)HC * NIN, (size_t)NIN * LR};
            for (int m = 0; m < 3; ++m) {
                std::vector<float> f(ns[m]);
                ref::dequant(ref::BF16, b + offs[m], f.data(), ns[m]);
                if (m == 1 && injf) std::memcpy(b + offs[m], f.data(), 4 * ns[m]);
                else ref::quantize_q8_0(f.data(), ns[m], b + offs[m]);
            }
        }
    }
    const int MAXT = 8;
    // GR_DOWN=SG_KS selects gr_down's variant (default 8_8: 8 simdgroups split one row)
    const std::string grd = std::getenv("GR_DOWN") ? std::getenv("GR_DOWN") : "8_8";
    int grd_sg = 8, grd_ks = 8;
    std::sscanf(grd.c_str(), "%d_%d", &grd_sg, &grd_ks);
    const int grd_rs = grd_sg / grd_ks;
    mtl::Buffer R = cx.alloc(4 * MAXT * NIN), R0 = cx.alloc(4 * MAXT * NIN), bo = cx.alloc(4 * MAXT * N),
                inj = cx.alloc(4 * MAXT * HC), xn = cx.alloc(4 * MAXT * NIN), lo = cx.alloc(4 * MAXT * LR),
                injo = cx.alloc(4 * MAXT * HC), xp = cx.alloc(4 * (4 * MAXT * N + MAXT * N / 16)),
                xr = cx.alloc(4 * (4 * MAXT * N + MAXT * N / 16));
    for (int i = 0; i < MAXT * NIN; ++i) R0.as<float>()[i] = rng.sym() * 2.0f;
    for (int i = 0; i < MAXT * N; ++i) bo.as<float>()[i] = rng.sym();
    for (int i = 0; i < MAXT * HC; ++i) inj.as<float>()[i] = rng.sym() * 3.0f;
    mtl::Pipeline pprep = cx.pipeline(lib, "xprep");
    int fails = 0;
    for (int nt : ntoks) {
        mtl::Pipeline pn = cx.pipeline(lib, "gr_norm"),
                      pd = cx.pipeline(lib, std::string(injf ? "gr_down_q8f_" : q8 ? "gr_down_q8_" : "gr_down_") + grd, mtl::Constants().i(1, nt)),
                      pu = cx.pipeline(lib, q8 ? "gr_up_q8" : "gr_up", mtl::Constants().i(1, nt));
        for (int apply = 0; apply < 2; ++apply) {
            GrArgs a{N, HC, LR, (uint32_t)nt, (uint32_t)apply, 1e-6f};
            std::memcpy(R.ptr, R0.ptr, R.bytes);
            auto enc = [&](int set, std::vector<mtl::Dispatch>& ds, int which) {
                const size_t o = (size_t)set * set_bytes;
                mtl::Dispatch d;
                if (which & 1) {
                    d.p = &pn;
                    d.args = {mtl::Arg::B(R), mtl::Arg::B(bo), mtl::Arg::B(inj), mtl::Arg::B(W, o + wd_bytes + wi_bytes + wu_bytes),
                              mtl::Arg::B(xn), mtl::Arg::V(a)};
                    d.grid[0] = HC;
                    d.grid[1] = nt;
                    d.tg[0] = 256;
                    ds.push_back(d);
                }
                if (which & 2) {
                    mtl::Dispatch e;
                    e.p = &pd;
                    e.args = {mtl::Arg::B(W, o), mtl::Arg::B(W, o + wd_bytes), mtl::Arg::B(xn), mtl::Arg::B(lo),
                              mtl::Arg::B(injo), mtl::Arg::V(a)};
                    e.grid[0] = (LR + HC + grd_rs - 1) / grd_rs;   // RS = SG / KS rows per threadgroup
                    e.tg[0] = 32 * grd_sg;
                    ds.push_back(e);
                }
                if (which & 4) {
                    mtl::Dispatch f;
                    f.p = &pu;
                    f.args = {mtl::Arg::B(W, o + wd_bytes + wi_bytes), mtl::Arg::B(lo), mtl::Arg::B(xn), mtl::Arg::B(xp),
                              mtl::Arg::V(a)};
                    f.grid[0] = N / 16;
                    f.tg[0] = 256;
                    ds.push_back(f);
                }
            };
            if (!bench) {
                std::vector<mtl::Dispatch> ds;
                enc(0, ds, 7);
                cx.run(ds);
                // reference
                const uint8_t* b = W.as<uint8_t>();
                std::vector<float> wd((size_t)LR * NIN), wi((size_t)HC * NIN), wu((size_t)NIN * LR);
                ref::dequant(wty, b, wd.data(), wd.size());
                ref::dequant(injf ? ref::F32 : wty, b + wd_bytes, wi.data(), wi.size());
                ref::dequant(wty, b + wd_bytes + wi_bytes, wu.data(), wu.size());
                const float* wn = (const float*)(b + wd_bytes + wi_bytes + wu_bytes);
                Err eR, elo, einj, emix;
                for (int t = 0; t < nt; ++t) {
                    std::vector<double> r(NIN), x(NIN), l(LR);
                    for (int c = 0; c < HC; ++c) {
                        const double g = apply ? 2.0 / (1.0 + std::exp(-(double)inj.as<float>()[t * HC + c] / HC)) : 0.0;
                        double ss = 0;
                        for (int d = 0; d < N; ++d) {
                            double v = R0.as<float>()[t * NIN + c * N + d];
                            if (apply) v += bo.as<float>()[t * N + d] * g;
                            r[c * N + d] = v;
                            ss += v * v;
                        }
                        const double rs = 1.0 / std::sqrt(ss / N + 1e-6);
                        for (int d = 0; d < N; ++d) x[c * N + d] = r[c * N + d] * wn[c * N + d] * rs;
                    }
                    for (int i = 0; i < NIN; ++i) eR.add(R.as<float>()[t * NIN + i], r[i]);
                    for (int k = 0; k < LR; ++k) {
                        double s = 0;
                        for (int i = 0; i < NIN; ++i) s += (double)wd[(size_t)k * NIN + i] * x[i];
                        const double z = s / HC;
                        l[k] = z / (1.0 + std::exp(-z));
                        elo.add(lo.as<float>()[t * LR + k], l[k]);
                    }
                    for (int c = 0; c < HC; ++c) {
                        double s = 0;
                        for (int i = 0; i < NIN; ++i) s += (double)wi[(size_t)c * NIN + i] * x[i];
                        einj.add(injo.as<float>()[t * HC + c], s);
                    }
                    for (int d = 0; d < N; ++d) {
                        double m = 0;
                        for (int c = 0; c < HC; ++c) {
                            double s = 0;
                            // the up projection reads the GPU's lo, so the check isolates gr_up
                            for (int k = 0; k < LR; ++k) s += (double)wu[(size_t)(c * N + d) * LR + k] * lo.as<float>()[t * LR + k];
                            m += x[c * N + d] / (1.0 + std::exp(-s));
                        }
                        emix.add(xp.as<float>()[t * N + d], m / HC);
                    }
                }
                // the prepared planes must be bitwise xprep's of the raw plane
                {
                    mtl::Dispatch d;
                    d.p = &pprep;
                    const uint32_t n = N, ntt = nt;
                    d.args = {mtl::Arg::B(xp), mtl::Arg::B(xr), mtl::Arg::V(n), mtl::Arg::V(ntt)};
                    d.tg[0] = 256;
                    d.grid[0] = (N / 16 * nt + 255) / 256;
                    cx.run({d});
                }
                const size_t elems = (size_t)4 * nt * N + (size_t)nt * N / 16;
                const bool planes = std::memcmp(xp.ptr, xr.ptr, 4 * elems) == 0;
                const bool ok = eR.l2() < 1e-6 && elo.l2() < 1e-5 && einj.l2() < 1e-5 && emix.l2() < 1e-5 && planes;
                fails += !ok;
                std::printf("parity ntok %d apply %d: R rel L2 %.2e, lo %.2e, inject %.2e, mixed %.2e, prepared planes "
                            "== xprep: %s  %s\n", nt, apply, eR.l2(), elo.l2(), einj.l2(), emix.l2(),
                            planes ? "bitwise" : "NO", ok ? "PASS" : "FAIL");
            } else {
                for (int which : {1, 2, 4, 7}) {
                    std::vector<mtl::Dispatch> ds;
                    for (int r = 0; r < 96; ++r) enc(r % SETS, ds, which);
                    cx.warm(ds, 0.1);
                    std::vector<double> ts;
                    for (int i = 0; i < 7; ++i) ts.push_back(cx.run(ds) / 96);
                    const double m = median(ts);
                    const double bytes = which == 1 ? 2.0 * nt * NIN * 4 + NIN * 4 + nt * NIN * 4
                                         : which == 2 ? wd_bytes + wi_bytes
                                         : which == 4 ? wu_bytes
                                                      : wd_bytes + wi_bytes + wu_bytes;
                    std::printf("bench ntok %d apply %d %-8s %7.2f us  %6.1f GB/s (%.2f MB)\n", nt, apply,
                                which == 1 ? "norm" : which == 2 ? "down" : which == 4 ? "up" : "all 3", m * 1e6,
                                bytes / m / 1e9, bytes / 1e6);
                }
            }
        }
    }
    if (!bench) std::printf("%s\n", fails ? "SOME FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
