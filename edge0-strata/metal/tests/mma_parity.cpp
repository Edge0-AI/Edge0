// metal/tests/mma_parity.cpp - the simdgroup-matrix GEMV (mma.metal): parity vs double for every dense format, each
// token of a window bitwise a one-token call, and its speed on the packs' shapes against the per-lane GEMV.
//
//   mma_parity            parity
//   mma_parity --bench    speed, 1 / 3 / 8 tokens
#include "mtl_rt.h"
#include "ref_quant.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
struct MmaArgs { uint32_t n_in, n_out, row_bytes, x_stride, y_stride; };
struct GemvArgs { uint32_t n_in, n_out, n_tok, row_bytes, x_stride, y_stride; };
double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
int xkind(int t) {
    switch (t) {
        case ref::Q2_0: return 1;
        case ref::Q4_K: case ref::Q5_K: return 2;
        case ref::Q6_K: case ref::Q3_K: return 3;
        default: return 0;
    }
}
}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    mtl::Context cx;
    const int lib = cx.compile({"quant.metal", "units.metal", "gemv.metal", "mma.metal"});
    std::printf("device %s\n", cx.device_name().c_str());
    const int MAXT = 8;
    mtl::Pipeline pprep = cx.pipeline(lib, "xprep");
    const std::vector<int> types = {ref::BF16, ref::Q8_0, ref::Q4_0, ref::Q5_0, ref::IQ4_NL, ref::IQ4_XS, ref::Q2_0,
                                    ref::IQ3_S, ref::Q3_K, ref::Q4_K, ref::Q5_K, ref::Q6_K};
    if (!bench) {
        int fails = 0;
        for (int t : types)
            for (int n_in : {2560, 640, 6144}) {
                if (n_in % ref::block_elems(t) || (n_in == 640 && ref::block_elems(t) == 256)) continue;
                const int n_out = 100;
                const size_t rb = ref::row_bytes(t, n_in);
                mtl::Buffer W = cx.alloc(rb * n_out), x = cx.alloc(4 * n_in * MAXT), y = cx.alloc(4 * n_out * MAXT),
                            y1 = cx.alloc(4 * n_out * MAXT);
                ref::random_blocks(t, W.as<uint8_t>(), n_out, n_in, 41 + t, 0.02f);
                ref::Rng r(9);
                for (int i = 0; i < n_in * MAXT; ++i) x.as<float>()[i] = r.sym();
                std::vector<float> wf((size_t)n_in * n_out);
                ref::dequant(t, W.ptr, wf.data(), wf.size());
                for (int nt : {1, 3, 8}) {
                    mtl::Pipeline p = cx.pipeline(lib, "gemv_mma_" + std::to_string(t) + "_2", mtl::Constants().i(4, nt));
                    MmaArgs a{(uint32_t)n_in, (uint32_t)n_out, (uint32_t)rb, (uint32_t)n_in, (uint32_t)n_out};
                    mtl::Dispatch d;
                    d.p = &p;
                    d.args = {mtl::Arg::B(W), mtl::Arg::B(x), mtl::Arg::B(y), mtl::Arg::V(a)};
                    d.grid[0] = (n_out + 15) / 16;
                    d.tg[0] = 64;
                    cx.run({d});
                    double se = 0, sr = 0;
                    for (int tt = 0; tt < nt; ++tt)
                        for (int o = 0; o < n_out; ++o) {
                            double ref = 0;
                            for (int k = 0; k < n_in; ++k) ref += (double)wf[(size_t)o * n_in + k] * x.as<float>()[tt * n_in + k];
                            const double e = y.as<float>()[tt * n_out + o] - ref;
                            se += e * e;
                            sr += ref * ref;
                        }
                    // the window property: token tt's column vs a one-token call on x shifted to row 0
                    int notex = 0;
                    if (nt > 1) {
                        mtl::Pipeline p1 = cx.pipeline(lib, "gemv_mma_" + std::to_string(t) + "_2", mtl::Constants().i(4, 1));
                        for (int tt = 0; tt < nt; ++tt) {
                            mtl::Dispatch d1 = d;
                            d1.p = &p1;
                            d1.args = {mtl::Arg::B(W), mtl::Arg::B(x, 4 * (size_t)tt * n_in), mtl::Arg::B(y1), mtl::Arg::V(a)};
                            // rows past the window come from the following tokens' memory: padding only
                            cx.run({d1});
                            notex += std::memcmp(y1.ptr, y.as<float>() + tt * n_out, 4 * n_out) != 0;
                        }
                    }
                    const double rel = std::sqrt(se / sr);
                    const bool ok = rel < 1e-5 && notex == 0;
                    fails += !ok;
                    std::printf("mma parity %-7s n_in %5d ntok %d: rel L2 %.2e, window == one token: %s  %s\n",
                                ref::type_name(t), n_in, nt, rel, nt == 1 ? "-" : notex ? "NO" : "bitwise", ok ? "PASS" : "FAIL");
                }
                cx.free(W);
                cx.free(x);
                cx.free(y);
                cx.free(y1);
            }
        std::printf("%s\n", fails ? "SOME FAILED" : "ALL PASS");
        return fails ? 1 : 0;
    }
    struct Shape { const char* name; int type, n_in, n_out; };
    const std::vector<Shape> shapes = {{"gdn qkv", ref::IQ4_XS, 2560, 10240}, {"gdn qkv", ref::Q3_K, 2560, 10240},
                                       {"gdn qkv", ref::Q4_K, 2560, 10240},   {"gdn qkv", ref::Q6_K, 2560, 10240},
                                       {"gdn out", ref::Q5_K, 6144, 2560},    {"qsa q", ref::Q2_0, 2560, 12288},
                                       {"router", ref::BF16, 2560, 512},      {"lm head", ref::Q5_K, 2560, 248320}};
    std::printf("%-9s %-7s %5s | %9s %9s | %s\n", "shape", "type", "ntok", "lane us", "mma us", "mma GB/s (best SG)");
    for (const Shape& s : shapes) {
        const size_t rb = ref::row_bytes(s.type, s.n_in), mb = rb * s.n_out;
        const int copies = (int)std::max<size_t>(4, std::min<size_t>(64, (1ull << 30) / mb));
        mtl::Buffer W = cx.alloc(mb * copies), x = cx.alloc(4 * s.n_in * MAXT), y = cx.alloc(4 * (size_t)s.n_out * MAXT),
                    xp = cx.alloc(4 * (4 * s.n_in * MAXT + s.n_in / 16 * MAXT));
        {
            std::vector<uint8_t> tmp(mb);
            ref::random_blocks(s.type, tmp.data(), s.n_out, s.n_in, 3, 0.02f);
            for (int c = 0; c < copies; ++c) std::memcpy(W.as<uint8_t>() + (size_t)c * mb, tmp.data(), mb);
        }
        ref::Rng r(5);
        for (int i = 0; i < s.n_in * MAXT; ++i) x.as<float>()[i] = r.sym();
        {
            mtl::Dispatch d;
            d.p = &pprep;
            const uint32_t n = s.n_in, nt = MAXT;
            d.args = {mtl::Arg::B(x), mtl::Arg::B(xp), mtl::Arg::V(n), mtl::Arg::V(nt)};
            d.tg[0] = 256;
            d.grid[0] = (s.n_in / 16 * MAXT + 255) / 256;
            cx.run({d});
        }
        const size_t plane = (size_t)4 * s.n_in * MAXT;
        for (int nt : {1, 3, 8}) {
            auto time = [&](std::vector<mtl::Dispatch>& ds) {
                cx.warm(ds, 0.1);
                std::vector<double> t;
                for (int i = 0; i < 5; ++i) t.push_back(cx.run(ds) / ds.size());
                return median(t);
            };
            // the per-lane GEMV with its usual tile
            const int L = s.n_in >= 6144 ? 32 : (s.n_out <= 640 || s.type == ref::Q2_0 || s.type == ref::BF16) ? 32 : 16;
            const int S = s.n_in >= 6144 ? 8 : 4, K = s.n_in >= 6144 ? 2 : 1;
            mtl::Pipeline pl = cx.pipeline(lib, "gemv_" + std::to_string(s.type) + "_" + std::to_string(L) + "_1_" +
                                                    std::to_string(S) + "_" + std::to_string(K), mtl::Constants().i(0, nt));
            GemvArgs ga{(uint32_t)s.n_in, (uint32_t)s.n_out, (uint32_t)nt, (uint32_t)rb, (uint32_t)s.n_in, (uint32_t)s.n_out};
            std::vector<mtl::Dispatch> dl;
            for (int k = 0; k < std::max(copies, 16); ++k) {
                mtl::Dispatch d;
                d.p = &pl;
                d.args = {mtl::Arg::B(W, (size_t)(k % copies) * mb), mtl::Arg::B(xp, plane * xkind(s.type)),
                          mtl::Arg::B(xp, plane * 4), mtl::Arg::B(y), mtl::Arg::V(ga)};
                const int rpt = (S / K) * (32 / L);
                d.grid[0] = (s.n_out + rpt - 1) / rpt;
                d.tg[0] = S * 32;
                dl.push_back(d);
            }
            const double tl = time(dl);
            double best = 1e9;
            int bsg = 0;
            for (int sgn : {1, 2, 4}) {
                mtl::Pipeline pm = cx.pipeline(lib, "gemv_mma_" + std::to_string(s.type) + "_" + std::to_string(sgn),
                                               mtl::Constants().i(4, nt));
                MmaArgs ma{(uint32_t)s.n_in, (uint32_t)s.n_out, (uint32_t)rb, (uint32_t)s.n_in, (uint32_t)s.n_out};
                std::vector<mtl::Dispatch> dm;
                for (int k = 0; k < std::max(copies, 16); ++k) {
                    mtl::Dispatch d;
                    d.p = &pm;
                    d.args = {mtl::Arg::B(W, (size_t)(k % copies) * mb), mtl::Arg::B(x), mtl::Arg::B(y), mtl::Arg::V(ma)};
                    d.grid[0] = (s.n_out + 8 * sgn - 1) / (8 * sgn);
                    d.tg[0] = 32 * sgn;
                    dm.push_back(d);
                }
                const double tm = time(dm);
                if (tm < best) { best = tm; bsg = sgn; }
            }
            std::printf("%-9s %-7s %5d | %9.2f %9.2f | %6.1f (SG %d)  lane %6.1f GB/s\n", s.name, ref::type_name(s.type), nt,
                        tl * 1e6, best * 1e6, mb / best / 1e9, bsg, mb / tl / 1e9);
        }
        cx.free(W);
        cx.free(x);
        cx.free(y);
        cx.free(xp);
    }
    return 0;
}
