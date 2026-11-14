// metal/tests/gemv_parity.cpp - the dense GEMV (gemv.metal) per format: parity against double, and speed per tile
// shape on the model's projection shapes.
//
//   gemv_parity                         parity: every format, n_in 2560/640/10240, 1..8 tokens
//   gemv_parity --bench [--pack Q2_0|IQ3_XXS] [--ntok N] [--all]   speed on the pack's shapes (best tile per shape)
//   gemv_parity --one TYPE NIN NOUT [--ntok N]                      sweep one shape
#include "gguf_lite.h"
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

struct GemvArgs { uint32_t n_in, n_out, n_tok, row_bytes, x_stride, y_stride; };
struct Cfg { int L, R, S, K; };
const std::vector<Cfg> kCfgs = {{32, 1, 4, 1}, {32, 2, 4, 1}, {16, 1, 4, 1}, {16, 2, 4, 1}, {8, 1, 4, 1},
                                {8, 2, 4, 1},  {4, 2, 4, 1},  {4, 1, 4, 1},  {32, 1, 4, 4}, {32, 2, 4, 2},
                                {32, 1, 8, 8}, {16, 1, 8, 4}, {32, 1, 8, 2}, {8, 1, 8, 1},  {1, 1, 4, 1},
                                {1, 2, 4, 1},  {2, 1, 4, 1},  {2, 2, 4, 1},  {1, 1, 2, 1},  {4, 4, 4, 1}};
int rows_per_tg(Cfg c) { return (c.S / c.K) * (32 / c.L) * c.R; }
int unit_of(int t) {
    switch (t) {
        case ref::Q2_0: case ref::IQ3_S: case ref::IQ3_XXS: case ref::IQ2_XXS: case ref::IQ2_XS: case ref::IQ2_S:
        case ref::BF16: case ref::F32: return 16;
        default: return 32;
    }
}
bool cfg_ok(Cfg c, int type, int n_in) {
    const int U = n_in / unit_of(type);
    return U % c.K == 0 && U / c.K >= c.L / 2;
}
std::string kname(int t, Cfg c) {
    return "gemv_" + std::to_string(t) + "_" + std::to_string(c.L) + "_" + std::to_string(c.R) + "_" +
           std::to_string(c.S) + "_" + std::to_string(c.K);
}
double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
// which prepared x a format reads (units.metal XK_*)
int xkind(int t) {
    switch (t) {
        case ref::Q2_0: return 1;
        case ref::Q4_K: case ref::Q5_K: return 2;
        case ref::Q6_K: case ref::Q3_K: return 3;
        default: return 0;
    }
}
// x [MAXT][n] -> XP [4][MAXT][n] + S16 [MAXT][n/16]
mtl::Dispatch xprep_dispatch(const mtl::Pipeline& p, const mtl::Buffer& x, const mtl::Buffer& xp, uint32_t n, uint32_t nt) {
    mtl::Dispatch d;
    d.p = &p;
    d.args = {mtl::Arg::B(x), mtl::Arg::B(xp), mtl::Arg::V(n), mtl::Arg::V(nt)};
    d.tg[0] = 256;
    d.grid[0] = (n / 16 * nt + 255) / 256;
    return d;
}

int type_of(const std::string& s) {
    for (int t : {ref::Q2_0, ref::Q4_0, ref::Q5_0, ref::Q8_0, ref::Q3_K, ref::Q4_K, ref::Q5_K, ref::Q6_K, ref::IQ4_NL,
                  ref::IQ4_XS, ref::IQ3_S, ref::BF16})
        if (s == ref::type_name(t)) return t;
    std::fprintf(stderr, "unknown type %s\n", s.c_str());
    std::exit(2);
}

struct Shape { const char* name; int type, n_in, n_out, per_token; };   // per_token: calls per decoded token

}  // namespace

int main(int argc, char** argv) {
    bool bench = false, all = false;
    int ntok = 1;
    std::string gguf_path;
    std::string pack = "Q2_0";
    std::vector<Shape> one;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bench") bench = true;
        else if (a == "--all") all = true;
        else if (a == "--ntok") ntok = std::atoi(argv[++i]);
        else if (a == "--pack") pack = argv[++i];
        else if (a == "--gguf") gguf_path = argv[++i];
        else if (a == "--one") {
            const int t = type_of(argv[i + 1]);
            one.push_back({"one", t, std::atoi(argv[i + 2]), std::atoi(argv[i + 3]), 1});
            i += 3;
            bench = true;
            all = true;
        } else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    mtl::Context cx;
    const int lib = cx.compile({"quant.metal", "units.metal", "gemv.metal"});
    std::printf("device %s\n", cx.device_name().c_str());
    const int MAXT = 8;
    mtl::Pipeline pprep = cx.pipeline(lib, "xprep");

    if (!gguf_path.empty()) {
        // real tensors: the first type-distinct ones of the file (experts: expert 0's matrix), 1 and 3 tokens
        gguf::File f;
        std::string err;
        if (!f.open(gguf_path, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
        mtl::Buffer Wf = cx.wrap(f.map, f.map_bytes);
        std::map<int, const gguf::Tensor*> pick;
        for (const auto& kv : f.tensors) {
            const gguf::Tensor& t = kv.second;
            if (t.shape.size() < 2 || t.name.find("token_embd") != std::string::npos ||
                t.name.find("per_layer") != std::string::npos)
                continue;
            const int ty = t.type;
            const bool gemv_type = ty == ref::BF16 || ty == ref::Q8_0 || ty == ref::Q4_0 || ty == ref::Q5_0 ||
                                   ty == ref::IQ4_NL || ty == ref::IQ4_XS || ty == ref::Q2_0 || ty == ref::IQ3_S ||
                                   ty == ref::Q3_K || ty == ref::Q4_K || ty == ref::Q5_K || ty == ref::Q6_K;
            if (gemv_type && !pick.count(ty)) pick[ty] = &t;
        }
        int fails = 0;
        for (const auto& kv : pick) {
            const gguf::Tensor& t = *kv.second;
            const int ty = t.type, n_in = (int)t.shape[0], n_out = std::min<int>((int)t.shape[1], 4096);
            const size_t rb = ref::row_bytes(ty, n_in);
            mtl::Buffer x = cx.alloc(4 * n_in * MAXT), y = cx.alloc(4 * (size_t)n_out * MAXT),
                        xp = cx.alloc(4 * (4 * n_in * MAXT + n_in / 16 * MAXT));
            ref::Rng r(3);
            for (int i = 0; i < n_in * MAXT; ++i) x.as<float>()[i] = r.sym();
            cx.run({xprep_dispatch(pprep, x, xp, n_in, MAXT)});
            const size_t plane = (size_t)4 * n_in * MAXT;
            std::vector<float> wf((size_t)n_in * n_out);
            ref::dequant(ty, (const uint8_t*)f.map + t.offset, wf.data(), wf.size());
            Cfg c = n_in >= 6144 ? Cfg{32, 1, 8, 2} : n_in <= 640 ? Cfg{8, 1, 4, 1} : Cfg{16, 1, 4, 1};
            if (!cfg_ok(c, ty, n_in)) c = Cfg{32, 1, 4, 1};
            for (int nt : {1, 3}) {
                mtl::Pipeline p = cx.pipeline(lib, kname(ty, c), mtl::Constants().i(0, nt));
                GemvArgs a{(uint32_t)n_in, (uint32_t)n_out, (uint32_t)nt, (uint32_t)rb, (uint32_t)n_in, (uint32_t)n_out};
                mtl::Dispatch d;
                d.p = &p;
                d.args = {mtl::Arg::B(Wf, t.offset), mtl::Arg::B(xp, plane * xkind(ty)), mtl::Arg::B(xp, plane * 4),
                          mtl::Arg::B(y), mtl::Arg::V(a)};
                d.grid[0] = (n_out + rows_per_tg(c) - 1) / rows_per_tg(c);
                d.tg[0] = c.S * 32;
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
                const double rel = std::sqrt(se / sr);
                const bool ok = rel < 1e-5 && std::isfinite(rel);
                fails += !ok;
                std::printf("real %-7s %-34s %5d x %5d ntok %d: rel L2 %.2e  %s\n", ref::type_name(ty), t.name.c_str(),
                            n_out, n_in, nt, rel, ok ? "PASS" : "FAIL");
            }
        }
        std::printf("%s\n", fails ? "SOME FAILED" : "ALL PASS");
        return fails ? 1 : 0;
    }
    if (!bench) {
        int fails = 0;
        for (int t : {ref::F32, ref::BF16, ref::Q8_0, ref::Q4_0, ref::Q5_0, ref::IQ4_NL, ref::IQ4_XS, ref::Q2_0, ref::IQ3_S,
                      ref::Q3_K, ref::Q4_K, ref::Q5_K, ref::Q6_K}) {
            for (int n_in : {2560, 640, 10240}) {
                if (n_in % ref::block_elems(t)) continue;
                const int n_out = 100;   // not a multiple of the rows per threadgroup on purpose
                const size_t rb = ref::row_bytes(t, n_in);
                mtl::Buffer W = cx.alloc(rb * n_out), x = cx.alloc(4 * n_in * MAXT), y = cx.alloc(4 * n_out * MAXT);
                mtl::Buffer xp = cx.alloc(4 * (4 * n_in * MAXT + n_in / 16 * MAXT));
                ref::random_blocks(t, W.as<uint8_t>(), n_out, n_in, 31 + t + n_in);
                ref::Rng r(7 + n_in);
                for (int i = 0; i < n_in * MAXT; ++i) x.as<float>()[i] = r.sym();
                cx.run({xprep_dispatch(pprep, x, xp, n_in, MAXT)});
                const size_t plane = (size_t)4 * n_in * MAXT;
                std::vector<float> wf((size_t)n_in * n_out);
                ref::dequant(t, W.ptr, wf.data(), wf.size());
                for (int nt : {1, 3, 8}) {
                    double worst = 0;
                    int not_exact = 0;
                    std::string worst_cfg;
                    for (Cfg c : kCfgs) {
                        if (!cfg_ok(c, t, n_in)) continue;
                        mtl::Pipeline p = cx.pipeline(lib, kname(t, c), mtl::Constants().i(0, nt));
                        GemvArgs a{(uint32_t)n_in, (uint32_t)n_out, (uint32_t)nt, (uint32_t)rb, (uint32_t)n_in,
                                   (uint32_t)n_out};
                        std::memset(y.ptr, 0, y.bytes);
                        mtl::Dispatch d;
                        d.p = &p;
                        d.args = {mtl::Arg::B(W), mtl::Arg::B(xp, plane * xkind(t)), mtl::Arg::B(xp, plane * 4),
                                  mtl::Arg::B(y), mtl::Arg::V(a)};
                        d.grid[0] = (n_out + rows_per_tg(c) - 1) / rows_per_tg(c);
                        d.tg[0] = c.S * 32;
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
                        const double rel = std::sqrt(se / sr);
                        if (rel > worst) { worst = rel; worst_cfg = kname(t, c); }
                        if (nt > 1) {
                            // every token of a window must be bitwise what a one-token call gives
                            std::vector<float> ywin((float*)y.ptr, (float*)y.ptr + nt * n_out);
                            mtl::Pipeline p1 = cx.pipeline(lib, kname(t, c), mtl::Constants().i(0, 1));
                            GemvArgs a1 = a;
                            a1.n_tok = 1;
                            for (int tt = 0; tt < nt; ++tt) {
                                mtl::Dispatch d1 = d;
                                d1.p = &p1;
                                d1.args = {mtl::Arg::B(W), mtl::Arg::B(xp, plane * xkind(t) + 4 * (size_t)tt * n_in),
                                           mtl::Arg::B(xp, plane * 4 + 4 * (size_t)tt * (n_in / 16)), mtl::Arg::B(y),
                                           mtl::Arg::V(a1)};
                                cx.run({d1});
                                if (std::memcmp(y.ptr, ywin.data() + tt * n_out, 4 * n_out) != 0) ++not_exact;
                            }
                        }
                    }
                    const bool ok = worst < 1e-5 && not_exact == 0;
                    fails += !ok;
                    std::printf("parity %-7s n_in %5d ntok %d: worst rel L2 over %zu tiles %.2e (%s), window == one "
                                "token: %s  %s\n",
                                ref::type_name(t), n_in, nt, kCfgs.size(), worst, worst_cfg.c_str(),
                                nt == 1 ? "-" : not_exact ? "NO" : "bitwise", ok ? "PASS" : "FAIL");
                }
                cx.free(W);
                cx.free(x);
                cx.free(y);
                cx.free(xp);
            }
        }
        std::printf("%s\n", fails ? "SOME FAILED" : "ALL PASS");
        return fails ? 1 : 0;
    }

    // ---- bench
    std::vector<Shape> shapes = one;
    if (shapes.empty()) {
        if (pack == "Q2_0")
            shapes = {
                {"hc down (x4/layer)", ref::BF16, 10240, 320, 4 * 48}, {"hc up (x4/layer)", ref::BF16, 320, 10240, 4 * 48},
                {"router", ref::BF16, 2560, 512, 48},                 {"gdn qkv", ref::IQ4_XS, 2560, 10240, 13},
                {"gdn qkv", ref::Q3_K, 2560, 10240, 18},              {"gdn qkv", ref::Q4_K, 2560, 10240, 4},
                {"gdn gate", ref::Q3_K, 2560, 6144, 25},              {"gdn out", ref::IQ4_XS, 6144, 2560, 17},
                {"gdn out", ref::Q4_K, 6144, 2560, 11},               {"qsa q", ref::Q3_K, 2560, 12288, 6},
                {"qsa out", ref::Q5_K, 6144, 2560, 4},                {"qsa k", ref::Q3_K, 2560, 512, 3},
                {"shexp gate", ref::Q2_0, 2560, 640, 21},             {"shexp down", ref::Q4_0, 640, 2560, 16},
                {"shexp down", ref::Q2_0, 640, 2560, 16},             {"indexer q", ref::BF16, 2560, 512, 12},
                {"ssm alpha", ref::BF16, 2560, 48, 36},               {"lm head", ref::Q5_K, 2560, 248320, 1},
                {"gdn qkv", ref::Q6_K, 2560, 10240, 1},
            };
        else
            shapes = {
                {"hc down (x4/layer)", ref::BF16, 10240, 320, 4 * 48}, {"hc up (x4/layer)", ref::BF16, 320, 10240, 4 * 48},
                {"gdn qkv", ref::IQ4_XS, 2560, 10240, 11},            {"gdn qkv", ref::Q6_K, 2560, 10240, 14},
                {"gdn qkv", ref::Q4_K, 2560, 10240, 8},               {"gdn out", ref::Q6_K, 6144, 2560, 19},
                {"gdn gate", ref::IQ3_S, 2560, 6144, 10},             {"gdn gate", ref::IQ4_XS, 2560, 6144, 11},
                {"qsa out", ref::Q6_K, 6144, 2560, 10},               {"qsa q", ref::IQ4_XS, 2560, 12288, 4},
                {"shexp down", ref::IQ4_NL, 640, 2560, 40},           {"shexp up", ref::IQ3_S, 2560, 640, 27},
                {"lm head", ref::Q5_K, 2560, 248320, 1},
            };
    }
    std::vector<int> ntoks = {ntok};
    std::printf("%-20s %-7s %6s %7s %5s %9s %9s %8s  %s\n", "shape", "type", "n_in", "n_out", "ntok", "us", "GB/s",
                "MB", "best tile (L,R,S,K)");
    for (const Shape& s : shapes) {
        const size_t rb = ref::row_bytes(s.type, s.n_in), mb = rb * s.n_out;
        const int copies = (int)std::max<size_t>(4, std::min<size_t>(256, (1ull << 30) / mb));
        mtl::Buffer W = cx.alloc(mb * copies), x = cx.alloc(4 * s.n_in * MAXT), y = cx.alloc(4 * (size_t)s.n_out * MAXT);
        {
            std::vector<uint8_t> tmp(mb);
            ref::random_blocks(s.type, tmp.data(), s.n_out, s.n_in, 99);
            for (int c = 0; c < copies; ++c) std::memcpy(W.as<uint8_t>() + (size_t)c * mb, tmp.data(), mb);
        }
        ref::Rng r(5);
        for (int i = 0; i < s.n_in * MAXT; ++i) x.as<float>()[i] = r.sym();
        mtl::Buffer xp = cx.alloc(4 * (4 * s.n_in * MAXT + s.n_in / 16 * MAXT));
        cx.run({xprep_dispatch(pprep, x, xp, s.n_in, MAXT)});
        const size_t plane = (size_t)4 * s.n_in * MAXT;
        for (int nt : ntoks) {
            double best = 1e9;
            Cfg bc{};
            const bool gx = std::getenv("GEMV_X") != nullptr;   // gemvx_* (x staged in threadgroup memory)
            const uint32_t tgm = (uint32_t)(4 * nt * (s.n_in + s.n_in / 16) + 15) / 16 * 16;
            for (Cfg c : gx ? std::vector<Cfg>{{32, 1, 4, 1}, {32, 2, 4, 1}, {16, 1, 4, 1}, {16, 2, 4, 1}, {8, 1, 4, 1},
                                               {8, 2, 4, 1}, {4, 2, 4, 1}, {16, 2, 8, 1}, {8, 2, 8, 1}, {16, 4, 4, 1},
                                               {32, 2, 8, 1}, {2, 2, 4, 1}}
                            : kCfgs) {
                if (!cfg_ok(c, s.type, s.n_in)) continue;
                if (!all && !(c.K == 1 || s.n_in >= 6144)) continue;
                if (gx && tgm > 32768) continue;
                const std::string kn = gx ? "gemvx_" + std::to_string(s.type) + "_" + std::to_string(c.L) + "_" + std::to_string(c.R) +
                                                 "_" + std::to_string(c.S)
                                          : kname(s.type, c);
                mtl::Pipeline p = cx.pipeline(lib, kn, mtl::Constants().i(0, nt).i(9, std::getenv("GEMV_PROBE") ? std::atoi(std::getenv("GEMV_PROBE")) : 0));
                GemvArgs a{(uint32_t)s.n_in, (uint32_t)s.n_out, (uint32_t)nt, (uint32_t)rb, (uint32_t)s.n_in,
                           (uint32_t)s.n_out};
                std::vector<mtl::Dispatch> ds;
                const int R = std::max(copies, 16);
                for (int k = 0; k < R; ++k) {
                    mtl::Dispatch d;
                    d.p = &p;
                    d.args = {mtl::Arg::B(W, (size_t)(k % copies) * mb), mtl::Arg::B(xp, plane * xkind(s.type)),
                              mtl::Arg::B(xp, plane * 4), mtl::Arg::B(y), mtl::Arg::V(a)};
                    d.grid[0] = (s.n_out + rows_per_tg(c) - 1) / rows_per_tg(c);
                    d.tg[0] = c.S * 32;
                    if (gx) d.tg_mem = tgm;
                    ds.push_back(d);
                }
                cx.warm(ds, 0.05);
                std::vector<double> t;
                for (int i = 0; i < 5; ++i) t.push_back(cx.run(ds) / R);
                const double m = median(t);
                if (one.size()) std::printf("   tile %2d,%d,%d,%d: %8.2f us  %6.1f GB/s\n", c.L, c.R, c.S, c.K, m * 1e6, mb / m / 1e9);
                if (m < best) { best = m; bc = c; }
            }
            std::printf("%-20s %-7s %6d %7d %5d %9.2f %9.1f %8.2f  %d,%d,%d,%d\n", s.name, ref::type_name(s.type), s.n_in,
                        s.n_out, nt, best * 1e6, mb / best / 1e9, mb / 1e6, bc.L, bc.R, bc.S, bc.K);
        }
        cx.free(W);
        cx.free(x);
        cx.free(y);
        cx.free(xp);
    }
    return 0;
}
