// metal/tests/gdn_parity.cpp - the fused GDN layer step (gdn.metal) against a double-precision CPU reference of the
// native contract, the window properties, and its speed.
//
//   gdn_parity            parity for windows of 1..4 tokens: y, the committed state and conv history; a verify
//                         window leaves the state untouched; its first m outputs are bitwise a fold of m
//   gdn_parity --bench    GPU time per layer: 36 distinct layer states rotated (113 MB, beyond the SLC)
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
constexpr int S = 128, HK = 16, HV = 48, C = (2 * HK + HV) * S, MAXT = 8;
struct GdnArgs { uint32_t h_k, h_v, channels, commit; float eps; uint32_t plane; };
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
    const int lib = cx.compile({"gdn.metal"}, "#define NO_GGML_COMMON 1\n");
    std::printf("device %s\n", cx.device_name().c_str());
    const int LAYERS = bench ? 36 : 1;
    const size_t st_elems = (size_t)S * HV * S;
    mtl::Buffer state = cx.alloc(4 * st_elems * LAYERS), st0 = cx.alloc(4 * st_elems), hist = cx.alloc(4 * C * 3 * LAYERS),
                h0 = cx.alloc(4 * C * 3), qkv = cx.alloc(4 * C * MAXT), al = cx.alloc(4 * HV * MAXT),
                be = cx.alloc(4 * HV * MAXT), z = cx.alloc(4 * HV * S * MAXT), cw = cx.alloc(4 * C * 4),
                dt = cx.alloc(4 * HV), sa = cx.alloc(4 * HV), gm = cx.alloc(4 * S), y = cx.alloc(4 * HV * S * MAXT),
                y2 = cx.alloc(4 * HV * S * MAXT), hist2 = cx.alloc(4 * C * 3 * LAYERS);
    ref::Rng r(17);
    for (size_t i = 0; i < st_elems; ++i) st0.as<float>()[i] = r.sym() * 0.1f;
    for (int i = 0; i < C * 3; ++i) h0.as<float>()[i] = r.sym();
    for (int i = 0; i < C * MAXT; ++i) qkv.as<float>()[i] = r.sym() * 2.0f;
    for (int i = 0; i < HV * MAXT; ++i) { al.as<float>()[i] = r.sym() * 2.0f; be.as<float>()[i] = r.sym() * 2.0f; }
    for (int i = 0; i < HV * S * MAXT; ++i) z.as<float>()[i] = r.sym() * 2.0f;
    for (int i = 0; i < C * 4; ++i) cw.as<float>()[i] = r.sym() * 0.5f;
    for (int i = 0; i < HV; ++i) { dt.as<float>()[i] = r.sym(); sa.as<float>()[i] = -std::exp(r.sym()); }
    for (int i = 0; i < S; ++i) gm.as<float>()[i] = 1.0f + 0.2f * r.sym();
    for (int l = 0; l < LAYERS; ++l) {
        std::memcpy(state.as<float>() + l * st_elems, st0.ptr, 4 * st_elems);
        std::memcpy(hist.as<float>() + l * C * 3, h0.ptr, 4 * C * 3);
    }
    auto disp = [&](const mtl::Pipeline& p, int layer, int commit, const mtl::Buffer& out) {
        GdnArgs a{HK, HV, C, (uint32_t)commit, 1e-6f, 0};
        mtl::Dispatch d;
        d.p = &p;
        d.args = {mtl::Arg::B(state, 4 * st_elems * layer), mtl::Arg::B(hist, 4 * (size_t)C * 3 * layer), mtl::Arg::B(qkv),
                  mtl::Arg::B(al), mtl::Arg::B(be), mtl::Arg::B(z), mtl::Arg::B(cw), mtl::Arg::B(dt), mtl::Arg::B(sa),
                  mtl::Arg::B(gm), mtl::Arg::B(out), mtl::Arg::V(a), mtl::Arg::B(hist2, 4 * (size_t)C * 3 * layer)};
        d.grid[0] = HV;
        d.tg[0] = std::getenv("GDN_IB8") ? 1024 : 512;
        return d;
    };
    {
        // every specialization the engine launches (GDN_NTOK 1..8): an M2 Max gave 448 for ntok >= 2 (#2323)
        const char* kn = std::getenv("GDN_IB8") ? "gdn_step8" : "gdn_step";
        const int need = std::getenv("GDN_IB8") ? 1024 : 512;
        int low = 0;
        for (int nt = 1; nt <= 8; ++nt) {
            mtl::Pipeline pn = cx.pipeline(lib, kn, mtl::Constants().i(2, nt));
            std::printf("%s pipeline GDN_NTOK %d: max %d threads per threadgroup (the kernel needs %d)%s\n", kn, nt, pn.max_threads,
                        need, pn.max_threads < need ? "  TOO FEW" : "");
            low += pn.max_threads < need;
        }
        if (low) {
            std::printf("gdn parity: %d specialization(s) cannot run with %d threads on this GPU: FAIL\n", low, need);
            return 1;
        }
    }
    if (bench) {
        for (int nt : {1, 2, 3, 4}) {
            const char* kn = std::getenv("GDN_IB8") ? "gdn_step8" : "gdn_step";
            mtl::Pipeline p = cx.pipeline(lib, kn, mtl::Constants().i(2, nt));
            for (int commit : {1, 0}) {
                std::vector<mtl::Dispatch> ds;
                for (int k = 0; k < 72; ++k) ds.push_back(disp(p, k % LAYERS, commit, y));
                cx.warm(ds, 0.1);
                std::vector<double> t;
                for (int i = 0; i < 7; ++i) t.push_back(cx.run(ds) / 72);
                const double m = median(t), bytes = 4.0 * st_elems * (commit ? 2 : 1);
                std::printf("gdn_step ntok %d %-7s %7.2f us per layer  state %.1f MB at %6.1f GB/s  (x36 layers: %.3f ms)\n",
                            nt, commit ? "commit" : "verify", m * 1e6, bytes / 1e6, bytes / m / 1e9, 36 * m * 1e3);
            }
        }
        return 0;
    }
    int fails = 0;
    for (int nt = 1; nt <= 4; ++nt) {
        mtl::Pipeline p = cx.pipeline(lib, std::getenv("GDN_IB8") ? "gdn_step8" : "gdn_step", mtl::Constants().i(2, nt));
        // determinism: the same verify call three times
        {
            std::memcpy(state.ptr, st0.ptr, 4 * st_elems);
            std::memcpy(hist.ptr, h0.ptr, 4 * C * 3);
            cx.run({disp(p, 0, 0, y2)});
            std::vector<float> first((float*)y2.ptr, (float*)y2.ptr + HV * S * nt);
            int diff = 0;
            for (int k = 0; k < 2; ++k) {
                cx.run({disp(p, 0, 0, y2)});
                diff += std::memcmp(first.data(), y2.ptr, 4 * HV * S * nt) != 0;
            }
            if (diff) std::printf("  ntok %d: NOT deterministic (%d of 2 repeats differ)\n", nt, diff);
        }
        // verify: state untouched
        std::memcpy(state.ptr, st0.ptr, 4 * st_elems);
        std::memcpy(hist.ptr, h0.ptr, 4 * C * 3);
        cx.run({disp(p, 0, 0, y2)});
        const bool untouched = std::memcmp(state.ptr, st0.ptr, 4 * st_elems) == 0 && std::memcmp(hist.ptr, h0.ptr, 4 * C * 3) == 0;
        cx.run({disp(p, 0, 1, y)});
        const bool same_y = std::memcmp(y.ptr, y2.ptr, 4 * HV * S * nt) == 0;
        // the reference
        std::vector<double> Sd(st_elems), Hd(C * 3);
        for (size_t i = 0; i < st_elems; ++i) Sd[i] = st0.as<float>()[i];
        for (int i = 0; i < C * 3; ++i) Hd[i] = h0.as<float>()[i];
        Err ey, es, eh;
        for (int t = 0; t < nt; ++t) {
            std::vector<double> cv(C);
            for (int c = 0; c < C; ++c) {
                const double in = qkv.as<float>()[t * C + c];
                double s = Hd[c * 3] * cw.as<float>()[c * 4] + Hd[c * 3 + 1] * cw.as<float>()[c * 4 + 1] +
                           Hd[c * 3 + 2] * cw.as<float>()[c * 4 + 2] + in * cw.as<float>()[c * 4 + 3];
                cv[c] = s / (1.0 + std::exp(-s));
                Hd[c * 3] = Hd[c * 3 + 1];
                Hd[c * 3 + 1] = Hd[c * 3 + 2];
                Hd[c * 3 + 2] = in;
            }
            for (int hh = 0; hh < 2 * HK; ++hh) {
                double ss = 0;
                for (int i = 0; i < S; ++i) ss += cv[hh * S + i] * cv[hh * S + i];
                const double rs = 1.0 / std::sqrt(ss / S + 1e-6 / S);
                for (int i = 0; i < S; ++i) cv[hh * S + i] = cv[hh * S + i] * rs / std::sqrt((double)S);
            }
            for (int h = 0; h < HV; ++h) {
                const int hk = h % HK;
                const double* q = &cv[hk * S];
                const double* k = &cv[HK * S + hk * S];
                const double* v = &cv[2 * HK * S + h * S];
                const double av = (double)al.as<float>()[t * HV + h] + dt.as<float>()[h];
                const double sp = av > 20 ? av : std::log1p(std::exp(av));
                const double g = std::exp(sp * sa.as<float>()[h]);
                const double b = 1.0 / (1.0 + std::exp(-(double)be.as<float>()[t * HV + h]));
                std::vector<double> o(S);
                for (int jj = 0; jj < S; ++jj) {
                    double kv = 0;
                    for (int i = 0; i < S; ++i) kv += Sd[((size_t)i * HV + h) * S + jj] * k[i];
                    const double delta = (v[jj] - g * kv) * b;
                    double acc = 0;
                    for (int i = 0; i < S; ++i) {
                        double& sv = Sd[((size_t)i * HV + h) * S + jj];
                        sv = g * sv + k[i] * delta;
                        acc += sv * q[i];
                    }
                    o[jj] = acc / std::sqrt((double)S);
                }
                double ss = 0;
                for (int jj = 0; jj < S; ++jj) ss += o[jj] * o[jj];
                const double rs = 1.0 / std::sqrt(ss / S + 1e-6);
                for (int jj = 0; jj < S; ++jj) {
                    const double zz = z.as<float>()[t * HV * S + h * S + jj];
                    ey.add(y.as<float>()[t * HV * S + h * S + jj], o[jj] * rs * gm.as<float>()[jj] / (1.0 + std::exp(-zz)));
                }
            }
        }
        for (size_t i = 0; i < st_elems; ++i) es.add(state.as<float>()[i], Sd[i]);
        for (int i = 0; i < C * 3; ++i) eh.add(hist2.as<float>()[i], Hd[i]);
        // fold property: a window of nt + 1 tokens (no commit) gives bitwise the same first nt outputs
        bool prefix = true;
        if (nt < 4) {
            mtl::Pipeline p2 = cx.pipeline(lib, std::getenv("GDN_IB8") ? "gdn_step8" : "gdn_step", mtl::Constants().i(2, nt + 1));
            std::memcpy(state.ptr, st0.ptr, 4 * st_elems);
            std::memcpy(hist.ptr, h0.ptr, 4 * C * 3);
            cx.run({disp(p2, 0, 0, y2)});
            prefix = std::memcmp(y.ptr, y2.ptr, 4 * HV * S * nt) == 0;
        }
        // partial commit (commit = 0x100 | k): the state and history after the first k tokens, bitwise those of a
        // k-token commit; y bitwise the verify window's
        bool partial = true;
        for (int k = 1; k < nt; ++k) {
            mtl::Pipeline pk = cx.pipeline(lib, std::getenv("GDN_IB8") ? "gdn_step8" : "gdn_step", mtl::Constants().i(2, k));
            mtl::Pipeline pn = cx.pipeline(lib, std::getenv("GDN_IB8") ? "gdn_step8" : "gdn_step", mtl::Constants().i(2, nt));
            std::memcpy(state.ptr, st0.ptr, 4 * st_elems);
            std::memcpy(hist.ptr, h0.ptr, 4 * C * 3);
            cx.run({disp(pk, 0, 1, y2)});
            std::vector<float> sk(state.as<float>(), state.as<float>() + st_elems), hk(hist2.as<float>(), hist2.as<float>() + C * 3);
            std::memcpy(state.ptr, st0.ptr, 4 * st_elems);
            cx.run({disp(pn, 0, 0, y)});
            std::vector<float> yv(y.as<float>(), y.as<float>() + HV * S * nt);
            std::memcpy(state.ptr, st0.ptr, 4 * st_elems);
            std::memset(hist2.ptr, 0xFF, 4 * C * 3);
            cx.run({disp(pn, 0, 0x100 | k, y2)});
            partial = partial && std::memcmp(state.ptr, sk.data(), 4 * st_elems) == 0 &&
                      std::memcmp(hist2.ptr, hk.data(), 4 * C * 3) == 0 && std::memcmp(y2.ptr, yv.data(), 4 * HV * S * nt) == 0;
        }
        const bool ok = untouched && same_y && prefix && partial && ey.l2() < 1e-5 && es.l2() < 1e-5 && eh.l2() == 0;
        fails += !ok;
        std::printf("gdn parity ntok %d: y rel L2 %.2e, state %.2e, history %.1e; verify leaves state: %s, verify y == "
                    "commit y: %s, first %d of a longer window bitwise: %s, partial commits bitwise: %s  %s\n", nt, ey.l2(), es.l2(),
                    eh.l2(), untouched ? "yes" : "NO", same_y ? "yes" : "NO", nt, prefix ? "yes" : "NO", partial ? "yes" : "NO",
                    ok ? "PASS" : "FAIL");
    }
    std::printf("%s\n", fails ? "SOME FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
