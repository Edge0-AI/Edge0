// metal/tests/qsa_parity.cpp - QSA decode (qsa.metal) stage by stage against a CPU reference, and its speed.
//
//   qsa_parity [P]          a context of P cells built by qsa_prep (default 3000: the selection is active above
//                           2051), then a window of 3 tokens: prep, score, top-k, attention, merge.  Each stage is
//                           checked against the CPU fed with the GPU's inputs to that stage; the ids exactly.
//   qsa_parity --bench [P]  GPU time of the window's kernels at context P, 1 and 3 tokens
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
constexpr int QH = 24, QKV = 2, QD = 256, IH = 4, ID = 128, BUDGET = 2051, CH = 64, NCH = (BUDGET + CH - 1) / CH;
struct QsaArgs { uint32_t pos0, pos_base, max_cells, n_tok; float eps; };
double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
struct Err {
    double se = 0, sr = 0;
    void add(double g, double r) { se += (g - r) * (g - r); sr += r * r; }
    double l2() const { return sr > 0 ? std::sqrt(se / sr) : std::sqrt(se); }
};
float h2f(uint16_t h) { return ref::f16_to_f32(h); }
}  // namespace

int main(int argc, char** argv) {
    bool bench = false;
    int P = 3000;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bench") bench = true;
        else P = std::atoi(argv[i]);
    }
    const int MAXT = 8, MAXC = std::max(4096, (P + 64) / 4 * 4 + 64);
    mtl::Context cx;
    const int lib = cx.compile({"qsa.metal", "qsa2.metal"}, "#define NO_GGML_COMMON 1\n");
    // QSA_TOPK=1: the v1 top-k; QSA_ATTN=1|2|3 (default 3): which attention (2 has its own chunking and merge)
    const int av = std::getenv("QSA_ATTN") ? std::atoi(std::getenv("QSA_ATTN")) : 1;
    const bool v2 = av == 2;
    mtl::Pipeline pprep = cx.pipeline(lib, "qsa_prep", mtl::Constants().b(24, std::getenv("QSA_MROPE") != nullptr)), pscore = cx.pipeline(lib, "qsa_score"),
                  ptopk = cx.pipeline(lib, std::getenv("QSA_TOPK") ? "qsa_topk" : "qsa_topk2"),
                  pattn = cx.pipeline(lib, av == 1 ? "qsa_attn" : av == 2 ? "qsa_attn2" : "qsa_attn3"),
                  pmerge = cx.pipeline(lib, v2 ? "qsa_merge2" : "qsa_merge");
    const int MB = MAXC / 4 + 2;
    // per-cell inputs for every cell (context + window)
    const int NC = P + MAXT;
    mtl::Buffer qfull = cx.alloc(4 * (size_t)NC * QH * 2 * QD), kc = cx.alloc(4 * (size_t)NC * QKV * QD),
                vc = cx.alloc(4 * (size_t)NC * QKV * QD), iq = cx.alloc(4 * (size_t)NC * IH * ID),
                ik = cx.alloc(4 * (size_t)NC * ID), qn = cx.alloc(4 * QD), kn = cx.alloc(4 * QD), iqn = cx.alloc(4 * ID),
                ikn = cx.alloc(4 * ID), cs = cx.alloc(8 * (size_t)(MAXC + 16) * 32), qcur = cx.alloc(4 * MAXT * QH * QD),
                qidx = cx.alloc(4 * MAXT * IH * ID), kq = cx.alloc((size_t)MAXC * QKV * QD), vq = cx.alloc((size_t)MAXC * QKV * QD),
                ksc = cx.alloc(2 * (size_t)MAXC * QKV * 4), vsc = cx.alloc(2 * (size_t)MAXC * QKV * 4), tail = cx.alloc(4 * 4 * ID),
                dead = cx.alloc(4 * ID), pooled = cx.alloc(4 * (size_t)MB * ID), scores = cx.alloc(4 * (size_t)MAXT * MB),
                ids = cx.alloc(4 * MAXT * BUDGET), part = cx.alloc(4 * (size_t)MAXT * 2 * NCH * QH * (QD + 2)),
                xp = cx.alloc(4 * (4 * MAXT * QH * QD + MAXT * QH * QD / 16));
    ref::Rng r(23);
    for (size_t i = 0; i < qfull.bytes / 4; ++i) qfull.as<float>()[i] = r.sym() * 2.0f;
    for (size_t i = 0; i < kc.bytes / 4; ++i) kc.as<float>()[i] = r.sym() * 2.0f;
    for (size_t i = 0; i < vc.bytes / 4; ++i) vc.as<float>()[i] = r.sym();
    for (size_t i = 0; i < iq.bytes / 4; ++i) iq.as<float>()[i] = r.sym();
    for (size_t i = 0; i < ik.bytes / 4; ++i) ik.as<float>()[i] = r.sym();
    for (int i = 0; i < QD; ++i) { qn.as<float>()[i] = 1.0f + 0.2f * r.sym(); kn.as<float>()[i] = 1.0f + 0.2f * r.sym(); }
    for (int i = 0; i < ID; ++i) { iqn.as<float>()[i] = 1.0f + 0.2f * r.sym(); ikn.as<float>()[i] = 1.0f + 0.2f * r.sym(); }
    const uint32_t pos_base = 0;
    // QSA_MROPE=1: qsa_prep with the M-RoPE table (function constant 24): random (t, h, w) positions per cell, the CPU
    // reference rotating pair i by row tab[cell * 3 + i % 3]; QSA_MROPE=id: the identity table, checked bitwise
    // against the plain kernel
    const char* mre = std::getenv("QSA_MROPE");
    const bool mrope = mre != nullptr, mid = mrope && std::string(mre) == "id";
    mtl::Buffer mtab = cx.alloc(4 * 3 * (size_t)(MAXC + 16));
    {
        ref::Rng rm(77);
        for (int c = 0; c < MAXC + 16; ++c)
            for (int k = 0; k < 3; ++k) mtab.as<int>()[c * 3 + k] = mid ? c : (int)(rm.next() % (uint64_t)(MAXC + 16));
    }
    for (int p = 0; p < MAXC + 16; ++p)
        for (int i = 0; i < 32; ++i) {
            const double inv = std::pow(1e7, -2.0 * i / 64.0), th = (double)p * inv;
            cs.as<float>()[(p * 32 + i) * 2] = (float)std::cos(th);
            cs.as<float>()[(p * 32 + i) * 2 + 1] = (float)std::sin(th);
        }
    auto prep = [&](uint32_t pos0, int nt) {
        QsaArgs a{pos0, pos_base, (uint32_t)MAXC, (uint32_t)nt, 1e-6f};
        mtl::Dispatch d;
        d.p = &pprep;
        d.args = {mtl::Arg::B(qfull, 4 * (size_t)pos0 * QH * 2 * QD), mtl::Arg::B(kc, 4 * (size_t)pos0 * QKV * QD),
                  mtl::Arg::B(vc, 4 * (size_t)pos0 * QKV * QD), mtl::Arg::B(iq, 4 * (size_t)pos0 * IH * ID),
                  mtl::Arg::B(ik, 4 * (size_t)pos0 * ID), mtl::Arg::B(qn), mtl::Arg::B(kn), mtl::Arg::B(iqn),
                  mtl::Arg::B(ikn), mtl::Arg::B(cs), mtl::Arg::B(qcur), mtl::Arg::B(qidx), mtl::Arg::B(kq),
                  mtl::Arg::B(vq), mtl::Arg::B(ksc), mtl::Arg::B(vsc), mtl::Arg::B(tail), mtl::Arg::B(dead),
                  mtl::Arg::B(pooled), mtl::Arg::V(a), mtl::Arg::B(mtab)};
        d.grid[0] = 33;
        d.grid[1] = nt;
        d.tg[0] = 256;
        return d;
    };
    auto window = [&](uint32_t pos0, int nt, std::vector<mtl::Dispatch>& ds) {
        QsaArgs a{pos0, pos_base, (uint32_t)MAXC, (uint32_t)nt, 1e-6f};
        ds.push_back(prep(pos0, nt));
        mtl::Dispatch s;
        s.p = &pscore;
        s.args = {mtl::Arg::B(qidx), mtl::Arg::B(pooled), mtl::Arg::B(dead), mtl::Arg::B(scores), mtl::Arg::V(a)};
        s.grid[0] = ((pos0 + nt) / 4 + 1 + 7) / 8;
        s.grid[1] = nt;
        s.tg[0] = 256;
        ds.push_back(s);
        mtl::Dispatch k;
        k.p = &ptopk;
        k.args = {mtl::Arg::B(scores), mtl::Arg::B(ids), mtl::Arg::V(a)};
        k.grid[0] = nt;
        k.tg[0] = 1024;
        ds.push_back(k);
        mtl::Dispatch at;
        at.p = &pattn;
        at.args = {mtl::Arg::B(qcur), mtl::Arg::B(kq), mtl::Arg::B(vq), mtl::Arg::B(ksc), mtl::Arg::B(vsc),
                   mtl::Arg::B(ids), mtl::Arg::B(part), mtl::Arg::V(a)};
        at.grid[0] = v2 ? (BUDGET + 31) / 32 : NCH;
        at.grid[1] = v2 ? QKV * 2 : QKV;
        at.grid[2] = nt;
        at.tg[0] = 256;
        ds.push_back(at);
        mtl::Dispatch m;
        m.p = &pmerge;
        m.args = {mtl::Arg::B(part), mtl::Arg::B(qfull, 4 * (size_t)pos0 * QH * 2 * QD), mtl::Arg::B(xp), mtl::Arg::V(a)};
        m.grid[0] = QH;
        m.grid[1] = nt;
        m.tg[0] = 256;
        ds.push_back(m);
    };
    // the context, one cell at a time
    {
        std::vector<mtl::Dispatch> ds;
        for (int p = 0; p < P; ++p) ds.push_back(prep(p, 1));
        cx.run(ds);
    }
    if (bench) {
        std::printf("device %s, context %d cells\n", cx.device_name().c_str(), P);
        for (int nt : {1, 3}) {
            std::vector<mtl::Dispatch> ds;
            window(P, nt, ds);
            const char* names[] = {"prep", "score", "topk", "attn", "merge"};
            double total = 0;
            for (int i = 0; i < 5; ++i) {
                std::vector<mtl::Dispatch> one(32, ds[i]);
                cx.warm(one, 0.05);
                std::vector<double> t;
                for (int k = 0; k < 7; ++k) t.push_back(cx.run(one) / 32);
                std::printf("  ntok %d %-6s %7.2f us\n", nt, names[i], median(t) * 1e6);
                total += median(t);
            }
            std::vector<double> t;
            for (int k = 0; k < 7; ++k) t.push_back(cx.run(ds));
            std::printf("  ntok %d window (5 dependent dispatches): %.2f us (sum of parts %.2f)\n", nt, median(t) * 1e6,
                        total * 1e6);
        }
        return 0;
    }
    const int NT = 3;
    std::vector<mtl::Dispatch> ds;
    window(P, NT, ds);
    // stage by stage, re-reading GPU state in between
    cx.run({ds[0]});
    int fails = 0;
    // ---- prep: qcur, qidx, the window cells' KV codes and the pooled keys vs CPU
    {
        Err eq, ei, ekv, ep;
        auto normrope = [&](const float* x, const float* g, int w, int absp, bool rope, std::vector<double>& out) {
            double ss = 0;
            for (int i = 0; i < w; ++i) ss += (double)x[i] * x[i];
            const double rs = 1.0 / std::sqrt(ss / w + 1e-6);
            out.assign(w, 0.0);
            for (int i = 0; i < w; ++i) out[i] = rs * x[i] * g[i];
            if (rope)
                for (int i = 0; i < 32; ++i) {
                    const int pp = mrope ? mtab.as<int>()[absp * 3 + i % 3] : absp;
                    const double c = cs.as<float>()[(pp * 32 + i) * 2], s = cs.as<float>()[(pp * 32 + i) * 2 + 1];
                    const double a = out[i], b = out[i + 32];
                    out[i] = a * c - b * s;
                    out[i + 32] = a * s + b * c;
                }
        };
        std::vector<double> o;
        for (int t = 0; t < NT; ++t) {
            const int cell = P + t;
            for (int h = 0; h < QH; ++h) {
                normrope(qfull.as<float>() + ((size_t)cell * QH + h) * 2 * QD, qn.as<float>(), QD, cell, true, o);
                for (int i = 0; i < QD; ++i) eq.add(qcur.as<float>()[(t * QH + h) * QD + i], o[i]);
            }
            for (int h = 0; h < IH; ++h) {
                normrope(iq.as<float>() + ((size_t)cell * IH + h) * ID, iqn.as<float>(), ID, cell, true, o);
                for (int i = 0; i < ID; ++i) ei.add(qidx.as<float>()[(t * IH + h) * ID + i], o[i]);
            }
            for (int h = 0; h < QKV; ++h) {
                normrope(kc.as<float>() + ((size_t)cell * QKV + h) * QD, kn.as<float>(), QD, cell, true, o);
                for (int i = 0; i < QD; ++i) {
                    const size_t row = (size_t)cell * QKV + h;
                    const double deq = (double)(int8_t)kq.as<int8_t>()[row * QD + i] * h2f(ksc.as<uint16_t>()[row * 4 + i / 64]);
                    ekv.add(deq, o[i]);
                }
            }
        }
        // pooled keys of the last few completed blocks
        const int nb = (P + NT) / 4;
        for (int b = std::max(0, nb - 8); b < nb; ++b) {
            std::vector<float> m(ID);
            for (int i = 0; i < ID; ++i)
                m[i] = (((ik.as<float>()[(4 * b) * ID + i] + ik.as<float>()[(4 * b + 1) * ID + i]) +
                         ik.as<float>()[(4 * b + 2) * ID + i]) + ik.as<float>()[(4 * b + 3) * ID + i]) * 0.25f;
            normrope(m.data(), ikn.as<float>(), ID, 4 * b, true, o);
            for (int i = 0; i < ID; ++i) ep.add(pooled.as<float>()[b * ID + i], o[i]);
        }
        // int8 quantization error is ~1/254 of max|x| per group: the KV check is a quantization tolerance
        const bool ok = eq.l2() < 1e-5 && ei.l2() < 1e-5 && ep.l2() < 1e-5 && ekv.l2() < 6e-3;
        fails += !ok;
        std::printf("prep: q rel L2 %.2e, indexer q %.2e, pooled keys %.2e, K int8 (vs float) %.2e  %s\n", eq.l2(),
                    ei.l2(), ep.l2(), ekv.l2(), ok ? "PASS" : "FAIL");
    }
    // ---- score
    cx.run({ds[1]});
    {
        Err es;
        for (int t = 0; t < NT; ++t) {
            const int n_kv = P + t + 1, n_bid = n_kv / 4;
            for (int b = 0; b <= n_bid; ++b) {
                const float* key = b == n_bid ? dead.as<float>() : pooled.as<float>() + b * ID;
                double s = 0;
                for (int h = 0; h < IH; ++h) {
                    double d = 0;
                    for (int i = 0; i < ID; ++i) d += (double)qidx.as<float>()[(t * IH + h) * ID + i] * key[i];
                    s += std::max(d, 0.0);
                }
                if (b == n_bid && (n_kv & 3)) {
                    // float(s + 1e9) keeps no bits of s: only check that the bias is there
                    if (!(scores.as<float>()[t * MB + b] >= 1e9f)) es.add(1.0, 0.0);
                } else {
                    es.add(scores.as<float>()[t * MB + b], s);
                }
            }
        }
        const bool ok = es.l2() < 1e-5;
        fails += !ok;
        std::printf("score: rel L2 %.2e (tail +1e9 checked separately)  %s\n", es.l2(), ok ? "PASS" : "FAIL");
    }
    // ---- top-k (exact, from the GPU's scores)
    cx.run({ds[2]});
    {
        int bad = 0;
        for (int t = 0; t < NT; ++t) {
            const int n_kv = P + t + 1, n_bid = n_kv / 4, width = std::min(n_kv, BUDGET);
            std::vector<int> cells(n_kv);
            for (int c = 0; c < n_kv; ++c) cells[c] = c;
            if (n_kv > BUDGET) {
                const float* sc = scores.as<float>() + t * MB;
                std::stable_sort(cells.begin(), cells.end(), [&](int x, int y) { return sc[x / 4] > sc[y / 4]; });
                cells.resize(width);
                std::sort(cells.begin(), cells.end());
            }
            (void)n_bid;
            for (int i = 0; i < width; ++i) bad += ids.as<int>()[t * BUDGET + i] != cells[i];
        }
        fails += bad != 0;
        std::printf("top-k: %d of %d ids differ (context %d: selection %s)  %s\n", bad, NT * std::min(P + 1, BUDGET), P,
                    P + 1 > BUDGET ? "active" : "identity", bad ? "FAIL" : "PASS");
    }
    // ---- attention + merge (from the GPU's qcur, ids and KV codes)
    cx.run({ds[3], ds[4]});
    {
        Err eo;
        for (int t = 0; t < NT; ++t) {
            const int n_kv = P + t + 1, width = std::min(n_kv, BUDGET);
            for (int h = 0; h < QH; ++h) {
                const int g = h / (QH / QKV);
                std::vector<double> s(width);
                double mx = -1e300;
                for (int c = 0; c < width; ++c) {
                    const size_t row = (size_t)ids.as<int>()[t * BUDGET + c] * QKV + g;
                    double d = 0;
                    for (int i = 0; i < QD; ++i)
                        d += (double)qcur.as<float>()[(t * QH + h) * QD + i] * (int8_t)kq.as<int8_t>()[row * QD + i] *
                             h2f(ksc.as<uint16_t>()[row * 4 + i / 64]);
                    s[c] = d / 16.0;
                    mx = std::max(mx, s[c]);
                }
                double l = 0;
                for (int c = 0; c < width; ++c) l += (s[c] = std::exp(s[c] - mx));
                for (int i = 0; i < QD; ++i) {
                    double o = 0;
                    for (int c = 0; c < width; ++c) {
                        const size_t row = (size_t)ids.as<int>()[t * BUDGET + c] * QKV + g;
                        o += s[c] * (int8_t)vq.as<int8_t>()[row * QD + i] * h2f(vsc.as<uint16_t>()[row * 4 + i / 64]);
                    }
                    o /= l;
                    const double gate = qfull.as<float>()[((size_t)(P + t) * QH + h) * 2 * QD + QD + i];
                    eo.add(xp.as<float>()[t * QH * QD + h * QD + i], o / (1.0 + std::exp(-gate)));
                }
            }
        }
        const bool ok = eo.l2() < 1e-5;
        fails += !ok;
        std::printf("attention + gate: rel L2 %.2e  %s\n", eo.l2(), ok ? "PASS" : "FAIL");
    }
    std::printf("%s\n", fails ? "SOME FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
