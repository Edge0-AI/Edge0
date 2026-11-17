// metal/tests/prompt_parity.cpp - the prompt-chunk glue (prompt.metal) against CPU references:
//   route_topk           softmax + top 10 + renormalised weights (moe_route's rule) for 300 tokens, a third of them
//                        with exact ties
//   the HC read as GEMMs gr_norm -> act_f16 -> [w_down; w_inject] GEMM -> gr_epi_down -> w_up GEMM -> gr_epi_up, on
//                        random BF16 weights, vs the exact formula (fp64) - the fp16 operands set the tolerance
//   swiglu_h, moe_out    elementwise
#include "mtl_rt.h"
#include "ref_quant.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
constexpr int N = 2560, HC = 4, LR = 320;
struct GrArgs { uint32_t n_embd, hc, hc_lr, n_tok, apply; float eps; };
struct EpiArgs { uint32_t n_tok, n_embd, hc, hc_lr; };
struct DqArgs { uint32_t n_in, n_out, row_bytes, perm; };
struct GemmArgs { int M, N, K, pad; };
double sig(double v) { return 1.0 / (1.0 + std::exp(-v)); }
}  // namespace

int main() {
    mtl::Context cx;
    const int lib = cx.compile({"quant.metal", "units.metal", "mma.metal", "dense_mm.metal", "gr.metal", "prompt.metal"});
    std::printf("device %s\n", cx.device_name().c_str());
    int fails = 0;
    // ---- route_topk
    {
        const int T = 300;
        mtl::Pipeline p = cx.pipeline(lib, "route_topk");
        mtl::Buffer lg = cx.alloc(4 * T * 512), ids = cx.alloc(4 * T * 10), w = cx.alloc(4 * T * 10);
        ref::Rng r(9);
        for (int t = 0; t < T; ++t)
            for (int e = 0; e < 512; ++e) {
                float v = r.sym() * 4.0f;
                if (t % 3 == 0) v = std::round(v * 4.0f) / 4.0f;
                lg.as<float>()[t * 512 + e] = v;
            }
        const uint32_t nt = T;
        mtl::Dispatch d;
        d.p = &p;
        d.args = {mtl::Arg::B(lg), mtl::Arg::B(ids), mtl::Arg::B(w), mtl::Arg::V(nt)};
        d.grid[0] = (T + 7) / 8;
        d.tg[0] = 256;
        cx.run({d});
        int bad = 0;
        double worst = 0;
        for (int t = 0; t < T; ++t) {
            const float* l = lg.as<float>() + t * 512;
            float mx = -INFINITY;
            for (int e = 0; e < 512; ++e) mx = std::max(mx, l[e]);
            std::vector<double> pr(512);
            double s = 0;
            for (int e = 0; e < 512; ++e) s += (pr[e] = std::exp((double)l[e] - mx));
            std::vector<int> order(512);
            for (int e = 0; e < 512; ++e) order[e] = e;
            std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return l[x] > l[y]; });
            double ss = 0;
            for (int k = 0; k < 10; ++k) ss += pr[order[k]] / s;
            for (int k = 0; k < 10; ++k) {
                if (ids.as<int>()[t * 10 + k] != order[k]) { ++bad; break; }
                const double ref = pr[order[k]] / s / std::max(ss, 6.103515625e-5);
                worst = std::max(worst, std::fabs(w.as<float>()[t * 10 + k] - ref) / ref);
            }
        }
        const bool ok = bad == 0 && worst < 2e-6;
        fails += !ok;
        std::printf("route_topk, %d tokens: id mismatches %d, worst weight rel err %.2e  %s\n", T, bad, worst, ok ? "PASS" : "FAIL");
    }
    // ---- the HC read as GEMMs
    for (int apply = 0; apply < 2; ++apply) {
        const int T = 70, NIN = HC * N, NR = LR + HC;
        mtl::Buffer R = cx.alloc(4 * (size_t)T * NIN), bo = cx.alloc(4 * (size_t)T * N), inj = cx.alloc(4 * T * HC),
                    wn = cx.alloc(4 * NIN), wdn = cx.alloc(2 * (size_t)LR * NIN), win = cx.alloc(2 * (size_t)HC * NIN),
                    wup = cx.alloc(2 * (size_t)NIN * LR), xn = cx.alloc(4 * (size_t)T * NIN), xnh = cx.alloc(2 * (size_t)T * NIN),
                    WD = cx.alloc(2 * (size_t)NR * NIN), WU = cx.alloc(2 * (size_t)NIN * LR), V = cx.alloc(4 * (size_t)T * NR),
                    loh = cx.alloc(2 * (size_t)T * LR), injo = cx.alloc(4 * T * HC), G = cx.alloc(4 * (size_t)T * NIN),
                    mixed = cx.alloc(4 * (size_t)T * N), mixh = cx.alloc(2 * (size_t)T * N);
        ref::Rng r(11 + apply);
        std::vector<float> R0((size_t)T * NIN);
        for (auto& v : R0) v = r.sym();
        std::copy(R0.begin(), R0.end(), R.as<float>());
        for (size_t i = 0; i < (size_t)T * N; ++i) bo.as<float>()[i] = r.sym();
        for (int i = 0; i < T * HC; ++i) inj.as<float>()[i] = r.sym() * 3.0f;
        for (int i = 0; i < NIN; ++i) wn.as<float>()[i] = 0.5f + r.uni();
        for (size_t i = 0; i < (size_t)LR * NIN; ++i) wdn.as<uint16_t>()[i] = ref::f32_to_bf16(r.sym() * 0.05f);
        for (size_t i = 0; i < (size_t)HC * NIN; ++i) win.as<uint16_t>()[i] = ref::f32_to_bf16(r.sym() * 0.05f);
        for (size_t i = 0; i < (size_t)NIN * LR; ++i) wup.as<uint16_t>()[i] = ref::f32_to_bf16(r.sym() * 0.2f);
        mtl::Pipeline pn = cx.pipeline(lib, "gr_norm"), pa = cx.pipeline(lib, "act_f16"), pdq = cx.pipeline(lib, "dq_f16_30"),
                      pg = cx.pipeline(lib, "gemm_f16_64_64_4"), ped = cx.pipeline(lib, "gr_epi_down"), peu = cx.pipeline(lib, "gr_epi_up");
        std::vector<mtl::Dispatch> ds;
        GrArgs ga{N, HC, LR, (uint32_t)T, (uint32_t)apply, 1e-6f};
        EpiArgs ea{(uint32_t)T, N, HC, LR};
        auto D = [&](const mtl::Pipeline& p, std::vector<mtl::Arg> a, uint32_t gx, uint32_t gy, uint32_t tg) {
            mtl::Dispatch d; d.p = &p; d.args = std::move(a); d.grid[0] = gx; d.grid[1] = gy; d.tg[0] = tg; ds.push_back(d);
        };
        auto dq = [&](const mtl::Buffer& w, const mtl::Buffer& out, size_t ooff, int n_in, int n_out) {
            DqArgs a{(uint32_t)n_in, (uint32_t)n_out, (uint32_t)(2 * n_in)};
            D(pdq, {mtl::Arg::B(w), mtl::Arg::B(out, ooff), mtl::Arg::V(a)}, (n_in / 16 + 63) / 64, n_out, 64);
        };
        auto gemm = [&](const mtl::Buffer& A, const mtl::Buffer& B, const mtl::Buffer& Y, int M, int Nn, int K) {
            GemmArgs g{M, Nn, K, 0};
            D(pg, {mtl::Arg::B(A), mtl::Arg::B(B), mtl::Arg::B(Y), mtl::Arg::V(g)}, (Nn + 63) / 64, (M + 63) / 64, 128);
        };
        const uint32_t nx = T * NIN;
        dq(wdn, WD, 0, NIN, LR);
        dq(win, WD, (size_t)2 * LR * NIN, NIN, HC);
        dq(wup, WU, 0, LR, NIN);
        D(pn, {mtl::Arg::B(R), mtl::Arg::B(bo), mtl::Arg::B(inj), mtl::Arg::B(wn), mtl::Arg::B(xn), mtl::Arg::V(ga)}, HC, T, 256);
        D(pa, {mtl::Arg::B(xn), mtl::Arg::B(xnh), mtl::Arg::V(nx)}, (nx + 255) / 256, 1, 256);
        gemm(xnh, WD, V, T, NR, NIN);
        D(ped, {mtl::Arg::B(V), mtl::Arg::B(loh), mtl::Arg::B(injo), mtl::Arg::V(ea)}, (T * NR + 255) / 256, 1, 256);
        gemm(loh, WU, G, T, NIN, LR);
        D(peu, {mtl::Arg::B(G), mtl::Arg::B(xn), mtl::Arg::B(mixed), mtl::Arg::B(mixh), mtl::Arg::V(ea)}, (T * N + 255) / 256, 1, 256);
        cx.run(ds);
        // the fused read: gr_norm_h (fp16 xn), the down GEMM, gr_epi_down, the up GEMM with the epilogue (w_up rows
        // interleaved by dq_f16's perm)
        mtl::Buffer mixed2 = cx.alloc(4 * (size_t)T * N), mixh2 = cx.alloc(2 * (size_t)T * N), WU2 = cx.alloc(2 * (size_t)NIN * LR),
                    xnh2 = cx.alloc(2 * (size_t)T * NIN), V2 = cx.alloc(4 * (size_t)T * NR), loh2 = cx.alloc(2 * (size_t)T * LR),
                    injo2 = cx.alloc(4 * T * HC);
        std::copy(R0.begin(), R0.end(), R.as<float>());
        {
            mtl::Pipeline pnh = cx.pipeline(lib, "gr_norm_h"), pup = cx.pipeline(lib, "gemm_hc_up_64_64_4");
            ds.clear();
            DqArgs da{(uint32_t)LR, (uint32_t)NIN, (uint32_t)(2 * LR), (uint32_t)HC};
            D(pdq, {mtl::Arg::B(wup), mtl::Arg::B(WU2), mtl::Arg::V(da)}, (LR / 16 + 63) / 64, NIN, 64);
            D(pnh, {mtl::Arg::B(R), mtl::Arg::B(bo), mtl::Arg::B(inj), mtl::Arg::B(wn), mtl::Arg::B(xnh2), mtl::Arg::V(ga)}, HC, T, 256);
            gemm(xnh2, WD, V2, T, NR, NIN);
            D(ped, {mtl::Arg::B(V2), mtl::Arg::B(loh2), mtl::Arg::B(injo2), mtl::Arg::V(ea)}, (T * NR + 255) / 256, 1, 256);
            GemmArgs g{T, NIN, LR, 0};
            D(pup, {mtl::Arg::B(loh2), mtl::Arg::B(WU2), mtl::Arg::B(xnh2), mtl::Arg::B(mixed2), mtl::Arg::B(mixh2), mtl::Arg::V(g),
                    mtl::Arg::V(ea)}, NIN / 64, (T + 63) / 64, 128);
            cx.run(ds);
        }
        // reference
        double se = 0, sr = 0, sei = 0, sri = 0, ser = 0, srr = 0, se2 = 0, sh2 = 0;
        std::vector<double> x(NIN), lo(LR);
        for (int t = 0; t < T; ++t) {
            for (int c = 0; c < HC; ++c) {
                const double g = apply ? 2.0 * sig(inj.as<float>()[t * HC + c] / (double)HC) : 0.0;
                double ss = 0;
                for (int d = 0; d < N; ++d) {
                    const size_t i = (size_t)t * NIN + c * N + d;
                    const double v = R0[i] + (apply ? bo.as<float>()[(size_t)t * N + d] * g : 0.0);
                    x[c * N + d] = v;
                    ss += v * v;
                    ser += (R.as<float>()[i] - v) * (R.as<float>()[i] - v);
                    srr += v * v;
                }
                const double rs = 1.0 / std::sqrt(ss / N + 1e-6);
                for (int d = 0; d < N; ++d) x[c * N + d] *= wn.as<float>()[c * N + d] * rs;
            }
            for (int k = 0; k < NR; ++k) {
                const uint16_t* w = k < LR ? wdn.as<uint16_t>() + (size_t)k * NIN : win.as<uint16_t>() + (size_t)(k - LR) * NIN;
                double s = 0;
                for (int i = 0; i < NIN; ++i) s += ref::bf16_to_f32(w[i]) * x[i];
                if (k < LR) lo[k] = s / HC * sig(s / HC);
                else {
                    const double gv = injo.as<float>()[t * HC + k - LR];
                    sei += (gv - s) * (gv - s);
                    sri += s * s;
                }
            }
            for (int d = 0; d < N; ++d) {
                double m = 0;
                for (int c = 0; c < HC; ++c) {
                    const uint16_t* w = wup.as<uint16_t>() + (size_t)(c * N + d) * LR;
                    double s = 0;
                    for (int k = 0; k < LR; ++k) s += ref::bf16_to_f32(w[k]) * lo[k];
                    m += x[c * N + d] * sig(s);
                }
                m /= HC;
                const double gv = mixed.as<float>()[(size_t)t * N + d], g2 = mixed2.as<float>()[(size_t)t * N + d],
                             h2 = ref::f16_to_f32(mixh2.as<uint16_t>()[(size_t)t * N + d]);
                se += (gv - m) * (gv - m);
                se2 += (g2 - m) * (g2 - m);
                sh2 += (h2 - m) * (h2 - m);
                sr += m * m;
            }
        }
        const double rm = std::sqrt(se / sr), ri = std::sqrt(sei / sri), rr = std::sqrt(ser / srr);
        const double r2 = std::sqrt(se2 / sr), rh2 = std::sqrt(sh2 / sr);
        const bool ok = rm < 2e-3 && ri < 2e-3 && rr < 1e-6 && r2 < 2e-3 && rh2 < 2e-3;
        fails += !ok;
        std::printf("HC read as GEMMs, %d tokens, apply %d: mixed rel err %.2e, inject %.2e, residual %.2e; fused read "
                    "(fp16 xn, up GEMM epilogue): mixed %.2e, fp16 %.2e  %s\n", T, apply, rm, ri, rr, r2, rh2, ok ? "PASS" : "FAIL");
    }
    // ---- swiglu_h, moe_out
    {
        const int n = 5000, T = 7, NE = 640;
        mtl::Pipeline ps = cx.pipeline(lib, "swiglu_h"), pm = cx.pipeline(lib, "moe_out");
        mtl::Buffer g = cx.alloc(4 * n), u = cx.alloc(4 * n), h = cx.alloc(2 * n), yr = cx.alloc(4 * T * NE),
                    ys = cx.alloc(4 * T * NE), sg = cx.alloc(4 * T), bo = cx.alloc(4 * T * NE);
        ref::Rng r(5);
        for (int i = 0; i < n; ++i) { g.as<float>()[i] = r.sym() * 6; u.as<float>()[i] = r.sym(); }
        for (int i = 0; i < T * NE; ++i) { yr.as<float>()[i] = r.sym(); ys.as<float>()[i] = r.sym(); }
        for (int i = 0; i < T; ++i) sg.as<float>()[i] = r.sym() * 4;
        const uint32_t nn = n;
        const uint32_t a2[2] = {(uint32_t)T, (uint32_t)NE};
        mtl::Dispatch d1; d1.p = &ps; d1.args = {mtl::Arg::B(g), mtl::Arg::B(u), mtl::Arg::B(h), mtl::Arg::V(nn)};
        d1.grid[0] = (n + 255) / 256; d1.tg[0] = 256;
        mtl::Dispatch d2; d2.p = &pm; d2.args = {mtl::Arg::B(yr), mtl::Arg::B(ys), mtl::Arg::B(sg), mtl::Arg::B(bo), mtl::Arg::V(a2)};
        d2.grid[0] = (T * NE + 255) / 256; d2.tg[0] = 256;
        cx.run({d1, d2});
        double w1 = 0, w2 = 0;
        for (int i = 0; i < n; ++i) {
            const double gv = g.as<float>()[i], ref = gv * sig(gv) * u.as<float>()[i];
            w1 = std::max(w1, std::fabs(ref::f16_to_f32(h.as<uint16_t>()[i]) - ref) / (std::fabs(ref) + 1e-3));
        }
        for (int i = 0; i < T * NE; ++i) {
            const double ref = yr.as<float>()[i] + sig(sg.as<float>()[i / NE]) * ys.as<float>()[i];
            w2 = std::max(w2, std::fabs(bo.as<float>()[i] - ref));
        }
        const bool ok = w1 < 1e-3 && w2 < 1e-5;
        fails += !ok;
        std::printf("swiglu_h worst rel err %.2e, moe_out worst abs err %.2e  %s\n", w1, w2, ok ? "PASS" : "FAIL");
    }
    std::printf("%s\n", fails ? "SOME FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
