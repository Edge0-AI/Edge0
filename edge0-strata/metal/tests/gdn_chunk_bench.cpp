// metal/tests/gdn_chunk_bench.cpp - GDN prompt stage microbenchmark and numerical check (30 Oct GDN-M, phase 1).
// The sequential scans of gdn_prompt.metal (gdn_scanr_2/4/8, production = gdn_scanr_4) against the chunked C = 32 form
// (gdn_cprep -> gdn_cstate -> gdn_cout) at the real Flash-Next shape (h_k 16, h_v 48, S 128, 10240 conv channels),
// T = 512 .. 32768, with the same gdn_pre in front and gdn_post behind (the whole GDN stage).  Checks against an FP64
// sequential oracle on a few heads (output o and final state), the pipeline thread limits, and continuation (a call
// split in two, chunked or sequential on either side, the second call on buffer offsets).
//   gdn_chunk_bench [--T 512,2048,..] [--check] [--regime normal|strong] [--heads 0,17,47] [--reps 5] [--noscan]
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
constexpr int HK = 16, HV = 48, CH = 10240, S = 128, GC = 32, NSC = 14;
struct GdnPArgs { uint32_t n_tok, h_k, h_v, channels; float eps; uint32_t f32; };
double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
std::vector<int> ints(const char* s) {
    std::vector<int> r;
    for (const char* p = s; *p;) { r.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; }
    return r;
}
}  // namespace

int main(int argc, char** argv) {
    std::vector<int> Ts = {512, 1024, 2048, 4096, 8192, 16384, 32768}, heads = {0, 17, 47};
    bool check = false, noscan = false, strong = false;
    int reps = 5;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--T") Ts = ints(argv[++i]);
        else if (a == "--heads") heads = ints(argv[++i]);
        else if (a == "--check") check = true;
        else if (a == "--noscan") noscan = true;
        else if (a == "--regime") strong = std::string(argv[++i]) == "strong";
        else if (a == "--reps") reps = std::atoi(argv[++i]);
    }
    const int TMAX = *std::max_element(Ts.begin(), Ts.end());
    mtl::Context cx;
    const int lib = cx.compile({"gdn_prompt.metal"}, "#define NO_GGML_COMMON 1\n");
    std::printf("device %s, regime %s, TMAX %d\n", cx.device_name().c_str(), strong ? "strong-decay" : "normal", TMAX);
    auto F = [&](size_t n) { return cx.alloc(4 * n); };
    const size_t NCM = (TMAX + GC - 1) / GC;
    mtl::Buffer qkv = F((size_t)TMAX * CH), al = F((size_t)TMAX * HV), be = F((size_t)TMAX * HV), z = F((size_t)TMAX * HV * S),
                cw = F(CH * 4), dt = F(HV), sa = F(HV), gam = F(S), st0 = F((size_t)S * HV * S), stX = F((size_t)S * HV * S),
                stRef = F((size_t)S * HV * S), h0 = F(CH * 3), hOut = F(CH * 3), yh = cx.alloc(2 * (size_t)TMAX * HV * S),
                yB = F((size_t)TMAX * HV * S), qn = F((size_t)(TMAX + 32) * HK * S), kn = F((size_t)(TMAX + 32) * HK * S),
                vc = F((size_t)TMAX * HV * S), gb = F(2 * (size_t)TMAX * HV), kq = F((size_t)TMAX * HK),
                o = F((size_t)TMAX * HV * S), oRef = F((size_t)TMAX * HV * S), cU = F((size_t)HV * NCM * GC * S),
                cW = F((size_t)HV * NCM * GC * S), clg = F((size_t)HV * NCM * GC), cD = F((size_t)HV * NCM * GC * S), cM = F((size_t)HV * NCM * GC * GC);
    ref::Rng r(21);
    for (size_t i = 0; i < (size_t)TMAX * CH; ++i) qkv.as<float>()[i] = r.sym() * 2.0f;
    for (size_t i = 0; i < (size_t)TMAX * HV; ++i) { al.as<float>()[i] = r.sym() * 3.0f; be.as<float>()[i] = r.sym() * 2.0f; }
    for (size_t i = 0; i < (size_t)TMAX * HV * S; ++i) z.as<float>()[i] = r.sym() * 2.0f;
    for (int i = 0; i < CH * 4; ++i) cw.as<float>()[i] = r.sym() * 0.6f;
    for (int i = 0; i < HV; ++i) { dt.as<float>()[i] = r.sym(); sa.as<float>()[i] = strong ? -5.0f - r.uni() * 25.0f : -0.2f - r.uni() * 2.0f; }
    for (int i = 0; i < S; ++i) gam.as<float>()[i] = 0.5f + r.uni();
    for (int i = 0; i < S * HV * S; ++i) st0.as<float>()[i] = r.sym() * 0.1f;
    for (int i = 0; i < CH * 3; ++i) h0.as<float>()[i] = r.sym();

    mtl::Pipeline pre = cx.pipeline(lib, "gdn_pre"), post = cx.pipeline(lib, "gdn_post");
    // the existing sequential scans: {name, threadgroups, threads}; index 1 = gdn_scanr_4 = production
    struct Sv { const char* nm; int grid, tg; };
    const Sv svs[NSC] = {{"gdn_scanr_2", HV * 4, 64}, {"gdn_scanr_4", HV * 4, 128}, {"gdn_scanr_8", HV * 4, 256},
                         {"gdn_scanr_4_8", HV * 4, 128}, {"gdn_scan_v", HV * 4, 256}, {"gdn_scan_1", HV * 4, 32},
                         {"gdn_scan_2", HV * 2, 64}, {"gdn_scan_4", HV, 128},
                         {"gdn_scanr_4_4", HV * 4, 128}, {"gdn_scanr_4_2", HV * 4, 128}, {"gdn_scanr_8_8", HV * 4, 256},
                         {"gdn_scanr_8_4", HV * 4, 256}, {"gdn_scanr_2_8", HV * 4, 64}, {"gdn_scanr_2_4", HV * 4, 64}};
    std::vector<mtl::Pipeline> scan;
    for (const Sv& v : svs) scan.push_back(cx.pipeline(lib, v.nm, std::string(v.nm) == "gdn_scan_v" || std::string(v.nm).find("scanr") != std::string::npos ? mtl::Constants().i(6, 0) : mtl::Constants()));
    mtl::Pipeline pcp = cx.pipeline(lib, "gdn_cprep"), pcs = cx.pipeline(lib, "gdn_cstate"), pco = cx.pipeline(lib, "gdn_cout"), pcp2 = cx.pipeline(lib, "gdn_cprep2"), pcs2 = cx.pipeline(lib, "gdn_cstate2");
    bool v2 = false;   // chunk_d builds the v2 chunked form (gdn_cprep2 + gdn_cstate2)
    std::printf("pipeline limits (max threads / simd / static tg mem; threads asked):");
    for (size_t i = 0; i < scan.size(); ++i) std::printf(" %s %d/%d/%d (%d)", svs[i].nm, scan[i].max_threads, scan[i].simd_width, scan[i].static_tg_mem, svs[i].tg);
    std::printf("\n  pre %d/%d/%d (256)  cprep %d/%d/%d (256)  cstate %d/%d/%d (256)  cout %d/%d/%d (128)  post %d/%d/%d (128)\n",
                pre.max_threads, pre.simd_width, pre.static_tg_mem, pcp.max_threads, pcp.simd_width, pcp.static_tg_mem, pcs.max_threads,
                pcs.simd_width, pcs.static_tg_mem, pco.max_threads, pco.simd_width, pco.static_tg_mem, post.max_threads, post.simd_width,
                post.static_tg_mem);
    std::printf("scratch at T: U+W+delta+lg = %.1f KB per token (%.0f MB at T 2048, %.0f MB at T 4096)\n",
                (3.0 * HV * S * 4 + HV * 4) / 1024.0, (3.0 * HV * S * 4 + HV * 4) * 2048 / 1e6,
                (3.0 * HV * S * 4 + HV * 4) * 4096 / 1e6);

    // token range [t0, t0 + n) as buffer offsets
    auto pre_d = [&](int t0, int n, std::vector<mtl::Dispatch>& ds, bool first) {
        GdnPArgs a{(uint32_t)n, HK, HV, CH, 1e-6f, 1u};
        mtl::Dispatch d;
        d.p = &pre;
        // history: the chunk reads hist (h0 for the first call, hOut afterwards), writes hOut
        d.args = {mtl::Arg::B(qkv, (size_t)t0 * CH * 4), mtl::Arg::B(first ? h0 : hOut), mtl::Arg::B(cw),
                  mtl::Arg::B(al, (size_t)t0 * HV * 4), mtl::Arg::B(be, (size_t)t0 * HV * 4), mtl::Arg::B(dt), mtl::Arg::B(sa),
                  mtl::Arg::B(qn, (size_t)t0 * HK * S * 4), mtl::Arg::B(kn, (size_t)t0 * HK * S * 4),
                  mtl::Arg::B(vc, (size_t)t0 * HV * S * 4), mtl::Arg::B(gb, (size_t)t0 * HV * 8), mtl::Arg::B(kq, (size_t)t0 * HK * 4),
                  mtl::Arg::B(hOut), mtl::Arg::V(a)};
        d.grid[0] = HK + HV / 2; d.grid[1] = n; d.tg[0] = 256;
        ds.push_back(d);
    };
    auto scan_d = [&](int which, int t0, int n, const mtl::Buffer& st, const mtl::Buffer& out, std::vector<mtl::Dispatch>& ds) {
        GdnPArgs a{(uint32_t)n, HK, HV, CH, 1e-6f, 1u};
        mtl::Dispatch d;
        d.p = &scan[which];
        d.args = {mtl::Arg::B(st), mtl::Arg::B(qn, (size_t)t0 * HK * S * 4), mtl::Arg::B(kn, (size_t)t0 * HK * S * 4),
                  mtl::Arg::B(vc, (size_t)t0 * HV * S * 4), mtl::Arg::B(gb, (size_t)t0 * HV * 8), mtl::Arg::B(kq, (size_t)t0 * HK * 4),
                  mtl::Arg::B(out, (size_t)t0 * HV * S * 4), mtl::Arg::V(a)};
        d.grid[0] = svs[which].grid; d.tg[0] = svs[which].tg;
        ds.push_back(d);
    };
    auto chunk_d = [&](int t0, int n, const mtl::Buffer& st, const mtl::Buffer& out, std::vector<mtl::Dispatch>& ds) {
        GdnPArgs a{(uint32_t)n, HK, HV, CH, 1e-6f, 1u};
        const uint32_t nc = (n + GC - 1) / GC;
        if (v2) {
            mtl::Dispatch f1; f1.p = &pcp2;
            f1.args = {mtl::Arg::B(qn, (size_t)t0 * HK * S * 4), mtl::Arg::B(kn, (size_t)t0 * HK * S * 4),
                       mtl::Arg::B(vc, (size_t)t0 * HV * S * 4), mtl::Arg::B(gb, (size_t)t0 * HV * 8), mtl::Arg::B(cU), mtl::Arg::B(cW),
                       mtl::Arg::B(clg), mtl::Arg::B(cM), mtl::Arg::V(a)};
            f1.grid[0] = HV; f1.grid[1] = nc; f1.tg[0] = 256; ds.push_back(f1);
            mtl::Dispatch f2; f2.p = &pcs2;
            f2.args = {mtl::Arg::B(st), mtl::Arg::B(qn, (size_t)t0 * HK * S * 4), mtl::Arg::B(kn, (size_t)t0 * HK * S * 4),
                       mtl::Arg::B(cU), mtl::Arg::B(cW), mtl::Arg::B(clg), mtl::Arg::B(cM), mtl::Arg::B(out, (size_t)t0 * HV * S * 4),
                       mtl::Arg::V(a)};
            f2.grid[0] = HV * 4; f2.tg[0] = 128; ds.push_back(f2);
            return;
        }
        mtl::Dispatch e1; e1.p = &pcp;
        e1.args = {mtl::Arg::B(qn, (size_t)t0 * HK * S * 4), mtl::Arg::B(kn, (size_t)t0 * HK * S * 4),
                   mtl::Arg::B(vc, (size_t)t0 * HV * S * 4), mtl::Arg::B(gb, (size_t)t0 * HV * 8), mtl::Arg::B(cU), mtl::Arg::B(cW),
                   mtl::Arg::B(clg), mtl::Arg::V(a)};
        e1.grid[0] = HV; e1.grid[1] = nc; e1.tg[0] = 256; ds.push_back(e1);
        mtl::Dispatch e2; e2.p = &pcs;
        e2.args = {mtl::Arg::B(st), mtl::Arg::B(qn, (size_t)t0 * HK * S * 4), mtl::Arg::B(kn, (size_t)t0 * HK * S * 4),
                   mtl::Arg::B(cU), mtl::Arg::B(cW), mtl::Arg::B(clg), mtl::Arg::B(cD), mtl::Arg::B(out, (size_t)t0 * HV * S * 4),
                   mtl::Arg::V(a)};
        e2.grid[0] = HV * 4; e2.tg[0] = 256; ds.push_back(e2);
        mtl::Dispatch e3; e3.p = &pco;
        e3.args = {mtl::Arg::B(qn, (size_t)t0 * HK * S * 4), mtl::Arg::B(kn, (size_t)t0 * HK * S * 4), mtl::Arg::B(cD),
                   mtl::Arg::B(clg), mtl::Arg::B(out, (size_t)t0 * HV * S * 4), mtl::Arg::V(a)};
        e3.grid[0] = HV; e3.grid[1] = nc; e3.tg[0] = 128; ds.push_back(e3);
    };
    auto post_d = [&](int t0, int n, const mtl::Buffer& out, std::vector<mtl::Dispatch>& ds) {
        GdnPArgs a{(uint32_t)n, HK, HV, CH, 1e-6f, 1u};
        mtl::Dispatch d;
        d.p = &post;
        d.args = {mtl::Arg::B(out, (size_t)t0 * HV * S * 4), mtl::Arg::B(z, (size_t)t0 * HV * S * 4), mtl::Arg::B(gam),
                  mtl::Arg::B(yh, (size_t)t0 * HV * S * 2), mtl::Arg::B(yB, (size_t)t0 * HV * S * 4), mtl::Arg::V(a)};
        d.grid[0] = HV; d.grid[1] = n; d.tg[0] = 128;
        ds.push_back(d);
    };
    mtl::Pipeline pf[3] = {cx.pipeline(lib, "gdn_scanf_4"), cx.pipeline(lib, "gdn_scanf_8"), cx.pipeline(lib, "gdn_scanf_2")};
    std::printf("fused scan+post pipelines (max threads / static tg mem; 512 asked): TB4 %d/%d  TB8 %d/%d  TB2  %d/%d\n", pf[0].max_threads,
                pf[0].static_tg_mem, pf[1].max_threads, pf[1].static_tg_mem, pf[2].max_threads, pf[2].static_tg_mem);
    auto fused_d = [&](int which, int t0, int n, const mtl::Buffer& st, std::vector<mtl::Dispatch>& ds) {
        GdnPArgs a{(uint32_t)n, HK, HV, CH, 1e-6f, 1u};
        mtl::Dispatch d;
        d.p = &pf[which];
        d.args = {mtl::Arg::B(st), mtl::Arg::B(qn, (size_t)t0 * HK * S * 4), mtl::Arg::B(kn, (size_t)t0 * HK * S * 4),
                  mtl::Arg::B(vc, (size_t)t0 * HV * S * 4), mtl::Arg::B(gb, (size_t)t0 * HV * 8), mtl::Arg::B(kq, (size_t)t0 * HK * 4),
                  mtl::Arg::B(z, (size_t)t0 * HV * S * 4), mtl::Arg::B(gam), mtl::Arg::B(yh, (size_t)t0 * HV * S * 2),
                  mtl::Arg::B(yB, (size_t)t0 * HV * S * 4), mtl::Arg::V(a)};
        d.grid[0] = HV; d.tg[0] = 512;
        ds.push_back(d);
    };
    auto timeit = [&](const std::vector<mtl::Dispatch>& ds) {
        cx.warm(ds, 0.3);
        std::vector<double> t;
        for (int i = 0; i < reps; ++i) t.push_back(cx.run(ds));
        return median(t) * 1e3;
    };
    auto copy_st0 = [&](mtl::Buffer& dst) { std::memcpy(dst.ptr, st0.ptr, (size_t)S * HV * S * 4); };

    // FP64 oracle for head h over tokens [t0, t0 + n) from the prepared tensors; state in/out [S][S] (i, j)
    auto oracle = [&](int h, int t0, int n, std::vector<double>& St, std::vector<double>& out) {
        const int hk = h % HK;
        out.assign((size_t)n * S, 0.0);
        std::vector<double> a(S), b(S);
        for (int t = t0; t < t0 + n; ++t) {
            const float* k = kn.as<float>() + ((size_t)t * HK + hk) * S;
            const float* q = qn.as<float>() + ((size_t)t * HK + hk) * S;
            const float* v = vc.as<float>() + ((size_t)t * HV + h) * S;
            const double g = gb.as<float>()[((size_t)t * HV + h) * 2], bt = gb.as<float>()[((size_t)t * HV + h) * 2 + 1];
            double kqv = 0;
            for (int i = 0; i < S; ++i) kqv += (double)k[i] * q[i];
            std::fill(a.begin(), a.end(), 0.0); std::fill(b.begin(), b.end(), 0.0);
            for (int i = 0; i < S; ++i) {
                const double* row = &St[(size_t)i * S];
                const double ki = k[i], qi = q[i];
                for (int j = 0; j < S; ++j) { a[j] += row[j] * ki; b[j] += row[j] * qi; }
            }
            std::vector<double> dl(S);
            for (int j = 0; j < S; ++j) {
                dl[j] = (v[j] - g * a[j]) * bt;
                out[(size_t)(t - t0) * S + j] = (g * b[j] + dl[j] * kqv) / std::sqrt((double)S);
            }
            for (int i = 0; i < S; ++i) {
                double* row = &St[(size_t)i * S];
                for (int j = 0; j < S; ++j) row[j] = g * row[j] + (double)k[i] * dl[j];
            }
        }
    };
    auto initial = [&](int h, std::vector<double>& St) {
        St.resize((size_t)S * S);
        for (int i = 0; i < S; ++i) for (int j = 0; j < S; ++j) St[(size_t)i * S + j] = st0.as<float>()[((size_t)i * HV + h) * S + j];
    };
    struct Err { double l2, mx; int nonfinite; };
    auto cmp_o = [&](const mtl::Buffer& ob, int h, int t0, int n, const std::vector<double>& ro) {
        double se = 0, sr = 0, mx = 0, rmax = 0; int nf = 0;
        for (int t = 0; t < n; ++t) for (int j = 0; j < S; ++j) {
            const float x = ob.as<float>()[((size_t)(t0 + t) * HV + h) * S + j];
            if (!std::isfinite(x)) { ++nf; continue; }
            const double e = x - ro[(size_t)t * S + j];
            se += e * e; sr += ro[(size_t)t * S + j] * ro[(size_t)t * S + j]; mx = std::max(mx, std::fabs(e)); rmax = std::max(rmax, std::fabs(ro[(size_t)t * S + j]));
        }
        return Err{std::sqrt(se / std::max(sr, 1e-30)), mx / std::max(rmax, 1e-30), nf};
    };
    auto cmp_s = [&](const mtl::Buffer& sb, int h, const std::vector<double>& rs) {
        double se = 0, sr = 0, mx = 0, rmax = 0; int nf = 0;
        for (int i = 0; i < S; ++i) for (int j = 0; j < S; ++j) {
            const float x = sb.as<float>()[((size_t)i * HV + h) * S + j];
            if (!std::isfinite(x)) { ++nf; continue; }
            const double e = x - rs[(size_t)i * S + j];
            se += e * e; sr += rs[(size_t)i * S + j] * rs[(size_t)i * S + j]; mx = std::max(mx, std::fabs(e)); rmax = std::max(rmax, std::fabs(rs[(size_t)i * S + j]));
        }
        return Err{std::sqrt(se / std::max(sr, 1e-30)), mx / std::max(rmax, 1e-30), nf};
    };

    // prepared tensors once for all TMAX tokens (gdn_pre per call carries the history; here one call, the tokens independent
    // of the scan variant).  For the continuation check the pre runs per split below.
    {
        std::vector<mtl::Dispatch> ds;
        // gdn_pre on a very long T reads hist only for the first 3 tokens; one call over TMAX is the same as pieces
        pre_d(0, TMAX, ds, true);
        cx.run(ds);
    }
    int fails = 0;
    if (check) {
        std::printf("== numerical checks vs FP64 oracle (rel L2 / max-abs over max|ref|, per head)\n");
        for (int T : Ts) {
            // the variants, each from st0, over tokens [0, T)
            struct V { const char* name; int kind; };  // kind: 0..2 scanr IB 2/4/8, 10 chunked
            for (V v : {V{"scanr_4", 1}, V{"chunked", 10}, V{"chunked2", 11}}) {
                copy_st0(stX);
                std::vector<mtl::Dispatch> ds;
                v2 = v.kind == 11; if (v.kind >= 10) chunk_d(0, T, stX, o, ds); else scan_d(v.kind, 0, T, stX, o, ds); v2 = false;
                cx.run(ds);
                for (int h : heads) {
                    std::vector<double> St, ro; initial(h, St); oracle(h, 0, T, St, ro);
                    Err eo = cmp_o(o, h, 0, T, ro), es = cmp_s(stX, h, St);
                    const bool ok = eo.l2 < 3e-3 && es.l2 < 3e-3 && !eo.nonfinite && !es.nonfinite;
                    fails += !ok;
                    std::printf("T %5d %-8s head %2d: o rel %.2e (max %.2e)  state rel %.2e (max %.2e)  nonfinite %d  %s\n", T, v.name,
                                h, eo.l2, eo.mx, es.l2, es.mx, eo.nonfinite + es.nonfinite, ok ? "PASS" : "FAIL");
                    if (T > 8192) break;   // one head at the largest sizes (oracle cost)
                }
            }
        }
        // continuation: T = A + B, second call on offsets and the first call's state; every pairing
        std::printf("== continuation (state carried across two calls, 2nd call on buffer offsets) vs oracle over the whole range\n");
        const std::pair<int, int> splits[] = {{37, 963}, {1024, 1024}, {2048, 2047}, {31, 33}, {1, 4095}};
        for (auto [A, B] : splits) {
            if (A + B > TMAX) continue;
            const char* nm[8] = {"chunk+chunk", "scan+chunk", "chunk+scan", "scan+scan", "chunk2+chunk2", "scan+chunk2", "chunk2+scan", "-"};
            for (int m = 0; m < 8; ++m) {
                v2 = m >= 4;
                if (v2 && m == 7) continue;   // scan+scan has no chunked part
                copy_st0(stX);
                std::vector<mtl::Dispatch> d1, d2;
                if (m % 4 == 0 || m % 4 == 2) chunk_d(0, A, stX, o, d1); else scan_d(1, 0, A, stX, o, d1);
                if (m % 4 == 0 || m % 4 == 1) chunk_d(A, B, stX, o, d2); else scan_d(1, A, B, stX, o, d2);
                cx.run(d1); cx.run(d2); v2 = false;
                for (int h : {heads[0], heads.back()}) {
                    std::vector<double> St, ro; initial(h, St); oracle(h, 0, A + B, St, ro);
                    Err eo = cmp_o(o, h, 0, A + B, ro), es = cmp_s(stX, h, St);
                    const bool ok = eo.l2 < 3e-3 && es.l2 < 3e-3 && !eo.nonfinite && !es.nonfinite;
                    fails += !ok;
                    std::printf("split %5d + %5d %-14s head %2d: o rel %.2e  state rel %.2e  %s\n", A, B, nm[m], h, eo.l2, es.l2, ok ? "PASS" : "FAIL");
                }
            }
        }
        std::printf("== bitwise: gdn_scanr_4_8 / _4_4 / _4_2 / _8_8 / _2_8 o and state vs gdn_scanr_4\n");
        for (int T : Ts) {
            if (T > 16384) continue;
            copy_st0(stX);
            std::vector<mtl::Dispatch> d0; scan_d(1, 0, T, stX, o, d0); cx.run(d0);
            std::vector<float> oref(o.as<float>(), o.as<float>() + (size_t)T * HV * S), sref(stX.as<float>(), stX.as<float>() + (size_t)S * HV * S);
            for (int w : {3, 8, 9, 10, 12}) {
                copy_st0(stX);
                std::memset(o.ptr, 0, (size_t)T * HV * S * 4);
                std::vector<mtl::Dispatch> d1; scan_d(w, 0, T, stX, o, d1); cx.run(d1);
                const bool eq = std::memcmp(o.ptr, oref.data(), oref.size() * 4) == 0 && std::memcmp(stX.ptr, sref.data(), sref.size() * 4) == 0;
                fails += !eq;
                std::printf("T %5d %-14s %s\n", T, svs[w].nm, eq ? "BITWISE EQUAL" : "DIFFERS");
            }
        }
        std::printf("== fused scan+post vs pre/scanr_4/post: y rel L2, fp16 y mismatches, state rel L2\n");
        for (int T : Ts) {
            if (T > 8192) continue;
            copy_st0(stX);
            std::vector<mtl::Dispatch> d0; scan_d(1, 0, T, stX, o, d0); post_d(0, T, o, d0); cx.run(d0);
            std::vector<float> yref(yB.as<float>(), yB.as<float>() + (size_t)T * HV * S), sref(stX.as<float>(), stX.as<float>() + (size_t)S * HV * S);
            std::vector<uint16_t> href(yh.as<uint16_t>(), yh.as<uint16_t>() + (size_t)T * HV * S);
            for (int w = 0; w < 3; ++w) {
                copy_st0(stX);
                std::memset(yB.ptr, 0, (size_t)T * HV * S * 4);
                std::vector<mtl::Dispatch> d1; fused_d(w, 0, T, stX, d1); cx.run(d1);
                double se = 0, sr = 0, ss = 0, ssr = 0; size_t hm = 0;
                for (size_t i = 0; i < (size_t)T * HV * S; ++i) { const double e = yB.as<float>()[i] - yref[i]; se += e * e; sr += (double)yref[i] * yref[i]; hm += yh.as<uint16_t>()[i] != href[i]; }
                for (size_t i = 0; i < (size_t)S * HV * S; ++i) { const double e = stX.as<float>()[i] - sref[i]; ss += e * e; ssr += (double)sref[i] * sref[i]; }
                const bool ok = std::sqrt(se / sr) < 1e-5 && std::sqrt(ss / ssr) < 1e-5;
                fails += !ok;
                std::printf("T %5d fused TB%-2d: y rel %.2e, fp16 mismatches %zu, state rel %.2e  %s\n", T, w == 0 ? 4 : w == 1 ? 8 : 2, std::sqrt(se / sr), hm, std::sqrt(ss / ssr), ok ? "PASS" : "FAIL");
            }
        }
        std::printf("%s\n", fails ? "SOME FAILED" : "ALL PASS");
    }
    // timing: gdn_pre, the scan variants, chunked parts, gdn_post; the stage = pre + scan + post
    std::printf("== timing (GPU ms per call, median of %d; stage = pre + scan + post)\n", reps);
    std::printf("%6s | %7s %7s |", "T", "pre", "post");
    for (const Sv& v : svs) std::printf(" %s", v.nm + 4);
    std::printf(" | chunked (cprep cstate cout) | stage scanr_4 / chunked / x\n");
    for (int T : Ts) {
        std::vector<mtl::Dispatch> dp, dpo, dc;
        pre_d(0, T, dp, true);
        post_d(0, T, o, dpo);
        const double tp = timeit(dp), tpo = timeit(dpo);
        std::printf("%6d | %7.3f %7.3f |", T, tp, tpo);
        double ts1 = 0, tbest = 1e30;
        int wbest = 1;
        for (int w = 0; w < NSC; ++w) {
            std::vector<mtl::Dispatch> d1;
            double t = 0;
            if (!noscan) { scan_d(w, 0, T, stX, o, d1); t = timeit(d1); }
            if (w == 1) ts1 = t;
            if (t > 0 && t < tbest) { tbest = t; wbest = w; }
            std::printf(" %8.3f", t);
        }
        chunk_d(0, T, stX, o, dc);
        const double tc = timeit(dc), tc1 = timeit({dc[0]}), tc2 = timeit({dc[1]}), tc3 = timeit({dc[2]});
        const double base = tp + ts1 + tpo, ch = tp + tc + tpo;
        std::printf(" | %8.3f (%7.3f %7.3f %7.3f) | %8.3f %8.3f %5.2f | best %s stage %.3f = x%.3f (scan x%.3f)\n", tc, tc1, tc2, tc3, base, ch, base / ch, svs[wbest].nm, tp + tbest + tpo, base / (tp + tbest + tpo), ts1 / tbest);
        {
            v2 = true;
            std::vector<mtl::Dispatch> dc2;
            chunk_d(0, T, stX, o, dc2);
            const double t2 = timeit(dc2), t2a = timeit({dc2[0]}), t2b = timeit({dc2[1]});
            v2 = false;
            std::printf("       chunked v2 %.3f (cprep2 %.3f cstate2 %.3f); stage pre+v2+post %.3f = x%.3f vs scanr_4 stage (scan x%.3f)\n", t2, t2a, t2b,
                        tp + t2 + tpo, base / (tp + t2 + tpo), ts1 / t2);
        }
        if (std::getenv("GDN_PROBE_RUN")) {
            v2 = true;
            std::vector<mtl::Dispatch> dd;
            chunk_d(0, T, stX, o, dd);
            for (int pr = 1; pr <= 3; ++pr) {
                mtl::Pipeline a1 = cx.pipeline(lib, "gdn_cprep2", mtl::Constants().i(6, pr)), a2 = cx.pipeline(lib, "gdn_cstate2", mtl::Constants().i(6, pr));
                mtl::Dispatch d1 = dd[0], d2 = dd[1];
                d1.p = &a1; d2.p = &a2;
                const double x1 = timeit({d1}), x2 = timeit({d2});
                std::printf("       probe %d: cprep2 %.3f cstate2 %.3f ms\n", pr, x1, x2);
            }
            v2 = false;
        }
        {
            double tf[3];
            for (int w = 0; w < 3; ++w) { std::vector<mtl::Dispatch> d1; fused_d(w, 0, T, stX, d1); tf[w] = timeit(d1); }
            const double fb = tp + std::min(tf[0], std::min(tf[1], tf[2]));
            std::printf("       fused scan+post TB4 %.3f TB8 %.3f TB2  %.3f ms; stage pre+fused best %.3f = x%.3f vs scanr_4 stage\n", tf[0], tf[1], tf[2], fb, base / fb);
        }
        std::fflush(stdout);
    }
    return fails ? 1 : 0;
}
