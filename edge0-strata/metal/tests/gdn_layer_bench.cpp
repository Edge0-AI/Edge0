// metal/tests/gdn_layer_bench.cpp - one Flash-Next GDN layer's prompt cost, piece by piece (30 Oct GDN-M phase 2).
// Real shapes (N 2560, qkv 10240, z 6144, alpha/beta 48, out 2560 x 6144), a chunk of T rows (the engine's prompt chunk is
// 2048): dq (Q6_K -> fp16 weights), the four input GEMMs separate and fused into one, tile choices, gdn_pre, the scan, gdn_post,
// the out GEMM; each alone (GPU time) and the layer in order, plus the TFLOP/s of every GEMM.  Random weights: a timing
// of the byte and FLOP stream, not a numerical check.
//   gdn_layer_bench [--T 2048] [--reps 7] [--scan gdn_scanr_4]
#include "mtl_rt.h"
#include "ref_quant.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
constexpr int N = 2560, HK = 16, HV = 48, CH = 10240, ZD = 6144, S = 128, TQ6K = 14;
struct GemmArgs { int M, N, K, pad; };
struct DqArgs { uint32_t n_in, n_out, row_bytes, perm; };
struct GdnPArgs { uint32_t n_tok, h_k, h_v, channels; float eps; uint32_t f32; };
double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
}  // namespace

int main(int argc, char** argv) {
    int T = 2048, reps = 7;
    std::string scan_name = "gdn_scanr_4";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--T") T = std::atoi(argv[++i]);
        else if (a == "--reps") reps = std::atoi(argv[++i]);
        else if (a == "--scan") scan_name = argv[++i];
    }
    mtl::Context cx;
    const int lib = cx.compile({"quant.metal", "units.metal", "mma.metal", "dense_mm.metal"});
    const int lgdn = cx.compile({"gdn.metal", "gdn_prompt.metal"}, "#define NO_GGML_COMMON 1\n");
    std::printf("device %s, chunk T %d, scan %s\n", cx.device_name().c_str(), T, scan_name.c_str());
    ref::Rng r(5);
    auto fill_half = [&](mtl::Buffer& b, size_t n, float sc) {
        for (size_t i = 0; i < n; ++i) b.as<uint16_t>()[i] = ref::f32_to_f16(r.sym() * sc);
    };
    const int rows[5] = {CH, ZD, 48, 48, N};   // qkv, z, alpha, beta, out
    const int kin[5] = {N, N, N, N, ZD};
    mtl::Buffer mixh = cx.alloc(2 * (size_t)T * N), ygh = cx.alloc(2 * (size_t)T * ZD);
    fill_half(mixh, (size_t)T * N, 1.0f);
    fill_half(ygh, (size_t)T * ZD, 1.0f);
    mtl::Buffer Wq[5], Wh[5], Y[5];
    mtl::Buffer qsrc[5];
    for (int i = 0; i < 5; ++i) {
        Wh[i] = cx.alloc(2 * (size_t)rows[i] * kin[i]);
        fill_half(Wh[i], (size_t)rows[i] * kin[i], 0.03f);
        Y[i] = cx.alloc(4 * (size_t)T * rows[i]);
        const size_t rb = ref::row_bytes(TQ6K, kin[i]);
        qsrc[i] = cx.alloc(rb * rows[i]);
        for (size_t j = 0; j < rb * rows[i]; ++j) qsrc[i].as<uint8_t>()[j] = (uint8_t)(r.uni() * 255.0f);
        // keep the fp16 scale of each Q6_K block sane (the last two bytes of a 210 byte block)
        for (size_t j = 0; j + 210 <= rb * rows[i]; j += 210) { qsrc[i].as<uint16_t>()[(j + 208) / 2] = ref::f32_to_f16(0.01f); }
    }
    const int NF = CH + ZD + 48 + 48;
    mtl::Buffer WF = cx.alloc(2 * (size_t)NF * N), YF = cx.alloc(4 * (size_t)T * NF);
    fill_half(WF, (size_t)NF * N, 0.03f);

    auto timeit = [&](const std::vector<mtl::Dispatch>& ds) {
        cx.warm(ds, 0.3);
        std::vector<double> t;
        for (int i = 0; i < reps; ++i) t.push_back(cx.run(ds));
        return median(t) * 1e3;
    };
    auto dq_d = [&](int i) {
        DqArgs a{(uint32_t)kin[i], (uint32_t)rows[i], (uint32_t)ref::row_bytes(TQ6K, kin[i]), 0};
        static mtl::Pipeline p = cx.pipeline(lib, "dq_f16_14");
        mtl::Dispatch d; d.p = &p;
        d.args = {mtl::Arg::B(qsrc[i]), mtl::Arg::B(Wh[i]), mtl::Arg::V(a)};
        d.grid[0] = (kin[i] / 32 + 63) / 64; d.grid[1] = rows[i]; d.tg[0] = 64;
        return d;
    };
    auto gemm_d = [&](const mtl::Buffer& A, const mtl::Buffer& B, const mtl::Buffer& Yb, int M, int Nn, int K, int tm, int tn, int sgn = 4) {
        static std::vector<std::pair<std::string, mtl::Pipeline*>> cache;
        const std::string fn = "gemm_f16_" + std::to_string(tm) + "_" + std::to_string(tn) + "_" + std::to_string(sgn);
        mtl::Pipeline* p = nullptr;
        for (auto& c : cache) if (c.first == fn) p = c.second;
        if (!p) { p = new mtl::Pipeline(cx.pipeline(lib, fn)); cache.push_back({fn, p}); }
        GemmArgs g{M, Nn, K, 0};
        mtl::Dispatch d; d.p = p;
        d.args = {mtl::Arg::B(A), mtl::Arg::B(B), mtl::Arg::B(Yb), mtl::Arg::V(g)};
        d.grid[0] = (Nn + tn - 1) / tn; d.grid[1] = (M + tm - 1) / tm; d.tg[0] = 32 * sgn;
        return d;
    };
    auto tfl = [&](int Nn, int K, double ms) { return 2.0 * T * Nn * K / (ms * 1e-3) / 1e12; };

    std::printf("== dq (Q6_K -> fp16), ms: qkv %.3f z %.3f alpha %.3f beta %.3f out %.3f\n", timeit({dq_d(0)}), timeit({dq_d(1)}),
                timeit({dq_d(2)}), timeit({dq_d(3)}), timeit({dq_d(4)}));
    std::printf("== GEMMs (tile TMxTN, 4 simdgroups): ms and TFLOP/s\n");
    struct Sh { const char* nm; int i; };
    struct Tl { int tm, tn, sg; };
    const std::vector<Tl> tiles = {{64, 32, 4}, {64, 64, 4}, {128, 64, 4}, {64, 128, 4}, {128, 128, 4}, {128, 128, 8}, {128, 64, 8},
                                   {64, 64, 8}, {256, 64, 8}, {64, 256, 8}, {128, 32, 4}, {32, 128, 4}};
    for (Sh s : {Sh{"qkv", 0}, Sh{"z", 1}, Sh{"alpha", 2}, Sh{"out", 4}}) {
        std::printf("%-6s N %5d K %4d:", s.nm, rows[s.i], kin[s.i]);
        const mtl::Buffer& A = s.i == 4 ? ygh : mixh;
        for (const Tl& tl : tiles) {
            if (rows[s.i] <= 64 && tl.tn > 32) continue;
            const double ms = timeit({gemm_d(A, Wh[s.i], Y[s.i], T, rows[s.i], kin[s.i], tl.tm, tl.tn, tl.sg)});
            std::printf("  %dx%d/%d %.3f (%.0f)", tl.tm, tl.tn, tl.sg, ms, tfl(rows[s.i], kin[s.i], ms));
        }
        std::printf("\n");
    }
    std::printf("fused qkv+z+alpha+beta N %d:", NF);
    double best_f = 1e9;
    for (const Tl& tl : tiles) {
        const double ms = timeit({gemm_d(mixh, WF, YF, T, NF, N, tl.tm, tl.tn, tl.sg)});
        best_f = std::min(best_f, ms);
        std::printf("  %dx%d/%d %.3f (%.0f)", tl.tm, tl.tn, tl.sg, ms, tfl(NF, N, ms));
    }
    std::printf("\n");
    // the four input GEMMs as the engine runs them, in order
    const double sep = timeit({gemm_d(mixh, Wh[0], Y[0], T, CH, N, 64, 64), gemm_d(mixh, Wh[1], Y[1], T, ZD, N, 64, 64),
                               gemm_d(mixh, Wh[2], Y[2], T, 48, N, 64, 32), gemm_d(mixh, Wh[3], Y[3], T, 48, N, 64, 32)});
    std::printf("== input projections: separate (engine tiles) %.3f ms, fused best %.3f ms\n", sep, best_f);

    // pre / scan / post
    mtl::Buffer qkv = Y[0], z = Y[1], al = Y[2], be = Y[3], cw = cx.alloc(4 * CH * 4), dt = cx.alloc(4 * HV), sa = cx.alloc(4 * HV),
                gam = cx.alloc(4 * S), st = cx.alloc(4 * (size_t)S * HV * S), h0 = cx.alloc(4 * CH * 3), h1 = cx.alloc(4 * CH * 3),
                qn = cx.alloc(4 * (size_t)T * HK * S), kn = cx.alloc(4 * (size_t)T * HK * S), vc = cx.alloc(4 * (size_t)T * HV * S),
                gb = cx.alloc(8 * (size_t)T * HV), kq = cx.alloc(4 * (size_t)T * HK), o = cx.alloc(4 * (size_t)T * HV * S),
                yh = cx.alloc(2 * (size_t)T * HV * S), y32 = cx.alloc(4);
    for (int i = 0; i < CH * 4; ++i) cw.as<float>()[i] = r.sym() * 0.6f;
    for (int i = 0; i < HV; ++i) { dt.as<float>()[i] = r.sym(); sa.as<float>()[i] = -0.2f - r.uni() * 2.0f; }
    for (int i = 0; i < S; ++i) gam.as<float>()[i] = 0.5f + r.uni();
    for (size_t i = 0; i < (size_t)T * CH; ++i) qkv.as<float>()[i] = r.sym() * 2.0f;
    for (size_t i = 0; i < (size_t)T * HV; ++i) { al.as<float>()[i] = r.sym() * 3.0f; be.as<float>()[i] = r.sym() * 2.0f; }
    for (size_t i = 0; i < (size_t)T * ZD; ++i) z.as<float>()[i] = r.sym() * 2.0f;
    mtl::Pipeline pre = cx.pipeline(lgdn, "gdn_pre"), post = cx.pipeline(lgdn, "gdn_post"),
                  scan = cx.pipeline(lgdn, scan_name, mtl::Constants().i(6, 0));
    GdnPArgs ga{(uint32_t)T, HK, HV, CH, 1e-6f, 0};
    mtl::Dispatch dpre; dpre.p = &pre;
    dpre.args = {mtl::Arg::B(qkv), mtl::Arg::B(h0), mtl::Arg::B(cw), mtl::Arg::B(al), mtl::Arg::B(be), mtl::Arg::B(dt), mtl::Arg::B(sa),
                 mtl::Arg::B(qn), mtl::Arg::B(kn), mtl::Arg::B(vc), mtl::Arg::B(gb), mtl::Arg::B(kq), mtl::Arg::B(h1), mtl::Arg::V(ga)};
    dpre.grid[0] = HK + HV / 2; dpre.grid[1] = T; dpre.tg[0] = 256;
    mtl::Dispatch dscan; dscan.p = &scan;
    dscan.args = {mtl::Arg::B(st), mtl::Arg::B(qn), mtl::Arg::B(kn), mtl::Arg::B(vc), mtl::Arg::B(gb), mtl::Arg::B(kq), mtl::Arg::B(o),
                  mtl::Arg::V(ga)};
    dscan.grid[0] = HV * 4; dscan.tg[0] = 128;
    mtl::Dispatch dpost; dpost.p = &post;
    dpost.args = {mtl::Arg::B(o), mtl::Arg::B(z), mtl::Arg::B(gam), mtl::Arg::B(yh), mtl::Arg::B(y32), mtl::Arg::V(ga)};
    dpost.grid[0] = HV; dpost.grid[1] = T; dpost.tg[0] = 128;
    const double tpre = timeit({dpre}), tscan = timeit({dscan}), tpost = timeit({dpost});
    std::printf("== gdn_pre %.3f, scan %.3f, gdn_post %.3f ms (stage %.3f)\n", tpre, tscan, tpost, tpre + tscan + tpost);
    double tF = 0;
    {   // the fused-stride kernels (function constant 7) against the separate ones: bit for bit
        for (int t = 0; t < T; ++t) {
            std::memcpy(YF.as<float>() + (size_t)t * NF, qkv.as<float>() + (size_t)t * CH, 4 * CH);
            std::memcpy(YF.as<float>() + (size_t)t * NF + CH, z.as<float>() + (size_t)t * ZD, 4 * ZD);
            std::memcpy(YF.as<float>() + (size_t)t * NF + CH + ZD, al.as<float>() + (size_t)t * 48, 4 * 48);
            std::memcpy(YF.as<float>() + (size_t)t * NF + CH + ZD + 48, be.as<float>() + (size_t)t * 48, 4 * 48);
        }
        std::memset(h1.ptr, 0, 4 * CH * 3);
        cx.run({dpre, dscan, dpost});
        std::vector<char> rqn((char*)qn.ptr, (char*)qn.ptr + 4 * (size_t)T * HK * S), rvc((char*)vc.ptr, (char*)vc.ptr + 4 * (size_t)T * HV * S),
            rgb((char*)gb.ptr, (char*)gb.ptr + 8 * (size_t)T * HV), ryh((char*)yh.ptr, (char*)yh.ptr + 2 * (size_t)T * HV * S),
            rh((char*)h1.ptr, (char*)h1.ptr + 4 * CH * 3);
        mtl::Pipeline preF = cx.pipeline(lgdn, "gdn_pre", mtl::Constants().i(7, NF)), postF = cx.pipeline(lgdn, "gdn_post", mtl::Constants().i(7, NF));
        mtl::Dispatch fpre = dpre, fpost = dpost;
        fpre.p = &preF; fpost.p = &postF;
        fpre.args[0] = mtl::Arg::B(YF, 0);
        fpre.args[3] = mtl::Arg::B(YF, 4 * (size_t)(CH + ZD));
        fpre.args[4] = mtl::Arg::B(YF, 4 * (size_t)(CH + ZD + 48));
        fpost.args[1] = mtl::Arg::B(YF, 4 * (size_t)CH);
        std::memset(qn.ptr, 0, rqn.size()); std::memset(vc.ptr, 0, rvc.size()); std::memset(gb.ptr, 0, rgb.size());
        std::memset(yh.ptr, 0, ryh.size()); std::memset(h1.ptr, 0, rh.size());
        cx.run({fpre, dscan, fpost});
        std::printf("fused-stride gdn_pre/gdn_post vs separate: qn %s, vc %s, gate/beta %s, history %s, fp16 y %s\n",
                    std::memcmp(qn.ptr, rqn.data(), rqn.size()) == 0 ? "equal" : "DIFFER", std::memcmp(vc.ptr, rvc.data(), rvc.size()) == 0 ? "equal" : "DIFFER",
                    std::memcmp(gb.ptr, rgb.data(), rgb.size()) == 0 ? "equal" : "DIFFER", std::memcmp(h1.ptr, rh.data(), rh.size()) == 0 ? "equal" : "DIFFER",
                    std::memcmp(yh.ptr, ryh.data(), ryh.size()) == 0 ? "equal" : "DIFFER");
        tF = timeit({fpre}) + timeit({fpost});
        std::printf("fused-stride pre + post %.3f ms (separate %.3f)\n", tF, tpre + tpost);
    }
    const double tout = timeit({gemm_d(ygh, Wh[4], Y[4], T, N, ZD, 64, 64)});
    double dqall = 0;
    for (int i = 0; i < 5; ++i) dqall += timeit({dq_d(i)});
    std::printf("== layer in engine order (GPU ms): dq %.3f + input GEMMs %.3f + pre/scan/post %.3f + out GEMM %.3f = %.3f (sum of parts)\n",
                dqall, sep, tpre + tscan + tpost, tout, dqall + sep + tpre + tscan + tpost + tout);
    {   // the layer as the engine now runs it on the M5 family: fused input GEMM, fused-stride pre/post, tiled out GEMM
        double tout_best = 1e9;
        for (auto [tm, tn] : std::vector<std::pair<int, int>>{{64, 64}, {128, 64}, {64, 128}})
            tout_best = std::min(tout_best, timeit({gemm_d(ygh, Wh[4], Y[4], T, N, ZD, tm, tn)}));
        const double base_l = dqall + sep + tpre + tscan + tpost + tout;
        const double new_l = dqall + best_f + tF + tscan + tout_best;
        std::printf("== layer: before %.3f ms, after (fused input GEMM %.3f, fused-stride pre+post %.3f, scan %.3f, out GEMM %.3f) %.3f ms = x%.3f\n",
                    base_l, best_f, tF, tscan, tout_best, new_l, base_l / new_l);
    }
    return 0;
}
