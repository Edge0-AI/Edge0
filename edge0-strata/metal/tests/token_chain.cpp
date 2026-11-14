// metal/tests/token_chain.cpp - one decode step of the whole model, every kernel of metal/ on the REAL weights of a
// GGUF (mmapped, wrapped no-copy as one MTLBuffer), each layer with its own tensor formats, all dispatches in one
// command buffer: the GPU time of a decode token (or a verify window) with these kernels, dispatch boundaries
// included.  Not a numerical check (the activations are synthetic): a timing of the real byte stream.
//
//   token_chain <shard-1.gguf> [--ntok N] [--ctx P] [--reps R] [--skip-qsa] [--only CLASS]
//
// Left out: the PLE path (one table row and two small projections at one layer) and the MTP draft.
#include "gguf_lite.h"
#include "mtl_rt.h"
#include "ref_quant.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <sys/mman.h>
#include <vector>

namespace {

constexpr int N = 2560, HC = 4, LR = 320, NFF = 640, TOPK = 10, NEXP = 512, MAXT = 8;
constexpr int P_MAXE = 128, P_GEXP = 16, P_GSTART = P_GEXP + P_MAXE, P_ETOK = P_GSTART + P_MAXE + 4,
              P_EW = P_ETOK + P_MAXE, P_TOKENT = P_EW + P_MAXE, P_INTS = P_TOKENT + P_MAXE, P_IGU = 4, P_IDN = 8;
struct GemvArgs { uint32_t n_in, n_out, n_tok, row_bytes, x_stride, y_stride; };
struct GrArgs { uint32_t n_embd, hc, hc_lr, n_tok, apply; float eps; };
struct GrArgs3 { uint32_t n_embd, hc, hc_lr, n_tok, apply; float eps; uint32_t rb_down, rb_up, comb, topk; };
struct MoeArgs {
    uint64_t gate_off, up_off, down_off, gu_stride, d_stride;
    uint32_t gu_row, d_row, n_embd, n_ff, n_tok, topk, n_expert, gx_gu, gx_dn, pad;
};
struct GdnArgs { uint32_t h_k, h_v, channels, commit; float eps; uint32_t plane; };
struct QsaArgs { uint32_t pos0, pos_base, max_cells, n_tok; float eps; };
struct GlueArgs { uint32_t n, n_tok, n_embd, topk; };
struct ArgArgs { uint32_t n_vocab, parts; };

int xkind(int t) {
    switch (t) {
        case ref::Q2_0: return 1;
        case ref::Q4_K: case ref::Q5_K: return 2;
        case ref::Q6_K: case ref::Q3_K: return 3;
        default: return 0;
    }
}
double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

// a prepared activation: 4 planes of n_tok rows of n, then the per-16 sums
struct Act {
    mtl::Buffer b;
    int n = 0;
    size_t plane_bytes() const { return (size_t)4 * MAXT * n; }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: token_chain <shard-1.gguf> [--ntok N] [--ctx P] [--reps R]\n"); return 2; }
    int ntok = 1, ctx = 1000, reps = 10;
    bool skip_qsa = false, graph = false;
    // the HC read: Q8_0 copies through the 3-dispatch read by default (the owner's decision, as STRATA_HC_Q8 on gfx12);
    // STRATA_HC_Q8=0 selects the exact BF16 path.  --hc 3: BF16, 3 dispatches; 2: BF16, 2 dispatches; q8: Q8_0 copies,
    // 2 dispatches; q8_3: Q8_0 copies, 3 dispatches
    const char* hcq8 = std::getenv("STRATA_HC_Q8");
    std::string hc_mode = hcq8 && std::string(hcq8) == "0" ? "3" : "q8_3";
    bool fuse = false;           // the GDN y prep in the step; no QSA score/top-k within the selection budget
    bool fold_comb = false;      // the MoE combine folded into the next 2-dispatch HC read (measured slower)
    std::string only, skip;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--ntok") ntok = std::atoi(argv[++i]);
        else if (a == "--ctx") ctx = std::atoi(argv[++i]);
        else if (a == "--reps") reps = std::atoi(argv[++i]);
        else if (a == "--skip-qsa") skip_qsa = true;
        else if (a == "--graph") graph = true;
        else if (a == "--hc") hc_mode = argv[++i];
        else if (a == "--fuse") fuse = true;
        else if (a == "--fold-comb") fold_comb = true;
        else if (a == "--only") only = argv[++i];
        else if (a == "--skip") skip = argv[++i];
    }
    gguf::File f;
    std::string err;
    if (!f.open(argv[1], err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
    mtl::Context cx;
    std::printf("device %s; %s: %zu tensors, %.1f GB; %d token(s), context %d\n", cx.device_name().c_str(),
                argv[1], f.tensors.size(), f.file_bytes / 1e9, ntok, ctx);
    // fault the file in (page cache), then wrap it
    {
        const double t0 = mtl::now_s();
        madvise(f.map, f.map_bytes, MADV_WILLNEED);
        volatile uint64_t s = 0;
        for (size_t o = 0; o < f.file_bytes; o += 16384) s += ((const uint8_t*)f.map)[o];
        std::printf("file touched in %.1f s\n", mtl::now_s() - t0);
    }
    mtl::Buffer W = cx.wrap(f.map, f.map_bytes);
    auto T = [&](const std::string& name) -> const gguf::Tensor& {
        const gguf::Tensor* t = f.find(name);
        if (!t) { std::fprintf(stderr, "missing tensor %s\n", name.c_str()); std::exit(2); }
        return *t;
    };

    const int lib = cx.compile({"quant.metal", "units.metal", "moe.metal", "gemv.metal", "gr.metal", "gr2.metal", "glue.metal",
                                "router.metal"});
    const int lgdn = cx.compile({"gdn.metal"}, "#define NO_GGML_COMMON 1\n");
    const int lqsa = cx.compile({"qsa.metal", "qsa2.metal"}, "#define NO_GGML_COMMON 1\n");
    const int lsmp = cx.compile({"sample.metal"}, "#define NO_GGML_COMMON 1\n");
    std::map<std::string, mtl::Pipeline> pcache;

    // ---- buffers
    const int MAXC = std::max(4096, (ctx + 64) / 4 * 4 + 64), MB = MAXC / 4 + 2;
    mtl::Buffer R = cx.alloc(4 * MAXT * HC * N), bo = cx.alloc(4 * MAXT * N), inj[2] = {cx.alloc(4 * MAXT * HC), cx.alloc(4 * MAXT * HC)},
                xn = cx.alloc(4 * MAXT * HC * N), lo = cx.alloc(4 * MAXT * LR), qkv = cx.alloc(4 * MAXT * 10240),
                z = cx.alloc(4 * MAXT * 6144), al = cx.alloc(4 * MAXT * 48), be = cx.alloc(4 * MAXT * 48),
                yg = cx.alloc(4 * MAXT * 6144), logits = cx.alloc(4 * MAXT * 512), ids = cx.alloc(4 * MAXT * TOPK),
                wts = cx.alloc(4 * MAXT * TOPK), plan = cx.alloc(4 * P_INTS), h = cx.alloc(4 * MAXT * TOPK * NFF),
                ye = cx.alloc(4 * MAXT * TOPK * N), gsh = cx.alloc(4 * MAXT * NFF), ush = cx.alloc(4 * MAXT * NFF),
                ysh = cx.alloc(4 * MAXT * N), sg = cx.alloc(4 * MAXT * 4), qfull = cx.alloc(4 * MAXT * 12288),
                kcur = cx.alloc(4 * MAXT * 512), vcur = cx.alloc(4 * MAXT * 512), iqr = cx.alloc(4 * MAXT * 512),
                ikr = cx.alloc(4 * MAXT * 128), qcur = cx.alloc(4 * MAXT * 24 * 256), qidx = cx.alloc(4 * MAXT * 4 * 128),
                scores = cx.alloc(4 * (size_t)MAXT * MB), qids = cx.alloc(4 * MAXT * 2051),
                part = cx.alloc(4 * (size_t)MAXT * 65 * 24 * 258), cs = cx.alloc(8 * (size_t)(MAXC + 16) * 32),
                lm = cx.alloc(4 * (size_t)MAXT * 248320), apv = cx.alloc(4 * MAXT * 120), api = cx.alloc(4 * MAXT * 120),
                tokout = cx.alloc(4 * MAXT);
    auto act = [&](int n) { Act a; a.n = n; a.b = cx.alloc(4 * ((size_t)4 * MAXT * n + (size_t)MAXT * n / 16)); return a; };
    Act XA = act(N), XF = act(N), XY = act(6144), XQ = act(6144), XH = act(NFF);
    for (size_t i = 0; i < R.bytes / 4; ++i) R.as<float>()[i] = 0.01f * (float)((i * 2654435761u) % 200) - 1.0f;
    for (int p = 0; p < MAXC + 16; ++p)
        for (int i = 0; i < 32; ++i) {
            const double th = (double)p * std::pow(1e7, -2.0 * i / 64.0);
            cs.as<float>()[(p * 32 + i) * 2] = (float)std::cos(th);
            cs.as<float>()[(p * 32 + i) * 2 + 1] = (float)std::sin(th);
        }
    // per-layer recurrent state and QSA caches
    std::vector<mtl::Buffer> gst, ghist, ghist2, kq, vq, ksc, vsc, tail, dead, pooled;
    for (int l = 0; l < 48; ++l) {
        if (l % 4 != 3) {
            gst.push_back(cx.alloc(4 * 128 * 48 * 128));
            ghist.push_back(cx.alloc(4 * 10240 * 3));
            ghist2.push_back(cx.alloc(4 * 10240 * 3));
        } else {
            kq.push_back(cx.alloc((size_t)MAXC * 2 * 256));
            vq.push_back(cx.alloc((size_t)MAXC * 2 * 256));
            ksc.push_back(cx.alloc(2 * (size_t)MAXC * 2 * 4));
            vsc.push_back(cx.alloc(2 * (size_t)MAXC * 2 * 4));
            tail.push_back(cx.alloc(4 * 4 * 128));
            dead.push_back(cx.alloc(4 * 128));
            pooled.push_back(cx.alloc(4 * (size_t)MB * 128));
        }
    }
    for (int t = 0; t < MAXT; ++t)
        for (int c = 0; c < 2051; ++c) qids.as<int>()[t * 2051 + c] = c;
    for (auto& b : vsc) for (size_t i = 0; i < b.bytes / 2; ++i) b.as<uint16_t>()[i] = 0x2000;   // small fp16 scales
    for (auto& b : ksc) for (size_t i = 0; i < b.bytes / 2; ++i) b.as<uint16_t>()[i] = 0x2000;

    std::vector<mtl::Dispatch> ds;
    std::vector<std::string> cls;   // class of each dispatch (for --only)
    std::map<std::string, double> bytes_by;
    bool nb = false;   // the next push is independent of the previous dispatch (no barrier before it)
    auto push = [&](const mtl::Dispatch& d0, const std::string& c, double bytes = 0) {
        mtl::Dispatch d = d0;
        d.barrier = !nb;
        nb = false;
        if (!only.empty() && c != only) return;
        if (!skip.empty() && c == skip) return;
        ds.push_back(d);
        cls.push_back(c);
        bytes_by[c] += bytes;
    };
    // pipelines whose names do not match the function (cached by key): create by function name first
    auto Pn = [&](int l, const std::string& fn, const std::string& key, const mtl::Constants& c) -> const mtl::Pipeline* {
        auto it = pcache.find(key);
        if (it == pcache.end()) it = pcache.emplace(key, cx.pipeline(l, fn, c)).first;
        return &it->second;
    };
    // ---- SLC prefetch (--prefetch): the next HC read's weights touched concurrently with the first projection GEMV
    // after the current read, so the latency-bound read finds them in the SLC
    const bool prefetch = std::getenv("CHAIN_PREFETCH") != nullptr;
    const bool albe = std::getenv("CHAIN_ALBE") != nullptr;   // ssm_alpha + ssm_beta as one gemv_cat
    // the router GEMV + moe_route as one router_route (default; CHAIN_ROUTE=0: separate)
    const bool routef = !(std::getenv("CHAIN_ROUTE") && std::string(std::getenv("CHAIN_ROUTE")) == "0");
    const bool combf = std::getenv("CHAIN_COMB") != nullptr;   // the MoE combine folded into the next gr_norm
    bool comb_pending = false;
    // routed gate/up: the i-quant grids from constant memory (QUANT_CTAB, function constant 23; bitwise; default,
    // CHAIN_CTAB=0: the threadgroup copy)
    const bool ctabf = !(std::getenv("CHAIN_CTAB") && std::string(std::getenv("CHAIN_CTAB")) == "0");
    mtl::Buffer rcounter = cx.alloc(16);
    struct Range { const mtl::Buffer* b; size_t off, bytes; };
    std::vector<Range> pending;
    mtl::Buffer sink = cx.alloc(16);
    const mtl::Pipeline* ptouch = nullptr;
    auto push_prefetch = [&]() {
        for (const Range& r : pending) {
            const uint32_t n16 = (uint32_t)(r.bytes / 16);
            mtl::Dispatch d;
            d.p = ptouch;
            d.args = {mtl::Arg::B(*r.b, r.off), mtl::Arg::B(sink), mtl::Arg::V(n16)};
            d.grid[0] = (n16 + 1023) / 1024;
            d.tg[0] = 256;
            nb = true;
            push(d, "prefetch", (double)r.bytes);
        }
        pending.clear();
    };
    // ---- GEMV by tensor, with a tile from the measured table
    auto gemv2 = [&](const gguf::Tensor& t, const Act& x, const mtl::Buffer& y, size_t y_stride, const std::string& c,
                     int ty_override = -1) {
        const int n_in = (int)t.shape[0], n_out = t.shape.size() > 1 ? (int)t.shape[1] : 1;
        const int ty = ty_override >= 0 ? ty_override : t.type;
        int L = 16, Rr = 1, S = 4, K = 1;
        if (n_in >= 6144) { L = 32; Rr = 1; S = 8; K = 2; }
        else if (n_in == 640) { L = 8; Rr = 1; S = 4; K = 1; }
        else if (n_out <= 640 || ty == ref::Q2_0 || ty == ref::BF16) { L = 32; Rr = 1; S = 4; K = 1; }
        const std::string fn = "gemv_" + std::to_string(ty) + "_" + std::to_string(L) + "_" + std::to_string(Rr) + "_" +
                               std::to_string(S) + "_" + std::to_string(K);
        const mtl::Pipeline* p = Pn(lib, fn, fn + "#" + std::to_string(ntok), mtl::Constants().i(0, ntok));
        GemvArgs a{(uint32_t)n_in, (uint32_t)n_out, (uint32_t)ntok, (uint32_t)ref::row_bytes(t.type, n_in), (uint32_t)n_in,
                   (uint32_t)y_stride};
        mtl::Dispatch d;
        d.p = p;
        d.args = {mtl::Arg::B(W, t.offset), mtl::Arg::B(x.b, x.plane_bytes() * xkind(ty)), mtl::Arg::B(x.b, x.plane_bytes() * 4),
                  mtl::Arg::B(y), mtl::Arg::V(a)};
        const int rpt = (S / K) * (32 / L) * Rr;
        d.grid[0] = (n_out + rpt - 1) / rpt;
        d.tg[0] = S * 32;
        push(d, c, (double)t.bytes);
        if (!pending.empty()) push_prefetch();
    };
    const mtl::Pipeline* pnorm = Pn(lib, "gr_norm", "gr_norm", {});
    const mtl::Pipeline* pnormc = Pn(lib, "gr_norm_comb", "gr_norm_comb", {});
    ptouch = Pn(lib, "slc_touch", "slc_touch", {});
    const mtl::Pipeline* pdown = Pn(lib, "gr_down_8_8", "gr_down_8_8", mtl::Constants().i(1, ntok));
    const mtl::Pipeline* pup = Pn(lib, "gr_up", "gr_up", mtl::Constants().i(1, ntok));
    const mtl::Pipeline* pdownq = Pn(lib, "gr_down_q8_8_8", "gr_down_q8_8_8", mtl::Constants().i(1, ntok));
    const mtl::Pipeline* pupq = Pn(lib, "gr_up_q8", "gr_up_q8", mtl::Constants().i(1, ntok));
    const mtl::Pipeline* pprep = Pn(lib, "xprep", "xprep", {});
    const mtl::Pipeline* proute = Pn(lib, "moe_route", "moe_route", {});
    const mtl::Pipeline* pswi = Pn(lib, "swiglu_prep", "swiglu_prep", {});
    const mtl::Pipeline* pcomb = Pn(lib, "moe_combine_sh", "moe_combine_sh", {});
    const mtl::Pipeline* pgdn = Pn(lgdn, "gdn_step", "gdn_step", mtl::Constants().i(2, ntok));
    const mtl::Pipeline* qprep = Pn(lqsa, "qsa_prep", "qsa_prep", {});
    const mtl::Pipeline* qscore = Pn(lqsa, "qsa_score", "qsa_score", {});
    const int qav = std::getenv("QSA_ATTN") ? std::atoi(std::getenv("QSA_ATTN")) : 1;
    const bool qv2 = qav == 2;
    const mtl::Pipeline* qtopk = Pn(lqsa, std::getenv("QSA_TOPK") ? "qsa_topk" : "qsa_topk2", "qsa_topk", {});
    const mtl::Pipeline* qattn = Pn(lqsa, qav == 1 ? "qsa_attn" : qv2 ? "qsa_attn2" : "qsa_attn3", "qsa_attn", {});
    const mtl::Pipeline* qmerge = Pn(lqsa, qv2 ? "qsa_merge2" : "qsa_merge", "qsa_merge", {});
    const mtl::Pipeline* pam1 = Pn(lsmp, "argmax_part", "argmax_part", {});
    const mtl::Pipeline* pam2 = Pn(lsmp, "argmax_final", "argmax_final", {});

    // the 2-dispatch read: residual and partial-dot ping-pong buffers, rs; the Q8_0 copies of the HC matrices
    mtl::Buffer Rb[2] = {R, cx.alloc(4 * MAXT * HC * N)}, Db[2] = {cx.alloc(4 * MAXT * (LR + HC) * HC), cx.alloc(4 * MAXT * (LR + HC) * HC)},
                rsb = cx.alloc(4 * MAXT * HC);
    int cur_r = 0, cur_d = 0;
    std::map<std::string, size_t> q8off;
    mtl::Buffer Q8;
    if (hc_mode == "q8" || hc_mode == "q8_3") {
        size_t tot = 0;
        for (const auto& kv : f.tensors) {
            const gguf::Tensor& t = kv.second;
            if (t.type != ref::BF16 || t.name.find("hc_") == std::string::npos || t.name.find("_norm") != std::string::npos) continue;
            uint64_t ne = 1;
            for (uint64_t d : t.shape) ne *= d;
            q8off[t.name] = tot;
            tot += ne / 32 * 34;
        }
        Q8 = cx.alloc((tot + 4095) / 4096 * 4096);
        const double t0 = mtl::now_s();
        std::vector<float> tmp;
        for (const auto& kv : q8off) {
            const gguf::Tensor& t = *f.find(kv.first);
            uint64_t ne = 1;
            for (uint64_t d : t.shape) ne *= d;
            tmp.resize(ne);
            ref::dequant(ref::BF16, (const uint8_t*)f.map + t.offset, tmp.data(), ne);
            ref::quantize_q8_0(tmp.data(), ne, Q8.as<uint8_t>() + kv.second);
        }
        std::printf("HC Q8_0 copies: %zu tensors, %.1f MB, quantized in %.1f s\n", q8off.size(), tot / 1e6, mtl::now_s() - t0);
    }
    int cur_inj = 0;
    auto gr_read = [&](const std::string& pre, bool apply, const Act& out, bool final_) {
        if (hc_mode != "3" && hc_mode != "q8_3") {
            const bool q8 = hc_mode == "q8";
            const gguf::Tensor& tn = T(pre + "_norm.weight");
            const gguf::Tensor& tdn = T(pre + "_down.weight");
            const gguf::Tensor& tup = T(pre + "_up.weight");
            const gguf::Tensor& tin = final_ ? tdn : T(pre + "_inject.weight");
            const int tt = std::min(ntok, 3);
            const std::string dn = std::string("gr_down3_") + (q8 ? "8_" : "30_") + std::to_string(tt) + "_2";
            const std::string up = std::string("gr_up3_") + (q8 ? "8" : "30");
            // after an MoE (the next layer's mixer read, the final read) the combine is folded in here
            const bool comb = fold_comb && apply && pre.find("hc_ffn") == std::string::npos;
            GrArgs3 a{N, HC, LR, MAXT, apply ? 1u : 0u, 1e-6f, (uint32_t)(q8 ? HC * N / 32 * 34 : HC * N * 2),
                      (uint32_t)(q8 ? LR / 32 * 34 : LR * 2), comb ? 1u : 0u, (uint32_t)TOPK};
            const mtl::Buffer& WB = q8 ? Q8 : W;
            const size_t od = q8 ? q8off.at(tdn.name) : tdn.offset, oi = q8 ? q8off.at(tin.name) : tin.offset,
                         ou = q8 ? q8off.at(tup.name) : tup.offset;
            mtl::Dispatch d;
            d.p = Pn(lib, dn, dn + "#" + std::to_string(ntok), mtl::Constants().i(1, ntok));
            d.args = {mtl::Arg::B(WB, od), mtl::Arg::B(WB, oi), mtl::Arg::B(Rb[cur_r]), mtl::Arg::B(Rb[1 - cur_r]),
                      mtl::Arg::B(bo), mtl::Arg::B(Db[cur_d]), mtl::Arg::B(W, tn.offset), mtl::Arg::B(Db[1 - cur_d]),
                      mtl::Arg::B(rsb), mtl::Arg::V(a), mtl::Arg::B(plan), mtl::Arg::B(ye), mtl::Arg::B(ysh), mtl::Arg::B(sg)};
            d.grid[0] = (LR + HC + 15) / 16;
            d.grid[1] = HC;
            d.tg[0] = 256;
            const double wb = q8 ? (tdn.bytes + (final_ ? 0 : tin.bytes)) * 34.0 / 64.0 : (double)tdn.bytes + (final_ ? 0 : tin.bytes);
            push(d, "hc", wb);
            cur_r = 1 - cur_r;
            cur_d = 1 - cur_d;
            mtl::Dispatch u;
            u.p = Pn(lib, up, up + "#" + std::to_string(ntok), mtl::Constants().i(1, ntok));
            u.args = {mtl::Arg::B(WB, ou), mtl::Arg::B(Db[cur_d]), mtl::Arg::B(Rb[cur_r]), mtl::Arg::B(out.b), mtl::Arg::V(a),
                      mtl::Arg::B(W, tn.offset), mtl::Arg::B(rsb)};
            u.grid[0] = N / 16;
            u.tg[0] = 256;
            push(u, "hc", q8 ? tup.bytes * 34.0 / 64.0 : (double)tup.bytes);
            return;
        }
        const gguf::Tensor& tn = T(pre + "_norm.weight");
        const gguf::Tensor& tdn = T(pre + "_down.weight");
        const gguf::Tensor& tup = T(pre + "_up.weight");
        const gguf::Tensor& tin = final_ ? tdn : T(pre + "_inject.weight");
        GrArgs a{N, HC, LR, (uint32_t)ntok, apply ? 1u : 0u, 1e-6f};
        const bool q83 = hc_mode == "q8_3";
        const mtl::Buffer& WB = q83 ? Q8 : W;
        const size_t od = q83 ? q8off.at(tdn.name) : tdn.offset, oi = q83 ? q8off.at(tin.name) : tin.offset,
                     ou = q83 ? q8off.at(tup.name) : tup.offset;
        mtl::Dispatch d1;
        d1.p = pnorm;
        d1.args = {mtl::Arg::B(R), mtl::Arg::B(bo), mtl::Arg::B(inj[cur_inj]), mtl::Arg::B(W, tn.offset), mtl::Arg::B(xn),
                   mtl::Arg::V(a)};
        if (comb_pending) {
            // the MoE combine folded into this norm (gr_norm_comb; bitwise moe_combine_sh + gr_norm)
            const uint32_t tk = TOPK;
            d1.p = pnormc;
            d1.args = {mtl::Arg::B(R), mtl::Arg::B(ye), mtl::Arg::B(inj[cur_inj]), mtl::Arg::B(W, tn.offset), mtl::Arg::B(xn),
                       mtl::Arg::V(a), mtl::Arg::B(plan), mtl::Arg::B(ysh), mtl::Arg::B(sg), mtl::Arg::V(tk)};
            comb_pending = false;
        }
        d1.grid[0] = HC;
        d1.grid[1] = ntok;
        d1.tg[0] = 256;
        push(d1, "hc", 0);
        mtl::Dispatch d2;
        d2.p = q83 ? pdownq : pdown;
        d2.args = {mtl::Arg::B(WB, od), mtl::Arg::B(WB, oi), mtl::Arg::B(xn), mtl::Arg::B(lo),
                   mtl::Arg::B(inj[1 - cur_inj]), mtl::Arg::V(a)};
        d2.grid[0] = LR + HC;
        d2.tg[0] = 256;
        push(d2, "hc", (double)tdn.bytes + (final_ ? 0 : tin.bytes));
        mtl::Dispatch d3;
        d3.p = q83 ? pupq : pup;
        d3.args = {mtl::Arg::B(WB, ou), mtl::Arg::B(lo), mtl::Arg::B(xn), mtl::Arg::B(out.b), mtl::Arg::V(a)};
        d3.grid[0] = N / 16;
        d3.tg[0] = 256;
        push(d3, "hc", (double)tup.bytes);
        cur_inj = 1 - cur_inj;
        if (prefetch && !final_) {
            // the next read: blk.l.hc_attn -> blk.l.hc_ffn -> blk.(l+1).hc_attn ... -> output_hc
            std::string nx;
            const size_t at = pre.find(".hc_attn");
            if (at != std::string::npos) nx = pre.substr(0, at) + ".hc_ffn";
            else {
                const int l = std::atoi(pre.c_str() + 4);
                nx = l == 47 ? std::string("output_hc") : "blk." + std::to_string(l + 1) + ".hc_attn";
            }
            const bool fin = nx == "output_hc";
            for (const std::string& part : fin ? std::vector<std::string>{"_down", "_up"}
                                               : std::vector<std::string>{"_down", "_inject", "_up"}) {
                const gguf::Tensor& tt = T(nx + part + ".weight");
                if (q83) {
                    uint64_t ne = 1;
                    for (uint64_t dd : tt.shape) ne *= dd;
                    pending.push_back({&Q8, q8off.at(tt.name), (size_t)(ne / 32 * 34) / 16 * 16});
                } else {
                    pending.push_back({&W, tt.offset, (size_t)tt.bytes / 16 * 16});
                }
            }
        }
    };
    auto xprep = [&](const mtl::Buffer& x, const Act& out, int n, const std::string& c) {
        mtl::Dispatch d;
        d.p = pprep;
        const uint32_t nn = n, nt = MAXT;
        d.args = {mtl::Arg::B(x), mtl::Arg::B(out.b), mtl::Arg::V(nn), mtl::Arg::V(nt)};
        d.tg[0] = 256;
        d.grid[0] = (n / 16 * MAXT + 255) / 256;
        push(d, c, 0);
    };
    // Every prepared activation has planes of MAXT rows (Act::plane_bytes); the producers that take a token count
    // for the plane stride (gr_up, qsa_merge, swiglu_prep, xprep) are given MAXT there.

    int gi = 0, qi = 0;
    for (int l = 0; l < 48; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        // ---- mixer
        gr_read(b + "hc_attn", l > 0, XA, false);
        if (l % 4 != 3) {
            gemv2(T(b + "attn_qkv.weight"), XA, qkv, 10240, "gdn proj");
            nb = true;
            gemv2(T(b + "attn_gate.weight"), XA, z, 6144, "gdn proj");
            nb = true;
            if (albe) {
                // ssm_alpha + ssm_beta in one dispatch (gemv_cat; bitwise the two GEMVs)
                const gguf::Tensor &ta = T(b + "ssm_alpha.weight"), &tb = T(b + "ssm_beta.weight");
                const std::string fn = "gemvc_" + std::to_string(ta.type) + "_32_1_4";
                mtl::Dispatch d;
                d.p = Pn(lib, fn, fn + "#" + std::to_string(ntok), mtl::Constants().i(0, ntok));
                GemvArgs ga{(uint32_t)N, 96, (uint32_t)ntok, (uint32_t)ref::row_bytes(ta.type, N), (uint32_t)N, 48};
                const uint32_t sp = 48;
                d.args = {mtl::Arg::B(W, ta.offset), mtl::Arg::B(XA.b, XA.plane_bytes() * xkind(ta.type)),
                          mtl::Arg::B(XA.b, XA.plane_bytes() * 4), mtl::Arg::B(al), mtl::Arg::V(ga), mtl::Arg::B(W, tb.offset),
                          mtl::Arg::B(be), mtl::Arg::V(sp)};
                d.grid[0] = 96 / 4;
                d.tg[0] = 128;
                push(d, "gdn proj", (double)ta.bytes + tb.bytes);
            } else {
                gemv2(T(b + "ssm_alpha.weight"), XA, al, 48, "gdn proj");
                nb = true;
                gemv2(T(b + "ssm_beta.weight"), XA, be, 48, "gdn proj");
            }
            GdnArgs a{16, 48, 10240, 1, 1e-6f, fuse ? (uint32_t)(MAXT * 6144) : 0u};
            mtl::Dispatch d;
            d.p = pgdn;
            d.args = {mtl::Arg::B(gst[gi]), mtl::Arg::B(ghist[gi]), mtl::Arg::B(qkv), mtl::Arg::B(al), mtl::Arg::B(be),
                      mtl::Arg::B(z), mtl::Arg::B(W, T(b + "ssm_conv1d.weight").offset), mtl::Arg::B(W, T(b + "ssm_dt.bias").offset),
                      mtl::Arg::B(W, T(b + "ssm_a").offset), mtl::Arg::B(W, T(b + "ssm_norm.weight").offset), mtl::Arg::B(fuse ? XY.b : yg),
                      mtl::Arg::V(a), mtl::Arg::B(ghist2[gi])};
            d.grid[0] = 48;
            d.tg[0] = 512;
            push(d, "gdn step", 2.0 * 4 * 128 * 48 * 128);
            if (!fuse) xprep(yg, XY, 6144, "gdn proj");
            gemv2(T(b + "ssm_out.weight"), XY, bo, N, "gdn proj");
            ++gi;
        } else if (!skip_qsa) {
            gemv2(T(b + "attn_q.weight"), XA, qfull, 12288, "qsa proj");
            nb = true;
            gemv2(T(b + "attn_k.weight"), XA, kcur, 512, "qsa proj");
            nb = true;
            gemv2(T(b + "attn_v.weight"), XA, vcur, 512, "qsa proj");
            nb = true;
            gemv2(T(b + "indexer.q_proj.weight"), XA, iqr, 512, "qsa proj");
            nb = true;
            gemv2(T(b + "indexer.k_proj.weight"), XA, ikr, 128, "qsa proj");
            QsaArgs a{(uint32_t)ctx, 0, (uint32_t)MAXC, (uint32_t)ntok, 1e-6f};
            mtl::Dispatch d;
            d.p = qprep;
            d.args = {mtl::Arg::B(qfull), mtl::Arg::B(kcur), mtl::Arg::B(vcur), mtl::Arg::B(iqr), mtl::Arg::B(ikr),
                      mtl::Arg::B(W, T(b + "attn_q_norm.weight").offset), mtl::Arg::B(W, T(b + "attn_k_norm.weight").offset),
                      mtl::Arg::B(W, T(b + "indexer.q_norm.weight").offset), mtl::Arg::B(W, T(b + "indexer.k_norm.weight").offset),
                      mtl::Arg::B(cs), mtl::Arg::B(qcur), mtl::Arg::B(qidx), mtl::Arg::B(kq[qi]), mtl::Arg::B(vq[qi]),
                      mtl::Arg::B(ksc[qi]), mtl::Arg::B(vsc[qi]), mtl::Arg::B(tail[qi]), mtl::Arg::B(dead[qi]),
                      mtl::Arg::B(pooled[qi]), mtl::Arg::V(a)};
            d.grid[0] = 33;
            d.grid[1] = ntok;
            d.tg[0] = 256;
            push(d, "qsa attn", 0);
            const bool sel = !(fuse && ctx + ntok <= 2051);   // within the budget the selection is the identity
            mtl::Dispatch s;
            s.p = qscore;
            s.args = {mtl::Arg::B(qidx), mtl::Arg::B(pooled[qi]), mtl::Arg::B(dead[qi]), mtl::Arg::B(scores), mtl::Arg::V(a)};
            s.grid[0] = ((ctx + ntok) / 4 + 1 + 7) / 8;
            s.grid[1] = ntok;
            s.tg[0] = 256;
            if (sel) push(s, "qsa attn", 0);
            mtl::Dispatch k;
            k.p = qtopk;
            k.args = {mtl::Arg::B(scores), mtl::Arg::B(qids), mtl::Arg::V(a)};
            k.grid[0] = ntok;
            k.tg[0] = 1024;
            if (sel) push(k, "qsa attn", 0);
            mtl::Dispatch at;
            at.p = qattn;
            at.args = {mtl::Arg::B(qcur), mtl::Arg::B(kq[qi]), mtl::Arg::B(vq[qi]), mtl::Arg::B(ksc[qi]), mtl::Arg::B(vsc[qi]),
                       mtl::Arg::B(qids), mtl::Arg::B(part), mtl::Arg::V(a)};
            at.grid[0] = qv2 ? 65 : 33;
            at.grid[1] = qv2 ? 4 : 2;
            at.grid[2] = ntok;
            at.tg[0] = 256;
            push(at, "qsa attn", 2.0 * 2 * 512 * std::min(ctx, 2051));
            mtl::Dispatch m;
            m.p = qmerge;
            QsaArgs am = a;
            am.n_tok = MAXT;   // the plane stride (see above)
            m.args = {mtl::Arg::B(part), mtl::Arg::B(qfull), mtl::Arg::B(XQ.b), mtl::Arg::V(am)};
            m.grid[0] = 24;
            m.grid[1] = ntok;
            m.tg[0] = 256;
            push(m, "qsa attn", 0);
            gemv2(T(b + "attn_output.weight"), XQ, bo, N, "qsa proj");
            ++qi;
        }
        // ---- MoE
        gr_read(b + "hc_ffn", true, XF, false);
        if (!routef) {
            gemv2(T(b + "ffn_gate_inp.weight"), XF, logits, 512, "router");
            nb = true;
        }
        const bool pair = fuse && std::getenv("CHAIN_PAIR");   // measured slower (+0.11 ms): opt-in
        if (pair) {
            // the shared expert's gate + up + SwiGLU in one dispatch, h raw into XH's raw plane
            const gguf::Tensor &tgs = T(b + "ffn_gate_shexp.weight"), &tus = T(b + "ffn_up_shexp.weight");
            const std::string fn = "gemv_pair_" + std::to_string(tgs.type) + "_" + std::to_string(tus.type);
            mtl::Dispatch d;
            d.p = Pn(lib, fn, fn + "#" + std::to_string(ntok), mtl::Constants().i(0, ntok));
            GemvArgs ga{(uint32_t)N, (uint32_t)NFF, (uint32_t)ntok, (uint32_t)ref::row_bytes(tgs.type, N), (uint32_t)N,
                        (uint32_t)NFF};
            const uint32_t rbu = (uint32_t)ref::row_bytes(tus.type, N);
            d.args = {mtl::Arg::B(W, tgs.offset), mtl::Arg::B(XF.b, XF.plane_bytes() * xkind(tgs.type)),
                      mtl::Arg::B(W, tus.offset), mtl::Arg::B(XF.b, XF.plane_bytes() * xkind(tus.type)),
                      mtl::Arg::B(XF.b, XF.plane_bytes() * 4), mtl::Arg::B(XH.b), mtl::Arg::V(ga), mtl::Arg::V(rbu)};
            d.grid[0] = NFF / 4;
            d.tg[0] = 128;
            push(d, "shared", (double)tgs.bytes + tus.bytes);
        } else {
            gemv2(T(b + "ffn_gate_shexp.weight"), XF, gsh, NFF, "shared");
            nb = true;
            gemv2(T(b + "ffn_up_shexp.weight"), XF, ush, NFF, "shared");
        }
        nb = true;
        gemv2(T(b + "ffn_gate_inp_shexp.weight"), XF, sg, 1, "shared");
        {
            MoeArgs a{};
            const gguf::Tensor &tg_ = T(b + "ffn_gate_exps.weight"), &tu = T(b + "ffn_up_exps.weight"),
                               &td = T(b + "ffn_down_exps.weight");
            a.gate_off = tg_.offset;
            a.up_off = tu.offset;
            a.down_off = td.offset;
            a.gu_row = (uint32_t)ref::row_bytes(tg_.type, N);
            a.d_row = (uint32_t)ref::row_bytes(td.type, NFF);
            a.gu_stride = (uint64_t)a.gu_row * NFF;
            a.d_stride = (uint64_t)a.d_row * N;
            a.n_embd = N;
            a.n_ff = NFF;
            a.n_tok = ntok;
            a.topk = TOPK;
            a.n_expert = NEXP;
            // tiles (lanes per row, rows per lane, simdgroups): CHAIN_GU / CHAIN_DN / CHAIN_DN4 override
            const bool dq2 = td.type == ref::Q2_0;
            int gl = 32, gr = 1, gs = 4, dl = dq2 ? 8 : 4, dr = dq2 ? 1 : 2, ds_ = 4;
            if (const char* e = std::getenv("CHAIN_GU")) std::sscanf(e, "%d,%d,%d", &gl, &gr, &gs);
            if (const char* e = std::getenv(dq2 ? "CHAIN_DN" : "CHAIN_DN4")) std::sscanf(e, "%d,%d,%d", &dl, &dr, &ds_);
            a.gx_gu = NFF / (gs * (32 / gl) * gr);
            a.gx_dn = N / (ds_ * (32 / dl) * dr);
            if (routef) {
                // the router GEMV + moe_route in one dispatch (router_route; bitwise), concurrent with the shared GEMVs
                const gguf::Tensor& tr = T(b + "ffn_gate_inp.weight");
                mtl::Dispatch r;
                r.p = Pn(lib, "router_route_30", "router_route#" + std::to_string(ntok), mtl::Constants().i(0, ntok));
                GemvArgs ga{(uint32_t)N, 512, (uint32_t)ntok, (uint32_t)ref::row_bytes(tr.type, N), (uint32_t)N, 512};
                r.args = {mtl::Arg::B(W, tr.offset), mtl::Arg::B(XF.b, 0), mtl::Arg::B(XF.b, XF.plane_bytes() * 4),
                          mtl::Arg::B(logits), mtl::Arg::V(ga), mtl::Arg::B(ids), mtl::Arg::B(wts), mtl::Arg::B(plan),
                          mtl::Arg::V(a), mtl::Arg::B(rcounter)};
                r.grid[0] = 512 / 16;
                r.tg[0] = 512;
                nb = true;
                push(r, "router", (double)tr.bytes);
            } else {
                mtl::Dispatch r;
                r.p = proute;
                r.args = {mtl::Arg::B(logits), mtl::Arg::B(ids), mtl::Arg::B(wts), mtl::Arg::B(plan), mtl::Arg::V(a)};
                r.tg[0] = 512;
                push(r, "router", 0);
            }
            {
                GlueArgs ga{NFF, MAXT, N, TOPK};
                mtl::Dispatch d;
                d.p = pswi;
                d.args = {mtl::Arg::B(gsh), mtl::Arg::B(ush), mtl::Arg::B(XH.b), mtl::Arg::V(ga)};
                d.grid[0] = NFF / 16;
                d.grid[1] = ntok;
                d.tg[0] = 16;
                nb = true;
                if (!pair) push(d, "shared", 0);
                nb = false;
            }
            const std::string gun = "moe_gu_" + std::to_string(tg_.type) + "_" + std::to_string(gl) + "_" +
                                    std::to_string(gr) + "_" + std::to_string(gs);
            const std::string dnn = "moe_down_" + std::to_string(dq2 ? 142 : td.type) + "_" + std::to_string(dl) + "_" +
                                    std::to_string(dr) + "_" + std::to_string(ds_);
            mtl::Dispatch g;
            g.p = Pn(lib, gun, gun + (ctabf ? "#ct" : ""), mtl::Constants().b(23, ctabf));
            g.args = {mtl::Arg::B(W), mtl::Arg::B(XF.b, XF.plane_bytes() * xkind(tg_.type)), mtl::Arg::B(h), mtl::Arg::B(plan),
                      mtl::Arg::V(a), mtl::Arg::B(XF.b, XF.plane_bytes() * 4)};
            g.tg[0] = 32 * gs;
            g.indirect = &plan;
            g.ind_off = P_IGU * 4;
            push(g, "experts", 10.0 * ntok * 2 * a.gu_stride);
            nb = true;
            {
                const gguf::Tensor& tds = T(b + "ffn_down_shexp.weight");
                // with --fuse h is raw: a Q2_0 down reads it through the raw-activation decoder (type 142)
                gemv2(tds, XH, ysh, N, "shared", pair && tds.type == ref::Q2_0 ? 142 : -1);
            }
            mtl::Dispatch dd;
            dd.p = Pn(lib, dnn, dnn, {});
            dd.args = {mtl::Arg::B(W), mtl::Arg::B(h), mtl::Arg::B(ye), mtl::Arg::B(plan), mtl::Arg::V(a), mtl::Arg::B(h)};
            dd.tg[0] = 32 * ds_;
            dd.indirect = &plan;
            dd.ind_off = P_IDN * 4;
            push(dd, "experts", 10.0 * ntok * a.d_stride);
        }
        {
            GlueArgs a{N, (uint32_t)ntok, N, TOPK};
            mtl::Dispatch d;
            d.p = pcomb;
            d.args = {mtl::Arg::B(ye), mtl::Arg::B(plan), mtl::Arg::B(ysh), mtl::Arg::B(sg), mtl::Arg::B(bo), mtl::Arg::V(a)};
            d.grid[0] = (ntok * N + 255) / 256;
            d.tg[0] = 256;
            if (combf && (hc_mode == "3" || hc_mode == "q8_3")) comb_pending = true;   // folded into the next gr_norm
            else if (!(fold_comb && hc_mode != "3")) push(d, "shared", 0);   // else folded into the next HC read
        }
    }
    // ---- the final mixer and the head
    gr_read("output_hc", true, XA, true);
    gemv2(T("output.weight"), XA, lm, 248320, "lm head");
    {
        ArgArgs a{248320, 120};
        mtl::Dispatch d1, d2;
        d1.p = pam1;
        d1.args = {mtl::Arg::B(lm), mtl::Arg::B(apv), mtl::Arg::B(api), mtl::Arg::V(a)};
        d1.grid[0] = 120;
        d1.grid[1] = ntok;
        d1.tg[0] = 256;
        push(d1, "lm head", 0);
        d2.p = pam2;
        d2.args = {mtl::Arg::B(apv), mtl::Arg::B(api), mtl::Arg::B(tokout), mtl::Arg::V(a)};
        d2.grid[0] = ntok;
        d2.tg[0] = 256;
        push(d2, "lm head", 0);
    }
    // the producers' plane stride must be MAXT rows: gr_up's and swiglu's args carry n_tok for the stride, so the
    // gr_up launches above use a.n_tok = ntok - patch them to MAXT (the kernels loop over GR_NTOK tokens anyway)
    for (auto& d : ds)
        if (d.p == pup || d.p == pupq || d.p == pnorm || d.p == pdown || d.p == pdownq) {
            GrArgs a;
            std::memcpy(&a, d.args.back().bytes.data(), sizeof a);
            if (d.p == pup || d.p == pupq) a.n_tok = MAXT;
            std::memcpy(d.args.back().bytes.data(), &a, sizeof a);
        }

    double total_bytes = 0;
    for (auto& kv : bytes_by) total_bytes += kv.second;
    std::printf("%zu dispatches; weight + state bytes %.1f MB\n", ds.size(), total_bytes / 1e6);
    auto run = [&] { return graph ? cx.run_graph(ds) : cx.run(ds); };
    run();
    const double tw = mtl::now_s();
    while (mtl::now_s() - tw < 1.0) run();
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) t.push_back(run());
    std::printf("HC read: %s\n", hc_mode == "3" ? "BF16, 3 dispatches" : hc_mode == "2" ? "BF16, 2 dispatches" : hc_mode == "q8_3" ? "Q8_0, 3 dispatches" : "Q8_0, 2 dispatches");
    std::printf("encoding: %s\n", graph ? "concurrent encoder, barriers at dependencies" : "serial encoder");
    const double m = median(t);
    std::printf("decode step (%d token(s)): GPU %.3f ms  (min %.3f, max %.3f)  -> %.1f GB/s effective; %.1f tok/s if "
                "every token were a window of %d\n", ntok, m * 1e3, *std::min_element(t.begin(), t.end()) * 1e3,
                *std::max_element(t.begin(), t.end()) * 1e3, total_bytes / m / 1e9, ntok / m, ntok);
    for (auto& kv : bytes_by) std::printf("  %-10s %8.1f MB\n", kv.first.c_str(), kv.second / 1e6);
    return 0;
}
