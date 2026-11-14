// metal/tests/prompt_chain.cpp - a whole PROMPT through every layer on the REAL weights of a GGUF (mmapped, wrapped
// no-copy), in chunks of C tokens (one command buffer a chunk): the GPU time of prompt processing with the prompt
// kernels of metal/.  A timing of the real byte and FLOP stream, not a numerical check: the tokens are random ids
// (their real embedding rows start every hyper-connection stream), the parity programs check the kernels.
//
//   prompt_chain <shard-1.gguf> [--prompt P,P,..] [--chunk C] [--breakdown] [--only CLASS] [--skip CLASS]
//
// Per layer and chunk (T tokens; activations fp32 rows, fp16 copies as the matrix units' operands):
//   HC read     gr_norm_h (fp16 xn) -> GEMM [w_down; w_inject] -> gr_epi_down -> gemm_hc_up (the GEMM with the
//               mixing epilogue);  PC_HC=plain: gr_norm -> act_f16 -> GEMM -> gr_epi_down -> GEMM -> gr_epi_up
//   GDN         GEMMs qkv / gate / alpha / beta -> gdn_pre (parallel) -> gdn_scan_r (the state scan) -> gdn_post
//               (output norm and gate, fp16) -> GEMM out;  PC_GDN=step: the decode step over the T tokens instead
//   QSA         GEMMs q / k / v / indexer q, k -> qsa_prep (T tokens) -> score + top-k (past the budget) -> attn ->
//               merge -> GEMM out
//   MoE         GEMM router -> route_topk -> plan (hist, scan, clear, place) -> gather -> mm_gu2 -> mm_down2 ->
//               combine; shared expert GEMMs gate, up -> swiglu_h -> GEMM down; GEMM shared gate; moe_out
// Every dense matrix is dequantized to fp16 (dq_f16) once a chunk, before its GEMM.  After the last chunk: the final
// HC read and the LM head on the LAST row only (the decode kernels: gr_norm / gr_down / gr_up, GEMV, argmax).
// Left out: the PLE path and the MTP draft.
#include "coop_probe.h"
#include "gguf_lite.h"
#include "mtl_rt.h"
#include "ref_quant.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <sys/mman.h>
#include <vector>

namespace {

constexpr int N = 2560, HC = 4, LR = 320, NFF = 640, TOPK = 10, NEXP = 512, NV = 248320, QB = 2051, QCHUNKS = 33;
struct GrArgs { uint32_t n_embd, hc, hc_lr, n_tok, apply; float eps; };
struct EpiArgs { uint32_t n_tok, n_embd, hc, hc_lr; };
struct DqArgs { uint32_t n_in, n_out, row_bytes, perm; };
struct GemmArgs { int M, N, K, pad; };
struct GemvArgs { uint32_t n_in, n_out, n_tok, row_bytes, x_stride, y_stride; };
struct GdnArgs { uint32_t h_k, h_v, channels, commit; float eps; uint32_t plane; };
struct GdnPArgs { uint32_t n_tok, h_k, h_v, channels; float eps; uint32_t f32; };
struct QsaArgs { uint32_t pos0, pos_base, max_cells, n_tok; float eps; };
struct MmPlanArgs { uint32_t n_ent, topk, n_expert, tm, n_blocks; };
struct MmArgs { uint64_t gate_off, up_off, down_off, gu_stride, d_stride; uint32_t gu_row, d_row, n_embd, n_ff; };
struct ArgArgs { uint32_t n_vocab, parts; };

double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
bool fmt_unit32(int t) { return t == ref::IQ4_XS || t == ref::IQ4_NL; }   // mm_gu2 needs a K chunk >= 32
int unit_of(int t) { return t == ref::Q2_0 || t == ref::IQ3_S || t == ref::BF16 ? 16 : 32; }

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: prompt_chain <shard-1.gguf> [--prompt P,..] [--chunk C] [--breakdown]\n"); return 2; }
    std::vector<int> prompts = {1024, 4096, 32768};
    int C = 2048, reps = 2;
    bool breakdown = false;
    std::string only, skip;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--prompt") {
            prompts.clear();
            for (char* s = argv[++i]; *s;) {
                prompts.push_back((int)std::strtol(s, &s, 10));
                if (*s == ',') ++s;
            }
        } else if (a == "--chunk") C = std::atoi(argv[++i]);
        else if (a == "--reps") reps = std::atoi(argv[++i]);
        else if (a == "--breakdown") breakdown = true;
        else if (a == "--only") only = argv[++i];
        else if (a == "--skip") skip = argv[++i];
    }
    const int PMAX = *std::max_element(prompts.begin(), prompts.end());
    gguf::File f;
    std::string err;
    if (!f.open(argv[1], err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
    mtl::Context cx;
    std::printf("device %s; %s: %zu tensors, %.1f GB; chunk %d\n", cx.device_name().c_str(), argv[1], f.tensors.size(),
                f.file_bytes / 1e9, C);
    {
        const double t0 = mtl::now_s();
        madvise(f.map, f.map_bytes, MADV_WILLNEED);
        volatile uint64_t s = 0;
        for (size_t o = 0; o < f.file_bytes; o += 16384) s += ((const uint8_t*)f.map)[o];
        std::printf("file touched in %.1f s\n", mtl::now_s() - t0);
    }
    mtl::Buffer W = cx.wrap(f.map, f.map_bytes);
    auto T_ = [&](const std::string& name) -> const gguf::Tensor& {
        const gguf::Tensor* t = f.find(name);
        if (!t) { std::fprintf(stderr, "missing tensor %s\n", name.c_str()); std::exit(2); }
        return *t;
    };
    const double tc0 = mtl::now_s();
    const int lib = cx.compile({"quant.metal", "units.metal", "mma.metal", "dense_mm.metal", "moe_mm.metal", "gr.metal",
                                "gemv.metal", "prompt.metal"});
    const int lgdn = cx.compile({"gdn.metal", "gdn_prompt.metal"}, "#define NO_GGML_COMMON 1\n");
    const int lqsa = cx.compile({"qsa.metal", "qsa_prompt.metal"}, "#define NO_GGML_COMMON 1\n");
    const int lsmp = cx.compile({"sample.metal"}, "#define NO_GGML_COMMON 1\n");
    std::printf("kernels compiled in %.1f s\n", mtl::now_s() - tc0);
    std::map<std::string, mtl::Pipeline> pcache;
    const bool k4 = coop_k4(cx);
    std::printf("matmul2d right-input layout: %s\n", k4 ? "runs of 4 k (per-lane run decoders)" : "other (per-element decode)");
    auto P = [&](int l, const std::string& fn, const mtl::Constants& c = {}, const std::string& key = "") -> const mtl::Pipeline* {
        const std::string k = key.empty() ? fn : key;
        auto it = pcache.find(k);
        if (it == pcache.end()) it = pcache.emplace(k, cx.pipeline(l, fn, c)).first;
        return &it->second;
    };

    // ---- buffers (MT = the chunk's rows)
    const size_t MT = C;
    const int MAXC = (PMAX + 64 + 63) / 64 * 64, MB = MAXC / 4 + 2;
    const int TM = 16;
    const int maxrows = (int)MT * TOPK + NEXP * TM, nblk_max = ((int)MT * TOPK + 1023) / 1024;
    auto F32 = [&](size_t n) { return cx.alloc(4 * n); };
    auto F16 = [&](size_t n) { return cx.alloc(2 * n); };
    mtl::Buffer R = F32(MT * HC * N), bo = F32(MT * N), inj[2] = {F32(MT * HC), F32(MT * HC)}, xn = F32(MT * HC * N),
                xnh = F16(MT * HC * N), V = F32(MT * (LR + HC)), loh = F16(MT * LR), G = F32(MT * HC * N), mixed = F32(MT * N),
                mixh = F16(MT * N);
    mtl::Buffer qkv = F32(MT * 10240), z = F32(MT * 6144), al = F32(MT * 48), be = F32(MT * 48), yg = F32(MT * 6144),
                ygh = F16(MT * 6144), gqn = F32(MT * 16 * 128), gkn = F32(MT * 16 * 128), gvc = F32(MT * 6144),
                ggb = F32(MT * 48 * 2), gkq = F32(MT * 16), go = F32(MT * 6144);
    const char* hc_env = std::getenv("PC_HC");
    const std::string hc_mode = hc_env ? hc_env : "fused";   // fused | plain
    const char* qsa_env = std::getenv("PC_QSA");
    // flash (one pass on the matrix units, no partials) | mm (split-K on the matrix units + merge) | v1 (qsa_attn,
    // fp32 + merge).  The M5: flash (= mm + merge or better); the M4: v1 attention (faster there) with the GEMM score.
    const std::string qsa_mode = qsa_env ? qsa_env : (k4 ? "flash" : "v1");
    // the expert GEMMs: "v2" (threadgroup-staged weights) or "v3:TM,TN,TK,SG" (register operands, TN rows a simdgroup)
    const char* gu_env = std::getenv("PC_GU");
    const char* dn_env = std::getenv("PC_DN");
    // defaults (M5 Max, moe_mm_parity --v3 --bench): gate/up 16,32,32,4 (every format), down 16,64,32,2
    // without the M5's right-input layout (the M4) the per-element v3 decode is slower than v2: v2 there
    const std::string gu_mode = gu_env ? gu_env : (k4 ? "v3:auto" : "v2"), dn_mode = dn_env ? dn_env : (k4 ? "v3:16,64,32,2" : "v2");
    auto v3_tile = [](const std::string& m, int* t) { std::sscanf(m.c_str() + 3, "%d,%d,%d,%d", &t[0], &t[1], &t[2], &t[3]); };
    int gu3[4] = {16, 64, 16, 2}, dn3[4] = {16, 64, 16, 2};
    if (gu_mode.rfind("v3:", 0) == 0 && gu_mode != "v3:auto") v3_tile(gu_mode, gu3);
    else if (gu_mode == "v3:auto") { gu3[0] = 16; gu3[1] = 32; gu3[2] = 32; gu3[3] = 4; }
    auto tl3 = [](const int* t) { return std::to_string(t[0]) + "_" + std::to_string(t[1]) + "_" + std::to_string(t[2]) + "_" + std::to_string(t[3]); };
    auto gu3_for = [&](int type) -> std::string {
        (void)type;   // one tile for every format since the paired-run decoders (W8)
        return tl3(gu3);
    };
    if (dn_mode.rfind("v3:", 0) == 0) v3_tile(dn_mode, dn3);
    const bool gu_v3 = gu_mode.rfind("v3:", 0) == 0, dn_v3 = dn_mode.rfind("v3:", 0) == 0;
    const char* gdn_env = std::getenv("PC_GDN");
    const std::string gdn_mode = gdn_env ? gdn_env : "scanr_4";
    mtl::Buffer qfull = F32(MT * 12288), kcur = F32(MT * 512), vcur = F32(MT * 512), iqr = F32(MT * 512), ikr = F32(MT * 128),
                qcur = F32(MT * 24 * 256), qcurh = F16(MT * 24 * 256), qidx = F32(MT * 4 * 128), scores = F32(MT * MB), qids = cx.alloc(4 * MT * QB),
                part = F32(MT * QCHUNKS * 24 * 258), XQ = F32(4 * MT * 6144 + MT * 384), xqh = F16(MT * 6144);
    mtl::Buffer logits = F32(MT * NEXP), ids = cx.alloc(4 * MT * TOPK), wts = F32(MT * TOPK),
                bcnt = cx.alloc(4 * (size_t)nblk_max * NEXP), boff = cx.alloc(4 * (size_t)nblk_max * NEXP), eoff = cx.alloc(4 * NEXP),
                tiles = cx.alloc(8 * (size_t)(maxrows / TM + 1)), meta = cx.alloc(64), stok = cx.alloc(4 * (size_t)maxrows),
                sw = F32(maxrows), pos = cx.alloc(4 * MT * TOPK), XS = F16((size_t)maxrows * N), HS = F16((size_t)maxrows * NFF),
                YE = F32((size_t)maxrows * N), yr = F32(MT * N), gsh = F32(MT * NFF), ush = F32(MT * NFF), hh = F16(MT * NFF),
                ys = F32(MT * N), sgv = F32(MT);
    // fp16 weight scratch, one per role
    mtl::Buffer WD = F16((size_t)(LR + HC) * HC * N), WU = F16((size_t)HC * N * LR), Wqkv = F16((size_t)10240 * N),
                Wz = F16((size_t)6144 * N), Wal = F16((size_t)48 * N), Wbe = F16((size_t)48 * N), Wout = F16((size_t)N * 6144),
                Wq = F16((size_t)12288 * N), Wk = F16((size_t)512 * N), Wv = F16((size_t)512 * N), Wiq = F16((size_t)512 * N),
                Wik = F16((size_t)128 * N), Wr = F16((size_t)NEXP * N), Wsg = F16((size_t)N), Wshg = F16((size_t)NFF * N),
                Wshu = F16((size_t)NFF * N), Wshd = F16((size_t)N * NFF);
    // the last row's final read and head
    mtl::Buffer xn1 = F32(HC * N), lo1 = F32(LR), inj1 = F32(HC), X1 = F32(4 * N + N / 16), lm = F32(NV), apv = F32(120),
                api = cx.alloc(4 * 120), tokout = cx.alloc(16);
    mtl::Buffer cs = cx.alloc(8 * (size_t)(MAXC + 16) * 32);
    for (int p = 0; p < MAXC + 16; ++p)
        for (int i = 0; i < 32; ++i) {
            const double th = (double)p * std::pow(1e7, -2.0 * i / 64.0);
            cs.as<float>()[(p * 32 + i) * 2] = (float)std::cos(th);
            cs.as<float>()[(p * 32 + i) * 2 + 1] = (float)std::sin(th);
        }
    std::vector<mtl::Buffer> gst, ghist[2], kq, vq, ksc, vsc, tail, dead, pooled;
    for (int l = 0; l < 48; ++l) {
        if (l % 4 != 3) {
            gst.push_back(F32(128 * 48 * 128));
            ghist[0].push_back(F32(10240 * 3));
            ghist[1].push_back(F32(10240 * 3));
        } else {
            kq.push_back(cx.alloc((size_t)MAXC * 2 * 256));
            vq.push_back(cx.alloc((size_t)MAXC * 2 * 256));
            ksc.push_back(cx.alloc(2 * (size_t)MAXC * 2 * 4));
            vsc.push_back(cx.alloc(2 * (size_t)MAXC * 2 * 4));
            tail.push_back(F32(4 * 128));
            dead.push_back(F32(128));
            pooled.push_back(F32((size_t)MB * 128));
        }
    }
    std::vector<int> gpar(36, 0);
    // embeddings of random token ids (the real rows)
    std::vector<float> E(MT * N);
    {
        const gguf::Tensor& te = T_("token_embd.weight");
        const size_t rb = ref::row_bytes(te.type, N);
        ref::Rng r(17);
        for (size_t t = 0; t < MT; ++t) {
            const uint64_t id = r.next() % 200000;
            ref::dequant(te.type, (const uint8_t*)f.map + te.offset + id * rb, E.data() + t * N, N);
        }
    }
    meta.as<uint32_t>()[2] = gu_v3 ? NFF / (gu3[1] * gu3[3]) : NFF / 64;
    meta.as<uint32_t>()[5] = dn_v3 ? N / (dn3[1] * dn3[3]) : N / 64;

    // ---- the chunk's dispatches
    std::vector<mtl::Dispatch> ds;
    std::vector<std::string> cls;
    auto push = [&](const mtl::Dispatch& d, const std::string& c) {
        if (!only.empty() && c != only) return;
        if (!skip.empty() && c == skip) return;
        ds.push_back(d);
        cls.push_back(c);
    };
    auto D = [&](const mtl::Pipeline* p, std::vector<mtl::Arg> a, uint32_t gx, uint32_t gy, uint32_t tg, const std::string& c,
                 uint32_t gz = 1) {
        mtl::Dispatch d;
        d.p = p;
        d.args = std::move(a);
        d.grid[0] = gx;
        d.grid[1] = gy;
        d.grid[2] = gz;
        d.tg[0] = tg;
        push(d, c);
    };
    const char* gk_env = std::getenv("PC_GEMM");
    auto gemm = [&](const mtl::Buffer& A, const mtl::Buffer& B, const mtl::Buffer& Y, int M, int Nn, int K, const std::string& c) {
        int tm = 64, tn = Nn <= 64 ? 32 : 64;
        if (gk_env && Nn > 64) std::sscanf(gk_env, "%d,%d", &tm, &tn);
        const std::string fn = "gemm_f16_" + std::to_string(tm) + "_" + std::to_string(tn) + "_4";
        GemmArgs g{M, Nn, K, 0};
        D(P(lib, fn), {mtl::Arg::B(A), mtl::Arg::B(B), mtl::Arg::B(Y), mtl::Arg::V(g)}, (Nn + tn - 1) / tn, (M + tm - 1) / tm,
          128, c);
    };
    auto dq = [&](const gguf::Tensor& t, const mtl::Buffer& out, size_t out_off = 0, uint32_t perm = 0) {
        const int n_in = (int)t.shape[0], n_out = t.shape.size() > 1 ? (int)t.shape[1] : 1;
        DqArgs a{(uint32_t)n_in, (uint32_t)n_out, (uint32_t)ref::row_bytes(t.type, n_in), perm};
        D(P(lib, "dq_f16_" + std::to_string(t.type)), {mtl::Arg::B(W, t.offset), mtl::Arg::B(out, out_off), mtl::Arg::V(a)},
          (n_in / unit_of(t.type) + 63) / 64, n_out, 64, "dq");
    };
    auto act16 = [&](const mtl::Buffer& x, const mtl::Buffer& xh, size_t n, const std::string& c) {
        const uint32_t nn = (uint32_t)n;
        D(P(lib, "act_f16"), {mtl::Arg::B(x), mtl::Arg::B(xh), mtl::Arg::V(nn)}, (nn + 255) / 256, 1, 256, c);
    };
    int cur_inj = 0;
    // the HC read of T rows: mixed (fp32) and mixh (fp16)
    auto hc_read = [&](const std::string& pre, bool apply, int T) {
        const gguf::Tensor &tn = T_(pre + "_norm.weight"), &tdn = T_(pre + "_down.weight"), &tin = T_(pre + "_inject.weight"),
                           &tup = T_(pre + "_up.weight");
        dq(tdn, WD);
        dq(tin, WD, (size_t)2 * LR * HC * N);
        GrArgs a{N, HC, LR, (uint32_t)T, apply ? 1u : 0u, 1e-6f};
        EpiArgs ea{(uint32_t)T, N, HC, LR};
        if (hc_mode == "fused") {
            // fp16 xn only; the up GEMM's epilogue mixes (w_up rows interleaved by the dequantization)
            dq(tup, WU, 0, HC);
            D(P(lib, "gr_norm_h"), {mtl::Arg::B(R), mtl::Arg::B(bo), mtl::Arg::B(inj[cur_inj]), mtl::Arg::B(W, tn.offset),
                                    mtl::Arg::B(xnh), mtl::Arg::V(a)}, HC, T, 256, "hc");
            gemm(xnh, WD, V, T, LR + HC, HC * N, "hc gemm");
            D(P(lib, "gr_epi_down"), {mtl::Arg::B(V), mtl::Arg::B(loh), mtl::Arg::B(inj[1 - cur_inj]), mtl::Arg::V(ea)},
              (T * (LR + HC) + 255) / 256, 1, 256, "hc");
            GemmArgs g{T, HC * N, LR, 0};
            D(P(lib, "gemm_hc_up_64_64_4"), {mtl::Arg::B(loh), mtl::Arg::B(WU), mtl::Arg::B(xnh), mtl::Arg::B(mixed),
                                             mtl::Arg::B(mixh), mtl::Arg::V(g), mtl::Arg::V(ea)},
              HC * N / 64, (T + 63) / 64, 128, "hc gemm");
            cur_inj = 1 - cur_inj;
            return;
        }
        dq(tup, WU);
        D(P(lib, "gr_norm"), {mtl::Arg::B(R), mtl::Arg::B(bo), mtl::Arg::B(inj[cur_inj]), mtl::Arg::B(W, tn.offset),
                              mtl::Arg::B(xn), mtl::Arg::V(a)}, HC, T, 256, "hc");
        act16(xn, xnh, (size_t)T * HC * N, "hc");
        gemm(xnh, WD, V, T, LR + HC, HC * N, "hc gemm");
        D(P(lib, "gr_epi_down"), {mtl::Arg::B(V), mtl::Arg::B(loh), mtl::Arg::B(inj[1 - cur_inj]), mtl::Arg::V(ea)},
          (T * (LR + HC) + 255) / 256, 1, 256, "hc");
        gemm(loh, WU, G, T, HC * N, LR, "hc gemm");
        D(P(lib, "gr_epi_up"), {mtl::Arg::B(G), mtl::Arg::B(xn), mtl::Arg::B(mixed), mtl::Arg::B(mixh), mtl::Arg::V(ea)},
          (T * N + 255) / 256, 1, 256, "hc");
        cur_inj = 1 - cur_inj;
    };
    auto build = [&](int pos0, int T, bool head) {
        ds.clear();
        cls.clear();
        int gi = 0, qi = 0;
        for (int l = 0; l < 48; ++l) {
            const std::string b = "blk." + std::to_string(l) + ".";
            hc_read(b + "hc_attn", l > 0, T);
            if (l % 4 != 3) {
                const gguf::Tensor &tq = T_(b + "attn_qkv.weight"), &tz = T_(b + "attn_gate.weight"),
                                   &ta = T_(b + "ssm_alpha.weight"), &tb = T_(b + "ssm_beta.weight"), &to = T_(b + "ssm_out.weight");
                dq(tq, Wqkv);
                dq(tz, Wz);
                dq(ta, Wal);
                dq(tb, Wbe);
                dq(to, Wout);
                gemm(mixh, Wqkv, qkv, T, 10240, N, "gdn proj");
                gemm(mixh, Wz, z, T, 6144, N, "gdn proj");
                gemm(mixh, Wal, al, T, 48, N, "gdn proj");
                gemm(mixh, Wbe, be, T, 48, N, "gdn proj");
                GdnArgs a{16, 48, 10240, 1, 1e-6f, 0};
                const int par = gpar[gi];
                if (gdn_mode != "step") {
                    GdnPArgs ga{(uint32_t)T, 16, 48, 10240, 1e-6f, 0};
                    D(P(lgdn, "gdn_pre"),
                      {mtl::Arg::B(qkv), mtl::Arg::B(ghist[par][gi]), mtl::Arg::B(W, T_(b + "ssm_conv1d.weight").offset),
                       mtl::Arg::B(al), mtl::Arg::B(be), mtl::Arg::B(W, T_(b + "ssm_dt.bias").offset),
                       mtl::Arg::B(W, T_(b + "ssm_a").offset), mtl::Arg::B(gqn), mtl::Arg::B(gkn), mtl::Arg::B(gvc), mtl::Arg::B(ggb),
                       mtl::Arg::B(gkq), mtl::Arg::B(ghist[1 - par][gi]), mtl::Arg::V(ga)},
                      16 + 24, T, 256, "gdn scan");
                    const int ib = gdn_mode == "scan_v" ? 8 : std::atoi(gdn_mode.c_str() + 6);
                    D(P(lgdn, "gdn_" + gdn_mode, gdn_mode == "scan_v" ? mtl::Constants() : mtl::Constants().i(6, 0)),
                      {mtl::Arg::B(gst[gi]), mtl::Arg::B(gqn), mtl::Arg::B(gkn), mtl::Arg::B(gvc), mtl::Arg::B(ggb), mtl::Arg::B(gkq),
                       mtl::Arg::B(go), mtl::Arg::V(ga)},
                      48 * 4, 1, 32 * ib, "gdn scan");
                    D(P(lgdn, "gdn_post"), {mtl::Arg::B(go), mtl::Arg::B(z), mtl::Arg::B(W, T_(b + "ssm_norm.weight").offset),
                                            mtl::Arg::B(ygh), mtl::Arg::B(yg), mtl::Arg::V(ga)},
                      48, T, 128, "gdn scan");
                    gemm(ygh, Wout, bo, T, N, 6144, "gdn proj");
                    ++gi;
                    goto moe;
                }
                D(P(lgdn, "gdn_step", mtl::Constants().i(2, T), "gdn_step#" + std::to_string(T)),
                  {mtl::Arg::B(gst[gi]), mtl::Arg::B(ghist[par][gi]), mtl::Arg::B(qkv), mtl::Arg::B(al), mtl::Arg::B(be),
                   mtl::Arg::B(z), mtl::Arg::B(W, T_(b + "ssm_conv1d.weight").offset), mtl::Arg::B(W, T_(b + "ssm_dt.bias").offset),
                   mtl::Arg::B(W, T_(b + "ssm_a").offset), mtl::Arg::B(W, T_(b + "ssm_norm.weight").offset), mtl::Arg::B(yg),
                   mtl::Arg::V(a), mtl::Arg::B(ghist[1 - par][gi])},
                  48, 1, 512, "gdn scan");
                act16(yg, ygh, (size_t)T * 6144, "gdn proj");
                gemm(ygh, Wout, bo, T, N, 6144, "gdn proj");
                ++gi;
            } else {
                const gguf::Tensor &tq = T_(b + "attn_q.weight"), &tk = T_(b + "attn_k.weight"), &tv = T_(b + "attn_v.weight"),
                                   &tiq = T_(b + "indexer.q_proj.weight"), &tik = T_(b + "indexer.k_proj.weight"),
                                   &to = T_(b + "attn_output.weight");
                dq(tq, Wq);
                dq(tk, Wk);
                dq(tv, Wv);
                dq(tiq, Wiq);
                dq(tik, Wik);
                dq(to, Wout);
                gemm(mixh, Wq, qfull, T, 12288, N, "qsa proj");
                gemm(mixh, Wk, kcur, T, 512, N, "qsa proj");
                gemm(mixh, Wv, vcur, T, 512, N, "qsa proj");
                gemm(mixh, Wiq, iqr, T, 512, N, "qsa proj");
                gemm(mixh, Wik, ikr, T, 128, N, "qsa proj");
                QsaArgs a{(uint32_t)pos0, 0, (uint32_t)MAXC, (uint32_t)T, 1e-6f};
                D(P(lqsa, "qsa_prep"),
                  {mtl::Arg::B(qfull), mtl::Arg::B(kcur), mtl::Arg::B(vcur), mtl::Arg::B(iqr), mtl::Arg::B(ikr),
                   mtl::Arg::B(W, T_(b + "attn_q_norm.weight").offset), mtl::Arg::B(W, T_(b + "attn_k_norm.weight").offset),
                   mtl::Arg::B(W, T_(b + "indexer.q_norm.weight").offset), mtl::Arg::B(W, T_(b + "indexer.k_norm.weight").offset),
                   mtl::Arg::B(cs), mtl::Arg::B(qcur), mtl::Arg::B(qidx), mtl::Arg::B(kq[qi]), mtl::Arg::B(vq[qi]),
                   mtl::Arg::B(ksc[qi]), mtl::Arg::B(vsc[qi]), mtl::Arg::B(tail[qi]), mtl::Arg::B(dead[qi]),
                   mtl::Arg::B(pooled[qi]), mtl::Arg::V(a)},
                  33, T, 256, "qsa prep");
                if (pos0 + T > QB) {   // some token's context is past the budget: select
                    if (!std::getenv("PC_QSA_SCORE"))   // the GEMM score unless PC_QSA_SCORE (= the scalar qsa_score)
                        D(P(lqsa, "qsa_score_mm"), {mtl::Arg::B(qidx), mtl::Arg::B(pooled[qi]), mtl::Arg::B(dead[qi]),
                                                    mtl::Arg::B(scores), mtl::Arg::V(a)},
                          ((pos0 + T) / 4 + 1 + 63) / 64, (T + 15) / 16, 128, "qsa score");
                    else
                    D(P(lqsa, "qsa_score"), {mtl::Arg::B(qidx), mtl::Arg::B(pooled[qi]), mtl::Arg::B(dead[qi]), mtl::Arg::B(scores),
                                             mtl::Arg::V(a)},
                      ((pos0 + T) / 4 + 1 + 7) / 8, T, 256, "qsa score");
                    D(P(lqsa, "qsa_topk"), {mtl::Arg::B(scores), mtl::Arg::B(qids), mtl::Arg::V(a)}, T, 1, 1024, "qsa sel");
                } else {
                    // within the budget the selection is the identity: qsa_topk only writes 0..n_kv-1
                    D(P(lqsa, "qsa_topk"), {mtl::Arg::B(scores), mtl::Arg::B(qids), mtl::Arg::V(a)}, T, 1, 1024, "qsa sel");
                }
                if (qsa_mode == "flash") {
                    const uint32_t nq = (uint32_t)T * 24 * 256;
                    D(P(lqsa, "qsa_q_f16"), {mtl::Arg::B(qcur), mtl::Arg::B(qcurh), mtl::Arg::V(nq)}, (nq + 255) / 256, 1, 256, "qsa attn");
                    D(P(lqsa, "qsa_flash", mtl::Constants().b(7, k4)),
                      {mtl::Arg::B(qcurh), mtl::Arg::B(kq[qi]), mtl::Arg::B(vq[qi]), mtl::Arg::B(ksc[qi]), mtl::Arg::B(vsc[qi]),
                       mtl::Arg::B(qids), mtl::Arg::B(qfull), mtl::Arg::B(xqh), mtl::Arg::V(a)},
                      2, T, 128, "qsa attn");
                    gemm(xqh, Wout, bo, T, N, 6144, "qsa proj");
                    ++qi;
                    goto moe;
                }
                if (qsa_mode == "mm") {
                    const uint32_t nq = (uint32_t)T * 24 * 256;
                    D(P(lqsa, "qsa_q_f16"), {mtl::Arg::B(qcur), mtl::Arg::B(qcurh), mtl::Arg::V(nq)}, (nq + 255) / 256, 1, 256, "qsa attn");
                    D(P(lqsa, "qsa_attn_mm", mtl::Constants().b(7, k4)), {mtl::Arg::B(qcurh), mtl::Arg::B(kq[qi]), mtl::Arg::B(vq[qi]), mtl::Arg::B(ksc[qi]),
                                               mtl::Arg::B(vsc[qi]), mtl::Arg::B(qids), mtl::Arg::B(part), mtl::Arg::V(a)},
                      QCHUNKS, 2, 32, "qsa attn", T);
                } else
                D(P(lqsa, "qsa_attn"), {mtl::Arg::B(qcur), mtl::Arg::B(kq[qi]), mtl::Arg::B(vq[qi]), mtl::Arg::B(ksc[qi]),
                                        mtl::Arg::B(vsc[qi]), mtl::Arg::B(qids), mtl::Arg::B(part), mtl::Arg::V(a)},
                  QCHUNKS, 2, 256, "qsa attn", T);
                D(P(lqsa, "qsa_merge"), {mtl::Arg::B(part), mtl::Arg::B(qfull), mtl::Arg::B(XQ), mtl::Arg::V(a)}, 24, T, 256,
                  "qsa merge");
                act16(XQ, xqh, (size_t)T * 6144, "qsa proj");
                gemm(xqh, Wout, bo, T, N, 6144, "qsa proj");
                ++qi;
            }
        moe:
            // ---- MoE
            hc_read(b + "hc_ffn", true, T);
            const gguf::Tensor &tr = T_(b + "ffn_gate_inp.weight"), &tsg = T_(b + "ffn_gate_inp_shexp.weight"),
                               &tg_ = T_(b + "ffn_gate_shexp.weight"), &tu_ = T_(b + "ffn_up_shexp.weight"),
                               &td_ = T_(b + "ffn_down_shexp.weight"), &eg = T_(b + "ffn_gate_exps.weight"),
                               &eu = T_(b + "ffn_up_exps.weight"), &ed = T_(b + "ffn_down_exps.weight");
            dq(tr, Wr);
            dq(tsg, Wsg);
            dq(tg_, Wshg);
            dq(tu_, Wshu);
            dq(td_, Wshd);
            gemm(mixh, Wr, logits, T, NEXP, N, "router");
            const uint32_t nt = T;
            D(P(lib, "route_topk"), {mtl::Arg::B(logits), mtl::Arg::B(ids), mtl::Arg::B(wts), mtl::Arg::V(nt)}, (T + 7) / 8, 1,
              256, "router");
            const int nent = T * TOPK, nblk = (nent + 1023) / 1024;
            MmPlanArgs pa{(uint32_t)nent, TOPK, NEXP, (uint32_t)TM, (uint32_t)nblk};
            const uint32_t nclr = maxrows;
            D(P(lib, "mm_clear"), {mtl::Arg::B(stok), mtl::Arg::V(nclr)}, (maxrows + 255) / 256, 1, 256, "experts");
            D(P(lib, "mm_hist"), {mtl::Arg::B(ids), mtl::Arg::B(bcnt), mtl::Arg::V(pa)}, nblk, 1, 1024, "experts");
            D(P(lib, "mm_scan"), {mtl::Arg::B(bcnt), mtl::Arg::B(boff), mtl::Arg::B(eoff), mtl::Arg::B(tiles), mtl::Arg::B(meta),
                                  mtl::Arg::V(pa)}, 1, 1, NEXP, "experts");
            D(P(lib, "mm_place"), {mtl::Arg::B(ids), mtl::Arg::B(wts), mtl::Arg::B(boff), mtl::Arg::B(eoff), mtl::Arg::B(stok),
                                   mtl::Arg::B(sw), mtl::Arg::B(pos), mtl::Arg::V(pa)}, nblk, 1, 1024, "experts");
            MmArgs ma{eg.offset, eu.offset, ed.offset, 0, 0, (uint32_t)ref::row_bytes(eg.type, N),
                      (uint32_t)ref::row_bytes(ed.type, NFF), N, NFF};
            ma.gu_stride = (uint64_t)ma.gu_row * NFF;
            ma.d_stride = (uint64_t)ma.d_row * N;
            const uint32_t ne = N;
            mtl::Dispatch g;
            g.p = P(lib, "mm_gather");
            g.args = {mtl::Arg::B(mixed), mtl::Arg::B(stok), mtl::Arg::B(XS), mtl::Arg::V(ne)};
            g.tg[0] = 256;
            g.indirect = &meta;
            g.ind_off = 8 * 4;
            push(g, "experts");
            mtl::Dispatch gu;
            gu.p = gu_v3 ? P(lib, "mm_gu3_" + std::to_string(eg.type) + "_" + gu3_for(eg.type), mtl::Constants().b(7, k4))
                         : P(lib, "mm_gu2_" + std::to_string(eg.type) + (fmt_unit32(eg.type) ? "_16_64_32_2" : "_16_64_16_2"),
                             mtl::Constants().i(5, 0));
            gu.args = {mtl::Arg::B(W), mtl::Arg::B(XS), mtl::Arg::B(HS), mtl::Arg::B(tiles), mtl::Arg::V(ma)};
            gu.tg[0] = gu_v3 ? 32 * gu3[3] : 64;
            gu.indirect = &meta;
            gu.ind_off = 2 * 4;
            push(gu, "experts gu");
            mtl::Dispatch dn;
            dn.p = dn_v3 ? P(lib, "mm_down3_" + std::to_string(ed.type) + "_" + tl3(dn3), mtl::Constants().b(7, k4))
                         : P(lib, "mm_down2_" + std::to_string(ed.type) + (ed.type == ref::IQ4_NL ? "_16_64_32_2" : "_16_64_16_2"));
            dn.args = {mtl::Arg::B(W), mtl::Arg::B(HS), mtl::Arg::B(YE), mtl::Arg::B(tiles), mtl::Arg::V(ma)};
            dn.tg[0] = dn_v3 ? 32 * dn3[3] : 64;
            dn.indirect = &meta;
            dn.ind_off = 5 * 4;
            push(dn, "experts dn");
            const uint32_t ca[4] = {(uint32_t)T, N, TOPK, 0};
            D(P(lib, "mm_combine"), {mtl::Arg::B(YE), mtl::Arg::B(pos), mtl::Arg::B(sw), mtl::Arg::B(yr), mtl::Arg::V(ca)},
              (T * N + 255) / 256, 1, 256, "experts");
            gemm(mixh, Wshg, gsh, T, NFF, N, "shared");
            gemm(mixh, Wshu, ush, T, NFF, N, "shared");
            const uint32_t nh = T * NFF;
            D(P(lib, "swiglu_h"), {mtl::Arg::B(gsh), mtl::Arg::B(ush), mtl::Arg::B(hh), mtl::Arg::V(nh)}, (nh + 255) / 256, 1, 256,
              "shared");
            gemm(hh, Wshd, ys, T, N, NFF, "shared");
            gemm(mixh, Wsg, sgv, T, 1, N, "shared");
            const uint32_t oa[2] = {(uint32_t)T, N};
            D(P(lib, "moe_out"), {mtl::Arg::B(yr), mtl::Arg::B(ys), mtl::Arg::B(sgv), mtl::Arg::B(bo), mtl::Arg::V(oa)},
              (T * N + 255) / 256, 1, 256, "shared");
        }
        if (head) {
            // the final read of the last row (decode kernels, 1 token) and the LM head
            const size_t lr = (size_t)T - 1;
            const gguf::Tensor &tn = T_("output_hc_norm.weight"), &tdn = T_("output_hc_down.weight"), &tup = T_("output_hc_up.weight");
            GrArgs a{N, HC, LR, 1, 1, 1e-6f};
            D(P(lib, "gr_norm"), {mtl::Arg::B(R, lr * HC * N * 4), mtl::Arg::B(bo, lr * N * 4), mtl::Arg::B(inj[cur_inj], lr * HC * 4),
                                  mtl::Arg::B(W, tn.offset), mtl::Arg::B(xn1), mtl::Arg::V(a)}, HC, 1, 256, "head");
            D(P(lib, "gr_down_8_8", mtl::Constants().i(1, 1), "gr_down_8_8#1"),
              {mtl::Arg::B(W, tdn.offset), mtl::Arg::B(W, tdn.offset), mtl::Arg::B(xn1), mtl::Arg::B(lo1), mtl::Arg::B(inj1),
               mtl::Arg::V(a)}, LR + HC, 1, 256, "head");
            D(P(lib, "gr_up", mtl::Constants().i(1, 1), "gr_up#1"),
              {mtl::Arg::B(W, tup.offset), mtl::Arg::B(lo1), mtl::Arg::B(xn1), mtl::Arg::B(X1), mtl::Arg::V(a)}, N / 16, 1, 256, "head");
            const gguf::Tensor& to = T_("output.weight");
            GemvArgs ga{N, NV, 1, (uint32_t)ref::row_bytes(to.type, N), N, NV};
            const std::string fn = "gemv_" + std::to_string(to.type) + "_16_1_4_1";
            const int xk = to.type == ref::Q4_K || to.type == ref::Q5_K ? 2 : (to.type == ref::Q6_K || to.type == ref::Q3_K) ? 3 : 0;
            D(P(lib, fn, mtl::Constants().i(0, 1), fn + "#1"),
              {mtl::Arg::B(W, to.offset), mtl::Arg::B(X1, (size_t)4 * N * xk), mtl::Arg::B(X1, (size_t)4 * N * 4), mtl::Arg::B(lm),
               mtl::Arg::V(ga)}, (NV + 7) / 8, 1, 128, "head");
            ArgArgs aa{NV, 120};
            D(P(lsmp, "argmax_part"), {mtl::Arg::B(lm), mtl::Arg::B(apv), mtl::Arg::B(api), mtl::Arg::V(aa)}, 120, 1, 256, "head");
            D(P(lsmp, "argmax_final"), {mtl::Arg::B(apv), mtl::Arg::B(api), mtl::Arg::B(tokout), mtl::Arg::V(aa)}, 1, 1, 256, "head");
        }
    };
    // a chunk: the embedding rows into every stream, then the chunk's command buffer; GDN history ping-pong flipped
    auto run_chunk = [&](int pos0, int T, bool head) {
        for (int t = 0; t < T; ++t)
            for (int c = 0; c < HC; ++c) std::memcpy(R.as<float>() + ((size_t)t * HC + c) * N, E.data() + (size_t)t * N, 4 * N);
        cur_inj = 0;
        build(pos0, T, head);
        const double g = cx.run(ds);
        for (auto& p : gpar) p ^= 1;
        return g;
    };
    auto prompt = [&](int Pn, double* first_chunk) {
        double tot = 0;
        for (int p0 = 0; p0 < Pn; p0 += C) {
            const int T = std::min(C, Pn - p0);
            const double g = run_chunk(p0, T, p0 + T >= Pn);
            if (p0 == 0 && first_chunk) *first_chunk = g;
            tot += g;
        }
        return tot;
    };
    std::set<std::string> all_cls;
    // warm: the first chunk until the clocks are up
    {
        const double t0 = mtl::now_s();
        int n = 0;
        while (mtl::now_s() - t0 < 1.5) { run_chunk(0, std::min(C, prompts[0]), false); ++n; }
        std::printf("warm-up: %d chunks; %zu dispatches a chunk\n", n, ds.size());
        all_cls.insert(cls.begin(), cls.end());
    }
    if (breakdown) {
        // each class alone on the first chunk (pos0 0) and on a chunk at pos0 = PMAX - C (QSA past the budget)
        if (PMAX > C) {   // the classes of a chunk past the budget (score) too
            run_chunk(std::max(0, PMAX - C), C, false);
            all_cls.insert(cls.begin(), cls.end());
        }
        for (int p0 : {0, std::max(0, PMAX - C)}) {
            const std::string keep_only = only;
            std::map<std::string, double> by;
            double all = 0;
            std::vector<std::string> names = {""};
            for (const auto& c : all_cls) names.push_back(c);
            for (const std::string& c : names) {
                only = c;
                std::vector<double> t;
                for (int i = 0; i < 3; ++i) t.push_back(run_chunk(p0, C, false));
                if (c.empty()) all = median(t);
                else by[c] = median(t);
            }
            only = keep_only;
            double sum = 0;
            for (auto& kv : by) sum += kv.second;
            std::printf("chunk of %d at pos %d: all %.2f ms (%.0f tok/s); classes alone (sum %.2f ms):\n", C, p0, all * 1e3, C / all,
                        sum * 1e3);
            for (auto& kv : by) std::printf("  %-10s %8.2f ms  %6.2f us a token\n", kv.first.c_str(), kv.second * 1e3, kv.second / C * 1e6);
        }
    }
    for (int Pn : prompts) {
        std::vector<double> t;
        double fc = 0;
        for (int r = 0; r < reps; ++r) t.push_back(prompt(Pn, &fc));
        const double m = median(t);
        std::printf("prompt %6d tokens (chunks of %d): GPU %8.1f ms -> %7.1f tok/s  (first chunk %.1f ms)\n", Pn, C, m * 1e3,
                    Pn / m, fc * 1e3);
    }
    return 0;
}
