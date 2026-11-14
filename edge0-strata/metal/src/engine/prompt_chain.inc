// metal/src/engine/prompt_chain.inc (included by window.cpp) - the prompt through mac-kern's prompt kernels
// (metal/tests/prompt_chain.cpp's chain, 6f6c555b), into the decode window's own state: chunks of C rows, every
// dense matrix dequantized to fp16 and multiplied on the matrix units, the routed experts as grouped GEMMs, the GDN
// state by the parallel prompt scan, QSA K/V and indexer keys into the window's caches.  Added for the engine:
// the PLE block after layer 0 (GEMMs + msl/ple.metal), every row's final residual (the drafter's prompt rows), the
// window's history parity, RoPE table and pending fold.  No head: the last prompt token runs as a decode window.
//
// The 8-row-window prompt path stays as the fallback: STRATA_METAL_PROMPT=window, and prompts shorter than
// STRATA_METAL_PROMPT_MIN rows (default 32).  STRATA_METAL_PROMPT_CHUNK = C (default 2048).  The kernel variants
// follow prompt_chain's defaults: matrix-unit register operands where the GPU has the M5's right-input layout
// (COOP_K4, function constant 7), else the older expert and attention kernels (the M4).

namespace {
constexpr int PC_QB = 2051, PC_QCHUNKS = 33, PC_STAGE_GU_MIN = 2048;
struct PcDqArgs { uint32_t n_in, n_out, row_bytes, perm; };
struct PcGemmArgs { int M, N, K, pad; };
struct PcEpiArgs { uint32_t n_tok, n_embd, hc, hc_lr; };
struct PcGdnPArgs { uint32_t n_tok, h_k, h_v, channels; float eps; uint32_t f32; };
struct PcMmPlanArgs { uint32_t n_ent, topk, n_expert, tm, n_blocks; };
struct PcMmArgs { uint64_t gate_off, up_off, down_off, gu_stride, d_stride; uint32_t gu_row, d_row, n_embd, n_ff; };
struct PcLatentArgs { uint32_t tokens, groups, hidden, width, topk, tile_rows; };
int pc_unit_of(int t) {
    return t == ref::Q2_0 || t == ref::IQ3_S || t == ref::BF16 || t == ref::F32 || t == 18 || t == 16 || t == 17 || t == 22 ? 16 : 32;
}
}  // namespace

bool Window::coop_k4() {
    if (const char* e = std::getenv("COOP_K4")) return std::atoi(e) != 0;
    const int lib = cx_->compile({"mm_bench.metal"}, "#define NO_GGML_COMMON 1\n");
    mtl::Pipeline p = cx_->pipeline(lib, "coop_layout_16_64_32");
    mtl::Buffer out = cx_->alloc(4 * 3 * 32 * 128 * 4);
    mtl::Dispatch d;
    d.p = &p;
    d.args = {mtl::Arg::B(out)};
    d.tg[0] = 32;
    cx_->run({d});
    const int* o = out.as<int>();
    bool k4 = true;
    for (int l = 0; l < 32 && k4; ++l) {
        const int* q = o + l * 128 * 4;
        const int cap = q[2];
        for (int i = 0; i + 3 < cap && k4; i += 4)
            for (int j = 1; j < 4; ++j)
                if (q[(i + j) * 4] != q[i * 4] + j || q[(i + j) * 4 + 1] != q[i * 4 + 1]) { k4 = false; break; }
    }
    cx_->free(out);
    return k4;
}

bool Window::pc_setup(int C) {
    PromptBufs& b = pb_;
    if (b.C >= C) return true;
    const double t0 = now_ms();
    const char* tune = std::getenv("STRATA_METAL_LATENT_PROMPT_TUNE");
    const bool tuned = latent_on_ && tune && std::strcmp(tune, "1") == 0;
    if (b.C == 0) {
        std::vector<std::string> kernels{"quant.metal", "units.metal", "mma.metal", "dense_mm.metal", "moe_mm.metal",
                                         "gr.metal", "gemv.metal", "prompt.metal"};
        if (latent_on_) kernels.push_back("latent_prompt.metal");
        if (tuned) kernels.push_back("latent_prompt_q3_stage.metal");
        b.lib = cx_->compile(kernels);
        b.lgdn = cx_->compile({"gdn.metal", "gdn_prompt.metal"}, "#define NO_GGML_COMMON 1\n");
        b.lqsa = cx_->compile({"qsa.metal", "qsa_prompt.metal"}, "#define NO_GGML_COMMON 1\n");
        b.k4 = coop_k4();
    }
    b = PromptBufs{b.lib, b.lgdn, b.lqsa, b.k4};   // (re)allocated below for the larger chunk
    b.v3e = !latent_on_ && b.k4 && T("blk.0.ffn_gate_exps.weight").type < 144;
    b.C = C;
    const size_t MT = (size_t) C;
    const int MB = (int) max_cells_ / 4 + 2;
    b.expert_tm = tuned ? 32 : 16;
    b.maxrows = (int) MT * TOPK + NEXP * b.expert_tm;
    const int nblk_max = ((int) MT * TOPK + 1023) / 1024;
    auto F32 = [&](size_t n) { return cx_->alloc(4 * n); };
    auto F16 = [&](size_t n) { return cx_->alloc(2 * n); };
    b.R = F32(MT * HC * N); b.bo = F32(MT * N); b.inj[0] = F32(MT * HC); b.inj[1] = F32(MT * HC); b.xn = F32(MT * HC * N);
    b.xnh = F16(MT * HC * N); b.V = F32(MT * (LR + HC)); b.loh = F16(MT * LR); b.mixed = F32(MT * N); b.mixh = F16(MT * N);
    b.fused = gdn_fused_inputs(cx_->device_name());
    if (b.fused) {   // the four GDN input projections as one GEMM: [qkv | z | alpha | beta] rows of 16480 floats
        b.yf = F32(MT * 16480);
        b.qkv = F32(1); b.z = F32(1); b.al = F32(1); b.be = F32(1);
    } else {
        b.qkv = F32(MT * 10240); b.z = F32(MT * 6144); b.al = F32(MT * 48); b.be = F32(MT * 48);
    }
    b.yg = F32(MT * 6144);
    b.ygh = F16(MT * 6144); b.gqn = F32(MT * 16 * 128); b.gkn = F32(MT * 16 * 128); b.gvc = F32(MT * 6144);
    b.ggb = F32(MT * 48 * 2); b.gkq = F32(MT * 16); b.go = F32(MT * 6144);
    b.qfull = F32(MT * 12288); b.kcur = F32(MT * 512); b.vcur = F32(MT * 512); b.iqr = F32(MT * 512); b.ikr = F32(MT * 128);
    b.qcur = F32(MT * 24 * 256); b.qcurh = F16(MT * 24 * 256); b.qidx = F32(MT * 4 * 128); b.scores = F32(MT * MB);
    b.qids = cx_->alloc(4 * MT * PC_QB); b.part = F32(MT * PC_QCHUNKS * 24 * 258); b.XQ = F32(4 * MT * 6144 + MT * 384);
    b.xqh = F16(MT * 6144);
    b.logits = F32(MT * NEXP); b.ids = cx_->alloc(4 * MT * TOPK); b.wts = F32(MT * TOPK);
    b.bcnt = cx_->alloc(4 * (size_t) nblk_max * NEXP); b.boff = cx_->alloc(4 * (size_t) nblk_max * NEXP);
    b.eoff = cx_->alloc(4 * NEXP); b.tiles = cx_->alloc(8 * (size_t) (b.maxrows / b.expert_tm + 1)); b.meta = cx_->alloc(64);
    b.stok = cx_->alloc(4 * (size_t) b.maxrows); b.sw = F32(b.maxrows); b.pos = cx_->alloc(4 * MT * TOPK);
    const int expert_width = latent_on_ ? (int) latent_args_.latent : N;
    b.XS = F16((size_t) b.maxrows * expert_width); b.HS = F16((size_t) b.maxrows * NFF);
    b.YE = F32((size_t) b.maxrows * expert_width);
    if (latent_on_) {
        const size_t groups = latent_args_.groups;
        b.lat_w = F16(groups * expert_width * N);
        b.lat_proj = F32(groups * MT * N);
        b.lat_pooled = F16(groups * MT * expert_width);
        // One layer of gate/up operands is reused across chunks and layers.
        if (tuned) b.lat_gu16 = F16(2ull * NEXP * NFF * expert_width);
    }
    b.yr = F32(MT * N); b.gsh = F32(MT * NFF); b.ush = F32(MT * NFF); b.hh = F16(MT * NFF); b.ys = F32(MT * N); b.sgv = F32(MT);
    b.WD = F16((size_t) (LR + HC) * HC * N); b.WU = F16((size_t) HC * N * LR);
    if (b.fused) {
        b.Wcat = F16((size_t) 16480 * N);
        b.Wqkv = F16(1); b.Wz = F16(1); b.Wal = F16(1); b.Wbe = F16(1);
    } else {
        b.Wqkv = F16((size_t) 10240 * N); b.Wz = F16((size_t) 6144 * N); b.Wal = F16((size_t) 48 * N); b.Wbe = F16((size_t) 48 * N);
    }
    b.Wout = F16((size_t) N * 6144);
    b.Wq = F16((size_t) 12288 * N); b.Wk = F16((size_t) 512 * N); b.Wv = F16((size_t) 512 * N); b.Wiq = F16((size_t) 512 * N);
    b.Wik = F16((size_t) 128 * N); b.Wr = F16((size_t) NEXP * N); b.Wsg = F16((size_t) N); b.Wshg = F16((size_t) NFF * N);
    b.Wshu = F16((size_t) NFF * N); b.Wshd = F16((size_t) N * NFF);
    // the PLE block's rows and projections
    b.pemb = F32(MT * N); b.pembh = F16(MT * N); b.pkey = F32(MT * 10240); b.pval = F32(MT * N); b.keyn = F32(MT * 10240);
    b.qn = F32(MT * 10240); b.gated = F32(MT * 10240); b.normW = F32(MT * 10240);
    if (had_on_) b.mixr = F32(MT * N);
    b.Wpk = F16((size_t) 10240 * N); b.Wpv = F16((size_t) N * N);
    // the expert GEMM tiles (prompt_chain's defaults): v3 gate/up 16,32,32,4 and down 16,64,32,2 with the M5 layout,
    // else v2
    b.gu3[0] = 16; b.gu3[1] = 32; b.gu3[2] = 32; b.gu3[3] = 4;
    b.dn3[0] = 16; b.dn3[1] = 64; b.dn3[2] = 32; b.dn3[3] = 2;
    b.meta.as<uint32_t>()[2] = b.v3e ? NFF / (b.gu3[1] * b.gu3[3]) : NFF / 64;
    b.meta.as<uint32_t>()[5] = b.v3e ? N / (b.dn3[1] * b.dn3[3]) : expert_width / 64;
    std::fprintf(stderr, "strata metal: prompt kernels (mac-kern's prompt chain) for chunks of %d rows, %s, ready in %.0f ms\n",
                 C, b.k4 ? "matrix-unit register operands (COOP_K4)" : "the older expert / attention kernels (no M5 layout)",
                 now_ms() - t0);
    return true;
}

// one chunk of nT rows at cells pos0.. (the window state is exactly at pos0 before, at pos0 + nT after)
bool Window::pc_chunk(const int64_t* tokens, int nT, int64_t pos0, std::string& err) {
    PromptBufs& b = pb_;
    strata::core::SessionState& ss = *ss_;
    const double th0 = now_ms();
    // ---- the rows: the embedding into every stream; the PLE table rows
    const gguf::Tensor& te = T("token_embd.weight");
    const size_t rb = ref::row_bytes(te.type, N);
    mrope_sync(pos0, nT);
    for (int t = 0; t < nT; ++t) {
        float* r = b.R.as<float>() + (size_t) t * HC * N;
        if (const float* img = img_row(pos0 + t)) std::memcpy(r, img, (size_t) N * 4);
        else ref::dequant(te.type, (const uint8_t*) W_.ptr + te.offset + (size_t) tokens[t] * rb, r, N);
        for (int c = 1; c < HC; ++c) std::memcpy(r + (size_t) c * N, r, (size_t) N * 4);
    }
    if (ple_on_) {
        std::vector<uint32_t> rows((size_t) nT * strata::kernels::PLE_N_HEADS);
        int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
        for (int t = 0; t < nT; ++t) {
            const int32_t tk = (int32_t) tokens[t];
            strata::kernels::ngram_rows(&tk, prev, 1, ss.ple.consts, rows.data() + (size_t) t * strata::kernels::PLE_N_HEADS);
            prev[0] = prev[1];
            prev[1] = tk;
        }
        if (!ss.ple.table->gather_batch(rows.data(), (size_t) nT, b.pemb.as<float>(), err)) return false;
    }
    b.host_ms += now_ms() - th0;
    std::vector<mtl::Dispatch> ds;
    ds.reserve(2400);
    static const bool stage_profile = std::getenv("STRATA_METAL_LATENT_PROMPT_PROFILE") != nullptr;
    static const bool latent_qsa_mm = [] {
        const char* e = std::getenv("STRATA_METAL_LATENT_PROMPT_QSA_MM");
        return e && std::strcmp(e, "1") == 0;
    }();
    const bool qsa_mm = b.k4 || (latent_on_ && latent_qsa_mm);
    std::vector<std::pair<const char*, size_t>> stage_ends;
    auto end_stage = [&](const char* name) {
        if (stage_profile && latent_on_) stage_ends.emplace_back(name, ds.size());
    };
    nb_ = false;
    if (pend_fold_ > 0) {   // the last window's commit, first
        fold(ds, pend_fold_, 1 - hp_);
        pend_fold_ = 0;
    }
    auto P_ = [&](int l, const std::string& fn, const mtl::Constants& c = {}, const std::string& key = "") {
        return P(l, fn, c, "pc:" + (key.empty() ? fn : key));
    };
    auto D = [&](const mtl::Pipeline* p, std::vector<mtl::Arg> a, uint32_t gx, uint32_t gy, uint32_t tg, uint32_t gz = 1) {
        mtl::Dispatch d;
        d.p = p;
        d.args = std::move(a);
        d.grid[0] = gx; d.grid[1] = gy; d.grid[2] = gz;
        d.tg[0] = tg;
        mtl::require_fit(d, "prompt-chain dispatch");
        ds.push_back(std::move(d));
    };
    auto gemm = [&](const mtl::Buffer& A, const mtl::Buffer& B, const mtl::Buffer& Y, int M, int Nn, int K,
                    size_t a_off = 0, size_t b_off = 0, size_t y_off = 0) {
        int tm = 64, tn = Nn <= 64 ? 32 : 64;
        gemm_tile(cx_->device_name(), Nn, K, tm, tn);
        const std::string fn = "gemm_f16_" + std::to_string(tm) + "_" + std::to_string(tn) + "_4";
        PcGemmArgs g{M, Nn, K, 0};
        D(P_(b.lib, fn), {mtl::Arg::B(A, a_off), mtl::Arg::B(B, b_off), mtl::Arg::B(Y, y_off), mtl::Arg::V(g)},
          (Nn + tn - 1) / tn, (M + tm - 1) / tm, 128);
    };
    auto dq = [&](const gguf::Tensor& t, const mtl::Buffer& out, size_t out_off = 0, uint32_t perm = 0) {
        const int n_in = (int) t.shape[0], n_out = t.shape.size() > 1 ? (int) t.shape[1] : 1;
        PcDqArgs a{(uint32_t) n_in, (uint32_t) n_out, (uint32_t) ref::row_bytes(t.type, n_in), perm};
        D(P_(b.lib, "dq_f16_" + std::to_string(t.type)), {mtl::Arg::B(W_, t.offset), mtl::Arg::B(out, out_off), mtl::Arg::V(a)},
          (n_in / pc_unit_of(t.type) + 63) / 64, n_out, 64);
    };
    auto act16 = [&](const mtl::Buffer& x, const mtl::Buffer& xh, size_t n) {
        const uint32_t nn = (uint32_t) n;
        D(P_(b.lib, "act_f16"), {mtl::Arg::B(x), mtl::Arg::B(xh), mtl::Arg::V(nn)}, (nn + 255) / 256, 1, 256);
    };
    const PcLatentArgs la{(uint32_t) nT, latent_args_.groups, N, latent_args_.latent, TOPK, b.expert_tm};
    auto latent_project = [&](size_t offset, uint32_t in, uint32_t out, uint32_t row_bytes,
                              const mtl::Buffer& input, bool group_inputs) {
        const PcDqArgs args{in, la.groups * out, row_bytes, 0};
        D(P_(b.lib, "dq_f16_2"), {mtl::Arg::B(latent_du_, offset), mtl::Arg::B(b.lat_w), mtl::Arg::V(args)},
          (in / 32 + 63) / 64, args.n_out, 64);
        for (uint32_t group = 0; group < la.groups; ++group)
            gemm(input, b.lat_w, b.lat_proj, nT, out, in, group_inputs ? 2ull * group * nT * in : 0,
                 2ull * group * out * in, 4ull * group * nT * out);
    };
    int cur_inj = 0;
    auto hc_read = [&](const std::string& pre, bool apply) {
        const gguf::Tensor &tn = T(pre + "_norm.weight"), &tdn = T(pre + "_down.weight"), &tin = T(pre + "_inject.weight"),
                           &tup = T(pre + "_up.weight");
        dq(tdn, b.WD);
        dq(tin, b.WD, (size_t) 2 * LR * HC * N);
        GrArgs a{N, HC, LR, (uint32_t) nT, apply ? 1u : 0u, eps_};
        PcEpiArgs ea{(uint32_t) nT, N, HC, LR};
        dq(tup, b.WU, 0, HC);
        D(P_(b.lib, "gr_norm_h"), {mtl::Arg::B(b.R), mtl::Arg::B(b.bo), mtl::Arg::B(b.inj[cur_inj]), mtl::Arg::B(W_, tn.offset),
                                  mtl::Arg::B(b.xnh), mtl::Arg::V(a)}, HC, nT, 256);
        gemm(b.xnh, b.WD, b.V, nT, LR + HC, HC * N);
        D(P_(b.lib, "gr_epi_down"), {mtl::Arg::B(b.V), mtl::Arg::B(b.loh), mtl::Arg::B(b.inj[1 - cur_inj]), mtl::Arg::V(ea)},
          (nT * (LR + HC) + 255) / 256, 1, 256);
        PcGemmArgs g{nT, HC * N, LR, 0};
        D(P_(b.lib, "gemm_hc_up_64_64_4"), {mtl::Arg::B(b.loh), mtl::Arg::B(b.WU), mtl::Arg::B(b.xnh), mtl::Arg::B(b.mixed),
                                           mtl::Arg::B(b.mixh), mtl::Arg::V(g), mtl::Arg::V(ea)},
          HC * N / 64, (nT + 63) / 64, 128);
        cur_inj = 1 - cur_inj;
    };
    const int hp = hp_;
    int gi = 0, qi = 0;
    for (int l = 0; l < 48; ++l) {
        const std::string bl = "blk." + std::to_string(l) + ".";
        hc_read(bl + "hc_attn", l > 0 && !(l == 1 && ple_on_));
        if (l % 4 != 3) {
            dq(T(bl + "ssm_out.weight"), b.Wout);
            PcGdnPArgs ga{(uint32_t) nT, 16, 48, 10240, eps_, 0};
            if (b.fused) {
                dq(T(bl + "attn_qkv.weight"), b.Wcat, 0);
                dq(T(bl + "attn_gate.weight"), b.Wcat, (size_t) 2 * 10240 * N);
                dq(T(bl + "ssm_alpha.weight"), b.Wcat, (size_t) 2 * (10240 + 6144) * N);
                dq(T(bl + "ssm_beta.weight"), b.Wcat, (size_t) 2 * (10240 + 6144 + 48) * N);
                gemm(b.mixh, b.Wcat, b.yf, nT, 16480, N);
                D(P_(b.lgdn, "gdn_pre", mtl::Constants().i(7, 16480), "gdn_pre_f"),
                  {mtl::Arg::B(b.yf), mtl::Arg::B(ghist_[hp][(size_t) gi]), mtl::Arg::B(W_, T(bl + "ssm_conv1d.weight").offset),
                   mtl::Arg::B(b.yf, (size_t) 4 * (10240 + 6144)), mtl::Arg::B(b.yf, (size_t) 4 * (10240 + 6144 + 48)),
                   mtl::Arg::B(W_, T(bl + "ssm_dt.bias").offset), mtl::Arg::B(W_, T(bl + "ssm_a").offset), mtl::Arg::B(b.gqn),
                   mtl::Arg::B(b.gkn), mtl::Arg::B(b.gvc), mtl::Arg::B(b.ggb), mtl::Arg::B(b.gkq),
                   mtl::Arg::B(ghist_[1 - hp][(size_t) gi]), mtl::Arg::V(ga)},
                  16 + 24, nT, 256);
            } else {
                dq(T(bl + "attn_qkv.weight"), b.Wqkv);
                dq(T(bl + "attn_gate.weight"), b.Wz);
                dq(T(bl + "ssm_alpha.weight"), b.Wal);
                dq(T(bl + "ssm_beta.weight"), b.Wbe);
                gemm(b.mixh, b.Wqkv, b.qkv, nT, 10240, N);
                gemm(b.mixh, b.Wz, b.z, nT, 6144, N);
                gemm(b.mixh, b.Wal, b.al, nT, 48, N);
                gemm(b.mixh, b.Wbe, b.be, nT, 48, N);
                D(P_(b.lgdn, "gdn_pre"),
                  {mtl::Arg::B(b.qkv), mtl::Arg::B(ghist_[hp][(size_t) gi]), mtl::Arg::B(W_, T(bl + "ssm_conv1d.weight").offset),
                   mtl::Arg::B(b.al), mtl::Arg::B(b.be), mtl::Arg::B(W_, T(bl + "ssm_dt.bias").offset),
                   mtl::Arg::B(W_, T(bl + "ssm_a").offset), mtl::Arg::B(b.gqn), mtl::Arg::B(b.gkn), mtl::Arg::B(b.gvc),
                   mtl::Arg::B(b.ggb), mtl::Arg::B(b.gkq), mtl::Arg::B(ghist_[1 - hp][(size_t) gi]), mtl::Arg::V(ga)},
                  16 + 24, nT, 256);
            }
            D(P_(b.lgdn, gdn_scan_name(cx_->device_name(), nT), mtl::Constants().i(6, 0)),
              {mtl::Arg::B(gst_[(size_t) gi]), mtl::Arg::B(b.gqn), mtl::Arg::B(b.gkn), mtl::Arg::B(b.gvc), mtl::Arg::B(b.ggb),
               mtl::Arg::B(b.gkq), mtl::Arg::B(b.go), mtl::Arg::V(ga)},
              48 * 4, 1, 32 * 4);
            if (b.fused)
                D(P_(b.lgdn, "gdn_post", mtl::Constants().i(7, 16480), "gdn_post_f"),
                  {mtl::Arg::B(b.go), mtl::Arg::B(b.yf, (size_t) 4 * 10240), mtl::Arg::B(W_, T(bl + "ssm_norm.weight").offset),
                   mtl::Arg::B(b.ygh), mtl::Arg::B(b.yg), mtl::Arg::V(ga)},
                  48, nT, 128);
            else
                D(P_(b.lgdn, "gdn_post"), {mtl::Arg::B(b.go), mtl::Arg::B(b.z), mtl::Arg::B(W_, T(bl + "ssm_norm.weight").offset),
                                          mtl::Arg::B(b.ygh), mtl::Arg::B(b.yg), mtl::Arg::V(ga)},
                  48, nT, 128);
            gemm(b.ygh, b.Wout, b.bo, nT, N, 6144);
            ++gi;
        } else {
            dq(T(bl + "attn_q.weight"), b.Wq);
            dq(T(bl + "attn_k.weight"), b.Wk);
            dq(T(bl + "attn_v.weight"), b.Wv);
            dq(T(bl + "indexer.q_proj.weight"), b.Wiq);
            dq(T(bl + "indexer.k_proj.weight"), b.Wik);
            dq(T(bl + "attn_output.weight"), b.Wout);
            gemm(b.mixh, b.Wq, b.qfull, nT, 12288, N);
            gemm(b.mixh, b.Wk, b.kcur, nT, 512, N);
            gemm(b.mixh, b.Wv, b.vcur, nT, 512, N);
            gemm(b.mixh, b.Wiq, b.iqr, nT, 512, N);
            gemm(b.mixh, b.Wik, b.ikr, nT, 128, N);
            QsaArgs a{(uint32_t) pos0, 0, (uint32_t) max_cells_, (uint32_t) nT, eps_};
            D(P_(b.lqsa, "qsa_prep", qsa_c(), qsa_k()),
              {mtl::Arg::B(b.qfull), mtl::Arg::B(b.kcur), mtl::Arg::B(b.vcur), mtl::Arg::B(b.iqr), mtl::Arg::B(b.ikr),
               mtl::Arg::B(W_, T(bl + "attn_q_norm.weight").offset), mtl::Arg::B(W_, T(bl + "attn_k_norm.weight").offset),
               mtl::Arg::B(W_, T(bl + "indexer.q_norm.weight").offset), mtl::Arg::B(W_, T(bl + "indexer.k_norm.weight").offset),
               mtl::Arg::B(cs_), mtl::Arg::B(b.qcur), mtl::Arg::B(b.qidx), mtl::Arg::B(kq_[(size_t) qi]), mtl::Arg::B(vq_[(size_t) qi]),
               mtl::Arg::B(ksc_[(size_t) qi]), mtl::Arg::B(vsc_[(size_t) qi]), mtl::Arg::B(tail_[(size_t) qi]),
               mtl::Arg::B(dead_[(size_t) qi]), mtl::Arg::B(pooled_[(size_t) qi]), mtl::Arg::V(a), mtl::Arg::B(mrope_)},
              33, nT, 256);
            if (pos0 + nT > PC_QB) {   // some row's context is past the budget: the indexer selects
                if (qsa_mm)
                    D(P_(b.lqsa, "qsa_score_mm"), {mtl::Arg::B(b.qidx), mtl::Arg::B(pooled_[(size_t) qi]), mtl::Arg::B(dead_[(size_t) qi]),
                                                 mtl::Arg::B(b.scores), mtl::Arg::V(a)},
                      (uint32_t) (((pos0 + nT) / 4 + 1 + 63) / 64), (nT + 15) / 16, 128);
                else
                    D(P_(b.lqsa, "qsa_score"), {mtl::Arg::B(b.qidx), mtl::Arg::B(pooled_[(size_t) qi]), mtl::Arg::B(dead_[(size_t) qi]),
                                              mtl::Arg::B(b.scores), mtl::Arg::V(a)},
                      (uint32_t) (((pos0 + nT) / 4 + 1 + 7) / 8), nT, 256);
            }
            D(P_(b.lqsa, "qsa_topk"), {mtl::Arg::B(b.scores), mtl::Arg::B(b.qids), mtl::Arg::V(a)}, nT, 1, 1024);
            if (qsa_mm) {
                const uint32_t nq = (uint32_t) nT * 24 * 256;
                D(P_(b.lqsa, "qsa_q_f16"), {mtl::Arg::B(b.qcur), mtl::Arg::B(b.qcurh), mtl::Arg::V(nq)}, (nq + 255) / 256, 1, 256);
                D(P_(b.lqsa, "qsa_attn_mm", mtl::Constants().b(7, b.k4), b.k4 ? "qsa_attn_mm#k4" : "qsa_attn_mm#scalar"),
                  {mtl::Arg::B(b.qcurh), mtl::Arg::B(kq_[(size_t) qi]), mtl::Arg::B(vq_[(size_t) qi]), mtl::Arg::B(ksc_[(size_t) qi]),
                   mtl::Arg::B(vsc_[(size_t) qi]), mtl::Arg::B(b.qids), mtl::Arg::B(b.part), mtl::Arg::V(a)},
                  PC_QCHUNKS, 2, 32, nT);
            } else {
                D(P_(b.lqsa, "qsa_attn"), {mtl::Arg::B(b.qcur), mtl::Arg::B(kq_[(size_t) qi]), mtl::Arg::B(vq_[(size_t) qi]),
                                          mtl::Arg::B(ksc_[(size_t) qi]), mtl::Arg::B(vsc_[(size_t) qi]), mtl::Arg::B(b.qids),
                                          mtl::Arg::B(b.part), mtl::Arg::V(a)},
                  PC_QCHUNKS, 2, 256, nT);
            }
            D(P_(b.lqsa, "qsa_merge"), {mtl::Arg::B(b.part), mtl::Arg::B(b.qfull), mtl::Arg::B(b.XQ), mtl::Arg::V(a)}, 24, nT, 256);
            act16(b.XQ, b.xqh, (size_t) nT * 6144);
            gemm(b.xqh, b.Wout, b.bo, nT, N, 6144);
            ++qi;
        }
        end_stage("attention");
        // ---- MoE
        hc_read(bl + "hc_ffn", true);
        dq(T(bl + "ffn_gate_inp.weight"), b.Wr);
        dq(T(bl + "ffn_gate_inp_shexp.weight"), b.Wsg);
        dq(T(bl + "ffn_gate_shexp.weight"), b.Wshg);
        dq(T(bl + "ffn_up_shexp.weight"), b.Wshu);
        dq(T(bl + "ffn_down_shexp.weight"), b.Wshd);
        gemm(b.mixh, b.Wr, b.logits, nT, NEXP, N);
        const uint32_t nt = (uint32_t) nT;
        D(P_(b.lib, "route_topk"), {mtl::Arg::B(b.logits), mtl::Arg::B(b.ids), mtl::Arg::B(b.wts), mtl::Arg::V(nt)}, (nT + 7) / 8, 1, 256);
        const int nent = nT * TOPK, nblk = (nent + 1023) / 1024;
        PcMmPlanArgs pa{(uint32_t) nent, TOPK, NEXP, b.expert_tm, (uint32_t) nblk};
        const uint32_t nclr = (uint32_t) b.maxrows;
        D(P_(b.lib, "mm_clear"), {mtl::Arg::B(b.stok), mtl::Arg::V(nclr)}, (b.maxrows + 255) / 256, 1, 256);
        D(P_(b.lib, "mm_hist"), {mtl::Arg::B(b.ids), mtl::Arg::B(b.bcnt), mtl::Arg::V(pa)}, nblk, 1, 1024);
        D(P_(b.lib, "mm_scan"), {mtl::Arg::B(b.bcnt), mtl::Arg::B(b.boff), mtl::Arg::B(b.eoff), mtl::Arg::B(b.tiles), mtl::Arg::B(b.meta),
                                mtl::Arg::V(pa)}, 1, 1, NEXP);
        D(P_(b.lib, "mm_place"), {mtl::Arg::B(b.ids), mtl::Arg::B(b.wts), mtl::Arg::B(b.boff), mtl::Arg::B(b.eoff), mtl::Arg::B(b.stok),
                                 mtl::Arg::B(b.sw), mtl::Arg::B(b.pos), mtl::Arg::V(pa)}, nblk, 1, 1024);
        PcMmArgs ma{};
        int gu_type, down_type;
        const mtl::Buffer* expert_weights = &W_;
        size_t ebase;
        const size_t labels = (size_t) l * NEXP * sizeof(int32_t);
        size_t du_offset = 0;
        if (latent_on_) {
            const auto& a = latent_args_;
            ma = {0, a.up_off, a.down_off, a.expert_bytes, a.expert_bytes, a.gu_row, a.down_row, a.latent, NFF};
            gu_type = ref::Q3_K;
            down_type = ref::IQ4_NL;
            expert_weights = &latent_a_;
            ebase = (size_t) l * NEXP * a.expert_bytes;
            du_offset = (size_t) l * (latent_dense_.d_bytes() + latent_dense_.u_bytes());
            latent_project(du_offset, N, la.width, (uint32_t) latent_dense_.d_row_bytes(), b.mixh, false);
        } else {
            const gguf::Tensor &eg = T(bl + "ffn_gate_exps.weight"), &eu = T(bl + "ffn_up_exps.weight"),
                               &ed = T(bl + "ffn_down_exps.weight");
            const auto gr = (uint32_t) ref::row_bytes(eg.type, N), dr = (uint32_t) ref::row_bytes(ed.type, NFF);
            ma = {eg.offset, eu.offset, ed.offset, (uint64_t) gr * NFF, (uint64_t) dr * N, gr, dr, N, NFF};
            gu_type = eg.type;
            down_type = ed.type;
            ebase = moe_rebase(ma.gate_off, ma.up_off, ma.down_off);
        }
        const uint32_t ne = N;
        const bool hg = !latent_on_ && had_gu(l), hd = !latent_on_ && had_dn(l);
        if (hg) had_rows(ds, b.lib, "pc:", b.mixed, 0, b.mixr, 0, nT, N, hsx_.handle ? &hsx_ : nullptr, false);
        mtl::Dispatch g;
        if (latent_on_) {
            g.p = P_(b.lib, "latent_prompt_gather");
            g.args = {mtl::Arg::B(b.lat_proj), mtl::Arg::B(latent_inv_, (size_t) l * la.width * sizeof(float)),
                      mtl::Arg::B(latent_labels_, labels), mtl::Arg::B(b.tiles), mtl::Arg::B(b.stok),
                      mtl::Arg::B(b.XS), mtl::Arg::V(la)};
        } else {
            g.p = P_(b.lib, "mm_gather");
            g.args = {mtl::Arg::B(hg ? b.mixr : b.mixed), mtl::Arg::B(b.stok), mtl::Arg::B(b.XS), mtl::Arg::V(ne)};
        }
        g.tg[0] = 256;
        g.indirect = &b.meta;
        g.ind_off = 8 * 4;
        ds.push_back(g);
        end_stage("route_D");
        auto tl3 = [](const int* t) { return std::to_string(t[0]) + "_" + std::to_string(t[1]) + "_" + std::to_string(t[2]) + "_" + std::to_string(t[3]); };
        mtl::Dispatch gu;
        const bool staged = b.lat_gu16.handle && nT >= PC_STAGE_GU_MIN;
        if (staged) {
            const uint32_t q3[4] = {ma.n_embd, ma.n_ff, ma.gu_row, uint32_t(ma.gu_stride)};
            D(P_(b.lib, "latent_prompt_q3_stage"),
              {mtl::Arg::B(*expert_weights, ebase), mtl::Arg::B(b.lat_gu16), mtl::Arg::V(q3)},
              (2 * ma.n_ff * (ma.n_embd / 32) + 127) / 128, NEXP, 128);
            gu.p = P_(b.lib, "latent_prompt_gu_f16");
        } else if (b.v3e) {
            const std::string fn = "mm_gu3_" + std::to_string(gu_type) + "_" + tl3(b.gu3);
            gu.p = P_(b.lib, fn, mtl::Constants().b(7, true), fn + "#k4");
        } else {
            const char* latent_tile = b.expert_tm == 32 ? "_32_64_32_2" : "_16_64_64_2";
            const std::string fn = "mm_gu2_" + std::to_string(gu_type) + (latent_on_ ? latent_tile : gu2_tile(gu_type));
            gu.p = P_(b.lib, fn, mtl::Constants().i(5, 0), fn + "#p0");
        }
        gu.args = {staged ? mtl::Arg::B(b.lat_gu16) : mtl::Arg::B(*expert_weights, ebase), mtl::Arg::B(b.XS), mtl::Arg::B(b.HS), mtl::Arg::B(b.tiles), mtl::Arg::V(ma)};
        gu.tg[0] = b.v3e ? 32 * b.gu3[3] : 64;
        gu.indirect = &b.meta;
        gu.ind_off = 2 * 4;
        mtl::require_fit(gu, "prompt gate/up dispatch");
        ds.push_back(gu);
        if (hd) had_rows_ind(ds, b.lib, "pc:", b.HS, b.meta);
        mtl::Dispatch dn;
        if (b.v3e) {
            const std::string fn = "mm_down3_" + std::to_string(down_type) + "_" + tl3(b.dn3);
            dn.p = P_(b.lib, fn, mtl::Constants().b(7, true), fn + "#k4");
        } else {
            const std::string tile = b.expert_tm == 32 ? "_32_64_32_2" : dn2_tile(down_type);
            dn.p = P_(b.lib, "mm_down2_" + std::to_string(down_type) + tile);
        }
        dn.args = {mtl::Arg::B(*expert_weights, ebase), mtl::Arg::B(b.HS), mtl::Arg::B(b.YE), mtl::Arg::B(b.tiles), mtl::Arg::V(ma)};
        dn.tg[0] = b.v3e ? 32 * b.dn3[3] : 64;
        dn.indirect = &b.meta;
        dn.ind_off = 5 * 4;
        mtl::require_fit(dn, "prompt down dispatch");
        ds.push_back(dn);
        end_stage("experts");
        if (latent_on_) {
            D(P_(b.lib, "latent_prompt_pool"),
              {mtl::Arg::B(b.YE), mtl::Arg::B(b.pos), mtl::Arg::B(b.ids), mtl::Arg::B(b.wts),
               mtl::Arg::B(latent_labels_, labels), mtl::Arg::B(b.lat_pooled), mtl::Arg::V(la)},
              (la.groups * la.tokens * la.width + 255) / 256, 1, 256);
            latent_project(du_offset + latent_dense_.d_bytes(), la.width, N, latent_args_.u_row, b.lat_pooled, true);
            D(P_(b.lib, "latent_prompt_sum"), {mtl::Arg::B(b.lat_proj), mtl::Arg::B(b.yr), mtl::Arg::V(la)},
              (nT * N + 255) / 256, 1, 256);
        } else {
            const uint32_t ca[4] = {(uint32_t) nT, N, TOPK, 0};
            D(P_(b.lib, "mm_combine"), {mtl::Arg::B(b.YE), mtl::Arg::B(b.pos), mtl::Arg::B(b.sw), mtl::Arg::B(b.yr), mtl::Arg::V(ca)},
              (nT * N + 255) / 256, 1, 256);
        }
        end_stage("U");
        gemm(b.mixh, b.Wshg, b.gsh, nT, NFF, N);
        gemm(b.mixh, b.Wshu, b.ush, nT, NFF, N);
        const uint32_t nh = (uint32_t) nT * NFF;
        D(P_(b.lib, "swiglu_h"), {mtl::Arg::B(b.gsh), mtl::Arg::B(b.ush), mtl::Arg::B(b.hh), mtl::Arg::V(nh)}, (nh + 255) / 256, 1, 256);
        gemm(b.hh, b.Wshd, b.ys, nT, N, NFF);
        gemm(b.mixh, b.Wsg, b.sgv, nT, 1, N);
        const uint32_t oa[2] = {(uint32_t) nT, N};
        D(P_(b.lib, "moe_out"), {mtl::Arg::B(b.yr), mtl::Arg::B(b.ys), mtl::Arg::B(b.sgv), mtl::Arg::B(b.bo), mtl::Arg::V(oa)},
          (nT * N + 255) / 256, 1, 256);
        end_stage("shared");
        if (l == 0 && ple_on_) {
            // layer 0's FFN write applied to R, then the PLE block (as the window: msl/ple.metal over nT rows)
            GrArgs a{N, HC, LR, (uint32_t) nT, 1u, eps_};
            D(P_(b.lib, "gr_norm"), {mtl::Arg::B(b.R), mtl::Arg::B(b.bo), mtl::Arg::B(b.inj[cur_inj]),
                                    mtl::Arg::B(W_, T("blk.1.hc_attn_norm.weight").offset), mtl::Arg::B(b.xn), mtl::Arg::V(a)},
              HC, nT, 256);
            act16(b.pemb, b.pembh, (size_t) nT * N);
            dq(T("blk.1.ple_key.weight"), b.Wpk);
            dq(T("blk.1.ple_value.weight"), b.Wpv);
            gemm(b.pembh, b.Wpk, b.pkey, nT, 10240, N);
            gemm(b.pembh, b.Wpv, b.pval, nT, N, N);
            struct PleArgs { uint32_t n_embd, hc; float eps; } pa2{N, HC, strata::kernels::NG_RMS_EPS};
            auto off = [&](const char* n) { return T(n).offset; };
            D(P(lple_, "ple_gnorm", {}, "ple_gnorm"), {mtl::Arg::B(b.pkey), mtl::Arg::B(W_, off("blk.1.ple_norm_key.weight")),
                                                       mtl::Arg::B(b.keyn), mtl::Arg::V(pa2)}, HC, nT, 256);
            D(P(lple_, "ple_gnorm", {}, "ple_gnorm"), {mtl::Arg::B(b.R), mtl::Arg::B(W_, off("blk.1.ple_norm_query.weight")),
                                                       mtl::Arg::B(b.qn), mtl::Arg::V(pa2)}, HC, nT, 256);
            D(P(lple_, "ple_gate_norm", {}, "ple_gate_norm"),
              {mtl::Arg::B(b.keyn), mtl::Arg::B(b.qn), mtl::Arg::B(b.pval), mtl::Arg::B(W_, off("blk.1.ple_norm_conv.weight")),
               mtl::Arg::B(b.gated), mtl::Arg::B(b.normW), mtl::Arg::V(pa2)}, HC, nT, 256);
            D(P(lple_, "ple_conv_add", {}, "ple_conv_add"),
              {mtl::Arg::B(pleHist_), mtl::Arg::B(b.normW), kw_arg(), mtl::Arg::B(b.gated),
               mtl::Arg::B(b.R), mtl::Arg::V(pa2)}, HC * N / 256, nT, 256);
            end_stage("PLE");
        }
    }
    // every row's final residual (the last write applied): the drafter's prompt rows
    {
        GrArgs a{N, HC, LR, (uint32_t) nT, 1u, eps_};
        D(P_(b.lib, "gr_norm"), {mtl::Arg::B(b.R), mtl::Arg::B(b.bo), mtl::Arg::B(b.inj[cur_inj]),
                                mtl::Arg::B(W_, T("output_hc_norm.weight").offset), mtl::Arg::B(b.xn), mtl::Arg::V(a)}, HC, nT, 256);
    }
    if (stage_ends.empty()) {
        b.gpu_ms += 1000.0 * cx_->run(ds);
    } else {
        end_stage("final");
        size_t first = 0;
        std::map<std::string, double> timings;
        for (const auto& [name, last] : stage_ends) {
            const std::vector<mtl::Dispatch> stage(ds.begin() + first, ds.begin() + last);
            const double ms = 1000.0 * cx_->run(stage);
            timings[name] += ms;
            b.gpu_ms += ms;
            first = last;
        }
        std::fprintf(stderr, "strata metal latent prompt profile: %d tokens at %lld", nT, (long long) pos0);
        for (const auto& [name, ms] : timings) std::fprintf(stderr, ", %s %.3f ms", name.c_str(), ms);
        std::fputc('\n', stderr);
    }
    hp_ = 1 - hp_;
    // the host side of the commit: the PLE history (the last 9 normalized rows) and the n-gram context
    if (ple_on_) {
        using namespace strata::kernels;
        const float* nw = b.normW.as<float>();
        float* hist = pleHist_.as<float>();
        for (int c = 0; c < NG_HC_DIM; ++c) {
            float* col = hist + (size_t) c * NG_HIST;
            float tmp[NG_HIST * 2];
            int k = 0;
            const int keep_old = std::max(0, NG_HIST - nT);
            for (int r = NG_HIST - keep_old; r < NG_HIST; ++r) tmp[k++] = col[r];
            for (int t = std::max(0, nT - NG_HIST); t < nT; ++t) tmp[k++] = nw[(size_t) t * NG_HC_DIM + c];
            for (int r = 0; r < NG_HIST; ++r) col[r] = tmp[r];
        }
    }
    for (int t = 0; t < nT; ++t) {
        ss.ple_prev[0] = ss.ple_prev[1];
        ss.ple_prev[1] = (int32_t) tokens[t];
    }
    last_T_ = 0;   // nothing to commit
    return true;
}

bool Window::prompt(const int64_t* tokens, int64_t n, int64_t pos0,
                    const std::function<bool(const float*, int64_t, int64_t, std::string&)>& on_chunk, std::string& err) {
    static const int C = [] {
        const char* e = std::getenv("STRATA_METAL_PROMPT_CHUNK");
        return std::max(64, std::min(8192, e ? std::atoi(e) : 2048));
    }();
    // Reserve the configured latent chunk once. The warm-up sends prompts of
    // increasing sizes, and the runtime owns these allocations until teardown.
    const int capacity = latent_on_ ? C : (int) std::min<int64_t>(C, std::max<int64_t>(n, 64));
    if (!pc_setup(capacity)) { err = "metal prompt: setup failed"; return false; }
    const double t0 = now_ms();
    pb_.gpu_ms = pb_.host_ms = pb_.mtp_ms = 0;
    for (int64_t c = 0; c < n; c += C) {
        const int nT = (int) std::min<int64_t>(C, n - c);
        if (pos0 + c + nT + 1 >= max_cells_) { err = "metal prompt: the prompt runs past the context"; return false; }
        if (!pc_chunk(tokens + c, nT, pos0 + c, err)) return false;
        if (on_chunk) {
            const double tm0 = now_ms();
            mpsrc_ = &pb_.R;   // the drafter's prompt rows are this chunk's final residuals
            const bool ok = on_chunk(nullptr, nT, pos0 + c, err);
            mpsrc_ = nullptr;
            pb_.mtp_ms += now_ms() - tm0;
            if (!ok) return false;
        }
    }
    if (std::getenv("STRATA_METAL_TRACE") || std::getenv("STRATA_METAL_PROFILE"))
        std::fprintf(stderr, "strata metal: prompt %lld tokens in %.1f ms (GPU %.1f ms, host rows %.1f ms, drafter rows %.1f ms) "
                             "-> %.1f tok/s\n", (long long) n, now_ms() - t0, pb_.gpu_ms, pb_.host_ms, pb_.mtp_ms,
                     1000.0 * (double) n / (now_ms() - t0));
    return true;
}
