// metal/src/engine/window.cpp - the decode window on Metal: Verifier::init/run/commit's Metal side (the one backend
// hook, src/core/verify.cpp under STRATA_GPU_METAL).  Everything above it - the decode loop, MTP control, the serve
// protocol, the PLE table reads, the file tier - is the shared host code.
//
// The window is mac-kern's chain (metal/kernels, tests/token_chain.cpp) on the model GGUF itself: shard 1 is mmapped
// and wrapped no-copy as one MTLBuffer, every weight is read where it lies (unified memory: no upload, no expert
// cache), and one window is encoded as a dependency graph in one command buffer.  Its own state lives here (the GDN
// recurrence and conv history, the int8 K/V and the indexer per QSA layer, the PLE history): the engine's
// SessionState buffers are not used by it, except the PLE token window (ss.ple_prev) and table.
//
// First version (MAC-PLAN Phase 3 start): windows of one token (no MTP drafts yet: --mtp-max-t 1), greedy.  The PLE
// block's projections run on the GPU; its norms, gate and conv (10240 values) on the CPU between two command buffers
// (layer 0 | layers 1..47 + head).
#include "gguf_lite.h"
#include "mtl_rt.h"
#include "resource_path.h"
#include "ref_quant.h"
#include "arena_budget.h"
#include "stream.h"
#include "../../tests/coop_probe.h"

#include "strata/core/layer.hpp"
#include "strata/core/conversation_cache.hpp"
#include "strata/core/latent_dense.hpp"
#include "strata/core/session.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/cpu/hadamard.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <functional>
#include <signal.h>
#include <unistd.h>
#include <map>
#include <memory>
#include <string>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <thread>
#include <vector>

namespace strata::gpu { uint64_t purge_device_allocations(size_t min_bytes); }   // metal/src/gpu/runtime.cpp

namespace strata::metal_engine {

// The prompt GDN scan kernel, chosen by chip and prompt length (gdn_prompt.metal, gdn_scan_r<IB, TB>; every choice is
// bit-identical to gdn_scanr_4: the token order and the arithmetic are the same, only the staged block size TB changes, which
// changes the threadgroup memory a core can hold at once).  Measured by metal/tests/gdn_chunk_bench:
//   Apple M4 (the base chip): gdn_scanr_4_8, 1.16x on the scan, 1.11x on the GDN prompt stage, T 512..32K
//   Apple M5 Max: gdn_scanr_4_4 from T >= 4096 (1.04x at 4K, 1.10x at 16K); below that the default is as fast
//   anything else (other chips, unknown names): gdn_scanr_4 as before
// STRATA_METAL_GDN_SCAN=<gdn_scanr_4 | gdn_scanr_4_8 | gdn_scanr_4_4 | gdn_scanr_4_2> forces a kernel (A/B and diagnosis).
static const char* gdn_scan_name(const std::string& device, int T) {
    static const std::string forced = [] {
        const char* e = std::getenv("STRATA_METAL_GDN_SCAN");
        const std::string s = e ? e : "";
        return (s == "gdn_scanr_4" || s == "gdn_scanr_4_8" || s == "gdn_scanr_4_4" || s == "gdn_scanr_4_2") ? s : std::string();
    }();
    if (!forced.empty()) return forced.c_str();
    if (device == "Apple M4") return "gdn_scanr_4_8";
    if (device == "Apple M5 Max" && T >= 4096) return "gdn_scanr_4_4";
    return "gdn_scanr_4";
}

// The M5 family's dense GEMM tiles for the prompt (bit-identical to the 64 x 64 tile: the tile only splits the output; measured on
// an M5 Max, T = 2048 / 4096): the GDN out projection (N 2560, K 6144) 64 x 128 (1.17 -> 0.99 ms), the wide input projections
// (N >= 6144) 128 x 64.  Other chips keep 64 x 64 (M4: no gain).  STRATA_METAL_GEMM_TILES=0 keeps the old tiles everywhere.
static void gemm_tile(const std::string& device, int Nn, int K, int& tm, int& tn) {
    static const bool off = [] { const char* e = std::getenv("STRATA_METAL_GEMM_TILES"); return e && std::string(e) == "0"; }();
    if (off || device.rfind("Apple M5", 0) != 0) return;
    if (Nn == 2560 && K == 6144) { tm = 64; tn = 128; }
    else if (Nn >= 6144) { tm = 128; tn = 64; }
}
// The four GDN input projections (qkv, z, alpha, beta) as ONE GEMM with a 16480 float output row (M5 family: 3.09 -> 2.87 ms a layer at
// T = 2048; gdn_pre / gdn_post read the pieces at that stride, function constant 7).  Bit-identical to the separate GEMMs.
// STRATA_METAL_GDN_FUSE=0 keeps the separate ones.
static bool gdn_fused_inputs(const std::string& device) {
    static const bool off = [] { const char* e = std::getenv("STRATA_METAL_GDN_FUSE"); return e && std::string(e) == "0"; }();
    return !off && device.rfind("Apple M5", 0) == 0;
}

namespace {

int g_slot_cap = 0;   // --batch N: the window keeps N + 1 sequences' running states side by side (set_batch_slots)

constexpr int N = 2560, HC = 4, LR = 320, NFF = 640, TOPK = 10, NEXP = 512, MAXT = 8, NVOCAB_MAX = 248320;
constexpr int P_MAXE = 128, P_GEXP = 16, P_GSTART = P_GEXP + P_MAXE, P_ETOK = P_GSTART + P_MAXE + 4,
              P_EW = P_ETOK + P_MAXE, P_TOKENT = P_EW + P_MAXE, P_INTS = P_TOKENT + P_MAXE, P_IGU = 4, P_IDN = 8;
struct GemvArgs { uint32_t n_in, n_out, n_tok, row_bytes, x_stride, y_stride; };
struct GrArgs { uint32_t n_embd, hc, hc_lr, n_tok, apply; float eps; };
struct MoeArgs {
    uint64_t gate_off, up_off, down_off, gu_stride, d_stride;
    uint32_t gu_row, d_row, n_embd, n_ff, n_tok, topk, n_expert, gx_gu, gx_dn, pad;
};
struct GdnArgs { uint32_t h_k, h_v, channels, commit; float eps; uint32_t plane; };
struct QsaArgs { uint32_t pos0, pos_base, max_cells, n_tok; float eps; };
struct GlueArgs { uint32_t n, n_tok, n_embd, topk; };
struct LatentRoutedArgs {
    uint32_t n_tok, groups, hidden, latent, ff, topk, plan_token_entry;
    uint32_t u_row, gu_row, down_row, up_off, down_off, expert_bytes;
};
struct ArgArgs { uint32_t n_vocab, parts; };
// the routed gate/up's i-quant grids from constant memory (QUANT_CTAB, function constant 23 of quant.metal): bitwise
// the threadgroup copy's values, 3-7% faster on the M5 Max; STRATA_METAL_CTAB=0 turns it off
bool metal_ctab() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_METAL_CTAB");
        return !(e && std::string(e) == "0");
    }();
    return on;
}

int xkind(int t) {
    switch (t) {
        case ref::Q2_0: return 1;
        case ref::Q4_K: case ref::Q5_K: return 2;
        case ref::Q6_K: case ref::Q3_K: return 3;
        default: return 0;
    }
}

// a prepared activation: 4 planes of MAXT rows of n, then the per-16 sums
struct Act {
    mtl::Buffer b;
    int n = 0;
    size_t plane_bytes() const { return (size_t) 4 * MAXT * n; }
};

double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

}  // namespace

// the prompt chain's state (prompt_chain.inc): kernel libraries, the variant, buffers for chunks of C rows
struct PromptBufs {
    int lib = -1, lgdn = -1, lqsa = -1;
    bool k4 = false;
    bool v3e = false;
    bool fused = false;          // the GDN input projections as one GEMM (gdn_fused_inputs)            // the M5's expert kernels (v3): not for the trellis types, which have only v2
    int C = 0, maxrows = 0;
    uint32_t expert_tm = 16;
    double gpu_ms = 0, host_ms = 0, mtp_ms = 0;
    int gu3[4] = {}, dn3[4] = {};
    mtl::Buffer R, bo, inj[2], xn, xnh, V, loh, mixed, mixh, qkv, z, al, be, yg, ygh, gqn, gkn, gvc, ggb, gkq, go, qfull,
        kcur, vcur, iqr, ikr, qcur, qcurh, qidx, scores, qids, part, XQ, xqh, logits, ids, wts, bcnt, boff, eoff, tiles, meta,
        stok, sw, pos, XS, HS, YE, yr, gsh, ush, hh, ys, sgv, WD, WU, Wqkv, Wz, Wal, Wbe, Wout, Wq, Wk, Wv, Wiq, Wik, Wr,
        Wsg, Wshg, Wshu, Wshd, yf, Wcat, pemb, pembh, pkey, pval, keyn, qn, gated, normW, Wpk, Wpv, mixr;
    mtl::Buffer lat_w, lat_proj, lat_pooled, lat_gu16;
};

// The vision path (--vision, GENI requests), from the shared host code: the image rows of the prompt being read
// (host rows by absolute position, null = embed the token; set by prompt_run for its duration) and the M-RoPE table
// (int32 [cells][3] (t, h, w); mrope_table_set: the host view of the device table the server fills per request).
const float* const* g_embd_rows = nullptr;
const int32_t* g_mrope_host = nullptr;
const int32_t* g_mrope_dev = nullptr;

class Window {
public:
    bool init(const strata::core::ModelGeometry& g, strata::core::SessionState& ss, const std::string& shard1, int max_t,
              int64_t max_cells, std::string& err);
    bool run(int T, const int32_t* tokens, int64_t pos0, bool head, int32_t* out, std::string& err);
    bool commit(int n_keep, std::string& err);
    bool finish(int Tn, const int32_t* tokens, int64_t pos0, bool head, int32_t* out, double t0);
    bool copy_logits(int t, float* host) const;
    int64_t vocab() const { return n_vocab_; }
    bool ready() const { return cx_ != nullptr; }
    bool streaming() const { return stream_; }
    bool latent() const { return latent_on_; }
    void report_profile() const;
    // --serve's conversation checkpoints: the window's running state (the GDN recurrences and conv histories, the
    // PLE history, each QSA layer's indexer tail and tail-block key); the KV and pooled keys are positional
    bool checkpoint_save(strata::core::ConversationCheckpoint& c, std::string& err);
    bool checkpoint_restore(const strata::core::ConversationCheckpoint& c, std::string& err);
    void reset();   // a new sequence (session_zero): the running state zeroed
    // ---- batch slots (--batch N): each slot is another sequence's running state (GDN recurrence and conv history, the PLE
    // history, each QSA layer's K/V, indexer tail and pooled keys).  The live members always hold ONE sequence's state (the
    // main one between calls); a slot's lives in a stash and is swapped in for its window (activate): the buffer objects stay
    // where they are (the cached dispatches point at them), only their contents are exchanged.
    bool slots_init(const std::vector<strata::core::SessionState*>& slots, std::string& err);
    int n_slots() const { return sets_.empty() ? 0 : (int) sets_.size() - 1; }
    int set_of(const strata::core::SessionState* s) const {
        auto it = set_of_ss_.find(s);
        return it == set_of_ss_.end() ? 0 : it->second;
    }
    // one window of S one-token rows, row t = slot rows[t] at position pos[t]: each runs and commits in its own state
    bool run_slots(const int* rows, int S, const int32_t* tokens, const int64_t* pos, int32_t* out, std::string& err);
    // slot b's state <-> the main one (the GPU is idle between windows): the running state and the first `upto` cells of K/V
    bool slot_copy(bool into_slot, int b, int64_t upto, bool kv_only, std::string& err);
    void activate(int set);
    void set_max_t(int t) { max_t_ = std::min(t, MAXT); }
    // ---- the MTP draft layer (mtp.hpp's math on mac-kern's kernels; the MTP GGUF from tools/mtp_pack.py)
    bool mtp_load(const std::string& gguf, int max_t, std::string& err);
    bool mtp_prefill(const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err);
    bool mtp_draft(int T, const int32_t* tokens, int64_t p, int a, int max_steps, int32_t* drafts, float* probs,
                   float min_p, int* n_drafts, std::string& err);
    bool mtp_loaded() const { return mtp_on_; }
    // the prompt through mac-kern's prompt kernels (prompt_chain.inc)
    bool prompt(const int64_t* tokens, int64_t n, int64_t pos0,
                const std::function<bool(const float*, int64_t, int64_t, std::string&)>& on_chunk, std::string& err);

private:
    const gguf::Tensor& T(const std::string& name) const {
        const gguf::Tensor* t = f_.find(name);
        if (!t) { std::fprintf(stderr, "strata metal: the model has no tensor %s\n", name.c_str()); std::exit(2); }
        return *t;
    }
    const mtl::Pipeline* P(int lib, const std::string& fn, const mtl::Constants& c, const std::string& key) {
        auto it = pcache_.find(key);
        if (it == pcache_.end()) {
            it = pcache_.emplace(key, cx_->pipeline(lib, fn, c)).first;
            pkey_of_[&it->second] = key;
        }
        return &it->second;
    }
    void push(std::vector<mtl::Dispatch>& ds, mtl::Dispatch d) {
        d.barrier = !nb_;
        nb_ = false;
        // timing only (WRONG answers): STRATA_METAL_SKIP=a,b,... runs the dispatches whose pipeline name contains one
        // of the words as empty grids (the marginal cost of a kernel class by leaving it out)
        static const std::vector<std::string> skip = [] {
            std::vector<std::string> v;
            if (const char* e = std::getenv("STRATA_METAL_SKIP")) {
                std::string s = e, w;
                for (char c : s + ",") { if (c == ',') { if (!w.empty()) v.push_back(w); w.clear(); } else w += c; }
            }
            return v;
        }();
        if (!skip.empty()) {
            const std::string& k = pkey_of_[d.p];
            for (const auto& w : skip)
                if (k.find(w) != std::string::npos) { d.indirect = nullptr; d.grid[0] = 0; break; }
        }
        mtl::require_fit(d, "engine dispatch (graph build)");   // a kernel Metal would skip silently: stop at the build, not at the first wrong answer
        ds.push_back(std::move(d));
    }
    std::map<const mtl::Pipeline*, std::string> pkey_of_;
    void gemv(std::vector<mtl::Dispatch>& ds, const gguf::Tensor& t, const Act& x, const mtl::Buffer& y, size_t y_stride,
              int ntok, size_t y_off = 0, const mtl::Buffer* wb = nullptr);
    void xprep(std::vector<mtl::Dispatch>& ds, const mtl::Buffer& x, const Act& out, int n);
    void gr_read(std::vector<mtl::Dispatch>& ds, const std::string& pre, bool apply, const Act& out, bool final_, int ntok);
    void layer(std::vector<mtl::Dispatch>& ds, int l, int ntok, int64_t pos0);
    void ple_cpu(int T);
    // the MTP layer: the input branches + attention HC read + K/V append for `n` rows (srcR rows, host tokens) at
    // cells cell0..; with `full` (n == 1) the rest of the layer too and the draft head into mlm_/tokout_.  The
    // embedding rows go to memb_ rows erow..; tokens == nullptr: memb_ row erow already holds the embedding (put
    // there by mtp_embed on the GPU).  `slot`: the dense attention's cell list (one per draft step of a round).
    void mtp_encode(std::vector<mtl::Dispatch>& ds, const mtl::Buffer& srcR, size_t src_off, int n, const int32_t* tokens,
                    int64_t cell0, bool full, int erow = 0, int slot = 0);
    // a draft round's step j on the GPU: the pick and its probability (dout_/dprob_ [j]), and with `embed` the picked
    // token's embedding into memb_ row 0 (the next step's input)
    void mtp_pick(std::vector<mtl::Dispatch>& ds, int j, bool embed);
    void gr_read_m(std::vector<mtl::Dispatch>& ds, const std::string& pre, mtl::Buffer& Rb, const mtl::Buffer& bo,
                   const mtl::Buffer& inj_in, const mtl::Buffer& inj_out, bool apply, const Act& out, bool final_, int ntok);
    const gguf::Tensor& TM(const std::string& name) const {
        const gguf::Tensor* x = fm_.find(name);
        if (!x) { std::fprintf(stderr, "strata metal: the MTP model has no tensor %s\n", name.c_str()); std::exit(2); }
        return *x;
    }
    size_t NM(const std::string& name) const {   // the folded (1 + w) copy of an F32 norm of the MTP model
        auto it = mnorm_off_.find(name);
        if (it == mnorm_off_.end()) { std::fprintf(stderr, "strata metal: no MTP norm %s\n", name.c_str()); std::exit(2); }
        return it->second;
    }

    strata::core::SessionState* ss_ = nullptr;        // the session whose sequence the live state is (the main one, or a batch slot's)
    strata::core::SessionState* main_ss_ = nullptr;   // the main session: it owns the PLE table (read by every slot)
    gguf::File f_;
    strata::core::LatentDense latent_dense_;
    strata::core::LatentA latent_experts_;
    std::unique_ptr<mtl::Context> cx_;
    mtl::Buffer W_;
    bool latent_on_ = false;
    bool latent_grouped_u_ = false;
    bool latent_fast_verify_ = false;
    bool fast_verify(int ntok) const {
        return latent_fast_verify_ && win_head_ && !rows_ && ntok > 1;
    }
    int latent_lib_ = -1;
    mtl::Buffer latent_du_, latent_a_, latent_labels_, latent_inv_, latent_d_, latent_down_;
    mtl::Buffer latent_group_active_;
    LatentRoutedArgs latent_args_{};
    bool latent_init(std::string& err);
    void latent_routed(std::vector<mtl::Dispatch>& ds, int layer, int ntok);
    // a model wrapped in chunks (mtl::Context::wrap): the routed experts are bound at the layer's first expert tensor,
    // their offsets made relative to it; returns that binding offset (0 when W_ is one buffer or the experts stream)
    size_t moe_rebase(uint64_t& g, uint64_t& u, uint64_t& d) const {
        if (stream_ || !W_.chunks) return 0;
        const uint64_t b = std::min({g, u, d});
        g -= b; u -= b; d -= b;
        return (size_t) b;
    }
    int lib_ = -1, lgdn_ = -1, lqsa_ = -1, lsmp_ = -1, lple_ = -1;
    std::map<std::string, mtl::Pipeline> pcache_;
    bool nb_ = false;
    int max_t_ = 1;
    int64_t max_cells_ = 0;
    int64_t n_vocab_ = 0;
    float eps_ = 1e-6f;
    // buffers
    mtl::Buffer R_, bo_, inj_[2], xn_, lo_, qkv_, z_, al_, be_, yg_, logits_, ids_, wts_, plan_, h_, ye_, gsh_, ush_,
        ysh_, sg_, qfull_, kcur_, vcur_, iqr_, ikr_, qcur_, qidx_, scores_, qids_, part_, cs_, lm_, apv_, api_, tokout_,
        emb_, pkey_, pval_, qkvL_, alL_, beL_, zL_, ydump_, keyn_, qn_, gated_, normW_, pleHist_;
    Act XA_, XF_, XY_, XQ_, XH_, XE_;
    int cur_inj_ = 0;
    // state
    std::vector<mtl::Buffer> gst_, ghist_[2], kq_, vq_, ksc_, vsc_, tail_, dead_, pooled_;
    // --batch N: every one of those is a view of set 0 in a bigger buffer holding all N + 1 sets (the slots' running states side
    // by side), so ONE dispatch can address any slot (rinfo_: the strides and each row's slot / cell / history parity)
    bool bigs_ok_ = false;
    std::vector<mtl::Buffer> gstBig_, hbBig_, kqBig_, vqBig_, kscBig_, vscBig_, tailBig_, deadBig_, pooledBig_;
    mtl::Buffer pleBig_, rinfo_;
    size_t st_gst_ = 0, st_hist_ = 0, st_kq_ = 0, st_sc_ = 0, st_tail_ = 0, st_dead_ = 0, st_pooled_ = 0, st_ple_ = 0;   // bytes between sets
    bool rows_ = false;       // the window being built / run holds one token of each of rows_n_ DIFFERENT sequences (batch slots)
    int rows_n_ = 0;
    strata::core::SessionState* rows_ss_[MAXT] = {};
    int rows_set_[MAXT] = {};
    static mtl::Buffer view_of(const mtl::Buffer& big, size_t off, size_t bytes) {
        mtl::Buffer v = big;
        v.ptr = (char*) big.ptr + off;
        v.bias = big.bias + off;
        v.bytes = bytes;
        return v;
    }
    bool run_rows(const int* rows, int S, const int32_t* tokens, const int64_t* pos, int32_t* out, std::string& err);
    bool finish_rows(int S, const int32_t* tokens, int32_t* out, double t0);
    struct SeqSet {
        std::vector<mtl::Buffer> gst, gh0, gh1, kq, vq, ksc, vsc, tail, dead, pooled;
        mtl::Buffer pleHist;
        std::vector<float> ple_hist;
        std::vector<float> last_norm;
        int hp = 0, pend_fold = 0, last_T = 0;
        bool last_head = true;
        int32_t last_tokens[MAXT] = {};
        strata::core::SessionState* ss = nullptr;
    };
    std::vector<std::unique_ptr<SeqSet>> sets_;     // [0] the spare the main state is exchanged with, [b + 1] slot b
    std::map<const strata::core::SessionState*, int> set_of_ss_;
    int active_ = 0;
    void swap_state(SeqSet& o);
    void alloc_set(SeqSet& o, bool real, int idx = 0);
    int hp_ = 0;                     // which of the two history buffers holds the committed conv history (all layers)
    // commit(n) of a window of several tokens: the GDN fold of the kept prefix waits for the next window and runs at
    // the start of its command buffer (one submit and wait less a round).  Nothing else reads the GDN state between
    // the two (the drafter reads the window's rows and its own caches).  STRATA_METAL_LAZY_FOLD=0: at commit.
    int pend_fold_ = 0;
    // gdn_step's partial commit (mac-kern 56526aed, commit = 0x100 | k): a verify window of T > 1 commits its first
    // row in place (it is always kept), so commit(1) needs no fold and commit(n) folds rows 1..n-1 only; a window
    // without the head (the prompt) commits all its rows in the step.  Opt-in (STRATA_METAL_PARTIAL=1): at --spec 2
    // it measured 0.3-0.8% slower on the M5 Max (the in-place state write costs every window, the fold it saves
    // only the rounds with no accepted draft).  Off: the window commits nothing and commit(n) folds n rows.
    bool partial_ = std::getenv("STRATA_METAL_PARTIAL") && std::string(std::getenv("STRATA_METAL_PARTIAL")) == "1";
    bool win_head_ = true;      // the window being encoded has the head
    bool last_head_ = true;     // the last window had it
    // token_chain's --fuse (mac-kern): the GDN step writes the prepared y planes itself (no xprep), and a window whose
    // context is within the selection budget skips the QSA score and top-k (the cell list is the identity: qids_
    // holds it from init, restored after a window that selected).  STRATA_METAL_FUSE=0: off.
    bool fuse_ = !(std::getenv("STRATA_METAL_FUSE") && std::string(std::getenv("STRATA_METAL_FUSE")) == "0");
    bool qdirty_ = false;
    // mac-kern's router_route (3f9ca996): the BF16 router GEMV and moe_route in one dispatch, beside the shared
    // expert's GEMVs (bitwise).  Opt-in (STRATA_METAL_ROUTE=1): in this window, where the router GEMV already runs
    // beside the shared GEMVs, it measured no faster (T2 GPU 13.035 vs 13.027 ms; the fused kernel alone 0.615 ms vs
    // 0.566 for the two) and 0.3-0.5% slower end to end on the M5 Max.
    bool routef_ = std::getenv("STRATA_METAL_ROUTE") && std::string(std::getenv("STRATA_METAL_ROUTE")) == "1";
    mtl::Buffer rcounter_;
    // STRATA_METAL_KPROF=T (test tooling): once, after the 20th cached window of T tokens with the head, the GPU time
    // of each kernel class alone (the window's own dispatches of that pipeline + weight role, run by themselves)
    void kprof(const std::vector<mtl::Dispatch>& ds, int Tn);
    void fold(std::vector<mtl::Dispatch>& ds, int n_keep, int hp);
    // the encoded window per (T, head, hp_): built once, then only the positions are patched (the host's encode cost)
    struct Cached {
        std::vector<mtl::Dispatch> ds;
        std::vector<int> cuts;                  // SSD streaming: the segment ends (one per layer, before its experts)
        std::vector<std::pair<int, int>> qsa;   // (dispatch, kind): 0 QsaArgs n_tok = T, 1 the merge (MAXT), 2 the scores
    };
    std::map<int, Cached> wcache_;
    std::vector<std::pair<int, int>>* rec_ = nullptr;
    // ---- SSD streaming (stream.h): the dense weights in W_, the routed experts in the arena
    std::unique_ptr<ExpertStream> st_;
    bool stream_ = false, st_pred_ = false;
    std::vector<int>* cuts_ = nullptr;
    mtl::Buffer predL_, predL2_;     // 48 x MAXT x 512: layer l + 1's (l + 2's) router logits from layer l's FFN input
    int seg_T_ = 0;
    bool stream_arena(std::string& err);
    double run_ds(Cached& c);
    // prompts on the matrix units (prompt_chunk below)
    struct PromptGemm;
    PromptGemm* pg_ = nullptr;
    const int32_t* cut_ids_ = nullptr;    // the routed ids the handshake reads (a prompt chunk's; else ids_)
    const mtl::Buffer* mtp_src_ = nullptr;   // the rows the draft layer's prompt K/V read (a prompt chunk's; else R_)
    double pg_ms_ = 0;
    int64_t pg_tok_ = 0;
public:
    bool prompt_gemm_on(int64_t n) const;
    bool prompt_gemm_init(std::string& err);
    bool prompt_chunk(const int32_t* tokens, int Tn, int64_t pos0, std::string& err);
    int prompt_chunk_rows() const;
    void prompt_region(bool hold) { if (stream_ && st_ && st_->ready()) st_->hold_region(hold); }
    void stream_report(FILE* f) const {
        if (!st_) return;
        st_->report(f);
        for (int t = 1; t <= MAXT; ++t)
            if (win_n_[t])
                std::fprintf(f, "stream windows T=%d     %d (%s): %.2f ms each (GPU incl. the host's turns %.2f ms)\n", t, win_n_[t],
                             t == 8 ? "prompt rows" : "decode", win_ms_[t] / win_n_[t], win_gpu_[t] / win_n_[t]);
    }
    int win_n_[MAXT + 1] = {};
    double win_ms_[MAXT + 1] = {}, win_gpu_[MAXT + 1] = {};
    void at_cut(int s) {
        st_->at_layer(s, seg_T_, TOPK, cut_ids_ ? cut_ids_ : ids_.as<int32_t>(), st_pred_ && !cut_ids_ ? predL_.as<float>() + (size_t) s * MAXT * 512 : nullptr,
                      predL2_.ptr && !cut_ids_ ? predL2_.as<float>() + (size_t) s * MAXT * 512 : nullptr);
    }
private:
    std::vector<float> ple_hist_;    // NG_HIST x hc_dim normalized rows, row-fastest
    bool ple_on_ = false;
    int last_T_ = 0;
    double last_gpu_ms_ = 0;
    int last_n_disp_ = 0;
    int32_t last_tokens_[MAXT] = {};
    std::vector<float> last_norm_;   // the window's normalized PLE rows (T x hc_dim), for commit
    // STRATA_METAL_LOGPOS=<path> (test tooling, one-token windows): per decode position the log-probability of the
    // token that came next (the next window's input: with --spec-follow, the forced text) and the top 64, in the
    // KL harness's row format (phaseA-tests cuda_p0/cuda_kl.py load_lp): pos, target, lp, argmax, 4 x 0, id:lp ...
    std::FILE* logpos_ = nullptr;
    bool lp_pend_ = false;
    int64_t lp_pos_ = 0, lp_first_ = -1;
    int32_t lp_arg_ = 0;
    double lp_lse_ = 0.0;
    std::vector<float> lp_row_;
    std::vector<std::pair<float, int32_t>> lp_top_;
    void logpos(int Tn, const int32_t* tokens, int64_t pos0, bool head);
    // STRATA_METAL_PROFILE=1: where a window's time goes, per window size (printed when the window goes away)
    struct Prof { int64_t n = 0; double pre = 0, rg = 0, gpu = 0, fin = 0, all = 0; };
    std::map<int, Prof> prof_;
    Prof cprof_, dprof_;   // commit (folds), the draft rounds
    double tp_pre_ = 0, tp_rg_ = 0;
public:
    ~Window();
    void prof_add(bool draft, double ms) { Prof& x = draft ? dprof_ : cprof_; ++x.n; x.all += ms; }
private:
    // the Q8_0 copies of the hyper-connection matrices (the default; STRATA_HC_Q8=0: the BF16 originals)
    // Gyro (agentionai's Hadamard-folded trellis experts, native_experts.txt v5 + hadamard.txt): the gate/up input is
    // H (s * x) and the down input H h, per block of 128; XR_ holds the rotated rows, hsx_/hsh_ the +/-1 signs
    bool had_on_ = false;
    int had_block_ = 128;
    bool gyro_dense_ = false;   // a trellis-expert file: its dense K-quants get their own GEMV tiles (gemv() below)
    bool had_gu(int l) const { return had_on_ && strata::kernels::cpu::hadamard_spec().gu(l); }
    bool had_dn(int l) const { return had_on_ && strata::kernels::cpu::hadamard_spec().down(l); }
    mtl::Buffer XR_, hsx_, hsh_;
    // one rotation dispatch: rows of `width` values (a multiple of 128) in place or from x to y; f16 = half rows
    void had_rows(std::vector<mtl::Dispatch>& ds, int lib, const std::string& pfx, const mtl::Buffer& x, size_t x_off,
                  const mtl::Buffer& y, size_t y_off, int rows, int width, const mtl::Buffer* signs, bool f16);
    // the prompt's h rows (half, padded rows of the expert tiles, their count in meta[8]): rotated in place, one
    // threadgroup a row through the indirect grid
    void had_rows_ind(std::vector<mtl::Dispatch>& ds, int lib, const std::string& pfx, const mtl::Buffer& h, const mtl::Buffer& meta);
    bool hc_q8_ = false;
    mtl::Buffer Q8_;
    std::map<std::string, size_t> q8off_;
    // MTP
    bool mtp_on_ = false;
    int lmtp_ = -1;
    gguf::File fm_;
    mtl::Buffer Wm_, mnorm_, mR_, memb_, men_, me2_, mhn_, mh2_, midx_, mbo_, minj_[2], mqids_, mlm_;
    std::map<std::string, size_t> mnorm_off_;
    std::vector<mtl::Buffer> mkq_, mvq_, mksc_, mvsc_, mtail_, mdead_, mpooled_;
    // the draft head over a token subset (data/draft_vocab.bin, as the CUDA drafter): its rows gathered from the head
    mtl::Buffer mhead_;
    gguf::Tensor mhead_t_;
    std::vector<int32_t> dvocab_;
    // the draft layer's BF16 matrices in Q8_0 (half the bytes a draft step reads; drafts only: the verify window
    // decides every token).  STRATA_METAL_MTP_Q8=0 keeps BF16.
    mtl::Buffer mq8_;
    std::map<std::string, gguf::Tensor> mq8t_;
    const mtl::Buffer& MB(const std::string& name) const { return mq8t_.count(name) ? mq8_ : Wm_; }
    const gguf::Tensor& TQ(const std::string& name) const {
        auto it = mq8t_.find(name);
        return it != mq8t_.end() ? it->second : TM(name);
    }
    Act XEn_, XHn_;
    PromptBufs pb_;
    mtl::Buffer mpemb_;                    // the drafter's prompt embedding rows (mtp_prefill, one command buffer)
    size_t mpemb_rows_ = 0;
    const mtl::Buffer* membp_ = nullptr;   // mtp_encode's embedding rows (nullptr: memb_)
    const mtl::Buffer* mpsrc_ = nullptr;   // the drafter's prompt rows (nullptr: R_)
    bool coop_k4();
    // the vision path: the embedding row of an absolute position (an image row, or null), the M-RoPE table rows a
    // window reads copied into mrope_ (qsa_prep's buffer 20, function constant 24)
    const float* img_row(int64_t pos) const { return g_embd_rows ? g_embd_rows[pos] : nullptr; }
    mtl::Buffer mrope_;
    // the PLE conv kernel as F16 (ple_conv_add reads half): the file's own F16 tensor, or an F32 one (Unsloth's)
    // narrowed with round-to-nearest-even as the host loader does
    mtl::Buffer plekw_;
    bool plekw_own_ = false;
    mtl::Arg kw_arg() { return plekw_own_ ? mtl::Arg::B(plekw_) : mtl::Arg::B(W_, T("blk.1.ple_conv1d.weight").offset); }
    const uint16_t* kw_host() {
        return plekw_own_ ? plekw_.as<uint16_t>()
                          : (const uint16_t*) ((const uint8_t*) W_.ptr + T("blk.1.ple_conv1d.weight").offset);
    }
    bool mrope_on() const { return g_mrope_host != nullptr; }
    void mrope_sync(int64_t c0, int64_t n) {
        if (!mrope_on()) return;
        const int64_t a = std::max<int64_t>(0, c0 & ~(int64_t) 3), b = std::min<int64_t>(max_cells_, c0 + n);
        if (b > a) std::memcpy(mrope_.as<int32_t>() + a * 3, g_mrope_host + a * 3, (size_t) (b - a) * 12);
    }
    mtl::Constants qsa_c() const { return mrope_on() ? mtl::Constants().b(24, true) : mtl::Constants(); }
    std::string qsa_k() const { return mrope_on() ? "qsa_prep#mrope" : "qsa_prep"; }
    int m5_layout_ = -1;   // the M5's matmul2d layout (coop_k4), asked once: the M4's tiles differ
    bool m5_layout() { if (m5_layout_ < 0) m5_layout_ = coop_k4() ? 1 : 0; return m5_layout_ == 1; }
    bool pc_setup(int C);
    bool pc_chunk(const int64_t* tokens, int nT, int64_t pos0, std::string& err);
    // the draft round in one command buffer (msl/mtp_chain.metal): STRATA_METAL_MTP_CHAIN=0 runs a command buffer
    // per step with the host's pick and embedding (as before)
    int lchain_ = -1;
    bool chain_ = false;
    mtl::Buffer dout_, dprob_, posv_, dmap_;
};

namespace {
// A repack (metal_repack.py: <file>.repack.lock) or setup's model preparation (.preparing.lock) beside the GPU's file holds a lock
// with its pid.  The repack and the pack move 100+ GB through the disk and one holds up to 20 GB of memory: an engine of 70-90 GB
// beside them is what froze the 128 GB M5 Max twice (watchdog panics, 2026-10-13).  Returns the pid of a live one, else 0.
// STRATA_METAL_IGNORE_REPACK=1 starts anyway (your choice).
long repack_pid_running(const std::string& model_file) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path dir = fs::path(model_file).parent_path().empty() ? fs::path(".") : fs::path(model_file).parent_path();
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        const std::string n = e.path().filename().string();
        const bool lock = n == ".preparing.lock" || (n.size() > 12 && n.compare(n.size() - 12, 12, ".repack.lock") == 0);
        if (!lock) continue;
        std::FILE* f = std::fopen(e.path().string().c_str(), "r");
        long pid = 0;
        if (f) { if (std::fscanf(f, "%ld", &pid) != 1) pid = 0; std::fclose(f); }
        if (pid > 0 && pid != (long) getpid() && (kill((pid_t) pid, 0) == 0 || errno == EPERM)) return pid;
    }
    return 0;
}
}  // namespace

bool Window::init(const strata::core::ModelGeometry& g, strata::core::SessionState& ss, const std::string& shard1,
                  int max_t, int64_t max_cells, std::string& err) {
    ss_ = &ss;
    main_ss_ = &ss;
    max_t_ = std::min(max_t, MAXT);
    max_cells_ = max_cells;
    if (g.n_embd != N || g.hc != HC || g.hc_lr != LR || g.n_ff != NFF || g.n_expert != NEXP || g.n_layers != 48) {
        err = "metal window: the geometry differs from Qwen3.8-Flash-Next's";
        return false;
    }
    eps_ = 1e-6f;
    const double t0 = now_ms();
    if (const long rp = repack_pid_running(shard1); rp > 0 && !(std::getenv("STRATA_METAL_IGNORE_REPACK") && std::string(std::getenv("STRATA_METAL_IGNORE_REPACK")) == "1")) {
        err = "metal window: a model repack or setup's preparation is running here (pid " + std::to_string(rp) + "): the engine is not started "
              "beside it (it froze a 128 GB Mac twice); wait for it to finish, or set STRATA_METAL_IGNORE_REPACK=1";
        return false;
    }
    if (!f_.open(shard1, err)) return false;
    cx_ = std::make_unique<mtl::Context>();
    if (!latent_init(err)) return false;
    stream_ = !latent_on_ && ExpertStream::wanted(f_, *cx_);
    if (stream_) {
        // the model does not fit: the dense weights are read into memory below, the experts stream from the SSD
        st_ = std::make_unique<ExpertStream>();
        if (!st_->open(f_, err)) return false;
    } else {
        {   // read ahead the weights, not the n-gram table: in a Gyro file it is 28.8 GB in the middle of the one file, read by the host from
            // the SSD a few rows at a time (src/ngram/ple_reader.cpp), and must not fill the page cache
            const gguf::Tensor* ple = f_.find("per_layer_token_embd.weight");
            if (ple == nullptr) {
                madvise(f_.map, f_.map_bytes, MADV_WILLNEED);
            } else {
                const size_t page = 16384;
                const size_t lo = (size_t) ple->offset / page * page, hi = ((size_t) (ple->offset + ple->bytes) + page - 1) / page * page;
                if (lo > 0) madvise(f_.map, lo, MADV_WILLNEED);
                if (hi < f_.map_bytes) madvise((uint8_t*) f_.map + hi, f_.map_bytes - hi, MADV_WILLNEED);
            }
        }
        // the chunks of a model bigger than one Metal buffer overlap by the largest range bound at one offset: a tensor,
        // or a layer's routed experts (gate, up and down, bound together at the first of them: moe_rebase)
        size_t span = 0;
        // (not the n-gram table: the host reads it from the SSD, the GPU never binds it - in a Gyro file it is 28.8 GB of the one file)
        for (const auto& kv : f_.tensors)
            if (kv.first != "per_layer_token_embd.weight") span = std::max<size_t>(span, kv.second.bytes);
        for (int l = 0; l < 48; ++l) {
            const std::string b = "blk." + std::to_string(l) + ".";
            const gguf::Tensor *g = f_.find(b + "ffn_gate_exps.weight"), *u = f_.find(b + "ffn_up_exps.weight"),
                               *d = f_.find(b + "ffn_down_exps.weight");
            if (!g || !u || !d) continue;
            const uint64_t lo = std::min({g->offset, u->offset, d->offset});
            const uint64_t hi = std::max({g->offset + g->bytes, u->offset + u->bytes, d->offset + d->bytes});
            span = std::max<size_t>(span, hi - lo);
        }
        W_ = cx_->wrap(f_.map, f_.map_bytes, span);
        if (W_.handle == nullptr) { err = "metal window: cannot wrap " + shard1 + " as a Metal buffer"; return false; }
    }
    n_vocab_ = (int64_t) T("output.weight").shape[1];
    if (const gguf::Tensor* e0 = f_.find("blk.0.ffn_gate_exps.weight")) gyro_dense_ = e0->type >= 144;
    lib_ = cx_->compile({"quant.metal", "units.metal", "moe.metal", "gemv.metal", "router.metal", "gr.metal", "gr2.metal",
                         "glue.metal"});
    rcounter_ = cx_->alloc(16);
    lgdn_ = cx_->compile({"gdn.metal"}, "#define NO_GGML_COMMON 1\n");
    lqsa_ = cx_->compile({"qsa.metal", "qsa2.metal"}, "#define NO_GGML_COMMON 1\n");
    lsmp_ = cx_->compile({"sample.metal"}, "#define NO_GGML_COMMON 1\n");
    lple_ = cx_->compile({strata_metal::resource("STRATA_METAL_ENGINE_KERNELS", "engine-msl", STRATA_METAL_ENGINE_KERNELS_DEFAULT) + "/ple.metal"}, "#define NO_GGML_COMMON 1\n");

    const int MAXC = (int) std::max<int64_t>(4096, (max_cells + 64) / 4 * 4 + 64), MB = MAXC / 4 + 2;
    max_cells_ = MAXC;
    auto A = [&](size_t bytes) { return cx_->alloc(bytes); };
    R_ = A(4 * MAXT * HC * N); bo_ = A(4 * MAXT * N); inj_[0] = A(4 * MAXT * HC); inj_[1] = A(4 * MAXT * HC);
    xn_ = A(4 * MAXT * HC * N); lo_ = A(4 * MAXT * LR); qkv_ = A(4 * MAXT * 10240); z_ = A(4 * MAXT * 6144);
    al_ = A(4 * MAXT * 48); be_ = A(4 * MAXT * 48); yg_ = A(4 * MAXT * 6144); logits_ = A(4 * MAXT * 512);
    ids_ = A(4 * MAXT * TOPK); wts_ = A(4 * MAXT * TOPK); plan_ = A(4 * P_INTS); h_ = A(4 * MAXT * TOPK * NFF);
    ye_ = A(4 * MAXT * TOPK * N); gsh_ = A(4 * MAXT * NFF); ush_ = A(4 * MAXT * NFF); ysh_ = A(4 * MAXT * N);
    sg_ = A(4 * MAXT * 4); qfull_ = A(4 * MAXT * 12288); kcur_ = A(4 * MAXT * 512); vcur_ = A(4 * MAXT * 512);
    iqr_ = A(4 * MAXT * 512); ikr_ = A(4 * MAXT * 128); qcur_ = A(4 * MAXT * 24 * 256); qidx_ = A(4 * MAXT * 4 * 128);
    scores_ = A(4 * (size_t) MAXT * MB); qids_ = A(4 * MAXT * 2051); part_ = A(4 * (size_t) MAXT * 65 * 24 * 258);
    cs_ = A(8 * (size_t) (MAXC + 16) * 32); lm_ = A(4 * (size_t) MAXT * n_vocab_); apv_ = A(4 * MAXT * 120);
    api_ = A(4 * MAXT * 120); tokout_ = A(4 * MAXT); emb_ = A(4 * MAXT * N); pkey_ = A(4 * MAXT * 10240);
    pval_ = A(4 * MAXT * N);
    // the GDN layers' window inputs, kept per layer so commit() can fold the accepted prefix into the state
    qkvL_ = A(4 * (size_t) 36 * MAXT * 10240); alL_ = A(4 * (size_t) 36 * MAXT * 48); beL_ = A(4 * (size_t) 36 * MAXT * 48);
    zL_ = A(4 * (size_t) 36 * MAXT * 6144); ydump_ = A(4 * MAXT * 6144);
    keyn_ = A(4 * MAXT * 10240); qn_ = A(4 * MAXT * 10240); gated_ = A(4 * MAXT * 10240); normW_ = A(4 * MAXT * 10240);
    mrope_ = A(12 * (size_t) (MAXC + 16));
    bigs_ok_ = g_slot_cap > 0;
    const size_t NSET = 1 + (size_t) std::max(0, g_slot_cap);
    auto R256 = [](size_t b) { return (b + 255) / 256 * 256; };
    st_gst_ = R256(4 * 128 * 48 * 128); st_hist_ = R256(4 * 10240 * 3); st_kq_ = R256((size_t) MAXC * 2 * 256);
    st_sc_ = R256(2 * (size_t) MAXC * 2 * 4); st_tail_ = R256(4 * 4 * 128); st_dead_ = R256(4 * 128);
    st_pooled_ = R256(4 * (size_t) MB * 128); st_ple_ = R256(4 * 9 * 10240);
    // the set-0 view of a new buffer of `bytes` per set (or a plain buffer when there are no batch slots)
    auto SV = [&](std::vector<mtl::Buffer>& bigs, size_t bytes, size_t stride, size_t per = 1, size_t which = 0) {
        if (!bigs_ok_) { bigs.push_back(mtl::Buffer()); return A(bytes); }
        if (which == 0) bigs.push_back(A(stride * per * NSET));
        return view_of(bigs.back(), which * stride, bytes);
    };
    if (bigs_ok_) { pleBig_ = A(st_ple_ * NSET); pleHist_ = view_of(pleBig_, 0, 4 * 9 * 10240); } else pleHist_ = A(4 * 9 * 10240);
    rinfo_ = A(4 * (8 + 4 * MAXT));
    auto act = [&](int n) { Act a; a.n = n; a.b = A(4 * ((size_t) 4 * MAXT * n + (size_t) MAXT * n / 16)); return a; };
    XA_ = act(N); XF_ = act(N); XY_ = act(6144); XQ_ = act(6144); XH_ = act(NFF); XE_ = act(N);
    {   // Gyro (native_experts.txt v5 + hadamard.txt): the rotation's signs and the rotated gate/up input
        const auto& hs = strata::kernels::cpu::hadamard_spec();
        had_on_ = hs.any();
        if (had_on_) {
            had_block_ = hs.block;
            if (had_block_ != 128) {
                err = "metal window: the pack's Hadamard block is " + std::to_string(had_block_) + "; the Metal rotation takes 128";
                return false;
            }
            XR_ = A(4 * MAXT * N);
            auto upload = [&](int w, mtl::Buffer& b) {
                if (const float* sg = strata::kernels::cpu::hadamard_signs(w)) {
                    b = A(4 * (size_t) w);
                    std::memcpy(b.ptr, sg, 4 * (size_t) w);
                }
            };
            upload(N, hsx_);
            upload(NFF, hsh_);
            std::fprintf(stderr, "strata metal: Hadamard-folded experts (block %d): the gate/up and down inputs are rotated on the GPU\n",
                         had_block_);
        }
    }
    // RoPE: cos/sin [pos][32] for the first 64 dims, NEOX pairs (i, i + 32), in double as build_rope_table
    const double base = f_.meta_num.count("qwen4exp.rope.freq_base") ? f_.meta_num["qwen4exp.rope.freq_base"] : 1e7;
    for (int p = 0; p < MAXC + 16; ++p)
        for (int i = 0; i < 32; ++i) {
            const double th = (double) p * std::pow(base, -2.0 * i / 64.0);
            cs_.as<float>()[(p * 32 + i) * 2] = (float) std::cos(th);
            cs_.as<float>()[(p * 32 + i) * 2 + 1] = (float) std::sin(th);
        }
    for (int t = 0; t < MAXT; ++t)
        for (int c = 0; c < 2051; ++c) qids_.as<int>()[t * 2051 + c] = c;
    for (int l = 0; l < 48; ++l) {
        if (l % 4 != 3) {
            gst_.push_back(SV(gstBig_, 4 * 128 * 48 * 128, st_gst_));
            ghist_[0].push_back(SV(hbBig_, 4 * 10240 * 3, st_hist_, 2, 0));   // [set][parity]: the two parities side by side
            ghist_[1].push_back(SV(hbBig_, 4 * 10240 * 3, st_hist_, 2, 1));
        } else {
            kq_.push_back(SV(kqBig_, (size_t) MAXC * 2 * 256, st_kq_));
            vq_.push_back(SV(vqBig_, (size_t) MAXC * 2 * 256, st_kq_));
            ksc_.push_back(SV(kscBig_, 2 * (size_t) MAXC * 2 * 4, st_sc_));
            vsc_.push_back(SV(vscBig_, 2 * (size_t) MAXC * 2 * 4, st_sc_));
            tail_.push_back(SV(tailBig_, 4 * 4 * 128, st_tail_));
            dead_.push_back(SV(deadBig_, 4 * 128, st_dead_));
            pooled_.push_back(SV(pooledBig_, 4 * (size_t) MB * 128, st_pooled_));
        }
    }
    // ---- the HC read in Q8_0 (the owner's default on Metal, as STRATA_HC_Q8 on gfx12): the BF16 down / up / inject
    // matrices quantized once at load (ggml's q8_0 rule, ref_quant), read by gr_down_q8 / gr_up_q8
    {
        const char* e = std::getenv("STRATA_HC_Q8");
        hc_q8_ = !(e && std::string(e) == "0");
    }
    {
        const double tq = now_ms();
        size_t tot = 0;
        std::vector<std::string> names;
        for (const auto& kv : f_.tensors) {
            const gguf::Tensor& x = kv.second;
            // BF16 (when the Q8_0 read is on) and, whatever the setting, the Gyro file's Q6_K / IQ4_NL ones: the decode window's
            // hyper-connection kernels read BF16 or Q8_0 only
            const bool conv = (x.type == ref::BF16 && hc_q8_) || x.type == ref::Q6_K || x.type == ref::IQ4_NL;
            if (!conv || x.name.find("hc_") == std::string::npos || x.name.find("_norm") != std::string::npos) continue;
            uint64_t ne = 1;
            for (uint64_t d : x.shape) ne *= d;
            q8off_[x.name] = tot;
            names.push_back(x.name);
            tot += ne / 32 * 34;
        }
        Q8_ = cx_->alloc((tot + 4095) / 4096 * 4096 + 4096);   // + a page: a file with Q8_0 HC already has no copies
        std::vector<std::thread> th;
        const unsigned nth = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
        for (unsigned k = 0; k < nth; ++k)
            th.emplace_back([&, k] {
                std::vector<float> tmp;
                std::vector<uint8_t> raw;
                for (size_t i = k; i < names.size(); i += nth) {
                    const gguf::Tensor& x = *f_.find(names[i]);
                    uint64_t ne = 1;
                    for (uint64_t d : x.shape) ne *= d;
                    tmp.resize(ne);
                    const uint8_t* src = (const uint8_t*) f_.map + x.offset;
                    if (stream_) {   // F_NOCACHE reads: the page cache is kept for the arena
                        raw.resize(x.bytes);
                        if (!st_->read(raw.data(), x.bytes, x.offset)) { std::fprintf(stderr, "strata metal: read failed\n"); std::exit(6); }
                        src = raw.data();
                    }
                    ref::dequant(x.type, src, tmp.data(), ne);
                    // a Q6_K / IQ4_NL matrix (Gyro) goes through BF16 first: the form the pack gives the CUDA engine
                    // (tools/iq_pack.py --compat-bf16), so the Q8_0 copy below is made from the same numbers
                    if (x.type != ref::BF16)
                        for (uint64_t i = 0; i < ne; ++i) tmp[i] = ref::bf16_to_f32(ref::f32_to_bf16(tmp[i]));
                    ref::quantize_q8_0(tmp.data(), ne, Q8_.as<uint8_t>() + q8off_[names[i]]);
                }
            });
        for (auto& x : th) x.join();
        std::fprintf(stderr, "strata metal: HC read in Q8_0 (STRATA_HC_Q8=0: BF16): %zu matrices, %.0f MB, made in %.0f ms\n",
                     names.size(), tot / 1e6, now_ms() - tq);
    }
    if (stream_) {
        std::vector<std::string> skip;   // the BF16 originals of the Q8_0 HC copies are not read again
        if (hc_q8_)
            for (const auto& kv : q8off_) skip.push_back(kv.first);
        if (!st_->load_dense(f_, *cx_, skip, W_, err)) return false;
        const char* e = std::getenv("STRATA_METAL_PREFETCH");
        st_pred_ = !(e && std::atoi(e) <= 0);
        if (st_pred_) predL_ = A(4 * (size_t) 48 * MAXT * 512);
        if (st_pred_ && st_->ahead2() > 0) predL2_ = A(4 * (size_t) 48 * MAXT * 512);
    }
    ple_on_ = ss.ple.ready() && f_.find("blk.1.ple_key.weight") != nullptr;
    if (ple_on_ && T("blk.1.ple_conv1d.weight").type == ref::F32) {
        const gguf::Tensor& k = T("blk.1.ple_conv1d.weight");
        const size_t ne = k.bytes / 4;
        plekw_ = cx_->alloc((ne * 2 + 4095) / 4096 * 4096);
        const float* src = (const float*) ((const uint8_t*) W_.ptr + k.offset);
        uint16_t* dst = plekw_.as<uint16_t>();
        for (size_t i = 0; i < ne; ++i) dst[i] = ref::f32_to_f16(src[i]);
        plekw_own_ = true;
    }
    ple_hist_.assign((size_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM, 0.0f);
    last_norm_.assign((size_t) MAXT * strata::kernels::NG_HC_DIM, 0.0f);
    std::fprintf(stderr, "strata metal: decode window on %s: %s %s (%.1f GB), %d cells, PLE %s, ready in %.0f ms\n",
                 cx_->device_name().c_str(), shard1.c_str(), stream_ ? "streamed from the SSD (experts)" : "wrapped",
                 f_.file_bytes / 1e9, MAXC, ple_on_ ? "on" : "OFF", now_ms() - t0);
    return true;
}

bool Window::latent_init(std::string& err) {
    const auto& layout = strata::kernels::cpu::expert_layout();
    if (!layout.native || layout.fmt.empty() || layout.fmt.front().n_embd == N) return true;
    constexpr uint32_t groups = 8, layers = 48;
    const auto declared_width = layout.fmt.front().n_embd;
    if (declared_width != 1280 && declared_width != 1792) {
        err = "metal latent: expected latent width 1280 or 1792";
        return false;
    }
    const auto width = uint32_t(declared_width);
    const char* enabled = std::getenv("STRATA_METAL_LATENT");
    const char* dense = std::getenv("STRATA_LATENT_DENSE");
    const char* experts = std::getenv("STRATA_LATENT_A");
    if (!enabled || std::strcmp(enabled, "1") || !dense || !*dense || !experts || !*experts) {
        err = "metal latent: set STRATA_METAL_LATENT=1, STRATA_LATENT_DENSE and STRATA_LATENT_A";
        return false;
    }
    if (layout.fmt.size() != layers || layout.n_expert != NEXP) {
        err = "metal latent: expected 48 layers and 512 experts";
        return false;
    }
    const size_t gu_row = ref::row_bytes(ref::Q3_K, width);
    const size_t down_row = ref::row_bytes(ref::IQ4_NL, NFF);
    const size_t up_off = gu_row * NFF, down_off = 2 * up_off;
    const size_t blob = down_off + down_row * width;
    std::vector<float> inverse;
    inverse.reserve(layers * width);
    for (const auto& fmt : layout.fmt) {
        if (fmt.n_embd != width || fmt.n_ff != NFF || fmt.gu_type != ref::Q3_K ||
            fmt.d_type != ref::IQ4_NL || fmt.gu_row != gu_row || fmt.d_row != down_row ||
            fmt.up_off != up_off || fmt.down_off != down_off || fmt.bytes != blob ||
            !fmt.gu_colscale || fmt.had_gu || fmt.had_d) {
            err = "metal latent: expected a uniform colscaled Q3_K/IQ4_NL expert layout";
            return false;
        }
        for (uint32_t i = 0; i < width; ++i) {
            const float v = fmt.gu_colscale[i];
            if (!std::isfinite(v) || v <= 0.0f) {
                err = "metal latent: invalid inverse scale";
                return false;
            }
            inverse.push_back(v);
        }
    }
    if (!latent_dense_.open(dense, layers, groups, N, width, err)) return false;
    if (latent_dense_.fmt() != 4) {
        err = "metal latent: D/U must be Q4_0 (STRATA_LATENT_DENSE_FMT=4)";
        return false;
    }
    if (!latent_experts_.open(experts, layers, NEXP, blob, err)) return false;
    const char* labels = std::getenv("STRATA_LATENT_LABELS");
    const auto label_path = labels && *labels ? std::filesystem::path(labels)
        : std::filesystem::path(experts).parent_path() / "labels.bin";
    std::ifstream lf(label_path, std::ios::binary);
    std::vector<int32_t> groups_by_expert(layers * NEXP);
    const size_t label_bytes = groups_by_expert.size() * sizeof(int32_t);
    if (!lf.read(reinterpret_cast<char*>(groups_by_expert.data()), label_bytes) ||
        lf.peek() != std::char_traits<char>::eof() ||
        std::any_of(groups_by_expert.begin(), groups_by_expert.end(),
                    [](int32_t group) { return group < 0 || group >= int32_t(groups); })) {
        err = "metal latent: invalid labels at " + label_path.string();
        return false;
    }
    const size_t layer_du = latent_dense_.d_bytes() + latent_dense_.u_bytes();
    const size_t layer_a = NEXP * blob;
    latent_du_ = cx_->wrap(const_cast<uint8_t*>(latent_dense_.d(0)), latent_dense_.bytes(), layer_du);
    latent_a_ = cx_->wrap(const_cast<uint8_t*>(latent_experts_.blob(0, 0)), layers * layer_a, layer_a);
    if (!latent_du_.handle || !latent_a_.handle) {
        err = "metal latent: cannot wrap the D/U or expert file";
        return false;
    }
    latent_labels_ = cx_->alloc(label_bytes);
    std::memcpy(latent_labels_.ptr, groups_by_expert.data(), label_bytes);
    latent_inv_ = cx_->alloc(inverse.size() * sizeof(float));
    std::memcpy(latent_inv_.ptr, inverse.data(), latent_inv_.bytes);
    latent_d_ = cx_->alloc(sizeof(float) * MAXT * groups * width);
    latent_down_ = cx_->alloc(sizeof(float) * MAXT * TOPK * width);
    const char* grouped = std::getenv("STRATA_METAL_LATENT_GROUP_U");
    latent_grouped_u_ = grouped && !std::strcmp(grouped, "1");
    if (latent_grouped_u_) latent_group_active_ = cx_->alloc(sizeof(uint32_t) * MAXT * groups);
    latent_args_ = {0, groups, N, width, NFF, TOPK, P_TOKENT, uint32_t(latent_dense_.u_row_bytes()),
                    uint32_t(gu_row), uint32_t(down_row), uint32_t(up_off), uint32_t(down_off), uint32_t(blob)};
    const char* verify_fast = std::getenv("STRATA_METAL_LATENT_VERIFY_FAST");
    latent_fast_verify_ = latent_grouped_u_ && verify_fast && std::strcmp(verify_fast, "1") == 0;
    latent_lib_ = cx_->compile({"quant.metal", "units.metal", "latent_routed.metal"});
    latent_on_ = true;
    std::fprintf(stderr, "strata metal: l%u Q3_K/IQ4_NL experts and Q4_0 D/U ready\n", width);
    if (latent_grouped_u_) std::fprintf(stderr, "strata metal: pool routed rows by group before U\n");
    return true;
}

void Window::latent_routed(std::vector<mtl::Dispatch>& ds, int layer, int ntok) {
    const auto& base = latent_args_;
    auto args = base;
    args.n_tok = ntok;
    const size_t du_offset = layer * (latent_dense_.d_bytes() + latent_dense_.u_bytes());
    const size_t a_offset = layer * size_t(NEXP) * args.expert_bytes;
    const size_t label_offset = layer * NEXP * sizeof(int32_t);
    const size_t scale_offset = layer * args.latent * sizeof(float);
    const uint32_t entries = ntok * args.topk;
    static const char* decode_tiles = std::getenv("STRATA_METAL_LATENT_DECODE_TILES");
    const bool narrow = ntok == 1 && decode_tiles && std::strcmp(decode_tiles, "1") == 0;
    const bool verify = fast_verify(ntok);
    const uint32_t gate_rows = verify ? 32 : narrow ? 16 : 4;
    const gguf::Tensor d_weight{"", {N, args.groups * args.latent}, ref::Q4_0,
                                du_offset, latent_dense_.d_bytes()};
    gemv(ds, d_weight, XF_, latent_d_, args.groups * args.latent, ntok, 0, &latent_du_);
    auto run = [&](const char* name, std::vector<mtl::Arg> buffers, uint32_t grid, uint32_t rows) {
        mtl::Dispatch d;
        d.p = P(latent_lib_, name, {}, name);
        d.args = std::move(buffers);
        d.args.push_back(mtl::Arg::V(args));
        d.grid[0] = grid;
        d.grid[1] = rows;
        d.tg[0] = 128;
        push(ds, d);
    };
    run("latent_colscale", {mtl::Arg::B(latent_d_), mtl::Arg::B(latent_inv_, scale_offset)},
        (ntok * args.groups * args.latent + 127) / 128, 1);
    const char* gate_kernel = verify ? "latent_gate_up_4"
                                    : narrow ? "latent_gate_up_8" : "latent_gate_up";
    run(gate_kernel,
        {mtl::Arg::B(latent_a_, a_offset),
         mtl::Arg::B(latent_labels_, label_offset), mtl::Arg::B(ids_), mtl::Arg::B(latent_d_), mtl::Arg::B(h_)},
        args.ff / gate_rows, entries);
    const bool down_narrow = verify || narrow;
    run(down_narrow ? "latent_down_4" : "latent_down", {mtl::Arg::B(latent_a_, a_offset), mtl::Arg::B(ids_),
                         mtl::Arg::B(h_), mtl::Arg::B(latent_down_)}, args.latent / (down_narrow ? 32 : 4), entries);
    const auto u_weights = mtl::Arg::B(latent_du_, du_offset + latent_dense_.d_bytes());
    if (latent_grouped_u_) {
        const uint32_t group_rows = ntok * args.groups;
        // D activations are no longer live after gate/up. Reuse that buffer for the grouped rows.
        run("latent_group_reduce", {mtl::Arg::B(latent_labels_, label_offset), mtl::Arg::B(ids_),
             mtl::Arg::B(wts_), mtl::Arg::B(latent_down_), mtl::Arg::B(latent_d_),
             mtl::Arg::B(latent_group_active_)}, (group_rows * args.latent + 127) / 128, 1);
        const char* u_kernel = verify ? "latent_group_u_4" : narrow ? "latent_group_u_8" : "latent_group_u";
        run(u_kernel, {u_weights, mtl::Arg::B(latent_d_),
             mtl::Arg::B(latent_group_active_), mtl::Arg::B(ye_)}, args.hidden / gate_rows, group_rows);
    } else {
        run("latent_u", {u_weights, mtl::Arg::B(latent_labels_, label_offset),
             mtl::Arg::B(ids_), mtl::Arg::B(wts_), mtl::Arg::B(plan_), mtl::Arg::B(latent_down_),
             mtl::Arg::B(ye_)}, (args.hidden + 3) / 4, entries);
    }
}

void Window::gemv(std::vector<mtl::Dispatch>& ds, const gguf::Tensor& t, const Act& x, const mtl::Buffer& y,
                  size_t y_stride, int ntok, size_t y_off, const mtl::Buffer* wb) {
    const int n_in = (int) t.shape[0], n_out = t.shape.size() > 1 ? (int) t.shape[1] : 1;
    const int ty = t.type;
    int L = 16, Rr = 1, S = 4, K = 1;
    if (n_in >= 6144) { L = 32; Rr = 1; S = 8; K = 2; }
    else if (n_in == 640) { L = 8; Rr = 1; S = 4; K = 1; }
    else if (n_out <= 640 || ty == ref::Q2_0 || ty == ref::BF16) { L = 32; Rr = 1; S = 4; K = 1; }
    static const bool gyro_tiles = !(std::getenv("STRATA_METAL_GYRO_TILES") && std::string(std::getenv("STRATA_METAL_GYRO_TILES")) == "0");   // off: the A/B arm
    if (gyro_dense_ && gyro_tiles && ntok >= 2) {
        // Gyro's dense weights are Q5_K / Q6_K / Q8_0 (3.6 GB a token).  gemv_parity --one on the M4, 2 tokens: Q5_K 2560->10240
        // 261 us with 16,1,4,1 against 210 with 2,1,4,1 (-20%), 2560->6144 158 -> 129, 6144->2560 162 -> 139, 2560->12288
        // 313 -> 247, and the shared down Q8_0 640->2560 36 -> 19 us with 4,2,4,1.  (The tile changes the order of the lane sums.)
        if (ty == ref::Q5_K && n_in == 2560 && n_out >= 6144) { L = 2; Rr = 1; S = 4; K = 1; }
        else if (ty == ref::Q5_K && n_in == 6144) { L = 2; Rr = 1; S = 4; K = 1; }
        else if (ty == ref::Q8_0 && n_in == 640) { L = 4; Rr = 2; S = 4; K = 1; }
    }
    {
        // windows of several tokens: NR rows a lane share each x load (the window's cost is its x traffic).  The sums
        // run in the same order for any NR, so the answers do not change.  STRATA_METAL_GEMV_NR (1, 2): the tile.
        static const int nr_env = [] { const char* e = std::getenv("STRATA_METAL_GEMV_NR"); return e ? std::atoi(e) : 1; }();
        if (ntok > 1 && (nr_env == 2 || fast_verify(ntok)) && n_out >= 2048) {
            if (S == 8 && K == 2) { S = 4; K = 2; }
            if (S == 8 && K == 1) S = 4;
            if (K == 1 || (L == 32 && K == 2)) Rr = 2;
        }
    }
    if (fast_verify(ntok)) {
        if (wb == &latent_du_ && ty == ref::Q4_0) {
            L = 8; Rr = 1; S = 4; K = 1;
        } else if (!wb && ty == ref::Q8_0) {
            if (n_in == N && n_out == n_vocab_) { L = 1; Rr = 1; S = 2; K = 1; }
            else if (n_in == 6144 && n_out == N) { L = 16; Rr = 2; S = 4; K = 1; }
            else if (n_in == N && n_out >= 10240) { L = 8; Rr = 1; S = 8; K = 1; }
            else if (n_in == N && n_out == 6144) { L = 8; Rr = 2; S = 4; K = 1; }
        }
    }
    const std::string fn = "gemv_" + std::to_string(ty) + "_" + std::to_string(L) + "_" + std::to_string(Rr) + "_" +
                           std::to_string(S) + "_" + std::to_string(K);
    const mtl::Pipeline* p = P(lib_, fn, mtl::Constants().i(0, ntok), fn + "#" + std::to_string(ntok));
    GemvArgs a{(uint32_t) n_in, (uint32_t) n_out, (uint32_t) ntok, (uint32_t) ref::row_bytes(t.type, n_in), (uint32_t) n_in,
               (uint32_t) y_stride};
    mtl::Dispatch d;
    d.p = p;
    d.args = {mtl::Arg::B(wb ? *wb : W_, t.offset), mtl::Arg::B(x.b, x.plane_bytes() * xkind(ty)),
              mtl::Arg::B(x.b, x.plane_bytes() * 4), mtl::Arg::B(y, y_off), mtl::Arg::V(a)};
    const int rpt = (S / K) * (32 / L) * Rr;
    d.grid[0] = (n_out + rpt - 1) / rpt;
    d.tg[0] = S * 32;
    push(ds, d);
}

void Window::xprep(std::vector<mtl::Dispatch>& ds, const mtl::Buffer& x, const Act& out, int n) {
    mtl::Dispatch d;
    d.p = P(lib_, "xprep", {}, "xprep");
    const uint32_t nn = n, nt = MAXT;
    d.args = {mtl::Arg::B(x), mtl::Arg::B(out.b), mtl::Arg::V(nn), mtl::Arg::V(nt)};
    d.tg[0] = 256;
    d.grid[0] = (n / 16 * MAXT + 255) / 256;
    push(ds, d);
}

// the hyper-connection read (gr.metal, 3 dispatches): the previous sublayer's write folded in when `apply`
void Window::gr_read(std::vector<mtl::Dispatch>& ds, const std::string& pre, bool apply, const Act& out, bool final_,
                     int ntok) {
    const gguf::Tensor& tn = T(pre + "_norm.weight");
    const gguf::Tensor& tdn = T(pre + "_down.weight");
    const gguf::Tensor& tup = T(pre + "_up.weight");
    const gguf::Tensor& tin = final_ ? tdn : T(pre + "_inject.weight");
    GrArgs a{N, HC, LR, (uint32_t) ntok, apply ? 1u : 0u, eps_};
    mtl::Dispatch d1;
    d1.p = P(lib_, "gr_norm", {}, "gr_norm");
    d1.args = {mtl::Arg::B(R_), mtl::Arg::B(bo_), mtl::Arg::B(inj_[cur_inj_]), mtl::Arg::B(W_, tn.offset), mtl::Arg::B(xn_),
               mtl::Arg::V(a)};
    d1.grid[0] = HC;
    d1.grid[1] = ntok;
    d1.tg[0] = 256;
    push(ds, d1);
    // each matrix: its load-time Q8_0 copy (q8off_), else the file's own tensor (Unsloth's are Q8_0 already, their
    // inject rows F32)
    auto src = [&](const gguf::Tensor& t) -> std::pair<const mtl::Buffer*, size_t> {
        auto it = q8off_.find(t.name);
        return it != q8off_.end() ? std::make_pair(&Q8_, it->second) : std::make_pair(&W_, (size_t) t.offset);
    };
    auto is_q8 = [&](const gguf::Tensor& t) { return q8off_.count(t.name) || t.type == ref::Q8_0; };
    const auto [bd, od] = src(tdn);
    const auto [bi, oi] = src(tin);
    const auto [bu, ou] = src(tup);
    const bool q8 = is_q8(tdn) && is_q8(tup);
    const bool injf = q8 && !final_ && !q8off_.count(tin.name) && tin.type == ref::F32;
    mtl::Dispatch d2;
    const char* dn = injf ? "gr_down_q8f_8_8" : q8 ? "gr_down_q8_8_8" : "gr_down_8_8";
    d2.p = P(lib_, dn, mtl::Constants().i(1, ntok), std::string(dn) + "#" + std::to_string(ntok));
    d2.args = {mtl::Arg::B(*bd, od), mtl::Arg::B(*bi, oi), mtl::Arg::B(xn_), mtl::Arg::B(lo_),
               mtl::Arg::B(inj_[1 - cur_inj_]), mtl::Arg::V(a)};
    d2.grid[0] = LR + HC;
    d2.tg[0] = 256;
    push(ds, d2);
    mtl::Dispatch d3;
    const char* up = q8 ? "gr_up_q8" : "gr_up";
    d3.p = P(lib_, up, mtl::Constants().i(1, ntok), std::string(up) + "#" + std::to_string(ntok));
    GrArgs au = a;
    au.n_tok = MAXT;   // the plane stride of the prepared activation
    d3.args = {mtl::Arg::B(*bu, ou), mtl::Arg::B(lo_), mtl::Arg::B(xn_), mtl::Arg::B(out.b), mtl::Arg::V(au)};
    d3.grid[0] = N / 16;
    d3.tg[0] = 256;
    push(ds, d3);
    cur_inj_ = 1 - cur_inj_;
}

namespace {
struct HadArgs { uint32_t width, x_stride, y_stride; float scale; uint32_t has_signs, pad; };   // quant.metal
}
// y = H (s * x) per block of 128 values of `rows` rows of `width` (x_off / y_off: byte offsets; f16: half rows, no signs)
void Window::had_rows(std::vector<mtl::Dispatch>& ds, int lib, const std::string& pfx, const mtl::Buffer& x, size_t x_off,
                      const mtl::Buffer& y, size_t y_off, int rows, int width, const mtl::Buffer* signs, bool f16) {
    HadArgs a{(uint32_t) width, (uint32_t) width, (uint32_t) width, 1.0f / std::sqrt((float) had_block_), signs ? 1u : 0u, 0u};
    const char* fn = f16 ? "had128_f16" : "had128_f32";
    mtl::Dispatch d;
    d.p = P(lib, fn, {}, pfx + fn);
    d.args = {mtl::Arg::B(x, x_off), mtl::Arg::B(y, y_off), mtl::Arg::B(signs ? *signs : x), mtl::Arg::V(a)};
    d.grid[0] = width / 128;
    d.grid[1] = rows;
    d.tg[0] = 128;
    push(ds, d);
}

void Window::had_rows_ind(std::vector<mtl::Dispatch>& ds, int lib, const std::string& pfx, const mtl::Buffer& h,
                          const mtl::Buffer& meta) {
    HadArgs a{(uint32_t) NFF, (uint32_t) NFF, (uint32_t) NFF, 1.0f / std::sqrt((float) had_block_), hsh_.handle ? 1u : 0u, 0u};
    mtl::Dispatch d;
    d.p = P(lib, "had128_f16_rows", {}, pfx + "had128_f16_rows");
    d.args = {mtl::Arg::B(h), mtl::Arg::B(h), mtl::Arg::B(hsh_.handle ? hsh_ : h), mtl::Arg::V(a)};
    d.tg[0] = 128;
    d.indirect = &meta;
    d.ind_off = 8 * 4;   // (rows, 1, 1): the padded rows of the tiles
    push(ds, d);
}

void Window::layer(std::vector<mtl::Dispatch>& ds, int l, int ntok, int64_t pos0) {
    const std::string b = "blk." + std::to_string(l) + ".";
    // ---- the mixer: layer 1's read after the PLE is not applied again (its pending write was applied before it)
    gr_read(ds, b + "hc_attn", l > 0 && !(l == 1 && ple_on_), XA_, false, ntok);
    if (l % 4 != 3) {
        const int gi = l - l / 4;
        // a one-token window commits in the step; a longer one leaves the state and commit() folds the kept prefix
        // from the inputs kept per layer (qkvL_ ...)
        const bool self_commit = ntok == 1;
        const uint32_t cm = self_commit ? 1u : !partial_ ? 0u : win_head_ ? (0x100u | 1u) : 1u;
        const size_t o_qkv = (size_t) gi * MAXT * 10240 * 4, o_ab = (size_t) gi * MAXT * 48 * 4, o_z = (size_t) gi * MAXT * 6144 * 4;
        gemv(ds, T(b + "attn_qkv.weight"), XA_, qkvL_, 10240, ntok, o_qkv);
        nb_ = true;
        gemv(ds, T(b + "attn_gate.weight"), XA_, zL_, 6144, ntok, o_z);
        nb_ = true;
        gemv(ds, T(b + "ssm_alpha.weight"), XA_, alL_, 48, ntok, o_ab);
        nb_ = true;
        gemv(ds, T(b + "ssm_beta.weight"), XA_, beL_, 48, ntok, o_ab);
        GdnArgs a{16, 48, 10240, cm, eps_, fuse_ ? (uint32_t) (MAXT * 6144) : 0u};
        mtl::Dispatch d;
        d.p = P(lgdn_, "gdn_step", mtl::Constants().i(2, ntok), "gdn_step#" + std::to_string(ntok));
        const int hp = hp_;
        if (rows_) {   // each row is a different sequence: its own state, parity and conv history (rinfo_), one grid row each
            d.p = P(lgdn_, "gdn_step", mtl::Constants().i(2, 1).b(31, true), "gdn_step#rows");
            GdnArgs ar{16, 48, 10240, 1u, eps_, fuse_ ? (uint32_t) (MAXT * 6144) : 0u};
            d.args = {mtl::Arg::B(gstBig_[(size_t) gi]), mtl::Arg::B(hbBig_[(size_t) gi]), mtl::Arg::B(qkvL_, o_qkv),
                      mtl::Arg::B(alL_, o_ab), mtl::Arg::B(beL_, o_ab), mtl::Arg::B(zL_, o_z),
                      mtl::Arg::B(W_, T(b + "ssm_conv1d.weight").offset),
                      mtl::Arg::B(W_, T(b + "ssm_dt.bias").offset), mtl::Arg::B(W_, T(b + "ssm_a").offset),
                      mtl::Arg::B(W_, T(b + "ssm_norm.weight").offset), mtl::Arg::B(fuse_ ? XY_.b : yg_), mtl::Arg::V(ar),
                      mtl::Arg::B(hbBig_[(size_t) gi]), mtl::Arg::B(rinfo_)};
            d.grid[0] = 48;
            d.grid[1] = (uint32_t) rows_n_;
            d.tg[0] = 512;
            push(ds, d);
            if (!fuse_) xprep(ds, yg_, XY_, 6144);
            gemv(ds, T(b + "ssm_out.weight"), XY_, bo_, N, ntok);
        } else {
        d.args = {mtl::Arg::B(gst_[(size_t) gi]), mtl::Arg::B(ghist_[hp][(size_t) gi]), mtl::Arg::B(qkvL_, o_qkv),
                  mtl::Arg::B(alL_, o_ab), mtl::Arg::B(beL_, o_ab), mtl::Arg::B(zL_, o_z),
                  mtl::Arg::B(W_, T(b + "ssm_conv1d.weight").offset),
                  mtl::Arg::B(W_, T(b + "ssm_dt.bias").offset), mtl::Arg::B(W_, T(b + "ssm_a").offset),
                  mtl::Arg::B(W_, T(b + "ssm_norm.weight").offset), mtl::Arg::B(fuse_ ? XY_.b : yg_), mtl::Arg::V(a),
                  mtl::Arg::B(ghist_[1 - hp][(size_t) gi])};
        d.grid[0] = 48;
        d.tg[0] = 512;
        push(ds, d);
        (void) self_commit;   // the history ping-pong turns after the window (run) or at commit
        if (!fuse_) xprep(ds, yg_, XY_, 6144);
        gemv(ds, T(b + "ssm_out.weight"), XY_, bo_, N, ntok);
        }
    } else {
        const int qi = l / 4;
        gemv(ds, T(b + "attn_q.weight"), XA_, qfull_, 12288, ntok);
        nb_ = true;
        gemv(ds, T(b + "attn_k.weight"), XA_, kcur_, 512, ntok);
        nb_ = true;
        gemv(ds, T(b + "attn_v.weight"), XA_, vcur_, 512, ntok);
        nb_ = true;
        gemv(ds, T(b + "indexer.q_proj.weight"), XA_, iqr_, 512, ntok);
        nb_ = true;
        gemv(ds, T(b + "indexer.k_proj.weight"), XA_, ikr_, 128, ntok);
        QsaArgs a{(uint32_t) pos0, 0, (uint32_t) max_cells_, (uint32_t) ntok, eps_};
        mtl::Dispatch d;
        const bool rw = rows_;   // (batch slots: pos0 is the highest row's cell, the rows have their own in rinfo_)
        const mtl::Constants rc = mtl::Constants().b(30, true);
        const mtl::Buffer& KQ = rw ? kqBig_[(size_t) qi] : kq_[(size_t) qi];
        const mtl::Buffer& VQ = rw ? vqBig_[(size_t) qi] : vq_[(size_t) qi];
        const mtl::Buffer& KS = rw ? kscBig_[(size_t) qi] : ksc_[(size_t) qi];
        const mtl::Buffer& VS = rw ? vscBig_[(size_t) qi] : vsc_[(size_t) qi];
        const mtl::Buffer& TL = rw ? tailBig_[(size_t) qi] : tail_[(size_t) qi];
        const mtl::Buffer& DD = rw ? deadBig_[(size_t) qi] : dead_[(size_t) qi];
        const mtl::Buffer& PL = rw ? pooledBig_[(size_t) qi] : pooled_[(size_t) qi];
        d.p = rw ? P(lqsa_, "qsa_prep", rc, "qsa_prep#rows") : P(lqsa_, "qsa_prep", qsa_c(), qsa_k());
        d.args = {mtl::Arg::B(qfull_), mtl::Arg::B(kcur_), mtl::Arg::B(vcur_), mtl::Arg::B(iqr_), mtl::Arg::B(ikr_),
                  mtl::Arg::B(W_, T(b + "attn_q_norm.weight").offset), mtl::Arg::B(W_, T(b + "attn_k_norm.weight").offset),
                  mtl::Arg::B(W_, T(b + "indexer.q_norm.weight").offset), mtl::Arg::B(W_, T(b + "indexer.k_norm.weight").offset),
                  mtl::Arg::B(cs_), mtl::Arg::B(qcur_), mtl::Arg::B(qidx_), mtl::Arg::B(KQ),
                  mtl::Arg::B(VQ), mtl::Arg::B(KS), mtl::Arg::B(VS),
                  mtl::Arg::B(TL), mtl::Arg::B(DD), mtl::Arg::B(PL),
                  mtl::Arg::V(a), mtl::Arg::B(mrope_)};
        if (rw) d.args.push_back(mtl::Arg::B(rinfo_));
        d.grid[0] = 33;
        d.grid[1] = ntok;
        d.tg[0] = 256;
        push(ds, d);
        if (rec_) rec_->push_back({(int) ds.size() - 1, 0});
        mtl::Dispatch s;
        s.p = rw ? P(lqsa_, "qsa_score", rc, "qsa_score#rows") : P(lqsa_, "qsa_score", {}, "qsa_score");
        s.args = {mtl::Arg::B(qidx_), mtl::Arg::B(PL), mtl::Arg::B(DD), mtl::Arg::B(scores_),
                  mtl::Arg::V(a)};
        if (rw) s.args.push_back(mtl::Arg::B(rinfo_));
        s.grid[0] = (uint32_t) (((pos0 + ntok) / 4 + 1 + 7) / 8);
        s.grid[1] = ntok;
        s.tg[0] = 256;
        const bool sel = !(fuse_ && pos0 + ntok <= 2051);
        if (sel) {
            push(ds, s);
            if (rec_) rec_->push_back({(int) ds.size() - 1, 2});
        }
        mtl::Dispatch k;
        k.p = rw ? P(lqsa_, "qsa_topk2", rc, "qsa_topk#rows") : P(lqsa_, "qsa_topk2", {}, "qsa_topk");
        k.args = {mtl::Arg::B(scores_), mtl::Arg::B(qids_), mtl::Arg::V(a)};
        if (rw) k.args.push_back(mtl::Arg::B(rinfo_));
        k.grid[0] = ntok;
        k.tg[0] = 1024;
        if (sel) {
            push(ds, k);
            if (rec_) rec_->push_back({(int) ds.size() - 1, 0});
        }
        mtl::Dispatch at;
        at.p = rw ? P(lqsa_, "qsa_attn", rc, "qsa_attn#rows") : P(lqsa_, "qsa_attn", {}, "qsa_attn");
        at.args = {mtl::Arg::B(qcur_), mtl::Arg::B(KQ), mtl::Arg::B(VQ), mtl::Arg::B(KS),
                   mtl::Arg::B(VS), mtl::Arg::B(qids_), mtl::Arg::B(part_), mtl::Arg::V(a)};
        if (rw) at.args.push_back(mtl::Arg::B(rinfo_));
        at.grid[0] = 33;
        at.grid[1] = 2;
        at.grid[2] = ntok;
        at.tg[0] = 256;
        push(ds, at);
        if (rec_) rec_->push_back({(int) ds.size() - 1, 0});
        mtl::Dispatch m;
        m.p = rw ? P(lqsa_, "qsa_merge", rc, "qsa_merge#rows") : P(lqsa_, "qsa_merge", {}, "qsa_merge");
        QsaArgs am = a;
        am.n_tok = MAXT;   // the plane stride
        m.args = {mtl::Arg::B(part_), mtl::Arg::B(qfull_), mtl::Arg::B(XQ_.b), mtl::Arg::V(am)};
        if (rw) m.args.push_back(mtl::Arg::B(rinfo_));
        m.grid[0] = 24;
        m.grid[1] = ntok;
        m.tg[0] = 256;
        push(ds, m);
        if (rec_) rec_->push_back({(int) ds.size() - 1, 1});
        gemv(ds, T(b + "attn_output.weight"), XQ_, bo_, N, ntok);
    }
    // ---- the MoE
    gr_read(ds, b + "hc_ffn", true, XF_, false, ntok);
    const gguf::Tensor& trt = T(b + "ffn_gate_inp.weight");
    const bool rf = !latent_on_ && routef_ && (trt.type == ref::BF16 || trt.type == ref::F32);
    if (!rf) {
        gemv(ds, trt, XF_, logits_, 512, ntok);
        nb_ = true;
    }
    if (stream_ && st_pred_ && l + 1 < 48) {   // the next layer's routing guessed from this layer's FFN input
        nb_ = !rf;   // beside the router GEMV; with router_route it is the first reader of XF_ here: a barrier
        gemv(ds, T("blk." + std::to_string(l + 1) + ".ffn_gate_inp.weight"), XF_, predL_, 512, ntok,
             (size_t) l * MAXT * 512 * 4);
        if (predL2_.ptr && l + 2 < 48) {
            nb_ = true;
            gemv(ds, T("blk." + std::to_string(l + 2) + ".ffn_gate_inp.weight"), XF_, predL2_, 512, ntok,
                 (size_t) l * MAXT * 512 * 4);
        }
        nb_ = true;
    }
    gemv(ds, T(b + "ffn_gate_shexp.weight"), XF_, gsh_, NFF, ntok);
    nb_ = true;
    gemv(ds, T(b + "ffn_up_shexp.weight"), XF_, ush_, NFF, ntok);
    nb_ = true;
    gemv(ds, T(b + "ffn_gate_inp_shexp.weight"), XF_, sg_, 1, ntok);
    if (latent_on_) {
        const bool overlap = fast_verify(ntok);
        MoeArgs a{};
        a.n_tok = ntok;
        a.topk = TOPK;
        a.n_expert = NEXP;
        mtl::Dispatch route;
        route.p = overlap ? P(lib_, "moe_route", mtl::Constants().b(28, true), "moe_route#only")
                             : P(lib_, "moe_route", {}, "moe_route");
        route.args = {mtl::Arg::B(logits_), mtl::Arg::B(ids_), mtl::Arg::B(wts_),
                      mtl::Arg::B(plan_), mtl::Arg::V(a)};
        route.tg[0] = overlap ? 32 * ntok : 512;
        push(ds, route);
        if (!overlap) latent_routed(ds, l, ntok);

        const GlueArgs glue{NFF, MAXT, N, TOPK};
        mtl::Dispatch shared;
        shared.p = P(lib_, "swiglu_prep", {}, "swiglu_prep");
        shared.args = {mtl::Arg::B(gsh_), mtl::Arg::B(ush_), mtl::Arg::B(XH_.b), mtl::Arg::V(glue)};
        shared.grid[0] = NFF / 16;
        shared.grid[1] = ntok;
        shared.tg[0] = 16;
        nb_ = overlap;
        push(ds, shared);
        if (overlap) {
            // Route, shared SwiGLU, and D projection read independent inputs.
            nb_ = true;
            latent_routed(ds, l, ntok);
            nb_ = true;   // the shared down projection is independent of the last routed dispatch
        }
        gemv(ds, T(b + "ffn_down_shexp.weight"), XH_, ysh_, N, ntok);
    } else {
        MoeArgs a{};
        const gguf::Tensor &tg_ = T(b + "ffn_gate_exps.weight"), &tu = T(b + "ffn_up_exps.weight"),
                           &td = T(b + "ffn_down_exps.weight");
        a.gate_off = tg_.offset;
        a.up_off = tu.offset;
        a.down_off = td.offset;
        if (stream_) {   // offsets inside the expert's arena slot
            a.gate_off = 0;
            a.up_off = st_->up_in_slot(l);
            a.down_off = st_->down_in_slot(l);
        }
        const size_t ebase = moe_rebase(a.gate_off, a.up_off, a.down_off);
        a.gu_row = (uint32_t) ref::row_bytes(tg_.type, N);
        a.d_row = (uint32_t) ref::row_bytes(td.type, NFF);
        a.gu_stride = (uint64_t) a.gu_row * NFF;
        a.d_stride = (uint64_t) a.d_row * N;
        a.n_embd = N;
        a.n_ff = NFF;
        a.n_tok = ntok;
        a.topk = TOPK;
        a.n_expert = NEXP;
        const bool dq2 = td.type == ref::Q2_0;
        // the trellis gate/up: 16 lanes a row, 2 rows a lane (expert_parity --bench --sweep, M4, TQK6, 10 experts: 166.6 us a layer
        // against 174.5 for 32,2,4 and 196.8 for 32,1,4; also the fastest at 2 and 3 tokens)
        const bool tq_ = tg_.type >= 144;
        const int gl = tq_ ? 16 : 32, gr = tq_ ? 2 : 1, gs = 4, ds_ = 4;
        // the down tile: Q2_0 8,1; else 4,2 - but on the M4 (no M5 layout) Q5_1 4,1 and Q8_0 8,1 (4,2 spills there:
        // 24-36 GB/s, mac-kern 513fc615)
        const bool m4 = !m5_layout();
        const int dl = dq2 || (m4 && td.type == ref::Q8_0) ? 8 : 4,
                  dr = dq2 || (m4 && (td.type == ref::Q8_0 || td.type == ref::Q5_1)) ? 1 : 2;
        a.gx_gu = NFF / (gs * (32 / gl) * gr);
        a.gx_dn = N / (ds_ * (32 / dl) * dr);
        mtl::Dispatch r;
        if (rf) {
            // the router GEMV + moe_route (bitwise both), beside the shared expert's GEMVs
            r.p = P(lib_, "router_route_" + std::to_string(trt.type), mtl::Constants().i(0, ntok), "router_route_" + std::to_string(trt.type) + "#" + std::to_string(ntok));
            GemvArgs ga{(uint32_t) N, 512, (uint32_t) ntok, (uint32_t) ref::row_bytes(trt.type, N), (uint32_t) N, 512};
            r.args = {mtl::Arg::B(W_, trt.offset), mtl::Arg::B(XF_.b, 0), mtl::Arg::B(XF_.b, XF_.plane_bytes() * 4),
                      mtl::Arg::B(logits_), mtl::Arg::V(ga), mtl::Arg::B(ids_), mtl::Arg::B(wts_), mtl::Arg::B(plan_),
                      mtl::Arg::V(a), mtl::Arg::B(rcounter_)};
            r.grid[0] = 512 / 16;
            r.tg[0] = 512;
            nb_ = true;
        } else {
            r.p = P(lib_, "moe_route", {}, "moe_route");
            r.args = {mtl::Arg::B(logits_), mtl::Arg::B(ids_), mtl::Arg::B(wts_), mtl::Arg::B(plan_), mtl::Arg::V(a)};
            r.tg[0] = 512;
        }
        push(ds, r);
        {
            GlueArgs ga{NFF, MAXT, N, TOPK};
            mtl::Dispatch d;
            d.p = P(lib_, "swiglu_prep", {}, "swiglu_prep");
            d.args = {mtl::Arg::B(gsh_), mtl::Arg::B(ush_), mtl::Arg::B(XH_.b), mtl::Arg::V(ga)};
            d.grid[0] = NFF / 16;
            d.grid[1] = ntok;
            d.tg[0] = 16;
            nb_ = !rf;   // with router_route the shared GEMVs are not behind a barrier yet: this one waits for them
            push(ds, d);
        }
        const std::string gun = "moe_gu_" + std::to_string(tg_.type) + "_" + std::to_string(gl) + "_" + std::to_string(gr) +
                                "_" + std::to_string(gs);
        const std::string dnn = "moe_down_" + std::to_string(dq2 ? 142 : td.type) + "_" + std::to_string(dl) + "_" +
                                std::to_string(dr) + "_" + std::to_string(ds_);
        mtl::Dispatch g;
        const mtl::Buffer& EW = stream_ ? st_->chunk(0) : W_;
        // function constant 23 (QUANT_CTAB): the i-quant grids from constant memory, no threadgroup copy (same values)
        const bool ct = metal_ctab();
        mtl::Constants ac = stream_ ? mtl::Constants().b(20, true) : mtl::Constants();
        ac.b(23, ct);
        const std::string ak = std::string(stream_ ? "#arena" : "") + (ct ? "#ct" : "");
        g.p = P(lib_, gun, ac, gun + ak);
        const bool hg = had_gu(l), hd = had_dn(l);
        if (hg) had_rows(ds, lib_, "d:", XF_.b, 0, XR_, 0, ntok, N, hsx_.handle ? &hsx_ : nullptr, false);   // before the host's turn
        g.args = {mtl::Arg::B(EW, ebase), hg ? mtl::Arg::B(XR_) : mtl::Arg::B(XF_.b, XF_.plane_bytes() * xkind(tg_.type)), mtl::Arg::B(h_),
                  mtl::Arg::B(plan_), mtl::Arg::V(a), mtl::Arg::B(XF_.b, XF_.plane_bytes() * 4)};
        if (stream_)
            for (int c = -1; c < 3; ++c)
                g.args.push_back(c < 0 ? mtl::Arg::B(st_->smap(), (size_t) l * 512 * 4) : mtl::Arg::B(st_->chunk(c + 1)));
        g.tg[0] = 32 * gs;
        g.indirect = &plan_;
        g.ind_off = P_IGU * 4;
        if (cuts_) cuts_->push_back((int) ds.size());   // the host's turn: this layer's experts put in place
        push(ds, g);
        nb_ = true;
        gemv(ds, T(b + "ffn_down_shexp.weight"), XH_, ysh_, N, ntok);
        if (hd) had_rows(ds, lib_, "d:", h_, 0, h_, 0, ntok * TOPK, NFF, hsh_.handle ? &hsh_ : nullptr, false);   // H h, in place
        mtl::Dispatch dd;
        dd.p = P(lib_, dnn, ac, dnn + ak);
        dd.args = {mtl::Arg::B(EW, ebase), mtl::Arg::B(h_), mtl::Arg::B(ye_), mtl::Arg::B(plan_), mtl::Arg::V(a), mtl::Arg::B(h_)};
        if (stream_)
            for (int c = -1; c < 3; ++c)
                dd.args.push_back(c < 0 ? mtl::Arg::B(st_->smap(), (size_t) l * 512 * 4) : mtl::Arg::B(st_->chunk(c + 1)));
        dd.tg[0] = 32 * ds_;
        dd.indirect = &plan_;
        dd.ind_off = P_IDN * 4;
        push(ds, dd);
    }
    {
        mtl::Dispatch d;
        if (latent_grouped_u_) {
            auto a = latent_args_;
            a.n_tok = ntok;
            d.p = P(latent_lib_, "latent_group_combine", {}, "latent_group_combine");
            d.args = {mtl::Arg::B(ye_), mtl::Arg::B(ysh_), mtl::Arg::B(sg_), mtl::Arg::B(bo_), mtl::Arg::V(a)};
        } else {
            GlueArgs a{N, (uint32_t) ntok, N, TOPK};
            d.p = P(lib_, "moe_combine_sh", {}, "moe_combine_sh");
            d.args = {mtl::Arg::B(ye_), mtl::Arg::B(plan_), mtl::Arg::B(ysh_), mtl::Arg::B(sg_), mtl::Arg::B(bo_), mtl::Arg::V(a)};
        }
        d.grid[0] = (ntok * N + 255) / 256;
        d.tg[0] = 256;
        push(ds, d);
    }
}

namespace {
float bf16f(uint16_t h) { uint32_t i = (uint32_t) h << 16; float f; std::memcpy(&f, &i, 4); return f; }
// ple.cu's gnorm_kernel: per stream, the squares rounded in f32, summed in double
void gnorm(const float* x, const float* w, float* y, int n_embd, int hc, float eps) {
    for (int c = 0; c < hc; ++c) {
        const float* xc = x + (size_t) c * n_embd;
        double acc = 0.0;
        for (int d = 0; d < n_embd; ++d) { const float sq = xc[d] * xc[d]; acc += (double) sq; }
        const float mean = (float) (acc / (double) n_embd);
        const float scale = 1.0f / std::sqrt(mean + eps);
        for (int d = 0; d < n_embd; ++d) y[(size_t) c * n_embd + d] = xc[d] * scale * w[(size_t) c * n_embd + d];
    }
}
}  // namespace

// The PLE block (ple.hpp) for the window's T rows on the CPU, after layer 0 (its pending write applied to R): R +=
// gated + silu(conv); the normalized rows go to last_norm_ (the history advances at commit).
void Window::ple_cpu(int Tn) {
    using namespace strata::kernels;
    const int hc_dim = NG_HC_DIM;
    const float* nkey = (const float*) ((const uint8_t*) W_.ptr + T("blk.1.ple_norm_key.weight").offset);
    const float* nqry = (const float*) ((const uint8_t*) W_.ptr + T("blk.1.ple_norm_query.weight").offset);
    const float* nconv = (const float*) ((const uint8_t*) W_.ptr + T("blk.1.ple_norm_conv.weight").offset);
    const uint16_t* kw = kw_host();
    std::vector<float> key(hc_dim), query(hc_dim), gated(hc_dim), norm(hc_dim), hist = ple_hist_;
    for (int t = 0; t < Tn; ++t) {
        float* hidden = R_.as<float>() + (size_t) t * HC * N;
        gnorm(pkey_.as<float>() + (size_t) t * hc_dim, nkey, key.data(), N, HC, NG_RMS_EPS);
        gnorm(hidden, nqry, query.data(), N, HC, NG_RMS_EPS);
        const float* value = pval_.as<float>() + (size_t) t * N;
        float gate[HC];
        const float inv_sqrt_n = 1.0f / std::sqrt((float) N);
        for (int c = 0; c < HC; ++c) {
            double acc = 0.0;
            for (int d = 0; d < N; ++d) acc += (double) (key[(size_t) c * N + d] * query[(size_t) c * N + d]);
            const float s = (float) acc * inv_sqrt_n;
            const float mag = std::sqrt(std::max(std::fabs(s), 1e-6f));
            const float sgn = s > 0.0f ? 1.0f : (s < 0.0f ? -1.0f : 0.0f);
            gate[c] = 1.0f / (1.0f + std::exp(-(sgn * mag)));
        }
        for (int i = 0; i < hc_dim; ++i) gated[(size_t) i] = value[i % N] * gate[i / N];
        gnorm(gated.data(), nconv, norm.data(), N, HC, NG_RMS_EPS);
        for (int c = 0; c < hc_dim; ++c) {
            float acc = 0.0f;
            for (int k = 0; k < PLE_CONV_KERNEL; ++k) {
                const int row = NG_HIST - (PLE_CONV_KERNEL - 1 - k) * NGRAM_SIZE;
                const float v = row == NG_HIST ? norm[(size_t) c] : hist[(size_t) row + (size_t) NG_HIST * c];
                acc += ref::f16_to_f32(kw[k + PLE_CONV_KERNEL * c]) * v;
            }
            const float conv = acc / (1.0f + std::exp(-acc));
            hidden[c] = hidden[c] + gated[(size_t) c] + conv;
        }
        std::memcpy(last_norm_.data() + (size_t) t * hc_dim, norm.data(), (size_t) hc_dim * 4);
        // the next row of this window sees this one in its history
        for (int c = 0; c < hc_dim; ++c) {
            float* col = hist.data() + (size_t) c * NG_HIST;
            for (int r = 0; r + 1 < NG_HIST; ++r) col[r] = col[r + 1];
            col[NG_HIST - 1] = norm[(size_t) c];
        }
    }
}

bool Window::run(int Tn, const int32_t* tokens, int64_t pos0, bool head, int32_t* out, std::string& err) {
    mtp_src_ = nullptr;
    if (Tn < 1 || Tn > MAXT || (head && Tn > max_t_)) { err = "metal window: window size out of range"; return false; }
    if (pos0 + Tn + 1 >= max_cells_) { err = "metal window: the window runs past the context"; return false; }
    const double t0 = now_ms();
    // ---- the embeddings (CPU dequant of the token rows straight into the shared buffer), broadcast to the streams
    const gguf::Tensor& te = T("token_embd.weight");
    const size_t rb = ref::row_bytes(te.type, N);
    for (int t = 0; t < Tn; ++t) {
        float* r = R_.as<float>() + (size_t) t * HC * N;
        if (const float* img = img_row(pos0 + t)) std::memcpy(r, img, (size_t) N * 4);
        else ref::dequant(te.type, (const uint8_t*) W_.ptr + te.offset + (size_t) tokens[t] * rb, r, N);
        for (int c = 1; c < HC; ++c) std::memcpy(r + (size_t) c * N, r, (size_t) N * 4);
    }
    // ---- the PLE rows (the table read is the shared host code's), projected on the GPU
    if (ple_on_) {
        strata::core::SessionState& ss = *ss_;
        uint32_t rows[MAXT * strata::kernels::PLE_N_HEADS];
        int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
        for (int t = 0; t < Tn; ++t) {
            if (rows_) { prev[0] = rows_ss_[t]->ple_prev[0]; prev[1] = rows_ss_[t]->ple_prev[1]; }   // (batch slots: each row its own sequence)
            strata::kernels::ngram_rows(&tokens[t], prev, 1, main_ss_->ple.consts, rows + t * strata::kernels::PLE_N_HEADS);
            if (!rows_) { prev[0] = prev[1]; prev[1] = tokens[t]; }
        }
        if (!main_ss_->ple.table->gather_batch(rows, (size_t) Tn, emb_.as<float>(), err)) return false;
    }
    tp_pre_ = now_ms();
    const bool sel = !(fuse_ && pos0 + Tn <= 2051);
    if (sel) qdirty_ = true;
    else if (qdirty_) {
        for (int t = 0; t < MAXT; ++t)
            for (int c = 0; c < 2051; ++c) qids_.as<int>()[t * 2051 + c] = c;
        qdirty_ = false;
    }
    const int key = rows_ ? 1000000 + Tn * 2 + (sel ? 1 : 0) + (int) (pend_fold_ * 100)
                          : (((((Tn * 2 + (head ? 1 : 0)) * 2 + hp_) * (MAXT + 1)) + pend_fold_) * 2 + (sel ? 1 : 0)) * 2 +
                                (mrope_on() ? 1 : 0);
    mrope_sync(pos0, Tn);
    const int fold_n = pend_fold_;
    pend_fold_ = 0;
    win_head_ = head;
    auto hit = wcache_.find(key);
    if (hit != wcache_.end()) {
        // the same window shape: only the positions change (QSA's cells, the score grid)
        Cached& c = hit->second;
        for (const auto& [i, kind] : c.qsa) {
            mtl::Dispatch& d = c.ds[(size_t) i];
            QsaArgs a{(uint32_t) pos0, 0, (uint32_t) max_cells_, (uint32_t) (kind == 1 ? MAXT : Tn), eps_};
            // the QsaArgs: the dispatch's last inline argument (qsa_prep has the M-RoPE table bound after it)
            mtl::Arg* qa = nullptr;
            for (auto& g : d.args)
                if (!g.bytes.empty()) qa = &g;
            std::memcpy(qa->bytes.data(), &a, sizeof a);
            if (kind == 2) d.grid[0] = (uint32_t) (((pos0 + Tn) / 4 + 1 + 7) / 8);
        }
        seg_T_ = Tn;
        const double r0 = now_ms();
        last_gpu_ms_ = 1000.0 * run_ds(c);
        {
            static const int kt = std::getenv("STRATA_METAL_KPROF") ? std::atoi(std::getenv("STRATA_METAL_KPROF")) : 0;
            static int seen = 0;
            if (kt == Tn && head && ++seen == 20 && !stream_) kprof(c.ds, Tn);
        }
        tp_rg_ = now_ms() - r0;
        last_n_disp_ = (int) c.ds.size();
        return rows_ ? finish_rows(Tn, tokens, out, t0) : finish(Tn, tokens, pos0, head, out, t0);
    }
    if (stream_ && !st_->ready() && !stream_arena(err)) return false;
    Cached& cw = wcache_[key];
    rec_ = &cw.qsa;
    cuts_ = stream_ ? &cw.cuts : nullptr;
    std::vector<mtl::Dispatch>& ds = cw.ds;
    nb_ = false;
    if (fold_n > 0) fold(ds, fold_n, 1 - hp_);   // the last window's commit (hp_ already turned)
    if (ple_on_) {
        xprep(ds, emb_, XE_, N);
        gemv(ds, T("blk.1.ple_key.weight"), XE_, pkey_, 10240, Tn);
        nb_ = true;
        gemv(ds, T("blk.1.ple_value.weight"), XE_, pval_, N, Tn);
    }
    cur_inj_ = 0;
    layer(ds, 0, Tn, pos0);
    if (ple_on_) {
        // layer 0's FFN write applied to R (gr_norm with apply writes R back; its xn is not used)
        const gguf::Tensor& tn = T("blk.1.hc_attn_norm.weight");
        GrArgs a{N, HC, LR, (uint32_t) Tn, 1u, eps_};
        mtl::Dispatch d1;
        d1.p = P(lib_, "gr_norm", {}, "gr_norm");
        d1.args = {mtl::Arg::B(R_), mtl::Arg::B(bo_), mtl::Arg::B(inj_[cur_inj_]), mtl::Arg::B(W_, tn.offset), mtl::Arg::B(xn_),
                   mtl::Arg::V(a)};
        d1.grid[0] = HC;
        d1.grid[1] = Tn;
        d1.tg[0] = 256;
        push(ds, d1);
        // the PLE block on the GPU (msl/ple.metal): norms, gate, conv, the residual update
        struct PleArgs { uint32_t n_embd, hc; float eps; } pa{N, HC, strata::kernels::NG_RMS_EPS};
        auto off = [&](const char* n) { return T(n).offset; };
        mtl::Dispatch g1, g2, g3, g4;
        g1.p = P(lple_, "ple_gnorm", {}, "ple_gnorm");
        g1.args = {mtl::Arg::B(pkey_), mtl::Arg::B(W_, off("blk.1.ple_norm_key.weight")), mtl::Arg::B(keyn_), mtl::Arg::V(pa)};
        g1.grid[0] = HC; g1.grid[1] = Tn; g1.tg[0] = 256;
        push(ds, g1);
        g2.p = g1.p;
        g2.args = {mtl::Arg::B(R_), mtl::Arg::B(W_, off("blk.1.ple_norm_query.weight")), mtl::Arg::B(qn_), mtl::Arg::V(pa)};
        g2.grid[0] = HC; g2.grid[1] = Tn; g2.tg[0] = 256;
        nb_ = true;   // beside the key norm
        push(ds, g2);
        g3.p = P(lple_, "ple_gate_norm", {}, "ple_gate_norm");
        g3.args = {mtl::Arg::B(keyn_), mtl::Arg::B(qn_), mtl::Arg::B(pval_), mtl::Arg::B(W_, off("blk.1.ple_norm_conv.weight")),
                   mtl::Arg::B(gated_), mtl::Arg::B(normW_), mtl::Arg::V(pa)};
        g3.grid[0] = HC; g3.grid[1] = Tn; g3.tg[0] = 256;
        push(ds, g3);
        g4.p = rows_ ? P(lple_, "ple_conv_add", mtl::Constants().b(32, true), "ple_conv_add#rows") : P(lple_, "ple_conv_add", {}, "ple_conv_add");
        g4.args = {mtl::Arg::B(rows_ ? pleBig_ : pleHist_), mtl::Arg::B(normW_), kw_arg(),
                   mtl::Arg::B(gated_), mtl::Arg::B(R_), mtl::Arg::V(pa)};
        if (rows_) g4.args.push_back(mtl::Arg::B(rinfo_));
        g4.grid[0] = HC * N / 256; g4.grid[1] = Tn; g4.tg[0] = 256;
        push(ds, g4);
    }
    for (int l = 1; l < 48; ++l) layer(ds, l, Tn, pos0);
    // ---- the final mixer and the head
    gr_read(ds, "output_hc", true, XA_, true, Tn);
    if (head) gemv(ds, T("output.weight"), XA_, lm_, (size_t) n_vocab_, Tn);
    if (head) {
        ArgArgs a{(uint32_t) n_vocab_, 120};
        mtl::Dispatch d1, d2;
        d1.p = P(lsmp_, "argmax_part", {}, "argmax_part");
        d1.args = {mtl::Arg::B(lm_), mtl::Arg::B(apv_), mtl::Arg::B(api_), mtl::Arg::V(a)};
        d1.grid[0] = 120;
        d1.grid[1] = Tn;
        d1.tg[0] = 256;
        push(ds, d1);
        d2.p = P(lsmp_, "argmax_final", {}, "argmax_final");
        d2.args = {mtl::Arg::B(apv_), mtl::Arg::B(api_), mtl::Arg::B(tokout_), mtl::Arg::V(a)};
        d2.grid[0] = Tn;
        d2.tg[0] = 256;
        push(ds, d2);
    }
    rec_ = nullptr;
    cuts_ = nullptr;
    seg_T_ = Tn;
    const double r0 = now_ms();
    last_gpu_ms_ = 1000.0 * run_ds(cw);
    tp_rg_ = now_ms() - r0;
    last_n_disp_ = (int) ds.size();
    return rows_ ? finish_rows(Tn, tokens, out, t0) : finish(Tn, tokens, pos0, head, out, t0);
}

bool Window::run_rows(const int* rows, int S, const int32_t* tokens, const int64_t* pos, int32_t* out, std::string& err) {
    if (S < 1 || S > MAXT || S > max_t_) { err = "metal window: batch window size out of range"; return false; }
    if (active_ != 0) activate(0);
    uint32_t* ri = rinfo_.as<uint32_t>() + 8;
    int64_t maxpos = 0;
    for (int t = 0; t < S; ++t) {
        if (rows[t] < 0 || rows[t] >= n_slots()) { err = "metal window: a batch row names a slot that does not exist"; return false; }
        for (int u = 0; u < t; ++u)
            if (rows[u] == rows[t]) { err = "metal window: two rows of one batch window are the same slot"; return false; }
        const int set = rows[t] + 1;
        rows_ss_[t] = sets_[(size_t) set]->ss;
        rows_set_[t] = set;
        ri[4 * t] = (uint32_t) set;
        ri[4 * t + 1] = (uint32_t) pos[t];
        ri[4 * t + 2] = (uint32_t) sets_[(size_t) set]->hp;
        ri[4 * t + 3] = 0;
        maxpos = std::max<int64_t>(maxpos, pos[t]);
    }
    rows_ = true;
    rows_n_ = S;
    const bool ok = run(S, tokens, maxpos, true, out, err);
    rows_ = false;
    return ok;
}

// the end of a window of batch slots: the picks, and each slot's commit (a one-token step committed its row: the other history
// parity is the committed one; the PLE history takes the row's normalized values; the session's token pair advances)
bool Window::finish_rows(int Tn, const int32_t* tokens, int32_t* out, double t0) {
    ++win_n_[Tn];
    win_ms_[Tn] += now_ms() - t0;
    win_gpu_[Tn] += last_gpu_ms_;
    for (int t = 0; t < Tn; ++t) out[t] = tokout_.as<int32_t>()[t];
    for (int t = 0; t < Tn; ++t) {
        SeqSet& s = *sets_[(size_t) rows_set_[t]];
        s.hp = 1 - s.hp;
        if (ple_on_) {
            const float* norm = normW_.as<float>() + (size_t) t * strata::kernels::NG_HC_DIM;
            for (int c = 0; c < strata::kernels::NG_HC_DIM; ++c) {
                float* col = s.pleHist.as<float>() + (size_t) c * strata::kernels::NG_HIST;
                for (int r = 0; r + 1 < strata::kernels::NG_HIST; ++r) col[r] = col[r + 1];
                col[strata::kernels::NG_HIST - 1] = norm[c];
            }
        }
        s.ss->ple_prev[0] = s.ss->ple_prev[1];
        s.ss->ple_prev[1] = tokens[t];
    }
    return true;
}

double Window::run_ds(Cached& c) {
    if (!stream_) return cx_->run_graph(c.ds);
    return cx_->run_graph_seg(c.ds, c.cuts, [](void* u, int s) { ((Window*) u)->at_cut(s); }, this);
}

namespace {
Window* g_stream_window = nullptr;
// What the process holds now, all of it (task_info TASK_VM_INFO): phys_footprint is its anonymous + wired + compressed memory
// (not the clean file pages it only maps), resident_size is the RSS (those pages too).  Bytes; 0 when the call fails.
struct ProcMem {
    double footprint = 0, rss = 0;
};
ProcMem proc_mem() {
    task_vm_info_data_t ti;
    mach_msg_type_number_t cnt = TASK_VM_INFO_COUNT;
    ProcMem r;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &ti, &cnt) == KERN_SUCCESS) {
        r.footprint = (double) ti.phys_footprint;
        r.rss = (double) ti.resident_size;
    }
    return r;
}
void stream_atexit() {
    if (g_stream_window) g_stream_window->stream_report(stderr);
}
}  // namespace

// The arena, once the rest is allocated (the MTP layer loads after init): what the GPU working set and the RAM leave
// with STRATA_METAL_KEEP_FREE_GB kept for macOS (default: 7 GB, or 15% of the RAM when that is more - the M5 Max
// froze with an 89 GB arena and 7 GB left beside a 50 GB n-gram table read through the file cache; arena_budget.h);
// STRATA_METAL_ARENA_GB sets it.
bool Window::stream_arena(std::string& err) {
    uint64_t phys = 0;
    size_t len = sizeof phys;
    sysctlbyname("hw.memsize", &phys, &len, nullptr, 0);
    const double gb = 1e9;
    if (!(std::getenv("STRATA_METAL_KEEP_HOST_WEIGHTS") && std::string(std::getenv("STRATA_METAL_KEEP_HOST_WEIGHTS")) == "1")) {
        const uint64_t g = strata::gpu::purge_device_allocations(32u << 20);
        std::fprintf(stderr, "strata metal stream: %.2f GB of the host code's device weight copies given back (unused by the "
                             "Metal window)\n", g / 1e9);
    }
    const double used = (double) cx_->allocated();
    // what macOS gives now without paging anything out (arena_avail: free, purgeable, the clean file cache; not other
    // programs' inactive memory, freed only by compressing or swapping it)
    auto avail_now = [&]() -> double {
        vm_statistics64_data_t vs;
        mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
        if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &vs, &cnt) == KERN_SUCCESS) {
            const double pg = (double) vm_page_size;
            return arena_avail(vs.free_count * pg, vs.purgeable_count * pg, vs.inactive_count * pg, vs.speculative_count * pg,
                               vs.external_page_count * pg);
        }
        return (double) phys - (double) cx_->allocated();
    };
    ArenaMem am;   // arena_budget.h: the rule, testable
    am.phys = (double) phys;
    am.working_set = (double) cx_->recommended_working_set();
    am.allocated = used;
    const ProcMem pm0 = proc_mem();   // everything the process holds, not only the Metal buffers (the old 67% rule missed it)
    am.proc_footprint = pm0.footprint;
    {   // the file cache macOS keeps before it compresses anonymous memory (a 4-byte count of pages, read into a zeroed 8)
        unsigned long long pages = 0;
        size_t pl = sizeof pages;
        if (sysctlbyname("vm.vm_page_filecache_min", &pages, &pl, nullptr, 0) == 0) am.filecache_min = (double) pages * (double) vm_page_size;
    }
    am.available = avail_now();
    if (main_ss_ && main_ss_->ple.table && main_ss_->ple.table->mode() == strata::kernels::PleIo::Mmap)
        am.ple_mapped = (double) ss_->ple.table->rows() * 192.0;   // a row is at most 192 bytes (Q8_0 + its scale)
    if (const char* e = std::getenv("STRATA_METAL_KEEP_FREE_GB")) am.keep_gb = std::atof(e);
    if (const char* e = std::getenv("STRATA_METAL_ARENA_GB")) am.arena_gb = std::atof(e);
    if (const char* e = std::getenv("STRATA_METAL_LEAVE_SHARE")) am.leave_share = std::atof(e);   // for measuring; the default is kLeaveShare
    const double keep = arena_keep(am);
    const double avail = am.available;
    double budget = arena_budget(am);
    if (budget < 2 * gb) {
        std::fprintf(stderr, "strata metal stream: only %.1f GB left for the expert arena (the GPU working set %.1f GB, RAM %.1f GB, "
                             "%.1f GB allocated): expect a low hit rate\n", budget / gb, cx_->recommended_working_set() / gb,
                     phys / gb, used / gb);
        budget = std::max(budget, 1.0 * gb);
    }
    std::string prof;
    if (const char* e = std::getenv("STRATA_METAL_EXPERT_PROFILE")) prof = e;
    else prof = strata_metal::resource(nullptr, "data", STRATA_METAL_DATA_DEFAULT) + "/expert-profile.bin";
    std::fprintf(stderr, "strata metal stream: %.2f GB allocated before the arena (the process holds %.2f GB, %.2f GB resident); "
                         "RAM %.1f GB (%.1f GB available, %.1f GB kept for macOS), GPU working set %.1f GB\n", used / gb,
                 pm0.footprint / gb, pm0.rss / gb, phys / gb, avail / gb, keep / gb, cx_->recommended_working_set() / gb);
    {   // the planned total: arena + the Metal buffers (or the process's own count when it is more) + the n-gram table's
        // reserve, against the cap (arena_budget.h: footprint_cap)
        const FootprintPlan fp = footprint_plan(am, budget);
        std::fprintf(stderr, "strata metal stream: planned resident footprint %.1f GB = arena %.1f + %.1f held before it + %.1f "
                             "reserved for the n-gram table; cap %.1f GB = the lesser of the GPU working set %.1f GB and the RAM "
                             "%.1f GB less %.1f GB left to macOS and its file cache%s\n",
                     fp.total / gb, fp.arena / gb, fp.base / gb, fp.ple / gb, fp.cap / gb, cx_->recommended_working_set() / gb,
                     phys / gb, (am.phys - fp.cap) / gb,
                     am.arena_gb >= 0 && fp.over() ? " - STRATA_METAL_ARENA_GB is above the cap: kept as set (your choice), but "
                                                     "a Mac that runs short of memory can freeze (watchdog panics seen at 90+ GB "
                                                     "resident of 128)" : "");
    }
    const char* epg = std::getenv("STRATA_METAL_PROMPT_GEMM");
    const bool pg_on = !(epg && std::string(epg) == "0");
    bool minimal = false;   // the arena's least layout (the min slots) is not shrunk further: it goes ahead, filled in steps
    for (int tries = 0;; ++tries) {
        const double planned = budget;
        std::function<uint64_t()> recheck = [&]() -> uint64_t { return (uint64_t) arena_shrink(am, planned, avail_now()); };
        if (st_->make_arena(*cx_, (uint64_t) budget, prof, pg_on ? 576 : 64, err, minimal ? std::function<uint64_t()>() : recheck,
                            [&](uint64_t step) { return arena_grow(am, avail_now(), (double) step); }))
            break;
        if (st_->shrink_to() == 0 || tries == 4) return false;
        if ((double) st_->shrink_to() + 64.0 * (1 << 20) >= budget) minimal = true;   // no smaller layout than this one
        std::fprintf(stderr, "strata metal stream: less memory is free than when the arena was planned: %.1f -> %.1f GB\n",
                     budget / gb, st_->shrink_to() / gb);
        budget = (double) st_->shrink_to();
    }
    if (pg_on && !prompt_gemm_init(err)) return false;
    {   // what the process really holds now, against the plan and the cap (a number that disagrees is a bug in the count)
        const ProcMem pm1 = proc_mem();
        const double cap = footprint_cap(am);
        std::fprintf(stderr, "strata metal stream: resident footprint after the fill: the process holds %.1f GB (%.1f GB resident), "
                             "planned %.1f GB, cap %.1f GB%s\n", pm1.footprint / gb, pm1.rss / gb,
                     footprint_plan(am, (double) st_->arena_used_bytes()).total / gb, cap / gb,
                     pm1.footprint > cap ? " - OVER THE CAP" : "");
    }
    g_stream_window = this;
    std::atexit(stream_atexit);
    return true;
}

void Window::logpos(int Tn, const int32_t* tokens, int64_t pos0, bool head) {
    static const char* path = std::getenv("STRATA_METAL_LOGPOS");
    if (path == nullptr || !head) return;
    if (logpos_ == nullptr) logpos_ = std::fopen(path, "ab");
    if (logpos_ == nullptr) return;
    if (lp_pend_ && Tn >= 1 && pos0 == lp_pos_ + 1) {
        const int32_t tgt = tokens[0];
        std::fprintf(logpos_, "%lld\t%d\t%.6f\t%d\t0\t0\t0\t0", (long long) (lp_pos_ - lp_first_), tgt,
                     (double) lp_row_[(size_t) tgt] - lp_lse_, lp_arg_);
        for (const auto& [v, id] : lp_top_) std::fprintf(logpos_, "\t%d:%.6f", id, (double) v - lp_lse_);
        std::fputc('\n', logpos_);
        std::fflush(logpos_);
    }
    lp_pend_ = false;
    if (Tn != 1) return;
    if (lp_first_ < 0) lp_first_ = pos0;
    const float* lg = lm_.as<float>();
    lp_row_.assign(lg, lg + n_vocab_);
    float mx = lg[0];
    int32_t am = 0;
    for (int64_t i = 1; i < n_vocab_; ++i) if (lg[i] > mx) { mx = lg[i]; am = (int32_t) i; }
    double s = 0.0;
    for (int64_t i = 0; i < n_vocab_; ++i) s += std::exp((double) (lg[i] - mx));
    lp_lse_ = (double) mx + std::log(s);
    lp_arg_ = am;
    lp_top_.clear();
    for (int64_t i = 0; i < n_vocab_; ++i) lp_top_.push_back({lg[i], (int32_t) i});
    std::partial_sort(lp_top_.begin(), lp_top_.begin() + 64, lp_top_.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
    lp_top_.resize(64);
    lp_pos_ = pos0;
    lp_pend_ = true;
}

void Window::kprof(const std::vector<mtl::Dispatch>& ds, int Tn) {
    std::map<const mtl::Pipeline*, std::string> pname;
    for (const auto& [k, v] : pcache_) pname[&v] = k.substr(0, k.find('#'));
    std::map<uint64_t, std::string> tname;
    for (const auto& [k, v] : f_.tensors) {
        std::string n = k;
        if (n.rfind("blk.", 0) == 0) n = n.substr(n.find('.', 4) + 1);
        tname[v.offset] = n;
    }
    std::map<size_t, std::string> q8name;
    for (const auto& [k, v] : q8off_) {
        std::string n = k;
        if (n.rfind("blk.", 0) == 0) n = n.substr(n.find('.', 4) + 1);
        q8name[v] = n;
    }
    std::map<std::string, std::vector<mtl::Dispatch>> by;
    for (const auto& d : ds) {
        std::string c = pname.count(d.p) ? pname[d.p] : "?";
        if (!d.args.empty() && d.args[0].buf == &W_ && tname.count(d.args[0].off) && c.rfind("moe_", 0) != 0)
            c += " " + tname[d.args[0].off];
        else if (!d.args.empty() && d.args[0].buf == &Q8_) c += " " + (q8name.count(d.args[0].off) ? q8name[d.args[0].off] : std::string("Q8"));
        mtl::Dispatch x = d;
        x.barrier = true;
        by[c].push_back(x);
    }
    auto med = [&](const std::vector<mtl::Dispatch>& v) {
        std::vector<double> t;
        for (int i = 0; i < 7; ++i) t.push_back(1000.0 * cx_->run_graph(v));
        std::sort(t.begin(), t.end());
        return t[3];
    };
    const double all = med(ds);
    std::vector<std::pair<double, std::string>> rows;
    double sum = 0;
    for (const auto& [c, v] : by) {
        const double m = med(v);
        sum += m;
        rows.push_back({m, c + " x" + std::to_string(v.size())});
    }
    std::sort(rows.rbegin(), rows.rend());
    std::fprintf(stderr, "strata metal kprof: window T%d, %zu dispatches, GPU %.3f ms; classes alone (serialized) sum %.3f ms:\n",
                 Tn, ds.size(), all, sum);
    for (const auto& [m, c] : rows) std::fprintf(stderr, "strata metal kprof: %8.3f ms %5.1f%%  %s\n", m, 100.0 * m / sum, c.c_str());
}

Window::~Window() {
    if (logpos_) std::fclose(logpos_);
    report_profile();
    if (pb_.lat_gu16.handle) cx_->free(pb_.lat_gu16);
}

void Window::report_profile() const {
    if (std::getenv("STRATA_METAL_PROFILE") == nullptr) return;
    auto line = [](const char* what, const Prof& x) {
        if (x.n == 0) return;
        std::fprintf(stderr, "strata metal profile: %-10s %6lld x  %.3f ms: host before %.3f, run_graph %.3f (GPU %.3f, so "
                             "encode/submit/wait %.3f), patch+finish %.3f\n", what, (long long) x.n, x.all / x.n, x.pre / x.n,
                     x.rg / x.n, x.gpu / x.n, (x.rg - x.gpu) / x.n, x.fin / x.n);
    };
    for (const auto& [k, x] : prof_) {
        const std::string w = "window T" + std::to_string(k / 2) + (k % 2 ? "" : " nohead");
        line(w.c_str(), x);
    }
    line("commit", cprof_);
    line("draft", dprof_);
}

bool Window::finish(int Tn, const int32_t* tokens, int64_t pos0, bool head, int32_t* out, double t0) {
    ++win_n_[Tn];
    win_ms_[Tn] += now_ms() - t0;
    win_gpu_[Tn] += last_gpu_ms_;
    static const bool prof = std::getenv("STRATA_METAL_PROFILE") != nullptr;
    if (prof) {
        Prof& x = prof_[Tn * 2 + (head ? 1 : 0)];
        const double t1 = now_ms();
        ++x.n; x.pre += tp_pre_ - t0; x.rg += tp_rg_; x.gpu += last_gpu_ms_; x.fin += t1 - tp_pre_ - tp_rg_;
        x.all += t1 - t0;
    }
    logpos(Tn, tokens, pos0, head);
    if (Tn == 1 || partial_) hp_ = 1 - hp_;   // the window committed its first row (or all of them) in the step
    last_head_ = head;
    for (int t = 0; t < Tn; ++t) out[t] = head ? tokout_.as<int32_t>()[t] : -1;
    last_T_ = Tn;
    for (int t = 0; t < Tn; ++t) last_tokens_[t] = tokens[t];
    static const bool trace = std::getenv("STRATA_METAL_TRACE") != nullptr;
    if (trace && head) {
        // the pick and its margin over the runner-up (a near tie says a backend difference may flip it)
        const float* lg = lm_.as<float>();
        int i1 = -1, i2 = -1;
        for (int i = 0; i < n_vocab_; ++i) {
            if (i1 < 0 || lg[i] > lg[i1]) { i2 = i1; i1 = i; }
            else if (i2 < 0 || lg[i] > lg[i2]) i2 = i;
        }
        std::fprintf(stderr, "strata metal: window pos %lld token %d -> %d (top %d %.4f, 2nd %d %.4f, margin %.4f) %.2f ms (GPU %.2f ms, %d dispatches)\n",
                     (long long) pos0, tokens[0], out[0], i1, lg[i1], i2, lg[i2], lg[i1] - lg[i2], now_ms() - t0, last_gpu_ms_, last_n_disp_);
    }
    return true;
}

void Window::fold(std::vector<mtl::Dispatch>& ds, int n_keep, int hp) {
    const size_t off = partial_ ? 1 : 0;   // rows 1.. (row 0 was committed in the window's step)
    for (int l = 0; l < 48; ++l) {
        if (l % 4 == 3) continue;
        const int gi = l - l / 4;
        const std::string b = "blk." + std::to_string(l) + ".";
        const size_t o_qkv = ((size_t) gi * MAXT + off) * 10240 * 4, o_ab = ((size_t) gi * MAXT + off) * 48 * 4,
                     o_z = ((size_t) gi * MAXT + off) * 6144 * 4;
        GdnArgs a{16, 48, 10240, 1u, eps_, 0};
        mtl::Dispatch d;
        d.p = P(lgdn_, "gdn_step", mtl::Constants().i(2, n_keep), "gdn_step#" + std::to_string(n_keep));
        d.args = {mtl::Arg::B(gst_[(size_t) gi]), mtl::Arg::B(ghist_[hp][(size_t) gi]), mtl::Arg::B(qkvL_, o_qkv),
                  mtl::Arg::B(alL_, o_ab), mtl::Arg::B(beL_, o_ab), mtl::Arg::B(zL_, o_z),
                  mtl::Arg::B(W_, T(b + "ssm_conv1d.weight").offset), mtl::Arg::B(W_, T(b + "ssm_dt.bias").offset),
                  mtl::Arg::B(W_, T(b + "ssm_a").offset), mtl::Arg::B(W_, T(b + "ssm_norm.weight").offset),
                  mtl::Arg::B(ydump_), mtl::Arg::V(a), mtl::Arg::B(ghist_[1 - hp][(size_t) gi])};
        d.grid[0] = 48;
        d.tg[0] = 512;
        nb_ = true;   // the layers' folds are independent of each other
        push(ds, d);
    }
    nb_ = false;   // what follows reads the folded state: a barrier
}

void Window::swap_state(SeqSet& o) {
    auto sw = [](std::vector<mtl::Buffer>& a, std::vector<mtl::Buffer>& b) {
        for (size_t i = 0; i < a.size() && i < b.size(); ++i) std::swap(a[i], b[i]);
    };
    sw(gst_, o.gst); sw(ghist_[0], o.gh0); sw(ghist_[1], o.gh1); sw(kq_, o.kq); sw(vq_, o.vq); sw(ksc_, o.ksc); sw(vsc_, o.vsc);
    sw(tail_, o.tail); sw(dead_, o.dead); sw(pooled_, o.pooled);
    std::swap(pleHist_, o.pleHist);
    ple_hist_.swap(o.ple_hist);
    last_norm_.swap(o.last_norm);
    std::swap(hp_, o.hp); std::swap(pend_fold_, o.pend_fold); std::swap(last_T_, o.last_T); std::swap(last_head_, o.last_head);
    for (int t = 0; t < MAXT; ++t) std::swap(last_tokens_[t], o.last_tokens[t]);
    std::swap(ss_, o.ss);
}

// the buffers of one sequence's running state, shaped like the live ones (real: allocated and zero; else empty placeholders)
void Window::alloc_set(SeqSet& o, bool real, int idx) {
    const size_t MAXC = (size_t) max_cells_, MB = MAXC / 4 + 2;
    if (real && bigs_ok_) {   // set idx's part of the big buffers
        for (size_t i = 0; i < gst_.size(); ++i) {
            o.gst.push_back(view_of(gstBig_[i], (size_t) idx * st_gst_, 4 * 128 * 48 * 128));
            o.gh0.push_back(view_of(hbBig_[i], ((size_t) idx * 2 + 0) * st_hist_, 4 * 10240 * 3));
            o.gh1.push_back(view_of(hbBig_[i], ((size_t) idx * 2 + 1) * st_hist_, 4 * 10240 * 3));
        }
        for (size_t j = 0; j < kq_.size(); ++j) {
            o.kq.push_back(view_of(kqBig_[j], (size_t) idx * st_kq_, MAXC * 2 * 256));
            o.vq.push_back(view_of(vqBig_[j], (size_t) idx * st_kq_, MAXC * 2 * 256));
            o.ksc.push_back(view_of(kscBig_[j], (size_t) idx * st_sc_, 2 * MAXC * 2 * 4));
            o.vsc.push_back(view_of(vscBig_[j], (size_t) idx * st_sc_, 2 * MAXC * 2 * 4));
            o.tail.push_back(view_of(tailBig_[j], (size_t) idx * st_tail_, 4 * 4 * 128));
            o.dead.push_back(view_of(deadBig_[j], (size_t) idx * st_dead_, 4 * 128));
            o.pooled.push_back(view_of(pooledBig_[j], (size_t) idx * st_pooled_, 4 * MB * 128));
        }
        o.pleHist = view_of(pleBig_, (size_t) idx * st_ple_, 4 * 9 * 10240);
        o.ple_hist.assign(ple_hist_.size(), 0.0f);
        return;
    }
    auto A = [&](size_t bytes) { return real ? cx_->alloc(bytes) : mtl::Buffer(); };
    for (size_t i = 0; i < gst_.size(); ++i) {
        o.gst.push_back(A(4 * 128 * 48 * 128));
        o.gh0.push_back(A(4 * 10240 * 3));
        o.gh1.push_back(A(4 * 10240 * 3));
    }
    for (size_t j = 0; j < kq_.size(); ++j) {
        o.kq.push_back(A(MAXC * 2 * 256));
        o.vq.push_back(A(MAXC * 2 * 256));
        o.ksc.push_back(A(2 * MAXC * 2 * 4));
        o.vsc.push_back(A(2 * MAXC * 2 * 4));
        o.tail.push_back(A(4 * 4 * 128));
        o.dead.push_back(A(4 * 128));
        o.pooled.push_back(A(4 * MB * 128));
    }
    o.pleHist = A(4 * 9 * 10240);
    if (real) o.ple_hist.assign(ple_hist_.size(), 0.0f);
}

bool Window::slots_init(const std::vector<strata::core::SessionState*>& slots, std::string& err) {
    if (!sets_.empty()) return true;   // (asked once)
    if (cx_ == nullptr) { err = "metal window: slots before the window is up"; return false; }
    std::vector<std::unique_ptr<SeqSet>> v;
    v.push_back(std::make_unique<SeqSet>());
    alloc_set(*v[0], false);
    v[0]->ss = ss_;
    set_of_ss_[ss_] = 0;
    uint64_t bytes = 0;
    for (size_t b = 0; b < slots.size(); ++b) {
        if ((int) b >= g_slot_cap) bigs_ok_ = false;   // more slots than the window was sized for: serial windows only
        auto x = std::make_unique<SeqSet>();
        alloc_set(*x, true, (int) b + 1);
        x->ss = slots[b];
        for (auto* vec : {&x->gst, &x->gh0, &x->gh1, &x->kq, &x->vq, &x->ksc, &x->vsc, &x->tail, &x->dead, &x->pooled})
            for (const auto& bf : *vec) bytes += bf.bytes;
        set_of_ss_[slots[b]] = (int) b + 1;
        v.push_back(std::move(x));
    }
    sets_ = std::move(v);
    active_ = 0;
    if (bigs_ok_) {   // the strides the row kernels need (elements of each array's type)
        uint32_t* h = rinfo_.as<uint32_t>();
        h[0] = (uint32_t) st_kq_; h[1] = (uint32_t) (st_sc_ / 2); h[2] = (uint32_t) (st_tail_ / 4); h[3] = (uint32_t) (st_dead_ / 4);
        h[4] = (uint32_t) (st_pooled_ / 4); h[5] = (uint32_t) (st_gst_ / 4); h[6] = (uint32_t) (st_hist_ / 4); h[7] = (uint32_t) (st_ple_ / 4);
    }
    std::fprintf(stderr, "strata metal: %zu batch slots, each with its own running state and K/V (%.2f GB each, %.2f GB in all)\n",
                 slots.size(), slots.empty() ? 0.0 : (double) bytes / (double) slots.size() / 1e9, (double) bytes / 1e9);
    return true;
}

// Makes `set` the live state.  The live members and the stashes exchange contents (two swaps through the active set's own stash).
void Window::activate(int set) {
    if (set == active_ || set < 0 || set >= (int) sets_.size()) return;
    swap_state(*sets_[(size_t) active_]);   // the stash of the active one now holds its state, the live members the spare
    swap_state(*sets_[(size_t) set]);       // ... and now the wanted one
    active_ = set;
}

bool Window::run_slots(const int* rows, int S, const int32_t* tokens, const int64_t* pos, int32_t* out, std::string& err) {
    if (sets_.empty()) { err = "metal window: no batch slots"; return false; }
    static const bool one_window = !(std::getenv("STRATA_METAL_BATCH_ROWS") && std::string(std::getenv("STRATA_METAL_BATCH_ROWS")) == "0");
    if (one_window && bigs_ok_) {
        activate(0);
        return run_rows(rows, S, tokens, pos, out, err);
    }
    bool ok = true;
    for (int t = 0; t < S && ok; ++t) {
        if (rows[t] < 0 || rows[t] >= n_slots()) { err = "metal window: a batch row names a slot that does not exist"; ok = false; break; }
        activate(rows[t] + 1);
        // a one-token window commits its row in its own step; commit(1) adds the PLE history and the session's token pair,
        // which read buffers the next window rewrites
        ok = run(1, tokens + t, pos[t], true, out + t, err) && commit(1, err);
    }
    activate(0);
    return ok;
}

bool Window::slot_copy(bool into_slot, int b, int64_t upto, bool kv_only, std::string& err) {
    if (b < 0 || b >= n_slots()) { err = "metal window: no such batch slot"; return false; }
    activate(0);
    SeqSet& o = *sets_[(size_t) b + 1];
    if (pend_fold_ > 0 || o.pend_fold > 0) { err = "metal window: a slot copy with a fold pending"; return false; }
    const size_t cells = (size_t) std::max<int64_t>(0, std::min<int64_t>(upto, max_cells_));
    auto cp = [&](const mtl::Buffer& live, const mtl::Buffer& stash, size_t n) {
        n = std::min({n, live.bytes, stash.bytes});
        if (into_slot) std::memcpy(stash.ptr, live.ptr, n); else std::memcpy(live.ptr, stash.ptr, n);
    };
    const int hs = into_slot ? hp_ : o.hp;   // the committed history parity of the source
    if (!kv_only) {
    if (into_slot) o.hp = hp_; else hp_ = o.hp;
    for (size_t i = 0; i < gst_.size(); ++i) {
        cp(gst_[i], o.gst[i], gst_[i].bytes);
        cp(ghist_[hs][i], hs ? o.gh1[i] : o.gh0[i], ghist_[hs][i].bytes);
    }
    }
    for (size_t j = 0; j < kq_.size(); ++j) {
        cp(kq_[j], o.kq[j], cells * 512);
        cp(vq_[j], o.vq[j], cells * 512);
        cp(ksc_[j], o.ksc[j], cells * 16);
        cp(vsc_[j], o.vsc[j], cells * 16);
        cp(pooled_[j], o.pooled[j], ((cells + 3) / 4 + 2) * 512);
        if (kv_only) continue;
        cp(tail_[j], o.tail[j], tail_[j].bytes);
        cp(dead_[j], o.dead[j], dead_[j].bytes);
    }
    if (kv_only) return true;
    cp(pleHist_, o.pleHist, pleHist_.bytes);
    if (into_slot) { o.ple_hist = ple_hist_; o.ss->ple_prev[0] = ss_->ple_prev[0]; o.ss->ple_prev[1] = ss_->ple_prev[1]; }
    else { ple_hist_ = o.ple_hist; ss_->ple_prev[0] = o.ss->ple_prev[0]; ss_->ple_prev[1] = o.ss->ple_prev[1]; }
    return true;
}

bool Window::checkpoint_save(strata::core::ConversationCheckpoint& c, std::string& err) {
    (void) err;
    if (pend_fold_ > 0) {   // the last commit's fold first: the checkpoint is the state after it
        std::vector<mtl::Dispatch> ds;
        nb_ = false;
        fold(ds, pend_fold_, 1 - hp_);
        cx_->run_graph(ds);
        pend_fold_ = 0;
    }
    const size_t gs = 4 * 128 * 48 * 128, hs = 4 * 10240 * 3, ts = 4 * 4 * 128, dsz = 4 * 128;
    c.gdn.resize(gst_.size() * (gs + hs));
    for (size_t i = 0; i < gst_.size(); ++i) {
        std::memcpy(c.gdn.data() + i * (gs + hs), gst_[i].ptr, gs);
        std::memcpy(c.gdn.data() + i * (gs + hs) + gs, ghist_[hp_][i].ptr, hs);
    }
    c.ple.assign((const uint8_t*) pleHist_.ptr, (const uint8_t*) pleHist_.ptr + 4 * 9 * 10240);
    c.tails.resize(tail_.size() * ts);
    c.dead.resize(dead_.size() * dsz);
    for (size_t j = 0; j < tail_.size(); ++j) {
        std::memcpy(c.tails.data() + j * ts, tail_[j].ptr, ts);
        std::memcpy(c.dead.data() + j * dsz, dead_[j].ptr, dsz);
    }
    c.block_pos.clear();
    return true;
}

void Window::reset() {
    pend_fold_ = 0;
    last_T_ = 0;
    auto zero = [](const mtl::Buffer& b, size_t n) { std::memset(b.ptr, 0, n); };
    for (size_t i = 0; i < gst_.size(); ++i) {
        zero(gst_[i], 4 * 128 * 48 * 128);
        zero(ghist_[0][i], 4 * 10240 * 3);
        zero(ghist_[1][i], 4 * 10240 * 3);
    }
    zero(pleHist_, 4 * 9 * 10240);
    for (size_t j = 0; j < tail_.size(); ++j) {
        zero(tail_[j], 4 * 4 * 128);
        zero(dead_[j], 4 * 128);
    }
    for (size_t j = 0; j < mtail_.size(); ++j) {
        zero(mtail_[j], 4 * 4 * 128);
        zero(mdead_[j], 4 * 128);
    }
}

bool Window::checkpoint_restore(const strata::core::ConversationCheckpoint& c, std::string& err) {
    const size_t gs = 4 * 128 * 48 * 128, hs = 4 * 10240 * 3, ts = 4 * 4 * 128, dsz = 4 * 128;
    if (c.gdn.size() != gst_.size() * (gs + hs) || c.ple.size() != (size_t) 4 * 9 * 10240 ||
        c.tails.size() != tail_.size() * ts || c.dead.size() != dead_.size() * dsz) {
        err = "metal window: the checkpoint is not this engine's (sizes differ)";
        return false;
    }
    pend_fold_ = 0;   // the restored state replaces whatever was pending
    last_T_ = 0;
    for (size_t i = 0; i < gst_.size(); ++i) {
        std::memcpy(gst_[i].ptr, c.gdn.data() + i * (gs + hs), gs);
        std::memcpy(ghist_[hp_][i].ptr, c.gdn.data() + i * (gs + hs) + gs, hs);
    }
    std::memcpy(pleHist_.ptr, c.ple.data(), c.ple.size());
    for (size_t j = 0; j < tail_.size(); ++j) {
        std::memcpy(tail_[j].ptr, c.tails.data() + j * ts, ts);
        std::memcpy(dead_[j].ptr, c.dead.data() + j * dsz, dsz);
    }
    return true;
}

bool Window::commit(int n_keep, std::string& err) {
    if (n_keep < 1 || n_keep > last_T_) { err = "metal window: commit count out of range"; return false; }
    using namespace strata::kernels;
    if (last_T_ > 1) {
        // the GDN layers: the kept prefix folded into the state (gdn_step over n_keep tokens from the window's own
        // inputs, with commit; bitwise the window's trajectory).  K/V and indexer cells past the prefix are rewritten
        // when those positions run again, before anything reads them.
        static const bool lazy = !(std::getenv("STRATA_METAL_LAZY_FOLD") && std::string(std::getenv("STRATA_METAL_LAZY_FOLD")) == "0");
        int nf = n_keep;
        if (partial_) {
            if (!last_head_) {   // the prompt's window committed every row in its step
                if (n_keep != last_T_) { err = "metal window: a prompt window commits all its rows"; return false; }
                nf = 0;
            } else {
                nf = n_keep - 1;   // row 0 is in
            }
        }
        if (nf > 0) {
            if (lazy) {
                pend_fold_ = nf;   // folded at the start of the next window (from the history parity before the turn)
            } else {
                std::vector<mtl::Dispatch> ds;
                nb_ = false;
                fold(ds, nf, hp_);
                cx_->run_graph(ds);
            }
            hp_ = 1 - hp_;
        }
    }
    strata::core::SessionState& ss = *ss_;
    for (int t = 0; t < n_keep; ++t) {
        if (ple_on_) {
            const float* norm = normW_.as<float>() + (size_t) t * NG_HC_DIM;
            for (int c = 0; c < NG_HC_DIM; ++c) {
                float* col = pleHist_.as<float>() + (size_t) c * NG_HIST;
                for (int r = 0; r + 1 < NG_HIST; ++r) col[r] = col[r + 1];
                col[NG_HIST - 1] = norm[c];
            }
        }
        ss.ple_prev[0] = ss.ple_prev[1];
        ss.ple_prev[1] = last_tokens_[t];
    }
    return true;
}


// ================================================================================================ the MTP draft layer
bool Window::mtp_load(const std::string& gguf_path, int max_t, std::string& err) {
    (void) max_t;
    if (!fm_.open(gguf_path, err)) return false;
    madvise(fm_.map, fm_.map_bytes, MADV_WILLNEED);
    Wm_ = cx_->wrap(fm_.map, fm_.map_bytes);
    if (Wm_.handle == nullptr) { err = "metal mtp: cannot wrap " + gguf_path; return false; }
    // vLLM's GemmaRMSNorm scales by (1 + w): the F32 norms folded once (tools/mtp_rt.py does the same for CUDA)
    size_t tot = 0;
    for (const auto& kv : fm_.tensors)
        if (kv.second.type == 0) { mnorm_off_[kv.first] = tot; tot += (kv.second.bytes + 255) / 256 * 256; }
    mnorm_ = cx_->alloc(tot + 256);
    for (const auto& kv : mnorm_off_) {
        const gguf::Tensor& x = fm_.tensors.at(kv.first);
        const float* src = (const float*) ((const uint8_t*) fm_.map + x.offset);
        float* dst = (float*) (mnorm_.as<uint8_t>() + kv.second);
        for (uint64_t i = 0; i < x.bytes / 4; ++i) dst[i] = src[i] + 1.0f;
    }
    lmtp_ = cx_->compile({strata_metal::resource("STRATA_METAL_ENGINE_KERNELS", "engine-msl", STRATA_METAL_ENGINE_KERNELS_DEFAULT) + "/mtp.metal"}, "#define NO_GGML_COMMON 1\n");
    {
        const char* e = std::getenv("STRATA_METAL_MTP_Q8");
        if (!(e && std::string(e) == "0")) {
            size_t tot = 0;
            std::vector<std::string> names;
            for (const auto& kv : fm_.tensors) {
                const gguf::Tensor& x = kv.second;
                uint64_t ne = 1;
                for (uint64_t d : x.shape) ne *= d;
                if (x.type != ref::BF16 || x.shape.size() < 2 || x.shape[0] % 32 != 0) continue;
                gguf::Tensor q = x;
                q.type = ref::Q8_0;
                q.offset = tot;
                q.bytes = ne / 32 * 34;
                mq8t_[x.name] = q;
                names.push_back(x.name);
                tot += (q.bytes + 255) / 256 * 256;
            }
            mq8_ = cx_->alloc(tot + 4096);
            std::vector<std::thread> th;
            const unsigned nth = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
            for (unsigned k = 0; k < nth; ++k)
                th.emplace_back([&, k] {
                    std::vector<float> tmp;
                    for (size_t i = k; i < names.size(); i += nth) {
                        const gguf::Tensor& x = fm_.tensors.at(names[i]);
                        const uint64_t ne = x.bytes / 2;
                        tmp.resize(ne);
                        ref::dequant(ref::BF16, (const uint8_t*) fm_.map + x.offset, tmp.data(), ne);
                        ref::quantize_q8_0(tmp.data(), ne, mq8_.as<uint8_t>() + mq8t_[names[i]].offset);
                    }
                });
            for (auto& x : th) x.join();
        }
    }
    auto A = [&](size_t bytes) { return cx_->alloc(bytes); };
    const int MAXC = (int) max_cells_, MB = MAXC / 4 + 2;
    mR_ = A(4 * MAXT * HC * N); memb_ = A(4 * MAXT * N); men_ = A(4 * MAXT * N); me2_ = A(4 * MAXT * N);
    mhn_ = A(4 * MAXT * HC * N); mh2_ = A(4 * HC * MAXT * N); midx_ = A(4 * MAXT * 640); mbo_ = A(4 * MAXT * N);
    minj_[0] = A(4 * MAXT * HC); minj_[1] = A(4 * MAXT * HC); mqids_ = A(4 * MAXT * 2051); mlm_ = A(4 * (size_t) n_vocab_);
    auto act = [&](int n) { Act x; x.n = n; x.b = A(4 * ((size_t) 4 * MAXT * n + (size_t) MAXT * n / 16)); return x; };
    XEn_ = act(N); XHn_ = act(N);
    mkq_.push_back(A((size_t) MAXC * 2 * 256)); mvq_.push_back(A((size_t) MAXC * 2 * 256));
    mksc_.push_back(A(2 * (size_t) MAXC * 2 * 4)); mvsc_.push_back(A(2 * (size_t) MAXC * 2 * 4));
    mtail_.push_back(A(4 * 4 * 128)); mdead_.push_back(A(4 * 128)); mpooled_.push_back(A(4 * (size_t) MB * 128));
    {
        std::string vp;
        // the subset setup put beside the draft layer (--draft-vocab: <mtp>/rt/draft_vocab.bin, as CUDA reads it),
        // else the repository's default
        const std::string beside = gguf_path.substr(0, gguf_path.find_last_of('/') + 1) + "rt/draft_vocab.bin";
        if (const char* e = std::getenv("STRATA_METAL_DRAFT_VOCAB")) vp = e;
        else if (FILE* t = std::fopen(beside.c_str(), "rb")) { std::fclose(t); vp = beside; }
        else vp = strata_metal::resource(nullptr, "data", STRATA_METAL_DATA_DEFAULT) + "/draft_vocab.bin";
        FILE* f = vp == "full" ? nullptr : std::fopen(vp.c_str(), "rb");
        if (f) {
            int32_t id;
            while (std::fread(&id, 4, 1, f) == 1) if (id >= 0 && id < n_vocab_) dvocab_.push_back(id);
            std::fclose(f);
        }
        if (!dvocab_.empty()) {
            const gguf::Tensor& h = T("output.weight");
            const size_t rb = ref::row_bytes(h.type, N);
            mhead_ = A(dvocab_.size() * rb + 4096);
            for (size_t i = 0; i < dvocab_.size(); ++i)
                std::memcpy(mhead_.as<uint8_t>() + i * rb, (const uint8_t*) W_.ptr + h.offset + (size_t) dvocab_[i] * rb, rb);
            mhead_t_ = h;
            mhead_t_.offset = 0;
            mhead_t_.shape = {(uint64_t) N, (uint64_t) dvocab_.size()};
            mhead_t_.bytes = dvocab_.size() * rb;
        }
    }
    mtp_on_ = true;
    {
        const int ty = T("token_embd.weight").type;
        const int ok_types[] = {30, 8, 2, 6, 20, 23, 42, 21, 11, 12, 13, 14, 18};
        const char* e = std::getenv("STRATA_METAL_MTP_CHAIN");
        chain_ = !(e && std::string(e) == "0") && std::find(std::begin(ok_types), std::end(ok_types), ty) != std::end(ok_types);
        if (chain_) {
            lchain_ = cx_->compile({"quant.metal", "units.metal", strata_metal::resource("STRATA_METAL_ENGINE_KERNELS", "engine-msl", STRATA_METAL_ENGINE_KERNELS_DEFAULT) + "/mtp_chain.metal"});
            dout_ = A(4 * MAXT); dprob_ = A(4 * MAXT); posv_ = A(4 * N);
            for (int i = 0; i < N; ++i) posv_.as<float>()[i] = (float) i;
            dmap_ = A(4 * std::max<size_t>(1, dvocab_.size()));
            if (!dvocab_.empty()) std::memcpy(dmap_.as<int32_t>(), dvocab_.data(), 4 * dvocab_.size());
            // parity: a few tokens' embeddings, GPU vs the host's dequant (the per-step path's)
            const gguf::Tensor& te = T("token_embd.weight");
            const size_t rb = ref::row_bytes(te.type, N);
            const int32_t probe[MAXT] = {0, 1, 271, 760, 6511, 11751, 100000, (int32_t) n_vocab_ - 1};
            double worst = 0.0;
            std::vector<float> ref_row(N);
            for (int k = 0; k < MAXT; ++k) {
                dout_.as<int32_t>()[0] = probe[k];
                std::vector<mtl::Dispatch> ds;
                nb_ = false;
                mtl::Dispatch d;
                d.p = P(lchain_, "mtp_embed_" + std::to_string(ty), {}, "mtp_embed_" + std::to_string(ty));
                struct EmbArgs { uint32_t n, row_bytes, j; } ea{(uint32_t) N, (uint32_t) rb, 0};
                d.args = {mtl::Arg::B(W_, te.offset), mtl::Arg::B(dout_), mtl::Arg::B(posv_), mtl::Arg::B(memb_), mtl::Arg::V(ea)};
                d.tg[0] = 256;
                push(ds, d);
                cx_->run_graph(ds);
                ref::dequant(te.type, (const uint8_t*) W_.ptr + te.offset + (size_t) probe[k] * rb, ref_row.data(), N);
                for (int i = 0; i < N; ++i) worst = std::max(worst, (double) std::fabs(ref_row[i] - memb_.as<float>()[i]));
            }
            std::fprintf(stderr, "strata metal: MTP draft round in one command buffer (embedding type %d on the GPU: max "
                                 "|GPU - host| %.3g over %d rows)\n", ty, worst, MAXT);
            if (worst > 1e-6) {
                std::fprintf(stderr, "strata metal: the GPU embedding differs from the host's: a command buffer per step\n");
                chain_ = false;
            }
        }
    }
    std::fprintf(stderr, "strata metal: MTP draft layer from %s (%.2f GB, %zu norms folded), draft head over %s tokens\n",
                 gguf_path.c_str(), fm_.file_bytes / 1e9, mnorm_off_.size(),
                 dvocab_.empty() ? "all" : std::to_string(dvocab_.size()).c_str());
    return true;
}

void Window::gr_read_m(std::vector<mtl::Dispatch>& ds, const std::string& pre, mtl::Buffer& Rb, const mtl::Buffer& bo,
                       const mtl::Buffer& inj_in, const mtl::Buffer& inj_out, bool apply, const Act& out, bool final_,
                       int ntok) {
    const gguf::Tensor& tdn = TQ(pre + ".input_mix_weight_down.weight");
    const gguf::Tensor& tup = TQ(pre + ".input_mix_weight_up.weight");
    const gguf::Tensor& tin = final_ ? tdn : TQ(pre + ".block_inject_weight.weight");
    const bool q8 = tdn.type == ref::Q8_0 && tup.type == ref::Q8_0 && tin.type == ref::Q8_0;
    const mtl::Buffer& WB = q8 ? mq8_ : Wm_;
    GrArgs a{N, HC, LR, (uint32_t) ntok, apply ? 1u : 0u, eps_};
    mtl::Dispatch d1;
    d1.p = P(lib_, "gr_norm", {}, "gr_norm");
    d1.args = {mtl::Arg::B(Rb), mtl::Arg::B(bo), mtl::Arg::B(inj_in), mtl::Arg::B(mnorm_, NM(pre + ".hc_norm.weight")),
               mtl::Arg::B(xn_), mtl::Arg::V(a)};
    d1.grid[0] = HC; d1.grid[1] = ntok; d1.tg[0] = 256;
    push(ds, d1);
    mtl::Dispatch d2;
    const char* dn = q8 ? "gr_down_q8_8_8" : "gr_down_8_8";
    d2.p = P(lib_, dn, mtl::Constants().i(1, ntok), std::string(dn) + "#" + std::to_string(ntok));
    d2.args = {mtl::Arg::B(WB, tdn.offset), mtl::Arg::B(WB, tin.offset), mtl::Arg::B(xn_), mtl::Arg::B(lo_),
               mtl::Arg::B(inj_out), mtl::Arg::V(a)};
    d2.grid[0] = LR + HC; d2.tg[0] = 256;
    push(ds, d2);
    mtl::Dispatch d3;
    const char* up = q8 ? "gr_up_q8" : "gr_up";
    d3.p = P(lib_, up, mtl::Constants().i(1, ntok), std::string(up) + "#" + std::to_string(ntok));
    GrArgs au = a;
    au.n_tok = MAXT;
    d3.args = {mtl::Arg::B(WB, tup.offset), mtl::Arg::B(lo_), mtl::Arg::B(xn_), mtl::Arg::B(out.b), mtl::Arg::V(au)};
    d3.grid[0] = N / 16; d3.tg[0] = 256;
    push(ds, d3);
}

void Window::mtp_encode(std::vector<mtl::Dispatch>& ds, const mtl::Buffer& srcR, size_t src_off, int n, const int32_t* tokens,
                        int64_t cell0, bool full, int erow, int slot) {
    struct RmsArgs { uint32_t n; float eps; };
    struct XpArgs { uint32_t n, n_tok, plane_tok, stride; };
    struct CombArgs { uint32_t n, hc, n_tok, h_stride; };
    // ---- the embedding rows (host tokens: dequantized on the CPU into the shared buffer)
    const gguf::Tensor& te = T("token_embd.weight");
    const size_t rb = ref::row_bytes(te.type, N);
    if (tokens)
        for (int i = 0; i < n; ++i)
            ref::dequant(te.type, (const uint8_t*) W_.ptr + te.offset + (size_t) tokens[i] * rb,
                         (membp_ ? *membp_ : memb_).as<float>() + (size_t) (erow + i) * N, N);
    const std::string L = "mtp.layers.0.";
    auto rms = [&](const mtl::Buffer& x, size_t xo, const std::string& w, mtl::Buffer& y, int width, int rows) {
        mtl::Dispatch d;
        d.p = P(lmtp_, "mtp_rms", {}, "mtp_rms");
        RmsArgs a{(uint32_t) width, eps_};
        d.args = {mtl::Arg::B(x, xo), mtl::Arg::B(mnorm_, NM(w)), mtl::Arg::B(y), mtl::Arg::V(a)};
        d.grid[0] = rows; d.tg[0] = 256;
        push(ds, d);
    };
    auto xp = [&](const mtl::Buffer& x, size_t xo, const Act& out, int width, int rows, int stride) {
        mtl::Dispatch d;
        d.p = P(lmtp_, "mtp_xprep", {}, "mtp_xprep");
        XpArgs a{(uint32_t) width, (uint32_t) rows, MAXT, (uint32_t) stride};
        d.args = {mtl::Arg::B(x, xo), mtl::Arg::B(out.b), mtl::Arg::V(a)};
        d.grid[0] = (width / 16 * rows + 255) / 256; d.tg[0] = 256;
        push(ds, d);
    };
    auto gm = [&](const std::string& name, const Act& x, const mtl::Buffer& y, size_t y_stride, int rows, size_t y_off = 0) {
        // a GEMV on the MTP model's tensor (the window's gemv() reads the main model's buffer)
        const gguf::Tensor& tt = TQ(name);
        const int n_in = (int) tt.shape[0], n_out = tt.shape.size() > 1 ? (int) tt.shape[1] : 1, ty = tt.type;
        int Lq = 16, Rr = 1, S = 4, K = 1;
        if (n_in >= 6144) { Lq = 32; Rr = 1; S = 8; K = 2; }
        else if (n_in == 640) { Lq = 8; Rr = 1; S = 4; K = 1; }
        else if (n_out <= 640 || ty == ref::Q2_0 || ty == ref::BF16) { Lq = 32; Rr = 1; S = 4; K = 1; }
        const std::string fn = "gemv_" + std::to_string(ty) + "_" + std::to_string(Lq) + "_" + std::to_string(Rr) + "_" +
                               std::to_string(S) + "_" + std::to_string(K);
        GemvArgs a{(uint32_t) n_in, (uint32_t) n_out, (uint32_t) rows, (uint32_t) ref::row_bytes(ty, n_in), (uint32_t) n_in,
                   (uint32_t) y_stride};
        mtl::Dispatch d;
        d.p = P(lib_, fn, mtl::Constants().i(0, rows), fn + "#" + std::to_string(rows));
        d.args = {mtl::Arg::B(MB(name), tt.offset), mtl::Arg::B(x.b, x.plane_bytes() * xkind(ty)),
                  mtl::Arg::B(x.b, x.plane_bytes() * 4), mtl::Arg::B(y, y_off), mtl::Arg::V(a)};
        const int rpt = (S / K) * (32 / Lq) * Rr;
        d.grid[0] = (n_out + rpt - 1) / rpt; d.tg[0] = S * 32;
        push(ds, d);
    };
    // ---- e = fc_embedding(rms(embed(tok)) * (1 + w)),  h = fc_hidden per stream (rms_10240(R) * (1 + w)),  R = h + e
    rms(membp_ ? *membp_ : memb_, (size_t) erow * N * 4, "mtp.pre_fc_norm_embedding.weight", men_, N, n);
    xp(men_, 0, XEn_, N, n, N);
    gm("mtp.fc_embedding.weight", XEn_, me2_, N, n);
    rms(srcR, src_off, "mtp.pre_fc_norm_hidden.weight", mhn_, HC * N, n);
    for (int c = 0; c < HC; ++c) {
        xp(mhn_, (size_t) c * N * 4, XHn_, N, n, HC * N);
        gm("mtp.fc_hidden.weight", XHn_, mh2_, N, n, (size_t) c * MAXT * N * 4);
    }
    {
        mtl::Dispatch d;
        d.p = P(lmtp_, "mtp_combine", {}, "mtp_combine");
        CombArgs a{N, HC, (uint32_t) n, (uint32_t) (MAXT * N)};
        d.args = {mtl::Arg::B(mh2_), mtl::Arg::B(me2_), mtl::Arg::B(mR_), mtl::Arg::V(a)};
        d.grid[0] = (n * HC * N + 255) / 256; d.tg[0] = 256;
        push(ds, d);
    }
    // ---- the attention hyper-connection, K/V (and the indexer keys) into the layer's own cache
    gr_read_m(ds, L + "attn_hyper_connection", mR_, mbo_, minj_[1], minj_[0], false, XA_, false, n);
    gm(L + "self_attn.k_proj.weight", XA_, kcur_, 512, n);
    nb_ = true;
    gm(L + "self_attn.v_proj.weight", XA_, vcur_, 512, n);
    nb_ = true;
    gm(L + "self_attn.indexer.index_qk_proj.weight", XA_, midx_, 640, n);
    if (full) {
        nb_ = true;
        gm(L + "self_attn.q_proj.weight", XA_, qfull_, 12288, n);
    }
    QsaArgs qa{(uint32_t) cell0, 0, (uint32_t) max_cells_, (uint32_t) n, eps_};
    {
        mtl::Dispatch d;
        d.p = P(lqsa_, "qsa_prep", qsa_c(), qsa_k());
        d.args = {mtl::Arg::B(qfull_), mtl::Arg::B(kcur_), mtl::Arg::B(vcur_), mtl::Arg::B(midx_), mtl::Arg::B(midx_, 512 * 4),
                  mtl::Arg::B(mnorm_, NM(L + "self_attn.q_norm.weight")), mtl::Arg::B(mnorm_, NM(L + "self_attn.k_norm.weight")),
                  mtl::Arg::B(mnorm_, NM(L + "self_attn.indexer.q_layernorm.weight")),
                  mtl::Arg::B(mnorm_, NM(L + "self_attn.indexer.k_layernorm.weight")),
                  mtl::Arg::B(cs_), mtl::Arg::B(qcur_), mtl::Arg::B(qidx_), mtl::Arg::B(mkq_[0]), mtl::Arg::B(mvq_[0]),
                  mtl::Arg::B(mksc_[0]), mtl::Arg::B(mvsc_[0]), mtl::Arg::B(mtail_[0]), mtl::Arg::B(mdead_[0]),
                  mtl::Arg::B(mpooled_[0]), mtl::Arg::V(qa), mtl::Arg::B(mrope_)};
        d.grid[0] = 33; d.grid[1] = n; d.tg[0] = 256;
        push(ds, d);
    }
    if (!full) return;
    // ---- dense attention over the window of cells up to this one (the draft layer's rule), the output gate, o_proj
    {
        int* ids = mqids_.as<int>() + (size_t) slot * 2051;
        const int64_t first = std::max<int64_t>(0, cell0 - 2050);
        for (int64_t c = first; c <= cell0; ++c) ids[c - first] = (int) c;
        mtl::Dispatch at;
        at.p = P(lqsa_, "qsa_attn", {}, "qsa_attn");
        at.args = {mtl::Arg::B(qcur_), mtl::Arg::B(mkq_[0]), mtl::Arg::B(mvq_[0]), mtl::Arg::B(mksc_[0]), mtl::Arg::B(mvsc_[0]),
                   mtl::Arg::B(mqids_, (size_t) slot * 2051 * 4), mtl::Arg::B(part_), mtl::Arg::V(qa)};
        at.grid[0] = 33; at.grid[1] = 2; at.grid[2] = 1; at.tg[0] = 256;
        push(ds, at);
        mtl::Dispatch m;
        m.p = P(lqsa_, "qsa_merge", {}, "qsa_merge");
        QsaArgs am = qa;
        am.n_tok = MAXT;
        m.args = {mtl::Arg::B(part_), mtl::Arg::B(qfull_), mtl::Arg::B(XQ_.b), mtl::Arg::V(am)};
        m.grid[0] = 24; m.grid[1] = 1; m.tg[0] = 256;
        push(ds, m);
        gm(L + "self_attn.o_proj.weight", XQ_, mbo_, N, 1);
    }
    // ---- the MLP hyper-connection (the attention write folded in), the MoE, the write + final mixer, the head
    gr_read_m(ds, L + "mlp_hyper_connection", mR_, mbo_, minj_[0], minj_[1], true, XF_, false, 1);
    gm(L + "mlp.gate.weight", XF_, logits_, 512, 1);
    nb_ = true;
    gm(L + "mlp.shared_expert.gate_proj.weight", XF_, gsh_, NFF, 1);
    nb_ = true;
    gm(L + "mlp.shared_expert.up_proj.weight", XF_, ush_, NFF, 1);
    nb_ = true;
    gm(L + "mlp.shared_expert_gate.weight", XF_, sg_, 1, 1);
    {
        MoeArgs a{};
        const gguf::Tensor &tgu = TM(L + "mlp.experts.gate_up_proj"), &td = TM(L + "mlp.experts.down_proj");
        a.gu_row = (uint32_t) ref::row_bytes(tgu.type, N);
        a.d_row = (uint32_t) ref::row_bytes(td.type, NFF);
        a.gate_off = tgu.offset;
        a.up_off = tgu.offset + (uint64_t) NFF * a.gu_row;   // [gate rows | up rows] per expert
        a.down_off = td.offset;
        a.gu_stride = (uint64_t) a.gu_row * 2 * NFF;
        a.d_stride = (uint64_t) a.d_row * N;
        a.n_embd = N; a.n_ff = NFF; a.n_tok = 1; a.topk = TOPK; a.n_expert = NEXP;
        const bool dq2 = td.type == ref::Q2_0;
        const int gl = 32, gr = 1, gs = 4, ds_ = 4;
        // the down tile: Q2_0 8,1; else 4,2 - but on the M4 (no M5 layout) Q5_1 4,1 and Q8_0 8,1 (4,2 spills there:
        // 24-36 GB/s, mac-kern 513fc615)
        const bool m4 = !m5_layout();
        const int dl = dq2 || (m4 && td.type == ref::Q8_0) ? 8 : 4,
                  dr = dq2 || (m4 && (td.type == ref::Q8_0 || td.type == ref::Q5_1)) ? 1 : 2;
        a.gx_gu = NFF / (gs * (32 / gl) * gr);
        a.gx_dn = N / (ds_ * (32 / dl) * dr);
        mtl::Dispatch r;
        r.p = P(lib_, "moe_route", {}, "moe_route");
        r.args = {mtl::Arg::B(logits_), mtl::Arg::B(ids_), mtl::Arg::B(wts_), mtl::Arg::B(plan_), mtl::Arg::V(a)};
        r.tg[0] = 512;
        push(ds, r);
        {
            GlueArgs ga{NFF, MAXT, N, TOPK};
            mtl::Dispatch d;
            d.p = P(lib_, "swiglu_prep", {}, "swiglu_prep");
            d.args = {mtl::Arg::B(gsh_), mtl::Arg::B(ush_), mtl::Arg::B(XH_.b), mtl::Arg::V(ga)};
            d.grid[0] = NFF / 16; d.grid[1] = 1; d.tg[0] = 16;
            nb_ = true;
            push(ds, d);
        }
        const std::string gun = "moe_gu_" + std::to_string(tgu.type) + "_" + std::to_string(gl) + "_" + std::to_string(gr) +
                                "_" + std::to_string(gs);
        const std::string dnn = "moe_down_" + std::to_string(dq2 ? 142 : td.type) + "_" + std::to_string(dl) + "_" +
                                std::to_string(dr) + "_" + std::to_string(ds_);
        mtl::Dispatch g;
        g.p = P(lib_, gun, mtl::Constants().b(23, metal_ctab()), gun + (metal_ctab() ? "#ct" : ""));
        g.args = {mtl::Arg::B(Wm_), mtl::Arg::B(XF_.b, XF_.plane_bytes() * xkind(tgu.type)), mtl::Arg::B(h_), mtl::Arg::B(plan_),
                  mtl::Arg::V(a), mtl::Arg::B(XF_.b, XF_.plane_bytes() * 4)};
        g.tg[0] = 32 * gs;
        g.indirect = &plan_;
        g.ind_off = P_IGU * 4;
        push(ds, g);
        nb_ = true;
        gm(L + "mlp.shared_expert.down_proj.weight", XH_, ysh_, N, 1);
        mtl::Dispatch dd;
        dd.p = P(lib_, dnn, {}, dnn);
        dd.args = {mtl::Arg::B(Wm_), mtl::Arg::B(h_), mtl::Arg::B(ye_), mtl::Arg::B(plan_), mtl::Arg::V(a), mtl::Arg::B(h_)};
        dd.tg[0] = 32 * ds_;
        dd.indirect = &plan_;
        dd.ind_off = P_IDN * 4;
        push(ds, dd);
        GlueArgs ca{N, 1, N, TOPK};
        mtl::Dispatch cb;
        cb.p = P(lib_, "moe_combine_sh", {}, "moe_combine_sh");
        cb.args = {mtl::Arg::B(ye_), mtl::Arg::B(plan_), mtl::Arg::B(ysh_), mtl::Arg::B(sg_), mtl::Arg::B(mbo_), mtl::Arg::V(ca)};
        cb.grid[0] = (N + 255) / 256; cb.tg[0] = 256;
        push(ds, cb);
    }
    // the layer's write (R updated in place: the next draft step's input residual) and the final mixer
    gr_read_m(ds, "mtp.hyper_connection_mixer", mR_, mbo_, minj_[1], minj_[0], true, XA_, true, 1);
    const int64_t nv = dvocab_.empty() ? n_vocab_ : (int64_t) dvocab_.size();
    if (dvocab_.empty()) gemv(ds, T("output.weight"), XA_, mlm_, (size_t) n_vocab_, 1);
    else gemv(ds, mhead_t_, XA_, mlm_, (size_t) nv, 1, 0, &mhead_);
    {
        ArgArgs a{(uint32_t) nv, 120};
        mtl::Dispatch d1, d2;
        d1.p = P(lsmp_, "argmax_part", {}, "argmax_part");
        d1.args = {mtl::Arg::B(mlm_), mtl::Arg::B(apv_), mtl::Arg::B(api_), mtl::Arg::V(a)};
        d1.grid[0] = 120; d1.grid[1] = 1; d1.tg[0] = 256;
        push(ds, d1);
        d2.p = P(lsmp_, "argmax_final", {}, "argmax_final");
        d2.args = {mtl::Arg::B(apv_), mtl::Arg::B(api_), mtl::Arg::B(tokout_), mtl::Arg::V(a)};
        d2.grid[0] = 1; d2.tg[0] = 256;
        push(ds, d2);
    }
}

// the prompt's K/V: cell cell0 + i pairs the main window's final residual row i with the token at cell0 + i + 1
bool Window::mtp_prefill(const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err) {
    if (!mtp_on_) { err = "metal mtp: not loaded"; return false; }
    mrope_sync(cell0, n);
    // every group of MAXT rows in one command buffer: the embedding rows of all of them in mpemb_ (row c + i)
    if ((int64_t) mpemb_rows_ < n) {
        if (mpemb_rows_ > 0) cx_->free(mpemb_);
        mpemb_rows_ = (size_t) std::max<int64_t>(n, 64);
        mpemb_ = cx_->alloc(4 * mpemb_rows_ * N);
    }
    membp_ = &mpemb_;
    std::vector<mtl::Dispatch> ds;
    nb_ = false;
    for (int64_t c = 0; c < n; c += MAXT) {
        const int k = (int) std::min<int64_t>(MAXT, n - c);
        mtp_encode(ds, mpsrc_ ? *mpsrc_ : mtp_src_ ? *mtp_src_ : R_, (size_t) c * HC * N * 4, k, next_tokens + c,
                   cell0 + c, false, (int) c, 0);
    }
    membp_ = nullptr;
    cx_->run_graph(ds);
    return true;
}

void Window::mtp_pick(std::vector<mtl::Dispatch>& ds, int j, bool embed) {
    struct PickArgs { uint32_t nv, j, use_map; };
    const int64_t nv = dvocab_.empty() ? n_vocab_ : (int64_t) dvocab_.size();
    mtl::Dispatch d;
    d.p = P(lchain_, "mtp_pick", {}, "mtp_pick");
    PickArgs pa{(uint32_t) nv, (uint32_t) j, dvocab_.empty() ? 0u : 1u};
    d.args = {mtl::Arg::B(mlm_), mtl::Arg::B(tokout_), mtl::Arg::B(dmap_), mtl::Arg::B(dout_), mtl::Arg::B(dprob_),
              mtl::Arg::V(pa)};
    d.tg[0] = 1024;
    push(ds, d);
    if (!embed) return;
    const gguf::Tensor& te = T("token_embd.weight");
    struct EmbArgs { uint32_t n, row_bytes, j; } ea{(uint32_t) N, (uint32_t) ref::row_bytes(te.type, N), (uint32_t) j};
    mtl::Dispatch e;
    const std::string fn = "mtp_embed_" + std::to_string(te.type);
    e.p = P(lchain_, fn, {}, fn);
    e.args = {mtl::Arg::B(W_, te.offset), mtl::Arg::B(dout_), mtl::Arg::B(posv_), mtl::Arg::B(memb_), mtl::Arg::V(ea)};
    e.tg[0] = 256;
    push(ds, e);
}

// One round (mtp.hpp's draft()): the window's rows 0..a-1 caught up (K/V), then the full layer on row a (draft 0) and
// on its own output for the next drafts.  By default the whole round is one command buffer (the pick, its
// probability and the next step's embedding on the GPU); STRATA_METAL_MTP_CHAIN=0: a command buffer per step, the
// host picking and dequantizing.  The embedding rows: the catch-up's at memb_ rows 0..a-1, draft 0's at row a.
bool Window::mtp_draft(int Tn, const int32_t* tokens, int64_t p, int a, int max_steps, int32_t* drafts, float* probs,
                       float min_p, int* n_drafts, std::string& err) {
    if (!mtp_on_) { err = "metal mtp: not loaded"; return false; }
    (void) Tn;
    mrope_sync(p, a + max_steps + 1);
    int n = 0;
    int32_t tok = tokens[a];
    if (chain_ && max_steps > 0) {
        std::vector<mtl::Dispatch> ds;
        nb_ = false;
        if (a > 0) mtp_encode(ds, R_, 0, a, tokens, p, false);
        mtp_encode(ds, R_, (size_t) a * HC * N * 4, 1, &tok, p + a, true, a, 0);
        mtp_pick(ds, 0, max_steps > 1);
        for (int j = 1; j < max_steps; ++j) {
            mtp_encode(ds, mR_, 0, 1, nullptr, p + a + j, true, 0, j);
            mtp_pick(ds, j, j + 1 < max_steps);
        }
        cx_->run_graph(ds);
        for (int j = 0; j < max_steps; ++j) {
            const float pj = dprob_.as<float>()[j];
            drafts[j] = dout_.as<int32_t>()[j];
            if (probs) probs[j] = pj;
            ++n;
            if (min_p > 0.0f && pj < min_p) break;
        }
        if (n_drafts) *n_drafts = n;
        return true;
    }
    for (int j = 0; j < max_steps; ++j) {
        std::vector<mtl::Dispatch> ds;
        nb_ = false;
        if (j == 0) {
            if (a > 0) mtp_encode(ds, R_, 0, a, tokens, p, false);
            mtp_encode(ds, R_, (size_t) a * HC * N * 4, 1, &tok, p + a, true, a, 0);
        } else {
            mtp_encode(ds, mR_, 0, 1, &tok, p + a + j, true);
        }
        cx_->run_graph(ds);
        const int32_t di = tokout_.as<int32_t>()[0];
        const int32_t d = dvocab_.empty() ? di : dvocab_[(size_t) di];
        // the draft's probability: softmax of the draft head's logits at the pick
        const float* lg = mlm_.as<float>();
        const float mx = lg[di];
        const int64_t nv = dvocab_.empty() ? n_vocab_ : (int64_t) dvocab_.size();
        double s = 0.0;
        for (int64_t i = 0; i < nv; ++i) s += std::exp((double) (lg[i] - mx));
        const float pj = (float) (1.0 / s);
        drafts[j] = d;
        if (probs) probs[j] = pj;
        tok = d;
        ++n;
        if (min_p > 0.0f && pj < min_p) break;
    }
    if (n_drafts) *n_drafts = n;
    return true;
}

bool Window::copy_logits(int t, float* host) const {
    if (t < 0 || t >= last_T_) return false;
    std::memcpy(host, lm_.as<float>() + (size_t) t * n_vocab_, (size_t) n_vocab_ * 4);
    return true;
}


// ================================================================================== prompts through the matrix units
// A prompt chunk of T rows (mac-kern's tests/prompt_chain.cpp, on the engine's state): every dense matrix dequantized
// to fp16 once a chunk and multiplied on the matrix units (MPP), the GDN scan, QSA over the chunk, the routed experts
// as grouped GEMMs (v2 on the M4, v3 on the M5).  The GDN state, the conv history, the K/V and indexer cells and the
// PLE history are the decode window's own, so decoding goes on from them.  With SSD streaming the experts of a layer
// are put in place at the layer's handshake (the arena's misses) and the next layer's are streamed in behind it.
// STRATA_METAL_PROMPT_GEMM=0: prompts through the decode window (8 rows at a time).  Its numerics differ from the
// window's (fp16 operands): STRATA_METAL_PROMPT_GEMM_MIN (default 64) keeps shorter prompts on the window path.
namespace {
struct PgDq { uint32_t n_in, n_out, row_bytes, perm; };
struct PgGemm { int M, N, K, pad; };
struct PgEpi { uint32_t n_tok, n_embd, hc, hc_lr; };
struct PgGdnP { uint32_t n_tok, h_k, h_v, channels; float eps; uint32_t f32; };
struct PgPlan { uint32_t n_ent, topk, n_expert, tm, n_blocks; };
struct PgMm { uint64_t gate_off, up_off, down_off, gu_stride, d_stride; uint32_t gu_row, d_row, n_embd, n_ff; };
int pg_unit(int t) { return t == ref::Q2_0 || t == ref::IQ3_S || t == ref::BF16 || t == ref::F32 ? 16 : 32; }
// the v2 (M4) expert tiles: the K-quants need TK 128, Q8_0 / IQ4_XS TK 32 (their 32-value units), the rest TK 16
std::string gu2_tile(int t) {
    if (t == ref::Q4_K || t == ref::Q5_K || t == ref::Q6_K) return "_16_16_128_2";   // the M4's best (513fc615)
    if (t == ref::Q8_0 || t == ref::IQ4_XS) return "_16_64_32_2";
    return "_16_64_16_2";
}
std::string dn2_tile(int t) { return t == ref::IQ4_NL || t == ref::Q5_1 || t == ref::Q8_0 ? "_16_64_32_2" : "_16_64_16_2"; }
}  // namespace

struct Window::PromptGemm {
    int MT = 0, QS = 256, TM = 16, maxrows = 0, nblk_max = 0;
    int lib = -1, lgdn = -1, lqsa = -1;
    bool k4 = false, gu_v3 = false, dn_v3 = false, flash = false;
    mtl::Buffer R, bo, inj[2], xnh, V, loh, mixed, mixh, qkv, z, al, be, ygh, gqn, gkn, gvc, ggb, gkq, go, yg, qfull, kcur, vcur,
        iqr, ikr, qcur, qcurh, qidx, scores, qids, part, XQ, xqh, logits, ids, wts, bcnt, boff, eoff, tiles, meta, stok, sw, pos,
        XS, HS, YE, yr, gsh, ush, hh, ys, sgv, emb, embh, pkey, pval, keyn, qn, gated, normW;
    mtl::Buffer WD, WU, Wqkv, Wz, Wal, Wbe, Wout, Wq, Wk, Wv, Wiq, Wik, Wr, Wsg, Wshg, Wshu, Wshd, Wpk, Wpv;
    mtl::Buffer mixr;   // Gyro: the rotated gate/up input rows [MT][N]
};

int Window::prompt_chunk_rows() const { return pg_ ? pg_->MT : 0; }

bool Window::prompt_gemm_on(int64_t n) const {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_METAL_PROMPT_GEMM");
        return e ? std::string(e) != "0" : true;
    }();
    static const int64_t mn = [] {
        const char* e = std::getenv("STRATA_METAL_PROMPT_GEMM_MIN");
        return e ? (int64_t) std::atoll(e) : (int64_t) 64;
    }();
    return on && n >= mn && pg_ != nullptr;
}

bool Window::prompt_gemm_init(std::string& err) {
    if (pg_) return true;
    if (stream_ && !st_->ready()) return stream_arena(err);   // it makes the prompt buffers after the arena
    const char* ec = std::getenv("STRATA_METAL_PROMPT_CHUNK");
    // Streaming: every chunk makes one pass over the layer's experts that are not in the arena, so a longer chunk
    // streams each expert fewer times. M5 Max, UD-Q5, warm: 1024 -> 700 tok/s at 4K, 2048 -> 920, 4096 -> 1,100 (32K the
    // same). The longer chunk's buffers (4.3 GB) come out of the arena only while a prompt runs, so it is the default on a
    // big Mac; a small one keeps 1024 (the buffers would cost it cache).
    uint64_t phys = 0;
    size_t plen = sizeof phys;
    sysctlbyname("hw.memsize", &phys, &plen, nullptr, 0);
    const int stream_chunk = phys >= (96ull << 30) ? 4096 : 1024;
    const int C = std::max(64, ec ? std::atoi(ec) : (stream_ ? stream_chunk : 2048)) / 64 * 64;
    auto pg = std::make_unique<PromptGemm>();
    PromptGemm& p = *pg;
    p.MT = C;
    p.QS = std::min(C, 256);
    p.lib = cx_->compile({"quant.metal", "units.metal", "mma.metal", "dense_mm.metal", "moe_mm.metal", "gr.metal", "gemv.metal",
                          "prompt.metal"});
    p.lgdn = cx_->compile({"gdn.metal", "gdn_prompt.metal"}, "#define NO_GGML_COMMON 1\n");
    p.lqsa = cx_->compile({"qsa.metal", "qsa_prompt.metal"}, "#define NO_GGML_COMMON 1\n");
    p.k4 = ::coop_k4(*cx_);
    const char* eg = std::getenv("STRATA_METAL_PROMPT_EXPERTS");   // v2 | v3 (default: v3 with the M5's layout)
    p.gu_v3 = p.dn_v3 = eg ? std::string(eg) == "v3" : p.k4;
    if (T("blk.0.ffn_gate_exps.weight").type >= 144) p.gu_v3 = p.dn_v3 = false;   // the trellis experts have the v2 kernels only
    const char* eq = std::getenv("STRATA_METAL_PROMPT_QSA");       // v1 | flash (default: flash with the M5's layout)
    p.flash = eq ? std::string(eq) == "flash" : p.k4;
    const size_t MT = (size_t) C;
    p.maxrows = C * TOPK + NEXP * p.TM;
    p.nblk_max = (C * TOPK + 1023) / 1024;
    const int MB = (int) max_cells_ / 4 + 2;
    // the buffers: measured first, then placed - on a streaming Mac inside the expert arena (a region of its slots
    // reserved only while a prompt runs: the prompt's scratch costs no decode memory), else allocated
    uint64_t need = 0;
    uint8_t* region = nullptr;
    bool measure = true;
    auto AB = [&](size_t bytes) -> mtl::Buffer {
        const size_t b = (bytes + 16383) / 16384 * 16384;
        if (measure) { need += b; return mtl::Buffer{}; }
        if (region == nullptr) return cx_->alloc(bytes);
        mtl::Buffer r = cx_->wrap(region, b);
        region += b;
        return r;
    };
    auto F32 = [&](size_t n) { return AB(4 * n); };
    auto F16 = [&](size_t n) { return AB(2 * n); };
    auto make = [&] {
        const size_t mr = (size_t) p.maxrows;
        p.R = F32(MT * HC * N); p.bo = F32(MT * N); p.inj[0] = F32(MT * HC); p.inj[1] = F32(MT * HC); p.xnh = F16(MT * HC * N);
        p.V = F32(MT * (LR + HC)); p.loh = F16(MT * LR); p.mixed = F32(MT * N); p.mixh = F16(MT * N);
        p.qkv = F32(MT * 10240); p.z = F32(MT * 6144); p.al = F32(MT * 48); p.be = F32(MT * 48); p.yg = F32(MT * 6144);
        p.ygh = F16(MT * 6144); p.gqn = F32(MT * 16 * 128); p.gkn = F32(MT * 16 * 128); p.gvc = F32(MT * 6144);
        p.ggb = F32(MT * 96); p.gkq = F32(MT * 16); p.go = F32(MT * 6144);
        p.qfull = F32(MT * 12288); p.kcur = F32(MT * 512); p.vcur = F32(MT * 512); p.iqr = F32(MT * 512); p.ikr = F32(MT * 128);
        p.qcur = F32(MT * 24 * 256); p.qcurh = F16(MT * 24 * 256); p.qidx = F32(MT * 4 * 128); p.scores = F32(MT * (size_t) MB);
        p.qids = AB(4 * MT * 2051);
        p.part = p.flash ? F32(16) : F32((size_t) p.QS * 33 * 24 * 258);
        p.XQ = F32(4 * MT * 6144 + 3 * (size_t) p.QS * 6144 + MT * 384); p.xqh = F16(MT * 6144);
        p.logits = F32(MT * NEXP); p.ids = AB(4 * MT * TOPK); p.wts = F32(MT * TOPK);
        p.bcnt = AB(4 * (size_t) p.nblk_max * NEXP); p.boff = AB(4 * (size_t) p.nblk_max * NEXP);
        p.eoff = AB(4 * NEXP); p.tiles = AB(8 * (mr / p.TM + 1)); p.meta = AB(64);
        p.stok = AB(4 * mr); p.sw = F32(mr); p.pos = AB(4 * MT * TOPK); p.XS = F16(mr * N); p.HS = F16(mr * NFF);
        p.YE = F32(mr * N); p.yr = F32(MT * N); p.gsh = F32(MT * NFF); p.ush = F32(MT * NFF); p.hh = F16(MT * NFF);
        p.ys = F32(MT * N); p.sgv = F32(MT);
        if (had_on_) p.mixr = F32(MT * N);
        p.WD = F16((size_t) (LR + HC) * HC * N); p.WU = F16((size_t) HC * N * LR); p.Wqkv = F16((size_t) 10240 * N);
        p.Wz = F16((size_t) 6144 * N); p.Wal = F16((size_t) 48 * N); p.Wbe = F16((size_t) 48 * N); p.Wout = F16((size_t) N * 6144);
        p.Wq = F16((size_t) 12288 * N); p.Wk = F16((size_t) 512 * N); p.Wv = F16((size_t) 512 * N); p.Wiq = F16((size_t) 512 * N);
        p.Wik = F16((size_t) 128 * N); p.Wr = F16((size_t) NEXP * N); p.Wsg = F16((size_t) N); p.Wshg = F16((size_t) NFF * N);
        p.Wshu = F16((size_t) NFF * N); p.Wshd = F16((size_t) N * NFF);
        if (ple_on_) {
            p.emb = F32(MT * N); p.embh = F16(MT * N); p.pkey = F32(MT * 10240); p.pval = F32(MT * N); p.keyn = F32(MT * 10240);
            p.qn = F32(MT * 10240); p.gated = F32(MT * 10240); p.normW = F32(MT * 10240);
            p.Wpk = F16((size_t) 10240 * N); p.Wpv = F16((size_t) N * N);
        }
    };
    make();
    measure = false;
    if (stream_) {
        region = st_->region(need);
        if (region == nullptr) { err = "metal prompt: no arena region for the prompt buffers"; return false; }
    }
    make();
    pg_ = pg.release();
    std::fprintf(stderr, "strata metal: prompts on the matrix units in chunks of %d (experts %s, QSA %s, matmul2d layout %s; "
                         "%.2f GB of buffers%s)\n", C, p.gu_v3 ? "v3" : "v2", p.flash ? "flash" : "v1", p.k4 ? "K4" : "other",
                 need / 1e9, stream_ ? ", in the expert arena while a prompt runs" : "");
    (void) err;
    return true;
}

bool Window::prompt_chunk(const int32_t* tokens, int Tn, int64_t pos0, std::string& err) {
    PromptGemm& p = *pg_;
    p.meta.as<uint32_t>()[2] = p.gu_v3 ? NFF / (32 * 4) : NFF / 64;
    p.meta.as<uint32_t>()[5] = p.dn_v3 ? N / (64 * 2) : N / 64;
    auto TN = [this](const std::string& n) -> const gguf::Tensor& { return this->T(n); };
    if (Tn < 1 || Tn > p.MT) { err = "metal prompt: chunk size out of range"; return false; }
    if (pos0 + Tn + 1 >= max_cells_) { err = "metal prompt: the chunk runs past the context"; return false; }
    if (stream_ && !st_->ready() && !stream_arena(err)) return false;
    if (pend_fold_ > 0) {   // the last decode window's commit, still pending
        std::vector<mtl::Dispatch> fd;
        nb_ = false;
        fold(fd, pend_fold_, 1 - hp_);
        cx_->run_graph(fd);
        pend_fold_ = 0;
    }
    const int T = Tn;
    // ---- the embeddings into every stream, the PLE rows
    {
        const gguf::Tensor& te = TN("token_embd.weight");
        const size_t rb = ref::row_bytes(te.type, N);
        mrope_sync(pos0, T);
        for (int t = 0; t < T; ++t) {
            float* r = p.R.as<float>() + (size_t) t * HC * N;
            if (const float* img = img_row(pos0 + t)) std::memcpy(r, img, (size_t) N * 4);
            else ref::dequant(te.type, (const uint8_t*) W_.ptr + te.offset + (size_t) tokens[t] * rb, r, N);
            for (int c = 1; c < HC; ++c) std::memcpy(r + (size_t) c * N, r, (size_t) N * 4);
        }
    }
    strata::core::SessionState& ss = *ss_;
    if (ple_on_) {
        std::vector<uint32_t> rows((size_t) T * strata::kernels::PLE_N_HEADS);
        int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
        for (int t = 0; t < T; ++t) {
            strata::kernels::ngram_rows(&tokens[t], prev, 1, ss.ple.consts, rows.data() + (size_t) t * strata::kernels::PLE_N_HEADS);
            prev[0] = prev[1];
            prev[1] = tokens[t];
        }
        if (!ss.ple.table->gather_batch(rows.data(), (size_t) T, p.emb.as<float>(), err)) return false;
    }
    std::vector<mtl::Dispatch> ds;
    std::vector<int> cuts;
    nb_ = false;
    auto D = [&](const mtl::Pipeline* pp, std::vector<mtl::Arg> a, uint32_t gx, uint32_t gy, uint32_t tg, uint32_t gz = 1) {
        mtl::Dispatch d;
        d.p = pp;
        d.args = std::move(a);
        d.grid[0] = gx; d.grid[1] = gy; d.grid[2] = gz;
        d.tg[0] = tg;
        push(ds, d);
    };
    auto PP = [&](int lib, const std::string& fn, const mtl::Constants& c = {}, const std::string& key = "") {
        return P(lib, fn, c, "pg:" + (key.empty() ? fn : key));
    };
    auto gemm = [&](const mtl::Buffer& A, const mtl::Buffer& B, const mtl::Buffer& Y, int M, int Nn, int K) {
        const int tm = 64, tn = Nn <= 64 ? 32 : 64;
        const std::string fn = "gemm_f16_" + std::to_string(tm) + "_" + std::to_string(tn) + "_4";
        PgGemm g{M, Nn, K, 0};
        D(PP(p.lib, fn), {mtl::Arg::B(A), mtl::Arg::B(B), mtl::Arg::B(Y), mtl::Arg::V(g)}, (Nn + tn - 1) / tn, (M + tm - 1) / tm, 128);
    };
    auto dqb = [&](const mtl::Buffer& src, size_t off, int type, int n_in, int n_out, const mtl::Buffer& out, size_t out_off = 0,
                   uint32_t perm = 0) {
        PgDq a{(uint32_t) n_in, (uint32_t) n_out, (uint32_t) ref::row_bytes(type, n_in), perm};
        D(PP(p.lib, "dq_f16_" + std::to_string(type)), {mtl::Arg::B(src, off), mtl::Arg::B(out, out_off), mtl::Arg::V(a)},
          (n_in / pg_unit(type) + 63) / 64, n_out, 64);
    };
    auto dq = [&](const gguf::Tensor& t, const mtl::Buffer& out, size_t out_off = 0, uint32_t perm = 0) {
        const int n_in = (int) t.shape[0], n_out = t.shape.size() > 1 ? (int) t.shape[1] : 1;
        if (hc_q8_ && q8off_.count(t.name)) dqb(Q8_, q8off_.at(t.name), ref::Q8_0, n_in, n_out, out, out_off, perm);
        else dqb(W_, t.offset, t.type, n_in, n_out, out, out_off, perm);
    };
    auto act16 = [&](const mtl::Buffer& x, const mtl::Buffer& xh, size_t n) {
        const uint32_t nn = (uint32_t) n;
        D(PP(p.lib, "act_f16"), {mtl::Arg::B(x), mtl::Arg::B(xh), mtl::Arg::V(nn)}, (nn + 255) / 256, 1, 256);
    };
    int ci = 0;
    auto hc_read = [&](const std::string& pre, bool apply) {
        const gguf::Tensor &tn = TN(pre + "_norm.weight"), &tdn = TN(pre + "_down.weight"), &tin = TN(pre + "_inject.weight"),
                           &tup = TN(pre + "_up.weight");
        dq(tdn, p.WD);
        dq(tin, p.WD, (size_t) 2 * LR * HC * N);
        dq(tup, p.WU, 0, HC);
        GrArgs a{N, HC, LR, (uint32_t) T, apply ? 1u : 0u, eps_};
        PgEpi ea{(uint32_t) T, N, HC, LR};
        D(PP(p.lib, "gr_norm_h"), {mtl::Arg::B(p.R), mtl::Arg::B(p.bo), mtl::Arg::B(p.inj[ci]), mtl::Arg::B(W_, tn.offset),
                                   mtl::Arg::B(p.xnh), mtl::Arg::V(a)}, HC, T, 256);
        gemm(p.xnh, p.WD, p.V, T, LR + HC, HC * N);
        D(PP(p.lib, "gr_epi_down"), {mtl::Arg::B(p.V), mtl::Arg::B(p.loh), mtl::Arg::B(p.inj[1 - ci]), mtl::Arg::V(ea)},
          (T * (LR + HC) + 255) / 256, 1, 256);
        PgGemm g{T, HC * N, LR, 0};
        D(PP(p.lib, "gemm_hc_up_64_64_4"), {mtl::Arg::B(p.loh), mtl::Arg::B(p.WU), mtl::Arg::B(p.xnh), mtl::Arg::B(p.mixed),
                                            mtl::Arg::B(p.mixh), mtl::Arg::V(g), mtl::Arg::V(ea)}, HC * N / 64, (T + 63) / 64, 128);
        ci = 1 - ci;
    };
    // the PLE projections (the window's gemv pair as GEMMs)
    if (ple_on_) {
        act16(p.emb, p.embh, (size_t) T * N);
        dq(TN("blk.1.ple_key.weight"), p.Wpk);
        dq(TN("blk.1.ple_value.weight"), p.Wpv);
        gemm(p.embh, p.Wpk, p.pkey, T, 10240, N);
        gemm(p.embh, p.Wpv, p.pval, T, N, N);
    }
    const int MB = (int) max_cells_ / 4 + 2;
    (void) MB;
    for (int l = 0; l < 48; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        hc_read(b + "hc_attn", l > 0 && !(l == 1 && ple_on_));
        if (l % 4 != 3) {
            const int gi = l - l / 4;
            dq(TN(b + "attn_qkv.weight"), p.Wqkv);
            dq(TN(b + "attn_gate.weight"), p.Wz);
            dq(TN(b + "ssm_alpha.weight"), p.Wal);
            dq(TN(b + "ssm_beta.weight"), p.Wbe);
            dq(TN(b + "ssm_out.weight"), p.Wout);
            gemm(p.mixh, p.Wqkv, p.qkv, T, 10240, N);
            gemm(p.mixh, p.Wz, p.z, T, 6144, N);
            gemm(p.mixh, p.Wal, p.al, T, 48, N);
            gemm(p.mixh, p.Wbe, p.be, T, 48, N);
            PgGdnP ga{(uint32_t) T, 16, 48, 10240, eps_, 0};
            D(PP(p.lgdn, "gdn_pre"),
              {mtl::Arg::B(p.qkv), mtl::Arg::B(ghist_[hp_][(size_t) gi]), mtl::Arg::B(W_, TN(b + "ssm_conv1d.weight").offset),
               mtl::Arg::B(p.al), mtl::Arg::B(p.be), mtl::Arg::B(W_, TN(b + "ssm_dt.bias").offset), mtl::Arg::B(W_, TN(b + "ssm_a").offset),
               mtl::Arg::B(p.gqn), mtl::Arg::B(p.gkn), mtl::Arg::B(p.gvc), mtl::Arg::B(p.ggb), mtl::Arg::B(p.gkq),
               mtl::Arg::B(ghist_[1 - hp_][(size_t) gi]), mtl::Arg::V(ga)}, 16 + 24, T, 256);
            D(PP(p.lgdn, gdn_scan_name(cx_->device_name(), T), mtl::Constants().i(6, 0), gdn_scan_name(cx_->device_name(), T)),
              {mtl::Arg::B(gst_[(size_t) gi]), mtl::Arg::B(p.gqn), mtl::Arg::B(p.gkn), mtl::Arg::B(p.gvc), mtl::Arg::B(p.ggb),
               mtl::Arg::B(p.gkq), mtl::Arg::B(p.go), mtl::Arg::V(ga)}, 48 * 4, 1, 32 * 4);
            D(PP(p.lgdn, "gdn_post"), {mtl::Arg::B(p.go), mtl::Arg::B(p.z), mtl::Arg::B(W_, TN(b + "ssm_norm.weight").offset),
                                       mtl::Arg::B(p.ygh), mtl::Arg::B(p.yg), mtl::Arg::V(ga)}, 48, T, 128);
            gemm(p.ygh, p.Wout, p.bo, T, N, 6144);
        } else {
            const int qi = l / 4;
            dq(TN(b + "attn_q.weight"), p.Wq);
            dq(TN(b + "attn_k.weight"), p.Wk);
            dq(TN(b + "attn_v.weight"), p.Wv);
            dq(TN(b + "indexer.q_proj.weight"), p.Wiq);
            dq(TN(b + "indexer.k_proj.weight"), p.Wik);
            dq(TN(b + "attn_output.weight"), p.Wout);
            gemm(p.mixh, p.Wq, p.qfull, T, 12288, N);
            gemm(p.mixh, p.Wk, p.kcur, T, 512, N);
            gemm(p.mixh, p.Wv, p.vcur, T, 512, N);
            gemm(p.mixh, p.Wiq, p.iqr, T, 512, N);
            gemm(p.mixh, p.Wik, p.ikr, T, 128, N);
            QsaArgs a{(uint32_t) pos0, 0, (uint32_t) max_cells_, (uint32_t) T, eps_};
            D(PP(p.lqsa, "qsa_prep", qsa_c(), qsa_k()),
              {mtl::Arg::B(p.qfull), mtl::Arg::B(p.kcur), mtl::Arg::B(p.vcur), mtl::Arg::B(p.iqr), mtl::Arg::B(p.ikr),
               mtl::Arg::B(W_, TN(b + "attn_q_norm.weight").offset), mtl::Arg::B(W_, TN(b + "attn_k_norm.weight").offset),
               mtl::Arg::B(W_, TN(b + "indexer.q_norm.weight").offset), mtl::Arg::B(W_, TN(b + "indexer.k_norm.weight").offset),
               mtl::Arg::B(cs_), mtl::Arg::B(p.qcur), mtl::Arg::B(p.qidx), mtl::Arg::B(kq_[(size_t) qi]), mtl::Arg::B(vq_[(size_t) qi]),
               mtl::Arg::B(ksc_[(size_t) qi]), mtl::Arg::B(vsc_[(size_t) qi]), mtl::Arg::B(tail_[(size_t) qi]),
               mtl::Arg::B(dead_[(size_t) qi]), mtl::Arg::B(pooled_[(size_t) qi]), mtl::Arg::V(a), mtl::Arg::B(mrope_)}, 33, T, 256);
            if (pos0 + T > 2051)
                D(PP(p.lqsa, "qsa_score_mm"), {mtl::Arg::B(p.qidx), mtl::Arg::B(pooled_[(size_t) qi]), mtl::Arg::B(dead_[(size_t) qi]),
                                               mtl::Arg::B(p.scores), mtl::Arg::V(a)},
                  (uint32_t) (((pos0 + T) / 4 + 1 + 63) / 64), (T + 15) / 16, 128);
            D(PP(p.lqsa, "qsa_topk"), {mtl::Arg::B(p.scores), mtl::Arg::B(p.qids), mtl::Arg::V(a)}, T, 1, 1024);
            if (p.flash) {
                const uint32_t nq = (uint32_t) T * 24 * 256;
                D(PP(p.lqsa, "qsa_q_f16"), {mtl::Arg::B(p.qcur), mtl::Arg::B(p.qcurh), mtl::Arg::V(nq)}, (nq + 255) / 256, 1, 256);
                D(PP(p.lqsa, "qsa_flash", mtl::Constants().b(7, p.k4), "qsa_flash"),
                  {mtl::Arg::B(p.qcurh), mtl::Arg::B(kq_[(size_t) qi]), mtl::Arg::B(vq_[(size_t) qi]), mtl::Arg::B(ksc_[(size_t) qi]),
                   mtl::Arg::B(vsc_[(size_t) qi]), mtl::Arg::B(p.qids), mtl::Arg::B(p.qfull), mtl::Arg::B(p.xqh), mtl::Arg::V(a)},
                  2, T, 128);
            } else {
                // v1 attention + merge in slices of QS rows (the partials of a slice only)
                for (int t0 = 0; t0 < T; t0 += p.QS) {
                    const int ns = std::min(p.QS, T - t0);
                    QsaArgs as{(uint32_t) (pos0 + t0), 0, (uint32_t) max_cells_, (uint32_t) ns, eps_};
                    D(PP(p.lqsa, "qsa_attn"), {mtl::Arg::B(p.qcur, (size_t) t0 * 24 * 256 * 4), mtl::Arg::B(kq_[(size_t) qi]),
                                               mtl::Arg::B(vq_[(size_t) qi]), mtl::Arg::B(ksc_[(size_t) qi]), mtl::Arg::B(vsc_[(size_t) qi]),
                                               mtl::Arg::B(p.qids, (size_t) t0 * 2051 * 4), mtl::Arg::B(p.part), mtl::Arg::V(as)},
                      33, 2, 256, (uint32_t) ns);
                    D(PP(p.lqsa, "qsa_merge"), {mtl::Arg::B(p.part), mtl::Arg::B(p.qfull, (size_t) t0 * 12288 * 4),
                                                mtl::Arg::B(p.XQ, (size_t) t0 * 6144 * 4), mtl::Arg::V(as)}, 24, ns, 256);
                }
                act16(p.XQ, p.xqh, (size_t) T * 6144);
            }
            gemm(p.xqh, p.Wout, p.bo, T, N, 6144);
        }
        if (l == 0 && ple_on_) {
            // layer 0's attention write applied to R happens in the FFN read below; the PLE block comes after the FFN
        }
        // ---- the MoE
        hc_read(b + "hc_ffn", true);
        const gguf::Tensor &tr = TN(b + "ffn_gate_inp.weight"), &tsg = TN(b + "ffn_gate_inp_shexp.weight"),
                           &tg_ = TN(b + "ffn_gate_shexp.weight"), &tu_ = TN(b + "ffn_up_shexp.weight"),
                           &td_ = TN(b + "ffn_down_shexp.weight"), &eg = TN(b + "ffn_gate_exps.weight"),
                           &ed = TN(b + "ffn_down_exps.weight");
        dq(tr, p.Wr);
        dq(tsg, p.Wsg);
        dq(tg_, p.Wshg);
        dq(tu_, p.Wshu);
        dq(td_, p.Wshd);
        gemm(p.mixh, p.Wr, p.logits, T, NEXP, N);
        const uint32_t nt = (uint32_t) T;
        D(PP(p.lib, "route_topk"), {mtl::Arg::B(p.logits), mtl::Arg::B(p.ids), mtl::Arg::B(p.wts), mtl::Arg::V(nt)}, (T + 7) / 8, 1, 256);
        const int nent = T * TOPK, nblk = (nent + 1023) / 1024;
        PgPlan pa{(uint32_t) nent, TOPK, NEXP, (uint32_t) p.TM, (uint32_t) nblk};
        const uint32_t nclr = (uint32_t) p.maxrows;
        D(PP(p.lib, "mm_clear"), {mtl::Arg::B(p.stok), mtl::Arg::V(nclr)}, (p.maxrows + 255) / 256, 1, 256);
        D(PP(p.lib, "mm_hist"), {mtl::Arg::B(p.ids), mtl::Arg::B(p.bcnt), mtl::Arg::V(pa)}, nblk, 1, 1024);
        D(PP(p.lib, "mm_scan"), {mtl::Arg::B(p.bcnt), mtl::Arg::B(p.boff), mtl::Arg::B(p.eoff), mtl::Arg::B(p.tiles),
                                 mtl::Arg::B(p.meta), mtl::Arg::V(pa)}, 1, 1, NEXP);
        D(PP(p.lib, "mm_place"), {mtl::Arg::B(p.ids), mtl::Arg::B(p.wts), mtl::Arg::B(p.boff), mtl::Arg::B(p.eoff),
                                  mtl::Arg::B(p.stok), mtl::Arg::B(p.sw), mtl::Arg::B(p.pos), mtl::Arg::V(pa)}, nblk, 1, 1024);
        PgMm ma{eg.offset, TN(b + "ffn_up_exps.weight").offset, ed.offset, 0, 0, (uint32_t) ref::row_bytes(eg.type, N),
                (uint32_t) ref::row_bytes(ed.type, NFF), N, NFF};
        ma.gu_stride = (uint64_t) ma.gu_row * NFF;
        ma.d_stride = (uint64_t) ma.d_row * N;
        if (stream_) { ma.gate_off = 0; ma.up_off = st_->up_in_slot(l); ma.down_off = st_->down_in_slot(l); }
        const size_t ebase = moe_rebase(ma.gate_off, ma.up_off, ma.down_off);
        const uint32_t ne_ = N;
        const bool hg = had_gu(l), hd = had_dn(l);
        if (hg) had_rows(ds, p.lib, "pg:", p.mixed, 0, p.mixr, 0, T, N, hsx_.handle ? &hsx_ : nullptr, false);
        {
            mtl::Dispatch g;
            g.p = PP(p.lib, "mm_gather");
            g.args = {mtl::Arg::B(hg ? p.mixr : p.mixed), mtl::Arg::B(p.stok), mtl::Arg::B(p.XS), mtl::Arg::V(ne_)};
            g.tg[0] = 256;
            g.indirect = &p.meta;
            g.ind_off = 8 * 4;
            push(ds, g);
        }
        const mtl::Buffer& EW = stream_ ? st_->chunk(0) : W_;
        auto arena_args = [&](mtl::Dispatch& d) {
            if (!stream_) return;
            d.args.push_back(mtl::Arg::B(st_->smap(), (size_t) l * 512 * 4));
            for (int c = 1; c < 4; ++c) d.args.push_back(mtl::Arg::B(st_->chunk(c)));
        };
        const std::string ak = stream_ ? "#arena" : "";
        mtl::Dispatch gu;
        if (p.gu_v3) {
            const std::string fn = "mm_gu3_" + std::to_string(eg.type) + "_16_32_32_4";
            mtl::Constants c = mtl::Constants().b(7, p.k4);
            if (stream_) c.b(20, true);
            gu.p = PP(p.lib, fn, c, fn + ak);
            gu.tg[0] = 32 * 4;
        } else {
            const std::string fn = "mm_gu2_" + std::to_string(eg.type) + gu2_tile(eg.type);
            mtl::Constants c = mtl::Constants().i(5, 0);
            if (stream_) c.b(20, true);
            gu.p = PP(p.lib, fn, c, fn + ak);
            gu.tg[0] = 64;
        }
        gu.args = {mtl::Arg::B(EW, ebase), mtl::Arg::B(p.XS), mtl::Arg::B(p.HS), mtl::Arg::B(p.tiles), mtl::Arg::V(ma)};
        arena_args(gu);
        gu.indirect = &p.meta;
        gu.ind_off = 2 * 4;
        if (stream_) cuts.push_back((int) ds.size());   // the host's turn: this layer's experts put in place
        push(ds, gu);
        if (hd) had_rows_ind(ds, p.lib, "pg:", p.HS, p.meta);
        mtl::Dispatch dn;
        if (p.dn_v3) {
            const std::string fn = "mm_down3_" + std::to_string(ed.type) + "_16_64_32_2";
            mtl::Constants c = mtl::Constants().b(7, p.k4);
            if (stream_) c.b(20, true);
            dn.p = PP(p.lib, fn, c, fn + ak);
            dn.tg[0] = 32 * 2;
        } else {
            const std::string fn = "mm_down2_" + std::to_string(ed.type) + dn2_tile(ed.type);
            mtl::Constants c;
            if (stream_) c.b(20, true);
            dn.p = PP(p.lib, fn, c, fn + ak);
            dn.tg[0] = 64;
        }
        dn.args = {mtl::Arg::B(EW, ebase), mtl::Arg::B(p.HS), mtl::Arg::B(p.YE), mtl::Arg::B(p.tiles), mtl::Arg::V(ma)};
        arena_args(dn);
        dn.indirect = &p.meta;
        dn.ind_off = 5 * 4;
        push(ds, dn);
        const uint32_t ca[4] = {(uint32_t) T, N, TOPK, 0};
        D(PP(p.lib, "mm_combine"), {mtl::Arg::B(p.YE), mtl::Arg::B(p.pos), mtl::Arg::B(p.sw), mtl::Arg::B(p.yr), mtl::Arg::V(ca)},
          (T * N + 255) / 256, 1, 256);
        gemm(p.mixh, p.Wshg, p.gsh, T, NFF, N);
        gemm(p.mixh, p.Wshu, p.ush, T, NFF, N);
        const uint32_t nh = (uint32_t) T * NFF;
        D(PP(p.lib, "swiglu_h"), {mtl::Arg::B(p.gsh), mtl::Arg::B(p.ush), mtl::Arg::B(p.hh), mtl::Arg::V(nh)}, (nh + 255) / 256, 1, 256);
        gemm(p.hh, p.Wshd, p.ys, T, N, NFF);
        gemm(p.mixh, p.Wsg, p.sgv, T, 1, N);
        const uint32_t oa[2] = {(uint32_t) T, N};
        D(PP(p.lib, "moe_out"), {mtl::Arg::B(p.yr), mtl::Arg::B(p.ys), mtl::Arg::B(p.sgv), mtl::Arg::B(p.bo), mtl::Arg::V(oa)},
          (T * N + 255) / 256, 1, 256);
        if (l == 0 && ple_on_) {
            // layer 0's FFN write applied to R, then the PLE block (as the window, for T rows)
            const gguf::Tensor& tn = TN("blk.1.hc_attn_norm.weight");
            GrArgs a{N, HC, LR, (uint32_t) T, 1u, eps_};
            D(P(lib_, "gr_norm", {}, "gr_norm"), {mtl::Arg::B(p.R), mtl::Arg::B(p.bo), mtl::Arg::B(p.inj[ci]), mtl::Arg::B(W_, tn.offset),
                                                  mtl::Arg::B(p.YE), mtl::Arg::V(a)}, HC, T, 256);
            struct PleArgs { uint32_t n_embd, hc; float eps; } pl{N, HC, strata::kernels::NG_RMS_EPS};
            auto off = [&](const char* n) { return TN(n).offset; };
            D(P(lple_, "ple_gnorm", {}, "ple_gnorm"), {mtl::Arg::B(p.pkey), mtl::Arg::B(W_, off("blk.1.ple_norm_key.weight")),
                                                      mtl::Arg::B(p.keyn), mtl::Arg::V(pl)}, HC, T, 256);
            nb_ = true;
            D(P(lple_, "ple_gnorm", {}, "ple_gnorm"), {mtl::Arg::B(p.R), mtl::Arg::B(W_, off("blk.1.ple_norm_query.weight")),
                                                      mtl::Arg::B(p.qn), mtl::Arg::V(pl)}, HC, T, 256);
            D(P(lple_, "ple_gate_norm", {}, "ple_gate_norm"),
              {mtl::Arg::B(p.keyn), mtl::Arg::B(p.qn), mtl::Arg::B(p.pval), mtl::Arg::B(W_, off("blk.1.ple_norm_conv.weight")),
               mtl::Arg::B(p.gated), mtl::Arg::B(p.normW), mtl::Arg::V(pl)}, HC, T, 256);
            D(P(lple_, "ple_conv_add", {}, "ple_conv_add"),
              {mtl::Arg::B(pleHist_), mtl::Arg::B(p.normW), kw_arg(), mtl::Arg::B(p.gated),
               mtl::Arg::B(p.R), mtl::Arg::V(pl)}, HC * N / 256, T, 256);
        }
    }
    // the last layer's write applied to R (the draft layer's K/V read the final residual rows)
    {
        const gguf::Tensor& tn = TN("output_hc_norm.weight");
        GrArgs a{N, HC, LR, (uint32_t) T, 1u, eps_};
        D(P(lib_, "gr_norm", {}, "gr_norm"), {mtl::Arg::B(p.R), mtl::Arg::B(p.bo), mtl::Arg::B(p.inj[ci]), mtl::Arg::B(W_, tn.offset),
                                              mtl::Arg::B(p.YE), mtl::Arg::V(a)}, HC, T, 256);
    }
    const double t0 = now_ms();
    if (stream_) {
        seg_T_ = T;
        cut_ids_ = p.ids.as<int32_t>();
        cx_->run_graph_seg(ds, cuts, [](void* u, int s) { ((Window*) u)->at_cut(s); }, this);
        cut_ids_ = nullptr;
    } else {
        cx_->run_graph(ds);
    }
    pg_ms_ += now_ms() - t0;
    pg_tok_ += T;
    static const bool trace = std::getenv("STRATA_METAL_TRACE") != nullptr;
    if (trace) std::fprintf(stderr, "strata metal: prompt chunk pos %lld, %d rows: %.0f ms\n", (long long) pos0, T, now_ms() - t0);
    // the window's state after the chunk: the conv history turned, the PLE history and token window advanced
    hp_ = 1 - hp_;
    if (ple_on_) {
        using namespace strata::kernels;
        float* hist = pleHist_.as<float>();
        const float* nw = p.normW.as<float>();
        for (int c = 0; c < NG_HC_DIM; ++c) {
            float col[NG_HIST];
            for (int r = 0; r < NG_HIST; ++r) {
                const int src = T - NG_HIST + r;   // row r of the new history: chunk row src, or the old history
                col[r] = src >= 0 ? nw[(size_t) src * NG_HC_DIM + c] : hist[(size_t) c * NG_HIST + (size_t) (r + T)];
            }
            std::memcpy(hist + (size_t) c * NG_HIST, col, sizeof col);
        }
    }
    if (T >= 2) { ss.ple_prev[0] = tokens[T - 2]; ss.ple_prev[1] = tokens[T - 1]; }
    else { ss.ple_prev[0] = ss.ple_prev[1]; ss.ple_prev[1] = tokens[0]; }
    mtp_src_ = &p.R;   // the draft layer's prompt K/V read this chunk's final rows
    return true;
}

// ---------------------------------------------------------------------------------------------- the hook's entry points
namespace {
std::string g_shard1;
}
void window_register(void* w);
bool mtp_load_pending(Window* w, std::string& err);
void set_model_shard(const char* shard1) { g_shard1 = shard1 ? shard1 : ""; }
void set_batch_slots(int n) { g_slot_cap = n > 0 ? std::min(n, 8) : 0; }

// One decode window per process: the prompt path (Prefill::init) may bring it up before the Verifier does.
namespace {
Window* g_the_window = nullptr;
}
void* window_create() {
    if (g_the_window == nullptr) g_the_window = new Window();
    return g_the_window;
}
void window_destroy(void* w) {
    if (w != nullptr && w == g_the_window) {
        if (g_stream_window == g_the_window) {
            g_the_window->stream_report(stderr);
            g_stream_window = nullptr;
        }
        delete g_the_window;
        g_the_window = nullptr;
        window_register(nullptr);
    }
}
bool window_init(void* w, const strata::core::ModelGeometry& g, strata::core::SessionState& ss, int max_t, int64_t max_cells,
                 int64_t* n_vocab, std::string& err) {
    Window* x = (Window*) w;
    if (!x->ready()) {
        if (g_shard1.empty()) { err = "metal window: the model shard is not set (--native SHARD1)"; return false; }
        if (!x->init(g, ss, g_shard1, MAXT, max_cells, err)) return false;
        window_register(w);
        if (!mtp_load_pending(x, err)) return false;
    }
    x->set_max_t(max_t);
    *n_vocab = x->vocab();
    return true;
}
bool window_ensure(const strata::core::ModelGeometry& g, strata::core::SessionState& ss, int64_t max_cells, std::string& err) {
    int64_t nv = 0;
    return window_init(window_create(), g, ss, MAXT, max_cells, &nv, err);
}
bool window_run(void* w, int T, const int32_t* tokens, int64_t pos0, bool head, int32_t* out, std::string& err) {
    // STRATA_LOGPOS (the quant yardstick, the KL harness): every window's head logits are read, also the prompt's
    // windows that sample nothing (head_sampling off)
    static const bool logpos = std::getenv("STRATA_LOGPOS") != nullptr;
    return ((Window*) w)->run(T, tokens, pos0, head || logpos, out, err);
}
void window_reset(const strata::core::SessionState* ss) {
    if (g_the_window == nullptr || !g_the_window->ready()) return;
    g_the_window->activate(g_the_window->set_of(ss));
    g_the_window->reset();
    g_the_window->activate(0);
}
bool window_checkpoint_save(strata::core::ConversationCheckpoint& c, const strata::core::SessionState* ss, std::string& err) {
    if (g_the_window == nullptr || !g_the_window->ready()) { err = "metal window: not up"; return false; }
    g_the_window->activate(g_the_window->set_of(ss));   // a batch slot's session: that slot's running state
    const bool ok = g_the_window->checkpoint_save(c, err);
    g_the_window->activate(0);
    return ok;
}
bool window_checkpoint_restore(const strata::core::ConversationCheckpoint& c, const strata::core::SessionState* ss, std::string& err) {
    if (g_the_window == nullptr || !g_the_window->ready()) { err = "metal window: not up"; return false; }
    g_the_window->activate(g_the_window->set_of(ss));
    const bool ok = g_the_window->checkpoint_restore(c, err);
    g_the_window->activate(0);
    return ok;
}
bool window_init_slots(void* w, const std::vector<strata::core::SessionState*>& slots, std::string& err) {
    return ((Window*) w)->slots_init(slots, err);
}
bool window_run_slots(void* w, const int* rows, int S, const int32_t* tokens, const int64_t* pos, int32_t* out, std::string& err) {
    const double t0 = now_ms();
    const bool ok = ((Window*) w)->run_slots(rows, S, tokens, pos, out, err);
    ((Window*) w)->prof_add(true, now_ms() - t0);
    return ok;
}
bool window_slot_copy_main(bool into_slot, int slot, int64_t upto, bool kv_only, std::string& err) {
    if (g_the_window == nullptr || !g_the_window->ready()) { err = "metal window: not up"; return false; }
    return g_the_window->slot_copy(into_slot, slot, upto, kv_only, err);
}
bool window_commit(void* w, int n_keep, std::string& err) {
    const double t0 = now_ms();
    const bool ok = ((Window*) w)->commit(n_keep, err);
    ((Window*) w)->prof_add(false, now_ms() - t0);
    return ok;
}
bool window_copy_logits(void* w, int t, float* host) { return ((Window*) w)->copy_logits(t, host); }

// the MTP hook (src/core/mtp.cpp under STRATA_GPU_METAL): the drafter lives beside the decode window it reads
namespace {
Window* g_window = nullptr;
}
void window_register(void* w) { g_window = (Window*) w; }
// the engine loads the drafter before the decode window exists: the path is kept and the window loads it at its init
std::string g_mtp_path;
int g_mtp_max_t = 0;
bool mtp_load(const std::string& rt_dir, int max_t, std::string& err) {
    (void) err;
    // the MTP model in GGUF form (tools/mtp_pack.py) beside the rt directory: STRATA_METAL_MTP_GGUF, else
    // <rt>/../mtp-q2_0.gguf
    std::string path;
    if (const char* e = std::getenv("STRATA_METAL_MTP_GGUF")) path = e;
    else {
        std::string d = rt_dir;
        while (!d.empty() && d.back() == '/') d.pop_back();
        const size_t s = d.rfind('/');
        path = (s == std::string::npos ? std::string(".") : d.substr(0, s)) + "/mtp-q2_0.gguf";
    }
    g_mtp_path = path;
    g_mtp_max_t = max_t;
    return true;
}
bool mtp_load_pending(Window* w, std::string& err) {
    return g_mtp_path.empty() || w->mtp_load(g_mtp_path, g_mtp_max_t, err);
}
// Prefill::run's Metal side: the prompt read through the window, MAXT rows at a time (no head), each window
// committed; on_chunk (the MTP's K/V) after each, with the window's final residual rows in place.
bool prompt_run_rows(const int64_t* tokens, int64_t n, int64_t pos0,
                     const std::function<bool(const float*, int64_t, int64_t, std::string&)>& on_chunk, std::string& err);
bool prompt_run(const int64_t* tokens, int64_t n, int64_t pos0, const float* const* embd_rows,
                const std::function<bool(const float*, int64_t, int64_t, std::string&)>& on_chunk, std::string& err) {
    struct Rows {
        explicit Rows(const float* const* r) { g_embd_rows = r; }
        ~Rows() { g_embd_rows = nullptr; }
    } rows(embd_rows);
    return prompt_run_rows(tokens, n, pos0, on_chunk, err);
}
bool prompt_run_rows(const int64_t* tokens, int64_t n, int64_t pos0,
                     const std::function<bool(const float*, int64_t, int64_t, std::string&)>& on_chunk, std::string& err) {
    if (g_window == nullptr) { err = "metal prompt: the decode window is not up yet"; return false; }
    static const bool force_windows = std::getenv("STRATA_METAL_PROMPT") && std::string(std::getenv("STRATA_METAL_PROMPT")) == "window";
    static const char* latent_prompt = std::getenv("STRATA_METAL_LATENT_PROMPT");
    if (g_window->latent() && !force_windows && latent_prompt && std::strcmp(latent_prompt, "1") == 0 && n >= 128)
        return g_window->prompt(tokens, n, pos0, on_chunk, err);
    const bool windows = force_windows || g_window->latent();
    static const int64_t min_rows = std::getenv("STRATA_METAL_PROMPT_MIN") ? std::atoll(std::getenv("STRATA_METAL_PROMPT_MIN")) : 32;
    if (!windows && !g_window->streaming() && n >= min_rows) return g_window->prompt(tokens, n, pos0, on_chunk, err);
    if (!windows && g_window->streaming()) {
        const char* e = std::getenv("STRATA_METAL_PROMPT_GEMM");
        const char* em = std::getenv("STRATA_METAL_PROMPT_GEMM_MIN");
        const int64_t mn = em ? std::atoll(em) : 64;
        if (!(e && std::string(e) == "0") && n >= mn) {
            if (!g_window->prompt_gemm_init(err)) return false;
            const int C = g_window->prompt_chunk_rows();
            std::vector<int32_t> tk32((size_t) C);
            g_window->prompt_region(true);
            bool ok = true;
            for (int64_t c = 0; c < n && ok; c += C) {
                const int k = (int) std::min<int64_t>(C, n - c);
                for (int i = 0; i < k; ++i) tk32[(size_t) i] = (int32_t) tokens[c + i];
                ok = g_window->prompt_chunk(tk32.data(), k, pos0 + c, err) && (!on_chunk || on_chunk(nullptr, k, pos0 + c, err));
            }
            g_window->prompt_region(false);
            return ok;
        }
    }
    int32_t tk[MAXT], out[MAXT];
    for (int64_t c = 0; c < n; c += MAXT) {
        const int k = (int) std::min<int64_t>(MAXT, n - c);
        for (int i = 0; i < k; ++i) tk[i] = (int32_t) tokens[c + i];
        if (!g_window->run(k, tk, pos0 + c, false, out, err) || !g_window->commit(k, err)) return false;
        if (on_chunk && !on_chunk(nullptr, k, pos0 + c, err)) return false;
    }
    if (n >= 1024) g_window->report_profile();
    return true;
}
bool mtp_prefill(const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err) {
    if (g_window == nullptr) { err = "metal mtp: no decode window"; return false; }
    return g_window->mtp_prefill(next_tokens, n, cell0, err);
}
bool mtp_draft(int T, const int32_t* tokens, int64_t p, int a, int max_steps, int32_t* drafts, float* probs, float min_p,
               int* n_drafts, std::string& err) {
    if (g_window == nullptr) { err = "metal mtp: no decode window"; return false; }
    const double t0 = now_ms();
    const bool ok = g_window->mtp_draft(T, tokens, p, a, max_steps, drafts, probs, min_p, n_drafts, err);
    g_window->prof_add(true, now_ms() - t0);
    return ok;
}

#include "prompt_chain.inc"

}  // namespace strata::metal_engine
namespace strata::gpu { void* host_view(const void* device_pointer); }
namespace strata::kernels {
void mrope_table_set(const int32_t* device_table) {
    strata::metal_engine::g_mrope_dev = device_table;
    strata::metal_engine::g_mrope_host =
        device_table ? (const int32_t*) strata::gpu::host_view(device_table) : nullptr;
}
const int32_t* mrope_table() { return strata::metal_engine::g_mrope_dev; }
}  // namespace strata::kernels
namespace strata::metal_engine {

}  // namespace strata::metal_engine
