// src/core/mtp.cpp - see include/strata/core/mtp.hpp.
#include "strata/core/mtp.hpp"
#include "strata/core/pin_budget.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/core/coupled_draft.hpp"
#include "strata/core/spec_prob.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/progress.hpp"

#include "strata/core/native_head.hpp"
#include "strata/core/peer_experts.hpp"
#include "strata/platform/memory.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/gfx_arch.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <sstream>
#include <vector>
#include "strata/platform/cpu_relax.hpp"

#ifdef STRATA_NATIVE_EXPERTS
#include "ggml.h"   // --mtp-q4: the head rows' and projections' formats, dequantized and requantized at load
#endif

#if defined(STRATA_GPU_METAL)
// The Metal backend's MTP drafter (metal/src/engine/window.cpp, beside the decode window it reads): load, prefill and
// draft hand over to it; bind has nothing to bind there.
namespace strata::metal_engine {
bool mtp_load(const std::string& rt_dir, int max_t, std::string& err);
bool mtp_prefill(const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err);
bool mtp_draft(int T, const int32_t* tokens, int64_t p, int a, int max_steps, int32_t* drafts, float* probs, float min_p,
               int* n_drafts, std::string& err);
}  // namespace strata::metal_engine
#endif

namespace strata::core {
namespace {

constexpr float EPS = 1e-6f;

// #783 PR-i: STRATA_MTP_CATCHUP_ALL=1 catches the drafter's K/V up for the whole verified window, rejected rows included
bool mtp_catchup_all() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_MTP_CATCHUP_ALL");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return on;
}
// STRATA_MTP_Q8_SM60=1 (opt-in, with STRATA_Q8_SM60=1): the drafter's own Q8_0 matrices (the draft head's subset,
// the projections) run on the Pascal GEMV too (the drafts differ in rounding only: the Pascal kernel sums in another
// order; the replies do not depend on the drafts)
bool mtp_q8_sm60() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_MTP_Q8_SM60");
        return v != nullptr && v[0] == '1';
    }();
    return on;
}
constexpr int GGML_Q8_0 = 8;
constexpr int GGML_Q4_0 = 2;
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

struct Bump {
    uint8_t* base = nullptr;
    uint64_t used = 0;
    template <typename T> T* take(uint64_t n) {
        T* p = base ? (T*) (base + used) : nullptr;
        used += (n * sizeof(T) + 255) & ~255ull;
        return p;
    }
};

bool mapped(size_t bytes, void** h, void** d) {
    if (strata::core::host_alloc_guarded(h, bytes, cudaHostAllocMapped | (peer_portable() ? cudaHostAllocPortable : 0)) != cudaSuccess) return false;
    std::memset(*h, 0, bytes);
    return cudaHostGetDevicePointer(d, *h, 0) == cudaSuccess;
}

strata::kernels::QsaShapes shapes_of(const ModelGeometry& g) { return qsa_shapes(g); }

// 64-bit seek/tell on a `FILE*`: `fseek`/`ftell` take a 32-bit `long` on Windows and would wrap past 2 GiB.
#if defined(_WIN32)
#define STRATA_FILE_SEEK64(f, o, w) _fseeki64((f), (long long) (o), (w))
#define STRATA_FILE_TELL64(f) _ftelli64(f)
#else
#define STRATA_FILE_SEEK64(f, o, w) fseeko((f), (off_t) (o), (w))
#define STRATA_FILE_TELL64(f) ftello(f)
#endif

bool read_file(const std::string& path, std::vector<uint8_t>& out) {
    // Loader fix (0.1.15+loaderfix.2): `ifstream::read` reaches the disk as 4095-byte reads on MSVC - the same
    // split `load_experts_ranges` had - so the drafter's 111 MiB dense blob paid 4 KiB per operation on a cold
    // start.  `fread` passes a request bigger than the stream buffer straight to `ReadFile()`.
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    // A `FILE*` has no destructor and the short-read path below returns early: the guard closes on every path.
    struct Closer {
        FILE* f;
        ~Closer() { if (f != nullptr) std::fclose(f); }
    } closer{f};
    if (STRATA_FILE_SEEK64(f, 0, SEEK_END) != 0) return false;
    const long long n = (long long) STRATA_FILE_TELL64(f);
    if (n < 0 || STRATA_FILE_SEEK64(f, 0, SEEK_SET) != 0) return false;
#if !defined(_WIN32)
    strata::platform::advise_willneed(fileno(f), 0, (uint64_t) n);
#endif
    out.resize((size_t) n);
    return n == 0 || std::fread(out.data(), 1, (size_t) n, f) == (size_t) n;
}

}  // namespace

MtpDrafter::~MtpDrafter() {
    if (cs_) cudaStreamSynchronize(cs_);
    for (auto& e : prefill_exec_) if (e) cudaGraphExecDestroy(e);
    for (auto& e : prefill_dev_exec_) if (e) cudaGraphExecDestroy(e);
    if (pf_dev_) cudaFree(pf_dev_);
    if (dense4_) cudaFree(dense4_);
    for (auto& e : round_exec_) if (e) cudaGraphExecDestroy(e);
    for (auto& e : step_exec_) if (e) cudaGraphExecDestroy(e);
    for (auto& e : round_exec_c_) if (e) cudaGraphExecDestroy(e);
    for (auto& e : step_exec_c_) if (e) cudaGraphExecDestroy(e);
    if (cparams_) cudaFree(cparams_);
    if (cring_) cudaFree(cring_);
    if (dinv_) cudaFree(dinv_);
    if (cscratch_) cudaFree(cscratch_);
    if (h_cparams_) cudaFreeHost(h_cparams_);
    if (h_chist_) cudaFreeHost(h_chist_);
    if (h_q_) cudaFreeHost(h_q_);
    if (cs_) cudaStreamDestroy(cs_);
    if (side_) cudaStreamDestroy(side_);
    if (sh_fork_) cudaEventDestroy(sh_fork_);
    if (sh_join_) cudaEventDestroy(sh_join_);
    if (owns_weights_ && dense_) {
        for (const auto& t : tensors_)   // the STRATA_MTP_Q8_SM60 registrations first (a no-op for the rest)
            if (t.kind == "q8_0") strata::kernels::native_q8_0_sm60_release(dense_ + t.off);
        cudaFree(dense_);
    }
    if (owns_weights_ && experts_) cudaFree(experts_);
    if (state_arena_) cudaFree(state_arena_);
    if (arena_) cudaFree(arena_);
    if (head_logits_) cudaFree(head_logits_);
    if (owns_draft_head_ && dhead_) {
        strata::kernels::native_q6_k_unpack(dhead_);
        strata::kernels::native_q8_0_sm60_release(dhead_);
        cudaFree(dhead_);
    }
    if (owns_draft_head_ && dvocab_) cudaFree(dvocab_);
    if (ev_chain_) cudaEventDestroy(ev_chain_);
    for (cudaEvent_t e : ev_step_) if (e) cudaEventDestroy(e);
    void* hosts[] = {h_tok_, h_step_, h_pos_, h_row_, h_out_, h_prob_, h_force_, h_floor_};
    for (void* h : hosts) if (h) cudaFreeHost(h);
}

const float* MtpDrafter::f32(const char* name) const {
    for (const auto& t : tensors_) if (t.name == name && t.kind == "f32") return (const float*) (dense_ + t.off);
    return nullptr;
}
const uint16_t* MtpDrafter::bf16(const char* name) const {
    for (const auto& t : tensors_) if (t.name == name && t.kind == "bf16") return (const uint16_t*) (dense_ + t.off);
    return nullptr;
}
const void* MtpDrafter::q8(const char* name) const {
    for (const auto& t : tensors_) if (t.name == name && t.kind == "q8_0") return dense_ + t.off;
    return nullptr;
}
const void* MtpDrafter::wq(const char* name, int& type) const {
    if (dense4_ != nullptr)
        for (const auto& e : q4_off_)
            if (e.first == name) { type = GGML_Q4_0; return dense4_ + e.second; }
    type = GGML_Q8_0;
    return q8(name);
}

// --mtp-q4: every Q8_0 tensor of dense.bin as Q4_0 (ggml's reference quantizer), in one device buffer
bool MtpDrafter::make_q4_dense(const std::vector<uint8_t>& blob, std::string& err) {
#ifdef STRATA_NATIVE_EXPERTS
    uint64_t total = 0;
    std::vector<std::pair<const Tensor*, uint64_t>> plan;
    for (const auto& t : tensors_) {
        if (t.kind != "q8_0" || t.cols % 32 != 0) continue;
        plan.push_back({&t, total});
        total += ((uint64_t) t.rows * (uint64_t) (t.cols / 32) * 18 + 255) & ~255ull;
    }
    std::vector<uint8_t> host((size_t) total);
    std::vector<float> row;
    const auto* q8t = ggml_get_type_traits(GGML_TYPE_Q8_0);
    for (const auto& [t, off] : plan) {
        row.resize((size_t) t->cols);
        const uint64_t in_row = (uint64_t) (t->cols / 32) * 34, out_row = (uint64_t) (t->cols / 32) * 18;
        for (int64_t r = 0; r < t->rows; ++r) {
            q8t->to_float(blob.data() + t->off + (uint64_t) r * in_row, row.data(), t->cols);
            ggml_quantize_chunk(GGML_TYPE_Q4_0, row.data(), host.data() + off + (uint64_t) r * out_row, 0, 1, t->cols,
                                nullptr);
        }
        q4_off_.push_back({t->name, off});
    }
    if (cudaMalloc((void**) &dense4_, (size_t) total) != cudaSuccess) { err = "mtp: the Q4_0 projections do not fit"; return false; }
    cudaMemcpy(dense4_, host.data(), (size_t) total, cudaMemcpyHostToDevice);
    vram_ += total;
    std::fprintf(stderr, "strata mtp: --mtp-q4: %zu projections as Q4_0 (%.1f MiB)\n", plan.size(), (double) total / 1048576.0);
    return true;
#else
    (void) blob;
    err = "mtp: --mtp-q4 needs a build with native experts (ggml)";
    return false;
#endif
}

// --mtp-q4: the draft head's subset rows as Q4_0 (dequantized from the main head's format)
bool MtpDrafter::make_q4_head(std::string& err) {
#ifdef STRATA_NATIVE_EXPERTS
    const int64_t N = g_->n_embd;
    const auto* tt = ggml_get_type_traits((ggml_type) head_->type());
    if (tt == nullptr || tt->to_float == nullptr || N % 32 != 0) { err = "mtp: --mtp-q4 cannot read the head's format"; return false; }
    const size_t in_row = head_->row_bytes(), out_row = (size_t) (N / 32) * 18;
    std::vector<uint8_t> src((size_t) n_dvocab_ * in_row), dst((size_t) n_dvocab_ * out_row);
    if (cudaMemcpy(src.data(), dhead_, src.size(), cudaMemcpyDeviceToHost) != cudaSuccess) { err = "mtp: reading the draft head"; return false; }
    std::vector<float> row((size_t) N);
    for (int64_t r = 0; r < n_dvocab_; ++r) {
        tt->to_float(src.data() + (size_t) r * in_row, row.data(), N);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, row.data(), dst.data() + (size_t) r * out_row, 0, 1, N, nullptr);
    }
    // the Q4_0 rows replace the subset in place (they are smaller)
    if (cudaMemcpy(dhead_, dst.data(), dst.size(), cudaMemcpyHostToDevice) != cudaSuccess) { err = "mtp: writing the draft head"; return false; }
    dhead_type_ = GGML_Q4_0;
    std::fprintf(stderr, "strata mtp: --mtp-q4: draft head over %lld tokens as Q4_0 (%.1f MiB read per step, was %.1f)\n",
                 (long long) n_dvocab_, (double) dst.size() / 1048576.0, (double) src.size() / 1048576.0);
    return true;
#else
    err = "mtp: --mtp-q4 needs a build with native experts (ggml)";
    return false;
#endif
}

bool MtpDrafter::load(const std::string& rt_dir, const ModelGeometry& g, SessionState& ss, int max_t, std::string& err,
                      int64_t window, const MtpDrafter* shared) {
#if defined(STRATA_GPU_METAL)
    (void) window; (void) shared;
    g_ = &g;
    ss_ = &ss;
    max_t_ = max_t;
    return strata::metal_engine::mtp_load(rt_dir, max_t, err);
#endif
    cudaGetDevice(&device_);   // a layer split's last stage on another GPU: the drafter lives there
    g_ = &g;
    ss_ = &ss;
    if (ple_ss_ == nullptr) ple_ss_ = &ss;
    max_t_ = max_t;
    rt_dir_ = rt_dir;
    // (the draft layer's kernels - the resident grouping, the grouped experts, the fused reads - are built for the narrow window)
    if (max_t < 1 || max_t > strata::kernels::kVerifyNarrowT) { err = "mtp: max_t out of range"; return false; }
    if (shared != nullptr) {
        if (shared->device_ != device_ || shared->g_ != &g || shared->dense_ == nullptr ||
            shared->experts_ == nullptr || shared->rt_dir_ != rt_dir) {
            err = "mtp: incompatible shared weights";
            return false;
        }
        dense_ = shared->dense_;
        experts_ = shared->experts_;
        tensors_ = shared->tensors_;
        owns_weights_ = false;
        owns_draft_head_ = false;
    }
    // Loader fix (0.1.15+loaderfix.2): the two reads below are the whole “drafter files” cost; reporting
    // them apart from the rest of the stage is what makes the next regression visible.
    const auto t_files = std::chrono::steady_clock::now();
    // ---- the index and the dense weights
    if (shared == nullptr) {
        std::ifstream idx(rt_dir + "/dense.txt");
        if (!idx) { err = "mtp: cannot open " + rt_dir + "/dense.txt (run tools/mtp_rt.py)"; return false; }
        std::string line;
        while (std::getline(idx, line)) {
            if (line.empty()) continue;
            std::istringstream is(line);
            Tensor t;
            is >> t.name >> t.kind >> t.rows >> t.cols >> t.off >> t.bytes;
            if (!is) { err = "mtp: malformed dense.txt line: " + line; return false; }
            tensors_.push_back(t);
        }
        std::vector<uint8_t> blob;
        if (!read_file(rt_dir + "/dense.bin", blob)) { err = "mtp: cannot read dense.bin"; return false; }
        const cudaError_t alloc = cudaMalloc((void**) &dense_, blob.size());
        if (alloc != cudaSuccess) {
            size_t free_bytes = 0, total_bytes = 0;
            const cudaError_t info = cudaMemGetInfo(&free_bytes, &total_bytes);
            err = "mtp: dense weights allocation failed (" + std::string(cudaGetErrorString(alloc)) +
                  "), requested " + std::to_string(blob.size() >> 20) + " MiB, CUDA0 free " +
                  (info == cudaSuccess ? std::to_string(free_bytes >> 20) + " MiB" : "unknown");
            return false;
        }
        cudaMemcpy(dense_, blob.data(), blob.size(), cudaMemcpyHostToDevice);
        vram_ += blob.size();
        if (q4_ && !make_q4_dense(blob, err)) return false;
    }
    // ---- the 512 routed experts, one blob each
    if (shared == nullptr) {
        const uint64_t bytes = (uint64_t) g.n_expert * strata::kernels::cpu::BLOB;
        // Loader fix (0.1.15+loaderfix.2): each 64 MiB read below reached the disk as ~16k 4095-byte reads under
        // MSVC's `basic_filebuf::xsgetn`, which is what made 675 MiB of drafter experts take minutes.
        FILE* f = std::fopen((rt_dir + "/experts.bin").c_str(), "rb");
        if (f == nullptr) { err = "mtp: cannot open experts.bin"; return false; }
        struct Closer {
            FILE* f;
            ~Closer() { if (f != nullptr) std::fclose(f); }
        } closer{f};
        if (cudaMalloc((void**) &experts_, bytes) != cudaSuccess) { err = "mtp: the 512 experts do not fit in VRAM"; return false; }
#if !defined(_WIN32)
        strata::platform::advise_willneed(fileno(f), 0, bytes);
#endif
        std::vector<uint8_t> chunk(64u << 20);
        for (uint64_t off = 0; off < bytes;) {
            const uint64_t n = std::min<uint64_t>(chunk.size(), bytes - off);
            if (std::fread(chunk.data(), 1, (size_t) n, f) != (size_t) n) { err = "mtp: experts.bin is truncated"; return false; }
            cudaMemcpy(experts_ + off, chunk.data(), n, cudaMemcpyHostToDevice);
            off += n;
        }
        vram_ += bytes;
    }
    const char* required[] = {"fc_embedding.weight", "fc_hidden.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                              "self_attn.v_proj.weight", "self_attn.o_proj.weight", "mlp.shared_expert.gate_proj.weight",
                              "mlp.shared_expert.up_proj.weight", "mlp.shared_expert.down_proj.weight"};
    for (const char* n : required) if (!q8(n)) { err = std::string("mtp: ") + n + " is missing (q8_0)"; return false; }
    // STRATA_MTP_Q8_SM60=1: the projections on the Pascal GEMV (--batch's shared drafters read the same pointers), as
    // packed copies like the main model's: the prompt path's draft_kv reads four of them in the GGUF layout (MMQ);
    // --mtp-q4 reads its Q4_0 copies instead
    if (shared == nullptr && dense4_ == nullptr && mtp_q8_sm60()) {
        int packed = 0;
        uint64_t packed_bytes = 0;
        for (const char* n : required)
            for (const auto& t : tensors_)
                if (t.name == n && t.kind == "q8_0" &&
                    strata::kernels::native_q8_0_sm60_pack_quiet(dense_ + t.off, (int) t.cols, (int) t.rows,
                                                                 ("MTP " + t.name).c_str())) {
                    ++packed;
                    packed_bytes += t.bytes;   // a copy is the same 34 bytes per 32 values
                }
        vram_ += packed_bytes;
        if (packed > 0)
            std::fprintf(stderr, "strata mtp: %d of %zu projections on the Pascal Q8_0 GEMV (STRATA_MTP_Q8_SM60=1, %.1f "
                                 "MiB more VRAM)\n", packed, sizeof(required) / sizeof(required[0]),
                         (double) packed_bytes / 1048576.0);
    }

    // ---- the layer's own K/V (dense attention: no indexer state is read)
    const strata::kernels::QsaShapes s = shapes_of(g);
    const int64_t max_cells = ss.qsa_states[ss.qsa_primary()].max_cells;
    // KV streaming: the drafter only reads its last `window` cells, so with streaming on its K/V is a ring of the
    // window (plus the cells a round writes ahead of its queries) over a host copy, refilled on a resume. The host copy
    // is pinned after the expert arena has pinned what it could: if it does not fit, the K/V stays whole in VRAM.
    int64_t ring = (window > 0 && window < max_cells) ? window + 4 * (int64_t) max_t + 64 : 0;
    // K8V4 never applies to the drafter: its own attention paths (below, and verify.cpp) handle whole formats
    // only, whatever ring shape it takes (0, a window, or the -1 fully-resident fallback).
    const bool kv_hybrid_was = qsa_kv_hybrid();
    const bool kv_int8_was = qsa_kv_int8();
    qsa_set_kv_hybrid(false);
    if (kv_hybrid_was) qsa_set_kv_int8(true);   // the drafter under --kv k8v4: plain INT8
    uint64_t sb = qsa_state_bytes(g, max_cells, false, ring);
    if (cudaMalloc(&state_arena_, sb) != cudaSuccess) { err = "mtp: the K/V state does not fit"; return false; }
    if (qsa_state_init(g, max_cells, state_arena_, st_, &ss.qsa_states[ss.qsa_primary()], ring) == 0) {
        if (st_.kv_mode == 0) { err = "mtp: state init failed"; return false; }
        std::fprintf(stderr, "strata mtp: no pinned RAM left for the draft layer's K/V copy; keeping it in VRAM\n");
        cudaGetLastError();
        cudaFree(state_arena_);
        st_ = QsaState{};
        ring = -1;   // fully resident
        sb = qsa_state_bytes(g, max_cells, false, ring);
        if (cudaMalloc(&state_arena_, sb) != cudaSuccess) { err = "mtp: the K/V state does not fit"; return false; }
        if (qsa_state_init(g, max_cells, state_arena_, st_, &ss.qsa_states[ss.qsa_primary()], ring) == 0) { err = "mtp: state init failed"; return false; }
    }
    qsa_set_kv_int8(kv_int8_was);
    qsa_set_kv_hybrid(kv_hybrid_was);
    qsa_state_zero(st_, g, nullptr);
    cudaDeviceSynchronize();
    vram_ += sb;

    // ---- buffers
    window_ = (window > 0 && window < max_cells) ? window : 0;
    if (floor_on_ && !mapped(64, (void**) &h_floor_, (void**) &m_floor_)) { err = "mtp: the window floor does not fit"; return false; }
    cap_ = (((window_ > 0 ? window_ : max_cells) + 63) / 64) * 64;
    attn_scratch_floats_ = (int64_t) strata::kernels::qsa_decode_attn_scratch_floats(cap_, s);
    const uint64_t T = (uint64_t) max_t, N = (uint64_t) g.n_embd, HC = (uint64_t) g.hc, K = (uint64_t) ss.k;
    const uint64_t NH = (uint64_t) g.n_head, HD = (uint64_t) g.head_dim, NKV = (uint64_t) g.n_head_kv;
    const uint64_t R2 = 2 * T;   // step/pos rows: T catch-up rows + up to T-2 chain steps
    bool ok = mapped(T * 4 + 64, (void**) &h_tok_, (void**) &m_tok_) &&
              mapped(R2 * 4 * 4 + 64, (void**) &h_step_, (void**) &m_step_) &&
              mapped(R2 * NH * 4 + 64, (void**) &h_pos_, (void**) &m_pos_) &&
              mapped(64, (void**) &h_row_, (void**) &m_row_) &&
              mapped(T * 4 + 64, (void**) &h_out_, (void**) &m_out_) &&
              mapped(T * 4 + 64, (void**) &h_prob_, (void**) &m_prob_) &&
              (!force_on_ || mapped(64, (void**) &h_force_, (void**) &m_force_));
    if (!ok) { err = "mtp: mapped staging failed"; return false; }
    if (h_force_ != nullptr) for (int j = 0; j < 16; ++j) h_force_[j] = -1;
    auto carve = [&](Bump& b) {
        tok_ = b.take<int32_t>(T); step_ = b.take<int32_t>(R2 * 4); pos_ = b.take<int32_t>(R2 * NH); row_ = b.take<int32_t>(4);
        ident_ = b.take<int32_t>(T * (uint64_t) cap_);
        Rin_ = b.take<float>(T * HC * N); R_ = b.take<float>(T * HC * N);
        emb_ = b.take<float>(T * N); en_ = b.take<float>(T * N); e2_ = b.take<float>(T * N);
        hn_ = b.take<float>(T * HC * N); h2_ = b.take<float>(T * HC * N);
        mixed_ = b.take<float>(T * N); inj_ = b.take<float>(T * HC); inj2_ = b.take<float>(T * HC);
        lo_ = b.take<float>(T * (uint64_t) g.hc_lr); rs_ = b.take<float>(T * HC); bo_ = b.take<float>(T * N);
        xn_ = b.take<float>(T * HC * N);
        xq_ = b.take<uint8_t>(strata::kernels::native_q8_1_bytes((int) (NH * HD), 8));
        qfull_ = b.take<float>(T * NH * 2 * HD); qcur_ = b.take<float>(T * NH * HD);
        kcur_ = b.take<float>(T * NKV * HD); vcur_ = b.take<float>(T * NKV * HD);
        attn_ = b.take<float>(T * NH * HD); attn32_ = b.take<float>(T * NH * HD);
        attn_scratch_ = b.take<float>((uint64_t) attn_scratch_floats_);   // the full layer runs one row at a time
        logits_ = b.take<float>(T * (uint64_t) g.n_expert); w_ = b.take<float>(T * K); ids_ = b.take<int32_t>(T * K);
        shared_ = b.take<float>(T * N); parts_ = b.take<float>(T * K * N); y_ = b.take<float>(T * N);
        sample_ = b.take<float>(T * N);
        {
            const strata::kernels::GrShapes gsh{g.n_embd, g.hc, g.hc_lr};
            uint8_t* gr_mem = b.take<uint8_t>(strata::kernels::gr_workspace_bytes(gsh));
            if (gr_mem != nullptr) strata::kernels::gr_workspace_init(gsh, gr_mem, own_gr_);
        }
        hit_slot_ = b.take<int32_t>(T * K); hit_dst_ = b.take<int32_t>(T * K); hit_count_ = b.take<int32_t>(4);
        grp_ptr_ = b.take<unsigned long long>(T * K); grp_start_ = b.take<int32_t>(T * K + 1);
        grp_counts_ = b.take<int32_t>(4);
        hit_xq_ = b.take<uint8_t>(T * (N / 32) * 34); hit_xs_ = b.take<float>(T * (N / 32));
        hit_scratch_ = b.take<uint8_t>(strata::kernels::moe_hit_grouped_scratch_bytes((int64_t) (T * K), g.n_embd, g.n_ff));
        sh_scratch_ = (float*) b.take<uint8_t>(strata::kernels::shared_expert_scratch_bytes(g.n_ff));
        x_bf16_ = b.take<uint16_t>(N);
        out_ids_ = b.take<int32_t>(T + 4);
        probs_ = b.take<float>(T + 4);
        arg_scratch_ = b.take<uint8_t>(strata::kernels::argmax_rows_scratch_bytes((int) T));
        top_scratch_ = b.take<uint8_t>(strata::kernels::row_top_prob_scratch_bytes((int) T));
        dummy_inj_ = b.take<float>(HC);
    };
    Bump count;
    carve(count);
    if (cudaMalloc(&arena_, count.used) != cudaSuccess) { err = "mtp: buffers do not fit"; return false; }
    cudaMemset(arena_, 0, count.used);
    Bump real;
    real.base = (uint8_t*) arena_;
    carve(real);
    vram_ += count.used;
    {
        std::vector<int32_t> id((size_t) (T * (uint64_t) cap_));
        for (uint64_t t = 0; t < T; ++t)
            for (int64_t i = 0; i < cap_; ++i) id[(size_t) (t * (uint64_t) cap_ + (uint64_t) i)] = (int32_t) i;
        cudaMemcpy(ident_, id.data(), id.size() * 4, cudaMemcpyHostToDevice);
    }
    // --pipeline-windows 2 (the forcing graphs): the chain shares its card with stage 1's windows and the next launch
    // on stage 0 waits for it, so its stream gets the highest priority there (STRATA_MTP_PRIORITY=0: the default)
    bool prio = false;
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)   // (HIP, Metal: the default priority)
    static const bool hi_prio = [] { const char* v = std::getenv("STRATA_MTP_PRIORITY"); return v == nullptr || std::atoi(v) != 0; }();
    int prio_lo = 0, prio_hi = 0;
    if (force_on_ && hi_prio && cudaDeviceGetStreamPriorityRange(&prio_lo, &prio_hi) == cudaSuccess)
        prio = cudaStreamCreateWithPriority(&cs_, cudaStreamNonBlocking, prio_hi) == cudaSuccess;
    cudaGetLastError();
#endif
#if defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
    // engine-latency, STRATA_MTP_HIP_PRIO (default on gfx1200/gfx1201, =1 elsewhere opt-in, =0 off): the draft stream at the
    // highest priority.  ROCm hands out hardware queues per priority (GPU_MAX_HW_QUEUES per level), so the drafter gets a
    // queue of its own: its round no longer waits behind the commit graph that the verifier's stream launched just before
    // it (the commit is only needed by the next window, and now runs beside the draft).  R9700, UD-Q4_K_XL: draft wall
    // 5.8 -> 3.5 ms a window (notes-latency.md).
    {
        static const bool hip_prio = [] {
            if (const char* v = std::getenv("STRATA_MTP_HIP_PRIO"); v != nullptr && *v != 0) return std::atoi(v) != 0;
            int dev = 0;
            cudaDeviceProp prop{};
            return cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&prop, dev) == cudaSuccess &&
                   strata::kernels::gfx_arch_is_gfx12(prop.gcnArchName);
        }();
        int lo = 0, hi = 0;
        if (hip_prio && hipDeviceGetStreamPriorityRange(&lo, &hi) == hipSuccess)
            prio = hipStreamCreateWithPriority(&cs_, hipStreamNonBlocking, hi) == hipSuccess;
        (void) cudaGetLastError();
    }
#endif
    if (!prio && cudaStreamCreateWithFlags(&cs_, cudaStreamNonBlocking) != cudaSuccess) { err = "mtp: stream"; return false; }
#if !defined(STRATA_USE_HIP)
    if (cudaStreamCreateWithFlags(&side_, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&sh_fork_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&sh_join_, cudaEventDisableTiming) != cudaSuccess) {
        err = "mtp: streams";
        return false;
    }
#else
    // HIP keeps the shared expert on cs_, so nothing uses this branch stream, and #1240 stopped creating it and its two
    // events (an R9700 lost prompt speed with them after the first request).  On gfx103x the stream's existence alone
    // makes prompts read faster: RX 6800, Windows, STRATA_HIP_PROMPT_F16=1, a 7.9K-token prompt at 798-808 tok/s
    // with it and 584-596 without, all of it in the prompt path's "wait copy" phase (3,081 -> 75 ms); the two events
    // change nothing.  So gfx103x creates the stream and no other HIP card does.  STRATA_MTP_SIDE_STREAM=1 / =0 forces
    // either.
    {
        int dev = 0;
        cudaDeviceProp prop{};
        bool on = cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&prop, dev) == cudaSuccess &&
                  std::strncmp(prop.gcnArchName, "gfx103", 6) == 0;
        if (const char* v = std::getenv("STRATA_MTP_SIDE_STREAM"); v != nullptr && *v != 0) on = std::atoi(v) != 0;
        if (on && cudaStreamCreateWithFlags(&side_, cudaStreamNonBlocking) != cudaSuccess) {
            err = "mtp: streams";
            return false;
        }
    }
#endif
    const double files_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_files).count();
    if (shared != nullptr) {
        std::fprintf(stderr, "strata mtp: shared draft weights, %.0f MiB of private state and buffers\n",
                     (double) vram_ / 1048576.0);
    } else {
        std::fprintf(stderr, "strata mtp: draft layer loaded, %.0f MiB of VRAM (experts %.0f, dense %.0f), files read in %.2f s (%.0f MiB/s)\n",
                     (double) vram_ / 1048576.0, (double) g.n_expert * strata::kernels::cpu::BLOB / 1048576.0,
                     (double) tensors_.back().off / 1048576.0, files_s,
                     files_s > 0 ? ((double) g.n_expert * strata::kernels::cpu::BLOB + (double) tensors_.back().off) /
                                       1048576.0 / files_s : 0.0);
    }
    return true;
}

namespace {
// STRATA_MTP_FULL_HEAD=1 (a diagnostic): draft over the whole vocabulary (the main head itself) instead of
// rt/draft_vocab.bin's subset - the reference head, to measure what the subset costs in acceptance
bool full_head_env() {
    static const bool on = [] { const char* v = std::getenv("STRATA_MTP_FULL_HEAD"); return v && v[0] == '1'; }();
    return on;
}
}  // namespace

bool MtpDrafter::top2_env() {
    static const bool on = [] { const char* v = std::getenv("STRATA_MTP_TOP2"); return v && v[0] == '1'; }();
    return on;
}

// STRATA_MTP_TOP2=1 (a diagnostic for tree drafts): draft j's runner-up under the draft layer, from the head logits
// the step just wrote (row 0), on the host.  Slow; only for measuring what a second branch would have caught.
void MtpDrafter::record_top2(int j) {
    const int64_t nv = dhead_ != nullptr ? n_dvocab_ : n_vocab_;
    if ((int64_t) lg_host_.size() < nv) lg_host_.resize((size_t) nv);
    if ((int) top2_.size() < max_t_) top2_.assign((size_t) max_t_, -1);
    if (j < 0 || j >= max_t_) return;
    if (cudaMemcpy(lg_host_.data(), head_logits_, (size_t) nv * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        top2_[(size_t) j] = -1;
        return;
    }
    int64_t b1 = -1, b2 = -1;
    for (int64_t i = 0; i < nv; ++i) {
        const float v = lg_host_[(size_t) i];
        if (b1 < 0 || v > lg_host_[(size_t) b1]) { b2 = b1; b1 = i; }
        else if (b2 < 0 || v > lg_host_[(size_t) b2]) b2 = i;
    }
    top2_[(size_t) j] = b2 < 0 ? -1 : dhead_ != nullptr ? dvocab_host_[(size_t) b2] : (int32_t) b2;
}

uint64_t MtpDrafter::bind_bytes(uint64_t head_row_bytes, int64_t n_vocab) const {
    uint64_t bytes = head_logits_ ? 0 : (uint64_t) max_t_ * (uint64_t) n_vocab * sizeof(float);
    if (dhead_ == nullptr && owns_draft_head_ && !full_head_env()) {
        if (FILE* f = std::fopen(vocab_file().c_str(), "rb")) {
            std::fseek(f, 0, SEEK_END);
            const long size = std::ftell(f);
            std::fclose(f);
            if (size >= 4 && size % 4 == 0) bytes += (uint64_t) (size / 4) * head_row_bytes + (uint64_t) size;
        }
    }
    // coupled draft sampling (STRATA_SPEC_COUPLED=1 only): an upper bound - the id -> subset map, the penalty ring,
    // the split scratch (64 lists of 64 entries at most) and the parameters
    if (coupled_draft_env() && cparams_ == nullptr)
        bytes += (uint64_t) n_vocab * 4 + (uint64_t) (kCoupledHistCap + max_t_) * 4 + 64ull * 64ull * 8ull + 4096;
    return bytes;
}

bool MtpDrafter::setup_coupled(std::string& err) {
    const int64_t nv = dhead_ != nullptr ? n_dvocab_ : n_vocab_;
    const size_t scratch = strata::kernels::coupled_draft_scratch_bytes((int) nv);
    if (scratch == 0) {
        std::fprintf(stderr, "strata mtp: STRATA_SPEC_COUPLED: %lld draft logits are too wide for the coupled sampler; "
                             "argmax drafts\n", (long long) nv);
        return true;
    }
    const size_t ring = (size_t) (kCoupledHistCap + max_t_) * sizeof(int32_t);
    if (cudaMalloc((void**) &cparams_, sizeof(strata::kernels::SamplerParams)) != cudaSuccess ||
        cudaMalloc((void**) &cring_, ring) != cudaSuccess || cudaMalloc(&cscratch_, scratch) != cudaSuccess ||
        !mapped(sizeof(strata::kernels::SamplerParams), (void**) &h_cparams_, (void**) &m_cparams_) ||
        !mapped((size_t) kCoupledHistCap * sizeof(int32_t), (void**) &h_chist_, (void**) &m_chist_)) {
        err = "mtp: the coupled draft sampler's buffers do not fit";
        return false;
    }
    if (spec_prob_env() &&
        !mapped((size_t) max_t_ * (size_t) kSpecQStride * sizeof(int32_t), (void**) &h_q_, (void**) &m_q_)) {
        err = "mtp: the draft distributions' buffer does not fit";
        return false;
    }
    cudaMemset(cring_, 0xff, ring);   // -1: no token
    std::fill(h_chist_, h_chist_ + kCoupledHistCap, -1);
    vram_ += sizeof(strata::kernels::SamplerParams) + ring + scratch;
    if (dhead_ != nullptr) {   // token id -> subset index (-1: not in the draft head), for the penalties
        std::vector<int32_t> sub((size_t) n_dvocab_), inv((size_t) n_vocab_, -1);
        cudaMemcpy(sub.data(), dvocab_, sub.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < sub.size(); ++i)
            if (sub[i] >= 0 && sub[i] < n_vocab_ && inv[(size_t) sub[i]] < 0) inv[(size_t) sub[i]] = (int32_t) i;
        if (cudaMalloc((void**) &dinv_, inv.size() * sizeof(int32_t)) != cudaSuccess) {
            err = "mtp: the coupled draft sampler's token map does not fit";
            return false;
        }
        cudaMemcpy(dinv_, inv.data(), inv.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
        vram_ += inv.size() * sizeof(int32_t);
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        err = std::string("mtp: coupled setup: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    coupled_ok_ = true;
    if (spec_prob_env())
        std::fprintf(stderr, "strata mtp: probabilistic draft acceptance on (STRATA_SPEC_PROB): sampled requests draft by "
                             "sampling the draft head's distribution over %lld tokens; %s\n", (long long) nv,
                     spec_gate_pick()
                         ? "pick gate: masked mu offers, min(1, p/mu) acceptance and (p-mu)+ rejection/NULL residuals"
                         : "top gate (compatibility): q offers, min(1, p/q) acceptance and (p-q)+ rejection residuals");
    else
        std::fprintf(stderr, "strata mtp: coupled draft sampling on (STRATA_SPEC_COUPLED): sampled requests draft with the "
                             "target's chain and Philox draw over %lld tokens\n", (long long) nv);
    return true;
}

void MtpDrafter::set_draft_sampling(const strata::kernels::SamplerParams& sp) {
    coupled_active_ = coupled_ok_ && !sp.greedy && sp.temperature > 0.0f && sp.temperature >= spec_min_temp();
    if (!coupled_active_) return;
    *h_cparams_ = sp;   // read by the next round graph (after the previous one has synced)
    std::fill(h_chist_, h_chist_ + kCoupledHistCap, -1);
}

void MtpDrafter::set_draft_history(const int32_t* tail, int64_t n_tail, int32_t next) {
    if (!coupled_active_) return;
    const int h = coupled_hist_len(h_cparams_->penalty_last_n, kCoupledHistCap);
    coupled_hist_base(tail, n_tail, next, h, h_chist_ + (kCoupledHistCap - h));
}

namespace {
// #474: what to change when the draft head's token subset does not fit the VRAM left - a smaller subset that setup
// ships (data/draft_vocab_*.bin; their token counts below), and how to pick it.  The start used to stop at "the draft
// head does not fit" with no hint.  Text only, after the failure: nothing changes for a start that fits.
void draft_head_hint(int64_t n_tokens, int64_t row_bytes) {
    size_t free_b = 0, total_b = 0;
    const bool have_free = cudaMemGetInfo(&free_b, &total_b) == cudaSuccess;
    auto mib = [&](int64_t n) { return (double) (n * row_bytes) / 1048576.0; };
    std::string free_s;
    if (have_free) {
        char b[64];
        std::snprintf(b, sizeof(b), " and %.0f MiB is free", (double) free_b / 1048576.0);
        free_s = b;
    }
    std::fprintf(stderr, "strata mtp: the draft head over %lld tokens needs %.0f MiB of VRAM%s.\n",
                 (long long) n_tokens, mib(n_tokens), free_s.c_str());
    struct Subset { const char* name; int64_t tokens; const char* what; };
    static const Subset smaller[] = {{"cyrillic", 58963, "English, code and the Cyrillic script"},
                                     {"en", 40525, "English and code"}};
    std::string opts;
    for (const Subset& s : smaller)
        if (s.tokens < n_tokens) {
            char b[160];
            std::snprintf(b, sizeof(b), "%s--draft-vocab %s (%s, ~%.0f MiB)", opts.empty() ? "" : " or ", s.name,
                          s.what, mib(s.tokens));
            opts += b;
        }
    if (!opts.empty())
        std::fprintf(stderr, "strata mtp: hint: a smaller draft vocabulary needs less VRAM: %s. Start once with it - "
                             "START-HERE.bat --draft-vocab en (Windows) or ./setup.sh --draft-vocab en - and the model "
                             "keeps it (\"draft_vocab\" in its strata-*.json config); or a smaller --context in "
                             "setup.\n",
                     opts.c_str());
    else
        std::fprintf(stderr, "strata mtp: hint: this is already the smallest shipped draft vocabulary: a smaller "
                             "--context (or closing what else uses the GPU) leaves it room.\n");
}
}  // namespace

bool MtpDrafter::bind(const WeightTable& wt, const NativeHead* head, const float* window_R, std::string& err,
                      const MtpDrafter* shared) {
#if defined(STRATA_GPU_METAL)
    (void) wt; (void) head; (void) window_R; (void) err; (void) shared;
    return true;
#endif
    const OnDevice on_device(device_);
    wt_ = &wt;
    head_ = head;
    window_R_ = window_R;
    const WeightRef* wo = wt.find("output.weight");
    if (!wo) { err = "mtp: output.weight is missing"; return false; }
    n_vocab_ = wo->ne1;
    if (head == nullptr || !head->loaded()) { err = "mtp: the draft layer needs the native head (--native)"; return false; }
    if (head_logits_ == nullptr &&
        cudaMalloc((void**) &head_logits_, (size_t) max_t_ * (size_t) n_vocab_ * sizeof(float)) != cudaSuccess) {
        err = "mtp: the draft logits do not fit";
        return false;
    }
    if (shared != nullptr) {
        if (shared->head_ != head || shared->device_ != device_ ||
            shared->n_vocab_ != n_vocab_) {
            err = "mtp: incompatible shared draft head";
            return false;
        }
        dhead_ = shared->dhead_;
        dhead_type_ = shared->dhead_type_;   // the subset's type (the main head's, or Q4_0): -1 failed every --batch-mtp step
        dvocab_ = shared->dvocab_;
        n_dvocab_ = shared->n_dvocab_;
        owns_draft_head_ = false;
    }
    // the draft head's token subset, when tools/draft_vocab.py wrote one
    if (dhead_ == nullptr && shared == nullptr && !full_head_env()) {
        std::vector<uint8_t> raw;
        if (read_file(vocab_file(), raw) && raw.size() >= 4 && raw.size() % 4 == 0) {
            n_dvocab_ = (int64_t) (raw.size() / 4);
            const int64_t row_bytes = (int64_t) head->row_bytes();   // a vocabulary row of the native head
            if (cudaMalloc((void**) &dvocab_, raw.size()) != cudaSuccess ||
                cudaMalloc((void**) &dhead_, (size_t) (n_dvocab_ * row_bytes)) != cudaSuccess) {
                err = "mtp: the draft head does not fit";
                draft_head_hint(n_dvocab_, row_bytes);
                return false;
            }
            cudaMemcpy(dvocab_, raw.data(), raw.size(), cudaMemcpyHostToDevice);
            dvocab_host_.resize((size_t) n_dvocab_);
            std::memcpy(dvocab_host_.data(), raw.data(), raw.size());
            const int head_dev = head->device();
            if (head_dev >= 0 && head_dev != device_) {
                // the main head is on another card (the drafter on the first stage of a layer split, 0.1.44): the rows are
                // gathered where the head is and the subset copied over once
                void* tmp = nullptr;
                int32_t* dv = nullptr;
                bool ok_x = true;
                {
                    const OnDevice on_head(head_dev);
                    ok_x = cudaMalloc((void**) &tmp, (size_t) (n_dvocab_ * row_bytes)) == cudaSuccess &&
                           cudaMalloc((void**) &dv, raw.size()) == cudaSuccess &&
                           cudaMemcpy(dv, raw.data(), raw.size(), cudaMemcpyHostToDevice) == cudaSuccess;
                    if (ok_x) {
                        strata::kernels::gather_rows((const uint8_t*) head->weights(), row_bytes, dv, n_dvocab_,
                                                     (uint8_t*) tmp, nullptr);
                        ok_x = cudaDeviceSynchronize() == cudaSuccess;
                    }
                }
                ok_x = ok_x && cudaMemcpy(dhead_, tmp, (size_t) (n_dvocab_ * row_bytes), cudaMemcpyDeviceToDevice) == cudaSuccess;
                {
                    const OnDevice on_head(head_dev);
                    if (tmp) cudaFree(tmp);
                    if (dv) cudaFree(dv);
                }
                if (!ok_x) { err = "mtp: the draft head's rows could not be copied from the head's card"; return false; }
            } else {
                strata::kernels::gather_rows((const uint8_t*) head->weights(), row_bytes, dvocab_, n_dvocab_, dhead_, nullptr);
            }
            cudaDeviceSynchronize();
            vram_ += (uint64_t) (n_dvocab_ * row_bytes) + raw.size();
            std::fprintf(stderr, "strata mtp: draft head over %lld tokens (%.1f MiB)\n", (long long) n_dvocab_,
                         (double) (n_dvocab_ * row_bytes) / 1048576.0);
            dhead_type_ = head->type();
            if (q4_head_ && !make_q4_head(err)) return false;
            if (dhead_type_ == 14 && strata::kernels::native_q6_k_packed_enabled())   // STRATA_Q6_PACKED=1
                strata::kernels::native_q6_k_pack(dhead_, (int) (row_bytes / 210) * 256, (int) n_dvocab_,
                                                  "MTP draft head");
            // STRATA_MTP_Q8_SM60=1: the subset on the Pascal GEMV like the main head, but packed in place - nothing but
            // record_rest's native_mmvq reads it (the prompt path never does), and a copy would cost its size again
            if (dhead_type_ == GGML_Q8_0 && owns_draft_head_ && mtp_q8_sm60() && row_bytes == g_->n_embd / 32 * 34 &&
                strata::kernels::native_q8_0_sm60_pack_in_place(dhead_, (int) g_->n_embd, (int) n_dvocab_,
                                                                "MTP draft head"))
                std::fprintf(stderr, "strata mtp: the draft head on the Pascal Q8_0 GEMV (STRATA_MTP_Q8_SM60=1)\n");
        }
    }
    if (coupled_draft_env() && cparams_ == nullptr && !setup_coupled(err)) return false;
    return true;
}

// STRATA_MTP_PROFILE=1 (measurement only): GPU stamps between the draft layer's stages in the round graph and in every
// step graph (each stamp is one more node, so the totals run a little high); the per-stage averages are printed every
// 300 rounds with STRATA_DRAFT_TIMING's line.  Off: no stamp is recorded and nothing changes.
namespace {
constexpr int kMpSlots = 24;
constexpr int kMpGraphs = 17;   // the round graph (0) and the steps 1..16
const char* const kMpNames[kMpSlots] = {"-", "embed+fc_emb", "hnorm+fc_hidden+add", "hc-read attn", "k/v+append",
                                        "q+norm-rope", "attention", "gate+o_proj", "hc-read mlp", "router",
                                        "experts", "shared", "combine+write", "hc-read mixer", "head gemv",
                                        "argmax+prob", "select", "", "", "", "", "", "", ""};
struct MtpProf {
    bool on = false;
    unsigned long long* d = nullptr;
    std::vector<unsigned long long> h;
    int base = -1;   // the graph being captured (-1: none, no stamps)
    double sum[2][kMpSlots] = {};
    int64_t n[2] = {0, 0};
    MtpProf() {
        const char* e = std::getenv("STRATA_MTP_PROFILE");
        on = e != nullptr && e[0] == '1';
    }
    void init() {
        if (!on || d != nullptr) return;
        if (cudaMalloc((void**) &d, (size_t) kMpSlots * kMpGraphs * 8) != cudaSuccess) { on = false; d = nullptr; (void) cudaGetLastError(); return; }
        cudaMemset(d, 0, (size_t) kMpSlots * kMpGraphs * 8);
        h.assign((size_t) kMpSlots * kMpGraphs, 0);
    }
    void stamp(int i, cudaStream_t cs) const {
        if (on && d != nullptr && base >= 0) strata::kernels::gpu_stamp(d, base * kMpSlots + i, cs);
    }
    void collect(int n_steps_run) {   // after the round's sync: the round graph and steps 1..n_steps_run-1
        if (!on || d == nullptr) return;
        cudaMemcpy(h.data(), d, h.size() * 8, cudaMemcpyDeviceToHost);
        for (int gi = 0; gi < n_steps_run && gi < kMpGraphs; ++gi) {
            const unsigned long long* s = h.data() + (size_t) gi * kMpSlots;
            const int k = gi == 0 ? 0 : 1;
            unsigned long long prev = s[0];
            if (prev == 0) continue;
            for (int i = 1; i < kMpSlots; ++i) {
                if (s[i] == 0 || s[i] < prev) continue;
                sum[k][i] += (double) (s[i] - prev);
                prev = s[i];
            }
            ++n[k];
        }
    }
    void report() {
        for (int k = 0; k < 2; ++k) {
            if (n[k] == 0) continue;
            std::string out;
            char b[96];
            double tot = 0;
            for (int i = 1; i < kMpSlots; ++i) {
                if (sum[k][i] <= 0) continue;
                tot += sum[k][i];
                std::snprintf(b, sizeof b, " %s %.3f", kMpNames[i], sum[k][i] / 1e6 / (double) n[k]);
                out += b;
            }
            std::fprintf(stderr, "strata mtp profile (%s, ms per graph over %lld):%s | total %.3f\n", k == 0 ? "round" : "step",
                         (long long) n[k], out.c_str(), tot / 1e6 / (double) n[k]);
            for (double& x : sum[k]) x = 0;
            n[k] = 0;
        }
    }
};
MtpProf g_mp;

// STRATA_VERIFY_NODES=1 also lists the draft graphs' nodes (kernel names on HIP; counts on CUDA)
void mtp_list_nodes(cudaGraph_t graph, const char* what) {
    if (std::getenv("STRATA_VERIFY_NODES") == nullptr || graph == nullptr) return;
#if defined(STRATA_GPU_METAL)
    (void) what;    // Metal has no graph nodes to list
#else
#if defined(STRATA_USE_HIP)
    using NodeT = hipGraphNode_t;
    using NodeTypeT = hipGraphNodeType;
#define MTP_GET_NODES hipGraphGetNodes
#define MTP_NODE_TYPE hipGraphNodeGetType
#else
    using NodeT = cudaGraphNode_t;
    using NodeTypeT = cudaGraphNodeType;
#define MTP_GET_NODES cudaGraphGetNodes
#define MTP_NODE_TYPE cudaGraphNodeGetType
#endif
    size_t nn = 0;
    MTP_GET_NODES(graph, nullptr, &nn);
    std::vector<NodeT> nodes(nn);
    MTP_GET_NODES(graph, nodes.data(), &nn);
    std::vector<std::pair<std::string, int>> kinds;
    for (NodeT nd : nodes) {
        NodeTypeT ty;
        MTP_NODE_TYPE(nd, &ty);
        std::string name = "type" + std::to_string((int) ty);
#if defined(STRATA_USE_HIP)
        if (ty == hipGraphNodeTypeKernel) {
            hipKernelNodeParams kp{};
            if (hipGraphKernelNodeGetParams(nd, &kp) == hipSuccess && kp.func != nullptr) {
                const char* fn = hipKernelNameRefByPtr(kp.func, nullptr);
                if (fn != nullptr) name = fn;
            }
        }
#else
        if (ty == cudaGraphNodeTypeKernel) name = "kernel";
#endif
        bool found = false;
        for (auto& k : kinds)
            if (k.first == name) { ++k.second; found = true; break; }
        if (!found) kinds.push_back({name, 1});
    }
#undef MTP_GET_NODES
#undef MTP_NODE_TYPE
    std::fprintf(stderr, "strata mtp: the %s graph has %zu nodes:", what, nn);
    for (auto& k : kinds) std::fprintf(stderr, "\n %4d x %.110s", k.second, k.first.c_str());
    std::fprintf(stderr, "\n");
#endif
}
}  // namespace

// The layer for T rows.  full = false stops after the K/V append (the prompt only needs the cache).
bool MtpDrafter::record_forward(int T, int step_row0, cudaStream_t cs, std::string& err) {
    // step_row0 >= 0: the full layer on step rows [step_row0, +T); step_row0 < 0: K/V only on rows [-1 - step_row0, +T)
    const bool full = step_row0 >= 0;
    if (full && T != 1) { err = "mtp: the full layer runs one row at a time (its attention scratch is sized for one)"; return false; }
    if (!record_front(T, full ? step_row0 : -1 - step_row0, cs, err)) return false;
    return !full || record_rest(step_row0, cs, err);
}

bool MtpDrafter::record_front(int T, int row0, cudaStream_t cs, std::string& err) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    const int64_t N = g.n_embd, HC = g.hc, NH = g.n_head, HD = g.head_dim, NKV = g.n_head_kv;
    const QsaShapes s = shapes_of(g);
    const int32_t* step = step_ + row0 * 4;
    const int32_t* pos = pos_ + row0 * NH;
    // --mtp-q4: each projection's format and weights for this pass (Q8_0 unless the Q4_0 copies exist)
    int wt_fc_embedding, wt_fc_hidden, wt_k_proj, wt_v_proj;
    const void* wp_fc_embedding = wq("fc_embedding.weight", wt_fc_embedding);
    const void* wp_fc_hidden = wq("fc_hidden.weight", wt_fc_hidden);
    const void* wp_k_proj = wq("self_attn.k_proj.weight", wt_k_proj);
    const void* wp_v_proj = wq("self_attn.v_proj.weight", wt_v_proj);
    try {
        // ---- the two input branches
        const WeightRef* we = wt_->find("token_embd.weight");
        if (!we) { err = "mtp: token_embd.weight is missing"; return false; }
        g_mp.stamp(0, cs);
        if (const NativeEmbed* ne = native_embed()) {   // plan v0.3 P6: the GGUF-form table
            ne->gather_dev(tok_, T, emb_, cs);
        } else {
            const auto* codes = (const uint8_t*) we->data;
            const auto* scales = (const float*) (codes + we->codes_bytes);
            const auto* offsets = we->has_offset ? (const float*) (codes + we->codes_bytes + we->scales_bytes) : nullptr;
            embedding_gather_dev(codes, scales, offsets, tok_, T, we->ne0, we->code_bits, we->code_bias, we->group_elems,
                                 (uint64_t) (we->ne0 / (8 / we->code_bits)), (uint64_t) (we->ne0 / we->group_elems), emb_, cs);
        }
        native_qsa_rms_norm_weighted(emb_, f32("pre_fc_norm_embedding.weight"), en_, (int) N, T, EPS, cs);
        native_quantize_q8_1(en_, xq_, (int) N, T, cs);
        native_mmvq(wt_fc_embedding, wp_fc_embedding, xq_, e2_, (int) N, (int) N, T, cs);
        g_mp.stamp(1, cs);
        if (hnorm_stream_)   // --mtp-hnorm stream: one RMS per stream, each scaled by its slice of the weight
            native_qsa_rms_norm_grouped(Rin_, f32("pre_fc_norm_hidden.weight"), hn_, (int) N, (int) HC, (int) (T * HC), EPS, cs);
        else
            native_qsa_rms_norm_weighted(Rin_, f32("pre_fc_norm_hidden.weight"), hn_, (int) (HC * N), T, EPS, cs);
        for (int c0 = 0; c0 < T * HC; c0 += 8) {
            const int nc = (int) std::min<int64_t>(8, T * HC - c0);
            native_quantize_q8_1(hn_ + (size_t) c0 * N, xq_, (int) N, nc, cs);
            native_mmvq(wt_fc_hidden, wp_fc_hidden, xq_, h2_ + (size_t) c0 * N, (int) N, (int) N, nc, cs);
        }
        add_streams_broadcast(h2_, e2_, R_, N, (int) HC, T, cs);
        g_mp.stamp(2, cs);
        // ---- the attention hyper-connection
        {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                fa[t].R = R_ + (size_t) t * HC * N; fa[t].R_out = R_ + (size_t) t * HC * N; fa[t].apply = false;
                fa[t].w_norm = f32("attn_hyper_connection.hc_norm.weight");
                fa[t].w_down = bf16("attn_hyper_connection.input_mix_weight_down.weight");
                fa[t].w_up = bf16("attn_hyper_connection.input_mix_weight_up.weight");
                fa[t].w_inject = bf16("attn_hyper_connection.block_inject_weight.weight");
                fa[t].eps = EPS; fa[t].lo = lo_ + t * g.hc_lr; fa[t].rs = rs_ + t * HC;
                fa[t].inject_out = inj_ + t * HC; fa[t].mixed = mixed_ + t * N;
            }
            fused_gr_read_multi(fa, T, xn_, cs);
        }
        g_mp.stamp(3, cs);
        // ---- attention: K/V into the layer's own cache
        native_quantize_q8_1(mixed_, xq_, (int) N, T, cs);
        native_mmvq(wt_k_proj, wp_k_proj, xq_, kcur_, (int) N, (int) (NKV * HD), T, cs);
        native_mmvq(wt_v_proj, wp_v_proj, xq_, vcur_, (int) N, (int) (NKV * HD), T, cs);
        // #783 PR-d (stuchapin909): the T tokens' K norm, K/V rotation and K/V append each run once over all T rows
        // (every row is independent; the rope keeps its per-token positions). STRATA_NO_BATCH_KV_STEP=1 appends per token.
        static const bool no_batch_kv = [] {
            const char* v = std::getenv("STRATA_NO_BATCH_KV_STEP");
            return v != nullptr && v[0] != '\0' && v[0] != '0';
        }();
        // #783 PR-f (stuchapin909): the per-head RMSNorm and the rope fused (bit-identical to the pair, rope_parity
        // check 6; STRATA_NO_NORM_ROPE=1 keeps the two; off on HIP until its parity check passes). K rows of one token
        // are consecutive in `pos`, tokens are NH apart, so a one-token window is the fusable case.
        const bool fuse_nr = native_rope_enabled() && native_norm_rope_usable((int) HD, (int) s.n_rot);
        if (T == 1 && fuse_nr) {
            native_qsa_rms_norm_rope(kcur_, (int) HD, f32("self_attn.k_norm.weight"), kcur_, (int) NKV, (int) HD,
                                     (int) s.n_rot, EPS, rope_scaling(), pos, cs);
        } else {
            native_qsa_rms_norm_weighted(kcur_, f32("self_attn.k_norm.weight"), kcur_, (int) HD, (int) (T * NKV), EPS, cs);
            for (int t = 0; t < T; ++t) {
                float* kc = kcur_ + t * NKV * HD;
                if (native_rope_enabled()) native_rope_apply(kc, kc, (int) NKV, (int) HD, (int) s.n_rot, rope_scaling(), pos + t * NH, cs);
                else rope_neox_apply(kc, kc, (int) NKV, (int) HD, (int) s.n_rot, st_.cos_tab, st_.sin_tab, pos + t * NH, cs);
            }
        }
        if (st_.kv_rot) {   // rotated K and V (kv_q4.hpp): Q4_0, and INT8 with STRATA_KV_ROT=1
            fwht256_inplace_cuda(kcur_, (int64_t) T * NKV, cs);
            fwht256_inplace_cuda(vcur_, (int64_t) T * NKV, cs);
        }
        // stored in the state's own format (#293 appended rotated INT8 K/V as Q4_0, into pools INT8 never has)
        if (!no_batch_kv && st_.kv_q4)
            kv_append_q4_steps(st_.k_q4, st_.v_q4, st_.page_table, step, 4, T, kcur_, vcur_, s, cs, &st_.host);
        else if (!no_batch_kv && st_.kv_int8)
            kv_append_q8_steps(st_.k_q, st_.v_q, st_.k_scale, st_.v_scale, st_.page_table, step, 4, kcur_, vcur_,
                               (int) (NKV * HD), T, s, cs, &st_.host);
        else
            for (int t = 0; t < T; ++t) {
                if (st_.kv_q4)
                    kv_append_q4_step(st_.k_q4, st_.v_q4, st_.page_table, step + t * 4, kcur_ + t * NKV * HD,
                                      vcur_ + t * NKV * HD, s, cs, &st_.host);
                else if (st_.kv_int8)
                    kv_append_q8_step(st_.k_q, st_.v_q, st_.k_scale, st_.v_scale, st_.page_table, step + t * 4,
                                      kcur_ + t * NKV * HD, vcur_ + t * NKV * HD, s, cs, &st_.host);
                else
                    kv_append_step(st_.k_pool, st_.v_pool, st_.page_table, step + t * 4, kcur_ + t * NKV * HD,
                                   vcur_ + t * NKV * HD, s, cs, &st_.host);
            }
        g_mp.stamp(4, cs);
    } catch (const std::exception& e) {
        err = std::string("mtp: ") + e.what();
        return false;
    }
    return true;
}

void MtpDrafter::norm_rope(float* data, const float* gamma, int rows, int cols, const int32_t* p, cudaStream_t cs) {
    using namespace strata::kernels;
    const QsaShapes s = shapes_of(*g_);
    native_qsa_rms_norm_weighted(data, gamma, data, cols, rows, EPS, cs);
    if (native_rope_enabled()) native_rope_apply(data, data, rows, cols, (int) s.n_rot, rope_scaling(), p, cs);
    else rope_neox_apply(data, data, rows, cols, (int) s.n_rot, st_.cos_tab, st_.sin_tab, p, cs);
}

bool MtpDrafter::record_rest(int step_row, cudaStream_t cs, std::string& err) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    const int T = 1;
    const int64_t N = g.n_embd, HC = g.hc, K = ss.k, NH = g.n_head, HD = g.head_dim;
    const QsaShapes s = shapes_of(g);
    const GrShapes gs{g.n_embd, g.hc, g.hc_lr};
    const int32_t* step = step_ + step_row * 4;
    const int32_t* pos = pos_ + step_row * NH;
    // --mtp-q4: the q and o projections' format and weights (Q8_0 unless the Q4_0 copies exist)
    int wt_q_proj, wt_o_proj;
    const void* wp_q_proj = wq("self_attn.q_proj.weight", wt_q_proj);
    const void* wp_o_proj = wq("self_attn.o_proj.weight", wt_o_proj);
    const bool fuse_nr = native_rope_enabled() && native_norm_rope_usable((int) HD, (int) s.n_rot);   // #783 PR-f
    try {
        // ---- dense attention over every cell
        native_mmvq(wt_q_proj, wp_q_proj, xq_, qfull_, (int) N, (int) (NH * 2 * HD), T, cs);
        // all T tokens' q rows at once (the rows of token t sit at pos[t * NH ..], so row r reads pos[r])
        if (fuse_nr) {
            native_qsa_rms_norm_rope(qfull_, (int) (2 * HD), f32("self_attn.q_norm.weight"), qcur_, (int) (T * NH),
                                     (int) HD, (int) s.n_rot, EPS, rope_scaling(), pos, cs);
        } else {
            if (cudaMemcpy2DAsync(qcur_, (size_t) HD * 4, qfull_, (size_t) HD * 2 * 4, (size_t) HD * 4,
                                  (size_t) (T * NH), cudaMemcpyDeviceToDevice, cs) != cudaSuccess) {
                err = "mtp: q split failed";
                return false;
            }
            norm_rope(qcur_, f32("self_attn.q_norm.weight"), (int) (T * NH), (int) HD, pos, cs);
        }
        if (st_.kv_rot) fwht256_inplace_cuda(qcur_, (int64_t) T * NH, cs);
        g_mp.stamp(5, cs);
        const QsaAttnPools pools = qsa_attn_pools(st_);
        if (floor_on_ && m_floor_ != nullptr)   // KVA: the window (or every cell) from the first real draft cell on
            window_ids_floor(const_cast<int32_t*>(step), T, window_ > 0 ? (int) window_ : 0x7fffffff, ident_, cap_, m_floor_, cs);
        else if (window_ > 0) window_ids(const_cast<int32_t*>(step), T, (int) window_, ident_, cap_, cs);
        // STRATA_QSA_SPLIT_ATTN=1: the draft layer's attention over its window, split the same way
        static const bool split_attn = [] { const char* v = std::getenv("STRATA_QSA_SPLIT_ATTN"); return v != nullptr && std::atoi(v) != 0; }();
        if (!(split_attn && qsa_decode_attn_split(qcur_, pools, ident_, step, cap_, s, attn_scratch_, (uint64_t) attn_scratch_floats_, attn_, T, cs)))
        qsa_decode_attn_batch(qcur_, pools, ident_, step, cap_, s, attn_scratch_, attn_, T, cs);
        if (st_.kv_rot) fwht256_inplace_cuda(attn_, (int64_t) T * NH, cs);
        g_mp.stamp(6, cs);
        native_qsa_gate_apply(attn_, qfull_, attn32_, (int) (T * NH), (int) HD, cs);
        native_quantize_q8_1(attn32_, xq_, (int) (NH * HD), T, cs);
        native_mmvq(wt_o_proj, wp_o_proj, xq_, bo_, (int) (NH * HD), (int) N, T, cs);
        g_mp.stamp(7, cs);
        // ---- the MLP hyper-connection (the attention write folded in)
        {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                fa[t].R = R_ + (size_t) t * HC * N; fa[t].R_out = R_ + (size_t) t * HC * N; fa[t].apply = true;
                fa[t].bo_prev = bo_ + t * N; fa[t].inj_prev = inj_ + t * HC;
                fa[t].w_norm = f32("mlp_hyper_connection.hc_norm.weight");
                fa[t].w_down = bf16("mlp_hyper_connection.input_mix_weight_down.weight");
                fa[t].w_up = bf16("mlp_hyper_connection.input_mix_weight_up.weight");
                fa[t].w_inject = bf16("mlp_hyper_connection.block_inject_weight.weight");
                fa[t].eps = EPS; fa[t].lo = lo_ + t * g.hc_lr; fa[t].rs = rs_ + t * HC;
                fa[t].inject_out = inj2_ + t * HC; fa[t].mixed = mixed_ + t * N;
            }
            fused_gr_read_multi(fa, T, xn_, cs);
        }
        g_mp.stamp(8, cs);
        // ---- MoE: router, the 512 resident experts, the shared expert (on a branch beside them on CUDA), the
        // combine, the write
        NativeSharedWeights nsw;
        nsw.gate_data = wq("mlp.shared_expert.gate_proj.weight", nsw.gate_type);
        nsw.up_data = wq("mlp.shared_expert.up_proj.weight", nsw.up_type);
        nsw.down_data = wq("mlp.shared_expert.down_proj.weight", nsw.down_type);
        nsw.q8_1 = xq_;   // the routed experts read hit_xq_
        const SForm none{};
        const bool need_bf16_x = !shared_expert_native_bf16_enabled();
        static const bool fuse_head_gr = [] {
            const char* v = std::getenv("STRATA_FUSE_HEAD_GR");
            return v != nullptr && std::atoi(v) != 0;
        }();
        static const bool head_mix_multi_on = [] {
#if defined(STRATA_USE_HIP)
            const char* v = std::getenv("STRATA_HEAD_MIX_MULTI");   // HIP: opt-in (=1)
            return v != nullptr && std::atoi(v) != 0;
#else
            const char* v = std::getenv("STRATA_HEAD_MIX_MULTI");
            return v == nullptr || std::atoi(v) != 0;
#endif
        }();
        // the shared expert only reads mixed_ / xq_ and writes shared_: a branch beside the router and the routed
        // experts, joined before the combine (CUDA; HIP keeps one stream: STRATA_MTP_SHARED_BRANCH=0 does too)
        static const bool branch_on = [] {
#if defined(STRATA_USE_HIP)
            return false;
#else
            const char* v = std::getenv("STRATA_MTP_SHARED_BRANCH");
            return v == nullptr || std::atoi(v) != 0;
#endif
        }();
        cudaStream_t sh_cs = cs;
        if (branch_on) {
            if (cudaEventRecord(sh_fork_, cs) != cudaSuccess || cudaStreamWaitEvent(side_, sh_fork_, 0) != cudaSuccess) {
                err = "mtp: the shared expert's branch";
                return false;
            }
            sh_cs = side_;
        }
        auto shared_rows = [&]() {
            for (int t = 0; t < T; ++t) {
                if (need_bf16_x) f32_to_bf16_bulk(mixed_ + t * N, x_bf16_, N, sh_cs);
                shared_expert(nullptr, nullptr, x_bf16_, none, nullptr, nullptr, nullptr, none, nullptr, nullptr, nullptr,
                              none, nullptr, nullptr, nullptr, bf16("mlp.shared_expert_gate.weight"), sh_scratch_,
                              shared_ + t * N, N, g.n_ff, 32, sh_cs, mixed_ + t * N, &nsw);
            }
        };
        if (branch_on) {
            shared_rows();
            if (cudaEventRecord(sh_join_, side_) != cudaSuccess) { err = "mtp: the shared expert's branch"; return false; }
        }
        // #783 PR-c (stuchapin909): a multi-token window routes in two launches (one router GEMV reading the weight once,
        // one top-10 over all rows - each row bitwise the single-token call) and combines in one
        if (T > 1 && native_router_enabled() && g.n_expert == 512 && K == 10) {
            bf16_gemv_fp32_mmvf_multi(mixed_, N, bf16("mlp.gate.weight"), logits_, g.n_expert, (int) N, (int) g.n_expert, T, cs);
            native_router_top10_multi(logits_, ids_, w_, T, cs);
        } else {
        for (int t = 0; t < T; ++t) {
            bf16_gemv_fp32_mmvf(mixed_ + t * N, bf16("mlp.gate.weight"), logits_ + t * g.n_expert, (int) N, (int) g.n_expert, cs);
            if (native_router_enabled() && g.n_expert == 512 && K == 10) native_router_top10(logits_ + t * g.n_expert, ids_ + t * K, w_ + t * K, cs);
            else router_top10(logits_ + t * g.n_expert, 1, (int) g.n_expert, (int) K, ids_ + t * K, w_ + t * K, cs);
        }
        }
        if (rr_res_ != nullptr && rr_margin_ > 0.0f && g.n_expert == 512 && K == 10) {
            try {
                native_route_resident(logits_, ids_, w_, rr_res_, T, rr_margin_, rr_lo_, rr_hi_, nullptr, cs);
            } catch (const std::exception& e) { err = std::string("mtp route-resident: ") + e.what(); return false; }
        }
        g_mp.stamp(9, cs);
        moe_group_resident(ids_, (int) (T * K), (int) K, experts_, (int64_t) strata::kernels::cpu::BLOB, grp_ptr_,
                           grp_start_, grp_counts_, hit_dst_, hit_slot_, cs);
        quantize_q8_0_scaled(mixed_, hit_xq_, hit_xs_, (int64_t) T * N, cs);
        moe_grouped_s2(grp_ptr_, grp_start_, grp_counts_, hit_dst_, hit_slot_, (int64_t) T * K, (int64_t) T * K, hit_xq_,
                       hit_xs_, hit_scratch_, parts_, cs);
        g_mp.stamp(10, cs);
        if (branch_on) {
            if (cudaStreamWaitEvent(cs, sh_join_, 0) != cudaSuccess) { err = "mtp: the shared expert's join"; return false; }
        } else {
            shared_rows();
        }
        g_mp.stamp(11, cs);
        static const bool no_multi_gr = [] {   // #783 PR-g: STRATA_NO_MULTI_GR=1 keeps the per-token GR calls
            const char* v = std::getenv("STRATA_NO_MULTI_GR");
            return v != nullptr && v[0] != '\0' && v[0] != '0';
        }();
        for (int t = 0; t < T; ++t) {
            if (!(T > 1 && native_moe_combine_enabled())) {
                if (native_moe_combine_enabled())
                    native_moe_combine(parts_ + (size_t) t * K * N, w_ + t * K, shared_ + t * N, y_ + t * N, N, K, cs);
                else
                    moe_combine(parts_ + (size_t) t * K * N, w_ + t * K, shared_ + t * N, y_ + t * N, N, K, cs);
            }
            if (!fuse_head_gr && no_multi_gr)
                gr_write(R_ + (size_t) t * HC * N, y_ + t * N, inj2_ + t * HC, gs, R_ + (size_t) t * HC * N, cs);
        }
        if (T > 1 && native_moe_combine_enabled())
            native_moe_combine_multi(parts_, w_, shared_, y_, N, K, T, cs);
        // #783 PR-g (stuchapin909): the T residual writes in one launch (each token's R, block output and injection are
        // its own, so doing them after the loop changes nothing)
        if (!fuse_head_gr && !no_multi_gr)
            gr_write_multi(R_, y_, inj2_, gs, R_, T, cs);
        g_mp.stamp(12, cs);
        // ---- the final mixer and the main model's head
        if (fuse_head_gr) {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                fa[t].R = R_ + (size_t) t * HC * N; fa[t].R_out = R_ + (size_t) t * HC * N; fa[t].apply = true;
                fa[t].bo_prev = y_ + t * N; fa[t].inj_prev = inj2_ + t * HC;
                fa[t].w_norm = f32("hyper_connection_mixer.hc_norm.weight");
                fa[t].w_down = bf16("hyper_connection_mixer.input_mix_weight_down.weight");
                fa[t].w_up = bf16("hyper_connection_mixer.input_mix_weight_up.weight");
                fa[t].w_inject = nullptr;
                fa[t].eps = EPS; fa[t].lo = lo_ + t * g.hc_lr; fa[t].rs = rs_ + t * HC;
                fa[t].inject_out = dummy_inj_; fa[t].mixed = sample_ + t * N;
            }
            fused_gr_read_multi(fa, T, xn_, cs);
        } else if (head_mix_multi_on) {
            // the window's rows in one read (CUDA; HIP keeps gr_read per row): bitwise the same sums
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                fa[t].R = R_ + (size_t) t * HC * N; fa[t].R_out = R_ + (size_t) t * HC * N; fa[t].apply = false;
                fa[t].w_norm = f32("hyper_connection_mixer.hc_norm.weight");
                fa[t].w_down = bf16("hyper_connection_mixer.input_mix_weight_down.weight");
                fa[t].w_up = bf16("hyper_connection_mixer.input_mix_weight_up.weight");
                fa[t].eps = EPS; fa[t].lo = lo_ + t * g.hc_lr; fa[t].rs = rs_ + t * HC; fa[t].mixed = sample_ + t * N;
            }
            fused_gr_read_multi(fa, T, xn_, cs);
        } else {
            for (int t = 0; t < T; ++t)
                gr_read(R_ + (size_t) t * HC * N, f32("hyper_connection_mixer.hc_norm.weight"),
                        bf16("hyper_connection_mixer.input_mix_weight_down.weight"),
                        bf16("hyper_connection_mixer.input_mix_weight_up.weight"), nullptr, EPS, gs, own_gr_,
                        sample_ + t * N, dummy_inj_, cs);
        }
        g_mp.stamp(13, cs);
        native_quantize_q8_1(sample_, xq_, (int) N, T, cs);
        const bool sub = dhead_ != nullptr;
        const int64_t nv = sub ? n_dvocab_ : n_vocab_;
        native_mmvq(sub ? dhead_type_ : head_->type(), sub ? dhead_ : head_->weights(), xq_, head_logits_, (int) N,
                    (int) nv, T, cs);
        g_mp.stamp(14, cs);
        if (coupled_rec_) {
            // coupled draft sampling: the target's chain and Philox draw for the row that will verify this draft
            // (counter = this cell + 1, from its step record), penalties over the ring; T == 1 (full layer)
            if (m_q_ != nullptr)   // STRATA_SPEC_PROB: a draw from q with its own stream, and q's list for the verifier
                spec_draft_sample(head_logits_, (int) nv, sub ? dvocab_ : nullptr, sub ? dinv_ : nullptr, (int) n_vocab_,
                                  cparams_, cring_, kCoupledHistCap, coupled_j_, step, cscratch_, out_ids_, probs_, m_q_,
                                  spec_gate_pick(), spec_draft_temp_scale(), cs);
            else
                coupled_draft_sample(head_logits_, (int) nv, sub ? dvocab_ : nullptr, sub ? dinv_ : nullptr, (int) n_vocab_,
                                     cparams_, cring_, kCoupledHistCap, coupled_j_, step, cscratch_, out_ids_, probs_, cs);
            return true;
        }
        if (argmax_rows_wanted()) {
            argmax_rows(head_logits_, T, (int) nv, arg_scratch_, out_ids_, cs);
        } else {
            SamplerParams sp;
            sp.greedy = true;
            sp.temperature = 0.0f;
            sample_tokens(head_logits_, T, (int) nv, nullptr, 0, sp, out_ids_, cs);
        }
        if (multi_block_head_ops()) row_top_prob_split(head_logits_, T, (int) nv, out_ids_, probs_, top_scratch_, cs);
        else row_top_prob(head_logits_, T, (int) nv, out_ids_, probs_, cs);
        if (sub) map_ids(out_ids_, dvocab_, T, cs);
        g_mp.stamp(15, cs);
    } catch (const std::exception& e) {
        err = std::string("mtp: ") + e.what();
        return false;
    }
    return true;
}

namespace {
bool finish_capture(cudaStream_t cs, bool ok, cudaGraphExec_t& exec, const char* what, std::string& err) {
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        return false;
    }
    cudaGraphExec_t got = nullptr;   // (a failed instantiate leaves nothing in the member)
    const cudaError_t ie = ce != cudaSuccess ? ce : cudaGraphInstantiate(&got, graph, 0);
    if (ie != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("mtp: ") + what + " capture: " + cudaGetErrorString(ie);   // the status that failed
        return false;
    }
    exec = got;
    mtp_list_nodes(graph, what);
    cudaGraphDestroy(graph);
    // an explicit upload: the first launch's implicit one blocked behind a device-side spin (verify.cpp); only a warm-up,
    // so a failure (no VRAM left) is cleared and the first launch uploads it
    if (cudaGraphUpload(exec, cs) != cudaSuccess || cudaStreamSynchronize(cs) != cudaSuccess) (void) cudaGetLastError();
    return true;
}
}  // namespace

bool MtpDrafter::capture_prefill(int T, std::string& err) {
    if (prefill_exec_[T]) return true;
    using namespace strata::kernels;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "mtp: begin capture"; return false; }
    copy_i32_from_mapped(tok_, m_tok_, T, cs_);
    copy_i32_from_mapped(step_, m_step_, (int64_t) T * 4, cs_);
    copy_i32_from_mapped(pos_, m_pos_, (int64_t) T * g_->n_head, cs_);
    const bool ok = record_forward(T, -1, cs_, err);   // K/V only, rows [0, T)
    return finish_capture(cs_, ok, prefill_exec_[T], "prefill", err);
}

bool MtpDrafter::capture_prefill_dev(int T, std::string& err) {
    if (prefill_dev_exec_[T]) return true;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "mtp: begin capture"; return false; }
    const bool ok = record_forward(T, -1, cs_, err);   // K/V only, rows [0, T); tok_/step_/pos_ filled before launch
    return finish_capture(cs_, ok, prefill_dev_exec_[T], "prefill (device inputs)", err);
}

bool MtpDrafter::capture_round(int T, bool coupled, std::string& err) {
    cudaGraphExec_t& exec = coupled ? round_exec_c_[T] : round_exec_[T];
    if (exec) return true;
    using namespace strata::kernels;
    const int64_t HCN = g_->hc * g_->n_embd;
    g_mp.init();   // (its allocation before the capture)
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "mtp: begin capture"; return false; }
    g_mp.base = 0;
    bool ok = true;
    // coupled: the request's chain and the penalty history's base, for this round's drafts
    if (coupled) coupled_draft_stage(m_cparams_, m_chist_, cparams_, cring_, kCoupledHistCap, cs_);
    // the catch-up: the layer's front for the window's T cells (their K/V), then its rest for row a only, on row
    // a's intermediates copied to row 0, at the cell the host staged in step row 2*max_t - 1; the draft chain is
    // one graph per step (`capture_step`) so the host can stop it when a draft is unlikely
    const int ra = 2 * max_t_ - 1;
    const int64_t N = g_->n_embd, NH = g_->n_head;
#if defined(STRATA_USE_HIP)   // AMD keeps its separate copies
    copy_i32_from_mapped(tok_, m_tok_, T, cs_);
    copy_i32_from_mapped(step_, m_step_, (int64_t) 2 * T * 4, cs_);
    copy_i32_from_mapped(pos_, m_pos_, (int64_t) 2 * T * NH, cs_);
    copy_i32_from_mapped(row_, m_row_, 2, cs_);
    copy_from_mapped(Rin_, window_R_, (int64_t) T * HCN, cs_);
    copy_i32_from_mapped(step_ + ra * 4, m_step_ + ra * 4, 4, cs_);
    copy_i32_from_mapped(pos_ + ra * NH, m_pos_ + ra * NH, NH, cs_);
#else   // the draft round's seven inputs in one launch
    const MappedCopy in[7] = {{tok_, m_tok_, T}, {step_, m_step_, (int64_t) 2 * T * 4}, {pos_, m_pos_, 2 * T * NH},
                              {row_, m_row_, 2}, {Rin_, window_R_, T * HCN}, {step_ + ra * 4, m_step_ + ra * 4, 4},
                              {pos_ + ra * NH, m_pos_ + ra * NH, NH}};
    copy_from_mapped_multi(in, 7, cs_);
#endif
    ok = record_front(T, 0, cs_, err);
    if (ok) {
        if (T > 1) {
            copy_row_to_first(row_, R_, HCN, inj_, g_->hc, mixed_, N, cs_);
            native_quantize_q8_1(mixed_, xq_, (int) N, 1, cs_);
        }
        coupled_rec_ = coupled;
        coupled_j_ = 0;
        ok = record_rest(ra, cs_, err);
        coupled_rec_ = false;
    }
    if (ok) mtp_select(R_, HCN, out_ids_, row_ + 1, Rin_, tok_, m_out_, 0, cs_, probs_, m_prob_);
    if (ok) g_mp.stamp(16, cs_);
    g_mp.base = -1;
    if (ok && force_on_ && !coupled) force_token(tok_, m_force_, 0, cs_);   // chain_launch: step 1's input
    return finish_capture(cs_, ok, exec, coupled ? "round (coupled)" : "round", err);
}

// Chain step j (1..max_t-2): one row at the cell staged in step row `max_t + j - 1`, from the previous step's
// residual and token (left in Rin_[0] / tok_[0] by mtp_select); draft j and its probability to the mapped outputs.
bool MtpDrafter::capture_step(int j, bool coupled, std::string& err) {
    cudaGraphExec_t& exec = coupled ? step_exec_c_[j] : step_exec_[j];
    if (exec) return true;
    using namespace strata::kernels;
    const int64_t HCN = g_->hc * g_->n_embd;
    const int row = max_t_ + j - 1;
    g_mp.init();
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "mtp: begin capture"; return false; }
#if defined(STRATA_USE_HIP)
    copy_i32_from_mapped(step_ + row * 4, m_step_ + row * 4, 4, cs_);
    copy_i32_from_mapped(pos_ + row * g_->n_head, m_pos_ + row * g_->n_head, g_->n_head, cs_);
#else
    const MappedCopy in[2] = {{step_ + row * 4, m_step_ + row * 4, 4},
                              {pos_ + row * g_->n_head, m_pos_ + row * g_->n_head, g_->n_head}};
    copy_from_mapped_multi(in, 2, cs_);
#endif
    coupled_rec_ = coupled;
    coupled_j_ = j;
    g_mp.base = j < kMpGraphs ? j : -1;
    bool ok = record_forward(1, row, cs_, err);
    coupled_rec_ = false;
    if (ok) mtp_select(R_, HCN, out_ids_, row_ + 1, Rin_, tok_, m_out_, j, cs_, probs_, m_prob_);
    if (ok) g_mp.stamp(16, cs_);
    g_mp.base = -1;
    if (ok && force_on_ && !coupled) force_token(tok_, m_force_, j, cs_);   // chain_launch: step j+1's input
    return finish_capture(cs_, ok, exec, coupled ? "step (coupled)" : "step", err);
}

void MtpDrafter::kv_restore(int64_t upto) {
    const OnDevice on_device(device_);
    if (st_.kv_mode != 2 || upto <= 0) return;
    // the ring's blocks below `upto`, from the host copy: a checkpoint resume may have left later cells in them
    const strata::kernels::QsaShapes s = shapes_of(*g_);
    const int64_t b1 = (upto + s.page_size - 1) / s.page_size, b0 = std::max<int64_t>(0, b1 - st_.n_slots);
    strata::kernels::kv_ring_restore(qsa_attn_pools(st_), st_.host, qsa_kv_format(st_), b0, b1, st_.n_slots, s, cs_);
    cudaStreamSynchronize(cs_);
}

bool MtpDrafter::prepare_prefill(std::string& err) {
    const OnDevice on_device(device_);
    const int64_t per_row = 1 + 4 + g_->n_head;
    if (pf_cap_ < (int64_t) max_t_ * per_row) {
        if (pf_dev_) cudaFree(pf_dev_);
        pf_dev_ = nullptr;
        pf_cap_ = 0;
        if (cudaMalloc((void**) &pf_dev_, (size_t) (max_t_ * per_row) * sizeof(int32_t)) != cudaSuccess) {
            err = "mtp prefill: the input records do not fit";
            return false;
        }
        pf_cap_ = max_t_ * per_row;
    }
    for (int T = 1; T <= max_t_; ++T)
        if (!capture_prefill_dev(T, err)) return false;   // (STRATA_MTP_PREFILL_SYNC's loop syncs anyway)
    return true;
}

// STRATA_MTP_PREFILL_SYNC: the prompt pass's per-group loop (see MtpDrafter::prefill); HIP's default
static bool prefill_per_group_sync() {
    static const bool v = [] {
        if (const char* e = std::getenv("STRATA_MTP_PREFILL_SYNC")) return std::atoi(e) != 0;
#if defined(STRATA_USE_HIP)
        return true;
#else
        return false;
#endif
    }();
    return v;
}

bool MtpDrafter::reserve_prefill(int64_t n, std::string& err) {
    const OnDevice on_device(device_);
    if (prefill_per_group_sync()) {
        for (int T = 1; T <= max_t_; ++T)
            if (!capture_prefill(T, err)) return false;
        return true;
    }
    if (!prepare_prefill(err)) return false;
    const int64_t per_row = 1 + 4 + g_->n_head;
    if (n <= 0) return true;
    // the records' pinned upload slots too (#1996's ring grows a slot at its first larger upload: a pinned free and
    // allocation, which take the driver's lock)
    if (!pf_ring_.reserve((size_t) (n * per_row) * sizeof(int32_t)))
        std::fprintf(stderr, "strata mtp: the prompt records' pinned upload slots could not be reserved (they grow at "
                             "first use)\n");
    if (pf_cap_ >= n * per_row) return true;
    if (pf_dev_) cudaFree(pf_dev_);
    pf_dev_ = nullptr;
    pf_cap_ = 0;
    if (cudaMalloc((void**) &pf_dev_, (size_t) (n * per_row) * sizeof(int32_t)) != cudaSuccess) {
        err = "mtp prefill: the input records do not fit";
        return false;
    }
    pf_cap_ = n * per_row;
    return true;
}

bool MtpDrafter::prefill(const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err,
                         bool sync) {
#if defined(STRATA_GPU_METAL)
    (void) R_rows; (void) sync;   // the rows are the decode window's own (the Metal drafter reads them in place)
    return strata::metal_engine::mtp_prefill(next_tokens, n, cell0, err);
#endif
    const OnDevice on_device(device_);
    const Clock::time_point t0 = Clock::now();
    const int64_t HCN = g_->hc * g_->n_embd;
    // cells the window can never reach again need no K/V
    const int64_t first_needed = this->first_needed();
    // E-4: every group's token / step / position records uploaded at once; each group is then device copies and a
    // graph on the one stream, with a single sync at the end (a group of <= max_t rows used to be staged in mapped
    // memory and synced before the next: ~5,500 host round trips on a 32K prompt).  The same work in the same
    // order.  STRATA_MTP_PREFILL_SYNC=1 keeps the old loop, =0 forces E-4.
    // HIP defaults to the old loop.  This pass runs whenever E-9 (Prefill::draft_kv) declines, which it does for the
    // drafter's ring (KV streaming, --kv-resident), and on gfx1201 / ROCm 7.2.4 the E-4 queue (a graph launch and four
    // device copies per group, ~2,000 groups per 8192-token chunk, no sync) sometimes never completes: the prompt hangs
    // in hipGraphLaunch or the final sync until the watchdog ends the engine.  R9700, full IQ3_XXS, --kv-resident
    // 32768, mixed load (chats, 32K and 7K prompts, deep follow-ups): E-4 hung in the first round on 2 of 2 tries,
    // the old loop ran 30 of 30 rounds clean, prompt speed unchanged.
    const bool per_group_sync = prefill_per_group_sync();
    const int64_t NHp = g_->n_head, per_row = 1 + 4 + NHp;
    if (!per_group_sync && n > 0) {
        if (pf_cap_ < n * per_row) {
            if (pf_dev_) cudaFree(pf_dev_);
            pf_dev_ = nullptr;
            pf_cap_ = 0;
            if (cudaMalloc((void**) &pf_dev_, (size_t) (n * per_row) * sizeof(int32_t)) != cudaSuccess) {
                err = "mtp prefill: the input records do not fit";
                return false;
            }
            pf_cap_ = n * per_row;
        }
        std::vector<int32_t> rec((size_t) (n * per_row));
        int32_t* tk = rec.data();
        int32_t* stp = tk + n;
        int32_t* ps = stp + 4 * n;
        for (int64_t i = 0; i < n; ++i) {
            const int64_t cell = cell0 + i;
            tk[i] = next_tokens[i];
            stp[i * 4 + 0] = (int32_t) cell;
            stp[i * 4 + 1] = (int32_t) (cell + 1);
            stp[i * 4 + 2] = (int32_t) ((cell + 1) / 4);
            stp[i * 4 + 3] = (int32_t) (cell + 1);
            for (int64_t h = 0; h < NHp; ++h) ps[i * NHp + h] = (int32_t) cell;
        }
        if (pf_ring_.h2d(pf_dev_, rec.data(), rec.size() * sizeof(int32_t), cs_) != cudaSuccess) {
            err = "mtp prefill: the input records upload failed";
            return false;
        }
        const int32_t* d_tk = pf_dev_;
        const int32_t* d_stp = d_tk + n;
        const int32_t* d_ps = d_stp + 4 * n;
        for (int64_t c = 0; c < n; c += max_t_) {
            const int T = (int) std::min<int64_t>(max_t_, n - c);
            if (cell0 + c + T <= first_needed) continue;
            if (!capture_prefill_dev(T, err)) return false;
            if (cudaMemcpyAsync(tok_, d_tk + c, (size_t) T * 4, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess ||
                cudaMemcpyAsync(step_, d_stp + c * 4, (size_t) T * 16, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess ||
                cudaMemcpyAsync(pos_, d_ps + c * NHp, (size_t) (T * NHp) * 4, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess ||
                cudaMemcpyAsync(Rin_, R_rows + (size_t) c * HCN, (size_t) T * HCN * sizeof(float), cudaMemcpyDeviceToDevice,
                                cs_) != cudaSuccess ||
                cudaGraphLaunch(prefill_dev_exec_[T], cs_) != cudaSuccess) {
                err = std::string("mtp prefill: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
        }
        if (!sync) {   // the caller orders on stream(); the records are staged (a pageable copy returns once taken)
            (void) cudaStreamQuery(cs_);
            ms_prefill += ms_since(t0);
            return true;
        }
        if (cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = std::string("mtp prefill: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        ms_prefill += ms_since(t0);
        return true;
    }
    for (int64_t c = 0; c < n; c += max_t_) {
        const int T = (int) std::min<int64_t>(max_t_, n - c);
        if (cell0 + c + T <= first_needed) continue;
        if (!capture_prefill(T, err)) return false;
        for (int t = 0; t < T; ++t) {
            const int64_t cell = cell0 + c + t;
            h_tok_[t] = next_tokens[c + t];
            h_step_[t * 4 + 0] = (int32_t) cell;
            h_step_[t * 4 + 1] = (int32_t) (cell + 1);
            h_step_[t * 4 + 2] = (int32_t) ((cell + 1) / 4);
            h_step_[t * 4 + 3] = (int32_t) (cell + 1);
            for (int64_t h = 0; h < g_->n_head; ++h) h_pos_[t * g_->n_head + h] = (int32_t) cell;
        }
        if (cudaMemcpyAsync(Rin_, R_rows + (size_t) c * HCN, (size_t) T * HCN * sizeof(float), cudaMemcpyDeviceToDevice,
                            cs_) != cudaSuccess ||
            cudaGraphLaunch(prefill_exec_[T], cs_) != cudaSuccess ||
            cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = std::string("mtp prefill: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
    }
    ms_prefill += ms_since(t0);
    return true;
}

// STRATA_DRAFT_TIMING=1: where a draft round's time goes, from cudaEvent pairs around each graph launch (GPU time of the round
// graph and of each step, and the idle gap the host's decision leaves between two steps) and the host's own wall time.
namespace {
struct DraftTiming {
    bool on = false;
    cudaEvent_t eb[16] = {}, ea[16] = {};
    double round_ms = 0, step_ms = 0, gap_ms = 0, wall_ms = 0, tail_ms = 0;
    double pre_ms = 0, launch_ms = 0, ple_ms = 0, wait0_ms = 0, waitj_ms = 0;   // host: staging, graph launch calls, PLE prefetch, spins
    int64_t rounds = 0, steps = 0;
    DraftTiming() {
        const char* e = std::getenv("STRATA_DRAFT_TIMING");
        on = e != nullptr && e[0] == '1';
    }
    void init() {
        if (!on || eb[0] != nullptr) return;
        for (int i = 0; i < 16; ++i)
            if (cudaEventCreate(&eb[i]) != cudaSuccess || cudaEventCreate(&ea[i]) != cudaSuccess) { on = false; (void) cudaGetLastError(); return; }
    }
    void report() const {
        if (rounds == 0) return;
        const double r = (double) rounds;
        std::fprintf(stderr, "strata mtp draft timing: %lld rounds, %.2f steps/round: wall %.3f ms = round graph %.3f + steps %.3f GPU "
                             "(%.3f per step) + host gaps between graphs %.3f + tail %.3f (per round)\n",
                     (long long) rounds, (double) steps / r, wall_ms / r, round_ms / r, step_ms / r,
                     steps > 0 ? step_ms / (double) steps : 0.0, gap_ms / r, tail_ms / r);
        std::fprintf(stderr, "strata mtp draft timing (host, per round): staging %.3f, graph launch calls %.3f, PLE prefetch %.3f, "
                             "wait for draft 0 %.3f, waits for the steps %.3f\n",
                     pre_ms / r, launch_ms / r, ple_ms / r, wait0_ms / r, waitj_ms / r);
    }
};
DraftTiming g_dt;
}  // namespace

bool MtpDrafter::draft(int T, const int32_t* tokens, int64_t p, int a, int32_t* drafts, std::string& err,
                       float* probs, float min_p, int* n_drafts, DraftResult* result) {
    if (result) { *result = {}; if (n_drafts) *n_drafts = 0; }
#if defined(STRATA_GPU_METAL)
    {
        if (T < 1 || T > max_t_ || a < 0 || a >= T) { err = "mtp: draft arguments out of range"; return false; }
        const int max_steps = (int) std::min<int64_t>(std::min(max_t_ - 1, max_drafts_), (int64_t) 1 << 30);
        const Clock::time_point t0 = Clock::now();
        int n = 0;
        if (!strata::metal_engine::mtp_draft(T, tokens, p, a, max_steps, drafts, probs, min_p, &n, err)) return false;
        for (int j = n; j < max_t_ - 1; ++j) { drafts[j] = 0; if (probs) probs[j] = 0.0f; }
        if (n_drafts) *n_drafts = n;
        if (result) *result = {n, max_steps};
        ms_draft += ms_since(t0);
        ++rounds;
        return true;
    }
#endif
    const OnDevice on_device(device_);
    if (T < 1 || T > max_t_ || a < 0 || a >= T) { err = "mtp: draft arguments out of range"; return false; }
    if (chain_live_) { err = "mtp: a pipelined chain is still in flight"; return false; }
    // #783 PR-i (stuchapin909): the round runs for the cells up to the accepted row a (T = a + 1): the K/V of the rejected
    // rows is not caught up - no later read reaches a cell past the one being drafted and the next round writes it first
    if (!mtp_catchup_all()) T = a + 1;
    const bool cp = coupled_active_;   // coupled draft sampling for this request: its own graphs
    if (!capture_round(T, cp, err)) return false;
    // Draft j writes its K/V at cell p + a + j, and the K/V ends at max_cells (its page table has no page past it):
    // near the context's end the round drafts only the cells that exist
    const int max_steps = (int) std::min<int64_t>(std::min(max_t_ - 1, max_drafts_), st_.max_cells - (p + a));
    for (int j = 1; j < max_steps; ++j)
        if (!capture_step(j, cp, err)) return false;
    const Clock::time_point t0 = Clock::now();
    const int64_t NH = g_->n_head;
    auto put = [&](int row, int64_t cell) {
        h_step_[row * 4 + 0] = (int32_t) cell;
        h_step_[row * 4 + 1] = (int32_t) (cell + 1);
        h_step_[row * 4 + 2] = (int32_t) ((cell + 1) / 4);
        h_step_[row * 4 + 3] = (int32_t) (cell + 1);
        for (int64_t h = 0; h < NH; ++h) h_pos_[row * NH + h] = (int32_t) cell;
    };
    for (int t = 0; t < T; ++t) {
        h_tok_[t] = tokens[t];
        put(t, p + t);
    }
    put(2 * max_t_ - 1, coupled_draft_cell(p, a, 0));   // p + a: draft 0's cell
    for (int j = 1; j < max_steps; ++j)
        put(max_t_ + j - 1, coupled_draft_cell(p, a, j));
    for (int j = 0; j < max_steps; ++j) ((volatile int32_t*) h_out_)[j] = -1;
    h_row_[0] = a;
    h_row_[1] = 0;
    if (h_force_ != nullptr) for (int j = 0; j < 16; ++j) h_force_[j] = -1;   // the forcing kernels: no-ops here
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (!stage_source_R(T, err)) return false;

    SessionState* const pss = ple_ss_ ? ple_ss_ : ss_;
    const bool do_ple = pss != nullptr && pss->ple.ready() && pss->ple.table != nullptr;
    int32_t ple_prev[2] = {do_ple ? pss->ple_prev[0] : 0, do_ple ? pss->ple_prev[1] : 0};
    const bool dth = g_dt.on;
    auto hms = [](Clock::time_point a0) { return std::chrono::duration<double, std::milli>(Clock::now() - a0).count(); };
    auto prefetch_ple = [&](int32_t tok) {
        if (!do_ple || tok < 0) return;
        const Clock::time_point tp = Clock::now();
        struct PleT { bool on; double& acc; Clock::time_point t; ~PleT() { if (on) acc += std::chrono::duration<double, std::milli>(Clock::now() - t).count(); } } ple_t{dth, g_dt.ple_ms, tp};
        uint32_t rows16[strata::kernels::PLE_N_HEADS];
        strata::kernels::ngram_rows(&tok, ple_prev, 1, pss->ple.consts, rows16);
        ple_prev[0] = ple_prev[1];
        ple_prev[1] = tok;
        pss->ple.table->prefetch_rows(rows16);
    };

    int n = 0;
    if (min_p <= 0.0f && !chain_rule_ && max_steps > 0) {
        if (cudaGraphLaunch(cp ? round_exec_c_[T] : round_exec_[T], cs_) != cudaSuccess) {
            err = std::string("mtp draft: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        for (int j = 1; j < max_steps; ++j) {
            if (cudaGraphLaunch(cp ? step_exec_c_[j] : step_exec_[j], cs_) != cudaSuccess) {
                err = std::string("mtp draft step: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
        }
        (void) cudaStreamQuery(cs_);
        prefetch_ple(tokens[a]);
        progress_at("mtp draft: waiting for the GPU");   // #2065
        for (int j = 0; j + 1 < max_steps; ++j) {
            uint32_t spins = 0;
            while (((volatile int32_t*) h_out_)[j] < 0) {
#if defined(_WIN32) || defined(__x86_64__)
                strata_cpu_pause();
#endif
                if ((++spins & 1023u) == 0 && cudaStreamQuery(cs_) != cudaErrorNotReady) break;
            }
            strata_load_fence();   // the token was seen: its probability is read after it (a no-op on x86)
            drafts[j] = ((volatile int32_t*) h_out_)[j];
            if (probs) probs[j] = ((volatile float*) h_prob_)[j];
            prefetch_ple(drafts[j]);
        }
        if (cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = std::string("mtp draft: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        const int last = max_steps - 1;
        drafts[last] = ((volatile int32_t*) h_out_)[last];
        if (probs) probs[last] = ((volatile float*) h_prob_)[last];
        prefetch_ple(drafts[last]);
        n = max_steps;
    } else if (max_steps > 0) {
        g_dt.init();
        const bool dt = g_dt.on;
        if (dt) g_dt.pre_ms += hms(t0);
        if (dt) cudaEventRecord(g_dt.eb[0], cs_);
        Clock::time_point tl = Clock::now();
        if (cudaGraphLaunch(cp ? round_exec_c_[T] : round_exec_[T], cs_) != cudaSuccess) {
            err = std::string("mtp draft: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        if (dt) g_dt.launch_ms += hms(tl);
        if (dt) cudaEventRecord(g_dt.ea[0], cs_);
        (void) cudaStreamQuery(cs_);
        prefetch_ple(tokens[a]);
        progress_at("mtp draft: waiting for the GPU");   // #2065: a stall here read as "decode", the verify window before it
        auto wait_step = [&](int j) {
            uint32_t spins = 0;
            while (((volatile int32_t*) h_out_)[j] < 0) {
#if defined(_WIN32) || defined(__x86_64__)
                strata_cpu_pause();
#endif
                if ((++spins & 1023u) == 0 && cudaStreamQuery(cs_) != cudaErrorNotReady) break;
            }
            strata_load_fence();
        };
        tl = Clock::now();
        wait_step(0);
        if (dt) g_dt.wait0_ms += hms(tl);
        drafts[0] = ((volatile int32_t*) h_out_)[0];
        float pj = ((volatile float*) h_prob_)[0];
        if (probs) probs[0] = pj;
        n = 1;
        for (int j = 1; j < max_steps && (chain_rule_ ? chain_rule_(j - 1, pj) : pj >= min_p); ++j) {
            if (dt) cudaEventRecord(g_dt.eb[j], cs_);
            tl = Clock::now();
            if (cudaGraphLaunch(cp ? step_exec_c_[j] : step_exec_[j], cs_) != cudaSuccess) {
                err = std::string("mtp draft step: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
            if (dt) g_dt.launch_ms += hms(tl);
            if (dt) cudaEventRecord(g_dt.ea[j], cs_);
            (void) cudaStreamQuery(cs_);
            prefetch_ple(drafts[j - 1]);
            tl = Clock::now();
            wait_step(j);
            if (dt) g_dt.waitj_ms += hms(tl);
            drafts[j] = ((volatile int32_t*) h_out_)[j];
            pj = ((volatile float*) h_prob_)[j];
            if (probs) probs[j] = pj;
            ++n;
        }
        if (chain_rule_ && n >= max_steps) (void) chain_rule_(n - 1, pj);   // the last step of a full chain
        prefetch_ple(drafts[n - 1]);
        const Clock::time_point t_pre_sync = Clock::now();
        if (cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = std::string("mtp draft: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        if (dt) {
            float ms = 0;
            double r = 0, s = 0, gp = 0;
            if (cudaEventElapsedTime(&ms, g_dt.eb[0], g_dt.ea[0]) == cudaSuccess) r = ms;
            for (int j = 1; j < n; ++j) {
                if (cudaEventElapsedTime(&ms, g_dt.eb[j], g_dt.ea[j]) == cudaSuccess) s += ms;
                if (cudaEventElapsedTime(&ms, g_dt.ea[j - 1], g_dt.eb[j]) == cudaSuccess) gp += ms;
            }
            g_dt.round_ms += r; g_dt.step_ms += s; g_dt.gap_ms += gp;
            g_dt.steps += n - 1; ++g_dt.rounds;
            g_dt.wall_ms += ms_since(t0);
            g_dt.tail_ms += std::chrono::duration<double, std::milli>(Clock::now() - t_pre_sync).count();
            if (g_dt.rounds % 300 == 0) g_dt.report();
        }
        if (g_mp.on) {
            g_mp.collect(n);
            if (g_mp.n[0] % 300 == 0) g_mp.report();
        }
    }
    for (int j = n; j < max_t_ - 1; ++j) { drafts[j] = 0; if (probs) probs[j] = 0.0f; }
    if (n_drafts) *n_drafts = n;
    if (result) *result = {n, max_steps};
    ms_draft += ms_since(t0);
    ++rounds;
    return true;
}

bool MtpDrafter::stage_source_R(int T, std::string& err) {
    if (src_R_ == nullptr || src_R_ == window_R_) return true;
    const Clock::time_point t_src = Clock::now();
    struct SrcTime { MtpDrafter& d; Clock::time_point t; ~SrcTime() { d.ms_src += ms_since(t); ++d.n_src; } } src_time{*this, t_src};
    if (cudaMemcpyAsync((void*) window_R_, src_R_, (size_t) T * (size_t) (g_->hc * g_->n_embd) * sizeof(float),
                        cudaMemcpyDeviceToDevice, cs_) != cudaSuccess) {
        err = "mtp: staging the window's residual rows failed";
        return false;
    }
    return true;
}

bool MtpDrafter::prepare_chain(std::string& err) {
    const OnDevice on_device(device_);
    if (!force_on_ || h_force_ == nullptr) { err = "mtp: chains need set_force_capture before load"; return false; }
    for (int T = 1; T <= max_t_; ++T)
        if (!capture_round(T, false, err)) return false;
    for (int j = 1; j <= max_t_ - 2; ++j)
        if (!capture_step(j, false, err)) return false;
    if (ev_chain_ == nullptr && cudaEventCreateWithFlags(&ev_chain_, cudaEventDisableTiming) != cudaSuccess) {
        err = "mtp: event creation failed";
        return false;
    }
    for (cudaEvent_t& e : ev_step_)
        if (e == nullptr && cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) {
            err = "mtp: event creation failed";
            return false;
        }
    return true;
}

bool MtpDrafter::chain_launch(int T, const int32_t* tokens, int64_t p, int a, const int32_t* force, int n_force,
                              int n_out, int n_early, std::string& err) {
    const OnDevice on_device(device_);
    if (chain_live_) { err = "mtp: a chain is already in flight"; return false; }
    if (!force_on_ || ev_chain_ == nullptr) { err = "mtp: chains need prepare_chain"; return false; }
    if (T < 1 || T > max_t_ || a < 0 || a >= T || n_out < 1 || n_out > max_t_ - 1 || n_force < 0 || n_force > n_out ||
        n_force > 16) {
        err = "mtp: chain arguments out of range";
        return false;
    }
    if (round_exec_[T] == nullptr) { err = "mtp: chain graphs not prepared"; return false; }
    for (int j = 1; j < n_out; ++j)
        if (step_exec_[j] == nullptr) { err = "mtp: chain graphs not prepared"; return false; }
    const Clock::time_point t0 = Clock::now();
    const int64_t NH = g_->n_head;
    auto put = [&](int row, int64_t cell) {
        h_step_[row * 4 + 0] = (int32_t) cell;
        h_step_[row * 4 + 1] = (int32_t) (cell + 1);
        h_step_[row * 4 + 2] = (int32_t) ((cell + 1) / 4);
        h_step_[row * 4 + 3] = (int32_t) (cell + 1);
        for (int64_t h = 0; h < NH; ++h) h_pos_[row * NH + h] = (int32_t) cell;
    };
    for (int t = 0; t < T; ++t) {
        h_tok_[t] = tokens[t];
        put(t, p + t);
    }
    put(2 * max_t_ - 1, p + a);                                      // output 0's cell
    for (int j = 1; j < n_out; ++j) put(max_t_ + j - 1, p + a + j);  // every step's cell, staged at once
    for (int j = 0; j < 16; ++j) h_force_[j] = j < n_force ? force[j] : -1;
    h_row_[0] = a;
    h_row_[1] = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (!stage_source_R(T, err)) return false;
    chain_early_ = std::max(0, std::min(n_early, n_out));
    bool ok = cudaGraphLaunch(round_exec_[T], cs_) == cudaSuccess;
    if (ok && chain_early_ >= 1) ok = cudaEventRecord(ev_step_[0], cs_) == cudaSuccess;
    for (int j = 1; ok && j < n_out; ++j) {
        ok = cudaGraphLaunch(step_exec_[j], cs_) == cudaSuccess;
        if (ok && j < chain_early_) ok = cudaEventRecord(ev_step_[j], cs_) == cudaSuccess;
    }
    if (!ok || cudaEventRecord(ev_chain_, cs_) != cudaSuccess) {
        err = std::string("mtp chain: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    (void) cudaStreamQuery(cs_);   // WDDM: submit now
    steps_seen_ = 0;
    chain_live_ = true;
    chain_n_ = n_out;
    ms_draft += ms_since(t0);
    ++rounds;
    return true;
}

int MtpDrafter::chain_outputs_ready(std::string& err) {
    if (!chain_live_) return chain_n_;
    const OnDevice on_device(device_);
    while (steps_seen_ < chain_early_) {
        const cudaError_t q = cudaEventQuery(ev_step_[steps_seen_]);
        if (q == cudaErrorNotReady) break;
        if (q != cudaSuccess) { err = std::string("mtp chain: ") + cudaGetErrorString(q); return -1; }
        chain_tok_[steps_seen_] = ((volatile int32_t*) h_out_)[steps_seen_];
        chain_prob_[steps_seen_] = ((volatile float*) h_prob_)[steps_seen_];
        ++steps_seen_;
    }
    return steps_seen_;
}

int MtpDrafter::chain_poll(std::string& err) {
    if (!chain_live_) return 1;
    const OnDevice on_device(device_);
    const cudaError_t q = cudaEventQuery(ev_chain_);
    if (q == cudaErrorNotReady) return 0;
    chain_live_ = false;
    if (q != cudaSuccess) { err = std::string("mtp chain: ") + cudaGetErrorString(q); return -1; }
    for (int j = 0; j < chain_n_; ++j) {
        chain_tok_[j] = ((volatile int32_t*) h_out_)[j];
        chain_prob_[j] = ((volatile float*) h_prob_)[j];
    }
    for (int j = chain_n_; j < 8; ++j) { chain_tok_[j] = 0; chain_prob_[j] = 0.0f; }
    return 1;
}

bool MtpDrafter::draft_first(int T, const float* R_row, int32_t token, int64_t cell, int32_t* drafts, std::string& err,
                             float* probs, float min_p, int* n_drafts, DraftResult* result) {
    if (result) { *result = {}; if (n_drafts) *n_drafts = 0; }
#if defined(STRATA_GPU_METAL)
    {   // the decode window's row 0 is the residual (the window that just ran)
        (void) R_row;
        std::vector<int32_t> toks((size_t) T, token);
        return draft(1, toks.data(), cell, 0, drafts, err, probs, min_p, n_drafts, result);
    }
#endif
    const OnDevice on_device(device_);
    // row 0 is the real pair; rows 1.. repeat it and only write cells the next round overwrites
    const int64_t HCN = g_->hc * g_->n_embd;
    for (int t = 0; t < T; ++t)
        if (cudaMemcpy((void*) (window_R_ + (size_t) t * HCN), R_row, (size_t) HCN * sizeof(float),
                       cudaMemcpyDeviceToDevice) != cudaSuccess) {
            err = "mtp: staging the first residual failed";
            return false;
        }
    std::vector<int32_t> toks((size_t) T, token);
    return draft(T, toks.data(), cell, 0, drafts, err, probs, min_p, n_drafts, result);
}

}  // namespace strata::core
