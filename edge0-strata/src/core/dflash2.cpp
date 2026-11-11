// src/core/dflash2.cpp - see include/strata/core/dflash2.hpp.
#include "strata/core/dflash2.hpp"

#include "strata/core/native_head.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/dflash2.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <atomic>
#include <initializer_list>
#include <stdexcept>

namespace strata::core {
namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

bool mapped(size_t bytes, void** h, void** d) {
    if (cudaHostAlloc(h, bytes, cudaHostAllocMapped) != cudaSuccess) return false;
    std::memset(*h, 0, bytes);
    return cudaHostGetDevicePointer(d, *h, 0) == cudaSuccess;
}

constexpr int GGML_Q8_0 = 8;

}  // namespace

Dflash2Drafter::~Dflash2Drafter() {
    if (device_ < 0) return;
    prof_report();
    prof_free();
    const OnDevice on_device(device_);
    if (cs_) cudaStreamSynchronize(cs_);
    for (cudaGraphExec_t& e : exec_)
        if (e) cudaGraphExecDestroy(e);
    auto fr = [](void* p) { if (p) cudaFree(p); };
    fr(blob_); fr(d_meta_); fr(xq_); fr(xqp_); fr(xdp_); fr(xqc_); fr(xdc_); fr(rt_ctx_); fr(rt_blk_);
    for (int l = 0; l < 8; ++l) { fr(kring_[l]); fr(vring_[l]); }
    fr(fcy_); fr(hctx_); fr(kc_); fr(vc_); fr(h_); fr(xn_); fr(xc_); fr(dyn_); fr(q_); fr(qn_); fr(k_); fr(v_); fr(kblk_); fr(ao_); fr(o_);
    fr(gate_); fr(up_); fr(mlp_); fr(part_); fr(fn_); fr(logits_); fr(topval_); fr(hsel_); fr(toptok_); fr(topscratch_);
    fr(dvocab_); fr(dhead_);
    if (h_meta_) cudaFreeHost(h_meta_);
    if (h_out_) cudaFreeHost(h_out_);
    if (cs_) cudaStreamDestroy(cs_);
    if (cs2_) cudaStreamDestroy(cs2_);
    if (ev_fork_) cudaEventDestroy(ev_fork_);
    if (ev_join_) cudaEventDestroy(ev_join_);
    for (cudaEvent_t e : ev_ctx_) if (e) cudaEventDestroy(e);
    if (ev_a_) cudaEventDestroy(ev_a_);
    if (ev_b_) cudaEventDestroy(ev_b_);
}

// STRATA_DF2_PROFILE=1 (measurement only): the round runs without its graph and an event follows every stage; the per-stage averages print at exit
struct Dflash2Drafter::Prof {
    std::vector<cudaEvent_t> ev;
    std::vector<std::string> name;
    std::vector<std::pair<std::string, double>> total;
    int64_t rounds = 0;
};

void Dflash2Drafter::prof_free() {
    delete prof_;
    prof_ = nullptr;
}

// STRATA_DF2_SKIP=attn,head,ctx,mlp,qkvo (measurement only): leaves those parts out of the round, so the graph's GPU time without them can be read
static bool skip_part(const char* what) {
    static const std::string list = [] { const char* e = std::getenv("STRATA_DF2_SKIP"); return std::string(e ? e : ""); }();
    return !list.empty() && list.find(what) != std::string::npos;
}

void Dflash2Drafter::stamp(const char* name) {
    if (!prof_) return;
    cudaEvent_t e;
    cudaEventCreate(&e);
    cudaEventRecord(e, cs_);
    prof_->ev.push_back(e);
    prof_->name.push_back(name);
}

void Dflash2Drafter::prof_collect() {
    if (!prof_ || prof_->ev.size() < 2) return;
    cudaStreamSynchronize(cs_);
    for (size_t i = 1; i < prof_->ev.size(); ++i) {
        float ms = 0;
        cudaEventElapsedTime(&ms, prof_->ev[i - 1], prof_->ev[i]);
        auto it = std::find_if(prof_->total.begin(), prof_->total.end(), [&](const auto& t) { return t.first == prof_->name[i]; });
        if (it == prof_->total.end()) prof_->total.emplace_back(prof_->name[i], ms); else it->second += ms;
    }
    for (cudaEvent_t e : prof_->ev) cudaEventDestroy(e);
    prof_->ev.clear();
    prof_->name.clear();
    ++prof_->rounds;
}

void Dflash2Drafter::prof_report() const {
    if (!prof_ || prof_->rounds == 0) return;
    double sum = 0;
    for (const auto& t : prof_->total) sum += t.second;
    std::fprintf(stderr, "strata dflash2 profile (%lld rounds, ms per round, total %.3f):", (long long) prof_->rounds, sum / (double) prof_->rounds);
    for (const auto& t : prof_->total) std::fprintf(stderr, " %s %.3f", t.first.c_str(), t.second / (double) prof_->rounds);
    std::fputc(10, stderr);
}

const Dflash2Drafter::Tensor& Dflash2Drafter::find(const std::string& name) const {
    for (const Tensor& t : tensors_)
        if (t.name == name) return t;
    throw std::runtime_error("dflash2: tensor " + name + " is missing from the pack");
}

bool Dflash2Drafter::load(const std::string& path, int device, std::string& err) {
    device_ = device;
    const OnDevice on_device(device_);
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) { err = "dflash2: cannot open " + path; return false; }
    struct Closer { std::FILE* f; ~Closer() { std::fclose(f); } } closer{f};
    uint32_t hdr[3];
    if (std::fread(hdr, 4, 3, f) != 3 || std::memcmp(hdr, "DF2P", 4) != 0 || hdr[1] != 1) { err = "dflash2: " + path + " is not a DF2P v1 pack"; return false; }
    std::string index(hdr[2], ' ');
    if (std::fread(index.data(), 1, hdr[2], f) != hdr[2]) { err = "dflash2: the pack's index is cut short"; return false; }
    const long data_off = 12 + (long) hdr[2];
    uint64_t total = 0;
    for (size_t i = 0; i < index.size();) {
        const size_t j = index.find('\n', i);
        if (j == std::string::npos) break;
        const std::string line = index.substr(i, j - i);
        i = j + 1;
        char a[256], b[64], c[256];
        long long v1, v2, v3, v4;
        if (std::sscanf(line.c_str(), "cfg %63s %255s", b, c) == 2) {
            const std::string key = b;
            if (key == "hidden") N_ = std::atoi(c);
            else if (key == "layers") L_ = std::atoi(c);
            else if (key == "heads") NH_ = std::atoi(c);
            else if (key == "kv_heads") NKV_ = std::atoi(c);
            else if (key == "head_dim") HD_ = std::atoi(c);
            else if (key == "inter") I_ = std::atoi(c);
            else if (key == "window") window_ = std::atoi(c);
            else if (key == "rope_theta") theta_ = (float) std::atof(c);
            else if (key == "eps") eps_ = (float) std::atof(c);
            else if (key == "conv_group") group_ = std::atoi(c);
            else if (key == "rank") rank_ = std::atoi(c);
            else if (key == "mask_token") mask_token_ = std::atoi(c);
            else if (key == "vocab") vocab_ = std::atoll(c);
            else if (key == "taps") {
                taps_.clear();
                for (const char* q = c; *q;) { taps_.push_back(std::atoi(q)); while (*q && *q != ',') ++q; if (*q == ',') ++q; }
            }
        } else if (std::sscanf(line.c_str(), "t %255s %63s %lld %lld %lld %lld", a, b, &v1, &v2, &v3, &v4) == 6) {
            Tensor t;
            t.name = a; t.dtype = b; t.rows = v1; t.cols = v2; t.off = (uint64_t) v3; t.bytes = (uint64_t) v4;
            tensors_.push_back(t);
            total = std::max(total, t.off + t.bytes);
        }
    }
    if (tensors_.empty() || NH_ != 32 || NKV_ != 8 || HD_ != 128 || rank_ != 256 || N_ != 2560 || L_ > 8 || window_ + 16 > kRing) {
        err = "dflash2: the pack's shape is not the Flash-Next drafter this engine was written for";
        return false;
    }
    if (cudaMalloc((void**) &blob_, total) != cudaSuccess) { err = "dflash2: the pack does not fit in VRAM (" + std::to_string(total >> 20) + " MiB)"; return false; }
    vram_ += total;
    uint64_t max_bytes = 0;
    for (const Tensor& t : tensors_) max_bytes = std::max(max_bytes, t.bytes);
    std::vector<uint8_t> stage(max_bytes);
    uint8_t* tmp = nullptr;
    if (cudaMalloc((void**) &tmp, max_bytes) != cudaSuccess) { err = "dflash2: upload scratch"; return false; }
    struct TmpFree { uint8_t* p; ~TmpFree() { cudaFree(p); } } tmp_free{tmp};
    for (const Tensor& t : tensors_) {
#if defined(_WIN32)
        _fseeki64(f, data_off + (long long) t.off, SEEK_SET);
#else
        fseeko(f, data_off + (off_t) t.off, SEEK_SET);
#endif
        if (std::fread(stage.data(), 1, t.bytes, f) != t.bytes) { err = "dflash2: the pack is shorter than its index"; return false; }
        if (t.dtype == "q8_0") {   // GGUF blocks -> the tiled qs plane at the tensor's offset, then the d plane (same total bytes)
            if (cudaMemcpy(tmp, stage.data(), t.bytes, cudaMemcpyHostToDevice) != cudaSuccess) { err = "dflash2: weight upload failed"; return false; }
            kernels::df2_q8_tile(tmp, blob_ + t.off, blob_ + t.off + (uint64_t) t.rows * (uint64_t) t.cols, (int) t.cols, t.rows * t.cols / 32, nullptr);
            if (cudaDeviceSynchronize() != cudaSuccess) { err = "dflash2: weight conversion failed"; return false; }
        } else if (cudaMemcpy(blob_ + t.off, stage.data(), t.bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "dflash2: weight upload failed";
            return false;
        }
    }
    // the draft round runs while the verifier's async commit still has work on the card: a high-priority stream keeps it from queueing behind that
    // (CUDA by default; on HIP only with STRATA_DF2_PRIO=1, where the engine's other streams keep the default priority)
    int prio_lo = 0, prio_hi = 0;
#if defined(STRATA_USE_HIP)
    static const bool hi_prio = [] { const char* e = std::getenv("STRATA_DF2_PRIO"); return e && e[0] == '1'; }();
    if (hi_prio) hipDeviceGetStreamPriorityRange(&prio_lo, &prio_hi);
    if (hipStreamCreateWithPriority(&cs_, hipStreamNonBlocking, hi_prio ? prio_hi : 0) != cudaSuccess ||
        hipStreamCreateWithPriority(&cs2_, hipStreamNonBlocking, hi_prio ? prio_hi : 0) != cudaSuccess) {
#else
    static const bool hi_prio = [] { const char* e = std::getenv("STRATA_DF2_PRIO"); return !(e && e[0] == '0'); }();
    cudaDeviceGetStreamPriorityRange(&prio_lo, &prio_hi);
    if (cudaStreamCreateWithPriority(&cs_, cudaStreamNonBlocking, hi_prio ? prio_hi : 0) != cudaSuccess ||
        cudaStreamCreateWithPriority(&cs2_, cudaStreamNonBlocking, hi_prio ? prio_hi : 0) != cudaSuccess) {
#endif
        err = "dflash2: stream";
        return false;
    }
    if (cudaEventCreateWithFlags(&ev_fork_, cudaEventDisableTiming) != cudaSuccess || cudaEventCreateWithFlags(&ev_join_, cudaEventDisableTiming) != cudaSuccess) {
        err = "dflash2: events";
        return false;
    }
    for (int l = 0; l < 8; ++l)
        if (cudaEventCreateWithFlags(&ev_ctx_[l], cudaEventDisableTiming) != cudaSuccess) { err = "dflash2: events"; return false; }
    if (const char* e = std::getenv("STRATA_DF2_PROFILE"); e && e[0] == '1') prof_ = new Prof();
    if (const char* e = std::getenv("STRATA_DF2_TIMING"); e && e[0] == '1') {
        timing_ = true;
        cudaEventCreate(&ev_a_);
        cudaEventCreate(&ev_b_);
    }
    return alloc(err);
}

bool Dflash2Drafter::alloc(std::string& err) {
    auto dev = [&](auto** p, size_t n) {
        using T = std::remove_pointer_t<std::remove_reference_t<decltype(*p)>>;
        if (cudaMalloc((void**) p, n * sizeof(T)) != cudaSuccess) return false;
        vram_ += n * sizeof(T);
        return true;
    };
    const size_t B = kRows;
    bool ok = dev(&d_meta_, 32) && dev(&xq_, kernels::native_q8_1_bytes(N_, kRows) + 4096) && dev(&xqp_, (size_t) 5 * N_ * 8 + 4096) && dev(&xdp_, (size_t) 5 * N_ / 32 * 8 + 64) && dev(&xqc_, (size_t) 5 * N_ * 8 + 4096) && dev(&xdc_, (size_t) 5 * N_ / 32 * 8 + 64) && dev(&rt_ctx_, B * 128) && dev(&rt_blk_, B * 128) &&
              dev(&fcy_, B * N_) && dev(&hctx_, B * N_) && dev(&kc_, B * NKV_ * HD_) && dev(&vc_, B * NKV_ * HD_) && dev(&h_, B * N_) && dev(&xn_, B * N_) &&
              dev(&xc_, B * N_) && dev(&dyn_, B * 640) && dev(&q_, B * NH_ * HD_) && dev(&qn_, B * NH_ * HD_) && dev(&k_, B * NKV_ * HD_) && dev(&v_, B * NKV_ * HD_) &&
              dev(&kblk_, B * NKV_ * HD_) && dev(&ao_, B * NH_ * HD_) && dev(&o_, B * N_) && dev(&gate_, B * I_) && dev(&up_, B * I_) && dev(&mlp_, B * N_) &&
              dev(&part_, (size_t) kernels::kDf2MaxChunks * 8 * 32 * 130) && dev(&fn_, B * N_) && dev(&topval_, kDrafts * 16) && dev(&toptok_, kDrafts * 16) &&
              dev(&hsel_, kDrafts * rank_) && dev(&topscratch_, (size_t) kDrafts * kernels::kDf2TopBlocks * kernels::kDf2TopK * 2);
    for (int l = 0; ok && l < L_; ++l) ok = dev(&kring_[l], (size_t) kRing * NKV_ * HD_) && dev(&vring_[l], (size_t) kRing * NKV_ * HD_);
    if (!ok) { err = "dflash2: the drafter's buffers do not fit in VRAM"; return false; }
    void* hd = nullptr;
    void* dd = nullptr;
    if (!mapped(32 * 4, (void**) &hd, &dd)) { err = "dflash2: mapped staging"; return false; }
    h_meta_ = (int32_t*) hd;
    m_meta_ = (int32_t*) dd;
    if (!mapped(64 * 4, &hd, &dd)) { err = "dflash2: mapped output"; return false; }
    h_out_ = (int32_t*) hd;
    out_tok_ = (int32_t*) dd;            // the select kernel writes the 7 picks here (host-visible) ...
    out_prob_ = (float*) dd + 16;        // ... and their probabilities at +16 ints
    cudaMemset(rt_ctx_, 0, B * 128 * 4);
    cudaMemset(xqp_, 0, (size_t) 5 * N_ * 8 + 4096);
    cudaMemset(xdp_, 0, ((size_t) 5 * N_ / 32 * 8 + 64) * 4);
    cudaMemset(xqc_, 0, (size_t) 5 * N_ * 8 + 4096);
    cudaMemset(xdc_, 0, ((size_t) 5 * N_ / 32 * 8 + 64) * 4);
    for (int l = 0; l < L_; ++l) { cudaMemset(kring_[l], 0, (size_t) kRing * NKV_ * HD_ * 4); cudaMemset(vring_[l], 0, (size_t) kRing * NKV_ * HD_ * 4); }
    return cudaDeviceSynchronize() == cudaSuccess;
}

bool Dflash2Drafter::bind_raw(EmbedFn embed, const void* head_w, int head_type, int64_t head_rows, const std::vector<int32_t>& vocab, std::string& err) {
    const OnDevice on_device(device_);
    embed_ = std::move(embed);
    head_w_ = head_w;
    head_type_ = head_type;
    n_dvocab_ = head_rows;
    if (!vocab.empty()) {
        n_dvocab_ = (int64_t) vocab.size();
        if (cudaMalloc((void**) &dvocab_, vocab.size() * 4) != cudaSuccess) { err = "dflash2: draft vocabulary"; return false; }
        vram_ += vocab.size() * 4;
        cudaMemcpy(dvocab_, vocab.data(), vocab.size() * 4, cudaMemcpyHostToDevice);
    }
    if ((n_dvocab_ + kernels::kDf2TopBlocks - 1) / kernels::kDf2TopBlocks > 4096) { err = "dflash2: the head has too many rows for the top-k"; return false; }
    if (!logits_ && cudaMalloc((void**) &logits_, (size_t) kDrafts * (size_t) n_dvocab_ * 4) != cudaSuccess) { err = "dflash2: the draft logits do not fit"; return false; }
    vram_ += (uint64_t) kDrafts * (uint64_t) n_dvocab_ * 4;
    bound_ = true;
    return true;
}

bool Dflash2Drafter::bind(const WeightTable& wt, const NativeHead* head, const std::string& vocab_file, std::string& err) {
    const OnDevice on_device(device_);
    if (head == nullptr || !head->loaded()) { err = "dflash2: the drafter needs the native head (--native)"; return false; }
    const WeightRef* we = wt.find("token_embd.weight");
    if (we == nullptr) { err = "dflash2: token_embd.weight is missing"; return false; }
    EmbedFn embed = [we](const int32_t* tok, int n, float* out, cudaStream_t cs) {
        if (const NativeEmbed* ne = native_embed()) {
            ne->gather_dev(tok, n, out, cs);
        } else {
            const auto* codes = (const uint8_t*) we->data;
            const auto* scales = (const float*) (codes + we->codes_bytes);
            const auto* offsets = we->has_offset ? (const float*) (codes + we->codes_bytes + we->scales_bytes) : nullptr;
            kernels::embedding_gather_dev(codes, scales, offsets, tok, n, we->ne0, we->code_bits, we->code_bias, we->group_elems,
                                          (uint64_t) (we->ne0 / (8 / we->code_bits)), (uint64_t) (we->ne0 / we->group_elems), out, cs);
        }
    };
    std::vector<int32_t> vocab;
    if (!vocab_file.empty()) {
        if (std::FILE* vf = std::fopen(vocab_file.c_str(), "rb")) {
            std::fseek(vf, 0, SEEK_END);
            const long size = std::ftell(vf);
            std::fseek(vf, 0, SEEK_SET);
            if (size >= 4 && size % 4 == 0) {
                vocab.resize((size_t) size / 4);
                if (std::fread(vocab.data(), 4, vocab.size(), vf) != vocab.size()) vocab.clear();
            }
            std::fclose(vf);
        }
    }
    const void* hw = head->weights();
    const int64_t row_bytes = (int64_t) head->row_bytes();
    int64_t head_rows = row_bytes > 0 ? (int64_t) (head->weight_bytes() / (uint64_t) row_bytes) : 0;
    if (!vocab.empty()) {
        int32_t* dv = nullptr;
        if (cudaMalloc((void**) &dv, vocab.size() * 4) != cudaSuccess || cudaMalloc((void**) &dhead_, (size_t) ((int64_t) vocab.size() * row_bytes)) != cudaSuccess) {
            if (dv) cudaFree(dv);
            err = "dflash2: the draft vocabulary's head rows do not fit";
            return false;
        }
        cudaMemcpy(dv, vocab.data(), vocab.size() * 4, cudaMemcpyHostToDevice);
        kernels::gather_rows((const uint8_t*) hw, row_bytes, dv, (int64_t) vocab.size(), dhead_, nullptr);
        cudaDeviceSynchronize();
        cudaFree(dv);
        vram_ += (uint64_t) ((int64_t) vocab.size() * row_bytes);
        hw = dhead_;
        head_rows = (int64_t) vocab.size();
    }
    if (!bind_raw(embed, hw, head->type(), head_rows, vocab, err)) return false;
    std::fprintf(stderr, "strata dflash2: %lld-token draft head (%.1f MiB), drafter %.0f MiB of VRAM\n", (long long) n_dvocab_,
                 (double) (n_dvocab_ * row_bytes) / 1048576.0, (double) vram_ / 1048576.0);
    return true;
}

void Dflash2Drafter::gemm(std::initializer_list<const char*> names, std::initializer_list<float*> outs, int ncols, cudaStream_t cs, const uint8_t* xq, const float* xd) {
    kernels::Df2Mat mats[3];
    int n = 0, n_in = 0;
    auto o = outs.begin();
    for (const char* name : names) {
        const Tensor& t = find(name);
        mats[n] = kernels::Df2Mat{blob_ + t.off, *o++, (int) t.rows};
        if (n == 0) n_in = (int) t.cols;
        ++n;
    }
    kernels::df2_gemm(mats, n, xq, xd, n_in, ncols, cs);
}

bool Dflash2Drafter::record_context(const float* taps, int n, cudaStream_t cs, std::string& err) {
    using namespace kernels;
    if (skip_part("ctx")) return true;
    try {
        const int tap_n = (int) taps_.size() * N_;
        df2_rope_table(d_meta_ + 0, n, theta_, rt_ctx_, cs);
        stamp("df2_rope_table");
        df2_quant_rows(taps, tap_n, n, xqc_, xdc_, cs);
        stamp("df2_quant_rows");
        gemm({"fc.weight"}, {fcy_}, n, cs, xqc_, xdc_);
        stamp("gemm:fc");
        df2_rms_quant(fcy_, f32("hidden_norm.weight"), nullptr, n, N_, eps_, xqc_, xdc_, cs);
        stamp("df2_rms_quant");
        for (int l = 0; l < L_; ++l) {
            const std::string p = "layers." + std::to_string(l) + ".";
            const std::string kp = p + "self_attn.k_proj.weight", vp = p + "self_attn.v_proj.weight";
            gemm({kp.c_str(), vp.c_str()}, {kc_, vc_}, n, cs, xqc_, xdc_);
            stamp("gemm:kp");
            df2_kv_ring(kc_, vc_, f32(p + "self_attn.k_norm.weight"), d_meta_ + 0, rt_ctx_, kring_[l], vring_[l], kRing, n, NKV_, eps_, cs);
            stamp("df2_kv_ring");
            if (fork_) cudaEventRecord(ev_ctx_[l], cs);   // layer l's keys and values are in the ring
        }
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

bool Dflash2Drafter::record_block(cudaStream_t cs, std::string& err) {
    using namespace kernels;
    try {
        const int B = kRows;
        df2_rope_table(d_meta_ + 8, B, theta_, rt_blk_, cs);
        stamp("df2_rope_table");
        embed_(d_meta_ + 24, B, h_, cs);
        stamp("embed_");
        df2_rms_quant(h_, f32("layers.0.input_layernorm.weight"), xn_, B, N_, eps_, xqp_, xdp_, cs);
        stamp("df2_rms_quant");
        for (int l = 0; l < L_; ++l) {
            const std::string p = "layers." + std::to_string(l) + ".";
            const float* base_a = f32(p + "attention_conv.base_kernel");
            const float* base_m = f32(p + "mlp_conv.base_kernel");
            const std::string akp = p + "attention_conv.kernel_projection.weight", mkp = p + "mlp_conv.kernel_projection.weight";
            const std::string qp = p + "self_attn.q_proj.weight", kp = p + "self_attn.k_proj.weight", vp = p + "self_attn.v_proj.weight";
            const std::string gp = p + "mlp.gate_proj.weight", up = p + "mlp.up_proj.weight";
            if (!skip_part("tiny")) gemm({akp.c_str()}, {dyn_}, B, cs, xqp_, xdp_);
            stamp("gemm:akp");
            if (!skip_part("small")) df2_conv_quant(xn_, dyn_, 640, base_a, B, N_, group_, xqp_, xdp_, cs);
            stamp("df2_conv_quant");
            if (!skip_part("qkvo")) gemm({qp.c_str(), kp.c_str(), vp.c_str()}, {q_, k_, v_}, B, cs, xqp_, xdp_);
            stamp("gemm:qp");
            if (!skip_part("small")) df2_norm_rope_qk(q_, k_, f32(p + "self_attn.q_norm.weight"), f32(p + "self_attn.k_norm.weight"), rt_blk_, qn_, kblk_, B, NH_, NKV_, eps_, cs);
            stamp("df2_norm_rope_qk");
            if (fork_) cudaStreamWaitEvent(cs, ev_ctx_[l], 0);   // the context update has written this layer's ring
            if (!skip_part("attn")) df2_attention(qn_, kring_[l], vring_[l], kblk_, v_, d_meta_ + 17, kRing, window_, part_, xqp_, xdp_, cs);
            stamp("df2_attention");
            if (!skip_part("qkvo")) gemm({(p + "self_attn.o_proj.weight").c_str()}, {o_}, B, cs, xqp_, xdp_);
            stamp("gemm:o_proj");
            if (!skip_part("small")) df2_conv_rms_quant(o_, dyn_ + 320, 640, base_a + 2 * N_, h_, f32(p + "post_attention_layernorm.weight"), xn_, B, N_, group_, eps_, xqp_, xdp_, cs);
            stamp("df2_conv_rms_quant");
            if (!skip_part("tiny")) gemm({mkp.c_str()}, {dyn_}, B, cs, xqp_, xdp_);
            stamp("gemm:mkp");
            if (!skip_part("small")) df2_conv_quant(xn_, dyn_, 640, base_m, B, N_, group_, xqp_, xdp_, cs);
            stamp("df2_conv_quant");
            if (!skip_part("mlp")) gemm({gp.c_str(), up.c_str()}, {gate_, up_}, B, cs, xqp_, xdp_);
            stamp("gemm:gp");
            if (!skip_part("small")) df2_swiglu_quant(gate_, up_, B, I_, xqp_, xdp_, cs);
            stamp("df2_swiglu_quant");
            if (!skip_part("mlp")) gemm({(p + "mlp.down_proj.weight").c_str()}, {mlp_}, B, cs, xqp_, xdp_);
            stamp("gemm:mlp.down_proj");
            // the next layer's input norm, or after the last layer the final norm (-> fn_, the rows the head and the selector read)
            const std::string next_gamma = l + 1 < L_ ? "layers." + std::to_string(l + 1) + ".input_layernorm.weight" : std::string("norm.weight");
            if (!skip_part("small")) df2_conv_rms_quant(mlp_, dyn_ + 320, 640, base_m + 2 * N_, h_, f32(next_gamma), l + 1 < L_ ? xn_ : fn_, B, N_, group_, eps_, xqp_, xdp_, cs);
            stamp("df2_conv_rms_quant");
        }
        native_quantize_q8_1(fn_ + N_, xq_, N_, kDrafts, cs);   // rows 1..7, the head's format
        df2_quant_rows(fn_ + N_, N_, kDrafts, xqp_, xdp_, cs);   // and the selector projection's
        if (!skip_part("head")) native_mmvq(head_type_, head_w_, xq_, logits_, N_, (int) n_dvocab_, kDrafts, cs);
        stamp("native_mmvq");
        df2_topk(logits_, (int) n_dvocab_, dvocab_, kDrafts, topscratch_, topval_, toptok_, cs);
        stamp("df2_topk");
        gemm({"candidate_selector.hidden_projection.weight"}, {hsel_}, kDrafts, cs, xqp_, xdp_);
        stamp("gemm:candidate_selector.hidden_projection");
        df2_select(hsel_, topval_, toptok_, reinterpret_cast<const uint16_t*>(blob_ + find("candidate_selector.predecessor_codebook").off),
                   reinterpret_cast<const uint16_t*>(blob_ + find("candidate_selector.successor_codebook").off), rank_, kDrafts, d_meta_ + 16, out_tok_,
                   out_prob_, cs);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

bool Dflash2Drafter::capture(int n, const float* taps, std::string& err) {
    if (exec_[n]) return true;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "dflash2: begin capture"; return false; }
    kernels::copy_i32_from_mapped(d_meta_, m_meta_, 32, cs_);
    // the context update (fc, the keys and values) and the block's front run side by side: they meet before each layer's attention
    // measured: -0.3 ms a round on an RTX 3060, +0.0..0.7 on a Strix Halo (gfx1151), so the HIP builds keep one stream (STRATA_DF2_FORK=1 / 0 overrides)
#if defined(STRATA_USE_HIP)
    static const bool fork_default = false;
#else
    static const bool fork_default = true;
#endif
    static const bool fork_on = [] { const char* e = std::getenv("STRATA_DF2_FORK"); return e && e[0] ? e[0] == '1' : fork_default; }();
    fork_ = fork_on && prof_ == nullptr && cs2_ != nullptr;
    if (fork_) {
        cudaEventRecord(ev_fork_, cs_);
        cudaStreamWaitEvent(cs2_, ev_fork_, 0);
    }
    bool ok = record_context(taps, n, fork_ ? cs2_ : cs_, err);
    if (fork_) cudaEventRecord(ev_join_, cs2_);
    if (ok) ok = record_block(cs_, err);
    if (fork_) cudaStreamWaitEvent(cs_, ev_join_, 0);
    fork_ = false;
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok) { if (graph) cudaGraphDestroy(graph); return false; }
    if (ce != cudaSuccess || cudaGraphInstantiate(&exec_[n], graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("dflash2: capture: ") + cudaGetErrorString(ce);
        return false;
    }
    cudaGraphDestroy(graph);
    cudaGraphUpload(exec_[n], cs_);
    cudaStreamSynchronize(cs_);
    return true;
}

bool Dflash2Drafter::prepare(const float* taps, std::string& err) {
    if (!bound_) { err = "dflash2: not bound"; return false; }
    if (prof_ != nullptr) return true;   // the profile runs without graphs
    const OnDevice on_device(device_);
    const Clock::time_point t0 = Clock::now();
    for (int i = 0; i < 8; ++i) h_meta_[i] = i;
    for (int n = 1; n <= kRows; ++n)
        if (!capture(n, taps, err)) return false;
    graph_taps_ = taps;
    ms_capture = ms_since(t0);
    return true;
}

bool Dflash2Drafter::context(const float* taps, int64_t n, int64_t pos0, std::string& err) {
    const OnDevice on_device(device_);
    const int tap_n = (int) taps_.size() * N_;
    for (int64_t r0 = 0; r0 < n; r0 += kRows) {
        const int m = (int) std::min<int64_t>(kRows, n - r0);
        for (int i = 0; i < m; ++i) h_meta_[i] = (int32_t) (pos0 + r0 + i);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        kernels::copy_i32_from_mapped(d_meta_, m_meta_, 32, cs_);
        if (!record_context(taps + (size_t) r0 * (size_t) tap_n, m, cs_, err)) return false;
        if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = std::string("dflash2: ") + cudaGetErrorString(cudaGetLastError()); return false; }
    }
    have_ctx_ = std::max<int64_t>(have_ctx_, pos0 + n);
    return true;
}

bool Dflash2Drafter::draft(const float* taps, int n_new, int64_t pos0, int32_t anchor, int32_t* drafts, float* probs, std::string& err) {
    if (!bound_) { err = "dflash2: not bound"; return false; }
    if (n_new < 1 || n_new > kRows) { err = "dflash2: draft rows out of range"; return false; }
    const OnDevice on_device(device_);
    const Clock::time_point t0 = Clock::now();
    if (graph_taps_ != nullptr && graph_taps_ != taps) { err = "dflash2: the taps buffer moved after capture"; return false; }
    if (!prof_ && !exec_[n_new]) {
        // the staging must hold valid positions while the capture runs the (empty) kernels' parameter checks
        for (int i = 0; i < 8; ++i) h_meta_[i] = (int32_t) (pos0 + i);
        if (!capture(n_new, taps, err)) return false;
        graph_taps_ = taps;
    }
    const int64_t s = pos0 + n_new;
    for (int i = 0; i < 8; ++i) { h_meta_[i] = (int32_t) (pos0 + std::min(i, n_new - 1)); h_meta_[8 + i] = (int32_t) (s + i); }
    h_meta_[16] = anchor;
    h_meta_[17] = (int32_t) s;
    h_meta_[24] = anchor;
    for (int i = 1; i < 8; ++i) h_meta_[24 + i] = mask_token_;
    for (int i = 0; i < kDrafts; ++i) { ((volatile int32_t*) h_out_)[i] = -1; }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (timing_) cudaEventRecord(ev_a_, cs_);
    if (prof_) {   // the same round without its graph, an event after every stage
        cudaStreamSynchronize(cs_);
        for (cudaEvent_t e : prof_->ev) cudaEventDestroy(e);   // the stamps of context() calls are not a round
        prof_->ev.clear();
        prof_->name.clear();
        kernels::copy_i32_from_mapped(d_meta_, m_meta_, 32, cs_);
        stamp("start");
        if (!record_context(taps, n_new, cs_, err) || !record_block(cs_, err)) return false;
        prof_collect();
    } else
    if (cudaGraphLaunch(exec_[n_new], cs_) != cudaSuccess) { err = std::string("dflash2 draft: ") + cudaGetErrorString(cudaGetLastError()); return false; }
    if (timing_) cudaEventRecord(ev_b_, cs_);
    (void) cudaStreamQuery(cs_);
    if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = std::string("dflash2 draft: ") + cudaGetErrorString(cudaGetLastError()); return false; }
    for (int i = 0; i < kDrafts; ++i) {
        drafts[i] = ((volatile int32_t*) h_out_)[i];
        if (probs) probs[i] = ((volatile float*) h_out_)[16 + i];
    }
    if (timing_) {
        float ms = 0;
        if (cudaEventElapsedTime(&ms, ev_a_, ev_b_) == cudaSuccess) ms_gpu += ms;
    }
    have_ctx_ = std::max<int64_t>(have_ctx_, s);
    ms_draft += ms_since(t0);
    ++rounds;
    return true;
}

bool Dflash2Drafter::debug_rows(std::vector<float>& rows, std::string& err) {
    rows.resize((size_t) kRows * (size_t) N_);
    if (cudaMemcpy(rows.data(), fn_, rows.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess) { err = "dflash2: debug copy"; return false; }
    return true;
}

}  // namespace strata::core
