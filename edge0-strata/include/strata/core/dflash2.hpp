// include/strata/core/dflash2.hpp - the DFlash2 drafter for Qwen3.8-Flash-Next on the GPU (CUDA and HIP).
//
// incoai/Qwen3.8-Flash-Next-DFlash2 (Apache-2.0) is a block drafter: 5 transformer layers of hidden 2560 that read the target's hidden
// states (the mean of the 4 hyper-connection streams after layers 5, 15, 25, 35, 45) as context and, from the last accepted token plus
// 7 mask tokens, produce 7 draft tokens in one forward pass.  A candidate selector (the head's top-16 per row, a rank-256 bilinear
// score between the previous pick and each candidate) picks the path left to right.  tools/dflash2_pack.py packs it as Q8_0.
//
//   context    taps (5 x 2560 per accepted row) -> fc (12800 -> 2560) -> hidden_norm -> per layer K (norm + rope) and V -> ring slot pos % R
//   block      8 rows: embed(anchor), embed(mask) x 7; per layer
//                rms -> dynamic 2-tap conv (prepare) -> q, k, v -> q/k norm + rope -> attention over the ring + the block's own keys
//                -> o -> conv (finish) + residual -> rms -> conv (prepare) -> gate/up -> silu*up -> down -> conv (finish) + residual
//   selector   final norm -> head over the draft vocabulary (rows 1..7) -> top-16 -> walk
//
// The ring holds only COMMITTED rows (an accepted window row's taps arrive after the verify window), and the block's keys stay outside
// it, so a rejected draft leaves nothing to roll back.  One CUDA graph per context-row count n = 1..8 holds the context update and the
// block; the host stages the positions and the anchor and waits for the 7 drafts.
#pragma once

#include "strata/core/session.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <string>
#include <vector>

namespace strata::core {

class NativeHead;

class Dflash2Drafter {
public:
    static constexpr int kRows = 8;           ///< block rows: the anchor + 7 masks
    static constexpr int kDrafts = kRows - 1;
    static constexpr int kRing = 2064;        ///< K/V slots per layer: the 2048 window + slack for rows written ahead of a rejection

    Dflash2Drafter() = default;
    ~Dflash2Drafter();
    Dflash2Drafter(const Dflash2Drafter&) = delete;
    Dflash2Drafter& operator=(const Dflash2Drafter&) = delete;

    /// Reads the pack (tools/dflash2_pack.py) to the GPU `device` and allocates the ring and the buffers (about 0.8 GB).
    bool load(const std::string& pack_path, int device, std::string& err);
    /// The target's embedding and head (and the draft vocabulary file, "" or missing: the whole head).
    bool bind(const WeightTable& wt, const NativeHead* head, const std::string& vocab_file, std::string& err);
    /// Tests: an embedding callback and a head given as raw rows of `head_type` (a native_mmvq type).
    using EmbedFn = std::function<void(const int32_t* tokens_dev, int n, float* out, cudaStream_t)>;
    bool bind_raw(EmbedFn embed, const void* head_w, int head_type, int64_t head_rows, const std::vector<int32_t>& vocab, std::string& err);

    /// A new request: the ring is empty again.
    void reset() { have_ctx_ = 0; }
    /// Context rows without a draft: `n` rows of taps (device, [n][5 * n_embd]) for positions pos0 ... pos0 + n - 1.
    bool context(const float* taps_dev, int64_t n, int64_t pos0, std::string& err);
    /// One round: the new context rows (n_new in 1..8: the committed window rows, taps device [n_new][5 * n_embd], positions pos0 ...),
    /// then the block for `anchor` (the last committed token, at pos0 + n_new).  `drafts` / `probs` get kDrafts entries (the
    /// selector's pick and its probability).  The taps pointer must be the same device buffer on every call (graphs are captured on it).
    bool draft(const float* taps_dev, int n_new, int64_t pos0, int32_t anchor, int32_t* drafts, float* probs, std::string& err);

    /// Captures the eight round graphs (n_new = 1..8) on `taps` now, so no round pays for a capture.
    bool prepare(const float* taps_dev, std::string& err);

    uint64_t vram_bytes() const { return vram_; }
    int64_t n_draft_vocab() const { return n_dvocab_; }
    int n_embd() const { return N_; }
    const std::vector<int>& tap_layers() const { return taps_; }
    cudaStream_t stream() const { return cs_; }
    /// Diagnostics (STRATA_DF2_DUMP): the final normalized block rows of the last draft(), host copy.
    bool debug_rows(std::vector<float>& rows, std::string& err);

    double ms_draft = 0, ms_gpu = 0, ms_ctx = 0, ms_capture = 0;
    int64_t rounds = 0;

private:
    struct Tensor { std::string name, dtype; int64_t rows = 0, cols = 0; uint64_t off = 0, bytes = 0; };
    const Tensor& find(const std::string& name) const;
    const void* q8(const std::string& name) const { return blob_ + find(name).off; }
    const float* f32(const std::string& name) const { return reinterpret_cast<const float*>(blob_ + find(name).off); }
    /// planar Q8 matrices (1..3 of them, same input width) times the planar activations in xqp_ / xdp_ (ncols columns)
    void gemm(std::initializer_list<const char*> names, std::initializer_list<float*> outs, int ncols, cudaStream_t cs, const uint8_t* xq, const float* xd);
    bool record_context(const float* taps, int n, cudaStream_t cs, std::string& err);
    bool record_block(cudaStream_t cs, std::string& err);
    bool capture(int n, const float* taps, std::string& err);
    bool alloc(std::string& err);
    struct Prof;
    Prof* prof_ = nullptr;
    void stamp(const char* name);
    void prof_collect();
    void prof_report() const;
    void prof_free();

    int device_ = -1;
    cudaStream_t cs_ = nullptr;
    std::vector<Tensor> tensors_;
    uint8_t* blob_ = nullptr;
    uint64_t vram_ = 0;
    int N_ = 2560, L_ = 5, NH_ = 32, NKV_ = 8, HD_ = 128, I_ = 7680, window_ = 2048, group_ = 16, rank_ = 256, mask_token_ = 0;
    float theta_ = 10000.0f, eps_ = 1e-6f;
    std::vector<int> taps_;
    int64_t vocab_ = 0;

    // target pieces
    EmbedFn embed_;
    const void* head_w_ = nullptr;
    int head_type_ = -1;
    int64_t n_dvocab_ = 0;
    int32_t* dvocab_ = nullptr;
    uint8_t* dhead_ = nullptr;      ///< the draft vocabulary's head rows (owned), or null with head_w_ the whole head
    bool bound_ = false;

    // device buffers
    int32_t* d_meta_ = nullptr;     ///< [0..7] context positions, [8..15] block positions, [16] anchor token, [17] anchor position, [24..31] block tokens
    int32_t* h_meta_ = nullptr;     ///< host staging of d_meta_ (mapped; m_meta_ is its device alias)
    int32_t* m_meta_ = nullptr;
    int32_t* h_out_ = nullptr;      ///< pinned: 7 draft tokens then 7 probabilities
    float* hp_out_ = nullptr;
    float *kring_[8] = {}, *vring_[8] = {};
    float *rt_ctx_ = nullptr, *rt_blk_ = nullptr;
    uint8_t* xq_ = nullptr;         ///< the head's q8_1 activations
    uint8_t* xqp_ = nullptr;        ///< planar activations of the drafter's own GEMMs (kernels/dflash2.hpp)
    float* xdp_ = nullptr;
    uint8_t* xqc_ = nullptr;        ///< the same for the context update, which runs beside the block
    float* xdc_ = nullptr;
    cudaStream_t cs2_ = nullptr;    ///< the context update's branch of the round graph
    cudaEvent_t ev_fork_ = nullptr, ev_join_ = nullptr, ev_ctx_[8] = {};
    bool fork_ = false;             ///< recording the forked graph: layer l's attention waits for ev_ctx_[l]
    float *fcy_ = nullptr, *hctx_ = nullptr, *kc_ = nullptr, *vc_ = nullptr;
    float *h_ = nullptr, *xn_ = nullptr, *xc_ = nullptr, *dyn_ = nullptr, *q_ = nullptr, *qn_ = nullptr, *k_ = nullptr, *v_ = nullptr, *kblk_ = nullptr;
    float *ao_ = nullptr, *o_ = nullptr, *gate_ = nullptr, *up_ = nullptr, *mlp_ = nullptr, *part_ = nullptr, *fn_ = nullptr;
    float *logits_ = nullptr, *topval_ = nullptr, *hsel_ = nullptr, *out_prob_ = nullptr;
    int32_t *toptok_ = nullptr, *out_tok_ = nullptr;
    uint32_t* topscratch_ = nullptr;
    cudaGraphExec_t exec_[kRows + 1] = {};
    const float* graph_taps_ = nullptr;
    int64_t have_ctx_ = 0;
    cudaEvent_t ev_a_ = nullptr, ev_b_ = nullptr;
    bool timing_ = false;
};

}  // namespace strata::core
