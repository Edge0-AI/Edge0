// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_vision.cuh - DeepSeek-V4.1's vision tower on one GPU (DeepSeek's inference/vision.py):
//   patches (14 x 14 RGB, scaled to [-1, 1]) -> Linear(588 -> 1024)
//   32 blocks: x += attn(rmsnorm(x)) (16 heads of 64, full bidirectional attention over the image, a 2D rotary on q
//              and k: the first half of each head's frequencies the patch row, the second its column);
//              x += w2(silu(gate) * up) with gate, up = w1(rmsnorm(x))
//   rmsnorm, then the aligner: each 3 x 3 block of patches (zero-padded at the right and bottom) as one 9216 row,
//   Linear(9216 -> 5120), GELU, Linear(5120 -> 5120): one LLM embedding row per block, in reading order.
// The BF16 weights live in pinned host memory and stream to the GPU a block at a time (two slots, the next block's
// copy under this one's work), so the tower costs the model ~350 MB of the GPU's memory, not ~1 GB; the GEMMs are
// cuBLAS (BF16 in, FP32 accumulation), the residual stream FP32.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace strata::ds4 {

struct VisionGeo {
    int layers = 0, dim = 1024, heads = 16, inter = 2816, patch = 14, merge = 3, out = 5120;
    float theta = 10000.0f, eps = 1e-6f;
    int max_tokens = 1024;   // an image's span in the prompt at most: n_h * (n_w + 1) + 2 tokens
};

class VisionTower {
public:
    /// read(name, shape, dst): the tensor's shape, and (dst not null) its BF16 data into dst; false: not there
    using Reader = std::function<bool(const std::string& name, std::vector<int64_t>& shape, void* dst)>;
    VisionTower() = default;
    VisionTower(const VisionTower&) = delete;
    VisionTower& operator=(const VisionTower&) = delete;
    ~VisionTower();
    bool load(const VisionGeo& g, int device, const Reader& read, std::string& err);
    /// Frees everything load took (a load that failed half way too); loaded() is false after it.
    void release();
    bool loaded() const { return dev_ >= 0; }
    int device() const { return dev_; }
    size_t device_bytes() const { return dev_bytes_; }
    /// The patches an image may have (its aligner grid within max_tokens).
    int max_patches() const { return max_n_; }
    /// rgb: height x width x 3 bytes, both sides multiples of the patch size -> the aligner's rows, BF16
    /// [n_h * n_w][out] in reading order (n_h x n_w: the patch grid over 3, rounded up).
    bool encode(const uint8_t* rgb, int width, int height, std::vector<uint16_t>& rows, int& n_h, int& n_w,
                std::string& err);

private:
    struct Piece {             // a weight the encode streams in: host bits, its size
        const uint16_t* host = nullptr;
        size_t n = 0;
    };
    VisionGeo g_;
    int dev_ = -1, at_ = -1, max_n_ = 0, qb_ = 0, rb_ = 0;   // (at_: the device its memory is on)
    size_t dev_bytes_ = 0, slot_n_ = 0;
    std::vector<void*> dallocs_;
    uint16_t* host_ = nullptr;          // pinned: every block's [wqkv, wo, w1, w2], then aligner w1, w2
    size_t block_n_ = 0;                // a block's weights (BF16 values)
    size_t al1_off_ = 0, al2_off_ = 0;  // the aligner's weights in host_
    // resident (small): the patch embedding, the norms and biases
    uint16_t *pe_w_ = nullptr, *pe_b_ = nullptr, *norm_ = nullptr, *al1_b_ = nullptr, *al2_b_ = nullptr;
    std::vector<uint16_t*> n1_, n2_, bqkv_, bo_;
    // work buffers (max_n_ patches)
    uint8_t* rgb_ = nullptr;
    uint16_t *patch_ = nullptr, *xn_ = nullptr, *qkv_ = nullptr, *o_ = nullptr, *mid_ = nullptr, *p_ = nullptr,
             *un_ = nullptr, *a16_ = nullptr, *outb_ = nullptr;
    float *x_ = nullptr, *y_ = nullptr, *s_ = nullptr, *inv_freq_ = nullptr;
    uint16_t* slot_[2] = {nullptr, nullptr};
    void* blas_ = nullptr;              // cublasHandle_t
    uint8_t* ws_ = nullptr;
    void *st_ = nullptr, *cs_ = nullptr;   // cudaStream_t: the work, the weight copies
    void *copied_[2] = {nullptr, nullptr}, *used_[2] = {nullptr, nullptr};   // cudaEvent_t per slot
    template <class T>
    bool dalloc(T*& p, size_t n, std::string& err);
};

}  // namespace strata::ds4
