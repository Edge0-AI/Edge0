// include/strata/core/latent_dense.hpp - the D/U Q8_0 matrices for the latent MoE.
//
// One file, one fixed layout:
//   layer l: D_q8 [G*ELL, H] (Q8_0 rows) then U_q8 [G*H, ELL]
// The file is mmap'd; the verify path uploads a layer's two matrices on demand.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::core {

class LatentDense {
public:
    LatentDense() = default;
    ~LatentDense();
    LatentDense(const LatentDense&) = delete;
    LatentDense& operator=(const LatentDense&) = delete;

    bool open(const std::string& path, int64_t n_layers, int64_t n_groups, int64_t n_embd, int64_t latent_dim,
              std::string& err);
    void close();

    bool mapped() const { return base_ != nullptr; }
    int64_t layers() const { return n_layers_; }
    int64_t groups() const { return n_groups_; }
    int64_t embd() const { return n_embd_; }
    int64_t latent() const { return latent_dim_; }
    // Q4DU: the dense pack's weight format, 8 (Q8_0, 34 B/block - the historical default) or
    // 4 (Q4_0, 18 B/block: {f16 d; 16 B nibbles}, element j in qs[j]'s low nibble, j+16 in its
    // high one, value = (code - 8) * d).  Derived once at open() from STRATA_LATENT_DENSE_FMT
    // and cross-checked against the file's size; every offset below flows from it.
    int fmt() const { return fmt_; }
    size_t row_b() const { return fmt_ == 4 ? 18 : 34; }
    size_t d_row_bytes() const { return (size_t) (n_embd_ / 32) * row_b(); }
    size_t u_row_bytes() const { return (size_t) (latent_dim_ / 32) * row_b(); }
    size_t d_bytes() const { return (size_t) n_groups_ * (size_t) latent_dim_ * d_row_bytes(); }
    size_t u_bytes() const { return (size_t) n_groups_ * (size_t) n_embd_ * u_row_bytes(); }
    uint64_t bytes() const { return bytes_; }

    const uint8_t* d(int64_t layer) const;
    const uint8_t* u(int64_t layer) const;

private:
    uint8_t* base_ = nullptr;
    size_t size_ = 0;
    uint64_t bytes_ = 0;
    void* map_ = nullptr;   // WIN32: the file-mapping handle; Linux: unused
    int64_t n_layers_ = 0, n_groups_ = 0, n_embd_ = 0, latent_dim_ = 0;
    int fmt_ = 8;
};


/// A pack (experts.bin) mapped for the direct-A smoke path.
class LatentA {
public:
    LatentA() = default;
    ~LatentA();
    LatentA(const LatentA&) = delete;
    LatentA& operator=(const LatentA&) = delete;
    bool open(const std::string& path, int64_t n_layers, int64_t n_expert, int64_t blob_bytes, std::string& err);
    void close();
    const uint8_t* blob(int64_t layer, int64_t expert) const;
    int64_t blob_bytes() const { return blob_bytes_; }
private:
    uint8_t* base_ = nullptr;
    size_t size_ = 0;
    void* map_ = nullptr;   // WIN32: the file-mapping handle; Linux: unused
    int64_t n_layers_ = 0, n_expert_ = 0, blob_bytes_ = 0;
};

}  // namespace strata::core
