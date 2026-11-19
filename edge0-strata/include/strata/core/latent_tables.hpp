// include/strata/core/latent_tables.hpp - per-device, per-layer U + labels upload for the latent MoE.
//
// The primary always runs the D stage (encoded x -> xl f32 rows, landing in mapped host memory). A peer or
// helper device that computes latent expert rows needs two pieces of the session state on ITS OWN device:
//   * the layer's U table (Q8_0/Q4_0, [G*H, ELL]) - the only large part;
//   * the group labels (labels.bin, one int32 per (layer, expert)) - tiny, uploaded whole per device.
// D is never needed off-device: the D stage runs on the primary and its f32 xl rows are mapped.
//
// This is the module that replaces the shared single-pair globals (latent_shared_d/u in latent_share.hpp) for
// multi-device use. It is independent of peer/helper/split wiring: it only answers "make sure device N can
// address layer L's U and labels", lazily.
#pragma once

#include "strata/core/latent_dense.hpp"

#include <cstdint>
#include <map>
#include <string>

namespace strata::core {

class LatentTables {
public:
    LatentTables() = default;
    ~LatentTables();
    LatentTables(const LatentTables&) = delete;
    LatentTables& operator=(const LatentTables&) = delete;

    /// Attach the session-level host state. Does not allocate or upload; open()/labels must outlive this object.
    /// `dense` is the mmap'd D/U file (already opened); `host_labels` points at `n_layers * E` int32 group labels;
    /// `n_embd` and `latent` are the model hidden and ELL so rows/offsets can be sanity-checked.
    void bind(LatentDense* dense, const int32_t* host_labels, int64_t n_layers, int64_t E,
              int64_t n_embd, int64_t latent);

    /// Release every device allocation.
    void close();

    /// Make sure `device` holds `layer`'s U (device) and the whole labels table (device). The current CUDA device
    /// is saved and restored. Returns false + `err` on a CUDA failure.
    bool ensure(int device, int64_t layer, std::string& err);

    int fmt() const { return dense_ ? dense_->fmt() : 8; }

    /// Device address of `layer`'s U on `device`, or null until ensure(device, layer) succeeded. Caller keeps the
    /// CUDA device set to `device` while using it.
    const uint8_t* u_dev(int device, int64_t layer) const;

    /// Device address of the whole labels table ([n_layers * E] int32) on `device`, or null if never uploaded there.
    /// Slice a layer with `labels_dev(device) + layer * E`.
    const int32_t* labels_dev(int device) const;

private:
    struct Dev {
        int32_t* labels = nullptr;   // [n_layers * E]
        uint8_t* u = nullptr;        // one layer's U (dense->u_bytes())
        int64_t u_layer = -1;
    };
    Dev* dev(int device);

    LatentDense* dense_ = nullptr;
    const int32_t* host_labels_ = nullptr;
    int64_t n_layers_ = 0, E_ = 0, n_embd_ = 0, latent_ = 0;
    std::map<int, Dev> devs_;
};

}  // namespace strata::core