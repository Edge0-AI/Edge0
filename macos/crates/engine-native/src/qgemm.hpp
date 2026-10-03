// Grouped quantized matmul via mlx gather_qmm, with an optional custom MSL path.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <mlx/mlx.h>

#include "safio.hpp"

namespace e0n {

enum class GemmPath { kMlxGatherQmm, kMslCustom };

struct ExpertMapDims {
  int64_t E;
  int hid;
  int ffn;
};
bool dims_of(const ModelFiles &files, int layer, ExpertMapDims &out, std::string &err);

class QuantLayer {
 public:
  static std::unique_ptr<QuantLayer> load(const ModelFiles &files, const ExpertMapDims &dims,
                                          int layer, GemmPath path, std::string &err);
  ~QuantLayer();

  int E() const { return e_; }
  int HID() const { return hid_; }
  int FFN() const { return ffn_; }

  // swiglu + down + residual; x is host [HID] float, updated in place.
  // layer_indices = per-layer expert sets (one proxy compute per layer).
  bool moe_proxy(std::vector<float> &x, const std::vector<std::vector<uint32_t>> &layer_indices,
                 std::string &err);

 private:
  struct Impl;
  explicit QuantLayer(std::unique_ptr<Impl> i);
  std::unique_ptr<Impl> impl_;
  int e_ = 0, hid_ = 0, ffn_ = 0;
};

std::vector<double> fp64_qmm_ref(const float *x, int T, int I, const uint32_t *W,
                                 const uint16_t *S, const uint16_t *B, int O, int G,
                                 const std::vector<int64_t> &sel);

bool gather_qmm_mlx(const uint32_t *W, const uint16_t *S, const uint16_t *B, int E_all, int O,
                    int I, int G, const float *x, int T, const std::vector<int64_t> &sel,
                    std::vector<float> &y, std::string &err);

// Lazy graph form: x and the return are mlx arrays; this call does not sync.
// Host weight pointers use a no-op deleter; the caller must keep staging alive until
// the layer-end eval. Output shape {sel.size(), T, O}.
mlx::core::array gather_qmm_graph(const uint32_t *W, const uint16_t *S, const uint16_t *B,
                                   int E_all, int O, int I, int G, const mlx::core::array &xa,
                                   int T, const std::vector<int64_t> &sel);

double max_rel(const std::vector<float> &got, const std::vector<double> &ref);

}  // namespace e0n
