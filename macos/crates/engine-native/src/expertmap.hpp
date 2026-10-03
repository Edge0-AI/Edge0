// (layer, expert) → contiguous row slices inside fused tensors. 35b: shape[0]=E;
// 8b: E folded into the row dim. Nine regions: {gate,up,down}_proj × {weight,scales,biases}.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "safio.hpp"

namespace e0n {

struct Region {
  uint64_t off, len;
  std::string shard;
  std::string name;   // "gate_proj.weight" etc., diagnostics only
};

class ExpertMap {
 public:
  static bool build(const ModelFiles& files, ExpertMap& out, std::string& err);

  int num_layers() const { return num_layers_; }
  int64_t num_experts() const { return num_experts_; }

  std::vector<Region> regions(int layer, int64_t expert) const;
  uint64_t expert_bytes(int layer, int64_t expert) const;

  const std::string& layer_prefix(int layer) const;

 private:
  struct Slice {
    uint64_t base, stride;
    std::string shard, name;
  };
  int num_layers_ = 0;
  int64_t num_experts_ = 0;
  std::vector<std::string> prefixes_;
  std::vector<std::map<std::string, Slice>> layers_;
};

}  // namespace e0n
