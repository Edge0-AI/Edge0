// Unmerged parallel LoRA: y = base(x) + (scale * ((x.f16 @ A^T) @ B^T)).astype(y.dtype), scale = alpha/r.
#pragma once
#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <mlx/mlx.h>

#include "fwd_common.hpp"

namespace e0n {
namespace lora {

enum Slot35 {
  S35_IN_A = 0, S35_IN_B, S35_IN_QKV, S35_IN_Z, S35_OUT_PROJ,  // linear_attn (30 layers)
  S35_Q, S35_K, S35_V, S35_O,                                   // self_attn (10 layers)
  S35_SH_GATE, S35_SH_UP, S35_SH_DOWN,                          // mlp.shared_expert (40 layers)
  S35_NSLOT = 12
};
enum Slot8b {
  S8B_Q = 0, S8B_K, S8B_V, S8B_O, S8B_B, S8B_F, S8B_G,  // KDA attention (18 layers)
  S8B_M_QB, S8B_M_KVB, S8B_M_DENSE, S8B_M_G,             // MLA attention (6 layers; g by family)
  S8B_D_GATE, S8B_D_UP, S8B_D_DOWN,                      // L0 dense mlp only
  S8B_NSLOT = 14
};

struct Meta {
  int64_t r = 0, alpha = 0, k = 0;
  float scale = 0.f;
  bool ok = false;  // metadata complete and self-consistent
};

class Bundle {
 public:
  // Single safetensors file (self-parsed header + __metadata__).
  // num_layers / family predicate: (li+1)%4==0 is full/MLA on both tiers.
  bool load_file(const std::string& path, bool is_8b, int num_layers, std::string& err);
  // In-memory tensors for tests; same core as load_file.
  bool load_raw(std::map<std::string, std::pair<std::string, std::vector<uint8_t>>> tensors,
                const std::string& meta_json, bool is_8b, int num_layers, std::string& err);

  bool empty() const { return pairs_.empty(); }
  const Meta& meta() const { return meta_; }
  int n_bound() const { return static_cast<int>(pairs_.size()); }
  int64_t bytes() const { return bytes_; }
  const std::vector<std::string>& bound_keys() const { return keys_sorted_; }
  int n_hits() const;
  void reset_hits() const { std::fill(hit_.begin(), hit_.end(), (char)0); }

  // y = base.apply(x); add delta if this (layer, slot) has a pair.
  // In/out mismatch vs QuantLin is a wiring bug and throws.
  mlx::core::array dressed(const QuantLin& lin, const mlx::core::array& x, int layer,
                           int slot) const;

  bool has_pair(int layer, int slot) const;
  bool pair_raw(int layer, int slot, std::vector<uint8_t>& a, std::vector<uint8_t>& b,
                int64_t& r, int64_t& in, int64_t& out) const;
  // Test-only error shapes: 0 = scale uses α (32); 1 = delta replaces base; 2 = f32 path.
  bool dressed_variant(const QuantLin& lin, const mlx::core::array& x, int layer, int slot,
                       int which, mlx::core::array& out, std::string& err) const;
  // Test-only: export the f16 delta bit pattern.
  bool delta_bits(int layer, int slot, const mlx::core::array& x, std::vector<uint16_t>& bits,
                  std::string& err) const;

 private:
  struct Pair {
    int layer = -1, slot = -1;
    int64_t r = 0, in = 0, out = 0;
    std::vector<uint8_t> ab, bb;  // stable host buffers (unique_ptr-owned, no realloc)
    mlx::core::array A{0.f}, B{0.f};
    std::string key;
  };
  const Pair* find_pair(int layer, int slot) const {
    if (layer < 0 || layer >= static_cast<int>(slots_.size()) || slot < 0 || slot >= ns_)
      return nullptr;
    int i = slots_[layer][slot];
    return i < 0 ? nullptr : pairs_[i].get();
  }

  Meta meta_;
  std::vector<std::unique_ptr<Pair>> pairs_;
  std::vector<std::vector<int>> slots_;  // slots_[layer][slot] → pair index, -1 unbound
  std::vector<std::string> keys_sorted_;
  mutable std::vector<char> hit_;
  // If E0_LORA_SCAFFOLD=<path.jsonl>, write base/delta prefix-64+summary on first dressed
  // call per key. Observe-only; does not change numeric paths.
  mutable std::string cap_path_;
  mutable std::set<std::string> cap_done_;
  int ns_ = 0;
  int64_t bytes_ = 0;
};

inline mlx::core::array apply_lora(const Bundle* lo, const QuantLin& lin,
                                   const mlx::core::array& x, int layer, int slot) {
  return lo ? lo->dressed(lin, x, layer, slot) : lin.apply(x);
}

}  // namespace lora
}  // namespace e0n
