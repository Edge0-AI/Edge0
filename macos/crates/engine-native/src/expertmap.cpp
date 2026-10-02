#include "expertmap.hpp"

#include <cstdio>

namespace e0n {
namespace {

uint64_t dtype_size(const std::string& d) {
  if (d == "BF16" || d == "F16") return 2;
  if (d == "F32" || d == "U32" || d == "I32") return 4;
  if (d == "F64") return 8;
  if (d == "U8" || d == "I8" || d == "BOOL") return 1;
  return 0;
}

constexpr const char* kParts[] = {"gate_proj", "up_proj", "down_proj"};
constexpr const char* kFields[] = {"weight", "scales", "biases"};

}  // namespace

bool ExpertMap::build(const ModelFiles& files, ExpertMap& out, std::string& err) {
  int64_t E = 0, L = 0;
  if (!files.config_int("num_experts", E) || E <= 0) {
    err = "config.json missing num_experts";
    return false;
  }
  if (!files.config_int("num_hidden_layers", L) || L <= 0) {
    err = "config.json missing num_hidden_layers";
    return false;
  }
  // qwen (35b): presence of language_model.model.layers.0.mlp.gate.weight.
  const bool qwen = files.find("language_model.model.layers.0.mlp.gate.weight") != nullptr;

  out.num_layers_ = static_cast<int>(L);
  out.num_experts_ = E;
  out.prefixes_.assign(static_cast<size_t>(L), "");
  out.layers_.clear();
  out.layers_.resize(static_cast<size_t>(L));

  for (int l = 0; l < static_cast<int>(L); l++) {
    const std::string base =
        qwen ? "language_model.model.layers." + std::to_string(l)
             : "model.layers." + std::to_string(l);
    std::string prefix;
    for (const std::string& cand :
         {base + ".mlp.experts", base + ".mlp.switch_mlp", base + ".mlp"}) {
      if (files.find(cand + ".gate_proj.weight")) { prefix = cand; break; }
    }
    if (prefix.empty()) {
      err = "layer " + std::to_string(l) + " has no MoE fused prefix";
      return false;
    }
    out.prefixes_[static_cast<size_t>(l)] = prefix;

    auto& slices = out.layers_[static_cast<size_t>(l)];
    for (const char* p : kParts) {
      for (const char* f : kFields) {
        const std::string name = std::string(p) + "." + f;
        const TensorInfo* t = files.find(prefix + "." + name);
        if (!t) {
          err = "missing tensor " + prefix + "." + name;
          return false;
        }
        const uint64_t es = dtype_size(t->dtype);
        if (es == 0) {
          err = "unknown dtype " + t->dtype + " for " + name;
          return false;
        }
        Slice s;
        if (t->shape.size() >= 3 && t->shape[0] == E) {
          // First dim = expert count (35b): one expert is a contiguous stride.
          s.stride = static_cast<uint64_t>(t->shape[1]) * static_cast<uint64_t>(t->shape[2]) * es;
        } else if (t->shape.size() == 2 && t->shape[0] % E == 0) {
          // E folded into the row dim (8b and 8b-L0 dense).
          s.stride = (static_cast<uint64_t>(t->shape[0]) / static_cast<uint64_t>(E)) *
                     static_cast<uint64_t>(t->shape[1]) * es;
        } else {
          err = "unsupported layout " + prefix + "." + name;
          return false;
        }
        s.base = t->begin;
        s.shard = t->shard;
        s.name = name;
        slices.emplace(name, s);
      }
    }
  }
  return true;
}

std::vector<Region> ExpertMap::regions(int layer, int64_t expert) const {
  std::vector<Region> out;
  if (layer < 0 || layer >= num_layers_ || expert < 0 || expert >= num_experts_) return out;
  out.reserve(9);
  for (const char* p : kParts) {
    for (const char* f : kFields) {
      auto it = layers_[static_cast<size_t>(layer)].find(std::string(p) + "." + f);
      if (it == layers_[static_cast<size_t>(layer)].end()) return {};
      const Slice& s = it->second;
      out.push_back(Region{s.base + static_cast<uint64_t>(expert) * s.stride, s.stride,
                           s.shard, s.name});
    }
  }
  return out;
}

uint64_t ExpertMap::expert_bytes(int layer, int64_t expert) const {
  uint64_t total = 0;
  for (auto& r : regions(layer, expert)) total += r.len;
  return total;
}

const std::string& ExpertMap::layer_prefix(int layer) const {
  static const std::string empty;
  if (layer < 0 || layer >= num_layers_) return empty;
  return prefixes_[static_cast<size_t>(layer)];
}

}  // namespace e0n
