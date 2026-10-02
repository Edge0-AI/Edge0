// Safetensors multi-shard header parse. A tensor lives in one shard; an expert's
// 9 regions may span shards — weight_map in model.safetensors.index.json is authoritative.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace e0n {

struct TensorInfo {
  std::string dtype;
  std::vector<int64_t> shape;
  uint64_t begin = 0, end = 0;  // absolute offset in the shard file (includes 8+header_len)
  std::string shard;
};

class ModelFiles {
 public:
  bool open(const std::string& dir, std::string& err);

  const TensorInfo* find(const std::string& tensor_name) const;
  std::string shard_path(const std::string& shard_name) const;
  std::vector<std::string> shard_names() const;
  uint64_t shard_size(const std::string& shard_name) const;  // 0 on failure

  bool read_tensor(const std::string& tensor_name, std::vector<uint8_t>& out,
                   std::string& err) const;

  bool config_int(const std::string& key, int64_t& out) const;
  bool config_bool(const std::string& key, bool& out) const;
  bool config_double(const std::string& key, double& out) const;
  bool config_quant(int64_t& bits, int64_t& group_size) const;
  // Per-module override (e.g. 35b mlp.gate is int8/g64 while the root is int4).
  // Longest-prefix match on the module path; else root quantization_config.
  bool config_quant_for(const std::string& tensor_name, int64_t& bits,
                        int64_t& group_size) const;
  bool has_file(const std::string& file_name) const;

 private:
  std::string dir_;
  std::map<std::string, TensorInfo> tensors_;
  std::map<std::string, uint64_t> shard_sizes_;
  std::string config_json_;
};

}  // namespace e0n
