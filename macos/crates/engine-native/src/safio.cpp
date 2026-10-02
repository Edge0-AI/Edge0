#include "safio.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace e0n {
namespace {

uint64_t file_size_or(const std::string& p, uint64_t fallback) {
  struct stat st;
  return ::stat(p.c_str(), &st) == 0 ? static_cast<uint64_t>(st.st_size) : fallback;
}

// u64 length + JSON; data-region offsets include the prefix.
bool parse_shard_header(const std::string& path, std::map<std::string, TensorInfo>& out,
                        std::string& err) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) { err = "open failed: " + path; return false; }
  uint64_t n = 0;
  if (::pread(fd, &n, 8, 0) != 8 || n == 0 || n > (1ull << 28)) {
    ::close(fd);
    err = "invalid header length: " + path;
    return false;
  }
  std::string js(n, '\0');
  uint64_t done = 0;
  while (done < n) {
    ssize_t got = ::pread(fd, js.data() + done, n - done, 8 + done);
    if (got <= 0) { ::close(fd); err = "header read failed: " + path; return false; }
    done += got;
  }
  ::close(fd);
  json h;
  try {
    h = json::parse(js);
  } catch (const std::exception& e) {
    err = "header JSON parse failed " + path + ": " + e.what();
    return false;
  }
  const uint64_t data_start = 8 + n;
  std::string base = path.substr(path.find_last_of('/') + 1);
  for (auto& [k, v] : h.items()) {
    if (k == "__metadata__" || k == "version" || !v.is_object()) continue;
    if (!v.contains("data_offsets")) continue;
    TensorInfo t;
    t.dtype = v.at("dtype").get<std::string>();
    t.shape = v.at("shape").get<std::vector<int64_t>>();
    auto offs = v.at("data_offsets").get<std::vector<uint64_t>>();
    t.begin = data_start + offs[0];
    t.end = data_start + offs[1];
    t.shard = base;
    out.emplace(k, std::move(t));
  }
  return true;
}

}  // namespace

bool ModelFiles::open(const std::string& dir, std::string& err) {
  dir_ = dir.empty() || dir.back() == '/' ? dir : dir + "/";
  std::error_code ec;
  if (!std::filesystem::is_directory(dir_, ec)) { err = "not a directory: " + dir_; return false; }

  std::vector<std::string> shards;
  for (auto& e : std::filesystem::directory_iterator(dir_, ec)) {
    std::string p = e.path().string();
    if (p.size() > 12 && p.compare(p.size() - 12, 12, ".safetensors") == 0) shards.push_back(p);
  }
  std::sort(shards.begin(), shards.end());
  if (shards.empty()) { err = "no .safetensors in directory: " + dir_; return false; }

  for (auto& s : shards) {
    std::string one_err;
    if (!parse_shard_header(s, tensors_, one_err)) { err = one_err; return false; }
    shard_sizes_[s.substr(s.find_last_of('/') + 1)] = file_size_or(s, 0);
  }

  std::ifstream idx(dir_ + "model.safetensors.index.json");
  if (idx) {
    try {
      json wm = json::parse(idx).at("weight_map");
      for (auto& [name, shard] : wm.items()) {
        auto it = tensors_.find(name);
        if (it != tensors_.end()) it->second.shard = shard.get<std::string>();
      }
    } catch (...) { /* corrupt index: keep header self-assignment */ }
  }

  std::ifstream cfg(dir_ + "config.json");
  if (cfg) {
    std::stringstream ss;
    ss << cfg.rdbuf();
    config_json_ = ss.str();
  }
  return true;
}

const TensorInfo* ModelFiles::find(const std::string& name) const {
  auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

std::string ModelFiles::shard_path(const std::string& shard_name) const {
  return dir_ + shard_name;
}

std::vector<std::string> ModelFiles::shard_names() const {
  std::vector<std::string> v;
  for (auto &[k, n] : shard_sizes_) v.push_back(k);
  return v;
}

bool ModelFiles::read_tensor(const std::string& tensor_name, std::vector<uint8_t>& out,
                             std::string& err) const {
  const TensorInfo* t = find(tensor_name);
  if (!t) { err = "missing tensor: " + tensor_name; return false; }
  out.resize(static_cast<size_t>(t->end - t->begin));
  int fd = ::open(shard_path(t->shard).c_str(), O_RDONLY);
  if (fd < 0) { err = "open failed: " + shard_path(t->shard); return false; }
  size_t done = 0;
  while (done < out.size()) {
    ssize_t got = ::pread(fd, out.data() + done, out.size() - done,
                          static_cast<off_t>(t->begin + done));
    if (got <= 0) {
      ::close(fd);
      err = "pread failed: " + tensor_name;
      return false;
    }
    done += static_cast<size_t>(got);
  }
  ::close(fd);
  return true;
}

uint64_t ModelFiles::shard_size(const std::string& shard_name) const {
  auto it = shard_sizes_.find(shard_name);
  return it == shard_sizes_.end() ? 0 : it->second;
}

bool ModelFiles::config_int(const std::string& key, int64_t& out) const {
  if (config_json_.empty()) return false;
  try {
    json c = json::parse(config_json_);
    if (c.contains(key) && c[key].is_number_integer()) { out = c[key].get<int64_t>(); return true; }
    if (c.contains("text_config")) {
      const json& tc = c["text_config"];
      if (tc.contains(key) && tc[key].is_number_integer()) { out = tc[key].get<int64_t>(); return true; }
    }
  } catch (...) {}
  return false;
}

bool ModelFiles::config_bool(const std::string& key, bool& out) const {
  if (config_json_.empty()) return false;
  try {
    json c = json::parse(config_json_);
    if (c.contains(key) && c[key].is_boolean()) { out = c[key].get<bool>(); return true; }
    if (c.contains("text_config")) {
      const json& tc = c["text_config"];
      if (tc.contains(key) && tc[key].is_boolean()) { out = tc[key].get<bool>(); return true; }
    }
  } catch (...) {}
  return false;
}

bool ModelFiles::config_double(const std::string& key, double& out) const {
  if (config_json_.empty()) return false;
  try {
    json c = json::parse(config_json_);
    if (c.contains(key) && c[key].is_number()) { out = c[key].get<double>(); return true; }
    if (c.contains("text_config")) {
      const json& tc = c["text_config"];
      if (tc.contains(key) && tc[key].is_number()) { out = tc[key].get<double>(); return true; }
    }
  } catch (...) {}
  return false;
}

bool ModelFiles::config_quant(int64_t& bits, int64_t& group_size) const {
  if (config_json_.empty()) return false;
  try {
    json c = json::parse(config_json_);
    const json* q = nullptr;
    if (c.contains("quantization_config")) q = &c["quantization_config"];
    else if (c.contains("text_config") && c["text_config"].contains("quantization_config"))
      q = &c["text_config"]["quantization_config"];
    if (q && q->contains("bits") && q->contains("group_size")) {
      bits = q->at("bits").get<int64_t>();
      group_size = q->at("group_size").get<int64_t>();
      return true;
    }
  } catch (...) {}
  return false;
}

bool ModelFiles::config_quant_for(const std::string& tensor_name, int64_t& bits,
                                  int64_t& group_size) const {
  if (config_json_.empty()) return false;
  try {
    json c = json::parse(config_json_);
    const json* q = nullptr;
    if (c.contains("quantization_config")) q = &c["quantization_config"];
    else if (c.contains("text_config") && c["text_config"].contains("quantization_config"))
      q = &c["text_config"]["quantization_config"];
    if (!q) return false;
    // Override keys are module paths; strip the last segment (.weight etc.) then
    // take the longest prefix match ("...mlp.gate" ⊂ "...mlp.gate.weight").
    auto dot = tensor_name.rfind('.');
    std::string mod = dot == std::string::npos ? tensor_name : tensor_name.substr(0, dot);
    std::string best;
    for (auto& [k, v] : q->items()) {
      if (!v.is_object() || !v.contains("bits") || !v.contains("group_size")) continue;
      if (mod.size() >= k.size() && mod.compare(0, k.size(), k) == 0 &&
          (k.size() > best.size()))
        best = k;
    }
    const json* use = q;
    if (!best.empty()) use = &(*q)[best];
    bits = use->at("bits").get<int64_t>();
    group_size = use->at("group_size").get<int64_t>();
    return true;
  } catch (...) {}
  return false;
}

bool ModelFiles::has_file(const std::string& file_name) const {
  struct stat st;
  return ::stat((dir_ + file_name).c_str(), &st) == 0;
}

}  // namespace e0n
