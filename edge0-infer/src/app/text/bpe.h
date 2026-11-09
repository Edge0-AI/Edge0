// Copyright (c) 2023-2026 The ggml authors. MIT license; see LICENSE.llama.cpp.
// adapted from
// https://github.com/ggml-org/llama.cpp/blob/7ab4ee7baad2d920464cbacfad4f4b07cf111fd2/src/llama-vocab.cpp
#pragma once
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
namespace edge0::text {
enum class Error { invalid_argument, incompatible };
class TextError : public std::runtime_error {
public:
  TextError(Error code, const std::string &message)
      : std::runtime_error(message), code(code) {}
  Error code;
};
struct TokenData {
  std::string model = "gpt2", pre = "qwen35";
  std::vector<std::string> tokens;
  std::vector<int32_t> types;
  std::vector<std::string> merges;
  int32_t bos = -1, eos = -1, eot = -1, padding = -1, eom = -1;
  bool add_bos = false, add_eos = false;
};
struct Fragment {
  std::string text;
  bool protocol = false;
};
bool valid_utf8(const std::string &text);
class Bpe {
public:
  explicit Bpe(TokenData data);
  std::vector<int32_t> encode(const std::string &text,
                              bool parse_special) const;
  std::vector<int32_t>
  encode_input(const std::vector<Fragment> &fragments) const;
  std::string token_to_piece(int32_t token, bool special) const;
  const std::set<int32_t> &eog() const { return eog_; }
  size_t size() const { return data_.tokens.size(); }

private:
  std::vector<int32_t> encode_fragments(const std::vector<Fragment> &fragments,
                                        bool standalone) const;
  void ordinary(const std::string &text, std::vector<int32_t> &out) const;
  TokenData data_;
  std::unordered_map<std::string, int32_t> ids_;
  std::map<std::pair<std::string, std::string>, int32_t> ranks_;
  std::vector<int32_t> specials_;
  std::set<int32_t> eog_;
};
} // namespace edge0::text
