#pragma once
#include "app/text/bpe.h"
#include "app/text/rules.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
inline void require(bool value, const std::string &message) {
  if (!value)
    throw std::runtime_error(message);
}
inline std::vector<std::string> split(const std::string &s, char delimiter) {
  std::vector<std::string> out;
  size_t start = 0;
  while (true) {
    auto p = s.find(delimiter, start);
    if (p == std::string::npos) {
      out.push_back(s.substr(start));
      return out;
    }
    out.push_back(s.substr(start, p - start));
    start = p + 1;
  }
}
inline std::string unhex(const std::string &s) {
  require(s.size() % 2 == 0, "odd hex");
  std::string out;
  for (size_t i = 0; i < s.size(); i += 2)
    out += static_cast<char>(std::stoul(s.substr(i, 2), nullptr, 16));
  return out;
}
inline std::string hex(const std::string &s) {
  std::string out;
  static constexpr char h[] = "0123456789abcdef";
  for (unsigned char c : s) {
    out += h[c >> 4];
    out += h[c & 15];
  }
  return out;
}
inline std::vector<int32_t> ids(const std::string &s) {
  std::istringstream stream(s);
  std::vector<int32_t> out;
  int32_t id;
  while (stream >> id)
    out.push_back(id);
  return out;
}
inline std::ifstream fixture(const std::string &name) {
  std::ifstream f(std::string(MOE_TEXT_FIXTURES) + "/" + name);
  require(f.good(), "cannot open fixture " + name);
  return f;
}
inline edge0::text::Bpe vocabulary() {
  const char *root = std::getenv("MOE_TINY");
  if (!root) throw std::runtime_error("MOE_TINY is required");
  auto rules = edge0::text::load_rules(std::string(root) + "/tiny-a");
  require(bool(rules), rules ? "" : rules.error().message);
  return rules.value().tokenizer;
}
