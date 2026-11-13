// Copyright (c) 2023-2026 The ggml authors. MIT license; see LICENSE.llama.cpp.
// adapted from
// https://github.com/ggml-org/llama.cpp/blob/7ab4ee7baad2d920464cbacfad4f4b07cf111fd2/src/llama-vocab.cpp
#include "app/text/bpe.h"
#include "app/text/unicode.h"
#include <algorithm>
#include <limits>
namespace edge0::text {
namespace {
const std::map<std::string, std::vector<std::string>> patterns = {{"qwen35", {
    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}"
    "\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| "
    "?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}}};
bool special_type(int32_t type) { return type == 2 || type == 3 || type == 4; }
} // namespace
Bpe::Bpe(TokenData data) : data_(std::move(data)) {
  if (data_.model != "gpt2")
    throw TextError(Error::incompatible, "expected gpt2");
  if (!patterns.count(data_.pre)) {
    std::string supported;
    for (const auto& entry : patterns) supported += " " + entry.first;
    throw TextError(Error::incompatible, "unsupported pre-tokenizer " + data_.pre + "; supported:" + supported);
  }
  if (data_.tokens.empty() || data_.tokens.size() != data_.types.size() ||
      data_.tokens.size() >
          static_cast<size_t>(std::numeric_limits<int32_t>::max()))
    throw TextError(Error::invalid_argument, "invalid vocabulary");
  const std::set<std::string> ends = {
      "<|eot_id|>",
      "<|im_end|>",
      "<|end|>",
      "<|return|>",
      "<|call|>",
      "<|flush|>",
      "<|calls|>",
      "<end_of_turn>",
      "<|endoftext|>",
      "</s>",
      "<|eom_id|>",
      "<EOT>",
      "_<EOT>",
      "[EOT]",
      "[EOS]",
      "<|end_of_text|>",
      "<end_of_utterance>",
      "<eos>",
      "<turn|>",
      "<|tool_response>",
      "<\xef\xbd\x9c"
      "end\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c>",
      "[e~[",
      "<|fim_pad|>",
      "<fim-pad>",
      "<fim_pad>",
      "<PAD>",
      "[PAD]",
      "<|fim_repo|>",
      "<|repo_name|>",
      "<fim-repo>",
      "<REPO>",
      "<reponame>",
      "<|file_sep|>"};
  for (size_t i = 0; i < data_.tokens.size(); ++i) {
    auto id = static_cast<int32_t>(i);
    const auto &s = data_.tokens[i];
    if (s.empty() || !ids_.emplace(s, id).second)
      throw TextError(Error::invalid_argument, "empty or duplicate token");
    if (ends.count(s)) {
      eog_.insert(id);
      data_.types[i] = 3;
    }
  }
  for (auto id : {data_.bos, data_.eos, data_.eot, data_.padding, data_.eom})
    if (id < -1 || id >= static_cast<int32_t>(size()))
      throw TextError(Error::invalid_argument, "invalid special ID");
  if ((data_.add_bos && data_.bos < 0) || (data_.add_eos && data_.eos < 0))
    throw TextError(Error::invalid_argument, "missing BOS/EOS");
  for (auto id : {data_.eos, data_.eot, data_.eom})
    if (id >= 0)
      eog_.insert(id);
  auto contains = [&](const std::string &s) { return ids_.count(s) != 0; };
  if (contains("<|end|>") && (contains("<|call|>") || contains("<|calls|>")) &&
      (contains("<|return|>") || contains("<|flush|>"))) {
    auto id = ids_.at("<|end|>");
    eog_.erase(id);
    data_.types[id] = 4;
  }
  if (contains("<|tool_response>") && contains("</s>")) {
    auto id = ids_.at("</s>");
    eog_.erase(id);
    data_.types[id] = 1;
  }
  for (const auto &s :
       {"<|channel|>", "<|message|>", "<|start|>", "<|constrain|>"})
    if (contains(s))
      data_.types[ids_.at(s)] = 4;
  for (size_t i = 0; i < data_.tokens.size(); ++i)
    if (special_type(data_.types[i]))
      specials_.push_back(static_cast<int32_t>(i));
  std::sort(specials_.begin(), specials_.end(), [&](int32_t a, int32_t b) {
    return data_.tokens[a].size() > data_.tokens[b].size();
  });
  for (size_t i = 0; i < data_.merges.size(); ++i) {
    const auto &s = data_.merges[i];
    auto p = s.find(' ');
    if (p == std::string::npos)
      throw TextError(Error::invalid_argument, "invalid merge");
    ranks_.emplace(std::make_pair(s.substr(0, p), s.substr(p + 1)),
                   static_cast<int32_t>(i));
  }
}
void Bpe::ordinary(const std::string &text, std::vector<int32_t> &out) const {
  for (const auto &word : unicode_regex_split(text, patterns.at(data_.pre))) {
    std::vector<std::string> symbols;
    for (size_t p = 0; p < word.size();) {
      auto n = unicode_len_utf8(word[p]);
      symbols.push_back(word.substr(p, n));
      p += n;
    }
    while (symbols.size() > 1) {
      int32_t best = std::numeric_limits<int32_t>::max();
      size_t at = symbols.size();
      for (size_t i = 0; i + 1 < symbols.size(); ++i) {
        auto it = ranks_.find({symbols[i], symbols[i + 1]});
        if (it != ranks_.end() && it->second < best) {
          best = it->second;
          at = i;
        }
      }
      if (at == symbols.size())
        break;
      symbols[at] += symbols[at + 1];
      symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(at + 1));
    }
    for (const auto &s : symbols) {
      auto it = ids_.find(s);
      if (it != ids_.end())
        out.push_back(it->second);
      else
        for (char c : s) {
          auto b = ids_.find(std::string(1, c));
          if (b != ids_.end())
            out.push_back(b->second);
        }
    }
  }
}
std::vector<int32_t> Bpe::encode(const std::string &text,
                                 bool parse_special) const {
  return encode_fragments({{text, parse_special}}, true);
}
std::vector<int32_t>
Bpe::encode_input(const std::vector<Fragment> &fragments) const {
  return encode_fragments(fragments, false);
}
std::vector<int32_t>
Bpe::encode_fragments(const std::vector<Fragment> &fragments,
                      bool standalone) const {
  std::string text;
  const bool has_protocol = std::any_of(fragments.begin(), fragments.end(),
      [](const Fragment& f) { return f.protocol && !f.text.empty(); });
  std::vector<bool> allowed;
  for (const auto &f : fragments) {
    text += f.text;
    if (has_protocol) allowed.insert(allowed.end(), f.text.size(), f.protocol);
  }
  if (!valid_utf8(text))
    throw TextError(Error::invalid_argument, "input is not valid UTF-8");
  struct Span {
    size_t begin, end;
    int32_t token;
  };
  std::vector<Span> spans = {{0, text.size(), -1}};
  for (auto id : specials_) {
    // Plain user input cannot contain control tokens. User-defined tokens
    // still match in standalone encoding, as required by that API.
    if (!has_protocol && !(standalone && data_.types[id] == 4)) continue;
    const auto &s = data_.tokens[id];
    std::vector<Span> next;
    for (auto span : spans) {
      if (span.token >= 0) {
        next.push_back(span);
        continue;
      }
      auto pos = span.begin;
      auto search = pos;
      while (search < span.end) {
        auto found = text.find(s, search);
        if (found == std::string::npos || found + s.size() > span.end)
          break;
        bool permitted =
            (standalone && data_.types[id] == 4) ||
            std::all_of(allowed.begin() + static_cast<std::ptrdiff_t>(found),
                        allowed.begin() +
                            static_cast<std::ptrdiff_t>(found + s.size()),
                        [](bool b) { return b; });
        if (!permitted) {
          search = found + 1;
          continue;
        }
        if (found > pos)
          next.push_back({pos, found, -1});
        next.push_back({found, found + s.size(), id});
        pos = found + s.size();
        search = pos;
      }
      if (pos < span.end)
        next.push_back({pos, span.end, -1});
    }
    spans = std::move(next);
  }
  std::vector<int32_t> out;
  if (data_.add_bos)
    out.push_back(data_.bos);
  for (auto span : spans) {
    if (span.token >= 0)
      out.push_back(span.token);
    else
      ordinary(text.substr(span.begin, span.end - span.begin), out);
  }
  if (data_.add_eos)
    out.push_back(data_.eos);
  return out;
}
std::string Bpe::token_to_piece(int32_t token, bool special) const {
  if (token < 0 || token >= static_cast<int32_t>(size()))
    throw TextError(Error::invalid_argument, "token out of range");
  auto type = data_.types[token];
  const auto &s = data_.tokens[token];
  if (type == 2 || type == 3)
    return special ? s : "";
  if (type == 4)
    return s;
  if (type == 6) {
    if (s.size() != 6)
      throw TextError(Error::invalid_argument, "invalid byte token");
    return std::string(
        1, static_cast<char>(std::stoul(s.substr(3, 2), nullptr, 16)));
  }
  if (type != 1)
    return "";
  std::string result;
  for (auto cp : unicode_cpts_from_utf8(s)) {
    auto u = unicode_cpt_to_utf8(cp);
    try {
      result += static_cast<char>(unicode_utf8_to_byte(u));
    } catch (const std::out_of_range &) {
      result += "[UNK_BYTE_0x";
      static constexpr char h[] = "0123456789abcdef";
      for (unsigned char c : u) {
        result += h[c >> 4];
        result += h[c & 15];
      }
      result += s + "]";
    }
  }
  return result;
}
} // namespace edge0::text
