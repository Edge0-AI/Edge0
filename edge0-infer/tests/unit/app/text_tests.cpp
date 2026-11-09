#include "app/text/bpe.h"
#include "app/text/decode.h"
#include "app/text/unicode.h"
#include "app/text_test_support.h"
#include "models/edge0_35b/text.h"
#include <algorithm>
#include <random>
using edge0::text::Bpe;
void bpe_test() {
  edge0::text::TokenData d;
  for (int i = 0; i < 256; ++i) {
    d.tokens.push_back(
        edge0::text::unicode_byte_to_utf8(static_cast<uint8_t>(i)));
    d.types.push_back(1);
  }
  d.tokens.insert(d.tokens.end(), {"ab", "bc", "abc", "<|im_end|>", "<think>",
                                   "<|endoftext|>", "custom"});
  d.types.insert(d.types.end(), {1, 1, 1, 3, 3, 3, 4});
  d.merges = {"b c", "a b"};
  d.eos = 261;
  {
    auto unknown=d;unknown.pre="unknown-rule";
    bool rejected=false;
    try { Bpe unsupported(unknown); }
    catch (const edge0::text::TextError& error) {
      rejected=error.code==edge0::text::Error::incompatible &&
          std::string(error.what()).find("unknown-rule")!=std::string::npos &&
          std::string(error.what()).find("qwen35")!=std::string::npos;
    }
    require(rejected,"unknown pre-tokenizer reports actual and supported names");
  }
  Bpe b(d);
  require(b.encode("abc", false) == std::vector<int32_t>({97, 257}),
          "rank priority");
  require(b.encode("<|im_end|>", true) == std::vector<int32_t>({259}),
          "special true");
  require(b.encode("<|im_end|>", false) != std::vector<int32_t>({259}),
          "special false");
  require(b.encode_input({{"a", false}, {"bc", false}}) ==
              b.encode("abc", false),
          "ordinary boundary merge");
  require(b.encode_input({{"<|im_", true}, {"end|>", false}}) ==
              b.encode("<|im_end|>", false),
          "mixed source marker");
  require(b.encode_input({{"<|im_", true}, {"end|>", true}}) ==
              std::vector<int32_t>({259}),
          "protocol split marker");
  require(b.encode_input({{"<|im_end|>", true}, {"<think>", false}}).front() ==
              259,
          "protocol retained");
  require(b.encode("e\u0301", false) != b.encode("\u00e9", false),
          "no normalization");
  for (int i = 0; i < 256; ++i)
    require(b.token_to_piece(i, true) == std::string(1, static_cast<char>(i)),
            "byte map");
  require(b.token_to_piece(259, false).empty(), "hidden control");
  require(b.eog() == std::set<int32_t>({259, 261}), "EOG");
  require(b.encode("custom", false) == std::vector<int32_t>({262}),
          "user-defined matching");
  require(b.encode_input({{"custom", false}}) != std::vector<int32_t>({262}),
          "user-defined body protected");
  for (const auto &s :
       {std::string("\x80"), std::string("\xed\xa0\x80"),
        std::string("\xf4\x90\x80\x80"), std::string("\xe2\x82")}) {
    bool rejected = false;
    try {
      b.encode(s, false);
    } catch (const edge0::text::TextError &) {
      rejected = true;
    }
    require(rejected, "invalid UTF8 rejected");
  }
  d.bos = 261;
  d.add_bos = true;
  d.add_eos = true;
  Bpe with(d);
  require(with.encode("a", false) == std::vector<int32_t>({261, 97, 261}),
          "metadata BOS/EOS");
}
void decode_test() {
  auto file = fixture("utf8_golden.tsv");
  std::string line;
  std::mt19937 rng(350017);
  size_t count = 0;
  while (std::getline(file, line)) {
    auto f = split(line, '\t');
    auto input = unhex(f[0]), expected = unhex(f[1]);
    edge0::text::Decoder whole;
    auto complete = whole.decode_increment(input);
    complete += whole.finish();
    require(complete == expected, "whole decode " + f[0]);
    for (int trial = 0; trial < 1000; ++trial) {
      edge0::text::Decoder decoder;
      std::string out;
      for (size_t p = 0; p < input.size();) {
        size_t n =
            std::min(input.size() - p, static_cast<size_t>(1 + rng() % 7));
        auto chunk = decoder.decode_increment(input.substr(p, n));
        require(edge0::text::valid_utf8(chunk), "unstable output");
        out += chunk;
        p += n;
      }
      out += decoder.finish();
      require(out == expected, "split decode " + f[0]);
      require(decoder.finish().empty(), "finish idempotent");
    }
    ++count;
  }
  edge0::text::Decoder d;
  require(d.decode_increment("\xe4\xbd").empty(), "incomplete retained");
  require(d.decode_increment("\xa0\xef\xbf\xbd") == "\xe4\xbd\xa0\xef\xbf\xbd",
          "valid replacement retained");
  std::cout << "decode cases=" << count << " cuts_each=1000\n";
}
std::vector<edge0::models::edge0_35b::Message> messages(const std::string &s) {
  std::vector<edge0::models::edge0_35b::Message> out;
  for (const auto &m : split(s, ';')) {
    auto p = m.find(':');
    out.push_back({m.substr(0, p), unhex(m.substr(p + 1)), {}});
  }
  return out;
}
std::vector<edge0::text::Fragment>
fragments(const edge0::models::edge0_35b::TextResult &r) {
  std::vector<edge0::text::Fragment> out;
  for (const auto &f : r.fragments)
    out.push_back({f.text, f.protocol});
  return out;
}
std::string rendered(const edge0::models::edge0_35b::TextResult &r) {
  std::string out;
  for (const auto &f : r.fragments)
    out += f.text;
  return out;
}
void template_test(bool tokens) {
  const auto declaration=edge0::models::edge0_35b::describe_text();
  require(declaration.template_sha256=="e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259" && declaration.model=="gpt2" && declaration.pre=="qwen35" && !declaration.default_thinking && declaration.tools && declaration.thinking_enabled,"text declaration");
  std::unique_ptr<Bpe> b;
  if (tokens)
    b = std::make_unique<Bpe>(vocabulary());
  auto file = fixture("template_golden.tsv");
  std::string line;
  while (std::getline(file, line)) {
    auto f = split(line, '\t');
    auto r = edge0::models::edge0_35b::text_format(messages(f[3]), {{}, false});
    require(!r.thinking, "thinking effective");
    require(rendered(r) == unhex(f[1]), "template bytes " + f[0]);
    if (b) {
      auto got = b->encode_input(fragments(r));
      auto want = ids(f[2]);
      if (got != want) {
        std::cerr << "got:";
        for (auto id : got)
          std::cerr << ' ' << id;
        std::cerr << "\nwant:";
        for (auto id : want)
          std::cerr << ' ' << id;
        std::cerr << '\n';
      }
      require(got == want, "template tokens " + f[0]);
    }
  }
  auto reject = [](std::vector<edge0::models::edge0_35b::Message> m,
                   edge0::models::edge0_35b::TextOptions o = {}) {
    bool bad = false;
    try {
      edge0::models::edge0_35b::text_format(m, o);
    } catch (const std::invalid_argument &) {
      bad = true;
    }
    require(bad, "invalid template must reject");
  };
  reject({});
  reject({{"assistant", "a", {}}});
  reject({{"user", "u", {}}, {"system", "s", {}}});
  reject({{"tool", "a", {}}});
  reject({{"unknown", "a", {}}});
  reject({{"user", "a", {{"id", "f", "{}"}}}});
  require(edge0::models::edge0_35b::text_format({{"user", "a", {}}}, {true, {}}).thinking, "thinking supported");
  require(!edge0::models::edge0_35b::text_format({{"user", "a", {}}}, {{}, true}).fragments.empty(), "preserve thinking supported");
  if (b) {
    auto first = edge0::models::edge0_35b::text_format({{"user", "Hello", {}}});
    auto first_tokens = b->encode_input(fragments(first));
    auto next = edge0::models::edge0_35b::text_format({{"user", "Hello", {}},
                                                 {"assistant", "Answer", {}},
                                                 {"user", "Next", {}}});
    auto next_tokens = b->encode_input(fragments(next));
    size_t prefix = 0;
    while (prefix < std::min(first_tokens.size(), next_tokens.size()) &&
           first_tokens[prefix] == next_tokens[prefix])
      ++prefix;
    require(prefix == first_tokens.size(), "preserved prompt prefix");
    std::cout << "prefix_tokens=" << prefix
              << " first_prompt_N=" << first_tokens.size()
              << " checkpoint_N_minus_1=" << first_tokens.size() - 1 << '\n';
  }
}
void corpus_test() {
  Bpe b(vocabulary());
  auto f = fixture("tokenize_golden.tsv");
  std::string line;
  size_t count = 0;
  while (std::getline(f, line)) {
    auto r = split(line, '\t');
    auto got = b.encode(unhex(r[2]), r[1] == "1");
    auto expected = ids(r[3]);
    if (got != expected) {
      std::cerr << "first mismatch " << r[0] << " parse=" << r[1] << " got:";
      for (auto id : got)
        std::cerr << ' ' << id;
      std::cerr << " expected:";
      for (auto id : expected)
        std::cerr << ' ' << id;
      std::cerr << '\n';
      throw std::runtime_error("corpus mismatch");
    }
    ++count;
  }
  auto pieces = fixture("pieces_golden.tsv");
  std::set<int32_t> eog;
  size_t n = 0;
  while (std::getline(pieces, line)) {
    auto r = split(line, '\t');
    auto id = static_cast<int32_t>(std::stoi(r[0]));
    require(b.token_to_piece(id, false) == unhex(r[1]), "piece false " + r[0]);
    require(b.token_to_piece(id, true) == unhex(r[2]), "piece true " + r[0]);
    if (r[3] == "1")
      eog.insert(id);
    ++n;
  }
  require(n == 248320 && n == b.size(), "complete vocabulary");
  require(eog == b.eog(), "complete EOG");
  std::cout << "corpus=" << count << " pieces=" << n
            << " both_special_modes eog=" << eog.size() << '\n';
}
int main(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--decode-steps") {
    std::string line;
    while (std::getline(std::cin, line)) {
      edge0::text::Decoder decoder;
      for (const auto &part : split(line, ';'))
        std::cout << hex(decoder.decode_increment(unhex(part))) << ':';
      std::cout << hex(decoder.finish()) << '\n';
      std::cout.flush();
    }
    return 0;
  }
  try {
    std::string mode = MOE_TEXT_MODE;
    if (mode == "bpe")
      bpe_test();
    else if (mode == "decode")
      decode_test();
    else if (mode == "template")
      template_test(false);
    else if (mode == "template_tokens")
      template_test(true);
    else
      corpus_test();
    std::cout << "passed " << mode << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
