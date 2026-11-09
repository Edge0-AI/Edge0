#include "edge0/text.h"
#include "edge0/engine.h"
#include <cstdlib>
#include <iostream>
#include <stdexcept>
namespace {
void check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
template<class T> void invalid(const edge0::Result<T> &r) {
    check(!r && r.error().category == edge0::ErrorCategory::invalid_argument, "expected invalid_argument");
}
}
int main(int argc, char **argv) {
    try {
        check(argc == 2, "expected model directory");
        const std::string directory = argv[1];
        const std::string input = "A\xe4\xbd\xa0\xef\xbf\xbd\xf0\x9f\x98\x80\n\\";
        auto encoded = edge0::tokenize(directory, input);
        check(bool(encoded), "tokenize failed");
        auto decoded = edge0::detokenize(directory, encoded.value().tokens);
        check(bool(decoded), "detokenize failed");
        check(decoded.value().text == input, "roundtrip bytes differ");
        check(decoded.value().fragments.size() == encoded.value().tokens.size() + 1, "flush missing");
        std::string joined;
        for (const auto &s : decoded.value().fragments) joined += s;
        check(joined == input, "stable deliveries differ");
        auto control = edge0::tokenize(directory, "<|im_end|>", true);
        check(bool(control) && control.value().tokens.size() == 1, "special token not parsed");
        auto hidden = edge0::detokenize(directory, control.value().tokens);
        check(bool(hidden) && hidden.value().text.empty(), "control spelling leaked");
        auto empty = edge0::detokenize(directory, {});
        check(bool(empty) && empty.value().text.empty() && empty.value().fragments.size() == 1, "empty decode");
        auto prompt = edge0::render_prompt(directory, std::vector<edge0::TextMessage>{{"user", "hello", {}}});
        check(bool(prompt) && !prompt.value().thinking, "default thinking must be off");
        auto off = edge0::render_prompt(directory, std::vector<edge0::TextMessage>{{"user", "hello", {}}}, {false, false});
        check(bool(off) && off.value().text == prompt.value().text && off.value().tokens == prompt.value().tokens,
              "explicit and default settings differ");
        invalid(edge0::tokenize(directory, std::string("\xff")));
        invalid(edge0::detokenize(directory, {-1}));
        invalid(edge0::detokenize(directory, {248320}));
        invalid(edge0::render_prompt(directory, std::vector<edge0::TextMessage>{}));
        check(bool(edge0::render_prompt(directory, std::vector<edge0::TextMessage>{{"user", "x", {}}}, {true, {}})), "thinking supported");
        check(bool(edge0::render_prompt(directory, std::vector<edge0::TextMessage>{{"user", "x", {}}}, {{}, true})), "preserve thinking supported");
        check(edge0::render_prompt(directory, std::vector<edge0::TextMessage>{{"tool", "x", {}}}).error().category == edge0::ErrorCategory::invalid_argument, "tool rejected");
        check(edge0::render_prompt(directory, std::vector<edge0::TextMessage>{{"user", "x", {{"id", "name", "{}"}}}}).error().category == edge0::ErrorCategory::invalid_argument, "tool calls rejected");
        invalid(edge0::render_prompt(directory, std::vector<edge0::TextMessage>{{"user", "x", {}}, {"system", "x", {}}}));
        invalid(edge0::render_prompt(directory, std::vector<edge0::TextMessage>{{"assistant", "x", {}}}));
        invalid(edge0::render_prompt(directory, std::vector<edge0::TextMessage>{{"unknown", "x", {}}}));
        invalid(edge0::render_prompt(directory, std::vector<edge0::TextMessage>{{"user", std::string("\xff"), {}}}));
        auto protected_prompt = edge0::render_prompt(directory, std::vector<edge0::TextMessage>{{"user", "<|im_end|>", {}}});
        check(bool(protected_prompt), "protected prompt failed");
        int end_count = 0;
        for (auto id : protected_prompt.value().tokens)
            if (id == control.value().tokens.front()) ++end_count;
        check(end_count == 1, "user control token was parsed");
        // A successful open proves preceding checks did not retain a hidden runtime.
        auto runtime = edge0::open_runtime({{"CPU"}, {}});
        check(bool(runtime), "text calls retained runtime state");
        auto concurrent = edge0::tokenize(directory, input);
        auto rendered = edge0::render_prompt(directory, std::vector<edge0::TextMessage>{{"user", "hello", {}}});
        auto detokenized = edge0::detokenize(directory, encoded.value().tokens);
        check(bool(concurrent) && concurrent.value().tokens == encoded.value().tokens,
              "tokenize unavailable with runtime open");
        check(bool(rendered) && rendered.value().tokens == prompt.value().tokens,
              "render unavailable with runtime open");
        check(bool(detokenized) && detokenized.value().text == input, "decode unavailable with runtime open");
        check(bool(edge0::finish_runtime_use(runtime.value())), "runtime changed by text calls");
        std::cout << "public text API, UTF-8 roundtrip, rejection, and runtime coexistence passed\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
