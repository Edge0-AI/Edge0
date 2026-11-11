#include "app/text/bpe.h"
#include "app/text/unicode.h"
#include <chrono>
#include <iostream>
int main() {
    edge0::text::TokenData data;
    for (int i = 0; i < 256; ++i) {
        data.tokens.push_back(edge0::text::unicode_byte_to_utf8(static_cast<uint8_t>(i)));
        data.types.push_back(1);
    }
    const std::string input(8192, 'a');
    for (std::size_t size = 1; size < input.size(); size *= 2) {
        const std::string half(size, 'a');
        data.merges.push_back(half + " " + half);
        data.tokens.push_back(half + half);
        data.types.push_back(1);
    }
    const auto expected = std::vector<int32_t>{static_cast<int32_t>(data.tokens.size() - 1)};
    edge0::text::Bpe tokenizer(std::move(data));
    if (tokenizer.encode(input, false) != expected) return 1;
    const auto start = std::chrono::steady_clock::now();
    for (int repeat = 0; repeat < 3; ++repeat)
        if (tokenizer.encode(input, false) != expected) return 1;
    std::cout << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() << '\n';
}
