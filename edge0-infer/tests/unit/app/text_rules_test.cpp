#include "app/text/rules.h"
#include "models/edge0_35b/assets.h"
#include "tests/support/check.h"
#include <iostream>
namespace {
edge0::model::AssetView metadata(const std::string &token, std::int32_t type) {
    edge0::model::AssetView assets;
    auto &m = assets["base"].metadata;
    m["tokenizer.ggml.model"] = std::string("gpt2");
    m["tokenizer.ggml.pre"] = std::string("qwen35");
    m["tokenizer.ggml.tokens"] = std::vector<std::string>{token};
    m["tokenizer.ggml.token_type"] = std::vector<std::int32_t>{type};
    m["tokenizer.ggml.merges"] = std::vector<std::string>{};
    return assets;
}
void rejected(const std::string &token, std::int32_t type, const char *reason) {
    auto result = edge0::text::load_rules(metadata(token, type), edge0::models::edge0_35b::adapter());
    CHECK(!result);
    CHECK(result.error().category == edge0::ErrorCategory::incompatible);
    CHECK(result.error().message.find("tokenizer.ggml.tokens[0]") != std::string::npos);
    CHECK(result.error().message.find(reason) != std::string::npos);
}
}
int main() {
    // Reject malformed metadata before a public piece/decode operation can throw.
    for (const char *token : {"<0xZZ>", "<0x1Z>", "ABC00D", "<0x1>", "<0x0>", "<0x000>", "x0x00>", "<0x00x",
                              "<0X00>", "<0xG0>", "<0x0G>", "<0x-1>", "<0x+1>", "<0x 0>"})
        rejected(token, 6, "byte token");
    for (const auto &entry : {std::make_pair("<0x00>", 0), std::make_pair("<0xAF>", 175),
                              std::make_pair("<0xaf>", 175), std::make_pair("<0xFF>", 255)}) {
        auto result = edge0::text::load_rules(metadata(entry.first, 6), edge0::models::edge0_35b::adapter());
        CHECK(result);
        auto piece = result.value().tokenizer.token_to_piece(0, false);
        CHECK(piece.size() == 1 && static_cast<unsigned char>(piece[0]) == entry.second);
    }
    // A byte-looking spelling is ordinary text unless token_type explicitly says byte.
    auto ordinary = edge0::text::load_rules(metadata("<0xZZ>", 1), edge0::models::edge0_35b::adapter());
    CHECK(ordinary);
    CHECK(ordinary.value().tokenizer.token_to_piece(0, false) == "<0xZZ>");
    for (const auto &token : {std::string("\xff"), std::string("\xc0\x80"),
                              std::string("\xed\xa0\x80"), std::string("\xe4\xbd")})
        for (auto type : {1, 3, 4}) rejected(token, type, "UTF-8");
    auto valid = edge0::text::load_rules(metadata("\xc4\xa0", 1), edge0::models::edge0_35b::adapter());
    CHECK(valid);
    CHECK(valid.value().tokenizer.token_to_piece(0, false) == " ");
    std::cout << "in-memory metadata rejects invalid byte spellings and UTF-8; valid pieces preserved\n";
}
