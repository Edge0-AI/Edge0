#pragma once
#include "app/text/bpe.h"
#include "model/adapter.h"
#include "edge0/text.h"
namespace edge0::text {
struct Rules {
    Bpe tokenizer;
    model::TextDescription description;
    const model::Adapter *adapter;
    std::string tokenizer_identity = {};
};
Result<Rules> load_rules(const model::AssetView &assets, const model::Adapter &adapter);
Result<Rules> load_rules(const std::string &directory);
Result<RenderedPrompt> render_input(const Rules&, const std::vector<TextMessage>&, const PromptOptions&);
Result<RenderedPrompt> render_input(const Rules&, const ChatInput&, const PromptOptions&);
} // namespace edge0::text
