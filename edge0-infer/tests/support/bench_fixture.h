#pragma once
#include "tests/support/model_fixture.h"
#include "models/edge0_35b/assets.h"

namespace fixture {
// Shape-valid zero-weight fixture for CLI metadata checks, not a numerical oracle.
inline void bench_assets(const fs::path& root, const fs::path& source) {
    auto view = metadata();
    view.at("lora").metadata["adapter.lora.alpha"] = 8.0f;
    auto& base = view.at("base").metadata;
    std::vector<std::string> vocabulary;
    for (char c = 'A'; c <= 'Z'; ++c) vocabulary.emplace_back(1, c);
    for (char c = 'a'; c <= 'f'; ++c) vocabulary.emplace_back(1, c);
    base["tokenizer.ggml.tokens"] = vocabulary;
    base["tokenizer.ggml.token_type"] = std::vector<std::int32_t>(vocabulary.size(), 1);
    base["tokenizer.ggml.merges"] = std::vector<std::string>{};
    const auto definition = edge0::models::edge0_35b::adapter().describe_assets(view);
    CHECK(definition);
    for (const auto* role : {"base", "lora", "prerouter"}) {
        std::vector<edge0::model::TensorRequirement> requirements;
        for (const auto& tensor : definition.value().tensors)
            if (tensor.role == role) requirements.push_back(tensor);
        gguf(root / (std::string(role) + ".gguf"), view.at(role).metadata, requirements);
    }
    write(root / "model.json", R"({"format_version":1,"architecture":"edge0_35b"})");
    write(root / "tokenizer.json", "{}");
    write(root / "generation_config.json", R"({"temperature":0.6,"top_k":8,"top_p":0.9,"repetition_penalty":1.1})");
    fs::copy_file(source / "tests/fixtures/models/edge0_35b/chat_template.jinja", root / "chat_template.jinja");
    catalog(root, {{"base", "base.gguf"}, {"lora", "lora.gguf"},
        {"prerouter", "prerouter.gguf"}, {"model", "model.json"}, {"tokenizer", "tokenizer.json"},
        {"generation_config", "generation_config.json"}, {"chat_template", "chat_template.jinja"}});
    write(root / "tokens.json", R"({"input":[0,1,2]})");
}
}
