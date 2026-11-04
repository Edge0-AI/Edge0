#include "tests/support/model_fixture.h"
struct Fake final : edge0::model::Adapter {
    bool missing_vocabulary = false;
    Fake() : Adapter({"base"}) {}
    std::string id() const override {
        return "fake";
    }
    edge0::model::TextDescription describe_text() const override {
        edge0::model::Sha256 h;
        h.update("fake template");
        return {h.finish(), "gpt2", "qwen35", false, false};
    }
    edge0::model::TextResult text_format(const std::vector<edge0::model::Message> &,
                                     const edge0::model::TextOptions &) const override {
        return {};
    }
    edge0::Result<edge0::model::ModelDefinition> describe_assets(const edge0::model::AssetView &) const override {
        edge0::model::ModelDefinition d;
        d.vocabulary_size=4;d.context_length=4096;d.layer_count=1;d.expert_count=1;d.experts_per_token=1;
        d.required_roles = {"base", "attachment", "chat_template"};
        d.tensors = {{"base", "a", "f32", {{4, 1, 1, 1}}}};
        if (missing_vocabulary) d.vocabulary_size=0;
        return d;
    }
};
int main() {
    using namespace fixture;
    using edge0::ErrorCategory;
    Fake fake;
    Catalog c;
    c.architecture = "unknown";
    error(select_adapter(c, {&fake}), ErrorCategory::unsupported, "fake");
    c.architecture = "fake";
    CHECK(select_adapter(c, {&fake}).value() == &fake);
    Temp t;
    auto p = t.path / "base.gguf";
    std::vector<TensorRequirement> req = {{"base", "a", "f32", {{4, 1, 1, 1}}}};
    Metadata m;
    m["tokenizer.ggml.model"] = std::string("gpt2");
    m["tokenizer.ggml.pre"] = std::string("qwen35");
    gguf(p, m, req);
    c.files["base"] = file(p, "base");
    error(check_compatibility(c, fake), ErrorCategory::incompatible, "attachment");
    c.files["attachment"] = {};
    write(t.path / "template", "fake template");
    c.files["chat_template"] = file(t.path / "template", "chat_template");
    c.files["extra"] = {};
    fake.missing_vocabulary=true;
    error(check_compatibility(c, fake), ErrorCategory::incompatible, "vocabulary_size");
    fake.missing_vocabulary=false;
    auto good = check_compatibility(c, fake);
    CHECK(good);
    CHECK(good.value().unused_roles == std::vector<std::string>{"extra"});
    for (const auto *key : {"tokenizer.ggml.model", "tokenizer.ggml.pre"}) {
        auto altered = m;
        altered[key] = std::string("unsupported-test-value");
        gguf(p, altered, req);
        auto rejected = check_compatibility(c, fake);
        error(rejected, ErrorCategory::incompatible, key);
        CHECK(rejected.error().message.find("expected") != std::string::npos);
        CHECK(rejected.error().message.find("actual unsupported-test-value") != std::string::npos);
    }
    gguf(p, m, req);
    for (int mode = 0; mode < 4; ++mode) {
        auto altered = req;
        if (mode == 0)
            altered.clear();
        if (mode == 1)
            altered.push_back({"base", "extra_tensor", "f32", {{4, 1, 1, 1}}});
        if (mode == 2)
            altered[0].type = "f16";
        if (mode == 3)
            altered[0].shape[0] = 8;
        gguf(p, m, altered);
        error(check_compatibility(c, fake), ErrorCategory::incompatible, mode == 1 ? "extra_tensor" : "a");
    }
    c.files.erase("base");
    error(check_compatibility(c, fake), ErrorCategory::incompatible, "base");
    std::cout << "adapter selection and complete compatibility rejection matrix passed\n";
}
