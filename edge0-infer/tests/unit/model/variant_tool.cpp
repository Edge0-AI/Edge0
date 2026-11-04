#include "tests/support/model_fixture.h"
int main(int argc, char **argv) {
    using namespace fixture;
    CHECK(argc == 4);
    std::string mode = argv[3];
    ggml_context *context = nullptr;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> source(
        gguf_init_from_file(argv[1], {false, &context}), gguf_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> owner(context, ggml_free);
    CHECK(source && owner);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> dest(gguf_init_empty(), gguf_free);
    gguf_set_kv(dest.get(), source.get());
    if (mode == "metadata")
        gguf_remove_key(dest.get(), "qwen3next.expert_count");
    for (std::int64_t i = 0; i < gguf_get_n_tensors(source.get()); ++i) {
        std::string name = gguf_get_tensor_name(source.get(), i);
        if (mode == "missing" && name == "blk.0.ffn_gate_exps.weight")
            continue;
        auto *tensor = ggml_get_tensor(owner.get(), name.c_str());
        CHECK(tensor);
        gguf_add_tensor(dest.get(), tensor);
        if (mode == "type" && name == "blk.0.attn_norm.weight")
            gguf_set_tensor_type(dest.get(), name.c_str(), GGML_TYPE_F16);
    }
    if (mode == "extra") {
        auto *tensor = ggml_get_tensor(owner.get(), "blk.0.attn_norm.weight");
        ggml_set_name(tensor, "unexpected_tensor");
        gguf_add_tensor(dest.get(), tensor);
    }
    CHECK(gguf_write_to_file(dest.get(), argv[2], false));
}
