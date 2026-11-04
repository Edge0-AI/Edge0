#pragma once
#include "model/adapter.h"
#include "model/sha256.h"
#include "gguf.h"
#include "tests/support/check.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
namespace fixture {
namespace fs = std::filesystem;
using namespace edge0::model;
struct Temp {
    fs::path path =
        fs::temp_directory_path() /
        ("moe-model-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Temp() {
        fs::create_directories(path);
    }
    ~Temp() {
        std::error_code e;
        fs::remove_all(path, e);
    }
};
inline void write(const fs::path &path, const std::string &s) {
    std::ofstream out(path, std::ios::binary);
    out << s;
    CHECK(out.good());
}
inline std::string read(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), {}};
}
inline void catalog(const fs::path &root, const std::vector<std::pair<std::string, std::string>> &files) {
    std::ofstream out(root / "assets.tsv");
    out << "role\tbytes\tsha256\tfilename\n";
    for (const auto &f : files) {
        auto h = sha256_file((root / f.second).string());
        CHECK(h);
        out << f.first << '\t' << fs::file_size(root / f.second) << '\t' << h.value() << '\t' << f.second
            << '\n';
    }
}
inline AssetFile file(const fs::path &p, const std::string &role = "base") {
    return {role, p.filename().string(), p.string(), fs::file_size(p), "", false};
}
inline AssetView metadata(std::uint32_t layers = 2, std::uint32_t experts = 2, std::uint32_t interval = 2) {
    AssetView view;
    auto &m = view["base"].metadata;
    m["general.architecture"] = std::string("qwen3next");
    const std::pair<const char *, std::uint32_t> values[] = {{"block_count", layers},
                                                             {"embedding_length", 64},
                                                             {"attention.head_count", 2},
                                                             {"attention.head_count_kv", 1},
                                                             {"attention.key_length", 32},
                                                             {"attention.value_length", 32},
                                                             {"rope.dimension_count", 16},
                                                             {"ssm.group_count", 2},
                                                             {"ssm.time_step_rank", 4},
                                                             {"ssm.state_size", 32},
                                                             {"ssm.conv_kernel", 4},
                                                             {"ssm.inner_size", 128},
                                                             {"expert_count", experts},
                                                             {"expert_used_count", 2},
                                                             {"expert_feed_forward_length", 64},
                                                             {"expert_shared_feed_forward_length", 96},
                                                             {"full_attention_interval", interval},
                                                             {"context_length", 2048}};
    for (const auto &pair : values)
        m[std::string("qwen3next.") + pair.first] = pair.second;
    m["qwen3next.rope.freq_base"] = 10000.0f;
    m["qwen3next.attention.layer_norm_rms_epsilon"] = 0.000001f;
    m["tokenizer.ggml.tokens"] = std::vector<std::string>(32, "x");
    m["tokenizer.ggml.model"] = std::string("gpt2");
    m["tokenizer.ggml.pre"] = std::string("qwen35");
    view["lora"].metadata["adapter.type"] = std::string("lora");
    Tensor rank;
    rank.ne = {{64, 4, 1, 1}};
    view["lora"].tensors["blk.0.ffn_gate_shexp.weight.lora_a"] = rank;
    auto& pr=view["prerouter"].metadata;
    std::vector<std::uint32_t> owners;
    for(std::uint32_t i=layers==40?6:layers==8?1:0;i+1<layers;++i) owners.push_back(i);
    pr["prerouter.owners"]=owners;
    pr["prerouter.feed_forward_length"]=std::uint32_t(layers==40?512:16);
    pr["prerouter.top_k"]=std::uint32_t(2);
    return view;
}
inline void set(gguf_context *g, const Metadata &metadata) {
    for (const auto &kv : metadata)
        std::visit(
            [&](const auto &v) {
                using T = std::decay_t<decltype(v)>;
#define SCALAR(TP, FN)                                                                                       \
    if constexpr (std::is_same_v<T, TP>)                                                                     \
        gguf_set_val_##FN(g, kv.first.c_str(), v);                                                           \
    else
                SCALAR(std::uint8_t, u8)
                SCALAR(std::int8_t, i8) SCALAR(std::uint16_t, u16) SCALAR(std::int16_t, i16)
                    SCALAR(std::uint32_t, u32) SCALAR(std::int32_t, i32) SCALAR(std::uint64_t, u64)
                        SCALAR(std::int64_t, i64) SCALAR(float, f32) SCALAR(double, f64) SCALAR(bool, bool)
#undef SCALAR
                            if constexpr (std::is_same_v<T, std::string>)
                                gguf_set_val_str(g, kv.first.c_str(), v.c_str());
                else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
                    std::vector<const char *> p;
                    for (const auto &s : v)
                        p.push_back(s.c_str());
                    gguf_set_arr_str(g, kv.first.c_str(), p.data(), p.size());
                }
                else if constexpr (std::is_same_v<T, std::vector<bool>>) {
                    std::vector<std::uint8_t> p;
                    for (bool b : v)
                        p.push_back(b);
                    gguf_set_arr_data(g, kv.first.c_str(), GGUF_TYPE_BOOL, p.data(), p.size());
                }
                else {
#define ARRAY(TP, TAG)                                                                                       \
    if constexpr (std::is_same_v<T, std::vector<TP>>)                                                        \
        gguf_set_arr_data(g, kv.first.c_str(), GGUF_TYPE_##TAG, v.data(), v.size());                         \
    else
                    ARRAY(std::uint8_t, UINT8)
                    ARRAY(std::int8_t, INT8) ARRAY(std::uint16_t, UINT16) ARRAY(std::int16_t, INT16)
                        ARRAY(std::uint32_t, UINT32) ARRAY(std::int32_t, INT32) ARRAY(std::uint64_t, UINT64)
                            ARRAY(std::int64_t, INT64) ARRAY(float, FLOAT32) ARRAY(double, FLOAT64)
#undef ARRAY
                    {}
                }
            },
            kv.second);
}
inline void gguf(const fs::path &p, const Metadata &metadata, const std::vector<TensorRequirement> &tensors) {
    std::unique_ptr<gguf_context, decltype(&gguf_free)> g(gguf_init_empty(), gguf_free);
    set(g.get(), metadata);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({ggml_tensor_overhead() * (tensors.size() + 1), nullptr, true}), ggml_free);
    CHECK(ctx);
    for (const auto &req : tensors) {
        auto type = req.type == "f32" ? GGML_TYPE_F32 : req.type == "f16" ? GGML_TYPE_F16 : GGML_TYPE_Q4_1;
        auto *t = ggml_new_tensor(ctx.get(), type, 4, req.shape.data());
        ggml_set_name(t, req.name.c_str());
        gguf_add_tensor(g.get(), t);
    }
    CHECK(gguf_write_to_file(g.get(), p.string().c_str(), true));
    std::uint64_t size = gguf_get_meta_size(g.get());
    for (std::int64_t i = 0; i < gguf_get_n_tensors(g.get()); ++i)
        size =
            std::max<std::uint64_t>(size, gguf_get_meta_size(g.get()) + gguf_get_tensor_offset(g.get(), i) +
                                              gguf_get_tensor_size(g.get(), i));
    fs::resize_file(p, size);
}
inline void assets(const fs::path &root, const Adapter &adapter, const fs::path &source,
                   const std::string &omitted = "") {
    auto view = metadata();
    auto d = adapter.describe_assets(view);
    CHECK(d);
    for (const auto &role : {"base", "lora", "prerouter"}) {
        std::vector<TensorRequirement> reqs;
        for (const auto &req : d.value().tensors)
            if (req.role == role && req.name != omitted)
                reqs.push_back(req);
        gguf(root / (std::string(role) + ".gguf"), view.at(role).metadata, reqs);
    }
    write(root / "model.json", "{\"format_version\":1,\"architecture\":\"edge0_35b\"}");
    write(root / "tokenizer.json", "{}");
    write(root / "generation_config.json", "{}");
    write(root / "provenance.json", "{}");
    fs::copy_file(source / "tests/fixtures/models/edge0_35b/chat_template.jinja",
                  root / "chat_template.jinja", fs::copy_options::overwrite_existing);
    catalog(root, {{"base", "base.gguf"},
                   {"lora", "lora.gguf"},
                   {"prerouter", "prerouter.gguf"},
                   {"tokenizer", "tokenizer.json"},
                   {"chat_template", "chat_template.jinja"},
                   {"generation_config", "generation_config.json"},
                   {"model", "model.json"},
                   {"provenance", "provenance.json"}});
}
template <class T> void error(const edge0::Result<T> &r, edge0::ErrorCategory cat, const std::string &text) {
    CHECK(!r);
    CHECK(r.error().category == cat);
    CHECK(r.error().message.find(text) != std::string::npos);
}
} // namespace fixture
