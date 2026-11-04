#include "edge0/assets.h"
#include "edge0/engine.h"
#include "models/edge0_35b/assets.h"
#include "tests/support/model_fixture.h"
int main(int argc, char **argv) {
    using namespace fixture;
    CHECK(argc >= 2);
    const auto &adapter = edge0::models::edge0_35b::adapter();
    if (argc >= 3) {
        fs::create_directories(argv[2]);
        assets(argv[2], adapter, argv[1], argc == 4 ? argv[3] : "");
        return 0;
    }
    Temp t;
    assets(t.path, adapter, argv[1]);
    auto r = edge0::inspect_assets({t.path.string(), false});
    CHECK(r);
    CHECK(r.value().layer_groups.at("attention") == std::vector<std::uint32_t>{1});
    CHECK(r.value().layer_groups.at("deltanet") == std::vector<std::uint32_t>{0});
    CHECK(r.value().expert_slices.size() == 12);
    CHECK((r.value().unused_roles == std::vector<std::string>{"model", "provenance"}));
    CHECK(r.value().integrity == "size_only");
    CHECK(sha256_file((t.path / "chat_template.jinja").string()).value() ==
          "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259");
    auto index = index_tensors(file(t.path / "base.gguf"));
    CHECK(index);
    CHECK(*read_metadata<std::string>(index.value().metadata, "tokenizer.ggml.model").value() == "gpt2");
    CHECK(*read_metadata<std::string>(index.value().metadata, "tokenizer.ggml.pre").value() == "qwen35");
    CHECK(edge0::inspect_assets({t.path.string(), true}).value().integrity == "digest");
    auto caps = edge0::query_capabilities(false);
    std::string cpu;
    for (const auto &device : caps.devices)
        if (device.backend == "cpu")
            cpu = device.name;
    CHECK(!cpu.empty());
    auto runtime = edge0::open_runtime({{cpu}, {}});
    CHECK(runtime);
    CHECK(edge0::inspect_assets({t.path.string(), false}));
    CHECK(edge0::finish_runtime_use(runtime.value()));
    assets(t.path, adapter, argv[1], "blk.0.ffn_gate_exps.weight");
    error(edge0::inspect_assets({t.path.string(), false}), edge0::ErrorCategory::incompatible,
          "blk.0.ffn_gate_exps.weight");
    std::cout << "small assets, text identity, slices, digest and live runtime isolation passed\n";
}
