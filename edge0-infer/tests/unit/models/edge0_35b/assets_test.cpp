#include "models/edge0_35b/assets.h"
#include "models/edge0_35b/compute.h"
#include <algorithm>
#include <set>
#include "model/loading.h"
#include "tests/support/runtime.h"
#include "tests/support/model_fixture.h"
int main(int argc, char **) {
    using namespace fixture;
    using edge0::ErrorCategory;
    const auto &adapter = edge0::models::edge0_35b::adapter();
    CHECK(adapter.id() == "edge0_35b");
    auto profile = [](bool full) {
        auto v = metadata(full ? 40 : 8, full ? 256 : 16, 4);
        auto &m = v["base"].metadata;
        for (const auto &entry : std::vector<std::pair<std::string, std::uint32_t>>{
                 {"embedding_length", full ? 2048U : 64U},
                 {"attention.head_count", full ? 16U : 2U},
                 {"attention.head_count_kv", full ? 2U : 1U},
                 {"attention.key_length", 256},
                 {"attention.value_length", 256},
                 {"rope.dimension_count", 64},
                 {"ssm.group_count", full ? 16U : 2U},
                 {"ssm.time_step_rank", full ? 32U : 4U},
                 {"ssm.state_size", 128},
                 {"expert_used_count", 4},
                 {"expert_feed_forward_length", full ? 512U : 64U},
                 {"expert_shared_feed_forward_length", full ? 512U : 64U}})
            m["qwen3next." + entry.first] = entry.second;
        m["tokenizer.ggml.tokens"] = std::vector<std::string>(248320, "x");
        v["lora"].tensors.begin()->second.ne[1] = full ? 16 : 4;
        v["prerouter"].metadata["prerouter.top_k"]=std::uint32_t(4);
        return v;
    };
    auto declarations = [&](bool full) {
        auto v = profile(full);
        const auto definition = adapter.describe_assets(v);
        CHECK(definition);
        auto states = edge0::models::edge0_35b::describe_states(definition.value());
        CHECK(states);
        CHECK(states.value().evolves_as_whole);
        CHECK(states.value().supports_save_restore);
        const auto& p = definition.value().parameters;
        CHECK(states.value().checkpoint_bytes == definition.value().layer_groups.at("deltanet").size() * sizeof(float) *
            ((p.at("conv_kernel") - 1) * p.at("state_dim") * (2 * p.at("n_key_head") + p.at("n_value_head")) +
             p.at("state_dim") * p.at("state_dim") * p.at("n_value_head")));
        CHECK(states.value().kv.size() == (full ? 20 : 4));
        for (std::size_t i = 0; i < states.value().kv.size(); ++i) {
            const auto& kv = states.value().kv[i];
            CHECK(kv.name == "attn." + std::to_string((i / 2) * 4 + 3) + (i % 2 ? ".v" : ".k"));
            CHECK(kv.type == GGML_TYPE_F16 && kv.token_axis == 1 && kv.alignment == 256);
            CHECK(kv.visibility == edge0::session::Visibility::causal);
            CHECK((kv.token_shape == std::vector<std::int64_t>{256, full ? 2 : 1}));
        }
        v["base"].tensors.clear(); v["lora"].tensors.clear();
        for (const auto& req : definition.value().tensors) {
            edge0::model::Tensor tensor;
            tensor.name = req.name; tensor.role = req.role; tensor.path = req.role + ".gguf";
            tensor.ne = req.shape; tensor.type = req.type; tensor.offset = 128;
            const auto type = req.type == "q4_1" ? GGML_TYPE_Q4_1 : req.type == "f16" ? GGML_TYPE_F16 : GGML_TYPE_F32;
            tensor.block_size = ggml_blck_size(type);
            tensor.nb[0] = ggml_type_size(type);
            tensor.nb[1] = ggml_row_size(type, tensor.ne[0]);
            for (int axis = 2; axis < 4; ++axis) tensor.nb[axis] = tensor.nb[axis-1] * tensor.ne[axis-1];
            tensor.bytes = tensor.nb[3] * tensor.ne[3];
            v[req.role].tensors.emplace(req.name, tensor);
        }
        for (bool lora : {true, false}) {
            auto weights = adapter.describe_weight_storage(v, {{"routing","actual"},{"lora", lora ? "on" : "off"},{"experts","resident"}});
            CHECK(weights);
            CHECK(weights.value().resident.size() == (full ? (lora ? 1263 : 703) : (lora ? 255 : 143)));
            std::set<std::string> ids;
            for (const auto& w : weights.value().resident) {
                CHECK(ids.insert(w.id).second && w.id == w.role + ":" + w.name);
                CHECK(lora || w.role == "base");
                const auto& t = v.at(w.role).tensors.at(w.name);
                CHECK(w.source.bytes == t.bytes && w.source.offset == t.offset && w.source.path == t.path);
                CHECK(t.type == ggml_type_name(w.source.layout.type));
                for (std::size_t axis = 0; axis < w.source.layout.dimensions.size(); ++axis)
                    CHECK(w.source.layout.dimensions[axis] == t.ne[axis]);
            }
        }
        auto slots=test::take(adapter.describe_weight_storage(v,{{"routing","actual"},{"prefill_read","used"}}));
        auto resident=test::take(adapter.describe_weight_storage(v,{{"routing","actual"},{"experts","resident"}}));
        CHECK(slots.pools.size()==(full?40:8) && slots.temporary.size()==3);
        CHECK(slots.resident.size()+slots.pools.size()*3==resident.resident.size());
        for(const auto& pool:slots.pools) {
            CHECK(pool.slots==4 && pool.experts.size()==(full?256:16));
            for(std::size_t expert=0;expert<pool.experts.size();++expert) for(std::size_t projection=0;projection<3;++projection) {
                const auto& first=pool.experts[0][projection]; const auto& source=pool.experts[expert][projection];
                CHECK(source.offset==first.offset+expert*first.bytes && source.bytes==first.bytes);
            }
        }
        auto backend=test::backend(0,nullptr);
        auto estimate=test::take(edge0::model::estimate_weights(slots,backend));
        auto nonexpert=slots; nonexpert.pools.clear(); nonexpert.temporary.clear();
        const auto base=test::take(edge0::model::estimate_weights(nonexpert,backend));
        const auto alignment=ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend->handle()));
        auto array_bytes=[&](std::size_t regions) {
            std::size_t bytes=0;
            for(const auto& source:slots.pools[0].experts[0]) bytes+=(source.bytes*regions+alignment-1)/alignment*alignment+(1+regions)*ggml_tensor_overhead();
            return bytes;
        };
        CHECK(estimate.pool_bytes==slots.pools.size()*array_bytes(4));
        CHECK(estimate.temporary_bytes==array_bytes(full?256:16));
        CHECK(estimate.total_bytes==base.total_bytes+estimate.pool_bytes+estimate.temporary_bytes);
        CHECK(estimate.total_bytes<test::take(edge0::model::estimate_weights(resident,backend)).total_bytes);
        for(const auto* value:{"3","abc","4.5","999999999999999999999999999999"})
            error(adapter.describe_weight_storage(v,{{"expert_slots",value}}),ErrorCategory::invalid_argument,"expert_slots");
        error(adapter.describe_weight_storage(v,{{"expert_slots",full?"257":"17"}}),ErrorCategory::invalid_argument,"expert_slots");
        error(adapter.describe_weight_storage(v,{{"expert_slots","4"},{"experts","resident"}}),ErrorCategory::invalid_argument,"expert_slots");
        error(adapter.describe_weight_storage(v,{{"experts","bad"}}),ErrorCategory::invalid_argument,"Invalid");
        error(adapter.describe_weight_storage(v,{{"prefill_read","bad"}}),ErrorCategory::invalid_argument,"Invalid");
        for(const auto* routing:{"predicted"}) {
            auto routed=test::take(adapter.describe_weight_storage(v,{{"routing",routing}}));
            CHECK(routed.resident.size()==slots.resident.size()+(full?99:18));
        }
        error(adapter.describe_weight_storage(v,{{"routing","prefetch"}}),ErrorCategory::invalid_argument,"routing");
        error(adapter.describe_weight_storage(v,{{"routing","predicted"},{"lora","off"}}),ErrorCategory::invalid_argument,"predicted");
        error(adapter.describe_weight_storage(v, {{"routing", "unknown"}}), ErrorCategory::invalid_argument, "routing");
        // Unknown option names are rejected by model::load, covered by its loading tests.
        error(adapter.describe_weight_storage(v, {{"routing","actual"},{"lora", "bad"}}), ErrorCategory::invalid_argument, "LoRA");
        v.erase("lora");
        error(adapter.describe_weight_storage(v, {}), ErrorCategory::incompatible, "lora");
        CHECK(adapter.describe_weight_storage(v, {{"routing","actual"},{"lora", "off"}}));
    };
    declarations(false); declarations(true);
    auto view = profile(true);
    auto check = [&](const AssetView &v) {
        auto d = adapter.describe_assets(v);
        CHECK(d);
        return d.value();
    };
    auto d = check(view);
    CHECK(d.layer_groups.at("attention").size() == 10 && d.layer_groups.at("deltanet").size() == 30);
    for (unsigned i = 0; i < 10; ++i)
        CHECK(d.layer_groups.at("attention")[i] == i * 4 + 3);
    auto count = [](const ModelDefinition &def, const std::string &role) {
        return std::count_if(def.tensors.begin(), def.tensors.end(),
                             [&](const auto &t) { return t.role == role; });
    };
    CHECK(count(d, "base") == 703 && count(d, "lora") == 560);
    auto tiny = profile(false);
    d = check(tiny);
    CHECK((d.layer_groups.at("attention") == std::vector<std::uint32_t>{3, 7}));
    CHECK(count(d, "base") == 143 && count(d, "lora") == 112);
    for (const auto &kv : view.at("base").metadata) {
        if (kv.first == "tokenizer.ggml.model" || kv.first == "tokenizer.ggml.pre")
            continue;
        auto bad = view;
        bad["base"].metadata.erase(kv.first);
        error(adapter.describe_assets(bad), ErrorCategory::incompatible, kv.first);
    }
    for (const auto &key : {"qwen3next.ssm.time_step_rank", "qwen3next.expert_used_count",
                            "qwen3next.attention.value_length", "qwen3next.rope.dimension_count"}) {
        auto bad = view;
        bad["base"].metadata[key] = std::uint32_t(999);
        error(adapter.describe_assets(bad), ErrorCategory::incompatible, key);
    }
    {
        auto bad = view;
        bad["base"].metadata["general.architecture"] = std::string("edge0_35b");
        error(adapter.describe_assets(bad), ErrorCategory::incompatible, "general.architecture");
        bad = view;
        bad["lora"].metadata["adapter.type"] = std::string("other");
        error(adapter.describe_assets(bad), ErrorCategory::incompatible, "adapter.type");
        bad = view;
        bad["base"].metadata["qwen3next.block_count"] = std::string("40");
        error(adapter.describe_assets(bad), ErrorCategory::incompatible, "type mismatch");
        bad = view;
        bad["base"].metadata["qwen3next.attention.recurrent_layers"] = std::vector<bool>(40, false);
        error(adapter.describe_assets(bad), ErrorCategory::incompatible, "recurrent_layers");
        std::vector<bool> recurrent;
        for (unsigned i = 0; i < 40; ++i)
            recurrent.push_back((i + 1) % 4 != 0);
        bad["base"].metadata["qwen3next.attention.recurrent_layers"] = recurrent;
        CHECK(adapter.describe_assets(bad));
    }
    {
        auto large_interval = view;
        large_interval["base"].metadata["qwen3next.full_attention_interval"] = std::uint32_t(1000000001);
        CHECK(check(large_interval).layer_groups.at("attention").empty());
    }
    if (argc > 1) {
        auto custom = metadata(5, 7, 3);
        auto &m = custom["base"].metadata;
        for (const auto &kv :
             std::vector<std::pair<std::string, std::uint32_t>>{{"embedding_length", 96},
                                                                {"attention.head_count", 5},
                                                                {"attention.head_count_kv", 3},
                                                                {"attention.key_length", 64},
                                                                {"attention.value_length", 64},
                                                                {"rope.dimension_count", 48},
                                                                {"ssm.group_count", 3},
                                                                {"ssm.time_step_rank", 6},
                                                                {"ssm.state_size", 16},
                                                                {"ssm.conv_kernel", 5},
                                                                {"expert_feed_forward_length", 128},
                                                                {"expert_shared_feed_forward_length", 160},
                                                                {"expert_used_count", 4}})
            m["qwen3next." + kv.first] = kv.second;
        m["tokenizer.ggml.tokens"] = std::vector<std::string>(41, "x");
        custom["lora"].tensors.begin()->second.ne[1] = 7;
        custom["prerouter"].metadata["prerouter.top_k"]=std::uint32_t(4);
        d = check(custom);
        for (const auto &req : d.tensors) {
            std::cout << req.role << '\t' << req.name << '\t' << req.type;
            for (auto dim : req.shape)
                std::cout << '\t' << dim;
            std::cout << '\n';
        }
    } else
        std::cout << "edge0_35b profiles, metadata invariants and missing-key matrix passed\n";
}
