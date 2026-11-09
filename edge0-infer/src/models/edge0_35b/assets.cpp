#include "models/edge0_35b/assets.h"
#include "models/edge0_35b/text.h"
#include "models/edge0_35b/output.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
namespace edge0::models::edge0_35b {
bool use_ahead_prefill(const model::ModelOptions& options, std::int64_t block_tokens) {
    const auto& read=options.at("prefill_read");
    return block_tokens>1 && (read=="ahead" ||
        (read=="auto" && std::uint64_t(block_tokens)>=std::stoull(options.at("prefill_ahead_min_tokens"))));
}
namespace {
using namespace model;
struct Rejected : std::runtime_error {
    using std::runtime_error::runtime_error;
};
template <class T> const T &get(const Metadata &m, const std::string &key) {
    auto value = read_metadata<T>(m, key);
    if (!value)
        throw Rejected(value.error().message);
    return *value.value();
}
class Edge0Adapter final : public Adapter {
  public:
    Edge0Adapter() : Adapter({"base", "lora", "prerouter"}) {}
    std::string id() const override {
        return "edge0_35b";
    }
    TextDescription describe_text() const override { return edge0::models::edge0_35b::describe_text(); }
    std::unique_ptr<OutputParser> output_parser(const TextOptions& options) const override {
        return edge0::models::edge0_35b::output_parser(options);
    }
    TextResult text_format(const std::vector<Message> &messages, const TextOptions &options) const override {
        return edge0::models::edge0_35b::text_format(messages, options);
    }
    Result<WeightDescription> describe_weight_storage(const AssetView& assets,
                                                      const ModelOptions& options) const override {
        for (const auto& item : options) {
            if (item.first == "routing" && item.second != "actual" && item.second != "predicted") {
                return Error{ErrorCategory::invalid_argument, "Invalid routing mode"};
            }
            if (item.first == "lora" && item.second != "on" && item.second != "off")
                return Error{ErrorCategory::invalid_argument, "LoRA must be on or off"};
        }
        const auto option = [&](const std::string& name,const std::string& fallback) { auto it=options.find(name); return it==options.end()?fallback:it->second; };
        const auto expert_read=option("expert_read","cached");
        if(expert_read!="cached" && expert_read!="direct") return Error{ErrorCategory::invalid_argument,"Invalid expert_read option"};
        if(expert_read=="direct" && !weights::supports_direct_read()) return Error{ErrorCategory::unsupported,"Direct expert reads are unavailable"};
        const auto read_mode=expert_read=="direct"?weights::ReadMode::direct:weights::ReadMode::cached;
        const auto mode=option("experts","slots"), read=option("prefill_read","auto");
        if ((mode!="slots" && mode!="resident") || (read!="used" && read!="layer" && read!="ahead" && read!="auto"))
            return Error{ErrorCategory::invalid_argument,"Invalid experts or prefill_read option"};
        if (mode=="resident" && options.count("expert_slots"))
            return Error{ErrorCategory::invalid_argument,"expert_slots is invalid with resident experts"};
        std::size_t capacity=0, expert_count=0;
        std::optional<ModelDefinition> definition;
        if (mode=="slots") {
            auto base=assets.find("base");
            if(base==assets.end()) return Error{ErrorCategory::incompatible,"Missing weight role: base"};
            definition=ModelDefinition{};
            for(const auto& entry:std::vector<std::pair<std::string,std::string>>{{"n_layer","block_count"},{"n_expert","expert_count"},{"top_k","expert_used_count"}}) {
                auto value=read_metadata<std::uint32_t>(base->second.metadata,"qwen3next."+entry.second);
                if(!value) return value.error();
                if(!*value.value()) return Error{ErrorCategory::incompatible,"Invalid expert metadata"};
                definition->parameters[entry.first]=*value.value();
            }
            for(std::uint32_t layer=0;layer<definition->parameters.at("n_layer");++layer)
                for(const auto* projection:{"gate","up","down"}) definition->expert_tensors.push_back({layer,projection,"blk."+std::to_string(layer)+".ffn_"+projection+"_exps.weight"});
            expert_count=definition->parameters.at("n_expert");
            const auto text=option("expert_slots",std::to_string(definition->parameters.at("top_k")));
            auto parsed=parse_count(text);
            if(!parsed || parsed.value()<definition->parameters.at("top_k") || parsed.value()>expert_count)
                return Error{ErrorCategory::invalid_argument,"expert_slots must be an integer from top_k to expert count"};
            capacity=static_cast<std::size_t>(parsed.value());
        }
        const bool lora = !options.count("lora") || options.at("lora") == "on";
        const auto routing=option("routing","predicted");
        if(routing=="predicted" && !lora)
            return Error{ErrorCategory::invalid_argument,"predicted requires lora=on"};
        WeightDescription result;
        result.resident_stages=StorageStages::both;
        result.pool_stages=StorageStages::decode;
        result.temporary_stages=StorageStages::prefill;
        result.temporary_copies=(read=="ahead" || read=="auto")?2:1;
        result.configuration = {{"routing", {"actual", "predicted"}, "predicted", false, "actual"},
                                {"lora", {"on", "off"}, "on"}, {"experts", {"slots","resident"}, "slots"},
                                {"prefill_read", {"used","layer","ahead","auto"}, "auto"},
                                {"prefill_ahead_min_tokens", {}, "3000", true},
                                {"expert_read", {"cached","direct"}, "cached"}};
        if (mode=="slots") {
            result.configuration.push_back({"expert_slots",{},std::to_string(definition->parameters.at("top_k"))});
            result.pools.resize(definition->parameters.at("n_layer"));
            for (std::size_t layer=0;layer<result.pools.size();++layer) {
                auto& pool=result.pools[layer]; pool.layer=static_cast<std::uint32_t>(layer); pool.slots=capacity; pool.experts.resize(expert_count);
            }
        }
        for (const auto* role : {"base", "lora", "prerouter"}) {
            if (std::string(role) == "prerouter" && routing == "actual") continue;
            if (std::string(role) == "lora" && !lora) continue;
            auto index = assets.find(role);
            if (index == assets.end())
                return Error{ErrorCategory::incompatible, std::string("Missing weight role: ") + role};
            for (const auto& item : index->second.tensors) {
                const auto& tensor = item.second;
                ggml_type type = GGML_TYPE_COUNT;
                for (int candidate = 0; candidate < GGML_TYPE_COUNT; ++candidate) {
                    const auto t = static_cast<ggml_type>(candidate);
                    if (tensor.type == ggml_type_name(t)) { type = t; break; }
                }
                if (type == GGML_TYPE_COUNT)
                    return Error{ErrorCategory::incompatible, "Unknown tensor type: " + tensor.type};
                std::size_t dimensions = 4;
                while (dimensions > 1 && tensor.ne[dimensions - 1] == 1) --dimensions;
                if (dimensions > 3)
                    return Error{ErrorCategory::incompatible, "Resident tensor has more than three dimensions"};
                auto slice = locate_slice(tensor, 0, static_cast<std::uint64_t>(tensor.ne[dimensions - 1]));
                if (!slice) return slice.error();
                exec::Layout layout{type, {tensor.ne.begin(), tensor.ne.begin() + dimensions}};
                if (mode=="slots" && std::string(role)=="base") {
                    const auto found=std::find_if(definition->expert_tensors.begin(),definition->expert_tensors.end(),[&](const ExpertTensor& e) { return e.name==tensor.name; });
                    if(found!=definition->expert_tensors.end()) {
                        const std::size_t projection=found->projection=="gate"?0:found->projection=="up"?1:2;
                        auto& pool=result.pools.at(found->layer);
                        exec::Layout single{type,{tensor.ne[0],tensor.ne[1]}};
                        pool.projections[projection]=single;
                        for(std::size_t expert=0;expert<expert_count;++expert) {
                            auto part=locate_slice(tensor,expert,1); if(!part) return part.error();
                            auto source=part.value();
                            pool.experts[expert][projection]={source.path,source.offset,static_cast<std::size_t>(source.bytes),single,read_mode};
                        }
                        continue;
                    }
                }
                result.resident.push_back({std::string(role) + ":" + tensor.name, role, tensor.name,
                    {slice.value().path, slice.value().offset, static_cast<std::size_t>(slice.value().bytes), layout}});
            }
        }
        if (mode=="slots") for(const auto& layout:result.pools.front().projections) result.temporary.push_back({layout,expert_count});
        return result;
    }
    Result<ModelDefinition> describe_assets(const AssetView &assets) const override {
        try {
            if (!assets.count("base") || !assets.count("lora") || !assets.count("prerouter"))
                throw Rejected("missing role base, lora or prerouter");
            const auto &b = assets.at("base").metadata;
            const auto &l = assets.at("lora");
            if (get<std::string>(b, "general.architecture") != "qwen3next")
                throw Rejected("general.architecture: expected qwen3next");
            if (get<std::string>(l.metadata, "adapter.type") != "lora")
                throw Rejected("adapter.type: expected lora");
            ModelDefinition d;
            d.required_roles = {"base", "lora", "prerouter", "tokenizer", "chat_template", "generation_config"};
            auto u = [&](const char *short_name, const char *suffix) {
                std::string key = "qwen3next.";
                key += suffix;
                auto n = get<std::uint32_t>(b, key);
                if (!n)
                    throw Rejected(key + ": expected positive value");
                d.parameters[short_name] = n;
                return std::int64_t(n);
            };
            auto n_layer = u("n_layer", "block_count"), n_embd = u("n_embd", "embedding_length"),
                 n_head = u("n_head", "attention.head_count"),
                 n_head_kv = u("n_head_kv", "attention.head_count_kv"),
                 head_dim = u("head_dim", "attention.key_length"),
                 rope_dim = u("rope_dim", "rope.dimension_count"),
                 n_key_head = u("n_key_head", "ssm.group_count"),
                 n_value_head = u("n_value_head", "ssm.time_step_rank"),
                 state_dim = u("state_dim", "ssm.state_size"),
                 conv_kernel = u("conv_kernel", "ssm.conv_kernel"), n_expert = u("n_expert", "expert_count"),
                 top_k = u("top_k", "expert_used_count"), n_ff = u("n_ff", "expert_feed_forward_length"),
                 n_shared = u("n_shared", "expert_shared_feed_forward_length"),
                 interval = u("full_attention_interval", "full_attention_interval");
            auto value_length = u("value_length", "attention.value_length");
            u("inner_size", "ssm.inner_size");
            u("context_length", "context_length");
            for (const auto *key :
                 {"qwen3next.rope.freq_base", "qwen3next.attention.layer_norm_rms_epsilon"}) {
                auto v = get<float>(b, key);
                if (!std::isfinite(v) || v <= 0)
                    throw Rejected(std::string(key) + ": expected finite positive value");
                d.numeric_parameters[key] = v;
            }
            if (value_length != head_dim)
                throw Rejected("qwen3next.attention.value_length: must equal key_length");
            if (n_value_head != 2 * n_key_head)
                throw Rejected("qwen3next.ssm.time_step_rank: must equal twice group_count");
            if (top_k > n_expert)
                throw Rejected("qwen3next.expert_used_count: exceeds expert_count");
            if (rope_dim > head_dim)
                throw Rejected("qwen3next.rope.dimension_count: exceeds key_length");
            auto n_vocab =
                static_cast<std::int64_t>(get<std::vector<std::string>>(b, "tokenizer.ggml.tokens").size());
            if (!n_vocab)
                throw Rejected("tokenizer.ggml.tokens: empty vocabulary");
            d.parameters["n_vocab"] = static_cast<std::uint64_t>(n_vocab);
            auto rank_tensor = l.tensors.find("blk.0.ffn_gate_shexp.weight.lora_a");
            if (rank_tensor == l.tensors.end())
                throw Rejected("blk.0.ffn_gate_shexp.weight.lora_a: missing LoRA rank tensor");
            auto lora_rank = rank_tensor->second.ne[1];
            if (lora_rank <= 0)
                throw Rejected("lora_rank: expected positive");
            d.parameters["lora_rank"] = static_cast<std::uint64_t>(lora_rank);
            auto product = [](std::initializer_list<std::int64_t> values, const char *key) {
                std::int64_t value = 1;
                for (auto n : values) {
                    if (n <= 0 || value > std::numeric_limits<std::int64_t>::max() / n)
                        throw Rejected(std::string(key) + ": shape product overflow");
                    value *= n;
                }
                return value;
            };
            product({head_dim, n_head, 2}, "qwen3next.attention.head_count");
            product({head_dim, n_head_kv}, "qwen3next.attention.head_count_kv");
            product({state_dim, 2 * n_key_head + n_value_head}, "qwen3next.ssm.state_size");
            product({state_dim, n_value_head}, "qwen3next.ssm.time_step_rank");
            product({2, lora_rank}, "lora_rank");
            auto recurrent = b.find("qwen3next.attention.recurrent_layers");
            const std::vector<bool> *flags = nullptr;
            if (recurrent != b.end()) {
                flags = &get<std::vector<bool>>(b, recurrent->first);
                if (flags->size() != static_cast<std::size_t>(n_layer))
                    throw Rejected(recurrent->first + ": length mismatch");
            }
            auto add = [&](const std::string &role, const std::string &name, const std::string &type,
                           std::initializer_list<std::int64_t> shape) {
                TensorRequirement req{role, name, type, {{1, 1, 1, 1}}};
                product(shape, name.c_str());
                std::copy(shape.begin(), shape.end(), req.shape.begin());
                d.tensors.push_back(std::move(req));
            };
            add("base", "token_embd.weight", "q4_1", {n_embd, n_vocab});
            add("base", "output.weight", "q4_1", {n_embd, n_vocab});
            add("base", "output_norm.weight", "f32", {n_embd});
            for (std::int64_t i = 0; i < n_layer; ++i) {
                bool attn = (i + 1) % interval == 0;
                if (flags && (*flags)[static_cast<std::size_t>(i)] == attn)
                    throw Rejected(
                        "qwen3next.attention.recurrent_layers: inconsistent with full_attention_interval");
                d.layer_groups[attn ? "attention" : "deltanet"].push_back(static_cast<std::uint32_t>(i));
                auto prefix = "blk." + std::to_string(i) + ".";
                auto a = [&](const std::string &name, const std::string &type,
                             std::initializer_list<std::int64_t> dims) {
                    add("base", prefix + name, type, dims);
                };
                auto paired = [&](const std::string &name, std::int64_t in, std::int64_t out,
                                  std::int64_t rank) {
                    a(name, "q4_1", {in, out});
                    add("lora", prefix + name + ".lora_a", "f16", {in, rank});
                    add("lora", prefix + name + ".lora_b", "f16", {rank, out});
                };
                a("attn_norm.weight", "f32", {n_embd});
                a("post_attention_norm.weight", "f32", {n_embd});
                if (attn) {
                    paired("attn_q.weight", n_embd, head_dim * n_head * 2, lora_rank);
                    paired("attn_k.weight", n_embd, head_dim * n_head_kv, lora_rank);
                    paired("attn_v.weight", n_embd, head_dim * n_head_kv, lora_rank);
                    paired("attn_output.weight", head_dim * n_head, n_embd, lora_rank);
                    a("attn_q_norm.weight", "f32", {head_dim});
                    a("attn_k_norm.weight", "f32", {head_dim});
                } else {
                    paired("attn_qkv.weight", n_embd, state_dim * (2 * n_key_head + n_value_head), lora_rank);
                    paired("attn_gate.weight", n_embd, state_dim * n_value_head, lora_rank);
                    paired("ssm_ba.weight", n_embd, 2 * n_value_head, 2 * lora_rank);
                    paired("ssm_out.weight", state_dim * n_value_head, n_embd, lora_rank);
                    a("ssm_conv1d.weight", "f32", {conv_kernel, state_dim * (2 * n_key_head + n_value_head)});
                    a("ssm_dt.bias", "f32", {n_value_head});
                    a("ssm_a", "f32", {n_value_head});
                    a("ssm_norm.weight", "f32", {state_dim});
                }
                a("ffn_gate_inp.weight", "f32", {n_embd, n_expert});
                for (const auto *projection : {"gate", "up", "down"}) {
                    auto name = std::string("ffn_") + projection + "_exps.weight";
                    bool down = std::string(projection) == "down";
                    a(name, "q4_1", {down ? n_ff : n_embd, down ? n_embd : n_ff, n_expert});
                    d.expert_tensors.push_back({static_cast<std::uint32_t>(i), projection, prefix + name});
                }
                paired("ffn_gate_shexp.weight", n_embd, n_shared, lora_rank);
                paired("ffn_up_shexp.weight", n_embd, n_shared, lora_rank);
                paired("ffn_down_shexp.weight", n_shared, n_embd, lora_rank);
                a("ffn_gate_inp_shexp.weight", "f32", {n_embd});
            }
            const auto& pr=assets.at("prerouter").metadata;
            const auto& owners=get<std::vector<std::uint32_t>>(pr,"prerouter.owners");
            const auto hidden=get<std::uint32_t>(pr,"prerouter.feed_forward_length");
            if(get<std::uint32_t>(pr,"prerouter.top_k")!=top_k)
                throw Rejected("prerouter.top_k: must equal expert_used_count");
            if(owners.empty() || owners.front()>=n_layer-1 || owners.size()!=std::size_t(n_layer-1-owners.front()))
                throw Rejected("prerouter.owners: expected contiguous owners ending at n_layer-2");
            if(!hidden) throw Rejected("prerouter.feed_forward_length: expected positive");
            // Fixed published profiles; synthetic shapes retain explicit metadata.
            if((n_layer==40 && (owners.front()!=6 || hidden!=512)) ||
               (n_layer==8 && (owners.front()!=1 || hidden!=16)))
                throw Rejected("prerouter.owners/feed_forward_length: profile mismatch");
            d.parameters["prerouter_first_owner"]=owners.front();
            d.parameters["prerouter_hidden"]=hidden;
            for(std::size_t j=0;j<owners.size();++j) {
                if(owners[j]!=owners.front()+j) throw Rejected("prerouter.owners: noncontiguous owners");
                const auto prefix="blk."+std::to_string(owners[j])+".prerouter_";
                add("prerouter",prefix+"fc1.weight","f16",{n_embd+2*n_expert,hidden});
                add("prerouter",prefix+"fc2.weight","f16",{hidden,n_expert});
                add("prerouter",prefix+"linear_init.weight","f16",{n_embd+2*n_expert,n_expert});
            }
            d.vocabulary_size=d.parameters.at("n_vocab");
            d.context_length=d.parameters.at("context_length");
            d.layer_count=d.parameters.at("n_layer");
            d.expert_count=d.parameters.at("n_expert");
            d.experts_per_token=d.parameters.at("top_k");
            d.layer_groups.try_emplace("attention");
            d.layer_groups.try_emplace("deltanet");
            for (auto& tensor : d.expert_tensors) tensor.experts=d.expert_count;
            return d;
        } catch (const Rejected &e) {
            return Error{ErrorCategory::incompatible, e.what()};
        }
    }
};
} // namespace
const model::Adapter &adapter() {
    static const Edge0Adapter value;
    return value;
}
} // namespace edge0::models::edge0_35b
