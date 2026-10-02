#include "registry.hpp"

namespace e0n {

std::vector<int> TierRouting::predicted() const {
  std::vector<int> v;
  for (int l = first_head_consumer; l <= last_head_consumer; l++) v.push_back(l);
  return v;
}

std::vector<int> TierRouting::routed() const {
  std::vector<int> v;
  for (int l = dense_prefix; l < first_head_consumer; l++) v.push_back(l);
  // 35b: last layer L39 does not consume predictions; 8b L23 == last_head_consumer
  // is consumed. The last-layer exception is qwen-family only.
  if (family == RouterFamily::kSoftmaxTopk) v.push_back(num_layers - 1);
  return v;
}

bool tier_routing(const std::string &tier_id, TierRouting &out) {
  if (tier_id == "edge0-35b") {
    out = TierRouting{};
    out.tier = tier_id;
    out.family = RouterFamily::kSoftmaxTopk;
    out.num_experts = 256;
    out.top_k = 4;
    out.norm_topk_prob = true;
    out.n_group = 0; out.topk_group = 0; out.routed_scaling = 1.0;
    out.expert_bias = false;
    out.num_layers = 40;
    out.first_head_consumer = 7; out.last_head_consumer = 38;
    out.first_owner = 6; out.last_owner = 38;
    out.dense_prefix = 0;
    out.prerouter_hidden = 512;
    out.head_file = "prerouter_edge0_35b.safetensors";
    out.lora_file = "lora_edge0_35b.safetensors";
    out.heads_baked_in_model = false;
    out.head_feature_teacher = false;  // qwen: block patch captures the executed set
    out.prefill_chunk = 2048;          // models/edge0_35b/__init__.py:63
    out.hot_window = 4;                // :64
    out.prefill_hot = 32;              // staged_k4(options.py:134)
    out.full_layer_prefill = false;    // options.py:133
    out.async_eval_prefill = false;    // qwen.py:176 gate=full_layer ⇒ False
    return true;
  }
  if (tier_id == "edge0-8b") {
    out = TierRouting{};
    out.tier = tier_id;
    out.family = RouterFamily::kSigmoidGroup;
    out.num_experts = 128;
    out.top_k = 8;
    out.norm_topk_prob = true;
    out.n_group = 8; out.topk_group = 4; out.routed_scaling = 2.5;
    out.expert_bias = true;
    out.num_layers = 24;
    out.first_head_consumer = 8; out.last_head_consumer = 23;
    out.first_owner = 7; out.last_owner = 22;
    out.dense_prefix = 1;
    out.prerouter_hidden = 512;
    out.head_file = "prerouter_edge0_8b.safetensors";
    out.lora_file = "lora_edge0_8b.safetensors";
    out.heads_baked_in_model = true;
    out.head_feature_teacher = true;  // ling: teacher top-8 feeds features
    out.prefill_chunk = 2048;          // models/edge0_8b/__init__.py:71
    out.hot_window = 1;
    out.prefill_hot = 0;               // prod_k8(options.py:189)
    out.full_layer_prefill = true;     // options.py:189
    out.async_eval_prefill = true;     // ling.py:217 gate=full_layer ⇒ True
    return true;
  }
  return false;
}

std::string routing_config_mismatch(const TierRouting &r, int64_t cfg_num_experts,
                                    int64_t cfg_num_layers) {
  if (cfg_num_experts > 0 && cfg_num_experts != r.num_experts)
    return "num_experts: config=" + std::to_string(cfg_num_experts) +
           " registry=" + std::to_string(r.num_experts);
  if (cfg_num_layers > 0 && cfg_num_layers != r.num_layers)
    return "num_layers: config=" + std::to_string(cfg_num_layers) +
           " registry=" + std::to_string(r.num_layers);
  return "";
}

}  // namespace e0n
