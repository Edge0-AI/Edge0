// Runtime routing facts that are not in config.json (config is only a cross-check).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace e0n {

enum class RouterFamily { kSoftmaxTopk, kSigmoidGroup };

struct TierRouting {
  std::string tier;              // "edge0-8b" / "edge0-35b" (= tier_id)
  RouterFamily family;
  int64_t num_experts;
  int top_k;                     // runtime (35b=4, not config's 8)
  bool norm_topk_prob;           // true on both tiers
  int n_group, topk_group;
  double routed_scaling;
  bool expert_bias;              // live gate scores include bias; head path does not
  int num_layers;
  int first_head_consumer;       // 35b 7 / 8b 8
  int last_head_consumer;        // 35b 38 / 8b 23 (35b layer 39 is not a consumer)
  int first_owner, last_owner;   // 35b 6/38 / 8b 7/22
  int dense_prefix;              // leading dense (no MoE) layers: 8b=1 (L0); 35b=0
  int prerouter_hidden;          // 512 on both live-head tiers
  std::string head_file;
  std::string lora_file;
  bool heads_baked_in_model;
  // Head-feature one-hot: false = executed set (qwen); true = teacher top-k (ling).
  bool head_feature_teacher;

  int prefill_chunk;          // 2048 both wrappers
  int hot_window;             // 35b=4 / 8b=1
  int prefill_hot;            // 35b staged_k4=32 / 8b prod_k8=0
  bool full_layer_prefill;    // 35b=false / 8b=true
  bool async_eval_prefill;    // = prefill_multi and full_layer

  std::vector<int> predicted() const;
  std::vector<int> routed() const;     // decode-time live-router layers, ascending
};

bool tier_routing(const std::string &tier_id, TierRouting &out);

// Empty string = match; non-empty = readable mismatch (caller decides whether to reject).
std::string routing_config_mismatch(const TierRouting &r, int64_t cfg_num_experts,
                                    int64_t cfg_num_layers);

}  // namespace e0n
