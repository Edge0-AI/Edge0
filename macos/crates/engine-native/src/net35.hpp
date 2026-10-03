// 35b (Qwen3.5-MoE) forward graph: mixed GatedDeltaNet and full attention.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <mlx/mlx.h>

#include "expertio.hpp"
#include "expertmap.hpp"
#include "fwd_common.hpp"
#include "hotstack.hpp"
#include "lora.hpp"
#include "provider.hpp"
#include "prerouter.hpp"
#include "registry.hpp"
#include "router.hpp"
#include "safio.hpp"

namespace e0n {

struct RouteRec {
  int layer = -1;
  bool predicted = false;  // false = live router (incl. pos-0 fallback and routed layers)
  Selection sel;
};

struct DecodeProbe {
  std::vector<int> owners;
  std::vector<std::vector<float>> m_in;
  std::vector<std::vector<double>> logits;
};

struct PrefillProbe {
  std::vector<std::vector<float>> per_block;  // [num_layers][H], block output (residual sum)
  std::vector<float> embed_last;              // layer-0 input (last embedding position)
};

struct State35 {
  PrerouterState pr;
  std::vector<mlx::core::array> conv;
  std::vector<mlx::core::array> recur;
  std::vector<mlx::core::array> kc;
  std::vector<mlx::core::array> vc;
  std::vector<char> is_linear;
  int64_t position = 0;
  bool started = false;
};

class Net35 {
 public:
  // lo = unmerged LoRA (nullptr = base). Loading is owned by the net (ExpertIo).
  static std::unique_ptr<Net35> create(std::shared_ptr<const ModelFiles> files,
                                       std::shared_ptr<const ExpertMap> emap, std::string &err,
                                       std::shared_ptr<const lora::Bundle> lo = nullptr);

  const TierRouting &reg() const { return reg_; }
  Prerouter &prerouter() { return *pr_; }
  const lora::Bundle *lora() const { return lo_.get(); }
  uint64_t dense_resident_bytes() const { return dense_bytes_; }
  void prefetch(const Step &plan);
  ExpertIo::Metrics io_metrics() const { return io_->snapshot(); }
  void shard_warm(const std::vector<std::string> &shard_names, bool seq);

  void init_state(State35 &st) const;
  void reset_state(State35 &st) const;

  // Batched: chunked (2048) graph, hot-32 stack. Returns last-position logits
  // (host f32 [V]) + last-chunk per-layer routing. Does not seed decode; first
  // decode uses the live router. Consumer layers are warmed via prefetch_from_prefill.
  bool prefill(const std::vector<int32_t> &tokens, State35 &st, std::vector<float> &logits_last,
               std::vector<RouteRec> &recs, std::string &err, PrefillProbe *probe = nullptr);
  // Decode T=1: previous token → logits [V] + routing; runs run_step_boundary at step end.
  bool step(int32_t token, State35 &st, std::vector<float> &logits, std::vector<RouteRec> &recs,
            std::string &err, DecodeProbe *probe = nullptr);

 private:
  Net35() = default;
  struct Layer {
    bool is_linear = true;
    std::vector<float> in_ln_w, post_ln_w;
    QuantLin in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj;
    std::vector<uint8_t> conv_wb;
    mlx::core::array conv_w{0.f};
    std::vector<float> A_log, dt_bias, gnorm_w;
    QuantLin q_proj, k_proj, v_proj, o_proj;
    std::vector<float> qn_w, kn_w;
    std::vector<float> gate_h;
    QuantLin gate_q;
    QuantLin sh_gate, sh_up, sh_down, sh_egate;
  };

  bool layer_fwd(Layer &L, int li, mlx::core::array &x, State35 &st, bool decode,
                 RouteRec &rec, mlx::core::array *m_in_out, std::string &err);
  bool moe_host(int li, const std::vector<float> &m_in, int T, const Selection &sel,
                std::vector<float> &out, std::string &err);

  mutable std::shared_ptr<const ModelFiles> files_;
  std::shared_ptr<const ExpertMap> emap_;
  std::unique_ptr<ExpertIo> io_;
  TierRouting reg_;
  std::vector<Layer> layers_;
  std::unique_ptr<Prerouter> pr_;
  std::shared_ptr<const lora::Bundle> lo_;
  mx::fast::CustomKernelFunction gdn_kernel_;

  int64_t H_ = 0, V_ = 0, E_ = 0;
  int64_t hk_ = 0, hv_ = 0, dk_ = 0, dv_ = 0, conv_k_ = 0, conv_dim_ = 0;
  int64_t nq_ = 0, nkv_ = 0, hd_ = 0;
  int64_t moe_ffn_ = 0, sh_ffn_ = 0;
  double eps_ = 1e-6, rope_base_ = 100000.0, full_interval_ = 4.0, partial_rot_ = 0.25;
  bool attn_gate_ = true;
  double clip_ = 1000.0;

  QuantLin embed_;
  QuantLin lm_head_;
  std::vector<float> final_norm_w_;
  uint64_t dense_bytes_ = 0;

  bool stage_all_gpu(State35 &st, std::string &err, DecodeProbe *probe);
  void reset_graph_caches() const;

  bool prefill_batched_(const std::vector<int32_t> &tokens, State35 &st,
                        std::vector<float> &logits_last, std::vector<RouteRec> &recs,
                        std::string &err, PrefillProbe *probe);
  bool forward_chunk_batched_(const int32_t *tok, int64_t T, State35 &st,
                              std::vector<float> &logits_out, bool to_host,
                              std::vector<RouteRec> &recs, std::string &err,
                              PrefillProbe *probe);
  void prefill_before_layer(int li);
  void prefill_end_(State35 &st);
  bool moe_hot_branch_(int li, const mlx::core::array &m_in, int64_t T,
                       const std::vector<int64_t> &flat, mlx::core::array &routed,
                       std::string &err);

  mutable std::vector<mlx::core::array> m_in_cache_, oh_cur_, oh_prev_;
  mutable std::vector<mlx::core::array> pred_inds_gpu_, pred_scores_gpu_;
  mutable std::vector<std::shared_ptr<Bundle>> step_bundles_;
  mlx::core::array hw1_{0.f}, hw2_{0.f}, hwl_{0.f};
  std::function<std::vector<mlx::core::array>(const std::vector<mlx::core::array> &)> moe_math_;
  std::function<std::vector<mlx::core::array>(const std::vector<mlx::core::array> &)>
      moe_math_sorted_;
};

}  // namespace e0n
