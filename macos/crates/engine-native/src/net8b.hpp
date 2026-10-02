// 8b (Ling 3.0 hybrid) forward graph: KDA/MLA mix, L0 dense FFN, MoE 128-choose-8 + shared.
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <mlx/mlx.h>

#include "expertio.hpp"
#include "expertmap.hpp"
#include "fwd_common.hpp"
#include "hotstack.hpp"
#include "lora.hpp"
#include "net35.hpp"  // RouteRec (shared per-layer routing record)
#include "prerouter.hpp"
#include "provider.hpp"
#include "registry.hpp"
#include "router.hpp"
#include "safio.hpp"

namespace e0n {

struct State8b {
  PrerouterState pr;
  std::vector<mlx::core::array> cq, ck, cv, ssm;
  std::vector<mlx::core::array> kc, vc;
  std::vector<char> is_mla;
  int64_t position = 0;
  bool started = false;
};

class Net8b {
 public:
  // lo = unmerged LoRA (nullptr = base). Loading is owned by the net (ExpertIo).
  static std::unique_ptr<Net8b> create(std::shared_ptr<const ModelFiles> files,
                                       std::shared_ptr<const ExpertMap> emap, std::string &err,
                                       std::shared_ptr<const lora::Bundle> lo = nullptr);

  const TierRouting &reg() const { return reg_; }
  Prerouter &prerouter() { return *pr_; }
  const lora::Bundle *lora() const { return lo_.get(); }
  uint64_t dense_resident_bytes() const { return dense_bytes_; }
  void prefetch(const Step &plan);
  ExpertIo::Metrics io_metrics() const { return io_->snapshot(); }
  void shard_warm(const std::vector<std::string> &shard_names, bool seq);

  void init_state(State8b &st) const;
  void reset_state(State8b &st) const;

  bool prefill(const std::vector<int32_t> &tokens, State8b &st, std::vector<float> &logits_last,
               std::vector<RouteRec> &recs, std::string &err, PrefillProbe *probe = nullptr);
  bool step(int32_t token, State8b &st, std::vector<float> &logits, std::vector<RouteRec> &recs,
            std::string &err, DecodeProbe *probe = nullptr);

 private:
  Net8b() = default;
  struct Layer {
    bool is_mla = false;
    bool has_moe = false;  // L0 (dense_prefix) is false
    std::vector<float> in_ln_w, post_ln_w;
    QuantLin k_q, k_k, k_v, k_f, k_g, k_b, k_o;
    std::vector<uint8_t> conv_qb, conv_kb, conv_vb;  // BF16 [proj,1,4]
    mlx::core::array conv_q{0.f}, conv_k{0.f}, conv_v{0.f};  // load-time swap to [proj,4,1]
    std::vector<float> A_log, dt_bias, onorm_w;
    QuantLin m_qa, m_qb, m_kva, m_kvb, m_dense, m_g;
    std::vector<float> m_qaln, m_kvaln;
    std::vector<float> gate_h;
    bool gate_bf16 = false;       // file is BF16 — wrap back to bf16 for same-domain mul
    std::vector<double> gate_bias;
    QuantLin sh_gate, sh_up, sh_down;
    QuantLin d_gate, d_up, d_down;
  };

  bool layer_fwd(Layer &L, int li, mlx::core::array &x, State8b &st, bool decode, bool prefill_seed,
                 RouteRec &rec, std::string &err);

  mutable std::shared_ptr<const ModelFiles> files_;
  std::shared_ptr<const ExpertMap> emap_;
  std::unique_ptr<ExpertIo> io_;
  TierRouting reg_;
  std::vector<Layer> layers_;
  std::unique_ptr<Prerouter> pr_;
  std::shared_ptr<const lora::Bundle> lo_;

  int64_t H_ = 0, V_ = 0, E_ = 0;
  int64_t nh_ = 0, hd_ = 0, proj_ = 0, conv_k_ = 0;
  int64_t qlora_ = 0, kvlora_ = 0, nope_ = 0, roped_ = 0, vhd_ = 0;
  int64_t moe_ffn_ = 0, sh_ffn_ = 0, dense_ffn_ = 0;
  int64_t group_ = 64;
  double eps_ = 1e-6, rope_theta_ = 6000000.0, kda_lb_ = -5.0;
  double clip_ = 1000.0;

  QuantLin embed_;
  QuantLin lm_head_;
  std::vector<float> final_norm_w_;
  uint64_t dense_bytes_ = 0;

  bool stage_all_gpu(State8b &st, std::string &err, DecodeProbe *probe);
  void reset_graph_caches() const;

  bool prefill_batched_(const std::vector<int32_t> &tokens, State8b &st,
                        std::vector<float> &logits_last, std::vector<RouteRec> &recs,
                        std::string &err, PrefillProbe *probe);
  bool forward_chunk_batched_(const int32_t *tok, int64_t T, State8b &st,
                              std::vector<float> &logits_out, bool to_host,
                              std::vector<RouteRec> &recs, std::string &err,
                              PrefillProbe *probe);
  void prefill_before_layer(int li);
  bool prefill_end_(State8b &st, std::string &err);

  mutable std::vector<mlx::core::array> m_in_cache_, last_topk_, prev_topk_oh_, pg_logits_;
  mutable std::vector<std::shared_ptr<Bundle>> step_bundles_;
  std::vector<mlx::core::array> gate_w32_, gate_b32_;
  mlx::core::array hw1_{0.f}, hw2_{0.f}, hwl_{0.f};
  std::function<std::vector<mlx::core::array>(const std::vector<mlx::core::array> &)> moe_math_;
  std::function<std::vector<mlx::core::array>(const std::vector<mlx::core::array> &)>
      moe_math_sorted_;

  mlx::core::fast::CustomKernelFunction kda_kernel_;
};

}  // namespace e0n
