#include "net35.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <numeric>  // iota (batched selk)
#include <set>

#include "dense.hpp"
#include "qgemm.hpp"

namespace e0n {
namespace mx = mlx::core;

namespace {

using S = mx::Shape;

namespace {
inline uint16_t f_to_bf16(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}
inline void bf_round(std::vector<float> &v) {
  for (float &x : v) {
    uint32_t u = static_cast<uint32_t>(f_to_bf16(x)) << 16;
    std::memcpy(&x, &u, 4);
  }
}
}  // namespace

mx::array first_k(const mx::array &a, int64_t k) {
  mx::Shape s1 = a.shape();
  mx::Shape s0 = a.shape();
  for (auto &v : s0) v = 0;
  s1[s1.size() - 1] = static_cast<mx::Shape::value_type>(k);
  return mx::slice(a, s0, s1);
}
// heads.py select_from_logits(44-51) ≡ moe/routing.py select_from_logits(13-24):
void select_softmax_gpu(const mx::array &logits, int64_t E, int64_t k, bool norm,
                        mx::array &idx, mx::array &w) {
  (void)E;  // shape is carried by logits; retained in the call signature for router symmetry.
  auto gates = mx::softmax(logits, std::vector<int>{-1}, true /*precise*/);  // heads.py:47
  idx = first_k(mx::argpartition(-gates, static_cast<int>(k - 1), -1), k);  // 48
  w = mx::take_along_axis(gates, idx, -1);                                // 49
  if (norm) w = w / mx::sum(w, std::vector<int>{-1}, true);               // 50(norm_topk_prob)
}

mx::array topk_onehot(const mx::array &inds, int64_t E) {
  auto eq = mx::equal(mx::expand_dims(inds, -1), mx::arange(0, static_cast<int>(E)));
  return mx::astype(mx::sum(eq, std::vector<int>{-2}, false), mx::float32);
}

mx::array gelu_erf16(const mx::array &x) {
  return mx::array(0.5f, mx::float16) * x *
         (mx::array(1.0f, mx::float16) +
          mx::erf(x / mx::array(static_cast<float>(std::sqrt(2.0)), mx::float16)));
}

std::vector<mx::array> moe_math_body(const std::vector<mx::array> &a) {
  auto xe = mx::reshape(a[0], mkshape({a[0].shape(0), a[0].shape(1), 1, 1, a[0].shape(2)}));  // layer.py:159
  auto gq = [&](const mx::array &xx, int o) {
    return mx::gather_qmm(xx, a[o], a[o + 1], a[o + 2], std::nullopt, a[10], true, 64, 4,
                          "affine", false);
  };
  auto x_up = gq(xe, 1);
  auto x_gate = gq(xe, 4);
  auto yh = (x_gate * mx::sigmoid(x_gate)) * x_up;  // _swiglu(x_up, x_gate)(layer.py:57-58,169)
  auto z = gq(yh, 7);
  return {mx::squeeze(z, -2)};                      // 172
}

std::vector<mx::array> moe_math_sorted_body(const std::vector<mx::array> &a) {
  auto gq = [&](const mx::array &xx, int o) {
    return mx::gather_qmm(xx, a[o], a[o + 1], a[o + 2], std::nullopt, a[10], true, 64, 4,
                          "affine", true);
  };
  auto x_up = gq(a[0], 1);
  auto x_gate = gq(a[0], 4);
  auto yh = (x_gate * mx::sigmoid(x_gate)) * x_up;             // _swiglu
  return {gq(yh, 7)};                                          // z=[T*k,1,H]
}

ExpertPtrs bundle_ptrs(const Bundle &b) {
  ExpertPtrs p;
  p.uw = b.at(0).data<uint32_t>();
  p.us = reinterpret_cast<const uint16_t *>(b.at(1).data<mx::bfloat16_t>());
  p.ub = reinterpret_cast<const uint16_t *>(b.at(2).data<mx::bfloat16_t>());
  p.gw = b.at(3).data<uint32_t>();
  p.gs = reinterpret_cast<const uint16_t *>(b.at(4).data<mx::bfloat16_t>());
  p.gb = reinterpret_cast<const uint16_t *>(b.at(5).data<mx::bfloat16_t>());
  p.dw = b.at(6).data<uint32_t>();
  p.ds = reinterpret_cast<const uint16_t *>(b.at(7).data<mx::bfloat16_t>());
  p.db = reinterpret_cast<const uint16_t *>(b.at(8).data<mx::bfloat16_t>());
  return p;
}

}  // namespace

std::unique_ptr<Net35> Net35::create(std::shared_ptr<const ModelFiles> files,
                                     std::shared_ptr<const ExpertMap> emap, std::string &err,
                                     std::shared_ptr<const lora::Bundle> lo) {
  std::unique_ptr<Net35> n(new Net35());
  n->files_ = files;
  n->emap_ = emap;
  n->lo_ = std::move(lo);
  if (!tier_routing("edge0-35b", n->reg_)) { err = "registry missing edge0-35b"; return nullptr; }
  int64_t cfg_layers = 0;
  if (files->config_int("num_hidden_layers", cfg_layers) && cfg_layers != n->reg_.num_layers) {
    err = "config layer count does not match registry";
    return nullptr;
  }
  auto ci = [&](const char *k, int64_t def, int64_t &o) {
    int64_t v;
    o = files->config_int(k, v) ? v : def;
  };
  auto cd = [&](const char *k, double def, double &o) {
    double v;
    o = files->config_double(k, v) ? v : def;
  };
  ci("hidden_size", 2048, n->H_);
  ci("vocab_size", 248320, n->V_);
  ci("moe_intermediate_size", 512, n->moe_ffn_);
  ci("shared_expert_intermediate_size", 512, n->sh_ffn_);
  ci("linear_num_key_heads", 16, n->hk_);
  ci("linear_num_value_heads", 32, n->hv_);
  ci("linear_key_head_dim", 128, n->dk_);
  ci("linear_value_head_dim", 128, n->dv_);
  ci("linear_conv_kernel_dim", 4, n->conv_k_);
  ci("num_attention_heads", 16, n->nq_);
  ci("num_key_value_heads", 2, n->nkv_);
  ci("head_dim", 256, n->hd_);
  int64_t interval = 4;
  ci("full_attention_interval", 4, interval);
  n->full_interval_ = static_cast<double>(interval);
  cd("rms_norm_eps", 1e-6, n->eps_);
  cd("rope_theta", 100000.0, n->rope_base_);
  cd("partial_rotary_factor", 0.25, n->partial_rot_);
  bool og = true;
  if (!files->config_bool("attn_output_gate", og)) og = true;
  n->attn_gate_ = og;

  const int64_t keyd = n->hk_ * n->dk_, vald = n->hv_ * n->dv_;
  n->conv_dim_ = 2 * keyd + vald;
  {
    static const std::string kGdnSource = R"gdn(
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        // g: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          if (true) {
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * g_[hv_idx];
              kv_mem += state[i] * k_[s_idx];
            }
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + k_[s_idx] * delta;
              out += state[i] * q_[s_idx];
            }
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {
              y[dv_idx] = static_cast<InT>(out);
            }
          }
          // Increment data pointers to next time step
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<InT>(state[i]);
        }
    )gdn";
    n->gdn_kernel_ = mx::fast::metal_kernel(
        "gated_delta_step", {"q", "k", "v", "g", "beta", "state_in", "T"}, {"y", "state_out"},
        kGdnSource);
  }

  n->layers_.resize(static_cast<size_t>(n->reg_.num_layers));
  const std::string M = "language_model.model.layers.";
  for (int64_t li = 0; li < n->reg_.num_layers; li++) {
    Layer &L = n->layers_[static_cast<size_t>(li)];
    const std::string P = M + std::to_string(li);
    L.is_linear = ((li + 1) % interval) != 0;
    auto q = [&](const std::string &name, QuantLin &dst) {
      if (!dst.load(*files, P + name, err)) { err += " @" + std::to_string(li); return false; }
      n->dense_bytes_ += dst.wb.size() + dst.sb.size() + dst.bb.size();
      return true;
    };
    auto vec = [&](const std::string &name, std::vector<float> &dstv) {
      DenseF32 d;
      if (!read_dense_f32(*files, P + name, d, err)) {
        err += " @" + std::to_string(li);
        return false;
      }
      dstv = std::move(d.v);
      n->dense_bytes_ += dstv.size() * 4;
      return true;
    };
    if (!vec(".input_layernorm.weight", L.in_ln_w) ||
        !vec(".post_attention_layernorm.weight", L.post_ln_w))
      return nullptr;
    if (L.is_linear) {
      if (!q(".linear_attn.in_proj_qkv.weight", L.in_proj_qkv) ||
          !q(".linear_attn.in_proj_z.weight", L.in_proj_z) ||
          !q(".linear_attn.in_proj_b.weight", L.in_proj_b) ||
          !q(".linear_attn.in_proj_a.weight", L.in_proj_a) ||
          !q(".linear_attn.out_proj.weight", L.out_proj) ||
          !vec(".linear_attn.A_log", L.A_log) || !vec(".linear_attn.dt_bias", L.dt_bias) ||
          !vec(".linear_attn.norm.weight", L.gnorm_w))
        return nullptr;
      if (static_cast<int64_t>(L.A_log.size()) != n->hv_ ||
          static_cast<int64_t>(L.dt_bias.size()) != n->hv_ ||
          static_cast<int64_t>(L.gnorm_w.size()) != n->dv_) {
        err = "GDN parameter vector length mismatch";
        return nullptr;
      }
      if (!files->read_tensor(P + ".linear_attn.conv1d.weight", L.conv_wb, err)) return nullptr;
      L.conv_w = mx::array(L.conv_wb.data(), mkshape({n->conv_dim_, n->conv_k_, 1}),
                           mx::bfloat16, [](void *) {});
      n->dense_bytes_ += L.conv_wb.size();
    } else {
      if (!q(".self_attn.q_proj.weight", L.q_proj) || !q(".self_attn.k_proj.weight", L.k_proj) ||
          !q(".self_attn.v_proj.weight", L.v_proj) || !q(".self_attn.o_proj.weight", L.o_proj) ||
          !vec(".self_attn.q_norm.weight", L.qn_w) || !vec(".self_attn.k_norm.weight", L.kn_w))
        return nullptr;
    }
    DenseF32 g;
    int64_t gbits = 4, ggs = 64;
    const std::string gw = P + ".mlp.gate.weight";
    if (!files->config_quant_for(gw, gbits, ggs)) { err = "gate has no quantization config"; return nullptr; }
    if (!read_dense_dequant(*files, gw, gbits, ggs, g, err)) return nullptr;
    if (g.rows != n->reg_.num_experts || g.cols != n->H_) {
      err = "gate shape mismatch";
      return nullptr;
    }
    L.gate_h = std::move(g.v);
    n->dense_bytes_ += L.gate_h.size() * 4;
    if (!L.gate_q.load(*files, gw, err)) { err += " @gate_q"; return nullptr; }
    n->dense_bytes_ += L.gate_q.wb.size() + L.gate_q.sb.size() + L.gate_q.bb.size();
    if (!q(".mlp.shared_expert.gate_proj.weight", L.sh_gate) ||
        !q(".mlp.shared_expert.up_proj.weight", L.sh_up) ||
        !q(".mlp.shared_expert.down_proj.weight", L.sh_down) ||
        !q(".mlp.shared_expert_gate.weight", L.sh_egate))
      return nullptr;
  }
  if (!n->embed_.load(*files, "language_model.model.embed_tokens.weight", err)) return nullptr;
  if (!n->lm_head_.load(*files, "language_model.lm_head.weight", err)) return nullptr;
  n->dense_bytes_ += n->embed_.wb.size() + n->embed_.sb.size() + n->embed_.bb.size() +
                     n->lm_head_.wb.size() + n->lm_head_.sb.size() + n->lm_head_.bb.size();
  {
    DenseF32 f;
    if (!read_dense_f32(*files, "language_model.model.norm.weight", f, err)) return nullptr;
    n->final_norm_w_ = std::move(f.v);
  }
  n->pr_ = std::make_unique<Prerouter>();
  if (!Prerouter::load(*files, n->reg_, n->H_, *n->pr_, err)) return nullptr;
  {
    const int64_t NL2 = n->reg_.num_layers;
    n->io_ = std::make_unique<ExpertIo>(files.get(), 64, 48, 8, 4);
    for (int64_t li = 0; li < NL2; li++) {
      ExpertIo::LayerCfg c;
      c.layer = int(li);
      c.prefix = emap->layer_prefix(int(li));
      c.E = n->reg_.num_experts;
      c.H = n->H_;
      c.F = n->moe_ffn_;
      c.group = 64;
      c.staged_n = int(n->reg_.top_k);
      c.incr_mode = true;
      c.incr_writeback = true;
      c.warm_willneed = true;
      c.prefill_hot = (int)n->reg_.prefill_hot;
      c.full_layer_prefill = n->reg_.full_layer_prefill;
      n->io_->add_layer(c);
    }
  }
  if (exec_mode() == ExecMode::Batched) {
    const char *cl = getenv("MLX_CACHE_LIMIT_MB");
    mx::set_cache_limit(static_cast<size_t>(cl ? atoi(cl) : 256) << 20);
    const int64_t NL = n->reg_.num_layers;
    n->m_in_cache_.assign(NL, mx::array(0.f));
    n->oh_cur_.assign(NL, mx::array(0.f));
    n->oh_prev_.assign(NL, mx::array(0.f));
    n->pred_inds_gpu_.assign(NL, mx::array(0.f));
    n->pred_scores_gpu_.assign(NL, mx::array(0.f));
    const int64_t nH = n->pr_->n_heads(), f_in = n->pr_->stacked_f_in(),
                  prh = n->pr_->stacked_prh(), hE = n->pr_->stacked_E();
    if (nH > 0) {
      auto wrap16 = [&](const std::vector<float> &v, std::initializer_list<int64_t> dims) {
        return mx::astype(mx::array((void *)v.data(), mkshape(dims), mx::float32, [](void *) {}),
                          mx::float16);
      };
      n->hw1_ = wrap16(n->pr_->stacked_fc1(), {nH, f_in, prh});
      n->hw2_ = wrap16(n->pr_->stacked_fc2(), {nH, prh, hE});
      n->hwl_ = wrap16(n->pr_->stacked_lin(), {nH, f_in, hE});
      mx::eval(n->hw1_, n->hw2_, n->hwl_);
    }
    // @core.compile(layer.py:157;staged_k4 use_compile=True，options.py:129).
    try {
      auto fn = mx::compile(&moe_math_body);
      const int64_t F = n->moe_ffn_;
      std::vector<mx::array> d = {
          mx::zeros(mkshape({1, 1, n->H_}), mx::bfloat16),
          mx::zeros(mkshape({2, F, n->H_ / 8}), mx::uint32),
          mx::zeros(mkshape({2, F, n->H_ / 64}), mx::bfloat16),
          mx::zeros(mkshape({2, F, n->H_ / 64}), mx::bfloat16),
          mx::zeros(mkshape({2, F, n->H_ / 8}), mx::uint32),
          mx::zeros(mkshape({2, F, n->H_ / 64}), mx::bfloat16),
          mx::zeros(mkshape({2, F, n->H_ / 64}), mx::bfloat16),
          mx::zeros(mkshape({2, n->H_, F / 8}), mx::uint32),
          mx::zeros(mkshape({2, n->H_, F / 64}), mx::bfloat16),
          mx::zeros(mkshape({2, n->H_, F / 64}), mx::bfloat16),
          mx::zeros(mkshape({1, 1, 2}), mx::int32)};
      auto out = fn(d);
      mx::eval(out[0]);
      n->moe_math_ = std::move(fn);
    } catch (const std::exception &ex) {
      std::fprintf(stderr, "edge0-35b: mlx compile unavailable (%s), MoE math falling back to eager\n",
                   ex.what());
      n->moe_math_ = &moe_math_body;
    }
    try {
      auto fn = mx::compile(&moe_math_sorted_body);
      const int64_t F = n->moe_ffn_;
      std::vector<mx::array> d = {
          mx::zeros(mkshape({16, 1, n->H_}), mx::bfloat16),
          mx::zeros(mkshape({2, F, n->H_ / 8}), mx::uint32),
          mx::zeros(mkshape({2, F, n->H_ / 64}), mx::bfloat16),
          mx::zeros(mkshape({2, F, n->H_ / 64}), mx::bfloat16),
          mx::zeros(mkshape({2, F, n->H_ / 8}), mx::uint32),
          mx::zeros(mkshape({2, F, n->H_ / 64}), mx::bfloat16),
          mx::zeros(mkshape({2, F, n->H_ / 64}), mx::bfloat16),
          mx::zeros(mkshape({2, n->H_, F / 8}), mx::uint32),
          mx::zeros(mkshape({2, n->H_, F / 64}), mx::bfloat16),
          mx::zeros(mkshape({2, n->H_, F / 64}), mx::bfloat16),
          mx::zeros(mkshape({16}), mx::int32)};
      auto out = fn(d);
      mx::eval(out[0]);
      n->moe_math_sorted_ = std::move(fn);
    } catch (const std::exception &ex) {
      std::fprintf(stderr, "edge0-35b: mlx compile unavailable (%s), sorted MoE math falling back to eager\n",
                   ex.what());
      n->moe_math_sorted_ = &moe_math_sorted_body;
    }
  }
  if (n->lo_) {
    const int nl = n->reg_.num_layers;
    const int full = nl / 4, lin = nl - full;
    const int want = 5 * lin + 4 * full + 3 * nl;
    if (n->lo_->n_bound() != want) {
      err = "LoRA bound count " + std::to_string(n->lo_->n_bound()) + " != tier structural count " +
            std::to_string(want);
      return nullptr;
    }
    n->dense_bytes_ += static_cast<uint64_t>(n->lo_->bytes());
  }
  return n;
}

void Net35::init_state(State35 &st) const {
  st.pr.init(reg_);
  const size_t NL = layers_.size();
  st.conv.assign(NL, mx::array(0.f));
  st.recur.assign(NL, mx::array(0.f));
  st.kc.assign(NL, mx::array(0.f));
  st.vc.assign(NL, mx::array(0.f));
  st.is_linear.assign(NL, 0);
  for (size_t i = 0; i < NL; i++) st.is_linear[i] = layers_[i].is_linear ? 1 : 0;
  st.position = 0;
  st.started = true;
  reset_graph_caches();
}

void Net35::reset_state(State35 &st) const {
  st.pr.reset();
  st.conv.assign(layers_.size(), mx::array(0.f));
  st.recur.assign(layers_.size(), mx::array(0.f));
  st.kc.assign(layers_.size(), mx::array(0.f));
  st.vc.assign(layers_.size(), mx::array(0.f));
  st.position = 0;
  reset_graph_caches();
}

bool Net35::layer_fwd(Layer &L, int li, mx::array &x, State35 &st, bool decode, RouteRec &rec,
                      mx::array *m_in_out, std::string &err) {
  const int64_t T = x.shape(1);
  const auto dt = x.dtype();
  const int64_t keyd = hk_ * dk_, vald = hv_ * dv_;
  auto norm_w = [&](const std::vector<float> &v) {
    return mx::astype(wrap_f32(v, mkshape({static_cast<int64_t>(v.size())})), dt);
  };
  mx::array xn = e0_rms_norm(x, norm_w(L.in_ln_w), eps_);

  mx::array r{0.f};
  if (L.is_linear) {
    const auto *lo = lo_.get();
    using LS = e0n::lora::Slot35;
    mx::array qkv = e0n::lora::apply_lora(lo, L.in_proj_qkv, xn, li, LS::S35_IN_QKV);  // [1,T,conv_dim]
    mx::array z = mx::reshape(e0n::lora::apply_lora(lo, L.in_proj_z, xn, li, LS::S35_IN_Z),
                              mkshape({1, T, hv_, dv_}));
    mx::array b = e0n::lora::apply_lora(lo, L.in_proj_b, xn, li, LS::S35_IN_B);  // [1,T,hv]
    mx::array a = e0n::lora::apply_lora(lo, L.in_proj_a, xn, li, LS::S35_IN_A);
    if (st.conv[li].ndim() == 0)
      st.conv[li] = mx::zeros(mkshape({1, conv_k_ - 1, conv_dim_}), dt);
    mx::array cat = mx::concatenate({st.conv[li], qkv}, 1);  // [1, K-1+T, C]
    st.conv[li] = mx::slice(cat, mkshape({0, T, 0}), mkshape({1, T + conv_k_ - 1, conv_dim_}));
    mx::array convout = e0_silu(mx::conv1d(
        cat, L.conv_w, 1, 0, 1, static_cast<int>(conv_dim_)));  // [1,T,C]
    auto parts = mx::split(convout, S{static_cast<int32_t>(keyd), static_cast<int32_t>(2 * keyd)},
                           -1);
    mx::array qf = mx::reshape(parts[0], mkshape({1, T, hk_, dk_}));
    mx::array kf = mx::reshape(parts[1], mkshape({1, T, hk_, dk_}));
    mx::array vf = mx::reshape(parts[2], mkshape({1, T, hv_, dv_}));
    float inv = 1.0f / std::sqrt(static_cast<float>(dk_));
    auto scalar_bf = [&](float v) { return mx::astype(mx::array(v), dt); };
    qf = scalar_bf(inv * inv) * e0_rms_norm(qf, std::nullopt, 1e-6);
    kf = scalar_bf(inv) * e0_rms_norm(kf, std::nullopt, 1e-6);
    mx::array beta = mx::sigmoid(b);
    mx::array alog = wrap_f32(L.A_log, mkshape({hv_}));
    mx::array dtb = mx::astype(wrap_f32(L.dt_bias, mkshape({hv_})), dt);  // bf16
    mx::array xab = a + dtb;
    mx::array sp = mx::log1p(mx::exp(-mx::abs(xab))) +
                   mx::maximum(xab, mx::astype(mx::array(0.f), dt));
    mx::array g = mx::astype(mx::exp(-mx::exp(alog) * mx::astype(sp, mx::float32)), dt);
    if (st.recur[li].ndim() == 0)
      st.recur[li] = mx::zeros(mkshape({1, hv_, dv_, dk_}), dt);  // gated_delta_update:275-278
    auto gouts = gdn_kernel_(
        {qf, kf, vf, g, beta, st.recur[li], mx::array((int)T, mx::int32)},
        {mkshape({1, T, hv_, dv_}), st.recur[li].shape()},
        {dt, dt},
        {32, static_cast<int>(dv_), static_cast<int>(hv_)},  // grid=(32, Dv, B*Hv)，B=1
        {32, 4, 1},                                          // threadgroup(gated_delta.py:206)
        {{"InT", dt},
         {"Dk", static_cast<int>(dk_)},
         {"Dv", static_cast<int>(dv_)},
         {"Hk", static_cast<int>(hk_)},
         {"Hv", static_cast<int>(hv_)}},
        std::nullopt, false, {});
    mx::array y_arr = gouts[0];
    st.recur[li] = gouts[1];
    mx::array ng = e0_rms_norm(y_arr, norm_w(L.gnorm_w), eps_);
    mx::array gated = e0_swiglu_precise(dt, z, ng);  // _precise_swiglu(out, z, x):silu(z)*x
    r = e0n::lora::apply_lora(lo_.get(), L.out_proj,
                              mx::reshape(gated, mkshape({1, T, vald})), li,
                              e0n::lora::S35_OUT_PROJ);
  } else {
    mx::array qo = e0n::lora::apply_lora(lo_.get(), L.q_proj, xn, li, e0n::lora::S35_Q);
    auto qs = mx::split(mx::reshape(qo, mkshape({1, T, nq_, hd_ * 2})),
                        S{static_cast<int32_t>(hd_)}, -1);
    mx::array qg = qs[0], gate = mx::reshape(qs[1], mkshape({1, T, nq_ * hd_}));
    mx::array q = mx::transpose(e0_rms_norm(qg, norm_w(L.qn_w), eps_), {0, 2, 1, 3});
    mx::array k = mx::transpose(
        e0_rms_norm(mx::reshape(e0n::lora::apply_lora(lo_.get(), L.k_proj, xn, li,
                                                   e0n::lora::S35_K),
                                   mkshape({1, T, nkv_, hd_})),
                    norm_w(L.kn_w), eps_),
        {0, 2, 1, 3});
    mx::array v = mx::transpose(mx::reshape(e0n::lora::apply_lora(lo_.get(), L.v_proj, xn, li,
                                                                        e0n::lora::S35_V),
                                                        mkshape({1, T, nkv_, hd_})),
                                {0, 2, 1, 3});
    const int rdim = static_cast<int>(hd_ * partial_rot_);
    q = mx::fast::rope(q, rdim, false, static_cast<float>(rope_base_), 1.0f,
                       static_cast<int>(st.position));
    k = mx::fast::rope(k, rdim, false, static_cast<float>(rope_base_), 1.0f,
                       static_cast<int>(st.position));
    if (st.kc[li].ndim() == 0) {
      st.kc[li] = k;
      st.vc[li] = v;
    } else {
      st.kc[li] = mx::concatenate({st.kc[li], k}, 2);
      st.vc[li] = mx::concatenate({st.vc[li], v}, 2);
    }
    mx::array o = mx::fast::scaled_dot_product_attention(
        q, st.kc[li], st.vc[li], 1.0f / std::sqrt(static_cast<float>(hd_)), "causal");
    o = mx::reshape(mx::transpose(o, {0, 2, 1, 3}), mkshape({1, T, nq_ * hd_}));
    if (attn_gate_) o = o * mx::sigmoid(gate);
    r = e0n::lora::apply_lora(lo_.get(), L.o_proj, o, li, e0n::lora::S35_O);
  }

  mx::array h = x + r;
  mx::array m_in = e0_rms_norm(h, norm_w(L.post_ln_w), eps_);
  if (m_in_out) *m_in_out = m_in;
  if (exec_mode() == ExecMode::Batched && decode && T == 1) {
    const int64_t E = reg_.num_experts, k = reg_.top_k;
    io_->note_call(li);
    const bool owner = li >= reg_.first_owner && li <= reg_.last_owner;   // 6..38
    const bool consumer = li >= reg_.first_head_consumer && li <= reg_.last_head_consumer;  // 7..38
    mx::array idx(0.f), w(0.f), routed(0.f);
    rec.layer = li;
    if (consumer && pred_inds_gpu_[static_cast<size_t>(li)].ndim() > 0) {
      idx = pred_inds_gpu_[static_cast<size_t>(li)];
      w = pred_scores_gpu_[static_cast<size_t>(li)];
      rec.predicted = true;
      rec.sel = st.pr.pred[static_cast<size_t>(li)];
      st.pr.pred_avail[static_cast<size_t>(li)] = false;
      io_->wait_staged(li);
      auto local = mx::take(io_->incr_slot_table(li), idx);  // layer.py:1338 take(_incr_slot_table, indices)
      std::vector<mx::array> args{m_in};
      for (auto &a : io_->incr_tensors(li)) args.push_back(a);
      args.push_back(local);
      auto &cx = exec_counters();
      cx.qmm_calls.fetch_add(3, std::memory_order_relaxed);
      cx.qmm_experts.fetch_add(3 * static_cast<uint64_t>(k), std::memory_order_relaxed);
      cx.qmm_weight_bytes.fetch_add(
          static_cast<uint64_t>(k) *
              (2 * (moe_ffn_ * (H_ / 8) * 4 + 2 * moe_ffn_ * (H_ / 64) * 2) +
               (H_ * (moe_ffn_ / 8) * 4 + 2 * H_ * (moe_ffn_ / 64) * 2)),
          std::memory_order_relaxed);
      routed = moe_math_(args)[0];  // layer.py:1341
    } else {
      const int F = static_cast<int>(moe_ffn_);
      const int gsh = static_cast<int>(H_ / 64), gsf = static_cast<int>(moe_ffn_ / 64);
      auto gl = L.gate_q.apply(m_in);
      select_softmax_gpu(gl, E, k, reg_.norm_topk_prob, idx, w);
      mx::eval(idx, w);
      exec_counters().host_syncs.fetch_add(1, std::memory_order_relaxed);
      auto idxh = mx::copy(mx::astype(idx, mx::int32), mx::default_stream(mx::Device::cpu));
      auto wh = mx::copy(mx::astype(w, mx::float32), mx::default_stream(mx::Device::cpu));
      mx::eval(idxh, wh);
      const int32_t *ip = idxh.data<int32_t>();
      const float *wp = wh.data<float>();
      std::vector<int64_t> flat(ip, ip + k);
      std::vector<int64_t> uniq = flat;
      std::sort(uniq.begin(), uniq.end());
      uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
      std::vector<int32_t> local_h(flat.size());
      for (size_t j = 0; j < flat.size(); j++)
        local_h[j] = static_cast<int32_t>(
            std::lower_bound(uniq.begin(), uniq.end(), flat[j]) - uniq.begin());
      auto bundles = io_->get_bundles(li, uniq);
      if (bundles.size() != uniq.size()) {
        err = "expert load missing L" + std::to_string(li);
        return false;
      }
      exec_counters().leases.fetch_add(uniq.size(), std::memory_order_relaxed);
      for (auto &[e, b] : bundles) step_bundles_.push_back(b);
      std::vector<mx::array> args{m_in};
      for (int p = 0; p < 9; p++) {
        std::vector<mx::array> rows;
        for (auto e : uniq) rows.push_back(bundles[e]->at(p));
        args.push_back(mx::stack(rows));
      }
      args.push_back(mx::array(local_h.begin(), mkshape({1, 1, k}), mx::int32));  // 1371
      auto &cx = exec_counters();
      cx.qmm_calls.fetch_add(3, std::memory_order_relaxed);
      cx.qmm_experts.fetch_add(3 * static_cast<uint64_t>(uniq.size()), std::memory_order_relaxed);
      cx.qmm_weight_bytes.fetch_add(
          static_cast<uint64_t>(uniq.size()) *
              (2 * (F * (H_ / 8) * 4 + 2 * F * gsh * 2) + (H_ * (F / 8) * 4 + 2 * H_ * gsf * 2)),
          std::memory_order_relaxed);
      routed = moe_math_(args)[0];
      rec.predicted = false;
      rec.sel.inds = uniq;
      rec.sel.scores.clear();
      for (int64_t e : uniq) {
        double s = 0.0;
        for (int64_t j = 0; j < k; j++)
          if (flat[j] == e) s = wp[j];
        rec.sel.scores.push_back(s);
      }
    }
    if (owner) {
      m_in_cache_[static_cast<size_t>(li)] = m_in;
      oh_cur_[static_cast<size_t>(li)] = topk_onehot(idx, E);
    }
    // mix+shared(install.py:190-194):(y*scores[...,None]).sum(-2) + sigmoid(egate(x))*shared(x)
    auto out = mx::sum(routed * mx::expand_dims(w, -1), std::vector<int>{-2}, false);
    auto shg = e0n::lora::apply_lora(lo_.get(), L.sh_gate, m_in, li, e0n::lora::S35_SH_GATE);
    auto shu = e0n::lora::apply_lora(lo_.get(), L.sh_up, m_in, li, e0n::lora::S35_SH_UP);
    mx::array sh =
        e0n::lora::apply_lora(lo_.get(), L.sh_down, e0_silu(shg) * shu, li, e0n::lora::S35_SH_DOWN);
    mx::array eg = mx::sigmoid(L.sh_egate.apply(m_in));
    x = h + out + sh * eg;
    if (clip_ > 0)
      x = mx::where(mx::isnan(x), mx::zeros_like(x),
                    mx::clip(x, mx::array(static_cast<float>(-clip_)),
                             mx::array(static_cast<float>(clip_))));
    return true;
  }
  if (exec_mode() == ExecMode::Batched) {
    const int64_t E = reg_.num_experts, k = reg_.top_k;
    io_->note_call(li);
    mx::array idx(0.f), w(0.f);
    rec.layer = li;
    rec.predicted = false;
    {
      auto gl = L.gate_q.apply(m_in);
      select_softmax_gpu(gl, E, k, reg_.norm_topk_prob, idx, w);  // qwen3_next.py:335-341
    }
    mx::eval(idx, w);
    exec_counters().host_syncs.fetch_add(1, std::memory_order_relaxed);
    auto idxh = mx::copy(mx::astype(idx, mx::int32), mx::default_stream(mx::Device::cpu));
    auto wh = mx::copy(mx::astype(w, mx::float32), mx::default_stream(mx::Device::cpu));
    mx::eval(idxh, wh);
    const int32_t *ip = idxh.data<int32_t>();
    const float *wp = wh.data<float>();
    std::vector<int64_t> flat(static_cast<size_t>(T * k));
    for (int64_t i = 0; i < T * k; i++) flat[static_cast<size_t>(i)] = ip[i];
    mx::array routed(0.f);
    const bool hot = !prefill_ondemand() && T * k > 8 && io_->hot_ready(li) &&
                     !io_->full_ready(li);  // __call__ guard(layer.py:1121-1122)
    if (hot) {
      io_->note_prefill_use(li, flat, (int)k,
                            li >= reg_.first_head_consumer && li <= reg_.last_head_consumer);
      if (!moe_hot_branch_(li, m_in, T, flat, routed, err)) return false;
    }
    if (!hot) {
      std::vector<int64_t> uniq = flat;                     // 1364 sorted(set(...))
      std::sort(uniq.begin(), uniq.end());
      uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
      std::vector<int32_t> local_h(static_cast<size_t>(T * k));  // 1370-1373 host remap
      for (int64_t i = 0; i < T * k; i++)
        local_h[static_cast<size_t>(i)] = static_cast<int32_t>(
            std::lower_bound(uniq.begin(), uniq.end(), flat[static_cast<size_t>(i)]) - uniq.begin());
      auto bundles = io_->get_bundles(li, uniq);            // 1368
      if (bundles.size() != uniq.size()) {
        err = "expert load missing L" + std::to_string(li);
        return false;
      }
      exec_counters().leases.fetch_add(uniq.size(), std::memory_order_relaxed);
      for (auto &[e, b] : bundles) step_bundles_.push_back(b);
      std::vector<mx::array> wargs;                         // 1376-1380 core.stack(rows)
      for (int p = 0; p < 9; p++) {
        std::vector<mx::array> rows;
        for (auto e : uniq) rows.push_back(bundles[e]->at(p));
        wargs.push_back(mx::stack(rows));
      }
      auto local_arr = mx::array(local_h.begin(), mkshape({1, T, k}), mx::int32);  // 1371
      const int F = static_cast<int>(moe_ffn_);
      auto &cx = exec_counters();
      cx.qmm_calls.fetch_add(3, std::memory_order_relaxed);
      cx.qmm_experts.fetch_add(3 * static_cast<uint64_t>(uniq.size()), std::memory_order_relaxed);
      cx.qmm_weight_bytes.fetch_add(
          static_cast<uint64_t>(uniq.size()) *
              (2 * (F * (H_ / 8) * 4 + 2 * F * (H_ / 64) * 2) +
               (H_ * (F / 8) * 4 + 2 * H_ * (F / 64) * 2)),
          std::memory_order_relaxed);
      if (T * k >= 64) {                            // 1384 do_sort = local2d.size >= 64
        auto flat_local = mx::reshape(local_arr, mkshape({T * k}));
        auto order = mx::argsort(flat_local, -1);   // _gather_sort(switch_layers.py:12-17)
        auto inv_order = mx::argsort(order, -1);
        auto xe = mx::reshape(m_in, mkshape({1, T, 1, 1, H_}));  // 1383 expand_dims((-2,-3))
        auto xs = mx::flatten(xe, 0, -3);           // [T,1,H]
        auto tok_rows = mx::take(
            xs, mx::floor_divide(order, mx::array((uint32_t)k, mx::uint32)), 0);  // x[order//M]
        auto local_sorted = mx::take(flat_local, order, 0);  // indices[order]
        std::vector<mx::array> sargs{tok_rows};
        for (auto &wa : wargs) sargs.push_back(wa);
        sargs.push_back(local_sorted);
        auto z = moe_math_sorted_(sargs)[0];        // [T*k,1,H]
        auto z_un = mx::take(z, inv_order, 0);
        routed = mx::squeeze(mx::reshape(z_un, mkshape({1, T, k, 1, H_})), -2);
      } else {
        std::vector<mx::array> args{m_in};
        for (auto &wa : wargs) args.push_back(wa);
        args.push_back(local_arr);
        routed = moe_math_(args)[0];
      }
    }
    // mix+shared(install.py:190-194):(y*scores[...,None]).sum(-2) + sigmoid(egate(x))*shared(x)
    auto out = mx::sum(routed * mx::expand_dims(w, -1), std::vector<int>{-2}, false);
    auto shg = e0n::lora::apply_lora(lo_.get(), L.sh_gate, m_in, li, e0n::lora::S35_SH_GATE);
    auto shu = e0n::lora::apply_lora(lo_.get(), L.sh_up, m_in, li, e0n::lora::S35_SH_UP);
    mx::array sh =
        e0n::lora::apply_lora(lo_.get(), L.sh_down, e0_silu(shg) * shu, li, e0n::lora::S35_SH_DOWN);
    mx::array eg = mx::sigmoid(L.sh_egate.apply(m_in));
    x = h + out + sh * eg;
    if (clip_ > 0)
      x = mx::where(mx::isnan(x), mx::zeros_like(x),
                    mx::clip(x, mx::array(static_cast<float>(-clip_)),
                             mx::array(static_cast<float>(clip_))));
    {
      std::vector<std::pair<int64_t, double>> lv;
      for (int64_t j = 0; j < k; j++)
        lv.push_back({ip[(T - 1) * k + j], wp[(T - 1) * k + j]});
      std::sort(lv.begin(), lv.end());
      for (auto &[e, s] : lv) {
        rec.sel.inds.push_back(e);
        rec.sel.scores.push_back(s);
      }
    }
    if (!hot) {
      mx::eval(x);
      step_bundles_.clear();
    }
    return true;
  }

  std::vector<float> mhost;
  if (!to_host32(m_in, mhost, err)) return false;  // [T,H]

  const int64_t E = reg_.num_experts;
  std::vector<Selection> sels(static_cast<size_t>(T));
  bool predicted = false;
  if (decode && st.pr.take_pred(li, sels[0])) {
    predicted = true;
  } else {
    std::vector<double> lg;
    {
      mx::array glh = L.gate_q.apply(m_in);
      std::vector<float> glf;
      if (!to_host32(glh, glf, err)) return false;
      lg.assign(glf.begin(), glf.end());
    }
    for (int64_t t = 0; t < T; t++)
      sels[static_cast<size_t>(t)] = select_softmax_topk(
          lg.data() + t * E, static_cast<int>(E), reg_.top_k, reg_.norm_topk_prob);
  }
  rec.layer = li;
  rec.predicted = predicted;
  rec.sel = sels[static_cast<size_t>(T - 1)];
  std::vector<float> last_row(mhost.begin() + (T - 1) * H_, mhost.end());
  st.pr.capture(li, std::move(last_row), rec.sel.inds);

  std::vector<int64_t> uni;
  for (auto &s : sels)
    for (auto e : s.inds)
      if (uni.empty() || uni.back() != e) uni.push_back(e);
  std::sort(uni.begin(), uni.end());
  uni.erase(std::unique(uni.begin(), uni.end()), uni.end());
  auto bundles = io_->get_bundles(li, uni);
  exec_counters().leases.fetch_add(uni.size(), std::memory_order_relaxed);
  std::vector<ExpertPtrs> ptrs(uni.size());
  for (size_t i = 0; i < uni.size(); i++)
    ptrs[i] = bundle_ptrs(*bundles[uni[i]]);
  std::vector<float> routed(static_cast<size_t>(T) * H_, 0.f);
  const int F = static_cast<int>(moe_ffn_);
  {
  for (size_t ei = 0; ei < uni.size(); ei++) {
    int64_t e = uni[ei];
    std::vector<int> toks;
    for (int64_t t = 0; t < T; t++)
      for (auto id : sels[static_cast<size_t>(t)].inds)
        if (id == e) toks.push_back(static_cast<int>(t));
    if (toks.empty()) continue;
    std::vector<float> rows(toks.size() * H_);
    for (size_t k = 0; k < toks.size(); k++)
      std::memcpy(rows.data() + k * H_, mhost.data() + static_cast<size_t>(toks[k]) * H_,
                  H_ * 4);
    std::vector<int64_t> one{0};
    std::vector<float> yg, yu;
    if (!gather_qmm_mlx(ptrs[ei].gw, ptrs[ei].gs, ptrs[ei].gb, 1, F, static_cast<int>(H_),
                        static_cast<int>(H_ / 64), rows.data(), static_cast<int>(toks.size()), one,
                        yg, err) ||
        !gather_qmm_mlx(ptrs[ei].uw, ptrs[ei].us, ptrs[ei].ub, 1, F, static_cast<int>(H_),
                        static_cast<int>(H_ / 64), rows.data(), static_cast<int>(toks.size()), one,
                        yu, err))
      return false;
    std::vector<float> yh(toks.size() * moe_ffn_);
    {
      bf_round(yg);
      bf_round(yu);
      auto bf = [](float v) {
        uint32_t u; std::memcpy(&u, &v, 4);
        u += 0x7FFFu + ((u >> 16) & 1u); u >>= 16; u <<= 16;
        float o; std::memcpy(&o, &u, 4); return o;
      };
      for (size_t k = 0; k < yh.size(); k++) {
        float g = yg[k];
        float sg = bf(1.0f / (1.0f + std::exp(-g)));
        float sl = bf(g * sg);
        yh[k] = bf(sl * yu[k]);
      }
    }
    std::vector<float> yd;
    if (!gather_qmm_mlx(ptrs[ei].dw, ptrs[ei].ds, ptrs[ei].db, 1, static_cast<int>(H_), F,
                        static_cast<int>(moe_ffn_ / 64), yh.data(),
                        static_cast<int>(toks.size()), one, yd, err))
      return false;
    bf_round(yd);
    for (size_t k = 0; k < toks.size(); k++) {
      int t = toks[k];
      double w = 0.0;
      auto &s = sels[static_cast<size_t>(t)];
      for (size_t j = 0; j < s.inds.size(); j++)
        if (s.inds[j] == e) w = s.scores[j];
      for (int64_t o = 0; o < H_; o++)
        routed[static_cast<size_t>(t) * H_ + o] += static_cast<float>(w * yd[k * H_ + o]);
    }
  }
  }

  auto shg = e0n::lora::apply_lora(lo_.get(), L.sh_gate, m_in, li, e0n::lora::S35_SH_GATE);
  auto shu = e0n::lora::apply_lora(lo_.get(), L.sh_up, m_in, li, e0n::lora::S35_SH_UP);
  mx::array sh = e0n::lora::apply_lora(
      lo_.get(), L.sh_down, e0_silu(shg) * shu, li, e0n::lora::S35_SH_DOWN);  // [1,T,H]
  mx::array eg = mx::sigmoid(L.sh_egate.apply(m_in));                      // [1,T,1]
  std::vector<float> shost;
  if (!to_host32(sh * eg, shost, err)) return false;
  for (int64_t t = 0; t < T; t++)
    for (int64_t o = 0; o < H_; o++)
      routed[static_cast<size_t>(t) * H_ + o] += shost[static_cast<size_t>(t) * H_ + o];
  mx::array y_arr = mx::astype(mx::reshape(wrap_f32(routed, mkshape({1, T, H_})),
                                           mkshape({1, T, H_})),
                               dt);
  x = h + y_arr;
  if (clip_ > 0) {
    x = mx::where(mx::isnan(x), mx::zeros_like(x),
                  mx::clip(x, mx::array(static_cast<float>(-clip_)),
                           mx::array(static_cast<float>(clip_))));
  }
  mx::eval(x);
  return true;
}

bool Net35::stage_all_gpu(State35 &st, std::string &err, DecodeProbe *probe) {
  const int64_t E = reg_.num_experts;
  std::vector<int> metas;
  std::vector<mx::array> ms, curs, prevs;
  for (int o : pr_->owners()) {
    if (m_in_cache_[static_cast<size_t>(o)].ndim() < 3 ||
        oh_cur_[static_cast<size_t>(o)].ndim() == 0)
      continue;
    metas.push_back(o);
    const auto &m = m_in_cache_[static_cast<size_t>(o)];
    mx::Shape s0 = m.shape(), s1 = m.shape();
    for (auto &v : s0) v = 0;
    s0[s0.size() - 2] = static_cast<mx::Shape::value_type>(m.shape(-2) - 1);
    ms.push_back(mx::slice(m, s0, s1));
    curs.push_back(oh_cur_[static_cast<size_t>(o)]);  // oh = executed(install:183-188)
    mx::array po(0.f);
    if (oh_prev_[static_cast<size_t>(o)].ndim() > 0)
      po = oh_prev_[static_cast<size_t>(o)];
    else
      po = mx::zeros(mkshape({1, 1, E}), mx::float32);
    prevs.push_back(po);
  }
  if (metas.empty()) return true;
  const int64_t n = static_cast<int64_t>(metas.size());
  auto feats = mx::astype(
      mx::concatenate({mx::stack(ms, 0), mx::stack(curs, 0), mx::stack(prevs, 0)}, -1),
      mx::float16);                            // stager.py:158-161
  auto f2 = mx::reshape(feats, mkshape({n, H_ + 2 * E}));  // 162
  auto h1 = mx::einsum("ni,nij->nj", {f2, hw1_});          // 163
  auto act = gelu_erf16(h1);
  auto lin = mx::einsum("ni,nij->nj", {f2, hwl_});
  auto fc2 = mx::einsum("ni,nij->nj", {act, hw2_});
  auto out = lin + fc2;                                      // 164-166
  auto logits_all = mx::reshape(out, mkshape({n, 1, 1, E}));    // 167
  mx::array inds_all(0.f), scores_all(0.f);
  select_softmax_gpu(logits_all, E, reg_.top_k, reg_.norm_topk_prob, inds_all, scores_all);  // _select:241-242
  // `first_k` is a strided view over the E-wide row. Materialize both outputs before
  // the single host read; reading data() from the view as if it were [n,k] walks into
  // the next row and corrupts the staged expert set (same fix as net8b).
  auto inds_h = mx::add(inds_all, mx::array((uint32_t)0, mx::uint32));
  auto scores_h = mx::add(scores_all, mx::array(0.0f, mx::float32));
  mx::eval(inds_h, scores_h);                  // stager:193 ONE eval
  exec_counters().host_syncs.fetch_add(1, std::memory_order_relaxed);
  const int k = static_cast<int>(reg_.top_k);
  const uint32_t *ip = inds_h.data<uint32_t>();  // 194 ONE tolist
  const float *sp = scores_h.data<float>();
  for (int64_t i = 0; i < n; i++) {
    const int o = metas[i], consumer = o + 1;  // 196-197
    Selection s;                                                     // _store:244-252
    std::vector<int64_t> eids;
    for (int j = 0; j < k; j++) eids.push_back(ip[i * k + j]);
    std::sort(eids.begin(), eids.end());
    eids.erase(std::unique(eids.begin(), eids.end()), eids.end());
    for (auto e : eids) {
      double v = 0.0;
      for (int j = 0; j < k; j++)
        if (ip[i * k + j] == static_cast<uint32_t>(e)) v = sp[i * k + j];
      s.inds.push_back(e);
      s.scores.push_back(v);
    }
    if (consumer >= reg_.first_head_consumer && consumer <= reg_.last_head_consumer) {
      st.pr.pred[static_cast<size_t>(consumer)] = s;
      st.pr.pred_avail[static_cast<size_t>(consumer)] = true;
      io_->stage_experts(consumer, s.inds);
      mx::Shape a0 = inds_all.shape(), a1 = inds_all.shape();  // pred_inds[p]/pred_scores[p]
      for (auto &v : a0) v = 0;
      a0[0] = static_cast<mx::Shape::value_type>(i);
      a1[0] = static_cast<mx::Shape::value_type>(i + 1);
      pred_inds_gpu_[static_cast<size_t>(consumer)] = mx::squeeze(mx::slice(inds_all, a0, a1), 0);     // [1,1,k]
      pred_scores_gpu_[static_cast<size_t>(consumer)] = mx::squeeze(mx::slice(scores_all, a0, a1), 0);
    }
    oh_prev_[static_cast<size_t>(o)] = curs[i];
  }
  st.pr.step++;
  if (probe) {
    auto lg32 = mx::astype(logits_all, mx::float32);
    mx::eval(lg32);
    exec_counters().host_syncs.fetch_add(1, std::memory_order_relaxed);
    const float *lp = lg32.data<float>();
    probe->owners.clear();
    probe->m_in.clear();
    probe->logits.clear();
    for (int64_t i = 0; i < n; i++) {
      probe->owners.push_back(metas[i]);
      probe->logits.emplace_back(lp + i * E, lp + (i + 1) * E);
      std::vector<float> mv;
      auto r32 = mx::astype(mx::reshape(ms[i], mkshape({H_})), mx::float32);
      if (!to_host32(r32, mv, err)) return false;
      probe->m_in.push_back(std::move(mv));
    }
  }
  return true;
}

void Net35::reset_graph_caches() const {
  const size_t NL = layers_.size();
  m_in_cache_.assign(NL, mx::array(0.f));
  oh_cur_.assign(NL, mx::array(0.f));
  oh_prev_.assign(NL, mx::array(0.f));
  pred_inds_gpu_.assign(NL, mx::array(0.f));
  pred_scores_gpu_.assign(NL, mx::array(0.f));
  for (size_t li = 0; li < layers_.size(); li++) io_->reset_layer(static_cast<int>(li));
  step_bundles_.clear();
}

void Net35::prefetch(const Step &plan) {
  if (pr_ && !pr_->owners().empty()) return;  // qwen.py:195(pg_stager is not None)
  std::map<int, std::vector<int64_t>> by_layer;
  for (auto &[li, e] : plan.experts) by_layer[li].push_back(e);
  for (auto &[li, es] : by_layer) io_->prefetch(li, es);
}

void Net35::shard_warm(const std::vector<std::string> &shard_names, bool seq) {
  for (auto &sn : shard_names) {
    io_->shard_advise_whole(sn);
    if (seq) io_->shard_seq_read(sn);
  }
}

bool Net35::prefill(const std::vector<int32_t> &tokens, State35 &st,
                    std::vector<float> &logits_last, std::vector<RouteRec> &recs,
                    std::string &err, PrefillProbe *probe) {
  if (exec_mode() == ExecMode::Batched)
    return prefill_batched_(tokens, st, logits_last, recs, err, probe);
  try {
    std::vector<float> xh;
    if (!embed_lookup(embed_, tokens, H_, V_, xh, err)) return false;
    const int64_t T = static_cast<int64_t>(tokens.size());
    if (probe) probe->embed_last.assign(xh.begin() + (T - 1) * H_, xh.end());
    mx::array x = mx::astype(mx::reshape(wrap_f32(xh, mkshape({1, T, H_})), mkshape({1, T, H_})),
                             mx::bfloat16);
    recs.clear();
    for (size_t li = 0; li < layers_.size(); li++) {
      RouteRec rec;
      if (!layer_fwd(layers_[li], static_cast<int>(li), x, st, false, rec, nullptr, err))
        return false;
      if (probe) {
        mx::array tail = mx::astype(
            mx::reshape(mx::slice(x, mkshape({0, T - 1, 0}), mkshape({1, T, H_})),
                        mkshape({1, H_})),
            mx::float32);
        std::vector<float> hv;
        if (!to_host32(tail, hv, err)) return false;
        probe->per_block.push_back(std::move(hv));
      }
      recs.push_back(std::move(rec));
    }
    mx::array xs = e0_rms_norm(
        x, mx::astype(wrap_f32(final_norm_w_, mkshape({H_})), mx::bfloat16), eps_);
    mx::array last = mx::reshape(mx::slice(xs, mkshape({0, T - 1, 0}), mkshape({1, T, H_})),
                                 mkshape({1, H_}));
    if (!to_host32(lm_head_.apply(last), logits_last, err)) return false;  // [V]
    st.position += T;
  } catch (const std::exception &ex) {
    err = std::string("prefill: ") + ex.what();
    return false;
  }
  return true;
}

bool Net35::prefill_batched_(const std::vector<int32_t> &tokens, State35 &st,
                             std::vector<float> &logits_last, std::vector<RouteRec> &recs,
                             std::string &err, PrefillProbe *probe) {
  try {
    const int64_t total = static_cast<int64_t>(tokens.size());
    const int64_t chunk = reg_.prefill_chunk > 0 ? reg_.prefill_chunk : total;  // base.py:39
    recs.clear();
    std::vector<float> lg;
    for (int64_t start = 0; start < total; start += chunk) {  // base.py:80-84
      const int64_t n = std::min(chunk, total - start);
      const bool last = start + n >= total;
      step_bundles_.clear();
      if (!forward_chunk_batched_(tokens.data() + start, n, st, lg, last, recs, err,
                                  last ? probe : nullptr))
        return false;
      st.position += n;  // base.py:83 self.pos += len(chunk)
      if (!stage_all_gpu(st, err, nullptr)) return false;
    }
    prefill_end_(st);
    logits_last = std::move(lg);
    return true;
  } catch (const std::exception &ex) {
    err = std::string("prefill: ") + ex.what();
    return false;
  }
}

bool Net35::forward_chunk_batched_(const int32_t *tok, int64_t T, State35 &st,
                                   std::vector<float> &logits_out, bool to_host,
                                   std::vector<RouteRec> &recs, std::string &err,
                                   PrefillProbe *probe) {
  std::vector<float> xh;
  if (!embed_lookup(embed_, std::vector<int32_t>(tok, tok + T), H_, V_, xh, err)) return false;
  if (probe) probe->embed_last.assign(xh.begin() + (T - 1) * H_, xh.end());
  mx::array x = mx::astype(mx::reshape(wrap_f32(xh, mkshape({1, T, H_})), mkshape({1, T, H_})),
                           mx::bfloat16);
  const bool cb = T > 1 && reg_.prefill_hot > 0 && !prefill_ondemand();
  std::vector<RouteRec> local_recs;
  for (size_t li = 0; li < layers_.size(); li++) {
    if (cb) prefill_before_layer(static_cast<int>(li));  // qwen3_5.py:285-287 before_layer_cb
    RouteRec rec;
    if (!layer_fwd(layers_[li], static_cast<int>(li), x, st, false, rec, nullptr, err))
      return false;
    if (probe) {
      mx::array tail = mx::astype(
          mx::reshape(mx::slice(x, mkshape({0, T - 1, 0}), mkshape({1, T, H_})),
                      mkshape({1, H_})),
          mx::float32);
      std::vector<float> hv;
      if (!to_host32(tail, hv, err)) return false;
      probe->per_block.push_back(std::move(hv));
    }
    local_recs.push_back(std::move(rec));
  }
  mx::array xs = e0_rms_norm(
      x, mx::astype(wrap_f32(final_norm_w_, mkshape({H_})), mx::bfloat16), eps_);
  mx::array lastrow =
      mx::reshape(mx::slice(xs, mkshape({0, T - 1, 0}), mkshape({1, T, H_})), mkshape({1, H_}));
  mx::array lg = lm_head_.apply(lastrow);  // [1,V](qwen.py:177)
  if (to_host) {
    if (!to_host32(lg, logits_out, err)) return false;
    recs = std::move(local_recs);
  } else {
    mx::eval(lg);
  }
  return true;
}

void Net35::prefill_before_layer(int li) {
  const int w = std::max(1, reg_.hot_window);  // hooks.py:31
  const int ahead = std::max(1, w / 2);        // :32
  io_->clear_full_layer(li - 1);
  if (reg_.prefill_hot) {                       // :40
    const int nL = static_cast<int>(layers_.size());
    for (int lj = li; lj < std::min(li + ahead + 1, nL); lj++) {
      io_->load_hot_layer(lj, reg_.prefill_hot);
      io_->materialize_hot(lj);
    }
    for (int lj = 0; lj < std::max(0, li - (w - ahead)); lj++)
      io_->dematerialize_hot(lj);
    return;
  }
}

// prefill_end_ = qwen.py:233-250 `_prefill_end`.
void Net35::prefill_end_(State35 &st) {
  for (size_t li = 0; li < layers_.size(); li++) io_->clear_full_layer(static_cast<int>(li));
  for (size_t li = 0; li < layers_.size(); li++) io_->refresh_hot_pins(static_cast<int>(li));
  for (int li = reg_.first_head_consumer; li <= reg_.last_head_consumer; li++)
    io_->prefetch_from_prefill(li);
  (void)st;
}

bool Net35::moe_hot_branch_(int li, const mx::array &m_in, int64_t T,
                            const std::vector<int64_t> &flat, mx::array &routed,
                            std::string &err) {
  const int64_t k = reg_.top_k, TK = static_cast<int64_t>(flat.size());
  const auto &key = io_->hot_key(li);         // _hot_key(1132)
  const auto &wst = io_->hot_weights(li);     // 9×[n+1,...](1131)
  if ((int)wst.size() != 9) { err = "hot window not materialized L" + std::to_string(li); return false; }
  std::map<int64_t, int32_t> pos;             // :1132 pos={e:i}
  for (size_t j = 0; j < key.size(); j++) pos[key[j]] = (int32_t)j;
  const int32_t n_rows = (int32_t)key.size();  // :1133
  std::vector<int32_t> g(TK);
  std::vector<int64_t> miss_rows;              // :1143
  for (int64_t i = 0; i < TK; i++) {
    auto it = pos.find(flat[i]);
    if (it == pos.end()) { g[i] = n_rows; miss_rows.push_back(i); }
    else g[i] = it->second;
  }
  auto remapped = mx::array(g.begin(), mkshape({T, k}), mx::int32);  // :1151-1152
  auto x_flat = mx::reshape(m_in, mkshape({T, H_}));                 // :1136
  auto xe = mx::reshape(x_flat, mkshape({T, 1, 1, H_}));             // :1153 expand_dims((-2,-3))
  auto flatr = mx::reshape(remapped, mkshape({TK}));
  auto order = mx::argsort(flatr, -1);                               // _gather_sort 1154
  auto inv_order = mx::argsort(order, -1);
  auto xs = mx::flatten(xe, 0, -3);                                  // [T,1,H]
  auto tok_rows = mx::take(xs, mx::floor_divide(order, mx::array((uint32_t)k, mx::uint32)), 0);
  auto local = mx::take(flatr, order, 0);                            // indices[order]
  const int F = static_cast<int>(moe_ffn_);
  auto gq = [&](const mx::array &xx, int o) {  // _gather_gate_up/down unfused+sorted(1095-1104,1156-1161)
    return mx::gather_qmm(xx, wst[o], wst[o + 1], wst[o + 2], std::nullopt, local, true, 64, 4,
                          "affine", true);
  };
  auto x_up = gq(tok_rows, 0);
  auto x_gate = gq(tok_rows, 3);
  auto yh = (x_gate * mx::sigmoid(x_gate)) * x_up;                   // _swiglu(1157;layer.py:57-58)
  auto z = mx::gather_qmm(yh, wst[6], wst[7], wst[8], std::nullopt, local, true, 64, 4,
                          "affine", true);
  auto out = mx::take(z, inv_order, 0);                              // _scatter_unsort(1162)
  if (!miss_rows.empty()) {
    std::map<int64_t, std::vector<int64_t>> by_expert;
    for (auto i : miss_rows) by_expert[flat[i]].push_back(i);
    auto &cx = exec_counters();
    cx.qmm_calls.fetch_add(3 * (uint64_t)by_expert.size(), std::memory_order_relaxed);
    cx.qmm_experts.fetch_add(3 * (uint64_t)by_expert.size(), std::memory_order_relaxed);
    cx.leases.fetch_add(by_expert.size(), std::memory_order_relaxed);
    for (auto &[e, tk] : by_expert) {
      auto b = io_->build(li, e);
      if (!b) { err = "hot miss build L" + std::to_string(li); return false; }
      std::vector<int32_t> rows, fpos;
      for (auto i : tk) { rows.push_back((int32_t)(i / k)); fpos.push_back((int32_t)i); }  // :1169,1217-1218
      const int64_t m = (int64_t)rows.size();
      auto rarr = mx::array(rows.begin(), mkshape({m}), mx::int32);
      auto xe_m = mx::reshape(mx::take(x_flat, rarr, 0), mkshape({m, 1, 1, H_}));  // :1176-1177
      auto one = mx::zeros(mkshape({1}), mx::int32);                                // :1182
      auto e1 = [&](int p) {  // _e1 view-like reshape(:1180-1181)
        return mx::reshape(b->at(p), mkshape({1, b->at(p).shape(0), b->at(p).shape(1)}));
      };
      auto gqm = [&](const mx::array &xx, int o) {
        return mx::gather_qmm(xx, e1(o), e1(o + 1), e1(o + 2), std::nullopt, one, true, 64, 4,
                              "affine", true);
      };
      auto gu = gqm(xe_m, 0);
      auto gg = gqm(xe_m, 3);
      auto zm = mx::gather_qmm((gg * mx::sigmoid(gg)) * gu, e1(6), e1(7), e1(8), std::nullopt,
                               one, true, 64, 4, "affine", true);                    // :1207-1215
      auto ze = mx::squeeze(mx::squeeze(zm, -2), -2);                                // :1216 [m,H]
      auto farr = mx::array(fpos.begin(), mkshape({m}), mx::int32);
      out = mx::scatter_add(
          out, {farr}, mx::reshape(ze, mkshape({m, 1, 1, H_})), {0});
    }
  }
  routed = mx::reshape(out, mkshape({1, T, k, H_}));
  auto &cx = exec_counters();
  cx.qmm_calls.fetch_add(3, std::memory_order_relaxed);
  cx.qmm_experts.fetch_add(3 * (uint64_t)std::min<int64_t>(n_rows, TK),
                           std::memory_order_relaxed);
  cx.qmm_weight_bytes.fetch_add(
      (uint64_t)std::min<int64_t>(n_rows, TK) *
          (2 * (F * (H_ / 8) * 4 + 2 * F * (H_ / 64) * 2) +
           (H_ * (F / 8) * 4 + 2 * H_ * (F / 64) * 2)),
      std::memory_order_relaxed);
  return true;
}

bool Net35::step(int32_t token, State35 &st, std::vector<float> &logits,
                 std::vector<RouteRec> &recs, std::string &err, DecodeProbe *probe) {
  try {
    step_bundles_.clear();
    std::vector<float> xh;
    if (!embed_lookup(embed_, {token}, H_, V_, xh, err)) return false;
    mx::array x = mx::astype(mx::reshape(wrap_f32(xh, mkshape({1, 1, H_})), mkshape({1, 1, H_})),
                             mx::bfloat16);
    recs.clear();
    for (size_t li = 0; li < layers_.size(); li++) {
      RouteRec rec;
      if (!layer_fwd(layers_[li], static_cast<int>(li), x, st, true, rec, nullptr, err))
        return false;
      recs.push_back(std::move(rec));
    }
    mx::array xs = e0_rms_norm(
        x, mx::astype(wrap_f32(final_norm_w_, mkshape({H_})), mx::bfloat16), eps_);
    if (!to_host32(lm_head_.apply(mx::reshape(xs, mkshape({1, H_}))), logits, err)) return false;
    st.position += 1;
    if (exec_mode() == ExecMode::Batched) {
      if (!stage_all_gpu(st, err, probe)) return false;
    } else if (probe) {
      probe->owners.clear(); probe->m_in.clear(); probe->logits.clear();
      for (int h = 0; h < pr_->n_heads(); h++) {
        int own = pr_->owner(h);
        if (!st.pr.cap_hidden[own].empty()) {
          probe->owners.push_back(own);
          probe->m_in.push_back(st.pr.cap_hidden[own]);
        }
      }
      std::vector<std::vector<double>> diag;
      if (!st.pr.run_step_boundary(*pr_, &diag, err)) return false;
      probe->logits = std::move(diag);
    } else if (!st.pr.run_step_boundary(*pr_, nullptr, err)) return false;
  } catch (const std::exception &ex) {
    err = std::string("step: ") + ex.what();
    return false;
  }
  return true;
}

}  // namespace e0n
