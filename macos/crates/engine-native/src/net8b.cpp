#include "net8b.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <string>

#include <mlx/linalg.h>

#include "dense.hpp"
#include "qgemm.hpp"

namespace e0n {
namespace mx = mlx::core;

namespace {

using S = mx::Shape;

mx::array conv_from_torch(std::vector<uint8_t> &bytes, int64_t C, int64_t K) {
  mx::array w(bytes.data(), mkshape({C, 1, K}), mx::bfloat16, [](void *) {});
  return mx::swapaxes(w, 1, 2);
}

mx::array first_k(const mx::array &a, int64_t k) {
  mx::Shape s1 = a.shape();
  mx::Shape s0 = a.shape();
  for (auto &v : s0) v = 0;
  s1[s1.size() - 1] = static_cast<mx::Shape::value_type>(k);
  return mx::slice(a, s0, s1);
}

// moe/routing.py group_select_from_logits(30-70) ≡ bailing_hybrid.py BailingGate.__call__
void select_group_gpu(const mx::array &logits, const mx::array *bias, int64_t k, int64_t n_group,
                      int64_t topk_group, double scaling, bool norm, mx::array &idx,
                      mx::array &w) {
  auto scores = mx::sigmoid(mx::astype(logits, mx::float32));  // routing.py:41
  mx::array select = scores;                                   // 42
  if (bias) select = scores + *bias;
  const int64_t k_drop = n_group - topk_group;                 // 47
  if (k_drop > 0) {
    mx::Shape g = scores.shape();                              // 49-50 (*b_shape, n_group, E/ng)
    g[g.size() - 1] =
        static_cast<mx::Shape::value_type>(g[g.size() - 1] / n_group);
    g.insert(g.end() - 1, static_cast<mx::Shape::value_type>(n_group));
    auto grouped = mx::reshape(select, g);
    auto gs = mx::sum(mx::topk(grouped, 2, -1), std::vector<int>{-1}, false);
    auto drop = first_k(mx::argpartition(gs, static_cast<int>(k_drop - 1), -1), k_drop);  // 54-55
    auto masked =
        mx::put_along_axis(grouped, mx::expand_dims(drop, -1),
                           mx::array(-std::numeric_limits<float>::infinity(), mx::float32), -2);
    select = mx::reshape(masked, scores.shape());              // 62
  }
  idx = first_k(mx::argpartition(-select, static_cast<int>(k - 1), -1), k);  // 65
  w = mx::take_along_axis(scores, idx, -1);
  if (norm) w = w / (mx::sum(w, std::vector<int>{-1}, true) + mx::array(1e-20f, mx::float32));  // 67-68
  w = w * mx::array(static_cast<float>(scaling), mx::float32);  // 69
}

mx::array gate_logits32(const mx::array &m_in, const mx::array &gate_w) {
  return mx::astype(mx::matmul(m_in, mx::transpose(gate_w)), mx::float32);
}

bool onehot_pos(const mx::array &idx, int64_t E, int64_t pos, mx::array &oh) {
  if (idx.ndim() < 3) return false;                            // stager.py:45-46
  const int64_t t = idx.shape(-2);
  const int64_t lo = pos >= 0 ? pos : t + pos;
  if (lo < 0 || lo >= t) return false;                         // 47-49
  mx::Shape s0 = idx.shape(), s1 = idx.shape();
  for (auto &v : s0) v = 0;
  s0[s0.size() - 2] = static_cast<mx::Shape::value_type>(lo);
  s1[s1.size() - 2] = static_cast<mx::Shape::value_type>(lo + 1);
  auto row = mx::slice(idx, s0, s1);                           // 51 [..., 1, k]
  auto eq = mx::equal(mx::expand_dims(row, -1), mx::arange(0, static_cast<int>(E)));  // 52-53
  oh = mx::astype(mx::sum(eq, std::vector<int>{-2}, false), mx::float32);  // 53-54
  return true;
}

mx::array rope_interleave_torch(const mx::array &x, int64_t Tpos_start, double theta,
                                int64_t D) {
  // 98: freqs = 1.0 / (theta ** (arange(0,D,2,f32)/D))
  auto ar = mx::divide(mx::arange(0, static_cast<int>(D), 2, mx::float32),
                       mx::array(static_cast<float>(D), mx::float32));
  auto freqs = mx::divide(mx::array(1.0f, mx::float32),
                          mx::power(mx::array(static_cast<float>(theta), mx::float32), ar));
  // 99: angles = outer(positions.f32, freqs);100: emb = cat(angles,angles)
  auto positions =
      mx::astype(mx::add(mx::arange(0, static_cast<int>(x.shape(2)), mx::int32),
                         mx::array(static_cast<int32_t>(Tpos_start), mx::int32)),
                 mx::float32);
  auto angles = mx::outer(positions, freqs);                   // [L,D/2]
  auto emb = mx::concatenate({angles, angles}, -1);            // 100 [L,D]
  auto cos = mx::cos(emb);                                     // 101
  auto sin = mx::sin(emb);
  // 102-103: xi = x.reshape(B,H,L,D/2,2).transpose(0,1,2,4,3).reshape(B,H,L,D)
  const int64_t B = x.shape(0), H = x.shape(1), Ld = x.shape(2);
  auto xi = mx::reshape(
      mx::transpose(mx::reshape(x, mkshape({B, H, Ld, D / 2, 2})), {0, 1, 2, 4, 3}),
      mkshape({B, H, Ld, D}));
  // 105: rh = cat(-xi[...,D/2:], xi[...,:D/2])
  mx::Shape s0 = xi.shape(), s1 = xi.shape();
  for (auto &v : s0) v = 0;
  s1[3] = static_cast<mx::Shape::value_type>(D / 2);
  auto lo = mx::slice(xi, s0, s1);                 // xi[..., :h]
  s0[3] = static_cast<mx::Shape::value_type>(D / 2);
  s1[3] = static_cast<mx::Shape::value_type>(D);
  auto hi = mx::slice(xi, s0, s1);                 // xi[..., h:]
  auto rh = mx::concatenate({mx::negative(hi), lo}, -1);
  auto cosb = mx::reshape(cos, mkshape({1, 1, Ld, D}));
  auto sinb = mx::reshape(sin, mkshape({1, 1, Ld, D}));
  return mx::add(mx::multiply(xi, cosb), mx::multiply(rh, sinb));
}

std::vector<mx::array> moe_math_body(const std::vector<mx::array> &a) {
  auto xe = mx::reshape(a[0], mkshape({a[0].shape(0), a[0].shape(1), 1, 1, a[0].shape(2)}));  // layer.py:159
  auto gq = [&](const mx::array &xx, int o) {                  // 160-167/169-171
    return mx::gather_qmm(xx, a[o], a[o + 1], a[o + 2], std::nullopt, a[10], true, 64, 4,
                          "affine", false);
  };
  auto x_up = gq(xe, 1);
  auto x_gate = gq(xe, 4);
  auto yh = (x_gate * mx::sigmoid(x_gate)) * x_up;             // _swiglu(x_up, x_gate)
  auto z = gq(yh, 7);
  return {mx::squeeze(z, -2)};                                 // 172
}

std::vector<mx::array> moe_math_sorted_body(const std::vector<mx::array> &a) {
  auto gq = [&](const mx::array &xx, int o) {                  // 181-192
    return mx::gather_qmm(xx, a[o], a[o + 1], a[o + 2], std::nullopt, a[10], true, 64, 4,
                          "affine", true);
  };
  auto x_up = gq(a[0], 1);
  auto x_gate = gq(a[0], 4);
  auto yh = (x_gate * mx::sigmoid(x_gate)) * x_up;             // _swiglu
  return {gq(yh, 7)};
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

std::unique_ptr<Net8b> Net8b::create(std::shared_ptr<const ModelFiles> files,
                                     std::shared_ptr<const ExpertMap> emap, std::string &err,
                                     std::shared_ptr<const lora::Bundle> lo) {
  std::unique_ptr<Net8b> n(new Net8b());
  n->lo_ = lo;
  n->files_ = files;
  n->emap_ = emap;
  if (!tier_routing("edge0-8b", n->reg_)) { err = "registry missing edge0-8b"; return nullptr; }
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
  ci("hidden_size", 1536, n->H_);
  ci("vocab_size", 157184, n->V_);
  ci("num_attention_heads", 16, n->nh_);
  ci("head_dim", 128, n->hd_);
  ci("short_conv_kernel_size", 4, n->conv_k_);
  ci("moe_intermediate_size", 512, n->moe_ffn_);
  {
    int64_t shi = 512, shn = 1;
    ci("moe_shared_expert_intermediate_size", 512, shi);
    ci("num_shared_experts", 1, shn);
    n->sh_ffn_ = shi * shn;
  }
  ci("intermediate_size", 4608, n->dense_ffn_);
  ci("q_lora_rank", 256, n->qlora_);
  ci("kv_lora_rank", 512, n->kvlora_);
  ci("qk_nope_head_dim", 128, n->nope_);
  ci("qk_rope_head_dim", 64, n->roped_);
  ci("v_head_dim", 128, n->vhd_);
  cd("rms_norm_eps", 1e-6, n->eps_);
  cd("rope_theta", 6000000.0, n->rope_theta_);
  cd("kda_lower_bound", -5.0, n->kda_lb_);
  n->proj_ = n->nh_ * n->hd_;
  int64_t qbits = 4, qgroup = 64;
  if (files->config_quant(qbits, qgroup) && qgroup > 0) n->group_ = qgroup;

  int64_t interval = 4, dense_prefix_cfg = 1;
  ci("layer_group_size", 4, interval);
  ci("first_k_dense_replace", 1, dense_prefix_cfg);
  if (dense_prefix_cfg != n->reg_.dense_prefix) { err = "config dense prefix does not match registry"; return nullptr; }

  const int64_t NL = n->reg_.num_layers;
  {
    static const std::string kKdaSource = R"gdn(
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

        // g: [B, T, Hv, Dk]
        auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          if (true) {
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * g_[s_idx];
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
          g_ += Hv * Dk;
          beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<InT>(state[i]);
        }
    )gdn";
    n->kda_kernel_ = mx::fast::metal_kernel(
        "gated_delta_step_vec", {"q", "k", "v", "g", "beta", "state_in", "T"},
        {"y", "state_out"}, kKdaSource);
  }
  n->layers_.resize(static_cast<size_t>(NL));
  const std::string M = "model.layers.";
  for (int64_t li = 0; li < NL; li++) {
    Layer &L = n->layers_[static_cast<size_t>(li)];
    const std::string P = M + std::to_string(li);
    L.is_mla = ((li + 1) % interval) == 0 || li >= NL / interval * interval;
    L.has_moe = li >= n->reg_.dense_prefix;
    auto q = [&](const std::string &name, QuantLin &dst) {
      if (!dst.load(*files, P + name, err)) { err += " @" + std::to_string(li); return false; }
      n->dense_bytes_ += dst.wb.size() + dst.sb.size() + dst.bb.size();
      return true;
    };
    auto vec = [&](const std::string &name, std::vector<float> &dstv) {
      DenseF32 d;
      if (!read_dense_f32(*files, P + name, d, err)) { err += " @" + std::to_string(li); return false; }
      dstv = std::move(d.v);
      n->dense_bytes_ += dstv.size() * 4;
      return true;
    };
    if (!vec(".input_layernorm.weight", L.in_ln_w) ||
        !vec(".post_attention_layernorm.weight", L.post_ln_w))
      return nullptr;
    if (!L.is_mla) {
      if (!q(".attention.q_proj.weight", L.k_q) || !q(".attention.k_proj.weight", L.k_k) ||
          !q(".attention.v_proj.weight", L.k_v) || !q(".attention.f_proj.weight", L.k_f) ||
          !q(".attention.g_proj.weight", L.k_g) || !q(".attention.b_proj.weight", L.k_b) ||
          !q(".attention.o_proj.weight", L.k_o) ||
          !vec(".attention.A_log", L.A_log) || !vec(".attention.dt_bias", L.dt_bias) ||
          !vec(".attention.o_norm.weight", L.onorm_w))
        return nullptr;
      if (static_cast<int64_t>(L.A_log.size()) != n->nh_ ||
          static_cast<int64_t>(L.dt_bias.size()) != n->proj_ ||
          static_cast<int64_t>(L.onorm_w.size()) != n->hd_) {
        err = "KDA parameter vector length mismatch";
        return nullptr;
      }
      if (!files->read_tensor(P + ".attention.q_conv1d.weight", L.conv_qb, err) ||
          !files->read_tensor(P + ".attention.k_conv1d.weight", L.conv_kb, err) ||
          !files->read_tensor(P + ".attention.v_conv1d.weight", L.conv_vb, err))
        return nullptr;
      auto ct = files->find(P + ".attention.q_conv1d.weight")->shape;
      if (ct.size() != 3 || ct[0] != n->proj_ || ct[2] != n->conv_k_) {
        err = "KDA conv shape is not [proj,1,ksize]";
        return nullptr;
      }
      L.conv_q = conv_from_torch(L.conv_qb, n->proj_, n->conv_k_);
      L.conv_k = conv_from_torch(L.conv_kb, n->proj_, n->conv_k_);
      L.conv_v = conv_from_torch(L.conv_vb, n->proj_, n->conv_k_);
      n->dense_bytes_ += L.conv_qb.size() + L.conv_kb.size() + L.conv_vb.size();
    } else {
      if (!q(".attention.q_a_proj.weight", L.m_qa) || !q(".attention.q_b_proj.weight", L.m_qb) ||
          !q(".attention.kv_a_proj_with_mqa.weight", L.m_kva) ||
          !q(".attention.kv_b_proj.weight", L.m_kvb) || !q(".attention.g_proj.weight", L.m_g) ||
          !q(".attention.dense.weight", L.m_dense) ||
          !vec(".attention.q_a_layernorm.weight", L.m_qaln) ||
          !vec(".attention.kv_a_layernorm.weight", L.m_kvaln))
        return nullptr;
      if (static_cast<int64_t>(L.m_qaln.size()) != n->qlora_ ||
          static_cast<int64_t>(L.m_kvaln.size()) != n->kvlora_) {
        err = "MLA LoRA normalize length mismatch";
        return nullptr;
      }
    }
    if (!L.has_moe) {
      if (!q(".mlp.gate_proj.weight", L.d_gate) || !q(".mlp.up_proj.weight", L.d_up) ||
          !q(".mlp.down_proj.weight", L.d_down))
        return nullptr;
      continue;
    }
    DenseF32 g;
    const std::string gw = P + ".mlp.gate.weight";
    const TensorInfo *gi = files->find(gw);
    if (!gi) { err = "missing gate"; return nullptr; }
    if (gi->dtype == "U32") {
      int64_t bits2 = 4, gs2 = 64;
      if (!files->config_quant_for(gw, bits2, gs2) ||
          !read_dense_dequant(*files, gw, bits2, gs2, g, err))
        return nullptr;
    } else if (!read_dense_f32(*files, gw, g, err)) {
      return nullptr;
    }
    if (g.rows != n->reg_.num_experts || g.cols != n->H_) { err = "gate shape mismatch"; return nullptr; }
    L.gate_h = std::move(g.v);
    L.gate_bf16 = gi->dtype != "U32";
    n->dense_bytes_ += L.gate_h.size() * 4;
    {
      DenseF32 eb;
      if (!read_dense_f32(*files, P + ".mlp.gate.expert_bias", eb, err)) return nullptr;
      if (static_cast<int64_t>(eb.v.size()) != n->reg_.num_experts) {
        err = "expert_bias length mismatch";
        return nullptr;
      }
      L.gate_bias.assign(eb.v.begin(), eb.v.end());
      n->dense_bytes_ += L.gate_bias.size() * 4;
    }
    if (!q(".mlp.shared_experts.gate_proj.weight", L.sh_gate) ||
        !q(".mlp.shared_experts.up_proj.weight", L.sh_up) ||
        !q(".mlp.shared_experts.down_proj.weight", L.sh_down))
      return nullptr;
  }
  if (!n->embed_.load(*files, "model.word_embeddings.weight", err)) return nullptr;
  if (!n->lm_head_.load(*files, "lm_head.weight", err)) return nullptr;
  n->dense_bytes_ += n->embed_.wb.size() + n->embed_.sb.size() + n->embed_.bb.size() +
                     n->lm_head_.wb.size() + n->lm_head_.sb.size() + n->lm_head_.bb.size();
  {
    DenseF32 f;
    if (!read_dense_f32(*files, "model.norm.weight", f, err)) return nullptr;
    n->final_norm_w_ = std::move(f.v);
  }
  n->pr_ = std::make_unique<Prerouter>();
  if (!Prerouter::load(*files, n->reg_, n->H_, *n->pr_, err)) return nullptr;
  {
    const int64_t NL2 = n->reg_.num_layers;
    n->io_ = std::make_unique<ExpertIo>(files.get(), 64, 48, 8, 4);
    for (int64_t li = 0; li < NL2; li++) {
      if (!n->layers_[size_t(li)].has_moe) continue;
      ExpertIo::LayerCfg c;
      c.layer = int(li);
      c.prefix = emap->layer_prefix(int(li));
      c.E = n->reg_.num_experts;
      c.H = n->H_;
      c.F = n->moe_ffn_;
      c.group = n->group_;
      c.staged_n = int(n->reg_.top_k);
      c.prefill_hot = (int)n->reg_.prefill_hot;
      c.full_layer_prefill = n->reg_.full_layer_prefill;
      n->io_->add_layer(c);
    }
  }
  if (exec_mode() == ExecMode::Batched) {
    const char *cl = getenv("MLX_CACHE_LIMIT_MB");
    mx::set_cache_limit(static_cast<size_t>(cl ? atoi(cl) : 256) << 20);
    const int64_t E = n->reg_.num_experts, NL = n->reg_.num_layers;
    n->m_in_cache_.assign(NL, mx::array(0.f));
    n->last_topk_.assign(NL, mx::array(0.f));
    n->prev_topk_oh_.assign(NL, mx::array(0.f));
    n->pg_logits_.assign(NL, mx::array(0.f));
    n->gate_w32_.assign(NL, mx::array(0.f));
    n->gate_b32_.assign(NL, mx::array(0.f));
    for (int64_t li = 0; li < NL; li++) {
      Layer &L = n->layers_[static_cast<size_t>(li)];
      if (!L.has_moe) continue;
      auto gw =
          mx::array(L.gate_h.data(), mkshape({E, n->H_}), mx::float32, [](void *) {});
      n->gate_w32_[li] = L.gate_bf16 ? mx::astype(gw, mx::bfloat16) : gw;
      std::vector<float> b32(static_cast<size_t>(E));
      for (int64_t e = 0; e < E; e++)
        b32[static_cast<size_t>(e)] = static_cast<float>(L.gate_bias[static_cast<size_t>(e)]);
      n->gate_b32_[li] = mx::array(b32.begin(), mkshape({E}), mx::float32);
    }
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
    try {
      auto fn = mx::compile(&moe_math_body);
      const int64_t F = n->moe_ffn_, gsh = n->H_ / 64, gsf = F / 64;
      std::vector<mx::array> d = {
          mx::zeros(mkshape({1, 1, n->H_}), mx::bfloat16),
          mx::zeros(mkshape({2, F, n->H_ / 8}), mx::uint32),
          mx::zeros(mkshape({2, F, gsh}), mx::bfloat16),
          mx::zeros(mkshape({2, F, gsh}), mx::bfloat16),
          mx::zeros(mkshape({2, F, n->H_ / 8}), mx::uint32),
          mx::zeros(mkshape({2, F, gsh}), mx::bfloat16),
          mx::zeros(mkshape({2, F, gsh}), mx::bfloat16),
          mx::zeros(mkshape({2, n->H_, F / 8}), mx::uint32),
          mx::zeros(mkshape({2, n->H_, gsf}), mx::bfloat16),
          mx::zeros(mkshape({2, n->H_, gsf}), mx::bfloat16),
          mx::zeros(mkshape({1, 1, 2}), mx::int32)};
      auto out = fn(d);
      mx::eval(out[0]);
      n->moe_math_ = std::move(fn);
    } catch (const std::exception &ex) {
      std::fprintf(stderr, "edge0-8b: mlx compile unavailable (%s), MoE math falls back to eager\n",
                   ex.what());
      n->moe_math_ = &moe_math_body;
    }
    try {
      auto fn = mx::compile(&moe_math_sorted_body);
      const int64_t F = n->moe_ffn_, gsh = n->H_ / 64, gsf = F / 64;
      std::vector<mx::array> d = {
          mx::zeros(mkshape({16, 1, n->H_}), mx::bfloat16),
          mx::zeros(mkshape({2, F, n->H_ / 8}), mx::uint32),
          mx::zeros(mkshape({2, F, gsh}), mx::bfloat16),
          mx::zeros(mkshape({2, F, gsh}), mx::bfloat16),
          mx::zeros(mkshape({2, F, n->H_ / 8}), mx::uint32),
          mx::zeros(mkshape({2, F, gsh}), mx::bfloat16),
          mx::zeros(mkshape({2, F, gsh}), mx::bfloat16),
          mx::zeros(mkshape({2, n->H_, F / 8}), mx::uint32),
          mx::zeros(mkshape({2, n->H_, gsf}), mx::bfloat16),
          mx::zeros(mkshape({2, n->H_, gsf}), mx::bfloat16),
          mx::zeros(mkshape({16}), mx::int32)};
      auto out = fn(d);
      mx::eval(out[0]);
      n->moe_math_sorted_ = std::move(fn);
    } catch (const std::exception &ex) {
      std::fprintf(stderr, "edge0-8b: mlx compile unavailable (%s), sorted MoE math falls back to eager\n",
                   ex.what());
      n->moe_math_sorted_ = &moe_math_sorted_body;
    }
  }
    {
      const char *pw = getenv("EDGE0_PREWARM");
      if (!pw) pw = getenv("LING_PREWARM");
      if (pw && !strcmp(pw, "1")) {
        auto tp0 = std::chrono::steady_clock::now();
        try {
          for (const auto &sn : files->shard_names()) {  // shard.advise_willneed + seq_read
            n->io_->shard_advise_whole(sn);
            n->io_->shard_seq_read(sn);
          }
          std::string perr;
          std::vector<float> lg;
          std::vector<RouteRec> recs;
          State8b pst;
          n->init_state(pst);
          const std::vector<int32_t> dummy(4, 0);  // prefill(dummy*4)
          n->prefill(dummy, pst, lg, recs, perr);
          n->step(0, pst, lg, recs, perr);          // step(dummy[0])
          n->reset_state(pst);
          std::fprintf(stderr, "[edge0-8b] prewarm done in %.1fs\n",
                       std::chrono::duration<double>(std::chrono::steady_clock::now() - tp0).count());
        } catch (...) { /* advisory */ }
      }
    }
  if (n->lo_) {
    const int nl = n->reg_.num_layers, mla = nl / 4;
    const int want = 7 * (nl - mla) + 4 * mla + 3;
    if (n->lo_->n_bound() != want) {
      err = "LoRA bind count " + std::to_string(n->lo_->n_bound()) + " != structural count " +
            std::to_string(want);
      return nullptr;
    }
    n->dense_bytes_ += static_cast<uint64_t>(n->lo_->bytes());
  }
  return n;
}

void Net8b::init_state(State8b &st) const {
  st.pr.init(reg_);
  const size_t NL = layers_.size();
  st.cq.assign(NL, mx::array(0.f));
  st.ck.assign(NL, mx::array(0.f));
  st.cv.assign(NL, mx::array(0.f));
  st.ssm.assign(NL, mx::array(0.f));
  st.kc.assign(NL, mx::array(0.f));
  st.vc.assign(NL, mx::array(0.f));
  st.is_mla.assign(NL, 0);
  for (size_t i = 0; i < NL; i++) st.is_mla[i] = layers_[i].is_mla ? 1 : 0;
  st.position = 0;
  st.started = true;
  reset_graph_caches();
}

void Net8b::reset_state(State8b &st) const {
  st.pr.reset();
  const size_t NL = layers_.size();
  st.cq.assign(NL, mx::array(0.f));
  st.ck.assign(NL, mx::array(0.f));
  st.cv.assign(NL, mx::array(0.f));
  st.ssm.assign(NL, mx::array(0.f));
  st.kc.assign(NL, mx::array(0.f));
  st.vc.assign(NL, mx::array(0.f));
  st.position = 0;
  reset_graph_caches();
}

bool Net8b::layer_fwd(Layer &L, int li, mx::array &x, State8b &st, bool decode, bool prefill_seed,
                      RouteRec &rec, std::string &err) {
  const int64_t T = x.shape(1);
  const auto dt = x.dtype();
  auto norm_w = [&](const std::vector<float> &v) {
    return mx::astype(wrap_f32(v, mkshape({static_cast<int64_t>(v.size())})), dt);
  };
  mx::array xn = e0_rms_norm(x, norm_w(L.in_ln_w), eps_);

  mx::array r{0.f};
  if (!L.is_mla) {
    const auto *lo = lo_.get();
    mx::array q_lin = e0n::lora::apply_lora(lo, L.k_q, xn, li, e0n::lora::S8B_Q),
        k_lin = e0n::lora::apply_lora(lo, L.k_k, xn, li, e0n::lora::S8B_K),
        v_lin = e0n::lora::apply_lora(lo, L.k_v, xn, li, e0n::lora::S8B_V);
    auto conv_step = [&](const mx::array &lin, mx::array &state, const mx::array &w) {
      if (state.ndim() == 0)
        state = mx::zeros(mkshape({1, conv_k_ - 1, proj_}), dt);
      mx::array cat = mx::concatenate({state, lin}, 1);  // [1, K-1+T, proj]
      state = mx::slice(cat, mkshape({0, T, 0}), mkshape({1, T + conv_k_ - 1, proj_}));
      return e0_silu(mx::conv1d(cat, w, 1, 0, 1, static_cast<int>(proj_)));  // [1,T,proj]
    };
    mx::array qc = conv_step(q_lin, st.cq[li], L.conv_q);
    mx::array kc = conv_step(k_lin, st.ck[li], L.conv_k);
    mx::array vc = conv_step(v_lin, st.cv[li], L.conv_v);
    mx::array qf = mx::astype(mx::reshape(qc, mkshape({1, T, nh_, hd_})), mx::float32);
    mx::array kf = mx::astype(mx::reshape(kc, mkshape({1, T, nh_, hd_})), mx::float32);
    mx::array vf = mx::astype(mx::reshape(vc, mkshape({1, T, nh_, hd_})), mx::float32);
    float sc = 1.0f / std::sqrt(static_cast<float>(hd_));
    qf = sc * qf / (mx::linalg::norm(qf, 2.0, -1, true) + 1e-6f);
    kf = kf / (mx::linalg::norm(kf, 2.0, -1, true) + 1e-6f);
    mx::array gate = mx::reshape(
        e0n::lora::apply_lora(lo, L.k_g, xn, li, e0n::lora::S8B_G), mkshape({1, T, nh_, hd_}));
    mx::array f = mx::astype(mx::reshape(e0n::lora::apply_lora(lo, L.k_f, xn, li, e0n::lora::S8B_F),
                                         mkshape({1, T, nh_, hd_})),
                               mx::float32);
    mx::array dtb = mx::reshape(wrap_f32(L.dt_bias, mkshape({nh_, hd_})), mkshape({1, 1, nh_, hd_}));
    mx::array alog = wrap_f32(L.A_log, mkshape({nh_, 1}));
    mx::array glog = static_cast<float>(kda_lb_) * mx::sigmoid(mx::exp(alog) * (f + dtb));
    mx::array gdec = mx::exp(glog);
    mx::array beta = mx::sigmoid(mx::astype(
        e0n::lora::apply_lora(lo, L.k_b, xn, li, e0n::lora::S8B_B), mx::float32));  // [1,T,H]
    if (st.ssm[li].ndim() == 0)
      st.ssm[li] = mx::zeros(mkshape({1, nh_, hd_, hd_}), mx::float32);
    auto kouts = kda_kernel_(
        {qf, kf, vf, gdec, beta, st.ssm[li], mx::array((int)T, mx::int32)},
        {mkshape({1, T, nh_, hd_}), st.ssm[li].shape()},
        {mx::float32, mx::float32},
        {32, static_cast<int>(hd_), static_cast<int>(nh_)},  // grid=(32, Dv, B*Hv)，B=1
        {32, 4, 1},                                          // threadgroup(gated_delta.py:206)
        {{"InT", mx::float32},
         {"Dk", static_cast<int>(hd_)},
         {"Dv", static_cast<int>(hd_)},
         {"Hk", static_cast<int>(nh_)},
         {"Hv", static_cast<int>(nh_)}},
        std::nullopt, false, {});
    mx::array y = kouts[0];  // [1,T,nh,hd] f32
    st.ssm[li] = kouts[1];
    mx::array gated =
        e0_rms_norm(mx::astype(y, dt), norm_w(L.onorm_w), eps_) * mx::sigmoid(gate);
    r = e0n::lora::apply_lora(lo, L.k_o, mx::reshape(gated, mkshape({1, T, proj_})), li,
                              e0n::lora::S8B_O);
  } else {
    mx::array q_lat = e0_rms_norm(L.m_qa.apply(xn), norm_w(L.m_qaln), eps_);
    mx::array qfull = mx::transpose(
        mx::reshape(e0n::lora::apply_lora(lo_.get(), L.m_qb, q_lat, li, e0n::lora::S8B_M_QB),
                    mkshape({1, T, nh_, nope_ + roped_})),
        {0, 2, 1, 3});
    mx::eval(qfull);
    auto qs = mx::split(qfull, S{static_cast<int32_t>(nope_)}, -1);
    mx::array q_nope = qs[0], q_pe = qs[1];
    mx::array comp = L.m_kva.apply(xn);  // [1,T,kvlora+rope]
    mx::eval(comp);
    auto cs = mx::split(comp, S{static_cast<int32_t>(kvlora_)}, -1);
    mx::array kv_lat = e0_rms_norm(cs[0], norm_w(L.m_kvaln), eps_);
    mx::array kv = mx::transpose(
        mx::reshape(
            e0n::lora::apply_lora(lo_.get(), L.m_kvb, kv_lat, li, e0n::lora::S8B_M_KVB),
            mkshape({1, T, nh_, nope_ + vhd_})),
        {0, 2, 1, 3});
    auto ks = mx::split(kv, S{static_cast<int32_t>(nope_)}, -1);
    mx::array k_nope = ks[0], values = ks[1];
    mx::array k_pe = mx::broadcast_to(
        rope_interleave_torch(
            mx::reshape(mx::slice(comp, mkshape({0, 0, kvlora_}),
                                  mkshape({1, T, kvlora_ + roped_})),
                        mkshape({1, 1, T, roped_})),
            st.position, rope_theta_, roped_),
        mkshape({1, nh_, T, roped_}));
    q_pe = rope_interleave_torch(q_pe, st.position, rope_theta_, roped_);
    mx::array queries = mx::concatenate({q_nope, q_pe}, -1);   // [1,nh,T,192]
    mx::array keys = mx::concatenate({k_nope, k_pe}, -1);
    if (st.kc[li].ndim() == 0) {
      st.kc[li] = keys;
      st.vc[li] = values;
    } else {
      st.kc[li] = mx::concatenate({st.kc[li], keys}, 2);
      st.vc[li] = mx::concatenate({st.vc[li], values}, 2);
    }
    float scale = 1.0f / std::sqrt(static_cast<float>(nope_ + roped_));
    mx::array o = mx::fast::scaled_dot_product_attention(
        queries, st.kc[li], st.vc[li], scale, "causal");  // [1,nh,T,vhd]
    o = mx::reshape(mx::transpose(o, {0, 2, 1, 3}), mkshape({1, T, nh_ * vhd_}));
    mx::array gate = mx::sigmoid(
        e0n::lora::apply_lora(lo_.get(), L.m_g, xn, li, e0n::lora::S8B_M_G));  // [1,T,nh] head-wise
    o = mx::reshape(mx::reshape(o, mkshape({1, T, nh_, vhd_})) *
                        mx::expand_dims(gate, -1),
                    mkshape({1, T, nh_ * vhd_}));
    r = e0n::lora::apply_lora(lo_.get(), L.m_dense, o, li, e0n::lora::S8B_M_DENSE);
  }

  mx::array h = x + r;
  mx::array m_in = e0_rms_norm(h, norm_w(L.post_ln_w), eps_);
  if (!L.has_moe) {
    mx::array y =
        e0n::lora::apply_lora(
            lo_.get(), L.d_down,
            e0_silu(e0n::lora::apply_lora(lo_.get(), L.d_gate, m_in, li, e0n::lora::S8B_D_GATE)) *
                e0n::lora::apply_lora(lo_.get(), L.d_up, m_in, li, e0n::lora::S8B_D_UP),
            li, e0n::lora::S8B_D_DOWN);
    x = h + y;
    if (clip_ > 0)
      x = mx::where(mx::isnan(x), mx::zeros_like(x),
                    mx::clip(x, mx::array(static_cast<float>(-clip_)),
                             mx::array(static_cast<float>(clip_))));
    if (!(exec_mode() == ExecMode::Batched && decode && T == 1))
      mx::eval(x);
    return true;
  }

  // bailing_hybrid.py BailingSparseMoE.__call__(679-695)/DecoderLayer.__call__(739-762):
  if (exec_mode() == ExecMode::Batched && decode && T == 1) {
    const int64_t k = reg_.top_k;
    io_->note_call(li);
    const bool owner = li >= reg_.first_owner && li <= reg_.last_owner;
    const bool consumer = li >= reg_.first_head_consumer && li <= reg_.last_head_consumer;
    if (owner) m_in_cache_[static_cast<size_t>(li)] = m_in;
    io_->wait_staged(li);
    const auto &st_st = io_->staged(li);
    const bool use_pg =
        consumer && pg_logits_[static_cast<size_t>(li)].ndim() > 0 &&
        !st.pr.pred[static_cast<size_t>(li)].inds.empty() &&
        st_st.active && st_st.exp == st.pr.pred[static_cast<size_t>(li)].inds;
    mx::array idx(0.f), w(0.f), routed(0.f), teach(0.f);
    rec.layer = li;
    if (use_pg) {
      select_group_gpu(pg_logits_[static_cast<size_t>(li)], nullptr, k, reg_.n_group,
                       reg_.topk_group, reg_.routed_scaling, reg_.norm_topk_prob, idx, w);
      if (owner) {
        auto gl = gate_logits32(m_in, gate_w32_[static_cast<size_t>(li)]);
        mx::array w_unused(0.f);
        select_group_gpu(gl, &gate_b32_[static_cast<size_t>(li)], k, reg_.n_group,
                         reg_.topk_group, reg_.routed_scaling, reg_.norm_topk_prob, teach,
                         w_unused);
      }
      rec.predicted = true;
      rec.sel = st.pr.pred[static_cast<size_t>(li)];
      st.pr.pred_avail[static_cast<size_t>(li)] = false;
      auto asm_ = io_->asm_for(li, st_st);
      auto local = mx::take(asm_.slot_of, idx);
      std::vector<mx::array> args{m_in};
      for (auto &a : asm_.wargs) args.push_back(a);
      args.push_back(local);
      auto &cx = exec_counters();
      cx.qmm_calls.fetch_add(3, std::memory_order_relaxed);
      cx.qmm_experts.fetch_add(3 * static_cast<uint64_t>(asm_.exp.size()), std::memory_order_relaxed);
      cx.qmm_weight_bytes.fetch_add(
          static_cast<uint64_t>(asm_.exp.size()) *
              (2 * (moe_ffn_ * (H_ / 8) * 4 + 2 * moe_ffn_ * (H_ / group_) * 2) +
               (H_ * (moe_ffn_ / 8) * 4 + 2 * H_ * (moe_ffn_ / group_) * 2)),
          std::memory_order_relaxed);
      routed = moe_math_(args)[0];  // layer.py:1359 self._moe_math(x, *wargs, local2d)
    } else {
      const int F = static_cast<int>(moe_ffn_);
      const int gsh = static_cast<int>(H_ / group_), gsf = static_cast<int>(moe_ffn_ / group_);
      auto gl = gate_logits32(m_in, gate_w32_[static_cast<size_t>(li)]);
      select_group_gpu(gl, &gate_b32_[static_cast<size_t>(li)], k, reg_.n_group, reg_.topk_group,
                       reg_.routed_scaling, reg_.norm_topk_prob, idx, w);
      if (owner) teach = idx;
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
      std::vector<int64_t> local_h(flat.size());
      for (size_t j = 0; j < flat.size(); j++)
        local_h[j] = static_cast<int32_t>(
            std::lower_bound(uniq.begin(), uniq.end(), flat[j]) - uniq.begin());
      auto bundles = io_->get_bundles(li, uniq);
      if (bundles.size() != uniq.size()) {
        err = "expert load missing items L" + std::to_string(li);
        return false;
      }
      exec_counters().leases.fetch_add(uniq.size(), std::memory_order_relaxed);
      for (auto &[e, b] : bundles) step_bundles_.push_back(b);
      std::vector<mx::array> args{m_in};
      for (int p = 0; p < 9; p++) {
        std::vector<mx::array> rows;
        for (auto e : uniq) rows.push_back(bundles[e]->at(p));
        args.push_back(mx::stack(rows));  // [U,...](1380)
      }
      args.push_back(mx::array(local_h.begin(), mkshape({1, 1, k}), mx::int32));  // 1371 core.array(list)
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
    if (owner) last_topk_[static_cast<size_t>(li)] = use_pg ? teach : idx;
    auto out = mx::sum(routed * mx::astype(mx::expand_dims(w, -1), routed.dtype()),
                       std::vector<int>{-2}, false);
    mx::array sh =
        L.sh_down.apply(e0_silu(L.sh_gate.apply(m_in)) * L.sh_up.apply(m_in));  // BailingMLP:503
    x = h + out + sh;
    if (clip_ > 0)
      x = mx::where(mx::isnan(x), mx::zeros_like(x),
                    mx::clip(x, mx::array(static_cast<float>(-clip_)),
                             mx::array(static_cast<float>(clip_))));
    return true;
  }

  //     before_layer_cb load-drop hooks.py:35-58 window=1 + async_eval_per_layer
  if (exec_mode() == ExecMode::Batched) {
    const int64_t k = reg_.top_k;
    const int64_t E = reg_.num_experts;
    io_->note_call(li);
    const bool owner = li >= reg_.first_owner && li <= reg_.last_owner;
    if (owner) m_in_cache_[static_cast<size_t>(li)] = m_in;
    e0n::TimeSink gt_cpu(e0n::timing_counters().gate_cpu_ns);
    auto gl = gate_logits32(m_in, gate_w32_[static_cast<size_t>(li)]);  // :527
    mx::array idx(0.f), w(0.f);
    select_group_gpu(gl, &gate_b32_[static_cast<size_t>(li)], k, reg_.n_group, reg_.topk_group,
                     reg_.routed_scaling, reg_.norm_topk_prob, idx, w);
    if (owner) last_topk_[static_cast<size_t>(li)] = idx;
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
    const bool full = !prefill_ondemand() && io_->full_ready(li) && T * k > 8 &&
                      !io_->hot_ready(li);
    if (full) {
      io_->note_prefill_use(li, flat, (int)k,
                            li >= reg_.first_head_consumer && li <= reg_.last_head_consumer);
      const auto &fw = io_->full_weights(li);
      if ((int)fw.size() != 9) { err = "full-layer stack not loaded L" + std::to_string(li); return false; }
      auto xe = mx::reshape(m_in, mkshape({1, T, 1, 1, H_}));           // 1246 expand_dims((-2,-3))
      auto iflat = mx::reshape(idx, mkshape({T * k}));
      auto order = mx::argsort(iflat, -1);                              // _gather_sort(switch_layers.py:12-17)
      auto inv_order = mx::argsort(order, -1);
      auto xs = mx::flatten(xe, 0, -3);                                 // [T,1,H]
      auto tok_rows = mx::take(
          xs, mx::floor_divide(order, mx::array((uint32_t)k, mx::uint32)), 0);  // x[order//M]
      auto local = mx::take(iflat, order, 0);
      auto gq = [&](const mx::array &xx, int o) {
        return mx::gather_qmm(xx, fw[o], fw[o + 1], fw[o + 2], std::nullopt, local, true, 64, 4,
                              "affine", true);
      };
      auto x_up = gq(tok_rows, 0);
      auto x_gate = gq(tok_rows, 3);
      auto yh = (x_gate * mx::sigmoid(x_gate)) * x_up;                   // _swiglu(1250;layer.py:57-58)
      auto z = gq(yh, 6);
      auto z_un = mx::take(z, inv_order, 0);                             // _scatter_unsort(1260)
      routed = mx::squeeze(mx::reshape(z_un, mkshape({1, T, k, 1, H_})), -2);
      auto &cx = exec_counters();
      cx.qmm_calls.fetch_add(3, std::memory_order_relaxed);
      cx.qmm_experts.fetch_add(3 * static_cast<uint64_t>(E), std::memory_order_relaxed);
      cx.qmm_weight_bytes.fetch_add(
          static_cast<uint64_t>(E) *
              (2 * (moe_ffn_ * (H_ / 8) * 4 + 2 * moe_ffn_ * (H_ / group_) * 2) +
               (H_ * (moe_ffn_ / 8) * 4 + 2 * H_ * (moe_ffn_ / group_) * 2)),
          std::memory_order_relaxed);
    }
    if (!full) {
      std::vector<int64_t> uniq = flat;                       // 1364 sorted(set(int(v) for v in flat))
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    std::vector<int32_t> local_h(static_cast<size_t>(T * k));  // 1370-1373 host remap
    for (int64_t i = 0; i < T * k; i++)
      local_h[static_cast<size_t>(i)] = static_cast<int32_t>(
          std::lower_bound(uniq.begin(), uniq.end(), flat[static_cast<size_t>(i)]) - uniq.begin());
    auto bundles = io_->get_bundles(li, uniq);
    if (bundles.size() != uniq.size()) {
      err = "expert load missing items L" + std::to_string(li);
      return false;
    }
    exec_counters().leases.fetch_add(uniq.size(), std::memory_order_relaxed);
    for (auto &[e, b] : bundles) step_bundles_.push_back(b);
    std::vector<mx::array> wargs;                            // 1376-1380 wargs=core.stack(rows)
    for (int p = 0; p < 9; p++) {
      std::vector<mx::array> rows;
      for (auto e : uniq) rows.push_back(bundles[e]->at(p));
      wargs.push_back(mx::stack(rows));
    }
    auto local_arr = mx::array(local_h.begin(), mkshape({1, T, k}), mx::int32);  // 1371 core.array(list)
    const int F = static_cast<int>(moe_ffn_);
    const int gsh = static_cast<int>(H_ / group_), gsf = static_cast<int>(moe_ffn_ / group_);
    auto &cx = exec_counters();
    cx.qmm_calls.fetch_add(3, std::memory_order_relaxed);
    cx.qmm_experts.fetch_add(3 * static_cast<uint64_t>(uniq.size()), std::memory_order_relaxed);
    cx.qmm_weight_bytes.fetch_add(
        static_cast<uint64_t>(uniq.size()) *
            (2 * (F * (H_ / 8) * 4 + 2 * F * gsh * 2) + (H_ * (F / 8) * 4 + 2 * H_ * gsf * 2)),
        std::memory_order_relaxed);
    if (T * k >= 64) {
      auto flat_local = mx::reshape(local_arr, mkshape({T * k}));
      auto order = mx::argsort(flat_local, -1);             // _gather_sort: order=argsort(flat)
      auto inv_order = mx::argsort(order, -1);              //                     inv=argsort(order)
      auto xe = mx::reshape(m_in, mkshape({1, T, 1, 1, H_}));  // 1383 expand_dims((-2,-3)) [B,T,1,1,H]
      auto xs = mx::flatten(xe, 0, -3);                     // 13-15 [T,1,H]
      auto tok_rows = mx::take(
          xs, mx::floor_divide(order, mx::array((uint32_t)k, mx::uint32)), 0);
      auto local_sorted = mx::take(flat_local, order, 0);   // indices[order]
      std::vector<mx::array> sargs{tok_rows};
      for (auto &wa : wargs) sargs.push_back(wa);
      sargs.push_back(local_sorted);
      auto z = moe_math_sorted_(sargs)[0];
      auto z_un = mx::take(z, inv_order, 0);
      routed = mx::squeeze(mx::reshape(z_un, mkshape({1, T, k, 1, H_})), -2);  // unflatten+squeeze(-2)
    } else {
      std::vector<mx::array> args{m_in};
      for (auto &wa : wargs) args.push_back(wa);
      args.push_back(local_arr);
      routed = moe_math_(args)[0];
    }
    }
    auto out = mx::sum(routed * mx::astype(mx::expand_dims(w, -1), routed.dtype()),
                       std::vector<int>{-2}, false);
    mx::array sh =
        L.sh_down.apply(e0_silu(L.sh_gate.apply(m_in)) * L.sh_up.apply(m_in));  // BailingMLP:503
    gt_cpu.stop();
    x = h + out + sh;  // :762
    if (clip_ > 0)
      x = mx::where(mx::isnan(x), mx::zeros_like(x),
                    mx::clip(x, mx::array(static_cast<float>(-clip_)),
                             mx::array(static_cast<float>(clip_))));
    rec.layer = li;
    rec.predicted = false;
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
    if (full) {
      mx::async_eval(x);
    } else {
      mx::eval(x);
      step_bundles_.clear();
    }
    return true;
  }

  std::vector<float> mhost;
  if (!to_host32(m_in, mhost, err)) return false;  // [T,H]

  e0n::TimeSink gt_cpu(e0n::timing_counters().gate_cpu_ns);
  const int64_t E = reg_.num_experts;
  const double *bias = L.gate_bias.empty() ? nullptr : L.gate_bias.data();
  std::vector<Selection> sels(static_cast<size_t>(T));
  bool predicted = false;
  if (decode && st.pr.take_pred(li, sels[0])) {
    predicted = true;
  } else {
    std::vector<double> lg;
    gate_logits(mhost.data(), static_cast<int>(T), static_cast<int>(H_), L.gate_h.data(),
                static_cast<int>(E), lg);
    for (int64_t t = 0; t < T; t++)
      sels[static_cast<size_t>(t)] = select_sigmoid_group(
          lg.data() + t * E, bias, static_cast<int>(E), reg_.top_k, reg_.n_group,
          reg_.topk_group, reg_.routed_scaling, reg_.norm_topk_prob);
  }
  rec.layer = li;
  rec.predicted = predicted;
  rec.sel = sels[static_cast<size_t>(T - 1)];

  const bool owner = li >= reg_.first_owner && li <= reg_.last_owner;
  if (owner) {
    std::vector<int64_t> feat = rec.sel.inds;
    if (predicted && reg_.head_feature_teacher) {
      std::vector<double> lg;
      gate_logits(mhost.data() + (T - 1) * H_, 1, static_cast<int>(H_), L.gate_h.data(),
                  static_cast<int>(E), lg);
      feat = select_sigmoid_group(lg.data(), bias, static_cast<int>(E), reg_.top_k, reg_.n_group,
                                  reg_.topk_group, reg_.routed_scaling, reg_.norm_topk_prob)
                 .inds;
    }
    std::vector<float> last_row(mhost.begin() + (T - 1) * H_, mhost.end());
    st.pr.capture(li, std::move(last_row), std::move(feat));
    if (prefill_seed && reg_.head_feature_teacher) {
      std::vector<int64_t> prev;
      if (T >= 2) {
        std::vector<double> lg;
        gate_logits(mhost.data() + (T - 2) * H_, 1, static_cast<int>(H_), L.gate_h.data(),
                    static_cast<int>(E), lg);
        prev = select_sigmoid_group(lg.data(), bias, static_cast<int>(E), reg_.top_k,
                                    reg_.n_group, reg_.topk_group, reg_.routed_scaling,
                                    reg_.norm_topk_prob)
                   .inds;
      }
      st.pr.prev_exec[li] = std::move(prev);
    }
  }
  gt_cpu.stop();

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
  const int gsh = static_cast<int>(H_ / group_), gsf = static_cast<int>(moe_ffn_ / group_);
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
    if (!gather_qmm_mlx(ptrs[ei].gw, ptrs[ei].gs, ptrs[ei].gb, 1, F, static_cast<int>(H_), gsh,
                        rows.data(), static_cast<int>(toks.size()), one, yg, err) ||
        !gather_qmm_mlx(ptrs[ei].uw, ptrs[ei].us, ptrs[ei].ub, 1, F, static_cast<int>(H_), gsh,
                        rows.data(), static_cast<int>(toks.size()), one, yu, err))
      return false;
    std::vector<float> yh(toks.size() * moe_ffn_);
    for (size_t k = 0; k < yh.size(); k++) yh[k] = silu_f(yg[k]) * yu[k];
    std::vector<float> yd;
    if (!gather_qmm_mlx(ptrs[ei].dw, ptrs[ei].ds, ptrs[ei].db, 1, static_cast<int>(H_), F, gsf,
                        yh.data(), static_cast<int>(toks.size()), one, yd, err))
      return false;
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

  mx::array sh =
      L.sh_down.apply(e0_silu(L.sh_gate.apply(m_in)) * L.sh_up.apply(m_in));  // [1,T,H]
  std::vector<float> shost;
  if (!to_host32(sh, shost, err)) return false;
  for (int64_t t = 0; t < T; t++)
    for (int64_t o = 0; o < H_; o++)
      routed[static_cast<size_t>(t) * H_ + o] += shost[static_cast<size_t>(t) * H_ + o];
  mx::array y_arr = mx::astype(wrap_f32(routed, mkshape({1, T, H_})), dt);
  x = h + y_arr;
  if (clip_ > 0) {
    x = mx::where(mx::isnan(x), mx::zeros_like(x),
                  mx::clip(x, mx::array(static_cast<float>(-clip_)),
                           mx::array(static_cast<float>(clip_))));
  }
  E0_TIME(e0n::timing_counters().eval_ns);
  mx::eval(x);
  return true;
}

bool Net8b::stage_all_gpu(State8b &st, std::string &err, DecodeProbe *probe) {
  const int64_t E = reg_.num_experts;
  std::vector<int> metas;
  std::vector<mx::array> ms, curs, prevs;
  for (int o : pr_->owners()) {
    const auto &m = m_in_cache_[static_cast<size_t>(o)];
    if (m.ndim() < 3) continue;
    metas.push_back(o);
    mx::Shape s0 = m.shape(), s1 = m.shape();
    for (auto &v : s0) v = 0;
    s0[s0.size() - 2] = static_cast<mx::Shape::value_type>(m.shape(-2) - 1);
    ms.push_back(mx::slice(m, s0, s1));
    mx::array co(0.f);                                  // 284 cur=pos-1 one-hot
    if (!onehot_pos(last_topk_[static_cast<size_t>(o)], E, -1, co))
      co = mx::zeros(mkshape({1, 1, E}), mx::float32);
    curs.push_back(co);
    mx::array po(0.f);
    if (prev_topk_oh_[static_cast<size_t>(o)].ndim() > 0)
      po = prev_topk_oh_[static_cast<size_t>(o)];
    else if (!onehot_pos(last_topk_[static_cast<size_t>(o)], E, -2, po))
      po = mx::zeros(mkshape({1, 1, E}), mx::float32);  // 292-293
    prevs.push_back(po);
  }
  if (metas.empty()) return true;                  // 185-186
  const int64_t n = static_cast<int64_t>(metas.size());
  auto s_ms = mx::stack(ms, 0), s_curs = mx::stack(curs, 0), s_prevs = mx::stack(prevs, 0);
  auto feats = mx::astype(mx::concatenate({s_ms, s_curs, s_prevs}, -1), mx::float16);  // 158-161
  auto f2 = mx::reshape(feats, mkshape({n, H_ + 2 * E}));  // 162
  auto gel = [](const mx::array &x) {
    return mx::array(0.5f, mx::float16) * x *
           (mx::array(1.0f, mx::float16) +
            mx::erf(x / mx::array(static_cast<float>(std::sqrt(2.0)), mx::float16)));
  };
  auto h1 = mx::einsum("ni,nij->nj", {f2, hw1_});  // 163
  auto out = mx::einsum("ni,nij->nj", {f2, hwl_}) +
             mx::einsum("ni,nij->nj", {gel(h1), hw2_});  // 165-166
  auto logits_all = mx::reshape(out, mkshape({n, 1, 1, E}));  // 167
  mx::array inds_all(0.f), scores_all(0.f);
  select_group_gpu(logits_all, nullptr, reg_.top_k, reg_.n_group, reg_.topk_group,
                   reg_.routed_scaling, reg_.norm_topk_prob, inds_all, scores_all);  // _select 296-300
  auto inds_h = mx::add(inds_all, mx::array((uint32_t)0, mx::uint32));
  auto scores_h = mx::add(scores_all, mx::array(0.0f, mx::float32));
  mx::eval(inds_h, scores_h);                      // 193 ONE eval
  exec_counters().host_syncs.fetch_add(1, std::memory_order_relaxed);
  const int k = static_cast<int>(reg_.top_k);
  const uint32_t *ip = inds_h.data<uint32_t>();  // 194 ONE tolist
  const float *sp = scores_h.data<float>();
  for (int64_t i = 0; i < n; i++) {
    const int o = metas[i], consumer = static_cast<int>(o + 1);
    Selection s;
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
    if (consumer <= reg_.last_head_consumer) {
      st.pr.pred[static_cast<size_t>(consumer)] = s;
      st.pr.pred_avail[static_cast<size_t>(consumer)] = true;
      io_->stage_experts(consumer, s.inds);
      mx::Shape a0 = logits_all.shape(), a1 = logits_all.shape();
      for (auto &v : a0) v = 0;
      a0[0] = static_cast<mx::Shape::value_type>(i);
      a1[0] = static_cast<mx::Shape::value_type>(i + 1);
      pg_logits_[static_cast<size_t>(consumer)] =
          mx::squeeze(mx::slice(logits_all, a0, a1), 0);
    }
    prev_topk_oh_[static_cast<size_t>(o)] = curs[i];
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

void Net8b::reset_graph_caches() const {
  const size_t NL = layers_.size();
  m_in_cache_.assign(NL, mx::array(0.f));
  last_topk_.assign(NL, mx::array(0.f));
  prev_topk_oh_.assign(NL, mx::array(0.f));
  pg_logits_.assign(NL, mx::array(0.f));
  for (auto &L : layers_)
    if (L.has_moe) io_->reset_layer(static_cast<int>(&L - layers_.data()));  // layer.py:1429-1441
  step_bundles_.clear();
}

void Net8b::prefetch(const Step &plan) {
  std::map<int, std::vector<int64_t>> by_layer;
  for (auto &[li, e] : plan.experts) by_layer[li].push_back(e);
  for (auto &[li, es] : by_layer) io_->prefetch(li, es);
}

void Net8b::shard_warm(const std::vector<std::string> &shard_names, bool seq) {
  for (auto &s : shard_names) {
    io_->shard_advise_whole(s);
    if (seq) io_->shard_seq_read(s);
  }
}

bool Net8b::prefill(const std::vector<int32_t> &tokens, State8b &st,
                    std::vector<float> &logits_last, std::vector<RouteRec> &recs,
                    std::string &err, PrefillProbe *probe) {
  if (exec_mode() == ExecMode::Batched)
    return prefill_batched_(tokens, st, logits_last, recs, err, probe);
  try {
    std::vector<float> xh;
    if (!embed_lookup(embed_, tokens, H_, V_, xh, err)) return false;
    const int64_t T = static_cast<int64_t>(tokens.size());
    if (probe) probe->embed_last.assign(xh.begin() + (T - 1) * H_, xh.end());
    mx::array x = mx::astype(wrap_f32(xh, mkshape({1, T, H_})), mx::bfloat16);
    recs.clear();
    for (size_t li = 0; li < layers_.size(); li++) {
      RouteRec rec;
      if (!layer_fwd(layers_[li], static_cast<int>(li), x, st, false, true, rec, err))
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
      if (layers_[li].has_moe) recs.push_back(std::move(rec));
    }
    mx::array xs = e0_rms_norm(
        x, mx::astype(wrap_f32(final_norm_w_, mkshape({H_})), mx::bfloat16), eps_);
    mx::array last = mx::reshape(mx::slice(xs, mkshape({0, T - 1, 0}), mkshape({1, T, H_})),
                                 mkshape({1, H_}));
    if (!to_host32(lm_head_.apply(last), logits_last, err)) return false;  // [V]
    st.position += T;
    if (!st.pr.run_step_boundary(*pr_, nullptr, err)) return false;
  } catch (const std::exception &ex) {
    err = std::string("prefill: ") + ex.what();
    return false;
  }
  return true;
}

bool Net8b::prefill_batched_(const std::vector<int32_t> &tokens, State8b &st,
                             std::vector<float> &logits_last, std::vector<RouteRec> &recs,
                             std::string &err, PrefillProbe *probe) {
  try {
    const int64_t total = static_cast<int64_t>(tokens.size());
    const int64_t chunk = reg_.prefill_chunk > 0 ? reg_.prefill_chunk : total;  // base.py:39(wrapper :71=2048)
    recs.clear();
    std::vector<float> lg;
    for (int64_t start = 0; start < total; start += chunk) {
      const int64_t n = std::min(chunk, total - start);
      const bool last = start + n >= total;
      step_bundles_.clear();
      if (!forward_chunk_batched_(tokens.data() + start, n, st, lg, last, recs, err,
                                  last ? probe : nullptr))
        return false;
      st.position += n;  // base.py:83
    }
    if (!prefill_end_(st, err)) return false;
    logits_last = std::move(lg);
    return true;
  } catch (const std::exception &ex) {
    err = std::string("prefill: ") + ex.what();
    return false;
  }
}

bool Net8b::forward_chunk_batched_(const int32_t *tok, int64_t T, State8b &st,
                                   std::vector<float> &logits_out, bool to_host,
                                   std::vector<RouteRec> &recs, std::string &err,
                                   PrefillProbe *probe) {
  std::vector<float> xh;
  if (!embed_lookup(embed_, std::vector<int32_t>(tok, tok + T), H_, V_, xh, err)) return false;
  if (probe) probe->embed_last.assign(xh.begin() + (T - 1) * H_, xh.end());
  mx::array x = mx::astype(wrap_f32(xh, mkshape({1, T, H_})), mx::bfloat16);
  const bool cb = T > 1 && !prefill_ondemand() &&
                  (reg_.full_layer_prefill || reg_.prefill_hot > 0);
  std::vector<RouteRec> local_recs;
  for (size_t li = 0; li < layers_.size(); li++) {
    if (cb) prefill_before_layer(static_cast<int>(li));
    RouteRec rec;
    if (!layer_fwd(layers_[li], static_cast<int>(li), x, st, false, true, rec, err))
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
    if (layers_[li].has_moe) local_recs.push_back(std::move(rec));
  }
  mx::array xs = e0_rms_norm(
      x, mx::astype(wrap_f32(final_norm_w_, mkshape({H_})), mx::bfloat16), eps_);
  mx::array lastrow =
      mx::reshape(mx::slice(xs, mkshape({0, T - 1, 0}), mkshape({1, T, H_})), mkshape({1, H_}));
  mx::array lg = lm_head_.apply(lastrow);  // ling.py:220
  if (to_host) {
    if (!to_host32(lg, logits_out, err)) return false;
    recs = std::move(local_recs);
  } else {
    mx::eval(lg);
  }
  return true;
}

void Net8b::prefill_before_layer(int li) {
  io_->clear_full_layer(li - 1);
  if (li >= 0 && li < (int)layers_.size() && layers_[size_t(li)].has_moe)
    io_->load_full_layer(li);
}

bool Net8b::prefill_end_(State8b &st, std::string &err) {
  for (size_t li = 0; li < layers_.size(); li++) io_->clear_full_layer(static_cast<int>(li));  // :278-279
  return stage_all_gpu(st, err, nullptr);
}

bool Net8b::step(int32_t token, State8b &st, std::vector<float> &logits,
                 std::vector<RouteRec> &recs, std::string &err, DecodeProbe *probe) {
  try {
    step_bundles_.clear();
    std::vector<float> xh;
    if (!embed_lookup(embed_, {token}, H_, V_, xh, err)) return false;
    mx::array x = mx::astype(wrap_f32(xh, mkshape({1, 1, H_})), mx::bfloat16);
    recs.clear();
    for (size_t li = 0; li < layers_.size(); li++) {
      RouteRec rec;
      if (!layer_fwd(layers_[li], static_cast<int>(li), x, st, true, false, rec, err))
        return false;
      if (layers_[li].has_moe) recs.push_back(std::move(rec));
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
