#include "prerouter.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <mlx/mlx.h>

#include "dense.hpp"

using mlx::core::array;
using mlx::core::Device;

namespace e0n {
namespace {

uint16_t to_f16_bits1(float v) {
  _Float16 h = static_cast<_Float16>(v);
  uint16_t b;
  std::memcpy(&b, &h, 2);
  return b;
}

float f16_bits_to_f(uint16_t b) {
  _Float16 h;
  std::memcpy(&h, &b, 2);
  return static_cast<float>(h);
}

std::vector<float> expand_f16(const std::vector<float> &v) {
  std::vector<float> o(v.size());
  for (size_t i = 0; i < v.size(); i++) o[i] = f16_bits_to_f(to_f16_bits1(v[i]));
  return o;
}

}  // namespace

struct Prerouter::Impl {
  TierRouting reg;
  int64_t hidden = 0, f_in = 0;
  std::vector<int> owners;
  std::vector<float> fc1, fc2, lin;
  bool ready = false;
};

Prerouter::Prerouter() : impl_(std::make_unique<Impl>()) {}
Prerouter::~Prerouter() = default;
Prerouter::Prerouter(Prerouter &&) noexcept = default;
Prerouter &Prerouter::operator=(Prerouter &&) noexcept = default;

int Prerouter::n_heads() const { return impl_ ? static_cast<int>(impl_->owners.size()) : 0; }
int Prerouter::owner(int i) const { return impl_->owners[i]; }

const std::vector<int> &Prerouter::owners() const { return impl_->owners; }
const std::vector<float> &Prerouter::stacked_fc1() const { return impl_->fc1; }
const std::vector<float> &Prerouter::stacked_fc2() const { return impl_->fc2; }
const std::vector<float> &Prerouter::stacked_lin() const { return impl_->lin; }
int64_t Prerouter::stacked_f_in() const { return impl_->f_in; }
int64_t Prerouter::stacked_prh() const { return impl_->reg.prerouter_hidden; }
int64_t Prerouter::stacked_E() const { return impl_->reg.num_experts; }

bool Prerouter::build(const std::vector<HeadW> &heads, const TierRouting &reg, int64_t hidden,
                      Prerouter &out, std::string &err) {
  if (heads.empty()) { err = "prerouter: zero heads"; return false; }
  const int64_t prh = reg.prerouter_hidden, E = reg.num_experts;
  int64_t f_in = -1;
  for (auto &h : heads) {
    if (h.fc1.empty() || h.fc2.empty() || h.lin.empty() ||
        static_cast<int64_t>(h.fc2.size()) != E * prh ||
        static_cast<int64_t>(h.lin.size()) != static_cast<int64_t>(h.fc1.size()) / prh * E) {
      err = "prerouter: header shape mismatch owner=" + std::to_string(h.owner);
      return false;
    }
    int64_t fi = static_cast<int64_t>(h.fc1.size()) / prh;
    if (fi != hidden + 2 * E) {
      err = "prerouter: f_in != hidden+2E (" + std::to_string(fi) + " vs " +
            std::to_string(hidden + 2 * E) + ") owner=" + std::to_string(h.owner);
      return false;
    }
    if (f_in < 0) f_in = fi;
    else if (f_in != fi) { err = "prerouter: f_in disagrees across heads"; return false; }
  }
  for (size_t i = 1; i < heads.size(); i++)
    if (heads[i].owner <= heads[i - 1].owner) { err = "prerouter: owners are not sorted"; return false; }

  auto im = std::make_unique<Prerouter::Impl>();
  im->reg = reg;
  im->hidden = hidden;
  im->f_in = f_in;
  const int O = static_cast<int>(heads.size());
  for (auto &h : heads) im->owners.push_back(h.owner);

  std::vector<float> fc1t(static_cast<size_t>(O) * f_in * prh),
      fc2s(static_cast<size_t>(O) * prh * E), lin_s(static_cast<size_t>(O) * f_in * E);
  for (int i = 0; i < O; i++) {
    const HeadW &h = heads[i];
    for (int64_t r = 0; r < prh; r++)
      for (int64_t c = 0; c < f_in; c++)
        fc1t[(static_cast<size_t>(i) * f_in + c) * prh + r] = h.fc1[r * f_in + c];
    for (int64_t e = 0; e < E; e++) {
      for (int64_t r = 0; r < prh; r++)
        fc2s[(static_cast<size_t>(i) * prh + r) * E + e] = h.fc2[e * prh + r];
      for (int64_t c = 0; c < f_in; c++)
        lin_s[(static_cast<size_t>(i) * f_in + c) * E + e] = h.lin[e * f_in + c];
    }
  }
  im->fc1 = expand_f16(fc1t);
  im->fc2 = expand_f16(fc2s);
  im->lin = expand_f16(lin_s);
  im->ready = true;
  out.impl_ = std::move(im);
  return true;
}

bool Prerouter::load(const ModelFiles &files, const TierRouting &reg, int64_t hidden,
                     Prerouter &out, std::string &err) {
  std::vector<HeadW> heads;
  for (int o = reg.first_owner; o <= reg.last_owner; o++) {
    HeadW h;
    h.owner = o;
    const std::string base = reg.head_file;
    DenseF32 fc1, fc2, lin;
    std::string k1 = "layers." + std::to_string(o) + ".fc1.weight";
    std::string k2 = "layers." + std::to_string(o) + ".fc2.weight";
    std::string k3 = "layers." + std::to_string(o) + ".linear_init.weight";
    if (!read_dense_f32(files, k1, fc1, err) || !read_dense_f32(files, k2, fc2, err) ||
        !read_dense_f32(files, k3, lin, err)) {
      if (err.find("missing tensor") == 0) err += " (header file " + base + " or missing from the directory)";
      return false;
    }
    if (fc1.rows != reg.prerouter_hidden || fc2.cols != reg.prerouter_hidden ||
        fc2.rows != reg.num_experts || lin.rows != reg.num_experts || fc1.cols != lin.cols) {
      err = "prerouter: header shape does not match registry owner=" + std::to_string(o);
      return false;
    }
    h.fc1 = std::move(fc1.v);
    h.fc2 = std::move(fc2.v);
    h.lin = std::move(lin.v);
    heads.push_back(std::move(h));
  }
  return build(heads, reg, hidden, out, err);
}

bool Prerouter::predict(const std::vector<Input> &in, std::vector<Selection> &out_sel,
                        std::vector<std::vector<double>> *out_logits, std::string &err) {
  if (!impl_ || !impl_->ready) { err = "prerouter: not built"; return false; }
  const int O = n_heads();
  if (static_cast<int>(in.size()) != O) { err = "prerouter: head count does not match input count"; return false; }
  const auto &reg = impl_->reg;
  const int64_t prh = reg.prerouter_hidden, E = reg.num_experts, f_in = impl_->f_in;

  std::vector<float> feats(static_cast<size_t>(O) * f_in, 0.f);
  for (int i = 0; i < O; i++) {
    if (!in[i].hidden) { err = "prerouter: missing header hidden input owner=" + std::to_string(owner(i)); return false; }
    float *row = feats.data() + static_cast<size_t>(i) * f_in;
    std::memcpy(row, in[i].hidden, sizeof(float) * impl_->hidden);
    for (int64_t e : in[i].cur_exec)
      if (e >= 0 && e < E) row[impl_->hidden + e] = 1.f;
    for (int64_t e : in[i].prev_exec)
      if (e >= 0 && e < E) row[impl_->hidden + E + e] = 1.f;
  }

  try {
    auto wrap = [](void *p, mlx::core::Shape shape, mlx::core::Dtype dt) {
      return array(p, std::move(shape), dt, [](void *) {});
    };
    auto a_feats = wrap(feats.data(), {O, 1, static_cast<int32_t>(f_in)}, mlx::core::float32);
    auto a_fc1 = wrap(impl_->fc1.data(),
                      {O, static_cast<int32_t>(f_in), static_cast<int32_t>(prh)},
                      mlx::core::float32);
    auto a_fc2 = wrap(impl_->fc2.data(),
                      {O, static_cast<int32_t>(prh), static_cast<int32_t>(E)},
                      mlx::core::float32);
    auto a_lin = wrap(impl_->lin.data(),
                      {O, static_cast<int32_t>(f_in), static_cast<int32_t>(E)},
                      mlx::core::float32);
    const auto f16 = mlx::core::float16;
    auto qf = mlx::core::astype(a_feats, f16);
    auto h = mlx::core::matmul(qf, mlx::core::astype(a_fc1, f16));  // [O,1,prh] f16
    auto gel = h * 0.5f * (1.0f + mlx::core::erf(h / static_cast<float>(std::sqrt(2.0))));
    auto logits = mlx::core::astype(
        mlx::core::matmul(gel, mlx::core::astype(a_fc2, f16)) +
            mlx::core::matmul(qf, mlx::core::astype(a_lin, f16)),
        mlx::core::float32);
    auto host = mlx::core::copy(logits, mlx::core::default_stream(Device::cpu));
    mlx::core::eval(host);
    const float *lp = host.data<float>();
    out_sel.assign(O, Selection{});
    if (out_logits) out_logits->assign(O, {});
    for (int i = 0; i < O; i++) {
      const double *rowp = nullptr;
      std::vector<double> drow(E);
      for (int64_t e = 0; e < E; e++) drow[e] = lp[static_cast<size_t>(i) * E + e];
      rowp = drow.data();
      if (out_logits) (*out_logits)[i] = drow;
      out_sel[i] = reg.family == RouterFamily::kSoftmaxTopk
                       ? select_softmax_topk(rowp, static_cast<int>(E), reg.top_k,
                                             reg.norm_topk_prob)
                       : select_sigmoid_group(rowp, nullptr, static_cast<int>(E), reg.top_k,
                                              reg.n_group, reg.topk_group, reg.routed_scaling,
                                              reg.norm_topk_prob);
    }
  } catch (const std::exception &ex) {
    err = std::string("prerouter mlx: ") + ex.what();
    return false;
  }
  return true;
}

// ---------------- PrerouterState ----------------

Step live_next_step(const PrerouterState &st) {
  Step plan;
  for (int l = st.reg.first_head_consumer; l <= st.reg.last_head_consumer; l++)
    if (!st.pred[l].inds.empty())
      for (auto e : st.pred[l].inds) plan.experts.emplace_back(l, e);
  return plan;
}

void PrerouterState::init(const TierRouting &r) {
  reg = r;
  cap_hidden.assign(r.num_layers, {});
  cap_exec.assign(r.num_layers, {});
  prev_exec.assign(r.num_layers, {});
  pred.assign(r.num_layers, Selection{});
  pred_avail.assign(r.num_layers, false);
  step = 0;
}

void PrerouterState::capture(int layer, std::vector<float> m_in, std::vector<int64_t> exec) {
  cap_hidden[layer] = std::move(m_in);
  std::sort(exec.begin(), exec.end());
  cap_exec[layer] = std::move(exec);
}

bool PrerouterState::run_step_boundary(Prerouter &pr,
                                       std::vector<std::vector<double>> *diag_logits,
                                       std::string &err) {
  const int O = pr.n_heads();
  for (int l = reg.first_head_consumer; l <= reg.last_head_consumer; l++) pred_avail[l] = false;
  std::vector<Prerouter::Input> in(O);
  for (int i = 0; i < O; i++) {
    int own = pr.owner(i);
    if (cap_hidden[own].empty()) continue;
    in[i].hidden = cap_hidden[own].data();
    in[i].cur_exec = cap_exec[own];
    in[i].prev_exec = prev_exec[own];
  }
  std::vector<Prerouter::Input> compact;
  std::vector<int> idx_of;
  for (int i = 0; i < O; i++)
    if (in[i].hidden) { compact.push_back(in[i]); idx_of.push_back(i); }
  if (compact.empty()) { swap(); return true; }

  std::vector<Selection> sel;
  std::vector<std::vector<double>> lg;
  if (!pr.predict(compact, sel, diag_logits ? &lg : nullptr, err)) return false;
  if (diag_logits) *diag_logits = std::move(lg);
  for (size_t j = 0; j < idx_of.size(); j++) {
    int own = pr.owner(idx_of[j]);
    int consumer = own + 1;
    if (consumer >= reg.first_head_consumer && consumer <= reg.last_head_consumer) {
      pred[consumer] = sel[j];
      pred_avail[consumer] = true;
    }
  }
  if (diag_logits) {
  }
  swap();
  return true;
}

void PrerouterState::swap() {
  prev_exec = cap_exec;
  for (auto &v : cap_hidden) v.clear();
  for (auto &v : cap_exec) v.clear();
  step++;
}

bool PrerouterState::take_pred(int layer, Selection &out) {
  if (layer < 0 || layer >= reg.num_layers || !pred_avail[layer] || pred[layer].inds.empty())
    return false;
  out = pred[layer];
  pred_avail[layer] = false;
  return true;
}

void PrerouterState::reset() {
  for (auto &v : cap_hidden) v.clear();
  for (auto &v : cap_exec) v.clear();
  for (auto &v : prev_exec) v.clear();
  for (auto &p : pred) p = Selection{};
  std::fill(pred_avail.begin(), pred_avail.end(), false);
  step = 0;
}

}  // namespace e0n
