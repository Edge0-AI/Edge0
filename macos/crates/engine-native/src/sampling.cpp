#include "sampling.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <set>

#include <nlohmann/json.hpp>

#include <mlx/mlx.h>

#include "fwd_common.hpp"

namespace mx = mlx::core;

namespace e0n {
namespace sampling {

namespace {
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// Single-row f32 mlx pipeline (upstream _mask_logits, penalty already applied).
// Vendored mlx topk returns the k largest values; min(topk) is the ascending-topk first element.
std::vector<float> mask_ops(const std::vector<float> &in, double temperature, int top_k,
                            double top_p) {
  if (in.empty()) return in;
  const int64_t V = (int64_t)in.size();
  mx::array l = wrap_f32(in, mkshape({1, V}));
  if (temperature > 0.0) {
    l = mx::divide(l, mx::array((float)temperature, mx::float32));
  }
  if (top_k > 0) {
    int k = top_k < V ? top_k : (int)V;  // min(top_k, V)
    auto top = mx::topk(l, k, -1);
    auto threshold = mx::min(top, -1, /*keepdims=*/true);  // k-th largest = first of ascending top-k
    l = mx::where(mx::less(l, threshold), mx::array(kNegInf, mx::float32), l);
  }
  if (top_p < 1.0) {
    // Descending cumsum without flip/negative stride (those produce a 0-size slice here):
    // descending_cumsum = total − ascending exclusive-cumsum. Threshold is the
    // (k-1)-th largest, located as ascending index (V−k).
    auto asc = mx::sort(l, -1);
    auto p_asc = mx::softmax(asc, -1);
    auto cum_exc = mx::subtract(mx::cumsum(p_asc, -1), p_asc);
    auto total = mx::sum(p_asc, -1, /*keepdims=*/true);
    auto cum_desc = mx::subtract(total, cum_exc);
    auto cutmask = mx::less_equal(cum_desc, mx::array((float)top_p, mx::float32));
    auto counts = mx::sum(mx::astype(cutmask, mx::int32), -1, /*keepdims=*/true);
    auto k = mx::maximum(counts, mx::array(1, mx::int32));
    auto pos = mx::subtract(mx::array((int32_t)V, mx::int32), k);
    auto threshold = mx::take_along_axis(asc, pos, -1);
    l = mx::where(mx::less(l, threshold), mx::array(kNegInf, mx::float32), l);
  }
  if (temperature <= 0.0) {
    auto m = mx::max(l, -1, /*keepdims=*/true);
    l = mx::where(mx::less(l, m), mx::array(kNegInf, mx::float32), l);
  }
  std::vector<float> out;
  std::string err;
  if (!to_host32(l, out, err)) return in;  // pipeline failure → leave unmasked
  return out;
}
}  // namespace

std::vector<float> mask_logits(const std::vector<float> &logits, double temperature,
                               int top_k, double top_p) {
  return mask_ops(logits, temperature, top_k, top_p);
}

std::vector<float> apply_repetition_penalty(const std::vector<float> &logits, double penalty,
                                            const std::vector<int32_t> &history) {
  std::vector<float> out = logits;
  if (penalty == 1.0 || history.empty()) return out;
  std::set<int32_t> uniq(history.begin(), history.end());
  for (int32_t id : uniq) {
    if (id < 0 || (size_t)id >= out.size()) continue;
    float v = out[id];
    out[id] = (v > 0.f) ? (float)((double)v / penalty) : v * (float)penalty;
  }
  return out;
}

void rng_seed(Rng &r, uint64_t seed) {
  uint64_t z = seed + 0x9E3779B97F4A7C15ull;
  auto sm = [](uint64_t &x) {
    x += 0x9E3779B97F4A7C15ull;
    uint64_t z2 = x;
    z2 = (z2 ^ (z2 >> 30)) * 0xBF58476D1CE4E5B9ull;
    z2 = (z2 ^ (z2 >> 27)) * 0x94D049BB133111EBull;
    return z2 ^ (z2 >> 31);
  };
  r.s[0] = sm(z);
  r.s[1] = sm(z) ^ 0xD1B54A32D192ED03ull;
}

uint64_t rng_next(Rng &r) {
  uint64_t z = (r.s[0] += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  uint64_t o = z ^ (z >> 31);
  r.s[1] += 0x9E3779B97F4A7C15ull;
  return o ^ r.s[1];
}

int32_t gumbel_draw(const std::vector<float> &masked, uint64_t seed, Rng &rng) {
  rng_seed(rng, seed);
  size_t b = 0;
  double best = -std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < masked.size(); i++) {
    if (std::isinf(masked[i])) continue;
    uint64_t u64 = rng_next(rng);
    double u = ((double)(u64 >> 11) + 1.0) / 9007199254740993.0;
    double g = masked[i] - std::log(-std::log(u));
    if (g > best) { best = g; b = i; }
  }
  return (int32_t)b;
}

int32_t sample_step(const std::vector<float> &logits, const Params &p, bool first,
                    const std::vector<int32_t> &history, Rng &rng) {
  size_t V = logits.size();
  if (V == 0) return -1;
  const bool ftg = first && p.first_token_greedy && p.temperature > 0.0;
  const bool greedy = p.temperature <= 0.0 || ftg;
  if (greedy) {
    std::vector<float> l = apply_repetition_penalty(logits, p.repetition_penalty, history);
    size_t b = 0;
    for (size_t i = 1; i < V; i++)
      if (l[i] > l[b]) b = i;  // ties take the smaller id (strict >)
    return (int32_t)b;
  }
  std::vector<float> l = apply_repetition_penalty(logits, p.repetition_penalty, history);
  l = mask_ops(l, p.temperature, p.top_k, p.top_p);
  if (p.has_seed) return gumbel_draw(l, p.seed, rng);
  return gumbel_draw(l, rng_next(rng), rng);  // no seed = stream-local (not stable across runs)
}

TierDefaults tier_defaults(const std::string &tier) {
  TierDefaults td;
  td.repetition_penalty = (tier == "edge0-8b") ? 1.1 : 1.0;
  return td;
}

Params parse_params(const std::string &params_json, const TierDefaults &td,
                    const std::vector<int32_t> &eos_from_config) {
  Params p;
  p.temperature = td.temperature;
  p.top_p = td.top_p;
  p.top_k = td.top_k;
  p.repetition_penalty = td.repetition_penalty;
  p.max_new_tokens = td.max_new_tokens;
  p.first_token_greedy = td.first_token_greedy;
  p.eos_ids = eos_from_config;

  nlohmann::json v = nlohmann::json::parse(params_json.empty() ? "{}" : params_json,
                                           nullptr, /*allow_exceptions=*/false);
  if (v.is_discarded() || !v.is_object()) return p;
  auto get = [&v](const char *k) -> const nlohmann::json * {
    auto it = v.find(k);
    return it == v.end() ? nullptr : &*it;
  };
  if (auto x = get("max_new_tokens"); x && x->is_number()) {
    int64_t n = x->get<int64_t>();
    if (n < 0) n = 0;
    if (n > 4096) n = 4096;
    p.max_new_tokens = (size_t)n;
  }
  if (auto x = get("temperature"); x && x->is_number()) p.temperature = x->get<double>();
  if (auto x = get("top_p"); x && x->is_number()) p.top_p = x->get<double>();
  if (auto x = get("top_k"); x && x->is_number()) p.top_k = (int)x->get<int64_t>();
  if (auto x = get("repetition_penalty"); x && x->is_number())
    p.repetition_penalty = x->get<double>();
  if (auto x = get("first_token_greedy"); x && x->is_boolean())
    p.first_token_greedy = x->get<bool>();
  if (auto x = get("seed"); x && (x->is_number_integer() || x->is_null())) {
    if (x->is_number_integer()) {
      p.has_seed = true;
      int64_t s = x->get<int64_t>();
      p.seed = s < 0 ? 0 : (uint64_t)s;
    } else {
      p.has_seed = false;
    }
  }
  if (auto x = get("teacher_tokens"); x && x->is_array()) {
    for (const auto &it : *x)
      if (it.is_number_integer()) p.teacher_tokens.push_back(it.get<int32_t>());
  }
  if (auto x = get("stop"); x && x->is_array()) {
    for (const auto &it : *x) {
      if (it.is_number_integer())
        p.stop_token_ids.push_back(it.get<int32_t>());
    }
  }
  return p;
}

}  // namespace sampling
}  // namespace e0n
