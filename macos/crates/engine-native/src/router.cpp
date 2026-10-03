#include "router.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace e0n {
namespace {

// Deterministic top-K: score descending, ties by id ascending (argpartition order is not the contract).
std::vector<int64_t> topk_ids(const std::vector<double> &v, int k) {
  std::vector<int64_t> ids(v.size());
  for (size_t i = 0; i < ids.size(); i++) ids[i] = static_cast<int64_t>(i);
  std::sort(ids.begin(), ids.end(), [&](int64_t a, int64_t b) {
    if (v[a] != v[b]) return v[a] > v[b];
    return a < b;
  });
  ids.resize(static_cast<size_t>(k));
  std::sort(ids.begin(), ids.end());  // canonical: ascending
  return ids;
}

}  // namespace

Selection select_softmax_topk(const double *logits, int E, int k, bool norm) {
  // precise softmax (fp64) → top-K → renormalize.
  double mx = logits[0];
  for (int i = 1; i < E; i++) mx = std::max(mx, logits[i]);
  std::vector<double> gates(static_cast<size_t>(E));
  double sum = 0.0;
  for (int i = 0; i < E; i++) {
    gates[i] = std::exp(logits[i] - mx);
    sum += gates[i];
  }
  for (auto &g : gates) g /= sum;
  Selection out;
  out.inds = topk_ids(gates, k);
  out.scores.resize(out.inds.size());
  double ssum = 0.0;
  for (size_t j = 0; j < out.inds.size(); j++) {
    out.scores[j] = gates[out.inds[j]];
    ssum += out.scores[j];
  }
  if (norm && ssum > 0.0)
    for (auto &s : out.scores) s /= ssum;
  return out;
}

Selection select_sigmoid_group(const double *logits, const double *bias, int E, int k,
                               int n_group, int topk_group, double scaling, bool norm) {
  std::vector<double> scores(static_cast<size_t>(E));
  for (int i = 0; i < E; i++) scores[i] = 1.0 / (1.0 + std::exp(-logits[i]));
  std::vector<double> select = scores;
  if (bias)
    for (int i = 0; i < E; i++) select[i] += bias[i];

  // Group mask: group score = sum of in-group top-2 select scores; drop the weakest k_drop groups (-inf).
  const int k_drop = n_group - topk_group;
  if (k_drop > 0) {
    const int per = E / n_group;
    std::vector<double> gscore(static_cast<size_t>(n_group));
    for (int g = 0; g < n_group; g++) {
      double m1 = -INFINITY, m2 = -INFINITY;  // top-2 (ties: two max passes)
      for (int j = 0; j < per; j++) {
        double s = select[static_cast<size_t>(g * per + j)];
        if (s >= m1) { m2 = m1; m1 = s; }
        else if (s > m2) m2 = s;
      }
      gscore[g] = m1 + m2;
    }
    // Drop the k_drop groups with the smallest scores; ties by group id ascending.
    std::vector<int> gids(static_cast<size_t>(n_group));
    for (int g = 0; g < n_group; g++) gids[g] = g;
    std::sort(gids.begin(), gids.end(), [&](int a, int b) {
      if (gscore[a] != gscore[b]) return gscore[a] < gscore[b];
      return a < b;
    });
    for (int d = 0; d < k_drop; d++) {
      int g = gids[d];
      for (int j = 0; j < per; j++) select[static_cast<size_t>(g * per + j)] = -INFINITY;
    }
  }

  Selection out;
  out.inds = topk_ids(select, k);
  out.scores.resize(out.inds.size());
  double ssum = 0.0;
  for (size_t j = 0; j < out.inds.size(); j++) {
    out.scores[j] = scores[out.inds[j]];  // weight = raw sigmoid (bias not in weights)
    ssum += out.scores[j];
  }
  if (norm)
    for (auto &s : out.scores) s /= (ssum + 1e-20);
  for (auto &s : out.scores) s *= scaling;
  return out;
}

void gate_logits(const float *x, int T, int H, const float *W, int E,
                 std::vector<double> &out) {
  out.assign(static_cast<size_t>(T) * E, 0.0);
  for (int t = 0; t < T; t++) {
    const float *xr = x + static_cast<size_t>(t) * H;
    for (int e = 0; e < E; e++) {
      const float *wr = W + static_cast<size_t>(e) * H;
      double acc = 0.0;
      for (int h = 0; h < H; h++) acc += static_cast<double>(xr[h]) * wr[h];
      out[static_cast<size_t>(t) * E + e] = acc;
    }
  }
}

}  // namespace e0n
