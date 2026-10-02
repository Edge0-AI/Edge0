// Host fp64 expert selection for the two router families (bit-identical to routing.py).
#pragma once
#include <cstdint>
#include <vector>

namespace e0n {

struct Selection {
  std::vector<int64_t> inds;
  std::vector<double> scores;
};

// logits length = E. k ≤ E. norm = norm_topk_prob.
Selection select_softmax_topk(const double *logits, int E, int k, bool norm);

// bias may be nullptr (exclude bias). E % n_group == 0 (8b: 128/8=16).
// topk_group == n_group means no group is dropped.
Selection select_sigmoid_group(const double *logits, const double *bias, int E, int k,
                               int n_group, int topk_group, double scaling, bool norm);

// Dense gate: x[T,H] × W[E,H]ᵀ → out[T,E], fp64 accumulate. Live graphs run gate on mlx.
void gate_logits(const float *x, int T, int H, const float *W, int E,
                 std::vector<double> &out);

}  // namespace e0n
