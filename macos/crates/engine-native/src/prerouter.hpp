// Prerouter heads and step-boundary batched prediction (stacked matmul, no per-head sync).
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "provider.hpp"
#include "registry.hpp"
#include "router.hpp"
#include "safio.hpp"

namespace e0n {

struct HeadW {
  int owner = -1;
  std::vector<float> fc1;  // [prh, f_in]
  std::vector<float> fc2;  // [E, prh]
  std::vector<float> lin;  // [E, f_in]
};

class Prerouter {
 public:
  // Load from the tier dir (heads from the registry; f_in = hidden + 2×E per head).
  // hidden is the tier hidden_size (35b 2048 / 8b 1536; same as gate weight columns).
  static bool load(const ModelFiles& files, const TierRouting& reg, int64_t hidden,
                   Prerouter& out, std::string& err);
  // Direct weights (CI / onboarding). Head order is the input order (must be ascending).
  static bool build(const std::vector<HeadW>& heads, const TierRouting& reg, int64_t hidden,
                    Prerouter& out, std::string& err);

  Prerouter();
  ~Prerouter();
  Prerouter(Prerouter&&) noexcept;
  Prerouter& operator=(Prerouter&&) noexcept;
  Prerouter(const Prerouter&) = delete;

  int n_heads() const;
  int owner(int i) const;  // source layer of head i (ascending)

  // Same stacked weights as host predict, layout = stager._stacked_weights:
  // fc1 [n,f_in,prh], fc2 [n,prh,E], lin [n,f_in,E].
  const std::vector<int> &owners() const;
  const std::vector<float> &stacked_fc1() const;
  const std::vector<float> &stacked_fc2() const;
  const std::vector<float> &stacked_lin() const;
  int64_t stacked_f_in() const;
  int64_t stacked_prh() const;
  int64_t stacked_E() const;

  struct Input {
    const float* hidden = nullptr;           // length = hidden_size (m_in)
    std::vector<int64_t> cur_exec;           // this token's exec set (ascending)
    std::vector<int64_t> prev_exec;          // previous token (empty = zero feature)
  };
  bool predict(const std::vector<Input>& in, std::vector<Selection>& out_sel,
               std::vector<std::vector<double>>* out_logits, std::string& err);

  struct Impl;

 private:
  std::unique_ptr<Impl> impl_;
};

// Cross-token double-buffer. Fields are public for the forward glue; only
// reset()/swap() mutate the batch structure.
struct PrerouterState {
  TierRouting reg;
  std::vector<std::vector<float>> cap_hidden;  // per layer: m_in (decode = last position)
  std::vector<std::vector<int64_t>> cap_exec;  // per layer: this step's exec set
  std::vector<std::vector<int64_t>> prev_exec;
  std::vector<Selection> pred;
  std::vector<bool> pred_avail;  // one-shot consume (cleared after take)
  long long step = 0;

  void init(const TierRouting& r);
  void capture(int layer, std::vector<float> m_in, std::vector<int64_t> exec);
  // Batch predict in ascending source-layer order → pred[layer+1] only if that
  // layer is in predicted(); 35b layer 38 → consumer 39 is computed then discarded.
  bool run_step_boundary(Prerouter& pr, std::vector<std::vector<double>>* diag_logits,
                         std::string& err);
  void swap();
  bool take_pred(int layer, Selection& out);
  void reset();
};

// Prefetch plan: predicted consumer layers that already have a prediction,
// (layer, expert) with layer then expert ascending. Live-router layers are omitted
// (their set is computed at the gate).
Step live_next_step(const PrerouterState &st);

}  // namespace e0n
