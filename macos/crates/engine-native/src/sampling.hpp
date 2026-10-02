// Sampling pipeline: mlx f32 mask ops plus host Gumbel-max categorical.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace e0n {
namespace sampling {

struct TierDefaults {
  double temperature = 0.7;
  double top_p = 0.95;
  int top_k = 64;
  double repetition_penalty = 1.0;  // 8b=1.1, 35b=1.0
  size_t max_new_tokens = 2048;
  bool first_token_greedy = true;
};

struct Params {
  double temperature = 0.7;
  double top_p = 0.95;
  int top_k = 64;
  double repetition_penalty = 1.0;
  size_t max_new_tokens = 64;
  std::vector<int32_t> eos_ids;           // from config.json at create (35b is an array)
  std::vector<int32_t> stop_token_ids;    // params "stop" as ids
  std::vector<std::vector<int32_t>> stop_seqs;  // params "stop" strings after tokenize
  bool has_seed = false;
  uint64_t seed = 0;
  std::vector<int32_t> teacher_tokens;  // oracle replay; step in and out follow this
  bool first_token_greedy = true;
};

TierDefaults tier_defaults(const std::string &tier);  // edge0-8b / edge0-35b; else generic

// Missing keys fall back to tier defaults; unknown keys are ignored.
Params parse_params(const std::string &params_json, const TierDefaults &td,
                    const std::vector<int32_t> &eos_from_config);

// logits = this step's host f32 [V]; history = prompt + emitted (same as upstream).
// rng is owned by the stream (seed is replayed before each draw).
struct Rng {
  uint64_t s[2];  // splitmix64; seed-determined, advanced per step
};
void rng_seed(Rng &r, uint64_t seed);
uint64_t rng_next(Rng &r);

// One sample. Greedy shortcut: temp≤0 (except first-token-greedy off) skips the
// mask chain and host-argnmaxes. first_token_greedy uses temperature=0 masks then argmax.
int32_t sample_step(const std::vector<float> &logits, const Params &p, bool first,
                    const std::vector<int32_t> &history, Rng &rng);

std::vector<float> mask_logits(const std::vector<float> &logits, double temperature,
                               int top_k, double top_p);
int32_t gumbel_draw(const std::vector<float> &masked, uint64_t seed, Rng &rng);
std::vector<float> apply_repetition_penalty(const std::vector<float> &logits,
                                            double penalty,
                                            const std::vector<int32_t> &history);

}  // namespace sampling
}  // namespace e0n
