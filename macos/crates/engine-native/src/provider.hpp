// Per-step (layer, expert) load plan. Implementations may swap replay for live routing.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace e0n {

struct Step {
  // Load order: (layer, expert), experts ascending within a layer.
  std::vector<std::pair<int, int64_t>> experts;
};

class StepPlanProvider {
 public:
  virtual ~StepPlanProvider() = default;
  // Next step; nullptr = end. The Step stays valid until the next next()/reset().
  virtual const Step *next() = 0;
  virtual void reset() = 0;
  virtual const char *source() const = 0;  // "routing-replay" | "prerouter" | ...
};

// Rebuild decode order from fixtures/<tier>/routing/records.jsonl (`kind=router`)
// grouped by (prompt_id, step). 35b samples cover layers 0–6; remaining layers
// reuse the matching-size expert set via L%7. Counted in steps_total().
bool make_routing_replay(const std::string &fixtures_tier_dir, const std::string &prompt_id,
                         int num_layers, int64_t num_experts,
                         std::unique_ptr<StepPlanProvider> &out, std::string &err);

namespace replay_meta {
uint64_t stale_regions(const StepPlanProvider *p);  // intervals that reuse a prefill set
uint64_t total_regions(const StepPlanProvider *p);
const std::vector<int> *step_ids(const StepPlanProvider *p);  // original step numbers
}  // namespace replay_meta

}  // namespace e0n
