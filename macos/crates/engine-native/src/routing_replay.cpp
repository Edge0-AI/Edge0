// Replay provider: `kind=router` records used as-is. 8b: layers 1–23 (L0 is dense);
// 35b: layers 0–39; top_k comes from the record.
#include "provider.hpp"

#include <algorithm>
#include <fstream>
#include <map>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace e0n {
namespace {

class RoutingReplay final : public StepPlanProvider {
 public:
  std::vector<Step> steps;               // decode step ascending (step 0..N)
  std::vector<int> step_ids;             // original step numbers
  size_t i = 0;
  Step cur;
  uint64_t stale_regions = 0;
  uint64_t total_regions = 0;

  const Step *next() override {
    if (i >= steps.size()) return nullptr;
    cur = steps[i++];
    return &cur;
  }
  void reset() override { i = 0; }
  const char *source() const override { return "routing-replay"; }
};

}  // namespace

bool make_routing_replay(const std::string &fixtures_tier_dir, const std::string &prompt_id,
                         int num_layers, int64_t num_experts,
                         std::unique_ptr<StepPlanProvider> &out, std::string &err) {
  std::ifstream f(fixtures_tier_dir + "/routing/records.jsonl");
  if (!f) { err = "no routing records: " + fixtures_tier_dir; return false; }

  // (step, layer) → experts; step=-1 is prefill and is not in the decode sequence.
  std::map<std::pair<int, int>, std::vector<int64_t>> sets;
  std::map<std::pair<int, int>, bool> is_stale;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    json r = json::parse(line, nullptr, false);
    if (r.is_discarded() || r.value("kind", "") != "router") continue;
    if (r.value("prompt_id", "") != prompt_id) continue;
    int step = r.value("step", -2);
    if (step < 0) continue;  // decode path only
    int layer = r.value("layer", -1);
    if (layer < 0 || layer >= num_layers) continue;  // 8b L0 has no records
    std::vector<int64_t> es;
    for (auto &e : r.at("experts")) es.push_back(e.get<int64_t>());
    std::sort(es.begin(), es.end());
    bool dup = es.size() > 1 && std::adjacent_find(es.begin(), es.end()) != es.end();
    if (dup) es.erase(std::unique(es.begin(), es.end()), es.end());
    auto key = std::make_pair(step, layer);
    if (sets.count(key)) continue;  // first record for a key wins
    sets[key] = std::move(es);
    is_stale[key] = r.value("routing_source", "").find("stale") != std::string::npos;
  }
  if (sets.empty()) {
    err = "prompt " + prompt_id + " has no decode routing records for this tier";
    return false;
  }

  auto rep = std::make_unique<RoutingReplay>();
  int cur_step = -1;
  for (auto &[key, es] : sets) {
    auto [step, layer] = key;
    for (int64_t e : es) {
      if (e < 0 || e >= num_experts) {
        err = "recorded expert id out of range: " + prompt_id + " L" + std::to_string(layer) +
              " E" + std::to_string(e);
        return false;
      }
    }
    if (step != cur_step) {
      rep->steps.emplace_back();
      rep->step_ids.push_back(step);
      cur_step = step;
    }
    auto &st = rep->steps.back();
    for (int64_t e : es) st.experts.emplace_back(layer, e);
    rep->total_regions += es.size();
    if (is_stale[key]) rep->stale_regions += es.size();
  }
  // map is ordered by (step, layer); experts within a layer are already sorted.
  out = std::move(rep);
  return true;
}

namespace replay_meta {
uint64_t stale_regions(const StepPlanProvider *p) {
  auto *r = dynamic_cast<const RoutingReplay *>(p);
  return r ? r->stale_regions : 0;
}
uint64_t total_regions(const StepPlanProvider *p) {
  auto *r = dynamic_cast<const RoutingReplay *>(p);
  return r ? r->total_regions : 0;
}
const std::vector<int> *step_ids(const StepPlanProvider *p) {
  auto *r = dynamic_cast<const RoutingReplay *>(p);
  return r ? &r->step_ids : nullptr;
}
}  // namespace replay_meta

}  // namespace e0n
