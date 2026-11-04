#pragma once
#include "model/adapter.h"
#include "session/storage.h"
#include "weights/storage.h"
#include "edge0/weight_activity.h"
#include "edge0/timing.h"
#include <chrono>
#include <memory>
#include <string>
#include <vector>

namespace edge0::model { class LoadedModel; }
namespace edge0::models {
struct StateDescription {
    std::vector<session::KVDeclaration> kv;
    std::vector<std::string> updated_entries;
    bool evolves_as_whole = false;
    bool supports_save_restore = false;
    std::size_t checkpoint_bytes = 0;
    std::size_t block_snapshot_bytes = 0;
};
enum class FinishReason { normal, failure };
enum class PlanMode { automatic, fresh };
struct GreedyOutput { std::int32_t token; float sum; };
struct GreedyTensors { ggml_tensor* token; ggml_tensor* sum; };
Result<GreedyTensors> append_greedy(exec::Graph& graph, ggml_tensor* logits);
struct ForwardProgress {
    bool submitted = false;
    WeightActivity activity;
    bool cancelled = false;
    std::size_t rolled_back_tokens = 0;
    double block_snapshot_seconds = 0;
};
struct ForwardRequest {
    std::vector<std::int32_t> tokens;
    session::Step step;
    exec::Phase phase;
    std::vector<std::size_t> output_positions;
    std::vector<std::string> observe;
    PlanMode plan_mode = PlanMode::automatic;
    bool greedy = false;
    bool retain_logits = true;
    ForwardProgress* progress = nullptr;
    bool breakdown = false;
    // The model currently writes the trace format. With a second model, return
    // structured records from forward and let the framework write the file.
    std::string route_trace = {};
    std::chrono::steady_clock::time_point (*clock_now)() = std::chrono::steady_clock::now;
    std::function<bool(std::size_t)> cancel_block = {};
    bool skip_block_rollback = false;
    std::shared_ptr<exec::Backend> backend = {};
};
struct ObservedTensor {
    std::string name;
    exec::Layout layout;
    std::vector<std::uint8_t> data;
};
class ForwardResult {
public:
    virtual ~ForwardResult() = default;
    virtual const exec::Completion& completion() const = 0;
    virtual Result<std::vector<float>> logits() = 0;
    virtual Result<std::vector<ObservedTensor>> observations() = 0;
    virtual exec::Status end_observation() = 0;
    virtual std::size_t weight_registrations() const = 0;
    virtual const std::vector<std::string>& updated_entries() const = 0;
    virtual Result<GreedyOutput> greedy_output() = 0;
    virtual std::size_t readback_bytes() const = 0;
    virtual bool reused_plan() const = 0;
    virtual WeightActivity weight_activity() const { return {}; }
    virtual std::optional<DecodeTiming> decode_timing() const { return {}; }
};
class ModelCompute {
public:
    virtual ~ModelCompute() = default;
    virtual bool supports_handoff() const { return false; }
    virtual exec::TransferStats transfer_stats(const model::LoadedModel&, session::StateStore&) const { return {}; }
    virtual exec::StageFacts stage_facts(const model::LoadedModel&, session::StateStore&,
        const std::shared_ptr<exec::Backend>&) const { return {}; }
    virtual exec::Status hand_off(const std::shared_ptr<model::LoadedModel>&,
        session::StateStore&, const std::shared_ptr<exec::Backend>&) const {
        return Error{ErrorCategory::unsupported, "Model state handoff is unsupported"};
    }
    virtual std::vector<std::string> observation_names(const model::LoadedModel&) const { return {}; }
    virtual Result<StateDescription> describe_states(const model::ModelDefinition& definition) const = 0;
    virtual exec::Status initialize_session(const std::shared_ptr<model::LoadedModel>& model,
        session::KVCache& kv, session::StateStore& states, exec::Budget& budget,
        const exec::Stages& stages) const = 0;
    virtual Result<std::unique_ptr<ForwardResult>> forward(
        const std::shared_ptr<model::LoadedModel>& model, const ForwardRequest& request,
        session::KVCache& kv, session::StateStore& states, exec::Budget& budget) const = 0;
    virtual Result<WeightActivity> finish_pending_weights(const std::shared_ptr<model::LoadedModel>&,
        session::StateStore&, FinishReason) const { return WeightActivity{}; }
    virtual exec::Status save_state(session::StateStore&) const { return Error{ErrorCategory::unsupported, "Checkpoint saving is unsupported"}; }
    virtual exec::Status restore_state(session::StateStore&) const { return Error{ErrorCategory::unsupported, "Checkpoint restoration is unsupported"}; }
    virtual exec::Status discard_saved_state(session::StateStore&) const { return Error{ErrorCategory::unsupported, "Checkpoint discard is unsupported"}; }
    virtual exec::Status reset_session(const std::shared_ptr<model::LoadedModel>& model,
        session::KVCache& kv, session::StateStore& states) const = 0;
    virtual void release_session(const std::shared_ptr<model::LoadedModel>& model,
        session::KVCache& kv, session::StateStore& states) const = 0;
    virtual void close_model(model::LoadedModel& model) const = 0;
};
}
