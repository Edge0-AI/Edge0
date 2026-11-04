#pragma once
#include "models/api/compute.h"
namespace edge0::models::edge0_35b {
inline const std::vector<std::uint32_t>& attention_layers(const model::ModelDefinition& definition) {
    return definition.layer_groups.at("attention");
}
inline const std::vector<std::uint32_t>& deltanet_layers(const model::ModelDefinition& definition) {
    return definition.layer_groups.at("deltanet");
}
Result<StateDescription> describe_states(const model::ModelDefinition& definition);
}
namespace edge0::models::edge0_35b {
struct DeltaState {
    std::shared_ptr<exec::StateStorage> convolution, recurrent;
};
exec::Status initialize_session(const std::shared_ptr<model::LoadedModel>& model,
    session::KVCache& kv, session::StateStore& states, exec::Budget& budget,
    const exec::Stages& stages);
exec::Status hand_off(const std::shared_ptr<model::LoadedModel>&, session::StateStore&, const std::shared_ptr<exec::Backend>&);
exec::Status reset_session(const std::shared_ptr<model::LoadedModel>& model,
    session::KVCache& kv, session::StateStore& states);
void release_session(const std::shared_ptr<model::LoadedModel>& model,
    session::KVCache& kv, session::StateStore& states);
void close_model(model::LoadedModel& model);
}
namespace edge0::models::edge0_35b {
std::vector<std::string> observation_names(const model::ModelDefinition& definition, bool prerouter = false);
Result<std::unique_ptr<ForwardResult>> forward(const std::shared_ptr<model::LoadedModel>& model,
    const ForwardRequest& request, session::KVCache& kv, session::StateStore& states, exec::Budget& budget);
}
namespace edge0::models::edge0_35b {
const ModelCompute& compute();
}

namespace edge0::models::edge0_35b {
struct CheckpointState {
    std::map<std::uint32_t, DeltaState> layers;
    bool valid = false;
};
exec::Status copy_block_snapshot(session::StateStore&, bool restore);
exec::Status save_state(session::StateStore&);
exec::Status restore_state(session::StateStore&);
exec::Status discard_saved_state(session::StateStore&);
struct Prediction { std::vector<std::int32_t> ids; std::vector<float> scores; };
struct PrerouterState {
    std::int64_t position = -1;
    std::vector<Prediction> predictions;
    std::vector<std::vector<std::int32_t>> previous;
    std::vector<std::shared_ptr<weights::Preparation>> pending;
};
Result<WeightActivity> finish_pending_weights(const std::shared_ptr<model::LoadedModel>& model,
    session::StateStore& states, FinishReason reason);
}
