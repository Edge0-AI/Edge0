#include "models/edge0_35b/compute.h"
#include <limits>
namespace edge0::models::edge0_35b {
Result<StateDescription> describe_states(const model::ModelDefinition& definition) {
    const auto heads = definition.parameters.find("n_head_kv");
    const auto width = definition.parameters.find("head_dim");
    if (heads == definition.parameters.end() || width == definition.parameters.end() ||
        heads->second == 0 || width->second == 0 ||
        heads->second > std::uint64_t(std::numeric_limits<std::int64_t>::max()) ||
        width->second > std::uint64_t(std::numeric_limits<std::int64_t>::max()))
        return Error{ErrorCategory::incompatible, "Invalid attention state dimensions"};
    StateDescription result;
    for (auto layer : deltanet_layers(definition)) result.updated_entries.push_back("deltanet." + std::to_string(layer));
    result.evolves_as_whole = !deltanet_layers(definition).empty();
    result.supports_save_restore = result.evolves_as_whole;
    if (result.evolves_as_whole) {
        const auto& p = definition.parameters;
        const auto width = p.at("state_dim"), heads = p.at("n_value_head");
        const auto channels = width * (2 * p.at("n_key_head") + heads);
        result.checkpoint_bytes = deltanet_layers(definition).size() * sizeof(float) *
            ((p.at("conv_kernel") - 1) * channels + width * width * heads);
    }
    result.block_snapshot_bytes = result.checkpoint_bytes;
    for (auto layer : attention_layers(definition)) {
        for (const auto* suffix : {"k", "v"}) {
            result.kv.push_back({"attn." + std::to_string(layer) + "." + suffix, GGML_TYPE_F16,
                {static_cast<std::int64_t>(width->second), static_cast<std::int64_t>(heads->second)},
                1, 256, session::Visibility::causal});
        }
    }
    return result;
}
}

#include "model/loading.h"
namespace edge0::models::edge0_35b {
exec::Status initialize_session(const std::shared_ptr<model::LoadedModel>& model,
    session::KVCache& kv, session::StateStore& states, exec::Budget& budget,
    const exec::Stages& stages) {
    (void)kv;
    if (!model) return Error{ErrorCategory::invalid_argument, "Missing model"};
    if (auto error = model->require_active()) return error;
    if (stages.prefill != model->stages.prefill || stages.decode != model->stages.decode) return Error{ErrorCategory::invalid_argument, "Session backend mismatch"};
    const auto& definition = model->assets.definition;
    const auto& p = definition.parameters;
    for (const auto layer : deltanet_layers(definition)) {
        auto state=states.get<DeltaState>("deltanet." + std::to_string(layer));
        if (!state) return state.error();
        if (state.value()) return Error{ErrorCategory::conflict, "DeltaNet state already initialized"};
    }
    for (const auto layer : deltanet_layers(definition)) {
        auto state = std::make_shared<DeltaState>();
        const auto width = static_cast<std::int64_t>(p.at("state_dim"));
        const auto heads = static_cast<std::int64_t>(p.at("n_value_head"));
        const auto channels = width * static_cast<std::int64_t>(2 * p.at("n_key_head") + p.at("n_value_head"));
        auto convolution = exec::allocate_state_storage(budget, stages,
            {GGML_TYPE_F32, {static_cast<std::int64_t>(p.at("conv_kernel")) - 1, channels}}, edge0::exec::Purpose::session_state);
        if (!convolution) return convolution.error();
        state->convolution = convolution.value();
        auto recurrent = exec::allocate_state_storage(budget, stages, {GGML_TYPE_F32, {width, width, heads}}, edge0::exec::Purpose::session_state);
        if (!recurrent) return recurrent.error();
        state->recurrent = recurrent.value();
        // llama.cpp 7ab4ee7, src/llama-memory-recurrent.cpp: new sequence state is zero.
        if (auto error = state->convolution->reset_zero()) return error;
        if (auto error = state->recurrent->reset_zero()) return error;
        if (auto error = states.put("deltanet." + std::to_string(layer), state)) return error;
    }
    for (const auto* name : {"aux.checkpoint", "aux.block_snapshot"}) {
    auto checkpoint = std::make_shared<CheckpointState>();
    if (auto error = states.put(name, checkpoint)) return error;
    for (const auto layer : deltanet_layers(definition)) {
        auto current_result=states.get<DeltaState>("deltanet." + std::to_string(layer));
        if (!current_result) return current_result.error();
        const auto current=current_result.value();
        auto& saved = checkpoint->layers[layer];
        for (const auto& pair : {std::make_pair(current->convolution, &saved.convolution), std::make_pair(current->recurrent, &saved.recurrent)}) {
            const auto* tensor = pair.first->tensor();
            auto storage = exec::allocate_state_storage(budget, stages,
                {tensor->type, {tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]}}, edge0::exec::Purpose::session_state);
            if (!storage) return storage.error();
            *pair.second = storage.value();
        }
    }
    }
    return {};
}
exec::Status reset_session(const std::shared_ptr<model::LoadedModel>& model,
    session::KVCache& kv, session::StateStore& states) {
    (void)kv;
    if (!model) return Error{ErrorCategory::invalid_argument, "Missing model"};
    auto finished=finish_pending_weights(model,states,FinishReason::normal);
    if(!finished) return finished.error();
    states.remove("aux.prerouter");
    if (auto error = discard_saved_state(states)) return error;
    for (const auto layer : deltanet_layers(model->assets.definition)) {
        auto state_result=states.get<DeltaState>("deltanet." + std::to_string(layer));
        if (!state_result) return state_result.error();
        auto state=state_result.value();
        if (!state) return Error{ErrorCategory::conflict, "DeltaNet state is not initialized"};
        if (auto error = state->convolution->reset_zero()) return error;
        if (auto error = state->recurrent->reset_zero()) return error;
    }
    return {};
}
void release_session(const std::shared_ptr<model::LoadedModel>& model,
    session::KVCache& kv, session::StateStore& states) {
    (void)kv;
    if (!model) return;
    finish_pending_weights(model,states,FinishReason::failure);
    states.remove_prefix("aux.");
    for (const auto layer : deltanet_layers(model->assets.definition))
        states.remove("deltanet." + std::to_string(layer));
}
void close_model(model::LoadedModel& model) {
    // No model-level auxiliary objects: all weight uses belong to individual graphs.
    (void)model;
}
}
namespace edge0::models::edge0_35b {
namespace {
class Compute final : public ModelCompute {
public:
    std::vector<std::string> observation_names(const model::LoadedModel& m) const override {
        return edge0_35b::observation_names(m.assets.definition, m.options.at("routing")!="actual");
    }
    exec::StageFacts stage_facts(const model::LoadedModel& model, session::StateStore& states,
        const std::shared_ptr<exec::Backend>& target) const override {
        exec::StageFacts facts;
        for (const auto layer : deltanet_layers(model.assets.definition)) {
            auto value = states.get<DeltaState>("deltanet." + std::to_string(layer));
            if (!value || !value.value()) continue;
            for (const auto& state : {value.value()->convolution, value.value()->recurrent}) {
                if (!facts.state_home) facts.state_home = state->allocation->backend;
                if (state->replicas && !state->replicas->shared) {
                    auto view = state->on(target);
                    if (view && !view.value()->current()) facts.handoff_bytes += ggml_nbytes(state->tensor());
                }
            }
        }
        return facts;
    }
    exec::TransferStats transfer_stats(const model::LoadedModel& model, session::StateStore& states) const override {
        exec::TransferStats result;
        auto add = [&](const DeltaState& state) {
            for (const auto& storage : {state.convolution, state.recurrent}) if (storage && storage->replicas) {
                const auto& stats = storage->replicas->stats;
                result.count += stats.count; result.bytes += stats.bytes; result.seconds += stats.seconds;
            }
        };
        for (const auto layer : deltanet_layers(model.assets.definition)) {
            auto value = states.get<DeltaState>("deltanet." + std::to_string(layer));
            if (value && value.value()) add(*value.value());
        }
        for (const auto* name : {"aux.checkpoint", "aux.block_snapshot"}) {
            auto value = states.get<CheckpointState>(name);
            if (value && value.value()) for (const auto& entry : value.value()->layers) add(entry.second);
        }
        return result;
    }
    bool supports_handoff() const override { return true; }
    exec::Status hand_off(const std::shared_ptr<model::LoadedModel>& model, session::StateStore& states,
        const std::shared_ptr<exec::Backend>& backend) const override {
        return edge0_35b::hand_off(model, states, backend);
    }
    Result<StateDescription> describe_states(const model::ModelDefinition& d) const override {
        return edge0_35b::describe_states(d);
    }
    exec::Status initialize_session(const std::shared_ptr<model::LoadedModel>& m,session::KVCache& kv,
        session::StateStore& states,exec::Budget& budget,const exec::Stages& stages) const override {
        return edge0_35b::initialize_session(m,kv,states,budget,stages);
    }
    Result<std::unique_ptr<ForwardResult>> forward(const std::shared_ptr<model::LoadedModel>& m,
        const ForwardRequest& request,session::KVCache& kv,session::StateStore& states,exec::Budget& budget) const override {
        return edge0_35b::forward(m,request,kv,states,budget);
    }
    Result<WeightActivity> finish_pending_weights(const std::shared_ptr<model::LoadedModel>& m,
        session::StateStore& states,FinishReason reason) const override {
        return edge0_35b::finish_pending_weights(m,states,reason);
    }
    exec::Status save_state(session::StateStore& s) const override { return edge0_35b::save_state(s); }
    exec::Status restore_state(session::StateStore& s) const override { return edge0_35b::restore_state(s); }
    exec::Status discard_saved_state(session::StateStore& s) const override { return edge0_35b::discard_saved_state(s); }
    exec::Status reset_session(const std::shared_ptr<model::LoadedModel>& m, session::KVCache& kv,
        session::StateStore& states) const override { return edge0_35b::reset_session(m, kv, states); }
    void release_session(const std::shared_ptr<model::LoadedModel>& m,session::KVCache& kv,session::StateStore& states) const override {
        edge0_35b::release_session(m,kv,states);
    }
    void close_model(model::LoadedModel& m) const override { edge0_35b::close_model(m); }
};
}
const ModelCompute& compute() { static const Compute implementation; return implementation; }
}

namespace edge0::models::edge0_35b {
Result<WeightActivity> finish_pending_weights(const std::shared_ptr<model::LoadedModel>& model,
    session::StateStore& states,FinishReason reason) {
    WeightActivity activity;
    auto state_result=states.get<PrerouterState>("aux.prerouter");
    if (!state_result) return state_result.error();
    auto state=state_result.value();
    if(!state) return activity;
    exec::Status failure;
    for(std::size_t layer=0;layer<state->pending.size();++layer) {
        auto& pending=state->pending[layer];
        if(!pending) continue;
        if(reason==FinishReason::failure || failure) model->pools.at(layer)->abandon(pending);
        else {
            auto ready=model->pools.at(layer)->await_ready(pending);
            if(!ready) failure=ready.error();
            else {
                activity.decode+=ready.value().activity;
                activity.prefetch.bytes+=ready.value().activity.bytes;
                activity.prefetch.tail_bytes+=ready.value().activity.bytes;
            }
        }
        pending.reset();
    }
    if(failure || reason==FinishReason::failure) {
        state->position=-1;
        for(auto& forecast:state->predictions) forecast={};
        for(auto& previous:state->previous) previous.clear();
    }
    if(failure) return *failure;
    return activity;
}
}

namespace edge0::models::edge0_35b {
namespace {
exec::Status copy_checkpoint(session::StateStore& states, bool restore, const char* name = "aux.checkpoint") {
    auto saved_result=states.get<CheckpointState>(name);
    if (!saved_result) return saved_result.error();
    auto saved=saved_result.value();
    if (!saved || (restore && !saved->valid)) return Error{ErrorCategory::conflict, "Checkpoint is unavailable"};
    if (!restore) saved->valid = false;
    for (const auto& entry : saved->layers) {
        auto current_result=states.get<DeltaState>("deltanet." + std::to_string(entry.first));
        if (!current_result) return current_result.error();
        const auto current=current_result.value();
        if (!current) return Error{ErrorCategory::conflict, "DeltaNet state is unavailable"};
        for (const auto& pair : {std::make_pair(current->convolution, entry.second.convolution), std::make_pair(current->recurrent, entry.second.recurrent)}) {
            if (!pair.first || !pair.second) return Error{ErrorCategory::conflict, "Checkpoint storage mismatch"};
            if (pair.first->replicas) {
                const auto backend = pair.first->allocation->backend;
                if (restore) if (auto error = exec::transfer(pair.second, backend)) return error;
                auto saved_view = pair.second->on(backend); if (!saved_view) return saved_view.error();
                const auto source = restore ? saved_view.value() : pair.first;
                const auto target = restore ? pair.first : saved_view.value();
                if (!source->current()) return Error{ErrorCategory::conflict, "Checkpoint source is stale"};
                ggml_backend_synchronize(backend->handle());
                ggml_backend_tensor_copy(source->tensor(), target->tensor());
                ggml_backend_synchronize(backend->handle());
                target->mark_written();
                continue;
            }
            if (pair.first->allocation->backend != pair.second->allocation->backend)
                return Error{ErrorCategory::conflict, "Checkpoint storage mismatch"};
            ggml_backend_tensor_copy(restore ? pair.second->tensor() : pair.first->tensor(),
                                     restore ? pair.first->tensor() : pair.second->tensor());
        }
    }
    saved->valid = true;
    if (restore) states.remove("aux.prerouter");
    return {};
}
}
exec::Status copy_block_snapshot(session::StateStore& states, bool restore) {
    return copy_checkpoint(states, restore, "aux.block_snapshot");
}
exec::Status save_state(session::StateStore& states) { return copy_checkpoint(states, false); }
exec::Status restore_state(session::StateStore& states) { return copy_checkpoint(states, true); }
exec::Status discard_saved_state(session::StateStore& states) {
    auto saved_result=states.get<CheckpointState>("aux.checkpoint");
    if (!saved_result) return saved_result.error();
    auto saved=saved_result.value();
    if (saved) saved->valid = false;
    return {};
}
}

namespace edge0::models::edge0_35b {
exec::Status hand_off(const std::shared_ptr<model::LoadedModel>& model, session::StateStore& states,
    const std::shared_ptr<exec::Backend>& backend) {
    if (!model || !backend || (backend != model->stages.prefill && backend != model->stages.decode))
        return Error{ErrorCategory::invalid_argument, "Invalid state handoff backend"};
    if (!model->stages.mixed()) return {};
    std::vector<std::pair<std::shared_ptr<exec::StateStorage>*, std::shared_ptr<exec::StateStorage>>> replacements;
    for (const auto layer : deltanet_layers(model->assets.definition)) {
        auto value = states.get<DeltaState>("deltanet." + std::to_string(layer));
        if (!value) return value.error();
        if (!value.value()) return Error{ErrorCategory::conflict, "DeltaNet state is unavailable"};
        for (auto* state : {&value.value()->convolution, &value.value()->recurrent}) {
            if (auto error = exec::transfer(*state, backend)) return error;
            auto next = (*state)->on(backend); if (!next) return next.error();
            replacements.push_back({state, next.value()});
        }
    }
    for (auto& replacement : replacements) *replacement.first = std::move(replacement.second);
    return {};
}
}
