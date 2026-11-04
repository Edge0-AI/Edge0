#include "app/generation_internal.h"
#include "app/runtime_hooks.h"
namespace edge0::app {
exec::Status prepare_stage_forward(const models::ModelCompute& compute, const std::shared_ptr<model::LoadedModel>& model,
    session::KVCache& kv, session::StateStore& states, const models::ForwardRequest& request, RuntimeRegistry* registry) {
    if (!model->stages.mixed()) return {};
    if (!request.backend) return Error{ErrorCategory::invalid_argument, "Missing forward backend"};
    if (registry && registry->hooks && registry->hooks->skip_handoff) return {};
    if (auto error = kv.make_current(request.backend, request.step.e)) return error;
    if (registry && registry->hooks) ++registry->hooks->hand_off_calls;
    return compute.hand_off(model, states, request.backend);
}
}
