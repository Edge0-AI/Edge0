#pragma once
#include "edge0/generation_check.h"
#include "app/runtime.h"
#include "app/text/rules.h"
#include "models/api/compute.h"
#include "model/loading.h"
#include "session/session.h"
#include <atomic>
#include <chrono>
#include <thread>
namespace edge0::app {
ReuseMethod reuse_method(session::ReuseKind);
OutputKind output_kind(model::ParsedKind);
exec::Status prepare_stage_forward(const models::ModelCompute&, const std::shared_ptr<model::LoadedModel>&,
    session::KVCache&, session::StateStore&, const models::ForwardRequest&, RuntimeRegistry* = nullptr);
exec::Status resolve_prefill_chunk(LoadConfiguration&, std::optional<std::int64_t>);
exec::Status resolve_context_capacity(LoadConfiguration&, std::optional<std::int64_t>,
    std::optional<std::int64_t>, std::uint64_t);
struct ModelHandle {
    RuntimeReference runtime;
    std::shared_ptr<RuntimeRegistry> registry;
    std::shared_ptr<model::LoadedModel> loaded;
    std::shared_ptr<text::Rules> rules;
    const models::ModelCompute* compute = nullptr;
    LoadConfiguration configuration;
    std::vector<AssetIdentity> assets;
    std::chrono::steady_clock::time_point (*clock_now)() = std::chrono::steady_clock::now;
    std::atomic<bool> unloading{false};
    ModelHandle(RuntimeReference r, std::shared_ptr<RuntimeRegistry> registry_value)
        : runtime(std::move(r)), registry(std::move(registry_value)) {}
};
struct SessionHandle {
    ModelReference model;
    std::shared_ptr<session::Session> state;
    std::atomic<bool> releasing{false};
    bool running = false;
    std::vector<GenerationBlock> blocks;
    std::atomic<bool> weights_failed{false};
};
void cleanup_registered(const std::shared_ptr<RuntimeRegistry>&);
// Internal injection point: simulated adapters use the same generation loop and compute API.
Result<ModelReference> register_model(const RuntimeReference&, std::shared_ptr<model::LoadedModel>,
    std::shared_ptr<text::Rules>, const models::ModelCompute&, const LoadConfiguration&,
    std::vector<AssetIdentity> assets = {});
}
