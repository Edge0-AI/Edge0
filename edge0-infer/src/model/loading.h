#pragma once
#include "model/adapter.h"
#include "exec/execution.h"
#include <atomic>
#include <functional>
namespace edge0::model {
struct WeightEstimate { std::size_t buffer_bytes = 0, metadata_bytes = 0, total_bytes = 0, resident_bytes = 0, pool_bytes = 0, temporary_bytes = 0; };
Result<WeightEstimate> estimate_weights(const WeightDescription& description,
    const std::shared_ptr<exec::Backend>& backend, ggml_backend_buffer_type_t type = nullptr);
struct LoadedWeight {
    std::shared_ptr<weights::Storage> storage;
    weights::Binding binding;
};
struct MixedWeights {
    exec::StagePolicy policy;
    std::vector<std::shared_ptr<exec::Backend>> candidates;
    std::map<exec::Backend*, std::map<std::string, LoadedWeight>> resident;
    std::map<exec::Backend*, std::pair<std::shared_ptr<weights::Storage>, std::shared_ptr<weights::Storage>>> temporary;
};
class LoadedModel {
public:
    explicit LoadedModel(exec::Budget budget) : weights(std::move(budget)) {}
    ~LoadedModel();
    exec::Status require_active() const;
    CompatibleAssets assets;
    ModelOptions options;
    GenerationDefaults generation_defaults;
    const Adapter* adapter = nullptr;
    exec::Stages stages;
    std::unique_ptr<MixedWeights> mixed;
    const std::map<std::string, LoadedWeight>& resident_on(const std::shared_ptr<exec::Backend>& backend) const {
        return mixed ? mixed->resident.at(backend.get()) : resident;
    }
    std::shared_ptr<weights::Storage> temporary_on(const std::shared_ptr<exec::Backend>& backend, bool next) const {
        if (!mixed) return next ? temporary_next : temporary;
        const auto& pair = mixed->temporary.at(backend.get()); return next ? pair.second : pair.first;
    }
    weights::Weights weights;
    std::map<std::string, LoadedWeight> resident;
    std::vector<std::shared_ptr<weights::Storage>> storages;
    std::vector<std::shared_ptr<weights::ExpertPool>> pools;
    std::shared_ptr<weights::Storage> temporary, temporary_next;
    std::vector<weights::PoolDescription> pool_descriptions;
    WeightEstimate estimate;
    std::function<void(LoadedModel&)> on_close;
    std::atomic<bool> unloading{false};
};
struct LoadReport { std::size_t weight_reads = 0; };
exec::Status validate_context_options(const WeightDescription&, const ModelOptions&, std::int64_t context);
struct LoadRequest {
    std::string directory, device;
    std::int64_t context = 4096;
    LoadReport* report = nullptr;
    ModelOptions options;
    std::function<void(std::size_t)> before_read;
    std::function<void(LoadedModel&)> on_close;
    std::string prefill_device;
    std::string stage_policy = "fixed";
    bool supports_handoff = false;
    exec::ConsumptionQuery consumption;
    exec::TransferHook transfer;
    std::optional<exec::StagePolicy> policy_override;
    std::function<bool(ggml_backend_t, ggml_backend_buffer_t)> direct;
};
Result<std::shared_ptr<LoadedModel>> load(const LoadRequest& request, const AdapterRegistry& adapters,
    exec::Budget& budget, const std::map<std::string, std::shared_ptr<exec::Backend>>& backends);
Result<MixedWeights> resolve_mixed_policy(const exec::Stages&, const LoadRequest&);
Result<WeightEstimate> estimate_mixed_weights(const WeightDescription&, const exec::Stages&, const MixedWeights&);
exec::Status load_mixed_weights(LoadedModel&, const WeightDescription&);
exec::Status request_unload(std::shared_ptr<LoadedModel>& model);
// Shared by single-backend and mixed resident packing: split a role so each
// storage stays within the backend buffer allocation limit.
struct BufferBound {
    ggml_backend_buffer_type_t type = nullptr;
    std::size_t limit = 0;
    std::size_t alignment = 1;
};
BufferBound buffer_bound(const std::vector<std::shared_ptr<exec::Backend>>&, const exec::ConsumptionQuery& = {});
Result<std::size_t> placed_bytes(const BufferBound&, const exec::Layout&);
Result<std::vector<std::vector<const ResidentWeight*>>> split_resident_group(
    const std::vector<const ResidentWeight*>&, const BufferBound&);
}
