#include "model/loading.h"
#include <algorithm>
#include <limits>

namespace edge0::model {
Result<MixedWeights> resolve_mixed_policy(const exec::Stages& stages, const LoadRequest& request) {
    auto policy = request.policy_override ? Result<exec::StagePolicy>(*request.policy_override) : exec::stage_policy(request.stage_policy);
    if (!policy) return policy.error();
    if (!policy.value().candidates) return Error{ErrorCategory::invalid_argument, "Stage policy has no candidate function"};
    auto values = policy.value().candidates(stages);
    if (values.empty()) return Error{ErrorCategory::invalid_argument, "Stage policy candidates are empty"};
    std::vector<std::shared_ptr<exec::Backend>> unique;
    for (const auto& value : values) {
        if (!value || (value != stages.prefill && value != stages.decode))
            return Error{ErrorCategory::invalid_argument, "Stage policy candidate is outside stages"};
        if (std::find(unique.begin(), unique.end(), value) == unique.end()) unique.push_back(value);
    }
    MixedWeights result; result.policy=policy.value(); result.candidates=std::move(unique);
    return result;
}
namespace {
std::vector<std::shared_ptr<exec::Backend>> consumers_for(model::StorageStages use, const exec::Stages& stages,
    const std::vector<std::shared_ptr<exec::Backend>>& candidates) {
    auto result = use == StorageStages::decode ? std::vector<std::shared_ptr<exec::Backend>>{} : candidates;
    if (use != StorageStages::prefill && std::find(result.begin(), result.end(), stages.decode) == result.end())
        result.push_back(stages.decode);
    return result;
}
exec::Status add_estimate(WeightEstimate& total, const WeightEstimate& item) {
    for (const auto& pair : {std::make_pair(&total.buffer_bytes, item.buffer_bytes),
        std::make_pair(&total.metadata_bytes, item.metadata_bytes), std::make_pair(&total.total_bytes, item.total_bytes),
        std::make_pair(&total.resident_bytes, item.resident_bytes), std::make_pair(&total.pool_bytes, item.pool_bytes),
        std::make_pair(&total.temporary_bytes, item.temporary_bytes)}) {
        if (*pair.first > std::numeric_limits<std::size_t>::max() - pair.second)
            return Error{ErrorCategory::incompatible, "Mixed weight estimate overflow"};
        *pair.first += pair.second;
    }
    return {};
}
}
Result<WeightEstimate> estimate_mixed_weights(const WeightDescription& description, const exec::Stages& stages,
    const MixedWeights& mixed) {
    WeightEstimate total;
    WeightDescription resident; resident.resident = description.resident;
    const auto both = consumers_for(description.resident_stages, stages, mixed.candidates);
    const auto home = exec::storage_home(both, stages.consumption);
    for (std::size_t i = 0; i < (home ? 1u : both.size()); ++i) {
        auto value = estimate_weights(resident, both[i], home); if (!value) return value.error();
        if (auto error = add_estimate(total, value.value())) return *error;
    }
    if (home) {
        // Each wrapper carries array and region descriptors; the physical data is shared.
        std::size_t views = 0;
        for (const auto& backend : both)
            if (backend != both.front() || exec::buffer_consumption(home, backend, stages.consumption) != exec::Consumption::direct) ++views;
        const auto bytes = description.resident.size() * 2 * ggml_tensor_overhead() * views;
        WeightEstimate aliases; aliases.metadata_bytes = aliases.total_bytes = aliases.resident_bytes = bytes;
        if (auto error = add_estimate(total, aliases)) return *error;
    }
    if (description.pool_stages != StorageStages::decode)
        return Error{ErrorCategory::unsupported, "Expert pools require a single decode consumer"};
    WeightDescription pools; pools.pools = description.pools;
    auto slots = estimate_weights(pools, stages.decode); if (!slots) return slots.error();
    if (auto error = add_estimate(total, slots.value())) return *error;
    WeightDescription temporary; temporary.temporary = description.temporary; temporary.temporary_copies = description.temporary_copies;
    const auto temporary_consumers=consumers_for(description.temporary_stages, stages, mixed.candidates);
    const auto temporary_home=temporary_consumers.size()>1 ? exec::storage_home(temporary_consumers, stages.consumption) : nullptr;
    for (std::size_t i=0;i<(temporary_home?1u:temporary_consumers.size());++i) {
        auto value=estimate_weights(temporary,temporary_consumers[i],temporary_home);if(!value)return value.error();
        if(auto error=add_estimate(total,value.value()))return *error;
    }
    if(temporary_home) {
        std::size_t tensors=0,views=0;
        for(const auto& array:description.temporary)tensors+=1+array.regions;
        for(const auto& backend:temporary_consumers)
            if(backend!=temporary_consumers.front() || exec::buffer_consumption(temporary_home,backend,stages.consumption)!=exec::Consumption::direct)++views;
        WeightEstimate aliases;
        aliases.metadata_bytes=aliases.total_bytes=aliases.temporary_bytes=tensors*description.temporary_copies*views*ggml_tensor_overhead();
        if(auto error=add_estimate(total,aliases))return *error;
    }
    return total;
}
exec::Status load_mixed_weights(LoadedModel& model, const WeightDescription& description) {
    const auto consumers = consumers_for(description.resident_stages, model.stages, model.mixed->candidates);
    struct Pending {
        std::shared_ptr<weights::Storage> storage;
        std::vector<weights::LoadItem> items;
        std::vector<std::string> ids;
        std::vector<std::shared_ptr<exec::Backend>> consumers;
    };
    std::vector<Pending> pending;
    std::map<std::string, std::vector<const ResidentWeight*>> groups;
    for (const auto& weight : description.resident) groups[weight.role].push_back(&weight);
    const auto bound = buffer_bound(consumers, model.stages.consumption);
    for (const auto& group : groups) {
        // One backend allocation cannot exceed its buffer limit. A role larger than that
        // limit is stored as several storages; each weight is still published by its id.
        auto parts = split_resident_group(group.second, bound);
        if (!parts) return parts.error();
        for (const auto& part : parts.value()) {
            std::vector<weights::Array> arrays;
            for (const auto* weight : part) arrays.push_back({weight->source.layout, 1});
            auto allocated = model.weights.create_storage(consumers, arrays, edge0::exec::Purpose::resident_weights, weights::Protection::region, model.stages.consumption);
            if (!allocated) return allocated.error();
            for (std::size_t i = 0; i < allocated.value().size(); ++i) {
                Pending load{allocated.value()[i], {}, {}, allocated.value().size() == 1 ? consumers :
                    std::vector<std::shared_ptr<exec::Backend>>{consumers[i]}};
                for (std::size_t j = 0; j < part.size(); ++j) {
                    const auto* weight = part[j];
                    load.ids.push_back(weight->id);
                    load.items.push_back({j, weight->id + "/stage/" + std::to_string(i), weight->source});
                }
                model.storages.push_back(load.storage); pending.push_back(std::move(load));
            }
        }
    }
    model.pool_descriptions = description.pools;
    for (const auto& pool : description.pools) {
        auto created = weights::ExpertPool::create_pools(model.weights, model.stages.decode, pool);
        if (!created) return created.error();
        // Upload reordering and whole-tensor or write-once constraints need validation on the first device requiring staged uploads.
        if (!model.weights.direct(model.stages.decode->handle(), created.value()->storage()->allocation()->buffer))
            return Error{ErrorCategory::unsupported, "Expert target cannot be written directly"};
        model.pools.push_back(created.value());
    }
    const auto temporary_consumers=consumers_for(description.temporary_stages,model.stages,model.mixed->candidates);
    if(!description.temporary.empty()) for(std::size_t copy=0;copy<description.temporary_copies;++copy) {
        auto allocated=model.weights.create_storage(temporary_consumers,description.temporary, edge0::exec::Purpose::experts,weights::Protection::group,model.stages.consumption);
        if(!allocated)return allocated.error();
        for(const auto& storage:allocated.value()) model.storages.push_back(storage);
        for(std::size_t i=0;i<temporary_consumers.size();++i) {
            auto& pair=model.mixed->temporary[temporary_consumers[i].get()];
            (copy?pair.second:pair.first)=allocated.value()[allocated.value().size()==1?0:i];
        }
    }
    // No payload is read until allocation finishes. Expert pools still require a directly writable decode backend.
    for (const auto& load : pending) {
        auto write = model.weights.begin_write(load.storage, load.items); if (!write) return write.error();
        if (auto error = model.weights.fill_regions(write.value())) return error;
        auto bindings = model.weights.complete(write.value()); if (!bindings) return bindings.error();
        for (const auto& consumer : load.consumers) for (std::size_t i = 0; i < load.ids.size(); ++i)
            model.mixed->resident[consumer.get()].emplace(load.ids[i], LoadedWeight{load.storage, bindings.value()[i]});
    }
    return {};
}
}
