#include "model/loading.h"
#include <algorithm>
#include <limits>
namespace edge0::model {
namespace {
bool add(std::size_t& value, std::size_t increment) {
    if (increment > std::numeric_limits<std::size_t>::max() - value) return false;
    value += increment;
    return true;
}
bool needs_staging(weights::Weights& weights, const std::shared_ptr<exec::Backend>& backend) {
    if (!backend) return false;
    auto* type = ggml_backend_get_default_buffer_type(backend->handle());
    if (!type) return true;
    const auto bytes = std::max<std::size_t>(ggml_backend_buft_get_alignment(type), 1);
    auto* buffer = ggml_backend_buft_alloc_buffer(type, bytes);
    if (!buffer) return true;
    const bool staged = !weights.direct(backend->handle(), buffer);
    ggml_backend_buffer_free(buffer);
    return staged;
}
}
BufferBound buffer_bound(const std::vector<std::shared_ptr<exec::Backend>>& consumers, const exec::ConsumptionQuery& consumption) {
    std::vector<ggml_backend_buffer_type_t> types;
    if (auto* home = exec::storage_home(consumers, consumption)) types.push_back(home);
    else for (const auto& backend : consumers)
        types.push_back(ggml_backend_get_default_buffer_type(backend->handle()));
    BufferBound bound{nullptr, std::numeric_limits<std::size_t>::max(), 1};
    for (auto* type : types) {
        if (!type) continue;
        const auto limit = ggml_backend_buft_get_max_size(type);
        if (limit < bound.limit) { bound.type = type; bound.limit = limit; bound.alignment = ggml_backend_buft_get_alignment(type); }
    }
    return bound;
}
Result<std::size_t> placed_bytes(const BufferBound& bound, const exec::Layout& layout) {
    if (!bound.type) return std::size_t{0};
    ggml_tensor tensor{};
    if (layout.dimensions.empty() || layout.dimensions.size() > GGML_MAX_DIMS ||
        layout.type < 0 || layout.type >= GGML_TYPE_COUNT || ggml_type_size(layout.type) == 0)
        return Error{ErrorCategory::invalid_argument, "Invalid tensor layout"};
    tensor.type = layout.type;
    std::fill(std::begin(tensor.ne), std::end(tensor.ne), 1);
    std::copy(layout.dimensions.begin(), layout.dimensions.end(), tensor.ne);
    const auto block = ggml_blck_size(layout.type);
    if (tensor.ne[0] <= 0 || tensor.ne[0] % block) return Error{ErrorCategory::invalid_argument, "Invalid block dimension"};
    tensor.nb[0] = ggml_type_size(layout.type);
    std::size_t bytes = tensor.nb[0];
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        const auto count = i == 0 ? tensor.ne[i] / block : tensor.ne[i];
        if (count <= 0 || static_cast<std::uint64_t>(count) > std::numeric_limits<std::size_t>::max() / bytes)
            return Error{ErrorCategory::invalid_argument, "Tensor size overflow"};
        bytes *= static_cast<std::size_t>(count);
        if (i + 1 < GGML_MAX_DIMS) tensor.nb[i + 1] = bytes;
    }
    const auto stored = ggml_backend_buft_get_alloc_size(bound.type, &tensor);
    if (stored > std::numeric_limits<std::size_t>::max() - bound.alignment) return Error{ErrorCategory::invalid_argument, "Tensor size overflow"};
    return (stored + bound.alignment - 1) / bound.alignment * bound.alignment;
}
Result<std::vector<std::vector<const ResidentWeight*>>> split_resident_group(
    const std::vector<const ResidentWeight*>& group, const BufferBound& bound) {
    std::vector<std::vector<const ResidentWeight*>> parts(1);
    std::size_t used = 0;
    for (const auto* weight : group) {
        auto bytes = placed_bytes(bound, weight->source.layout);
        if (!bytes) return bytes.error();
        if (bytes.value() > bound.limit)
            return Error{ErrorCategory::invalid_argument, "Storage exceeds backend buffer limit"};
        if (!parts.back().empty() && used > bound.limit - bytes.value()) {
            parts.emplace_back(); used = 0;
        }
        parts.back().push_back(weight);
        used += bytes.value();
    }
    return parts;
}
Result<WeightEstimate> estimate_weights(const WeightDescription& description,
    const std::shared_ptr<exec::Backend>& backend, ggml_backend_buffer_type_t type) {
    if (!backend) return Error{ErrorCategory::invalid_argument, "Missing weight backend"};
    if (!type) type = ggml_backend_get_default_buffer_type(backend->handle());
    const auto alignment = ggml_backend_buft_get_alignment(type);
    if (!alignment) return Error{ErrorCategory::backend_error, "Invalid buffer alignment"};
    WeightEstimate result;
    auto account = [&](const std::vector<weights::Array>& arrays, std::size_t& subtotal) -> exec::Status {
      for (const auto& array : arrays) {
        auto layout = array.layout;
        if (layout.dimensions.empty() || layout.dimensions.size()>3) return Error{ErrorCategory::incompatible,"Invalid array layout"};
        if (!array.regions || array.regions > 1000000) return Error{ErrorCategory::incompatible,"Invalid weight region count"};
        layout.dimensions.push_back(static_cast<std::int64_t>(array.regions));
        if (layout.dimensions.empty() || layout.dimensions.size() > 4 || layout.type < 0 ||
            layout.type >= GGML_TYPE_COUNT || !ggml_type_size(layout.type))
            return Error{ErrorCategory::incompatible, "Invalid resident weight layout"};
        ggml_tensor tensor{};
        tensor.type = layout.type;
        std::fill(std::begin(tensor.ne), std::end(tensor.ne), 1);
        std::copy(layout.dimensions.begin(), layout.dimensions.end(), tensor.ne);
        const auto block = ggml_blck_size(layout.type);
        if (tensor.ne[0] <= 0 || tensor.ne[0] % block)
            return Error{ErrorCategory::incompatible, "Invalid resident weight block dimension"};
        tensor.nb[0] = ggml_type_size(layout.type);
        std::size_t bytes = tensor.nb[0];
        for (int axis = 0; axis < 4; ++axis) {
            const auto extent = axis == 0 ? tensor.ne[axis] / block : tensor.ne[axis];
            if (extent <= 0 || std::uint64_t(extent) > std::numeric_limits<std::size_t>::max() / bytes)
                return Error{ErrorCategory::incompatible, "Resident weight size overflow"};
            bytes *= std::size_t(extent);
            if (axis < 3) tensor.nb[axis + 1] = bytes;
        }
        const auto allocation = ggml_backend_buft_get_alloc_size(type, &tensor);
        if (allocation > std::numeric_limits<std::size_t>::max() - alignment ||
            !add(result.buffer_bytes, (allocation + alignment - 1) / alignment * alignment) ||
            !add(result.metadata_bytes, (1 + array.regions) * ggml_tensor_overhead()))
            return Error{ErrorCategory::incompatible, "Weight estimate overflow"};
        if (!add(subtotal, (allocation + alignment - 1) / alignment * alignment) ||
            !add(subtotal, (1 + array.regions) * ggml_tensor_overhead()))
            return Error{ErrorCategory::incompatible,"Weight estimate overflow"};
      }
      return {};
    };
    for (const auto& weight : description.resident)
        if (auto e=account({{weight.source.layout,1}},result.resident_bytes)) return *e;
    for (const auto& pool : description.pools) {
        std::vector<weights::Array> arrays;
        for (const auto& layout : pool.projections) arrays.push_back({layout,pool.slots});
        if (auto e=account(arrays,result.pool_bytes)) return *e;
    }
    if(description.temporary_copies<1 || description.temporary_copies>2)
        return Error{ErrorCategory::incompatible,"Invalid temporary storage count"};
    for(std::size_t copy=0;copy<description.temporary_copies;++copy)
        if (auto e=account(description.temporary,result.temporary_bytes)) return *e;
    result.total_bytes = result.buffer_bytes;
    if (!add(result.total_bytes, result.metadata_bytes))
        return Error{ErrorCategory::incompatible, "Weight estimate overflow"};
    return result;
}
LoadedModel::~LoadedModel() {
    if (on_close) on_close(*this);
    for (auto& pool : pools) pool->close();
    if (temporary) weights.close(temporary);
    if (temporary_next) weights.close(temporary_next);
    resident.clear();
    for (auto it = storages.rbegin(); it != storages.rend(); ++it) weights.release_storage(*it);
}
exec::Status LoadedModel::require_active() const {
    if (unloading.load()) return Error{ErrorCategory::conflict, "Model is unloading"};
    return {};
}
exec::Status request_unload(std::shared_ptr<LoadedModel>& model) {
    if (!model) return Error{ErrorCategory::invalid_argument, "Missing model"};
    model->unloading.store(true);
    model.reset();
    return {};
}
exec::Status validate_context_options(const WeightDescription& description, const ModelOptions& options, std::int64_t context) {
    for (const auto& option : description.configuration) {
        if (!option.context_tokens) continue;
        const auto found=options.find(option.name);
        const auto& text=found==options.end()?option.default_value:found->second;
        const auto value=parse_count(text);
        if (!value || context<=0 || value.value()<1 || value.value()>std::uint64_t(context))
            return Error{ErrorCategory::invalid_argument,option.name+" must be an integer from 1 to context capacity"};
    }
    return {};
}
Result<std::shared_ptr<LoadedModel>> load(const LoadRequest& request, const AdapterRegistry& adapters,
    exec::Budget& budget, const std::map<std::string, std::shared_ptr<exec::Backend>>& backends) {
    if (request.report) *request.report = {};
    const auto selected = backends.find(request.device);
    if (selected == backends.end() || !selected->second) return Error{ErrorCategory::unsupported, "Device is outside runtime scope"};
    const auto prefill = request.prefill_device.empty() ? request.device : request.prefill_device;
    if (!backends.count(prefill)) return Error{ErrorCategory::unsupported, "Prefill device is outside runtime scope"};
    auto stages = exec::resolve_stages(request.device, request.device, selected->second->facility);
    if (prefill != request.device) {
        if (!request.supports_handoff) return Error{ErrorCategory::unsupported, "Model does not support state handoff"};
        stages = exec::resolve_stage_backends(prefill, request.device, backends);
    }
    if (!stages) return stages.error();
    auto resolved_stages = stages.value();
    if (request.stage_policy != "fixed" && !request.policy_override)
        return Error{ErrorCategory::invalid_argument, "Unknown stage policy: " + request.stage_policy};
    resolved_stages.consumption = request.consumption; resolved_stages.transfer = request.transfer;
    auto defaults = read_generation_defaults(request.directory);
    if (!defaults) return defaults.error();
    auto catalog = open_catalog(request.directory, adapters);
    if (!catalog) return catalog.error();
    auto adapter = select_adapter(catalog.value(), adapters);
    if (!adapter) return adapter.error();
    auto assets = check_compatibility(catalog.value(), *adapter.value());
    if (!assets) return assets.error();
    auto declaration = adapter.value()->describe_weight_storage(assets.value().indexes, request.options);
    if (!declaration) return declaration.error();
    for (const auto& option : request.options) {
        const auto& items=declaration.value().configuration;
        if (std::none_of(items.begin(),items.end(),[&](const ConfigurationItem& item) { return item.name==option.first; }))
            return Error{ErrorCategory::invalid_argument,"Unknown model option: "+option.first};
    }
    if (auto error=validate_context_options(declaration.value(),request.options,request.context)) return *error;
    std::unique_ptr<MixedWeights> mixed;
    if(resolved_stages.mixed()) {
        auto policy=resolve_mixed_policy(resolved_stages,request);if(!policy)return policy.error();
        mixed=std::make_unique<MixedWeights>(policy.value());
    }
    auto estimate = mixed ? estimate_mixed_weights(declaration.value(), resolved_stages, *mixed) :
        estimate_weights(declaration.value(), resolved_stages.prefill);
    if (!estimate) return estimate.error();
    auto result = std::make_shared<LoadedModel>(budget);
    struct ReportReads {
        LoadedModel& model;
        LoadReport* report;
        ~ReportReads() { if (report) report->weight_reads = model.weights.read_count(); }
    } report_reads{*result, request.report};
    result->weights.before_read = request.before_read;
    if (request.direct) result->weights.direct = request.direct;
    std::vector<std::shared_ptr<exec::Backend>> staging_targets{resolved_stages.prefill};
    if (mixed) {
        for (const auto& backend : mixed->candidates)
            if (std::find(staging_targets.begin(), staging_targets.end(), backend) == staging_targets.end())
                staging_targets.push_back(backend);
        if (std::find(staging_targets.begin(), staging_targets.end(), resolved_stages.decode) == staging_targets.end())
            staging_targets.push_back(resolved_stages.decode);
    }
    std::size_t staging = 0;
    for (const auto& backend : staging_targets)
        if (needs_staging(result->weights, backend)) { staging = result->weights.staging_bytes(); break; }
    auto accounted = estimate.value();
    if (staging && !add(accounted.total_bytes, staging))
        return Error{ErrorCategory::incompatible, "Weight estimate overflow"};
    const auto available = budget.limit() - budget.current();
    if (accounted.total_bytes > available)
        return Error{ErrorCategory::budget_exceeded, "Weights require " + std::to_string(accounted.total_bytes) +
            " bytes (resident=" + std::to_string(accounted.resident_bytes) + ", pools=" +
            std::to_string(accounted.pool_bytes) + ", temporary=" + std::to_string(accounted.temporary_bytes) + ", buffer=" + std::to_string(accounted.buffer_bytes) + ", metadata=" +
            std::to_string(accounted.metadata_bytes) + "), available=" + std::to_string(available)};
    if (staging) if (auto error = result->weights.ensure_staging()) return *error;
    result->generation_defaults = defaults.value();
    result->adapter = adapter.value();
    result->assets = assets.value();
    result->stages = resolved_stages;
    result->mixed = std::move(mixed);
    result->estimate = accounted;
    for (const auto& option : declaration.value().configuration)
        result->options[option.name] = request.options.count(option.name) ? request.options.at(option.name) : option.default_value;
    if (result->stages.mixed()) {
        if (auto error = load_mixed_weights(*result, declaration.value())) return *error;
        result->on_close = request.on_close;
        return result;
    }
    std::map<std::string, std::vector<const ResidentWeight*>> groups;
    for (const auto& weight : declaration.value().resident) groups[weight.role].push_back(&weight);
    // Allocate all groups before any payload read, splitting roles that exceed the backend buffer limit.
    struct Pending { std::shared_ptr<weights::Storage> storage; std::vector<weights::LoadItem> items; };
    std::vector<Pending> pending;
    const auto bound = buffer_bound({result->stages.prefill}, result->stages.consumption);
    for (const auto& group : groups) {
        auto parts = split_resident_group(group.second, bound);
        if (!parts) return parts.error();
        for (const auto& part : parts.value()) {
            std::vector<weights::Array> arrays;
            for (const auto* weight : part) arrays.push_back({weight->source.layout, 1});
            auto storage = result->weights.create_storage(result->stages.prefill, arrays, edge0::exec::Purpose::resident_weights);
            if (!storage) return storage.error();
            Pending load{storage.value(), {}};
            for (std::size_t i = 0; i < part.size(); ++i)
                load.items.push_back({i, part[i]->id, part[i]->source});
            result->storages.push_back(load.storage);
            pending.push_back(std::move(load));
        }
    }
    result->pool_descriptions = declaration.value().pools;
    for (const auto& pool : declaration.value().pools) {
        auto created = weights::ExpertPool::create_pools(result->weights,result->stages.prefill,pool);
        if (!created) return created.error();
        result->pools.push_back(created.value());
    }
    if (!declaration.value().temporary.empty()) {
        auto created=result->weights.create_storage(result->stages.prefill,declaration.value().temporary, edge0::exec::Purpose::experts,weights::Protection::group);
        if (!created) return created.error();
        result->temporary=created.value();
        if(declaration.value().temporary_copies==2) {
            auto next=result->weights.create_storage(result->stages.prefill,declaration.value().temporary, edge0::exec::Purpose::experts,weights::Protection::group);
            if(!next) return next.error();
            result->temporary_next=next.value();
        }
    }
    for (const auto& load : pending) {
        auto planned = result->weights.plan_resident_load(load.storage, load.items);
        if (!planned) return planned.error();
        auto write = result->weights.begin_write(load.storage, planned.value());
        if (!write) return write.error();
        if (auto error = result->weights.fill_regions(write.value())) return *error;
        auto bindings = result->weights.complete(write.value());
        if (!bindings) return bindings.error();
        for (const auto& binding : bindings.value()) result->resident.emplace(binding.id, LoadedWeight{load.storage, binding});
    }
    result->on_close = request.on_close;
    return result;
}
}
