#include "edge0/engine.h"
#include "app/runtime.h"
#include "app/adapters.h"
#include "app/generation_internal.h"
#include "platform/backend.h"
#include "app/runtime_hooks.h"
#include "exec/budget.h"
#include "edge0/build_identity.h"
#include <limits>
#include "ggml.h"
#include <algorithm>
#include <memory>
#include <mutex>
#include <set>

namespace edge0 {
namespace {
struct RuntimeState {
    std::shared_ptr<RuntimeRegistry> registry = std::make_shared<RuntimeRegistry>();
    std::vector<std::shared_ptr<platform::BackendFacility>> facilities;
    RuntimeCpuThreads threads{};
    LogReceiver receiver;
    std::unique_ptr<exec::Budget> budget;
    BudgetSource budget_source = BudgetSource::device_default;
    ggml_log_callback previous_callback = nullptr;
    void* previous_data = nullptr;
};
std::mutex runtime_mutex;
std::unique_ptr<RuntimeState> current;
std::uint64_t generation = 0;

void forward_log(enum ggml_log_level level, const char* text, void* data) {
    LogLevel mapped = LogLevel::none;
    switch (level) {
        case GGML_LOG_LEVEL_DEBUG: mapped = LogLevel::debug; break;
        case GGML_LOG_LEVEL_INFO: mapped = LogLevel::info; break;
        case GGML_LOG_LEVEL_WARN: mapped = LogLevel::warning; break;
        case GGML_LOG_LEVEL_ERROR: mapped = LogLevel::error; break;
        case GGML_LOG_LEVEL_CONT: mapped = LogLevel::continuation; break;
        case GGML_LOG_LEVEL_NONE: break;
    }
    (*static_cast<const LogReceiver*>(data))(mapped, text);
}
}

Result<RuntimeReference> open_runtime(const RuntimeOptions& options) {
    std::lock_guard<std::mutex> lock(runtime_mutex);
    if (current) return Error{ErrorCategory::conflict, "A runtime is already open"};
    const std::set<std::string> unique(options.devices.begin(), options.devices.end());
    if (options.devices.empty() || unique.size() != options.devices.size())
        return Error{ErrorCategory::invalid_argument, "Devices must be explicit, nonempty and unique"};
    const auto defaults = platform::default_cpu_threads();
    const auto logical = std::thread::hardware_concurrency();
    bool has_cpu = false;
    for (const auto& name : options.devices) {
        const auto device = ggml_backend_dev_by_name(name.c_str());
        if (device && name == ggml_backend_dev_name(device) && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) has_cpu = true;
    }
    for (auto value : {options.cpu_threads, options.cpu_prefill_threads})
        if (value && (!has_cpu || *value < 1 || static_cast<unsigned>(*value) > (logical ? logical : 512u)))
            return Error{ErrorCategory::invalid_argument, "CPU threads require a CPU device and a value from 1 to the logical CPU count"};
    current = std::make_unique<RuntimeState>();
    current->threads = {options.cpu_threads.value_or(defaults.decode), options.cpu_prefill_threads.value_or(defaults.prefill),
        options.cpu_threads ? CpuThreadSource::explicit_value : CpuThreadSource::platform_default,
        options.cpu_prefill_threads ? CpuThreadSource::explicit_value : CpuThreadSource::platform_default};
    current->receiver = options.log_receiver;
    ggml_log_get(&current->previous_callback, &current->previous_data);
    if (current->receiver) ggml_log_set(forward_log, &current->receiver);
    for (const auto& device : options.devices) {
        auto result = platform::BackendFacility::create(device, {current->threads.cpu_threads, current->threads.cpu_prefill_threads});
        if (!result.first) {
            current->facilities.clear();
            ggml_log_set(current->previous_callback, current->previous_data);
            current.reset();
            return Error{ErrorCategory::backend_error, result.second};
        }
        current->facilities.push_back(std::make_shared<platform::BackendFacility>(std::move(result.first)));
    }
    std::size_t limit = std::numeric_limits<std::size_t>::max();
    const auto reported = platform::devices();
    for (const auto& name : options.devices) {
        const auto device = std::find_if(reported.begin(), reported.end(),
            [&](const auto& item) { return item.name == name; });
        limit = std::min(limit, device == reported.end() ? std::size_t(0) : device->memory_total_bytes);
    }
    if (limit == 0 || (options.memory_budget_bytes &&
        (*options.memory_budget_bytes == 0 || *options.memory_budget_bytes > limit))) {
        current->facilities.clear();
        ggml_log_set(current->previous_callback, current->previous_data);
        current.reset();
        return Error{ErrorCategory::invalid_argument, "Memory budget must be positive and not exceed device memory"};
    }
    current->budget_source = options.memory_budget_bytes ? BudgetSource::explicit_value : BudgetSource::device_default;
    current->budget = std::make_unique<exec::Budget>(options.memory_budget_bytes.value_or(limit));
    return RuntimeReference(++generation, options.devices);
}

Result<std::string> finish_runtime_use(const RuntimeReference& runtime) {
    std::lock_guard<std::mutex> lock(runtime_mutex);
    if (!current || runtime.generation_ != generation)
        return Error{ErrorCategory::runtime_ended, "Runtime has ended"};
    auto registry = current->registry;
    {
        std::lock_guard<std::mutex> guard(registry->mutex);
        if (registry->active) return Error{ErrorCategory::busy, "Runtime has an active call"};
        registry->active = true;
    }
    app::cleanup_registered(registry);
    for (const auto& facility : current->facilities)
        if (facility.use_count() != 1) {
            std::lock_guard<std::mutex> guard(registry->mutex);
            registry->active = false;
            return Error{ErrorCategory::conflict, "Runtime execution resources are still in use"};
        }
    { std::lock_guard<std::mutex> guard(registry->mutex); registry->ended = true; }
    current->facilities.clear();
    ggml_log_set(current->previous_callback, current->previous_data);
    current.reset();
    return std::string("Safely ended");
}

Result<RuntimeResources> RuntimeAccess::acquire(const RuntimeReference& runtime) {
    std::lock_guard<std::mutex> lock(runtime_mutex);
    if (!current || runtime.generation_ != generation)
        return Error{ErrorCategory::runtime_ended, "Runtime has ended"};
    RuntimeResources result{*current->budget, {}, current->registry};
    for (std::size_t i = 0; i < runtime.devices_.size(); ++i) {
        const auto& name = runtime.devices_[i];
        auto stages = exec::resolve_stages(name, name, current->facilities[i]);
        if (!stages) return stages.error();
        result.backends.emplace(name, stages.value().prefill);
    }
    if (result.registry->hooks) for (const auto& alias : result.registry->hooks->aliases)
        result.backends.emplace(alias);
    return result;
}

Result<RuntimeBudget> query_runtime_budget(const RuntimeReference& runtime) {
    std::lock_guard<std::mutex> lock(runtime_mutex);
    if (!current || runtime.generation_ != generation)
        return Error{ErrorCategory::runtime_ended, "Runtime has ended"};
    return RuntimeBudget{current->budget->limit(), current->budget_source};
}

Result<RuntimeCpuThreads> query_runtime_cpu_threads(const RuntimeReference& runtime) {
    std::lock_guard<std::mutex> lock(runtime_mutex);
    if (!current || runtime.generation_ != generation)
        return Error{ErrorCategory::runtime_ended, "Runtime has ended"};
    return current->threads;
}

Capabilities query_capabilities(bool probe, const LogReceiver& log_receiver) {
    std::lock_guard<std::mutex> lock(runtime_mutex);
    LogReceiver receiver = log_receiver;
    struct RestoreLog {
        bool active;
        ggml_log_callback callback = nullptr;
        void* data = nullptr;
        ~RestoreLog() { if (active) ggml_log_set(callback, data); }
    } restore{!current && bool(log_receiver)};
    if (restore.active) {
        ggml_log_get(&restore.callback, &restore.data);
        ggml_log_set(forward_log, &receiver);
    }
    Capabilities result;
    const auto defaults = platform::default_cpu_threads();
    result.cpu_threads = {defaults.decode, defaults.prefill, CpuThreadSource::platform_default, CpuThreadSource::platform_default};
    result.engine_version = MOE_VERSION;
    result.ggml_origin = MOE_GGML_COMMIT;
    result.ggml_version = MOE_GGML_VERSION;
    result.source_commit = MOE_BUILD_SOURCE_COMMIT;
    result.source_modified = MOE_BUILD_SOURCE_MODIFIED == 0 ? TriState::no
        : MOE_BUILD_SOURCE_MODIFIED == 1 ? TriState::yes : TriState::unknown;
    result.source_reason = MOE_BUILD_SOURCE_REASON;
    for (const auto& entry : app::compiled_models()) result.models.push_back(entry.adapter->id());
    auto compiled = platform::compiled_backends();
    const auto registered = compiled;
    auto devices = platform::devices();
    // Always report the two preset backend families, even when uncompiled.
    for (const auto& name : {std::string("cpu"), std::string("metal")})
        if (std::find(compiled.begin(), compiled.end(), name) == compiled.end()) compiled.push_back(name);
    std::sort(compiled.begin(), compiled.end());
    for (const auto& name : compiled) {
        const bool present = std::find(registered.begin(), registered.end(), name) != registered.end();
        BackendCapability backend{name, present ? TriState::yes : TriState::no,
            TriState::unknown, TriState::unknown, {}};
        if (probe && present) {
            backend.initializable = TriState::no;
            backend.reason = "No registered device";
            for (const auto& device : devices) {
                if (device.backend != name) continue;
                auto initialized = platform::probe(device.name);
                if (initialized.first) {
                    backend.initializable = TriState::yes;
                    backend.reason.clear();
                    break;
                }
                backend.reason = initialized.second;
            }
        }
        result.backends.push_back(std::move(backend));
    }
    for (const auto& device : devices)
        result.devices.push_back({device.backend, device.name, device.description,
            device.memory_total_bytes, device.memory_free_bytes});
    return result;
}
}
