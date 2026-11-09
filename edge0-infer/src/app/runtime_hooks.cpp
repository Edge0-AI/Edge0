#include "app/runtime_hooks.h"
namespace edge0::app {
Result<std::shared_ptr<RuntimeHooks>> runtime_hooks(const RuntimeReference& runtime) {
    auto resources = RuntimeAccess::acquire(runtime); if (!resources) return resources.error();
    auto registry = resources.value().registry;
    std::lock_guard<std::mutex> lock(registry->mutex);
    if (registry->active || !registry->models.empty())
        return Error{ErrorCategory::conflict, "Install runtime hooks before loading models"};
    if (!registry->hooks) registry->hooks = std::make_shared<RuntimeHooks>();
    return registry->hooks;
}
exec::Status add_backend_alias(const RuntimeReference& runtime, const std::string& alias, const std::string& native_device) {
    auto resources = RuntimeAccess::acquire(runtime); if (!resources) return resources.error();
    if (alias.empty() || resources.value().backends.count(alias))
        return Error{ErrorCategory::invalid_argument, "Backend alias is empty or already registered"};
    auto hooks = runtime_hooks(runtime); if (!hooks) return hooks.error();
    auto facility = platform::BackendFacility::create(native_device);
    if (!facility.first) return Error{ErrorCategory::unsupported, facility.second};
    auto owner = std::make_shared<platform::BackendFacility>(std::move(facility.first));
    auto stages = exec::resolve_stages(native_device, native_device, owner);
    if (!stages) return stages.error();
    hooks.value()->aliases.emplace(alias, stages.value().prefill);
    return {};
}
}
