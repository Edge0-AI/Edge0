#include "app/adapters.h"
#include "models/edge0_35b/assets.h"
#include "models/edge0_35b/compute.h"
namespace edge0::app {
std::vector<CompiledModel> compiled_models() {
    return {{&models::edge0_35b::adapter(), &models::edge0_35b::compute()}};
}
model::AdapterRegistry compiled_adapters() {
    model::AdapterRegistry adapters;
    for (const auto& entry : compiled_models()) adapters.push_back(entry.adapter);
    return adapters;
}
Result<CompiledModel> select_compiled(const model::Catalog& catalog) {
    auto selected = model::select_adapter(catalog, compiled_adapters());
    if (!selected) return selected.error();
    for (const auto& entry : compiled_models())
        if (entry.adapter == selected.value()) return entry;
    return Error{ErrorCategory::unsupported, "Selected adapter has no compute registration"};
}
model::ModelOptions apply_reference_values(const std::vector<model::ConfigurationItem>& items,
    model::ModelOptions options) {
    for (const auto& item : items)
        if (item.reference_value) options.emplace(item.name, *item.reference_value);
    return options;
}
} // namespace edge0::app
