#pragma once
#include "model/adapter.h"
#include "models/api/compute.h"
namespace edge0::app {
struct CompiledModel {
    const model::Adapter* adapter;
    const models::ModelCompute* compute;
};
std::vector<CompiledModel> compiled_models();
model::AdapterRegistry compiled_adapters();
Result<CompiledModel> select_compiled(const model::Catalog& catalog);
model::ModelOptions apply_reference_values(const std::vector<model::ConfigurationItem>& items,
    model::ModelOptions options);
}
