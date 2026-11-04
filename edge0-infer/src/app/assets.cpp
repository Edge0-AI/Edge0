#include "edge0/assets.h"
#include "app/adapters.h"
namespace edge0 {
Result<AssetReport> inspect_assets(const InspectOptions &options) {
    auto catalog = model::open_catalog(options.directory, options.digest ? model::Integrity::digest
                                                                         : model::Integrity::size_only);
    if (!catalog)
        return catalog.error();
    auto adapter = model::select_adapter(catalog.value(), app::compiled_adapters());
    if (!adapter)
        return adapter.error();
    auto compatible = model::check_compatibility(catalog.value(), *adapter.value());
    if (!compatible)
        return compatible.error();
    const auto &c = compatible.value();
    const auto &d = c.definition;
    AssetReport report;
    report.architecture = catalog.value().architecture;
    report.adapter = adapter.value()->id();
    report.integrity = options.digest ? "digest" : "size_only";
    auto arch = model::read_metadata<std::string>(c.indexes.at("base").metadata, "general.architecture");
    if (!arch)
        return arch.error();
    report.gguf_architecture = *arch.value();
    for (const auto &entry : catalog.value().files) {
        const auto &f = entry.second;
        report.files.push_back({f.role, f.filename, f.bytes, f.digest_verified});
    }
    report.unused_roles = c.unused_roles;
    report.parameters = d.parameters;
    report.numeric_parameters = d.numeric_parameters;
    report.layer_groups = d.layer_groups;
    report.layers = d.layer_count;
    report.experts = d.expert_count;
    report.experts_per_token = d.experts_per_token;
    for (const auto &entry : c.indexes)
        report.tensor_counts[entry.first] = entry.second.tensors.size();
    for (const auto &req : d.tensors)
        report.tensor_requirements.push_back({req.role, req.name, req.type, req.shape});
    for (const auto &expert : d.expert_tensors) {
        const auto &t = c.indexes.at("base").tensors.at(expert.name);
        for (std::uint32_t e = 0; e < expert.experts; ++e) {
            auto slice = model::locate_slice(t, e, 1);
            if (!slice)
                return slice.error();
            report.expert_slices.push_back({expert.layer, e, expert.projection,
                                            catalog.value().files.at("base").filename, t.type,
                                            slice.value().offset, slice.value().bytes});
        }
    }
    return report;
}
} // namespace edge0
