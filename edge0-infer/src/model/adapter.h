#pragma once
#include "model/asset_index.h"
#include "model/text.h"
#include "weights/expert_slots.h"
#include <memory>
namespace edge0::model {
using AssetView = std::map<std::string, TensorIndex>;
struct ExpertTensor {
    std::uint32_t layer;
    std::string projection, name;
    std::uint64_t experts = 0;
};
struct ModelDefinition {
    std::vector<std::string> required_roles;
    std::vector<TensorRequirement> tensors;
    std::map<std::string, std::uint64_t> parameters;
    std::map<std::string, double> numeric_parameters;
    std::map<std::string, std::vector<std::uint32_t>> layer_groups;
    std::uint64_t vocabulary_size = 0, context_length = 0, layer_count = 0, expert_count = 0, experts_per_token = 0;
    std::vector<ExpertTensor> expert_tensors;
};
struct ConfigurationItem {
    std::string name;
    std::vector<std::string> values;
    std::string default_value;
    bool context_tokens = false;
    std::optional<std::string> reference_value = {};
};
Result<std::uint64_t> parse_count(const std::string& text);
using ModelOptions = std::map<std::string, std::string>;
struct ResidentWeight {
    std::string id, role, name;
    weights::Source source;
};
enum class StorageStages { prefill, decode, both };
struct WeightDescription {
    std::vector<ConfigurationItem> configuration;
    std::vector<ResidentWeight> resident;
    std::vector<weights::PoolDescription> pools;
    std::vector<weights::Array> temporary;
    std::size_t temporary_copies = 1;
    StorageStages resident_stages = StorageStages::both;
    StorageStages pool_stages = StorageStages::decode;
    StorageStages temporary_stages = StorageStages::prefill;
};
class Adapter {
  public:
    explicit Adapter(std::vector<std::string> roles) : gguf_roles(std::move(roles)) {}
    virtual ~Adapter() = default;
    virtual std::string id() const = 0;
    // Roles are declared before metadata can be inspected.
    const std::vector<std::string> gguf_roles;
    virtual TextDescription describe_text() const = 0;
    virtual TextResult text_format(const std::vector<Message> &, const TextOptions &) const = 0;
    virtual std::unique_ptr<OutputParser> output_parser(const TextOptions&) const {
        return std::make_unique<OutputParser>();
    }
    virtual Result<WeightDescription> describe_weight_storage(
        const AssetView&, const ModelOptions&) const {
        return Error{ErrorCategory::unsupported, "Weight storage declaration unavailable"};
    }
    virtual Result<ModelDefinition> describe_assets(const AssetView &assets) const = 0;
};
using AdapterRegistry = std::vector<const Adapter *>;
struct CompatibleAssets {
    ModelDefinition definition;
    AssetView indexes;
    std::vector<std::string> unused_roles;
};
Result<const Adapter *> select_adapter(const Catalog &catalog, const AdapterRegistry &registry);
Result<CompatibleAssets> check_compatibility(const Catalog &catalog, const Adapter &adapter);
} // namespace edge0::model
