#include "model/adapter.h"
#include "model/sha256.h"
#include <algorithm>
#include <set>
#include <limits>
namespace edge0::model {
Result<std::uint64_t> parse_count(const std::string& text) {
    std::uint64_t value=0;
    if (text.empty()) return Error{ErrorCategory::invalid_argument, "Expected a decimal count"};
    for (char digit : text) {
        if (digit<'0' || digit>'9' || value>(std::numeric_limits<std::uint64_t>::max()-9)/10)
            return Error{ErrorCategory::invalid_argument, "Invalid decimal count"};
        value=value*10+std::uint64_t(digit-'0');
    }
    return value;
}
Result<const Adapter *> select_adapter(const Catalog &catalog, const AdapterRegistry &registry) {
    std::string known;
    for (auto *adapter : registry) {
        if (adapter->id() == catalog.architecture)
            return adapter;
        known += " " + adapter->id();
    }
    return Error{ErrorCategory::unsupported,
                 "architecture " + catalog.architecture + "; compiled adapters:" + known};
}
Result<CompatibleAssets> check_compatibility(const Catalog &catalog, const Adapter &adapter) {
    CompatibleAssets out;
    for (const auto &role : adapter.gguf_roles) {
        auto file = catalog.files.find(role);
        if (file == catalog.files.end())
            return Error{ErrorCategory::incompatible, "missing required role " + role};
        auto index = index_tensors(file->second);
        if (!index)
            return index.error();
        out.indexes.emplace(role, index.value());
    }
    auto definition = adapter.describe_assets(out.indexes);
    if (!definition)
        return definition.error();
    out.definition = definition.value();
    for (const auto& field : {std::make_pair("vocabulary_size", out.definition.vocabulary_size),
            {"context_length", out.definition.context_length}, {"layer_count", out.definition.layer_count},
            {"expert_count", out.definition.expert_count}, {"experts_per_token", out.definition.experts_per_token}})
        if (!field.second) return Error{ErrorCategory::incompatible, std::string("Missing positive ") + field.first};
    for (const auto &role : out.definition.required_roles)
        if (!catalog.files.count(role))
            return Error{ErrorCategory::incompatible, "missing required role " + role};
    const auto text = adapter.describe_text();
    {
        auto file = catalog.files.find("chat_template");
        if (file == catalog.files.end())
            return Error{ErrorCategory::incompatible, "missing required role chat_template"};
        auto digest = sha256_file(file->second.path);
        if (!digest)
            return digest.error();
        if (digest.value() != text.template_sha256)
            return Error{ErrorCategory::incompatible, "template identity mismatch: expected " +
                         text.template_sha256 + "; actual " + digest.value()};
        const auto base = out.indexes.find("base");
        if (base == out.indexes.end())
            return Error{ErrorCategory::incompatible, "missing GGUF role base"};
        for (const auto &entry : {std::make_pair("tokenizer.ggml.model", text.model),
                                  std::make_pair("tokenizer.ggml.pre", text.pre)}) {
            auto actual = read_metadata<std::string>(base->second.metadata, entry.first);
            if (!actual)
                return actual.error();
            if (*actual.value() != entry.second)
                return Error{ErrorCategory::incompatible, std::string(entry.first) + ": expected " +
                             entry.second + "; actual " + *actual.value()};
        }
    }
    std::map<std::string, std::set<std::string>> expected;
    for (const auto &req : out.definition.tensors) {
        auto index = out.indexes.find(req.role);
        if (index == out.indexes.end())
            return Error{ErrorCategory::incompatible, "missing GGUF role " + req.role};
        auto t = find_tensor(index->second, req);
        if (!t)
            return t.error();
        expected[req.role].insert(req.name);
    }
    for (const auto &role : out.indexes)
        for (const auto &t : role.second.tensors)
            if (!expected[role.first].count(t.first))
                return Error{ErrorCategory::incompatible,
                             role.first + "/" + t.first + ": expected absent; actual extra tensor"};
    for (const auto &role : catalog.files)
        if (std::find(out.definition.required_roles.begin(), out.definition.required_roles.end(),
                      role.first) == out.definition.required_roles.end())
            out.unused_roles.push_back(role.first);
    return out;
}
} // namespace edge0::model
