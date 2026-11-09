#pragma once
#include "edge0/result.h"
#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>
namespace edge0::model {
struct GenerationDefaults {
    std::optional<double> temperature, top_p, repetition_penalty;
    std::optional<std::int64_t> top_k;
};
Result<GenerationDefaults> read_generation_defaults(const std::string& directory);
enum class Integrity { size_only, digest };
struct AssetFile {
    std::string role, filename, path;
    std::uint64_t bytes = 0;
    std::string sha256;
    bool digest_verified = false;
};
struct Catalog {
    std::string architecture;
    Integrity integrity = Integrity::size_only;
    std::map<std::string, AssetFile> files;
};
using MetadataValue =
    std::variant<std::uint8_t, std::int8_t, std::uint16_t, std::int16_t, std::uint32_t, std::int32_t, float,
                 bool, std::string, std::uint64_t, std::int64_t, double, std::vector<std::uint8_t>,
                 std::vector<std::int8_t>, std::vector<std::uint16_t>, std::vector<std::int16_t>,
                 std::vector<std::uint32_t>, std::vector<std::int32_t>, std::vector<float>, std::vector<bool>,
                 std::vector<std::string>, std::vector<std::uint64_t>, std::vector<std::int64_t>,
                 std::vector<double>>;
using Metadata = std::map<std::string, MetadataValue>;
struct Tensor {
    std::string name, role, path, type;
    std::array<std::int64_t, 4> ne{{1, 1, 1, 1}};
    std::array<std::uint64_t, 4> nb{};
    std::uint64_t offset = 0, bytes = 0;
    std::uint64_t block_size = 1;
};
struct TensorRequirement {
    std::string role, name, type;
    std::array<std::int64_t, 4> shape{{1, 1, 1, 1}};
};
struct TensorIndex {
    AssetFile file;
    Metadata metadata;
    std::map<std::string, Tensor> tensors;
};
struct Slice {
    std::string path, type;
    std::uint64_t offset = 0, bytes = 0;
};
Result<Catalog> open_catalog(const std::string &directory, Integrity integrity = Integrity::size_only);
Result<TensorIndex> index_tensors(const AssetFile &file);
template <class T> Result<const T *> read_metadata(const Metadata &metadata, const std::string &key) {
    auto found = metadata.find(key);
    if (found == metadata.end())
        return Error{ErrorCategory::incompatible, key + ": missing metadata key"};
    auto value = std::get_if<T>(&found->second);
    if (!value)
        return Error{ErrorCategory::incompatible, key + ": metadata type mismatch"};
    return value;
}
Result<const Tensor *> find_tensor(const TensorIndex &index, const TensorRequirement &required);
Result<Slice> locate_slice(const Tensor &tensor, std::uint64_t first, std::uint64_t count);
} // namespace edge0::model
