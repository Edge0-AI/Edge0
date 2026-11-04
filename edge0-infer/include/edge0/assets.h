#pragma once
#include "edge0/result.h"
#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>
namespace edge0 {
struct InspectOptions {
    std::string directory;
    bool digest = false;
};
struct InspectedFile {
    std::string role, filename;
    std::uint64_t bytes;
    bool digest_verified;
};
struct InspectedTensor {
    std::string role, name, type;
    std::array<std::int64_t, 4> shape;
};
struct ExpertSlice {
    std::uint32_t layer, expert;
    std::string projection, file, type;
    std::uint64_t offset, length;
};
struct AssetReport {
    std::string architecture, adapter, gguf_architecture, integrity;
    std::vector<InspectedFile> files;
    std::vector<std::string> unused_roles;
    std::map<std::string, std::uint64_t> parameters, tensor_counts;
    std::map<std::string, double> numeric_parameters;
    std::map<std::string, std::vector<std::uint32_t>> layer_groups;
    std::uint64_t layers = 0, experts = 0, experts_per_token = 0;
    std::vector<ExpertSlice> expert_slices;
    std::vector<InspectedTensor> tensor_requirements;
};
// Developer inspection only; application model loading uses the engine API.
// Does not open a runtime or initialize a backend. Default mode reads headers
// only; digest mode streams complete files without loading backend weights.
Result<AssetReport> inspect_assets(const InspectOptions &options);
} // namespace edge0
