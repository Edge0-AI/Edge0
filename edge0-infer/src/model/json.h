#pragma once
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace edge0::model {
// Objects preserve insertion order. Numbers retain their JSON spelling until serialization.
struct Json {
    enum class Kind { null, boolean, number, string, array, object };
    Kind kind = Kind::null;
    std::string value;
    std::vector<Json> elements;
    std::vector<std::pair<std::string, Json>> members;
    const Json* find(std::string_view key) const;
    static Json parse(std::string_view);
    std::string dump() const;
};
std::string json_quote(std::string_view);
} // namespace edge0::model
