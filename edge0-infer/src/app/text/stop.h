#pragma once
#include <optional>
#include <string>
#include <vector>
namespace edge0::text {
class StopMatcher {
public:
    explicit StopMatcher(std::vector<std::string> stops):stops_(std::move(stops)) {}
    std::string feed(const std::string&);
    std::string finish();
    const std::optional<std::string>& matched() const {return matched_;}
private:
    std::vector<std::string> stops_;
    std::string pending_;
    std::optional<std::string> matched_;
};
}
