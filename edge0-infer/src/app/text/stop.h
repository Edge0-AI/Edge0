#pragma once
#include <optional>
#include <string>
#include <vector>
namespace edge0::text {
class StopMatcher {
public:
    explicit StopMatcher(std::vector<std::string> stops);
    std::string feed(const std::string&);
    std::string finish();
    const std::optional<std::string>& matched() const {return matched_;}
private:
    std::vector<std::string> stops_;
    std::vector<std::vector<std::size_t>> failure_;
    std::vector<std::size_t> lengths_;
    std::string pending_;
    std::optional<std::string> matched_;
};
}
