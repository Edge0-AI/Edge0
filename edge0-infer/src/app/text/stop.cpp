#include "app/text/stop.h"
#include <algorithm>
namespace edge0::text {
StopMatcher::StopMatcher(std::vector<std::string> stops):stops_(std::move(stops)),lengths_(stops_.size()) {
    for (const auto& stop : stops_) {
        std::vector<std::size_t> failure(stop.size());
        for (std::size_t i = 1, length = 0; i < stop.size(); ++i) {
            while (length && stop[i] != stop[length]) length = failure[length - 1];
            if (stop[i] == stop[length]) ++length;
            failure[i] = length;
        }
        failure_.push_back(std::move(failure));
    }
}
std::string StopMatcher::feed(const std::string& text) {
    if(matched_) return {};
    if(stops_.empty()) return text;
    for (const auto& stop : stops_) if (stop.empty()) {
        matched_ = stop; pending_.clear(); return {};
    }
    const auto previous = pending_.size();
    pending_+=text;
    for (std::size_t at = 0; at < text.size(); ++at) {
        std::size_t longest = 0;
        for (std::size_t i = 0; i < stops_.size(); ++i) {
            const auto& stop = stops_[i];
            auto& length = lengths_[i];
            while (length && text[at] != stop[length]) length = failure_[i][length - 1];
            if (text[at] == stop[length]) ++length;
            // The first completing byte wins. At that byte the longest match
            // starts earliest, preserving the existing overlap rule.
            if (length == stop.size() && length > longest) {
                longest = length; matched_ = stop;
            }
        }
        if (matched_) {
            auto result = pending_.substr(0, previous + at + 1 - longest);
            pending_.clear(); return result;
        }
    }
    std::size_t hold=0;
    for (const auto length : lengths_) hold = std::max(hold, length);
    auto result=pending_.substr(0,pending_.size()-hold);
    pending_.erase(0,result.size());return result;
}
std::string StopMatcher::finish() {
    if(matched_) return {};
    std::fill(lengths_.begin(), lengths_.end(), 0);
    auto result=std::move(pending_);pending_.clear();return result;
}
}
