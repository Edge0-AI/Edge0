#include "app/text/stop.h"
#include <algorithm>
#include <limits>
namespace edge0::text {
std::string StopMatcher::feed(const std::string& text) {
    if(matched_) return {};
    if(stops_.empty()) return text;
    pending_+=text;
    std::size_t best_end=std::numeric_limits<std::size_t>::max(),best_start=best_end;
    for(const auto& stop:stops_) {
        const auto start=pending_.find(stop);
        if(start==std::string::npos) continue;
        const auto end=start+stop.size();
        if(end<best_end || (end==best_end && start<best_start)) {
            best_end=end;best_start=start;matched_=stop;
        }
    }
    if(matched_) {
        auto result=pending_.substr(0,best_start);pending_.clear();return result;
    }
    std::size_t hold=0;
    for(const auto& stop:stops_)
        for(std::size_t size=1;size<stop.size() && size<=pending_.size();++size)
            if(pending_.compare(pending_.size()-size,size,stop,0,size)==0) hold=std::max(hold,size);
    auto result=pending_.substr(0,pending_.size()-hold);
    pending_.erase(0,result.size());return result;
}
std::string StopMatcher::finish() {
    if(matched_) return {};
    auto result=std::move(pending_);pending_.clear();return result;
}
}
