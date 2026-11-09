#include "edge0/generation.h"
namespace cli {
std::vector<edge0::TextMessage> completed_history(const std::vector<edge0::TextMessage>& candidate,
    const std::vector<edge0::TextMessage>& previous, const edge0::GenerationResult& result) {
    const bool normal = result.stop_reason == edge0::StopReason::length || result.stop_reason == edge0::StopReason::end_of_generation ||
        result.stop_reason==edge0::StopReason::stop_string || result.stop_reason==edge0::StopReason::tool_calls;
    if (!result.state_valid || result.error || (!normal &&
        (result.stop_reason != edge0::StopReason::cancelled || result.text.empty()))) return previous;
    auto history = candidate; history.push_back({"assistant", result.text,result.tool_calls,result.reasoning}); return history;
}
}
