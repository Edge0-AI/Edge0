#include "app/text/history.h"
#include <algorithm>
namespace edge0::text {
std::size_t match_history(const std::vector<std::int32_t>& target, const session::Snapshot& cached, const session::Identity& identity) {
    if (!(cached.identity == identity)) return 0;
    std::size_t p = 0;
    while (p < target.size() && p < cached.tokens.size() && target[p] == cached.tokens[p]) ++p;
    return p;
}
Result<std::vector<std::int32_t>> prepare_suffix(const std::vector<std::int32_t>& input, std::size_t k) {
    if (k >= input.size()) return Error{ErrorCategory::invalid_argument, "Empty or invalid input suffix"};
    return std::vector<std::int32_t>(input.begin() + k, input.end());
}
FinishedTurn finish_turn(Decoder& decoder, std::size_t input_count, const std::vector<std::int32_t>& accepted,
    std::int64_t committed, bool ended, bool flush) {
    FinishedTurn result;
    if (flush) result.text = decoder.finish();
    if (!accepted.empty() && committed >= 0 && static_cast<std::uint64_t>(committed) >= input_count &&
        static_cast<std::uint64_t>(committed) - input_count == accepted.size() - 1) {
        result.tail = accepted.back(); result.tail_is_end = ended;
    }
    return result;
}
}
