#pragma once
#include "app/text/decode.h"
#include "session/session.h"
namespace edge0::text {
std::size_t match_history(const std::vector<std::int32_t>& target, const session::Snapshot&, const session::Identity&);
Result<std::vector<std::int32_t>> prepare_suffix(const std::vector<std::int32_t>&, std::size_t k);
struct FinishedTurn {
    std::string text;
    std::optional<std::int32_t> tail;
    bool tail_is_end = false;
};
FinishedTurn finish_turn(Decoder&, std::size_t input_count, const std::vector<std::int32_t>& accepted,
    std::int64_t committed, bool ended, bool flush = true);
}
