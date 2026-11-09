#pragma once
#include "edge0/generation.h"

namespace example_chat {
inline edge0::GenerationOverrides generation_overrides(bool thinking, std::optional<std::int64_t> max_tokens, bool token_events, double temperature = 0.2) {
    edge0::GenerationOverrides overrides;
    overrides.thinking = thinking;
    overrides.temperature = temperature;
    if (max_tokens) overrides.max_tokens = *max_tokens;
    if (token_events) overrides.logprobs = true;
    return overrides;
}
}
