#pragma once
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace strata::core {
// Native protocol: logit_bias=id:value,id:value. -100 is a hard exclusion, not a finite penalty.
inline bool parse_logit_bias(const std::string& text, int vocab, std::vector<float>& out, std::string& error) {
    out.clear();
    if (text.empty() || vocab <= 0) { error = "empty logit_bias or invalid vocabulary"; return false; }
    std::vector<float> bias((size_t) vocab, 0.0f);
    std::vector<bool> seen((size_t) vocab, false);
    const char* p = text.c_str();
    int banned = 0;
    for (;;) {
        if (*p < '0' || *p > '9') { error = "invalid logit_bias token ID"; return false; }
        char* end = nullptr;
        const unsigned long id = std::strtoul(p, &end, 10);
        if (*end != ':' || id >= (unsigned long) vocab || seen[id]) {
            error = "logit_bias ID outside vocabulary, duplicated, or missing ':'"; return false;
        }
        p = end + 1;
        const float value = std::strtof(p, &end);
        if (end == p || !std::isfinite(value) || value < -100 || value > 100 || (*end && *end != ',')) {
            error = "invalid logit_bias value (expected -100..100)"; return false;
        }
        seen[id] = true;
        bias[id] = value == -100 ? -std::numeric_limits<float>::infinity() : value;
        banned += value == -100;
        if (!*end) break;
        p = end + 1;
    }
    if (banned == vocab) { error = "logit_bias bans the entire vocabulary"; return false; }
    out = std::move(bias);
    return true;
}
} // namespace strata::core
