#pragma once

#include <charconv>
#include <cstdint>
#include <string_view>

namespace strata::program::logits_selection {

// Storage selection only: callers must still condition on every input token.
// Positive decimal int64, with no sign, whitespace, suffix or overflow.
inline bool parse_stride(std::string_view text, int64_t& result) {
    if (text.empty() || text.front() < '0' || text.front() > '9') return false;
    int64_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || value <= 0) return false;
    result = value;
    return true;
}

// Zero-based multiples of stride, followed by the final input position once.
inline constexpr bool selected(int64_t position, int64_t total, int64_t stride) {
    return total > 0 && stride > 0 && position >= 0 && position < total &&
           (position % stride == 0 || position == total - 1);
}

inline constexpr int64_t row_count(int64_t total, int64_t stride) {
    return total > 0 && stride > 0 ? (total - 1) / stride + 1 + ((total - 1) % stride != 0) : 0;
}

} // namespace strata::program::logits_selection
