#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>

namespace strata::core::conversation_detail {
inline bool add(size_t& total, size_t n) {
    if (n > std::numeric_limits<size_t>::max() - total) return false;
    total += n;
    return true;
}
inline bool product(size_t& out, std::initializer_list<uint64_t> factors) {
    size_t value = 1;
    for (uint64_t factor : factors) {
        if (factor > std::numeric_limits<size_t>::max() ||
            (factor && value > std::numeric_limits<size_t>::max() / factor)) return false;
        value *= (size_t) factor;
    }
    out = value;
    return true;
}
} // namespace strata::core::conversation_detail
