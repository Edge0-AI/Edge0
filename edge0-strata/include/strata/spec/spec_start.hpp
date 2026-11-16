// include/strata/spec/spec_start.hpp - where the speculative loop of `strata generate --spec` begins, and whether it runs.
//
// The CLI's prompt path leaves the position the speculative loop starts at in one variable.  Until 0.2.2 it was 0 for "the
// loop is not used", which is also the position a native pack starts the loop at when the prompt is a single token (its
// last prompt token is window 0 at position 0): such a prompt never speculated.  -1 is the "not used" value now.
#pragma once

#include <cstdint>

namespace strata::spec {

inline constexpr int64_t kSpecNotStarted = -1;

/// The loop runs when the prompt path named a start position (0 is a real one), tokens are still wanted and the text did
/// not already end.
inline constexpr bool spec_loop_runs(int64_t spec_pos, int64_t produced, int64_t max_new, bool ended) {
    return spec_pos >= 0 && produced < max_new && !ended;
}

}  // namespace strata::spec
