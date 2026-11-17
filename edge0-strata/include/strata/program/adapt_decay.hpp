#pragma once
// --adapt-decay: the factor the adaptive tier's usage counts are multiplied by after each adaptation (#1284).
// Inside 0..1 it forgets old routing; above 1 the counts grow without end (inf after a few hundred adaptations) and the
// swap gains turn NaN, which breaks the sort that orders the swaps.  An unusable value is not refused: the engine says
// so and runs with the default.
#include <cmath>
#include <cstdio>

namespace strata::program {

constexpr float kAdaptDecayDefault = 0.7f;

/// `v` itself when it is finite and in 0..1 (0 forgets everything at once, 1 never forgets: both work), else the default
/// with a line on stderr.
inline float adapt_decay_checked(double v, std::FILE* log = stderr) {
    if (std::isfinite(v) && v >= 0.0 && v <= 1.0) return (float) v;
    if (log != nullptr)
        std::fprintf(log, "strata: --adapt-decay %g is not between 0 and 1; using %g (counts above 1 grow without end and the swap gains turn NaN)\n",
                     v, (double) kAdaptDecayDefault);
    return kAdaptDecayDefault;
}

}  // namespace strata::program
