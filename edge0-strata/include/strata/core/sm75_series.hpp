// strata/core/sm75_series.hpp - the Turing (RTX 20, compute capability 7.5) kernel series: one opt-in switch.
//
// The Turing kernels of PRs #2069-#2072, #2081, #2091-#2093, #2145-#2146, #2220-#2223, #1821 were measured on one RTX
// 2080 Ti (and a CMP 50HX) and are bitwise equal to the kernels they replace.  We have no sm_75 card to confirm them, so
// they are ON ONLY WHEN ASKED FOR:
//
//   STRATA_SM75_PERF=1   turns on every bitwise Turing kernel of the series on a compute capability 7.5 card.
//
// Each kernel keeps the author's own switch, which wins over STRATA_SM75_PERF in both directions (=0 off, =1 on, on any
// card where the PR said so for tests).  Nothing here changes a card that is not a 7.5: the check is the card's compute
// capability (STRATA_EMULATE_CC=75 counts, so a 75-virtual build on an sm_86 card runs the series for tests).
//
// The numerics-changing Turing options are NOT part of this switch and keep their own explicit opt-ins:
//   STRATA_SELECT_SIMT=1|2 (FP32 tiled block scorer), STRATA_PA_V2T=1 (v2 prompt attention), STRATA_PF_FUSED_SM75=1
//   (fused int8 prompt experts), STRATA_BF16_TC=1 (FP16 tensor-core projections).
#pragma once

#include "strata/core/emulate.hpp"

#include <cstdlib>

namespace strata {

/// STRATA_SM75_PERF=1 (read once)
inline bool sm75_series() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_SM75_PERF");
        return e != nullptr && std::atoi(e) != 0;
    }();
    return on;
}

/// the device (raw cudaDevAttrComputeCapabilityMajor / Minor) is a Turing card, or emulates one (STRATA_EMULATE_CC=75)
inline bool is_sm75(int major, int minor) { return cc_major_of(major) == 7 && cc_minor_of(minor) == 5; }

/// the default of a bitwise Turing kernel switch: on for a Turing card when STRATA_SM75_PERF=1
inline bool sm75_default(int major, int minor) { return is_sm75(major, minor) && sm75_series(); }

/// a Turing switch with its own variable: `name` set means 0 off / anything else on (Turing only); unset follows the series
inline bool sm75_switch(const char* name, int major, int minor) {
    if (!is_sm75(major, minor)) return false;
    const char* v = std::getenv(name);
    return v != nullptr ? std::atoi(v) != 0 : sm75_series();
}

}  // namespace strata
