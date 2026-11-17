#pragma once
// STRATA_FUSE_DECODE=1: the operator fusions of PR #1097 (gate/up GEMV + SwiGLU in one kernel, SwiGLU + intermediate
// quantizer in one kernel, every activation image of x in one pass, two BF16 projections in one kernel).  They are
// compiled only into the experimental build (STRATA_EXPERIMENTAL_SM60, the CUDA 12 engine for Pascal / Volta) and are
// bitwise what the unfused sequences produce.  Opt-in: unset or 0 keeps the unfused kernels.
#include <cstdlib>

namespace strata::kernels {
inline bool fuse_decode() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_FUSE_DECODE");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}
}  // namespace strata::kernels
