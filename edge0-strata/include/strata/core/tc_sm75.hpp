// include/strata/core/tc_sm75.hpp - sm_75 (RTX 20 / Turing) tensor-core paths.
//
// Turing has no tf32 MMA and no cp.async, so three of this engine's prompt-path kernels are compiled out below
// sm_80 and fall back to slower paths (see `qsa_prompt_attn.cu`, `qsa_select.cu`).  It does have fp16 MMA
// (m16n8k8; m16n8k16 is sm_80+) and cuBLAS has no bf16 GEMM kernel below sm_80 - it runs an FFMA one, which
// measured 19.5% of the prompt's GPU time on a 2x2080 Ti.  These switches select the Turing paths.
//
// **ONLY sm_75 IS EVER AFFECTED.**  The predicate below answers true for compute capability 7.5 and for nothing
// else: cc 7.0/7.2 (Volta) keeps its own behaviour, and every cc >= 8 keeps the paths the engine was written
// with.  A machine that is not a 2080 Ti cannot be moved onto them by an environment variable.
// The kernels themselves are gated a second time, at compile time, by `__CUDA_ARCH__ == 750`.
//
// 0.1.45: the Turing paths are opt-in (nobody here has an sm_75 card to confirm them): STRATA_SM75_PERF=1
// (sm75_series.hpp) turns them on, STRATA_TC_SM75=0 / 1 forces this one off / on (still only on a 7.5 card).
#pragma once

#include "strata/core/sm75_series.hpp"

namespace strata::core {

inline bool tc_sm75_enabled(int cc_major, int cc_minor) {
    return strata::sm75_switch("STRATA_TC_SM75", cc_major, cc_minor);   // 7.5 and only 7.5 - never another SM model
}

}  // namespace strata::core
