// Strata. Which way the DeepSeek V4.1 engine reads a prompt: the chunked pass (the "flow": each chunk goes through every layer at
// once, the experts the GPUs do not hold streamed to the GPU once a chunk) or windows of 16 tokens (the CPU lane computes the experts
// it holds, for every window again).  The choice is the prompt's length against one number, the same on every box and in every session,
// so a prompt always takes the same path and gives the same bits (the two paths round differently).
//
// Why 96.  First prompt of a fresh process, the chunked pass warmed up (the BLAS library's start paid while the user types), r9700
// 2.00 pack, milliseconds windows / flow:  1 card  28: 1144 / 2252, 100: 3384 / 3922, 300: 9310 / 7281, 600: 17349 / 8165, 1500: 42469 / 10920;
// 4 cards  28: 585 / 922, 100: 1161 / 1264, 300: 2837 / 1923, 600: 5300 / 2675, 1500: 13353 / 4181 - the crossover is ~120-140 tokens
// there.  Boxes whose experts come mostly from an SSD or a slower CPU lane cross earlier (P2003 RTX 3060, no warm-up: 28: 3322 / 4063,
// 100: 8893 / 6817, 300: 26137 / 11417; the owner's RTX 5070 ~100; p620 100: 11839 / 7580), so a box-independent 96 costs the
// fast-CPU boxes at most 0.5 s on a 100-token prompt and saves the slow ones 2-4 s; 600 tokens and up take the flow everywhere.
#pragma once

#include <cstdlib>

namespace strata::ds4 {

constexpr int kPromptFlowMin = 96;   // tokens from which a prompt goes through the chunked pass

// env: DS4_PICK_MIN (null: the default)
inline int prompt_flow_min(const char* env) {
    if (env == nullptr || *env == 0) return kPromptFlowMin;
    const int v = std::atoi(env);
    return v < 1 ? 1 : v;
}

inline bool prompt_uses_flow(int tokens, int flow_min) { return tokens >= flow_min; }

}  // namespace strata::ds4
