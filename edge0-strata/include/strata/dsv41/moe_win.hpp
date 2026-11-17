// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// include/strata/dsv41/moe_win.hpp - a verify window's MoE entries grouped by expert (the grouping alone).
// Extracted from Project Maya's include/strata/kernels/glm_fast.hpp (MIT, see third_party/project-maya/NOTICE.md).
#pragma once
#include <cstddef>
#include <cuda_runtime.h>

namespace strata::dsv41k::glmf {
constexpr int kMaxMvCols = 16;
struct MoeWin {
    const unsigned long long* plan_ptr = nullptr;   // [T][8] the tokens' plans (0: an entry the GPU does not compute)
    const float* plan_w = nullptr;                  // [T][8]
    const float* cpu_part = nullptr;                // [T][n_embd] the host's answers to disk misses, when flagged
    const int* cpu_flag = nullptr;                  // [T]
    const void* xq = nullptr;                       // the tokens' FFN inputs (q8_1), xq_stride bytes apart
    size_t xq_stride = 0;
    void* hq = nullptr;                             // scratch: [T * 8][n_ff / 32] q8_1
    float* part = nullptr;                          // scratch: [T * 8][n_embd]
    const float* sh_out = nullptr;                  // [T][n_embd] the shared expert's down rows, or null
    float* out = nullptr;                           // [T][n_embd], out_stride floats apart
    size_t out_stride = 0;
    unsigned long long* grp_ptr = nullptr;          // scratch: [T * 8]
    int* grp_start = nullptr;                       // scratch: [T * 8 + 1]
    int* grp_ent = nullptr;                         // scratch: [T * 8]
    int* n_groups = nullptr;                        // scratch: [1]
    int T = 0, k = 0, n_embd = 0, n_ff = 0;
    float limit = 0.0f;
    size_t down_off = 0;
};
constexpr int kMaxWinTokens = kMaxMvCols;   // the window's tokens moe_win_experts groups (16 x 8 entries)
void moe_win_group(const MoeWin& w, cudaStream_t s);
}  // namespace strata::dsv41k::glmf
