// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/dsv41/vendor/moe_win.cu - the window's expert grouping kernel.
// Extracted from Project Maya's src/kernels/cuda/glm_fast.cu (MIT, see third_party/project-maya/NOTICE.md).
#include "../ds4_launch.h"   // strata_ds4_launch: <<<>>> on CUDA and HIP, a dispatch on Metal
#include "strata/dsv41/moe_win.hpp"
#include <cstdio>
#include <cstdlib>

namespace strata::dsv41k::glmf {
namespace {
constexpr int kWinEnt = kMaxWinTokens * 8;

// the entries grouped by expert slot (in order of first appearance), each group's members in entry order (t * 8 + i)
// (the tables in shared memory: as locals they were a 2.6 KB stack a thread, which the driver reserves for every thread
// the device can hold - ~420 MB on a V100 - and a launch failed "out of memory" once the expert pool had the rest)
__global__ void moe_win_group_kernel(const __grid_constant__ MoeWin w) {
    __shared__ unsigned long long gp[kWinEnt];
    __shared__ int cnt[kWinEnt], gof[kWinEnt], fill[kWinEnt + 1];
    if (threadIdx.x != 0) return;
    int ng = 0;
    for (int t = 0; t < w.T; ++t)
        for (int i = 0; i < 8; ++i) {
            const int e = t * 8 + i;
            gof[e] = -1;
            const unsigned long long p = i < w.k ? w.plan_ptr[e] : 0ull;
            if (p == 0ull) continue;
            int g = 0;
            while (g < ng && gp[g] != p) ++g;
            if (g == ng) {
                gp[ng] = p;
                cnt[ng++] = 0;
            }
            ++cnt[g];
            gof[e] = g;
        }
    fill[0] = 0;
    for (int g = 0; g < ng; ++g) {
        w.grp_ptr[g] = gp[g];
        w.grp_start[g] = fill[g];
        fill[g + 1] = fill[g] + cnt[g];
    }
    w.grp_start[ng] = fill[ng];
    for (int e = 0; e < w.T * 8; ++e)
        if (gof[e] >= 0) w.grp_ent[fill[gof[e]]++] = e;
    *w.n_groups = ng;
}
}  // namespace

void moe_win_group(const MoeWin& w, cudaStream_t s) {
    strata_ds4_launch(moe_win_group_kernel, 1, 32, 0, s, w);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "dsv41: moe_win_group: %s\n", cudaGetErrorString(e));
        std::abort();
    }
}
}  // namespace strata::dsv41k::glmf
