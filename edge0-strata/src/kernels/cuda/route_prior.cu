// src/kernels/cuda/route_prior.cu - STRATA_ROUTE_PRIOR=lambda: cache-conditioned routing (opt-in, changes the output).
//
// "Mixture of Cache-Conditional Experts" (arXiv 2412.00099), without training: when a token's 10 experts are picked,
// an expert already in VRAM gets a `lambda` bonus on its logit.  The SELECTION uses the biased logits; the WEIGHTS stay
// the router's (softmax over the 512 unbiased logits), renormalised over the 10 picked like the native router.  Fewer
// missed experts to copy over PCIe or compute on the CPU.  Unlike STRATA_ROUTE_RESIDENT (a missed tail expert swapped
// for the best resident one within a logit margin), every rank can move, by an amount the logits decide.
#include "strata/kernels/route_prior.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>

namespace strata::kernels {
namespace {

__device__ __forceinline__ float wmax(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m, 32));
    return v;
}
__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m, 32);
    return v;
}

// one warp per token; 512 experts = 16 values per lane (as the native router)
__global__ void route_prior_k(const float* __restrict__ logits, const int32_t* __restrict__ res, float lambda,
                                  int32_t* __restrict__ ids, float* __restrict__ weights) {
    const float* lg = logits + (size_t) blockIdx.x * 512;
    int32_t* id = ids + (size_t) blockIdx.x * 10;
    float* w = weights + (size_t) blockIdx.x * 10;
    const int lane = threadIdx.x;
    float p[16], s[16];
    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) { p[i] = lg[lane + 32 * i]; mx = fmaxf(mx, p[i]); }
    mx = wmax(mx);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int e = lane + 32 * i;
        s[i] = p[i] + (res[e] >= 0 ? lambda : 0.0f);      // the selection score
        p[i] = expf(p[i] - mx);                            // the router's probability (unnormalised)
        sum += p[i];
    }
    (void) wsum(sum);
    float sel_p = 0.0f, sel_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = -FLT_MAX, bp = 0.0f;
        int be = 1 << 30;
#pragma unroll
        for (int i = 0; i < 16; ++i)
            if (s[i] > best) { best = s[i]; be = lane + 32 * i; bp = p[i]; }
#pragma unroll
        for (int m = 16; m; m >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, m, 32);
            const int oe = __shfl_xor_sync(0xffffffffu, be, m, 32);
            const float op = __shfl_xor_sync(0xffffffffu, bp, m, 32);
            if (ob > best || (ob == best && oe < be)) { best = ob; be = oe; bp = op; }
        }
        if ((be & 31) == lane) s[be >> 5] = -FLT_MAX;
        if (lane == rank) { id[rank] = be; sel_p = bp; }
        sel_sum += bp;                                      // the same on every lane
    }
    if (lane < 10) w[lane] = sel_p / fmaxf(sel_sum, 1e-30f);
}

}  // namespace

void route_prior_top10(const float* logits, const int32_t* res_layer, float lambda, int32_t* ids, float* weights,
                             int n_tok, void* stream) {
    if (n_tok <= 0) return;
    route_prior_k<<<n_tok, 32, 0, (cudaStream_t) stream>>>(logits, res_layer, lambda, ids, weights);
}

}  // namespace strata::kernels
