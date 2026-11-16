// src/prefill/lat_q3k_swiglu.cu - see include/strata/prefill/lat_q3k_swiglu.hpp.
//
// M2b of the q3k latent A stage: SwiGLU and the q8_1 of H, the pass moe_fused.cu's latent_kernel<true> does in its
// epilogue, here on the flat gate/up rows M2a writes.  A warp is one 32-value block of one entry, so the block's amax
// is one xor reduction over the warp; the code position and the block's d are q8_1_store's (iq_kernels.cu:2865: the
// amax, `q8_1_finite(amax / 127)`, `q8_1_quant`), i.e. the engine's own quantizer bit for bit.
#include "strata/prefill/lat_q3k_swiglu.hpp"
#include "strata/kernels/q8_1_finite.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::prefill {
namespace {

void dck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "lat q3k swiglu: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

constexpr int kFF = 640;            // features per entry
constexpr int kBlocks = kFF / 32;   // 20 q8_1 blocks per entry
constexpr int kHqRow = 800;         // 10 chunks x 80 B
constexpr int kThreads = 256;       // 8 warps

__global__ void __launch_bounds__(kThreads)
lat_q3k_swiglu_kernel(const float* __restrict__ gate, const float* __restrict__ up, int64_t rows,
                      uint8_t* __restrict__ hq, float* __restrict__ hout) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const long long i = (long long) blockIdx.x * (kThreads / 32) + wid;   // one warp per (entry, 32-value block)
    if (i >= rows * kBlocks) return;
    const long long e = i / kBlocks;
    const int b = (int) (i % kBlocks);
    const int v = lane;
    const int64_t at = e * kFF + 32 * b + v;
    const float g = gate[at], u = up[at];
    // the engine's product, rounded on its own (__fmul_rn), as moe_fused.cu's latent epilogue and swiglu_q8_1_entries
    const float h = __fmul_rn(__fdiv_rn(g, 1.0f + __expf(-g)), u);
    if (hout != nullptr) hout[at] = h;
    float am = fabsf(h);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    const float d = strata::kernels::q8_1_finite(__fdiv_rn(am, 127.0f));
    const int8_t qv = strata::kernels::q8_1_quant(h, d, am);
    uint8_t* row = hq + (size_t) e * kHqRow + (size_t) (b >> 1) * 80;
    // q8_1_store's permuted position: sub-block 2 * (b & 1) + (v / 16), byte 4 * (v & 3) + ((v & 15) / 4)
    row[16 * (2 * (b & 1) + (v >> 4)) + 4 * (v & 3) + ((v & 15) >> 2)] = (uint8_t) qv;
    if (lane == 0) *(__half*) (row + 64 + 2 * (b & 1)) = __float2half(d);
#else
    (void) gate; (void) up; (void) rows; (void) hq; (void) hout;
#endif
}

}  // namespace

void lat_q3k_swiglu_q8(const float* gate, const float* up, int64_t rows, void* hq, float* hout, void* stream) {
    if (rows <= 0 || gate == nullptr || up == nullptr || hq == nullptr) return;
#if defined(__HIPCC__)
    std::fprintf(stderr, "lat q3k swiglu: the latent A stage's kernels are CUDA-only\n");
    std::exit(1);
#else
    const long long n = rows * kBlocks;                  // one warp a block
    const long long blocks = (n + (kThreads / 32) - 1) / (kThreads / 32);
    lat_q3k_swiglu_kernel<<<(unsigned) blocks, kThreads, 0, (cudaStream_t) stream>>>(gate, up, rows, (uint8_t*) hq,
                                                                                    hout);
    dck(cudaGetLastError(), "launch");
#endif
}

}  // namespace strata::prefill
