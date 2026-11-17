// src/prefill/kernels.cu - see include/strata/prefill/kernels.hpp.
#include "strata/prefill/kernels.hpp"
#include "strata/prefill/gdn_chunk.hpp"
#include "strata/prefill/gdn_probe.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/gfx_arch.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/core/emulate.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::prefill {
namespace {

constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;
constexpr int S = 128, HK = 16, HV = 48, C = 10240;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ uint16_t bf(float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
// The BF16 GEMMs' second operand (STRATA_PREFILL_BF16X2): what the BF16 image `hi` left out of f, itself in BF16.
// W.hi + W.lo carries ~16 mantissa bits of the activation - the decode path's FP32 x to within ~1e-5.
__device__ __forceinline__ uint16_t bf_lo(float f, uint16_t hi) { return bf(f - __uint_as_float((uint32_t) hi << 16)); }
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ uint16_t hf(float f) { return __half_as_ushort(__float2half_rn(f)); }
// A SwiGLU product for an FP16 GEMM: saturated, so a token with a massive activation cannot turn into inf and then
// NaN in the down projection (decode's q8_1 has room to ~8e6; FP16 ends at 65504).  A NaN stays NaN (fminf/fmaxf
// would make it -65504 and hide where it came from); finite values below 65504 round exactly as before.
// STRATA_F16_RANGE=1 (set_f16_range): what reaches FP16's range on this device - the largest |value| the activation
// images converted (hf_sat is their one funnel) and how many were beyond 65504 or NaN; W and Y are counted in gemm.cu.
// Off, the cost is one flag read per value.  The max is kept as float bits (positive floats order as unsigned ints),
// and a thread takes the atomic only when it beats the value it read, so the atomics stop once the max has settled.
#if defined(__HIPCC__)   // HIP only: the CUDA kernels stay exactly as they were (no flag read in hf_sat)
__device__ int g_f16_range_on = 0;
__device__ unsigned g_f16_range_max = 0;
__device__ unsigned long long g_f16_range_over = 0;
__device__ __forceinline__ void f16_range_note(float f) {
    const float a = fabsf(f);
    const unsigned bits = __float_as_uint(a);
    if (bits > g_f16_range_max) atomicMax(&g_f16_range_max, bits);
    if (a > 65504.0f || isnan(f)) atomicAdd(&g_f16_range_over, 1ull);
}
#endif
__device__ __forceinline__ uint16_t hf_sat(float f) {
#if defined(__HIPCC__)
    if (g_f16_range_on) f16_range_note(f);
#endif
    return hf(isnan(f) ? f : fminf(fmaxf(f, -65504.0f), 65504.0f));
}

// Turing's FP16 image of the hyper-connection's xn16 (gemm.cu's bf16_to_f16_kernel applied to a BF16 value): the
// writers store what the BF16 tensor-core path would convert, so the GEMM reads it without a conversion pass
__device__ __forceinline__ uint16_t bf_to_hf_conv(uint16_t b) {
    float f = __uint_as_float((uint32_t) b << 16);
    if (f > 65504.0f && !isinf(f)) f = 65504.0f;
    else if (f < -65504.0f && !isinf(f)) f = -65504.0f;
    return __half_as_ushort(__float2half_rn(f));
}
// The prompt path's 16-bit activation image for the BF16-weight GEMMs: BF16, or FP16 where the GEMM library is fast
// only in FP16 (prompt_f16() in gemm.cu: rocBLAS on gfx103x).  Set once per device before the first prompt.
#if defined(__HIPCC__)   // HIP only (#835): the CUDA kernels stay exactly as they were, they never read the flag
__device__ int g_act_f16 = 0;
__device__ __forceinline__ uint16_t act16(float f) { return g_act_f16 ? hf_sat(f) : bf(f); }
#else
__device__ __forceinline__ uint16_t act16(float f) { return bf(f); }
#endif
// block-wide sum for blockDim.x <= 1024, result broadcast
__device__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    float t = (threadIdx.x < nw) ? sh[threadIdx.x] : 0.0f;
    if (w == 0) t = warp_sum(t);
    if (threadIdx.x == 0) sh[0] = t;
    __syncthreads();
    return sh[0];
}
void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "prefill %s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}
unsigned blocks_for(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }

// ---------------------------------------------------------------- hyper-connection
__global__ void gr_norm_kernel(const float* __restrict__ R, const float* __restrict__ w, float eps,
                               float* __restrict__ xn, uint16_t* __restrict__ xn16, uint16_t* __restrict__ xn16_lo,
                               int f16img) {
    __shared__ float sh[32];
    const int64_t row = blockIdx.x;                 // t * 4 + c
    const int c = (int) (row % HC);
    const float* r = R + row * N;
    float ss = 0.0f;
    for (int d = threadIdx.x; d < N; d += blockDim.x) ss += r[d] * r[d];
    const float rs = rsqrtf(block_sum(ss, sh) / (float) N + eps);
    for (int d = threadIdx.x; d < N; d += blockDim.x) {
        const float v = r[d] * rs * w[c * N + d];
        xn[row * N + d] = v;
        const uint16_t h = act16(v);
        xn16[row * N + d] = f16img ? bf_to_hf_conv(h) : h;
        if (xn16_lo) xn16_lo[row * N + d] = bf_lo(v, h);
    }
}
// F-1: the row scale only (and the BF16 image); gr_mix_r_kernel recomputes r * rs * w itself, in the same order,
// so the FP32 copy of the normalized rows (T x 10240 floats) is neither written nor read
__global__ void gr_norm_rs_kernel(const float* __restrict__ R, const float* __restrict__ w, float eps,
                                  float* __restrict__ rs_out, uint16_t* __restrict__ xn16,
                                  uint16_t* __restrict__ xn16_lo, int64_t ldx, int f16img) {
    __shared__ float sh[32];
    const int64_t row = blockIdx.x;                 // t * 4 + c
    const int c = (int) (row % HC);
    const float* r = R + row * N;
    const int64_t xo = (row / HC) * ldx + (int64_t) c * N;   // the BF16 image's row (token stride ldx)
    float ss = 0.0f;
    for (int d = threadIdx.x; d < N; d += blockDim.x) ss += r[d] * r[d];
    const float rs = rsqrtf(block_sum(ss, sh) / (float) N + eps);
    if (threadIdx.x == 0) rs_out[row] = rs;
    for (int d = threadIdx.x; d < N; d += blockDim.x) {
        const float v = r[d] * rs * w[c * N + d];
        const uint16_t h = act16(v);
        xn16[xo + d] = f16img ? bf_to_hf_conv(h) : h;
        if (xn16_lo) xn16_lo[xo + d] = bf_lo(v, h);
    }
}
__global__ void gr_mix_r_kernel(const float* __restrict__ R, const float* __restrict__ rs, const float* __restrict__ w,
                                const float* __restrict__ g, float* __restrict__ mixed, uint16_t* __restrict__ mixed16,
                                int64_t T, uint16_t* __restrict__ mixed_h, uint16_t* __restrict__ mixed16_lo) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    const int64_t t = i / N, d = i % N;
    float s = 0.0f;
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const int64_t j = t * D + c * N + d;
        const float x = R[j] * rs[t * HC + c] * w[c * N + d];   // gr_norm_kernel's value, bit for bit
        s = fmaf(x, sigm(g[j]), s);
    }
    s /= (float) HC;
    mixed[i] = s;
    if (mixed16) {
        const uint16_t h = act16(s);
        mixed16[i] = h;
        if (mixed16_lo) mixed16_lo[i] = bf_lo(s, h);
    }
    if (mixed_h) mixed_h[i] = hf(s);
}
// F-2: gr_write_kernel for one row (t, c), then gr_norm_rs_kernel's reduction over it with the next half's norm
// weights - the same thread-to-element mapping (256 threads, stride 256) and block_sum, so rs and the BF16 image are
// the same bits, and R is not read back
constexpr int GRW_PER = (N + 255) / 256;
__global__ void __launch_bounds__(256) gr_write_norm_rs_kernel(float* __restrict__ R, const float* __restrict__ bo,
                                                               const float* __restrict__ inj, int64_t inj_ld,
                                                               const float* __restrict__ w, float eps,
                                                               float* __restrict__ rs_out, uint16_t* __restrict__ xn16,
                                                               uint16_t* __restrict__ xn16_lo, int64_t ldx, int f16img) {
    __shared__ float sh[32];
    const int64_t row = blockIdx.x;                 // t * 4 + c
    const int64_t t = row / HC;
    const int c = (int) (row % HC);
    float* r = R + row * N;
    const int64_t xo = t * ldx + (int64_t) c * N;
    const float sc = 2.0f * sigm(inj[t * inj_ld + c] / (float) HC);
    float v[GRW_PER];
    float ss = 0.0f;
    int k = 0;
#pragma unroll
    for (int d = threadIdx.x; d < N; d += 256, ++k) {
        const float x = fmaf(bo[t * N + d], sc, r[d]);
        r[d] = x;
        v[k] = x;
        ss += x * x;
    }
    const float rs = rsqrtf(block_sum(ss, sh) / (float) N + eps);
    if (threadIdx.x == 0) rs_out[row] = rs;
    k = 0;
#pragma unroll
    for (int d = threadIdx.x; d < N; d += 256, ++k) {
        const float x = v[k] * rs * w[c * N + d];
        const uint16_t h = act16(x);
        xn16[xo + d] = f16img ? bf_to_hf_conv(h) : h;
        if (xn16_lo) xn16_lo[xo + d] = bf_lo(x, h);
    }
}
// S23 (opt-in STRATA_CVEC_FUSE=1): a steered layer's FFN write, its control vector and the next half's norm in one
// pass over R (was gr_write + cvec_apply + gr_norm_rs: three).  The write is gr_write_kernel's, the vector
// cvec_kernel's (the same per-thread dot order over d = tid + 256 k, the same two-level xor reduction), the norm
// gr_norm_rs_kernel's (the same ss order and block_sum): the same bits.
__global__ void __launch_bounds__(256) gr_write_cvec_norm_rs_kernel(
    float* __restrict__ R, const float* __restrict__ bo, const float* __restrict__ inj, int64_t inj_ld,
    const float* __restrict__ v_l, const float* __restrict__ s_l, const int* __restrict__ on, int mode,
    const float* __restrict__ w, float eps, float* __restrict__ rs_out, uint16_t* __restrict__ xn16,
    uint16_t* __restrict__ xn16_lo, int64_t ldx, int f16img) {
    __shared__ float sh[32];
    __shared__ float part[8];
    const int64_t row = blockIdx.x;                 // t * 4 + c
    const int64_t t = row / HC;
    const int c = (int) (row % HC);
    float* r = R + row * N;
    const float sc = 2.0f * sigm(inj[t * inj_ld + c] / (float) HC);
    const float s = *s_l;
    const bool steer = *on != 0 && s != 0.0f;
    float x[GRW_PER];
    float dot = 0.0f;
    int k = 0;
#pragma unroll
    for (int d = threadIdx.x; d < N; d += 256, ++k) {
        const float xv = fmaf(bo[t * N + d], sc, r[d]);
        x[k] = xv;
        if (steer && mode == 0) dot = fmaf(xv, v_l[d], dot);
    }
    if (steer && mode == 0) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
        if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = dot;
        __syncthreads();
        if (threadIdx.x < 32) {
            float p = threadIdx.x < 8 ? part[threadIdx.x] : 0.0f;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) p += __shfl_xor_sync(0xffffffffu, p, o);
            if (threadIdx.x == 0) part[0] = p;
        }
        __syncthreads();
        dot = part[0] * s;
    }
    float ss = 0.0f;
    k = 0;
#pragma unroll
    for (int d = threadIdx.x; d < N; d += 256, ++k) {
        float xv = x[k];
        if (steer) xv = mode == 0 ? fmaf(-dot, v_l[d], xv) : xv + v_l[d];
        r[d] = xv;
        x[k] = xv;
        ss += xv * xv;
    }
    const float rs = rsqrtf(block_sum(ss, sh) / (float) N + eps);
    if (threadIdx.x == 0) rs_out[row] = rs;
    k = 0;
#pragma unroll
    for (int d = threadIdx.x; d < N; d += 256, ++k) {
        const float y = x[k] * rs * w[c * N + d];
        const uint16_t h = act16(y);
        xn16[t * ldx + (int64_t) c * N + d] = f16img ? bf_to_hf_conv(h) : h;
        if (xn16_lo) xn16_lo[t * ldx + (int64_t) c * N + d] = bf_lo(y, h);
    }
}
__global__ void gr_silu_kernel(const float* __restrict__ lo, uint16_t* __restrict__ lo16, uint16_t* __restrict__ lo16_lo,
                               int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = lo[i] / (float) HC;
    const float v = x / (1.0f + __expf(-x));
    const uint16_t h = act16(v);
    lo16[i] = h;
    if (lo16_lo) lo16_lo[i] = bf_lo(v, h);
}
__global__ void gr_mix_kernel(const float* __restrict__ xn, const float* __restrict__ g, float* __restrict__ mixed,
                              uint16_t* __restrict__ mixed16, int64_t T, uint16_t* __restrict__ mixed_h,
                              uint16_t* __restrict__ mixed16_lo) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    const int64_t t = i / N, d = i % N;
    float s = 0.0f;
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const int64_t j = t * D + c * N + d;
        s = fmaf(xn[j], sigm(g[j]), s);
    }
    s /= (float) HC;
    mixed[i] = s;
    if (mixed16) {
        const uint16_t h = act16(s);
        mixed16[i] = h;
        if (mixed16_lo) mixed16_lo[i] = bf_lo(s, h);
    }
    if (mixed_h) mixed_h[i] = hf(s);
}
__global__ void gr_write_kernel(float* __restrict__ R, const float* __restrict__ bo, const float* __restrict__ inj,
                                int64_t inj_ld, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * D) return;
    const int64_t t = i / D, c = (i % D) / N, d = i % N;
    R[i] = fmaf(bo[t * N + d], 2.0f * sigm(inj[t * inj_ld + c] / (float) HC), R[i]);
}
__global__ void gr_broadcast_kernel(const float* __restrict__ e, float* __restrict__ R, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * D) return;
    const int64_t t = i / D, d = i % N;
    R[i] = e[t * N + d];
}

// ---------------------------------------------------------------- GDN
__global__ void gdn_gates_kernel(const float* __restrict__ ab, const float* __restrict__ dt,
                                 const float* __restrict__ ssm_a, float* __restrict__ gate, float* __restrict__ beta,
                                 int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * HV) return;
    const int64_t t = i / HV, h = i % HV;
    const float v = ab[t * 2 * HV + h] + dt[h];
    gate[i] = (v > 20.0f ? v : log1pf(__expf(v))) * ssm_a[h];
    beta[i] = sigm(ab[t * 2 * HV + HV + h]);
}
// one thread per channel, walks the chunk; then a second kernel normalises
__global__ void gdn_conv_kernel(float* __restrict__ hist, const float* __restrict__ qkv, const float* __restrict__ w,
                                float* __restrict__ h, int64_t T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float v0 = hist[c * 3], v1 = hist[c * 3 + 1], v2 = hist[c * 3 + 2];
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int64_t t = 0; t < T; ++t) {
        const float x = qkv[t * C + c];
        const float s = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
        h[t * C + c] = s / (1.0f + __expf(-s));
        v0 = v1; v1 = v2; v2 = x;
    }
    hist[c * 3] = v0; hist[c * 3 + 1] = v1; hist[c * 3 + 2] = v2;
}
// C-3: the same 4-tap causal conv, tiled over tokens: thread (c, tile) reads its tile's 3 predecessors from the
// chunk (or the history before it) instead of carrying them - the conv reads inputs, not its own outputs, so the
// tiles are independent. The same expression per element (so the same bits); the history is written afterwards.
constexpr int CONV_TILE = 64;
__global__ void gdn_conv_tiled_kernel(const float* __restrict__ hist, const float* __restrict__ qkv,
                                      const float* __restrict__ w, float* __restrict__ h, int64_t T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int64_t t0 = (int64_t) blockIdx.y * CONV_TILE;
    if (t0 >= T) return;
    const int64_t t1 = t0 + CONV_TILE < T ? t0 + CONV_TILE : T;
    auto input = [&](int64_t t) -> float { return t >= 0 ? qkv[t * C + c] : hist[c * 3 + (int) (t + 3)]; };
    float v0 = input(t0 - 3), v1 = input(t0 - 2), v2 = input(t0 - 1);
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int64_t t = t0; t < t1; ++t) {
        const float x = qkv[t * C + c];
        const float s = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
        h[t * C + c] = s / (1.0f + __expf(-s));
        v0 = v1; v1 = v2; v2 = x;
    }
}
// Aurora (S23, opt-in STRATA_GDN_CONVL2=1): gdn_conv_tiled_kernel with gdn_l2_kernel folded in.  A block's 128 threads
// are the 128 channels of one head, so for the q / k heads (block < 2 HK) the norm's sum of squares is the block's:
// the same warp_sum, the same four partials added in the same order, the same v * rsqrtf(ss + eps) on the same
// stored value - the same bits, without writing h and reading it back (16 tokens per round).
__global__ void gdn_conv_l2_kernel(const float* __restrict__ hist, const float* __restrict__ qkv,
                                   const float* __restrict__ w, float* __restrict__ h, int64_t T, float eps) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t t0 = (int64_t) blockIdx.y * CONV_TILE;
    if (t0 >= T) return;
    const int64_t t1 = t0 + CONV_TILE < T ? t0 + CONV_TILE : T;
    auto input = [&](int64_t t) -> float { return t >= 0 ? qkv[t * C + c] : hist[c * 3 + (int) (t + 3)]; };
    float v0 = input(t0 - 3), v1 = input(t0 - 2), v2 = input(t0 - 1);
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int64_t t = t0; t < t1; ++t) {   // gdn_conv_tiled_kernel's loop, verbatim
        const float x = qkv[t * C + c];
        const float s = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
        h[t * C + c] = s / (1.0f + __expf(-s));
        v0 = v1; v1 = v2; v2 = x;
    }
    if (blockIdx.x >= 2 * HK) return;   // v channels: no norm
    __shared__ float part[16][4];
    for (int64_t tb = t0; tb < t1; tb += 16) {
        const int n = (int) (t1 - tb < 16 ? t1 - tb : 16);
        for (int j = 0; j < n; ++j) {   // (each thread reads back its own stored value)
            const float v = h[(tb + j) * C + c];
            // gdn_l2_kernel as the compiler built it: the first xor step adds the partner's rounded square to an
            // unrounded own one (an fma), the other steps are plain adds - spelled out so no contraction can differ
            float sq = __fmaf_rn(v, v, __shfl_xor_sync(0xffffffffu, __fmul_rn(v, v), 16));
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) sq = __fadd_rn(sq, __shfl_xor_sync(0xffffffffu, sq, o));
            if ((threadIdx.x & 31) == 0) part[j][threadIdx.x >> 5] = sq;
        }
        __syncthreads();
        for (int j = 0; j < n; ++j) {
            const float ss = part[j][0] + part[j][1] + part[j][2] + part[j][3];
            h[(tb + j) * C + c] = h[(tb + j) * C + c] * rsqrtf(ss + eps);
        }
        __syncthreads();
    }
}
// STRATA_GDN_CONVL2=2: the conv, SiLU and q / k L2 norm in ONE pass, a warp per (head, run of CONVW_TOK tokens): a lane owns
// four consecutive channels of the head's 128 (a float4 each way), walks the tokens keeping the three previous inputs in
// registers, and the head's sum of squares is one 5-step warp reduction a token - no shared memory, no barrier, h written
// once, normalised.  The conv + SiLU value is the expression of gdn_conv_tiled_kernel (same bits); the sum of squares is
// added in another order than gdn_l2_kernel's (FP32-level difference).  The first three inputs of a run are re-read from
// qkv (L2 hits; 3 / CONVW_TOK extra reads).  Bytes a token: 41 KB read, 41 KB written, against 114 KB for conv + l2.
constexpr int CONVW_TOK = 32;
__global__ void __launch_bounds__(128) gdn_conv_w_kernel(const float* __restrict__ hist, const float* __restrict__ qkv,
                                                         const float* __restrict__ w, float* __restrict__ h, int64_t T,
                                                         float eps) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t tb = ((int64_t) blockIdx.y * 4 + warp) * CONVW_TOK;
    if (tb >= T) return;
    const int64_t te = tb + CONVW_TOK < T ? tb + CONVW_TOK : T;
    const int c0 = blockIdx.x * 128 + lane * 4;
    const bool norm = blockIdx.x < 2 * HK;   // q / k heads
    float wr[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int k = 0; k < 4; ++k) wr[i][k] = w[(c0 + i) * 4 + k];
    float4 p0, p1, p2;   // inputs t - 3, t - 2, t - 1
    auto input = [&](int64_t t) -> float4 {
        if (t >= 0) return *reinterpret_cast<const float4*>(qkv + t * C + c0);
        return make_float4(hist[(c0 + 0) * 3 + (int) (t + 3)], hist[(c0 + 1) * 3 + (int) (t + 3)],
                           hist[(c0 + 2) * 3 + (int) (t + 3)], hist[(c0 + 3) * 3 + (int) (t + 3)]);
    };
    p0 = input(tb - 3); p1 = input(tb - 2); p2 = input(tb - 1);
#pragma unroll 4
    for (int64_t t = tb; t < te; ++t) {
        const float4 x = *reinterpret_cast<const float4*>(qkv + t * C + c0);
        const float xv[4] = {x.x, x.y, x.z, x.w}, a0[4] = {p0.x, p0.y, p0.z, p0.w}, a1[4] = {p1.x, p1.y, p1.z, p1.w},
                    a2[4] = {p2.x, p2.y, p2.z, p2.w};
        float o[4];
        float sq = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float s = a0[i] * wr[i][0] + a1[i] * wr[i][1] + a2[i] * wr[i][2] + xv[i] * wr[i][3];
            o[i] = s / (1.0f + __expf(-s));
            sq = fmaf(o[i], o[i], sq);
        }
        if (norm) {
#pragma unroll
            for (int m = 16; m > 0; m >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, m);
            const float rn = rsqrtf(sq + eps);
#pragma unroll
            for (int i = 0; i < 4; ++i) o[i] *= rn;
        }
        *reinterpret_cast<float4*>(h + t * C + c0) = make_float4(o[0], o[1], o[2], o[3]);
        p0 = p1; p1 = p2; p2 = x;
    }
}
// the history after the chunk: its last three inputs (the older history where the chunk is shorter than 3)
__global__ void gdn_conv_hist_kernel(float* __restrict__ hist, const float* __restrict__ qkv, int64_t T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float v[3];
    for (int k = 0; k < 3; ++k) {
        const int64_t t = T - 3 + k;
        v[k] = t >= 0 ? qkv[t * C + c] : hist[c * 3 + (int) (t + 3)];
    }
    hist[c * 3] = v[0]; hist[c * 3 + 1] = v[1]; hist[c * 3 + 2] = v[2];
}
__global__ void gdn_l2_kernel(float* __restrict__ h, float eps) {
    // block (t, head) over the 32 q/k heads, 128 threads
    const int64_t t = blockIdx.y;
    const int head = blockIdx.x;
    float* x = h + t * C + head * S;
    const float v = x[threadIdx.x];
    float sq = warp_sum(v * v);
    __shared__ float part[4];
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
    __syncthreads();
    const float ss = part[0] + part[1] + part[2] + part[3];
    x[threadIdx.x] = v * rsqrtf(ss + eps);
}
constexpr int RG = 4, RPG = S / RG;
constexpr int CB = 32, NCB = S / CB;  // value columns per block kernel / blocks per head
// v100/gdn-chunk: Volta-only (sm_70), inside the experimental build.  Main's recurrence kernels above are
// untouched for every other architecture; the chain-split is dispatched only when the device is cc 7.0.
#if defined(STRATA_EXPERIMENTAL_SM60)
// v100/gdn-chunk: the same recurrence with WARP-LOCAL reductions - no __syncthreads at all.  gdn_rec_cols_*'
// per-token shape is two cross-warp reductions (k^T W and q^T W) behind two barriers each (six barriers per
// token in the pipelined kernel); here a warp owns 8 columns and the 4 row-groups are lanes within the warp,
// so the reductions are __shfl's.  The arithmetic is the same expressions in the same order - the four
// row-group partials still sum as ((p0+p1)+p2)+p3 - so this kernel is BIT-EXACT against gdn_rec_cols_pipe
// (the parity test asserts it).  The state has the same (S, h_v, S) layout and the same j-fastest column.
__global__ void __launch_bounds__(S) gdn_rec_cols_warp_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                             const float* __restrict__ gate,
                                                             const float* __restrict__ beta,
                                                             float* __restrict__ oc_out, int64_t T) {
    const int head = blockIdx.x / NCB, cb = blockIdx.x % NCB;
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;   // warp w: columns w*8..w*8+7
    const int col = cb * CB + w * 8 + (l & 7);              // lane: column within the warp's 8
    const int rg = l >> 3;                                  // lane group: 4 row-groups of 32 rows
    const int qh = head % HK;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * HV + head) * S + col;
    const size_t rs = (size_t) HV * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * rs];
    const unsigned full = 0xffffffffu;
    for (int64_t t = 0; t < T; ++t) {
        const float* ht = h + t * C;
        const float* krow = ht + HK * S + qh * S + rg * RPG;
        const float* qrow = ht + qh * S + rg * RPG;
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], krow[r], kv);
        // ((p0 + p1) + p2) + p3 over the row-groups, exactly as red[0]+red[1]+red[2]+red[3]
        float p0 = __shfl_sync(full, kv, (l & 7));
        float p1 = __shfl_sync(full, kv, (l & 7) + 8);
        float p2 = __shfl_sync(full, kv, (l & 7) + 16);
        float p3 = __shfl_sync(full, kv, (l & 7) + 24);
        const float kv_col = ((p0 + p1) + p2) + p3;
        const float g = __expf(gate[t * HV + head]);
        const float delta = (ht[2 * HK * S + head * S + col] - g * kv_col) * beta[t * HV + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], krow[r] * delta);
            o = fmaf(s[r], qrow[r], o);
        }
        float q0 = __shfl_sync(full, o, (l & 7));
        float q1 = __shfl_sync(full, o, (l & 7) + 8);
        float q2 = __shfl_sync(full, o, (l & 7) + 16);
        float q3 = __shfl_sync(full, o, (l & 7) + 24);
        oc_out[t * HV * S + head * S + col] = (((q0 + q1) + q2) + q3) * rsqrtf((float) S);
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * rs] = s[r];
}
#endif
__global__ void __launch_bounds__(S * RG) gdn_rec_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                         const float* __restrict__ gate,
                                                         const float* __restrict__ beta, const float* __restrict__ z,
                                                         const float* __restrict__ gamma, float eps,
                                                         float* __restrict__ y, uint16_t* __restrict__ y16, int64_t T,
                                                         int64_t ld16) {
    __shared__ float sk[S], sq[S], red[RG][S], wsum[16];
    const int head = blockIdx.x, col = threadIdx.x, rg = threadIdx.y, tid = rg * S + col;
    const int qh = head % HK;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * HV + head) * S + col;
    const size_t rs = (size_t) HV * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * rs];
    const float g_col = gamma[col];
    for (int64_t t = 0; t < T; ++t) {
        const float* ht = h + t * C;
        __syncthreads();
        if (tid < S) { sq[tid] = ht[qh * S + tid]; sk[tid] = ht[HK * S + qh * S + tid]; }
        __syncthreads();
        const float g = __expf(gate[t * HV + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red[rg][col] = kv;
        __syncthreads();
        const float kv_col = red[0][col] + red[1][col] + red[2][col] + red[3][col];
        const float delta = (ht[2 * HK * S + head * S + col] - g * kv_col) * beta[t * HV + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        __syncthreads();
        red[rg][col] = o;
        __syncthreads();
        float oc = 0.0f, sp = 0.0f;
        if (rg == 0) {
            oc = (red[0][col] + red[1][col] + red[2][col] + red[3][col]) * rsqrtf((float) S);
            sp = oc * oc;
        }
        sp = warp_sum(sp);
        if ((tid & 31) == 0) wsum[tid >> 5] = sp;
        __syncthreads();
        if (rg == 0) {
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float v = oc * rsqrtf(ss / (float) S + eps) * g_col * sigm(z[t * HV * S + head * S + col]);
            y[t * HV * S + head * S + col] = v;
            y16[t * ld16 + head * S + col] = hf(v);
        }
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * rs] = s[r];
}

// D-2: the recurrence with the value columns split over 4 blocks per head (4x the blocks of the kernel above, a
// quarter of its threads per __syncthreads), and the output norm - the only step that couples the head's columns -
// in its own kernel. Per column the same arithmetic in the same order (the 4 row-group partial sums added as
// red[0] + red[1] + red[2] + red[3]; the norm's warp sums over the same 32-column warps): the same bits.

__global__ void __launch_bounds__(CB * RG) gdn_rec_cols_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                                const float* __restrict__ gate,
                                                                const float* __restrict__ beta,
                                                                float* __restrict__ oc_out, int64_t T) {
    __shared__ float sk[S], sq[S], red[RG][CB];
    const int head = blockIdx.x / NCB, cb = blockIdx.x % NCB;
    const int c = threadIdx.x, rg = threadIdx.y, tid = rg * CB + c, col = cb * CB + c;
    const int qh = head % HK;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * HV + head) * S + col;
    const size_t rs = (size_t) HV * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * rs];
    for (int64_t t = 0; t < T; ++t) {
        const float* ht = h + t * C;
        __syncthreads();
        if (tid < S) { sq[tid] = ht[qh * S + tid]; sk[tid] = ht[HK * S + qh * S + tid]; }
        __syncthreads();
        const float g = __expf(gate[t * HV + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red[rg][c] = kv;
        __syncthreads();
        const float kv_col = red[0][c] + red[1][c] + red[2][c] + red[3][c];
        const float delta = (ht[2 * HK * S + head * S + col] - g * kv_col) * beta[t * HV + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        __syncthreads();
        red[rg][c] = o;
        __syncthreads();
        if (rg == 0) oc_out[t * HV * S + head * S + col] = (red[0][c] + red[1][c] + red[2][c] + red[3][c]) * rsqrtf((float) S);
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * rs] = s[r];
}
// gdn_rec_cols_kernel with the next token's inputs (q/k rows, v, gate, beta) loaded into registers while this token
// computes (software pipelining).  The same arithmetic in the same order: the same bits, and the same CB-column split.
// STRATA_GDN_PIPELINE=0: gdn_rec_cols_kernel.
__global__ void __launch_bounds__(CB * RG) gdn_rec_cols_pipe_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                                      const float* __restrict__ gate,
                                                                      const float* __restrict__ beta,
                                                                      float* __restrict__ oc_out, int64_t T) {
    constexpr int NT = CB * RG, LPT = S / NT;   // threads, q/k rows loaded per thread
    __shared__ float sk[S], sq[S], red[RG][CB];
    const int head = blockIdx.x / NCB, cb = blockIdx.x % NCB;
    const int c = threadIdx.x, rg = threadIdx.y, tid = rg * CB + c, col = cb * CB + c;
    const int qh = head % HK;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * HV + head) * S + col;
    const size_t rs = (size_t) HV * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * rs];
    float nq[LPT], nk[LPT], nv = 0.0f, ng = 0.0f, nb = 0.0f;
    auto fetch = [&](int64_t t) {
        const float* ht = h + t * C;
#pragma unroll
        for (int u = 0; u < LPT; ++u) { nq[u] = ht[qh * S + tid + u * NT]; nk[u] = ht[HK * S + qh * S + tid + u * NT]; }
        nv = ht[2 * HK * S + head * S + col];
        ng = gate[t * HV + head];
        nb = beta[t * HV + head];
    };
    if (T > 0) fetch(0);
    for (int64_t t = 0; t < T; ++t) {
        float cq[LPT], ck[LPT];
#pragma unroll
        for (int u = 0; u < LPT; ++u) { cq[u] = nq[u]; ck[u] = nk[u]; }
        const float cv = nv, cg = ng, cbt = nb;
        __syncthreads();
#pragma unroll
        for (int u = 0; u < LPT; ++u) { sq[tid + u * NT] = cq[u]; sk[tid + u * NT] = ck[u]; }
        __syncthreads();
        if (t + 1 < T) fetch(t + 1);
        const float g = __expf(cg);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red[rg][c] = kv;
        __syncthreads();
        const float kv_col = red[0][c] + red[1][c] + red[2][c] + red[3][c];
        const float delta = (cv - g * kv_col) * cbt;
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        __syncthreads();
        red[rg][c] = o;
        __syncthreads();
        if (rg == 0) oc_out[t * HV * S + head * S + col] = (red[0][c] + red[1][c] + red[2][c] + red[3][c]) * rsqrtf((float) S);
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * rs] = s[r];
}
// v100/gdn-chunk: gdn_rec_cols_pipe_kernel with the two 32-deep FMA chains (k^T W and q^T W) split into 4
// accumulators each.  The chain is the recurrence's per-token floor (a 32-step dependent FFMA chain is ~128
// cycles before the barriers); the split keeps the load pattern, the barriers and the (S, h_v, S) state
// layout untouched.  FP32-level, NOT bit-exact - the four partial sums group the 32 products differently -
// so it is opt-in by measured benefit (STRATA_GDN_REC_WARP=2; gdn_chunk_parity bounds the error).
#if defined(STRATA_EXPERIMENTAL_SM60)
__global__ void __launch_bounds__(CB * RG) gdn_rec_cols_pipe_fast_kernel(float* __restrict__ state,
                                                                        const float* __restrict__ h,
                                                                        const float* __restrict__ gate,
                                                                        const float* __restrict__ beta,
                                                                        float* __restrict__ oc_out, int64_t T) {
    constexpr int NT = CB * RG, LPT = S / NT;
    __shared__ float sk[S], sq[S], red[RG][CB];
    const int head = blockIdx.x / NCB, cb = blockIdx.x % NCB;
    const int c = threadIdx.x, rg = threadIdx.y, tid = rg * CB + c, col = cb * CB + c;
    const int qh = head % HK;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * HV + head) * S + col;
    const size_t rs = (size_t) HV * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * rs];
    float nq[LPT], nk[LPT], nv = 0.0f, ng = 0.0f, nb = 0.0f;
    auto fetch = [&](int64_t t) {
        const float* ht = h + t * C;
#pragma unroll
        for (int u = 0; u < LPT; ++u) { nq[u] = ht[qh * S + tid + u * NT]; nk[u] = ht[HK * S + qh * S + tid + u * NT]; }
        nv = ht[2 * HK * S + head * S + col];
        ng = gate[t * HV + head];
        nb = beta[t * HV + head];
    };
    if (T > 0) fetch(0);
    for (int64_t t = 0; t < T; ++t) {
        float cq[LPT], ck[LPT];
#pragma unroll
        for (int u = 0; u < LPT; ++u) { cq[u] = nq[u]; ck[u] = nk[u]; }
        const float cv = nv, cg = ng, cbt = nb;
        __syncthreads();
#pragma unroll
        for (int u = 0; u < LPT; ++u) { sq[tid + u * NT] = cq[u]; sk[tid + u * NT] = ck[u]; }
        __syncthreads();
        if (t + 1 < T) fetch(t + 1);
        const float g = __expf(cg);
        float kv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv[r & 3] = fmaf(s[r], sk[rg * RPG + r], kv[r & 3]);
        red[rg][c] = (kv[0] + kv[1]) + (kv[2] + kv[3]);
        __syncthreads();
        const float kv_col = red[0][c] + red[1][c] + red[2][c] + red[3][c];
        const float delta = (cv - g * kv_col) * cbt;
        float o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o[r & 3] = fmaf(s[r], sq[rg * RPG + r], o[r & 3]);
        }
        __syncthreads();
        red[rg][c] = (o[0] + o[1]) + (o[2] + o[3]);
        __syncthreads();
        if (rg == 0) oc_out[t * HV * S + head * S + col] = (red[0][c] + red[1][c] + red[2][c] + red[3][c]) * rsqrtf((float) S);
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * rs] = s[r];
}
#endif

#if !defined(__HIPCC__)
// The recurrence with one thread for the three value heads that share a key head (head % HK): column c of heads
// qh, qh + 16 and qh + 32, row group rg.  gdn_rec_cols_pipe_kernel spends its time in shared memory, not in
// arithmetic: every thread of a warp needs the same 32 q and k values per token (the k twice), and a warp receives one
// such broadcast value per clock however wide the load.  Here every q/k value a thread loads feeds three heads, and a
// token's k row goes into registers once for both of its uses.  The inputs come in blocks of GDN_TB tokens, copied to
// shared memory by cp.async while the block before computes (one token ahead is shorter than a load from L2 takes),
// and the two cross-row-group sums have their own arrays, so a token needs 2 __syncthreads instead of 5: the second
// one of a token orders every read of rkv before the next token's writes, the next token's first one every read of ro
// before the writes after it.  64 blocks instead of 192.  Per value head and column the same arithmetic in the same
// order: the same bits (src/prefill/gdn_rec_parity.cu checks them and times the variants: 1.41x on a 4080 Super).
// sm_80+ cards that hold its 64 blocks at once (gdn_keyhead_ok); STRATA_GDN_KEYHEAD=0: gdn_rec_cols_pipe_kernel.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#define STRATA_GDN_CP_ASYNC 0   // Turing builds: plain copies (never launched there, see gdn_keyhead_ok)
#else
#define STRATA_GDN_CP_ASYNC 1
#endif
__device__ __forceinline__ void gdn_cp4(float* smem, const float* gmem) {
#if STRATA_GDN_CP_ASYNC
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"((unsigned) __cvta_generic_to_shared(smem)), "l"(gmem));
#else
    *smem = *gmem;
#endif
}
__device__ __forceinline__ void gdn_cp16(float* smem, const float* gmem) {
#if STRATA_GDN_CP_ASYNC
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(smem)), "l"(gmem));
#else
    *reinterpret_cast<float4*>(smem) = *reinterpret_cast<const float4*>(gmem);
#endif
}
__device__ __forceinline__ void gdn_cp_commit() {
#if STRATA_GDN_CP_ASYNC
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
__device__ __forceinline__ void gdn_cp_wait_prev() {   // every group but the newest has landed
#if STRATA_GDN_CP_ASYNC
    asm volatile("cp.async.wait_group 1;\n" ::);
#endif
}
constexpr int GDN_TB = 8, VPK = HV / HK;   // tokens per staged block, value heads per key head
__global__ void __launch_bounds__(CB * RG) gdn_rec_kh_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                               const float* __restrict__ gate,
                                                               const float* __restrict__ beta,
                                                               float* __restrict__ oc_out, int64_t T) {
    constexpr int TB = GDN_TB, NT = CB * RG, QKP = S / 4, VP = CB / 4;   // threads, 16-byte pieces of a q/k row, of v
    __shared__ __align__(16) float sq[2][TB][S];
    __shared__ __align__(16) float sk[2][TB][S];
    __shared__ __align__(16) float sv[2][TB][VPK][CB];
    __shared__ float sg[2][TB][VPK], sb[2][TB][VPK], rkv[VPK][RG][CB], ro[VPK][RG][CB];
    const int qh = blockIdx.x / NCB, cb = blockIdx.x % NCB;
    const int c = threadIdx.x, rg = threadIdx.y, tid = rg * CB + c, col = cb * CB + c;
    float s[VPK][RPG];
    const size_t rs = (size_t) HV * S;
#pragma unroll
    for (int j = 0; j < VPK; ++j) {
        const float* base = state + ((size_t) (rg * RPG) * HV + qh + j * HK) * S + col;
#pragma unroll
        for (int r = 0; r < RPG; ++r) s[j][r] = base[r * rs];
    }
    const int64_t nblk = (T + TB - 1) / TB;
    auto stage = [&](int64_t k) {   // tokens [k * TB, k * TB + TB) into buffer k & 1
        const int bb = (int) (k & 1);
        const int64_t t0 = k * TB;
        for (int p = tid; p < TB * 2 * QKP; p += NT) {
            const int i = p / (2 * QKP), w = p % (2 * QKP), isk = w / QKP, jj = (w % QKP) * 4;
            if (t0 + i < T)
                gdn_cp16(isk ? &sk[bb][i][jj] : &sq[bb][i][jj], h + (t0 + i) * C + (isk ? HK * S : 0) + qh * S + jj);
        }
        for (int p = tid; p < TB * VPK * VP; p += NT) {
            const int i = p / (VPK * VP), w = p % (VPK * VP), j = w / VP, jj = (w % VP) * 4;
            if (t0 + i < T)
                gdn_cp16(&sv[bb][i][j][jj], h + (t0 + i) * C + 2 * HK * S + (qh + j * HK) * S + cb * CB + jj);
        }
        for (int p = tid; p < 2 * TB * VPK; p += NT) {
            const int isb = p / (TB * VPK), w = p % (TB * VPK), i = w / VPK, j = w % VPK;
            if (t0 + i < T)
                gdn_cp4(isb ? &sb[bb][i][j] : &sg[bb][i][j], (isb ? beta : gate) + (t0 + i) * HV + qh + j * HK);
        }
    };
    if (nblk > 0) stage(0);
    gdn_cp_commit();
    for (int64_t k = 0; k < nblk; ++k) {
        // buffer (k + 1) & 1 was read by block k - 1, whose last token's second __syncthreads every thread has passed
        if (k + 1 < nblk) stage(k + 1);
        gdn_cp_commit();
        gdn_cp_wait_prev();
        __syncthreads();
        const int bb = (int) (k & 1);
        const int n = (int) ((T - k * TB) < TB ? (T - k * TB) : TB);
        for (int i = 0; i < n; ++i) {
            const int64_t t = k * TB + i;
            float kc[RPG];
#pragma unroll
            for (int r = 0; r < RPG; ++r) kc[r] = sk[bb][i][rg * RPG + r];
            float g[VPK], kv[VPK], delta[VPK], o[VPK];
#pragma unroll
            for (int j = 0; j < VPK; ++j) { g[j] = __expf(sg[bb][i][j]); kv[j] = 0.0f; o[j] = 0.0f; }
#pragma unroll
            for (int r = 0; r < RPG; ++r)
#pragma unroll
                for (int j = 0; j < VPK; ++j) kv[j] = fmaf(s[j][r], kc[r], kv[j]);
#pragma unroll
            for (int j = 0; j < VPK; ++j) rkv[j][rg][c] = kv[j];
            __syncthreads();
#pragma unroll
            for (int j = 0; j < VPK; ++j) {
                const float kv_col = rkv[j][0][c] + rkv[j][1][c] + rkv[j][2][c] + rkv[j][3][c];
                delta[j] = (sv[bb][i][j][c] - g[j] * kv_col) * sb[bb][i][j];
            }
#pragma unroll
            for (int r = 0; r < RPG; ++r) {
                const float qr = sq[bb][i][rg * RPG + r];
#pragma unroll
                for (int j = 0; j < VPK; ++j) {
                    s[j][r] = fmaf(g[j], s[j][r], kc[r] * delta[j]);
                    o[j] = fmaf(s[j][r], qr, o[j]);
                }
            }
#pragma unroll
            for (int j = 0; j < VPK; ++j) ro[j][rg][c] = o[j];
            __syncthreads();
            if (rg < VPK)   // row group j writes head j's output
                oc_out[t * HV * S + (qh + rg * HK) * S + col] =
                    (ro[rg][0][c] + ro[rg][1][c] + ro[rg][2][c] + ro[rg][3][c]) * rsqrtf((float) S);
        }
    }
#pragma unroll
    for (int j = 0; j < VPK; ++j) {
        float* base = state + ((size_t) (rg * RPG) * HV + qh + j * HK) * S + col;
#pragma unroll
        for (int r = 0; r < RPG; ++r) base[r * rs] = s[j][r];
    }
}
// gdn_rec_kh_kernel where it pays: a CUDA card with cp.async (sm_80+) that holds all 64 of its blocks at once (each
// walks the whole chunk, so blocks left for a second wave would double the time).  The busiest SM sets the pace: from
// 64 SMs up this kernel has one block per SM, below that two on some SMs, while the kernel before has ceil(192 / SMs).
// With the engine's grids on fewer SMs (gdn_rec_parity --bench), on a 4080 SUPER / a 3090: 1.40-1.42x / 1.28-1.29x at
// 64 SMs and more, 1.02-1.04x / 1.03-1.06x at 48 to 63 (two blocks against four: a draw, slower on neither),
// 1.28-1.31x / 1.28-1.30x at 39 to 47, 1.53-1.57x / 1.54-1.55x at 32 to 38.  Per call, from the current device (a
// layer split can mix cards).
bool gdn_keyhead_ok() {
    static const bool off = [] { const char* v = std::getenv("STRATA_GDN_KEYHEAD"); return v != nullptr && std::atoi(v) == 0; }();
    if (off) return false;
    static int known[64] = {};   // per device: 0 not asked yet, 1 yes, 2 no
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (known[dev] == 0) {
        int major = 0, sms = 0, per_sm = 0;
        const bool yes = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                         cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess && major >= 8 &&
                         cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, gdn_rec_kh_kernel, CB * RG, 0) ==
                             cudaSuccess &&
                         (int64_t) per_sm * sms >= (int64_t) HK * NCB;
        if (!yes) cudaGetLastError();
        known[dev] = yes ? 1 : 2;
    }
    return known[dev] == 1;
}
// STRATA_GDN_CHUNKED=1 (opt-in): the recurrence in chunks of GCH tokens (the WY form of the gated delta rule, as
// flash-linear-attention computes it), all in FP32.  Per value head, with g_t = exp(gate_t), gamma_t the gate's sum
// from the chunk's start to t (inclusive) and S0 the state before the chunk:
//   A[t][i] = beta_t exp(gamma_t - gamma_i) k_t.k_i (i < t),   T = (I + A)^-1 (unit lower triangular),
//   Y_t = beta_t (v_t - exp(gamma_t) S0^T k_t),   D = T Y (the delta rule's corrections, one per token),
//   o_t = exp(gamma_t) S0^T q_t + sum_{i <= t} exp(gamma_t - gamma_i) (q_t.k_i) D_i,
//   S = exp(gamma_last) S0 + sum_t exp(gamma_last - gamma_t) k_t D_t^T,
// the same state and outputs as the token-by-token kernels up to FP32 rounding (another order of the sums).
// gdn_chunk_prep_kernel builds T, P = exp(gamma_t - gamma_i) q_t.k_i (i <= t) and gamma for every chunk of a
// super-block at once (a block per chunk and key head: its three value heads share k and q); gdn_chunk_scan_kernel then
// walks the super-block's chunks per (value head, 16 columns): the state slice in registers, the products as small
// register-tiled matmuls from shared memory.  Every exponent is <= 0 (gates are negative), so nothing overflows.
// STRATA_GDN_CHUNKED=1 takes prompt chunks of this many tokens and more: on an RTX 5090 the two launches cost more
// than they save below ~100 tokens (gdn_rec_parity --bench: 32 tokens 0.58x gdn_rec_kh_kernel, 96 1.03x, 128 1.20x,
// 512 1.67x, 2048+ 1.81x)
constexpr int64_t kGdnChunkedMin = 128;
constexpr int64_t kGdnTcMin = 128;        // STRATA_GDN_TC: the two launches cost ~20 us; below this the kh kernel wins
constexpr int GCH = 32;            // tokens per chunk (the prep's solve is a lane per token)
constexpr int GSB = 2048;          // tokens per super-block: the prep's scratch is GSB / GCH chunks of T, P and gamma
constexpr int GDV = 16;            // value columns per scan block
constexpr int GKP = S + 4;         // the scan's padded q / k rows (8 rows of a warp land on 8 banks)
constexpr int GTP = GCH + 1;       // padded T / P rows
static_assert(GCH == 32, "gdn_chunk_prep_kernel: a lane per token of the chunk");
constexpr size_t kChunkPrepSmem = sizeof(float) * (2 * GCH * (S + 1) + 2 * GCH * GTP);   // A over k / q
static_assert(VPK * GCH * GTP <= 2 * GCH * (S + 1), "gdn_chunk_prep_kernel: A fits where k and q were");
constexpr size_t kChunkScanSmem = sizeof(float) * (3 * GCH * GKP + VPK * (S * GDV + 3 * GCH * GDV));

__global__ void __launch_bounds__(256) gdn_chunk_prep_kernel(const float* __restrict__ h, const float* __restrict__ gate,
                                                             const float* __restrict__ beta, float* __restrict__ tm,
                                                             float* __restrict__ pm, float* __restrict__ gm, int64_t t0,
                                                             int64_t T) {
    extern __shared__ __align__(16) float gsm[];
    float* sk = gsm;                       // [GCH][S + 1]
    float* sq = sk + GCH * (S + 1);        // [GCH][S + 1]
    float* skk = sq + GCH * (S + 1);       // [GCH][GTP]  k_t.k_i
    float* sqk = skk + GCH * GTP;          // [GCH][GTP]  q_t.k_i
    float* sa = gsm;                       // [VPK][GCH][GTP]  A, then T below the diagonal (over k and q:
                                           // read by then, so two blocks share an SM)
    const int qh = blockIdx.y, ch = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int64_t tb = t0 + (int64_t) ch * GCH;
    const int n = (int) min((int64_t) GCH, T - tb);
    {   // k and q as float4, every load issued before the first store
        constexpr int NP = GCH * S / 4 / 256;   // float4 a thread, each of k and q
        float4 kv[NP], qv[NP];
#pragma unroll
        for (int m = 0; m < NP; ++m) {
            const int p = tid + 256 * m, t = p / (S / 4), r = (p % (S / 4)) * 4;
            kv[m] = t < n ? *reinterpret_cast<const float4*>(h + (tb + t) * C + HK * S + qh * S + r) : make_float4(0.f, 0.f, 0.f, 0.f);
            qv[m] = t < n ? *reinterpret_cast<const float4*>(h + (tb + t) * C + qh * S + r) : make_float4(0.f, 0.f, 0.f, 0.f);
        }
#pragma unroll
        for (int m = 0; m < NP; ++m) {
            const int p = tid + 256 * m, t = p / (S / 4), r = (p % (S / 4)) * 4;
            float* dk = sk + t * (S + 1) + r;
            float* dq = sq + t * (S + 1) + r;
            dk[0] = kv[m].x; dk[1] = kv[m].y; dk[2] = kv[m].z; dk[3] = kv[m].w;
            dq[0] = qv[m].x; dq[1] = qv[m].y; dq[2] = qv[m].z; dq[3] = qv[m].w;
        }
    }
    __syncthreads();
    {   // lane i, rows warp + 8m: k_t.k_i and q_t.k_i on eight independent sums (rows 129 apart: 32 banks)
        constexpr int RW = GCH / 8;
        float kk[RW] = {}, qk[RW] = {};
        for (int r = 0; r < S; ++r) {
            const float ki = sk[lane * (S + 1) + r];
#pragma unroll
            for (int m = 0; m < RW; ++m) {
                kk[m] = fmaf(sk[(warp + 8 * m) * (S + 1) + r], ki, kk[m]);
                qk[m] = fmaf(sq[(warp + 8 * m) * (S + 1) + r], ki, qk[m]);
            }
        }
#pragma unroll
        for (int m = 0; m < RW; ++m) {
            const int t = warp + 8 * m;
            skk[t * GTP + lane] = lane <= t ? kk[m] : 0.0f;
            sqk[t * GTP + lane] = lane <= t ? qk[m] : 0.0f;
        }
    }
    __syncthreads();
    if (warp >= VPK) return;
    const int vh = qh + warp * HK, i = lane;   // warp j: value head j of the key head; lane i: token i (column i)
    float* a = sa + warp * GCH * GTP;
    const float b = i < n ? beta[(tb + i) * HV + vh] : 0.0f;
    float gam = i < n ? gate[(tb + i) * HV + vh] : 0.0f;   // past the end: gate 0, beta 0 (no effect on S)
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const float u = __shfl_up_sync(0xffffffffu, gam, o);
        if (lane >= o) gam += u;
    }
    for (int t = 0; t < GCH; ++t) {
        const float gt = __shfl_sync(0xffffffffu, gam, t), bt = __shfl_sync(0xffffffffu, b, t);
        a[t * GTP + i] = i < t ? bt * expf(gt - gam) * skk[t * GTP + i] : (i == t ? 1.0f : 0.0f);
    }
    __syncwarp();
    // T = (I + A)^-1 by forward substitution, row by row in place: T[t][i] = -sum_{k < t} A[t][k] T[k][i] over the
    // finished rows (diagonal 1, zeros above it: the terms k < i add an exact 0, so the same bits as summing from k = i)
    // - a trip the whole warp shares, so the loads go ahead of the sums
    for (int t = 1; t < GCH; ++t) {
        float v = 0.0f;
#pragma unroll 8
        for (int k = 0; k < t; ++k) v = fmaf(-a[t * GTP + k], a[k * GTP + i], v);
        __syncwarp();
        if (i < t) a[t * GTP + i] = v;
        __syncwarp();
    }
    float* T_ = tm + ((int64_t) ch * HV + vh) * GCH * GCH;
    float* P_ = pm + ((int64_t) ch * HV + vh) * GCH * GCH;
    for (int t = 0; t < GCH; ++t) {
        const float gt = __shfl_sync(0xffffffffu, gam, t);
        T_[t * GCH + i] = a[t * GTP + i];
        P_[t * GCH + i] = i <= t ? expf(gt - gam) * sqk[t * GTP + i] : 0.0f;
    }
    gm[((int64_t) ch * HV + vh) * GCH + i] = gam;
}

__device__ __forceinline__ void gdn_pf_l2(const float* p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }
__device__ __forceinline__ void gdn_cp_wait_all() {
#if STRATA_GDN_CP_ASYNC
    asm volatile("cp.async.wait_all;\n" ::);
#endif
}
// A block per (key head, 16 value columns): 128 threads for each of the key head's three value heads, which share the
// staged k and q rows.  128 blocks, one an SM, in one wave (each walks every chunk: a second wave would double the
// time).  A thread owns one token's row of the products (4 columns) and holds that token's T and P rows in registers,
// loaded at the chunk's start so they land during the first product; the next chunk's k (a second buffer), q (after
// this chunk's last read of q), v, gamma and beta load while this chunk computes.  Three barriers a chunk.
__global__ void __launch_bounds__(128 * VPK) gdn_chunk_scan_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                                   const float* __restrict__ beta, const float* __restrict__ tm,
                                                                   const float* __restrict__ pm, const float* __restrict__ gm,
                                                                   float* __restrict__ oc_out, int64_t t0, int nch, int64_t T) {
    extern __shared__ __align__(16) float gsm[];
    constexpr int NT = 128 * VPK, HEAD = S * GDV + 3 * GCH * GDV;   // threads; a head's shared floats
    float* kbuf = gsm;                     // [2][GCH][GKP]  k, double-buffered
    float* sq = kbuf + 2 * GCH * GKP;      // [GCH][GKP]
    const int qh = blockIdx.x, c0 = blockIdx.y * GDV, j = threadIdx.x / 128, tid = threadIdx.x % 128;
    const int vh = qh + j * HK;
    float* ss = sq + GCH * GKP + j * HEAD; // [S][GDV]  the state slice at the chunk's start
    float* sy = ss + S * GDV;              // [GCH][GDV]  Y
    float* sd = sy + GCH * GDV;            // [GCH][GDV]  D
    float* sdd = sd + GCH * GDV;           // [GCH][GDV]  exp(gamma_last - gamma_t) D
    const int rq = tid >> 2, vq = tid & 3;   // the update's tile: rows 4rq..4rq+3; both tiles: columns 4vq..4vq+3
    const int tt = tid >> 2, vv = 4 * vq;    // the products' tile: token tt
    const size_t rs = (size_t) HV * S;
    auto stage_kq = [&](int64_t c_t0, int n, float* dst, bool isq) {   // k or q rows of a chunk, cp.async
        for (int p = threadIdx.x; p < GCH * (S / 4); p += NT) {
            const int t = p / (S / 4), r = (p % (S / 4)) * 4;
            if (t < n) gdn_cp16(dst + t * GKP + r, h + (c_t0 + t) * C + (isq ? 0 : HK * S) + qh * S + r);
            else *reinterpret_cast<float4*>(dst + t * GKP + r) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    };
    // this tile's v, gamma, beta and the chunk's last gamma (registers)
    auto load_vgb = [&](int ch, int64_t c_t0, int n, float4& v4, float& gam, float& bet, float& glast) {
        v4 = tt < n ? __ldg(reinterpret_cast<const float4*>(h + (c_t0 + tt) * C + 2 * HK * S + vh * S + c0 + vv))
                    : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const float* g = gm + ((int64_t) ch * HV + vh) * GCH;
        gam = __ldg(g + tt);
        glast = __ldg(g + GCH - 1);
        bet = tt < n ? __ldg(beta + (c_t0 + tt) * HV + vh) : 0.0f;
    };
    float s[4][4];
#pragma unroll
    for (int x = 0; x < 4; ++x) {
#pragma unroll
        for (int y = 0; y < 4; ++y) s[x][y] = state[(size_t) (4 * rq + x) * rs + (size_t) vh * S + c0 + vv + y];
        *reinterpret_cast<float4*>(&ss[(4 * rq + x) * GDV + vv]) = make_float4(s[x][0], s[x][1], s[x][2], s[x][3]);
    }
    const float out_scale = rsqrtf((float) S);
    float4 v4;
    float gam, bet, glast;
    {
        const int n0 = (int) min((int64_t) GCH, T - t0);
        stage_kq(t0, n0, kbuf, false);
        stage_kq(t0, n0, sq, true);
        gdn_cp_commit();
        load_vgb(0, t0, n0, v4, gam, bet, glast);
        gdn_cp_wait_all();
        __syncthreads();
    }
    for (int ch = 0; ch < nch; ++ch) {
        const int64_t c_t0 = t0 + (int64_t) ch * GCH;
        const int n = (int) min((int64_t) GCH, T - c_t0);
        const bool more = ch + 1 < nch;
        const int64_t n_t0 = c_t0 + GCH;
        const int nn = more ? (int) min((int64_t) GCH, T - n_t0) : 0;
        const float* sk = kbuf + (ch & 1) * GCH * GKP;
        // this token's T and P rows into registers now: they land while the first product runs
        float tr[GCH], pr[GCH];
        {
            const float4* T4 = reinterpret_cast<const float4*>(tm + (((int64_t) ch * HV + vh) * GCH + tt) * GCH);
            const float4* P4 = reinterpret_cast<const float4*>(pm + (((int64_t) ch * HV + vh) * GCH + tt) * GCH);
#pragma unroll
            for (int i = 0; i < GCH / 4; ++i) {
                const float4 a = __ldg(T4 + i), b = __ldg(P4 + i);
                tr[4 * i] = a.x; tr[4 * i + 1] = a.y; tr[4 * i + 2] = a.z; tr[4 * i + 3] = a.w;
                pr[4 * i] = b.x; pr[4 * i + 1] = b.y; pr[4 * i + 2] = b.z; pr[4 * i + 3] = b.w;
            }
        }
        float4 nv4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float ngam = 0.0f, nbet = 0.0f, nglast = 0.0f;
        if (more) {   // the next chunk's v, gamma and beta; the one after into L2 (its k and q: after barrier A)
            load_vgb(ch + 1, n_t0, nn, nv4, ngam, nbet, nglast);
            const int64_t f_t0 = n_t0 + GCH;
            const int t = threadIdx.x >> 3, l = threadIdx.x & 7;
            if (t < GCH && f_t0 + t < T) gdn_pf_l2(h + (f_t0 + t) * C + (l < 4 ? HK * S : 0) + qh * S + 32 * (l & 3));
            if (tid < GCH) {
                gdn_pf_l2(tm + (((int64_t) (ch + 1) * HV + vh) * GCH + tid) * GCH);
                gdn_pf_l2(pm + (((int64_t) (ch + 1) * HV + vh) * GCH + tid) * GCH);
            }
        }
        // X = K S0 and Q S0 for token tt, then Y = beta (V - exp(gamma) X)
        float x[4] = {0.f, 0.f, 0.f, 0.f}, qs[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
        for (int r = 0; r < S; r += 4) {
            const float4 k4 = *reinterpret_cast<const float4*>(&sk[tt * GKP + r]);
            const float4 q4 = *reinterpret_cast<const float4*>(&sq[tt * GKP + r]);
            const float kk[4] = {k4.x, k4.y, k4.z, k4.w}, qq[4] = {q4.x, q4.y, q4.z, q4.w};
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const float4 s4 = *reinterpret_cast<const float4*>(&ss[(r + jj) * GDV + vv]);
                x[0] = fmaf(kk[jj], s4.x, x[0]); x[1] = fmaf(kk[jj], s4.y, x[1]);
                x[2] = fmaf(kk[jj], s4.z, x[2]); x[3] = fmaf(kk[jj], s4.w, x[3]);
                qs[0] = fmaf(qq[jj], s4.x, qs[0]); qs[1] = fmaf(qq[jj], s4.y, qs[1]);
                qs[2] = fmaf(qq[jj], s4.z, qs[2]); qs[3] = fmaf(qq[jj], s4.w, qs[3]);
            }
        }
        const float e = expf(gam);
        *reinterpret_cast<float4*>(&sy[tt * GDV + vv]) = make_float4(bet * (v4.x - e * x[0]), bet * (v4.y - e * x[1]),
                                                                     bet * (v4.z - e * x[2]), bet * (v4.w - e * x[3]));
        __syncthreads();   // (A) Y complete; every read of q done
        if (more) {        // the next chunk's k into the other buffer, its q into the single q buffer
            stage_kq(n_t0, nn, kbuf + ((ch + 1) & 1) * GCH * GKP, false);
            stage_kq(n_t0, nn, sq, true);
            gdn_cp_commit();
        }
        // D = T Y (T is zero above its diagonal), and D decayed to the chunk's end for the update
        float d[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < GCH; ++i) {
            const float4 y4 = *reinterpret_cast<const float4*>(&sy[i * GDV + vv]);
            d[0] = fmaf(tr[i], y4.x, d[0]); d[1] = fmaf(tr[i], y4.y, d[1]);
            d[2] = fmaf(tr[i], y4.z, d[2]); d[3] = fmaf(tr[i], y4.w, d[3]);
        }
        const float f = expf(glast - gam);
        *reinterpret_cast<float4*>(&sd[tt * GDV + vv]) = make_float4(d[0], d[1], d[2], d[3]);
        *reinterpret_cast<float4*>(&sdd[tt * GDV + vv]) = make_float4(f * d[0], f * d[1], f * d[2], f * d[3]);
        __syncthreads();   // (B) D complete
        // o = exp(gamma) Q S0 + P D (P is zero above its diagonal), the output
        float o[4] = {e * qs[0], e * qs[1], e * qs[2], e * qs[3]};
#pragma unroll
        for (int i = 0; i < GCH; ++i) {
            const float4 d4 = *reinterpret_cast<const float4*>(&sd[i * GDV + vv]);
            o[0] = fmaf(pr[i], d4.x, o[0]); o[1] = fmaf(pr[i], d4.y, o[1]);
            o[2] = fmaf(pr[i], d4.z, o[2]); o[3] = fmaf(pr[i], d4.w, o[3]);
        }
        if (tt < n)
            *reinterpret_cast<float4*>(oc_out + (c_t0 + tt) * HV * S + vh * S + c0 + vv) =
                make_float4(o[0] * out_scale, o[1] * out_scale, o[2] * out_scale, o[3] * out_scale);
        // S = exp(gamma_last) S0 + K^T (decayed D)
        const float gL = expf(glast);
#pragma unroll
        for (int xx = 0; xx < 4; ++xx)
#pragma unroll
            for (int y = 0; y < 4; ++y) s[xx][y] *= gL;
#pragma unroll 4
        for (int t = 0; t < GCH; ++t) {
            const float4 k4 = *reinterpret_cast<const float4*>(&sk[t * GKP + 4 * rq]);
            const float4 d4 = *reinterpret_cast<const float4*>(&sdd[t * GDV + vv]);
            const float kx[4] = {k4.x, k4.y, k4.z, k4.w}, dy[4] = {d4.x, d4.y, d4.z, d4.w};
#pragma unroll
            for (int xx = 0; xx < 4; ++xx)
#pragma unroll
                for (int y = 0; y < 4; ++y) s[xx][y] = fmaf(kx[xx], dy[y], s[xx][y]);
        }
#pragma unroll
        for (int xx = 0; xx < 4; ++xx)
            *reinterpret_cast<float4*>(&ss[(4 * rq + xx) * GDV + vv]) = make_float4(s[xx][0], s[xx][1], s[xx][2], s[xx][3]);
        v4 = nv4; gam = ngam; bet = nbet; glast = nglast;
        gdn_cp_wait_all();
        __syncthreads();   // (C) the new ss, the next k and q landed; every read of sd, sdd and this k buffer done
    }
#pragma unroll
    for (int xx = 0; xx < 4; ++xx)
#pragma unroll
        for (int y = 0; y < 4; ++y) state[(size_t) (4 * rq + xx) * rs + (size_t) vh * S + c0 + vv + y] = s[xx][y];
}
// the same scan on the tensor cores (STRATA_GDN_TC=1, gdn_rec_tc below); src/prefill/gdn_rec_parity.cu includes the
// same text
#include "gdn_tc.cuh"

// The scratch, one a device: allocated on first use at its largest (GSB / GCH chunks of T, P and gamma: 25.6 MB) and
// kept (a device runs its prompt on one stream).  A stream-ordered allocation and free per call, even from a pool that
// keeps its memory, cost host time the GPU waited for (a 2K prompt's recurrence took 78 ms against 34 ms for
// gdn_rec_kh_kernel).
float* gdn_chunk_scratch() {
    static float* bufs[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return nullptr; }
    if (bufs[dev] == nullptr) {
        const size_t n = (size_t) (GSB / GCH) * HV * (2 * GCH * GCH + GCH);
        if (cudaMalloc((void**) &bufs[dev], n * sizeof(float)) != cudaSuccess) {
            cudaGetLastError();
            bufs[dev] = nullptr;
        }
    }
    return bufs[dev];
}

// The whole recurrence of T tokens in super-blocks of GSB: a prep and a scan launch each, the scratch (T, P and gamma
// of a super-block's chunks: up to GSB / GCH x 48 heads x (2 x 32 x 32 + 32) floats, 25.6 MB) from gdn_chunk_scratch.
// y: the [T][HV][S] output.  An error return means nothing was launched (the caller takes another kernel): the scratch
// could not be had, or the card has fewer than 128 SMs (a second wave of the scan would double it) or too little
// shared memory for it (Turing).
cudaError_t gdn_rec_chunked(float* state, const float* h, const float* gate, const float* beta, float* y, int64_t T,
                            cudaStream_t s) {
    if (T <= 0) return cudaSuccess;
    {   // sm_80+ (cp.async), the scan's 128 blocks one an SM in one wave, and its 93.7 KB of shared memory (else:
        // another kernel)
        int dev = 0, major = 0, sms = 0, smem = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess) {
            cudaGetLastError();
            return cudaErrorNotSupported;
        }
        // STRATA_GDN_CHUNKED=2 skips the SM-count test (measurement on a card under 128 SMs: the scan then runs in
        // several waves and is expected to be slower than the default kernel; the architecture and shared-memory
        // tests stay)
        static const bool any_sms_env = [] { const char* v = std::getenv("STRATA_GDN_CHUNKED"); return v != nullptr && std::atoi(v) >= 2; }();
        const GdnProbe* pr = gdn_probe_get();   // tests only: the same as =2, per call
        const bool any_sms = any_sms_env || (pr != nullptr && pr->any_sms);
        if (strata::cc_major_of(major) < 8 || (!any_sms && sms < HK * (S / GDV)) || (size_t) strata::smem_optin_of(smem) < kChunkScanSmem)
            return cudaErrorNotSupported;
    }
    float* scratch = gdn_chunk_scratch();
    if (scratch == nullptr) return cudaErrorMemoryAllocation;
    // the scratch segment: a test may ask for a smaller one than GSB (a multiple of GCH); production always GSB
    int64_t gsb = GSB;
    if (const GdnProbe* pr = gdn_probe_get())
        if (pr->gsb >= GCH && pr->gsb < GSB) gsb = (pr->gsb / GCH) * GCH;
    const int nb = (int) ((std::min<int64_t>(gsb, T) + GCH - 1) / GCH);
    const size_t tsz = (size_t) nb * HV * GCH * GCH;
    float *tm = scratch, *pm = tm + tsz, *gm = pm + tsz;
    {   // once a device
        static bool attrs[64] = {};
        int dev = 0;
        cudaGetDevice(&dev);
        if (dev >= 0 && dev < 64 && !attrs[dev]) {
            cudaFuncSetAttribute(gdn_chunk_prep_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kChunkPrepSmem);
            cudaFuncSetAttribute(gdn_chunk_scan_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kChunkScanSmem);
            attrs[dev] = true;
        }
    }
    for (int64_t t0 = 0; t0 < T; t0 += gsb) {
        const int nch = (int) ((std::min<int64_t>(gsb, T - t0) + GCH - 1) / GCH);
        gdn_chunk_prep_kernel<<<dim3((unsigned) nch, HK), 256, kChunkPrepSmem, s>>>(h, gate, beta, tm, pm, gm, t0, T);
        gdn_chunk_scan_kernel<<<dim3(HK, S / GDV), 128 * VPK, kChunkScanSmem, s>>>(state, h, beta, tm, pm, gm, y, t0, nch, T);
    }
    return cudaSuccess;   // launch errors surface at the caller's check; an error above means nothing ran
}
// STRATA_GDN_TC=1: gdn_rec_chunked with the scan on the tensor cores (gdn_tc_scan_kernel, gdn_tc.cuh) and the prep
// that also writes its f16 images (gdn_tc_prep_kernel, scratch gdn_tc_scratch), 48 blocks of 256 threads in one wave.  An error return means nothing was launched: the card is below
// sm_80, has fewer than 48 SMs, or less than 122 KB of shared memory a block.
// value tiles a head for the split tensor-core scan: STRATA_GDN_TC_TILES=1|2 (or the probe's tc_tiles) asks; 1 otherwise
int gdn_tc_tiles() {
    const GdnProbe* pr = gdn_probe_get();
    if (pr != nullptr && pr->tc_tiles > 0) return pr->tc_tiles >= 2 ? 2 : 1;
    static const int e = [] { const char* v = std::getenv("STRATA_GDN_TC_TILES"); return v ? std::atoi(v) : 0; }();
    if (e > 0) return e >= 2 ? 2 : 1;
    // automatic: two tiles when all 2 x 48 blocks fit the card in one wave (measured: PRO 5000 / 110 SMs rec 1.47x; A6000 / 84 SMs
    // is 1.25x SLOWER with two tiles, its 96 blocks need a second wave)
    static int sms_of[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return 1; }
    if (sms_of[dev] == 0) {
        int n = 0;
        if (cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) { cudaGetLastError(); n = 1; }
        sms_of[dev] = n;
    }
    return sms_of[dev] >= 2 * HV ? 2 : 1;
}
// the output norm's operands, when the split scan fuses it (see gdn_tc_scan_kernel NORM)
struct GdnTcNorm {
    const float* z;
    const float* gamma;
    float eps;
    uint16_t* y16;
    int64_t ld16;
};
cudaError_t gdn_rec_tc(float* state, const float* h, const float* gate, const float* beta, float* y, int64_t T,
                       cudaStream_t s, bool split = false, int np = 3, const GdnTcNorm* nm = nullptr, int tiles = 1) {
    if (T <= 0) return cudaSuccess;
    {
        int dev = 0, major = 0, sms = 0, smem = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess) {
            cudaGetLastError();
            return cudaErrorNotSupported;
        }
        const GdnProbe* pr = gdn_probe_get();   // tests only: fewer SMs than value heads run the 48 blocks in several waves
        const bool any_sms = pr != nullptr && pr->any_sms;
        if (strata::cc_major_of(major) < 8 || (!any_sms && sms < HV) ||
            (size_t) strata::smem_optin_of(smem) < (split ? (nm ? kTcScanSmemSplitNorm : kTcScanSmemSplit) : kTcScanSmem<false>))
            return cudaErrorNotSupported;
    }
    const GtcScratch sc = gdn_tc_scratch();
    if (sc.ti == nullptr) return cudaErrorMemoryAllocation;
    {   // once a device
        static bool attrs[3][64] = {};
        const int ai = nm && split ? 2 : (split ? 1 : 0);
        int dev = 0;
        cudaGetDevice(&dev);
        if (dev >= 0 && dev < 64 && !attrs[ai][dev]) {
            if (split) cudaFuncSetAttribute(gdn_tc_scan_kernel<false, true, 3, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kTcScanSmemSplit);
            if (ai == 2) {
                cudaFuncSetAttribute(gdn_tc_scan_kernel<true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kTcScanSmemSplitNorm);
                cudaFuncSetAttribute(gdn_tc_scan_kernel<true, true, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kTcScanSmemSplitNorm);
                cudaFuncSetAttribute(gdn_tc_scan_kernel<true, true, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kTcScanSmemSplitNorm);
            } else if (split) {
                cudaFuncSetAttribute(gdn_tc_scan_kernel<false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kTcScanSmemSplit);
                cudaFuncSetAttribute(gdn_tc_scan_kernel<false, true, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kTcScanSmemSplit);
                cudaFuncSetAttribute(gdn_tc_scan_kernel<false, true, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kTcScanSmemSplit);
            }
            else cudaFuncSetAttribute(gdn_tc_scan_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kTcScanSmem<false>);
            attrs[ai][dev] = true;
        }
    }
    for (int64_t t0 = 0; t0 < T; t0 += GSB) {
        const int nch = (int) ((std::min<int64_t>(GSB, T - t0) + GCH - 1) / GCH);
        gdn_tc_prep_kernel<<<dim3((unsigned) nch, HK), 256, kTcPrepSmem, s>>>(h, gate, beta, sc.ti, sc.te, sc.sc, sc.kq, sc.pi, t0, T);
        if (tiles == 2 && split && !nm && np == 3) {
            gdn_tc_scan_kernel<false, true, 3, 2><<<HV * 2, GTC_CW * 16, kTcScanSmemSplit, s>>>(state, h, sc.ti, sc.te, sc.sc, sc.kq, sc.pi, y, t0, nch, T);
        } else if (nm && split) {
#define STRATA_TCN(NPV) gdn_tc_scan_kernel<true, true, NPV><<<HV, GTC_CW * 32, kTcScanSmemSplitNorm, s>>>(state, h, sc.ti, sc.te, sc.sc, sc.kq, sc.pi, y, t0, nch, T, nm->z, nm->gamma, nm->eps, nm->y16, nm->ld16)
            if (np == 2) STRATA_TCN(2);
            else if (np == 1) STRATA_TCN(1);
            else STRATA_TCN(3);
#undef STRATA_TCN
        } else if (split && np == 2)
            gdn_tc_scan_kernel<false, true, 2><<<HV, GTC_CW * 32, kTcScanSmemSplit, s>>>(state, h, sc.ti, sc.te, sc.sc, sc.kq, sc.pi, y, t0, nch, T);
        else if (split && np == 1)
            gdn_tc_scan_kernel<false, true, 1><<<HV, GTC_CW * 32, kTcScanSmemSplit, s>>>(state, h, sc.ti, sc.te, sc.sc, sc.kq, sc.pi, y, t0, nch, T);
        else if (split)
            gdn_tc_scan_kernel<false, true><<<HV, GTC_CW * 32, kTcScanSmemSplit, s>>>(state, h, sc.ti, sc.te, sc.sc, sc.kq, sc.pi, y, t0, nch, T);
        else
            gdn_tc_scan_kernel<false><<<HV, GTC_CW * 32, kTcScanSmem<false>, s>>>(state, h, sc.ti, sc.te, sc.sc, sc.kq, sc.pi, y, t0, nch, T);
    }
    return cudaSuccess;
}
// Why gdn_rec_chunked (tc false) or gdn_rec_tc (tc true) launched nothing, in words: the same tests asked again.  Only
// the gdn_probe.hpp `reason` uses it (a test says what it timed); the dispatch itself never calls it.
const char* gdn_chunk_why(bool tc, cudaError_t e, bool any_sms, bool split = false) {
    static thread_local char buf[200];
    if (e == cudaErrorMemoryAllocation) return "scratch allocation failed";
    int dev = 0, major = 0, sms = 0, smem = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess) {
        cudaGetLastError();
        return "device attributes unavailable";
    }
    const int cc = strata::cc_major_of(major);
    const size_t need = tc ? (split ? kTcScanSmemSplit : (size_t) kTcScanSmem<false>) : kChunkScanSmem;
    const int sm_need = tc ? HV : HK * (S / GDV);
    if (cc < 8) std::snprintf(buf, sizeof buf, "compute capability %d < 8 (cp.async)", cc);
    else if (!tc && !any_sms && sms < sm_need)
        std::snprintf(buf, sizeof buf, "%d SMs < %d: the scan would run in several waves (STRATA_GDN_CHUNKED=2 bypasses)", sms, sm_need);
    else if (tc && !any_sms && sms < sm_need) std::snprintf(buf, sizeof buf, "%d SMs < %d (one block per value head)", sms, sm_need);
    else if ((size_t) strata::smem_optin_of(smem) < need)
        std::snprintf(buf, sizeof buf, "shared memory per block %d B < %zu B needed", strata::smem_optin_of(smem), need);
    else std::snprintf(buf, sizeof buf, "not supported (%s)", cudaGetErrorString(e));
    return buf;
}
#endif
#if defined(__HIPCC__)   // the Aurora (gfx11) recurrence kernels: AMD builds only
// Aurora (S23): the recurrence with the column-split grid of gdn_rec_cols_pipe_kernel (four 32-column blocks per value
// head) but FOUR LANES per column, lane g holding the column's state rows 32g .. 32g + 31 in registers - exactly the
// row group the old kernel's thread (col, rg = g) held - so a token needs no __syncthreads: the four kv / o partials
// meet through wave shuffles and are added in the old order red[0] + red[1] + red[2] + red[3] (the same bits;
// tests/hip/gdn_rec_head.cpp).  The q/k rows, v, gate and beta are staged in LDS HT tokens at a time, the next chunk
// loaded into registers while this one computes, and a token's k and q rows are read into registers before its
// products (one LDS latency per token, not one per load).  The old kernels took ~4 barriers per token and ~3.5 us
// per token and layer on the Radeon 8060S; the output norm runs after it in gdn_out_norm_warp_kernel.
// STRATA_GDN_HEAD=0: the old kernels.
constexpr int HT = 4;                           // tokens per staged chunk (LDS ~10 KB: all 192 blocks resident)
constexpr int QCB = 32;                         // columns per block
constexpr int QTH = QCB * 4;                    // threads per block: 4 lanes per column
constexpr int QRS = 36;                         // staged row-group stride (32 rows + 4 pad: the 4 groups' reads
                                                // land in different LDS banks)
constexpr int QPF = HT * 2 * S / 4 / QTH;       // float4 of q/k each thread prefetches per chunk (2)
__global__ void __launch_bounds__(QTH) gdn_rec_quad_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                           const float* __restrict__ gate, const float* __restrict__ beta,
                                                           float* __restrict__ oc_out, int64_t T) {
    __shared__ __align__(16) float sqk[2][HT][2][4 * QRS];   // [buffer][token][q | k][group][row (padded)]
    __shared__ float sgb[2][2][HT];                         // [buffer][gate | beta][token]
    __shared__ float sv[2][HT][QCB];                        // [buffer][token][column] v
    const int head = blockIdx.x / (S / QCB), cb = blockIdx.x % (S / QCB);
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int g = lane >> 3, cl = wave * 8 + (lane & 7), col = cb * QCB + cl;   // row group, column in block / head
    const int qh = head % HK;
    float s[32];
    {
        const float* base = state + ((size_t) (g * 32) * HV + head) * S + col;
#pragma unroll
        for (int r = 0; r < 32; ++r) s[r] = base[(size_t) r * HV * S];
    }
    float4 pf[QPF];
    float pv[HT], pgb = 0.0f;
    // chunk c's inputs into registers: q/k (float4 i of the chunk's [token][q|k][S/4] block), the column's v (lanes of
    // group 0) and gate (threads 0..HT-1) / beta (HT..2HT-1) of token `tid % HT`
    auto fetch = [&](int64_t c) {
        const int64_t t0 = c * HT;
#pragma unroll
        for (int u = 0; u < QPF; ++u) {
            const int i = tid + u * QTH;
            const int tt = i / (2 * S / 4), rem = i % (2 * S / 4), part = rem / (S / 4), r4 = rem % (S / 4);
            const int64_t t = t0 + tt;
            pf[u] = t < T ? *reinterpret_cast<const float4*>(h + t * C + part * HK * S + qh * S + r4 * 4)
                          : make_float4(0.f, 0.f, 0.f, 0.f);
        }
        if (g == 0) {
#pragma unroll
            for (int tt = 0; tt < HT; ++tt) {
                const int64_t t = t0 + tt;
                pv[tt] = t < T ? h[t * C + 2 * HK * S + head * S + col] : 0.0f;
            }
        }
        if (tid < 2 * HT) {
            const int64_t t = t0 + (tid % HT);
            pgb = t < T ? (tid < HT ? gate : beta)[t * HV + head] : 0.0f;
        }
    };
    const float rs_s = rsqrtf((float) S);
    const int src0 = lane & 7;                       // lane of group 0 for this column; group j is src0 + 8 j
    const int64_t nch = (T + HT - 1) / HT;
    if (nch > 0) fetch(0);
    for (int64_t c = 0; c < nch; ++c) {
        const int b = (int) (c & 1);
        if (g == 0) {
#pragma unroll
            for (int tt = 0; tt < HT; ++tt) sv[b][tt][cl] = pv[tt];
        }
#pragma unroll
        for (int u = 0; u < QPF; ++u) {
            const int i = tid + u * QTH;
            const int tt = i / (2 * S / 4), rem = i % (2 * S / 4), part = rem / (S / 4), r4 = rem % (S / 4);
            const int row = r4 * 4;
            *reinterpret_cast<float4*>(&sqk[b][tt][part][(row >> 5) * QRS + (row & 31)]) = pf[u];
        }
        if (tid < 2 * HT) sgb[b][tid / HT][tid % HT] = pgb;
        __syncthreads();                               // chunk c staged (buffer b was last read two chunks ago)
        if (c + 1 < nch) fetch(c + 1);
        const int64_t t0 = c * HT;
        const int nt = (int) (T - t0 < HT ? T - t0 : HT);
        for (int tt = 0; tt < nt; ++tt) {
            float k[32], q[32];
            {
                const float4* k4 = reinterpret_cast<const float4*>(&sqk[b][tt][1][g * QRS]);
                const float4* q4 = reinterpret_cast<const float4*>(&sqk[b][tt][0][g * QRS]);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float4 kk = k4[j], qq = q4[j];
                    k[4 * j] = kk.x; k[4 * j + 1] = kk.y; k[4 * j + 2] = kk.z; k[4 * j + 3] = kk.w;
                    q[4 * j] = qq.x; q[4 * j + 1] = qq.y; q[4 * j + 2] = qq.z; q[4 * j + 3] = qq.w;
                }
            }
            const float g_exp = __expf(sgb[b][0][tt]);
            float kv = 0.0f;
#pragma unroll
            for (int r = 0; r < 32; ++r) kv = fmaf(s[r], k[r], kv);
            const float p0 = __shfl_sync(0xffffffffu, kv, src0), p1 = __shfl_sync(0xffffffffu, kv, src0 + 8),
                        p2 = __shfl_sync(0xffffffffu, kv, src0 + 16), p3 = __shfl_sync(0xffffffffu, kv, src0 + 24);
            const float kv_col = p0 + p1 + p2 + p3;
            const float cvt = sv[b][tt][cl], cbt = sgb[b][1][tt];
            const float delta = (cvt - g_exp * kv_col) * cbt;
            float o = 0.0f;
#pragma unroll
            for (int r = 0; r < 32; ++r) {
                s[r] = fmaf(g_exp, s[r], k[r] * delta);
                o = fmaf(s[r], q[r], o);
            }
            const float o0 = __shfl_sync(0xffffffffu, o, src0), o1 = __shfl_sync(0xffffffffu, o, src0 + 8),
                        o2 = __shfl_sync(0xffffffffu, o, src0 + 16), o3 = __shfl_sync(0xffffffffu, o, src0 + 24);
            if (g == 0) oc_out[(t0 + tt) * HV * S + head * S + col] = (o0 + o1 + o2 + o3) * rs_s;
        }
    }
    {
        float* base = state + ((size_t) (g * 32) * HV + head) * S + col;
#pragma unroll
        for (int r = 0; r < 32; ++r) base[(size_t) r * HV * S] = s[r];
    }
}
// Aurora (S23, opt-in STRATA_GDN_PP=1): gdn_rec_quad_kernel with the same arithmetic in the same order, two changes in
// how it is compiled.  (1) The VGPR count: the 32 state-row addresses of the first loads were kept live across the
// whole walk to be reused by the final stores (64 VGPRs, 196 in all = 7 waves a SIMD for the 192 blocks of 4 waves
// that need 9.6): the stores' lane offset goes through an asm so the addresses are made again, 133 VGPRs, all blocks
// resident.  (2) A full chunk's four tokens run with the state alternating between two register sets (the update
// writes the other set), so the compiler emits no 32 register copies after the update.
__global__ void __launch_bounds__(QTH) gdn_rec_quad_pp_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                           const float* __restrict__ gate, const float* __restrict__ beta,
                                                           float* __restrict__ oc_out, int64_t T) {
    __shared__ __align__(16) float sqk[2][HT][2][4 * QRS];   // [buffer][token][q | k][group][row (padded)]
    __shared__ float sgb[2][2][HT];                         // [buffer][gate | beta][token]
    __shared__ float sv[2][HT][QCB];                        // [buffer][token][column] v
    const int head = blockIdx.x / (S / QCB), cb = blockIdx.x % (S / QCB);
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int g = lane >> 3, cl = wave * 8 + (lane & 7), col = cb * QCB + cl;   // row group, column in block / head
    const int qh = head % HK;
    float s[32];
    {
        const float* sb = state + (size_t) head * S;
        const uint32_t voff = (uint32_t) ((g * 32) * HV * S + col);
#pragma unroll
        for (int r = 0; r < 32; ++r) s[r] = sb[(size_t) r * HV * S + voff];
    }
    float4 pf[QPF];
    float pv[HT], pgb = 0.0f;
    // chunk c's inputs into registers: q/k (float4 i of the chunk's [token][q|k][S/4] block), the column's v (lanes of
    // group 0) and gate (threads 0..HT-1) / beta (HT..2HT-1) of token `tid % HT`
    auto fetch = [&](int64_t c) {
        const int64_t t0 = c * HT;
#pragma unroll
        for (int u = 0; u < QPF; ++u) {
            const int i = tid + u * QTH;
            const int tt = i / (2 * S / 4), rem = i % (2 * S / 4), part = rem / (S / 4), r4 = rem % (S / 4);
            const int64_t t = t0 + tt;
            pf[u] = t < T ? *reinterpret_cast<const float4*>(h + t * C + part * HK * S + qh * S + r4 * 4)
                          : make_float4(0.f, 0.f, 0.f, 0.f);
        }
        if (g == 0) {
#pragma unroll
            for (int tt = 0; tt < HT; ++tt) {
                const int64_t t = t0 + tt;
                pv[tt] = t < T ? h[t * C + 2 * HK * S + head * S + col] : 0.0f;
            }
        }
        if (tid < 2 * HT) {
            const int64_t t = t0 + (tid % HT);
            pgb = t < T ? (tid < HT ? gate : beta)[t * HV + head] : 0.0f;
        }
    };
    const float rs_s = rsqrtf((float) S);
    const int src0 = lane & 7;                       // lane of group 0 for this column; group j is src0 + 8 j
    const int64_t nch = (T + HT - 1) / HT;
    if (nch > 0) fetch(0);
    for (int64_t c = 0; c < nch; ++c) {
        const int b = (int) (c & 1);
        if (g == 0) {
#pragma unroll
            for (int tt = 0; tt < HT; ++tt) sv[b][tt][cl] = pv[tt];
        }
#pragma unroll
        for (int u = 0; u < QPF; ++u) {
            const int i = tid + u * QTH;
            const int tt = i / (2 * S / 4), rem = i % (2 * S / 4), part = rem / (S / 4), r4 = rem % (S / 4);
            const int row = r4 * 4;
            *reinterpret_cast<float4*>(&sqk[b][tt][part][(row >> 5) * QRS + (row & 31)]) = pf[u];
        }
        if (tid < 2 * HT) sgb[b][tid / HT][tid % HT] = pgb;
        __syncthreads();                               // chunk c staged (buffer b was last read two chunks ago)
        if (c + 1 < nch) fetch(c + 1);
        const int64_t t0 = c * HT;
        const int nt = (int) (T - t0 < HT ? T - t0 : HT);
        auto token = [&](int tt, float (&sin)[32], float (&sout)[32]) {
            float k[32], q[32];
            {
                const float4* k4 = reinterpret_cast<const float4*>(&sqk[b][tt][1][g * QRS]);
                const float4* q4 = reinterpret_cast<const float4*>(&sqk[b][tt][0][g * QRS]);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float4 kk = k4[j], qq = q4[j];
                    k[4 * j] = kk.x; k[4 * j + 1] = kk.y; k[4 * j + 2] = kk.z; k[4 * j + 3] = kk.w;
                    q[4 * j] = qq.x; q[4 * j + 1] = qq.y; q[4 * j + 2] = qq.z; q[4 * j + 3] = qq.w;
                }
            }
            const float g_exp = __expf(sgb[b][0][tt]);
            float kv = 0.0f;
#pragma unroll
            for (int r = 0; r < 32; ++r) kv = fmaf(sin[r], k[r], kv);
            const float p0 = __shfl_sync(0xffffffffu, kv, src0), p1 = __shfl_sync(0xffffffffu, kv, src0 + 8),
                        p2 = __shfl_sync(0xffffffffu, kv, src0 + 16), p3 = __shfl_sync(0xffffffffu, kv, src0 + 24);
            const float kv_col = p0 + p1 + p2 + p3;
            const float cvt = sv[b][tt][cl], cbt = sgb[b][1][tt];
            const float delta = (cvt - g_exp * kv_col) * cbt;
            float o = 0.0f;
#pragma unroll
            for (int r = 0; r < 32; ++r) {
                sout[r] = fmaf(g_exp, sin[r], k[r] * delta);
                o = fmaf(sout[r], q[r], o);
            }
            const float o0 = __shfl_sync(0xffffffffu, o, src0), o1 = __shfl_sync(0xffffffffu, o, src0 + 8),
                        o2 = __shfl_sync(0xffffffffu, o, src0 + 16), o3 = __shfl_sync(0xffffffffu, o, src0 + 24);
            if (g == 0) oc_out[(t0 + tt) * HV * S + head * S + col] = (o0 + o1 + o2 + o3) * rs_s;
        };
        if (nt == HT) {   // the state alternates between two register sets: no copy after the update
            float s2[32];
            token(0, s, s2);
            token(1, s2, s);
            token(2, s, s2);
            token(3, s2, s);
        } else {
            for (int tt = 0; tt < nt; ++tt) {
                float s2[32];
                token(tt, s, s2);
#pragma unroll
                for (int r = 0; r < 32; ++r) s[r] = s2[r];
            }
        }
    }
    {
        float* sb = state + (size_t) head * S;
        uint32_t voff = (uint32_t) ((g * 32) * HV * S + col);
        asm volatile("" : "+v"(voff));   // (so the loads' 32 addresses are not kept alive across the whole loop)
#pragma unroll
        for (int r = 0; r < 32; ++r) sb[(size_t) r * HV * S + voff] = s[r];
    }
}

// Aurora (S23, opt-in STRATA_GDN_PP=2): gdn_rec_quad_kernel with two columns per lane (a wave: 4 row groups x 8 lanes x
// 2 columns = 16 columns; a block of 4 waves 64 columns, 96 blocks).  A lane's two columns share the k / q rows it
// reads from shared memory (half the LDS reads and shuffles per column, the kernel's limit) and give each wave two
// independent dependency chains; the 384 waves fit in one round at 230 VGPRs.  Per column the same arithmetic in
// the same order (the same bits as gdn_rec_quad_kernel).  The store of the state goes through an asm for the reason
// given at gdn_rec_quad_pp_kernel.

// ---- C2: two columns per lane (a wave: 4 row groups x 8 lanes x 2 columns = 16 columns; a block of 4 waves 64
// columns).  A lane's two columns share the k / q rows it reads from LDS (half the LDS reads and shuffles per column
// and two independent dependency chains in each wave).  Per column the same arithmetic in the same order.
constexpr int C2CB = 64;                          // columns per block
constexpr int C2PF = HT * 2 * S / 4 / QTH;        // float4 of q/k each thread prefetches per chunk (2)
__global__ void __launch_bounds__(QTH) gdn_rec_quad2c_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                             const float* __restrict__ gate, const float* __restrict__ beta,
                                                             float* __restrict__ oc_out, int64_t T) {
    __shared__ __align__(16) float sqk[2][HT][2][4 * QRS];
    __shared__ float sgb[2][2][HT];
    __shared__ float sv[2][HT][C2CB];
    const int head = blockIdx.x / (S / C2CB), cb = blockIdx.x % (S / C2CB);
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int g = lane >> 3, cl0 = wave * 16 + (lane & 7), cl1 = cl0 + 8;   // the lane's two columns in the block
    const int col0 = cb * C2CB + cl0, col1 = cb * C2CB + cl1;
    const int qh = head % HK;
    float s0[32], s1[32];
    {
        const float* sb = state + (size_t) head * S;
        const uint32_t voff = (uint32_t) ((g * 32) * HV * S);
#pragma unroll
        for (int r = 0; r < 32; ++r) { s0[r] = sb[(size_t) r * HV * S + voff + col0]; s1[r] = sb[(size_t) r * HV * S + voff + col1]; }
    }
    float4 pf[C2PF];
    float pv[2][HT], pgb = 0.0f;
    auto fetch = [&](int64_t c) {
        const int64_t t0 = c * HT;
#pragma unroll
        for (int u = 0; u < C2PF; ++u) {
            const int i = tid + u * QTH;
            const int tt = i / (2 * S / 4), rem = i % (2 * S / 4), part = rem / (S / 4), r4 = rem % (S / 4);
            const int64_t t = t0 + tt;
            pf[u] = t < T ? *reinterpret_cast<const float4*>(h + t * C + part * HK * S + qh * S + r4 * 4)
                          : make_float4(0.f, 0.f, 0.f, 0.f);
        }
        if (g == 0) {
#pragma unroll
            for (int tt = 0; tt < HT; ++tt) {
                const int64_t t = t0 + tt;
                pv[0][tt] = t < T ? h[t * C + 2 * HK * S + head * S + col0] : 0.0f;
                pv[1][tt] = t < T ? h[t * C + 2 * HK * S + head * S + col1] : 0.0f;
            }
        }
        if (tid < 2 * HT) {
            const int64_t t = t0 + (tid % HT);
            pgb = t < T ? (tid < HT ? gate : beta)[t * HV + head] : 0.0f;
        }
    };
    const float rs_s = rsqrtf((float) S);
    const int src0 = lane & 7;
    const int64_t nch = (T + HT - 1) / HT;
    if (nch > 0) fetch(0);
    for (int64_t c = 0; c < nch; ++c) {
        const int b = (int) (c & 1);
        if (g == 0) {
#pragma unroll
            for (int tt = 0; tt < HT; ++tt) { sv[b][tt][cl0] = pv[0][tt]; sv[b][tt][cl1] = pv[1][tt]; }
        }
#pragma unroll
        for (int u = 0; u < C2PF; ++u) {
            const int i = tid + u * QTH;
            const int tt = i / (2 * S / 4), rem = i % (2 * S / 4), part = rem / (S / 4), r4 = rem % (S / 4);
            const int row = r4 * 4;
            *reinterpret_cast<float4*>(&sqk[b][tt][part][(row >> 5) * QRS + (row & 31)]) = pf[u];
        }
        if (tid < 2 * HT) sgb[b][tid / HT][tid % HT] = pgb;
        __syncthreads();
        if (c + 1 < nch) fetch(c + 1);
        const int64_t t0 = c * HT;
        const int nt = (int) (T - t0 < HT ? T - t0 : HT);
        for (int tt = 0; tt < nt; ++tt) {
            float k[32], q[32];
            {
                const float4* k4 = reinterpret_cast<const float4*>(&sqk[b][tt][1][g * QRS]);
                const float4* q4 = reinterpret_cast<const float4*>(&sqk[b][tt][0][g * QRS]);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float4 kk = k4[j], qq = q4[j];
                    k[4 * j] = kk.x; k[4 * j + 1] = kk.y; k[4 * j + 2] = kk.z; k[4 * j + 3] = kk.w;
                    q[4 * j] = qq.x; q[4 * j + 1] = qq.y; q[4 * j + 2] = qq.z; q[4 * j + 3] = qq.w;
                }
            }
            const float g_exp = __expf(sgb[b][0][tt]);
            float kv0 = 0.0f, kv1 = 0.0f;
#pragma unroll
            for (int r = 0; r < 32; ++r) { kv0 = fmaf(s0[r], k[r], kv0); kv1 = fmaf(s1[r], k[r], kv1); }
            const float a0 = __shfl_sync(0xffffffffu, kv0, src0), a1 = __shfl_sync(0xffffffffu, kv0, src0 + 8),
                        a2 = __shfl_sync(0xffffffffu, kv0, src0 + 16), a3 = __shfl_sync(0xffffffffu, kv0, src0 + 24);
            const float b0 = __shfl_sync(0xffffffffu, kv1, src0), b1 = __shfl_sync(0xffffffffu, kv1, src0 + 8),
                        b2 = __shfl_sync(0xffffffffu, kv1, src0 + 16), b3 = __shfl_sync(0xffffffffu, kv1, src0 + 24);
            const float kvc0 = a0 + a1 + a2 + a3, kvc1 = b0 + b1 + b2 + b3;
            const float cbt = sgb[b][1][tt];
            const float d0 = (sv[b][tt][cl0] - g_exp * kvc0) * cbt, d1 = (sv[b][tt][cl1] - g_exp * kvc1) * cbt;
            float o0 = 0.0f, o1 = 0.0f;
#pragma unroll
            for (int r = 0; r < 32; ++r) {
                s0[r] = fmaf(g_exp, s0[r], k[r] * d0);
                s1[r] = fmaf(g_exp, s1[r], k[r] * d1);
                o0 = fmaf(s0[r], q[r], o0);
                o1 = fmaf(s1[r], q[r], o1);
            }
            const float x0 = __shfl_sync(0xffffffffu, o0, src0), x1 = __shfl_sync(0xffffffffu, o0, src0 + 8),
                        x2 = __shfl_sync(0xffffffffu, o0, src0 + 16), x3 = __shfl_sync(0xffffffffu, o0, src0 + 24);
            const float y0 = __shfl_sync(0xffffffffu, o1, src0), y1 = __shfl_sync(0xffffffffu, o1, src0 + 8),
                        y2 = __shfl_sync(0xffffffffu, o1, src0 + 16), y3 = __shfl_sync(0xffffffffu, o1, src0 + 24);
            if (g == 0) {
                oc_out[(t0 + tt) * HV * S + head * S + col0] = (x0 + x1 + x2 + x3) * rs_s;
                oc_out[(t0 + tt) * HV * S + head * S + col1] = (y0 + y1 + y2 + y3) * rs_s;
            }
        }
    }
    {
        float* sb = state + (size_t) head * S;
        uint32_t voff = (uint32_t) ((g * 32) * HV * S);
        asm volatile("" : "+v"(voff));
#pragma unroll
        for (int r = 0; r < 32; ++r) { sb[(size_t) r * HV * S + voff + col0] = s0[r]; sb[(size_t) r * HV * S + voff + col1] = s1[r]; }
    }
}

// gdn_out_norm_kernel's body over a grid-stride walk of the (token, head) rows: the old kernel launched a 128-thread
// block per row (T * 48 blocks; ~45 GB/s on the 8060S).  The same code per row (the same compiled arithmetic: a wave
// per row with four values per lane rounded differently in its first xor step).
template <bool WY>
__global__ void __launch_bounds__(S) gdn_out_norm_loop_kernel(const float* __restrict__ z, const float* __restrict__ gamma,
                                                              float eps, float* __restrict__ y, uint16_t* __restrict__ y16,
                                                              int64_t rows, int64_t ld16) {
    __shared__ float wsum[4];
    const int col = threadIdx.x;
    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const size_t at = (size_t) row * S + col;
        const float oc = y[at];
        // (warp_sum as the compiler built it: the first xor step adds the partner's rounded square to an unrounded own
        // one, an fma; spelled out so neither instance's contraction can differ)
        float sp = __fmaf_rn(oc, oc, __shfl_xor_sync(0xffffffffu, __fmul_rn(oc, oc), 16));
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) sp = __fadd_rn(sp, __shfl_xor_sync(0xffffffffu, sp, o));
        __syncthreads();                               // the previous row's sums are read
        if ((col & 31) == 0) wsum[col >> 5] = sp;
        __syncthreads();
        const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
        const float bp = oc * rsqrtf(ss / (float) S + eps) * gamma[col], sg = sigm(z[at]);
        const size_t o16 = (size_t) (row / HV) * ld16 + (size_t) (row % HV) * S + col;
        if constexpr (WY) {
            const float v = bp * sg;
            y[at] = v;
            y16[o16] = hf(v);
        } else {
            // STRATA_GDN_NOY=1: nothing reads the FP32 result (403 MB per 16K chunk).  The FP16 value is what the
            // compiler makes of hf(v) next to the store: v_fma_mixlo_f16, the product rounded to FP16 once
            unsigned r;
            asm("v_fma_mixlo_f16 %0, %1, %2, 0" : "=v"(r) : "v"(sg), "v"(bp));
            y16[o16] = (uint16_t) r;
        }
    }
}
// STRATA_GDN_CHUNKED=1 on the gfx11 matrix cores: the recurrence in chunks of 16 tokens (the WY form of the CUDA
// chunked path above), FP16 operands and FP32 accumulation, the output norm fused.  Per value head, gamma_t the
// gate's sum from the chunk's start to t (inclusive; the code's sg) and S0 the state before the chunk:
//   A[t][i] = beta_t exp(gamma_t - gamma_i) k_t.k_i (i < t),  T = (I + A)^-1,  P[t][i] = exp(gamma_t - gamma_i) q_t.k_i (i <= t),
//   Y = beta (V - exp(gamma) K S0),  D = T Y,  o = exp(gamma) Q S0 + P D,
//   S = exp(gamma_last) S0 + K^T (exp(gamma_last - gamma) D).
// gdn_wmma_prep_kernel computes T and P of every chunk at once (they do not depend on the state); gdn_wmma_scan_kernel
// walks the chunks, a block per value head and a wave per 16 value columns, the state in the waves' WMMA accumulators
// (FP32), each product a 16 x 16 x 16 WMMA: k, q, T, P, D and the state as an operand are rounded to FP16 (relative
// 2^-11), so the output differs from the FP32 kernels by more than their rounding (tests/hip/gdn_chunked.cpp holds it
// to the bounds of its Error::ok).  The norm and the z gate run in the scan.
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
#define STRATA_GDN_WMMA 1   // device pass with the gfx11 WMMA intrinsic (the list of gfx_arch_is_gfx11_wmma)
#endif
constexpr int GW_CK = 16;          // tokens per chunk (a WMMA tile)
constexpr int GW_NW = S / 16;      // the scan's waves per block: one per 16 value columns
constexpr int GW_VPK = HV / HK;    // value heads per key head: the prep's waves per block
constexpr int GW_KP = S + 8;       // the scan's padded FP16 k / q rows
constexpr int GW_AHEAD = 3;        // the scan warms the cache this many chunks ahead
constexpr int64_t kGdnWmmaMin = 64;   // STRATA_GDN_CHUNKED=1 takes prompt chunks of this many tokens and more
// gfx12: the chunked kernels are faster than the sequential ones at every call length from 64 tokens up (1.7x to 2.5x on the
// layer), but the prompt-speed gain depends on the prompt: IQ3_S on an R9700, +3.9% at 4K, +5.0% at 8K (median of 7 pairs: borderline), +6.9% at 16K, +10.1% at
// 32K.  It first started at 12000 tokens for the old +5% bar; the owner then chose every gain (2026-10-30), and the final gate
// with the start at 64 measured +1.7% to +3.2% at 1-3K, +3.5% at 4K, +4.7% at 16K, +6.1% at 32K (20 pairs), KL 0.0025 mean /
// 0.0381 max, 55 of 61 tasks, a 30-minute soak clean.  So gfx12 takes it from the same 64 tokens as gfx11.
constexpr int64_t kGdnW12MinT = kGdnWmmaMin;
typedef _Float16 gw_h16 __attribute__((ext_vector_type(16)));
typedef _Float16 gw_h8 __attribute__((ext_vector_type(8)));
typedef _Float16 gw_h2 __attribute__((ext_vector_type(2)));
typedef float gw_f8 __attribute__((ext_vector_type(8)));
typedef float gw_f4 __attribute__((ext_vector_type(4)));   // (HIP's float4 copies through memcpy, which kept a staged
                                                           // copy in scratch: a store that waits for the load)
__device__ __forceinline__ gw_f8 gw_mma(gw_h16 a, gw_h16 b, gw_f8 c) {
#if defined(STRATA_GDN_WMMA)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
    (void) a; (void) b; (void) c;
    __builtin_trap();   // not a gfx11 device pass: the runtime gate never launches it
#endif
}
// An accumulator tile (lane l: rows 2v + (l >> 4) for v = 0..7, column l & 15) as a B operand (lane l: column l & 15,
// rows 0..15, both halves of the wave alike): pairs of rows rounded to FP16 and packed, the other half's pairs fetched
// across (permlanex16), then interleaved by a byte permute whose selector depends on the lane's half (gw_sel).
struct GwSel { unsigned even, odd; };
__device__ __forceinline__ GwSel gw_sel(int hi) { return hi ? GwSel{0x01000504u, 0x03020706u} : GwSel{0x05040100u, 0x07060302u}; }
__device__ __forceinline__ gw_h16 gw_acc_to_b(gw_f8 x, GwSel ps) {
    unsigned wd[8];
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const gw_h2 p = {(_Float16) x[2 * m], (_Float16) x[2 * m + 1]};
        const unsigned own = __builtin_bit_cast(unsigned, p);
#if defined(STRATA_GDN_WMMA)
        const unsigned other = __builtin_amdgcn_permlanex16(own, own, 0x76543210u, 0xfedcba98u, false, false);
#else
        const unsigned other = own;
#endif
        wd[2 * m] = __builtin_amdgcn_perm(other, own, ps.even);
        wd[2 * m + 1] = __builtin_amdgcn_perm(other, own, ps.odd);
    }
    return __builtin_bit_cast(gw_h16, wd);
}
__device__ __forceinline__ gw_h16 gw_ld(const _Float16* p) {   // 32 bytes of LDS, 16-byte aligned
    const uint4 a = *reinterpret_cast<const uint4*>(p), b = *reinterpret_cast<const uint4*>(p + 8);
    const uint4 ab[2] = {a, b};
    return __builtin_bit_cast(gw_h16, ab);
}
// The wave's own LDS writes complete (read back across its lanes).  A gfx11 instruction: gfx12 device passes never run these kernels.
__device__ __forceinline__ void gw_lds_wait() {
#if defined(STRATA_GDN_WMMA)
    __asm__ volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
#endif
}
// A barrier over LDS only.  __syncthreads() is a workgroup fence over all memory, which on gfx11 also waits for every
// global load in flight: the scan's loads of the next chunk would have to land before each barrier.
__device__ __forceinline__ void gw_lds_barrier() {
#if defined(STRATA_GDN_WMMA)
    __asm__ volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
#else
    __syncthreads();   // the gfx11-only instruction must not reach a gfx12 device pass
#endif
}

// A block per (key head, chunk) - the grid's x the key head, so the blocks in flight read the same token rows (blocks
// that walked the key heads' slices of far-apart rows took 14 ms against 2.2 ms for 16K tokens on gfx1151) - and a wave per value head of the key head.
// T then P, FP16 row-major (1 KB), at tp[(chunk * HV + value head) * 512].
__global__ void __launch_bounds__(32 * GW_VPK) gdn_wmma_prep_kernel(const float* __restrict__ h, const float* __restrict__ gate,
                                                                 const float* __restrict__ beta, _Float16* __restrict__ tp,
                                                                 int64_t T) {
    __shared__ float sa[GW_VPK][GW_CK * GW_CK], sg[GW_VPK][GW_CK], sb[GW_VPK][GW_CK];
    const int qh = blockIdx.x, ch = blockIdx.y, w = threadIdx.x >> 5, l = threadIdx.x & 31, c = l & 15, hi = l >> 4;
    const int vh = qh + w * HK;
    const int64_t t0 = (int64_t) ch * GW_CK;
    gw_f8 kk = {}, qk = {};   // k_t.k_i and q_t.k_i (lane: column i = c, rows t = 2v + hi)
    {
        const bool ok = t0 + c < T;   // past the end: zero rows
        const float* ht = h + (ok ? t0 + c : 0) * C;
#pragma unroll
        for (int j = 0; j < S / 16; ++j) {
            gw_h16 ka, qa;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const gw_f4 k4 = *reinterpret_cast<const gw_f4*>(ht + HK * S + qh * S + 16 * j + 4 * i);
                const gw_f4 q4 = *reinterpret_cast<const gw_f4*>(ht + qh * S + 16 * j + 4 * i);
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    ka[4 * i + e] = (_Float16) (ok ? k4[e] : 0.0f);
                    qa[4 * i + e] = (_Float16) (ok ? q4[e] : 0.0f);
                }
            }
            kk = gw_mma(ka, ka, kk);   // B = K^T: lane i holds column i of K^T, row i of K - the A operand itself
            qk = gw_mma(qa, ka, qk);
        }
    }
    if (l < GW_CK) {   // past the end: gate 0, beta 0
        const bool ok = t0 + l < T;
        float g = ok ? gate[(t0 + l) * HV + vh] : 0.0f;
#pragma unroll
        for (int o = 1; o < GW_CK; o <<= 1) {
            const float u = __shfl_up(g, o, GW_CK);
            if (l >= o) g += u;
        }
        sg[w][l] = g;
        sb[w][l] = ok ? beta[(t0 + l) * HV + vh] : 0.0f;
    }
    gw_lds_wait();   // the wave's own LDS writes, read across its lanes
    _Float16* out = tp + ((int64_t) ch * HV + vh) * (2 * GW_CK * GW_CK);
#pragma unroll
    for (int v = 0; v < 8; ++v) {
        const int t = 2 * v + hi;
        sa[w][t * GW_CK + c] = c < t ? sb[w][t] * __expf(sg[w][t] - sg[w][c]) * kk[v] : 0.0f;
        out[GW_CK * GW_CK + t * GW_CK + c] = (_Float16) (c <= t ? __expf(sg[w][t] - sg[w][c]) * qk[v] : 0.0f);
    }
    gw_lds_wait();
    if (l < GW_CK) {   // lane c: column c of T = (I + A)^-1 by forward substitution
        float tc[GW_CK];
#pragma unroll
        for (int t = 0; t < GW_CK; ++t) {
            float v = t == c ? 1.0f : 0.0f;
#pragma unroll
            for (int k = 0; k < t; ++k) v = fmaf(-sa[w][t * GW_CK + k], tc[k], v);
            tc[t] = v;
            out[t * GW_CK + c] = (_Float16) v;
        }
    }
}

// A thread's slice of a chunk in the scan (token threadIdx / 16, columns (threadIdx % 16) * 8 .. + 7): k, q and v,
// loaded a chunk ahead, and one word of the chunk's T and P; lanes 0..15 of wave 0 also gate and beta of token l.
// Rows past the end read the last row (finite: their gate and beta become 0 at staging, which keeps them out of the
// state, of the real rows' products and of the output) - no select on the loaded values, which would wait for them.
struct GwChunk {
    gw_f4 k0, k1, q0, q1, v0, v1;
    float g, b;
    unsigned tp;
};
__device__ __forceinline__ void gw_load(GwChunk& in, const float* __restrict__ h, const float* __restrict__ gate,
                                        const float* __restrict__ beta, const unsigned* __restrict__ tp, int64_t t0,
                                        int64_t T, int vh, int qh) {
    const int t = threadIdx.x >> 4, r = (threadIdx.x & 15) * 8;
    const float* ht = h + std::min<int64_t>(t0 + t, T - 1) * C;
    in.tp = tp[((t0 / GW_CK) * HV + vh) * (GW_CK * GW_CK) + threadIdx.x];
    in.k0 = *reinterpret_cast<const gw_f4*>(ht + HK * S + qh * S + r);
    in.k1 = *reinterpret_cast<const gw_f4*>(ht + HK * S + qh * S + r + 4);
    in.q0 = *reinterpret_cast<const gw_f4*>(ht + qh * S + r);
    in.q1 = *reinterpret_cast<const gw_f4*>(ht + qh * S + r + 4);
    in.v0 = *reinterpret_cast<const gw_f4*>(ht + 2 * HK * S + vh * S + r);
    in.v1 = *reinterpret_cast<const gw_f4*>(ht + 2 * HK * S + vh * S + r + 4);
    if (threadIdx.x < GW_CK) {
        const int64_t tl = std::min<int64_t>(t0 + threadIdx.x, T - 1);
        in.g = gate[tl * HV + vh];
        in.b = beta[tl * HV + vh];
    }
}
// Brings a later chunk's lines (this block's k, q, v and z rows: 256 lines of 128 bytes, one a thread) into the cache
// with a load whose result is not used, so the real loads of that chunk hit the cache.  An ordinary load: the compiler
// tracks it, and the result is consumed (by an empty asm) one chunk later, after the store of the previous chunk's output
// rows, where the wait for the loads issued before that point also covers it.
__device__ __forceinline__ unsigned gw_touch(const float* __restrict__ h, const float* __restrict__ z, int64_t t0,
                                             int64_t T, int vh, int qh) {
    const int t = threadIdx.x >> 4, part = threadIdx.x & 15, kind = part >> 2;
    if (t0 + t >= T) return 0u;
    const float* p = kind == 0 ? h + (t0 + t) * C + HK * S + qh * S
                   : kind == 1 ? h + (t0 + t) * C + qh * S
                   : kind == 2 ? h + (t0 + t) * C + 2 * HK * S + vh * S
                               : z + (t0 + t) * HV * S + vh * S;
    return __float_as_uint(p[32 * (part & 3)]);
}
// Per chunk: stage it (k / q FP16, k also transposed, v, the gate's running sum, beta, T and P); barrier (1); store
// the previous chunk's output rows (the z gate on its normed output, 16 bytes a thread); X = K S0, QS = Q S0; Y, D,
// o and the wave's sums of o^2; the state update; barrier (2); the norm.  LDS is single-buffered: an array written in
// the staging of chunk c + 1 or after its barrier (2) was last read before that wave reached barrier (1) of c + 1.
__global__ void __launch_bounds__(GW_NW * 32) gdn_wmma_scan_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                                   const float* __restrict__ gate, const float* __restrict__ beta,
                                                                   const float* __restrict__ z, const float* __restrict__ gamma,
                                                                   const unsigned* __restrict__ tp, float eps,
                                                                   uint16_t* __restrict__ y16, int64_t T, int64_t ld16) {
    __shared__ __attribute__((aligned(16))) _Float16 sk[GW_CK * GW_KP], sq[GW_CK * GW_KP], skt[S * GW_CK],
        stp[2 * GW_CK * GW_CK];
    __shared__ __attribute__((aligned(16))) float sv[GW_CK * S], so[GW_CK * S], sn[GW_CK * GW_NW];
    __shared__ float sg[GW_CK], sb[GW_CK], sgam[S];
    const _Float16 *st_ = stp, *sp_ = stp + GW_CK * GW_CK;   // T, P
    const int vh = blockIdx.x, qh = vh % HK, w = threadIdx.x >> 5, l = threadIdx.x & 31, c = l & 15, hi = l >> 4;
    const int col = 16 * w + c;                                       // this lane's column in the products
    const int st = threadIdx.x >> 4, sr = (threadIdx.x & 15) * 8;    // this thread's staging slice
    const size_t rs = (size_t) HV * S;
    const GwSel ps = gw_sel(hi);
    gw_f8 s[S / 16];   // the state: rows 16j + 2v + hi, column col
#pragma unroll
    for (int j = 0; j < S / 16; ++j)
#pragma unroll
        for (int v = 0; v < 8; ++v) s[j][v] = state[(size_t) (16 * j + 2 * v + hi) * rs + (size_t) vh * S + col];
    if (threadIdx.x < S) sgam[threadIdx.x] = gamma[threadIdx.x];
    const float out_scale = rsqrtf((float) S);
    GwChunk cur;
    gw_load(cur, h, gate, beta, tp, 0, T, vh, qh);
    unsigned touched = 0;
    gw_f4 z0 = {}, z1 = {};   // the z slice of the chunk whose output rows are stored next
    auto store_rows = [&](int64_t c0) {   // the output rows of the chunk at c0 (its norm in so): z gate, FP16
        if (c0 + st >= T) return;
        const float* o = &so[st * S + sr];
        gw_h8 r;
#pragma unroll
        for (int i = 0; i < 8; ++i) r[i] = (_Float16) (o[i] * sigm(i < 4 ? z0[i] : z1[i - 4]));
        *reinterpret_cast<uint4*>(y16 + (c0 + st) * ld16 + vh * S + sr) = __builtin_bit_cast(uint4, r);
    };
    for (int64_t t0 = 0; t0 < T; t0 += GW_CK) {
        {   // stage this chunk
            gw_h8 kh, qq;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                kh[i] = (_Float16) (i < 4 ? cur.k0[i] : cur.k1[i - 4]);
                qq[i] = (_Float16) (i < 4 ? cur.q0[i] : cur.q1[i - 4]);
                skt[(sr + i) * GW_CK + st] = kh[i];
            }
            *reinterpret_cast<uint4*>(&sk[st * GW_KP + sr]) = __builtin_bit_cast(uint4, kh);
            *reinterpret_cast<uint4*>(&sq[st * GW_KP + sr]) = __builtin_bit_cast(uint4, qq);
            *reinterpret_cast<gw_f4*>(&sv[st * S + sr]) = cur.v0;
            *reinterpret_cast<gw_f4*>(&sv[st * S + sr + 4]) = cur.v1;
            reinterpret_cast<unsigned*>(stp)[threadIdx.x] = cur.tp;
            if (threadIdx.x < GW_CK) {   // past the end: gate 0, beta 0
                const bool ok = t0 + l < T;
                float g = ok ? cur.g : 0.0f;
#pragma unroll
                for (int o = 1; o < GW_CK; o <<= 1) {
                    const float u = __shfl_up(g, o, GW_CK);
                    if (l >= o) g += u;
                }
                sg[l] = g;
                sb[l] = ok ? cur.b : 0.0f;
            }
        }
        if (t0 + GW_CK < T) gw_load(cur, h, gate, beta, tp, t0 + GW_CK, T, vh, qh);   // lands during this chunk
        const unsigned touch_now = gw_touch(h, z, t0 + GW_AHEAD * GW_CK, T, vh, qh);
        gw_lds_barrier();   // (1) this chunk staged; the previous chunk's norm in so
        if (t0 > 0) store_rows(t0 - GW_CK);
        __asm__ volatile("" ::"v"(touched));   // the previous touch, consumed here
        touched = touch_now;
        {   // this chunk's z slice, for its output rows after barrier (1) of the next chunk
            const int64_t tz = std::min<int64_t>(t0 + st, T - 1);
            z0 = *reinterpret_cast<const gw_f4*>(z + tz * HV * S + vh * S + sr);
            z1 = *reinterpret_cast<const gw_f4*>(z + tz * HV * S + vh * S + sr + 4);
        }
        gw_f8 x = {}, qs = {};   // K S0 and Q S0, this wave's columns
#pragma unroll
        for (int j = 0; j < S / 16; ++j) {
            const gw_h16 b = gw_acc_to_b(s[j], ps);
            x = gw_mma(gw_ld(sk + c * GW_KP + 16 * j), b, x);
            qs = gw_mma(gw_ld(sq + c * GW_KP + 16 * j), b, qs);
        }
        gw_f8 y;
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int t = 2 * v + hi;
            y[v] = sb[t] * (sv[t * S + col] - __expf(sg[t]) * x[v]);
        }
        const gw_f8 d = gw_mma(gw_ld(st_ + c * GW_CK), gw_acc_to_b(y, ps), gw_f8{});
        gw_f8 o;
#pragma unroll
        for (int v = 0; v < 8; ++v) o[v] = __expf(sg[2 * v + hi]) * qs[v];
        o = gw_mma(gw_ld(sp_ + c * GW_CK), gw_acc_to_b(d, ps), o);
#pragma unroll
        for (int v = 0; v < 8; ++v) {   // the wave's 16 columns of o^2 per token, then the 8 waves' through LDS
            o[v] *= out_scale;
            float p = o[v] * o[v];
#pragma unroll
            for (int m = 1; m < 16; m <<= 1) p += __shfl_xor(p, m);
            if (c == 0) sn[(2 * v + hi) * GW_NW + w] = p;
        }
        const float gl = sg[GW_CK - 1];
        gw_f8 dd;
#pragma unroll
        for (int v = 0; v < 8; ++v) dd[v] = __expf(gl - sg[2 * v + hi]) * d[v];
        const gw_h16 db = gw_acc_to_b(dd, ps);
        const float eg = __expf(gl);
#pragma unroll
        for (int j = 0; j < S / 16; ++j) s[j] = gw_mma(gw_ld(skt + (16 * j + c) * GW_CK), db, s[j] * eg);
        gw_lds_barrier();   // (2) the sums of o^2
        const float g_col = sgam[col];
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int t = 2 * v + hi;
            const gw_f4 n0 = *reinterpret_cast<const gw_f4*>(&sn[t * GW_NW]), n1 = *reinterpret_cast<const gw_f4*>(&sn[t * GW_NW + 4]);
            const float ss = n0[0] + n0[1] + n0[2] + n0[3] + n1[0] + n1[1] + n1[2] + n1[3];
            so[t * S + col] = o[v] * rsqrtf(ss * (1.0f / S) + eps) * g_col;
        }
    }
    __asm__ volatile("" ::"v"(touched));
    gw_lds_barrier();
    store_rows((T - 1) / GW_CK * GW_CK);
    __asm__ volatile("" : "+s"(state));   // the state's addresses recomputed here, not kept (spilled) since the start
#pragma unroll
    for (int j = 0; j < S / 16; ++j)
#pragma unroll
        for (int v = 0; v < 8; ++v) state[(size_t) (16 * j + 2 * v + hi) * rs + (size_t) vh * S + col] = s[j][v];
}
#include "gdn_wmma12.inc"
// The opt-in chunked recurrence on gfx11 and gfx12 (STRATA_GDN_CHUNKED=1): false (nothing launched) on another part.  y, the
// FP32 scratch of the other kernels (T x 6144 floats), holds T and P (1 KB per chunk and value head: T x 3 KB).
bool gdn_rec_wmma(float* state, const float* h, const float* gate, const float* beta, const float* z, const float* gamma,
                  float eps, float* y, uint16_t* y16, int64_t T, cudaStream_t s, int64_t ld16) {
    if (T < kGdnWmmaMin || ld16 % 8 != 0) return false;   // the rows are stored 16 bytes at a time
    // A layer-split worker can change devices: a process-wide answer for the first card can launch a trapping
    // gfx11 kernel on a later gfx12 card.  Cache only the current thread's last device, as properties are not cheap.
    static thread_local int last_device = -1;
    static thread_local bool gfx11 = false, gfx12 = false;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    if (dev != last_device) {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return false;
        gfx11 = strata::kernels::gfx_arch_is_gfx11_wmma(p.gcnArchName);
        gfx12 = strata::kernels::gfx_arch_is_gfx12_wmma(p.gcnArchName);
        last_device = dev;
    }
    if (!gfx11 && !gfx12) return false;
    if (gfx12 && T < kGdnW12MinT) return false;
    const int64_t nch = (T + GW_CK - 1) / GW_CK;
    _Float16* tp = reinterpret_cast<_Float16*>(y);
    if (gfx12) {
        gdn_w12_prep_kernel<false><<<dim3(HK, (unsigned) nch), 32 * GW_VPK, 0, s>>>(h, nullptr, gate, beta, tp, T);
        static const bool lb = [] { const char* v = std::getenv("STRATA_GDN_W12_LB"); return v == nullptr || std::atoi(v) != 0; }();
        if (lb)
            gdn_w12_scan_kernel<true, false><<<HV, GW_NW * 32, 0, s>>>(state, h, nullptr, gate, beta, z, gamma,
                                                                       reinterpret_cast<const unsigned*>(tp), eps, y16, T, ld16);
        else
            gdn_w12_scan_kernel<false, false><<<HV, GW_NW * 32, 0, s>>>(state, h, nullptr, gate, beta, z, gamma,
                                                                        reinterpret_cast<const unsigned*>(tp), eps, y16, T, ld16);
        return true;
    }
    gdn_wmma_prep_kernel<<<dim3(HK, (unsigned) nch), 32 * GW_VPK, 0, s>>>(h, gate, beta, tp, T);
    gdn_wmma_scan_kernel<<<HV, GW_NW * 32, 0, s>>>(state, h, gate, beta, z, gamma, reinterpret_cast<const unsigned*>(tp),
                                                   eps, y16, T, ld16);
    return true;
}
// STRATA_GDN_CHUNKED=1 on gfx12 with the conv folded in (gdn_wmma12f.inc; STRATA_GDN_FUSE=0 keeps conv and recurrence apart):
// the work of gdn_conv + gdn_recurrence in three launches, no h buffer.  False (nothing launched) when the chunked path or
// the fusion is off, the device is not gfx12, or the call is short / unaligned: the caller then runs the two entry points.
bool gdn_fused_impl(float* history, const float* qkv, const float* conv_w, float* hbuf, float* state, const float* gate,
                    const float* beta, const float* z, const float* gamma, float eps, float l2eps, float* y, uint16_t* y16,
                    int64_t T, void* stream, int64_t ld16) {
    static const bool on = [] {
        const char* c = std::getenv("STRATA_GDN_CHUNKED");
        const char* f = std::getenv("STRATA_GDN_FUSE");
        return c != nullptr && std::atoi(c) != 0 && (f == nullptr || std::atoi(f) != 0);
    }();
    if (!on || T < kGdnWmmaMin) return false;
    if (ld16 <= 0) ld16 = (int64_t) HV * S;
    if (ld16 % 8 != 0) return false;
    static thread_local int last_device = -1;
    static thread_local bool gfx12 = false;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    if (dev != last_device) {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return false;
        gfx12 = strata::kernels::gfx_arch_is_gfx12_wmma(p.gcnArchName);
        last_device = dev;
    }
    if (!gfx12 || T < kGdnW12MinT) return false;
    cudaStream_t s = (cudaStream_t) stream;
    const int64_t nch = (T + GW_CK - 1) / GW_CK;
    _Float16* tp = reinterpret_cast<_Float16*>(y);
    _Float16* h16 = reinterpret_cast<_Float16*>(hbuf);   // the FP32 h buffer holds the FP16 one
    gdn_conv_l2_h16_kernel<<<dim3(C / 128, (unsigned) ((T + CONV_TILE - 1) / CONV_TILE)), 128, 0, s>>>(history, qkv, conv_w, h16, T, l2eps);
    gdn_conv_hist_kernel<<<C / 128, 128, 0, s>>>(history, qkv, T);   // the history after the chunk (the conv above read the old one)
    gdn_w12_prep_kernel<true><<<dim3(HK, (unsigned) nch), 32 * GW_VPK, 0, s>>>(nullptr, h16, gate, beta, tp, T);
    static const bool lb = [] { const char* v = std::getenv("STRATA_GDN_W12_LB"); return v == nullptr || std::atoi(v) != 0; }();
    if (lb)
        gdn_w12_scan_kernel<true, true><<<HV, GW_NW * 32, 0, s>>>(state, nullptr, h16, gate, beta, z, gamma,
                                                                  reinterpret_cast<const unsigned*>(tp), eps, y16, T, ld16);
    else
        gdn_w12_scan_kernel<false, true><<<HV, GW_NW * 32, 0, s>>>(state, nullptr, h16, gate, beta, z, gamma,
                                                                   reinterpret_cast<const unsigned*>(tp), eps, y16, T, ld16);
    check("gdn_conv_recurrence_fused");
    return true;
}
#endif   // __HIPCC__
__global__ void __launch_bounds__(S) gdn_out_norm_kernel(const float* __restrict__ z, const float* __restrict__ gamma,
                                                         float eps, float* __restrict__ y, uint16_t* __restrict__ y16,
                                                         int64_t ld16) {
    __shared__ float wsum[4];
    const int64_t t = blockIdx.x;
    const int head = blockIdx.y, col = threadIdx.x;
    const size_t at = (size_t) t * HV * S + (size_t) head * S + col;
    const float oc = y[at];
    float sp = warp_sum(oc * oc);
    if ((col & 31) == 0) wsum[col >> 5] = sp;
    __syncthreads();
    const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
    const float v = oc * rsqrtf(ss / (float) S + eps) * gamma[col] * sigm(z[t * HV * S + head * S + col]);
    y16[(size_t) t * ld16 + (size_t) head * S + col] = hf(v);   // (the FP32 normalized value is not stored: nothing reads it)
}

// ---------------------------------------------------------------- MoE
template <int REG>
__global__ void route_kernel(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ wout,
                             int64_t T) {
    const int64_t t = (int64_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (t >= T) return;
    const int lane = threadIdx.x & 31;
    const float* lg = logits + t * (REG * 32);
    float v[REG];
#pragma unroll
    for (int i = 0; i < REG; ++i) v[i] = lg[lane + i * 32];
    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < REG; ++i) mx = fmaxf(mx, v[i]);
    mx = warp_max(mx);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < REG; ++i) { v[i] = expf(v[i] - mx); sum += v[i]; }
    const float rcp = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < REG; ++i) { v[i] *= rcp; if (isnan(v[i])) v[i] = -FLT_MAX; }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = v[0];
        int ex = lane;
#pragma unroll
        for (int i = 1; i < REG; ++i) if (v[i] > best) { best = v[i]; ex = lane + i * 32; }
#pragma unroll
        for (int m = 16; m; m >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, m);
            const int oi = __shfl_xor_sync(0xffffffffu, ex, m);
            if (ob > best || (ob == best && oi < ex)) { best = ob; ex = oi; }
        }
        if ((ex & 31) == lane) { v[ex / 32] = -INFINITY; selected_sum += best; }
        if (lane == 0) ids[t * 10 + rank] = ex;
        if (rank == lane) selected = best;
    }
    selected_sum = fmaxf(warp_sum(selected_sum), 6.103515625e-5f);
    if (lane < 10) wout[t * 10 + lane] = selected / selected_sum;
}
// Strata blob: gate/up codes [1280][640 B], down codes [2560][160 B], gate/up scales [1280][40] f16, down scales [2560][10] f16
template <bool HALF>
__global__ void blob_dequant_kernel(const uint8_t* __restrict__ blob, uint16_t* __restrict__ gu16,
                                    uint16_t* __restrict__ d16) {
    constexpr size_t O_D_CODES = (size_t) 1280 * 640, O_GU_SC = O_D_CODES + (size_t) 2560 * 160,
                     O_D_SC = O_GU_SC + (size_t) 1280 * 40 * 2;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // one thread per 4 weights (one code byte)
    const int64_t n_gu = 1280LL * 640, n_d = 2560LL * 160;
    if (i < n_gu) {
        const int64_t row = i / 640, byte = i % 640;
        const uint8_t c = blob[row * 640 + byte];
        const uint8_t* sp = blob + O_GU_SC + (size_t) (row * 40 + (byte * 4) / 64) * 2;
        const float d = __half2float(__ushort_as_half((uint16_t) (sp[0] | (sp[1] << 8))));
        uint16_t* o = gu16 + row * 2560 + byte * 4;
#pragma unroll
        for (int k = 0; k < 4; ++k) { const float v = (float) (((c >> (2 * k)) & 3) - 1) * d; o[k] = HALF ? hf(v) : bf(v); }
    } else if (i < n_gu + n_d) {
        const int64_t j = i - n_gu, row = j / 160, byte = j % 160;
        const uint8_t c = blob[O_D_CODES + row * 160 + byte];
        const uint8_t* sp = blob + O_D_SC + (size_t) (row * 10 + (byte * 4) / 64) * 2;
        const float d = __half2float(__ushort_as_half((uint16_t) (sp[0] | (sp[1] << 8))));
        uint16_t* o = d16 + row * 640 + byte * 4;
#pragma unroll
        for (int k = 0; k < 4; ++k) { const float v = (float) (((c >> (2 * k)) & 3) - 1) * d; o[k] = HALF ? hf(v) : bf(v); }
    }
}
__global__ void swiglu_il_kernel(const float* __restrict__ gu, uint16_t* __restrict__ h16, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 640) return;
    const int64_t r = i / 640, k = i % 640;
    const float g = gu[r * 1280 + 2 * k], u = gu[r * 1280 + 2 * k + 1];
    h16[i] = hf_sat(g / (1.0f + __expf(-g)) * u);
}
// The Volta prompt experts (gemm_iq_f16_grouped): SwiGLU of the gate/up product's rows (gate 0..n_ff, up n_ff..2 n_ff,
// or interleaved) to the FP16 input of the down product, saturated as swiglu_il_kernel
__global__ void swiglu_split_f16_kernel(const float* __restrict__ gu, uint16_t* __restrict__ h16, int64_t rows, int n_ff,
                                        bool interleaved) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n_ff) return;
    const int64_t r = i / n_ff, k = i % n_ff;
    const float* row = gu + r * 2 * n_ff;
    const float g = interleaved ? row[2 * k] : row[k], u = interleaved ? row[2 * k + 1] : row[n_ff + k];
    h16[i] = hf_sat(g / (1.0f + __expf(-g)) * u);
}
__global__ void swiglu_pair_kernel(const float* __restrict__ g, const float* __restrict__ u, uint16_t* __restrict__ h16,
                                   int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 640) return;
    const float a = g[i];
    h16[i] = hf_sat(a / (1.0f + __expf(-a)) * u[i]);
}
__global__ void gather_rows16_kernel(const uint16_t* __restrict__ x, const int32_t* __restrict__ src,
                                     uint16_t* __restrict__ dst, int64_t n, int64_t width) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // one uint4 (8 bf16)
    const int64_t per = width / 8;
    if (i >= n * per) return;
    const int64_t r = i / per, j = i % per;
    reinterpret_cast<uint4*>(dst)[r * per + j] = reinterpret_cast<const uint4*>(x)[(int64_t) src[r] * per + j];
}
__global__ void moe_combine_kernel(const float* __restrict__ Dm, const int32_t* __restrict__ slot,
                                   const float* __restrict__ w, const float* __restrict__ shared,
                                   const float* __restrict__ sg, float* __restrict__ bo, int64_t T,
                                   const float* __restrict__ row_sd) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    const int64_t t = i / N, d = i % N;
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < 10; ++k) {
        const int64_t r = slot[t * 10 + k];
        const float v = row_sd ? Dm[r * N + d] * row_sd[r] : Dm[r * N + d];   // the product scale_down_rows stored
        s = fmaf(w[t * 10 + k], v, s);
    }
    bo[i] = s + shared[i] * sigm(sg[t]);
}

// W (X5): four columns per thread, one float4 load per slot; the per-element expression and the k order are unchanged
// (bitwise equal to moe_combine_kernel). Needs 16-byte aligned Dm / shared / bo (N * 4 bytes per row is a multiple of 16).
__global__ void moe_combine4_kernel(const float* __restrict__ Dm, const int32_t* __restrict__ slot,
                                    const float* __restrict__ w, const float4* __restrict__ shared,
                                    const float* __restrict__ sg, float4* __restrict__ bo, int64_t T,
                                    const float* __restrict__ row_sd) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // float4 index
    if (i >= T * (N / 4)) return;
    const int64_t t = i / (N / 4), d = (i % (N / 4)) * 4;
    float4 s = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 v[10];
    float ww[10];
#pragma unroll
    for (int k = 0; k < 10; ++k) {
        ww[k] = w[t * 10 + k];
        const int64_t r = slot[t * 10 + k];
        v[k] = *reinterpret_cast<const float4*>(Dm + r * N + d);
        if (row_sd) {   // NVFP4 in MMQ: the row's s_down, as moe_combine_kernel applies it
            const float sd = row_sd[r];
            v[k] = make_float4(v[k].x * sd, v[k].y * sd, v[k].z * sd, v[k].w * sd);
        }
    }
#pragma unroll
    for (int k = 0; k < 10; ++k) {
        s.x = fmaf(ww[k], v[k].x, s.x); s.y = fmaf(ww[k], v[k].y, s.y);
        s.z = fmaf(ww[k], v[k].z, s.z); s.w = fmaf(ww[k], v[k].w, s.w);
    }
    const float g = sigm(sg[t]);
    const float4 sh = shared[i];
    bo[i] = make_float4(s.x + sh.x * g, s.y + sh.y * g, s.z + sh.z * g, s.w + sh.w * g);
}

__global__ void moe_shared_init_kernel(const float* __restrict__ shared, const float* __restrict__ sg,
                                       float* __restrict__ acc, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    acc[i] = shared[i] * sigm(sg[i / N]);
}
// grid (n_u, ceil(N / 256)): one token of the group per block row, its rows summed in order
__global__ void moe_scatter_add_kernel(const float* __restrict__ Dm, const int32_t* __restrict__ u_tok,
                                       const int32_t* __restrict__ u_off, const int32_t* __restrict__ u_rows,
                                       const float* __restrict__ u_w, float* __restrict__ acc) {
    const int64_t u = blockIdx.x;
    const int64_t d = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= N) return;
    float s = 0.0f;
    for (int32_t j = u_off[u]; j < u_off[u + 1]; ++j) s = fmaf(u_w[j], Dm[(int64_t) u_rows[j] * N + d], s);
    acc[(int64_t) u_tok[u] * N + d] += s;
}

// ---------------------------------------------------------------- QSA helpers
__global__ void rms_rows_kernel(float* __restrict__ x, const float* __restrict__ w, int64_t cols, int64_t ld, float eps) {
    __shared__ float sh[32];
    float* r = x + (int64_t) blockIdx.x * ld;
    float ss = 0.0f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) ss += r[c] * r[c];
    const float s = rsqrtf(block_sum(ss, sh) / (float) cols + eps);
    __syncthreads();
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) r[c] = s * r[c] * w[c];
}
// TAB (#280, STRATA_ROPE_TABLE=1): the angles from the session's float64 table.  The host launches <false> whenever
// no table applies - the default - so the default kernel is 0.1.31's code exactly (the table read is not in it;
// with it merely skipped at run time, the compiled default path changed its results).
template <bool TAB>
__global__ void rope_kernel(float* __restrict__ x, int64_t heads, int64_t dim, int64_t ld, int64_t pos0,
                            float theta_scale, float freq_scale, float corr_low, float corr_high,
                            float ext_factor, float mscale, const int32_t* __restrict__ mtab,
                            strata::kernels::RopeTab rt) {
    const int64_t row = blockIdx.x;             // t * heads + h
    const int pair = threadIdx.x;               // 0..31
    const int64_t t = row / heads, h = row % heads;
    float* p = x + t * ld + h * dim;
    float c, s;
    if (!(TAB && strata::kernels::rope_tab_cs(rt, strata::kernels::mrope_pos(mtab, (int) (pos0 + t), pair), pair, c,
                                              s))) {
        const float theta_extrap =
            (float) strata::kernels::mrope_pos(mtab, (int) (pos0 + t), pair) * powf(theta_scale, (float) pair);
        strata::kernels::rope_scaled_angle(theta_extrap, freq_scale, corr_low, corr_high, ext_factor, mscale,
                                           pair, c, s);
    }
    const float a = p[pair], b = p[pair + 32];
    p[pair] = a * c - b * s;
    p[pair + 32] = a * s + b * c;
}
__global__ void split_q_kernel(const float* __restrict__ qf, float* __restrict__ q, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * 24 * 256) return;
    const int64_t t = i / (24 * 256), h = (i / 256) % 24, d = i % 256;
    q[i] = qf[t * 24 * 512 + h * 512 + d];
}
__global__ void gate_attn_kernel(const float* __restrict__ a, const float* __restrict__ qf, uint16_t* __restrict__ o16,
                                 int64_t T, int64_t ld16) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * 24 * 256) return;
    const int64_t t = i / (24 * 256), h = (i / 256) % 24, d = i % 256;
    o16[t * ld16 + h * 256 + d] = hf(a[i] * (1.0f / (1.0f + expf(-qf[t * 24 * 512 + h * 512 + 256 + d]))));
}

// one block per (token, kv head, 64-value group); 64 threads. KV streaming: the pool page only if the block is
// resident (table >= 0), and the host copy and the prompt path's staging pool (both identity layout) when given.
__global__ void kv_append_kernel(const float* __restrict__ K, const float* __restrict__ V, int64_t pos0,
                                 const int32_t* __restrict__ table, int64_t page_size, uint16_t* k_pool,
                                 uint16_t* v_pool, int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
                                 strata::kernels::KvHostPools host, strata::kernels::KvHostPools stage) {
    const int64_t t = blockIdx.x;
    const int kvh = blockIdx.y, g = blockIdx.z >> 1;
    const bool is_v = (blockIdx.z & 1) != 0;
    const int d = g * 64 + threadIdx.x;
    const float x = (is_v ? V : K)[t * 512 + kvh * 256 + d];
    const int64_t pos = pos0 + t;
    const int64_t page = table[pos / page_size];
    const int64_t row = (page * 2 + kvh) * page_size + pos % page_size;
    const int64_t row_id = ((pos / page_size) * 2 + kvh) * page_size + pos % page_size;
    if (k_pool != nullptr) {
        const uint16_t h = hf(x);
        if (page >= 0) (is_v ? v_pool : k_pool)[row * 256 + d] = h;
        if (host.k_pool != nullptr) (is_v ? host.v_pool : host.k_pool)[row_id * 256 + d] = h;
        if (stage.k_pool != nullptr) (is_v ? stage.v_pool : stage.k_pool)[row_id * 256 + d] = h;
        return;
    }
    float a = fabsf(x);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    __shared__ float wm[2];
    if ((threadIdx.x & 31) == 0) wm[threadIdx.x >> 5] = a;
    __syncthreads();
    const float amax = fmaxf(wm[0], wm[1]);
    const uint16_t sb = hf(amax / 127.0f);
    const float sf = __half2float(__ushort_as_half(sb));
    int q = 0;
    if (sf > 0.0f) { q = __float2int_rn(x / sf); q = q < -127 ? -127 : (q > 127 ? 127 : q); }
    if (page >= 0) {
        (is_v ? v_q : k_q)[row * 256 + d] = (int8_t) q;
        if (threadIdx.x == 0) (is_v ? v_scale : k_scale)[row * 4 + g] = sb;
    }
    if (host.k_q != nullptr) {
        (is_v ? host.v_q : host.k_q)[row_id * 256 + d] = (int8_t) q;
        if (threadIdx.x == 0) (is_v ? host.v_scale : host.k_scale)[row_id * 4 + g] = sb;
    }
    if (stage.k_q != nullptr) {
        (is_v ? stage.v_q : stage.k_q)[row_id * 256 + d] = (int8_t) q;
        if (threadIdx.x == 0) (is_v ? stage.v_scale : stage.k_scale)[row_id * 4 + g] = sb;
    }
}
__global__ void to_f16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
#if defined(__HIPCC__)
        y[i] = hf_sat(x[i]);   // saturated and counted like every other FP16 image (the hc / embedding image on gfx103x)
#else
        y[i] = hf(x[i]);       // CUDA: as it was
#endif
}
__global__ void round_f16_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        y[i] = __half2float(__float2half_rn(x[i]));
}
__global__ void to_bf16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, uint16_t* __restrict__ ylo,
                               int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const uint16_t h = act16(x[i]);
        y[i] = h;
        if (ylo) ylo[i] = bf_lo(x[i], h);
    }
}

// S23 (opt-in STRATA_HC_UPMIX=1, gfx11): the hyper-connection read's up projection with gr_mix_r as its epilogue.  The
// up GEMM's FP32 output `gated` (T x 10240: 671 MB per 16K chunk) is never written or read back: a block computes a
// tile of 64 tokens x 64 columns for all 4 streams on the matrix cores (BF16 in, FP32 accumulate over K = 320; the
// lo16 and w_up tiles staged in LDS), and each lane already holds the 4 streams' values of its (token, column) - the
// mix (x_c = R * rs * w, sum of x_c * sigmoid(gated_c), / 4) runs in registers, in gr_mix_r_kernel's order.  The
// GEMM sums K in another order than hipBLASLt: rounding-level, hence opt-in and quality-gated.
#if defined(__HIPCC__)
typedef __bf16 um_b16 __attribute__((ext_vector_type(16)));
typedef float um_f8 __attribute__((ext_vector_type(8)));
__device__ __forceinline__ um_f8 um_wmma(um_b16 a, um_b16 b, um_f8 c) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
#else
    (void) a; (void) b; return c;   // not a gfx11 device pass: never launched (runtime gate)
#endif
}
constexpr int UM_T = 256, UM_D = 16, UM_LD = LR + 8;   // a block: 256 tokens x 16 columns x the 4 streams
__device__ __forceinline__ um_b16 um_frag(const uint16_t* p) {   // 16 BF16 (32 bytes, 16-byte aligned)
    const uint4 a = *reinterpret_cast<const uint4*>(p), b = *reinterpret_cast<const uint4*>(p + 8);
    const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    return __builtin_bit_cast(um_b16, w);
}
// The block's w_up rows (16 columns x 4 streams x all 320 k) are staged in LDS once; each wave streams its own 16
// tokens' lo16 rows from global memory (every A fragment feeds the 4 streams' WMMAs) - one barrier per block.
__global__ void __launch_bounds__(512) gr_upmix_kernel(const uint16_t* __restrict__ lo16, const uint16_t* __restrict__ wu,
                                                       const float* __restrict__ R, const float* __restrict__ rs,
                                                       const float* __restrict__ w, float* __restrict__ mixed,
                                                       uint16_t* __restrict__ mixed16, uint16_t* __restrict__ mixed_h,
                                                       int64_t T) {
    __shared__ __align__(16) uint16_t sB[HC * UM_D][UM_LD];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    // blocks ordered column-tile fastest: the blocks in flight share a token tile, so the epilogue's R reads run
    // along rows (token-tile fastest scattered them over 512 rows x 40 KB per block: 28 ms instead of 3.7 at 8K)
    const int64_t t0 = (int64_t) blockIdx.y * UM_T + 16 * wave;
    const int d0 = blockIdx.x * UM_D;
    for (int i = tid; i < HC * UM_D * (LR / 8); i += 512) {
        const int rr = i / (LR / 8), q = i % (LR / 8);
        const int c = rr / UM_D, dd = rr % UM_D;
        *reinterpret_cast<uint4*>(&sB[rr][8 * q]) = *reinterpret_cast<const uint4*>(wu + (size_t) (c * N + d0 + dd) * LR + 8 * q);
    }
    __syncthreads();
    if (t0 >= T) return;
    const int64_t ta = t0 + l16 < T ? t0 + l16 : T - 1;
    const uint16_t* arow = lo16 + ta * LR;
    um_f8 acc[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) acc[c] = um_f8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int ks = 0; ks < LR; ks += 16) {
        const um_b16 a = um_frag(arow + ks);
#pragma unroll
        for (int c = 0; c < HC; ++c) acc[c] = um_wmma(a, um_frag(&sB[c * UM_D + l16][ks]), acc[c]);
    }
    const int d = d0 + l16;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int64_t t = t0 + 2 * i + hi;
        if (t >= T) continue;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            const float x = R[t * D + c * N + d] * rs[t * HC + c] * w[c * N + d];   // gr_mix_r_kernel's order
            s = fmaf(x, sigm(acc[c][i]), s);
        }
        s /= (float) HC;
        mixed[t * N + d] = s;
        if (mixed16) mixed16[t * N + d] = act16(s);
        if (mixed_h) mixed_h[t * N + d] = hf(s);
    }
}
#endif
#if !defined(__HIPCC__)
// S23 on CUDA (sm_80+; opt-in STRATA_HC_UPMIX=1, see hc_upmix in prefill.cpp): the hyper-connection read's up
// projection (lo16 x w_up^T, BF16 in, FP32 accumulate over K = 320) with gr_mix_r as its epilogue, so `gated`
// (T x 10240 FP32, 336 MB for an 8,192-token chunk) is neither written by the GEMM nor read back by the mix - gr_mix_r
// reads R and `gated` at ~96% of the bandwidth already, so only fewer bytes make the read faster.  A block: 128 tokens
// x 32 columns x the 4 streams (128 output columns), K staged in 32-deep slices (cp.async, double-buffered), 8 warps of
// 32 tokens x 16 columns x 4 streams on mma.m16n8k16 (ldmatrix); a lane then holds its (token, column) pairs for all 4
// streams, and the mix runs in registers in gr_mix_r_kernel's order (x_c = R * rs * w; s = fmaf(x_c, sigm(g_c), s);
// s / 4) with the BF16 / FP16 images.  The epilogue's R rows are asked into L2 at the start.  With cuBLAS's up GEMM
// summing K in the same order (CUDA 13.3 on an RTX 5090 from 33 tokens: gr_upmix_parity) `mixed` is the default pair's
// bits.
constexpr int UC_BM = 128, UC_BD = 32;   // tokens, columns a stream
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800   // cp.async, bf16 mma: the kernel is empty before sm_80
constexpr int UC_KC = 32, UC_LD = UC_KC + 8;   // k a slice, a staged row
__device__ __forceinline__ void uc_pf_l2(const float* p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }
__device__ __forceinline__ void uc_cp16(void* s, const void* g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(s)), "l"(g));
}
__device__ __forceinline__ void uc_ldm4(unsigned (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned) __cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void uc_mma(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
#endif
__global__ void __launch_bounds__(256, 2) gr_upmix_cuda_kernel(const uint16_t* __restrict__ lo16,
                                                              const uint16_t* __restrict__ wu,
                                                              const float* __restrict__ R, const float* __restrict__ rs,
                                                              const float* __restrict__ w, float* __restrict__ mixed,
                                                              uint16_t* __restrict__ mixed16,
                                                              uint16_t* __restrict__ mixed_h,
                                                              int64_t T) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(16) uint16_t sA[2][UC_BM][UC_LD];
    __shared__ __align__(16) uint16_t sB[2][HC * UC_BD][UC_LD];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp >> 1, wn = warp & 1;
    // blocks column-tile fastest: the blocks in flight share a token tile (its lo16 rows stay in L2, the epilogue's R
    // reads run along rows)
    const int64_t t0 = (int64_t) blockIdx.y * UC_BM;
    const int d0 = blockIdx.x * UC_BD;
    // the epilogue's R rows (128 tokens x 4 streams x 128 bytes) asked into L2 now, so they arrive under the GEMM
    for (int i = tid; i < UC_BM * HC; i += 256) {
        const int64_t t = t0 + i / HC;
        if (t < T) uc_pf_l2(R + t * D + (i % HC) * N + d0);
    }
    auto stage = [&](int kc, int buf) {
        for (int i = tid; i < UC_BM * (UC_KC / 8); i += 256) {
            const int r = i / (UC_KC / 8), q = i % (UC_KC / 8);
            const int64_t t = t0 + r < T ? t0 + r : T - 1;   // the last tile's missing rows read a real row, unused
            uc_cp16(&sA[buf][r][q * 8], lo16 + t * LR + kc * UC_KC + q * 8);
        }
        for (int i = tid; i < HC * UC_BD * (UC_KC / 8); i += 256) {
            const int r = i / (UC_KC / 8), q = i % (UC_KC / 8);
            uc_cp16(&sB[buf][r][q * 8], wu + ((int64_t) (r / UC_BD) * N + d0 + r % UC_BD) * LR + kc * UC_KC + q * 8);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    float acc[2][HC][2][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int c = 0; c < HC; ++c)
#pragma unroll
            for (int s = 0; s < 2; ++s)
#pragma unroll
                for (int i = 0; i < 4; ++i) acc[mt][c][s][i] = 0.0f;
    constexpr int NK = LR / UC_KC;
    stage(0, 0);
    for (int kc = 0; kc < NK; ++kc) {
        const int buf = kc & 1;
        if (kc + 1 < NK) {
            stage(kc + 1, buf ^ 1);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < UC_KC / 16; ++ks) {
            unsigned a[2][4];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) uc_ldm4(a[mt], &sA[buf][wm * 32 + mt * 16 + (lane & 15)][ks * 16 + (lane >> 4) * 8]);
#pragma unroll
            for (int c = 0; c < HC; ++c) {
                unsigned b[4];   // b[0..1]: columns 0-7 of this warp's 16, b[2..3]: 8-15 (k 0-7, 8-15 each)
                uc_ldm4(b, &sB[buf][c * UC_BD + wn * 16 + (lane & 7) + ((lane >> 4) << 3)][ks * 16 + ((lane >> 3) & 1) * 8]);
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    uc_mma(acc[mt][c][0], a[mt], b[0], b[1]);
                    uc_mma(acc[mt][c][1], a[mt], b[2], b[3]);
                }
            }
        }
        __syncthreads();   // every read of `buf` before stage(kc + 2) refills it
    }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int64_t t = t0 + wm * 32 + mt * 16 + (lane >> 2) + h * 8;
            if (t >= T) continue;
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                const int d = d0 + wn * 16 + s * 8 + (lane & 3) * 2;
                float sm[2] = {0.0f, 0.0f};
#pragma unroll
                for (int c = 0; c < HC; ++c) {
                    const float2 r2 = *reinterpret_cast<const float2*>(R + t * D + c * N + d);
                    const float2 w2 = *reinterpret_cast<const float2*>(w + c * N + d);
                    const float rsc = rs[t * HC + c];
                    const float x0 = r2.x * rsc * w2.x, x1 = r2.y * rsc * w2.y;   // gr_mix_r_kernel's order
                    sm[0] = fmaf(x0, sigm(acc[mt][c][s][h * 2 + 0]), sm[0]);
                    sm[1] = fmaf(x1, sigm(acc[mt][c][s][h * 2 + 1]), sm[1]);
                }
                sm[0] /= (float) HC;
                sm[1] /= (float) HC;
                *reinterpret_cast<float2*>(mixed + t * N + d) = make_float2(sm[0], sm[1]);
                if (mixed16)
                    *reinterpret_cast<uint32_t*>(mixed16 + t * N + d) =
                        (uint32_t) act16(sm[0]) | ((uint32_t) act16(sm[1]) << 16);
                if (mixed_h)
                    *reinterpret_cast<uint32_t*>(mixed_h + t * N + d) = (uint32_t) hf(sm[0]) | ((uint32_t) hf(sm[1]) << 16);
            }
        }
#endif
}
#endif
}  // namespace

void kv_append(const float* K, const float* V, int64_t T, int64_t pos0, const int32_t* page_table, int64_t page_size,
               uint16_t* k_pool, uint16_t* v_pool, int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
               void* stream, const strata::kernels::KvHostPools* host, const strata::kernels::KvHostPools* stage) {
    if (T <= 0) return;
    kv_append_kernel<<<dim3((unsigned) T, 2, 8), 64, 0, (cudaStream_t) stream>>>(
        K, V, pos0, page_table, page_size, k_pool, v_pool, k_q, v_q, k_scale, v_scale,
        host ? *host : strata::kernels::KvHostPools{}, stage ? *stage : strata::kernels::KvHostPools{});
    check("kv_append");
}
void set_act_f16(bool on) {
#if !defined(__HIPCC__)
    (void) on;   // CUDA: the activation image is always BF16
    return;
#else
    const int v = on ? 1 : 0;
    if (cudaMemcpyToSymbol(g_act_f16, &v, sizeof v) != cudaSuccess) {
        std::fprintf(stderr, "prefill: setting the FP16 activation image failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        std::exit(1);
    }
#endif
}
void set_f16_range(bool on) {
#if defined(__HIPCC__)
    const int v = on ? 1 : 0;
    if (cudaMemcpyToSymbol(g_f16_range_on, &v, sizeof v) != cudaSuccess) {
        std::fprintf(stderr, "prefill: setting the FP16 range check failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        std::exit(1);
    }
#else
    (void) on;
#endif
}
bool f16_range_read(float& max_abs, unsigned long long& over, bool reset) {
#if !defined(__HIPCC__)
    (void) max_abs; (void) over; (void) reset;
    return false;
#else
    unsigned bits = 0; unsigned long long o = 0;
    if (cudaMemcpyFromSymbol(&bits, g_f16_range_max, sizeof bits) != cudaSuccess ||
        cudaMemcpyFromSymbol(&o, g_f16_range_over, sizeof o) != cudaSuccess) { cudaGetLastError(); return false; }
    union { unsigned u; float f; } c; c.u = bits; max_abs = c.f; over = o;
    if (reset) {
        const unsigned z = 0; const unsigned long long zz = 0;
        cudaMemcpyToSymbol(g_f16_range_max, &z, sizeof z); cudaMemcpyToSymbol(g_f16_range_over, &zz, sizeof zz);
    }
    return true;
#endif
}
void to_f16(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    to_f16_kernel<<<(unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 256, 0, (cudaStream_t) stream>>>(x, y, n);
    check("to_f16");
}
void round_f16(const float* x, float* y, int64_t n, void* stream) {
    if (n <= 0) return;
    round_f16_kernel<<<(unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 256, 0, (cudaStream_t) stream>>>(x, y, n);
    check("round_f16");
}
void to_bf16(const float* x, uint16_t* y, int64_t n, void* stream, uint16_t* ylo) {
    if (n <= 0) return;
    to_bf16_kernel<<<(unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 256, 0, (cudaStream_t) stream>>>(x, y, ylo, n);
    check("to_bf16");
}

void gr_norm(const float* R, const float* w_norm, float eps, float* xn, uint16_t* xn16, int64_t T, void* stream,
             uint16_t* xn16_lo, bool f16img) {
    gr_norm_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(R, w_norm, eps, xn, xn16, xn16_lo, f16img ? 1 : 0);
    check("gr_norm");
}
void gr_norm_rs(const float* R, const float* w_norm, float eps, float* rs, uint16_t* xn16, int64_t T, void* stream,
                uint16_t* xn16_lo, int64_t ldx, bool f16img) {
    gr_norm_rs_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(R, w_norm, eps, rs, xn16, xn16_lo,
                                                                                ldx > 0 ? ldx : D, f16img ? 1 : 0);
    check("gr_norm_rs");
}
void gr_mix_r(const float* R, const float* rs, const float* w_norm, const float* gated, float* mixed, uint16_t* mixed16,
              int64_t T, void* stream, uint16_t* mixed_h, uint16_t* mixed16_lo) {
    gr_mix_r_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(R, rs, w_norm, gated, mixed, mixed16, T,
                                                                          mixed_h, mixed16_lo);
    check("gr_mix_r");
}
bool gr_upmix(const uint16_t* lo16, const uint16_t* w_up, const float* R, const float* rs, const float* w_norm,
              float* mixed, uint16_t* mixed16, uint16_t* mixed_h, int64_t T, void* stream) {
#if defined(__HIPCC__)
    static const bool gfx11 = [] {
        int dev = 0;
        cudaDeviceProp p;
        if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&p, dev) != cudaSuccess) return false;
        return strata::kernels::gfx_arch_is_gfx11_wmma(p.gcnArchName);
    }();
    if (!gfx11 || T <= 0) return false;
    gr_upmix_kernel<<<dim3(N / UM_D, (unsigned) ((T + UM_T - 1) / UM_T)), 512, 0, (cudaStream_t) stream>>>(
        lo16, w_up, R, rs, w_norm, mixed, mixed16, mixed_h, T);
    check("gr_upmix");
    return true;
#else
    static int ok_dev[64] = {};   // per device: 0 not asked yet, 1 sm_80+, 2 no
    int dev = 0;
    if (T <= 0 || cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (ok_dev[dev] == 0) {
        int major = 0;
        ok_dev[dev] = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                      strata::cc_major_of(major) >= 8 ? 1 : 2;
        cudaGetLastError();
    }
    if (ok_dev[dev] != 1) return false;
    gr_upmix_cuda_kernel<<<dim3(N / UC_BD, (unsigned) ((T + UC_BM - 1) / UC_BM)), 256, 0, (cudaStream_t) stream>>>(
        lo16, w_up, R, rs, w_norm, mixed, mixed16, mixed_h, T);
    check("gr_upmix");
    return true;
#endif
}
void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w_norm_next, float eps,
                      float* rs, uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo, int64_t ldx, bool f16img) {
    gr_write_norm_rs_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(R, bo, inj, inj_ld, w_norm_next, eps,
                                                                                      rs, xn16, xn16_lo,
                                                                                      ldx > 0 ? ldx : D, f16img ? 1 : 0);
    check("gr_write_norm_rs");
}
void gr_write_cvec_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* v_l,
                           const float* s_l, const int* on, int mode, const float* w_norm_next, float eps, float* rs,
                           uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo, int64_t ldx, bool f16img) {
    gr_write_cvec_norm_rs_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(
        R, bo, inj, inj_ld, v_l, s_l, on, mode, w_norm_next, eps, rs, xn16, xn16_lo, ldx > 0 ? ldx : D, f16img ? 1 : 0);
    check("gr_write_cvec_norm_rs");
}
void gr_silu(const float* lo, uint16_t* lo16, int64_t T, void* stream, uint16_t* lo16_lo) {
    gr_silu_kernel<<<blocks_for(T * LR), 256, 0, (cudaStream_t) stream>>>(lo, lo16, lo16_lo, T * LR);
    check("gr_silu");
}
void gr_mix(const float* xn, const float* gated, float* mixed, uint16_t* mixed16, int64_t T, void* stream,
            uint16_t* mixed_h, uint16_t* mixed16_lo) {
    gr_mix_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(xn, gated, mixed, mixed16, T, mixed_h, mixed16_lo);
    check("gr_mix");
}
void gr_write(float* R, const float* bo, const float* inj, int64_t inj_ld, int64_t T, void* stream) {
    gr_write_kernel<<<blocks_for(T * D), 256, 0, (cudaStream_t) stream>>>(R, bo, inj, inj_ld, T);
    check("gr_write");
}
void gr_broadcast(const float* e, float* R, int64_t T, void* stream) {
    gr_broadcast_kernel<<<blocks_for(T * D), 256, 0, (cudaStream_t) stream>>>(e, R, T);
    check("gr_broadcast");
}
void gdn_gates(const float* ab, const float* dt, const float* ssm_a, float* gate, float* beta, int64_t T, void* stream) {
    gdn_gates_kernel<<<blocks_for(T * HV), 256, 0, (cudaStream_t) stream>>>(ab, dt, ssm_a, gate, beta, T);
    check("gdn_gates");
}
void gdn_conv(float* history, const float* qkv, const float* conv_w, float* h, int64_t T, float eps, void* stream) {
    static const bool serial = std::getenv("STRATA_GDN_CONV_SERIAL") != nullptr;   // the old walk (A/B)
    if (serial || T <= CONV_TILE) {
        gdn_conv_kernel<<<C / 128, 128, 0, (cudaStream_t) stream>>>(history, qkv, conv_w, h, T);
    } else {
        static const int fused_mode = [] { const char* v = std::getenv("STRATA_GDN_CONVL2"); return v ? std::atoi(v) : 0; }();
        const bool fused = fused_mode != 0;
        if (fused_mode >= 2) {
            gdn_conv_w_kernel<<<dim3(C / 128, (unsigned) ((T + 4 * CONVW_TOK - 1) / (4 * CONVW_TOK))), 128, 0,
                                (cudaStream_t) stream>>>(history, qkv, conv_w, h, T, eps);
            gdn_conv_hist_kernel<<<C / 128, 128, 0, (cudaStream_t) stream>>>(history, qkv, T);
            check("gdn_conv");
            return;
        }
        if (fused) {
            gdn_conv_l2_kernel<<<dim3(C / 128, (unsigned) ((T + CONV_TILE - 1) / CONV_TILE)), 128, 0,
                                 (cudaStream_t) stream>>>(history, qkv, conv_w, h, T, eps);
            gdn_conv_hist_kernel<<<C / 128, 128, 0, (cudaStream_t) stream>>>(history, qkv, T);
            check("gdn_conv");
            return;
        }
        gdn_conv_tiled_kernel<<<dim3(C / 128, (unsigned) ((T + CONV_TILE - 1) / CONV_TILE)), 128, 0,
                                (cudaStream_t) stream>>>(history, qkv, conv_w, h, T);
        gdn_conv_hist_kernel<<<C / 128, 128, 0, (cudaStream_t) stream>>>(history, qkv, T);
    }
    gdn_l2_kernel<<<dim3(2 * HK, (unsigned) T), S, 0, (cudaStream_t) stream>>>(h, eps);
    check("gdn_conv");
}
// The recurrence dispatch, shared by the two entry points below so the state-only form cannot drift from the full
// one.  `with_out` false runs exactly the same kernels minus the output normalization (the CUDA-only KVA path).
static void gdn_recurrence_impl(int variant, float* state, const float* h, const float* gate, const float* beta,
                                const float* z, const float* gamma, float eps, float* y, uint16_t* y16, int64_t T,
                                void* stream, int64_t ld16, bool with_out) {
    if (ld16 <= 0) ld16 = (int64_t) HV * S;
    static const bool serial = std::getenv("STRATA_GDN_REC_HEADS") != nullptr;   // the one-block-per-head kernel (A/B)
#if defined(__HIPCC__)
    if (variant == 0 || variant == 1) {
        // STRATA_GDN_CHUNKED=1 (opt-in): the chunked recurrence on the gfx11 matrix cores, FP16 operands (other bits;
        // a shorter chunk or another part takes the kernels below)
        static const bool chunked = [] { const char* v = std::getenv("STRATA_GDN_CHUNKED"); return v != nullptr && std::atoi(v) != 0; }();
        if (chunked && !serial && gdn_rec_wmma(state, h, gate, beta, z, gamma, eps, y, y16, T, (cudaStream_t) stream, ld16)) {
            check("gdn_recurrence (chunked, WMMA)");
            return;
        }
    }
    if (variant == 3) {   // diagnostics: the quad recurrence + the old norm kernel
        if (T > 0) {
            gdn_rec_quad_kernel<<<HV * (S / QCB), QTH, 0, (cudaStream_t) stream>>>(state, h, gate, beta, y, T);
            if (with_out)
                gdn_out_norm_kernel<<<dim3((unsigned) T, HV), S, 0, (cudaStream_t) stream>>>(z, gamma, eps, y, y16, ld16);
        }
        return;
    }
    if (variant == 4 || variant == 5) {   // diagnostics: the quad recurrence's PP / two-column kernels + the norm
        if (T > 0) {
            if (variant == 5)
                gdn_rec_quad2c_kernel<<<HV * (S / C2CB), QTH, 0, (cudaStream_t) stream>>>(state, h, gate, beta, y, T);
            else
                gdn_rec_quad_pp_kernel<<<HV * (S / QCB), QTH, 0, (cudaStream_t) stream>>>(state, h, gate, beta, y, T);
            if (with_out)
                gdn_out_norm_loop_kernel<true><<<(unsigned) std::min<int64_t>(T * HV, 4096), S, 0, (cudaStream_t) stream>>>(
                    z, gamma, eps, y, y16, T * HV, ld16);
        }
        check("gdn_recurrence (quad pp)");
        return;
    }
    if (variant == 1) {
        if (T > 0) {
            static const int pp = [] { const char* v = std::getenv("STRATA_GDN_PP"); return v ? std::atoi(v) : 0; }();
            if (pp == 2)
                gdn_rec_quad2c_kernel<<<HV * (S / C2CB), QTH, 0, (cudaStream_t) stream>>>(state, h, gate, beta, y, T);
            else if (pp)
                gdn_rec_quad_pp_kernel<<<HV * (S / QCB), QTH, 0, (cudaStream_t) stream>>>(state, h, gate, beta, y, T);
            else
                gdn_rec_quad_kernel<<<HV * (S / QCB), QTH, 0, (cudaStream_t) stream>>>(state, h, gate, beta, y, T);
            static const bool noy = [] { const char* v = std::getenv("STRATA_GDN_NOY"); return v && std::atoi(v) != 0; }();
            if (with_out && noy)
                gdn_out_norm_loop_kernel<false><<<(unsigned) std::min<int64_t>(T * HV, 4096), S, 0, (cudaStream_t) stream>>>(
                    z, gamma, eps, y, y16, T * HV, ld16);
            else if (with_out)
                gdn_out_norm_loop_kernel<true><<<(unsigned) std::min<int64_t>(T * HV, 4096), S, 0, (cudaStream_t) stream>>>(
                    z, gamma, eps, y, y16, T * HV, ld16);
        }
        check("gdn_recurrence (quad)");
        return;
    }
#else
    (void) variant;   // the quad recurrence kernels are AMD-only
#endif
#if defined(STRATA_EXPERIMENTAL_SM60)
    // v100/gdn-chunk: the chunked recurrence (src/prefill/gdn_chunk.cu) on Volta, where the token chain is the
    // cost.  STRATA_GDN_CHUNK=0 forces the recurrence, =1 forces the chunked path (the parity test's A/B switch);
    // unset: cc == 7.0 only, every other architecture keeps the recurrence bit-for-bit as before.  Read per call
    // on purpose - cheap next to the work, and the tests flip it between two launches.
    const char* chunk_v = std::getenv("STRATA_GDN_CHUNK");
    const int chunk_env = chunk_v && *chunk_v ? std::atoi(chunk_v) : -1;
    static const bool volta = gdn_chunk_available();   // runtime cc == 7.0 (the kernels are compiled for sm_70 only)
    const bool chunked = !serial && T > 0 && (chunk_env >= 0 ? chunk_env != 0 : false);
    // v100/gdn-chunk: the sync-free warp-reduction recurrence (bit-exact vs gdn_rec_cols_pipe) is the
    // measured-fast path on Volta; the chunked recurrence is opt-in (STRATA_GDN_CHUNK=1/2) - correct and
    // measured, but slower than this on the V100 (bench/gdn_chunk_bench).  STRATA_GDN_REC_WARP=0 restores
    // the old pipelined kernel bit-for-bit.
    // STRATA_GDN_REC_WARP (Volta only, opt-in): unset or 0 -> the pipelined kernel (the default; the chain split changes the
    // summation order, so it is not the default without a V100 check here), 2 -> the chain-split kernel (measured +6..9%, FP32-level), the
    // old pipelined kernel bit-for-bit is 0, 1 -> the sync-free bit-exact control.
    const char* warp_v = std::getenv("STRATA_GDN_REC_WARP");
    const int warp_env = warp_v && *warp_v ? std::atoi(warp_v) : 0;
    const bool warp_rec = !serial && !chunked && T > 0 && volta && warp_env != 0;
    if (chunked && T >= 64) {
        // the chunked prefix + the recurrence over the tail (< 64 tokens), then the out norm over all of T
        const int64_t T64 = T - T % 64;
        gdn_chunk_recurrence(state, h, gate, beta, y, T64, stream);
        if (T > T64)
            gdn_rec_cols_pipe_kernel<<<HV * NCB, dim3(CB, RG), 0, (cudaStream_t) stream>>>(
                state, h + T64 * C, gate + T64 * HV, beta + T64 * HV, y + T64 * (size_t) HV * S, T - T64);
        if (with_out)
            gdn_out_norm_kernel<<<dim3((unsigned) T, HV), S, 0, (cudaStream_t) stream>>>(z, gamma, eps, y, y16, ld16);
    } else if (warp_rec) {
        // =2: the 4-accumulator chain split on the pipelined structure (FP32-level, not bit-exact);
        // =1: the sync-free bit-exact warp reduction (measured slower - kept as the bitwise control)
        if (warp_env == 2)
            gdn_rec_cols_pipe_fast_kernel<<<HV * NCB, dim3(CB, RG), 0, (cudaStream_t) stream>>>(state, h, gate, beta,
                                                                                                y, T);
        else
            gdn_rec_cols_warp_kernel<<<HV * NCB, S, 0, (cudaStream_t) stream>>>(state, h, gate, beta, y, T);
        if (with_out)
            gdn_out_norm_kernel<<<dim3((unsigned) T, HV), S, 0, (cudaStream_t) stream>>>(z, gamma, eps, y, y16, ld16);
    } else
#endif
    if ((serial && with_out) || T <= 0 || variant == 2) {   // the fused kernel needs z/gamma: only with the output
        gdn_rec_kernel<<<HV, dim3(S, RG), 0, (cudaStream_t) stream>>>(state, h, gate, beta, z, gamma, eps, y, y16, T, ld16);
    } else {
        static const bool pipe = [] { const char* v = std::getenv("STRATA_GDN_PIPELINE"); return v == nullptr || std::atoi(v) != 0; }();
        bool normed_any = false;   // the tensor-core scan wrote y16 (the norm kernel below is skipped)
#if !defined(__HIPCC__)
        // STRATA_GDN_CHUNKED=1 (opt-in): the recurrence in chunks for prompt chunks of kGdnChunkedMin+ tokens (other
        // bits; a shorter chunk, a card it does not fit or a failed scratch allocation takes the kernels below)
        static const bool chunked = [] { const char* v = std::getenv("STRATA_GDN_CHUNKED"); return v != nullptr && std::atoi(v) != 0; }();
        // STRATA_GDN_TC=1 (opt-in): the chunked recurrence with the scan on the tensor cores (gdn_tc.cuh) for prompt
        // chunks of kGdnTcMin+ tokens: within 1e-4 of the kernels below (gdn_rec_parity), 3.75x gdn_rec_kh_kernel on a
        // CMP 170HX; a shorter chunk, a card it does not fit or a failed scratch allocation takes the kernels below
        static const bool tc = [] { const char* v = std::getenv("STRATA_GDN_TC"); return v != nullptr && std::atoi(v) != 0; }();
        // A test's probe (gdn_probe.hpp) reads back the path that ran and may force one; without a probe this is the
        // dispatch it was.  A forced path that cannot run launches nothing (selected = none, reason says why).
        GdnProbe* pr = gdn_probe_get();
        const int force = pr ? pr->force : (int) kGdnAuto;
        int sel = kGdnNone, wanted = force;
        bool normed = false;   // the scan wrote y16 itself
        const char* why = "";
        const cudaStream_t cs = (cudaStream_t) stream;
        // the split tensor-core scan can write y16 itself (STRATA_GDN_TC_FUSE=0 off): then the norm kernel below is skipped
        static const bool tc_fuse = [] { const char* v = std::getenv("STRATA_GDN_TC_FUSE"); return v == nullptr || std::atoi(v) != 0; }();
        const GdnTcNorm tcn = {z, gamma, eps, y16, ld16};
        const int tcl = gdn_tc_tiles();
        const GdnTcNorm* tcnp = with_out && tc_fuse && tcl == 1 ? &tcn : nullptr;   // 2 tiles a head: the norm kernel (a head's norm spans both)
        if (force != kGdnAuto) {
            if (force == kGdnTc || force == kGdnTcSplit || force == kGdnTcS2 || force == kGdnTcS1) {
                const bool sp = force != kGdnTc;
                const int np = force == kGdnTcS2 ? 2 : force == kGdnTcS1 ? 1 : 3;
                cudaError_t e = T >= kGdnTcMin ? gdn_rec_tc(state, h, gate, beta, y, T, cs, sp, np, sp ? tcnp : nullptr, sp ? tcl : 1) : cudaErrorInvalidValue;
                if (e == cudaSuccess) { sel = force; normed = sp && tcnp != nullptr; }
                else why = T < kGdnTcMin ? "T below the tc minimum (128)" : gdn_chunk_why(true, e, false, sp);
            } else if (force == kGdnChunked) {
                cudaError_t e = T >= kGdnChunkedMin ? gdn_rec_chunked(state, h, gate, beta, y, T, cs) : cudaErrorInvalidValue;
                if (e == cudaSuccess) sel = kGdnChunked;
                else why = T < kGdnChunkedMin ? "T below the chunked minimum (128)" : gdn_chunk_why(false, e, pr->any_sms);
            } else if (force == kGdnKeyHead) {
                if (gdn_keyhead_ok()) {
                    gdn_rec_kh_kernel<<<HK * NCB, dim3(CB, RG), 0, cs>>>(state, h, gate, beta, y, T);
                    sel = kGdnKeyHead;
                } else why = "keyhead kernel not eligible here (needs sm_80+ with all 64 blocks resident, or STRATA_GDN_KEYHEAD=0 is set)";
            } else if (force == kGdnPipe) {
                gdn_rec_cols_pipe_kernel<<<HV * NCB, dim3(CB, RG), 0, cs>>>(state, h, gate, beta, y, T);
                sel = kGdnPipe;
            } else if (force == kGdnCols) {
                gdn_rec_cols_kernel<<<HV * NCB, dim3(CB, RG), 0, cs>>>(state, h, gate, beta, y, T);
                sel = kGdnCols;
            } else why = "unknown path";
        } else {
            if (tc && T >= kGdnTcMin) {
                wanted = kGdnTc;
                cudaError_t e = gdn_rec_tc(state, h, gate, beta, y, T, cs);
                if (e == cudaSuccess) sel = kGdnTc;
                else {   // the 99-KB cards (sm_86 / 89 / 120): the single-buffered v / T / P plan
                    if (pr) why = gdn_chunk_why(true, e, false);
                    // STRATA_GDN_TC_NP=1|2 (experiment, not a default): fewer split passes a product on this plan, other bits
                    static const int tc_np = [] { const char* v = std::getenv("STRATA_GDN_TC_NP"); const int x = v ? std::atoi(v) : 3; return x == 1 || x == 2 ? x : 3; }();
                    e = gdn_rec_tc(state, h, gate, beta, y, T, cs, true, tc_np, tcnp, tcl);
                    if (e == cudaSuccess) { sel = kGdnTcSplit; why = ""; normed = tcnp != nullptr; }
                }
            }
            if (sel == kGdnNone && chunked && T >= kGdnChunkedMin) {
                if (wanted == kGdnAuto) wanted = kGdnChunked;
                const cudaError_t e = gdn_rec_chunked(state, h, gate, beta, y, T, cs);
                if (e == cudaSuccess) sel = kGdnChunked;
                else if (pr && wanted == kGdnChunked) why = gdn_chunk_why(false, e, false);
            }
            if (sel == kGdnNone) {
                if (pipe && gdn_keyhead_ok()) {   // the value heads of a key head in one thread (same bits)
                    gdn_rec_kh_kernel<<<HK * NCB, dim3(CB, RG), 0, cs>>>(state, h, gate, beta, y, T);
                    sel = kGdnKeyHead;
                } else if (pipe) {   // the software-pipelined loads (same bits)
                    gdn_rec_cols_pipe_kernel<<<HV * NCB, dim3(CB, RG), 0, cs>>>(state, h, gate, beta, y, T);
                    sel = kGdnPipe;
                } else {
                    gdn_rec_cols_kernel<<<HV * NCB, dim3(CB, RG), 0, cs>>>(state, h, gate, beta, y, T);
                    sel = kGdnCols;
                }
                if (wanted == kGdnAuto) wanted = sel;
            }
        }
        if (pr) { pr->selected = sel; pr->wanted = wanted == kGdnAuto ? sel : wanted; pr->reason = why; ++pr->calls; }
        normed_any = normed;
        if (sel == kGdnNone) { check("gdn_recurrence (forced path not eligible: nothing launched)"); return; }
#else
        if (pipe)   // the software-pipelined loads (same bits)
            gdn_rec_cols_pipe_kernel<<<HV * NCB, dim3(CB, RG), 0, (cudaStream_t) stream>>>(state, h, gate, beta, y, T);
        else
            gdn_rec_cols_kernel<<<HV * NCB, dim3(CB, RG), 0, (cudaStream_t) stream>>>(state, h, gate, beta, y, T);
#endif
        if (with_out && !normed_any)
            gdn_out_norm_kernel<<<dim3((unsigned) T, HV), S, 0, (cudaStream_t) stream>>>(z, gamma, eps, y, y16, ld16);
    }
    check("gdn_recurrence");
}
void gdn_recurrence_variant(int variant, float* state, const float* h, const float* gate, const float* beta,
                            const float* z, const float* gamma, float eps, float* y, uint16_t* y16, int64_t T,
                            void* stream, int64_t ld16) {
    gdn_recurrence_impl(variant, state, h, gate, beta, z, gamma, eps, y, y16, T, stream, ld16, /*with_out=*/true);
}
void gdn_recurrence(float* state, const float* h, const float* gate, const float* beta, const float* z,
                    const float* gamma, float eps, float* y, uint16_t* y16, int64_t T, void* stream, int64_t ld16) {
    // Aurora (S23): four lanes per column + the grid-stride norm, the same bits (tests/hip/gdn_rec_head.cpp);
    // STRATA_GDN_HEAD=1 (on by default on gfx1151; off with STRATA_GDN_HEAD=0 or STRATA_GDN_REC_HEADS) takes them
    // (AMD builds only; the arch defaults set STRATA_GDN_HEAD=1 on gfx1151 and leave other cards on the earlier kernels)
    static const bool head = [] { const char* v = std::getenv("STRATA_GDN_HEAD"); return v != nullptr && std::atoi(v) != 0; }();
    static const bool serial = std::getenv("STRATA_GDN_REC_HEADS") != nullptr;
    gdn_recurrence_impl(head && !serial ? 1 : 0, state, h, gate, beta, z, gamma, eps, y, y16, T, stream, ld16, true);
}
/// KVA: the same recurrence, bit for bit, without the output normalisation (its readers - the output projection
/// and the gated residual write - have no consumer when only the state advances).  `y` is still the scratch the
/// recurrence kernels write; `z`/`gamma` are not read.
bool gdn_conv_recurrence_fused(float* history, const float* qkv, const float* conv_w, float* hbuf, float* state, const float* gate,
                               const float* beta, const float* z, const float* gamma, float eps, float l2eps, float* y,
                               uint16_t* y16, int64_t T, void* stream, int64_t ld16) {
#if defined(__HIPCC__)
    return gdn_fused_impl(history, qkv, conv_w, hbuf, state, gate, beta, z, gamma, eps, l2eps, y, y16, T, stream, ld16);
#else
    (void) history; (void) qkv; (void) conv_w; (void) hbuf; (void) state; (void) gate; (void) beta; (void) z; (void) gamma; (void) eps;
    (void) l2eps; (void) y; (void) y16; (void) T; (void) stream; (void) ld16;
    return false;
#endif
}
void gdn_recurrence_state_only(float* state, const float* h, const float* gate, const float* beta, float* y,
                               int64_t T, void* stream) {
    static const bool head = [] { const char* v = std::getenv("STRATA_GDN_HEAD"); return v != nullptr && std::atoi(v) != 0; }();
    static const bool serial = std::getenv("STRATA_GDN_REC_HEADS") != nullptr;
    gdn_recurrence_impl(head && !serial ? 1 : 0, state, h, gate, beta, nullptr, nullptr, 0.0f, y, nullptr, T,
                        stream, 0, /*with_out=*/false);
}
void route(const float* logits, int32_t* ids, float* weights, int64_t T, int64_t n_expert, void* stream) {
    if (n_expert == 512)
        route_kernel<16><<<(unsigned) ((T + 7) / 8), 256, 0, (cudaStream_t) stream>>>(logits, ids, weights, T);
    else if (n_expert == 256)
        route_kernel<8><<<(unsigned) ((T + 7) / 8), 256, 0, (cudaStream_t) stream>>>(logits, ids, weights, T);
    else
        strata::kernels::router_top10(logits, (int) T, (int) n_expert, 10, ids, weights, stream);
    check("route");
}
void blob_dequant(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream) {
    blob_dequant_kernel<false><<<blocks_for(1280LL * 640 + 2560LL * 160), 256, 0, (cudaStream_t) stream>>>(blob, gu16, down16);
    check("blob_dequant");
}
void blob_dequant_f16(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream) {
    blob_dequant_kernel<true><<<blocks_for(1280LL * 640 + 2560LL * 160), 256, 0, (cudaStream_t) stream>>>(blob, gu16, down16);
    check("blob_dequant_f16");
}
void swiglu_interleaved(const float* gu, uint16_t* h16, int64_t n, void* stream) {
    if (n <= 0) return;
    swiglu_il_kernel<<<blocks_for(n * 640), 256, 0, (cudaStream_t) stream>>>(gu, h16, n);
    check("swiglu_interleaved");
}
void swiglu_split_f16(const float* gu, uint16_t* h16, int64_t rows, int n_ff, bool interleaved, void* stream) {
    if (rows <= 0) return;
    swiglu_split_f16_kernel<<<blocks_for(rows * n_ff), 256, 0, (cudaStream_t) stream>>>(gu, h16, rows, n_ff, interleaved);
    check("swiglu_split_f16");
}
void swiglu_pair(const float* g, const float* u, uint16_t* h16, int64_t n, void* stream) {
    swiglu_pair_kernel<<<blocks_for(n * 640), 256, 0, (cudaStream_t) stream>>>(g, u, h16, n);
    check("swiglu_pair");
}
namespace {
__global__ void copy_i32_kernel(int32_t* __restrict__ dst, const int32_t* __restrict__ src, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        dst[i] = src[i];
}
}  // namespace
namespace {
__global__ void copy_f4_kernel(float4* __restrict__ dst, const float4* __restrict__ src, int64_t n4) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (int64_t) gridDim.x * blockDim.x)
        dst[i] = src[i];
}
}  // namespace
void copy_f32_wide(float* dst, const float* src, int64_t n, void* stream) {
    if (n <= 0) return;
    const int64_t n4 = n / 4, b = std::min<int64_t>((n4 + 255) / 256, 4096);
    if (n4 > 0) copy_f4_kernel<<<(unsigned) b, 256, 0, (cudaStream_t) stream>>>((float4*) dst, (const float4*) src, n4);
    if (n % 4) copy_i32((int32_t*) dst + n4 * 4, (const int32_t*) src + n4 * 4, n % 4, stream);
    check("copy_f32_wide");
}
void copy_i32(int32_t* dst, const int32_t* src, int64_t n, void* stream) {
    if (n <= 0) return;
    const int64_t b = (n + 255) / 256;
    copy_i32_kernel<<<(unsigned) (b < 256 ? b : 256), 256, 0, (cudaStream_t) stream>>>(dst, src, n);
    check("copy_i32");
}
void gather_rows16(const uint16_t* x16, const int32_t* src, uint16_t* dst16, int64_t n, int64_t width, void* stream) {
    if (n <= 0) return;
    gather_rows16_kernel<<<blocks_for(n * (width / 8)), 256, 0, (cudaStream_t) stream>>>(x16, src, dst16, n, width);
    check("gather_rows16");
}
namespace {
__global__ void moe_combine_peer_kernel(const float* __restrict__ Dm, const int32_t* __restrict__ slot,
                                        const float* __restrict__ w, const float* __restrict__ shared,
                                        const float* __restrict__ sg, const float* __restrict__ peer, int64_t rows_local,
                                        float* __restrict__ bo, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    const int64_t t = i / N, d = i % N;
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < 10; ++k) {
        const int32_t r = slot[t * 10 + k];
        if (r < rows_local) s = fmaf(w[t * 10 + k], Dm[(int64_t) r * N + d], s);
    }
    bo[i] = (s + peer[i]) + shared[i] * sigm(sg[t]);
}
__global__ void peer_scatter_add_kernel(float* __restrict__ sum, const float* __restrict__ rows,
                                        const float* __restrict__ wk, const int32_t* __restrict__ pair, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * N) return;
    const int64_t r = i / N, d = i % N;
    const int32_t p = pair[r];
    float* o = sum + (int64_t) (p / 10) * N + d;
    *o = fmaf(wk[p], rows[i], *o);
}
__global__ void moe_combine_peer16_kernel(const float* __restrict__ Dm, const int32_t* __restrict__ slot,
                                          const float* __restrict__ w, const float* __restrict__ shared,
                                          const float* __restrict__ sg, const uint16_t* __restrict__ peer16,
                                          int64_t rows_local, float* __restrict__ bo, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    const int64_t t = i / N, d = i % N;
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < 10; ++k) {
        const int32_t r = slot[t * 10 + k];
        if (r < rows_local) s = fmaf(w[t * 10 + k], Dm[(int64_t) r * N + d], s);
    }
    bo[i] = (s + __half2float(__ushort_as_half(peer16[i]))) + shared[i] * sigm(sg[t]);
}
__global__ void peer_gather_add_kernel(float* __restrict__ sum, const float* __restrict__ rows, int64_t r0,
                                       const float* __restrict__ wk, const int32_t* __restrict__ pair,
                                       const int32_t* __restrict__ tok, const int32_t* __restrict__ start,
                                       const int32_t* __restrict__ list) {
    const int64_t d = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= N) return;
    const int b = blockIdx.y;
    float* o = sum + (int64_t) tok[b] * N + d;
    float s = *o;
    for (int i = start[b]; i < start[b + 1]; ++i) {
        const int r = list[i];
        s = fmaf(wk[pair[r]], rows[(r - r0) * N + d], s);
    }
    *o = s;
}
__global__ void sums_to_f16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = hf_sat(x[i]);
}
__global__ void f16_to_f32_wide_kernel(float4* __restrict__ dst, const uint4* __restrict__ src, int64_t n8) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // eight values
    if (i >= n8) return;
    const uint4 v = src[i];
    const __half2* h = reinterpret_cast<const __half2*>(&v);
    const float2 a = __half22float2(h[0]), b = __half22float2(h[1]), c = __half22float2(h[2]), e = __half22float2(h[3]);
    dst[2 * i] = make_float4(a.x, a.y, b.x, b.y);
    dst[2 * i + 1] = make_float4(c.x, c.y, e.x, e.y);
}
}  // namespace
void moe_combine_peer16(const float* Dm, const int32_t* slot, const float* w, const float* shared, const float* sg,
                        const uint16_t* peer16, int64_t rows_local, float* bo, int64_t T, void* stream) {
    moe_combine_peer16_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(Dm, slot, w, shared, sg, peer16,
                                                                                      rows_local, bo, T);
    check("moe_combine_peer16");
}
void peer_gather_add(float* sum, const float* rows, int64_t r0, const float* wk, const int32_t* pair, const int32_t* tok,
                     const int32_t* start, const int32_t* list, int64_t n_tok, void* stream) {
    if (n_tok <= 0) return;
    peer_gather_add_kernel<<<dim3((unsigned) ((N + 255) / 256), (unsigned) n_tok), 256, 0, (cudaStream_t) stream>>>(
        sum, rows, r0, wk, pair, tok, start, list);
    check("peer_gather_add");
}
void sums_to_f16(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    sums_to_f16_kernel<<<blocks_for(n), 256, 0, (cudaStream_t) stream>>>(x, y, n);
    check("sums_to_f16");
}
void f16_to_f32_wide(float* dst, const uint16_t* src, int64_t n, void* stream) {
    if (n <= 0) return;
    f16_to_f32_wide_kernel<<<blocks_for(n / 8), 256, 0, (cudaStream_t) stream>>>(reinterpret_cast<float4*>(dst),
                                                                                    reinterpret_cast<const uint4*>(src), n / 8);
    check("f16_to_f32_wide");
}
void moe_combine_peer(const float* Dm, const int32_t* slot, const float* w, const float* shared, const float* sg,
                      const float* peer, int64_t rows_local, float* bo, int64_t T, void* stream) {
    moe_combine_peer_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(Dm, slot, w, shared, sg, peer, rows_local,
                                                                                    bo, T);
    check("moe_combine_peer");
}
void peer_scatter_add(float* sum, const float* rows, const float* wk, const int32_t* pair, int64_t n, void* stream) {
    if (n <= 0) return;
    peer_scatter_add_kernel<<<blocks_for(n * N), 256, 0, (cudaStream_t) stream>>>(sum, rows, wk, pair, n);
    check("peer_scatter_add");
}
void moe_combine(const float* Dm, const int32_t* slot, const float* w, const float* shared, const float* sg, float* bo,
                 int64_t T, void* stream, const float* row_sd) {
#ifndef STRATA_W_NO_COMB
    if (((reinterpret_cast<uintptr_t>(Dm) | reinterpret_cast<uintptr_t>(shared) | reinterpret_cast<uintptr_t>(bo)) & 15) == 0) {
        moe_combine4_kernel<<<blocks_for(T * (N / 4)), 256, 0, (cudaStream_t) stream>>>(
            Dm, slot, w, reinterpret_cast<const float4*>(shared), sg, reinterpret_cast<float4*>(bo), T, row_sd);
        check("moe_combine");
        return;
    }
#endif
    moe_combine_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(Dm, slot, w, shared, sg, bo, T, row_sd);
    check("moe_combine");
}
void moe_shared_init(const float* shared, const float* sg, float* acc, int64_t T, void* stream) {
    if (T <= 0) return;
    moe_shared_init_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(shared, sg, acc, T);
    check("moe_shared_init");
}
void moe_scatter_add(const float* Dm, const int32_t* u_tok, const int32_t* u_off, const int32_t* u_rows,
                     const float* u_w, int64_t n_u, float* acc, void* stream) {
    if (n_u <= 0) return;
    const dim3 grid((unsigned) n_u, (unsigned) ((N + 255) / 256));
    moe_scatter_add_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(Dm, u_tok, u_off, u_rows, u_w, acc);
    check("moe_scatter_add");
}
void rms_rows(float* x, const float* w, int64_t rows, int64_t cols, int64_t ld, float eps, void* stream) {
    if (rows <= 0) return;
    rms_rows_kernel<<<(unsigned) rows, 256, 0, (cudaStream_t) stream>>>(x, w, cols, ld, eps);
    check("rms_rows");
}
void rope(float* x, int64_t T, int64_t heads, int64_t dim, int64_t ld, int64_t pos0,
          const strata::kernels::RopeScaling& scaling, void* stream) {
    // The engine validates the resolved config at startup with the same rule (generate.cpp), so this only
    // fires for a caller that bypassed it; the prompt path has no error return here, so it stops the process.
    if (const char* why = strata::kernels::rope_scaling_invalid(scaling)) {
        std::fprintf(stderr, "prefill rope: invalid rope scaling: %s\n", why);
        std::exit(1);
    }
    const float theta_scale = powf((float) scaling.freq_base, -2.0f / 64.0f);
    const strata::kernels::RopeKernelArgs k = scaling.kernel_args(64);   // none: the identity constants
    const strata::kernels::RopeTab rt = strata::kernels::rope_table_for(scaling);
    if (rt.cos != nullptr)
        rope_kernel<true><<<(unsigned) (T * heads), 32, 0, (cudaStream_t) stream>>>(
            x, heads, dim, ld, pos0, theta_scale, k.freq_scale, k.corr_low, k.corr_high, k.ext_factor, k.attn_factor,
            strata::kernels::mrope_table(), rt);
    else
        rope_kernel<false><<<(unsigned) (T * heads), 32, 0, (cudaStream_t) stream>>>(
            x, heads, dim, ld, pos0, theta_scale, k.freq_scale, k.corr_low, k.corr_high, k.ext_factor, k.attn_factor,
            strata::kernels::mrope_table(), rt);
    check("rope");
}
void split_q(const float* q_full, float* q, int64_t T, void* stream) {
    split_q_kernel<<<blocks_for(T * 24 * 256), 256, 0, (cudaStream_t) stream>>>(q_full, q, T);
    check("split_q");
}
void gate_attn(const float* attn, const float* q_full, uint16_t* out16, int64_t T, void* stream, int64_t ld16) {
    gate_attn_kernel<<<blocks_for(T * 24 * 256), 256, 0, (cudaStream_t) stream>>>(attn, q_full, out16, T,
                                                                                ld16 > 0 ? ld16 : 24 * 256);
    check("gate_attn");
}

}  // namespace strata::prefill
