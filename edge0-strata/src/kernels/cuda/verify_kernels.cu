// src/kernels/cuda/verify_kernels.cu - see include/strata/kernels/verify_kernels.hpp.
//
// The per-token arithmetic of every kernel here is transcribed from its single-token original (fused_gdn.cu,
// elementwise.cu) with the same operation order, so a verify window reproduces plain decode bit for bit.
#include "strata/kernels/verify_kernels.hpp"
#include "strata/core/geometry.hpp"
#include "strata/core/emulate.hpp"
#define STRATA_Q81_FASTMATH_DIV 1   // #3948: the q8_1 stores divide as native_mmvq.cu (--use_fast_math) does, see q8_1_finite.hpp
#include "strata/kernels/q8_1_finite.hpp"   // #606: q8_1_ds
#include "strata/kernels/window_start.hpp"  // KVA: the draft layer's window floor
#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/pdl.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <utility>
#include <vector>

namespace strata::kernels {
namespace {

constexpr int S = 128;          // GDN state size
static_assert(S == strata::core::kCompiledGeometry.ssm_state_size, "verify_kernels is compiled for core/geometry.hpp");
constexpr int RG = 4;
constexpr int RPG = S / RG;

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

// `commit` (one token): the history then keeps it - [hist1, hist2, x_0], each thread its channel's, after reading them
__device__ __forceinline__ void gdn_conv_l2_body(const int bx, const int by, float* hist, const float* __restrict__ qkv,
                                                              const float* __restrict__ w, float* __restrict__ h,
                                                              int C, int qk_heads, float eps, int t_begin, bool commit) {
    __shared__ float part[S / 32];
    const int t = t_begin + by;
    const int c = bx * S + threadIdx.x;
    // the window of token t: [hist0, hist1, hist2, x_0, ..., x_t], its last four entries
    float win[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = t + j;          // index into [hist(3) | x...]
        win[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
    }
    const float v0 = win[0], v1 = win[1], v2 = win[2], x = qkv[(size_t) t * C + c];
    if (commit) {
        hist[c * 3] = v1;
        hist[c * 3 + 1] = v2;
        hist[c * 3 + 2] = x;
    }
    float sum = v0 * w[c * 4] + v1 * w[c * 4 + 1] + v2 * w[c * 4 + 2] + x * w[c * 4 + 3];
    float y = sum / (1.0f + __expf(-sum));
    if ((int) bx < qk_heads) {
        float sq = y * y;
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
        if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
        __syncthreads();
        const float ss = part[0] + part[1] + part[2] + part[3];
        y *= rsqrtf(ss + eps);
    }
    h[(size_t) t * C + c] = y;
}

__global__ void __launch_bounds__(S) gdn_conv_l2_multi_kernel(float* hist, const float* __restrict__ qkv,
                                                              const float* __restrict__ w, float* __restrict__ h,
                                                              int C, int qk_heads, float eps, int t_begin, bool commit) {
    gdn_conv_l2_body(blockIdx.x, blockIdx.y, hist, qkv, w, h, C, qk_heads, eps, t_begin, commit);
}

__global__ void gdn_conv_commit_kernel(float* __restrict__ hist, const float* __restrict__ qkv, int C,
                                       const int32_t* __restrict__ n_keep) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int n = *n_keep;
    if (n <= 0) return;
    float seq[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = n + j;          // the last three of [hist(3) | x_0..x_{n-1}]
        seq[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
    }
    hist[c * 3] = seq[0];
    hist[c * 3 + 1] = seq[1];
    hist[c * 3 + 2] = seq[2];
}

template <int MAX_T = kVerifyNarrowT, bool EXACT_T = false>
__device__ __forceinline__ void gdn_ab_body(const int bx, const float* __restrict__ x, const uint16_t* __restrict__ wa,
                                                          const uint16_t* __restrict__ wb,
                                                          const float* __restrict__ dt,
                                                          const float* __restrict__ ssm_a, float* __restrict__ gate,
                                                          float* __restrict__ beta, int n, int h_v, int T) {
    const int row = bx * 2 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= 2 * h_v) return;
    const bool is_beta = row >= h_v;
    const int r = is_beta ? row - h_v : row;
    const uint4* w4 = reinterpret_cast<const uint4*>((is_beta ? wb : wa) + (size_t) r * n);
    float acc[MAX_T];
#pragma unroll
    for (int t = 0; t < MAX_T; ++t) acc[t] = 0.0f;
    for (int j = lane; j < n / 8; j += 32) {
        const uint4 wv = __ldg(w4 + j);
        const float w0 = __uint_as_float(wv.x << 16), w1 = __uint_as_float(wv.x & 0xffff0000u);
        const float w2 = __uint_as_float(wv.y << 16), w3 = __uint_as_float(wv.y & 0xffff0000u);
        const float w4f = __uint_as_float(wv.z << 16), w5 = __uint_as_float(wv.z & 0xffff0000u);
        const float w6 = __uint_as_float(wv.w << 16), w7 = __uint_as_float(wv.w & 0xffff0000u);
#pragma unroll
        for (int t = 0; t < MAX_T; ++t) {
            if (!EXACT_T && t >= T) break;
            const float* xt = x + (size_t) t * n;
            const float4 xa = __ldg(reinterpret_cast<const float4*>(xt + j * 8));
            const float4 xb = __ldg(reinterpret_cast<const float4*>(xt + j * 8 + 4));
            float a = acc[t];
            a = fmaf(w0, xa.x, a); a = fmaf(w1, xa.y, a);
            a = fmaf(w2, xa.z, a); a = fmaf(w3, xa.w, a);
            a = fmaf(w4f, xb.x, a); a = fmaf(w5, xb.y, a);
            a = fmaf(w6, xb.z, a); a = fmaf(w7, xb.w, a);
            acc[t] = a;
        }
    }
#pragma unroll
    for (int t = 0; t < MAX_T; ++t) {
        if (!EXACT_T && t >= T) break;
        float a = acc[t];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane != 0) continue;
        if (is_beta) {
            beta[(size_t) t * h_v + r] = 1.0f / (1.0f + __expf(-a));
        } else {
            const float v = a + dt[r];
            const float sp = v > 20.0f ? v : log1pf(__expf(v));
            gate[(size_t) t * h_v + r] = sp * ssm_a[r];
        }
    }
}

template <int MAX_T = kVerifyNarrowT, bool EXACT_T = false>
__global__ void __launch_bounds__(64) gdn_ab_multi_kernel(const float* __restrict__ x, const uint16_t* __restrict__ wa,
                                                          const uint16_t* __restrict__ wb,
                                                          const float* __restrict__ dt,
                                                          const float* __restrict__ ssm_a, float* __restrict__ gate,
                                                          float* __restrict__ beta, int n, int h_v, int T) {
    gdn_ab_body<MAX_T, EXACT_T>(blockIdx.x, x, wa, wb, dt, ssm_a, gate, beta, n, h_v, T);
}

// 143-amdk (STRATA_FUSE_CONVAB): the conv + L2 norm and the a/b projections in ONE launch.  They read different inputs (the qkv
// projection and the layer input) and write different outputs, so they only share a node: blocks [0, conv) run the conv body,
// the following blocks (token row 0 only) the a/b body, each exactly as its own kernel.
template <int MAX_T, bool EXACT_T>
__global__ void __launch_bounds__(S) gdn_conv_ab_kernel(int conv_blocks, float* hist, const float* __restrict__ qkv,
                                                        const float* __restrict__ w, float* __restrict__ h, int C, int qk_heads,
                                                        float eps, int t_begin, bool commit, const float* __restrict__ x,
                                                        const uint16_t* __restrict__ wa, const uint16_t* __restrict__ wb,
                                                        const float* __restrict__ dt, const float* __restrict__ ssm_a,
                                                        float* __restrict__ gate, float* __restrict__ beta, int n, int h_v, int T) {
    const int ab_blocks = (2 * h_v + 1) / 2;   // the a/b blocks first: their dot chain is the longer one, so it starts with the first wave
    if ((int) blockIdx.x >= ab_blocks) {
        gdn_conv_l2_body((int) blockIdx.x - ab_blocks, (int) blockIdx.y, hist, qkv, w, h, C, qk_heads, eps, t_begin, commit);
    } else if (blockIdx.y == 0 && threadIdx.x < 64) {
        gdn_ab_body<MAX_T, EXACT_T>((int) blockIdx.x, x, wa, wb, dt, ssm_a, gate, beta, n, h_v, T);
    }
}

// State-only commit kernel: each head's 128 independent state columns are split across 4 blocks of 32 columns
// (48 * 4 = 192 blocks of 128 threads across all SMs, vs 48 blocks of 512 threads), with no sq/o/norm/z/y work.
// DBG (STRATA_DBG_GDN=1, #937): a window count the launch never allowed (n_keep is read from device memory; n_max is the
// rows the window holds) is reported and the commit skipped, where the fault would be a memory-aperture violation
__device__ __forceinline__ void gdn_step_commit_body(float* __restrict__ state, const float* __restrict__ hbuf, int C,
                                                     const float* __restrict__ gate, const float* __restrict__ beta,
                                                     int h_k, int h_v, int n);

template <bool DBG>
__global__ void __launch_bounds__(32 * RG) gdn_step_commit_kernel(float* __restrict__ state,
                                                                  const float* __restrict__ hbuf, int C,
                                                                  const float* __restrict__ gate,
                                                                  const float* __restrict__ beta,
                                                                  int h_k, int h_v,
                                                                  const int32_t* __restrict__ n_keep, int n_max) {
    const int n = *n_keep;
    if (n <= 0) return;
    if (DBG && n > n_max) {
        if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
            printf("strata DBG: gdn_step_commit: n_keep %d is past the window's %d rows: commit skipped (#937)\n", n, n_max);
        return;
    }
    gdn_step_commit_body(state, hbuf, C, gate, beta, h_k, h_v, n);
}

// engine-latency: gdn_step_commit_kernel for every GDN layer of the commit in one launch (blockIdx.z = the layer); each
// layer's blocks run gdn_step_commit_body on that layer's state and rows, exactly as its own launch did
__global__ void __launch_bounds__(32 * RG) gdn_step_commit_layers_kernel(float* __restrict__ state, long long state_stride,
                                                                         const float* __restrict__ hbuf, long long h_stride,
                                                                         int C, const float* __restrict__ gate,
                                                                         const float* __restrict__ beta, long long gb_stride,
                                                                         int h_k, int h_v, const int32_t* __restrict__ n_keep) {
    const int n = *n_keep;
    if (n <= 0) return;
    const long long z = blockIdx.z;
    gdn_step_commit_body(state + z * state_stride, hbuf + z * h_stride, C, gate + z * gb_stride, beta + z * gb_stride, h_k,
                         h_v, n);
}

// engine-latency: gdn_conv_commit_kernel for every GDN layer in one launch (blockIdx.y = the layer)
__global__ void gdn_conv_commit_layers_kernel(float* __restrict__ hist0, long long hist_stride, const float* __restrict__ qkv0,
                                              long long qkv_stride, int C, const int32_t* __restrict__ n_keep) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int n = *n_keep;
    if (n <= 0) return;
    float* hist = hist0 + (long long) blockIdx.y * hist_stride;
    const float* qkv = qkv0 + (long long) blockIdx.y * qkv_stride;
    float seq[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = n + j;          // the last three of [hist(3) | x_0..x_{n-1}]
        seq[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
    }
    hist[c * 3] = seq[0];
    hist[c * 3 + 1] = seq[1];
    hist[c * 3 + 2] = seq[2];
}

__device__ __forceinline__ void gdn_step_commit_body(float* __restrict__ state, const float* __restrict__ hbuf, int C,
                                                     const float* __restrict__ gate, const float* __restrict__ beta,
                                                     int h_k, int h_v, int n) {
    __shared__ float sk[S];
    __shared__ float red[RG][32];
    const int head = blockIdx.x;
    const int col_local = threadIdx.x;
    const int col = blockIdx.y * 32 + col_local;
    const int rg = threadIdx.y;
    const int tid = rg * 32 + col_local;    // 0 .. 127
    const int qh = head % h_k;
    const int qk = S * h_k;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < n; ++t) {
        const float* ht = hbuf + (size_t) t * C;
        __syncthreads();
        sk[tid] = ht[qk + qh * S + tid];
        __syncthreads();
        const float g = __expf(gate[(size_t) t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red[rg][col_local] = kv;
        __syncthreads();
        const float kv_col = red[0][col_local] + red[1][col_local] + red[2][col_local] + red[3][col_local];
        const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
#pragma unroll
        for (int r = 0; r < RPG; ++r) s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
}

// S26 (STRATA_QFUSE=1): the q8_1 image of a 32-column block, written by the warp that holds those columns with
// native_quantize_q8_1_kernel's quantizer (the same XOR-tree max and sum, d = amax / 127, roundf(x / d), ds = (d, sum)).
struct GdnQ81 { half2 ds; int8_t qs[32]; };
__device__ __forceinline__ void gdn_q8_1_store(GdnQ81* __restrict__ xq, size_t idx, float xi) {
    // no contraction: xi is a product here, and fma(a, b, shfl) would round the first sum unlike the separate
    // quantizer, which reads xi from memory (S26 harness: ds.sum differed by 1 ulp without this)
#pragma clang fp contract(off)
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    const float d = q8_1_finite(q81_div(amax, 127.0f));   // #606: as native_quantize_q8_1_kernel - finite blocks bit for bit (#3948: and its division)
    const int8_t q = q8_1_quant(xi, d, amax);
    xq[idx / 32].qs[idx % 32] = q;
    if (idx % 32 == 0) xq[idx / 32].ds = q8_1_ds(d, sum);   // #606: clamped (the QFUSE path bypassed the finite helper)
}

template <bool ALL_OUT, bool Q>
__global__ void __launch_bounds__(S * RG) gdn_step_norm_multi_kernel(float* __restrict__ state,
                                                                     const float* __restrict__ hbuf, int C,
                                                                     const float* __restrict__ gate,
                                                                     const float* __restrict__ beta,
                                                                     const float* __restrict__ z,
                                                                     const float* __restrict__ gamma, float eps,
                                                                     float* __restrict__ y, int h_k, int h_v, int T,
                                                                     const int32_t* __restrict__ n_keep, int t_out_begin,
                                                                     GdnQ81* __restrict__ xq, GdnGroups grp = {}) {
    if (grp.n > 0) {   // group blockIdx.y - its slot's state and its rows of the window's buffers
        const int gi = blockIdx.y, f = grp.first[gi];
        state = grp.state[gi];
        hbuf += (size_t) f * C;
        gate += (size_t) f * h_v;
        beta += (size_t) f * h_v;
        z += (size_t) f * S * h_v;
        y += (size_t) f * S * h_v;
        T = grp.ntok[gi];
        n_keep = grp.keep[gi];
    }
    __shared__ float sk[2][S], sq[2][S];
    __shared__ float red_kv[RG][S];
    __shared__ float red_o[RG][S];
    __shared__ float wsum[S / 32];
    const int head = blockIdx.x;
    const int col = threadIdx.x;
    const int rg = threadIdx.y;
    const int qh = head % h_k;
    const int qk = S * h_k;             // q at [0, qk), k at [qk, 2qk), v at [2qk, ...)
    const int value_dim = S * h_v;
    const int n = ALL_OUT ? T : (n_keep ? *n_keep : T);
    if (n > 0) {
        if (rg == 0) sk[0][col] = hbuf[qk + qh * S + col];
        else if (rg == 1 && (ALL_OUT || 0 >= t_out_begin)) sq[0][col] = hbuf[qh * S + col];
    }
    const float gam = (rg == 0) ? gamma[col] : 0.0f;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = ALL_OUT ? __ldg(&base[r * row_stride]) : base[r * row_stride];
    for (int t = 0; t < n; ++t) {
        const int cur = t & 1, nxt = (t + 1) & 1;
        const float* ht = hbuf + (size_t) t * C;
        const bool need_out = ALL_OUT || (t >= t_out_begin);
        __syncthreads();
        if (t + 1 < n) {
            const float* ht_next = ht + C;
            const bool need_next = ALL_OUT || (t + 1 >= t_out_begin);
            if (rg == 0) sk[nxt][col] = ht_next[qk + qh * S + col];
            else if (rg == 1 && need_next) sq[nxt][col] = ht_next[qh * S + col];
        }
        const float g = __expf(gate[(size_t) t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[cur][rg * RPG + r], kv);
        red_kv[rg][col] = kv;
        __syncthreads();
        const float kv_col = red_kv[0][col] + red_kv[1][col] + red_kv[2][col] + red_kv[3][col];
        const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
        if (!need_out) {
#pragma unroll
            for (int r = 0; r < RPG; ++r) s[r] = fmaf(g, s[r], sk[cur][rg * RPG + r] * delta);
            continue;
        }
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[cur][rg * RPG + r] * delta);
            o = fmaf(s[r], sq[cur][rg * RPG + r], o);
        }
        red_o[rg][col] = o;
        __syncthreads();
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (red_o[0][col] + red_o[1][col] + red_o[2][col] + red_o[3][col]) * rsqrtf((float) S);
            sq_part = oc * oc;
        }
        // every thread takes the shuffles, as the single-token kernel does: inside the `rg == 0` branch hipcc on
        // gfx1151 gave another rounding for 3% of the outputs (gdn_parity section 5 on Aurora)
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
        if (rg == 0 && (col & 31) == 0) wsum[col >> 5] = sq_part;
        __syncthreads();
        if (rg == 0) {
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = rsqrtf(ss / (float) S + eps);
            const float zz = z[(size_t) t * value_dim + head * S + col];
            const float yv = oc * scale * gam * (1.0f / (1.0f + __expf(-zz)));
            y[(size_t) t * value_dim + head * S + col] = yv;
            if constexpr (Q) gdn_q8_1_store(xq, (size_t) (t - t_out_begin) * value_dim + head * S + col, yv);
        }
    }
    if (!ALL_OUT && n_keep != nullptr && n > 0) {
#pragma unroll
        for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
    }
}

// S25 (STRATA_GDN_SPLIT=1): the same recurrence with each head's 128 state columns over 4 blocks of 32 columns
// (128 threads: the same (column, row group) threads, each with the same 32 state rows), so 4x the blocks of
// gdn_step_norm_multi_kernel - which ran one 512-thread block per head (32 blocks) and read the 2 MB state of a layer
// at ~60 GB/s. Every column's arithmetic is the original's; the output norm, which needs all 128 columns of a head,
// moves to gdn_out_norm_kernel: this kernel leaves the unnormalized output `oc` in y and the second kernel sums the
// squares by the same warps (columns 32w..32w+31, the same butterfly) in the same order.
// S26: OUT = false (the commit, whose outputs nobody reads) drops the output reduction and its two barriers; the
// norm kernel below has the old kernel's code shape (S x RG threads, rg 0 holding oc), which makes it bitwise equal -
// the 128-thread version differed from the old kernel by 1 ulp in ~7% of the outputs.
constexpr int GS_COLS = 32;
template<bool OUT>
__global__ void __launch_bounds__(GS_COLS * RG) gdn_step_split_kernel(float* __restrict__ state,
                                                                      const float* __restrict__ hbuf, int C,
                                                                      const float* __restrict__ gate,
                                                                      const float* __restrict__ beta,
                                                                      float* __restrict__ y, int h_k, int h_v, int T,
                                                                      const int32_t* __restrict__ n_keep,
                                                                      int t_out_begin) {
    __shared__ float sk[S], sq[S];
    __shared__ float red[RG][GS_COLS];
    const int head = blockIdx.x;
    const int c0 = blockIdx.y * GS_COLS;
    const int lc = threadIdx.x;                 // local column
    const int col = c0 + lc;
    const int rg = threadIdx.y;
    const int tid = rg * GS_COLS + lc;
    const int qh = head % h_k;
    const int qk = S * h_k;
    const int value_dim = S * h_v;
    const int n = n_keep ? *n_keep : T;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < n; ++t) {
        const float* ht = hbuf + (size_t) t * C;
        __syncthreads();
        if (tid < S) { sk[tid] = ht[qk + qh * S + tid]; sq[tid] = ht[qh * S + tid]; }
        __syncthreads();
        const float g = __expf(gate[(size_t) t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red[rg][lc] = kv;
        __syncthreads();
        const float kv_col = red[0][lc] + red[1][lc] + red[2][lc] + red[3][lc];
        const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        if (!OUT) continue;
        __syncthreads();
        red[rg][lc] = o;
        __syncthreads();
        if (rg == 0 && t >= t_out_begin)
            y[(size_t) t * value_dim + head * S + col] =
                (red[0][lc] + red[1][lc] + red[2][lc] + red[3][lc]) * rsqrtf((float) S);
    }
    if (n_keep != nullptr && n > 0) {
#pragma unroll
        for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
    }
}


// S26: the state-only replay (the commit) with its per-token inputs prefetched - gate, beta and the v column of every
// row up front, the next row's k / q while the current one computes. The arithmetic and its order are
// gdn_step_split_kernel<false>'s (gfx1151 harness: state bitwise equal; 44.5 -> 39.1 us at 1 row, 47.4 -> 40.5 at 2,
// 53.8 -> 42.9 at 4, h_v 48). With outputs the prefetch lost (register pressure), so the window keeps the plain split.
template<int MAXT>
__global__ void __launch_bounds__(GS_COLS * RG) gdn_state_replay_kernel(float* __restrict__ state,
                                                                        const float* __restrict__ hbuf, int C,
                                                                        const float* __restrict__ gate,
                                                                        const float* __restrict__ beta, int h_k,
                                                                        int h_v, int T, const int32_t* __restrict__ n_keep) {
    __shared__ float sk[S], sq[S];
    __shared__ float red[RG][GS_COLS];
    const int head = blockIdx.x;
    const int c0 = blockIdx.y * GS_COLS;
    const int lc = threadIdx.x;
    const int col = c0 + lc;
    const int rg = threadIdx.y;
    const int tid = rg * GS_COLS + lc;
    const int qh = head % h_k;
    const int qk = S * h_k;
    const int n = n_keep ? *n_keep : T;
    float gv[MAXT], bv[MAXT], vv[MAXT];
#pragma unroll
    for (int t = 0; t < MAXT; ++t) {
        if (t < n) {
            gv[t] = gate[(size_t) t * h_v + head];
            bv[t] = beta[(size_t) t * h_v + head];
            vv[t] = hbuf[(size_t) t * C + 2 * qk + head * S + col];
        }
    }
    float kn = 0.0f, qn = 0.0f;
    if (n > 0) { kn = hbuf[qk + qh * S + tid]; qn = hbuf[qh * S + tid]; }
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
#pragma unroll
    for (int t = 0; t < MAXT; ++t) {
        if (t >= n) break;
        __syncthreads();
        sk[tid] = kn; sq[tid] = qn;
        __syncthreads();
        if (t + 1 < n) {
            const float* hn = hbuf + (size_t) (t + 1) * C;
            kn = hn[qk + qh * S + tid]; qn = hn[qh * S + tid];
        }
        const float g = __expf(gv[t]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red[rg][lc] = kv;
        __syncthreads();
        const float kv_col = red[0][lc] + red[1][lc] + red[2][lc] + red[3][lc];
        const float delta = (vv[t] - g * kv_col) * bv[t];
        float o = 0.0f;   // unused: kept because the measured code had it (without it: 51 us instead of 39 at 1 row)
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        (void) o;
    }
    if (n_keep != nullptr && n > 0) {
#pragma unroll
        for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
    }
}
// y = oc * rsqrt(mean(oc^2) + eps) * gamma * sigmoid(z) per head and token: gdn_step_norm_multi_kernel's tail with its
// code shape (S x RG threads; rg 0 holds oc, the other row groups contribute 0 to wsum), reading oc from y
template<bool Q>
__global__ void __launch_bounds__(S * RG) gdn_out_norm_kernel(const float* __restrict__ z, const float* __restrict__ gamma,
                                                             float eps, float* __restrict__ y, int h_v, int T,
                                                             const int32_t* __restrict__ n_keep, int t_out_begin,
                                                             GdnQ81* __restrict__ xq) {
    __shared__ float wsum[S * RG / 32];
    const int head = blockIdx.x, t = blockIdx.y + t_out_begin, col = threadIdx.x, rg = threadIdx.y;
    const int tid = rg * S + col;
    const int n = n_keep ? *n_keep : T;
    if (t >= n) return;
    const int value_dim = S * h_v;
    float oc = 0.0f, sq_part = 0.0f;
    if (rg == 0) {
        oc = y[(size_t) t * value_dim + head * S + col];
        sq_part = oc * oc;
    }
    for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
    if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
    __syncthreads();
    if (rg == 0) {
        const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
        const float scale = rsqrtf(ss / (float) S + eps);
        const float zz = z[(size_t) t * value_dim + head * S + col];
        if constexpr (Q) {
            const float yv = oc * scale * gamma[col] * (1.0f / (1.0f + __expf(-zz)));
            y[(size_t) t * value_dim + head * S + col] = yv;
            gdn_q8_1_store(xq, (size_t) (t - t_out_begin) * value_dim + head * S + col, yv);
        } else {
            y[(size_t) t * value_dim + head * S + col] = oc * scale * gamma[col] * (1.0f / (1.0f + __expf(-zz)));
        }
    }
}

__global__ void embedding_gather_dev_kernel(const uint8_t* __restrict__ codes, const float* __restrict__ scales,
                                            const float* __restrict__ offsets, const int32_t* __restrict__ tokens,
                                            int64_t n, int code_bits, int code_bias, int group_elems,
                                            unsigned long long row_codes, unsigned long long row_groups,
                                            float* __restrict__ out) {
    const int t = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned long long token = (unsigned long long) tokens[t];
    const uint8_t* c = codes + token * row_codes;
    const float* sc = scales + token * row_groups;
    const float* of = offsets ? offsets + token * row_groups : nullptr;
    const int per_byte = 8 / code_bits;
    const unsigned mask = (1u << code_bits) - 1u;
    const int code = (c[i / per_byte] >> ((i % per_byte) * code_bits)) & mask;
    const int64_t group = i / group_elems;
    const float product = __fmul_rn((float) (code + code_bias), sc[group]);
    out[(size_t) t * n + i] = __fadd_rn(product, of ? of[group] : 0.0f);
}

__global__ void broadcast_streams_kernel(const float* __restrict__ x, float* __restrict__ R, int64_t n, int hc) {
    const int t = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * hc) return;
    R[(size_t) t * n * hc + i] = x[(size_t) t * n + i % n];
}

__global__ void hc_mean_streams_kernel(const float* __restrict__ R, float* __restrict__ out, int64_t n, int hc) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = 0.0f;
    for (int c = 0; c < hc; ++c) s += R[(size_t) c * n + i];
    out[i] = s / (float) hc;
}

__global__ void route_marginal_stats_kernel(const int32_t* __restrict__ ids, int n, int k, unsigned long long* acc) {
    const int r = threadIdx.x;
    if (r >= n) return;
    int fresh = 0;
    for (int j = 0; j < k; ++j) {
        const int32_t e = ids[r * k + j];
        bool seen = false;
        for (int j2 = 0; j2 < j && !seen; ++j2) seen = ids[r * k + j2] == e;
        for (int r2 = 0; r2 < r && !seen; ++r2)
            for (int j2 = 0; j2 < k && !seen; ++j2) seen = ids[r2 * k + j2] == e;
        if (!seen) ++fresh;
    }
    atomicAdd(acc + r, (unsigned long long) fresh);
}

__global__ void copy_indexed_kernel(float* __restrict__ dst, const float* __restrict__ src, int64_t stride,
                                    const int32_t* __restrict__ index, int64_t n) {
    const int idx = *index;
    if (idx < 0) return;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        dst[i] = src[(size_t) idx * stride + i];
}

__global__ void fetch_blobs_kernel(const unsigned long long* __restrict__ src, const int32_t* __restrict__ n,
                                   uint4* __restrict__ dst, long long per) {
    const long long total = (long long) *n * per;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long long) gridDim.x * blockDim.x) {
        const long long k = i / per, off = i - k * per;
        dst[i] = ((const uint4*) src[k])[off];
    }
}

__global__ void res_patch_kernel(int32_t* __restrict__ table, const volatile int32_t* pairs, int n) {
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += gridDim.x * blockDim.x) table[pairs[2 * k]] = pairs[2 * k + 1];
}

__global__ void rebase_ptrs_kernel(unsigned long long* ptr, const int32_t* n, unsigned long long base, long long bytes) {
    const int k = threadIdx.x;
    if (k < *n) ptr[k] = base + (unsigned long long) k * (unsigned long long) bytes;
}

__global__ void add_streams_broadcast_kernel(const float* __restrict__ h, const float* __restrict__ e,
                                             float* __restrict__ R, int64_t n, int hc) {
    const int t = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * hc) return;
    R[(size_t) t * n * hc + i] = h[(size_t) t * n * hc + i] + e[(size_t) t * n + i % n];
}

__global__ void ident_hits_kernel(const int32_t* __restrict__ ids, int n, int32_t* __restrict__ slot,
                                  int32_t* __restrict__ dst, int32_t* __restrict__ count) {
    const int i = threadIdx.x;
    if (i < n) { slot[i] = ids[i]; dst[i] = i; }
    if (i == 0) *count = n;
}

// E = the widest element the row size divides into (16, 4 or 1 bytes): a Q6_K head row of 2560 values is 2100 bytes
template<typename E>
__global__ void gather_rows_kernel(const E* __restrict__ src, long long row_e, const int32_t* __restrict__ ids,
                                   long long n, E* __restrict__ dst) {
    const long long total = n * row_e;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long long) gridDim.x * blockDim.x) {
        const long long r = i / row_e, o = i - r * row_e;
        dst[i] = src[(long long) ids[r] * row_e + o];
    }
}

__global__ void map_ids_kernel(int32_t* ids, const int32_t* __restrict__ table, int n) {
    const int i = threadIdx.x;
    if (i < n) ids[i] = table[ids[i]];
}

__global__ void row_top_prob_kernel(const float* __restrict__ logits, int n_vocab, const int32_t* __restrict__ ids,
                                    float* __restrict__ probs) {
    __shared__ float part[32];
    const int t = blockIdx.x;
    const float* l = logits + (size_t) t * n_vocab;
    const float m = l[ids[t]];
    float s = 0.0f;
    for (int i = threadIdx.x; i < n_vocab; i += blockDim.x) s += __expf(l[i] - m);
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        for (int w = 0; w < (int) (blockDim.x >> 5); ++w) tot += part[w];
        probs[t] = 1.0f / tot;
    }
}

__global__ void mtp_select_kernel(const float* __restrict__ R_src, int64_t stride, const int32_t* __restrict__ ids,
                                  const int32_t* __restrict__ row_dev, float* __restrict__ R_dst,
                                  int32_t* __restrict__ tok_dst, int32_t* out, int j, const float* probs, float* out_p) {
    const int row = *row_dev;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < stride; i += (int64_t) gridDim.x * blockDim.x)
        R_dst[i] = R_src[(size_t) row * stride + i];
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const int32_t tok = ids[row];
        *tok_dst = tok;
        if (probs != nullptr && out_p != nullptr) ((volatile float*) out_p)[j] = probs[row];
        __threadfence_system();
        if (out != nullptr) ((volatile int32_t*) out)[j] = tok;
    }
}

__global__ void dense_steps_kernel(const int32_t* __restrict__ cells, int n, int32_t* __restrict__ steps) {
    const int i = threadIdx.x;
    if (i >= n) return;
    const int c = cells[i];
    steps[i * 4 + 0] = c;
    steps[i * 4 + 1] = c + 1;
    steps[i * 4 + 2] = (c + 1) / 4;
    steps[i * 4 + 3] = c + 1;
}

// --pipeline-windows: the drafter's chain, teacher forced (see force_token)
__global__ void force_token_kernel(int32_t* tok, const int32_t* force, int j) {
    const int32_t f = ((const volatile int32_t*) force)[j];
    if (f >= 0) *tok = f;
}

}  // namespace

void fetch_blobs(const unsigned long long* src, const int32_t* n, uint8_t* dst, int64_t blob_bytes, int cap, void* stream) {
    if (cap <= 0) return;
    if (blob_bytes % 16 != 0) { std::fprintf(stderr, "fetch_blobs: blob size must be a multiple of 16\n"); std::exit(1); }
    fetch_blobs_kernel<<<48 * 8, 256, 0, (cudaStream_t) stream>>>(src, n, (uint4*) dst, (long long) (blob_bytes / 16));
    check("fetch_blobs");
}

void res_patch(int32_t* table, const int32_t* pairs, int n, void* stream) {
    if (n <= 0) return;
    res_patch_kernel<<<(n + 255) / 256 < 64 ? (n + 255) / 256 : 64, 256, 0, (cudaStream_t) stream>>>(table, pairs, n);
    check("res_patch");
}

void rebase_ptrs(unsigned long long* ptr, const int32_t* n, uint8_t* base, int64_t blob_bytes, void* stream) {
    rebase_ptrs_kernel<<<1, 128, 0, (cudaStream_t) stream>>>(ptr, n, (unsigned long long) base, (long long) blob_bytes);
    check("rebase_ptrs");
}

void add_streams_broadcast(const float* h, const float* e, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    add_streams_broadcast_kernel<<<dim3((unsigned) ((n_embd * hc + 255) / 256), (unsigned) n_tok), 256, 0,
                                   (cudaStream_t) stream>>>(h, e, R, n_embd, hc);
    check("add_streams_broadcast");
}

void ident_hits(const int32_t* ids, int n, int32_t* slot, int32_t* dst, int32_t* count, void* stream) {
    if (n < 1 || n > 1024) { std::fprintf(stderr, "ident_hits: n out of range\n"); std::exit(1); }
    ident_hits_kernel<<<1, 1024, 0, (cudaStream_t) stream>>>(ids, n, slot, dst, count);
    check("ident_hits");
}

namespace {
__global__ void copy_row_to_first_kernel(const int32_t* __restrict__ row_dev, float* a, int64_t a_n, float* b,
                                         int64_t b_n, float* c, int64_t c_n) {
    const int64_t row = *row_dev;
    if (row == 0) return;
    const int64_t i0 = (int64_t) blockIdx.x * blockDim.x + threadIdx.x, st = (int64_t) gridDim.x * blockDim.x;
    for (int64_t i = i0; i < a_n; i += st) a[i] = a[row * a_n + i];
    for (int64_t i = i0; i < b_n; i += st) b[i] = b[row * b_n + i];
    for (int64_t i = i0; i < c_n; i += st) c[i] = c[row * c_n + i];
}
}  // namespace

void copy_row_to_first(const int32_t* row_dev, float* a, int64_t a_n, float* b, int64_t b_n, float* c, int64_t c_n,
                       void* stream) {
    copy_row_to_first_kernel<<<16, 256, 0, (cudaStream_t) stream>>>(row_dev, a, a_n, b, b_n, c, c_n);
    check("copy_row_to_first");
}

void mtp_select(const float* R_src, int64_t R_stride, const int32_t* ids, const int32_t* row_dev, float* R_dst,
                int32_t* tok_dst, int32_t* out, int j, void* stream, const float* probs, float* out_p) {
    mtp_select_kernel<<<1, 256, 0, (cudaStream_t) stream>>>(R_src, R_stride, ids, row_dev, R_dst, tok_dst, out, j,
                                                            probs, out_p);
    check("mtp_select");
}

void force_token(int32_t* tok, const int32_t* force, int j, void* stream) {
    force_token_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(tok, force, j);
    check("force_token");
}

void gather_rows(const uint8_t* src, int64_t row_bytes, const int32_t* ids, int64_t n, uint8_t* dst, void* stream) {
    cudaStream_t s = (cudaStream_t) stream;
    if (row_bytes % 16 == 0)
        gather_rows_kernel<<<48 * 8, 256, 0, s>>>((const uint4*) src, row_bytes / 16, ids, n, (uint4*) dst);
    else if (row_bytes % 4 == 0)
        gather_rows_kernel<<<48 * 8, 256, 0, s>>>((const uint32_t*) src, row_bytes / 4, ids, n, (uint32_t*) dst);
    else
        gather_rows_kernel<<<48 * 8, 256, 0, s>>>(src, row_bytes, ids, n, dst);
    check("gather_rows");
}

void map_ids(int32_t* ids, const int32_t* table, int n, void* stream) {
    map_ids_kernel<<<1, 64, 0, (cudaStream_t) stream>>>(ids, table, n);
    check("map_ids");
}

void row_top_prob(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* stream) {
    row_top_prob_kernel<<<n_rows, 1024, 0, (cudaStream_t) stream>>>(logits, n_vocab, ids, probs);
    check("row_top_prob");
}

#if !defined(__HIPCC__)   // warp-32 kernels: an AMD card keeps the one-block row_top_prob and sampler_greedy_kernel
namespace {

constexpr int kArgMaxBlocks = 128, kArgThreads = 256, kTopBlocks = 8;

// the larger value, on equality the lower index: sampler_greedy_kernel's order
__device__ __forceinline__ void arg_take(float ov, int oi, float& bv, int& bi) {
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
}

// a block's (value, index) pick, in thread 0; `none` is the index of no candidate
__device__ __forceinline__ void arg_block(float& bv, int& bi, int none) {
    __shared__ float sv[32];
    __shared__ int si[32];
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, bv, off);
        const int oi = __shfl_down_sync(0xffffffffu, bi, off);
        arg_take(ov, oi, bv, bi);
    }
    const int warp = (int) (threadIdx.x >> 5), lane = (int) (threadIdx.x & 31);
    if (lane == 0) { sv[warp] = bv; si[warp] = bi; }
    __syncthreads();
    if (warp == 0) {
        const int nw = (int) (blockDim.x >> 5);
        bv = lane < nw ? sv[lane] : __int_as_float(0xff800000);
        bi = lane < nw ? si[lane] : none;
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, bv, off);
            const int oi = __shfl_down_sync(0xffffffffu, bi, off);
            arg_take(ov, oi, bv, bi);
        }
    }
}

// A row's scratch, whatever the launch's row count (one scratch serves windows of every size): its counter, then
// the blocks' values and indices.
struct ArgRow {
    unsigned counter;
    unsigned pad[3];
    float v[kArgMaxBlocks];
    int i[kArgMaxBlocks];
};

// grid (blocks, rows): block b scans [b * per_block, +per_block) of its row
__global__ void argmax_rows_kernel(const float* __restrict__ logits, int n, int per_block, ArgRow* __restrict__ rows,
                                   int32_t* out) {
    const int row = (int) blockIdx.y, b = (int) blockIdx.x, nb = (int) gridDim.x;
    unsigned* counter = &rows[row].counter;
    float* pv = rows[row].v;
    int* pi = rows[row].i;
    const float* l = logits + (size_t) row * n;
    const int lo = b * per_block, hi = min(n, lo + per_block);
    float bv = __int_as_float(0xff800000);   // -inf
    int bi = n;
    for (int v = lo + (int) threadIdx.x; v < hi; v += (int) blockDim.x) {
        const float s = l[v];
        if (s > bv) { bv = s; bi = v; }
    }
    arg_block(bv, bi, n);
    __shared__ bool last;
    if (threadIdx.x == 0) {
        pv[b] = bv;
        pi[b] = bi;
        __threadfence();
        last = atomicAdd(counter, 1u) == (unsigned) (nb - 1);
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    bv = __int_as_float(0xff800000);
    bi = n;
    if ((int) threadIdx.x < nb) {
        bv = ((volatile const float*) pv)[threadIdx.x];
        bi = ((volatile const int*) pi)[threadIdx.x];
    }
    arg_block(bv, bi, n);
    if (threadIdx.x == 0) {
        out[row] = bi < n ? bi : 0;   // no value above -inf: 0, as the one-block kernel
        *counter = 0;
    }
}

// a row's top-probability scratch: its counter, then its 32 warps' sums
struct TopRow {
    unsigned counter;
    unsigned pad[3];
    float parts[32];
};

// grid (kTopBlocks, rows), 1024 / kTopBlocks threads: thread j of block b is row_top_prob's thread b * blockDim + j
__global__ void row_top_prob_split_kernel(const float* __restrict__ logits, int n_vocab, const int32_t* __restrict__ ids,
                                          float* __restrict__ probs, TopRow* __restrict__ rows) {
    const int row = (int) blockIdx.y;
    float* parts = rows[row].parts;
    const float* l = logits + (size_t) row * n_vocab;
    const float m = l[ids[row]];
    const int vt = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    float s = 0.0f;
    for (int i = vt; i < n_vocab; i += 1024) s += __expf(l[i] - m);
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) parts[vt >> 5] = s;
    __threadfence();
    __syncthreads();
    __shared__ bool last;
    if (threadIdx.x == 0) last = atomicAdd(&rows[row].counter, 1u) == (unsigned) (gridDim.x - 1);
    __syncthreads();
    if (!last || threadIdx.x != 0) return;
    __threadfence();
    float tot = 0.0f;
    for (int w = 0; w < 32; ++w) tot += ((volatile const float*) parts)[w];
    probs[row] = 1.0f / tot;
    rows[row].counter = 0;
}

}  // namespace

uint64_t argmax_rows_scratch_bytes(int n_rows) { return (uint64_t) n_rows * sizeof(ArgRow); }

void argmax_rows(const float* logits, int n_rows, int n, void* scratch, int32_t* out, void* stream) {
    if (n_rows <= 0) return;
    const int nb = std::min(kArgMaxBlocks, std::max(1, (n + 4095) / 4096));
    const int per_block = (n + nb - 1) / nb;
    argmax_rows_kernel<<<dim3((unsigned) nb, (unsigned) n_rows), kArgThreads, 0, (cudaStream_t) stream>>>(
        logits, n, per_block, (ArgRow*) scratch, out);
    check("argmax_rows");
}

uint64_t row_top_prob_scratch_bytes(int n_rows) { return (uint64_t) n_rows * sizeof(TopRow); }

void row_top_prob_split(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* scratch,
                        void* stream) {
    if (n_rows <= 0) return;
    row_top_prob_split_kernel<<<dim3(kTopBlocks, (unsigned) n_rows), 1024 / kTopBlocks, 0, (cudaStream_t) stream>>>(
        logits, n_vocab, ids, probs, (TopRow*) scratch);
    check("row_top_prob_split");
}
#else
uint64_t argmax_rows_scratch_bytes(int) { return 0; }
void argmax_rows(const float*, int, int, void*, int32_t*, void*) {
    std::fprintf(stderr, "argmax_rows: not built for this backend\n");
    std::exit(1);
}
uint64_t row_top_prob_scratch_bytes(int) { return 0; }
void row_top_prob_split(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void*,
                        void* stream) {
    row_top_prob(logits, n_rows, n_vocab, ids, probs, stream);
}
#endif

namespace {
// grid (kNonfiniteBlocks, rows): block b of row t scans [b * per_block, +per_block) and writes its own word.  The
// test is on the bits (exponent all ones), so --use_fast_math cannot fold it away.
__global__ void logits_nonfinite_kernel(const float* __restrict__ logits, int n, int per_block, uint32_t* flags) {
    const int row = (int) blockIdx.y, b = (int) blockIdx.x;
    const float* l = logits + (size_t) row * n;
    const int lo = b * per_block, hi = min(n, lo + per_block);
    int bad = 0;
    for (int v = lo + (int) threadIdx.x; v < hi; v += (int) blockDim.x)
        bad |= (__float_as_uint(l[v]) & 0x7f800000u) == 0x7f800000u;
    bad = __syncthreads_or(bad);
    if (threadIdx.x == 0) flags[row * kNonfiniteBlocks + b] = bad ? 1u : 0u;
}
}  // namespace

void logits_nonfinite_rows(const float* logits, int n_rows, int n, uint32_t* flags, void* stream) {
    if (n_rows <= 0 || n <= 0) return;
    const int per_block = (n + kNonfiniteBlocks - 1) / kNonfiniteBlocks;
    logits_nonfinite_kernel<<<dim3((unsigned) kNonfiniteBlocks, (unsigned) n_rows), 256, 0, (cudaStream_t) stream>>>(
        logits, n, per_block, flags);
    check("logits_nonfinite_rows");
}

bool multi_block_head_ops() {
#if defined(__HIPCC__)
    return false;
#else
    static const bool on = [] {
        const char* v = std::getenv("STRATA_MULTI_BLOCK_ARGMAX");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
#endif
}

bool argmax_rows_wanted() {
#if defined(__HIPCC__)
    return false;
#else
    // sm_80 to sm_89, and sm_7x with STRATA_ARGMAX_ROWS_SM70=1: sample_tokens' greedy pick there is the one-block kernel (sm_90+ has the cluster kernel)
    static const bool want = [] {
        int dev = 0, major = 0;
        if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess)
            return false;
        if (major == 7) {   // STRATA_ARGMAX_ROWS_SM70=1: opt-in on Volta (measured on sm_70 only)
            const char* e = std::getenv("STRATA_ARGMAX_ROWS_SM70");
            return e != nullptr && std::atoi(e) != 0;
        }
        return major == 8;
    }();
    return want && multi_block_head_ops();
#endif
}

namespace {
__global__ void window_ids_kernel(int32_t* steps, int window, int32_t* ids, long long stride) {
    const int q = blockIdx.y;
    int32_t* st = steps + q * 4;
    const int n_kv = st[1];
    const int start = n_kv > window ? n_kv - window : 0;
    const int width = n_kv - start;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < width; j += gridDim.x * blockDim.x)
        ids[q * stride + j] = start + j;
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) st[3] = width;
}
}  // namespace

void window_ids(int32_t* steps, int n, int window, int32_t* ids, int64_t ids_stride, void* stream) {
    window_ids_kernel<<<dim3(8, (unsigned) n), 256, 0, (cudaStream_t) stream>>>(steps, window, ids, (long long) ids_stride);
    check("window_ids");
}

namespace {
// window_ids with a floor read from (mapped) memory when the kernel RUNS: the captured graph keeps the pointer, the
// value changes between sessions.  A floor of 0 gives window_ids' selection exactly (strata::kernels::window_start).
__global__ void window_ids_floor_kernel(int32_t* steps, int window, int32_t* ids, long long stride,
                                        const volatile int32_t* floor_cell) {
    const int q = blockIdx.y;
    int32_t* st = steps + q * 4;
    const int n_kv = st[1];
    const int start = strata::kernels::window_start(n_kv, window, *floor_cell);
    const int width = n_kv - start;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < width; j += gridDim.x * blockDim.x)
        ids[q * stride + j] = start + j;
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) st[3] = width;
}
}  // namespace

void window_ids_floor(int32_t* steps, int n, int window, int32_t* ids, int64_t ids_stride, const int32_t* floor_cell,
                      void* stream) {
    window_ids_floor_kernel<<<dim3(8, (unsigned) n), 256, 0, (cudaStream_t) stream>>>(steps, window, ids, (long long) ids_stride,
                                                                                    floor_cell);
    check("window_ids_floor");
}

void dense_steps(const int32_t* cells, int n, int32_t* steps, void* stream) {
    dense_steps_kernel<<<1, 64, 0, (cudaStream_t) stream>>>(cells, n, steps);
    check("dense_steps");
}

void gdn_conv_l2_multi(const float* history, const float* qkv, const float* conv_w, float* h, int channels,
                       int qk_heads, float eps, int n_tok, void* stream, int t_begin, bool commit) {
    if (!history || !qkv || !conv_w || !h || channels % S != 0 || n_tok < 1 || n_tok > kVerifyMaxT ||
        (commit && (n_tok != 1 || t_begin != 0))) {
        std::fprintf(stderr, "gdn_conv_l2_multi: invalid arguments\n");
        std::exit(1);
    }
    gdn_conv_l2_multi_kernel<<<dim3((unsigned) (channels / S), (unsigned) n_tok), S, 0, (cudaStream_t) stream>>>(
        const_cast<float*>(history), qkv, conv_w, h, channels, qk_heads, eps, t_begin, commit);
    check("gdn_conv_l2_multi");
}

void gdn_conv_ab_multi(const float* history, const float* qkv, const float* conv_w, float* h, int channels, int qk_heads,
                       float eps, int n_tok, int t_begin, bool commit, const float* x, const uint16_t* w_alpha,
                       const uint16_t* w_beta, const float* dt, const float* ssm_a, float* gate, float* beta, int n_embd,
                       int h_v, void* stream) {
    if (!history || !qkv || !conv_w || !h || channels % S != 0 || n_tok < 1 || n_tok > kVerifyMaxT || n_embd % 8 != 0 ||
        (commit && (n_tok != 1 || t_begin != 0))) {
        std::fprintf(stderr, "gdn_conv_ab_multi: invalid arguments\n");
        std::exit(1);
    }
    const int conv_blocks = channels / S, ab_blocks = (2 * h_v + 1) / 2;
    const dim3 grid((unsigned) (conv_blocks + ab_blocks), (unsigned) n_tok);
    cudaStream_t st = (cudaStream_t) stream;
    float* hist = const_cast<float*>(history);
    switch (n_tok) {
        case 1: gdn_conv_ab_kernel<1, true><<<grid, S, 0, st>>>(conv_blocks, hist, qkv, conv_w, h, channels, qk_heads, eps, t_begin, commit, x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok); break;
        case 2: gdn_conv_ab_kernel<2, true><<<grid, S, 0, st>>>(conv_blocks, hist, qkv, conv_w, h, channels, qk_heads, eps, t_begin, commit, x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok); break;
        case 3: gdn_conv_ab_kernel<3, true><<<grid, S, 0, st>>>(conv_blocks, hist, qkv, conv_w, h, channels, qk_heads, eps, t_begin, commit, x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok); break;
        case 4: gdn_conv_ab_kernel<4, true><<<grid, S, 0, st>>>(conv_blocks, hist, qkv, conv_w, h, channels, qk_heads, eps, t_begin, commit, x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok); break;
        case 5: gdn_conv_ab_kernel<5, true><<<grid, S, 0, st>>>(conv_blocks, hist, qkv, conv_w, h, channels, qk_heads, eps, t_begin, commit, x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok); break;
        case 6: gdn_conv_ab_kernel<6, true><<<grid, S, 0, st>>>(conv_blocks, hist, qkv, conv_w, h, channels, qk_heads, eps, t_begin, commit, x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok); break;
        case 7: case 8: gdn_conv_ab_kernel<kVerifyNarrowT, false><<<grid, S, 0, st>>>(conv_blocks, hist, qkv, conv_w, h, channels, qk_heads, eps, t_begin, commit, x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok); break;
        default: gdn_conv_ab_kernel<kVerifyMaxT, false><<<grid, S, 0, st>>>(conv_blocks, hist, qkv, conv_w, h, channels, qk_heads, eps, t_begin, commit, x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok); break;
    }
    check("gdn_conv_ab_multi");
}

void gdn_conv_commit(float* history, const float* qkv, int channels, const int32_t* n_keep, void* stream) {
    gdn_conv_commit_kernel<<<(unsigned) ((channels + 255) / 256), 256, 0, (cudaStream_t) stream>>>(history, qkv,
                                                                                                 channels, n_keep);
    check("gdn_conv_commit");
}

bool gdn_commit_layers(float* state, int64_t state_stride, int64_t conv_off, const float* qkv, int64_t qkv_stride,
                       const float* h, int64_t h_stride, const float* gate, const float* beta, int64_t gb_stride,
                       int conv_channels, int h_k, int h_v, int n_max, int n_layers, const int32_t* n_keep, void* stream) {
    // the per-layer commit's kernel choice (gdn_step_norm_multi below): only its state-only kernel is batched here
    static const bool per_layer = [] {
        const char* sp = std::getenv("STRATA_GDN_SPLIT");
        const char* cs = std::getenv("STRATA_GDN_COMMIT_SPLIT");
        const char* dbg = std::getenv("STRATA_DBG_GDN");
        return (sp && sp[0] == '1') || (cs && cs[0] == '0') || (dbg && dbg[0] == '1');
    }();
    if (per_layer || n_layers <= 0 || n_layers > 65535 || n_max < 1 || n_max > kVerifyMaxT || n_keep == nullptr ||
        h_v <= 0 || h_k <= 0 || h_v % h_k != 0)
        return false;
    cudaStream_t st = (cudaStream_t) stream;
    gdn_conv_commit_layers_kernel<<<dim3((unsigned) ((conv_channels + 255) / 256), (unsigned) n_layers), 256, 0, st>>>(
        state + conv_off, (long long) state_stride, qkv, (long long) qkv_stride, conv_channels, n_keep);
    gdn_step_commit_layers_kernel<<<dim3((unsigned) h_v, 4u, (unsigned) n_layers), dim3(32, RG), 0, st>>>(
        state, (long long) state_stride, h, (long long) h_stride, conv_channels, gate, beta, (long long) gb_stride, h_k, h_v,
        n_keep);
    check("gdn_commit_layers");
    return true;
}

void gdn_ab_multi(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, int n_tok, void* stream) {
    if (n_embd % 8 != 0 || n_tok < 1 || n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_ab_multi: invalid arguments\n");
        std::exit(1);
    }
    const unsigned blocks = (unsigned) ((2 * h_v + 1) / 2);
    cudaStream_t st = (cudaStream_t) stream;
    switch (n_tok) {
        case 1: gdn_ab_multi_kernel<1, true><<<blocks, 64, 0, st>>>(x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok); break;
        case 2: gdn_ab_multi_kernel<2, true><<<blocks, 64, 0, st>>>(x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok); break;
        case 3: gdn_ab_multi_kernel<3, true><<<blocks, 64, 0, st>>>(x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok); break;
        case 4: gdn_ab_multi_kernel<4, true><<<blocks, 64, 0, st>>>(x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok); break;
        case 5: gdn_ab_multi_kernel<5, true><<<blocks, 64, 0, st>>>(x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok); break;
        case 6: gdn_ab_multi_kernel<6, true><<<blocks, 64, 0, st>>>(x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok); break;
        case 7: case 8: gdn_ab_multi_kernel<kVerifyNarrowT, false><<<blocks, 64, 0, st>>>(x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok); break;
        default: gdn_ab_multi_kernel<kVerifyMaxT, false><<<blocks, 64, 0, st>>>(x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok); break;
    }
    check("gdn_ab_multi");
}

void gdn_step_norm_groups(const GdnGroups& groups, const float* h, int conv_channels, const float* gate, const float* beta,
                          const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, bool commit,
                          void* stream) {
    if (groups.n < 1 || groups.n > 8 || !h || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v % h_k) {
        std::fprintf(stderr, "gdn_step_norm_groups: invalid arguments\n");
        std::exit(1);
    }
    for (int i = 0; i < groups.n; ++i)
        if (!groups.state[i] || groups.ntok[i] < 1 || groups.ntok[i] > kVerifyMaxT || (commit && !groups.keep[i])) {
            std::fprintf(stderr, "gdn_step_norm_groups: invalid group %d\n", i);
            std::exit(1);
        }
    const dim3 g1((unsigned) h_v, (unsigned) groups.n), b1(S, RG);
    // the kernel gdn_step_norm_multi picks for these calls: a window's rows (all outputs, no keep) or a commit of
    // one-row groups (keep, t_out_begin 0)
    if (commit)
        gdn_step_norm_multi_kernel<false, false><<<g1, b1, 0, (cudaStream_t) stream>>>(
            nullptr, h, conv_channels, gate, beta, z, gamma, eps, y, h_k, h_v, 0, nullptr, 0, nullptr, groups);
    else
        gdn_step_norm_multi_kernel<true, false><<<g1, b1, 0, (cudaStream_t) stream>>>(
            nullptr, h, conv_channels, gate, beta, z, gamma, eps, y, h_k, h_v, 0, nullptr, 0, nullptr, groups);
    check("gdn_step_norm_groups");
}

void gdn_step_norm_multi(float* state, const float* h, int conv_channels, const float* gate, const float* beta,
                         const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int n_tok,
                         const int32_t* n_keep, void* stream, int t_out_begin, void* xq_out) {
    GdnQ81* const xq = static_cast<GdnQ81*>(xq_out);   // S26: non-null = also the outputs' q8_1 (STRATA_QFUSE)
    if (!state || !h || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v % h_k || n_tok < 1 ||
        n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_step_norm_multi: invalid arguments\n");
        std::exit(1);
    }
    static const bool split = [] { const char* v = std::getenv("STRATA_GDN_SPLIT"); return v && v[0] == '1'; }();
    // S26: the split runs where it is faster (gfx1151 harness, h_v 48: the commit 8-19% per call; a window of 2+ rows
    // 3-13%; one row: no gain, the old kernel stays) - bitwise equal in every case
    if (split && (t_out_begin >= n_tok || n_tok - t_out_begin >= 2)) {
        const dim3 grid((unsigned) h_v, S / GS_COLS), block(GS_COLS, RG);
        if (t_out_begin >= n_tok) {   // no outputs (the commit): the state only
            if (n_tok > kVerifyNarrowT)
                gdn_state_replay_kernel<kVerifyMaxT><<<grid, block, 0, (cudaStream_t) stream>>>(
                    state, h, conv_channels, gate, beta, h_k, h_v, n_tok, n_keep);
            else
                gdn_state_replay_kernel<kVerifyNarrowT><<<grid, block, 0, (cudaStream_t) stream>>>(
                    state, h, conv_channels, gate, beta, h_k, h_v, n_tok, n_keep);
        } else {
            gdn_step_split_kernel<true><<<grid, block, 0, (cudaStream_t) stream>>>(
                state, h, conv_channels, gate, beta, y, h_k, h_v, n_tok, n_keep, t_out_begin);
            const dim3 ng((unsigned) h_v, (unsigned) (n_tok - t_out_begin));
            if (xq) gdn_out_norm_kernel<true><<<ng, dim3(S, RG), 0, (cudaStream_t) stream>>>(z, gamma, eps, y, h_v, n_tok, n_keep, t_out_begin, xq);
            else gdn_out_norm_kernel<false><<<ng, dim3(S, RG), 0, (cudaStream_t) stream>>>(z, gamma, eps, y, h_v, n_tok, n_keep, t_out_begin, nullptr);
        }
        check("gdn_step_norm_multi (split)");
        return;
    }
    static const bool commit_split = [] {
        const char* e = std::getenv("STRATA_GDN_COMMIT_SPLIT");
        return !e || e[0] != '0';
    }();
    if (commit_split && n_keep != nullptr && t_out_begin >= n_tok) {
        static const bool dbg = [] { const char* e = std::getenv("STRATA_DBG_GDN"); return e && e[0] == '1'; }();
        if (dbg) gdn_step_commit_kernel<true><<<dim3((unsigned) h_v, 4u), dim3(32, RG), 0, (cudaStream_t) stream>>>(
            state, h, conv_channels, gate, beta, h_k, h_v, n_keep, n_tok);
        else gdn_step_commit_kernel<false><<<dim3((unsigned) h_v, 4u), dim3(32, RG), 0, (cudaStream_t) stream>>>(
            state, h, conv_channels, gate, beta, h_k, h_v, n_keep, n_tok);
        check("gdn_step_commit");
        return;
    }
    const dim3 g1((unsigned) h_v), b1(S, RG);
    const bool all_out = n_keep == nullptr && t_out_begin <= 0;
    if (xq && t_out_begin < n_tok) {
        if (all_out) gdn_step_norm_multi_kernel<true, true><<<g1, b1, 0, (cudaStream_t) stream>>>(
            state, h, conv_channels, gate, beta, z, gamma, eps, y, h_k, h_v, n_tok, nullptr, 0, xq);
        else gdn_step_norm_multi_kernel<false, true><<<g1, b1, 0, (cudaStream_t) stream>>>(
            state, h, conv_channels, gate, beta, z, gamma, eps, y, h_k, h_v, n_tok, n_keep, t_out_begin, xq);
    } else if (all_out) {
        gdn_step_norm_multi_kernel<true, false><<<g1, b1, 0, (cudaStream_t) stream>>>(
            state, h, conv_channels, gate, beta, z, gamma, eps, y, h_k, h_v, n_tok, nullptr, 0, nullptr);
    } else {
        gdn_step_norm_multi_kernel<false, false><<<g1, b1, 0, (cudaStream_t) stream>>>(
            state, h, conv_channels, gate, beta, z, gamma, eps, y, h_k, h_v, n_tok, n_keep, t_out_begin, nullptr);
    }
    check("gdn_step_norm_multi");
}

namespace {
__global__ void wait_flag_ge_kernel(const volatile uint32_t* flag, uint32_t value) {
    while (*flag < value) strata_spin_pause();
    __threadfence_system();
}

// the GPU's clock in ns (the same sources as gpu_stamp_kernel)
__device__ __forceinline__ unsigned long long strata_now_ns() {
    unsigned long long t;
#if defined(STRATA_HIP_GFX906)
    t = wall_clock64() * 40ull;
#elif defined(__HIPCC__)
    t = wall_clock64() * 10ull;
#else
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
#endif
    return t;
}

// the overlapped split's hand-off wait, bounded: spin until *flag >= value, or give up after timeout_ns (0: never),
// marking *err = 1 (mapped) so the host fails the window instead of using a stale hand-off.  The window's later
// kernels still run (on whatever the hand-off buffer holds) so the graph completes and no stream is left hanging.
__global__ void wait_handoff_kernel(const volatile uint32_t* flag, uint32_t value, uint32_t* err,
                                    unsigned long long timeout_ns) {
    const unsigned long long t0 = timeout_ns ? strata_now_ns() : 0ull;
    while (*flag < value) {
        strata_spin_pause();
        if (timeout_ns && strata_now_ns() - t0 > timeout_ns) {
            *(volatile uint32_t*) err = 1u;
            __threadfence_system();
            return;
        }
    }
    __threadfence_system();
}

__global__ void handoff_publish_kernel(float* dst, const float* __restrict__ a, int64_t na, const float* __restrict__ b,
                                       int64_t nb, const float* __restrict__ c, int64_t nc, uint32_t* counter,
                                       uint32_t* flag, const volatile uint32_t* drop) {
    const int64_t n = na + nb + nc;
    volatile float* const out = dst;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        out[i] = i < na ? a[i] : i < na + nb ? b[i - na] : c[i - na - nb];
    __threadfence_system();   // this thread's stores reach host memory before the block counts itself done
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t prev = atomicAdd(counter, 1u);
        if (prev == gridDim.x - 1) {   // every block's payload is out: raise the next stage's flag
            *(volatile uint32_t*) counter = 0u;
            __threadfence_system();
            if (drop == nullptr || *drop == 0u) {   // test hook (STRATA_TEST_HANDOFF_DROP): a lost publish
                *(volatile uint32_t*) flag = 1u;
                __threadfence_system();
            }
        }
    }
}

// wait_flag_ge, with the GPU's clock as it arrives (t_in) and as it leaves (t_out), either may be null
__global__ void wait_flag_ge_stamped_kernel(const volatile uint32_t* flag, uint32_t value, volatile unsigned long long* t_in,
                                            volatile unsigned long long* t_out) {
    if (t_in != nullptr) *t_in = strata_now_ns();
    while (*flag < value) strata_spin_pause();
    __threadfence_system();
    if (t_out != nullptr) *t_out = strata_now_ns();
}
}  // namespace

void wait_handoff(const uint32_t* flag, uint32_t value, uint32_t* err, unsigned long long timeout_ns, void* stream) {
    wait_handoff_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(flag, value, err, timeout_ns);
    check("wait_handoff");
}

void handoff_publish(float* dst, const float* a, int64_t na, const float* b, int64_t nb, const float* c, int64_t nc,
                     uint32_t* counter, uint32_t* flag, void* stream, const uint32_t* drop) {
    const int64_t n = na + nb + nc;
    if (n <= 0 || dst == nullptr || counter == nullptr || flag == nullptr) {
        std::fprintf(stderr, "handoff_publish: invalid arguments\n");
        std::exit(1);
    }
    const int64_t nb256 = (n + 255) / 256;
    const int blocks = (int) (nb256 < 64 ? nb256 : 64);
    handoff_publish_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>(dst, a, na, b, nb, c, nc, counter, flag, drop);
    check("handoff_publish");
}

namespace {
// one thread per window entry: up to kVerifyNarrowT tokens x 10 routed experts (80), so 128 (#646 had 64: a window of
// 7+ tokens lost its last entries).  A window of 9..16 rows (up to 160 entries) runs the same kernel on 256 threads;
// one block size per window class, so a narrow window launches exactly the 128-thread kernel it always did.
constexpr int kResidentPlanNarrow = 128;
constexpr int kResidentPlanMax = kVerifyMaxT > kVerifyNarrowT ? 256 : kResidentPlanNarrow;
static_assert(kVerifyNarrowT * 10 <= kResidentPlanNarrow, "resident_plan: one thread per entry");
static_assert(kVerifyMaxT * 10 <= kResidentPlanMax, "resident_plan: one thread per entry");
// the parallel scan keeps one partial sum per warp in s_wsum[NT / 32] and packs (entries << 16 | groups) in an int
static_assert(kResidentPlanMax <= 256 && kResidentPlanMax % 32 == 0, "resident_plan: at most 8 warps");
static_assert(kResidentPlanMax < 32768, "resident_plan: the packed entry count must stay below 2^15");
template <int NT>
__global__ void __launch_bounds__(NT) resident_plan_kernel(const int32_t* __restrict__ ids, int n, int k, const int32_t* __restrict__ res,
                                     int n_expert, const uint8_t* cache_base, const unsigned long long* slot_off,
                                     long long blob, int32_t* __restrict__ pl, long long capx, uint32_t* skip,
                                     uint32_t ring, volatile uint32_t* plan_err) {
    constexpr int NW = NT / 32;
    __shared__ int32_t s_ids[NT];
    __shared__ int32_t s_excl[NT];
    __shared__ int32_t s_wsum[NW];
    __shared__ int s_bad;
    const int tid = threadIdx.x;
    if (tid == 0) s_bad = 0;
    __syncthreads();

    int32_t eid = -1;
    int32_t slot = -1;
    if (tid < n) {
        eid = ids[tid];
        s_ids[tid] = eid;
        slot = (eid >= 0 && eid < n_expert) ? res[eid] : -1;
        if (slot < 0) atomicOr(&s_bad, 1);
    }
    __syncthreads();
    if (s_bad) {
        if (tid == 0 && skip != nullptr) *skip = 0;
        if (tid == 0 && skip == nullptr) {   // #871: the all-resident graph has no host plan to fall back on
            pl[0] = 0; pl[1] = 0; pl[2] = 0;   // an empty plan: no expert runs on a stale pointer
            if (plan_err != nullptr) { *plan_err = 1; __threadfence_system(); }
        }
        return;
    }

    int first_j = tid;
    int rank_in_group = 0;
    int count_same = 0;
    if (tid < n) {
        for (int j = 0; j < n; ++j) {
            if (s_ids[j] == eid) {
                if (j < first_j) first_j = j;
                if (j < tid) ++rank_in_group;
                ++count_same;
            }
        }
    }
    const bool is_first = (tid < n && first_j == tid && slot >= 0);
    const int my_cnt = is_first ? count_same : 0;
    const int my_pack = (my_cnt << 16) | (is_first ? 1 : 0);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    int pref = my_pack;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const int up = __shfl_up_sync(0xffffffffu, pref, d);
        if (lane >= d) pref += up;
    }
    s_excl[tid] = pref - my_pack;
    if (lane == 31) {
        s_wsum[warp] = pref;
    }
    __syncthreads();

    int32_t* counts = pl;
    int32_t* start = pl + 4;
    int32_t* dst = start + capx + 1;
    int32_t* tok = dst + capx;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    unsigned long long* ptr = (unsigned long long*) (pl + ptr_off);
    int32_t* start2 = pl + ptr_off + 4 * capx;

    if (is_first) {
        int tot = s_excl[tid];
#pragma unroll
        for (int w = 0; w < NW; ++w) {
            if (w < warp) tot += s_wsum[w];
        }
        const int grp_idx = tot & 0xffff;
        const int ent_start = tot >> 16;
        ptr[grp_idx] = (unsigned long long) (cache_base + (slot_off ? (size_t) slot_off[slot] : (size_t) slot * (size_t) blob));
        start[grp_idx] = ent_start;
    }
    if (tid < n) {
        const int fj_warp = first_j >> 5;
        int fj_tot = s_excl[first_j];
#pragma unroll
        for (int w = 0; w < NW; ++w) {
            if (w < fj_warp) fj_tot += s_wsum[w];
        }
        const int out_idx = (fj_tot >> 16) + rank_in_group;
        dst[out_idx] = tid;
        tok[out_idx] = tid / k;
    }
    if (tid == 0) {
        int groups;
        if constexpr (NW == 4) groups = (s_wsum[0] + s_wsum[1] + s_wsum[2] + s_wsum[3]) & 0xffff;   // (the 128-thread kernel as it was)
        else {
            groups = 0;
#pragma unroll
            for (int w = 0; w < NW; ++w) groups += s_wsum[w];
            groups &= 0xffff;
        }
        start[groups] = n;
        start2[0] = n;
        counts[0] = groups;
        counts[1] = n;
        counts[2] = 0;
        if (skip != nullptr) {
            __threadfence();
            *skip = ring;
        }
    }
}
// The same plan in one block of 128 threads (n <= 128): thread i owns entry i. Groups are the distinct experts in
// order of first occurrence; group g's entries are its occurrences in increasing i - exactly the loop above
// (S26: the one-thread loop took ~44 us per call on gfx1151, 48 per window).
__global__ void resident_plan_par_kernel(const int32_t* __restrict__ ids, int n, int k, const int32_t* __restrict__ res,
                                         int n_expert, const uint8_t* cache_base, const unsigned long long* slot_off,
                                         long long blob, int32_t* __restrict__ pl, long long capx, uint32_t* skip,
                                         uint32_t ring) {
    __shared__ int32_t s_id[128];
    __shared__ int s_first[128], s_size[128];
    __shared__ int s_bad;
    const int i = threadIdx.x;
    if (i == 0) s_bad = 0;
    __syncthreads();
    int32_t e = -1;
    if (i < n) {
        e = ids[i];
        s_id[i] = e;
        if (e < 0 || e >= n_expert || res[e] < 0) s_bad = 1;
    }
    __syncthreads();
    if (s_bad) { if (i == 0) *skip = 0; return; }
    int first = i, rank = 0, size = 0;
    if (i < n) {
        for (int j = 0; j < n; ++j) {
            const bool same = s_id[j] == e;
            if (same && j < first) first = j;
            if (same && j < i) ++rank;
            size += same;
        }
    }
    if (i < n) { s_first[i] = first; s_size[i] = size; }
    __syncthreads();
    int32_t* counts = pl;
    int32_t* start = pl + 4;
    int32_t* dst = start + capx + 1;
    int32_t* tok = dst + capx;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    unsigned long long* ptr = (unsigned long long*) (pl + ptr_off);
    int32_t* start2 = pl + ptr_off + 4 * capx;
    if (i < n) {
        // group of entry i = the number of first occurrences before `first`; its start = their sizes' sum
        int group = 0, gstart = 0;
        for (int j = 0; j < first; ++j)
            if (s_first[j] == j) { ++group; gstart += s_size[j]; }
        if (i == first) {
            const int32_t slot = res[e];
            ptr[group] = (unsigned long long) (cache_base + (slot_off ? (size_t) slot_off[slot] : (size_t) slot * (size_t) blob));
            start[group] = gstart;
        }
        dst[gstart + rank] = i;
        tok[gstart + rank] = i / k;
        if (i == 0) {
            int groups = 0;
            for (int j = 0; j < n; ++j) groups += s_first[j] == j;
            start[groups] = n;
            start2[0] = n;
            counts[0] = groups;
            counts[1] = n;
            counts[2] = 0;
        }
    }
    __threadfence();
    __syncthreads();
    if (i == 0) *skip = ring;
}
__global__ void wait_flag_ge_or_kernel(const volatile uint32_t* flag, uint32_t value, const volatile uint32_t* skip) {
    if (*skip == value) return;
    while (*flag < value) strata_spin_pause();
    __threadfence_system();
}
__global__ void copy_i32_unless_kernel(int32_t* __restrict__ dst, const volatile int32_t* src, int n,
                                       const uint32_t* skip, uint32_t value) {
    if (*skip == value) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}
__global__ void copy_or_zero_kernel(float4* __restrict__ dst, const volatile float4* src, long long n4,
                                    const uint32_t* skip, uint32_t value) {
    const bool zero = *skip == value;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (long long) gridDim.x * blockDim.x)
        dst[i] = zero ? make_float4(0.f, 0.f, 0.f, 0.f) : const_cast<const float4*>(src)[i];
}
}  // namespace

void resident_plan(const int32_t* ids, int n_entries, int k, const int32_t* res_layer, int n_expert,
                   const uint8_t* cache_base, const unsigned long long* slot_off, long long blob, int32_t* plan,
                   long long capx, uint32_t* skip, uint32_t ring, void* stream, uint32_t* plan_err) {
    if (n_entries > kResidentPlanNarrow) {   // a 9..16-row window (kResidentPlanMax == 256 only in a wide build)
        if (kResidentPlanMax > kResidentPlanNarrow && n_entries <= kResidentPlanMax)
            resident_plan_kernel<kResidentPlanMax><<<1, kResidentPlanMax, 0, (cudaStream_t) stream>>>(ids, n_entries, k, res_layer, n_expert, cache_base, slot_off,
                                                              blob, plan, capx, skip, ring, plan_err);
        else {
            std::fprintf(stderr, "resident_plan: %d entries, at most %d\n", n_entries, kResidentPlanMax);
            std::exit(1);
        }
    } else {
        resident_plan_kernel<kResidentPlanNarrow><<<1, kResidentPlanNarrow, 0, (cudaStream_t) stream>>>(ids, n_entries, k, res_layer, n_expert, cache_base, slot_off,
                                                              blob, plan, capx, skip, ring, plan_err);
    }
    check("resident_plan");
}
void wait_flag_ge_or(const uint32_t* flag, uint32_t value, const uint32_t* skip, void* stream) {
    wait_flag_ge_or_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(flag, value, skip);
    check("wait_flag_ge_or");
}
void copy_i32_from_mapped_unless(int32_t* dst, const int32_t* src, long long n, const uint32_t* skip, uint32_t value,
                                 void* stream) {
    if (n <= 0) return;
    copy_i32_unless_kernel<<<1, 128, 0, (cudaStream_t) stream>>>(dst, (const volatile int32_t*) src, (int) n, skip, value);
    check("copy_i32_from_mapped_unless");
}
void copy_or_zero_from_mapped(float* dst, const float* src, long long n, const uint32_t* skip, uint32_t value,
                              void* stream) {
    if (n <= 0) return;
    const long long n4 = n / 4;
    const int blocks = (int) ((n4 + 255) / 256 < 64 ? (n4 + 255) / 256 : 64);
    copy_or_zero_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>((float4*) dst, (const volatile float4*) src, n4, skip,
                                                                    value);
    check("copy_or_zero_from_mapped");
}

namespace {
__global__ void wait_copy_i32_kernel(const volatile uint32_t* flag, uint32_t value, int32_t* __restrict__ dst,
                                     const volatile int32_t* src, int n) {
    if (threadIdx.x == 0) {
        while (*flag < value) strata_spin_pause();
        __threadfence_system();
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}
}  // namespace
void wait_copy_i32_from_mapped(const uint32_t* flag, uint32_t value, int32_t* dst, const int32_t* src, int64_t n,
                               void* stream) {
    wait_copy_i32_kernel<<<1, 128, 0, (cudaStream_t) stream>>>(flag, value, dst, src, (int) n);
    check("wait_copy_i32_from_mapped");
}

void wait_flag_ge(const uint32_t* flag, uint32_t value, void* stream) {
    wait_flag_ge_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(flag, value);
    check("wait_flag_ge");
}

void wait_flag_ge_stamped(const uint32_t* flag, uint32_t value, unsigned long long* t_in, unsigned long long* t_out,
                          void* stream) {
    wait_flag_ge_stamped_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(flag, value, t_in, t_out);
    check("wait_flag_ge_stamped");
}

void embedding_gather_dev(const uint8_t* codes, const float* scales, const float* offsets, const int32_t* tokens,
                          int n_tok, int64_t n, int code_bits, int code_bias, int group_elems, uint64_t row_codes,
                          uint64_t row_groups, float* out, void* stream) {
    embedding_gather_dev_kernel<<<dim3((unsigned) ((n + 255) / 256), (unsigned) n_tok), 256, 0,
                                  (cudaStream_t) stream>>>(codes, scales, offsets, tokens, n, code_bits, code_bias,
                                                           group_elems, row_codes, row_groups, out);
    check("embedding_gather_dev");
}

void broadcast_streams(const float* x, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    broadcast_streams_kernel<<<dim3((unsigned) ((n_embd * hc + 255) / 256), (unsigned) n_tok), 256, 0,
                               (cudaStream_t) stream>>>(x, R, n_embd, hc);
    check("broadcast_streams");
}

void hc_mean_streams(const float* R, float* out, int64_t n_embd, int hc, void* stream) {
    hc_mean_streams_kernel<<<(unsigned) ((n_embd + 255) / 256), 256, 0, (cudaStream_t) stream>>>(R, out, n_embd, hc);
    check("hc_mean_streams");
}

void route_marginal_stats(const int32_t* ids, int n, int k, unsigned long long* acc, void* stream) {
    route_marginal_stats_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(ids, n, k, acc);
    check("route_marginal_stats");
}

void copy_indexed(float* dst, const float* src, int64_t stride, const int32_t* index, int64_t n, void* stream) {
    const unsigned blocks = (unsigned) ((n + 255) / 256 < 64 ? (n + 255) / 256 : 64);
    copy_indexed_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>(dst, src, stride, index, n);
    check("copy_indexed");
}

// a GPU timestamp (ns, %globaltimer) into buf[i] - the verify window's stage profiler.  Pdl: launched inside a PDL
// stretch, the stamp lets the next kernel launch at once and takes its time when the one before it has finished.
namespace {
template <bool Pdl>
__global__ void gpu_stamp_kernel(unsigned long long* buf, int i) {
    if constexpr (Pdl) {
        pdl_trigger();
        pdl_wait();
    }
    unsigned long long t;
#if defined(STRATA_HIP_GFX906)
    t = wall_clock64() * 40ull;   // gfx906: the wall clock runs at 25 MHz (hipDeviceAttributeWallClockRate) -> ns
#elif defined(__HIPCC__)
    t = wall_clock64() * 10ull;   // gfx10.3 / gfx11 / gfx12: a constant 100 MHz counter, in ns
#else
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
#endif
    buf[i] = t;
}
}  // namespace
namespace {
__global__ void noop_kernel() {}
}  // namespace
void gpu_noop(void* stream) { noop_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(); }
void gpu_stamp(unsigned long long* buf, int i, void* stream) {
    if (pdl_scope()) launch_pdl(gpu_stamp_kernel<true>, dim3(1), dim3(1), 0, (cudaStream_t) stream, buf, i);
    else gpu_stamp_kernel<false><<<1, 1, 0, (cudaStream_t) stream>>>(buf, i);
}

namespace {
__global__ void copy_rows_strided_kernel(float4* __restrict__ dst, const float4* __restrict__ src, long long rows,
                                         int w4, int src_w4) {
    const long long total = rows * w4;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long long) gridDim.x * blockDim.x) {
        const long long r = i / w4, c = i - r * w4;
        dst[i] = src[r * src_w4 + c];
    }
}
}  // namespace
void copy_rows_strided(float* dst, const float* src, int64_t rows, int64_t w, int64_t src_w, void* stream) {
    if (rows <= 0 || w <= 0) return;
    if ((w & 3) || (src_w & 3) || src_w < w || ((uintptr_t) dst & 15) || ((uintptr_t) src & 15)) {
        std::fprintf(stderr, "copy_rows_strided: widths must be multiples of 4 floats (src_w >= w), pointers 16-byte "
                             "aligned\n");
        std::exit(1);
    }
    const long long total = (long long) rows * (w / 4);
    const unsigned blocks = (unsigned) ((total + 255) / 256 < 256 ? (total + 255) / 256 : 256);
    copy_rows_strided_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>((float4*) dst, (const float4*) src, rows,
                                                                         (int) (w / 4), (int) (src_w / 4));
    check("copy_rows_strided");
}

// ---- programmatic dependent launch (pdl.hpp)
bool& pdl_scope() {
    static thread_local bool on = false;
    return on;
}

bool pdl_supported() {
#if defined(__HIPCC__) || !defined(CUDART_VERSION) || CUDART_VERSION < 12030
    return false;   // HIP; or a CUDA runtime without the capture-dependency query pdl_launch_ok needs
#else
    // per device (a layer split runs on several): 1 = runs, 2 = does not.  sm_90+ (the card's, or STRATA_EMULATE_CC's),
    // and code built for it: on a build with only older code the driver JIT-compiles PTX without griddepcontrol.
    static std::atomic<int> state[64];
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    int s = state[dev].load(std::memory_order_relaxed);
    if (s == 0) {
        static const bool env_on = [] { const char* v = std::getenv("STRATA_DF_PDL"); return v != nullptr && std::atoi(v) != 0; }();   // opt-in until the A/B says otherwise
        int major = 0;
        cudaFuncAttributes fa{};
        const bool on = env_on && cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                        strata::cc_major_of(major) >= 9 &&
                        cudaFuncGetAttributes(&fa, gpu_stamp_kernel<true>) == cudaSuccess && fa.ptxVersion >= 90 &&
                        fa.binaryVersion >= 90;
        cudaGetLastError();
        s = on ? 1 : 2;
        state[dev].store(s, std::memory_order_relaxed);
    }
    return s == 1;
#endif
}

#if !defined(__HIPCC__)
bool pdl_launch_ok(const void* kernel, cudaStream_t stream) {
#if !defined(CUDART_VERSION) || CUDART_VERSION < 12030
    (void) kernel; (void) stream;
    return false;
#else
    // the kernel's own code is sm_90+ (each kernel checked once per device: its PTX, or a JIT from older PTX, decides)
    {
        static std::mutex mu;
        static std::vector<std::pair<std::pair<const void*, int>, bool>> seen;
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) { cudaGetLastError(); return false; }
        std::lock_guard<std::mutex> lock(mu);
        bool found = false, ok = false;
        for (const auto& e : seen)
            if (e.first.first == kernel && e.first.second == dev) { found = true; ok = e.second; break; }
        if (!found) {
            cudaFuncAttributes fa{};
            ok = cudaFuncGetAttributes(&fa, kernel) == cudaSuccess && fa.ptxVersion >= 90 && fa.binaryVersion >= 90;
            cudaGetLastError();
            seen.push_back({{kernel, dev}, ok});
        }
        if (!ok) return false;
    }
    // every node the next captured node will depend on is a kernel (CUDA allows a programmatic edge only between two
    // kernel nodes): never after a memcpy, memset, host or event node.  A join of branches gives several kernels, a
    // programmatic edge from each; STRATA_DF_PDL=2 allows a single predecessor only.
    static const int mode = [] { const char* v = std::getenv("STRATA_DF_PDL"); return v ? std::atoi(v) : 1; }();
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    const cudaGraphNode_t* deps = nullptr;
    const cudaGraphEdgeData* edges = nullptr;
    size_t n = 0;
#if CUDART_VERSION >= 13000
    const cudaError_t e = cudaStreamGetCaptureInfo(stream, &status, nullptr, nullptr, &deps, &edges, &n);
#else
    const cudaError_t e = cudaStreamGetCaptureInfo_v3(stream, &status, nullptr, nullptr, &deps, &edges, &n);
#endif
    if (e != cudaSuccess) { cudaGetLastError(); return false; }
    if (status != cudaStreamCaptureStatusActive || n == 0 || deps == nullptr) return false;
    if (mode == 2 && n != 1) return false;
    for (size_t i = 0; i < n; ++i) {
        if (edges != nullptr && (edges[i].type != cudaGraphDependencyTypeDefault || edges[i].from_port != 0 ||
                                 edges[i].to_port != 0))
            return false;
        cudaGraphNodeType type;
        if (cudaGraphNodeGetType(deps[i], &type) != cudaSuccess) { cudaGetLastError(); return false; }
        if (type != cudaGraphNodeTypeKernel) return false;
    }
    return true;
#endif
}
#endif

}  // namespace strata::kernels
