// src/kernels/cuda/gdn.cu - P2.S2: the gated delta-net's non-projection parts.
//
// See include/strata/kernels/gdn.hpp for the state layout (S, h_v, S) and for why the recurrence needs no
// barrier at all: every line of it touches only one (j, h) column, so one thread owns a column end to end.
#include "strata/kernels/gdn.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int JTHREADS = 32;   ///< threads along j, the state's fast axis
/// Heads staged per block.  **THIS IS A LATENCY-HIDING KNOB, NOT A CAPACITY ONE, AND IT WAS SET WRONG.**
///
/// The kernel is one thread per `(h, j)` COLUMN of the state, so the thread count is fixed at
/// `S * h_v` = 128 * 48 = **6,144** no matter how the work is packed - and `MAX_H` decides only how many of
/// those columns share a block:
///
///     MAX_H = 8   grid = (S/32, h_v/8) = (4, 6)  =  24 blocks x 32 threads  ->  1 warp on 24 of 48 SMs
///     MAX_H = 1   grid = (S/32, h_v/1) = (4, 48) = 192 blocks x 32 threads  ->  4 warps on every SM
///
/// Each thread walks `i` in 0..S-1 TWICE per head with a strided global load each step, so it is entirely
/// latency-bound: a warp that issues one 128-byte transaction and then waits ~600 cycles needs other warps to
/// run meanwhile, and at 24 blocks there were none.  Measured 0.2800 ms per GDN block - 10 ms of a 71.5 ms
/// token, 14%, in a kernel touching 12.6 MB that a 643 GB/s card should do in 0.020 ms.
///
/// The arithmetic does not change at all: the same columns are computed by the same code, in different blocks.
constexpr int MAX_H = 1;       ///< heads staged in shared memory per block

/// ggml_compute_softplus_f32: log1p(exp(x)), with the large-x branch that avoids overflow.
__device__ __forceinline__ float softplus_f(float x) { return x > 20.0f ? x : log1pf(expf(x)); }
__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

/// One step of the recurrence.  One thread per (h, j) column of the state.
///
/// `ks`/`qs` stage `k[idx[h]]` and `q[idx[h]]` for the block's heads.  Within a warp h is constant and j
/// varies, so the staged row is read with a single broadcast - which is also why the GLOBAL loads of
/// `q`/`k` are done once per block into shared instead of once per thread.
__global__ void gdn_step_kernel(float* __restrict__ state, const float* __restrict__ q,
                                const float* __restrict__ k, const float* __restrict__ v,
                                const float* __restrict__ gate, const float* __restrict__ beta,
                                float* __restrict__ o, int S, int h_k, int h_v) {
    __shared__ float ks[MAX_H][128];
    __shared__ float qs[MAX_H][128];
    __shared__ float dec_s[MAX_H];
    __shared__ float beta_s[MAX_H];

    const int h0 = blockIdx.y * MAX_H;
    const int nh = min(MAX_H, h_v - h0);
    const int j = blockIdx.x * JTHREADS + threadIdx.x;

    // stage this block's q/k rows and its per-head scalars
    for (int hi = 0; hi < nh; ++hi) {
        const int h = h0 + hi;
        const int src = h % h_k;                       // MODULO head pairing
        for (int i = threadIdx.x; i < S; i += JTHREADS) {
            ks[hi][i] = k[src * S + i];
            qs[hi][i] = q[src * S + i];
        }
        if (threadIdx.x == 0) {
            dec_s[hi] = expf(gate[h]);
            beta_s[hi] = beta[h];
        }
    }
    __syncthreads();
    if (j >= S) return;

    for (int hi = 0; hi < nh; ++hi) {
        const int h = h0 + hi;
        const float dec = dec_s[hi], b = beta_s[hi];
        float* col = state + (size_t) h * S + j;        // (S, h_v, S) with j fastest
        const size_t stride = (size_t) h_v * S;

        // pass 1: decay the state and contract it against k.  Decay BEFORE the update - PROPERTY 5.
        float sk = 0.0f;
        for (int i = 0; i < S; ++i) {
            const float s = col[(size_t) i * stride] * dec;
            col[(size_t) i * stride] = s;
            sk += s * ks[hi][i];
        }
        const float d = (v[(size_t) h * S + j] - sk) * b;
        // pass 2: the rank-1 update, then read out against q using the UPDATED state
        float dot = 0.0f;
        for (int i = 0; i < S; ++i) {
            const float s = col[(size_t) i * stride] + ks[hi][i] * d;
            col[(size_t) i * stride] = s;
            dot += s * qs[hi][i];
        }
        o[(size_t) h * S + j] = dot;
    }
}

/// `ggml_ssm_conv`: out[c] = sum_i inp[i][c] * kW[c*d_conv + i], inp = [conv_state | x], and the state slides.
/// One thread per channel; d_conv is 4, so the serial work per thread is a 4-tap dot and an unrolled shift.
__global__ void gdn_conv_kernel(float* __restrict__ cs, const float* __restrict__ x,
                                const float* __restrict__ kW, float* __restrict__ out, int C, int dc) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float* st = cs + (size_t) c * (dc - 1);
    const float* w = kW + (size_t) c * dc;
    float acc = 0.0f;
    for (int i = 0; i < dc - 1; ++i) acc += st[i] * w[i];   // kernel[0] reads the OLDEST state row
    acc += x[c] * w[dc - 1];                                // the new input lands in the LAST tap
    out[c] = acc;
    for (int i = 0; i < dc - 2; ++i) st[i] = st[i + 1];     // slide: drop the oldest, append the newest
    st[dc - 2] = x[c];
}

/// `build_gdn_l2_norm`, one warp per row: x / sqrt(sum(x^2) + eps) over the last axis.
///
/// The sum is accumulated in DOUBLE, as `ref/gdn.py` does (`l2_norm(q.astype(np.float64), eps)`), because the
/// reference deliberately computes this one in float64 and then casts - so matching its precision is part of
/// transcribing it.
__global__ void gdn_l2_kernel(float* __restrict__ x, int cols, float eps) {
    const int row = blockIdx.x;
    float* p = x + (size_t) row * cols;
    double acc = 0.0;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) acc += (double) p[i] * (double) p[i];
    // warp reduction
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    __shared__ double ssum;
    if (threadIdx.x == 0) ssum = acc;
    __syncthreads();
    const float inv = (float) (1.0 / sqrt(ssum + (double) eps));
    for (int i = threadIdx.x; i < cols; i += blockDim.x) p[i] *= inv;
}

/// y = rms_norm(o, eps) * ssm_norm * sigmoid(z), one warp per head, norm over that head's S values.
__global__ void gdn_out_norm_kernel(const float* __restrict__ o, const float* __restrict__ z,
                                    const float* __restrict__ ssm_norm, float* __restrict__ y, int S,
                                    float eps) {
    const int h = blockIdx.x;
    const float* po = o + (size_t) h * S;
    const float* pz = z + (size_t) h * S;
    float* py = y + (size_t) h * S;
    double acc = 0.0;
    for (int i = threadIdx.x; i < S; i += blockDim.x) acc += (double) po[i] * (double) po[i];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    __shared__ double ssum;
    if (threadIdx.x == 0) ssum = acc;
    __syncthreads();
    const float inv = (float) (1.0 / sqrt(ssum / (double) S + (double) eps));
    for (int i = threadIdx.x; i < S; i += blockDim.x)
        py[i] = po[i] * inv * ssm_norm[i] * sigmoid_f(pz[i]);
}

}  // namespace

void gdn_step(float* state, const float* q, const float* k, const float* v, const float* gate,
              const float* beta, float* o, const GdnShapes& s, void* stream) {
    if (s.S <= 0 || s.h_k <= 0 || s.h_v <= 0) return;
    if (s.S > 128) {
        std::fprintf(stderr, "gdn_step: S = %lld exceeds the staged 128 (`ks`/`qs` are [8][128])\n",
                     (long long) s.S);
        std::exit(1);
    }
    const dim3 grid((unsigned) ((s.S + JTHREADS - 1) / JTHREADS),
                    (unsigned) ((s.h_v + MAX_H - 1) / MAX_H));
    gdn_step_kernel<<<grid, JTHREADS, 0, (cudaStream_t) stream>>>(state, q, k, v, gate, beta, o, (int) s.S,
                                                                  (int) s.h_k, (int) s.h_v);
    if (stream == nullptr) {
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "gdn_step: %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
    }
}

void gdn_conv_step(float* conv_state, const float* x, const float* kW, float* out, int64_t channels,
                   int64_t d_conv, void* stream) {
    if (channels <= 0 || d_conv < 1) return;
    const int blocks = (int) ((channels + 255) / 256);
    gdn_conv_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>(conv_state, x, kW, out, (int) channels,
                                                               (int) d_conv);
    if (stream == nullptr) {
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "gdn_conv_step: %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
    }
}

void gdn_l2_norm(float* x, int64_t rows, int64_t cols, float eps, void* stream) {
    if (rows <= 0 || cols <= 0) return;
    if (cols > 1024) {
        std::fprintf(stderr, "gdn_l2_norm: cols = %lld exceeds the 1024-wide warp reduction\n", (long long) cols);
        std::exit(1);
    }
    gdn_l2_kernel<<<(unsigned) rows, 32, 0, (cudaStream_t) stream>>>(x, (int) cols, eps);
    if (stream == nullptr) {
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "gdn_l2_norm: %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
    }
}

/// `beta = sigmoid(beta)`, in place, over the `h_v` per-head scalars.
///
/// **THIS IS `build_layer_attn_linear` L889, AND LEAVING IT OUT WAS THE C1 BUG.**  The reference computes
/// `beta = ggml_sigmoid(ssm_beta @ cur)` and hands the RESULT to the delta net; `gdn_step_kernel` is documented
/// as `d[h,j] = (v[h,j] - sk[h,j]) * beta[h]` and applies no sigmoid of its own, so the layer glue owes it one.
/// The engine passed the raw pre-activation, which is unbounded and signed - the delta-net write strength then
/// stops being a fraction and the whole recurrence is wrong from layer 0.
///
/// **IT SURVIVED `gdn_parity` BECAUSE THAT TEST SUPPLIES ITS OWN `beta`.**  The kernel was always right; only
/// the glue between the projection and the kernel was wrong, and `ref/gdn.py` (`beta = sigmoid(wbeta @ x)`) and
/// `ref/model.py` (`beta = G.sigmoid(proj(...))`) both get it right - so the two transcriptions agreed with
/// each other and neither was ever compared against the C++ that actually runs.
__global__ void gdn_beta_gate_kernel(float* __restrict__ beta, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) beta[i] = sigmoid_f(beta[i]);
}

void gdn_beta_gate(float* beta, int64_t h_v, void* stream) {
    if (beta == nullptr || h_v <= 0) return;
    const int n = (int) h_v;
    gdn_beta_gate_kernel<<<(unsigned) ((n + 127) / 128), 128, 0, (cudaStream_t) stream>>>(beta, n);
    if (stream == nullptr) {
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "gdn_beta_gate: %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
    }
}

void gdn_out_norm(const float* o, const float* z, const float* ssm_norm, float* y, int64_t h_v, int64_t S,
                  float eps, void* stream) {
    if (h_v <= 0 || S <= 0) return;
    gdn_out_norm_kernel<<<(unsigned) h_v, 32, 0, (cudaStream_t) stream>>>(o, z, ssm_norm, y, (int) S, eps);
    if (stream == nullptr) {
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "gdn_out_norm: %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
    }
}

}  // namespace strata::kernels
