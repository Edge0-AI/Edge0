// src/qwen35/q36ref.cu - NEW-MODELS-PLAN.md phase T1: the reference path for Qwen3.6-35B-A3B (GGUF architecture
// qwen35moe), exactness first.  One GPU, the whole model resident in VRAM (UD-Q4_K_XL: 21.3 GiB on a 32 GB card),
// one token at a time (the prompt too), fp32 activations, every weight read in its GGUF form.  Two kernel sets:
// --kernels plain, this file's own plain kernels (the reference), and --kernels strata (default), Strata's native
// MMVQ for the quantized matmuls (dense and routed experts) and its fused GDN step with the SiLU-gated norm - the
// engine's kernels on the new family's shapes.  No expert cache, no MTP, no graphs: checked against llama.cpp
// (per-layer hidden states, greedy tokens, teacher-forced KL: tools/qwen35/).
//
// The forward pass, as llama.cpp's src/models/qwen35moe.cpp builds it:
//   x = token_embd[tok]
//   per layer l < 40:  h = rms(x) * attn_norm
//       GDN (l % 4 != 3): qkv = W_qkv h (8192), z = W_gate h (4096), b = sigmoid(W_beta h), a = W_alpha h (32 each)
//            c = silu(causal conv4(qkv)); q, k = l2norm(c[0:2048]), l2norm(c[2048:4096]) (16 heads), v = c[4096:] (32)
//            g = softplus(a + dt_bias) * ssm_a;  per V head h (K head h % 16):  S = S * exp(g_h);
//            d = (v - S k) * b_h;  S += d k^T;  o = S q / sqrt(128);  o = rms(o) * ssm_norm * silu(z_h)
//            y = W_ssm_out o
//       attention (l % 4 == 3): attn_q -> per head [q 256 | gate 256]; q, k = rms(.) * q_norm/k_norm; NEOX RoPE over the
//            first 64 dims (base 1e7; text positions: the M-RoPE sections all carry the same position); GQA 16 / 2;
//            y = W_o (softmax(q k^T / 16) v * sigmoid(gate))
//       x += y;  h2 = rms(x) * post_attention_norm
//       MoE: p = softmax(W_router h2) over 256, top-8, renormalised; sum_e p_e W_down_e (silu(W_gate_e h2) * W_up_e h2)
//            + sigmoid(w_shexp_gate . h2) * W_down_s (silu(W_gate_s h2) * W_up_s h2);  x += moe
//   logits = W_output (rms(x) * output_norm)
//
// Usage: strata-q36ref --model Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --tokens 248045,846,198,... [--n 64] [--ctx 4096]
//                      [--dump-layers FILE] [--dump-logits FILE] [--act q8|f32] [--kernels strata|plain]
//                      [--force IDS] [--dump-full-logits F]
//   prints the greedy ids ("gen: id id ..."); --dump-layers writes every processed position's 40 layer outputs (f32,
//   [pos][layer][2048]); --dump-logits writes each greedy step's top-2 (id, logit) pairs for the near-tie report;
//   --force feeds the given continuation instead of the greedy ids and --dump-full-logits writes every step's
//   logits (f32, [step][vocab]) for the teacher-forced KL check (tools/qwen35/kl.py).
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/family_layout.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_router.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

#define CK(x)                                                                                                          \
    do {                                                                                                               \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "q36ref: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

// ------------------------------------------------------------------------------------------------ dequantisation
// ggml's block formats, 32 values at a time ("chunk" c of a row).  The formulas are ggml-quants.c's dequantize_row_*.
enum : int { T_F32 = 0, T_F16 = 1, T_Q8_0 = 8, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14, T_BF16 = 30 };

__device__ __forceinline__ float h2f(uint16_t h) {
    const uint32_t s = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e == 0) {
        const float v = ldexpf((float) m, -24);
        return s ? -v : v;
    }
    const uint32_t f = e == 31 ? (s | 0x7f800000u | (m << 13)) : (s | ((e + 112) << 23) | (m << 13));
    return __uint_as_float(f);
}
__device__ __forceinline__ uint16_t ld16(const uint8_t* p) { return (uint16_t) (p[0] | (p[1] << 8)); }

__device__ __forceinline__ void scale_min_k4(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

/// The 32 values of chunk `c` of one row (`row` = the row's first byte).
template <int T> __device__ void deq32(const uint8_t* row, int c, float* out) {
    if constexpr (T == T_F32) {
        const float* r = (const float*) row + 32 * c;
        for (int l = 0; l < 32; ++l) out[l] = r[l];
    } else if constexpr (T == T_F16) {
        const uint8_t* r = row + 64 * c;
        for (int l = 0; l < 32; ++l) out[l] = h2f(ld16(r + 2 * l));
    } else if constexpr (T == T_BF16) {
        const uint8_t* r = row + 64 * c;
        for (int l = 0; l < 32; ++l) out[l] = __uint_as_float((uint32_t) ld16(r + 2 * l) << 16);
    } else if constexpr (T == T_Q8_0) {
        const uint8_t* b = row + 34 * c;
        const float d = h2f(ld16(b));
        const int8_t* q = (const int8_t*) (b + 2);
        for (int l = 0; l < 32; ++l) out[l] = d * (float) q[l];
    } else if constexpr (T == T_Q4_K) {
        const uint8_t* b = row + 144 * (c >> 3);
        const int j = c & 7;
        const float d = h2f(ld16(b)), dmin = h2f(ld16(b + 2));
        int sc, m;
        scale_min_k4(j, b + 4, sc, m);
        const float d1 = d * (float) sc, m1 = dmin * (float) m;
        const uint8_t* qs = b + 16 + 32 * (j >> 1);
        for (int l = 0; l < 32; ++l) out[l] = d1 * (float) ((j & 1) ? (qs[l] >> 4) : (qs[l] & 0xF)) - m1;
    } else if constexpr (T == T_Q5_K) {
        const uint8_t* b = row + 176 * (c >> 3);
        const int j = c & 7;
        const float d = h2f(ld16(b)), dmin = h2f(ld16(b + 2));
        int sc, m;
        scale_min_k4(j, b + 4, sc, m);
        const float d1 = d * (float) sc, m1 = dmin * (float) m;
        const uint8_t* qh = b + 16;
        const uint8_t* qs = b + 48 + 32 * (j >> 1);
        for (int l = 0; l < 32; ++l) {
            const int q = ((j & 1) ? (qs[l] >> 4) : (qs[l] & 0xF)) + ((qh[l] >> j) & 1 ? 16 : 0);
            out[l] = d1 * (float) q - m1;
        }
    } else if constexpr (T == T_Q6_K) {
        const uint8_t* b = row + 210 * (c >> 3);
        const int j = c & 7, n = j >> 2, r = j & 3;
        const uint8_t* ql = b + 64 * n;
        const uint8_t* qh = b + 128 + 32 * n;
        const int8_t* sc = (const int8_t*) (b + 192) + 8 * n;
        const float d = h2f(ld16(b + 208));
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            int q;
            if (r == 0) q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
            else if (r == 1) q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
            else if (r == 2) q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
            else q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
            out[l] = d * (float) sc[is + 2 * r] * (float) (q - 32);
        }
    }
}

__device__ float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

constexpr int kThreads = 128;

/// y[sel][o] = W_e[o] . x[sel] for row o (grid.x), selection `sel` (grid.y): expert ids[sel] (or 0 without ids),
/// x of selection sel at x + sel * x_stride.  `beta` != 0 adds into y.
template <int T>
__global__ void k_gemv(const uint8_t* W, size_t row_bytes, size_t expert_bytes, const int* ids, int n_in, const float* x,
                       size_t x_stride, float* y, int n_out) {
    __shared__ float sh[kThreads];
    const int o = blockIdx.x, sel = blockIdx.y;
    const int e = ids ? ids[sel] : 0;
    const uint8_t* row = W + (size_t) e * expert_bytes + (size_t) o * row_bytes;
    const float* xs = x + (size_t) sel * x_stride;
    float acc = 0.f, v[32];
    for (int c = threadIdx.x; c < n_in / 32; c += blockDim.x) {
        deq32<T>(row, c, v);
        const float* xc = xs + 32 * c;
        for (int l = 0; l < 32; ++l) acc += v[l] * xc[l];
    }
    acc = block_sum(acc, sh);
    if (threadIdx.x == 0) y[(size_t) sel * n_out + o] = acc;
}

template <int T> __global__ void k_row(const uint8_t* W, size_t row_bytes, int row, int n, float* out) {
    const uint8_t* r = W + (size_t) row * row_bytes;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < n / 32; c += gridDim.x * blockDim.x) deq32<T>(r, c, out + 32 * c);
}

/// y = x / sqrt(mean(x^2) + eps) * w, per group of n (grid = groups); w (length n) may be null.
__global__ void k_rms(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ float sh[kThreads];
    const float* xg = x + (size_t) blockIdx.x * n;
    float* yg = y + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xg[i] * xg[i];
    s = block_sum(s, sh);
    const float r = 1.0f / sqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yg[i] = xg[i] * r * (w ? w[i] : 1.0f);
}

__device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }
__device__ __forceinline__ float sigm(float v) { return 1.0f / (1.0f + expf(-v)); }
__device__ __forceinline__ float softplus(float v) { return v > 20.f ? v : log1pf(expf(v)); }

/// The causal 4-tap conv over the 8192 qkv channels with its 3-input state, then SiLU; the state moves by one.
__global__ void k_conv(const float* qkv, float* state, const float* w, float* out, int C, int K) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float acc = 0.f;
    for (int k = 0; k < K - 1; ++k) acc += state[(size_t) k * C + c] * w[(size_t) c * K + k];
    acc += qkv[c] * w[(size_t) c * K + K - 1];
    for (int k = 0; k < K - 2; ++k) state[(size_t) k * C + c] = state[(size_t) (k + 1) * C + c];
    state[(size_t) (K - 2) * C + c] = qkv[c];
    out[c] = silu(acc);
}

/// One gated delta-rule step per V head (grid = HV, block = D = 128): S[i][j] row i owned by thread i.
__global__ void k_gdn(const float* conv, const float* alpha, const float* dt_bias, const float* ssm_a,
                      const float* beta_raw, float* S, float* out, int HK, int /*HV: the grid*/, int D, float eps) {
    extern __shared__ float sm[];
    float* q = sm;
    float* k = sm + D;
    float* red = sm + 2 * D;
    const int h = blockIdx.x, kh = h % HK, i = threadIdx.x;
    const float qi = conv[kh * D + i], ki = conv[HK * D + kh * D + i], vi = conv[2 * HK * D + h * D + i];
    const float qn = block_sum(qi * qi, red), kn = block_sum(ki * ki, red);
    q[i] = qi / sqrtf(qn + eps) / sqrtf((float) D);
    k[i] = ki / sqrtf(kn + eps);
    __syncthreads();
    const float g = expf(softplus(alpha[h] + dt_bias[h]) * ssm_a[h]);
    const float b = sigm(beta_raw[h]);
    float* Si = S + ((size_t) h * D + i) * D;
    float sk = 0.f;
    for (int j = 0; j < D; ++j) {
        Si[j] *= g;
        sk += Si[j] * k[j];
    }
    const float d = (vi - sk) * b;
    float o = 0.f;
    for (int j = 0; j < D; ++j) {
        Si[j] += d * k[j];
        o += Si[j] * q[j];
    }
    out[h * D + i] = o;
}

/// The fused path's gate and beta, in place: alpha -> softplus(alpha + dt) * ssm_a (exp'd by the step), beta -> sigmoid.
__global__ void k_gdn_gb(float* alpha, const float* dt_bias, const float* ssm_a, float* beta, int HV) {
    const int h = threadIdx.x;
    if (h >= HV) return;
    alpha[h] = softplus(alpha[h] + dt_bias[h]) * ssm_a[h];
    beta[h] = sigm(beta[h]);
}

/// o = rms(o) * w * silu(z), per V head (grid = HV, block = D).
__global__ void k_gated_norm(float* o, const float* w, const float* z, int D, float eps) {
    __shared__ float sh[kThreads];
    float* oh = o + (size_t) blockIdx.x * D;
    const float v = oh[threadIdx.x];
    const float s = block_sum(v * v, sh);
    oh[threadIdx.x] = v / sqrtf(s / (float) D + eps) * w[threadIdx.x] * silu(z[blockIdx.x * D + threadIdx.x]);
}

/// The attention layer's per-token preparation: q (from the [q | gate] pairs) and k RMS-normed per head with their
/// weights, NEOX RoPE over the first n_rot dims, k and v appended to the cache at `pos`, the gate kept.
__global__ void k_attn_prep(const float* qg, const float* kx, const float* vx, const float* qw, const float* kw,
                            float* q, float* gate, float* kc, float* vc, int H, int /*HKV: the grid*/, int D, int n_rot, int pos,
                            int ctx, float base, float eps) {
    __shared__ float sh[kThreads];
    __shared__ float buf[512];
    const int head = blockIdx.x;   // 0..H-1: query heads, H..H+HKV-1: key heads
    const bool isq = head < H;
    const float* src = isq ? qg + (size_t) head * 2 * D : kx + (size_t) (head - H) * D;
    float s = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) s += src[i] * src[i];
    s = block_sum(s, sh);
    const float r = 1.0f / sqrtf(s / (float) D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) buf[i] = src[i] * r * (isq ? qw[i] : kw[i]);
    __syncthreads();
    float* dst = isq ? q + (size_t) head * D : kc + ((size_t) (head - H) * ctx + pos) * D;
    const int half = n_rot / 2;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = buf[i];
        if (i < n_rot) {
            const int p = i < half ? i : i - half;
            const float theta = (float) pos * powf(base, -2.0f * (float) p / (float) n_rot);
            const float cs = cosf(theta), sn = sinf(theta);
            v = i < half ? buf[i] * cs - buf[i + half] * sn : buf[i - half] * sn + buf[i] * cs;
        }
        dst[i] = v;
    }
    if (isq)
        for (int i = threadIdx.x; i < D; i += blockDim.x) gate[(size_t) head * D + i] = qg[(size_t) head * 2 * D + D + i];
    else
        for (int i = threadIdx.x; i < D; i += blockDim.x)
            vc[((size_t) (head - H) * ctx + pos) * D + i] = vx[(size_t) (head - H) * D + i];
}

/// softmax(q k^T * scale) v for one query head per block (GQA: KV head = head / (H / HKV)), times sigmoid(gate).
__global__ void k_attn(const float* q, const float* kc, const float* vc, const float* gate, float* out, int H, int HKV,
                       int D, int n_pos, int ctx, float scale) {
    extern __shared__ float sc[];
    __shared__ float sh[kThreads];
    const int h = blockIdx.x, kvh = h / (H / HKV);
    const float* qh = q + (size_t) h * D;
    for (int p = threadIdx.x; p < n_pos; p += blockDim.x) {
        const float* kp = kc + ((size_t) kvh * ctx + p) * D;
        float s = 0.f;
        for (int i = 0; i < D; ++i) s += qh[i] * kp[i];
        sc[p] = s * scale;
    }
    __syncthreads();
    float mx = -INFINITY;
    for (int p = threadIdx.x; p < n_pos; p += blockDim.x) mx = fmaxf(mx, sc[p]);
    sh[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    mx = sh[0];
    __syncthreads();
    float sum = 0.f;
    for (int p = threadIdx.x; p < n_pos; p += blockDim.x) {
        sc[p] = expf(sc[p] - mx);
        sum += sc[p];
    }
    sum = block_sum(sum, sh);
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float acc = 0.f;
        for (int p = 0; p < n_pos; ++p) acc += sc[p] * vc[((size_t) kvh * ctx + p) * D + i];
        out[(size_t) h * D + i] = acc / sum * sigm(gate[(size_t) h * D + i]);
    }
}

__global__ void k_swiglu(const float* g, const float* u, float* h, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = silu(g[i]) * u[i];
}
__global__ void k_add(float* x, const float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}
/// x += sum_s w[s] * y[s] (the routed experts), + sigmoid(gs) * ys (the shared expert)
__global__ void k_moe_sum(float* x, const float* y, const float* w, int k, const float* ys, const float* gs, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.f;
    for (int s = 0; s < k; ++s) acc += w[s] * y[(size_t) s * n + i];
    x[i] += acc + sigm(gs[0]) * ys[i];
}

// ------------------------------------------------------------------------------------------------ the model
struct Tensor {
    const uint8_t* d = nullptr;   // device
    int type = -1;
    std::vector<int64_t> shape;
    size_t row_bytes = 0;
};

size_t row_bytes_of(int type, int64_t n) {
    int be = 0, bb = 0;
    if (!strata::block_geometry((uint32_t) type, be, bb)) return 0;
    return (size_t) (n / be) * (size_t) bb;
}

template <int T>
void gemv_t(const Tensor& w, size_t expert_bytes, const int* ids, int n_sel, const float* x, size_t x_stride, float* y) {
    const int n_in = (int) w.shape[0], n_out = (int) w.shape[1];
    k_gemv<T><<<dim3(n_out, n_sel), kThreads>>>(w.d, w.row_bytes, expert_bytes, ids, n_in, x, x_stride, y, n_out);
}

/// llama.cpp's GPU matmuls on quantized weights (MMVQ for one token, MMQ for a batch) read the activations as q8_1:
/// per 32 values d = amax / 127, q = round(x / d).  --act q8 (the default) gives this path the same rounding, so
/// its numbers follow llama.cpp's instead of drifting from them by the activation rounding; --act f32 reads them as
/// they are.  `x` (n_sel vectors of n, `stride` apart) -> `out` (packed, n apart).
__global__ void k_act_q8(const float* x, size_t stride, int n, float* out) {
    const int sel = blockIdx.y, blk = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, l = threadIdx.x % 32;
    if (blk * 32 >= n) return;
    const float v = x[(size_t) sel * stride + blk * 32 + l];
    __shared__ float am[kThreads];
    am[threadIdx.x] = fabsf(v);
    __syncthreads();
    const int base = threadIdx.x - l;
    float amax = 0.f;
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, am[base + i]);
    const float d = amax / 127.0f;
    const float q = amax == 0.0f ? 0.0f : roundf(v / d);
    out[(size_t) sel * n + blk * 32 + l] = q * d;
}

bool g_act_q8 = true;
float* g_act_buf = nullptr;
constexpr size_t kActBuf = 1 << 16;

// --kernels strata (the default): the quantized matmuls run on Strata's native MMVQ (llama.cpp's pinned MMVQ: q8_1
// activations, one launch per routed expert) and the GDN step on Strata's fused kernels (conv + SiLU + L2, then the
// step with the SiLU-gated norm); --kernels plain keeps every op on this file's own kernels (the reference).  The
// Strata kernels run on a blocking stream, so they and the null-stream work here stay in order.
bool g_strata = true;
cudaStream_t g_st = nullptr;
void* g_q8 = nullptr;
std::vector<int> g_ids_host;   // the routed experts of the current layer (the native expert path launches per id)

void gemv(const Tensor& w, const float* x, float* y, const int* ids = nullptr, int n_sel = 1, size_t x_stride = 0) {
    const size_t eb = w.shape.size() == 3 ? w.row_bytes * (size_t) w.shape[1] : 0;
    const bool quant = w.type == T_Q8_0 || w.type == T_Q4_K || w.type == T_Q5_K || w.type == T_Q6_K;
    if (quant && g_strata) {
        const int n_in = (int) w.shape[0], n_out = (int) w.shape[1], cols = x_stride ? n_sel : 1;
        if (x_stride && x_stride != (size_t) n_in) {
            std::fprintf(stderr, "q36ref: the native path wants contiguous input columns\n");
            std::exit(1);
        }
        strata::kernels::native_quantize_q8_1(x, g_q8, n_in, cols, g_st);
        const size_t qb = strata::kernels::native_q8_1_bytes(n_in, 1);
        for (int sel = 0; sel < n_sel; ++sel) {
            const uint8_t* we = w.d + (ids ? (size_t) g_ids_host[(size_t) sel] * eb : 0);
            strata::kernels::native_mmvq(w.type, we, (const char*) g_q8 + (x_stride ? (size_t) sel * qb : 0),
                                         y + (size_t) sel * n_out, n_in, n_out, 1, g_st);
        }
        CK(cudaGetLastError());
        return;
    }
    if (quant && g_act_q8) {
        const int n = (int) w.shape[0], sel = x_stride ? n_sel : 1;
        if ((size_t) n * sel > kActBuf) {
            std::fprintf(stderr, "q36ref: activation scratch too small\n");
            std::exit(1);
        }
        k_act_q8<<<dim3((n / 32 + kThreads / 32 - 1) / (kThreads / 32), sel), kThreads>>>(x, x_stride, n, g_act_buf);
        x = g_act_buf;
        x_stride = x_stride ? (size_t) n : 0;
    }
    switch (w.type) {
    case T_F32: gemv_t<T_F32>(w, eb, ids, n_sel, x, x_stride, y); break;
    case T_F16: gemv_t<T_F16>(w, eb, ids, n_sel, x, x_stride, y); break;
    case T_BF16: gemv_t<T_BF16>(w, eb, ids, n_sel, x, x_stride, y); break;
    case T_Q8_0: gemv_t<T_Q8_0>(w, eb, ids, n_sel, x, x_stride, y); break;
    case T_Q4_K: gemv_t<T_Q4_K>(w, eb, ids, n_sel, x, x_stride, y); break;
    case T_Q5_K: gemv_t<T_Q5_K>(w, eb, ids, n_sel, x, x_stride, y); break;
    case T_Q6_K: gemv_t<T_Q6_K>(w, eb, ids, n_sel, x, x_stride, y); break;
    default:
        std::fprintf(stderr, "q36ref: no kernel for ggml type %d (%s)\n", w.type, strata::ggml_type_name((uint32_t) w.type));
        std::exit(1);
    }
    CK(cudaGetLastError());
}

/// A 1-D tensor as f32 on the device (F32 as is, BF16 widened at load).
struct Model {
    strata::core::ModelGeometry g;
    int64_t top_k = 8;
    float eps = 1e-6f, rope_base = 1e7f;
    int n_rot = 64;
    std::map<std::string, Tensor> t;
    std::map<std::string, float*> f32;   // the small vectors, f32

    const Tensor& W(const std::string& n) const {
        auto it = t.find(n);
        if (it == t.end()) {
            std::fprintf(stderr, "q36ref: missing tensor %s\n", n.c_str());
            std::exit(1);
        }
        return it->second;
    }
    const float* F(const std::string& n) const {
        auto it = f32.find(n);
        if (it == f32.end()) {
            std::fprintf(stderr, "q36ref: missing vector %s\n", n.c_str());
            std::exit(1);
        }
        return it->second;
    }
};

float h2f_host(uint16_t h) {
    const uint32_t s = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e == 0) return (s ? -1.f : 1.f) * std::ldexp((float) m, -24);
    const uint32_t f = e == 31 ? (s | 0x7f800000u | (m << 13)) : (s | ((e + 112) << 23) | (m << 13));
    float r;
    std::memcpy(&r, &f, 4);
    return r;
}

bool load(const std::string& path, Model& m, std::string& err) {
    const auto t0 = std::chrono::steady_clock::now();
    const strata::GgufModel gm = strata::GgufModel::open(path);
    if (!strata::core::geometry_from_gguf(gm.meta(), m.g, m.top_k, err)) return false;
    if (m.g.family != strata::ModelFamily::Qwen35Moe) {
        err = "this reference path is for Qwen3.6-35B-A3B (qwen35moe) only";
        return false;
    }
    size_t checked = 0;
    if (!strata::core::check_gguf_layout(gm, m.g, err, &checked)) return false;
    const std::string p = "qwen35moe.";
    if (const strata::MetaValue* v = gm.meta().get(p + "attention.layer_norm_rms_epsilon")) m.eps = (float) v->num();
    if (const strata::MetaValue* v = gm.meta().get(p + "rope.freq_base")) m.rope_base = (float) v->num();
    if (const strata::MetaValue* v = gm.meta().get(p + "rope.dimension_count")) m.n_rot = (int) v->u;
    uint64_t bytes = 0;
    for (size_t s = 0; s < gm.size(); ++s) {
        const strata::GgufFile& f = gm.shard(s);
        for (const strata::TensorInfo& ti : f.tensors()) {
            const uint64_t nb = strata::tensor_payload_bytes(ti);
            const uint8_t* src = f.tensor_data(ti);
            // the MTP block (blk.40) is not run by this path: not loaded
            if (ti.name.rfind("blk." + std::to_string(m.g.n_layers) + ".", 0) == 0) continue;
            const bool conv = ti.name.size() > 17 && ti.name.compare(ti.name.size() - 17, 17, "ssm_conv1d.weight") == 0;
            if (ti.shape.size() == 1 || conv) {
                // the small vectors (and the conv taps): f32 on the device
                std::vector<float> h(ti.elements());
                if (ti.type == T_F32) std::memcpy(h.data(), src, h.size() * 4);
                else if (ti.type == T_BF16)
                    for (size_t i = 0; i < h.size(); ++i) {
                        const uint32_t u = (uint32_t) (src[2 * i] | (src[2 * i + 1] << 8)) << 16;
                        std::memcpy(&h[i], &u, 4);
                    }
                else if (ti.type == T_F16)
                    for (size_t i = 0; i < h.size(); ++i) h[i] = h2f_host((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
                else {
                    err = ti.name + ": a 1-D tensor of type " + ti.type_name();
                    return false;
                }
                float* d = nullptr;
                CK(cudaMalloc(&d, h.size() * 4));
                CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
                m.f32[ti.name] = d;
                bytes += h.size() * 4;
                continue;
            }
            Tensor w;
            w.type = (int) ti.type;
            for (uint64_t d : ti.shape) w.shape.push_back((int64_t) d);
            w.row_bytes = row_bytes_of(w.type, w.shape[0]);
            uint8_t* d = nullptr;
            CK(cudaMalloc(&d, nb));
            CK(cudaMemcpy(d, src, nb, cudaMemcpyHostToDevice));
            w.d = d;
            m.t[ti.name] = w;
            bytes += nb;
        }
    }
    std::fprintf(stderr, "q36ref: %s: %zu tensors (layout checked), %.2f GiB on the GPU in %.1f s\n", path.c_str(),
                 checked, (double) bytes / (1 << 30),
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return true;
}

struct State {
    int ctx = 4096, pos = 0;
    // scratch
    float *x, *h, *y, *qkv, *z, *alpha, *beta, *conv_out, *gdn_out, *qg, *k, *v, *q, *gate, *att, *router, *eg, *eu, *eh,
        *ey, *sg, *su, *sh, *sy, *sgate, *wts, *logits;
    int* ids;
    std::vector<float*> conv_state, S, kc, vc;   // per layer (GDN: conv + S; attention: K/V cache)
};

float* dalloc(size_t n) {
    float* p = nullptr;
    CK(cudaMalloc(&p, n * 4));
    CK(cudaMemset(p, 0, n * 4));
    return p;
}

void init_state(const Model& m, State& s) {
    const auto& g = m.g;
    const int N = (int) g.n_embd, FF = (int) g.n_ff, K = (int) m.top_k;
    s.x = dalloc(N); s.h = dalloc(N); s.y = dalloc(N);
    s.qkv = dalloc(g.ssm_conv_channels); s.z = dalloc(g.ssm_value_dim); s.alpha = dalloc(g.ssm_v_heads);
    s.beta = dalloc(g.ssm_v_heads); s.conv_out = dalloc(g.ssm_conv_channels); s.gdn_out = dalloc(g.ssm_value_dim);
    s.qg = dalloc(2 * g.n_head * g.head_dim); s.k = dalloc(g.n_head_kv * g.head_dim); s.v = dalloc(g.n_head_kv * g.head_dim);
    s.q = dalloc(g.n_head * g.head_dim); s.gate = dalloc(g.n_head * g.head_dim); s.att = dalloc(g.n_head * g.head_dim);
    s.router = dalloc(g.n_expert);
    s.eg = dalloc((size_t) K * FF); s.eu = dalloc((size_t) K * FF); s.eh = dalloc((size_t) K * FF); s.ey = dalloc((size_t) K * N);
    s.sg = dalloc(g.n_ff_shexp); s.su = dalloc(g.n_ff_shexp); s.sh = dalloc(g.n_ff_shexp); s.sy = dalloc(N); s.sgate = dalloc(1);
    s.wts = dalloc(K);
    s.logits = dalloc(g.n_vocab);
    CK(cudaMalloc(&s.ids, K * sizeof(int)));
    g_act_buf = dalloc(kActBuf);
    CK(cudaStreamCreate(&g_st));   // blocking: ordered with this file's null-stream launches and copies
    CK(cudaMalloc(&g_q8, strata::kernels::native_q8_1_bytes((int) kActBuf, 1)));
    for (int64_t l = 0; l < g.n_layers; ++l) {
        if (strata::core::is_qsa_layer(g, l)) {
            s.conv_state.push_back(nullptr);
            s.S.push_back(nullptr);
            s.kc.push_back(dalloc((size_t) g.n_head_kv * s.ctx * g.head_dim));
            s.vc.push_back(dalloc((size_t) g.n_head_kv * s.ctx * g.head_dim));
        } else {
            s.conv_state.push_back(dalloc((size_t) (g.ssm_d_conv - 1) * g.ssm_conv_channels));
            s.S.push_back(dalloc((size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size));
            s.kc.push_back(nullptr);
            s.vc.push_back(nullptr);
        }
    }
}

std::string L(int64_t l, const char* s) { return "blk." + std::to_string(l) + "." + s; }

void embed(const Model& m, State& s, int tok) {
    const Tensor& e = m.W("token_embd.weight");
    const int N = (int) m.g.n_embd;
    switch (e.type) {
    case T_Q8_0: k_row<T_Q8_0><<<4, 128>>>(e.d, e.row_bytes, tok, N, s.x); break;
    case T_Q4_K: k_row<T_Q4_K><<<4, 128>>>(e.d, e.row_bytes, tok, N, s.x); break;
    case T_Q6_K: k_row<T_Q6_K><<<4, 128>>>(e.d, e.row_bytes, tok, N, s.x); break;
    case T_BF16: k_row<T_BF16><<<4, 128>>>(e.d, e.row_bytes, tok, N, s.x); break;
    case T_F16: k_row<T_F16><<<4, 128>>>(e.d, e.row_bytes, tok, N, s.x); break;
    default:
        std::fprintf(stderr, "q36ref: token_embd type %d has no row reader\n", e.type);
        std::exit(1);
    }
    CK(cudaGetLastError());
}

/// One token through the model at position s.pos; the layer outputs into `layers` (host, n_layers x N) when given.
void forward(const Model& m, State& s, int tok, float* layers) {
    const auto& g = m.g;
    const int N = (int) g.n_embd, K = (int) m.top_k, FF = (int) g.n_ff, FS = (int) g.n_ff_shexp;
    const int B = 256;
    embed(m, s, tok);
    for (int64_t l = 0; l < g.n_layers; ++l) {
        k_rms<<<1, kThreads>>>(s.x, m.F(L(l, "attn_norm.weight")), s.h, N, m.eps);
        if (!strata::core::is_qsa_layer(g, l)) {
            const int HK = (int) g.ssm_k_heads, HV = (int) g.ssm_v_heads, D = (int) g.ssm_state_size;
            const int C = (int) g.ssm_conv_channels;
            gemv(m.W(L(l, "attn_qkv.weight")), s.h, s.qkv);
            gemv(m.W(L(l, "attn_gate.weight")), s.h, s.z);
            gemv(m.W(L(l, "ssm_beta.weight")), s.h, s.beta);
            gemv(m.W(L(l, "ssm_alpha.weight")), s.h, s.alpha);
            if (g_strata && D == 128 && g.ssm_d_conv == 4) {
                // Strata's fused GDN (its own state layouts: conv history [channel][3], S [row][head][col])
                strata::kernels::fused_gdn_conv_l2(s.conv_state[l], s.qkv, m.F(L(l, "ssm_conv1d.weight")), s.conv_out,
                                                   C, 2 * HK, m.eps, g_st);
                k_gdn_gb<<<1, 64, 0, g_st>>>(s.alpha, m.F(L(l, "ssm_dt.bias")), m.F(L(l, "ssm_a")), s.beta, HV);
                strata::kernels::fused_gdn_step_norm_silu(s.S[l], s.conv_out, s.conv_out + HK * D,
                                                          s.conv_out + 2 * HK * D, s.alpha, s.beta, s.z,
                                                          m.F(L(l, "ssm_norm.weight")), m.eps, s.gdn_out, HK, HV, g_st);
            } else {
                k_conv<<<(C + B - 1) / B, B>>>(s.qkv, s.conv_state[l], m.F(L(l, "ssm_conv1d.weight")), s.conv_out, C,
                                               (int) g.ssm_d_conv);
                k_gdn<<<HV, D, (2 * D + D) * sizeof(float)>>>(s.conv_out, s.alpha, m.F(L(l, "ssm_dt.bias")),
                                                               m.F(L(l, "ssm_a")), s.beta, s.S[l], s.gdn_out, HK, HV, D,
                                                               m.eps);
                k_gated_norm<<<HV, D>>>(s.gdn_out, m.F(L(l, "ssm_norm.weight")), s.z, D, m.eps);
            }
            gemv(m.W(L(l, "ssm_out.weight")), s.gdn_out, s.y);
        } else {
            const int H = (int) g.n_head, HKV = (int) g.n_head_kv, D = (int) g.head_dim;
            gemv(m.W(L(l, "attn_q.weight")), s.h, s.qg);
            gemv(m.W(L(l, "attn_k.weight")), s.h, s.k);
            gemv(m.W(L(l, "attn_v.weight")), s.h, s.v);
            k_attn_prep<<<H + HKV, kThreads>>>(s.qg, s.k, s.v, m.F(L(l, "attn_q_norm.weight")),
                                               m.F(L(l, "attn_k_norm.weight")), s.q, s.gate, s.kc[l], s.vc[l], H, HKV, D,
                                               m.n_rot, s.pos, s.ctx, m.rope_base, m.eps);
            k_attn<<<H, kThreads, (size_t) (s.pos + 1) * sizeof(float)>>>(s.q, s.kc[l], s.vc[l], s.gate, s.att, H, HKV, D,
                                                                          s.pos + 1, s.ctx, 1.0f / sqrtf((float) D));
            gemv(m.W(L(l, "attn_output.weight")), s.att, s.y);
        }
        CK(cudaGetLastError());
        k_add<<<(N + B - 1) / B, B>>>(s.x, s.y, N);
        k_rms<<<1, kThreads>>>(s.x, m.F(L(l, "post_attention_norm.weight")), s.h, N, m.eps);
        // router: softmax over every expert, top-k, renormalised (host side: exact and simple)
        gemv(m.W(L(l, "ffn_gate_inp.weight")), s.h, s.router);
        if (g_strata && strata::kernels::native_router_topk_supported((int) g.n_expert, K)) {
            // Strata's router (llama.cpp's topk-moe contract) for 256 / 8; the ids come back for the per-expert launches
            strata::kernels::native_router_topk(s.router, s.ids, s.wts, (int) g.n_expert, K, 1, g_st);
            g_ids_host.resize((size_t) K);
            CK(cudaMemcpy(g_ids_host.data(), s.ids, K * sizeof(int), cudaMemcpyDeviceToHost));
        } else {
            std::vector<float> r((size_t) g.n_expert);
            CK(cudaMemcpy(r.data(), s.router, r.size() * 4, cudaMemcpyDeviceToHost));
            const float mx = *std::max_element(r.begin(), r.end());
            double sum = 0;
            for (float& v : r) sum += (v = std::exp(v - mx));
            for (float& v : r) v = (float) (v / sum);
            std::vector<int> idx((size_t) g.n_expert);
            for (int i = 0; i < (int) idx.size(); ++i) idx[i] = i;
            std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                              [&](int a, int b) { return r[a] > r[b] || (r[a] == r[b] && a < b); });
            std::vector<float> w(K);
            float ws = 0;
            for (int i = 0; i < K; ++i) ws += (w[i] = r[idx[i]]);
            for (float& v : w) v /= ws;
            g_ids_host.assign(idx.begin(), idx.begin() + K);
            CK(cudaMemcpy(s.ids, idx.data(), K * sizeof(int), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(s.wts, w.data(), K * 4, cudaMemcpyHostToDevice));
        }
        gemv(m.W(L(l, "ffn_gate_exps.weight")), s.h, s.eg, s.ids, K, 0);
        gemv(m.W(L(l, "ffn_up_exps.weight")), s.h, s.eu, s.ids, K, 0);
        k_swiglu<<<(K * FF + B - 1) / B, B>>>(s.eg, s.eu, s.eh, K * FF);
        gemv(m.W(L(l, "ffn_down_exps.weight")), s.eh, s.ey, s.ids, K, FF);
        // shared expert with its sigmoid scalar gate
        gemv(m.W(L(l, "ffn_gate_shexp.weight")), s.h, s.sg);
        gemv(m.W(L(l, "ffn_up_shexp.weight")), s.h, s.su);
        k_swiglu<<<(FS + B - 1) / B, B>>>(s.sg, s.su, s.sh, FS);
        gemv(m.W(L(l, "ffn_down_shexp.weight")), s.sh, s.sy);
        {
            Tensor gs;
            gs.d = (const uint8_t*) m.F(L(l, "ffn_gate_inp_shexp.weight"));
            gs.type = T_F32;
            gs.shape = {N, 1};
            gs.row_bytes = (size_t) N * 4;
            gemv(gs, s.h, s.sgate);
        }
        k_moe_sum<<<(N + B - 1) / B, B>>>(s.x, s.ey, s.wts, K, s.sy, s.sgate, N);
        CK(cudaGetLastError());
        if (layers) CK(cudaMemcpy(layers + (size_t) l * N, s.x, (size_t) N * 4, cudaMemcpyDeviceToHost));
    }
    k_rms<<<1, kThreads>>>(s.x, m.F("output_norm.weight"), s.h, N, m.eps);
    gemv(m.W("output.weight"), s.h, s.logits);
    CK(cudaDeviceSynchronize());
    ++s.pos;
}

std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> v;
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && (s[i] == ',' || s[i] == ' ' || s[i] == '\n' || s[i] == '\r')) ++i;
        if (i >= s.size()) break;
        v.push_back(std::atoi(s.c_str() + i));
        while (i < s.size() && s[i] != ',' && s[i] != ' ' && s[i] != '\n') ++i;
    }
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model, tokens, dump_layers, dump_logits, force, dump_full;
    int n_gen = 64, ctx = 4096;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "q36ref: %s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--model") model = next();
        else if (a == "--tokens") tokens = next();
        else if (a == "--tokens-file") {
            FILE* f = std::fopen(next().c_str(), "rb");
            if (!f) { std::fprintf(stderr, "q36ref: cannot read the tokens file\n"); return 2; }
            char buf[4096];
            size_t n;
            while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) tokens.append(buf, n);
            std::fclose(f);
        } else if (a == "--n") n_gen = std::atoi(next().c_str());
        else if (a == "--ctx") ctx = std::atoi(next().c_str());
        else if (a == "--dump-layers") dump_layers = next();
        else if (a == "--dump-logits") dump_logits = next();
        else if (a == "--force") force = next();
        else if (a == "--kernels") {
            const std::string v = next();
            if (v != "strata" && v != "plain") { std::fprintf(stderr, "q36ref: --kernels strata or plain\n"); return 2; }
            g_strata = v == "strata";
        }
        else if (a == "--dump-full-logits") dump_full = next();
        else if (a == "--act") {
            const std::string v = next();
            if (v != "q8" && v != "f32") { std::fprintf(stderr, "q36ref: --act q8 or f32\n"); return 2; }
            g_act_q8 = v == "q8";
        }
        else {
            std::fprintf(stderr, "usage: strata-q36ref --model GGUF --tokens ID,ID,... | --tokens-file F [--n 64] [--ctx 4096] "
                                 "[--dump-layers FILE] [--dump-logits FILE] [--act q8|f32] [--kernels strata|plain] "
                                 "[--force ID,ID,...] [--dump-full-logits FILE]\n");
            return 2;
        }
    }
    const std::vector<int> prompt = parse_ids(tokens);
    // --force: the continuation is these tokens (teacher forcing: the KL check reads both engines on one sequence);
    // the greedy ids are still printed
    const std::vector<int> forced = parse_ids(force);
    if (!forced.empty()) n_gen = (int) forced.size() + 1;
    if (model.empty() || prompt.empty()) {
        std::fprintf(stderr, "q36ref: --model and --tokens are required\n");
        return 2;
    }
    Model m;
    std::string err;
    try {
        if (!load(model, m, err)) {
            std::fprintf(stderr, "q36ref: %s\n", err.c_str());
            return 1;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "q36ref: %s\n", e.what());
        return 1;
    }
    State s;
    s.ctx = ctx;
    init_state(m, s);
    const int N = (int) m.g.n_embd, NL = (int) m.g.n_layers, V = (int) m.g.n_vocab;
    FILE* fl = dump_layers.empty() ? nullptr : std::fopen(dump_layers.c_str(), "wb");
    FILE* fg = dump_logits.empty() ? nullptr : std::fopen(dump_logits.c_str(), "wb");
    FILE* ff = dump_full.empty() ? nullptr : std::fopen(dump_full.c_str(), "wb");
    std::vector<float> layers((size_t) NL * N), logits((size_t) V);
    const auto t0 = std::chrono::steady_clock::now();
    int next = -1;
    std::vector<int> gen;
    const int total = (int) prompt.size() + n_gen - 1;
    if (total > ctx) {
        std::fprintf(stderr, "q36ref: %d positions do not fit --ctx %d\n", total, ctx);
        return 2;
    }
    for (int i = 0; i < total; ++i) {
        const int tok = i < (int) prompt.size() ? prompt[i] : next;
        forward(m, s, tok, fl ? layers.data() : nullptr);
        if (fl) std::fwrite(layers.data(), 4, layers.size(), fl);
        if (i + 1 >= (int) prompt.size()) {
            CK(cudaMemcpy(logits.data(), s.logits, (size_t) V * 4, cudaMemcpyDeviceToHost));
            int a = 0, b = -1;
            for (int j = 1; j < V; ++j)
                if (logits[j] > logits[a]) a = j;
            for (int j = 0; j < V; ++j)
                if (j != a && (b < 0 || logits[j] > logits[b])) b = j;
            next = gen.size() < forced.size() ? forced[gen.size()] : a;
            gen.push_back(a);
            if (ff) std::fwrite(logits.data(), 4, logits.size(), ff);
            if (fg) {
                const float rec[4] = {(float) a, logits[a], (float) b, logits[b]};
                std::fwrite(rec, 4, 4, fg);
            }
        }
    }
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (fl) std::fclose(fl);
    if (fg) std::fclose(fg);
    if (ff) std::fclose(ff);
    std::printf("gen:");
    for (int t : gen) std::printf(" %d", t);
    std::printf("\n");
    std::fprintf(stderr, "q36ref: %zu prompt + %zu generated positions in %.2f s (%.2f tok/s, one token at a time)\n",
                 prompt.size(), gen.size(), sec, (double) total / sec);
    return 0;
}
