// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_vision.cu - DeepSeek-V4.1's vision tower (see ds4_vision.cuh).
//
// One encode, N patches on a gh x gw grid:
//   patchify (the reference's (x / 255 - 0.5) / 0.5, BF16) -> x = patch_embed(patches)               FP32 [N][1024]
//   per block, its weights in slot (b % 2) while block b + 1's copy runs:
//     xn = norm1(x); qkv = rope(wqkv(xn) + bias)                       (rows in pieces of rb_: the FP32 GEMM output)
//     per query piece of qb_ rows, per head: S = Q K^T / 8 (FP32), P = softmax(S) (BF16), O[:, head] = P V;
//       x[piece] += wo(O) + bias
//     xn = norm2(x); per row piece: x += w2(silu(gate) * up), gate | up = w1(xn)
//   xn = norm(x); the 3 x 3 unfold; GELU(w1 + b1); w2 + b2 -> BF16 rows (the aligner's weights stream in pieces too)
#include "ds4_launch.h"   // strata_ds4_launch: <<<>>> on CUDA and HIP, a dispatch on Metal
#include "ds4_vision.cuh"

#include <cublas_v2.h>
#include "ds4_numerics.cuh"
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>

namespace strata::ds4 {
namespace {

#define VK(call)                                                                      \
    do {                                                                              \
        cudaError_t e_ = (call);                                                      \
        if (e_ != cudaSuccess) {                                                      \
            err = std::string("vision: ") + #call + ": " + cudaGetErrorString(e_);    \
            return false;                                                             \
        }                                                                             \
    } while (0)
#define VB(call)                                                                                      \
    do {                                                                                              \
        cublasStatus_t e_ = (call);                                                                   \
        if (e_ != CUBLAS_STATUS_SUCCESS) {                                                            \
            err = std::string("vision: ") + #call + ": cuBLAS status " + std::to_string((int) e_);   \
            return false;                                                                             \
        }                                                                                             \
    } while (0)

constexpr int kT = 256;                  // threads of the row kernels
constexpr int kSmT = 512, kSmPer = 20;   // softmax: a row of up to 10240 keys in registers
constexpr size_t kWs = 16u << 20;        // cuBLAS workspace

__device__ __forceinline__ float bf(uint16_t b) { return __uint_as_float((uint32_t) b << 16); }
__device__ __forceinline__ uint16_t bfr(float f) { return bf16_rn(f); }

__device__ float block_reduce(float v, bool is_max) {
    __shared__ float red[32];
    __shared__ float tot;
    for (int o = 16; o > 0; o >>= 1) {
        const float u = __shfl_xor_sync(0xffffffffu, v, o);
        v = is_max ? fmaxf(v, u) : v + u;
    }
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    __syncthreads();   // (an earlier call's readers of red / tot are done)
    if (lane == 0) red[w] = v;
    __syncthreads();
    if (w == 0) {
        v = lane < nw ? red[lane] : (is_max ? -FLT_MAX : 0.0f);
        for (int o = 16; o > 0; o >>= 1) {
            const float u = __shfl_xor_sync(0xffffffffu, v, o);
            v = is_max ? fmaxf(v, u) : v + u;
        }
        if (lane == 0) tot = v;
    }
    __syncthreads();
    return tot;
}

// patch i of the gh x gw grid: its 3 x p x p values, channel-major (the reference's reshape / permute)
__global__ void patchify_kernel(const uint8_t* __restrict__ rgb, int width, int gw, int p, uint16_t* __restrict__ out) {
    const int i = blockIdx.x, e = threadIdx.x, pp = p * p;
    if (e >= 3 * pp) return;
    const int c = e / pp, r = e % pp, y = (i / gw) * p + r / p, x = (i % gw) * p + r % p;
    const float u = (float) rgb[((size_t) y * width + x) * 3 + c];
    out[(size_t) i * 3 * pp + e] = bfr(__fdiv_rn(__fsub_rn(__fdiv_rn(u, 255.0f), 0.5f), 0.5f));
}

// x[r] += y[r] (y null: nothing) + bias
__global__ void add_rows_kernel(float* __restrict__ x, const float* __restrict__ y, int ldy,
                                const uint16_t* __restrict__ bias, int dim) {
    const int r = blockIdx.x;
    for (int e = threadIdx.x; e < dim; e += blockDim.x) {
        float v = x[(size_t) r * dim + e];
        if (y != nullptr) v += y[(size_t) r * ldy + e];
        if (bias != nullptr) v += bf(bias[e]);
        x[(size_t) r * dim + e] = v;
    }
}

__global__ void rmsnorm_kernel(const float* __restrict__ x, const uint16_t* __restrict__ w, int dim, float eps,
                               uint16_t* __restrict__ out) {
    const int r = blockIdx.x;
    const float* xr = x + (size_t) r * dim;
    float ss = 0.0f;
    for (int e = threadIdx.x; e < dim; e += blockDim.x) ss += xr[e] * xr[e];
    ss = block_reduce(ss, false);
    const float rs = rsqrtf(ss / (float) dim + eps);
    for (int e = threadIdx.x; e < dim; e += blockDim.x) out[(size_t) r * dim + e] = bfr(bf(w[e]) * (xr[e] * rs));
}

// a piece's rows of wqkv's output (+ bias, rounded to BF16 as the reference's linear) into qkv, q and k rotated: the
// head's first half pairs with its second; frequency j < hd / 4 turns with the patch's row, the rest with its column
__global__ void qkv_rope_kernel(const float* __restrict__ y, const uint16_t* __restrict__ bias, int r0, int gw,
                                const float* __restrict__ inv_freq, int heads, int hd, uint16_t* __restrict__ qkv) {
    const int r = blockIdx.x, gr = r0 + r, dim = heads * hd, half = hd / 2, nf = hd / 4;
    const int row = gr / gw, col = gr % gw;
    const float* yr = y + (size_t) r * 3 * dim;
    uint16_t* o = qkv + (size_t) gr * 3 * dim;
    for (int i = threadIdx.x; i < 2 * heads * half; i += blockDim.x) {
        const int base = (i / half) * hd, d = i % half;
        const float x1 = bf(bfr(yr[base + d] + bf(bias[base + d])));
        const float x2 = bf(bfr(yr[base + d + half] + bf(bias[base + d + half])));
        float sn, cs;
        sincosf((float) (d < nf ? row : col) * inv_freq[d % nf], &sn, &cs);
        o[base + d] = bfr(x1 * cs - x2 * sn);
        o[base + d + half] = bfr(x2 * cs + x1 * sn);
    }
    for (int e = 2 * dim + threadIdx.x; e < 3 * dim; e += blockDim.x) o[e] = bfr(yr[e] + bf(bias[e]));
}

__global__ void __launch_bounds__(kSmT) softmax_kernel(const float* __restrict__ s, int n, uint16_t* __restrict__ p) {
    const float* sr = s + (size_t) blockIdx.x * n;
    uint16_t* pr = p + (size_t) blockIdx.x * n;
    float v[kSmPer];
    float m = -FLT_MAX;
#pragma unroll
    for (int k = 0; k < kSmPer; ++k) {
        const int e = threadIdx.x + k * kSmT;
        v[k] = e < n ? sr[e] : -FLT_MAX;
        m = fmaxf(m, v[k]);
    }
    m = block_reduce(m, true);
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < kSmPer; ++k) {
        const int e = threadIdx.x + k * kSmT;
        v[k] = e < n ? expf(v[k] - m) : 0.0f;
        sum += v[k];
    }
    const float inv = 1.0f / block_reduce(sum, false);
#pragma unroll
    for (int k = 0; k < kSmPer; ++k) {
        const int e = threadIdx.x + k * kSmT;
        if (e < n) pr[e] = bfr(v[k] * inv);
    }
}

// mid = silu(gate) * up, gate | up the halves of w1's row (BF16 steps, as the reference's)
__global__ void silu_mul_kernel(const float* __restrict__ y, int inter, uint16_t* __restrict__ mid) {
    const int r = blockIdx.y, e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= inter) return;
    const float gt = bf(bfr(y[(size_t) r * 2 * inter + e])), up = bf(bfr(y[(size_t) r * 2 * inter + inter + e]));
    mid[(size_t) r * inter + e] = bfr(bf(bfr(gt / (1.0f + expf(-gt)))) * up);
}

// the aligner's input: block (bi, bj) of the m x m patches as one row, value c * m * m + kh * m + kw (F.unfold's
// order), zero past the grid
__global__ void unfold_kernel(const uint16_t* __restrict__ xn, int gh, int gw, int m, int dim, int lw,
                              uint16_t* __restrict__ un) {
    const int l = blockIdx.x, bi = l / lw, bj = l % lw, mm = m * m;
    for (int f = threadIdx.x; f < dim * mm; f += blockDim.x) {
        const int c = f / mm, k = f % mm, y = bi * m + k / m, x = bj * m + k % m;
        un[(size_t) l * dim * mm + f] = y < gh && x < gw ? xn[((size_t) y * gw + x) * dim + c] : (uint16_t) 0;
    }
}

// out = act(bf16(y + bias)): GELU (erf) or none
__global__ void bias_act_kernel(const float* __restrict__ y, const uint16_t* __restrict__ bias, int n, int gelu,
                                uint16_t* __restrict__ out) {
    const int r = blockIdx.y, e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    float v = bf(bfr(y[(size_t) r * n + e] + bf(bias[e])));
    if (gelu) v = 0.5f * v * (1.0f + erff(v * 0.70710678118654752f));
    out[(size_t) r * n + e] = bfr(v);
}

int cdiv(int a, int b) { return (a + b - 1) / b; }

// y[r][o] = alpha * sum_k x[r][k] W[o][k] (row-major W [out][in] with row stride ldw, x [rows][ldx], y [rows][ldy])
bool gemm(cublasHandle_t h, const uint16_t* W, int ldw, const uint16_t* x, int ldx, void* y, cudaDataType yt, int ldy,
          int rows, int in, int out, float alpha, std::string& err) {
    const float beta = 0.0f;
    VB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, out, rows, in, &alpha, W, CUDA_R_16BF, ldw, x, CUDA_R_16BF, ldx, &beta,
                    y, yt, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    return true;
}

}  // namespace

template <class T>
bool VisionTower::dalloc(T*& p, size_t n, std::string& err) {
    void* v = nullptr;
    const size_t b = (n * sizeof(T) + 255) / 256 * 256;
    if (cudaMalloc(&v, b) != cudaSuccess) {
        err = "vision: cudaMalloc " + std::to_string(b >> 20) + " MiB on CUDA" + std::to_string(at_) + " failed";
        return false;
    }
    dallocs_.push_back(v);
    dev_bytes_ += b;
    p = (T*) v;
    return true;
}

VisionTower::~VisionTower() { release(); }

void VisionTower::release() {
    if (at_ < 0) return;
    int cur = 0;
    cudaGetDevice(&cur);
    cudaSetDevice(at_);
    for (int i = 0; i < 2; ++i) {
        if (copied_[i]) cudaEventDestroy((cudaEvent_t) copied_[i]);
        if (used_[i]) cudaEventDestroy((cudaEvent_t) used_[i]);
        copied_[i] = used_[i] = nullptr;
    }
    if (blas_) cublasDestroy((cublasHandle_t) blas_);
    if (st_) cudaStreamDestroy((cudaStream_t) st_);
    if (cs_) cudaStreamDestroy((cudaStream_t) cs_);
    for (void* p : dallocs_) cudaFree(p);
    if (host_) cudaFreeHost(host_);
    cudaSetDevice(cur);
    blas_ = st_ = cs_ = nullptr;
    dallocs_.clear();
    host_ = nullptr;
    dev_ = at_ = -1;
    dev_bytes_ = 0;
}

bool VisionTower::load(const VisionGeo& g, int device, const Reader& read, std::string& err) {
    g_ = g;
    const int D = g.dim, I = g.inter, P = g.patch, M = g.merge, hd = g.heads > 0 ? g.dim / g.heads : 0;
    if (g.layers <= 0 || hd * g.heads != D || hd % 4 != 0 || P <= 0 || M <= 0 || g.max_tokens < 5) {
        err = "vision: geometry not covered";
        return false;
    }
    // the largest aligner grid within max_tokens (n_h * (n_w + 1) + 2): the work buffers' size
    int cells = 0;
    for (int h = 1; h <= g.max_tokens - 2; ++h) {
        const int w = (g.max_tokens - 2) / h - 1;
        if (w >= 1) cells = std::max(cells, h * w);
    }
    max_n_ = cells * M * M;
    if (max_n_ > kSmT * kSmPer) {
        err = "vision: " + std::to_string(max_n_) + " patches are more than the attention covers";
        return false;
    }
    cudaSetDevice(device);
    at_ = device;
    dev_bytes_ = 0;
    // the checkpoint's tensors, checked before anything is read
    struct Want {
        std::string name;
        std::vector<int64_t> shape;
    };
    std::vector<Want> want = {{"vision.patch_embed.proj.weight", {D, 3 * P * P}},
                              {"vision.patch_embed.proj.bias", {D}},
                              {"vision.norm.weight", {D}},
                              {"aligner.w1.weight", {g.out, (int64_t) D * M * M}},
                              {"aligner.w1.bias", {g.out}},
                              {"aligner.w2.weight", {g.out, g.out}},
                              {"aligner.w2.bias", {g.out}}};
    for (int l = 0; l < g.layers; ++l) {
        const std::string B = "vision.blocks." + std::to_string(l) + ".";
        want.push_back({B + "norm1.weight", {D}});
        want.push_back({B + "norm2.weight", {D}});
        want.push_back({B + "attn.wqkv.weight", {3 * D, D}});
        want.push_back({B + "attn.wqkv.bias", {3 * D}});
        want.push_back({B + "attn.wo.weight", {D, D}});
        want.push_back({B + "attn.wo.bias", {D}});
        want.push_back({B + "mlp.w1.weight", {2 * I, D}});
        want.push_back({B + "mlp.w2.weight", {D, I}});
    }
    for (const Want& w : want) {
        std::vector<int64_t> s;
        if (!read(w.name, s, nullptr)) {
            err = "vision: " + w.name + " is not in the checkpoint";
            return false;
        }
        if (s != w.shape) {
            err = "vision: " + w.name + " has another shape than the config's";
            return false;
        }
    }
    // the streamed weights, pinned
    block_n_ = (size_t) 3 * D * D + (size_t) D * D + (size_t) 2 * I * D + (size_t) D * I;
    al1_off_ = block_n_ * g.layers;
    al2_off_ = al1_off_ + (size_t) g.out * D * M * M;
    const size_t host_n = al2_off_ + (size_t) g.out * g.out;
    if (cudaHostAlloc((void**) &host_, host_n * 2, cudaHostAllocDefault) != cudaSuccess) {
        host_ = nullptr;
        err = "vision: pinning " + std::to_string((host_n * 2) >> 20) + " MiB of host memory failed";
        return false;
    }
    std::vector<int64_t> s;
    for (int l = 0; l < g.layers; ++l) {
        const std::string B = "vision.blocks." + std::to_string(l) + ".";
        uint16_t* b = host_ + (size_t) l * block_n_;
        if (!read(B + "attn.wqkv.weight", s, b) || !read(B + "attn.wo.weight", s, b + (size_t) 3 * D * D) ||
            !read(B + "mlp.w1.weight", s, b + (size_t) 4 * D * D) ||
            !read(B + "mlp.w2.weight", s, b + (size_t) 4 * D * D + (size_t) 2 * I * D)) {
            err = "vision: block " + std::to_string(l) + ": read failed";
            return false;
        }
    }
    if (!read("aligner.w1.weight", s, host_ + al1_off_) || !read("aligner.w2.weight", s, host_ + al2_off_)) {
        err = "vision: the aligner: read failed";
        return false;
    }
    // the small ones on the device
    auto up = [&](const std::string& name, size_t n, uint16_t*& d) -> bool {
        std::vector<uint16_t> h(n);
        std::vector<int64_t> sh;
        if (!read(name, sh, h.data())) {
            err = "vision: " + name + ": read failed";
            return false;
        }
        if (!dalloc(d, n, err)) return false;
        VK(cudaMemcpy(d, h.data(), n * 2, cudaMemcpyHostToDevice));
        return true;
    };
    bool ok = up("vision.patch_embed.proj.weight", (size_t) D * 3 * P * P, pe_w_) &&
              up("vision.patch_embed.proj.bias", D, pe_b_) && up("vision.norm.weight", D, norm_) &&
              up("aligner.w1.bias", g.out, al1_b_) && up("aligner.w2.bias", g.out, al2_b_);
    n1_.assign(g.layers, nullptr);
    n2_.assign(g.layers, nullptr);
    bqkv_.assign(g.layers, nullptr);
    bo_.assign(g.layers, nullptr);
    for (int l = 0; l < g.layers && ok; ++l) {
        const std::string B = "vision.blocks." + std::to_string(l) + ".";
        ok = up(B + "norm1.weight", D, n1_[l]) && up(B + "norm2.weight", D, n2_[l]) &&
             up(B + "attn.wqkv.bias", 3 * D, bqkv_[l]) && up(B + "attn.wo.bias", D, bo_[l]);
    }
    if (!ok) return false;
    {   // the rotary's frequencies: theta ^ -(2i / (hd / 2)), hd / 4 of them
        const int nf = hd / 4;
        std::vector<float> f(nf);
        for (int i = 0; i < nf; ++i) f[i] = 1.0f / std::pow(g.theta, (float) (2 * i) / (float) (hd / 2));
        if (!dalloc(inv_freq_, nf, err)) return false;
        VK(cudaMemcpy(inv_freq_, f.data(), nf * 4, cudaMemcpyHostToDevice));
    }
    // the work buffers: the patch grid's rows whole (the residual, its norm, q / k / v), the GEMM outputs a piece of
    // rows at a time, the attention's scores a piece of queries at a time (~48 MB)
    const int L = cells;
    rb_ = 2048;
    qb_ = std::max(16, (int) ((48u << 20) / ((size_t) max_n_ * 4)) / 16 * 16);
    slot_n_ = std::max(block_n_, (size_t) 1 << 22);
    const size_t y_n = (size_t) rb_ * std::max({3 * D, 2 * I, D});
    const size_t qkv_n = std::max({(size_t) max_n_ * 3 * D,
                                   (size_t) max_n_ * 3 * P * P + ((size_t) max_n_ * 3 * P * P + 1) / 2,   // patches, pixels
                                   (size_t) L * D * M * M + (size_t) 2 * L * g.out});                      // the aligner's
    if ((size_t) L * g.out > y_n || (size_t) max_n_ * D > y_n) {
        err = "vision: work buffer sizes";
        return false;
    }
    ok = dalloc(x_, (size_t) max_n_ * D, err) && dalloc(xn_, (size_t) max_n_ * D, err) && dalloc(qkv_, qkv_n, err) &&
         dalloc(y_, y_n, err) && dalloc(o_, (size_t) max_n_ * D, err) && dalloc(mid_, (size_t) rb_ * I, err) &&
         dalloc(s_, (size_t) qb_ * max_n_, err) && dalloc(p_, (size_t) qb_ * max_n_, err) &&
         dalloc(slot_[0], slot_n_, err) && dalloc(slot_[1], slot_n_, err) && dalloc(ws_, kWs, err);
    if (!ok) return false;
    patch_ = qkv_;   // (before block 0's qkv)
    rgb_ = (uint8_t*) (qkv_ + (size_t) max_n_ * 3 * P * P);
    un_ = qkv_;      // (after the last block's attention)
    a16_ = un_ + (size_t) L * D * M * M;
    outb_ = a16_ + (size_t) L * g.out;
    cublasHandle_t h = nullptr;
    VB(cublasCreate(&h));
    blas_ = h;
    cudaStream_t st = nullptr, cs = nullptr;
    VK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    VK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
    st_ = st;
    cs_ = cs;
    VB(cublasSetStream(h, st));
    VB(cublasSetWorkspace(h, ws_, kWs));
    for (int i = 0; i < 2; ++i) {
        cudaEvent_t a = nullptr, b = nullptr;
        VK(cudaEventCreateWithFlags(&a, cudaEventDisableTiming));
        VK(cudaEventCreateWithFlags(&b, cudaEventDisableTiming));
        copied_[i] = a;
        used_[i] = b;
    }
    dev_ = device;
    return true;
}

bool VisionTower::encode(const uint8_t* rgb, int width, int height, std::vector<uint16_t>& rows, int& n_h, int& n_w,
                         std::string& err) {
    if (dev_ < 0) {
        err = "vision: not loaded";
        return false;
    }
    const VisionGeo& g = g_;
    const int D = g.dim, I = g.inter, P = g.patch, M = g.merge, hd = D / g.heads, H3 = 3 * D;
    if (width <= 0 || height <= 0 || width % P != 0 || height % P != 0) {
        err = "vision: an image of " + std::to_string(width) + " x " + std::to_string(height) +
              " pixels (both sides must be multiples of " + std::to_string(P) + ")";
        return false;
    }
    const int gh = height / P, gw = width / P, N = gh * gw;
    n_h = cdiv(gh, M);
    n_w = cdiv(gw, M);
    const int L = n_h * n_w;
    if (N > max_n_ || n_h * (n_w + 1) + 2 > g.max_tokens) {
        err = "vision: an image of " + std::to_string(width) + " x " + std::to_string(height) + " pixels is more than " +
              std::to_string(g.max_tokens) + " image tokens";
        return false;
    }
    cudaSetDevice(dev_);
    cublasHandle_t h = (cublasHandle_t) blas_;
    cudaStream_t s = (cudaStream_t) st_, cs = (cudaStream_t) cs_;
    // the weights in the order the encode takes them: the blocks, then the aligner's in pieces of a slot
    std::vector<Piece> pcs;
    for (int l = 0; l < g.layers; ++l) pcs.push_back({host_ + (size_t) l * block_n_, block_n_});
    const int in1 = D * M * M, r1 = (int) std::min<size_t>(g.out, slot_n_ / in1), r2 = (int) std::min<size_t>(g.out, slot_n_ / g.out);
    for (int o0 = 0; o0 < g.out; o0 += r1)
        pcs.push_back({host_ + al1_off_ + (size_t) o0 * in1, (size_t) std::min(r1, g.out - o0) * in1});
    for (int o0 = 0; o0 < g.out; o0 += r2)
        pcs.push_back({host_ + al2_off_ + (size_t) o0 * g.out, (size_t) std::min(r2, g.out - o0) * g.out});
    auto fetch = [&](size_t i) -> bool {
        const int k = (int) (i % 2);
        VK(cudaStreamWaitEvent(cs, (cudaEvent_t) used_[k], 0));
        VK(cudaMemcpyAsync(slot_[k], pcs[i].host, pcs[i].n * 2, cudaMemcpyHostToDevice, cs));
        VK(cudaEventRecord((cudaEvent_t) copied_[k], cs));
        return true;
    };
    size_t pi = 0;   // the piece being used
    auto take = [&]() -> const uint16_t* {   // the next piece's slot, once its copy is in (the one after it started)
        const int k = (int) (pi % 2);
        if (pi + 1 < pcs.size() && !fetch(pi + 1)) return nullptr;
        if (cudaStreamWaitEvent(s, (cudaEvent_t) copied_[k], 0) != cudaSuccess) {
            err = "vision: waiting for a weight copy failed";
            return nullptr;
        }
        return slot_[k];
    };
    auto done = [&]() {   // the piece's work is queued: its slot is free once that ran
        cudaEventRecord((cudaEvent_t) used_[pi % 2], s);
        ++pi;
    };
    if (!fetch(0)) return false;

    // the patches
    VK(cudaMemcpyAsync(rgb_, rgb, (size_t) N * 3 * P * P, cudaMemcpyHostToDevice, s));
    strata_ds4_launch(patchify_kernel, N, 3 * P * P, 0, s, rgb_, width, gw, P, patch_);
    if (!gemm(h, pe_w_, 3 * P * P, patch_, 3 * P * P, x_, CUDA_R_32F, D, N, 3 * P * P, D, 1.0f, err)) return false;
    strata_ds4_launch(add_rows_kernel, N, kT, 0, s, x_, nullptr, 0, pe_b_, D);

    const float scale = 1.0f / std::sqrt((float) hd);
    const int qb = (int) std::max<size_t>(1, std::min<size_t>((size_t) N, (size_t) qb_ * max_n_ / (size_t) N));
    for (int l = 0; l < g.layers; ++l) {
        const uint16_t* wqkv = take();
        if (wqkv == nullptr) return false;
        const uint16_t *wo = wqkv + (size_t) H3 * D, *w1 = wo + (size_t) D * D, *w2 = w1 + (size_t) 2 * I * D;
        strata_ds4_launch(rmsnorm_kernel, N, kT, 0, s, x_, n1_[l], D, g.eps, xn_);
        for (int r0 = 0; r0 < N; r0 += rb_) {
            const int nr = std::min(rb_, N - r0);
            if (!gemm(h, wqkv, D, xn_ + (size_t) r0 * D, D, y_, CUDA_R_32F, H3, nr, D, H3, 1.0f, err)) return false;
            strata_ds4_launch(qkv_rope_kernel, nr, kT, 0, s, y_, bqkv_[l], r0, gw, inv_freq_, g.heads, hd, qkv_);
        }
        for (int q0 = 0; q0 < N; q0 += qb) {
            const int nq = std::min(qb, N - q0);
            const float one = 1.0f, zero = 0.0f;
            for (int hh = 0; hh < g.heads; ++hh) {
                // S^T (keys x queries, column-major) = K Q^T: row-major S [query][key]
                VB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, nq, hd, &scale, qkv_ + D + hh * hd, CUDA_R_16BF, H3,
                                qkv_ + (size_t) q0 * H3 + hh * hd, CUDA_R_16BF, H3, &zero, s_, CUDA_R_32F, N,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                strata_ds4_launch(softmax_kernel, nq, kSmT, 0, s, s_, N, p_);
                // O^T (hd x queries) = V^T P^T: row-major O [query][head's hd] at column hh * hd
                VB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, hd, nq, N, &one, qkv_ + 2 * D + hh * hd, CUDA_R_16BF, H3,
                                p_, CUDA_R_16BF, N, &zero, o_ + hh * hd, CUDA_R_16BF, D, CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT));
            }
            if (!gemm(h, wo, D, o_, D, y_, CUDA_R_32F, D, nq, D, D, 1.0f, err)) return false;
            strata_ds4_launch(add_rows_kernel, nq, kT, 0, s, x_ + (size_t) q0 * D, y_, D, bo_[l], D);
        }
        strata_ds4_launch(rmsnorm_kernel, N, kT, 0, s, x_, n2_[l], D, g.eps, xn_);
        for (int r0 = 0; r0 < N; r0 += rb_) {
            const int nr = std::min(rb_, N - r0);
            if (!gemm(h, w1, D, xn_ + (size_t) r0 * D, D, y_, CUDA_R_32F, 2 * I, nr, D, 2 * I, 1.0f, err)) return false;
            strata_ds4_launch(silu_mul_kernel, dim3(cdiv(I, kT), nr), kT, 0, s, y_, I, mid_);
            if (!gemm(h, w2, I, mid_, I, y_, CUDA_R_32F, D, nr, I, D, 1.0f, err)) return false;
            strata_ds4_launch(add_rows_kernel, nr, kT, 0, s, x_ + (size_t) r0 * D, y_, D, nullptr, D);
        }
        done();
    }
    strata_ds4_launch(rmsnorm_kernel, N, kT, 0, s, x_, norm_, D, g.eps, xn_);

    // the aligner
    strata_ds4_launch(unfold_kernel, L, kT, 0, s, xn_, gh, gw, M, D, n_w, un_);
    for (int o0 = 0; o0 < g.out; o0 += r1) {
        const uint16_t* w = take();
        if (w == nullptr) return false;
        if (!gemm(h, w, in1, un_, in1, y_ + o0, CUDA_R_32F, g.out, L, in1, std::min(r1, g.out - o0), 1.0f, err))
            return false;
        done();
    }
    strata_ds4_launch(bias_act_kernel, dim3(cdiv(g.out, kT), L), kT, 0, s, y_, al1_b_, g.out, 1, a16_);
    for (int o0 = 0; o0 < g.out; o0 += r2) {
        const uint16_t* w = take();
        if (w == nullptr) return false;
        if (!gemm(h, w, g.out, a16_, g.out, y_ + o0, CUDA_R_32F, g.out, L, g.out, std::min(r2, g.out - o0), 1.0f, err))
            return false;
        done();
    }
    strata_ds4_launch(bias_act_kernel, dim3(cdiv(g.out, kT), L), kT, 0, s, y_, al2_b_, g.out, 0, outb_);
    rows.resize((size_t) L * g.out);
    VK(cudaMemcpyAsync(rows.data(), outb_, rows.size() * 2, cudaMemcpyDeviceToHost, s));
    VK(cudaStreamSynchronize(s));
    VK(cudaGetLastError());
    return true;
}

}  // namespace strata::ds4
