// q36_kernels.cuh - decode kernels of the Qwen3.6 engine (q36_engine.cu), written to give the SAME BITS as the shared
// kernels they replace (native_small_mmvq_kernel, k_moe_out, k_add_rms, k_group, native_quantize_q8_1) with a layout that
// keeps more bytes in flight and fewer launches.  Every kernel here is checked bit for bit against the old one on the
// real activations (STRATA_Q36_VERIFY=1, --no-graph).
#pragma once

#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/q8_1_finite.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace q36k {

struct QB {   // a q8_1 activation block (what native_quantize_q8_1 writes)
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(QB) == 36, "q8_1 block");

__device__ __forceinline__ float wsum(float x) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o, 32);
    return x;
}
__device__ __forceinline__ float wmax(float x) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, o, 32));
    return x;
}
__device__ __forceinline__ float sigm(float v) { return 1.0f / (1.0f + expf(-v)); }

// The shared CUDA kernels these ones replace (native_mmvq.cu, native_router.cu) are compiled with --use_fast_math (CMakeLists.txt);
// this file is not, so on CUDA the division and the exponential they use are spelled out as the instructions that flag selects
// (div.approx.ftz, rcp.approx.ftz, ex2.approx.ftz via __expf).  HIP builds have no such flag: plain operators.
#if defined(__CUDACC__) && !defined(__HIPCC__)
__device__ __forceinline__ float fdiv_fm(float a, float b) {
    float r;
    asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ float frcp_fm(float a) {
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}
__device__ __forceinline__ float fexp_fm(float a) { return __expf(a); }
#else
__device__ __forceinline__ float fdiv_fm(float a, float b) { return a / b; }
__device__ __forceinline__ float frcp_fm(float a) { return 1.0f / a; }
__device__ __forceinline__ float fexp_fm(float a) { return expf(a); }
#endif

/// One lane's element `v` of a 32-element q8_1 block (the warp is the block): native_quantize_q8_1_kernel's body.
/// `y` points at the row's blocks, `i` is the element index within the row.
__device__ __forceinline__ void q8_store(QB* __restrict__ y, int i, float v, int lane) {
    const float amax = wmax(fabsf(v));
    const float sum = wsum(v);
    const float d = strata::kernels::q8_1_finite(fdiv_fm(amax, 127.0f));
    int8_t q = 0;   // strata::kernels::q8_1_quant with the division of the fast-math TU
    if (amax != 0.0f) {
        const float qf = roundf(fdiv_fm(v, d));
        q = (int8_t) (qf > 127.0f ? 127.0f : qf < -127.0f ? -127.0f : qf);
    }
    y[i >> 5].qs[lane] = q;
    if (lane == 0) y[i >> 5].ds = strata::kernels::q8_1_ds(d, sum);
}

/// A Q8_0 row (GGUF blocks of 34 bytes, or the qs / d planes with PK) against NC activation columns, with the arithmetic of
/// native_small_mmvq_kernel / native_mmvq_multi_kernel's exact layout: the 4 warps of the old block are the 4 partial sums
/// P_w of one lane here (block kbx = 8 w + lane / 4 + 32 j, ints 2 (lane % 4) and +1), added in warp order, then the xor
/// tree.  One warp per row, every load of a lane issued before the first dot (NJ = ceil(nb / 32) terms per partial).
/// res[c] = the row's value for column c in every lane.
template <int NC, int NJ, bool PK>
__device__ __forceinline__ void q8_row_dot(const uint8_t* __restrict__ w, const half* __restrict__ dpl, const QB* __restrict__ x,
                                           int row, int nb, int lane, float (&res)[NC]) {
    const int kqs = 2 * (lane & 3);
    const int kb0 = lane >> 2;
    int v0[4][NJ], v1[4][NJ];
    float d0[4][NJ];
    bool ok[4][NJ];
#pragma unroll
    for (int p = 0; p < 4; ++p)
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            const int kbx = kb0 + 8 * p + 32 * j;
            ok[p][j] = kbx < nb;
            if (!ok[p][j]) continue;
            const size_t blk = (size_t) row * nb + kbx;
            if (PK) {
                const int* q = reinterpret_cast<const int*>(w + blk * 32) + kqs;
                v0[p][j] = q[0];
                v1[p][j] = q[1];
                d0[p][j] = __half2float(dpl[blk]);
            } else {
                const uint16_t* b = reinterpret_cast<const uint16_t*>(w + blk * 34);
                d0[p][j] = __half2float(__ushort_as_half(b[0]));
                v0[p][j] = (int) b[1 + 2 * kqs] | ((int) b[2 + 2 * kqs] << 16);
                v1[p][j] = (int) b[3 + 2 * kqs] | ((int) b[4 + 2 * kqs] << 16);
            }
        }
    float P[NC][4];
#pragma unroll
    for (int c = 0; c < NC; ++c)
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            float t = 0.f;
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                if (!ok[p][j]) continue;
                const QB* xb = x + (size_t) c * nb + kb0 + 8 * p + 32 * j;
                const int* u = reinterpret_cast<const int*>(xb->qs);
                int sumi = 0;
                sumi = STRATA_DP4A(v0[p][j], u[kqs], sumi);
                sumi = STRATA_DP4A(v1[p][j], u[kqs + 1], sumi);
                const float d1 = __low2float(xb->ds);
                t += d0[p][j] * d1 * float(sumi);
            }
            P[c][p] = t;
        }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        float t = P[c][0];
        t += P[c][1];
        t += P[c][2];
        t += P[c][3];
        res[c] = wsum(t);
    }
}

/// y[c][row] = W[row] . x[c], NC columns, 8 rows (warps) per 256-thread block.
template <int NC, int NJ, bool PK>
__global__ void __launch_bounds__(256) k_q8_rows(const uint8_t* __restrict__ w, const half* __restrict__ dpl,
                                                 const QB* __restrict__ x, float* __restrict__ y, int nb, int n_out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int stride = gridDim.x * (blockDim.x >> 5);   // a grid-stride loop over the rows (see launch_q8_rows)
    for (int row = blockIdx.x * (blockDim.x >> 5) + warp; row < n_out; row += stride) {
        float res[NC];
        q8_row_dot<NC, NJ, PK>(w, dpl, x, row, nb, lane, res);
        if (lane == 0)
#pragma unroll
            for (int c = 0; c < NC; ++c) y[(size_t) c * n_out + row] = res[c];
    }
}

template <int NC, int NJ, bool PK>
inline void launch_q8_rows_t(const void* w, const half* dpl, const void* x, float* y, int nb, int n_out, int warps, int max_warps,
                           cudaStream_t s) {
    const int rows = n_out < max_warps ? n_out : max_warps;
    k_q8_rows<NC, NJ, PK><<<(unsigned) ((rows + warps - 1) / warps), warps * 32, 0, s>>>(static_cast<const uint8_t*>(w), dpl,
                                                                                    static_cast<const QB*>(x), y, nb, n_out);
}

template <int NC, int NJ, bool PK>
/// At most 1024 warps in the grid, a grid-stride loop over the rows beyond: a 2048-row product took 21 us on gfx1201 as 2048 warps and
/// 5.2 us as 1024 (a grid above 32768 threads is several times slower there; the same rows, the same sums).
inline void launch_q8_rows(const void* w, const half* dpl, const void* x, float* y, int nb, int n_out, cudaStream_t s) {
    launch_q8_rows_t<NC, NJ, PK>(w, dpl, x, y, nb, n_out, 8, 1024, s);
}

/// The shared expert's gate and up products and SwiGLU + q8_1 in one launch: 32 rows (one q8_1 block of h) per 1024-thread
/// block, warp w computing gate row and up row 32 b + w, then warp 0 applying native_swiglu_quantize_q8_1_kernel's body to
/// the 32 pairs.  Column c of the h image is at hq + c * (n_out / 32).  n_out % 32 == 0.
template <int NC, int NJ, bool PK>
__global__ void __launch_bounds__(1024) k_sh_gu(const uint8_t* __restrict__ wg, const half* __restrict__ dg, const uint8_t* __restrict__ wu,
                                                const half* __restrict__ du, const QB* __restrict__ x, QB* __restrict__ hq,
                                                int nb, int n_out) {
    __shared__ float g[NC][32], u[NC][32];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * 32 + warp;
    float rg[NC], ru[NC];
    q8_row_dot<NC, NJ, PK>(wg, dg, x, row, nb, lane, rg);
    q8_row_dot<NC, NJ, PK>(wu, du, x, row, nb, lane, ru);
    if (lane == 0)
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            g[c][warp] = rg[c];
            u[c][warp] = ru[c];
        }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const float gi = g[c][lane];
            const float xi = __fmul_rn(__fdividef(gi, __fadd_rn(1.0f, __expf(-gi))), u[c][lane]);
            q8_store(hq + (size_t) c * (n_out >> 5), blockIdx.x * 32 + lane, xi, lane);
        }
    }
}

/// k_add_rms (x += y; h = rms(x) * w, one block of 256 per row) and the q8_1 image of h in the same launch.
__global__ void __launch_bounds__(256) k_add_rms_q(float* x, const float* y, const float* w, float* h, QB* hq, int n, float eps) {
    __shared__ float sh[256];
    float* xr = x + (size_t) blockIdx.x * n;
    const float* yr = y + (size_t) blockIdx.x * n;
    float* hr = h + (size_t) blockIdx.x * n;
    QB* qr = hq + (size_t) blockIdx.x * (n >> 5);
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += 256) {
        const float v = xr[i] + yr[i];
        xr[i] = v;
        s += v * v;
    }
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
        __syncthreads();
    }
    const float r = rsqrtf(sh[0] / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += 256) {
        const float hv = __fmul_rn(__fmul_rn(xr[i], r), w[i]);   // no FMA with the quantizer's sums: the old h was stored first
        hr[i] = hv;
        q8_store(qr, i, ((volatile float*) hr)[i], threadIdx.x & 31);   // the stored value, as the standalone quantizer reads it
    }
}

/// k_moe_out for T <= 8 rows (decode, verify windows) with n <= 2048 and n % 256 == 0: 1024 threads per row instead of 256
/// (the elements' loads in parallel), the squares summed per old thread (element t, t + 256, ... in order) and the
/// 256-wide tree as before: the same bits.  x += sum_k w_k part_k (+ gate * shared); h = rms(x) * wn; with hq the q8_1
/// image of h as well (the next layer's input).
__global__ void __launch_bounds__(1024) k_moe_out_small(float* x, const float* parts, const int32_t* prow, const float* wts,
                                                        int K_, const float* shared, const float* sgate, int sg_stride,
                                                        const float* wn, float* h, int n, float eps, QB* hq = nullptr) {
    __shared__ float vv[2048];
    __shared__ float sh[256];
    __shared__ const float* pk[16];
    const int t = blockIdx.x;
    float* xr = x + (size_t) t * n;
    if (threadIdx.x < K_) pk[threadIdx.x] = parts + (size_t) (prow ? prow[t * K_ + threadIdx.x] : t * K_ + threadIdx.x) * n;
    __syncthreads();
    const float* wr = wts + (size_t) t * K_;
    const float gs = sigm(sgate[(size_t) t * sg_stride]);
    for (int i = threadIdx.x; i < n; i += 1024) {
        float acc = wr[0] * pk[0][i];
        for (int k = 1; k < K_; ++k) acc = fmaf(wr[k], pk[k][i], acc);
        acc += gs * shared[(size_t) t * n + i];
        const float v = xr[i] + acc;
        xr[i] = v;
        vv[i] = v;
    }
    if (wn == nullptr) return;
    __syncthreads();
    if (threadIdx.x < 256) {
        float s = 0.f;
        for (int i = threadIdx.x; i < n; i += 256) s += vv[i] * vv[i];
        sh[threadIdx.x] = s;
    }
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float r = rsqrtf(sh[0] / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += 1024) {
        const float hv = __fmul_rn(__fmul_rn(vv[i], r), wn[i]);
        h[(size_t) t * n + i] = hv;
        if (hq) q8_store(hq + (size_t) t * (n >> 5), i, ((volatile float*) h)[(size_t) t * n + i], threadIdx.x & 31);
    }
}

/// k_group / k_group_hy for n_ent <= 64 entries: one block of 64 threads ranks each entry by (expert id, entry) in
/// parallel instead of one thread walking all the experts; the same group / entry arrays.  HY: only the entries whose
/// expert has a VRAM slot (slot_of >= 0), the group's blob at blobs + slot * blob.
template <bool HY>
__global__ void k_group_small(const int32_t* ids, int n_ent, int K_, const int32_t* slot_of, const uint8_t* blobs,
                              unsigned long long blob, unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups,
                              int32_t* ent_dst, int32_t* ent_tok) {
    __shared__ int sid[64];
    const int e = threadIdx.x;
    const int id = e < n_ent ? ids[e] : 0;
    const bool valid = e < n_ent && (!HY || slot_of[id] >= 0);
    int rank = 0;
    if (valid) {
        for (int e2 = 0; e2 < n_ent; ++e2) {
            const int id2 = ids[e2];
            if (HY && slot_of[id2] < 0) continue;
            if (id2 < id || (id2 == id && e2 < e)) ++rank;
        }
        ent_dst[rank] = e;
        ent_tok[rank] = e / K_;
        sid[rank] = id;
    }
    __syncthreads();
    int nv = 0;
    for (int e2 = 0; e2 < n_ent; ++e2) nv += (!HY || slot_of[ids[e2]] >= 0) ? 1 : 0;
    if (valid && (rank == 0 || sid[rank - 1] != sid[rank])) {
        int g = 0;
        for (int q = 1; q < rank; ++q) g += sid[q - 1] != sid[q] ? 1 : 0;
        g += rank > 0 ? 1 : 0;
        grp_ptr[g] = (unsigned long long) (blobs + (size_t) (HY ? slot_of[id] : id) * blob);
        grp_start[g] = rank;
    }
    if (e == 0) {
        int G = nv > 0 ? 1 : 0;
        for (int q = 1; q < nv; ++q) G += sid[q - 1] != sid[q] ? 1 : 0;
        grp_start[G] = nv;
        *n_groups = G;
    }
}

/// route_topk<256, 8> (native_router.cu, copied: the same expressions) for n_tok <= 8 tokens (one warp each) and, in the same
/// launch, k_group_small's grouping of the picks.  block (32, 8).  HY as k_group_small.
template <bool HY>
__global__ void __launch_bounds__(256) k_route_group(const float* __restrict__ logits, int32_t* __restrict__ ids,
                                                     float* __restrict__ weights, int n_tok, int ld, const int32_t* slot_of,
                                                     const uint8_t* blobs, unsigned long long blob,
                                                     unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups,
                                                     int32_t* ent_dst, int32_t* ent_tok) {
    constexpr int NE = 256, K = 8, V = NE / 32;
    __shared__ int sids[64], sid[64];
    const int tk = threadIdx.y;
    const int lane = threadIdx.x;
    if (tk < n_tok) {
        const float* lg = logits + (size_t) tk * ld;
        float values[V];
#pragma unroll
        for (int i = 0; i < V; ++i) values[i] = lg[lane + i * 32];
        float maximum = -INFINITY;
#pragma unroll
        for (int i = 0; i < V; ++i) maximum = max(maximum, values[i]);
        maximum = wmax(maximum);
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < V; ++i) {
            values[i] = fexp_fm(values[i] - maximum);
            sum += values[i];
        }
        const float reciprocal = frcp_fm(wsum(sum));
#pragma unroll
        for (int i = 0; i < V; ++i) {
            values[i] *= reciprocal;
            if (__isnanf(values[i])) values[i] = -FLT_MAX;
        }
        float selected = 0.0f, selected_sum = 0.0f;
        for (int rank = 0; rank < K; ++rank) {
            float best = values[0];
            int expert = lane;
#pragma unroll
            for (int i = 1; i < V; ++i) {
                if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
            }
#pragma unroll
            for (int mask = 16; mask; mask >>= 1) {
                const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
                const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
                if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
            }
            if ((expert & 31) == lane) {
                values[expert / 32] = -INFINITY;
                ids[(size_t) tk * K + rank] = expert;
                sids[tk * K + rank] = expert;
                selected_sum += best;
            }
            if (rank == lane) selected = best;
        }
        selected_sum = max(wsum(selected_sum), 6.103515625e-5f);
        const float inverse_selected_sum = frcp_fm(selected_sum);
        if (lane < K) weights[(size_t) tk * K + lane] = selected * inverse_selected_sum;
    }
    __syncthreads();
    const int n_ent = n_tok * K;
    const int e = tk * 32 + lane;   // 0 .. 255: the entries are the first n_ent threads
    const int id = e < n_ent ? sids[e] : 0;
    const bool valid = e < n_ent && (!HY || slot_of[id] >= 0);
    int rank = 0;
    if (valid) {
        for (int e2 = 0; e2 < n_ent; ++e2) {
            const int id2 = sids[e2];
            if (HY && slot_of[id2] < 0) continue;
            if (id2 < id || (id2 == id && e2 < e)) ++rank;
        }
        ent_dst[rank] = e;
        ent_tok[rank] = e / K;
        sid[rank] = id;
    }
    __syncthreads();
    int nv = 0;
    for (int e2 = 0; e2 < n_ent; ++e2) nv += (!HY || slot_of[sids[e2]] >= 0) ? 1 : 0;
    if (valid && (rank == 0 || sid[rank - 1] != sid[rank])) {
        int g = 0;
        for (int q = 1; q < rank; ++q) g += sid[q - 1] != sid[q] ? 1 : 0;
        g += rank > 0 ? 1 : 0;
        grp_ptr[g] = (unsigned long long) (blobs + (size_t) (HY ? slot_of[id] : id) * blob);
        grp_start[g] = rank;
    }
    if (e == 0) {
        int G = nv > 0 ? 1 : 0;
        for (int q = 1; q < nv; ++q) G += sid[q - 1] != sid[q] ? 1 : 0;
        grp_start[G] = nv;
        *n_groups = G;
    }
}

}  // namespace q36k
