// src/kernels/cuda/qsa_pattn.cu - see include/strata/kernels/qsa_pattn.hpp.
//
// Every value below is computed by the operation the production path uses, on the same operands, in the same order:
//   - score of (cell, head): each lane l's partial k8[0] * q[8l] + ... + k8[7] * q[8l + 7] with k8[e] = (float) code * sc,
//     the 32 partials summed in warp_sum's pairs (reduce12's tree), times `scale`;
//   - chunk softmax: m = warp_max over fmaxf(s[lane], s[lane + 32]); e = __expf(s - m) (0 for a masked or absent cell);
//     l = warp_sum(e[lane] + e[lane + 32]);
//   - chunk value: acc[h] = fmaf(e[c][h], v, acc[h]) over the chunk's cells in ascending order, masked cells skipped,
//     v = (float) code * sc;
//   - merge (attn_merge_kernel): M = fmaxf over the chunks' m; for each chunk in order with m != -FLT_MAX:
//     w = __expf(m - M), L = fmaf(l, w, L), acc = fmaf(acc_c, w, acc); out = L > 0 ? acc / L : 0.
// qsa_pattn_parity compares the outputs with qsa_decode_attn_batch's byte for byte on synthetic selections.
// Only the default variant of the contributor's PR (#3943) is kept: ps_score_x_kernel + ps_value_d4_kernel.
#include "strata/kernels/qsa_pattn.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/core/geometry.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if !defined(__HIPCC__) && defined(STRATA_EXPERIMENTAL_SM60)
#define STRATA_PATTN_BUILT 1   // the CUDA build with STRATA_EXPERIMENTAL_SM60 only (not HIP, not Metal)
#include <cuda_fp16.h>
#else
#define STRATA_PATTN_BUILT 0
#endif

namespace strata::kernels {

// STRATA_PATTN_SM60=1 turns it on (a tester switch, read once).  In a build without the sm_6x kernels it says so and stays off.
int pattn_mode() {
    static const int m = [] {
        const char* v = std::getenv("STRATA_PATTN_SM60");
        const int x = v != nullptr && *v != '\0' ? std::atoi(v) : 0;
        if (x == 0) return 0;
#if STRATA_PATTN_BUILT
        std::fprintf(stderr, "strata pattn: STRATA_PATTN_SM60=%d: the prompt path's QSA attention on the sm_6x score / "
                             "value kernels (bit for bit qsa_decode_attn_batch; int8 K/V)\n", x);
        return 1;
#else
        std::fprintf(stderr, "strata pattn: STRATA_PATTN_SM60 is set, but this build has no sm_6x kernels "
                             "(CUDA with STRATA_EXPERIMENTAL_SM60 only); the usual attention runs\n");
        return 0;
#endif
    }();
    return m;
}

#if STRATA_PATTN_BUILT
namespace {

constexpr int HD = 256;
constexpr int G = 12;
static_assert(HD == strata::core::kCompiledGeometry.head_dim &&
              G == strata::core::kCompiledGeometry.n_head / strata::core::kCompiledGeometry.n_head_kv,
              "qsa_pattn is compiled for core/geometry.hpp");
constexpr int CHUNK = 64;
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int NSC = HD / KV_Q8_GROUP;   // scales per row (4)

__device__ __forceinline__ float warp_sum_x(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 16); v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4); v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
}
__device__ __forceinline__ float warp_max_x(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 16)); v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4)); v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    return v;
}

// pool row of cell `c0 + t` of query selection `ids` (attn_chunk_kernel_pre75's srow), -1 masked / absent
__device__ __forceinline__ long long cell_row(const QsaAttnPools& p, const int32_t* __restrict__ ids, int c0, int t,
                                              int n_here, int n_kv_heads, int kvh, int page_size) {
    long long r = -1;
    if (t < n_here) {
        const int cell = __ldg(ids + c0 + t);
        const long long page = (long long) __ldg(p.page_table + cell / page_size);
        if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
    }
    return r;
}

// the chunk's K codes (16-byte loads) and scales (as float) into shared memory; masked / absent cells read row 0
__device__ __forceinline__ void stage_rows(const int8_t* __restrict__ codes, const uint16_t* __restrict__ scales,
                                           const long long* srow, int n_here, int t, int8_t (*sk)[HD],
                                           float (*sks)[NSC]) {
    uint4 kv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int idx = t + THREADS * j, c = idx >> 4, part = idx & 15;
        const long long row = srow[c];
        const long long rr = (c < n_here && row >= 0) ? row : 0;
        kv[j] = __ldg(reinterpret_cast<const uint4*>(codes + rr * HD) + part);
    }
    float sc;
    {
        const int c = t >> 2, g = t & 3;
        const long long row = srow[c];
        const long long rr = (c < n_here && row >= 0) ? row : 0;
        sc = __half2float(__ushort_as_half(__ldg(scales + rr * NSC + g)));
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) reinterpret_cast<uint4*>(&sk[0][0])[t + THREADS * j] = kv[j];
    sks[t >> 2][t & 3] = sc;
}

constexpr int PS_BLK = G * CHUNK + 2 * CHUNK + 2 * G;   // floats per (query, KV head, chunk): e, rows (int64), m, l
constexpr int NCM = 33;   // chunks the value kernel keeps (cap <= 2112)

// ---------------------------------------------------------------------------------------------------------------------
// ps_score_x_kernel: the score phase two cells at a time with a select-free reduce-scatter.  reduce12's selects exist
// because the two lanes of an exchange send different slots; here every lane keeps the same slots and sends the same
// slots, and the lane-dependent part is moved into which query head each register slot holds (q is loaded once, so the
// permutation costs nothing per cell) and which of the two cells is "first":
//   slot i (0..11) of lane L holds head P_L(i) = 6 ((i / 6) ^ b3) + ((i % 6 + 3 b2) % 6)       (b_k: bit k of L)
//   first = cell B if b4 else cell A, second = the other one; part1[i] / part2[i] = lane partials of (first / second,
//   head P_L(i)) - the same expression as attn_chunk_kernel_pre75.
//   level 16: r[i]  = part1[i] + shfl(part2[i], 16)   (partner: b4 flipped, so its second is my first, same P)
//   level 8 : r6[i] = r[i] + shfl(r[i + 6], 8)         (P_{L^8}(i) = P_L(i + 6))
//   level 4 : r3[i] = r6[i] + shfl(r6[i + 3], 4)       (P_{L^4}(i) = P_L(i + 3), i < 3)
//   level 2 / 1: reduce24's (heads 6 b3 + 3 b2 + {0, 1, 2}; o0 head +2 b1, o1 head +1 where b1 = 0)
// At every level the value at lane L is its partial at L plus its partial at L ^ o, the same operands warp_sum adds,
// so every score is reduce12's bit for bit.  A pair with an absent / masked cell runs the same code on row 0's staged
// codes and that cell's outputs are not stored (it gets -FLT_MAX).  25 shuffles, 25 adds and 4 selects per pair
// instead of 2 x (15 + 15 + ~30).
__device__ __forceinline__ int px_head(int i, int lane) {
    const int b3 = (lane >> 3) & 1, b2 = (lane >> 2) & 1;
    return 6 * ((i / 6) ^ b3) + ((i % 6 + 3 * b2) % 6);
}
__device__ __forceinline__ void load_q_perm(const float* __restrict__ qh, int lane, float4 (&qa)[G], float4 (&qb)[G]) {
#pragma unroll
    for (int i = 0; i < G; ++i) {
        const float4* p4 = reinterpret_cast<const float4*>(qh + (size_t) px_head(i, lane) * HD + lane * 8);
        qa[i] = __ldg(p4);
        qb[i] = __ldg(p4 + 1);
    }
}
__device__ __forceinline__ void dequant8(const int8_t* krow, float sc, int lane, float (&k8)[8]) {
    const uint2 raw = *reinterpret_cast<const uint2*>(krow + lane * 8);
    const int8_t* cb = reinterpret_cast<const int8_t*>(&raw);
#pragma unroll
    for (int e = 0; e < 8; ++e) k8[e] = (float) cb[e] * sc;
}
__device__ __forceinline__ float dot8(const float (&k8)[8], const float4& qa, const float4& qb) {
    return k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w + k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z +
           k8[7] * qb.w;
}
template <int UNR = 1>
__device__ __forceinline__ void score_chunk_x(const float4 (&qa)[G], const float4 (&qb)[G], const int8_t (*sk)[HD],
                                              const float (*sks)[NSC], const long long* srow, float (*sp)[CHUNK],
                                              int n_here, float scale, int warp, int lane) {
    const bool h16 = (lane & 16) != 0, h2 = (lane & 2) != 0, even = (lane & 1) == 0;
    const int hb = 6 * ((lane >> 3) & 1) + 3 * ((lane >> 2) & 1);
    const int h0 = hb + (h2 ? 2 : 0), h1 = hb + 1;
#pragma unroll UNR
    for (int i = 0; i < CHUNK / WARPS; i += 2) {
        const int ca = warp + WARPS * i, cb = warp + WARPS * (i + 1);
        const bool va = ca < n_here && srow[ca] >= 0, vb = cb < n_here && srow[cb] >= 0;
        if (!va && !vb) {
            if (lane < G) { sp[lane][ca] = -FLT_MAX; sp[lane][cb] = -FLT_MAX; }
            continue;
        }
        const int c1 = h16 ? cb : ca, c2 = h16 ? ca : cb;   // first / second (the staged rows of absent cells are row 0's)
        float r[G];
        {
            float k8[8];
            dequant8(sk[c2], sks[c2][lane >> 3], lane, k8);
            float p2[G];
#pragma unroll
            for (int s = 0; s < G; ++s) p2[s] = dot8(k8, qa[s], qb[s]);
            dequant8(sk[c1], sks[c1][lane >> 3], lane, k8);
#pragma unroll
            for (int s = 0; s < G; ++s) r[s] = dot8(k8, qa[s], qb[s]) + __shfl_xor_sync(0xffffffffu, p2[s], 16);
        }
        float r6[6];
#pragma unroll
        for (int s = 0; s < 6; ++s) r6[s] = r[s] + __shfl_xor_sync(0xffffffffu, r[s + 6], 8);
        float r3[3];
#pragma unroll
        for (int s = 0; s < 3; ++s) r3[s] = r6[s] + __shfl_xor_sync(0xffffffffu, r6[s + 3], 4);
        const float x0 = (h2 ? r3[2] : r3[0]) + __shfl_xor_sync(0xffffffffu, h2 ? r3[0] : r3[2], 2);
        const float x1 = r3[1] + __shfl_xor_sync(0xffffffffu, r3[1], 2);
        const float o0 = x0 + __shfl_xor_sync(0xffffffffu, x0, 1);
        const float o1 = x1 + __shfl_xor_sync(0xffffffffu, x1, 1);
        const bool vmine = h16 ? vb : va;
        if (even && vmine) {
            sp[h0][c1] = o0 * scale;
            if (!h2) sp[h1][c1] = o1 * scale;
        }
        if (lane < G) {
            if (!va) sp[lane][ca] = -FLT_MAX;
            if (!vb) sp[lane][cb] = -FLT_MAX;
        }
    }
}

// ps_score_kernel with score_chunk_x (q loaded in the permuted slot order)
template <int UNR>
__global__ void __launch_bounds__(THREADS, 2) ps_score_x_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                                const int32_t* __restrict__ ids,
                                                                const int32_t* __restrict__ step, int n_kv_heads,
                                                                int page_size, float scale, float* __restrict__ blocks,
                                                                int n_chunks, int cap) {
    q += (size_t) blockIdx.z * (size_t) (n_kv_heads * G) * HD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * kStepCount;
    const int chunk = blockIdx.x, kvh = blockIdx.y;
    float* const blk = blocks + (((size_t) blockIdx.z * n_kv_heads + kvh) * n_chunks + chunk) * PS_BLK;
    __shared__ __align__(16) int8_t sk[CHUNK][HD];
    __shared__ __align__(16) float sks[CHUNK][NSC];
    __shared__ float sp[G][CHUNK];
    __shared__ long long srow[CHUNK];
    const int n_ids = __ldg(step + kStepWidth);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = min(CHUNK, n_ids - c0);
    if (n_here <= 0) {
        if (t < G) { blk[G * CHUNK + 2 * CHUNK + t] = -FLT_MAX; blk[G * CHUNK + 2 * CHUNK + G + t] = 0.0f; }
        return;
    }
    if (t < CHUNK) {
        const long long r = cell_row(p, ids, c0, t, n_here, n_kv_heads, kvh, page_size);
        srow[t] = r;
        reinterpret_cast<long long*>(blk + G * CHUNK)[t] = r;
    }
    float4 qa[G], qb[G];
    load_q_perm(q + (size_t) (kvh * G) * HD, lane, qa, qb);
    __syncthreads();
    stage_rows(p.k_q, p.k_scale, srow, n_here, t, sk, sks);
    __syncthreads();
    score_chunk_x<UNR>(qa, qb, sk, sks, srow, sp, n_here, scale, warp, lane);
    __syncthreads();
    for (int h = warp; h < G; h += WARPS) {
        const float a = sp[h][lane], b = sp[h][lane + 32];
        const float m = warp_max_x(fmaxf(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0) ? __expf(a - m) : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? __expf(b - m) : 0.0f;
        blk[h * CHUNK + lane] = ea;
        blk[h * CHUNK + lane + 32] = eb;
        const float l = warp_sum_x(ea + eb);
        if (lane == 0) { blk[G * CHUNK + 2 * CHUNK + h] = m; blk[G * CHUNK + 2 * CHUNK + G + h] = l; }
    }
}

// ps_value_d4_kernel: the value kernel with four dims a thread (64 threads, one CTA per (KV head, query)): a cell's four
// codes are one 32-bit load (a warp reads 128 contiguous bytes), its 12 weights three broadcast float4 reads for 48
// fmaf.  V loads run four cells ahead.  Per (dim, head) the fold and the merge are ps_value_kernel's.
__global__ void __launch_bounds__(64) ps_value_d4_kernel(QsaAttnPools p, const int32_t* __restrict__ steps,
                                                         const float* __restrict__ blocks, int n_kv_heads, int n_chunks,
                                                         float* __restrict__ attn) {
    constexpr int NT = 64;
    const int kvh = blockIdx.x, z = blockIdx.y;
    const int t = threadIdx.x;
    const float* const blk0 = blocks + ((size_t) z * n_kv_heads + kvh) * n_chunks * PS_BLK;
    __shared__ float sm_m[NCM][G], sm_w[NCM][G], sm_l[NCM][G];
    __shared__ float sL[G];
    __shared__ __align__(16) float spt[CHUNK][G];
    __shared__ long long srow[CHUNK];
    __shared__ int s_masked;
    const int width = __ldg(steps + (size_t) z * kStepCount + kStepWidth);
    const int n_act = width > 0 ? min(n_chunks, (width - 1) / CHUNK + 1) : 0;
    for (int i = t; i < n_act * G; i += NT) {
        const int c = i / G, h = i % G;
        sm_m[c][h] = __ldg(blk0 + (size_t) c * PS_BLK + G * CHUNK + 2 * CHUNK + h);
        sm_l[c][h] = __ldg(blk0 + (size_t) c * PS_BLK + G * CHUNK + 2 * CHUNK + G + h);
    }
    __syncthreads();
    if (t < G) {
        float M = -FLT_MAX;
        for (int c = 0; c < n_act; ++c) M = fmaxf(M, sm_m[c][t]);
        float L = 0.0f;
        for (int c = 0; c < n_act; ++c) {
            const float m = sm_m[c][t];
            const float w = __expf(m - M);
            sm_w[c][t] = w;
            const float L_next = fmaf(sm_l[c][t], w, L);
            L = (m == -FLT_MAX) ? L : L_next;
        }
        sL[t] = L;
    }
    const int d0 = 4 * t, g = d0 / KV_Q8_GROUP;
    float acc[4][G];
#pragma unroll
    for (int h = 0; h < G; ++h)
#pragma unroll
        for (int d = 0; d < 4; ++d) acc[d][h] = 0.0f;
    for (int c = 0; c < n_act; ++c) {
        const float* blk = blk0 + (size_t) c * PS_BLK;
        const int n_here = min(CHUNK, width - c * CHUNK);
        __syncthreads();
        if (t == 0) s_masked = 0;
        __syncthreads();
        {
            const long long r = __ldg(reinterpret_cast<const long long*>(blk + G * CHUNK) + t);
            srow[t] = r;
            if (t < n_here && r < 0) s_masked = 1;
        }
#pragma unroll
        for (int j = 0; j < G * CHUNK / NT; ++j) {
            const int idx = t + NT * j, h = idx / CHUNK, cell = idx % CHUNK;
            spt[cell][h] = __ldg(blk + idx);
        }
        __syncthreads();
        float a[4][G];
#pragma unroll
        for (int h = 0; h < G; ++h)
#pragma unroll
            for (int d = 0; d < 4; ++d) a[d][h] = 0.0f;
        const auto fold4 = [&](int cl, uint32_t codes, float sc) {
            const float v0 = (float) (int8_t) (codes & 0xFF) * sc, v1 = (float) (int8_t) ((codes >> 8) & 0xFF) * sc;
            const float v2 = (float) (int8_t) ((codes >> 16) & 0xFF) * sc, v3 = (float) (int8_t) (codes >> 24) * sc;
            const float4* w4 = reinterpret_cast<const float4*>(spt[cl]);
            const float4 w0 = w4[0], w1 = w4[1], w2 = w4[2];
            const float ww[G] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w, w2.x, w2.y, w2.z, w2.w};
#pragma unroll
            for (int h = 0; h < G; ++h) {
                a[0][h] = fmaf(ww[h], v0, a[0][h]);
                a[1][h] = fmaf(ww[h], v1, a[1][h]);
                a[2][h] = fmaf(ww[h], v2, a[2][h]);
                a[3][h] = fmaf(ww[h], v3, a[3][h]);
            }
        };
        const auto vld = [&](int cl, uint32_t& codes, float& sc) {
            const long long r = srow[cl];
            const long long rr = r >= 0 ? r : 0;
            codes = __ldg(reinterpret_cast<const uint32_t*>(p.v_q + rr * HD + d0));
            sc = __half2float(__ushort_as_half(__ldg(p.v_scale + rr * NSC + g)));
        };
        if (!s_masked) {
            int cl = 0;
            for (; cl + 4 <= n_here; cl += 4) {
                uint32_t k0, k1, k2, k3;
                float s0, s1, s2, s3;
                vld(cl, k0, s0); vld(cl + 1, k1, s1); vld(cl + 2, k2, s2); vld(cl + 3, k3, s3);
                fold4(cl, k0, s0);
                fold4(cl + 1, k1, s1);
                fold4(cl + 2, k2, s2);
                fold4(cl + 3, k3, s3);
            }
            for (; cl < n_here; ++cl) {
                uint32_t k0;
                float s0;
                vld(cl, k0, s0);
                fold4(cl, k0, s0);
            }
        } else {
            for (int cl = 0; cl < n_here; ++cl) {
                if (srow[cl] < 0) continue;
                uint32_t k0;
                float s0;
                vld(cl, k0, s0);
                fold4(cl, k0, s0);
            }
        }
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const bool live = !(sm_m[c][h] == -FLT_MAX);
            const float w = sm_w[c][h];
#pragma unroll
            for (int d = 0; d < 4; ++d) {
                const float acc_next = fmaf(a[d][h], w, acc[d][h]);
                acc[d][h] = live ? acc_next : acc[d][h];
            }
        }
    }
    __syncthreads();
    float* out = attn + ((size_t) z * n_kv_heads + kvh) * G * HD;
#pragma unroll
    for (int h = 0; h < G; ++h) {
        const float L = sL[h];
        *reinterpret_cast<float4*>(out + (size_t) h * HD + d0) =
            make_float4(L > 0.0f ? acc[0][h] / L : 0.0f, L > 0.0f ? acc[1][h] / L : 0.0f,
                        L > 0.0f ? acc[2][h] / L : 0.0f, L > 0.0f ? acc[3][h] / L : 0.0f);
    }
}

bool sm6x_device() {
    static int cc[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (cc[dev] == 0) {
        int major = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        cc[dev] = major;
    }
    return cc[dev] == 6;
}

}  // namespace

bool qsa_pattn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                     const QsaShapes& s, float* scratch, uint64_t scratch_floats, float* attn, int64_t n_q,
                     void* stream) {
    return sm6x_device() && qsa_pattn_batch_any_arch(q, pools, ids, steps, cap, s, scratch, scratch_floats, attn, n_q, stream);
}

bool qsa_pattn_batch_any_arch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                              int64_t cap, const QsaShapes& s, float* scratch, uint64_t scratch_floats, float* attn,
                              int64_t n_q, void* stream) {
    if (n_q <= 0) return true;
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !steps ||
        !pools.page_table || pools.k_q == nullptr || pools.v_q == nullptr || pools.k_q4 != nullptr ||
        pools.v_q4 != nullptr || !pools.k_scale || !pools.v_scale)
        return false;
    const cudaStream_t st = (cudaStream_t) stream;
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    const float scale = 1.0f / sqrtf((float) HD);
    if (n_chunks > NCM) return false;   // a longer selection keeps the usual path
    // the scratch holds the blocks of one batch of queries (probabilities, rows, m and l of every chunk), evenly split
    const uint64_t per_q = (uint64_t) s.n_head_kv * (uint64_t) n_chunks * PS_BLK;
    const int64_t max_batch = (int64_t) (scratch_floats / per_q);
    if (max_batch < 1) return false;
    const int64_t n_batches = (n_q + max_batch - 1) / max_batch;
    const int64_t batch = (n_q + n_batches - 1) / n_batches;
    for (int64_t t0 = 0; t0 < n_q; t0 += batch) {
        const int64_t nb = n_q - t0 < batch ? n_q - t0 : batch;
        const dim3 g1((unsigned) n_chunks, (unsigned) s.n_head_kv, (unsigned) nb);
        const dim3 g2((unsigned) s.n_head_kv, (unsigned) nb);
        const float* qb = q + (size_t) t0 * s.n_head * HD;
        const int32_t* ib = ids + (size_t) t0 * cap;
        const int32_t* sb = steps + (size_t) t0 * kStepCount;
        ps_score_x_kernel<1><<<g1, THREADS, 0, st>>>(qb, pools, ib, sb, (int) s.n_head_kv, (int) s.page_size, scale,
                                                     scratch, n_chunks, (int) cap);
        ps_value_d4_kernel<<<g2, 64, 0, st>>>(pools, sb, scratch, (int) s.n_head_kv, n_chunks,
                                              attn + (size_t) t0 * s.n_head * HD);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        // a launch that did not start: say so once, and let the caller run the usual attention over the whole chunk
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::fprintf(stderr, "strata pattn: launch failed (%s); the usual attention runs instead\n", cudaGetErrorString(e));
        }
        return false;
    }
    return true;
}

#else  // not the experimental sm_60 CUDA build
bool qsa_pattn_batch(const float*, const QsaAttnPools&, const int32_t*, const int32_t*, int64_t, const QsaShapes&,
                     float*, uint64_t, float*, int64_t, void*) { return false; }
bool qsa_pattn_batch_any_arch(const float*, const QsaAttnPools&, const int32_t*, const int32_t*, int64_t, const QsaShapes&,
                              float*, uint64_t, float*, int64_t, void*) { return false; }
#endif

}  // namespace strata::kernels
