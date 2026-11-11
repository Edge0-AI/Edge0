// src/kernels/cuda/qsa_prompt_attn.cu - see include/strata/kernels/qsa_prompt_attn.hpp.
#include "strata/core/emulate.hpp"
#include "strata/core/sm120_series.hpp"
#include "strata/core/defaults_026.hpp"
#include "strata/core/sm75_series.hpp"
#include "strata/core/geometry.hpp"
#include "strata/core/gfx12_platform_defaults.hpp"
#include <cstdlib>
#include <cstring>
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/gfx_arch.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_q4.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>

#include <atomic>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace strata::kernels {
namespace {

constexpr int HD = 256;           // head_dim
constexpr int G = 12;             // query heads per KV head
static_assert(HD == strata::core::kCompiledGeometry.head_dim && G == strata::core::kCompiledGeometry.n_head / strata::core::kCompiledGeometry.n_head_kv, "qsa_prompt_attn is compiled for core/geometry.hpp");
#ifndef D1_CH
#define D1_CH 32
#endif
constexpr int CH = D1_CH;         // cells per chunk
constexpr int THREADS = 128;      // 4 warps: scores by cell (8 each), p.v by dimension (64 each = one int8 scale group)
constexpr int QS = HD + 8;        // q row stride in halves (bank-conflict-free fragment loads)

// The MMA below needs sm_75 or newer (Turing runs it as two k=8 steps); cp.async needs sm_80. Builds for pre-sm_75
// cards compile the MMA to a trap; qsa_prompt_attn_batch refuses such a device at run time, so the old kernel runs
// there.  Turing compiles cp_async16 to a trap as well and takes the v1 kernel instead of launch_i8.
#if defined(__HIPCC__)          // AMD: no mma.sync / cp.async; the host keeps the old kernel (below)
#define STRATA_PA_SM80 0
#elif !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define STRATA_PA_SM80 1
#else
#define STRATA_PA_SM80 0
#endif

// m16n8k16 with f16 inputs needs sm_80.  Turing (sm_75) has m16n8k8 with the SAME A/B/C register mapping, so the
// k=16 step is two k=8 steps on the fragments as they are already laid out: a[0]/a[1] are rows gid/gid+8 at k columns
// 2*tig..2*tig+1 (b[0]'s k rows), a[2]/a[3] the same rows at k columns 2*tig+8..2*tig+9 (b[1]'s k rows).  The
// products then add into the same FP32 C registers in the order hi-part-0, hi-part-1, which is the order the k16
// instruction accumulates in as well - but the sum now rounds twice, so the two paths do not agree bit for bit.
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
#if !STRATA_PA_SM80 && (defined(__HIPCC__) || !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 750)
    __trap();   // AMD and pre-Turing builds: no mma.sync (the host keeps the old kernel there)
#elif !STRATA_PA_SM80
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(b[0]));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[2]), "r"(a[3]), "r"(b[1]));
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

// Two int8 codes (low byte first) as an exact half2: 1024 + (c + 128) built in the mantissa, minus 1152.
__device__ __forceinline__ uint32_t i8x2_to_h2(uint32_t x) {
    uint32_t y = ((x & 0xffu) | ((x & 0xff00u) << 8)) ^ 0x00800080u;
    y |= 0x64006400u;
    __half2 h = *reinterpret_cast<__half2*>(&y);
    h = __hsub2(h, __halves2half2(__float2half(1152.f), __float2half(1152.f)));
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ uint32_t pack_h2(float lo_k, float hi_k) {   // element k in the low half
    __half2 h = __floats2half2_rn(lo_k, hi_k);
    return *reinterpret_cast<uint32_t*>(&h);
}

// KV_MODE 1: int8 codes + fp16 scale per 64 values. KV_MODE 0: fp16 values (scales 1).
// KV_MODE 3 (hybrid K8V4): K as mode 1, V as mode 0 - the row's q4_0 blocks are dequantized to fp16 at
// gather, so everything downstream of the load is the mode-0 V path; the caller un-rotates the output.
// KV_MODE 4 (Q4_0 K and V, `--kv q4_0`): mode 1 with a scale per 32 values - each q4_0 block's codes enter as exact
// int8 (code - 8) and its fp16 scale is applied in FP32, as mode 1's are; the caller rotates q and un-rotates the
// output (kv_q4.hpp).
template <int KV_MODE, bool PF = false>
struct Smem {
    using KElem = typename std::conditional<KV_MODE == 0, __half, int8_t>::type;
    using VElem = typename std::conditional<KV_MODE == 1 || KV_MODE == 4, int8_t, __half>::type;
    static constexpr int KROW = KV_MODE == 0 ? HD + 8 : HD + 16;   // elements; 16-byte aligned rows, banks spread
    static constexpr int VROW = (KV_MODE == 1 || KV_MODE == 4) ? HD + 16 : HD + 8;
    static constexpr int NG = KV_MODE == 4 ? HD / QK4_0 : 4;      // scale groups per row (q4_0: 8 of 32, else 4 of 64)
    __half qh[16][QS];
    __half ql[16][QS];
    // int8 K/V (mode 1): K is read only by the scores and V only by p.v, so they share one buffer - V is stored over K
    // after the scores (prompt_attn_kernel).  37.9 -> 29.2 KB: two blocks fit an SM's 64 KB on Turing instead of one.
    struct Rows { KElem k[CH][KROW]; VElem v[CH][VROW]; };
    struct SharedRows { union { KElem k[CH][KROW]; VElem v[CH][VROW]; }; };
    typename std::conditional<KV_MODE == 1 && PF, SharedRows, Rows>::type kv;
    float ks[CH][NG];
    float vs[CH][NG];
    float s[16][CH + 1];
    float qmax[THREADS / 32];
    float alpha[16];
    float lsum[16];
    float mrow[16];
    long long row[CH];
};

template <int KV_MODE, bool PF = false>
__global__ void __launch_bounds__(THREADS) prompt_attn_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                              const int32_t* __restrict__ ids,
                                                              const int32_t* __restrict__ steps, int n_kv_heads,
                                                              int page_size, float scale_log2, float* __restrict__ attn,
                                                              int cap) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem<KV_MODE, PF>& S = *reinterpret_cast<Smem<KV_MODE, PF>*>(smem_raw);
    const int qi = blockIdx.x, kvh = blockIdx.y;
    const int n_head = n_kv_heads * G;
    q += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    attn += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    ids += (size_t) qi * cap;
    const int n = __ldg(steps + (size_t) qi * kStepCount + kStepWidth);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int gid = lane >> 2, tig = lane & 3;

    // q: 12 heads + 4 zero rows, scaled by a power of two that puts its largest value near 2^14 (exact, and the
    // lo halves stay out of FP16's subnormal range), then split into hi + lo halves
    float qm = 0.0f;
    for (int i = t; i < G * HD; i += THREADS) qm = fmaxf(qm, fabsf(q[i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, o));
    if (lane == 0) S.qmax[warp] = qm;
    __syncthreads();
    qm = fmaxf(fmaxf(S.qmax[0], S.qmax[1]), fmaxf(S.qmax[2], S.qmax[3]));
    int qe = 0;
    if (qm > 0.0f) frexpf(qm, &qe);                 // qm < 2^qe
    const float qup = ldexpf(1.0f, 14 - qe), qdown = ldexpf(scale_log2, qe - 14);
    for (int i = t; i < 16 * HD; i += THREADS) {
        const int h = i / HD, d = i % HD;
        const float x = h < G ? q[(size_t) h * HD + d] * qup : 0.0f;
        const __half hi = __float2half_rn(x);
        S.qh[h][d] = hi;
        S.ql[h][d] = __float2half_rn(x - __half2float(hi));
    }
    if (t < 16) { S.mrow[t] = -INFINITY; S.lsum[t] = 0.0f; }

    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;

    // int8 K/V: the next chunk's rows are loaded into registers while this chunk computes (Turing has no cp.async, and
    // at 1 block per SM the gather's latency was the kernel's largest stall), then stored to shared memory exactly as
    // the gather below stores them: the same bytes in the same places, the same arithmetic after.
    constexpr bool PREFETCH = KV_MODE == 1 && PF;
    constexpr int KP = HD / 16;                               // 16-byte pieces per int8 row
    constexpr int PPT = PREFETCH ? CH * KP / THREADS : 1;     // pieces per thread
    static_assert(!PREFETCH || (CH * KP) % THREADS == 0 && CH * 4 == THREADS, "prefetch layout");
    uint4 pk[PPT], pv[PPT];
    float pa = 0.0f, pb = 0.0f;
    auto row_of = [&](int c0, int c) -> long long {          // as S.row below; -1 past the chunk's cells
        if (c >= min(CH, n - c0)) return -1;
        const int cell = ids[c0 + c];
        const long long page = (long long) p.page_table[cell / page_size];
        return (page * n_kv_heads + kvh) * page_size + (cell % page_size);
    };
    auto fetch = [&](int c0) {
        if constexpr (PREFETCH) {
#pragma unroll
            for (int j = 0; j < PPT; ++j) {
                const int i = t + j * THREADS, c = i / KP, pc = i % KP;
                const long long r = row_of(c0, c);
                pk[j] = r >= 0 ? __ldg(reinterpret_cast<const uint4*>(p.k_q + r * HD) + pc) : make_uint4(0, 0, 0, 0);
                pv[j] = r >= 0 ? __ldg(reinterpret_cast<const uint4*>(p.v_q + r * HD) + pc) : make_uint4(0, 0, 0, 0);
            }
            const int c = t / 4, g = t % 4;
            const long long r = row_of(c0, c);
            pa = r >= 0 ? __half2float(__ushort_as_half(p.k_scale[r * (HD / KV_Q8_GROUP) + g])) : 0.0f;
            pb = r >= 0 ? __half2float(__ushort_as_half(p.v_scale[r * (HD / KV_Q8_GROUP) + g])) : 0.0f;
        }
    };
    if constexpr (PREFETCH) fetch(0);

    for (int c0 = 0; c0 < n; c0 += CH) {
        const int nh = min(CH, n - c0);
        if constexpr (PREFETCH) {
            __syncthreads();   // the previous chunk's p.v is done with k, v, s
#pragma unroll
            for (int j = 0; j < PPT; ++j) {
                const int i = t + j * THREADS, c = i / KP, pc = i % KP;
                *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(&S.kv.k[c][0]) + pc * 16) = pk[j];
            }
            S.ks[t / 4][t % 4] = pa;
            S.vs[t / 4][t % 4] = pb;
            __syncthreads();
        } else {
        if (t < CH) {
            long long r = -1;
            if (t < nh) {
                const int cell = ids[c0 + t];
                const long long page = (long long) p.page_table[cell / page_size];
                r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
            }
            S.row[t] = r;
        }
        __syncthreads();   // rows ready; the previous chunk's p.v is done with k, v, s
        // gather the chunk's K and V rows (16-byte pieces; K8V4's V as q4_0 blocks dequantized to fp16)
        // and their scales
        if constexpr (KV_MODE == 4) {
            // q4_0 K and V: one (K or V, cell, block) per item - the block's 16 code bytes (2-byte aligned: eight
            // 16-bit loads) become 32 exact int8 codes (element j in the low nibble of byte j, j + 16 in the high
            // one), its fp16 scale goes to ks/vs
            constexpr int BLKS = HD / QK4_0;
            constexpr int BYTES = BLKS * (int) sizeof(block_q4_0);
            for (int i = t; i < 2 * CH * BLKS; i += THREADS) {
                const int kv = i / (CH * BLKS), rem = i % (CH * BLKS), c = rem / BLKS, b = rem % BLKS;
                const long long r = S.row[c];
                uint4 lo = make_uint4(0, 0, 0, 0), hi = make_uint4(0, 0, 0, 0);
                float d = 0.0f;
                if (r >= 0) {
                    const block_q4_0* blk = reinterpret_cast<const block_q4_0*>((kv == 0 ? p.k_q4 : p.v_q4) + r * BYTES) + b;
                    d = __half2float(__ushort_as_half(__ldg(&blk->d)));
                    const uint16_t* q16 = reinterpret_cast<const uint16_t*>(blk->qs);
                    uint32_t w[4];
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        w[j] = (uint32_t) __ldg(q16 + 2 * j) | ((uint32_t) __ldg(q16 + 2 * j + 1) << 16);
                    // per byte: the low nibble minus 8 is element j, the high one minus 8 element j + 16
                    lo = make_uint4(__vsub4(w[0] & 0x0F0F0F0Fu, 0x08080808u), __vsub4(w[1] & 0x0F0F0F0Fu, 0x08080808u),
                                    __vsub4(w[2] & 0x0F0F0F0Fu, 0x08080808u), __vsub4(w[3] & 0x0F0F0F0Fu, 0x08080808u));
                    hi = make_uint4(__vsub4((w[0] >> 4) & 0x0F0F0F0Fu, 0x08080808u),
                                    __vsub4((w[1] >> 4) & 0x0F0F0F0Fu, 0x08080808u),
                                    __vsub4((w[2] >> 4) & 0x0F0F0F0Fu, 0x08080808u),
                                    __vsub4((w[3] >> 4) & 0x0F0F0F0Fu, 0x08080808u));
                }
                int8_t* dst = kv == 0 ? &S.kv.k[c][b * QK4_0] : &S.kv.v[c][b * QK4_0];
                reinterpret_cast<uint4*>(dst)[0] = lo;
                reinterpret_cast<uint4*>(dst)[1] = hi;
                (kv == 0 ? S.ks : S.vs)[c][b] = d;
            }
        } else {
            constexpr int KPIECES = HD * (int) sizeof(typename Smem<KV_MODE>::KElem) / 16;   // per K row
            for (int i = t; i < CH * KPIECES; i += THREADS) {
                const int c = i / KPIECES, pc = i % KPIECES;
                const long long r = S.row[c];
                uint4 kx = make_uint4(0, 0, 0, 0);
                if (r >= 0) {
                    if constexpr (KV_MODE == 0)
                        kx = __ldg(reinterpret_cast<const uint4*>(p.k_pool + r * HD) + pc);
                    else   // modes 1 and 3: the K side is INT8
                        kx = __ldg(reinterpret_cast<const uint4*>(p.k_q + r * HD) + pc);
                }
                *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(&S.kv.k[c][0]) + pc * 16) = kx;
            }
            if constexpr (KV_MODE == 3) {   // V: dequantize the row's q4_0 blocks straight into the fp16 V row
                constexpr int BLKS = HD / QK4_0;
                constexpr int BYTES = BLKS * (int) sizeof(block_q4_0);
                for (int i = t; i < CH * BLKS; i += THREADS) {
                    const int c = i / BLKS, b = i % BLKS;
                    const long long r = S.row[c];
#pragma unroll
                    for (int j = 0; j < QK4_0; ++j) S.kv.v[c][b * QK4_0 + j] = __half(0);
                    if (r >= 0) {
                        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + r * BYTES) + b;
                        const float d = __half2float(__ushort_as_half(blk->d));
#pragma unroll
                        for (int j = 0; j < QK4_0 / 2; ++j) {
                            S.kv.v[c][b * QK4_0 + j] = __float2half_rn((float) ((int)(blk->qs[j] & 0x0F) - 8) * d);
                            S.kv.v[c][b * QK4_0 + j + QK4_0 / 2] =
                                __float2half_rn((float) ((int)(blk->qs[j] >> 4) - 8) * d);
                        }
                    }
                }
            } else {
                constexpr int VPIECES = HD * (int) sizeof(typename Smem<KV_MODE>::VElem) / 16;   // per V row
                for (int i = t; i < CH * VPIECES; i += THREADS) {
                    const int c = i / VPIECES, pc = i % VPIECES;
                    const long long r = S.row[c];
                    uint4 vx = make_uint4(0, 0, 0, 0);
                    if (r >= 0) {
                        if constexpr (KV_MODE == 1)
                            vx = __ldg(reinterpret_cast<const uint4*>(p.v_q + r * HD) + pc);
                        else
                            vx = __ldg(reinterpret_cast<const uint4*>(p.v_pool + r * HD) + pc);
                    }
                    *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(&S.kv.v[c][0]) + pc * 16) = vx;
                }
            }
            for (int i = t; i < CH * 4; i += THREADS) {
                const int c = i / 4, g = i % 4;
                const long long r = S.row[c];
                float a = 0.0f, b = 0.0f;
                if (r >= 0) {
                    if constexpr (KV_MODE == 1) {
                        a = __half2float(__ushort_as_half(p.k_scale[r * (HD / KV_Q8_GROUP) + g]));
                        b = __half2float(__ushort_as_half(p.v_scale[r * (HD / KV_Q8_GROUP) + g]));
                    } else if constexpr (KV_MODE == 3) {   // K as int8, V dequantized to fp16 (scale 1)
                        a = __half2float(__ushort_as_half(p.k_scale[r * (HD / KV_Q8_GROUP) + g]));
                        b = 1.0f;
                    } else {
                        a = b = 1.0f;
                    }
                }
                S.ks[c][g] = a;
                S.vs[c][g] = b;
            }
        }
        __syncthreads();
        }
        // scores: warp w takes cells 8w..8w+7 (one n-tile) over all 256 dims, per scale group (64 dims; q4_0's 32)
        constexpr int NG = Smem<KV_MODE>::NG, KPG = HD / 16 / NG;   // groups per row, 16-dim MMA steps per group
#pragma unroll
        for (int nt = 0; nt < CH / 32; ++nt) {
            const int cb = (warp + 4 * nt) * 8;
            float sc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int g = 0; g < NG; ++g) {
                float tg[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int kk = 0; kk < KPG; ++kk) {
                    const int k0 = (g * KPG + kk) * 16;
                    uint32_t ah[4], al[4], b[2];
                    ah[0] = *reinterpret_cast<const uint32_t*>(&S.qh[gid][k0 + 2 * tig]);
                    ah[1] = *reinterpret_cast<const uint32_t*>(&S.qh[gid + 8][k0 + 2 * tig]);
                    ah[2] = *reinterpret_cast<const uint32_t*>(&S.qh[gid][k0 + 2 * tig + 8]);
                    ah[3] = *reinterpret_cast<const uint32_t*>(&S.qh[gid + 8][k0 + 2 * tig + 8]);
                    al[0] = *reinterpret_cast<const uint32_t*>(&S.ql[gid][k0 + 2 * tig]);
                    al[1] = *reinterpret_cast<const uint32_t*>(&S.ql[gid + 8][k0 + 2 * tig]);
                    al[2] = *reinterpret_cast<const uint32_t*>(&S.ql[gid][k0 + 2 * tig + 8]);
                    al[3] = *reinterpret_cast<const uint32_t*>(&S.ql[gid + 8][k0 + 2 * tig + 8]);
                    if constexpr (KV_MODE != 0) {   // modes 1 and 3: the K side is INT8 codes
                        b[0] = i8x2_to_h2(*reinterpret_cast<const uint16_t*>(&S.kv.k[cb + gid][k0 + 2 * tig]));
                        b[1] = i8x2_to_h2(*reinterpret_cast<const uint16_t*>(&S.kv.k[cb + gid][k0 + 2 * tig + 8]));
                    } else {
                        b[0] = *reinterpret_cast<const uint32_t*>(&S.kv.k[cb + gid][k0 + 2 * tig]);
                        b[1] = *reinterpret_cast<const uint32_t*>(&S.kv.k[cb + gid][k0 + 2 * tig + 8]);
                    }
                    mma16816(tg, ah, b);
#ifndef D1_NO_QLO
                    mma16816(tg, al, b);
#endif
                }
                const float s0 = S.ks[cb + 2 * tig][g], s1 = S.ks[cb + 2 * tig + 1][g];
                sc[0] = fmaf(tg[0], s0, sc[0]);
                sc[1] = fmaf(tg[1], s1, sc[1]);
                sc[2] = fmaf(tg[2], s0, sc[2]);
                sc[3] = fmaf(tg[3], s1, sc[3]);
            }
            const int c = cb + 2 * tig;
            S.s[gid][c] = c < nh ? sc[0] * qdown : -INFINITY;
            S.s[gid][c + 1] = c + 1 < nh ? sc[1] * qdown : -INFINITY;
            S.s[gid + 8][c] = c < nh ? sc[2] * qdown : -INFINITY;
            S.s[gid + 8][c + 1] = c + 1 < nh ? sc[3] * qdown : -INFINITY;
        }
        __syncthreads();
        if constexpr (PREFETCH) {   // the scores are done with K: V goes over it (read after the softmax's barrier)
#pragma unroll
            for (int j = 0; j < PPT; ++j) {
                const int i = t + j * THREADS, c = i / KP, pc = i % KP;
                *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(&S.kv.v[c][0]) + pc * 16) = pv[j];
            }
            if (c0 + CH < n) fetch(c0 + CH);   // the next chunk's rows, in flight while this one finishes
        }
        // online softmax: row t/8, 4 cells per thread, 8 threads per row (lanes 8r..8r+7 of a warp)
        {
            constexpr int PER = CH / 8;
            const int r = t >> 3, sub = t & 7;
            float x[PER], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < PER; ++j) { x[j] = S.s[r][sub * PER + j]; mx = fmaxf(mx, x[j]); }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : exp2f(x[j] - m_new);
                S.s[r][sub * PER + j] = e;
                sum += e;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            __syncwarp();
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : exp2f(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = fmaf(S.lsum[r], a, sum);
                S.mrow[r] = m_new;
            }
        }
        __syncthreads();
        // p.v: warp w owns dims [64w, 64w+64), which is int8 scale group w (q4_0: groups 2w and 2w+1, four n-tiles
        // each). A group's scale is folded into p relative to the chunk's largest magnitude (q4_0's scales are signed:
        // ggml's d = max / -8), times 2^14 (|p'| <= 2^14: inside FP16's range, its lo half out of the subnormals); the
        // chunk's sum is then added to the running one in FP32 with the factor taken back out
        {
            constexpr int GPW = NG / 4, JPG = 8 / GPW;   // scale groups per warp, n-tiles per group
            float vdown_g[GPW];
            float tmp[8][4];
#pragma unroll
            for (int j = 0; j < 8; ++j) tmp[j][0] = tmp[j][1] = tmp[j][2] = tmp[j][3] = 0.0f;
#pragma unroll
            for (int gi = 0; gi < GPW; ++gi) {
            const int vg = warp * GPW + gi;
            float vmax = 0.0f;
#pragma unroll
            for (int c = lane; c < CH; c += 32) vmax = fmaxf(vmax, fabsf(S.vs[c][vg]));
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, o));
            const float vup = vmax > 0.0f ? 16384.0f / vmax : 0.0f;
            vdown_g[gi] = vmax * (1.0f / 16384.0f);
#pragma unroll
            for (int ks = 0; ks < CH / 16; ++ks) {
                const int cA = ks * 16 + 2 * tig, cB = cA + 8;
                const float w0 = S.vs[cA][vg] * vup, w1 = S.vs[cA + 1][vg] * vup, w2 = S.vs[cB][vg] * vup,
                            w3 = S.vs[cB + 1][vg] * vup;
                const float p00 = S.s[gid][cA] * w0, p01 = S.s[gid][cA + 1] * w1;
                const float p10 = S.s[gid + 8][cA] * w0, p11 = S.s[gid + 8][cA + 1] * w1;
                const float p02 = S.s[gid][cB] * w2, p03 = S.s[gid][cB + 1] * w3;
                const float p12 = S.s[gid + 8][cB] * w2, p13 = S.s[gid + 8][cB + 1] * w3;
                uint32_t ah[4], al[4];
                ah[0] = pack_h2(p00, p01);
                ah[1] = pack_h2(p10, p11);
                ah[2] = pack_h2(p02, p03);
                ah[3] = pack_h2(p12, p13);
                {
                    const __half2* h = reinterpret_cast<const __half2*>(ah);
                    float2 f;
                    f = __half22float2(h[0]); al[0] = pack_h2(p00 - f.x, p01 - f.y);
                    f = __half22float2(h[1]); al[1] = pack_h2(p10 - f.x, p11 - f.y);
                    f = __half22float2(h[2]); al[2] = pack_h2(p02 - f.x, p03 - f.y);
                    f = __half22float2(h[3]); al[3] = pack_h2(p12 - f.x, p13 - f.y);
                }
#pragma unroll
                for (int jj = 0; jj < JPG; ++jj) {
                    const int j = gi * JPG + jj;
                    const int d = warp * 64 + j * 8 + gid;
                    uint32_t b[2];
                    if constexpr (KV_MODE == 1 || KV_MODE == 4) {
                        const uint32_t x0 = (uint8_t) S.kv.v[cA][d] | ((uint32_t) (uint8_t) S.kv.v[cA + 1][d] << 8);
                        const uint32_t x1 = (uint8_t) S.kv.v[cB][d] | ((uint32_t) (uint8_t) S.kv.v[cB + 1][d] << 8);
                        b[0] = i8x2_to_h2(x0);
                        b[1] = i8x2_to_h2(x1);
                    } else {
                        const __half2 h0 = __halves2half2(S.kv.v[cA][d], S.kv.v[cA + 1][d]);
                        const __half2 h1 = __halves2half2(S.kv.v[cB][d], S.kv.v[cB + 1][d]);
                        b[0] = *reinterpret_cast<const uint32_t*>(&h0);
                        b[1] = *reinterpret_cast<const uint32_t*>(&h1);
                    }
                    mma16816(tmp[j], ah, b);
#ifndef D1_NO_PLO
                    mma16816(tmp[j], al, b);
#endif
                }
            }
            }
            const float a0 = S.alpha[gid], a1 = S.alpha[gid + 8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float vdown = vdown_g[j / JPG];
                acc[j][0] = fmaf(acc[j][0], a0, tmp[j][0] * vdown);
                acc[j][1] = fmaf(acc[j][1], a0, tmp[j][1] * vdown);
                acc[j][2] = fmaf(acc[j][2], a1, tmp[j][2] * vdown);
                acc[j][3] = fmaf(acc[j][3], a1, tmp[j][3] * vdown);
            }
        }
    }
    __syncthreads();
    const float l0 = S.lsum[gid], l1 = S.lsum[gid + 8];
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int d = warp * 64 + j * 8 + 2 * tig;
        *reinterpret_cast<float2*>(attn + (size_t) gid * HD + d) = make_float2(acc[j][0] * i0, acc[j][1] * i0);
        if (gid + 8 < G)
            *reinterpret_cast<float2*>(attn + (size_t) (gid + 8) * HD + d) = make_float2(acc[j][2] * i1, acc[j][3] * i1);
    }
}

// ---- v2 (int8 KV): warp w owns dims [64w, 64w+64) for both q.k and p.v, which is also int8 scale group w. So a
// warp needs only its own 64-byte slice of each K and V row: it gathers it itself with cp.async into its own
// double-buffered stage while it computes the previous chunk, and q stays in registers. Only the q.k partial sums
// (one per dim group) cross warps, and they are added in a fixed order: the result is deterministic.
constexpr int CH2 = 32;

struct Smem2 {
    int8_t kv[2][4][2][CH2][64];   // stage, warp, K/V, cell, 64 dims in 16-byte pieces XOR-swizzled by the cell
    float sc[2][4][2][CH2];        // stage, warp, K/V scale of the cell for the warp's group
    float part[4][16][CH2 + 1];    // q.k per dim group
    float p[16][CH2 + 1];
    float qmax[4];
    float alpha[16];
    float lsum[16];
    float mrow[16];
};
// Turing (register staging): one stage - the next chunk's pieces wait in registers until this chunk's p.v is done -
// and the q.k partials of cell group nt (8 cells) in the 512 bytes of the warp's own K slice that held those cells
// (read by then), so a block takes ~19.7 KB and three fit a Turing SM (one did, at 45.6 KB: four warps on four
// schedulers, every latency exposed)
struct Smem2T {
    int8_t kv[1][4][2][CH2][64];
    float sc[1][4][2][CH2];
    float p[16][CH2 + 1];
    float qmax[4];
    float alpha[16];
    float lsum[16];
    float mrow[16];
};

__device__ __forceinline__ int swz(int cell, int byte) {   // byte offset of (cell, byte) in a stage slice
    return cell * 64 + ((((byte >> 4) ^ (cell >> 1)) & 3) << 4) + (byte & 15);
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
#if !STRATA_PA_SM80
    (void) smem; (void) gmem; (void) valid;
    __trap();   // not reached on Turing (the register-staged pipeline replaces it there)
#else
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(gmem), "r"(valid ? 16 : 0));
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if STRATA_PA_SM80
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
__device__ __forceinline__ void cp_async_wait1() {
#if STRATA_PA_SM80
    asm volatile("cp.async.wait_group 1;\n" ::);
#endif
}

__global__ void __launch_bounds__(THREADS) prompt_attn_i8_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                                 const int32_t* __restrict__ ids,
                                                                 const int32_t* __restrict__ steps, int n_kv_heads,
                                                                 int page_size, float scale_log2,
                                                                 float* __restrict__ attn, int cap) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
#if STRATA_PA_SM80
    Smem2& S = *reinterpret_cast<Smem2*>(smem_raw);
#define PA2_PART(w, r, c) S.part[w][r][c]
#else
    Smem2T& S = *reinterpret_cast<Smem2T*>(smem_raw);
    // the partial of (row r, cell c) of warp w: cell group c / 8's 512 bytes of w's K slice, [row][cell % 8]
#define PA2_PART(w, r, c) reinterpret_cast<const float*>(&S.kv[0][w][0][0][0])[((c) >> 3) * 128 + (r) * 8 + ((c) & 7)]
#endif
    const int qi = blockIdx.x, kvh = blockIdx.y;
    const int n_head = n_kv_heads * G;
    q += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    attn += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    ids += (size_t) qi * cap;
    const int n = __ldg(steps + (size_t) qi * kStepCount + kStepWidth);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int gid = lane >> 2, tig = lane & 3;
    const int dim0 = warp * 64;

    // q: the power-of-two prescale over all 12 heads (as v1), then this warp's 64 dims as hi/lo A fragments
    float qm = 0.0f;
    for (int i = t; i < G * HD; i += THREADS) qm = fmaxf(qm, fabsf(q[i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, o));
    if (lane == 0) S.qmax[warp] = qm;
    if (t < 16) { S.mrow[t] = -INFINITY; S.lsum[t] = 0.0f; }
    __syncthreads();
    qm = fmaxf(fmaxf(S.qmax[0], S.qmax[1]), fmaxf(S.qmax[2], S.qmax[3]));
    int qe = 0;
    if (qm > 0.0f) frexpf(qm, &qe);
    const float qup = ldexpf(1.0f, 14 - qe), qdown = ldexpf(scale_log2, qe - 14);
    uint32_t qh[4][4], ql[4][4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = gid + (r & 1) * 8, col = dim0 + kk * 16 + 2 * tig + (r >> 1) * 8;
            float2 x = make_float2(0.f, 0.f);
            if (row < G) x = *reinterpret_cast<const float2*>(q + (size_t) row * HD + col);
            x.x *= qup;
            x.y *= qup;
            const __half2 hi = __floats2half2_rn(x.x, x.y);
            const float2 hf = __half22float2(hi);
            const __half2 lo = __floats2half2_rn(x.x - hf.x, x.y - hf.y);
            qh[kk][r] = *reinterpret_cast<const uint32_t*>(&hi);
            ql[kk][r] = *reinterpret_cast<const uint32_t*>(&lo);
        }
    }

    // the chunk pipeline: cells two chunks ahead, their pool rows one chunk ahead, the data (cp.async) one ahead
    const int n_chunks = (n + CH2 - 1) / CH2;
    auto cell_of = [&](int c) -> int { return c < n ? __ldg(ids + c) : -1; };
    auto row_of = [&](int cell) -> long long {
        if (cell < 0) return -1;
        const long long page = (long long) __ldg(p.page_table + cell / page_size);
        return (page * n_kv_heads + kvh) * page_size + (cell % page_size);
    };
#if !STRATA_PA_SM80
    // Turing (no cp.async): the warp's next pieces wait in registers and are stored into their stage after this
    // chunk's p.v (store_stage) - the same bytes in the same places as the cp.async pipeline
    uint4 rk[4], rv[4];
    auto store_stage = [&](int st) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = lane + 32 * j, cell = idx >> 2, pc = idx & 3;
            *reinterpret_cast<uint4*>(&S.kv[st][warp][0][0][0] + swz(cell, pc * 16)) = rk[j];
            *reinterpret_cast<uint4*>(&S.kv[st][warp][1][0][0] + swz(cell, pc * 16)) = rv[j];
        }
    };
#endif
    auto issue = [&](long long r, int st, float& ksr, float& vsr) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = lane + 32 * j, cell = idx >> 2, pc = idx & 3;
            const long long rr = __shfl_sync(0xffffffffu, r, cell);
            const bool ok = rr >= 0;
            const size_t off = ok ? (size_t) rr * HD + dim0 + pc * 16 : 0;
#if STRATA_PA_SM80
            cp_async16(&S.kv[st][warp][0][0][0] + swz(cell, pc * 16), p.k_q + off, ok);
            cp_async16(&S.kv[st][warp][1][0][0] + swz(cell, pc * 16), p.v_q + off, ok);
#else
            (void) st;
            rk[j] = ok ? __ldg(reinterpret_cast<const uint4*>(p.k_q + off)) : make_uint4(0, 0, 0, 0);
            rv[j] = ok ? __ldg(reinterpret_cast<const uint4*>(p.v_q + off)) : make_uint4(0, 0, 0, 0);
#endif
        }
        ksr = r >= 0 ? __half2float(__ushort_as_half(__ldg(p.k_scale + r * (HD / KV_Q8_GROUP) + warp))) : 0.0f;
        vsr = r >= 0 ? __half2float(__ushort_as_half(__ldg(p.v_scale + r * (HD / KV_Q8_GROUP) + warp))) : 0.0f;
    };
    float ksn, vsn;
    issue(row_of(cell_of(lane)), 0, ksn, vsn);
    cp_async_commit();
#if !STRATA_PA_SM80
    store_stage(0);
#endif
    S.sc[0][warp][0][lane] = ksn;
    S.sc[0][warp][1][lane] = vsn;
    long long r_next = row_of(cell_of(CH2 + lane));
    int cell_next2 = cell_of(2 * CH2 + lane);

    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;

    for (int ci = 0; ci < n_chunks; ++ci) {
#if STRATA_PA_SM80
        const int st = ci & 1, nst = st ^ 1, c0 = ci * CH2;
#else
        const int st = 0, nst = 0, c0 = ci * CH2;
#endif
        const bool more = ci + 1 < n_chunks;
        if (more) issue(r_next, nst, ksn, vsn);
        cp_async_commit();
        r_next = row_of(cell_next2);
        cell_next2 = cell_of((ci + 3) * CH2 + lane);
        cp_async_wait1();
        __syncwarp();
        const int8_t* K = &S.kv[st][warp][0][0][0];
        const int8_t* V = &S.kv[st][warp][1][0][0];
        // q.k over this warp's 64 dims, times the cell's K scale for this group
#pragma unroll
        for (int nt = 0; nt < CH2 / 8; ++nt) {
            float tg[4] = {0.f, 0.f, 0.f, 0.f};
            const int cell = nt * 8 + gid;
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                uint32_t b[2];
                b[0] = i8x2_to_h2(*reinterpret_cast<const uint16_t*>(K + swz(cell, kk * 16 + 2 * tig)));
                b[1] = i8x2_to_h2(*reinterpret_cast<const uint16_t*>(K + swz(cell, kk * 16 + 2 * tig + 8)));
                mma16816(tg, qh[kk], b);
                mma16816(tg, ql[kk], b);
            }
            const int c = nt * 8 + 2 * tig;
            const float s0 = S.sc[st][warp][0][c], s1 = S.sc[st][warp][0][c + 1];
#if STRATA_PA_SM80
            S.part[warp][gid][c] = tg[0] * s0;
            S.part[warp][gid][c + 1] = tg[1] * s1;
            S.part[warp][gid + 8][c] = tg[2] * s0;
            S.part[warp][gid + 8][c + 1] = tg[3] * s1;
#else
            __syncwarp();   // every lane has read the group's cells: their 512 bytes take its partials
            float* pp = reinterpret_cast<float*>(&S.kv[0][warp][0][0][0]) + nt * 128;
            pp[gid * 8 + 2 * tig] = tg[0] * s0;
            pp[gid * 8 + 2 * tig + 1] = tg[1] * s1;
            pp[(gid + 8) * 8 + 2 * tig] = tg[2] * s0;
            pp[(gid + 8) * 8 + 2 * tig + 1] = tg[3] * s1;
#endif
        }
        __syncthreads();
        // online softmax over the four groups' sum (fixed order): row t/8, 4 cells per thread
        {
            const int r = t >> 3, sub = t & 7;
            float x[4], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = sub * 4 + j;
                x[j] = c0 + c < n ? (((PA2_PART(0, r, c) + PA2_PART(1, r, c)) + PA2_PART(2, r, c)) + PA2_PART(3, r, c)) * qdown
                                  : -INFINITY;
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : exp2f(x[j] - m_new);
                S.p[r][sub * 4 + j] = e;
                sum += e;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            __syncwarp();
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : exp2f(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = fmaf(S.lsum[r], a, sum);
                S.mrow[r] = m_new;
            }
        }
        __syncthreads();
        // p.v over this warp's 64 dims (as v1)
        {
            float vmax = S.sc[st][warp][1][lane];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, o));
            const float vup = vmax > 0.0f ? 16384.0f / vmax : 0.0f, vdown = vmax * (1.0f / 16384.0f);
            float tmp[8][4];
#pragma unroll
            for (int j = 0; j < 8; ++j) tmp[j][0] = tmp[j][1] = tmp[j][2] = tmp[j][3] = 0.0f;
#pragma unroll
            for (int ks = 0; ks < CH2 / 16; ++ks) {
                const int cA = ks * 16 + 2 * tig, cB = cA + 8;
                const float w0 = S.sc[st][warp][1][cA] * vup, w1 = S.sc[st][warp][1][cA + 1] * vup,
                            w2 = S.sc[st][warp][1][cB] * vup, w3 = S.sc[st][warp][1][cB + 1] * vup;
                const float p00 = S.p[gid][cA] * w0, p01 = S.p[gid][cA + 1] * w1;
                const float p10 = S.p[gid + 8][cA] * w0, p11 = S.p[gid + 8][cA + 1] * w1;
                const float p02 = S.p[gid][cB] * w2, p03 = S.p[gid][cB + 1] * w3;
                const float p12 = S.p[gid + 8][cB] * w2, p13 = S.p[gid + 8][cB + 1] * w3;
                uint32_t ah[4], al[4];
                ah[0] = pack_h2(p00, p01);
                ah[1] = pack_h2(p10, p11);
                ah[2] = pack_h2(p02, p03);
                ah[3] = pack_h2(p12, p13);
                {
                    const __half2* h = reinterpret_cast<const __half2*>(ah);
                    float2 f;
                    f = __half22float2(h[0]); al[0] = pack_h2(p00 - f.x, p01 - f.y);
                    f = __half22float2(h[1]); al[1] = pack_h2(p10 - f.x, p11 - f.y);
                    f = __half22float2(h[2]); al[2] = pack_h2(p02 - f.x, p03 - f.y);
                    f = __half22float2(h[3]); al[3] = pack_h2(p12 - f.x, p13 - f.y);
                }
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int d = j * 8 + gid;
                    const uint32_t x0 = (uint8_t) V[swz(cA, d)] | ((uint32_t) (uint8_t) V[swz(cA + 1, d)] << 8);
                    const uint32_t x1 = (uint8_t) V[swz(cB, d)] | ((uint32_t) (uint8_t) V[swz(cB + 1, d)] << 8);
                    uint32_t b[2];
                    b[0] = i8x2_to_h2(x0);
                    b[1] = i8x2_to_h2(x1);
                    mma16816(tmp[j], ah, b);
                    mma16816(tmp[j], al, b);
                }
            }
            const float a0 = S.alpha[gid], a1 = S.alpha[gid + 8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[j][0] = fmaf(acc[j][0], a0, tmp[j][0] * vdown);
                acc[j][1] = fmaf(acc[j][1], a0, tmp[j][1] * vdown);
                acc[j][2] = fmaf(acc[j][2], a1, tmp[j][2] * vdown);
                acc[j][3] = fmaf(acc[j][3], a1, tmp[j][3] * vdown);
            }
        }
        if (more) {
            S.sc[nst][warp][0][lane] = ksn;
            S.sc[nst][warp][1][lane] = vsn;
#if !STRATA_PA_SM80
            store_stage(nst);
#endif
        }
        __syncwarp();
    }
    __syncthreads();
    const float l0 = S.lsum[gid], l1 = S.lsum[gid + 8];
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int d = dim0 + j * 8 + 2 * tig;
        *reinterpret_cast<float2*>(attn + (size_t) gid * HD + d) = make_float2(acc[j][0] * i0, acc[j][1] * i0);
        if (gid + 8 < G)
            *reinterpret_cast<float2*>(attn + (size_t) (gid + 8) * HD + d) = make_float2(acc[j][2] * i1, acc[j][3] * i1);
    }
}

#undef PA2_PART
__global__ void __launch_bounds__(THREADS) decode_attn_i8_split_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                                 const int32_t* __restrict__ ids,
                                                                 const int32_t* __restrict__ steps, int n_kv_heads,
                                                                 int page_size, float scale_log2,
                                                                 float* __restrict__ part, int cap) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem2& S = *reinterpret_cast<Smem2*>(smem_raw);
    const int qi = blockIdx.x, kvh = blockIdx.y, sp = blockIdx.z, nsp = gridDim.z;
    const int n_head = n_kv_heads * G;
    q += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    // partials of (query, split): acc [n_head][HD] unnormalized, then m / l [n_head][2]
    float* pacc = part + (size_t) (qi * nsp + sp) * n_head * (HD + 2) + (size_t) kvh * G * HD;
    float* pml = part + (size_t) (qi * nsp + sp) * n_head * (HD + 2) + (size_t) n_head * HD + (size_t) kvh * G * 2;
    ids += (size_t) qi * cap;
    const int n = __ldg(steps + (size_t) qi * kStepCount + kStepWidth);
    // this split's chunks: [ci0, ci1) of the query's ceil(n / CH2)
    const int n_all = (n + CH2 - 1) / CH2, per = (n_all + nsp - 1) / nsp;
    const int ci0 = sp * per, ci1 = min(n_all, ci0 + per);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int gid = lane >> 2, tig = lane & 3;
    const int dim0 = warp * 64;

    // q: the power-of-two prescale over all 12 heads (as v1), then this warp's 64 dims as hi/lo A fragments
    float qm = 0.0f;
    for (int i = t; i < G * HD; i += THREADS) qm = fmaxf(qm, fabsf(q[i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, o));
    if (lane == 0) S.qmax[warp] = qm;
    if (t < 16) { S.mrow[t] = -INFINITY; S.lsum[t] = 0.0f; }
    __syncthreads();
    qm = fmaxf(fmaxf(S.qmax[0], S.qmax[1]), fmaxf(S.qmax[2], S.qmax[3]));
    int qe = 0;
    if (qm > 0.0f) frexpf(qm, &qe);
    const float qup = ldexpf(1.0f, 14 - qe), qdown = ldexpf(scale_log2, qe - 14);
    uint32_t qh[4][4], ql[4][4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = gid + (r & 1) * 8, col = dim0 + kk * 16 + 2 * tig + (r >> 1) * 8;
            float2 x = make_float2(0.f, 0.f);
            if (row < G) x = *reinterpret_cast<const float2*>(q + (size_t) row * HD + col);
            x.x *= qup;
            x.y *= qup;
            const __half2 hi = __floats2half2_rn(x.x, x.y);
            const float2 hf = __half22float2(hi);
            const __half2 lo = __floats2half2_rn(x.x - hf.x, x.y - hf.y);
            qh[kk][r] = *reinterpret_cast<const uint32_t*>(&hi);
            ql[kk][r] = *reinterpret_cast<const uint32_t*>(&lo);
        }
    }

    if (ci0 >= ci1) {   // an empty split: m = -inf, l = 0, acc = 0
        for (int i = t; i < G * HD; i += THREADS) pacc[(size_t) (i / HD) * HD + i % HD] = 0.0f;
        if (t < G) { pml[2 * t] = -INFINITY; pml[2 * t + 1] = 0.0f; }
        return;
    }
    // the chunk pipeline: cells two chunks ahead, their pool rows one chunk ahead, the data (cp.async) one ahead
    auto cell_of = [&](int c) -> int { return c < n ? __ldg(ids + c) : -1; };
    auto row_of = [&](int cell) -> long long {
        if (cell < 0) return -1;
        const long long page = (long long) __ldg(p.page_table + cell / page_size);
        return (page * n_kv_heads + kvh) * page_size + (cell % page_size);
    };
    auto issue = [&](long long r, int st, float& ksr, float& vsr) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = lane + 32 * j, cell = idx >> 2, pc = idx & 3;
            const long long rr = __shfl_sync(0xffffffffu, r, cell);
            const bool ok = rr >= 0;
            const size_t off = ok ? (size_t) rr * HD + dim0 + pc * 16 : 0;
            cp_async16(&S.kv[st][warp][0][0][0] + swz(cell, pc * 16), p.k_q + off, ok);
            cp_async16(&S.kv[st][warp][1][0][0] + swz(cell, pc * 16), p.v_q + off, ok);
        }
        ksr = r >= 0 ? __half2float(__ushort_as_half(__ldg(p.k_scale + r * (HD / KV_Q8_GROUP) + warp))) : 0.0f;
        vsr = r >= 0 ? __half2float(__ushort_as_half(__ldg(p.v_scale + r * (HD / KV_Q8_GROUP) + warp))) : 0.0f;
    };
    float ksn, vsn;
    issue(row_of(cell_of(ci0 * CH2 + lane)), 0, ksn, vsn);
    cp_async_commit();
    S.sc[0][warp][0][lane] = ksn;
    S.sc[0][warp][1][lane] = vsn;
    long long r_next = row_of(cell_of((ci0 + 1) * CH2 + lane));
    int cell_next2 = cell_of((ci0 + 2) * CH2 + lane);

    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;

    for (int ci = ci0; ci < ci1; ++ci) {
        const int st = (ci - ci0) & 1, c0 = ci * CH2;
        const bool more = ci + 1 < ci1;
        if (more) issue(r_next, st ^ 1, ksn, vsn);
        cp_async_commit();
        r_next = row_of(cell_next2);
        cell_next2 = cell_of((ci + 3) * CH2 + lane);
        cp_async_wait1();
        __syncwarp();
        const int8_t* K = &S.kv[st][warp][0][0][0];
        const int8_t* V = &S.kv[st][warp][1][0][0];
        // q.k over this warp's 64 dims, times the cell's K scale for this group
#pragma unroll
        for (int nt = 0; nt < CH2 / 8; ++nt) {
            float tg[4] = {0.f, 0.f, 0.f, 0.f};
            const int cell = nt * 8 + gid;
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                uint32_t b[2];
                b[0] = i8x2_to_h2(*reinterpret_cast<const uint16_t*>(K + swz(cell, kk * 16 + 2 * tig)));
                b[1] = i8x2_to_h2(*reinterpret_cast<const uint16_t*>(K + swz(cell, kk * 16 + 2 * tig + 8)));
                mma16816(tg, qh[kk], b);
                mma16816(tg, ql[kk], b);
            }
            const int c = nt * 8 + 2 * tig;
            const float s0 = S.sc[st][warp][0][c], s1 = S.sc[st][warp][0][c + 1];
            S.part[warp][gid][c] = tg[0] * s0;
            S.part[warp][gid][c + 1] = tg[1] * s1;
            S.part[warp][gid + 8][c] = tg[2] * s0;
            S.part[warp][gid + 8][c + 1] = tg[3] * s1;
        }
        __syncthreads();
        // online softmax over the four groups' sum (fixed order): row t/8, 4 cells per thread
        {
            const int r = t >> 3, sub = t & 7;
            float x[4], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = sub * 4 + j;
                x[j] = c0 + c < n ? (((S.part[0][r][c] + S.part[1][r][c]) + S.part[2][r][c]) + S.part[3][r][c]) * qdown
                                  : -INFINITY;
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : exp2f(x[j] - m_new);
                S.p[r][sub * 4 + j] = e;
                sum += e;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            __syncwarp();
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : exp2f(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = fmaf(S.lsum[r], a, sum);
                S.mrow[r] = m_new;
            }
        }
        __syncthreads();
        // p.v over this warp's 64 dims (as v1)
        {
            float vmax = S.sc[st][warp][1][lane];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, o));
            const float vup = vmax > 0.0f ? 16384.0f / vmax : 0.0f, vdown = vmax * (1.0f / 16384.0f);
            float tmp[8][4];
#pragma unroll
            for (int j = 0; j < 8; ++j) tmp[j][0] = tmp[j][1] = tmp[j][2] = tmp[j][3] = 0.0f;
#pragma unroll
            for (int ks = 0; ks < CH2 / 16; ++ks) {
                const int cA = ks * 16 + 2 * tig, cB = cA + 8;
                const float w0 = S.sc[st][warp][1][cA] * vup, w1 = S.sc[st][warp][1][cA + 1] * vup,
                            w2 = S.sc[st][warp][1][cB] * vup, w3 = S.sc[st][warp][1][cB + 1] * vup;
                const float p00 = S.p[gid][cA] * w0, p01 = S.p[gid][cA + 1] * w1;
                const float p10 = S.p[gid + 8][cA] * w0, p11 = S.p[gid + 8][cA + 1] * w1;
                const float p02 = S.p[gid][cB] * w2, p03 = S.p[gid][cB + 1] * w3;
                const float p12 = S.p[gid + 8][cB] * w2, p13 = S.p[gid + 8][cB + 1] * w3;
                uint32_t ah[4], al[4];
                ah[0] = pack_h2(p00, p01);
                ah[1] = pack_h2(p10, p11);
                ah[2] = pack_h2(p02, p03);
                ah[3] = pack_h2(p12, p13);
                {
                    const __half2* h = reinterpret_cast<const __half2*>(ah);
                    float2 f;
                    f = __half22float2(h[0]); al[0] = pack_h2(p00 - f.x, p01 - f.y);
                    f = __half22float2(h[1]); al[1] = pack_h2(p10 - f.x, p11 - f.y);
                    f = __half22float2(h[2]); al[2] = pack_h2(p02 - f.x, p03 - f.y);
                    f = __half22float2(h[3]); al[3] = pack_h2(p12 - f.x, p13 - f.y);
                }
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int d = j * 8 + gid;
                    const uint32_t x0 = (uint8_t) V[swz(cA, d)] | ((uint32_t) (uint8_t) V[swz(cA + 1, d)] << 8);
                    const uint32_t x1 = (uint8_t) V[swz(cB, d)] | ((uint32_t) (uint8_t) V[swz(cB + 1, d)] << 8);
                    uint32_t b[2];
                    b[0] = i8x2_to_h2(x0);
                    b[1] = i8x2_to_h2(x1);
                    mma16816(tmp[j], ah, b);
                    mma16816(tmp[j], al, b);
                }
            }
            const float a0 = S.alpha[gid], a1 = S.alpha[gid + 8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[j][0] = fmaf(acc[j][0], a0, tmp[j][0] * vdown);
                acc[j][1] = fmaf(acc[j][1], a0, tmp[j][1] * vdown);
                acc[j][2] = fmaf(acc[j][2], a1, tmp[j][2] * vdown);
                acc[j][3] = fmaf(acc[j][3], a1, tmp[j][3] * vdown);
            }
        }
        if (more) {
            S.sc[st ^ 1][warp][0][lane] = ksn;
            S.sc[st ^ 1][warp][1][lane] = vsn;
        }
        __syncwarp();
    }
    __syncthreads();
    // the split's unnormalized sums and each head row's maximum and sum (the merge rescales and divides)
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int d = dim0 + j * 8 + 2 * tig;
        *reinterpret_cast<float2*>(pacc + (size_t) gid * HD + d) = make_float2(acc[j][0], acc[j][1]);
        if (gid + 8 < G)
            *reinterpret_cast<float2*>(pacc + (size_t) (gid + 8) * HD + d) = make_float2(acc[j][2], acc[j][3]);
    }
    if (t < G) { pml[2 * t] = S.mrow[t]; pml[2 * t + 1] = S.lsum[t]; }
}

// the splits of each (query, head) merged with the log-sum-exp rescale (base 2, as the kernel's online softmax), in
// split order; a block per (query, head), a thread per dimension
__global__ void __launch_bounds__(HD) decode_attn_merge_kernel(const float* __restrict__ part, int n_head, int nsp,
                                                               float* __restrict__ attn) {
    const int qi = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const size_t qs = (size_t) qi * nsp * n_head * (HD + 2);
    float M = -INFINITY;
    for (int s = 0; s < nsp; ++s) M = fmaxf(M, part[qs + (size_t) s * n_head * (HD + 2) + (size_t) n_head * HD + 2 * h]);
    float L = 0.0f, o = 0.0f;
    for (int s = 0; s < nsp; ++s) {
        const float* ps = part + qs + (size_t) s * n_head * (HD + 2);
        const float m = ps[(size_t) n_head * HD + 2 * h], l = ps[(size_t) n_head * HD + 2 * h + 1];
        const float w = m == -INFINITY ? 0.0f : exp2f(m - M);
        L = fmaf(l, w, L);
        o = fmaf(ps[(size_t) h * HD + d], w, o);
    }
    attn[((size_t) qi * n_head + h) * HD + d] = L > 0.0f ? o * (1.0f / L) : 0.0f;
}



bool launch_i8(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
               const QsaShapes& s, float* attn, int64_t n_q, cudaStream_t st) {
    static bool attr[64] = {};   // the shared-memory opt-in is per device (a layer split runs this on several)
    static int t75[64] = {};     // 1: Turing (the kernel's single-stage layout, Smem2T), 2: not
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) return false;
    if (!t75[dev]) {
        int major = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        t75[dev] = strata::cc_major_of(major) < 8 ? 1 : 2;   // STRATA_EMULATE_CC counts
    }
    const int bytes = (int) (t75[dev] == 1 ? sizeof(Smem2T) : sizeof(Smem2));
    if (!attr[dev]) {
        if (cudaFuncSetAttribute(prompt_attn_i8_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) !=
            cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        attr[dev] = true;
    }
    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        prompt_attn_i8_kernel<<<dim3((unsigned) nb, (unsigned) s.n_head_kv), THREADS, bytes, st>>>(
            q + q0 * s.n_head * HD, pools, ids + q0 * cap, steps + q0 * kStepCount, (int) s.n_head_kv,
            (int) s.page_size, scale_log2, attn + q0 * s.n_head * HD, (int) cap);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_prompt_attn_batch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}

// ---- v3 (int8 KV, opt-in STRATA_PROMPT_ATTN_IMMA=1): v2's block, pipeline and online softmax, with both products on
// INT8 tensor cores (mma m16n8k32 s8, int32 accumulation).  K and V enter as the int8 codes they are - no int8 -> fp16
// conversion - and the other operand as THREE 8-bit parts: q per (head, 64-dim group) and p * v_scale per (head, chunk)
// scaled to 127, then X = rint(x * 65536) split exactly into h (signed), l (signed) and m (unsigned) bytes - 24 bits,
// as fine as FP32 itself and finer than the FP16 hi + lo split v2 uses (~22 bits).  Each part's int32 sum is exact; the three meet in
// FP32 as h + (l * 256 + m) / 65536 (m unsigned: a u8 x s8 MMA).  INT8 MMA runs ~3.1x FP16's rate on sm_120 (732 vs 233 measured), so three INT8
// MMAs cost about one FP16 MMA where v2 spends two.  p.v takes each thread's 8 consecutive dims across the 8 n-tiles
// (n-tile j, column n = dim n * 8 + j), so a thread reads 8 bytes of a V row per cell and byte-transposes 4 cells.
__device__ __forceinline__ void imma16832(int* c, const uint32_t* a, const uint32_t* b) {
#if !STRATA_PA_SM80
    __trap();
#else
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}
// the low part's MMA: unsigned A (m in 0..255), signed B
__device__ __forceinline__ void imma16832_us(int* c, const uint32_t* a, const uint32_t* b) {
#if !STRATA_PA_SM80
    __trap();
#else
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}
// x (|x| <= 127) as X = rint(x * 65536) = h * 65536 + l * 256 + m exactly: h, l signed bytes, m an unsigned one
__device__ __forceinline__ void split3(float x, int& h, int& l, int& m) {
    const int X = __float2int_rn(x * 65536.0f);
    h = (X + 32768) >> 16;
    const int r = X - (h << 16);   // -32768 .. 32767
    l = r >> 8;                    // -128 .. 127
    m = r & 255;                   // 0 .. 255
}
__device__ __forceinline__ uint32_t pack4(int a, int b, int c, int d) {
    return (uint32_t) (a & 0xff) | ((uint32_t) (b & 0xff) << 8) | ((uint32_t) (c & 0xff) << 16) | ((uint32_t) (d & 0xff) << 24);
}
__device__ __forceinline__ float join3(int h, int l, int m) {   // the parts' exact int32 sums -> h + (l*256 + m)/65536
    return (float) h + (float) ((l << 8) + m) * (1.0f / 65536.0f);
}
// four words of four bytes each (word i = cell i) -> four words of the same byte position across the cells
__device__ __forceinline__ void tr4x4(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3, uint32_t* o) {
    const uint32_t t0 = __byte_perm(w0, w1, 0x5140), t1 = __byte_perm(w0, w1, 0x7362);
    const uint32_t t2 = __byte_perm(w2, w3, 0x5140), t3 = __byte_perm(w2, w3, 0x7362);
    o[0] = __byte_perm(t0, t2, 0x5410);
    o[1] = __byte_perm(t0, t2, 0x7632);
    o[2] = __byte_perm(t1, t3, 0x5410);
    o[3] = __byte_perm(t1, t3, 0x7632);
}

// v3's shared memory: ONE stage of K and one of V per warp (a warp gathers and reads only its own 64-dim slices, so
// reusing a buffer needs only that warp's own order): K(c + 1) is gathered while the block runs softmax(c) and
// p.v(c), V(c + 1) while it runs q.k(c + 1) and softmax(c + 1); and only the 12 real head rows of the partial sums
// and p.  25.5 KB instead of v2's 45.6 KB: three blocks an SM instead of two (one block an SM ran 1.54x slower than
// two - this kernel waits on latency, not on its MMAs).
struct Smem3 {
    int8_t kv[4][2][CH2][64];   // warp, K/V, cell, 64 dims in 16-byte pieces XOR-swizzled by the cell
    float sc[4][2][CH2];        // warp, K/V scale of the cell for the warp's group
    float part[4][G][CH2 + 1];  // q.k per dim group (the 12 head rows)
    float p[G][CH2 + 1];
    float alpha[16];
    float lsum[16];
    float mrow[16];
};

__global__ void __launch_bounds__(THREADS, 3) prompt_attn_i8v3_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                                      const int32_t* __restrict__ ids,
                                                                      const int32_t* __restrict__ steps, int n_kv_heads,
                                                                      int page_size, float scale_log2,
                                                                      float* __restrict__ attn, int cap) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem3& S = *reinterpret_cast<Smem3*>(smem_raw);
    const int qi = blockIdx.x, kvh = blockIdx.y;
    const int n_head = n_kv_heads * G;
    q += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    attn += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    ids += (size_t) qi * cap;
    const int n = __ldg(steps + (size_t) qi * kStepCount + kStepWidth);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int gid = lane >> 2, tig = lane & 3;
    const int dim0 = warp * 64;
    if (t < 16) { S.mrow[t] = -INFINITY; S.lsum[t] = 0.0f; S.alpha[t] = 0.0f; }

    // q: rows gid / gid + 8 (heads; 12..15 are pad), this warp's 64 dims as the A fragments of two k-steps of 32,
    // three 8-bit parts with a scale per (row, 64-dim group)
    uint32_t qa[3][2][4];
    float qinv[2];
    {
        float x[2][2][2][4];   // [row half][k-step][k half][4 dims]
        float mx[2] = {0.0f, 0.0f};
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
            const int row = gid + rh * 8;
#pragma unroll
            for (int ks = 0; ks < 2; ++ks)
#pragma unroll
                for (int kh = 0; kh < 2; ++kh) {
                    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                    if (row < G) v = *reinterpret_cast<const float4*>(q + (size_t) row * HD + dim0 + ks * 32 + kh * 16 + tig * 4);
                    x[rh][ks][kh][0] = v.x; x[rh][ks][kh][1] = v.y; x[rh][ks][kh][2] = v.z; x[rh][ks][kh][3] = v.w;
                    mx[rh] = fmaxf(mx[rh], fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
                }
#pragma unroll
            for (int o = 1; o < 4; o <<= 1) mx[rh] = fmaxf(mx[rh], __shfl_xor_sync(0xffffffffu, mx[rh], o));
        }
        float up[2];
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
            up[rh] = mx[rh] > 0.0f ? 127.0f / mx[rh] : 0.0f;
            qinv[rh] = mx[rh] * (1.0f / 127.0f);
        }
#pragma unroll
        for (int ks = 0; ks < 2; ++ks)
#pragma unroll
            for (int r = 0; r < 4; ++r) {   // a0: row gid, k 0..15 half; a1: row gid + 8; a2: row gid, k 16..31; a3
                const int rh = r & 1, kh = r >> 1;
                int hh[4], ll[4], mm[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) split3(x[rh][ks][kh][e] * up[rh], hh[e], ll[e], mm[e]);
                qa[0][ks][r] = pack4(hh[0], hh[1], hh[2], hh[3]);
                qa[1][ks][r] = pack4(ll[0], ll[1], ll[2], ll[3]);
                qa[2][ks][r] = pack4(mm[0], mm[1], mm[2], mm[3]);
            }
    }

    const int n_chunks = (n + CH2 - 1) / CH2;
    auto cell_of = [&](int c) -> int { return c < n ? __ldg(ids + c) : -1; };
    auto row_of = [&](int cell) -> long long {
        if (cell < 0) return -1;
        const long long page = (long long) __ldg(p.page_table + cell / page_size);
        return (page * n_kv_heads + kvh) * page_size + (cell % page_size);
    };
    // this warp's 64-dim slice of K (kv = 0) or V (1) of the chunk whose rows the lanes hold, and the cells' scales
    auto issue = [&](long long r, int kv) {
        const int8_t* pool = kv == 0 ? p.k_q : p.v_q;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = lane + 32 * j, cell = idx >> 2, pc = idx & 3;
            const long long rr = __shfl_sync(0xffffffffu, r, cell);
            const bool ok = rr >= 0;
            const size_t off = ok ? (size_t) rr * HD + dim0 + pc * 16 : 0;
            cp_async16(&S.kv[warp][kv][0][0] + swz(cell, pc * 16), pool + off, ok);
        }
        const uint16_t* sc = kv == 0 ? p.k_scale : p.v_scale;
        S.sc[warp][kv][lane] = r >= 0 ? __half2float(__ushort_as_half(__ldg(sc + r * (HD / KV_Q8_GROUP) + warp))) : 0.0f;
    };
    long long r_nxt = row_of(cell_of(lane));   // chunk 0
    issue(r_nxt, 0);
    cp_async_commit();
    issue(r_nxt, 1);
    cp_async_commit();
    r_nxt = row_of(cell_of(CH2 + lane));       // chunk 1
    int cell_nn = cell_of(2 * CH2 + lane);     // chunk 2
    __syncthreads();

    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;
    const int8_t* K = &S.kv[warp][0][0][0];
    const int8_t* V = &S.kv[warp][1][0][0];

    for (int ci = 0; ci < n_chunks; ++ci) {
        const int c0 = ci * CH2;
        const bool more = ci + 1 < n_chunks;
        cp_async_wait1();   // K(ci) is in (pending: V(ci))
        __syncwarp();
        // q.k over this warp's 64 dims: n-tile nt holds cells nt * 8 .. + 7 (column n = cell nt * 8 + n)
#pragma unroll
        for (int nt = 0; nt < CH2 / 8; ++nt) {
            int ch[4] = {0, 0, 0, 0}, cl[4] = {0, 0, 0, 0}, cm[4] = {0, 0, 0, 0};
            const int cell = nt * 8 + gid;
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                uint32_t b[2];
                b[0] = *reinterpret_cast<const uint32_t*>(K + swz(cell, ks * 32 + tig * 4));
                b[1] = *reinterpret_cast<const uint32_t*>(K + swz(cell, ks * 32 + 16 + tig * 4));
                imma16832(ch, qa[0][ks], b);
                imma16832(cl, qa[1][ks], b);
                imma16832_us(cm, qa[2][ks], b);
            }
            const int c = nt * 8 + 2 * tig;
            const float s0 = S.sc[warp][0][c], s1 = S.sc[warp][0][c + 1];
            S.part[warp][gid][c] = join3(ch[0], cl[0], cm[0]) * qinv[0] * s0;
            S.part[warp][gid][c + 1] = join3(ch[1], cl[1], cm[1]) * qinv[0] * s1;
            if (gid + 8 < G) {
                S.part[warp][gid + 8][c] = join3(ch[2], cl[2], cm[2]) * qinv[1] * s0;
                S.part[warp][gid + 8][c + 1] = join3(ch[3], cl[3], cm[3]) * qinv[1] * s1;
            }
        }
        __syncwarp();   // every lane has read K(ci) and its scales
        if (more) issue(r_nxt, 0);
        cp_async_commit();
        __syncthreads();
        // online softmax over the four groups' sum (fixed order): rows 0..11, 8 threads a row (warp 3 idles)
        if (t < G * 8) {
            const int r = t >> 3, sub = t & 7;
            float x[4], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = sub * 4 + j;
                x[j] = c0 + c < n ? (((S.part[0][r][c] + S.part[1][r][c]) + S.part[2][r][c]) + S.part[3][r][c]) * scale_log2
                                  : -INFINITY;
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : exp2f(x[j] - m_new);
                S.p[r][sub * 4 + j] = e;
                sum += e;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            __syncwarp();
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : exp2f(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = fmaf(S.lsum[r], a, sum);
                S.mrow[r] = m_new;
            }
        }
        __syncthreads();
        cp_async_wait1();   // V(ci) is in (pending: K(ci + 1))
        __syncwarp();
        // p.v over this warp's 64 dims: A = p * v_scale (rows gid / gid + 8, k = the chunk's 32 cells) in three parts
        // scaled per row; B = V (k = cells, n-tile j column n = dim n * 8 + j)
        {
            uint32_t pa[3][4];
            float pinv[2];
            {   // a scale per row: a chunk whose p are all small still gets its 24 bits (one per warp and chunk lost
                // ~3x of the accuracy against FP64)
                float a[4][4], mx[2] = {0.0f, 0.0f};
#pragma unroll
                for (int r = 0; r < 4; ++r) {   // a0: row gid, cells tig * 4 ..; a1: row gid + 8; a2 / a3: cells 16 + ...
                    const int row = gid + (r & 1) * 8, cb = (r >> 1) * 16 + tig * 4;
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        a[r][e] = row < G ? S.p[row][cb + e] * S.sc[warp][1][cb + e] : 0.0f;
                        mx[r & 1] = fmaxf(mx[r & 1], a[r][e]);
                    }
                }
                float up[2];
#pragma unroll
                for (int rh = 0; rh < 2; ++rh) {
#pragma unroll
                    for (int o = 1; o < 4; o <<= 1) mx[rh] = fmaxf(mx[rh], __shfl_xor_sync(0xffffffffu, mx[rh], o));
                    up[rh] = mx[rh] > 0.0f ? 127.0f / mx[rh] : 0.0f;
                    pinv[rh] = mx[rh] * (1.0f / 127.0f);
                }
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    int hh[4], ll[4], mm[4];
#pragma unroll
                    for (int e = 0; e < 4; ++e) split3(a[r][e] * up[r & 1], hh[e], ll[e], mm[e]);
                    pa[0][r] = pack4(hh[0], hh[1], hh[2], hh[3]);
                    pa[1][r] = pack4(ll[0], ll[1], ll[2], ll[3]);
                    pa[2][r] = pack4(mm[0], mm[1], mm[2], mm[3]);
                }
            }
            // B: this thread's 8 dims (gid * 8 .. + 7) of cells tig * 4 .. + 3 (b0) and 16 + tig * 4 .. + 3 (b1)
            uint32_t bv[2][8];   // [k half][dim j]
#pragma unroll
            for (int kh = 0; kh < 2; ++kh) {
                uint32_t lo[4], hi[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const uint2 w = *reinterpret_cast<const uint2*>(V + swz(kh * 16 + tig * 4 + e, gid * 8));
                    lo[e] = w.x;
                    hi[e] = w.y;
                }
                tr4x4(lo[0], lo[1], lo[2], lo[3], &bv[kh][0]);
                tr4x4(hi[0], hi[1], hi[2], hi[3], &bv[kh][4]);
            }
            __syncwarp();   // every lane has read V(ci) and its scales
            if (more) issue(r_nxt, 1);
            cp_async_commit();
            const float a0 = S.alpha[gid], a1 = S.alpha[gid + 8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                int ch[4] = {0, 0, 0, 0}, cl[4] = {0, 0, 0, 0}, cm[4] = {0, 0, 0, 0};
                const uint32_t b[2] = {bv[0][j], bv[1][j]};
                imma16832(ch, pa[0], b);
                imma16832(cl, pa[1], b);
                imma16832_us(cm, pa[2], b);
                acc[j][0] = fmaf(acc[j][0], a0, join3(ch[0], cl[0], cm[0]) * pinv[0]);
                acc[j][1] = fmaf(acc[j][1], a0, join3(ch[1], cl[1], cm[1]) * pinv[0]);
                acc[j][2] = fmaf(acc[j][2], a1, join3(ch[2], cl[2], cm[2]) * pinv[1]);
                acc[j][3] = fmaf(acc[j][3], a1, join3(ch[3], cl[3], cm[3]) * pinv[1]);
            }
        }
        r_nxt = row_of(cell_nn);
        cell_nn = cell_of((ci + 3) * CH2 + lane);
    }
    __syncthreads();
    const float l0 = S.lsum[gid], l1 = gid + 8 < G ? S.lsum[gid + 8] : 0.0f;
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {   // column 2 * tig is dim 16 * tig + j, column 2 * tig + 1 dim 16 * tig + 8 + j
        const int d0 = dim0 + 16 * tig + j, d1 = d0 + 8;
        attn[(size_t) gid * HD + d0] = acc[j][0] * i0;
        attn[(size_t) gid * HD + d1] = acc[j][1] * i0;
        if (gid + 8 < G) {
            attn[(size_t) (gid + 8) * HD + d0] = acc[j][2] * i1;
            attn[(size_t) (gid + 8) * HD + d1] = acc[j][3] * i1;
        }
    }
}

bool launch_i8v3(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                 const QsaShapes& s, float* attn, int64_t n_q, cudaStream_t st) {
    static bool attr[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    const int bytes = (int) sizeof(Smem3);
    if (dev < 0 || dev >= 64) return false;
    if (!attr[dev]) {
        if (cudaFuncSetAttribute(prompt_attn_i8v3_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) !=
            cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        // the shared-memory side of the L1 split as large as it goes, so three blocks fit an SM
        cudaFuncSetAttribute(prompt_attn_i8v3_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        cudaGetLastError();
        attr[dev] = true;
    }
    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        prompt_attn_i8v3_kernel<<<dim3((unsigned) nb, (unsigned) s.n_head_kv), THREADS, bytes, st>>>(
            q + q0 * s.n_head * HD, pools, ids + q0 * cap, steps + q0 * kStepCount, (int) s.n_head_kv,
            (int) s.page_size, scale_log2, attn + q0 * s.n_head * HD, (int) cap);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_prompt_attn_batch (v3): %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}

template <int KV_MODE, bool PF = false>
bool launch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
            const QsaShapes& s, float* attn, int64_t n_q, cudaStream_t st) {
    static bool attr[64] = {};   // per device, as above
    int dev = 0;
    cudaGetDevice(&dev);
    const int bytes = (int) sizeof(Smem<KV_MODE, PF>);
    if (dev < 0 || dev >= 64) return false;
    if (!attr[dev]) {
        if (cudaFuncSetAttribute(prompt_attn_kernel<KV_MODE, PF>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) !=
            cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        attr[dev] = true;
    }
    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        prompt_attn_kernel<KV_MODE, PF><<<dim3((unsigned) nb, (unsigned) s.n_head_kv), THREADS, bytes, st>>>(
            q + q0 * s.n_head * HD, pools, ids + q0 * cap, steps + q0 * kStepCount, (int) s.n_head_kv,
            (int) s.page_size, scale_log2, attn + q0 * s.n_head * HD, (int) cap);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_prompt_attn_batch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}


// PR #600: compiled only into the experimental build (-DSTRATA_EXPERIMENTAL_SM60=ON), the one build that runs on a
// Volta card; the ready-made engine has none of it (no extra kernels to load, the same code as before).
#if !defined(__HIPCC__) && defined(STRATA_EXPERIMENTAL_SM60)
// ---- sm_70 (Volta): v1's structure on mma.m8n8k4 ------------------------------------------------------------------
// Volta's tensor cores only have m8n8k4 (FP16 in, FP32 accumulate): per warp FOUR independent 8x8x4 products, one per
// quad-pair (QP q = lanes 4q..4q+3 and 4q+16..4q+19; thread j = (lane & 3) + 4 * (lane >> 4) within it). Layout, measured on
// a V100 (tools-free microtest, and equal to turbomind's SM70_MMA_884): the thread holds row j of A (k 0..3 as two
// registers) and column j of B (k 0..3); C register i is row (lane & 1) + 4 * (lane >> 4) + 2 * ((i >> 1) & 1),
// column (lane & 2) + 4 * (i >> 2) + (i & 1) of its QP's 8x8 tile.
//   q.k: warp w owns dims [64w, 64w+64) = int8 scale group w; QP q takes cells 8q..8q+7 of the 32-cell chunk; 16 k-steps
//        of 4 dims, two 8-row blocks (12 heads + 4 zero rows), q as hi + lo halves. The warp's partial (times the
//        cells' K scale) goes to S.part[w]; the softmax adds the four groups in a fixed order (deterministic).
//   p.v: warp w owns dims [64w, 64w+64) again; QP q takes the 16 dims 64w+16q.. (two n-tiles of 8); k = the 32 cells in
//        steps of 4; p carries the V scale (as v1) and is split into hi + lo halves.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
#define STRATA_PA_VOLTA 1
#else
#define STRATA_PA_VOLTA 0
#endif

__device__ __forceinline__ void mma884(float* c, uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
#if STRATA_PA_VOLTA
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
#else
    __trap();   // m8n8k4 is a Volta instruction; the host only launches this kernel on sm_70
#endif
}

template <int KV_MODE>
struct Smem70 {
    using KElem = typename std::conditional<KV_MODE == 0, __half, int8_t>::type;
    using VElem = typename std::conditional<KV_MODE == 1, int8_t, __half>::type;
    static constexpr int KROW = KV_MODE == 0 ? HD + 8 : HD + 16;
    static constexpr int VROW = KV_MODE == 1 ? HD + 16 : HD + 8;
    __half qh[16][QS];
    __half ql[16][QS];
    KElem k[CH][KROW];
    VElem v[CH][VROW];
    float ks[CH][4];
    float vs[CH][4];
    float part[4][16][CH + 4];   // q.k per dim group (scaled by the cell's K scale)
    float s[16][CH + 4];         // the chunk's probabilities
    float qmax[THREADS / 32];
    float alpha[16];
    float lsum[16];
    float mrow[16];
    long long row[2][CH];        // double-buffered pool rows: the next chunk's lookups hide under p.v
};

template <int KV_MODE>
__global__ void __launch_bounds__(THREADS) prompt_attn_v70_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                                  const int32_t* __restrict__ ids,
                                                                  const int32_t* __restrict__ steps, int n_kv_heads,
                                                                  int page_size, float scale_log2,
                                                                  float* __restrict__ attn, int cap) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem70<KV_MODE>& S = *reinterpret_cast<Smem70<KV_MODE>*>(smem_raw);
    const int qi = blockIdx.x, kvh = blockIdx.y;
    const int n_head = n_kv_heads * G;
    q += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    attn += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    ids += (size_t) qi * cap;
    const int n = __ldg(steps + (size_t) qi * kStepCount + kStepWidth);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int qp = (lane >> 2) & 3, jj = (lane & 3) + 4 * (lane >> 4);   // quad-pair, thread within it
    const int crow = (lane & 1) + 4 * (lane >> 4);                        // C row of register 0 (+2 per register pair)
    const int ccol = lane & 2;                                           // C column of register 0 (+4 per half, +1 per reg)

    // q: 12 heads + 4 zero rows, scaled by a power of two that puts its largest value near 2^14 (exact, and the
    // lo halves stay out of FP16's subnormal range), then split into hi + lo halves
    float qm = 0.0f;
    for (int i = t; i < G * HD; i += THREADS) qm = fmaxf(qm, fabsf(q[i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, o));
    if (lane == 0) S.qmax[warp] = qm;
    __syncthreads();
    qm = fmaxf(fmaxf(S.qmax[0], S.qmax[1]), fmaxf(S.qmax[2], S.qmax[3]));
    int qe = 0;
    if (qm > 0.0f) frexpf(qm, &qe);                 // qm < 2^qe
    const float qup = ldexpf(1.0f, 14 - qe), qdown = ldexpf(scale_log2, qe - 14);
    for (int i = t; i < 16 * HD; i += THREADS) {
        const int h = i / HD, d = i % HD;
        const float x = h < G ? q[(size_t) h * HD + d] * qup : 0.0f;
        const __half hi = __float2half_rn(x);
        S.qh[h][d] = hi;
        S.ql[h][d] = __float2half_rn(x - __half2float(hi));
    }
    if (t < 16) { S.mrow[t] = -INFINITY; S.lsum[t] = 0.0f; }

    float acc[2][2][8];
#pragma unroll
    for (int rb = 0; rb < 2; ++rb)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[rb][nt][i] = 0.0f;

    // the pool rows of a chunk: -1 past the selection and for a page the KV streaming left non-resident (the
    // decode kernel masks those, so this kernel does too).  Chunk 0's rows fill one buffer here; every later
    // chunk's lookups fill the other just before the previous p.v, where their global latency hides.
    int rbuf = 0;
    if (t < CH) {
        long long r = -1;
        if (t < min(CH, n)) {
            const int cell = ids[t];
            const long long page = (long long) p.page_table[cell / page_size];
            r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        S.row[0][t] = r;
    }
    for (int c0 = 0; c0 < n; c0 += CH, rbuf ^= 1) {
        const int nh = min(CH, n - c0);
        __syncthreads();   // rows ready; the previous chunk's p.v is done with k, v
        // gather the chunk's K and V rows (16-byte pieces; K8V4's V as q4_0 blocks dequantized to fp16)
        // and their scales
        {
            constexpr int KPIECES = HD * (int) sizeof(typename Smem<KV_MODE>::KElem) / 16;   // per K row
            constexpr int KPT = (CH * KPIECES + THREADS - 1) / THREADS;
            static_assert((CH * KPIECES) % THREADS == 0, "the staging pieces divide evenly");
            // issue every global load of a thread before any of its stores: they then overlap each other instead
            // of paying the HBM latency one at a time
            uint4 kx[KPT];
            int kc[KPT], kp[KPT];
#pragma unroll
            for (int s = 0; s < KPT; ++s) {
                const int i = t + s * THREADS;
                kc[s] = i / KPIECES; kp[s] = i % KPIECES;
                const long long r = S.row[rbuf][kc[s]];
                kx[s] = make_uint4(0, 0, 0, 0);
                if (r >= 0) {
                    if constexpr (KV_MODE == 0)
                        kx[s] = __ldg(reinterpret_cast<const uint4*>(p.k_pool + r * HD) + kp[s]);
                    else   // modes 1 and 3: the K side is INT8
                        kx[s] = __ldg(reinterpret_cast<const uint4*>(p.k_q + r * HD) + kp[s]);
                }
            }
#pragma unroll
            for (int s = 0; s < KPT; ++s)
                *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(&S.k[kc[s]][0]) + kp[s] * 16) = kx[s];
            if constexpr (KV_MODE == 3) {   // V: dequantize the row's q4_0 blocks straight into the fp16 V row
                constexpr int BLKS = HD / QK4_0;
                constexpr int BYTES = BLKS * (int) sizeof(block_q4_0);
                for (int i = t; i < CH * BLKS; i += THREADS) {
                    const int c = i / BLKS, b = i % BLKS;
                    const long long r = S.row[rbuf][c];
#pragma unroll
                    for (int j = 0; j < QK4_0; ++j) S.v[c][b * QK4_0 + j] = __half(0);
                    if (r >= 0) {
                        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + r * BYTES) + b;
                        const float d = __half2float(__ushort_as_half(blk->d));
#pragma unroll
                        for (int j = 0; j < QK4_0 / 2; ++j) {
                            S.v[c][b * QK4_0 + j] = __float2half_rn((float) ((int)(blk->qs[j] & 0x0F) - 8) * d);
                            S.v[c][b * QK4_0 + j + QK4_0 / 2] =
                                __float2half_rn((float) ((int)(blk->qs[j] >> 4) - 8) * d);
                        }
                    }
                }
            } else {
                constexpr int VPIECES = HD * (int) sizeof(typename Smem<KV_MODE>::VElem) / 16;   // per V row
                constexpr int VPT = (CH * VPIECES + THREADS - 1) / THREADS;
                static_assert((CH * VPIECES) % THREADS == 0, "the staging pieces divide evenly");
                uint4 vx[VPT];
                int vc[VPT], vp[VPT];
#pragma unroll
                for (int s = 0; s < VPT; ++s) {
                    const int i = t + s * THREADS;
                    vc[s] = i / VPIECES; vp[s] = i % VPIECES;
                    const long long r = S.row[rbuf][vc[s]];
                    vx[s] = make_uint4(0, 0, 0, 0);
                    if (r >= 0) {
                        if constexpr (KV_MODE == 1)
                            vx[s] = __ldg(reinterpret_cast<const uint4*>(p.v_q + r * HD) + vp[s]);
                        else
                            vx[s] = __ldg(reinterpret_cast<const uint4*>(p.v_pool + r * HD) + vp[s]);
                    }
                }
#pragma unroll
                for (int s = 0; s < VPT; ++s)
                    *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(&S.v[vc[s]][0]) + vp[s] * 16) = vx[s];
            }
            for (int i = t; i < CH * 4; i += THREADS) {
                const int c = i / 4, g = i % 4;
                const long long r = S.row[rbuf][c];
                float a = 0.0f, b = 0.0f;
                if (r >= 0) {
                    if constexpr (KV_MODE == 1) {
                        a = __half2float(__ushort_as_half(p.k_scale[r * (HD / KV_Q8_GROUP) + g]));
                        b = __half2float(__ushort_as_half(p.v_scale[r * (HD / KV_Q8_GROUP) + g]));
                    } else if constexpr (KV_MODE == 3) {   // K as int8, V dequantized to fp16 (scale 1)
                        a = __half2float(__ushort_as_half(p.k_scale[r * (HD / KV_Q8_GROUP) + g]));
                        b = 1.0f;
                    } else {
                        a = b = 1.0f;
                    }
                }
                S.ks[c][g] = a;
                S.vs[c][g] = b;
            }
        }
        __syncthreads();
        // scores: warp w = dim group w (64 dims), QP q = cells 8q..8q+7. The hi and lo halves of q accumulate in
        // separate m8n8k4 chains (tg, tgl), added in FP32 at the end: Volta's tensor cores truncate as they
        // accumulate, and in one chain behind the hi products most of the lo half was lost (error vs FP64 at 32K
        // 5.1e-6 -> 2.3e-6, the FP32 kernel's level). The same for p.v below.
        {
            float tg[2][8], tgl[2][8];
#pragma unroll
            for (int rb = 0; rb < 2; ++rb)
#pragma unroll
                for (int i = 0; i < 8; ++i) tg[rb][i] = tgl[rb][i] = 0.0f;
            const int cell_b = qp * 8 + jj;
#pragma unroll
            for (int ks = 0; ks < 16; ++ks) {
                const int d0 = warp * 64 + ks * 4;
                uint32_t b0, b1;
                if constexpr (KV_MODE != 0) {   // modes 1 and 3: the K side is INT8 codes, exact in FP16
                    const uint32_t raw = *reinterpret_cast<const uint32_t*>(&S.k[cell_b][d0]);
                    b0 = i8x2_to_h2(raw & 0xffffu);
                    b1 = i8x2_to_h2(raw >> 16);
                } else {
                    const uint2 raw = *reinterpret_cast<const uint2*>(&S.k[cell_b][d0]);
                    b0 = raw.x;
                    b1 = raw.y;
                }
#pragma unroll
                for (int rb = 0; rb < 2; ++rb) {
                    const uint2 ah = *reinterpret_cast<const uint2*>(&S.qh[rb * 8 + jj][d0]);
                    const uint2 al = *reinterpret_cast<const uint2*>(&S.ql[rb * 8 + jj][d0]);
                    mma884(tg[rb], ah.x, ah.y, b0, b1);
                    mma884(tgl[rb], al.x, al.y, b0, b1);
                }
            }
#pragma unroll
            for (int rb = 0; rb < 2; ++rb)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int row = rb * 8 + crow + 2 * ((i >> 1) & 1);
                    const int cell = qp * 8 + ccol + 4 * (i >> 2) + (i & 1);
                    S.part[warp][row][cell] = (tg[rb][i] + tgl[rb][i]) * S.ks[cell][warp];
                }
        }
        __syncthreads();
        // online softmax: row t/8, 4 cells per thread, 8 threads per row (lanes 8r..8r+7 of a warp)
        {
            constexpr int PER = CH / 8;
            const int r = t >> 3, sub = t & 7;
            float x[PER], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                const int c = sub * PER + j;
                const float sum4 = ((S.part[0][r][c] + S.part[1][r][c]) + S.part[2][r][c]) + S.part[3][r][c];
                x[j] = c < nh ? sum4 * qdown : -INFINITY;
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : exp2f(x[j] - m_new);
                S.s[r][sub * PER + j] = e;
                sum += e;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            __syncwarp();
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : exp2f(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = fmaf(S.lsum[r], a, sum);
                S.mrow[r] = m_new;
            }
        }
        __syncthreads();
        __syncthreads();
        // p.v: warp w owns dims [64w, 64w+64) (int8 scale group w), QP q its 16 dims 64w+16q..; the V scale is folded
        // into p relative to the chunk's largest, times 2^14 (as v1)
        {
            float vmax = 0.0f;
#pragma unroll
            for (int c = lane; c < CH; c += 32) vmax = fmaxf(vmax, S.vs[c][warp]);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, o));
            const float vup = vmax > 0.0f ? 16384.0f / vmax : 0.0f, vdown = vmax * (1.0f / 16384.0f);
            float tmp[2][2][8], tmpl[2][2][8];   // the hi and lo halves of p in separate chains (see the scores)
#pragma unroll
            for (int rb = 0; rb < 2; ++rb)
#pragma unroll
                for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                    for (int i = 0; i < 8; ++i) tmp[rb][nt][i] = tmpl[rb][nt][i] = 0.0f;
#pragma unroll
            for (int ks = 0; ks < CH / 4; ++ks) {
                const int c0 = ks * 4;
                float w[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) w[e] = S.vs[c0 + e][warp] * vup;
                uint2 ah[2], al[2];
#pragma unroll
                for (int rb = 0; rb < 2; ++rb) {
                    const float4 pv = *reinterpret_cast<const float4*>(&S.s[rb * 8 + jj][c0]);
                    const float p0 = pv.x * w[0], p1 = pv.y * w[1], p2 = pv.z * w[2], p3 = pv.w * w[3];
                    ah[rb].x = pack_h2(p0, p1);
                    ah[rb].y = pack_h2(p2, p3);
                    const float2 f01 = __half22float2(*reinterpret_cast<const __half2*>(&ah[rb].x));
                    const float2 f23 = __half22float2(*reinterpret_cast<const __half2*>(&ah[rb].y));
                    al[rb].x = pack_h2(p0 - f01.x, p1 - f01.y);
                    al[rb].y = pack_h2(p2 - f23.x, p3 - f23.y);
                }
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const int d = warp * 64 + qp * 16 + nt * 8 + jj;
                    uint32_t b0, b1;
                    if constexpr (KV_MODE == 1) {
                        const uint32_t x0 = (uint8_t) S.v[c0][d] | ((uint32_t) (uint8_t) S.v[c0 + 1][d] << 8);
                        const uint32_t x1 = (uint8_t) S.v[c0 + 2][d] | ((uint32_t) (uint8_t) S.v[c0 + 3][d] << 8);
                        b0 = i8x2_to_h2(x0);
                        b1 = i8x2_to_h2(x1);
                    } else {
                        const __half2 h0 = __halves2half2(S.v[c0][d], S.v[c0 + 1][d]);
                        const __half2 h1 = __halves2half2(S.v[c0 + 2][d], S.v[c0 + 3][d]);
                        b0 = *reinterpret_cast<const uint32_t*>(&h0);
                        b1 = *reinterpret_cast<const uint32_t*>(&h1);
                    }
#pragma unroll
                    for (int rb = 0; rb < 2; ++rb) {
                        mma884(tmp[rb][nt], ah[rb].x, ah[rb].y, b0, b1);
                        mma884(tmpl[rb][nt], al[rb].x, al[rb].y, b0, b1);
                    }
                }
            }
#pragma unroll
            for (int rb = 0; rb < 2; ++rb) {
                float a_[2];
                a_[0] = S.alpha[rb * 8 + crow];
                a_[1] = S.alpha[rb * 8 + crow + 2];
#pragma unroll
                for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                    for (int i = 0; i < 8; ++i)
                        acc[rb][nt][i] = fmaf(acc[rb][nt][i], a_[(i >> 1) & 1], (tmp[rb][nt][i] + tmpl[rb][nt][i]) * vdown);
            }
        }
        // the next chunk's pool rows, into the other buffer: the two global lookups per cell hide under p.v
        if (c0 + CH < n && t < CH) {
            long long r = -1;
            const int idx = c0 + CH + t;
            if (idx < n) {
                const int cell = ids[idx];
                const long long page = (long long) p.page_table[cell / page_size];
                r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
            }
            S.row[rbuf ^ 1][t] = r;
        }
    }
    __syncthreads();
#pragma unroll
    for (int rb = 0; rb < 2; ++rb)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = rb * 8 + crow + 2 * h;
            if (row >= G) continue;
            const float l = S.lsum[row];
            const float inv = l > 0.0f ? 1.0f / l : 0.0f;
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const int i = hh * 4 + h * 2;               // registers i, i+1: row h, columns ccol + 4 * hh + {0, 1}
                    const int d = warp * 64 + qp * 16 + nt * 8 + ccol + 4 * hh;
                    *reinterpret_cast<float2*>(attn + (size_t) row * HD + d) =
                        make_float2(acc[rb][nt][i] * inv, acc[rb][nt][i + 1] * inv);
                }
        }
}

template <int KV_MODE>
bool launch70(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
              const QsaShapes& s, float* attn, int64_t n_q, cudaStream_t st) {
    static bool attr[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    const int bytes = (int) sizeof(Smem70<KV_MODE>);
    if (dev < 0 || dev >= 64) return false;
    if (!attr[dev]) {
        if (cudaFuncSetAttribute(prompt_attn_v70_kernel<KV_MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) !=
            cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        attr[dev] = true;
    }
    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        prompt_attn_v70_kernel<KV_MODE><<<dim3((unsigned) nb, (unsigned) s.n_head_kv), THREADS, bytes, st>>>(
            q + q0 * s.n_head * HD, pools, ids + q0 * cap, steps + q0 * kStepCount, (int) s.n_head_kv,
            (int) s.page_size, scale_log2, attn + q0 * s.n_head * HD, (int) cap);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_prompt_attn_batch (sm_70): %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}
#else   // AMD, or a build without STRATA_EXPERIMENTAL_SM60: no m8n8k4 kernel (the caller keeps the old one)
template <int KV_MODE>
bool launch70(const float*, const QsaAttnPools&, const int32_t*, const int32_t*, int64_t, const QsaShapes&, float*,
              int64_t, cudaStream_t) {
    return false;
}
#endif  // !__HIPCC__ && STRATA_EXPERIMENTAL_SM60

#if defined(__HIPCC__)
// ---- S6: the int8-KV prompt attention on RDNA4 matrix cores (opt-in: STRATA_HIP_WMMA=1, gfx12 only). The design of
// the v2 kernel above with gfx12's v_wmma_f32_16x16x16_f16 (wave32) in place of m16n8k16: wave w owns dims
// [64w, 64w+64) (int8 scale group w) for q.k and p.v; q and p are split into FP16 hi + lo parts, the int8 codes enter
// exactly as FP16, the scales are applied in FP32 and the four groups' q.k partials are added in a fixed order.
// FP32-level accuracy, deterministic, but not bitwise equal to qsa_decode_attn_batch (another summation order).
// Fragment layout (16x16x16, wave32, checked on gfx1201): A lane l holds A[l % 16][(l / 16) * 8 + i], B lane l holds
// B[(l / 16) * 8 + i][l % 16], C/D lane l holds D[(l / 16) * 8 + i][l % 16], i = 0..7.
// gfx11 (RDNA3 / RDNA3.5: gfx110x, gfx115x) has the same instruction with another layout: A lane l holds all 16 k of
// row l % 16 (lanes 16..31 repeat lanes 0..15), B lane l all 16 k of column l % 16, and C/D lane l holds
// D[2 * i + l / 16][l % 16]. PA_KL is the k values per lane (8 or 16), PA_KOFF(half) the first one, PA_ROW the C row.
#if defined(__gfx1200__) || defined(__gfx1201__)
#define STRATA_PA_WMMA 1
#define STRATA_PA_GFX11 0
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || \
    defined(__gfx1151__)
#define STRATA_PA_WMMA 1
#define STRATA_PA_GFX11 1
#else
#define STRATA_PA_WMMA 0
#define STRATA_PA_GFX11 0
#endif
#if STRATA_PA_GFX11
constexpr int PA_KL = 16;
#define PA_KOFF(half) 0
#define PA_ROW(i, half) (2 * (i) + (half))
#else
constexpr int PA_KL = 8;
#define PA_KOFF(half) ((half) * 8)
#define PA_ROW(i, half) ((half) * 8 + (i))
#endif
typedef _Float16 wh8 __attribute__((ext_vector_type(PA_KL)));
typedef float wf8 __attribute__((ext_vector_type(8)));
constexpr int WCH = 32;          // cells per chunk (two 16-cell tiles)
constexpr int WVS = 80;          // staged V row stride in bytes (64 codes, 16-byte aligned, banks spread)
struct alignas(16) SmemW {
    float part[4][16][WCH + 1];  // q.k per dim group
    float p[16][WCH + 1];
    float vs[4][WCH];            // V scale per (group, cell)
    int valid[WCH];              // the cell has a resident pool row
    uint8_t v[4][WCH][WVS];      // per wave: the chunk's V codes of its group, one row per cell
    float qmax[4];
    float alpha[16];
    float lsum[16];
    float mrow[16];
};

__device__ __forceinline__ wf8 wmma_f16(wh8 a, wh8 b, wf8 c) {
#if STRATA_PA_WMMA && STRATA_PA_GFX11
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#elif STRATA_PA_WMMA
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
#else
    __builtin_trap();
    return c;
#endif
}

__device__ __forceinline__ wh8 i8_to_h(const uint32_t* x) {   // PA_KL codes, exact: |code| <= 128
    wh8 h;
#pragma unroll
    for (int i = 0; i < PA_KL; ++i) h[i] = (_Float16) (int) (int8_t) (x[i / 4] >> (8 * (i % 4)));
    return h;
}

// SPLIT = true (default): q and p carried as FP16 hi + lo pairs (two WMMAs each: FP32-level). S23 (opt-in
// STRATA_PA_FAST=1): SPLIT = false, one FP16 image each - half the matrix work, rounding-level (FP16 attention as the
// usual flash-attention kernels), quality-gated
template <bool SPLIT>
__global__ void __launch_bounds__(THREADS) prompt_attn_wmma_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                                   const int32_t* __restrict__ ids,
                                                                   const int32_t* __restrict__ steps, int n_kv_heads,
                                                                   int page_size, float scale_log2,
                                                                   float* __restrict__ attn, int cap) {
#if STRATA_PA_WMMA
    __shared__ SmemW S;
    const int qi = blockIdx.x, kvh = blockIdx.y;
    const int n_head = n_kv_heads * G;
    q += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    attn += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    ids += (size_t) qi * cap;
    const int n = __ldg(steps + (size_t) qi * kStepCount + kStepWidth);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int col = lane & 15, half = lane >> 4;   // fragment column / k half
    const int dim0 = warp * 64;

    float qm = 0.0f;
    for (int i = t; i < G * HD; i += THREADS) qm = fmaxf(qm, fabsf(q[i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, o));
    if (lane == 0) S.qmax[warp] = qm;
    if (t < 16) { S.mrow[t] = -INFINITY; S.lsum[t] = 0.0f; }
    __syncthreads();
    qm = fmaxf(fmaxf(S.qmax[0], S.qmax[1]), fmaxf(S.qmax[2], S.qmax[3]));
    int qe = 0;
    if (qm > 0.0f) frexpf(qm, &qe);
    const float qup = ldexpf(1.0f, 14 - qe), qdown = ldexpf(scale_log2, qe - 14);
    // q A fragments of this wave's 64 dims: row = col (heads 12..15 zero), k = dim0 + kk*16 + PA_KOFF(half) + i
    wh8 qh[4], ql[4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
        float x[PA_KL];
        if (col < G) {
            const float4* src = reinterpret_cast<const float4*>(q + (size_t) col * HD + dim0 + kk * 16 + PA_KOFF(half));
#pragma unroll
            for (int j = 0; j < PA_KL / 4; ++j) {
                const float4 a = src[j];
                x[4 * j] = a.x; x[4 * j + 1] = a.y; x[4 * j + 2] = a.z; x[4 * j + 3] = a.w;
            }
        } else {
#pragma unroll
            for (int i = 0; i < PA_KL; ++i) x[i] = 0.0f;
        }
#pragma unroll
        for (int i = 0; i < PA_KL; ++i) {
            const float v = x[i] * qup;
            const _Float16 hi = (_Float16) v;
            qh[kk][i] = hi;
            ql[kk][i] = (_Float16) (v - (float) hi);
        }
    }

    wf8 acc[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[j] = wf8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const int n_chunks = (n + WCH - 1) / WCH;
    // chunk ci's cell for this lane (lane = cell index): its pool row, or -1 (past the selection, or a block the KV
    // streaming left non-resident: masked as in the decode kernel)
    auto row_of = [&](int ci) -> long long {
        const int c0 = ci * WCH;
        long long r = -1;
        if (c0 + lane < n) {
            const int cell = __ldg(ids + c0 + lane);
            const long long page = (long long) __ldg(p.page_table + cell / page_size);
            if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        return r;
    };
    // a chunk's global reads: this wave's V slice (64 bytes of each row) and V scale, and the K fragments (8 or 16
    // consecutive codes per lane) and K scales of its two 16-cell tiles
    struct Chunk {
        long long row;
        uint4 v[4];
        float vsc, ksc[2];
        uint32_t kx[2][4][PA_KL / 4];
    };
    auto fetch = [&](long long row, Chunk& d) {
        d.row = row;
        d.v[0] = d.v[1] = d.v[2] = d.v[3] = make_uint4(0, 0, 0, 0);
        d.vsc = 0.0f;
        if (row >= 0) {
            const uint4* src = reinterpret_cast<const uint4*>(p.v_q + row * HD + dim0);
            d.v[0] = __ldg(src); d.v[1] = __ldg(src + 1); d.v[2] = __ldg(src + 2); d.v[3] = __ldg(src + 3);
            d.vsc = __half2float(__ushort_as_half(__ldg(p.v_scale + row * (HD / KV_Q8_GROUP) + warp)));
        }
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const long long rr = __shfl(row, nt * 16 + col);
            d.ksc[nt] = 0.0f;
            if (rr >= 0) d.ksc[nt] = __half2float(__ushort_as_half(__ldg(p.k_scale + rr * (HD / KV_Q8_GROUP) + warp)));
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
                for (int i = 0; i < PA_KL / 4; ++i) d.kx[nt][kk][i] = 0;
                if (rr >= 0) {
                    const uint32_t* src = reinterpret_cast<const uint32_t*>(p.k_q + rr * HD + dim0 + kk * 16 + PA_KOFF(half));
#if STRATA_PA_GFX11
                    const uint4 v = __ldg(reinterpret_cast<const uint4*>(src));
                    d.kx[nt][kk][0] = v.x; d.kx[nt][kk][1] = v.y; d.kx[nt][kk][2] = v.z; d.kx[nt][kk][3] = v.w;
#else
                    const uint2 v = __ldg(reinterpret_cast<const uint2*>(src));
                    d.kx[nt][kk][0] = v.x; d.kx[nt][kk][1] = v.y;
#endif
                }
            }
        }
    };
    auto chunk = [&](const Chunk& cd) {
        // stage this wave's V slice and scale
        {
            uint4* dst = reinterpret_cast<uint4*>(&S.v[warp][lane][0]);
            dst[0] = cd.v[0]; dst[1] = cd.v[1]; dst[2] = cd.v[2]; dst[3] = cd.v[3];
            S.vs[warp][lane] = cd.vsc;
            if (warp == 0) S.valid[lane] = cd.row >= 0;
        }
        // q.k over this wave's 64 dims for two 16-cell tiles, times the cell's K scale for this group
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            wf8 s = wf8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const wh8 b = i8_to_h(cd.kx[nt][kk]);
                s = wmma_f16(qh[kk], b, s);
                if constexpr (SPLIT) s = wmma_f16(ql[kk], b, s);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) S.part[warp][PA_ROW(i, half)][nt * 16 + col] = s[i] * cd.ksc[nt];
        }
        __syncthreads();
        // online softmax over the four groups' sum (fixed order): row t/8, 4 cells per thread
        {
            const int r = t >> 3, sub = t & 7;
            float x[4], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = sub * 4 + j;
                // past the selection, or a masked cell (non-resident page, as in the decode kernel): no weight
                x[j] = S.valid[c] ? (((S.part[0][r][c] + S.part[1][r][c]) + S.part[2][r][c]) + S.part[3][r][c]) * qdown
                                  : -INFINITY;
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : exp2f(x[j] - m_new);
                S.p[r][sub * 4 + j] = e;
                sum += e;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : exp2f(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = fmaf(S.lsum[r], a, sum);
                S.mrow[r] = m_new;
            }
        }
        __syncthreads();
        // p.v over this wave's 64 dims: 4 tiles of 16 dims, 2 k-steps of 16 cells
        {
            float vmax = S.vs[warp][lane];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, o));
            const float vup = vmax > 0.0f ? 16384.0f / vmax : 0.0f, vdown = vmax * (1.0f / 16384.0f);
            wf8 tmp[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) tmp[j] = wf8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                const int cb = ks * 16 + PA_KOFF(half);   // this lane's PA_KL cells
                wh8 ah, al;
#pragma unroll
                for (int i = 0; i < PA_KL; ++i) {
                    const float pv = S.p[col][cb + i] * (S.vs[warp][cb + i] * vup);
                    const _Float16 hi = (_Float16) pv;
                    ah[i] = hi;
                    al[i] = (_Float16) (pv - (float) hi);
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int d = j * 16 + col;
                    wh8 b;
#pragma unroll
                    for (int i = 0; i < PA_KL; ++i) b[i] = (_Float16) (int) (int8_t) S.v[warp][cb + i][d];
                    tmp[j] = wmma_f16(ah, b, tmp[j]);
                    if constexpr (SPLIT) tmp[j] = wmma_f16(al, b, tmp[j]);
                }
            }
            float a[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) a[i] = S.alpha[PA_ROW(i, half)];
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[j][i] = fmaf(acc[j][i], a[i], tmp[j][i] * vdown);
        }
        // no barrier here: `v`/`vs` are this wave's own, and `p`/`alpha`/`valid` are rewritten only after the next
        // chunk's first barrier (which every wave reaches after its p.v)
    };
    // all of a chunk's reads are issued before its compute (they were interleaved with the q.k WMMAs and the V
    // staging): 32K context, 2,048 queries, gfx1151, STRATA_PA_FAST=1: 12.49 -> 11.27 ms; the same bits
    for (int ci = 0; ci < n_chunks; ++ci) {
        Chunk d;
        fetch(row_of(ci), d);
        chunk(d);
    }
    float inv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float l = S.lsum[PA_ROW(i, half)];
        inv[i] = l > 0.0f ? 1.0f / l : 0.0f;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = PA_ROW(i, half);
            if (row < G) attn[(size_t) row * HD + dim0 + j * 16 + col] = acc[j][i] * inv[i];
        }
#else
    __builtin_trap();
#endif
}

// gfx12 (RDNA4): on by default on Linux (148-amd; STRATA_HIP_WMMA=0 turns it off; Windows: opt-in with =1 until tested there).  gfx11 (RDNA3 / RDNA3.5): only on request
// (STRATA_HIP_WMMA=1).  STRATA_HIP_WMMA=auto (151-amd) takes the matrix-core kernel on a gfx12 card for a batch of
// STRATA_HIP_WMMA_MIN_Q (64) or more queries and keeps the FP32 kernel for smaller calls.  The output differs from the
// old kernel's in its last bits (FP32-level accuracy, another summation order): teacher-forced KL against the old kernel
// 0.0003 (28 prompts) / 0.0014 (8 long documents) on an R9700.  A launch that fails switches the path off for the rest of
// the process (the caller then runs the FP32 kernel) instead of ending it.
int hip_wmma_mode_env() {   // -1 unset (the arch default), 0 off, 1 forced, 2 auto
    static const int mode = [] {
        const char* e = std::getenv("STRATA_HIP_WMMA");
        if (e == nullptr || e[0] == '\0') return -1;
        if (e[0] == '1') return 1;
        if (!std::strcmp(e, "auto")) return 2;
        return 0;
    }();
    return mode;
}
std::atomic<bool> g_wmma_failed{false};
// the device's class: 1 gfx11 matrix-core part, 2 gfx12 (RDNA4), 3 anything else
int hip_wmma_arch(int dev) {
    static int arch[64] = {};
    if (dev < 0 || dev >= 64) return 3;
    if (arch[dev] == 0) {
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, dev) != hipSuccess) { (void) hipGetLastError(); return 3; }
        arch[dev] = std::strncmp(prop.gcnArchName, "gfx12", 5) == 0 ? 2
                    : strata::kernels::gfx_arch_is_gfx11_wmma(prop.gcnArchName) ? 1 : 3;
        const int mode = hip_wmma_mode_env();
        static bool told = false;
        if (!told && (mode >= 1 || arch[dev] == 2)) {
            told = true;
            const bool on = (arch[dev] == 2 && (mode >= 1 || strata::core::kGfx12PromptDefaultsOn)) || (arch[dev] == 1 && mode >= 1);
            if (mode >= 1 || on)
                std::fprintf(stderr, "strata: %sthe prompt attention on matrix cores %s (%s)%s\n", mode >= 1 ? "STRATA_HIP_WMMA: " : "",
                             !on ? "unavailable" : (mode == 2 ? "chosen per call" : "on"), prop.gcnArchName,
                             mode >= 1 ? "" : " by default; STRATA_HIP_WMMA=0 turns it off");
            else
                std::fprintf(stderr, "strata: the prompt attention on matrix cores is off by default on Windows (opt-in until tested on Windows); "
                                     "STRATA_HIP_WMMA=1 turns it on (%s)\n", prop.gcnArchName);
        }
    }
    return arch[dev];
}
// auto: where the matrix-core kernel beats the FP32 one.  Measured on the R9700 (gfx1201) only; the gfx11 parts take
// it only when forced with =1.
bool hip_wmma_auto_pick(int arch, int64_t n_q, int64_t cap) {
    if (arch != 2) return false;
    static const int64_t min_q = [] { const char* e = std::getenv("STRATA_HIP_WMMA_MIN_Q"); return e ? (int64_t) std::atoll(e) : (int64_t) 64; }();
    static const int64_t min_cap = [] { const char* e = std::getenv("STRATA_HIP_WMMA_MIN_CAP"); return e ? (int64_t) std::atoll(e) : (int64_t) 0; }();
    return n_q >= min_q && cap >= min_cap;
}
bool hip_wmma_usable(int64_t n_q, int64_t cap) {
    const int mode = hip_wmma_mode_env();
    if (mode == 0 || g_wmma_failed.load(std::memory_order_relaxed)) return false;
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) { (void) hipGetLastError(); return false; }
    const int arch = hip_wmma_arch(dev);
    if (arch == 3) return false;
    if (mode == 1) return true;
    if (mode == 2) return hip_wmma_auto_pick(arch, n_q, cap);
    return arch == 2 && strata::core::kGfx12PromptDefaultsOn;   // unset: the gfx12 default (Linux; Windows is opt-in until tested there)
}

bool launch_wmma(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                 const QsaShapes& s, float* attn, int64_t n_q, cudaStream_t st) {
    {   // STRATA_QSA_STAT=1 (diagnostic): how much consecutive queries' selections overlap (the row-union attention's sharing factor)
        static const bool stat_on = std::getenv("STRATA_QSA_STAT") != nullptr;
        static int stat_calls = 0;
        if (stat_on && n_q >= 2048 && stat_calls < 4) {
            ++stat_calls;
            cudaStreamSynchronize(st);
            std::vector<int32_t> h((size_t) (n_q * cap)), hs((size_t) (n_q * kStepCount));
            cudaMemcpy(h.data(), ids, h.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(hs.data(), steps, hs.size() * 4, cudaMemcpyDeviceToHost);
            for (int R : {4, 8, 16, 32}) {
                double mean_n = 0, mean_u = 0; int64_t groups = 0;
                for (int64_t g0 = n_q / 2; g0 + R <= n_q; g0 += 4 * R) {   // the second half of the chunk (longer contexts)
                    std::vector<int32_t> all;
                    int64_t sum_n = 0;
                    for (int r = 0; r < R; ++r) {
                        const int n = hs[(size_t) (g0 + r) * kStepCount + kStepWidth];
                        sum_n += n;
                        all.insert(all.end(), h.begin() + (size_t) (g0 + r) * cap, h.begin() + (size_t) (g0 + r) * cap + n);
                    }
                    std::sort(all.begin(), all.end());
                    const int64_t u = std::unique(all.begin(), all.end()) - all.begin();
                    mean_n += (double) sum_n / R; mean_u += (double) u; ++groups;
                }
                if (groups) std::fprintf(stderr, "strata qsa stat: call %d n_q %lld cap %lld rows/group %d: mean selection %.0f, mean union %.0f (x%.2f; K/V reads per row %.2fx lower)\n",
                                         stat_calls, (long long) n_q, (long long) cap, R, mean_n / groups, mean_u / groups, mean_u / mean_n, R * mean_n / mean_u);
            }
        }
    }
    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        static const bool fast = [] { const char* e = std::getenv("STRATA_PA_FAST"); return e != nullptr && e[0] == '1'; }();
        static const int64_t min_t = [] { const char* e = std::getenv("STRATA_PF_SWITCH_MIN_T"); return e ? (int64_t) std::atoll(e) : (int64_t) 0; }();
        if (fast && n_q >= min_t)
            prompt_attn_wmma_kernel<false><<<dim3((unsigned) nb, (unsigned) s.n_head_kv), THREADS, 0, st>>>(
                q + q0 * s.n_head * HD, pools, ids + q0 * cap, steps + q0 * kStepCount, (int) s.n_head_kv,
                (int) s.page_size, scale_log2, attn + q0 * s.n_head * HD, (int) cap);
        else
            prompt_attn_wmma_kernel<true><<<dim3((unsigned) nb, (unsigned) s.n_head_kv), THREADS, 0, st>>>(
                q + q0 * s.n_head * HD, pools, ids + q0 * cap, steps + q0 * kStepCount, (int) s.n_head_kv,
                (int) s.page_size, scale_log2, attn + q0 * s.n_head * HD, (int) cap);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_prompt_attn_batch (wmma): %s; the FP32 prompt attention runs from here on\n", cudaGetErrorString(e));
        g_wmma_failed.store(true);
        return false;
    }
    return true;
}
#endif

int g_pa_imma = -1;   // STRATA_PROMPT_ATTN_IMMA: -1 not read yet; qsa_prompt_attn_set_imma (tests) overrides it
}  // namespace

// STRATA_QSA_SPLIT_ATTN=1 (the idea of fork Eddoursul fd23d91): the decode attention through the
// prompt path's int8 tensor-core kernel with each query's cells split among up to 20 blocks (one wave), merged with the
// log-sum-exp rescale.  int8 KV on sm_80+ only (false otherwise: the caller keeps qsa_decode_attn_batch).  The
// partials live in the decode scratch (n_q x splits x n_head x (HD + 2) floats, within qsa_decode_attn_scratch_floats).
bool qsa_decode_attn_split(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, uint64_t scratch_floats, float* attn, int64_t n_q,
                           void* stream) {
#if defined(__HIPCC__)
    return false;
#else
    if (n_q <= 0) return true;
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !ids || !steps || !pools.page_table ||
        pools.k_q == nullptr || pools.v_q == nullptr || pools.k_scale == nullptr || pools.v_scale == nullptr ||
        pools.k_q4 != nullptr || pools.v_q4 != nullptr)
        return false;
    static int cc[64] = {}, sms[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (cc[dev] == 0) {
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        cudaDeviceGetAttribute(&sms[dev], cudaDevAttrMultiProcessorCount, dev);
        cc[dev] = 10 * major + minor;
    }
    if (cc[dev] < 80) return false;
    static bool attr[64] = {};
    const int bytes = (int) sizeof(Smem2);
    if (!attr[dev]) {
        if (cudaFuncSetAttribute(decode_attn_i8_split_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        attr[dev] = true;
    }
    // splits: up to 20, about two blocks per SM in one wave, never more than the capacity's chunks (the scratch)
    const int64_t chunks_cap = (cap + CH2 - 1) / CH2;
    int64_t nsp = (2 * (int64_t) sms[dev] + n_q * s.n_head_kv - 1) / (n_q * s.n_head_kv);
    nsp = std::max<int64_t>(1, std::min<int64_t>({nsp, 20, chunks_cap}));
    if ((uint64_t) (n_q * nsp * s.n_head * (HD + 2)) > scratch_floats) return false;   // the caller's scratch
    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    cudaStream_t st = (cudaStream_t) stream;
    decode_attn_i8_split_kernel<<<dim3((unsigned) n_q, (unsigned) s.n_head_kv, (unsigned) nsp), THREADS, bytes, st>>>(
        q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size, scale_log2, scratch, (int) cap);
    decode_attn_merge_kernel<<<dim3((unsigned) n_q, (unsigned) s.n_head), HD, 0, st>>>(scratch, (int) s.n_head, (int) nsp, attn);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_decode_attn_split: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
#endif
}
void qsa_prompt_attn_set_imma(int on) { g_pa_imma = on < 0 ? -1 : (on ? 1 : 0); }
bool qsa_prompt_attn_matrix_cores_requested() {
#if defined(__HIPCC__)
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) { (void) hipGetLastError(); return false; }
    const int mode = hip_wmma_mode_env(), arch = hip_wmma_arch(dev);
    return mode != 0 && arch != 3 && !g_wmma_failed.load(std::memory_order_relaxed) && (mode >= 1 || (arch == 2 && strata::core::kGfx12PromptDefaultsOn));
#else
    return false;
#endif
}

bool qsa_hip_wmma_active() { return qsa_prompt_attn_matrix_cores_requested(); }

bool qsa_prompt_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return true;
    bool volta = false, turing = false, t75 = false;   // per call, from the CURRENT device (a layer split can mix Turing with newer cards)
    {   // sm_75 or newer: the MMA above compiles for both.  sm_80+ runs the cp.async kernel (launch_i8); Turing has
        // no cp.async, so it runs the v1 kernel (launch<1>, same accuracy, another summation order).  An older card
        // keeps the old kernel.
        // #371: the compute capability with its minor - sm_70 (V100) has no m16n8k8 (the kernels trap below sm_75)
#if defined(STRATA_HIP_GFX906)
        return false;   // gfx906: the tensor-core and WMMA kernels are not for it; the FP32 kernel runs
#endif
        static int cc[64] = {};
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
        if (cc[dev] == 0) {
            int major = 0, minor = 0;
            if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
                cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
                cudaGetLastError();
                return false;
            }
            // STRATA_QSA_WARP=1|attn (an A/B arm): the pre-sm_80 kernels on any card that has the Turing MMA (sm_75
            // and newer), as RTX 20 runs them.  A card below sm_75 has no m16n8k8: its builds hold a __trap() in those
            // kernels, so forcing them there killed the engine ("prefill copy_i32: unspecified launch failure" on a
            // Tesla P100); such a card ignores the switch and keeps the kernel it runs without it.
            const char* w = std::getenv("STRATA_QSA_WARP");
            const int real_cc = 10 * strata::cc_major_of(major) + strata::cc_minor_of(minor);
            cc[dev] = w && (!std::strcmp(w, "1") || !std::strcmp(w, "attn")) && real_cc >= 75 ? 75 : real_cc;
        }
        if (cc[dev] < 70) return false;
        volta = cc[dev] < 75;     // sm_70: the m8n8k4 kernel (STRATA_PA_VOLTA); STRATA_QSA_WARP=1 still forces the old one
        turing = cc[dev] < 80;
        t75 = cc[dev] == 75;      // a Turing card, or one run as such (STRATA_EMULATE_CC=75, STRATA_QSA_WARP=1)
    }
#if defined(__HIPCC__)
    // the tensor-core kernels are compiled out on AMD (its major version is not a CUDA sm); RDNA4 has its own int8-KV
    // matrix-core kernel, opt-in (STRATA_HIP_WMMA=1); everything else keeps the old kernel
    (void) turing; (void) volta;
    if (pools.k_q != nullptr && pools.v_q != nullptr && pools.k_scale != nullptr && pools.v_scale != nullptr &&
        pools.k_q4 == nullptr && pools.v_q4 == nullptr && s.head_dim == HD && s.n_head == (int64_t) G * s.n_head_kv &&
        cap > 0 && ids && steps && pools.page_table && hip_wmma_usable(n_q, cap))
        return launch_wmma(q, pools, ids, steps, cap, s, attn, n_q, (cudaStream_t) stream);
    return false;
#endif
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !ids || !steps || !pools.page_table)
        return false;
    cudaStream_t st = (cudaStream_t) stream;
    if (pools.k_q4 != nullptr) {   // Q4_0 K and V (--kv q4_0): mode 4.  STRATA_PROMPT_ATTN_Q4=0: the old kernel (A/B)
        static const bool q4_off = [] {
            const char* v = std::getenv("STRATA_PROMPT_ATTN_Q4");
            return v != nullptr && v[0] == '0';
        }();
        // Turing runs mode 4 as pairs of m16n8k8 MMAs (another summation order than the FP32 kernel it replaces): manual
        // opt-in STRATA_PROMPT_ATTN_Q4_TC=1 (#1581, c3c: qsa_prompt_attn_parity at 4096/512 and 1500/512 on an RTX 2080 Super).
        static const bool q4_tc = [] {
            const char* v = std::getenv("STRATA_PROMPT_ATTN_Q4_TC");
            return v != nullptr && v[0] == '1';
        }();
        if (q4_off || volta || (turing && !q4_tc) || pools.v_q4 == nullptr) return false;
        return launch<4>(q, pools, ids, steps, cap, s, attn, n_q, st);
    }
    if (pools.k_q != nullptr && pools.v_q4 != nullptr) {   // hybrid K8V4: int8 K + dequantized-q4 V
        if (!pools.k_scale) return false;
        if (volta) return launch70<3>(q, pools, ids, steps, cap, s, attn, n_q, st);
        return launch<3>(q, pools, ids, steps, cap, s, attn, n_q, st);
    }
    if (pools.k_q != nullptr) {
        if (!pools.v_q || !pools.k_scale || !pools.v_scale) return false;
        // STRATA_PROMPT_ATTN_V1=1 (debug): the first version, same accuracy, another summation order - the control
        // for how far the model amplifies an FP32-level change.  Turing always takes it: v2's cp.async does not
        // exist before sm_80.
        static const bool v1 = std::getenv("STRATA_PROMPT_ATTN_V1") != nullptr;
        if (volta) return launch70<1>(q, pools, ids, steps, cap, s, attn, n_q, st);
        // Turing: v2 with register staging instead of cp.async, opt-in (STRATA_PA_V2T=1) until measured
        static const bool v2t = [] { const char* e = std::getenv("STRATA_PA_V2T"); return e != nullptr && e[0] == '1'; }();
        if (v1 || (turing && !v2t)) {
            // #1821 (xrip): v1 with the next chunk's rows prefetched in registers and K/V sharing one buffer (two blocks an
            // SM), the same bits.  STRATA_PA_V1_PREFETCH=0 / 1 forces it off / on; unset: a Turing card with
            // STRATA_SM75_PERF=1.
            static const int pf_env = [] { const char* e = std::getenv("STRATA_PA_V1_PREFETCH"); return e ? (std::atoi(e) != 0 ? 1 : 0) : -1; }();
            if (pf_env == 1 || (pf_env < 0 && t75 && strata::sm75_series())) return launch<1, true>(q, pools, ids, steps, cap, s, attn, n_q, st);
            return launch<1>(q, pools, ids, steps, cap, s, attn, n_q, st);
        }
        // STRATA_PROMPT_ATTN_IMMA (sm_80+): both products on INT8 tensor cores (prompt_attn_i8v3_kernel).  ON by default since 0.2.7 on every
        // card that has them (Ampere and newer, CUDA builds; not Turing, Volta or Pascal, which have no int8 m16n8k32 MMA this kernel needs):
        // +1.6% / +1.8% prompt at 8K / 16K on the RTX PRO 5000 (sm_120) and +1.1% / +1.3% at 8K / 16K on 2x RTX A4000 (sm_86), decode unchanged,
        // 5 pairs each; another summation order (~1e-7), so a near-tie can flip a prompt's first token.  Sm_80, 8.9, 9.0 and the datacentre
        // Blackwells run the same kernel unmeasured.  STRATA_PROMPT_ATTN_IMMA=0 is the off switch (the v2 i8 kernel of 0.2.6), =1 as the default.
        if (g_pa_imma < 0) {
            const char* v = std::getenv("STRATA_PROMPT_ATTN_IMMA");
            g_pa_imma = v == nullptr ? 2 : (v[0] == '1' ? 1 : 0);   // 2: the default (on where the kernel can run)
        }
        if (!turing && (g_pa_imma == 1 || (g_pa_imma == 2 && strata::prompt_attn_imma_default(false, 80, strata::qwen_defaults_026())))) return launch_i8v3(q, pools, ids, steps, cap, s, attn, n_q, st);
        return launch_i8(q, pools, ids, steps, cap, s, attn, n_q, st);
    }
    if (!pools.k_pool || !pools.v_pool) return false;
    if (volta) return launch70<0>(q, pools, ids, steps, cap, s, attn, n_q, st);
    return launch<0>(q, pools, ids, steps, cap, s, attn, n_q, st);
}

}  // namespace strata::kernels
