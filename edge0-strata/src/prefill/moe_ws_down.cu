// src/prefill/moe_ws_down.cu - weight-stationary down products (see moe_ws.hpp): Q2_0 and IQ4_NL, [2560 x 640] read in
// place at the blob's down offset, K stopping at 640 (nothing past a matrix is read).  A CTA = 128 or 256 threads =
// as many weight rows of one expert (a row per thread, all of the chunk's H rows TPP at a time, the sums in registers);
// per K step the next step's weight bytes and q8_1 words are loaded into registers before this step's compute and
// stored to shared memory after it (one barrier pair per step).  The H rows are ONE mmq::swiglu_quant over all of the
// layer's rows: [k/128][rows] block_q8_1_mmq, of which blocks 0-4 are read.  (Measured on a GP100 die: ws / per-group
// MMQ 0.24-0.29 at 2.5 rows per expert, 0.38-0.49 at 10, 0.52-0.68 at 35, 0.61-0.74 at 160; bit-identical,
// compute-sanitizer clean with exact-size matrices.)
#include "common.cuh"
#include "mmq.cuh"

#include "moe_ws.cuh"

#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace strata::prefill::ws::detail {
namespace {

constexpr int DN_ROWS = 2560, DN_K = 640;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill ws: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// ---- Q2_0: 18-B blocks {f16 d; 16 B of 2-bit codes}, 180-B rows; a step is 128 values = 2 blocks = 36 B of a row.
// MMQ's values {-1, 0, 1, 2}[code] (ggml_cuda_mmq_load_tiles_q2_0's PRMT, the selector masked to 3 bits) and the
// activations' int8 are both made exact fp16 pairs; a 32-value block is 16 HFMA2 into 2 half2 sums, each fp16 lane
// summing 8 products of magnitude <= 256 - integers <= 2048, exact - added in fp32: the float MMQ's I2F(sumi) gives.
// Epilogue as MMQ's sm_60 SASS: per 32-value block FMUL.FTZ (dx * dy), FFMA.FTZ (prod, sum, acc), blocks in K order.
namespace q20 {
constexpr int NT = 128, ROWB = 180, NSTEP = DN_K / 128, QB = 144, PIECES = DN_ROWS / NT;

__device__ __forceinline__ uint32_t prmt(const uint32_t a, const uint32_t b, const uint32_t s) {
    uint32_t d;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
    return d;
}
__device__ __forceinline__ uint32_t hfma2(const uint32_t a, const uint32_t b, const uint32_t c) {
    uint32_t d;
    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}
__device__ __forceinline__ uint32_t hadd2(const uint32_t a, const uint32_t b) {
    uint32_t d;
    asm("add.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}
// the two fp16 lanes of h converted exactly to f32 and added
__device__ __forceinline__ float h2sum(const uint32_t h) {
    float f;
    asm("{\n\t.reg .f16 lo, hi;\n\t.reg .f32 a, b;\n\tmov.b32 {lo, hi}, %1;\n\t"
        "cvt.f32.f16 a, lo;\n\tcvt.f32.f16 b, hi;\n\tadd.f32 %0, a, b;\n\t}"
        : "=f"(f) : "r"(h));
    return f;
}
__device__ __forceinline__ float f16_to_f32(const uint32_t bits16) {
    float f;
    asm("{\n\t.reg .b16 h;\n\tcvt.u16.u32 h, %1;\n\tcvt.f32.f16 %0, h;\n\t}" : "=f"(f) : "r"(bits16));
    return f;
}

// the 16 values of the int16 qa = x2 & 0xFFFF, qb = x2 >> 16 of a block's codes (value i at bits 2i) as fp16 pairs
// (v0,v4) (v1,v5) (v2,v6) (v3,v7), qa's to wa and qb's to wb: prmt(H, 0, sel) with nibbles 1 / 3 of sel the code with
// its low bit flipped and nibbles 0 / 2 zero (byte 0 of H, the zero high byte)
__device__ __forceinline__ void dec2_h16(const uint32_t x2, uint32_t (&wa)[4], uint32_t (&wb)[4]) {
    constexpr uint32_t H = 0x3C40BC00u;   // bytes 00 BC 40 3C = fp16 high bytes of {0, -1, +2, +1}[c ^ 1] = {-1,0,1,2}[c]
    const uint32_t x = x2 ^ 0x55555555u;
    const uint32_t m0 = (x << 4) & 0x30303030u, m1 = (x << 2) & 0x30303030u, m2 = x & 0x30303030u, m3 = (x >> 2) & 0x30303030u;
    wa[0] = prmt(H, 0u, m0);
    wa[1] = prmt(H, 0u, m1);
    wa[2] = prmt(H, 0u, m2);
    wa[3] = prmt(H, 0u, m3);
    wb[0] = prmt(H, 0u, m0 >> 16);
    wb[1] = prmt(H, 0u, m1 >> 16);
    wb[2] = prmt(H, 0u, m2 >> 16);
    wb[3] = prmt(H, 0u, m3 >> 16);
}

// the 8 int8 of activation words A (values 0-3) and B (4-7) as fp16 pairs (a0,a4) (a1,a5) (a2,a6) (a3,a7):
// u = a + 128; fp16 bits 0x64uu = 1024 + u exactly; minus 1152 = a exactly
__device__ __forceinline__ void act_h16(const uint32_t A, const uint32_t B, uint32_t (&a)[4]) {
    const uint32_t P1 = prmt(A, B, 0x5140u) ^ 0x80808080u;   // u0 u4 u1 u5
    const uint32_t P2 = prmt(A, B, 0x7362u) ^ 0x80808080u;   // u2 u6 u3 u7
    constexpr uint32_t M = 0x64646464u, NEG1152 = 0xE480E480u;
    a[0] = hadd2(prmt(P1, M, 0x4140u), NEG1152);
    a[1] = hadd2(prmt(P1, M, 0x4342u), NEG1152);
    a[2] = hadd2(prmt(P2, M, 0x4140u), NEG1152);
    a[3] = hadd2(prmt(P2, M, 0x4342u), NEG1152);
}

// the exact integer dot of one 32-value block (weights w[16], activations a0..a3 = 16 fp16 pairs) as a float
__device__ __forceinline__ float block_sum(const uint32_t (&w)[16], const uint4 a0, const uint4 a1, const uint4 a2,
                                           const uint4 a3) {
    uint32_t p = hfma2(w[0], a0.x, 0u), q = hfma2(w[1], a0.y, 0u);
    p = hfma2(w[2], a0.z, p); q = hfma2(w[3], a0.w, q);
    p = hfma2(w[4], a1.x, p); q = hfma2(w[5], a1.y, q);
    p = hfma2(w[6], a1.z, p); q = hfma2(w[7], a1.w, q);
    p = hfma2(w[8], a2.x, p); q = hfma2(w[9], a2.y, q);
    p = hfma2(w[10], a2.z, p); q = hfma2(w[11], a2.w, q);
    p = hfma2(w[12], a3.x, p); q = hfma2(w[13], a3.y, q);
    p = hfma2(w[14], a3.z, p); q = hfma2(w[15], a3.w, q);
    return h2sum(p) + h2sum(q);
}

// TPP tokens per pass; tokens guarded in pairs (a pair's token past the count runs on a clamped row, not written).
// Per step every thread stages exactly 9 words of the CTA's 128 x 36 B (row stride 9 words: conflict-free decode reads)
// and threads < 8 TPP one 16-value q8_1 piece each.
template <int TPP>
__global__ void __launch_bounds__(NT, 4) dn_kernel(const Chunk* __restrict__ chunks, const char* __restrict__ w,
                                                   const int64_t* __restrict__ xoff, const int64_t xadd,
                                                   const char* __restrict__ xq, const int total_rows,
                                                   float* __restrict__ dst) {
    constexpr int G = 2, WPR = 9, AQ = TPP * 8, AI = (AQ + NT - 1) / NT;
    static_assert(TPP % G == 0, "");
    __shared__ int wsm[NT * WPR];
    __shared__ __align__(16) uint32_t act_s[TPP][4][16];
    __shared__ __align__(8) float dy_s[4][TPP];
    int r0;
    const Chunk c = cta_work<PIECES, NT>(chunks, r0);
    const char* __restrict__ W = w + xoff[c.j] + xadd + (size_t) r0 * ROWB;
    const int tid = threadIdx.x;
    int woff[WPR];   // word tid + NT i of the step's NT x 9 words: row (tid + NT i) / 9, word (tid + NT i) % 9
#pragma unroll
    for (int i = 0; i < WPR; ++i) {
        const int idx = tid + i * NT, r = idx / WPR;
        woff[i] = r * ROWB + 4 * (idx - r * WPR);
    }
    int wpre[WPR];
    uint4 apre[AI];
    float dpre[AI];
    auto load_w = [&](const int st) {
        const char* __restrict__ Ws = W + st * 36;
#pragma unroll
        for (int i = 0; i < WPR; ++i) wpre[i] = __ldg((const int*) (Ws + woff[i]));
    };
    auto store_w = [&]() {
#pragma unroll
        for (int i = 0; i < WPR; ++i) wsm[tid + i * NT] = wpre[i];
    };
    auto load_a = [&](const int st, const int trow0, const int nt) {
#pragma unroll
        for (int i = 0; i < AI; ++i) {
            const int idx = tid + i * NT;
            if (AQ % NT == 0 || idx < AQ) {
                const int t = idx >> 3, m = idx & 7;
                const char* blk = xq + ((size_t) st * total_rows + trow0 + min(t, nt - 1)) * QB;
                apre[i] = __ldg((const uint4*) (blk + 16) + m);   // values 16m .. 16m + 15 of the 128
                dpre[i] = __ldg((const float*) blk + (m >> 1));   // d of their 32-value block
            }
        }
    };
    auto store_a = [&]() {
#pragma unroll
        for (int i = 0; i < AI; ++i) {
            const int idx = tid + i * NT;
            if (AQ % NT == 0 || idx < AQ) {
                const int t = idx >> 3, m = idx & 7, b = m >> 1, hf = m & 1;
                uint32_t p0[4], p1[4];
                act_h16(apre[i].x, apre[i].y, p0);
                act_h16(apre[i].z, apre[i].w, p1);
                uint4* d = (uint4*) &act_s[t][b][8 * hf];
                d[0] = make_uint4(p0[0], p0[1], p0[2], p0[3]);
                d[1] = make_uint4(p1[0], p1[1], p1[2], p1[3]);
                dy_s[b][t] = dpre[i];   // both pieces of a block write the same value
            }
        }
    };

    load_w(0);
    load_a(0, c.row0, min(TPP, c.rows));
    for (int p0 = 0; p0 < c.rows; p0 += TPP) {
        const int nt = min(TPP, c.rows - p0);
        const int trow0 = c.row0 + p0;
        float acc[TPP];
#pragma unroll
        for (int t = 0; t < TPP; ++t) acc[t] = 0.0f;
#pragma unroll 1
        for (int st = 0; st < NSTEP; ++st) {
            __syncthreads();
            store_w();
            store_a();
            __syncthreads();
            if (st + 1 < NSTEP) {
                load_w(st + 1);
                load_a(st + 1, trow0, nt);
            } else if (p0 + TPP < c.rows) {
                load_w(0);
                load_a(0, trow0 + TPP, min(TPP, c.rows - p0 - TPP));
            }
            // the row's 9 step words: block 0 = d0 | qs0 at bytes 0..17 (the codes 2 bytes into a word: funnel shift),
            // block 1 = d1 in the top half of word 4, qs1 = words 5..8
            const uint32_t* cw = (const uint32_t*) (wsm + tid * WPR);
#pragma unroll 1
            for (int b = 0; b < 4; ++b) {   // 32-value block: Q2_0 block b / 2, half b % 2 (int16 4h .. 4h + 3), K order
                const int pq = b >> 1, h = b & 1;
                const int base = pq ? 5 + 2 * h : 2 * h, sh = pq ? 0 : 16;
                const uint32_t c0 = cw[base], c1 = cw[base + 1], c2 = cw[min(base + 2, WPR - 1)];
                const float dx = f16_to_f32(pq ? cw[4] >> 16 : cw[0] & 0xFFFFu);
                uint32_t wv[16], wa[4], wb[4];
                dec2_h16(__funnelshift_r(c0, c1, sh), wa, wb);   // int16 0 and 1 of the half
#pragma unroll
                for (int k = 0; k < 4; ++k) { wv[k] = wa[k]; wv[4 + k] = wb[k]; }
                dec2_h16(__funnelshift_r(c1, c2, sh), wa, wb);   // int16 2 and 3
#pragma unroll
                for (int k = 0; k < 4; ++k) { wv[8 + k] = wa[k]; wv[12 + k] = wb[k]; }
#pragma unroll
                for (int t0 = 0; t0 < TPP; t0 += G) {
                    if (t0 < nt) {
                        const float2 d2 = *(const float2*) &dy_s[b][t0];
#pragma unroll
                        for (int t = t0; t < t0 + G; ++t) {
                            const uint4* a4 = (const uint4*) act_s[t][b];
                            const float s = block_sum(wv, a4[0], a4[1], a4[2], a4[3]);
                            acc[t] = __fmaf_rn(__fmul_rn(dx, t == t0 ? d2.x : d2.y), s, acc[t]);
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int t = 0; t < TPP; ++t)
            if (t < nt) dst[(size_t) (trow0 + t) * DN_ROWS + r0 + tid] = acc[t];
    }
}
}  // namespace q20

// ---- IQ4_NL: 18-B blocks {f16 d; 16 B of 4-bit indices into kvalues_iq4nl}, 360-B rows; a step is SB blocks (SB 4:
// 72 B of a row, 8-byte aligned; SB 2: 36 B).  The int8 values are get_int_from_table_16's table lookups (the 16-byte
// table in registers) without its final byte reorder - the same values paired (v4k, v4k+1), (v4k+16, v4k+17), ... and
// the activations paired the same way (PRMT 0x9180 / 0xB3A2); one XMAD chain per (row, token, block), then MMQ's sm_60
// FMUL.FTZ (dx * dy), I2F (sumi), FFMA.FTZ, blocks in K order.  Staging: 3 rows per warp and k (lanes 0-26, a row's 9
// 4- or 8-byte pieces), the last k guarded so no row past the CTA's is read.
namespace nl4 {
constexpr int NB = DN_K / QK4_NL, BB = (int) sizeof(block_iq4_nl), ROWB = NB * BB;   // 20, 18, 360
static_assert(NB == 20 && BB == 18 && ROWB == 360, "");

// bytes (0,1) and (2,3) of v, each sign-extended to 16 bits
__device__ __forceinline__ void split01(const int v, int& p01, int& p23) {
    asm("prmt.b32 %0, %1, 0, 0x9180;" : "=r"(p01) : "r"(v));
    asm("prmt.b32 %0, %1, 0, 0xB3A2;" : "=r"(p23) : "r"(v));
}
// the 16-byte kvalues_iq4nl table as 4 words, read byte by byte
__device__ __forceinline__ void load_tab(uint32_t (&tab)[4]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t v = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) v |= (uint32_t) (uint8_t) kvalues_iq4nl[4 * i + b] << (8 * b);
        tab[i] = v;
    }
}
// a 32-value block from its 4 qs words (q[k]: values 4k..4k+3 in the low nibbles, 16+4k.. in the high ones): 16 int16
// pair words against the activation pairs split01 makes of activation words m = 0..7 (at act[2m], act[2m + 1]).
// get_int_from_table_16's body (CUDA branch) up to its two lookup words tmp[0] = (v4k, v4k+16, v4k+1, v4k+17),
// tmp[1] = (v4k+2, v4k+18, v4k+3, v4k+19)
__device__ __forceinline__ void dec(const int (&q)[4], const uint32_t (&tab)[4], int (&wv)[16]) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint32_t q4 = (uint32_t) q[k];
        uint32_t tmp[2];
        const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
#pragma unroll
        for (uint32_t i = 0; i < 2; ++i) {
            const uint32_t shift = 16 * i;
            const uint32_t low = __byte_perm(tab[0], tab[1], q4 >> shift);
            const uint32_t high = __byte_perm(tab[2], tab[3], q4 >> shift);
            tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
        }
        split2((int) tmp[0], wv[2 * k], wv[2 * (k + 4)]);           // (v4k, v4k+1) | (v16+4k, v17+4k)
        split2((int) tmp[1], wv[2 * k + 1], wv[2 * (k + 4) + 1]);   // (v4k+2, v4k+3) | (v18+4k, v19+4k)
    }
}
// block PAR of block pair p of the staged step (9 words per pair: even block d = lo16(w0), qs = bytes 2..17 by funnel
// shifts; odd block d = hi16(w4), qs = w5..w8)
template <int PAR>
__device__ __forceinline__ float load_blk(const int* __restrict__ cw, const int p, int (&q)[4]) {
    const int* c = cw + 9 * p;
    if constexpr (PAR == 0) {
#pragma unroll
        for (int k = 0; k < 4; ++k) q[k] = (int) __funnelshift_r((uint32_t) c[k], (uint32_t) c[k + 1], 16);
        return __half2float(__ushort_as_half((unsigned short) ((uint32_t) c[0] & 0xFFFFu)));
    } else {
#pragma unroll
        for (int k = 0; k < 4; ++k) q[k] = c[5 + k];
        return __half2float(__ushort_as_half((unsigned short) ((uint32_t) c[4] >> 16)));
    }
}

template <int NT, int TPP, int SB, int G, int PAR>
__device__ __forceinline__ void blk_compute(const int* __restrict__ cw, const int p, const uint32_t (&tab)[4],
                                            const int (*act_s)[SB][16], const float (*dy_s)[SB], const int nt,
                                            float (&acc)[TPP]) {
    const int b = 2 * p + PAR;   // block within the step
    int wv[16];
    float dx;
    {
        int q[4];
        dx = load_blk<PAR>(cw, p, q);
        dec(q, tab, wv);
    }
    // tokens in groups of G behind one guard (a group's tokens past nt run on stale shared memory, not written)
#pragma unroll
    for (int t0 = 0; t0 < TPP; t0 += G) {
        if (t0 < nt) {
#pragma unroll
            for (int t = t0; t < t0 + G; ++t) {
                const int4* a4 = (const int4*) act_s[t][b];
                const int4 a0 = a4[0], a1 = a4[1], a2 = a4[2], a3 = a4[3];
                const float dy = dy_s[t][b];
                int s0 = 0;
                s0 = mad2(wv[0], a0.x, s0); s0 = mad2(wv[1], a0.y, s0); s0 = mad2(wv[2], a0.z, s0); s0 = mad2(wv[3], a0.w, s0);
                s0 = mad2(wv[4], a1.x, s0); s0 = mad2(wv[5], a1.y, s0); s0 = mad2(wv[6], a1.z, s0); s0 = mad2(wv[7], a1.w, s0);
                s0 = mad2(wv[8], a2.x, s0); s0 = mad2(wv[9], a2.y, s0); s0 = mad2(wv[10], a2.z, s0); s0 = mad2(wv[11], a2.w, s0);
                s0 = mad2(wv[12], a3.x, s0); s0 = mad2(wv[13], a3.y, s0); s0 = mad2(wv[14], a3.z, s0); s0 = mad2(wv[15], a3.w, s0);
                acc[t] = __fmaf_rn(__fmul_rn(dx, dy), __int2float_rn(s0), acc[t]);
            }
        }
    }
}

template <int NT, int TPP, int SB, int G>
__global__ void __launch_bounds__(NT, 1) dn_kernel(const Chunk* __restrict__ chunks, const char* __restrict__ w,
                                                   const int64_t* __restrict__ xoff, const int64_t xadd,
                                                   const block_q8_1_mmq* __restrict__ xq, const int total_rows,
                                                   float* __restrict__ dst) {
    static_assert((SB == 2 || SB == 4) && NT % 32 == 0 && TPP % G == 0, "");
    constexpr int NW = NT / 32, WPS = 9 * SB / 2, S = WPS | 1, NSTEP = NB / SB, PIECES = DN_ROWS / NT;
    constexpr int KI = (NT + 3 * NW - 1) / (3 * NW);
    constexpr int AW = 9 * SB;                     // activation words per token and step (SB d + 8 SB qs words)
    constexpr int AI = (TPP * AW + NT - 1) / NT;
    using PT = typename std::conditional<SB == 4, int2, int>::type;
    __shared__ int wsm[NT * S];
    __shared__ __align__(16) int act_s[TPP][SB][16];
    __shared__ float dy_s[TPP][SB];
    uint32_t tab[4];
    load_tab(tab);
    int r0;
    const Chunk c = cta_work<PIECES, NT>(chunks, r0);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int sub = lane / 9, pc = lane - 9 * sub;   // sub 3: lanes 27-31, idle
    const int srow0 = 3 * warp + sub;
    const bool sact = sub < 3;
    const char* __restrict__ Wg = w + xoff[c.j] + xadd + (size_t) (r0 + srow0) * ROWB + pc * (int) sizeof(PT);
    int* __restrict__ ws_st = wsm + srow0 * S + pc * (int) (sizeof(PT) / 4);
    PT pre[KI];
    int apre[AI];
    auto load_w = [&](const int st) {
        if (sact) {
            const char* g = Wg + st * (SB * BB);
#pragma unroll
            for (int k = 0; k < KI; ++k)
                if (k < KI - 1 || srow0 + 3 * NW * k < NT) pre[k] = __ldg((const PT*) (g + (size_t) k * 3 * NW * ROWB));
        }
    };
    auto store_w = [&]() {
        if (sact) {
#pragma unroll
            for (int k = 0; k < KI; ++k)
                if (k < KI - 1 || srow0 + 3 * NW * k < NT) {
                    if constexpr (SB == 4) {
                        ws_st[k * 3 * NW * S] = pre[k].x;
                        ws_st[k * 3 * NW * S + 1] = pre[k].y;
                    } else {
                        ws_st[k * 3 * NW * S] = pre[k];
                    }
                }
        }
    };
    // word wd of token t's step: d4[b0 + wd] for wd < SB, else qs word 8 b0 + (wd - SB)
    auto load_a = [&](const int st, const int trow0, const int nt) {
        const int kb = (st * SB) >> 2, b0 = (st * SB) & 3;   // the 128-value q8_1 block and the first 32-block in it
#pragma unroll
        for (int i = 0; i < AI; ++i) {
            const int idx = threadIdx.x + i * NT;
            if (idx < nt * AW) {
                const int t = idx / AW, wd = idx - t * AW;
                const int* blk = (const int*) (xq + (size_t) kb * total_rows + trow0 + t);
                apre[i] = __ldg(blk + (wd < SB ? b0 + wd : 4 + 8 * b0 + (wd - SB)));
            }
        }
    };
    auto store_a = [&](const int nt) {
#pragma unroll
        for (int i = 0; i < AI; ++i) {
            const int idx = threadIdx.x + i * NT;
            if (idx < nt * AW) {
                const int t = idx / AW, wd = idx - t * AW;
                if (wd < SB) {
                    dy_s[t][wd] = __int_as_float(apre[i]);
                } else {
                    const int m = wd - SB, bb = m >> 3, mm = m & 7;
                    int lo, hi;
                    split01(apre[i], lo, hi);
                    *(int2*) &act_s[t][bb][2 * mm] = make_int2(lo, hi);
                }
            }
        }
    };
    load_w(0);
    load_a(0, c.row0, min(TPP, c.rows));
    const int* cw = wsm + threadIdx.x * S;
    for (int p0 = 0; p0 < c.rows; p0 += TPP) {
        const int nt = min(TPP, c.rows - p0);
        const int trow0 = c.row0 + p0;
        float acc[TPP];
#pragma unroll
        for (int t = 0; t < TPP; ++t) acc[t] = 0.0f;
#pragma unroll 1
        for (int st = 0; st < NSTEP; ++st) {
            __syncthreads();
            store_w();
            store_a(nt);
            __syncthreads();
            if (st + 1 < NSTEP) {
                load_w(st + 1);
                load_a(st + 1, trow0, nt);
            } else if (p0 + TPP < c.rows) {
                load_w(0);
                load_a(0, trow0 + TPP, min(TPP, c.rows - p0 - TPP));
            }
#pragma unroll 1
            for (int p = 0; p < SB / 2; ++p) {
                blk_compute<NT, TPP, SB, G, 0>(cw, p, tab, act_s, dy_s, nt, acc);
                blk_compute<NT, TPP, SB, G, 1>(cw, p, tab, act_s, dy_s, nt, acc);
            }
        }
#pragma unroll
        for (int t = 0; t < TPP; ++t)
            if (t < nt) dst[(size_t) (trow0 + t) * DN_ROWS + r0 + threadIdx.x] = acc[t];
    }
}

// ---- STRATA_WS_FAST=1 (ws::fast(), moe_ws.hpp): the 256 x 32 tier (2-block steps, pairs of tokens) with fewer
// instructions per pass, the same Dm rows bit for bit: every token's operations and their order are dn_kernel's.
// (Prototyped as a bench variant on one GP100 die: memcmp-equal to dn_kernel on real IQ4_NL layers at T 1024 and
// 1536, lognormal sigma 0.5 / 1.0 and a flat-32 control; 4.7-9.2% faster, 122 -> 102 registers, still two CTAs per
// SM.  moe_ws_down_bench checks and times it against dn_kernel.)
//   DUFF  a switch on the number of token pairs falls through the 16 unrolled pairs (a jump table) instead of a guard
//         per pair and block
//   DYT   dy as [block][token]: a pair's two dy in one LDS.64
template <int NT, int TPP, int SB, int G, int PAR>
__device__ __forceinline__ void blk_compute_fast(const int* __restrict__ cw, const int p, const uint32_t (&tab)[4],
                                                 const int (*act_s)[SB][16], const float* __restrict__ dyb, const int nt,
                                                 float (&acc)[TPP]) {
    static_assert(TPP / G == 16 && G == 2, "the switch below: 16 token pairs");
    const int b = 2 * p + PAR;   // block within the step
    int wv[16];
    float dx;
    {
        int q[4];
        dx = load_blk<PAR>(cw, p, q);
        dec(q, tab, wv);
    }
    // pairs past nt's run on stale shared memory and are not written, as dn_kernel's
    auto grp = [&](auto K) {
        constexpr int t0 = decltype(K)::value * G;
        const float2 d2 = *(const float2*) &dyb[b * TPP + t0];
#pragma unroll
        for (int t = t0; t < t0 + G; ++t) {
            const int4* a4 = (const int4*) act_s[t][b];
            const int4 a0 = a4[0], a1 = a4[1], a2 = a4[2], a3 = a4[3];
            int s0 = 0;
            s0 = mad2(wv[0], a0.x, s0); s0 = mad2(wv[1], a0.y, s0); s0 = mad2(wv[2], a0.z, s0); s0 = mad2(wv[3], a0.w, s0);
            s0 = mad2(wv[4], a1.x, s0); s0 = mad2(wv[5], a1.y, s0); s0 = mad2(wv[6], a1.z, s0); s0 = mad2(wv[7], a1.w, s0);
            s0 = mad2(wv[8], a2.x, s0); s0 = mad2(wv[9], a2.y, s0); s0 = mad2(wv[10], a2.z, s0); s0 = mad2(wv[11], a2.w, s0);
            s0 = mad2(wv[12], a3.x, s0); s0 = mad2(wv[13], a3.y, s0); s0 = mad2(wv[14], a3.z, s0); s0 = mad2(wv[15], a3.w, s0);
            acc[t] = __fmaf_rn(__fmul_rn(dx, t == t0 ? d2.x : d2.y), __int2float_rn(s0), acc[t]);
        }
    };
    using std::integral_constant;
    switch ((nt + G - 1) / G) {
        case 16: grp(integral_constant<int, 15>{}); [[fallthrough]];
        case 15: grp(integral_constant<int, 14>{}); [[fallthrough]];
        case 14: grp(integral_constant<int, 13>{}); [[fallthrough]];
        case 13: grp(integral_constant<int, 12>{}); [[fallthrough]];
        case 12: grp(integral_constant<int, 11>{}); [[fallthrough]];
        case 11: grp(integral_constant<int, 10>{}); [[fallthrough]];
        case 10: grp(integral_constant<int, 9>{}); [[fallthrough]];
        case 9: grp(integral_constant<int, 8>{}); [[fallthrough]];
        case 8: grp(integral_constant<int, 7>{}); [[fallthrough]];
        case 7: grp(integral_constant<int, 6>{}); [[fallthrough]];
        case 6: grp(integral_constant<int, 5>{}); [[fallthrough]];
        case 5: grp(integral_constant<int, 4>{}); [[fallthrough]];
        case 4: grp(integral_constant<int, 3>{}); [[fallthrough]];
        case 3: grp(integral_constant<int, 2>{}); [[fallthrough]];
        case 2: grp(integral_constant<int, 1>{}); [[fallthrough]];
        case 1: grp(integral_constant<int, 0>{}); [[fallthrough]];
        default: break;
    }
}

// dn_kernel with blk_compute_fast and dy_s as [SB][TPP]; everything else is dn_kernel's
template <int NT, int TPP, int SB, int G>
__global__ void __launch_bounds__(NT, 1) dn_kernel_fast(const Chunk* __restrict__ chunks, const char* __restrict__ w,
                                                        const int64_t* __restrict__ xoff, const int64_t xadd,
                                                        const block_q8_1_mmq* __restrict__ xq, const int total_rows,
                                                        float* __restrict__ dst) {
    static_assert((SB == 2 || SB == 4) && NT % 32 == 0 && TPP % G == 0, "");
    constexpr int NW = NT / 32, WPS = 9 * SB / 2, S = WPS | 1, NSTEP = NB / SB, PIECES = DN_ROWS / NT;
    constexpr int KI = (NT + 3 * NW - 1) / (3 * NW);
    constexpr int AW = 9 * SB;                     // activation words per token and step (SB d + 8 SB qs words)
    constexpr int AI = (TPP * AW + NT - 1) / NT;
    using PT = typename std::conditional<SB == 4, int2, int>::type;
    __shared__ int wsm[NT * S];
    __shared__ __align__(16) int act_s[TPP][SB][16];
    __shared__ __align__(8) float dy_s[SB * TPP];  // [SB][TPP]: dy of token t, block b at b * TPP + t
    uint32_t tab[4];
    load_tab(tab);
    int r0;
    const Chunk c = cta_work<PIECES, NT>(chunks, r0);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int sub = lane / 9, pc = lane - 9 * sub;   // sub 3: lanes 27-31, idle
    const int srow0 = 3 * warp + sub;
    const bool sact = sub < 3;
    const char* __restrict__ Wg = w + xoff[c.j] + xadd + (size_t) (r0 + srow0) * ROWB + pc * (int) sizeof(PT);
    int* __restrict__ ws_st = wsm + srow0 * S + pc * (int) (sizeof(PT) / 4);
    PT pre[KI];
    int apre[AI];
    auto load_w = [&](const int st) {
        if (sact) {
            const char* g = Wg + st * (SB * BB);
#pragma unroll
            for (int k = 0; k < KI; ++k)
                if (k < KI - 1 || srow0 + 3 * NW * k < NT) pre[k] = __ldg((const PT*) (g + (size_t) k * 3 * NW * ROWB));
        }
    };
    auto store_w = [&]() {
        if (sact) {
#pragma unroll
            for (int k = 0; k < KI; ++k)
                if (k < KI - 1 || srow0 + 3 * NW * k < NT) {
                    if constexpr (SB == 4) {
                        ws_st[k * 3 * NW * S] = pre[k].x;
                        ws_st[k * 3 * NW * S + 1] = pre[k].y;
                    } else {
                        ws_st[k * 3 * NW * S] = pre[k];
                    }
                }
        }
    };
    auto load_a = [&](const int st, const int trow0, const int nt) {
        const int kb = (st * SB) >> 2, b0 = (st * SB) & 3;
#pragma unroll
        for (int i = 0; i < AI; ++i) {
            const int idx = threadIdx.x + i * NT;
            if (idx < nt * AW) {
                const int t = idx / AW, wd = idx - t * AW;
                const int* blk = (const int*) (xq + (size_t) kb * total_rows + trow0 + t);
                apre[i] = __ldg(blk + (wd < SB ? b0 + wd : 4 + 8 * b0 + (wd - SB)));
            }
        }
    };
    auto store_a = [&](const int nt) {
#pragma unroll
        for (int i = 0; i < AI; ++i) {
            const int idx = threadIdx.x + i * NT;
            if (idx < nt * AW) {
                const int t = idx / AW, wd = idx - t * AW;
                if (wd < SB) {
                    dy_s[wd * TPP + t] = __int_as_float(apre[i]);
                } else {
                    const int m = wd - SB, bb = m >> 3, mm = m & 7;
                    int lo, hi;
                    split01(apre[i], lo, hi);
                    *(int2*) &act_s[t][bb][2 * mm] = make_int2(lo, hi);
                }
            }
        }
    };
    load_w(0);
    load_a(0, c.row0, min(TPP, c.rows));
    const int* cw = wsm + threadIdx.x * S;
    for (int p0 = 0; p0 < c.rows; p0 += TPP) {
        const int nt = min(TPP, c.rows - p0);
        const int trow0 = c.row0 + p0;
        float acc[TPP];
#pragma unroll
        for (int t = 0; t < TPP; ++t) acc[t] = 0.0f;
#pragma unroll 1
        for (int st = 0; st < NSTEP; ++st) {
            __syncthreads();
            store_w();
            store_a(nt);
            __syncthreads();
            if (st + 1 < NSTEP) {
                load_w(st + 1);
                load_a(st + 1, trow0, nt);
            } else if (p0 + TPP < c.rows) {
                load_w(0);
                load_a(0, trow0 + TPP, min(TPP, c.rows - p0 - TPP));
            }
#pragma unroll 1
            for (int p = 0; p < SB / 2; ++p) {
                blk_compute_fast<NT, TPP, SB, G, 0>(cw, p, tab, act_s, dy_s, nt, acc);
                blk_compute_fast<NT, TPP, SB, G, 1>(cw, p, tab, act_s, dy_s, nt, acc);
            }
        }
#pragma unroll
        for (int t = 0; t < TPP; ++t)
            if (t < nt) dst[(size_t) (trow0 + t) * DN_ROWS + r0 + threadIdx.x] = acc[t];
    }
}
}  // namespace nl4

}  // namespace

// Q2_0: 8 tokens per pass below 5 rows per expert, then 16 (equal at 5)
void dn_q2_0(const Layer& L, double u, cudaStream_t s) {
    if (L.nchunks <= 0) return;
    const unsigned grid = (unsigned) (L.nchunks * q20::PIECES);
    if (u < 5.0) q20::dn_kernel<8><<<grid, q20::NT, 0, s>>>(L.chunks, (const char*) L.w, L.xoff, L.down_off, (const char*) L.hq,
                                                            (int) L.rows, L.dm);
    else q20::dn_kernel<16><<<grid, q20::NT, 0, s>>>(L.chunks, (const char*) L.w, L.xoff, L.down_off, (const char*) L.hq,
                                                     (int) L.rows, L.dm);
    ck(cudaGetLastError(), "down Q2_0");
}

// IQ4_NL: 128 rows x 8 tokens (4-block steps) below 5 rows per expert, 256 x 16 below 20, then 256 x 32 (2-block steps);
// `fast` (STRATA_WS_FAST=1): the 256 x 32 tier runs dn_kernel_fast instead - the same Dm rows
void dn_iq4_nl(const Layer& L, double u, cudaStream_t s, bool fast) {
    if (L.nchunks <= 0) return;
    const char* w = (const char*) L.w;
    const block_q8_1_mmq* x = (const block_q8_1_mmq*) L.hq;
    if (u < 5.0)
        nl4::dn_kernel<128, 8, 4, 2><<<(unsigned) (L.nchunks * (DN_ROWS / 128)), 128, 0, s>>>(L.chunks, w, L.xoff, L.down_off, x,
                                                                                           (int) L.rows, L.dm);
    else if (u < 20.0)
        nl4::dn_kernel<256, 16, 4, 2><<<(unsigned) (L.nchunks * (DN_ROWS / 256)), 256, 0, s>>>(L.chunks, w, L.xoff, L.down_off, x,
                                                                                             (int) L.rows, L.dm);
    else if (fast)
        nl4::dn_kernel_fast<256, 32, 2, 2><<<(unsigned) (L.nchunks * (DN_ROWS / 256)), 256, 0, s>>>(L.chunks, w, L.xoff,
                                                                                                  L.down_off, x,
                                                                                                  (int) L.rows, L.dm);
    else
        nl4::dn_kernel<256, 32, 2, 2><<<(unsigned) (L.nchunks * (DN_ROWS / 256)), 256, 0, s>>>(L.chunks, w, L.xoff, L.down_off, x,
                                                                                             (int) L.rows, L.dm);
    ck(cudaGetLastError(), "down IQ4_NL");
}

}  // namespace strata::prefill::ws::detail
