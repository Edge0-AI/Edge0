// src/prefill/moe_ws_gu_iq2.cu - weight-stationary gate/up products (see moe_ws.hpp): IQ2_XXS, IQ2_XS and IQ2_S,
// [1280 x 2560] read in place.  A CTA = 128 threads = 128 weight rows of one expert (a row per thread, all of the
// chunk's activation rows TPP at a time, the sums in registers); K advances 128 values a step (half a 256-value block
// = 4 sub-blocks): the step's bytes of the 128 rows and the step's q8_1 block of each token go to shared memory (the
// activations split to int16 pairs once), then each thread decodes its row's 4 sub-blocks once per pass and runs them
// against the tokens.  Bits: per 32-value sub-block MMQ's int8 values (ggml_cuda_mmq_load_tiles_iq2_*) and scale
// expressions, then MMQ's epilogue (IQ2_XXS: acc = fma(dx * dy, float(sum), acc); IQ2_XS / IQ2_S, a scale per 16
// values: t = fma(float(sum 0-15), d0, 0), t = fma(float(sum 16-31), d1, t), acc = fma(t, dy, acc)), sub-blocks in K
// order.  (Measured on a GP100 die: ws / per-group MMQ 0.26-0.33 at 2.5
// rows per expert, 0.46-0.52 at 10, 0.67-0.76 at 35, 0.75-0.82 at 160; bit-identical, compute-sanitizer clean with
// exact-size matrices.)
//
// Decode: every iq2xxs / iq2xs / iq2s_grid byte is 0x08, 0x19 or 0x2B, so a grid entry is 8 nibble codes (0, 1, 2)
// and a value's sign adds 4 to its nibble: __byte_perm(0x002B1908, 0x00D5E7F8, codes | signs) is MMQ's
// __vsub4(g ^ s, s) word, one PRMT per 4 values (checked against MMQ's expressions for every entry and sign byte).
// Staging without per-lane branches: rows are 660 / 740 / 820 B, so block kc starts 2 * (kc & 1) bytes into a 4-byte
// word for the whole CTA - an even block's words are two aligned loads and a funnel shift (the second load ends at most
// 2 bytes into block kc + 1 of the same row), an odd block's are aligned; d sits the other way round (an odd block's d
// is the high half of the word 2 bytes before it, inside block kc - 1).  Every load of a lane is issued before its
// stores, and nothing outside a row is read (the last row's last block is odd).  Token groups of TG: one `t < nt`
// branch per group, so the XMAD chains of the group's tokens interleave (a group's tokens past the count are not
// written).
#include "common.cuh"
#include "mmq.cuh"

#include "moe_ws.cuh"

#include <cstdio>
#include <cstdlib>

namespace strata::prefill::ws::detail {
namespace {

constexpr int GU_ROWS = 1280, GU_K = 2560, NB = GU_K / QK_K, NSTEP = 2 * NB, NT = 128, NW = NT / 32, PIECES = GU_ROWS / NT;
constexpr uint32_t PX = 0x002B1908u, PY = 0x00D5E7F8u;   // codes 0-2: +8 +25 +43 | codes 4-6: -8 -25 -43

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill ws: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// the step's activations of IQ2_XS / IQ2_S: nt consecutive 144-B block_q8_1_mmq, d4 -> dy_s, int8 -> int16 pairs
template <int TPP>
__device__ __forceinline__ void stage_act(const block_q8_1_mmq* __restrict__ xq, const int hc, const int total_rows,
                                          const int trow0, const int nt, int (*act_s)[4][16], float (*dy_s)[4]) {
    constexpr int NA = (TPP * 36 + NT - 1) / NT;
    int av[NA];
    const int* xr = (const int*) (xq + (size_t) hc * total_rows + trow0);
#pragma unroll
    for (int i = 0; i < NA; ++i) {
        const int idx = threadIdx.x + i * NT;
        if (idx < nt * 36) av[i] = __ldg(xr + idx);
    }
#pragma unroll
    for (int i = 0; i < NA; ++i) {
        const int idx = threadIdx.x + i * NT;
        if (idx < nt * 36) {
            const int t = idx / 36, wd = idx - t * 36;
            if (wd < 4) {
                dy_s[t][wd] = __int_as_float(av[i]);
            } else {
                const int m = wd - 4;
                int lo, hi;
                split2(av[i], lo, hi);
                *(int2*) &act_s[t][m / 8][2 * (m % 8)] = make_int2(lo, hi);
            }
        }
    }
}

// IQ2_XS / IQ2_S: sub-block s's 16 int16-pair words wv and scales d0 / d1 against the tokens (vec_dot_q8_0_16_q8_1's
// epilogue).  CVT 0: four XMAD chains and I2F; 1: two chains seeded with kMagicI (exact for the 16-value sums,
// |sum| <= 16 * 43 * 127 < 2^22)
template <int TPP, int CVT, int TG>
__device__ __forceinline__ void mac16(const int (&wv)[16], const float dx0, const float dx1, const int s,
                                      const int (*act_s)[4][16], const float (*dy_s)[4], const int nt, float (&acc)[TPP]) {
#pragma unroll
    for (int t0 = 0; t0 < TPP; t0 += TG) {
        if (t0 < nt) {
#pragma unroll
            for (int t = t0; t < t0 + TG; ++t) {
                const int4* a4 = (const int4*) act_s[t][s];
                const int4 a0 = a4[0], a1 = a4[1], a2 = a4[2], a3 = a4[3];
                const float dy = dy_s[t][s];
                float fa, fb;   // float(sum of values 0-15), float(sum of values 16-31): exact
                if constexpr (CVT == 0) {
                    int s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                    s0 = mad2(wv[0], a0.x, s0); s1 = mad2(wv[1], a0.y, s1); s0 = mad2(wv[2], a0.z, s0); s1 = mad2(wv[3], a0.w, s1);
                    s0 = mad2(wv[4], a1.x, s0); s1 = mad2(wv[5], a1.y, s1); s0 = mad2(wv[6], a1.z, s0); s1 = mad2(wv[7], a1.w, s1);
                    s2 = mad2(wv[8], a2.x, s2); s3 = mad2(wv[9], a2.y, s3); s2 = mad2(wv[10], a2.z, s2); s3 = mad2(wv[11], a2.w, s3);
                    s2 = mad2(wv[12], a3.x, s2); s3 = mad2(wv[13], a3.y, s3); s2 = mad2(wv[14], a3.z, s2); s3 = mad2(wv[15], a3.w, s3);
                    fa = __int2float_rn(s0 + s1);
                    fb = __int2float_rn(s2 + s3);
                } else {
                    int sa = kMagicI, sb = kMagicI;
                    sa = mad2(wv[0], a0.x, sa); sb = mad2(wv[8], a2.x, sb);
                    sa = mad2(wv[1], a0.y, sa); sb = mad2(wv[9], a2.y, sb);
                    sa = mad2(wv[2], a0.z, sa); sb = mad2(wv[10], a2.z, sb);
                    sa = mad2(wv[3], a0.w, sa); sb = mad2(wv[11], a2.w, sb);
                    sa = mad2(wv[4], a1.x, sa); sb = mad2(wv[12], a3.x, sb);
                    sa = mad2(wv[5], a1.y, sa); sb = mad2(wv[13], a3.y, sb);
                    sa = mad2(wv[6], a1.z, sa); sb = mad2(wv[14], a3.z, sb);
                    sa = mad2(wv[7], a1.w, sa); sb = mad2(wv[15], a3.w, sb);
                    fa = __fsub_rn(__int_as_float(sa), kMagicF);
                    fb = __fsub_rn(__int_as_float(sb), kMagicF);
                }
                float tf = __fmaf_rn(fa, dx0, 0.0f);
                tf = __fmaf_rn(fb, dx1, tf);
                acc[t] = __fmaf_rn(tf, dy, acc[t]);
            }
        }
    }
}

// d0 / d1 = ((ls & 15 | ls >> 4) * d + d / 2) / 4 as MMQ's sm_60 SASS computes them
__device__ __forceinline__ void scales16(const int ls, const int dbits, float& d0, float& d1) {
    const float d = __half2float(__ushort_as_half((unsigned short) (dbits & 0xFFFF)));
    const float dh = __fmul_rn(d, 0.5f);
    d0 = __fmul_rn(__fmaf_rn(d, __int2float_rn(ls & 0x0F), dh), 0.25f);
    d1 = __fmul_rn(__fmaf_rn(d, __int2float_rn(ls >> 4), dh), 0.25f);
}

// ---- IQ2_XXS: 66-B blocks; per row and step 8 qs words (4 grid bytes, then aux32: 4 x 7 sign bits + the scale) and
// d (stride 9).  Every lane slot is one word type: qs words rows tid/8 + 16 i (word tid%8), d the thread's own row.
// A partial pass's token groups run past the count on copies of row nt - 1 (not written); a full pass has no branch.
namespace xxs2 {
constexpr int BB = (int) sizeof(block_iq2_xxs), ROWB = NB * BB, CWS = 9;
static_assert(BB == 66 && ROWB % 4 == 0, "block_iq2_xxs");

// nibble b = the code of byte b of iq2xxs_grid[i]
__device__ __forceinline__ uint32_t grid_codes(const int i) {
    const uint2 g = ((const uint2*) iq2xxs_grid)[i];
    uint32_t w = 0;
#pragma unroll
    for (int b = 0; b < 8; ++b) {
        const uint32_t v = ((b < 4 ? g.x : g.y) >> (8 * (b & 3))) & 0xFF;
        w |= ((uint32_t) (v > 0x08u) + (uint32_t) (v > 0x19u)) << (4 * b);
    }
    return w;
}
// a 7-bit sign field: MMQ's unpack_ksigns byte (bit 7 = parity) spread to bit 2 of nibble b
__device__ __forceinline__ uint32_t sign_spread(const int v) {
    const uint32_t s = (uint32_t) v ^ ((uint32_t) (__popc(v) & 1) << 7);
    uint32_t w = 0;
#pragma unroll
    for (int b = 0; b < 8; ++b) w |= ((s >> b) & 1u) << (4 * b + 2);
    return w;
}

template <int TPP, int CH, int TG, bool FULL>
__device__ __forceinline__ void sb_compute(const int* __restrict__ cw, const int s, const uint32_t* __restrict__ tabA,
                                           const uint32_t* __restrict__ tabB, const int (*act_s)[4][16],
                                           const float (*dy_s)[4], const int nt, float (&acc)[TPP]) {
    int wv[16];
    float dx;
    {
        const uint32_t q2 = (uint32_t) cw[2 * s + 0];
        const uint32_t aux32 = (uint32_t) cw[2 * s + 1];
#pragma unroll
        for (int l = 0; l < QR2_XXS; ++l) {
            const uint32_t sel = tabA[(q2 >> (8 * l)) & 0xFF] | tabB[(aux32 >> (7 * l)) & 0x7F];
            split2((int) __byte_perm(PX, PY, sel), wv[4 * l + 0], wv[4 * l + 1]);
            split2((int) __byte_perm(PX, PY, sel >> 16), wv[4 * l + 2], wv[4 * l + 3]);
        }
        const int ls = aux32 >> 27 | 1;
        const float d = __ushort_as_half((unsigned short) (cw[8] & 0xFFFF));
        dx = d * ls / 8;   // MMQ: (d * scale + d / 2) / 4
    }
#pragma unroll
    for (int t0 = 0; t0 < TPP; t0 += TG) {
        if (FULL || t0 < nt) {
#pragma unroll
            for (int t = t0; t < t0 + TG; ++t) {
                const int4* a4 = (const int4*) act_s[t][s];
                const int4 a0 = a4[0], a1 = a4[1], a2 = a4[2], a3 = a4[3];
                const int a[16] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w, a2.x, a2.y, a2.z, a2.w, a3.x, a3.y, a3.z, a3.w};
                const float dy = dy_s[t][s];
                int sc[CH];
                sc[0] = kMagicI;
#pragma unroll
                for (int c = 1; c < CH; ++c) sc[c] = 0;
#pragma unroll
                for (int m = 0; m < 16; ++m) sc[m % CH] = mad2(wv[m], a[m], sc[m % CH]);
                int sum = sc[0];
#pragma unroll
                for (int c = 1; c < CH; ++c) sum += sc[c];
                acc[t] = __fmaf_rn(__fmul_rn(dx, dy), __fsub_rn(__int_as_float(sum), kMagicF), acc[t]);
            }
        }
    }
}

template <int TPP, int CH, int TG>
__global__ void __launch_bounds__(NT, 1) gu_kernel(const Chunk* __restrict__ chunks, const char* __restrict__ w,
                                                  const int64_t* __restrict__ xoff, const block_q8_1_mmq* __restrict__ xq,
                                                  const int total_rows, float* __restrict__ dst) {
    static_assert(TPP % TG == 0 && (TPP * 32) % NT == 0 && TPP * 4 <= NT, "");
    constexpr int NDS = 8;                      // qs-word slots per lane: rows tid/8 + i*NT/8, word tid%8
    constexpr int RS = NT / 8;                  // rows per slot
    constexpr int NA = TPP * 32 / NT;           // activation qs-word slots per lane
    __shared__ __align__(16) uint32_t tabA[256];
    __shared__ __align__(16) uint32_t tabB[128];
    __shared__ __align__(16) int act_s[TPP][4][16];   // [token][32-block of the step][8 words split to 16]
    __shared__ float dy_s[TPP][4];
    __shared__ int wsm[NT * CWS];
    const int tid = threadIdx.x;
    for (int i = tid; i < 256; i += NT) tabA[i] = grid_codes(i);
    for (int i = tid; i < 128; i += NT) tabB[i] = sign_spread(i);
    int r0;
    const Chunk c = cta_work<PIECES, NT>(chunks, r0);
    const char* __restrict__ W = w + xoff[c.j] + (size_t) r0 * ROWB;
    const char* __restrict__ Wq = W + (size_t) (tid >> 3) * ROWB + 4 * (tid & 7);   // + i*RS*ROWB + kc*BB + 32h
    const char* __restrict__ Wd = W + (size_t) tid * ROWB;                          // + kc*BB
    int* __restrict__ sq = wsm + (tid >> 3) * CWS + (tid & 7);                      // + i*RS*CWS
    int* __restrict__ sd = wsm + tid * CWS + 8;
    const int* cw = wsm + tid * CWS;
    for (int p0 = 0; p0 < c.rows; p0 += TPP) {
        const int nt = min(TPP, c.rows - p0);
        const int trow0 = c.row0 + p0;
        float acc[TPP];
#pragma unroll
        for (int t = 0; t < TPP; ++t) acc[t] = 0.0f;
#pragma unroll 1
        for (int hc = 0; hc < NSTEP; ++hc) {
            const int kc = hc >> 1, h = hc & 1;
            const int* xr = (const int*) (xq + (size_t) hc * total_rows + trow0);   // nt consecutive 36-word blocks
            __syncthreads();
            int av[NA], dv = 0;
#pragma unroll
            for (int i = 0; i < NA; ++i) {
                const int idx = tid + i * NT, t = idx >> 5;
                av[i] = __ldg(xr + min(t, nt - 1) * 36 + 4 + (idx & 31));   // clamped: no branch, never past row nt-1
            }
            if (TPP * 4 == NT || tid < TPP * 4) dv = __ldg(xr + min(tid >> 2, nt - 1) * 36 + (tid & 3));
            const char* bq = Wq + kc * BB + 32 * h;
            if (kc & 1) {   // odd block: qs words aligned; d = high half of the word 2 bytes before the block
                int v[NDS];
#pragma unroll
                for (int i = 0; i < NDS; ++i) v[i] = __ldg((const int*) (bq + 2 + (size_t) i * RS * ROWB));
                const int vd = __ldg((const int*) (Wd + kc * BB - 2));
#pragma unroll
                for (int i = 0; i < NDS; ++i) sq[i * RS * CWS] = v[i];
                *sd = (int) ((uint32_t) vd >> 16);
            } else {        // even block: qs words 2 bytes into a word: two aligned words + funnel shift; d = low half
                int lo[NDS], hi[NDS];
#pragma unroll
                for (int i = 0; i < NDS; ++i) {
                    lo[i] = __ldg((const int*) (bq + (size_t) i * RS * ROWB));
                    hi[i] = __ldg((const int*) (bq + 4 + (size_t) i * RS * ROWB));
                }
                const int vd = __ldg((const int*) (Wd + kc * BB));
#pragma unroll
                for (int i = 0; i < NDS; ++i) sq[i * RS * CWS] = (int) __funnelshift_r((uint32_t) lo[i], (uint32_t) hi[i], 16);
                *sd = vd;
            }
#pragma unroll
            for (int i = 0; i < NA; ++i) {
                const int idx = tid + i * NT, t = idx >> 5, wd = idx & 31;
                int lo, hi;
                split2(av[i], lo, hi);
                *(int2*) &act_s[t][wd >> 3][2 * (wd & 7)] = make_int2(lo, hi);
            }
            if (TPP * 4 == NT || tid < TPP * 4) dy_s[tid >> 2][tid & 3] = __int_as_float(dv);
            __syncthreads();
            if (nt == TPP) {
#pragma unroll 1
                for (int s = 0; s < 4; ++s) sb_compute<TPP, CH, TG, true>(cw, s, tabA, tabB, act_s, dy_s, nt, acc);
            } else {
#pragma unroll 1
                for (int s = 0; s < 4; ++s) sb_compute<TPP, CH, TG, false>(cw, s, tabA, tabB, act_s, dy_s, nt, acc);
            }
        }
#pragma unroll
        for (int t = 0; t < TPP; ++t)
            if (t < nt) dst[(size_t) (trow0 + t) * GU_ROWS + r0 + tid] = acc[t];
    }
}
}  // namespace xxs2

// ---- IQ2_XS: 74-B blocks; per row and step the qs u16 [16h, 16h + 16) (8 words), the scales [4h, 4h + 4) and d:
// 10 words, row stride 11, staged by lanes 0-9 / 16-25 of each warp (two rows per warp and k).  The grid codes and
// ksigns' spread signs (q2 & 0x1FF, q2 >> 9) are tables generated from ggml's iq2xs_grid.
namespace xs2 {
constexpr int BB = 74, ROWB = NB * BB, CW = 10, CWS = 11, KR = NT / (2 * NW);

__device__ const uint32_t kIq2xsCode[512] = {
    0x00000000u, 0x00000002u, 0x00000011u, 0x00000020u, 0x00000022u, 0x00000101u, 0x00000110u, 0x00000112u,
    0x00000121u, 0x00000200u, 0x00000202u, 0x00000211u, 0x00000220u, 0x00001001u, 0x00001010u, 0x00001012u,
    0x00001021u, 0x00001100u, 0x00001102u, 0x00001111u, 0x00001120u, 0x00001201u, 0x00001210u, 0x00002000u,
    0x00002002u, 0x00002011u, 0x00002020u, 0x00002101u, 0x00002110u, 0x00002121u, 0x00002200u, 0x00010001u,
    0x00010010u, 0x00010012u, 0x00010021u, 0x00010100u, 0x00010102u, 0x00010111u, 0x00010120u, 0x00010122u,
    0x00010201u, 0x00010210u, 0x00011000u, 0x00011002u, 0x00011011u, 0x00011020u, 0x00011101u, 0x00011110u,
    0x00011200u, 0x00011220u, 0x00012001u, 0x00012010u, 0x00012100u, 0x00020000u, 0x00020002u, 0x00020011u,
    0x00020020u, 0x00020101u, 0x00020110u, 0x00020200u, 0x00021001u, 0x00021010u, 0x00021100u, 0x00021111u,
    0x00022000u, 0x00022022u, 0x00100001u, 0x00100010u, 0x00100012u, 0x00100021u, 0x00100100u, 0x00100102u,
    0x00100111u, 0x00100120u, 0x00100201u, 0x00100210u, 0x00101000u, 0x00101002u, 0x00101011u, 0x00101020u,
    0x00101101u, 0x00101110u, 0x00101112u, 0x00101200u, 0x00102001u, 0x00102010u, 0x00102100u, 0x00110000u,
    0x00110002u, 0x00110011u, 0x00110020u, 0x00110101u, 0x00110110u, 0x00110200u, 0x00111001u, 0x00111010u,
    0x00111100u, 0x00111201u, 0x00112000u, 0x00120001u, 0x00120010u, 0x00120100u, 0x00120212u, 0x00121000u,
    0x00121002u, 0x00122010u, 0x00200000u, 0x00200002u, 0x00200011u, 0x00200020u, 0x00200022u, 0x00200101u,
    0x00200110u, 0x00200200u, 0x00200211u, 0x00201001u, 0x00201010u, 0x00201100u, 0x00201120u, 0x00202000u,
    0x00202200u, 0x00202222u, 0x00210001u, 0x00210010u, 0x00210100u, 0x00211000u, 0x00212001u, 0x00212021u,
    0x00220000u, 0x00220200u, 0x00220220u, 0x00222112u, 0x00222200u, 0x01000001u, 0x01000010u, 0x01000012u,
    0x01000021u, 0x01000100u, 0x01000102u, 0x01000111u, 0x01000120u, 0x01000201u, 0x01000210u, 0x01001000u,
    0x01001002u, 0x01001011u, 0x01001020u, 0x01001101u, 0x01001110u, 0x01001200u, 0x01001222u, 0x01002001u,
    0x01002010u, 0x01002100u, 0x01010000u, 0x01010002u, 0x01010011u, 0x01010020u, 0x01010101u, 0x01010110u,
    0x01010200u, 0x01011001u, 0x01011010u, 0x01011100u, 0x01012000u, 0x01012110u, 0x01012112u, 0x01020001u,
    0x01020010u, 0x01020012u, 0x01020100u, 0x01021000u, 0x01021200u, 0x01100000u, 0x01100002u, 0x01100011u,
    0x01100020u, 0x01100101u, 0x01100110u, 0x01100200u, 0x01101001u, 0x01101010u, 0x01101021u, 0x01101100u,
    0x01101210u, 0x01102000u, 0x01110001u, 0x01110010u, 0x01110100u, 0x01111000u, 0x01120000u, 0x01120110u,
    0x01121021u, 0x01200001u, 0x01200010u, 0x01200100u, 0x01200102u, 0x01201000u, 0x01201110u, 0x01202012u,
    0x01210000u, 0x01210011u, 0x01211212u, 0x01221101u, 0x01222221u, 0x02000000u, 0x02000002u, 0x02000011u,
    0x02000020u, 0x02000022u, 0x02000101u, 0x02000110u, 0x02000200u, 0x02001001u, 0x02001010u, 0x02001100u,
    0x02002000u, 0x02002200u, 0x02010001u, 0x02010010u, 0x02010100u, 0x02011000u, 0x02011020u, 0x02011211u,
    0x02020000u, 0x02020202u, 0x02022000u, 0x02022220u, 0x02100001u, 0x02100010u, 0x02100100u, 0x02100221u,
    0x02101000u, 0x02110000u, 0x02111001u, 0x02111102u, 0x02112121u, 0x02120001u, 0x02120122u, 0x02122212u,
    0x02200000u, 0x02200020u, 0x02200022u, 0x02200200u, 0x02201111u, 0x02202020u, 0x02202202u, 0x02211220u,
    0x02212100u, 0x02220020u, 0x02220200u, 0x02222002u, 0x02222020u, 0x02222022u, 0x10000001u, 0x10000010u,
    0x10000012u, 0x10000021u, 0x10000100u, 0x10000102u, 0x10000111u, 0x10000120u, 0x10000201u, 0x10000210u,
    0x10001000u, 0x10001002u, 0x10001011u, 0x10001020u, 0x10001022u, 0x10001101u, 0x10001110u, 0x10001200u,
    0x10001211u, 0x10002001u, 0x10002010u, 0x10002100u, 0x10010000u, 0x10010002u, 0x10010011u, 0x10010020u,
    0x10010101u, 0x10010110u, 0x10010200u, 0x10011001u, 0x10011010u, 0x10011100u, 0x10012000u, 0x10012011u,
    0x10012202u, 0x10020001u, 0x10020010u, 0x10020100u, 0x10020102u, 0x10020221u, 0x10021000u, 0x10100000u,
    0x10100002u, 0x10100011u, 0x10100020u, 0x10100101u, 0x10100110u, 0x10100121u, 0x10100200u, 0x10101001u,
    0x10101010u, 0x10101100u, 0x10102000u, 0x10102110u, 0x10110001u, 0x10110010u, 0x10110100u, 0x10110210u,
    0x10111000u, 0x10112122u, 0x10120000u, 0x10120022u, 0x10121010u, 0x10121100u, 0x10200001u, 0x10200010u,
    0x10200100u, 0x10201000u, 0x10201011u, 0x10201110u, 0x10201202u, 0x10210000u, 0x10210101u, 0x10211010u,
    0x10211100u, 0x10211221u, 0x10220010u, 0x11000000u, 0x11000002u, 0x11000011u, 0x11000020u, 0x11000101u,
    0x11000110u, 0x11000200u, 0x11000220u, 0x11001001u, 0x11001010u, 0x11001100u, 0x11002000u, 0x11010001u,
    0x11010010u, 0x11010100u, 0x11010111u, 0x11011000u, 0x11011002u, 0x11020000u, 0x11021010u, 0x11022222u,
    0x11100001u, 0x11100010u, 0x11100100u, 0x11100201u, 0x11101000u, 0x11101200u, 0x11102001u, 0x11102201u,
    0x11110000u, 0x11110020u, 0x11112000u, 0x11112020u, 0x11120201u, 0x11121220u, 0x11122201u, 0x11200000u,
    0x11200110u, 0x11201001u, 0x11201100u, 0x11202121u, 0x11210122u, 0x11211000u, 0x11211002u, 0x11222011u,
    0x12000001u, 0x12000010u, 0x12000100u, 0x12001000u, 0x12001110u, 0x12001202u, 0x12002012u, 0x12002221u,
    0x12010000u, 0x12020210u, 0x12021022u, 0x12022102u, 0x12100000u, 0x12100112u, 0x12110100u, 0x12111000u,
    0x12111011u, 0x12112210u, 0x12200001u, 0x12201222u, 0x12210211u, 0x12220012u, 0x12221110u, 0x12221202u,
    0x20000000u, 0x20000002u, 0x20000011u, 0x20000020u, 0x20000101u, 0x20000110u, 0x20000200u, 0x20000222u,
    0x20001001u, 0x20001010u, 0x20001100u, 0x20002000u, 0x20002002u, 0x20002220u, 0x20002222u, 0x20010001u,
    0x20010010u, 0x20010012u, 0x20010100u, 0x20011000u, 0x20011101u, 0x20011121u, 0x20020000u, 0x20020200u,
    0x20022000u, 0x20022002u, 0x20022200u, 0x20022220u, 0x20100001u, 0x20100010u, 0x20100100u, 0x20100102u,
    0x20100111u, 0x20101000u, 0x20101200u, 0x20102021u, 0x20110000u, 0x20111010u, 0x20112211u, 0x20120120u,
    0x20121222u, 0x20200000u, 0x20200020u, 0x20200211u, 0x20201122u, 0x20202000u, 0x20202002u, 0x20202220u,
    0x20210012u, 0x20220202u, 0x20222000u, 0x20222020u, 0x20222112u, 0x20222220u, 0x21000001u, 0x21000010u,
    0x21000100u, 0x21001000u, 0x21001112u, 0x21002010u, 0x21010000u, 0x21010202u, 0x21011210u, 0x21021112u,
    0x21022021u, 0x21100000u, 0x21100011u, 0x21101010u, 0x21101100u, 0x21101120u, 0x21110221u, 0x21112100u,
    0x21112102u, 0x21121001u, 0x21201101u, 0x21202212u, 0x21211021u, 0x21220111u, 0x21221200u, 0x22000000u,
    0x22000002u, 0x22000020u, 0x22000022u, 0x22000200u, 0x22000222u, 0x22002200u, 0x22011101u, 0x22011121u,
    0x22012212u, 0x22020000u, 0x22020002u, 0x22020020u, 0x22020222u, 0x22022000u, 0x22022200u, 0x22101000u,
    0x22102111u, 0x22121211u, 0x22122120u, 0x22200022u, 0x22200200u, 0x22200202u, 0x22200220u, 0x22202200u,
    0x22202220u, 0x22210010u, 0x22212010u, 0x22212012u, 0x22220220u, 0x22220222u, 0x22222101u, 0x22222222u,
};
__device__ const uint32_t kIq2xsSign[128] = {
    0x00000000u, 0x40000004u, 0x40000040u, 0x00000044u, 0x40000400u, 0x00000404u, 0x00000440u, 0x40000444u,
    0x40004000u, 0x00004004u, 0x00004040u, 0x40004044u, 0x00004400u, 0x40004404u, 0x40004440u, 0x00004444u,
    0x40040000u, 0x00040004u, 0x00040040u, 0x40040044u, 0x00040400u, 0x40040404u, 0x40040440u, 0x00040444u,
    0x00044000u, 0x40044004u, 0x40044040u, 0x00044044u, 0x40044400u, 0x00044404u, 0x00044440u, 0x40044444u,
    0x40400000u, 0x00400004u, 0x00400040u, 0x40400044u, 0x00400400u, 0x40400404u, 0x40400440u, 0x00400444u,
    0x00404000u, 0x40404004u, 0x40404040u, 0x00404044u, 0x40404400u, 0x00404404u, 0x00404440u, 0x40404444u,
    0x00440000u, 0x40440004u, 0x40440040u, 0x00440044u, 0x40440400u, 0x00440404u, 0x00440440u, 0x40440444u,
    0x40444000u, 0x00444004u, 0x00444040u, 0x40444044u, 0x00444400u, 0x40444404u, 0x40444440u, 0x00444444u,
    0x44000000u, 0x04000004u, 0x04000040u, 0x44000044u, 0x04000400u, 0x44000404u, 0x44000440u, 0x04000444u,
    0x04004000u, 0x44004004u, 0x44004040u, 0x04004044u, 0x44004400u, 0x04004404u, 0x04004440u, 0x44004444u,
    0x04040000u, 0x44040004u, 0x44040040u, 0x04040044u, 0x44040400u, 0x04040404u, 0x04040440u, 0x44040444u,
    0x44044000u, 0x04044004u, 0x04044040u, 0x44044044u, 0x04044400u, 0x44044404u, 0x44044440u, 0x04044444u,
    0x04400000u, 0x44400004u, 0x44400040u, 0x04400044u, 0x44400400u, 0x04400404u, 0x04400440u, 0x44400444u,
    0x44404000u, 0x04404004u, 0x04404040u, 0x44404044u, 0x04404400u, 0x44404404u, 0x44404440u, 0x04404444u,
    0x44440000u, 0x04440004u, 0x04440040u, 0x44440044u, 0x04440400u, 0x44440404u, 0x44440440u, 0x04440444u,
    0x04444000u, 0x44444004u, 0x44444040u, 0x04444044u, 0x44444400u, 0x04444404u, 0x04444440u, 0x44444444u,
};

template <int TPP, int CVT, int TG>
__global__ void __launch_bounds__(NT, 1) gu_kernel(const Chunk* __restrict__ chunks, const char* __restrict__ w,
                                                  const int64_t* __restrict__ xoff, const block_q8_1_mmq* __restrict__ xq,
                                                  const int total_rows, float* __restrict__ dst) {
    static_assert(TPP % TG == 0, "");
    __shared__ uint32_t tabA_s[512];
    __shared__ uint32_t tabB_s[128];
    __shared__ __align__(16) int act_s[TPP][4][16];   // [token][sub-block][8 int8x4 words split to 16 int16 pairs]
    __shared__ float dy_s[TPP][4];
    __shared__ int wsm[NT * CWS];
    for (int i = threadIdx.x; i < 512; i += NT) tabA_s[i] = kIq2xsCode[i];
    for (int i = threadIdx.x; i < 128; i += NT) tabB_s[i] = kIq2xsSign[i];
    int r0;
    const Chunk c = cta_work<PIECES, NT>(chunks, r0);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int ci = lane & 15, rsub = lane >> 4;   // lanes 0-9 stage one row, 16-25 the next; ci = the compact word
    const bool cact = ci < CW;
    const char* __restrict__ Wr = w + xoff[c.j] + (size_t) (r0 + 2 * warp + rsub) * ROWB;   // + k * 2 NW rows
    int* __restrict__ wsl = wsm + (2 * warp + rsub) * CWS + ci;                              // + k * 2 NW * CWS
    const int* cw = wsm + threadIdx.x * CWS;
    for (int p0 = 0; p0 < c.rows; p0 += TPP) {
        const int nt = min(TPP, c.rows - p0);
        const int trow0 = c.row0 + p0;
        float acc[TPP];
#pragma unroll
        for (int t = 0; t < TPP; ++t) acc[t] = 0.0f;
#pragma unroll 1
        for (int hc = 0; hc < NSTEP; ++hc) {
            const int kc = hc >> 1, h = hc & 1;
            // compact word ci's byte offset inside the block: qs words (ci 0-7), scales (ci 8); d (ci 9) apart
            const int boff = ci < 8 ? 2 + 32 * h + 4 * ci : 66 + 4 * h;
            __syncthreads();
            if (cact) {
                const char* blk0 = Wr + kc * BB;
                if (kc & 1) {   // odd block: words aligned, one load each; d = high half of the word at blk - 2
                    const int o = ci < 9 ? boff : -2, sh = ci < 9 ? 0 : 16;
                    int v[KR];
#pragma unroll
                    for (int k = 0; k < KR; ++k) v[k] = __ldg((const int*) (blk0 + (size_t) k * 2 * NW * ROWB + o));
#pragma unroll
                    for (int k = 0; k < KR; ++k) wsl[k * 2 * NW * CWS] = __funnelshift_r(v[k], v[k], sh);
                } else {        // even block: words 2 bytes into a word, two loads + funnel shift; d = low half at blk
                    const int o0 = ci < 9 ? boff - 2 : 0, o1 = ci < 9 ? boff + 2 : 0, sh = ci < 9 ? 16 : 0;
                    int v0[KR], v1[KR];
#pragma unroll
                    for (int k = 0; k < KR; ++k) {
                        const char* b = blk0 + (size_t) k * 2 * NW * ROWB;
                        v0[k] = __ldg((const int*) (b + o0));
                        v1[k] = __ldg((const int*) (b + o1));
                    }
#pragma unroll
                    for (int k = 0; k < KR; ++k) wsl[k * 2 * NW * CWS] = __funnelshift_r(v0[k], v1[k], sh);
                }
            }
            stage_act<TPP>(xq, hc, total_rows, trow0, nt, act_s, dy_s);
            __syncthreads();
#pragma unroll 1
            for (int s = 0; s < 4; ++s) {   // cw: [0..7] qs u16 pairs (sub-block s = words 2s, 2s+1), [8] scales, [9] d
                int wv[16];
                float dx0, dx1;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const uint32_t q2 = ((uint32_t) cw[2 * s + (l >> 1)] >> (16 * (l & 1))) & 0xFFFFu;
                    const uint32_t sel = tabA_s[q2 & 0x1FFu] | tabB_s[q2 >> 9];
                    split2((int) __byte_perm(PX, PY, sel), wv[4 * l + 0], wv[4 * l + 1]);
                    split2((int) __byte_perm(PX, PY, sel >> 16), wv[4 * l + 2], wv[4 * l + 3]);
                }
                scales16((cw[8] >> (8 * s)) & 0xFF, cw[9], dx0, dx1);
                mac16<TPP, CVT, TG>(wv, dx0, dx1, s, act_s, dy_s, nt, acc);
            }
        }
#pragma unroll
        for (int t = 0; t < TPP; ++t)
            if (t < nt) dst[(size_t) (trow0 + t) * GU_ROWS + r0 + threadIdx.x] = acc[t];
    }
}
}  // namespace xs2

// ---- IQ2_S: 82-B blocks; per row and step the grid bytes qs[16h..] (4 words), the sign bytes qs[32 + 16h..] (4),
// qh[4h..4h + 3], scales[4h..4h + 3] and d: 11 words, row stride 11, staged by lanes 0-10 / 16-26 of each warp.  The grid
// index is 10 bits (qs | qh bits), the sign byte is explicit: tables generated from ggml's iq2s_grid.
namespace s2 {
constexpr int BB = (int) sizeof(block_iq2_s), ROWB = NB * BB, CW = 11, CWS = 11, KR = NT / (2 * NW);
static_assert(BB == 82, "block_iq2_s");

__device__ const uint32_t kIq2sCode[1024] = {
    0x00000000u, 0x00000002u, 0x00000011u, 0x00000020u, 0x00000022u, 0x00000101u, 0x00000110u, 0x00000112u,
    0x00000121u, 0x00000200u, 0x00000202u, 0x00000211u, 0x00000220u, 0x00001001u, 0x00001010u, 0x00001012u,
    0x00001021u, 0x00001100u, 0x00001102u, 0x00001111u, 0x00001120u, 0x00001201u, 0x00001210u, 0x00001212u,
    0x00001221u, 0x00002000u, 0x00002002u, 0x00002011u, 0x00002020u, 0x00002101u, 0x00002110u, 0x00002200u,
    0x00002211u, 0x00002222u, 0x00010001u, 0x00010010u, 0x00010012u, 0x00010021u, 0x00010100u, 0x00010102u,
    0x00010111u, 0x00010120u, 0x00010201u, 0x00010210u, 0x00011000u, 0x00011002u, 0x00011011u, 0x00011020u,
    0x00011101u, 0x00011110u, 0x00011112u, 0x00011121u, 0x00011200u, 0x00011211u, 0x00011220u, 0x00012001u,
    0x00012010u, 0x00012100u, 0x00012102u, 0x00012111u, 0x00012201u, 0x00012210u, 0x00020000u, 0x00020002u,
    0x00020011u, 0x00020020u, 0x00020101u, 0x00020110u, 0x00020200u, 0x00020222u, 0x00021001u, 0x00021010u,
    0x00021012u, 0x00021021u, 0x00021100u, 0x00021111u, 0x00022000u, 0x00022011u, 0x00022022u, 0x00022110u,
    0x00022202u, 0x00100001u, 0x00100010u, 0x00100012u, 0x00100021u, 0x00100100u, 0x00100102u, 0x00100111u,
    0x00100120u, 0x00100201u, 0x00100210u, 0x00100212u, 0x00100221u, 0x00101000u, 0x00101002u, 0x00101011u,
    0x00101020u, 0x00101022u, 0x00101101u, 0x00101110u, 0x00101112u, 0x00101121u, 0x00101200u, 0x00101202u,
    0x00101211u, 0x00102001u, 0x00102010u, 0x00102012u, 0x00102021u, 0x00102100u, 0x00102111u, 0x00102120u,
    0x00102201u, 0x00102210u, 0x00110000u, 0x00110002u, 0x00110011u, 0x00110020u, 0x00110022u, 0x00110101u,
    0x00110110u, 0x00110112u, 0x00110121u, 0x00110200u, 0x00110211u, 0x00110220u, 0x00111001u, 0x00111010u,
    0x00111012u, 0x00111021u, 0x00111100u, 0x00111102u, 0x00111111u, 0x00111120u, 0x00111201u, 0x00111210u,
    0x00112000u, 0x00112002u, 0x00112011u, 0x00112020u, 0x00112101u, 0x00112110u, 0x00112200u, 0x00120001u,
    0x00120010u, 0x00120012u, 0x00120021u, 0x00120100u, 0x00120111u, 0x00121000u, 0x00121011u, 0x00121020u,
    0x00121101u, 0x00121110u, 0x00121200u, 0x00122001u, 0x00122010u, 0x00122100u, 0x00200000u, 0x00200002u,
    0x00200011u, 0x00200020u, 0x00200101u, 0x00200110u, 0x00200112u, 0x00200121u, 0x00200200u, 0x00200211u,
    0x00200222u, 0x00201001u, 0x00201010u, 0x00201012u, 0x00201021u, 0x00201100u, 0x00201102u, 0x00201111u,
    0x00201120u, 0x00201201u, 0x00201210u, 0x00202000u, 0x00202011u, 0x00202110u, 0x00202222u, 0x00210001u,
    0x00210010u, 0x00210100u, 0x00210102u, 0x00210111u, 0x00210120u, 0x00210201u, 0x00211000u, 0x00211011u,
    0x00211020u, 0x00211101u, 0x00211110u, 0x00211200u, 0x00212001u, 0x00212100u, 0x00220000u, 0x00220101u,
    0x00220110u, 0x00220202u, 0x00220220u, 0x00220222u, 0x00221100u, 0x00222121u, 0x01000001u, 0x01000010u,
    0x01000012u, 0x01000021u, 0x01000100u, 0x01000102u, 0x01000111u, 0x01000120u, 0x01000201u, 0x01000210u,
    0x01000212u, 0x01001000u, 0x01001002u, 0x01001011u, 0x01001020u, 0x01001101u, 0x01001110u, 0x01001112u,
    0x01001121u, 0x01001200u, 0x01001202u, 0x01001211u, 0x01001220u, 0x01002001u, 0x01002010u, 0x01002012u,
    0x01002100u, 0x01002111u, 0x01002120u, 0x01002201u, 0x01002210u, 0x01010000u, 0x01010002u, 0x01010011u,
    0x01010020u, 0x01010022u, 0x01010101u, 0x01010110u, 0x01010112u, 0x01010121u, 0x01010200u, 0x01010202u,
    0x01010211u, 0x01010220u, 0x01011001u, 0x01011010u, 0x01011012u, 0x01011021u, 0x01011100u, 0x01011102u,
    0x01011111u, 0x01011120u, 0x01011201u, 0x01011210u, 0x01012000u, 0x01012002u, 0x01012011u, 0x01012020u,
    0x01012101u, 0x01012110u, 0x01020001u, 0x01020010u, 0x01020021u, 0x01020100u, 0x01020111u, 0x01020201u,
    0x01020210u, 0x01021000u, 0x01021011u, 0x01021101u, 0x01021110u, 0x01022001u, 0x01022010u, 0x01022100u,
    0x01100000u, 0x01100002u, 0x01100011u, 0x01100020u, 0x01100101u, 0x01100110u, 0x01100112u, 0x01100121u,
    0x01100200u, 0x01100211u, 0x01100220u, 0x01101001u, 0x01101010u, 0x01101012u, 0x01101021u, 0x01101100u,
    0x01101102u, 0x01101111u, 0x01101120u, 0x01101201u, 0x01101210u, 0x01102000u, 0x01102002u, 0x01102011u,
    0x01102020u, 0x01102101u, 0x01102110u, 0x01102200u, 0x01110001u, 0x01110010u, 0x01110012u, 0x01110021u,
    0x01110100u, 0x01110102u, 0x01110111u, 0x01110120u, 0x01110201u, 0x01110210u, 0x01111000u, 0x01111002u,
    0x01111011u, 0x01111020u, 0x01111101u, 0x01111110u, 0x01111200u, 0x01112001u, 0x01112010u, 0x01112100u,
    0x01120000u, 0x01120011u, 0x01120020u, 0x01120101u, 0x01120110u, 0x01120200u, 0x01121001u, 0x01121010u,
    0x01121100u, 0x01122000u, 0x01122222u, 0x01200001u, 0x01200010u, 0x01200012u, 0x01200021u, 0x01200100u,
    0x01200111u, 0x01200120u, 0x01200201u, 0x01201000u, 0x01201002u, 0x01201011u, 0x01201020u, 0x01201101u,
    0x01201110u, 0x01201200u, 0x01202001u, 0x01202010u, 0x01210000u, 0x01210002u, 0x01210011u, 0x01210020u,
    0x01210101u, 0x01210110u, 0x01210200u, 0x01211001u, 0x01211010u, 0x01211100u, 0x01211221u, 0x01212202u,
    0x01220010u, 0x01220100u, 0x01221000u, 0x01221112u, 0x02000000u, 0x02000002u, 0x02000011u, 0x02000020u,
    0x02000101u, 0x02000110u, 0x02000112u, 0x02000121u, 0x02000200u, 0x02000211u, 0x02000222u, 0x02001001u,
    0x02001010u, 0x02001100u, 0x02001102u, 0x02001111u, 0x02001210u, 0x02002000u, 0x02002022u, 0x02002110u,
    0x02002222u, 0x02010001u, 0x02010010u, 0x02010100u, 0x02010102u, 0x02010111u, 0x02010201u, 0x02011000u,
    0x02011002u, 0x02011011u, 0x02011101u, 0x02011110u, 0x02011200u, 0x02012001u, 0x02012010u, 0x02012100u,
    0x02020000u, 0x02020022u, 0x02020202u, 0x02020220u, 0x02020222u, 0x02021010u, 0x02021100u, 0x02022020u,
    0x02022022u, 0x02022220u, 0x02100001u, 0x02100010u, 0x02100012u, 0x02100021u, 0x02100100u, 0x02100111u,
    0x02100120u, 0x02100201u, 0x02100210u, 0x02101000u, 0x02101002u, 0x02101011u, 0x02101020u, 0x02101101u,
    0x02101110u, 0x02101200u, 0x02102001u, 0x02102010u, 0x02102100u, 0x02110000u, 0x02110011u, 0x02110020u,
    0x02110101u, 0x02110110u, 0x02110200u, 0x02111001u, 0x02111010u, 0x02111100u, 0x02111212u, 0x02112000u,
    0x02120001u, 0x02120010u, 0x02120100u, 0x02121000u, 0x02121121u, 0x02200000u, 0x02200011u, 0x02200101u,
    0x02200110u, 0x02201001u, 0x02201010u, 0x02201100u, 0x02202022u, 0x02202222u, 0x02210001u, 0x02210010u,
    0x02210100u, 0x02212111u, 0x02220022u, 0x02220202u, 0x02221210u, 0x02222020u, 0x02222022u, 0x10000001u,
    0x10000010u, 0x10000012u, 0x10000021u, 0x10000100u, 0x10000102u, 0x10000111u, 0x10000120u, 0x10000122u,
    0x10000201u, 0x10000210u, 0x10000212u, 0x10001000u, 0x10001002u, 0x10001011u, 0x10001020u, 0x10001022u,
    0x10001101u, 0x10001110u, 0x10001112u, 0x10001121u, 0x10001200u, 0x10001202u, 0x10001211u, 0x10002001u,
    0x10002010u, 0x10002100u, 0x10002111u, 0x10002120u, 0x10002201u, 0x10002210u, 0x10010000u, 0x10010002u,
    0x10010011u, 0x10010020u, 0x10010101u, 0x10010110u, 0x10010112u, 0x10010121u, 0x10010200u, 0x10010202u,
    0x10010211u, 0x10011001u, 0x10011010u, 0x10011012u, 0x10011021u, 0x10011100u, 0x10011102u, 0x10011111u,
    0x10011120u, 0x10011201u, 0x10011210u, 0x10012000u, 0x10012002u, 0x10012011u, 0x10012020u, 0x10012101u,
    0x10012110u, 0x10012200u, 0x10020001u, 0x10020010u, 0x10020100u, 0x10020102u, 0x10020111u, 0x10020120u,
    0x10020210u, 0x10021000u, 0x10021011u, 0x10021020u, 0x10021101u, 0x10021110u, 0x10021200u, 0x10022001u,
    0x10022010u, 0x10100000u, 0x10100002u, 0x10100011u, 0x10100020u, 0x10100022u, 0x10100101u, 0x10100110u,
    0x10100112u, 0x10100121u, 0x10100200u, 0x10100202u, 0x10100211u, 0x10100220u, 0x10101001u, 0x10101010u,
    0x10101012u, 0x10101021u, 0x10101100u, 0x10101102u, 0x10101111u, 0x10101120u, 0x10101201u, 0x10101210u,
    0x10102000u, 0x10102002u, 0x10102011u, 0x10102020u, 0x10102101u, 0x10102110u, 0x10102200u, 0x10110001u,
    0x10110010u, 0x10110012u, 0x10110021u, 0x10110100u, 0x10110102u, 0x10110111u, 0x10110120u, 0x10110201u,
    0x10110210u, 0x10111000u, 0x10111002u, 0x10111011u, 0x10111020u, 0x10111101u, 0x10111110u, 0x10111200u,
    0x10111222u, 0x10112001u, 0x10112010u, 0x10112100u, 0x10120000u, 0x10120002u, 0x10120011u, 0x10120020u,
    0x10120101u, 0x10120110u, 0x10120200u, 0x10121001u, 0x10121010u, 0x10121100u, 0x10122000u, 0x10122211u,
    0x10200001u, 0x10200010u, 0x10200021u, 0x10200100u, 0x10200102u, 0x10200111u, 0x10200120u, 0x10200201u,
    0x10200210u, 0x10201000u, 0x10201002u, 0x10201011u, 0x10201020u, 0x10201101u, 0x10201110u, 0x10201200u,
    0x10202010u, 0x10202100u, 0x10210000u, 0x10210002u, 0x10210011u, 0x10210020u, 0x10210101u, 0x10210110u,
    0x10210200u, 0x10211001u, 0x10211010u, 0x10211100u, 0x10212000u, 0x10212112u, 0x10220001u, 0x10220010u,
    0x10220100u, 0x10221000u, 0x11000000u, 0x11000002u, 0x11000011u, 0x11000020u, 0x11000101u, 0x11000110u,
    0x11000112u, 0x11000121u, 0x11000200u, 0x11000202u, 0x11000211u, 0x11000220u, 0x11001001u, 0x11001010u,
    0x11001012u, 0x11001021u, 0x11001100u, 0x11001102u, 0x11001111u, 0x11001120u, 0x11001201u, 0x11001210u,
    0x11002000u, 0x11002002u, 0x11002011u, 0x11002020u, 0x11002101u, 0x11002110u, 0x11010001u, 0x11010010u,
    0x11010012u, 0x11010021u, 0x11010100u, 0x11010102u, 0x11010111u, 0x11010120u, 0x11010201u, 0x11010210u,
    0x11011000u, 0x11011002u, 0x11011011u, 0x11011020u, 0x11011101u, 0x11011110u, 0x11011200u, 0x11012001u,
    0x11012010u, 0x11012100u, 0x11020000u, 0x11020011u, 0x11020020u, 0x11020101u, 0x11020110u, 0x11020200u,
    0x11021001u, 0x11021010u, 0x11021100u, 0x11021221u, 0x11022000u, 0x11100001u, 0x11100010u, 0x11100012u,
    0x11100021u, 0x11100100u, 0x11100102u, 0x11100111u, 0x11100120u, 0x11100201u, 0x11100210u, 0x11101000u,
    0x11101002u, 0x11101011u, 0x11101020u, 0x11101101u, 0x11101110u, 0x11101200u, 0x11102001u, 0x11102010u,
    0x11102100u, 0x11110000u, 0x11110002u, 0x11110011u, 0x11110020u, 0x11110101u, 0x11110110u, 0x11110200u,
    0x11111001u, 0x11111010u, 0x11111100u, 0x11112000u, 0x11120001u, 0x11120010u, 0x11120100u, 0x11120212u,
    0x11121000u, 0x11200000u, 0x11200002u, 0x11200011u, 0x11200020u, 0x11200101u, 0x11200110u, 0x11200200u,
    0x11201001u, 0x11201010u, 0x11201100u, 0x11201122u, 0x11202000u, 0x11210001u, 0x11210010u, 0x11210100u,
    0x11211000u, 0x11220000u, 0x11220121u, 0x11222011u, 0x11222220u, 0x12000001u, 0x12000010u, 0x12000012u,
    0x12000100u, 0x12000102u, 0x12000111u, 0x12000120u, 0x12000201u, 0x12000210u, 0x12001000u, 0x12001011u,
    0x12001020u, 0x12001101u, 0x12001110u, 0x12001200u, 0x12002010u, 0x12002100u, 0x12010000u, 0x12010002u,
    0x12010011u, 0x12010020u, 0x12010101u, 0x12010110u, 0x12010200u, 0x12011001u, 0x12011010u, 0x12011100u,
    0x12012000u, 0x12012121u, 0x12020010u, 0x12020100u, 0x12021000u, 0x12021112u, 0x12022201u, 0x12100000u,
    0x12100011u, 0x12100020u, 0x12100101u, 0x12100110u, 0x12100200u, 0x12101001u, 0x12101010u, 0x12101100u,
    0x12102000u, 0x12110001u, 0x12110010u, 0x12110100u, 0x12111000u, 0x12111022u, 0x12111220u, 0x12112102u,
    0x12120000u, 0x12122110u, 0x12200001u, 0x12200010u, 0x12200100u, 0x12201211u, 0x12202120u, 0x12210000u,
    0x12210222u, 0x12221002u, 0x12222201u, 0x20000000u, 0x20000002u, 0x20000011u, 0x20000020u, 0x20000101u,
    0x20000110u, 0x20000121u, 0x20000200u, 0x20000211u, 0x20001001u, 0x20001010u, 0x20001100u, 0x20001102u,
    0x20001111u, 0x20001120u, 0x20001201u, 0x20002000u, 0x20002011u, 0x20002101u, 0x20002110u, 0x20010001u,
    0x20010010u, 0x20010021u, 0x20010100u, 0x20010102u, 0x20010111u, 0x20010120u, 0x20010201u, 0x20010210u,
    0x20011000u, 0x20011002u, 0x20011011u, 0x20011020u, 0x20011101u, 0x20011110u, 0x20012001u, 0x20012010u,
    0x20012100u, 0x20012221u, 0x20020000u, 0x20020011u, 0x20020022u, 0x20020101u, 0x20020110u, 0x20021001u,
    0x20021010u, 0x20021100u, 0x20100001u, 0x20100010u, 0x20100012u, 0x20100021u, 0x20100100u, 0x20100102u,
    0x20100111u, 0x20100120u, 0x20100201u, 0x20101000u, 0x20101002u, 0x20101011u, 0x20101020u, 0x20101101u,
    0x20101110u, 0x20101200u, 0x20102001u, 0x20102010u, 0x20102100u, 0x20110000u, 0x20110002u, 0x20110011u,
    0x20110020u, 0x20110101u, 0x20110110u, 0x20110200u, 0x20111001u, 0x20111010u, 0x20111100u, 0x20112000u,
    0x20112022u, 0x20120001u, 0x20120010u, 0x20120100u, 0x20120221u, 0x20121000u, 0x20200000u, 0x20200011u,
    0x20200101u, 0x20200110u, 0x20201001u, 0x20201010u, 0x20201100u, 0x20202202u, 0x20210001u, 0x20210010u,
    0x20211000u, 0x20211211u, 0x20220202u, 0x20221120u, 0x20221122u, 0x20222002u, 0x20222202u, 0x21000001u,
    0x21000010u, 0x21000021u, 0x21000100u, 0x21000102u, 0x21000111u, 0x21000120u, 0x21000210u, 0x21001000u,
    0x21001002u, 0x21001011u, 0x21001020u, 0x21001101u, 0x21001110u, 0x21001200u, 0x21002001u, 0x21002010u,
    0x21002100u, 0x21010000u, 0x21010011u, 0x21010101u, 0x21010110u, 0x21011001u, 0x21011010u, 0x21011100u,
    0x21011122u, 0x21020001u, 0x21020010u, 0x21020100u, 0x21021000u, 0x21022212u, 0x21100000u, 0x21100002u,
    0x21100011u, 0x21100020u, 0x21100101u, 0x21100110u, 0x21100200u, 0x21101001u, 0x21101010u, 0x21101100u,
    0x21102000u, 0x21102112u, 0x21110001u, 0x21110010u, 0x21110100u, 0x21111000u, 0x21112120u, 0x21112201u,
    0x21120000u, 0x21121012u, 0x21121210u, 0x21200001u, 0x21200010u, 0x21200100u, 0x21200212u, 0x21201000u,
    0x21202221u, 0x21210000u, 0x21211021u, 0x21211102u, 0x21222100u, 0x22000000u, 0x22000011u, 0x22000022u,
    0x22000110u, 0x22000202u, 0x22000222u, 0x22001001u, 0x22001010u, 0x22001100u, 0x22002202u, 0x22002222u,
    0x22011000u, 0x22011211u, 0x22020002u, 0x22020022u, 0x22020202u, 0x22020220u, 0x22020222u, 0x22022002u,
    0x22022020u, 0x22022022u, 0x22022220u, 0x22100001u, 0x22100010u, 0x22100100u, 0x22101000u, 0x22102021u,
    0x22102210u, 0x22110000u, 0x22110121u, 0x22121101u, 0x22200022u, 0x22200220u, 0x22202202u, 0x22211110u,
    0x22212012u, 0x22220020u, 0x22220022u, 0x22220200u, 0x22220202u, 0x22220220u, 0x22222020u, 0x22222222u,
};
__device__ const uint32_t kIq2sSign[256] = {
    0x00000000u, 0x00000004u, 0x00000040u, 0x00000044u, 0x00000400u, 0x00000404u, 0x00000440u, 0x00000444u,
    0x00004000u, 0x00004004u, 0x00004040u, 0x00004044u, 0x00004400u, 0x00004404u, 0x00004440u, 0x00004444u,
    0x00040000u, 0x00040004u, 0x00040040u, 0x00040044u, 0x00040400u, 0x00040404u, 0x00040440u, 0x00040444u,
    0x00044000u, 0x00044004u, 0x00044040u, 0x00044044u, 0x00044400u, 0x00044404u, 0x00044440u, 0x00044444u,
    0x00400000u, 0x00400004u, 0x00400040u, 0x00400044u, 0x00400400u, 0x00400404u, 0x00400440u, 0x00400444u,
    0x00404000u, 0x00404004u, 0x00404040u, 0x00404044u, 0x00404400u, 0x00404404u, 0x00404440u, 0x00404444u,
    0x00440000u, 0x00440004u, 0x00440040u, 0x00440044u, 0x00440400u, 0x00440404u, 0x00440440u, 0x00440444u,
    0x00444000u, 0x00444004u, 0x00444040u, 0x00444044u, 0x00444400u, 0x00444404u, 0x00444440u, 0x00444444u,
    0x04000000u, 0x04000004u, 0x04000040u, 0x04000044u, 0x04000400u, 0x04000404u, 0x04000440u, 0x04000444u,
    0x04004000u, 0x04004004u, 0x04004040u, 0x04004044u, 0x04004400u, 0x04004404u, 0x04004440u, 0x04004444u,
    0x04040000u, 0x04040004u, 0x04040040u, 0x04040044u, 0x04040400u, 0x04040404u, 0x04040440u, 0x04040444u,
    0x04044000u, 0x04044004u, 0x04044040u, 0x04044044u, 0x04044400u, 0x04044404u, 0x04044440u, 0x04044444u,
    0x04400000u, 0x04400004u, 0x04400040u, 0x04400044u, 0x04400400u, 0x04400404u, 0x04400440u, 0x04400444u,
    0x04404000u, 0x04404004u, 0x04404040u, 0x04404044u, 0x04404400u, 0x04404404u, 0x04404440u, 0x04404444u,
    0x04440000u, 0x04440004u, 0x04440040u, 0x04440044u, 0x04440400u, 0x04440404u, 0x04440440u, 0x04440444u,
    0x04444000u, 0x04444004u, 0x04444040u, 0x04444044u, 0x04444400u, 0x04444404u, 0x04444440u, 0x04444444u,
    0x40000000u, 0x40000004u, 0x40000040u, 0x40000044u, 0x40000400u, 0x40000404u, 0x40000440u, 0x40000444u,
    0x40004000u, 0x40004004u, 0x40004040u, 0x40004044u, 0x40004400u, 0x40004404u, 0x40004440u, 0x40004444u,
    0x40040000u, 0x40040004u, 0x40040040u, 0x40040044u, 0x40040400u, 0x40040404u, 0x40040440u, 0x40040444u,
    0x40044000u, 0x40044004u, 0x40044040u, 0x40044044u, 0x40044400u, 0x40044404u, 0x40044440u, 0x40044444u,
    0x40400000u, 0x40400004u, 0x40400040u, 0x40400044u, 0x40400400u, 0x40400404u, 0x40400440u, 0x40400444u,
    0x40404000u, 0x40404004u, 0x40404040u, 0x40404044u, 0x40404400u, 0x40404404u, 0x40404440u, 0x40404444u,
    0x40440000u, 0x40440004u, 0x40440040u, 0x40440044u, 0x40440400u, 0x40440404u, 0x40440440u, 0x40440444u,
    0x40444000u, 0x40444004u, 0x40444040u, 0x40444044u, 0x40444400u, 0x40444404u, 0x40444440u, 0x40444444u,
    0x44000000u, 0x44000004u, 0x44000040u, 0x44000044u, 0x44000400u, 0x44000404u, 0x44000440u, 0x44000444u,
    0x44004000u, 0x44004004u, 0x44004040u, 0x44004044u, 0x44004400u, 0x44004404u, 0x44004440u, 0x44004444u,
    0x44040000u, 0x44040004u, 0x44040040u, 0x44040044u, 0x44040400u, 0x44040404u, 0x44040440u, 0x44040444u,
    0x44044000u, 0x44044004u, 0x44044040u, 0x44044044u, 0x44044400u, 0x44044404u, 0x44044440u, 0x44044444u,
    0x44400000u, 0x44400004u, 0x44400040u, 0x44400044u, 0x44400400u, 0x44400404u, 0x44400440u, 0x44400444u,
    0x44404000u, 0x44404004u, 0x44404040u, 0x44404044u, 0x44404400u, 0x44404404u, 0x44404440u, 0x44404444u,
    0x44440000u, 0x44440004u, 0x44440040u, 0x44440044u, 0x44440400u, 0x44440404u, 0x44440440u, 0x44440444u,
    0x44444000u, 0x44444004u, 0x44444040u, 0x44444044u, 0x44444400u, 0x44444404u, 0x44444440u, 0x44444444u,
};

template <int TPP, int CVT, int TG>
__global__ void __launch_bounds__(NT, 1) gu_kernel(const Chunk* __restrict__ chunks, const char* __restrict__ w,
                                                  const int64_t* __restrict__ xoff, const block_q8_1_mmq* __restrict__ xq,
                                                  const int total_rows, float* __restrict__ dst) {
    static_assert(TPP % TG == 0, "");
    __shared__ __align__(16) uint32_t tabA_s[1024];
    __shared__ __align__(16) uint32_t tabB_s[256];
    __shared__ __align__(16) int act_s[TPP][4][16];
    __shared__ float dy_s[TPP][4];
    __shared__ int wsm[NT * CWS];
    for (int i = threadIdx.x; i < 1024; i += NT) tabA_s[i] = kIq2sCode[i];
    for (int i = threadIdx.x; i < 256; i += NT) tabB_s[i] = kIq2sSign[i];
    int r0;
    const Chunk c = cta_work<PIECES, NT>(chunks, r0);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int ci = lane & 15, rsub = lane >> 4;   // lanes 0-10 stage one row, 16-26 the next; ci = the compact word
    const bool cact = ci < CW;
    const char* __restrict__ Wr = w + xoff[c.j] + (size_t) (r0 + 2 * warp + rsub) * ROWB;   // + k * 2 NW rows
    int* __restrict__ wsl = wsm + (2 * warp + rsub) * CWS + ci;                              // + k * 2 NW * CWS
    const int* cw = wsm + threadIdx.x * CWS;
    for (int p0 = 0; p0 < c.rows; p0 += TPP) {
        const int nt = min(TPP, c.rows - p0);
        const int trow0 = c.row0 + p0;
        float acc[TPP];
#pragma unroll
        for (int t = 0; t < TPP; ++t) acc[t] = 0.0f;
#pragma unroll 1
        for (int hc = 0; hc < NSTEP; ++hc) {
            const int kc = hc >> 1, h = hc & 1;
            // compact word ci's byte offset inside the block (ci 0-9); d (ci 10) sits at +0
            const int boff = ci < 4 ? 2 + 16 * h + 4 * ci : ci < 8 ? 18 + 16 * h + 4 * ci : ci == 8 ? 66 + 4 * h : 74 + 4 * h;
            __syncthreads();
            if (cact) {
                const char* blk0 = Wr + kc * BB;
                if (kc & 1) {   // odd block: words aligned, one load each; d = high half of the word at blk - 2
                    const int o = ci < 10 ? boff : -2, sh = ci < 10 ? 0 : 16;
                    int v[KR];
#pragma unroll
                    for (int k = 0; k < KR; ++k) v[k] = __ldg((const int*) (blk0 + (size_t) k * 2 * NW * ROWB + o));
#pragma unroll
                    for (int k = 0; k < KR; ++k) wsl[k * 2 * NW * CWS] = __funnelshift_r(v[k], v[k], sh);
                } else {        // even block: words 2 bytes into a word, two loads + funnel shift; d = low half at blk
                    const int o0 = ci < 10 ? boff - 2 : 0, o1 = ci < 10 ? boff + 2 : 0, sh = ci < 10 ? 16 : 0;
                    int v0[KR], v1[KR];
#pragma unroll
                    for (int k = 0; k < KR; ++k) {
                        const char* b = blk0 + (size_t) k * 2 * NW * ROWB;
                        v0[k] = __ldg((const int*) (b + o0));
                        v1[k] = __ldg((const int*) (b + o1));
                    }
#pragma unroll
                    for (int k = 0; k < KR; ++k) wsl[k * 2 * NW * CWS] = __funnelshift_r(v0[k], v1[k], sh);
                }
            }
            stage_act<TPP>(xq, hc, total_rows, trow0, nt, act_s, dy_s);
            __syncthreads();
#pragma unroll 1
            for (int s = 0; s < 4; ++s) {   // cw: [0..3] grid bytes, [4..7] sign bytes, [8] qh, [9] scales, [10] d
                int wv[16];
                float dx0, dx1;
                const int qs_packed = cw[s], signs_packed_32 = cw[4 + s], qh = (cw[8] >> (8 * s)) & 0xFF;
#pragma unroll
                for (int l = 0; l < QR2_S; ++l) {
                    const int idx = ((qs_packed >> (8 * l)) & 0xFF) | ((qh << (8 - 2 * l)) & 0x300);
                    const uint32_t sel = tabA_s[idx] | tabB_s[(signs_packed_32 >> (8 * l)) & 0xFF];
                    split2((int) __byte_perm(PX, PY, sel), wv[4 * l + 0], wv[4 * l + 1]);
                    split2((int) __byte_perm(PX, PY, sel >> 16), wv[4 * l + 2], wv[4 * l + 3]);
                }
                scales16((cw[9] >> (8 * s)) & 0xFF, cw[10], dx0, dx1);
                mac16<TPP, CVT, TG>(wv, dx0, dx1, s, act_s, dy_s, nt, acc);
            }
        }
#pragma unroll
        for (int t = 0; t < TPP; ++t)
            if (t < nt) dst[(size_t) (trow0 + t) * GU_ROWS + r0 + threadIdx.x] = acc[t];
    }
}
}  // namespace s2

template <typename F>
void go(F kernel, const Layer& L, cudaStream_t s) {
    kernel<<<(unsigned) (L.nchunks * PIECES), NT, 0, s>>>(L.chunks, (const char*) L.w, L.xoff, (const block_q8_1_mmq*) L.xq,
                                                         (int) L.rows, L.gu);
}

}  // namespace

// tokens per pass by u (the measured crossovers): IQ2_XXS 8 below 5 rows per expert, 16 below 25, then 32 (4-token
// groups); IQ2_XS 8 below 6, 16 below 20, then 16 in 4-token groups; IQ2_S 8 below 5, 16 below 25, then 32
void gu_iq2_xxs(const Layer& L, double u, cudaStream_t s) {
    if (L.nchunks <= 0) return;
    if (u < 5.0) go(xxs2::gu_kernel<8, 3, 2>, L, s);
    else if (u < 25.0) go(xxs2::gu_kernel<16, 3, 2>, L, s);
    else go(xxs2::gu_kernel<32, 2, 4>, L, s);
    ck(cudaGetLastError(), "gate/up IQ2_XXS");
}

void gu_iq2_xs(const Layer& L, double u, cudaStream_t s) {
    if (L.nchunks <= 0) return;
    if (u < 6.0) go(xs2::gu_kernel<8, 1, 2>, L, s);
    else if (u < 20.0) go(xs2::gu_kernel<16, 0, 2>, L, s);
    else go(xs2::gu_kernel<16, 1, 4>, L, s);
    ck(cudaGetLastError(), "gate/up IQ2_XS");
}

void gu_iq2_s(const Layer& L, double u, cudaStream_t s) {
    if (L.nchunks <= 0) return;
    if (u < 5.0) go(s2::gu_kernel<8, 0, 2>, L, s);
    else if (u < 25.0) go(s2::gu_kernel<16, 1, 2>, L, s);
    else go(s2::gu_kernel<32, 1, 4>, L, s);
    ck(cudaGetLastError(), "gate/up IQ2_S");
}

}  // namespace strata::prefill::ws::detail
