// src/prefill/moe_ws_gu_iq3.cu - weight-stationary gate/up products (see moe_ws.hpp): IQ3_XXS and IQ3_S, [1280 x 2560]
// read in place.  A CTA = 128 threads = 128 weight rows of one expert (a row per thread, all of the chunk's activation
// rows TPP at a time, the sums in registers); K advances 128 values a step (half a 256-value block = 4 sub-blocks):
// the step's bytes of the 128 rows and the step's q8_1 block of each token go to shared memory (the activations split
// to int16 pairs once), then each thread decodes its row's 4 sub-blocks once per pass and runs them against the tokens.
// Bits: per 32-value sub-block MMQ's int8 values (ggml_cuda_mmq_load_tiles_iq3_*: grid entries, its __vcmpne4 sign
// masks applied as (g ^ s) + (s & 0x01010101) == __vsub4(g ^ s, s), every grid byte >= 1) and scale expression, then
// MMQ's epilogue acc = fma(dx * dy, float(sum), acc), sub-blocks in K order.  (Measured on a GP100 die:
// ws / per-group MMQ 0.31-0.35 at 2.5 rows per expert, 0.49-0.52 at 10, 0.67-0.76 at 35,
// 0.78-0.79 at 160; bit-identical, compute-sanitizer clean with exact-size matrices.)
//
// Staging without per-lane branches: rows are 980 / 1100 B, so block kc starts 2 * (kc & 1) bytes into a 4-byte word
// for the whole CTA - an even block's words are two aligned loads and a funnel shift (the second load ends at most 2
// bytes into block kc + 1 of the same row), an odd block's are aligned; d sits the other way round (an odd block's d is
// the high half of the word 2 bytes before it, inside block kc - 1).  Every load of a lane is issued before its stores,
// and nothing outside a row is read (the last row's last block is odd).
#include "common.cuh"
#include "mmq.cuh"

#include "moe_ws.cuh"

#include <cstdio>
#include <cstdlib>

namespace strata::prefill::ws::detail {
namespace {

constexpr int GU_ROWS = 1280, GU_K = 2560, NB = GU_K / QK_K, NSTEP = 2 * NB, NT = 128, PIECES = GU_ROWS / NT;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill ws: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// ---- IQ3_XXS: 98-B blocks; per row and step 8 grid-index words, 4 aux words (signs + scale) and d (stride 13).
// Every lane slot is one word type: grid words rows tid/8 + 16 i (word tid%8), aux words rows tid/4 + 32 i (word
// tid%4), d the thread's own row.  Token groups of TG: one branch per group on a partial pass (a group's tokens past
// the count run on copies of row nt - 1 and are not written), none on a full pass.
namespace xxs3 {
constexpr int BB = (int) sizeof(block_iq3_xxs), ROWB = NB * BB, CWS = 13;
static_assert(BB == 98 && ROWB % 4 == 0, "block_iq3_xxs");

template <int TPP, int CH, int TG, bool FULL>
__device__ __forceinline__ void sb_compute(const int* __restrict__ cw, const int s, const uint32_t* __restrict__ grid,
                                           const int2* __restrict__ lut, const int (*act_s)[4][16],
                                           const float (*dy_s)[4], const int nt, float (&acc)[TPP]) {
    int wv[16];
    float dx;
    {
        const uint32_t qa = (uint32_t) cw[2 * s + 0], qb = (uint32_t) cw[2 * s + 1];   // 8 grid indices
        const uint32_t aux32 = (uint32_t) cw[8 + s];
#pragma unroll
        for (int l = 0; l < QR3_XXS; ++l) {
            const uint32_t qw = l < 2 ? qa : qb;
            const uint32_t g0 = grid[(qw >> (16 * (l & 1))) & 0xFF], g1 = grid[(qw >> (16 * (l & 1) + 8)) & 0xFF];
            const int2 sg = lut[(aux32 >> (7 * l)) & 0x7F];
            split2((int) ((g0 ^ (uint32_t) sg.x) + ((uint32_t) sg.x & 0x01010101u)), wv[4 * l + 0], wv[4 * l + 1]);
            split2((int) ((g1 ^ (uint32_t) sg.y) + ((uint32_t) sg.y & 0x01010101u)), wv[4 * l + 2], wv[4 * l + 3]);
        }
        const int ls = aux32 >> 28;
        const float d = __ushort_as_half((unsigned short) (cw[12] & 0xFFFF));
        dx = (ls*d + d/2)/2;   // MMQ's expression, verbatim
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
    constexpr int NG = 8, RG = NT / 8;          // grid-index-word slots: rows tid/8 + i*RG, word tid%8
    constexpr int NX = 4, RX = NT / 4;          // aux-word slots: rows tid/4 + i*RX, word tid%4
    constexpr int NA = TPP * 32 / NT;
    __shared__ uint32_t grid_s[256];
    __shared__ int2 lut_s[128];
    __shared__ __align__(16) int act_s[TPP][4][16];
    __shared__ float dy_s[TPP][4];
    __shared__ int wsm[NT * CWS];
    const int tid = threadIdx.x;
    for (int i = tid; i < 256; i += NT) grid_s[i] = iq3xxs_grid[i];
    for (int i = tid; i < 128; i += NT) {
        const uint32_t signs = unpack_ksigns((uint8_t) i);
        lut_s[i] = make_int2(__vcmpne4(signs & 0x08040201, 0), __vcmpne4(signs & 0x80402010, 0));
    }
    int r0;
    const Chunk c = cta_work<PIECES, NT>(chunks, r0);
    const char* __restrict__ W = w + xoff[c.j] + (size_t) r0 * ROWB;
    const char* __restrict__ Wg = W + (size_t) (tid >> 3) * ROWB + 4 * (tid & 7);        // + i*RG*ROWB + kc*BB + 32h
    const char* __restrict__ Wx = W + (size_t) (tid >> 2) * ROWB + 64 + 4 * (tid & 3);   // + i*RX*ROWB + kc*BB + 16h
    const char* __restrict__ Wd = W + (size_t) tid * ROWB;
    int* __restrict__ sg_ = wsm + (tid >> 3) * CWS + (tid & 7);
    int* __restrict__ sx_ = wsm + (tid >> 2) * CWS + 8 + (tid & 3);
    int* __restrict__ sd_ = wsm + tid * CWS + 12;
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
            const int* xr = (const int*) (xq + (size_t) hc * total_rows + trow0);
            __syncthreads();
            int av[NA], dv = 0;
#pragma unroll
            for (int i = 0; i < NA; ++i) {
                const int idx = tid + i * NT, t = idx >> 5;
                av[i] = __ldg(xr + min(t, nt - 1) * 36 + 4 + (idx & 31));   // clamped: no branch, never past row nt-1
            }
            if (TPP * 4 == NT || tid < TPP * 4) dv = __ldg(xr + min(tid >> 2, nt - 1) * 36 + (tid & 3));
            const char* bg = Wg + kc * BB + 32 * h;   // even kc: the aligned word 2 bytes before the data word
            const char* bx = Wx + kc * BB + 16 * h;
            if (kc & 1) {   // odd block: data words aligned (at +2); d = high half of the word 2 bytes before the block
                int vg[NG], vx[NX];
#pragma unroll
                for (int i = 0; i < NG; ++i) vg[i] = __ldg((const int*) (bg + 2 + (size_t) i * RG * ROWB));
#pragma unroll
                for (int i = 0; i < NX; ++i) vx[i] = __ldg((const int*) (bx + 2 + (size_t) i * RX * ROWB));
                const int vd = __ldg((const int*) (Wd + kc * BB - 2));
#pragma unroll
                for (int i = 0; i < NG; ++i) sg_[i * RG * CWS] = vg[i];
#pragma unroll
                for (int i = 0; i < NX; ++i) sx_[i * RX * CWS] = vx[i];
                *sd_ = (int) ((uint32_t) vd >> 16);
            } else {        // even block: data words 2 bytes into a word: two aligned words + funnel shift; d = low half
                int lg[NG], hg[NG], lx[NX], hx[NX];
#pragma unroll
                for (int i = 0; i < NG; ++i) {
                    lg[i] = __ldg((const int*) (bg + (size_t) i * RG * ROWB));
                    hg[i] = __ldg((const int*) (bg + 4 + (size_t) i * RG * ROWB));
                }
#pragma unroll
                for (int i = 0; i < NX; ++i) {
                    lx[i] = __ldg((const int*) (bx + (size_t) i * RX * ROWB));
                    hx[i] = __ldg((const int*) (bx + 4 + (size_t) i * RX * ROWB));
                }
                const int vd = __ldg((const int*) (Wd + kc * BB));
#pragma unroll
                for (int i = 0; i < NG; ++i) sg_[i * RG * CWS] = (int) __funnelshift_r((uint32_t) lg[i], (uint32_t) hg[i], 16);
#pragma unroll
                for (int i = 0; i < NX; ++i) sx_[i * RX * CWS] = (int) __funnelshift_r((uint32_t) lx[i], (uint32_t) hx[i], 16);
                *sd_ = vd;
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
                for (int s = 0; s < 4; ++s) sb_compute<TPP, CH, TG, true>(cw, s, grid_s, lut_s, act_s, dy_s, nt, acc);
            } else {
#pragma unroll 1
                for (int s = 0; s < 4; ++s) sb_compute<TPP, CH, TG, false>(cw, s, grid_s, lut_s, act_s, dy_s, nt, acc);
            }
        }
#pragma unroll
        for (int t = 0; t < TPP; ++t)
            if (t < nt) dst[(size_t) (trow0 + t) * GU_ROWS + r0 + tid] = acc[t];
    }
}
}  // namespace xxs3

// ---- IQ3_S: 110-B blocks; per row and step a compact copy of the bytes the 4 sub-blocks read - qs 32 B, signs 16 B,
// qh 4 B, d and the 2 scale bytes: 14 words, row stride 15 - staged by lanes 0-13 / 16-29 of each warp (two rows per
// warp and k), each word built from two aligned loads by one PRMT.  The sign masks come from a 256-entry table of MMQ's
// own __vcmpne4 expressions.  Token groups of TG behind one `t < nt` branch.
namespace s3 {
constexpr int BB = 110, ROWB = NB * BB, CW = 14, CWS = 15, NW = NT / 32, KN = NT / (2 * NW);

__device__ __forceinline__ int2 mmq_signs(const int sb) {   // ggml_cuda_mmq_load_tiles_iq3_s's, verbatim
    const int s0 = __vcmpne4(((sb & 0x03) << 7) | ((sb & 0x0C) << 21), 0x00000000);
    const int s1 = __vcmpne4(((sb & 0x30) << 3) | ((sb & 0xC0) << 17), 0x00000000);
    return make_int2(s0, s1);
}

// sub-block s of the staged half: kqsx = 4 h + s's 8 int8x4 words and scale.  cw: [0..7] qs, [8..11] sign bytes,
// [12] qh, [13] d | the half's 2 scale bytes << 16
__device__ __forceinline__ float dec(const int* __restrict__ cw, const int s, const uint32_t* __restrict__ grid,
                                     const int2* __restrict__ lut, int (&q)[8]) {
    const int2 qs_packed = make_int2(cw[2 * s + 0], cw[2 * s + 1]);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = (cw[12] >> (8 * s)) & 0xFF;
    const int signs_packed_32 = cw[8 + s];
    const int w13 = cw[13];
#pragma unroll
    for (int l = 0; l < QR3_S; ++l) {
        const int2 grid_pos = make_int2(grid[qs[2 * l + 0] | ((qh << (8 - 2 * l)) & 0x100)],
                                        grid[qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 0x100)]);
        const int2 sg = lut[(signs_packed_32 >> (8 * l)) & 0xFF];
        q[2 * l + 0] = (grid_pos.x ^ sg.x) + (sg.x & 0x01010101);
        q[2 * l + 1] = (grid_pos.y ^ sg.y) + (sg.y & 0x01010101);
    }
    const int ls = 1 + 2 * (((w13 >> (16 + 8 * (s >> 1))) >> (4 * (s & 1))) & 0x0F);
    const float d = __ushort_as_half((unsigned short) (w13 & 0xFFFF));
    return ls * d;
}

// compact word ci of step half h in a block of parity odd: its two aligned words (offsets from the block) and selector
__device__ __forceinline__ void stage_map(const int ci, const int h, const int odd, int& o0, int& o1, int& sel) {
    const int off = ci < 8 ? 2 + 32 * h + 4 * ci : ci < 12 ? 74 + 16 * h + 4 * (ci - 8) : 66 + 4 * h;
    const bool dw = ci == 13;
    o0 = dw ? (odd ? -2 : 0) : (odd ? off : off - 2);
    o1 = dw ? (odd ? 106 : 104 + 4 * h) : (odd ? off : off + 2);
    sel = dw ? (odd ? (h ? 0x7632 : 0x5432) : (h ? 0x5410 : 0x7610)) : (odd ? 0x3210 : 0x5432);
}

template <int TPP, int TG>
__device__ __forceinline__ void sb_compute(const int* __restrict__ cw, const int s, const uint32_t* __restrict__ grid,
                                           const int2* __restrict__ lut, const int (*act_s)[4][16], const float (*dy_s)[4],
                                           const int nt, float (&acc)[TPP]) {
    int q[8];
    const float dx = dec(cw, s, grid, lut, q);
    int wv[16];
#pragma unroll
    for (int m = 0; m < 8; ++m) split2(q[m], wv[2 * m], wv[2 * m + 1]);
#pragma unroll
    for (int t0 = 0; t0 < TPP; t0 += TG) {
        if (t0 < nt) {
#pragma unroll
            for (int t = t0; t < t0 + TG; ++t) {
                const int4* a4 = (const int4*) act_s[t][s];
                const int4 a0 = a4[0], a1 = a4[1], a2 = a4[2], a3 = a4[3];
                const float dy = dy_s[t][s];
                int s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                s0 = mad2(wv[0], a0.x, s0); s1 = mad2(wv[1], a0.y, s1); s2 = mad2(wv[2], a0.z, s2); s3 = mad2(wv[3], a0.w, s3);
                s0 = mad2(wv[4], a1.x, s0); s1 = mad2(wv[5], a1.y, s1); s2 = mad2(wv[6], a1.z, s2); s3 = mad2(wv[7], a1.w, s3);
                s0 = mad2(wv[8], a2.x, s0); s1 = mad2(wv[9], a2.y, s1); s2 = mad2(wv[10], a2.z, s2); s3 = mad2(wv[11], a2.w, s3);
                s0 = mad2(wv[12], a3.x, s0); s1 = mad2(wv[13], a3.y, s1); s2 = mad2(wv[14], a3.z, s2); s3 = mad2(wv[15], a3.w, s3);
                acc[t] = __fmaf_rn(__fmul_rn(dx, dy), __int2float_rn((s0 + s1) + (s2 + s3)), acc[t]);
            }
        }
    }
}

template <int TPP, int TG>
__global__ void __launch_bounds__(NT) gu_kernel(const Chunk* __restrict__ chunks, const char* __restrict__ w,
                                                const int64_t* __restrict__ xoff, const block_q8_1_mmq* __restrict__ xq,
                                                const int total_rows, float* __restrict__ dst) {
    static_assert(TPP % TG == 0, "");
    constexpr size_t KSTR = (size_t) 2 * NW * ROWB;   // a staging lane's rows are 2 * NW apart
    __shared__ uint32_t grid_s[512];
    __shared__ int2 lut_s[256];
    __shared__ __align__(16) int act_s[TPP][4][16];
    __shared__ float dy_s[TPP][4];
    __shared__ int wsm[NT * CWS];
    for (int i = threadIdx.x; i < 512; i += NT) grid_s[i] = iq3s_grid[i];
    for (int i = threadIdx.x; i < 256; i += NT) lut_s[i] = mmq_signs(i);
    int r0;
    const Chunk c = cta_work<PIECES, NT>(chunks, r0);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int ci = lane & 15, rsub = lane >> 4;   // lanes 0-13 stage one row, 16-29 the next; ci = the compact word
    const bool cact = ci < CW;
    const char* __restrict__ Wr = w + xoff[c.j] + (size_t) (r0 + 2 * warp + rsub) * ROWB;   // + k * KSTR
    int* __restrict__ wsl = wsm + (2 * warp + rsub) * CWS + ci;                              // + k * 2 NW * CWS
    for (int p0 = 0; p0 < c.rows; p0 += TPP) {
        const int nt = min(TPP, c.rows - p0);
        const int trow0 = c.row0 + p0;
        float acc[TPP];
#pragma unroll
        for (int t = 0; t < TPP; ++t) acc[t] = 0.0f;
#pragma unroll 1
        for (int hc = 0; hc < NSTEP; ++hc) {
            __syncthreads();
            if (cact) {   // branch-free inside: every lane issues its 2 * KN loads, then its KN stores
                int o0, o1, sel;
                stage_map(ci, hc & 1, (hc >> 1) & 1, o0, o1, sel);
                const char* b = Wr + (hc >> 1) * BB;
                int v0[KN], v1[KN];
#pragma unroll
                for (int k = 0; k < KN; ++k) v0[k] = __ldg((const int*) (b + k * KSTR + o0));
#pragma unroll
                for (int k = 0; k < KN; ++k) v1[k] = __ldg((const int*) (b + k * KSTR + o1));
#pragma unroll
                for (int k = 0; k < KN; ++k) wsl[k * 2 * NW * CWS] = (int) __byte_perm(v0[k], v1[k], sel);
            }
            {   // the step's activations: nt consecutive 144-B blocks, loads first, then split + store
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
            __syncthreads();
            const int* cw = wsm + threadIdx.x * CWS;
#pragma unroll 1
            for (int s = 0; s < 4; ++s) sb_compute<TPP, TG>(cw, s, grid_s, lut_s, act_s, dy_s, nt, acc);
        }
#pragma unroll
        for (int t = 0; t < TPP; ++t)
            if (t < nt) dst[(size_t) (trow0 + t) * GU_ROWS + r0 + threadIdx.x] = acc[t];
    }
}
}  // namespace s3

template <typename F>
void go(F kernel, const Layer& L, cudaStream_t s) {
    kernel<<<(unsigned) (L.nchunks * PIECES), NT, 0, s>>>(L.chunks, (const char*) L.w, L.xoff, (const block_q8_1_mmq*) L.xq,
                                                         (int) L.rows, L.gu);
}

}  // namespace

// tokens per pass by u (the measured crossovers): IQ3_XXS 8 below 5 rows per expert, 16 below 25, then 32 (4-token
// groups); IQ3_S 8 below 4, 16 below 15, then 32
void gu_iq3_xxs(const Layer& L, double u, cudaStream_t s) {
    if (L.nchunks <= 0) return;
    if (u < 5.0) go(xxs3::gu_kernel<8, 4, 2>, L, s);
    else if (u < 25.0) go(xxs3::gu_kernel<16, 4, 2>, L, s);
    else go(xxs3::gu_kernel<32, 2, 4>, L, s);
    ck(cudaGetLastError(), "gate/up IQ3_XXS");
}

void gu_iq3_s(const Layer& L, double u, cudaStream_t s) {
    if (L.nchunks <= 0) return;
    if (u < 4.0) go(s3::gu_kernel<8, 2>, L, s);
    else if (u < 15.0) go(s3::gu_kernel<16, 2>, L, s);
    else go(s3::gu_kernel<32, 4>, L, s);
    ck(cudaGetLastError(), "gate/up IQ3_S");
}

}  // namespace strata::prefill::ws::detail
