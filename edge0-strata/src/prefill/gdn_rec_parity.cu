// src/prefill/gdn_rec_parity.cu - the prompt path's DeltaNet recurrence, bit for bit and timed.
//
// src/prefill/kernels.cu runs gdn_rec_kh_kernel (the three value heads that share a key head in one thread, the
// inputs staged in blocks of 8 tokens) on CUDA sm_80+ cards that hold all 64 of its blocks at once, and
// gdn_rec_cols_pipe_kernel elsewhere and with STRATA_GDN_KEYHEAD=0.  Both are copied below verbatim, with the output
// norm (gdn_out_norm_kernel) and the norm as it was before, which also stored the normalized value in FP32 although
// nothing reads it.  The two recurrences do the same arithmetic in the same order per value head and column, so the
// output, the state after it and the FP16 output must be the same bits.  STRATA_GDN_CHUNKED=1's recurrence in chunks
// (copied below too) sums in another order: it must stay within 1e-4 of gdn_rec_kh_kernel (largest |difference| over
// largest |value|, the output and the state), also at T = 8192 and 32768; up to 8192 both are also measured against
// the recurrence in FP64.  STRATA_GDN_TC=1's tensor-core scan (gdn_tc.cuh, included below) is held to the same 1e-4,
// also on stress inputs (the state 100x smaller, 50x larger or zero, v 50x larger, slow gates, coherent keys).
//
// Synthetic inputs of the model's shape (q/k L2-normalized per key head as gdn_l2_kernel leaves them), no model.
//   gdn_rec_parity            the bit checks (T from 1 to 4099, a non-zero state as after an earlier chunk)
//   gdn_rec_parity --long     and the tensor-core scan at T = 51820 (needs ~7 GB)
//   gdn_rec_parity --selftest-frag   the tensor-core scan's fragment helpers alone, one warp each, against FP64
//   gdn_rec_parity --bench-tc the tensor-core series alone (gdn_rec_kh_kernel and the scan alternating, nothing else:
//                             for A / B runs of two builds back to back)
//   gdn_rec_parity --bench    and the timings at T = 2048, 8192 and 32768 (one layer, all 48 value heads); the two
//                             variants' launches alternate, since a card under load steps its clock down after a
//                             fraction of a second (see native_grouped_parity); then both at 1 to 6 blocks per
//                             SM, as on cards with more or fewer SMs (scale), and both with the engine's grids on
//                             32 to all of this card's SMs, the others held by a sleeping kernel (fewer_sms)
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

namespace {

constexpr int S = 128, HK = 16, HV = 48, C = 10240;   // as kernels.cu: head size, key heads, value heads, q|k|v width
constexpr int RG = 4, RPG = S / RG;
constexpr int CB = 32, NCB = S / CB;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ uint16_t hf(float f) { return __half_as_ushort(__float2half_rn(f)); }

// ------------------------------------------------------------------ the engine's kernels (src/prefill/kernels.cu)
// gdn_rec_cols_kernel with the next token's inputs (q/k rows, v, gate, beta) loaded into registers while this token
// computes (software pipelining).  The same arithmetic in the same order: the same bits, and the same CB-column split.
// STRATA_GDN_PIPELINE=0: gdn_rec_cols_kernel.
__global__ void __launch_bounds__(CB * RG) gdn_rec_cols_pipe_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                                      const float* __restrict__ gate,
                                                                      const float* __restrict__ beta,
                                                                      float* __restrict__ oc_out, int64_t T) {
    constexpr int NT = CB * RG, LPT = S / NT;   // threads, q/k rows loaded per thread
    __shared__ float sk[S], sq[S], red[RG][CB];
    const int head = blockIdx.x / NCB, cb = blockIdx.x % NCB;
    const int c = threadIdx.x, rg = threadIdx.y, tid = rg * CB + c, col = cb * CB + c;
    const int qh = head % HK;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * HV + head) * S + col;
    const size_t rs = (size_t) HV * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * rs];
    float nq[LPT], nk[LPT], nv = 0.0f, ng = 0.0f, nb = 0.0f;
    auto fetch = [&](int64_t t) {
        const float* ht = h + t * C;
#pragma unroll
        for (int u = 0; u < LPT; ++u) { nq[u] = ht[qh * S + tid + u * NT]; nk[u] = ht[HK * S + qh * S + tid + u * NT]; }
        nv = ht[2 * HK * S + head * S + col];
        ng = gate[t * HV + head];
        nb = beta[t * HV + head];
    };
    if (T > 0) fetch(0);
    for (int64_t t = 0; t < T; ++t) {
        float cq[LPT], ck[LPT];
#pragma unroll
        for (int u = 0; u < LPT; ++u) { cq[u] = nq[u]; ck[u] = nk[u]; }
        const float cv = nv, cg = ng, cbt = nb;
        __syncthreads();
#pragma unroll
        for (int u = 0; u < LPT; ++u) { sq[tid + u * NT] = cq[u]; sk[tid + u * NT] = ck[u]; }
        __syncthreads();
        if (t + 1 < T) fetch(t + 1);
        const float g = __expf(cg);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red[rg][c] = kv;
        __syncthreads();
        const float kv_col = red[0][c] + red[1][c] + red[2][c] + red[3][c];
        const float delta = (cv - g * kv_col) * cbt;
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        __syncthreads();
        red[rg][c] = o;
        __syncthreads();
        if (rg == 0) oc_out[t * HV * S + head * S + col] = (red[0][c] + red[1][c] + red[2][c] + red[3][c]) * rsqrtf((float) S);
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * rs] = s[r];
}
// The recurrence with one thread for the three value heads that share a key head (head % HK): column c of heads
// qh, qh + 16 and qh + 32, row group rg.  gdn_rec_cols_pipe_kernel spends its time in shared memory, not in
// arithmetic: every thread of a warp needs the same 32 q and k values per token (the k twice), and a warp receives one
// such broadcast value per clock however wide the load.  Here every q/k value a thread loads feeds three heads, and a
// token's k row goes into registers once for both of its uses.  The inputs come in blocks of GDN_TB tokens, copied to
// shared memory by cp.async while the block before computes (one token ahead is shorter than a load from L2 takes),
// and the two cross-row-group sums have their own arrays, so a token needs 2 __syncthreads instead of 5: the second
// one of a token orders every read of rkv before the next token's writes, the next token's first one every read of ro
// before the writes after it.  64 blocks instead of 192.  Per value head and column the same arithmetic in the same
// order: the same bits (src/prefill/gdn_rec_parity.cu checks them and times the variants: 1.41x on a 4080 Super).
// sm_80+ (gdn_keyhead_ok); STRATA_GDN_KEYHEAD=0: gdn_rec_cols_pipe_kernel.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#define STRATA_GDN_CP_ASYNC 0   // Turing builds: plain copies (never launched there, see gdn_keyhead_ok)
#else
#define STRATA_GDN_CP_ASYNC 1
#endif
__device__ __forceinline__ void gdn_cp4(float* smem, const float* gmem) {
#if STRATA_GDN_CP_ASYNC
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"((unsigned) __cvta_generic_to_shared(smem)), "l"(gmem));
#else
    *smem = *gmem;
#endif
}
__device__ __forceinline__ void gdn_cp16(float* smem, const float* gmem) {
#if STRATA_GDN_CP_ASYNC
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(smem)), "l"(gmem));
#else
    *reinterpret_cast<float4*>(smem) = *reinterpret_cast<const float4*>(gmem);
#endif
}
__device__ __forceinline__ void gdn_cp_commit() {
#if STRATA_GDN_CP_ASYNC
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
__device__ __forceinline__ void gdn_cp_wait_prev() {   // every group but the newest has landed
#if STRATA_GDN_CP_ASYNC
    asm volatile("cp.async.wait_group 1;\n" ::);
#endif
}
constexpr int GDN_TB = 8, VPK = HV / HK;   // tokens per staged block, value heads per key head
__global__ void __launch_bounds__(CB * RG) gdn_rec_kh_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                               const float* __restrict__ gate,
                                                               const float* __restrict__ beta,
                                                               float* __restrict__ oc_out, int64_t T) {
    constexpr int TB = GDN_TB, NT = CB * RG, QKP = S / 4, VP = CB / 4;   // threads, 16-byte pieces of a q/k row, of v
    __shared__ __align__(16) float sq[2][TB][S];
    __shared__ __align__(16) float sk[2][TB][S];
    __shared__ __align__(16) float sv[2][TB][VPK][CB];
    __shared__ float sg[2][TB][VPK], sb[2][TB][VPK], rkv[VPK][RG][CB], ro[VPK][RG][CB];
    const int qh = blockIdx.x / NCB, cb = blockIdx.x % NCB;
    const int c = threadIdx.x, rg = threadIdx.y, tid = rg * CB + c, col = cb * CB + c;
    float s[VPK][RPG];
    const size_t rs = (size_t) HV * S;
#pragma unroll
    for (int j = 0; j < VPK; ++j) {
        const float* base = state + ((size_t) (rg * RPG) * HV + qh + j * HK) * S + col;
#pragma unroll
        for (int r = 0; r < RPG; ++r) s[j][r] = base[r * rs];
    }
    const int64_t nblk = (T + TB - 1) / TB;
    auto stage = [&](int64_t k) {   // tokens [k * TB, k * TB + TB) into buffer k & 1
        const int bb = (int) (k & 1);
        const int64_t t0 = k * TB;
        for (int p = tid; p < TB * 2 * QKP; p += NT) {
            const int i = p / (2 * QKP), w = p % (2 * QKP), isk = w / QKP, jj = (w % QKP) * 4;
            if (t0 + i < T)
                gdn_cp16(isk ? &sk[bb][i][jj] : &sq[bb][i][jj], h + (t0 + i) * C + (isk ? HK * S : 0) + qh * S + jj);
        }
        for (int p = tid; p < TB * VPK * VP; p += NT) {
            const int i = p / (VPK * VP), w = p % (VPK * VP), j = w / VP, jj = (w % VP) * 4;
            if (t0 + i < T)
                gdn_cp16(&sv[bb][i][j][jj], h + (t0 + i) * C + 2 * HK * S + (qh + j * HK) * S + cb * CB + jj);
        }
        for (int p = tid; p < 2 * TB * VPK; p += NT) {
            const int isb = p / (TB * VPK), w = p % (TB * VPK), i = w / VPK, j = w % VPK;
            if (t0 + i < T)
                gdn_cp4(isb ? &sb[bb][i][j] : &sg[bb][i][j], (isb ? beta : gate) + (t0 + i) * HV + qh + j * HK);
        }
    };
    if (nblk > 0) stage(0);
    gdn_cp_commit();
    for (int64_t k = 0; k < nblk; ++k) {
        // buffer (k + 1) & 1 was read by block k - 1, whose last token's second __syncthreads every thread has passed
        if (k + 1 < nblk) stage(k + 1);
        gdn_cp_commit();
        gdn_cp_wait_prev();
        __syncthreads();
        const int bb = (int) (k & 1);
        const int n = (int) ((T - k * TB) < TB ? (T - k * TB) : TB);
        for (int i = 0; i < n; ++i) {
            const int64_t t = k * TB + i;
            float kc[RPG];
#pragma unroll
            for (int r = 0; r < RPG; ++r) kc[r] = sk[bb][i][rg * RPG + r];
            float g[VPK], kv[VPK], delta[VPK], o[VPK];
#pragma unroll
            for (int j = 0; j < VPK; ++j) { g[j] = __expf(sg[bb][i][j]); kv[j] = 0.0f; o[j] = 0.0f; }
#pragma unroll
            for (int r = 0; r < RPG; ++r)
#pragma unroll
                for (int j = 0; j < VPK; ++j) kv[j] = fmaf(s[j][r], kc[r], kv[j]);
#pragma unroll
            for (int j = 0; j < VPK; ++j) rkv[j][rg][c] = kv[j];
            __syncthreads();
#pragma unroll
            for (int j = 0; j < VPK; ++j) {
                const float kv_col = rkv[j][0][c] + rkv[j][1][c] + rkv[j][2][c] + rkv[j][3][c];
                delta[j] = (sv[bb][i][j][c] - g[j] * kv_col) * sb[bb][i][j];
            }
#pragma unroll
            for (int r = 0; r < RPG; ++r) {
                const float qr = sq[bb][i][rg * RPG + r];
#pragma unroll
                for (int j = 0; j < VPK; ++j) {
                    s[j][r] = fmaf(g[j], s[j][r], kc[r] * delta[j]);
                    o[j] = fmaf(s[j][r], qr, o[j]);
                }
            }
#pragma unroll
            for (int j = 0; j < VPK; ++j) ro[j][rg][c] = o[j];
            __syncthreads();
            if (rg < VPK)   // row group j writes head j's output
                oc_out[t * HV * S + (qh + rg * HK) * S + col] =
                    (ro[rg][0][c] + ro[rg][1][c] + ro[rg][2][c] + ro[rg][3][c]) * rsqrtf((float) S);
        }
    }
#pragma unroll
    for (int j = 0; j < VPK; ++j) {
        float* base = state + ((size_t) (rg * RPG) * HV + qh + j * HK) * S + col;
#pragma unroll
        for (int r = 0; r < RPG; ++r) base[r * rs] = s[j][r];
    }
}

// src/prefill/kernels.cu's chunked recurrence (STRATA_GDN_CHUNKED=1), copied as it is there
// STRATA_GDN_CHUNKED=1 (opt-in): the recurrence in chunks of GCH tokens (the WY form of the gated delta rule, as
// flash-linear-attention computes it), all in FP32.  Per value head, with g_t = exp(gate_t), gamma_t the gate's sum
// from the chunk's start to t (inclusive) and S0 the state before the chunk:
//   A[t][i] = beta_t exp(gamma_t - gamma_i) k_t.k_i (i < t),   T = (I + A)^-1 (unit lower triangular),
//   Y_t = beta_t (v_t - exp(gamma_t) S0^T k_t),   D = T Y (the delta rule's corrections, one per token),
//   o_t = exp(gamma_t) S0^T q_t + sum_{i <= t} exp(gamma_t - gamma_i) (q_t.k_i) D_i,
//   S = exp(gamma_last) S0 + sum_t exp(gamma_last - gamma_t) k_t D_t^T,
// the same state and outputs as the token-by-token kernels up to FP32 rounding (another order of the sums).
// gdn_chunk_prep_kernel builds T, P = exp(gamma_t - gamma_i) q_t.k_i (i <= t) and gamma for every chunk of a
// super-block at once (a block per chunk and key head: its three value heads share k and q); gdn_chunk_scan_kernel then
// walks the super-block's chunks per (value head, 16 columns): the state slice in registers, the products as small
// register-tiled matmuls from shared memory.  Every exponent is <= 0 (gates are negative), so nothing overflows.
constexpr int GCH = 32;            // tokens per chunk (the prep's solve is a lane per token)
constexpr int GSB = 2048;          // tokens per super-block: the prep's scratch is GSB / GCH chunks of T, P and gamma
constexpr int GDV = 16;            // value columns per scan block
constexpr int GKP = S + 4;         // the scan's padded q / k rows (8 rows of a warp land on 8 banks)
constexpr int GTP = GCH + 1;       // padded T / P rows
static_assert(GCH == 32, "gdn_chunk_prep_kernel: a lane per token of the chunk");
constexpr size_t kChunkPrepSmem = sizeof(float) * (2 * GCH * (S + 1) + 2 * GCH * GTP);   // A over k / q
static_assert(VPK * GCH * GTP <= 2 * GCH * (S + 1), "gdn_chunk_prep_kernel: A fits where k and q were");
constexpr size_t kChunkScanSmem = sizeof(float) * (3 * GCH * GKP + VPK * (S * GDV + 3 * GCH * GDV));

__global__ void __launch_bounds__(256) gdn_chunk_prep_kernel(const float* __restrict__ h, const float* __restrict__ gate,
                                                             const float* __restrict__ beta, float* __restrict__ tm,
                                                             float* __restrict__ pm, float* __restrict__ gm, int64_t t0,
                                                             int64_t T) {
    extern __shared__ __align__(16) float gsm[];
    float* sk = gsm;                       // [GCH][S + 1]
    float* sq = sk + GCH * (S + 1);        // [GCH][S + 1]
    float* skk = sq + GCH * (S + 1);       // [GCH][GTP]  k_t.k_i
    float* sqk = skk + GCH * GTP;          // [GCH][GTP]  q_t.k_i
    float* sa = gsm;                       // [VPK][GCH][GTP]  A, then T below the diagonal (over k and q:
                                           // read by then, so two blocks share an SM)
    const int qh = blockIdx.y, ch = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int64_t tb = t0 + (int64_t) ch * GCH;
    const int n = (int) min((int64_t) GCH, T - tb);
    {   // k and q as float4, every load issued before the first store
        constexpr int NP = GCH * S / 4 / 256;   // float4 a thread, each of k and q
        float4 kv[NP], qv[NP];
#pragma unroll
        for (int m = 0; m < NP; ++m) {
            const int p = tid + 256 * m, t = p / (S / 4), r = (p % (S / 4)) * 4;
            kv[m] = t < n ? *reinterpret_cast<const float4*>(h + (tb + t) * C + HK * S + qh * S + r) : make_float4(0.f, 0.f, 0.f, 0.f);
            qv[m] = t < n ? *reinterpret_cast<const float4*>(h + (tb + t) * C + qh * S + r) : make_float4(0.f, 0.f, 0.f, 0.f);
        }
#pragma unroll
        for (int m = 0; m < NP; ++m) {
            const int p = tid + 256 * m, t = p / (S / 4), r = (p % (S / 4)) * 4;
            float* dk = sk + t * (S + 1) + r;
            float* dq = sq + t * (S + 1) + r;
            dk[0] = kv[m].x; dk[1] = kv[m].y; dk[2] = kv[m].z; dk[3] = kv[m].w;
            dq[0] = qv[m].x; dq[1] = qv[m].y; dq[2] = qv[m].z; dq[3] = qv[m].w;
        }
    }
    __syncthreads();
    {   // lane i, rows warp + 8m: k_t.k_i and q_t.k_i on eight independent sums (rows 129 apart: 32 banks)
        constexpr int RW = GCH / 8;
        float kk[RW] = {}, qk[RW] = {};
        for (int r = 0; r < S; ++r) {
            const float ki = sk[lane * (S + 1) + r];
#pragma unroll
            for (int m = 0; m < RW; ++m) {
                kk[m] = fmaf(sk[(warp + 8 * m) * (S + 1) + r], ki, kk[m]);
                qk[m] = fmaf(sq[(warp + 8 * m) * (S + 1) + r], ki, qk[m]);
            }
        }
#pragma unroll
        for (int m = 0; m < RW; ++m) {
            const int t = warp + 8 * m;
            skk[t * GTP + lane] = lane <= t ? kk[m] : 0.0f;
            sqk[t * GTP + lane] = lane <= t ? qk[m] : 0.0f;
        }
    }
    __syncthreads();
    if (warp >= VPK) return;
    const int vh = qh + warp * HK, i = lane;   // warp j: value head j of the key head; lane i: token i (column i)
    float* a = sa + warp * GCH * GTP;
    const float b = i < n ? beta[(tb + i) * HV + vh] : 0.0f;
    float gam = i < n ? gate[(tb + i) * HV + vh] : 0.0f;   // past the end: gate 0, beta 0 (no effect on S)
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const float u = __shfl_up_sync(0xffffffffu, gam, o);
        if (lane >= o) gam += u;
    }
    for (int t = 0; t < GCH; ++t) {
        const float gt = __shfl_sync(0xffffffffu, gam, t), bt = __shfl_sync(0xffffffffu, b, t);
        a[t * GTP + i] = i < t ? bt * expf(gt - gam) * skk[t * GTP + i] : (i == t ? 1.0f : 0.0f);
    }
    __syncwarp();
    // T = (I + A)^-1 by forward substitution, row by row in place: T[t][i] = -sum_{k < t} A[t][k] T[k][i] over the
    // finished rows (diagonal 1, zeros above it: the terms k < i add an exact 0, so the same bits as summing from k = i)
    // - a trip the whole warp shares, so the loads go ahead of the sums
    for (int t = 1; t < GCH; ++t) {
        float v = 0.0f;
#pragma unroll 8
        for (int k = 0; k < t; ++k) v = fmaf(-a[t * GTP + k], a[k * GTP + i], v);
        __syncwarp();
        if (i < t) a[t * GTP + i] = v;
        __syncwarp();
    }
    float* T_ = tm + ((int64_t) ch * HV + vh) * GCH * GCH;
    float* P_ = pm + ((int64_t) ch * HV + vh) * GCH * GCH;
    for (int t = 0; t < GCH; ++t) {
        const float gt = __shfl_sync(0xffffffffu, gam, t);
        T_[t * GCH + i] = a[t * GTP + i];
        P_[t * GCH + i] = i <= t ? expf(gt - gam) * sqk[t * GTP + i] : 0.0f;
    }
    gm[((int64_t) ch * HV + vh) * GCH + i] = gam;
}

__device__ __forceinline__ void gdn_pf_l2(const float* p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }
__device__ __forceinline__ void gdn_cp_wait_all() {
#if STRATA_GDN_CP_ASYNC
    asm volatile("cp.async.wait_all;\n" ::);
#endif
}
// A block per (key head, 16 value columns): 128 threads for each of the key head's three value heads, which share the
// staged k and q rows.  128 blocks, one an SM, in one wave (each walks every chunk: a second wave would double the
// time).  A thread owns one token's row of the products (4 columns) and holds that token's T and P rows in registers,
// loaded at the chunk's start so they land during the first product; the next chunk's k (a second buffer), q (after
// this chunk's last read of q), v, gamma and beta load while this chunk computes.  Three barriers a chunk.
__global__ void __launch_bounds__(128 * VPK) gdn_chunk_scan_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                                   const float* __restrict__ beta, const float* __restrict__ tm,
                                                                   const float* __restrict__ pm, const float* __restrict__ gm,
                                                                   float* __restrict__ oc_out, int64_t t0, int nch, int64_t T) {
    extern __shared__ __align__(16) float gsm[];
    constexpr int NT = 128 * VPK, HEAD = S * GDV + 3 * GCH * GDV;   // threads; a head's shared floats
    float* kbuf = gsm;                     // [2][GCH][GKP]  k, double-buffered
    float* sq = kbuf + 2 * GCH * GKP;      // [GCH][GKP]
    const int qh = blockIdx.x, c0 = blockIdx.y * GDV, j = threadIdx.x / 128, tid = threadIdx.x % 128;
    const int vh = qh + j * HK;
    float* ss = sq + GCH * GKP + j * HEAD; // [S][GDV]  the state slice at the chunk's start
    float* sy = ss + S * GDV;              // [GCH][GDV]  Y
    float* sd = sy + GCH * GDV;            // [GCH][GDV]  D
    float* sdd = sd + GCH * GDV;           // [GCH][GDV]  exp(gamma_last - gamma_t) D
    const int rq = tid >> 2, vq = tid & 3;   // the update's tile: rows 4rq..4rq+3; both tiles: columns 4vq..4vq+3
    const int tt = tid >> 2, vv = 4 * vq;    // the products' tile: token tt
    const size_t rs = (size_t) HV * S;
    auto stage_kq = [&](int64_t c_t0, int n, float* dst, bool isq) {   // k or q rows of a chunk, cp.async
        for (int p = threadIdx.x; p < GCH * (S / 4); p += NT) {
            const int t = p / (S / 4), r = (p % (S / 4)) * 4;
            if (t < n) gdn_cp16(dst + t * GKP + r, h + (c_t0 + t) * C + (isq ? 0 : HK * S) + qh * S + r);
            else *reinterpret_cast<float4*>(dst + t * GKP + r) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    };
    // this tile's v, gamma, beta and the chunk's last gamma (registers)
    auto load_vgb = [&](int ch, int64_t c_t0, int n, float4& v4, float& gam, float& bet, float& glast) {
        v4 = tt < n ? __ldg(reinterpret_cast<const float4*>(h + (c_t0 + tt) * C + 2 * HK * S + vh * S + c0 + vv))
                    : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const float* g = gm + ((int64_t) ch * HV + vh) * GCH;
        gam = __ldg(g + tt);
        glast = __ldg(g + GCH - 1);
        bet = tt < n ? __ldg(beta + (c_t0 + tt) * HV + vh) : 0.0f;
    };
    float s[4][4];
#pragma unroll
    for (int x = 0; x < 4; ++x) {
#pragma unroll
        for (int y = 0; y < 4; ++y) s[x][y] = state[(size_t) (4 * rq + x) * rs + (size_t) vh * S + c0 + vv + y];
        *reinterpret_cast<float4*>(&ss[(4 * rq + x) * GDV + vv]) = make_float4(s[x][0], s[x][1], s[x][2], s[x][3]);
    }
    const float out_scale = rsqrtf((float) S);
    float4 v4;
    float gam, bet, glast;
    {
        const int n0 = (int) min((int64_t) GCH, T - t0);
        stage_kq(t0, n0, kbuf, false);
        stage_kq(t0, n0, sq, true);
        gdn_cp_commit();
        load_vgb(0, t0, n0, v4, gam, bet, glast);
        gdn_cp_wait_all();
        __syncthreads();
    }
    for (int ch = 0; ch < nch; ++ch) {
        const int64_t c_t0 = t0 + (int64_t) ch * GCH;
        const int n = (int) min((int64_t) GCH, T - c_t0);
        const bool more = ch + 1 < nch;
        const int64_t n_t0 = c_t0 + GCH;
        const int nn = more ? (int) min((int64_t) GCH, T - n_t0) : 0;
        const float* sk = kbuf + (ch & 1) * GCH * GKP;
        // this token's T and P rows into registers now: they land while the first product runs
        float tr[GCH], pr[GCH];
        {
            const float4* T4 = reinterpret_cast<const float4*>(tm + (((int64_t) ch * HV + vh) * GCH + tt) * GCH);
            const float4* P4 = reinterpret_cast<const float4*>(pm + (((int64_t) ch * HV + vh) * GCH + tt) * GCH);
#pragma unroll
            for (int i = 0; i < GCH / 4; ++i) {
                const float4 a = __ldg(T4 + i), b = __ldg(P4 + i);
                tr[4 * i] = a.x; tr[4 * i + 1] = a.y; tr[4 * i + 2] = a.z; tr[4 * i + 3] = a.w;
                pr[4 * i] = b.x; pr[4 * i + 1] = b.y; pr[4 * i + 2] = b.z; pr[4 * i + 3] = b.w;
            }
        }
        float4 nv4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float ngam = 0.0f, nbet = 0.0f, nglast = 0.0f;
        if (more) {   // the next chunk's v, gamma and beta; the one after into L2 (its k and q: after barrier A)
            load_vgb(ch + 1, n_t0, nn, nv4, ngam, nbet, nglast);
            const int64_t f_t0 = n_t0 + GCH;
            const int t = threadIdx.x >> 3, l = threadIdx.x & 7;
            if (t < GCH && f_t0 + t < T) gdn_pf_l2(h + (f_t0 + t) * C + (l < 4 ? HK * S : 0) + qh * S + 32 * (l & 3));
            if (tid < GCH) {
                gdn_pf_l2(tm + (((int64_t) (ch + 1) * HV + vh) * GCH + tid) * GCH);
                gdn_pf_l2(pm + (((int64_t) (ch + 1) * HV + vh) * GCH + tid) * GCH);
            }
        }
        // X = K S0 and Q S0 for token tt, then Y = beta (V - exp(gamma) X)
        float x[4] = {0.f, 0.f, 0.f, 0.f}, qs[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
        for (int r = 0; r < S; r += 4) {
            const float4 k4 = *reinterpret_cast<const float4*>(&sk[tt * GKP + r]);
            const float4 q4 = *reinterpret_cast<const float4*>(&sq[tt * GKP + r]);
            const float kk[4] = {k4.x, k4.y, k4.z, k4.w}, qq[4] = {q4.x, q4.y, q4.z, q4.w};
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const float4 s4 = *reinterpret_cast<const float4*>(&ss[(r + jj) * GDV + vv]);
                x[0] = fmaf(kk[jj], s4.x, x[0]); x[1] = fmaf(kk[jj], s4.y, x[1]);
                x[2] = fmaf(kk[jj], s4.z, x[2]); x[3] = fmaf(kk[jj], s4.w, x[3]);
                qs[0] = fmaf(qq[jj], s4.x, qs[0]); qs[1] = fmaf(qq[jj], s4.y, qs[1]);
                qs[2] = fmaf(qq[jj], s4.z, qs[2]); qs[3] = fmaf(qq[jj], s4.w, qs[3]);
            }
        }
        const float e = expf(gam);
        *reinterpret_cast<float4*>(&sy[tt * GDV + vv]) = make_float4(bet * (v4.x - e * x[0]), bet * (v4.y - e * x[1]),
                                                                     bet * (v4.z - e * x[2]), bet * (v4.w - e * x[3]));
        __syncthreads();   // (A) Y complete; every read of q done
        if (more) {        // the next chunk's k into the other buffer, its q into the single q buffer
            stage_kq(n_t0, nn, kbuf + ((ch + 1) & 1) * GCH * GKP, false);
            stage_kq(n_t0, nn, sq, true);
            gdn_cp_commit();
        }
        // D = T Y (T is zero above its diagonal), and D decayed to the chunk's end for the update
        float d[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < GCH; ++i) {
            const float4 y4 = *reinterpret_cast<const float4*>(&sy[i * GDV + vv]);
            d[0] = fmaf(tr[i], y4.x, d[0]); d[1] = fmaf(tr[i], y4.y, d[1]);
            d[2] = fmaf(tr[i], y4.z, d[2]); d[3] = fmaf(tr[i], y4.w, d[3]);
        }
        const float f = expf(glast - gam);
        *reinterpret_cast<float4*>(&sd[tt * GDV + vv]) = make_float4(d[0], d[1], d[2], d[3]);
        *reinterpret_cast<float4*>(&sdd[tt * GDV + vv]) = make_float4(f * d[0], f * d[1], f * d[2], f * d[3]);
        __syncthreads();   // (B) D complete
        // o = exp(gamma) Q S0 + P D (P is zero above its diagonal), the output
        float o[4] = {e * qs[0], e * qs[1], e * qs[2], e * qs[3]};
#pragma unroll
        for (int i = 0; i < GCH; ++i) {
            const float4 d4 = *reinterpret_cast<const float4*>(&sd[i * GDV + vv]);
            o[0] = fmaf(pr[i], d4.x, o[0]); o[1] = fmaf(pr[i], d4.y, o[1]);
            o[2] = fmaf(pr[i], d4.z, o[2]); o[3] = fmaf(pr[i], d4.w, o[3]);
        }
        if (tt < n)
            *reinterpret_cast<float4*>(oc_out + (c_t0 + tt) * HV * S + vh * S + c0 + vv) =
                make_float4(o[0] * out_scale, o[1] * out_scale, o[2] * out_scale, o[3] * out_scale);
        // S = exp(gamma_last) S0 + K^T (decayed D)
        const float gL = expf(glast);
#pragma unroll
        for (int xx = 0; xx < 4; ++xx)
#pragma unroll
            for (int y = 0; y < 4; ++y) s[xx][y] *= gL;
#pragma unroll 4
        for (int t = 0; t < GCH; ++t) {
            const float4 k4 = *reinterpret_cast<const float4*>(&sk[t * GKP + 4 * rq]);
            const float4 d4 = *reinterpret_cast<const float4*>(&sdd[t * GDV + vv]);
            const float kx[4] = {k4.x, k4.y, k4.z, k4.w}, dy[4] = {d4.x, d4.y, d4.z, d4.w};
#pragma unroll
            for (int xx = 0; xx < 4; ++xx)
#pragma unroll
                for (int y = 0; y < 4; ++y) s[xx][y] = fmaf(kx[xx], dy[y], s[xx][y]);
        }
#pragma unroll
        for (int xx = 0; xx < 4; ++xx)
            *reinterpret_cast<float4*>(&ss[(4 * rq + xx) * GDV + vv]) = make_float4(s[xx][0], s[xx][1], s[xx][2], s[xx][3]);
        v4 = nv4; gam = ngam; bet = nbet; glast = nglast;
        gdn_cp_wait_all();
        __syncthreads();   // (C) the new ss, the next k and q landed; every read of sd, sdd and this k buffer done
    }
#pragma unroll
    for (int xx = 0; xx < 4; ++xx)
#pragma unroll
        for (int y = 0; y < 4; ++y) state[(size_t) (4 * rq + xx) * rs + (size_t) vh * S + c0 + vv + y] = s[xx][y];
}

// The scratch, one a device: allocated on first use at its largest (GSB / GCH chunks of T, P and gamma: 25.6 MB) and
// kept (a device runs its prompt on one stream).  A stream-ordered allocation and free per call, even from a pool that
// keeps its memory, cost host time the GPU waited for (a 2K prompt's recurrence took 78 ms against 34 ms for
// gdn_rec_kh_kernel).
float* gdn_chunk_scratch() {
    static float* bufs[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return nullptr; }
    if (bufs[dev] == nullptr) {
        const size_t n = (size_t) (GSB / GCH) * HV * (2 * GCH * GCH + GCH);
        if (cudaMalloc((void**) &bufs[dev], n * sizeof(float)) != cudaSuccess) {
            cudaGetLastError();
            bufs[dev] = nullptr;
        }
    }
    return bufs[dev];
}

// The whole recurrence of T tokens in super-blocks of GSB: a prep and a scan launch each, the scratch (T, P and gamma
// of a super-block's chunks: up to GSB / GCH x 48 heads x (2 x 32 x 32 + 32) floats, 25.6 MB) from gdn_chunk_scratch.
// y: the [T][HV][S] output.  An error return means nothing was launched (the caller takes another kernel): the scratch
// could not be had, or the card has fewer than 128 SMs (a second wave of the scan would double it) or too little
// shared memory for it (Turing).
cudaError_t gdn_rec_chunked(float* state, const float* h, const float* gate, const float* beta, float* y, int64_t T,
                            cudaStream_t s) {
    if (T <= 0) return cudaSuccess;
    {   // sm_80+ (cp.async), the scan's 128 blocks one an SM in one wave, and its 93.7 KB of shared memory (else:
        // another kernel)
        int dev = 0, major = 0, sms = 0, smem = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess) {
            cudaGetLastError();
            return cudaErrorNotSupported;
        }
        // as in kernels.cu: STRATA_GDN_CHUNKED=2 skips the SM-count test (to measure a card under 128 SMs)
        static const bool any_sms = [] { const char* v = std::getenv("STRATA_GDN_CHUNKED"); return v != nullptr && std::atoi(v) >= 2; }();
        if (major < 8 || (!any_sms && sms < HK * (S / GDV)) || (size_t) smem < kChunkScanSmem)
            return cudaErrorNotSupported;
    }
    float* scratch = gdn_chunk_scratch();
    if (scratch == nullptr) return cudaErrorMemoryAllocation;
    const int nb = (int) ((std::min<int64_t>(GSB, T) + GCH - 1) / GCH);
    const size_t tsz = (size_t) nb * HV * GCH * GCH;
    float *tm = scratch, *pm = tm + tsz, *gm = pm + tsz;
    {   // once a device
        static bool attrs[64] = {};
        int dev = 0;
        cudaGetDevice(&dev);
        if (dev >= 0 && dev < 64 && !attrs[dev]) {
            cudaFuncSetAttribute(gdn_chunk_prep_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kChunkPrepSmem);
            cudaFuncSetAttribute(gdn_chunk_scan_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kChunkScanSmem);
            attrs[dev] = true;
        }
    }
    for (int64_t t0 = 0; t0 < T; t0 += GSB) {
        const int nch = (int) ((std::min<int64_t>(GSB, T - t0) + GCH - 1) / GCH);
        gdn_chunk_prep_kernel<<<dim3((unsigned) nch, HK), 256, kChunkPrepSmem, s>>>(h, gate, beta, tm, pm, gm, t0, T);
        gdn_chunk_scan_kernel<<<dim3(HK, S / GDV), 128 * VPK, kChunkScanSmem, s>>>(state, h, beta, tm, pm, gm, y, t0, nch, T);
    }
    return cudaSuccess;   // launch errors surface at the caller's check; an error above means nothing ran
}
// src/prefill/kernels.cu's tensor-core scan (STRATA_GDN_TC=1): the same text, included there too
#include "gdn_tc.cuh"
// src/prefill/kernels.cu's gdn_rec_tc, copied as it is there (with the raw device attributes, as gdn_rec_chunked above)
cudaError_t gdn_rec_tc(float* state, const float* h, const float* gate, const float* beta, float* y, int64_t T,
                       cudaStream_t s) {
    if (T <= 0) return cudaSuccess;
    {
        int dev = 0, major = 0, sms = 0, smem = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess) {
            cudaGetLastError();
            return cudaErrorNotSupported;
        }
        if (major < 8 || sms < HV || (size_t) smem < kTcScanSmem<false>) return cudaErrorNotSupported;
    }
    const GtcScratch sc = gdn_tc_scratch();
    if (sc.ti == nullptr) return cudaErrorMemoryAllocation;
    {   // once a device
        static bool attrs[64] = {};
        int dev = 0;
        cudaGetDevice(&dev);
        if (dev >= 0 && dev < 64 && !attrs[dev]) {
            cudaFuncSetAttribute(gdn_tc_scan_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kTcScanSmem<false>);
            attrs[dev] = true;
        }
    }
    for (int64_t t0 = 0; t0 < T; t0 += GSB) {
        const int nch = (int) ((std::min<int64_t>(GSB, T - t0) + GCH - 1) / GCH);
        gdn_tc_prep_kernel<<<dim3((unsigned) nch, HK), 256, kTcPrepSmem, s>>>(h, gate, beta, sc.ti, sc.te, sc.sc, sc.kq, sc.pi, t0, T);
        gdn_tc_scan_kernel<false><<<HV, GTC_CW * 32, kTcScanSmem<false>, s>>>(state, h, sc.ti, sc.te, sc.sc, sc.kq, sc.pi, y, t0, nch, T);
    }
    return cudaSuccess;
}
// the output norm over a head's 128 columns, into the FP16 copy the out projection reads (the FP32 output before the
// norm stays in its scratch buffer)
__global__ void __launch_bounds__(S) gdn_out_norm_kernel(const float* __restrict__ z, const float* __restrict__ gamma,
                                                         float eps, const float* __restrict__ y,
                                                         uint16_t* __restrict__ y16) {
    __shared__ float wsum[4];
    const int64_t t = blockIdx.x;
    const int head = blockIdx.y, col = threadIdx.x;
    const size_t at = (size_t) t * HV * S + (size_t) head * S + col;
    const float oc = y[at];
    float sp = warp_sum(oc * oc);
    if ((col & 31) == 0) wsum[col >> 5] = sp;
    __syncthreads();
    const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
    const float v = oc * rsqrtf(ss / (float) S + eps) * gamma[col] * sigm(z[t * HV * S + head * S + col]);
    y16[at] = hf(v);
}

// ------------------------------------------------------------------ the output norm before (0.1.31)
__global__ void __launch_bounds__(S) gdn_out_norm_old_kernel(const float* __restrict__ z, const float* __restrict__ gamma,
                                                             float eps, float* __restrict__ y,
                                                             uint16_t* __restrict__ y16) {
    __shared__ float wsum[4];
    const int64_t t = blockIdx.x;
    const int head = blockIdx.y, col = threadIdx.x;
    const size_t at = (size_t) t * HV * S + (size_t) head * S + col;
    const float oc = y[at];
    float sp = warp_sum(oc * oc);
    if ((col & 31) == 0) wsum[col >> 5] = sp;
    __syncthreads();
    const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
    const float v = oc * rsqrtf(ss / (float) S + eps) * gamma[col] * sigm(z[t * HV * S + head * S + col]);
    y[at] = v;
    y16[at] = hf(v);
}

// ------------------------------------------------------------------ synthetic inputs
__device__ __forceinline__ uint32_t mix(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
    return (uint32_t) x;
}
__device__ __forceinline__ float uni(uint64_t key) { return ((mix(key) >> 8) + 0.5f) * (1.0f / 16777216.0f); }
__device__ __forceinline__ float gauss(uint64_t key) {
    const float u1 = uni(key * 2 + 1), u2 = uni(key * 2 + 2);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}
__global__ void fill_gauss(float* x, int64_t n, float scale, uint64_t seed, float offset = 0.0f) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        x[i] = offset + scale * gauss(seed * 0x9E3779B97F4A7C15ULL + (uint64_t) i);
}
// the q and k rows L2-normalized per key head, as gdn_l2_kernel leaves them for the recurrence
__global__ void l2_rows(float* h, float eps) {
    float* x = h + (int64_t) blockIdx.y * C + (int64_t) blockIdx.x * S;
    const float v = x[threadIdx.x];
    __shared__ float part[4];
    const float sq = warp_sum(v * v);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
    __syncthreads();
    x[threadIdx.x] = v * rsqrtf(part[0] + part[1] + part[2] + part[3] + eps);
}
// gate = log of the decay (mostly near 1, some strong; slow: -0.001 u, so the state grows for long), beta in (0, 1)
__global__ void fill_gates(float* gate, float* beta, int64_t n, uint64_t seed, bool slow = false) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const float u = uni(seed * 0x9E3779B97F4A7C15ULL + 7 * (uint64_t) i);
        gate[i] = slow ? -0.001f * u : -0.02f - 0.8f * u * u * u;
        beta[i] = 1.0f / (1.0f + expf(-gauss(seed * 0xD1B54A32D192ED03ULL + (uint64_t) i)));
    }
}
// coherent keys: every token's k row = a key head's k_0 + 0.05 x the row that was there (before l2_rows): the
// chunk's T = (I + A)^-1 then carries large entries, the reason the prep scales its T images by a power of two of their own
__global__ void coherent_k(float* h, uint64_t seed) {
    float* x = h + (int64_t) blockIdx.y * C + HK * S + (int64_t) blockIdx.x * S;
    x[threadIdx.x] = gauss(seed * 0x9E3779B97F4A7C15ULL + (uint64_t) (blockIdx.x * S + threadIdx.x)) + 0.05f * x[threadIdx.x];
}
__global__ void scale_v(float* h, int64_t T, float scale) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < T * HV * S; i += (int64_t) gridDim.x * blockDim.x)
        h[(i / (HV * S)) * C + 2 * HK * S + i % (HV * S)] *= scale;
}
__global__ void count_diff(const uint32_t* a, const uint32_t* b, int64_t n, unsigned long long* out) {
    unsigned long long d = 0;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        d += a[i] != b[i];
    if (d) atomicAdd(out, d);
}

// ------------------------------------------------------------------ holding SMs (fewer_sms)
__device__ __forceinline__ uint64_t gtime() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ int sm_id() {
    unsigned id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
    return (int) id;
}
__device__ __forceinline__ void nap(unsigned ns) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    __nanosleep(ns);
#else
    (void) ns;
#endif
}
// limits that a run that works never reaches (a launch here takes milliseconds); they keep a run that does not work
// from waiting forever, and the host drops its time
constexpr uint64_t kHoldLimitNs = 1000000000ull, kWaitLimitNs = 500000000ull;
// One block per SM: it asks for all the shared memory a block may have (launched with that much dynamic shared
// memory), so no block of another kernel fits beside it.  It notes its SM and sleeps until release says gen.
__global__ void hold_sm(unsigned long long* sitting, const volatile unsigned* release, unsigned gen, int* where,
                        unsigned* late) {
    where[blockIdx.x] = sm_id();
    __threadfence();
    atomicAdd(sitting, 1ull);
    const uint64_t t0 = gtime();
    while (*release != gen) {
        if (gtime() - t0 > kHoldLimitNs) { atomicAdd(late, 1u); return; }
        nap(20000);
    }
}
// on the timed launch's stream, before it: returns once all holding blocks so far sit on their SMs
__global__ void wait_sitting(const volatile unsigned long long* sitting, unsigned long long all, unsigned* late) {
    const uint64_t t0 = gtime();
    while (*sitting < all) {
        if (gtime() - t0 > kWaitLimitNs) { atomicAdd(late, 1u); return; }
        nap(1000);
    }
}
__global__ void release_sms(volatile unsigned* release, unsigned gen) { *release = gen; }
// Where the blocks of a grid land: launched like the recurrence it stands for (the same grid and block size, and as
// much dynamic shared memory as lets as many blocks share an SM), every block notes its SM and stays until all of them
// have landed, so they all sit at once as the recurrence's blocks do for the whole chunk.
__global__ void land(int* where, unsigned* landed, unsigned* apart) {
    extern __shared__ unsigned char land_smem[];   // only its size matters
    if (threadIdx.x != 0 || threadIdx.y != 0) return;
    where[blockIdx.x] = sm_id();
    atomicAdd(landed, 1u);
    const uint64_t t0 = gtime();
    while (*(volatile unsigned*) landed < gridDim.x) {
        if (gtime() - t0 > kWaitLimitNs) { atomicAdd(apart, 1u); return; }   // they did not all fit at once
        nap(1000);
    }
    (void) land_smem;
}

template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc((void**) &p, n * sizeof(T) + 16), "cudaMalloc");
    return p;
}
unsigned long long diff_words(const void* a, const void* b, size_t bytes) {
    static unsigned long long* d = dalloc<unsigned long long>(1);
    ck(cudaMemset(d, 0, sizeof(*d)), "memset");
    count_diff<<<1024, 256>>>((const uint32_t*) a, (const uint32_t*) b, (int64_t) (bytes / 4), d);
    unsigned long long h = 0;
    ck(cudaMemcpy(&h, d, sizeof(h), cudaMemcpyDeviceToHost), "diff");
    return h;
}

// The model's shapes with gaussian values; the defaults are the bit checks' inputs (the state as after an earlier
// chunk, scale 0.05).  The others stress the tensor-core scan's f16 range handling: the state 100x smaller, 50x larger
// or zero (a fresh conversation), v 50x larger, slow gates, coherent keys.
struct Inputs {
    int64_t T = 0;
    float *h = nullptr, *gate = nullptr, *beta = nullptr, *z = nullptr, *gamma = nullptr, *state0 = nullptr;
    explicit Inputs(int64_t t, uint64_t seed, float state_scale = 0.05f, float v_scale = 1.0f, bool slow_gates = false,
                    bool coherent_keys = false)
        : T(t) {
        h = dalloc<float>((size_t) T * C);
        gate = dalloc<float>((size_t) T * HV);
        beta = dalloc<float>((size_t) T * HV);
        z = dalloc<float>((size_t) T * HV * S);
        gamma = dalloc<float>(S);
        state0 = dalloc<float>((size_t) S * HV * S);
        fill_gauss<<<4096, 256>>>(h, T * C, 1.0f, seed);
        if (coherent_keys) coherent_k<<<dim3(HK, (unsigned) T), S>>>(h, seed + 5);
        l2_rows<<<dim3(2 * HK, (unsigned) T), S>>>(h, 1e-6f);
        if (v_scale != 1.0f) scale_v<<<4096, 256>>>(h, T, v_scale);
        fill_gates<<<1024, 256>>>(gate, beta, T * HV, seed + 1, slow_gates);
        fill_gauss<<<4096, 256>>>(z, T * HV * S, 1.0f, seed + 2);
        fill_gauss<<<1, 128>>>(gamma, S, 0.1f, seed + 3, 1.0f);
        fill_gauss<<<1024, 256>>>(state0, (int64_t) S * HV * S, state_scale, seed + 4);
        ck(cudaDeviceSynchronize(), "inputs");
    }
    Inputs(const Inputs&) = delete;
    Inputs& operator=(const Inputs&) = delete;
    ~Inputs() { cudaFree(h); cudaFree(gate); cudaFree(beta); cudaFree(z); cudaFree(gamma); cudaFree(state0); }
};

enum Rec { kBefore, kKeyHead, kRecCount };
void run_rec(Rec v, float* state, const Inputs& in, float* oc, cudaStream_t s = 0) {
    if (v == kBefore)
        gdn_rec_cols_pipe_kernel<<<HV * NCB, dim3(CB, RG), 0, s>>>(state, in.h, in.gate, in.beta, oc, in.T);
    else
        gdn_rec_kh_kernel<<<HK * NCB, dim3(CB, RG), 0, s>>>(state, in.h, in.gate, in.beta, oc, in.T);
}

int check_bits(int64_t T, uint64_t seed) {
    Inputs in(T, seed);
    const size_t st = (size_t) S * HV * S, oc_n = (size_t) T * HV * S;
    float* state[kRecCount];
    float* oc[kRecCount];
    for (int v = 0; v < kRecCount; ++v) {
        state[v] = dalloc<float>(st);
        oc[v] = dalloc<float>(oc_n);
        ck(cudaMemcpy(state[v], in.state0, st * 4, cudaMemcpyDeviceToDevice), "state copy");
        run_rec((Rec) v, state[v], in, oc[v]);
    }
    ck(cudaDeviceSynchronize(), "recurrence");
    int fails = 0;
    const unsigned long long d_oc = diff_words(oc[kBefore], oc[kKeyHead], oc_n * 4),
                             d_st = diff_words(state[kBefore], state[kKeyHead], st * 4);
    if (d_oc || d_st) {
        std::printf("FAIL T=%lld recurrence: %llu of %zu outputs and %llu of %zu state values differ\n", (long long) T,
                    d_oc, oc_n, d_st, st);
        ++fails;
    }
    // the output norm: before (in place, with the FP32 store) against now (the FP16 store only), on the same output
    uint16_t* y16a = dalloc<uint16_t>(oc_n);
    uint16_t* y16b = dalloc<uint16_t>(oc_n);
    float* yb = dalloc<float>(oc_n);
    ck(cudaMemcpy(yb, oc[kBefore], oc_n * 4, cudaMemcpyDeviceToDevice), "y copy");
    gdn_out_norm_kernel<<<dim3((unsigned) T, HV), S>>>(in.z, in.gamma, 1e-6f, yb, y16b);
    gdn_out_norm_old_kernel<<<dim3((unsigned) T, HV), S>>>(in.z, in.gamma, 1e-6f, oc[kBefore], y16a);
    ck(cudaDeviceSynchronize(), "norm");
    const unsigned long long d16 = diff_words(y16a, y16b, oc_n * 2);
    if (d16) {
        std::printf("FAIL T=%lld output norm: %llu of %zu FP16 pairs differ\n", (long long) T, d16, oc_n / 2);
        ++fails;
    }
    std::printf("T=%-6lld %s\n", (long long) T, fails ? "DIFFERENT" : "the same bits (output, state, FP16 output)");
    for (int v = 0; v < kRecCount; ++v) { cudaFree(state[v]); cudaFree(oc[v]); }
    cudaFree(y16a); cudaFree(y16b); cudaFree(yb);
    return fails;
}

// The chunked recurrence against gdn_rec_kh_kernel (other bits: the largest |difference| over the largest |value|, the
// output and the state after the chunk); a failure above 1e-4 (FP32 sums in another order stay near 1e-6).
double max_rel(const float* a, const float* b, size_t n) {
    std::vector<float> x(n), y(n);
    ck(cudaMemcpy(x.data(), a, n * 4, cudaMemcpyDeviceToHost), "copy");
    ck(cudaMemcpy(y.data(), b, n * 4, cudaMemcpyDeviceToHost), "copy");
    double md = 0.0, mv = 0.0;
    for (size_t i = 0; i < n; ++i) {
        md = std::max(md, (double) std::fabs(x[i] - y[i]));
        mv = std::max(mv, (double) std::fabs(x[i]));
    }
    return mv > 0.0 ? md / mv : md;
}
// The recurrence in FP64 (a thread per value head and column, the state in local memory): the reference both FP32
// recurrences are measured against
__global__ void gdn_rec_f64_kernel(double* __restrict__ state, const float* __restrict__ h, const float* __restrict__ gate,
                                   const float* __restrict__ beta, double* __restrict__ out, int64_t T) {
    const int head = blockIdx.x, col = threadIdx.x, qh = head % HK;
    double s[S];
    for (int r = 0; r < S; ++r) s[r] = state[((size_t) r * HV + head) * S + col];
    for (int64_t t = 0; t < T; ++t) {
        const float* ht = h + t * C;
        const double g = exp((double) gate[t * HV + head]), b = beta[t * HV + head];
        double kv = 0.0;
        for (int r = 0; r < S; ++r) kv += s[r] * ht[HK * S + qh * S + r];
        const double delta = (ht[2 * HK * S + head * S + col] - g * kv) * b;
        double o = 0.0;
        for (int r = 0; r < S; ++r) {
            s[r] = g * s[r] + ht[HK * S + qh * S + r] * delta;
            o += s[r] * ht[qh * S + r];
        }
        out[(t * HV + head) * S + col] = o / sqrt((double) S);
    }
    for (int r = 0; r < S; ++r) state[((size_t) r * HV + head) * S + col] = s[r];
}
__global__ void to_f64(const float* x, double* y, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) y[i] = x[i];
}
double max_rel_f64(const float* a, const double* ref, size_t n) {   // max |a - ref| / max |ref|
    std::vector<float> x(n);
    std::vector<double> y(n);
    ck(cudaMemcpy(x.data(), a, n * 4, cudaMemcpyDeviceToHost), "copy");
    ck(cudaMemcpy(y.data(), ref, n * 8, cudaMemcpyDeviceToHost), "copy");
    double md = 0.0, mv = 0.0;
    for (size_t i = 0; i < n; ++i) {
        md = std::max(md, std::fabs(x[i] - y[i]));
        mv = std::max(mv, std::fabs(y[i]));
    }
    return mv > 0.0 ? md / mv : md;
}
// a recurrence of gdn_rec_chunked's shape (the FP32 chunked pair, the tensor-core scan) on the inputs given
using RecLaunch = cudaError_t (*)(float*, const float*, const float*, const float*, float*, int64_t, cudaStream_t);
int check_variant(const Inputs& in, const char* label, RecLaunch launch, const char* needs) {
    const int64_t T = in.T;
    const size_t st = (size_t) S * HV * S, oc_n = (size_t) T * HV * S;
    float* s0 = dalloc<float>(st);
    float* s1 = dalloc<float>(st);
    float* o0 = dalloc<float>(oc_n);
    float* o1 = dalloc<float>(oc_n);
    ck(cudaMemcpy(s0, in.state0, st * 4, cudaMemcpyDeviceToDevice), "state copy");
    ck(cudaMemcpy(s1, in.state0, st * 4, cudaMemcpyDeviceToDevice), "state copy");
    gdn_rec_kh_kernel<<<HK * NCB, dim3(CB, RG)>>>(s0, in.h, in.gate, in.beta, o0, in.T);
    const cudaError_t e = launch(s1, in.h, in.gate, in.beta, o1, in.T, 0);
    if (e == cudaErrorNotSupported) {
        std::printf("     T=%-6lld %s: not on this card (%s)\n", (long long) T, label, needs);
        cudaFree(s0); cudaFree(s1); cudaFree(o0); cudaFree(o1);
        return 0;
    }
    ck(e, label);
    ck(cudaDeviceSynchronize(), label);
    const double eo = max_rel(o0, o1, oc_n), es = max_rel(s0, s1, st);
    const bool bad = !(eo < 1e-4) || !(es < 1e-4);
    std::printf("%s T=%-6lld %s: output %.2e, state %.2e (max |diff| / max |value|)\n", bad ? "FAIL" : "    ",
                (long long) T, label, eo, es);
    if (T <= 8192) {   // both against FP64 (its output: 8 bytes a value, so not at 32K)
        double* sd = dalloc<double>(st);
        double* od = dalloc<double>(oc_n);
        to_f64<<<1024, 256>>>(in.state0, sd, (int64_t) st);
        gdn_rec_f64_kernel<<<HV, S>>>(sd, in.h, in.gate, in.beta, od, in.T);
        ck(cudaDeviceSynchronize(), "FP64 run");
        std::printf("            against FP64: gdn_rec_kh_kernel output %.2e, state %.2e; this one output %.2e, state %.2e\n",
                    max_rel_f64(o0, od, oc_n), max_rel_f64(s0, sd, st), max_rel_f64(o1, od, oc_n), max_rel_f64(s1, sd, st));
        cudaFree(sd); cudaFree(od);
    }
    cudaFree(s0); cudaFree(s1); cudaFree(o0); cudaFree(o1);
    return bad ? 1 : 0;
}
int check_chunked(int64_t T, uint64_t seed) {
    Inputs in(T, seed);
    return check_variant(in, "chunked (STRATA_GDN_CHUNKED)", gdn_rec_chunked, "sm_80+, 128 SMs, 93.7 KB a block");
}
constexpr const char* kTcNeeds = "sm_80+, 48 SMs, 122 KB a block";
int check_tc(int64_t T, uint64_t seed) {
    Inputs in(T, seed);
    return check_variant(in, "tc (STRATA_GDN_TC=1)", gdn_rec_tc, kTcNeeds);
}
// the tensor-core scan on the stress inputs (Inputs): its f16 range handling has a case for each
int check_tc_stress(int64_t T, uint64_t seed) {
    struct Case { const char* label; float state_scale, v_scale; bool slow_gates, coherent_keys; };
    const Case cases[] = {{"tc (STRATA_GDN_TC=1), state0 x 0.01", 5e-4f, 1.0f, false, false},
                          {"tc (STRATA_GDN_TC=1), state0 x 50", 2.5f, 1.0f, false, false},
                          {"tc (STRATA_GDN_TC=1), state0 = 0", 0.0f, 1.0f, false, false},
                          {"tc (STRATA_GDN_TC=1), v x 50", 0.05f, 50.0f, false, false},
                          {"tc (STRATA_GDN_TC=1), slow gates", 0.05f, 1.0f, true, false},
                          {"tc (STRATA_GDN_TC=1), coherent keys", 0.05f, 1.0f, false, true}};
    int fails = 0;
    for (const Case& c : cases) {
        Inputs in(T, seed++, c.state_scale, c.v_scale, c.slow_gates, c.coherent_keys);
        fails += check_variant(in, c.label, gdn_rec_tc, kTcNeeds);
    }
    return fails;
}

// ------------------------------------------------------------------ the tensor-core scan's fragment self-test (--selftest-frag)
// One warp each, random inputs, the reference in double on the host: (b) a 16x128 . 128x32 product through
// gtc_prod_state (the f16 split with the state's 2^es and k's 2^8, ldmatrix), (c) a 16x32 . 32x128 update through
// gtc_update_state (ldmatrix.trans, the 2^ed scale, the accumulator's pre / post scale), (d) a 16x32 . 32x32 product
// through gtc_prod_dt on a full random T of wide range (the prep's 2^et on the B images), (e) the triangle-skipped
// tile sets of gtc_prod_dt against the full ones on triangular T / P: the same bits, since the skipped tiles add
// exact zeros.  Most bugs of a tensor-core kernel are fragment-layout bugs: these catch them before the kernel does.
// ((a), the C -> A tf32 shuffle, went with the tf32 form of D^T.)
__device__ void st_stage_k(uint16_t* kh, uint16_t* kl, const float* k) {   // k: 32 x 128 f32 -> the hi / lo images x 2^8
    for (int p = threadIdx.x; p < GCH * S / 4; p += 32) {
        const int t = p / (S / 4), r = (p % (S / 4)) * 4;
        gtc_split_store4(kh + t * GTC_KP + r, kl + t * GTC_KP + r, *reinterpret_cast<const float4*>(k + t * S + r), 256.0f);
    }
    __syncwarp();
}
// a 16-row matrix (row stride ld) as 8-column tiles in the C layout: tile j, lane (gid, tig) holds (gid, 8j + 2tig),
// (gid, 8j + 2tig + 1), (gid + 8, 8j + 2tig), (gid + 8, 8j + 2tig + 1)
template <int NT> __device__ void st_load_c(float (&c)[NT][4], const float* m, int ld) {
    const int lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        c[j][0] = m[gid * ld + 8 * j + 2 * tig];
        c[j][1] = m[gid * ld + 8 * j + 2 * tig + 1];
        c[j][2] = m[(gid + 8) * ld + 8 * j + 2 * tig];
        c[j][3] = m[(gid + 8) * ld + 8 * j + 2 * tig + 1];
    }
}
template <int NT> __device__ void st_store_c(const float (&c)[NT][4], float* m, int ld, float scale) {
    const int lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        m[gid * ld + 8 * j + 2 * tig] = c[j][0] * scale;
        m[gid * ld + 8 * j + 2 * tig + 1] = c[j][1] * scale;
        m[(gid + 8) * ld + 8 * j + 2 * tig] = c[j][2] * scale;
        m[(gid + 8) * ld + 8 * j + 2 * tig + 1] = c[j][3] * scale;
    }
}
template <int NT> __device__ float st_max_abs(const float (&c)[NT][4]) {
    float m = 0.0f;
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) m = fmaxf(m, fabsf(c[j][q]));
    return gtc_warp_max(m);
}
__global__ void st_state_kernel(const float* a, const float* k, float* out) {   // out (16 x 32) = a (16 x 128) k^T
    __shared__ __align__(16) uint16_t kh[GCH * GTC_KP], kl[GCH * GTC_KP];
    st_stage_k(kh, kl, k);
    float s[16][4];
    st_load_c<16>(s, a, S);
    const int es = gtc_scale_exp(st_max_abs<16>(s));
    float xt[4][4] = {}, qs[4][4] = {};
    gtc_prod_state(s, gtc_pow2(es), kh, kl, kh, kl, xt, qs);
    st_store_c<4>(xt, out, GCH, gtc_pow2(-(8 + es)));
}
__global__ void st_update_kernel(const float* s0, const float* dd, const float* k, float eL, float* out) {   // out (16 x 128) = eL s0 + dd (16 x 32) k
    __shared__ __align__(16) uint16_t kh[GCH * GTC_KP], kl[GCH * GTC_KP];
    st_stage_k(kh, kl, k);
    float s[16][4], dt[4][4], ddt[4][4];
    st_load_c<16>(s, s0, S);
    st_load_c<4>(dt, dd, GCH);
    const int ed = gtc_scale_exp(st_max_abs<4>(dt));
    const float sD = gtc_pow2(ed), sPre = eL * gtc_pow2(ed + 8);
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) ddt[j][q] = dt[j][q] * sD;
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) s[j][q] *= sPre;
    gtc_update_state(s, ddt, kh, kl);
    st_store_c<16>(s, out, S, gtc_pow2(-(ed + 8)));
}
// out (16 x 32) = d p^T: p's images x 2^eb (the scan's P carry 2^8, its T the prep's 2^et), d x 2^ed from its maximum
template <bool SKIP> __global__ void st_dt_kernel(const float* d, const float* p, int eb, float* out) {
    __shared__ __align__(16) uint16_t ph[GCH * GTC_PP], pl[GCH * GTC_PP];
    for (int q = threadIdx.x; q < GCH * GCH / 4; q += 32) {
        const int t = q / (GCH / 4), i = (q % (GCH / 4)) * 4;
        gtc_split_store4(ph + t * GTC_PP + i, pl + t * GTC_PP + i, *reinterpret_cast<const float4*>(p + t * GCH + i), gtc_pow2(eb));
    }
    __syncwarp();
    float dt[4][4], ot[4][4] = {};
    st_load_c<4>(dt, d, GCH);
    const int ed = gtc_scale_exp(st_max_abs<4>(dt));
    gtc_prod_dt<SKIP>(dt, gtc_pow2(ed), ph, pl, ot);
    st_store_c<4>(ot, out, GCH, gtc_pow2(-(ed + eb)));
}
int st_scale_exp(const std::vector<float>& m) {   // the prep's gtc_scale_exp on the host: max |m| into [2^14, 2^15)
    float mx = 0.0f;
    for (float x : m) mx = std::max(mx, std::fabs(x));
    if (mx == 0.0f) return 0;
    int e = 0;
    std::frexp(mx, &e);   // mx = f 2^e, f in [0.5, 1): floor(log2 mx) = e - 1
    e = 14 - (e - 1);
    return e < -40 ? -40 : (e > 40 ? 40 : e);
}
struct StRng {   // xorshift64, host side
    uint64_t s;
    explicit StRng(uint64_t seed) : s(seed) {}
    float uni() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return ((float) (s >> 40) + 0.5f) / 16777216.0f; }
    float gauss() { return std::sqrt(-2.0f * std::log(uni())) * std::cos(6.2831853f * uni()); }
    float logmag(float lo, float hi) { return (uni() < 0.5f ? -1.0f : 1.0f) * std::pow(10.0f, lo + (hi - lo) * uni()); }
};
float* st_upload(const std::vector<float>& v) {
    float* d = dalloc<float>(v.size());
    ck(cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice), "upload");
    return d;
}
std::vector<float> st_download(const float* d, size_t n) {
    std::vector<float> v(n);
    ck(cudaDeviceSynchronize(), "self-test kernel");
    ck(cudaMemcpy(v.data(), d, n * 4, cudaMemcpyDeviceToHost), "download");
    return v;
}
double st_max_rel(const std::vector<float>& got, const std::vector<double>& ref) {
    double md = 0.0, mv = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        md = std::max(md, std::fabs((double) got[i] - ref[i]));
        mv = std::max(mv, std::fabs(ref[i]));
    }
    return mv > 0.0 ? md / mv : md;
}
int selftest_frag() {
    std::printf("--selftest-frag: the tensor-core scan's fragment helpers, one warp each\n");
    int fails = 0;
    StRng rng(0x5e1f7e57ULL);
    auto unit_rows = [&](std::vector<float>& k, int rows, int cols) {   // gaussian rows, L2-normalized
        for (int t = 0; t < rows; ++t) {
            double ss = 0.0;
            for (int r = 0; r < cols; ++r) { k[t * cols + r] = rng.gauss(); ss += (double) k[t * cols + r] * k[t * cols + r]; }
            for (int r = 0; r < cols; ++r) k[t * cols + r] = (float) (k[t * cols + r] / std::sqrt(ss));
        }
    };
    {   // (b) X^T = S^T K^T: f16x3 with the state's 2^es and k's 2^8, B by ldmatrix
        std::vector<float> a(16 * S), k(GCH * S);
        for (float& x : a) x = rng.logmag(-3.0f, 2.0f);
        unit_rows(k, GCH, S);
        std::vector<double> ref(16 * GCH);
        for (int c = 0; c < 16; ++c)
            for (int i = 0; i < GCH; ++i) {
                double v = 0.0;
                for (int r = 0; r < S; ++r) v += (double) a[c * S + r] * k[i * S + r];
                ref[c * GCH + i] = v;
            }
        float* da = st_upload(a);
        float* dk = st_upload(k);
        float* dout = dalloc<float>(16 * GCH);
        st_state_kernel<<<1, 32>>>(da, dk, dout);
        const double rel = st_max_rel(st_download(dout, 16 * GCH), ref);
        const bool bad = !(rel <= 2e-6);
        std::printf("  (b) X^T = S^T K^T (16x128 . 128x32, f16x3, 2^es 2^8, ldmatrix): max |diff| / max |ref| %.2e  %s\n", rel, bad ? "FAIL" : "ok");
        fails += bad;
        cudaFree(da); cudaFree(dk); cudaFree(dout);
    }
    {   // (c) S^T = eL S^T + Dd^T K: ldmatrix.trans, 2^ed, the accumulator scaled before and after
        std::vector<float> s0(16 * S), dd(16 * GCH), k(GCH * S);
        for (float& x : s0) x = rng.logmag(-3.0f, 2.0f);
        for (float& x : dd) x = rng.logmag(-3.0f, 2.0f);
        unit_rows(k, GCH, S);
        const float eL = 0.25f + 0.75f * rng.uni();
        std::vector<double> ref(16 * S);
        for (int c = 0; c < 16; ++c)
            for (int r = 0; r < S; ++r) {
                double v = (double) eL * s0[c * S + r];
                for (int i = 0; i < GCH; ++i) v += (double) dd[c * GCH + i] * k[i * S + r];
                ref[c * S + r] = v;
            }
        float* ds = st_upload(s0);
        float* dd_ = st_upload(dd);
        float* dk = st_upload(k);
        float* dout = dalloc<float>(16 * S);
        st_update_kernel<<<1, 32>>>(ds, dd_, dk, eL, dout);
        const double rel = st_max_rel(st_download(dout, 16 * S), ref);
        const bool bad = !(rel <= 2e-6);
        std::printf("  (c) S^T = eL S^T + Dd^T K (16x32 . 32x128, f16x3, ldmatrix.trans, 2^ed pre / post scale): %.2e  %s\n", rel, bad ? "FAIL" : "ok");
        fails += bad;
        cudaFree(ds); cudaFree(dd_); cudaFree(dk); cudaFree(dout);
    }
    {   // (d) D^T = Y^T T^T on f16x3 through gtc_prod_dt, a full random T of wide range with the prep's 2^et
        std::vector<float> y(16 * GCH), t(GCH * GCH);
        for (float& x : y) x = rng.logmag(-3.0f, 2.0f);
        for (float& x : t) x = rng.logmag(-2.0f, 3.0f);
        std::vector<double> ref(16 * GCH);
        for (int c = 0; c < 16; ++c)
            for (int i = 0; i < GCH; ++i) {
                double v = 0.0;
                for (int ip = 0; ip < GCH; ++ip) v += (double) y[c * GCH + ip] * t[i * GCH + ip];
                ref[c * GCH + i] = v;
            }
        float* dy = st_upload(y);
        float* dt_ = st_upload(t);
        float* dout = dalloc<float>(16 * GCH);
        st_dt_kernel<false><<<1, 32>>>(dy, dt_, st_scale_exp(t), dout);
        const double rel = st_max_rel(st_download(dout, 16 * GCH), ref);
        const bool bad = !(rel <= 2e-6);
        std::printf("  (d) D^T = Y^T T^T (16x32 . 32x32, f16x3, Y x 2^ey, T x 2^et): %.2e  %s\n", rel, bad ? "FAIL" : "ok");
        fails += bad;
        cudaFree(dy); cudaFree(dt_); cudaFree(dout);
    }
    {   // (e) the triangle-skipped tile sets against the full ones: T unit lower triangular, P lower triangular
        std::vector<float> y(16 * GCH), t(GCH * GCH), d(16 * GCH), p(GCH * GCH);
        for (float& x : y) x = rng.logmag(-3.0f, 2.0f);
        for (float& x : d) x = rng.logmag(-3.0f, 2.0f);
        for (int i = 0; i < GCH; ++i)
            for (int j = 0; j < GCH; ++j) {
                t[i * GCH + j] = j < i ? rng.logmag(-2.0f, 3.0f) : (j == i ? 1.0f : 0.0f);
                p[i * GCH + j] = j <= i ? 2.0f * rng.uni() - 1.0f : 0.0f;
            }
        std::vector<double> ref_d(16 * GCH), ref_o(16 * GCH);
        for (int c = 0; c < 16; ++c)
            for (int i = 0; i < GCH; ++i) {
                double v = 0.0, w = 0.0;
                for (int ip = 0; ip < GCH; ++ip) { v += (double) y[c * GCH + ip] * t[i * GCH + ip]; w += (double) d[c * GCH + ip] * p[i * GCH + ip]; }
                ref_d[c * GCH + i] = v;
                ref_o[c * GCH + i] = w;
            }
        float* dy = st_upload(y);
        float* dt_ = st_upload(t);
        float* dd_ = st_upload(d);
        float* dp = st_upload(p);
        float* full = dalloc<float>(16 * GCH);
        float* skip = dalloc<float>(16 * GCH);
        st_dt_kernel<false><<<1, 32>>>(dy, dt_, st_scale_exp(t), full);
        st_dt_kernel<true><<<1, 32>>>(dy, dt_, st_scale_exp(t), skip);
        const std::vector<float> f1 = st_download(full, 16 * GCH), s1 = st_download(skip, 16 * GCH);
        const unsigned long long diff1 = std::memcmp(f1.data(), s1.data(), 16 * GCH * 4) != 0 ? diff_words(full, skip, 16 * GCH * 4) : 0;
        const double rel1 = st_max_rel(s1, ref_d);
        st_dt_kernel<false><<<1, 32>>>(dd_, dp, 8, full);
        st_dt_kernel<true><<<1, 32>>>(dd_, dp, 8, skip);
        const std::vector<float> f2 = st_download(full, 16 * GCH), s2 = st_download(skip, 16 * GCH);
        const unsigned long long diff2 = std::memcmp(f2.data(), s2.data(), 16 * GCH * 4) != 0 ? diff_words(full, skip, 16 * GCH * 4) : 0;
        const double rel2 = st_max_rel(s2, ref_o);
        const bool bad = diff1 != 0 || diff2 != 0 || !(rel1 <= 2e-6) || !(rel2 <= 2e-6);
        std::printf("  (e) triangle skipping: D^T 10 of 16 tiles %llu words differ (vs FP64 %.2e), o^T 6 of 8 tiles %llu words differ (vs FP64 %.2e)  %s\n",
                    diff1, rel1, diff2, rel2, bad ? "FAIL" : "ok");
        fails += bad;
        cudaFree(dy); cudaFree(dt_); cudaFree(dd_); cudaFree(dp); cudaFree(full); cudaFree(skip);
    }
    std::printf("--selftest-frag: %d failures\n", fails);
    return fails;
}

float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0f : v[v.size() / 2];
}

// tc_only (--bench-tc): the tensor-core series alone, for A / B runs of two builds back to back
void bench(int64_t T, bool tc_only = false) {
    Inputs in(T, 0xB3);
    const size_t st = (size_t) S * HV * S, oc_n = (size_t) T * HV * S;
    float* state = dalloc<float>(st);
    float* oc = dalloc<float>(oc_n);
    uint16_t* y16 = dalloc<uint16_t>(oc_n);
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "event");
    ck(cudaEventCreate(&e1), "event");
    auto timed = [&](auto&& launch) {
        ck(cudaMemcpy(state, in.state0, st * 4, cudaMemcpyDeviceToDevice), "state");
        cudaEventRecord(e0);
        launch();
        cudaEventRecord(e1);
        ck(cudaEventSynchronize(e1), "timed");
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        return ms;
    };
    const int warm = 3, rounds = T >= 16384 ? 10 : 20;
    std::vector<float> t_rec[kRecCount], t_norm[2];
    for (int i = 0; !tc_only && i < warm + rounds; ++i) {
        for (int j = 0; j < kRecCount; ++j) {
            const Rec v = (Rec) ((i + j) % kRecCount);   // a different variant first in each round
            const float ms = timed([&] { run_rec(v, state, in, oc); });
            if (i >= warm) t_rec[v].push_back(ms);
        }
    }
    for (int i = 0; !tc_only && i < warm + rounds; ++i) {
        for (int j = 0; j < 2; ++j) {
            const int v = (i + j) % 2;
            const float ms = timed([&] {
                if (v == 0) gdn_out_norm_old_kernel<<<dim3((unsigned) T, HV), S>>>(in.z, in.gamma, 1e-6f, oc, y16);
                else gdn_out_norm_kernel<<<dim3((unsigned) T, HV), S>>>(in.z, in.gamma, 1e-6f, oc, y16);
            });
            if (i >= warm) t_norm[v].push_back(ms);
        }
    }
    std::vector<float> t_kc[2];   // gdn_rec_kh_kernel and the chunked recurrence, alternating
    const bool can_chunk = !tc_only && gdn_rec_chunked(state, in.h, in.gate, in.beta, oc, in.T, 0) == cudaSuccess &&
                           cudaDeviceSynchronize() == cudaSuccess;
    for (int i = 0; can_chunk && i < warm + rounds; ++i)
        for (int j = 0; j < 2; ++j) {
            const int v = (i + j) % 2;
            const float ms = timed([&] {
                if (v == 0) gdn_rec_kh_kernel<<<HK * NCB, dim3(CB, RG)>>>(state, in.h, in.gate, in.beta, oc, in.T);
                else ck(gdn_rec_chunked(state, in.h, in.gate, in.beta, oc, in.T, 0), "chunked");
            });
            if (i >= warm) t_kc[v].push_back(ms);
        }
    std::vector<float> t_tc[2];   // gdn_rec_kh_kernel and the tensor-core scan, alternating
    const bool can_tc = gdn_rec_tc(state, in.h, in.gate, in.beta, oc, in.T, 0) == cudaSuccess &&
                        cudaDeviceSynchronize() == cudaSuccess;
    for (int i = 0; can_tc && i < warm + rounds; ++i)
        for (int j = 0; j < 2; ++j) {
            const int v = (i + j) % 2;
            const float ms = timed([&] {
                if (v == 0) gdn_rec_kh_kernel<<<HK * NCB, dim3(CB, RG)>>>(state, in.h, in.gate, in.beta, oc, in.T);
                else ck(gdn_rec_tc(state, in.h, in.gate, in.beta, oc, in.T, 0), "tc");
            });
            if (i >= warm) t_tc[v].push_back(ms);
        }
    const float r0 = median(t_rec[kBefore]), r1 = median(t_rec[kKeyHead]), n0 = median(t_norm[0]), n1 = median(t_norm[1]);
    std::printf("  T = %lld tokens, one layer (48 value heads), medians of %d alternating runs:\n", (long long) T, rounds);
    if (tc_only) {
        const float kh = median(t_tc[0]), tc = median(t_tc[1]);
        std::printf("    recurrence, gdn_rec_kh_kernel                  %8.2f ms  (%.3f us per token)\n", kh, 1000.0f * kh / (float) T);
        std::printf("    recurrence, tensor-core chunked (STRATA_GDN_TC=1) %8.2f ms  (%.3f us per token)  %.2fx gdn_rec_kh_kernel\n",
                    tc, 1000.0f * tc / (float) T, kh / tc);
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaFree(state); cudaFree(oc); cudaFree(y16);
        return;
    }
    std::printf("    recurrence, before (gdn_rec_cols_pipe_kernel)  %8.2f ms  (%.3f us per token)\n", r0,
                1000.0f * r0 / (float) T);
    std::printf("    recurrence, gdn_rec_kh_kernel                  %8.2f ms  (%.3f us per token)  %.2fx\n", r1,
                1000.0f * r1 / (float) T, r0 / r1);
    if (can_chunk) {
        const float kh = median(t_kc[0]), kc = median(t_kc[1]);
        std::printf("    recurrence, chunked (STRATA_GDN_CHUNKED=1)     %8.2f ms  (%.3f us per token)  %.2fx gdn_rec_kh_kernel\n",
                    kc, 1000.0f * kc / (float) T, kh / kc);
    }
    if (can_tc) {
        const float kh = median(t_tc[0]), tc = median(t_tc[1]);
        std::printf("    recurrence, tensor-core chunked (STRATA_GDN_TC=1) %8.2f ms  (%.3f us per token)  %.2fx gdn_rec_kh_kernel\n",
                    tc, 1000.0f * tc / (float) T, kh / tc);
    }
    std::printf("    output norm, before (FP32 and FP16 store)      %8.2f ms\n", n0);
    std::printf("    output norm, FP16 store only                   %8.2f ms  %.2fx\n", n1, n0 / n1);
    std::printf("    36 DeltaNet layers per %lld-token block: recurrence + norm %.0f ms -> %.0f ms (-%.0f ms)\n",
                (long long) T, 36.0f * (r0 + n0), 36.0f * (r1 + n1), 36.0f * (r0 + n0 - r1 - n1));
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(state); cudaFree(oc); cudaFree(y16);
}

// gdn_rec_kh_kernel against the chunked recurrence on short chunks (the chunked one's launches, host side included:
// the GPU is idle at each start), where kernels.cu's kGdnChunkedMin comes from
void chunked_sweep() {
    std::printf("  short chunks, gdn_rec_kh_kernel -> chunked (medians of 21 alternating runs):\n");
    for (const int64_t T : {32, 64, 96, 128, 192, 256, 512, 1024}) {
        Inputs in(T, 0xC5);
        const size_t st = (size_t) S * HV * S;
        float* state = dalloc<float>(st);
        float* oc = dalloc<float>((size_t) T * HV * S);
        cudaEvent_t e0, e1;
        ck(cudaEventCreate(&e0), "event");
        ck(cudaEventCreate(&e1), "event");
        const bool can = gdn_rec_chunked(state, in.h, in.gate, in.beta, oc, in.T, 0) == cudaSuccess &&
                         cudaDeviceSynchronize() == cudaSuccess;
        std::vector<float> t[2];
        for (int i = 0; can && i < 5 + 21; ++i)
            for (int j = 0; j < 2; ++j) {
                const int v = (i + j) % 2;
                ck(cudaMemcpy(state, in.state0, st * 4, cudaMemcpyDeviceToDevice), "state");
                cudaEventRecord(e0);
                if (v == 0) gdn_rec_kh_kernel<<<HK * NCB, dim3(CB, RG)>>>(state, in.h, in.gate, in.beta, oc, in.T);
                else ck(gdn_rec_chunked(state, in.h, in.gate, in.beta, oc, in.T, 0), "chunked");
                cudaEventRecord(e1);
                ck(cudaEventSynchronize(e1), "timed");
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, e0, e1);
                if (i >= 5) t[v].push_back(ms);
            }
        if (can) {
            const float kh = median(t[0]), kc = median(t[1]);
            std::printf("    T = %-5lld %8.1f us -> %8.1f us  %.2fx\n", (long long) T, 1000.0f * kh, 1000.0f * kc, kh / kc);
        } else {
            std::printf("    chunked: not on this card\n");
        }
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        cudaFree(state); cudaFree(oc);
        if (!can) return;
    }
}

// The two kernels at a given number of blocks per SM, as on cards with more or fewer SMs than this one: the kernel
// before puts ceil(192 / SMs) of its blocks on an SM, gdn_rec_kh_kernel ceil(64 / SMs), and the busiest SM sets the
// pace (every block walks the whole chunk).  b blocks per SM here: b rounds of one block per SM, each round one launch
// of SMs blocks (the first heads; split in launches of at most the engine's grid), every launch on its own stream with
// its own state and output, all started together, so the block scheduler puts one block of each round on every SM.
// The kernels are the ones above, unchanged.
void scale(int64_t T) {
    int dev = 0, sms = 0, per_sm[kRecCount] = {};
    ck(cudaGetDevice(&dev), "device");
    ck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev), "SMs");
    ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm[kBefore], gdn_rec_cols_pipe_kernel, CB * RG, 0), "occupancy");
    ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm[kKeyHead], gdn_rec_kh_kernel, CB * RG, 0), "occupancy");
    constexpr int kMaxB = 6, kMaxLaunches = 6;
    const int full[kRecCount] = {HV * NCB, HK * NCB};   // the grids as the engine launches them
    auto parts = [&](int v) { return (sms + full[v] - 1) / full[v]; };   // launches per round
    Inputs in(T, 0xC7);
    const size_t st = (size_t) S * HV * S, oc_n = (size_t) T * HV * S;
    float* state[kMaxLaunches];
    float* oc[kMaxLaunches];
    cudaStream_t stream[kMaxLaunches];
    cudaEvent_t e0, e1, done[kMaxLaunches];
    ck(cudaEventCreate(&e0), "event");
    ck(cudaEventCreate(&e1), "event");
    for (int c = 0; c < kMaxLaunches; ++c) {
        state[c] = dalloc<float>(st);
        oc[c] = dalloc<float>(oc_n);
        ck(cudaStreamCreateWithFlags(&stream[c], cudaStreamNonBlocking), "stream");
        ck(cudaEventCreate(&done[c]), "event");
    }
    struct Case { int v, b; };
    std::vector<Case> cases;
    for (int v = 0; v < kRecCount; ++v)
        for (int b = 1; b <= std::min(kMaxB, per_sm[v]); ++b)
            if (b * parts(v) <= kMaxLaunches) cases.push_back({v, b});
    auto run = [&](const Case& cs) {
        for (int c = 0; c < kMaxLaunches; ++c) ck(cudaMemcpy(state[c], in.state0, st * 4, cudaMemcpyDeviceToDevice), "state");
        ck(cudaDeviceSynchronize(), "sync");
        const int np = parts(cs.v), n = cs.b * np;
        cudaEventRecord(e0, stream[0]);
        for (int c = 0; c < n; ++c) {
            const int k = c % np, grid = sms / np + (k < sms % np ? 1 : 0);   // round c / np, part k
            if (c > 0) cudaStreamWaitEvent(stream[c], e0, 0);
            if (cs.v == kBefore)
                gdn_rec_cols_pipe_kernel<<<grid, dim3(CB, RG), 0, stream[c]>>>(state[c], in.h, in.gate, in.beta, oc[c], T);
            else
                gdn_rec_kh_kernel<<<grid, dim3(CB, RG), 0, stream[c]>>>(state[c], in.h, in.gate, in.beta, oc[c], T);
            cudaEventRecord(done[c], stream[c]);
        }
        for (int c = 1; c < n; ++c) cudaStreamWaitEvent(stream[0], done[c], 0);
        cudaEventRecord(e1, stream[0]);
        ck(cudaEventSynchronize(e1), "scale");
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        return ms;
    };
    const int warm = 2, rounds = 9;
    std::vector<std::vector<float>> t(cases.size());
    for (int i = 0; i < warm + rounds; ++i)
        for (size_t j = 0; j < cases.size(); ++j) {
            const size_t k = (i + j) % cases.size();   // a different case first in each round
            const float ms = run(cases[k]);
            if (i >= warm) t[k].push_back(ms);
        }
    auto us = [&](int v, int b) {   // per token, or < 0 if not measured
        for (size_t k = 0; k < cases.size(); ++k)
            if (cases[k].v == v && cases[k].b == b) return 1000.0f * median(t[k]) / (float) T;
        return -1.0f;
    };
    std::printf("  per token at a given number of blocks per SM (%lld tokens, medians of %d alternating runs; this card has "
                "%d SMs):\n", (long long) T, rounds, sms);
    std::printf("    blocks per SM   before      gdn_rec_kh_kernel\n");
    for (int b = 1; b <= kMaxB; ++b) {
        const float u0 = us(kBefore, b), u1 = us(kKeyHead, b);
        if (u0 < 0 && u1 < 0) continue;
        char a0[32] = "-", a1[32] = "-";
        if (u0 >= 0) std::snprintf(a0, sizeof a0, "%.3f us", u0);
        if (u1 >= 0) std::snprintf(a1, sizeof a1, "%.3f us", u1);
        std::printf("    %8d        %-11s %s\n", b, a0, a1);
    }
    std::printf("  so on a card with N SMs (the kernel before puts ceil(192 / N) blocks on an SM, gdn_rec_kh_kernel "
                "ceil(64 / N), at most %d):\n", per_sm[kKeyHead]);
    const int bands[][2] = {{192, 0}, {96, 191}, {64, 95}, {48, 63}, {39, 47}, {32, 38}};
    for (const auto& bd : bands) {
        const int nsm = bd[0], b0 = (192 + nsm - 1) / nsm, b1 = (64 + nsm - 1) / nsm;
        const float u0 = us(kBefore, b0), u1 = us(kKeyHead, b1);
        char range[32];
        if (bd[1]) std::snprintf(range, sizeof range, "%d-%d SMs", bd[0], bd[1]);
        else std::snprintf(range, sizeof range, "%d+ SMs", bd[0]);
        if (u0 < 0 || u1 < 0)
            std::printf("    %-12s before %d / new %d per SM: not measured on this card\n", range, b0, b1);
        else
            std::printf("    %-12s before %d / new %d per SM: %.3f -> %.3f us per token, %.2fx\n", range, b0, b1, u0, u1,
                        u0 / u1);
    }
    for (int c = 0; c < kMaxLaunches; ++c) {
        cudaFree(state[c]);
        cudaFree(oc[c]);
        cudaStreamDestroy(stream[c]);
        cudaEventDestroy(done[c]);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
}

// The engine's grids on N of this card's SMs.  scale() puts as many blocks on every SM as the busiest SM of a card with
// N SMs gets; on that card only some SMs get that many (192 blocks on 80 SMs: 3 on 32 SMs, 2 on the other 48), the
// rest of the card has less to do, and the estimate came out 4-9 % above the real launch on two cards.  Here the
// recurrences run as the engine launches them, while hold_sm keeps all but N SMs: its blocks go out first, a small
// kernel on the timed launch's stream waits until they all sit, and one after the timed launch lets them go.  land()
// shows for every N where the two grids land: none on a held SM, and how many blocks share the busiest SM.  The
// recurrences are the ones above, unchanged; L2, memory and clock stay this card's.
void fewer_sms(int64_t T) {
    int dev = 0, sms = 0, optin = 0, smem_sm = 0, reserved = 0, hold_per_sm = 0, per_sm[kRecCount] = {};
    ck(cudaGetDevice(&dev), "device");
    ck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev), "SMs");
    ck(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev), "shared memory");
    ck(cudaDeviceGetAttribute(&smem_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev), "shared memory");
    ck(cudaDeviceGetAttribute(&reserved, cudaDevAttrReservedSharedMemoryPerBlock, dev), "shared memory");
    for (const void* f : {(const void*) hold_sm, (const void*) land}) {
        ck(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, optin), "shared memory");
        ck(cudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout, (int) cudaSharedmemCarveoutMaxShared),
           "carveout");
    }
    // Loaded now rather than at a first launch: with lazy loading (CUDA 12's default) a kernel's first launch may wait
    // until the device is idle, and so for hold_sm, which waits for release_sms.
    cudaFuncAttributes fa[kRecCount], fx;
    ck(cudaFuncGetAttributes(&fa[kBefore], gdn_rec_cols_pipe_kernel), "attributes");
    ck(cudaFuncGetAttributes(&fa[kKeyHead], gdn_rec_kh_kernel), "attributes");
    for (const void* f : {(const void*) hold_sm, (const void*) wait_sitting, (const void*) release_sms, (const void*) land})
        ck(cudaFuncGetAttributes(&fx, f), "attributes");
    ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&hold_per_sm, hold_sm, 1, optin), "occupancy");
    ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm[kBefore], gdn_rec_cols_pipe_kernel, CB * RG, 0), "occupancy");
    ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm[kKeyHead], gdn_rec_kh_kernel, CB * RG, 0), "occupancy");
    const int grid[kRecCount] = {HV * NCB, HK * NCB};   // as the engine launches them
    std::printf("  the engine's grids on N of this card's %d SMs (%lld tokens; the other SMs held by a sleeping kernel, see "
                "fewer_sms):\n", sms, (long long) T);
    const int left = smem_sm - optin - reserved;   // shared memory a holding block leaves on its SM
    if (hold_per_sm != 1 || (int) fa[kBefore].sharedSizeBytes + reserved <= left ||
        (int) fa[kKeyHead].sharedSizeBytes + reserved <= left) {
        std::printf("    not on this card: a block of a recurrence would fit beside a holding block (%d bytes left)\n",
                    left);
        return;
    }
    // land(): as much dynamic shared memory as still lets as many of its blocks share an SM as of the recurrence
    int land_smem[kRecCount];
    for (int v = 0; v < kRecCount; ++v) {
        int lo = 0, hi = optin;
        while (lo < hi) {
            const int mid = (lo + hi + 1) / 2;
            int nb = 0;
            ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, land, CB * RG, mid), "occupancy");
            if (nb >= per_sm[v]) lo = mid;
            else hi = mid - 1;
        }
        land_smem[v] = lo;
    }
    // N: all SMs, and below that the SM counts of cards and the band edges, as long as both grids fit at once
    std::vector<int> ns = {sms};
    for (const int n : {170, 144, 132, 128, 112, 108, 96, 95, 84, 82, 80, 76, 72, 70, 66, 64, 63, 60, 56, 52, 48, 47, 46, 44,
                        40, 39, 38, 36, 34, 32})
        if (n < sms && (int64_t) n * per_sm[kBefore] >= grid[kBefore] && (int64_t) n * per_sm[kKeyHead] >= grid[kKeyHead])
            ns.push_back(n);

    Inputs in(T, 0xD5);
    const size_t st = (size_t) S * HV * S, oc_n = (size_t) T * HV * S;
    float* state = dalloc<float>(st);
    float* oc = dalloc<float>(oc_n);
    unsigned long long* sitting = dalloc<unsigned long long>(1);
    unsigned* flags = dalloc<unsigned>(4);
    unsigned *release = flags, *late = flags + 1, *landed = flags + 2, *apart = flags + 3;
    int* where = dalloc<int>((size_t) sms + grid[kBefore]);   // the holding blocks' SMs, then land()'s
    ck(cudaMemset(sitting, 0, sizeof(*sitting)), "memset");
    ck(cudaMemset(flags, 0, 4 * sizeof(unsigned)), "memset");
    cudaStream_t s_hold, s_run;
    ck(cudaStreamCreateWithFlags(&s_hold, cudaStreamNonBlocking), "stream");
    ck(cudaStreamCreateWithFlags(&s_run, cudaStreamNonBlocking), "stream");
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "event");
    ck(cudaEventCreate(&e1), "event");
    unsigned long long all = 0;   // holding blocks launched so far: sitting counts every one that sat down
    unsigned gen = 0;
    auto hold = [&](int n) {   // all but n SMs, before the next launch on s_run
        if (n >= sms) return;
        ++gen;
        all += (unsigned long long) (sms - n);
        hold_sm<<<sms - n, 1, optin, s_hold>>>(sitting, release, gen, where, late);
        ck(cudaGetLastError(), "hold_sm");
        const cudaError_t q = cudaStreamQuery(s_hold);   // hands the launch to the GPU now (Windows collects launches)
        if (q != cudaErrorNotReady) ck(q, "hold_sm");
        (void) cudaGetLastError();   // not ready is no error
        wait_sitting<<<1, 1, 0, s_run>>>(sitting, all, late);
    };
    auto let_go = [&](int n) {
        if (n < sms) release_sms<<<1, 1, 0, s_run>>>(release, gen);
    };
    auto settle = [&](const char* what) {
        ck(cudaStreamSynchronize(s_run), what);
        ck(cudaStreamSynchronize(s_hold), what);
    };
    auto count = [&](const unsigned* c) {   // after settle()
        unsigned h = 0;
        ck(cudaMemcpy(&h, c, sizeof(h), cudaMemcpyDeviceToHost), "count");
        return h;
    };
    auto done = [&] {
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaStreamDestroy(s_hold);
        cudaStreamDestroy(s_run);
        cudaFree(state); cudaFree(oc); cudaFree(sitting); cudaFree(flags); cudaFree(where);
    };

    // where the grids land, once per N
    struct Landing { bool held = true; int busiest[kRecCount] = {}, on[kRecCount] = {}; };
    std::vector<Landing> lands(ns.size());
    for (size_t i = 0; i < ns.size(); ++i) {
        const int n = ns[i];
        for (int v = 0; v < kRecCount; ++v) {
            bool sat = false;
            unsigned apart0 = 0;
            for (int attempt = 0; attempt < 2 && !sat; ++attempt) {
                const unsigned late0 = count(late);
                apart0 = count(apart);
                ck(cudaMemsetAsync(landed, 0, sizeof(unsigned), s_run), "memset");
                hold(n);
                land<<<grid[v], dim3(CB, RG), land_smem[v], s_run>>>(where + sms, landed, apart);
                ck(cudaGetLastError(), "land");
                let_go(n);
                settle("land");
                sat = count(late) == late0;
            }
            if (!sat) {   // twice: the holding kernel and the launches beside it do not run at the same time here
                std::printf("    not on this card: the other SMs could not be held (the kernel holding them and the "
                            "launches beside it did not run at the same time)\n");
                done();
                return;
            }
            std::vector<int> w((size_t) sms + grid[v]);
            ck(cudaMemcpy(w.data(), where, w.size() * sizeof(int), cudaMemcpyDeviceToHost), "where");
            std::map<int, int> held, blocks;
            for (int b = 0; b < (n < sms ? sms - n : 0); ++b) ++held[w[b]];
            for (int b = 0; b < grid[v]; ++b) ++blocks[w[sms + b]];
            bool ok = count(apart) == apart0 && (int) held.size() == (n < sms ? sms - n : 0) && (int) blocks.size() <= n;
            int most = 0, on = 0;
            for (const auto& sb : blocks) {
                if (held.count(sb.first)) ok = false;
                if (sb.second > most) { most = sb.second; on = 0; }
                if (sb.second == most) ++on;
            }
            lands[i].held = lands[i].held && ok;
            lands[i].busiest[v] = most;
            lands[i].on[v] = on;
        }
    }

    // the timings: every N and recurrence once per round, a different one first in each round
    struct Case { size_t i; Rec v; };
    std::vector<Case> cases;
    for (size_t i = 0; i < ns.size(); ++i)
        if (lands[i].held)
            for (int v = 0; v < kRecCount; ++v) cases.push_back({i, (Rec) v});
    auto run = [&](const Case& cs) {   // ms, or < 0 if the SMs were not held for the whole launch
        const int n = ns[cs.i];
        const unsigned late0 = count(late);
        ck(cudaMemcpyAsync(state, in.state0, st * 4, cudaMemcpyDeviceToDevice, s_run), "state");
        hold(n);
        ck(cudaEventRecord(e0, s_run), "event");
        run_rec(cs.v, state, in, oc, s_run);
        ck(cudaEventRecord(e1, s_run), "event");
        let_go(n);
        settle("fewer SMs");
        if (count(late) != late0) return -1.0f;
        float ms = 0.0f;
        ck(cudaEventElapsedTime(&ms, e0, e1), "event");
        return ms;
    };
    const int warm = 2, rounds = 15, max_dropped = 5;
    std::vector<std::vector<float>> t(cases.size());
    int dropped = 0;
    for (int r = 0; r < warm + rounds && dropped <= max_dropped; ++r)
        for (size_t j = 0; j < cases.size() && dropped <= max_dropped; ++j) {
            const size_t k = (r + j) % cases.size();
            const float ms = run(cases[k]);
            if (ms < 0.0f) ++dropped;
            else if (r >= warm) t[k].push_back(ms);
        }
    if (dropped > max_dropped) {   // each such run waits for a limit of a second or so
        std::printf("    stopped: in %d runs the other SMs were not held for the whole launch\n", dropped);
        done();
        return;
    }
    auto us = [&](size_t i, int v) {   // per token, or < 0 if not measured
        for (size_t k = 0; k < cases.size(); ++k)
            if (cases[k].i == i && cases[k].v == v && !t[k].empty()) return 1000.0f * median(t[k]) / (float) T;
        return -1.0f;
    };
    auto blocks_on = [](char* out, size_t size, int most, int on) {
        std::snprintf(out, size, "%d block%s on %d SM%s", most, most == 1 ? "" : "s", on, on == 1 ? "" : "s");
    };
    std::printf("  (medians of %d alternating runs; L2, memory and clock stay this card's)\n", rounds);
    std::printf("      N   before: busiest SM   new: busiest SM      us per token, before -> new\n");
    for (size_t i = 0; i < ns.size(); ++i) {
        if (!lands[i].held) {
            std::printf("    %3d   the other SMs were not held (a block landed on one, or a wait ran out)\n", ns[i]);
            continue;
        }
        char b0[40], b1[40];
        blocks_on(b0, sizeof b0, lands[i].busiest[kBefore], lands[i].on[kBefore]);
        blocks_on(b1, sizeof b1, lands[i].busiest[kKeyHead], lands[i].on[kKeyHead]);
        const float u0 = us(i, kBefore), u1 = us(i, kKeyHead);
        if (u0 < 0.0f || u1 < 0.0f) std::printf("    %3d   %-20s %-20s not measured\n", ns[i], b0, b1);
        else std::printf("    %3d   %-20s %-20s %.3f -> %.3f   %.2fx\n", ns[i], b0, b1, u0, u1, u0 / u1);
    }
    if (dropped) std::printf("    (%d runs dropped: the other SMs were not held for the whole launch)\n", dropped);
    std::printf("  so on a card with N SMs, as measured here (the kernel before puts ceil(192 / N) blocks on the busiest SM, "
                "gdn_rec_kh_kernel ceil(64 / N)):\n");
    const int bands[][2] = {{96, 191}, {64, 95}, {48, 63}, {39, 47}, {32, 38}};
    for (const auto& bd : bands) {
        const int b0 = (192 + bd[0] - 1) / bd[0], b1 = (64 + bd[0] - 1) / bd[0];
        float lo = 0.0f, hi = 0.0f;
        int cnt = 0, nmin = 0, nmax = 0;
        for (size_t i = 0; i < ns.size(); ++i) {
            const float u0 = us(i, kBefore), u1 = us(i, kKeyHead);
            if (ns[i] < bd[0] || ns[i] > bd[1] || u0 < 0.0f || u1 < 0.0f) continue;
            const float x = u0 / u1;
            if (cnt == 0 || x < lo) lo = x;
            if (cnt == 0 || x > hi) hi = x;
            if (cnt == 0 || ns[i] < nmin) nmin = ns[i];
            if (cnt == 0 || ns[i] > nmax) nmax = ns[i];
            ++cnt;
        }
        char range[32];
        std::snprintf(range, sizeof range, "%d-%d SMs", bd[0], bd[1]);
        if (cnt > 0)
            std::printf("    %-12s before %d / new %d per SM: %.2fx to %.2fx (%d to %d SMs)\n", range, b0, b1, lo, hi, nmin,
                        nmax);
        else if (bd[0] > sms)
            std::printf("    %-12s before %d / new %d per SM: more SMs than this card has (see the estimate above)\n", range,
                        b0, b1);
        else
            std::printf("    %-12s before %d / new %d per SM: not measured\n", range, b0, b1);
    }
    done();
}

}  // namespace

int main(int argc, char** argv) {
    bool do_bench = false, do_self = false, do_long = false, do_bench_tc = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0) do_bench = true;
        if (std::strcmp(argv[i], "--selftest-frag") == 0) do_self = true;
        if (std::strcmp(argv[i], "--long") == 0) do_long = true;
        if (std::strcmp(argv[i], "--bench-tc") == 0) do_bench_tc = true;
    }
    cudaDeviceProp p{};
    ck(cudaGetDeviceProperties(&p, 0), "device");
    std::printf("gdn_rec_parity on %s (sm_%d%d)\n", p.name, p.major, p.minor);
    int fails = 0;
    if (do_self) {   // the tensor-core helpers alone (one warp each, any sm_80+ card; the device code below sm_80 traps)
        if (p.major < 8) { std::printf("--selftest-frag: the tensor-core helpers need sm_80 or newer (skipped)\n"); return 77; }
        fails = selftest_frag();
        std::printf("gdn_rec_parity: %d failures\n", fails);
        return fails ? 1 : 0;
    }
    if (do_bench_tc) {   // the tensor-core timings alone
        for (const int64_t T : {(int64_t) 2048, (int64_t) 8192, (int64_t) 32768}) bench(T, true);
        return 0;
    }
    const int64_t Ts[] = {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 257, 1000, 4099};
    uint64_t seed = 11;
    for (const int64_t T : Ts) fails += check_bits(T, seed++);
    for (const int64_t T : Ts) fails += check_chunked(T, seed++);
    for (const int64_t T : {(int64_t) 8192, (int64_t) 32768}) fails += check_chunked(T, seed++);
    for (const int64_t T : Ts) fails += check_tc(T, seed++);
    for (const int64_t T : {(int64_t) 8192, (int64_t) 32768}) fails += check_tc(T, seed++);
    for (const int64_t T : {(int64_t) 4099, (int64_t) 32768}) { fails += check_tc_stress(T, seed); seed += 8; }
    if (do_long) fails += check_tc(51820, seed++);   // the engine's longest prompt (about 7 GB of inputs and outputs)
    if (do_bench) {
        std::printf("--bench: the prompt path's DeltaNet recurrence\n");
        bench(2048);
        bench(8192);
        bench(32768);
        chunked_sweep();
        scale(8192);
        fewer_sms(8192);
    }
    std::printf("gdn_rec_parity: %d failures\n", fails);
    return fails ? 1 : 0;
}
