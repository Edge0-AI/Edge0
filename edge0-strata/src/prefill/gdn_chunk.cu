// src/prefill/gdn_chunk.cu - see include/strata/prefill/gdn_chunk.hpp for the math and the contract.
//
// Three kernels, the same three stages as llama.cpp-v100's gdn-chunk-sm70 (cumsum / kkt / fwd), hand-written
// plain CUDA - FP32 throughout, no FP16 staging and no tensor cores yet (the TileLang m8n8k4 output of the
// reference is a later, speed-only step).  What the chunking buys is the shape of the serial chain: T tokens
// become T/64 chunk steps, each step a batch of dense matmuls over 64 tokens instead of one token's rank-1
// update behind two __syncthreads.
//
// Per CTA: one (v head, 32-column slice) of the state, all chunks in order (the state slice lives in
// registers - the chunk-serial scan of the reference's fwd kernel).  The per-chunk 64x64 matrices A and P are
// computed once per (chunk, head) by the kkt kernel and read by the four column slices.
#include "strata/prefill/gdn_chunk.hpp"

#include <cuda_runtime.h>
#if !defined(__HIPCC__)
#include <cuda_fp16.h>
#include <mma.h>
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace strata::prefill {
namespace {

constexpr int S = 128, HK = 16, HV = 48, C = 10240;  // the geometry of src/prefill/kernels.cu
constexpr int CH = 64;                               // tokens per chunk (the reference's chunk)
constexpr int NS = 32;                               // value columns per fwd CTA
constexpr int NSE = S / NS;                          // 4 slices per head
constexpr int SEG = 32;                              // chunks per pipeline segment (sizes the workspace)
// the wmma kernel's dynamic shared memory: Ks/Qs f16, W16 f16, Ug f32 (Gram scratch / U out), As/Ps f16,
// dlt f32 and its f16 image - 84 KB, past the 48 KB default (opt-in), under the 96 KB Volta ceiling
constexpr int WMMA_SMEM = (2 * CH * S + S * NS) * (int) sizeof(__half) + S * NS * (int) sizeof(float) +
                          2 * CH * CH * (int) sizeof(__half) + CH * NS * (int) sizeof(float) +
                          CH * NS * (int) sizeof(__half);

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "gdn_chunk %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// g_cumsum[lt * HV + h] = inclusive cumsum of gate over the segment's tokens (lt = segment-local token).
// One block of 64 per (chunk, head): a Kogge-Stone scan of 64 values, two tokens' worth of nothing.
__global__ void __launch_bounds__(CH) gdn_chunk_cumsum_kernel(float* __restrict__ gcum,
                                                              const float* __restrict__ gate, int ntok) {
    const int h = blockIdx.y;
    const int lt = blockIdx.x * CH + threadIdx.x;
    __shared__ float sh[CH];
    sh[threadIdx.x] = lt < ntok ? gate[(size_t) lt * HV + h] : 0.0f;
    __syncthreads();
#pragma unroll
    for (int off = 1; off < CH; off <<= 1) {
        const float add = threadIdx.x >= off ? sh[threadIdx.x - off] : 0.0f;
        __syncthreads();
        sh[threadIdx.x] += add;
        __syncthreads();
    }
    if (lt < ntok) gcum[(size_t) lt * HV + h] = sh[threadIdx.x];
}

// Per (chunk, head): the decayed Gram matrices the fwd kernel's solve and output sum read,
//   A[t][s] = beta_t exp(c_t - c_s) (k_t . k_s)  for s < t   (the (I + strict_lower) solve matrix)
//   P[t][s] = exp(c_t - c_s) (q_t . k_s)          for s <= t  (the output sum)
// in f32 (CH x CH per chunk-head; the upper triangle of A and P is never written and never read).  Every
// decay factor is a decay from a LATER token back to an EARLIER one - <= 1 for a real GDN gate - so nothing
// here can overflow, and delta stays O(beta v) in the fwd solve (gdn_chunk.hpp derives the form).
__global__ void __launch_bounds__(128) gdn_chunk_kkt_kernel(float* __restrict__ A, float* __restrict__ P,
                                                            const float* __restrict__ h,
                                                            const float* __restrict__ beta,
                                                            const float* __restrict__ gcum, int nchunks) {
    // 2 * CH * SP f32: the k and q tiles (opt-in, > 48 KB).  SP pads the row so the dot loop's row-strided
    // reads do not all land in one shared bank (S = 128 floats is a whole number of bank rows).
    constexpr int SP = S + 1;
    extern __shared__ float sh[];
    float* ks = sh;
    float* qs = sh + CH * SP;
    const int hh = blockIdx.y, ch = blockIdx.x, qh = hh % HK;
    const int64_t base = (size_t) ch * CH;
    const float* kh = h + base * C + (size_t) HK * S + (size_t) qh * S;
    const float* qh_ = h + base * C + (size_t) qh * S;
    for (int i = threadIdx.x; i < CH * S; i += blockDim.x) {
        const int t = i / S, j = i % S;
        ks[t * SP + j] = kh[(size_t) t * C + j];
        qs[t * SP + j] = qh_[(size_t) t * C + j];
    }
    __syncthreads();
    float* Ap = A + ((size_t) ch * HV + hh) * CH * CH;
    float* Pp = P + ((size_t) ch * HV + hh) * CH * CH;
    for (int p = threadIdx.x; p < CH * CH; p += blockDim.x) {
        const int t = p / CH, s = p % CH;
        if (s > t) continue;
        const float dec = expf(gcum[(base + t) * HV + hh] - gcum[(base + s) * HV + hh]);
        float pq = 0.0f, pq1 = 0.0f;
#pragma unroll 4
        for (int i = 0; i < S; i += 2) {
            pq = fmaf(qs[t * SP + i], ks[s * SP + i], pq);
            pq1 = fmaf(qs[t * SP + i + 1], ks[s * SP + i + 1], pq1);
        }
        Pp[t * CH + s] = (pq + pq1) * dec;
        if (s < t) {
            float pk = 0.0f, pk1 = 0.0f;
#pragma unroll 4
            for (int i = 0; i < S; i += 2) {
                pk = fmaf(ks[t * SP + i], ks[s * SP + i], pk);
                pk1 = fmaf(ks[t * SP + i + 1], ks[s * SP + i + 1], pk1);
            }
            Ap[t * CH + s] = (pk + pk1) * dec * beta[(base + t) * HV + hh];
        }
    }
}

// The chunk-serial scan: one CTA per (v head, 32-column slice), state in registers, all chunks in order.
// Per token: the two row-block reductions (k_t.W_in, q_t.W_in) give u_t and qW_t; the w==0 lane of each
// column solves its delta and writes o.  The A/P tiles are staged to shared memory at the top of each chunk:
// the solve and output sums are serial FMA chains, and a global load inside the chain (L2, ~200 cycles)
// throttles the whole CTA to it.  One __syncthreads per token (the shared dlt row plus the partials), one
// after the token loop (the last dlt row before the state update).
__global__ void __launch_bounds__(128) gdn_chunk_fwd_kernel(float* __restrict__ state, float* __restrict__ oc,
                                                            const float* __restrict__ h,
                                                            const float* __restrict__ beta,
                                                            const float* __restrict__ gcum,
                                                            const float* __restrict__ A,
                                                            const float* __restrict__ P, int nchunks) {
    const int hh = blockIdx.y, sl = blockIdx.x, j0 = sl * NS, qh = hh % HK;
    const int w = threadIdx.x >> 5, c = threadIdx.x & 31, col = j0 + c;
    // red is double-buffered by token parity: the w==0 lane reads token t's partials while the other warps
    // may already be staging token t+1's into the other slot (one barrier per token, no reader/writer race)
    __shared__ float red[2][2][4][NS], dlt[CH][NS], As[CH][CH], Ps[CH][CH], ex[CH][2];
    float W[NS];
#pragma unroll
    for (int r = 0; r < NS; ++r)
        W[r] = state[((size_t) (w * NS + r) * HV + hh) * S + col];
    const float rsqrt_S = rsqrtf((float) S);
    for (int ch = 0; ch < nchunks; ++ch) {
        const int64_t base = (size_t) ch * CH;
        const float sg = gcum[(base + CH - 1) * HV + hh];  // sigma: the chunk's total log-decay
        const float* Ap = A + ((size_t) ch * HV + hh) * CH * CH;
        const float* Pp = P + ((size_t) ch * HV + hh) * CH * CH;
        // stage the two 64x64 tiles and the per-token decay factors (exp(c_t), exp(sigma - c_t))
        for (int i = threadIdx.x; i < CH * CH; i += blockDim.x) {
            As[i / CH][i % CH] = Ap[i];
            Ps[i / CH][i % CH] = Pp[i];
        }
        if (threadIdx.x < CH) {
            const float gc = gcum[(base + threadIdx.x) * HV + hh];
            ex[threadIdx.x][0] = expf(gc);
            ex[threadIdx.x][1] = expf(sg - gc);
        }
        __syncthreads();
        const float sg_exp = ex[CH - 1][0];  // exp(sigma)
        for (int t = 0; t < CH; ++t) {
            const float* kt = h + (base + t) * C + (size_t) HK * S + (size_t) qh * S + w * NS;
            const float* qt = h + (base + t) * C + (size_t) qh * S + w * NS;
            float up = 0.0f, up1 = 0.0f, qp = 0.0f, qp1 = 0.0f;
#pragma unroll
            for (int r = 0; r < NS; r += 2) {
                const float w0 = W[r], w1 = W[r + 1];
                up = fmaf(kt[r], w0, up);
                up1 = fmaf(kt[r + 1], w1, up1);
                qp = fmaf(qt[r], w0, qp);
                qp1 = fmaf(qt[r + 1], w1, qp1);
            }
            red[t & 1][0][w][c] = up + up1;
            red[t & 1][1][w][c] = qp + qp1;
            __syncthreads();
            if (w == 0) {
                const float u_t = red[t & 1][0][0][c] + red[t & 1][0][1][c] + red[t & 1][0][2][c] + red[t & 1][0][3][c];
                const float qW_t =
                    red[t & 1][1][0][c] + red[t & 1][1][1][c] + red[t & 1][1][2][c] + red[t & 1][1][3][c];
                const float b = beta[(base + t) * HV + hh];
                const float v_t = h[(base + t) * C + (size_t) 2 * HK * S + (size_t) hh * S + col];
                // delta_t = beta_t (v_t - exp(c_t) u_t) - sum_{s<t} A[t][s] delta_s
                float d = b * (v_t - ex[t][0] * u_t), d1 = 0.0f;
#pragma unroll 1
                for (int s = 0; s < t; s += 2) {
                    d = fmaf(-As[t][s], dlt[s][c], d);
                    if (s + 1 < t) d1 = fmaf(-As[t][s + 1], dlt[s + 1][c], d1);
                }
                d += d1;
                dlt[t][c] = d;
                float o = ex[t][0] * qW_t + Ps[t][t] * d, o1 = 0.0f;
#pragma unroll 1
                for (int s = 0; s < t; s += 2) {
                    o = fmaf(Ps[t][s], dlt[s][c], o);
                    if (s + 1 < t) o1 = fmaf(Ps[t][s + 1], dlt[s + 1][c], o1);
                }
                oc[((base + t) * HV + hh) * S + col] = (o + o1) * rsqrt_S;
            }
        }
        __syncthreads();  // the last dlt row is visible for the update
        // W_out = exp(sigma) W_in + sum_s k_s exp(sigma - c_s) delta_s^T   (the decay folded into the delta)
#pragma unroll
        for (int r = 0; r < NS; ++r) W[r] *= sg_exp;
        for (int s = 0; s < CH; ++s) {
            const float* kt = h + (base + s) * C + (size_t) HK * S + (size_t) qh * S + w * NS;
            const float ds = dlt[s][c] * ex[s][1];
#pragma unroll
            for (int r = 0; r < NS; ++r) W[r] = fmaf(kt[r], ds, W[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < NS; ++r) state[((size_t) (w * NS + r) * HV + hh) * S + col] = W[r];
}

// ---------------------------------------------------------------- wmma (m16n16k16) variant
//
// The FMA scan above is correct but latency-bound: per token it streams k_t and q_t from global into
// dependent FMA chains, and V100 FMA throughput cannot pay for that at this granularity (measured on the
// V100: ~2x slower than the recurrence it was meant to beat).  This variant runs the same math the way
// llama.cpp-v100's gdn-chunk-sm70 does it - q/k/v and the state staged through FP16, the matmuls on the
// FP16 tensor cores with FP32 accumulate, the triangular solve still FMA - and folds the kkt stage's Gram
// matrices into the same CTA (recomputed per column slice, which the tensor cores make cheap).
//
// Per chunk: stage k/q, -> f16; Gram K K^T and Q K^T -> A and P (decay + beta baked); the GEMM
// U = [K; Q] W gives u_t and qW_t for every token at once; the solve and output sums are the FMA scan's;
// W += K^T (delta * decay) on the tensor cores with W itself as the f32 accumulator fragments - the state
// never leaves the registers across chunks.  W's f16 image exists only as the U GEMM's operand.
//
// Precision: the FP16 staging is the reference's "f32 -> f16 暂存" (10-bit mantissa on the operands);
// accumulations, the solve and the state are f32.  FP32-level, not bit-exact - see gdn_chunk_parity.
#if !defined(__HIPCC__)

__global__ void __launch_bounds__(128) gdn_chunk_wmma_kernel(float* __restrict__ state, float* __restrict__ oc,
                                                             const float* __restrict__ h,
                                                             const float* __restrict__ beta,
                                                             const float* __restrict__ gcum, int nchunks) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700   // named in every device pass of a multi-arch build; only sm_70+ has wmma
    namespace wmma = ::nvcuda::wmma;
    // smem (84.5 KB, opt-in):  Ks/Qs the chunk's k and q rows f16; W16 the state slice f16 (U's operand);
    //  Ug[128][32] f32 aliasing the 64x64 Gram scratch and then U = [K;Q] W;  As/Ps f16;  dlt f32 + f16.
    extern __shared__ __align__(1024) unsigned char smem[];
    __half* Ks = (__half*) smem;                                   // [CH][S]
    __half* Qs = Ks + CH * S;                                      // [CH][S]
    __half* W16 = Qs + CH * S;                                     // [S][NS]
    float* Ug = (float*) (W16 + S * NS);                           // [S][NS] f32  (Gram scratch / U out)
    __half* As = (__half*) (Ug + S * NS);                          // [CH][CH]
    __half* Ps = As + CH * CH;                                     // [CH][CH]
    float* dlt = (float*) (Ps + CH * CH);                          // [CH][NS]
    __half* dlt16 = (__half*) (dlt + CH * NS);                     // [CH][NS]

    const int hh = blockIdx.y, sl = blockIdx.x, j0 = sl * NS, qh = hh % HK;
    const int w = threadIdx.x >> 5, c = threadIdx.x & 31, col = j0 + c;
    // the state slice in through smem (load_matrix_sync of an accumulator tile is a shared-memory op on
    // sm_70): Ug first as the f32 staging, then W's f32 image in the fragments and its f16 image in W16
    for (int i = threadIdx.x; i < S * NS; i += blockDim.x) {
        const int r = i / NS, j = i % NS;
        Ug[i] = state[(size_t) r * HV * S + (size_t) hh * S + j0 + j];
        W16[i] = __float2half_rn(Ug[i]);
    }
    __syncthreads();
    // W as f32 accumulator fragments: warp w owns rows w*32..w*32+31 of the 128x32 slice, 2x2 tiles
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> wF[2][2];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int nj = 0; nj < 2; ++nj)
            wmma::load_matrix_sync(wF[mi][nj], Ug + (size_t) (w * 32 + 16 * mi) * NS + 16 * nj, NS,
                                   wmma::mem_row_major);
    const float rsqrt_S = rsqrtf((float) S);
    for (int ch = 0; ch < nchunks; ++ch) {
        const int64_t base = (size_t) ch * CH;
        const float sg = gcum[(base + CH - 1) * HV + hh];  // sigma: the chunk's total log-decay
        // stage the chunk's k and q rows through f16 - float4 loads, four in flight per thread: a scalar
        // load loop here is a serial global-latency chain per thread (measured: it WAS the kernel's floor)
        for (int i4 = threadIdx.x; i4 < CH * S / 4; i4 += blockDim.x) {
            const int t = (i4 * 4) / S, j4 = (i4 * 4) % S;
            const float4 k4 = *reinterpret_cast<const float4*>(h + (base + t) * C + (size_t) HK * S + (size_t) qh * S + j4);
            const float4 q4 = *reinterpret_cast<const float4*>(h + (base + t) * C + (size_t) qh * S + j4);
            Ks[t * S + j4 + 0] = __float2half_rn(k4.x);
            Ks[t * S + j4 + 1] = __float2half_rn(k4.y);
            Ks[t * S + j4 + 2] = __float2half_rn(k4.z);
            Ks[t * S + j4 + 3] = __float2half_rn(k4.w);
            Qs[t * S + j4 + 0] = __float2half_rn(q4.x);
            Qs[t * S + j4 + 1] = __float2half_rn(q4.y);
            Qs[t * S + j4 + 2] = __float2half_rn(q4.z);
            Qs[t * S + j4 + 3] = __float2half_rn(q4.w);
        }
        // the state slice as f16: the fragments to Ug, then to W16
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int nj = 0; nj < 2; ++nj)
                wmma::store_matrix_sync(Ug + (size_t) (w * 32 + 16 * mi) * NS + 16 * nj, wF[mi][nj], NS,
                                        wmma::mem_row_major);
        __syncthreads();
        for (int i = threadIdx.x; i < S * NS; i += blockDim.x) W16[i] = __float2half_rn(Ug[i]);
        __syncthreads();  // W16 done before Ug becomes the Gram scratch
        // Gram:  Ug = K K^T, then As[t][s] = beta_t exp(c_t-c_s) Ug[t][s];  Ug = Q K^T, then Ps (no beta)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> aF;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bF;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> gF[4];
            for (int pass = 0; pass < 2; ++pass) {
#pragma unroll
                for (int nj = 0; nj < 4; ++nj) wmma::fill_fragment(gF[nj], 0.0f);
                for (int kk = 0; kk < S / 16; ++kk) {
                    wmma::load_matrix_sync(aF, (pass ? Qs : Ks) + (size_t) (w * 16) * S + kk * 16, S);
                    for (int nj = 0; nj < 4; ++nj) {
                        wmma::load_matrix_sync(bF, Ks + kk * 16 + (size_t) (16 * nj) * S, S);
                        wmma::mma_sync(gF[nj], aF, bF, gF[nj]);
                    }
                }
                for (int nj = 0; nj < 4; ++nj)
                    wmma::store_matrix_sync(Ug + (size_t) (w * 16) * CH + 16 * nj, gF[nj], CH, wmma::mem_row_major);
                __syncthreads();
                // fold decay (and beta) into the f16 tiles
                for (int p = threadIdx.x; p < CH * CH; p += blockDim.x) {
                    const int t = p / CH, s = p % CH;
                    const float dec = expf(gcum[(base + t) * HV + hh] - gcum[(base + s) * HV + hh]);
                    if (pass)  // Ps: s <= t live, the upper triangle zeroed - the o GEMM reads the whole tile
                        Ps[t * CH + s] = s <= t ? __float2half_rn(Ug[t * CH + s] * dec) : __float2half_rn(0.0f);
                    else if (s < t)  // As: strict lower only, the solve reads exactly that
                        As[t * CH + s] = __float2half_rn(Ug[t * CH + s] * dec * beta[(base + t) * HV + hh]);
                }
                __syncthreads();
            }
        }
        // U = [K; Q] W16 : rows 0..63 are u_t = k_t.W_in, rows 64..127 are qW_t = q_t.W_in.  The 128 output
        // rows split as 32 per warp: warps 0-1 the K half, warps 2-3 the Q half.
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> aF;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bF[2];
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> uF[2][2];
            const __half* src = w < 2 ? Ks : Qs;
            const int row0 = (w & 1) * 32;
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                for (int nj = 0; nj < 2; ++nj) wmma::fill_fragment(uF[mi][nj], 0.0f);
            for (int kk = 0; kk < S / 16; ++kk) {
                for (int nj = 0; nj < 2; ++nj)
                    wmma::load_matrix_sync(bF[nj], W16 + kk * 16 * NS + 16 * nj, NS);
#pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    wmma::load_matrix_sync(aF, src + (size_t) (row0 + 16 * mi) * S + kk * 16, S);
                    for (int nj = 0; nj < 2; ++nj) wmma::mma_sync(uF[mi][nj], aF, bF[nj], uF[mi][nj]);
                }
            }
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                for (int nj = 0; nj < 2; ++nj)
                    wmma::store_matrix_sync(Ug + (size_t) (w * 32 + 16 * mi) * NS + 16 * nj, uF[mi][nj], NS,
                                            wmma::mem_row_major);
            __syncthreads();
        }
        // the solve - 8 independent accumulator chains so the smem loads of one group hide behind the FMAs
        // of the last (the 2-chain was smem-latency-bound and WAS half the kernel's runtime)
        for (int t = 0; t < CH; ++t) {
            if (w == 0) {
                const float gc = gcum[(base + t) * HV + hh];
                const float ex0 = expf(gc);
                const float b = beta[(base + t) * HV + hh];
                const float v_t = h[(base + t) * C + (size_t) 2 * HK * S + (size_t) hh * S + col];
                float d[8];
                d[0] = b * (v_t - ex0 * Ug[t * NS + c]);
#pragma unroll
                for (int q = 1; q < 8; ++q) d[q] = 0.0f;
                int s0 = 0;
#pragma unroll 1
                for (; s0 + 8 <= t; s0 += 8) {
#pragma unroll
                    for (int q = 0; q < 8; ++q)
                        d[q] = fmaf(-__half2float(As[t * CH + s0 + q]), dlt[(s0 + q) * NS + c], d[q]);
                }
#pragma unroll 1
                for (; s0 < t; ++s0) d[0] = fmaf(-__half2float(As[t * CH + s0]), dlt[s0 * NS + c], d[0]);
                dlt[t * NS + c] = (d[0] + d[1] + d[2] + d[3]) + (d[4] + d[5] + d[6] + d[7]);
            }
        }
        __syncthreads();  // the deltas are complete
        for (int i = threadIdx.x; i < CH * NS; i += blockDim.x) dlt16[i] = __float2half_rn(dlt[i]);
        __syncthreads();
        // o_intra = Ps delta on the tensor cores (the o FMA sum was the other latency chain), staged into the
        // As tile (dead since the solve); 64 output rows split 16 per warp
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> aF;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bF[2];
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> oF[2];
            for (int nj = 0; nj < 2; ++nj) wmma::fill_fragment(oF[nj], 0.0f);
            for (int kk = 0; kk < CH / 16; ++kk) {
                for (int nj = 0; nj < 2; ++nj)
                    wmma::load_matrix_sync(bF[nj], dlt16 + (size_t) (kk * 16) * NS + 16 * nj, NS);
                wmma::load_matrix_sync(aF, Ps + (size_t) (w * 16) * CH + kk * 16, CH);
                for (int nj = 0; nj < 2; ++nj) wmma::mma_sync(oF[nj], aF, bF[nj], oF[nj]);
            }
            for (int nj = 0; nj < 2; ++nj)
                wmma::store_matrix_sync(Ug + S * NS + (size_t) (w * 16) * NS + 16 * nj, oF[nj], NS,
                                        wmma::mem_row_major);  // (the As tile's byte range)
        }
        __syncthreads();
        for (int i = threadIdx.x; i < CH * NS; i += blockDim.x) {
            const int t = i / NS;
            oc[((base + t) * HV + hh) * S + j0 + (i % NS)] =
                (Ug[S * NS + i] + expf(gcum[(base + t) * HV + hh]) * Ug[(64 + t) * NS + (i % NS)]) * rsqrt_S;
        }
        __syncthreads();
        // dlt16 = f16(delta_s * exp(sigma - c_s)) - the decay folded in for the state update
        for (int i = threadIdx.x; i < CH * NS; i += blockDim.x) {
            const int s = i / NS;
            dlt16[i] = __float2half_rn(__half2float(dlt16[i]) * expf(sg - gcum[(base + s) * HV + hh]));
        }
        const float sg_exp = expf(sg);
        __syncthreads();
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int nj = 0; nj < 2; ++nj)
#pragma unroll
                for (int q = 0; q < 8; ++q) wF[mi][nj].x[q] *= sg_exp;
        {
            {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> aF;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bF[2];
            for (int kk = 0; kk < CH / 16; ++kk) {
                for (int nj = 0; nj < 2; ++nj)
                    wmma::load_matrix_sync(bF[nj], dlt16 + (size_t) (kk * 16) * NS + 16 * nj, NS);
#pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    wmma::load_matrix_sync(aF, Ks + (w * 32 + 16 * mi) + (size_t) (kk * 16) * S, S);
                    for (int nj = 0; nj < 2; ++nj) wmma::mma_sync(wF[mi][nj], aF, bF[nj], wF[mi][nj]);
                }
            } }
        }
    }
    // the state slice out through smem as well
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int nj = 0; nj < 2; ++nj)
            wmma::store_matrix_sync(Ug + (size_t) (w * 32 + 16 * mi) * NS + 16 * nj, wF[mi][nj], NS,
                                    wmma::mem_row_major);
    __syncthreads();
    for (int i = threadIdx.x; i < S * NS; i += blockDim.x) {
        const int r = i / NS, j = i % NS;
        state[(size_t) r * HV * S + (size_t) hh * S + j0 + j] = Ug[i];
    }
#endif
}
#endif  // !HIPCC

// The grow-only workspace, one per (thread, device): a layer split runs the chunked recurrence on more than
// one card, and each card's pointers are only valid on that card - a single per-thread buffer would hand a
// second card the first card's pointers.
struct Ws {
    float* A = nullptr;      // [SEG][HV][CH][CH]
    float* P = nullptr;      // [SEG][HV][CH][CH]
    float* gcum = nullptr;   // [SEG*CH][HV]
    size_t cap = 0;          // in chunks
    void ensure(size_t chunks) {
        if (chunks <= cap) return;
        if (A) cudaFree(A);
        if (P) cudaFree(P);
        if (gcum) cudaFree(gcum);
        cudaGetLastError();   // a free of another device's pointer must not surface in the mallocs below
        cap = chunks;
        const size_t mat = chunks * HV * CH * CH;
        if (cudaMalloc((void**) &A, mat * sizeof(float)) != cudaSuccess ||
            cudaMalloc((void**) &P, mat * sizeof(float)) != cudaSuccess ||
            cudaMalloc((void**) &gcum, (size_t) chunks * CH * HV * sizeof(float)) != cudaSuccess) {
            int dev = 0;
            cudaGetDevice(&dev);
            size_t fb = 0, tb = 0;
            cudaMemGetInfo(&fb, &tb);
            std::fprintf(stderr, "gdn_chunk: workspace alloc failed (%zu chunks, device %d, %.1f of %.1f MiB free): %s\n",
                         chunks, dev, (double) fb / 1048576.0, (double) tb / 1048576.0,
                         cudaGetErrorString(cudaGetLastError()));
            std::exit(1);
        }
    }
    ~Ws() {
        // A prompt's stage threads are created per request (std::async in the layer split): without this, each
        // request would leak its device's ~50 MB workspace until the card runs out.
        if (A) cudaFree(A);
        if (P) cudaFree(P);
        if (gcum) cudaFree(gcum);
    }
};

Ws& ws() {
    static thread_local Ws w[8];   // [device]: the cards a layer split can run on in one process
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        cudaGetLastError();
        dev = 0;
    }
    if (dev < 0 || dev >= 8) dev = 0;
    return w[dev];
}

int gdn_chunk_cc_of(int dev) {
    // Per DEVICE: a layer split runs on more than one card, and the capability is a card property - a
    // process-global cache would dispatch a second card by the first card's capability.
    static int cc[64] = {};
    if (dev < 0 || dev >= 64) return 0;
    if (cc[dev] == 0) {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) {
            cudaGetLastError();
            return 0;
        }
        cc[dev] = p.major * 10 + p.minor;
    }
    return cc[dev];
}

int gdn_chunk_cc() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return gdn_chunk_cc_of(dev);
}

}  // namespace

bool gdn_chunk_available() {
#if defined(__HIPCC__) || !defined(STRATA_EXPERIMENTAL_SM60)
    return false;  // the chunked path is compiled and dispatched only in the Volta experimental build
#else
    return gdn_chunk_cc() == 70;
#endif
}

void gdn_chunk_recurrence(float* state, const float* h, const float* gate, const float* beta, float* oc,
                          int64_t T, void* stream) {
#if defined(__HIPCC__)
    (void) state; (void) h; (void) gate; (void) beta; (void) oc; (void) T; (void) stream;
    std::fprintf(stderr, "gdn_chunk_recurrence: not built for HIP\n");
    std::exit(1);
#else
    if (T <= 0) return;
    const int64_t T64 = T - T % CH;
    if (T64 <= 0) return;
    cudaStream_t cs = (cudaStream_t) stream;
    // the wmma path is the default; STRATA_GDN_CHUNK=2 selects the FP32-only FMA scan (the parity test's
    // oracle - same math, no FP16 staging) and anything below sm_70 has no wmma to use.  Read per call (the
    // tests flip it between launches).
    const char* forced_v = std::getenv("STRATA_GDN_CHUNK");
    const int forced = forced_v && *forced_v ? std::atoi(forced_v) : -1;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        cudaGetLastError();
        dev = 0;
    }
#if !defined(__HIPCC__)
    const bool wmma_ok = gdn_chunk_cc_of(dev) == 70;
#else
    const bool wmma_ok = false;
#endif
    // The shared-memory opt-in is a property of the DEVICE's context, not the process: a layer split runs
    // this call on more than one card, and each card needs its own cudaFuncSetAttribute before that card's
    // past-48KB launches (a process-global std::call_once left the second card at the 48 KB default, and its
    // first chunked launch failed with "invalid argument").  Per device, once; every attempt clears the
    // runtime error so a refused opt-in cannot surface in a later check() as a spurious failure.
    static std::mutex attr_mu;
    static bool attr_done[64] = {};
    static bool wmma_attr[64] = {};
    {
        std::lock_guard<std::mutex> lk(attr_mu);
        if (dev >= 0 && dev < 64 && !attr_done[dev]) {
            const cudaError_t ek = cudaFuncSetAttribute(gdn_chunk_kkt_kernel,
                                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                        2 * CH * (S + 1) * (int) sizeof(float));
            const cudaError_t ew = cudaFuncSetAttribute(gdn_chunk_wmma_kernel,
                                                        cudaFuncAttributeMaxDynamicSharedMemorySize, WMMA_SMEM);
            cudaGetLastError();
            (void) ek;   // a refused opt-in leaves the FMA path's own launch to report it
            wmma_attr[dev] = ew == cudaSuccess;
            attr_done[dev] = true;
        }
    }
    const bool use_wmma = wmma_ok && forced != 2 && dev >= 0 && dev < 64 && wmma_attr[dev];
    Ws& w = ws();
    const size_t chunks_total = (size_t) (T64 / CH);
    w.ensure(chunks_total < SEG ? (chunks_total ? chunks_total : 1) : SEG);
    // STRATA_GDN_CHUNK_PROF=1: per-stage milliseconds over the whole call (a dev tool, not a hot path)
    static const bool prof = std::getenv("STRATA_GDN_CHUNK_PROF") != nullptr;
    cudaEvent_t pe0{}, pe1{};
    float ms[3] = {0, 0, 0};
    if (prof) cudaEventCreate(&pe0), cudaEventCreate(&pe1);
    for (int64_t t0 = 0; t0 < T64; t0 += (int64_t) SEG * CH) {
        const int64_t left = (T64 - t0) / CH;
        const int nch = (int) (left < SEG ? left : SEG);
        const int ntok = nch * CH;
        if (prof) cudaEventRecord(pe0, cs);
        gdn_chunk_cumsum_kernel<<<dim3((unsigned) nch, HV), CH, 0, cs>>>(w.gcum, gate + t0 * HV, ntok);
        check("cumsum");
        if (prof) {
            cudaEventRecord(pe1, cs);
            cudaEventSynchronize(pe1);
            cudaEventElapsedTime(&ms[0], pe0, pe1);
            cudaEventRecord(pe0, cs);
        }
        if (use_wmma) {
            gdn_chunk_wmma_kernel<<<dim3(NSE, HV), 128, WMMA_SMEM, cs>>>(
                state, oc + t0 * HV * S, h + t0 * C, beta + t0 * HV, w.gcum, nch);
            check("wmma");
        } else {
            gdn_chunk_kkt_kernel<<<dim3((unsigned) nch, HV), 128, 2 * CH * (S + 1) * sizeof(float), cs>>>(
                w.A, w.P, h + t0 * C, beta + t0 * HV, w.gcum, nch);
            check("kkt");
        }
        if (prof) {
            cudaEventRecord(pe1, cs);
            cudaEventSynchronize(pe1);
            cudaEventElapsedTime(&ms[1], pe0, pe1);
            cudaEventRecord(pe0, cs);
        }
        if (!use_wmma) {
            gdn_chunk_fwd_kernel<<<dim3(NSE, HV), 128, 0, cs>>>(state, oc + t0 * HV * S, h + t0 * C, beta + t0 * HV,
                                                                w.gcum, w.A, w.P, nch);
            check("fwd");
        }
        if (prof) {
            cudaEventRecord(pe1, cs);
            cudaEventSynchronize(pe1);
            cudaEventElapsedTime(&ms[2], pe0, pe1);
        }
    }
    if (prof) {
        std::printf("gdn_chunk prof T=%lld: cumsum %.3fms stage2 %.3fms stage3 %.3fms\n", (long long) T, ms[0],
                    ms[1], ms[2]);
        cudaEventDestroy(pe0);
        cudaEventDestroy(pe1);
    }
#endif
}

}  // namespace strata::prefill
