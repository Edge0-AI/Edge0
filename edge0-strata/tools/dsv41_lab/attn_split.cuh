// tools/dsv41_lab/attn_split.cuh - split-KV ("flash-decoding") version of DeepSeek's sparse attention for a few queries
// (decode T = 1, DSpark verify T = 2..5): 64 heads x d 512, MQA (K = V), window rows (BF16) + selected compressed rows (kKvFp4:
// E2M1 codes + E4M3 scales per 16) + a learned sink logit per head.  Same inputs and outputs as strata::ds4::sparse_attn
// (AttnRows gen 0, ckv kKvFp4); the shipped kernel runs one 256-thread block per (head, query) = 64 blocks on 48 SMs and takes
// ~125 us per query; this one splits the keys over R-row chunks and groups G heads per block, which share the decoded rows.
//   phase 1: block (head group, chunk, query): decode R rows to BF16 in shared memory (exact: window rows are BF16, packed
//            rows have <= 7 significant bits), scores for G heads (warp per row, lanes 16 values, shuffle reduction), local
//            softmax, partial output (thread = 2 output values) -> scratch {m, l, o[512]} per (query, head, chunk)
//   phase 2: block (head, query): merge the chunks and the sink.
// Plain C++ (no PTX), shuffles and shared memory only: the HIP port is a rename.
#pragma once
#include "../../src/dsv41/ds4_kernels.cuh"
#include "../../src/dsv41/ds4_numerics.cuh"
#include <cuda_fp16.h>

namespace strata::ds4lab {

using strata::ds4::AttnRows;

__device__ __forceinline__ float bf2f_(uint16_t b) { return __uint_as_float((uint32_t) b << 16); }
__device__ __forceinline__ uint16_t f2bf_exact(float f) { return (uint16_t) (__float_as_uint(f) >> 16); }   // exact values only
__device__ __forceinline__ float e2m1_x2p14(uint32_t c) {
    return __half2float(__ushort_as_half((unsigned short) (((c & 8u) << 12) | ((c & 7u) << 9))));
}
__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float wmax(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// d = 512 only (a thread owns 2 output values, a lane 16 values of a row)
template <int G, int R>
__global__ void __launch_bounds__(256) attn_part(const float* __restrict__ q, const uint16_t* __restrict__ win,
                                                 const uint8_t* __restrict__ ckv, size_t ckv_row_bytes, const int* __restrict__ sel,
                                                 int sel_stride, AttnRows ar, int heads, float scale, float* __restrict__ part,
                                                 float* __restrict__ ml, int nsplit) {
    constexpr int D = 512;
    __shared__ __align__(16) uint16_t rows[R][D];
    __shared__ float sc[G][R];
    __shared__ float4 pr[R][(G + 3) / 4];
    __shared__ float mloc[G], lloc[G];
    const int hg = blockIdx.x, chunk = blockIdx.y, t = blockIdx.z;
    const int n_win = ar.nwin[t], n_sel = ar.nsel[t], total = n_win + n_sel;
    const int j0 = chunk * R;
    if (j0 >= total) return;
    const int nr = min(R, total - j0);
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    // q of this block's heads: lane's 16 values per head in registers (issued first: its latency overlaps the row loads)
    float qv[G][16];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const float4* qp = (const float4*) (q + ((size_t) t * heads + hg * G + g) * D + lane * 16);
#pragma unroll
        for (int i = 0; i < 4; ++i) { const float4 v = qp[i]; qv[g][4 * i] = v.x; qv[g][4 * i + 1] = v.y; qv[g][4 * i + 2] = v.z; qv[g][4 * i + 3] = v.w; }
    }
    // ---- stage rows: warp per row, lane = 16 values; all of a warp's row loads are issued before any is converted
    constexpr int RPW = (R + 7) / 8;
    uint4 wa[RPW][2];
    uint2 cwv[RPW];
    uint32_t sbv[RPW];
    int rowoff[RPW];
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
        const int jj = wp + 8 * i, j = j0 + jj;
        rowoff[i] = -1;
        if (jj < nr) {
            if (j < n_win) {
                const uint4* src = (const uint4*) (win + (size_t) (ar.win0[t] + j) * D + lane * 16);
                wa[i][0] = src[0];
                wa[i][1] = src[1];
            } else {
                rowoff[i] = sel[(size_t) t * sel_stride + (j - n_win)];
            }
        }
    }
#pragma unroll
    for (int i = 0; i < RPW; ++i)
        if (rowoff[i] >= 0) {
            const uint8_t* r = ckv + (size_t) rowoff[i] * ckv_row_bytes;
            cwv[i] = *(const uint2*) (r + lane * 8);
            sbv[i] = r[D / 2 + lane];
        }
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
        const int jj = wp + 8 * i;
        if (jj >= nr) continue;
        uint4* dst = (uint4*) &rows[jj][lane * 16];
        if (rowoff[i] < 0) {
            dst[0] = wa[i][0];
            dst[1] = wa[i][1];
        } else {
            const float s = strata::ds4::e4m3_value((uint8_t) sbv[i]) * 16384.0f;   // (the nibbles decode to value * 2^-14)
            uint32_t o[8];
#pragma unroll
            for (int w = 0; w < 2; ++w) {
                const uint32_t c = w ? cwv[i].y : cwv[i].x;
#pragma unroll
                for (int x = 0; x < 4; ++x) {
                    const uint32_t b = c >> (8 * x);
                    const uint16_t lo = f2bf_exact(e2m1_x2p14(b) * s), hi = f2bf_exact(e2m1_x2p14(b >> 4) * s);
                    o[4 * w + x] = (uint32_t) lo | ((uint32_t) hi << 16);
                }
            }
            dst[0] = make_uint4(o[0], o[1], o[2], o[3]);
            dst[1] = make_uint4(o[4], o[5], o[6], o[7]);
        }
    }
    __syncthreads();
    // ---- scores
    for (int jj = wp; jj < nr; jj += 8) {
        const uint4 a = *(const uint4*) &rows[jj][lane * 16], b = *(const uint4*) &rows[jj][lane * 16 + 8];
        float kv[16];
        const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) { kv[2 * i] = __uint_as_float(w[i] << 16); kv[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u); }
#pragma unroll
        for (int g = 0; g < G; ++g) {
            float s = 0.f;
#pragma unroll
            for (int i = 0; i < 16; ++i) s += qv[g][i] * kv[i];
            s = wsum(s);
            if (lane == 0) sc[g][jj] = s * scale;
        }
    }
    __syncthreads();
    // ---- local softmax: warp g handles head g
    if (wp < G) {
        float m = -INFINITY;
        for (int j = lane; j < nr; j += 32) m = fmaxf(m, sc[wp][j]);
        m = wmax(m);
        float l = 0.f;
        for (int j = lane; j < nr; j += 32) { const float p = __expf(sc[wp][j] - m); sc[wp][j] = p; l += p; }
        l = wsum(l);
        if (lane == 0) { mloc[wp] = m; lloc[wp] = l; }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < nr * G; i += 256) { const int j = i / G, g = i % G; ((float*) &pr[j][0])[g] = sc[g][j]; }
    __syncthreads();
    // ---- partial output: thread = values 2 tid, 2 tid + 1 of every head
    float o0[G], o1[G];
#pragma unroll
    for (int g = 0; g < G; ++g) o0[g] = o1[g] = 0.f;
    const int c = threadIdx.x * 2;
    for (int j = 0; j < nr; ++j) {
        const uint32_t kw = *(const uint32_t*) &rows[j][c];
        const float k0 = __uint_as_float(kw << 16), k1 = __uint_as_float(kw & 0xffff0000u);
        const float* p = (const float*) &pr[j][0];
#pragma unroll
        for (int g = 0; g < G; ++g) { o0[g] += p[g] * k0; o1[g] += p[g] * k1; }
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const size_t bi = ((size_t) t * heads + hg * G + g) * nsplit + chunk;
        *(float2*) (part + bi * D + c) = make_float2(o0[g], o1[g]);
        if (threadIdx.x == 0) { ml[bi * 2] = mloc[g]; ml[bi * 2 + 1] = lloc[g]; }
    }
}

__global__ void __launch_bounds__(256) attn_merge(const float* __restrict__ part, const float* __restrict__ ml, const float* __restrict__ sink,
                                                  int heads, int nsplit, int R, AttnRows ar, float* __restrict__ out) {
    constexpr int D = 512;
    const int h = blockIdx.x, t = blockIdx.y;
    const int total = ar.nwin[t] + ar.nsel[t];
    const int ns = (total + R - 1) / R;
    const size_t base = ((size_t) t * heads + h) * nsplit;
    float M = sink[h];
    for (int s = 0; s < ns; ++s) M = fmaxf(M, ml[(base + s) * 2]);
    float den = __expf(sink[h] - M);
    for (int s = 0; s < ns; ++s) den += ml[(base + s) * 2 + 1] * __expf(ml[(base + s) * 2] - M);
    const int c = threadIdx.x * 2;
    float a0 = 0.f, a1 = 0.f;
    for (int s = 0; s < ns; ++s) {
        const float w = __expf(ml[(base + s) * 2] - M);
        const float2 v = *(const float2*) (part + (base + s) * D + c);
        a0 += w * v.x;
        a1 += w * v.y;
    }
    *(float2*) (out + ((size_t) t * heads + h) * D + c) = make_float2(a0 / den, a1 / den);
}

struct AttnSplitScratch {
    float* part = nullptr;   // [T][heads][nsplit][512]
    float* ml = nullptr;     // [T][heads][nsplit][2]
    int nsplit = 0, max_t = 0;
};

// total_max = the largest nwin + nsel of the T queries
template <int G, int R>
void sparse_attn_split(const float* q, const uint16_t* win, const uint8_t* ckv, size_t ckv_row_bytes, const int* sel, int sel_stride,
                       const AttnRows& ar, const float* sink, int heads, int T, float scale, float* out, const AttnSplitScratch& sc,
                       int total_max, cudaStream_t s) {
    const int nsplit = (total_max + R - 1) / R;
    attn_part<G, R><<<dim3(heads / G, nsplit, T), 256, 0, s>>>(q, win, ckv, ckv_row_bytes, sel, sel_stride, ar, heads, scale, sc.part, sc.ml, sc.nsplit);
    attn_merge<<<dim3(heads, T), 256, 0, s>>>(sc.part, sc.ml, sink, heads, sc.nsplit, R, ar, out);
}

}  // namespace strata::ds4lab
