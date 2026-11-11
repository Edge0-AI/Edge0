// tools/dsv41_lab/tqk_gemv_c.cuh - v3 with a COMPUTED codebook instead of the 2048-entry table (LUT-free): the step's four weights
// come from 2 or 4 LCG words, x_i = s * A_i + B_i, w = (x_i & 0x8FFF8FFF) ^ 0x3B603B60 read as two fp16 (QTIP 3INST's trick;
// computed_codes.hpp).  CODE 0 "3single": the two words are the A fragment (a0, a2) as they are: 2 IMAD + 2 LOP3 a step.
// CODE 1 "3sum": a0 = HADD2(w0, w1), a2 = HADD2(w2, w3): 4 IMAD + 4 LOP3 + 2 HADD2.  No shared-memory table, no __syncthreads.
// Everything else (TQL layout, prepared B fragments, mma) is gemv_v3.
#pragma once
#include "tqk_gemv3.cuh"

namespace tqlab {

__device__ __forceinline__ uint32_t cword(uint32_t s, int i) {
    constexpr uint32_t A[4] = {89226354u, 0x9E3779B1u, 0x85EBCA6Bu, 0xC2B2AE35u};
    constexpr uint32_t B[4] = {64248484u, 0x7F4A7C15u, 0x165667B1u, 0x27D4EB2Fu};
    return ((s * A[i] + B[i]) & 0x8FFF8FFFu) ^ 0x3B603B60u;
}
__device__ __forceinline__ uint32_t hadd2_u(uint32_t a, uint32_t b) {
    __half2 x = *(__half2*) &a, y = *(__half2*) &b;
    const __half2 r = __hadd2(x, y);
    return *(const uint32_t*) &r;
}

template <int K, int NB, int T, int EXACT, int CODE>
__global__ void __launch_bounds__(128) gemv_v3c(const Job* __restrict__ jobs, const uint32_t* __restrict__ lut_unused, int in_unused) {
    const Job job = jobs[blockIdx.y];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
    const int tile0 = (blockIdx.x * (blockDim.x >> 5) + warp) * T;
    if (tile0 * 16 >= job.rows) return;
    const int nblk = job.in / 128, NP = nblk / 2;
    float tot[T][4];
#pragma unroll
    for (int t = 0; t < T; ++t)
#pragma unroll
        for (int i = 0; i < 4; ++i) tot[t][i] = 0.f;
    const int src_lane = 4 * g + ((q + 3) & 3);
    for (int P = 0; P < NP; ++P) {
        uint32_t bh[2][8][2], bl[2][8][2];
#pragma unroll
        for (int bk = 0; bk < 2; ++bk) {
            if (g < NB) {
                const uint4* xp = ((const uint4*) job.x) + (((size_t) (2 * P + bk) * 8) * NB + g) * 4 + q;
#pragma unroll
                for (int m = 0; m < 8; ++m) {
                    const uint4 v = __ldg(xp + (size_t) m * NB * 4);
                    bh[bk][m][0] = v.x;
                    bh[bk][m][1] = v.y;
                    if (EXACT) { bl[bk][m][0] = v.z; bl[bk][m][1] = v.w; }
                }
            } else {
#pragma unroll
                for (int m = 0; m < 8; ++m) { bh[bk][m][0] = bh[bk][m][1] = bl[bk][m][0] = bl[bk][m][1] = 0u; }
            }
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const size_t base = (size_t) (tile0 + t) * NP + P;
            uint32_t r[K];
#pragma unroll
            for (int j = 0; j < K; ++j) r[j] = __ldcs(((const uint32_t*) job.w) + (base * K + j) * 32 + lane);
            const uint64_t dq = __ldcs(job.sc + base * 8 + g);
            uint64_t win[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int a = (8 * K * i) >> 5, sh = (8 * K * i) & 31;
                const uint32_t w0 = r[a];
                const uint32_t w1 = a + 1 < K ? r[a + 1 < K ? a + 1 : 0] : 0u;
                const uint32_t w2 = a + 2 < K ? r[a + 2 < K ? a + 2 : 0] : 0u;
                const uint32_t lo = __funnelshift_r(w0, w1, sh), hi = __funnelshift_r(w1, w2, sh);
                const uint64_t ch = (((uint64_t) hi << 32) | lo) & ((1ull << (8 * K)) - 1ull);
                const uint32_t fw = __shfl_sync(0xffffffffu, lo, src_lane);
                win[i] = ch | ((uint64_t) (fw & ((1u << (15 - K)) - 1u)) << (8 * K));
            }
#pragma unroll
            for (int bk = 0; bk < 2; ++bk) {
                float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int m = 0; m < 8; ++m) {
                    uint32_t a0, a1, a2, a3;   // (row g, slots 2q..), (row g+8, ..), (row g, slots 2q+8..), (row g+8, ..)
                    {
                        const uint32_t s = (uint32_t) (win[2 * bk + 0] >> ((7 - m) * K)) & kMask;
                        if (CODE == 0) { a0 = cword(s, 0); a2 = cword(s, 1); }
                        else { a0 = hadd2_u(cword(s, 0), cword(s, 1)); a2 = hadd2_u(cword(s, 2), cword(s, 3)); }
                    }
                    {
                        const uint32_t s = (uint32_t) (win[2 * bk + 1] >> ((7 - m) * K)) & kMask;
                        if (CODE == 0) { a1 = cword(s, 0); a3 = cword(s, 1); }
                        else { a1 = hadd2_u(cword(s, 0), cword(s, 1)); a3 = hadd2_u(cword(s, 2), cword(s, 3)); }
                    }
                    mma16816(acc, a0, a1, a2, a3, bh[bk][m][0], bh[bk][m][1]);
                    if (EXACT) mma16816(acc, a0, a1, a2, a3, bl[bk][m][0], bl[bk][m][1]);
                }
                const float d0 = __half2float(__ushort_as_half((unsigned short) (dq >> (16 * (2 * bk + 0)))));
                const float d1 = __half2float(__ushort_as_half((unsigned short) (dq >> (16 * (2 * bk + 1)))));
                tot[t][0] += d0 * acc[0];
                tot[t][1] += d0 * acc[1];
                tot[t][2] += d1 * acc[2];
                tot[t][3] += d1 * acc[3];
            }
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int col = 2 * q + (i & 1), row = (tile0 + t) * 16 + g + 8 * (i >> 1);
            if (col < NB) job.y[(size_t) col * job.ldy + row] = tot[t][i];
        }
}

}  // namespace tqlab
