// tools/dsv41_lab/tqk_gemv3.cuh - v3: tensor-core GEMV over the TQL layout (tqk_repack.hpp); include after tqk_gemv.cuh.
// Every lane's stream bits and the scales arrive in K + 1 fully coalesced loads per (16-row tile, block pair) - K words and one
// u64 shared by the quad - instead of ~18 u16 loads over 8 cache lines.  The 15-bit states of a lane's 8 steps need 15 - K
// bits from lane q-1's chunk (circular): one shuffle per chunk.
#pragma once
#include "tqk_gemv.cuh"

namespace tqlab {

template <int K, int NB, int T, int EXACT>
__global__ void __launch_bounds__(128) gemv_v3(const Job* __restrict__ jobs, const uint32_t* __restrict__ lut_g, int in) {
    __shared__ uint32_t lut[2048];
    for (int i = threadIdx.x; i < 2048; i += blockDim.x) lut[i] = lut_g[i];
    __syncthreads();
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
            // windows of the 4 chunks: own 8K bits + the low 15-K bits of lane q-1's chunk
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
            uint32_t ea[2][2][8], eb[2][2][8];   // [bk][h][m]
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int m = 0; m < 8; ++m) {
                    const uint32_t s = (uint32_t) (win[i] >> ((7 - m) * K)) & kMask;
                    const uint32_t x = s * kHash;
                    ea[i >> 1][i & 1][m] = lut[x >> 21];
                    eb[i >> 1][i & 1][m] = lut[(x >> 10) & 2047u];
                }
#pragma unroll
            for (int bk = 0; bk < 2; ++bk) {
                float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int m = 0; m < 8; ++m) {
                    mma16816(acc, ea[bk][0][m], ea[bk][1][m], eb[bk][0][m], eb[bk][1][m], bh[bk][m][0], bh[bk][m][1]);
                    if (EXACT) mma16816(acc, ea[bk][0][m], ea[bk][1][m], eb[bk][0][m], eb[bk][1][m], bl[bk][m][0], bl[bk][m][1]);
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

namespace tqlab {
// activations -> B fragments, once per call: out[(((blk * 8 + m) * nb + c) * 4 + q)] = {hi(x0,x1), hi(x2,x3), lo(x0,x1), lo(x2,x3)}
// for x[c][blk * 128 + 32 q + 4 m .. +3]  (hi = fp16(x), lo = fp16(x - hi): hi + lo is x to ~2^-22).
__global__ void prep_b(const float* __restrict__ x, int ldx, int nb, int in, uint4* __restrict__ out) {
    const int nblk = in / 128;
    const size_t gid = (size_t) blockIdx.x * blockDim.x + threadIdx.x, n = (size_t) nb * nblk * 32;
    if (gid >= n) return;
    const int m = gid & 7, q = (gid >> 3) & 3;
    const size_t cb = gid >> 5;   // c * nblk + blk
    const int c = (int) (cb / nblk), blk = (int) (cb % nblk);
    const float4 v = *(const float4*) (x + (size_t) c * ldx + blk * 128 + 32 * q + 4 * m);
    const __half2 h0 = __floats2half2_rn(v.x, v.y), h1 = __floats2half2_rn(v.z, v.w);
    out[(((size_t) blk * 8 + m) * nb + c) * 4 + q] = make_uint4(h2bits(h0), h2bits(h1), h2bits(__floats2half2_rn(v.x - __low2float(h0), v.y - __high2float(h0))),
                          h2bits(__floats2half2_rn(v.z - __low2float(h1), v.w - __high2float(h1))));
}
}  // namespace tqlab
