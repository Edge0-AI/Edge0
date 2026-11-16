// tools/dsv41_lab/tqk_gemv.cuh - CUDA GEMV / small-batch GEMM kernels for the trellis expert formats at DeepSeek shapes
// (TQK4X/5X = types 225/224 of f-dsv41-enc, TQK6/7 = ggml 145/146), weights stay in the file layout:
//   row = nblk blocks of {fp16 d; 4K stream bytes}, block = 128 weights = 32 steps of 4; step t's 15-bit state is the
//   stream window at bit (31-t)*K (LSB first, circular); w = d * codebook(state); codebook = int16 table / 8192 (the
//   CPU decoder's table: tqenc::point_table()).
// Plain C++ in the kernels (no inline PTX, no tensor-core intrinsics) so the HIP port is a rename.
#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tqlab {

constexpr uint32_t kMask = 0x7fffu;
constexpr uint32_t kHash = 0x9e3779b1u;

template <int K> struct Fmt {
    static constexpr int kBits = K;
    static constexpr int kBlockBytes = 2 + 4 * K;
    static constexpr int kWords = 2 * K;   // u16 words of the stream
};

// One job = one matrix (rows x in) applied to NB columns of activations (F32, [col][in]) -> Y[col][rows].
struct Job {
    const uint8_t* w;       // rows * row_bytes, row_bytes = in / 128 * block bytes (2-byte aligned)
    const float* x;         // [NB][ldx]
    float* y;               // [NB][ldy]
    int rows;
    int ldx, ldy;
    const uint64_t* sc;     // v3 (TQL layout) only: the scales, see tqk_repack.hpp
    int in = 0;             // v3: weights per row (x is then the prepared B-fragment array, see prep_b)
};

// the four unscaled weights of the step with state s from the float2 point table
__device__ __forceinline__ void step_w(const float2* __restrict__ lut, uint32_t s, float2& a, float2& b) {
    const uint32_t x = (s & kMask) * kHash;
    a = lut[x >> 21];
    b = lut[(x >> 10) & 2047u];
}

// the 4 states of steps 4l..4l+3 of a block whose stream starts at qs (2-byte aligned)
template <int K>
__device__ __forceinline__ void states4(const uint16_t* __restrict__ qs, int l, uint32_t s[4]) {
    constexpr int nwords = 2 * K;
    const int off3 = (28 - 4 * l) * K, wb = off3 >> 4, r = off3 & 15;
    const int i1 = wb + 1 >= nwords ? wb + 1 - nwords : wb + 1;
    const int i2 = wb + 2 >= nwords ? wb + 2 - nwords : wb + 2;
    const int i3 = wb + 3 >= nwords ? wb + 3 - nwords : wb + 3;
    const uint64_t v = (uint64_t) qs[wb] | ((uint64_t) qs[i1] << 16) | ((uint64_t) qs[i2] << 32) | ((uint64_t) qs[i3] << 48);
#pragma unroll
    for (int i = 0; i < 4; ++i) s[i] = (uint32_t) (v >> (r + (3 - i) * K)) & kMask;
}

// ------------------------------------------------------------------------------------------------------------
// v1: warp per RPW rows, lane = 16-weight unit (4 steps), 32 units = 4 blocks per iteration.  F32 activations from
// global/L1, point table in shared memory as float2.
template <int K, int NB, int RPW>
__global__ void __launch_bounds__(256) gemv_v1(const Job* __restrict__ jobs, const float2* __restrict__ lut_g, int in) {
    __shared__ float2 lut[2048];
    for (int i = threadIdx.x; i < 2048; i += blockDim.x) lut[i] = lut_g[i];
    __syncthreads();
    const Job job = jobs[blockIdx.y];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * (blockDim.x >> 5) + warp) * RPW;
    if (row0 >= job.rows) return;
    constexpr int bsz = Fmt<K>::kBlockBytes;
    const int nblk = in / 128;
    const size_t row_bytes = (size_t) nblk * bsz;
    float acc[RPW][NB];
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
        for (int c = 0; c < NB; ++c) acc[r][c] = 0.f;
    const int l = lane & 7, bsub = lane >> 3;   // unit lane in block, block within the group of 4
    for (int b0 = 0; b0 < nblk; b0 += 4) {
        const int b = b0 + bsub;
        if (b >= nblk) break;
        float4 xv[NB][4];
#pragma unroll
        for (int c = 0; c < NB; ++c) {
            const float4* xp = (const float4*) (job.x + (size_t) c * job.ldx + b * 128 + 16 * l);
#pragma unroll
            for (int i = 0; i < 4; ++i) xv[c][i] = __ldg(xp + i);
        }
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const int row = row0 + r;
            if (row >= job.rows) continue;
            const uint8_t* blk = job.w + (size_t) row * row_bytes + (size_t) b * bsz;
            const float d = __half2float(__ushort_as_half(*(const uint16_t*) blk));
            uint32_t s[4];
            states4<K>((const uint16_t*) (blk + 2), l, s);
            float u[NB];
#pragma unroll
            for (int c = 0; c < NB; ++c) u[c] = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float2 a, bb;
                step_w(lut, s[i], a, bb);
#pragma unroll
                for (int c = 0; c < NB; ++c)
                    u[c] += a.x * xv[c][i].x + a.y * xv[c][i].y + bb.x * xv[c][i].z + bb.y * xv[c][i].w;
            }
#pragma unroll
            for (int c = 0; c < NB; ++c) acc[r][c] += d * u[c];
        }
    }
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
        for (int c = 0; c < NB; ++c) {
            float v = acc[r][c];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0 && row0 + r < job.rows) job.y[(size_t) c * job.ldy + row0 + r] = v;
        }
}

// plain dequantisation of whole rows: out[row][in] = d * codebook (F32), one thread per 4-weight step
template <int K>
__global__ void dequant_f32(const uint8_t* __restrict__ w, const float2* __restrict__ lut_g, float* __restrict__ out,
                            int64_t nblocks) {
    __shared__ float2 lut[2048];
    for (int i = threadIdx.x; i < 2048; i += blockDim.x) lut[i] = lut_g[i];
    __syncthreads();
    const int64_t gid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // unit = 4 steps
    const int64_t b = gid >> 3;
    if (b >= nblocks) return;
    const int l = (int) (gid & 7);
    const uint8_t* blk = w + b * Fmt<K>::kBlockBytes;
    const float d = __half2float(__ushort_as_half(*(const uint16_t*) blk));
    uint32_t s[4];
    states4<K>((const uint16_t*) (blk + 2), l, s);
    float4* o = (float4*) (out + b * 128 + 16 * l);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 a, bb;
        step_w(lut, s[i], a, bb);
        o[i] = make_float4(d * a.x, d * a.y, d * bb.x, d * bb.y);
    }
}


// ------------------------------------------------------------------------------------------------------------
// v2: tensor cores.  The point table is exactly representable in fp16 (int16 table / 8192, |n| < 1024 or <= 11
// significant bits), so a trellis step's four weights are two half2 words straight out of an 8 KB u32 table: no
// conversion, and they are the A fragment of mma.m16n8k16 as they are.  Thread (g = lane/4, q = lane%4) of a 16-row
// tile holds rows g and g+8; mma m of a 128-weight block takes step t = 8q + m of both rows (k slots 2q, 2q+1, 2q+8,
// 2q+9 = weights 4t..4t+3) against the activations x[col g][4t..4t+3] (B fragment, columns 0..NB-1 of the 8).  The k
// order inside an mma is free as long as A and B agree.  The activations are split x = hi + lo (two fp16) so the
// product is accurate to ~2^-22 relative (EXACT = 1; EXACT = 0 uses hi only, fp16-rounded activations).  The block
// scale d is applied per row after the block's 8 (16) mma.  D lands in rows g / g+8, columns 2q, 2q+1: stored directly.
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t h2bits(__half2 h) { return *(uint32_t*) &h; }

// the 4 states of steps 4l..4l+3 -> the 8 table entries needed: es[i][0..1]
template <int K, int NB, int T, int EXACT>
__global__ void __launch_bounds__(128) gemv_v2(const Job* __restrict__ jobs, const uint32_t* __restrict__ lut_g, int in) {
    __shared__ uint32_t lut[2048];
    for (int i = threadIdx.x; i < 2048; i += blockDim.x) lut[i] = lut_g[i];
    __syncthreads();
    const Job job = jobs[blockIdx.y];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
    const int row0 = (blockIdx.x * (blockDim.x >> 5) + warp) * 16 * T;
    if (row0 >= job.rows) return;
    constexpr int bsz = Fmt<K>::kBlockBytes;
    const int nblk = in / 128;
    const size_t row_bytes = (size_t) nblk * bsz;
    float tot[T][4];
#pragma unroll
    for (int t = 0; t < T; ++t)
#pragma unroll
        for (int i = 0; i < 4; ++i) tot[t][i] = 0.f;
    for (int b = 0; b < nblk; ++b) {
        uint32_t bh[8][2], bl[8][2];
        if (g < NB) {
            const float4* xp = (const float4*) (job.x + (size_t) g * job.ldx + b * 128 + 32 * q);
#pragma unroll
            for (int m = 0; m < 8; ++m) {
                const float4 v = __ldg(xp + m);
                const __half2 h0 = __floats2half2_rn(v.x, v.y), h1 = __floats2half2_rn(v.z, v.w);
                bh[m][0] = h2bits(h0); bh[m][1] = h2bits(h1);
                if (EXACT) {
                    bl[m][0] = h2bits(__floats2half2_rn(v.x - __low2float(h0), v.y - __high2float(h0)));
                    bl[m][1] = h2bits(__floats2half2_rn(v.z - __low2float(h1), v.w - __high2float(h1)));
                }
            }
        } else {
#pragma unroll
            for (int m = 0; m < 8; ++m) { bh[m][0] = bh[m][1] = bl[m][0] = bl[m][1] = 0u; }
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
            uint32_t ea[2][8], eb[2][8];   // [row half][m]
            float dd[2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = row0 + t * 16 + g + 8 * h;
                const uint8_t* blk = job.w + (size_t) row * row_bytes + (size_t) b * bsz;
                dd[h] = __half2float(__ushort_as_half(*(const uint16_t*) blk));
#pragma unroll
                for (int u = 0; u < 2; ++u) {
                    uint32_t s[4];
                    states4<K>((const uint16_t*) (blk + 2), 2 * q + u, s);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const uint32_t x = s[i] * kHash;
                        ea[h][4 * u + i] = lut[x >> 21];
                        eb[h][4 * u + i] = lut[(x >> 10) & 2047u];
                    }
                }
            }
            float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int m = 0; m < 8; ++m) {
                mma16816(acc, ea[0][m], ea[1][m], eb[0][m], eb[1][m], bh[m][0], bh[m][1]);
                if (EXACT) mma16816(acc, ea[0][m], ea[1][m], eb[0][m], eb[1][m], bl[m][0], bl[m][1]);
            }
            tot[t][0] += dd[0] * acc[0]; tot[t][1] += dd[0] * acc[1];
            tot[t][2] += dd[1] * acc[2]; tot[t][3] += dd[1] * acc[3];
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int col = 2 * q + (i & 1), row = row0 + t * 16 + g + 8 * (i >> 1);
            if (col < NB) job.y[(size_t) col * job.ldy + row] = tot[t][i];
        }
}

}  // namespace tqlab
