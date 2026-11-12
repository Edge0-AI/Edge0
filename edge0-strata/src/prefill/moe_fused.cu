// src/prefill/moe_fused.cu - see include/strata/prefill/moe_fused.hpp (#136: the Q2_0 pack's prompt experts on
// Strata's own int8 tensor-core kernels, with the grouping on the GPU).
//
// The arithmetic.  A Q2_0 block is 64 weights w = d_w * (q - 1), q in 0..3, four codes per byte, weight j of a block
// in byte j / 4 at bits 2 * (j % 4).  An activation block of 32 values is x = d_x * a with a = round(x / d_x) in
// -127..127.  Over one 32-value half-block
//     sum w x = d_w * d_x * (sum q a - sum a),
// so the int8 product takes the codes as they are (0..3 is a valid s8) and the offset is the activations' own sum,
// stored with the block.  The dot product (|sum q a| <= 32 * 3 * 127 < 2^22) becomes a float exactly with one
// integer add: as_float(0x4B400000 + dot) = 1.5 * 2^23 + dot, and the block stores c = -(1.5 * 2^23 + sum a), so
// as_float(0x4B400000 + dot) + c = dot - sum a with no rounding.  Per output and 64-block that is
//     acc += d_w * (d_x0 * t0 + d_x1 * t1)        (t = dot - sum a of each half)
// - two integer adds, two float adds and three multiply-adds.
//
// The K order.  mma.sync m16n8k32 takes four consecutive k of a row (A) or column (B) per register.  Four consecutive
// weights are one code byte; four codes of one register instead come from the same 2-bit field of four consecutive
// bytes - (word >> 2t) & 0x03030303 holds weights t, t+4, t+8, t+12 of a 16-weight word.  So the kernels use that order
// for the k of a half-block: mma k index 16h + 4t + j stands for weight (and activation) 16h + t + 4j.  The
// activations are stored in it (perm32), and a lane's two B registers (h = 0 and 1) are adjacent: one 8-byte load.
#include "strata/prefill/moe_fused.hpp"
#include "strata/kernels/gfx_arch.hpp"
#include "strata/core/emulate.hpp"
#include "strata/kernels/q8_1_finite.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace strata::prefill::fused {
namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill fused experts: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

constexpr int AB = 80;                  // activation bytes per 64 values: 64 codes, then {d0, c0, d1, c1}
constexpr int MAGIC = 0x4B400000;       // the bits of 1.5 * 2^23
constexpr float MAGICF = 12582912.0f;
// The Strata Q2_0 blob (moe_mmq.cu strata_q2_kernel, kernels.cu blob_dequant_kernel): gate/up codes [1280][640 B]
// (rows interleaved: gate of feature f at row 2f, its up at 2f + 1), down codes [2560][160 B], gate/up scales
// [1280][40] fp16, down scales [2560][10] fp16.
constexpr size_t O_GU_CODES = 0, O_D_CODES = (size_t) 1280 * 640, O_GU_SC = O_D_CODES + (size_t) 2560 * 160,
                 O_D_SC = O_GU_SC + (size_t) 1280 * 40 * 2;
constexpr int THREADS = 512;            // 16 warps: 4 along the weight rows (64 each) x 4 along the tile rows (16 each)
constexpr int WROWS = 256;              // weight rows per work item: 128 features of gate/up, or 256 outputs of down
constexpr int STAGES = 4;               // cp.async pipeline depth, one 64-weight block of K per stage
constexpr int STAGE_BYTES = WROWS * 16 + kTileRows * AB;
constexpr size_t smem_bytes(bool gu) {
    return (size_t) WROWS * (gu ? 40 : 10) * 2 + STAGES * STAGE_BYTES + kTileRows * 4;
}

// the stored position of value k of a 32-value half-block (see "The K order")
__host__ __device__ __forceinline__ int perm32(int k) { return 8 * (k & 3) + 4 * (k >> 4) + ((k & 15) >> 2); }
// where value k of a 32-value half-block is stored: perm32 for the CUDA kernel; on HIP the gfx11 WMMA kernel's order
// (value 4b + t of a 16-value word at 4t + b: see expert_w11_kernel)
__host__ __device__ __forceinline__ int act_pos(int k) {
#if defined(__HIPCC__)
    return (k & 16) | ((k & 3) << 2) | ((k >> 2) & 3);
#else
    return perm32(k);
#endif
}

struct Tables {
    int32_t *cnt, *off, *fill, *ts;      // per expert: rows, first row (E + 1), placement cursor, first tile (E + 1)
    int2* tiles;                         // {expert, first row}, in expert order
};
size_t tiles_at(int n_expert) { return ((size_t) (4 * n_expert + 2) * 4 + 15) / 16 * 16; }
Tables tables(void* scratch, int n_expert) {
    int32_t* p = (int32_t*) scratch;
    return {p, p + n_expert, p + 2 * n_expert + 1, p + 3 * n_expert + 1,
            (int2*) ((uint8_t*) scratch + tiles_at(n_expert))};
}

// ---- activations: one warp per 64 values
__global__ void quant_act_kernel(const float* __restrict__ x, int64_t nblk, uint8_t* __restrict__ xa) {
    const int64_t w = (int64_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (w >= nblk) return;
    const float v0 = x[w * 64 + lane], v1 = x[w * 64 + 32 + lane];
    float a0 = fabsf(v0), a1 = fabsf(v1);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        a0 = fmaxf(a0, __shfl_xor_sync(0xffffffffu, a0, o));
        a1 = fmaxf(a1, __shfl_xor_sync(0xffffffffu, a1, o));
    }
    const int q0 = a0 > 0.0f ? __float2int_rn(v0 * (127.0f / a0)) : 0;
    const int q1 = a1 > 0.0f ? __float2int_rn(v1 * (127.0f / a1)) : 0;
    int s0 = q0, s1 = q1;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        s0 += __shfl_xor_sync(0xffffffffu, s0, o);
        s1 += __shfl_xor_sync(0xffffffffu, s1, o);
    }
    uint8_t* out = xa + w * AB;
    out[act_pos(lane)] = (uint8_t) (int8_t) q0;
    out[32 + act_pos(lane)] = (uint8_t) (int8_t) q1;
    if (lane == 0)
        *(float4*) (out + 64) = make_float4(a0 / 127.0f, -(MAGICF + (float) s0), a1 / 127.0f, -(MAGICF + (float) s1));
}

// ---- grouping: counts, offsets and tiles, placement
__global__ void count_kernel(const int32_t* __restrict__ ids, int64_t n, int E, int32_t* __restrict__ cnt) {
    extern __shared__ int32_t hist[];
    for (int e = threadIdx.x; e < E; e += blockDim.x) hist[e] = 0;
    __syncthreads();
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const int e = ids[i];
        if ((unsigned) e < (unsigned) E) atomicAdd(&hist[e], 1);
    }
    __syncthreads();
    for (int e = threadIdx.x; e < E; e += blockDim.x)
        if (hist[e]) atomicAdd(&cnt[e], hist[e]);
}

// exclusive prefix sum over the block (blockDim a multiple of 32); *total = the block's sum
__device__ int block_excl_sum(int v, int* total, int* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) sh[w] = x;
    __syncthreads();
    if (w == 0) {
        int s = lane < nw ? sh[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += y;
        }
        sh[lane] = s;
    }
    __syncthreads();
    const int r = x - v + (w > 0 ? sh[w - 1] : 0);
    *total = sh[nw - 1];
    __syncthreads();   // sh is reused by the next call
    return r;
}

// one block, a thread per expert (E <= 1024)
__global__ void scan_kernel(Tables tb, int E) {
    __shared__ int sh[32];
    const int e = threadIdx.x;
    const int c = e < E ? tb.cnt[e] : 0, nt = (c + kTileRows - 1) / kTileRows;
    int rows = 0, tiles = 0;
    const int o = block_excl_sum(c, &rows, sh), to = block_excl_sum(nt, &tiles, sh);
    if (e < E) {
        tb.off[e] = o;
        tb.fill[e] = o;
        tb.ts[e] = to;
        for (int i = 0; i < nt; ++i) tb.tiles[to + i] = make_int2(e, o + i * kTileRows);
    }
    if (e == 0) {
        tb.off[E] = rows;
        tb.ts[E] = tiles;
    }
}

__global__ void place_kernel(const int32_t* __restrict__ ids, int64_t n, int k, int E, Tables tb,
                             int32_t* __restrict__ slot, int32_t* __restrict__ src) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int e = ids[i];
    if ((unsigned) e >= (unsigned) E) { slot[i] = 0; return; }   // (the router never writes one)
    const int p = atomicAdd(&tb.fill[e], 1);
    slot[i] = p;
    src[p] = (int32_t) (i / k);
}

// ---- the expert products
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(dst)),
                 "l"(src));
}
__device__ __forceinline__ void cp4(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"((unsigned) __cvta_generic_to_shared(dst)),
                 "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
// d = A (16 x 32 s8, row) * B (32 x 8 s8, col), int32, from zero
__device__ __forceinline__ void mma_s8(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
}
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
// Turing (STRATA_PF_FUSED_SM75): no cp.async - the stages are filled by plain copies (the same bytes; the barriers
// around them stay) - and no m16n8k32 with s8 operands: it is four m8n8k16, rows g / g + 8 by k 0-15 / 16-31 (the
// m16n8k32 fragments: a[0] / a[1] rows g / g + 8 at k 4 tig.., a[2] / a[3] the same at k 16 + 4 tig.., b0 / b1 the
// two k halves).  The int32 sums are exact in any order: the same d as the sm_80 instruction.
__device__ __forceinline__ void cp16(void* dst, const void* src) { *(uint4*) dst = __ldg((const uint4*) src); }
__device__ __forceinline__ void cp4(void* dst, const void* src) { *(uint32_t*) dst = __ldg((const uint32_t*) src); }
__device__ __forceinline__ void cp_commit() {}
template <int N> __device__ __forceinline__ void cp_wait() {}
__device__ __forceinline__ void mma8_s8(int& c0, int& c1, uint32_t a, uint32_t b) {
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n"
        : "+r"(c0), "+r"(c1) : "r"(a), "r"(b));
}
__device__ __forceinline__ void mma_s8(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    d[0] = d[1] = d[2] = d[3] = 0;
    mma8_s8(d[0], d[1], a[0], b0);
    mma8_s8(d[0], d[1], a[2], b1);
    mma8_s8(d[2], d[3], a[1], b0);
    mma8_s8(d[2], d[3], a[3], b1);
}
#endif

// One work item = (tile of up to 64 routed rows of expert e, a block of 256 weight rows); a persistent grid walks
// the batch's items (their count is on the device).  GU: gate/up rows of 128 features against the rows' tokens'
// activations (`act` per token, 2560 values), SwiGLU, H to int8 per 32 features into `out` (per row, 640 values).
// !GU: down rows (256 outputs) against H (`act` per row), FP32 into `dm` per row.
// Warp (wf, wt): weight rows 64 wf.. as four m16 tiles (GU: a tile = 8 features, its gate rows as mma rows 0-7 and
// its up rows as 8-15, so a lane holds gate and up of one feature: SwiGLU in registers), tile rows 16 wt.. as two n8.
template <bool GU>
__global__ void __launch_bounds__(THREADS, 1)
expert_kernel(const Batch b, const Tables tb, const uint8_t* __restrict__ act, const int32_t* __restrict__ src,
              uint8_t* __restrict__ out, float* __restrict__ dm) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    constexpr int NKB = GU ? 40 : 10;                 // 64-weight blocks along K
    constexpr int NFB = GU ? 5 : 10;                  // weight-row blocks per tile
    constexpr int CODE_LD = GU ? 640 : 160;           // code bytes per weight row
    constexpr size_t O_CODES = GU ? O_GU_CODES : O_D_CODES, O_SC = GU ? O_GU_SC : O_D_SC;
    constexpr int ACT_LD = NKB * AB;                  // bytes per activation row: 3200 (a token), 800 (a row's H)
    constexpr unsigned M2 = 0x03030303u;
    extern __shared__ __align__(16) uint8_t smem[];
    const __half* wsc = (const __half*) smem;                     // [256][NKB] the work item's weight scales
    uint8_t* stages = smem + WROWS * NKB * 2;                     // [STAGES] x {codes [256][16], act [64][AB]}
    int* srow = (int*) (stages + STAGES * STAGE_BYTES);           // [64] the activation row of each tile row

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, tig = lane & 3;
    const int wf = warp & 3, nb0 = 16 * (warp >> 2);
    int ra[4], rb[4];                                             // the local weight rows of mma rows g and g + 8
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (GU) { ra[i] = 2 * (32 * wf + 8 * i + g); rb[i] = ra[i] + 1; }
        else { ra[i] = 64 * wf + 16 * i + g; rb[i] = ra[i] + 8; }
    }
    const int t0 = tb.ts[b.e0], nwork = (tb.ts[b.e1] - t0) * NFB;
    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tl = tb.tiles[t0 + w / NFB];
        const int fb = w % NFB, e = tl.x, row0 = tl.y, nrows = min(kTileRows, tb.off[e + 1] - row0);
        const uint8_t* blob = b.blob[e - b.e0];
        const int rbase = fb * WROWS;
        __syncthreads();                                          // the previous item is done with the buffers
        if (tid < kTileRows) {
            const int r = row0 + min(tid, nrows - 1);             // rows past the tile's end repeat its last one
            srow[tid] = GU ? src[r] : r;
        }
        for (int c = tid; c < WROWS * 5; c += THREADS) {          // the scales: 5 x 16 B (GU) or 5 x 4 B a row
            const int r = c / 5, q = c % 5;
            if (GU) cp16(smem + r * 80 + q * 16, blob + O_SC + (size_t) (rbase + r) * 80 + q * 16);
            else cp4(smem + r * 20 + q * 4, blob + O_SC + (size_t) (rbase + r) * 20 + q * 4);
        }
        __syncthreads();                                          // srow
        auto load = [&](int kb) {
            uint8_t* st = stages + (kb % STAGES) * STAGE_BYTES;
            if (tid < WROWS) cp16(st + tid * 16, blob + O_CODES + (size_t) (rbase + tid) * CODE_LD + kb * 16);
            const int a = tid - (THREADS - kTileRows * 5);        // the last 320 threads: 64 rows x 5 x 16 B
            if (a >= 0) {
                const int r = a / 5, q = a % 5;
                cp16(st + WROWS * 16 + r * AB + q * 16, act + (size_t) srow[r] * ACT_LD + kb * AB + q * 16);
            }
        };
#pragma unroll
        for (int s = 0; s < STAGES - 1; ++s) {
            if (s < NKB) load(s);
            cp_commit();
        }
        // a warp whose rows are all past the tile's end only takes part in the loads
        const bool on0 = nb0 < nrows, on1 = nb0 + 8 < nrows;
        float acc[4][2][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int n = 0; n < 2; ++n)
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[i][n][q] = 0.0f;
        for (int kb = 0; kb < NKB; ++kb) {
            cp_wait<STAGES - 2>();
            __syncthreads();
            if (kb + STAGES - 1 < NKB) load(kb + STAGES - 1);
            cp_commit();
            if (!on0) continue;
            const uint8_t* st = stages + (kb % STAGES) * STAGE_BYTES;
            const uint8_t* sa = st + WROWS * 16;
            uint2 bq[2][2];                                       // [n8 tile][half]: the B registers
            float4 sc[2][2];                                      // [n8 tile][column 2 tig + c]: {d0, c0, d1, c1}
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
                const uint8_t* r = sa + (nb0 + 8 * n + g) * AB + 8 * tig;
                bq[n][0] = *(const uint2*) r;
                bq[n][1] = *(const uint2*) (r + 32);
                sc[n][0] = *(const float4*) (sa + (nb0 + 8 * n + 2 * tig) * AB + 64);
                sc[n][1] = *(const float4*) (sa + (nb0 + 8 * n + 2 * tig + 1) * AB + 64);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint4 ca = *(const uint4*) (st + ra[i] * 16), cb = *(const uint4*) (st + rb[i] * 16);
                const float da = __half2float(wsc[ra[i] * NKB + kb]), db = __half2float(wsc[rb[i] * NKB + kb]);
                const int s = 2 * tig;
                const uint32_t a0[4] = {(ca.x >> s) & M2, (cb.x >> s) & M2, (ca.y >> s) & M2, (cb.y >> s) & M2};
                const uint32_t a1[4] = {(ca.z >> s) & M2, (cb.z >> s) & M2, (ca.w >> s) & M2, (cb.w >> s) & M2};
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
                    int d0[4], d1[4];
                    mma_s8(d0, a0, bq[n][0].x, bq[n][0].y);
                    mma_s8(d1, a1, bq[n][1].x, bq[n][1].y);
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const float4 z = sc[n][q & 1];
                        const float u0 = __int_as_float(MAGIC + d0[q]) + z.y, u1 = __int_as_float(MAGIC + d1[q]) + z.w;
                        acc[i][n][q] = fmaf(q < 2 ? da : db, fmaf(z.z, u1, z.x * u0), acc[i][n][q]);
                    }
                }
            }
        }
        if (!on0) continue;
        if (GU) {
            // SwiGLU of feature 32 wf + 8 i + g (lane g of m tile i), tile rows nb0 + 8 n + 2 tig + c; H to int8 per
            // row over the warp's 32 features (= one 32-value half-block of the down product's K)
            const int blk = 4 * fb + wf;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    float h[4], am = 0.0f;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float gt = acc[i][n][c], up = acc[i][n][2 + c];
                        h[i] = gt / (1.0f + __expf(-gt)) * up;
                        am = fmaxf(am, fabsf(h[i]));
                    }
#pragma unroll
                    for (int o = 4; o < 32; o <<= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
                    const float inv = am > 0.0f ? 127.0f / am : 0.0f;
                    int qv[4], sum = 0;
#pragma unroll
                    for (int i = 0; i < 4; ++i) { qv[i] = __float2int_rn(h[i] * inv); sum += qv[i]; }
#pragma unroll
                    for (int o = 4; o < 32; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    if (r < nrows) {
                        uint8_t* o = out + (size_t) (row0 + r) * (10 * AB) + (blk >> 1) * AB;
                        const int hh = blk & 1;
#pragma unroll
                        for (int i = 0; i < 4; ++i) o[32 * hh + perm32(8 * i + g)] = (uint8_t) (int8_t) qv[i];
                        if (g == 0) *(float2*) (o + 64 + 8 * hh) = make_float2(am / 127.0f, -(MAGICF + (float) sum));
                    }
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int r = nb0 + 8 * n + 2 * tig + (q & 1);
                        if (r < nrows) dm[(size_t) (row0 + r) * 2560 + rbase + (q < 2 ? ra[i] : rb[i])] = acc[i][n][q];
                    }
                }
        }
    }
#endif
}

#if defined(__HIPCC__)
// ---- Aurora (S23): the same products on gfx11 (RDNA3 / RDNA3.5) matrix cores, v_wmma_i32_16x16x16_iu8 (wave32).
// Fragments (gfx11): A lane l holds the 16 k of row l % 16 (lanes 16..31 repeat lanes 0..15), B lane l the 16 k of
// column l % 16, C/D lane l holds D[2i + l / 16][l % 16], i = 0..7.  A = 16 weight rows, B = 16 tile rows (tokens):
// a lane's eight results are eight weight rows of ONE tile row, whose activation scales it already holds.
// The K order of a 16-weight word (one k-step): A register t = (word >> 2t) & 0x03030303 holds weights t, t+4, t+8,
// t+12, so k index p = 4t + b stands for weight 4b + t; the activations are stored in that order (act_pos).  The
// integer dot of each 32-value half (two k-steps) is exact and the epilogue is the CUDA kernel's, so the numbers are
// those of expert_kernel (another K order inside exact integer sums).
// No LDS staging: a wave reads its weight rows (16 B per row and 64-weight block) and its tile rows' activation
// blocks (80 B) straight from global memory (L2: an expert's rows are read by every tile, a token's by every weight
// block), one block ahead in registers; the work item's weight scales go to LDS once.
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
#define STRATA_FUSED_W11 1
#else
#define STRATA_FUSED_W11 0
#endif
typedef int w11_i4 __attribute__((ext_vector_type(4)));
typedef int w11_i8 __attribute__((ext_vector_type(8)));
constexpr int W_ROWS = 128;                       // weight rows per work item (GU: 64 features = one H block)
constexpr int W_THREADS = 256;                    // 8 waves: 4 along the weight rows (32 each) x 2 along the tile (32)

__device__ __forceinline__ w11_i8 wmma_iu8(w11_i4 a, w11_i4 b, w11_i8 c) {
#if STRATA_FUSED_W11
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, a, true, b, c, false);
#else
    __builtin_trap();
    return c;
#endif
}

template <bool GU>
__global__ void __launch_bounds__(W_THREADS)
expert_w11_kernel(const Batch b, const Tables tb, const uint8_t* __restrict__ act, const int32_t* __restrict__ src,
                  uint8_t* __restrict__ out, float* __restrict__ dm) {
#if STRATA_FUSED_W11
    constexpr int NKB = GU ? 40 : 10;                 // 64-weight blocks along K
    constexpr int NFB = (GU ? 1280 : 2560) / W_ROWS;  // work items per tile
    constexpr int CODE_LD = GU ? 640 : 160;           // code bytes per weight row
    constexpr size_t O_CODES = GU ? O_GU_CODES : O_D_CODES, O_SC = GU ? O_GU_SC : O_D_SC;
    constexpr int ACT_LD = NKB * AB;                  // bytes per activation row
    constexpr unsigned M2 = 0x03030303u;
    __shared__ __align__(16) uint16_t wsc[W_ROWS * NKB];        // the item's weight scales (fp16), [row][kb]
    __shared__ int srow[kTileRows];
    __shared__ float hs[GU ? kTileRows : 1][GU ? 65 : 1];      // GU: H of the item's 64 features, [tile row][feature]
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wm = wave & 3, wn = wave >> 2, l16 = lane & 15, hi = lane >> 4;
    const int t0 = tb.ts[b.e0], nwork = (tb.ts[b.e1] - t0) * NFB;
    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tl = tb.tiles[t0 + w / NFB];
        const int fb = w % NFB, e = tl.x, row0 = tl.y, nrows = min(kTileRows, tb.off[e + 1] - row0);
        const uint8_t* blob = b.blob[e - b.e0];
        const int rbase = fb * W_ROWS;
        __syncthreads();                                          // the previous item is done with srow / wsc / hs
        if (tid < kTileRows) {
            const int r = row0 + min(tid, nrows - 1);             // rows past the tile's end repeat its last one
            srow[tid] = GU ? src[r] : r;
        }
        {
            const uint4* s4 = reinterpret_cast<const uint4*>(blob + O_SC + (size_t) rbase * NKB * 2);
            for (int i = tid; i < W_ROWS * NKB / 8; i += W_THREADS) reinterpret_cast<uint4*>(wsc)[i] = s4[i];
        }
        __syncthreads();
        const bool on = 32 * wn < nrows;                          // a wave past the tile's rows only stages
        float acc[2][2][8];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[mt][nt][i] = 0.0f;
        if (on) {
            const uint8_t* arow[2];
            const uint8_t* brow[2];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) arow[mt] = blob + O_CODES + (size_t) (rbase + 32 * wm + 16 * mt + l16) * CODE_LD;
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) brow[nt] = act + (size_t) srow[32 * wn + 16 * nt + l16] * ACT_LD;
            uint4 ac[2], bq[2][4];
            float4 bs[2];
            auto fetch = [&](int kb, uint4 (&a)[2], uint4 (&q)[2][4], float4 (&s)[2]) {
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) a[mt] = *reinterpret_cast<const uint4*>(arow[mt] + kb * 16);
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const uint4* p = reinterpret_cast<const uint4*>(brow[nt] + kb * AB);
                    q[nt][0] = p[0]; q[nt][1] = p[1]; q[nt][2] = p[2]; q[nt][3] = p[3];
                    s[nt] = *reinterpret_cast<const float4*>(brow[nt] + kb * AB + 64);
                }
            };
            fetch(0, ac, bq, bs);
            for (int kb = 0; kb < NKB; ++kb) {
                uint4 nac[2], nbq[2][4];
                float4 nbs[2];
                if (kb + 1 < NKB) fetch(kb + 1, nac, nbq, nbs);
                w11_i8 c[2][2][2];                                // [half][mt][nt]
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                        for (int nt = 0; nt < 2; ++nt) c[h][mt][nt] = w11_i8{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
                for (int ks = 0; ks < 4; ++ks) {                  // k-step = 16-weight word ks of the block
                    w11_i4 A[2];
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        const unsigned wd = ks == 0 ? ac[mt].x : ks == 1 ? ac[mt].y : ks == 2 ? ac[mt].z : ac[mt].w;
                        A[mt] = w11_i4{(int) (wd & M2), (int) ((wd >> 2) & M2), (int) ((wd >> 4) & M2),
                                       (int) ((wd >> 6) & M2)};
                    }
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
                        const uint4 q = bq[nt][ks];
                        const w11_i4 B = w11_i4{(int) q.x, (int) q.y, (int) q.z, (int) q.w};
#pragma unroll
                        for (int mt = 0; mt < 2; ++mt) c[ks >> 1][mt][nt] = wmma_iu8(A[mt], B, c[ks >> 1][mt][nt]);
                    }
                }
                // the epilogue: per 64-weight block acc += d_w * (d_x0 * (dot0 - sum0) + d_x1 * (dot1 - sum1))
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    float dw[8];
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int m = 32 * wm + 16 * mt + 2 * i + hi;
                        dw[i] = __half2float(__ushort_as_half(wsc[m * NKB + kb]));
                    }
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
                        const float4 z = bs[nt];
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const float u0 = __int_as_float(MAGIC + c[0][mt][nt][i]) + z.y;
                            const float u1 = __int_as_float(MAGIC + c[1][mt][nt][i]) + z.w;
                            acc[mt][nt][i] = fmaf(dw[i], fmaf(z.z, u1, z.x * u0), acc[mt][nt][i]);
                        }
                    }
                }
                if (kb + 1 < NKB) {
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) ac[mt] = nac[mt];
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
                        bs[nt] = nbs[nt];
#pragma unroll
                        for (int j = 0; j < 4; ++j) bq[nt][j] = nbq[nt][j];
                    }
                }
            }
        }
        if constexpr (GU) {
            // SwiGLU: row 2f is feature f's gate (lanes 0..15), 2f + 1 its up (lanes 16..31) - the same (mt, i)
            if (on) {
#pragma unroll
                for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const float up = __shfl_xor_sync(0xffffffffu, acc[mt][nt][i], 16);
                            const float gt = acc[mt][nt][i];
                            if (hi == 0) hs[32 * wn + 16 * nt + l16][16 * wm + 8 * mt + i] = gt / (1.0f + __expf(-gt)) * up;
                        }
            }
            __syncthreads();
            // H block fb (features 64 fb .. 64 fb + 63) to int8: a thread per (tile row, half)
            if (tid < 2 * kTileRows) {
                const int r = tid >> 1, hh = tid & 1;
                if (r < nrows) {
                    float am = 0.0f;
#pragma unroll 8
                    for (int j = 0; j < 32; ++j) am = fmaxf(am, fabsf(hs[r][32 * hh + j]));
                    const float inv = am > 0.0f ? 127.0f / am : 0.0f;
                    uint8_t* o = out + (size_t) (row0 + r) * (10 * AB) + (size_t) fb * AB;
                    int sum = 0;
                    uint32_t wd[8] = {};
#pragma unroll
                    for (int j = 0; j < 32; ++j) {
                        const int qv = __float2int_rn(hs[r][32 * hh + j] * inv);
                        sum += qv;
                        const int p = act_pos(j);
                        wd[p >> 2] |= (uint32_t) (uint8_t) (int8_t) qv << (8 * (p & 3));
                    }
                    uint4* o4 = reinterpret_cast<uint4*>(o + 32 * hh);
                    o4[0] = make_uint4(wd[0], wd[1], wd[2], wd[3]);
                    o4[1] = make_uint4(wd[4], wd[5], wd[6], wd[7]);
                    *reinterpret_cast<float2*>(o + 64 + 8 * hh) = make_float2(am / 127.0f, -(MAGICF + (float) sum));
                }
            }
        } else if (on) {
#pragma unroll
            for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const int r = 32 * wn + 16 * nt + l16;
                    if (r < nrows) {
                        float* d = dm + (size_t) (row0 + r) * 2560 + rbase + 32 * wm + 16 * mt + hi;
#pragma unroll
                        for (int i = 0; i < 8; ++i) d[2 * i] = acc[mt][nt][i];
                    }
                }
        }
    }
#else
    __builtin_trap();
#endif
}
#endif  // __HIPCC__

unsigned blocks(int64_t n, int per) { return (unsigned) ((n + per - 1) / per); }

struct DevInfo {
    bool done = false;
    int cc = 0, sms = 0, occ_gu = 0, occ_d = 0;
};
std::mutex g_mu;
DevInfo g_dev[32];

#if defined(__HIPCC__)
// Resident blocks per multiprocessor for the gfx11 WMMA prompt-expert kernels' persistent grids.  HIP counts a gfx11
// WGP (two CUs, which these kernels use as one in the default WGP mode) as one multiprocessor, and the occupancy query
// answers 1 block for these kernels where 2 run side by side - so the grid was half the GPU's room.  gfx1151 (20
// WGPs), 8192 tokens x top 10, IQ3_S / IQ4_NL, one layer: 1 block per WGP 60.7 ms, 2 44.7, 3 49.9, 4 45.2 (the same
// work items and arithmetic: the results do not change).  STRATA_PF_OCC=N sets the blocks per WGP (an experiment knob).
static int wgp_blocks(int occ) {
    static const int env = [] {
        const char* v = std::getenv("STRATA_PF_OCC");
        return v != nullptr ? std::atoi(v) : 0;
    }();
    return env > 0 ? env : 2 * std::max(occ, 1);
}
#endif
const DevInfo& dev_info() {
    int dev = 0;
    cudaGetDevice(&dev);
    std::lock_guard<std::mutex> lk(g_mu);
    DevInfo& d = g_dev[dev & 31];
    if (d.done) return d;
    d.done = true;
#if defined(STRATA_HIP_GFX906)
    // gfx906 reports compute capability 9.0 through HIP, but the mma.sync bodies above are CUDA sm_80+ only and
    // empty in a hipcc build: never available here (the prompt path keeps its own expert GEMMs)
    return d;
#elif defined(__HIPCC__)
    {   // gfx11 (RDNA3 / RDNA3.5) with this build's code for it: the WMMA kernels
        cudaDeviceProp prop;
        cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, dev);
        hipFuncAttributes fa{};
        if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess || !strata::kernels::gfx_arch_is_gfx11_wmma(prop.gcnArchName) ||
            hipFuncGetAttributes(&fa, reinterpret_cast<const void*>(expert_w11_kernel<true>)) != hipSuccess) {
            cudaGetLastError();
            return d;
        }
        if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&d.occ_gu, expert_w11_kernel<true>, W_THREADS, 0) != hipSuccess ||
            hipOccupancyMaxActiveBlocksPerMultiprocessor(&d.occ_d, expert_w11_kernel<false>, W_THREADS, 0) != hipSuccess)
            d.occ_gu = d.occ_d = 0;
        d.cc = d.occ_gu >= 1 && d.occ_d >= 1 ? 80 : 0;
        cudaGetLastError();
        return d;
    }
#else
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, dev);
    d.cc = 10 * strata::cc_major_of(major) + strata::cc_minor_of(minor);   // STRATA_EMULATE_CC counts (tests)
    // Turing (sm_75): the same kernels with plain copies and m8n8k16 MMAs, opt-in (STRATA_PF_FUSED_SM75=1)
    static const bool sm75 = [] { const char* v = std::getenv("STRATA_PF_FUSED_SM75"); return v != nullptr && v[0] == '1'; }();
    if (d.cc < 75 || (d.cc < 80 && !sm75)) { d.cc = d.cc < 80 ? 0 : d.cc; return d; }
    // the device code of this card's arch must have the kernels (a build without an sm_75+ target has an empty body)
    cudaFuncAttributes fa{};
    if (cudaFuncGetAttributes(&fa, expert_kernel<true>) != cudaSuccess || fa.ptxVersion < (d.cc >= 80 ? 80 : 75)) {
        cudaGetLastError();
        d.cc = 0;
        return d;
    }
    ck(cudaFuncSetAttribute(expert_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem_bytes(true)),
       "smem gu");
    ck(cudaFuncSetAttribute(expert_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem_bytes(false)),
       "smem down");
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&d.occ_gu, expert_kernel<true>, THREADS, smem_bytes(true));
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&d.occ_d, expert_kernel<false>, THREADS, smem_bytes(false));
    if (d.occ_gu < 1 || d.occ_d < 1) d.cc = 0;                   // does not fit this card: the MMQ path
    cudaGetLastError();
#endif
    return d;
}

// ---- latent (edge0): the latent pack's A stage - the NVFP4 native blobs on this file's fused batch structure.
// The blob is the native layout of native_expert_layout(40, 40, ELL, I): [gate rows | up rows | down rows] of
// raw GGUF blocks, then the NVFP4 tail {s_gate, s_up, s_down, 0} (the expert's global scales).  A 64-value NVFP4
// block is 4 UE4M3 sub-scales (one per 16 values) and 32 bytes of E2M1 codes; a sub-block's first 8 values sit in
// the low nibbles of its 8 code bytes, the last 8 in the high nibbles (vec_dot_nvfp4_q8_1's order).  Doubled, the
// E2M1 values are the ints 0, +-1, +-2, +-3, +-4, +-6, +-8, +-12 (kvalues_fp4), so a 16-value sub-block is an
// exact s8 x s8 dot on mma.sync m16n8k16, its UE4M3 scale (halved, as ue4m3_to_f32_half decodes it) applied with
// the activation block's own fp16 scale in the epilogue - the same terms as the native path's vec_dot, another
// summation order.  The activations are the xl rows' own q8_1 blocks (quantize_q8_1_rows: the same codes and
// fp16 scales native_expert_grouped reads), staged into shared memory with each sub-block's 16 codes permuted so
// that a lane's fragment is one aligned word; the H the gate/up epilogue writes is the same q8_1 quantization as
// q8_1_store (the amax, the d = amax / 127 and the code rounding, bit for bit) in that permuted layout, and the
// down kernel reads it back the same way.  STRATA_LATENT_FUSED=0 in prefill.cpp keeps native_expert_grouped.
constexpr int LTHREADS = 512;             // as the Q2_0 kernel: 4 warps along the weight rows x 4 along the tile
constexpr int LSTAGES = 2;                // the decode / permute stages (plain loads: the bytes go through registers)
constexpr int LWPITCH = 76;               // a decoded weight row of a stage: 64 int8 codes, 4 fp16 sub-scales, pad
constexpr int LAPITCH = 80;               // a staged activation row: 16 codes per sub-block, {d0, d1} fp16, pad
inline size_t lat_smem_bytes() {
    return (size_t) LSTAGES * WROWS * LWPITCH + (size_t) LSTAGES * kTileRows * LAPITCH + (size_t) kTileRows * 4;
}

#if !defined(__HIPCC__)

// iq_kernels.cu's ue4m3_to_f32_half: the UE4M3 value halved (kvalues_fp4 holds the E2M1 values doubled)
__device__ __forceinline__ float lat_ue4m3_half(const uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    const uint32_t exp = (x >> 3) & 0xF, man = x & 0x7;
    return exp == 0 ? (float) man * 0.0009765625f : __uint_as_float(((exp + 119u) << 23) | (man << 20));
}

// the four nibbles of `q` (one code per byte) to the four doubled-E2M1 bytes (kvalues_fp4)
__device__ __forceinline__ uint32_t lat_kv4(const uint32_t q) {
    // a byte_perm selector is only 16 bits: the four per-byte codes (value in bits 0-2, sign in
    // bit 3 of each byte of q) must be gathered into the selector's four nibbles first - nibble
    // i = q byte i's code. (Passing q & 0x07070707 as the selector left bytes 2-3 of q unused:
    // only the low two bytes of a 32-bit value form the four selector nibbles.)
    const uint32_t sel = (q & 0x00000007u) | ((q & 0x00000700u) >> 4) | ((q & 0x00070000u) >> 8) |
                         ((q & 0x07000000u) >> 12);
    const uint32_t pl = __byte_perm(0x03020100u, 0x0C080604u, sel);   // {0, 1, 2, 3} {4, 6, 8, 12}
    const uint32_t ph = __byte_perm(0xFDFEFF00u, 0xF4F8FAFCu, sel);   // their negatives
    // the final selector, still 16 bits: nibble i = i (pl side), or i + 4 (the ph side) when q
    // byte i's sign bit is set - the sign bits land on nibble i's bit 2 (value 4)
    const uint32_t sg = ((q & 0x00000008u) >> 1) | ((q & 0x00000800u) >> 5) |
                        ((q & 0x00080000u) >> 9) | ((q & 0x08000000u) >> 13);
    return __byte_perm(pl, ph, 0x3210u | sg);
}

// One 16-value sub-block's 8 code bytes (w0 = bytes 0-3, w1 = bytes 4-7; the first 8 values in the low nibbles,
// the last 8 in the high ones) decoded and permuted into out[0..3]: out[k]'s bytes hold values k, 4+k, 8+k and
// 12+k - the word a lane reads at 4 * its tig for the mma's k fragments.
__device__ __forceinline__ void lat_decode_sub(const uint32_t w0, const uint32_t w1, uint32_t (&out)[4]) {
    const uint32_t lo0 = w0 & 0x0F0F0F0Fu, lo1 = w1 & 0x0F0F0F0Fu, hi0 = (w0 >> 4) & 0x0F0F0F0Fu,
                   hi1 = (w1 >> 4) & 0x0F0F0F0Fu;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint32_t lo = __byte_perm(__byte_perm(lo0, lo1, k | ((k + 4) << 4)),
                                        __byte_perm(hi0, hi1, k | ((k + 4) << 4)), 0x5410u);
        out[k] = lat_kv4(lo);
    }
}

// d += A (16 x 16 s8, row) * B (16 x 8 s8, col), int32
__device__ __forceinline__ void mma_s8_k16(int (&d)[4], const uint32_t a0, const uint32_t a1, const uint32_t b0) {
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0));
}

// q8_1_store's quantized value (iq_kernels.cu divides as a unit without --use_fast_math does: div.rn)
__device__ __forceinline__ int8_t lat_q8(const float xi, const float d, const float amax) {
    if (amax == 0.0f) return 0;
    const float q = roundf(__fdiv_rn(xi, d));
    return (int8_t) (q > 127.0f ? 127.0f : q < -127.0f ? -127.0f : q);
}

// One work item = (a tile of up to 64 routed rows of expert e, a block of the geometry's weight rows); a
// persistent grid walks the batch's items.  GU: gate feature f's row (blob row f) and its up row (up_off + f) as
// the mma tile's rows g and g + 8 (SwiGLU in registers), the item's 128 features against the rows' entries' q8_1
// activations, H quantized as q8_1_store into `hbuf`.  !GU: 256 down rows (256 outputs) against H, FP32 into
// `dm' per row, the expert's s_down on the store.  The mma runs per 16-value sub-block (the UE4M3 granularity):
// the decoded weight codes and the staged activation codes share the permuted layout above, so both fragments
// are one aligned word per lane.
template <bool GU>
__global__ void __launch_bounds__(LTHREADS, 1)
latent_kernel(const WideBatch b, const Tables tb, const LatentGeom g, const uint8_t* __restrict__ act,
              const int32_t* __restrict__ src, uint8_t* __restrict__ hbuf, float* __restrict__ dm) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const int NCH = GU ? (int) (g.ell >> 6) : (int) (g.iff >> 6);   // 64-value chunks along K (28 / 10)
    constexpr int NFB = GU ? 5 : 7;                                 // work items per tile (128 features / 256 rows)
    const size_t apitch = GU ? (size_t) (g.ell / 32) * 36 : (size_t) (g.iff >> 6) * LAPITCH;
    const int achunk = GU ? 72 : LAPITCH;                           // a staged row's source chunk (72 / 80 bytes)
    const size_t hpitch = (size_t) (g.iff >> 6) * LAPITCH;          // an H row (10 chunks of 80)
    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t* wst[2] = {smem, smem + WROWS * LWPITCH};
    uint8_t* ast[2] = {smem + 2 * WROWS * LWPITCH, smem + 2 * WROWS * LWPITCH + kTileRows * LAPITCH};
    int* srow = (int*) (smem + 2 * WROWS * LWPITCH + 2 * kTileRows * LAPITCH);   // [64] the act row of each tile row

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, mg = lane >> 2, tig = lane & 3;
    const int wf = warp & 3, nb0 = 16 * (warp >> 2);
    const int wrow = tid >> 1, whalf = tid & 1;                     // the weight rows: 2 threads a row, 2 subs each
    const int arow = tid >> 1, ahalf = tid & 1;                     // the act rows: threads 0..127, 2 a row
    uint32_t wq[4], aq[8];                                          // the next chunk's raw bytes
    uint32_t wd = 0, adw = 0;                                       // its weight sub-scale bytes / the act ds word

    const int t0 = tb.ts[b.e0], nwork = (tb.ts[b.e1] - t0) * NFB;
    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tl = tb.tiles[t0 + w / NFB];
        const int fb = w % NFB, e = tl.x, row0 = tl.y;
        // the latent grouping's off[] is a permutation (the resident experts' block comes first, then the staged
        // ones), so a tile's rows end at off[e] + cnt[e] - never at off[e + 1], which crosses a block boundary
        const int nrows = min(kTileRows, tb.cnt[e] - (row0 - tb.off[e]));
        const uint8_t* blob = b.blob[e - b.e0];
        if (blob == nullptr) continue;   // an expert of another launch of this layer: this batch has no blob for it
        // this thread's weight row of the item (GU: gate rows 0..127, up rows 128..255 of the stage)
        const uint8_t* wsrc;
        if (GU) wsrc = wrow < 128 ? blob + (size_t) (128 * fb + wrow) * g.gu_row
                                  : blob + g.up_off + (size_t) (128 * fb + wrow - 128) * g.gu_row;
        else wsrc = blob + g.down_off + (size_t) (256 * fb + wrow) * g.d_row;
        // the expert's global scales (the NVFP4 tail; 1 without one)
        float s_gate = 1.0f, s_up = 1.0f, s_down = 1.0f;
        if (g.tail_off) {
            const float* tail = (const float*) (blob + g.tail_off);
            s_gate = tail[0];
            s_up = tail[1];
            s_down = tail[2];
        }
        __syncthreads();                                              // the previous item is done with the buffers
        if (tid < kTileRows) {
            const int r = row0 + min(tid, nrows - 1);                // rows past the tile's end repeat its last one
            srow[tid] = GU ? src[r] : r;
        }
        __syncthreads();                                              // srow

        auto raw_load = [&](int kb) {
            const uint32_t* wp = (const uint32_t*) (wsrc + 36 * kb);
            wq[0] = wp[1 + 4 * whalf];
            wq[1] = wp[2 + 4 * whalf];
            wq[2] = wp[3 + 4 * whalf];
            wq[3] = wp[4 + 4 * whalf];
            wd = wp[0] >> (16 * whalf);                               // the block's d[2 * whalf .. +1]
            if (tid < 2 * kTileRows) {
                const uint32_t* ap = (const uint32_t*) (act + (size_t) srow[arow] * apitch + (size_t) achunk * kb);
#pragma unroll
                for (int i = 0; i < 8; ++i) aq[i] = ap[GU ? 1 + 9 * ahalf + i : 8 * ahalf + i];
                adw = ap[GU ? 9 * ahalf : 16];   // GU: the block's ds word; !GU: the row's {d0, d1}
            }
        };
        auto stage_store = [&](int kb) {
            uint8_t* wr = wst[kb & 1] + (size_t) wrow * LWPITCH;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int s = 2 * whalf + h;
                uint32_t out[4];
                lat_decode_sub(wq[2 * h], wq[2 * h + 1], out);
                *(uint32_t*) (wr + 16 * s) = out[0];
                *(uint32_t*) (wr + 16 * s + 4) = out[1];
                *(uint32_t*) (wr + 16 * s + 8) = out[2];
                *(uint32_t*) (wr + 16 * s + 12) = out[3];
                *(__half*) (wr + 68 + 2 * s) = __float2half(lat_ue4m3_half((uint8_t) (wd >> (8 * h))));
            }
            if (tid < 2 * kTileRows) {
                uint8_t* ar = ast[kb & 1] + (size_t) arow * LAPITCH;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int s = 2 * ahalf + h;
                    const uint32_t* wp = &aq[4 * h];
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        if (GU) {
                            const uint32_t p1 = __byte_perm(wp[0], wp[1], k | ((k + 4) << 4));
                            const uint32_t p2 = __byte_perm(wp[2], wp[3], k | ((k + 4) << 4));
                            *(uint32_t*) (ar + 16 * s + 4 * k) = __byte_perm(p1, p2, 0x5410u);
                        } else {
                            *(uint32_t*) (ar + 16 * s + 4 * k) = wp[k];
                        }
                    }
                }
                if (GU) *(__half*) (ar + 64 + 2 * ahalf) = __ushort_as_half((unsigned short) (adw & 0xFFFF));
                else *(uint32_t*) (ar + 64) = adw;
            }
        };

        raw_load(0);
        stage_store(0);
        if (NCH > 1) raw_load(1);
        const bool on0 = nb0 < nrows, on1 = nb0 + 8 < nrows;
        float acc[4][2][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int n = 0; n < 2; ++n)
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[i][n][q] = 0.0f;
        for (int kb = 0; kb < NCH; ++kb) {
            __syncthreads();
            if (on0) {
                const uint8_t* wc = wst[kb & 1];
                const uint8_t* ac = ast[kb & 1];
                for (int s = 0; s < 4; ++s) {
                    uint32_t bq[2];
                    float dx0[2], dx1[2];               // the OUTPUT rows' (2 tig, 2 tig + 1) own ds
                    for (int n = 0; n < 2; ++n) {
                        if (n == 1 && !on1) break;
                        const uint8_t* ar = ac + (size_t) (nb0 + 8 * n + mg) * LAPITCH;
                        bq[n] = *(const uint32_t*) (ar + 16 * s + 4 * tig);
                        const half2 ds0 = *(const half2*) (ac + (size_t) (nb0 + 8 * n + 2 * tig) * LAPITCH + 64);
                        const half2 ds1 = *(const half2*) (ac + (size_t) (nb0 + 8 * n + 2 * tig + 1) * LAPITCH + 64);
                        dx0[n] = __half2float((s & 2) ? ds0.y : ds0.x);
                        dx1[n] = __half2float((s & 2) ? ds1.y : ds1.x);
                    }
                    for (int i = 0; i < 4; ++i) {
                        const int ra = GU ? 32 * wf + 8 * i + mg : 64 * wf + 16 * i + mg;
                        const int rb = GU ? ra + 128 : ra + 8;
                        const uint32_t a0 = *(const uint32_t*) (wc + (size_t) ra * LWPITCH + 16 * s + 4 * tig);
                        const uint32_t a1 = *(const uint32_t*) (wc + (size_t) rb * LWPITCH + 16 * s + 4 * tig);
                        const float da = __half2float(*(const __half*) (wc + (size_t) ra * LWPITCH + 68 + 2 * s));
                        const float db = __half2float(*(const __half*) (wc + (size_t) rb * LWPITCH + 68 + 2 * s));
                        for (int n = 0; n < 2; ++n) {
                            if (n == 1 && !on1) break;
                            int d[4] = {0, 0, 0, 0};
                            mma_s8_k16(d, a0, a1, bq[n]);
                            acc[i][n][0] += (da * dx0[n]) * (float) d[0];
                            acc[i][n][1] += (da * dx1[n]) * (float) d[1];
                            acc[i][n][2] += (db * dx0[n]) * (float) d[2];
                            acc[i][n][3] += (db * dx1[n]) * (float) d[3];
                        }
                    }
                }
            }
            __syncthreads();
            if (kb + 1 < NCH) {
                stage_store(kb + 1);
                if (kb + 2 < NCH) raw_load(kb + 2);
            }
        }
        if (!on0) continue;
        if (GU) {
            // SwiGLU of feature 32 * (4 * fb + wf) + 8 i + mg (lane mg of m tile i), tile rows nb0 + 8 n + 2 tig + c;
            // H to q8_1 as q8_1_store (the amax over the warp's 32 features, d = amax / 127, the code rounding)
            const int blk = 4 * fb + wf;
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
                for (int c = 0; c < 2; ++c) {
                    float h[4], am = 0.0f;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float gt = acc[i][n][c] * s_gate, up = acc[i][n][2 + c] * s_up;
                        h[i] = __fmul_rn(__fdiv_rn(gt, 1.0f + __expf(-gt)), up);
                        am = fmaxf(am, fabsf(h[i]));
                    }
                    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 16));
                    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 8));
                    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4));
                    const float d = strata::kernels::q8_1_finite(__fdiv_rn(am, 127.0f));
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    if (r < nrows) {
                        uint8_t* o = hbuf + (size_t) (row0 + r) * hpitch + (size_t) (blk >> 1) * LAPITCH;
                        const int hh = blk & 1;
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const int v = 8 * i + mg;
                            o[16 * (2 * hh + (v >> 4)) + 4 * (v & 3) + ((v & 15) >> 2)] = (uint8_t) lat_q8(h[i], d, am);
                        }
                        if (mg == 0) *(__half*) (o + 64 + 2 * hh) = __float2half(d);
                    }
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int r = nb0 + 8 * n + 2 * tig + (q & 1);
                        if (r < nrows)
                            dm[(size_t) (row0 + r) * g.ell + 256 * fb +
                               (q < 2 ? 64 * wf + 16 * i + mg : 64 * wf + 16 * i + mg + 8)] = acc[i][n][q] * s_down;
                    }
                }
        }
    }
#endif
}

#endif  // !__HIPCC__


// ---- latent (edge0): the D and U dense projections on the same fused structure.  Both are Q8_0 weight rows x
// q8_1 activation rows - a 32-value block of each side is one exact s8 x s8 dot on mma.m16n8k16 (two sub-blocks
// share the Q8_0 block's fp16 scale, staged into all four sub-scale slots), the block scales applied in the
// epilogue per sub-block: the same terms as iq_mmvq(8)'s and latent_u_k's dp4a dots, another summation order.
// D (latent_dense_kernel<true>): the layer input's q8_1 rows [M = T, K = N] against all G*ell weight rows, the
// FP32 products quantized straight to the xl rows' own q8_1 blocks (q8_1_store bit for bit: the warp's amax and
// sum, d = amax / 127, the code rounding) - the xl FP32 round trip of iq_mmvq + quantize_q8_1_rows never runs.
// U (<false>): each entry's ylq row against its label's H weight rows (the entries bucketed by label on the
// host, one tile list per group), FP32 into Dm.
template <bool DU>
__global__ void __launch_bounds__(LTHREADS, 1)
latent_dense_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ xq,
                    const int32_t* __restrict__ rows, const int32_t* __restrict__ goff,
                    void* __restrict__ outq, float* __restrict__ outf,
                    int K, int64_t ell, int G, int H, int n_groups, int64_t M, int nwork, int fmt) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const int NCH = K >> 6;                       // 64-value chunks along K (40 for D, 28 for U)
    const int NFB = DU ? (int) ((size_t) G * ell >> 8) : H >> 8;   // 256-output items per tile (56 / 10)
    const size_t wrowb = (size_t) (K / 32) * (fmt == 4 ? 18 : 34);  // Q4DU: a Q8_0 (34) or Q4_0 (18) weight row
    const size_t apitch = (size_t) (K / 32) * 36; // a q8_1 activation row
    const int eblk = (int) (ell >> 5);            // q8_1 blocks in an xl row (56)
    const int ipg = (int) (ell >> 8);             // the item rows per output group (7)
    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t* wst[2] = {smem, smem + WROWS * LWPITCH};
    uint8_t* ast[2] = {smem + 2 * WROWS * LWPITCH, smem + 2 * WROWS * LWPITCH + kTileRows * LAPITCH};
    int* srow = (int*) (smem + 2 * WROWS * LWPITCH + 2 * kTileRows * LAPITCH);   // [64] the act row of each tile row

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, mg = lane >> 2, tig = lane & 3;
    const int wf = warp & 3, nb0 = 16 * (warp >> 2);
    const int wrow = tid >> 1, whalf = tid & 1;                     // the weight rows: 2 threads a row, a block each
    const int arow = tid >> 1, ahalf = tid & 1;                     // the act rows: threads 0..127, 2 a row
    uint32_t wq[8], aq[8];                                          // the next chunk's raw bytes
    uint32_t adw = 0;                                               // the act ds word
    __half wd16;                                                    // the weight block's scale

    for (int w_ = blockIdx.x; w_ < nwork; w_ += gridDim.x) {
        const int fb = w_ % NFB, t = w_ / NFB;
        int base, nrows;
        const uint8_t* wsec;
        if (DU) {
            base = 64 * t;
            nrows = min(kTileRows, (int) (M - base));
            wsec = w;
        } else {
            int g = 0, tb = 0;
            while (g < n_groups) {
                const int nt = (goff[g + 1] - goff[g] + 63) >> 6;
                if (t < tb + nt) break;
                tb += nt;
                ++g;
            }
            if (g >= n_groups) continue;
            base = goff[g] + 64 * (t - tb);
            nrows = min(kTileRows, goff[g + 1] - base);
            wsec = w + (size_t) g * (size_t) H * wrowb;
        }
        const uint8_t* wsrc = wsec + (size_t) (256 * fb + wrow) * wrowb;
        __syncthreads();                                              // the previous item is done with the buffers
        if (tid < kTileRows) {
            const int r = base + min(tid, nrows - 1);                // rows past the tile's end repeat its last one
            srow[tid] = DU ? r : rows[r];
        }
        __syncthreads();                                              // srow

        auto raw_load = [&](int kb) {
            // Q4DU fmt 4: one Q4_0 block {fp16 d; uint8 nib[16]} - element j in byte j's low nibble, j+16
            // in its high one, value (code - 8).  Expanded to the SAME 32-int8 qs layout as Q8_0 (wq byte
            // order = elements 4j..4j+3), so stage_store and the mma chain below are shared verbatim; the
            // centred codes are the exact integer form of iq_kernels.cu's q4_0_centred.
            if (fmt == 4) {
            const unsigned short* bp = (const unsigned short*) (wsrc + 18 * (2 * kb + whalf));
            wd16 = __ushort_as_half(bp[0]);
            const uint8_t* nbp = (const uint8_t*) (bp + 1);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint32_t v = 0;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int p = 4 * j + k;   // element p of the block
                    // p < 16: byte p's low nibble; p >= 16: byte (p-16)'s high nibble
                    const int code = (p < 16) ? (nbp[p] & 0xF) : ((nbp[p - 16] >> 4) & 0xF);
                    v |= ((uint32_t) ((code - 8) & 0xFF)) << (8 * k);   // centred, masked
                }
                wq[j] = v;
            }
            } else {
            // one Q8_0 block: {fp16 d; int8 qs[32]} - the qs words assembled from ushort pairs (a block is only
            // 2-byte aligned within its row)
            const unsigned short* bp = (const unsigned short*) (wsrc + 34 * (2 * kb + whalf));
            wd16 = __ushort_as_half(bp[0]);
#pragma unroll
            for (int j = 0; j < 8; ++j) wq[j] = (uint32_t) bp[1 + 2 * j] | ((uint32_t) bp[2 + 2 * j] << 16);
            }
            if (tid < 2 * kTileRows) {
                const uint32_t* ap = (const uint32_t*) (xq + (size_t) srow[arow] * apitch + (size_t) (72 * kb));
#pragma unroll
                for (int i = 0; i < 8; ++i) aq[i] = ap[1 + 9 * ahalf + i];
                adw = ap[9 * ahalf];                                 // the block's ds word (d in the low half)
            }
        };
        auto stage_store = [&](int kb) {
            uint8_t* wr = wst[kb & 1] + (size_t) wrow * LWPITCH;
#pragma unroll
            for (int h = 0; h < 2; ++h) {                             // this block's two 16-value sub-blocks
                const int s = 2 * whalf + h;
                const uint32_t* wp = &wq[4 * h];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const uint32_t p1 = __byte_perm(wp[0], wp[1], k | ((k + 4) << 4));
                    const uint32_t p2 = __byte_perm(wp[2], wp[3], k | ((k + 4) << 4));
                    *(uint32_t*) (wr + 16 * s + 4 * k) = __byte_perm(p1, p2, 0x5410u);
                }
            }
            // the Q8_0 block's one scale covers both its sub-blocks - but only THIS thread's
            // two slots: whalf 0 stages block 2kb (slots 68/70), whalf 1 block 2kb+1 (72/74).
            // (all four slots as before was a write race that lost the second block's scale)
            *(__half*) (wr + 68 + 4 * whalf) = wd16;
            *(__half*) (wr + 70 + 4 * whalf) = wd16;
            if (tid < 2 * kTileRows) {
                uint8_t* ar = ast[kb & 1] + (size_t) arow * LAPITCH;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int s = 2 * ahalf + h;
                    const uint32_t* wp = &aq[4 * h];
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const uint32_t p1 = __byte_perm(wp[0], wp[1], k | ((k + 4) << 4));
                        const uint32_t p2 = __byte_perm(wp[2], wp[3], k | ((k + 4) << 4));
                        *(uint32_t*) (ar + 16 * s + 4 * k) = __byte_perm(p1, p2, 0x5410u);
                    }
                }
                *(__half*) (ar + 64 + 2 * ahalf) = __ushort_as_half((unsigned short) (adw & 0xFFFF));
            }
        };

        raw_load(0);
        stage_store(0);
        if (NCH > 1) raw_load(1);
        const bool on0 = nb0 < nrows, on1 = nb0 + 8 < nrows;
        float acc[4][2][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int n = 0; n < 2; ++n)
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[i][n][q] = 0.0f;
        for (int kb = 0; kb < NCH; ++kb) {
            __syncthreads();
            if (on0) {
                const uint8_t* wc = wst[kb & 1];
                const uint8_t* ac = ast[kb & 1];
                for (int s = 0; s < 4; ++s) {
                    uint32_t bq[2];
                    float dx0[2], dx1[2];               // the OUTPUT rows' (2 tig, 2 tig + 1) own ds
                    for (int n = 0; n < 2; ++n) {
                        if (n == 1 && !on1) break;
                        const uint8_t* ar = ac + (size_t) (nb0 + 8 * n + mg) * LAPITCH;
                        bq[n] = *(const uint32_t*) (ar + 16 * s + 4 * tig);
                        const half2 ds0 = *(const half2*) (ac + (size_t) (nb0 + 8 * n + 2 * tig) * LAPITCH + 64);
                        const half2 ds1 = *(const half2*) (ac + (size_t) (nb0 + 8 * n + 2 * tig + 1) * LAPITCH + 64);
                        dx0[n] = __half2float((s & 2) ? ds0.y : ds0.x);
                        dx1[n] = __half2float((s & 2) ? ds1.y : ds1.x);
                    }
                    for (int i = 0; i < 4; ++i) {
                        const int ra = 64 * wf + 16 * i + mg, rb = ra + 8;
                        const uint32_t a0 = *(const uint32_t*) (wc + (size_t) ra * LWPITCH + 16 * s + 4 * tig);
                        const uint32_t a1 = *(const uint32_t*) (wc + (size_t) rb * LWPITCH + 16 * s + 4 * tig);
                        const float da = __half2float(*(const __half*) (wc + (size_t) ra * LWPITCH + 68 + 2 * s));
                        const float db = __half2float(*(const __half*) (wc + (size_t) rb * LWPITCH + 68 + 2 * s));
                        for (int n = 0; n < 2; ++n) {
                            if (n == 1 && !on1) break;
                            int d[4] = {0, 0, 0, 0};
                            mma_s8_k16(d, a0, a1, bq[n]);
                            acc[i][n][0] += (da * dx0[n]) * (float) d[0];
                            acc[i][n][1] += (da * dx1[n]) * (float) d[1];
                            acc[i][n][2] += (db * dx0[n]) * (float) d[2];
                            acc[i][n][3] += (db * dx1[n]) * (float) d[3];
                        }
                    }
                }
            }
            __syncthreads();
            if (kb + 1 < NCH) {
                stage_store(kb + 1);
                if (kb + 2 < NCH) raw_load(kb + 2);
            }
        }
        if (!on0) continue;
        if (DU) {
            // the warp's 64 outputs are two q8_1 blocks of the (token, group) xl row: quantized as q8_1_store
            // (the amax and the sum over the block's 32 values, d = amax / 127, the code rounding, ds = {d, sum})
            const int g_out = fb / ipg, rr = fb % ipg;               // the output group, the item within it
            const size_t xrowb = (size_t) eblk * 36;                 // an xlq row
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
                for (int qc = 0; qc < 2; ++qc) {
                    const int r_t = nb0 + 8 * n + 2 * tig + qc;      // the token's tile row
                    if (r_t >= nrows) continue;
                    const int t_tok = srow[r_t];
                    uint8_t* xrow = (uint8_t*) outq + (size_t) (t_tok * G + g_out) * xrowb;
                    for (int b = 0; b < 2; ++b) {                    // the two 32-value blocks of the warp
                        float am = 0.0f, sm = 0.0f;
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            am = fmaxf(am, fabsf(acc[2 * b + h][n][qc]));
                            am = fmaxf(am, fabsf(acc[2 * b + h][n][2 + qc]));
                            sm += acc[2 * b + h][n][qc] + acc[2 * b + h][n][2 + qc];
                        }
                        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 16));
                        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 8));
                        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4));
                        sm += __shfl_xor_sync(0xffffffffu, sm, 16);
                        sm += __shfl_xor_sync(0xffffffffu, sm, 8);
                        sm += __shfl_xor_sync(0xffffffffu, sm, 4);
                        const float d = strata::kernels::q8_1_finite(__fdiv_rn(am, 127.0f));
                        uint8_t* ob = xrow + (size_t) (8 * rr + 2 * wf + b) * 36;
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            ob[4 + 16 * h + mg] = (uint8_t) lat_q8(acc[2 * b + h][n][qc], d, am);
                            ob[4 + 16 * h + 8 + mg] = (uint8_t) lat_q8(acc[2 * b + h][n][2 + qc], d, am);
                        }
                        if (mg == 0) *(half2*) ob = strata::kernels::q8_1_ds(d, sm);
                    }
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int r_t = nb0 + 8 * n + 2 * tig + (q & 1);
                        if (r_t < nrows)
                            outf[(size_t) srow[r_t] * H + 256 * fb + 64 * wf + 16 * i + mg + 8 * (q >> 1)] =
                                acc[i][n][q];
                    }
                }
        }
    }
#endif
}

// ---- the U-stage's group tables, built on the device (a mid-layer pinned h2d waits behind the blob prefetch's
// DMA queue - 20-30 ms a layer once stream_all's prefetch runs deep, while these three tiny launches cost ~50 us).
// goff[0..n_groups] ends up holding the per-group run ends (0 then the exclusive prefix of the counts), rows[0..n)
// the entry indices bucketed by their expert's label; cursors is n_groups + 1 scratch for the scatter.
__global__ void lat_u_count_k(const int32_t* __restrict__ ids, const int32_t* __restrict__ labels,
                              int64_t n, int n_groups, int32_t* __restrict__ goff) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const int32_t e = ids[i];
        atomicAdd(&goff[1 + (e >= 0 ? labels[e] : 0) % n_groups], 1);
    }
}
__global__ void lat_u_prefix_k(int32_t* goff, int32_t* cursors, int n_groups) {
    int acc = 0;
    for (int g = 0; g < n_groups; ++g) {
        cursors[g] = acc;
        acc += goff[g + 1];
        goff[g + 1] = acc;
    }
    cursors[n_groups] = acc;
}
__global__ void lat_u_scatter_k(const int32_t* __restrict__ ids, const int32_t* __restrict__ labels,
                                int64_t n, int n_groups, int32_t* __restrict__ cursors,
                                int32_t* __restrict__ rows) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const int32_t e = ids[i];
        const int g = (e >= 0 ? labels[e] : 0) % n_groups;
        rows[atomicAdd(&cursors[g], 1)] = (int32_t) i;
    }
}

struct LatentDevInfo {
    bool checked = false, ok = false;
    int sms = 0, occ_gu = 0, occ_d = 0, occ_dd = 0, occ_du = 0;
};
std::mutex g_lat_mu;
LatentDevInfo g_lat_dev[32];

const LatentDevInfo& lat_dev() {
    int dev = 0;
    cudaGetDevice(&dev);
    std::lock_guard<std::mutex> lk(g_lat_mu);
    LatentDevInfo& d = g_lat_dev[dev & 31];
    if (!d.checked) {
        d.checked = true;
#if !defined(__HIPCC__)
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, dev);
        if (10 * major + minor < 80) return d;    // the s8 mma of m16n8k16
        cudaFuncAttributes fa{};
        if (cudaFuncGetAttributes(&fa, latent_kernel<true>) != cudaSuccess || fa.ptxVersion < 80) {
            cudaGetLastError();
            return d;
        }
        ck(cudaFuncSetAttribute(latent_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) lat_smem_bytes()),
           "latent smem gu");
        ck(cudaFuncSetAttribute(latent_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) lat_smem_bytes()),
           "latent smem down");
        ck(cudaFuncSetAttribute(latent_dense_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int) lat_smem_bytes()), "latent dense smem d");
        ck(cudaFuncSetAttribute(latent_dense_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int) lat_smem_bytes()), "latent dense smem u");
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&d.occ_gu, latent_kernel<true>, LTHREADS, lat_smem_bytes());
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&d.occ_d, latent_kernel<false>, LTHREADS, lat_smem_bytes());
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&d.occ_dd, latent_dense_kernel<true>, LTHREADS, lat_smem_bytes());
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&d.occ_du, latent_dense_kernel<false>, LTHREADS, lat_smem_bytes());
        cudaGetLastError();
        d.ok = d.occ_gu >= 1 && d.occ_d >= 1 && d.occ_dd >= 1 && d.occ_du >= 1;
#endif
    }
    return d;
}

}  // namespace

bool built() { return true; }
bool available() { return dev_info().cc >= 75; }
// 0.1.36: on by default for the Q2_0 pack (+13-23% prompts on an RTX 5070; teacher-forced against the FP16 path as
// close as MMQ at 8K and closer at 32K: phaseA-tests s18-tf); STRATA_PF_FUSED=0 keeps MMQ.  The native packs' kernels
// (moe_fused_iq) stay opt-in: STRATA_PF_FUSED=1 (requested()).
bool enabled() {
    static const bool env = [] {
        const char* v = std::getenv("STRATA_PF_FUSED");
#if defined(__HIPCC__)
        return v != nullptr && v[0] == '1';   // Aurora: the gfx11 WMMA kernels are opt-in (STRATA_PF_FUSED=1)
#else
        return v == nullptr || v[0] != '0';
#endif
    }();
    return env && available();
}
bool requested() {
    static const bool env = [] {
        const char* v = std::getenv("STRATA_PF_FUSED");
        return v != nullptr && v[0] == '1';
    }();
    return env && available();
}

size_t act_bytes(int64_t rows, int64_t cols) { return (size_t) rows * (size_t) (cols / 64) * AB; }
size_t group_bytes(int64_t n, int n_expert) {
    return tiles_at(n_expert) + ((size_t) (n + kTileRows - 1) / kTileRows + (size_t) n_expert) * sizeof(int2);
}

void quantize_act(const float* x, int64_t rows, int64_t cols, void* xa, void* stream) {
    if (rows <= 0) return;
    const int64_t nblk = rows * (cols / 64);
    quant_act_kernel<<<blocks(nblk, 8), 256, 0, (cudaStream_t) stream>>>(x, nblk, (uint8_t*) xa);
    ck(cudaGetLastError(), "quantize_act");
}

void group(const int32_t* ids, int64_t n, int k, int n_expert, void* scratch, int32_t* slot, int32_t* src,
           void* stream) {
    if (n_expert < 1 || n_expert > 1024) {
        std::fprintf(stderr, "prefill fused experts: %d experts (the grouping takes 1 to 1024)\n", n_expert);
        std::exit(1);
    }
    const cudaStream_t s = (cudaStream_t) stream;
    const Tables tb = tables(scratch, n_expert);
    ck(cudaMemsetAsync(tb.cnt, 0, (size_t) n_expert * 4, s), "group memset");
    if (n > 0) {
        const unsigned nb = std::min<unsigned>(blocks(n, 1024), 4u * (unsigned) std::max(dev_info().sms, 1));
        count_kernel<<<nb, 1024, (size_t) n_expert * 4, s>>>(ids, n, n_expert, tb.cnt);
    }
    scan_kernel<<<1, (unsigned) ((n_expert + 31) / 32 * 32), 0, s>>>(tb, n_expert);
    if (n > 0) place_kernel<<<blocks(n, 256), 256, 0, s>>>(ids, n, k, n_expert, tb, slot, src);
    ck(cudaGetLastError(), "group");
}

void experts(const Batch& b, int n_expert, int64_t n, const void* scratch, const void* xa, const int32_t* src,
             void* ha, float* dm, void* stream) {
    if (b.e1 <= b.e0 || n <= 0) return;
    const DevInfo& d = dev_info();
    const cudaStream_t s = (cudaStream_t) stream;
    const Tables tb = tables(const_cast<void*>(scratch), n_expert);
    // the most tiles the batch can have: every row in it, plus a partial tile per expert
    const int64_t tiles = (n + kTileRows - 1) / kTileRows + (b.e1 - b.e0);
#if defined(__HIPCC__)
    const unsigned g_gu = (unsigned) std::min<int64_t>(tiles * (1280 / W_ROWS), (int64_t) d.sms * wgp_blocks(d.occ_gu));
    const unsigned g_d = (unsigned) std::min<int64_t>(tiles * (2560 / W_ROWS), (int64_t) d.sms * wgp_blocks(d.occ_d));
    expert_w11_kernel<true><<<g_gu, W_THREADS, 0, s>>>(b, tb, (const uint8_t*) xa, src, (uint8_t*) ha, nullptr);
    expert_w11_kernel<false><<<g_d, W_THREADS, 0, s>>>(b, tb, (const uint8_t*) ha, src, nullptr, dm);
#else
    const unsigned g_gu = (unsigned) std::min<int64_t>(tiles * 5, (int64_t) d.sms * d.occ_gu);
    const unsigned g_d = (unsigned) std::min<int64_t>(tiles * 10, (int64_t) d.sms * d.occ_d);
    expert_kernel<true><<<g_gu, THREADS, smem_bytes(true), s>>>(b, tb, (const uint8_t*) xa, src, (uint8_t*) ha,
                                                                nullptr);
    expert_kernel<false><<<g_d, THREADS, smem_bytes(false), s>>>(b, tb, (const uint8_t*) ha, src, nullptr, dm);
#endif
    ck(cudaGetLastError(), "experts");
}

bool latent_fused_available() { return lat_dev().ok; }

size_t latent_group_bytes(int64_t n, int n_expert) { return group_bytes(n, n_expert); }

size_t latent_h_off(int64_t n, int n_expert) { return (group_bytes(n, n_expert) + 255) & ~(size_t) 255; }

size_t latent_scratch_bytes(int64_t n, int n_expert, int64_t iff) {
    return latent_h_off(n, n_expert) + (size_t) n * (size_t) (iff >> 6) * LAPITCH;
}

// the tables experts_latent reads: the layer's whole grouping (the per-expert entry counts and offsets of the
// host grouping, the 64-row tile list in expert id order) in the layout of `group`'s scratch.  `fill` stays zero:
// the latent entries are already grouped by the host, no placement runs.
void group_latent(const int32_t* cnt, const int32_t* off, int n_expert, void* host, size_t bytes) {
    std::memset(host, 0, bytes);
    int32_t* p = (int32_t*) host;
    int32_t* cnt_d = p;
    int32_t* off_d = p + n_expert;
    int32_t* ts_d = p + 3 * n_expert + 1;
    int2* tiles = (int2*) ((uint8_t*) host + tiles_at(n_expert));
    for (int e = 0; e < n_expert; ++e) cnt_d[e] = cnt[e];
    for (int e = 0; e <= n_expert; ++e) off_d[e] = off[e];
    int t = 0;
    for (int e = 0; e < n_expert; ++e) {
        ts_d[e] = t;
        const int c = cnt[e], nt = (c + kTileRows - 1) / kTileRows;
        for (int i = 0; i < nt; ++i) tiles[t + i] = make_int2(e, off[e] + i * kTileRows);
        t += nt;
    }
    ts_d[n_expert] = t;
}

// The batch's products on the latent geometry: gate/up from `xa` (the xl rows' q8_1 blocks, [rows][ell] of
// 36-byte blocks) through the rows `src` of the tables (src[row] = that entry's xl row), SwiGLU, H as q8_1 into
// `hbuf` (n rows of latent_h_off-sized scratch), down into `dm` [n][ell] FP32 at the tables' rows.  `n` bounds the
// launch grid.  The blobs are the native NVFP4 layout `g` describes; an expert of the tables without a blob in
// `b` (it belongs to another launch of the same layer) is skipped.
void experts_latent(const WideBatch& b, const LatentGeom& g, int n_expert, int64_t n, const void* scratch,
                    const void* xa, const int32_t* src, void* hbuf, float* dm, void* stream) {
    if (b.e1 <= b.e0 || n <= 0) return;
    const LatentDevInfo& d = lat_dev();
    if (!d.ok) {
        std::fprintf(stderr, "prefill fused experts (latent): the kernels are not available on this device\n");
        std::exit(1);
    }
    const cudaStream_t s = (cudaStream_t) stream;
    const Tables tb = tables(const_cast<void*>(scratch), n_expert);
    // the most tiles the batch can have: every row in it, plus a partial tile per expert
    const int64_t tiles = (n + kTileRows - 1) / kTileRows + (b.e1 - b.e0);
    const unsigned g_gu = (unsigned) std::min<int64_t>(tiles * 5, (int64_t) d.sms * d.occ_gu);
    const unsigned g_d = (unsigned) std::min<int64_t>(tiles * 7, (int64_t) d.sms * d.occ_d);
#if !defined(__HIPCC__)   // amd-hip port: the latent A kernels are CUDA-only (lat_dev().ok is false on HIP, so this is not reached)
    latent_kernel<true><<<g_gu, LTHREADS, lat_smem_bytes(), s>>>(b, tb, g, (const uint8_t*) xa, src,
                                                                 (uint8_t*) hbuf, nullptr);
    ck(cudaGetLastError(), "experts_latent gu");
    latent_kernel<false><<<g_d, LTHREADS, lat_smem_bytes(), s>>>(b, tb, g, (const uint8_t*) hbuf, src, nullptr, dm);
    ck(cudaGetLastError(), "experts_latent down");
#endif   // !__HIPCC__ (amd-hip port)
    ck(cudaGetLastError(), "experts_latent");
}

// The latent D projection on the fused dense kernel: the layer input's q8_1 rows [M = T, K = N] (`xq`,
// quantize_q8_1_rows of the mixed rows) against the layer's G*ell Q8_0 weight rows (`w`, the dense pack's D
// section), the products quantized to the xl rows' own q8_1 blocks in `outq` (row t*G + g) - the FP32 xl round
// trip of iq_mmvq + quantize_q8_1_rows never runs.
void latent_dense_d(const void* w, const void* xq, int64_t M, int K, int G, int64_t ell, void* outq, void* stream,
                     int fmt) {
    const LatentDevInfo& d = lat_dev();
    if (!d.ok) {
        std::fprintf(stderr, "prefill fused experts (latent dense): the kernels are not available on this device\n");
        std::exit(1);
    }
    const cudaStream_t s = (cudaStream_t) stream;
    const int64_t nwork = ((M + kTileRows - 1) / kTileRows) * (int64_t) (((size_t) G * ell) >> 8);
    const unsigned grid = (unsigned) std::min<int64_t>(nwork, (int64_t) d.sms * d.occ_dd);
    latent_dense_kernel<true><<<grid, LTHREADS, lat_smem_bytes(), s>>>(
        (const uint8_t*) w, (const uint8_t*) xq, nullptr, nullptr, outq, nullptr, K, ell, G, 0, 0, M, nwork,
        fmt);
    ck(cudaGetLastError(), "latent_dense_d");
}

// The latent U projection: each entry's ylq row (`xq`, [M = T*K, K = ELL]) against its label's H Q8_0 weight
// rows (`w`, the dense pack's U section, n_groups groups of H rows).  The entries are bucketed by their label
// on the host: `goff` [n_groups + 1] the per-group run in `rows` [M] (the entry index list), both in device
// memory; `nwork` the tile-item count that bounds the launch grid.  FP32 into `out` [M, H].
void latent_dense_u(const void* w, const void* yq, const int32_t* rows, const int32_t* goff, int n_groups,
                    int64_t nwork, int64_t M, int K, int H, float* out, void* stream, int fmt) {
    const LatentDevInfo& d = lat_dev();
    if (!d.ok) {
        std::fprintf(stderr, "prefill fused experts (latent dense): the kernels are not available on this device\n");
        std::exit(1);
    }
    const cudaStream_t s = (cudaStream_t) stream;
    const unsigned grid = (unsigned) std::min<int64_t>(nwork, (int64_t) d.sms * d.occ_du);
    latent_dense_kernel<false><<<grid, LTHREADS, lat_smem_bytes(), s>>>(
        (const uint8_t*) w, (const uint8_t*) yq, rows, goff, nullptr, out, K, 0, 0, H, n_groups, M, nwork,
        fmt);
    ck(cudaGetLastError(), "latent_dense_u");
}

// The U tables on the device: goff[0..n_groups] (the per-group run ends), rows[0..n) (the entries bucketed by
// their expert's label), cursors (n_groups + 1 scratch) - all from ids[] and labels[] that are already resident.
// The caller bounds the launch grid without the counts: ceil(n / 64) + n_groups tiles cover every split.
void latent_dense_u_tables(const int32_t* ids, const int32_t* labels, int64_t n, int n_groups, int32_t* goff,
                           int32_t* cursors, int32_t* rows, void* stream) {
    const cudaStream_t s = (cudaStream_t) stream;
    ck(cudaMemsetAsync(goff, 0, (size_t) (n_groups + 1) * 4, s), "latent u tables memset");
    lat_u_count_k<<<64, 256, 0, s>>>(ids, labels, n, n_groups, goff);
    lat_u_prefix_k<<<1, 1, 0, s>>>(goff, cursors, n_groups);
    lat_u_scatter_k<<<64, 256, 0, s>>>(ids, labels, n, n_groups, cursors, rows);
    ck(cudaGetLastError(), "latent_dense_u_tables");
}

}  // namespace strata::prefill::fused
