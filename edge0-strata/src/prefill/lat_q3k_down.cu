// src/prefill/lat_q3k_down.cu - see include/strata/prefill/lat_q3k_down.hpp.
//
// M1 of the q3k latent A stage (q3k_fused/BRIEF.md): the down half, IQ4_NL weights, K = 640, 1280 output rows.
// It is moe_fused.cu's NVFP4 latent down kernel (latent_kernel<false>, "the latent (edge0)" section) with only the
// block decode replaced - the same work item (256 output rows of one expert), the same 512-thread block, the same
// double-buffered staged rows, the same mma.sync m16n8k16 s8 products and the same epilogue:
//   staged weight row (LWPITCH 76): 4 x 16 int8 codes at 16 * s, four fp16 sub-scales at 68 + 2 * s
//   staged activation row (LAPITCH 80): the same 64 codes at 16 * s, then {d0, d1} (its two 32-value blocks' fp16
//   scales) at 64; the H rows the gate/up epilogue writes are already in that order (q8_1_store's), so the down
//   reads them as they are.
// The mma multiplies word k, byte c of a weight sub-block with word k, byte c of the activation sub-block, so both
// sides must hold the sub-block's value 4c + k there.
//
// The decode.  An IQ4_NL block is 18 bytes for 32 values - half d, then uint8 qs[16]: value j (0..15) is qs[j]'s low
// nibble through kvalues_iq4nl, value j + 16 its high nibble (ggml's dequantize_row_iq4_nl).  A 64-value chunk of a
// row is exactly two such blocks = 36 bytes, the size of an NVFP4 chunk, so the chunk stride (36), raw_load's shape
// and every stage buffer stay as they are.  The block's one d serves both of its 16-value halves (the NVFP4 rows
// carry a sub-scale per 16 values), so the four sub-scale slots get the two blocks' d's twice each.
//
// The expert's blob is read where it is: blob[g] + 704000 + row * 360 for the group's rows, no copy, no gather.
#include "strata/prefill/lat_q3k_down.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ggml.h"

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace strata::prefill {
namespace {

void dck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "lat q3k down: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// The q3k latent pack's geometry (q3k_fused/BRIEF.md): native_expert_layout(11, 20, 1280, 640).
constexpr int64_t kNE = 1280;          // the down's output rows (ELL)
constexpr int64_t kFF = 640;           // the down's K (n_ff)
constexpr size_t kDownOff = 704000;    // the blob's down section
constexpr size_t kDRow = 360;          // 20 IQ4_NL blocks x 18 B
constexpr int kChunks = (int) (kFF >> 6);    // 10 chunks of 64 values along K
constexpr int LHQ = kChunks * 80;            // a staged H row: 800 B

constexpr int LTHREADS = 512;          // as moe_fused.cu's latent kernel
constexpr int WROWS = 256;             // the down rows of one work item
constexpr int LWPITCH = 76;            // a staged weight row
constexpr int LAPITCH = 80;            // a staged activation row
constexpr int kTileRows = 64;          // routed rows (a group's entries) per tile
constexpr int kItems = (int) (kNE / WROWS);   // 5 work items along the output rows

inline size_t lat_smem_bytes() {
    return (size_t) 2 * WROWS * LWPITCH + (size_t) 2 * kTileRows * LAPITCH + (size_t) kTileRows * 4;
}

#if !defined(__HIPCC__)

// One 16-value sub-block's 4 code words of 4 qs words (q[m] = values 4m .. 4m + 3, the natural order) permuted into
// the staged order: out[k]'s byte c is the sub-block's value 4c + k.
__device__ __forceinline__ void lat_stage_codes(const uint32_t* A, uint32_t (&out)[4]) {
#pragma unroll
    for (int k = 0; k < 4; ++k)
        out[k] = __byte_perm(__byte_perm(A[0], A[1], k | ((k + 4) << 4)),
                             __byte_perm(A[2], A[3], k | ((k + 4) << 4)), 0x5410u);
}

__global__ void __launch_bounds__(LTHREADS, 1)
lat_q3k_down_kernel(const uint8_t* const* __restrict__ blob, const int32_t* __restrict__ starts, int n_groups,
                    const uint8_t* __restrict__ act, float* __restrict__ out, int64_t ld,
                    const int32_t* __restrict__ counts, const int32_t* __restrict__ ent_dst) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t* wst[2] = {smem, smem + WROWS * LWPITCH};
    uint8_t* ast[2] = {smem + 2 * WROWS * LWPITCH, smem + 2 * WROWS * LWPITCH + kTileRows * LAPITCH};
    int* srow = (int*) (smem + 2 * WROWS * LWPITCH + 2 * kTileRows * LAPITCH);   // [64] the act row of each tile row

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, mg = lane >> 2, tig = lane & 3;
    const int wf = warp & 3, nb0 = 16 * (warp >> 2);
    const int wrow = tid >> 1, whalf = tid & 1;                     // the weight rows: 2 threads a row, 1 block each
    const int arow = tid >> 1, ahalf = tid & 1;                     // the act rows: 128 threads, 2 a row
    uint32_t wq[4];                                                 // this thread's 16 qs bytes of a chunk
    uint32_t wd = 0;                                                // the block's d (fp16 bits)
    uint32_t aq[8], adw = 0;                                        // the chunk's act words / its {d0, d1}
    const int fb = blockIdx.x;                                      // this launch's 256 output rows of the item
    for (int g = blockIdx.y; g < n_groups; g += gridDim.y) {
        const uint8_t* bl = blob[g];
        if (bl == nullptr) continue;                                // an expert of another launch of this layer
        const int e0 = starts[g];
        // the group's row count: counts[] when the caller has it (the engine's flushes are not in group order),
        // otherwise the next group's start
        const int e1 = counts != nullptr ? e0 + counts[g] : starts[g + 1];
        // A group's 64-row tiles are split over blockIdx.z (the launcher widens z only when the groups alone do not
        // fill the device): a block row takes the tiles z, z + gridDim.z, ...
        for (int row0 = e0 + (int) blockIdx.z * kTileRows; row0 < e1; row0 += (int) gridDim.z * kTileRows) {
            const int nrows = min(kTileRows, e1 - row0);
            // the tile rows the products read: nb0 = 16 * (warp >> 2) and on0 / on1, so 16-row groups of them -
            // staging the rows past that is work no lane consumes (a 4-entry group read 64 of them before)
            const int arows = min(kTileRows, (nrows + 15) & ~15);
            // this thread's weight row of the item: row WROWS * fb + wrow of the group's 1280 down rows
            const uint8_t* wsrc = bl + kDownOff + (size_t) (WROWS * fb + wrow) * kDRow;
            __syncthreads();                                        // the previous item is done with the buffers
            if (tid < kTileRows) srow[tid] = row0 + min(tid, nrows - 1);   // rows past the tile repeat its last one
            __syncthreads();                                        // srow

            auto raw_load = [&](int kb) {
                // The chunk is 36 B = 9 words and its start is 4-byte aligned (704000, 360 and 36 are); an IQ4_NL
                // block's qs starts 2 B into it, so a block's 16 qs bytes are NOT word aligned - each thread reads
                // the 5 aligned words its block's qs spans and shifts the two halves' d out of the first.
                const uint32_t* w9 = (const uint32_t*) (wsrc + 36 * kb + 4 * (4 * whalf));
                const uint32_t b0 = w9[0], b1 = w9[1], b2 = w9[2], b3 = w9[3], b4 = w9[4];
                if (whalf) {                                        // block 1: d at bytes 18,19, qs at 20..35
                    wq[0] = b1; wq[1] = b2; wq[2] = b3; wq[3] = b4;
                    wd = b0 >> 16;
                } else {                                            // block 0: d at bytes 0,1, qs at 2..17
                    wq[0] = (b0 >> 16) | (b1 << 16);
                    wq[1] = (b1 >> 16) | (b2 << 16);
                    wq[2] = (b2 >> 16) | (b3 << 16);
                    wq[3] = (b3 >> 16) | (b4 << 16);
                    wd = b0 & 0xFFFFu;
                }
                if (tid < 2 * arows) {
                    const uint32_t* ap = (const uint32_t*) (act + (size_t) srow[arow] * LHQ + (size_t) LAPITCH * kb);
#pragma unroll
                    for (int i = 0; i < 8; ++i) aq[i] = ap[8 * ahalf + i];
                    adw = ap[16];                                   // the chunk's {d0, d1}
                }
            };
            auto stage_store = [&](int kb) {
                uint8_t* wr = wst[kb & 1] + (size_t) wrow * LWPITCH;
                // 32 codes in natural order: q[m] holds values 4m .. 4m + 3, the low nibbles of qs[4m .. 4m + 3]
                // (values 0..15) then the high ones (16..31); value j is qs[j]'s low nibble, j + 16 its high one
                uint32_t q[8];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    uint32_t lo = 0, hi = 0;
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        const uint32_t byte = (wq[k] >> (8 * c)) & 0xFF;
                        lo |= (uint32_t) (uint8_t) kvalues_iq4nl[byte & 0xF] << (8 * c);
                        hi |= (uint32_t) (uint8_t) kvalues_iq4nl[byte >> 4] << (8 * c);
                    }
                    q[k] = lo;
                    q[4 + k] = hi;
                }
                const unsigned short dh = (unsigned short) wd;      // the block's d, this chunk's 2 sub-blocks
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int s = 2 * whalf + h;
                    uint32_t o[4];
                    lat_stage_codes(q + 4 * h, o);
#pragma unroll
                    for (int k = 0; k < 4; ++k) *(uint32_t*) (wr + 16 * s + 4 * k) = o[k];
                    *(__half*) (wr + 68 + 2 * s) = __ushort_as_half(dh);
                }
                if (tid < 2 * arows) {
                    uint8_t* ar = ast[kb & 1] + (size_t) arow * LAPITCH;
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int s = 2 * ahalf + h;
                        const uint32_t* wp = &aq[4 * h];
#pragma unroll
                        for (int k = 0; k < 4; ++k) *(uint32_t*) (ar + 16 * s + 4 * k) = wp[k];
                    }
                    *(uint32_t*) (ar + 64) = adw;
                }
            };

            raw_load(0);
            stage_store(0);
            if (kChunks > 1) raw_load(1);
            const bool on0 = nb0 < nrows, on1 = nb0 + 8 < nrows;
            float acc[4][2][4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int n = 0; n < 2; ++n)
#pragma unroll
                    for (int q = 0; q < 4; ++q) acc[i][n][q] = 0.0f;
            for (int kb = 0; kb < kChunks; ++kb) {
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
                            const int ra = 64 * wf + 16 * i + mg;
                            const int rb = ra + 8;
                            const uint32_t a0 = *(const uint32_t*) (wc + (size_t) ra * LWPITCH + 16 * s + 4 * tig);
                            const uint32_t a1 = *(const uint32_t*) (wc + (size_t) rb * LWPITCH + 16 * s + 4 * tig);
                            const float da = __half2float(*(const __half*) (wc + (size_t) ra * LWPITCH + 68 + 2 * s));
                            const float db = __half2float(*(const __half*) (wc + (size_t) rb * LWPITCH + 68 + 2 * s));
                            for (int n = 0; n < 2; ++n) {
                                if (n == 1 && !on1) break;
                                int d[4] = {0, 0, 0, 0};
                                asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "
                                    "{%0,%1,%2,%3};\n"
                                    : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                                    : "r"(a0), "r"(a1), "r"(bq[n]));
                                acc[i][n][0] += (da * dx0[n]) * (float) d[0];
                                acc[i][n][1] += (da * dx1[n]) * (float) d[1];
                                acc[i][n][2] += (db * dx0[n]) * (float) d[2];
                                acc[i][n][3] += (db * dx1[n]) * (float) d[3];
                            }
                        }
                    }
                }
                __syncthreads();
                if (kb + 1 < kChunks) {
                    stage_store(kb + 1);
                    if (kb + 2 < kChunks) raw_load(kb + 2);
                }
            }
            if (!on0) continue;
            // the expert's s_down (NVFP4's tail) is 1 for a q3k blob (tail_off 0): the products go out as they are
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int r = nb0 + 8 * n + 2 * tig + (q & 1);
                        if (r < nrows)
                            out[(size_t) (ent_dst != nullptr ? ent_dst[row0 + r] : (row0 + r)) * ld + WROWS * fb +
                                (q < 2 ? 64 * wf + 16 * i + mg : 64 * wf + 16 * i + mg + 8)] = acc[i][n][q];
                    }
                }
        }
    }
#endif
}

#endif  // !__HIPCC__

}  // namespace

void lat_q3k_down(const uint8_t* const* blob, const int32_t* starts, int n_groups, const void* hq, float* out,
                  int64_t ld, void* stream, const int32_t* counts, const int32_t* ent_dst) {
    if (n_groups <= 0 || blob == nullptr || starts == nullptr || hq == nullptr || out == nullptr) return;
#if defined(__HIPCC__)
    std::fprintf(stderr, "lat q3k down: the latent A stage's kernels are CUDA-only\n");
    std::exit(1);
#else
    static const bool once = [] {
        dck(cudaFuncSetAttribute(lat_q3k_down_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int) lat_smem_bytes()), "smem");
        return true;
    }();
    (void) once;
    static const int sms = [] {
        int dev = 0, n = 0;
        return cudaGetDevice(&dev) == cudaSuccess && cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) ==
                                                       cudaSuccess && n > 0
                   ? n
                   : 0;   // 0: never widen z (the launch then cannot lose anything)
    }();
    const unsigned gy = (unsigned) std::min(n_groups, 512);
    // A launch of a few experts (a verify window's, or one big group) has too few (group, 256-row) items to fill
    // the device: the group's tiles go over blockIdx.z as well.  With many groups the extra z blocks would only exit.
    const unsigned gz = (sms > 0 && (int) (gy * kItems) < sms) ? 8u : 1u;
    lat_q3k_down_kernel<<<dim3((unsigned) kItems, gy, gz), LTHREADS, lat_smem_bytes(), (cudaStream_t) stream>>>(
        blob, starts, n_groups, (const uint8_t*) hq, out, ld, counts, ent_dst);
    dck(cudaGetLastError(), "launch");
#endif
}

}  // namespace strata::prefill
