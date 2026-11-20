// src/prefill/lat_dense_w11.cu - see include/strata/prefill/lat_dense_w11.hpp.
//
// One work item is 128 weight rows against a tile of up to 64 activation rows; a persistent grid walks the items.
// A workgroup is 8 waves: 4 along the weight rows (32 each) and 2 along the tile (32 each), so a wave owns a
// 32 x 32 block of outputs as four 16 x 16 WMMA tiles.
//
// Fragments (gfx11, wave32): A lane l holds the 16 k of row l % 16, B lane l the 16 k of column l % 16, and C / D
// lane l holds D[2i + l / 16][l % 16] for i = 0..7.  A is 16 weight rows and B is 16 activation rows, so a lane's
// eight results are eight weight rows of one activation row: the activation scale is the lane's own, and the eight
// weight scales come from LDS, where the item's scales are staged once.
#include "strata/prefill/lat_dense_w11.hpp"

#include "strata/kernels/gfx_arch.hpp"
#include "strata/kernels/q8_1_finite.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace strata::prefill::dense_w11 {

namespace {

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
#define STRATA_DENSE_W11 1
#else
#define STRATA_DENSE_W11 0
#endif

typedef int w11_i4 __attribute__((ext_vector_type(4)));
typedef int w11_i8 __attribute__((ext_vector_type(8)));

constexpr int kTile = 64;            // activation rows per tile
constexpr int kRows = 128;           // weight rows per work item
constexpr int kThreads = 256;        // 8 waves
constexpr int kMaxBlocks = 80;       // 32-value blocks along K: 2560 / 32 (D reads the model width)

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill latent dense (gfx11): %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// signed x signed: both sides are int8 codes
__device__ __forceinline__ w11_i8 wmma_s8(w11_i4 a, w11_i4 b, w11_i8 c) {
#if STRATA_DENSE_W11
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
#else
    __builtin_trap();
    return c;
#endif
}

// DU = true: the D projection (q8_1 out, optional FP32 out).  DU = false: the U projection (FP32 out).
template <bool DU>
__global__ void __launch_bounds__(kThreads)
dense_w11_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ xq, const int32_t* __restrict__ rows,
                 const int32_t* __restrict__ goff, uint8_t* __restrict__ outq, float* __restrict__ outf,
                 const float* __restrict__ inv, int K, int ell, int G, int H, int n_groups, int64_t M, int nwork) {
#if STRATA_DENSE_W11
    const int NKB = K >> 5;                                    // 32-value blocks along K
    const int NFB = (DU ? G * ell : H) / kRows;                // work items per tile
    const size_t wrowb = (size_t) NKB * 34;                    // a Q8_0 weight row: {fp16 d; int8 qs[32]} blocks
    const size_t apitch = (size_t) NKB * 36;                   // a q8_1 activation row: {fp16 d, fp16 sum; int8 qs[32]}
    __shared__ uint16_t wsc[kRows * kMaxBlocks];               // the item's weight scales (fp16 bits), [row][block]
    __shared__ int srow[kTile];                                // the activation row of each tile row
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wm = wave & 3, wn = wave >> 2, l16 = lane & 15, hi = lane >> 4;
    for (int w_ = blockIdx.x; w_ < nwork; w_ += gridDim.x) {
        const int fb = w_ % NFB, t = w_ / NFB;
        int base, nrows;
        const uint8_t* wsec;
        if constexpr (DU) {
            base = kTile * t;
            nrows = min(kTile, (int) (M - base));
            wsec = w;
        } else {
            int g = 0, tb = 0;
            while (g < n_groups) {
                const int nt = (goff[g + 1] - goff[g] + kTile - 1) / kTile;
                if (t < tb + nt) break;
                tb += nt;
                ++g;
            }
            if (g >= n_groups) continue;                       // the grid bound has more tiles than the runs do
            base = goff[g] + kTile * (t - tb);
            nrows = min(kTile, goff[g + 1] - base);
            wsec = w + (size_t) g * (size_t) H * wrowb;
        }
        if (nrows <= 0) continue;
        const int rbase = fb * kRows;
        __syncthreads();                                       // the previous item is done with srow / wsc
        if (tid < kTile) {
            const int r = base + min(tid, nrows - 1);          // rows past the tile's end repeat its last one
            srow[tid] = DU ? r : rows[r];
        }
        {
            const int row = tid >> 1;                          // two threads a weight row, alternate blocks each
            const uint8_t* p = wsec + (size_t) (rbase + row) * wrowb;
            for (int kb = tid & 1; kb < NKB; kb += 2)
                wsc[row * kMaxBlocks + kb] = *reinterpret_cast<const uint16_t*>(p + (size_t) kb * 34);
        }
        __syncthreads();
        const bool on = 32 * wn < nrows;                       // a wave past the tile's rows has nothing to do
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
            for (int mt = 0; mt < 2; ++mt) arow[mt] = wsec + (size_t) (rbase + 32 * wm + 16 * mt + l16) * wrowb;
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) brow[nt] = xq + (size_t) srow[32 * wn + 16 * nt + l16] * apitch;
            for (int kb = 0; kb < NKB; ++kb) {
                w11_i4 A[2][2], B[2][2];                       // [tile][k-step]
                float dx[2];
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    // a Q8_0 block is only 2-byte aligned within its row: the codes as ushort pairs
                    const uint16_t* q = reinterpret_cast<const uint16_t*>(arow[mt] + (size_t) kb * 34 + 2);
#pragma unroll
                    for (int ks = 0; ks < 2; ++ks)
#pragma unroll
                        for (int j = 0; j < 4; ++j)
                            A[mt][ks][j] = (int) ((uint32_t) q[8 * ks + 2 * j] | ((uint32_t) q[8 * ks + 2 * j + 1] << 16));
                }
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const uint32_t* q = reinterpret_cast<const uint32_t*>(brow[nt] + (size_t) kb * 36);
                    dx[nt] = __half2float(__ushort_as_half((unsigned short) (q[0] & 0xFFFFu)));
#pragma unroll
                    for (int ks = 0; ks < 2; ++ks)
#pragma unroll
                        for (int j = 0; j < 4; ++j) B[nt][ks][j] = (int) q[1 + 4 * ks + j];
                }
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    float dw[8];
#pragma unroll
                    for (int i = 0; i < 8; ++i)
                        dw[i] = __half2float(__ushort_as_half(wsc[(32 * wm + 16 * mt + 2 * i + hi) * kMaxBlocks + kb]));
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
                        w11_i8 c = w11_i8{0, 0, 0, 0, 0, 0, 0, 0};
                        c = wmma_s8(A[mt][0], B[nt][0], c);
                        c = wmma_s8(A[mt][1], B[nt][1], c);    // the block's exact integer dots
#pragma unroll
                        for (int i = 0; i < 8; ++i) acc[mt][nt][i] = fmaf(dw[i] * dx[nt], (float) c[i], acc[mt][nt][i]);
                    }
                }
            }
        }
        if (!on) continue;
        if constexpr (DU) {
            // the wave's 32 weight rows are one q8_1 block of the (token, group) xl row.  A lane holds 16 of a
            // token's 32 values (rows 16 mt + 2 i + hi) and lane ^ 16 the other 16: the block's amax and sum are
            // the pair's, the scale and the codes as q8_1_store makes them.
            const int j0 = rbase + 32 * wm;                    // the block's first weight row
            const int g_out = j0 / ell, c0 = j0 % ell;         // the output group, the block's first column in it
            const size_t xrowb = (size_t) (ell >> 5) * 36;     // an xlq row
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int r = 32 * wn + 16 * nt + l16;
                float v[2][8];
                float am = 0.0f, sm = 0.0f;
#pragma unroll
                for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int cj = c0 + 16 * mt + 2 * i + hi;
                        v[mt][i] = inv != nullptr ? acc[mt][nt][i] * inv[cj] : acc[mt][nt][i];
                        am = fmaxf(am, fabsf(v[mt][i]));
                        sm += v[mt][i];
                    }
                am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 16));
                sm += __shfl_xor_sync(0xffffffffu, sm, 16);
                if (r >= nrows) continue;
                const size_t orow = (size_t) srow[r] * (size_t) G + (size_t) g_out;
                const float d = strata::kernels::q8_1_finite(am / 127.0f);
                uint8_t* ob = outq + orow * xrowb + (size_t) (c0 >> 5) * 36;
#pragma unroll
                for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                    for (int i = 0; i < 8; ++i)
                        ob[4 + 16 * mt + 2 * i + hi] = (uint8_t) strata::kernels::q8_1_quant(v[mt][i], d, am);
                if (hi == 0) *reinterpret_cast<half2*>(ob) = strata::kernels::q8_1_ds(d, sm);
                if (outf != nullptr) {
                    float* o = outf + orow * (size_t) ell + c0 + hi;
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                        for (int i = 0; i < 8; ++i) o[16 * mt + 2 * i] = v[mt][i];
                }
            }
        } else {
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int r = 32 * wn + 16 * nt + l16;
                if (r >= nrows) continue;
                float* o = outf + (size_t) srow[r] * (size_t) H + rbase + 32 * wm + hi;
#pragma unroll
                for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                    for (int i = 0; i < 8; ++i) o[16 * mt + 2 * i] = acc[mt][nt][i];
            }
        }
    }
#else
    __builtin_trap();
#endif
}

struct DevInfo {
    bool done = false, ok = false;
    int sms = 0, occ_d = 0, occ_u = 0;
};
std::mutex g_mu;
DevInfo g_dev[32];

// As moe_fused.cu's wgp_blocks: HIP counts a gfx11 WGP as one multiprocessor and the occupancy query answers one
// block where two run side by side.  STRATA_PF_OCC=N sets the blocks per WGP.
int wgp_blocks(int occ) {
    static const int env = [] {
        const char* v = std::getenv("STRATA_PF_OCC");
        return v != nullptr ? std::atoi(v) : 0;
    }();
    return env > 0 ? env : 2 * std::max(occ, 1);
}

const DevInfo& dev_info() {
    int dev = 0;
    cudaGetDevice(&dev);
    std::lock_guard<std::mutex> lk(g_mu);
    DevInfo& d = g_dev[dev & 31];
    if (d.done) return d;
    d.done = true;
#if defined(__HIPCC__) && !defined(STRATA_HIP_GFX906)
    cudaDeviceProp prop;
    cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, dev);
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess || !strata::kernels::gfx_arch_is_gfx11_wmma(prop.gcnArchName)) {
        cudaGetLastError();
        return d;
    }
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&d.occ_d, dense_w11_kernel<true>, kThreads, 0) != hipSuccess ||
        hipOccupancyMaxActiveBlocksPerMultiprocessor(&d.occ_u, dense_w11_kernel<false>, kThreads, 0) != hipSuccess)
        d.occ_d = d.occ_u = 0;
    cudaGetLastError();
    d.ok = d.sms > 0 && d.occ_d >= 1 && d.occ_u >= 1;
#endif
    return d;
}

void need(const DevInfo& d, int K, const char* what) {
    if (!d.ok || (K >> 5) > kMaxBlocks || (K & 31) != 0) {
        std::fprintf(stderr, "prefill latent dense (gfx11): %s is not available here (K = %d)\n", what, K);
        std::exit(1);
    }
}

}  // namespace

bool available(int fmt) { return fmt == 8 && dev_info().ok; }

void project_d(const void* w, const void* xq, int64_t M, int K, int G, int64_t ell, void* outq, float* xl_f32,
               const float* inv, void* stream) {
    const DevInfo& d = dev_info();
    need(d, K, "the D projection");
    if (M <= 0) return;
    if (ell % kRows != 0) {
        std::fprintf(stderr, "prefill latent dense (gfx11): the latent width %lld is not a multiple of %d\n",
                     (long long) ell, kRows);
        std::exit(1);
    }
    const int64_t nwork = ((M + kTile - 1) / kTile) * (int64_t) ((size_t) G * (size_t) ell / kRows);
    const unsigned grid = (unsigned) std::min<int64_t>(nwork, (int64_t) d.sms * wgp_blocks(d.occ_d));
    dense_w11_kernel<true><<<grid, kThreads, 0, (cudaStream_t) stream>>>(
        (const uint8_t*) w, (const uint8_t*) xq, nullptr, nullptr, (uint8_t*) outq, xl_f32, inv, K, (int) ell, G, 0,
        0, M, (int) nwork);
    ck(cudaGetLastError(), "the D projection");
}

void project_u(const void* w, const void* yq, const int32_t* rows, const int32_t* goff, int n_groups, int64_t tiles,
               int64_t M, int K, int H, float* out, void* stream) {
    const DevInfo& d = dev_info();
    need(d, K, "the U projection");
    if (M <= 0 || tiles <= 0) return;
    if (H % kRows != 0) {
        std::fprintf(stderr, "prefill latent dense (gfx11): the model width %d is not a multiple of %d\n", H, kRows);
        std::exit(1);
    }
    const int64_t nwork = tiles * (int64_t) (H / kRows);
    const unsigned grid = (unsigned) std::min<int64_t>(nwork, (int64_t) d.sms * wgp_blocks(d.occ_u));
    dense_w11_kernel<false><<<grid, kThreads, 0, (cudaStream_t) stream>>>(
        (const uint8_t*) w, (const uint8_t*) yq, rows, goff, nullptr, out, nullptr, K, 0, 0, H, n_groups, M,
        (int) nwork);
    ck(cudaGetLastError(), "the U projection");
}

}  // namespace strata::prefill::dense_w11
