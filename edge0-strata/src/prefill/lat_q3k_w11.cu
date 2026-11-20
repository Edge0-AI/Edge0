// src/prefill/lat_q3k_w11.cu - see include/strata/prefill/lat_q3k_w11.hpp.
//
// One work item is 256 weight rows against a tile of up to 64 routed rows of one expert; a persistent grid walks
// the batch's items.  Gate/up: the item is 128 features, their 128 gate rows and 128 up rows.  Down: 256 output rows.
// A workgroup is 8 waves along the weight rows.  A wave runs two 16-row WMMA tiles of weights against the tile's
// four 16-row activation tiles, so a weight tile is decoded once for all 64 rows.  For gate/up the two weight tiles
// are the gate rows and the up rows of the same 16 features, so SwiGLU stays inside the wave.
//
// Fragments (gfx11, wave32): A lane l holds the 16 k of row l % 16, B lane l the 16 k of column l % 16, and C / D
// lane l holds D[2i + l / 16][l % 16] for i = 0..7.  A k-step is 16 values: one Q3_K sub-block, and half of an
// IQ4_NL or q8_1 block.
//
// Q3_K: a weight is d * (scale - 32) * (code - 4), with one 6-bit scale per sub-block.  A lane's A operand is one
// row's sub-block, so it has one scale: the lane multiplies it into its 16 codes, as (4 - code) * (scale - 32),
// which is in [-128, 124] and so an int8.  The matrix core then sums a whole 32-value activation block exactly, and
// only d (one per 256 weights) and the activation block's scale are left for the FP32 step.  These are the terms of
// ggml's dequantized dot product, in another summation order.
//
// The weights are decoded in registers from the GGUF blocks, where the expert already is.  The item's FP32 weight
// scales go to LDS once: Q3_K's d per super-block (negated: the operand carries the other sign), IQ4_NL's d per block.
#include "strata/prefill/lat_q3k_w11.hpp"

#include "strata/kernels/gfx_arch.hpp"
#include "strata/kernels/q8_1_finite.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace strata::prefill::q3k_w11 {

namespace {

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
#define STRATA_Q3K_W11 1
#else
#define STRATA_Q3K_W11 0
#endif

typedef int w11_i4 __attribute__((ext_vector_type(4)));
typedef int w11_i8 __attribute__((ext_vector_type(8)));

constexpr int kTile = fused::kTileRows;   // routed rows per tile
constexpr int kRows = 256;                // weight rows per work item
constexpr int kThreads = 256;             // 8 waves
constexpr int kMaxSc = 20;                // FP32 scales per weight row: Q3_K super-blocks, or IQ4_NL blocks of 640
constexpr int kHChunk = 80;               // an H row's 64-feature chunk: 64 codes, two fp16 scales, pad

struct Tabs {
    const int32_t *cnt, *off, *ts;
    const int2* tiles;
};

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill latent experts (gfx11): %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

__device__ __forceinline__ w11_i8 wmma_s8(w11_i4 a, w11_i4 b, w11_i8 c) {
#if STRATA_Q3K_W11
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
#else
    __builtin_trap();
    return c;
#endif
}

// 16 bytes at p as four words.  A GGUF row is only 2-byte aligned (a Q3_K row is 770 B at ell = 1792): the aligned
// words around it are read and shifted, and the fifth word only when the row needs it.
__device__ __forceinline__ void ld16(const uint8_t* p, uint32_t (&w)[4]) {
    const uintptr_t u = (uintptr_t) p;
    const uint32_t* q = reinterpret_cast<const uint32_t*>(u & ~(uintptr_t) 3);
    const int sh = 8 * (int) (u & 3);
    const uint32_t q4 = sh ? q[4] : 0u;
    const uint32_t v[5] = {q[0], q[1], q[2], q[3], q4};
#pragma unroll
    for (int i = 0; i < 4; ++i) w[i] = (uint32_t) ((((uint64_t) v[i + 1] << 32) | v[i]) >> sh);
}
__device__ __forceinline__ float ldh(const uint8_t* p) {
    return __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(p)));
}

// per-byte a + b (mod 256), no carry between the bytes
__device__ __forceinline__ uint32_t add8(const uint32_t a, const uint32_t b) {
    return ((a & 0x7F7F7F7Fu) + (b & 0x7F7F7F7Fu)) ^ ((a ^ b) & 0x80808080u);
}

// four IQ4_NL codes (one per byte, 0..15) to their int8 values: ggml's kvalues_iq4nl as two 8-byte tables
__device__ __forceinline__ uint32_t iq4nl_word(const uint32_t n) {
    constexpr uint64_t lo = 0xF6EADDCFBFAD9881ull;   // -127 -104 -83 -65 -49 -35 -22 -10
    constexpr uint64_t hi = 0x7159453526190D01ull;   //    1   13  25  38  53  69  89 113
    uint32_t o = 0;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const uint32_t idx = (n >> (8 * c)) & 15u;
        const uint64_t t = (idx & 8u) ? hi : lo;
        o |= (uint32_t) ((t >> (8 * (idx & 7u))) & 0xFFu) << (8 * c);
    }
    return o;
}

// GU = true: gate/up, SwiGLU, H as q8_1 into `out`.  GU = false: down, FP32 into `dm`.
template <bool GU>
__global__ void __launch_bounds__(kThreads)
latent_w11_kernel(const fused::WideBatch b, const Tabs tb, const fused::LatentGeom g, const uint8_t* __restrict__ act,
                  const int32_t* __restrict__ src, uint8_t* __restrict__ out, float* __restrict__ dm) {
#if STRATA_Q3K_W11
    const int ell = (int) g.ell, iff = (int) g.iff;
    const int NKB = (GU ? ell : iff) >> 5;                     // 32-value blocks along K
    const int NFB = GU ? iff / 128 : ell / kRows;              // work items per tile
    const size_t apitch = GU ? (size_t) (ell >> 5) * 36 : (size_t) (iff >> 6) * kHChunk;
    const size_t hpitch = (size_t) (iff >> 6) * kHChunk;
    __shared__ float wd[kRows * kMaxSc];                       // the item's FP32 weight scales, [row][block]
    __shared__ int srow[kTile];                                // the activation row of each tile row
    __shared__ float ham[GU ? 8 * kTile : 1];                  // gate/up: |H| max of a wave's 16 features, [wave][tile row]
    const int tid = threadIdx.x, lane = tid & 31, wv = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    const int t0 = tb.ts[b.e0], nwork = (tb.ts[b.e1] - t0) * NFB;
    for (int w_ = blockIdx.x; w_ < nwork; w_ += gridDim.x) {
        const int2 tl = tb.tiles[t0 + w_ / NFB];
        const int fb = w_ % NFB, e = tl.x, row0 = tl.y;
        // off[] is a permutation for the latent grouping, so a tile's rows end at off[e] + cnt[e]
        const int nrows = min(kTile, tb.cnt[e] - (row0 - tb.off[e]));
        const uint8_t* blob = b.blob[e - b.e0];
        if (blob == nullptr || nrows <= 0) continue;           // an expert of another launch of this layer
        float s_gate = 1.0f, s_up = 1.0f, s_down = 1.0f;
        if (g.tail_off) {
            const float* tail = reinterpret_cast<const float*>(blob + g.tail_off);
            s_gate = tail[0];
            s_up = tail[1];
            s_down = tail[2];
        }
        // the item's weight row `lr` (0..255).  Gate/up: 0..127 the gate rows of the item's features, 128..255 their up rows
        auto wrow = [&](int lr) -> const uint8_t* {
            if constexpr (GU) return blob + (lr >= 128 ? g.up_off : 0) + (size_t) (128 * fb + (lr & 127)) * g.gu_row;
            else return blob + g.down_off + (size_t) (kRows * fb + lr) * g.d_row;
        };
        // the wave's A tile `a` (0..1), its row r16 (0..15), as an item row
        auto lrow = [&](int a, int r16) -> int {
            if constexpr (GU) return 128 * a + 16 * wv + r16;
            else return 32 * wv + 16 * a + r16;
        };
        __syncthreads();                                       // the previous item is done with the LDS tables
        if (tid < kTile) {
            const int r = row0 + min(tid, nrows - 1);          // rows past the tile's end repeat its last one
            srow[tid] = GU ? src[r] : r;
        }
        {
            const uint8_t* p = wrow(tid);                      // a thread a weight row
            if constexpr (GU) {
                // block_q3_K: hmask[32], qs[64], scales[12], fp16 d.  The operand is (4 - code) * (scale - 32): -d here
                for (int sb = 0; sb < (NKB >> 3); ++sb) wd[tid * kMaxSc + sb] = -ldh(p + (size_t) 110 * sb + 108);
            } else {
                for (int kb = 0; kb < NKB; ++kb) wd[tid * kMaxSc + kb] = ldh(p + (size_t) 18 * kb);
            }
        }
        __syncthreads();
        float acc[2][4][8];
#pragma unroll
        for (int a = 0; a < 2; ++a)
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[a][nt][i] = 0.0f;
        const uint8_t* arow[2];
        const uint8_t* brow[4];
#pragma unroll
        for (int a = 0; a < 2; ++a) arow[a] = wrow(lrow(a, l16));
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) brow[nt] = act + (size_t) srow[16 * nt + l16] * apitch;
        float dw[2][8];                                        // the FP32 scales of the lane's weight rows
        uint32_t sw[2][3];                                     // Q3_K: the super-block's 12 scale bytes
        for (int kb = 0; kb < NKB; ++kb) {
            if constexpr (GU) {
                if ((kb & 7) == 0) {                           // a new 256-weight super-block
#pragma unroll
                    for (int a = 0; a < 2; ++a) {
                        uint32_t t[4];
                        ld16(arow[a] + (size_t) 110 * (kb >> 3) + 96, t);
                        sw[a][0] = t[0];
                        sw[a][1] = t[1];
                        sw[a][2] = t[2];
#pragma unroll
                        for (int i = 0; i < 8; ++i) dw[a][i] = wd[lrow(a, 2 * i + hi) * kMaxSc + (kb >> 3)];
                    }
                }
            } else {
#pragma unroll
                for (int a = 0; a < 2; ++a)
#pragma unroll
                    for (int i = 0; i < 8; ++i) dw[a][i] = wd[lrow(a, 2 * i + hi) * kMaxSc + kb];
            }
            w11_i4 A[2][2];                                    // [tile][k-step]
#pragma unroll
            for (int a = 0; a < 2; ++a) {
                if constexpr (GU) {
                    // ggml's dequantize_row_q3_K: sub-block sb of the super-block reads 16 bytes of qs at
                    // 32 * half + 16 * gg with the 2-bit field j, and bit j + 4 * half of 16 bytes of hmask
                    const uint8_t* q = arow[a] + (size_t) 110 * (kb >> 3);
#pragma unroll
                    for (int ks = 0; ks < 2; ++ks) {
                        const int sb = 2 * (kb & 7) + ks, half = sb >> 3, j = (sb >> 1) & 3, gg = sb & 1, hb = j + 4 * half;
                        const uint32_t byte = (sw[a][(sb & 7) >> 2] >> (8 * (sb & 3))) & 0xFFu;
                        const uint32_t nib = (byte >> (4 * (sb >> 3))) & 0xFu;
                        const uint32_t hi2 = (sw[a][2] >> (8 * (sb & 3) + 2 * (sb >> 2))) & 3u;
                        const uint32_t S = (uint32_t) ((int) (nib | (hi2 << 4)) - 32) & 0xFFu;   // scale - 32, a byte
                        // (4 - code) * S = c * S - 3 S with c = 7 - code: the three bit planes of c select S, 2 S, 4 S
                        const uint32_t r0 = S * 0x01010101u, r1 = ((2u * S) & 0xFFu) * 0x01010101u,
                                       r2 = ((4u * S) & 0xFFu) * 0x01010101u, rk = ((0u - 3u * S) & 0xFFu) * 0x01010101u;
                        uint32_t qv[4], hv[4];
                        ld16(q + 32 + 32 * half + 16 * gg, qv);
                        ld16(q + 16 * gg, hv);
#pragma unroll
                        for (int m = 0; m < 4; ++m) {
                            const uint32_t c = (((qv[m] >> (2 * j)) & 0x03030303u) | (((hv[m] >> hb) & 0x01010101u) << 2)) ^
                                               0x07070707u;
                            const uint32_t p0 = ((c & 0x01010101u) * 0xFFu) & r0;
                            const uint32_t p1 = (((c >> 1) & 0x01010101u) * 0xFFu) & r1;
                            const uint32_t p2 = (((c >> 2) & 0x01010101u) * 0xFFu) & r2;
                            A[a][ks][m] = (int) add8(add8(p0, p1), add8(p2, rk));
                        }
                    }
                } else {
                    // an IQ4_NL block: fp16 d, qs[16].  Value j is the low nibble of qs[j], value 16 + j the high.
                    // The block is 2-byte aligned: qs is either word aligned, or it starts 2 bytes into a word
                    const uint8_t* q = arow[a] + (size_t) 18 * kb;
                    uint32_t n[4];
                    if ((uintptr_t) q & 2) {
                        const uint32_t* c = reinterpret_cast<const uint32_t*>(q + 2);
#pragma unroll
                        for (int m = 0; m < 4; ++m) n[m] = c[m];
                    } else {
                        const uint32_t* c = reinterpret_cast<const uint32_t*>(q);
                        const uint32_t v[5] = {c[0], c[1], c[2], c[3], *reinterpret_cast<const uint16_t*>(q + 16)};
#pragma unroll
                        for (int m = 0; m < 4; ++m) n[m] = (v[m] >> 16) | (v[m + 1] << 16);
                    }
#pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        A[a][0][m] = (int) iq4nl_word(n[m] & 0x0F0F0F0Fu);
                        A[a][1][m] = (int) iq4nl_word((n[m] >> 4) & 0x0F0F0F0Fu);
                    }
                }
            }
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                if (16 * nt >= nrows) continue;                // the tile has no rows here
                w11_i4 B[2];
                float dx;
                if constexpr (GU) {                            // a q8_1 block: {fp16 d, fp16 sum; int8 qs[32]}
                    const uint32_t* q = reinterpret_cast<const uint32_t*>(brow[nt] + (size_t) 36 * kb);
                    dx = __half2float(__ushort_as_half((unsigned short) (q[0] & 0xFFFFu)));
#pragma unroll
                    for (int ks = 0; ks < 2; ++ks)
#pragma unroll
                        for (int j = 0; j < 4; ++j) B[ks][j] = (int) q[1 + 4 * ks + j];
                } else {                                       // an H chunk: 64 codes, then the two blocks' fp16 d
                    const uint8_t* q = brow[nt] + (size_t) kHChunk * (kb >> 1);
                    const uint32_t* c = reinterpret_cast<const uint32_t*>(q + 32 * (kb & 1));
                    dx = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(q + 64 + 2 * (kb & 1))));
#pragma unroll
                    for (int ks = 0; ks < 2; ++ks)
#pragma unroll
                        for (int j = 0; j < 4; ++j) B[ks][j] = (int) c[4 * ks + j];
                }
#pragma unroll
                for (int a = 0; a < 2; ++a) {
                    w11_i8 c = w11_i8{0, 0, 0, 0, 0, 0, 0, 0};
                    c = wmma_s8(A[a][0], B[0], c);
                    c = wmma_s8(A[a][1], B[1], c);             // the block's exact integer dots
#pragma unroll
                    for (int i = 0; i < 8; ++i) acc[a][nt][i] = fmaf(dw[a][i] * dx, (float) c[i], acc[a][nt][i]);
                }
            }
        }
        if constexpr (GU) {
            // a q8_1 block of H is 32 features: this wave's 16 and those of wave ^ 1.  A lane holds 8 of a tile row's
            // features (2 i + hi) and lane ^ 16 the other 8.  The scale and the codes are as q8_1_store makes them.
            float h[4][8];
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                float am = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float gt = acc[0][nt][i] * s_gate, up = acc[1][nt][i] * s_up;
                    h[nt][i] = gt / (1.0f + expf(-gt)) * up;
                    am = fmaxf(am, fabsf(h[nt][i]));
                }
                am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 16));
                if (hi == 0) ham[wv * kTile + 16 * nt + l16] = am;
            }
            __syncthreads();
            const int f0 = 16 * wv;                            // the wave's first feature in the item
            const int hh = (f0 >> 5) & 1;                      // its block in the 64-feature chunk
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int r = 16 * nt + l16;
                if (r >= nrows) continue;
                const float am = fmaxf(ham[wv * kTile + r], ham[(wv ^ 1) * kTile + r]);
                const float d = strata::kernels::q8_1_finite(am / 127.0f);
                uint8_t* o = out + (size_t) (row0 + r) * hpitch + (size_t) (2 * fb + (f0 >> 6)) * kHChunk;
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    o[(f0 & 63) + 2 * i + hi] = (uint8_t) strata::kernels::q8_1_quant(h[nt][i], d, am);
                if (hi == 0 && (wv & 1) == 0) *reinterpret_cast<__half*>(o + 64 + 2 * hh) = __float2half(d);
            }
        } else {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int r = 16 * nt + l16;
                if (r >= nrows) continue;
                float* o = dm + (size_t) (row0 + r) * (size_t) ell + kRows * fb + 32 * wv + hi;
#pragma unroll
                for (int a = 0; a < 2; ++a)
#pragma unroll
                    for (int i = 0; i < 8; ++i) o[16 * a + 2 * i] = acc[a][nt][i] * s_down;
            }
        }
    }
#else
    __builtin_trap();
#endif
}

struct DevInfo {
    bool done = false, ok = false;
    int sms = 0, occ_gu = 0, occ_d = 0;
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
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&d.occ_gu, latent_w11_kernel<true>, kThreads, 0) != hipSuccess ||
        hipOccupancyMaxActiveBlocksPerMultiprocessor(&d.occ_d, latent_w11_kernel<false>, kThreads, 0) != hipSuccess)
        d.occ_gu = d.occ_d = 0;
    cudaGetLastError();
    d.ok = d.sms > 0 && d.occ_gu >= 1 && d.occ_d >= 1;
#endif
    return d;
}

}  // namespace

bool available() { return dev_info().ok; }

void experts(const fused::WideBatch& b, const fused::LatentGeom& g, const int32_t* cnt, const int32_t* off,
             const int32_t* ts, const void* tiles, int64_t n, const void* xa, const int32_t* src, void* hbuf, float* dm,
             void* stream) {
    if (b.e1 <= b.e0 || n <= 0) return;
    const DevInfo& d = dev_info();
    // the geometry these kernels take: Q3_K gate/up rows of whole 256-value super-blocks, IQ4_NL down rows
    const bool geom = d.ok && g.ell > 0 && g.ell % 256 == 0 && (g.ell >> 8) <= kMaxSc && g.iff > 0 && g.iff % 128 == 0 &&
                      (g.iff >> 5) <= kMaxSc && g.gu_row == (size_t) (g.ell / 256) * 110 &&
                      g.d_row == (size_t) (g.iff / 32) * 18;
    if (!geom) {
        std::fprintf(stderr,
                     "prefill latent experts (gfx11): not available here (ell %lld, iff %lld, rows of %zu and %zu bytes)\n",
                     (long long) g.ell, (long long) g.iff, g.gu_row, g.d_row);
        std::exit(1);
    }
    const cudaStream_t s = (cudaStream_t) stream;
    const Tabs tb{cnt, off, ts, (const int2*) tiles};
    // the most tiles the batch can have: every row in it, plus a partial tile per expert
    const int64_t tiles_max = (n + kTile - 1) / kTile + (b.e1 - b.e0);
    const unsigned g_gu =
        (unsigned) std::min<int64_t>(tiles_max * (g.iff / 128), (int64_t) d.sms * wgp_blocks(d.occ_gu));
    const unsigned g_d =
        (unsigned) std::min<int64_t>(tiles_max * (g.ell / kRows), (int64_t) d.sms * wgp_blocks(d.occ_d));
    latent_w11_kernel<true><<<g_gu, kThreads, 0, s>>>(b, tb, g, (const uint8_t*) xa, src, (uint8_t*) hbuf, nullptr);
    ck(cudaGetLastError(), "gate/up");
    latent_w11_kernel<false><<<g_d, kThreads, 0, s>>>(b, tb, g, (const uint8_t*) hbuf, src, nullptr, dm);
    ck(cudaGetLastError(), "down");
}

}  // namespace strata::prefill::q3k_w11
