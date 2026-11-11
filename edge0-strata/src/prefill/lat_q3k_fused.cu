// src/prefill/lat_q3k_fused.cu - see include/strata/prefill/lat_q3k_fused.hpp.
//
// M3 of the q3k latent A stage: the whole A stage in one call - gate/up (Q3_K), SwiGLU, the q8_1 of H, then the down
// (IQ4_NL) - in TWO launches, the shape experts_latent has: the gate/up kernel's epilogue is moe_fused.cu's
// latent_kernel<true> epilogue (SwiGLU + q8_1 into the staged H rows), and the down is the shipped lat_q3k_down
// kernel called as it is.  Nothing between them goes through the host.
//
// The gate/up half here is lat_q3k_gu.cu's kernel (frozen, sha256 53a56961...) with ONE difference, its epilogue:
// M2a stores the products to gate[]/up[] and M2b quantizes them in a second pass, this one keeps the products in
// registers and runs SwiGLU + q8_1 there (exactly as latent_kernel<true>: the block's amax over the warp's mg
// dimension via xor 16/8/4 and the four i in registers, `h = __fmul_rn(__fdiv_rn(gt, 1 + __expf(-gt)), up)`,
// `q8_1_finite(amax / 127)`, `q8_1_quant`), which is byte for byte what M2b's warp-per-block quantizer writes.
// Everything else - the decode, the stage buffers, the mma, the work item, the grid - is that file's, unchanged.
//
// The staged weight row built from Q3_K instead of NVFP4:
//   staged row (80 B): 4 x 16 int8 codes at 16 * s, four fp32 sub-scales at 64 + 4 * s
//   a lane multiplies word k, byte c of its weight sub-block with word k, byte c of the activation sub-block, so both
//   hold the sub-block's value 4c + k; the sub-block's scale multiplies the whole 16-value product.
//
// The decode.  block_q3_K (110 B, 256 values): hmask[32], qs[64], scales[12], half d.  Value v of the super-block is
//   code(v) = ((qs[B] >> 2t) & 3) - 4 + 4 * ((hmask[B] >> m) & 1)                     in [-4, 3]
//   scale   = d * (sc[s] - 32),  s = v / 16,  sc = the 6-bit scale of that sub-block
// with B = 32 * (v / 128) + 16 * ((v / 16) & 1) + (v & 15), t = (v / 32) & 3 and m = t + 4 * (v / 128) - exactly
// ggml-quants.c's dequantize_row_q3_K (:1305): its 16 sub-blocks per 128-value half, its `shift += 2; m <<= 1`.
// The 16 6-bit scales come from the 12 scale bytes the same way: sub-block s takes the low (s < 8) or high nibble of
// byte s & 7 and the two high bits 2 * (s / 4) of byte 8 + (s & 3).
// A 64-value chunk of a row is four sub-blocks: 32 qs bytes and 32 hmask bytes of one half at ONE 2-bit field.  Two
// threads a row: thread `whalf` takes the two sub-blocks of field j = 2 * (chunk & 1) + whalf.
//
// The rows are 550 B apart, so an address inside a row is 4-byte aligned or 2 mod 4: the 16-byte groups are read with
// one funnel shift (lat_ld4).  The reads can reach up to 6 bytes past a row's last scale byte, which is inside the
// blob (the down section follows the gate/up rows) and never past the blob.
#include "strata/prefill/lat_q3k_fused.hpp"
#include "strata/prefill/lat_q3k_down.hpp"
#include "strata/kernels/q8_1_finite.hpp"

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
        std::fprintf(stderr, "lat q3k fused: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

constexpr int64_t kNE = 1280;          // K (ell): 5 Q3_K super-blocks
constexpr int64_t kFF = 640;           // gate/up rows per projection
constexpr size_t kUpOff = 352000;      // native_expert_layout(11, 20, 1280, 640)
constexpr size_t kGuRow = 550;         // 5 x 110 B
constexpr int kChunks = 20;            // 1280 / 64
constexpr int kItems = 5;              // 640 features / 128 a work item

constexpr int LTHREADS = 512;
constexpr int WROWS = 256;             // 128 gate rows + their 128 up rows
constexpr int LWPITCH = 80;            // 4 x 16 codes, 4 fp32 sub-scales
constexpr int LAPITCH = 80;            // 64 activation codes, {d0, d1}, pad
constexpr int kTileRows = 64;
constexpr int kXqRow = 1440;           // 40 q8_1 blocks of 36 B
constexpr int kHqRow = 800;            // a staged H row: 10 chunks of 80 B (n_ff / 32 = 20 blocks)

inline size_t lat_smem_bytes() {
    return (size_t) 2 * WROWS * LWPITCH + (size_t) 2 * kTileRows * LAPITCH + (size_t) kTileRows * 4;
}

#if !defined(__HIPCC__)

// 16 bytes at p as four words; `sh` is 0 or 16, from the row's 2-byte misalignment (550 B rows)
__device__ __forceinline__ void lat_ld4(const uint8_t* p, int sh, uint32_t (&w)[4]) {
    const uint32_t* q = (const uint32_t*) (p - (sh >> 3));
    w[0] = __funnelshift_r(q[0], q[1], sh);
    w[1] = __funnelshift_r(q[1], q[2], sh);
    w[2] = __funnelshift_r(q[2], q[3], sh);
    w[3] = __funnelshift_r(q[3], q[4], sh);
}

// One 16-value sub-block's 4 code words (A[m] byte c = the sub-block's value 4m + c) into the staged order: out[k]'s
// byte c is the sub-block's value 4c + k (the order the staged activations use, so both mma operands agree).
__device__ __forceinline__ void lat_stage_codes(const uint32_t* A, uint32_t (&out)[4]) {
#pragma unroll
    for (int k = 0; k < 4; ++k)
        out[k] = __byte_perm(__byte_perm(A[0], A[1], k | ((k + 4) << 4)),
                             __byte_perm(A[2], A[3], k | ((k + 4) << 4)), 0x5410u);
}

// This thread's two sub-blocks of one 64-value chunk, staged.
struct GuStaged {
    uint32_t w[2][4];
    float sc[2];
};

__device__ __forceinline__ GuStaged lat_gu_decode(const uint8_t* blk, int kb, int whalf) {
    const int cl = kb & 3;                        // the chunk inside the 256-value super-block
    const int half = cl >> 1;                     // the dequantizer's n = 128 * half
    const int j = 2 * (cl & 1) + whalf;           // the 2-bit field (and, with +4*half, the hmask bit)
    const int hb = j + 4 * half;                  // the hmask bit of this thread's two sub-blocks
    const int sh = 8 * ((int) ((uintptr_t) blk & 3));
    uint32_t qg[2][4], hg[2][4], sw[4];
    lat_ld4(blk + 32 + 32 * half + 0, sh, qg[0]);     // the even sub-block's 16 qs bytes
    lat_ld4(blk + 32 + 32 * half + 16, sh, qg[1]);    // the odd one's
    // the hmask is NOT advanced across the 128-value halves (ggml reads hm[l] / hm[l + 16] both times, the bit
    // j + 4 * half picking the half): its two 16-byte groups are 0 and 16
    lat_ld4(blk + 0, sh, hg[0]);
    lat_ld4(blk + 16, sh, hg[1]);
    lat_ld4(blk + 96, sh, sw);                        // scales[12] then d (and 2 pad bytes)
    const float d = __half2float(__ushort_as_half((unsigned short) (sw[3] & 0xFFFFu)));
    GuStaged r;
#pragma unroll
    for (int g = 0; g < 2; ++g) {                 // sub-block s_local = 2 * whalf + g
        const int s = 4 * cl + 2 * whalf + g;
        uint32_t nat[4];
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const uint32_t lo2 = (qg[g][m] >> (2 * j)) & 0x03030303u;             // 2 low bits per value
            const uint32_t hi1 = (hg[g][m] >> hb) & 0x01010101u;                  // the high bit -> 4
            // the -4 offset ggml's vec_dot folds in with a second dp4a, folded into the code here: [0, 7] - 4 =
            // [-4, 3].  Byte-wise, not a 32-bit subtract: a byte underflow would borrow from its neighbour.
            nat[m] = __vsub4(lo2 | (4u * hi1), 0x04040404u);
        }
        lat_stage_codes(nat, r.w[g]);
        // the sub-block's 6-bit scale: nibble (low/high per half) of byte s & 7, two high bits of byte 8 + (s & 3)
        const uint32_t nib_byte = (sw[(s & 7) >> 2] >> (8 * (s & 3))) & 0xFFu;
        const uint32_t nib = (nib_byte >> (4 * (s >> 3))) & 0xFu;
        const uint32_t hi2 = (sw[2] >> (8 * (s & 3) + 2 * (s >> 2))) & 3u;
        r.sc[g] = d * (float) ((int) (nib | (hi2 << 4)) - 32);
    }
    return r;
}

// ---- the kernel: 128 gate rows and their up rows per work item ------------------------------------------------
__global__ void __launch_bounds__(LTHREADS, 1)
lat_q3k_fused_gu_kernel(const uint8_t* const* __restrict__ blob, const int32_t* __restrict__ starts, int n_groups,
                        const uint8_t* __restrict__ xq, uint8_t* __restrict__ hbuf,
                        const int32_t* __restrict__ counts, const int32_t* __restrict__ ent_tok) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t* wst[2] = {smem, smem + WROWS * LWPITCH};
    uint8_t* ast[2] = {smem + 2 * WROWS * LWPITCH, smem + 2 * WROWS * LWPITCH + kTileRows * LAPITCH};
    int* srow = (int*) (smem + 2 * WROWS * LWPITCH + 2 * kTileRows * LAPITCH);

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, mg = lane >> 2, tig = lane & 3;
    const int wf = warp & 3, nb0 = 16 * (warp >> 2);
    const int wrow = tid >> 1, whalf = tid & 1;         // 2 threads a weight row, one 32-value half each
    const int arow = tid >> 1, ahalf = tid & 1;

    const int fb = blockIdx.x;                          // this launch's 128 features of the item
    for (int g = blockIdx.y; g < n_groups; g += gridDim.y) {
        const uint8_t* bl = blob[g];
        if (bl == nullptr) continue;
        const int e0 = starts[g];
        // the group's row count: counts[] when the caller has it (the engine's flushes are not in group order)
        const int e1 = counts != nullptr ? e0 + counts[g] : starts[g + 1];
        for (int row0 = e0 + (int) blockIdx.z * kTileRows; row0 < e1; row0 += (int) gridDim.z * kTileRows) {
            const int nrows = min(kTileRows, e1 - row0);
            const int arows = min(kTileRows, (nrows + 15) & ~15);
            // the stage's row: gate rows 128 * fb .. , then their up rows 128 rows later
            const int fr = 128 * fb + (wrow & 127);
            const uint8_t* wsrc = wrow < 128 ? bl + (size_t) fr * kGuRow : bl + kUpOff + (size_t) fr * kGuRow;
            __syncthreads();
            // the activation row of a tile row: the entry itself, or the token row ent_tok gives it (H stays
            // entry-indexed: row0 + r is the entry)
            if (tid < kTileRows) {
                const int er = row0 + min(tid, nrows - 1);
                srow[tid] = ent_tok != nullptr ? ent_tok[er] : er;
            }
            __syncthreads();

            GuStaged wc[2];
            uint32_t aq[8], adw = 0;
            auto load_chunk = [&](int kb) {
                const uint8_t* blk = wsrc + 110 * (kb >> 2);
                wc[kb & 1] = lat_gu_decode(blk, kb, whalf);
                if (tid < 2 * arows) {
                    // the plain q8_1 row: 72 B a chunk (two 32-value blocks), word 0 of a block is its ds
                    const uint32_t* ap = (const uint32_t*) (xq + (size_t) srow[arow] * kXqRow + (size_t) 72 * kb);
#pragma unroll
                    for (int i = 0; i < 8; ++i) aq[i] = ap[1 + 9 * ahalf + i];
                    adw = ap[9 * ahalf];
                }
            };
            auto store_chunk = [&](int kb) {
                uint8_t* wr = wst[kb & 1] + (size_t) wrow * LWPITCH;
                const GuStaged& w = wc[kb & 1];
#pragma unroll
                for (int gg = 0; gg < 2; ++gg) {
                    const int s = 2 * whalf + gg;
#pragma unroll
                    for (int k = 0; k < 4; ++k) *(uint32_t*) (wr + 16 * s + 4 * k) = w.w[gg][k];
                    *(float*) (wr + 64 + 4 * s) = w.sc[gg];
                }
                if (tid < 2 * arows) {
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
                    *(__half*) (ar + 64 + 2 * ahalf) = __ushort_as_half((unsigned short) (adw & 0xFFFFu));
                }
            };

            load_chunk(0);
            store_chunk(0);
            if (kChunks > 1) load_chunk(1);
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
                    const uint8_t* wc8 = wst[kb & 1];
                    const uint8_t* ac = ast[kb & 1];
                    for (int s = 0; s < 4; ++s) {
                        uint32_t bq[2];
                        float dx0[2], dx1[2];
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
                            const int ra = 32 * wf + 8 * i + mg;          // the gate row (its up row is ra + 128)
                            const int rb = ra + 128;
                            const uint32_t a0 = *(const uint32_t*) (wc8 + (size_t) ra * LWPITCH + 16 * s + 4 * tig);
                            const uint32_t a1 = *(const uint32_t*) (wc8 + (size_t) rb * LWPITCH + 16 * s + 4 * tig);
                            const float da = *(const float*) (wc8 + (size_t) ra * LWPITCH + 64 + 4 * s);
                            const float db = *(const float*) (wc8 + (size_t) rb * LWPITCH + 64 + 4 * s);
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
                    store_chunk(kb + 1);
                    if (kb + 2 < kChunks) load_chunk(kb + 2);
                }
            }
            if (!on0) continue;
            // SwiGLU of feature 32 * (4 * fb + wf) + 8 i + mg, H to q8_1 as q8_1_store (the amax over the warp's 32
            // features, d = q8_1_finite(amax / 127), the code rounding) into the staged row the down half reads -
            // moe_fused.cu's latent_kernel<true> epilogue, with no NVFP4 tail (the q3k pack has none: scales 1)
            const int blk = 4 * fb + wf;
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
                for (int c = 0; c < 2; ++c) {
                    float h[4], am = 0.0f;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float gt = acc[i][n][c], uv = acc[i][n][2 + c];
                        h[i] = __fmul_rn(__fdiv_rn(gt, 1.0f + __expf(-gt)), uv);
                        am = fmaxf(am, fabsf(h[i]));
                    }
                    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 16));
                    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 8));
                    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4));
                    const float d = strata::kernels::q8_1_finite(__fdiv_rn(am, 127.0f));
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    if (r < nrows) {
                        uint8_t* o = hbuf + (size_t) (row0 + r) * kHqRow + (size_t) (blk >> 1) * LAPITCH;
                        const int hh = blk & 1;
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const int v = 8 * i + mg;
                            o[16 * (2 * hh + (v >> 4)) + 4 * (v & 3) + ((v & 15) >> 2)] =
                                (uint8_t) strata::kernels::q8_1_quant(h[i], d, am);
                        }
                        if (mg == 0) *(__half*) (o + 64 + 2 * hh) = __float2half(d);
                    }
                }
            }
        }
    }
#endif
}

#endif  // !__HIPCC__

int lat_sms() {
    static const int sms = [] {
        int dev = 0, n = 0;
        return cudaGetDevice(&dev) == cudaSuccess &&
                       cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess && n > 0
                   ? n
                   : 0;
    }();
    return sms;
}

}  // namespace

void lat_q3k_fused(const uint8_t* const* blob, const int32_t* starts, int n_groups, const void* xq, void* hbuf,
                   float* out, int64_t ld, void* stream, const int32_t* counts, const int32_t* ent_tok,
                   const int32_t* ent_dst) {
    if (n_groups <= 0 || blob == nullptr || starts == nullptr || xq == nullptr || hbuf == nullptr || out == nullptr)
        return;
#if defined(__HIPCC__)
    std::fprintf(stderr, "lat q3k fused: the latent A stage's kernels are CUDA-only\n");
    std::exit(1);
#else
    static const bool once = [] {
        dck(cudaFuncSetAttribute(lat_q3k_fused_gu_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int) lat_smem_bytes()), "smem");
        return true;
    }();
    (void) once;
    const int sms = lat_sms();
    const unsigned gy = (unsigned) std::min(n_groups, 512);
    const unsigned gz = (sms > 0 && (int) (gy * kItems) < sms) ? 8u : 1u;
    lat_q3k_fused_gu_kernel<<<dim3((unsigned) kItems, gy, gz), LTHREADS, lat_smem_bytes(), (cudaStream_t) stream>>>(
        blob, starts, n_groups, (const uint8_t*) xq, (uint8_t*) hbuf, counts, ent_tok);
    dck(cudaGetLastError(), "launch gu+swiglu+q8_1");
    // the down: the shipped M1 kernel, on the H rows just written (same stream, so ordered)
    lat_q3k_down(blob, starts, n_groups, hbuf, out, ld, (void*) stream, counts, ent_dst);
#endif
}

}  // namespace strata::prefill
