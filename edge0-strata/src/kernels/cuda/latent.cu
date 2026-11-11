// src/kernels/cuda/latent.cu - the latent MoE's engine-side kernels (new file).
//
// Bit-identical dot to iq_mmvq(8)'s vec_dot_q8_0_q8_1: per block
//   acc += dx * dw * sum_k(xs[k] * ws[k])
// where dx is the q8_1 block's f16 scale (struct {half ds[2]; int8 qs[32]}: the first half,
// bytes 0..1; the values follow at bytes 4..35) and dw the q8_0 block's f16 scale
// ({half d; int8 qs[32]}: bytes 0..1; values at 2..33).  The per-block 32-lane sum is
// computed directly (same integer arithmetic as the dp4a lanes, no alignment assumptions).
#include "strata/kernels/latent.hpp"
#include "strata/kernels/dp4a.hpp"

#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace strata::kernels {

__global__ void latent_plan_rows_k(int32_t* tok, const int32_t* dst, const int32_t* counts, const int32_t* ids,
                                   const int32_t* labels, int E, int base_token, int G, int K) {
    const int nj = counts[1];   // all GPU entries (the VRAM share + the PCIe share)
    for (int j = threadIdx.x; j < nj; j += blockDim.x) {
        const int d = dst[j];
        const int32_t e = ids[d];
        const int g = (e >= 0 && e < E) ? labels[e] : 0;
        tok[j] = (base_token + d / K) * G + g;
    }
}

void latent_plan_rows(int32_t* tok, const int32_t* dst, const int32_t* counts, const int32_t* ids,
                      const int32_t* labels_dev, int64_t E, int base_token, int G, int K, void* stream) {
    latent_plan_rows_k<<<1, 64, 0, (cudaStream_t) stream>>>(tok, dst, counts, ids, labels_dev, (int) E, base_token, G, K);
}

// latent_u_k optimized: coalesced dp4a, one warp per output row.
// Old version: one THREAD per row with scalar byte loads and no reuse -> ~25x read
// amplification, ~0.84 ms/layer. New: lanes stride the ELL words of a row as int32
// (built from 2-byte-aligned ushort pairs, q8_0 stores {half d; i8 qs[32]} at +2),
// dp4a per word, warp-shuffle reduction. U[g] rows are read once per (route,row)
// and shared across routes through L2 as before, but now at stream bandwidth.
__device__ __forceinline__ int latent_ld4_i8(const uint8_t* p) {
    const unsigned short a = *(const unsigned short*) p;
    const unsigned short b = *(const unsigned short*) (p + 2);
    return (int) ((unsigned) a | ((unsigned) b << 16));   // a = p0|p1<<8, b<<16 = p2<<16|p3<<24
}


// Q4DU: the weight block's dot with a q8_1 activation block, fmt 8 (Q8_0) or 4 (Q4_0).
// The q8_1 block is always {f16 d; f16 s; int8 qs[32]} (36 B, values at +4); the weight block
// is {f16 d; int8 qs[32]} (34 B) for fmt 8, {f16 d; uint8 nib[16]} (18 B) for fmt 4 with
// element j in byte j's low nibble and element j+16 in its high one, value (code - 8) * d -
// the same centred codes as iq_kernels.cu's vec_dot_q4_0_q8_1 (q4_0_centred).
__device__ __forceinline__ float lat_blk_dot_q4(const uint8_t* wb, const uint8_t* xb, int j) {
    // nibble words are built from byte loads: a Q4_0 block is only 2-byte aligned inside its
    // row (18 B stride), a raw int dereference at +2 is misaligned and faults on sm_120.
    const float dw = __half2float(*(const __half*) wb);
    const float dx = __half2float(*(const __half*) xb);
    // 4 bytes at wb+2+j: each byte's low nibble is element (j+k), its high nibble element (j+k+16).
    int w0 = 0, w16 = 0, xw = 0, xw16 = 0;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int byte_ = (int) wb[2 + j + k];
        w0 |= (((byte_ & 0xF) - 8) & 0xFF) << (8 * k);          // mask: a negative code's sign
        w16 |= ((((byte_ >> 4) & 0xF) - 8) & 0xFF) << (8 * k);  // bits must not bleed into the
        xw |= (int) (uint32_t) xb[4 + j + k] << (8 * k);        // neighbouring byte of the word
        xw16 |= (int) (uint32_t) xb[4 + j + k + 16] << (8 * k); // (dp4a reads s8 itself)
    }
    return dx * dw * (float) (__dp4a(xw, w0, 0) + __dp4a(xw16, w16, 0));
}

__global__ void latent_u_k(const uint8_t* __restrict__ u_q8, const uint8_t* __restrict__ ylq,
                           const int32_t* __restrict__ ids, const int32_t* __restrict__ labels, int E, int cap_entries,
                           int ELL, int H, float* __restrict__ out, int fmt) {
    const int r = blockIdx.x;
    if (r >= cap_entries) return;
    const int32_t e = ids[r];
    const int g = (e >= 0 && e < E) ? labels[e] : 0;
    const int nb = ELL / 32;            // value blocks per row
    const int wrow = nb * (fmt == 4 ? 18 : 34);   // Q4DU: the weight row's bytes
    const uint8_t* w = u_q8 + (size_t) g * (size_t) H * (size_t) wrow;
    const uint8_t* x = ylq + (size_t) r * (size_t) (nb * 36);
    float* o = out + (size_t) r * (size_t) H;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int h = (int) (blockIdx.y * (blockDim.x >> 5)) + warp;
    if (h >= H) return;
    const uint8_t* wb = w + (size_t) h * (size_t) wrow;
    float facc = 0.0f;
    if (fmt == 4) {
        for (int wi = lane; wi < nb * 4; wi += 32) {   // 4 int4 words per block (16 nibble bytes)
            const int b = wi >> 2, j = (wi & 3) * 4;
            facc += lat_blk_dot_q4(wb + (size_t) b * 18, x + (size_t) b * 36, j);
        }
    } else {
    for (int wi = lane; wi < nb * 8; wi += 32) {
        const int b = wi >> 3, j = (wi & 7) * 4;
        const float dx = __half2float(*(const __half*) (x + (size_t) b * 36));
        const float dw = __half2float(*(const __half*) (wb + (size_t) b * 34));
        const int xw = *(const int*) (x + (size_t) b * 36 + 4 + j);
        const int ww = latent_ld4_i8(wb + (size_t) b * 34 + 2 + j);
        facc += dx * dw * (float) __dp4a(xw, ww, 0);
    }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) facc += __shfl_down_sync(0xffffffffu, facc, off);
    if (lane == 0) o[h] = facc;
}

void latent_u_grouped(const uint8_t* u_q8, const uint8_t* ylq, const int32_t* ids, const int32_t* labels_dev,
                      int64_t E, int64_t cap_entries, int ELL, int H, float* out, void* stream, int fmt) {
    if (cap_entries <= 0) return;
    if (u_q8 == nullptr || ylq == nullptr || ids == nullptr || labels_dev == nullptr || out == nullptr) {
        std::fprintf(stderr, "latent_u_grouped NULL ptr: u=%p ylq=%p ids=%p labels=%p out=%p E=%lld cap=%lld ELL=%d H=%d\n",
                     (const void*) u_q8, (const void*) ylq, (const void*) ids, (const void*) labels_dev,
                     (const void*) out, (long long) E, (long long) cap_entries, ELL, H);
        std::fflush(stderr);
        std::abort();
    }
    const int warps_per_block = 8;   // 256 threads
    const int rows_per_block = warps_per_block;
    dim3 grid((unsigned) cap_entries, (unsigned) ((H + rows_per_block - 1) / rows_per_block));
    latent_u_k<<<grid, warps_per_block * 32, 0, (cudaStream_t) stream>>>(u_q8, ylq, ids, labels_dev, (int) E,
                                                                         (int) cap_entries, ELL, H, out, fmt);
}

// knife-0 (STRATA_LATENT_U_GPU): the CPU share's U computed on the GPU and written straight into parts_.
// One block per routed row r (like latent_u_k), 8 warps.  A GPU row (r in dst[0..count)) copies hit_out;
// a CPU row computes U[labels[ids[r]]] . hq[r] with the same dp4a dot as latent_u_k (bit-compatible to ~1e-7).
__global__ void latent_u_matvec_k(const uint8_t* __restrict__ u_q8, const uint8_t* __restrict__ hq,
                                  const int32_t* __restrict__ ids, const int32_t* __restrict__ labels,
                                  const int32_t* __restrict__ dst, const int32_t* __restrict__ count,
                                  int E, int n_entries, int ELL, int H, float* __restrict__ parts, int fmt) {
    const int g = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int hout = (int) (blockIdx.y * (blockDim.x >> 5)) + warp;
    const int nb = ELL / 32, wrow = nb * (fmt == 4 ? 18 : 34);   // Q4DU
    __shared__ int c_shared;
    __shared__ uint8_t hq_s[2016];   // nb*36 = 56*36, ELL=1792 (latent) fixed
    if (threadIdx.x == 0) c_shared = *count;
    __syncthreads();
    const int c = c_shared;
    if (hout >= H) return;
    const uint8_t* wb = u_q8 + (size_t) g * (size_t) H * (size_t) wrow + (size_t) hout * (size_t) wrow;
    for (int r = 0; r < n_entries; ++r) {
        const int32_t e = ids[r];
        if (e < 0 || e >= E || labels[e] != g) continue;
        bool gpu = false;
        for (int i = 0; i < c; ++i) if (dst[i] == r) { gpu = true; break; }
        if (gpu) continue;
        const uint8_t* x = hq + (size_t) r * (size_t) (nb * 36);
        for (int i = threadIdx.x; i < nb * 36; i += blockDim.x) hq_s[i] = x[i];   // once per block
        __syncthreads();
        float facc = 0.0f;
        if (fmt == 4) {
            for (int wi = lane; wi < nb * 4; wi += 32) {
                const int b = wi >> 2, j = (wi & 3) * 4;
                facc += lat_blk_dot_q4(wb + (size_t) b * 18, hq_s + (size_t) b * 36, j);
            }
        } else {
        for (int wi = lane; wi < nb * 8; wi += 32) {
            const int b = wi >> 3, j = (wi & 7) * 4;
            const float dx = __half2float(*(const __half*) (hq_s + (size_t) b * 36));
            const float dw = __half2float(__ldg((const __half*) (wb + (size_t) b * 34)));
            const int xw = *(const int*) (hq_s + (size_t) b * 36 + 4 + j);
            const int ww = latent_ld4_i8(wb + (size_t) b * 34 + 2 + j);
            facc += dx * dw * (float) __dp4a(xw, ww, 0);
        }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) facc += __shfl_down_sync(0xffffffffu, facc, off);
        if (lane == 0) parts[(size_t) r * (size_t) H + hout] = facc;
        __syncthreads();
    }
}

void latent_u_matvec(const uint8_t* u_q8, const uint8_t* hq, const int32_t* ids, const int32_t* labels_dev,
                     const int32_t* dst, const int32_t* count, int64_t E, int64_t n_entries, int ELL, int H,
                     int G, float* parts, void* stream, int fmt) {
    if (n_entries <= 0 || G <= 0) return;
    const int warps_per_block = 8;
    dim3 grid((unsigned) G, (unsigned) ((H + warps_per_block - 1) / warps_per_block));
    latent_u_matvec_k<<<grid, warps_per_block * 32, 0, (cudaStream_t) stream>>>(
        u_q8, hq, ids, labels_dev, dst, count, (int) E, (int) n_entries, ELL, H, parts, fmt);
}

}  // namespace strata::kernels