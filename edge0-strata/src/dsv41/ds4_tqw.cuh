// Strata. gfx12 (RDNA4, R9700 = gfx1201) WMMA form of the trellis expert GEMV of the dsv41 CUDA lab (tools/dsv41_lab:
// tqk_gemv3.cuh, wmma_emu.hpp = the specification, hip/tqk_gemv_wmma.hip.cpp = its first HIP form).  Layout "TQW" (see
// wmma_emu.hpp::repack_tqw): a wave = T tiles of 16 rows, lane l = r + 16 kg owns row r and 16 consecutive steps of every
// 128-weight block; activations are the B operand (NB <= 16 columns), split x = hi + lo in two fp16 for ~2^-22 accuracy.
// CODE 0: the table codebook (K5/K6/K7 of the shipped files: two lookups into an 8 KB LDS table of half2 per step);
// CODE 1: the computed "3single" code (no table; the new K4/K5 types if they take it).
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

namespace strata::ds4::tqw {

typedef _Float16 half8_t __attribute__((ext_vector_type(8)));
typedef float float8_t __attribute__((ext_vector_type(8)));

constexpr uint32_t kMask = 0x7fffu;

struct Job {
    const uint32_t* words;     // TQW words  [tile][P][K][32 lanes]
    const uint16_t* scales;    // [tile][P][16 rows][2 blocks] fp16
    const half8_t* xb;         // prepared B fragments (prep_b)
    float* y;                  // [NB][ldy]
    int rows, in, ldy;
};

__device__ __forceinline__ void step_words(const uint32_t* lut, uint32_t s, uint32_t& w0, uint32_t& w1, int code) {
    if (code == 1) {
        w0 = ((s * 89226354u + 64248484u) & 0x8FFF8FFFu) ^ 0x3B603B60u;
        w1 = ((s * 0x9E3779B1u + 0x7F4A7C15u) & 0x8FFF8FFFu) ^ 0x3B603B60u;
    } else {
        const uint32_t x = s * 0x9e3779b1u;
        w0 = lut[x >> 21];
        w1 = lut[(x >> 10) & 2047u];
    }
}

// activations -> B fragments.  out[(((blk * 8 + m) * 2 + kg) * nb + n) * 2 + {0 = hi, 1 = lo}] = 8 halves of x[n][blk * 128 + 4 * (16 kg + 2 m) + j]
__global__ void prep_b(const float* x, int ldx, int nb, int in, half8_t* out) {
    const int nblk = in / 128;
    const size_t gid = (size_t) blockIdx.x * blockDim.x + threadIdx.x, n_all = (size_t) nblk * 8 * 2 * nb;
    if (gid >= n_all) return;
    const int n = (int) (gid % nb), kg = (int) ((gid / nb) % 2), m = (int) ((gid / (2 * nb)) % 8), blk = (int) (gid / (16 * nb));
    const float* p = x + (size_t) n * ldx + blk * 128 + 4 * (16 * kg + 2 * m);
    half8_t hi, lo;
#pragma unroll
    for (int j = 0; j < 8; ++j) { hi[j] = (_Float16) p[j]; lo[j] = (_Float16) (p[j] - (float) hi[j]); }
    out[gid * 2] = hi;
    out[gid * 2 + 1] = lo;
}

template <int K, int NB, int T, int EXACT, int CODE>
__global__ void __launch_bounds__(128) gemv(const Job* jobs, const uint32_t* __restrict__ lut_g) {
    __shared__ uint32_t lut[CODE == 0 ? 2048 : 1];
    if (CODE == 0) {
        for (int i = threadIdx.x; i < 2048; i += blockDim.x) lut[i] = lut_g[i];
        __syncthreads();
    }
    const Job job = jobs[blockIdx.y];
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31, r = lane & 15, kg = lane >> 4;
    const int tile0 = (blockIdx.x * (blockDim.x >> 5) + wave) * T;
    if (tile0 * 16 >= job.rows) return;
    const int nblk = job.in / 128, NP = nblk / 2;
    float tot[T][8];
#pragma unroll
    for (int t = 0; t < T; ++t)
#pragma unroll
        for (int i = 0; i < 8; ++i) tot[t][i] = 0.f;
    for (int P = 0; P < NP; ++P) {
        uint32_t rec[T][K];
#pragma unroll
        for (int t = 0; t < T; ++t)
#pragma unroll
            for (int j = 0; j < K; ++j) rec[t][j] = __builtin_nontemporal_load(job.words + (((size_t) (tile0 + t) * NP + P) * K + j) * 32 + lane);
#pragma unroll
        for (int bk = 0; bk < 2; ++bk) {
            half8_t bh[8], bl[8];
            if (r < NB) {
#pragma unroll
                for (int m = 0; m < 8; ++m) {
                    const half8_t* p = job.xb + (((((size_t) (2 * P + bk) * 8 + m) * 2 + kg) * NB + r) * 2);
                    bh[m] = p[0];
                    if (EXACT) bl[m] = p[1];
                }
            } else {
#pragma unroll
                for (int m = 0; m < 8; ++m) { bh[m] = half8_t{0, 0, 0, 0, 0, 0, 0, 0}; bl[m] = bh[m]; }
            }
#pragma unroll
            for (int t = 0; t < T; ++t) {
                // chunk bk of the record: bits [16 K bk, 16 K bk + 16 K); window 0 = steps u 0..7 (+ 15-K bits of the partner chunk),
                // window 1 = steps u 8..15
                constexpr int NW = K;   // (words of a record)
                auto bits = [&](int pos, int n) -> uint64_t {   // n <= 56 bits at record bit pos (pos is a compile-time constant here)
                    const int a = pos >> 5, sh = pos & 31;
                    const uint32_t w0 = rec[t][a], w1 = a + 1 < NW ? rec[t][a + 1 < NW ? a + 1 : 0] : 0u, w2 = a + 2 < NW ? rec[t][a + 2 < NW ? a + 2 : 0] : 0u;
                    // (w2:w1:w0) >> sh, low 64 bits
                    const uint64_t out = (sh == 0) ? (((uint64_t) w1 << 32) | w0) : ((((uint64_t) w1 << 32) | w0) >> sh) | ((uint64_t) w2 << (64 - sh));
                    return out & ((1ull << n) - 1ull);
                };
                const int b0 = 16 * K * bk;
                const uint64_t lo8 = bits(b0, 8 * K), hi8 = bits(b0 + 8 * K, 8 * K);
                const uint32_t mine = (uint32_t) (lo8 & 0xffffffffu);   // the bits the PARTNER needs: this lane's chunk bits [0, 15 - K)
                const uint32_t pc = __shfl_xor((int) mine, 16);
                const uint64_t win0 = hi8 | ((uint64_t) (pc & ((1u << (15 - K)) - 1u)) << (8 * K));
                const uint64_t win1 = lo8 | ((hi8 & ((1ull << (15 - K)) - 1ull)) << (8 * K));
                float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
                for (int m = 0; m < 8; ++m) {
                    half8_t a;
#pragma unroll
                    for (int s2 = 0; s2 < 2; ++s2) {
                        const int u = 2 * m + s2;
                        const uint32_t st = (uint32_t) (u < 8 ? win0 >> ((7 - u) * K) : win1 >> ((15 - u) * K)) & kMask;
                        uint32_t w0, w1;
                        step_words(lut, st, w0, w1, CODE);
                        a[4 * s2 + 0] = __builtin_bit_cast(_Float16, (uint16_t) (w0 & 0xffffu));
                        a[4 * s2 + 1] = __builtin_bit_cast(_Float16, (uint16_t) (w0 >> 16));
                        a[4 * s2 + 2] = __builtin_bit_cast(_Float16, (uint16_t) (w1 & 0xffffu));
                        a[4 * s2 + 3] = __builtin_bit_cast(_Float16, (uint16_t) (w1 >> 16));
                    }
                    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bh[m], acc);
                    if (EXACT) acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bl[m], acc);
                }
                // scales of rows 8 kg .. 8 kg + 7 of this block: 8 consecutive (row, bk) pairs of 2 -> stride 2
                const uint16_t* sp = job.scales + (((size_t) (tile0 + t) * NP + P) * 16 + 8 * kg) * 2 + bk;
#pragma unroll
                for (int i = 0; i < 8; ++i) tot[t][i] += (float) __builtin_bit_cast(_Float16, sp[2 * i]) * acc[i];
            }
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t)
        if (r < NB)
#pragma unroll
            for (int i = 0; i < 8; ++i) job.y[(size_t) r * job.ldy + (tile0 + t) * 16 + 8 * kg + i] = tot[t][i];
}

}  // namespace strata::ds4::tqw
