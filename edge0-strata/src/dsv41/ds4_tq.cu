// Strata. The trellis routed experts (TQK6 / TQK7 / TQK5X / TQK4X / TQK5T / TQK4T, ds4_iq.cuh's is_tq) on the device for the
// DeepSeek-V4.1 engine path.  The weights stay as the file has them (a row of k values = k / 128 blocks of {fp16 d; 4 K stream
// bytes}); the blob in VRAM is the file's bytes, so the expert cache, the tensor-parallel cuts and the CPU lane read the same
// layout.  Arithmetic as the CPU lane's (ds4_tq_cpu.cpp): the activations are int8 with an F32 scale per 32 values (the
// engine's q8_quant), the weights exact (fp16 codebook values times the block's fp16 scale); the sums are F32.
//
// gfx12 (RDNA4, R9700): v_wmma_f32_16x16x16_f16.  The weights are the B operand (16 rows of a tile = the 16 columns, a lane
// holds one row's 8 consecutive k), the activations the A operand (16 entries = the 16 rows, split x = hi + lo in two fp16
// so the product keeps ~2^-22), D[entry][weight row] in F32: a lane holds the entries 8 kg .. 8 kg + 7 of its row, so the
// block scale d is the lane's own.  Lane l = r + 16 kg owns row r and the 16 steps 16 kg .. 16 kg + 15 of every block, whose
// 15-bit states are one 128-bit window of the stream (the tile code of tools/dsv41_lab/wmma_emu.hpp, checked there against a
// double reference and on the card by tests/dsv41/tqw_hip_test.cpp).  Elsewhere the same lane mapping in plain FMAs.
#include "ds4_launch.h"   // strata_ds4_launch: <<<>>> on CUDA and HIP, a dispatch on Metal
#include "ds4_iq.cuh"

#include <cuda_fp16.h>

#include "ggml-common-tq.h"   // tq_lut_f16 (the 2048-point codebook as fp16 bit patterns, __device__)

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#if defined(STRATA_USE_HIP) && !defined(DS4_TQ_PORTABLE) && (defined(__GFX12__) || defined(__gfx1200__) || defined(__gfx1201__))
#define DS4_TQ_WMMA 1
#endif

#if !defined(STRATA_USE_HIP) && defined(__CUDACC__)   // (HIP's non-sync shuffle name, for the portable lane on CUDA)
#define __shfl_xor(v, m) __shfl_xor_sync(0xffffffffu, (v), (m))
#endif
namespace strata::ds4::gguf::iq::tq {
namespace {

constexpr int kWaves = 4, kThreads = kWaves * 32;   // a block: 4 waves, a wave = one 16-row tile of each matrix
constexpr int kTileRows = 16;
constexpr int kMaxEnt = 16;   // entries (columns) a pass

#if defined(DS4_TQ_WMMA)
typedef _Float16 half8_t __attribute__((ext_vector_type(8)));
typedef float float8_t __attribute__((ext_vector_type(8)));
#endif

constexpr bool computed_code(int type) { return type == kTQK5X || type == kTQK4X; }

// ---------------------------------------------------------------- the codebook and the stream
// a step's 15-bit state -> the four weights as two words of two fp16 (lo, hi): w0 w1 | w2 w3
template <bool COMPUTED>
__device__ __forceinline__ void step_words(const uint32_t* lut, uint32_t s, uint32_t& w0, uint32_t& w1) {
    if (COMPUTED) {
        w0 = ((s * 89226354u + 64248484u) & 0x8FFF8FFFu) ^ 0x3B603B60u;
        w1 = ((s * 0x9E3779B1u + 0x7F4A7C15u) & 0x8FFF8FFFu) ^ 0x3B603B60u;
    } else {
        const uint32_t x = s * 0x9e3779b1u;
        w0 = lut[x >> 21];
        w1 = lut[(x >> 10) & 2047u];
    }
}
__device__ __forceinline__ float half_bits_f(uint32_t b) { return __half2float(__ushort_as_half((unsigned short) b)); }

// the 128-bit window of a lane's 16 steps in one block record: record = {u16 d; stream}, lane half kg takes stream bits
// [(16 - 16 kg) K, (32 - 16 kg) K) plus the 15 - K bits after them (circular), so state(u), u = 0..15, is the 15 bits at
// (15 - u) K of the window.
template <int K>
__device__ __forceinline__ void stream_window(const uint8_t* rec, int kg, uint64_t& lo, uint64_t& hi) {
    const uint16_t* st = (const uint16_t*) (rec + 2);
    constexpr int NW = (15 * K + 15 + 15) / 16;   // u16 words that hold 15 K + 15 bits
    uint64_t w[2] = {0ull, 0ull};
    if (kg == 1) {
#pragma unroll
        for (int i = 0; i < NW; ++i) w[i >> 2] |= (uint64_t) st[i] << (16 * (i & 3));
    } else {   // the stream's upper half [16 K, 32 K) = K words; the bits past its end wrap to the stream's start
        const uint16_t* up = st + K;
#pragma unroll
        for (int i = 0; i < K; ++i) w[i >> 2] |= (uint64_t) up[i] << (16 * (i & 3));
        const uint64_t wrap = (uint64_t) st[0] & ((1ull << (15 - K)) - 1ull);   // 15 - K <= 11 bits: inside word 0
        const int p = 16 * K;
        w[p >> 6] |= wrap << (p & 63);   // (p & 63) + 15 - K <= 56: never past the word
    }
    lo = w[0];
    hi = w[1];
}
__device__ __forceinline__ uint32_t state_at(uint64_t lo, uint64_t hi, int pos) {   // 15 bits at bit pos of the 128-bit window
    uint64_t v;
    if (pos + 15 <= 64) v = lo >> pos;
    else if (pos >= 64) v = hi >> (pos - 64);
    else v = (lo >> pos) | (hi << (64 - pos));
    return (uint32_t) v & 0x7fffu;
}

// ---------------------------------------------------------------- activations: where a lane's 8 values of an entry come from
// Each loader: valid(m) says entry m (this pass's column) exists; load8(m, k0, v) the 8 values k0..k0+7 as F32.
struct ActQ8 {   // int8 rows with an F32 scale a 32: entry m = row rows[m] of q (n values a row), scales d (n / 32 a row)
    const int8_t* q;
    const float* d;
    const int* rows;
    int n;
    __device__ __forceinline__ void load8(int m, int k0, float* v) const {
        const size_t r = (size_t) rows[m];
        const int2 w = *(const int2*) (q + r * n + k0);
        const float sc = d[r * (n / 32) + (k0 >> 5)];
        const int8_t* b = (const int8_t*) &w;
#pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = (float) b[j] * sc;
    }
};
struct ActF32 {   // F32 rows: entry m = row rows[m] (ld floats a row)
    const float* x;
    const int* rows;
    size_t ld;
    __device__ __forceinline__ void load8(int m, int k0, float* v) const {
        const float4* p = (const float4*) (x + (size_t) rows[m] * ld + k0);
        const float4 a = p[0], b = p[1];
        v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
    }
};
struct ActF16 {   // F16 rows: entry m = row rows[m] (ld halves a row)
    const uint16_t* x;
    const int* rows;
    size_t ld;
    __device__ __forceinline__ void load8(int m, int k0, float* v) const {
        const uint4 w = *(const uint4*) (x + (size_t) rows[m] * ld + k0);
        const uint32_t u[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[2 * j] = half_bits_f(u[j] & 0xffffu);
            v[2 * j + 1] = half_bits_f(u[j] >> 16);
        }
    }
};

// ---------------------------------------------------------------- the stream window by 32-bit loads
// The same 128-bit window as stream_window, from aligned 32-bit loads (a record is 2 + 4 K bytes: its scale, then the stream, at
// any 2-byte alignment): the array starts at the 4-byte floor of the record, the stream at bit 16 or 32 of it.  Loads past the
// end of the record are skipped (the window bits above what a lane needs are junk anyway).
template <int K>
__device__ __forceinline__ void stream_window_u32(const uint8_t* rec, int kg, uint64_t& lo, uint64_t& hi) {
    constexpr int bsz = 2 + 4 * K;
    const uintptr_t ad = (uintptr_t) rec;
    const uint32_t* pa = (const uint32_t*) (ad & ~(uintptr_t) 3);
    const int off = 16 + 8 * (int) (ad & 3);                    // the stream's first bit in the array: 16 or 32
    const int start = off + (kg ? 0 : 16 * K);
    const int w0 = start >> 5, s = start & 31;
    const int nwords = (int) (((ad & 3) + bsz + 3) >> 2);       // words that hold the record
    uint32_t v[5];
#pragma unroll
    for (int i = 0; i < 5; ++i) v[i] = (w0 + i < nwords) ? pa[w0 + i] : 0u;
    uint32_t r[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) r[i] = (uint32_t) ((((uint64_t) v[i + 1] << 32) | v[i]) >> s);
    lo = ((uint64_t) r[1] << 32) | r[0];
    hi = ((uint64_t) r[3] << 32) | r[2];
    if (!kg) {   // the upper half [16 K, 32 K) of the stream (16 K >= 64 bits); the bits past its end wrap to the stream's start
        hi &= (1ull << (16 * K - 64)) - 1ull;
        const uint32_t first = off == 16 ? (pa[0] >> 16) : pa[1];
        const uint64_t wrap = (uint64_t) first & ((1ull << (15 - K)) - 1ull);
        const int p = 16 * K - 64;
        hi |= wrap << p;
    }
}

// ---------------------------------------------------------------- NM 16-row tiles (matrices sharing the activations) over up to 16 entries
// acc_out[mi][i] (i < 8) = D[entry 8 kg + i][row r] of matrix mi's tile: sum over the row's k values of x_entry * w, k % 128 == 0.
// wrows[mi]: the tile's row r (this lane's) of matrix mi in the blob; nblk blocks.  The activation fragments are made once for all NM.
template <int K, bool COMPUTED, int NM, bool WORDS, class Act>
__device__ __forceinline__ void tile_dot(const uint8_t* const* wrows, int nblk, const uint32_t* lut, const Act& act, int ne,
                                         float (*acc_out)[8]) {
    constexpr int bsz = 2 + 4 * K;
    const int lane = threadIdx.x & 31, r = lane & 15, kg = lane >> 4;
    float tot[NM][8];
#pragma unroll
    for (int mi = 0; mi < NM; ++mi)
#pragma unroll
        for (int i = 0; i < 8; ++i) tot[mi][i] = 0.0f;
    const bool have = r < ne;   // this lane's A rows (the entry m = r)
    for (int b = 0; b < nblk; ++b) {
        float d[NM];
        uint64_t lo[NM], hi[NM];
#pragma unroll
        for (int mi = 0; mi < NM; ++mi) {
            const uint8_t* rec = wrows[mi] + (size_t) b * bsz;
            d[mi] = half_bits_f(*(const uint16_t*) rec);
            if (WORDS) stream_window_u32<K>(rec, kg, lo[mi], hi[mi]);
            else stream_window<K>(rec, kg, lo[mi], hi[mi]);
        }
#if defined(DS4_TQ_WMMA)
        float8_t acc[NM];
#pragma unroll
        for (int mi = 0; mi < NM; ++mi) acc[mi] = float8_t{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            half8_t ah, al;
            {
                float v[8];
                if (have) act.load8(r, b * 128 + 64 * kg + 8 * m, v);
                else {
#pragma unroll
                    for (int j = 0; j < 8; ++j) v[j] = 0.0f;
                }
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const _Float16 h = (_Float16) v[j];
                    ah[j] = h;
                    al[j] = (_Float16) (v[j] - (float) h);
                }
            }
#pragma unroll
            for (int mi = 0; mi < NM; ++mi) {
                half8_t wf;
#pragma unroll
                for (int s2 = 0; s2 < 2; ++s2) {
                    const int u = 2 * m + s2;
                    uint32_t w0, w1;
                    step_words<COMPUTED>(lut, state_at(lo[mi], hi[mi], (15 - u) * K), w0, w1);
                    wf[4 * s2 + 0] = __builtin_bit_cast(_Float16, (uint16_t) (w0 & 0xffffu));
                    wf[4 * s2 + 1] = __builtin_bit_cast(_Float16, (uint16_t) (w0 >> 16));
                    wf[4 * s2 + 2] = __builtin_bit_cast(_Float16, (uint16_t) (w1 & 0xffffu));
                    wf[4 * s2 + 3] = __builtin_bit_cast(_Float16, (uint16_t) (w1 >> 16));
                }
                acc[mi] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ah, wf, acc[mi]);   // D[m = entry][n = weight row]
                acc[mi] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(al, wf, acc[mi]);
            }
        }
#pragma unroll
        for (int mi = 0; mi < NM; ++mi)
#pragma unroll
            for (int i = 0; i < 8; ++i) tot[mi][i] += d[mi] * acc[mi][i];
#else
        // plain FMAs: this lane's row r, its 64 weights of the block against every entry (lane r + 16 kg' sums the halves)
        float part[NM][kMaxEnt];
#pragma unroll
        for (int mi = 0; mi < NM; ++mi)
#pragma unroll
            for (int e = 0; e < kMaxEnt; ++e) part[mi][e] = 0.0f;
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            float w[NM][8];
#pragma unroll
            for (int mi = 0; mi < NM; ++mi)
#pragma unroll
                for (int s2 = 0; s2 < 2; ++s2) {
                    uint32_t w0, w1;
                    step_words<COMPUTED>(lut, state_at(lo[mi], hi[mi], (15 - (2 * m + s2)) * K), w0, w1);
                    w[mi][4 * s2 + 0] = half_bits_f(w0 & 0xffffu);
                    w[mi][4 * s2 + 1] = half_bits_f(w0 >> 16);
                    w[mi][4 * s2 + 2] = half_bits_f(w1 & 0xffffu);
                    w[mi][4 * s2 + 3] = half_bits_f(w1 >> 16);
                }
#pragma unroll
            for (int e = 0; e < kMaxEnt; ++e)
                if (e < ne) {
                    float v[8];
                    act.load8(e, b * 128 + 64 * kg + 8 * m, v);
#pragma unroll
                    for (int mi = 0; mi < NM; ++mi)
#pragma unroll
                        for (int j = 0; j < 8; ++j) part[mi][e] = fmaf(w[mi][j], v[j], part[mi][e]);
                }
        }
        // entries 0..7 into tot[i] of the kg = 0 lane, 8..15 of the kg = 1 lane, so the output indexing matches the WMMA's
#pragma unroll
        for (int mi = 0; mi < NM; ++mi)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float p0 = part[mi][i], p1 = part[mi][8 + i];
                const float q0 = __shfl_xor_sync(0xffffffffu, p0, 16), q1 = __shfl_xor_sync(0xffffffffu, p1, 16);
                tot[mi][i] += d[mi] * (kg == 0 ? p0 + q0 : p1 + q1);
            }
#endif
    }
#pragma unroll
    for (int mi = 0; mi < NM; ++mi)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc_out[mi][i] = tot[mi][i];
}

__device__ __forceinline__ float swiglu(float gv, float uv, float limit) {   // ds4_iq.cu's
    if (limit > 0.0f) {
        gv = fminf(gv, limit);
        uv = fminf(fmaxf(uv, -limit), limit);
    }
    return __fmul_rn(__fdiv_rn(gv, __fadd_rn(1.0f, expf(-gv))), uv);
}

template <int K, bool COMPUTED>
__device__ __forceinline__ void load_lut(uint32_t* lut) {
    if (!COMPUTED) {
        // the table as the CPU decoder has it (tq_cpu.cpp's int16 state table): fp16 * 2^13 rounded half away from zero, over
        // 2^13 - exact for values >= 2^-3 and within 2^-14 below; the result is still an exact fp16 (fewer significant bits)
        for (int i = threadIdx.x; i < 2048; i += blockDim.x) {
            uint32_t pair = 0;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const float v = half_bits_f(tq_lut_f16[2 * i + j]) * 8192.0f;
                const int n = (int) (v < 0.0f ? v - 0.5f : v + 0.5f);
                pair |= (uint32_t) __half_as_ushort(__float2half_rn((float) n / 8192.0f)) << (16 * j);
            }
            lut[i] = pair;
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------- token / window experts
__device__ __forceinline__ bool group_of(const Args& a, int grp, const uint8_t*& blob, int& m0, int& m1) {
    if (a.grp_ptr == nullptr) {
        const unsigned long long p = a.plan_ptr[grp];
        if (p == 0ull) return false;
        blob = (const uint8_t*) p;
        m0 = grp;
        m1 = grp + 1;
        return true;
    }
    if (grp >= *a.n_groups) return false;
    blob = (const uint8_t*) a.grp_ptr[grp];
    m0 = a.grp_start[grp];
    m1 = a.grp_start[grp + 1];
    return true;
}

// grid (n_ff / 64, groups): a wave = a 16-row tile of the gate and of the up of the group's expert, 16 entries a pass
template <int K, bool COMPUTED, bool WORDS>
__global__ void __launch_bounds__(kThreads) gate_up_kernel(const Args a) {
    __shared__ uint32_t lut[COMPUTED ? 1 : 2048];
    __shared__ int ent[kMaxEnt], xr[kMaxEnt];
    const uint8_t* blob;
    int m0, m1;
    if (!group_of(a, blockIdx.y, blob, m0, m1)) return;   // (uniform)
    load_lut<K, COMPUTED>(lut);
    const int lane = threadIdx.x & 31, r = lane & 15, kg = lane >> 4, wave = threadIdx.x >> 5;
    const int tile = blockIdx.x * kWaves + wave;
    const int row = tile * kTileRows + r;
    const size_t rb = (size_t) (a.n_embd / 128) * (2 + 4 * K);
    const uint8_t* g = blob + a.tr[0] + (size_t) row * rb;
    const uint8_t* u = blob + a.tr[1] + (size_t) row * rb;
    for (int mm = m0; mm < m1; mm += kMaxEnt) {
        const int ne = min(kMaxEnt, m1 - mm);
        __syncthreads();
        if ((int) threadIdx.x < ne) {
            const int e = a.grp_ptr != nullptr ? a.grp_ent[mm + threadIdx.x] : mm + threadIdx.x;
            ent[threadIdx.x] = e;
            xr[threadIdx.x] = a.grp_ptr != nullptr ? e >> 3 : 0;
        }
        __syncthreads();
        if (tile * kTileRows >= a.n_ff) continue;
        const ActQ8 act{a.xq, a.xd, xr, a.n_embd};
        const uint8_t* wr[2] = {g, u};
        float gu[2][8];
        tile_dot<K, COMPUTED, 2, WORDS>(wr, a.n_embd / 128, lut, act, ne, gu);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int m = 8 * kg + i;
            if (m < ne) a.h[(size_t) ent[m] * a.n_ff + row] = swiglu(gu[0][i], gu[1][i], a.limit);
        }
    }
}

// grid (n_embd / 64, groups): the down's tile of a wave against the entries' int8 h
template <int K, bool COMPUTED, bool WORDS>
__global__ void __launch_bounds__(kThreads) down_kernel(const Args a) {
    __shared__ uint32_t lut[COMPUTED ? 1 : 2048];
    __shared__ int ent[kMaxEnt];
    const uint8_t* blob;
    int m0, m1;
    if (!group_of(a, blockIdx.y, blob, m0, m1)) return;
    load_lut<K, COMPUTED>(lut);
    const int lane = threadIdx.x & 31, r = lane & 15, kg = lane >> 4, wave = threadIdx.x >> 5;
    const int tile = blockIdx.x * kWaves + wave;
    const int row = tile * kTileRows + r;
    const size_t rb = (size_t) (a.n_ff / 128) * (2 + 4 * K);
    const uint8_t* d = blob + a.tr[2] + (size_t) row * rb;
    for (int mm = m0; mm < m1; mm += kMaxEnt) {
        const int ne = min(kMaxEnt, m1 - mm);
        __syncthreads();
        if ((int) threadIdx.x < ne) ent[threadIdx.x] = a.grp_ptr != nullptr ? a.grp_ent[mm + threadIdx.x] : mm + threadIdx.x;
        __syncthreads();
        if (tile * kTileRows >= a.n_embd) continue;
        const ActQ8 act{a.hq, a.hd, ent, a.n_ff};
        const uint8_t* wr[1] = {d};
        float dd[1][8];
        tile_dot<K, COMPUTED, 1, WORDS>(wr, a.n_ff / 128, lut, act, ne, dd);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int m = 8 * kg + i;
            if (m < ne) a.part[(size_t) ent[m] * a.n_embd + row] = dd[0][i];
        }
    }
}

// ---------------------------------------------------------------- the prompt path's sets
// grid (n_ff / 64, chunks of 16 rows, n_exp): expert e = blockIdx.z owns the sorted rows [bounds[e], bounds[e + 1]); local
// row = sorted row - row0.  gate / up: F32 x[idx[local]] (ldx) -> F16 swiglu h[local * ldo + c]; down: F16 xh[local] (ldx) ->
// F32 y[local * ldo + c].
template <int K, bool COMPUTED, bool GATE_UP, bool WORDS>
__global__ void __launch_bounds__(kThreads) set_kernel(const SetArgs a) {
    __shared__ uint32_t lut[COMPUTED ? 1 : 2048];
    __shared__ int rowsel[kMaxEnt];
    const int e = blockIdx.z;
    const int lo = a.bounds[e] - a.row0, hi = a.bounds[e + 1] - a.row0;
    const int r0 = lo + blockIdx.y * kMaxEnt;
    if (r0 >= hi) return;   // (uniform)
    load_lut<K, COMPUTED>(lut);
    const int ne = min(kMaxEnt, hi - r0);
    const int lane = threadIdx.x & 31, r = lane & 15, kg = lane >> 4, wave = threadIdx.x >> 5;
    const int N = GATE_UP ? a.n_ff : a.n_embd, Kin = GATE_UP ? a.n_embd : a.n_ff;
    const int tile = blockIdx.x * kWaves + wave;
    const int row = tile * kTileRows + r;
    const size_t rb = (size_t) (Kin / 128) * (2 + 4 * K);
    const uint8_t* blob = a.wbase + (size_t) e * a.stride;
    if ((int) threadIdx.x < ne) rowsel[threadIdx.x] = GATE_UP ? a.idx[r0 + threadIdx.x] : r0 + (int) threadIdx.x;
    __syncthreads();
    if (tile * kTileRows >= N) return;
    float res[2][8];
    if (GATE_UP) {
        const ActF32 act{a.xf, rowsel, (size_t) a.ldx};
        const uint8_t* wr[2] = {blob + a.tr[0] + (size_t) row * rb, blob + a.tr[1] + (size_t) row * rb};
        tile_dot<K, COMPUTED, 2, WORDS>(wr, Kin / 128, lut, act, ne, res);
    } else {
        const ActF16 act{a.xh, rowsel, (size_t) a.ldx};
        const uint8_t* wr[1] = {blob + a.tr[2] + (size_t) row * rb};
        tile_dot<K, COMPUTED, 1, WORDS>(wr, Kin / 128, lut, act, ne, res);
    }
    const int kgl = threadIdx.x >> 4 & 1;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int m = 8 * kgl + i;
        if (m >= ne) continue;
        const size_t lr = (size_t) (r0 + m);
        if (GATE_UP) a.h[lr * a.ldo + row] = __half_as_ushort(__float2half_rn(swiglu(res[0][i], res[1][i], a.limit)));
        else a.y[lr * a.ldo + row] = res[0][i];
    }
}

// ---------------------------------------------------------------- GEMV jobs
// grid (n / 64): y[c * ycs + r] = alpha (row r . x_c) + bias[r] for ncol (<= 16) int8 activation columns
template <int K, bool COMPUTED, bool WORDS>
__global__ void __launch_bounds__(kThreads) mv_kernel(const MvJob j) {
    __shared__ uint32_t lut[COMPUTED ? 1 : 2048];
    __shared__ int cols[kMaxEnt];
    load_lut<K, COMPUTED>(lut);
    if ((int) threadIdx.x < kMaxEnt) cols[threadIdx.x] = (int) threadIdx.x;
    __syncthreads();
    const int lane = threadIdx.x & 31, r = lane & 15, kg = lane >> 4, wave = threadIdx.x >> 5;
    const int tile = blockIdx.x * kWaves + wave;
    if (tile * kTileRows >= j.n) return;
    const int row = tile * kTileRows + r;
    const ActQ8 act{j.xq, j.xd, cols, j.k};
    const uint8_t* wr[1] = {j.w + (size_t) row * j.ld};
    float accm[1][8];
    tile_dot<K, COMPUTED, 1, WORDS>(wr, j.k / 128, lut, act, j.ncol, accm);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int m = 8 * kg + i;
        if (m < j.ncol && row < j.n) j.y[(size_t) m * j.ycs + row] = j.alpha * accm[0][i] + (j.bias != nullptr ? j.bias[row] : 0.0f);
    }
}

// ---------------------------------------------------------------- dispatch
#define DS4_TQ_SWITCH(t, F)                         \
    switch (t) {                                    \
        case kTQK6: F(6, false); break;             \
        case kTQK7: F(7, false); break;             \
        case kTQK5X: F(5, true); break;             \
        case kTQK4X: F(4, true); break;             \
        case kTQK5T: F(5, false); break;            \
        case kTQK4T: F(4, false); break;            \
        default: return false;                      \
    }

// 32-bit stream loads (STRATA_DS4_TQ_WORDS=1) or the 16-bit ones (default)
bool use_words() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_DS4_TQ_WORDS");
        return e != nullptr && std::atoi(e) != 0;   // (no gain measured on the R9700: 13.8 vs 14.2 ms a token)
    }();
    return on;
}

bool bad(const char* what) {
    std::fprintf(stderr, "ds4 tq: %s\n", what);
    return false;
}
}  // namespace

bool gate_up(const Args& a, int groups, int ne_max, cudaStream_t s) {
    const bool w = use_words();
    (void) ne_max;
    if (a.n_embd % 128 != 0 || a.n_ff % (kWaves * kTileRows) != 0) return bad("gate_up: geometry not covered");
    const dim3 grid((unsigned) (a.n_ff / (kWaves * kTileRows)), (unsigned) groups);
#define GU_(K, C) { if (w) strata_ds4_launch(gate_up_kernel<K, C, true>, grid, kThreads, 0, s, a); else strata_ds4_launch(gate_up_kernel<K, C, false>, grid, kThreads, 0, s, a); }
    DS4_TQ_SWITCH(a.tg, GU_)
#undef GU_
    return true;
}

bool down(const Args& a, int groups, int ne_max, cudaStream_t s) {
    const bool w = use_words();
    (void) ne_max;
    if (a.n_ff % 128 != 0 || a.n_embd % (kWaves * kTileRows) != 0) return bad("down: geometry not covered");
    const dim3 grid((unsigned) (a.n_embd / (kWaves * kTileRows)), (unsigned) groups);
#define DN_(K, C) { if (w) strata_ds4_launch(down_kernel<K, C, true>, grid, kThreads, 0, s, a); else strata_ds4_launch(down_kernel<K, C, false>, grid, kThreads, 0, s, a); }
    DS4_TQ_SWITCH(a.td, DN_)
#undef DN_
    return true;
}

bool set_gate_up(const SetArgs& a, cudaStream_t s) {
    const bool w = use_words();
    if (a.n_embd % 128 != 0 || a.n_ff % (kWaves * kTileRows) != 0) return bad("set_gate_up: geometry not covered");
    const dim3 grid((unsigned) (a.n_ff / (kWaves * kTileRows)), (unsigned) ((a.max_rows + kMaxEnt - 1) / kMaxEnt), (unsigned) a.n_exp);
#define SG_(K, C) { if (w) strata_ds4_launch(set_kernel<K, C, true, true>, grid, kThreads, 0, s, a); else strata_ds4_launch(set_kernel<K, C, true, false>, grid, kThreads, 0, s, a); }
    DS4_TQ_SWITCH(a.tg, SG_)
#undef SG_
    return true;
}

bool set_down(const SetArgs& a, cudaStream_t s) {
    const bool w = use_words();
    if (a.n_ff % 128 != 0 || a.n_embd % (kWaves * kTileRows) != 0) return bad("set_down: geometry not covered");
    const dim3 grid((unsigned) (a.n_embd / (kWaves * kTileRows)), (unsigned) ((a.max_rows + kMaxEnt - 1) / kMaxEnt), (unsigned) a.n_exp);
#define SD_(K, C) { if (w) strata_ds4_launch(set_kernel<K, C, false, true>, grid, kThreads, 0, s, a); else strata_ds4_launch(set_kernel<K, C, false, false>, grid, kThreads, 0, s, a); }
    DS4_TQ_SWITCH(a.td, SD_)
#undef SD_
    return true;
}

bool mv(const MvJob* jobs, int n, cudaStream_t s) {
    for (int i = 0; i < n; ++i) {
        const MvJob& j = jobs[i];
        const bool w = use_words();
        if (j.k % 128 != 0 || j.n % kTileRows != 0 || j.ncol < 1 || j.ncol > kMaxEnt) return bad("mv: geometry not covered");
        const dim3 grid((unsigned) ((j.n + kWaves * kTileRows - 1) / (kWaves * kTileRows)));
#define MV_(K, C) { if (w) strata_ds4_launch(mv_kernel<K, C, true>, grid, kThreads, 0, s, j); else strata_ds4_launch(mv_kernel<K, C, false>, grid, kThreads, 0, s, j); }
        DS4_TQ_SWITCH(j.type, MV_)
#undef MV_
    }
    return true;
}

}  // namespace strata::ds4::gguf::iq::tq
