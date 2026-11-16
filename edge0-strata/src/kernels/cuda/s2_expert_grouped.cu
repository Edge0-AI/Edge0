// src/kernels/cuda/s2_expert_grouped.cu - R4's grouped GPU expert.  Read the header first.
//
// THE ARITHMETIC IS `src/kernels/cpu/expert.cpp`'s, over the same bytes, one warp per output row.
//
//     sum_j (c_j - 1) * d_w * xhat_j * d_x   ==   d_w * d_x * ( sum_j c_j*xhat_j  -  sum_j xhat_j )
//
// Both sums are INT8 x INT8, so both are `__dp4a` - four multiply-accumulates per instruction, exact in
// integer and with no dequantization inside the loop.  The weight scale `d_w` is fp16 (one per 64 elements)
// and the activation scale `d_x` is fp16 (one per 32 elements), so the float work is one multiply-add per
// 32-element chunk rather than one per element.
//
// **THE SUMMATION ORDER IS NOT THE CPU's AND CANNOT BE.**  Lane `L` takes chunks `L, L+32, ...` and the
// partials are reduced through shuffle; the CPU walks every chunk in order with its own accumulator shape.
// The two agree to float rounding and not to the bit, which is the same contract `s_gemv_parity` carries for
// the same reason.  `bench/micro/moe_hit_parity.cu` is the check.
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/core/geometry.hpp"
#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/f16_bits.hpp"

#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/fuse_decode.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {
namespace {

// THE BLOB'S OWN GEOMETRY, from `include/strata/kernels/cpu/expert.hpp`.  Restated as literals because that
// header is the CPU path's and this file must not silently follow it if the two ever disagree: the sizes below
// are what the CPU kernel's indexing computes, and `moe_hit_parity` compares the two end to end.
constexpr int H = 2560;
constexpr int FF = 640;
static_assert(H == strata::core::kCompiledGeometry.n_embd && FF == strata::core::kCompiledGeometry.n_ff, "s2_expert_grouped is compiled for core/geometry.hpp");
constexpr int QK = 64;                       // Q2_0's group: one fp16 scale per 64 weights
constexpr int ROW_GU = H / 4;                // 640 B of codes per gate/up row (2 bits per element)
constexpr int ROW_D = FF / 4;                // 160 B per down row
constexpr int SC_GU = H / QK;                // 40 fp16 scales per gate/up row
constexpr int SC_D = FF / QK;                // 10 per down row
constexpr size_t O_D_CODES = (size_t) 2 * FF * ROW_GU;
constexpr size_t O_GU_SCALES = O_D_CODES + (size_t) H * ROW_D;
constexpr size_t O_D_SCALES = O_GU_SCALES + (size_t) 2 * FF * SC_GU * 2;

__device__ __forceinline__ float f16_at(const uint8_t* p) {
    return __half2float(__ushort_as_half((uint16_t) (p[0] | (p[1] << 8))));
}

/// **ONE S2 ROW AGAINST A Q8_0 ACTIVATION, WARP-WIDE.**  Every lane accumulates its own float partial over a
/// strided set of 32-element chunks and the caller reduces; `chunk` indices are absolute so the caller can
/// start the lane at any offset.
///
/// Returns the lane's partial.  `codes` is `n_in/4` bytes and `scales` `n_in/QK` fp16, both for THIS row.
///
/// **`x_scales` IS R4.2h AND IT IS OPTIONAL ON PURPOSE.**  When it is non-null the activation's multiplier
/// comes from an fp32 array instead of the block's fp16 `d`.  The CPU pool multiplies by the fp32
/// `ActQ::scale` (`cpu/expert.cpp:92`), and the two disagreed by **4.761e-04 relative on 80 of 80 chunks**
/// (`bench/micro/act_quant_parity.cu`), which is what made a cache hit compute a different expert from a
/// cache miss.  Null keeps the previous fp16 behaviour **exactly**, so `moe_hit_parity` - which passes no
/// scales - still measures the kernel it always measured.
__device__ __forceinline__ float row_dot_s2_q8(const uint8_t* __restrict__ codes,
                                               const uint8_t* __restrict__ scales,
                                               const uint8_t* __restrict__ x_q8_0, int n_chunks, int lane,
                                               const float* __restrict__ x_scales = nullptr) {
    float acc = 0.0f;
    for (int c = lane; c < n_chunks; c += 32) {
        const uint8_t* cb = codes + (size_t) c * 8;             // 8 code bytes = 32 elements
        const uint8_t* xb = x_q8_0 + (size_t) c * 34;           // one block_q8_0
        const float dx = x_scales ? x_scales[c] : f16_at(xb);
        const int8_t* xq = (const int8_t*) (xb + 2);

        // ---- THE CODES, EXPANDED TO ONE BYTE PER ELEMENT, THEN FOUR AT A TIME INTO `dp4a`.
        //
        // The packed form is LSB-first: element 4j+k is bits [2k, 2k+2) of code byte j.  `dp4a` needs both
        // operands as packed int8, so each code byte becomes a word whose four bytes are its four 2-bit
        // fields - which is exactly `(c & 3) | ((c>>2)&3)<<8 | ((c>>4)&3)<<16 | ((c>>6)&3)<<24`.
        int s = 0;      // sum of code * x
        int hx = 0;     // sum of x        - the weight-independent term, as ones * x
        const int ones = 0x01010101;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const unsigned cbyte = cb[j];
            const int cw = (int) ((cbyte & 3u) | (((cbyte >> 2) & 3u) << 8) | (((cbyte >> 4) & 3u) << 16) |
                                  (((cbyte >> 6) & 3u) << 24));
            // **`memcpy`, NOT A CAST.**  A `block_q8_0` is 34 BYTES - two of header then 32 of int8 - so the
            // activation data at offset 2 is never 4-byte aligned, and `*(const int*)(xq + 4*j)` faults with
            // "misaligned address".  It faulted exactly that way on this kernel's first run.  `memcpy` of a
            // known 4 bytes compiles to whatever load is legal for the alignment, which is the point of using
            // it rather than reasoning about which cast happens to work.
            int xw;
            memcpy(&xw, xq + 4 * j, 4);
            s = STRATA_DP4A(cw, xw, s);
            hx = STRATA_DP4A(ones, xw, hx);
        }
        // One weight scale per 64 elements, so per TWO 32-element chunks.
        const float dw = f16_at(scales + (size_t) (c >> 1) * 2);
        acc += dw * dx * (float) (s - hx);
    }
    return acc;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, off);
    return v;
}

/// GATE AND UP, ONE WARP PER ROW.
///
/// **THE ROWS ARE INTERLEAVED AND THE FIRST VERSION OF THIS GOT IT WRONG.**  In the blob, row-SLOT `i` of the
/// `2*FF` gate/up rows is gate row `i/2` when `i` is even and up row `(i-1)/2` when it is odd - that is what
/// `O_GU_CODES + (2r)*ROW_GU` and `+ (2r+1)*ROW_GU` mean in `expert.cpp`.  This kernel decoded the correct
/// code row for slot `i` and then wrote it to output slot `i` of a layout whose first `FF` entries are gate and
/// whose last `FF` are up, which pairs `silu(gate[r]) * up[r]` with the WRONG `up` for every `r`.  It produced
/// finite, plausible numbers.  `moe_hit_parity` caught it on the first run, at worst relative error 2.2e+03.
///
/// So slot `i` is DECODED from row-slot `i` and WRITTEN to the output slot its parity says it belongs to.
__global__ void gu_kernel(const uint8_t* __restrict__ blob_base, const int32_t* __restrict__ slot_index,
                          long long blob_bytes, const uint8_t* __restrict__ x_q8_0,
                          const float* __restrict__ x_scales, float* __restrict__ gate_up, int n_hits,
                          const int32_t* __restrict__ d_count = nullptr,
                          const int32_t* __restrict__ dst_index = nullptr, int tok_div = 0) {
    const int warps_per_block = (int) (blockDim.x >> 5);
    const long long slot = (long long) blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    const long long rows_per_hit = 2LL * FF;
    const long long total = (long long) n_hits * rows_per_hit;
    if (slot >= total) return;
    const int h = (int) (slot / rows_per_hit);
    if (d_count != nullptr && h >= *d_count) return;     // token graph: capacity layout, device count
    const int i = (int) (slot % rows_per_hit);
    const int lane = threadIdx.x & 31;

    const uint8_t* blob = blob_base + (size_t) slot_index[h] * (size_t) blob_bytes;
    if (tok_div > 0) {   // plan v0.3 P6 verify window: each hit reads its own token's activation
        const int tok = dst_index[h] / tok_div;
        x_q8_0 += (size_t) tok * (size_t) (H / 32) * 34;
        if (x_scales != nullptr) x_scales += (size_t) tok * (size_t) (H / 32);
    }
    const float acc = row_dot_s2_q8(blob + (size_t) i * ROW_GU,
                                    blob + O_GU_SCALES + (size_t) i * SC_GU * 2, x_q8_0, H / 32, lane, x_scales);
    const float s = warp_sum(acc);
    if (lane != 0) return;
    // ---- THE OUTPUT LAYOUT IS GATE-MAJOR, AND THAT IS NOT COSMETIC.
    //
    // The first version wrote `gate_up[h*2FF + {0..FF-1}] = gate` and `[h*2FF + FF..] = up`, i.e. a per-hit
    // [gate | up] pair, and then called the shared `swiglu_kernel` and `quantize_q8_0` over the whole buffer.
    // Both of those walk a CONTIGUOUS range, so with more than one hit they read hit 0's UP rows where they
    // wanted hit 1's GATE rows - finite numbers, wrong expert.  With one hit it would have passed.
    //
    // Gate-major removes the mismatch instead of adding a stride to two other kernels: every hit's gate rows
    // are contiguous from 0, every hit's up rows are contiguous from `n_hits * FF`, and the swiglu's output
    // lands in the first `n_hits * FF` floats exactly where `quantize_q8_0` reads it.
    const int r = i >> 1;
    const size_t base = (i & 1) ? ((size_t) n_hits * FF + (size_t) h * FF) : ((size_t) h * FF);
    gate_up[base + (size_t) r] = s;
}

// `silu(gate) * up` lives in `strata/kernels/swiglu.cuh` (the fused kernels call it in their epilogues);
// the standalone pass it replaced is gone.  The scratch layout is unchanged: `[0, n_pairs)` ends up holding
// the products, `[n_pairs, 2*n_pairs)` the raw up rows.
/// `silu(gate) * up`, in place, over a GATE-MAJOR buffer: `[0, n_pairs)` is every hit's gate and
/// `[n_pairs, 2*n_pairs)` is every hit's up, so hit `h`'s row `r` meets itself at `h*FF + r`.
///
/// SiLU on the GATE and multiplied by up - the reading `docs/semantics.md` records, and the one that is wrong
/// the other way round in a way that still produces a finite number.
__global__ void swiglu_kernel(float* __restrict__ gate_up, long long n_pairs) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pairs) return;
    const float g = gate_up[i];
    const float u = gate_up[n_pairs + i];
    gate_up[i] = (g / (1.0f + __expf(-g))) * u;
}

__global__ void swiglu_quantize_q8_0_scaled_kernel(float* __restrict__ gate_up, long long n_pairs,
                                                   uint8_t* __restrict__ blocks, float* __restrict__ scales) {
    const long long n_blocks = n_pairs >> 5;
    const long long b = ((long long) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (b >= n_blocks) return;
    const int lane = threadIdx.x & 31;
    const long long idx = b * 32 + lane;
    const float g = gate_up[idx];
    const float u = gate_up[n_pairs + idx];
    const float xv = (g / (1.0f + __expf(-g))) * u;
    gate_up[idx] = xv;
    uint8_t* out = blocks + b * 34;

    float amax = fabsf(xv);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    const float s = amax > 0.f ? amax / 127.f : 0.f;
    const float inv = s > 0.f ? 1.f / s : 0.f;
    if (lane == 0) {
        scales[b] = s;
        const uint16_t d16bits = f16_from_f32(s);
        out[0] = (uint8_t) (d16bits & 0xFF);
        out[1] = (uint8_t) (d16bits >> 8);
    }
    const float t = xv * inv;
    const float r = t + (t >= 0.f ? 0.5f : -0.5f);
    int v = (int) r;
    v = v < -127 ? -127 : (v > 127 ? 127 : v);
    out[2 + lane] = (uint8_t) (int8_t) v;
}

/// DOWN, ONE WARP PER ROW, reading the quantized intermediate the caller produced.
///
/// `dst_index[h]` is which row of the shared output buffer hit `h` fills - see the header.  It is the router's
/// slot, not `h`, and the two differ on every layer where some experts are resident and some are not.
__global__ void down_kernel(const uint8_t* __restrict__ blob_base, const int32_t* __restrict__ slot_index,
                            const int32_t* __restrict__ dst_index, long long blob_bytes,
                            const uint8_t* __restrict__ h_q8_0, const float* __restrict__ h_scales,
                            float* __restrict__ out, int n_hits, const int32_t* __restrict__ d_count = nullptr) {
    const int warps_per_block = (int) (blockDim.x >> 5);
    const long long row = (long long) blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    const long long total = (long long) n_hits * H;
    if (row >= total) return;
    const int h = (int) (row / H);
    if (d_count != nullptr && h >= *d_count) return;
    const int r = (int) (row % H);
    const int lane = threadIdx.x & 31;

    const uint8_t* blob = blob_base + (size_t) slot_index[h] * (size_t) blob_bytes;
    const uint8_t* xb = h_q8_0 + (size_t) h * (size_t) (FF / 32) * 34;
    const float acc = row_dot_s2_q8(blob + O_D_CODES + (size_t) r * ROW_D,
                                    blob + O_D_SCALES + (size_t) r * SC_D * 2, xb, FF / 32, lane,
                                    h_scales ? h_scales + (size_t) h * (size_t) (FF / 32) : nullptr);
    const float s = warp_sum(acc);
    if (lane == 0) out[(size_t) dst_index[h] * H + r] = s;
}

// ---- THE SAME INTEGERS FROM FEWER INSTRUCTIONS.  `STRATA_OLD_GROUPED=1` keeps the kernels above.
//
// **INSIDE A CHUNK, WHICH CODE MEETS WHICH ACTIVATION IN A `dp4a` WORD IS FREE.**  `s = sum_e code_e * x_e` and
// `hx = sum_e x_e` are exact integer sums (|s| <= 32 * 3 * 128), so any grouping of the 32 products into words
// gives the same two integers.  The grouping used here costs no transposition of the codes: `(w >> 2f) & 0x03030303`
// holds field `f` of each of a code word's four bytes - elements `4b + f`, plus 16 for the second word - so it is
// the ACTIVATIONS that are regrouped, once per chunk, into
//
//     X[4h + f] = { x[16h + f], x[16h + 4 + f], x[16h + 8 + f], x[16h + 12 + f] }
//
// a 4x4 byte transpose of each half of the chunk, eight `__byte_perm`.  The kernels above spend seven integer
// operations per code BYTE and a second `dp4a` chain for `hx` on every row; here a code word costs seven for 16
// elements and `hx` is summed once per chunk.  `s`, `hx` and the float expression `dw * dx * (float) (s - hx)` are
// unchanged, so every output is bitwise the previous kernels' - `s2_expert_grouped_parity` checks exactly that.

// fp16 at an even address, one 16-bit load (`f16_at` reads it as two bytes); same value.
__device__ __forceinline__ float f16_ld(const uint8_t* p) {
    return __half2float(__ushort_as_half(*(const unsigned short*) p));
}

// A chunk's two code words as the eight `dp4a` operands that pair with `X[0..7]`: `m[4h + f]` byte `b` is the
// code of element `16h + 4b + f`.
__device__ __forceinline__ void expand_codes(uint2 cb, int m[8]) {
    const unsigned M = 0x03030303u;
    m[0] = (int) (cb.x & M);
    m[1] = (int) ((cb.x >> 2) & M);
    m[2] = (int) ((cb.x >> 4) & M);
    m[3] = (int) ((cb.x >> 6) & M);
    m[4] = (int) (cb.y & M);
    m[5] = (int) ((cb.y >> 2) & M);
    m[6] = (int) ((cb.y >> 4) & M);
    m[7] = (int) ((cb.y >> 6) & M);
}

// The 32 int8 of the `block_q8_0` at `xb` regrouped into `X[0..7]` (see above), and their sum `hx`.
//
// **NINE ALIGNED WORD LOADS INSTEAD OF 32 BYTE LOADS.**  The caller guarantees a 4-byte aligned activation row, so
// chunk `c`'s int8 start at `34c + 2`, i.e. 0 bytes (odd `c`) or 2 bytes (even `c`) into an aligned word; the
// words are read aligned and funnel-shifted into place.  The ninth word is read only in the 2-byte case, and then
// it lies inside block `c + 1` of the same row: both row lengths (80 and 20 chunks) are even, so an even chunk is
// never a row's last.
__device__ __forceinline__ int load_x_chunk(const uint8_t* __restrict__ xb, int X[8]) {
    const uint8_t* q = xb + 2;
    const int off = (int) ((uintptr_t) q & 3);
    const unsigned* p = (const unsigned*) (q - off);
    const unsigned sh = (unsigned) off * 8;
    unsigned v[9];
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = __ldg(p + j);
    v[8] = off != 0 ? __ldg(p + 8) : 0u;
    unsigned n[8];
    int hx = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        // natural word j: x[4j .. 4j+3]; a 64-bit shift (sh is 0, 8, 16 or 24) rather than __funnelshift_r, so the same
        // source needs no CUDA-only intrinsic.
        n[j] = (unsigned) ((((unsigned long long) v[j + 1] << 32) | v[j]) >> sh);
        hx = STRATA_DP4A(0x01010101, (int) n[j], hx);
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const unsigned t0 = __byte_perm(n[4 * h], n[4 * h + 1], 0x5140);      // a0 b0 a1 b1
        const unsigned t1 = __byte_perm(n[4 * h], n[4 * h + 1], 0x7362);      // a2 b2 a3 b3
        const unsigned t2 = __byte_perm(n[4 * h + 2], n[4 * h + 3], 0x5140);  // c0 d0 c1 d1
        const unsigned t3 = __byte_perm(n[4 * h + 2], n[4 * h + 3], 0x7362);  // c2 d2 c3 d3
        X[4 * h + 0] = (int) __byte_perm(t0, t2, 0x5410);                    // a0 b0 c0 d0
        X[4 * h + 1] = (int) __byte_perm(t0, t2, 0x7632);                    // a1 b1 c1 d1
        X[4 * h + 2] = (int) __byte_perm(t1, t3, 0x5410);                    // a2 b2 c2 d2
        X[4 * h + 3] = (int) __byte_perm(t1, t3, 0x7632);                    // a3 b3 c3 d3
    }
    return hx;
}

// `s` for one chunk: the eight expanded code words against the eight regrouped activation words.
__device__ __forceinline__ int chunk_s(const int m[8], const int X[8]) {
    int s = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) s = STRATA_DP4A(m[j], X[j], s);
    return s;
}

/// `gu_kernel`, new: ONE WARP PER (gate, up) PAIR - row-slots `2r` and `2r + 1`, adjacent in the blob - so each
/// activation chunk is loaded and regrouped once for both rows.  Per row, the lane's chunks, their order, the float
/// expression and the shuffle reduction are `row_dot_s2_q8`'s; the outputs land where `gu_kernel` puts them.
__global__ void gu_pair_kernel(const uint8_t* __restrict__ blob_base, const int32_t* __restrict__ slot_index,
                               long long blob_bytes, const uint8_t* __restrict__ x_q8_0,
                               const float* __restrict__ x_scales, float* __restrict__ gate_up, int n_hits,
                               const int32_t* __restrict__ d_count, const int32_t* __restrict__ dst_index,
                               int tok_div) {
    const int warps_per_block = (int) (blockDim.x >> 5);
    const long long pair = (long long) blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    const long long total = (long long) n_hits * FF;
    if (pair >= total) return;
    const int h = (int) (pair / FF);
    if (d_count != nullptr && h >= *d_count) return;
    const int r = (int) (pair % FF);
    const int lane = threadIdx.x & 31;

    const uint8_t* blob = blob_base + (size_t) slot_index[h] * (size_t) blob_bytes;
    if (tok_div > 0) {
        const int tok = dst_index[h] / tok_div;
        x_q8_0 += (size_t) tok * (size_t) (H / 32) * 34;
        if (x_scales != nullptr) x_scales += (size_t) tok * (size_t) (H / 32);
    }
    const uint8_t* codes = blob + (size_t) (2 * r) * ROW_GU;               // gate row; the up row follows it
    const uint8_t* scales = blob + O_GU_SCALES + (size_t) (2 * r) * SC_GU * 2;
    float acc_g = 0.0f, acc_u = 0.0f;
    for (int c = lane; c < H / 32; c += 32) {
        const uint8_t* xb = x_q8_0 + (size_t) c * 34;
        const float dx = x_scales ? x_scales[c] : f16_ld(xb);
        int X[8], m[8];
        const int hx = load_x_chunk(xb, X);
        expand_codes(*(const uint2*) (codes + (size_t) c * 8), m);
        const float dw_g = f16_ld(scales + (size_t) (c >> 1) * 2);
        acc_g += dw_g * dx * (float) (chunk_s(m, X) - hx);
        expand_codes(*(const uint2*) (codes + ROW_GU + (size_t) c * 8), m);
        const float dw_u = f16_ld(scales + SC_GU * 2 + (size_t) (c >> 1) * 2);
        acc_u += dw_u * dx * (float) (chunk_s(m, X) - hx);
    }
    const float sg = warp_sum(acc_g);
    const float su = warp_sum(acc_u);
    if (lane != 0) return;
    gate_up[(size_t) h * FF + (size_t) r] = sg;                                // gate-major, as in `gu_kernel`
    gate_up[(size_t) n_hits * FF + (size_t) h * FF + (size_t) r] = su;
}

/// `down_kernel`, new: ONE WARP PER PAIR OF ROWS `r, r + 1` of one hit, the intermediate's chunk loaded once for both.
__global__ void down_pair_kernel(const uint8_t* __restrict__ blob_base, const int32_t* __restrict__ slot_index,
                                 const int32_t* __restrict__ dst_index, long long blob_bytes,
                                 const uint8_t* __restrict__ h_q8_0, const float* __restrict__ h_scales,
                                 float* __restrict__ out, int n_hits, const int32_t* __restrict__ d_count) {
    const int warps_per_block = (int) (blockDim.x >> 5);
    const long long pair = (long long) blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    const long long total = (long long) n_hits * (H / 2);
    if (pair >= total) return;
    const int h = (int) (pair / (H / 2));
    if (d_count != nullptr && h >= *d_count) return;
    const int r = 2 * (int) (pair % (H / 2));
    const int lane = threadIdx.x & 31;

    const uint8_t* blob = blob_base + (size_t) slot_index[h] * (size_t) blob_bytes;
    const uint8_t* xrow = h_q8_0 + (size_t) h * (size_t) (FF / 32) * 34;
    const float* xs = h_scales ? h_scales + (size_t) h * (size_t) (FF / 32) : nullptr;
    const uint8_t* codes = blob + O_D_CODES + (size_t) r * ROW_D;
    const uint8_t* scales = blob + O_D_SCALES + (size_t) r * SC_D * 2;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (int c = lane; c < FF / 32; c += 32) {
        const uint8_t* xb = xrow + (size_t) c * 34;
        const float dx = xs ? xs[c] : f16_ld(xb);
        int X[8], m[8];
        const int hx = load_x_chunk(xb, X);
        expand_codes(*(const uint2*) (codes + (size_t) c * 8), m);
        const float dw0 = f16_ld(scales + (size_t) (c >> 1) * 2);
        acc0 += dw0 * dx * (float) (chunk_s(m, X) - hx);
        expand_codes(*(const uint2*) (codes + ROW_D + (size_t) c * 8), m);
        const float dw1 = f16_ld(scales + SC_D * 2 + (size_t) (c >> 1) * 2);
        acc1 += dw1 * dx * (float) (chunk_s(m, X) - hx);
    }
    const float s0 = warp_sum(acc0);
    const float s1 = warp_sum(acc1);
    if (lane != 0) return;
    const size_t o = (size_t) dst_index[h] * H + (size_t) r;
    out[o] = s0;
    out[o + 1] = s1;
}

// The CPU subtracts the weight bias after its eight FMA accumulators have been reduced. Moving the
// subtraction into each integer dot, as the legacy kernel does, changes rounding even with equal scales.
__global__ void activation_correction_kernel(const uint8_t* q8, const float* scales, float* hx, int chunks) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= chunks) return;
    const int8_t* q = (const int8_t*) (q8 + (size_t) c * 34 + 2);
    int sum = 0;
    for (int j = 0; j < 32; ++j) sum += q[j];
    hx[c] = __fmul_rn(scales[c], (float) sum);
}

__device__ __forceinline__ int dot4(const uint8_t* codes, const int8_t* q) {
    const unsigned c = *codes;
    const int cw = (int) ((c & 3u) | (((c >> 2) & 3u) << 8) |
                          (((c >> 4) & 3u) << 16) | (((c >> 6) & 3u) << 24));
    int xw;
    memcpy(&xw, q, sizeof xw);
    return STRATA_DP4A(cw, xw, 0);
}

__device__ __forceinline__ float row_dot_cpu_order(const uint8_t* codes, const uint8_t* scales,
                                                   const uint8_t* xq, const float* xs,
                                                   const float* hx, int blocks, int lane) {
    float acc = 0.0f;
    float corr = 0.0f;
    for (int b = 0; b < blocks; ++b) {
        const float d = f16_at(scales + 2 * b);
        const int lo = dot4(codes + b * 16 + lane,
                           (const int8_t*) (xq + (size_t) (2 * b) * 34 + 2) + lane * 4);
        const int hi = dot4(codes + b * 16 + 8 + lane,
                           (const int8_t*) (xq + (size_t) (2 * b + 1) * 34 + 2) + lane * 4);
        acc = __fmaf_rn(__fmul_rn(d, xs[2 * b]), (float) lo, acc);
        acc = __fmaf_rn(__fmul_rn(d, xs[2 * b + 1]), (float) hi, acc);
        if (lane == 0)
            corr = __fadd_rn(corr, __fmul_rn(d, __fadd_rn(hx[2 * b], hx[2 * b + 1])));
    }
    // _mm_add_ps(low128, high128), then two _mm_hadd_ps. The pair order is 4, 1, 2;
    // a standard shuffle tree in the order 4, 2, 1 is a different floating-point expression.
    constexpr unsigned mask = 0xffffffffu;
    acc = __fadd_rn(acc, __shfl_down_sync(mask, acc, 4, 8));
    acc = __fadd_rn(acc, __shfl_down_sync(mask, acc, 1, 8));
    acc = __fadd_rn(acc, __shfl_down_sync(mask, acc, 2, 8));
    return __fsub_rn(acc, corr);  // Only lane zero is consumed.
}

template <bool DOWN>
__global__ void cpu_order_projection_kernel(const uint8_t* blob_base, const int32_t* slots,
                                              const int32_t* destinations, long long blob_bytes,
                                              const uint8_t* xq, const float* xs, const float* hx,
                                              float* out, int n_hits) {
    constexpr int rows_per_hit = DOWN ? H : 2 * FF;
    const int row = blockIdx.x * (blockDim.x / 8) + threadIdx.x / 8;
    if (row >= n_hits * rows_per_hit) return;
    const int h = row / rows_per_hit;
    const int r = row % rows_per_hit;
    const int lane = threadIdx.x & 7;
    const uint8_t* blob = blob_base + (size_t) slots[h] * (size_t) blob_bytes;
    const int chunks_offset = DOWN ? h * (FF / 32) : 0;
    const uint8_t* codes = DOWN ? blob + O_D_CODES + (size_t) r * ROW_D : blob + (size_t) r * ROW_GU;
    const uint8_t* scales = DOWN ? blob + O_D_SCALES + (size_t) r * SC_D * 2
                                 : blob + O_GU_SCALES + (size_t) r * SC_GU * 2;
    const float value = row_dot_cpu_order(codes, scales, xq + (size_t) chunks_offset * 34,
                                           xs + chunks_offset, hx + chunks_offset,
                                           DOWN ? SC_D : SC_GU, lane);
    if (lane != 0) return;
    if (DOWN) out[(size_t) destinations[h] * H + r] = value;
    else out[((r & 1) ? (size_t) n_hits * FF : 0) + (size_t) h * FF + (r >> 1)] = value;
}

__global__ void cpu_order_swiglu_kernel(float* gu, int pairs) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pairs) return;
    const float g = gu[i];
    // Accurate fp32 exponential; __expf's approximation would introduce an additional source of error.
    // CPU/GPU libc last-bit differences are diagnosed separately by the micro, not hidden with FP64 here.
    const float eg = expf(-g);
    gu[i] = __fmul_rn(__fdiv_rn(g, __fadd_rn(1.0f, eg)), gu[pairs + i]);
}

__global__ void cpu_order_quantize_kernel(const float* x, uint8_t* blocks, float* scales,
                                           float* hx, int chunks) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= chunks) return;
    const float* xb = x + c * 32;
    uint8_t* out = blocks + (size_t) c * 34;
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(xb[j]));
    const float s = amax > 0.0f ? __fdiv_rn(amax, 127.0f) : 0.0f;
    const float inv = s > 0.0f ? __fdiv_rn(1.0f, s) : 0.0f;
    scales[c] = s;
    const uint16_t bits = __half_as_ushort(__float2half_rn(s));
    out[0] = (uint8_t) bits;
    out[1] = (uint8_t) (bits >> 8);
    int sum = 0;
    for (int j = 0; j < 32; ++j) {
        const float t = __fmul_rn(xb[j], inv);
        int v = (int) __fadd_rn(t, t >= 0.0f ? 0.5f : -0.5f);
        v = v < -127 ? -127 : (v > 127 ? 127 : v);
        out[2 + j] = (uint8_t) (int8_t) v;
        sum += v;
    }
    hx[c] = __fmul_rn(s, (float) sum);
}

constexpr int THREADS = 256;

void check(const char* who, void* stream) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s launch: %s\n", who, cudaGetErrorString(e));
        std::exit(1);
    }
    // Deliberately NOT synchronising for a non-null stream: this is called once per layer from a captured
    // graph's worth of work, and `finish()`'s null-stream sync in `s_gemv.cu` is the pattern that made a whole
    // round of measurements the driver's cost instead of the kernel's (Memory/ERRORS.md RC-7).
    (void) stream;
}

// The previous kernels stay selectable for A/B - `STRATA_OLD_GROUPED=1` in the environment, or
// `moe_grouped_select_old` (the parity test runs both in one process).  The environment is read once, on first use;
// the choice is made at each launch, so a captured graph keeps the kernels it was captured with.
std::atomic<int> g_select_old{-1};
// Which kernels the last launch of an entry point used: 1 = new, 0 = previous, -1 = none since the last query.
std::atomic<int> g_last_path{-1};

bool old_kernels() {
    const int s = g_select_old.load(std::memory_order_relaxed);
    if (s >= 0) return s != 0;
    static const bool env = [] {
        const char* e = std::getenv("STRATA_OLD_GROUPED");
        return e != nullptr && e[0] == '1';
    }();
    return env;
}

// `STRATA_GROUPED_PAIR_MIN_HITS=N`: the per-hit path keeps the previous one-warp-per-row kernels below N hits of
// capacity.  The new per-hit kernels launch half the warps (two rows each), so one hit's gate/up is 80 blocks - fewer
// than an RTX 5090's SMs - and whether that costs time at one to three hits is what `--bench` measures.  Bitwise
// the same either way; default 0 (always the new kernels).  Ignored while `moe_grouped_select_old` forces a choice.
long long pair_min_hits() {
    if (g_select_old.load(std::memory_order_relaxed) >= 0) return 0;
    static const long long n = [] {
        const char* e = std::getenv("STRATA_GROUPED_PAIR_MIN_HITS");
        return e != nullptr ? std::atoll(e) : 0LL;
    }();
    return n;
}

// The new kernels read the activations as aligned words (`load_x_chunk`), so they need 4-byte aligned activation
// rows - the input's and the intermediate's in `scratch`; the per-hit ones also read a blob's codes as uint2, which
// needs an 8-byte aligned arena and slot size (the grouped kernels above already did).  Anything else keeps the
// previous kernels rather than issuing a misaligned load.
bool new_grouped(const void* x_q8_0, const void* scratch) {
    const bool fast = !old_kernels() && ((uintptr_t) x_q8_0 & 3) == 0 && ((uintptr_t) scratch & 3) == 0;
    g_last_path.store(fast ? 1 : 0, std::memory_order_relaxed);
    return fast;
}

bool new_hit(const void* blob_base, long long blob_bytes, const void* x_q8_0, const void* scratch, long long cap) {
    const bool fast = new_grouped(x_q8_0, scratch) && ((uintptr_t) blob_base & 7) == 0 && (blob_bytes & 7) == 0 &&
                      cap >= pair_min_hits();
    g_last_path.store(fast ? 1 : 0, std::memory_order_relaxed);
    return fast;
}

// The per-hit path's two projections, previous or new kernels (one warp per row, or per pair of rows).
void launch_hit_gu(bool fast, const uint8_t* blob_base, const int32_t* slot_index, long long blob_bytes,
                   const uint8_t* x_q8_0, const float* x_scales, float* gate_up, long long cap,
                   const int32_t* d_count, const int32_t* dst_index, int tok_div, cudaStream_t cs) {
    const int warps = THREADS / 32;
    if (fast) {
        const long long pairs = cap * (long long) FF;
        gu_pair_kernel<<<(unsigned) ((pairs + warps - 1) / warps), THREADS, 0, cs>>>(
            blob_base, slot_index, blob_bytes, x_q8_0, x_scales, gate_up, (int) cap, d_count, dst_index, tok_div);
    } else {
        const long long rows = cap * 2LL * FF;
        gu_kernel<<<(unsigned) ((rows + warps - 1) / warps), THREADS, 0, cs>>>(
            blob_base, slot_index, blob_bytes, x_q8_0, x_scales, gate_up, (int) cap, d_count, dst_index, tok_div);
    }
}

void launch_hit_down(bool fast, const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                     long long blob_bytes, const uint8_t* h_q8_0, const float* h_scales, float* out, long long cap,
                     const int32_t* d_count, cudaStream_t cs) {
    const int warps = THREADS / 32;
    if (fast) {
        const long long pairs = cap * (long long) (H / 2);
        down_pair_kernel<<<(unsigned) ((pairs + warps - 1) / warps), THREADS, 0, cs>>>(
            blob_base, slot_index, dst_index, blob_bytes, h_q8_0, h_scales, out, (int) cap, d_count);
    } else {
        const long long rows = cap * (long long) H;
        down_kernel<<<(unsigned) ((rows + warps - 1) / warps), THREADS, 0, cs>>>(
            blob_base, slot_index, dst_index, blob_bytes, h_q8_0, h_scales, out, (int) cap, d_count);
    }
}

}  // namespace

void moe_grouped_select_old(int old) { g_select_old.store(old < 0 ? -1 : (old != 0 ? 1 : 0)); }

int moe_grouped_last_path() { return g_last_path.exchange(-1, std::memory_order_relaxed); }

uint64_t moe_hit_grouped_scratch_bytes(int64_t n_hits, int64_t n_embd, int64_t n_ff) {
    if (n_hits <= 0) return 0;
    const uint64_t gu = (uint64_t) n_hits * (uint64_t) (2 * n_ff) * 4;
    const uint64_t q8 = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 34;
    // R4.2h: the fp32 scales for the INTERMEDIATE's own quantization, one per 32-element chunk per hit.
    const uint64_t hs = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 4;
    const uint64_t xh = (uint64_t) (n_embd / 32) * 4;
    return ((gu + 15) & ~15ull) + ((q8 + 15) & ~15ull) + 2 * ((hs + 15) & ~15ull) +
           ((xh + 15) & ~15ull);
}

void moe_hit_grouped_s2(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                        int64_t n_hits, int64_t blob_bytes, const uint8_t* x_q8_0, void* scratch, float* out,
                        void* stream, const float* x_scales) {
    if (n_hits <= 0) return;
    cudaStream_t cs = (cudaStream_t) stream;
    const bool fast = new_hit(blob_base, blob_bytes, x_q8_0, scratch, n_hits);

    const uint64_t gu_bytes = ((uint64_t) n_hits * (uint64_t) (2 * FF) * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) n_hits * (uint64_t) (FF / 32) * 34 + 15) & ~15ull;
    float* gate_up = (float*) scratch;
    uint8_t* h_q8_0 = (uint8_t*) scratch + gu_bytes;
    float* h_scales = (float*) ((uint8_t*) scratch + gu_bytes + q8_bytes);

    // 1. gate + up, one launch for every row of every hit.
    {
        launch_hit_gu(fast, blob_base, slot_index, blob_bytes, x_q8_0, x_scales, gate_up, n_hits, nullptr, nullptr,
                      0, cs);
        check("moe_hit_grouped_s2/gu", stream);
    }
    // 2+3. silu(gate) * up, then the intermediate's own contract - ONE kernel where two ran.  The products
    //    land in the gate half exactly as `swiglu_kernel` left them (the parity test compares that scratch),
    //    and the q8_0 blocks are the same bytes `quantize_q8_0` produced.  `swilu_kind` 1 is this file's
    //    `swiglu_kernel` expression (see `strata/kernels/swiglu.cuh`).
    //    The down weight is Q2_0, whose `vec_dot_type` is Q8_0 - `ggml_mul_mat`'s rule, not a choice.  With
    //    `x_scales` the intermediate takes the CPU's contract (`quantize_q8_0_scaled`), R4.2h.
    {
        const long long pairs = n_hits * (long long) FF;
#if defined(STRATA_EXPERIMENTAL_SM60)
        if (strata::kernels::fuse_decode()) {
        if (x_scales != nullptr)
            swilu_quantize_q8_0_scaled(gate_up, gate_up + pairs, gate_up, pairs, 1, h_q8_0, h_scales, stream);
        else
            swilu_quantize_q8_0(gate_up, gate_up + pairs, gate_up, pairs, 1, h_q8_0, stream);
        } else
#endif
        {
        const unsigned blocks = (unsigned) ((pairs + THREADS - 1) / THREADS);
        swiglu_kernel<<<blocks, THREADS, 0, cs>>>(gate_up, pairs);
        check("moe_hit_grouped_s2/swiglu", stream);
        if (x_scales != nullptr) quantize_q8_0_scaled(gate_up, h_q8_0, h_scales, n_hits * (int64_t) FF, stream);
        else quantize_q8_0(gate_up, h_q8_0, n_hits * (int64_t) FF, stream);
        }
    }
    // 4. down.
    {
        launch_hit_down(fast, blob_base, slot_index, dst_index, blob_bytes, h_q8_0,
                        x_scales != nullptr ? h_scales : nullptr, out, n_hits, nullptr, cs);
        check("moe_hit_grouped_s2/down", stream);
    }
}

namespace {
// Plan v0.3 P4 token graph: which of this layer's routed experts are resident, decided ON THE DEVICE from the
// static residency row, so no host step sits between the ring and the hit kernels.  One warp; k <= 32.
__global__ void hit_select_kernel(const int32_t* __restrict__ ids, const int32_t* __restrict__ res_row, int k,
                                  int n_expert, int32_t* __restrict__ slot, int32_t* __restrict__ dst,
                                  int32_t* __restrict__ count) {
    const int lane = threadIdx.x;
    int s = -1;
    if (lane < k) {
        const int e = ids[lane];
        if (e >= 0 && e < n_expert) s = res_row[e];
    }
    const unsigned hit = __ballot_sync(0xffffffffu, s >= 0);
    if (s >= 0) {
        const int at = __popc(hit & ((1u << lane) - 1u));
        slot[at] = s;
        dst[at] = lane;
    }
    if (lane == 0) *count = __popc(hit);
}

// Plan v0.3 P6: the same for up to 128 routed entries (a verify window of T tokens x k): four warps, ballots
// compacted in entry order.
__global__ void hit_select_multi_kernel(const int32_t* __restrict__ ids, const int32_t* __restrict__ res_row, int n,
                                        int n_expert, int32_t* __restrict__ slot, int32_t* __restrict__ dst,
                                        int32_t* __restrict__ count) {
    __shared__ int warp_count[4];
    const int i = threadIdx.x, lane = i & 31, warp = i >> 5;
    int s = -1;
    if (i < n) {
        const int e = ids[i];
        if (e >= 0 && e < n_expert) s = res_row[e];
    }
    const unsigned hit = __ballot_sync(0xffffffffu, s >= 0);
    if (lane == 0) warp_count[warp] = __popc(hit);
    __syncthreads();
    int before = 0;
    for (int w = 0; w < warp; ++w) before += warp_count[w];
    if (s >= 0) {
        const int at = before + __popc(hit & ((1u << lane) - 1u));
        slot[at] = s;
        dst[at] = i;
    }
    if (i == 0) *count = warp_count[0] + warp_count[1] + warp_count[2] + warp_count[3];
}

__global__ void add_hits_kernel(float* __restrict__ parts, const float* __restrict__ hit_out,
                                const int32_t* __restrict__ dst, const int32_t* __restrict__ count, int n_embd) {
    const int h = blockIdx.y;
    if (h >= *count) return;
    const size_t row = (size_t) dst[h] * (size_t) n_embd;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_embd; i += gridDim.x * blockDim.x)
        parts[row + i] += hit_out[row + i];
}
}  // namespace

void moe_hit_select(const int32_t* ids, const int32_t* res_row, int k, int n_expert, int32_t* slot, int32_t* dst,
                    int32_t* count, void* stream) {
    if (k < 1 || k > 32) { std::fprintf(stderr, "moe_hit_select: k must be 1..32\n"); std::exit(1); }
    hit_select_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(ids, res_row, k, n_expert, slot, dst, count);
    check("moe_hit_select", stream);
}

void moe_hit_grouped_s2_dev(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                            const int32_t* d_count, int64_t cap, int64_t blob_bytes, const uint8_t* x_q8_0,
                            void* scratch, float* out, void* stream, const float* x_scales) {
    if (cap <= 0) return;
    cudaStream_t cs = (cudaStream_t) stream;
    const bool fast = new_hit(blob_base, blob_bytes, x_q8_0, scratch, cap);
    const uint64_t gu_bytes = ((uint64_t) cap * (uint64_t) (2 * FF) * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) cap * (uint64_t) (FF / 32) * 34 + 15) & ~15ull;
    float* gate_up = (float*) scratch;
    uint8_t* h_q8_0 = (uint8_t*) scratch + gu_bytes;
    float* h_scales = (float*) ((uint8_t*) scratch + gu_bytes + q8_bytes);
    {
        launch_hit_gu(fast, blob_base, slot_index, blob_bytes, x_q8_0, x_scales, gate_up, cap, d_count, nullptr, 0,
                      cs);
        check("moe_hit_grouped_s2_dev/gu", stream);
    }
    // silu + intermediate quantization in one kernel (same full-capacity range as the two kernels it replaces,
    // sentinel rows included - the parity test compares that scratch).
    {
        const long long pairs = cap * (long long) FF;
#if defined(STRATA_EXPERIMENTAL_SM60)
        if (strata::kernels::fuse_decode()) {
        if (x_scales != nullptr)
            swilu_quantize_q8_0_scaled(gate_up, gate_up + pairs, gate_up, pairs, 1, h_q8_0, h_scales, stream);
        else
            swilu_quantize_q8_0(gate_up, gate_up + pairs, gate_up, pairs, 1, h_q8_0, stream);
        } else
#endif
        {
        swiglu_kernel<<<(unsigned) ((pairs + THREADS - 1) / THREADS), THREADS, 0, cs>>>(gate_up, pairs);
        check("moe_hit_grouped_s2_dev/swiglu", stream);
        if (x_scales != nullptr) quantize_q8_0_scaled(gate_up, h_q8_0, h_scales, cap * (int64_t) FF, stream);
        else quantize_q8_0(gate_up, h_q8_0, cap * (int64_t) FF, stream);
        }
    }
    {
        launch_hit_down(fast, blob_base, slot_index, dst_index, blob_bytes, h_q8_0,
                        x_scales != nullptr ? h_scales : nullptr, out, cap, d_count, cs);
        check("moe_hit_grouped_s2_dev/down", stream);
    }
}

void moe_hit_select_multi(const int32_t* ids, const int32_t* res_row, int n, int n_expert, int32_t* slot, int32_t* dst,
                          int32_t* count, void* stream) {
    if (n < 1 || n > 128) { std::fprintf(stderr, "moe_hit_select_multi: n must be 1..128\n"); std::exit(1); }
    hit_select_multi_kernel<<<1, 128, 0, (cudaStream_t) stream>>>(ids, res_row, n, n_expert, slot, dst, count);
    check("moe_hit_select_multi", stream);
}

void moe_hit_grouped_s2_multi(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                              const int32_t* d_count, int64_t cap, int64_t blob_bytes, const uint8_t* x_q8_0,
                              const float* x_scales, int k_per_token, void* scratch, float* out, void* stream) {
    if (cap <= 0) return;
    cudaStream_t cs = (cudaStream_t) stream;
    const bool fast = new_hit(blob_base, blob_bytes, x_q8_0, scratch, cap);
    const uint64_t gu_bytes = ((uint64_t) cap * (uint64_t) (2 * FF) * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) cap * (uint64_t) (FF / 32) * 34 + 15) & ~15ull;
    float* gate_up = (float*) scratch;
    uint8_t* h_q8_0 = (uint8_t*) scratch + gu_bytes;
    float* h_scales = (float*) ((uint8_t*) scratch + gu_bytes + q8_bytes);
    {
        launch_hit_gu(fast, blob_base, slot_index, blob_bytes, x_q8_0, x_scales, gate_up, cap, d_count, dst_index,
                      k_per_token, cs);
        check("moe_hit_grouped_s2_multi/gu", stream);
    }
    {
        const long long pairs = cap * (long long) FF;
#if defined(STRATA_EXPERIMENTAL_SM60)
        if (strata::kernels::fuse_decode()) {
        if (x_scales != nullptr)
            swilu_quantize_q8_0_scaled(gate_up, gate_up + pairs, gate_up, pairs, 1, h_q8_0, h_scales, stream);
        else
            swilu_quantize_q8_0(gate_up, gate_up + pairs, gate_up, pairs, 1, h_q8_0, stream);
        } else
#endif
        {
        swiglu_kernel<<<(unsigned) ((pairs + THREADS - 1) / THREADS), THREADS, 0, cs>>>(gate_up, pairs);
        check("moe_hit_grouped_s2_multi/swiglu", stream);
        if (x_scales != nullptr) quantize_q8_0_scaled(gate_up, h_q8_0, h_scales, cap * (int64_t) FF, stream);
        else quantize_q8_0(gate_up, h_q8_0, cap * (int64_t) FF, stream);
        }
    }
    {
        launch_hit_down(fast, blob_base, slot_index, dst_index, blob_bytes, h_q8_0,
                        x_scales != nullptr ? h_scales : nullptr, out, cap, d_count, cs);
        check("moe_hit_grouped_s2_multi/down", stream);
    }
}

namespace {
// Plan v0.3 P6: experts GROUPED - one blob pointer per group (a VRAM slot or a mapped host blob read over PCIe),
// every entry of the group (a token routed to that expert) computed from ONE read of each row.  Per entry the
// arithmetic is `row_dot_s2_q8`'s, chunk by chunk in the same lane order, so every entry is bitwise the per-entry
// hit kernel's.
constexpr int GU_CHUNKS = (H / 32 + 31) / 32;   // 3: chunks of a gate/up row per lane (80 chunks / 32 lanes)
constexpr int GMAX = 8;                          // entries per group (tokens routed to one expert in a window)
// a group holds one entry per token of the window routed to its expert, and a kernel keeps at most GMAX of them
// (`min(..., GMAX)`): a longer group would drop entries without a word.  A window of 9..16 rows can route more tokens
// than that to one expert, so `moe_grouped_s2` takes the window's largest group and runs the kernel again with
// SKIP = GMAX over the entries from the GMAX-th on (the staging is GMAX entries wide in shared memory: 51 KB at 16
// would not compile on CUDA, whose static limit is 48 KB).  Every entry's output is its own dot products, so the
// passes are bitwise the one wide pass; SKIP = 0 is the kernel as it was.
constexpr int kGroupPasses = (kVerifyMaxT + GMAX - 1) / GMAX;
static_assert(kGroupPasses * GMAX >= kVerifyMaxT, "a verify window's group can exceed the passes' entries");
constexpr int GU_ROWS = 32;                      // gate/up rows per block: 4 per warp
constexpr int D_ROWS = 64;                       // down rows per block: 8 per warp

// One activation chunk's contribution, `row_dot_s2_q8`'s inner body with the 32 int8 of the chunk already in
// aligned words: same dp4a sequence, same float expression, so the result is bitwise the per-entry kernel's.
__device__ __forceinline__ float chunk_dot(uint2 cb, const int* xw, float dw, float dx) {
    const uint8_t* cbytes = (const uint8_t*) &cb;
    int s = 0, hx = 0;
    const int ones = 0x01010101;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const unsigned cbyte = cbytes[j];
        const int cw = (int) ((cbyte & 3u) | (((cbyte >> 2) & 3u) << 8) | (((cbyte >> 4) & 3u) << 16) |
                              (((cbyte >> 6) & 3u) << 24));
        s = STRATA_DP4A(cw, xw[j], s);
        hx = STRATA_DP4A(ones, xw[j], hx);
    }
    return dw * dx * (float) (s - hx);
}

// Gate/up: a block = GU_ROWS rows of ONE group.  The group's activations (each entry's token row of x_q8_0 and
// its fp32 scales) are staged once into shared memory as aligned words; each warp then walks its rows, loading
// each lane's code chunks once and dotting them with every entry.
template <int SKIP>
__global__ void __launch_bounds__(256) gu_grouped_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                         const int32_t* __restrict__ grp_start,
                                                         const int32_t* __restrict__ n_groups,
                                                         const int32_t* __restrict__ ent_tok,
                                                         const uint8_t* __restrict__ x_q8_0,
                                                         const float* __restrict__ x_scales,
                                                         float* __restrict__ gate_up, int cap_entries) {
    __shared__ int xs_q[GMAX][H / 4];          // the entries' int8 activations as words (2560 B each)
    __shared__ float xs_d[GMAX][H / 32];
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g] + SKIP, ne = min(grp_start[g + 1] - e0, GMAX);
    if (SKIP > 0 && ne <= 0) return;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    for (int i = t; i < ne * (H / 32); i += blockDim.x) {
        const int k = i / (H / 32), c = i - k * (H / 32);
        const uint8_t* xb = x_q8_0 + (size_t) ent_tok[e0 + k] * (size_t) (H / 32) * 34 + (size_t) c * 34;
        xs_d[k][c] = x_scales ? x_scales[(size_t) ent_tok[e0 + k] * (H / 32) + c] : f16_at(xb);
        const int8_t* q = (const int8_t*) (xb + 2);
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            int v;
            memcpy(&v, q + 4 * w, 4);
            xs_q[k][c * 8 + w] = v;
        }
    }
    __syncthreads();
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const int row0 = blockIdx.x * GU_ROWS;
    for (int rr = warp; rr < GU_ROWS; rr += 8) {
        const int i = row0 + rr;
        const uint8_t* codes = blob + (size_t) i * ROW_GU;
        const uint8_t* scales = blob + O_GU_SCALES + (size_t) i * SC_GU * 2;
        uint2 cb[GU_CHUNKS];
        float dw[GU_CHUNKS];
#pragma unroll
        for (int q = 0; q < GU_CHUNKS; ++q) {
            const int c = lane + 32 * q;
            if (c < H / 32) {
                cb[q] = *(const uint2*) (codes + (size_t) c * 8);
                dw[q] = f16_at(scales + (size_t) (c >> 1) * 2);
            }
        }
        for (int k = 0; k < ne; ++k) {
            float acc = 0.0f;
#pragma unroll
            for (int q = 0; q < GU_CHUNKS; ++q) {
                const int c = lane + 32 * q;
                if (c >= H / 32) break;
                acc += chunk_dot(cb[q], &xs_q[k][c * 8], dw[q], xs_d[k][c]);
            }
            const float sum = warp_sum(acc);
            if (lane == 0) {
                const int e = e0 + k, r = i >> 1;
                const size_t base = (i & 1) ? ((size_t) cap_entries * FF + (size_t) e * FF) : ((size_t) e * FF);
                gate_up[base + (size_t) r] = sum;
            }
        }
    }
}

// Down: a block = D_ROWS rows of ONE group; the entries' quantized intermediates staged once.  A down row is 20
// chunks, so lanes 0..19 each hold one chunk, as in the per-entry kernel.
template <int SKIP>
__global__ void __launch_bounds__(256) down_grouped_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                           const int32_t* __restrict__ grp_start,
                                                           const int32_t* __restrict__ n_groups,
                                                           const int32_t* __restrict__ ent_dst,
                                                           const uint8_t* __restrict__ h_q8_0,
                                                           const float* __restrict__ h_scales,
                                                           float* __restrict__ out) {
    __shared__ int hs_q[GMAX][FF / 4];
    __shared__ float hs_d[GMAX][FF / 32];
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g] + SKIP, ne = min(grp_start[g + 1] - e0, GMAX);
    if (SKIP > 0 && ne <= 0) return;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    for (int i = t; i < ne * (FF / 32); i += blockDim.x) {
        const int k = i / (FF / 32), c = i - k * (FF / 32);
        const uint8_t* xb = h_q8_0 + (size_t) (e0 + k) * (size_t) (FF / 32) * 34 + (size_t) c * 34;
        hs_d[k][c] = h_scales ? h_scales[(size_t) (e0 + k) * (FF / 32) + c] : f16_at(xb);
        const int8_t* q = (const int8_t*) (xb + 2);
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            int v;
            memcpy(&v, q + 4 * w, 4);
            hs_q[k][c * 8 + w] = v;
        }
    }
    __syncthreads();
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const int row0 = blockIdx.x * D_ROWS;
    for (int rr = warp; rr < D_ROWS; rr += 8) {
        const int r = row0 + rr;
        const uint8_t* codes = blob + O_D_CODES + (size_t) r * ROW_D;
        const uint8_t* scales = blob + O_D_SCALES + (size_t) r * SC_D * 2;
        const int c = lane;
        uint2 cb = make_uint2(0, 0);
        float dw = 0.0f;
        if (c < FF / 32) {
            cb = *(const uint2*) (codes + (size_t) c * 8);
            dw = f16_at(scales + (size_t) (c >> 1) * 2);
        }
        for (int k = 0; k < ne; ++k) {
            float acc = 0.0f;
            if (c < FF / 32) acc += chunk_dot(cb, &hs_q[k][c * 8], dw, hs_d[k][c]);
            const float sum = warp_sum(acc);
            if (lane == 0) out[(size_t) ent_dst[e0 + k] * H + r] = sum;
        }
    }
}

// ---- The grouped kernels with the staged activations laid out for the reads.
//
// **THE LAYOUT ABOVE IS AN 8-WAY BANK CONFLICT.**  `xs_q[k][c * 8 + j]` puts lane `L`'s word `j` at `8L + j`, so
// lanes `L, L + 4, L + 8, ...` share a bank on every one of the chunk's eight reads; and `chunk_dot` recomputes
// `hx`, which depends on the entry alone, for every row.  Here the staging writes, once per (entry, chunk):
//
//     xs_w[j][k * NC + c]   word j of the regrouped chunk (`load_x_chunk`), so lanes read consecutive words
//     xs_dh[k * NC + c]     (dx, hx) - one 64-bit read per chunk instead of a float and eight `dp4a`
//
// Word-major, the staging writes are consecutive as well (the flat index `i` IS `k * NC + c`), so neither side
// conflicts and no padding is needed.  Each warp takes its rows TWO at a time (adjacent rows: gate `r` and up `r`,
// or down rows `r, r + 1`), so every shared read serves both; the codes of both are expanded once per row, not
// once per entry.  Per (row, entry) the lane's chunks, their order, the float expression and the shuffle
// reduction are the kernels' above, so every output is bitwise theirs.

// Stages chunk `i = k * NC + c` of a group: its regrouped words at `xs_w[j * STRIDE + i]` and (dx, hx) at `xs_dh[i]`.
template <int STRIDE>
__device__ __forceinline__ void stage_chunk(const uint8_t* __restrict__ xb, float dx, int* xs_w, int2* xs_dh, int i) {
    int X[8];
    const int hx = load_x_chunk(xb, X);
#pragma unroll
    for (int j = 0; j < 8; ++j) xs_w[j * STRIDE + i] = X[j];
    xs_dh[i] = make_int2(__float_as_int(dx), hx);
}

template <int SKIP>
__global__ void __launch_bounds__(256) gu_grouped_t_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                           const int32_t* __restrict__ grp_start,
                                                           const int32_t* __restrict__ n_groups,
                                                           const int32_t* __restrict__ ent_tok,
                                                           const uint8_t* __restrict__ x_q8_0,
                                                           const float* __restrict__ x_scales,
                                                           float* __restrict__ gate_up, int cap_entries) {
    constexpr int NC = H / 32;
    __shared__ int xs_w[8 * GMAX * NC];          // 20 KB: word j of entry k's chunk c at [j][k * NC + c]
    __shared__ int2 xs_dh[GMAX * NC];            // 5 KB: (dx as bits, hx)
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g] + SKIP, ne = min(grp_start[g + 1] - e0, GMAX);
    if (SKIP > 0 && ne <= 0) return;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    for (int i = t; i < ne * NC; i += blockDim.x) {
        const int k = i / NC, c = i - k * NC;
        const int tok = ent_tok[e0 + k];
        const uint8_t* xb = x_q8_0 + (size_t) tok * (size_t) NC * 34 + (size_t) c * 34;
        const float dx = x_scales ? x_scales[(size_t) tok * NC + c] : f16_ld(xb);
        stage_chunk<GMAX * NC>(xb, dx, xs_w, xs_dh, i);
    }
    __syncthreads();
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const int row0 = blockIdx.x * GU_ROWS;                 // even: a pair is always (gate r, up r)
    for (int pp = warp; pp < GU_ROWS / 2; pp += 8) {
        const int i = row0 + 2 * pp;
        const uint8_t* codes = blob + (size_t) i * ROW_GU;
        const uint8_t* scales = blob + O_GU_SCALES + (size_t) i * SC_GU * 2;
        int m0[GU_CHUNKS][8], m1[GU_CHUNKS][8];
        float dw0[GU_CHUNKS], dw1[GU_CHUNKS];
#pragma unroll
        for (int q = 0; q < GU_CHUNKS; ++q) {
            const int c = lane + 32 * q;
            if (c < NC) {
                expand_codes(*(const uint2*) (codes + (size_t) c * 8), m0[q]);
                expand_codes(*(const uint2*) (codes + ROW_GU + (size_t) c * 8), m1[q]);
                dw0[q] = f16_ld(scales + (size_t) (c >> 1) * 2);
                dw1[q] = f16_ld(scales + SC_GU * 2 + (size_t) (c >> 1) * 2);
            }
        }
        for (int k = 0; k < ne; ++k) {
            float acc0 = 0.0f, acc1 = 0.0f;
#pragma unroll
            for (int q = 0; q < GU_CHUNKS; ++q) {
                const int c = lane + 32 * q;
                if (c >= NC) break;
                const int at = k * NC + c;
                int X[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) X[j] = xs_w[j * (GMAX * NC) + at];
                const int2 dh = xs_dh[at];
                const float dx = __int_as_float(dh.x);
                acc0 += dw0[q] * dx * (float) (chunk_s(m0[q], X) - dh.y);
                acc1 += dw1[q] * dx * (float) (chunk_s(m1[q], X) - dh.y);
            }
            const float s0 = warp_sum(acc0);
            const float s1 = warp_sum(acc1);
            if (lane == 0) {
                const int e = e0 + k, r = i >> 1;
                gate_up[(size_t) e * FF + (size_t) r] = s0;
                gate_up[(size_t) cap_entries * FF + (size_t) e * FF + (size_t) r] = s1;
            }
        }
    }
}

template <int SKIP>
__global__ void __launch_bounds__(256) down_grouped_t_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                             const int32_t* __restrict__ grp_start,
                                                             const int32_t* __restrict__ n_groups,
                                                             const int32_t* __restrict__ ent_dst,
                                                             const uint8_t* __restrict__ h_q8_0,
                                                             const float* __restrict__ h_scales,
                                                             float* __restrict__ out) {
    constexpr int NC = FF / 32;
    __shared__ int hs_w[8 * GMAX * NC];
    __shared__ int2 hs_dh[GMAX * NC];
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g] + SKIP, ne = min(grp_start[g + 1] - e0, GMAX);
    if (SKIP > 0 && ne <= 0) return;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    for (int i = t; i < ne * NC; i += blockDim.x) {
        const int k = i / NC, c = i - k * NC;
        const uint8_t* xb = h_q8_0 + (size_t) (e0 + k) * (size_t) NC * 34 + (size_t) c * 34;
        const float dx = h_scales ? h_scales[(size_t) (e0 + k) * NC + c] : f16_ld(xb);
        stage_chunk<GMAX * NC>(xb, dx, hs_w, hs_dh, i);
    }
    __syncthreads();
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const int row0 = blockIdx.x * D_ROWS;
    for (int pp = warp; pp < D_ROWS / 2; pp += 8) {
        const int r = row0 + 2 * pp;
        const uint8_t* codes = blob + O_D_CODES + (size_t) r * ROW_D;
        const uint8_t* scales = blob + O_D_SCALES + (size_t) r * SC_D * 2;
        const int c = lane;                                // lanes 0..19 hold one chunk each, as above
        int m0[8], m1[8];
        float dw0 = 0.0f, dw1 = 0.0f;
        if (c < NC) {
            expand_codes(*(const uint2*) (codes + (size_t) c * 8), m0);
            expand_codes(*(const uint2*) (codes + ROW_D + (size_t) c * 8), m1);
            dw0 = f16_ld(scales + (size_t) (c >> 1) * 2);
            dw1 = f16_ld(scales + SC_D * 2 + (size_t) (c >> 1) * 2);
        }
        for (int k = 0; k < ne; ++k) {
            float acc0 = 0.0f, acc1 = 0.0f;
            if (c < NC) {
                const int at = k * NC + c;
                int X[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) X[j] = hs_w[j * (GMAX * NC) + at];
                const int2 dh = hs_dh[at];
                const float dx = __int_as_float(dh.x);
                acc0 += dw0 * dx * (float) (chunk_s(m0, X) - dh.y);
                acc1 += dw1 * dx * (float) (chunk_s(m1, X) - dh.y);
            }
            const float s0 = warp_sum(acc0);
            const float s1 = warp_sum(acc1);
            if (lane == 0) {
                const size_t o = (size_t) ent_dst[e0 + k] * H + (size_t) r;
                out[o] = s0;
                out[o + 1] = s1;
            }
        }
    }
}
}  // namespace

namespace {
// Plan v0.3 P6: groups built on the device when every expert is resident at `base + id * blob` (the MTP layer):
// one block of 128 threads, groups in first-appearance order, entries of a group in routing order.
__global__ void group_resident_kernel(const int32_t* __restrict__ ids, int n, int k_per_tok, const uint8_t* base,
                                      long long blob, unsigned long long* __restrict__ grp_ptr,
                                      int32_t* __restrict__ grp_start, int32_t* __restrict__ counts,
                                      int32_t* __restrict__ ent_dst, int32_t* __restrict__ ent_tok) {
    __shared__ int e_s[128], first_s[128], size_s[128], gidx_s[128], gstart_s[129];
    const int i = threadIdx.x;
    const int e = i < n ? ids[i] : -1;
    e_s[i] = e;
    __syncthreads();
    int first = i, rank = 0, size = 0;
    if (i < n) {
        for (int j = 0; j < i; ++j)
            if (e_s[j] == e) { if (first == i) first = j; ++rank; }
        if (first == i)
            for (int j = i; j < n; ++j) size += e_s[j] == e;
    }
    first_s[i] = first;
    size_s[i] = (i < n && first == i) ? size : 0;
    __syncthreads();
    if (i == 0) {
        int gi = 0, acc = 0;
        for (int j = 0; j < n; ++j)
            if (first_s[j] == j) {
                gidx_s[j] = gi;
                gstart_s[gi] = acc;
                grp_ptr[gi] = (unsigned long long) (base + (size_t) e_s[j] * (size_t) blob);
                grp_start[gi] = acc;
                acc += size_s[j];
                ++gi;
            }
        grp_start[gi] = acc;
        counts[0] = gi;
        counts[1] = acc;
    }
    __syncthreads();
    if (i < n) {
        const int at = gstart_s[gidx_s[first]] + rank;
        ent_dst[at] = i;
        ent_tok[at] = i / k_per_tok;
    }
}
}  // namespace

void moe_group_resident(const int32_t* ids, int n, int k_per_tok, const uint8_t* base, int64_t blob,
                        unsigned long long* grp_ptr, int32_t* grp_start, int32_t* counts, int32_t* ent_dst,
                        int32_t* ent_tok, void* stream) {
    if (n < 1 || n > 128) { std::fprintf(stderr, "moe_group_resident: n must be 1..128\n"); std::exit(1); }
    group_resident_kernel<<<1, 128, 0, (cudaStream_t) stream>>>(ids, n, k_per_tok, base, (long long) blob, grp_ptr,
                                                               grp_start, counts, ent_dst, ent_tok);
    check("moe_group_resident", stream);
}

static void grouped_s2_impl(const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                            const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries,
                            const uint8_t* x_q8_0, const float* x_scales, void* scratch, float* out, void* stream,
                            int max_group) {
    if (cap_groups <= 0 || cap_entries <= 0) return;
    if (max_group < 1 || max_group > kGroupPasses * GMAX) {
        std::fprintf(stderr, "moe_grouped_s2: a group of %d entries, at most %d\n", max_group, kGroupPasses * GMAX);
        std::exit(1);
    }
    // a group of more than GMAX entries (a window of 9..16 rows, wide builds only): the entries from the GMAX-th on, too
    constexpr bool kWide = kGroupPasses > 1;
    const bool second = kWide && max_group > GMAX;
    cudaStream_t cs = (cudaStream_t) stream;
    const uint64_t gu_bytes = ((uint64_t) cap_entries * (uint64_t) (2 * FF) * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) cap_entries * (uint64_t) (FF / 32) * 34 + 15) & ~15ull;
    float* gate_up = (float*) scratch;
    uint8_t* h_q8_0 = (uint8_t*) scratch + gu_bytes;
    float* h_scales = (float*) ((uint8_t*) scratch + gu_bytes + q8_bytes);
    const bool fast = new_grouped(x_q8_0, scratch);
    {
        const dim3 grid((unsigned) (2 * FF / GU_ROWS), (unsigned) cap_groups);
        if (fast) {
            gu_grouped_t_kernel<0><<<grid, 256, 0, cs>>>(grp_ptr, grp_start, n_groups, ent_tok, x_q8_0, x_scales,
                                                         gate_up, (int) cap_entries);
            if constexpr (kWide)
                if (second)
                    gu_grouped_t_kernel<GMAX><<<grid, 256, 0, cs>>>(grp_ptr, grp_start, n_groups, ent_tok, x_q8_0, x_scales,
                                                                    gate_up, (int) cap_entries);
        } else {
            gu_grouped_kernel<0><<<grid, 256, 0, cs>>>(grp_ptr, grp_start, n_groups, ent_tok, x_q8_0, x_scales, gate_up,
                                                       (int) cap_entries);
            if constexpr (kWide)
                if (second)
                    gu_grouped_kernel<GMAX><<<grid, 256, 0, cs>>>(grp_ptr, grp_start, n_groups, ent_tok, x_q8_0, x_scales,
                                                                  gate_up, (int) cap_entries);
        }
        check("moe_grouped_s2/gu", stream);
    }
    // #783 PR-k (stuchapin909): SwiGLU and the scaled Q8_0 quantize of its result in one kernel (gate_up still gets the
    // activation written back, so the parity checks read the same buffer); STRATA_S2_SWIGLU_Q8=0 keeps the two launches
    static const bool swiglu_q8 = [] {
        const char* e = std::getenv("STRATA_S2_SWIGLU_Q8");
        return e == nullptr || e[0] != '0';
    }();
    static_assert(FF % 32 == 0, "the fused SwiGLU + Q8_0 kernel works in whole 32-value blocks");
    if (fast && x_scales != nullptr && swiglu_q8) {
        const long long pairs = cap_entries * (long long) FF;
        const long long n_blocks = pairs >> 5;
        const int warps = THREADS / 32;
        swiglu_quantize_q8_0_scaled_kernel<<<(unsigned) ((n_blocks + warps - 1) / warps), THREADS, 0, cs>>>(
            gate_up, pairs, h_q8_0, h_scales);
        check("moe_grouped_s2/swiglu_q8", stream);
    } else {
        const long long pairs = cap_entries * (long long) FF;
#if defined(STRATA_EXPERIMENTAL_SM60)
        if (strata::kernels::fuse_decode()) {
        if (x_scales != nullptr)
            swilu_quantize_q8_0_scaled(gate_up, gate_up + pairs, gate_up, pairs, 1, h_q8_0, h_scales, stream);
        else
            swilu_quantize_q8_0(gate_up, gate_up + pairs, gate_up, pairs, 1, h_q8_0, stream);
        } else
#endif
        {
        swiglu_kernel<<<(unsigned) ((pairs + THREADS - 1) / THREADS), THREADS, 0, cs>>>(gate_up, pairs);
        check("moe_grouped_s2/swiglu", stream);
        if (x_scales != nullptr) quantize_q8_0_scaled(gate_up, h_q8_0, h_scales, cap_entries * (int64_t) FF, stream);
        else quantize_q8_0(gate_up, h_q8_0, cap_entries * (int64_t) FF, stream);
        }
    }
    {
        const dim3 grid((unsigned) (H / D_ROWS), (unsigned) cap_groups);
        const float* hs = x_scales != nullptr ? h_scales : nullptr;
        if (fast) {
            down_grouped_t_kernel<0><<<grid, 256, 0, cs>>>(grp_ptr, grp_start, n_groups, ent_dst, h_q8_0, hs, out);
            if constexpr (kWide)
                if (second) down_grouped_t_kernel<GMAX><<<grid, 256, 0, cs>>>(grp_ptr, grp_start, n_groups, ent_dst, h_q8_0, hs, out);
        } else {
            down_grouped_kernel<0><<<grid, 256, 0, cs>>>(grp_ptr, grp_start, n_groups, ent_dst, h_q8_0, hs, out);
            if constexpr (kWide)
                if (second) down_grouped_kernel<GMAX><<<grid, 256, 0, cs>>>(grp_ptr, grp_start, n_groups, ent_dst, h_q8_0, hs, out);
        }
        check("moe_grouped_s2/down", stream);
    }
}

void moe_grouped_s2(const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                    const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries,
                    const uint8_t* x_q8_0, const float* x_scales, void* scratch, float* out, void* stream) {
    grouped_s2_impl(grp_ptr, grp_start, n_groups, ent_dst, ent_tok, cap_groups, cap_entries, x_q8_0, x_scales, scratch, out, stream,
                    GMAX);
}

#if STRATA_VERIFY_MAX_T > 8
void moe_grouped_s2_wide(const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                         const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries,
                         const uint8_t* x_q8_0, const float* x_scales, void* scratch, float* out, void* stream,
                         int max_group) {
    grouped_s2_impl(grp_ptr, grp_start, n_groups, ent_dst, ent_tok, cap_groups, cap_entries, x_q8_0, x_scales, scratch, out, stream,
                    max_group);
}
#endif

__global__ void add_hits_mapped_kernel(float* __restrict__ parts, const float* __restrict__ hit_out,
                                        const volatile float* __restrict__ ymiss, const int32_t* __restrict__ dst,
                                        const int32_t* __restrict__ count, const volatile uint32_t* __restrict__ wait_flag,
                                        uint32_t wait_value, int n_embd, int64_t rows_total) {
    // LATENT-CUT-A: wait_flag + copy_rows_from_mapped + moe_hit_add fused. The grid walks ALL
    // routed rows r (one block per row, like the copy kernel). Row r in dst (the GPU's share):
    // parts[r] = 0 + hit_out[r] (the copy kernel zeroed it, moe_hit_add added hit - one
    // rounding, kept here as the single add). Row r NOT in dst (the CPU pool's):
    // parts[r] = ymiss[r] (the copy stored it; the add kernel then left it - hit_out is zero
    // on those rows only if the U stage wrote none, so we add hit_out[r] the same way the old
    // moe_hit_add did NOT: it only walked dst. Faithful: store ymiss, no hit add).
    if (threadIdx.x == 0) {   // 143-amdk: the wait folded into every row block (as the original)
        while (*wait_flag < wait_value) strata_spin_pause();
        __threadfence_system();
    }
    __syncthreads();
    const int r = blockIdx.x;   // the routed position: m_ymiss is indexed by it
    __shared__ int hit;
    if (threadIdx.x == 0) {
        int h = 0;
        const int c = *count;
        for (int i = 0; i < c; ++i) h |= dst[i] == r;
        hit = h;
    }
    __syncthreads();
    float* p = parts + (int64_t) r * (int64_t) n_embd;
    if (hit) {
        // the GPU's own row: zero then add the hit (the old two-step exactly: copy zeroed, add added)
        for (int i = threadIdx.x; i < n_embd; i += blockDim.x)
            p[i] = hit_out[(int64_t) r * (int64_t) n_embd + i];
    } else {
        // the CPU pool's row: store the mapped row (the copy kernel's store), no hit add
        const volatile float* ym = ymiss + (int64_t) r * (int64_t) n_embd;
        for (int i = threadIdx.x; i < n_embd; i += blockDim.x)
            p[i] = ym[i];
    }
}

void moe_hit_add(float* parts, const float* hit_out, const int32_t* dst, const int32_t* count, int64_t cap,
                 int64_t n_embd, void* stream) {
    if (cap <= 0) return;
    const dim3 grid((unsigned) ((n_embd + 255) / 256 < 8 ? (n_embd + 255) / 256 : 8), (unsigned) cap);
    add_hits_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(parts, hit_out, dst, count, (int) n_embd);
    check("moe_hit_add", stream);
}

// knife-0: the u0 path's GPU-row write is an ASSIGNMENT (add_hits_mapped_kernel: p[i] = hit_out[r*H+i]), not an add:
// in the latent branch nothing writes parts for those rows before the merge (the VRAM share writes hit_out, the PCIe
// share lat_yl_grp), so an add there keeps the previous window's stale rows (a wrong first token).
__global__ void set_hits_kernel(float* __restrict__ parts, const float* __restrict__ hit_out,
                                const int32_t* __restrict__ dst, const int32_t* __restrict__ count, int n_embd) {
    const int h = blockIdx.y;
    if (h >= *count) return;
    const size_t row = (size_t) dst[h] * (size_t) n_embd;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_embd; i += gridDim.x * blockDim.x)
        parts[row + i] = hit_out[row + i];
}
void moe_hit_set(float* parts, const float* hit_out, const int32_t* dst, const int32_t* count, int64_t cap,
                 int64_t n_embd, void* stream) {
    if (cap <= 0) return;
    const dim3 grid((unsigned) ((n_embd + 255) / 256 < 8 ? (n_embd + 255) / 256 : 8), (unsigned) cap);
    set_hits_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(parts, hit_out, dst, count, (int) n_embd);
    check("moe_hit_set", stream);
}

void moe_hit_add_mapped(float* parts, const float* hit_out, const volatile float* ymiss, const int32_t* dst,
                        const int32_t* count, int64_t cap, int64_t n_embd, const volatile uint32_t* wait_flag,
                        uint32_t wait_value, void* stream) {
    // LATENT-CUT-A: the fused replacement of wait_flag_ge + copy_rows_from_mapped + moe_hit_add.
    // Grid: one block per ROUTED row (cap = n*K), 128 threads - the copy kernel's shape.
    if (cap <= 0) return;
    add_hits_mapped_kernel<<<(unsigned) cap, 128, 0, (cudaStream_t) stream>>>(parts, hit_out, ymiss, dst, count,
                                                                               wait_flag, wait_value, (int) n_embd, cap);
    check("moe_hit_add_mapped", stream);
}

void moe_hit_grouped_s2_cpu_order(const uint8_t* blob_base, const int32_t* slot_index,
                                 const int32_t* dst_index, int64_t n_hits, int64_t blob_bytes,
                                 const uint8_t* x_q8_0, void* scratch, float* out, void* stream,
                                 const float* x_scales, float* gate_up_trace) {
    if (n_hits <= 0) return;
    if (x_scales == nullptr) {
        std::fprintf(stderr, "moe_hit_grouped_s2_cpu_order requires fp32 activation scales\n");
        std::exit(1);
    }
    cudaStream_t cs = (cudaStream_t) stream;
    const uint64_t gu_bytes = ((uint64_t) n_hits * 2 * FF * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) n_hits * (FF / 32) * 34 + 15) & ~15ull;
    const uint64_t scale_bytes = ((uint64_t) n_hits * (FF / 32) * 4 + 15) & ~15ull;
    float* gu = (float*) scratch;
    uint8_t* hq = (uint8_t*) scratch + gu_bytes;
    float* hs = (float*) (hq + q8_bytes);
    float* hh = (float*) ((uint8_t*) hs + scale_bytes);
    float* xh = (float*) ((uint8_t*) hh + scale_bytes);
    activation_correction_kernel<<<(H / 32 + THREADS - 1) / THREADS, THREADS, 0, cs>>>(
        x_q8_0, x_scales, xh, H / 32);
    check("cpu_order/input_correction", stream);
    const int rows_per_block = THREADS / 8;
    cpu_order_projection_kernel<false><<<(unsigned) ((n_hits * 2 * FF + rows_per_block - 1) / rows_per_block),
                                            THREADS, 0, cs>>>(
        blob_base, slot_index, dst_index, blob_bytes, x_q8_0, x_scales, xh, gu, (int) n_hits);
    check("cpu_order/gate_up", stream);
    if (gate_up_trace != nullptr &&
        cudaMemcpyAsync(gate_up_trace, gu, (size_t) n_hits * 2 * FF * sizeof(float),
                        cudaMemcpyDeviceToDevice, cs) != cudaSuccess) {
        std::fprintf(stderr, "cpu_order/gate_up_trace copy failed\n");
        std::exit(1);
    }
    cpu_order_swiglu_kernel<<<(unsigned) ((n_hits * FF + THREADS - 1) / THREADS), THREADS, 0, cs>>>(
        gu, (int) n_hits * FF);
    check("cpu_order/swiglu", stream);
    cpu_order_quantize_kernel<<<(unsigned) ((n_hits * (FF / 32) + THREADS - 1) / THREADS), THREADS, 0, cs>>>(
        gu, hq, hs, hh, (int) n_hits * (FF / 32));
    check("cpu_order/intermediate_quantize", stream);
    cpu_order_projection_kernel<true><<<(unsigned) ((n_hits * H + rows_per_block - 1) / rows_per_block),
                                           THREADS, 0, cs>>>(
        blob_base, slot_index, dst_index, blob_bytes, hq, hs, hh, out, (int) n_hits);
    check("cpu_order/down", stream);
}

}  // namespace strata::kernels
