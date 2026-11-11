// src/kernels/cuda/s_gemv_pair.cu - gate/up dual GEMV + SwiGLU in ONE kernel (llama.cpp's `mul_mat_gated`).
//
// The shared expert ran THREE launches for its two projections and their SwiGLU: the gate GEMV, the up
// GEMV, then a pass over both halves.  This kernel computes `out[o] = silu(gate_row_o) * up_row_o` in one
// launch - one read of the activation for both rows - and never materialises `up`.
//
// BITWISE against the unfused pipeline: each row is accumulated by the SAME row function the standalone
// kernel uses (`strata/kernels/s_rowdev.cuh`, shared with nothing else), and the SwiGLU expression is the
// shared one (`strata/kernels/swiglu.cuh`).  A float store/load is lossless, so computing the product from
// the register values is what the two-kernel pipeline computed from memory.  `fusions_parity` checks this
// bitwise.
//
// The kernel supports every (gate form, up form) pair the S-family dispatch produces: a side is either the
// S2 row (`s2_gemv_q8`, code_bits 2, Q8_0 activations) or an S4/S8 row (`s_gemv_q8_*_split`, Q8_0 or Q8_K
// activations).  The per-side kind is a run-time integer picked ONCE per row, outside the accumulation
// loops.  Anything else (a non-32 `tpr`, an unknown width) falls back to the unfused sequence.
#include "strata/kernels/s_gemv.hpp"
#include "strata/kernels/s_rowdev.cuh"
#include "strata/kernels/swiglu.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

// side kinds: 0 = S2 row (Q8_0 act), 1 = split<Q8K>, 2 = split<Q8_0>; widths 4 and 8 fold into `kind`
// below via the template argument chosen at dispatch.
enum { SideS2 = 0, SideQ8K = 1, SideQ80 = 2 };

__device__ __forceinline__ float side_row(int kind, int cb, const uint8_t* x80, const uint8_t* x8k,
                                          const uint8_t* codes, const float* scales, const float* off,
                                          long long n_in, long long o, int lane, int bias, int codebook,
                                          int group_shift, int has_offset, int code_bits,
                                          float* s2partial, int tpr) {
    if (kind == SideS2) {
        row_s2_q8(x80, codes, scales, n_in, o, lane, tpr, s2partial);
        return s2partial[0];
    }
    const uint8_t* x = kind == SideQ8K ? x8k : x80;
    switch (code_bits) {
        case 4:
            return kind == SideQ8K ? row_s_q8<4, true>(x, codes, scales, off, n_in, o, lane, bias, codebook,
                                                       group_shift, has_offset)
                                   : row_s_q8<4, false>(x, codes, scales, off, n_in, o, lane, bias, codebook,
                                                        group_shift, has_offset);
        default:
            return kind == SideQ8K ? row_s_q8<8, true>(x, codes, scales, off, n_in, o, lane, bias, codebook,
                                                       group_shift, has_offset)
                                   : row_s_q8<8, false>(x, codes, scales, off, n_in, o, lane, bias, codebook,
                                                        group_shift, has_offset);
    }
}

struct SideArgs {
    int kind, code_bits, bias, codebook, group_shift, has_offset;
    const uint8_t *codes, *x80, *x8k;
    const float* scales;
    const float* off;
};

// ONE WARP PER OUTPUT PAIR: gate row o and up row o, two accumulations, one SwiGLU store.
__global__ void pair_kernel(SideArgs g, SideArgs u, float* __restrict__ out, long long n_in, long long n_out,
                            int swilu_kind, int tpr) {
    __shared__ float s2p[8][32];                     // the S2 tree, per warp (8 warps per block)
    const int warp_in_block = threadIdx.x >> 5;
    const long long o = (long long) blockIdx.x * (blockDim.x >> 5) + warp_in_block;
    if (o >= n_out) return;
    const int lane = threadIdx.x & 31;
    const float ag = side_row(g.kind, g.code_bits, g.x80, g.x8k, g.codes, g.scales, g.off, n_in, o, lane,
                              g.bias, g.codebook, g.group_shift, g.has_offset, g.code_bits,
                              s2p[warp_in_block], tpr);
    const float au = side_row(u.kind, u.code_bits, u.x80, u.x8k, u.codes, u.scales, u.off, n_in, o, lane,
                              u.bias, u.codebook, u.group_shift, u.has_offset, u.code_bits,
                              s2p[warp_in_block], tpr);
    if (lane != 0) return;
    float v;
    if (swilu_kind == 0) v = swilu_legacy(ag, au);
    else if (swilu_kind == 1) v = swilu_fast(ag, au);
    else v = swilu_native(ag, au);
    out[o] = v;
}

bool side_of(const SForm& f, int64_t n_in, const uint8_t* x80, const uint8_t* x8k, const uint8_t* codes,
             const float* scales, const float* off, SideArgs* out) {
    if (f.code_bits == 2) {
        out->kind = SideS2;
    } else if (f.code_bits == 4 || f.code_bits == 8) {
        out->kind = f.act_kind == 1 ? SideQ8K : SideQ80;
    } else {
        return false;
    }
    out->code_bits = f.code_bits;
    out->bias = f.code_bias;
    out->codebook = (int) f.codebook;
    out->group_shift = 0;
    while ((1 << out->group_shift) < f.group_elems) ++out->group_shift;
    if ((1 << out->group_shift) != f.group_elems) return false;
    out->has_offset = f.has_offset ? 1 : 0;
    out->codes = codes;
    out->scales = scales;
    out->off = off;
    out->x80 = x80;
    out->x8k = x8k;
    (void) n_in;
    return true;
}

}  // namespace

bool s_gemv_pair_silu(const uint8_t* x_q8_0, const uint8_t* x_q8k, const SForm& form_g, const uint8_t* codes_g,
                      const float* scales_g, const float* off_g, const SForm& form_u, const uint8_t* codes_u,
                      const float* scales_u, const float* off_u, float* out, int64_t n_in, int64_t n_out,
                      int threads_per_row, int swilu_kind, void* stream) {
    if (n_in <= 0 || n_out <= 0) return false;
    // the S2 row's tree is shaped by `threads_per_row`; the standalone kernel takes any tpr, but this
    // warp-per-pair form reproduces it only at 32 (the engine's TPR is 32 everywhere).
    if (threads_per_row != 32) return false;
    SideArgs g, u;
    if (!side_of(form_g, n_in, x_q8_0, x_q8k, codes_g, scales_g, off_g, &g)) return false;
    if (!side_of(form_u, n_in, x_q8_0, x_q8k, codes_u, scales_u, off_u, &u)) return false;

    const int threads = 256;                                  // 8 warps = 8 output pairs
    const unsigned grid = (unsigned) ((n_out + 7) / 8);
    pair_kernel<<<grid, threads, 0, (cudaStream_t) stream>>>(g, u, out, n_in, n_out, swilu_kind, threads_per_row);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "s_gemv_pair_silu launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}

}  // namespace strata::kernels
