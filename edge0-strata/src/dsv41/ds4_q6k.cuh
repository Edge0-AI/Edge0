// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_q6k.cuh - one value of a ggml Q6_K superblock (256 values, 210 bytes: ql[128] the low 4 bits, qh[64] the
// high 2, scales[16] int8 a 16 values, d F16), as ggml's dequantize_row_q6_K computes it: d * scale * (q - 32), the two
// products in that order in F32 - the same bits (the Engram tables a GGUF may hold in Q6_K).
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

namespace strata::ds4 {

constexpr int kQ6kBytes = 210;

__device__ __forceinline__ float q6k_value(const uint8_t* __restrict__ b, int i) {
    const uint8_t* ql = b;
    const uint8_t* qh = b + 128;
    const int8_t* sc = (const int8_t*) (b + 192);
    // (the F16 scale's bits as a half: HIP's headers take no brace-initialized __half_raw from them)
    const float d = __half2float(__ushort_as_half((unsigned short) (b[208] | (b[209] << 8))));
    const int h = i >> 7, w = i & 127, quarter = w >> 5, l = w & 31;   // (ggml's loop: half h, 32-value quarters)
    ql += h * 64;
    qh += h * 32;
    sc += h * 8;
    const int lo = quarter & 1 ? ql[l + 32] : ql[l];
    const int q = ((quarter & 2 ? lo >> 4 : lo & 0xF) | (((qh[l] >> (2 * quarter)) & 3) << 4)) - 32;
    return __fmul_rn(__fmul_rn(d, (float) sc[(l >> 4) + 2 * quarter]), (float) q);
}

}  // namespace strata::ds4
