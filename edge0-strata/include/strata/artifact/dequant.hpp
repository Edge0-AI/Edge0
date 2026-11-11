#pragma once
// src/artifact/dequant.cpp - P1.S3: scalar reference dequantizers, transcribed from ggml.
//
// P1.S3 asks for every encoding present in the Q2_0 file. This starts with the ones on the critical
// path (the routed experts are all Q2_0) plus the trivial element types, and is structured so the
// rest drop in. Source of truth per encoding is quoted in each function.
//
// Q2_0 is the one that matters, and its contract is settled from source in docs/q2_0-contract.md:
//   block = { fp16 d; uint8 qs[16] }  = 64 elements, 18 bytes
//   code {0,1,2,3} -> symbol {-1,0,+1,+2};  v = (code - 1) * d
//   4 codes per byte, LSB-first: byte j/4, bits (j%4)*2
//   QK2_0 = 64 is PROVEN from the artifact's own offset brackets, not assumed (the llama.cpp PR that
//   introduced the type used 128; this artifact is 64).
//
// Validation: `--check <file.gguf>` dequantizes real blocks and asserts the STRUCTURAL invariant that
// every output value is one of {-d, 0, +d, +2d} for that block's own scale. That catches bit-order,
// sign and stride bugs, which are the failure modes a dequantizer actually has. It is deliberately
// not a correctness-vs-ggml test - that needs ggml linked, which is P1.S3's next step.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#define STRATA_GGUF_MAIN_DISABLED 1
#include "strata/artifact/gguf_reader.hpp"

namespace strata {

// ---- fp16/bf16 -> fp32. Both handle subnormals and infinities; naive shifts do not.
inline float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, f;
    if (exp == 0) {
        if (man == 0)
            f = sign << 31;
        else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400u)) {
                man <<= 1;
                --exp;
            }
            man &= 0x3FFu;
            f = (sign << 31) | (exp << 23) | (man << 13);
        }
    } else if (exp == 31)
        f = (sign << 31) | 0x7F800000u | (man << 13);
    else
        f = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

inline float bf16_to_fp32(uint16_t h) {
    const uint32_t f = (uint32_t)h << 16;
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

inline uint16_t read_u16(const uint8_t* p) {
    return (uint16_t)(p[0] | (p[1] << 8));
}

// ---- Q2_0: 64 elements from an 18-byte block. Mirrors dequantize_row_q2_0 (ggml master 3cf03257).
inline void dequantize_q2_0(const uint8_t* block, float* out) {
    const float d = fp16_to_fp32(read_u16(block));
    const uint8_t* qs = block + 2;
    for (int j = 0; j < 64; ++j) {
        const int byte_index = j / 4;
        const int bit_offset = (j % 4) * 2;
        const int code = (qs[byte_index] >> bit_offset) & 0x03;
        out[j] = (float)(code - 1) * d; // code {0,1,2,3} -> symbol {-1,0,+1,+2}
    }
}

// ---- Q8_0: 32 elements from a 34-byte block.
inline void dequantize_q8_0(const uint8_t* block, float* out) {
    const float d = fp16_to_fp32(read_u16(block));
    const int8_t* qs = (const int8_t*)(block + 2);
    for (int j = 0; j < 32; ++j) out[j] = (float)qs[j] * d;
}

// ---- Q4_0: 32 elements from an 18-byte block. Low nibble is element j, high nibble is j+16.
inline void dequantize_q4_0(const uint8_t* block, float* out) {
    const float d = fp16_to_fp32(read_u16(block));
    const uint8_t* qs = block + 2;
    for (int j = 0; j < 16; ++j) {
        out[j] = (float)((int)(qs[j] & 0x0F) - 8) * d;
        out[j + 16] = (float)((int)(qs[j] >> 4) - 8) * d;
    }
}

// ---- Q5_0: 32 elements from a 22-byte block. Same nibble layout as Q4_0 plus a 5th bit per element
// packed in qh[4]; the 32 bits of qh are the high bits of elements 0..31 in order.
inline void dequantize_q5_0(const uint8_t* block, float* out) {
    const float d = fp16_to_fp32(read_u16(block));
    const uint8_t* qh = block + 2;
    const uint8_t* qs = block + 6;
    const uint32_t h =
        (uint32_t)qh[0] | ((uint32_t)qh[1] << 8) | ((uint32_t)qh[2] << 16) | ((uint32_t)qh[3] << 24);
    for (int j = 0; j < 16; ++j) {
        const int x0 = (int)(qs[j] & 0x0F) | (int)(((h >> j) & 1u) << 4);
        const int x1 = (int)(qs[j] >> 4) | (int)(((h >> (j + 16)) & 1u) << 4);
        out[j] = (float)(x0 - 16) * d;
        out[j + 16] = (float)(x1 - 16) * d;
    }
}
// ---- Q5_1: 32 elements from a 24-byte block: fp16 d, fp16 m, then Q5_0's qh[4] and qs[16]. The 5-bit codes are
// unsigned and the block's minimum m is added instead of Q5_0's fixed -16 offset (dequantize_row_q5_1).
inline void dequantize_q5_1(const uint8_t* block, float* out) {
    const float d = fp16_to_fp32(read_u16(block));
    const float m = fp16_to_fp32(read_u16(block + 2));
    const uint8_t* qh = block + 4;
    const uint8_t* qs = block + 8;
    const uint32_t h =
        (uint32_t)qh[0] | ((uint32_t)qh[1] << 8) | ((uint32_t)qh[2] << 16) | ((uint32_t)qh[3] << 24);
    for (int j = 0; j < 16; ++j) {
        const int x0 = (int)(qs[j] & 0x0F) | (int)(((h >> j) & 1u) << 4);
        const int x1 = (int)(qs[j] >> 4) | (int)(((h >> (j + 16)) & 1u) << 4);
        out[j] = (float)x0 * d + m;
        out[j + 16] = (float)x1 * d + m;
    }
}
// ---- IQ4_NL: 32 elements from an 18-byte block. Non-linear 4-bit codebook, NOT a linear grid, so
// the table must be exact. Copied verbatim from ggml-common.h `kvalues_iq4nl` at 3cf03257:
//     -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
// Layout matches dequantize_row_iq4_nl (ggml-quants.c): low nibble is element j, high nibble j+16.
static const int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                         1,    13,   25,  38,  53,  69,  89,  113};

inline void dequantize_iq4_nl(const uint8_t* block, float* out) {
    const float d = fp16_to_fp32(read_u16(block));
    const uint8_t* qs = block + 2;
    for (int j = 0; j < 16; ++j) {
        out[j] = d * (float)kvalues_iq4nl[qs[j] & 0x0F];
        out[j + 16] = d * (float)kvalues_iq4nl[qs[j] >> 4];
    }
}
// ---- Q6_K: 256 elements from a 210-byte super-block, transcribed from dequantize_row_q6_K
// (ggml-quants.c at 3cf03257). Layout: ql[128] low nibbles, qh[64] high 2 bits, scales[16] int8,
// fp16 d LAST. Processed in two 128-element halves, each advancing ql by 64, qh by 32, sc by 8.
// The `sc` indices are is+{0,2,4,6} with is = l/16 - so each 32-element group draws from 8 of the 16
// scales, and the four output positions within a 32-group use different scales. Easy to get subtly
// wrong, which is why it is transcribed rather than recalled.
inline void dequantize_q6_K(const uint8_t* block, float* out) {
    const uint8_t* ql = block;                       // 128 bytes
    const uint8_t* qh = block + 128;                 // 64
    const int8_t* sc = (const int8_t*)(block + 192); // 16
    const float d = fp16_to_fp32(read_u16(block + 208));
    float* y = out;
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = (int)((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = (int)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l + 0] = d * (float)sc[is + 0] * (float)q1;
            y[l + 32] = d * (float)sc[is + 2] * (float)q2;
            y[l + 64] = d * (float)sc[is + 4] * (float)q3;
            y[l + 96] = d * (float)sc[is + 6] * (float)q4;
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}
// ---- Q4_K: 256 elements from a 144-byte super-block. Layout is `fp16 d`, `fp16 dmin` (so the FIRST
// four bytes are the scale pair, unlike Q6_K where d is last), `scales[12]`, `qs[128]`.
// Both functions transcribed from ggml-quants.c at 3cf03257. The 6-bit scale/min unpacking in
// get_scale_min_k4 is the part that is impossible to recall reliably: the low 4 bits of the second
// half come from q[j+4], and the high 2 bits from the low halves' top bits (q[j-4] and q[j]).
static inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

inline void dequantize_q4_K(const uint8_t* block, float* out) {
    const float d = fp16_to_fp32(read_u16(block));      // d
    const float mn = fp16_to_fp32(read_u16(block + 2)); // dmin
    const uint8_t* scales = block + 4;                  // 12 bytes
    const uint8_t* q = block + 16;                      // 128 bytes
    float* y = out;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4(is + 0, scales, sc, m);
        const float d1 = d * (float)sc, m1 = mn * (float)m;
        get_scale_min_k4(is + 1, scales, sc, m);
        const float d2 = d * (float)sc, m2 = mn * (float)m;
        for (int l = 0; l < 32; ++l) *y++ = d1 * (float)(q[l] & 0x0F) - m1;
        for (int l = 0; l < 32; ++l) *y++ = d2 * (float)(q[l] >> 4) - m2;
        q += 32;
        is += 2;
    }
}
// ---- Q5_K: 256 elements from a 176-byte super-block: `fp16 d`, `fp16 dmin`, `scales[12]`,
// `qh[32]`, `qs[128]`. Same shape as Q4_K plus the 5th bit plane. Transcribed from
// dequantize_row_q5_K (ggml-quants.c, 3cf03257). The 5th bits are selected by a mask that SHIFTS LEFT
// BY 2 each 64-element group (u1 = 1,4,16,64; u2 = 2,8,32,128) - a recalled version would very likely
// have used a fixed mask or a shift by 1.
inline void dequantize_q5_K(const uint8_t* block, float* out) {
    const float d = fp16_to_fp32(read_u16(block));
    const float mn = fp16_to_fp32(read_u16(block + 2));
    const uint8_t* scales = block + 4; // 12
    const uint8_t* qh = block + 16;    // 32
    const uint8_t* ql = block + 48;    // 128
    float* y = out;
    int is = 0;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4(is + 0, scales, sc, m);
        const float d1 = d * (float)sc, m1 = mn * (float)m;
        get_scale_min_k4(is + 1, scales, sc, m);
        const float d2 = d * (float)sc, m2 = mn * (float)m;
        for (int l = 0; l < 32; ++l) *y++ = d1 * (float)((ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) - m1;
        for (int l = 0; l < 32; ++l) *y++ = d2 * (float)((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
        ql += 32;
        is += 2;
        u1 = (uint8_t)(u1 << 2);
        u2 = (uint8_t)(u2 << 2);
    }
}
// ---- Q3_K: 256 elements from a 110-byte super-block: `hmask[32]`, `qs[64]`, `scales[12]`, `fp16 d`
// LAST. Transcribed from dequantize_row_q3_K (ggml-quants.c, 3cf03257). This type has NO `dmin` - its
// scales are SIGNED 6-bit with a `- 32` bias - so it does not follow the Q4_K/Q5_K pattern at all.
// Two details that cannot be recalled:
//   * the 12 scale bytes are bit-repacked through three 32-bit masks before use (kmask1/2 below),
//     because 16 six-bit scales do not fit in 12 bytes linearly;
//   * the low 2 bits are offset by -4 when the high bit is CLEAR, i.e. the correction is NEGATIVE.
inline void dequantize_q3_K(const uint8_t* block, float* out) {
    const uint8_t* hm = block;     // 32
    const uint8_t* q = block + 32; // 64
    const float d_all = fp16_to_fp32(read_u16(block + 108));

    const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
    uint32_t aux[4] = {0, 0, 0, 0};
    std::memcpy(aux, block + 96, 12);
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t* scales = (const int8_t*)aux;

    float* y = out;
    int is = 0;
    uint8_t m = 1; // hoisted: ggml declares this BEFORE the n loop, so it does not reset between halves
    for (int n = 0; n < 256; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            float dl = d_all * (float)(scales[is++] - 32);
            for (int l = 0; l < 16; ++l)
                *y++ = dl * (float)((int8_t)((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4));
            dl = d_all * (float)(scales[is++] - 32);
            for (int l = 0; l < 16; ++l)
                *y++ = dl * (float)((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
            shift += 2;
            m = (uint8_t)(m << 1);
        }
        q += 32;
    }
}
// ---- IQ4_XS: 256 elements from a 136-byte super-block: `fp16 d`, `uint16 scales_h`, `scales_l[4]`,
// `qs[128]`. Transcribed from dequantize_row_iq4_xs (ggml-quants.c, 3cf03257). Reuses the IQ4_NL
// codebook. Eight 32-element groups; each group's scale is 6 bits assembled from the LOW nibble of
// `scales_l[ib/2]` (selected by ib%2) and TWO bits of `scales_h` at shift 2*ib - so the high bits come
// from a 16-bit field indexed by group, not from a byte array. Then `- 32`, as in Q3_K.
inline void dequantize_iq4_xs(const uint8_t* block, float* out) {
    const float d = fp16_to_fp32(read_u16(block));
    const uint16_t scales_h = read_u16(block + 2);
    const uint8_t* scales_l = block + 4; // 4
    const uint8_t* qs = block + 8;       // 128
    float* y = out;
    for (int ib = 0; ib < 8; ++ib) {
        const int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F) | (((scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * (float)(ls - 32);
        for (int j = 0; j < 16; ++j) {
            y[j + 0] = dl * (float)kvalues_iq4nl[qs[j] & 0x0F];
            y[j + 16] = dl * (float)kvalues_iq4nl[qs[j] >> 4];
        }
        y += 32;
        qs += 16;
    }
}
// ---- element types
inline void dequantize_f32(const uint8_t* p, float* out, int n) {
    std::memcpy(out, p, (size_t)n * 4);
}
inline void dequantize_f16(const uint8_t* p, float* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = fp16_to_fp32(read_u16(p + 2 * i));
}
inline void dequantize_bf16(const uint8_t* p, float* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = bf16_to_fp32(read_u16(p + 2 * i));
}

} // namespace strata
