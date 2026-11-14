// metal/src/ref_quant.cpp - see ref_quant.h.  The i-quant loops are transcribed from ggml-quants.c
// (dequantize_row_iq2_xxs / iq2_xs / iq2_s / iq3_xxs / iq3_s / iq4_nl / iq4_xs) at 3cf03257; the K-quants and the
// 32-blocks from Strata's include/strata/artifact/dequant.hpp (themselves transcribed from the same commit).
#include "ref_quant.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include "ggml-common-tq.h"                  // the trellis block structs and the codebook tq_lut_f16 (Gyro)
#include "../../src/kernels/cpu/tq_states.hpp"   // the CPU trellis state extraction (the reference decoder)

namespace ref {

const char* type_name(int t) {
    switch (t) {
        case F32: return "F32"; case F16: return "F16"; case BF16: return "BF16";
        case Q4_0: return "Q4_0"; case Q5_0: return "Q5_0"; case Q5_1: return "Q5_1"; case Q8_0: return "Q8_0";
        case Q3_K: return "Q3_K"; case Q4_K: return "Q4_K"; case Q5_K: return "Q5_K"; case Q6_K: return "Q6_K";
        case IQ2_XXS: return "IQ2_XXS"; case IQ2_XS: return "IQ2_XS"; case IQ2_S: return "IQ2_S";
        case IQ3_XXS: return "IQ3_XXS"; case IQ3_S: return "IQ3_S"; case IQ4_NL: return "IQ4_NL";
        case IQ4_XS: return "IQ4_XS"; case Q2_0: return "Q2_0"; case IQ1_S: return "IQ1_S"; case IQ1_M: return "IQ1_M";
        case TQ2_T: return "TQ2_T"; case TQK6: return "TQK6"; case TQK7: return "TQK7";
    }
    return "?";
}

int block_elems(int t) {
    switch (t) {
        case F32: case F16: case BF16: return 1;
        case Q4_0: case Q5_0: case Q5_1: case Q8_0: case IQ4_NL: return 32;
        case Q2_0: return 64;
        case TQ2_T: case TQK6: case TQK7: return 128;
        default: return 256;
    }
}

int block_bytes(int t) {
    switch (t) {
        case F32: return 4; case F16: case BF16: return 2;
        case Q4_0: return 18; case Q5_0: return 22; case Q5_1: return 24; case Q8_0: return 34; case IQ4_NL: return 18; case Q2_0: return 18;
        case Q3_K: return 110; case Q4_K: return 144; case Q5_K: return 176; case Q6_K: return 210;
        case IQ2_XXS: return 66; case IQ2_XS: return 74; case IQ2_S: return 82; case IQ3_XXS: return 98;
        case IQ3_S: return 110; case IQ4_XS: return 136; case IQ1_S: return 50; case IQ1_M: return 56;
        case TQ2_T: return 34; case TQK6: return 26; case TQK7: return 30;
    }
    std::fprintf(stderr, "ref: unknown type %d\n", t);
    std::exit(2);
}

float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, f;
    if (exp == 0) {
        if (man == 0)
            f = sign << 31;
        else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400u)) { man <<= 1; --exp; }
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

uint16_t f32_to_f16(float f) {
    _Float16 h = (_Float16)f;
    uint16_t r;
    std::memcpy(&r, &h, 2);
    return r;
}

float bf16_to_f32(uint16_t h) {
    const uint32_t f = (uint32_t)h << 16;
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

uint16_t f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

static inline uint16_t rd16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }

static void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// The trellis types (ggml 144 TQ2_T, 145 TQK6, 146 TQK7): 128 weights = 32 steps of 4; step t's state comes from
// the CPU engine's own extraction (src/kernels/cpu/tq_states.hpp), x = state * 0x9e3779b1, the two codebook points
// lut[x >> 21] and lut[(x >> 10) & 2047] (fp16 pairs) give weights 4t..4t+3, times the fp16 scale.
static void deq_tq(int kb, const uint8_t* b, float* y) {
    uint32_t st[32];
    if (kb == 8) strata::kernels::cpu::tq_detail::tq_states<8>(b + 2, st);
    else if (kb == 6) strata::kernels::cpu::tq_detail::tq_states<6>(b + 2, st);
    else strata::kernels::cpu::tq_detail::tq_states<7>(b + 2, st);
    const float d = f16_to_f32(rd16(b));
    for (int t = 0; t < 32; ++t) {
        const uint32_t x = st[t] * 0x9e3779b1u;
        const uint32_t pa = x >> 21, pb = (x >> 10) & 2047u;
        y[4 * t + 0] = f16_to_f32(tq_lut_f16[2 * pa]) * d;
        y[4 * t + 1] = f16_to_f32(tq_lut_f16[2 * pa + 1]) * d;
        y[4 * t + 2] = f16_to_f32(tq_lut_f16[2 * pb]) * d;
        y[4 * t + 3] = f16_to_f32(tq_lut_f16[2 * pb + 1]) * d;
    }
}

static void deq_block(int t, const uint8_t* b, float* y) {
    switch (t) {
        case TQ2_T: deq_tq(8, b, y); return;
        case TQK6: deq_tq(6, b, y); return;
        case TQK7: deq_tq(7, b, y); return;
        case Q2_0: {
            const float d = f16_to_f32(rd16(b));
            for (int j = 0; j < 64; ++j) y[j] = (float)(((b[2 + j / 4] >> ((j % 4) * 2)) & 3) - 1) * d;
            return;
        }
        case Q8_0: {
            const float d = f16_to_f32(rd16(b));
            for (int j = 0; j < 32; ++j) y[j] = (float)(int8_t)b[2 + j] * d;
            return;
        }
        case Q4_0: {
            const float d = f16_to_f32(rd16(b));
            for (int j = 0; j < 16; ++j) {
                y[j] = (float)((b[2 + j] & 15) - 8) * d;
                y[j + 16] = (float)((b[2 + j] >> 4) - 8) * d;
            }
            return;
        }
        case Q5_0: {
            const float d = f16_to_f32(rd16(b));
            uint32_t h;
            std::memcpy(&h, b + 2, 4);
            const uint8_t* qs = b + 6;
            for (int j = 0; j < 16; ++j) {
                const int x0 = (qs[j] & 15) | (((h >> j) & 1) << 4), x1 = (qs[j] >> 4) | (((h >> (j + 16)) & 1) << 4);
                y[j] = (float)(x0 - 16) * d;
                y[j + 16] = (float)(x1 - 16) * d;
            }
            return;
        }
        case Q5_1: {   // ggml dequantize_row_q5_1: y = x * d + m
            const float d = f16_to_f32(rd16(b)), m = f16_to_f32(rd16(b + 2));
            uint32_t h;
            std::memcpy(&h, b + 4, 4);
            const uint8_t* qs = b + 8;
            for (int j = 0; j < 16; ++j) {
                const int x0 = (qs[j] & 15) | (((h >> j) & 1) << 4), x1 = (qs[j] >> 4) | (((h >> (j + 16)) & 1) << 4);
                y[j] = (float)x0 * d + m;
                y[j + 16] = (float)x1 * d + m;
            }
            return;
        }
        case IQ4_NL: {
            const float d = f16_to_f32(rd16(b));
            for (int j = 0; j < 16; ++j) {
                y[j] = d * kvalues_iq4nl[b[2 + j] & 15];
                y[j + 16] = d * kvalues_iq4nl[b[2 + j] >> 4];
            }
            return;
        }
        case IQ4_XS: {
            const float d = f16_to_f32(rd16(b));
            const uint16_t sh = rd16(b + 2);
            const uint8_t* sl = b + 4;
            const uint8_t* qs = b + 8;
            for (int ib = 0; ib < 8; ++ib) {
                const int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 15) | (((sh >> (2 * ib)) & 3) << 4);
                const float dl = d * (ls - 32);
                for (int j = 0; j < 16; ++j) {
                    y[j] = dl * kvalues_iq4nl[qs[j] & 15];
                    y[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4];
                }
                y += 32;
                qs += 16;
            }
            return;
        }
        case Q6_K: {
            const uint8_t* ql = b;
            const uint8_t* qh = b + 128;
            const int8_t* sc = (const int8_t*)(b + 192);
            const float d = f16_to_f32(rd16(b + 208));
            for (int n = 0; n < 256; n += 128) {
                for (int l = 0; l < 32; ++l) {
                    const int is = l / 16;
                    const int q1 = (int)((ql[l] & 15) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    const int q2 = (int)((ql[l + 32] & 15) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    const int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    const int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    y[l] = d * sc[is] * q1;
                    y[l + 32] = d * sc[is + 2] * q2;
                    y[l + 64] = d * sc[is + 4] * q3;
                    y[l + 96] = d * sc[is + 6] * q4;
                }
                y += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
            return;
        }
        case Q4_K: {
            const float d = f16_to_f32(rd16(b)), mn = f16_to_f32(rd16(b + 2));
            const uint8_t* sc = b + 4;
            const uint8_t* q = b + 16;
            int is = 0;
            for (int j = 0; j < 256; j += 64) {
                uint8_t s, m;
                get_scale_min_k4(is, sc, s, m);
                const float d1 = d * s, m1 = mn * m;
                get_scale_min_k4(is + 1, sc, s, m);
                const float d2 = d * s, m2 = mn * m;
                for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 15) - m1;
                for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
                q += 32;
                is += 2;
            }
            return;
        }
        case Q5_K: {
            const float d = f16_to_f32(rd16(b)), mn = f16_to_f32(rd16(b + 2));
            const uint8_t* sc = b + 4;
            const uint8_t* qh = b + 16;
            const uint8_t* ql = b + 48;
            int is = 0;
            uint8_t u1 = 1, u2 = 2;
            for (int j = 0; j < 256; j += 64) {
                uint8_t s, m;
                get_scale_min_k4(is, sc, s, m);
                const float d1 = d * s, m1 = mn * m;
                get_scale_min_k4(is + 1, sc, s, m);
                const float d2 = d * s, m2 = mn * m;
                for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 15) + ((qh[l] & u1) ? 16 : 0)) - m1;
                for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
                ql += 32;
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
            return;
        }
        case Q3_K: {
            const uint8_t* hm = b;
            const uint8_t* q = b + 32;
            const float d_all = f16_to_f32(rd16(b + 108));
            const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
            uint32_t aux[4] = {0, 0, 0, 0};
            std::memcpy(aux, b + 96, 12);
            const uint32_t tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            const int8_t* scales = (const int8_t*)aux;
            int is = 0;
            uint8_t m = 1;
            for (int n = 0; n < 256; n += 128) {
                int shift = 0;
                for (int j = 0; j < 4; ++j) {
                    float dl = d_all * (scales[is++] - 32);
                    for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
                    dl = d_all * (scales[is++] - 32);
                    for (int l = 0; l < 16; ++l)
                        *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                    shift += 2;
                    m <<= 1;
                }
                q += 32;
            }
            return;
        }
        case IQ1_S: {   // ggml dequantize_row_iq1_s: dl = d (2 s + 1), y = dl (grid + delta)
            const float d = f16_to_f32(rd16(b));
            const uint8_t* qs = b + 2;
            for (int ib = 0; ib < 8; ++ib) {
                const uint16_t qh = rd16(b + 34 + 2 * ib);
                const float dl = d * (float)(2 * ((qh >> 12) & 7) + 1);
                const float delta = qh & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA;
                for (int l = 0; l < 4; ++l) {
                    const int8_t* grid = (const int8_t*)(iq1s_grid + (qs[l] | (((qh >> 3 * l) & 7) << 8)));
                    for (int j = 0; j < 8; ++j) y[j] = dl * (grid[j] + delta);
                    y += 8;
                }
                qs += 4;
            }
            return;
        }
        case IQ1_M: {   // ggml dequantize_row_iq1_m: the f16 scale in the top nibbles of the four u16 scales
            const uint8_t* qs = b;
            const uint8_t* qh = b + 32;
            const uint16_t* sc = (const uint16_t*)(b + 48);
            uint16_t s16 = (uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
            const float d = f16_to_f32(s16);
            for (int ib = 0; ib < 8; ++ib) {
                const float dl1 = d * (float)(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 7) + 1);
                const float dl2 = d * (float)(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 7) + 1);
                uint16_t idx[4];
                idx[0] = qs[0] | ((qh[0] << 8) & 0x700);
                idx[1] = qs[1] | ((qh[0] << 4) & 0x700);
                idx[2] = qs[2] | ((qh[1] << 8) & 0x700);
                idx[3] = qs[3] | ((qh[1] << 4) & 0x700);
                const float delta[4] = {qh[0] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA, qh[0] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA,
                                        qh[1] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA, qh[1] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA};
                for (int l = 0; l < 4; ++l) {
                    const int8_t* grid = (const int8_t*)(iq1s_grid + idx[l]);
                    for (int j = 0; j < 8; ++j) y[j] = (l < 2 ? dl1 : dl2) * (grid[j] + delta[l]);
                    y += 8;
                }
                qs += 4;
                qh += 2;
            }
            return;
        }
        case IQ2_XXS: {
            const float d = f16_to_f32(rd16(b));
            const uint8_t* qs = b + 2;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                uint32_t aux32[2];
                std::memcpy(aux32, qs + 8 * ib32, 8);
                const uint8_t* aux8 = (const uint8_t*)aux32;
                const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
                for (int l = 0; l < 4; ++l) {
                    const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[l]);
                    const uint8_t signs = ksigns_iq2xs[(aux32[1] >> 7 * l) & 127];
                    for (int j = 0; j < 8; ++j) y[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                    y += 8;
                }
            }
            return;
        }
        case IQ2_XS: {
            const float d = f16_to_f32(rd16(b));
            const uint8_t* q8 = b + 2;
            const uint8_t* scales = b + 66;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const float db0 = d * (0.5f + (scales[ib32] & 15)) * 0.25f;
                const float db1 = d * (0.5f + (scales[ib32] >> 4)) * 0.25f;
                for (int l = 0; l < 4; ++l) {
                    const uint16_t q = rd16(q8 + 2 * (4 * ib32 + l));
                    const uint8_t* grid = (const uint8_t*)(iq2xs_grid + (q & 511));
                    const uint8_t signs = ksigns_iq2xs[q >> 9];
                    const float db = l < 2 ? db0 : db1;
                    for (int j = 0; j < 8; ++j) y[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                    y += 8;
                }
            }
            return;
        }
        case IQ2_S: {
            const float d = f16_to_f32(rd16(b));
            const uint8_t* qs = b + 2;
            const uint8_t* qh = b + 66;
            const uint8_t* scales = b + 74;
            const uint8_t* signs = qs + 32;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const float db0 = d * (0.5f + (scales[ib32] & 15)) * 0.25f;
                const float db1 = d * (0.5f + (scales[ib32] >> 4)) * 0.25f;
                for (int l = 0; l < 4; ++l) {
                    const float dl = l < 2 ? db0 : db1;
                    const uint8_t* grid = (const uint8_t*)(iq2s_grid + (qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)));
                    for (int j = 0; j < 8; ++j) y[j] = dl * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1.f : 1.f);
                    y += 8;
                }
                qs += 4;
                signs += 4;
            }
            return;
        }
        case IQ3_XXS: {
            const float d = f16_to_f32(rd16(b));
            const uint8_t* qs = b + 2;
            const uint8_t* sas = b + 2 + 64;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                uint32_t aux32;
                std::memcpy(&aux32, sas + 4 * ib32, 4);
                const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
                for (int l = 0; l < 4; ++l) {
                    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * l) & 127];
                    const uint8_t* g1 = (const uint8_t*)(iq3xxs_grid + qs[2 * l + 0]);
                    const uint8_t* g2 = (const uint8_t*)(iq3xxs_grid + qs[2 * l + 1]);
                    for (int j = 0; j < 4; ++j) {
                        y[j + 0] = db * g1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
                        y[j + 4] = db * g2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
                    }
                    y += 8;
                }
                qs += 8;
            }
            return;
        }
        case IQ3_S: {
            const float d = f16_to_f32(rd16(b));
            const uint8_t* qs = b + 2;
            const uint8_t* qh = b + 66;
            const uint8_t* signs = b + 74;
            const uint8_t* scales = b + 106;
            for (int ib32 = 0; ib32 < 8; ib32 += 2) {
                const float db1 = d * (1 + 2 * (scales[ib32 / 2] & 15));
                const float db2 = d * (1 + 2 * (scales[ib32 / 2] >> 4));
                for (int l = 0; l < 4; ++l) {
                    const uint8_t* g1 = (const uint8_t*)(iq3s_grid + (qs[2 * l + 0] | ((qh[0] << (8 - 2 * l)) & 256)));
                    const uint8_t* g2 = (const uint8_t*)(iq3s_grid + (qs[2 * l + 1] | ((qh[0] << (7 - 2 * l)) & 256)));
                    for (int j = 0; j < 4; ++j) {
                        y[j + 0] = db1 * g1[j] * (signs[l] & kmask_iq2xs[j + 0] ? -1.f : 1.f);
                        y[j + 4] = db1 * g2[j] * (signs[l] & kmask_iq2xs[j + 4] ? -1.f : 1.f);
                    }
                    y += 8;
                }
                qs += 8;
                signs += 4;
                for (int l = 0; l < 4; ++l) {
                    const uint8_t* g1 = (const uint8_t*)(iq3s_grid + (qs[2 * l + 0] | ((qh[1] << (8 - 2 * l)) & 256)));
                    const uint8_t* g2 = (const uint8_t*)(iq3s_grid + (qs[2 * l + 1] | ((qh[1] << (7 - 2 * l)) & 256)));
                    for (int j = 0; j < 4; ++j) {
                        y[j + 0] = db2 * g1[j] * (signs[l] & kmask_iq2xs[j + 0] ? -1.f : 1.f);
                        y[j + 4] = db2 * g2[j] * (signs[l] & kmask_iq2xs[j + 4] ? -1.f : 1.f);
                    }
                    y += 8;
                }
                qh += 2;
                qs += 8;
                signs += 4;
            }
            return;
        }
    }
    std::fprintf(stderr, "ref: dequant of %s not implemented\n", type_name(t));
    std::exit(2);
}

void dequant(int t, const void* src, float* dst, size_t n) {
    const uint8_t* s = (const uint8_t*)src;
    if (t == F32) { std::memcpy(dst, s, n * 4); return; }
    if (t == F16) { for (size_t i = 0; i < n; ++i) dst[i] = f16_to_f32(rd16(s + 2 * i)); return; }
    if (t == BF16) { for (size_t i = 0; i < n; ++i) dst[i] = bf16_to_f32(rd16(s + 2 * i)); return; }
    const int be = block_elems(t), bb = block_bytes(t);
    for (size_t i = 0; i < n / be; ++i) deq_block(t, s + i * bb, dst + i * be);
}

void quantize_q8_0(const float* x, size_t n, uint8_t* out) {
    for (size_t b = 0; b < n / 32; ++b) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[b * 32 + j]));
        const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        const uint16_t h = f32_to_f16(d);
        uint8_t* o = out + b * 34;
        o[0] = (uint8_t)h;
        o[1] = (uint8_t)(h >> 8);
        for (int j = 0; j < 32; ++j) o[2 + j] = (uint8_t)(int8_t)std::roundf(x[b * 32 + j] * id);
    }
}

static void put16(uint8_t* p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }

void random_blocks(int t, uint8_t* dst, int rows, int cols, uint64_t seed, float scale) {
    Rng r(seed);
    const size_t n = (size_t)rows * cols;
    if (t == F32 || t == F16 || t == BF16) {
        for (size_t i = 0; i < n; ++i) {
            const float v = r.sym() * scale * 2.0f;
            if (t == F32) std::memcpy(dst + 4 * i, &v, 4);
            else put16(dst + 2 * i, t == F16 ? f32_to_f16(v) : f32_to_bf16(v));
        }
        return;
    }
    const int be = block_elems(t), bb = block_bytes(t);
    const size_t nb = n / be;
    for (size_t i = 0; i < nb; ++i) {
        uint8_t* b = dst + i * bb;
        for (int k = 0; k < bb; ++k) b[k] = (uint8_t)r.next();
        // the scale fields: magnitude so that |w| ~ scale.  Typical code magnitudes differ per type.
        const float u = 0.5f + r.uni();
        switch (t) {
            case Q2_0: put16(b, f32_to_f16(scale * u)); break;
            case Q8_0: put16(b, f32_to_f16(scale * u / 64)); break;
            case Q4_0: case IQ4_NL: put16(b, f32_to_f16(scale * u / (t == Q4_0 ? 4 : 64))); break;
            case Q5_0: put16(b, f32_to_f16(scale * u / 8)); break;
            case Q5_1: put16(b, f32_to_f16(scale * u / 16)); put16(b + 2, f32_to_f16(-scale * u)); break;
            case IQ4_XS: put16(b, f32_to_f16(scale * u / 64 / 16)); break;
            case Q4_K: case Q5_K: put16(b, f32_to_f16(scale * u / 32 / 8)); put16(b + 2, f32_to_f16(scale * u / 32 / 2)); break;
            case Q6_K: put16(b + 208, f32_to_f16(scale * u / 32 / 64)); break;
            case Q3_K: put16(b + 108, f32_to_f16(scale * u / 32 / 2)); break;
            case IQ2_XXS: case IQ2_XS: case IQ2_S: put16(b, f32_to_f16(scale * u / 4)); break;
            case IQ1_S: put16(b, f32_to_f16(scale * u / 2)); break;
            case IQ1_M: { uint16_t h = f32_to_f16(scale * u / 2); for (int j = 0; j < 4; ++j) { uint16_t w = (uint16_t)(b[48 + 2 * j] | (b[49 + 2 * j] << 8)); w = (uint16_t)((w & 0x0fff) | (((h >> (4 * j)) & 0xf) << 12)); b[48 + 2 * j] = (uint8_t)w; b[49 + 2 * j] = (uint8_t)(w >> 8); } break; }
            case IQ3_XXS: put16(b, f32_to_f16(scale * u / 8)); break;
            case IQ3_S: put16(b, f32_to_f16(scale * u / 64)); break;
            case TQ2_T: case TQK6: case TQK7: put16(b, f32_to_f16(scale * u / 1.5f)); break;   // codebook |point| ~ 1
            default: break;
        }
    }
}

}  // namespace ref
