// src/kernels/cpu/kq_scalar.cpp - scalar fallbacks for the edge0 latent CPU lane on non-x86 hosts
// (Windows ARM64, DGX Spark).  The AVX-2 versions live in kq_avx2.cpp and are chosen at run time by
// cpu_avx2_ok(); on a host without them these run instead - same numbers, one row/token at a time.
// The row math mirrors pool.cpp's static lat_*_row functions (the reference for the AVX2 kernels).
#include "strata/kernels/cpu/kq_avx2.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cmath>

namespace strata::kernels::cpu {

void lat_quant_q8_1_qs_avx2(const float* x, const float* d_ptr, int8_t* q) {
    // The AVX2 version's scalar contract (probe_qs.cpp): divide by d (not 1/d), round
    // ties-away-from-zero like roundf, clamp to [-127, 127].
    const float d = *d_ptr;
    for (int j = 0; j < 32; ++j) {
        const float v = x[j] / d;
        int r = (int) std::lroundf(v);   // roundf: half away from zero
        if (r > 127) r = 127;
        if (r < -127) r = -127;
        q[j] = (int8_t) r;
    }
}

void lat_q8_0_q8_1_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                        float* const* out, int r0, int r1) {
    const int nb = n / 32;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * row_bytes;
        for (int t = 0; t < nt; ++t) {
            const uint8_t* xq = (const uint8_t*) act[t];
            float acc = 0.0f;
            for (int b = 0; b < nb; ++b) {
                const uint8_t* wb = row + (size_t) b * 34;
                const uint8_t* xb = xq + (size_t) b * 36;
                const float dw = f32_from_f16((uint16_t)(wb[0] | (wb[1] << 8)));
                const float dx = f32_from_f16((uint16_t)(xb[0] | (xb[1] << 8)));
                const int8_t* ws = (const int8_t*)(wb + 2);
                const int8_t* xs = (const int8_t*)(xb + 4);
                int s = 0;
                for (int j = 0; j < 32; ++j) s += (int) xs[j] * (int) ws[j];
                acc += dw * dx * (float) s;
            }
            out[t][r] = acc;
        }
    }
}

void lat_q4_0_q8_1_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                        float* const* out, int r0, int r1) {
    const int nb = n / 32;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * row_bytes;
        for (int t = 0; t < nt; ++t) {
            const uint8_t* xq = (const uint8_t*) act[t];
            float acc = 0.0f;
            for (int b = 0; b < nb; ++b) {
                const uint8_t* wb = row + (size_t) b * 18;
                const uint8_t* xb = xq + (size_t) b * 36;
                const float dw = f32_from_f16((uint16_t)(wb[0] | (wb[1] << 8)));
                const float dx = f32_from_f16((uint16_t)(xb[0] | (xb[1] << 8)));
                const int8_t* xs = (const int8_t*)(xb + 4);
                int s = 0;
                for (int j = 0; j < 16; ++j) {
                    const int byte_ = wb[2 + j];
                    s += (int) xs[j] * ((int)(byte_ & 0xF) - 8);
                    s += (int) xs[j + 16] * ((int)(byte_ >> 4) - 8);
                }
                acc += dw * dx * (float) s;
            }
            out[t][r] = acc;
        }
    }
}

}  // namespace strata::kernels::cpu
