// tools/dsv41_lab/computed_codes.hpp - LUT-free ("computed") codebooks for the 15-bit-state, 4-weights-a-step trellis, after
// QTIP (Tseng et al., Cornell-RelaxML/qtip lib/codebook/bitshift.py): 1MAD and 3INST decode a state with a 32-bit LCG and a few
// bit tricks instead of a table.  QTIP emits one weight per LCG word (V = 1); a step of our format has four weights, so every
// variant here uses 2 to 4 LCG words per state.  Host-side definition (the encoder takes it through the TQE_CODE environment
// variable, tools/trellis/tqenc.cpp) and the CUDA kernels use the same constants (tqk_gemv_c.cuh).
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace tqcode {

// LCG words of a state: x_i = s * kA[i] + kB[i]
inline constexpr uint32_t kA[4] = {89226354u, 0x9E3779B1u, 0x85EBCA6Bu, 0xC2B2AE35u};
inline constexpr uint32_t kB[4] = {64248484u, 0x7F4A7C15u, 0x165667B1u, 0x27D4EB2Fu};
inline constexpr uint32_t kMask3 = 0x8FFF8FFFu, kXor3 = 0x3B603B60u;   // QTIP 3INST: (x & 0x8FFF8FFF) ^ 0x3B603B60 = two fp16

inline float h2f(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    float f;
    if (e == 0) f = std::ldexp((float) m, -24);
    else if (e == 31) f = INFINITY;
    else f = std::ldexp(1.0f + m / 1024.0f, (int) e - 15);
    return sign ? -f : f;
}

// float -> fp16 -> float, round to nearest even (what HADD2 does to the exact sum of two fp16 values)
inline float rn16(float x) {
    if (x == 0.0f) return x;
    int e;
    std::frexp(x, &e);   // |x| in [2^(e-1), 2^e)
    const int q = (e - 1 < -14 ? -14 : e - 1) - 10;
    const float quantum = std::ldexp(1.0f, q);
    return std::nearbyint(x / quantum) * quantum;
}

inline uint32_t word3(uint32_t s, int i) { return ((s * kA[i] + kB[i]) & kMask3) ^ kXor3; }

enum Kind { kNone = 0, k1mad, k3inst, k3sum, k3single };

inline Kind kind_of(const char* n) {
    if (!n || !*n) return kNone;
    if (!std::strcmp(n, "1mad")) return k1mad;
    if (!std::strcmp(n, "3inst")) return k3inst;
    if (!std::strcmp(n, "3sum")) return k3sum;
    if (!std::strcmp(n, "3single")) return k3single;
    return kNone;
}

// the four (unnormalised) weights of state s
inline void code4(Kind k, uint32_t s, float out[4]) {
    s &= 0x7fffu;
    switch (k) {
        case k1mad:
            for (int i = 0; i < 4; ++i) {
                const uint32_t x = s * kA[i] + kB[i];
                const int y = (int) (x & 255) + (int) ((x >> 8) & 255) + (int) ((x >> 16) & 255) + (int) (x >> 24);
                out[i] = (float) (y - 510) / 147.800537109375f;
            }
            break;
        case k3inst:
            for (int i = 0; i < 4; ++i) { const uint32_t r = word3(s, i); out[i] = rn16(h2f((uint16_t) (r >> 16)) + h2f((uint16_t) (r & 0xffffu))); }
            break;
        case k3sum: {
            const uint32_t r0 = word3(s, 0), r1 = word3(s, 1), r2 = word3(s, 2), r3 = word3(s, 3);
            out[0] = rn16(h2f((uint16_t) (r0 & 0xffffu)) + h2f((uint16_t) (r1 & 0xffffu)));
            out[1] = rn16(h2f((uint16_t) (r0 >> 16)) + h2f((uint16_t) (r1 >> 16)));
            out[2] = rn16(h2f((uint16_t) (r2 & 0xffffu)) + h2f((uint16_t) (r3 & 0xffffu)));
            out[3] = rn16(h2f((uint16_t) (r2 >> 16)) + h2f((uint16_t) (r3 >> 16)));
            break;
        }
        default: {
            const uint32_t r0 = word3(s, 0), r1 = word3(s, 1);
            out[0] = h2f((uint16_t) (r0 & 0xffffu));
            out[1] = h2f((uint16_t) (r0 >> 16));
            out[2] = h2f((uint16_t) (r1 & 0xffffu));
            out[3] = h2f((uint16_t) (r1 >> 16));
        }
    }
}

}  // namespace tqcode
