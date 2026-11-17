// src/kernels/cpu/tq_states.hpp - the 32 trellis states of one TQ2_T / TQK6 / TQK7 block in plain C++ (all three
// kernels: scalar code beat byte-shuffle versions on AVX2 and AVX-512).  Internal to src/kernels/cpu/tq_*.cpp.
//
// TQ2_T (K = 8): state t = qs[(t + 31) % 32] << 8 | qs[t], masked to 15 bits.
// TQK (K = 6, 7): the 16 stream bits from bit (31 - t)*K, LSB first, circular modulo 32*K bits, masked to 15 bits.
// Every window is one unaligned 32-bit little-endian read inside the block's 4*K bytes, except the windows that run
// past the end (t = 0 and 1), which read a 64-bit view of the last four bytes followed by the first four.
#pragma once

#include <cstdint>
#include <cstring>

namespace strata::kernels::cpu::tq_detail {

inline constexpr uint32_t kTqStateMask = 0x7fffu;

inline uint32_t tq_ld32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;   // x86 (and every target Strata builds for) is little-endian
}

/// State t alone (t a constant after unrolling); `wrap` = tq_wrap<KB>(qs) for the TQK windows past the end.
template <int KB>
inline uint64_t tq_wrap(const uint8_t* qs) {
    if constexpr (KB == 8) return 0;
    else return (uint64_t) tq_ld32(qs + 4 * KB - 4) | ((uint64_t) tq_ld32(qs) << 32);
}
template <int KB>
inline uint32_t tq_state(const uint8_t* qs, uint64_t wrap, int t) {
    if constexpr (KB == 8) {
        return (((uint32_t) qs[(t + 31) & 31] << 8) | qs[t]) & kTqStateMask;
    } else {
        constexpr int NB = 4 * KB;
        const int off = (31 - t) * KB, b = off >> 3;
        const uint32_t v = b + 4 <= NB ? tq_ld32(qs + b) >> (off & 7) : (uint32_t) (wrap >> (off - 8 * (NB - 4)));
        return v & kTqStateMask;
    }
}

template <int KB>
inline void tq_states(const uint8_t* qs, uint32_t* st) {
    if constexpr (KB == 8) {
        st[0] = (((uint32_t) qs[31] << 8) | qs[0]) & kTqStateMask;
#if defined(__GNUC__)
#pragma GCC unroll 31
#endif
        for (int t = 1; t < 32; ++t) st[t] = (((uint32_t) qs[t - 1] << 8) | qs[t]) & kTqStateMask;
    } else {
        constexpr int NB = 4 * KB;
        const uint64_t wrap = (uint64_t) tq_ld32(qs + NB - 4) | ((uint64_t) tq_ld32(qs) << 32);
#if defined(__GNUC__)
#pragma GCC unroll 32
#endif
        for (int t = 0; t < 32; ++t) {
            const int off = (31 - t) * KB, b = off >> 3;
            const uint32_t v = b + 4 <= NB ? tq_ld32(qs + b) >> (off & 7) : (uint32_t) (wrap >> (off - 8 * (NB - 4)));
            st[t] = v & kTqStateMask;
        }
    }
}

/// TQP (Gyro-L): 64 pair-trellis steps; state t = the 16 stream bits from bit (63 - t)*KB, LSB first,
/// circular over 8*KB bytes (tqp_state in the fork's ggml-common.h).
template <int KB>
inline uint32_t tq_state_tqp(const uint8_t* qs, int t) {
    constexpr int NB = 8 * KB;
    const uint32_t off = (uint32_t) (63 - t) * (uint32_t) KB;
    const uint32_t i0 = off >> 3;
    const uint32_t i1 = (i0 + 1) % (uint32_t) NB;
    const uint32_t i2 = (i0 + 2) % (uint32_t) NB;
    const uint32_t w = (uint32_t) qs[i0] | ((uint32_t) qs[i1] << 8) | ((uint32_t) qs[i2] << 16);
    return (w >> (off & 7)) & kTqStateMask;
}
template <int KB>
inline void tq_states_tqp(const uint8_t* qs, uint32_t* st) {
#if defined(__GNUC__)
#pragma GCC unroll 64
#endif
    for (int t = 0; t < 64; ++t) st[t] = tq_state_tqp<KB>(qs, t);
}

/// fp16 bits -> float, exact (the block scales and the Q8_0 activation scales).
inline float tq_half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu, man = h & 0x3ffu, bits;
    if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (man << 13);
    } else if (exp != 0) {
        bits = sign | ((exp + 112u) << 23) | (man << 13);
    } else if (man == 0) {
        bits = sign;
    } else {   // subnormal: man * 2^-24, exact in float
        const float m = (float) man * (1.0f / 16777216.0f);
        return sign ? -m : m;
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

inline uint16_t tq_ld16(const uint8_t* p) {
    uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}

}  // namespace strata::kernels::cpu::tq_detail
