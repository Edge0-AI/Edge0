// src/kernels/cpu/tq_computed.hpp - the LUT-free ("computed") codebook of the experimental trellis types TQK5X / TQK4X
// (version 2): a state's four weights come from two 32-bit LCG words instead of the hashed 2048-point table.
//
//   r_i = ((s * A_i + B_i) & 0x8FFF8FFF) ^ 0x3B603B60        i = 0, 1   (s = the 15-bit state)
//   weights 0, 1 = the low and high fp16 of r_0, weights 2, 3 = those of r_1
//
// (QTIP's 3INST word, Tseng et al., Cornell-RelaxML/qtip lib/codebook/bitshift.py; the variant "3single" of the
// f-dsv41-cudalab study, which measured it at the SQNR of the table on real DeepSeek experts and a GEMV without
// shared-memory lookups.)  The mask fixes the top exponent bits, so |w| lies in [0.125, 2) and the int16 table value
// round(w * 2^13) used by the CPU kernels is an exact integer (1024..16383): decoder and encoder agree to the bit.
// The existing types (TQ2_T, TQK6, TQK7) and the table-based K5 / K4 (ids 226 / 227, "version 1") do not use it.
#pragma once

#include <cstdint>

#include "tq_states.hpp"

namespace strata::kernels::cpu::tq_detail {

inline constexpr uint32_t kCompA[2] = {89226354u, 0x9E3779B1u};
inline constexpr uint32_t kCompB[2] = {64248484u, 0x7F4A7C15u};
inline constexpr uint32_t kCompMask = 0x8FFF8FFFu, kCompXor = 0x3B603B60u;

inline uint32_t comp_word(uint32_t s, int i) { return ((s * kCompA[i] + kCompB[i]) & kCompMask) ^ kCompXor; }

/// The four weights of state s at scale 2^13 (int16, exact).
inline void comp_state4(uint32_t s, int16_t out[4]) {
    const uint32_t r0 = comp_word(s & 0x7fffu, 0), r1 = comp_word(s & 0x7fffu, 1);
    const uint16_t h[4] = {(uint16_t) (r0 & 0xffffu), (uint16_t) (r0 >> 16), (uint16_t) (r1 & 0xffffu), (uint16_t) (r1 >> 16)};
    for (int j = 0; j < 4; ++j) out[j] = (int16_t) (tq_half_to_float(h[j]) * 8192.0f);
}

}  // namespace strata::kernels::cpu::tq_detail
