// tools/dsv41_lab/wmma_emu.hpp - host emulation of the gfx12 (RDNA4, R9700 = gfx1201) port of gemv_v3, written lane by lane exactly
// as the HIP kernel in hip/tqk_gemv_wmma.hip.cpp does it, so the data flow (layout "TQW", step -> k-slot map, window bits, shuffles, B
// fragments, D scaling) is checked on any machine against the double reference BEFORE it runs on the card.  The only assumption
// that needs the card is the lane layout of v_wmma_f32_16x16x16_f16 (hip/wmma_probe.hip.cpp checks it in one launch).
//
// Assumed layout, wave32, __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v8h a, v8h b, v8f c):
//   A[m][k]: lane l holds a[j] = A[m = l % 16][k = 8 * (l / 16) + j], j = 0..7        (no replication between the lane halves)
//   B[k][n]: lane l holds b[j] = B[k = 8 * (l / 16) + j][n = l % 16]
//   D[m][n]: lane l holds c[i] = D[m = 8 * (l / 16) + i][n = l % 16], i = 0..7
#pragma once
#include "computed_codes.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

namespace tqw {

struct Half8 { uint16_t h[8]; };
struct Float8 { float f[8]; };

// D = A * B + C on 32 lanes
inline void wmma_f32_16x16x16_f16(const Half8 a[32], const Half8 b[32], Float8 c[32]) {
    float A[16][16], B[16][16], D[16][16];
    for (int l = 0; l < 32; ++l)
        for (int j = 0; j < 8; ++j) {
            A[l % 16][8 * (l / 16) + j] = tqcode::h2f(a[l].h[j]);
            B[8 * (l / 16) + j][l % 16] = tqcode::h2f(b[l].h[j]);
        }
    for (int l = 0; l < 32; ++l)
        for (int i = 0; i < 8; ++i) D[8 * (l / 16) + i][l % 16] = c[l].f[i];
    for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 16; ++n) {
            double s = D[m][n];
            for (int k = 0; k < 16; ++k) s += (double) A[m][k] * B[k][n];
            D[m][n] = (float) s;
        }
    for (int l = 0; l < 32; ++l)
        for (int i = 0; i < 8; ++i) c[l].f[i] = D[8 * (l / 16) + i][l % 16];
}

// ---- layout TQW (gfx12): a tile = 16 rows, a pair P = blocks 2P, 2P+1.  Lane l = r + 16 kg of the wave owns row r and, of every
// block, the 16 steps u = 0..15 of the half kg (steps t = 16 kg + u): stream bits [(16 - 16 kg) K, (32 - 16 kg) K) of the block.
// Its two chunks (bk = 0, 1) are 32 K bits = K words:  words[((tile * NP + P) * K + j) * 32 + lane].
// Scales: sc[((tile * NP + P) * 16 + r) * 2 + bk] (u16 fp16 d of row r, block 2P + bk): a lane reads the 8 rows 8 kg .. 8 kg + 7 = 32 bytes.
inline void repack_tqw(int K, const uint8_t* w, int rows, int nblk, std::vector<uint32_t>& words, std::vector<uint16_t>& sc) {
    const int bsz = 2 + 4 * K, NP = nblk / 2, NT = rows / 16;
    words.assign((size_t) NT * NP * K * 32, 0u);
    sc.assign((size_t) NT * NP * 32, 0);
    for (int t = 0; t < NT; ++t)
        for (int P = 0; P < NP; ++P) {
            const size_t base = (size_t) t * NP + P;
            for (int r = 0; r < 16; ++r)
                for (int bk = 0; bk < 2; ++bk) {
                    const uint8_t* blk = w + ((size_t) (t * 16 + r) * nblk + 2 * P + bk) * bsz;
                    std::memcpy(&sc[(base * 16 + r) * 2 + bk], blk, 2);
                    for (int kg = 0; kg < 2; ++kg) {
                        const int lane = r + 16 * kg;
                        for (int b = 0; b < 16 * K; ++b) {   // chunk bit b -> record bit 16 K bk + b
                            const int src = (16 - 16 * kg) * K + b;
                            const uint32_t bit = (blk[2 + (src >> 3)] >> (src & 7)) & 1u;
                            const int dst = 16 * K * bk + b;
                            words[(base * K + (dst >> 5)) * 32 + lane] |= bit << (dst & 31);
                        }
                    }
                }
        }
}

// B fragments (prep_b of the CUDA path, gfx12 form): for column n < nb, block blk, mma m, k-group kg:  8 halves hi and 8 halves lo of
// x[n][blk * 128 + 4 * (16 kg + 2 m) + j], j = 0..7   (hi = fp16(x), lo = fp16(x - hi))
inline uint16_t f2h_rn(float f) {   // fp16 round to nearest even (host)
    uint32_t u; std::memcpy(&u, &f, 4);
    const uint32_t s = (u >> 16) & 0x8000u; int e = (int) ((u >> 23) & 0xff) - 127 + 15; uint32_t m = u & 0x7fffffu;
    if (((u >> 23) & 0xff) == 0xff) return (uint16_t) (s | 0x7c00u | (m ? 0x200u : 0));
    if (e >= 31) return (uint16_t) (s | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return (uint16_t) s;
        m |= 0x800000u; const int sh = 14 - e; uint32_t hm = m >> sh; const uint32_t rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (hm & 1))) ++hm;
        return (uint16_t) (s | hm);
    }
    uint32_t hm = m >> 13; const uint32_t rem = m & 0x1fffu;
    uint32_t v = ((uint32_t) e << 10) | hm;
    if (rem > 0x1000u || (rem == 0x1000u && (v & 1))) ++v;
    return (uint16_t) (s | v);
}

// the A fragment of a lane for one trellis step: (word 0, word 1) -> 4 fp16 (weights 4t .. 4t+3).  computed code "3single":
// the two LCG words as they are.  table: the two half2 entries of the hash.
using StepFn = void (*)(uint32_t state, uint32_t out[2]);

// The kernel, wave by wave.  x: nb columns of F32 activations [nb][in].  Returns y[nb][rows].
inline void gemv_tqw(int K, int nb, int rows, int in, const std::vector<uint32_t>& words, const std::vector<uint16_t>& sc,
                     const std::vector<float>& x, StepFn step, std::vector<float>& y, bool exact = true) {
    const int nblk = in / 128, NP = nblk / 2, NT = rows / 16;
    y.assign((size_t) nb * rows, 0.f);
    for (int t = 0; t < NT; ++t) {
        float tot[32][8] = {};
        for (int P = 0; P < NP; ++P) {
            const size_t base = (size_t) t * NP + P;
            // per lane: the record, the two chunk windows (8 steps each) with the partner bits
            uint64_t win[32][2][2];   // [lane][bk][half of the chunk: 0 = steps u 0..7, 1 = u 8..15]
            for (int lane = 0; lane < 32; ++lane) {
                uint32_t r[8];
                for (int j = 0; j < K; ++j) r[j] = words[(base * K + j) * 32 + lane];
                for (int bk = 0; bk < 2; ++bk) {
                    // chunk = record bits [16 K bk, 16 K (bk + 1)), up to 80 bits: read as two 64-bit pieces
                    auto bits = [&](int pos, int n) {   // n <= 64 bits at record bit pos
                        uint64_t v = 0;
                        for (int i = 0; i < n; ++i) { const int p = pos + i; v |= (uint64_t) ((r[p >> 5] >> (p & 31)) & 1u) << i; }
                        return v;
                    };
                    const uint64_t lo8 = bits(16 * K * bk, 8 * K);          // chunk bits [0, 8K)  -> steps u 8..15
                    const uint64_t hi8 = bits(16 * K * bk + 8 * K, 8 * K);  // chunk bits [8K, 16K) -> steps u 0..7
                    win[lane][bk][1] = lo8 | (hi8 & ((1ull << (15 - K)) - 1ull)) << (8 * K);   // u 8..15 end at bit 7K + 15 <= 16K - 1
                    win[lane][bk][0] = hi8;                                   // partner bits added below
                }
            }
            // partner bits: the low 15 - K bits of the chunk of lane ^ 16 (same bk)
            for (int lane = 0; lane < 32; ++lane)
                for (int bk = 0; bk < 2; ++bk) {
                    const uint64_t pc = win[lane ^ 16][bk][1] & ((1ull << (8 * K)) - 1ull);   // partner chunk bits [0, 8K) (its steps u 8..15)
                    win[lane][bk][0] |= (pc & ((1ull << (15 - K)) - 1ull)) << (8 * K);
                }
            for (int bk = 0; bk < 2; ++bk) {
                Float8 acc[32] = {};
                for (int m = 0; m < 8; ++m) {
                    Half8 a[32], bh[32], bl[32];
                    for (int lane = 0; lane < 32; ++lane) {
                        const int kg = lane / 16;
                        for (int s2 = 0; s2 < 2; ++s2) {
                            const int u = 2 * m + s2;   // local step 0..15
                            // u 0..7 live in window 0 at offset (7 - u) K, u 8..15 in window 1 at offset (15 - u) K
                            const uint32_t st = (uint32_t) ((u < 8 ? win[lane][bk][0] >> ((7 - u) * K) : win[lane][bk][1] >> ((15 - u) * K))) & 0x7fffu;
                            uint32_t w2[2];
                            step(st, w2);
                            a[lane].h[4 * s2 + 0] = (uint16_t) (w2[0] & 0xffffu); a[lane].h[4 * s2 + 1] = (uint16_t) (w2[0] >> 16);
                            a[lane].h[4 * s2 + 2] = (uint16_t) (w2[1] & 0xffffu); a[lane].h[4 * s2 + 3] = (uint16_t) (w2[1] >> 16);
                        }
                        const int n = lane % 16;
                        for (int j = 0; j < 8; ++j) {
                            float xv = 0.f;
                            if (n < nb) xv = x[(size_t) n * in + (2 * P + bk) * 128 + 4 * (16 * kg + 2 * m) + j];
                            const uint16_t hi = f2h_rn(xv);
                            bh[lane].h[j] = hi;
                            bl[lane].h[j] = f2h_rn(xv - tqcode::h2f(hi));
                        }
                    }
                    wmma_f32_16x16x16_f16(a, bh, acc);
                    if (exact) wmma_f32_16x16x16_f16(a, bl, acc);
                }
                for (int lane = 0; lane < 32; ++lane)
                    for (int i = 0; i < 8; ++i) {
                        const int row = 8 * (lane / 16) + i;
                        tot[lane][i] += tqcode::h2f(sc[(base * 16 + row) * 2 + bk]) * acc[lane].f[i];
                    }
            }
        }
        for (int lane = 0; lane < 32; ++lane) {
            const int n = lane % 16;
            if (n >= nb) continue;
            for (int i = 0; i < 8; ++i) y[(size_t) n * rows + t * 16 + 8 * (lane / 16) + i] = tot[lane][i];
        }
    }
}

}  // namespace tqw
