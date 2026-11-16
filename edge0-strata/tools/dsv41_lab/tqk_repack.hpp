// tools/dsv41_lab/tqk_repack.hpp - host reference of the "TQL" kernel layout of a trellis matrix (lossless bit permutation of
// the file layout; rows % 16 == 0, blocks per row even).  See tqk_gemv3.cuh for why.
//
// A tile = 16 rows, a pair P = blocks 2P, 2P+1.  Lane (g = lane/4, q = lane%4) of the mma tile owns, of each of the 4
// (block bk, row g + 8 h) stream chunks, the 8K bits at stream bit (24 - 8q) K  (steps 8q .. 8q+7 of the block), chunk
// i = 2 bk + h.  Its 4 chunks are 32 K bits = K words:  words[((tile * NP + P) * K + j) * 32 + lane].  The scales:
// sc[(tile * NP + P) * 8 + g] = 4 x u16, slot 2 bk + h = fp16 d of (row g + 8 h, block 2P + bk).
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>

namespace tqlab {

// file layout -> TQL: words (K * 32 per tile-pair) and scales (8 u64 per tile-pair)
inline void repack_tql(int K, const uint8_t* w, int rows, int nblk, std::vector<uint32_t>& words, std::vector<uint64_t>& sc) {
    const int bsz = 2 + 4 * K, NP = nblk / 2, NT = rows / 16;
    words.assign((size_t) NT * NP * K * 32, 0u);
    sc.assign((size_t) NT * NP * 8, 0ull);
    for (int t = 0; t < NT; ++t)
        for (int P = 0; P < NP; ++P) {
            const size_t base = (size_t) t * NP + P;
            for (int g = 0; g < 8; ++g) {
                uint64_t dq = 0;
                for (int c = 0; c < 4; ++c) {
                    const int bk = c >> 1, h = c & 1;
                    const uint8_t* blk = w + ((size_t) (t * 16 + g + 8 * h) * nblk + 2 * P + bk) * bsz;
                    uint16_t d;
                    std::memcpy(&d, blk, 2);
                    dq |= (uint64_t) d << (16 * c);
                }
                sc[base * 8 + g] = dq;
                for (int q = 0; q < 4; ++q) {
                    const int lane = 4 * g + q;
                    for (int i = 0; i < 4; ++i) {
                        const int bk = i >> 1, h = i & 1;
                        const uint8_t* blk = w + ((size_t) (t * 16 + g + 8 * h) * nblk + 2 * P + bk) * bsz;
                        for (int b = 0; b < 8 * K; ++b) {   // chunk bit b -> record bit 8 K i + b
                            const int src = (24 - 8 * q) * K + b;
                            const uint32_t bit = (blk[2 + (src >> 3)] >> (src & 7)) & 1u;
                            const int dst = 8 * K * i + b;
                            words[(base * K + (dst >> 5)) * 32 + lane] |= bit << (dst & 31);
                        }
                    }
                }
            }
        }
}

}  // namespace tqlab
