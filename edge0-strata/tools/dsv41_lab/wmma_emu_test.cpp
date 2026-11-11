// tools/dsv41_lab/wmma_emu_test.cpp - the gfx12 port's data flow (wmma_emu.hpp) against the double reference on decoded weights.
//   ./wmma_emu_test            table codebook (shipped K5/K6/K7)
//   TQE_CODE=3single ./wmma_emu_test    computed code
// Host only; no GPU needed.
#include "../trellis/tqenc.hpp"
#include "wmma_emu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static void step_table(uint32_t s, uint32_t out[2]) {
    static std::vector<uint32_t> lut;
    if (lut.empty()) {
        const float* pt = tqenc::point_table();
        lut.resize(2048);
        for (int i = 0; i < 2048; ++i) lut[i] = tqw::f2h_rn(pt[2 * i]) | ((uint32_t) tqw::f2h_rn(pt[2 * i + 1]) << 16);
    }
    const uint32_t x = (s & 0x7fffu) * 0x9e3779b1u;
    out[0] = lut[x >> 21];
    out[1] = lut[(x >> 10) & 2047u];
}
static void step_3single(uint32_t s, uint32_t out[2]) { out[0] = tqcode::word3(s & 0x7fffu, 0); out[1] = tqcode::word3(s & 0x7fffu, 1); }

int main() {
    const bool computed = tqcode::kind_of(std::getenv("TQE_CODE")) == tqcode::k3single;
    if (std::getenv("TQE_CODE") && !computed) { std::printf("only TQE_CODE=3single is wired here\n"); return 2; }
    std::mt19937_64 rng(7);
    int fails = 0;
    const int Ks[4] = {4, 5, 6, 7}, types[4] = {225, 224, 145, 146};
    for (int ki = 0; ki < 4; ++ki) {
        const int K = Ks[ki], type = types[ki], bsz = 2 + 4 * K;
        const int rows = 48, in = 768, nblk = in / 128;
        std::vector<uint8_t> w((size_t) rows * nblk * bsz);
        for (auto& b : w) b = (uint8_t) rng();
        for (int64_t b = 0; b < (int64_t) rows * nblk; ++b) { const uint16_t d = tqw::f2h_rn(0.02f + 0.001f * (float) (rng() % 400)); std::memcpy(&w[(size_t) b * bsz], &d, 2); }
        std::vector<float> dec((size_t) rows * in);
        tqenc::decode_blocks(type, w.data(), (int64_t) rows * nblk, dec.data());
        std::vector<uint32_t> words; std::vector<uint16_t> sc;
        tqw::repack_tqw(K, w.data(), rows, nblk, words, sc);
        for (int nb : {1, 3, 5}) {
            std::vector<float> x((size_t) nb * in);
            std::uniform_real_distribution<float> u(-1.f, 1.f);
            for (auto& v : x) v = u(rng);
            std::vector<float> y;
            tqw::gemv_tqw(K, nb, rows, in, words, sc, x, computed ? step_3single : step_table, y);
            double worst = 0;
            for (int c = 0; c < nb; ++c)
                for (int r = 0; r < rows; ++r) {
                    double s = 0, sa = 0;
                    for (int i = 0; i < in; ++i) { const double t = (double) dec[(size_t) r * in + i] * x[(size_t) c * in + i]; s += t; sa += std::fabs(t); }
                    worst = std::max(worst, std::fabs(y[(size_t) c * rows + r] - s) / (sa + 1e-30));
                }
            const bool ok = worst < 3e-7;
            std::printf("K=%d nb=%d (%s): emulated gfx12 flow vs double reference %.2e %s\n", K, nb, computed ? "3single" : "table", worst, ok ? "ok" : "FAIL");
            fails += !ok;
        }
    }
    std::printf("%s\n", fails ? "FAILED" : "ALL OK");
    return fails;
}
