// tools/dsv41_lab/code_sqnr.cpp - weight SQNR of the trellis encoder on N(0,1) blocks for a codebook chosen by TQE_CODE
// (unset = the shipped table).   TQE_CODE=3sum ./code_sqnr [type 224] [blocks 16384] [thorough 0|1]
// Rate-distortion limit of a Gaussian at K bits per 4 weights: 6.02 K / 4 dB.
#include "../trellis/tqenc.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    const int type = argc > 1 ? std::atoi(argv[1]) : 224;
    const int64_t nb = argc > 2 ? std::atoll(argv[2]) : 16384;
    const bool thorough = argc > 3 && std::atoi(argv[3]);
    const int K = tqenc::type_kbits(type);
    std::mt19937_64 rng(12345);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> w((size_t) nb * 128);
    for (auto& v : w) v = nd(rng);
    std::vector<uint8_t> enc((size_t) nb * tqenc::block_bytes(type));
    tqenc::Params p;
    if (thorough) { p.iters = 3; p.ncand = 3; p.scale_tries = 3; }
    int64_t nonfinite = 0, zero = 0;
    tqenc::encode_blocks(type, w.data(), nullptr, nb, enc.data(), p, 0, nullptr, &nonfinite, &zero);
    std::vector<float> dec((size_t) nb * 128);
    tqenc::decode_blocks(type, enc.data(), nb, dec.data());
    double sq = 0, ee = 0;
    for (size_t i = 0; i < w.size(); ++i) { sq += (double) w[i] * w[i]; ee += (double) (dec[i] - w[i]) * (dec[i] - w[i]); }
    const double sqnr = 10 * std::log10(sq / ee), lim = 6.0206 * K / 4.0;
    std::printf("TQE_CODE=%-8s type %d K=%d %s: SQNR %.3f dB, limit %.3f dB, gap %.3f dB (rel MSE %.4f)\n", std::getenv("TQE_CODE") ? std::getenv("TQE_CODE") : "table", type, K,
                thorough ? "thorough" : "fast", sqnr, lim, lim - sqnr, ee / sq);
    return 0;
}
