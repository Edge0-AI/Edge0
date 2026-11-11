// tools/trellis/tqenc_hip_test.cpp - GPU encoder vs the CPU encoder: correctness (GPU bytes decode, by the CPU decoder,
// to exactly the weights whose error the GPU reported), quality (SQNR, planted recovery) and speed (weights/s).
// Run under the box locks:  flock -s /data/gpu.lock flock -x ~/locks/gpu3.lock env HIP_VISIBLE_DEVICES=3 ./tqenc_hip_test
#include "tqenc.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

extern "C" double tqh_encode(int type, const float* w, const float* h, int64_t nblocks, uint8_t* dst, int iters, int ncand,
                             int scale_tries, int device, double* err_out, int64_t* stats, double* timing, int fixed_scale_bits);

using namespace tqenc;
static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { ++g_fail; std::printf("  FAIL line %d: %s  ", __LINE__, #c); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static const char* tname(int t) { return t == TQK6 ? "TQK6" : t == TQK7 ? "TQK7" : t == TQK5T ? "TQK5T" : t == TQK4T ? "TQK4T" : t == TQK5X ? "TQK5X" : "TQK4X"; }

int main(int argc, char** argv) {
    const long big = argc > 1 ? std::atol(argv[1]) : 131072;   // blocks for the speed runs
    std::mt19937_64 rng(77);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int types[6] = {TQK4T, TQK5T, TQK4X, TQK5X, TQK6, TQK7};
    for (int type : types) {
        const int bb = block_bytes(type);
        std::printf("== %s\n", tname(type));
        // 1. correctness + CPU comparison on 2048 Gaussian blocks, without and with importance weights
        {
            const int nb = 2048;
            std::vector<float> w((size_t) nb * 128), h((size_t) nb * 128);
            for (auto& v : w) v = 0.02f * nd(rng);
            for (auto& v : h) v = 0.2f + std::fabs(nd(rng));
            for (int wi = 0; wi < 2; ++wi) {
                const float* hp = wi ? h.data() : nullptr;
                std::vector<uint8_t> g((size_t) nb * bb), c((size_t) nb * bb);
                std::vector<double> ge(nb), ce(nb);
                int64_t st[2];
                double tm[2];
                const double gt = tqh_encode(type, w.data(), hp, nb, g.data(), 1, 1, 1, 0, ge.data(), st, tm, 0);
                Params p;
                p.iters = 1;
                int64_t nf, zb;
                const double ct = encode_blocks(type, w.data(), hp, nb, c.data(), p, 8, ce.data(), &nf, &zb);
                std::vector<float> dec((size_t) nb * 128);
                decode_blocks(type, g.data(), nb, dec.data());
                double worst = 0;
                for (int b = 0; b < nb; ++b) {
                    double e2 = 0;
                    for (int i = 0; i < 128; ++i) {
                        const double d = (double) w[b * 128 + i] - dec[b * 128 + i];
                        e2 += (hp ? hp[b * 128 + i] : 1.0) * d * d;
                    }
                    worst = std::max(worst, std::fabs(e2 - ge[b]) / std::max(1e-30, e2 + ge[b]));
                }
                CHECK(worst < 1e-5, "GPU reported error differs from the decoded error: %.2e", worst);
                int same = 0;
                for (int b = 0; b < nb; ++b) same += std::memcmp(&g[(size_t) b * bb], &c[(size_t) b * bb], bb) == 0;
                std::printf("  importance %-3s GPU total error %.6e, CPU %.6e (GPU/CPU %.5f); identical blocks %d/%d; decoded-vs-reported worst %.1e\n",
                            wi ? "yes" : "no", gt, ct, gt / ct, same, nb, worst);
                CHECK(gt <= ct * 1.002, "GPU error is more than 0.2%% above the CPU's");
            }
        }
        // 2. planted solutions: weights decoded from random streams must be recovered (the scale is found by the refinement)
        {
            const int nb = 2048;
            std::vector<uint8_t> blk((size_t) nb * bb);
            for (int b = 0; b < nb; ++b) {
                const float d = std::ldexp(0.5f + (float) (rng() % 1000) / 1000.f, -8 + (int) (rng() % 6));
                const uint16_t dh = float_to_half(d);
                blk[(size_t) b * bb] = dh & 255;
                blk[(size_t) b * bb + 1] = dh >> 8;
                for (int i = 2; i < bb; ++i) blk[(size_t) b * bb + i] = (uint8_t) rng();
            }
            std::vector<float> w((size_t) nb * 128);
            decode_blocks(type, blk.data(), nb, w.data());
            std::vector<uint8_t> g((size_t) nb * bb);
            std::vector<double> ge(nb);
            tqh_encode(type, w.data(), nullptr, nb, g.data(), 3, 1, 1, 0, ge.data(), nullptr, nullptr, 0);
            int ok = 0;
            for (int b = 0; b < nb; ++b) ok += ge[b] == 0.0;
            std::printf("  planted: %d/%d blocks recovered exactly (scale unknown)\n", ok, nb);
            CHECK(ok > nb * 0.9, "planted recovery low");
        }
        // 3. speed on big Gaussian batches
        {
            std::vector<float> w((size_t) big * 128);
            for (auto& v : w) v = nd(rng);
            std::vector<uint8_t> g((size_t) big * bb);
            double en = 0;
            for (float v : w) en += (double) v * v;
            struct Cfg { const char* n; int iters, ncand, tries; } cfgs[3] = {{"fast (1 pass + 1)", 0, 1, 1}, {"default (iters 1)", 1, 1, 1}, {"rot3 iters3", 3, 3, 1}};
            for (auto& c : cfgs) {
                double tm[2];
                int64_t stt[2];
                const double t0 = now();
                const double e = tqh_encode(type, w.data(), nullptr, big, g.data(), c.iters, c.ncand, c.tries, 0, nullptr, stt, tm, 0);
                const double s = now() - t0;
                {
                    std::vector<float> dec((size_t) big * 128);
                    decode_blocks(type, g.data(), big, dec.data());
                    double e2 = 0;
                    for (size_t i = 0; i < dec.size(); ++i) e2 += ((double) w[i] - dec[i]) * ((double) w[i] - dec[i]);
                    std::printf("  [check: energy %.6e, reported error %.6e, decoded error %.6e -> SQNR %.3f dB]\n", en, e, e2, 10 * std::log10(en / e2));
                }
                std::printf("  %-18s SQNR %.3f dB  %.2f M weights/s (kernels only %.2f M/s, %.0f passes)  total %.2fs  [zero blocks %lld, nonfinite %lld]\n", c.n,
                            10 * std::log10(en / e), (double) big * 128 / s / 1e6, (double) big * 128 / tm[0] / 1e6, tm[1], s,
                            (long long) stt[1], (long long) stt[0]);
            }
        }
    }
    std::printf("\n%s (%d failures)\n", g_fail ? "FAILED" : "OK", g_fail);
    return g_fail ? 1 : 0;
}
