// Strata. Parity + speed of the gfx12 WMMA trellis GEMV (src/dsv41/ds4_tqw.cuh) on a R9700 (gfx1201):
//   tqw_hip_test [--k 4,5,6,7] [--code 0|1] [--bench]
// Parity: random legal streams (any bit pattern is one), host-decoded weights (tqenc::decode_blocks: the table codebook, or the
// computed code when TQE_CODE=3single, with --code 1) against the kernel's y, relative to sum|w x| per output.
// --bench: the DeepSeek decode shape (12 gate/up matrices of 2304 x 5120), GB/s of weight bytes.
#include "../../src/dsv41/ds4_tqw.cuh"
#include "../../tools/dsv41_lab/wmma_emu.hpp"
#include "../../tools/trellis/tqenc.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace strata::ds4::tqw;
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); std::exit(2); } } while (0)

template <typename T> static T* up(const std::vector<T>& v) { T* d; CK(hipMalloc(&d, v.size() * sizeof(T))); CK(hipMemcpy(d, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice)); return d; }

template <int K, int NB, int CODE> static void launch(const Job* jd, const uint32_t* lut, int rows, int njobs) {
    constexpr int T = 2;
    dim3 grid((rows + 16 * T * 4 - 1) / (16 * T * 4), njobs);
    gemv<K, NB, T, 1, CODE><<<grid, 128>>>(jd, lut);
}
template <int K, int CODE> static void launch_nb(int nb, const Job* jd, const uint32_t* lut, int rows, int njobs) {
    switch (nb) {
        case 1: launch<K, 1, CODE>(jd, lut, rows, njobs); break;
        case 3: launch<K, 3, CODE>(jd, lut, rows, njobs); break;
        default: launch<K, 5, CODE>(jd, lut, rows, njobs); break;
    }
}
template <int CODE> static void launch_k(int K, int nb, const Job* jd, const uint32_t* lut, int rows, int njobs) {
    switch (K) {
        case 4: launch_nb<4, CODE>(nb, jd, lut, rows, njobs); break;
        case 5: launch_nb<5, CODE>(nb, jd, lut, rows, njobs); break;
        case 6: launch_nb<6, CODE>(nb, jd, lut, rows, njobs); break;
        default: launch_nb<7, CODE>(nb, jd, lut, rows, njobs); break;
    }
}
static void run(int code, int K, int nb, const Job* jd, const uint32_t* lut, int rows, int njobs) {
    if (code) launch_k<1>(K, nb, jd, lut, rows, njobs); else launch_k<0>(K, nb, jd, lut, rows, njobs);
}

int main(int argc, char** argv) {
    int code = std::getenv("TQE_CODE") ? 1 : 0;
    bool bench = false;
    std::vector<int> Ks = {4, 5, 6, 7};
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bench") bench = true;
        else if (a == "--code" && i + 1 < argc) code = std::atoi(argv[++i]);
        else if (a == "--k" && i + 1 < argc) { Ks.clear(); for (const char* p = argv[++i]; *p;) { Ks.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; } }
    }
    std::vector<uint32_t> hl(2048);
    {
        const float* pt = tqenc::point_table();
        for (int i = 0; i < 2048; ++i) hl[i] = tqw::f2h_rn(pt[2 * i]) | ((uint32_t) tqw::f2h_rn(pt[2 * i + 1]) << 16);
    }
    const uint32_t* dlut = up(hl);
    std::mt19937_64 rng(7);
    int fails = 0;
    const int types[8] = {0, 0, 0, 0, 225, 224, 145, 146};
    for (int K : Ks) {
        const int type = types[K], bsz = 2 + 4 * K;
        const int rows = 96, in = 1536, nblk = in / 128;
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
            const uint32_t* dw = up(words); const uint16_t* ds = up(sc); const float* dx = up(x);
            half8_t* xb; CK(hipMalloc(&xb, (size_t) nblk * 8 * 2 * nb * 2 * sizeof(half8_t)));
            const size_t nall = (size_t) nblk * 8 * 2 * nb;
            prep_b<<<(unsigned) ((nall + 127) / 128), 128>>>(dx, in, nb, in, xb);
            float* dy; CK(hipMalloc(&dy, (size_t) nb * rows * 4)); CK(hipMemset(dy, 0, (size_t) nb * rows * 4));
            std::vector<Job> jv = {Job{dw, ds, xb, dy, rows, in, rows}};
            const Job* dj = up(jv);
            run(code, K, nb, dj, dlut, rows, 1);
            CK(hipDeviceSynchronize());
            std::vector<float> y((size_t) nb * rows);
            CK(hipMemcpy(y.data(), dy, y.size() * 4, hipMemcpyDeviceToHost));
            double worst = 0;
            for (int c = 0; c < nb; ++c)
                for (int r = 0; r < rows; ++r) {
                    double s = 0, sa = 0;
                    for (int i = 0; i < in; ++i) { const double t = (double) dec[(size_t) r * in + i] * x[(size_t) c * in + i]; s += t; sa += std::fabs(t); }
                    worst = std::max(worst, std::fabs(y[(size_t) c * rows + r] - s) / (sa + 1e-30));
                }
            const bool ok = worst < 3e-7;
            std::printf("K=%d nb=%d (%s): HIP wmma kernel vs double reference %.2e %s\n", K, nb, code ? "computed" : "table", worst, ok ? "ok" : "FAIL");
            fails += !ok;
        }
    }
    if (bench) {
        for (int K : Ks) {
            const int bsz = 2 + 4 * K, in = 5120, rows = 2304, nblk = in / 128;
            std::vector<uint8_t> w((size_t) rows * nblk * bsz);
            for (auto& b : w) b = (uint8_t) rng();
            std::vector<uint32_t> words; std::vector<uint16_t> sc;
            tqw::repack_tqw(K, w.data(), rows, nblk, words, sc);
            for (int nb : {1, 5}) {
                std::vector<float> x((size_t) nb * in, 0.5f);
                const float* dx = up(x);
                half8_t* xb; CK(hipMalloc(&xb, (size_t) nblk * 8 * 2 * nb * 2 * sizeof(half8_t)));
                const size_t nall = (size_t) nblk * 8 * 2 * nb;
                prep_b<<<(unsigned) ((nall + 127) / 128), 128>>>(dx, in, nb, in, xb);
                float* dy; CK(hipMalloc(&dy, (size_t) nb * rows * 4));
                std::vector<Job> jv;   // 12 distinct buffers, so the L2 cannot serve a repeated matrix
                for (int j = 0; j < 12; ++j) jv.push_back(Job{up(words), up(sc), xb, dy, rows, in, rows});
                const Job* dj = up(jv);
                const double bytes = 12.0 * rows * nblk * bsz;
                for (int rep = 0; rep < 3; ++rep) run(code, K, nb, dj, dlut, rows, 12);
                CK(hipDeviceSynchronize());
                const auto t0 = std::chrono::steady_clock::now();
                const int reps = 50;
                for (int rep = 0; rep < reps; ++rep) run(code, K, nb, dj, dlut, rows, 12);
                CK(hipDeviceSynchronize());
                const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
                std::printf("bench K=%d nb=%d %s: 12 gate/up matrices %.1f us, %.0f GB/s, %.2f Tw/s\n", K, nb, code ? "computed" : "table", us, bytes / us / 1e3,
                            12.0 * rows * in / us / 1e6);
            }
        }
    }
    std::printf("%s\n", fails ? "FAILED" : "ALL OK");
    return fails;
}
