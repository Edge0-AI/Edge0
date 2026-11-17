// src/kernels/ple_fp8_parity.cpp - the FP8 PLE table (tools/ple_fp8_pack.py) read through PleTable, against values
// decoded from the checkpoint itself, and against the IQ4_NL table it replaces.
//
//     ple_fp8_parity <ple-fp8.gguf> <reference.bin> [<iq4nl shard.gguf>]
//
// reference.bin: int32 n, uint32 rows[n], float32 values[n][160] - the rows as torch decodes the checkpoint's
// F8_E4M3 bytes, times its weight_scale (n a multiple of 16 for the gather_batch check). Every row must match BIT FOR BIT through
// both I/O modes (Direct and Mmap) and through the prompt path's gather_batch. With the IQ4_NL shard the same rows
// are compared too: same table, so a correlation near 0.997 - a shifted row order would be ~0.
#include "strata/kernels/ngram.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace k = strata::kernels;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: ple_fp8_parity <ple-fp8.gguf> <reference.bin> [<iq4nl shard.gguf>]\n");
        return 2;
    }
    std::ifstream f(argv[2], std::ios::binary);
    int32_t n = 0;
    f.read((char*) &n, 4);
    std::vector<uint32_t> rows((size_t) n);
    std::vector<float> ref((size_t) n * k::PLE_HEAD_DIM);
    f.read((char*) rows.data(), (std::streamsize) rows.size() * 4);
    f.read((char*) ref.data(), (std::streamsize) ref.size() * 4);
    if (!f || n <= 0) { std::fprintf(stderr, "cannot read %s\n", argv[2]); return 2; }

    int bad = 0;
    for (const k::PleIo mode : {k::PleIo::Direct, k::PleIo::Mmap}) {
        k::PleTable t;
        std::string err;
        k::PleIoOptions io;
        io.mode = mode;
        if (!t.open(argv[1], err, io)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
        std::vector<float> got((size_t) n * k::PLE_HEAD_DIM);
        for (int i = 0; i < n; ++i) t.read_row(rows[(size_t) i], got.data() + (size_t) i * k::PLE_HEAD_DIM);
        int diff = 0;
        for (size_t j = 0; j < got.size(); ++j) diff += std::memcmp(&got[j], &ref[j], 4) != 0;
        std::vector<float> batch((size_t) n * k::PLE_HEAD_DIM);   // the prompt path: all rows as one request
        const bool ok = t.gather_batch(rows.data(), (size_t) n / k::PLE_N_HEADS, batch.data(), err);
        int bdiff = 0;
        const size_t nb = (size_t) n / k::PLE_N_HEADS * k::PLE_N_HEADS * k::PLE_HEAD_DIM;
        for (size_t j = 0; ok && j < nb; ++j) bdiff += std::memcmp(&batch[j], &ref[j], 4) != 0;
        std::printf("%-6s %s, %llu rows: read_row %d/%zu values differ, gather_batch %s %d differ\n",
                    mode == k::PleIo::Direct ? "Direct" : "Mmap", t.format(), (unsigned long long) t.rows(), diff,
                    got.size(), ok ? "ok," : "FAILED,", bdiff);
        bad += diff + bdiff + (ok ? 0 : 1) + (std::strcmp(t.format(), std::getenv("PLE_EXPECT_FORMAT") ? std::getenv("PLE_EXPECT_FORMAT") : "F8_E4M3") != 0);
    }
    if (argc > 3) {
        k::PleTable q;
        std::string err;
        if (!q.open(argv[3], err)) { std::fprintf(stderr, "open %s: %s\n", argv[3], err.c_str()); return 1; }
        double cmin = 1, rsum = 0;
        int counted = 0;
        std::vector<float> v(k::PLE_HEAD_DIM);
        for (int i = 0; i < n; ++i) {
            q.read_row(rows[(size_t) i], v.data());
            const float* r = ref.data() + (size_t) i * k::PLE_HEAD_DIM;
            double sa = 0, sb = 0, sab = 0, sa2 = 0, sb2 = 0, e2 = 0, r2 = 0;
            for (int j = 0; j < k::PLE_HEAD_DIM; ++j) {
                sa += r[j]; sb += v[j]; sab += (double) r[j] * v[j]; sa2 += (double) r[j] * r[j]; sb2 += (double) v[j] * v[j];
                e2 += ((double) v[j] - r[j]) * ((double) v[j] - r[j]); r2 += (double) r[j] * r[j];
            }
            if (r2 == 0) continue;                                      // an all-zero row (the table has some)
            const double m = k::PLE_HEAD_DIM;
            const double c = (sab - sa * sb / m) / std::sqrt((sa2 - sa * sa / m) * (sb2 - sb * sb / m));
            cmin = std::fmin(cmin, c);
            rsum += std::sqrt(e2 / r2);
            ++counted;
        }
        std::printf("IQ4_NL table, same rows: correlation min %.4f, error vs FP8 %.1f%% mean (%d non-zero rows)\n",
                    cmin, 100 * rsum / (counted ? counted : 1), counted);
        if (cmin < 0.98) ++bad;
    }
    std::printf("RESULT: %s\n", bad ? "MISMATCH" : "FP8 rows exact");
    return bad ? 1 : 0;
}
