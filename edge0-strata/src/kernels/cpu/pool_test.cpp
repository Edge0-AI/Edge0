// src/kernels/cpu/pool_test.cpp - P2.S3's test for the expert pool.
//
// The pool's correctness claims are small and specific, so they are checked directly rather than through a
// timing number:
//
//   1. THE SAME ANSWER AS SERIAL.  Every job's output must be bit-identical to running that expert serially.
//      Each worker owns its own `ExpertScratch`, and the jobs share one read-only activation, so there is no
//      legitimate source of difference - "close enough" here would be hiding a data race.
//   2. EVERY JOB RUNS EXACTLY ONCE.  Claiming is `head.fetch_add`, and an off-by-one in the bound is the
//      classic pool bug: it either drops a job or runs one twice.  Checked with a sentinel-filled output
//      buffer, so a dropped job is visible as an untouched slot rather than as a slightly wrong number.
//   3. REPEATED BATCHES.  A pool that works once and hangs or corrupts on the second `run()` is the failure
//      mode the park protocol exists to prevent, so `run()` is called many times in a row.
//   4. A BATCH BIGGER AND SMALLER THAN THE WORKER COUNT, because `n < workers` leaves most workers claiming
//      nothing and `n > workers` is the real case (10 experts, 5 workers).
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {

bool read_blob(const char* path, long long index, std::vector<uint8_t>& out) {
    out.assign(cpu::BLOB, 0);
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return false;
#if defined(_MSC_VER)
    if (_fseeki64(f, index * (long long) cpu::BLOB, SEEK_SET) != 0) { std::fclose(f); return false; }
#else
    if (fseeko(f, (off_t) index * (off_t) cpu::BLOB, SEEK_SET) != 0) { std::fclose(f); return false; }
#endif
    const size_t got = std::fread(out.data(), 1, out.size(), f);
    std::fclose(f);
    return got == out.size();
}

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    const char* path = "pack/full/experts.bin";
    long long layer = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") selftest = true;
        else if (a == "--file" && i + 1 < argc) path = argv[++i];
        else if (a == "--layer" && i + 1 < argc) layer = std::atoll(argv[++i]);
        else { std::fprintf(stderr, "usage: pool_test [--selftest] [--file P] [--layer N]\n"); return 2; }
    }

    const cpu::CpuFeatures feat = cpu::cpu_features();
    if (!feat.usable()) {
        std::printf("  CPU lacks %s - the VNNI path cannot run here; pool test SKIPPED, not passed.\n",
                    feat.reason());
        std::printf("\npool: 0 failures, 1 SKIPPED\n");
        return 0;
    }

    int bad = 0;

    // ---- ten experts off one layer, which is exactly what a token uses
    const int NEXP = 10;
    std::vector<std::vector<uint8_t>> blobs((size_t) NEXP);
    for (int e = 0; e < NEXP; ++e)
        if (!read_blob(path, layer * 512 + e, blobs[(size_t) e])) {
            std::fprintf(stderr, "cannot read expert %d of layer %lld from %s\n", e, layer, path);
            return 2;
        }

    std::mt19937 rng(99);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> x(cpu::H);
    for (auto& v : x) v = gauss(rng);

    cpu::ActQ act;
    cpu::act_quant_q8_1(x.data(), cpu::H, act);

    // ---- serial reference
    std::vector<float> ref((size_t) NEXP * cpu::H);
    {
        cpu::ExpertScratch ws;
        for (int e = 0; e < NEXP; ++e)
            cpu::s2_expert_vnni_q(blobs[(size_t) e].data(), act, ref.data() + (size_t) e * cpu::H, ws);
    }

    // ---- the pool
    const std::vector<int> cores = cpu::physical_cores(true);
    const int hw = (int) std::thread::hardware_concurrency();
    std::printf("  %-44s %d logical, %d physical (skipping the first)\n", "cores the pool will use",
                hw, (int) cores.size());

    cpu::ExpertPool pool;
    std::printf("  %-44s %d\n", "workers", pool.workers());

    std::vector<cpu::ExpertJob> jobs((size_t) NEXP);
    std::vector<float> got((size_t) NEXP * cpu::H);
    std::vector<float> weights((size_t) NEXP);
    for (int e = 0; e < NEXP; ++e) weights[(size_t) e] = 0.1f * (float) (e + 1);

    // ---- 1 + 2: same answer as serial, and every job exactly once.
    // The output buffer starts at a sentinel no real result can equal, so a job that never runs shows up as
    // an untouched slot instead of as a plausible number.
    const float SENTINEL = -1.2345e33f;
    for (int e = 0; e < NEXP; ++e) {
        jobs[(size_t) e].blob = blobs[(size_t) e].data();
        jobs[(size_t) e].act = &act;
        jobs[(size_t) e].out = got.data() + (size_t) e * cpu::H;
        jobs[(size_t) e].weight = weights[(size_t) e];
        jobs[(size_t) e].slot = e;
        std::fill(jobs[(size_t) e].out, jobs[(size_t) e].out + cpu::H, SENTINEL);
    }
    pool.run(jobs.data(), NEXP);

    long long not_run = 0, diff = 0;
    float worst = 0.f;
    for (int e = 0; e < NEXP; ++e)
        for (int i = 0; i < cpu::H; ++i) {
            const float a = ref[(size_t) e * cpu::H + i], b = got[(size_t) e * cpu::H + i];
            if (b == SENTINEL) { ++not_run; continue; }
            if (std::memcmp(&a, &b, 4) != 0) ++diff;
            worst = std::fmax(worst, std::fabs(a - b));
        }
    std::printf("  %-44s %s (%lld of %d outputs untouched)\n", "every job ran exactly once",
                not_run ? "*** NO ***" : "yes", not_run, NEXP * cpu::H);
    if (not_run) ++bad;
    // BIT-IDENTICAL, not "close".  Each worker has a private scratch and the activation is shared read-only,
    // so any difference at all is a race or a scratch collision - and a tolerance would hide exactly that.
    std::printf("  %-44s %s (%lld of %d differ, worst |d| %.3e)\n", "identical to the serial run",
                diff ? "*** NO ***" : "yes", diff, NEXP * cpu::H, (double) worst);
    if (diff) ++bad;

    // ---- 3: repeated batches.  A park protocol that works once and corrupts on the second call is the
    //        exact failure this loop is here to catch.
    {
        int repeats_bad = 0;
        for (int r = 0; r < 200; ++r) {
            for (int e = 0; e < NEXP; ++e) std::fill(jobs[(size_t) e].out, jobs[(size_t) e].out + cpu::H, SENTINEL);
            pool.run(jobs.data(), NEXP);
            for (int e = 0; e < NEXP && !repeats_bad; ++e)
                for (int i = 0; i < cpu::H; ++i)
                    if (std::memcmp(&ref[(size_t) e * cpu::H + i], &got[(size_t) e * cpu::H + i], 4) != 0) {
                        ++repeats_bad;
                        break;
                    }
        }
        std::printf("  %-44s %s (200 consecutive batches)\n", "repeated batches stay correct",
                    repeats_bad ? "*** NO ***" : "yes");
        if (repeats_bad) ++bad;
    }

    // ---- 4: batch sizes either side of the worker count
    {
        int size_bad = 0;
        for (int n : {1, 2, pool.workers() - 1 > 0 ? pool.workers() - 1 : 1, pool.workers(),
                      pool.workers() + 1, NEXP}) {
            if (n < 1 || n > NEXP) continue;
            for (int e = 0; e < n; ++e) std::fill(jobs[(size_t) e].out, jobs[(size_t) e].out + cpu::H, SENTINEL);
            pool.run(jobs.data(), n);
            for (int e = 0; e < n && !size_bad; ++e)
                for (int i = 0; i < cpu::H; ++i)
                    if (std::memcmp(&ref[(size_t) e * cpu::H + i], &got[(size_t) e * cpu::H + i], 4) != 0) {
                        ++size_bad;
                        break;
                    }
        }
        std::printf("  %-44s %s (1, w-1, w, w+1, 10)\n", "batch sizes around the worker count",
                    size_bad ? "*** NO ***" : "yes");
        if (size_bad) ++bad;
    }

    // ---- and the number that matters for the ledger: c for one layer, 10 experts over 48 layers.
    //
    // BEST OF N, not a mean.  The first version divided a 10-run total by 10 and reported 6.604 ms per layer -
    // 2.1 GB/s against L9's measured 42.55 - purely because a CUDA build was running on the same 6-core
    // machine at the time.  A mean over a contended machine measures the contention; a best-of measures the
    // instrument, and every other timing tool in this project already takes the best.  The spread is printed
    // so a contended run is visible rather than being read as a regression.
    {
        const int REPS = 20;
        std::vector<double> t((size_t) REPS, 0.0);
        for (int r = 0; r < REPS; ++r) {
            const double t0 = now_ms();
            pool.run(jobs.data(), NEXP);
            t[(size_t) r] = now_ms() - t0;
        }
        std::vector<double> sorted = t;
        std::sort(sorted.begin(), sorted.end());
        const double best = sorted.front(), median = sorted[sorted.size() / 2], worst = sorted.back();
        const double gbs = (double) NEXP * cpu::BLOB / (best * 1e-3) / 1e9;
        std::printf("\n  %-44s %7.3f ms   (%.1f GB/s)\n", "10 experts, one layer (best of 20)",
                    best, gbs);
        std::printf("  %-44s %7.3f / %7.3f ms   (spread %.2fx)\n", "median / worst", median, worst,
                    median / (best > 0 ? best : 1));
        std::printf("  %-44s %7.3f ms   -> %.1f tok/s for the full 48 layers\n",
                    "extrapolated to 48 layers", best * 48, 1000.0 / (best * 48));
        std::printf("  L9 measured 42.55 GB/s on 6 cores; this pool uses %d workers (core 0 is left to the\n",
                    pool.workers());
        std::printf("  host loop), so %.1f GB/s x 6/%d = %.1f GB/s is the per-core comparison.\n",
                    gbs, pool.workers(), gbs * 6.0 / pool.workers());
        if (median / (best > 0 ? best : 1) > 1.5)
            std::printf("  *** the spread is over 1.5x: this machine was CONTENDED and the median is not a\n"
                        "      property of the pool.  Re-run on a quiet machine before quoting it. ***\n");
    }

    std::printf("\npool: %d failures\n", bad);
    if (bad) return 1;
    if (selftest) std::printf("pool_test OK\n");
    return 0;
}
