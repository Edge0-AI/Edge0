// src/kernels/cpu/pool_stress.cpp - issue #29: the expert pool under the load a big-VRAM card gives it.
//
// With most experts resident on the GPU, many layers hand the pool nothing, the workers go to sleep in the middle
// of a request, and the next tiny batch (one or two experts) is often finished by the host and the awake workers
// before a sleeper has even woken.  That late wake-up is what the park/claim protocol has to survive.  This drives
// hundreds of thousands of such batches - 0-4 jobs, run() and run_split(), with pauses long enough for the workers
// to sleep - and checks that every batch returns, with every job's output written.  A watchdog thread turns a
// hang into a failure instead of a frozen test.
//
//   pool_stress [seconds]        (default 20)
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

namespace cpu = strata::kernels::cpu;

int main(int argc, char** argv) {
    const double seconds = argc > 1 ? std::atof(argv[1]) : 20.0;
    const cpu::CpuFeatures feat = cpu::cpu_features();
    if (!feat.usable()) {
        std::printf("CPU lacks %s: pool_stress SKIPPED\n", feat.reason());
        return 0;
    }
    constexpr int NJ = 4;
    std::vector<std::vector<uint8_t>> blobs(NJ, std::vector<uint8_t>(cpu::BLOB, 0));
    std::vector<float> x(cpu::H, 0.25f);
    cpu::ActQ act;
    cpu::act_quant_q8_1(x.data(), cpu::H, act);
    std::vector<float> out((size_t) NJ * cpu::H);
    std::vector<cpu::ExpertJob> jobs(NJ);
    for (int e = 0; e < NJ; ++e) {
        jobs[(size_t) e].blob = blobs[(size_t) e].data();
        jobs[(size_t) e].act = &act;
        jobs[(size_t) e].out = out.data() + (size_t) e * cpu::H;
        jobs[(size_t) e].slot = e;
    }

    cpu::ExpertPool pool;
    std::printf("pool_stress: %d workers + host, %.0f s\n", pool.workers(), seconds);

    // the watchdog: a batch that does not return within 10 s is a hang
    std::atomic<long long> beat{0};
    std::atomic<bool> finished{false};
    std::thread dog([&] {
        long long last = -1;
        auto since = std::chrono::steady_clock::now();
        while (!finished.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            const long long b = beat.load();
            const auto now = std::chrono::steady_clock::now();
            if (b != last) { last = b; since = now; continue; }
            if (now - since > std::chrono::seconds(10)) {
                std::printf("\npool_stress: HANG - batch %lld has not returned for 10 s\n", b);
                std::fflush(stdout);
                std::_Exit(3);
            }
        }
    });

    std::mt19937 rng(29);
    const float SENTINEL = -1.2345e33f;
    long long batches = 0, sleeps = 0, missed = 0;
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - t0 < std::chrono::duration<double>(seconds)) {
        // now and then a gap long enough for every worker to go to sleep (the pool sleeps after 20 ms)
        if (rng() % 64 == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(21 + (int) (rng() % 10)));
            ++sleeps;
        } else if (rng() % 4 == 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(rng() % 300));
        }
        const int n = (int) (rng() % (NJ + 1));
        for (int e = 0; e < n; ++e) std::fill(jobs[(size_t) e].out, jobs[(size_t) e].out + cpu::H, SENTINEL);
        if (rng() % 2) pool.run(jobs.data(), n);
        else pool.run_split(jobs.data(), n);
        for (int e = 0; e < n; ++e)
            if (jobs[(size_t) e].out[0] == SENTINEL || jobs[(size_t) e].out[cpu::H - 1] == SENTINEL) ++missed;
        beat.fetch_add(1);
        ++batches;
    }
    finished.store(true);
    dog.join();
    std::printf("pool_stress: %lld batches (%lld after the workers slept), %lld jobs not run\n", batches, sleeps,
                missed);
    if (missed) {
        std::printf("pool_stress: FAILED\n");
        return 1;
    }
    std::printf("pool_stress OK\n");
    return 0;
}
