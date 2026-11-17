// Linux/x86 CUDA reproducer for #1975. Run "lazy" and "warm" as separate processes.
// See bench/results/2026-10-11-rtx-5090-callback-affinity/README.md for the build command.
#include <cuda_runtime_api.h>
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct State {
    std::atomic<bool> done{false};
    cpu_set_t mask{};
};

static void callback(void* p) {
    auto* s = (State*) p;
    sched_getaffinity(0, sizeof s->mask, &s->mask);
    s->done.store(true, std::memory_order_release);
}

static void check(cudaError_t e) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    const bool warm = argc > 1 && std::strcmp(argv[1], "warm") == 0;
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    State state;
    if (warm) {
        check(cudaLaunchHostFunc(stream, callback, &state));
        check(cudaStreamSynchronize(stream));
    }
    cpu_set_t host;
    CPU_ZERO(&host);
    CPU_SET(0, &host);
    if (sched_setaffinity(0, sizeof host, &host) != 0) {
        std::perror("sched_setaffinity");
        return 1;
    }
    std::vector<double> times;
    for (int i = 0; i < 200; ++i) {
        state.done.store(false, std::memory_order_release);
        const auto start = std::chrono::steady_clock::now();
        check(cudaLaunchHostFunc(stream, callback, &state));
        while (!state.done.load(std::memory_order_acquire)) {
            asm volatile("pause");
            if (std::chrono::steady_clock::now() - start > std::chrono::seconds(5)) return 2;
        }
        times.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count());
    }
    std::sort(times.begin(), times.end());
    std::printf("%s: callback affinity:", warm ? "before pin warm-up" : "after pin lazy creation");
    for (int i = 0; i < CPU_SETSIZE; ++i)
        if (CPU_ISSET(i, &state.mask)) std::printf(" %d", i);
    std::printf("; median %.2f us, p95 %.2f us\n", times[100], times[190]);
    check(cudaStreamSynchronize(stream));
    check(cudaStreamDestroy(stream));
}
