#include "strata/core/host_graph_launch.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <thread>

static void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}
static void gpu(cudaError_t e) { check(e == cudaSuccess, cudaGetErrorString(e)); }
__global__ void wait_for_cpu(volatile unsigned* flag) {
    while (*flag == 0) { }
}
__global__ void tick(unsigned* value) { ++*value; }

int main(int argc, char** argv) {
    const bool threaded = !(argc > 1 && std::strcmp(argv[1], "--synchronous") == 0);
    cudaStream_t stream;
    gpu(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    unsigned *host, *flag, *value;
    gpu(cudaHostAlloc(&host, sizeof(unsigned), cudaHostAllocMapped));
    gpu(cudaHostGetDevicePointer(&flag, host, 0));
    gpu(cudaMalloc(&value, sizeof(unsigned)));
    gpu(cudaMemset(value, 0, sizeof(unsigned)));
    *host = 0;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    gpu(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    wait_for_cpu<<<1, 1, 0, stream>>>(flag);
    // Enough nodes to exercise launch backpressure under Compute Sanitizer.
    for (int i = 0; i < 4096; ++i) tick<<<1, 1, 0, stream>>>(value);
    gpu(cudaStreamEndCapture(stream, &graph));
    gpu(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    std::atomic<bool> done{false}, timed_out{false};
    std::thread watchdog([&] {
        const auto start = std::chrono::steady_clock::now();
        while (!done && std::chrono::steady_clock::now() - start < std::chrono::seconds(8))
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        if (!done) { timed_out = true; *(volatile unsigned*)host = 1; }
    });
    {
        strata::core::HostGraphLaunch launch(exec, stream, 0, threaded,
                                            [&] { *(volatile unsigned*)host = 1; });
        gpu(launch.status());
        check(launch.query() == cudaErrorNotReady, "graph waits for CPU service");
        *(volatile unsigned*)host = 1;
        gpu(launch.finish());
        gpu(cudaStreamSynchronize(stream));
    }
    done = true;
    watchdog.join();
    check(!timed_out, "CPU service proceeds without launch deadlock");
    unsigned count = 0;
    gpu(cudaMemcpy(&count, value, sizeof(unsigned), cudaMemcpyDeviceToHost));
    check(count == 4096, "all graph nodes executed");
    if (threaded) {
        // An early service failure must release the graph before joining.
        *host = 0;
        std::atomic<bool> released{false};
        {
            strata::core::HostGraphLaunch launch(exec, stream, 0, true,
                [&] { released = true; *(volatile unsigned*)host = 1; });
        }
        check(released, "early return releases device waits");
        gpu(cudaStreamSynchronize(stream));
        gpu(cudaMemcpy(&count, value, sizeof(unsigned), cudaMemcpyDeviceToHost));
        check(count == 8192, "early return drains the actual graph");
        if (argc > 1 && std::strcmp(argv[1], "--errors") == 0) {
            strata::core::HostGraphLaunch invalid(exec, stream, -1, true, [] {});
            check(invalid.finish() == cudaErrorInvalidDevice, "launch-thread device error propagated");
            (void)cudaGetLastError();
        }
    }
    gpu(cudaGraphExecDestroy(exec));
    gpu(cudaGraphDestroy(graph));
    gpu(cudaFree(value));
    gpu(cudaFreeHost(host));
    gpu(cudaStreamDestroy(stream));
    std::puts("PASS: graph launch, CPU doorbell service and cleanup");
}
