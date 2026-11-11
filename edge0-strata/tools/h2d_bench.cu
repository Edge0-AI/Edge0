// 0.1.43 5.2: do two streams move host->device data faster than one (both copy engines)?  And what does one layer's
// worth of expert-sized copies cost end to end (issue, copy, host function, sync)?
//   nvcc -O2 -o h2d_bench tools/h2d_bench.cu && ./h2d_bench [device]
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1); } } while (0)
using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

int main(int argc, char** argv) {
    int dev = argc > 1 ? std::atoi(argv[1]) : 0;
    CK(cudaSetDevice(dev));
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, dev));
    std::printf("%s asyncEngineCount=%d\n", p.name, p.asyncEngineCount);
    const size_t blob = 1382400;                    // a Q2_0 expert
    const size_t nb = 256;                          // blobs in the arena
    const size_t arena_bytes = nb * blob;
    uint8_t* host = nullptr;
    // like the engine's arena: plain pages registered with cudaHostRegister (PORTABLE)
    host = (uint8_t*) aligned_alloc(2 << 20, (arena_bytes + (2 << 20) - 1) / (2 << 20) * (2 << 20));
    for (size_t i = 0; i < arena_bytes; i += 4096) host[i] = (uint8_t) i;
    CK(cudaHostRegister(host, arena_bytes, cudaHostRegisterPortable));
    uint8_t* dbuf;
    CK(cudaMalloc(&dbuf, 2 * 64 * blob));
    cudaStream_t s[2];
    for (int i = 0; i < 2; ++i) CK(cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking));

    // 1. throughput: N blobs, one stream vs two streams (alternate blobs)
    for (int nstreams = 1; nstreams <= 2; ++nstreams) {
        double best = 1e9;
        for (int rep = 0; rep < 7; ++rep) {
            auto t0 = Clock::now();
            for (size_t i = 0; i < nb; ++i) CK(cudaMemcpyAsync(dbuf + (i % 64) * blob, host + i * blob, blob, cudaMemcpyHostToDevice, s[i % nstreams]));
            for (int i = 0; i < nstreams; ++i) CK(cudaStreamSynchronize(s[i]));
            best = std::min(best, ms_since(t0));
        }
        std::printf("throughput %d stream(s): %.3f ms for %zu blobs = %.2f GB/s\n", nstreams, best, nb, arena_bytes / best / 1e6);
    }
    // 1b. the same with each stream copying one big contiguous range
    for (int nstreams = 1; nstreams <= 2; ++nstreams) {
        double best = 1e9;
        for (int rep = 0; rep < 7; ++rep) {
            auto t0 = Clock::now();
            size_t half = 64 * blob / nstreams;
            for (int i = 0; i < nstreams; ++i) CK(cudaMemcpyAsync(dbuf + i * half, host + i * half, half, cudaMemcpyHostToDevice, s[i]));
            for (int i = 0; i < nstreams; ++i) CK(cudaStreamSynchronize(s[i]));
            best = std::min(best, ms_since(t0));
        }
        std::printf("big range %d stream(s): %.3f ms for %zu bytes = %.2f GB/s\n", nstreams, best, 64 * blob, 64 * blob / best / 1e6);
    }
    // 2. per-layer latency: k blobs, then a host function, then wait for it (the engine's flag B)
    for (int k : {1, 2, 4, 8, 16}) {
        for (int nstreams = 1; nstreams <= 2; ++nstreams) {
            std::vector<double> v;
            for (int rep = 0; rep < 200; ++rep) {
                static volatile int flag;
                flag = 0;
                auto t0 = Clock::now();
                for (int i = 0; i < k; ++i) CK(cudaMemcpyAsync(dbuf + i * blob, host + ((rep * 13 + i * 7) % nb) * blob, blob, cudaMemcpyHostToDevice, s[i % nstreams]));
                if (nstreams == 2) {
                    cudaEvent_t ev;
                    CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
                    CK(cudaEventRecord(ev, s[1]));
                    CK(cudaStreamWaitEvent(s[0], ev, 0));
                    CK(cudaEventDestroy(ev));
                }
                CK(cudaLaunchHostFunc(s[0], [](void*) { flag = 1; }, nullptr));
                while (!flag) {}
                v.push_back(ms_since(t0));
            }
            std::sort(v.begin(), v.end());
            std::printf("layer of %2d blobs, %d stream(s): median %.3f ms  min %.3f  p90 %.3f  (ideal at 11 GB/s %.3f)\n", k, nstreams, v[v.size() / 2], v[0], v[v.size() * 9 / 10], k * blob / 11e6);
        }
    }
    return 0;
}
