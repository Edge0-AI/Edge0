// src/core/dflash2_launch_bench.cpp - what a dependent kernel costs inside a captured graph on this GPU: N tiny kernels in a chain, and N kernels of 256 blocks.
//
//     dflash2_launch_bench [n=100]
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

__global__ void tiny(float* x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] * 1.0001f + 1.0f;
}

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 100;
    float* x;
    cudaMalloc((void**) &x, 1 << 24);
    cudaMemset(x, 0, 1 << 24);
    cudaStream_t cs;
    cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking);
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int blocks : {1, 64, 256, 1024}) {
        cudaGraph_t g;
        cudaGraphExec_t ge;
        cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal);
        for (int i = 0; i < n; ++i) tiny<<<blocks, 256, 0, cs>>>(x, blocks * 256);
        cudaStreamEndCapture(cs, &g);
        cudaGraphInstantiate(&ge, g, 0);
        cudaGraphLaunch(ge, cs);
        cudaStreamSynchronize(cs);
        cudaEventRecord(a, cs);
        for (int it = 0; it < 50; ++it) cudaGraphLaunch(ge, cs);
        cudaEventRecord(b, cs);
        cudaEventSynchronize(b);
        float ms;
        cudaEventElapsedTime(&ms, a, b);
        std::printf("graph of %d chained kernels, %4d blocks: %.2f us per kernel\n", n, blocks, 1000.0 * ms / 50 / n);
        // one launch from an idle stream: event, launch, event, wait (what the drafter does each round)
        {
            double tot = 0, wall = 0;
            for (int it = 0; it < 20; ++it) {
                cudaStreamSynchronize(cs);
                auto t0 = std::chrono::steady_clock::now();
                cudaEventRecord(a, cs);
                cudaGraphLaunch(ge, cs);
                cudaEventRecord(b, cs);
                cudaStreamSynchronize(cs);
                wall += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                float m;
                cudaEventElapsedTime(&m, a, b);
                tot += m;
            }
            std::printf("single launch from idle, %d kernels, %4d blocks: events %.3f ms, wall %.3f ms\n", n, blocks, tot / 20, wall / 20);
        }
        // the same without a graph
        cudaEventRecord(a, cs);
        for (int it = 0; it < 5; ++it)
            for (int i = 0; i < n; ++i) tiny<<<blocks, 256, 0, cs>>>(x, blocks * 256);
        cudaEventRecord(b, cs);
        cudaEventSynchronize(b);
        cudaEventElapsedTime(&ms, a, b);
        std::printf("stream of %d chained kernels, %4d blocks: %.2f us per kernel\n", n, blocks, 1000.0 * ms / 5 / n);
    }
    return 0;
}
