// tools/sm60_dp4a_check.cu - proves the sm_60 STRATA_DP4A fallback in include/strata/kernels/dp4a.hpp is
// bit-exact with a plain signed byte-wise dot product, and times it against that byte loop.
//
// It includes the committed header, so the check covers the code the kernels compile, not a copy.  On sm_61+
// STRATA_DP4A is the hardware __dp4a and the check still applies.  No CMake target; build by hand:
//
//   nvcc -O3 -arch=sm_60 -Iinclude tools/sm60_dp4a_check.cu -o sm60_dp4a_check && ./sm60_dp4a_check
//
// Exit status 0 = no mismatches and no CUDA error.
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#include "strata/kernels/dp4a.hpp"

// Reference: the signed-byte semantics __dp4a(int, int, int) has for these operands.
__device__ __forceinline__ int byte_dp4a(const int a, const int b, const int c) {
    const int8_t* a8 = (const int8_t*) &a;
    const int8_t* b8 = (const int8_t*) &b;
    return c + a8[0] * b8[0] + a8[1] * b8[1] + a8[2] * b8[2] + a8[3] * b8[3];
}

// splitmix32: every output byte depends on the whole state, so all byte lanes see all byte pairs.
__device__ uint32_t rnd(uint32_t& s) {
    uint32_t z = (s += 0x9e3779b9u);
    z = (z ^ (z >> 16)) * 0x85ebca6bu;
    z = (z ^ (z >> 13)) * 0xc2b2ae35u;
    return z ^ (z >> 16);
}

constexpr int BLOCKS = 1024, THREADS = 256, ITERS = 4096;  // 1024 * 256 * 4096 = ~1.07e9 cases

// Per thread: iterations 0-15 pair every edge byte (0x7f, 0x80, 0xff, 0x00) in a with every edge byte in b.
// Iteration 16 is exhaustive across the grid: the 262,144 threads are exactly 4 lanes x 256 x 256, so every
// (lane, a byte, b byte) combination is tested once, with random bytes in the other lanes.  The rest is random.
__global__ void check(unsigned long long* mismatches, int iters) {
    const int edge[4] = {0x7f7f7f7f, (int) 0x80808080u, (int) 0xffffffffu, 0};
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t s = t * 0x2545f491u;
    unsigned long long bad = 0;
    for (int i = 0; i < iters; ++i) {
        int a = (int) rnd(s), b = (int) rnd(s), c = (int) rnd(s);
        if (i < 16) { a = edge[i & 3]; b = edge[i >> 2]; }
        if (i == 16) {
            const int sh = 8 * (t >> 16);
            a = (int) (((uint32_t) a & ~(0xffu << sh)) | (((t >> 8) & 0xffu) << sh));
            b = (int) (((uint32_t) b & ~(0xffu << sh)) | ((t & 0xffu) << sh));
        }
        bad += byte_dp4a(a, b, c) != STRATA_DP4A(a, b, c);
    }
    atomicAdd(mismatches, bad);
}

template <bool STRATA>
__global__ void bench(int* out, int iters) {
    int a = threadIdx.x * 0x01020304, b = blockIdx.x * 0x05060708 + 1, acc = 0;
    for (int i = 0; i < iters; ++i) {
        acc = STRATA ? STRATA_DP4A(a, b, acc) : byte_dp4a(a, b, acc);
        a ^= acc; b += 0x01010101;
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template <bool STRATA>
float time_it(int* out) {
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    bench<STRATA><<<BLOCKS, THREADS>>>(out, 1 << 12);  // warm-up
    cudaEventRecord(t0);
    bench<STRATA><<<BLOCKS, THREADS>>>(out, 1 << 16);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms;
}

int main() {
    static_assert(BLOCKS * THREADS == 4 * 256 * 256, "iteration 16 needs one thread per (lane, a byte, b byte)");
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    unsigned long long* d_bad; int* d_out;
    cudaMalloc(&d_bad, sizeof(*d_bad)); cudaMemset(d_bad, 0, sizeof(*d_bad));
    cudaMalloc(&d_out, BLOCKS * THREADS * sizeof(int));
    check<<<BLOCKS, THREADS>>>(d_bad, ITERS);
    unsigned long long bad = 0;
    cudaError_t e = cudaMemcpy(&bad, d_bad, sizeof(bad), cudaMemcpyDeviceToHost);
    float byte_ms = time_it<false>(d_out), strata_ms = time_it<true>(d_out);
    printf("cases=%llu mismatches=%llu\n", (unsigned long long) BLOCKS * THREADS * ITERS, bad);
    printf("byte loop %.1f ms  STRATA_DP4A %.1f ms  speedup %.2fx\n", byte_ms, strata_ms, byte_ms / strata_ms);
    if (e == cudaSuccess) e = cudaGetLastError();
    printf("cuda: %s\n", cudaGetErrorString(e));
    cudaFree(d_bad); cudaFree(d_out);
    return (bad == 0 && e == cudaSuccess) ? 0 : 1;
}
