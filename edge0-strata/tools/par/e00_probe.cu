// E00 capability probe: peer access, GPU<->GPU / H2D / D2H transfer latency+bandwidth at plan sizes, RAM bandwidth.
// build: nvcc -O2 -std=c++17 e00_probe.cu -o e00_probe -lpthread ; run: ./e00_probe > e00.json
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>
#include <algorithm>
using Clk = std::chrono::steady_clock;
static double now_us() { return std::chrono::duration<double, std::micro>(Clk::now().time_since_epoch()).count(); }
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(2);} } while (0)
static const size_t SIZES[] = {4096, 16384, 65536, 262144, 1 << 20, 8 << 20, 64 << 20};

template <class F> double med(F f, int reps) {
    f(); f();
    std::vector<double> v;
    for (int i = 0; i < reps; ++i) { double t = now_us(); f(); v.push_back(now_us() - t); }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
int main() {
    int n = 0; CK(cudaGetDeviceCount(&n));
    printf("{\n \"gpus\": [");
    for (int d = 0; d < n; ++d) {
        cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, d));
        size_t fr, tot; CK(cudaSetDevice(d)); CK(cudaMemGetInfo(&fr, &tot));
        printf("%s{\"index\":%d,\"name\":\"%s\",\"sm\":%d,\"sms\":%d,\"free_mib\":%zu,\"total_mib\":%zu,\"pci_bus\":%d}", d ? "," : "", d, p.name, p.major * 10 + p.minor, p.multiProcessorCount, fr >> 20, tot >> 20, p.pciBusID);
    }
    printf("],\n \"peer\": [");
    bool first = true;
    for (int a = 0; a < n; ++a) for (int b = 0; b < n; ++b) if (a != b) {
        int can = 0; cudaDeviceGetP2PAttribute(&can, cudaDevP2PAttrAccessSupported, a, b);
        int nat = 0; cudaDeviceGetP2PAttribute(&nat, cudaDevP2PAttrNativeAtomicSupported, a, b);
        printf("%s{\"from\":%d,\"to\":%d,\"access\":%d,\"native_atomics\":%d}", first ? "" : ",", a, b, can, nat); first = false;
    }
    printf("],\n \"transfers\": [");
    first = true;
    auto emit = [&](const char* kind, int a, int b, size_t sz, double us) {
        double gbs = (double) sz / us / 1e3;
        printf("%s\n  {\"kind\":\"%s\",\"a\":%d,\"b\":%d,\"bytes\":%zu,\"median_us\":%.2f,\"gb_s\":%.3f}", first ? "" : ",", kind, a, b, sz, us, gbs); first = false;
    };
    size_t maxs = 64 << 20;
    void* h = nullptr; CK(cudaMallocHost(&h, maxs)); memset(h, 1, maxs);
    std::vector<void*> dbuf(n), dbuf2(n);
    std::vector<cudaStream_t> st(n);
    for (int d = 0; d < n; ++d) { CK(cudaSetDevice(d)); CK(cudaMalloc(&dbuf[d], maxs)); CK(cudaMalloc(&dbuf2[d], maxs)); CK(cudaStreamCreateWithFlags(&st[d], cudaStreamNonBlocking)); }
    for (int d = 0; d < n; ++d) {
        CK(cudaSetDevice(d));
        for (size_t sz : SIZES) {
            emit("h2d_pinned", d, -1, sz, med([&] { CK(cudaMemcpyAsync(dbuf[d], h, sz, cudaMemcpyHostToDevice, st[d])); CK(cudaStreamSynchronize(st[d])); }, 30));
            emit("d2h_pinned", d, -1, sz, med([&] { CK(cudaMemcpyAsync(h, dbuf[d], sz, cudaMemcpyDeviceToHost, st[d])); CK(cudaStreamSynchronize(st[d])); }, 30));
        }
        cudaStream_t s2; CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
        double us = med([&] { CK(cudaMemcpyAsync(dbuf[d], h, maxs / 2, cudaMemcpyHostToDevice, st[d])); CK(cudaMemcpyAsync((char*) h + maxs / 2, dbuf2[d], maxs / 2, cudaMemcpyDeviceToHost, s2)); CK(cudaStreamSynchronize(st[d])); CK(cudaStreamSynchronize(s2)); }, 10);
        emit("h2d_d2h_bidir_total", d, -1, maxs, us);
    }
    for (int a = 0; a < n; ++a) for (int b = 0; b < n; ++b) if (a != b) {
        int can = 0; cudaDeviceCanAccessPeer(&can, a, b);
        if (can) { CK(cudaSetDevice(a)); cudaDeviceEnablePeerAccess(b, 0); cudaGetLastError(); }
        CK(cudaSetDevice(a));
        for (size_t sz : SIZES)
            emit(can ? "p2p_peer_memcpy" : "gpu_to_gpu_memcpy_staged", a, b, sz, med([&] { CK(cudaMemcpyPeerAsync(dbuf2[b], b, dbuf[a], a, sz, st[a])); CK(cudaStreamSynchronize(st[a])); }, 30));
    }
    if (n >= 2) {
        CK(cudaSetDevice(0)); cudaDeviceEnablePeerAccess(1, 0); cudaGetLastError(); CK(cudaSetDevice(1)); cudaDeviceEnablePeerAccess(0, 0); cudaGetLastError();
        double us = med([&] { CK(cudaMemcpyPeerAsync(dbuf2[1], 1, dbuf[0], 0, maxs, st[0])); CK(cudaMemcpyPeerAsync(dbuf2[0], 0, dbuf[1], 1, maxs, st[1])); CK(cudaStreamSynchronize(st[0])); CK(cudaStreamSynchronize(st[1])); }, 10);
        emit("p2p_bidir_total_bytes", 0, 1, 2 * maxs, us);
        void* h2 = nullptr; CK(cudaMallocHost(&h2, maxs));
        us = med([&] { CK(cudaMemcpyAsync(dbuf[0], h, maxs, cudaMemcpyHostToDevice, st[0])); CK(cudaMemcpyAsync(dbuf[1], h2, maxs, cudaMemcpyHostToDevice, st[1])); CK(cudaStreamSynchronize(st[0])); CK(cudaStreamSynchronize(st[1])); }, 10);
        emit("h2d_both_gpus_total_bytes", 0, 1, 2 * maxs, us);
    }
    printf("\n ],\n");
    size_t N = (512u << 20) / sizeof(double);
    std::vector<double> A(N, 1.0), Bv(N, 2.0), C(N, 0.0);
    printf(" \"ram_triad_gb_s\": {");
    unsigned hw = std::thread::hardware_concurrency();
    bool f2 = true;
    for (unsigned t : {1u, 2u, 4u, 8u, 16u, hw}) {
        if (t > hw) continue;
        double best = 1e30;
        for (int r = 0; r < 3; ++r) {
            double t0 = now_us();
            std::vector<std::thread> th;
            for (unsigned i = 0; i < t; ++i) th.emplace_back([&, i] { size_t lo = N * i / t, hi = N * (i + 1) / t; for (size_t k = lo; k < hi; ++k) C[k] = A[k] + 3.0 * Bv[k]; });
            for (auto& x : th) x.join();
            best = std::min(best, now_us() - t0);
        }
        printf("%s\"%u\": %.1f", f2 ? "" : ", ", t, 3.0 * N * sizeof(double) / best / 1e3); f2 = false;
    }
    printf("},\n \"hw_threads\": %u\n}\n", hw);
    return 0;
}
