// hwbench: host RAM bandwidth, PCIe host->device, and unbuffered overlapped NVMe reads (Windows).
//   hwbench ram [threads] [MiB]
//   hwbench pcie
//   hwbench nvme <file> <block_KiB> <qd> <seconds> [seq|rand]
#include <cuda_runtime.h>
#include <windows.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static void bench_ram(int threads, size_t mib) {
    size_t n = mib << 20;
    std::vector<char*> src(threads), dst(threads);
    for (int t = 0; t < threads; t++) {
        src[t] = (char*)_aligned_malloc(n, 4096); dst[t] = (char*)_aligned_malloc(n, 4096);
        memset(src[t], t + 1, n); memset(dst[t], 0, n);
    }
    for (int mode = 0; mode < 2; mode++) {
        double best = 0;
        for (int rep = 0; rep < 4; rep++) {
            std::vector<std::thread> th; volatile uint64_t sink = 0;
            double t0 = now();
            for (int t = 0; t < threads; t++)
                th.emplace_back([&, t] {
                    if (mode == 0) memcpy(dst[t], src[t], n);
                    else { uint64_t s = 0; const uint64_t* p = (const uint64_t*)src[t]; for (size_t i = 0; i < n / 8; i += 1) s += p[i]; sink = sink + s; }
                });
            for (auto& x : th) x.join();
            double dt = now() - t0; double gb = (mode == 0 ? 2.0 : 1.0) * n * threads / 1e9 / dt;  // copy counts read+write
            if (gb > best) best = gb;
        }
        printf("ram %s threads=%d size/thread=%zu MiB best=%.1f GB/s (%s)\n", mode == 0 ? "copy" : "read", threads, mib, best, mode == 0 ? "read+write counted" : "read only");
    }
}

static void bench_pcie() {
    for (size_t mib : {1, 4, 16, 64, 256}) {
        size_t n = mib << 20; void *h, *d; cudaStream_t s; cudaStreamCreate(&s);
        cudaMallocHost(&h, n); cudaMalloc(&d, n); memset(h, 1, n);
        void* hp = _aligned_malloc(n, 4096); memset(hp, 1, n);
        for (int pinned = 1; pinned >= 0; pinned--) {
            void* src = pinned ? h : hp;
            for (int w = 0; w < 3; w++) cudaMemcpyAsync(d, src, n, cudaMemcpyHostToDevice, s);
            cudaStreamSynchronize(s);
            int reps = (int)std::max<size_t>(8, (1024u / mib)); double t0 = now();
            for (int i = 0; i < reps; i++) cudaMemcpyAsync(d, src, n, cudaMemcpyHostToDevice, s);
            cudaStreamSynchronize(s); double dt = now() - t0;
            printf("pcie H2D %s %4zu MiB: %.2f GB/s\n", pinned ? "pinned  " : "pageable", mib, (double)n * reps / 1e9 / dt);
        }
        cudaFreeHost(h); cudaFree(d); _aligned_free(hp); cudaStreamDestroy(s);
    }
}

static void bench_nvme(const char* path, size_t block, int qd, double secs, bool seq) {
    HANDLE f = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
    if (f == INVALID_HANDLE_VALUE) { printf("open failed %lu\n", GetLastError()); return; }
    LARGE_INTEGER sz; GetFileSizeEx(f, &sz);
    HANDLE port = CreateIoCompletionPort(f, nullptr, 0, 1);
    uint64_t nblk = (uint64_t)sz.QuadPart / block;
    struct Slot { OVERLAPPED ov; char* buf; };
    std::vector<Slot> slots(qd);
    std::mt19937_64 rng(12345);
    uint64_t seqpos = 0, bytes = 0, ops = 0;
    std::vector<double> lat; std::vector<double> t_issue(qd);
    auto issue = [&](int i) {
        uint64_t b = seq ? (seqpos++ % nblk) : (rng() % nblk);
        uint64_t off = b * block;
        memset(&slots[i].ov, 0, sizeof(OVERLAPPED));
        slots[i].ov.Offset = (DWORD)(off & 0xffffffff); slots[i].ov.OffsetHigh = (DWORD)(off >> 32);
        t_issue[i] = now();
        if (!ReadFile(f, slots[i].buf, (DWORD)block, nullptr, &slots[i].ov) && GetLastError() != ERROR_IO_PENDING) { printf("ReadFile err %lu\n", GetLastError()); exit(1); }
    };
    for (int i = 0; i < qd; i++) { slots[i].buf = (char*)VirtualAlloc(nullptr, block, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE); issue(i); }
    double t0 = now(), tend = t0 + secs;
    while (true) {
        DWORD nb; ULONG_PTR key; OVERLAPPED* ov;
        if (!GetQueuedCompletionStatus(port, &nb, &key, &ov, 10000) && !ov) { printf("wait err\n"); break; }
        int i = (int)(((Slot*)((char*)ov - offsetof(Slot, ov))) - slots.data());
        bytes += nb; ops++; lat.push_back(now() - t_issue[i]);
        if (now() < tend) issue(i); else if (--qd == 0) break;
    }
    double dt = now() - t0;
    std::sort(lat.begin(), lat.end());
    printf("nvme %s block=%zuKiB qd=%zu: %.2f GB/s, %.0f IOPS, lat p50=%.2f ms p99=%.2f ms\n", seq ? "seq " : "rand", block >> 10, slots.size(),
           bytes / 1e9 / dt, ops / dt, lat[lat.size() / 2] * 1e3, lat[lat.size() * 99 / 100] * 1e3);
    CloseHandle(f);
}

// GPU kernels reading host memory over PCIe in place (zero-copy): the pinned + mapped (or registered) arena as a weights source
__global__ void zc_sum(const int4* __restrict__ p, size_t n, unsigned long long* out) {
    unsigned long long acc = 0;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const int4 v = p[i];
        acc += (unsigned) v.x ^ (unsigned) v.y ^ (unsigned) v.z ^ (unsigned) v.w;
    }
    if (acc == 0x123456789ull) atomicAdd(out, acc);
}
static void bench_zc(size_t mib) {
    const size_t n = mib << 20;
    for (int mode = 0; mode < 2; ++mode) {
        void *h = nullptr, *dp = nullptr;
        if (mode == 0) {
            if (cudaHostAlloc(&h, n, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) { printf("hostalloc failed\n"); continue; }
        } else {
            h = VirtualAlloc(nullptr, n, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            memset(h, 1, n);
            if (cudaHostRegister(h, n, cudaHostRegisterMapped | cudaHostRegisterPortable) != cudaSuccess) { printf("hostregister failed\n"); continue; }
        }
        memset(h, 1, n);
        cudaHostGetDevicePointer(&dp, h, 0);
        unsigned long long* out; cudaMalloc(&out, 8);
        for (int blocks : {64, 256, 1024}) {
            zc_sum<<<blocks, 256>>>((const int4*) dp, n / 16, out);
            cudaDeviceSynchronize();
            cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
            cudaEventRecord(a);
            for (int r = 0; r < 4; ++r) zc_sum<<<blocks, 256>>>((const int4*) dp, n / 16, out);
            cudaEventRecord(b); cudaEventSynchronize(b);
            float ms; cudaEventElapsedTime(&ms, a, b);
            printf("zero-copy %s %zu MiB, %d blocks: %.2f GB/s\n", mode == 0 ? "cudaHostAlloc(mapped)" : "VirtualAlloc+cudaHostRegister(mapped)", mib, blocks, 4.0 * n / 1e9 / (ms / 1e3));
        }
        if (mode == 0) cudaFreeHost(h); else { cudaHostUnregister(h); VirtualFree(h, 0, MEM_RELEASE); }
        cudaFree(out);
    }
}

int main(int argc, char** argv) {
    std::string m = argc > 1 ? argv[1] : "";
    if (m == "ram") bench_ram(argc > 2 ? atoi(argv[2]) : 6, argc > 3 ? atoi(argv[3]) : 512);
    else if (m == "pcie") bench_pcie();
    else if (m == "zc") bench_zc(argc > 2 ? atoi(argv[2]) : 1024);
    else if (m == "nvme" && argc > 5) bench_nvme(argv[2], (size_t)atoi(argv[3]) << 10, atoi(argv[4]), atof(argv[5]), argc > 6 && std::string(argv[6]) == "seq");
    else printf("usage: hwbench ram|pcie|nvme ...\n");
}
