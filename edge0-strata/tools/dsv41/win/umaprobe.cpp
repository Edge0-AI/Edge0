// umaprobe (HIP, Windows): what a unified-memory APU (Strix Halo, gfx1151) lets the GPU use, and how fast.
//   umaprobe info                       device figures, hipMemGetInfo, the DXGI video memory budget, system RAM
//   umaprobe alloc <GiB cap>            hipMalloc 1 GiB chunks until failure or the cap: how much, where RAM goes, GPU read GB/s per region
//   umaprobe reg <GiB> [pagesMB]        hipHostRegister of GiB of VirtualAlloc'd memory: works?, time, GPU read GB/s through it
//   umaprobe regchunk <GiB> <chunk GiB>  register in pieces
//   umaprobe hostalloc <GiB>            hipHostMalloc (mapped) of GiB: works?, GPU read GB/s
//   umaprobe copy                       hipMemcpy H2D / D2H pageable and pinned
// Build: tools\dsv41\win\build_umaprobe.bat (ROCm clang, gfx1151).  Internal tool, not shipped.
#include <hip/hip_runtime.h>
#include <windows.h>
#include <psapi.h>
#include <dxgi1_4.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("%s:%d %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static double ram_avail_gib() { MEMORYSTATUSEX m{}; m.dwLength = sizeof m; GlobalMemoryStatusEx(&m); return m.ullAvailPhys / 1073741824.0; }
static double ram_total_gib() { MEMORYSTATUSEX m{}; m.dwLength = sizeof m; GlobalMemoryStatusEx(&m); return m.ullTotalPhys / 1073741824.0; }

static void dxgi_line(const char* tag) {
    static IDXGIAdapter3* ad = nullptr;
    if (!ad) {
        hipDeviceProp_t p{}; hipGetDeviceProperties(&p, 0);
        LUID luid; std::memcpy(&luid, p.luid, sizeof luid);
        IDXGIFactory4* f = nullptr; CreateDXGIFactory1(__uuidof(IDXGIFactory4), (void**) &f);
        if (f) { f->EnumAdapterByLuid(luid, __uuidof(IDXGIAdapter3), (void**) &ad); f->Release(); }
    }
    if (!ad) { std::printf("%s dxgi: no adapter\n", tag); return; }
    DXGI_QUERY_VIDEO_MEMORY_INFO l{}, n{};
    ad->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &l);
    ad->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &n);
    std::printf("%s dxgi local budget %.1f GiB usage %.1f | non-local budget %.1f GiB usage %.1f | RAM avail %.1f / %.1f GiB\n", tag,
                l.Budget / 1073741824.0, l.CurrentUsage / 1073741824.0, n.Budget / 1073741824.0, n.CurrentUsage / 1073741824.0, ram_avail_gib(), ram_total_gib());
}

__global__ void read_kernel(const uint4* __restrict__ p, size_t n, uint32_t* out) {
    uint32_t acc = 0;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const uint4 v = p[i];
        acc += v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) out[0] = acc;
}

static double gpu_read_gbps(const void* p, size_t bytes, int reps = 5) {
    uint32_t* o; CK(hipMalloc(&o, 64));
    const size_t n = bytes / 16;
    const int blocks = 4096, threads = 256;
    read_kernel<<<blocks, threads>>>((const uint4*) p, n, o);
    CK(hipDeviceSynchronize());
    double best = 0;
    for (int r = 0; r < reps; ++r) {
        const double t0 = now();
        read_kernel<<<blocks, threads>>>((const uint4*) p, n, o);
        CK(hipDeviceSynchronize());
        const double g = bytes / 1e9 / (now() - t0);
        if (g > best) best = g;
    }
    hipFree(o);
    return best;
}

static void info() {
    int nd = 0; CK(hipGetDeviceCount(&nd));
    for (int d = 0; d < nd; ++d) {
        hipDeviceProp_t p{}; CK(hipGetDeviceProperties(&p, d));
        std::printf("dev %d %s arch %s CUs %d totalGlobalMem %.2f GiB integrated %d canMapHostMemory %d managed %d pageableMemoryAccess %d unifiedAddr %d\n", d, p.name, p.gcnArchName,
                    p.multiProcessorCount, p.totalGlobalMem / 1073741824.0, p.integrated, p.canMapHostMemory, p.managedMemory, p.pageableMemoryAccess, p.unifiedAddressing);
    }
    size_t f, t; CK(hipMemGetInfo(&f, &t));
    std::printf("hipMemGetInfo free %.2f total %.2f GiB\n", f / 1073741824.0, t / 1073741824.0);
    dxgi_line("info");
    {   // the adapter's own split: the BIOS carve-out (dedicated) and the shared (system RAM) part
        hipDeviceProp_t p{}; hipGetDeviceProperties(&p, 0);
        LUID luid; std::memcpy(&luid, p.luid, sizeof luid);
        IDXGIFactory4* f = nullptr; CreateDXGIFactory1(__uuidof(IDXGIFactory4), (void**) &f);
        IDXGIAdapter1* a = nullptr;
        if (f && SUCCEEDED(f->EnumAdapterByLuid(luid, __uuidof(IDXGIAdapter1), (void**) &a))) {
            DXGI_ADAPTER_DESC1 d{}; a->GetDesc1(&d);
            std::printf("adapter dedicated video %.1f GiB, dedicated system %.1f GiB, shared system %.1f GiB\n", d.DedicatedVideoMemory / 1073741824.0,
                        d.DedicatedSystemMemory / 1073741824.0, d.SharedSystemMemory / 1073741824.0);
            a->Release();
        }
        if (f) f->Release();
    }
}

static void alloc(int cap) {
    info();
    std::vector<void*> ch;
    const size_t C = 1ull << 30;
    for (int i = 0; i < cap; ++i) {
        void* p = nullptr;
        const double t0 = now();
        if (hipMalloc(&p, C) != hipSuccess) { (void) hipGetLastError(); std::printf("hipMalloc failed at chunk %d (%d GiB allocated)\n", i, i); break; }
        CK(hipMemset(p, 1, C));
        ch.push_back(p);
        if ((i + 1) % 8 == 0 || i == 0) {
            size_t f, t; hipMemGetInfo(&f, &t);
            char tag[64]; std::snprintf(tag, sizeof tag, "after %d GiB (%.2fs/chunk) memfree %.1f", i + 1, now() - t0, f / 1073741824.0);
            dxgi_line(tag);
        }
    }
    std::printf("allocated %zu GiB\n", ch.size());
    // read bandwidth per 4 GiB window: the first (carve-out?) and the last (shared?)
    for (size_t w : {size_t(0), ch.size() / 2, ch.size() >= 4 ? ch.size() - 1 : 0}) {
        if (ch.empty()) break;
        std::printf("GPU read chunk %zu (1 GiB): %.1f GB/s\n", w, gpu_read_gbps(ch[w], C));
    }
    for (void* p : ch) hipFree(p);
}

static void reg(int gib, int pages_mb) {
    info();
    const size_t n = (size_t) gib << 30;
    double t0 = now();
    void* p = VirtualAlloc(nullptr, n, MEM_COMMIT | MEM_RESERVE | (pages_mb > 0 ? MEM_LARGE_PAGES : 0), PAGE_READWRITE);
    if (!p) { std::printf("VirtualAlloc %d GiB failed %lu\n", gib, GetLastError()); return; }
    std::memset(p, 7, n);
    std::printf("committed+touched %d GiB in %.1f s\n", gib, now() - t0);
    dxgi_line("committed");
    t0 = now();
    hipError_t e = hipHostRegister(p, n, hipHostRegisterPortable | hipHostRegisterMapped);
    std::printf("hipHostRegister(%d GiB, mapped|portable): %s in %.2f s\n", gib, hipGetErrorString(e), now() - t0);
    if (e != hipSuccess) { (void) hipGetLastError(); e = hipHostRegister(p, n, hipHostRegisterPortable); std::printf("hipHostRegister(portable only): %s\n", hipGetErrorString(e)); (void) hipGetLastError(); }
    dxgi_line("registered");
    if (e == hipSuccess) {
        void* dp = nullptr;
        if (hipHostGetDevicePointer(&dp, p, 0) != hipSuccess) { (void) hipGetLastError(); dp = p; }
        const size_t w = std::min<size_t>(n, 4ull << 30);
        std::printf("GPU read of registered host memory (%zu GiB window): %.1f GB/s%s\n", w >> 30, gpu_read_gbps(dp, w), dp == p ? " (same ptr)" : " (device ptr)");
        t0 = now();
        for (size_t o = 0; o + (1ull << 30) <= n; o += (1ull << 30)) { /* touch pattern: sequential 1 GiB reads */ }
        void* d; if (hipMalloc(&d, 1ull << 30) == hipSuccess) {
            t0 = now(); for (int i = 0; i < 4; ++i) CK(hipMemcpy(d, p, 1ull << 30, hipMemcpyHostToDevice));
            std::printf("hipMemcpy H2D from registered memory: %.1f GB/s\n", 4.0 / (now() - t0));
            hipFree(d);
        }
        hipHostUnregister(p);
    }
    VirtualFree(p, 0, MEM_RELEASE);
}

static void regchunk(int gib, int chunk) {
    info();
    const size_t C = (size_t) chunk << 30;
    std::vector<void*> v;
    double t0 = now();
    for (int o = 0; o < gib; o += chunk) {
        void* p = VirtualAlloc(nullptr, C, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (!p) { std::printf("VirtualAlloc failed at %d GiB\n", o); break; }
        std::memset(p, 5, C);
        const double t1 = now();
        const hipError_t e = hipHostRegister(p, C, hipHostRegisterPortable | hipHostRegisterMapped);
        std::printf("register chunk at %d GiB (%d GiB): %s in %.2f s\n", o, chunk, hipGetErrorString(e), now() - t1);
        std::fflush(stdout);
        if (e != hipSuccess) { (void) hipGetLastError(); break; }
        v.push_back(p);
    }
    std::printf("registered %zu chunks in %.1f s\n", v.size(), now() - t0);
    dxgi_line("regchunk");
    for (size_t i = 0; i < v.size(); i += std::max<size_t>(1, v.size() / 3)) {
        void* dp = nullptr; if (hipHostGetDevicePointer(&dp, v[i], 0) != hipSuccess) { (void) hipGetLastError(); dp = v[i]; }
        std::printf("GPU read chunk %zu: %.1f GB/s\n", i, gpu_read_gbps(dp, std::min<size_t>(C, 4ull << 30)));
    }
    for (void* p : v) { hipHostUnregister(p); VirtualFree(p, 0, MEM_RELEASE); }
}


// regsub: the engine's store pattern: one big VirtualAlloc(MEM_RESERVE), committed in chunks of N MiB (untouched or touched), each registered mapped
static void regsub(int chunk_mib, int nchunks, int touch) {
    info();
    const size_t C = (size_t) chunk_mib << 20;
    uint8_t* base = (uint8_t*) VirtualAlloc(nullptr, C * (size_t) nchunks, MEM_RESERVE, PAGE_READWRITE);
    if (!base) { std::printf("reserve failed\n"); return; }
    std::vector<void*> dev;
    for (int i = 0; i < nchunks; ++i) {
        uint8_t* p = base + C * (size_t) i;
        if (!VirtualAlloc(p, C, MEM_COMMIT, PAGE_READWRITE)) { std::printf("commit failed at %d\n", i); break; }
        if (touch) std::memset(p, 7, C);
        hipError_t e = hipHostRegister(p, C, hipHostRegisterPortable | hipHostRegisterMapped);
        void* dp = nullptr;
        if (e == hipSuccess) e = hipHostGetDevicePointer(&dp, p, 0);
        std::printf("chunk %d: register+devptr %s, dev %p host %p\n", i, hipGetErrorString(e), dp, (void*) p);
        if (e != hipSuccess) { (void) hipGetLastError(); break; }
        dev.push_back(dp);
        if (!touch) std::memset(p, 7, C);   // (the store's reads fill it after the registration)
    }
    for (size_t i = 0; i < dev.size(); ++i) {
        std::printf("GPU read chunk %zu: %.1f GB/s\n", i, gpu_read_gbps(dev[i], C));
        std::fflush(stdout);
    }
}

static void hostalloc(int gib) {
    info();
    void* p = nullptr;
    const size_t n = (size_t) gib << 30;
    const double t0 = now();
    hipError_t e = hipHostMalloc(&p, n, hipHostMallocMapped | hipHostMallocPortable);
    std::printf("hipHostMalloc(%d GiB, mapped): %s in %.2f s\n", gib, hipGetErrorString(e), now() - t0);
    if (e != hipSuccess) return;
    std::memset(p, 3, n);
    dxgi_line("hostmalloc");
    void* dp = nullptr; if (hipHostGetDevicePointer(&dp, p, 0) != hipSuccess) { (void) hipGetLastError(); dp = p; }
    std::printf("GPU read of hipHostMalloc memory: %.1f GB/s\n", gpu_read_gbps(dp, std::min<size_t>(n, 4ull << 30)));
    hipHostFree(p);
}

static void copy() {
    for (size_t mib : {16, 64, 256, 1024}) {
        const size_t n = mib << 20; void *d, *hp, *hg; hipStream_t s; CK(hipStreamCreate(&s)); CK(hipMalloc(&d, n));
        CK(hipHostMalloc(&hp, n, hipHostMallocDefault)); hg = _aligned_malloc(n, 4096); std::memset(hp, 1, n); std::memset(hg, 1, n);
        for (int pinned = 1; pinned >= 0; --pinned) {
            void* src = pinned ? hp : hg;
            for (int w = 0; w < 2; ++w) CK(hipMemcpyAsync(d, src, n, hipMemcpyHostToDevice, s));
            CK(hipStreamSynchronize(s));
            const int reps = (int) std::max<size_t>(4, 2048 / mib); double t0 = now();
            for (int i = 0; i < reps; ++i) CK(hipMemcpyAsync(d, src, n, hipMemcpyHostToDevice, s));
            CK(hipStreamSynchronize(s));
            std::printf("H2D %s %5zu MiB: %.1f GB/s\n", pinned ? "pinned  " : "pageable", mib, (double) n * reps / 1e9 / (now() - t0));
        }
        hipFree(d); hipHostFree(hp); _aligned_free(hg); hipStreamDestroy(s);
    }
}

// ver: which amdhip64 this process really loaded (path, file version), the HIP runtime / driver versions
static void ver() {
    int rv = 0, dv = 0; hipRuntimeGetVersion(&rv); hipDriverGetVersion(&dv);
    std::printf("hipRuntimeGetVersion %d, hipDriverGetVersion %d\n", rv, dv);
    HMODULE h = GetModuleHandleA("amdhip64_7.dll");
    if (!h) h = GetModuleHandleA("amdhip64.dll");
    char path[MAX_PATH] = {0};
    if (h) GetModuleFileNameA(h, path, MAX_PATH);
    std::printf("loaded amdhip64: %s\n", path);
    DWORD hd = 0, sz = path[0] ? GetFileVersionInfoSizeA(path, &hd) : 0;
    if (sz) {
        std::vector<char> b(sz);
        VS_FIXEDFILEINFO* fi = nullptr; UINT n = 0;
        if (GetFileVersionInfoA(path, 0, sz, b.data()) && VerQueryValueA(b.data(), "\\", (void**) &fi, &n) && fi)
            std::printf("file version %u.%u.%u.%u\n", HIWORD(fi->dwFileVersionMS), LOWORD(fi->dwFileVersionMS), HIWORD(fi->dwFileVersionLS), LOWORD(fi->dwFileVersionLS));
    }
    hipDeviceProp_t p{}; hipGetDeviceProperties(&p, 0);
    std::printf("device %s %s\n", p.name, p.gcnArchName);
    info();
}

__global__ void sum_kernel(const unsigned long long* __restrict__ p, size_t n, unsigned long long* out) {
    unsigned long long acc = 0;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) acc += p[i];
    atomicAdd(out, acc);
}
static void m1_line(const char* what, int seg, double secs, double avail0, size_t reg_total) {
    PROCESS_MEMORY_COUNTERS_EX pm{}; pm.cb = sizeof pm; GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*) &pm, sizeof pm);
    const double avail = ram_avail_gib();
    std::printf("M1 %s seg %d: %.2fs | registered %.1f GiB | avail %.1f GiB (fell %.1f = %.2fx the registered) | private %.1f GiB ws %.1f GiB faults %lu\n", what, seg, secs,
                reg_total / 1073741824.0, avail, avail0 - avail, reg_total ? (avail0 - avail) / (reg_total / 1073741824.0) : 0.0, pm.PrivateUsage / 1073741824.0,
                pm.WorkingSetSize / 1073741824.0, pm.PageFaultCount);
    dxgi_line("   ");
    std::fflush(stdout);
}
// m1 <segment MiB> <total GiB> <mapped|coarse|plain>: reserve total, then per segment commit + touch (a known pattern), register (flags per mode),
// device pointer, a checksum kernel that reads every page and compares with the pattern's sum; every step logs DXGI, RAM, commit, faults.
// 'coarse' = hipHostRegisterMapped | hipExtHostRegisterCoarseGrained; 'plain' = portable only (no map).  A fresh process per arm, a watchdog outside.
static void m1(int seg_mib, int total_gib, const std::string& mode) {
    ver();
    const size_t S = (size_t) seg_mib << 20, T = (size_t) total_gib << 30;
    const int nseg = (int) (T / S);
    uint8_t* base = (uint8_t*) VirtualAlloc(nullptr, T, MEM_RESERVE, PAGE_READWRITE);
    if (!base) { std::printf("reserve failed\n"); return; }
    unsigned flags = hipHostRegisterPortable;
    if (mode == "mapped") flags |= hipHostRegisterMapped;
    if (mode == "coarse") flags |= hipHostRegisterMapped | hipExtHostRegisterCoarseGrained;
    unsigned long long* dsum = nullptr; CK(hipMalloc(&dsum, 8));
    const double avail0 = ram_avail_gib();
    m1_line("start", 0, 0, avail0, 0);
    size_t reg = 0;
    for (int i = 0; i < nseg; ++i) {
        uint8_t* p = base + S * (size_t) i;
        double t0 = now();
        if (!VirtualAlloc(p, S, MEM_COMMIT, PAGE_READWRITE)) { std::printf("commit failed at segment %d\n", i); break; }
        unsigned long long* w = (unsigned long long*) p; const size_t nw = S / 8;
        unsigned long long expect = 0;
        for (size_t k = 0; k < nw; ++k) { w[k] = (unsigned long long) i * 1000003ull + k; expect += w[k]; }
        const double tt = now() - t0;
        t0 = now();
        hipError_t e = hipHostRegister(p, S, flags);
        void* dp = p;
        if (e == hipSuccess && (flags & hipHostRegisterMapped)) { e = hipHostGetDevicePointer(&dp, p, 0); }
        const double tr = now() - t0;
        if (e != hipSuccess) { std::printf("M1 register segment %d FAILED: %s (after %.2fs)\n", i, hipGetErrorString(e), tr); (void) hipGetLastError(); m1_line("failed", i, tr, avail0, reg); break; }
        reg += S;
        unsigned long long zero = 0; CK(hipMemcpy(dsum, &zero, 8, hipMemcpyHostToDevice));
        t0 = now();
        sum_kernel<<<2048, 256>>>((const unsigned long long*) dp, nw, dsum);
        hipError_t ek = hipDeviceSynchronize();
        const double tk = now() - t0;
        unsigned long long got = 0; hipMemcpy(&got, dsum, 8, hipMemcpyDeviceToHost);
        std::printf("M1 segment %d: touch %.2fs register %.2fs read %.3fs (%.0f GB/s) checksum %s\n", i, tt, tr, tk, S / 1e9 / (std::max)(tk, 1e-9), ek != hipSuccess ? hipGetErrorString(ek) : (got == expect ? "OK" : "MISMATCH"));
        if (ek != hipSuccess) { m1_line("kernel fault", i, tk, avail0, reg); break; }
        if (i % (std::max)(1, nseg / 8) == 0 || i == nseg - 1) m1_line("step", i, tr, avail0, reg);
    }
    m1_line("end", nseg, 0, avail0, reg);
}

int main(int argc, char** argv) {
    const std::string m = argc > 1 ? argv[1] : "";
    if (m == "ver") { ver(); return 0; }
    if (m == "m1") { m1(argc > 2 ? std::atoi(argv[2]) : 1024, argc > 3 ? std::atoi(argv[3]) : 8, argc > 4 ? argv[4] : "mapped"); return 0; }
    if (m == "info") info();
    else if (m == "alloc") alloc(argc > 2 ? std::atoi(argv[2]) : 80);
    else if (m == "reg") reg(argc > 2 ? std::atoi(argv[2]) : 8, argc > 3 ? std::atoi(argv[3]) : 0);
    else if (m == "regchunk") regchunk(argc > 2 ? std::atoi(argv[2]) : 32, argc > 3 ? std::atoi(argv[3]) : 4);
    else if (m == "regsub") regsub(argc > 2 ? std::atoi(argv[2]) : 660, argc > 3 ? std::atoi(argv[3]) : 8, argc > 4 ? std::atoi(argv[4]) : 0);
    else if (m == "hostalloc") hostalloc(argc > 2 ? std::atoi(argv[2]) : 8);
    else if (m == "copy") copy();
    else std::printf("usage: umaprobe info|alloc N|reg GiB|hostalloc GiB|copy\n");
}
