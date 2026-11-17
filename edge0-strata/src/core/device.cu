// src/core/device.cu - P2.S1: the CUDA side of the runtime core.
#include "strata/core/device.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

namespace strata::core {

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw CudaError(std::string(what) + ": " + cudaGetErrorString(e), (int) e);
    }
}

// A NaN pattern, not zero.  Zeros read from uninitialised memory are indistinguishable from real zeros in a
// dequantized weight or a masked attention score, which is exactly the kind of wrong-but-plausible value the
// Phase 1 harnesses kept catching.
#if !defined(STRATA_GPU_METAL)
__global__ void poison_kernel(float* p, uint64_t n_floats) {
    const uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_floats) p[i] = __int_as_float(0x7fc00000);
}
#endif

#if defined(STRATA_USE_HIP)
#if !defined(STRATA_HIP_ARCHS)
#error "STRATA_HIP_ARCHS (the compiled HIP architectures) is set by cmake/hip_backend.cmake"
#endif
// "gfx1201:sramecc-:xnack-" -> "gfx1201"
std::string base_arch(const char* gcn_arch_name) {
    std::string arch(gcn_arch_name);
    const size_t colon = arch.find(':');
    if (colon != std::string::npos) arch.resize(colon);
    return arch;
}

bool compiled_for(const std::string& arch) {
    const std::string list = STRATA_HIP_ARCHS;
    size_t a = 0;
    while (a <= list.size()) {
        size_t b = list.find(',', a);
        if (b == std::string::npos) b = list.size();
        if (!arch.empty() && list.compare(a, b - a, arch) == 0 && b - a == arch.size()) return true;
        a = b + 1;
    }
    return false;
}

std::string arch_problem(const cudaDeviceProp& p, int ordinal) {
    const std::string arch = base_arch(p.gcnArchName);
    const std::string card = "GPU " + std::to_string(ordinal) + " (" + p.name + ", " + arch + ")";
    if (!compiled_for(arch)) {
        return card + " is not an architecture this Strata engine was compiled for (" + STRATA_HIP_ARCHS +
               "); compile it for this card (./setup.sh --backend hip, or -DCMAKE_HIP_ARCHITECTURES=" + arch +
               ", docs/AMD_HIP.md) or choose another GPU with HIP_VISIBLE_DEVICES";
    }
    if (p.warpSize != 32) {
        return card + " runs wave" + std::to_string(p.warpSize) + "; Strata's HIP kernels need wave32";
    }
    return "";
}
#endif

}  // namespace

const char* compiled_gpu_archs() {
#if defined(STRATA_USE_HIP)
    return STRATA_HIP_ARCHS;
#else
    return "";
#endif
}

int device_count() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {   // HIP without a usable device reports an error, not 0
        cudaGetLastError();
        return 0;
    }
    return count < 0 ? 0 : count;
}

// STRATA_PRIMARY_DEVICE: see the declaration in device.hpp.  0 = upstream's behavior (the runtime's first
// visible device).  Set once, before any context exists, by generate.cpp's STRATA_PRIMARY_DEVICE handling.
static int g_primary_device = 0;

void set_primary_device(int ordinal) { g_primary_device = ordinal; }
int primary_device() { return g_primary_device; }

bool device_summary(int ordinal, std::string& name, std::string& detail) {
    cudaDeviceProp p{};
    if (ordinal < 0 || ordinal >= device_count() || cudaGetDeviceProperties(&p, ordinal) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    char buf[160];
#if defined(STRATA_USE_HIP)
    std::snprintf(buf, sizeof(buf), "arch %s, %.1f GiB, wave%d", base_arch(p.gcnArchName).c_str(),
                  (double) p.totalGlobalMem / (1024.0 * 1024 * 1024), p.warpSize);
#else
    std::snprintf(buf, sizeof(buf), "compute capability %d.%d, %.1f GiB", p.major, p.minor,
                  (double) p.totalGlobalMem / (1024.0 * 1024 * 1024));
#endif
    name = p.name;
    detail = buf;
    return true;
}

}  // namespace strata::core

#if defined(STRATA_USE_HIP) && defined(_WIN32)
// The free VRAM figure on Windows HIP (include/strata/hip_compat/cuda_runtime.h maps cudaMemGetInfo here).
//
// hipMemGetInfo (ROCclr's PAL backend, Device::globalFreeMemory) is the card's size minus this process's own
// allocations: it asks Windows for this process's usage only, never for what the desktop and other programs hold.
// So on a card that also drives the desktop the engine counted ~930 MiB that was not there, the expert cache filled
// the card, and WDDM moved memory out to system RAM: decode at 30 tok/s instead of 41 (RX 6800, HIP SDK 7.2).
// Windows itself gives each process a video memory budget (DXGI QueryVideoMemoryInfo) that does account for the
// others.  Measured on that card: the budget sits 0.8 GiB below the card's size while this process is small and
// 1.8 GiB below once it holds 11 GiB; allocations past it still succeed, and are what Windows moves out.  The
// free figure here is hipMemGetInfo's lowered by what the budget withholds (the card's size minus the budget).
// Subtracting from HIP's own figure, rather than taking the budget minus DXGI's usage, keeps memory the HIP runtime
// has freed and holds in its cache counted as free, as hipMemGetInfo counts it.  dxgi.dll is loaded at run time, so
// nothing new is linked; when it or the card's adapter cannot be found, hipMemGetInfo's figure stands.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_4.h>
#include <atomic>
#include <cstdlib>
#include <mutex>

namespace strata::hip_compat {
namespace {
IDXGIAdapter3* budget_adapter(int device) {
    constexpr int kMaxDevices = 16;
    static std::mutex mu;
    static IDXGIAdapter3* adapters[kMaxDevices] = {};
    static bool tried[kMaxDevices] = {};
    if (device < 0 || device >= kMaxDevices) return nullptr;
    std::lock_guard<std::mutex> lock(mu);
    if (tried[device]) return adapters[device];
    tried[device] = true;
    hipDeviceProp_t p{};
    if (hipGetDeviceProperties(&p, device) != hipSuccess) {
        (void) hipGetLastError();
        return nullptr;
    }
    LUID luid{};
    std::memcpy(&luid, p.luid, sizeof luid);
    using CreateFactory = HRESULT(WINAPI*)(REFIID, void**);
    static HMODULE dxgi = LoadLibraryW(L"dxgi.dll");   // kept for the process' lifetime, like the adapters
    const auto create = dxgi ? (CreateFactory) (void*) GetProcAddress(dxgi, "CreateDXGIFactory1") : nullptr;
    IDXGIFactory4* factory = nullptr;
    if (create == nullptr || FAILED(create(__uuidof(IDXGIFactory4), (void**) &factory))) return nullptr;
    IDXGIAdapter3* adapter = nullptr;
    if (FAILED(factory->EnumAdapterByLuid(luid, __uuidof(IDXGIAdapter3), (void**) &adapter))) adapter = nullptr;
    factory->Release();
    adapters[device] = adapter;
    return adapter;
}
}  // namespace

// #873 / #654: on some Windows setups the first hipMemGetInfo of a process fails with hipErrorInvalidValue because the ROCm
// runtime has not finished initialising against the Adrenalin WDDM driver, and the first kernel launch after that
// crashes (0xC0000005 in amdhip64_7.dll).  A hipFree(0) before the first query initialises the context, and one
// failed query gets one retry after 250 ms.  Both happen once per process, before the DXGI budget below; any other
// error, and a failure of the retry, are returned as they are (a real failure is never retried).
static hipError_t first_mem_get_info(size_t* free_bytes, size_t* total_bytes) {
    static std::atomic<bool> warmed{false}, retried{false};
    if (!warmed.exchange(true)) (void) hipFree(nullptr);
    hipError_t e = hipMemGetInfo(free_bytes, total_bytes);
    if (e == hipErrorInvalidValue && !retried.exchange(true)) {
        (void) hipGetLastError();
        std::fprintf(stderr, "strata: hipMemGetInfo failed (%s); retrying once after 250 ms\n", hipGetErrorString(e));
        Sleep(250);
        e = hipMemGetInfo(free_bytes, total_bytes);
        std::fprintf(stderr, "strata: hipMemGetInfo retry: %s\n", e == hipSuccess ? "ok" : hipGetErrorString(e));
    }
    return e;
}

hipError_t mem_get_info(size_t* free_bytes, size_t* total_bytes) {
    if (free_bytes == nullptr || total_bytes == nullptr) return hipMemGetInfo(free_bytes, total_bytes);
    const hipError_t e = first_mem_get_info(free_bytes, total_bytes);
    static const bool off = [] {
        const char* v = std::getenv("STRATA_WDDM_BUDGET");
        return v != nullptr && std::atoi(v) == 0;
    }();
    if (e != hipSuccess || off || free_bytes == nullptr || total_bytes == nullptr) return e;
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) {
        (void) hipGetLastError();
        return e;
    }
    IDXGIAdapter3* adapter = budget_adapter(device);
    DXGI_QUERY_VIDEO_MEMORY_INFO local{};
    if (adapter == nullptr || FAILED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local)) ||
        local.Budget == 0 || local.Budget >= *total_bytes)
        return e;
    const size_t withheld = *total_bytes - (size_t) local.Budget;   // the desktop's and other programs' share
    static std::atomic<bool> said{false};
    if (!said.exchange(true)) {
        std::fprintf(stderr, "strata: Windows budgets %llu of this card's %llu MiB for this process; free VRAM is "
                             "counted within that (STRATA_WDDM_BUDGET=0: off)\n",
                     (unsigned long long) (local.Budget >> 20), (unsigned long long) (*total_bytes >> 20));
    }
    *free_bytes = *free_bytes > withheld ? *free_bytes - withheld : 0;
    return e;
}
}  // namespace strata::hip_compat
#endif

namespace strata::core {

std::string gpu_arch_problem(int ordinal) {
#if defined(STRATA_USE_HIP)
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || ordinal < 0 || ordinal >= count) {
        cudaGetLastError();
        return "";
    }
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, ordinal) != cudaSuccess) {
        cudaGetLastError();
        return "";
    }
    return arch_problem(p, ordinal);
#else
    (void) ordinal;
    return "";
#endif
}

std::string device_code_error() {
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906) || defined(STRATA_GPU_METAL)
    return "";   // gpu_arch_problem() checks the HIP architectures against STRATA_HIP_ARCHS, before this point
#else
    // every .cu of the engine is compiled for the same CMAKE_CUDA_ARCHITECTURES, so this kernel stands for all
    cudaFuncAttributes a{};
    const cudaError_t e = cudaFuncGetAttributes(&a, poison_kernel);
    if (e == cudaSuccess) return {};
    cudaGetLastError();
    return cudaGetErrorString(e);
#endif
}

DeviceInfo device_info(int ordinal) {
    int count = 0;
    check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count == 0) {
#if defined(STRATA_USE_HIP)
        throw CudaError(std::string("no HIP device is present; this engine was compiled for ") + STRATA_HIP_ARCHS, -1);
#else
        throw CudaError("no CUDA device is present; Strata needs an NVIDIA GPU (RTX 20 series or newer)", -1);
#endif
    }
    if (ordinal < 0 || ordinal >= count) {
        throw CudaError("device ordinal " + std::to_string(ordinal) + " is out of range (have " +
                            std::to_string(count) + ")",
                        -1);
    }
    DeviceInfo d;
    d.ordinal = ordinal;
    check(cudaSetDevice(ordinal), "cudaSetDevice");

    cudaDeviceProp p{};
    check(cudaGetDeviceProperties(&p, ordinal), "cudaGetDeviceProperties");
    d.name = p.name;
    d.cc_major = p.major;
    d.cc_minor = p.minor;
    d.multi_processor_count = p.multiProcessorCount;

    size_t free_b = 0, total_b = 0;
    check(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
    d.free_bytes = free_b;
    d.total_bytes = total_b;

    check(cudaDriverGetVersion(&d.driver_version), "cudaDriverGetVersion");
    check(cudaRuntimeGetVersion(&d.runtime_version), "cudaRuntimeGetVersion");

    // The engine supports compute capability 7.5 and newer (Turing: the QSA scorer's tf32 mma has a portable
    // fp32-FMA fallback below sm_80, the tensor-core prompt kernels refuse and fall back).  Compiling for a
    // supported arch is enforced by CMake; RUNNING on an older card is caught here, because a binary can be carried
    // to a machine with an older card and would otherwise silently take whatever path the driver chose.  The HIP
    // backend checks the card against the architectures the binary was compiled for (and wave32).
#if defined(STRATA_HIP_GFX906)
    // AMD gfx906 (the STRATA_HIP_GFX906 compat build, not STRATA_USE_HIP): the build targets one gfx arch
    // (CMAKE_HIP_ARCHITECTURES, wave64); a card of another arch fails at the first kernel launch with
    // hipErrorInvalidDeviceFunction.  gcnArchName says which it is.
    {
        std::string arch(p.gcnArchName);
        if (const size_t colon = arch.find(':'); colon != std::string::npos) arch.resize(colon);
        d.arch = arch;
    }
    if (p.gcnArchName[0] != 0) d.name += std::string(" (") + p.gcnArchName + ")";
#elif defined(STRATA_USE_HIP)
    d.arch = base_arch(p.gcnArchName);
    if (const std::string why = arch_problem(p, ordinal); !why.empty()) throw CudaError(why, -1);
#else
    // #236: the experimental build (-DSTRATA_EXPERIMENTAL_SM60=ON: Pascal sm_60, Volta sm_70) runs on the cards it
    // was built for - refusing them below 7.5 there made the flag useless; the release engine keeps 7.5
#if defined(STRATA_EXPERIMENTAL_SM60)
    constexpr int kMinCc = 60;
    const char* const kNeed = "6.0 or newer (this is the experimental Pascal / Volta build)";
#else
    constexpr int kMinCc = 75;
    const char* const kNeed = "7.5 or newer (RTX 20 / 30 / 40 / 50 series)";
#endif
    if (d.cc_major * 10 + d.cc_minor < kMinCc) {
        throw CudaError("device " + d.name + " reports compute capability " + std::to_string(d.cc_major) +
                            "." + std::to_string(d.cc_minor) + "; Strata needs compute capability " + kNeed,
                        -1);
    }
#endif
    return d;
}

DeviceArena::DeviceArena(uint64_t bytes, int ordinal, bool poison)
    : capacity_(bytes), ordinal_(ordinal), poison_(poison) {
    if (bytes == 0) throw CudaError("DeviceArena of 0 bytes", -1);
    check(cudaSetDevice(ordinal), "cudaSetDevice");
    // One allocation for the whole region.  cudaMalloc of a large block is the thing that can fail late, so it
    // happens once, here, before anything depends on it.
    check(cudaMalloc(&base_, (size_t) bytes), "cudaMalloc");
#if defined(STRATA_GPU_METAL)   // unified memory: the CPU writes the pattern through the allocation's CPU view
    if (poison_) {
        uint32_t* h = (uint32_t*) strata::gpu::host_view(base_);
        for (uint64_t i = 0; i < bytes / sizeof(float); ++i) h[i] = 0x7fc00000u;
    }
#else
    if (poison_) {
        const int threads = 256;
        const uint64_t n = bytes / sizeof(float);
        const uint64_t blocks = (n + threads - 1) / threads;
        // gridDim.x is 32-bit, so a large region needs a loop.  12 GB of floats is 3e9 elements = 1.2e7
        // blocks, which fits, but the loop keeps it correct for any size rather than for today's sizes.
        const uint64_t max_blocks = 0x7FFFFFFFull;
        for (uint64_t b = 0; b < blocks; b += max_blocks) {
            const uint64_t chunk = (blocks - b < max_blocks) ? (blocks - b) : max_blocks;
            poison_kernel<<<(unsigned) chunk, threads>>>((float*) base_ + b * threads, n - b * threads);
            check(cudaGetLastError(), "poison_kernel");
        }
        check(cudaDeviceSynchronize(), "poison sync");
    }
#endif
}

DeviceArena::~DeviceArena() {
    if (base_) cudaFree(base_);          // best effort: a destructor must not throw
}

void* DeviceArena::alloc(uint64_t bytes, uint64_t align) {
    if (bytes == 0) return nullptr;
    if (align == 0 || (align & (align - 1)) != 0) {
        throw CudaError("DeviceArena::alloc alignment must be a power of two", -1);
    }
    const uint64_t start = (used_ + align - 1) & ~(align - 1);
    if (start + bytes > capacity_) {
        char msg[256];
        std::snprintf(msg, sizeof(msg),
                      "DeviceArena out of memory: asked for %llu B at offset %llu (align %llu) in a %llu B "
                      "region - the plan from P1.S9 did not close",
                      (unsigned long long) bytes, (unsigned long long) start, (unsigned long long) align,
                      (unsigned long long) capacity_);
        throw CudaError(msg, -1);
    }
    used_ = start + bytes;
    return (char*) base_ + start;
}

}  // namespace strata::core
