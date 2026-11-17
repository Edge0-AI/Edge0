// src/core/expert_cache.cpp - R4's slot storage and residency table.  Read the header first.
#include "strata/core/expert_cache.hpp"

// #533's segmented cache uses CUDA's virtual memory management (cuMem*): not on HIP, neither the RDNA backend nor
// the gfx906 compat build (PR #638), which compiles this file as HIP without STRATA_USE_HIP
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906) || defined(STRATA_GPU_METAL)
#define STRATA_EC_NO_VMM 1
#endif

#include <thread>
#include <mutex>
#include <cuda_runtime.h>
#if !defined(STRATA_EC_NO_VMM)
#include <cuda.h>   // #533: the virtual memory management types (the functions come through the runtime's entry points)
#endif

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <cstdlib>
#include <utility>
#include <cstring>

// (not in a SYCL build: its CUDA names are a compatibility layer without the PCI ids NVML is matched by)
#if defined(_WIN32) && !defined(STRATA_EC_NO_VMM) && !defined(SYCL_LANGUAGE_VERSION)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#define STRATA_EC_NVML_FREE 1
#endif

namespace strata::core {

#if defined(STRATA_EC_NVML_FREE)
namespace {

// STRATA_VRAM_FREE=device (opt-in, Windows + NVIDIA).  Under WDDM cudaMemGetInfo answers for THIS process: memory
// another process holds on the card (the image encoder, a browser, another engine) is not in its free figure.
// Measured on an RTX 5090, driver 610.88: with 6 GiB written by another process, cudaMemGetInfo in a second process
// still read 30,994 MiB free where NVML read 24,348.  NVML (nvml.dll, installed with every driver) counts the whole
// card.  The library is loaded from the system directory only, at first use; false when it cannot answer.
bool nvml_device_free(size_t& out) {
    struct Memory { unsigned long long total, free, used; };
    typedef int (*InitFn)(void);
    typedef int (*ByBusFn)(const char*, void**);
    typedef int (*MemFn)(void*, Memory*);
    struct Lib {
        ByBusFn by_bus = nullptr;
        MemFn mem = nullptr;
        Lib() {
            HMODULE h = LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
            if (h == nullptr) return;
            const auto init = (InitFn) (void*) GetProcAddress(h, "nvmlInit_v2");
            const auto bus = (ByBusFn) (void*) GetProcAddress(h, "nvmlDeviceGetHandleByPciBusId_v2");
            const auto m = (MemFn) (void*) GetProcAddress(h, "nvmlDeviceGetMemoryInfo");
            if (init == nullptr || bus == nullptr || m == nullptr || init() != 0) return;
            by_bus = bus;
            mem = m;
        }
    };
    static const Lib lib;
    if (lib.mem == nullptr) return false;
    int dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return false;
    char bus[32];
    std::snprintf(bus, sizeof bus, "%08X:%02X:%02X.0", (unsigned) prop.pciDomainID, (unsigned) prop.pciBusID,
                  (unsigned) prop.pciDeviceID);
    void* handle = nullptr;
    Memory m{};
    if (lib.by_bus(bus, &handle) != 0 || lib.mem(handle, &m) != 0) return false;
    out = (size_t) m.free;
    return true;
}

}  // namespace
#endif

namespace {
// STRATA_EXPERT_NIBBLE: a write into a sealed cache (see ExpertCache::seal).  Says which call and stops: the slots hold
// 4-bit codes the GGUF bytes would overwrite, or a GGUF expert where a 4-bit one is expected - a wrong reply, not an error.
[[noreturn]] void sealed_write(const char* call, int32_t slot) {
    std::fprintf(stderr, "strata: STRATA_EXPERT_NIBBLE: ExpertCache::%s(slot %d) after the cache was sealed - a GGUF "
                         "blob would be written into the 4-bit expert slots: stopping\n", call, (int) slot);
    std::fflush(stderr);
    std::abort();
}
}  // namespace

// See the header.  NVIDIA's DGX Spark Porting Guide (section 5.5, "Memory reporting on UMA systems") recommends the same:
// not to rely on cudaMemGetInfo alone but to count the memory the OS can reclaim.  Swap is not counted here (unlike
// NVIDIA's reference snippet): an expert cache that pushes the system into swap would be far slower than a smaller one.
size_t device_free_bytes() {
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
#if defined(__linux__) && !defined(STRATA_HIP_GFX906)   // (the gfx906 compat layer has no cudaDevAttrIntegrated; that GPU is discrete)
    // per device: a box can mix an integrated GPU (an APU) with a discrete one, and the answer is the CURRENT device's
    static std::atomic<int> uma_cache[64];   // 0 unknown, 1 integrated (unified memory), 2 discrete
    bool unified_memory = false;
    {
        int dev = 0, v = 0;
        if (cudaGetDevice(&dev) == cudaSuccess && dev >= 0 && dev < 64) {
            int c = uma_cache[dev].load(std::memory_order_acquire);
            if (c == 0) {
                c = cudaDeviceGetAttribute(&v, cudaDevAttrIntegrated, dev) == cudaSuccess && v ? 1 : 2;
                uma_cache[dev].store(c, std::memory_order_release);
            }
            unified_memory = c == 1;
        }
    }
    if (unified_memory) {
        if (FILE* m = std::fopen("/proc/meminfo", "r")) {
            char line[256];
            unsigned long long kb = 0;
            while (std::fgets(line, sizeof line, m))
                if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) break;
            std::fclose(m);
            // STRATA_UMA_HEADROOM_GIB: a whole number of GiB, 0..1024; anything else keeps the default 6 (said once)
            static const long gib = [] {
                const char* h = std::getenv("STRATA_UMA_HEADROOM_GIB");
                if (h == nullptr) return 6L;
                char* end = nullptr;
                const long v = std::strtol(h, &end, 10);
                if (end != h && *end == '\0' && v >= 0 && v <= 1024) return v;
                std::fprintf(stderr, "strata: STRATA_UMA_HEADROOM_GIB=%s is not a whole number of GiB (0-1024): using 6\n", h);
                return 6L;
            }();
            const unsigned long long head = (unsigned long long) gib << 30;
            const unsigned long long avail = kb << 10;
            if (avail > head && avail - head > free_b) free_b = (size_t) (avail - head);
        }
    }
#endif
#if defined(STRATA_EC_NVML_FREE)
    static const bool device_wide = [] {
        const char* v = std::getenv("STRATA_VRAM_FREE");
        return v != nullptr && std::strcmp(v, "device") == 0;
    }();
    if (device_wide) {
        size_t nvml_b = 0;
        const bool got = nvml_device_free(nvml_b);
        static std::atomic<bool> said{false};
        if (!said.exchange(true)) {
            if (got)
                std::fprintf(stderr, "strata generate: STRATA_VRAM_FREE=device: free VRAM is read for the whole card "
                                     "(NVML: %lld MiB now; CUDA's figure for this process alone: %lld MiB)\n",
                             (long long) (nvml_b >> 20), (long long) (free_b >> 20));
            else
                std::fprintf(stderr, "strata generate: STRATA_VRAM_FREE=device: NVML did not answer (nvml.dll); free "
                                     "VRAM stays CUDA's figure for this process\n");
        }
        if (got) free_b = std::min(free_b, nvml_b);
    }
#endif
    return free_b;
}

bool read_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                         std::vector<std::pair<int32_t, int32_t>>& ranked, int64_t& slots, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "read_expert_profile: cannot open " + path;
        return false;
    }
    char magic[4] = {0, 0, 0, 0};
    uint32_t hdr[5] = {0, 0, 0, 0, 0};
    if (std::fread(magic, 1, 4, f) != 4 || std::fread(hdr, 4, 5, f) != 5) {
        std::fclose(f);
        err = "read_expert_profile: " + path + " is too short to hold a header";
        return false;
    }
    if (std::memcmp(magic, "STRP", 4) != 0) {
        std::fclose(f);
        err = "read_expert_profile: " + path + " does not start with STRP";
        return false;
    }
    const uint32_t version = hdr[0], nl = hdr[1], ne = hdr[2], want = hdr[3], n_ranked = hdr[4];
    // the version comes first: the fields after it are only known to be these in a version 1 file, so a later
    // format is refused by its number rather than read as if it were this one
    if (version != 1) {
        std::fclose(f);
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "read_expert_profile: %s is a version %u profile but this engine reads version 1 - it was "
                      "written by a different release's tools/make_profile.py or --expert-profile-save",
                      path.c_str(), version);
        err = buf;
        return false;
    }
    if ((int64_t) nl != n_layers || (int64_t) ne != n_expert) {
        std::fclose(f);
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "read_expert_profile: %s is %ux%u but this model is %lldx%lld - it is a profile for a "
                      "different artifact", path.c_str(), nl, ne, (long long) n_layers, (long long) n_expert);
        err = buf;
        return false;
    }
    if (n_ranked > want) {
        std::fclose(f);
        err = "read_expert_profile: the header claims more ranked pairs than slots";
        return false;
    }
    ranked.assign(n_ranked, {0, 0});
    std::vector<uint16_t> raw((size_t) n_ranked * 2);
    if (n_ranked > 0 && std::fread(raw.data(), 2, (size_t) n_ranked * 2, f) != (size_t) n_ranked * 2) {
        std::fclose(f);
        err = "read_expert_profile: the ranked list is truncated";
        return false;
    }
    std::fclose(f);
    for (uint32_t i = 0; i < n_ranked; ++i) {
        const int32_t l = (int32_t) raw[(size_t) i * 2], e = (int32_t) raw[(size_t) i * 2 + 1];
        if (l < 0 || l >= n_layers || e < 0 || e >= n_expert) {
            char buf[256];
            std::snprintf(buf, sizeof buf, "read_expert_profile: pair %u is (layer %d, expert %d), out of range",
                          i, l, e);
            err = buf;
            return false;
        }
        ranked[(size_t) i] = {l, e};
    }
    slots = (int64_t) want;
    return true;
}

std::vector<std::pair<int32_t, int32_t>> rank_learned_profile(int64_t n_layers, int64_t n_expert,
                                                              const std::vector<uint8_t>& resident,
                                                              const std::vector<double>& heat,
                                                              const std::vector<std::pair<int32_t, int32_t>>& prior) {
    const size_t n = (size_t) (n_layers * n_expert);
    std::vector<int64_t> prior_rank(n, INT64_MAX);
    for (size_t r = 0; r < prior.size(); ++r) {
        const auto [l, e] = prior[r];
        if (l >= 0 && l < n_layers && e >= 0 && e < n_expert) {
            int64_t& pr = prior_rank[(size_t) (l * n_expert + e)];
            if (pr == INT64_MAX) pr = (int64_t) r;
        }
    }
    std::vector<int64_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = (int64_t) i;
    auto res = [&](int64_t i) { return (size_t) i < resident.size() && resident[(size_t) i] != 0; };
    auto ht = [&](int64_t i) { return (size_t) i < heat.size() ? heat[(size_t) i] : 0.0; };
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
        if (res(a) != res(b)) return res(a);
        if (ht(a) != ht(b)) return ht(a) > ht(b);
        if (prior_rank[(size_t) a] != prior_rank[(size_t) b]) return prior_rank[(size_t) a] < prior_rank[(size_t) b];
        return a < b;
    });
    std::vector<std::pair<int32_t, int32_t>> ranked(n);
    for (size_t r = 0; r < n; ++r)
        ranked[r] = {(int32_t) (order[r] / n_expert), (int32_t) (order[r] % n_expert)};
    return ranked;
}

bool write_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                          const std::vector<std::pair<int32_t, int32_t>>& ranked, std::string& err) {
    if (n_layers <= 0 || n_expert <= 0 || n_layers > 65535 || n_expert > 65535) {
        err = "write_expert_profile: the model's layout does not fit the format";
        return false;
    }
    std::vector<int32_t> table((size_t) (n_layers * n_expert), -1);
    std::vector<uint16_t> pairs;
    pairs.reserve(ranked.size() * 2);
    for (size_t r = 0; r < ranked.size(); ++r) {
        const auto [l, e] = ranked[r];
        if (l < 0 || l >= n_layers || e < 0 || e >= n_expert) {
            err = "write_expert_profile: a ranked pair is out of range";
            return false;
        }
        table[(size_t) (l * n_expert + e)] = (int32_t) r;
        pairs.push_back((uint16_t) l);
        pairs.push_back((uint16_t) e);
    }
    const uint32_t hdr[5] = {1u, (uint32_t) n_layers, (uint32_t) n_expert, (uint32_t) ranked.size(),
                             (uint32_t) ranked.size()};
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) {
        err = "write_expert_profile: cannot create " + tmp;
        return false;
    }
    // the format is little-endian (make_profile.py's "<"): so is every machine this engine runs on
    bool ok = std::fwrite("STRP", 1, 4, f) == 4 && std::fwrite(hdr, 4, 5, f) == 5 &&
              (pairs.empty() || std::fwrite(pairs.data(), 2, pairs.size(), f) == pairs.size()) &&
              std::fwrite(table.data(), 4, table.size(), f) == table.size();
    ok = (std::fclose(f) == 0) && ok;
    std::error_code ec;
    if (ok) std::filesystem::rename(tmp, path, ec);   // replaces an existing file (MoveFileEx / rename(2))
    if (!ok || ec) {
        std::filesystem::remove(tmp, ec);
        err = "write_expert_profile: cannot write " + path;
        return false;
    }
    return true;
}

ExpertCache::~ExpertCache() { close(); }

// ---- #533: the segmented arena (--vram-elastic).  The driver API's virtual memory functions, looked up through the
// runtime (no link against the driver library): one address range for the whole arena, backed by physical segments,
// and the tail's segments unmapped / mapped again later.  Nothing here runs unless a segment size was set.
#if !defined(STRATA_EC_NO_VMM)
namespace {
struct Vmm {
    CUresult (CUDAAPI* device_get)(CUdevice*, int) = nullptr;
    CUresult (CUDAAPI* attribute)(int*, CUdevice_attribute, CUdevice) = nullptr;
    CUresult (CUDAAPI* granularity)(size_t*, const CUmemAllocationProp*, CUmemAllocationGranularity_flags) = nullptr;
    CUresult (CUDAAPI* reserve)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long) = nullptr;
    CUresult (CUDAAPI* address_free)(CUdeviceptr, size_t) = nullptr;
    CUresult (CUDAAPI* create)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*,
                               unsigned long long) = nullptr;
    CUresult (CUDAAPI* release)(CUmemGenericAllocationHandle) = nullptr;
    CUresult (CUDAAPI* map)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long) = nullptr;
    CUresult (CUDAAPI* unmap)(CUdeviceptr, size_t) = nullptr;
    CUresult (CUDAAPI* set_access)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t) = nullptr;
    bool ok = false;
};

template <class F> bool entry(const char* name, F& f) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q{};
#if CUDART_VERSION >= 12050
    const cudaError_t e = cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &q);
#else
    const cudaError_t e = cudaGetDriverEntryPoint(name, &p, cudaEnableDefault, &q);
#endif
    if (e != cudaSuccess || q != cudaDriverEntryPointSuccess || p == nullptr) {
        (void) cudaGetLastError();
        return false;
    }
    f = reinterpret_cast<F>(p);
    return true;
}

const Vmm& vmm() {
    static const Vmm v = [] {
        Vmm x;
        x.ok = entry("cuDeviceGet", x.device_get) && entry("cuDeviceGetAttribute", x.attribute) &&
               entry("cuMemGetAllocationGranularity", x.granularity) && entry("cuMemAddressReserve", x.reserve) &&
               entry("cuMemAddressFree", x.address_free) && entry("cuMemCreate", x.create) &&
               entry("cuMemRelease", x.release) && entry("cuMemMap", x.map) && entry("cuMemUnmap", x.unmap) &&
               entry("cuMemSetAccess", x.set_access);
        return x;
    }();
    return v;
}

CUmemAllocationProp device_prop(int dev) {
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;
    return prop;
}

// one segment: a physical allocation mapped at `va`, readable and writable by this device
bool map_segment(const Vmm& v, int dev, CUdeviceptr va, size_t bytes, unsigned long long& handle) {
    const CUmemAllocationProp prop = device_prop(dev);
    CUmemGenericAllocationHandle h = 0;
    if (v.create(&h, bytes, &prop, 0) != CUDA_SUCCESS) return false;
    if (v.map(va, bytes, 0, h, 0) != CUDA_SUCCESS) {
        v.release(h);
        return false;
    }
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = dev;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (v.set_access(va, bytes, &access, 1) != CUDA_SUCCESS) {
        v.unmap(va, bytes);
        v.release(h);
        return false;
    }
    handle = (unsigned long long) h;
    return true;
}
}  // namespace
#endif

bool ExpertCache::open_segmented(uint64_t want, std::string& err) {
#if defined(STRATA_EC_NO_VMM)
    (void) want;
    err = "ExpertCache: --vram-elastic (a segmented expert cache) is CUDA-only for now";
    return false;
#else
    const Vmm& v = vmm();
    int dev = 0, supported = 0;
    CUdevice cu = 0;
    if (!v.ok || cudaGetDevice(&dev) != cudaSuccess || v.device_get(&cu, dev) != CUDA_SUCCESS ||
        v.attribute(&supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, cu) != CUDA_SUCCESS ||
        !supported) {
        err = "ExpertCache: --vram-elastic needs the driver's virtual memory management, which this GPU or driver "
              "does not offer";
        return false;
    }
    const CUmemAllocationProp prop = device_prop(dev);
    size_t gran = 0;
    if (v.granularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) != CUDA_SUCCESS || gran == 0) {
        err = "ExpertCache: cannot read the driver's allocation granularity";
        return false;
    }
    const uint64_t g = (uint64_t) gran;
    const uint64_t total = (want + g - 1) / g * g;
    seg_ = (int64_t) (((uint64_t) seg_req_ + g - 1) / g * g);
    CUdeviceptr va = 0;
    if (v.reserve(&va, (size_t) total, 0, 0, 0) != CUDA_SUCCESS) {
        err = "ExpertCache: cannot reserve the address range of the segmented expert cache";
        return false;
    }
    base_ = reinterpret_cast<uint8_t*>(va);
    reserved_ = total;
    for (uint64_t at = 0; at < total; at += (uint64_t) seg_) {
        segs_.push_back(0);
        seg_size_.push_back((int64_t) std::min<uint64_t>((uint64_t) seg_, total - at));
    }
    for (size_t i = 0; i < segs_.size(); ++i) {
        if (!map_segment(v, dev, va + (CUdeviceptr) ((uint64_t) i * (uint64_t) seg_), (size_t) seg_size_[i],
                         segs_[i])) {
            char buf[200];
            std::snprintf(buf, sizeof buf, "ExpertCache: cudaMalloc failed: segment %zu of %zu (%.2f GiB) of the "
                          "segmented cache could not be allocated", i + 1, segs_.size(),
                          (double) seg_size_[i] / 1073741824.0);
            err = buf;   // "cudaMalloc failed": the auto cache's smaller-retry path reads it as an allocation failure
            release_segmented();
            return false;
        }
        mapped_segs_ = (int64_t) i + 1;
    }
    return true;
#endif
}

void ExpertCache::release_segmented() {
#if !defined(STRATA_EC_NO_VMM)
    const Vmm& v = vmm();
    if (base_ != nullptr) cudaDeviceSynchronize();
    const CUdeviceptr va = reinterpret_cast<CUdeviceptr>(base_);
    for (size_t i = 0; i < segs_.size(); ++i)
        if (segs_[i] != 0) {
            v.unmap(va + (CUdeviceptr) ((uint64_t) i * (uint64_t) seg_), (size_t) seg_size_[i]);
            v.release((CUmemGenericAllocationHandle) segs_[i]);
        }
    if (base_ != nullptr && reserved_ > 0) v.address_free(va, (size_t) reserved_);
#endif
    segs_.clear();
    seg_size_.clear();
    mapped_segs_ = 0;
    reserved_ = 0;
    base_ = nullptr;
}

int64_t ExpertCache::mapped_bytes() const {
    if (segs_.empty()) return base_ != nullptr ? full_bytes() : 0;
    int64_t b = 0;
    for (int64_t i = 0; i < mapped_segs_; ++i) b += seg_size_[(size_t) i];
    return b;
}

int64_t ExpertCache::slots_within(int64_t bytes) const {
    if (bytes >= full_bytes()) return slots_;
    if (bytes <= 0) return 0;
    if (off_.empty()) return blob_ > 0 ? bytes / blob_ : 0;
    // off_[i + 1] is slot i's end: count the slots whose end is at most `bytes`
    return (int64_t) (std::upper_bound(off_.begin() + 1, off_.end(), (uint64_t) bytes) - (off_.begin() + 1));
}

bool ExpertCache::shrink(int64_t keep_bytes, std::string& err) {
    if (segs_.empty()) {
        err = "the expert cache is not segmented (the engine needs --vram-elastic)";
        return false;
    }
#if defined(STRATA_EC_NO_VMM)
    (void) keep_bytes;
    return false;
#else
    const Vmm& v = vmm();
    int64_t keep = 0, at = 0;   // the segments [0, keep) hold the first keep_bytes
    while (keep < (int64_t) segs_.size() && at < keep_bytes) at += seg_size_[(size_t) keep++];
    if (keep >= mapped_segs_) return true;
    if (cudaDeviceSynchronize() != cudaSuccess) {
        err = std::string("the device failed before the cache shrank: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    const CUdeviceptr va = reinterpret_cast<CUdeviceptr>(base_);
    for (int64_t i = mapped_segs_ - 1; i >= keep; --i) {
        const CUdeviceptr p = va + (CUdeviceptr) ((uint64_t) i * (uint64_t) seg_);
        if (v.unmap(p, (size_t) seg_size_[(size_t) i]) != CUDA_SUCCESS ||
            v.release((CUmemGenericAllocationHandle) segs_[(size_t) i]) != CUDA_SUCCESS) {
            err = "the driver refused to release an expert-cache segment";
            live_slots_ = slots_within(mapped_bytes());
            return false;
        }
        segs_[(size_t) i] = 0;
        mapped_segs_ = i;
    }
    live_slots_ = slots_within(mapped_bytes());
    return true;
#endif
}

bool ExpertCache::grow(int64_t want_bytes, std::string& err) {
    if (segs_.empty()) {
        err = "the expert cache is not segmented (the engine needs --vram-elastic)";
        return false;
    }
#if defined(STRATA_EC_NO_VMM)
    (void) want_bytes;
    return false;
#else
    const Vmm& v = vmm();
    int dev = 0;
    cudaGetDevice(&dev);
    const CUdeviceptr va = reinterpret_cast<CUdeviceptr>(base_);
    int64_t at = mapped_bytes();
    while (mapped_segs_ < (int64_t) segs_.size() && at + seg_size_[(size_t) mapped_segs_] <= want_bytes) {
        const size_t i = (size_t) mapped_segs_;
        if (!map_segment(v, dev, va + (CUdeviceptr) ((uint64_t) i * (uint64_t) seg_), (size_t) seg_size_[i],
                         segs_[i])) {
            (void) cudaGetLastError();
            err = "the driver has no VRAM for another expert-cache segment";
            live_slots_ = slots_within(mapped_bytes());
            return false;
        }
        at += seg_size_[i];
        ++mapped_segs_;
    }
    live_slots_ = slots_within(mapped_bytes());
    return true;
#endif
}

#if defined(STRATA_USE_HIP)
bool ExpertCache::ensure_blocking_staging(std::size_t bytes, std::string& err) {
    if (bytes <= blocking_staging_bytes_) return true;
    void* next = nullptr;
    const cudaError_t status = cudaHostAlloc(&next, bytes, cudaHostAllocDefault);
    if (status != cudaSuccess) {
        err = std::string("ExpertCache: HIP blocking staging allocation: ") + cudaGetErrorString(status);
        return false;
    }
    if (blocking_staging_) (void) cudaFreeHost(blocking_staging_);
    blocking_staging_ = static_cast<uint8_t*>(next);
    blocking_staging_bytes_ = bytes;
    return true;
}
#endif

namespace {
bool g_cache_vmm = false;   // set_vmm
}  // namespace

void ExpertCache::set_vmm(bool enabled) { g_cache_vmm = enabled; }

bool ExpertCache::open(int64_t n_slots, int64_t n_layers, int64_t n_expert, int64_t blob_bytes,
                       std::string& err) {
    close();
    if (n_slots <= 0) {
        err = "ExpertCache: n_slots must be positive";
        return false;
    }
    if (n_layers <= 0 || n_expert <= 0 || blob_bytes <= 0) {
        err = "ExpertCache: n_layers, n_expert and blob_bytes must all be positive";
        return false;
    }

    const uint64_t want = (uint64_t) n_slots * (uint64_t) blob_bytes;

    // ---- **THE ALLOCATION IS CHECKED AGAINST THE CARD, NOT AGAINST THE REQUEST.**
    //
    // `cudaMalloc` failing is the easy case. The one that matters is a machine where the weights already own
    // most of VRAM: the cache then takes what is left and `slots()` would report the number ASKED FOR while
    // `device_slot()` walks off the end. So the free-VRAM figure is read and compared BEFORE the allocation,
    // and the two numbers are named in the refusal.
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
        free_b = device_free_bytes();   // unified memory: what the OS can give back counts (see the header)
        if ((uint64_t) free_b < want) {
            char buf[320];
            std::snprintf(buf, sizeof buf,
                          "ExpertCache: %lld slots x %lld B = %.2f GiB, but only %.2f GiB of VRAM is free "
                          "(%.2f GiB of %.2f GiB total). Lower --expert-cache.",
                          (long long) n_slots, (long long) blob_bytes, (double) want / 1073741824.0,
                          (double) free_b / 1073741824.0, (double) (total_b - free_b) / 1073741824.0,
                          (double) total_b / 1073741824.0);
            err = buf;
            return false;
        }
    }

    if (seg_req_ > 0) {   // #533: --vram-elastic: physical segments behind one address range (zeroed below)
        if (!open_segmented(want, err)) return false;
    } else if (g_cache_vmm && vmm_available()) {
        // the elastic K/V: every chunk mapped now; the K/V may later take some of them (and give them back)
        auto r = std::make_unique<VmmRange>();
        if (!r->reserve(want) || !r->map_range(0, r->chunks(), [] { return (VmmChunk) 0; })) {
            char buf[256];
            std::snprintf(buf, sizeof buf, "ExpertCache: mapping %.2f GiB of VRAM failed: out of memory",
                          (double) want / 1073741824.0);
            err = buf;
            return false;
        }
        base_ = r->base();
        vmm_ = std::move(r);
    } else if (cudaMalloc((void**) &base_, (size_t) want) != cudaSuccess) {
        base_ = nullptr;
        char buf[256];
        std::snprintf(buf, sizeof buf, "ExpertCache: cudaMalloc(%.2f GiB) failed: %s",
                      (double) want / 1073741824.0, cudaGetErrorString(cudaGetLastError()));
        err = buf;
        return false;
    }
    // Zeroed so a slot read before it is filled is a DETERMINISTIC wrong answer rather than whatever the
    // allocator handed back.  A stale block of a previous process's memory would still sum to finite floats.
    if (cudaMemset(base_, 0, (size_t) want) != cudaSuccess) {
        err = "ExpertCache: cudaMemset of the slot arena failed";
        close();
        return false;
    }

    residency_.assign((size_t) (n_layers * n_expert), kNotResident);
    slots_ = n_slots;
    live_slots_ = n_slots;
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    blob_ = blob_bytes;
#if defined(STRATA_USE_HIP)
    if (!ensure_blocking_staging((std::size_t) blob_, err)) {
        close();
        return false;
    }
#endif
    next_free_ = 0;
    fills_ = 0;
    admitted_ = 0;
    // R4.2g: each layer starts at the bottom of its own range.  Built here rather than lazily so `admit`
    // stays allocation-free on the token path.
    layer_next_.assign((size_t) (n_layers > 0 ? n_layers : 0), 0);
    for (int64_t l = 0; l < n_layers; ++l) {
        int64_t lo = 0, hi = 0;
        layer_slot_range(l, lo, hi);
        layer_next_[(size_t) l] = (int32_t) lo;
    }
    return true;
}

bool ExpertCache::open_sized(const std::vector<int64_t>& slot_bytes, int64_t n_layers, int64_t n_expert,
                             std::string& err) {
    if (slot_bytes.empty()) { err = "ExpertCache: no slots"; return false; }
    int64_t mx = 0;
    std::vector<uint64_t> off(slot_bytes.size() + 1, 0);
    for (size_t i = 0; i < slot_bytes.size(); ++i) {
        // 256-byte aligned slots, so every blob starts where the kernels' vector loads expect it (and the
        // STRATA_MMQ_DIRECT guard after each blob, 0 by default)
        off[i + 1] = off[i] + (uint64_t) slot_stride(slot_bytes[i], guard_req_);
        mx = slot_bytes[i] > mx ? slot_bytes[i] : mx;
    }
    // one allocation of the summed size, through the uniform path's checks: n "slots" of 1 byte
    if (!open((int64_t) off.back(), n_layers, n_expert, 1, err)) return false;
    slots_ = (int64_t) slot_bytes.size();
    live_slots_ = slots_;
    blob_ = mx;
    off_ = std::move(off);
    guard_ = guard_req_;
#if defined(STRATA_USE_HIP)
    if (!ensure_blocking_staging((std::size_t) blob_, err)) {
        close();
        return false;
    }
#endif
    // #369: each layer's cursor at the bottom of its own range, as open() seeds it - open() above ran on byte-sized
    // "slots", so its seeds are not slot indices
    layer_next_.assign((size_t) (n_layers > 0 ? n_layers : 0), 0);
    for (int64_t l = 0; l < n_layers; ++l) {
        int64_t lo = 0, hi = 0;
        layer_slot_range(l, lo, hi);
        layer_next_[(size_t) l] = (int32_t) lo;
    }
    return true;
}

void ExpertCache::close() {
#if defined(STRATA_USE_HIP)
    if (blocking_staging_) (void) cudaFreeHost(blocking_staging_);
    blocking_staging_ = nullptr;
    blocking_staging_bytes_ = 0;
#endif
    off_.clear();
    guard_ = 0;
    if (!segs_.empty()) {
        release_segmented();
    } else if (vmm_) {
        vmm_.reset();   // unmaps and frees every chunk it still holds
        base_ = nullptr;
    } else if (base_ != nullptr) {
        cudaFree(base_);
        base_ = nullptr;
    }
    residency_.clear();
    slots_ = 0;
    live_slots_ = 0;
    n_layers_ = 0;
    n_expert_ = 0;
    blob_ = 0;
    next_free_ = 0;
    fills_ = 0;
    admitted_ = 0;
    sealed_ = false;
    layer_next_.clear();
}

/// R4.2g.  Layer `l` owns `[l*q, (l+1)*q)` with `q = slots_ / n_layers_`; the LAST layer takes whatever is
/// left over, so the ranges always cover `0..slots_` exactly and no slot is orphaned by the division.
void ExpertCache::layer_slot_range(int64_t layer, int64_t& lo, int64_t& hi) const {
    lo = 0;
    hi = 0;
    if (n_layers_ <= 0 || slots_ <= 0 || layer < 0 || layer >= n_layers_) return;
    const int64_t q = slots_ / n_layers_;
    lo = layer * q;
    hi = (layer == n_layers_ - 1) ? slots_ : (layer + 1) * q;
}

int32_t ExpertCache::slot_of(int64_t layer, int64_t expert) const {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return kNotResident;
    return residency_[(size_t) (layer * n_expert_ + expert)];
}

int32_t ExpertCache::admit(int64_t layer, int64_t expert) {
    if (sealed_) return kNotResident;   // STRATA_EXPERT_NIBBLE: nothing is admitted (and so filled) after the startup fill
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return kNotResident;
    const size_t at = (size_t) (layer * n_expert_ + expert);
    if (residency_[at] != kNotResident) return residency_[at];
    // R4.2g: THE PER-LAYER PATH.  Same "no eviction" rule, but the ceiling is this layer's own range rather
    // than one counter shared by all 48 - which is what confined the measured hit rate to 2.97%.
    if (per_layer_) {
        if (layer_next_.empty()) return kNotResident;
        int64_t lo = 0, hi = 0;
        layer_slot_range(layer, lo, hi);
        if ((int64_t) layer_next_[(size_t) layer] >= hi) return kNotResident;   // this layer's quota is full
        residency_[at] = layer_next_[(size_t) layer]++;
        ++admitted_;
        return residency_[at];
    }
    if (next_free_ >= slots_) return kNotResident;   // full: no eviction, deliberately - see the header
    residency_[at] = (int32_t) next_free_;
    return (int32_t) next_free_++;
}

uint8_t* ExpertCache::device_slot(int32_t slot) {
    if (slot < 0 || slot >= slots_) return nullptr;
    if (!off_.empty()) return base_ + off_[(size_t) slot];
    return base_ + (size_t) slot * (size_t) blob_;
}

const uint8_t* ExpertCache::device_slot(int32_t slot) const {
    if (slot < 0 || slot >= slots_) return nullptr;
    if (!off_.empty()) return base_ + off_[(size_t) slot];
    return base_ + (size_t) slot * (size_t) blob_;
}

bool ExpertCache::fill_slot(int32_t slot, const uint8_t* host_blob, void* stream, std::string& err, int64_t bytes) {
    if (sealed_) sealed_write("fill_slot", slot);
    const size_t n = (size_t) (bytes > 0 && bytes <= blob_ ? bytes : blob_);
    uint8_t* dst = device_slot(slot);
    if (dst == nullptr) {
        err = "ExpertCache::fill_slot: slot " + std::to_string(slot) + " is outside 0.." +
              std::to_string(slots_ - 1);
        return false;
    }
    if (host_blob == nullptr) {
        err = "ExpertCache::fill_slot: the host blob is null";
        return false;
    }
    const cudaError_t e = cudaMemcpyAsync(dst, host_blob, n, cudaMemcpyHostToDevice,
                                          (cudaStream_t) stream);
    if (e != cudaSuccess) {
        err = std::string("ExpertCache::fill_slot: ") + cudaGetErrorString(e);
        return false;
    }
    ++fills_;
    return true;
}

bool ExpertCache::fill_slot_blocking(int32_t slot, const uint8_t* host_blob, std::string& err, int64_t bytes) {
    if (sealed_) sealed_write("fill_slot_blocking", slot);
    const size_t n = (size_t) (bytes > 0 && bytes <= blob_ ? bytes : blob_);
    uint8_t* dst = device_slot(slot);
    if (dst == nullptr) {
        err = "ExpertCache::fill_slot_blocking: slot outside the arena";
        return false;
    }
    if (host_blob == nullptr) {
        err = "ExpertCache::fill_slot_blocking: the host blob is null";
        return false;
    }
#if defined(STRATA_USE_HIP)
    // Bound HIP's pageable-source staging to one expert instead of repeatedly
    // registering regions of the mmap. The blocking copy completes before reuse.
    if (!blocking_staging_ || n > blocking_staging_bytes_) {
        err = "ExpertCache::fill_slot_blocking: HIP staging buffer is too small";
        return false;
    }
    std::memcpy(blocking_staging_, host_blob, n);
    const cudaError_t e = cudaMemcpy(dst, blocking_staging_, n, cudaMemcpyHostToDevice);
#else
    const cudaError_t e = cudaMemcpy(dst, host_blob, n, cudaMemcpyHostToDevice);
#endif
    if (e != cudaSuccess) {
        err = std::string("ExpertCache::fill_slot_blocking: ") + cudaGetErrorString(e);
        return false;
    }
    ++fills_;
    return true;
}

bool ExpertCache::fill_slot_queued(int32_t slot, const uint8_t* host_blob, std::string& err, int64_t bytes) {
    if (sealed_) sealed_write("fill_slot_queued", slot);
    const size_t n = (size_t) (bytes > 0 && bytes <= blob_ ? bytes : blob_);
    uint8_t* dst = device_slot(slot);
    if (dst == nullptr || host_blob == nullptr) {
        err = dst == nullptr ? "ExpertCache::fill_slot_queued: slot outside the arena"
                             : "ExpertCache::fill_slot_queued: the host blob is null";
        return false;
    }
    const cudaError_t e = cudaMemcpyAsync(dst, host_blob, n, cudaMemcpyHostToDevice, (cudaStream_t) 0);
    if (e != cudaSuccess) {
        err = std::string("ExpertCache::fill_slot_queued: ") + cudaGetErrorString(e);
        return false;
    }
    ++fills_;
    return true;
}

bool ExpertCache::fill_slot_queued_parts(int32_t slot, const BlobPart* parts, int n, std::string& err) {
    if (sealed_) sealed_write("fill_slot_queued_parts", slot);
    uint8_t* dst = device_slot(slot);
    if (dst == nullptr || parts == nullptr || n <= 0) {
        err = "ExpertCache::fill_slot_queued_parts: slot outside the arena or no pieces";
        return false;
    }
    uint64_t total = 0;
    for (int i = 0; i < n; ++i) total += parts[i].n;
    if (total > (uint64_t) blob_) {
        err = "ExpertCache::fill_slot_queued_parts: the pieces are larger than a slot";
        return false;
    }
    for (int i = 0; i < n; ++i) {
        const cudaError_t e = cudaMemcpyAsync(dst + parts[i].at, parts[i].p, (size_t) parts[i].n, cudaMemcpyHostToDevice,
                                              (cudaStream_t) 0);
        if (e != cudaSuccess) {
            err = std::string("ExpertCache::fill_slot_queued_parts: ") + cudaGetErrorString(e);
            return false;
        }
    }
    ++fills_;
    return true;
}

bool ExpertCache::sync_queued(std::string& err) {
    const cudaError_t e = cudaStreamSynchronize((cudaStream_t) 0);
    if (e != cudaSuccess) {
        err = std::string("ExpertCache::sync_queued: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

bool ExpertCache::verify_slot(int32_t slot, const uint8_t* host_blob, std::string& err, int64_t bytes) {
    const int64_t nb = bytes > 0 && bytes <= blob_ ? bytes : blob_;
    const uint8_t* src = device_slot(slot);
    if (src == nullptr) {
        err = "ExpertCache::verify_slot: slot outside the arena";
        return false;
    }
    // `cudaMemcpy` and not `cudaMemcpyAsync`: this is a startup check, and a check that can be read before it
    // has happened is not a check.  It also synchronises the fills queued before it, which is what makes the
    // comparison meaningful.
    std::vector<uint8_t> got((size_t) nb);
    const cudaError_t e = cudaMemcpy(got.data(), src, (size_t) nb, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        err = std::string("ExpertCache::verify_slot: ") + cudaGetErrorString(e);
        return false;
    }
    if (std::memcmp(got.data(), host_blob, (size_t) nb) != 0) {
        size_t first = 0;
        while (first < (size_t) nb && got[first] == host_blob[first]) ++first;
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "ExpertCache::verify_slot: slot %d differs from the arena at byte %llu (of %lld)",
                      (int) slot, (unsigned long long) first, (long long) blob_);
        err = buf;
        return false;
    }
    return true;
}

}  // namespace strata::core
