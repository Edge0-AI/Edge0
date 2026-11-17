// include/strata/program/gpu_report.hpp - #3797: what to tell the user when a GPU allocation at start fails.
//
// "cudaMalloc(818947840) for the weight arena failed: 320 MiB of 21998 MiB free" says that the card is full, not by
// whom. The usual answer is another program (or an engine that is still exiting), so the report names, per card, the
// memory that is free and the largest other processes that hold some. The text is a warning added to the message that
// was already there; nothing about the failure changes.
//
// `format_gpu_memory_report` is a pure function (tests/program/gpu_report_test.cpp). `query_other_processes` asks
// the NVIDIA driver library (NVML, loaded at run time: no link-time dependency, so a build without it, an AMD card,
// a missing driver library or a failed call just returns false and the report leaves the process list out). It
// never throws and never ends the process.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace strata::program::gpu_report {

struct Process {
    uint32_t pid = 0;
    std::string name;            ///< the program's file name; empty when the system would not say
    uint64_t used_mib = 0;       ///< 0 = the driver does not report it (Windows WDDM does not for most programs)
};

struct Card {
    int index = 0;               ///< the CUDA device number
    std::string name;            ///< "NVIDIA GeForce RTX 2080 Ti"; may be empty
    uint64_t free_mib = 0;
    uint64_t total_mib = 0;
    bool processes_known = false;       ///< the driver answered the process query (the list may still be empty)
    std::vector<Process> processes;     ///< every process that holds memory on this card, in any order
};

namespace detail {
inline std::string size_text(uint64_t mib) {
    char b[48];
    if (mib >= 10240) std::snprintf(b, sizeof b, "%.1f GiB", (double) mib / 1024.0);
    else std::snprintf(b, sizeof b, "%llu MiB", (unsigned long long) mib);
    return b;
}
inline std::string base_name(const std::string& path) {
    const size_t cut = path.find_last_of("/\\");
    return cut == std::string::npos ? path : path.substr(cut + 1);
}
}  // namespace detail

/// One line per card: the free and total memory and, when the driver listed them, the largest other processes
/// (`self_pid` and any process of 0 MiB that the driver does list are left out of "largest"; at most `max_procs`
/// are named). Ends with no newline. `cards` empty: an empty string.
inline std::string format_gpu_memory_report(const std::vector<Card>& cards, uint32_t self_pid, size_t max_procs = 3) {
    std::string out;
    for (const Card& c : cards) {
        if (!out.empty()) out += "\n";
        out += "strata generate:   CUDA" + std::to_string(c.index);
        if (!c.name.empty()) out += " (" + c.name + ")";
        out += ": " + std::to_string((unsigned long long) c.free_mib) + " MiB free of " +
               std::to_string((unsigned long long) c.total_mib) + " MiB";   // (as the engine's other memory lines)
        if (!c.processes_known) {
            out += "; the driver does not say which programs use it here";
            continue;
        }
        std::vector<Process> others;
        size_t unsized = 0;
        for (const Process& p : c.processes) {
            if (p.pid == self_pid) continue;
            if (p.used_mib == 0) { ++unsized; continue; }
            others.push_back(p);
        }
        std::sort(others.begin(), others.end(), [](const Process& a, const Process& b) {
            return a.used_mib != b.used_mib ? a.used_mib > b.used_mib : a.pid < b.pid;
        });
        if (others.empty() && unsized == 0) {
            out += "; the driver lists no other program using it";
            continue;
        }
        out += "; other programs on it: ";
        for (size_t i = 0; i < others.size() && i < max_procs; ++i) {
            if (i > 0) out += ", ";
            const std::string n = detail::base_name(others[i].name);
            out += (n.empty() ? "pid " + std::to_string(others[i].pid) : n + " (pid " + std::to_string(others[i].pid) + ")") + " " +
                   detail::size_text(others[i].used_mib);
        }
        if (others.size() > max_procs) out += ", and " + std::to_string(others.size() - max_procs) + " more";
        if (unsized > 0) out += std::string(others.empty() ? "" : "; ") + std::to_string(unsized) +
                                (unsized == 1 ? " program whose memory the driver does not report" : " programs whose memory the driver does not report");
    }
    return out;
}

/// "The split planned to keep N MiB free on this card, M MiB are free" for the low-VRAM warning: how far the engine's
/// own estimate was off. Empty when the card has at least what was planned (nothing to explain).
inline std::string planned_vs_free_text(int64_t planned_mib, int64_t free_mib) {
    if (planned_mib <= 0 || free_mib >= planned_mib) return {};
    return "the layer split planned to keep " + std::to_string((long long) planned_mib) + " MiB free here after loading; " +
           std::to_string((long long) std::max<int64_t>(free_mib, 0)) + " MiB are free, so " +
           std::to_string((long long) (planned_mib - std::max<int64_t>(free_mib, 0))) + " MiB went to something the plan did not count";
}

// ---------------------------------------------------------------------------------------------- the driver query
namespace detail {
struct NvmlProc {            // nvmlProcessInfo_v2_t / the layout the _v3 calls fill
    unsigned int pid;
    unsigned long long usedGpuMemory;
    unsigned int gpuInstanceId;
    unsigned int computeInstanceId;
};
constexpr unsigned long long kNvmlNotAvailable = ~0ull;
using NvmlInit = int (*)();
using NvmlHandleByPci = int (*)(const char*, void**);
using NvmlProcs = int (*)(void*, unsigned int*, NvmlProc*);
using NvmlProcName = int (*)(unsigned int, char*, unsigned int);

struct Nvml {
    void* lib = nullptr;
    NvmlHandleByPci by_pci = nullptr;
    NvmlProcs compute = nullptr;
    NvmlProcs graphics = nullptr;
    NvmlProcName name = nullptr;
    NvmlInit shutdown = nullptr;
};

inline void* open_lib() {
#if defined(_WIN32)
    return (void*) LoadLibraryA("nvml.dll");
#else
    for (const char* n : {"libnvidia-ml.so.1", "libnvidia-ml.so"})
        if (void* h = dlopen(n, RTLD_LAZY | RTLD_LOCAL)) return h;
    return nullptr;
#endif
}
inline void* sym(void* lib, const char* n) {
#if defined(_WIN32)
    return (void*) GetProcAddress((HMODULE) lib, n);
#else
    return dlsym(lib, n);
#endif
}
inline void close_lib(void* lib) {
#if defined(_WIN32)
    if (lib) FreeLibrary((HMODULE) lib);
#else
    if (lib) dlclose(lib);
#endif
}
}  // namespace detail

/// The processes that hold memory on the card with this PCI bus id ("0000:65:00.0", what cudaDeviceGetPCIBusId
/// returns). true: the driver answered (the list may be empty). false: it could not be asked. Best effort.
inline bool query_other_processes(const char* pci_bus_id, std::vector<Process>& out) {
    out.clear();
    if (pci_bus_id == nullptr || pci_bus_id[0] == '\0') return false;
    detail::Nvml n;
    n.lib = detail::open_lib();
    if (n.lib == nullptr) return false;
    bool ok = false;
    do {
        auto init = (detail::NvmlInit) detail::sym(n.lib, "nvmlInit_v2");
        n.shutdown = (detail::NvmlInit) detail::sym(n.lib, "nvmlShutdown");
        n.by_pci = (detail::NvmlHandleByPci) detail::sym(n.lib, "nvmlDeviceGetHandleByPciBusId_v2");
        n.name = (detail::NvmlProcName) detail::sym(n.lib, "nvmlSystemGetProcessName");
        // _v3 and _v2 take the 24-byte record above; the unversioned (v1) call has another record, so it is not used
        n.compute = (detail::NvmlProcs) detail::sym(n.lib, "nvmlDeviceGetComputeRunningProcesses_v3");
        if (!n.compute) n.compute = (detail::NvmlProcs) detail::sym(n.lib, "nvmlDeviceGetComputeRunningProcesses_v2");
        n.graphics = (detail::NvmlProcs) detail::sym(n.lib, "nvmlDeviceGetGraphicsRunningProcesses_v3");
        if (!n.graphics) n.graphics = (detail::NvmlProcs) detail::sym(n.lib, "nvmlDeviceGetGraphicsRunningProcesses_v2");
        if (!init || !n.by_pci || !n.compute || init() != 0) break;
        void* dev = nullptr;
        if (n.by_pci(pci_bus_id, &dev) != 0 || dev == nullptr) { if (n.shutdown) n.shutdown(); break; }
        for (detail::NvmlProcs call : {n.compute, n.graphics}) {
            if (!call) continue;
            std::vector<detail::NvmlProc> recs(128);
            unsigned int count = (unsigned int) recs.size();
            int rc = call(dev, &count, recs.data());
            if (rc == 7 && count > recs.size()) {            // NVML_ERROR_INSUFFICIENT_SIZE: `count` is the number
                recs.resize((size_t) count + 8);
                count = (unsigned int) recs.size();
                rc = call(dev, &count, recs.data());
            }
            if (rc != 0) continue;
            ok = true;
            for (unsigned int i = 0; i < count && i < recs.size(); ++i) {
                bool seen = false;
                for (Process& p : out)
                    if (p.pid == recs[i].pid) { seen = true; break; }
                if (seen) continue;
                Process p;
                p.pid = recs[i].pid;
                p.used_mib = recs[i].usedGpuMemory == detail::kNvmlNotAvailable ? 0 : (uint64_t) (recs[i].usedGpuMemory >> 20);
                if (n.name) {
                    char buf[260] = {};
                    if (n.name(p.pid, buf, (unsigned int) sizeof buf) == 0) p.name = buf;
                }
                out.push_back(std::move(p));
            }
        }
        if (n.shutdown) n.shutdown();
    } while (false);
    detail::close_lib(n.lib);
    return ok;
}

}  // namespace strata::program::gpu_report
