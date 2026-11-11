// src/core/device_main.cpp - `strata-device`: report the GPU, the plan, and exercise the arena.
//
// This is P2.S1's "startup prints the memory plan vs actual cudaMemGetInfo" bullet, on its own so it can run
// without the model.  It is also the run-time half of the sm_120 policy: CMake refuses to COMPILE for another
// architecture, and this refuses to RUN on one.
#include "strata/core/device.hpp"
#include "strata/plan/plan.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#if defined(STRATA_USE_HIP) && defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

static std::string human(uint64_t b) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.3f GiB (%llu B)", (double) b / (1024.0 * 1024 * 1024),
                  (unsigned long long) b);
    return buf;
}

int main(int argc, char** argv) {
    bool selftest = false;
    bool list_devices = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--selftest") == 0) selftest = true;
        else if (std::strcmp(argv[i], "--list-devices") == 0) list_devices = true;
        else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            std::printf("usage: strata-device [--selftest] [--list-devices]\n"
                        "  --list-devices  every GPU the runtime enumerates, numbered as HIP_VISIBLE_DEVICES /\n"
                        "                  CUDA_VISIBLE_DEVICES number them, and whether this binary can run it\n");
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return 2;
        }
    }

    // The runtime's numbering, which setup needs on Windows: there an integrated Radeon is HIP device 0 and pushes the
    // discrete card to 1, while setup finds the cards in the display-adapter order (#325).  No arch check here - the
    // cards this binary has no code for are part of the answer.  Format (setup.py's hip_devices parses it):
    //   device N: <name>
    //     arch gfx1201, 15.9 GiB, wave32          (CUDA: compute capability 12.0, 11.9 GiB)
    //     cannot run: <why>                       (only for a card this binary cannot run)
    if (list_devices) {
        const int count = strata::core::device_count();
        if (count == 0) std::printf("(no GPU device)\n");
        for (int ordinal = 0; ordinal < count; ++ordinal) {
            std::string name, detail;
            if (!strata::core::device_summary(ordinal, name, detail)) {
                std::printf("device %d: (the runtime cannot describe it)\n", ordinal);
                continue;
            }
            std::printf("device %d: %s\n  %s\n", ordinal, name.c_str(), detail.c_str());
            if (const std::string why = strata::core::gpu_arch_problem(ordinal); !why.empty())
                std::printf("  cannot run: %s\n", why.c_str());
        }
        return 0;
    }

    try {
        const strata::core::DeviceInfo d = strata::core::device_info(0);
        std::printf("device %d: %s\n", d.ordinal, d.name.c_str());
#if defined(STRATA_USE_HIP)
        std::printf("  HIP arch            %s wave32 (compiled for %s)\n", d.arch.c_str(),
                    strata::core::compiled_gpu_archs());
#else
        std::printf("  compute capability  %d.%d   (sm_%d%d)\n", d.cc_major, d.cc_minor, d.cc_major, d.cc_minor);
#endif
        std::printf("  multiprocessors     %d\n", d.multi_processor_count);
        std::printf("  VRAM total / free   %s / %s\n", human(d.total_bytes).c_str(), human(d.free_bytes).c_str());
        std::printf("  driver / runtime    %d / %d\n", d.driver_version, d.runtime_version);
#if defined(STRATA_USE_HIP) && defined(_WIN32)
        // #468 #461: which HIP runtime this process loaded - the one beside the exe, or an AMD driver's System32 copy
        if (HMODULE h = GetModuleHandleA("amdhip64_7.dll")) {
            char path[MAX_PATH] = {};
            if (GetModuleFileNameA(h, path, MAX_PATH) > 0) std::printf("  HIP runtime         %s\n", path);
        }
#endif

        // The planner's view against the card's.  A plan that does not fit in what is actually FREE is the
        // failure this print exists to make visible at startup rather than at token 4000.
        const auto plan = strata::plan::make_plan(20480, strata::plan::Geometry{}, strata::plan::Costs{});
        std::printf("\n%s", strata::plan::to_string(plan).c_str());
        std::printf("  card free           %s\n", human(d.free_bytes).c_str());
        std::printf("  plan + KV vs free   %s\n",
                    plan.vram_budget <= d.free_bytes ? "FITS" : "*** DOES NOT FIT ***");

        if (selftest) {
            // Exercise the arena for real: allocate, write from the host, read back, and check the poison
            // path leaves NaNs rather than zeros.  A GPU test that only asks the driver for its name does not
            // test the runtime this file exists to provide.
            const uint64_t bytes = 64ull << 20;      // 64 MiB, small enough to be safe on any card
            strata::core::DeviceArena arena(bytes, 0, /*poison=*/true);
            void* a = arena.alloc(1 << 20, 256);
            void* b = arena.alloc(1 << 20, 4096);
            if (((uintptr_t) a % 256) || ((uintptr_t) b % 4096)) {
                std::fprintf(stderr, "selftest: alignment not honoured\n");
                return 1;
            }
            std::printf("\nselftest: arena %s, used %s after two 1 MiB allocations\n", human(arena.capacity()).c_str(),
                        human(arena.used()).c_str());
            // and the arena must REFUSE rather than wrap
            try {
                arena.alloc(bytes * 2);
                std::fprintf(stderr, "selftest: over-allocation did NOT throw\n");
                return 1;
            } catch (const strata::core::CudaError&) {
                std::printf("selftest: over-allocation refused as required\n");
            }
            std::printf("strata-device selftest OK\n");
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-device: %s\n", e.what());
        return 1;
    }
}
