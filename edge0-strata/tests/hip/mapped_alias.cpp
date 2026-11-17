// Is a hipHostGetDevicePointer alias usable on this stack? (#325)
//
// tests/hip/handoff.cpp fails on gfx1201/Windows with "separate copy/ring timeout": it does a
// hipMemcpyAsync(..., hipMemcpyDeviceToDevice) into a hipHostGetDevicePointer alias, then waits for
// a doorbell that copy should have published. This test pins down the properties that decide it.
//
// On Linux the alias is a device address of the pinned host memory, a D2D copy lands, the host sees
// it, and a kernel reads it. On Windows (#325, RX 9070 XT) the runtime hands back the HOST pointer
// itself (unified addressing; this test prints both addresses), and a D2D copy into it does not
// publish - hip_handoff's timeout there. The engine itself runs on Windows (the reporter's serving
// run gave correct answers), so kernels reading mapped memory through that address work; only the
// copy-engine path does not. src/core/native_head.cpp keeps the token embedding in VRAM when the
// alias is refused outright.
//
// Reporting, not gating: a Windows host is EXPECTED to fail some checks below, so this test exits 0
// whenever it ran at all, and 77 (skipped) without a device. When the alias IS the host pointer the
// D2D copy is not attempted (it may fault rather than report); the kernel read is, since that is
// what the engine relies on.

#include "strata/kernels/elementwise.hpp"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <vector>

namespace {

int not_held = 0;

void report(const char* what, bool holds) {
    std::printf("  %-44s %s\n", what, holds ? "holds" : "DOES NOT HOLD");
    if (!holds) ++not_held;
}

// A kernel reads `n` floats through `alias` into VRAM; the result is copied back and compared.
bool kernel_reads(const float* alias, const std::vector<float>& want) {
    const int64_t n = (int64_t) want.size();
    float* out = nullptr;
    if (hipMalloc(reinterpret_cast<void**>(&out), want.size() * sizeof(float)) != hipSuccess) return false;
    strata::kernels::copy_from_mapped(out, alias, n, nullptr);
    std::vector<float> got(want.size(), -1.0f);
    const bool ran = hipDeviceSynchronize() == hipSuccess &&
                     hipMemcpy(got.data(), out, want.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess;
    hipFree(out);
    hipGetLastError();
    return ran && got == want;
}

}  // namespace

int main() {
    int runtime = 0;
    if (hipRuntimeGetVersion(&runtime) != hipSuccess) return 1;
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count < 1) {
        std::printf("no HIP device\n");
        return 77;  // ctest "skipped"
    }
    std::printf("HIP runtime %d, %d device(s)\n", runtime, count);

    const int64_t n = 4096;                       // enough elements for copy_from_mapped's float4 path
    const size_t bytes = size_t(n) * sizeof(float);
    std::vector<float> pattern(size_t(n), 0.0f);
    for (int64_t i = 0; i < n; ++i) pattern[size_t(i)] = float(i + 1);

    float* host = nullptr;
    if (hipHostMalloc(reinterpret_cast<void**>(&host), bytes, hipHostMallocMapped) != hipSuccess) return 1;
    void* alias = nullptr;
    if (hipHostGetDevicePointer(&alias, host, 0) != hipSuccess || alias == nullptr) {
        std::printf("hipHostGetDevicePointer failed: native_head.cpp keeps the embedding in VRAM here\n");
        hipGetLastError();
        hipHostFree(host);
        return 0;
    }
    const bool distinct = alias != reinterpret_cast<void*>(host);
    std::printf("\n  host  %p\n  alias %p\n  -> %s\n\n", static_cast<void*>(host), alias,
                distinct ? "a distinct device address" : "the host pointer itself (unified addressing)");
    report("the alias is a distinct device address", distinct);

    // What the engine relies on: a kernel reading mapped host memory through the alias.
    for (int64_t i = 0; i < n; ++i) host[i] = pattern[size_t(i)];
    report("a kernel reads the alias correctly", kernel_reads(reinterpret_cast<const float*>(alias), pattern));

    // What tests/hip/handoff.cpp attempts: a DeviceToDevice copy INTO the alias, seen by the host.
    if (distinct) {
        float* device = nullptr;
        std::vector<float> twice(pattern);
        for (float& v : twice) v *= 2.0f;
        if (hipMalloc(reinterpret_cast<void**>(&device), bytes) == hipSuccess &&
            hipMemcpy(device, twice.data(), bytes, hipMemcpyHostToDevice) == hipSuccess) {
            const hipError_t copy = hipMemcpy(alias, device, bytes, hipMemcpyDeviceToDevice);
            report("a D2D copy into the alias succeeds", copy == hipSuccess);
            bool seen = hipDeviceSynchronize() == hipSuccess;
            for (int64_t i = 0; seen && i < n; ++i) seen = host[i] == twice[size_t(i)];
            report("the host sees a D2D copy into the alias", seen);
        }
        if (device) hipFree(device);
        hipGetLastError();
    } else {
        std::printf("  the D2D copy into the alias is not attempted: with alias == host it may fault rather than\n"
                    "  report (tests/hip/handoff's timeout on Windows)\n");
    }

    std::printf("\n%d check(s) did not hold (a Windows host is expected to fail some)\n", not_held);
    hipHostFree(host);
    return 0;
}
