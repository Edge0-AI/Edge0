// strata_ds4_launch(kernel, grid, block, shared, stream, args...): the engine's one spelling of a kernel launch.  On CUDA it is
// kernel<<<grid, block, shared, stream>>>(args...); on Metal metal/include/metal_compat/cuda_launch.h maps the same call onto
// a dispatch.  (Every file that launches kernels includes this header.)
#pragma once
#if (defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__)) && !defined(STRATA_METAL)
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
template <class... KA, class... CA>
inline void strata_ds4_launch_at(const char* file, int line, void (*kernel)(KA...), dim3 grid, dim3 block, size_t shared,
                                 cudaStream_t stream, CA&&... args) {
    kernel<<<grid, block, shared, stream>>>(static_cast<KA>(args)...);
    static const bool check = std::getenv("DS4_LAUNCH_CHECK") != nullptr;   // report the launch that failed (not a later call)
    if (check) {
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess)
            std::fprintf(stderr, "ds4 launch failed at %s:%d: %s (grid %u,%u,%u block %u,%u,%u shared %zu)\n", file, line,
                         cudaGetErrorString(e), grid.x, grid.y, grid.z, block.x, block.y, block.z, shared);
    }
}
#define strata_ds4_launch(...) strata_ds4_launch_at(__FILE__, __LINE__, __VA_ARGS__)

// (the Metal debug dumps in ds4_model.cu name these; on CUDA there is no deferred queue and a device pointer is not a host view)
inline void ds4_metal_flush() {}
namespace strata::gpu {
inline void* host_view(const void* p) { return const_cast<void*>(p); }
}
#endif

// ds4_dbg("NAME"): the Mac engine's opt-in diagnostics (DS4_DBG_*, DS4_CPU_LOG, DS4_TLOG, ...).  On Metal it is an environment
// lookup; everywhere else it is the constant false, so the CUDA and HIP engines carry none of those checks on their hot paths.
#if defined(STRATA_METAL)
#include <cstdlib>
inline bool ds4_dbg(const char* name) { return std::getenv(name) != nullptr; }
#else
constexpr bool ds4_dbg(const char*) { return false; }
#endif
