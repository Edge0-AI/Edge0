// include/strata/kernels/pdl.hpp - programmatic dependent launch (PDL) for the verify window's kernel chain.
//
// On sm_90+ a kernel launched with cudaLaunchAttributeProgrammaticStreamSerialization may start while the kernel
// before it is still running: everything before `pdl_wait()` overlaps that kernel (and its launch latency), everything
// after it sees the kernel's writes, exactly as after an ordinary launch.  The kernels that opt in only LOAD constant
// weights before `pdl_wait()` and write nothing, so every value they compute is unchanged.  `pdl_trigger()` lets the
// next kernel launch early (it still waits for this grid to finish before it reads anything).  Their activations and
// outputs are not `__restrict__` on CUDA (STRATA_PDL_RESTRICT): with it the compiler may load them before the wait
// (the reason llama.cpp's PDL drops __restrict__ too); the weights keep it.
//
// Stream capture turns the attribute into a programmatic graph edge, which CUDA allows only between two kernel nodes.
// `launch_pdl` therefore uses it only while `pdl_scope()` is set (the verify window's recorder, on a device where
// `pdl_supported()`), only for a kernel whose code was built for sm_90+ (PTX of an older architecture, JIT-compiled,
// has no griddepcontrol.wait: such a kernel would race), and only when every node the capturing stream's next node
// depends on is a kernel - never after a memcpy, a memset, a host node or an event node.  After a join of branches
// (several kernels) each edge is programmatic and the wait covers them all; STRATA_DF_PDL=2 allows one predecessor
// only.  Everything else is an ordinary launch.  On HIP all of this is an ordinary launch, the device functions are
// no-ops and the kernels keep their plain loops (kPdlPrefetch).
#pragma once

#include <cuda_runtime.h>

#include <utility>

namespace strata::kernels {

/// Whether launches on this thread may carry the PDL attribute (set by the verify window's recorder).
bool& pdl_scope();

/// The current device runs PDL: a CUDA 12.3+ build, sm_90+ (STRATA_EMULATE_CC counts), code built for it, and
/// STRATA_DF_PDL is set (1, or 2 for single-predecessor edges only).  Always false on HIP.
bool pdl_supported();

#if (defined(__CUDACC__) || defined(__HIPCC__))

// STRATA_PDL_RESTRICT never goes on a __global__ signature: the host pass and the stub must see the same signature on
// every architecture (MSVC rejects the template stubs otherwise, and so does g++ when the stub is generated from a
// device pass).  A kernel takes plain pointers and declares `const T* STRATA_PDL_RESTRICT p = p_;` locals in its body.
// On CUDA the activation pointers lose __restrict__ so the compiler cannot load them before pdl_wait() - but only
// where PDL can run (sm_70 and newer here).  On Pascal and older (__CUDA_ARCH__ < 700, no PDL) `__restrict__` is what
// lets the compiler use the read-only data path (LDG.CI): without it decode on sm_61 halves (#1469).
#if defined(__HIPCC__)
#define STRATA_PDL_RESTRICT __restrict__
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
#define STRATA_PDL_RESTRICT __restrict__
#else
#define STRATA_PDL_RESTRICT
#endif

#if defined(__HIPCC__)
inline constexpr bool kPdlPrefetch = false;   // HIP: no PDL, the kernels keep their plain loops
#else
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
inline constexpr bool kPdlPrefetch = false;   // Pascal and older: no PDL, the prefetch only costs registers (#1469)
#else
inline constexpr bool kPdlPrefetch = true;    // CUDA: the weights a kernel can load before pdl_wait() are loaded there
#endif
/// `kernel` may be launched with the PDL attribute into `stream` now (see the header comment).
bool pdl_launch_ok(const void* kernel, cudaStream_t stream);
#endif

__device__ __forceinline__ void pdl_wait() {
#if !defined(__HIPCC__) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void pdl_trigger() {
#if !defined(__HIPCC__) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;" :::);
#endif
}

/// `kernel<<<grid, block, smem, stream>>>(args...)`, with the PDL attribute when `pdl_launch_ok` says so.  A launch
/// error stays the thread's last error for the caller's own check (cudaGetLastError) to report.
template <typename... KArgs, typename... Args>
inline void launch_pdl(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t stream,
                       Args&&... args) {
#if !defined(__HIPCC__)
    if (pdl_scope() && pdl_launch_ok((const void*) kernel, stream)) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = grid;
        cfg.blockDim = block;
        cfg.dynamicSmemBytes = smem;
        cfg.stream = stream;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        (void) cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...);
        return;
    }
#endif
    kernel<<<grid, block, smem, stream>>>(std::forward<Args>(args)...);
}

#endif  // __CUDACC__ || __HIPCC__

}  // namespace strata::kernels
