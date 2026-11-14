// The CUDA function-space qualifiers and the few launch/kernel spellings the shared sources use.  On Metal they say
#include "cuda_launch.h"
// nothing: a "kernel" here is an MSL function compiled at run time and dispatched by the host runtime, so the qualifiers
// are empty and __forceinline__ is plain inline.  Part of the metal_compat layer (M5 Max, 2026-10-23).
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __global__
#define __global__
#endif
#ifndef __shared__
#define __shared__
#endif
#ifndef __constant__
#define __constant__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#ifndef __launch_bounds__
#define __launch_bounds__(...)
#endif
#ifndef __restrict__
#define __restrict__
#endif

// metal/include/metal_compat/cuda_runtime.h - the CUDA runtime names the host files use, on Metal.
//
// The HIP build's approach (include/strata/hip_compat/cuda_runtime.h maps every cudaX the host code calls to hipX),
// applied to Metal: this directory goes first on the Metal build's include path, so the shared host files compile for
// Metal UNCHANGED, and the CUDA / HIP builds never see this file (their objects stay byte-identical by construction).
// Every name forwards to strata::gpu's Metal implementation (metal/src/gpu/runtime.cpp), whose semantics are the
// CUDA ones.  Only the names in the Phase 1 inventory (phaseA-tests/mac-design.md §1.3) are provided; a host file that
// uses another one fails to compile on Metal, which is the point.
#pragma once

// CUDA's alignment qualifier
#ifndef __align__
#define __align__(n) alignas(n)
#endif

#include "cuda_device_builtins.h"

#include <cmath>

using std::fmaxf;
using std::fminf;
using std::sqrtf;
using std::fabsf;
using std::powf;
using std::expf;
using std::logf;

// CUDA's headers put min/max in scope for integers; the sources rely on it
template <typename T> static inline T min(T a, T b) { return strata_min(a, b); }
template <typename T> static inline T max(T a, T b) { return strata_max(a, b); }

#include <cstddef>
#include <cstdint>

#include <immintrin.h>               // metal_compat/immintrin.h: CUDA's headers bring the x86 one in on a PC
#include "strata/gpu/runtime.hpp"    // metal/include/strata/gpu: the Metal runtime

namespace strata::gpu::compat_detail { inline Error bad() { return kErrorUnsupported; } }

// ---- types and constants
using cudaError_t = strata::gpu::Error;
using cudaStream_t = strata::gpu::Stream;
using cudaEvent_t = strata::gpu::Event;
using cudaGraph_t = strata::gpu::Graph;
using cudaGraphExec_t = strata::gpu::GraphExec;
using cudaGraphNode_t = void*;
using cudaDeviceProp = strata::gpu::DeviceProp;
using cudaMemcpyKind = strata::gpu::CopyKind;
using cudaHostFn_t = strata::gpu::HostFn;
inline constexpr cudaError_t cudaSuccess = strata::gpu::kSuccess;
inline constexpr cudaError_t cudaErrorNotReady = strata::gpu::kNotReady;
inline constexpr cudaError_t cudaErrorMemoryAllocation = strata::gpu::kErrorMemoryAllocation;
inline constexpr cudaError_t cudaErrorPeerAccessAlreadyEnabled = strata::gpu::kErrorPeerAccessAlreadyEnabled;
inline constexpr cudaError_t cudaErrorStreamCaptureUnsupported = strata::gpu::kErrorUnsupported;
inline constexpr cudaMemcpyKind cudaMemcpyHostToDevice = strata::gpu::kHostToDevice;
inline constexpr cudaMemcpyKind cudaMemcpyDeviceToHost = strata::gpu::kDeviceToHost;
inline constexpr cudaMemcpyKind cudaMemcpyDeviceToDevice = strata::gpu::kDeviceToDevice;
inline constexpr cudaMemcpyKind cudaMemcpyDefault = strata::gpu::kCopyDefault;
inline constexpr unsigned cudaHostAllocDefault = strata::gpu::kHostAllocDefault;
inline constexpr unsigned cudaHostAllocPortable = strata::gpu::kHostAllocPortable;
inline constexpr unsigned cudaHostAllocMapped = strata::gpu::kHostAllocMapped;
inline constexpr unsigned cudaHostRegisterPortable = strata::gpu::kHostRegisterPortable;
inline constexpr unsigned cudaHostRegisterMapped = strata::gpu::kHostRegisterMapped;
inline constexpr unsigned cudaHostRegisterReadOnly = strata::gpu::kHostRegisterReadOnly;
inline constexpr unsigned cudaEventDefault = strata::gpu::kEventDefault;
inline constexpr unsigned cudaEventDisableTiming = strata::gpu::kEventDisableTiming;
inline constexpr unsigned cudaEventBlockingSync = strata::gpu::kEventBlockingSync;
inline constexpr unsigned cudaStreamNonBlocking = strata::gpu::kStreamNonBlocking;
enum cudaStreamCaptureStatus { cudaStreamCaptureStatusNone = 0, cudaStreamCaptureStatusActive = 1, cudaStreamCaptureStatusInvalidated = 2 };
static inline strata::gpu::Error cudaStreamIsCapturing(strata::gpu::Stream, cudaStreamCaptureStatus* st) {
    if (st) *st = cudaStreamCaptureStatusNone;
    return strata::gpu::kSuccess;
}

enum cudaStreamCaptureMode { cudaStreamCaptureModeGlobal = 0, cudaStreamCaptureModeThreadLocal = 1, cudaStreamCaptureModeRelaxed = 2 };
// (the compute capability is asked by the 0.1.45 Turing defaults in verify.cpp: no answer here, which reads as "not a Turing card")
enum cudaDeviceAttr {
    cudaDevAttrClockRate = 13, cudaDevAttrMultiProcessorCount = 16, cudaDevAttrIntegrated = 18,
    cudaDevAttrComputeCapabilityMajor = 75, cudaDevAttrComputeCapabilityMinor = 76
};

// ---- errors, devices
inline cudaError_t cudaGetLastError() { return strata::gpu::get_last_error(); }
inline cudaError_t cudaPeekAtLastError() { return strata::gpu::peek_last_error(); }
inline const char* cudaGetErrorString(cudaError_t e) { return strata::gpu::error_string(e); }
inline cudaError_t cudaGetDevice(int* d) { return strata::gpu::get_device(d); }
inline cudaError_t cudaSetDevice(int d) { return strata::gpu::set_device(d); }
inline cudaError_t cudaGetDeviceCount(int* n) { return strata::gpu::device_count(n); }
inline cudaError_t cudaDeviceSynchronize() { return strata::gpu::device_synchronize(); }
inline cudaError_t cudaMemGetInfo(size_t* f, size_t* t) { return strata::gpu::mem_get_info(f, t); }
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int d) { return strata::gpu::get_device_properties(p, d); }
inline cudaError_t cudaRuntimeGetVersion(int* v) { return strata::gpu::runtime_version(v); }
inline cudaError_t cudaDriverGetVersion(int* v) { return strata::gpu::runtime_version(v); }
inline cudaError_t cudaDeviceGetAttribute(int* v, cudaDeviceAttr a, int d) {
    switch (a) {
        case cudaDevAttrIntegrated: return strata::gpu::device_integrated(v, d);
        case cudaDevAttrMultiProcessorCount: return strata::gpu::device_sm_count(v, d);
        case cudaDevAttrClockRate: return strata::gpu::device_clock_khz(v, d);
        case cudaDevAttrComputeCapabilityMajor:
        case cudaDevAttrComputeCapabilityMinor: return strata::gpu::compat_detail::bad();
    }
    return strata::gpu::compat_detail::bad();
}
inline cudaError_t cudaDeviceCanAccessPeer(int* v, int d, int p) { return strata::gpu::device_can_access_peer(v, d, p); }
inline cudaError_t cudaDeviceEnablePeerAccess(int p, unsigned f) { return strata::gpu::device_enable_peer_access(p, f); }
inline cudaError_t cudaDeviceDisablePeerAccess(int) { return cudaSuccess; }   // a Mac has one GPU: nothing to disable

// ---- memory and copies
// the deferred command buffer mtl_rt batches the port's dispatches into: a copy (or any host read of a device result)
// has to commit it first, or it reads memory the GPU has not written yet.  Defined in ds4_finish.cpp.
extern "C" void ds4_metal_flush();
extern "C" void ds4_metal_kick();

template <class T> inline cudaError_t cudaMalloc(T** p, size_t n) { return strata::gpu::malloc_device(p, n); }
inline cudaError_t cudaFree(void* p) { return strata::gpu::free_device(p); }
template <class T> inline cudaError_t cudaHostAlloc(T** p, size_t n, unsigned f) { return strata::gpu::host_alloc(p, n, f); }
template <class T> inline cudaError_t cudaMallocHost(T** p, size_t n) { return strata::gpu::malloc_host(p, n); }
inline cudaError_t cudaFreeHost(void* p) { return strata::gpu::free_host(p); }
template <class T> inline cudaError_t cudaHostGetDevicePointer(T** d, void* h, unsigned f) { return strata::gpu::host_device_pointer(d, h, f); }
inline cudaError_t cudaHostRegister(void* p, size_t n, unsigned f) { return strata::gpu::host_register(p, n, f); }
inline cudaError_t cudaHostUnregister(void* p) { return strata::gpu::host_unregister(p); }
inline cudaError_t cudaMemset(void* p, int v, size_t n) { return strata::gpu::memset_device(p, v, n); }
inline cudaError_t cudaMemsetAsync(void* p, int v, size_t n, cudaStream_t s = nullptr) { return strata::gpu::memset_async(p, v, n, s); }
inline cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind k) { return strata::gpu::memcpy_sync(d, s, n, k); }
inline cudaError_t cudaMemcpyAsync(void* d, const void* s, size_t n, cudaMemcpyKind k, cudaStream_t st = nullptr) {
    return strata::gpu::memcpy_async(d, s, n, k, st);
}
inline cudaError_t cudaMemcpy2DAsync(void* d, size_t dp, const void* s, size_t sp, size_t w, size_t h, cudaMemcpyKind k,
                                     cudaStream_t st = nullptr) {
    return strata::gpu::memcpy_2d_async(d, dp, s, sp, w, h, k, st);
}
inline cudaError_t cudaMemcpyPeerAsync(void* d, int dd, const void* s, int sd, size_t n, cudaStream_t st = nullptr) {
    return strata::gpu::memcpy_peer_async(d, dd, s, sd, n, st);
}
inline cudaError_t cudaMemcpyPeer(void* d, int dd, const void* s, int sd, size_t n) {
    const cudaError_t e = cudaMemcpyPeerAsync(d, dd, s, sd, n, nullptr);
    return e != cudaSuccess ? e : cudaDeviceSynchronize();
}

// ---- streams, events, host functions
inline cudaError_t cudaStreamCreate(cudaStream_t* s) { return strata::gpu::stream_create(s); }
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned f) { return strata::gpu::stream_create(s, f); }
inline cudaError_t cudaDeviceGetStreamPriorityRange(int*, int*) { return strata::gpu::compat_detail::bad(); }   // no priorities: callers fall back to a plain stream
inline cudaError_t cudaStreamCreateWithPriority(cudaStream_t* s, unsigned f, int) { return strata::gpu::stream_create(s, f); }
inline cudaError_t cudaStreamDestroy(cudaStream_t s) { return strata::gpu::stream_destroy(s); }
inline cudaError_t cudaStreamSynchronize(cudaStream_t s) { return strata::gpu::stream_synchronize(s); }
inline cudaError_t cudaStreamQuery(cudaStream_t s) { return strata::gpu::stream_query(s); }
inline cudaError_t cudaStreamWaitEvent(cudaStream_t s, cudaEvent_t e, unsigned f = 0) { return strata::gpu::stream_wait_event(s, e, f); }
inline cudaError_t cudaLaunchHostFunc(cudaStream_t s, cudaHostFn_t fn, void* a) { return strata::gpu::launch_host_func(s, fn, a); }
inline cudaError_t cudaEventCreate(cudaEvent_t* e) { return strata::gpu::event_create(e); }
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned f) { return strata::gpu::event_create(e, f); }
inline cudaError_t cudaEventDestroy(cudaEvent_t e) { return strata::gpu::event_destroy(e); }
inline cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t s = nullptr) { return strata::gpu::event_record(e, s); }
inline cudaError_t cudaEventQuery(cudaEvent_t e) { return strata::gpu::event_query(e); }
inline cudaError_t cudaEventSynchronize(cudaEvent_t e) { return strata::gpu::event_synchronize(e); }
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t a, cudaEvent_t b) { return strata::gpu::event_elapsed_ms(ms, a, b); }

// ---- graphs
inline cudaError_t cudaStreamBeginCapture(cudaStream_t s, cudaStreamCaptureMode m) {
    return m == cudaStreamCaptureModeThreadLocal ? strata::gpu::begin_capture(s) : strata::gpu::compat_detail::bad();
}
inline cudaError_t cudaStreamEndCapture(cudaStream_t s, cudaGraph_t* g) { return strata::gpu::end_capture(s, g); }
inline cudaError_t cudaGraphGetNodes(cudaGraph_t g, cudaGraphNode_t* nodes, size_t* n) {
    return nodes == nullptr ? strata::gpu::graph_node_count(g, n) : strata::gpu::compat_detail::bad();
}
inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t* x, cudaGraph_t g, cudaGraphNode_t*, char*, size_t) {
    return strata::gpu::graph_instantiate(x, g);
}
inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t* x, cudaGraph_t g, unsigned long long f = 0) {
    return strata::gpu::graph_instantiate(x, g, f);
}
inline cudaError_t cudaGraphLaunch(cudaGraphExec_t x, cudaStream_t s) { return strata::gpu::graph_launch(x, s); }
inline cudaError_t cudaGraphUpload(cudaGraphExec_t x, cudaStream_t s) { return strata::gpu::graph_upload(x, s); }
inline cudaError_t cudaGraphDestroy(cudaGraph_t g) { return strata::gpu::graph_destroy(g); }
inline cudaError_t cudaGraphExecDestroy(cudaGraphExec_t x) { return strata::gpu::graph_exec_destroy(x); }

// cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind): a strided copy, used by the vision tower's patch rows and
// the Engram tables' column slices.  The Metal runtime has no pitched copy yet, so this reports failure rather than
// copying nothing silently.
inline strata::gpu::Error cudaMemcpy2D(void*, size_t, const void*, size_t, size_t, size_t, strata::gpu::CopyKind) {
    return strata::gpu::kErrorUnsupported;   // cudaMemcpy2D_unimplemented_marker
}

// occupancy query: the Metal runtime decides its own threadgroup limits, so answer "one block" and let the caller size grids
template <typename K, typename... A>
static inline strata::gpu::Error cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* n, K&&, int, A&&...) {
    if (n) *n = 1;
    return strata::gpu::kSuccess;
}
