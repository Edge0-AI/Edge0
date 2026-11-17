// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// include/strata/dsv41/hip_or_cuda.h - the runtime the dsv41 *tests* need, whichever backend the build has.
//
// The engine's own sources take their runtime from the backend's headers (ds4_*.cu compile as HIP or as CUDA).  The tests
// are plain .cpp files that launch kernels and copy buffers, and they used to include <hip/hip_runtime.h> unconditionally,
// which no CUDA toolkit ships: every CUDA box had to pass a shim on -I to build them (found on l620 and z440, 2026-10-23).
// This header is that shim, in the tree: HIP names on a HIP build, mapped onto the CUDA runtime otherwise.
//
// Two details that a first version got wrong (z440, 2026-10-23): HIP's hipMalloc is a *template* taking T**, so a plain
// void** overload does not bind `hipMalloc(&uint8_ptr)`; and hipMemset has no CUDA counterpart under that name.
#pragma once

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>

using hipError_t = cudaError_t;
using hipEvent_t = cudaEvent_t;
using hipStream_t = cudaStream_t;
constexpr hipError_t hipSuccess = cudaSuccess;
constexpr int hipMemcpyHostToDevice = cudaMemcpyHostToDevice;
constexpr int hipMemcpyDeviceToHost = cudaMemcpyDeviceToHost;

inline hipError_t hipMalloc(void** p, size_t n) { return cudaMalloc(p, n); }
template <typename T>
inline hipError_t hipMalloc(T** p, size_t n) { return cudaMalloc(reinterpret_cast<void**>(p), n); }
inline hipError_t hipFree(void* p) { return cudaFree(p); }
inline hipError_t hipMemcpy(void* dst, const void* src, size_t n, int kind) {
    return cudaMemcpy(dst, src, n, static_cast<cudaMemcpyKind>(kind));
}
template <typename T>
inline hipError_t hipMemcpy(T* dst, const T* src, size_t n, int kind) {
    return cudaMemcpy(static_cast<void*>(dst), static_cast<const void*>(src), n, static_cast<cudaMemcpyKind>(kind));
}
inline hipError_t hipMemset(void* p, int v, size_t n) { return cudaMemset(p, v, n); }
inline hipError_t hipDeviceSynchronize() { return cudaDeviceSynchronize(); }
inline hipError_t hipSetDevice(int d) { return cudaSetDevice(d); }
inline const char* hipGetErrorString(hipError_t e) { return cudaGetErrorString(e); }
inline hipError_t hipEventCreate(hipEvent_t* e) { return cudaEventCreate(e); }
inline hipError_t hipEventRecord(hipEvent_t e, hipStream_t s = nullptr) { return cudaEventRecord(e, s); }
inline hipError_t hipEventSynchronize(hipEvent_t e) { return cudaEventSynchronize(e); }
inline hipError_t hipEventElapsedTime(float* ms, hipEvent_t a, hipEvent_t b) { return cudaEventElapsedTime(ms, a, b); }
#endif
