// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// include/strata/dsv41/hip_extra.h - CUDA runtime names the DeepSeek-V4.1 engine path uses that Strata's HIP shim
// (include/strata/hip_compat) does not map.  Force-included into the dsv41 targets only, so no other target changes.
// The mappings are Project Maya's include/strata/hip_compat (MIT, third_party/project-maya/NOTICE.md).
#pragma once
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#ifndef __grid_constant__
#define __grid_constant__
#endif
#ifndef cudaDevAttrMemoryClockRate
#define cudaDevAttrMemoryClockRate hipDeviceAttributeMemoryClockRate
#endif
#ifndef cudaDevAttrGlobalMemoryBusWidth
#define cudaDevAttrGlobalMemoryBusWidth hipDeviceAttributeMemoryBusWidth
#endif
#ifndef cudaDevAttrCanUseHostPointerForRegisteredMem
#define cudaDevAttrCanUseHostPointerForRegisteredMem hipDeviceAttributeCanUseHostPointerForRegisteredMem
#endif
#ifndef cudaDeviceGetPCIBusId
#define cudaDeviceGetPCIBusId hipDeviceGetPCIBusId
#endif
#ifndef cudaDeviceGetStreamPriorityRange
#define cudaDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange
#endif
#ifndef cudaOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#endif
#ifndef cudaErrorUnknown
#define cudaErrorUnknown hipErrorUnknown
#endif
#ifndef cudaErrorInvalidConfiguration
#define cudaErrorInvalidConfiguration hipErrorInvalidConfiguration
#endif
#ifndef cudaMemcpy2D
#define cudaMemcpy2D hipMemcpy2D
#endif
#ifndef cudaStreamCreateWithPriority
#define cudaStreamCreateWithPriority hipStreamCreateWithPriority
#endif
#ifndef cudaStreamLegacy
#define cudaStreamLegacy hipStreamLegacy
#endif
#ifndef CUBLAS_GEMM_DEFAULT_TENSOR_OP
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#endif
#ifndef cudaDataType
#define cudaDataType hipDataType
#endif
