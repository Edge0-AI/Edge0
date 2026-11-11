// strata_hip.h - the CUDA runtime/cuBLAS names Strata uses, mapped onto HIP/hipBLAS for AMD GPUs (gfx906).
//
// This directory is put FIRST on the include path of a STRATA_ENABLE_HIP build only; its `cuda_runtime.h`,
// `cuda_fp16.h` and `cublas_v2.h` include this header instead of the NVIDIA ones, so the engine's sources stay
// one tree.  The mapping list follows llama.cpp's ggml-cuda/vendors/hip.h (MIT) where the two overlap.
//
// The warp contract: every Strata kernel is written for a 32-lane warp.  gfx906 runs 64-lane wavefronts, so
// here a CUDA warp is a LOGICAL 32-lane half of a wavefront: every shuffle defaults to width 32 (a
// `__shfl_down_sync(m, v, 1)` at lane 31 must not read lane 32), `__ballot_sync` returns the 32-bit mask of the
// caller's own half, and `__syncwarp` is a wavefront barrier plus an LDS fence.  That keeps every kernel's
// arithmetic, lane mapping and reduction order exactly as on NVIDIA; wave64-native rewrites of the hot kernels
// are separate, measured changes.
#pragma once

#ifndef HIP_DISABLE_WARP_SYNC_BUILTINS
#define HIP_DISABLE_WARP_SYNC_BUILTINS 1
#endif
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cstdint>
#include <cstdlib>

#define STRATA_HIP_GFX906 1

// ---- runtime: types and constants ------------------------------------------------------------------------
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaErrorNotReady hipErrorNotReady
#define cudaErrorMemoryAllocation hipErrorOutOfMemory
#define cudaErrorStreamCaptureUnsupported hipErrorStreamCaptureUnsupported
#define cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t
#define cudaGraph_t hipGraph_t
#define cudaGraphExec_t hipGraphExec_t
#define cudaGraphNode_t hipGraphNode_t
#define cudaDeviceProp hipDeviceProp_t
#define cudaMemcpyKind hipMemcpyKind
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDefault hipMemcpyDefault
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#define cudaStreamCaptureModeRelaxed hipStreamCaptureModeRelaxed
#define cudaStreamCaptureModeGlobal hipStreamCaptureModeGlobal
#define cudaStreamCaptureStatus hipStreamCaptureStatus
#define cudaStreamCaptureStatusNone hipStreamCaptureStatusNone
#define cudaStreamCaptureStatusActive hipStreamCaptureStatusActive
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventBlockingSync hipEventBlockingSync
#define cudaEventDefault hipEventDefault
#define cudaHostAllocDefault hipHostMallocDefault
#define cudaHostAllocPortable hipHostMallocPortable
#define cudaHostAllocMapped hipHostMallocMapped
#define cudaHostAllocWriteCombined hipHostMallocWriteCombined
#define cudaHostRegisterDefault hipHostRegisterDefault
#define cudaHostRegisterPortable hipHostRegisterPortable
#define cudaHostRegisterMapped hipHostRegisterMapped
#define cudaHostRegisterReadOnly hipHostRegisterReadOnly
#define cudaDeviceScheduleSpin hipDeviceScheduleSpin
#define cudaDeviceScheduleYield hipDeviceScheduleYield
#define cudaDeviceScheduleBlockingSync hipDeviceScheduleBlockingSync
#define cudaDeviceMapHost hipDeviceMapHost
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncAttributePreferredSharedMemoryCarveout hipFuncAttributePreferredSharedMemoryCarveout
// CUDA's cudaSharedmemCarveoutMaxShared is a percentage (100 = give the block all of the carveout); HIP has no
// counterpart constant, so spell it out (the carveout attribute takes an int percentage).
#define cudaSharedmemCarveoutMaxShared 100
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define cudaDevAttrClockRate hipDeviceAttributeClockRate
#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor
#define cudaDevAttrMaxSharedMemoryPerBlockOptin hipDeviceAttributeSharedMemPerBlockOptin
#define cudaDevAttrMemoryClockRate hipDeviceAttributeMemoryClockRate
#define cudaDevAttrGlobalMemoryBusWidth hipDeviceAttributeMemoryBusWidth

// ---- runtime: functions ----------------------------------------------------------------------------------
#define cudaGetErrorString hipGetErrorString
#define cudaGetErrorName hipGetErrorName
#define cudaGetLastError hipGetLastError
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
// The shared-memory opt-in limit: some HIP versions report 0 for it on GCN cards; the real ceiling there is
// the plain per-block limit (64 KB of LDS on gfx906).
inline hipError_t strata_device_get_attribute(int* v, hipDeviceAttribute_t a, int dev) {
    hipError_t e = hipDeviceGetAttribute(v, a, dev);
    if (e == hipSuccess && a == hipDeviceAttributeSharedMemPerBlockOptin && *v <= 0)
        e = hipDeviceGetAttribute(v, hipDeviceAttributeMaxSharedMemoryPerBlock, dev);
    return e;
}
#define cudaDeviceGetAttribute strata_device_get_attribute
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaDeviceDisablePeerAccess hipDeviceDisablePeerAccess
#define cudaDriverGetVersion hipDriverGetVersion
#define cudaRuntimeGetVersion hipRuntimeGetVersion
#define cudaSetDeviceFlags hipSetDeviceFlags
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemGetInfo hipMemGetInfo
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyPeerAsync hipMemcpyPeerAsync
#define cudaMemcpyPeer hipMemcpyPeer
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
// Mapped host memory is how the GPU and the CPU talk inside a running graph (the doorbells, the CPU experts'
// results, the verify flags): a kernel spins on a word the CPU writes, and the CPU on a word the kernel writes.
// On AMD that needs FINE-GRAINED (coherent) memory - coarse-grained host memory may sit in the GPU's L2 for the
// whole kernel and the spin never sees the other side.  So every mapped allocation is made coherent here.
inline hipError_t strata_host_alloc(void** p, size_t bytes, unsigned int flags) {
    if (flags & hipHostMallocMapped) flags |= hipHostMallocCoherent;
    return hipHostMalloc(p, bytes, flags);
}
template <typename T> inline hipError_t strata_host_alloc(T** p, size_t bytes, unsigned int flags) {
    return strata_host_alloc(reinterpret_cast<void**>(p), bytes, flags);
}
#define cudaHostAlloc strata_host_alloc
#define cudaMallocHost(ptr, size) hipHostMalloc(ptr, size, hipHostMallocDefault)
#define cudaFreeHost hipHostFree
#define cudaHostRegister hipHostRegister
#define cudaHostUnregister hipHostUnregister
#define cudaHostGetDevicePointer hipHostGetDevicePointer
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamQuery hipStreamQuery
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaStreamIsCapturing hipStreamIsCapturing
#define cudaLaunchHostFunc hipLaunchHostFunc
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventQuery hipEventQuery
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaGraphLaunch hipGraphLaunch
#define cudaGraphUpload hipGraphUpload
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaGraphGetNodes hipGraphGetNodes
#define cudaGraphNodeType hipGraphNodeType
#define cudaGraphNodeGetType hipGraphNodeGetType
#define cudaGraphNodeTypeKernel hipGraphNodeTypeKernel
#define cudaGraphNodeTypeMemcpy hipGraphNodeTypeMemcpy
#define cudaGraphNodeTypeMemset hipGraphNodeTypeMemset
#define cudaKernelNodeParams hipKernelNodeParams
#define cudaGraphKernelNodeGetParams hipGraphKernelNodeGetParams
inline hipError_t cudaFuncGetName(const char** name, const void*) { *name = nullptr; return hipErrorNotSupported; }
// A function, not a macro: the call sites pass kernels as cudaFuncSetAttribute(kernel<A, B, C>, ...) and a
// function-like macro would split its arguments on the commas inside the template argument list (0.1.38's
// hip_compat/cuda_runtime.h had this as a template function too; the macro broke fused_gr.cu on ROCm 7.2.1).
template <typename Kernel>
inline hipError_t cudaFuncSetAttribute(Kernel kernel, hipFuncAttribute attribute, int value) {
    return hipFuncSetAttribute(reinterpret_cast<const void*>(kernel), attribute, value);
}
#define cudaMemcpyToSymbol(sym, src, ...) hipMemcpyToSymbol(HIP_SYMBOL(sym), src, __VA_ARGS__)
#define cudaMemcpyFromSymbol(dst, sym, ...) hipMemcpyFromSymbol(dst, HIP_SYMBOL(sym), __VA_ARGS__)

// CUDA 12 has a 3-argument cudaGraphInstantiate(exec, graph, flags) and the older 5-argument one; HIP spells
// the first hipGraphInstantiateWithFlags.  Both are used in the tree.
inline hipError_t cudaGraphInstantiate(hipGraphExec_t* exec, hipGraph_t graph, unsigned long long flags) {
    return hipGraphInstantiateWithFlags(exec, graph, flags);
}
inline hipError_t cudaGraphInstantiate(hipGraphExec_t* exec, hipGraph_t graph, hipGraphNode_t* err_node,
                                       char* log, size_t log_size) {
    return hipGraphInstantiate(exec, graph, err_node, log, log_size);
}
// cudaInitDevice(dev, deviceFlags, flags): initialise a device and its scheduling flags WITHOUT making it the
// current device (CUDA's contract; the layer split relies on it - a changed current device put GPU 0's expert
// cache on GPU 1).
inline hipError_t cudaInitDevice(int device, unsigned int device_flags, unsigned int /*flags*/) {
    int prev = 0;
    hipError_t e = hipGetDevice(&prev);
    if (e != hipSuccess) return e;
    e = hipSetDevice(device);
    if (e != hipSuccess) return e;
    if (hipSetDeviceFlags(device_flags) != hipSuccess) (void) hipGetLastError();   // context already active
    return hipSetDevice(prev);
}

// ---- device side -----------------------------------------------------------------------------------------
#if defined(__HIP_DEVICE_COMPILE__) || defined(__HIPCC__)

// Logical 32-lane warps (see the header comment).  The width argument, when given, wins.
#define __shfl_sync(mask, var, lane, ...) __shfl(var, lane, STRATA_SHFL_W(__VA_ARGS__))
#define __shfl_up_sync(mask, var, delta, ...) __shfl_up(var, delta, STRATA_SHFL_W(__VA_ARGS__))
#define __shfl_down_sync(mask, var, delta, ...) __shfl_down(var, delta, STRATA_SHFL_W(__VA_ARGS__))
#define __shfl_xor_sync(mask, var, lanemask, ...) __shfl_xor(var, lanemask, STRATA_SHFL_W(__VA_ARGS__))
// STRATA_SHFL_W() -> 32, STRATA_SHFL_W(w) -> w
#define STRATA_SHFL_W(...) STRATA_SHFL_W_I(dummy, ##__VA_ARGS__, STRATA_SHFL_W_GIVEN, STRATA_SHFL_W_DEFAULT)(__VA_ARGS__)
#define STRATA_SHFL_W_I(d, a, b, ...) b
#define STRATA_SHFL_W_GIVEN(w) (w)
#define STRATA_SHFL_W_DEFAULT(...) 32
#define __all_sync(mask, pred) __all(pred)
#define __any_sync(mask, pred) __any(pred)

static __device__ __forceinline__ unsigned int strata_ballot32(int pred) {
    const unsigned long long b = __ballot(pred);
    return (unsigned int) (b >> (__lane_id() & 32u));
}
#define __ballot_sync(mask, pred) strata_ballot32(pred)
#define __activemask() ((unsigned int) (__ballot(1) >> (__lane_id() & 32u)))

static __device__ __forceinline__ void strata_syncwarp() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}
#define __syncwarp(...) strata_syncwarp()

// __nanosleep(ns): s_sleep counts 64 clocks per unit (~40 ns at the 1.7 GHz shader clock); a spin loop only
// needs "back off a little", not an exact time.
static __device__ __forceinline__ void strata_nanosleep(unsigned int ns) {
    if (ns >= 400) __builtin_amdgcn_s_sleep(8);
    else __builtin_amdgcn_s_sleep(1);
}
#define __nanosleep(ns) strata_nanosleep(ns)
#define __trap() __builtin_trap()

// __dp4a(a, b, c): signed 8-bit dot product plus accumulator; gfx906 has v_dot4_i32_i8.
static __device__ __forceinline__ int strata_dp4a(int a, int b, int c) {
#if defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx942__) || defined(__GFX11__) || defined(__GFX12__)
    return __builtin_amdgcn_sdot4(a, b, c, false);
#else
    const int8_t* va = reinterpret_cast<const int8_t*>(&a);
    const int8_t* vb = reinterpret_cast<const int8_t*>(&b);
    return c + va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2] + va[3] * vb[3];
#endif
}
static __device__ __forceinline__ unsigned int strata_dp4a(unsigned int a, unsigned int b, unsigned int c) {
    const uint8_t* va = reinterpret_cast<const uint8_t*>(&a);
    const uint8_t* vb = reinterpret_cast<const uint8_t*>(&b);
    return c + va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2] + va[3] * vb[3];
}
#define __dp4a strata_dp4a

// SIMD-in-a-word byte ops (CUDA semantics: __vsub4 wraps per byte, __vsubss4 saturates, __vcmp*4 give 0xff)
typedef int8_t strata_i8x4 __attribute__((ext_vector_type(4)));
typedef uint8_t strata_u8x4 __attribute__((ext_vector_type(4)));
static __device__ __forceinline__ unsigned int strata_vsub4(unsigned int a, unsigned int b) {
    const strata_u8x4 va = __builtin_bit_cast(strata_u8x4, a);
    const strata_u8x4 vb = __builtin_bit_cast(strata_u8x4, b);
    return __builtin_bit_cast(unsigned int, (strata_u8x4) (va - vb));
}
static __device__ __forceinline__ int strata_vsubss4(int a, int b) {
    const strata_i8x4 va = __builtin_bit_cast(strata_i8x4, a);
    const strata_i8x4 vb = __builtin_bit_cast(strata_i8x4, b);
    return __builtin_bit_cast(int, (strata_i8x4) __builtin_elementwise_sub_sat(va, vb));
}
static __device__ __forceinline__ unsigned int strata_vcmpeq4(unsigned int a, unsigned int b) {
    const strata_u8x4 va = __builtin_bit_cast(strata_u8x4, a);
    const strata_u8x4 vb = __builtin_bit_cast(strata_u8x4, b);
    strata_u8x4 c;
#pragma unroll
    for (int i = 0; i < 4; ++i) c[i] = va[i] == vb[i] ? 0xff : 0x00;
    return __builtin_bit_cast(unsigned int, c);
}
static __device__ __forceinline__ unsigned int strata_vcmpne4(unsigned int a, unsigned int b) {
    return ~strata_vcmpeq4(a, b);
}
#define __vsub4(a, b) ((int) strata_vsub4((unsigned int) (a), (unsigned int) (b)))
#define __vsubss4(a, b) strata_vsubss4((int) (a), (int) (b))
#define __vcmpeq4(a, b) strata_vcmpeq4((unsigned int) (a), (unsigned int) (b))
#define __vcmpne4(a, b) strata_vcmpne4((unsigned int) (a), (unsigned int) (b))


// __fmul_rn / __fadd_rn / __fsub_rn: on NVIDIA these are never contracted into an FMA (that is their point - the
// engine uses them where a result must match a reference bit for bit).  HIP spells them as plain operators, which
// clang's default -ffp-contract=fast-honor-pragmas may fuse.  The pragma keeps each one a separate rounding.
static __device__ __forceinline__ float strata_fmul_rn(float a, float b) {
#pragma clang fp contract(off)
    return a * b;
}
static __device__ __forceinline__ float strata_fadd_rn(float a, float b) {
#pragma clang fp contract(off)
    return a + b;
}
static __device__ __forceinline__ float strata_fsub_rn(float a, float b) {
#pragma clang fp contract(off)
    return a - b;
}
#define __fmul_rn(a, b) strata_fmul_rn((a), (b))
#define __fadd_rn(a, b) strata_fadd_rn((a), (b))
#define __fsub_rn(a, b) strata_fsub_rn((a), (b))


// __byte_perm(x, y, s): HIP's own version builds an 8-byte array and indexes it - the array lands in SCRATCH
// memory, and every IQ4 codebook lookup went through it (a 0.9 MB IQ4_NL GEMV took 290 us, ~3 GB/s).  gfx906 has
// v_perm_b32, which selects bytes of {src0:src1} (src1 low) by byte selectors 0-7: the same operation, one
// instruction.  CUDA reads the low 3 bits of each of the four low selector nibbles.
static __device__ __forceinline__ unsigned int strata_byte_perm(unsigned int x, unsigned int y, unsigned int s) {
    const unsigned int sel = (s & 0x7u) | ((s & 0x70u) << 4) | ((s & 0x700u) << 8) | ((s & 0x7000u) << 12);
    return __builtin_amdgcn_perm(y, x, sel);
}
#define __byte_perm(x, y, s) strata_byte_perm((unsigned int) (x), (unsigned int) (y), (unsigned int) (s))

#endif  // device side
