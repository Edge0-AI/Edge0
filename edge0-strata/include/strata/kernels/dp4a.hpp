// include/strata/kernels/dp4a.hpp - the two instructions the kernels use that Pascal (sm_60) does not have.
//
// The engine is written for compute capability 7.5 and newer and refuses older cards at CMake configure time
// (a Pascal build passes -DSTRATA_EXPERIMENTAL_SM60=ON).  A GP100-class card is compute capability 6.0, which is
// missing exactly two things the i-quant kernels rely on:
//
//   * `__dp4a`, a byte-wise dot product, available from 6.1 (Pascal GP10x/GV11x).  The reference for its
//     fallback is llama.cpp's own (`ggml/src/ggml-cuda/common.cuh`), since the kernels in this directory are
//     transcribed from llama.cpp's vecdotq.cuh and that wrapper's operands are the same sites: it reinterprets
//     the operands as SIGNED bytes and accumulates in int32, which is what `__dp4a` does for these calls.  The
//     fallback below computes exactly that with four PTX `vmad` byte-select instructions, so it is bit-exact
//     rather than merely close (tools/sm60_dp4a_check.cu checks it against the byte-wise form).
//   * `__nanosleep`, available from 7.0 (Volta).  It only paces single-thread doorbell waits, so a loop that
//     spins without it is correct, merely busier.
//
// Both are compile-time selections on `__CUDA_ARCH__`, so one source tree builds for Pascal and for RTX
// 20/30/40/50 without a #define at the call sites.  A HIP build never takes either fallback: `__CUDA_ARCH__` is
// undefined there, so `STRATA_DP4A` is `__dp4a` (which `hip_compat/intrinsics.hpp` provides) and the pause is
// HIP's own `__nanosleep`.
#pragma once

#include <cstdint>
#if defined(STRATA_HIP_GFX906)
#include <cuda_runtime.h>   // gfx906: the compat layer (__forceinline__, __nanosleep, __dp4a)
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 610
__device__ __forceinline__ int strata_dp4a(const int a, const int b, const int c) {
    // PTX vmad with byte selectors: compiles to four VMAD.S8.S8 (cuobjdump -sass), no byte extraction.
    // Bit-exact with the byte-wise form it replaces: 1.07e9 cases (edge bytes, every per-lane byte pair, random),
    // 0 mismatches, ~2.2x faster in isolation on a Tesla P100 (tools/sm60_dp4a_check.cu).
    int r = c;
    asm("vmad.s32.s32.s32 %0, %1.b0, %2.b0, %0;\n\t"
        "vmad.s32.s32.s32 %0, %1.b1, %2.b1, %0;\n\t"
        "vmad.s32.s32.s32 %0, %1.b2, %2.b2, %0;\n\t"
        "vmad.s32.s32.s32 %0, %1.b3, %2.b3, %0;"
        : "+r"(r) : "r"(a), "r"(b));
    return r;
}
#define STRATA_DP4A(a, b, c) strata_dp4a((a), (b), (c))
#else
#define STRATA_DP4A(a, b, c) __dp4a((a), (b), (c))
#endif

/// Yield the thread while a doorbell flag is being polled.  The length of the pause is a backoff hint, not a
/// contract, and only pre-Volta CUDA has no `__nanosleep` at all.
__device__ __forceinline__ void strata_spin_pause() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
    // sm_6x: the loop spins.  Every call site is a single-thread doorbell wait, so nothing else is delayed.
#else
    __nanosleep(100);
#endif
}
