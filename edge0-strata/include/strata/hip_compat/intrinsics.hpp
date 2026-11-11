#pragma once

// CUDA device intrinsics used by Strata kernels on supported wave32 HIP targets.
// This header is included only from the HIP cuda_runtime compatibility shim.
#if defined(__HIPCC__)
#include <hip/hip_version.h>

#include <cstdint>

namespace strata::hip_compat {

__device__ __forceinline__ int signed_byte(uint32_t word, int lane) {
    const unsigned value = (word >> (lane * 8)) & 0xffu;
    return value < 0x80u ? static_cast<int>(value) : static_cast<int>(value) - 0x100;
}

// CUDA's signed __dp4a: four signed byte products accumulated modulo 2^32.
__device__ __forceinline__ int dp4a(int a, int b, int c) {
#if (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || \
     defined(__gfx1151__) || defined(__gfx1200__) || \
     defined(__gfx1201__)) && __has_builtin(__builtin_amdgcn_sudot4)
    // RDNA3 and RDNA4 expose the signed/unsigned dot4 form (v_dot4_i32_iu8). Mark
    // both packed operands signed to preserve CUDA __dp4a semantics; keep the
    // portable path for other HIP compilers/targets.
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#elif (defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__)) && __has_builtin(__builtin_amdgcn_sdot4)
    // RDNA2 has no sudot4 (that is gfx11+), but it has the plain signed v_dot4_i32_i8 (dot1-insts): the same
    // signed x signed byte products accumulated modulo 2^32, no clamp.
    return __builtin_amdgcn_sdot4(a, b, c, false);
#elif defined(__gfx1012__) && !defined(STRATA_GFX1012_PORTABLE_DOT)
    // RDNA1 has no native signed dot4. SDWA selects and sign-extends byte
    // operands directly, avoiding separate shift/mask/sign-extension work.
    // Adapted from pinned llama.cpp ggml-cuda/common.cuh (MIT; see
    // third_party/ggml/LICENSE), same modulo-2^32 accumulation contract.
    int lo, hi;
    asm("v_mul_i32_i24 %1, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0\n"
        "v_mul_i32_i24 %2, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1\n"
        "v_add3_u32 %0, %1, %2, %0\n"
        "v_mul_i32_i24 %1, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2\n"
        "v_mul_i32_i24 %2, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3\n"
        "v_add3_u32 %0, %1, %2, %0\n"
        // c changes before the high-byte reads; it must not alias a or b.
        : "+&v"(c), "=&v"(lo), "=&v"(hi) : "v"(a), "v"(b));
    return c;
#else
    const uint32_t ua = static_cast<uint32_t>(a);
    const uint32_t ub = static_cast<uint32_t>(b);
    uint32_t sum = static_cast<uint32_t>(c);
#pragma unroll
    for (int lane = 0; lane < 4; ++lane)
        sum += static_cast<uint32_t>(signed_byte(ua, lane) * signed_byte(ub, lane));
    return static_cast<int>(sum);
#endif
}

// CUDA's __byte_perm (default mode): result byte i is byte s.nibble[i] & 7 of the pair {y:x}, x the low word.
// HIP's own version indexes a byte array in private memory, which becomes scratch traffic in the i-quant table
// lookups; v_perm_b32 is the same selection in one instruction (selector bytes 0-3 pick from its second
// operand, 4-7 from its first), so the nibble selector is spread to bytes and masked to 0-7.
__device__ __forceinline__ uint32_t byte_perm(uint32_t x, uint32_t y, uint32_t s) {
    const uint32_t sel = (s & 0x7u) | ((s & 0x70u) << 4) | ((s & 0x700u) << 8) | ((s & 0x7000u) << 12);
    return __builtin_amdgcn_perm(y, x, sel);
}

// The packed byte operations below work on all four lanes at once (SWAR, Hacker's Delight 2-18): gfx11/gfx12 have
// no packed 8-bit subtract, and a per-lane loop costs the Q3_K/Q6_K dots and the i-quant sign expansion several
// times the instructions.  kHigh is each lane's top bit.
constexpr uint32_t kHigh = 0x80808080u;

// CUDA's packed byte subtract wraps independently in each unsigned byte lane: subtract the low seven bits with the
// minuend's top bit forced on (so no borrow crosses a lane), then fix each top bit by XOR.
__device__ __forceinline__ int vsub4(int a, int b) {
    const uint32_t ua = static_cast<uint32_t>(a);
    const uint32_t ub = static_cast<uint32_t>(b);
    return static_cast<int>(((ua | kHigh) - (ub & ~kHigh)) ^ ((ua ^ ~ub) & kHigh));
}

// CUDA's packed signed-byte saturating subtract: the wrapping difference, and in each lane that overflowed (operand
// signs differ and the result's sign differs from the minuend's) the bound on the minuend's side, 0x7f or 0x80.
__device__ __forceinline__ int vsubss4(int a, int b) {
    const uint32_t ua = static_cast<uint32_t>(a);
    const uint32_t ub = static_cast<uint32_t>(b);
    const uint32_t d = static_cast<uint32_t>(vsub4(a, b));
    const uint32_t overflow = (ua ^ ub) & (ua ^ d) & kHigh;
    const uint32_t mask = (overflow >> 7) * 0xffu;
    const uint32_t bound = 0x7f7f7f7fu + ((ua & kHigh) >> 7);
    return static_cast<int>((d & ~mask) | (bound & mask));
}

// CUDA's four-lane byte compare, returning 0xff for each unequal lane and 0 otherwise: a lane of a ^ b is nonzero
// when its low seven bits carry into the top bit on adding 0x7f, or its top bit is already set.
__device__ __forceinline__ int vcmpne4(int a, int b) {
    const uint32_t t = static_cast<uint32_t>(a) ^ static_cast<uint32_t>(b);
    const uint32_t nonzero = (((t & ~kHigh) + ~kHigh) | t) & kHigh;
    return static_cast<int>((nonzero >> 7) * 0xffu);
}

// CUDA's mask argument describes participating lanes. The current kernel set uses full
// wave32 masks; reject future partial-mask use instead of silently dropping its semantics.
__device__ __forceinline__ void require_full_wave_mask(uint32_t mask) {
    if (mask != 0xffffffffu) __builtin_trap();
}

// Older HIP has no __syncwarp. A wave barrier alone does not order memory.
// Release/acquire fences cover the shared-memory exchange used by attention.
#if HIP_VERSION_MAJOR < 7
__device__ __forceinline__ void syncwarp(uint32_t mask = 0xffffffffu) {
    require_full_wave_mask(mask);
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}
#endif

template <typename T>
__device__ __forceinline__ T shfl_xor_sync(uint32_t mask, T value, int lane_mask, int width = 32) {
    require_full_wave_mask(mask);
    return __shfl_xor(value, lane_mask, width);
}

template <typename T>
__device__ __forceinline__ T shfl_down_sync(uint32_t mask, T value, unsigned delta, int width = 32) {
    require_full_wave_mask(mask);
    return __shfl_down(value, delta, width);
}

template <typename T>
__device__ __forceinline__ T shfl_up_sync(uint32_t mask, T value, unsigned delta, int width = 32) {
    require_full_wave_mask(mask);
    return __shfl_up(value, delta, width);
}

template <typename T>
__device__ __forceinline__ T shfl_sync(uint32_t mask, T value, int source_lane, int width = 32) {
    require_full_wave_mask(mask);
    return __shfl(value, source_lane, width);
}

__device__ __forceinline__ unsigned ballot_sync(uint32_t mask, int predicate) {
    require_full_wave_mask(mask);
    return static_cast<unsigned>(__ballot(predicate));
}

}  // namespace strata::hip_compat

#define __dp4a(a, b, c) (::strata::hip_compat::dp4a((a), (b), (c)))
#define __byte_perm(x, y, s) (::strata::hip_compat::byte_perm((x), (y), (s)))
#define __vsub4(a, b) (::strata::hip_compat::vsub4((a), (b)))
#define __vsubss4(a, b) (::strata::hip_compat::vsubss4((a), (b)))
#define __vcmpne4(a, b) (::strata::hip_compat::vcmpne4((a), (b)))
#define __shfl_xor_sync(...) (::strata::hip_compat::shfl_xor_sync(__VA_ARGS__))
#define __shfl_down_sync(...) (::strata::hip_compat::shfl_down_sync(__VA_ARGS__))
#define __shfl_up_sync(...) (::strata::hip_compat::shfl_up_sync(__VA_ARGS__))
#define __shfl_sync(...) (::strata::hip_compat::shfl_sync(__VA_ARGS__))
#define __ballot_sync(mask, predicate) (::strata::hip_compat::ballot_sync((mask), (predicate)))
#if HIP_VERSION_MAJOR < 7
#define __syncwarp(...) (::strata::hip_compat::syncwarp(__VA_ARGS__))
#endif
// AMD's sleep instruction accepts only 0..15; the synchronization loops use it as a
// backoff hint, so use its smallest portable delay independently of CUDA cycle counts.
#define __nanosleep(cycles) __builtin_amdgcn_s_sleep(1)

#endif  // defined(__HIPCC__)
