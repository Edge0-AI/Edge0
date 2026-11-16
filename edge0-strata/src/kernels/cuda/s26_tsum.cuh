// S26: several warp sums at once (P per-lane values -> P sums), bitwise equal to P separate xor-butterfly warp sums
// (`for (o = 16; o > 0; o >>= 1) v += shfl_xor(v, o)`).  At offset 16 a lane keeps half of its P values and adds the
// partner's copy of that half, at 8 half again, ... until one value is left, then plain xor steps for the remaining
// offsets.  Every level adds the same pair of values as the butterfly does (fp add commutes, so x + y == y + x
// bitwise), only each lane carries fewer of them: P - 1 + (5 - log2 P) cross-lane moves instead of 5 P.  Value j's sum
// ends in the lanes whose bits 4, 3, .. (MSB first) spell j: tsum_lane<P>(j) is one of them.  On AMD the moves are
// DPP row_xmask / permlanex16 instead of ds_bpermute.
#pragma once

#include <cuda_runtime.h>   // (before the first use of __forceinline__ / __float_as_int: a gfx906 build gets them from its shim)

namespace strata::kernels::s26ts {

#if defined(__HIP_PLATFORM_AMD__) && !defined(STRATA_HIP_GFX906)   // (permlanex16 / DPP row_xmask: gfx10+ only)
__device__ __forceinline__ float xmov16(float v) {
    return __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(v), __float_as_int(v), 0x76543210u, 0xfedcba98u,
                                                       false, false));
}
template <int O>
__device__ __forceinline__ float xmov(float v) {
    if constexpr (O == 16) return xmov16(v);
    else return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x160 + O, 0xf, 0xf, false));
}
#else
template <int O>
__device__ __forceinline__ float xmov(float v) { return __shfl_xor_sync(0xffffffffu, v, O); }
#endif

__host__ __device__ constexpr int pow2_ceil(int t) { return t <= 1 ? 1 : t <= 2 ? 2 : t <= 4 ? 4 : t <= 8 ? 8 : t <= 16 ? 16 : 32; }
__host__ __device__ constexpr int log2c(int p) { return p <= 1 ? 0 : p == 2 ? 1 : p == 4 ? 2 : p == 8 ? 3 : p == 16 ? 4 : 5; }

template <int O>
__device__ __forceinline__ float tplain(float x) {
    if constexpr (O >= 1) { x += xmov<O>(x); return tplain<O / 2>(x); }
    else return x;
}
// v[0..CNT) (CNT a power of two <= 32): this lane's values; returns the sum of value tsum_token<CNT>(lane)
template <int CNT, int O = 16>
__device__ __forceinline__ float tsum(float* v, int lane) {
    if constexpr (CNT > 1) {
        constexpr int H = CNT / 2;
        const bool hi = (lane & O) != 0;
#pragma unroll
        for (int j = 0; j < H; ++j) {
            const float send = hi ? v[j] : v[j + H];
            const float keep = hi ? v[j + H] : v[j];
            v[j] = keep + xmov<O>(send);
        }
        return tsum<H, O / 2>(v, lane);
    } else {
        return tplain<O>(v[0]);
    }
}
template <int P>
__device__ __forceinline__ int tsum_token(int lane) {
    constexpr int L = log2c(P);
    int k = 0;
#pragma unroll
    for (int b = 0; b < L; ++b) k = (k << 1) | ((lane >> (4 - b)) & 1);
    return k;
}
template <int P>
__device__ __forceinline__ int tsum_lane(int k) {
    constexpr int L = log2c(P);
    int lane = 0;
#pragma unroll
    for (int b = 0; b < L; ++b) lane |= ((k >> (L - 1 - b)) & 1) << (4 - b);
    return lane;
}

}  // namespace strata::kernels::s26ts
