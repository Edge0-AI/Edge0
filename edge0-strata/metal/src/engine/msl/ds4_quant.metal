// metal/src/engine/msl/ds4_quant.metal - the DeepSeek engine's activation quantiser, the first dsv41 kernel ported.
//
// CUDA's q8 path (ds4_gguf.cu:201, q8_quant_kernel): per 32 values d = max|x| / 127 as an f16, then q = round(x / d) clamped to
// [-128, 127] as int8.  Verified bit-exact against the same arithmetic on the CPU: 4096 blocks, 131,072 values, 0 differences,
// worst scale error 4.6e-04 (half rounding) - metal/tests/dsv41/q8_quant_parity.mm.
//
// The block comes from the flat thread id (thread_position_in_grid / 32), not threadgroup_position_in_grid: the latter arrived
// as 0 in every group on this Mac's driver and made every block recompute block 0.
#include <metal_stdlib>

using namespace metal;

// x: n values; ldx: the stride in values between consecutive blocks of 32 (32 for a contiguous row)
kernel void ds4_q8_quant(device const float* x [[buffer(0)]], constant uint& ldx [[buffer(1)]],
                         device char* q [[buffer(2)]], device half* d [[buffer(3)]],
                         uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    const uint b = gid / 32u;
    const int base = int(b) * int(ldx);
    const float v = x[base + int(lane)];
    float amax = fabs(v);
    amax = max(amax, simd_shuffle_xor(amax, 16));
    amax = max(amax, simd_shuffle_xor(amax, 8));
    amax = max(amax, simd_shuffle_xor(amax, 4));
    amax = max(amax, simd_shuffle_xor(amax, 2));
    amax = max(amax, simd_shuffle_xor(amax, 1));
    const float scale = amax / 127.0f;
    const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    int iq = int(rint(v * inv));
    iq = clamp(iq, -128, 127);
    q[int(b) * 32 + int(lane)] = char(iq);
    if (lane == 0) d[int(b)] = half(scale);
}
