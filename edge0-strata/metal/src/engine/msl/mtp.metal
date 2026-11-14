// metal/src/engine/msl/mtp.metal - the small steps of the MTP draft layer's input branches (include/strata/core/mtp.hpp)
// around mac-kern's kernels (GEMVs, the hyper-connection read, QSA, MoE, the head):
//
//   mtp_rms      y[r] = x[r] * rsqrt(mean(x[r]^2) + eps) * w        rows of width n (w: (1 + w) folded by the host)
//   mtp_xprep    xprep (units.metal) of rows read with a stride: the prepared activation of one stream of rows
//   mtp_combine  R[t][c][d] = h[c][t][d] + e[t][d]                  the streams of fc_hidden plus fc_embedding
//
// Prepared activations: [4 kinds][n_tok][n] then S16 [n_tok][n / 16] (units.metal's xprep layout).

#include <metal_stdlib>
using namespace metal;

struct RmsArgs { uint n; float eps; };

// grid (rows), 256 threads
kernel void mtp_rms(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]], device float* y [[buffer(2)]],
                    constant RmsArgs& a [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]],
                    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    device const float* xr = x + tg.x * a.n;
    float ss = 0.0f;
    for (uint d = tid; d < a.n; d += 256) ss = fma(xr[d], xr[d], ss);
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += part[i];
    const float rs = rsqrt(tot / float(a.n) + a.eps);
    device float* yr = y + tg.x * a.n;
    for (uint d = tid; d < a.n; d += 256) yr[d] = xr[d] * rs * w[d];
}

struct XpArgs { uint n; uint n_tok; uint plane_tok; uint stride; };

// one thread per 16 values of one row: row t at x + t * stride; out planes of plane_tok rows
kernel void mtp_xprep(device const float* x [[buffer(0)]], device float* xp [[buffer(1)]], constant XpArgs& a [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    const uint per = a.n / 16;
    if (gid >= per * a.n_tok) return;
    const uint t = gid / per, i = gid % per;
    device const float* src = x + t * a.stride + 16 * i;
    const uint plane = a.plane_tok * a.n;
    device float* raw = xp + t * a.n + 16 * i;
    float v[16];
    for (int j = 0; j < 16; ++j) v[j] = src[j];
    float s0 = 0, s1 = 0;
    for (int j = 0; j < 16; j += 2) { s0 += v[j]; s1 += v[j + 1]; }
    const float q2s[8] = {1.0f, 0.25f, 0.0625f, 0.015625f, 0.00390625f, 0.0009765625f, 0.000244140625f, 0.00006103515625f};
    const float p4s[4] = {1.0f, 0.00390625f, 1.52587890625e-05f, 5.9604644775390625e-08f};
    for (int j = 0; j < 16; ++j) {
        raw[j] = v[j];
        raw[plane + j] = v[j] * q2s[j & 7];
        raw[2 * plane + j] = v[j] * p4s[j & 3];
        raw[3 * plane + j] = (j & 1) ? v[j] * 0.00390625f : v[j];
    }
    xp[4 * plane + t * per + i] = s0 + s1;
}

struct CombArgs { uint n; uint hc; uint n_tok; uint h_stride; };   // h: stream c's rows at h + c * h_stride

kernel void mtp_combine(device const float* h [[buffer(0)]], device const float* e [[buffer(1)]], device float* R [[buffer(2)]],
                        constant CombArgs& a [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    const uint per = a.hc * a.n;
    if (gid >= a.n_tok * per) return;
    const uint t = gid / per, c = (gid % per) / a.n, d = gid % a.n;
    R[gid] = h[c * a.h_stride + t * a.n + d] + e[t * a.n + d];
}
