// metal/src/engine/msl/ds4_init.metal - the first kernels the engine asks for, ported from ds4_kernels.cu.
//
// ds4_kernels.cu:1477  __global__ void init_pre_kernel(float* pre, int T) {
//     const int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < T * 8) pre[i] = (i % 8) == 0 ? 1.0f : 0.0f;
// }
#include <metal_stdlib>

using namespace metal;

// the grid is threadgroups of 128; the flat id is what CUDA's blockIdx*blockDim+threadIdx was
kernel void ds4_init_pre(device float* pre [[buffer(0)]], constant uint& T [[buffer(1)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid < T * 8u) pre[gid] = (gid % 8u) == 0u ? 1.0f : 0.0f;
}

// ds4_kernels.cu:130  embed_kernel: v = bf16(rows[t*dim + e]); h[(t*hc + s)*dim + e] = v for each of hc copies
kernel void ds4_embed(device const ushort* rows [[buffer(0)]], device float* h [[buffer(1)]],
                      constant uint& dim [[buffer(2)]], constant uint& hc [[buffer(3)]],
                      uint2 tgp [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_threadgroup]]) {
    const uint e = tgp.x * 256u + uint(lane);
    if (e >= dim) return;
    const uint t = tgp.y;
    const uint bits = uint(rows[(size_t) t * dim + e]) << 16;   // BF16 -> F32 is the top half
    const float v = as_type<float>(bits);
    for (uint s = 0; s < hc; ++s) h[((size_t) t * hc + s) * dim + e] = v;
}

// ds4_model.cu:2737  cublasSgemm(OP_T, OP_N, n_mix, T, hcd, ...) - the mHC mixes: mixes[t][m] = h[t] . fn[m]
// one thread an output element; both operands F32, alpha = 1, beta = 0 (every call site seen)
kernel void ds4_sgemm_tn(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
                         device float* C [[buffer(2)]], constant uint& m [[buffer(3)]], constant uint& nt [[buffer(4)]],
                         constant uint& k [[buffer(5)]], constant uint& lda [[buffer(6)]], constant uint& ldb [[buffer(7)]],
                         constant uint& ldc [[buffer(8)]], uint gid [[thread_position_in_grid]]) {
    const uint total = m * nt;
    if (gid >= total) return;
    const uint t = gid / m, mm = gid % m;
    const device float* a = A + (size_t) mm * lda;    // row mm of A
    const device float* b = B + (size_t) t * ldb;     // row t of B
    float acc = 0.0f;
    for (uint i = 0; i < k; ++i) acc += a[i] * b[i];
    C[(size_t) t * ldc + mm] = acc;
}

// ds4_kernels.cu:1894  finish_kernel<false> - RMS norm (f.w), mrope (f.rd), rounding (f.quant), then store to x or dst.
// The mrope and rounding branches are not written yet; the host wrapper will decline until they are.
struct Ds4Finish {
    device float* w;
    float eps;
    int rd, pos, inverse, vpp, pos_step;
    device float* freqs;
    int quant;
    device ushort* dst;
    long dst_stride;
    device uchar* pdst;
    long pdst_stride;
};

static inline float ds4_warp_sum(float v) {
    v += simd_shuffle_xor(v, 16);
    v += simd_shuffle_xor(v, 8);
    v += simd_shuffle_xor(v, 4);
    v += simd_shuffle_xor(v, 2);
    v += simd_shuffle_xor(v, 1);
    return v;
}
static inline float ds4_block_sum(float v, threadgroup float* red, ushort lane, ushort w) {
    v = ds4_warp_sum(v);
    if (lane == 0) red[w] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (w == 0) {
        float t = lane < 8 ? red[lane] : 0.0f;
        t = ds4_warp_sum(t);
        if (lane == 0) red[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return red[0];
}

// ds4_kernels.cu:65 pow2_ceil / :88 fp8_rt / :90 e2m1_rt - the KV quantiser's scale and roundings, shared by finish_pack
// (the packed rows) and ds4_finish's rounding branch
static inline float ds4_pow2_ceil(float x) {
    const uint b = as_type<uint>(x);
    const int e = int((b >> 23) & 0xffu) - 127 + ((b & 0x7fffffu) != 0u ? 1 : 0);
    return as_type<float>(uint(e + 127) << 23);
}

static inline float ds4_e4m3_round(float v) {   // E4M3, nearest even (the input is clamped)
    const float a = fabs(v);
    if (!(a == a)) return v;
    float r;
    if (a >= 448.0f) r = 448.0f;
    else {
        const int e = int((as_type<uint>(a) >> 23) & 0xffu) - 127;
        const float step = e < -6 ? 0.001953125f : as_type<float>(uint(e - 3 + 127) << 23);
        r = a == 0.0f ? 0.0f : rint(a / step) * step;
        r = fmin(r, 448.0f);
    }
    return copysign(r, v);
}

static inline float ds4_e2m1_rt(float v) {   // E2M1 magnitudes 0 .5 1 1.5 2 3 4 6, nearest, ties to even
    const float a = fabs(v);
    float r;
    if (a <= 0.25f) r = 0.0f;
    else if (a < 0.75f) r = 0.5f;
    else if (a <= 1.25f) r = 1.0f;
    else if (a < 1.75f) r = 1.5f;
    else if (a <= 2.5f) r = 2.0f;
    else if (a < 3.5f) r = 3.0f;
    else if (a <= 5.0f) r = 4.0f;
    else r = 6.0f;
    return copysign(r, v);
}

kernel void ds4_finish(device float* x [[buffer(0)]], constant long& stride [[buffer(1)]],
                       constant uint& len [[buffer(2)]], constant Ds4Finish& f [[buffer(3)]],
                       uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                       ushort w [[simdgroup_index_in_threadgroup]], uint b [[threadgroup_position_in_grid]]) {
    threadgroup float red[8];
    device float* v = x + (long) b * stride;
    const uint nj = (len + 255u) / 256u;
    float r[32];
    float ss = 0.0f;
    for (uint j = 0; j < 32u; ++j) {
        if (j >= nj) break;
        const uint e = tid + j * 256u;
        r[j] = e < len ? v[e] : 0.0f;
        ss += r[j] * r[j];
    }
    if (f.w != nullptr) {
        const float tot = ds4_block_sum(ss, red, lane, w);
        const float inv = rsqrt(tot / float(len) + f.eps);
        for (uint j = 0; j < 32u; ++j) {
            if (j >= nj) break;
            const uint e = tid + j * 256u;
            if (e < len) r[j] = r[j] * inv * f.w[e];
        }
    }
    if (f.rd > 0) {
        // mrope: the last rd values move in (re, im) pairs a lane apart (ds4_kernels.cu:161): the partner comes from a
        // lane xor 1, the angle from the token's position and the frequency of the pair
        const uint r0 = len - uint(f.rd);
        const uint pos = uint(f.pos) + (b / uint(f.vpp)) * uint(f.pos_step);
        for (uint j = 0; j < 32u; ++j) {
            if (j >= nj) break;
            const uint e = tid + j * 256u;
            const float other = simd_shuffle_xor(r[j], 1);
            if (e >= r0 && e < len) {
                const uint i = (e - r0) >> 1;
                const float ang = float(pos) * f.freqs[i];
                const float c = cos(ang);
                const float sn = f.inverse ? -sin(ang) : sin(ang);
                r[j] = ((e - r0) & 1u) == 0u ? r[j] * c - other * sn : other * sn + r[j] * c;
            }
        }
    }
    if (f.quant != 0) {   // the KV rounding without the pack: the group's scale, then the value rounded to the format
        const uint g = f.quant == 2 ? 16u : 32u;
        for (uint j = 0; j < 32u; ++j) {
            if (j >= nj) break;
            const uint e = tid + j * 256u;
            float a = e < len ? fabs(r[j]) : 0.0f;
            for (uint o = g / 2u; o > 0u; o >>= 1) a = fmax(a, simd_shuffle_xor(a, ushort(o)));
            if (e >= len) continue;
            if (f.quant == 1) {
                const float s = ds4_pow2_ceil(fmax(a, 1e-4f) * (1.0f / 448.0f));
                r[j] = ds4_e4m3_round(fmin(fmax(r[j] / s, -448.0f), 448.0f)) * s;
            } else if (f.quant == 2) {
                const float s = ds4_e4m3_round(fmax(a, 6.0f * 0.001953125f) / 6.0f);
                r[j] = ds4_e2m1_rt(fmin(fmax(r[j] / s, -6.0f), 6.0f)) * s;
            } else {
                const float s = ds4_pow2_ceil(fmax(a, 6.0f * 1.1754943508222875e-38f) * (1.0f / 6.0f));
                r[j] = ds4_e2m1_rt(fmin(fmax(r[j] / s, -6.0f), 6.0f)) * s;
            }
        }
    }
    for (uint j = 0; j < 32u; ++j) {
        if (j >= nj) break;
        const uint e = tid + j * 256u;
        if (e >= len) continue;
        if (f.dst != nullptr) {
            const uint bits = as_type<uint>(r[j]);
            f.dst[(long) b * f.dst_stride + (long) e] = ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);   // BF16 round to nearest even, the vector's own row
        } else {
            v[e] = r[j];
        }
    }
}

// ds4_kernels.cu:1677  hc_post_inplace_kernel: h[t][k][e] = post[t][k]*y[t][e] + sum_j comb[t][j][k]*h_old[t][j][e]
kernel void ds4_hc_post(device const float* y [[buffer(0)]], device float* h [[buffer(1)]],
                        device const float* post [[buffer(2)]], device const float* comb [[buffer(3)]],
                        constant uint& dim [[buffer(4)]], constant uint& hc [[buffer(5)]],
                        uint2 tgp [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_threadgroup]]) {
    const uint e = tgp.x * 256u + uint(lane);
    if (e >= dim) return;
    const uint t = tgp.y;
    device float* r = h + (size_t) t * hc * dim;
    float rv[8];
    for (uint j = 0; j < 8u; ++j) if (j < hc) rv[j] = r[(size_t) j * dim + e];
    const float yv = y[(size_t) t * dim + e];
    for (uint k = 0; k < 8u; ++k) {
        if (k >= hc) break;
        float a = post[(size_t) t * 8u + k] * yv;
        for (uint j = 0; j < 8u; ++j) if (j < hc) a += comb[(size_t) t * 64u + j * hc + k] * rv[j];
        r[(size_t) k * dim + e] = a;
    }
}

// ds4_kernels.cu:1692  hc_mix_scale_kernel: mixes[t][i] *= rsqrt(sum(h[t])^2 / hcdim + eps), one threadgroup a token
kernel void ds4_hc_mix_scale(device const float* h [[buffer(0)]], device float* mixes [[buffer(1)]],
                             constant uint& hcdim [[buffer(2)]], constant uint& n_mix [[buffer(3)]],
                             constant float& eps [[buffer(4)]], uint b [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                             ushort w [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[8];
    const device float* x = h + (size_t) b * hcdim;
    float ss = 0.0f;
    for (uint k = tid * 4u; k < hcdim; k += 256u * 4u) {
        const float4 hv = *(const device float4*) (x + k);
        ss += hv.x * hv.x + hv.y * hv.y + hv.z * hv.z + hv.w * hv.w;
    }
    ss = ds4_block_sum(ss, red, lane, w);
    const float inv = rsqrt(ss / float(hcdim) + eps);
    if (tid < n_mix) mixes[(size_t) b * 32u + tid] *= inv;
}

// ds4_kernels.cu:1017  hc_mix_kernel: mixes[t][m] = (h[t] . fn[m]) * rsqrt(mean(h[t]^2) + eps), one threadgroup a (m, t)
kernel void ds4_hc_mix(device const float* h [[buffer(0)]], device const float* fn [[buffer(1)]],
                       device float* mixes [[buffer(2)]], constant uint& hcdim [[buffer(3)]],
                       constant float& eps [[buffer(4)]], uint2 tgp [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                       ushort w [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[8];
    const uint m = tgp.x, t = tgp.y;
    const device float* x = h + (size_t) t * hcdim;
    const device float* f = fn + (size_t) m * hcdim;
    float acc = 0.0f, ss = 0.0f;
    for (uint k = tid * 4u; k < hcdim; k += 256u * 4u) {
        const float4 hv = *(const device float4*) (x + k);
        const float4 fv = *(const device float4*) (f + k);
        acc += hv.x * fv.x + hv.y * fv.y + hv.z * fv.z + hv.w * fv.w;
        ss += hv.x * hv.x + hv.y * hv.y + hv.z * hv.z + hv.w * hv.w;
    }
    acc = ds4_block_sum(acc, red, lane, w);
    ss = ds4_block_sum(ss, red, lane, w);
    if (tid == 0) mixes[(size_t) t * 32u + m] = acc * rsqrt(ss / float(hcdim) + eps);
}

// hc_mix for a prompt chunk: one threadgroup a token for four mixes a pass (the token's row is read once per four mixes instead of once per
// mix).  Every mix's partial sums keep the single-mix kernel's order, so the result is bit-identical.
kernel void ds4_hc_mix_all(device const float* h [[buffer(0)]], device const float* fn [[buffer(1)]],
                           device float* mixes [[buffer(2)]], constant uint& hcdim [[buffer(3)]],
                           constant float& eps [[buffer(4)]], constant uint& n_mix [[buffer(5)]],
                           uint t [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                           ushort w [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[8];
    const device float* x = h + (size_t) t * hcdim;
    float ss = 0.0f;
    for (uint k = tid * 4u; k < hcdim; k += 256u * 4u) {
        const float4 hv = *(const device float4*) (x + k);
        ss += hv.x * hv.x + hv.y * hv.y + hv.z * hv.z + hv.w * hv.w;
    }
    ss = ds4_block_sum(ss, red, lane, w);
    const float inv = rsqrt(ss / float(hcdim) + eps);
    for (uint m0 = 0; m0 < n_mix; m0 += 4u) {
        const uint nm = min(4u, n_mix - m0);
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        const device float* f0 = fn + (size_t) m0 * hcdim;
        const device float* f1 = f0 + (nm > 1u ? hcdim : 0u);
        const device float* f2 = f0 + (nm > 2u ? 2u * hcdim : 0u);
        const device float* f3 = f0 + (nm > 3u ? 3u * hcdim : 0u);
        for (uint k = tid * 4u; k < hcdim; k += 256u * 4u) {
            const float4 hv = *(const device float4*) (x + k);
            const float4 v0 = *(const device float4*) (f0 + k);
            const float4 v1 = *(const device float4*) (f1 + k);
            const float4 v2 = *(const device float4*) (f2 + k);
            const float4 v3 = *(const device float4*) (f3 + k);
            a0 += hv.x * v0.x + hv.y * v0.y + hv.z * v0.z + hv.w * v0.w;
            a1 += hv.x * v1.x + hv.y * v1.y + hv.z * v1.z + hv.w * v1.w;
            a2 += hv.x * v2.x + hv.y * v2.y + hv.z * v2.z + hv.w * v2.w;
            a3 += hv.x * v3.x + hv.y * v3.y + hv.z * v3.z + hv.w * v3.w;
        }
        a0 = ds4_block_sum(a0, red, lane, w);
        a1 = ds4_block_sum(a1, red, lane, w);
        a2 = ds4_block_sum(a2, red, lane, w);
        a3 = ds4_block_sum(a3, red, lane, w);
        if (tid == 0) {
            mixes[(size_t) t * 32u + m0] = a0 * inv;
            if (nm > 1u) mixes[(size_t) t * 32u + m0 + 1u] = a1 * inv;
            if (nm > 2u) mixes[(size_t) t * 32u + m0 + 2u] = a2 * inv;
            if (nm > 3u) mixes[(size_t) t * 32u + m0 + 3u] = a3 * inv;
        }
    }
}

// hc mix + split in one dispatch.  The mix is ds4_hc_mix_all's (four mixes a pass, every one keeping the single-mix
// kernel's summation order, so the values are bit-identical), then the same threadgroup runs the sinkhorn split of the
// 24 mixes into pre / post / comb.  Two dispatches become one, and nothing else changes.
kernel void ds4_hc_mix_split(device const float* h [[buffer(0)]], device const float* fn [[buffer(1)]],
                             device float* mixes [[buffer(2)]], constant uint& hcdim [[buffer(3)]],
                             constant float& eps [[buffer(4)]], constant uint& n_mix [[buffer(5)]],
                             device const float* scale [[buffer(6)]], device const float* base [[buffer(7)]],
                             device float* pre [[buffer(8)]], device float* post [[buffer(9)]],
                             device float* comb [[buffer(10)]], constant uint& hc [[buffer(11)]],
                             constant uint& iters [[buffer(12)]], constant float& hc_eps [[buffer(13)]],
                             uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]], ushort wid [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[8];
    const device float* x = h + (size_t) t * hcdim;
    float ss = 0.0f;
    for (uint k = tid * 4u; k < hcdim; k += 256u * 4u) {
        const float4 hv = *(const device float4*) (x + k);
        ss += hv.x * hv.x + hv.y * hv.y + hv.z * hv.z + hv.w * hv.w;
    }
    ss = ds4_block_sum(ss, red, lane, wid);
    const float inv = rsqrt(ss / float(hcdim) + eps);
    for (uint m0 = 0; m0 < n_mix; m0 += 4u) {
        const uint nm = min(4u, n_mix - m0);
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        const device float* f0 = fn + (size_t) m0 * hcdim;
        const device float* f1 = f0 + (nm > 1u ? hcdim : 0u);
        const device float* f2 = f0 + (nm > 2u ? 2u * hcdim : 0u);
        const device float* f3 = f0 + (nm > 3u ? 3u * hcdim : 0u);
        for (uint k = tid * 4u; k < hcdim; k += 256u * 4u) {
            const float4 hv = *(const device float4*) (x + k);
            const float4 v0 = *(const device float4*) (f0 + k);
            const float4 v1 = *(const device float4*) (f1 + k);
            const float4 v2 = *(const device float4*) (f2 + k);
            const float4 v3 = *(const device float4*) (f3 + k);
            a0 += hv.x * v0.x + hv.y * v0.y + hv.z * v0.z + hv.w * v0.w;
            a1 += hv.x * v1.x + hv.y * v1.y + hv.z * v1.z + hv.w * v1.w;
            a2 += hv.x * v2.x + hv.y * v2.y + hv.z * v2.z + hv.w * v2.w;
            a3 += hv.x * v3.x + hv.y * v3.y + hv.z * v3.z + hv.w * v3.w;
        }
        a0 = ds4_block_sum(a0, red, lane, wid);
        a1 = ds4_block_sum(a1, red, lane, wid);
        a2 = ds4_block_sum(a2, red, lane, wid);
        a3 = ds4_block_sum(a3, red, lane, wid);
        if (tid == 0) {
            mixes[(size_t) t * 32u + m0] = a0 * inv;
            if (nm > 1u) mixes[(size_t) t * 32u + m0 + 1u] = a1 * inv;
            if (nm > 2u) mixes[(size_t) t * 32u + m0 + 2u] = a2 * inv;
            if (nm > 3u) mixes[(size_t) t * 32u + m0 + 3u] = a3 * inv;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // the split: ds4_hc_split4's math, in the first warp (every lane of it takes the same path)
    const device float* mx = mixes + (size_t) t * 32u;
    if (tid < 4u) {
        pre[(size_t) t * 8u + tid] = 1.0f / (1.0f + exp(-(mx[tid] * scale[0] + base[tid]))) + hc_eps;
        post[(size_t) t * 8u + tid] = 2.0f / (1.0f + exp(-(mx[4u + tid] * scale[1] + base[4u + tid])));
    }
    if (tid < 32u) {
        const bool on = lane < 16u;
        float c = on ? mx[8u + lane] * scale[2] + base[8u + lane] : -3.402823466e+38f;
        float m = fmax(c, simd_shuffle_xor(c, ushort(1)));
        m = fmax(m, simd_shuffle_xor(m, ushort(2)));
        c = on ? exp(c - m) : 0.0f;
        float rs = c + simd_shuffle_xor(c, ushort(1));
        rs += simd_shuffle_xor(rs, ushort(2));
        c = c / rs + hc_eps;
        c = on ? c : 0.0f;
        float cs = c + simd_shuffle_xor(c, ushort(4));
        cs += simd_shuffle_xor(cs, ushort(8));
        c = c / (cs + hc_eps);
        for (uint it = 1; it < iters; ++it) {
            float r = c + simd_shuffle_xor(c, ushort(1));
            r += simd_shuffle_xor(r, ushort(2));
            c = c / (r + hc_eps);
            float q = c + simd_shuffle_xor(c, ushort(4));
            q += simd_shuffle_xor(q, ushort(8));
            c = c / (q + hc_eps);
        }
        if (on) comb[(size_t) t * 64u + lane] = c;
    }
}

// ds4_kernels.cu:268  hc_split_kernel - the sinkhorn normalisation of the hyper-connection mix.
// This is the SCALAR path of the CUDA kernel (lane 0 of the threadgroup does the token, a local 8x8 matrix), applied for every
// hc including 4: it computes the same mathematics as the warp-lane version, only without the shuffles.  Correctness first -
// the lane-parallel hc == 4 version can be added behind a profile.
kernel void ds4_hc_split(device const float* mixes [[buffer(0)]], device const float* scale [[buffer(1)]],
                         device const float* base [[buffer(2)]], device float* pre [[buffer(3)]],
                         device float* post [[buffer(4)]], device float* comb [[buffer(5)]],
                         constant uint& hc [[buffer(6)]], constant uint& iters [[buffer(7)]],
                         constant float& eps [[buffer(8)]], uint b [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_threadgroup]]) {
    if (lane != 0u) return;
    const device float* mx = mixes + (size_t) b * 32u;
    device float* pr = pre + (size_t) b * 8u;
    device float* po = post + (size_t) b * 8u;
    device float* cb = comb + (size_t) b * 64u;
    for (uint i = 0; i < 8u; ++i) {
        if (i >= hc) break;
        pr[i] = 1.0f / (1.0f + exp(-(mx[i] * scale[0] + base[i]))) + eps;
        po[i] = 2.0f / (1.0f + exp(-(mx[hc + i] * scale[1] + base[hc + i])));
    }
    float c[8][8];
    for (uint j = 0; j < 8u; ++j) {
        if (j >= hc) break;
        float m = -3.402823466e+38f;
        for (uint k = 0; k < 8u; ++k) {
            if (k >= hc) break;
            c[j][k] = mx[2u * hc + j * hc + k] * scale[2] + base[2u * hc + j * hc + k];
            m = fmax(m, c[j][k]);
        }
        float s = 0.0f;
        for (uint k = 0; k < 8u; ++k) {
            if (k >= hc) break;
            c[j][k] = exp(c[j][k] - m);
            s += c[j][k];
        }
        for (uint k = 0; k < 8u; ++k) {
            if (k >= hc) break;
            c[j][k] = c[j][k] / s + eps;
        }
    }
    for (uint k = 0; k < 8u; ++k) {
        if (k >= hc) break;
        float s = 0.0f;
        for (uint j = 0; j < 8u; ++j) if (j < hc) s += c[j][k];
        for (uint j = 0; j < 8u; ++j) if (j < hc) c[j][k] /= s + eps;
    }
    for (uint it = 1; it < iters; ++it) {
        for (uint j = 0; j < 8u; ++j) {
            if (j >= hc) break;
            float s = 0.0f;
            for (uint k = 0; k < 8u; ++k) if (k < hc) s += c[j][k];
            for (uint k = 0; k < 8u; ++k) if (k < hc) c[j][k] /= s + eps;
        }
        for (uint k = 0; k < 8u; ++k) {
            if (k >= hc) break;
            float s = 0.0f;
            for (uint j = 0; j < 8u; ++j) if (j < hc) s += c[j][k];
            for (uint j = 0; j < 8u; ++j) if (j < hc) c[j][k] /= s + eps;
        }
    }
    for (uint j = 0; j < 8u; ++j) {
        if (j >= hc) break;
        for (uint k = 0; k < 8u; ++k) { if (k >= hc) break; cb[j * hc + k] = c[j][k]; }
    }
}

// ds4_kernels.cu:1938  moe_sort - count each virtual slot's picks, prefix-sum the counts, then fill in token order.
// kPicks = 8 (ds4_kernels.cuh:19): idx is [T][8], so q -> (t = q/8, i = q%8).
kernel void ds4_bounds_scan(device int* bounds [[buffer(0)]], constant uint& n_vs [[buffer(1)]],
                            uint tid [[thread_index_in_threadgroup]]) {
    if (tid != 0u) return;
    bounds[0] = 0;
    for (uint s = 0; s < n_vs; ++s) bounds[s + 1] += bounds[s];
}

// simd_ballot returns a simd_vote, which does not cast to an integer in this toolchain; the same counts come from the
// prefix-sum and sum primitives, which give a lane its offset inside its simdgroup and the group its total.
static inline int ds4_warp_prefix(bool hit) { return simd_prefix_exclusive_sum(hit ? 1 : 0); }
static inline int ds4_warp_total(bool hit) { return simd_sum(hit ? 1 : 0); }

kernel void ds4_moe_sort_count(device const int* idx [[buffer(0)]], constant uint& T [[buffer(1)]],
                               constant uint& kk [[buffer(2)]], device const int* vs [[buffer(3)]],
                               device int* bounds [[buffer(4)]], uint slot [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
    threadgroup int warp_n[8];
    threadgroup int base_s;
    const uint wp = tid / 32u, lane = tid % 32u;
    if (tid == 0u) base_s = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n = T * kk;
    for (uint q = tid; q < n; q += 256u) {
        const uint t = q / kk, i = q - t * kk;
        const bool hit = vs[idx[t * 8u + i]] == int(slot);
        // one vote a warp: the running offset needs each warp's count, accumulated in slot order
        const int wtot = ds4_warp_total(hit);
        if ((tid % 32u) == 0u) warp_n[wp] = wtot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int total = 0;
        for (uint w = 0; w < 8u; ++w) total += warp_n[w];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) base_s += total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) bounds[slot + 1] = base_s;
}

kernel void ds4_moe_sort_fill(device const int* idx [[buffer(0)]], constant uint& T [[buffer(1)]],
                              constant uint& kk [[buffer(2)]], device const int* vs [[buffer(3)]],
                              device int* bounds [[buffer(4)]], device int* row_tok [[buffer(5)]],
                              device int* pos [[buffer(6)]], uint slot [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]]) {
    threadgroup int warp_n[8];
    threadgroup int base_s;
    const uint wp = tid / 32u, lane = tid % 32u;
    if (tid == 0u) base_s = bounds[slot];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n = T * kk;
    for (uint q = tid; q < n; q += 256u) {
        const uint t = q / kk, i = q - t * kk;
        const bool hit = vs[idx[t * 8u + i]] == int(slot);
        const int wpre = ds4_warp_prefix(hit), wtot = ds4_warp_total(hit);
        if ((tid % 32u) == 0u) warp_n[wp] = wtot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int before = 0, total = 0;
        for (uint w = 0; w < 8u; ++w) { if (w < wp) before += warp_n[w]; total += warp_n[w]; }
        if (hit) {
            const int r = base_s + before + wpre;
            row_tok[r] = int(t);
            pos[t * 8u + i] = r;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) base_s += total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ds4_kernels.cu:1690  engram_dequant_tok_kernel - one hash-table row a threadgroup: IQ4_NL blocks to F32.
// fmt 2 (this model's Engram): an 18-byte block is fp16 d then 16 packed nibbles; the low nibble of q[j] is value j and the
// high nibble value j + 16, each an index into ggml-common.h's kvalues_iq4nl (ds4_kernels.cu:1280).
constant int ds4_kIq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

kernel void ds4_engram_dequant(device const uchar* src [[buffer(0)]], constant ulong& per [[buffer(1)]],
                               constant uint& cols [[buffer(2)]], constant uint& hd [[buffer(3)]],
                               device float* out [[buffer(4)]], constant int& fmt [[buffer(5)]],
                               uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint t = r / cols, c = r - t * cols;
    const device uchar* rows = src + (size_t) t * per + (size_t) c * hd;
    if (fmt == 2) {
        for (uint e = tid; e < hd; e += 256u) {
            const device uchar* b = rows + (size_t) (e / 32u) * 18u;
            const uint q = b[2u + (e & 15u)];
            const int v = (e & 16u) ? int(q >> 4) : int(q & 15u);
            const ushort dh = ushort(b[0]) | (ushort(b[1]) << 8);
            out[(size_t) r * hd + e] = float(as_type<half>(dh)) * float(ds4_kIq4nl[v]);
        }
        return;
    }
    // fmt 1 (Q6_K) and fmt 0 (E4M3 x scale) are not ported: write NaN, so a wrong path shows as NaN in the output rather than
    // as a silent zero.
    for (uint e = tid; e < hd; e += 256u) out[(size_t) r * hd + e] = NAN;
}

// ds4_kernels.cu:341  hc_pre_norm_kernel - per token: a = sum_s mix[s]*h[token][s][e], one RMS over dim, then x = a*inv
// (times w when given).  One threadgroup a token, 1024 threads; a thread holds at most 8 values (dim <= 8*1024 = 8192).
kernel void ds4_hc_pre_norm(device const float* h [[buffer(0)]], device const float* mix [[buffer(1)]],
                            device const float* w [[buffer(2)]], constant uint& dim [[buffer(3)]],
                            constant uint& hc [[buffer(4)]], constant float& eps [[buffer(5)]],
                            device float* x [[buffer(6)]], constant uint& has_w [[buffer(7)]],
                            uint b [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]], ushort wg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[32];
    const device float* hh = h + (size_t) b * hc * dim;
    const device float* mm = mix + (size_t) b * 8u;
    device float* xx = x + (size_t) b * dim;
    float r[8];
    float ss = 0.0f;
    uint n = 0;
    for (uint e = tid; e < dim; e += 1024u, ++n) {
        float a = 0.0f;
        for (uint s = 0; s < 8u; ++s) if (s < hc) a += mm[s] * hh[(size_t) s * dim + e];
        if (n < 8u) r[n] = a;
        ss += a * a;
    }
    // a 1024-thread reduction: 32 simdgroups, each into red[], then the first simdgroup sums those
    float v = ds4_warp_sum(ss);
    if (lane == 0u) red[wg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (wg == 0u) {
        float t = lane < 32u ? red[lane] : 0.0f;
        t = ds4_warp_sum(t);
        if (lane == 0u) red[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = rsqrt(red[0] / float(dim) + eps);
    n = 0;
    for (uint e = tid; e < dim; e += 1024u, ++n) {
        if (n >= 8u) break;
        xx[e] = has_w != 0u ? r[n] * inv * w[e] : r[n];
    }
}

// ds4_gguf.cu:186  q8_quant_kernel: a warp a 32-group - amax across the warp, sc = amax/127, q = rint(v/sc), d = sc.
// grid = (ceil((k/32) / 8 warps), rows), threadgroup = 256 (8 warps of 32).
kernel void ds4_q8_quant(device const float* x [[buffer(0)]], constant ulong& ldx [[buffer(1)]],
                         constant uint& k [[buffer(2)]], device char* q [[buffer(3)]],
                         device float* d [[buffer(4)]], uint2 tgp [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]]) {
    const uint warp = tid / 32u, lane = tid % 32u;
    const uint b = tgp.x * 8u + warp;         // the 32-group index
    const uint r = tgp.y;
    if ((size_t) b * 32u >= (size_t) k) return;
    const device float* xr = x + (size_t) r * ldx;
    const float v = xr[(size_t) b * 32u + lane];
    const float am = simd_max(fabs(v));
    const float sc = am / 127.0f;
    q[(size_t) r * k + (size_t) b * 32u + lane] = char(am > 0.0f ? rint(v / sc) : 0.0f);
    if (lane == 0u) d[(size_t) r * (k / 32u) + b] = sc;
}

// ds4_gguf.cu:371  mx_mv_kernel - the MXFP4 expert matrix-vector.
// grid = (ceil(n / 8 warps), jobs), threadgroup = 256.  A warp a row; a lane strides the 32-value blocks; NC columns of
// activations share each decoded block (the T=8 window), then a warp sum and y[c][row] = alpha*v + bias[row].
// Block layout (kMxBytes = 17 a 32 values): the E8M0 scale byte, then 16 bytes whose low nibble is value j and high nibble
// value j+16, each an index into the 16-entry codebook {0,1,2,3,4,6,8,12} and its negatives.
struct Ds4MxJob {
    device const void* w;
    ulong ld;           // bytes a row
    device const float* x;
    ulong xcs;
    device float* y;
    ulong ycs;
    device const float* bias;
    float alpha;
    int k, n, ncol;
    device const char* xq;
    device const float* xd;
};

static inline float ds4_e8m0(uint e) {   // 2^(e - 128), ggml's GGML_E8M0_TO_FP32_HALF
    return as_type<float>(e < 2u ? (0x00200000u << e) : ((e - 1u) << 23));
}

static inline void ds4_mx_block(device const uchar* p, threadgroup const float* lut, thread float* w) {
    const float d = ds4_e8m0(uint(p[0]));
    for (uint j = 0; j < 16u; ++j) {
        const uint q = uint(p[1u + j]);
        w[j] = lut[q & 15u] * d;
        w[j + 16u] = lut[q >> 4] * d;
    }
}

kernel void ds4_mx_mv(device const uchar* jobs [[buffer(0)]],
                      uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float lut[16];
    if (tid < 16u) {
        const float v[8] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f};
        lut[tid] = tid < 8u ? v[tid] : -v[tid - 8u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device Ds4MxJob& b = *(const device Ds4MxJob*) (jobs + (size_t) tgp.y * sizeof(Ds4MxJob));
    const uint row = tgp.x * 8u + (tid >> 5);
    if (int(row) >= b.n) return;
    const device uchar* w = (const device uchar*) b.w + (size_t) row * b.ld;
    thread float acc[8];
    for (uint c = 0; c < 8u; ++c) acc[c] = 0.0f;
    const uint nblk = uint(b.k) / 32u;
    for (uint kb = lane; kb < nblk; kb += 32u) {
        float wv[32];
        ds4_mx_block(w + (size_t) kb * 17u, lut, wv);
        for (uint c = 0; c < 8u; ++c) {
            if (int(c) >= b.ncol) break;
            const device float* x = b.x + c * b.xcs + (size_t) kb * 32u;
            float a = acc[c];
            for (uint j = 0; j < 32u; ++j) a = fma(wv[j], x[j], a);
            acc[c] = a;
        }
    }
    for (uint c = 0; c < 8u; ++c) {
        if (int(c) >= b.ncol) break;
        float v = acc[c];
        v += simd_shuffle_xor(v, 16);
        v += simd_shuffle_xor(v, 8);
        v += simd_shuffle_xor(v, 4);
        v += simd_shuffle_xor(v, 2);
        v += simd_shuffle_xor(v, 1);
        if (lane == 0u) b.y[(size_t) c * b.ycs + row] = b.alpha * v + (b.bias != nullptr ? b.bias[row] : 0.0f);
    }
}

// ds4_gguf.cu:320  q8_mv_kernel - the Q8_0 matvec against quantised activations.
// grid = (ceil(n / 8), jobs), threadgroup 256: a warp a row, a lane strides the 34-byte blocks (fp16 d + 32 int8), the dot is
// integer (CUDA's __dp4a in four-value groups; here the same sum written out), then acc += (dw * xd[c]) * sum.
// The staged-prefetch (ST) and vector variants of the CUDA kernel are prefetch strategies only - this reads the blocks directly.
kernel void ds4_q8_mv(device const uchar* jobs [[buffer(0)]], uint2 tgp [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const device Ds4MxJob& b = *(const device Ds4MxJob*) (jobs + (size_t) tgp.y * sizeof(Ds4MxJob));
    const uint row = tgp.x * 8u + (tid >> 5);
    if (int(row) >= b.n) return;
    const device uchar* w = (const device uchar*) b.w + (size_t) row * b.ld;
    const uint nb = uint(b.k) / 32u;
    float acc[8];
    for (uint c = 0; c < 8u; ++c) acc[c] = 0.0f;
    for (uint kb = lane; kb < nb; kb += 32u) {
        const device uchar* p = w + (size_t) kb * 34u;
        const ushort dh = ushort(p[0]) | (ushort(p[1]) << 8);
        const float dw = float(as_type<half>(dh));
        int wv[32];
        for (uint j = 0; j < 32u; ++j) wv[j] = int((char) p[2u + j]);
        for (uint c = 0; c < 8u; ++c) {
            if (int(c) >= b.ncol) break;
            const device char* xq = b.xq + (size_t) c * b.k + (size_t) kb * 32u;
            int sum = 0;
            for (uint j = 0; j < 32u; ++j) sum += wv[j] * int(xq[j]);
            acc[c] = fma(dw * b.xd[(size_t) c * nb + kb], float(sum), acc[c]);
        }
    }
    for (uint c = 0; c < 8u; ++c) {
        if (int(c) >= b.ncol) break;
        float v = acc[c];
        v += simd_shuffle_xor(v, 16);
        v += simd_shuffle_xor(v, 8);
        v += simd_shuffle_xor(v, 4);
        v += simd_shuffle_xor(v, 2);
        v += simd_shuffle_xor(v, 1);
        if (lane == 0u) b.y[(size_t) c * b.ycs + row] = b.alpha * v + (b.bias != nullptr ? b.bias[row] : 0.0f);
    }
}

// ds4_kernels.cu:1092  ring_to_ext_kernel - one ring row of the KV cache to BF16: each value an E4M3 code times its group's
// E8M0 power of two (row[e] and row[d + e/32]); rows before the window start write zero.
static inline float ds4_e4m3(uint b) {   // E4M3 (bias 7, no infinities, max 448): sign, 4-bit exp, 3-bit mantissa
    const float s = (b & 0x80u) ? -1.0f : 1.0f;
    const uint ex = (b >> 3) & 0xfu, m = b & 7u;
    if (ex == 0u) return s * float(m) * (1.0f / 8.0f) * (1.0f / 64.0f);
    if (ex == 15u && m == 7u) return NAN;
    return s * exp2(float(ex) - 7.0f) * (1.0f + float(m) * 0.125f);
}
static inline float ds4_ue8m0(uint b) { return as_type<float>(b << 23); }   // 2^(b - 127)
static inline ushort ds4_f2bf16(float f) {
    const uint u = as_type<uint>(f);
    return ushort((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}


// ds4_kernels.cu:1104  ext_to_ring_kernel - the window's BF16 rows packed back into the E4M3 ring, each 32-value group at the
// smallest power-of-two scale it fits under (k), so v / 2^k is an E4M3 value times a power of two.
static inline uint ds4_e4m3_code(float v, int k) {
    const uint sign = (v < 0.0f) ? 0x80u : 0u;
    float a = fabs(v) * exp2(-float(k));
    if (a == 0.0f) return sign;
    int e = int(floor(log2(a)));
    int mant = int(round((a / exp2(float(e)) - 1.0f) * 8.0f));
    int ex = e + 7;
    if (mant >= 8) { mant = 0; ++ex; }
    if (ex <= 0) { mant = 0; ex = 0; }
    if (ex > 15) { mant = 7; ex = 15; }
    return sign | (uint(ex) << 3) | uint(mant & 7);
}

kernel void ds4_ext_to_ring(device const ushort* ext [[buffer(0)]], device uchar* ring [[buffer(1)]],
                            device uchar* backup [[buffer(2)]], constant long& p0 [[buffer(3)]],
                            constant uint& n [[buffer(4)]], constant uint& T [[buffer(5)]],
                            constant uint& d [[buffer(6)]], constant ulong& rb [[buffer(7)]], constant uint& has_backup [[buffer(8)]],
                            uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    if (T - t > n) return;                       // overwritten by a later row of the window anyway
    device uchar* row = ring + (size_t) ((size_t(p0) + t) % n) * rb;
    if (has_backup != 0u) {   // (a bound buffer is never a null pointer here: the host says whether there is a backup)
        for (size_t i = tid; i < rb; i += 256u) backup[(size_t) t * rb + i] = row[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const device ushort* src = ext + (size_t) (n + t) * d;
    for (uint e0 = 0; e0 < d; e0 += 256u) {
        const uint e = e0 + tid;
        const float v = e < d ? as_type<float>(uint(src[e]) << 16) : 0.0f;   // (BF16 bits, not half)
        uint a = as_type<uint>(v) & 0x7fffffffu;             // magnitudes ordered as bits
        for (uint o = 16u; o > 0u; o >>= 1u) a = max(a, simd_shuffle_xor(a, o));
        const int k = a == 0u ? 0 : int(a >> 23) - 135 + (((a & 0x7fffffu) > 0x600000u) ? 1 : 0);
        if (e >= d) continue;
        row[e] = uchar(ds4_e4m3_code(v, k));
        if ((e & 31u) == 0u) row[d + e / 32u] = uchar(k + 127);
    }
}


// ds4_kernels.cu:916  unpack32 - 16 threads expand one MXFP4 KV row (288 B at d=512) into 32 BF16 values a thread:
// 16 code bytes at part*16, two E4M3 scale bytes at 256 + part*2, and the 4-bit code is itself an FP4 bit pattern
// (e2m1_val14: sign and exponent/mantissa bits spliced into a half).
static inline float ds4_e2m1_val14(uint c) {
    const ushort h = (ushort) (((c & 8u) << 12) | ((c & 7u) << 9));
    return float(as_type<half>(h));
}

// ds4_kernels.cu:937  unpack_sel_kernel - the selected rows of the compressed cache into the attention scratch, one row a
// group of 16 threads.  n_sel comes either from the generated schedule (gen: min(kcap, (pos+1)/R)) or from nsel[t].
kernel void ds4_unpack_sel(device const uchar* ckv [[buffer(0)]], device const int* sel [[buffer(1)]],
                           constant uint& sel_stride [[buffer(2)]], constant int& gen [[buffer(3)]],
                           constant long& p0 [[buffer(4)]], constant int& t0 [[buffer(5)]],
                           constant int& R [[buffer(6)]], constant int& kcap [[buffer(7)]],
                           device const int* nsel [[buffer(8)]], constant uint& cap [[buffer(9)]],
                           device ushort* out [[buffer(10)]], constant ulong& rb [[buffer(11)]],
                           uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint t = tgp.y;
    int n_sel;
    if (gen != 0) {
        const long pa = p0 + long(t0) + long(t);
        n_sel = R > 0 ? int(min(long(kcap), (pa + 1) / long(R))) : 0;
    } else {
        n_sel = nsel[t];
    }
    const uint i = tgp.x * 16u + (tid >> 4u), part = tid & 15u;
    if (int(i) >= n_sel) return;
    const size_t r = (size_t) t * cap + i;
    const device uchar* rr = ckv + (size_t) sel[(size_t) t * sel_stride + i] * rb;
    const uint4 cw = *(const device uint4*) (rr + part * 16u);
    const float k0 = ds4_e4m3(uint(rr[256u + part * 2u])) * 16384.0f;
    const float k1 = ds4_e4m3(uint(rr[256u + part * 2u + 1u])) * 16384.0f;
    const uint wv[4] = {cw.x, cw.y, cw.z, cw.w};
    uint o[16];
    for (uint k = 0; k < 16u; ++k) {
        const uint b = wv[k >> 2] >> (8u * (k & 3u));
        const float ks = k < 8u ? k0 : k1;
        const float lo = ds4_e2m1_val14(b) * ks, hi = ds4_e2m1_val14(b >> 4) * ks;
        o[k] = (as_type<uint>(lo) >> 16) | (as_type<uint>(hi) & 0xffff0000u);
    }
    device uint* dst = (device uint*) (out + r * 512u + part * 32u);
    for (uint q = 0; q < 4u; ++q) {
        dst[q * 4u + 0u] = o[4u * q + 0u];
        dst[q * 4u + 1u] = o[4u * q + 1u];
        dst[q * 4u + 2u] = o[4u * q + 2u];
        dst[q * 4u + 3u] = o[4u * q + 3u];
    }
}


// ds4_kernels.cu:1772  moe_combine_kernel - a token's expert outputs back into its row: for each pick whose routed row is in
// [r0, r1), acc += w[t][i] * y[r - r0][e]; untouched when no pick falls in range.
kernel void ds4_moe_combine(device float* acc [[buffer(0)]], device const float* y [[buffer(1)]],
                            device const int* pos [[buffer(2)]], device const float* w [[buffer(3)]],
                            constant int& kk [[buffer(4)]], constant int& r0 [[buffer(5)]],
                            constant int& r1 [[buffer(6)]], constant uint& dim [[buffer(7)]],
                            uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint e = tgp.x * 256u + tid;
    if (e >= dim) return;
    const uint t = tgp.y;
    float a = acc[(size_t) t * dim + e];
    bool any = false;
    for (int i = 0; i < 8; ++i) {
        if (i >= kk) break;
        const int r = pos[(size_t) t * 8u + i];
        if (r < r0 || r >= r1) continue;
        a = fma(w[(size_t) t * 8u + i], y[(size_t) (r - r0) * dim + e], a);
        any = true;
    }
    if (any) acc[(size_t) t * dim + e] = a;
}


// ds4_kernels.cu:362  hc_post_kernel - like hc_post_inplace, but reads a separate residual and writes a separate output:
// out[t][k][e] = post[t][k]*y[t][e] + sum_j comb[t][j][k]*res[t][j][e]
kernel void ds4_hc_post_v(device const float* y [[buffer(0)]], device const float* res [[buffer(1)]],
                          device const float* post [[buffer(2)]], device const float* comb [[buffer(3)]],
                          device float* out [[buffer(4)]], constant uint& dim [[buffer(5)]],
                          constant uint& hc [[buffer(6)]], uint2 tgp [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]]) {
    const uint i = tgp.x * 256u + tid;
    if (i >= dim * hc) return;
    const uint t = tgp.y, k = i / dim, e = i - k * dim;
    const device float* r = res + (size_t) t * hc * dim;
    float a = post[(size_t) t * 8u + k] * y[(size_t) t * dim + e];
    for (uint j = 0; j < 8u; ++j) {
        if (j >= hc) break;
        a += comb[(size_t) t * 64u + j * hc + k] * r[(size_t) j * dim + e];
    }
    out[(size_t) t * hc * dim + i] = a;
}


// ds4_kernels.cu:441  gemv_f32_kernel - a warp a row of F32 weights: float4 loads of W[row] and x[t], four FMAs a step,
// lane-strided over k/4, then the warp sum into y[t][row].
kernel void ds4_gemv_f32(device const float* W [[buffer(0)]], device const float* x [[buffer(1)]],
                         constant uint& n [[buffer(2)]], constant uint& kk [[buffer(3)]],
                         device float* y [[buffer(4)]], constant uint& ldy [[buffer(5)]],
                         uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]]) {
    const uint row = tgp.x * 8u + (tid >> 5);
    if (row >= n) return;
    const uint t = tgp.y;
    const device float4* w = (const device float4*) (W + (size_t) row * kk);
    const device float4* xv = (const device float4*) (x + (size_t) t * kk);
    float acc = 0.0f;
    for (uint c = lane; c < kk / 4u; c += 32u) {
        const float4 a = w[c], b = xv[c];
        acc = fma(a.x, b.x, acc);
        acc = fma(a.y, b.y, acc);
        acc = fma(a.z, b.z, acc);
        acc = fma(a.w, b.w, acc);
    }
    acc += simd_shuffle_xor(acc, 16);
    acc += simd_shuffle_xor(acc, 8);
    acc += simd_shuffle_xor(acc, 4);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 1);
    if (lane == 0u) y[(size_t) t * ldy + row] = acc;
}


// ds4_kernels.cu:1353  route_publish_kernel - the MoE router: sc = sqrt(softplus(logit)), bs = sc + bias, then k rounds of a
// block-wide argmax (a strided max, a warp reduction, a 16-warp pass), the picks' sc summed into the weights, and each pick's
// device address published to plan_ptr.  The prefetch requests (req) are NOT published here yet: that is the schedule, not the
// answer, and it can be turned on separately.
kernel void ds4_route_publish(device const float* logits [[buffer(0)]], device const float* bias [[buffer(1)]],
                              constant uint& n [[buffer(2)]], constant uint& kk [[buffer(3)]],
                              constant float& route_scale [[buffer(4)]], device const ulong* vram [[buffer(5)]],
                              device int* idx [[buffer(6)]], device float* w [[buffer(7)]],
                              device ulong* plan_ptr [[buffer(8)]], device const float* bias_vl [[buffer(9)]],
                              constant uint& vl [[buffer(10)]], constant uint& has_vl [[buffer(11)]],
                              uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sc[1024];
    threadgroup float bs[1024];
    threadgroup float bv[16];
    threadgroup int bi[16];
    threadgroup int pick[8];
    const device float* lg = logits + (size_t) t * 1024u;
    const device float* b = (has_vl != 0u && ((vl >> t) & 1u) != 0u) ? bias_vl : bias;
    for (uint i = tid; i < n; i += 512u) {
        const float l = lg[i];
        const float sp = l > 20.0f ? l : log(1.0f + exp(l));
        sc[i] = sqrt(sp);
        bs[i] = sc[i] + b[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint lane = tid & 31u, wp = tid >> 5u;
    for (uint r = 0; r < 8u; ++r) {
        if (r >= kk) break;
        float best = -3.402823466e+38f;
        int bix = 0x7fffffff;
        for (uint i = tid; i < n; i += 512u) {
            if (bs[i] > best || (bs[i] == best && int(i) < bix)) { best = bs[i]; bix = int(i); }
        }
        for (uint o = 16u; o > 0u; o >>= 1u) {
            const float ov = simd_shuffle_xor(best, (ushort) o);
            const int oi = simd_shuffle_xor(bix, (ushort) o);
            if (ov > best || (ov == best && oi < bix)) { best = ov; bix = oi; }
        }
        if (lane == 0u && wp < 16u) { bv[wp] = best; bi[wp] = bix; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float bb = -3.402823466e+38f;
            int ix = 0x7fffffff;
            for (uint q = 0; q < 16u; ++q) {
                if (bv[q] > bb || (bv[q] == bb && bi[q] < ix)) { bb = bv[q]; ix = bi[q]; }
            }
            pick[r] = ix;
            bs[ix] = -3.402823466e+38f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        float sum = 0.0f;
        for (uint r = 0; r < 8u; ++r) { if (r >= kk) break; sum += sc[pick[r]]; }
        for (uint r = 0; r < 8u; ++r) {
            if (r >= kk) break;
            const uint p_ = uint(pick[r]);
            const float wv = sc[p_] / (sum + 1e-20f) * route_scale;
            idx[(size_t) t * 8u + r] = int(p_);
            w[(size_t) t * 8u + r] = wv;
            plan_ptr[(size_t) t * 8u + r] = vram[p_];
        }
    }
}


// ds4_kernels.cu  swiglu_kernel - the expert MLP's activation: h = silu(g) * u, with the clamp when lim > 0
kernel void ds4_swiglu(device const float* g [[buffer(0)]], device const float* u [[buffer(1)]],
                       constant uint& n [[buffer(2)]], constant float& lim [[buffer(3)]],
                       device float* h [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    float gv = g[gid], uv = u[gid];
    if (lim > 0.0f) {                                   // both inputs clamped, then the gate itself
        gv = fmin(gv, lim);
        uv = fmin(fmax(uv, -lim), lim);
    }
    h[gid] = (gv / (1.0f + exp(-gv))) * uv;             // silu(g) * u
}

kernel void ds4_ring_to_ext(device const uchar* ring [[buffer(0)]], device ushort* ext [[buffer(1)]],
                            constant long& p0 [[buffer(2)]], constant uint& n [[buffer(3)]],
                            constant uint& d [[buffer(4)]], constant ulong& rb [[buffer(5)]],
                            uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const long p = p0 - long(n) + long(r);
    const device uchar* row = ring + (size_t) (p >= 0 ? (p % long(n)) : 0) * rb;
    for (uint e = tid; e < d; e += 256u) {
        ext[(size_t) r * d + e] = p >= 0 ? ds4_f2bf16(ds4_e4m3(uint(row[e])) * ds4_ue8m0(uint(row[d + e / 32u]))) : ushort(0);
    }
}

// ---------------------------------------------------------------- ds4_gguf.cu:577 mx_combine_kernel
//
// A token's expert outputs summed in plan order (an expert off the device adds 0), the CPU lane's part added when its
// flag is set, then the shared expert's - ds4_gguf.cu:169's combine().  Grid (n_embd / 128, T), 128 threads.
struct Ds4Comb {
    device const ulong* plan_ptr;   // [T][8]: nonzero when the device holds that expert
    device const float* plan_w;     // [T][8]
    device const float* part;       // [entries][n_embd]
    device const float* cpu_part;   // [T][cpu_stride], or null
    device const int* cpu_flag;     // [T], or null
    device const float* sh_out;     // [T][n_embd], or null
    device float* out;
    long cpu_stride, out_stride;
    int k, n_embd;
};

kernel void ds4_mx_combine(constant Ds4Comb& a [[buffer(0)]], uint2 tgp [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]]) {
    const uint r = tgp.x * 128u + tid;
    if (int(r) >= a.n_embd) return;
    const int t = int(tgp.y);
    float m = 0.0f;
    for (int i = 0; i < a.k; ++i) {   // kPicks <= 8
        const int e = t * 8 + i;
        const float pi = a.plan_ptr[e] != 0ul ? a.part[(size_t) e * (size_t) a.n_embd + r] : 0.0f;
        m = fma(a.plan_w[t * 8 + i], pi, m);
    }
    if (a.cpu_flag != nullptr && a.cpu_flag[t] != 0) m += a.cpu_part[(size_t) t * (size_t) a.cpu_stride + r];
    m += a.sh_out != nullptr ? a.sh_out[(size_t) t * (size_t) a.n_embd + r] : 0.0f;
    a.out[(size_t) t * (size_t) a.out_stride + r] = m;
}

// ---------------------------------------------------------------- ds4_kernels.cu:1286 engram_dequant_kernel / :1311 engram_gate_kernel
//
// The Engram tables' row decoder and gate (this model's tables are fmt 2: IQ4_NL, an 18-byte block of 32 values whose
// nibbles index ds4_kIq4nl above).
kernel void ds4_engram_dequant_rows(device const uchar* rows [[buffer(0)]], constant uint& hd [[buffer(1)]],
                                    device float* out [[buffer(2)]], constant int& fmt [[buffer(3)]],
                                    uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    if (fmt == 2) {
        const device uchar* row = rows + (size_t) r * (size_t) hd;
        for (uint e = tid; e < hd; e += 256u) {
            const device uchar* b = row + (size_t) (e / 32u) * 18u;
            const uint q = b[2u + (e & 15u)];
            const int v = (e & 16u) ? int(q >> 4) : int(q & 15u);
            const ushort dh = ushort(b[0]) | (ushort(b[1]) << 8);
            out[(size_t) r * (size_t) hd + e] = float(as_type<half>(dh)) * float(ds4_kIq4nl[v]);
        }
        return;
    }
    // fmt 1 (Q6_K) and fmt 0 (E4M3 x scale) are not ported: NaN rather than a silent zero (as ds4_engram_dequant does)
    for (uint e = tid; e < hd; e += 256u) out[(size_t) r * (size_t) hd + e] = NAN;
}

// ds4_kernels.cu's block_sum for a 1024-thread threadgroup: a warp sum a warp, then every warp sums the 32 partials (the
// same result in all of them, so no warp diverges at a barrier)
static inline float ds4_block_sum32(float v, threadgroup float* red, ushort lane, ushort w) {
    v = ds4_warp_sum(v);
    if (lane == 0) red[w] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = lane < 32u ? red[lane] : 0.0f;
    t = ds4_warp_sum(t);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return t;
}

kernel void ds4_engram_gate(device float* h [[buffer(0)]], device const float* kv [[buffer(1)]],
                            device const float* w [[buffer(2)]], constant uint& hc [[buffer(3)]],
                            constant uint& dim [[buffer(4)]], constant float& eps [[buffer(5)]],
                            uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]], ushort wg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[32];
    const uint k = tgp.x, t = tgp.y;
    device float* hk = h + ((size_t) t * hc + (size_t) k) * (size_t) dim;
    const device float* kvt = kv + (size_t) t * (size_t) (hc + 1u) * (size_t) dim;
    const device float* key = kvt + (size_t) k * (size_t) dim;
    const device float* val = kvt + (size_t) hc * (size_t) dim;
    const device float* wk = w + (size_t) k * (size_t) dim;
    float sh = 0.0f, sk = 0.0f, sd = 0.0f;
    for (uint e = tid; e < dim; e += 1024u) {
        const float hv = hk[e], kve = key[e];
        sh += hv * hv;
        sk += kve * kve;
        sd += hv * wk[e] * kve;
    }
    sh = ds4_block_sum32(sh, red, lane, wg);
    sk = ds4_block_sum32(sk, red, lane, wg);
    sd = ds4_block_sum32(sd, red, lane, wg);
    const float rstd = rsqrt(sh / float(dim) + eps) * rsqrt(sk / float(dim) + eps);
    const float dot = sd * rstd * rsqrt(float(dim));
    const float sq = copysign(sqrt(fmax(fabs(dot), 1e-6f)), dot);
    const float gate = 1.0f / (1.0f + exp(-sq));
    for (uint e = tid; e < dim; e += 1024u) hk[e] = hk[e] + gate * val[e];
}

// ds4_gguf.cu:72  bf16_mv_kernel<NC, true> - a BF16 matrix's matvec against F32 activations (the engram WKV here, and the
// BF16 dense matrices the jobs' f == 0 names).  Same frame as ds4_mx_mv above: a warp a row, a lane strides the row; here a
// lane 8 BF16 weights (16 bytes = one uint4) and 8 activations, so the row is walked in 256-value steps.
// A pair's F32 is bit-exact - the low half zeroed - which is bf16_mv_kernel's __uint_as_float(w << 16).
kernel void ds4_bf16_mv(device const uchar* jobs [[buffer(0)]], uint2 tgp [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const device Ds4MxJob& b = *(const device Ds4MxJob*) (jobs + (size_t) tgp.y * sizeof(Ds4MxJob));
    const uint row = tgp.x * 8u + (tid >> 5);
    if (int(row) >= b.n) return;
    const device ushort* w = (const device ushort*) b.w + (size_t) row * b.ld;
    thread float acc[16];   // (the CUDA template's largest NC; ncol cuts the rest)
    for (uint c = 0; c < 16u; ++c) acc[c] = 0.0f;
    for (uint i = uint(lane) * 8u; i < uint(b.k); i += 256u) {
        const uint4 q = *(const device uint4*) (w + i);
        const float wf[8] = {as_type<float>(q.x << 16), as_type<float>(q.x & 0xffff0000u),
                             as_type<float>(q.y << 16), as_type<float>(q.y & 0xffff0000u),
                             as_type<float>(q.z << 16), as_type<float>(q.z & 0xffff0000u),
                             as_type<float>(q.w << 16), as_type<float>(q.w & 0xffff0000u)};
        for (uint c = 0; c < 16u; ++c) {
            if (int(c) >= b.ncol) break;
            const device float4* xp = (const device float4*) (b.x + c * b.xcs + (size_t) i);
            const float4 x0 = xp[0], x1 = xp[1];
            float a = acc[c];
            a = fma(wf[0], x0.x, a);
            a = fma(wf[1], x0.y, a);
            a = fma(wf[2], x0.z, a);
            a = fma(wf[3], x0.w, a);
            a = fma(wf[4], x1.x, a);
            a = fma(wf[5], x1.y, a);
            a = fma(wf[6], x1.z, a);
            a = fma(wf[7], x1.w, a);
            acc[c] = a;
        }
    }
    for (uint c = 0; c < 16u; ++c) {
        if (int(c) >= b.ncol) break;
        float v = acc[c];
        v += simd_shuffle_xor(v, 16);
        v += simd_shuffle_xor(v, 8);
        v += simd_shuffle_xor(v, 4);
        v += simd_shuffle_xor(v, 2);
        v += simd_shuffle_xor(v, 1);
        if (lane == 0u) b.y[(size_t) c * b.ycs + row] = b.alpha * v + (b.bias != nullptr ? b.bias[row] : 0.0f);
    }
}

// ---------------------------------------------------------------- ds4_kernels.cu:524 compress_kernel / :582 index_score_kernel
//
// The compressor (ratio > 1 layers): each token's kv / score row into its slot, and every ratio'th token pooled by a
// per-value softmax over the slots, scaled by a row RMS and the norm weights.  Grid (1), 512 threads (d <= 1024 values).
static inline float ds4_block_sum_n(float v, threadgroup float* red, ushort lane, ushort w, uint nw) {
    v = ds4_warp_sum(v);
    if (lane == 0) red[w] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = lane < nw ? red[lane] : 0.0f;
    t = ds4_warp_sum(t);   // (every warp sums the same partials: no divergence at a barrier)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return t;
}

kernel void ds4_compress(device const float* kv [[buffer(0)]], device const float* score [[buffer(1)]],
                         device float* kv_state [[buffer(2)]], device float* score_state [[buffer(3)]],
                         constant int& ratio [[buffer(4)]], constant int& p0 [[buffer(5)]], constant int& T [[buffer(6)]],
                         constant uint& d [[buffer(7)]], device const float* wn [[buffer(8)]],
                         constant float& eps [[buffer(9)]], device float* latent [[buffer(10)]],
                         uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                         ushort wg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[32];
    const float kfmax = -3.4028234663852886e+38f;   // -FLT_MAX, as the CUDA writes it
    int g = 0;
    for (int t = 0; t < T; ++t) {
        const int slot = (p0 + t) % ratio;
        const bool pool = (p0 + t + 1) % ratio == 0;
        for (uint e = tid; e < d; e += 512u) {
            kv_state[(size_t) slot * d + e] = kv[(size_t) t * d + e];
            score_state[(size_t) slot * d + e] = score[(size_t) t * d + e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!pool) continue;
        float val[2] = {0.0f, 0.0f};   // (d <= 2 * 512)
        float ss = 0.0f;
        uint n = 0;
        for (uint e = tid; e < d; e += 512u, ++n) {
            float m = kfmax;
            for (int s = 0; s < ratio; ++s) m = fmax(m, score_state[(size_t) s * d + e]);
            float den = 0.0f;
            for (int s = 0; s < ratio; ++s) den += exp(score_state[(size_t) s * d + e] - m);
            float acc = 0.0f;
            for (int s = 0; s < ratio; ++s)
                acc += kv_state[(size_t) s * d + e] * (exp(score_state[(size_t) s * d + e] - m) / den);
            val[n] = acc;
            ss += acc * acc;
        }
        const float inv = rsqrt(ds4_block_sum_n(ss, red, lane, wg, 16u) / float(d) + eps);
        n = 0;
        for (uint e = tid; e < d; e += 512u, ++n) latent[(size_t) g * d + e] = val[n] * inv * wn[e];
        ++g;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ---------------------------------------------------------------- ds4_kernels.cu:582 index_score_kernel<128, KF>
//
// A thread a position: its key row (kfmt 3: 64 bytes of E2M1 codes (value 2i in the low nibble) then 4 UE8M0 scales; kfmt
// 0: BF16) against every head's query (a threadgroup tile), relu, then the heads' weighted sum.  Grid (ceil(n / 256), T).
struct Ds4Rows {   // ds4_kernels.cuh:115 Rows (kWin = 16)
    int n[16];
    int k[16];
    int gen, R, kcap, ccap, blk;
    long p0;
};

static inline void ds4_rows_at(constant Ds4Rows& r, int t, thread int& n, thread int& k) {
    if (r.gen == 0) {
        n = r.n[t];
        k = r.k[t];
        return;
    }
    const int nn = int((r.p0 + t + 1) / r.R);
    const int cb = min(r.ccap, (nn + r.blk - 1) / r.blk);
    switch (r.gen) {
        case 1: n = nn; k = min(r.kcap, nn); break;
        case 2: n = nn; k = cb; break;
        case 3: n = cb * r.blk; k = min(r.kcap, nn); break;
        default: n = (nn + r.blk - 1) / r.blk; k = r.kcap; break;
    }
}

kernel void ds4_index_score(device const float* q [[buffer(0)]], device const float* wq [[buffer(1)]],
                            device const uchar* kc [[buffer(2)]], device const int* cand [[buffer(3)]],
                            device float* out [[buffer(4)]], constant Ds4Rows& lim [[buffer(5)]],
                            constant int& heads [[buffer(6)]], constant int& n [[buffer(7)]],
                            constant int& cand_stride [[buffer(8)]], constant int& blk [[buffer(9)]],
                            constant long& out_stride [[buffer(10)]], constant int& kfmt [[buffer(11)]],
                            constant int& has_cand [[buffer(12)]], uint2 tgp [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]]) {
    constexpr int HD = 128;
    threadgroup float qs[32 * HD + 32];
    const int t = int(tgp.y);
    for (uint i = tid; i < uint(heads) * uint(HD); i += 256u) qs[i] = q[(size_t) t * (size_t) heads * HD + i];
    for (uint i = tid; i < uint(heads); i += 256u) qs[uint(heads) * HD + i] = wq[(size_t) t * 64 + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int i = int(tgp.x) * 256 + int(tid);
    if (i >= n) return;
    device float* o = out + (long) t * out_stride;
    int ln, lk;
    ds4_rows_at(lim, t, ln, lk);
    int p = i;
    if (has_cand != 0) {
        if (i >= lk * blk) {
            o[i] = -INFINITY;
            return;
        }
        p = cand[(size_t) t * (size_t) cand_stride + (size_t) (i / blk)] * blk + i % blk;
    }
    if (p >= ln) {
        o[i] = -INFINITY;
        return;
    }
    const size_t kb = kfmt == 3 ? (size_t) ((HD / 2 + HD / 32 + 15) / 16 * 16) : (size_t) HD * 2;
    const device uchar* kr = kc + (size_t) p * kb;
    float acc[32];
    for (uint h = 0; h < 32u; ++h) acc[h] = 0.0f;
    for (int c = 0; c < HD; c += 8) {
        float kvv[8];
        if (kfmt == 3) {   // 8 codes in 4 bytes, the group's power-of-two scale (2^(b - 127))
            const uint cw = uint(kr[c / 2]) | (uint(kr[c / 2 + 1]) << 8) | (uint(kr[c / 2 + 2]) << 16) |
                            (uint(kr[c / 2 + 3]) << 24);
            const float ks = as_type<float>(uint(kr[HD / 2 + c / 32]) << 23);
            for (uint x = 0; x < 8u; ++x) kvv[x] = ds4_e2m1_val14(cw >> (4 * x)) * 16384.0f * ks;
        } else {
            const device ushort* b = (const device ushort*) (kr + (size_t) c * 2);
            for (uint x = 0; x < 8u; ++x) kvv[x] = as_type<float>(uint(b[x]) << 16);
        }
        for (uint h = 0; h < 32u; ++h) {
            if (int(h) >= heads) break;
            threadgroup const float* qh = qs + h * HD + c;
            for (uint x = 0; x < 8u; ++x) acc[h] += qh[x] * kvv[x];
        }
    }
    float s = 0.0f;
    for (uint h = 0; h < 32u; ++h) {
        if (int(h) >= heads) break;
        s += fmax(acc[h], 0.0f) * qs[uint(heads) * HD + h];
    }
    o[i] = s;
}

// The sparse-attention indexer with the GPU's matrix units.  The scalar ds4_index_score spends 16.3 s of a 32K prompt
// (30 % of it) doing 32 head-dots a (query, key) pair one FMA at a time; this stages 16 fp4 keys and one query's q as
// halves and runs the 32 x 128 x 16 product through simdgroup_matrix, which measured 3.17-6.10x faster in the standalone
// harness (tools/ds4/idx_harness/idx_mma) with every output matching the scalar path to half precision (<= 7e-4
// relative).  Same arithmetic, same order per (query, key) up to that rounding.
//
// The fp4 weights are staged as e2m1 * ks (<= 12, half-safe) with the 16384 applied once in the epilogue - the full
// 16384 * ks reaches 2e5 and overflows half.
kernel void ds4_index_score_mma(device const float* q [[buffer(0)]], device const float* wq [[buffer(1)]],
                                device const uchar* kc [[buffer(2)]], device float* out [[buffer(4)]],
                                constant Ds4Rows& lim [[buffer(5)]], constant int& heads [[buffer(6)]],
                                constant int& n [[buffer(7)]], constant int& T [[buffer(8)]],
                                constant long& out_stride [[buffer(9)]],
                                uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int HD = 128;
    constexpr int KB = 80;      // the fp4 row: 64 code bytes + 4 scale bytes, padded to 80
    constexpr int NK = 16;      // keys a threadgroup scores (two key-blocks of 8)
    threadgroup half qst[32 * HD];      // [head][dim]
    threadgroup half kst[HD * NK];      // [dim][key], e2m1 * ks
    threadgroup float part[4][NK];      // the four head-blocks' partial sums
    threadgroup float accs[8][64];      // a simdgroup's 8x8 tile
    const int k0 = int(tgp.x) * NK;
    const int t = int(tgp.y);
    if (t >= T) return;
    for (uint i = tid; i < uint(heads) * uint(HD); i += 256u) qst[i] = half(q[(size_t) t * (size_t) heads * HD + i]);
    for (int key = int(tid) / 8; key < NK; key += 32) {
        const int idx = k0 + key;
        if (idx >= n) continue;
        const device uchar* kr = kc + (size_t) idx * (size_t) KB;
        for (int c = 0; c < HD; c += 8) {
            const uint cw = uint(kr[c / 2]) | (uint(kr[c / 2 + 1]) << 8) | (uint(kr[c / 2 + 2]) << 16) |
                            (uint(kr[c / 2 + 3]) << 24);
            const float ks = as_type<float>(uint(kr[HD / 2 + c / 32]) << 23);
            for (uint x = 0; x < 8u; ++x) kst[(c + int(x)) * NK + key] = half(ds4_e2m1_val14(cw >> (4 * x)) * ks);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int hb = int(sg) / 2, kb = int(sg) % 2;   // eight heads x eight keys a simdgroup
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int c = 0; c < HD; c += 8) {
        simdgroup_half8x8 a, b;
        simdgroup_load(a, qst, HD, ulong2(ulong(c), ulong(hb) * 8ul));
        simdgroup_load(b, kst, NK, ulong2(ulong(kb) * 8ul, ulong(c)));
        simdgroup_multiply_accumulate(acc, a, b, acc);
    }
    simdgroup_store(acc, accs[sg], 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const uint lane = tid % 32u;
    if (lane < 8u) {
        const int key = kb * 8 + int(lane);
        float ssum = 0.0f;
        for (uint h = 0; h < 8u; ++h) {
            const int hh = hb * 8 + int(h);
            if (hh >= heads) break;
            ssum += fmax(accs[sg][h * 8 + lane] * 16384.0f, 0.0f) * wq[(size_t) t * 64 + hh];
        }
        part[hb][key] = ssum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < uint(NK)) {
        const int key = int(tid);
        const int idx = k0 + key;
        if (idx >= n) return;
        int ln, lk;
        ds4_rows_at(lim, t, ln, lk);
        device float* o = out + (long) t * out_stride;
        o[idx] = idx >= ln ? -INFINITY : (part[0][key] + part[1][key] + part[2][key] + part[3][key]);
    }
}

// ---------------------------------------------------------------- ds4_kernels.cu:142 finish_kernel<true> (the KV pack)
//
// The same load, RMS and mrope as ds4_finish above, then the row written as the cache's packed codes (kv_row_bytes): the
// quantiser's scale byte a group (32 values, or 16 for the 4-bit MXFP4-style rows), the codes themselves from bit 0 up -
// for the 4-bit formats two values a byte, the second from the next lane.  Written from the CUDA of the same name.



static inline uint ds4_e2m1_code(float q) {   // (bit 3 the sign, -0 kept; the magnitudes as 0 .. 7)
    const float a = fabs(q);
    const uint m = a <= 1.5f ? uint(a * 2.0f) : a <= 2.0f ? 4u : a <= 3.0f ? 5u : a <= 4.0f ? 6u : 7u;
    return m | ((as_type<uint>(q) >> 31) << 3);
}

kernel void ds4_finish_pack(device float* x [[buffer(0)]], constant long& stride [[buffer(1)]],
                            constant uint& len [[buffer(2)]], constant Ds4Finish& f [[buffer(3)]],
                            uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                            ushort w [[simdgroup_index_in_threadgroup]], uint b [[threadgroup_position_in_grid]]) {
    threadgroup float red[8];
    device float* v = x + (long) b * stride;
    const uint nj = (len + 255u) / 256u;
    float r[32];
    float ss = 0.0f;
    for (uint j = 0; j < 32u; ++j) {
        if (j >= nj) break;
        const uint e = tid + j * 256u;
        r[j] = e < len ? v[e] : 0.0f;
        ss += r[j] * r[j];
    }
    if (f.w != nullptr) {
        const float tot = ds4_block_sum(ss, red, lane, w);
        const float inv = rsqrt(tot / float(len) + f.eps);
        for (uint j = 0; j < 32u; ++j) {
            if (j >= nj) break;
            const uint e = tid + j * 256u;
            if (e < len) r[j] = r[j] * inv * f.w[e];
        }
    }
    if (f.rd > 0) {
        const uint r0 = len - uint(f.rd);
        const uint pos = uint(f.pos) + (b / uint(f.vpp)) * uint(f.pos_step);
        for (uint j = 0; j < 32u; ++j) {
            if (j >= nj) break;
            const uint e = tid + j * 256u;
            const float other = simd_shuffle_xor(r[j], 1);
            if (e >= r0 && e < len) {
                const uint i = (e - r0) >> 1;
                const float ang = float(pos) * f.freqs[i];
                const float c = cos(ang);
                const float sn = f.inverse ? -sin(ang) : sin(ang);
                r[j] = ((e - r0) & 1u) == 0u ? r[j] * c - other * sn : other * sn + r[j] * c;
            }
        }
    }
    // the packed row: a group's scale and the group's codes as the format's (quant 1: d E4M3 codes at pr[e] and d/32
    // scale bytes at pr[len + e/32]; 2: E2M1 pairs at pr[e/2] with E4M3 scales at pr[len/2 + e/16]; 3: E2M1 pairs with
    // UE8M0 scales at pr[len/2 + e/32])
    const uint g = f.quant == 2 ? 16u : 32u;
    device uchar* pr = f.pdst + (long) b * f.pdst_stride;
    for (uint j = 0; j < 32u; ++j) {
        if (j >= nj) break;
        const uint e = tid + j * 256u;
        float a = e < len ? fabs(r[j]) : 0.0f;
        for (uint o = g / 2u; o > 0u; o >>= 1) a = fmax(a, simd_shuffle_xor(a, ushort(o)));
        float q = 0.0f, s = 1.0f;
        if (e < len) {
            if (f.quant == 1) {
                s = ds4_pow2_ceil(fmax(a, 1e-4f) * (1.0f / 448.0f));
                q = ds4_e4m3_round(fmin(fmax(r[j] / s, -448.0f), 448.0f));
            } else if (f.quant == 2) {
                s = ds4_e4m3_round(fmax(a, 6.0f * 0.001953125f) / 6.0f);
                q = ds4_e2m1_rt(fmin(fmax(r[j] / s, -6.0f), 6.0f));
            } else {
                s = ds4_pow2_ceil(fmax(a, 6.0f * 1.1754943508222875e-38f) * (1.0f / 6.0f));
                q = ds4_e2m1_rt(fmin(fmax(r[j] / s, -6.0f), 6.0f));
            }
            r[j] = q * s;
        }
        if (f.quant == 1) {
            if (e < len) pr[e] = uchar(ds4_e4m3_code(q, 0));
        } else {
            const uint c = ds4_e2m1_code(q);
            const uint hi = simd_shuffle_down(c, 1);   // (value e + 1: the next lane's)
            if (e < len && (e & 1u) == 0u) pr[e >> 1] = uchar(c | (hi << 4));
        }
        if (e < len && e % g == 0u)
            pr[(f.quant == 1 ? len : len / 2u) + e / g] =
                uchar(f.quant == 2 ? ds4_e4m3_code(s, 0) : (as_type<uint>(s) >> 23));
    }
    device ushort* d = f.dst != nullptr ? f.dst + (long) b * f.dst_stride : nullptr;
    for (uint j = 0; j < 32u; ++j) {
        if (j >= nj) break;
        const uint e = tid + j * 256u;
        if (e >= len) continue;
        if (d != nullptr) {
            const uint bits = as_type<uint>(r[j]);
            d[e] = ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);   // BF16 round to nearest even
        }
    }
}

// ds4_kernels.cu:401  gemv_f16_rows_kernel - F16 weights against F32 activations (the indexer's wproj here): a warp a row
// of the matrix, a lane 8 halves (16 bytes, the uint4 above) and 8 activations a step, kRowsT = 2 tokens a block.  Grid
// (ceil(n / 8), ceil(T / 2)), threadgroup 256.
kernel void ds4_gemv_f16_rows(device const ushort* W [[buffer(0)]], device const float* x [[buffer(1)]],
                              constant ulong& ldx [[buffer(2)]], constant int& n [[buffer(3)]],
                              constant int& k [[buffer(4)]], constant int& T [[buffer(5)]],
                              constant float& alpha [[buffer(6)]], device float* y [[buffer(7)]],
                              constant ulong& ldy [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int kRowsT = 2;
    const int row = int(tgp.x) * 8 + int(tid >> 5);
    const int t0 = int(tgp.y) * kRowsT, nt = min(kRowsT, T - t0);
    if (row >= n) return;
    const device ushort* w = W + (size_t) row * (size_t) k;
    float acc[kRowsT];
    for (int t = 0; t < kRowsT; ++t) acc[t] = 0.0f;
    for (int c = int(lane) * 8; c < k; c += 32 * 8) {
        const uint4 q = *(const device uint4*) (w + c);
        const float2 f0 = float2(as_type<half2>(q.x)), f1 = float2(as_type<half2>(q.y));
        const float2 f2 = float2(as_type<half2>(q.z)), f3 = float2(as_type<half2>(q.w));
        const float2 fs[4] = {f0, f1, f2, f3};
        float xv[kRowsT][8];
        for (int t = 0; t < kRowsT; ++t)
            if (t < nt) {
                const device float4* xp = (const device float4*) (x + (size_t) (t0 + t) * ldx + (size_t) c);
                const float4 a4 = xp[0], b4 = xp[1];
                xv[t][0] = a4.x;
                xv[t][1] = a4.y;
                xv[t][2] = a4.z;
                xv[t][3] = a4.w;
                xv[t][4] = b4.x;
                xv[t][5] = b4.y;
                xv[t][6] = b4.z;
                xv[t][7] = b4.w;
            }
        for (int i = 0; i < 4; ++i) {
            const float2 f = fs[i];
            for (int t = 0; t < kRowsT; ++t)
                if (t < nt) acc[t] += f.x * xv[t][2 * i] + f.y * xv[t][2 * i + 1];
        }
    }
    for (int t = 0; t < kRowsT; ++t)
        if (t < nt) {
            float a = acc[t];
            a += simd_shuffle_xor(a, 16);
            a += simd_shuffle_xor(a, 8);
            a += simd_shuffle_xor(a, 4);
            a += simd_shuffle_xor(a, 2);
            a += simd_shuffle_xor(a, 1);
            if (lane == 0u) y[(size_t) (t0 + t) * ldy + row] = alpha * a;
        }
}

// ---------------------------------------------------------------- ds4_kernels.cu:650 topk_kernel
//
// A token's k largest scores: four 8-bit radix passes over the order keys of the row (a threadgroup of 1024, shared
// histograms), then the picks written in index order by a scan a pass, so the attention sums them in the same order every
// run and the ties kept are the lowest positions.
static inline uint ds4_ord_key(float f) {
    const uint b = as_type<uint>(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

kernel void ds4_topk(device const float* vals [[buffer(0)]], constant long& stride [[buffer(1)]],
                     constant Ds4Rows& r [[buffer(2)]], device const int* cand [[buffer(3)]],
                     constant int& cand_stride [[buffer(4)]], constant int& blk [[buffer(5)]],
                     device int* out [[buffer(6)]], constant int& out_stride [[buffer(7)]],
                     constant int& has_cand [[buffer(8)]], uint tgp [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                     ushort wid [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint hist[256];
    threadgroup uint prefix_s, mask_s;
    threadgroup int remaining_s, cnt_gt, cnt_eq;
    threadgroup int wgt[32], weq[32];
    const int t = int(tgp);
    int n, k;
    ds4_rows_at(r, t, n, k);
    device const float* v = vals + (long) t * stride;
    device int* o = out + (size_t) t * (size_t) out_stride;
    device const int* cd = has_cand != 0 ? cand + (size_t) t * (size_t) cand_stride : nullptr;
    if (n <= 0) return;
    if (k >= n) {
        for (uint i = tid; i < uint(n); i += 1024u)
            o[i] = cd != nullptr ? cd[i / uint(blk)] * blk + int(i % uint(blk)) : int(i);
        return;
    }
    if (tid == 0u) {
        prefix_s = 0u;
        mask_s = 0u;
        remaining_s = k;
        cnt_gt = 0;
        cnt_eq = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (uint i = tid; i < 256u; i += 1024u) hist[i] = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint prefix = prefix_s, mask = mask_s;
        for (int i = int(tid); i < n; i += 1024) {
            const uint u = ds4_ord_key(v[i]);
            if ((u & mask) == prefix) {
                threadgroup atomic_uint* hp = (threadgroup atomic_uint*) &hist[(u >> shift) & 0xffu];
                atomic_fetch_add_explicit(hp, 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            const int rem = remaining_s;
            int cum = 0;
            for (int d = 255; d >= 0; --d) {
                const int h = int(hist[d]);
                if (cum + h >= rem) {
                    prefix_s = prefix | (uint(d) << shift);
                    mask_s = mask | (0xffu << shift);
                    remaining_s = rem - cum;
                    break;
                }
                cum += h;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint thr = prefix_s;
    const int remaining = remaining_s;
    const int n_gt = k - remaining;
    for (int i0 = 0; i0 < n; i0 += 1024) {
        const int i = i0 + int(tid);
        const uint u = i < n ? ds4_ord_key(v[i]) : 0u;
        const bool gt = i < n && u > thr, eq = i < n && u == thr;
        const auto bgv = simd_ballot(gt);
        const auto bev = simd_ballot(eq);
        const uint bg = uint((simd_vote::vote_t)bgv), be = uint((simd_vote::vote_t)bev);
        if (lane == 0u) {
            wgt[wid] = int(popcount(bg));
            weq[wid] = int(popcount(be));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (wid == 0u) {
            const int g0 = wgt[lane], e0 = weq[lane];
            int gs = g0, es = e0;
            for (uint o2 = 1u; o2 < 32u; o2 <<= 1) {
                const int a = simd_shuffle_up(gs, ushort(o2)), b = simd_shuffle_up(es, ushort(o2));
                if (uint(lane) >= o2) {
                    gs += a;
                    es += b;
                }
            }
            wgt[lane] = cnt_gt + gs - g0;
            weq[lane] = cnt_eq + es - e0;
            simdgroup_barrier(mem_flags::mem_none);
            if (lane == 31u) {
                cnt_gt += gs;
                cnt_eq += es;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int slot = -1;
        const uint below = (1u << lane) - 1u;
        if (gt) slot = wgt[wid] + int(popcount(bg & below));
        else if (eq) {
            const int e = weq[wid] + int(popcount(be & below));
            if (e < remaining) slot = n_gt + e;
        }
        if (slot >= 0) o[slot] = cd != nullptr ? cd[i / blk] * blk + i % blk : i;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ds4_kernels.cu:751 block_max_kernel - a block of `blk` positions' largest score (the indexer's candidate blocks), the
// newest position's block pinned in at +inf.  Grid (ceil(nb_max / 256), T), 256 threads.
kernel void ds4_block_max(device const float* score [[buffer(0)]], constant long& stride [[buffer(1)]],
                          constant Ds4Rows& r [[buffer(2)]], constant int& blk [[buffer(3)]],
                          device float* bm [[buffer(4)]], constant long& bm_stride [[buffer(5)]],
                          uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int b = int(tgp.x) * 256 + int(tid), t = int(tgp.y);
    int n, k_;
    ds4_rows_at(r, t, n, k_);
    const int nb = (n + blk - 1) / blk;
    if (b >= nb) return;
    float m = -INFINITY;
    for (int j = 0; j < blk; ++j) {
        const int p = b * blk + j;
        if (p < n) m = fmax(m, score[(long) t * stride + p]);
    }
    if (b == (n - 1) / blk) m = INFINITY;   // the newest position's block is pinned in
    bm[(long) t * bm_stride + b] = m;
}

// ---------------------------------------------------------------- ds4_kernels.cu:1141 hc_mean_kernel / :1150 iota_rows_kernel
//
// The two small shapes the decode loop asks for: a token's mean over its hc streams, and a row's indices.
kernel void ds4_hc_mean(device const float* h [[buffer(0)]], constant int& hc [[buffer(1)]],
                        constant int& dim [[buffer(2)]], device float* out [[buffer(3)]],
                        constant long& out_stride [[buffer(4)]], uint2 tgp [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]]) {
    const int e = int(tgp.x) * 256 + int(tid), t = int(tgp.y);
    if (e >= dim) return;
    float a = 0.0f;
    for (int s = 0; s < hc; ++s) a += h[((size_t) t * (size_t) hc + (size_t) s) * (size_t) dim + (size_t) e];
    out[(long) t * out_stride + e] = a / float(hc);
}

kernel void ds4_iota_rows(device int* out [[buffer(0)]], constant int& n [[buffer(1)]], constant int& stride [[buffer(2)]],
                          uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint t = tg;
    for (uint i = tid; i < uint(n); i += 64u) out[(size_t) t * (size_t) stride + i] = int(i);
}

// ---------------------------------------------------------------- vendor/sampler.cu:93 sampler_greedy_kernel
//
// The plain argmax over the vocabulary, ONE BLOCK (1024) a token: a thread's strided candidates with the repetition
// penalties applied (a threadgroup bitmap of the history window says which tokens can be penalised at all), a warp
// reduction, then warp 0's - a tie keeps the larger value and the SMALLER index, so this is the serial scan's answer.
kernel void ds4_sampler_greedy(device const float* logits [[buffer(0)]], constant int& n_vocab [[buffer(1)]],
                               device const int* history [[buffer(2)]], constant int& history_len [[buffer(3)]],
                               constant int& plen [[buffer(4)]], constant float& pen_repeat [[buffer(5)]],
                               constant float& pen_freq [[buffer(6)]], constant float& pen_present [[buffer(7)]],
                               device const float* logit_bias [[buffer(8)]], constant int& has_bias [[buffer(9)]],
                               device int* out [[buffer(10)]], uint tg [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                               ushort wid [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint penal_bits[4096];   // (n_vocab + 31) / 32 words (the host checks it fits)
    threadgroup float sv[32];
    threadgroup int si[32];
    const int t = int(tg);
    device const float* l = logits + (size_t) t * (size_t) n_vocab;
    device const int* hrow = history != nullptr ? history + (size_t) t * (size_t) history_len : nullptr;
    int hlen = 0;
    if (hrow != nullptr) {
        hlen = min(plen, history_len);
        if (hlen < 0) hlen = 0;
        hrow += history_len - hlen;   // the window is the tail
    }
    const int bits_words = (n_vocab + 31) / 32;
    const bool use_bits = hrow != nullptr && hlen > 0 && bits_words > 0;
    if (use_bits) {
        for (int w = int(tid); w < bits_words; w += 1024) penal_bits[w] = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = int(tid); i < hlen; i += 1024)
            if (hrow[i] >= 0 && hrow[i] < n_vocab) {
                threadgroup atomic_uint* pb = (threadgroup atomic_uint*) &penal_bits[hrow[i] >> 5];
                atomic_fetch_or_explicit(pb, 1u << (hrow[i] & 31), memory_order_relaxed);
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float bv = -INFINITY;
    int best = n_vocab;   // (the "no candidate" index: it loses every comparison)
    for (int v = int(tid); v < n_vocab; v += 1024) {
        float s = has_bias != 0 ? l[v] + logit_bias[v] : l[v];
        int cnt = 0;
        if (use_bits && (penal_bits[v >> 5] & (1u << (v & 31))) != 0u)
            for (int i = 0; i < hlen; ++i)
                if (hrow[i] == v) ++cnt;
        if (cnt > 0) {
            if (s <= 0.0f) s *= pen_repeat;
            else s /= pen_repeat;
            s -= float(cnt) * pen_freq + 1.0f * pen_present;
        }
        if (s > bv) {
            bv = s;
            best = v;
        }
    }
    for (uint o = 16u; o > 0u; o >>= 1) {
        const float ov = simd_shuffle_down(bv, ushort(o));
        const int oi = simd_shuffle_down(best, ushort(o));
        if (ov > bv || (ov == bv && oi < best)) {
            bv = ov;
            best = oi;
        }
    }
    if (lane == 0u) {
        sv[wid] = bv;
        si[wid] = best;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (wid == 0u) {
        float wv = sv[lane];
        int wi = si[lane];
        for (uint o = 16u; o > 0u; o >>= 1) {
            const float ov = simd_shuffle_down(wv, ushort(o));
            const int oi = simd_shuffle_down(wi, ushort(o));
            if (ov > wv || (ov == wv && oi < wi)) {
                wv = ov;
                wi = oi;
            }
        }
        if (lane == 0u) out[t] = (wi < n_vocab) ? wi : 0;   // (a tie between two -inf candidates: 0)
    }
}

// ---------------------------------------------------------------- sparse attention (the prompt path)
//
// ds4_kernels.cu:954 sparse_attn_packed_kernel<G> and :774 sparse_attn_kernel<false>.  The packed kernel stages sixteen
// rows of a tile in threadgroup memory (the window rows as they are, the compressed rows decoded to BF16) once for all
// G heads of a threadgroup, then does each head's dots, softmax and weighted sum.  The BF16 kernel is the same arithmetic
// for row sets already stored as BF16 (a DSpark block); one head and query a threadgroup.
struct Ds4AttnRows {   // ds4_kernels.cuh:149 AttnRows (kWin = 16)
    int win0[16], nwin[16], nsel[16];
    int gen, t0, win, R, kcap;
    long p0;
};

static inline void ds4_attn_counts(constant Ds4AttnRows& ar, int t, thread int& n_win, thread int& n_sel, thread int& w0) {
    if (ar.gen != 0) {
        const long ta = (long) ar.t0 + (long) t;
        const long pa = ar.p0 + ta;
        const long a = (long) ar.win - ar.p0, b = ta + 1;
        const long first = a > b ? a : b;   // max(win - p0, ta + 1)
        w0 = int(first);
        n_win = int((long) ar.win + ta - first + 1);
        n_sel = ar.R > 0 ? int(((pa + 1) / (long) ar.R) < (long) ar.kcap ? (pa + 1) / (long) ar.R : (long) ar.kcap) : 0;
    } else {
        w0 = ar.win0[t];
        n_win = ar.nwin[t];
        n_sel = ar.nsel[t];
    }
    if (n_win < 0) n_win = 0;
    if (n_sel < 0) n_sel = 0;
}

static inline float ds4_bf16_u32(uint v) { return as_type<float>(v << 16); }

// 32 values of a packed compressed row (kKvFp4, 512 values) as BF16 in four uint4s
static inline void ds4_attn_unpack32(const device uchar* rr, int part, threadgroup uint4* dst) {
    const uint4 cw = *(const device uint4*) (rr + (size_t) part * 16u);
    const float k0 = ds4_e4m3(uint(rr[256u + uint(part) * 2u])) * 16384.0f;
    const float k1 = ds4_e4m3(uint(rr[256u + uint(part) * 2u + 1u])) * 16384.0f;
    const uint wv[4] = {cw.x, cw.y, cw.z, cw.w};
    uint o[16];
    for (uint k = 0; k < 16u; ++k) {
        const uint b = wv[k >> 2] >> (8u * (k & 3u));
        const float ks = k < 8u ? k0 : k1;
        const float lo = ds4_e2m1_val14(b) * ks, hi = ds4_e2m1_val14(b >> 4) * ks;
        o[k] = (as_type<uint>(lo) >> 16) | (as_type<uint>(hi) & 0xffff0000u);
    }
    dst[0] = uint4(o[0], o[1], o[2], o[3]);
    dst[1] = uint4(o[4], o[5], o[6], o[7]);
    dst[2] = uint4(o[8], o[9], o[10], o[11]);
    dst[3] = uint4(o[12], o[13], o[14], o[15]);
}

// rows [s0, s0 + n) of the tile at t0 into tile4; a thread stages one row's 32 values
static inline void ds4_attn_stage(device const ushort* wbase, device const uchar* ckv, device const int* sl,
                                  int n_win, int t0, int s0, int n, int sr, int part, threadgroup uint4* tile4) {
    if (sr >= n) return;
    const int j = s0 + sr;
    threadgroup uint4* dst = tile4 + (size_t) sr * 64u + (size_t) part * 4u;
    const int jj = t0 + j;
    if (jj < n_win) {
        const device uint4* src = (const device uint4*) (wbase + (size_t) jj * 512u + (size_t) part * 32u);
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = src[3];
        return;
    }
    const device uchar* rr = ckv + (size_t) sl[jj - n_win] * 288u;
    ds4_attn_unpack32(rr, part, dst);
}

static inline float ds4_attn_warp_max(float v) {
    v = fmax(v, simd_shuffle_xor(v, 16));
    v = fmax(v, simd_shuffle_xor(v, 8));
    v = fmax(v, simd_shuffle_xor(v, 4));
    v = fmax(v, simd_shuffle_xor(v, 2));
    v = fmax(v, simd_shuffle_xor(v, 1));
    return v;
}

template <int G>
static inline void ds4_sparse_packed_run(device const float* q, device const ushort* win, device const uchar* ckv,
                                         device const int* sel, uint sel_stride, constant Ds4AttnRows& ar,
                                         device const float* sink, uint heads, float scale, device float* out,
                                         threadgroup uint4* tile4, threadgroup float* sc, threadgroup float* m_run_s,
                                         threadgroup float* tscale_s, threadgroup float* den_s, uint2 tgp, uint tid,
                                         ushort lane_in, ushort wp_in) {
    constexpr int W = 8 / G;   // warps a head
    const int hb = int(tgp.x) * G, t = int(tgp.y);
    int n_win, n_sel, w0;
    ds4_attn_counts(ar, t, n_win, n_sel, w0);
    const device ushort* wbase = win + (size_t) w0 * 512u;
    const device int* sl = sel + (size_t) t * (size_t) sel_stride;
    const uint lane = uint(lane_in), wp = uint(wp_in);
    const uint gw = wp / uint(W), wr = wp % uint(W);
    float qr[16];
    for (int x = 0; x < 16; ++x)
        qr[x] = q[((size_t) t * heads + uint(hb) + gw) * 512u + lane * 16u + uint(x)];
    if (tid < uint(G)) {
        m_run_s[tid] = -1e30f;
        den_s[tid] = 0.0f;
    }
    float o0[G], o1[G];
    for (int g = 0; g < G; ++g) o0[g] = o1[g] = 0.0f;
    const int c = int(tid) * 2;
    const int sr = int(tid) >> 4, part = int(tid) & 15;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int total = n_win + n_sel;
    for (int t0 = 0; t0 < total; t0 += 64) {
        const int nt = min(64, total - t0);
        for (int s0 = 0; s0 < nt; s0 += 16) {   // the dots
            const int n = min(16, nt - s0);
            ds4_attn_stage(wbase, ckv, sl, n_win, t0, s0, n, sr, part, tile4);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int r = int(wr); r < n; r += W) {
                const threadgroup uint4* rp = tile4 + (size_t) r * 64u + lane * 2u;
                const uint4 a = rp[0], b = rp[1];
                const uint av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
                float s = 0.0f;
                for (int x = 0; x < 4; ++x) {
                    s += qr[2 * x] * ds4_bf16_u32(av[x] & 0xffffu);
                    s += qr[2 * x + 1] * ds4_bf16_u32(av[x] >> 16);
                }
                for (int x = 0; x < 4; ++x) {
                    s += qr[8 + 2 * x] * ds4_bf16_u32(bv[x] & 0xffffu);
                    s += qr[8 + 2 * x + 1] * ds4_bf16_u32(bv[x] >> 16);
                }
                s = ds4_warp_sum(s);
                if (lane == 0u) sc[uint(gw) * 64u + uint(s0 + r)] = s * scale;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (wp < uint(G)) {   // head wp's softmax over the tile
            threadgroup float* scg = sc + wp * 64u;
            float m = -INFINITY;
            for (int j = int(lane); j < nt; j += 32) m = fmax(m, scg[j]);
            m = ds4_attn_warp_max(m);
            const float m_old = m_run_s[wp];
            const float m_new = fmax(m_old, m);
            float s = 0.0f;
            for (int j = int(lane); j < nt; j += 32) {
                const float p = exp(scg[j] - m_new);
                scg[j] = p;
                s += p;
            }
            s = ds4_warp_sum(s);
            if (lane == 0u) {
                tscale_s[wp] = exp(m_old - m_new);
                den_s[wp] = den_s[wp] * tscale_s[wp] + s;
                m_run_s[wp] = m_new;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int g = 0; g < G; ++g) {
            const float cs = tscale_s[g];
            o0[g] *= cs;
            o1[g] *= cs;
        }
        const threadgroup uint* tu = (const threadgroup uint*) tile4;
        for (int s0 = 0; s0 < nt; s0 += 16) {   // the weighted sums
            const int n = min(16, nt - s0);
            ds4_attn_stage(wbase, ckv, sl, n_win, t0, s0, n, sr, part, tile4);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int r = 0; r < n; ++r) {
                const uint pr = tu[(size_t) r * 256u + uint(tid)];
                const float v0 = ds4_bf16_u32(pr & 0xffffu), v1 = ds4_bf16_u32(pr >> 16);
                for (int g = 0; g < G; ++g) {
                    const float p = sc[uint(g) * 64u + uint(s0 + r)];
                    o0[g] = fma(p, v0, o0[g]);
                    o1[g] = fma(p, v1, o1[g]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (int g = 0; g < G; ++g) {
        const float den = den_s[g] + exp(sink[hb + g] - m_run_s[g]);
        device float* ov = out + ((size_t) t * heads + uint(hb + g)) * 512u;
        ov[c] = o0[g] / den;
        ov[c + 1] = o1[g] / den;
    }
}

#define DS4_SPARSE_PACKED_SIG(NAME)                                                                                    \
    kernel void NAME(device const float* q [[buffer(0)]], device const ushort* win [[buffer(1)]],                      \
                     device const uchar* ckv [[buffer(2)]], device const int* sel [[buffer(3)]],                       \
                     constant uint& sel_stride [[buffer(4)]], constant Ds4AttnRows& ar [[buffer(5)]],                  \
                     device const float* sink [[buffer(6)]], constant uint& heads [[buffer(7)]],                       \
                     constant float& scale [[buffer(8)]], device float* out [[buffer(9)]],                            \
                     uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],             \
                     ushort lane [[thread_index_in_simdgroup]], ushort wp [[simdgroup_index_in_threadgroup]])

#define DS4_SPARSE_PACKED_BODY(G)                                                                                      \
    {                                                                                                                  \
        threadgroup uint4 tile4[16u * 64u];                                                                            \
        threadgroup float sc[8u * 64u];                                                                                \
        threadgroup float m_run_s[8u], tscale_s[8u], den_s[8u];                                                        \
        ds4_sparse_packed_run<G>(q, win, ckv, sel, sel_stride, ar, sink, heads, scale, out, tile4, sc, m_run_s,        \
                                 tscale_s, den_s, tgp, tid, lane, wp);                                                 \
    }

DS4_SPARSE_PACKED_SIG(ds4_sparse_attn_g8) DS4_SPARSE_PACKED_BODY(8)
DS4_SPARSE_PACKED_SIG(ds4_sparse_attn_g4) DS4_SPARSE_PACKED_BODY(4)
DS4_SPARSE_PACKED_SIG(ds4_sparse_attn_g2) DS4_SPARSE_PACKED_BODY(2)
DS4_SPARSE_PACKED_SIG(ds4_sparse_attn_g1) DS4_SPARSE_PACKED_BODY(1)

#undef DS4_SPARSE_PACKED_SIG
#undef DS4_SPARSE_PACKED_BODY

#ifndef DS4_MX_PLO
#define DS4_MX_PLO 0   // 1: the softmax weights as BF16 high + low (rms error vs the scalar kernel 2e-6 instead of 3e-4)
#endif
struct Ds4AttnRaw { uint4 v[4]; int kind; };   // a thread's 32 values of one row, as read: kind 0 window (v[0..3] BF16), 1 compressed (v[0] codes, v[1].x the two scale bytes), 2 none

static inline void ds4_attn_ld(device const ushort* wbase, device const uchar* ckv, device const int* sl, int n_win, int t0, int n,
                               int sr, int pt, thread Ds4AttnRaw& r) {
    r.kind = 2;
    if (sr >= n) return;
    const int jj = t0 + sr;
    if (jj < n_win) {
        const device uint4* src = (const device uint4*) (wbase + (size_t) jj * 512u + (size_t) pt * 32u);
        r.v[0] = src[0]; r.v[1] = src[1]; r.v[2] = src[2]; r.v[3] = src[3];
        r.kind = 0;
    } else {
        const device uchar* rr = ckv + (size_t) sl[jj - n_win] * 288u;
        r.v[0] = *(const device uint4*) (rr + (size_t) pt * 16u);
        r.v[1] = uint4(uint(rr[256u + uint(pt) * 2u]), uint(rr[256u + uint(pt) * 2u + 1u]), 0u, 0u);
        r.kind = 1;
    }
}

static inline void ds4_attn_st(thread const Ds4AttnRaw& r, int sr, int pt, threadgroup uint4* tile4) {
    threadgroup uint4* dst = tile4 + (size_t) sr * 64u + (size_t) pt * 4u;
    if (r.kind == 0) {
        dst[0] = r.v[0]; dst[1] = r.v[1]; dst[2] = r.v[2]; dst[3] = r.v[3];
    } else if (r.kind == 1) {
        const float k0 = ds4_e4m3(r.v[1].x) * 16384.0f;
        const float k1 = ds4_e4m3(r.v[1].y) * 16384.0f;
        const uint wv[4] = {r.v[0].x, r.v[0].y, r.v[0].z, r.v[0].w};
        uint o[16];
        for (uint k = 0; k < 16u; ++k) {
            const uint b = wv[k >> 2] >> (8u * (k & 3u));
            const float ks = k < 8u ? k0 : k1;
            const float lo = ds4_e2m1_val14(b) * ks, hi = ds4_e2m1_val14(b >> 4) * ks;
            o[k] = (as_type<uint>(lo) >> 16) | (as_type<uint>(hi) & 0xffff0000u);
        }
        dst[0] = uint4(o[0], o[1], o[2], o[3]);
        dst[1] = uint4(o[4], o[5], o[6], o[7]);
        dst[2] = uint4(o[8], o[9], o[10], o[11]);
        dst[3] = uint4(o[12], o[13], o[14], o[15]);
    } else {
        dst[0] = uint4(0); dst[1] = uint4(0); dst[2] = uint4(0); dst[3] = uint4(0);
    }
}

// ---------------------------------------------------------------- sparse attention on the matrix units (G = 8 heads a threadgroup)
//
// The same attention as ds4_sparse_attn_g8 (a token, eight heads, the window rows plus the selected compressed rows, softmax with the head's
// sink) but the dots and the weighted sum run on simdgroup 8x8 matrices: eight rows a pass are staged once as BF16 (exact), q and the
// softmax weights are split into a BF16 high and low part (so the products keep about 16 bits), accumulation is fp32, and the softmax is the
// online one (the running maximum rescales the accumulators through a diagonal matrix).
#ifndef DS4_MX_QK_LO
#define DS4_MX_QK_LO 0   // 0: the QK dot uses the BF16 high part of q only (faster, ~8 fewer bits)
#endif
// ATTN_PROBE (function constant 25): a timing probe.  1 skips the P.V accumulate, 2 skips the QK matmul - the numbers
// it produces are wrong on purpose; it exists to say which half of the pass the 20 s of a 32K prompt is in.
constant int ATTN_PROBE [[function_constant(25)]];
constant int attn_probe = is_function_constant_defined(ATTN_PROBE) ? ATTN_PROBE : 0;

kernel void ds4_sparse_attn_mx(device const float* q [[buffer(0)]], device const ushort* win [[buffer(1)]],
                               device const uchar* ckv [[buffer(2)]], device const int* sel [[buffer(3)]],
                               constant uint& sel_stride [[buffer(4)]], constant Ds4AttnRows& ar [[buffer(5)]],
                               device const float* sink [[buffer(6)]], constant uint& heads [[buffer(7)]],
                               constant float& scale [[buffer(8)]], device float* out [[buffer(9)]],
                               uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                               ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint4 tile4[8u * 64u];
    threadgroup float qbuf[4096];            // q hi / lo as BF16 (8 KB each), later the output rows as float
    threadgroup float part[8u * 64u];
    threadgroup float ssm[64];
    threadgroup bfloat phi[64], plo[64];
    threadgroup float dg[64];
    threadgroup float m_s[8], den_s[8], sc_s[8];
    threadgroup bfloat* qhi = (threadgroup bfloat*) qbuf;
    threadgroup bfloat* qlo = qhi + 4096;
    threadgroup const bfloat* kt = (threadgroup const bfloat*) tile4;

    const int hb = int(tgp.x) * 8, t = int(tgp.y);
    int n_win, n_sel, w0;
    ds4_attn_counts(ar, t, n_win, n_sel, w0);
    const device ushort* wbase = win + (size_t) w0 * 512u;
    const device int* sl = sel + (size_t) t * (size_t) sel_stride;
    for (uint x = 0; x < 16u; ++x) {
        const uint idx = tid * 16u + x, h = idx >> 9, d = idx & 511u;
        const float v = q[((size_t) t * heads + uint(hb) + h) * 512u + d];
        const bfloat hi = bfloat(v);
        qhi[idx] = hi;
        if (DS4_MX_QK_LO) qlo[idx] = bfloat(v - float(hi));   // (unused when the QK dot takes the high part only)
    }
    if (tid < 8u) {
        m_s[tid] = -1e30f;
        den_s[tid] = 0.0f;
    }
    simdgroup_float8x8 O[8];
    for (int c = 0; c < 8; ++c) O[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const int sr = int(tid) >> 4, pt = int(tid) & 15;
    const int total = n_win + n_sel;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    Ds4AttnRaw raw;
    raw.kind = 2;
    if (sr < 8) ds4_attn_ld(wbase, ckv, sl, n_win, 0, min(8, total), sr, pt, raw);
    for (int t0 = 0; t0 < total; t0 += 8) {
        const int n = min(8, total - t0);
        if (sr < 8 && attn_probe != 5) ds4_attn_st(raw, sr, pt, tile4);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sr < 8) ds4_attn_ld(wbase, ckv, sl, n_win, t0 + 8, min(8, total - t0 - 8), sr, pt, raw);   // the next pass's rows, in flight during this one
        if (attn_probe != 2 && attn_probe != 6) {   // S = q . K^T over this simdgroup's eighth of the 512 values
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int kb = int(sg) * 8; kb < int(sg) * 8 + 8; ++kb) {
                simdgroup_bfloat8x8 a_hi, a_lo, b;
                simdgroup_load(b, kt, 512, ulong2(ulong(kb) * 8ul, 0ul), true);
                simdgroup_load(a_hi, qhi, 512, ulong2(ulong(kb) * 8ul, 0ul));
                simdgroup_load(a_lo, qlo, 512, ulong2(ulong(kb) * 8ul, 0ul));
                simdgroup_multiply_accumulate(acc, a_hi, b, acc);
                if (DS4_MX_QK_LO) simdgroup_multiply_accumulate(acc, a_lo, b, acc);   // (the low part of q: exactness vs speed)
            }
            simdgroup_store(acc, part + uint(sg) * 64u, 8);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 64u && attn_probe != 4 && attn_probe != 6) {
            float s = 0.0f;
            for (uint g = 0; g < 8u; ++g) s += part[g * 64u + tid];
            ssm[tid] = s * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (attn_probe != 3) {   // head sg's online softmax over the pass's rows
            const bool valid = int(lane) < n;
            const float s = valid ? ssm[uint(sg) * 8u + uint(lane)] : -INFINITY;
            const float m = ds4_attn_warp_max(s);
            const float m_old = m_s[sg];
            const float m_new = fmax(m_old, m);
            const float p = valid ? exp(s - m_new) : 0.0f;
            const float ps = ds4_warp_sum(p);
            if (lane < 8u) {
                const bfloat hi = bfloat(p);
                phi[uint(sg) * 8u + uint(lane)] = hi;
                plo[uint(sg) * 8u + uint(lane)] = bfloat(p - float(hi));
            }
            if (lane == 0u) {
                const float cs = exp(m_old - m_new);
                sc_s[sg] = cs;
                den_s[sg] = den_s[sg] * cs + ps;
                m_s[sg] = m_new;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 64u) dg[tid] = (tid >> 3) == (tid & 7u) ? sc_s[tid >> 3] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (attn_probe != 1 && attn_probe != 6) {   // O = diag(rescale) O + P V on this simdgroup's 64 columns
            simdgroup_float8x8 D;
            simdgroup_bfloat8x8 ph, pl;
            simdgroup_load(D, dg, 8);
            simdgroup_load(ph, phi, 8);
            simdgroup_load(pl, plo, 8);
            for (int c = 0; c < 8; ++c) {
                simdgroup_float8x8 tmp;
                simdgroup_multiply(tmp, D, O[c]);
                simdgroup_bfloat8x8 v;
                simdgroup_load(v, kt, 512, ulong2(ulong(sg) * 64ul + ulong(c) * 8ul, 0ul));
                simdgroup_multiply_accumulate(tmp, ph, v, tmp);
#if DS4_MX_PLO
                simdgroup_multiply_accumulate(tmp, pl, v, tmp);
#endif
                O[c] = tmp;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int c = 0; c < 8; ++c) simdgroup_store(O[c], qbuf, 512, ulong2(ulong(sg) * 64ul + ulong(c) * 8ul, 0ul));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint x = 0; x < 16u; ++x) {
        const uint idx = tid * 16u + x, h = idx >> 9, d = idx & 511u;
        const float den = den_s[h] + exp(sink[uint(hb) + h] - m_s[h]);
        out[((size_t) t * heads + uint(hb) + h) * 512u + d] = qbuf[idx] / den;
    }
}

// TWO QUERIES A THREADGROUP, written out in full.  The staged keys are shared; each query keeps its own accumulators,
// its own online-softmax state and its own key count (the causal shape makes the second query's set a superset, so the
// union staged is the second query's own set and the first masks its tail).
kernel void ds4_sparse_attn_mx2(device const float* q [[buffer(0)]], device const ushort* win [[buffer(1)]],
                                device const uchar* ckv [[buffer(2)]], device const int* sel [[buffer(3)]],
                                constant uint& sel_stride [[buffer(4)]], constant Ds4AttnRows& ar [[buffer(5)]],
                                device const float* sink [[buffer(6)]], constant uint& heads [[buffer(7)]],
                                constant float& scale [[buffer(8)]], device float* out [[buffer(9)]],
                                uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint4 tile4[8u * 64u];
    threadgroup float qbuf[4096];            // two queries' q as BF16 (8 KB each), later a query's output rows
    threadgroup float part[8u * 64u];
    threadgroup float ssm[64];
    threadgroup bfloat phi[64], plo[64];
    threadgroup float dg[64];
    threadgroup float m_s[16], den_s[16], sc_s[16];
    threadgroup bfloat* qhiA = (threadgroup bfloat*) qbuf;
    threadgroup bfloat* qhiB = qhiA + 4096;   // a query's q is 4096 bf16
    threadgroup float* obuf = qbuf;           // 8 heads x 512 floats: qbuf again, once the key loop is done
    threadgroup const bfloat* kt = (threadgroup const bfloat*) tile4;

    const int hb = int(tgp.x) * 8;
    const int tA = int(tgp.y) * 2;
    const int rows_total = (ar.gen == 0 && ar.t0 > 0) ? ar.t0 : 0x7fffffff;
    const int nq = (tA + 1 < rows_total) ? 2 : 1;
    int nwA, nsA, w0A, nwB, nsB, w0B;
    ds4_attn_counts(ar, tA, nwA, nsA, w0A);
    nwB = nwA; nsB = nsA; w0B = w0A;
    if (nq > 1) ds4_attn_counts(ar, tA + 1, nwB, nsB, w0B);
    const int totalA = nwA + nsA;
    const int totalB = (nq > 1) ? nwB + nsB : totalA;
    const int total = totalA > totalB ? totalA : totalB;
    const device ushort* wbase = win + (size_t) w0A * 512u;
    const device int* sl = sel + (size_t) tA * (size_t) sel_stride;
    for (uint x = 0; x < 16u; ++x) {
        const uint idx = tid * 16u + x, h = idx >> 9, d = idx & 511u;
        qhiA[idx] = bfloat(q[((size_t) tA * heads + uint(hb) + h) * 512u + d]);
        if (nq > 1) qhiB[idx] = bfloat(q[((size_t) (tA + 1) * heads + uint(hb) + h) * 512u + d]);
    }
    if (tid < 16u) {
        m_s[tid] = -1e30f;
        den_s[tid] = 0.0f;
    }
    simdgroup_float8x8 OA[8], OB[8];
    for (int c = 0; c < 8; ++c) {
        OA[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        OB[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    const int sr = int(tid) >> 4, pt = int(tid) & 15;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    Ds4AttnRaw raw;
    raw.kind = 2;
    if (sr < 8) ds4_attn_ld(wbase, ckv, sl, nwB, 0, min(8, total), sr, pt, raw);
    for (int t0 = 0; t0 < total; t0 += 8) {
        const int np = min(8, total - t0);
        (void) np;
        if (sr < 8) ds4_attn_st(raw, sr, pt, tile4);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sr < 8) ds4_attn_ld(wbase, ckv, sl, nwB, t0 + 8, min(8, total - t0 - 8), sr, pt, raw);
        {   // query A: S = q . K^T
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int kb = int(sg) * 8; kb < int(sg) * 8 + 8; ++kb) {
                simdgroup_bfloat8x8 a_hi, b;
                simdgroup_load(b, kt, 512, ulong2(ulong(kb) * 8ul, 0ul), true);
                simdgroup_load(a_hi, qhiA, 512, ulong2(ulong(kb) * 8ul, 0ul));
                simdgroup_multiply_accumulate(acc, a_hi, b, acc);
            }
            simdgroup_store(acc, part + uint(sg) * 64u, 8);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 64u) {
            float s = 0.0f;
            for (uint g = 0; g < 8u; ++g) s += part[g * 64u + tid];
            ssm[tid] = s * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   // query A: online softmax
            const int nq_here = min(8, totalA - t0);
            const bool valid = int(lane) < nq_here;
            const float s = valid ? ssm[uint(sg) * 8u + uint(lane)] : -INFINITY;
            const float m = ds4_attn_warp_max(s);
            const float m_old = m_s[0 + sg];
            const float m_new = fmax(m_old, m);
            const float p = valid ? exp(s - m_new) : 0.0f;
            const float ps = ds4_warp_sum(p);
            if (lane < 8u) {
                const bfloat hi = bfloat(p);
                phi[uint(sg) * 8u + uint(lane)] = hi;
                plo[uint(sg) * 8u + uint(lane)] = bfloat(p - float(hi));
            }
            if (lane == 0u) {
                const float cs = exp(m_old - m_new);
                sc_s[0 + sg] = cs;
                den_s[0 + sg] = den_s[0 + sg] * cs + ps;
                m_s[0 + sg] = m_new;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 64u) dg[tid] = (tid >> 3) == (tid & 7u) ? sc_s[0 + (tid >> 3)] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   // query A: O = diag(rescale) O + P V
            simdgroup_float8x8 D;
            simdgroup_bfloat8x8 ph, pl;
            simdgroup_load(D, dg, 8);
            simdgroup_load(ph, phi, 8);
            simdgroup_load(pl, plo, 8);
            for (int c = 0; c < 8; ++c) {
                simdgroup_float8x8 tmp;
                simdgroup_multiply(tmp, D, OA[c]);
                simdgroup_bfloat8x8 v;
                simdgroup_load(v, kt, 512, ulong2(ulong(sg) * 64ul + ulong(c) * 8ul, 0ul));
                simdgroup_multiply_accumulate(tmp, ph, v, tmp);
#if DS4_MX_PLO
                simdgroup_multiply_accumulate(tmp, pl, v, tmp);
#endif
                OA[c] = tmp;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (nq > 1) {
        {   // query B: S = q . K^T
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int kb = int(sg) * 8; kb < int(sg) * 8 + 8; ++kb) {
                simdgroup_bfloat8x8 a_hi, b;
                simdgroup_load(b, kt, 512, ulong2(ulong(kb) * 8ul, 0ul), true);
                simdgroup_load(a_hi, qhiB, 512, ulong2(ulong(kb) * 8ul, 0ul));
                simdgroup_multiply_accumulate(acc, a_hi, b, acc);
            }
            simdgroup_store(acc, part + uint(sg) * 64u, 8);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 64u) {
            float s = 0.0f;
            for (uint g = 0; g < 8u; ++g) s += part[g * 64u + tid];
            ssm[tid] = s * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   // query B: online softmax
            const int nq_here = min(8, totalB - t0);
            const bool valid = int(lane) < nq_here;
            const float s = valid ? ssm[uint(sg) * 8u + uint(lane)] : -INFINITY;
            const float m = ds4_attn_warp_max(s);
            const float m_old = m_s[8 + sg];
            const float m_new = fmax(m_old, m);
            const float p = valid ? exp(s - m_new) : 0.0f;
            const float ps = ds4_warp_sum(p);
            if (lane < 8u) {
                const bfloat hi = bfloat(p);
                phi[uint(sg) * 8u + uint(lane)] = hi;
                plo[uint(sg) * 8u + uint(lane)] = bfloat(p - float(hi));
            }
            if (lane == 0u) {
                const float cs = exp(m_old - m_new);
                sc_s[8 + sg] = cs;
                den_s[8 + sg] = den_s[8 + sg] * cs + ps;
                m_s[8 + sg] = m_new;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 64u) dg[tid] = (tid >> 3) == (tid & 7u) ? sc_s[8 + (tid >> 3)] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   // query B: O = diag(rescale) O + P V
            simdgroup_float8x8 D;
            simdgroup_bfloat8x8 ph, pl;
            simdgroup_load(D, dg, 8);
            simdgroup_load(ph, phi, 8);
            simdgroup_load(pl, plo, 8);
            for (int c = 0; c < 8; ++c) {
                simdgroup_float8x8 tmp;
                simdgroup_multiply(tmp, D, OB[c]);
                simdgroup_bfloat8x8 v;
                simdgroup_load(v, kt, 512, ulong2(ulong(sg) * 64ul + ulong(c) * 8ul, 0ul));
                simdgroup_multiply_accumulate(tmp, ph, v, tmp);
#if DS4_MX_PLO
                simdgroup_multiply_accumulate(tmp, pl, v, tmp);
#endif
                OB[c] = tmp;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (int c = 0; c < 8; ++c) simdgroup_store(OA[c], obuf, 512, ulong2(ulong(sg) * 64ul + ulong(c) * 8ul, 0ul));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint x = 0; x < 16u; ++x) {
        const uint idx = tid * 16u + x, h = idx >> 9, d = idx & 511u;
        out[((size_t) tA * heads + uint(hb) + h) * 512u + d] = obuf[idx] / (den_s[h] + exp(sink[uint(hb) + h] - m_s[h]));
    }
    if (nq > 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int c = 0; c < 8; ++c) simdgroup_store(OB[c], obuf, 512, ulong2(ulong(sg) * 64ul + ulong(c) * 8ul, 0ul));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint x = 0; x < 16u; ++x) {
            const uint idx = tid * 16u + x, h = idx >> 9, d = idx & 511u;
            out[((size_t) (tA + 1) * heads + uint(hb) + h) * 512u + d] =
                obuf[idx] / (den_s[8u + h] + exp(sink[uint(hb) + h] - m_s[8u + h]));
        }
    }
}

// kContig == false (the row set is BF16 already): one head and query a threadgroup, a warp a row's dot and warp 0 the
// softmax, every thread two output values.
static inline void ds4_sparse_bf16_run(device const float* q, device const ushort* win, device const ushort* ckv,
                                       device const int* sel, uint sel_stride, constant Ds4AttnRows& ar,
                                       device const float* sink, uint heads, float scale, device float* out,
                                       threadgroup float* qs, threadgroup float* sc, threadgroup int* rowix,
                                       threadgroup float* m_run_s, threadgroup float* tscale_s,
                                       threadgroup float* den_s, uint2 tgp, uint tid, ushort lane_in,
                                       ushort wp_in) {
    const int h = int(tgp.x), t = int(tgp.y);
    int n_win, n_sel, w0;
    ds4_attn_counts(ar, t, n_win, n_sel, w0);
    const device ushort* wbase = win + (size_t) w0 * 512u;
    const device int* sl = sel + (size_t) t * (size_t) sel_stride;
    const uint lane = uint(lane_in), wp = uint(wp_in);
    for (uint i = tid; i < 512u; i += 256u) qs[i] = q[((size_t) t * heads + uint(h)) * 512u + i];
    if (tid == 0u) {
        m_run_s[0] = -1e30f;
        den_s[0] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o0 = 0.0f, o1 = 0.0f;
    const int c = int(tid) * 2;
    const int total = n_win + n_sel;
    for (int t0 = 0; t0 < total; t0 += 64) {
        const int nt = min(64, total - t0);
        if (tid < uint(nt)) {
            const int j = t0 + int(tid);
            rowix[tid] = j < n_win ? -1 - j : sl[j - n_win];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int jj = int(wp); jj < nt; jj += 8) {
            const int code = rowix[jj];
            const device ushort* row = code < 0 ? wbase + (size_t) (-1 - code) * 512u
                                                : ckv + (size_t) code * 512u;
            const device uint4* rp = (const device uint4*) (row + lane * 16u);
            const uint4 a = rp[0], b = rp[1];
            const uint av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
            float s = 0.0f;
            for (int x = 0; x < 4; ++x) {
                s += qs[lane * 16u + uint(2 * x)] * ds4_bf16_u32(av[x] & 0xffffu);
                s += qs[lane * 16u + uint(2 * x + 1)] * ds4_bf16_u32(av[x] >> 16);
            }
            for (int x = 0; x < 4; ++x) {
                s += qs[lane * 16u + uint(8 + 2 * x)] * ds4_bf16_u32(bv[x] & 0xffffu);
                s += qs[lane * 16u + uint(9 + 2 * x)] * ds4_bf16_u32(bv[x] >> 16);
            }
            s = ds4_warp_sum(s);
            if (lane == 0u) sc[jj] = s * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (wp == 0u) {
            float m = -INFINITY;
            for (int j = int(lane); j < nt; j += 32) m = fmax(m, sc[j]);
            m = ds4_attn_warp_max(m);
            const float m_old = m_run_s[0];
            const float m_new = fmax(m_old, m);
            float s = 0.0f;
            for (int j = int(lane); j < nt; j += 32) {
                const float p = exp(sc[j] - m_new);
                sc[j] = p;
                s += p;
            }
            s = ds4_warp_sum(s);
            if (lane == 0u) {
                tscale_s[0] = exp(m_old - m_new);
                den_s[0] = den_s[0] * tscale_s[0] + s;
                m_run_s[0] = m_new;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float cs = tscale_s[0];
        o0 *= cs;
        o1 *= cs;
        for (int j = 0; j < nt; ++j) {
            const int code = rowix[j];
            const device ushort* row = code < 0 ? wbase + (size_t) (-1 - code) * 512u
                                                : ckv + (size_t) code * 512u;
            const uint pr = *(const device uint*) (row + c);
            o0 = fma(sc[j], ds4_bf16_u32(pr & 0xffffu), o0);
            o1 = fma(sc[j], ds4_bf16_u32(pr >> 16), o1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float den = den_s[0] + exp(sink[h] - m_run_s[0]);
    device float* ov = out + ((size_t) t * heads + uint(h)) * 512u;
    ov[c] = o0 / den;
    ov[c + 1] = o1 / den;
}

kernel void ds4_sparse_attn_bf16(device const float* q [[buffer(0)]], device const ushort* win [[buffer(1)]],
                                 device const ushort* ckv [[buffer(2)]], device const int* sel [[buffer(3)]],
                                 constant uint& sel_stride [[buffer(4)]], constant Ds4AttnRows& ar [[buffer(5)]],
                                 device const float* sink [[buffer(6)]], constant uint& heads [[buffer(7)]],
                                 constant float& scale [[buffer(8)]], device float* out [[buffer(9)]],
                                 uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                 ushort lane [[thread_index_in_simdgroup]], ushort wp [[simdgroup_index_in_threadgroup]]) {
    threadgroup float qs[512];
    threadgroup float sc[64];
    threadgroup int rowix[64];
    threadgroup float m_run_s, tscale_s, den_s;
    ds4_sparse_bf16_run(q, win, ckv, sel, sel_stride, ar, sink, heads, scale, out, qs, sc, rowix, &m_run_s,
                        &tscale_s, &den_s, tgp, tid, lane, wp);
}

// ds4_kernels.cu:461 route_kernel - softplus scores, their top k with bias, weights from the unbiased scores.  One
// threadgroup (512) a token, as the CUDA kernel; rows 1024 values apart, the picks kPicks = 8 apart.
kernel void ds4_route(device const float* logits [[buffer(0)]], device const float* bias [[buffer(1)]],
                      constant uint& n [[buffer(2)]], constant uint& k [[buffer(3)]], constant float& route_scale [[buffer(4)]],
                      device int* idx [[buffer(5)]], device float* w [[buffer(6)]], device const float* bias_vl [[buffer(7)]],
                      constant uint& has_vl [[buffer(8)]], device const uint* vlw [[buffer(9)]],
                      uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]], ushort wp [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sc[1024], bs[1024];
    threadgroup float bv[32];
    threadgroup int bi[32];
    threadgroup int pick[16];
    const uint t = tg;
    device const float* l = logits + (size_t) t * 1024u;
    device const float* bp = bias;
    if (has_vl != 0u && ((vlw[t >> 5u] >> (t & 31u)) & 1u) != 0u) bp = bias_vl;
    for (uint i = tid; i < n; i += 512u) {
        const float v = l[i];
        const float sp = v > 20.0f ? v : log(1.0f + exp(v));
        sc[i] = sqrt(sp);
        bs[i] = sc[i] + bp[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = 0; r < k; ++r) {
        float best = -INFINITY;
        int bix = 0x7fffffff;
        for (uint i = tid; i < n; i += 512u)
            if (bs[i] > best || (bs[i] == best && int(i) < bix)) {
                best = bs[i];
                bix = int(i);
            }
        for (uint o = 16u; o > 0u; o >>= 1u) {
            const float ov = simd_shuffle_xor(best, o);
            const int oi = simd_shuffle_xor(bix, o);
            if (ov > best || (ov == best && oi < bix)) {
                best = ov;
                bix = oi;
            }
        }
        if (lane == 0u) {
            bv[wp] = best;
            bi[wp] = bix;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float b = -INFINITY;
            int ix = 0x7fffffff;
            for (uint q = 0; q < 16u; ++q)
                if (bv[q] > b || (bv[q] == b && bi[q] < ix)) {
                    b = bv[q];
                    ix = bi[q];
                }
            pick[r] = ix;
            bs[ix] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        float s = 0.0f;
        for (uint r = 0; r < k; ++r) s += sc[pick[r]];
        for (uint r = 0; r < k; ++r) {
            idx[(size_t) t * 8u + r] = pick[r];
            w[(size_t) t * 8u + r] = sc[pick[r]] / (s + 1e-20f) * route_scale;
        }
    }
}

// ds4_kernels.cu:1336 gather_kernel and :1342 combine_kernel - a token's gathered expert-output rows and the MoE's sum.
struct Ds4Gather {
    int src[16 * 8];
};
struct Ds4Combine {
    int n[16];
    int row[16][8];
    float w[16][8];
    int cpu[16];
};

kernel void ds4_gather(device const float* x [[buffer(0)]], constant Ds4Gather& g [[buffer(1)]],
                       constant uint& cols [[buffer(2)]], device float* out [[buffer(3)]],
                       uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint r = tgp.y;
    const uint c = tgp.x * 256u + tid;
    if (c < cols) out[(size_t) r * cols + c] = x[(size_t) g.src[r] * cols + c];
}

kernel void ds4_combine(device const float* ed [[buffer(0)]], device const float* cpu [[buffer(1)]],
                        device const float* sh [[buffer(2)]], constant Ds4Combine& cm [[buffer(3)]],
                        constant uint& dim [[buffer(4)]], constant uint& has_cpu [[buffer(5)]],
                        constant uint& has_sh [[buffer(6)]], device float* out [[buffer(7)]],
                        uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint e = tgp.x * 256u + tid, t = tgp.y;
    if (e >= dim) return;
    float a = 0.0f;
    for (int j = 0; j < cm.n[t]; ++j) a = fma(cm.w[t][j], ed[(size_t) cm.row[t][j] * dim + e], a);
    if (has_cpu != 0u && cm.cpu[t] != 0) a += cpu[(size_t) t * dim + e];
    if (has_sh != 0u) a += sh[(size_t) t * dim + e];
    out[(size_t) t * dim + e] = a;
}

// ---------------------------------------------------------------- ds4_kernels.cu:1156 markov_kernel / :1265 confidence_kernel
//
// The drafter's head (DSpark's Markov chain and its confidence): markov_add is a warp a vocabulary row - the previous
// token's E row (BF16) staged as floats in the threadgroup, then E . H[v] added to logits[v] (H is F16, the pack's
// markov_head.head.weight); confidence is a threadgroup a token over x and the same E row against the head's weights,
// its block sum through the logistic.
struct Ds4MarkovJob {
    device float* logits;       // [V]: += the bias
    device const ushort* E;     // [vocab][R] BF16
    device const ushort* H;     // [vocab][R] F16
    device const int* tok;      // the previous token (one int)
    int V, R;
};

struct Ds4ConfJob {
    device const float* x;      // [T][dim]
    device const ushort* E;     // [vocab][R] BF16
    device const int* tok;      // [T]
    device const float* w;      // [dim + R]
    device float* out;          // [T]
    int dim, R;
};

// a BF16 bit pattern as its float (ds4_kernels.cu:28 bf16f: the top half, no rounding)
static inline float ds4_bf16f(ushort b) { return as_type<float>(uint(b) << 16u); }

// a warp a vocabulary row: logits[v] += E[*tok] . H[v]
kernel void ds4_markov_add(constant Ds4MarkovJob& j [[buffer(0)]], uint tg [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float er[1024];   // (R <= 1024: the host wrapper refuses more)
    const device ushort* erow = j.E + (size_t) (*j.tok) * (size_t) j.R;
    for (int i = int(tid); i < j.R; i += 256) er[i] = ds4_bf16f(erow[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int v = int(tg) * 8 + int(tid >> 5);
    if (v >= j.V) return;
    const device ushort* hr = j.H + (size_t) v * (size_t) j.R;
    float a = 0.0f;
    for (int c = int(lane) * 8; c < j.R; c += 256) {
        for (int i = 0; i < 8; ++i) a += float(as_type<half>(hr[c + i])) * er[c + i];
    }
    a = ds4_warp_sum(a);
    if (lane == 0u) j.logits[v] += a;
}

// a threadgroup a token: the logistic of the head's sum over x and the token's E row
kernel void ds4_confidence(constant Ds4ConfJob& j [[buffer(0)]], uint tg [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                           ushort w [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[32];
    const int t = int(tg);
    const device float* xt = j.x + (size_t) t * (size_t) j.dim;
    const device ushort* er = j.E + (size_t) j.tok[t] * (size_t) j.R;
    float a = 0.0f;
    for (int i = int(tid); i < j.dim + j.R; i += 1024) a += j.w[i] * (i < j.dim ? xt[i] : ds4_bf16f(er[i - j.dim]));
    a = ds4_block_sum_n(a, red, lane, w, 32u);
    if (tid == 0u) j.out[t] = 1.0f / (1.0f + exp(-a));
}

// embed by raw device addresses (the runtime's MTLResidencySet makes gpuAddress pointers valid kernel arguments)
struct Ds4EmbedArgs {
    device ushort* rows;
    device float* h;
    uint dim;
    uint hc;
};
kernel void ds4_embed_j(device const Ds4EmbedArgs* a [[buffer(0)]], uint2 tgp [[threadgroup_position_in_grid]],
                        ushort lane [[thread_index_in_threadgroup]]) {
    const uint e = tgp.x * 256u + uint(lane);
    if (e >= a->dim) return;
    const uint t = tgp.y;
    const uint bits = uint(a->rows[(size_t) t * a->dim + e]) << 16;
    const float v = as_type<float>(bits);
    for (uint s = 0; s < a->hc; ++s) a->h[((size_t) t * a->hc + s) * a->dim + e] = v;
}

// (diagnostic) write a constant through a device pointer passed as a value, and one passed in a buffer
kernel void ds4_test_write(device float* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n) out[gid] = 3.0f;
}
kernel void ds4_test_write_v(device const uint64_t* p [[buffer(0)]], constant uint& n [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    device float* out = (device float*) (uintptr_t) p[0];
    if (gid < n) out[gid] = 7.0f;
}

// (diagnostic) report the arguments the kernel received
kernel void ds4_args_probe(device float* out [[buffer(0)]], constant uint& dim [[buffer(1)]],
                           constant uint& hc [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid == 0) { out[0] = float(dim); out[1] = float(hc); out[2] = 1.0f; }
}

// hc == 4 (every layer of this model): the warp-lane form of the CUDA kernel - lane j*4+k holds comb[j][k], the row and column
// sums are xor-shuffles (1, 2 within a row; 4, 8 within a column).  One simdgroup a token instead of one lane.
kernel void ds4_hc_split4(device const float* mixes [[buffer(0)]], device const float* scale [[buffer(1)]],
                          device const float* base [[buffer(2)]], device float* pre [[buffer(3)]],
                          device float* post [[buffer(4)]], device float* comb [[buffer(5)]],
                          constant uint& hc [[buffer(6)]], constant uint& iters [[buffer(7)]],
                          constant float& eps [[buffer(8)]], uint b [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_threadgroup]]) {
    const device float* mx = mixes + (size_t) b * 32u;
    if (lane < 4u) {
        pre[(size_t) b * 8u + lane] = 1.0f / (1.0f + exp(-(mx[lane] * scale[0] + base[lane]))) + eps;
        post[(size_t) b * 8u + lane] = 2.0f / (1.0f + exp(-(mx[4u + lane] * scale[1] + base[4u + lane])));
    }
    const bool on = lane < 16u;
    float c = on ? mx[8u + lane] * scale[2] + base[8u + lane] : -3.402823466e+38f;
    float m = fmax(c, simd_shuffle_xor(c, ushort(1)));
    m = fmax(m, simd_shuffle_xor(m, ushort(2)));
    c = on ? exp(c - m) : 0.0f;
    float rs = c + simd_shuffle_xor(c, ushort(1));
    rs += simd_shuffle_xor(rs, ushort(2));
    c = c / rs + eps;
    c = on ? c : 0.0f;
    float cs = c + simd_shuffle_xor(c, ushort(4));
    cs += simd_shuffle_xor(cs, ushort(8));
    c = c / (cs + eps);
    for (uint it = 1; it < iters; ++it) {
        float r = c + simd_shuffle_xor(c, ushort(1));
        r += simd_shuffle_xor(r, ushort(2));
        c = c / (r + eps);
        float s = c + simd_shuffle_xor(c, ushort(4));
        s += simd_shuffle_xor(s, ushort(8));
        c = c / (s + eps);
    }
    if (on) comb[(size_t) b * 64u + lane] = c;
}

// Plain byte copies and fills on the context's queue: cudaMemcpyAsync / cudaMemsetAsync between runtime allocations (unified
// memory: a "device" and a "host" buffer are both just memory) ride in the same command buffer as the kernels around them,
// so their order is the program order and a layer needs no queue-to-queue handshake.
kernel void ds4_copy4(device const uint* src [[buffer(0)]], device uint* dst [[buffer(1)]], constant ulong& n4 [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    if (ulong(gid) < n4) dst[gid] = src[gid];
}
kernel void ds4_copy1(device const uchar* src [[buffer(0)]], device uchar* dst [[buffer(1)]], constant ulong& n [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    if (ulong(gid) < n) dst[gid] = src[gid];
}
kernel void ds4_fill1(device uchar* dst [[buffer(0)]], constant ulong& n [[buffer(1)]], constant uint& v [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    if (ulong(gid) < n) dst[gid] = uchar(v);
}

// ---------------------------------------------------------------------------------------------------------------------
// The prompt chunk's dense GEMM operands (see ds4_gemm_f16_nt on the host): float rows -> half, a quantized matrix -> half rows,
// strided half rows packed, and the float result scattered into the engine's strided output.

kernel void ds4_rows_f16(device const float* x [[buffer(0)]], constant ulong& ldx [[buffer(1)]],
                         device const int* idx [[buffer(2)]], constant uint& has_idx [[buffer(3)]],
                         constant uint& ncols [[buffer(4)]], device half* out [[buffer(5)]],
                         uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint r = tgp.y, c = tgp.x * 256u + tid;
    if (c >= ncols) return;
    const ulong row = has_idx != 0u ? ulong(idx[r]) : ulong(r);
    out[(ulong) r * ncols + c] = half(x[row * ldx + c]);
}

static inline void ds4_kscale2(const device uchar* q, int j, thread int& d, thread int& m) {
    if (j < 4) {
        d = int(q[j]) & 63;
        m = int(q[j + 4]) & 63;
    } else {
        d = (int(q[j + 4]) & 0xf) | ((int(q[j - 4]) >> 6) << 4);
        m = (int(q[j + 4]) >> 4) | ((int(q[j]) >> 6) << 4);
    }
}

// a thread a 32-value sub-block of a Q5_K matrix (n rows of k values, ld bytes a row) -> out[n][k] half
kernel void ds4_dq_q5k(device const uchar* W [[buffer(0)]], constant ulong& ld [[buffer(1)]], constant uint& k [[buffer(2)]],
                       constant uint& n [[buffer(3)]], device half* out [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    const uint nb = k / 256u;
    const ulong total = (ulong) n * nb * 8ul;
    if (ulong(gid) >= total) return;
    const uint ib = gid & 7u;
    const uint blk = (gid >> 3) % nb;
    const uint row = uint(ulong(gid) / (8ul * nb));
    const device uchar* p = W + (ulong) row * ld + (ulong) blk * 176ul;
    const float d = float(as_type<half>(*(const device ushort*) p));
    const float dmin = float(as_type<half>(*(const device ushort*) (p + 2)));
    int sc, mm;
    ds4_kscale2(p + 4, int(ib), sc, mm);
    const device uchar* qh = p + 16;
    const device uchar* qs = p + 48 + 32u * (ib >> 1);
    const uint sh = 4u * (ib & 1u);
    device packed_half4* o = (device packed_half4*) (out + (ulong) row * k + (ulong) blk * 256ul + ib * 32u);
    const float ds = d * float(sc), dm = dmin * float(mm);
    for (uint l = 0; l < 32u; l += 4u) {   // four values a step: 4-byte loads, 8-byte stores (the same arithmetic a value)
        const uint4 a = uint4(*(const device packed_uchar4*) (qs + l));
        const uint4 b = uint4(*(const device packed_uchar4*) (qh + l));
        const uint4 q = ((a >> sh) & 15u) | (((b >> ib) & 1u) << 4);
        o[l >> 2] = packed_half4(half4(float4(ds) * float4(q) - float4(dm)));
    }
}

// Q6_K (210-byte blocks: ql[128], qh[64], int8 scales[16], f16 d; the Unsloth sets' attention, shared-expert, head and engram_kv
// matrices): a thread four values of a block (two 128-value halves x 32 lanes), ggml's dequantize_row_q6_K arithmetic
// ((d * scale) * (q - 32), in this order) -> out[n][k] half
kernel void ds4_dq_q6k(device const uchar* W [[buffer(0)]], constant ulong& ld [[buffer(1)]], constant uint& k [[buffer(2)]],
                       constant uint& n [[buffer(3)]], device half* out [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    const uint nb = k / 256u;
    const ulong total = (ulong) n * nb * 64ul;
    if (ulong(gid) >= total) return;
    const uint l = gid & 31u;
    const uint hf = (gid >> 5) & 1u;
    const uint blk = (gid >> 6) % nb;
    const uint row = uint(ulong(gid) / (64ul * nb));
    const device uchar* p = W + (ulong) row * ld + (ulong) blk * 210ul;
    const float d = float(as_type<half>(*(const device ushort*) (p + 208)));
    const device uchar* ql = p + 64u * hf;
    const device uchar* qh = p + 128u + 32u * hf;
    const device char* sc = (const device char*) (p + 192u + 8u * hf);
    const uint is = l >> 4;
    const uint qlo = uint(ql[l]), qhi = uint(ql[l + 32u]), h = uint(qh[l]);
    const int q1 = int((qlo & 0xfu) | (((h >> 0) & 3u) << 4)) - 32;
    const int q2 = int((qhi & 0xfu) | (((h >> 2) & 3u) << 4)) - 32;
    const int q3 = int((qlo >> 4) | (((h >> 4) & 3u) << 4)) - 32;
    const int q4 = int((qhi >> 4) | (((h >> 6) & 3u) << 4)) - 32;
    device half* o = out + (ulong) row * k + (ulong) blk * 256ul + 128ul * hf + l;
    o[0] = half(d * float(sc[is + 0u]) * float(q1));
    o[32] = half(d * float(sc[is + 2u]) * float(q2));
    o[64] = half(d * float(sc[is + 4u]) * float(q3));
    o[96] = half(d * float(sc[is + 6u]) * float(q4));
}

// Q8_0: 34-byte blocks (fp16 d, 32 int8)
kernel void ds4_dq_q8(device const uchar* W [[buffer(0)]], constant ulong& ld [[buffer(1)]], constant uint& k [[buffer(2)]],
                      constant uint& n [[buffer(3)]], device half* out [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    const uint ng = k / 32u;
    const ulong total = (ulong) n * ng;
    if (ulong(gid) >= total) return;
    const uint g = gid % ng, row = gid / ng;
    const device uchar* p = W + (ulong) row * ld + (ulong) g * 34ul;
    const float d = float(as_type<half>(*(const device ushort*) p));
    device half* o = out + (ulong) row * k + (ulong) g * 32ul;
    for (uint l = 0; l < 32u; ++l) o[l] = half(d * float(char(p[2 + l])));
}

// BF16 values (ld values a row) -> half
kernel void ds4_dq_bf16(device const ushort* W [[buffer(0)]], constant ulong& ld [[buffer(1)]], constant uint& k [[buffer(2)]],
                        constant uint& n [[buffer(3)]], device half* out [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    const ulong total = (ulong) n * k;
    if (ulong(gid) >= total) return;
    const ulong r = ulong(gid) / k, c = ulong(gid) - r * k;
    out[gid] = half(as_type<float>(uint(W[r * ld + c]) << 16));
}

// half rows with a row stride -> contiguous rows
kernel void ds4_pack_h16(device const ushort* x [[buffer(0)]], constant ulong& ld [[buffer(1)]], constant uint& k [[buffer(2)]],
                         device ushort* out [[buffer(3)]], uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint c = tgp.x * 256u + tid;
    if (c < k) out[(ulong) tgp.y * k + c] = x[(ulong) tgp.y * ld + c];
}

// y[r][c] = alpha * t[r][c] + beta * y[r][c]
kernel void ds4_axpby_rows(device const float* t [[buffer(0)]], device float* y [[buffer(1)]], constant ulong& ldy [[buffer(2)]],
                           constant uint& n [[buffer(3)]], constant float& alpha [[buffer(4)]], constant float& beta [[buffer(5)]],
                           uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint c = tgp.x * 256u + tid;
    if (c >= n) return;
    const ulong yi = (ulong) tgp.y * ldy + c;
    y[yi] = alpha * t[(ulong) tgp.y * n + c] + (beta != 0.0f ? beta * y[yi] : 0.0f);
}

// ds4_q8_quant for several jobs in one dispatch (a matvec call's activations): grid (groups, max rows, jobs)
struct Ds4Q8Spec {
    device const float* x;
    ulong ldx;
    uint rows;
    uint k;
    device char* q;
    device float* d;
};
kernel void ds4_q8_quant_multi(device const Ds4Q8Spec* specs [[buffer(0)]], uint3 tgp [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
    const device Ds4Q8Spec& s = specs[tgp.z];
    const uint warp = tid / 32u, lane = tid % 32u;
    const uint b = tgp.x * 8u + warp;
    const uint r = tgp.y;
    if (r >= s.rows || (size_t) b * 32u >= (size_t) s.k) return;
    const device float* xr = s.x + (size_t) r * s.ldx;
    const float v = xr[(size_t) b * 32u + lane];
    const float am = simd_max(fabs(v));
    const float sc = am / 127.0f;
    s.q[(size_t) r * s.k + (size_t) b * 32u + lane] = char(am > 0.0f ? rint(v / sc) : 0.0f);
    if (lane == 0u) s.d[(size_t) r * (s.k / 32u) + b] = sc;
}
