// src/kernels/cuda/iq_kernels.cu - see include/strata/kernels/iq_kernels.hpp.
//
// The dot products (vec_dot_*_q8_1), the dequantizers and the q8_1 quantizer are transcribed from llama.cpp
// (ggml/src/ggml-cuda/vecdotq.cuh, dequantize.cuh, quantize.cu at the commit in third_party/ggml/VERSION.txt;
// MIT license, third_party/ggml/LICENSE).  The block structs and codebook grids come from its ggml-common.h,
// included unchanged.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/core/expert_nibble.hpp"
#include "strata/core/sm75_series.hpp"
#include "strata/core/sm120_series.hpp"
#include "strata/kernels/gfx_arch.hpp"
#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/q8_1_finite.hpp"
#include "s26_tsum.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"
#include "ggml-common-tq.h"   // APR trellis types (agention llama.cpp fork)
#include "tq_lut_i8.cuh"       // int8 copy of the trellis codebook (one global scale) for the dp4a dot

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <utility>
#include <cstdlib>
#include <map>
#include <mutex>
#include <tuple>
#include <type_traits>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

// ---------------------------------------------------------------- llama.cpp helpers (vecdotq.cuh)
__device__ __forceinline__ int get_int_b2(const void* x, const int& i32) {
    const uint16_t* x16 = (const uint16_t*) x;
    int x32 = x16[2 * i32 + 0] << 0;
    x32 |= x16[2 * i32 + 1] << 16;
    return x32;
}
__device__ __forceinline__ int get_int_b4(const void* x, const int& i32) { return ((const int*) x)[i32]; }
__device__ __forceinline__ uint32_t unpack_ksigns(const uint8_t v) {
    const uint32_t p = __popc(v) & 1;
    const uint32_t s = v ^ p << 7;
    return s * 0x01010101;
}
__device__ __forceinline__ int2 get_int_from_table_16(const int& q4, const int8_t* table) {
#if defined(STRATA_HIP_GFX906)
    // AMD: llama.cpp's HIP lookup (vecdotq.cuh) - v_perm_b32 takes 3-bit byte indices, so the low and high halves
    // of the table are looked up and the index MSB picks between them: 4 perms per 8 values.
    const uint32_t* v32 = reinterpret_cast<const uint32_t*>(table);
    const uint32_t q_even = (uint32_t) q4, q_odd = (uint32_t) q4 >> 4;
    const uint32_t el = __builtin_amdgcn_perm(v32[1], v32[0], q_even & 0x07070707u);
    const uint32_t ol = __builtin_amdgcn_perm(v32[1], v32[0], q_odd & 0x07070707u);
    const uint32_t eh = __builtin_amdgcn_perm(v32[3], v32[2], q_even & 0x07070707u);
    const uint32_t oh = __builtin_amdgcn_perm(v32[3], v32[2], q_odd & 0x07070707u);
    return make_int2((int) __builtin_amdgcn_perm(eh, el, 0x03020100u | ((q_even & 0x08080808u) >> 1)),
                     (int) __builtin_amdgcn_perm(oh, ol, 0x03020100u | ((q_odd & 0x08080808u) >> 1)));
#else
    const uint32_t* table32 = (const uint32_t*) table;
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
#endif
}
#define ggml_cuda_dp4a(a, b, c) STRATA_DP4A((a), (b), (c))

// ---------------------------------------------------------------- the dot products (vecdotq.cuh)
__device__ __forceinline__ float vec_dot_q2_0_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q2_0* bq2_0 = (const block_q2_0*) vbq + kbx;
    const float d2 = bq2_0->d;
    const int16_t* qs = (const int16_t*) bq2_0->qs + iqs * 4;
    const block_q8_1* bq8_1_chunk = bq8_1 + iqs;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = qs[j];
        const int u = get_int_b4(bq8_1_chunk->qs, j * 2 + 0);
        const int v = get_int_b4(bq8_1_chunk->qs, j * 2 + 1);
        const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
        const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
        const int qx = __byte_perm(qe, qo, 0x5140);
        const int qy = __byte_perm(qe, qo, 0x7362);
        sumi = ggml_cuda_dp4a(u, qx, sumi);
        sumi = ggml_cuda_dp4a(v, qy, sumi);
    }
    const float d8 = __low2float(bq8_1_chunk->ds);
    return d2 * d8 * sumi;
}

__device__ __forceinline__ float vec_dot_iq2_xxs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                      const int& kbx, const int& iqs) {
    const block_iq2_xxs* bq2 = (const block_iq2_xxs*) vbq + kbx;
    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t* aux8 = (const uint8_t*) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = ((const uint2*) iq2xxs_grid)[aux8[k0 / 2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 0);
        sumi = ggml_cuda_dp4a(grid0, u0, sumi);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 1);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }
    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq2_xs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                     const int& kbx, const int& iqs) {
    const block_iq2_xs* bq2 = (const block_iq2_xs*) vbq + kbx;
    const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
    const uint16_t* q2 = (const uint16_t*) &q2_packed;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = ((const uint2*) iq2xs_grid)[q2[l0 / 2] & 0x1FF];
        const uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq2_s_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_iq2_s* bq2 = (const block_iq2_s*) vbq + kbx;
    const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq2->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int* grid_pos = (const int*) (iq2s_grid + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
        const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4(grid_pos[0] ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos[1] ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq3_xxs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                      const int& kbx, const int& iqs) {
    const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* q3 = (const uint8_t*) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 2;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq3_s_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
    const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq3->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                                        iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
        const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    sumi *= 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq1_m_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_iq1_m* bq1 = (const block_iq1_m*) vbq + kbx;
    const int qs_packed = get_int_b4(bq1->qs, iqs);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    int sumi[2] = {0, 0};
    float sumf[2] = {0.0f, 0.0f};
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qhl = bq1->qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
        const int grid = iq1s_grid_gpu[qs[l0 / 2] | ((qhl & 0x07) << 8)];
        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;
        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
        sumi[l0 / 4] = ggml_cuda_dp4a(grid0, u0, sumi[l0 / 4]);
        sumi[l0 / 4] = ggml_cuda_dp4a(grid1, u1, sumi[l0 / 4]);
        const float delta = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f * IQ1M_DELTA / 0x08);
        int sumy = 0;
        sumy = ggml_cuda_dp4a(u0, 0x01010101, sumy);
        sumy = ggml_cuda_dp4a(u1, 0x01010101, sumy);
        sumf[l0 / 4] += delta * sumy;
    }
    const uint16_t* sc = (const uint16_t*) bq1->scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    const float d = __half2float(scale.f16) * __low2float(bq8_1[iqs].ds);
    const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
    const int sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
    const int sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
    return d * ((sumi[0] + sumf[0]) * sc0 + (sumi[1] + sumf[1]) * sc1);
}

__device__ __forceinline__ float vec_dot_iq4_nl_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                     const int& kbx, const int& iqs) {
    const block_iq4_nl* bq4 = (const block_iq4_nl*) vbq + kbx;
    const int* q8 = (const int*) bq8_1->qs + iqs;
    int sumi = 0;
#pragma unroll
    for (int l = 0; l < 2; ++l) {
        const int aux_q4 = get_int_b2(bq4->qs, iqs + l);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);
    }
    const float d = __half2float(bq4->d) * __low2float(bq8_1->ds);
    return d * sumi;
}

// IQ4_XS: 256 values as 8 sub-blocks of 32 (6-bit scale each); one call covers one sub-block (iqs = 4 * sub-block),
// and `bq8_1` is the super-block's first q8_1 block, so the call's activation is bq8_1[iqs / 4].  The GSQ-RCO IQ3_S
// file keeps one layer's routed gate/up experts in this format.
__device__ __forceinline__ float vec_dot_iq4_xs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                     const int& kbx, const int& iqs) {
    const block_iq4_xs* bq4 = (const block_iq4_xs*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        const int u0 = get_int_b4(bq8_1[iqs / 4].qs, j + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 4].qs, j + 4);
        sumi = ggml_cuda_dp4a(v.x, u0, sumi);
        sumi = ggml_cuda_dp4a(v.y, u1, sumi);
    }
    const int ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    const float d = __half2float(bq4->d) * __low2float(bq8_1[iqs / 4].ds);
    return d * sumi;
}

// ---------------------------------------------------------------- Unsloth's UD-Q4_K_XL experts
// Q4_K / Q5_K gate/up and Q5_1 / Q8_0 down: llama.cpp's vec_dot_*_q8_1 (vecdotq.cuh, VDR 2 each), transcribed; the
// Q5_1 min term is the one departure (below).  Via eddoursul/Strata 8029fa9 (iq_dot.cuh) and #255 (Q8_0,
// gopinath87607), which agree with llama.cpp and with each other.
constexpr int VDR_Q4_K = 2, VDR_Q5_K = 2, VDR_Q5_1 = 2, VDR_Q5_0 = 2, VDR_Q4_1 = 2, VDR_Q4_0 = 2, VDR_Q8_0 = 2, VDR_Q6_K = 1;

__device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(const int* __restrict__ v, const int* __restrict__ u,
                                                             const uint8_t* __restrict__ sc, const uint8_t* __restrict__ m,
                                                             const half2& dm4, const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;
        const int dot1 = ggml_cuda_dp4a(v1i, u[2 * i + 1], ggml_cuda_dp4a(v0i, u[2 * i + 0], 0));
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2 * i + 1], ggml_cuda_dp4a(0x01010101, u[2 * i + 0], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);   // the min times the sum of the QUANTIZED activations
    }
    const float2 dm4f = __half22float2(dm4);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}
__device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_vmmq(const int* __restrict__ vl, const int* __restrict__ vh,
                                                             const int* __restrict__ u, const uint8_t* __restrict__ sc,
                                                             const uint8_t* __restrict__ m, const half2& dm5,
                                                             const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0F0F0F0F;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = ggml_cuda_dp4a(v0i, u[2 * i + 0], ggml_cuda_dp4a(v1i, u[2 * i + 1], 0));
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2 * i + 0], ggml_cuda_dp4a(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm5f = __half22float2(dm5);
    return dm5f.x * sumf_d - dm5f.y * sumf_m;
}
// the 6-bit scales and mins of the 32-value group pair bq8_offset / 2, branchless (llama.cpp; shared by Q4_K, Q5_K)
__device__ __forceinline__ void k_scale_min(const uint8_t* scales8, int bq8_offset, uint16_t aux[2]) {
    const uint16_t* scales = (const uint16_t*) scales8;
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm + 0];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = (uint32_t) -(int32_t) (j >= 2);
    aux[0] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
}
__device__ __forceinline__ float vec_dot_q4_K_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q4_K* bq4_K = (const block_q4_K*) vbq + kbx;
    int v[2];
    int u[2 * QR4_K];
    float d8[QR4_K];
    const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
    const int* q4 = (const int*) (bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = q4[0];
    v[1] = q4[4];
    uint16_t aux[2];
    k_scale_min(bq4_K->scales, bq8_offset, aux);
    const uint8_t* sc = (const uint8_t*) aux;
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
}
__device__ __forceinline__ float vec_dot_q5_K_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q5_K* bq5_K = (const block_q5_K*) vbq + kbx;
    int vl[2];
    int vh[2];
    int u[2 * QR5_K];
    float d8[QR5_K];
    const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
    const int* ql = (const int*) (bq5_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const int* qh = (const int*) (bq5_K->qh + 4 * ((iqs / 2) % 4));
    vl[0] = ql[0];
    vl[1] = ql[4];
    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;
    uint16_t aux[2];
    k_scale_min(bq5_K->scales, bq8_offset, aux);
    const uint8_t* sc = (const uint8_t*) aux;
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);
}

// Q3_K gate/up: llama.cpp's vec_dot_q3_K_q8_1 (vecdotq.cuh), the mmvq variant with VDR = 2: each call reads TWO
// qs words (vl[0] = word iqs, vl[1] = word iqs+1) and the matching two hmask words, against QR3_K=4 q8_1 blocks.
// The three shifted scale ints carry the 6-bit signed scales (sc-32).  Fmt<11> calls this with iqs = 0, 2, .., 14
// (QI3_K = QK_K/(4*QR3_K) = 16, QI8_1 = 8, ipb = QI3_K/VDR = 8, step = 2), so the two words of every call span all
// 16 qs words of the 256-value block.  The previous VDR = 1 body read ONE word per call and therefore only the even
// words (half of every block) - exactly the wrong-output bug this replaces.
constexpr int VDR_Q3_K = 2;
__device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmvq(const int* __restrict__ vl, const int* __restrict__ vh,
                                                             const int* __restrict__ u, const int* __restrict__ scales,
                                                             const float& d3, const float* __restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        const int sc_low  = (scales[i / 2] >> (16 * (i % 2))) & 0x0F;
        const int sc_high = (scales[2] >> (16 * (i % 2) + 2 * (i / 2))) & 0x03;
        const int sc = (sc_low | (sc_high << 4)) - 32;              // 6-bit signed
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < VDR_Q3_K; ++j) {
            const int vil = (vl[j] >> (2 * i)) & 0x03030303;        // 2 low bits per value
            const int vih = ((vh[j] >> i) << 2) & 0x04040404;       // high bit -> 4
            sumi = ggml_cuda_dp4a(vil | vih,  u[VDR_Q3_K * i + j], sumi);
            sumi = ggml_cuda_dp4a((int) 0xFCFCFCFC, u[VDR_Q3_K * i + j], sumi);   // -4 per value
        }
        sumf += d8[i] * (sumi * sc);
    }
    return d3 * sumf;
}
__device__ __forceinline__ float vec_dot_q3_K_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q3_K* bq3_K = (const block_q3_K*) vbq + kbx;
    const int bq8_offset = QR3_K * (iqs / (QI3_K / 2));
    const int iqs8 = iqs % QI8_1;
    const float d = bq3_K->d;
    int vl[VDR_Q3_K];
    int vh[VDR_Q3_K];
#pragma unroll
    for (int j = 0; j < VDR_Q3_K; ++j) {
        vl[j] = get_int_b2(bq3_K->qs, iqs + j);
        vh[j] = get_int_b2(bq3_K->hmask, iqs8 + j) >> bq8_offset;
    }
    const int sc_shift = 8 * (iqs8 / (QI8_1 / 2)) + 4 * (iqs / (QI3_K / 2));
    int scales[3];
#pragma unroll
    for (int l = 0; l < 3; ++l) scales[l] = get_int_b2(bq3_K->scales, l) >> sc_shift;
    int u[VDR_Q3_K * QR3_K];
    float d8[QR3_K];
#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
#pragma unroll
        for (int j = 0; j < VDR_Q3_K; ++j) {
            u[VDR_Q3_K * i + j] = get_int_b4(bq8_1[bq8_offset + i].qs, iqs8 + j);
        }
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }
    return vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, scales, d, d8);
}

// Q5_1: llama.cpp's integer chain, but the min term multiplies the sum of the QUANTIZED activations (dp4a with
// 0x01010101, times d8) instead of the q8_1 block's `ds.y`, which our quantizer (like llama.cpp's) fills with the sum
// of the ORIGINAL activations.  That is ggml-cpu's convention (its q8_1 `s` is d * sum(q)) and the one the K-quant
// mins above use; the scaled and the min term then see the same activation (eddoursul/Strata measured 1.1-1.2%
// against 1.9% relative error per expert).  Result: sumi * (d5 * d8) + sumu * (m5 * d8).
// Q5_0: llama.cpp's integer chain verbatim (vecdotq.cuh vec_dot_q5_0_q8_1_impl) -- symmetric (one delta, no
// learned min), so unlike Q5_1 there is no min-term rounding choice to make: the constant -16 per-weight offset
// is accounted for via the q8_1 block's `ds.y` (sum of the ORIGINAL activations), exactly as upstream.
__device__ __forceinline__ float vec_dot_q5_0_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q5_0* bq5_0 = (const block_q5_0*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q5_0; ++i) {
        const int vl = get_int_b2(bq5_0->qs, iqs + i);
        const int vh = get_int_b2(bq5_0->qh, 0) >> (4 * (iqs + i));
        const int u0 = get_int_b4(bq8_1->qs, iqs + i), u1 = get_int_b4(bq8_1->qs, iqs + i + QI5_0);
        int vi0 = (vl >> 0) & 0x0F0F0F0F;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi0, u0, sumi);
        int vi1 = (vl >> 4) & 0x0F0F0F0F;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi1, u1, sumi);
    }
    const float d5 = __half2float(bq5_0->d);
    const float d8 = __low2float(bq8_1->ds);
    const float s8 = __high2float(bq8_1->ds);
    return d5 * (sumi * d8 - (16.0f * VDR_Q5_0 / QI5_0) * s8);
}
// Q4_0 (plain llama-quantize Q4_0 files, e.g. bartowski's: gate/up and most down projections): the codes are centred
// (q - 8 as signed bytes) before the dot, the exact integer form ggml-cpu's q4_0 x q8_0 dot uses, rather than
// llama.cpp's CUDA chain (uncentred codes minus 8 x the q8_1 block's float sum `ds.y`).  On a real Q4_0 expert
// (2560 x 640) the CUDA chain measured 1.4e-2 relative error against the float product, the centred form 5.3e-3;
// with the CUDA chain native_expert_parity failed its 3e-2 limit on two layers of a Q4_0 Flash-Next file.  The
// centring is the same idea as the Q5_1 min term above: the offset is applied to the activations' own int8 codes.
__device__ __forceinline__ int q4_0_centred(const int v) {
    // Unsigned: (b & 0x08080808) * 0x1E reaches 0xF0F0F0F0, which overflows a signed int - undefined behaviour the
    // sm_60 build (no __dp4a, strata_dp4a fallback) optimised into a wrong value (native_expert_parity gpu rel 1.7e4).
    const uint32_t b = (uint32_t) v ^ 0x08080808u;   // per byte: 0..15 -> (q - 8) as a 4-bit two's complement value
    return (int) (b | ((b & 0x08080808u) * 0x1Eu));  // sign-extend each nibble into its byte (0x08 * 0x1E = 0xF0)
}
__device__ __forceinline__ float vec_dot_q4_0_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q4_0* bq4 = (const block_q4_0*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q4_0; ++i) {
        const int v = get_int_b2(bq4->qs, iqs + i);
        sumi = ggml_cuda_dp4a(q4_0_centred((v >> 0) & 0x0F0F0F0F), get_int_b4(bq8_1->qs, iqs + i), sumi);
        sumi = ggml_cuda_dp4a(q4_0_centred((v >> 4) & 0x0F0F0F0F), get_int_b4(bq8_1->qs, iqs + i + QI4_0), sumi);
    }
    return __half2float(bq4->d) * __low2float(bq8_1->ds) * sumi;
}
// Q4_1 (llama-quantize puts it on a few down projections of its Q4_0 files): the Q5_1 chain above without the fifth
// bit, the same min-term choice (the activations' own int8 codes): sumi * (d4 * d8) + sumu * (m4 * d8).
__device__ __forceinline__ float vec_dot_q4_1_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q4_1* bq4_1 = (const block_q4_1*) vbq + kbx;
    int sumi = 0, sumu = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q4_1; ++i) {
        const int v = get_int_b4(bq4_1->qs, iqs + i);
        const int u0 = get_int_b4(bq8_1->qs, iqs + i), u1 = get_int_b4(bq8_1->qs, iqs + i + QI4_1);
        sumi = ggml_cuda_dp4a((v >> 0) & 0x0F0F0F0F, u0, sumi);
        sumi = ggml_cuda_dp4a((v >> 4) & 0x0F0F0F0F, u1, sumi);
        sumu = ggml_cuda_dp4a(0x01010101, u1, ggml_cuda_dp4a(0x01010101, u0, sumu));
    }
    const float2 dm4 = __half22float2(bq4_1->dm);
    const float d8 = __low2float(bq8_1->ds);
    return sumi * (dm4.x * d8) + sumu * (dm4.y * d8);
}
__device__ __forceinline__ float vec_dot_q5_1_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q5_1* bq5_1 = (const block_q5_1*) vbq + kbx;
    int sumi = 0, sumu = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q5_1; ++i) {
        const int vl = get_int_b4(bq5_1->qs, iqs + i);
        const int vh = get_int_b4(bq5_1->qh, 0) >> (4 * (iqs + i));
        const int u0 = get_int_b4(bq8_1->qs, iqs + i), u1 = get_int_b4(bq8_1->qs, iqs + i + QI5_1);
        int vi0 = (vl >> 0) & 0x0F0F0F0F;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi0, u0, sumi);
        int vi1 = (vl >> 4) & 0x0F0F0F0F;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi1, u1, sumi);
        sumu = ggml_cuda_dp4a(0x01010101, u1, ggml_cuda_dp4a(0x01010101, u0, sumu));
    }
    const float2 dm5 = __half22float2(bq5_1->dm);
    const float d8 = __low2float(bq8_1->ds);
    return sumi * (dm5.x * d8) + sumu * (dm5.y * d8);
}
// Q6_K: llama.cpp's integer chain verbatim (vecdotq.cuh vec_dot_q6_K_q8_1_impl_mmvq / vec_dot_q6_K_q8_1) -- signed
// per-16-element scales, the -32 offset folded into the per-byte value, the q8_1 block's ds.low for the scales,
// upstream exactly as pulled (no min-term choice exists: Q6_K has no learned min).
__device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmvq(const int vl, const int vh, const int* __restrict__ u,
                                                             const int8_t* __restrict__ scales, const float d,
                                                             const float* __restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = __vsubss4(vil | vih, 0x20202020);   // vi = (vil | vih) - 32
        sumf += d8[i] * (STRATA_DP4A(vi, u[i], 0) * sc);
    }
    return d * sumf;
}
__device__ __forceinline__ float vec_dot_q6_K_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q6_K* bq6_K = (const block_q6_K*) vbq + kbx;
    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
    const int scale_offset = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
    const int vh_shift = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));
    const int vl = get_int_b2(bq6_K->ql, iqs);
    const int vh = get_int_b2(bq6_K->qh, (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4)) >> vh_shift;
    const int8_t* scales = bq6_K->scales + scale_offset;
    int u[QR6_K];
    float d8[QR6_K];
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        u[i] = get_int_b4(bq8_1[bq8_offset + 2 * i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + 2 * i].ds);
    }
    return vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, scales, __half2float(bq6_K->d), d8);
}
__device__ __forceinline__ float vec_dot_q8_0_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q8_0* bq8_0 = (const block_q8_0*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q8_0; ++i)
        sumi = ggml_cuda_dp4a(get_int_b2(bq8_0->qs, iqs + i), get_int_b4(bq8_1->qs, iqs + i), sumi);
    const float d8_0 = __half2float(bq8_0->d), d8_1 = __low2float(bq8_1->ds);
    return d8_0 * d8_1 * ((float) sumi);
}
// NVFP4 (ModelOpt): 64 values = 4 sub-blocks of 16, one UE4M3 scale each, E2M1 codes split-half like Q4_0.
// kvalues_fp4 holds the E2M1 values doubled (0, 1, 2, 3, 4, 6, 8, 12), so the scale is halved here - exactly
// ggml's ggml_cuda_ue4m3_to_fp32, decoded in software so the GPU and ggml-cpu agree bit for bit (NaN -> 0).
__device__ __forceinline__ float ue4m3_to_f32_half(uint8_t x) {
    // The same value as ldexpf(1 + man/8, exp - 7) * 0.5 (ldexpf(man, -9) * 0.5 for exp 0), built from its bits: every
    // such value is exact in FP32, so this is equal for all 256 inputs - without the ldexpf and the branches it cost
    // in the decode experts' inner loop.  Normal: 2^(exp-8) * (1 + man/8) = exponent field exp + 119, mantissa man<<20.
    if (x == 0 || x == 0x7F) return 0.0f;
    const uint32_t exp = (x >> 3) & 0xF, man = x & 0x7;
    return exp == 0 ? (float) man * 0.0009765625f : __uint_as_float(((exp + 119u) << 23) | (man << 20));
}

// One call = two sub-blocks (32 values); `bq8_1` is the block's first q8_1 block (two per 64 values).
__device__ __forceinline__ float vec_dot_nvfp4_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_nvfp4* bq4 = (const block_nvfp4*) vbq + kbx;
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int iqs0 = iqs + 2 * i, iqs1 = iqs0 + 1, is = iqs0 >> 1;
        const int2 v0 = get_int_from_table_16(get_int_b4(bq4->qs, iqs0), kvalues_fp4);
        const int2 v1 = get_int_from_table_16(get_int_b4(bq4->qs, iqs1), kvalues_fp4);
        const block_q8_1* bq8 = bq8_1 + (is >> 1);
        const int i8 = (is & 1) << 2;
        int sumi = ggml_cuda_dp4a(v0.x, get_int_b4(bq8->qs, i8 + 0), 0);
        sumi = ggml_cuda_dp4a(v0.y, get_int_b4(bq8->qs, i8 + 2), sumi);
        sumi = ggml_cuda_dp4a(v1.x, get_int_b4(bq8->qs, i8 + 1), sumi);
        sumi = ggml_cuda_dp4a(v1.y, get_int_b4(bq8->qs, i8 + 3), sumi);
        sum += ue4m3_to_f32_half(bq4->d[is]) * __low2float(bq8->ds) * (float) sumi;
    }
    return sum;
}

// ---------------------------------------------------------------- APR trellis types (ggml 144 TQ2_T, 145 TQK6, 146 TQK7)
// Transcribed from the agention llama.cpp fork's ggml-cuda/tq.cuh (6d3f2b1b8; format in ggml-common-tq.h).  A
// 128-value block is 8 lanes of 4 trellis steps (16 values); the dot of lane l reads q8_1 block l/2, ints 4(l&1)..
template<int TY> struct TqInfo;
template<> struct TqInfo<144> { using block_t = block_tq2_t; };   // byte-aligned, no K
template<> struct TqInfo<145> { using block_t = block_tqk6; static constexpr int K = 6; };
template<> struct TqInfo<146> { using block_t = block_tqk7; static constexpr int K = 7; };
// Gyro-L: pair trellis (64 steps x K bits, 2 weights per step), the TQ2_T codebook's first pair
template<> struct TqInfo<147> { using block_t = block_tqp5; static constexpr int K = 5; };
template<> struct TqInfo<148> { using block_t = block_tqp6; static constexpr int K = 6; };
template<> struct TqInfo<149> { using block_t = block_tqp7; static constexpr int K = 7; };
__device__ __forceinline__ float tq_h2f(uint32_t bits) { return __half2float(__ushort_as_half((unsigned short) bits)); }
// the four unscaled weights of the step with state s: (a.x, a.y, b.x, b.y)
__device__ __forceinline__ void tq_step(uint32_t s, float2& a, float2& b) {
    const uint32_t* lut = (const uint32_t*) tq_lut_f16;
    const uint32_t x = (s & TQ_STATE_MASK) * 0x9e3779b1u;
    const uint32_t pa = lut[x >> 21], pb = lut[(x >> 10) & 2047u];
    a = make_float2(tq_h2f(pa & 0xFFFFu), tq_h2f(pa >> 16));
    b = make_float2(tq_h2f(pb & 0xFFFFu), tq_h2f(pb >> 16));
}
// the states of steps 4l..4l+3 of one block (qs is 2-byte aligned: every trellis block has an even size)
// Compute capability 6.0 (Tesla P100): the block's words are read with ld.global.nc (the weights are read-only; a generic
// pointer made these LD.E), and TQK6's lane window needs only three words (34 bits from a bit offset of 0 or 8), TQK7's
// four.  The same bits either way.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 610
#define STRATA_TQ_LD16(p) ((uint32_t) __ldg((const unsigned short*) (p)))
#else
#define STRATA_TQ_LD16(p) ((uint32_t) *(p))
#endif
template<int TY>
__device__ __forceinline__ void tq_states4(const uint8_t* qs, int l, uint32_t s[4]) {
    const uint16_t* qs16 = (const uint16_t*) qs;
    if constexpr (TY == 144) {   // s_t = qs[(t + 31) % 32] << 8 | qs[t]
        const uint32_t w01 = STRATA_TQ_LD16(qs16 + 2 * l + 0), w23 = STRATA_TQ_LD16(qs16 + 2 * l + 1);
        const uint32_t prev = qs[(4 * l + 31) & 31];
        const uint32_t b0 = w01 & 0xFF, b1 = w01 >> 8, b2 = w23 & 0xFF, b3 = w23 >> 8;
        s[0] = (prev << 8) | b0;
        s[1] = (b0 << 8) | b1;
        s[2] = (b1 << 8) | b2;
        s[3] = (b2 << 8) | b3;
    } else {
        // step t's state: the 16 stream bits from bit (31 - t)*K, circular over 2K u16 words; steps 4l..4l+3 all
        // lie in the four words from (28 - 4l)*K >> 4
        constexpr int K = TqInfo<TY>::K, nwords = 2 * K;
        const int off3 = (28 - 4 * l) * K, wb = off3 >> 4, r = off3 & 15;
        const int wi1 = wb + 1 >= nwords ? wb + 1 - nwords : wb + 1;
        const int wi2 = wb + 2 >= nwords ? wb + 2 - nwords : wb + 2;
        const int wi3 = wb + 3 >= nwords ? wb + 3 - nwords : wb + 3;
        uint64_t v = (uint64_t) STRATA_TQ_LD16(qs16 + wb) | ((uint64_t) STRATA_TQ_LD16(qs16 + wi1) << 16) |
                     ((uint64_t) STRATA_TQ_LD16(qs16 + wi2) << 32);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 610
        if constexpr (K == 6) {   // r is 0 or 8 here: bits r .. r + 33 < 48, the fourth word is never read
        } else
#endif
        v |= (uint64_t) STRATA_TQ_LD16(qs16 + wi3) << 48;
#pragma unroll
        for (int i = 0; i < 4; ++i) s[i] = (uint32_t) (v >> (r + (3 - i) * K)) & 0xFFFFu;
    }
}
// lane iqs/4 of block kbx against the q8_1 blocks aligned with the block's start (4 per trellis block).  Split in
// two, like the IQ types' decode-once kernels below: tq_load decodes the lane's 16 codebook values (weight side only),
// tq_apply does the activation side.  vec_dot_tq_q8_1 is apply(load()), and the decode-once kernels call tq_load once
// per weight part and tq_apply per entry: the same integer and float operations in the same order, so the two kernel
// families are bitwise equal to EACH OTHER (native_expert_parity: "decode-once vs per-entry kernels").  Neither is
// bitwise equal to a float decode of the weights, see Numerics.
// The lane's 16 codebook values as 4 ints of packed int8: a trellis step's 4 weights are two 2-byte entries = one
// int, so the apply side is 4 dp4a instead of 16 int8->float + 16 FMAs, and a decoded lane takes 5 registers
// instead of 17.
//
// Numerics.  Decode (these kernels) reads the int8 codebook tq_lut_i8: one global scale TQ_LUT_I8_SCALE, relative
// RMS error 0.85% of the codebook std, far below the trellis quantization error.  The prompt path decodes the
// trellis otherwise: the FP16 path (dq_tq below) through the fp16 codebook tq_lut_f16, which is what ggml's to_float
// reads, and the MMQ path through the GGML fork's tile loader (the int8 codebook too on the fork's main since
// 2026-10-03; the fp16 codebook requantized to int8 per 16 weights before that and with GGML_CUDA_TQ_F32_DOT).
// Decode and prompt therefore agree within tolerance, not bitwise.  native_expert_parity bounds this path: one
// expert (int8 codebook, q8_1 activations) against ggml's float dequantizer and a float SwiGLU, relative L1 error
// sum|gpu - ref| / sum|ref| < 3e-2 (the bound every expert type's CPU and GPU paths get there); the prompt path's
// fp16 dequantizers against to_float at < 1e-6, their fp16 output equal to the fp16 of the float one.
struct TqLane { int q[4]; float d; };
// STAGED: the int8 codebook is read from shared memory (s_lut: the 2048 u16 of tq_lut_i8, copied by stage_iq_grid)
// instead of the global table.  The same 16-bit entries, so the same weights.
template<int TY, bool STAGED = false>
__device__ __forceinline__ TqLane tq_load(const void* __restrict__ vbq, int kbx, int iqs, const uint16_t* __restrict__ s_lut = nullptr) {
    const auto* b = (const typename TqInfo<TY>::block_t*) vbq + kbx;
    uint32_t s[4];
    tq_states4<TY>(b->qs, iqs / 4, s);
    TqLane r;
    const uint16_t* lut = STAGED ? s_lut : tq_lut_i8;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t x = (s[i] & TQ_STATE_MASK) * 0x9e3779b1u;
        r.q[i] = (int) ((uint32_t) lut[x >> 21] | ((uint32_t) lut[(x >> 10) & 2047u] << 16));
    }
    r.d = tq_h2f(STRATA_TQ_LD16(&b->d));
    return r;
}
__device__ __forceinline__ float tq_apply(const TqLane& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
    const int l = iqs / 4;
    const block_q8_1* b8 = bq8_1 + l / 2;
    const int* q8 = (const int*) b8->qs + 4 * (l & 1);
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) sumi = ggml_cuda_dp4a(r.q[i], q8[i], sumi);
    return r.d * (__low2float(b8->ds) * TQ_LUT_I8_SCALE) * (float) sumi;
}
template<int TY>
__device__ __forceinline__ float vec_dot_tq_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                 const int& kbx, const int& iqs) {
    return tq_apply(tq_load<TY>(vbq, kbx, iqs), bq8_1, iqs);
}

// ---- TQP (Gyro-L): 64 trellis steps of 2 weights; a lane is 8 steps (16 values).
// State of step t: the 16 stream bits from bit (63 - t)*K, LSB-first, circular over 64*K bits
// (tqp_state in the fork's ggml-common.h). The step's 2 weights are tq2t_lut_f16[2*hyb(2*s)+j],
// i.e. the FIRST pair of TQ2_T state s = tq_lut_i8[x >> 21] (tq_lut_i8 holds the pair as two int8).
// A lane's 8 states are the 16-bit fields at bit i*K of the bit stream starting at its 8th-to-last
// step, so the whole lane needs K + 2 bytes (7, 8 or 9 for K = 5, 6, 7): one byte-built 64-bit
// window plus one byte for K = 7's first state (which ends at bit 65), then eight shifts.  That is
// 7-9 loads per lane instead of 24.  The bytes are loaded (not one u64) because a block's qs array
// is only 2-byte aligned (block strides are 42/50/58 bytes).
template<int TY>
__device__ __forceinline__ void tqp_states8(const uint8_t* __restrict__ qs, int lane, uint32_t s[8]) {
    constexpr int K = TqInfo<TY>::K, NB = 8 * K, NW = K + 2 < 8 ? K + 2 : 8;
    const int pidx = (7 - lane) * K;          // the lane's first byte, always < NB
    uint64_t w = 0;                           // bits pidx*8 .. pidx*8+63
#pragma unroll
    for (int j = 0; j < NW; ++j) {
        const int idx0 = pidx + j;
        w |= (uint64_t) qs[idx0 < NB ? idx0 : idx0 - NB] << (8 * j);
    }
    const int i8 = pidx + 8;
    const uint32_t hi = K == 7 ? (uint32_t) qs[i8 < NB ? i8 : i8 - NB] : 0u;   // the 65th bit
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int sh = (7 - i) * K;                               // 0 .. 7K (<= 49)
        const uint32_t lo = (uint32_t) (w >> sh);
        const uint32_t top = sh > 48 ? (uint32_t) (hi << (64 - sh)) : 0u;
        s[i] = (lo | top) & 0xFFFFu;
    }
}
// The per-state form: 3 byte loads and a shift, no window.  Used by dq_tqp (the f16 expert uploads), which
// benches faster with it than with the window (the window's register pressure costs the copy kernels).
template<int TY>
__device__ __forceinline__ void tqp_states8_cheap(const uint8_t* __restrict__ qs, int lane, uint32_t s[8]) {
    constexpr int K = TqInfo<TY>::K, NB = 8 * K;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t off = (uint32_t) (63 - (8 * lane + i)) * (uint32_t) K;
        const uint32_t i0 = off >> 3;
        const uint32_t i1 = i0 + 1 < (uint32_t) NB ? i0 + 1 : 0;
        const uint32_t i2 = i1 + 1 < (uint32_t) NB ? i1 + 1 : 0;
        const uint32_t w = (uint32_t) qs[i0] | ((uint32_t) qs[i1] << 8) | ((uint32_t) qs[i2] << 16);
        s[i] = (w >> (off & 7)) & 0xFFFFu;
    }
}
template<int TY>
__device__ __forceinline__ TqLane tq_load_tqp(const void* __restrict__ vbq, int kbx, int iqs) {
    const auto* b = (const typename TqInfo<TY>::block_t*) vbq + kbx;
    uint32_t s[8];
    tqp_states8<TY>(b->qs, iqs / 4, s);
    TqLane r;
#pragma unroll
    for (int i = 0; i < 4; ++i) {   // two steps (2+2 weights) per int
        const uint32_t x0 = (s[2 * i] & TQ_STATE_MASK) * 0x9e3779b1u;
        const uint32_t x1 = (s[2 * i + 1] & TQ_STATE_MASK) * 0x9e3779b1u;
        r.q[i] = (int) ((uint32_t) tq_lut_i8[x0 >> 21] | ((uint32_t) tq_lut_i8[x1 >> 21] << 16));
    }
    r.d = tq_h2f(b->d);
    return r;
}
template<int TY>
__device__ __forceinline__ float vec_dot_tqp_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                  const int& kbx, const int& iqs) {
    return tq_apply(tq_load_tqp<TY>(vbq, kbx, iqs), bq8_1, iqs);
}

// ---------------------------------------------------------------- the formats
// qk = values per block, ipb = dot calls per block (qi / vdr), step = the iqs stride between calls.
template<int TY> struct Fmt;
template<> struct Fmt<16> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_xxs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<17> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_xs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<18> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq3_xxs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<20> { static constexpr int qk = 32, ipb = 2, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq4_nl_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<21> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq3_s_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<23> { static constexpr int qk = 256, ipb = 8, step = 4;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq4_xs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<22> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_s_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<29> { static constexpr int qk = 256, ipb = 8, step = 1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq1_m_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<42> { static constexpr int qk = 64, ipb = 2, step = 1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q2_0_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<12> { static constexpr int qk = 256, ipb = QI4_K / VDR_Q4_K, step = VDR_Q4_K;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q4_K_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<13> { static constexpr int qk = 256, ipb = QI5_K / VDR_Q5_K, step = VDR_Q5_K;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q5_K_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<11> { static constexpr int qk = 256, ipb = QI3_K / VDR_Q3_K, step = VDR_Q3_K;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q3_K_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<14> { static constexpr int qk = 256, ipb = QI6_K / VDR_Q6_K, step = VDR_Q6_K;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q6_K_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<7> { static constexpr int qk = 32, ipb = QI5_1 / VDR_Q5_1, step = VDR_Q5_1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q5_1_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<2> { static constexpr int qk = 32, ipb = QI4_0 / VDR_Q4_0, step = VDR_Q4_0;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q4_0_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<3> { static constexpr int qk = 32, ipb = QI4_1 / VDR_Q4_1, step = VDR_Q4_1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q4_1_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<6> { static constexpr int qk = 32, ipb = QI5_0 / VDR_Q5_0, step = VDR_Q5_0;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q5_0_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<8> { static constexpr int qk = 32, ipb = QI8_0 / VDR_Q8_0, step = VDR_Q8_0;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q8_0_q8_1(v, y, kbx, iqs); } };
// APR trellis types: 8 lanes of 4 steps per 128-value block, iqs = 4 * lane
template<> struct Fmt<144> { static constexpr int qk = 128, ipb = 8, step = 4;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_tq_q8_1<144>(v, y, kbx, iqs); } };
template<> struct Fmt<145> { static constexpr int qk = 128, ipb = 8, step = 4;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_tq_q8_1<145>(v, y, kbx, iqs); } };
template<> struct Fmt<146> { static constexpr int qk = 128, ipb = 8, step = 4;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_tq_q8_1<146>(v, y, kbx, iqs); } };
// Gyro-L pair trellis: 8 lanes of 8 steps (16 values) per 128-value block
#define STRATA_TQP_FMT(T) template<> struct Fmt<T> { static constexpr int qk = 128, ipb = 8, step = 4; \
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_tqp_q8_1<T>(v, y, kbx, iqs); } };
STRATA_TQP_FMT(147)
STRATA_TQP_FMT(148)
STRATA_TQP_FMT(149)
#undef STRATA_TQP_FMT

// The formats of each role, one list each so a type cannot be in one switch and missing from another.  Every
// entry is a kernel template for each CUDA architecture of the build, hence two lists rather than one.
#ifdef STRATA_Q6K_EXPERTS   // opt-in build (-DSTRATA_Q6K_EXPERTS=ON): one more instance per kernel, loaded at start
#define STRATA_GU_FMTS(X) X(16) X(17) X(18) X(21) X(22) X(23) X(29) X(42) X(11) X(12) X(13) X(14) X(6) X(2) X(3) X(8) X(40) X(144) X(145) X(147) X(148) X(149)
#else
#define STRATA_GU_FMTS(X) X(16) X(17) X(18) X(21) X(22) X(23) X(29) X(42) X(12) X(13) X(6) X(2) X(3) X(8) X(40) X(144) X(145) X(147) X(148) X(149)
#endif
// Q5_K / Q6_K downs: Qwen3.6-35B-A3B (FF 512, whole 256-blocks per row); Flash-Next's FF 640 never selects them
#ifdef STRATA_Q6K_EXPERTS
#define STRATA_D_FMTS(X) X(20) X(23) X(42) X(7) X(6) X(2) X(3) X(8) X(13) X(14) X(40) X(144) X(146) X(147) X(148) X(149)
#else
#define STRATA_D_FMTS(X) X(20) X(23) X(42) X(7) X(6) X(2) X(3) X(8) X(13) X(40) X(144) X(146) X(147) X(148) X(149)
#endif
#define STRATA_MMVQ_FMTS(X) X(16) X(17) X(18) X(20) X(21) X(22) X(23) X(29) X(42) X(12) X(13) X(7) X(6) X(2) X(3) X(8) X(40) X(144) X(145) X(146) X(147) X(148) X(149)
template<> struct Fmt<40> { static constexpr int qk = 64, ipb = 2, step = 4;     // NVFP4: 2 calls of 32 values
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_nvfp4_q8_1(v, y, kbx, iqs); } };

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// One row against one q8_1 activation, the whole warp: call k = (block, part) is lane-strided.
template<int TY>
__device__ __forceinline__ float row_dot(const uint8_t* row, const block_q8_1* x, int nb, int lane) {
    using F = Fmt<TY>;
    float s = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        s += F::dot(row, x + kbx * (F::qk / 32), kbx, iqs);
    }
    return warp_sum(s);
}

template<int TY>
__global__ void __launch_bounds__(128) mmvq_kernel(const uint8_t* __restrict__ w, size_t row_bytes,
                                                   const block_q8_1* __restrict__ x, float* __restrict__ y, int n_in,
                                                   int n_out, int ncols) {
    const int row = blockIdx.x * 4 + threadIdx.y;
    if (row >= n_out) return;
    const int lane = threadIdx.x;
    const int nb = n_in / Fmt<TY>::qk;
    const uint8_t* wr = w + (size_t) row * row_bytes;
    for (int c = 0; c < ncols; ++c) {
        const float s = row_dot<TY>(wr, x + (size_t) c * (n_in / 32), nb, lane);
        if (lane == 0) y[(size_t) c * n_out + row] = s;
    }
}

// ---------------------------------------------------------------- decode once, apply to every column
// The kernels above call Fmt<TY>::dot once per (call, column): each column re-reads the weight words and redoes
// the grid lookups and sign unpacking.  Here each dot is split, as native_mmvq.cu's multi-column traits are, into
// `load` (everything that depends only on the weight: the signed grid words, the integer scales, the fp16 block
// scale as a float) and `apply` (the activation loads, the dp4a chain in the same order, the same integer scale
// step and the same float expression).  `apply(load(...))` does the dot's integer and float operations in the same
// order on the same values, so a column of the kernels below is BITWISE equal to the same column of mmvq_kernel /
// native_gu_kernel / native_down_kernel (iq_multi_parity checks it; STRATA_OLD_IQ_MMVQ=1 keeps the old kernels).
template<int TY> struct Split;
// Formats with a Split below take the decode-once kernels; the others (Q4_K, Q5_K, Q5_1, Q8_0: UD-Q4_K_XL) the
// per-entry ones, which call Fmt<TY>::dot per column exactly as before #242 (the launchers test kSplit at compile
// time, so the multi kernels are never instantiated for a type without a Split).
template<int TY> inline constexpr bool kSplit = false;
// The K-quant / Q8_0 / Q5_1 formats (UD-Q4_K_XL) have Split traits (SplitK below), but their decode-once multi kernels
// run only with STRATA_KQ_SPLIT=1 until the A/B on the measured cards decides (bitwise the old kernels either way).
template<int TY> inline constexpr bool kq_type = (TY == 12 || TY == 13 || TY == 8);   // not Q5_1: its Split apply is not bitwise the old dot (FMA contraction), see notes-151-amd
inline bool g_kq_split = [] { const char* v = std::getenv("STRATA_KQ_SPLIT"); return v != nullptr && v[0] == '1'; }();
template<int TY> inline bool split_off() { return kq_type<TY> && !g_kq_split; }
template<int TY> inline constexpr bool kTqType = (TY == 144 || TY == 145 || TY == 146);
// the trellis types stage their int8 codebook (4 KB) too, but only where it pays: tq_stage_on() below (compute capability 6.0)
template<int TY> inline constexpr bool kStageIqGrid = (TY == 16 || TY == 17 || TY == 18 || TY == 21 || TY == 22 || TY == 29 || kTqType<TY>);

template<> inline constexpr bool kSplit<16> = true;
template<> struct Split<16> {   // IQ2_XXS
    struct W { int g[8]; int ls; float dw; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ s_grid = nullptr) {
        const block_iq2_xxs* bq2 = (const block_iq2_xxs*) vbq + kbx;
        const int q2 = get_int_b2(bq2->qs, iqs);
        const uint8_t* aux8 = (const uint8_t*) &q2;
        const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
        const uint2* grid_lut;
        if constexpr (STAGE_GRID) grid_lut = (const uint2*) s_grid;
        else grid_lut = (const uint2*) iq2xxs_grid;
        W r;
#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            const uint2 grid_pos = grid_lut[aux8[k0 / 2]];
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            r.g[k0 + 0] = __vsub4(grid_pos.x ^ signs0, signs0);
            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            r.g[k0 + 1] = __vsub4(grid_pos.y ^ signs1, signs1);
        }
        r.ls = aux32 >> 27 | 1;
        r.dw = __half2float(bq2->d);
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi);
        sumi = sumi * r.ls / 8;
        const float d = r.dw * __low2float(bq8_1[iqs / 2].ds);
        return d * sumi;
    }
    // `apply` with the q8_1 block already in registers: u = its 8 words, d8 = __low2float(ds).  Same operations in
    // the same order (the EXPDEC kernels, see expdec below).
    __device__ static float apply_u(const W& r, const int (&u)[8], float d8) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = ggml_cuda_dp4a(r.g[j], u[j], sumi);
        sumi = sumi * r.ls / 8;
        const float d = r.dw * d8;
        return d * sumi;
    }
};
// IQ2_XS and IQ2_S share the apply: two half sums, two 4-bit scales
struct SplitLs2 {
    struct W { int g[8]; int ls0, ls1; float dw; };
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi0 = 0, sumi1 = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) sumi0 = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi0);
#pragma unroll
        for (int j = 4; j < 8; ++j) sumi1 = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi1);
        const int sumi = (sumi0 * r.ls0 + sumi1 * r.ls1 + (sumi0 + sumi1) / 2) / 4;
        const float d = r.dw * __low2float(bq8_1[iqs / 2].ds);
        return d * sumi;
    }
    __device__ static float apply_u(const W& r, const int (&u)[8], float d8) {
        int sumi0 = 0, sumi1 = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) sumi0 = ggml_cuda_dp4a(r.g[j], u[j], sumi0);
#pragma unroll
        for (int j = 4; j < 8; ++j) sumi1 = ggml_cuda_dp4a(r.g[j], u[j], sumi1);
        const int sumi = (sumi0 * r.ls0 + sumi1 * r.ls1 + (sumi0 + sumi1) / 2) / 4;
        const float d = r.dw * d8;
        return d * sumi;
    }
};
template<> inline constexpr bool kSplit<17> = true;
template<> struct Split<17> : SplitLs2 {   // IQ2_XS
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ s_grid = nullptr) {
        const block_iq2_xs* bq2 = (const block_iq2_xs*) vbq + kbx;
        const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
        const uint16_t* q2 = (const uint16_t*) &q2_packed;
        const uint2* grid_lut;
        if constexpr (STAGE_GRID) grid_lut = (const uint2*) s_grid;
        else grid_lut = (const uint2*) iq2xs_grid;
        W r;
        r.ls0 = bq2->scales[iqs / 2] & 0x0F;
        r.ls1 = bq2->scales[iqs / 2] >> 4;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const uint2 grid_pos = grid_lut[q2[l0 / 2] & 0x1FF];
            const uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            r.g[l0 + 0] = __vsub4(grid_pos.x ^ signs0, signs0);
            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            r.g[l0 + 1] = __vsub4(grid_pos.y ^ signs1, signs1);
        }
        r.dw = __half2float(bq2->d);
        return r;
    }
};
template<> inline constexpr bool kSplit<22> = true;
template<> struct Split<22> : SplitLs2 {   // IQ2_S
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ s_grid = nullptr) {
        const block_iq2_s* bq2 = (const block_iq2_s*) vbq + kbx;
        const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq2->qh[iqs / 2];
        const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
        const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
        const uint64_t* grid_lut;
        if constexpr (STAGE_GRID) grid_lut = (const uint64_t*) s_grid;
        else grid_lut = iq2s_grid;
        W r;
        r.ls0 = bq2->scales[iqs / 2] & 0x0F;
        r.ls1 = bq2->scales[iqs / 2] >> 4;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int* grid_pos = (const int*) (grid_lut + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
            const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
            r.g[l0 + 0] = __vsub4(grid_pos[0] ^ signs0, signs0);
            r.g[l0 + 1] = __vsub4(grid_pos[1] ^ signs1, signs1);
        }
        r.dw = __half2float(bq2->d);
        return r;
    }
};
template<> inline constexpr bool kSplit<18> = true;
template<> struct Split<18> {   // IQ3_XXS
    struct W { int g[8]; int ls; float dw; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ s_grid = nullptr) {
        const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
        const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
        const uint8_t* q3 = (const uint8_t*) &q3_packed;
        const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
        const uint32_t* grid = STAGE_GRID ? s_grid : iq3xxs_grid;
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int2 grid_pos = make_int2(grid[q3[l0 + 0]], grid[q3[l0 + 1]]);
            if constexpr (STAGE_GRID) {
                // The staged tables hold one mask per 7-bit sign field; bit 7 of the field is the next group's
                // first sign, which unpack_ksigns cancels, so masking it off is the same value.
                const uint32_t v = (aux32 >> (7 * l0 / 2)) & 0x7f;
                const uint32_t ml = s_grid[256 + v], mh = s_grid[384 + v];
                r.g[l0 + 0] = (int) ((grid_pos.x ^ ml) + (ml & 0x01010101u));
                r.g[l0 + 1] = (int) ((grid_pos.y ^ mh) + (mh & 0x01010101u));
            } else {
                const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
                const int signs0 = __vcmpne4(signs & 0x08040201, 0);
                r.g[l0 + 0] = __vsub4(grid_pos.x ^ signs0, signs0);
                const int signs1 = __vcmpne4(signs & 0x80402010, 0);
                r.g[l0 + 1] = __vsub4(grid_pos.y ^ signs1, signs1);
            }
        }
        r.ls = aux32 >> 28;
        r.dw = __half2float(bq3->d);
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi);
        sumi = (r.ls * sumi + sumi / 2) / 2;
        const float d = r.dw * __low2float(bq8_1[iqs / 2].ds);
        return d * sumi;
    }
    __device__ static float apply_u(const W& r, const int (&u)[8], float d8) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = ggml_cuda_dp4a(r.g[j], u[j], sumi);
        sumi = (r.ls * sumi + sumi / 2) / 2;
        const float d = r.dw * d8;
        return d * sumi;
    }
};
template<> inline constexpr bool kSplit<21> = true;
template<> struct Split<21> {   // IQ3_S
    struct W { int g[8]; int ls; float dw; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ s_grid = nullptr) {
        const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
        const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq3->qh[iqs / 2];
        const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
        const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
        const uint32_t* grid = STAGE_GRID ? s_grid : iq3s_grid;
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int2 grid_pos = make_int2(grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                                            grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
            if constexpr (STAGE_GRID) {
                // The staged tables hold the byte masks of a sign byte's bits 0-3 and 4-7; every iq3s_grid byte is
                // odd, so (g ^ mask) + (mask & 0x01010101) negates the masked bytes with no carry into the next.
                const uint32_t ml = s_grid[512 + signs_packed_8[l0 / 2]], mh = s_grid[768 + signs_packed_8[l0 / 2]];
                r.g[l0 + 0] = (int) (((uint32_t) grid_pos.x ^ ml) + (ml & 0x01010101u));
                r.g[l0 + 1] = (int) (((uint32_t) grid_pos.y ^ mh) + (mh & 0x01010101u));
            } else {
                const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
                const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
                r.g[l0 + 0] = __vsub4(grid_pos.x ^ signs0, signs0);
                r.g[l0 + 1] = __vsub4(grid_pos.y ^ signs1, signs1);
            }
        }
        r.ls = 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
        r.dw = __half2float(bq3->d);
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi);
        sumi *= r.ls;
        const float d = r.dw * __low2float(bq8_1[iqs / 2].ds);
        return d * sumi;
    }
    __device__ static float apply_u(const W& r, const int (&u)[8], float d8) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = ggml_cuda_dp4a(r.g[j], u[j], sumi);
        sumi *= r.ls;
        const float d = r.dw * d8;
        return d * sumi;
    }
};
template<> inline constexpr bool kSplit<29> = true;
template<> struct Split<29> {   // IQ1_M
    struct W { int g[8]; float delta[4]; int sc0, sc1; float dw; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ s_grid = nullptr) {
        const block_iq1_m* bq1 = (const block_iq1_m*) vbq + kbx;
        const int qs_packed = get_int_b4(bq1->qs, iqs);
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const uint32_t* grid_lut;
        if constexpr (STAGE_GRID) grid_lut = s_grid;
        else grid_lut = iq1s_grid_gpu;
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int qhl = bq1->qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
            const int grid = grid_lut[qs[l0 / 2] | ((qhl & 0x07) << 8)];
            r.g[l0 + 0] = (grid >> 0) & 0x0F0F0F0F;
            r.g[l0 + 1] = (grid >> 4) & 0x0F0F0F0F;
            r.delta[l0 / 2] = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f * IQ1M_DELTA / 0x08);
        }
        const uint16_t* sc = (const uint16_t*) bq1->scales;
        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
        r.dw = __half2float(scale.f16);
        const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
        r.sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
        r.sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi[2] = {0, 0};
        float sumf[2] = {0.0f, 0.0f};
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
            const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
            sumi[l0 / 4] = ggml_cuda_dp4a(r.g[l0 + 0], u0, sumi[l0 / 4]);
            sumi[l0 / 4] = ggml_cuda_dp4a(r.g[l0 + 1], u1, sumi[l0 / 4]);
            int sumy = 0;
            sumy = ggml_cuda_dp4a(u0, 0x01010101, sumy);
            sumy = ggml_cuda_dp4a(u1, 0x01010101, sumy);
            sumf[l0 / 4] += r.delta[l0 / 2] * sumy;
        }
        const float d = r.dw * __low2float(bq8_1[iqs].ds);
        return d * ((sumi[0] + sumf[0]) * r.sc0 + (sumi[1] + sumf[1]) * r.sc1);
    }
};
template<> inline constexpr bool kSplit<20> = true;
template<> struct Split<20> {   // IQ4_NL
    struct W { int2 v[2]; float dw; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
        const block_iq4_nl* bq4 = (const block_iq4_nl*) vbq + kbx;
        W r;
#pragma unroll
        for (int l = 0; l < 2; ++l) r.v[l] = get_int_from_table_16(get_int_b2(bq4->qs, iqs + l), kvalues_iq4nl);
        r.dw = __half2float(bq4->d);
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        const int* q8 = (const int*) bq8_1->qs + iqs;
        int sumi = 0;
#pragma unroll
        for (int l = 0; l < 2; ++l) {
            sumi = ggml_cuda_dp4a(r.v[l].x, q8[l + 0], sumi);
            sumi = ggml_cuda_dp4a(r.v[l].y, q8[l + 4], sumi);
        }
        const float d = r.dw * __low2float(bq8_1->ds);
        return d * sumi;
    }
    // `apply` with the block's words in registers: p0 = words iqs, iqs + 1; p1 = words iqs + 4, iqs + 5; d8 = __low2float(ds)
    __device__ static float apply_u(const W& r, int2 p0, int2 p1, float d8) {
        int sumi = 0;
        sumi = ggml_cuda_dp4a(r.v[0].x, p0.x, sumi);
        sumi = ggml_cuda_dp4a(r.v[0].y, p1.x, sumi);
        sumi = ggml_cuda_dp4a(r.v[1].x, p0.y, sumi);
        sumi = ggml_cuda_dp4a(r.v[1].y, p1.y, sumi);
        const float d = r.dw * d8;
        return d * sumi;
    }
};
template<> inline constexpr bool kSplit<23> = true;
template<> struct Split<23> {   // IQ4_XS
    struct W { int2 v[4]; int ls; float dw; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
        const block_iq4_xs* bq4 = (const block_iq4_xs*) vbq + kbx;
        W r;
#pragma unroll
        for (int j = 0; j < 4; ++j) r.v[j] = get_int_from_table_16(get_int_b4(bq4->qs, iqs + j), kvalues_iq4nl);
        r.ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
        r.dw = __half2float(bq4->d);
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int u0 = get_int_b4(bq8_1[iqs / 4].qs, j + 0);
            const int u1 = get_int_b4(bq8_1[iqs / 4].qs, j + 4);
            sumi = ggml_cuda_dp4a(r.v[j].x, u0, sumi);
            sumi = ggml_cuda_dp4a(r.v[j].y, u1, sumi);
        }
        sumi *= r.ls - 32;
        const float d = r.dw * __low2float(bq8_1[iqs / 4].ds);
        return d * sumi;
    }
};
template<> inline constexpr bool kSplit<42> = true;
template<> struct Split<42> {   // Q2_0
    struct W { int qx[4], qy[4]; float d2; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
        const block_q2_0* bq2_0 = (const block_q2_0*) vbq + kbx;
        W r;
        r.d2 = bq2_0->d;
        const int16_t* qs = (const int16_t*) bq2_0->qs + iqs * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = qs[j];
            const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
            const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
            r.qx[j] = __byte_perm(qe, qo, 0x5140);
            r.qy[j] = __byte_perm(qe, qo, 0x7362);
        }
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        const block_q8_1* bq8_1_chunk = bq8_1 + iqs;
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int u = get_int_b4(bq8_1_chunk->qs, j * 2 + 0);
            const int v = get_int_b4(bq8_1_chunk->qs, j * 2 + 1);
            sumi = ggml_cuda_dp4a(u, r.qx[j], sumi);
            sumi = ggml_cuda_dp4a(v, r.qy[j], sumi);
        }
        const float d8 = __low2float(bq8_1_chunk->ds);
        return r.d2 * d8 * sumi;
    }
    // `apply` with the q8_1 block (2 kbx + iqs) in registers: u = its 8 words, d8 = __low2float(ds)
    __device__ static float apply_u(const W& r, const int (&u)[8], float d8) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumi = ggml_cuda_dp4a(u[j * 2 + 0], r.qx[j], sumi);
            sumi = ggml_cuda_dp4a(u[j * 2 + 1], r.qy[j], sumi);
        }
        return r.d2 * d8 * sumi;
    }
};
template<int TY> struct S27;
template<> struct S27<12> {   // Q4_K
    struct W { int v[2]; uint16_t aux[2]; half2 dm; };
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_q4_K* b = (const block_q4_K*) vbq + kbx;
        W r;
        const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
        const int* q4 = (const int*) (b->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
        r.v[0] = q4[0];
        r.v[1] = q4[4];
        k_scale_min(b->scales, bq8_offset, r.aux);
        r.dm = b->dm;
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
        int u[2 * QR4_K];
        float d8[QR4_K];
#pragma unroll
        for (int i = 0; i < QR4_K; ++i) {
            const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
            d8[i] = __low2float(bq8i->ds);
            const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
            u[2 * i + 0] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const uint8_t* sc = (const uint8_t*) r.aux;
        const uint8_t* m = sc + 2;
        return vec_dot_q4_K_q8_1_impl_vmmq(r.v, u, sc, m, r.dm, d8);
    }
};
template<> struct S27<13> {   // Q5_K
    struct W { int vl[2]; int vh[2]; uint16_t aux[2]; half2 dm; };
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_q5_K* b = (const block_q5_K*) vbq + kbx;
        W r;
        const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
        const int* ql = (const int*) (b->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
        const int* qh = (const int*) (b->qh + 4 * ((iqs / 2) % 4));
        r.vl[0] = ql[0];
        r.vl[1] = ql[4];
        r.vh[0] = qh[0] >> bq8_offset;
        r.vh[1] = qh[4] >> bq8_offset;
        k_scale_min(b->scales, bq8_offset, r.aux);
        r.dm = b->dm;
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
        int u[2 * QR5_K];
        float d8[QR5_K];
#pragma unroll
        for (int i = 0; i < QR5_K; ++i) {
            const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
            d8[i] = __low2float(bq8i->ds);
            const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
            u[2 * i + 0] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const uint8_t* sc = (const uint8_t*) r.aux;
        const uint8_t* m = sc + 2;
        return vec_dot_q5_K_q8_1_impl_vmmq(r.vl, r.vh, u, sc, m, r.dm, d8);
    }
};
template<> struct S27<7> {   // Q5_1
    struct W { int vl[VDR_Q5_1]; int qh; half2 dm; };
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_q5_1* b = (const block_q5_1*) vbq + kbx;
        W r;
#pragma unroll
        for (int i = 0; i < VDR_Q5_1; ++i) r.vl[i] = get_int_b4(b->qs, iqs + i);
        r.qh = get_int_b4(b->qh, 0);
        r.dm = b->dm;
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0, sumu = 0;
#pragma unroll
        for (int i = 0; i < VDR_Q5_1; ++i) {
            const int vl = r.vl[i];
            const int vh = r.qh >> (4 * (iqs + i));
            const int u0 = get_int_b4(bq8_1->qs, iqs + i), u1 = get_int_b4(bq8_1->qs, iqs + i + QI5_1);
            int vi0 = (vl >> 0) & 0x0F0F0F0F;
            vi0 |= (vh << 4) & 0x00000010;
            vi0 |= (vh << 11) & 0x00001000;
            vi0 |= (vh << 18) & 0x00100000;
            vi0 |= (vh << 25) & 0x10000000;
            sumi = ggml_cuda_dp4a(vi0, u0, sumi);
            int vi1 = (vl >> 4) & 0x0F0F0F0F;
            vi1 |= (vh >> 12) & 0x00000010;
            vi1 |= (vh >> 5) & 0x00001000;
            vi1 |= (vh << 2) & 0x00100000;
            vi1 |= (vh << 9) & 0x10000000;
            sumi = ggml_cuda_dp4a(vi1, u1, sumi);
            sumu = ggml_cuda_dp4a(0x01010101, u1, ggml_cuda_dp4a(0x01010101, u0, sumu));
        }
        const float2 dm5 = __half22float2(r.dm);
        const float d8 = __low2float(bq8_1->ds);
        return sumi * (dm5.x * d8) + sumu * (dm5.y * d8);
    }
};
template<> struct S27<8> {   // Q8_0
    struct W { int q[VDR_Q8_0]; float d; };
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_q8_0* b = (const block_q8_0*) vbq + kbx;
        W r;
#pragma unroll
        for (int i = 0; i < VDR_Q8_0; ++i) r.q[i] = get_int_b2(b->qs, iqs + i);
        r.d = __half2float(b->d);
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < VDR_Q8_0; ++i) sumi = ggml_cuda_dp4a(r.q[i], get_int_b4(bq8_1->qs, iqs + i), sumi);
        const float d8_1 = __low2float(bq8_1->ds);
        return r.d * d8_1 * ((float) sumi);
    }
};
// K-quants and Q5_1 / Q8_0 (UD-Q4_K_XL): the S27 load / apply (below the same file used them first for the V2K
// grouped kernels) as Split traits, so the generic multi-column kernels decode a weight once per call for these
// formats too.  Bitwise the old per-column Fmt<TY>::dot (the S27 apply is that dot's own expression; iq_multi_parity).
// STRATA_OLD_IQ_MMVQ=1 keeps the old kernels; STRATA_KQ_SPLIT=0 keeps them for these four formats only.
template<int TY> struct SplitK : S27<TY> {
    template<bool STAGE_GRID = false>
    __device__ static typename S27<TY>::W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
        return S27<TY>::load(vbq, kbx, iqs);
    }
};
template<> inline constexpr bool kSplit<12> = true;
template<> inline constexpr bool kSplit<13> = true;
template<> inline constexpr bool kSplit<8> = true;
template<> struct Split<12> : SplitK<12> {};
template<> struct Split<13> : SplitK<13> {};
template<> struct Split<8> : SplitK<8> {};
template<> inline constexpr bool kSplit<40> = true;
template<> struct Split<40> {   // NVFP4: vec_dot_nvfp4_q8_1's two 16-value sub-blocks, weights decoded once per call
    struct W { int2 v0[2], v1[2]; float sc[2]; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
        const block_nvfp4* bq4 = (const block_nvfp4*) vbq + kbx;
        W r;
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int iqs0 = iqs + 2 * i, iqs1 = iqs0 + 1, is = iqs0 >> 1;
            r.v0[i] = get_int_from_table_16(get_int_b4(bq4->qs, iqs0), kvalues_fp4);
            r.v1[i] = get_int_from_table_16(get_int_b4(bq4->qs, iqs1), kvalues_fp4);
            r.sc[i] = ue4m3_to_f32_half(bq4->d[is]);
        }
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        float sum = 0.0f;                              // the same operations, in the same order, as the dot
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int is = (iqs + 2 * i) >> 1;
            const block_q8_1* bq8 = bq8_1 + (is >> 1);
            const int i8 = (is & 1) << 2;
            int sumi = ggml_cuda_dp4a(r.v0[i].x, get_int_b4(bq8->qs, i8 + 0), 0);
            sumi = ggml_cuda_dp4a(r.v0[i].y, get_int_b4(bq8->qs, i8 + 2), sumi);
            sumi = ggml_cuda_dp4a(r.v1[i].x, get_int_b4(bq8->qs, i8 + 1), sumi);
            sumi = ggml_cuda_dp4a(r.v1[i].y, get_int_b4(bq8->qs, i8 + 3), sumi);
            sum += r.sc[i] * __low2float(bq8->ds) * (float) sumi;
        }
        return sum;
    }
};
// APR trellis types: the lane's 16 decoded codebook values (tq_load / tq_apply above)
template<> inline constexpr bool kSplit<144> = true;
template<> inline constexpr bool kSplit<145> = true;
template<> inline constexpr bool kSplit<146> = true;
template<int TY> struct SplitTq {
    using W = TqLane;
    template<bool STAGE_GRID = false>   // STAGE_GRID: the codebook in shared memory (s_grid, see stage_iq_grid)
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ s_grid = nullptr) {
        return tq_load<TY, STAGE_GRID>(vbq, kbx, iqs, (const uint16_t*) s_grid);
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) { return tq_apply(r, bq8_1, iqs); }
};
template<> struct Split<144> : SplitTq<144> {};
template<> struct Split<145> : SplitTq<145> {};
template<> struct Split<146> : SplitTq<146> {};
// Gyro-L TQP (147-149): the pair layout needs tq_load_tqp, not tq_load, so its split is its own trait.
// Until this existed the launchers took the per-entry mmvq/native kernels for TQP, and row_dot_multi's
// callers re-decoded the lane once per activation.
template<int TY> struct SplitTqp {
    using W = TqLane;
    template<bool STAGE_GRID = false>   // (no grid table to stage: the codebook is read where it is)
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
        return tq_load_tqp<TY>(vbq, kbx, iqs);
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) { return tq_apply(r, bq8_1, iqs); }
};
template<> inline constexpr bool kSplit<147> = true;
template<> inline constexpr bool kSplit<148> = true;
template<> inline constexpr bool kSplit<149> = true;
template<> struct Split<147> : SplitTqp<147> {};
template<> struct Split<148> : SplitTqp<148> {};
template<> struct Split<149> : SplitTqp<149> {};

// One row against the n <= NC activations x + off[0..n) (n >= 1, warp-uniform; offsets in q8_1 blocks, 32-bit to
// spare registers), the whole warp.  Per activation this is row_dot: the same calls k, lane-strided the same way,
// summed in the same order, then the same warp_sum.  Only the weight side moves out of the per-activation loop.
template<int TY, int NC, bool EXACT_N = false, bool STAGE_GRID = false>
__device__ __forceinline__ void row_dot_multi(const uint8_t* row, const block_q8_1* x, const int (&off)[NC], int n,
                                              int nb, int lane, float (&s)[NC], const uint32_t* __restrict__ s_grid = nullptr) {
    using F = Fmt<TY>;
    using S = Split<TY>;
#pragma unroll
    for (int c = 0; c < NC; ++c) s[c] = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::template load<STAGE_GRID>(row, kbx, iqs, s_grid);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
#pragma unroll
    for (int c = 0; c < NC; ++c)
        if (EXACT_N || c < n) s[c] = warp_sum(s[c]);
}

// 8-lane sub-warp row dot for K = nb * ipb == 40 (TD == 20, IQ4_NL with n_ff == 640):
// 4 rows per warp (32 rows per 256-thread block), 100% active lanes, bitwise identical addition tree to row_dot_multi.
template<int TY, int NC, bool EXACT_N = false, bool STAGE_GRID = false>
__device__ __forceinline__ void row_dot_40_sub8(const uint8_t* row, const block_q8_1* x, const int (&off)[NC], int n,
                                                int t, float (&s)[NC], const uint32_t* __restrict__ s_grid = nullptr) {
    using F = Fmt<TY>;
    using S = Split<TY>;
#pragma unroll
    for (int c = 0; c < NC; ++c) s[c] = 0.0f;
    for (int k = t; k < 40; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::template load<STAGE_GRID>(row, kbx, iqs, s_grid);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
    {
        const int k = t + 16, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::template load<STAGE_GRID>(row, kbx, iqs, s_grid);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            if (EXACT_N || c < n) {
                float s16 = 0.0f;
                s16 += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
                s[c] = __fadd_rn(s[c], s16);
            }
        }
    }
    float s8[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) s8[c] = 0.0f;
    {
        const int k = t + 8, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::template load<STAGE_GRID>(row, kbx, iqs, s_grid);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s8[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
    {
        const int k = t + 24, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::template load<STAGE_GRID>(row, kbx, iqs, s_grid);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            if (EXACT_N || c < n) {
                float s24 = 0.0f;
                s24 += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
                s[c] = __fadd_rn(s[c], __fadd_rn(s8[c], s24));
            }
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        if (EXACT_N || c < n) {
#pragma unroll
            for (int o = 4; o > 0; o >>= 1) s[c] += __shfl_xor_sync(0xffffffffu, s[c], o);
        }
    }
}

// 8-lane sub-warp row dot for K = nb * ipb == 20 (TD == 40, NVFP4 with n_ff == 640): 4 rows per warp (32 rows per
// 256-thread block).  Bitwise row_dot_multi's tree: warp_sum adds lane i + 16 (terms k and k + 16; lanes 20-31 hold
// +0) then i + 8, then 4, 2, 1 - so lane t here takes k = t (+ k = t + 16 for t < 4, else + 0) and k = t + 8 (+ 0).
template<int TY, int NC, bool EXACT_N = false>
__device__ __forceinline__ void row_dot_20_sub8(const uint8_t* row, const block_q8_1* x, const int (&off)[NC], int n,
                                                int t, float (&s)[NC]) {
    using F = Fmt<TY>;
    using S = Split<TY>;
#pragma unroll
    for (int c = 0; c < NC; ++c) s[c] = 0.0f;
    {
        const int k = t, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::load(row, kbx, iqs);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
    float s16[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) s16[c] = 0.0f;
    if (t < 4) {
        const int k = t + 16, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::load(row, kbx, iqs);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s16[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
    float s8[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) s8[c] = 0.0f;
    {
        const int k = t + 8, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::load(row, kbx, iqs);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s8[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        if (EXACT_N || c < n) {
            s[c] = __fadd_rn(__fadd_rn(s[c], s16[c]), __fadd_rn(s8[c], 0.0f));
#pragma unroll
            for (int o = 4; o > 0; o >>= 1) s[c] += __shfl_xor_sync(0xffffffffu, s[c], o);
        }
    }
}

// 16-lane sub-warp row dot for K = nb * ipb == 20 (TD == 42, Q2_0, and TD == 40, NVFP4, with n_ff == 640):
// 2 rows per warp (16 rows per 256-thread block), bitwise identical addition tree to row_dot_multi.
template<int TY, int NC, bool EXACT_N = false>
__device__ __forceinline__ void row_dot_20_sub16(const uint8_t* row, const block_q8_1* x, const int (&off)[NC], int n,
                                                 int t, float (&s)[NC]) {
    using F = Fmt<TY>;
    using S = Split<TY>;
#pragma unroll
    for (int c = 0; c < NC; ++c) s[c] = 0.0f;
    {
        const int k = t, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::load(row, kbx, iqs);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
    float s16[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) s16[c] = 0.0f;
    if (t < 4) {
        const int k = t + 16, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::load(row, kbx, iqs);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s16[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        if (EXACT_N || c < n) {
            s[c] = __fadd_rn(s[c], s16[c]);
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) s[c] += __shfl_xor_sync(0xffffffffu, s[c], o);
        }
    }
}

// 16-lane sub-warp row dot for K = nb * ipb == 80 (all kSplit gate/up formats with n_embd == 2560):
// 2 rows per warp (16 rows per 256-thread block), 100% active lanes, bitwise identical addition tree to row_dot_multi.
template<int TY, int NC, bool EXACT_N = false, bool STAGE_GRID = false>
__device__ __forceinline__ void row_dot_80_sub16(const uint8_t* row, const block_q8_1* x, const int (&off)[NC], int n,
                                                 int t, float (&s)[NC], const uint32_t* __restrict__ s_grid = nullptr) {
    using F = Fmt<TY>;
    using S = Split<TY>;
#pragma unroll
    for (int c = 0; c < NC; ++c) s[c] = 0.0f;
    for (int k = t; k < 80; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::template load<STAGE_GRID>(row, kbx, iqs, s_grid);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
    float s16[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) s16[c] = 0.0f;
    for (int k = t + 16; k < 64; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::template load<STAGE_GRID>(row, kbx, iqs, s_grid);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s16[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        if (EXACT_N || c < n) {
            s[c] = __fadd_rn(s[c], s16[c]);
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) s[c] += __shfl_xor_sync(0xffffffffu, s[c], o);
        }
    }
}

// 8-lane sub-warp row dot for K = nb * ipb == 160 (the APR trellis gate/up types at n_embd 2560): 4 rows per warp (32 per
// 256-thread block).  Bitwise row_dot_multi's tree: its lane L sums the calls k = L, L + 32, ... L + 128 in that order, then
// warp_sum adds lane + 16, then + 8, then 4, 2, 1.  Here lane t of 8 keeps the four sums of the lanes t, t + 8, t + 16 and
// t + 24 (v = 0..3) and adds them (v0 + v2) + (v1 + v3), which is what the first two steps give; 3 steps follow.  A trellis
// block is 8 lanes, so lane t always takes lane t of block v + 4 j: its bit offsets are the same for all 20 calls.
template<int TY, int NC, bool EXACT_N = false, bool STAGE_GRID = false>
__device__ __forceinline__ void row_dot_160_sub8(const uint8_t* row, const block_q8_1* x, const int (&off)[NC], int n,
                                                 int t, float (&s)[NC], const uint32_t* __restrict__ s_grid = nullptr) {
    using F = Fmt<TY>;
    using S = Split<TY>;
    static_assert(F::ipb == 8 && F::step == 4, "one trellis block is 8 lanes of 4 steps");
    float p[4][NC];
#pragma unroll
    for (int v = 0; v < 4; ++v)
#pragma unroll
        for (int c = 0; c < NC; ++c) p[v][c] = 0.0f;
#pragma unroll
    for (int j = 0; j < 5; ++j) {
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const int kbx = v + 4 * j, iqs = F::step * t;
            const typename S::W w = S::template load<STAGE_GRID>(row, kbx, iqs, s_grid);
#pragma unroll
            for (int c = 0; c < NC; ++c)
                if (EXACT_N || c < n) p[v][c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        if (EXACT_N || c < n) {
            s[c] = __fadd_rn(__fadd_rn(p[0][c], p[2][c]), __fadd_rn(p[1][c], p[3][c]));
#pragma unroll
            for (int o = 4; o > 0; o >>= 1) s[c] += __shfl_xor_sync(0xffffffffu, s[c], o);
        }
    }
}

// 147-verify: row_dot_80_sub16 for two rows (a gate row and its up row) at once, both rows' weight loads of a step issued
// before either is applied; each row's sums are row_dot_80_sub16's, in its order (the same bits).
template<int TY, int NC, bool EXACT_N = false, bool STAGE_GRID = false>
__device__ __forceinline__ void row_dot2_80_sub16(const uint8_t* ra, const uint8_t* rb, const block_q8_1* x, const int (&off)[NC],
                                                  int n, int t, float (&sa)[NC], float (&sb)[NC],
                                                  const uint32_t* __restrict__ s_grid = nullptr) {
    using F = Fmt<TY>;
    using S = Split<TY>;
#pragma unroll
    for (int c = 0; c < NC; ++c) { sa[c] = 0.0f; sb[c] = 0.0f; }
    for (int k = t; k < 80; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W wa = S::template load<STAGE_GRID>(ra, kbx, iqs, s_grid);
        const typename S::W wb = S::template load<STAGE_GRID>(rb, kbx, iqs, s_grid);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) {
                sa[c] += S::apply(wa, x + off[c] + kbx * (F::qk / 32), iqs);
                sb[c] += S::apply(wb, x + off[c] + kbx * (F::qk / 32), iqs);
            }
    }
    float a16[NC], b16[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) { a16[c] = 0.0f; b16[c] = 0.0f; }
    for (int k = t + 16; k < 64; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W wa = S::template load<STAGE_GRID>(ra, kbx, iqs, s_grid);
        const typename S::W wb = S::template load<STAGE_GRID>(rb, kbx, iqs, s_grid);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) {
                a16[c] += S::apply(wa, x + off[c] + kbx * (F::qk / 32), iqs);
                b16[c] += S::apply(wb, x + off[c] + kbx * (F::qk / 32), iqs);
            }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        if (EXACT_N || c < n) {
            sa[c] = __fadd_rn(sa[c], a16[c]);
            sb[c] = __fadd_rn(sb[c], b16[c]);
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) {
                sa[c] += __shfl_xor_sync(0xffffffffu, sa[c], o);
                sb[c] += __shfl_xor_sync(0xffffffffu, sb[c], o);
            }
        }
    }
}

// The sign-mask tables that follow the codebook words in a staged IQ3_XXS / IQ3_S table (see stage_iq_grid): the byte
// masks of every 7-bit sign field (IQ3_XXS, 2 x 128 words at +256) and of a sign byte's bits 0-3 and 4-7 (IQ3_S, 2 x 256
// words at +512).  Split<18>::load<true> and Split<21>::load<true> read them, so every kernel that stages these two
// codebooks (the default kernels through stage_iq_grid, the STRATA_EXPDEC window kernels through EdGrid) fills them too.
__device__ __forceinline__ void stage_iq3xxs_masks(uint32_t* s_buf, int tid, int nthreads) {
    for (int u = tid; u < 128; u += nthreads) {
        const uint32_t s = (uint32_t) u | ((__popc((uint32_t) u) & 1u) << 7);   // 8th sign = parity of the 7
        uint32_t ml = 0, mh = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            ml |= ((s >> i) & 1u) * (0xffu << (8 * i));
            mh |= ((s >> (4 + i)) & 1u) * (0xffu << (8 * i));
        }
        s_buf[256 + u] = ml;
        s_buf[384 + u] = mh;
    }
}
__device__ __forceinline__ void stage_iq3s_masks(uint32_t* s_buf, int tid, int nthreads) {
    for (int u = tid; u < 256; u += nthreads) {
        uint32_t ml = 0, mh = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            ml |= (((uint32_t) u >> i) & 1u) * (0xffu << (8 * i));
            mh |= (((uint32_t) u >> (4 + i)) & 1u) * (0xffu << (8 * i));
        }
        s_buf[512 + u] = ml;
        s_buf[768 + u] = mh;
    }
}

template<int TY, bool STAGE_GRID = kStageIqGrid<TY>>
__device__ __forceinline__ const uint32_t* stage_iq_grid(uint32_t* s_buf, int tid, int nthreads) {
    if constexpr (!STAGE_GRID) {
        return nullptr;
    } else if constexpr (TY == 16) {
        uint2* dst = (uint2*) s_buf;
        const uint2* src = (const uint2*) iq2xxs_grid;
        for (int i = tid; i < 256; i += nthreads) dst[i] = src[i];
        __syncthreads();
        return s_buf;
    } else if constexpr (TY == 17) {
        uint2* dst = (uint2*) s_buf;
        const uint2* src = (const uint2*) iq2xs_grid;
        for (int i = tid; i < 512; i += nthreads) dst[i] = src[i];
        __syncthreads();
        return s_buf;
    } else if constexpr (TY == 22) {
        uint2* dst = (uint2*) s_buf;
        const uint2* src = (const uint2*) iq2s_grid;
        for (int i = tid; i < 1024; i += nthreads) dst[i] = src[i];
        __syncthreads();
        return s_buf;
    } else if constexpr (TY == 29) {
        uint2* dst = (uint2*) s_buf;
        const uint2* src = (const uint2*) iq1s_grid_gpu;
        for (int i = tid; i < 1024; i += nthreads) dst[i] = src[i];
        __syncthreads();
        return s_buf;
    } else if constexpr (TY == 18) {
        // 256 codebook words, then two 128-word tables of the per-byte sign masks (signs 0-3, signs 4-7).  A
        // mask depends only on the 7 bits it is unpacked from, and iq3xxs_grid's bytes are all >= 4, so the
        // consumer applies it as (g ^ mask) + (mask & 0x01010101): a negate per set byte, with no carry into
        // the next.  One pass per block replaces the per-lookup unpack_ksigns / __vcmpne4 / __vsub4 chain.
        // Ported from shinbunbun/llama-cpp-p100-patches (MIT), patches 29 `mmvq-iq3xxs-grid-smem` and 30
        // `mmvq-ksigns-smem`, measured there on a Tesla P100 (sm_60).
        for (int i = tid; i < 256; i += nthreads) s_buf[i] = iq3xxs_grid[i];
        stage_iq3xxs_masks(s_buf, tid, nthreads);
        __syncthreads();
        return s_buf;
    } else if constexpr (kTqType<TY>) {
        // the trellis types' int8 codebook: 2048 u16 (4 KB), read by 8 random lookups per 16 weights
        const uint32_t* src = (const uint32_t*) tq_lut_i8;
        for (int i = tid; i < 1024; i += nthreads) s_buf[i] = src[i];
        __syncthreads();
        return s_buf;
    } else if constexpr (TY == 21) {
        // 512 codebook words, then two 256-word tables: the byte masks of a sign byte's bits 0-3 and of its bits
        // 4-7 (IQ3_S stores one sign byte per 8 values).  As for IQ3_XXS, one pass per block replaces the
        // per-lookup __vcmpne4 / __vsub4 chain, and the codebook reads leave global memory.
        for (int i = tid; i < 512; i += nthreads) s_buf[i] = iq3s_grid[i];
        stage_iq3s_masks(s_buf, tid, nthreads);
        __syncthreads();
        return s_buf;
    } else {
        return nullptr;
    }
}

template<int TY, bool STAGE_GRID = kStageIqGrid<TY>>
struct IqGridWords {
    static constexpr int value = !STAGE_GRID ? 4 : (TY == 22 || TY == 29) ? 2048 : (TY == 17 || TY == 21 || kTqType<TY>) ? 1024 : (TY == 16 || TY == 18) ? 512 : 4;
};

// mmvq_kernel with the columns taken NC at a time.  After warp_sum every lane holds the same sum, so lane c stores
// column c.
// #778: `__shared__ alignas(16)` does not compile for gfx1030 (HIP); the attribute after the declarator does, but nvcc
// with MSVC as host does not take it.  Same layout either way.
#if defined(__HIPCC__)
#define STRATA_SHARED_ALIGN16_U32(name, ...) __shared__ uint32_t name[__VA_ARGS__] __attribute__((aligned(16)))
#else
#define STRATA_SHARED_ALIGN16_U32(name, ...) __shared__ alignas(16) uint32_t name[__VA_ARGS__]
#endif
template<int TY, int NC, bool STAGE_GRID = kStageIqGrid<TY>, bool EXACT_N = false>
__global__ void __launch_bounds__(128) mmvq_multi_kernel(const uint8_t* __restrict__ w, size_t row_bytes,
                                                         const block_q8_1* __restrict__ x, float* __restrict__ y,
                                                         int n_in, int n_out, int ncols) {
    STRATA_SHARED_ALIGN16_U32(s_grid_buf, IqGridWords<TY, STAGE_GRID>::value);
    const uint32_t* s_grid = stage_iq_grid<TY, STAGE_GRID>(s_grid_buf, threadIdx.y * 32 + threadIdx.x, 128);
    const int row = blockIdx.x * 4 + threadIdx.y;
    if (row >= n_out) return;
    const int lane = threadIdx.x;
    const int nb = n_in / Fmt<TY>::qk, xb = n_in / 32;
    const uint8_t* wr = w + (size_t) row * row_bytes;
    if constexpr (EXACT_N) {
        int off[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) off[c] = c * xb;
        float s[NC];
        row_dot_multi<TY, NC, true, STAGE_GRID>(wr, x, off, NC, nb, lane, s, s_grid);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (lane == c) y[(size_t) c * n_out + row] = s[c];
    } else {
        for (int c0 = 0; c0 < ncols; c0 += NC) {
            const int n = min(NC, ncols - c0);
            int off[NC];
#pragma unroll
            for (int c = 0; c < NC; ++c) off[c] = (c0 + min(c, n - 1)) * xb;
            float s[NC];
            row_dot_multi<TY, NC, false, STAGE_GRID>(wr, x, off, n, nb, lane, s, s_grid);
#pragma unroll
            for (int c = 0; c < NC; ++c)
                if (c < n && lane == c) y[(size_t) (c0 + c) * n_out + row] = s[c];
        }
    }
}

// ---------------------------------------------------------------- grouped native experts
constexpr int GU_ROWS = 8;     // rows per block (one warp each)

// Group g is computed by block row g % gridDim.y: rows y, y + gridDim.y, ... < *n_groups.  The callers size the
// launch for the most groups a call can have (cap: every entry its own expert), but the count is only known on the
// device, and a verify window's PCIe call usually has no group at all: with one block row per possible group every
// unused one was cap x (2 n_ff / 8 + n_embd / 8) blocks that started only to return.  A smaller gridDim.y strides
// instead.  Each group's rows are computed by the same warp code in the same order either way, so the results do
// not depend on gridDim.y (gridDim.y = cap_groups is the one-row-per-group launch).
template<int TG>
__global__ void __launch_bounds__(256) native_gu_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                        const int32_t* __restrict__ grp_start,
                                                        const int32_t* __restrict__ n_groups,
                                                        const int32_t* __restrict__ ent_tok,
                                                        const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                        float* __restrict__ gate, float* __restrict__ up) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * GU_ROWS + warp;             // 0 .. 2*n_ff
    if (row >= 2 * L.n_ff) return;
    const bool is_up = row >= L.n_ff;
    const int r = is_up ? row - (int) L.n_ff : row;
    const size_t off = (is_up ? L.up_off : 0) + (size_t) r * L.gu_row;
    const int nb = (int) (L.n_embd / Fmt<TG>::qk), xb = (int) (L.n_embd / 32);
    const int ng = *n_groups;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {
        const uint8_t* blob = (const uint8_t*) grp_ptr[g];
        const uint8_t* wr = blob + off;
        // NVFP4: the expert's global scales from the blob's tail; s_down goes on the down output (native_down_kernel):
        // folded into up it left the hidden near 1e-5, whose q8_1 block scale is an FP16 subnormal
        float scale = 1.0f;
        if (L.tail_off) {
            const float* tail = (const float*) (blob + L.tail_off);
            scale = is_up ? tail[1] : tail[0];
        }
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        for (int e = e0; e < e1; ++e) {
            const float s = row_dot<TG>(wr, xq + (size_t) ent_tok[e] * xb, nb, lane);
            if (lane == 0) (is_up ? up : gate)[(size_t) e * L.n_ff + r] = s * scale;
        }
    }
}

// native_gu_kernel with the group's entries taken GRP_NC at a time, each weight part decoded once per pass.
// A group has at most one entry per token of the window (kVerifyMaxT = 8; setup writes --spec 4, and a split window
// has halves of <= 4), so 4 takes such windows in one pass; 8 would take ~64-80 registers against ~48 (ptxas -v).
constexpr int GRP_NC = 4;

template<int TG, bool STAGE_GRID = kStageIqGrid<TG>, bool SUB16 = true>
__global__ void __launch_bounds__(256) native_gu_multi_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                              const int32_t* __restrict__ grp_start,
                                                              const int32_t* __restrict__ n_groups,
                                                              const int32_t* __restrict__ ent_tok,
                                                              const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                              float* __restrict__ gate, float* __restrict__ up) {
    const int ng = *n_groups;
    if (blockIdx.y >= ng) return;
    STRATA_SHARED_ALIGN16_U32(s_grid_buf, IqGridWords<TG, STAGE_GRID>::value);
    const uint32_t* s_grid = stage_iq_grid<TG, STAGE_GRID>(s_grid_buf, threadIdx.x, 256);
    const int nb = (int) (L.n_embd / Fmt<TG>::qk), xb = (int) (L.n_embd / 32);
    if constexpr (SUB16 && (TG == 144 || TG == 145)) {
        if (nb * Fmt<TG>::ipb == 160) {   // the trellis rows (tq_sub8_on): 160 dots at n_embd 2560, 8 lanes a row
            const int subwarp = threadIdx.x >> 3, t = threadIdx.x & 7;
            const int row = blockIdx.x * 32 + subwarp;            // 0 .. 2*n_ff
            if (row >= 2 * L.n_ff) return;
            const bool is_up = row >= L.n_ff;
            const int r = is_up ? row - (int) L.n_ff : row;
            const size_t w_off = (is_up ? L.up_off : 0) + (size_t) r * L.gu_row;
            float* dst = is_up ? up : gate;
            for (int g = blockIdx.y; g < ng; g += gridDim.y) {
                const uint8_t* wr = (const uint8_t*) grp_ptr[g] + w_off;
                const float sc = L.tail_off ? ((const float*) (grp_ptr[g] + L.tail_off))[is_up ? 1 : 0] : 1.0f;
                const int e0 = grp_start[g], e1 = grp_start[g + 1];
                for (int e = e0; e < e1; e += GRP_NC) {
                    const int n = min(GRP_NC, e1 - e);
                    if (n == 1) {
                        const int off1[1] = { ent_tok[e] * 80 };
                        float s1[1];
                        row_dot_160_sub8<TG, 1, true, STAGE_GRID>(wr, xq, off1, 1, t, s1, s_grid);
                        if (t == 0) dst[(size_t) e * L.n_ff + r] = s1[0] * sc;
                    } else if (n == 2) {
                        const int off2[2] = { ent_tok[e] * 80, ent_tok[e + 1] * 80 };
                        float s2[2];
                        row_dot_160_sub8<TG, 2, true, STAGE_GRID>(wr, xq, off2, 2, t, s2, s_grid);
                        if (t < 2) dst[(size_t) (e + t) * L.n_ff + r] = s2[t] * sc;
                    } else if (n == 3) {
                        const int off3[3] = { ent_tok[e] * 80, ent_tok[e + 1] * 80, ent_tok[e + 2] * 80 };
                        float s3[3];
                        row_dot_160_sub8<TG, 3, true, STAGE_GRID>(wr, xq, off3, 3, t, s3, s_grid);
                        if (t < 3) dst[(size_t) (e + t) * L.n_ff + r] = s3[t] * sc;
                    } else {
                        const int off4[4] = { ent_tok[e] * 80, ent_tok[e + 1] * 80, ent_tok[e + 2] * 80, ent_tok[e + 3] * 80 };
                        float s4[4];
                        row_dot_160_sub8<TG, 4, true, STAGE_GRID>(wr, xq, off4, 4, t, s4, s_grid);
                        if (t < 4) dst[(size_t) (e + t) * L.n_ff + r] = s4[t] * sc;
                    }
                }
            }
            return;
        }
    } else if constexpr (SUB16) {
        if (xb == 80 && nb * Fmt<TG>::ipb == 80) {   // row_dot_80_sub16 covers exactly 80 dots (not the 128-value TQ blocks)
            const int subwarp = threadIdx.x >> 4, t = threadIdx.x & 15;
            const int row = blockIdx.x * 16 + subwarp;            // 0 .. 2*n_ff
            if (row >= 2 * L.n_ff) return;
            const bool is_up = row >= L.n_ff;
            const int r = is_up ? row - (int) L.n_ff : row;
            const size_t w_off = (is_up ? L.up_off : 0) + (size_t) r * L.gu_row;
            float* dst = is_up ? up : gate;
            for (int g = blockIdx.y; g < ng; g += gridDim.y) {
                const uint8_t* wr = (const uint8_t*) grp_ptr[g] + w_off;
                // NVFP4: the expert's s_gate / s_up on the FP32 output, as native_gu_kernel (1 otherwise: exact)
                const float sc = L.tail_off ? ((const float*) (grp_ptr[g] + L.tail_off))[is_up ? 1 : 0] : 1.0f;
                const int e0 = grp_start[g], e1 = grp_start[g + 1];
                for (int e = e0; e < e1; e += GRP_NC) {
                    const int n = min(GRP_NC, e1 - e);
                    if (n == 1) {
                        const int off1[1] = { ent_tok[e] * 80 };
                        float s1[1];
                        row_dot_80_sub16<TG, 1, true, STAGE_GRID>(wr, xq, off1, 1, t, s1, s_grid);
                        if (t == 0) dst[(size_t) e * L.n_ff + r] = s1[0] * sc;
                    } else if (n == 2) {
                        const int off2[2] = { ent_tok[e] * 80, ent_tok[e + 1] * 80 };
                        float s2[2];
                        row_dot_80_sub16<TG, 2, true, STAGE_GRID>(wr, xq, off2, 2, t, s2, s_grid);
                        if (t < 2) dst[(size_t) (e + t) * L.n_ff + r] = s2[t] * sc;
                    } else if (n == 3) {
                        const int off3[3] = { ent_tok[e] * 80, ent_tok[e + 1] * 80, ent_tok[e + 2] * 80 };
                        float s3[3];
                        row_dot_80_sub16<TG, 3, true, STAGE_GRID>(wr, xq, off3, 3, t, s3, s_grid);
                        if (t < 3) dst[(size_t) (e + t) * L.n_ff + r] = s3[t] * sc;
                    } else {
                        const int off4[4] = { ent_tok[e] * 80, ent_tok[e + 1] * 80, ent_tok[e + 2] * 80, ent_tok[e + 3] * 80 };
                        float s4[4];
                        row_dot_80_sub16<TG, 4, true, STAGE_GRID>(wr, xq, off4, 4, t, s4, s_grid);
                        if (t < 4) dst[(size_t) (e + t) * L.n_ff + r] = s4[t] * sc;
                    }
                }
            }
            return;
        }
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    // cc 6.0 trellis experts (staged codebook): a block takes the row chunks blockIdx.x, + gridDim.x, ... so the 4 KB
    // codebook staging above is paid once for several chunks (the launch shrinks the grid to match); every other
    // launch has one chunk per block and this is the single pass it always was
    constexpr bool kRowLoop = kTqType<TG> && STAGE_GRID;
    if constexpr (kRowLoop) {
        // cc 6.0 trellis gate/up: the group's entries (GRP_NC at a time) are copied once to shared memory by the block, and the
        // block's row chunks all apply them from there: the apply reads each entry's q8_1 block with five 4-byte loads per
        // lane, which from global memory is what an extra entry costs on this card.  The same words, the same operations
        // in the same order (row_dot_multi), so the same sums.
        if (xb <= 80) {
            __shared__ uint32_t s_x[GRP_NC * 80 * 9];
            const block_q8_1* xs = reinterpret_cast<const block_q8_1*>(s_x);
            for (int g = blockIdx.y; g < ng; g += gridDim.y) {
                const uint8_t* gp = (const uint8_t*) grp_ptr[g];
                const int e0 = grp_start[g], e1 = grp_start[g + 1];
                for (int e = e0; e < e1; e += GRP_NC) {
                    const int n = min(GRP_NC, e1 - e);
                    __syncthreads();
                    for (int i = 0; i < n; ++i) {
                        const uint32_t* src = reinterpret_cast<const uint32_t*>(xq + (size_t) ent_tok[e + i] * xb);
                        for (int j = threadIdx.x; j < xb * 9; j += 256) s_x[i * xb * 9 + j] = src[j];
                    }
                    __syncthreads();
                    for (int rc = blockIdx.x; rc * GU_ROWS < 2 * L.n_ff; rc += gridDim.x) {
                        const int row = rc * GU_ROWS + warp;
                        if (row >= 2 * L.n_ff) continue;
                        const bool is_up = row >= L.n_ff;
                        const int r = is_up ? row - (int) L.n_ff : row;
                        const size_t w_off = (is_up ? L.up_off : 0) + (size_t) r * L.gu_row;
                        float* dst = is_up ? up : gate;
                        const uint8_t* wr = gp + w_off;
                        const float sc = L.tail_off ? ((const float*) (grp_ptr[g] + L.tail_off))[is_up ? 1 : 0] : 1.0f;
                        if (n == 1) {
                            const int off1[1] = { 0 };
                            float s1[1];
                            row_dot_multi<TG, 1, true, STAGE_GRID>(wr, xs, off1, 1, nb, lane, s1, s_grid);
                            if (lane == 0) dst[(size_t) e * L.n_ff + r] = s1[0] * sc;
                        } else if (n == 2) {
                            const int off2[2] = { 0, xb };
                            float s2[2];
                            row_dot_multi<TG, 2, true, STAGE_GRID>(wr, xs, off2, 2, nb, lane, s2, s_grid);
                            if (lane < 2) dst[(size_t) (e + lane) * L.n_ff + r] = s2[lane] * sc;
                        } else if (n == 3) {
                            const int off3[3] = { 0, xb, 2 * xb };
                            float s3[3];
                            row_dot_multi<TG, 3, true, STAGE_GRID>(wr, xs, off3, 3, nb, lane, s3, s_grid);
                            if (lane < 3) dst[(size_t) (e + lane) * L.n_ff + r] = s3[lane] * sc;
                        } else {
                            const int off4[4] = { 0, xb, 2 * xb, 3 * xb };
                            float s4[4];
                            row_dot_multi<TG, 4, true, STAGE_GRID>(wr, xs, off4, 4, nb, lane, s4, s_grid);
                            if (lane < 4) dst[(size_t) (e + lane) * L.n_ff + r] = s4[lane] * sc;
                        }
                    }
                }
            }
            return;
        }
    }
    for (int rc = blockIdx.x; ; rc += gridDim.x) {
    const int row = rc * GU_ROWS + warp;             // 0 .. 2*n_ff
    if (row >= 2 * L.n_ff) break;
    const bool is_up = row >= L.n_ff;
    const int r = is_up ? row - (int) L.n_ff : row;
    const size_t w_off = (is_up ? L.up_off : 0) + (size_t) r * L.gu_row;
    float* dst = is_up ? up : gate;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {   // the group stride, as native_gu_kernel
        const uint8_t* wr = (const uint8_t*) grp_ptr[g] + w_off;
                // NVFP4: the expert's s_gate / s_up on the FP32 output, as native_gu_kernel (1 otherwise: exact)
                const float sc = L.tail_off ? ((const float*) (grp_ptr[g] + L.tail_off))[is_up ? 1 : 0] : 1.0f;
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        for (int e = e0; e < e1; e += GRP_NC) {
            const int n = min(GRP_NC, e1 - e);
            if (n == 1) {
                const int off1[1] = { ent_tok[e] * xb };
                float s1[1];
                row_dot_multi<TG, 1, true, STAGE_GRID>(wr, xq, off1, 1, nb, lane, s1, s_grid);
                if (lane == 0) dst[(size_t) e * L.n_ff + r] = s1[0] * sc;
            } else if (n == 2) {
                const int off2[2] = { ent_tok[e] * xb, ent_tok[e + 1] * xb };
                float s2[2];
                row_dot_multi<TG, 2, true, STAGE_GRID>(wr, xq, off2, 2, nb, lane, s2, s_grid);
                if (lane < 2) dst[(size_t) (e + lane) * L.n_ff + r] = s2[lane] * sc;
            } else if (n == 3) {
                const int off3[3] = { ent_tok[e] * xb, ent_tok[e + 1] * xb, ent_tok[e + 2] * xb };
                float s3[3];
                row_dot_multi<TG, 3, true, STAGE_GRID>(wr, xq, off3, 3, nb, lane, s3, s_grid);
                if (lane < 3) dst[(size_t) (e + lane) * L.n_ff + r] = s3[lane] * sc;
            } else {
                const int off4[4] = { ent_tok[e] * xb, ent_tok[e + 1] * xb, ent_tok[e + 2] * xb, ent_tok[e + 3] * xb };
                float s4[4];
                row_dot_multi<TG, 4, true, STAGE_GRID>(wr, xq, off4, 4, nb, lane, s4, s_grid);
                if (lane < 4) dst[(size_t) (e + lane) * L.n_ff + r] = s4[lane] * sc;
            }
        }
    }
    if constexpr (!kRowLoop) break;
    }
}

// APR (prism.hadamard): y = H_block (signs * x) per block of each row, block = blockDim.x (a power of two <= 1024),
// one thread block per (row block, row).  The butterflies are the CPU's (hadamard.cpp) in the same order, so the
// rotated values are the same floats.
template<typename T> __device__ __forceinline__ float ld_f(const T* p) { return (float) *p; }
template<> __device__ __forceinline__ float ld_f<__half>(const __half* p) { return __half2float(*p); }
template<typename T> __device__ __forceinline__ void st_f(T* p, float v) { *p = v; }
template<> __device__ __forceinline__ void st_f<__half>(__half* p, float v) { *p = __float2half(v); }
template<typename T>
__global__ void hadamard_rows_kernel(const T* __restrict__ x, T* __restrict__ y, int64_t width,
                                     const float* __restrict__ signs, float scale) {
    extern __shared__ float hbuf[];
    const int n = blockDim.x, i = threadIdx.x;
    const int64_t col = (int64_t) blockIdx.x * n + i, at = (int64_t) blockIdx.y * width + col;
    float v = ld_f<T>(x + at);
    if (signs) v *= signs[col];
    hbuf[i] = v;
    for (int len = 1; len < n; len <<= 1) {
        __syncthreads();
        if ((i & len) == 0) {
            const float a = hbuf[i], c = hbuf[i + len];
            hbuf[i] = a + c;
            hbuf[i + len] = a - c;
        }
    }
    __syncthreads();
    st_f<T>(y + at, hbuf[i] * scale);
}

__global__ void swiglu_entries_kernel(const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ h,
                                      long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    h[i] = (g / (1.0f + __expf(-g))) * up[i];
}

template<int TD>
__global__ void __launch_bounds__(256) native_down_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                          const int32_t* __restrict__ grp_start,
                                                          const int32_t* __restrict__ n_groups,
                                                          const int32_t* __restrict__ ent_dst,
                                                          const block_q8_1* __restrict__ hq, NativeExpertLayout L,
                                                          float* __restrict__ out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * 8 + warp;
    if (r >= L.n_embd) return;
    const size_t off = L.down_off + (size_t) r * L.d_row;
    const int nb = (int) (L.n_ff / Fmt<TD>::qk), hb = (int) (L.n_ff / 32);
    const int ng = *n_groups;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {   // as native_gu_kernel
        const uint8_t* blob = (const uint8_t*) grp_ptr[g];
        const uint8_t* wr = blob + off;
        const float sd = L.tail_off ? ((const float*) (blob + L.tail_off))[2] : 1.0f;   // NVFP4: s_down
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        for (int e = e0; e < e1; ++e) {
            const float s = row_dot<TD>(wr, hq + (size_t) e * hb, nb, lane);
            if (lane == 0) out[(size_t) ent_dst[e] * L.n_embd + r] = s * sd;
        }
    }
}

// the 8-lane down rows: IQ4_NL's K = 40, NVFP4's K = 20
template<int TD, int NC, bool EXACT_N = false, bool STAGE_GRID = false>
__device__ __forceinline__ void row_dot_sub8(const uint8_t* row, const block_q8_1* x, const int (&off)[NC], int n, int t,
                                             float (&s)[NC], const uint32_t* __restrict__ s_grid = nullptr) {
    if constexpr (TD == 40) row_dot_20_sub8<TD, NC, EXACT_N>(row, x, off, n, t, s);
    else row_dot_40_sub8<TD, NC, EXACT_N, STAGE_GRID>(row, x, off, n, t, s, s_grid);
}

// native_down_kernel with the entries taken GRP_NC at a time
template<int TD, bool SUB_DOWN = true, bool STAGE_TQ = false>
__global__ void __launch_bounds__(256) native_down_multi_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                                const int32_t* __restrict__ grp_start,
                                                                const int32_t* __restrict__ n_groups,
                                                                const int32_t* __restrict__ ent_dst,
                                                                const block_q8_1* __restrict__ hq, NativeExpertLayout L,
                                                                float* __restrict__ out) {
    const int ng = *n_groups;
    if (blockIdx.y >= ng) return;
    STRATA_SHARED_ALIGN16_U32(s_grid_buf, IqGridWords<TD, STAGE_TQ>::value);   // the trellis codebook on a cc 6.0 card, else 4 unused words
    const uint32_t* s_grid = stage_iq_grid<TD, STAGE_TQ>(s_grid_buf, threadIdx.x, 256);
    __shared__ uint32_t s_hq_buf[GRP_NC * 20 * 9];
    const int nb = (int) (L.n_ff / Fmt<TD>::qk), hb = (int) (L.n_ff / 32);
    if constexpr (SUB_DOWN && kTqType<TD> && STAGE_TQ) {
        // cc 6.0 trellis down rows (K = 40 dots at n_ff 640, the IQ4_NL row shape): 8 lanes a row, 32 rows a chunk, and a
        // block takes the chunks blockIdx.x, + gridDim.x, ... of one staged codebook and one staged set of activations
        // (the launch shrinks the grid to match).  The sums are row_dot_40_sub8's, as the IQ4_NL rows.
        if (hb == 20) {
            const int t = threadIdx.x & 7;
            const block_q8_1* s_hq = reinterpret_cast<const block_q8_1*>(s_hq_buf);
            for (int g = blockIdx.y; g < ng; g += gridDim.y) {
                const uint8_t* gp = (const uint8_t*) grp_ptr[g];
                const float sd = L.tail_off ? ((const float*) (grp_ptr[g] + L.tail_off))[2] : 1.0f;
                const int e0 = grp_start[g], e1 = grp_start[g + 1];
                for (int e = e0; e < e1; e += GRP_NC) {
                    const int n = min(GRP_NC, e1 - e);
                    const uint32_t* src = reinterpret_cast<const uint32_t*>(hq + (size_t) e * 20);
                    const int words = n * (20 * 9);
                    __syncthreads();
                    for (int i = threadIdx.x; i < words; i += 256) s_hq_buf[i] = src[i];
                    __syncthreads();
                    for (int rc = blockIdx.x; rc * 32 < L.n_embd; rc += gridDim.x) {
                        const int r = rc * 32 + (threadIdx.x >> 3);
                        if (r >= L.n_embd) continue;
                        const uint8_t* wr = gp + L.down_off + (size_t) r * L.d_row;
                        if (n == 1) {
                            const int off1[1] = { 0 };
                            float s1[1];
                            row_dot_sub8<TD, 1, true, STAGE_TQ>(wr, s_hq, off1, 1, t, s1, s_grid);
                            if (t == 0) out[(size_t) ent_dst[e] * L.n_embd + r] = s1[0] * sd;
                        } else if (n == 2) {
                            const int off2[2] = { 0, 20 };
                            float s2[2];
                            row_dot_sub8<TD, 2, true, STAGE_TQ>(wr, s_hq, off2, 2, t, s2, s_grid);
                            if (t < 2) out[(size_t) ent_dst[e + t] * L.n_embd + r] = s2[t] * sd;
                        } else if (n == 3) {
                            const int off3[3] = { 0, 20, 40 };
                            float s3[3];
                            row_dot_sub8<TD, 3, true, STAGE_TQ>(wr, s_hq, off3, 3, t, s3, s_grid);
                            if (t < 3) out[(size_t) ent_dst[e + t] * L.n_embd + r] = s3[t] * sd;
                        } else {
                            const int off4[4] = { 0, 20, 40, 60 };
                            float s4[4];
                            row_dot_sub8<TD, 4, true, STAGE_TQ>(wr, s_hq, off4, 4, t, s4, s_grid);
                            if (t < 4) out[(size_t) ent_dst[e + t] * L.n_embd + r] = s4[t] * sd;
                        }
                    }
                }
            }
            return;
        }
    } else
    if constexpr (SUB_DOWN && (TD == 20 || TD == 40)) {   // IQ4_NL (K 40) and NVFP4 (K 20): 8-lane rows
        if (hb == 20) {
            const int r = blockIdx.x * 32 + (threadIdx.x >> 3);
            const int t = threadIdx.x & 7;
            const bool valid_r = (r < L.n_embd);
            const size_t w_off = L.down_off + (size_t) (valid_r ? r : 0) * L.d_row;
            const block_q8_1* s_hq = reinterpret_cast<const block_q8_1*>(s_hq_buf);
            for (int g = blockIdx.y; g < ng; g += gridDim.y) {
                const uint8_t* wr = (const uint8_t*) grp_ptr[g] + w_off;
                const float sd = L.tail_off ? ((const float*) (grp_ptr[g] + L.tail_off))[2] : 1.0f;   // NVFP4: s_down
                const int e0 = grp_start[g], e1 = grp_start[g + 1];
                for (int e = e0; e < e1; e += GRP_NC) {
                    const int n = min(GRP_NC, e1 - e);
                    const uint32_t* src = reinterpret_cast<const uint32_t*>(hq + (size_t) e * 20);
                    const int words = n * (20 * 9);
                    __syncthreads();
                    for (int i = threadIdx.x; i < words; i += 256) s_hq_buf[i] = src[i];
                    __syncthreads();
                    if (valid_r) {
                        if (n == 1) {
                            const int off1[1] = { 0 };
                            float s1[1];
                            row_dot_sub8<TD, 1, true, STAGE_TQ>(wr, s_hq, off1, 1, t, s1, s_grid);
                            if (t == 0) out[(size_t) ent_dst[e] * L.n_embd + r] = s1[0] * sd;
                        } else if (n == 2) {
                            const int off2[2] = { 0, 20 };
                            float s2[2];
                            row_dot_sub8<TD, 2, true, STAGE_TQ>(wr, s_hq, off2, 2, t, s2, s_grid);
                            if (t < 2) out[(size_t) ent_dst[e + t] * L.n_embd + r] = s2[t] * sd;
                        } else if (n == 3) {
                            const int off3[3] = { 0, 20, 40 };
                            float s3[3];
                            row_dot_sub8<TD, 3, true, STAGE_TQ>(wr, s_hq, off3, 3, t, s3, s_grid);
                            if (t < 3) out[(size_t) ent_dst[e + t] * L.n_embd + r] = s3[t] * sd;
                        } else {
                            const int off4[4] = { 0, 20, 40, 60 };
                            float s4[4];
                            row_dot_sub8<TD, 4, true, STAGE_TQ>(wr, s_hq, off4, 4, t, s4, s_grid);
                            if (t < 4) out[(size_t) ent_dst[e + t] * L.n_embd + r] = s4[t] * sd;
                        }
                    }
                }
            }
            return;
        }
    } else if constexpr (SUB_DOWN && TD == 42) {
        if (hb == 20) {
            const int r = blockIdx.x * 16 + (threadIdx.x >> 4);
            const int t = threadIdx.x & 15;
            const bool valid_r = (r < L.n_embd);
            const size_t w_off = L.down_off + (size_t) (valid_r ? r : 0) * L.d_row;
            const block_q8_1* s_hq = reinterpret_cast<const block_q8_1*>(s_hq_buf);
            for (int g = blockIdx.y; g < ng; g += gridDim.y) {
                const uint8_t* wr = (const uint8_t*) grp_ptr[g] + w_off;
                const float sd = L.tail_off ? ((const float*) (grp_ptr[g] + L.tail_off))[2] : 1.0f;   // NVFP4: s_down
                const int e0 = grp_start[g], e1 = grp_start[g + 1];
                for (int e = e0; e < e1; e += GRP_NC) {
                    const int n = min(GRP_NC, e1 - e);
                    const uint32_t* src = reinterpret_cast<const uint32_t*>(hq + (size_t) e * 20);
                    const int words = n * (20 * 9);
                    __syncthreads();
                    for (int i = threadIdx.x; i < words; i += 256) s_hq_buf[i] = src[i];
                    __syncthreads();
                    if (valid_r) {
                        if (n == 1) {
                            const int off1[1] = { 0 };
                            float s1[1];
                            row_dot_20_sub16<TD, 1, true>(wr, s_hq, off1, 1, t, s1);
                            if (t == 0) out[(size_t) ent_dst[e] * L.n_embd + r] = s1[0] * sd;
                        } else if (n == 2) {
                            const int off2[2] = { 0, 20 };
                            float s2[2];
                            row_dot_20_sub16<TD, 2, true>(wr, s_hq, off2, 2, t, s2);
                            if (t < 2) out[(size_t) ent_dst[e + t] * L.n_embd + r] = s2[t] * sd;
                        } else if (n == 3) {
                            const int off3[3] = { 0, 20, 40 };
                            float s3[3];
                            row_dot_20_sub16<TD, 3, true>(wr, s_hq, off3, 3, t, s3);
                            if (t < 3) out[(size_t) ent_dst[e + t] * L.n_embd + r] = s3[t] * sd;
                        } else {
                            const int off4[4] = { 0, 20, 40, 60 };
                            float s4[4];
                            row_dot_20_sub16<TD, 4, true>(wr, s_hq, off4, 4, t, s4);
                            if (t < 4) out[(size_t) ent_dst[e + t] * L.n_embd + r] = s4[t] * sd;
                        }
                    }
                }
            }
            return;
        }
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * 8 + warp;
    const bool valid_r = (r < L.n_embd);
    const size_t w_off = L.down_off + (size_t) (valid_r ? r : 0) * L.d_row;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {   // the group stride, as native_gu_kernel
        const uint8_t* wr = (const uint8_t*) grp_ptr[g] + w_off;
                const float sd = L.tail_off ? ((const float*) (grp_ptr[g] + L.tail_off))[2] : 1.0f;   // NVFP4: s_down
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        if (hb == 20) {
            const block_q8_1* s_hq = reinterpret_cast<const block_q8_1*>(s_hq_buf);
            for (int e = e0; e < e1; e += GRP_NC) {
                const int n = min(GRP_NC, e1 - e);
                const uint32_t* src = reinterpret_cast<const uint32_t*>(hq + (size_t) e * 20);
                const int words = n * (20 * 9);
                __syncthreads();
                for (int i = threadIdx.x; i < words; i += 256) s_hq_buf[i] = src[i];
                __syncthreads();
                if (valid_r) {
                    if (n == 1) {
                        const int off1[1] = { 0 };
                        float s1[1];
                        row_dot_multi<TD, 1, true, STAGE_TQ>(wr, s_hq, off1, 1, nb, lane, s1, s_grid);
                        if (lane == 0) out[(size_t) ent_dst[e] * L.n_embd + r] = s1[0] * sd;
                    } else if (n == 2) {
                        const int off2[2] = { 0, 20 };
                        float s2[2];
                        row_dot_multi<TD, 2, true, STAGE_TQ>(wr, s_hq, off2, 2, nb, lane, s2, s_grid);
                        if (lane < 2) out[(size_t) ent_dst[e + lane] * L.n_embd + r] = s2[lane] * sd;
                    } else {
                        int off[GRP_NC];
#pragma unroll
                        for (int c = 0; c < GRP_NC; ++c) off[c] = min(c, n - 1) * 20;
                        float s[GRP_NC];
                        row_dot_multi<TD, GRP_NC, false, STAGE_TQ>(wr, s_hq, off, n, nb, lane, s, s_grid);
#pragma unroll
                        for (int c = 0; c < GRP_NC; ++c)
                            if (c < n && lane == c) out[(size_t) ent_dst[e + c] * L.n_embd + r] = s[c] * sd;
                    }
                }
            }
        } else if (valid_r) {
            for (int e = e0; e < e1; e += GRP_NC) {
                const int n = min(GRP_NC, e1 - e);
                int off[GRP_NC];
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c) off[c] = (e + min(c, n - 1)) * hb;
                float s[GRP_NC];
                row_dot_multi<TD, GRP_NC, false, STAGE_TQ>(wr, hq, off, n, nb, lane, s, s_grid);
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c)
                    if (c < n && lane == c) out[(size_t) ent_dst[e + c] * L.n_embd + r] = s[c] * sd;
            }
        }
    }
}

// ---------------------------------------------------------------- EXPDEC (STRATA_EXPDEC=1): the group's activations
// in shared memory, R rows per lane
//
// native_gu_multi_kernel / native_down_multi_kernel decode each weight part once and apply it to every entry of the
// group, but the gate/up apply re-reads the entry's q8_1 block from global memory with 9 scattered 4-byte loads
// (36-byte blocks: a warp's load touches 32 sectors), so an entry costs about 60% of a whole expert (nsys, 2080 Ti,
// L02 at 131k: ~2.6 us per expert and ~1.6 us per entry and layer) although it reads nothing from DRAM.  Here a
// block stages its group's entries (ED_NC at a time, as the default) in shared memory laid out so that a call's
// words are two 16-byte loads (gate/up, Q2_0 down: the block's two halves in separate planes, conflict-free) or two
// 8-byte loads (IQ4_NL down), the fp16 block scale already a float; every lane computes R rows at once, so the
// activation words are loaded once per R rows; the IQ3_XXS / IQ3_S codebooks are staged too (IQ2_* already were),
// and a block takes `rb` rows (auto: the most that keep ~STRATA_EXPDEC_MINBLK blocks in flight) instead of 16.
// Each (row, entry) sum is the default kernel's sum tree (row_dot_80_sub16 / row_dot_40_sub8 / row_dot_20_sub16: the
// same lane does the same calls in the same order with the same operations, `apply_u` = `apply` on the block's
// words), so the output is bitwise the same (expdec_bench checks it; docs/EXPERTDEC.md).
#ifndef STRATA_EXPDEC_MINB
#define STRATA_EXPDEC_MINB 4   // gate/up __launch_bounds__ min blocks per SM by default (STRATA_EXPDEC_MINB at runtime): 4 = up to 64
                               // registers, as the default kernels (bench: equal on the 2080 Ti, 0.87-0.95 on the 3060); 2 = up to 128
#endif
constexpr int ED_NC = GRP_NC;   // entries staged at once
constexpr int ED_XB = 80;       // q8_1 blocks of a gate/up row (n_embd 2560)
constexpr int ED_HB = 20;       // q8_1 blocks of a down row (n_ff 640)

// the codebook of a gate/up format in shared memory: words, and the copy (every thread of the block; sync after)
template<int TG> struct EdGrid;
template<> struct EdGrid<16> { static constexpr int words = 512;
    __device__ static void copy(uint32_t* s, int tid) { const uint2* src = (const uint2*) iq2xxs_grid; uint2* dst = (uint2*) s; for (int i = tid; i < 256; i += 256) dst[i] = src[i]; } };
template<> struct EdGrid<17> { static constexpr int words = 1024;
    __device__ static void copy(uint32_t* s, int tid) { const uint2* src = (const uint2*) iq2xs_grid; uint2* dst = (uint2*) s; for (int i = tid; i < 512; i += 256) dst[i] = src[i]; } };
template<> struct EdGrid<22> { static constexpr int words = 2048;
    __device__ static void copy(uint32_t* s, int tid) { const uint2* src = (const uint2*) iq2s_grid; uint2* dst = (uint2*) s; for (int i = tid; i < 1024; i += 256) dst[i] = src[i]; } };
// (IQ3_XXS / IQ3_S: the codebook and, behind it, the sign-mask tables Split<TG>::load<true> reads - stage_iq_grid's layout)
template<> struct EdGrid<18> { static constexpr int words = 512;
    __device__ static void copy(uint32_t* s, int tid) {
        for (int i = tid; i < 256; i += 256) s[i] = iq3xxs_grid[i];
        stage_iq3xxs_masks(s, tid, 256);
    } };
template<> struct EdGrid<21> { static constexpr int words = 1024;
    __device__ static void copy(uint32_t* s, int tid) {
        for (int i = tid; i < 512; i += 256) s[i] = iq3s_grid[i];
        stage_iq3s_masks(s, tid, 256);
    } };

// One entry's q8_1 row (XB blocks of 36 bytes, 4-byte aligned) into shared memory as entry c: the 8 words of block k
// in planes (HALVES: s_u[(2 c + h) XB + k] = words 4h..4h+3, two 16-byte loads per call, conflict-free over 16
// consecutive k; else s_u[(c XB + k) 2 + h], the block's 32 bytes together, for the IQ4_NL pairs) and the fp16 scale
// as s_ds[c XB + k] = __low2float(ds), the value `apply` computes.  All 256 threads.
template<int XB, bool HALVES>
__device__ __forceinline__ void ed_stage_entry(int4* __restrict__ s_u, float* __restrict__ s_ds, int c,
                                               const block_q8_1* __restrict__ src, int tid) {
    int* su = (int*) s_u;
    const int* w32 = (const int*) src;
    for (int w = tid; w < XB * 9; w += 256) {
        const int k = w / 9, j = w - k * 9;
        const int v = w32[w];
        if (j == 0) {
            const half2 h = *reinterpret_cast<const half2*>(&v);
            s_ds[c * XB + k] = __low2float(h);
        } else {
            const int q = j - 1;
            if constexpr (HALVES) su[((c * 2 + (q >> 2)) * XB + k) * 4 + (q & 3)] = v;
            else su[(c * XB + k) * 8 + q] = v;
        }
    }
}
template<int XB>
__device__ __forceinline__ void ed_load_u(const int4* __restrict__ s_u, int c, int k, int (&u)[8]) {
    const int4 a = s_u[(c * 2 + 0) * XB + k], b = s_u[(c * 2 + 1) * XB + k];
    u[0] = a.x; u[1] = a.y; u[2] = a.z; u[3] = a.w;
    u[4] = b.x; u[5] = b.y; u[6] = b.z; u[7] = b.w;
}

// R gate/up rows against the NC staged entries, lane t of a 16-lane sub-warp: row_dot_80_sub16's tree (calls t,
// t+32, t+64 into s; t+16, t+48 into s16; s = __fadd_rn(s, s16); xor 8-4-2-1), every call's weight part decoded once
// per row and the entry's block loaded once per NC x R products.
template<int TG, int R, int NC>
__device__ __forceinline__ void ed_gu_rows(const uint8_t* const* wr, const int4* __restrict__ s_u,
                                           const float* __restrict__ s_ds, const uint32_t* __restrict__ s_grid, int t,
                                           float (&s)[R][NC]) {
    using F = Fmt<TG>;
    using S = Split<TG>;
    float s16[R][NC];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < NC; ++c) { s[r][c] = 0.0f; s16[r][c] = 0.0f; }
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int k = t + 32 * j;
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        typename S::W w[R];
#pragma unroll
        for (int r = 0; r < R; ++r) w[r] = S::template load<true>(wr[r], kbx, iqs, s_grid);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            int u[8];
            ed_load_u<ED_XB>(s_u, c, k, u);
            const float d8 = s_ds[c * ED_XB + k];
#pragma unroll
            for (int r = 0; r < R; ++r) s[r][c] += S::apply_u(w[r], u, d8);
        }
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int k = t + 16 + 32 * j;
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        typename S::W w[R];
#pragma unroll
        for (int r = 0; r < R; ++r) w[r] = S::template load<true>(wr[r], kbx, iqs, s_grid);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            int u[8];
            ed_load_u<ED_XB>(s_u, c, k, u);
            const float d8 = s_ds[c * ED_XB + k];
#pragma unroll
            for (int r = 0; r < R; ++r) s16[r][c] += S::apply_u(w[r], u, d8);
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            s[r][c] = __fadd_rn(s[r][c], s16[r][c]);
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) s[r][c] += __shfl_xor_sync(0xffffffffu, s[r][c], o);
        }
}

// gate/up: 16 sub-warps of 16 lanes, R rows each = 16 R rows per pass, `rb` rows per block (a multiple of 16 R; 2 n_ff
// is one too, so a pass is whole or absent for the entire block).  Group stride as native_gu_multi_kernel.
template<int TG, int R, int MINB>
__global__ void __launch_bounds__(256, MINB) expdec_gu_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                                            const int32_t* __restrict__ grp_start,
                                                                            const int32_t* __restrict__ n_groups,
                                                                            const int32_t* __restrict__ ent_tok,
                                                                            const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                                            float* __restrict__ gate, float* __restrict__ up, int rb) {
    const int ng = *n_groups;
    if (blockIdx.y >= ng) return;
    __shared__ alignas(16) int4 s_u[ED_NC * 2 * ED_XB];
    __shared__ float s_ds[ED_NC * ED_XB];
    STRATA_SHARED_ALIGN16_U32(s_grid_buf, EdGrid<TG>::words);
    const int tid = threadIdx.x;
    EdGrid<TG>::copy(s_grid_buf, tid);   // visible after the staging's __syncthreads
    const int sw = tid >> 4, t = tid & 15;
    const int nrow = 2 * (int) L.n_ff, n_ff = (int) L.n_ff;
    const int row_base = blockIdx.x * rb;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {
        const uint8_t* blob = (const uint8_t*) grp_ptr[g];
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        for (int e = e0; e < e1; e += ED_NC) {
            const int n = min(ED_NC, e1 - e);
            __syncthreads();
            for (int c = 0; c < n; ++c) ed_stage_entry<ED_XB, true>(s_u, s_ds, c, xq + (size_t) ent_tok[e + c] * ED_XB, tid);
            __syncthreads();
            for (int p = 0; p * 16 * R < rb; ++p) {
                const int row0 = row_base + p * 16 * R + sw * R;
                if (row0 >= nrow) break;   // uniform over the block
                const bool is_up = row0 >= n_ff;
                const int r0 = is_up ? row0 - n_ff : row0;
                const uint8_t* wr[R];
#pragma unroll
                for (int r = 0; r < R; ++r) wr[r] = blob + (is_up ? L.up_off : 0) + (size_t) (r0 + r) * L.gu_row;
                float* dst = is_up ? up : gate;
                auto run = [&](auto nc) {
                    constexpr int NC = decltype(nc)::value;
                    float s[R][NC];
                    ed_gu_rows<TG, R, NC>(wr, s_u, s_ds, s_grid_buf, t, s);
#pragma unroll
                    for (int c = 0; c < NC; ++c)
                        if (t == c)
#pragma unroll
                            for (int r = 0; r < R; ++r) dst[(size_t) (e + c) * n_ff + r0 + r] = s[r][c];
                };
                switch (n) {
                    case 1: run(std::integral_constant<int, 1>{}); break;
                    case 2: run(std::integral_constant<int, 2>{}); break;
                    case 3: run(std::integral_constant<int, 3>{}); break;
                    default: run(std::integral_constant<int, 4>{}); break;
                }
            }
        }
    }
}

// R IQ4_NL down rows, lane t of an 8-lane sub-warp: row_dot_40_sub8's tree.  s_p: the block's 8 words together
// (ed_stage_entry<ED_HB, false>), read as the int2 pairs (iqs, iqs + 1) and (iqs + 4, iqs + 5).
template<int R, int NC>
__device__ __forceinline__ void ed_d20_rows(const uint8_t* const* wr, const int2* __restrict__ s_p,
                                            const float* __restrict__ s_ds, int t, float (&s)[R][NC]) {
    using F = Fmt<20>;
    using S = Split<20>;
    auto term = [&](int k, int r, int c, const typename S::W& w) -> float {
        const int kbx = k / F::ipb, par = k & 1;
        const int2 p0 = s_p[(c * ED_HB + kbx) * 4 + par], p1 = s_p[(c * ED_HB + kbx) * 4 + 2 + par];
        return S::apply_u(w, p0, p1, s_ds[c * ED_HB + kbx]);
    };
    auto loadw = [&](int k, typename S::W (&w)[R]) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
#pragma unroll
        for (int r = 0; r < R; ++r) w[r] = S::load(wr[r], kbx, iqs);
    };
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < NC; ++c) s[r][c] = 0.0f;
#pragma unroll
    for (int j = 0; j < 2; ++j) {   // k = t, t + 32
        const int k = t + 32 * j;
        typename S::W w[R];
        loadw(k, w);
#pragma unroll
        for (int c = 0; c < NC; ++c)
#pragma unroll
            for (int r = 0; r < R; ++r) s[r][c] += term(k, r, c, w[r]);
    }
    {
        const int k = t + 16;
        typename S::W w[R];
        loadw(k, w);
#pragma unroll
        for (int c = 0; c < NC; ++c)
#pragma unroll
            for (int r = 0; r < R; ++r) {
                float s16 = 0.0f;
                s16 += term(k, r, c, w[r]);
                s[r][c] = __fadd_rn(s[r][c], s16);
            }
    }
    float s8[R][NC];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < NC; ++c) s8[r][c] = 0.0f;
    {
        const int k = t + 8;
        typename S::W w[R];
        loadw(k, w);
#pragma unroll
        for (int c = 0; c < NC; ++c)
#pragma unroll
            for (int r = 0; r < R; ++r) s8[r][c] += term(k, r, c, w[r]);
    }
    {
        const int k = t + 24;
        typename S::W w[R];
        loadw(k, w);
#pragma unroll
        for (int c = 0; c < NC; ++c)
#pragma unroll
            for (int r = 0; r < R; ++r) {
                float s24 = 0.0f;
                s24 += term(k, r, c, w[r]);
                s[r][c] = __fadd_rn(s[r][c], __fadd_rn(s8[r][c], s24));
            }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < NC; ++c)
#pragma unroll
            for (int o = 4; o > 0; o >>= 1) s[r][c] += __shfl_xor_sync(0xffffffffu, s[r][c], o);
}

// R Q2_0 down rows, lane t of a 16-lane sub-warp: row_dot_20_sub16's tree (call t; call t + 16 for t < 4 into s16).
template<int R, int NC>
__device__ __forceinline__ void ed_d42_rows(const uint8_t* const* wr, const int4* __restrict__ s_u,
                                            const float* __restrict__ s_ds, int t, float (&s)[R][NC]) {
    using F = Fmt<42>;
    using S = Split<42>;
    {
        const int k = t, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        typename S::W w[R];
#pragma unroll
        for (int r = 0; r < R; ++r) w[r] = S::load(wr[r], kbx, iqs);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            int u[8];
            ed_load_u<ED_HB>(s_u, c, k, u);
            const float d8 = s_ds[c * ED_HB + k];
#pragma unroll
            for (int r = 0; r < R; ++r) { s[r][c] = 0.0f; s[r][c] += S::apply_u(w[r], u, d8); }
        }
    }
    float s16[R][NC];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < NC; ++c) s16[r][c] = 0.0f;
    if (t < 4) {
        const int k = t + 16, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        typename S::W w[R];
#pragma unroll
        for (int r = 0; r < R; ++r) w[r] = S::load(wr[r], kbx, iqs);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            int u[8];
            ed_load_u<ED_HB>(s_u, c, k, u);
            const float d8 = s_ds[c * ED_HB + k];
#pragma unroll
            for (int r = 0; r < R; ++r) s16[r][c] += S::apply_u(w[r], u, d8);
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            s[r][c] = __fadd_rn(s[r][c], s16[r][c]);
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) s[r][c] += __shfl_xor_sync(0xffffffffu, s[r][c], o);
        }
}

// down (IQ4_NL: 32 sub-warps of 8 lanes; Q2_0: 16 of 16), R rows per lane, `rb` rows per block (a multiple of the
// rows per pass; n_embd too).  The entries' h rows are contiguous in hq (entry e at e * ED_HB).
template<int TD, int R>
__global__ void __launch_bounds__(256, STRATA_EXPDEC_MINB) expdec_down_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                                              const int32_t* __restrict__ grp_start,
                                                                              const int32_t* __restrict__ n_groups,
                                                                              const int32_t* __restrict__ ent_dst,
                                                                              const block_q8_1* __restrict__ hq, NativeExpertLayout L,
                                                                              float* __restrict__ out, int rb) {
    const int ng = *n_groups;
    if (blockIdx.y >= ng) return;
    constexpr int LANES = TD == 20 ? 8 : 16;
    constexpr int RPP = (256 / LANES) * R;   // rows per pass
    __shared__ alignas(16) int4 s_u[ED_NC * 2 * ED_HB];
    __shared__ float s_ds[ED_NC * ED_HB];
    const int tid = threadIdx.x;
    const int sw = tid / LANES, t = tid % LANES;
    const int nrow = (int) L.n_embd;
    const int row_base = blockIdx.x * rb;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {
        const uint8_t* blob = (const uint8_t*) grp_ptr[g] + L.down_off;
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        for (int e = e0; e < e1; e += ED_NC) {
            const int n = min(ED_NC, e1 - e);
            __syncthreads();
            for (int c = 0; c < n; ++c) ed_stage_entry<ED_HB, TD == 42>(s_u, s_ds, c, hq + (size_t) (e + c) * ED_HB, tid);
            __syncthreads();
            for (int p = 0; p * RPP < rb; ++p) {
                const int row0 = row_base + p * RPP + sw * R;
                if (row0 >= nrow) break;   // uniform over the block
                const uint8_t* wr[R];
#pragma unroll
                for (int r = 0; r < R; ++r) wr[r] = blob + (size_t) (row0 + r) * L.d_row;
                auto run = [&](auto nc) {
                    constexpr int NC = decltype(nc)::value;
                    float s[R][NC];
                    if constexpr (TD == 20) ed_d20_rows<R, NC>(wr, (const int2*) s_u, s_ds, t, s);
                    else ed_d42_rows<R, NC>(wr, s_u, s_ds, t, s);
#pragma unroll
                    for (int c = 0; c < NC; ++c)
                        if (t == c) {
                            const size_t o = (size_t) ent_dst[e + c] * L.n_embd + row0;
#pragma unroll
                            for (int r = 0; r < R; ++r) out[o + r] = s[r][c];
                        }
                };
                switch (n) {
                    case 1: run(std::integral_constant<int, 1>{}); break;
                    case 2: run(std::integral_constant<int, 2>{}); break;
                    case 3: run(std::integral_constant<int, 3>{}); break;
                    default: run(std::integral_constant<int, 4>{}); break;
                }
            }
        }
    }
}

// ---------------------------------------------------------------- q8_1 (quantize.cu)
// value i of a q8_1 row set (the warp holds block i / 32, lane = i % 32)
__device__ __forceinline__ void q8_1_store(const float xi, block_q8_1* __restrict__ y, const long long i) {
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float d = q8_1_finite(amax / 127.0f);   // #606: q8_1_finite.hpp - the same bits for every finite block
    const int8_t q = q8_1_quant(xi, d, amax);
    const long long ib = i / 32, iqs = i % 32;
    y[ib].qs[iqs] = q;
    if (iqs == 0) y[ib].ds = q8_1_ds(d, sum);
}

__global__ void quantize_q8_1_kernel(const float* __restrict__ x, block_q8_1* __restrict__ y, long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    q8_1_store(x[i], y, i);
}

// swiglu_entries_kernel and quantize_q8_1_kernel in one pass, over the call's own entries only:
// [grp_start[0], grp_start[*n_groups]) (a call's entries are contiguous; the verify window's PCIe call starts after
// its VRAM call) instead of all cap_entries twice - nothing reads the others: the down kernel reads its groups'.
// n_ff is a multiple of 32, so a warp is one q8_1 block and the bounds are warp-uniform.  The values are the two
// kernels': the SwiGLU product is rounded on its own - it must not contract into the first add of q8_1_store's block
// sum, as it could not when it went through memory (CUDA: __fmul_rn; HIP's __fmul_rn is a plain product, so there
// the product is written here under contract(off)) - then q8_1_store unchanged: the same blocks, bit for bit.
__global__ void __launch_bounds__(256) swiglu_q8_1_entries_kernel(const float* __restrict__ gate,
                                                                  const float* __restrict__ up,
                                                                  const int32_t* __restrict__ grp_start,
                                                                  const int32_t* __restrict__ n_groups, int n_ff,
                                                                  block_q8_1* __restrict__ hq) {
#if defined(__HIPCC__)
#pragma clang fp contract(off)
#endif
    const long long lo = (long long) grp_start[0] * n_ff, hi = (long long) grp_start[*n_groups] * n_ff;
    for (long long i = lo + (long long) blockIdx.x * blockDim.x + threadIdx.x; i < hi;
         i += (long long) gridDim.x * blockDim.x) {
        const float g = gate[i];
#if defined(__HIPCC__)
        const float h = (g / (1.0f + __expf(-g))) * up[i];
#else
        const float h = __fmul_rn(g / (1.0f + __expf(-g)), up[i]);
#endif
        q8_1_store(h, hq, i);
    }
}

// 147-verify (STRATA_GU_SWIGLU=1): native_gu_multi_kernel's 16-lane gate and up rows, SwiGLU and the q8_1 blocks in ONE
// launch.  A block owns 32 consecutive hidden columns r of a group: its 32 sub-warps each compute gate row r and up row
// r (the same row_dot_80_sub16 calls, entries GRP_NC at a time, the same NVFP4 tail scale), so the product
// h = silu(gate) * up is formed in the block and warp k quantizes entry k's 32 values with q8_1_store - the q8_1 block
// of those columns.  Every value is the separate kernels': the dots are the same calls, gate and up are the same floats
// (written to memory before), the product is rounded on its own as in swiglu_q8_1_entries_kernel, and q8_1_store is the
// same.  It saves the gate/up round trip through memory and one launch per layer.
template<int TG, bool STAGE_GRID>
__global__ void __launch_bounds__(512) native_gu_swiglu_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                               const int32_t* __restrict__ grp_start,
                                                               const int32_t* __restrict__ n_groups,
                                                               const int32_t* __restrict__ ent_tok,
                                                               const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                               block_q8_1* __restrict__ hq) {
#if defined(__HIPCC__)
#pragma clang fp contract(off)
#endif
    const int ng = *n_groups;
    if ((int) blockIdx.y >= ng) return;
    STRATA_SHARED_ALIGN16_U32(s_grid_buf, IqGridWords<TG, STAGE_GRID>::value);
    const uint32_t* s_grid = stage_iq_grid<TG, STAGE_GRID>(s_grid_buf, threadIdx.x, 512);
    __shared__ float sh[GRP_NC][32];
    const int sw = threadIdx.x >> 4, t = threadIdx.x & 15, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * 32 + sw;                       // the hidden column (gate row r, up row r)
    const size_t g_off = (size_t) r * L.gu_row, u_off = L.up_off + (size_t) r * L.gu_row;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {
        const uint8_t* blob = (const uint8_t*) grp_ptr[g];
        const float sg = L.tail_off ? ((const float*) (blob + L.tail_off))[0] : 1.0f;   // NVFP4: s_gate / s_up
        const float su = L.tail_off ? ((const float*) (blob + L.tail_off))[1] : 1.0f;
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        for (int e = e0; e < e1; e += GRP_NC) {
            const int n = min(GRP_NC, e1 - e);
            float gv[GRP_NC], uv[GRP_NC];
            if (n == 1) {
                const int off1[1] = { ent_tok[e] * 80 };
                float a1[1], b1[1];
                row_dot2_80_sub16<TG, 1, true, STAGE_GRID>(blob + g_off, blob + u_off, xq, off1, 1, t, a1, b1, s_grid);
                gv[0] = a1[0]; uv[0] = b1[0];
            } else if (n == 2) {
                const int off2[2] = { ent_tok[e] * 80, ent_tok[e + 1] * 80 };
                float a2[2], b2[2];
                row_dot2_80_sub16<TG, 2, true, STAGE_GRID>(blob + g_off, blob + u_off, xq, off2, 2, t, a2, b2, s_grid);
#pragma unroll
                for (int c = 0; c < 2; ++c) { gv[c] = a2[c]; uv[c] = b2[c]; }
            } else if (n == 3) {
                const int off3[3] = { ent_tok[e] * 80, ent_tok[e + 1] * 80, ent_tok[e + 2] * 80 };
                float a3[3], b3[3];
                row_dot2_80_sub16<TG, 3, true, STAGE_GRID>(blob + g_off, blob + u_off, xq, off3, 3, t, a3, b3, s_grid);
#pragma unroll
                for (int c = 0; c < 3; ++c) { gv[c] = a3[c]; uv[c] = b3[c]; }
            } else {
                const int off4[4] = { ent_tok[e] * 80, ent_tok[e + 1] * 80, ent_tok[e + 2] * 80, ent_tok[e + 3] * 80 };
                float a4[4], b4[4];
                row_dot2_80_sub16<TG, 4, true, STAGE_GRID>(blob + g_off, blob + u_off, xq, off4, 4, t, a4, b4, s_grid);
#pragma unroll
                for (int c = 0; c < 4; ++c) { gv[c] = a4[c]; uv[c] = b4[c]; }
            }
            // sub-lane c holds entry c's sums (as the separate kernel's store): h of column r, entry c
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) {
                if (c < n && t == c) {
                    const float gg = gv[c] * sg, uu = uv[c] * su;
#if defined(__HIPCC__)
                    sh[c][sw] = (gg / (1.0f + __expf(-gg))) * uu;
#else
                    sh[c][sw] = __fmul_rn(gg / (1.0f + __expf(-gg)), uu);
#endif
                }
            }
            __syncthreads();
            if (warp < n) q8_1_store(sh[warp][lane], hq, (long long) (e + warp) * L.n_ff + (long long) blockIdx.x * 32 + lane);
            __syncthreads();
        }
    }
}

// ---------------------------------------------------------------- dequant (dequantize.cuh)
template<typename dst_t> __device__ __forceinline__ dst_t cvt(float v);
template<> __device__ __forceinline__ float cvt<float>(float v) { return v; }
template<> __device__ __forceinline__ __half cvt<__half>(float v) { return __float2half(v); }

template<typename dst_t>
__device__ void dq_iq2_xxs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq2_xxs* x = (const block_iq2_xxs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint8_t* aux8 = (const uint8_t*) q2;
    const uint8_t* grid = (const uint8_t*) (iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}
template<typename dst_t>
__device__ void dq_iq2_xs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq2_xs* x = (const block_iq2_xs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint8_t* grid = (const uint8_t*) (iq2xs_grid + (q2[il] & 511));
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}
template<typename dst_t>
__device__ void dq_iq2_s(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq2_s* x = (const block_iq2_s*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* grid = (const uint8_t*) (iq2s_grid + (x[ibs].qs[4 * ib + il] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 0x300)));
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[ibs].qs[QK_K / 8 + 4 * ib + il];
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}
template<typename dst_t>
__device__ void dq_iq3_xxs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq3_xxs* x = (const block_iq3_xxs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* q3 = x[ibs].qs + 8 * ib;
    const uint16_t* gas = (const uint16_t*) (x[ibs].qs + QK_K / 4) + 2 * ib;
    const uint8_t* grid1 = (const uint8_t*) (iq3xxs_grid + q3[2 * il + 0]);
    const uint8_t* grid2 = (const uint8_t*) (iq3xxs_grid + q3[2 * il + 1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
        y[j + 4] = cvt<dst_t>(d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
    }
}
template<typename dst_t>
__device__ void dq_iq3_s(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq3_s* x = (const block_iq3_s*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* qs = x[ibs].qs + 8 * ib;
    const uint8_t* grid1 = (const uint8_t*) (iq3s_grid + (qs[2 * il + 0] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 256)));
    const uint8_t* grid2 = (const uint8_t*) (iq3s_grid + (qs[2 * il + 1] | ((x[ibs].qh[ib] << (7 - 2 * il)) & 256)));
    const float d = (float) x[ibs].d * (1 + 2 * ((x[ibs].scales[ib / 2] >> 4 * (ib % 2)) & 0xf));
    const uint8_t signs = x[ibs].signs[4 * ib + il];
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
        y[j + 4] = cvt<dst_t>(d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
    }
}
template<typename dst_t>
__device__ void dq_iq1_m(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq1_m* x = (const block_iq1_m*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* sc = (const uint16_t*) x[ibs].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int64_t ib16 = 2 * ib + il / 2;
    const float d = (float) scale.f16 * (2 * ((sc[ib16 / 4] >> 3 * (ib16 % 4)) & 0x7) + 1);
    const float delta = x[ibs].qh[2 * ib + il / 2] & (0x08 << 4 * (il % 2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2];
    const int8_t* q = (const int8_t*) grid32;
    grid32[0] = iq1s_grid_gpu[x[ibs].qs[4 * ib + il] | (((x[ibs].qh[2 * ib + il / 2] >> 4 * (il % 2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * (q[j] + delta));
}
template<typename dst_t>
__device__ void dq_iq4_nl(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq4_nl* x = (const block_iq4_nl*) vx + ibs * (QK_K / QK4_NL);
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 4 * il;
    const uint8_t* q4 = x[ib].qs + 4 * il;
    const float d = (float) x[ib].d;
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j + 16] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] >> 4]);
    }
}
// Q3_K (the Q2_0 file's token_embd): llama.cpp's dequantize_block_q3_K, its 64 threads folded onto 32
template<typename dst_t>
__device__ void dq_q3_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q3_K* x = (const block_q3_K*) vx + ibs;
    for (int tt = tid; tt < 64; tt += 32) {
        const int r = tt / 4, t2 = r / 2, is0 = r % 2;
        const int l0 = 16 * is0 + 4 * (tt % 4);
        const int n = t2 / 4, j = t2 - 4 * n;
        const uint8_t m = (uint8_t) (1 << (4 * n + j));
        const int is = 8 * n + 2 * j + is0;
        const int shift = 2 * j;
        const int8_t us = is < 4  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 8] >> 0) & 3) << 4)) :
                          is < 8  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 4] >> 2) & 3) << 4)) :
                          is < 12 ? (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is + 0] >> 4) & 3) << 4)) :
                                    (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is - 4] >> 6) & 3) << 4));
        const float dl = (float) x->d * (us - 32);
        dst_t* y = yy + 128 * n + 32 * j;
        const uint8_t* q = x->qs + 32 * n;
        const uint8_t* hm = x->hmask;
        for (int l = l0; l < l0 + 4; ++l) y[l] = cvt<dst_t>(dl * ((int8_t) ((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4)));
    }
}
template<typename dst_t>
__device__ void dq_iq4_xs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq4_xs* x = (const block_iq4_xs*) vx + ibs;
    const int il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 4 * il;
    const uint8_t* q4 = x->qs + 16 * ib + 4 * il;
    const float d = (float) x->d * ((((x->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x->scales_h >> 2 * ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j + 16] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] >> 4]);
    }
}
template<typename dst_t>
__device__ void dq_q2_0(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    // one "superblock" = 256 values = 4 blocks of 64; thread tid writes 8 values
    const block_q2_0* x = (const block_q2_0*) vx + ibs * 4;
    const int b = tid / 8, part = tid % 8;          // block 0..3, 8 values each
    const float d = (float) x[b].d;
    for (int j = 0; j < 8; ++j) {
        const int i = part * 8 + j;
        const int code = (x[b].qs[i / 4] >> ((i % 4) * 2)) & 3;
#if defined(__HIPCC__)
        // HIP compilers fold the half path's negative scale times +0 to +0 (HIP 5.7 on gfx1012, gfx1151 on ROCm
        // 7.13 - PR #895, gfx1200/gfx1201 on ROCm 7.10 - #1474); the gating arch list only ever chased the newest
        // report (#1540).  Preserve the scale's sign, as the FP32/CPU paths do: value-identical where the fold never happens.
        if constexpr (std::is_same_v<dst_t, __half>) {
            if (code == 1) {
                yy[b * 64 + i] = __ushort_as_half(__half_as_ushort(x[b].d) & 0x8000u);
                continue;
            }
        }
#endif
        yy[b * 64 + i] = cvt<dst_t>(d * (float) (code - 1));
    }
}

// llama.cpp's dequantize_q4_K / dequantize_q5_K (dequantize.cuh; q5_K's 64 threads folded onto 32) and the 32-value
// blocks of Q5_1 / Q8_0, 8 of them per 256-value "superblock" (thread tid writes 8 values of block tid % 8).
__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}
template<typename dst_t>
__device__ void dq_q4_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q4_K* x = (const block_q4_K*) vx;
    const int64_t il = tid / 8, ir = tid % 8, is = 2 * il;
    const int n = 4;
    dst_t* y = yy + 64 * il + n * ir;
    const float dall = __low2half(x[ibs].dm);
    const float dmin = __high2half(x[ibs].dm);
    const uint8_t* q = x[ibs].qs + 32 * il + n * ir;
    uint8_t sc, m;
    get_scale_min_k4((int) is + 0, x[ibs].scales, sc, m);
    const float d1 = dall * sc, m1 = dmin * m;
    get_scale_min_k4((int) is + 1, x[ibs].scales, sc, m);
    const float d2 = dall * sc, m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = cvt<dst_t>(d1 * (q[l] & 0xF) - m1);
        y[l + 32] = cvt<dst_t>(d2 * (q[l] >> 4) - m2);
    }
}
template<typename dst_t>
__device__ void dq_q5_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q5_K* x = (const block_q5_K*) vx;
    for (int tt = tid; tt < 64; tt += 32) {
        const int il = tt / 16, ir = tt % 16, is = 2 * il;
        dst_t* y = yy + 64 * il + 2 * ir;
        const float dall = __low2half(x[ibs].dm);
        const float dmin = __high2half(x[ibs].dm);
        const uint8_t* ql = x[ibs].qs + 32 * il + 2 * ir;
        const uint8_t* qh = x[ibs].qh + 2 * ir;
        uint8_t sc, m;
        get_scale_min_k4(is + 0, x[ibs].scales, sc, m);
        const float d1 = dall * sc, m1 = dmin * m;
        get_scale_min_k4(is + 1, x[ibs].scales, sc, m);
        const float d2 = dall * sc, m2 = dmin * m;
        uint8_t hm = (uint8_t) (1 << (2 * il));
        y[0] = cvt<dst_t>(d1 * ((ql[0] & 0xF) + (qh[0] & hm ? 16 : 0)) - m1);
        y[1] = cvt<dst_t>(d1 * ((ql[1] & 0xF) + (qh[1] & hm ? 16 : 0)) - m1);
        hm <<= 1;
        y[32] = cvt<dst_t>(d2 * ((ql[0] >> 4) + (qh[0] & hm ? 16 : 0)) - m2);
        y[33] = cvt<dst_t>(d2 * ((ql[1] >> 4) + (qh[1] & hm ? 16 : 0)) - m2);
    }
}
template<typename dst_t>
__device__ void dq_q5_0(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    // symmetric: one delta (no min), the high bit folded in then centered by the format's fixed -16 offset
    const block_q5_0* x = (const block_q5_0*) vx + ibs * (QK_K / QK5_0);
    const int ib = tid % 8, il = tid / 8;
    const float d = __half2float(x[ib].d);
    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));
    dst_t* y = yy + 32 * ib;
    for (int j = 0; j < 4; ++j) {
        const int iqs = 4 * il + j;
        const int xh_0 = ((qh >> (iqs + 0)) << 4) & 0x10;
        const int xh_1 = ((qh >> (iqs + 12))) & 0x10;
        y[iqs] = cvt<dst_t>(((float) ((x[ib].qs[iqs] & 0xf) | xh_0) - 16.0f) * d);
        y[iqs + 16] = cvt<dst_t>(((float) ((x[ib].qs[iqs] >> 4) | xh_1) - 16.0f) * d);
    }
}
template<typename dst_t>
__device__ void dq_q4_0(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q4_0* x = (const block_q4_0*) vx + ibs * (QK_K / QK4_0);
    const int ib = tid % 8, il = tid / 8;
    const float d = __half2float(x[ib].d);
    dst_t* y = yy + 32 * ib;
    for (int j = 0; j < 4; ++j) {
        const int iqs = 4 * il + j;                   // llama.cpp's dequantize_q4_0 for value pairs iqs, iqs + 16
        y[iqs] = cvt<dst_t>(((float) (x[ib].qs[iqs] & 0xf) - 8.0f) * d);
        y[iqs + 16] = cvt<dst_t>(((float) (x[ib].qs[iqs] >> 4) - 8.0f) * d);
    }
}
template<typename dst_t>
__device__ void dq_q4_1(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q4_1* x = (const block_q4_1*) vx + ibs * (QK_K / QK4_1);
    const int ib = tid % 8, il = tid / 8;
    const float2 dm = __half22float2(x[ib].dm);
    dst_t* y = yy + 32 * ib;
    for (int j = 0; j < 4; ++j) {
        const int iqs = 4 * il + j;                   // llama.cpp's dequantize_q4_1 for value pairs iqs, iqs + 16
        y[iqs] = cvt<dst_t>((float) (x[ib].qs[iqs] & 0xf) * dm.x + dm.y);
        y[iqs + 16] = cvt<dst_t>((float) (x[ib].qs[iqs] >> 4) * dm.x + dm.y);
    }
}
template<typename dst_t>
__device__ void dq_q5_1(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q5_1* x = (const block_q5_1*) vx + ibs * (QK_K / QK5_1);
    const int ib = tid % 8, il = tid / 8;
    const float2 dm = __half22float2(x[ib].dm);
    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));
    dst_t* y = yy + 32 * ib;
    for (int j = 0; j < 4; ++j) {
        const int iqs = 4 * il + j;                   // llama.cpp's dequantize_q5_1 for value pairs iqs, iqs + 16
        const int xh_0 = ((qh >> (iqs + 0)) << 4) & 0x10;
        const int xh_1 = ((qh >> (iqs + 12))) & 0x10;
        y[iqs] = cvt<dst_t>((float) ((x[ib].qs[iqs] & 0xf) | xh_0) * dm.x + dm.y);
        y[iqs + 16] = cvt<dst_t>((float) ((x[ib].qs[iqs] >> 4) | xh_1) * dm.x + dm.y);
    }
}
template<typename dst_t>
__device__ void dq_q6_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    // llama.cpp's dequantize_q6_K (dequantize.cuh, written for 64 threads) folded onto the kernel's 32: each of the
    // 64 thread roles runs at tid and tid + 32.
    const block_q6_K* x = (const block_q6_K*) vx;
    for (int tt = tid; tt < 64; tt += 32) {
        const int64_t ip = tt / 32;                 // 0 or 1
        const int64_t il = tt - 32 * ip;            // 0...31
        const int64_t is = 8 * ip + il / 16;
        dst_t* y = yy + 128 * ip + il;
        const float d = __half2float(x[ibs].d);
        const uint8_t* ql = x[ibs].ql + 64 * ip + il;
        const uint8_t qh = x[ibs].qh[32 * ip + il];
        const int8_t* sc = x[ibs].scales + is;
        y[0]  = cvt<dst_t>(d * sc[0] * ((int8_t) ((ql[0]  & 0xF) | (((qh >> 0) & 3) << 4)) - 32));
        y[32] = cvt<dst_t>(d * sc[2] * ((int8_t) ((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32));
        y[64] = cvt<dst_t>(d * sc[4] * ((int8_t) ((ql[0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32));
        y[96] = cvt<dst_t>(d * sc[6] * ((int8_t) ((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32));
    }
}
template<typename dst_t>
__device__ void dq_q8_0(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q8_0* x = (const block_q8_0*) vx + ibs * (QK_K / QK8_0);
    const int ib = tid % 8, il = tid / 8;
    const float d = __half2float(x[ib].d);
    dst_t* y = yy + 32 * ib + 8 * il;
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>((float) x[ib].qs[8 * il + j] * d);
}

// BF16 (the token embedding as the checkpoint ships it, tools/embd_bf16_pack.py): 8 values per thread.
template<typename dst_t>
__device__ void dq_bf16(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const uint16_t* x = (const uint16_t*) vx + ibs * 256 + tid * 8;
    for (int j = 0; j < 8; ++j) yy[tid * 8 + j] = cvt<dst_t>(__uint_as_float((uint32_t) x[j] << 16));
}

// NVFP4: one "superblock" = 256 values = 4 blocks of 64; thread tid writes 8 values of block tid / 8.
// Sub-block s (16 values) keeps its codes in qs[8 s .. 8 s + 7]: byte j = value j (low nibble) and j + 8 (high).
template<typename dst_t>
__device__ void dq_nvfp4(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_nvfp4* x = (const block_nvfp4*) vx + ibs * 4;
    const int b = tid / 8, part = tid % 8;
    for (int j = 0; j < 8; ++j) {
        const int i = part * 8 + j, s = i / 16, k = i % 16;
        const uint8_t byte = x[b].qs[s * 8 + (k & 7)];
        const int code = k < 8 ? (byte & 0x0F) : (byte >> 4);
        yy[b * 64 + i] = cvt<dst_t>((float) kvalues_fp4[code] * ue4m3_to_f32_half(x[b].d[s]));
    }
}

// APR trellis types: one 256-value call = two 128-blocks; thread tid decodes block tid/16, lane (tid/2)%8, steps
// 2(tid%2)..2(tid%2)+1 (8 values)
template<int TY, typename dst_t>
__device__ void dq_tq(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const int blk = tid >> 4, l = (tid >> 1) & 7, half = tid & 1;
    const auto* b = (const typename TqInfo<TY>::block_t*) vx + 2 * ibs + blk;
    const float d = tq_h2f(b->d);
    uint32_t s[4];
    tq_states4<TY>(b->qs, l, s);
    dst_t* y = yy + blk * 128 + 16 * l + 8 * half;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        float2 a, c;
        tq_step(s[2 * half + i], a, c);
        y[4 * i + 0] = cvt<dst_t>(d * a.x);
        y[4 * i + 1] = cvt<dst_t>(d * a.y);
        y[4 * i + 2] = cvt<dst_t>(d * c.x);
        y[4 * i + 3] = cvt<dst_t>(d * c.y);
    }
}

// TQP (Gyro-L): pair trellis; the step's 2 weights are the first pair of TQ2_T state s.
__device__ __forceinline__ float2 tqp_pair(uint32_t s) {
    const uint32_t* lut = (const uint32_t*) tq_lut_f16;
    const uint32_t x = (s & TQ_STATE_MASK) * 0x9e3779b1u;
    const uint32_t pa = lut[x >> 21];
    return make_float2(tq_h2f(pa & 0xFFFFu), tq_h2f(pa >> 16));
}
// one 256-value call = two 128-blocks; thread tid decodes block tid/16, lane (tid>>1)&7, steps
// 8*l + 4*half .. +3 (8 values, 2 per step)
template<int TY, typename dst_t>
__device__ void dq_tqp(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const int blk = tid >> 4, l = (tid >> 1) & 7, half = tid & 1;
    const auto* b = (const typename TqInfo<TY>::block_t*) vx + 2 * ibs + blk;
    const float d = tq_h2f(b->d);
    uint32_t s[8];
    tqp_states8_cheap<TY>(b->qs, l, s);
    dst_t* y = yy + blk * 128 + 16 * l + 8 * half;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 a = tqp_pair(s[4 * half + i]);
        y[2 * i + 0] = cvt<dst_t>(d * a.x);
        y[2 * i + 1] = cvt<dst_t>(d * a.y);
    }
}

// Every type below must also be in is_iq() (BF16: embed_type_supported): the host entry points refuse the others,
// so the default is unreachable.
template<typename dst_t>
__device__ __forceinline__ void dq_dispatch(int ty, const void* vx, int64_t ibs, dst_t* y, int tid) {
    switch (ty) {
        case 16: dq_iq2_xxs(vx, ibs, y, tid); break;
        case 17: dq_iq2_xs(vx, ibs, y, tid); break;
        case 18: dq_iq3_xxs(vx, ibs, y, tid); break;
        case 20: dq_iq4_nl(vx, ibs, y, tid); break;
        case 21: dq_iq3_s(vx, ibs, y, tid); break;
        case 22: dq_iq2_s(vx, ibs, y, tid); break;
        case 29: dq_iq1_m(vx, ibs, y, tid); break;
        case 23: dq_iq4_xs(vx, ibs, y, tid); break;
        case 11: dq_q3_k(vx, ibs, y, tid); break;
        case 42: dq_q2_0(vx, ibs, y, tid); break;
        case 12: dq_q4_k(vx, ibs, y, tid); break;
        case 13: dq_q5_k(vx, ibs, y, tid); break;
        case 14: dq_q6_k(vx, ibs, y, tid); break;   // the Q6K_EXPERTS kernels, and the token embedding (Gyro-S)
        case 7: dq_q5_1(vx, ibs, y, tid); break;
        case 6: dq_q5_0(vx, ibs, y, tid); break;
        case 2: dq_q4_0(vx, ibs, y, tid); break;
        case 3: dq_q4_1(vx, ibs, y, tid); break;
        case 40: dq_nvfp4(vx, ibs, y, tid); break;
        case 8: dq_q8_0(vx, ibs, y, tid); break;
        case 30: dq_bf16(vx, ibs, y, tid); break;
        case 144: dq_tq<144>(vx, ibs, y, tid); break;
        case 145: dq_tq<145>(vx, ibs, y, tid); break;
        case 146: dq_tq<146>(vx, ibs, y, tid); break;
        case 147: dq_tqp<147>(vx, ibs, y, tid); break;
        case 148: dq_tqp<148>(vx, ibs, y, tid); break;
        case 149: dq_tqp<149>(vx, ibs, y, tid); break;
        default: break;
    }
}

// flat: superblock i -> y + 256 i
template<typename dst_t>
__global__ void dequant_flat_kernel(int ty, const void* __restrict__ vx, dst_t* __restrict__ y,
                                    const float* __restrict__ scale) {
    const int64_t i = blockIdx.x;
    dq_dispatch<dst_t>(ty, vx, i, y + i * QK_K, threadIdx.x);
    if (scale != nullptr) {       // NVFP4 down: s_down (the weights stay in FP16's normal range)
        __syncwarp();
        const float sc = *scale;
        for (int k = 0; k < 8; ++k) {
            const int j = threadIdx.x * 8 + k;
            y[i * QK_K + j] = cvt<dst_t>((float) y[i * QK_K + j] * sc);
        }
    }
}
// gate/up: superblock i of a role matrix (n_embd/256 per row) -> interleaved row 2r + parity
__global__ void dequant_gu_kernel(int ty, const void* __restrict__ gate, const void* __restrict__ up, int64_t per_row,
                                  __half* __restrict__ y, const float* __restrict__ tail) {
    const int64_t i = blockIdx.x;
    const int parity = blockIdx.y;
    const int64_t r = i / per_row, c = i % per_row;
    __half* dst = y + ((2 * r + parity) * per_row + c) * QK_K;
    dq_dispatch<__half>(ty, parity ? up : gate, i, dst, threadIdx.x);
    if (tail != nullptr) {        // NVFP4: gate * s_gate, up * s_up (s_down on the down output: FP16 would underflow)
        __syncwarp();
        const float sc = parity ? tail[1] : tail[0];
        for (int k = 0; k < 8; ++k) {
            const int j = threadIdx.x * 8 + k;
            dst[j] = __float2half(__half2float(dst[j]) * sc);
        }
    }
}

// the types dq_dispatch dequantizes
bool is_iq(int t) {
    return t == 16 || t == 17 || t == 18 || t == 20 || t == 21 || t == 22 || t == 23 || t == 29 || t == 42 || t == 11 ||
#ifdef STRATA_Q6K_EXPERTS
           t == 12 || t == 13 || t == 14 || t == 7 || t == 6 || t == 2 || t == 3 || t == 8 || t == 40 || t == 144 || t == 145 || t == 146 || t == 147 || t == 148 || t == 149;
#else
           t == 12 || t == 13 || t == 7 || t == 6 || t == 2 || t == 3 || t == 8 || t == 40 || t == 144 || t == 145 || t == 146 || t == 147 || t == 148 || t == 149;
#endif
}
// values per block of the types the grouped expert kernels take (0 = none)
int gu_qk(int t) {
    switch (t) {
#define STRATA_QK(T) case T: return Fmt<T>::qk;
        STRATA_GU_FMTS(STRATA_QK)
#undef STRATA_QK
        default: return 0;
    }
}
int d_qk(int t) {
    switch (t) {
#define STRATA_QK(T) case T: return Fmt<T>::qk;
        STRATA_D_FMTS(STRATA_QK)
#undef STRATA_QK
        default: return 0;
    }
}
bool gu_split(int t) {
    switch (t) {
#define STRATA_SP(T) case T: return kSplit<T> && !split_off<T>();
        STRATA_GU_FMTS(STRATA_SP)
#undef STRATA_SP
        default: return false;
    }
}

bool env_on(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}
// STRATA_OLD_IQ_MMVQ=1 keeps the per-column kernels (bitwise equal to the new ones; kept for A/B timing)
bool g_old_kernels = env_on("STRATA_OLD_IQ_MMVQ");
bool g_no_sub16_gu = env_on("STRATA_NO_SUB16_GU");
// STRATA_IQ_STAGE_GRID=0 disables staging 64-bit i-quant codebook tables into shared memory
bool g_stage_grid = [] {
    const char* v = std::getenv("STRATA_IQ_STAGE_GRID");
    return v == nullptr || v[0] == '\0' || v[0] != '0';
}();
// The single-matrix mmvq (128-thread blocks, one per 4 rows) pays the 2-8 KB staging per block: on the RTX 5070 its
// IQ2_XS / IQ2_S / IQ1_M calls were 10-20% slower staged at 3-8 columns (iq_multi_parity --bench), while the grouped
// expert kernels gain.  So mmvq stages only with STRATA_IQ_STAGE_GRID_MMVQ=1 (bitwise the same either way).
bool g_stage_grid_mmvq = g_stage_grid && [] {
    const char* v = std::getenv("STRATA_IQ_STAGE_GRID_MMVQ");
    return v != nullptr && v[0] == '1';
}();
// IQ3_XXS's and IQ3_S's codebooks and sign-mask tables staged in shared memory by the grouped expert kernels (the
// single-matrix mmvq does not): bitwise the same either way.  OPT-IN: STRATA_IQ_STAGE_TABLES=1.  The PR (#1424) made it
// the default on compute capability 6.0 (GP100: no __dp4a, these kernels decode-bound at 70-140 GB/s); on a Tesla P100
// (IQ3_XXS and IQ3_S packs, 5-6 interleaved pairs each) the decode was 0.97-1.03x of the unstaged kernels, noise, so
// it is not on by default.  On an RTX 3090 Ti, forced on, the IQ3_S layers took 3-13% longer.
bool stage_tables_on() {
    if (!g_stage_grid) return false;
    static const bool on = [] {
        const char* v = std::getenv("STRATA_IQ_STAGE_TABLES");
        return v && v[0] && v[0] != '0';
    }();
    return on;
}

// The trellis types (Gyro: TQ2_T / TQK6 / TQK7) read a 4 KB int8 codebook with 8 random lookups per 16 weights.  The
// grouped expert kernels stage it in shared memory on compute capability 6.0 (Tesla P100: no dp4a, ~28 ALU ops per
// weight in the decode, the global lookups are replayed per distinct line) and leave it in global memory elsewhere
// (the newer cards' L1 absorbs it).  STRATA_TQ_STAGE=1 / 0 forces it either way; bitwise the same result both ways.
bool tq_stage_on() {
    static const int env = [] { const char* v = std::getenv("STRATA_TQ_STAGE"); return v != nullptr && v[0] != '\0' ? (v[0] != '0' ? 1 : 0) : -1; }();
    if (env >= 0) return env == 1 && g_stage_grid;
#if defined(__HIPCC__)
    return false;
#else
    if (!g_stage_grid) return false;
    static int per_dev[64];   // 0 unknown, 1 no, 2 yes
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (!per_dev[dev]) {
        int maj = 0, mnr = 0;
        const bool ok = cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                        cudaDeviceGetAttribute(&mnr, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess;
        if (!ok) cudaGetLastError();
        per_dev[dev] = ok && maj * 10 + mnr < 61 ? 2 : 1;
    }
    return per_dev[dev] == 2;
#endif
}
// Everywhere but cc 6.0 (tq_stage_on) the trellis kernels stage the codebook too, and the 160-dot gate/up rows are 8-lane rows (4 per
// warp, 32 per block, row_dot_160_sub8) while the down rows are the staged 8-lane rows above: on an RTX 3060 (Gyro-M, 24 groups of one
// token) gate/up 221 -> 140 us and down 182 -> 86 us.  Bitwise the same sums as before.  STRATA_TQ_STAGE=0 or STRATA_TQ_SUB8=0 turn it off
// (CUDA only: the AMD builds keep the old kernels), STRATA_IQ_STAGE_GRID=0 as for every staged table.
bool tq_sub8_on() {
#if defined(__HIPCC__)
    return false;
#else
    static const bool on = [] {
        const char* v = std::getenv("STRATA_TQ_STAGE");
        if (v != nullptr && v[0] == '0') return false;
        v = std::getenv("STRATA_TQ_SUB8");
        return !(v != nullptr && v[0] == '0');
    }();
    return on && g_stage_grid && !g_no_sub16_gu && !tq_stage_on();
#endif
}
// row chunks per block of the staged trellis gate/up and down kernels (STRATA_TQ_ROWCH, default 4)
int tq_row_chunks() {
    static const int n = [] {
        const char* v = std::getenv("STRATA_TQ_ROWCH");
        const int k = v != nullptr ? std::atoi(v) : 4;
        return k < 1 ? 1 : k > 32 ? 32 : k;
    }();
    return n;
}

template<int TY, bool STAGE_GRID = kStageIqGrid<TY>>
void launch_mmvq_multi_nc(dim3 grid, dim3 block, cudaStream_t s, const uint8_t* W, size_t rb, const block_q8_1* X,
                          float* y, int n_in, int n_out, int ncols) {
    switch (ncols) {
        case 1: mmvq_multi_kernel<TY, 1, STAGE_GRID, true><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, 1); break;
        case 2: mmvq_multi_kernel<TY, 2, STAGE_GRID, true><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, 2); break;
        case 3: mmvq_multi_kernel<TY, 3, STAGE_GRID, true><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, 3); break;
        case 4: mmvq_multi_kernel<TY, 4, STAGE_GRID, true><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, 4); break;
        case 5: mmvq_multi_kernel<TY, 5, STAGE_GRID, true><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, 5); break;
        case 6: mmvq_multi_kernel<TY, 6, STAGE_GRID, true><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, 6); break;
        default:
            if (ncols <= 1) mmvq_multi_kernel<TY, 1, STAGE_GRID, false><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, ncols);
            else mmvq_multi_kernel<TY, 8, STAGE_GRID, false><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, ncols);
            break;
    }
}

template<int TY>
void launch_mmvq(const uint8_t* W, size_t rb, const block_q8_1* X, float* y, int n_in, int n_out, int ncols,
                 cudaStream_t s) {
    const dim3 grid((unsigned) ((n_out + 3) / 4)), block(32, 4);
    if constexpr (!kSplit<TY>) mmvq_kernel<TY><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, ncols);
    else if (g_old_kernels || split_off<TY>()) mmvq_kernel<TY><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, ncols);
    else if constexpr (kStageIqGrid<TY>) {
        if (!g_stage_grid_mmvq) {
            launch_mmvq_multi_nc<TY, false>(grid, block, s, W, rb, X, y, n_in, n_out, ncols);
            return;
        }
        launch_mmvq_multi_nc<TY, true>(grid, block, s, W, rb, X, y, n_in, n_out, ncols);
    } else {
        launch_mmvq_multi_nc<TY, false>(grid, block, s, W, rb, X, y, n_in, n_out, ncols);
    }
}


// ---------------------------------------------------------------- S26 STRATA_EXPERT_V2=1 (opt-in): IQ3_S gate/up + IQ4_NL down
// Bitwise equal to native_gu_multi_kernel / native_down_multi_kernel (S26 harness, real UD-IQ4_XS expert blobs, T 1-3,
// memcmp of every output): each row's lane sums and warp_sum are the same; a warp interleaves RPW rows (more loads in
// flight) and reads the chunk's activation rows from LDS and, for IQ3_S, the grid table from LDS (the same values).
#if defined(__HIPCC__)
#define STRATA_NT_LOAD(p) __builtin_nontemporal_load(p)
#else   // (opt-in S26 kernels, AMD-tuned: a CUDA build compiles them with plain loads)
#define STRATA_NT_LOAD(p) (*(p))
#endif
template<bool NT> __device__ __forceinline__ int ld16i(const uint16_t* p) {
    if constexpr (NT) return (int) STRATA_NT_LOAD(p); else return (int) *p;
}
template<bool NT> __device__ __forceinline__ int ldb2(const void* x, int i32) {
    const uint16_t* x16 = (const uint16_t*) x;
    int x32 = ld16i<NT>(x16 + 2 * i32 + 0) << 0;
    x32 |= ld16i<NT>(x16 + 2 * i32 + 1) << 16;
    return x32;
}
template<bool NT> __device__ __forceinline__ int ld8i(const uint8_t* p) {
    if constexpr (NT) return (int) STRATA_NT_LOAD(p); else return (int) *p;
}
template<bool NT> __device__ __forceinline__ float ldhalf(const half* p) {
    return __half2float(__ushort_as_half((unsigned short) ld16i<NT>(reinterpret_cast<const uint16_t*>(p))));
}
template<int TY, bool NT> struct S26Split;
template<bool NT> struct S26Split<21, NT> {   // IQ3_S: Split<21>::load with the loads made explicit
    using W = Split<21>::W;
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
        const int2 qs_packed = make_int2(ldb2<NT>(bq3->qs, iqs + 0), ldb2<NT>(bq3->qs, iqs + 1));
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = ld8i<NT>(bq3->qh + iqs / 2);
        const int signs_packed_32 = ldb2<NT>(bq3->signs, iqs / 2);
        const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int2 grid_pos = make_int2(iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                                            iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
            const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
            r.g[l0 + 0] = __vsub4(grid_pos.x ^ signs0, signs0);
            r.g[l0 + 1] = __vsub4(grid_pos.y ^ signs1, signs1);
        }
        r.ls = 1 + 2 * ((ld8i<NT>(bq3->scales + iqs / 4) >> ((iqs << 1) & 0x04)) & 0x0F);
        r.dw = ldhalf<NT>(&bq3->d);
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ b, int iqs) { return Split<21>::apply(r, b, iqs); }
};
template<bool NT> struct S26Split<20, NT> {   // IQ4_NL
    using W = Split<20>::W;
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_iq4_nl* bq4 = (const block_iq4_nl*) vbq + kbx;
        W r;
#pragma unroll
        for (int l = 0; l < 2; ++l) r.v[l] = get_int_from_table_16(ldb2<NT>(bq4->qs, iqs + l), kvalues_iq4nl);
        r.dw = ldhalf<NT>(&bq4->d);
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ b, int iqs) { return Split<20>::apply(r, b, iqs); }
};

// S26 v13+: v6 (4 interleaved rows per warp) with the IQ3_S grid table (LT) and / or the chunk's activation rows (LX)
// read from LDS: the same table values and the same q8_1 bytes, so the same arithmetic and bits.
struct S26IQ3S {   // Split<21>::load with the grid from `grid` (LT) or iq3s_grid itself
    using W = Split<21>::W;
    template<bool LT = true>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* grid) {
        const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
        const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq3->qh[iqs / 2];
        const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
        const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int i0 = qs[l0 + 0] | ((qh << (8 - l0)) & 0x100), i1 = qs[l0 + 1] | ((qh << (7 - l0)) & 0x100);
            const int2 grid_pos = LT ? make_int2(grid[i0], grid[i1]) : make_int2(iq3s_grid[i0], iq3s_grid[i1]);
            const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
            r.g[l0 + 0] = __vsub4(grid_pos.x ^ signs0, signs0);
            r.g[l0 + 1] = __vsub4(grid_pos.y ^ signs1, signs1);
        }
        r.ls = 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
        r.dw = __half2float(bq3->d);
        return r;
    }
};

constexpr int S26_XMAX = 2560 / 32;   // q8_1 blocks of one activation row (n_embd 2560)

// S26 BAL: step I of a warp's RPW x NI items (item p = 32 I + lane): rows QA / QB (lanes >= THR take QB)
template<int I, int NI, int RPW> struct S26Bal {
    static constexpr int P0 = 32 * I, QA = P0 / NI, QB0 = (P0 + 31) / NI, QB = QB0 < RPW ? QB0 : RPW - 1;
    static constexpr int THR = (QA + 1) * NI - P0;
    static_assert(NI >= 32, "a step spans at most two rows");
};

template<bool LT, bool LX, int RPW, bool SL = false, bool TS = false>
__global__ void __launch_bounds__(256) s26_gu_l_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                       const int32_t* __restrict__ grp_start,
                                                       const int32_t* __restrict__ n_groups,
                                                       const int32_t* __restrict__ ent_tok,
                                                       const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                       float* __restrict__ gate, float* __restrict__ up) {
    __shared__ uint32_t s_grid[LT ? 512 : 1];
    __shared__ block_q8_1 s_x[LX ? GRP_NC * S26_XMAX : 1];
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    if constexpr (LT)
        for (int i = threadIdx.x; i < 512; i += 256) s_grid[i] = iq3s_grid[i];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = blockIdx.x * GU_ROWS * RPW + warp;
    const bool live = row0 + GU_ROWS * (RPW - 1) < 2 * L.n_ff;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint8_t* wr[RPW];
    int rr[RPW];
    bool upq[RPW];
#pragma unroll
    for (int q = 0; q < RPW; ++q) {
        const int row = row0 + q * GU_ROWS;
        upq[q] = row >= L.n_ff;
        rr[q] = upq[q] ? row - (int) L.n_ff : row;
        wr[q] = blob + (upq[q] ? L.up_off : 0) + (size_t) rr[q] * L.gu_row;
    }
    const uint32_t* grid = LT ? s_grid : reinterpret_cast<const uint32_t*>(iq3s_grid);
    const int nb = (int) (L.n_embd / 256), xb = (int) (L.n_embd / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; e += GRP_NC) {
        const int n = min(GRP_NC, e1 - e);
        const block_q8_1* xbase = xq;
        int off[GRP_NC];
        if constexpr (LX) {
            __syncthreads();   // the previous chunk is done with s_x
            for (int i = threadIdx.x; i < n * xb; i += 256) {
                const int c = i / xb, j = i - c * xb;
                memcpy(&s_x[c * S26_XMAX + j], &xq[(size_t) ent_tok[e + c] * xb + j], sizeof(s_x[0]));
            }
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) off[c] = min(c, n - 1) * S26_XMAX;
            xbase = s_x;
        } else {
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) off[c] = ent_tok[e + min(c, n - 1)] * xb;
        }
        if constexpr (LT || LX) __syncthreads();   // s_grid / s_x ready
        if (!live) continue;
        float s[RPW][GRP_NC];
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) s[q][c] = 0.0f;
        for (int k = lane; k < nb * 8; k += 32) {
            const int kbx = k / 8, iqs = 2 * (k % 8);
            Split<21>::W w[RPW];
#pragma unroll
            for (int q = 0; q < RPW; ++q) { if constexpr (SL && !LT) w[q] = S26Split<21, false>::load(wr[q], kbx, iqs); else w[q] = S26IQ3S::load<LT>(wr[q], kbx, iqs, grid); }
#pragma unroll
            for (int q = 0; q < RPW; ++q)
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c)
                    if (c < n) s[q][c] += Split<21>::apply(w[q], xbase + off[c] + kbx * 8, iqs);
        }
        if constexpr (TS) {   // S26: all RPW x GRP_NC sums in one transposed butterfly (bitwise the same sums)
            constexpr int P = s26ts::pow2_ceil(RPW * GRP_NC);
            float v[P];
#pragma unroll
            for (int j = 0; j < P; ++j) v[j] = j < RPW * GRP_NC ? s[j / GRP_NC][j % GRP_NC] : 0.0f;
            const float sum = s26ts::tsum<P>(v, lane);
            const int j = s26ts::tsum_token<P>(lane);
            if (lane == s26ts::tsum_lane<P>(j) && j < RPW * GRP_NC) {
                const int c = j % GRP_NC;
#pragma unroll
                for (int q = 0; q < RPW; ++q)
                    if (j / GRP_NC == q && c < n) (upq[q] ? up : gate)[(size_t) (e + c) * L.n_ff + rr[q]] = sum;
            }
            continue;
        }
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                if (c < n) s[q][c] = warp_sum(s[q][c]);
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                if (c < n && lane == c) (upq[q] ? up : gate)[(size_t) (e + c) * L.n_ff + rr[q]] = s[q][c];
    }
}

constexpr int S26_HMAX = 640 / 32;    // q8_1 blocks of one SwiGLU row (n_ff 640)

template<bool LX, int RPW, bool SL = false, bool TS = false, int BAL = 0>
__global__ void __launch_bounds__(256) s26_down_l_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                         const int32_t* __restrict__ grp_start,
                                                         const int32_t* __restrict__ n_groups,
                                                         const int32_t* __restrict__ ent_dst,
                                                         const block_q8_1* __restrict__ hq, NativeExpertLayout L,
                                                         float* __restrict__ out) {
    __shared__ block_q8_1 s_h[LX ? GRP_NC * S26_HMAX : 1];
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = blockIdx.x * 8 * RPW + warp;
    const bool live = r0 + 8 * (RPW - 1) < L.n_embd;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint8_t* wr[RPW];
#pragma unroll
    for (int q = 0; q < RPW; ++q) wr[q] = blob + L.down_off + (size_t) (r0 + 8 * q) * L.d_row;
    const int nb = (int) (L.n_ff / 32), hb = (int) (L.n_ff / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; e += GRP_NC) {
        const int n = min(GRP_NC, e1 - e);
        const block_q8_1* hbase = hq;
        int off[GRP_NC];
        if constexpr (LX) {
            __syncthreads();
            for (int i = threadIdx.x; i < n * hb; i += 256) {
                const int c = i / hb, j = i - c * hb;
                memcpy(&s_h[c * S26_HMAX + j], &hq[(size_t) (e + c) * hb + j], sizeof(s_h[0]));
            }
            __syncthreads();
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) off[c] = min(c, n - 1) * S26_HMAX;
            hbase = s_h;
        } else {
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) off[c] = (e + min(c, n - 1)) * hb;
        }
        if (!live) continue;
        float s[RPW][GRP_NC];
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) s[q][c] = 0.0f;
        if constexpr (BAL > 0) {   // S26: 5 items per lane instead of 8 / 4, in BAL groups (loads first)
            constexpr int NI = 2 * S26_HMAX, IT = RPW * NI / 32;
            static_assert(RPW * NI % 32 == 0, "whole steps");
            auto group = [&]<int G0, int... J>(std::integer_sequence<int, J...>) {
                Split<20>::W w[sizeof...(J)];
                int kk[sizeof...(J)];
                bool hh[sizeof...(J)];
                ([&] {
                    using B = S26Bal<G0 + J, NI, RPW>;
                    hh[J] = B::QB != B::QA && lane >= B::THR;
                    kk[J] = B::P0 + lane - (hh[J] ? B::QB : B::QA) * NI;
                    const uint8_t* rp = hh[J] ? wr[B::QB] : wr[B::QA];
                    if constexpr (SL) w[J] = S26Split<20, false>::load(rp, kk[J] / 2, 2 * (kk[J] % 2));
                    else w[J] = Split<20>::load(rp, kk[J] / 2, 2 * (kk[J] % 2));
                }(), ...);
                ([&] {
                    using B = S26Bal<G0 + J, NI, RPW>;
#pragma unroll
                    for (int c = 0; c < GRP_NC; ++c)
                        if (c < n) {
                            const float v = Split<20>::apply(w[J], hbase + off[c] + kk[J] / 2, 2 * (kk[J] % 2));
                            if (hh[J]) s[B::QB][c] += v; else s[B::QA][c] += v;
                        }
                }(), ...);
            };
            constexpr int G = (IT + BAL - 1) / BAL;
            [&]<int... Q>(std::integer_sequence<int, Q...>) {
                (group.template operator()<Q * G>(std::make_integer_sequence<int, (IT - Q * G < G ? IT - Q * G : G)>{}), ...);
            }(std::make_integer_sequence<int, BAL>{});
        } else
        for (int k = lane; k < nb * 2; k += 32) {
            const int kbx = k / 2, iqs = 2 * (k % 2);
            Split<20>::W w[RPW];
#pragma unroll
            for (int q = 0; q < RPW; ++q) { if constexpr (SL) w[q] = S26Split<20, false>::load(wr[q], kbx, iqs); else w[q] = Split<20>::load(wr[q], kbx, iqs); }
#pragma unroll
            for (int q = 0; q < RPW; ++q)
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c)
                    if (c < n) s[q][c] += Split<20>::apply(w[q], hbase + off[c] + kbx, iqs);
        }
        if constexpr (TS) {   // S26: all RPW x GRP_NC sums in one transposed butterfly (bitwise the same sums)
            constexpr int P = s26ts::pow2_ceil(RPW * GRP_NC);
            float v[P];
#pragma unroll
            for (int j = 0; j < P; ++j) v[j] = j < RPW * GRP_NC ? s[j / GRP_NC][j % GRP_NC] : 0.0f;
            const float sum = s26ts::tsum<P>(v, lane);
            const int j = s26ts::tsum_token<P>(lane);
            const int q = j / GRP_NC, c = j % GRP_NC;
            if (lane == s26ts::tsum_lane<P>(j) && j < RPW * GRP_NC && c < n)
                out[(size_t) ent_dst[e + c] * L.n_embd + r0 + 8 * q] = sum;
            continue;
        }
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                if (c < n) s[q][c] = warp_sum(s[q][c]);
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                if (c < n && lane == c) out[(size_t) ent_dst[e + c] * L.n_embd + r0 + 8 * q] = s[q][c];
    }
}


// S26: swiglu_entries_kernel + quantize_q8_1_kernel in one launch (the same h expression, then the same block quantizer,
// contraction off so the product cannot fuse into the first sum); h is still written (it is the scratch the old pair used)
__global__ void s26_swiglu_q8_1_kernel(const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ h,
                                       block_q8_1* __restrict__ y, long long n) {
#pragma clang fp contract(off)
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;   // n is a multiple of 32: whole warps only
    const float g = gate[i];
    const float xi = (g / (1.0f + __expf(-g))) * up[i];
    h[i] = xi;
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float d = q8_1_finite(amax / 127.0f);   // #606: as native_quantize_q8_1_kernel - finite blocks bit for bit
    const int8_t q = q8_1_quant(xi, d, amax);
    const long long ib = i / 32, iqs = i % 32;
    y[ib].qs[iqs] = q;
    if (iqs == 0) y[ib].ds = q8_1_ds(d, sum);   // #606: clamped scale/sum (the S26 path bypassed the finite helper)
}
template<bool LT, bool LX, int RG, int RD, bool SL = false, bool FQ = false, bool TS = false, int BD = 0>
void s26_launch_l(const NativeExpertLayout& L, int64_t cap_groups, cudaStream_t s, const unsigned long long* grp_ptr,
                  const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok,
                  const block_q8_1* X, float* gate, float* up, float* h, block_q8_1* hq, float* out, long long nh) {
    const dim3 ggu((unsigned) (2 * L.n_ff / (GU_ROWS * RG)), (unsigned) cap_groups);
    s26_gu_l_kernel<LT, LX, RG, SL, TS><<<ggu, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
    if (FQ) {
        s26_swiglu_q8_1_kernel<<<(unsigned) ((nh + 255) / 256), 256, 0, s>>>(gate, up, h, hq, nh);
    } else {
        swiglu_entries_kernel<<<(unsigned) ((nh + 255) / 256), 256, 0, s>>>(gate, up, h, nh);
        quantize_q8_1_kernel<<<(unsigned) ((nh + 255) / 256), 256, 0, s>>>(h, hq, nh);
    }
    const dim3 gd((unsigned) (L.n_embd / (8 * RD)), (unsigned) cap_groups);
    s26_down_l_kernel<LX, RD, SL, TS, BD><<<gd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
}
// ---------------------------------------------------------------- stream B: UD-Q4_K_XL grouped decode experts
// STRATA_EXPERT_V2K=1: s26_gu_l_kernel / s26_down_l_kernel's structure (4 interleaved rows per warp, the chunk's q8_1
// activation rows in LDS, SwiGLU + q8_1 fused, optionally STRATA_TSUM's transposed butterfly) for Q4_K / Q5_K gate/up
// and Q5_1 / Q8_0 down, which otherwise run native_gu_kernel / native_down_kernel (one entry at a time, the weight
// words re-read per column).  Each Fmt<TY>::dot is split into S27<TY>::load (everything that depends on the weight) and
// apply (the activation words and the same integer / float expression), the same impl functions, so a row's lane-strided
// k order, its terms and warp_sum are the old kernels': every output is bitwise the old one's (iq harness, memcmp).

template<int TG, bool LX, int RPW, bool TS>
__global__ void __launch_bounds__(256) s27_gu_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                     const int32_t* __restrict__ grp_start,
                                                     const int32_t* __restrict__ n_groups,
                                                     const int32_t* __restrict__ ent_tok,
                                                     const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                     float* __restrict__ gate, float* __restrict__ up) {
    using F = Fmt<TG>;
    using S = S27<TG>;
    __shared__ block_q8_1 s_x[LX ? GRP_NC * S26_XMAX : 1];
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = blockIdx.x * GU_ROWS * RPW + warp;
    const bool live = row0 + GU_ROWS * (RPW - 1) < 2 * L.n_ff;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint8_t* wr[RPW];
    int rr[RPW];
    bool upq[RPW];
#pragma unroll
    for (int q = 0; q < RPW; ++q) {
        const int row = row0 + q * GU_ROWS;
        upq[q] = row >= L.n_ff;
        rr[q] = upq[q] ? row - (int) L.n_ff : row;
        wr[q] = blob + (upq[q] ? L.up_off : 0) + (size_t) rr[q] * L.gu_row;
    }
    const int nb = (int) (L.n_embd / F::qk), xb = (int) (L.n_embd / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; e += GRP_NC) {
        const int n = min(GRP_NC, e1 - e);
        const block_q8_1* xbase = xq;
        int off[GRP_NC];
        if constexpr (LX) {
            __syncthreads();   // the previous chunk is done with s_x
            for (int i = threadIdx.x; i < n * xb; i += 256) {
                const int c = i / xb, j = i - c * xb;
                memcpy(&s_x[c * S26_XMAX + j], &xq[(size_t) ent_tok[e + c] * xb + j], sizeof(s_x[0]));
            }
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) off[c] = min(c, n - 1) * S26_XMAX;
            xbase = s_x;
            __syncthreads();
        } else {
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) off[c] = ent_tok[e + min(c, n - 1)] * xb;
        }
        if (!live) continue;
        float s[RPW][GRP_NC];
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) s[q][c] = 0.0f;
        for (int k = lane; k < nb * F::ipb; k += 32) {
            const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
            typename S::W w[RPW];
#pragma unroll
            for (int q = 0; q < RPW; ++q) w[q] = S::load(wr[q], kbx, iqs);
#pragma unroll
            for (int q = 0; q < RPW; ++q)
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c)
                    if (c < n) s[q][c] += S::apply(w[q], xbase + off[c] + kbx * (F::qk / 32), iqs);
        }
        if constexpr (TS) {
            constexpr int P = s26ts::pow2_ceil(RPW * GRP_NC);
            float v[P];
#pragma unroll
            for (int j = 0; j < P; ++j) v[j] = j < RPW * GRP_NC ? s[j / GRP_NC][j % GRP_NC] : 0.0f;
            const float sum = s26ts::tsum<P>(v, lane);
            const int j = s26ts::tsum_token<P>(lane);
            if (lane == s26ts::tsum_lane<P>(j) && j < RPW * GRP_NC) {
                const int c = j % GRP_NC;
#pragma unroll
                for (int q = 0; q < RPW; ++q)
                    if (j / GRP_NC == q && c < n) (upq[q] ? up : gate)[(size_t) (e + c) * L.n_ff + rr[q]] = sum;
            }
            continue;
        }
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                if (c < n) s[q][c] = warp_sum(s[q][c]);
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                if (c < n && lane == c) (upq[q] ? up : gate)[(size_t) (e + c) * L.n_ff + rr[q]] = s[q][c];
    }
}

template<int TD, bool LX, int RPW, bool TS>
__global__ void __launch_bounds__(256) s27_down_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                       const int32_t* __restrict__ grp_start,
                                                       const int32_t* __restrict__ n_groups,
                                                       const int32_t* __restrict__ ent_dst,
                                                       const block_q8_1* __restrict__ hq, NativeExpertLayout L,
                                                       float* __restrict__ out) {
    using F = Fmt<TD>;
    using S = S27<TD>;
    __shared__ block_q8_1 s_h[LX ? GRP_NC * S26_HMAX : 1];
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = blockIdx.x * 8 * RPW + warp;
    const bool live = r0 + 8 * (RPW - 1) < L.n_embd;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint8_t* wr[RPW];
#pragma unroll
    for (int q = 0; q < RPW; ++q) wr[q] = blob + L.down_off + (size_t) (r0 + 8 * q) * L.d_row;
    const int nb = (int) (L.n_ff / F::qk), hb = (int) (L.n_ff / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; e += GRP_NC) {
        const int n = min(GRP_NC, e1 - e);
        const block_q8_1* hbase = hq;
        int off[GRP_NC];
        if constexpr (LX) {
            __syncthreads();
            for (int i = threadIdx.x; i < n * hb; i += 256) {
                const int c = i / hb, j = i - c * hb;
                memcpy(&s_h[c * S26_HMAX + j], &hq[(size_t) (e + c) * hb + j], sizeof(s_h[0]));
            }
            __syncthreads();
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) off[c] = min(c, n - 1) * S26_HMAX;
            hbase = s_h;
        } else {
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) off[c] = (e + min(c, n - 1)) * hb;
        }
        if (!live) continue;
        float s[RPW][GRP_NC];
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) s[q][c] = 0.0f;
        for (int k = lane; k < nb * F::ipb; k += 32) {
            const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
            typename S::W w[RPW];
#pragma unroll
            for (int q = 0; q < RPW; ++q) w[q] = S::load(wr[q], kbx, iqs);
#pragma unroll
            for (int q = 0; q < RPW; ++q)
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c)
                    if (c < n) s[q][c] += S::apply(w[q], hbase + off[c] + kbx * (F::qk / 32), iqs);
        }
        if constexpr (TS) {
            constexpr int P = s26ts::pow2_ceil(RPW * GRP_NC);
            float v[P];
#pragma unroll
            for (int j = 0; j < P; ++j) v[j] = j < RPW * GRP_NC ? s[j / GRP_NC][j % GRP_NC] : 0.0f;
            const float sum = s26ts::tsum<P>(v, lane);
            const int j = s26ts::tsum_token<P>(lane);
            const int q = j / GRP_NC, c = j % GRP_NC;
            if (lane == s26ts::tsum_lane<P>(j) && j < RPW * GRP_NC && c < n)
                out[(size_t) ent_dst[e + c] * L.n_embd + r0 + 8 * q] = sum;
            continue;
        }
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                if (c < n) s[q][c] = warp_sum(s[q][c]);
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                if (c < n && lane == c) out[(size_t) ent_dst[e + c] * L.n_embd + r0 + 8 * q] = s[q][c];
    }
}

template<int TG, int TD, bool LX, int RG, int RD, bool TS>
void s27_launch(const NativeExpertLayout& L, int64_t cap_groups, cudaStream_t s, const unsigned long long* grp_ptr,
                const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok,
                const block_q8_1* X, float* gate, float* up, float* h, block_q8_1* hq, float* out, long long nh) {
    const dim3 ggu((unsigned) (2 * L.n_ff / (GU_ROWS * RG)), (unsigned) cap_groups);
    s27_gu_kernel<TG, LX, RG, TS><<<ggu, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
    s26_swiglu_q8_1_kernel<<<(unsigned) ((nh + 255) / 256), 256, 0, s>>>(gate, up, h, hq, nh);
    const dim3 gd((unsigned) (L.n_embd / (8 * RD)), (unsigned) cap_groups);
    s27_down_kernel<TD, LX, RD, TS><<<gd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
}
template<int TG, int TD>
void s27_launch_ts(bool ts, const NativeExpertLayout& L, int64_t cap_groups, cudaStream_t s,
                   const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                   const int32_t* ent_dst, const int32_t* ent_tok, const block_q8_1* X, float* gate, float* up, float* h,
                   block_q8_1* hq, float* out, long long nh) {
    if (ts) s27_launch<TG, TD, true, 4, 4, true>(L, cap_groups, s, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, X, gate, up, h, hq, out, nh);
    else s27_launch<TG, TD, true, 4, 4, false>(L, cap_groups, s, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, X, gate, up, h, hq, out, nh);
}

// ---- EXPDEC launch: knobs, shapes, rows per block
int env_int_or(const char* name, int dflt) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' ? std::atoi(v) : dflt;
}
// read at first use (not at load), so a per-architecture default set at start (nv_defaults.hpp) is seen; native_expert_set_expdec overrides
int g_expdec = -1;
int expdec_mode() { if (g_expdec < 0) g_expdec = env_on("STRATA_EXPDEC") ? 1 : 0; return g_expdec; }
int g_expdec_r = env_int_or("STRATA_EXPDEC_R", 1);          // gate/up rows per lane: 1, 2 (bench: R=1 wins on the 2080 Ti)
int g_expdec_dr = env_int_or("STRATA_EXPDEC_DR", 4);        // down rows per lane: 1, 2, 4 (Q2_0: R=4 wins on both cards)
int g_expdec_minb = env_int_or("STRATA_EXPDEC_MINB", STRATA_EXPDEC_MINB);   // gate/up __launch_bounds__ min blocks: 2 (<= 128 regs) or 4 (<= 64)
int g_expdec_d20 = env_int_or("STRATA_EXPDEC_D20", 0);      // 1: EXPDEC also for the IQ4_NL down (bench: no gain, default kernel)
// 1: EXPDEC for the down too (Q2_0; IQ4_NL with _D20).  Default 0: in the engine (nsys of a 131k decode) the Q2_0 down
// with EXPDEC was ~10% slower than the default kernel although the bench alone had it at 0.78, so by default EXPDEC
// takes the gate/up only (-20/-23% at T >= 2 on an RTX 2080 Ti in the engine).
int g_expdec_down = env_int_or("STRATA_EXPDEC_DOWN", 0);
int g_expdec_rb = env_int_or("STRATA_EXPDEC_RB", 0);        // gate/up rows per block (0 = auto)
int g_expdec_drb = env_int_or("STRATA_EXPDEC_DRB", 0);      // down rows per block (0 = auto)
int g_expdec_minblk = env_int_or("STRATA_EXPDEC_MINBLK", 800);   // auto: the most rows per block that keep this many blocks
int expdec_r() { return g_expdec_r == 2 ? 2 : 1; }
int expdec_dr() { return g_expdec_dr == 1 || g_expdec_dr == 2 ? g_expdec_dr : 4; }

bool expdec_gu_ok(const NativeExpertLayout& L) {
    const bool fmt = L.gu_type == 16 || L.gu_type == 17 || L.gu_type == 18 || L.gu_type == 21 || L.gu_type == 22;
    return fmt && L.n_embd == 32 * ED_XB && L.n_ff % 64 == 0 && L.n_ff > 0;
}
bool expdec_down_ok(const NativeExpertLayout& L) {
    return g_expdec_down && ((L.d_type == 20 && g_expdec_d20) || L.d_type == 42) && L.n_ff == 32 * ED_HB && L.n_embd % 128 == 0 &&
           L.n_embd > 0;
}
// rows per block: `knob` rounded up to the rows per pass, or (0) doubled from one pass while ceil(nrow / rb) x
// cap_groups blocks stay >= g_expdec_minblk (cap_groups: the launch's group rows; every block row g >= n_groups returns)
int expdec_rows_per_block(int nrow, int rpp, int64_t cap_groups, int knob) {
    if (knob > 0) {
        int rb = (knob + rpp - 1) / rpp * rpp;
        return rb > nrow ? nrow : rb;
    }
    int rb = rpp;
    const int64_t cap = cap_groups > 0 ? cap_groups : 1;
    while (rb * 2 <= nrow && ((int64_t) ((nrow + 2 * rb - 1) / (2 * rb))) * cap >= g_expdec_minblk) rb *= 2;
    return rb;
}

template<int TG, int R, int MINB>
void expdec_gu_r(int64_t gy, int64_t cap_groups, cudaStream_t s, const unsigned long long* grp_ptr, const int32_t* grp_start,
                 const int32_t* n_groups, const int32_t* ent_tok, const block_q8_1* X, const NativeExpertLayout& L,
                 float* gate, float* up) {
    const int nrow = 2 * (int) L.n_ff;
    const int rb = expdec_rows_per_block(nrow, 16 * R, cap_groups, g_expdec_rb);
    const dim3 grid((unsigned) ((nrow + rb - 1) / rb), (unsigned) gy);
    expdec_gu_kernel<TG, R, MINB><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, rb);
}
template<int TG>
void expdec_gu_t(int64_t gy, int64_t cap_groups, cudaStream_t s, const unsigned long long* grp_ptr, const int32_t* grp_start,
                 const int32_t* n_groups, const int32_t* ent_tok, const block_q8_1* X, const NativeExpertLayout& L,
                 float* gate, float* up) {
    const bool b4 = g_expdec_minb >= 4;
    if (expdec_r() == 2) {
        if (b4) expdec_gu_r<TG, 2, 4>(gy, cap_groups, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
        else expdec_gu_r<TG, 2, 2>(gy, cap_groups, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
    } else {
        if (b4) expdec_gu_r<TG, 1, 4>(gy, cap_groups, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
        else expdec_gu_r<TG, 1, 2>(gy, cap_groups, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
    }
}
// One line on stderr per (role, format, taken or not), the first time STRATA_EXPDEC=1 meets it: the proof in the
// engine's log that the EXPDEC kernels are the ones launched (or captured into the window's graph), with the
// choices in force.
void expdec_note(const NativeExpertLayout& L, bool gu, bool taken, cudaStream_t s, int64_t gy, int64_t cap_groups) {
    static bool seen[2][64][2] = {};
    const int ty = gu ? L.gu_type : L.d_type;
    if (ty < 0 || ty >= 64 || seen[gu ? 0 : 1][ty][taken ? 1 : 0]) return;
    seen[gu ? 0 : 1][ty][taken ? 1 : 0] = true;
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(s, &cs);
    int dev = -1;
    cudaGetDevice(&dev);
    if (!taken) {
        std::fprintf(stderr, "strata expdec: CUDA%d %s type %d (n_embd %lld, n_ff %lld) not covered: the default kernel runs\n", dev,
                     gu ? "gate/up" : "down", ty, (long long) L.n_embd, (long long) L.n_ff);
        return;
    }
    if (gu) {
        const int R = expdec_r(), rb = expdec_rows_per_block(2 * (int) L.n_ff, 16 * R, cap_groups, g_expdec_rb);
        std::fprintf(stderr, "strata expdec: CUDA%d gate/up type %d: expdec_gu_kernel R %d MINB %d, %d rows per block, grid %d x %lld "
                             "(cap %lld groups)%s\n", dev, ty, R, g_expdec_minb >= 4 ? 4 : 2, rb, (2 * (int) L.n_ff + rb - 1) / rb,
                     (long long) gy, (long long) cap_groups, cs != cudaStreamCaptureStatusNone ? ", captured into a graph" : "");
    } else {
        const int R = expdec_dr(), lanes = ty == 20 ? 8 : 16;
        const int rb = expdec_rows_per_block((int) L.n_embd, (256 / lanes) * R, cap_groups, g_expdec_drb);
        std::fprintf(stderr, "strata expdec: CUDA%d down type %d: expdec_down_kernel R %d, %d rows per block, grid %d x %lld "
                             "(cap %lld groups)%s\n", dev, ty, R, rb, ((int) L.n_embd + rb - 1) / rb, (long long) gy,
                     (long long) cap_groups, cs != cudaStreamCaptureStatusNone ? ", captured into a graph" : "");
    }
}

// false: EXPDEC is off or does not take this layout (the caller launches the default kernels)
bool expdec_gu(const NativeExpertLayout& L, int64_t gy, int64_t cap_groups, cudaStream_t s, const unsigned long long* grp_ptr,
               const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_tok, const block_q8_1* X, float* gate,
               float* up) {
    if (!expdec_mode() || g_old_kernels) return false;
    if (!expdec_gu_ok(L)) { expdec_note(L, true, false, s, gy, cap_groups); return false; }
    expdec_note(L, true, true, s, gy, cap_groups);
    switch (L.gu_type) {
        case 16: expdec_gu_t<16>(gy, cap_groups, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 17: expdec_gu_t<17>(gy, cap_groups, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 18: expdec_gu_t<18>(gy, cap_groups, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 21: expdec_gu_t<21>(gy, cap_groups, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        default: expdec_gu_t<22>(gy, cap_groups, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
    }
    return true;
}
template<int TD, int R>
void expdec_down_r(int64_t gy, int64_t cap_groups, cudaStream_t s, const unsigned long long* grp_ptr, const int32_t* grp_start,
                   const int32_t* n_groups, const int32_t* ent_dst, const block_q8_1* hq, const NativeExpertLayout& L, float* out) {
    constexpr int LANES = TD == 20 ? 8 : 16;
    const int nrow = (int) L.n_embd;
    const int rb = expdec_rows_per_block(nrow, (256 / LANES) * R, cap_groups, g_expdec_drb);
    const dim3 grid((unsigned) ((nrow + rb - 1) / rb), (unsigned) gy);
    expdec_down_kernel<TD, R><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out, rb);
}
template<int TD>
void expdec_down_t(int64_t gy, int64_t cap_groups, cudaStream_t s, const unsigned long long* grp_ptr, const int32_t* grp_start,
                   const int32_t* n_groups, const int32_t* ent_dst, const block_q8_1* hq, const NativeExpertLayout& L, float* out) {
    switch (expdec_dr()) {
        case 1: expdec_down_r<TD, 1>(gy, cap_groups, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 2: expdec_down_r<TD, 2>(gy, cap_groups, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        default: expdec_down_r<TD, 4>(gy, cap_groups, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
    }
}
bool expdec_down(const NativeExpertLayout& L, int64_t gy, int64_t cap_groups, cudaStream_t s, const unsigned long long* grp_ptr,
                 const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_dst, const block_q8_1* hq, float* out) {
    if (!expdec_mode() || g_old_kernels) return false;
    if (!expdec_down_ok(L)) { expdec_note(L, false, false, s, gy, cap_groups); return false; }
    expdec_note(L, false, true, s, gy, cap_groups);
    if (L.d_type == 20) expdec_down_t<20>(gy, cap_groups, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
    else expdec_down_t<42>(gy, cap_groups, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
    return true;
}

template<int TG>
void launch_gu(dim3 grid, cudaStream_t s, const unsigned long long* grp_ptr, const int32_t* grp_start,
               const int32_t* n_groups, const int32_t* ent_tok, const block_q8_1* X, const NativeExpertLayout& L,
               float* gate, float* up) {
    // the sub-16 rows (row_dot_80_sub16) assume 80 dots per row: (n_embd / qk) * ipb == 80. The APR trellis types
    // (qk 128, ipb 8: 160 dots at n_embd 2560) take the full-warp multi kernel, as before the sub-16 kernels.
    const bool g_no_sub16_gu = ::strata::kernels::g_no_sub16_gu || (L.n_embd / Fmt<TG>::qk) * Fmt<TG>::ipb != 80;
    if constexpr (!kSplit<TG>) native_gu_kernel<TG><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
    else if (g_old_kernels || split_off<TG>()) native_gu_kernel<TG><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
    else if constexpr (kStageIqGrid<TG>) {
        if constexpr (TG == 144 || TG == 145) {   // tq_sub8_on: the 8-lane trellis rows, 32 rows a block
            if (tq_sub8_on() && (L.n_embd / Fmt<TG>::qk) * Fmt<TG>::ipb == 160) {
                dim3 g8 = grid;
                g8.x = (unsigned) ((2 * L.n_ff + 31) / 32);
                native_gu_multi_kernel<TG, true, true><<<g8, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
                return;
            }
        }
        if (!g_stage_grid || ((TG == 18 || TG == 21) && !stage_tables_on()) || (kTqType<TG> && !tq_stage_on())) {
            if (g_no_sub16_gu) native_gu_multi_kernel<TG, false, false><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
            else native_gu_multi_kernel<TG, false, true><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
            return;
        }
        dim3 sg = grid;
        if constexpr (kTqType<TG>) sg.x = (grid.x + tq_row_chunks() - 1) / tq_row_chunks();   // the staged codebook is paid once per chunks
        if (g_no_sub16_gu) native_gu_multi_kernel<TG, true, false><<<sg, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
        else native_gu_multi_kernel<TG, true, true><<<sg, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
    } else {
        if (g_no_sub16_gu) native_gu_multi_kernel<TG, false, false><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
        else native_gu_multi_kernel<TG, false, true><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
    }
}

// 147-verify: STRATA_GU_SWIGLU=1 runs gate/up, SwiGLU and the q8_1 blocks as native_gu_swiglu_kernel, where launch_gu would
// run native_gu_multi_kernel's 16-lane path (n_embd 2560, the split formats); bitwise the same hq.  Unset: on for a compute
// capability 12.0 card (sm120_series.hpp; RTX PRO 5000, 6 pairs: decode +0.6% story, +0.9% code, +0.5% 8K, 6/6 each, the
// same tokens), off elsewhere; STRATA_GU_SWIGLU=0 turns it off.
bool gu_swiglu_on() {
    static const int env = [] { const char* v = std::getenv("STRATA_GU_SWIGLU"); return v != nullptr ? (std::atoi(v) != 0 ? 1 : 0) : -1; }();
    if (env >= 0) return env != 0;
#if defined(__HIPCC__)
    return false;
#else
    static int per_dev[64];   // 0 unknown, 1 no, 2 yes
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (!per_dev[dev]) {
        int maj = 0, mnr = 0;
        const bool ok = cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                        cudaDeviceGetAttribute(&mnr, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess;
        if (!ok) cudaGetLastError();
        per_dev[dev] = ok && strata::sm120_default(maj, mnr) ? 2 : 1;
    }
    return per_dev[dev] == 2;
#endif
}
template<int TG>
bool launch_gu_swiglu(const NativeExpertLayout& L, int64_t gy, cudaStream_t s, const unsigned long long* grp_ptr,
                      const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_tok, const block_q8_1* X,
                      block_q8_1* hq) {
    if constexpr (!kSplit<TG>) {
        return false;
    } else {
        if (g_old_kernels || split_off<TG>() || g_no_sub16_gu || L.n_embd != 2560 || L.n_ff % 32 != 0 || L.n_ff <= 0) return false;
        // the 16-lane rows assume 80 dots a row (as launch_gu): the 256-value K-quants and the APR trellis types (TQ2_T /
        // TQK6 gate/up: qk 128, ipb 8) have 160 at n_embd 2560 and take launch_gu's full-warp multi kernel.  Gyro-S (TQK6
        // gate/up) went through here on every 12.0 card from 147-verify on: half of each gate/up row, looping answers.
        if ((L.n_embd / Fmt<TG>::qk) * Fmt<TG>::ipb != 80) return false;
        const dim3 grid((unsigned) (L.n_ff / 32), (unsigned) gy);
        bool stage = false;
        if constexpr (kStageIqGrid<TG>) stage = g_stage_grid && !((TG == 18 || TG == 21) && !stage_tables_on()) && !(kTqType<TG> && !tq_stage_on());
        if constexpr (kStageIqGrid<TG>) {
            if (stage) native_gu_swiglu_kernel<TG, true><<<grid, 512, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, hq);
            else native_gu_swiglu_kernel<TG, false><<<grid, 512, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, hq);
        } else {
            native_gu_swiglu_kernel<TG, false><<<grid, 512, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, hq);
        }
        return true;
    }
}

template<int TD>
void launch_down(dim3 grid, cudaStream_t s, const unsigned long long* grp_ptr, const int32_t* grp_start,
                 const int32_t* n_groups, const int32_t* ent_dst, const block_q8_1* hq, const NativeExpertLayout& L,
                 float* out) {
    if constexpr (!kSplit<TD>) native_down_kernel<TD><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
    else if (g_old_kernels || split_off<TD>()) native_down_kernel<TD><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
    else if constexpr (kTqType<TD>) {   // the trellis down types: the codebook staged where tq_stage_on() or tq_sub8_on()
        if (tq_stage_on() || tq_sub8_on()) {
            if (g_no_sub16_gu) native_down_multi_kernel<TD, false, true><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
            else native_down_multi_kernel<TD, true, true><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
        }
        else native_down_multi_kernel<TD, true, false><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
    }
    else if (g_no_sub16_gu) native_down_multi_kernel<TD, false><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
    else native_down_multi_kernel<TD, true><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
}
// ---------------------------------------------------------------- STRATA_DOWN_WS: the decode down projection, staged
// (opt-in, default off; cc 6.0 only).  native_down_multi_kernel's IQ4_NL (K 40, n_ff 640) and Q2_0 (K 20) rows read each
// weight as 2-byte loads, an item at a time, straight from global memory, and a block (32 / 16 rows) waits on that.
// Here a block copies its rows (IQ4_NL 32, Q2_0 64: 11,520 bytes) into shared memory with 16-byte loads - one round
// trip, all in flight - reads the plan words before the group test and the entries' destination rows before the weights
// land, and then runs the production row dots on the copy: the same 8 / 16-lane sub-warp, lane t, items, float
// expressions and order (row_dot_40_sub8 / row_dot_20_sub16: s += item t (+ t + 32), s16, s8, s24, the __fadd_rn's,
// xor 4..1 / 8..1, * sd).  Two integer-only changes inside an item: the dp4a emulation's byte products are taken from
// the table lookup's bytes with vmad byte selectors (no interleaving byte_perm), and IQ4_NL's lookup uses raw prmt
// selectors (bit 3, sign replicate, only touches the bytes the low/high merge drops; lhsi's nibbles are 0..7, so no
// 0x7777 masks).  An item's int32 sum is the sum of the same int8 x int8 products, so every output is bit for bit
// native_down_multi_kernel's (memcmp equal on all 48 layers of the Swift 1.5 IQ3_S pack, all 512 experts, 1-6
// entries a group and T = 1..4 plans; native_expert_bench checks it against mode 0 and mode 8 by output hash).
#if defined(STRATA_EXPERIMENTAL_SM60) && !defined(__HIPCC__) && !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && \
    !defined(STRATA_GPU_METAL)
#define STRATA_DOWN_WS_BUILT 1   // the CUDA build with STRATA_EXPERIMENTAL_SM60 only
#else
#define STRATA_DOWN_WS_BUILT 0
#endif
#if STRATA_DOWN_WS_BUILT
#define STRATA_DWS_VMAD(r, a, sa, b, sb) asm("vmad.s32.s32.s32 %0, %1.b" #sa ", %2.b" #sb ", %0;" : "+r"(r) : "r"(a), "r"(b))

__device__ __forceinline__ uint32_t dws_prmt(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// get_int_from_table_16(q4, kvalues_iq4nl) before its 0x6420 / 0x7531 interleave: t0 = T(v0) T(v16) T(v1) T(v17),
// t1 = T(v2) T(v18) T(v3) T(v19).  low = prmt(T0..7, q) is right for codes < 8, high = prmt(T8..15, q ^ 0x8..) for
// codes >= 8, and the merge with lhsi (nibbles 0..7) takes exactly those bytes - the same bytes as __byte_perm's masked
// selectors give.
__device__ __forceinline__ void dws_iq4_tab(uint32_t q4, int& t0, int& t1) {
    const uint32_t* table32 = (const uint32_t*) kvalues_iq4nl;
    const uint32_t lhsi = 0x32103210u | ((q4 & 0x88888888u) >> 1);
    const uint32_t qx = q4 ^ 0x88888888u;
    t0 = (int) dws_prmt(dws_prmt(table32[0], table32[1], q4), dws_prmt(table32[2], table32[3], qx), lhsi);
    t1 = (int) dws_prmt(dws_prmt(table32[0], table32[1], q4 >> 16), dws_prmt(table32[2], table32[3], qx >> 16), lhsi >> 16);
}

struct DwsIq4W { int t0[2], t1[2]; float dw; };

__device__ __forceinline__ DwsIq4W dws_iq4_load(const void* __restrict__ vbq, int kbx, int iqs) {
    const block_iq4_nl* bq4 = (const block_iq4_nl*) vbq + kbx;
    DwsIq4W r;
#pragma unroll
    for (int l = 0; l < 2; ++l) dws_iq4_tab((uint32_t) get_int_b2(bq4->qs, iqs + l), r.t0[l], r.t1[l]);
    r.dw = __half2float(bq4->d);
    return r;
}

// = Split<20>::apply(Split<20>::load(...)): x_l = (t0.b0 t0.b2 t1.b0 t1.b2) . q8[l], y_l = (t0.b1 t0.b3 t1.b1 t1.b3) . q8[l + 4]
__device__ __forceinline__ float dws_iq4_apply(const DwsIq4W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
    const int* q8 = (const int*) bq8_1->qs + iqs;
    int sumi = 0;
#pragma unroll
    for (int l = 0; l < 2; ++l) {
        const int A = q8[l + 0], B = q8[l + 4];
        STRATA_DWS_VMAD(sumi, r.t0[l], 0, A, 0);
        STRATA_DWS_VMAD(sumi, r.t0[l], 2, A, 1);
        STRATA_DWS_VMAD(sumi, r.t1[l], 0, A, 2);
        STRATA_DWS_VMAD(sumi, r.t1[l], 2, A, 3);
        STRATA_DWS_VMAD(sumi, r.t0[l], 1, B, 0);
        STRATA_DWS_VMAD(sumi, r.t0[l], 3, B, 1);
        STRATA_DWS_VMAD(sumi, r.t1[l], 1, B, 2);
        STRATA_DWS_VMAD(sumi, r.t1[l], 3, B, 3);
    }
    const float d = r.dw * __low2float(bq8_1->ds);
    return d * sumi;
}

// row_dot_40_sub8<20, NC, true> (t < 8: its k loop is k = t, t + 32), the five items' words loaded first
template<int NC>
__device__ __forceinline__ void dws_row40(const uint8_t* row, const block_q8_1* x, int t, float (&s)[NC]) {
    const int k0 = t, k1 = t + 32, k2 = t + 16, k3 = t + 8, k4 = t + 24;
    const DwsIq4W w0 = dws_iq4_load(row, k0 / 2, 2 * (k0 % 2));
    const DwsIq4W w1 = dws_iq4_load(row, k1 / 2, 2 * (k1 % 2));
    const DwsIq4W w2 = dws_iq4_load(row, k2 / 2, 2 * (k2 % 2));
    const DwsIq4W w3 = dws_iq4_load(row, k3 / 2, 2 * (k3 % 2));
    const DwsIq4W w4 = dws_iq4_load(row, k4 / 2, 2 * (k4 % 2));
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const block_q8_1* xc = x + 20 * c;
        s[c] = 0.0f;
        s[c] += dws_iq4_apply(w0, xc + k0 / 2, 2 * (k0 % 2));
        s[c] += dws_iq4_apply(w1, xc + k1 / 2, 2 * (k1 % 2));
        {
            float s16 = 0.0f;
            s16 += dws_iq4_apply(w2, xc + k2 / 2, 2 * (k2 % 2));
            s[c] = __fadd_rn(s[c], s16);
        }
        float s8 = 0.0f;
        s8 += dws_iq4_apply(w3, xc + k3 / 2, 2 * (k3 % 2));
        {
            float s24 = 0.0f;
            s24 += dws_iq4_apply(w4, xc + k4 / 2, 2 * (k4 % 2));
            s[c] = __fadd_rn(s[c], __fadd_rn(s8, s24));
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
        for (int o = 4; o > 0; o >>= 1) s[c] += __shfl_xor_sync(0xffffffffu, s[c], o);
    }
}

// Split<42>::load before its 0x5140 / 0x7362 interleave (qx = qe.b0 qo.b0 qe.b1 qo.b1, qy = qe.b2 qo.b2 qe.b3 qo.b3)
struct DwsQ2W { int qe[4], qo[4]; float d2; };

__device__ __forceinline__ DwsQ2W dws_q2_load(const void* __restrict__ vbq, int kbx, int iqs) {
    const block_q2_0* bq2_0 = (const block_q2_0*) vbq + kbx;
    DwsQ2W r;
    r.d2 = bq2_0->d;
    const int16_t* qs = (const int16_t*) bq2_0->qs + iqs * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = qs[j];
        r.qe[j] = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
        r.qo[j] = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
    }
    return r;
}

// = Split<42>::apply, the 32 products summed in two int32 chains
__device__ __forceinline__ float dws_q2_apply(const DwsQ2W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
    const block_q8_1* bq8_1_chunk = bq8_1 + iqs;
    int a[2] = { 0, 0 };
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int u = get_int_b4(bq8_1_chunk->qs, j * 2 + 0);
        const int v = get_int_b4(bq8_1_chunk->qs, j * 2 + 1);
        int& x = a[j & 1];
        STRATA_DWS_VMAD(x, u, 0, r.qe[j], 0);
        STRATA_DWS_VMAD(x, u, 1, r.qo[j], 0);
        STRATA_DWS_VMAD(x, u, 2, r.qe[j], 1);
        STRATA_DWS_VMAD(x, u, 3, r.qo[j], 1);
        STRATA_DWS_VMAD(x, v, 0, r.qe[j], 2);
        STRATA_DWS_VMAD(x, v, 1, r.qo[j], 2);
        STRATA_DWS_VMAD(x, v, 2, r.qe[j], 3);
        STRATA_DWS_VMAD(x, v, 3, r.qo[j], 3);
    }
    const int sumi = a[0] + a[1];
    const float d8 = __low2float(bq8_1_chunk->ds);
    return r.d2 * d8 * sumi;
}

// row_dot_20_sub16<42, NC, true>
template<int NC>
__device__ __forceinline__ void dws_row20(const uint8_t* row, const block_q8_1* x, int t, float (&s)[NC]) {
    const DwsQ2W w = dws_q2_load(row, t / 2, t % 2);
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        s[c] = 0.0f;
        s[c] += dws_q2_apply(w, x + 20 * c + (t / 2) * 2, t % 2);
    }
    float s16[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) s16[c] = 0.0f;
    if (t < 4) {
        const int k = t + 16;
        const DwsQ2W w2 = dws_q2_load(row, k / 2, k % 2);
#pragma unroll
        for (int c = 0; c < NC; ++c) s16[c] += dws_q2_apply(w2, x + 20 * c + (k / 2) * 2, k % 2);
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        s[c] = __fadd_rn(s[c], s16[c]);
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) s[c] += __shfl_xor_sync(0xffffffffu, s[c], o);
    }
}

// NC entries of one row; every lane of the sub-warp holds the same sum (butterfly), lane c stores entry c
template<int TD, int NC>
__device__ __forceinline__ void dws_rows(const uint8_t* wr, const block_q8_1* s_hq, int t, float sd, float* __restrict__ out,
                                         const int (&dst)[NC], int64_t n_embd, int r) {
    float s[NC];
    if constexpr (TD == 20) dws_row40<NC>(wr, s_hq, t, s);
    else dws_row20<NC>(wr, s_hq, t, s);
#pragma unroll
    for (int c = 0; c < NC; ++c)
        if (t == c) out[(size_t) dst[c] * n_embd + r] = s[c] * sd;
}

// grid (n_embd / RB, gy), 256 threads: RB = 32 IQ4_NL rows (one pass of 8-lane rows) or 64 Q2_0 rows (four passes of
// 16-lane rows); entries NCMAX at a time (1: a T = 1 window, every group one entry; 4: as GRP_NC)
template<int TD, int NCMAX>
__global__ void __launch_bounds__(256, 4) native_down_ws_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                                const int32_t* __restrict__ grp_start,
                                                                const int32_t* __restrict__ n_groups,
                                                                const int32_t* __restrict__ ent_dst,
                                                                const block_q8_1* __restrict__ hq, NativeExpertLayout L,
                                                                float* __restrict__ out) {
    constexpr int LPR = TD == 20 ? 8 : 16, RPP = 256 / LPR, P = TD == 20 ? 1 : 4, RB = RPP * P;
    constexpr int DROW = TD == 20 ? 360 : 180, WB = RB * DROW, NV = (WB / 16 + 255) / 256;
    static_assert(WB % 16 == 0, "16-byte copies");
    STRATA_SHARED_ALIGN16_U32(s_w, WB / 4);
    STRATA_SHARED_ALIGN16_U32(s_hq_buf, NCMAX * 20 * 9);
    const int tid = threadIdx.x, t = tid % LPR, sub = tid / LPR;
    const int r0 = blockIdx.x * RB;
    const int nrow = min(RB, (int) L.n_embd - r0);
    const block_q8_1* s_hq = reinterpret_cast<const block_q8_1*>(s_hq_buf);
    // the plan words of this block row's first group before the test (gridDim.y <= cap_groups: in bounds)
    int g = blockIdx.y;
    unsigned long long gpu = grp_ptr[g];
    int e0 = grp_start[g], e1 = grp_start[g + 1];
    const int ng = *n_groups;
    for (; g < ng; g += gridDim.y) {   // the group stride, as native_down_multi_kernel
        if (g != (int) blockIdx.y) {
            __syncthreads();
            gpu = grp_ptr[g];
            e0 = grp_start[g];
            e1 = grp_start[g + 1];
        }
        const uint8_t* gp = (const uint8_t*) gpu + L.down_off + (size_t) r0 * DROW;
        const float sd = L.tail_off ? ((const float*) (gpu + L.tail_off))[2] : 1.0f;
        int dst[NCMAX];
#pragma unroll
        for (int c = 0; c < NCMAX; ++c) dst[c] = ent_dst[min(e0 + c, e1 - 1)];
        const int nbytes = nrow * DROW;
        if ((((uintptr_t) gp) & 15) == 0 && (nbytes & 15) == 0) {
            const uint4* src = reinterpret_cast<const uint4*>(gp);
            uint4* dw = reinterpret_cast<uint4*>(s_w);
            uint4 v[NV];
#pragma unroll
            for (int k = 0; k < NV; ++k)
                if (tid + 256 * k < (nbytes >> 4)) v[k] = __ldg(src + tid + 256 * k);
#pragma unroll
            for (int k = 0; k < NV; ++k)
                if (tid + 256 * k < (nbytes >> 4)) dw[tid + 256 * k] = v[k];
        } else {
            const uint16_t* s2 = reinterpret_cast<const uint16_t*>(gp);
            uint16_t* d2 = reinterpret_cast<uint16_t*>(s_w);
            for (int i = tid; i < (nbytes >> 1); i += 256) d2[i] = s2[i];
        }
        for (int e = e0; e < e1; e += NCMAX) {
            const int n = min(NCMAX, e1 - e);
            if (e != e0) {
                __syncthreads();
#pragma unroll
                for (int c = 0; c < NCMAX; ++c) dst[c] = ent_dst[min(e + c, e1 - 1)];
            }
            const uint32_t* src = reinterpret_cast<const uint32_t*>(hq + (size_t) e * 20);
            for (int i = tid; i < n * 180; i += 256) s_hq_buf[i] = src[i];
            __syncthreads();
#pragma unroll
            for (int p = 0; p < P; ++p) {
                const int rl = p * RPP + sub;
                if (rl >= nrow) continue;
                const uint8_t* wr = reinterpret_cast<const uint8_t*>(s_w) + rl * DROW;
                const int r = r0 + rl;
                if constexpr (NCMAX == 1) {
                    const int d1[1] = { dst[0] };
                    dws_rows<TD, 1>(wr, s_hq, t, sd, out, d1, L.n_embd, r);
                } else {
                    static_assert(NCMAX == 4, "NCMAX 1 or 4");
                    if (n == 1) {
                        const int d1[1] = { dst[0] };
                        dws_rows<TD, 1>(wr, s_hq, t, sd, out, d1, L.n_embd, r);
                    } else if (n == 2) {
                        const int d2[2] = { dst[0], dst[1] };
                        dws_rows<TD, 2>(wr, s_hq, t, sd, out, d2, L.n_embd, r);
                    } else if (n == 3) {
                        const int d3[3] = { dst[0], dst[1], dst[2] };
                        dws_rows<TD, 3>(wr, s_hq, t, sd, out, d3, L.n_embd, r);
                    } else {
                        const int d4[4] = { dst[0], dst[1], dst[2], dst[3] };
                        dws_rows<TD, 4>(wr, s_hq, t, sd, out, d4, L.n_embd, r);
                    }
                }
            }
        }
    }
}
#undef STRATA_DWS_VMAD

// STRATA_DOWN_WS=1 on a compute capability 6.0 device: the IQ4_NL / Q2_0 down rows at n_ff 640 take
// native_down_ws_kernel (whatever the expert mode: it is bitwise mode 0's, and mode 8's matches mode 0)
bool down_ws_on(const NativeExpertLayout& L) {
    static const bool env = env_on("STRATA_DOWN_WS");
    if (!env || g_old_kernels || g_no_sub16_gu || L.n_ff != 640 || (L.d_type != 20 && L.d_type != 42)) return false;
    static int per_dev[64];   // 0 unknown, 1 no, 2 yes
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    if (!per_dev[dev]) {
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        per_dev[dev] = major == 6 && minor == 0 ? 2 : 1;
    }
    return per_dev[dev] == 2;
}

void launch_down_ws(const NativeExpertLayout& L, int64_t gy, int64_t cap_groups, cudaStream_t s, const unsigned long long* grp_ptr,
                    const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_dst, const block_q8_1* hq, float* out) {
    const int rb = L.d_type == 20 ? 32 : 64;
    const dim3 grid((unsigned) ((L.n_embd + rb - 1) / rb), (unsigned) gy);
    const bool one = cap_groups <= 10;   // a T = 1 window: ten groups of one entry
    if (L.d_type == 20) {
        if (one) native_down_ws_kernel<20, 1><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
        else native_down_ws_kernel<20, 4><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
    } else {
        if (one) native_down_ws_kernel<42, 1><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
        else native_down_ws_kernel<42, 4><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
    }
}
#endif  // STRATA_DOWN_WS_BUILT


// ---------------------------------------------------------------- STRATA_EXPERT_NIBBLE: gate/up on 4-bit slots
// (opt-in, default off; strata/core/expert_nibble.hpp has the slot layout and the item code).  A converted layer's
// VRAM slot holds each 32-weight item of a gate/up row as 16 bytes of 4-bit codes (one 16-byte load), its integer
// scale byte and the superblock's fp16 d bit for bit; the kernels below decode an item into the production
// Split<TG>::W (or IQ3_S's offset form) with 1-4 instructions per four weights instead of the codebook lookups, then
// run the production integer chain and float tail in the production lane / item / addition order (row_dot_80_sub16:
// k = t, t+32, t+64 into s, t+16, t+48 into s16, __fadd_rn, xor 8, 4, 2, 1), so every gate / up value is bit for bit
// native_gu_multi_kernel<TG, *, true>'s.  Two skeletons in the shape of native_gu_multi_kernel (grid (80, gy), 256
// threads, 16 lanes a row, a group's entries GRP_NC at a time, the group stride):
//   S  the entries' q8_1 blocks staged in shared memory, each item's code loaded and decoded where it is used, the
//      production apply on it (IQ3_S: v = 2u - 15 restored per byte - no sum of x)
//   P  the lane's five items' codes, scales and d loaded before the staging (in flight during it), the entries' qs
//      words staged as 16-byte rows and ds as a float, the production chain written out on the registers (IQ3_S:
//      sumi = 2 dot(u, x) - 15 sum(x) with sum(x) from the staged words: the same int32)
// Then the load-time transcoder, the GGUF rebuild (the start-up check and the prompt path's gather) and the slot hash.
// Built only with STRATA_EXPERT_NIBBLE_BUILT (a CUDA build with STRATA_EXPERIMENTAL_SM60); elsewhere a gu_nib layout
// stops in native_expert_grouped, and the switch is refused at start.
#if STRATA_EXPERT_NIBBLE_BUILT
namespace nibk = strata::core::nibble;

__device__ __forceinline__ uint32_t nib_prmt(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));   // the sign-replicate modes too
    return r;
}

// One item's code -> the production W (what Split<TG>::load gives for the same GGUF item, byte for byte).
template<int TG> struct NibLoad;
template<> struct NibLoad<22> {   // IQ2_S: the nibble is the PRMT selector into {8,25,43,0} {-8,-25,-43,0}
    using W = Split<22>::W;
    __device__ static W load(uint4 q, uint32_t sc, __half d) {
        constexpr uint32_t T0 = 0x002B1908u, T1 = 0x00D5E7F8u;
        W r;
        const uint32_t qw[4] = { q.x, q.y, q.z, q.w };
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            r.g[2 * m + 0] = (int) nib_prmt(T0, T1, qw[m]);
            r.g[2 * m + 1] = (int) nib_prmt(T0, T1, qw[m] >> 16);
        }
        r.ls0 = sc & 0x0F;
        r.ls1 = sc >> 4;
        r.dw = __half2float(d);
        return r;
    }
};
template<> struct NibLoad<18> {   // IQ3_XXS: magnitude PRMT into {4..62}, sign-replicated byte masks, (g ^ m) + (m & 1s)
    using W = Split<18>::W;
    __device__ static W load(uint4 q, uint32_t sc, __half d) {
        constexpr uint32_t T0 = 0x1C140C04u, T1 = 0x3E342C24u;   // {4,12,20,28} {36,44,52,62}
        W r;
        const uint32_t qw[4] = { q.x, q.y, q.z, q.w };
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const uint32_t w = qw[m], ms = w & 0x77777777u, a = w << 4;
            const uint32_t m0 = nib_prmt(a, w, 0xD9C8u), m1 = nib_prmt(a, w, 0xFBEAu);
            const uint32_t g0 = nib_prmt(T0, T1, ms), g1 = nib_prmt(T0, T1, ms >> 16);
            r.g[2 * m + 0] = (int) ((g0 ^ m0) + (m0 & 0x01010101u));
            r.g[2 * m + 1] = (int) ((g1 ^ m1) + (m1 & 0x01010101u));
        }
        r.ls = (int) sc;
        r.dw = __half2float(d);
        return r;
    }
};
template<> struct NibLoad<21> {   // IQ3_S: v = 2u - 15 per byte, ((2u | 0x80) - 0x0F) ^ 0x80 (every byte >= 0x80: no borrow)
    using W = Split<21>::W;
    __device__ static W load(uint4 q, uint32_t sc, __half d) {
        W r;
        const uint32_t qw[4] = { q.x, q.y, q.z, q.w };
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const uint32_t lo2 = (qw[m] << 1) & 0x1E1E1E1Eu, hi2 = (qw[m] >> 3) & 0x1E1E1E1Eu;   // 2u per byte
            r.g[2 * m + 0] = (int) (((lo2 | 0x80808080u) - 0x0F0F0F0Fu) ^ 0x80808080u);
            r.g[2 * m + 1] = (int) (((hi2 | 0x80808080u) - 0x0F0F0F0Fu) ^ 0x80808080u);
        }
        r.ls = (int) sc;
        r.dw = __half2float(d);
        return r;
    }
};

// S skeleton row dot over one converted row (codes, scales, d of row `row`), x the staged blocks
template<int TG, int NC>
__device__ __forceinline__ void nib_row_dot_s(const uint4* __restrict__ crow, const uint8_t* __restrict__ srow,
                                              const __half* __restrict__ drow, const block_q8_1* x, const int (&off)[NC],
                                              int t, float (&s)[NC]) {
    using Q = NibLoad<TG>;
    const int iqs = 2 * (t & 7), kb = t >> 3;
    const block_q8_1* xb[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) { s[c] = 0.0f; xb[c] = x + off[c] + kb * 8; }
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const typename Q::W w = Q::load(crow[t + 32 * i], srow[t + 32 * i], drow[kb + 4 * i]);
#pragma unroll
        for (int c = 0; c < NC; ++c) s[c] += Split<TG>::apply(w, xb[c] + 32 * i, iqs);
    }
    float s16[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) s16[c] = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const typename Q::W w = Q::load(crow[t + 16 + 32 * i], srow[t + 16 + 32 * i], drow[kb + 2 + 4 * i]);
#pragma unroll
        for (int c = 0; c < NC; ++c) s16[c] += Split<TG>::apply(w, xb[c] + 16 + 32 * i, iqs);
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        s[c] = __fadd_rn(s[c], s16[c]);
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) s[c] += __shfl_xor_sync(0xffffffffu, s[c], o);
    }
}

// One pass of the S skeleton: the n (1..GRP_NC) entries from e staged, then this thread's row against them.
// Block-uniform (e, n): every thread reaches both barriers.
template<int TG>
__device__ __forceinline__ void nib_pass_s(block_q8_1* s_x, const uint4* crow, const uint8_t* srow, const __half* drow,
                                           const int32_t* __restrict__ ent_tok, const block_q8_1* __restrict__ xq,
                                           int e, int n, int t, float* dst, int r) {
    int tok[GRP_NC];
#pragma unroll
    for (int c = 0; c < GRP_NC; ++c) tok[c] = ent_tok[e + min(c, n - 1)];
    __syncthreads();
    for (int i = threadIdx.x; i < n * 80 * 9; i += 256) {   // 80 blocks x 36 B an entry, as 9 words a block
        const int c = i / 720, rem = i % 720;
        ((int*) s_x)[c * 720 + rem] = ((const int*) (xq + (size_t) tok[c] * 80))[rem];
    }
    __syncthreads();
    if (n == 1) {
        const int o[1] = { 0 }; float s[1];
        nib_row_dot_s<TG, 1>(crow, srow, drow, s_x, o, t, s);
        if (t == 0) dst[(size_t) e * nibk::kNFf + r] = s[0];
    } else if (n == 2) {
        const int o[2] = { 0, 80 }; float s[2];
        nib_row_dot_s<TG, 2>(crow, srow, drow, s_x, o, t, s);
        if (t < 2) dst[(size_t) (e + t) * nibk::kNFf + r] = s[t];
    } else if (n == 3) {
        const int o[3] = { 0, 80, 160 }; float s[3];
        nib_row_dot_s<TG, 3>(crow, srow, drow, s_x, o, t, s);
        if (t < 3) dst[(size_t) (e + t) * nibk::kNFf + r] = s[t];
    } else {
        const int o[4] = { 0, 80, 160, 240 }; float s[4];
        nib_row_dot_s<TG, 4>(crow, srow, drow, s_x, o, t, s);
        if (t < 4) dst[(size_t) (e + t) * nibk::kNFf + r] = s[t];
    }
}

// P skeleton: x as two LDS.128 per item and entry, ds as the float __low2float gives
struct NibXQ { int4 a, b; };

template<int TG>
__device__ __forceinline__ float nib_p_apply(const uint4& q, uint32_t sc, __half dh, const NibXQ& x, float dx, int xsum) {
    if constexpr (TG == 22) {   // SplitLs2::apply on the registers
        const typename NibLoad<22>::W w = NibLoad<22>::load(q, sc, dh);
        int sumi0 = 0, sumi1 = 0;
        sumi0 = ggml_cuda_dp4a(w.g[0], x.a.x, sumi0); sumi0 = ggml_cuda_dp4a(w.g[1], x.a.y, sumi0);
        sumi0 = ggml_cuda_dp4a(w.g[2], x.a.z, sumi0); sumi0 = ggml_cuda_dp4a(w.g[3], x.a.w, sumi0);
        sumi1 = ggml_cuda_dp4a(w.g[4], x.b.x, sumi1); sumi1 = ggml_cuda_dp4a(w.g[5], x.b.y, sumi1);
        sumi1 = ggml_cuda_dp4a(w.g[6], x.b.z, sumi1); sumi1 = ggml_cuda_dp4a(w.g[7], x.b.w, sumi1);
        const int sumi = (sumi0 * w.ls0 + sumi1 * w.ls1 + (sumi0 + sumi1) / 2) / 4;
        const float d = w.dw * dx;
        return d * sumi;
    } else if constexpr (TG == 18) {   // Split<18>::apply on the registers
        const typename NibLoad<18>::W w = NibLoad<18>::load(q, sc, dh);
        int sumi = 0;
        sumi = ggml_cuda_dp4a(w.g[0], x.a.x, sumi); sumi = ggml_cuda_dp4a(w.g[1], x.a.y, sumi);
        sumi = ggml_cuda_dp4a(w.g[2], x.a.z, sumi); sumi = ggml_cuda_dp4a(w.g[3], x.a.w, sumi);
        sumi = ggml_cuda_dp4a(w.g[4], x.b.x, sumi); sumi = ggml_cuda_dp4a(w.g[5], x.b.y, sumi);
        sumi = ggml_cuda_dp4a(w.g[6], x.b.z, sumi); sumi = ggml_cuda_dp4a(w.g[7], x.b.w, sumi);
        sumi = (w.ls * sumi + sumi / 2) / 2;
        const float d = w.dw * dx;
        return d * sumi;
    } else {   // IQ3_S, offset form: dot(v, x) = 2 dot(u, x) - 15 sum(x), exact (|2 dot(u, x)| < 2^17); then Split<21>'s
        int u[8];
        const uint32_t qw[4] = { q.x, q.y, q.z, q.w };
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            u[2 * m + 0] = (int) (qw[m] & 0x0F0F0F0Fu);
            u[2 * m + 1] = (int) ((qw[m] >> 4) & 0x0F0F0F0Fu);
        }
        int sumu = 0;
        sumu = ggml_cuda_dp4a(u[0], x.a.x, sumu); sumu = ggml_cuda_dp4a(u[1], x.a.y, sumu);
        sumu = ggml_cuda_dp4a(u[2], x.a.z, sumu); sumu = ggml_cuda_dp4a(u[3], x.a.w, sumu);
        sumu = ggml_cuda_dp4a(u[4], x.b.x, sumu); sumu = ggml_cuda_dp4a(u[5], x.b.y, sumu);
        sumu = ggml_cuda_dp4a(u[6], x.b.z, sumu); sumu = ggml_cuda_dp4a(u[7], x.b.w, sumu);
        int sumi = 2 * sumu - 15 * xsum;
        sumi *= (int) sc;
        const float d = __half2float(dh) * dx;
        return d * sumi;
    }
}

// the lane's items in the production order: k = t, t+32, t+64 (sum s), then t+16, t+48 (sum s16)
template<int TG, int NC>
__device__ __forceinline__ void nib_row_dot_p(const uint4 (&q)[5], const uint32_t (&sc)[5], const __half (&dh)[5],
                                              const NibXQ* __restrict__ sx, const float* __restrict__ sd,
                                              const int* __restrict__ sxs, int t, float (&s)[NC]) {
    constexpr int K[5] = { 0, 32, 64, 16, 48 };
#pragma unroll
    for (int c = 0; c < NC; ++c) s[c] = 0.0f;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int k = t + K[i];
#pragma unroll
        for (int c = 0; c < NC; ++c)
            s[c] += nib_p_apply<TG>(q[i], sc[i], dh[i], sx[c * 80 + k], sd[c * 80 + k], TG == 21 ? sxs[c * 80 + k] : 0);
    }
    float s16[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) s16[c] = 0.0f;
#pragma unroll
    for (int i = 3; i < 5; ++i) {
        const int k = t + K[i];
#pragma unroll
        for (int c = 0; c < NC; ++c)
            s16[c] += nib_p_apply<TG>(q[i], sc[i], dh[i], sx[c * 80 + k], sd[c * 80 + k], TG == 21 ? sxs[c * 80 + k] : 0);
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        s[c] = __fadd_rn(s[c], s16[c]);
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) s[c] += __shfl_xor_sync(0xffffffffu, s[c], o);
    }
}

// The P skeleton's work on one group: this lane's five items into registers, then the passes.  Block-uniform.
template<int TG>
__device__ __forceinline__ void nib_group_p(NibXQ* s_x, float* s_d, int* s_xs, const uint4* crow, const uint8_t* srow,
                                            const __half* drow, const int32_t* __restrict__ ent_tok,
                                            const block_q8_1* __restrict__ xq, int e0, int e1, int t, float* dst, int r) {
    const int kb = t >> 3;
    uint4 q[5]; uint32_t sc[5]; __half dh[5];
    q[0] = crow[t]; q[1] = crow[t + 32]; q[2] = crow[t + 64]; q[3] = crow[t + 16]; q[4] = crow[t + 48];
    sc[0] = srow[t]; sc[1] = srow[t + 32]; sc[2] = srow[t + 64]; sc[3] = srow[t + 16]; sc[4] = srow[t + 48];
    dh[0] = drow[kb]; dh[1] = drow[kb + 4]; dh[2] = drow[kb + 8]; dh[3] = drow[kb + 2]; dh[4] = drow[kb + 6];
    for (int e = e0; e < e1; e += GRP_NC) {
        const int n = min(GRP_NC, e1 - e);
        __syncthreads();
        for (int i = threadIdx.x; i < n * 80; i += 256) {
            const int c = i >= 240 ? 3 : i >= 160 ? 2 : i >= 80 ? 1 : 0, kk = i - 80 * c;
            const int tok = ent_tok[e + c];
            const int* src = (const int*) (xq + (size_t) tok * 80 + kk);
            NibXQ v;
            v.a = make_int4(src[1], src[2], src[3], src[4]);
            v.b = make_int4(src[5], src[6], src[7], src[8]);
            s_x[i] = v;
            s_d[i] = __low2float(xq[(size_t) tok * 80 + kk].ds);
            if constexpr (TG == 21) {   // sum(x): the production dp4a against 0x01010101 over the 8 qs words
                int xs = 0;
                xs = ggml_cuda_dp4a(v.a.x, 0x01010101, xs); xs = ggml_cuda_dp4a(v.a.y, 0x01010101, xs);
                xs = ggml_cuda_dp4a(v.a.z, 0x01010101, xs); xs = ggml_cuda_dp4a(v.a.w, 0x01010101, xs);
                xs = ggml_cuda_dp4a(v.b.x, 0x01010101, xs); xs = ggml_cuda_dp4a(v.b.y, 0x01010101, xs);
                xs = ggml_cuda_dp4a(v.b.z, 0x01010101, xs); xs = ggml_cuda_dp4a(v.b.w, 0x01010101, xs);
                s_xs[i] = xs;
            }
        }
        __syncthreads();
        if (n == 1) {
            float s[1]; nib_row_dot_p<TG, 1>(q, sc, dh, s_x, s_d, s_xs, t, s);
            if (t == 0) dst[(size_t) e * nibk::kNFf + r] = s[0];
        } else if (n == 2) {
            float s[2]; nib_row_dot_p<TG, 2>(q, sc, dh, s_x, s_d, s_xs, t, s);
            if (t < 2) dst[(size_t) (e + t) * nibk::kNFf + r] = s[t];
        } else if (n == 3) {
            float s[3]; nib_row_dot_p<TG, 3>(q, sc, dh, s_x, s_d, s_xs, t, s);
            if (t < 3) dst[(size_t) (e + t) * nibk::kNFf + r] = s[t];
        } else {
            float s[4]; nib_row_dot_p<TG, 4>(q, sc, dh, s_x, s_d, s_xs, t, s);
            if (t < 4) dst[(size_t) (e + t) * nibk::kNFf + r] = s[t];
        }
    }
}

// SKEL kSkelS or kSkelP for every group.  grid (80, gy): 2 x 640 rows = 80 x 16, so every thread has a row and
// reaches every barrier.
constexpr int kNibSWords = GRP_NC * 80 * (int) (sizeof(block_q8_1) / 4);   // S: the blocks
constexpr int kNibXqWords = GRP_NC * 80 * (int) (sizeof(NibXQ) / 4);         // P: the qs rows, then ds, then sum(x)
template<int TG, int SKEL>
inline constexpr int kNibSmemWords = SKEL == nibk::kSkelS ? kNibSWords : kNibXqWords + GRP_NC * 80 + (TG == 21 ? GRP_NC * 80 : 0);
template<int TG, int SKEL>
__global__ void __launch_bounds__(256) nib_gu_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                     const int32_t* __restrict__ grp_start,
                                                     const int32_t* __restrict__ n_groups,
                                                     const int32_t* __restrict__ ent_tok,
                                                     const block_q8_1* __restrict__ xq, float* __restrict__ gate,
                                                     float* __restrict__ up) {
    const int ng = *n_groups;
    if (blockIdx.y >= (unsigned) ng) return;
    STRATA_SHARED_ALIGN16_U32(s_buf, kNibSmemWords<TG, SKEL>);
    const int subwarp = threadIdx.x >> 4, t = threadIdx.x & 15;
    const int row = blockIdx.x * 16 + subwarp;   // 0 .. 2 n_ff
    const bool is_up = row >= nibk::kNFf;
    const int r = is_up ? row - (int) nibk::kNFf : row;
    float* dst = is_up ? up : gate;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {
        const uint8_t* slot = (const uint8_t*) grp_ptr[g];
        const uint4* crow = (const uint4*) (slot + nibk::kCodesOff) + (size_t) row * nibk::kItems;
        const uint8_t* srow = slot + nibk::kScOff + (size_t) row * nibk::kItems;
        const __half* drow = (const __half*) (slot + nibk::kDOff) + (size_t) row * nibk::kSuper;
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        if constexpr (SKEL == nibk::kSkelP) {
            nib_group_p<TG>((NibXQ*) s_buf, (float*) (s_buf + kNibXqWords), (int*) (s_buf + kNibXqWords + GRP_NC * 80),
                            crow, srow, drow, ent_tok, xq, e0, e1, t, dst, r);
        } else {
            for (int e = e0; e < e1; e += GRP_NC)
                nib_pass_s<TG>((block_q8_1*) s_buf, crow, srow, drow, ent_tok, xq, e, min(GRP_NC, e1 - e), t, dst, r);
        }
    }
}

#include "nib_gu_proto.cuh"   // STRATA_NIB_GU_PROTO: the staged-once 4-bit gate/up kernel (id 31)

int g_nib_variant = -1;   // native_expert_nib_set_variant (the bench / parity); -1 = expert_nibble.hpp's gu_skeleton

// STRATA_NIB_GU_PROTO: unset / 0 = off (the S / P skeleton per format); 1 (or 31) = gu-proto 31 (the activations staged
// once per two row blocks with 16-byte loads, the production row dot).  Read once.
int nib_gu_proto_env() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_NIB_GU_PROTO");
        int id = e != nullptr ? std::atoi(e) : 0;
        if (id == 1) id = 31;
        if (id != 31) id = 0;
        if (id != 0) std::fprintf(stderr, "native_expert_grouped: STRATA_NIB_GU_PROTO: 4-bit gate/up kernel gu-proto %d\n", id);
        return id;
    }();
    return v;
}

// The gate/up of a converted layer's VRAM groups: native_gu_multi_kernel's grid and outputs.
void launch_gu_nib(const NativeExpertLayout& L, dim3 grid, cudaStream_t s, const unsigned long long* grp_ptr,
                   const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_tok, const block_q8_1* X,
                   float* gate, float* up) {
    if (g_nib_variant >= 10) {   // gu-proto (nib_gu_proto.cuh), forced by the bench / parity
        if (((uintptr_t) X & 15) != 0 ||
            !nibx_launch(L.gu_type, g_nib_variant, grid, s, grp_ptr, grp_start, n_groups, ent_tok, X, gate, up)) {
            std::fprintf(stderr, "native_expert_grouped: gu-proto variant %d for type %d is not built (or x is not 16-aligned)\n",
                         g_nib_variant, L.gu_type);
            std::exit(1);
        }
        return;
    }
    // STRATA_NIB_GU_PROTO (opt-in, default off): the gu-proto kernel in place of the S / P skeleton - bitwise the
    // same gate / up (iq_nibble_parity (7), native_expert_bench --nibble --vs).  The 16-byte staging needs x 16-aligned
    // (verify's nat_xq_ rows are: a 256-aligned buffer, 2,880 B a token); otherwise the production kernel runs.
    if (g_nib_variant < 0) {
        const int pv = nib_gu_proto_env();
        if (pv != 0 && ((uintptr_t) X & 15) == 0 &&
            nibx_launch(L.gu_type, pv, grid, s, grp_ptr, grp_start, n_groups, ent_tok, X, gate, up))
            return;
    }
    const int skel = g_nib_variant >= 0 ? (g_nib_variant == 1 ? nibk::kSkelP : nibk::kSkelS) : nibk::gu_skeleton(L.gu_type);
#define STRATA_NIB_GU(TG, S) \
    case TG * 2 + S: nib_gu_kernel<TG, S><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, gate, up); break;
    switch (L.gu_type * 2 + skel) {
        STRATA_NIB_GU(22, 0) STRATA_NIB_GU(22, 1)
        STRATA_NIB_GU(18, 0) STRATA_NIB_GU(18, 1)
        STRATA_NIB_GU(21, 0) STRATA_NIB_GU(21, 1)
        default:
            std::fprintf(stderr, "native_expert_grouped: 4-bit gate/up type %d at %lld x %lld is not built\n", L.gu_type,
                         (long long) L.n_embd, (long long) L.n_ff);
            std::exit(1);
    }
#undef STRATA_NIB_GU
}

// ---- the load-time transcoder (expert_nibble.hpp's transcode)
template<int TG>
__device__ __forceinline__ uint32_t nib_code(int v, bool& ok) {
    if constexpr (TG == 21) {   // v odd in -15..15 -> u = (v + 15) / 2
        if ((v & 1) == 0 || v < -15 || v > 15) { ok = false; return 0; }
        return (uint32_t) ((v + 15) >> 1);
    } else if constexpr (TG == 18) {   // index of {4, 12, ..., 52, 62} | sign << 3
        const int a = v < 0 ? -v : v;
        const int i = a == 62 ? 7 : ((a & 7) == 4 && a <= 52) ? (a - 4) >> 3 : -1;
        if (i < 0) { ok = false; return 0; }
        return (uint32_t) i | (v < 0 ? 8u : 0u);
    } else {   // IQ2_S: index of {8, 25, 43} | sign << 2
        const int a = v < 0 ? -v : v;
        const int i = a == 8 ? 0 : a == 25 ? 1 : a == 43 ? 2 : -1;
        if (i < 0) { ok = false; return 0; }
        return (uint32_t) i | (v < 0 ? 4u : 0u);
    }
}

// The GGUF blocks (ggml-common.h): their size and the fp16 d at their start
template<int TG> struct NibBlk;
template<> struct NibBlk<22> { using B = block_iq2_s; };
template<> struct NibBlk<18> { using B = block_iq3_xxs; };
template<> struct NibBlk<21> { using B = block_iq3_s; };

// One thread per (expert, row, item): the production load, the code, scale and d written, read back from memory and
// decoded with the kernels' own decoders, compared with the load's W (and, for IQ3_S, the P skeleton's u against v);
// `bad` counts the items that differ or hold a weight outside the code set.
template<int TG>
__global__ void __launch_bounds__(256) nib_transcode_kernel(const uint8_t* __restrict__ src, uint64_t raw_bytes,
                                                            uint64_t gu_row, uint64_t up_off,
                                                            const unsigned long long* __restrict__ slots, long long n,
                                                            unsigned long long* __restrict__ bad) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * nibk::kRows * nibk::kItems) return;
    const int k = (int) (i % nibk::kItems);
    const int row = (int) ((i / nibk::kItems) % nibk::kRows);
    const long long ex = i / (nibk::kItems * nibk::kRows);
    const bool is_up = row >= nibk::kNFf;
    const int r = is_up ? row - (int) nibk::kNFf : row;
    const uint8_t* wr = src + (size_t) ex * raw_bytes + (is_up ? up_off : 0) + (size_t) r * gu_row;
    const int kbx = k / Fmt<TG>::ipb, iqs = Fmt<TG>::step * (k % Fmt<TG>::ipb);
    using S = Split<TG>;
    const typename S::W w = S::template load<false>(wr, kbx, iqs, nullptr);
    bool ok = true;
    uint32_t words[4];
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const uint32_t g0 = (uint32_t) w.g[2 * m], g1 = (uint32_t) w.g[2 * m + 1];
        uint32_t word = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const uint32_t c0 = nib_code<TG>((int) (int8_t) (g0 >> (8 * b)), ok);
            const uint32_t c1 = nib_code<TG>((int) (int8_t) (g1 >> (8 * b)), ok);
            if constexpr (TG == 21) word |= (c0 | c1 << 4) << (8 * b);
            else word |= c0 << (4 * b) | c1 << (16 + 4 * b);
        }
        words[m] = word;
    }
    uint32_t sc;
    if constexpr (TG == 22) sc = (uint32_t) (w.ls0 | w.ls1 << 4);
    else sc = (uint32_t) w.ls;
    ok = ok && sc < 256u;
    uint8_t* slot = (uint8_t*) slots[ex];
    uint4* cp = (uint4*) (slot + nibk::code_off(row, k));
    uint8_t* sp = slot + nibk::scale_off(row, k);
    *cp = make_uint4(words[0], words[1], words[2], words[3]);
    *sp = (uint8_t) sc;
    const __half dsrc = ((const typename NibBlk<TG>::B*) wr + kbx)->d;   // the block's d, bit for bit
    __half dread = dsrc;
    if (k % Fmt<TG>::ipb == 0) {   // the superblock's first item writes its d
        __half* dp = (__half*) (slot + nibk::d_off(row, kbx));
        *dp = dsrc;
        dread = __ushort_as_half(*(volatile const unsigned short*) dp);
        ok = ok && __half_as_ushort(dread) == __half_as_ushort(dsrc);
    }
    // the code and the scale read back from memory and decoded as the kernels do
    const uint4 qb = __ldcv(cp);
    const uint32_t sb = *(volatile const uint8_t*) sp;
    const typename NibLoad<TG>::W v = NibLoad<TG>::load(qb, sb, dread);
#pragma unroll
    for (int j = 0; j < 8; ++j) ok = ok && v.g[j] == w.g[j];
    if constexpr (TG == 22) ok = ok && v.ls0 == w.ls0 && v.ls1 == w.ls1;
    else ok = ok && v.ls == w.ls;
    ok = ok && __float_as_uint(v.dw) == __float_as_uint(w.dw);
    if constexpr (TG == 21) {   // the P skeleton's u: 2u - 15 per byte is v
        const uint32_t qw[4] = { qb.x, qb.y, qb.z, qb.w };
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const uint32_t u0 = qw[m] & 0x0F0F0F0Fu, u1 = (qw[m] >> 4) & 0x0F0F0F0Fu;
            ok = ok && __vsub4(__vadd4(u0, u0), 0x0F0F0F0Fu) == (uint32_t) w.g[2 * m] &&
                 __vsub4(__vadd4(u1, u1), 0x0F0F0F0Fu) == (uint32_t) w.g[2 * m + 1];
        }
    }
    if (!ok) atomicAdd(bad, 1ull);
}

// ---- the GGUF rebuild (expert_nibble.hpp's check_gguf and gather_gguf)
// The reverse codebooks: a group's magnitude pattern -> its grid index (0xFFFFFFFF: not in the grid).  32-bit
// entries for atomicCAS, which also counts a pattern two grid entries share (none in ggml's grids; then refused).
constexpr int kNibRev22 = 6561, kNibRev18 = 4096, kNibRev21 = 4096;   // 3^8 (IQ2_S: 8 weights), 8^4, 8^4 (4 weights)
__device__ uint32_t g_nib_rev22[kNibRev22];
__device__ uint32_t g_nib_rev18[kNibRev18];
__device__ uint32_t g_nib_rev21[kNibRev21];
__device__ unsigned long long g_nib_rev_bad;

__device__ __forceinline__ int nib_mag22(uint32_t b) { return b == 8 ? 0 : b == 25 ? 1 : b == 43 ? 2 : -1; }
__device__ __forceinline__ int nib_mag18(uint32_t b) { return b == 62 ? 7 : ((b & 7) == 4 && b <= 52) ? (int) (b - 4) >> 3 : -1; }
__device__ __forceinline__ int nib_mag21(uint32_t b) { return (b & 1) && b <= 15 ? (int) (b - 1) >> 1 : -1; }

__global__ void nib_rev_clear_kernel() {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kNibRev22) g_nib_rev22[i] = 0xFFFFFFFFu;
    if (i < kNibRev18) g_nib_rev18[i] = 0xFFFFFFFFu;
    if (i < kNibRev21) g_nib_rev21[i] = 0xFFFFFFFFu;
    if (i == 0) g_nib_rev_bad = 0;
}
__global__ void nib_rev_fill_kernel() {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t* tab = nullptr;
    int key = 0;
    bool ok = true;
    uint32_t idx = 0;
    if (i < 1024) {   // iq2s_grid: 8 bytes of {8, 25, 43}
        const uint64_t g = iq2s_grid[i];
        for (int j = 7; j >= 0; --j) { const int m = nib_mag22((uint32_t) (g >> (8 * j)) & 0xFF); ok = ok && m >= 0; key = key * 3 + (m < 0 ? 0 : m); }
        tab = g_nib_rev22; idx = (uint32_t) i;
    } else if (i < 1024 + 256) {   // iq3xxs_grid: 4 bytes of {4, ..., 62}
        const uint32_t g = iq3xxs_grid[i - 1024];
        for (int j = 3; j >= 0; --j) { const int m = nib_mag18((g >> (8 * j)) & 0xFF); ok = ok && m >= 0; key = key * 8 + (m < 0 ? 0 : m); }
        tab = g_nib_rev18; idx = (uint32_t) (i - 1024);
    } else if (i < 1024 + 256 + 512) {   // iq3s_grid: 4 bytes of {1, 3, ..., 15}
        const uint32_t g = iq3s_grid[i - 1280];
        for (int j = 3; j >= 0; --j) { const int m = nib_mag21((g >> (8 * j)) & 0xFF); ok = ok && m >= 0; key = key * 8 + (m < 0 ? 0 : m); }
        tab = g_nib_rev21; idx = (uint32_t) (i - 1280);
    } else {
        return;
    }
    if (!ok || atomicCAS(tab + key, 0xFFFFFFFFu, idx) != 0xFFFFFFFFu) atomicAdd(&g_nib_rev_bad, 1ull);
}

// The bytes of GGUF block `s` of row `row`'s item `ib` (0..7) rebuilt from the slot into `blk` (shared memory, the
// block's first byte): the codes give each weight's magnitude index and sign, the reverse codebook the grid index,
// the scale byte and d are the block's own.  Disjoint bytes per item (IQ3_S: the even item writes the shared scale
// byte); item 0 writes d.
template<int TG>
__device__ __forceinline__ void nib_gguf_item(uint8_t* blk, int ib, const uint8_t* __restrict__ slot, int row, int k) {
    const uint4 q4 = *(const uint4*) (slot + nibk::code_off(row, k));
    const uint32_t w[4] = { q4.x, q4.y, q4.z, q4.w };
    const uint32_t sc = slot[nibk::scale_off(row, k)];
    if (ib == 0) {
        const uint8_t* dp = slot + nibk::d_off(row, k / 8);
        blk[0] = dp[0];
        blk[1] = dp[1];
    }
    if constexpr (TG == 22) {   // block_iq2_s: d, qs[32] grid low bits, qs[32] signs, qh[8], scales[8]
        uint32_t qh = 0;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            int key = 0;
            uint32_t sg = 0;
#pragma unroll
            for (int j = 7; j >= 0; --j) {
                const uint32_t c = j < 4 ? (w[l] >> (4 * j)) & 15u : (w[l] >> (16 + 4 * (j - 4))) & 15u;
                key = key * 3 + (int) (c & 3u);
                sg |= ((c >> 2) & 1u) << j;
            }
            const uint32_t idx = key < kNibRev22 ? g_nib_rev22[key] : 0xFFFFFFFFu;
            blk[2 + 4 * ib + l] = (uint8_t) idx;
            blk[2 + 32 + 4 * ib + l] = (uint8_t) sg;
            qh |= ((idx >> 8) & 3u) << (2 * l);
        }
        blk[66 + ib] = (uint8_t) qh;
        blk[74 + ib] = (uint8_t) sc;
    } else if constexpr (TG == 18) {   // block_iq3_xxs: d, qs[64] grid indices, 8 x aux32 (4 x 7 sign bits | scale << 28)
        uint32_t aux = sc << 28;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            uint32_t sg = 0;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                int key = 0;
#pragma unroll
                for (int j = 3; j >= 0; --j) {
                    const uint32_t c = (w[l] >> (16 * h + 4 * j)) & 15u;
                    key = key * 8 + (int) (c & 7u);
                    sg |= ((c >> 3) & 1u) << (4 * h + j);
                }
                blk[2 + 8 * ib + 2 * l + h] = (uint8_t) g_nib_rev18[key];
            }
            aux |= (sg & 0x7Fu) << (7 * l);   // the 8th sign is the parity of the other 7 (ksigns_iq2xs)
        }
        blk[2 + 64 + 4 * ib + 0] = (uint8_t) aux;
        blk[2 + 64 + 4 * ib + 1] = (uint8_t) (aux >> 8);
        blk[2 + 64 + 4 * ib + 2] = (uint8_t) (aux >> 16);
        blk[2 + 64 + 4 * ib + 3] = (uint8_t) (aux >> 24);
    } else {   // block_iq3_s: d, qs[64] grid low bits, qh[8], signs[32], scales[4] (two items' nibbles each)
        uint32_t qh = 0;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            uint32_t sg = 0;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                int key = 0;
#pragma unroll
                for (int j = 3; j >= 0; --j) {
                    const uint32_t u = (w[l] >> (8 * j + 4 * h)) & 15u;   // v = 2u - 15: negative for u <= 7
                    key = key * 8 + (int) (u <= 7u ? 7u - u : u - 8u);
                    sg |= (u <= 7u ? 1u : 0u) << (4 * h + j);
                }
                const uint32_t idx = g_nib_rev21[key];
                blk[2 + 8 * ib + 2 * l + h] = (uint8_t) idx;
                qh |= ((idx >> 8) & 1u) << (2 * l + h);
            }
            blk[74 + 4 * ib + l] = (uint8_t) sg;
        }
        blk[66 + ib] = (uint8_t) qh;
        if ((ib & 1) == 0) {
            const uint32_t sc1 = slot[nibk::scale_off(row, k + 1)];
            blk[106 + ib / 2] = (uint8_t) (((sc - 1u) >> 1) | (((sc1 - 1u) >> 1) << 4));
        }
    }
}

constexpr int kNibGgufRows = 8;        // gate/up rows a block rebuilds (in shared memory, then one coalesced pass)
constexpr int kNibGgufThreads = 128;
constexpr int kNibGgufWords = 8;       // uint4 per thread in the down / tail part
struct NibGgufArgs {
    const unsigned long long* dev;     // the slots' device addresses (check), or nullptr: `p`
    const uint8_t* p[16];              // the slots (gather: an MMQ group, at most mmq::kGatherGroupMax)
    int first, n;
    uint8_t* gu;                       // gather: the rebuilt [gate | up]; check: the raw blobs' first byte
    uint64_t gu_stride;
    uint8_t* d;                        // gather: the down rows; check: the raw blobs' down rows
    uint64_t d_stride;
    uint64_t gu_row, down16, zero16;   // GGUF row bytes; down rows and MMQ tail in uint4
    unsigned long long* bad;           // check: 32-bit words / uint4 that differ
};
// blockIdx.y: expert first + y.  blockIdx.x < kRows / kNibGgufRows: rows [8x, 8x + 8) rebuilt into shared memory,
// then written (gather) or compared (check) as 32-bit words; the rest: the down rows as uint4, and (gather, the last
// expert) the zeroed tails after gate/up and down.
template<int TG, bool CHECK>
__global__ void __launch_bounds__(kNibGgufThreads) nib_gguf_kernel(NibGgufArgs a) {
    constexpr int BB = (int) sizeof(typename NibBlk<TG>::B);
    __shared__ uint32_t s_rows[kNibGgufRows * 1100 / 4];
    const int q = a.first + (int) blockIdx.y;
    const uint8_t* slot = a.dev != nullptr ? (const uint8_t*) a.dev[q] : a.p[q];
    constexpr int kRowBlocks = (int) (nibk::kRows / kNibGgufRows);
    unsigned long long nbad = 0;
    if ((int) blockIdx.x < kRowBlocks) {
        const int r0 = (int) blockIdx.x * kNibGgufRows;
        uint8_t* sb = (uint8_t*) s_rows;
        for (int u = threadIdx.x; u < kNibGgufRows * (int) nibk::kItems; u += kNibGgufThreads) {
            const int rr = u / (int) nibk::kItems, k = u % (int) nibk::kItems;
            nib_gguf_item<TG>(sb + (size_t) rr * a.gu_row + (size_t) (k >> 3) * BB, k & 7, slot, r0 + rr, k);
        }
        __syncthreads();
        const int nw = (int) (kNibGgufRows * a.gu_row / 4);
        uint32_t* dst = (uint32_t*) (a.gu + (size_t) q * a.gu_stride + (size_t) r0 * a.gu_row);
        for (int i = threadIdx.x; i < nw; i += kNibGgufThreads) {
            if constexpr (CHECK) nbad += dst[i] != s_rows[i];
            else dst[i] = s_rows[i];
        }
    } else {
        const uint4* src = (const uint4*) (slot + nibk::kDownOff);
        uint4* dd = (uint4*) (a.d + (size_t) q * a.d_stride);
        const int64_t base = ((int64_t) blockIdx.x - kRowBlocks) * (kNibGgufThreads * kNibGgufWords);
#pragma unroll
        for (int w = 0; w < kNibGgufWords; ++w) {
            const int64_t i = base + (int64_t) w * kNibGgufThreads + threadIdx.x;
            if (i < (int64_t) a.down16) {
                if constexpr (CHECK) {
                    const uint4 x = src[i], y = dd[i];
                    nbad += x.x != y.x || x.y != y.y || x.z != y.z || x.w != y.w;
                } else {
                    dd[i] = src[i];
                }
            } else if (!CHECK && q == a.n - 1 && i < (int64_t) (a.down16 + 2 * a.zero16)) {
                const int64_t z = i - (int64_t) a.down16;
                if (z < (int64_t) a.zero16) ((uint4*) (a.gu + (size_t) a.n * a.gu_stride))[z] = make_uint4(0, 0, 0, 0);
                else ((uint4*) (a.d + (size_t) a.n * a.d_stride))[z - (int64_t) a.zero16] = make_uint4(0, 0, 0, 0);
            }
        }
    }
    if constexpr (CHECK) {
        if (nbad) atomicAdd(a.bad, nbad);
    }
}

// sum_i (w_i ^ i) * kHashMul mod 2^64 over the first `bytes` bytes of each slot (little-endian words, the last one
// zero-extended): expert_nibble.hpp's hash_host.  A sum, so the blocks' order of adding does not change it.
__global__ void __launch_bounds__(256) nib_hash_kernel(const unsigned long long* __restrict__ slots, uint64_t bytes,
                                                       unsigned long long* __restrict__ out) {
    const uint8_t* p = (const uint8_t*) slots[blockIdx.y];
    const uint64_t n16 = bytes / 16, nw = (bytes + 3) / 4;
    uint64_t h = 0;
    for (uint64_t q = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x; q < n16; q += (uint64_t) gridDim.x * blockDim.x) {
        const uint4 v = ((const uint4*) p)[q];
        const uint32_t i = (uint32_t) (4 * q);
        h += (uint64_t) (v.x ^ i) * nibk::kHashMul + (uint64_t) (v.y ^ (i + 1)) * nibk::kHashMul +
             (uint64_t) (v.z ^ (i + 2)) * nibk::kHashMul + (uint64_t) (v.w ^ (i + 3)) * nibk::kHashMul;
    }
    if (blockIdx.x == 0)
        for (uint64_t i = 4 * n16 + threadIdx.x; i < nw; i += blockDim.x) {
            uint32_t w = 0;
            for (uint64_t b = 4 * i; b < 4 * i + 4 && b < bytes; ++b) w |= (uint32_t) p[b] << (8 * (b - 4 * i));
            h += (uint64_t) (w ^ (uint32_t) i) * nibk::kHashMul;
        }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) h += __shfl_xor_sync(0xffffffffu, h, o);
    __shared__ uint64_t s_h[8];
    if ((threadIdx.x & 31) == 0) s_h[threadIdx.x >> 5] = h;
    __syncthreads();
    if (threadIdx.x == 0) {
        uint64_t t = 0;
        for (int w = 0; w < (int) (blockDim.x >> 5); ++w) t += s_h[w];
        atomicAdd(out + blockIdx.y, (unsigned long long) t);
    }
}
#endif  // STRATA_EXPERT_NIBBLE_BUILT

// STRATA_EXPERT_NIBBLE: whether a layout's gate/up is a converted slot's 4-bit codes (constant false in a build
// without the kernels, so its dispatch below is today's code)
constexpr bool nib_layout(const NativeExpertLayout& L) {
#if STRATA_EXPERT_NIBBLE_BUILT
    return L.gu_nib;
#else
    static_cast<void>(L);
    return false;
#endif
}

}  // namespace

void iq_set_old_kernels(bool old) { g_old_kernels = old; }
bool iq_old_kernels() { return g_old_kernels; }

bool iq_supported(int t) noexcept { return is_iq(t); }
// BF16 and Q6_K: embedding only (dq_dispatch has them; no expert or prompt-path kernels take them)
bool embed_type_supported(int t) noexcept { return is_iq(t) || t == 30 || t == 14; }

size_t iq_row_bytes(int t, int64_t n) noexcept {
    switch (t) {
        case 16: return (size_t) (n / 256) * sizeof(block_iq2_xxs);
        case 17: return (size_t) (n / 256) * sizeof(block_iq2_xs);
        case 18: return (size_t) (n / 256) * sizeof(block_iq3_xxs);
        case 20: return (size_t) (n / 32) * sizeof(block_iq4_nl);
        case 21: return (size_t) (n / 256) * sizeof(block_iq3_s);
        case 22: return (size_t) (n / 256) * sizeof(block_iq2_s);
        case 29: return (size_t) (n / 256) * sizeof(block_iq1_m);
        case 23: return (size_t) (n / 256) * sizeof(block_iq4_xs);
        case 11: return (size_t) (n / 256) * sizeof(block_q3_K);
        case 42: return (size_t) (n / 64) * sizeof(block_q2_0);
        case 12: return (size_t) (n / 256) * sizeof(block_q4_K);
        case 13: return (size_t) (n / 256) * sizeof(block_q5_K);
        case 14: return (size_t) (n / 256) * sizeof(block_q6_K);
        case 7: return (size_t) (n / 32) * sizeof(block_q5_1);
        case 6: return (size_t) (n / 32) * sizeof(block_q5_0);
        case 2: return (size_t) (n / 32) * sizeof(block_q4_0);
        case 3: return (size_t) (n / 32) * sizeof(block_q4_1);
        case 40: return (size_t) (n / 64) * sizeof(block_nvfp4);
        case 8: return (size_t) (n / 32) * sizeof(block_q8_0);
        case 30: return (size_t) n * 2;   // BF16: the token embedding only (iq_embed_rows, iq_dequant_f32)
        case 144: return (size_t) (n / 128) * sizeof(block_tq2_t);   // APR trellis types
        case 145: return (size_t) (n / 128) * sizeof(block_tqk6);
        case 146: return (size_t) (n / 128) * sizeof(block_tqk7);
        case 147: return (size_t) (n / 128) * sizeof(block_tqp5);
        case 148: return (size_t) (n / 128) * sizeof(block_tqp6);
        case 149: return (size_t) (n / 128) * sizeof(block_tqp7);
        default: return 0;
    }
}

void quantize_q8_1_rows(const float* x, int64_t n_rows, int64_t n_cols, void* y, void* stream) {
    const long long n = (long long) n_rows * n_cols;
    if (n <= 0) return;
    quantize_q8_1_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(x, (block_q8_1*) y, n);
    check("quantize_q8_1_rows");
}

void iq_mmvq(int t, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    const size_t rb = iq_row_bytes(t, n_in);
    cudaStream_t s = (cudaStream_t) stream;
    const auto* W = (const uint8_t*) w;
    const auto* X = (const block_q8_1*) x_q8_1;
    switch (t) {
#define STRATA_MMVQ(T) case T: launch_mmvq<T>(W, rb, X, y, n_in, n_out, ncols, s); break;
        STRATA_MMVQ_FMTS(STRATA_MMVQ)
#undef STRATA_MMVQ
        default: std::fprintf(stderr, "iq_mmvq: type %d is not supported\n", t); std::exit(1);
    }
    check("iq_mmvq");
}

void iq_dequant_f16(int t, const void* src, int64_t n, uint16_t* dst, void* stream, const float* scale) {
    if (n % 256 != 0 || !is_iq(t)) { std::fprintf(stderr, "iq_dequant_f16: bad arguments\n"); std::exit(1); }
    dequant_flat_kernel<__half><<<(unsigned) (n / 256), 32, 0, (cudaStream_t) stream>>>(t, src, (__half*) dst, scale);
    check("iq_dequant_f16");
}

namespace {
__global__ void embed_rows_kernel(int ty, const uint8_t* __restrict__ table, size_t row_bytes,
                                  const int32_t* __restrict__ tokens, int64_t n_embd, float* __restrict__ y) {
    const int t = blockIdx.y;
    const int64_t b = blockIdx.x;
    const uint8_t* row = table + (size_t) tokens[t] * row_bytes;
    dq_dispatch<float>(ty, row, b, y + (size_t) t * n_embd + b * QK_K, threadIdx.x);
}
}  // namespace

void iq_embed_rows(int t, const void* table, size_t row_bytes, const int32_t* tokens, int64_t n_tok, int64_t n_embd,
                   float* out, void* stream) {
    if (n_tok <= 0) return;
    if (n_embd % 256 != 0 || !embed_type_supported(t)) { std::fprintf(stderr, "iq_embed_rows: bad arguments\n"); std::exit(1); }
    embed_rows_kernel<<<dim3((unsigned) (n_embd / 256), (unsigned) n_tok), 32, 0, (cudaStream_t) stream>>>(
        t, (const uint8_t*) table, row_bytes, tokens, n_embd, out);
    check("iq_embed_rows");
}

void iq_dequant_f32(int t, const void* src, int64_t n, float* dst, void* stream) {
    if (n % 256 != 0 || !embed_type_supported(t)) { std::fprintf(stderr, "iq_dequant_f32: bad arguments\n"); std::exit(1); }
    dequant_flat_kernel<float><<<(unsigned) (n / 256), 32, 0, (cudaStream_t) stream>>>(t, src, dst, nullptr);
    check("iq_dequant_f32");
}

void iq_dequant_gu_f16(int t, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* dst, void* stream,
                       const float* tail) {
    // checked like the other entry points: an unknown type used to leave `dst` unwritten, a wrong prompt and no error
    if (n_embd % 256 != 0 || !is_iq(t)) { std::fprintf(stderr, "iq_dequant_gu_f16: type %d / %lld\n", t, (long long) n_embd); std::exit(1); }
    const int64_t per_row = n_embd / 256;
    dequant_gu_kernel<<<dim3((unsigned) (n_ff * per_row), 2), 32, 0, (cudaStream_t) stream>>>(t, gate, up, per_row,
                                                                                           (__half*) dst, tail);
    check("iq_dequant_gu_f16");
}

// A bad block is a pack or caller error (expert_layout_load already refuses a block that does not divide the expert
// widths): it exits with a message, like check() does for a failed launch.
namespace {
void hadamard_bad_block(const char* what, int block, int64_t width) {
    std::fprintf(stderr, "%s: Hadamard block %d is not a power of two in [2, 1024] dividing the row width %lld "
                         "(hadamard.txt's block and the expert widths disagree)\n", what, block, (long long) width);
    std::exit(1);
}
constexpr int64_t kHadamardMaxRows = 65535;   // gridDim.y
bool hadamard_block_ok(int block, int64_t width) {
    return block >= 2 && block <= 1024 && (block & (block - 1)) == 0 && width % block == 0;
}
}  // namespace

void hadamard_rows(const float* x, float* y, int64_t rows, int64_t width, int block, const float* signs, void* stream) {
    if (rows <= 0) return;
    if (!hadamard_block_ok(block, width)) hadamard_bad_block("hadamard_rows", block, width);
    // gridDim.y is limited to 65535: a prompt chunk's T * K rows (16K tokens x 10 experts) go up in slices
    for (int64_t r0 = 0; r0 < rows; r0 += kHadamardMaxRows) {
        const int64_t n = std::min<int64_t>(kHadamardMaxRows, rows - r0);
        hadamard_rows_kernel<float><<<dim3((unsigned) (width / block), (unsigned) n), block, block * sizeof(float),
                                      (cudaStream_t) stream>>>(x + r0 * width, y + r0 * width, width, signs,
                                                               1.0f / std::sqrt((float) block));
    }
    check("hadamard_rows");
}

void hadamard_rows_f16(uint16_t* x, int64_t rows, int64_t width, int block, const float* signs, void* stream) {
    if (rows <= 0) return;
    if (!hadamard_block_ok(block, width)) hadamard_bad_block("hadamard_rows_f16", block, width);
    for (int64_t r0 = 0; r0 < rows; r0 += kHadamardMaxRows) {   // gridDim.y <= 65535, as in hadamard_rows
        const int64_t n = std::min<int64_t>(kHadamardMaxRows, rows - r0);
        hadamard_rows_kernel<__half><<<dim3((unsigned) (width / block), (unsigned) n), block, block * sizeof(float),
                                       (cudaStream_t) stream>>>((const __half*) x + r0 * width, (__half*) x + r0 * width,
                                                                width, signs, 1.0f / std::sqrt((float) block));
    }
    check("hadamard_rows_f16");
}

const float* hadamard_device_signs(const float* host, int width) {
    if (host == nullptr || width <= 0) return nullptr;
    static std::mutex mu;
    static std::map<std::tuple<int, const float*, int>, float*> table;   // (device, host vector, width)
    int dev = 0;
    cudaGetDevice(&dev);
    std::lock_guard<std::mutex> lock(mu);
    float*& d = table[std::make_tuple(dev, host, width)];
    if (d == nullptr) {
        if (cudaMalloc((void**) &d, (size_t) width * sizeof(float)) != cudaSuccess ||
            cudaMemcpy(d, host, (size_t) width * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            std::fprintf(stderr, "hadamard_device_signs: %s\n", cudaGetErrorString(cudaGetLastError()));
            std::exit(1);
        }
    }
    return d;
}

// Q3_K shared column scales: x = diag(1/s) x row-wise, the ONE multiply every expert-activation consumer sees
// (hadamard_rows above is the structural template: one row per gridDim.y, width split over gridDim.x, the
// kHadamardMaxRows slices keep gridDim.y <= 65535).
__global__ void lat_colscale_rows_kernel(float* __restrict__ x, int width, const float* __restrict__ inv) {
    const int c = (int) blockIdx.x * blockDim.x + threadIdx.x;   // one column per lane, one row per blockIdx.y
    if (c < width) x[(int64_t) blockIdx.y * width + c] *= inv[c];
}

void lat_colscale_rows(float* x, int64_t rows, int width, const float* inv, void* stream) {
    if (rows <= 0 || inv == nullptr) return;
    const int block = 128;
    for (int64_t r0 = 0; r0 < rows; r0 += kHadamardMaxRows) {   // gridDim.y <= 65535, as in hadamard_rows
        const int64_t n = std::min<int64_t>(kHadamardMaxRows, rows - r0);
        lat_colscale_rows_kernel<<<dim3((unsigned) ((width + block - 1) / block), (unsigned) n), block, 0,
                                   (cudaStream_t) stream>>>(x + r0 * width, width, inv);
    }
    check("lat_colscale_rows");
}

const float* lat_gu_colscale_dev(int layer, int width) {
    int stored = 0;
    const float* host = lat_gu_colscale_host(layer, &stored);
    if (host == nullptr || width <= 0) return nullptr;
    if (stored != width) {   // the manifest's n_embd is the truth; a mismatched caller would read OOB host memory
        std::fprintf(stderr, "lat_gu_colscale_dev: layer %d stores %d values, the caller wants width %d\n",
                     layer, stored, width);
        return nullptr;
    }
    static std::mutex mu;
    static std::map<std::tuple<int, int, const float*>, float*> table;   // (device, layer, host vector)
    int dev = 0;
    cudaGetDevice(&dev);
    std::lock_guard<std::mutex> lock(mu);
    float*& d = table[std::make_tuple(dev, layer, host)];
    if (d == nullptr) {
        if (cudaMalloc((void**) &d, (size_t) width * sizeof(float)) != cudaSuccess ||
            cudaMemcpy(d, host, (size_t) width * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            std::fprintf(stderr, "lat_gu_colscale_dev: %s\n", cudaGetErrorString(cudaGetLastError()));
            std::exit(1);
        }
    }
    return d;
}

bool native_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept {
    const int qg = gu_qk(gu_type), qd = d_qk(d_type);
    // a type without row bytes would make every row (and the up/down offsets) 0: refused, not run wrong
    return qg > 0 && qd > 0 && iq_row_bytes(gu_type, n_embd) > 0 && iq_row_bytes(d_type, n_ff) > 0 && is_iq(gu_type) && is_iq(d_type) && n_embd % qg == 0 && n_ff % qd == 0 &&
           n_embd % 256 == 0 && (n_ff * n_embd) % 256 == 0;
}

NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) {
    NativeExpertLayout L;
    L.gu_type = gu_type;
    L.d_type = d_type;
    L.n_embd = n_embd;
    L.n_ff = n_ff;
    L.gu_row = iq_row_bytes(gu_type, n_embd);
    L.d_row = iq_row_bytes(d_type, n_ff);
    L.up_off = (size_t) n_ff * L.gu_row;
    L.down_off = 2 * L.up_off;
    L.bytes = L.down_off + (size_t) n_embd * L.d_row;
    // NVFP4: {s_gate, s_up, s_down, 0} after down (tools/iq_pack.py) whenever any projection is NVFP4; a mixed
    // expert's other projection has 1.0 there, so the kernels below multiply by it unconditionally
    if (gu_type == 40 || d_type == 40) {
        L.tail_off = L.bytes;
        L.bytes += 16;
    }
    return L;
}

NativeExpertLayout native_expert_layout_nib(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) {
    namespace nb = strata::core::nibble;
    NativeExpertLayout L = native_expert_layout(gu_type, d_type, n_embd, n_ff);
    L.gu_nib = true;
    L.gu_row = (size_t) (nb::kItems * nb::kItemBytes);   // a row's codes (the scales and d follow all the codes)
    L.up_off = nb::kUpCodesOff;
    L.down_off = nb::kDownOff;
    L.bytes = nb::slot_bytes_for((uint64_t) n_embd * L.d_row);
    return L;
}

bool native_expert_nib_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept {
#if STRATA_EXPERT_NIBBLE_BUILT
    namespace nb = strata::core::nibble;
    return nb::fmt_built(gu_type) && n_embd == nb::kNEmbd && n_ff == nb::kNFf && d_type != 40 &&
           native_expert_supported(gu_type, d_type, n_embd, n_ff);
#else
    (void) gu_type; (void) d_type; (void) n_embd; (void) n_ff;
    return false;
#endif
}

void native_expert_nib_set_variant(int v) {
#if STRATA_EXPERT_NIBBLE_BUILT
    g_nib_variant = v;
#else
    (void) v;
#endif
}

bool native_expert_nib_variant_attr(int v, int gu_type, int* regs, int* local_bytes, int* blocks_per_sm) {
#if STRATA_EXPERT_NIBBLE_BUILT
    cudaFuncAttributes a{};
    if (!nibx_launch(gu_type, v, dim3(80, 1), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &a)) return false;
    *regs = a.numRegs;
    *local_bytes = (int) a.localSizeBytes;
    *blocks_per_sm = g_nibx_occ;
    return true;
#else
    (void) v; (void) gu_type; (void) regs; (void) local_bytes; (void) blocks_per_sm;
    return false;
#endif
}

size_t native_expert_scratch_bytes(int64_t cap, int64_t n_ff) {
    const size_t f = (size_t) cap * (size_t) n_ff * sizeof(float);
    return 3 * ((f + 255) & ~(size_t) 255) + (((size_t) cap * (size_t) (n_ff / 32) * sizeof(block_q8_1) + 255) & ~(size_t) 255);
}


// The AMD layouts below (STRATA_EXP_MODE) also build for CUDA: Volta takes mode 8 by default (exp_mode), every
// other CUDA card keeps mode 0, the CUDA layout above.
#if 1  // dflash2-kern: the layouts build for every HIP arch too (default mode stays 0 off gfx906; STRATA_EXP_MODE picks one)
#define STRATA_EXP_LAYOUTS 1
#else
#define STRATA_EXP_LAYOUTS 0
#endif
#if defined(STRATA_HIP_GFX906)
constexpr int kExpModeDefault = 7;
constexpr int kExpFallback = 2;   // modes 5-8 on a format without an LDS kernel: R2
constexpr bool kExpLds16 = false; // IQ2_XXS gate/up: R2 as measured on gfx906
#else
constexpr int kExpModeDefault = 0;
constexpr int kExpFallback = 0;   // ... the CUDA layout (V100, IQ2_XXS gate/up: R2 183 us vs 113)
constexpr bool kExpLds16 = true;  // IQ2_XXS gate/up in LDS too
#endif
#if STRATA_EXP_LAYOUTS
// ---- AMD layouts for the grouped native experts (STRATA_EXP_MODE; 0 = the CUDA one above).
// 1 (W64): a row per 64-lane wavefront - 4x the wavefronts, ~1-2 calls per lane, a 64-lane butterfly.
// 2 (R2):  a 32-lane logical warp computes TWO rows in one loop - two independent load chains in flight.
// Either way a (row, entry) sum does not depend on the window size.
template<int TY>
__device__ __forceinline__ float row_dot64(const uint8_t* row, const block_q8_1* x, int nb, int lane) {
    using F = Fmt<TY>;
    float s = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += 64) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        s += F::dot(row, x + kbx * (F::qk / 32), kbx, iqs);
    }
#pragma unroll
#if defined(__HIPCC__)
    for (int o = 32; o > 0; o >>= 1) s += __shfl_xor(s, o, 64);
#else
    __trap();   // a 64-lane wavefront: mode 1 is gfx906's only
#endif
    return s;
}
template<int TY>
__device__ __forceinline__ void row_dot2(const uint8_t* r0, const uint8_t* r1, const block_q8_1* x, int nb, int lane,
                                         float& o0, float& o1) {
    using F = Fmt<TY>;
    float s0 = 0.0f, s1 = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const block_q8_1* xk = x + kbx * (F::qk / 32);
        const float a = F::dot(r0, xk, kbx, iqs);
        const float b = F::dot(r1, xk, kbx, iqs);
        s0 += a;
        s1 += b;
    }
    o0 = warp_sum(s0);
    o1 = warp_sum(s1);
}
template<int TY, int NR>
__device__ __forceinline__ void row_dotn(const uint8_t* const* r, const block_q8_1* x, int nb, int lane, float* o) {
    using F = Fmt<TY>;
    float acc[NR];
#pragma unroll
    for (int i = 0; i < NR; ++i) acc[i] = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const block_q8_1* xk = x + kbx * (F::qk / 32);
        float d[NR];
#pragma unroll
        for (int i = 0; i < NR; ++i) d[i] = F::dot(r[i], xk, kbx, iqs);
#pragma unroll
        for (int i = 0; i < NR; ++i) acc[i] += d[i];
    }
#pragma unroll
    for (int i = 0; i < NR; ++i) o[i] = warp_sum(acc[i]);
}
template<int TG, int MODE>
__global__ void __launch_bounds__(256) native_gu_amd_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                            const int32_t* __restrict__ grp_start,
                                                            const int32_t* __restrict__ n_groups,
                                                            const int32_t* __restrict__ ent_tok,
                                                            const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                            float* __restrict__ gate, float* __restrict__ up) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const int nb = (int) (L.n_embd / Fmt<TG>::qk), xb = (int) (L.n_embd / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    const int nrow = 2 * (int) L.n_ff;
    auto wrow = [&](int row) -> const uint8_t* {
        const bool is_up = row >= L.n_ff;
        return blob + (is_up ? L.up_off : 0) + (size_t) (is_up ? row - (int) L.n_ff : row) * L.gu_row;
    };
    auto put = [&](int row, int e, float v) {
        const bool is_up = row >= L.n_ff;
        (is_up ? up : gate)[(size_t) e * L.n_ff + (is_up ? row - (int) L.n_ff : row)] = v;
    };
    if constexpr (MODE == 1) {
        const int lane = threadIdx.x & 63, row = blockIdx.x * 4 + (threadIdx.x >> 6);
        if (row >= nrow) return;
        const uint8_t* wr = wrow(row);
        for (int e = e0; e < e1; ++e) {
            const float v = row_dot64<TG>(wr, xq + (size_t) ent_tok[e] * xb, nb, lane);
            if (lane == 0) put(row, e, v);
        }
    } else if constexpr (MODE == 4) {
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        const int row0 = blockIdx.x * 32 + warp;
        if (row0 >= nrow) return;
        const uint8_t* w[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) w[i] = wrow(row0 + 8 * i < nrow ? row0 + 8 * i : row0);
        for (int e = e0; e < e1; ++e) {
            float o[4];
            row_dotn<TG, 4>(w, xq + (size_t) ent_tok[e] * xb, nb, lane, o);
            if (lane == 0)
#pragma unroll
                for (int i = 0; i < 4; ++i) if (row0 + 8 * i < nrow) put(row0 + 8 * i, e, o[i]);
        }
    } else {
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        const int row0 = blockIdx.x * 16 + warp, row1 = row0 + 8;
        if (row0 >= nrow) return;
        const bool two = row1 < nrow;
        const uint8_t* w0 = wrow(row0);
        const uint8_t* w1 = wrow(two ? row1 : row0);
        for (int e = e0; e < e1; ++e) {
            float a, b;
            row_dot2<TG>(w0, w1, xq + (size_t) ent_tok[e] * xb, nb, lane, a, b);
            if (lane == 0) { put(row0, e, a); if (two) put(row1, e, b); }
        }
    }
}
template<int TD, int MODE>
__global__ void __launch_bounds__(256) native_down_amd_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                              const int32_t* __restrict__ grp_start,
                                                              const int32_t* __restrict__ n_groups,
                                                              const int32_t* __restrict__ ent_dst,
                                                              const block_q8_1* __restrict__ hq, NativeExpertLayout L,
                                                              float* __restrict__ out) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const int nb = (int) (L.n_ff / Fmt<TD>::qk), hb = (int) (L.n_ff / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    const int nrow = (int) L.n_embd;
    if constexpr (MODE == 1) {
        const int lane = threadIdx.x & 63, r = blockIdx.x * 4 + (threadIdx.x >> 6);
        if (r >= nrow) return;
        const uint8_t* wr = blob + L.down_off + (size_t) r * L.d_row;
        for (int e = e0; e < e1; ++e) {
            const float v = row_dot64<TD>(wr, hq + (size_t) e * hb, nb, lane);
            if (lane == 0) out[(size_t) ent_dst[e] * L.n_embd + r] = v;
        }
    } else if constexpr (MODE == 4) {
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        const int r0 = blockIdx.x * 32 + warp;
        if (r0 >= nrow) return;
        const uint8_t* w[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) w[i] = blob + L.down_off + (size_t) (r0 + 8 * i < nrow ? r0 + 8 * i : r0) * L.d_row;
        for (int e = e0; e < e1; ++e) {
            float o[4];
            row_dotn<TD, 4>(w, hq + (size_t) e * hb, nb, lane, o);
            if (lane == 0)
#pragma unroll
                for (int i = 0; i < 4; ++i) if (r0 + 8 * i < nrow) out[(size_t) ent_dst[e] * L.n_embd + r0 + 8 * i] = o[i];
        }
    } else {
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        const int r0 = blockIdx.x * 16 + warp, r1 = r0 + 8;
        if (r0 >= nrow) return;
        const bool two = r1 < nrow;
        const uint8_t* w0 = blob + L.down_off + (size_t) r0 * L.d_row;
        const uint8_t* w1 = blob + L.down_off + (size_t) (two ? r1 : r0) * L.d_row;
        for (int e = e0; e < e1; ++e) {
            float a, b;
            row_dot2<TD>(w0, w1, hq + (size_t) e * hb, nb, lane, a, b);
            if (lane == 0) {
                out[(size_t) ent_dst[e] * L.n_embd + r0] = a;
                if (two) out[(size_t) ent_dst[e] * L.n_embd + r1] = b;
            }
        }
    }
}
// ---- 5 (LDS): gate/up of the grid formats (IQ2_S, IQ3_XXS, IQ3_S) with the codebook grid and the group's q8_1
// activations in LDS.  The bench showed gate/up compute-bound, not bandwidth-bound (T = 2 costs ~1.8x T = 1 on the
// same weights, 110-130 GB/s): each call issues its grid lookups and eight activation loads through the vector
// memory path after the weight load.  Same calls, same lane-strided k, same R2 row pairs, same warp_sum: the
// output is bitwise the mode-2 one.
template<int TY> struct GridOf;
template<> struct GridOf<22> { using T = uint64_t; static constexpr int N = 1024; __device__ static const T* src() { return iq2s_grid; } };
template<> struct GridOf<16> { using T = uint64_t; static constexpr int N = 256; __device__ static const T* src() { return iq2xxs_grid; } };
template<> struct GridOf<18> { using T = uint32_t; static constexpr int N = 256; __device__ static const T* src() { return iq3xxs_grid; } };
template<> struct GridOf<21> { using T = uint32_t; static constexpr int N = 512; __device__ static const T* src() { return iq3s_grid; } };
template<> struct GridOf<23> { using T = uint32_t; static constexpr int N = 1; __device__ static const T* src() { return iq3s_grid; } };   // IQ4_XS: no grid

// SG = 1 (mode 6): the signs without byte-SIMD ops, which gfx906 lacks (strata_vcmpne4 / strata_vsub4 unpack to
// ~12 word ops each).  m = the sign nibble as 0x00/0xff bytes; (g ^ m) is g or -g-1 per byte, so
// dp4a(g ^ m, u) - dp4a(m, u) = sum(s g u) exactly - the same integers, the same floats.
__device__ __forceinline__ int nib_mask(uint32_t n) {
    const uint32_t b = __umul24(n & 0xFu, 0x204081u) & 0x01010101u;
    return (int) ((b << 8) - b);
}
template<int TY, int SG> struct DotG;
template<int SG> struct DotG<22, SG> { __device__ static __forceinline__ float f(const void* vbq, const block_q8_1* bq8_1, int kbx, int iqs,
                                                      const uint64_t* grid) {
    const block_iq2_s* bq2 = (const block_iq2_s*) vbq + kbx;
    const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq2->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int* grid_pos = (const int*) (grid + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        int& acc = l0 < 4 ? sumi0 : sumi1;
        if constexpr (SG == 1) {
            const int m0 = nib_mask(signs_packed_8[l0 / 2]), m1 = nib_mask(signs_packed_8[l0 / 2] >> 4);
            acc = ggml_cuda_dp4a(grid_pos[0] ^ m0, u0, acc);
            acc = ggml_cuda_dp4a(grid_pos[1] ^ m1, u1, acc);
            acc -= ggml_cuda_dp4a(m1, u1, ggml_cuda_dp4a(m0, u0, 0));
        } else {
            const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
            const int grid_l = __vsub4(grid_pos[0] ^ signs0, signs0);
            const int grid_h = __vsub4(grid_pos[1] ^ signs1, signs1);
            acc = ggml_cuda_dp4a(grid_l, u0, acc);
            acc = ggml_cuda_dp4a(grid_h, u1, acc);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
} };
template<int SG> struct DotG<18, SG> { __device__ static __forceinline__ float f(const void* vbq, const block_q8_1* bq8_1, int kbx, int iqs,
                                                      const uint32_t* grid) {
    const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* q3 = (const uint8_t*) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(grid[q3[l0 + 0]], grid[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if constexpr (SG == 1) {
            const int m0 = nib_mask(signs), m1 = nib_mask(signs >> 4);
            sumi = ggml_cuda_dp4a(grid_pos.x ^ m0, u0, sumi);
            sumi = ggml_cuda_dp4a(grid_pos.y ^ m1, u1, sumi);
            sumi -= ggml_cuda_dp4a(m1, u1, ggml_cuda_dp4a(m0, u0, 0));
        } else {
            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
            sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
            sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
        }
    }
    const int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 2;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
} };
template<int SG> struct DotG<21, SG> { __device__ static __forceinline__ float f(const void* vbq, const block_q8_1* bq8_1, int kbx, int iqs,
                                                      const uint32_t* grid) {
    const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
    const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq3->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                                        grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if constexpr (SG == 1) {
            const int m0 = nib_mask(signs_packed_8[l0 / 2]), m1 = nib_mask(signs_packed_8[l0 / 2] >> 4);
            sumi = ggml_cuda_dp4a(grid_pos.x ^ m0, u0, sumi);
            sumi = ggml_cuda_dp4a(grid_pos.y ^ m1, u1, sumi);
            sumi -= ggml_cuda_dp4a(m1, u1, ggml_cuda_dp4a(m0, u0, 0));
        } else {
            const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
            const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
            const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
            sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
            sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
        }
    }
    sumi *= 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
} };

// IQ2_XXS: vec_dot_iq2_xxs_q8_1 with the grid in LDS - the same integers, the same floats
template<int SG> struct DotG<16, SG> { __device__ static __forceinline__ float f(const void* vbq, const block_q8_1* bq8_1, int kbx, int iqs,
                                                      const uint64_t* grid) {
    const block_iq2_xxs* bq2 = (const block_iq2_xxs*) vbq + kbx;
    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t* aux8 = (const uint8_t*) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = *(const uint2*) (grid + aux8[k0 / 2]);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 1);
        if constexpr (SG == 1) {
            const uint32_t v = (aux32 >> (7 * k0 / 2)) & 0x7F, sg = v | ((__popc(v) & 1) << 7);
            const int m0 = nib_mask(sg), m1 = nib_mask(sg >> 4);
            sumi = ggml_cuda_dp4a((int) grid_pos.x ^ m0, u0, sumi);
            sumi = ggml_cuda_dp4a((int) grid_pos.y ^ m1, u1, sumi);
            sumi -= ggml_cuda_dp4a(m1, u1, ggml_cuda_dp4a(m0, u0, 0));
        } else {
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            sumi = ggml_cuda_dp4a((int) __vsub4(grid_pos.x ^ signs0, signs0), u0, sumi);
            sumi = ggml_cuda_dp4a((int) __vsub4(grid_pos.y ^ signs1, signs1), u1, sumi);
        }
    }
    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
} };

template<int SG> struct DotG<23, SG> { __device__ static __forceinline__ float f(const void* vbq, const block_q8_1* bq8_1, int kbx, int iqs,
                                                      const uint32_t*) { return vec_dot_iq4_xs_q8_1(vbq, bq8_1, kbx, iqs); } };

constexpr int LDS_RB = 64;    // gate/up rows per block (4 R2 passes of 16)
constexpr int LDS_NT = 4;     // entries whose activations sit in LDS at once

template<int TG, int SG, bool TI = false>
__global__ void __launch_bounds__(256) native_gu_lds_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                            const int32_t* __restrict__ grp_start,
                                                            const int32_t* __restrict__ n_groups,
                                                            const int32_t* __restrict__ ent_tok,
                                                            const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                            float* __restrict__ gate, float* __restrict__ up) {
    using GT = typename GridOf<TG>::T;
    using F = Fmt<TG>;
    __shared__ GT sgrid[GridOf<TG>::N];
    extern __shared__ int sx_raw[];   // LDS_NT tokens x xb blocks of q8_1 (36 bytes = 9 ints each)
    const int g = blockIdx.y;
    if (g >= *n_groups) return;       // uniform over the block
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const GT* gsrc = GridOf<TG>::src();
    for (int i = tid; i < GridOf<TG>::N; i += 256) sgrid[i] = gsrc[i];
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const int nb = (int) (L.n_embd / F::qk), xb = (int) (L.n_embd / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    const int nrow = 2 * (int) L.n_ff;
    const block_q8_1* sx = (const block_q8_1*) sx_raw;
    auto wrow = [&](int row) -> const uint8_t* {
        const bool is_up = row >= L.n_ff;
        return blob + (is_up ? L.up_off : 0) + (size_t) (is_up ? row - (int) L.n_ff : row) * L.gu_row;
    };
    auto put = [&](int row, int e, float v) {
        const bool is_up = row >= L.n_ff;
        (is_up ? up : gate)[(size_t) e * L.n_ff + (is_up ? row - (int) L.n_ff : row)] = v;
    };
    for (int c0 = e0; c0 < e1; c0 += LDS_NT) {
        const int cn = min(LDS_NT, e1 - c0);
        __syncthreads();
        for (int j = 0; j < cn; ++j) {
            const int* src = (const int*) (xq + (size_t) ent_tok[c0 + j] * xb);
            for (int i = tid; i < xb * 9; i += 256) sx_raw[j * xb * 9 + i] = src[i];
        }
        __syncthreads();
        for (int p = 0; p < LDS_RB / 16; ++p) {
            const int row0 = blockIdx.x * LDS_RB + p * 16 + warp, row1 = row0 + 8;
            if (row0 >= nrow) break;
            const bool two = row1 < nrow;
            const uint8_t* w0 = wrow(row0);
            const uint8_t* w1 = wrow(two ? row1 : row0);
            if constexpr (TI) {   // mode 7: the tokens inside the k loop - one weight load per k for all of them
                auto pass = [&](auto ntc) {
                    constexpr int NTC = decltype(ntc)::value;
                    float s0[NTC], s1[NTC];
#pragma unroll
                    for (int j = 0; j < NTC; ++j) { s0[j] = 0.0f; s1[j] = 0.0f; }
                    for (int k = lane; k < nb * F::ipb; k += 32) {
                        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
#pragma unroll
                        for (int j = 0; j < NTC; ++j) {
                            const block_q8_1* xk = sx + (size_t) j * xb + kbx * (F::qk / 32);
                            const float a = DotG<TG, SG>::f(w0, xk, kbx, iqs, sgrid);
                            const float b = DotG<TG, SG>::f(w1, xk, kbx, iqs, sgrid);
                            s0[j] += a;
                            s1[j] += b;
                        }
                    }
#pragma unroll
                    for (int j = 0; j < NTC; ++j) {
                        const float a = warp_sum(s0[j]), b = warp_sum(s1[j]);
                        if (lane == 0) { put(row0, c0 + j, a); if (two) put(row1, c0 + j, b); }
                    }
                };
                switch (cn) {
                    case 1: pass(std::integral_constant<int, 1>{}); break;
                    case 2: pass(std::integral_constant<int, 2>{}); break;
                    case 3: pass(std::integral_constant<int, 3>{}); break;
                    default: pass(std::integral_constant<int, 4>{}); break;
                }
                continue;
            }
            for (int j = 0; j < cn; ++j) {
                const block_q8_1* x = sx + (size_t) j * xb;
                float s0 = 0.0f, s1 = 0.0f;
                for (int k = lane; k < nb * F::ipb; k += 32) {
                    const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
                    const block_q8_1* xk = x + kbx * (F::qk / 32);
                    const float a = DotG<TG, SG>::f(w0, xk, kbx, iqs, sgrid);
                    const float b = DotG<TG, SG>::f(w1, xk, kbx, iqs, sgrid);
                    s0 += a;
                    s1 += b;
                }
                s0 = warp_sum(s0);
                s1 = warp_sum(s1);
                if (lane == 0) { put(row0, c0 + j, s0); if (two) put(row1, c0 + j, s1); }
            }
        }
    }
}

// ---- 7 (down): the group's q8_1 h rows in LDS, the tokens inside the k loop, the R2 row pairs of mode 2 -
// the same per-(row, entry) sums in the same order: bitwise the mode-2 output.
template<int TD>
__global__ void __launch_bounds__(256) native_down_lds_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                              const int32_t* __restrict__ grp_start,
                                                              const int32_t* __restrict__ n_groups,
                                                              const int32_t* __restrict__ ent_dst,
                                                              const block_q8_1* __restrict__ hq, NativeExpertLayout L,
                                                              float* __restrict__ out) {
    using F = Fmt<TD>;
    extern __shared__ int sh_raw[];   // LDS_NT entries x hb blocks of q8_1
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const int nb = (int) (L.n_ff / F::qk), hb = (int) (L.n_ff / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    const int nrow = (int) L.n_embd;
    const block_q8_1* sh = (const block_q8_1*) sh_raw;
    for (int c0 = e0; c0 < e1; c0 += LDS_NT) {
        const int cn = min(LDS_NT, e1 - c0);
        __syncthreads();
        {
            const int* src = (const int*) (hq + (size_t) c0 * hb);   // the chunk's entries are contiguous
            for (int i = tid; i < cn * hb * 9; i += 256) sh_raw[i] = src[i];
        }
        __syncthreads();
        for (int p = 0; p < LDS_RB / 16; ++p) {
            const int r0 = blockIdx.x * LDS_RB + p * 16 + warp, r1 = r0 + 8;
            if (r0 >= nrow) break;
            const bool two = r1 < nrow;
            const uint8_t* w0 = blob + L.down_off + (size_t) r0 * L.d_row;
            const uint8_t* w1 = blob + L.down_off + (size_t) (two ? r1 : r0) * L.d_row;
            auto pass = [&](auto ntc) {
                constexpr int NTC = decltype(ntc)::value;
                float s0[NTC], s1[NTC];
#pragma unroll
                for (int j = 0; j < NTC; ++j) { s0[j] = 0.0f; s1[j] = 0.0f; }
                for (int k = lane; k < nb * F::ipb; k += 32) {
                    const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
#pragma unroll
                    for (int j = 0; j < NTC; ++j) {
                        const block_q8_1* xk = sh + (size_t) j * hb + kbx * (F::qk / 32);
                        const float a = F::dot(w0, xk, kbx, iqs);
                        const float b = F::dot(w1, xk, kbx, iqs);
                        s0[j] += a;
                        s1[j] += b;
                    }
                }
#pragma unroll
                for (int j = 0; j < NTC; ++j) {
                    const float a = warp_sum(s0[j]), b = warp_sum(s1[j]);
                    if (lane == 0) {
                        out[(size_t) ent_dst[c0 + j] * L.n_embd + r0] = a;
                        if (two) out[(size_t) ent_dst[c0 + j] * L.n_embd + r1] = b;
                    }
                }
            };
            switch (cn) {
                case 1: pass(std::integral_constant<int, 1>{}); break;
                case 2: pass(std::integral_constant<int, 2>{}); break;
                case 3: pass(std::integral_constant<int, 3>{}); break;
                default: pass(std::integral_constant<int, 4>{}); break;
            }
        }
    }
}

// ---- 8: mode 7's gate/up with SwiGLU and the q8_1 quantization in the epilogue.  A block takes h rows
// [r0, r0 + 32): gate rows r0.. and up rows r0.. (64 weight rows, the same per-(row, entry) sums as mode 7), so
// one 32-lane warp per entry holds exactly one q8_1 block of h and runs quantize_q8_1_kernel's own code on it -
// bitwise the separate kernels' hq, two launches fewer per call.
template<int TG>
__global__ void __launch_bounds__(256) native_gu_fused_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                              const int32_t* __restrict__ grp_start,
                                                              const int32_t* __restrict__ n_groups,
                                                              const int32_t* __restrict__ ent_tok,
                                                              const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                              block_q8_1* __restrict__ hq) {
    using GT = typename GridOf<TG>::T;
    using F = Fmt<TG>;
    __shared__ GT sgrid[GridOf<TG>::N];
    __shared__ float res[LDS_NT][64];
    extern __shared__ int sx_raw[];
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const GT* gsrc = GridOf<TG>::src();
    for (int i = tid; i < GridOf<TG>::N; i += 256) sgrid[i] = gsrc[i];
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const int nb = (int) (L.n_embd / F::qk), xb = (int) (L.n_embd / 32), hb = (int) (L.n_ff / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    const int r0 = blockIdx.x * 32;
    const block_q8_1* sx = (const block_q8_1*) sx_raw;
    auto wrow = [&](int i) -> const uint8_t* {   // i < 32: gate row r0 + i, else up row r0 + i - 32
        return blob + (i < 32 ? (size_t) 0 : L.up_off) + (size_t) (r0 + (i & 31)) * L.gu_row;
    };
    for (int c0 = e0; c0 < e1; c0 += LDS_NT) {
        const int cn = min(LDS_NT, e1 - c0);
        __syncthreads();
        for (int j = 0; j < cn; ++j) {
            const int* src = (const int*) (xq + (size_t) ent_tok[c0 + j] * xb);
            for (int i = tid; i < xb * 9; i += 256) sx_raw[j * xb * 9 + i] = src[i];
        }
        __syncthreads();
        for (int p = 0; p < 4; ++p) {
            const int i0 = p * 16 + warp, i1 = i0 + 8;
            const uint8_t* w0 = wrow(i0);
            const uint8_t* w1 = wrow(i1);
            auto pass = [&](auto ntc) {
                constexpr int NTC = decltype(ntc)::value;
                float s0[NTC], s1[NTC];
#pragma unroll
                for (int j = 0; j < NTC; ++j) { s0[j] = 0.0f; s1[j] = 0.0f; }
                for (int k = lane; k < nb * F::ipb; k += 32) {
                    const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
#pragma unroll
                    for (int j = 0; j < NTC; ++j) {
                        const block_q8_1* xk = sx + (size_t) j * xb + kbx * (F::qk / 32);
                        const float a = DotG<TG, 1>::f(w0, xk, kbx, iqs, sgrid);
                        const float b = DotG<TG, 1>::f(w1, xk, kbx, iqs, sgrid);
                        s0[j] += a;
                        s1[j] += b;
                    }
                }
#pragma unroll
                for (int j = 0; j < NTC; ++j) {
                    const float a = warp_sum(s0[j]), b = warp_sum(s1[j]);
                    if (lane == 0) { res[j][i0] = a; res[j][i1] = b; }
                }
            };
            switch (cn) {
                case 1: pass(std::integral_constant<int, 1>{}); break;
                case 2: pass(std::integral_constant<int, 2>{}); break;
                case 3: pass(std::integral_constant<int, 3>{}); break;
                default: pass(std::integral_constant<int, 4>{}); break;
            }
        }
        __syncthreads();
        if (warp < cn) {   // swiglu_entries_kernel + quantize_q8_1_kernel, one block of 32 h values per entry
            const float gg = res[warp][lane], uu = res[warp][32 + lane];
            const float xi = (gg / (1.0f + __expf(-gg))) * uu;
            float amax = fabsf(xi), sum = xi;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
                sum += __shfl_xor_sync(0xffffffffu, sum, o);
            }
            const float d = q8_1_finite(amax / 127.0f);   // #606, as q8_1_store: finite blocks bit for bit
            const int8_t q = q8_1_quant(xi, d, amax);
            block_q8_1* y = hq + (size_t) (c0 + warp) * hb + blockIdx.x;
            y->qs[lane] = q;
            if (lane == 0) y->ds = q8_1_ds(d, sum);
        }
    }
}

// dflash2-kern: Strix Halo (gfx1151) takes mode 8 (gate/up + SwiGLU + q8_1 in one launch, the group's activations in LDS) for the
// IQ packs' gate/up: native_expert_bench on the IQ3_S pack's real rows, 45 groups of one entry (a 8-token verify window's call),
// gate/up -25% (IQ2_S), -8% (IQ3_XXS), -6% (IQ3_S), bitwise equal to mode 0.  STRATA_GFX1151_EXP8=0 keeps the old layout.
bool gfx1151_exp8(int dev) {
#if defined(__HIPCC__) && !defined(STRATA_HIP_GFX906)
    static int per_dev[64];   // 0 unknown, 1 no, 2 yes
    if (dev < 0 || dev >= 64) return false;
    if (!per_dev[dev]) {
        const char* v = std::getenv("STRATA_GFX1151_EXP8");
        cudaDeviceProp p;
        per_dev[dev] = !(v != nullptr && v[0] == '0') && cudaGetDeviceProperties(&p, dev) == cudaSuccess &&
                               strata::kernels::gfx_arch_is_gfx1151(p.gcnArchName) ? 2 : 1;
        (void) cudaGetLastError();
    }
    return per_dev[dev] == 2;
#else
    (void) dev;
    return false;
#endif
}
int g_exp_mode = -1;   // native_expert_set_mode (the bench); -1 = STRATA_EXP_MODE, else the card's default
int exp_mode() {
    static const int m = [] { const char* v = std::getenv("STRATA_EXP_MODE"); return v ? std::atoi(v) : -1; }();
    if (g_exp_mode >= 0) return g_exp_mode;
    if (m >= 0) return m;
#if defined(STRATA_HIP_GFX906)
    return kExpModeDefault;
#else
    // Volta (sm_70): mode 8, the grid and the group's activations in shared memory with SwiGLU + q8_1 fused
    // (V100-SXM2, a verify window's VRAM call: 227 -> 159 us); every other CUDA card keeps the CUDA layout
    static int per_dev[64];   // 0 unknown, else mode + 1
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return kExpModeDefault;
    if (!per_dev[dev] && gfx1151_exp8(dev)) per_dev[dev] = 1 + 8;
    if (!per_dev[dev]) {
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        const char* sm70 = std::getenv("STRATA_SM70_TABLE");   // PR 1401: Volta's mode 8 is opt-in until confirmed on a V100
        // Pascal GP100 (sm_60): mode 8 too, on by default (Tesla P100: bitwise equal on the model's layers, the down
        // projection 116 -> 94 us and IQ3_S gate/up 360 -> 229 us in a 28-group VRAM call); STRATA_SM60_TABLE=0 keeps the CUDA layout
        const char* sm60 = std::getenv("STRATA_SM60_TABLE");
        const bool p100 = major == 6 && minor == 0 && (sm60 == nullptr || std::atoi(sm60) != 0);
        // Pascal with dp4a (sm_61): mode 8 as well, on by default (Tesla P40, native_expert_bench on the IQ3_S pack's
        // layers, an 18-group call at 2 tokens: IQ3_S 390 -> 255 us, IQ3_XXS 345 -> 252 us, IQ2_S 290 -> 238 us,
        // IQ4_XS 328 -> 261 us, bitwise equal); STRATA_SM61_TABLE=0 keeps the CUDA layout
        const char* sm61 = std::getenv("STRATA_SM61_TABLE");
        const bool p40 = major == 6 && minor == 1 && (sm61 == nullptr || std::atoi(sm61) != 0);
        // Turing (sm_75): mode 4 (native_expert_bench, real Q2_0 rows of Flash-Next, bitwise equal to mode 0 on an RTX
        // 2080 Ti: 10 experts x 1 entry 47.3 -> 41.4 us, 30 x 1 120.2 -> 100.2 us); the down projection keeps the CUDA
        // layout in windows of 4+ tokens (sm75_down_em below).  STRATA_SM75_EXP=0 keeps the CUDA layout.
        // (off unless STRATA_SM75_PERF=1; STRATA_SM75_EXP=0 / 1 forces it off / on)
        const bool turing = strata::sm75_switch("STRATA_SM75_EXP", major, minor);
        per_dev[dev] = 1 + ((major == 7 && minor == 0 && sm70 != nullptr && std::atoi(sm70) != 0) || p100 || p40 ? 8 : turing ? 4 : kExpModeDefault);
    }
    return per_dev[dev] - 1;
#endif
}
#endif
int g_exp_phase = 0;   // the bench: 0 all, 1 gate/up + swiglu + quantize only, 2 down only, 3 gate/up only
#if STRATA_EXP_LAYOUTS && !defined(__HIPCC__)
// The down projection's mode on Turing's default: mode 4 is slower than the CUDA layout once groups have 3+ entries
// (12 x 3: 36.5 -> 40.8 us, 16 x 4: 56.9 -> 68.9) - windows of 4+ tokens (cap_groups > 30) keep mode 0 there.
int sm75_down_em(int em0, int64_t cap_groups) {
    if (em0 != 4 || g_exp_mode >= 0 || std::getenv("STRATA_EXP_MODE") != nullptr || cap_groups <= 30) return em0;
    int dev = 0, major = 0, minor = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return em0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    return strata::is_sm75(major, minor) ? 0 : em0;
}
#else
int sm75_down_em(int em0, int64_t) { return em0; }
#endif

void native_expert_set_mode(int mode, int phase) {
#if STRATA_EXP_LAYOUTS
    g_exp_mode = mode;
#else
    (void) mode;
#endif
    g_exp_phase = phase;
}

namespace {
bool g_grouped_v1 = env_on("STRATA_GROUPED_V1");
}  // namespace

void native_grouped_set_v1(bool v1) { g_grouped_v1 = v1; }

void native_expert_set_expdec(int on) { g_expdec = on ? 1 : 0; }
int native_expert_expdec() { return expdec_mode(); }
void native_expert_set_expdec_params(const NativeExpertExpdecParams& p) {
    if (p.gu_rows_per_lane > 0) g_expdec_r = p.gu_rows_per_lane;
    if (p.down_rows_per_lane > 0) g_expdec_dr = p.down_rows_per_lane;
    if (p.gu_min_blocks_per_sm > 0) g_expdec_minb = p.gu_min_blocks_per_sm;
    if (p.down_iq4_nl >= 0) g_expdec_d20 = p.down_iq4_nl;
    if (p.down >= 0) g_expdec_down = p.down;
    if (p.gu_rows_per_block >= 0) g_expdec_rb = p.gu_rows_per_block;
    if (p.down_rows_per_block >= 0) g_expdec_drb = p.down_rows_per_block;
    if (p.min_blocks > 0) g_expdec_minblk = p.min_blocks;
}
NativeExpertExpdecParams native_expert_expdec_params() {
    NativeExpertExpdecParams p;
    p.gu_rows_per_lane = expdec_r();
    p.down_rows_per_lane = expdec_dr();
    p.gu_min_blocks_per_sm = g_expdec_minb >= 4 ? 4 : 2;
    p.down_iq4_nl = g_expdec_d20 ? 1 : 0;
    p.down = g_expdec_down ? 1 : 0;
    p.gu_rows_per_block = g_expdec_rb;
    p.down_rows_per_block = g_expdec_drb;
    p.min_blocks = g_expdec_minblk;
    return p;
}
bool native_expert_expdec_takes(const NativeExpertLayout& L, bool& gu, bool& down) {
    gu = expdec_gu_ok(L);
    down = expdec_down_ok(L);
    return gu || down;
}

void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                           const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                           int64_t cap_entries, const void* x_q8_1, void* scratch, float* out, void* stream,
                           int64_t grid_groups) {
    if (cap_groups <= 0 || cap_entries <= 0) return;
    if (L.n_ff % 32 != 0) { std::fprintf(stderr, "native_expert_grouped: n_ff %lld\n", (long long) L.n_ff); std::exit(1); }
    cudaStream_t s = (cudaStream_t) stream;
    const size_t f = (size_t) cap_entries * (size_t) L.n_ff * sizeof(float), fa = (f + 255) & ~(size_t) 255;
    float* gate = (float*) scratch;
    float* up = (float*) ((uint8_t*) scratch + fa);
    float* h = (float*) ((uint8_t*) scratch + 2 * fa);
    block_q8_1* hq = (block_q8_1*) ((uint8_t*) scratch + 3 * fa);
    const auto* X = (const block_q8_1*) x_q8_1;
#if !STRATA_EXPERT_NIBBLE_BUILT
    if (L.gu_nib) {   // STRATA_EXPERT_NIBBLE's slots: the switch is refused at start in a build without the kernels
        std::fprintf(stderr, "native_expert_grouped: 4-bit gate/up slots are not built into this engine\n");
        std::exit(1);
    }
#endif
    // STRATA_EXPERT_NIBBLE (nib_layout): only the gate/up step below takes another kernel (launch_gu_nib, in place of
    // launch_gu); SwiGLU, the q8_1 quantization and the down rows (GGUF, at L.down_off) are the layout's own
    static const bool v2 = [] { const char* v = std::getenv("STRATA_EXPERT_V2"); return v && v[0] == '1'; }();
    if (v2 && !nib_layout(L) && L.gu_type == 21 && L.d_type == 20 && L.n_embd == 2560 && L.n_ff == 640) {   // S26: see s26_gu_l_kernel
        // S26 STRATA_TSUM=1: the sums as one transposed butterfly per warp (s26_tsum.cuh; bitwise the same values)
        static const bool ts = [] { const char* v = std::getenv("STRATA_TSUM"); return v && v[0] == '1'; }();
        // (+ down rows' items spread evenly over the lanes, one load group: bitwise, harness 1.09-1.12x vs 1.03-1.06x;
        // the same for gate / up was 0.6-0.8x)
        if (ts) s26_launch_l<true, true, 4, 4, true, true, true, 1>(L, cap_groups, s, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, X, gate, up, h,
                                                             hq, out, (long long) cap_entries * L.n_ff);
        else s26_launch_l<true, true, 4, 4, true, true>(L, cap_groups, s, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, X, gate, up, h,
                                                        hq, out, (long long) cap_entries * L.n_ff);
        check("native_expert_grouped (v2)");
        return;
    }
    // stream B: UD-Q4_K_XL's Q4_K / Q5_K gate/up and Q5_1 / Q8_0 down (see S27 above)
    // MODEL-UPDATE: on by default on gfx12 (every output the old kernels' bits; R9700, UD-Q4_K_XL decode, 5 interleaved
    // pairs: story +1.2% 5/5, code +0.8% 5/5, after 6K +0.2% 4/5); STRATA_EXPERT_V2K=0 turns it off, =1 turns it on
    // elsewhere (not measured on other cards)
    static const bool v2k = [] {
        const char* v = std::getenv("STRATA_EXPERT_V2K");
        if (v != nullptr && v[0] != '\0') return v[0] == '1';
#if defined(__HIPCC__) && !defined(STRATA_HIP_GFX906)
        int dev = 0;
        cudaDeviceProp p;
        const bool g12 = cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&p, dev) == cudaSuccess &&
                         strata::kernels::gfx_arch_is_gfx12(p.gcnArchName);
        (void) cudaGetLastError();
        return g12;
#else
        return false;
#endif
    }();
    // Qwen3.6-35B-A3B (n_embd 2048, n_ff 512, Q4_K / Q5_K gate+up, Q5_K down): the same kernels, the geometry only sizes the grid
    const bool q36geo = L.n_embd == 2048 && L.n_ff == 512 && L.d_type == 13;
    if (v2k && (L.gu_type == 12 || L.gu_type == 13) &&
        (((L.d_type == 7 || L.d_type == 8) && L.n_embd == 2560 && L.n_ff == 640) || q36geo)) {
        static const bool ts = [] { const char* v = std::getenv("STRATA_TSUM"); return v && v[0] == '1'; }();
        const long long nh = (long long) cap_entries * L.n_ff;
#define STRATA_V2K(G, D) s27_launch_ts<G, D>(ts, L, cap_groups, s, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, X, gate, up, h, hq, out, nh)
        if (q36geo) {
            // the gate/up of the V2K path (4 rows per warp, the chunk's activations in LDS) + SwiGLU/q8_1, and the old down kernel:
            // for n_ff 512 the new down kernel measured slower (15.1 vs 13.4 us per layer on gfx1201, R9700)
            const dim3 ggu((unsigned) (2 * L.n_ff / (GU_ROWS * 4)), (unsigned) cap_groups);
            if (L.gu_type == 12) s27_gu_kernel<12, true, 4, false><<<ggu, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
            else s27_gu_kernel<13, true, 4, false><<<ggu, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
            s26_swiglu_q8_1_kernel<<<(unsigned) ((nh + 255) / 256), 256, 0, s>>>(gate, up, h, hq, nh);
            const dim3 gd((unsigned) ((L.n_embd + 7) / 8), (unsigned) cap_groups);
            launch_down<13>(gd, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
        }
        else if (L.gu_type == 12 && L.d_type == 7) STRATA_V2K(12, 7);
        else if (L.gu_type == 12) STRATA_V2K(12, 8);
        else if (L.d_type == 7) STRATA_V2K(13, 7);
        else STRATA_V2K(13, 8);
#undef STRATA_V2K
        check("native_expert_grouped (v2k)");
        return;
    }
    const bool v1 = g_grouped_v1;
    const int64_t gy = (v1 || grid_groups <= 0 || grid_groups > cap_groups) ? cap_groups : grid_groups;
    const int gu_rows = (!g_old_kernels && !g_no_sub16_gu && L.n_embd == 2560 && gu_split(L.gu_type) &&
                         gu_qk(L.gu_type) != 128 && L.gu_type != 12 && L.gu_type != 13) ? 16 : GU_ROWS;   // qk 128 (APR trellis): no sub-16 rows, see launch_gu
    const dim3 ggu((unsigned) ((2 * L.n_ff + gu_rows - 1) / gu_rows), (unsigned) gy);
#if STRATA_EXP_LAYOUTS
#if defined(STRATA_HIP_GFX906)
    const int em0 = exp_mode();
#else
    // CUDA: the AMD layouts launch a block row per possible group, so a call that strides (a verify window's PCIe
    // call, grid_groups 1..cap) keeps the CUDA layout: V100, 0 groups 11.3 vs 6.5 us, 1 group of 2 29.6 vs 21.3
    const int em0 = (grid_groups > 0 && grid_groups < cap_groups) ? 0 : exp_mode();
#endif
#endif
#if STRATA_EXP_LAYOUTS
    const bool fused_gu = !nib_layout(L) && em0 == 8 && ((kExpLds16 && L.gu_type == 16) || L.gu_type == 18 || L.gu_type == 21 || L.gu_type == 22 || L.gu_type == 23) &&
                          L.n_ff % 32 == 0;
    if (fused_gu && g_exp_phase != 2) {
        const dim3 gl((unsigned) (L.n_ff / 32), (unsigned) cap_groups);
        const size_t sh = (size_t) LDS_NT * (size_t) (L.n_embd / 32) * sizeof(block_q8_1);
        switch (L.gu_type) {
            case 16: native_gu_fused_kernel<16><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, hq); break;
            case 18: native_gu_fused_kernel<18><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, hq); break;
            case 21: native_gu_fused_kernel<21><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, hq); break;
            case 23: native_gu_fused_kernel<23><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, hq); break;
            default: native_gu_fused_kernel<22><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, hq); break;
        }
        check("native_expert_grouped/gu fused");
    }
#endif
    // 147-verify STRATA_GU_SWIGLU=1: gate/up + SwiGLU + q8_1 in one launch where the CUDA layout's 16-lane kernel would run
    bool gs_done = false;
#if STRATA_EXP_LAYOUTS
    const bool gs_layout = !fused_gu && em0 == 0;
#else
    const bool gs_layout = true;
#endif
    if (gs_layout && !nib_layout(L) && g_exp_phase == 0 && !v1 && gu_swiglu_on() && !(expdec_mode() && !g_old_kernels && expdec_gu_ok(L))) {
        switch (L.gu_type) {
#define STRATA_GS(T) case T: gs_done = launch_gu_swiglu<T>(L, gy, s, grp_ptr, grp_start, n_groups, ent_tok, X, hq); break;
            STRATA_GU_FMTS(STRATA_GS)
#undef STRATA_GS
            default: break;
        }
        if (gs_done) check("native_expert_grouped/gu+swiglu");
    }
#if STRATA_EXP_LAYOUTS
    if (g_exp_phase != 2 && !fused_gu && !gs_done) {
#else
    if (g_exp_phase != 2 && !gs_done) {
#endif
#if STRATA_EXP_LAYOUTS
    const bool lds_gu = !nib_layout(L) && (em0 == 5 || em0 == 6 || em0 == 7 || em0 == 8) && ((kExpLds16 && L.gu_type == 16) || L.gu_type == 18 || L.gu_type == 21 || L.gu_type == 22 || L.gu_type == 23);
    if (lds_gu) {
        const dim3 gl((unsigned) ((2 * L.n_ff + LDS_RB - 1) / LDS_RB), (unsigned) cap_groups);
        const size_t sh = (size_t) LDS_NT * (size_t) (L.n_embd / 32) * sizeof(block_q8_1);
        switch (L.gu_type) {
            case 16: if (em0 >= 7) native_gu_lds_kernel<16, 1, true><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); else if (em0 == 6) native_gu_lds_kernel<16, 1><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); else native_gu_lds_kernel<16, 0><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
            case 18: if (em0 >= 7) native_gu_lds_kernel<18, 1, true><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); else if (em0 == 6) native_gu_lds_kernel<18, 1><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); else native_gu_lds_kernel<18, 0><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
            case 21: if (em0 >= 7) native_gu_lds_kernel<21, 1, true><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); else if (em0 == 6) native_gu_lds_kernel<21, 1><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); else native_gu_lds_kernel<21, 0><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
            case 23: if (em0 >= 7) native_gu_lds_kernel<23, 1, true><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); else native_gu_lds_kernel<23, 1><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
            default: if (em0 >= 7) native_gu_lds_kernel<22, 1, true><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); else if (em0 == 6) native_gu_lds_kernel<22, 1><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); else native_gu_lds_kernel<22, 0><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        }
    } else {
    const int em = em0 >= 5 ? kExpFallback : em0;
    const dim3 ggu_amd((unsigned) ((2 * L.n_ff + (em == 1 ? 3 : em == 4 ? 31 : 15)) / (em == 1 ? 4 : em == 4 ? 32 : 16)), (unsigned) cap_groups);
#define STRATA_GU_AMD(T) \
    case T: if (em == 1) native_gu_amd_kernel<T, 1><<<ggu_amd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); \
            else if (em == 4) native_gu_amd_kernel<T, 4><<<ggu_amd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); \
            else native_gu_amd_kernel<T, 2><<<ggu_amd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
    // the AMD layouts cover the IQ packs' gate/up types; Unsloth's Q4_K / Q5_K take the CUDA layout below
    const bool amd_gu = L.gu_type == 16 || L.gu_type == 17 || L.gu_type == 18 || L.gu_type == 21 || L.gu_type == 22 ||
                        L.gu_type == 23 || L.gu_type == 29 || L.gu_type == 42;
    if ((em == 1 || em == 2 || em == 4) && amd_gu && !nib_layout(L)) {
        switch (L.gu_type) {
            STRATA_GU_AMD(16) STRATA_GU_AMD(17) STRATA_GU_AMD(18) STRATA_GU_AMD(21) STRATA_GU_AMD(22) STRATA_GU_AMD(23)
            STRATA_GU_AMD(29) STRATA_GU_AMD(42)
        }
    } else
#undef STRATA_GU_AMD
#endif
#if STRATA_EXPERT_NIBBLE_BUILT
    if (nib_layout(L))   // STRATA_EXPERT_NIBBLE: a converted layer's VRAM slots, native_gu_multi_kernel's grid and bits
        launch_gu_nib(L, dim3((unsigned) (strata::core::nibble::kRows / 16), (unsigned) gy), s, grp_ptr, grp_start,
                      n_groups, ent_tok, X, gate, up);
    else
#endif
    if (!expdec_gu(L, gy, cap_groups, s, grp_ptr, grp_start, n_groups, ent_tok, X, gate, up))   // STRATA_EXPDEC=1
    switch (L.gu_type) {
#define STRATA_GU(T) case T: launch_gu<T>(ggu, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        STRATA_GU_FMTS(STRATA_GU)
#undef STRATA_GU
        default: std::fprintf(stderr, "native_expert_grouped: gate/up type %d\n", L.gu_type); std::exit(1);
    }
#if STRATA_EXP_LAYOUTS
    }
#endif
    check("native_expert_grouped/gu");
    if (g_exp_phase == 3) return;   // the bench: gate/up only
    const long long nh = (long long) cap_entries * L.n_ff;
#if defined(STRATA_HIP_GFX906)
    // gfx906: the separate SwiGLU and q8_1 passes stay the default (the A/B baseline); the RDNA one-pass
    // kernel (bitwise the same) with STRATA_HIP_SWIGLU_FUSED=1
    static const bool sw_fused = env_on("STRATA_HIP_SWIGLU_FUSED");
    const bool sw_v1 = v1 || !sw_fused;
#else
    const bool sw_v1 = v1;
#endif
    // APR: Hadamard-folded down weights take the rotated h, so h exists as floats first (the separate passes)
    if (sw_v1 || L.had_block > 0) {
        swiglu_entries_kernel<<<(unsigned) ((nh + 255) / 256), 256, 0, s>>>(gate, up, h, nh);
        if (L.had_block > 0)   // APR: the down weights are Hadamard-folded, so is their input (in place, every entry row)
            hadamard_rows(h, h, cap_entries, L.n_ff, L.had_block, L.d_signs, s);
        quantize_q8_1_kernel<<<(unsigned) ((nh + 255) / 256), 256, 0, s>>>(h, hq, nh);
    } else {
        swiglu_q8_1_entries_kernel<<<(unsigned) ((nh + 255) / 256), 256, 0, s>>>(gate, up, grp_start, n_groups,
                                                                                (int) L.n_ff, hq);
    }
    check("native_expert_grouped/swiglu");
    }
    if (g_exp_phase == 1) return;
    // STRATA_LAT_DOWN_S26 (opt-in): the S26 down rows for a geometry the S26 path's n_embd 2560 gate excludes - the
    // latent pack's 1280 / n_ff 640.  s26_down_l_kernel carries no n_embd constant (r0, live and the output row are all
    // runtime) and its LDS row is n_ff/32 = 20 blocks, so it is the IQ4_NL latent down as it stands: the fp4 latent
    // kernel's 1280/640 structure, with 16-byte weight blocks instead of the 2-byte loads native_down_multi_kernel
    // issues one item at a time.  Bitwise native_down_multi_kernel's (S26 harness).
    static const bool lat_down_s26 = env_on("STRATA_LAT_DOWN_S26");
    if (lat_down_s26 && L.d_type == 20 && L.n_ff == 640 && L.n_embd > 0 && L.n_embd % 32 == 0) {
        const dim3 gl((unsigned) ((L.n_embd + 31) / 32), (unsigned) cap_groups);
        s26_down_l_kernel<true, 4><<<gl, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
        check("native_expert_grouped/down lat s26");
        return;
    }
    const bool tq_sub8 = (L.d_type == 144 || L.d_type == 146) && !g_old_kernels && !g_no_sub16_gu && L.n_ff == 640 && (tq_stage_on() || tq_sub8_on());   // the trellis down rows as 8-lane rows
    const int d_rows = (!g_old_kernels && !g_no_sub16_gu && L.n_ff == 640) ? (L.d_type == 20 || L.d_type == 40 || tq_sub8 ? 32 : (L.d_type == 42 ? 16 : 8)) : 8;
    const int tq_ch = tq_sub8 ? tq_row_chunks() : 1;   // the staged trellis down kernel takes this many 32-row chunks a block
    const dim3 gd((unsigned) (((L.n_embd + d_rows - 1) / d_rows + tq_ch - 1) / tq_ch), (unsigned) gy);
#if STRATA_DOWN_WS_BUILT
    if (down_ws_on(L)) {   // STRATA_DOWN_WS=1 (cc 6.0): the staged down rows, bitwise native_down_multi_kernel's
        launch_down_ws(L, gy, cap_groups, s, grp_ptr, grp_start, n_groups, ent_dst, hq, out);
        check("native_expert_grouped/down ws");
        return;
    }
#endif
#if STRATA_EXP_LAYOUTS
    const int em0d = sm75_down_em(em0, cap_groups);
    int dev_d = 0;
    cudaGetDevice(&dev_d);
    // gfx1151: the LDS down kernel is slower than the CUDA layout for IQ4_NL (86 vs 63 us, 24 groups) and faster for Q2_0 (-18%)
    const bool lds_down_ok = L.d_type == 42 || (L.d_type == 20 && !gfx1151_exp8(dev_d));
    if ((em0d == 7 || em0d == 8) && lds_down_ok) {
        const dim3 gl((unsigned) ((L.n_embd + LDS_RB - 1) / LDS_RB), (unsigned) cap_groups);
        const size_t sh = (size_t) LDS_NT * (size_t) (L.n_ff / 32) * sizeof(block_q8_1);
        if (L.d_type == 20) native_down_lds_kernel<20><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
        else native_down_lds_kernel<42><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
        check("native_expert_grouped/down");
        return;
    }
    const int em = em0d >= 5 ? kExpFallback : em0d;
    const dim3 gd_amd((unsigned) ((L.n_embd + (em == 1 ? 3 : em == 4 ? 31 : 15)) / (em == 1 ? 4 : em == 4 ? 32 : 16)), (unsigned) cap_groups);
#define STRATA_D_AMD(T) \
    case T: if (em == 1) native_down_amd_kernel<T, 1><<<gd_amd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); \
            else if (em == 4) native_down_amd_kernel<T, 4><<<gd_amd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); \
            else native_down_amd_kernel<T, 2><<<gd_amd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
    // the AMD layouts cover the IQ packs' down types; the others (Unsloth's Q4_K / Q5_K / Q5_1 / Q8_0) take the
    // CUDA layout below
    if ((em == 1 || em == 2 || em == 4) && (L.d_type == 20 || L.d_type == 23 || L.d_type == 42)) {
        switch (L.d_type) {
            STRATA_D_AMD(20) STRATA_D_AMD(23) STRATA_D_AMD(42)
        }
    } else
#undef STRATA_D_AMD
#endif
    if (!expdec_down(L, gy, cap_groups, s, grp_ptr, grp_start, n_groups, ent_dst, hq, out))   // STRATA_EXPDEC=1
    switch (L.d_type) {
#define STRATA_DOWN(T) case T: launch_down<T>(gd, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        STRATA_D_FMTS(STRATA_DOWN)
#undef STRATA_DOWN
        default: std::fprintf(stderr, "native_expert_grouped: down type %d\n", L.d_type); std::exit(1);
    }
    check("native_expert_grouped/down");
}

}  // namespace strata::kernels

// ---------------------------------------------------------------- STRATA_EXPERT_NIBBLE: expert_nibble.hpp's functions
namespace strata::core::nibble {

#if STRATA_EXPERT_NIBBLE_BUILT
namespace {
int g_rev_ready[64];   // per device: 0 not built, 1 built, 2 a grid entry not unique (refused)

bool rev_ready() {
    int dev = 0;
    return cudaGetDevice(&dev) == cudaSuccess && dev >= 0 && dev < 64 && g_rev_ready[dev] == 1;
}
}  // namespace
#endif

int prepare(void* stream) {
#if STRATA_EXPERT_NIBBLE_BUILT
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return (int) cudaErrorInvalidDevice;
    if (g_rev_ready[dev] == 1) return 0;
    if (g_rev_ready[dev] == 2) return (int) cudaErrorInvalidValue;
    cudaStream_t s = (cudaStream_t) stream;
    strata::kernels::nib_rev_clear_kernel<<<(strata::kernels::kNibRev22 + 255) / 256, 256, 0, s>>>();
    strata::kernels::nib_rev_fill_kernel<<<(1024 + 256 + 512 + 255) / 256, 256, 0, s>>>();
    cudaError_t e = cudaGetLastError();
    unsigned long long bad = 0;
    if (e == cudaSuccess) e = cudaStreamSynchronize(s);
    if (e == cudaSuccess) e = cudaMemcpyFromSymbol(&bad, strata::kernels::g_nib_rev_bad, sizeof bad);
    if (e != cudaSuccess) return (int) e;
    g_rev_ready[dev] = bad == 0 ? 1 : 2;
    return bad == 0 ? 0 : (int) cudaErrorInvalidValue;
#else
    (void) stream;
    return 1;   // cudaErrorInvalidValue: not in this build
#endif
}

int transcode(const LayerVram& L, const void* staging, int64_t n, const unsigned long long* slot_ptrs,
              unsigned long long* bad, void* stream) {
#if STRATA_EXPERT_NIBBLE_BUILT
    if (n <= 0) return 0;
    if (!L.on || !fmt_built(L.gu_type) || L.raw_gu_row != raw_gu_row(L.gu_type) ||
        L.raw_up_off != (uint64_t) kNFf * L.raw_gu_row || L.raw_bytes < 2 * L.raw_up_off)
        return (int) cudaErrorInvalidValue;
    const long long nthr = (long long) n * kRows * kItems;
    const dim3 grid((unsigned) ((nthr + 255) / 256));
    cudaStream_t s = (cudaStream_t) stream;
    const uint8_t* src = (const uint8_t*) staging;
#define STRATA_NIB_TC(TG) \
    strata::kernels::nib_transcode_kernel<TG><<<grid, 256, 0, s>>>(src, L.raw_bytes, L.raw_gu_row, L.raw_up_off, slot_ptrs, n, bad)
    switch (L.gu_type) {
        case kIQ2_S: STRATA_NIB_TC(22); break;
        case kIQ3_XXS: STRATA_NIB_TC(18); break;
        case kIQ3_S: STRATA_NIB_TC(21); break;
        default: return (int) cudaErrorInvalidValue;
    }
#undef STRATA_NIB_TC
    return (int) cudaGetLastError();
#else
    (void) L; (void) staging; (void) n; (void) slot_ptrs; (void) bad; (void) stream;
    return 1;   // cudaErrorInvalidValue: not in this build
#endif
}

#if STRATA_EXPERT_NIBBLE_BUILT
namespace {
int launch_gguf(const LayerVram& L, strata::kernels::NibGgufArgs& a, bool check, cudaStream_t s) {
    namespace K = strata::kernels;
    if (!L.on || !fmt_built(L.gu_type) || L.raw_gu_row != raw_gu_row(L.gu_type) || L.down_bytes % 16 != 0 ||
        L.raw_up_off != (uint64_t) kNFf * L.raw_gu_row || L.raw_down_off != 2 * L.raw_up_off)
        return (int) cudaErrorInvalidValue;
    if (!rev_ready()) return (int) cudaErrorNotReady;   // prepare() was not run on this device
    a.gu_row = L.raw_gu_row;
    a.down16 = L.down_bytes / 16;
    const int64_t tail = (int64_t) (a.down16 + 2 * a.zero16);
    const int64_t per = (int64_t) K::kNibGgufThreads * K::kNibGgufWords;
    const dim3 grid((unsigned) (kRows / K::kNibGgufRows + (tail + per - 1) / per), (unsigned) (a.n - a.first));
#define STRATA_NIB_GG(TG) \
    (check ? K::nib_gguf_kernel<TG, true><<<grid, K::kNibGgufThreads, 0, s>>>(a) \
           : K::nib_gguf_kernel<TG, false><<<grid, K::kNibGgufThreads, 0, s>>>(a))
    switch (L.gu_type) {
        case kIQ2_S: STRATA_NIB_GG(22); break;
        case kIQ3_XXS: STRATA_NIB_GG(18); break;
        case kIQ3_S: STRATA_NIB_GG(21); break;
        default: return (int) cudaErrorInvalidValue;
    }
#undef STRATA_NIB_GG
    return (int) cudaGetLastError();
}
}  // namespace
#endif

int check_gguf(const LayerVram& L, const void* staging, int64_t n, const unsigned long long* slot_ptrs,
               unsigned long long* bad, void* stream) {
#if STRATA_EXPERT_NIBBLE_BUILT
    if (n <= 0) return 0;
    if (n > 65535) return (int) cudaErrorInvalidValue;
    strata::kernels::NibGgufArgs a{};
    a.dev = slot_ptrs;
    a.first = 0;
    a.n = (int) n;
    a.gu = (uint8_t*) staging;
    a.gu_stride = L.raw_bytes;
    a.d = (uint8_t*) staging + L.raw_down_off;
    a.d_stride = L.raw_bytes;
    a.zero16 = 0;
    a.bad = bad;
    if (((uintptr_t) staging | L.raw_bytes) % 16 != 0) return (int) cudaErrorInvalidValue;
    return launch_gguf(L, a, true, (cudaStream_t) stream);
#else
    (void) L; (void) staging; (void) n; (void) slot_ptrs; (void) bad; (void) stream;
    return 1;   // cudaErrorInvalidValue: not in this build
#endif
}

int gather_gguf(const LayerVram& L, const uint8_t* const* slots, int first, int n, void* gu_dst, uint64_t gu_stride,
                void* d_dst, uint64_t d_stride, uint64_t zero_bytes, void* stream) {
#if STRATA_EXPERT_NIBBLE_BUILT
    if (first < 0 || n <= first || n > 16) return (int) cudaErrorInvalidValue;
    uintptr_t al = (uintptr_t) gu_dst | (uintptr_t) d_dst | gu_stride | d_stride | zero_bytes;
    for (int q = first; q < n; ++q) al |= (uintptr_t) slots[q];
    if (al % 16 != 0 || gu_stride < (uint64_t) kRows * L.raw_gu_row || d_stride < L.down_bytes) return (int) cudaErrorInvalidValue;
    strata::kernels::NibGgufArgs a{};
    a.dev = nullptr;
    for (int q = first; q < n; ++q) a.p[q] = slots[q];
    a.first = first;
    a.n = n;
    a.gu = (uint8_t*) gu_dst;
    a.gu_stride = gu_stride;
    a.d = (uint8_t*) d_dst;
    a.d_stride = d_stride;
    a.zero16 = zero_bytes / 16;
    a.bad = nullptr;
    return launch_gguf(L, a, false, (cudaStream_t) stream);
#else
    (void) L; (void) slots; (void) first; (void) n; (void) gu_dst; (void) gu_stride; (void) d_dst; (void) d_stride;
    (void) zero_bytes; (void) stream;
    return 1;   // cudaErrorInvalidValue: not in this build
#endif
}

int hash_slots(const LayerVram& L, const unsigned long long* slot_ptrs, int64_t n, unsigned long long* out, void* stream) {
#if STRATA_EXPERT_NIBBLE_BUILT
    if (n <= 0) return 0;
    if (L.hashed == 0) return (int) cudaErrorInvalidValue;
    cudaStream_t s = (cudaStream_t) stream;
    const cudaError_t e = cudaMemsetAsync(out, 0, (size_t) n * sizeof(unsigned long long), s);
    if (e != cudaSuccess) return (int) e;
    // ~2.7 MB a slot: 32 blocks of 256 threads each, ~21 16-byte loads a thread
    strata::kernels::nib_hash_kernel<<<dim3(32, (unsigned) n), 256, 0, s>>>(slot_ptrs, L.hashed, out);
    return (int) cudaGetLastError();
#else
    (void) L; (void) slot_ptrs; (void) n; (void) out; (void) stream;
    return 1;   // cudaErrorInvalidValue: not in this build
#endif
}

}  // namespace strata::core::nibble

// ---- Volta prompt experts (STRATA_PF_WMMA): a group's experts as one launch of FP16 tensor-core tiles, the weights
// dequantized into shared memory by dq_dispatch (the FP16 path's own formulas), the activations FP16, FP32 sums.
// Replaces MMQ's dp4a products on sm_70, and the FP16 path's dequantize-to-global + one cuBLAS call per expert.
#if !defined(__HIPCC__)
#include <mma.h>
namespace strata::kernels {
namespace {
constexpr int WG_BM = 64, WG_BN = 128, WG_THREADS = 256, WG_XK = 128, WG_MT = 4;
// A block: BN = 128 weight rows of one expert x up to WG_MT * BM = 256 of its token rows (blockIdx.y walks an expert's
// rows in such chunks), so every weight superblock is dequantized ONCE per chunk into shared memory (WK 256: an IQ
// superblock through dq_dispatch; WK 128: two Q2_0 blocks, or four IQ4_NL blocks); the activations stream through a
// BM x 128 tile.  8 warps of 32 x 32 per m-tile (2 x 2 fragments), FP32 accumulators for the chunk's m-tiles in registers.
template<int WK, bool Q20, bool IQNL = false>
__global__ void __launch_bounds__(WG_THREADS) gemm_iq_f16_grouped_kernel(int ty, const uint8_t* __restrict__ W,
                                                                         size_t expert_bytes, size_t row_bytes, int K,
                                                                         const __half* __restrict__ X, int ldx,
                                                                         const int32_t* __restrict__ bounds, int ch_per_e,
                                                                         float* __restrict__ Y, int ldy) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700   // the sm_60 pass of a multi-arch build has no nvcuda::wmma (the host never launches these there)
    using namespace nvcuda;
    constexpr int LDW = WK + 8, LDX = WG_XK + 8;
    extern __shared__ __align__(16) unsigned char wg_smem[];
    __half* sw = reinterpret_cast<__half*>(wg_smem);            // [BN][LDW]
    __half* sx = sw + WG_BN * LDW;                               // [BM][LDX]
    const int e = blockIdx.y / ch_per_e, ch = blockIdx.y % ch_per_e;
    const int c0 = bounds[e] + ch * WG_MT * WG_BM, r1 = bounds[e + 1];
    if (c0 >= r1) return;
    const int crow = min(WG_MT * WG_BM, r1 - c0);
    const int nmt = (crow + WG_BM - 1) / WG_BM;
    const int n0 = blockIdx.x * WG_BN;
    const uint8_t* we = W + (size_t) e * expert_bytes;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp >> 2, wn = warp & 3;                     // rows wm*32, cols wn*32 of an m-tile
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[WG_MT][2][2];
#pragma unroll
    for (int t = 0; t < WG_MT; ++t)
#pragma unroll
        for (int a = 0; a < 2; ++a)
#pragma unroll
            for (int b = 0; b < 2; ++b) wmma::fill_fragment(acc[t][a][b], 0.0f);
    // the activation tiles in order (k0, m-tile, half): the next one's 4 uint4 per thread are loaded into registers
    // while the current one is multiplied (the first before the weights' dequantization)
    constexpr int XQ = WG_BM * (WG_XK / 8) / WG_THREADS;        // 4
    const int nx = WK / WG_XK, per_k = nmt * nx, total = (K / WK) * per_k;
    uint4 xr[XQ];
    auto load_x = [&](int it) {
        const int kb = it / per_k, rem = it % per_k, t = rem / nx, xh = rem % nx;
        const int rows = min(WG_BM, crow - t * WG_BM);
#pragma unroll
        for (int q = 0; q < XQ; ++q) {
            const int i = tid + q * WG_THREADS, r = i / (WG_XK / 8), c = (i % (WG_XK / 8)) * 8;
            xr[q] = r < rows ? *reinterpret_cast<const uint4*>(X + (size_t) (c0 + t * WG_BM + r) * ldx + kb * WK + xh * WG_XK + c)
                             : make_uint4(0, 0, 0, 0);
        }
    };
    load_x(0);
    int it = 0;
    for (int k0 = 0; k0 < K; k0 += WK) {
        if constexpr (IQNL) {
            // IQ4_NL down (WK 128 = 4 x block_iq4_nl): one thread per (row, 64-value half) - the four blocks of the
            // 128-value window sit in two halves of two blocks, and each thread's values are exactly dq_iq4_nl's
            const int r = tid >> 1, half = tid & 1;
            const block_iq4_nl* bp = reinterpret_cast<const block_iq4_nl*>(we + (size_t) (n0 + r) * row_bytes) + k0 / 32 + half * 2;
            __half* o = sw + r * LDW + half * 64;
#pragma unroll
            for (int bl = 0; bl < 2; ++bl) {
                const block_iq4_nl* b = bp + bl;
                const float d = (float) b->d;
#pragma unroll
                for (int p = 0; p < 16; ++p) {
                    o[bl * 32 + p] = cvt<__half>(d * kvalues_iq4nl[b->qs[p] & 0xf]);
                    o[bl * 32 + p + 16] = cvt<__half>(d * kvalues_iq4nl[b->qs[p] >> 4]);
                }
            }
        } else if constexpr (!Q20) {
            for (int r = warp; r < WG_BN; r += WG_THREADS / 32)
                dq_dispatch<__half>(ty, we + (size_t) (n0 + r) * row_bytes, k0 / 256, sw + r * LDW, lane);
        } else {
            const int r = tid >> 1, blk = tid & 1;   // one thread per (row, 64-value block)
            const block_q2_0* b = reinterpret_cast<const block_q2_0*>(we + (size_t) (n0 + r) * row_bytes) + k0 / 64 + blk;
            const float d = (float) b->d;
            uint4 q4;
            memcpy(&q4, b->qs, 16);
            const uint8_t* qb = reinterpret_cast<const uint8_t*>(&q4);
            __half* o = sw + r * LDW + blk * 64;
#pragma unroll
            for (int c = 0; c < 16; ++c) {
                const int v = qb[c];
                *reinterpret_cast<__half2*>(o + 4 * c) = __floats2half2_rn(d * (float) ((v & 3) - 1), d * (float) (((v >> 2) & 3) - 1));
                *reinterpret_cast<__half2*>(o + 4 * c + 2) = __floats2half2_rn(d * (float) (((v >> 4) & 3) - 1), d * (float) (((v >> 6) & 3) - 1));
            }
        }
#pragma unroll
        for (int t = 0; t < WG_MT; ++t) {
            if (t >= nmt) break;
#pragma unroll 1
            for (int xh = 0; xh < WK / WG_XK; ++xh, ++it) {
                __syncthreads();                                 // the previous X tile is consumed (and W is written)
#pragma unroll
                for (int q = 0; q < XQ; ++q) {
                    const int i = tid + q * WG_THREADS, r = i / (WG_XK / 8), c = (i % (WG_XK / 8)) * 8;
                    *reinterpret_cast<uint4*>(sx + r * LDX + c) = xr[q];
                }
                __syncthreads();
                if (it + 1 < total) load_x(it + 1);
#pragma unroll
                for (int kk = 0; kk < WG_XK; kk += 16) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[2];
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b[2];
#pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        wmma::load_matrix_sync(a[f], sx + (wm * 32 + f * 16) * LDX + kk, LDX);
                        wmma::load_matrix_sync(b[f], sw + (wn * 32 + f * 16) * LDW + xh * WG_XK + kk, LDW);
                    }
#pragma unroll
                    for (int fa = 0; fa < 2; ++fa)
#pragma unroll
                        for (int fb = 0; fb < 2; ++fb) wmma::mma_sync(acc[t][fa][fb], a[fa], b[fb], acc[t][fa][fb]);
                }
            }
        }
        __syncthreads();                                         // W is consumed before the next superblock
    }
    float* so = reinterpret_cast<float*>(wg_smem);               // [BM][BN + 4] per m-tile
    constexpr int LDO = WG_BN + 4;
#pragma unroll
    for (int t = 0; t < WG_MT; ++t) {
        if (t >= nmt) break;
        const int rows = min(WG_BM, crow - t * WG_BM);
#pragma unroll
        for (int fa = 0; fa < 2; ++fa)
#pragma unroll
            for (int fb = 0; fb < 2; ++fb)
                wmma::store_matrix_sync(so + (wm * 32 + fa * 16) * LDO + wn * 32 + fb * 16, acc[t][fa][fb], LDO, wmma::mem_row_major);
        __syncthreads();
        for (int i = tid; i < rows * (WG_BN / 4); i += WG_THREADS) {
            const int r = i / (WG_BN / 4), c = (i % (WG_BN / 4)) * 4;
            *reinterpret_cast<float4*>(Y + (size_t) (c0 + t * WG_BM + r) * ldy + n0 + c) = *reinterpret_cast<const float4*>(so + r * LDO + c);
        }
        __syncthreads();
    }
#endif
}

// ---- STRATA_PF_WMMA2=1 (opt-in): IQ2_S gate/up and IQ4_NL down, tiled like a Volta MoE kernel in the
// PR author's own unpublished llama.cpp patches.
// A block: W2_RB = 128 weight rows of one expert x W2_CB = 64 of its sorted rows, K stages of W2_KT = 64. Thread t dequantizes
// row t % 128, 32-value sub-block t / 128 of each K stage from the raw stage in shared memory (one IQ2_S super-block or four IQ4_NL
// blocks per row, read by the whole block as 4-byte words, the next one already in registers). 8 warps = 4 (32 weight rows) x 2
// (32 sorted rows), 2 x 2 wmma 16x16x16 fragments each, FP32 sums.
constexpr int W2_RB = 128, W2_CB = 64, W2_KT = 64, W2_NT = 256, W2_LD = W2_KT + 8;
template <int TY> struct w2_traits;
template <> struct w2_traits<22> { static constexpr int raw_k = 256, raw_b = 82, segw = 21, segw_s = 21, ntab = 1024 + 16; };
template <> struct w2_traits<20> { static constexpr int raw_k = 128, raw_b = 72, segw = 18, segw_s = 19, ntab = 128; };
template <int TY> __host__ __device__ constexpr int w2_nl() { return (W2_RB * w2_traits<TY>::segw + W2_NT - 1) / W2_NT; }

__device__ __forceinline__ uint32_t w2_ld_b2(const uint8_t* p) {   // 4 bytes at a 2-byte aligned address
    const uint16_t* q = reinterpret_cast<const uint16_t*>(p);
    return (uint32_t) q[0] | ((uint32_t) q[1] << 16);
}
__device__ __forceinline__ half2 w2_h2(uint32_t v) { half2 h; memcpy(&h, &v, 4); return h; }
__device__ __forceinline__ uint32_t w2_u32(half2 h) { uint32_t v; memcpy(&v, &h, 4); return v; }

// the raw stage rs of the block's 128 rows: segw words per row from the 4-byte aligned start at or below rs * raw_b
template <int TY>
__device__ __forceinline__ void w2_raw_ld(uint32_t (&R)[w2_nl<TY>()], const uint8_t* __restrict__ xb, size_t row_bytes, int rs, int tid) {
    using tr = w2_traits<TY>;
    const int off = (rs * tr::raw_b) & ~3;
#pragma unroll
    for (int i = 0; i < w2_nl<TY>(); ++i) {
        const int w = tid + i * W2_NT;
        if ((W2_RB * tr::segw) % W2_NT == 0 || w < W2_RB * tr::segw)
            R[i] = *reinterpret_cast<const uint32_t*>(xb + (size_t) (w / tr::segw) * row_bytes + off + 4 * (w % tr::segw));
    }
}
template <int TY>
__device__ __forceinline__ void w2_raw_st(const uint32_t (&R)[w2_nl<TY>()], uint32_t* __restrict__ raw, int tid) {
    using tr = w2_traits<TY>;
#pragma unroll
    for (int i = 0; i < w2_nl<TY>(); ++i) {
        const int w = tid + i * W2_NT;
        if ((W2_RB * tr::segw) % W2_NT == 0 || w < W2_RB * tr::segw) raw[(w / tr::segw) * tr::segw_s + w % tr::segw] = R[i];
    }
}

// sub-block `sb` (0..7) of the IQ2_S super-block at rr -> 32 halves at o, natural order, bit for bit dq_iq2_s's values:
// cvt<__half>(dl * grid[j] * +-1) with dl = d * (0.5 + scale) * 0.25 in FP32 (the sign flips the rounded half, which is the same)
__device__ __forceinline__ float w2_b2f(uint32_t w, int j) { return __int_as_float(0x4B000000 | ((w >> 8 * j) & 0xff)) - 8388608.0f; }
__device__ __forceinline__ void w2_dq_iq2_s(const uint8_t* rr, int sb, __half* o, const uint2* __restrict__ tab) {
    const uint2* sgn = tab + 1024;
    const uint32_t sc = rr[2 + 64 + 8 + sb], qh = rr[2 + 64 + sb];
    const uint32_t qs = w2_ld_b2(rr + 2 + 4 * sb), sg = w2_ld_b2(rr + 2 + 32 + 4 * sb);
    const float d = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(rr)));
    const float dl0 = d * (0.5f + (float) (sc & 0xf)) * 0.25f;
    const float dl1 = d * (0.5f + (float) (sc >> 4)) * 0.25f;
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const float dl = il < 2 ? dl0 : dl1;
        const int idx = (int) (((qs >> 8 * il) & 0xff) | ((qh << (8 - 2 * il)) & 0x300));
        const uint2 gw = tab[idx];
        const uint32_t sbits = (sg >> 8 * il) & 0xff;
        const uint2 s0 = sgn[sbits & 15], s1 = sgn[sbits >> 4];
        uint4 out;
        out.x = w2_u32(__floats2half2_rn(dl * w2_b2f(gw.x, 0), dl * w2_b2f(gw.x, 1))) ^ s0.x;
        out.y = w2_u32(__floats2half2_rn(dl * w2_b2f(gw.x, 2), dl * w2_b2f(gw.x, 3))) ^ s0.y;
        out.z = w2_u32(__floats2half2_rn(dl * w2_b2f(gw.y, 0), dl * w2_b2f(gw.y, 1))) ^ s1.x;
        out.w = w2_u32(__floats2half2_rn(dl * w2_b2f(gw.y, 2), dl * w2_b2f(gw.y, 3))) ^ s1.y;
        *reinterpret_cast<uint4*>(o + 8 * il) = out;
    }
}
// IQ4_NL block at b -> 32 halves at o, natural order (value j: the low nibble of byte j, value j + 16: its high nibble); the product
// of two exact halves rounded once, as cvt<__half>(d * kvalues_iq4nl[q])
__device__ __forceinline__ void w2_dq_iq4_nl(const uint8_t* b, __half* o, const half2* __restrict__ kv) {
    const half2 d2 = __half2half2(__ushort_as_half(*reinterpret_cast<const uint16_t*>(b)));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t q = w2_ld_b2(b + 2 + 4 * i);
        const half2 t0 = kv[q & 0xff], t1 = kv[(q >> 8) & 0xff], t2 = kv[(q >> 16) & 0xff], t3 = kv[q >> 24];
        uint2 lo, hi;
        lo.x = w2_u32(__hmul2(__lows2half2(t0, t1), d2));
        lo.y = w2_u32(__hmul2(__lows2half2(t2, t3), d2));
        hi.x = w2_u32(__hmul2(__highs2half2(t0, t1), d2));
        hi.y = w2_u32(__hmul2(__highs2half2(t2, t3), d2));
        *reinterpret_cast<uint2*>(o + 4 * i) = lo;
        *reinterpret_cast<uint2*>(o + 16 + 4 * i) = hi;
    }
}

template <int TY>
__global__ void __launch_bounds__(W2_NT, 2) gemm_iq_f16_w2_kernel(const uint8_t* __restrict__ W, size_t expert_bytes, size_t row_bytes,
                                                                  int K, const __half* __restrict__ X, int ldx,
                                                                  const int32_t* __restrict__ bounds, int ch_per_e,
                                                                  float* __restrict__ Y, int ldy) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700   // the sm_60 pass of a multi-arch build has no nvcuda::wmma (the host never launches these there)
    using namespace nvcuda;
    using tr = w2_traits<TY>;
    constexpr int NL = w2_nl<TY>(), NSUB = tr::raw_k / W2_KT;
    __shared__ __align__(128) __half xs[W2_RB * W2_LD];
    __shared__ __align__(128) __half ys[W2_CB * W2_LD];
    __shared__ uint2 tab[tr::ntab];
    __shared__ uint32_t raw[W2_RB * tr::segw_s];

    const int e = blockIdx.y / ch_per_e, ch = blockIdx.y % ch_per_e;
    const int c0 = bounds[e] + ch * W2_CB, r1 = bounds[e + 1];
    if (c0 >= r1) return;
    const int ncol = min(W2_CB, r1 - c0);
    const int n0 = blockIdx.x * W2_RB;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wr = warp & 3, wc = warp >> 2;                     // weight rows wr*32, sorted rows wc*32
    const uint8_t* xb = W + (size_t) e * expert_bytes + (size_t) n0 * row_bytes;
    const int nrs = K / tr::raw_k;

    uint32_t R[NL];
    w2_raw_ld<TY>(R, xb, row_bytes, 0, tid);
    const int yq = tid & 7, yc = tid >> 3;                       // 16-byte part yq of sorted rows yc and yc + 32
    uint4 Yr[2];
    auto load_y = [&](int k0) {
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            const int c = yc + 32 * u;
            Yr[u] = c < ncol ? *reinterpret_cast<const uint4*>(X + (size_t) (c0 + c) * ldx + k0 + 8 * yq) : make_uint4(0, 0, 0, 0);
        }
    };
    load_y(0);
    if constexpr (TY == 20) {
        reinterpret_cast<half2*>(tab)[tid] = __halves2half2(__int2half_rn(kvalues_iq4nl[tid & 15]), __int2half_rn(kvalues_iq4nl[tid >> 4]));
    } else {
        for (int i = tid; i < 1024; i += W2_NT) tab[i] = reinterpret_cast<const uint2*>(iq2s_grid)[i];
        if (tid < 16)   // a sign nibble -> masks of its bits 0-1 and 2-3 (bit 2p -> bit 15, bit 2p + 1 -> bit 31)
            tab[1024 + tid] = make_uint2(((tid >> 0) & 1u) << 15 | ((tid >> 1) & 1u) << 31, ((tid >> 2) & 1u) << 15 | ((tid >> 3) & 1u) << 31);
    }
    w2_raw_st<TY>(R, raw, tid);
    if (nrs > 1) w2_raw_ld<TY>(R, xb, row_bytes, 1, tid);
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < 2; ++b) wmma::fill_fragment(acc[a][b], 0.0f);
    const int xr = tid % W2_RB, sub = tid / W2_RB;
    __half* xrow = xs + xr * W2_LD + 32 * sub;
    const uint8_t* rawrow = reinterpret_cast<const uint8_t*>(raw + xr * tr::segw_s);

    for (int rs = 0; rs < nrs; ++rs) {
        const uint8_t* rr = rawrow + ((rs * tr::raw_b) & 3);
#pragma unroll
        for (int h = 0; h < NSUB; ++h) {
            const int k0 = (rs * NSUB + h) * W2_KT;
            if constexpr (TY == 20) w2_dq_iq4_nl(rr + 18 * (2 * h + sub), xrow, reinterpret_cast<const half2*>(tab));
            else w2_dq_iq2_s(rr, 2 * h + sub, xrow, tab);
#pragma unroll
            for (int u = 0; u < 2; ++u) *reinterpret_cast<uint4*>(ys + (yc + 32 * u) * W2_LD + 8 * yq) = Yr[u];
            __syncthreads();
            if (k0 + W2_KT < K) load_y(k0 + W2_KT);
            if (wc * 32 < ncol) {                                // warp-uniform: sorted rows past the expert's skip their mma
#pragma unroll
                for (int kk = 0; kk < W2_KT; kk += 16) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[2];
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b[2];
#pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        wmma::load_matrix_sync(a[f], ys + (wc * 32 + f * 16) * W2_LD + kk, W2_LD);
                        wmma::load_matrix_sync(b[f], xs + (wr * 32 + f * 16) * W2_LD + kk, W2_LD);
                    }
#pragma unroll
                    for (int fa = 0; fa < 2; ++fa) {
                        if (wc * 32 + fa * 16 >= ncol) break;
#pragma unroll
                        for (int fb = 0; fb < 2; ++fb) wmma::mma_sync(acc[fa][fb], a[fa], b[fb], acc[fa][fb]);
                    }
                }
            }
            if (h == NSUB - 1 && rs + 1 < nrs) {                 // nobody reads raw between the barrier above and the one below
                w2_raw_st<TY>(R, raw, tid);
                if (rs + 2 < nrs) w2_raw_ld<TY>(R, xb, row_bytes, rs + 2, tid);
            }
            __syncthreads();
        }
    }
    // full 16-row tiles straight to Y; the expert's last partial one through the warp's 16 x 16 scratch (ys is free now)
    float* scr = reinterpret_cast<float*>(ys) + warp * 256;
#pragma unroll
    for (int fa = 0; fa < 2; ++fa) {
        const int t0 = wc * 32 + fa * 16;
        if (t0 >= ncol) break;
#pragma unroll
        for (int fb = 0; fb < 2; ++fb) {
            const int nb = n0 + wr * 32 + fb * 16;
            if (t0 + 16 <= ncol) {
                wmma::store_matrix_sync(Y + (size_t) (c0 + t0) * ldy + nb, acc[fa][fb], ldy, wmma::mem_row_major);
            } else {
                wmma::store_matrix_sync(scr, acc[fa][fb], 16, wmma::mem_row_major);
                __syncwarp();
                for (int i = lane; i < 256; i += 32)
                    if (t0 + i / 16 < ncol) Y[(size_t) (c0 + t0 + i / 16) * ldy + nb + i % 16] = scr[i];
                __syncwarp();
            }
        }
    }
#endif
}

bool w2_on() {
    static const bool v = [] { const char* e = std::getenv("STRATA_PF_WMMA2"); return e && e[0] == '1'; }();
    return v;
}
}  // namespace

bool gemm_iq_f16_grouped(int ty, const void* W, size_t expert_bytes, int n_out, int K, const void* X, int ldx,
                         const int32_t* bounds, int n_experts, int max_rows, float* Y, int ldy, void* stream) {
    if (n_out % WG_BN != 0 || n_experts <= 0 || max_rows <= 0 || ldy % 4 != 0) return false;
    {   // the tiles are tensor-core kernels: a pre-Volta card has no body for them (the sm_60 pass compiles them empty)
        int cur = 0, major = 0;
        if (cudaGetDevice(&cur) != cudaSuccess || cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, cur) != cudaSuccess ||
            major < 7)
            return false;
    }
    // STRATA_PF_WMMA2=1 (opt-in): IQ2_S gate/up and IQ4_NL down through gemm_iq_f16_w2_kernel (above)
    if (w2_on() && n_out % W2_RB == 0 && ldx % 8 == 0 && ldy % 8 == 0 && (uintptr_t) X % 16 == 0 && (uintptr_t) Y % 32 == 0 &&
        (uintptr_t) W % 4 == 0 && expert_bytes % 4 == 0 && ((ty == 22 && K % 512 == 0) || (ty == 20 && K % 128 == 0))) {
        const size_t rb = iq_row_bytes(ty, K);
        const int ch = (max_rows + W2_CB - 1) / W2_CB;
        if (rb % 4 == 0 && (long long) n_experts * ch <= 65535) {
            const dim3 grid2((unsigned) (n_out / W2_RB), (unsigned) (n_experts * ch));
            cudaStream_t s2 = (cudaStream_t) stream;
            if (ty == 22)
                gemm_iq_f16_w2_kernel<22><<<grid2, W2_NT, 0, s2>>>((const uint8_t*) W, expert_bytes, rb, K, (const __half*) X, ldx, bounds, ch, Y, ldy);
            else
                gemm_iq_f16_w2_kernel<20><<<grid2, W2_NT, 0, s2>>>((const uint8_t*) W, expert_bytes, rb, K, (const __half*) X, ldx, bounds, ch, Y, ldy);
            return cudaGetLastError() == cudaSuccess;
        }
    }
    const bool q20 = ty == 42;
    const bool nl = ty == 20 && K % 256 != 0;   // IQ4_NL down (K=640): there is no WK=256 path for this K
    if (q20 || nl ? K % 128 != 0 : (K % 256 != 0 || !is_iq(ty))) return false;
    const size_t row_bytes = iq_row_bytes(ty, K);
    const int mt = (max_rows + WG_MT * WG_BM - 1) / (WG_MT * WG_BM);   // row chunks per expert
    const dim3 grid((unsigned) (n_out / WG_BN), (unsigned) (n_experts * mt));
    cudaStream_t s = (cudaStream_t) stream;
    auto smem = [](int wk) { return (size_t) WG_BN * (wk + 8) * 2 + (size_t) WG_BM * (WG_XK + 8) * 2; };
    // the shared-memory attribute belongs to the current device, so it is set on each card: with one flag for the
    // process a layer split's second card launched without it and the prompt stopped ("the WMMA expert tiles could not launch")
    static bool attr[64];
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    if (!attr[dev]) {
        cudaFuncSetAttribute(gemm_iq_f16_grouped_kernel<256, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem(256));
        cudaFuncSetAttribute(gemm_iq_f16_grouped_kernel<128, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem(128));
        cudaFuncSetAttribute(gemm_iq_f16_grouped_kernel<128, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem(128));
        attr[dev] = true;
    }
    if (q20)
        gemm_iq_f16_grouped_kernel<128, true><<<grid, WG_THREADS, smem(128), s>>>(ty, (const uint8_t*) W, expert_bytes, row_bytes, K,
                                                                                 (const __half*) X, ldx, bounds, mt, Y, ldy);
    else if (nl)
        gemm_iq_f16_grouped_kernel<128, false, true><<<grid, WG_THREADS, smem(128), s>>>(ty, (const uint8_t*) W, expert_bytes, row_bytes, K,
                                                                                         (const __half*) X, ldx, bounds, mt, Y, ldy);
    else
        gemm_iq_f16_grouped_kernel<256, false><<<grid, WG_THREADS, smem(256), s>>>(ty, (const uint8_t*) W, expert_bytes, row_bytes, K,
                                                                                  (const __half*) X, ldx, bounds, mt, Y, ldy);
    return cudaGetLastError() == cudaSuccess;
}
}  // namespace strata::kernels
#else
namespace strata::kernels {
bool gemm_iq_f16_grouped(int, const void*, size_t, int, int, const void*, int, const int32_t*, int, int, float*, int, void*) {
    return false;
}
}  // namespace strata::kernels
#endif
