// src/prefill/moe_mmq_tab.cuh - STRATA_MMQ_DIRECT: llama.cpp's MMQ outer scheduler (ggml-cuda mmq.cuh, MIT) with a
// per-expert pointer table, so the prompt path's MMQ reads resident experts in their expert-cache slots instead of
// a gathered copy.  Included only by the per-type instance TUs (CMake: mmq_tab_instance.cu.in); moe_mmq.cu sees
// the declarations in moe_mmq_tab.hpp.
//
// Mirrored from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d (the fetched pin; the vendored third_party tree's
// mmq.cuh is byte-identical to it), ggml/src/ggml-cuda/mmq.cuh:
//   mul_mat_q          lines  953-1238  -> mul_mat_q_tab
//   launch_mul_mat_q   lines 1395-1475  -> launch_mul_mat_q_tab
//   mul_mat_q_switch_J lines 1477-1558  -> mul_mat_q_tab_switch_J
// fallback = false only (mul_mat_q_tab_case refuses rows % 128 != 0).  Everything else is the upstream text, and
// mul_mat_q_process_tile, the tile loaders, vec_dot, write_back and mul_mat_q_stream_k_fixup are upstream's own
// templates from the same mmq.cuh, compiled in the same target with the same flags.  Deliberate deviations:
//   1. upstream's `const char * __restrict__ x` parameter becomes 16 such parameters x0..x15 (the table, see
//      STRATA_MMQ_TAB_PARAMS); at the three offset_x sites the expert's base is the one picked by
//      fastdiv(zt, channel_ratio), and the fastdiv(zt, channel_ratio)*stride_channel_x term leaves offset_x
//      (stride_channel_x is still passed, unused);
//   2. the launcher raises the shared-memory limit of mul_mat_q_tab<type, J> only - upstream's line for
//      mul_mat_q<type, J, true> is dropped, so no fallback=true kernel is instantiated.
// The launch geometry (J, tiles, the stream-k grid and k split, the fixup) is computed from the same mmq_args as
// upstream's, so every output tile is summed in the same order: the results are bitwise the same as gathering the
// experts `stride_channel_x` apart (tests/cuda/prefill_mmq_tab_test.cu checks it per type and J).
// CMake turns STRATA_MMQ_TAB off unless the SHA-256 of mmq.cuh's mirrored region (mul_mat_q_process_tile through
// mul_mat_q_case, which includes struct mmq_args) is one this copy was verified against; when the llama.cpp pin
// moves, re-sync this file, re-run the test and record the new hash in CMakeLists.txt (_strata_mmq_tab_known).
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include "mmq.cuh"
#include "moe_mmq_tab.hpp"

#include <type_traits>

namespace strata::prefill::mmq {

// A second net behind the configure fingerprint: the upstream templates this copy calls keep their signatures.
static_assert(std::is_same<decltype(&mul_mat_q_process_tile<GGML_TYPE_Q8_0, 8, false, false>),
                           void (*)(const char*, int, const int*, const int*, float*, float*, const float*, int, int, int,
                                    int, int, int, int)>::value,
              "mmq.cuh: mul_mat_q_process_tile changed - re-sync src/prefill/moe_mmq_tab.cuh");
static_assert(std::is_same<decltype(&mul_mat_q_stream_k_fixup<GGML_TYPE_Q8_0, 8, false>),
                           void (*)(const int32_t*, const int32_t*, float*, float*, uint3, int, int, int, uint3, int,
                                    uint3, int, uint3)>::value,
              "mmq.cuh: mul_mat_q_stream_k_fixup changed - re-sync src/prefill/moe_mmq_tab.cuh");

// The table reaches the kernel as 16 separate `const char * __restrict__` parameters, like upstream's `x`, and an
// expert's entry is picked with a select chain: a pointer picked from read-only kernel parameters keeps upstream's
// non-coherent global loads (LDG.CONSTANT) of the weights; a dynamically indexed array parameter would be copied to
// local memory and its pointers read through generic loads (measured ~1.2-2.5x slower products).
#define STRATA_MMQ_TAB_PARAMS                                                                                     \
    const char * __restrict__ x0, const char * __restrict__ x1, const char * __restrict__ x2,                  \
    const char * __restrict__ x3, const char * __restrict__ x4, const char * __restrict__ x5,                  \
    const char * __restrict__ x6, const char * __restrict__ x7, const char * __restrict__ x8,                  \
    const char * __restrict__ x9, const char * __restrict__ x10, const char * __restrict__ x11,                \
    const char * __restrict__ x12, const char * __restrict__ x13, const char * __restrict__ x14,               \
    const char * __restrict__ x15
#define STRATA_MMQ_TAB_PICK(z)                                                                                    \
    ((z) == 0 ? x0 : (z) == 1 ? x1 : (z) == 2 ? x2 : (z) == 3 ? x3 : (z) == 4 ? x4 : (z) == 5 ? x5 :            \
     (z) == 6 ? x6 : (z) == 7 ? x7 : (z) == 8 ? x8 : (z) == 9 ? x9 : (z) == 10 ? x10 : (z) == 11 ? x11 :        \
     (z) == 12 ? x12 : (z) == 13 ? x13 : (z) == 14 ? x14 : x15)
#define STRATA_MMQ_TAB_ARGS(t)                                                                                    \
    (t).p[0], (t).p[1], (t).p[2], (t).p[3], (t).p[4], (t).p[5], (t).p[6], (t).p[7], (t).p[8], (t).p[9],        \
    (t).p[10], (t).p[11], (t).p[12], (t).p[13], (t).p[14], (t).p[15]

// ---- mirrored: mul_mat_q (fallback = false, per-expert base from `tab`) ----
template <ggml_type type, int J>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(type, J, false), ggml_cuda_mmq_get_occupancy(type, J, false))
static __global__ void mul_mat_q_tab(
        STRATA_MMQ_TAB_PARAMS, const int * __restrict__ y, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, float * __restrict__ dst, float * __restrict__ tmp_fixup,
        const float * __restrict__ y_scale,
        const uint3 blocks_per_ne00, const int nrows_x, const int ncols_dst, const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const uint3 channel_ratio, const uint3 nchannels_y, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const uint3 nsamples_y, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const uint3 ntx) {
    constexpr bool fallback = false;  // STRATA: the tab copy has no fallback (rows % 128 == 0) instantiation

    // Skip unused template specializations for faster compilation:
    if (ggml_cuda_mmq_get_config(type, J, fallback).type == GGML_TYPE_COUNT) {
        NO_DEVICE_CODE;
        return;
    }

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int qk        = ggml_cuda_type_traits<type>::qk;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback);

    const uint32_t nty = (nrows_x + I - 1) / I; // Number of tiles y

    // Initialize the ids for writing back data with just the index.
    // For regular matrix multiplications this is never changed.
    // For MoE the correct indices are loaded from ids_dst.
    extern __shared__ int ids_dst_shared[]; // Stored at beginning of shared memory.
#pragma unroll
    for (int j0 = 0; j0 < J; j0 += nwarps*warp_size) {
        const int j = j0 + threadIdx.y*warp_size + threadIdx.x;

        if (j0 + nwarps*warp_size > J && j >= J) {
            break;
        }

        ids_dst_shared[j] = j;
    }
    __syncthreads();

    if constexpr (!ggml_cuda_mmq_get_stream_k(type, J, fallback)) {
        const uint2 tmp2 = fast_div_modulo(blockIdx.z, nchannels_y);
        const int wt = tmp2.x;
        const int zt = tmp2.y;
        const int jt = blockIdx.y;
        const int it = blockIdx.x;

        // Defaults for regular matrix multiplication:
        int col_low    = 0;
        int col_high   = ncols_dst;
        int col_diff   = ncols_dst;
        int offset_y       = wt*stride_sample_y   + zt*stride_channel_y;
        int offset_dst     = wt*stride_sample_dst + zt*stride_channel_dst + jt*J*stride_col_dst;
        int offset_y_scale;
        if constexpr (type == GGML_TYPE_NVFP4) {
            offset_y_scale = wt*nchannels_y.z*ncols_y + zt*ncols_y;
        } else {
            GGML_UNUSED(offset_y_scale);
        }

        if (ids_dst) {
            col_low  = expert_bounds[zt + 0];
            col_high = expert_bounds[zt + 1];
            col_diff = col_high - col_low;

            offset_y   = 0;
            offset_dst = 0;
            if constexpr (type == GGML_TYPE_NVFP4) {
                offset_y_scale = 0;
            }

            if (jt*J >= col_diff) {
                return;
            }

            // __syncthreads(); // There is no previous tile that could cause a race condition.
#pragma unroll
            for (int j0 = 0; j0 < J; j0 += nwarps*warp_size) {
                const int j = j0 + threadIdx.y*warp_size + threadIdx.x;

                if (j0 + nwarps*warp_size > J && j >= J) {
                    break;
                }

                ids_dst_shared[j] = ids_dst[col_low + jt*J + j];
            }
            __syncthreads();
        }

        offset_y   += (col_low + jt*J)*(sizeof(block_q8_1_mmq)/sizeof(int));
        offset_dst += it*I;
        const float * y_scale_tile = nullptr;
        if constexpr (type == GGML_TYPE_NVFP4) {
            offset_y_scale += col_low + jt*J;
            y_scale_tile = y_scale ? y_scale + offset_y_scale : nullptr;
        }

        const int tile_x_max_i = nrows_x  - it*I - 1;
        const int tile_y_max_j = col_diff - jt*J - 1;

        const char * xz = STRATA_MMQ_TAB_PICK(fastdiv(zt, channel_ratio));  // STRATA: per-expert base
        const int offset_x = fastdiv(wt, sample_ratio)*stride_sample_x + it*I*stride_row_x;

        constexpr bool fixup = false;
        mul_mat_q_process_tile<type, J, fallback, fixup>
            (xz, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, y_scale_tile,
             stride_row_x, ncols_y, stride_col_dst,
             tile_x_max_i, tile_y_max_j, 0, blocks_per_ne00.z);
        return;
    }

    constexpr int ITER_K          = ggml_cuda_mmq_get_K_vram(type, J, fallback);
    constexpr int blocks_per_iter = ITER_K / qk;

    // kbc == k block continuous, current index in continuous ijk space.
    int kbc      = int64_t(blockIdx.x)    *(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / gridDim.x;
    int kbc_stop = int64_t(blockIdx.x + 1)*(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / gridDim.x;

    kbc      -= fastmodulo(kbc,      blocks_per_ne00) % blocks_per_iter;
    kbc_stop -= fastmodulo(kbc_stop, blocks_per_ne00) % blocks_per_iter;

    // kb0 == k index when doing the matrix multiplication for an output tile.
    int kb0_start = fastmodulo(kbc, blocks_per_ne00);
    int kb0_stop  = min(blocks_per_ne00.z, uint32_t(kb0_start + kbc_stop - kbc));
    while (kbc < kbc_stop && kb0_stop == int(blocks_per_ne00.z)) {
        int tmp = fastdiv(kbc, blocks_per_ne00);
        uint2 tmp2 = fast_div_modulo(tmp, ntx);
        const int jt = tmp2.y;
        tmp = tmp2.x;
        tmp2 = fast_div_modulo(tmp, nchannels_y);
        const int zt = tmp2.y;
        tmp = tmp2.x;
        tmp2 = fast_div_modulo(tmp, nsamples_y);
        const int wt = tmp2.y;
        const int it = tmp2.x;

        // Defaults for regular matrix multiplication:
        int col_low    = 0;
        int col_high   = ncols_dst;
        int col_diff   = ncols_dst;
        int offset_y       = wt*stride_sample_y   + zt*stride_channel_y;
        int offset_dst     = wt*stride_sample_dst + zt*stride_channel_dst + jt*J*stride_col_dst;
        int offset_y_scale;
        if constexpr (type == GGML_TYPE_NVFP4) {
            offset_y_scale = wt*nchannels_y.z*ncols_y + zt*ncols_y;
        } else {
            GGML_UNUSED(offset_y_scale);
        }

        if (ids_dst) {
            col_low  = expert_bounds[zt + 0];
            col_high = expert_bounds[zt + 1];
            col_diff = col_high - col_low;

            offset_y   = 0;
            offset_dst = 0;
            if constexpr (type == GGML_TYPE_NVFP4) {
                offset_y_scale = 0;
            }

            if (jt*J >= col_diff) {
                kbc += blocks_per_ne00.z;
                kbc -= fastmodulo(kbc, blocks_per_ne00);

                kb0_start = 0;
                kb0_stop  = min(blocks_per_ne00.z, uint32_t(kbc_stop - kbc));

                continue;
            }

            __syncthreads();
#pragma unroll
            for (int j0 = 0; j0 < J; j0 += nwarps*warp_size) {
                const int j = j0 + threadIdx.y*warp_size + threadIdx.x;

                if (j0 + nwarps*warp_size > J && j >= J) {
                    break;
                }

                ids_dst_shared[j] = ids_dst[col_low + jt*J + j];
            }
            __syncthreads();
        }

        offset_y += (col_low + jt * J) * (sizeof(block_q8_1_mmq) / sizeof(int));
        offset_dst += it*I;
        const float * y_scale_tile = nullptr;
        if constexpr (type == GGML_TYPE_NVFP4) {
            offset_y_scale += col_low + jt * J;
            y_scale_tile = y_scale ? y_scale + offset_y_scale : nullptr;
        }

        const int tile_x_max_i = nrows_x  - it*I - 1;
        const int tile_y_max_j = col_diff - jt*J - 1;

        const char * xz = STRATA_MMQ_TAB_PICK(fastdiv(zt, channel_ratio));  // STRATA: per-expert base
        const int offset_x = fastdiv(wt, sample_ratio)*stride_sample_x + it*I*stride_row_x;

        constexpr bool fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
        mul_mat_q_process_tile<type, J, fallback, fixup>
            (xz, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, y_scale_tile,
             stride_row_x, ncols_y, stride_col_dst,
             tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);

        kbc += blocks_per_ne00.z;
        kbc -= fastmodulo(kbc, blocks_per_ne00);

        kb0_start = 0;
        kb0_stop  = min(blocks_per_ne00.z, uint32_t(kbc_stop - kbc));
    }

    if (kbc >= kbc_stop) {
        return;
    }

    int tmp = fastdiv(kbc, blocks_per_ne00);
    uint2 tmp2 = fast_div_modulo(tmp, ntx);
    const int jt = tmp2.y;
    tmp = tmp2.x;
    tmp2 = fast_div_modulo(tmp, nchannels_y);
    const int zt = tmp2.y;
    tmp = tmp2.x;
    tmp2 = fast_div_modulo(tmp, nsamples_y);
    const int wt = tmp2.y;
    const int it = tmp2.x;

    // Defaults for regular matrix multiplication:
    int col_low    = 0;
    int col_high   = ncols_dst;
    int col_diff   = ncols_dst;
    int offset_y       = wt*stride_sample_y   + zt*stride_channel_y;
    int offset_dst     = wt*stride_sample_dst + zt*stride_channel_dst + jt*J*stride_col_dst;
    int offset_y_scale;
    if constexpr (type == GGML_TYPE_NVFP4) {
        offset_y_scale = wt*nchannels_y.z*ncols_y + zt*ncols_y;
    } else {
        GGML_UNUSED(offset_y_scale);
    }

    if (ids_dst) {
        col_low  = expert_bounds[zt + 0];
        col_high = expert_bounds[zt + 1];
        col_diff = col_high - col_low;

        offset_y   = 0;
        offset_dst = 0;
        if constexpr (type == GGML_TYPE_NVFP4) {
            offset_y_scale = 0;
        }

        if (jt*J >= col_diff) {
            return;
        }

        // The memory layout for the fixup buffer is always contiguous, therefore reset ids:
        __syncthreads();
#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps*warp_size) {
            const int j = j0 + threadIdx.y*warp_size + threadIdx.x;

            if (j0 + nwarps*warp_size > J && j >= J) {
                break;
            }

            ids_dst_shared[j] = j;
        }
        __syncthreads();
    }

    offset_y += (col_low + jt * J) * (sizeof(block_q8_1_mmq) / sizeof(int));
    offset_dst += it*I;
    const float * y_scale_tile = nullptr;
    if constexpr (type == GGML_TYPE_NVFP4) {
        offset_y_scale += col_low + jt * J;
        y_scale_tile = y_scale ? y_scale + offset_y_scale : nullptr;
    }

    const int tile_x_max_i = nrows_x  - it*I - 1;
    const int tile_y_max_j = col_diff - jt*J - 1;

    const char * xz = STRATA_MMQ_TAB_PICK(fastdiv(zt, channel_ratio));  // STRATA: per-expert base
    const int offset_x = fastdiv(wt, sample_ratio)*stride_sample_x + it*I*stride_row_x;

    constexpr bool fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
    mul_mat_q_process_tile<type, J, fallback, fixup>
        (xz, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, y_scale_tile,
         stride_row_x, ncols_y, stride_col_dst,
         tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);
}

// ---- mirrored: launch_mul_mat_q (fallback = false) ----
template <ggml_type type, int J>
static void launch_mul_mat_q_tab(ggml_backend_cuda_context & ctx, const mmq_args & args, const mmq_x_tab & tab, cudaStream_t stream) {
    constexpr bool fallback = false;
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;
    const int warp_size = ggml_cuda_info().devices[id].warp_size;

    const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc);
    GGML_ASSERT(config.nthreads % warp_size == 0);
    const int nwarps = config.nthreads / warp_size;
    const int nbytes_shared = mmq_get_nbytes_shared(config, cc);

    const dim3 block_dims(warp_size, nwarps, 1);

    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q_tab<type, J>), nbytes_shared);  // STRATA: no fallback=true line

    const int nty  = (args.nrows_x   + config.I - 1) / config.I;
    const int ntx  = (args.ncols_max + config.J - 1) / config.J;
    const int ntzw = args.nchannels_y * args.nsamples_y;
    const dim3 block_nums_xy_tiling(nty, ntx, ntzw);

    GGML_ASSERT(args.nchannels_y % args.nchannels_x == 0);
    GGML_ASSERT(args.nsamples_y  % args.nsamples_x  == 0);
    const int channel_ratio = args.nchannels_y / args.nchannels_x;
    const int sample_ratio  = args.nsamples_y  / args.nsamples_x;

    const uint3 blocks_per_ne00_fd = init_fastdiv_values(args.ncols_x / ggml_cuda_type_traits<type>::qk);
    const uint3 ntx_fd             = init_fastdiv_values(ntx);
    const uint3 nchannels_y_fd     = init_fastdiv_values(args.nchannels_y);
    const uint3 nsamples_y_fd      = init_fastdiv_values(args.nsamples_y);
    const uint3 channel_ratio_fd   = init_fastdiv_values(channel_ratio);
    const uint3 sample_ratio_fd    = init_fastdiv_values(sample_ratio);

    if (!ggml_cuda_mmq_get_stream_k(type, J, fallback, cc)) {
        mul_mat_q_tab<type, J><<<block_nums_xy_tiling, block_dims, nbytes_shared, stream>>>
            (STRATA_MMQ_TAB_ARGS(tab), args.y, args.ids_dst, args.expert_bounds, args.dst, nullptr, args.y_scale,
             blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
             channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
             sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
             ntx_fd);
        return;
    }

    // For the stream-k kernel it is possible to run it with tiling by setting the number of CUDA blocks equal to the number of tiles.
    // This is worthwhile if the efficiency of tiling is high and skipping the fixup kernel is more important.
    const int ntiles_dst = ntx * nty * ntzw;
    const int tiles_nwaves = (ntiles_dst + nsm - 1) / nsm;
    const int tiles_efficiency_percent = 100 * ntiles_dst / (nsm*tiles_nwaves);
    const dim3 block_nums_stream_k(GGML_CUDA_CC_IS_NVIDIA(cc) && tiles_efficiency_percent >= 90 ? ntiles_dst : nsm, 1, 1);

    GGML_ASSERT(ntiles_dst * blocks_per_ne00_fd.z < (1 << 30)); // Assert that variable kbc will not overflow.

    const bool fixup_needed = ntiles_dst % block_nums_stream_k.x != 0;

    ggml_cuda_pool & pool = ctx.pool(id);
    ggml_cuda_pool_alloc<float> tmp_fixup(pool);
    if (fixup_needed) {
        tmp_fixup.alloc(block_nums_stream_k.x * config.J*config.I);
    }

    const dim3 block_nums_fixup(block_nums_stream_k.x, config.I/warp_size, 1);
    const dim3 block_dims_fixup(block_dims.x, block_dims.y/2, block_dims.z);

    mul_mat_q_tab<type, J><<<block_nums_stream_k, block_dims, nbytes_shared, stream>>>
        (STRATA_MMQ_TAB_ARGS(tab), args.y, args.ids_dst, args.expert_bounds, args.dst, tmp_fixup.ptr, args.y_scale,
         blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
         channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
         sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
         ntx_fd);

    if (!fixup_needed) {
        return;
    }

    CUDA_CHECK(cudaGetLastError());
    mul_mat_q_stream_k_fixup<type, J, fallback><<<block_nums_fixup, block_dims_fixup, 0, stream>>>
        (args.ids_dst, args.expert_bounds, args.dst, tmp_fixup.ptr, blocks_per_ne00_fd, args.nrows_x, args.ncols_dst,
         args.nrows_dst, nchannels_y_fd, args.stride_channel_dst, nsamples_y_fd, args.stride_sample_dst,
         ntx_fd);
}

// ---- mirrored: mul_mat_q_switch_J (fallback = false) ----
template <ggml_type type>
static void mul_mat_q_tab_switch_J(ggml_backend_cuda_context & ctx, const mmq_args & args, const mmq_x_tab & tab, cudaStream_t stream) {
    constexpr bool fallback = false;
    const int    id    = ggml_cuda_get_device();
    const int    cc    = ggml_cuda_info().devices[id].cc;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    int J_best        = 0;
    int ntiles_J_best = INT_MAX;

    for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8) {
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc);
        if (config.type == GGML_TYPE_COUNT) {
            continue;
        }

        if (mmq_get_nbytes_shared(config, cc) > smpbo) {
            continue;
        }

        const int ntiles_x = (args.ncols_opt + config.J - 1) / config.J;

        if (ntiles_x < ntiles_J_best) {
            J_best = J;
            ntiles_J_best = ntiles_x;
        }
    }

    switch (J_best) {
        case   8:
            launch_mul_mat_q_tab<type, 8>(ctx, args, tab, stream);
            break;
        case  16:
            launch_mul_mat_q_tab<type, 16>(ctx, args, tab, stream);
            break;
        case  24:
            launch_mul_mat_q_tab<type, 24>(ctx, args, tab, stream);
            break;
        case  32:
            launch_mul_mat_q_tab<type, 32>(ctx, args, tab, stream);
            break;
        case  40:
            launch_mul_mat_q_tab<type, 40>(ctx, args, tab, stream);
            break;
        case  48:
            launch_mul_mat_q_tab<type, 48>(ctx, args, tab, stream);
            break;
        case  56:
            launch_mul_mat_q_tab<type, 56>(ctx, args, tab, stream);
            break;
        case  64:
            launch_mul_mat_q_tab<type, 64>(ctx, args, tab, stream);
            break;
        case  72:
            launch_mul_mat_q_tab<type, 72>(ctx, args, tab, stream);
            break;
        case  80:
            launch_mul_mat_q_tab<type, 80>(ctx, args, tab, stream);
            break;
        case  88:
            launch_mul_mat_q_tab<type, 88>(ctx, args, tab, stream);
            break;
        case  96:
            launch_mul_mat_q_tab<type, 96>(ctx, args, tab, stream);
            break;
        case 104:
            launch_mul_mat_q_tab<type, 104>(ctx, args, tab, stream);
            break;
        case 112:
            launch_mul_mat_q_tab<type, 112>(ctx, args, tab, stream);
            break;
        case 120:
            launch_mul_mat_q_tab<type, 120>(ctx, args, tab, stream);
            break;
        case 128:
            launch_mul_mat_q_tab<type, 128>(ctx, args, tab, stream);
            break;
        default:
            fprintf(stderr, "J_best=%d\n", J_best);
            GGML_ABORT("fatal error");
            break;
    }
}

template <ggml_type T>
bool mul_mat_q_tab_case(ggml_backend_cuda_context& ctx, const mmq_args& a, const mmq_x_tab& tab, cudaStream_t s) {
    if (a.nrows_x % 128 != 0) return false;   // upstream would take the fallback configs: not mirrored
    mul_mat_q_tab_switch_J<T>(ctx, a, tab, s);
    return true;
}

}  // namespace strata::prefill::mmq
