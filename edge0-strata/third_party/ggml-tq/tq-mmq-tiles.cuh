// third_party/ggml-tq/tq-mmq-tiles.cuh - the MMQ tile loader of the trellis types (TQ2_T, TQK6, TQK7), appended to ggml-cuda's
// mmq-load-tiles.cuh by cmake/strata_ggml_tq_overlay.cmake when ggml comes from upstream llama.cpp (the agentionai fork has
// it natively).  The function is agentionai/llama.cpp's (ggml/src/ggml-cuda/mmq-load-tiles.cuh, MIT, the ggml authors,
// written by Laurent Zuijdwijk), unchanged; the other lines of this patch only route the trellis types to it the way
// IQ2_XS is routed (same Q8_0_16 tile layout, same dot kernels, same configs).
#pragma once
#include "tq.cuh"

// Trellis types (TQ2_T, TQK6, TQK7): a 128-weight block is TQ_LANES lanes of 16 weights. Each lane
// decodes its 4 trellis steps and requantizes them to int8 with one scale per 16 weights, filling
// the Q8_0_16 tile layout (as IQ2_XS). One MMQ_ITER_K slice of a row is 2 blocks = 16 lanes.
template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_tq(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    // ggml_cuda_mmq_get_util_funcs names this loader for every type; only trellis types reach it.
    constexpr ggml_type tq_type = ggml_cuda_type_is_tq(type) ? type : GGML_TYPE_TQ2_T;
    typedef typename tq_block_info<tq_type>::block_t block_t;
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(type, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    static_assert(MMQ_ITER_K % QK_TQK == 0, "bad MMQ_ITER_K");
    constexpr int threads_per_row = (MMQ_ITER_K / QK_TQK) * TQ_LANES; // 16 lanes x 16 weights = 256
    static_assert(warp_size % threads_per_row == 0, "bad warp size");
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;
    const int kbx  = kqsx / TQ_LANES;
    const int lane = kqsx % TQ_LANES;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_t * bxi = (const block_t *) x + kbx0 + i*stride + kbx;

        int q[TQ_STEPS_LANE];
        const float dq = tq_lane_to_q8<tq_type>(bxi, lane, q);

#pragma unroll
        for (int k = 0; k < TQ_STEPS_LANE; ++k) {
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride + 4*kqsx + k] = q[k];
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 4*kqsx + k] = q[k];
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride                             + kqsx] = dq;
#else
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + kqsx] = dq;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}
