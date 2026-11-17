// src/prefill/moe_mmq_tab.hpp - STRATA_MMQ_DIRECT: declarations of the pointer-table MMQ (moe_mmq_tab.cuh).
// moe_mmq.cu includes only this header, so the tab kernels are instantiated once, in the per-type TUs that CMake
// generates from mmq_tab_instance.cu.in (STRATA_MMQ_TAB on), never implicitly in moe_mmq.cu.
#pragma once

#include "mmq.cuh"

namespace strata::prefill::mmq {

/// Expert z's weight matrix starts at p[z] (z < the launch's nchannels <= 16).
struct mmq_x_tab {
    const char* p[16];
};

/// llama.cpp's mul_mat_q_case<T> for fallback=false, with the per-expert base address from `tab` instead of
/// x + z * stride_channel_x.  false (nothing launched) when the rows are not a multiple of 128.
template <ggml_type T>
bool mul_mat_q_tab_case(ggml_backend_cuda_context& ctx, const mmq_args& a, const mmq_x_tab& tab, cudaStream_t s);

extern template bool mul_mat_q_tab_case<GGML_TYPE_Q2_0>(ggml_backend_cuda_context&, const mmq_args&, const mmq_x_tab&, cudaStream_t);
extern template bool mul_mat_q_tab_case<GGML_TYPE_IQ2_XXS>(ggml_backend_cuda_context&, const mmq_args&, const mmq_x_tab&, cudaStream_t);
extern template bool mul_mat_q_tab_case<GGML_TYPE_IQ2_XS>(ggml_backend_cuda_context&, const mmq_args&, const mmq_x_tab&, cudaStream_t);
extern template bool mul_mat_q_tab_case<GGML_TYPE_IQ2_S>(ggml_backend_cuda_context&, const mmq_args&, const mmq_x_tab&, cudaStream_t);
extern template bool mul_mat_q_tab_case<GGML_TYPE_IQ3_XXS>(ggml_backend_cuda_context&, const mmq_args&, const mmq_x_tab&, cudaStream_t);
extern template bool mul_mat_q_tab_case<GGML_TYPE_IQ3_S>(ggml_backend_cuda_context&, const mmq_args&, const mmq_x_tab&, cudaStream_t);
extern template bool mul_mat_q_tab_case<GGML_TYPE_IQ4_NL>(ggml_backend_cuda_context&, const mmq_args&, const mmq_x_tab&, cudaStream_t);
extern template bool mul_mat_q_tab_case<GGML_TYPE_IQ4_XS>(ggml_backend_cuda_context&, const mmq_args&, const mmq_x_tab&, cudaStream_t);
extern template bool mul_mat_q_tab_case<GGML_TYPE_Q8_0>(ggml_backend_cuda_context&, const mmq_args&, const mmq_x_tab&, cudaStream_t);

}  // namespace strata::prefill::mmq
