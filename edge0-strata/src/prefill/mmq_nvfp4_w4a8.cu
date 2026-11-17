// src/prefill/mmq_nvfp4_w4a8.cu - NVFP4 MMQ with 8-bit activations on Blackwell (W4A8).
//
// On sm_120 mmq.cuh compiles NVFP4 as FP4 x FP4 (W4A4): the activations are rounded to E2M1 too, and the first
// token's distribution drifts by KL 0.0004-0.03 from the exact FP16 prefill (8 prompts). Every other GPU gets
// mmq.cuh's int8 path for NVFP4 instead - weights decoded to int8, activations Q8_1, int8 tensor cores - which is
// the precision decode already runs at (vec_dot_nvfp4_q8_1, ggml-cpu's Q8_0). This TU compiles that path for
// sm_120 by hiding Blackwell's FP4 MMA from mmq.cuh on both sides: the device code (BLACKWELL_MMA_AVAILABLE) and
// the host's tile config (blackwell_mma_available(cc)) - they must agree, the host sizes shared memory for the
// device's layout. The two non-static templates are renamed so the linker cannot fold them with the W4A4 instance.
#include "common.cuh"

#undef BLACKWELL_MMA_AVAILABLE
#define blackwell_mma_available(cc) false
#define mul_mat_q_switch_J strata_w4a8_mul_mat_q_switch_J
#define mul_mat_q_case strata_w4a8_mul_mat_q_case
#include "mmq.cuh"

DECL_MMQ_CASE(GGML_TYPE_NVFP4);

namespace strata::prefill::mmq {
void run_nvfp4_w4a8(ggml_backend_cuda_context& ctx, const mmq_args& a, cudaStream_t s) {
    strata_w4a8_mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, a, s);
}
}  // namespace strata::prefill::mmq
