// src/prefill/mmq_nvfp4_w4a4.cu - NVFP4 MMQ with FP4 activations on Blackwell (W4A4, opt-in: STRATA_PREFILL_NVFP4=w4a4).
//
// RTX 50's FP4 MMA exists only in the arch-specific target (sm_120a), so this is the one translation unit CMake
// builds with 12x -> 12xa (strata_mmq_w4a4); everything else, the default NVFP4 path included, keeps the plain
// architectures and runs on every Blackwell card (sm_121 too). Here mmq.cuh's NVFP4 tile loader and quantize.cu's
// FP4 activation quantizer see BLACKWELL_MMA_AVAILABLE together - they must agree on the activations' layout.
// Their non-static symbols are renamed so they cannot fold with strata_mmq's plain-arch copies.
#include "common.cuh"

#define mul_mat_q_switch_J strata_w4a4_mul_mat_q_switch_J
#define mul_mat_q_case strata_w4a4_mul_mat_q_case
#define quantize_row_q8_1_cuda strata_w4a4_quantize_row_q8_1_cuda
#define quantize_mmq_q8_1_cuda strata_w4a4_quantize_mmq_q8_1_cuda
#define quantize_scatter_mmq_q8_1_cuda strata_w4a4_quantize_scatter_mmq_q8_1_cuda
#define quantize_scatter_mmq_fp4_cuda strata_w4a4_quantize_scatter_mmq_fp4_cuda
#define quantize_mmq_fp4_cuda strata_w4a4_quantize_mmq_fp4_cuda
#include "mmq.cuh"
#include "quantize.cu"

DECL_MMQ_CASE(GGML_TYPE_NVFP4);

namespace {
// 1 when this unit's device code has the FP4 MMA (built for the running card's 12xa); a card without an image
// here (an architecture not in the list, or sm_121 against a 120a build) fails the launch: W4A4 is then off
__global__ void w4a4_probe_kernel(int* out) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    *out = 1;
#else
    *out = 0;
#endif
}
}  // namespace

namespace strata::prefill::mmq {
bool w4a4_available() {
    int* d = nullptr;
    int h = 0;
    if (cudaMalloc(&d, sizeof(int)) != cudaSuccess) { (void) cudaGetLastError(); return false; }
    w4a4_probe_kernel<<<1, 1>>>(d);
    const bool ran = cudaGetLastError() == cudaSuccess && cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess;
    (void) cudaGetLastError();
    cudaFree(d);
    return ran && h == 1;
}
void run_nvfp4_w4a4(ggml_backend_cuda_context& ctx, const mmq_args& a, cudaStream_t s) {
    strata_w4a4_mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, a, s);
}
void quantize_nvfp4_w4a4(const float* x, const int32_t* ids, void* xq, float* yscale, bool aligned, int64_t cols,
                         int64_t ld, int64_t rows, int64_t padded, cudaStream_t s) {
    strata_w4a4_quantize_mmq_fp4_cuda(x, ids, xq, yscale, GGML_TYPE_NVFP4, aligned, cols, ld, rows * ld, rows * ld,
                                      padded, rows, 1, 1, s);
}
}  // namespace strata::prefill::mmq
