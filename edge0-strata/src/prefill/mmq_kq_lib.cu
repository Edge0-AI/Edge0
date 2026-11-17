// src/prefill/mmq_kq_lib.cu - strata_kq, the shared library with llama.cpp's MMQ prompt kernels for the K-quants
// (Q4_K, Q5_K, Q5_1, Q6_K: the experts of Unsloth's UD-Q4/Q5/Q6_K_XL files).  MODEL-UPDATE Phase 1: the engine loads
// every kernel of its own binary when it starts (CUDA_MODULE_LOADING=EAGER, src/program/generate.cpp), so kernels in
// the engine cost every model VRAM; these four instances were ~3.8 MB of sm_60 code (3 expert slots of IQ3_XXS on a
// P100).  Here they load only when moe_mmq.cu opens this library - for a model with these experts, before the expert
// cache is sized - and never otherwise.  The library has its own copy of ggml_cuda_host.cu's symbols and of the static
// CUDA runtime; it works in the engine's context on the stream it is given.
#include "common.cuh"
#include "mmq.cuh"

#include "strata/prefill/mmq_kq_abi.hpp"

#include <mutex>

#if defined(_WIN32)
#define STRATA_KQ_EXPORT extern "C" __declspec(dllexport)
#else
#define STRATA_KQ_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {
std::mutex g_mu;
ggml_backend_cuda_context* g_ctx[GGML_CUDA_MAX_DEVICES] = {};

ggml_backend_cuda_context* ctx_for(int dev) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_ctx[dev] == nullptr) g_ctx[dev] = new ggml_backend_cuda_context(dev);
    return g_ctx[dev];
}
}  // namespace

// the interface version and the argument block's size: both sides are compiled from the same mmq.cuh
STRATA_KQ_EXPORT int strata_kq_abi() { return STRATA_KQ_ABI_VERSION * 100000 + (int) sizeof(mmq_args); }

// Starts this library's CUDA runtime in the engine's context on `device`: with CUDA_MODULE_LOADING=EAGER (the engine
// sets it before its first CUDA call) that loads every kernel here now, while the VRAM for it is still free.
// 0 = ready, else the cudaError_t.
STRATA_KQ_EXPORT int strata_kq_init(int device) {
    cudaError_t e = cudaSetDevice(device);
    if (e == cudaSuccess) e = cudaFree(nullptr);
    return (int) e;
}

// One MMQ product (moe_mmq.cu's Context::run for these types) on `device`, on the engine's `stream`.  0 = launched, -1 =
// a type this library does not have, else the cudaError_t of the launch.
STRATA_KQ_EXPORT int strata_kq_run(int device, int type, const void* args, void* stream) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) return -1;
    if (cudaSetDevice(device) != cudaSuccess) return (int) cudaGetLastError();
    ggml_backend_cuda_context& ctx = *ctx_for(device);
    const mmq_args& a = *(const mmq_args*) args;
    const cudaStream_t s = (cudaStream_t) stream;
    switch ((ggml_type) type) {
        case GGML_TYPE_Q4_K: mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, a, s); break;
        case GGML_TYPE_Q5_K: mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, a, s); break;
        case GGML_TYPE_Q5_1: mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, a, s); break;
        case GGML_TYPE_Q6_K: mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, a, s); break;
        default: return -1;
    }
    return (int) cudaGetLastError();
}
