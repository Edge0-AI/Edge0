// src/prefill/gemm_bf16_parity.cu - Gemm::bf16 against cuBLAS's own BF16 product (GPU, synthetic, no model).
//
// Below sm_80 (no BF16 tensor cores) Gemm::bf16 converts the weight and the activations to FP16 and runs the FP16
// tensor-core GEMM (Volta by default, Turing with STRATA_BF16_TC=1) or widens them to fp32 (Pascal); everywhere else
// it is the cuBLAS BF16 call itself.  (On Pascal the cuBLAS BF16 reference itself may be refused: the test then fails
// at the reference, which says so.)  This
// checks the result against cublasGemmEx on the same BF16 inputs, within fp32-accumulation rounding, over the prompt
// path's shapes: the hyper-connection down / up projections, the router and indexer rows, the PLE value matrix, a T
// large enough to slice the activations, beta = 1 accumulation (the bf16x2 low parts) and an output row stride wider
// than N.  --bench adds the time of each against the cuBLAS BF16 product.
#include "strata/prefill/gemm.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

uint16_t to_bf16(float f) {   // round to nearest even
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

bool ok_or(cudaError_t e, const char* what) {
    if (e != cudaSuccess) std::printf("FAIL: %s: %s\n", what, cudaGetErrorString(e));
    return e == cudaSuccess;
}

struct Shape {
    int64_t T, N, K, ldy;
    const char* what;
};

}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    int dev = 0, maj = 0, min = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&min, cudaDevAttrComputeCapabilityMinor, dev);
    std::printf("gemm_bf16_parity: compute capability %d.%d\n", maj, min);

    cudaStream_t st = nullptr;
    if (!ok_or(cudaStreamCreate(&st), "stream")) return 1;
    strata::prefill::Gemm gemm;
    std::string err;
    if (!gemm.init(st, 0, err)) { std::printf("FAIL: %s\n", err.c_str()); return 1; }
    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { std::printf("FAIL: cublasCreate\n"); return 1; }
    cublasSetStream(h, st);

    const Shape shapes[] = {
        {4096, 320, 10240, 0, "hc down (activations sliced)"},
        {4096, 10240, 320, 0, "hc up"},
        {777, 2560, 2560, 0, "PLE value"},
        {512, 512, 2560, 0, "router / indexer q"},
        {300, 4, 10240, 0, "hc inject"},
        {2048, 1, 2560, 0, "single row"},
        {333, 128, 2560, 136, "row stride wider than N"},
    };
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    int failures = 0;
    for (const Shape& s : shapes) {
        const int64_t ldy = s.ldy > 0 ? s.ldy : s.N;
        std::vector<uint16_t> x((size_t) (s.T * s.K)), xlo(x.size()), w((size_t) (s.N * s.K));
        for (size_t i = 0; i < x.size(); ++i) {
            const float v = 3.0f * nd(rng);          // activations: normalized, a few units
            x[i] = to_bf16(v);
            xlo[i] = to_bf16(1e-3f * nd(rng));       // a bf16x2 low part
        }
        for (auto& v : w) v = to_bf16(0.02f * nd(rng));
        uint16_t *dx = nullptr, *dxlo = nullptr, *dw = nullptr;
        float *dy = nullptr, *dr = nullptr;
        const size_t ybytes = (size_t) (s.T * ldy) * 4;
        bool ok = ok_or(cudaMalloc((void**) &dx, x.size() * 2), "x") && ok_or(cudaMalloc((void**) &dxlo, x.size() * 2), "xlo") &&
                  ok_or(cudaMalloc((void**) &dw, w.size() * 2), "w") && ok_or(cudaMalloc((void**) &dy, ybytes), "y") &&
                  ok_or(cudaMalloc((void**) &dr, ybytes), "ref");
        if (!ok) return 1;
        cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dxlo, xlo.data(), xlo.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice);
        cudaMemset(dy, 0, ybytes);
        cudaMemset(dr, 0, ybytes);

        // the product under test: X . W^T, then the low part added with beta = 1
        gemm.bf16(dx, dw, dy, s.T, s.N, s.K, ldy);
        gemm.bf16(dxlo, dw, dy, s.T, s.N, s.K, ldy, 1.0f);
        // the reference: cuBLAS on the BF16 inputs, the same two calls
        const float one = 1.0f, zero = 0.0f;
        auto ref = [&](const uint16_t* X, float beta) {
            return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, (int) s.N, (int) s.T, (int) s.K, &one, dw, CUDA_R_16BF,
                                (int) s.K, X, CUDA_R_16BF, (int) s.K, &beta, dr, CUDA_R_32F, (int) ldy,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        };
        if (ref(dx, zero) != CUBLAS_STATUS_SUCCESS || ref(dxlo, one) != CUBLAS_STATUS_SUCCESS) {
            std::printf("FAIL: reference cublasGemmEx\n");
            return 1;
        }
        if (!ok_or(cudaStreamSynchronize(st), "sync")) return 1;
        std::vector<float> y((size_t) (s.T * ldy)), r(y.size());
        cudaMemcpy(y.data(), dy, ybytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(r.data(), dr, ybytes, cudaMemcpyDeviceToHost);
        double worst = 0.0, mag = 1e-30;
        int bad = 0;
        for (int64_t t = 0; t < s.T; ++t)
            for (int64_t c = 0; c < s.N; ++c) {
                const float a = y[(size_t) (t * ldy + c)], b = r[(size_t) (t * ldy + c)];
                if (!std::isfinite(a)) ++bad;
                worst = std::max(worst, (double) std::fabs(a - b));
                mag = std::max(mag, (double) std::fabs(b));
            }
        // the untouched columns of a wider row stride stay as they were
        for (int64_t t = 0; t < s.T && ldy > s.N; ++t)
            for (int64_t c = s.N; c < ldy; ++c)
                if (y[(size_t) (t * ldy + c)] != 0.0f) ++bad;
        const bool pass = bad == 0 && worst <= 1e-4 * mag;
        if (!pass) ++failures;
        std::printf("  %-32s T %5lld N %5lld K %5lld ldy %5lld: worst |diff| %.3e of max |ref| %.3e (rel %.2e) %s\n",
                    s.what, (long long) s.T, (long long) s.N, (long long) s.K, (long long) ldy, worst, mag, worst / mag,
                    pass ? "pass" : "FAIL");

        if (bench) {
            auto time = [&](auto&& f) {
                for (int i = 0; i < 3; ++i) f();
                cudaStreamSynchronize(st);
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < 20; ++i) f();
                cudaStreamSynchronize(st);
                return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 20;
            };
            const double us_gemm = time([&] { gemm.bf16(dx, dw, dy, s.T, s.N, s.K, ldy); });
            const double us_ref = time([&] { ref(dx, zero); });
            std::printf("      bench: Gemm::bf16 %9.1f us   cuBLAS BF16 %9.1f us   (%.2fx)\n", us_gemm, us_ref,
                        us_ref / us_gemm);
        }
        cudaFree(dx); cudaFree(dxlo); cudaFree(dw); cudaFree(dy); cudaFree(dr);
    }
    cublasDestroy(h);
    cudaStreamDestroy(st);
    std::printf("gemm_bf16_parity: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
