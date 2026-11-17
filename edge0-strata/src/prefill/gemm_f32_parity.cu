// src/prefill/gemm_f32_parity.cu - Pascal: Gemm::f16's opt-in FP32 route (cublasSgemm on FP32 copies of the FP16
// operands, STRATA_DENSE_F32=1) against the cuBLAS FP16-operand call it replaces, bitwise, over the dense projections'
// shapes and a sweep of chunk sizes T (GPU, synthetic, no model).  Whether the two agree is a property of the cuBLAS
// build (measured: 12.4.5.8 on a Tesla P40, 5,201 sizes): run this before setting STRATA_DENSE_F32=1.
//
//     build/gemm_f32_parity [--quick | --full] [--bench]
// default: every 7th T from the route's first to 8192, every T of 7000-7100 and around the powers of two, 8192;
// --quick: every 251st T; --full: every T.  --bench prints both times for a few T.  Exit 0: every output bitwise
// equal (or the route is not built here: not the experimental CUDA 12 engine).
#include "strata/prefill/gemm.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace {
__global__ void fill_f16(__half* p, int64_t n, unsigned seed, float scale) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned x = (unsigned) i * 2654435761u + seed;
    x ^= x >> 16; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
    p[i] = __float2half(((float) (x & 0xffff) / 32768.0f - 1.0f) * scale);
}
__global__ void count_diff(const unsigned* a, const unsigned* b, int64_t n, unsigned* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && a[i] != b[i]) atomicAdd(out, 1u);
}
bool ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) std::printf("FAIL: %s: %s\n", what, cudaGetErrorString(e));
    return e == cudaSuccess;
}
struct Shape { const char* what; int64_t N, K, t_min; };
}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    bool quick = false, full = false, bench = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        quick |= a == "--quick"; full |= a == "--full"; bench |= a == "--bench";
    }
    int dev = 0, maj = 0, min = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&min, cudaDevAttrComputeCapabilityMinor, dev);
    std::printf("gemm_f32_parity: compute capability %d.%d\n", maj, min);
    cudaStream_t st = nullptr;
    if (!ok(cudaStreamCreate(&st), "stream")) return 1;
    strata::prefill::Gemm gemm;
    std::string err;
    if (!gemm.init(st, 0, err)) { std::printf("FAIL: %s\n", err.c_str()); return 1; }
    // the shapes the route takes and its first T for each (gemm.cu, dense_f32_min_t)
    const Shape shapes[] = {{"gdn qkv", 10240, 2560, 256}, {"qsa query", 12288, 2560, 256}, {"gdn gate", 6144, 2560, 512},
                            {"gdn out", 2560, 6144, 4096}, {"qsa out", 2560, 4096, 4096}};
    const int64_t TMAX = 8192;
    unsigned* cnt = nullptr;
    if (!ok(cudaMalloc((void**) &cnt, 4), "counter")) return 1;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    long long cases = 0, bad_cases = 0;
    for (const Shape& s : shapes) {
        const int64_t N = s.N, K = s.K;
        uint16_t *W = nullptr, *X = nullptr;
        float *Ya = nullptr, *Yb = nullptr;
        if (!ok(cudaMalloc((void**) &W, (size_t) N * K * 2), "W") || !ok(cudaMalloc((void**) &X, (size_t) TMAX * K * 2), "X") ||
            !ok(cudaMalloc((void**) &Ya, (size_t) TMAX * N * 4), "Ya") || !ok(cudaMalloc((void**) &Yb, (size_t) TMAX * N * 4), "Yb")) return 1;
        fill_f16<<<(unsigned) ((N * K + 255) / 256), 256, 0, st>>>((__half*) W, N * K, 1u + (unsigned) N, 0.05f);
        fill_f16<<<(unsigned) ((TMAX * K + 255) / 256), 256, 0, st>>>((__half*) X, TMAX * K, 7u + (unsigned) K, 2.0f);
        std::set<int64_t> ts;
        const int64_t step = full ? 1 : quick ? 251 : 7;
        for (int64_t t = s.t_min; t <= TMAX; t += step) ts.insert(t);
        ts.insert(s.t_min); ts.insert(TMAX);
        if (!quick) {
            for (int64_t t = 7000; t <= 7100; ++t) if (t >= s.t_min) ts.insert(t);
            for (int64_t p = 256; p <= TMAX; p *= 2) for (int64_t d = -3; d <= 3; ++d) if (p + d >= s.t_min && p + d <= TMAX) ts.insert(p + d);
            for (int64_t t = 6540; t <= 6570; ++t) if (t >= s.t_min) ts.insert(t);   // around the activation slices' first split (K 2560)
        }
        long long bad = 0, bad_outputs = 0;
        int64_t first_bad = 0;
        for (int64_t T : ts) {
            strata::prefill::gemm_set_dense_f32(0);
            gemm.f16(X, W, Ya, T, N, K);
            strata::prefill::gemm_set_dense_f32(1);
            gemm.f16(X, W, Yb, T, N, K);
            cudaMemsetAsync(cnt, 0, 4, st);
            count_diff<<<(unsigned) ((T * N + 255) / 256), 256, 0, st>>>((const unsigned*) Ya, (const unsigned*) Yb, T * N, cnt);
            unsigned nd = 0;
            cudaMemcpyAsync(&nd, cnt, 4, cudaMemcpyDeviceToHost, st);
            if (!ok(cudaStreamSynchronize(st), "sync")) return 1;
            ++cases;
            if (nd) { if (!bad) first_bad = T; ++bad; bad_outputs += nd; }
        }
        std::printf("%-10s %5lld x %-5lld: %zu sizes from T %lld to %lld, %lld with different bits%s\n", s.what, (long long) N, (long long) K, ts.size(),
                    (long long) *ts.begin(), (long long) *ts.rbegin(), bad, bad ? "" : " (bitwise equal)");
        if (bad) std::printf("FAIL %s: first at T %lld, %lld outputs in all\n", s.what, (long long) first_bad, bad_outputs);
        bad_cases += bad;
        if (bench) {
            for (int64_t T : {(int64_t) 560, (int64_t) 2048, (int64_t) 4096, (int64_t) 7040, (int64_t) 7041}) {
                if (T < s.t_min) continue;
                double ms[2] = {0, 0};
                for (int on = 0; on < 2; ++on) {
                    strata::prefill::gemm_set_dense_f32(on);
                    gemm.f16(X, W, Ya, T, N, K);
                    cudaEventRecord(e0, st);
                    for (int i = 0; i < 3; ++i) gemm.f16(X, W, Ya, T, N, K);
                    cudaEventRecord(e1, st);
                    cudaEventSynchronize(e1);
                    float t = 0;
                    cudaEventElapsedTime(&t, e0, e1);
                    ms[on] = t / 3;
                }
                std::printf("BENCH %-10s T %5lld | cuBLAS FP16 %7.2f ms | FP32 copies %7.2f ms (the conversions included) | %.2fx\n", s.what, (long long) T, ms[0],
                            ms[1], ms[0] / ms[1]);
            }
        }
        cudaFree(W); cudaFree(X); cudaFree(Ya); cudaFree(Yb);
    }
    strata::prefill::gemm_set_dense_f32(-1);
    std::printf("gemm_f32_parity: %lld sizes, %s\n", cases, bad_cases ? "FAILED" : "OK (bitwise)");
    return bad_cases ? 1 : 0;
}
