// gemm_sweep.cu - P40: cublasGemmEx (FP16 operands, FP32 sums) against cublasSgemm on FP32 copies, a sweep over T for
// the dense prompt projections' shapes: time of each and whether the two outputs are bitwise equal.
//   nvcc -O3 -arch=sm_61 gemm_sweep.cu -o gemm_sweep -lcublas && ./gemm_sweep
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

__global__ void fill_f16(__half* p, long n, unsigned seed, float scale) {
    long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned x = (unsigned) i * 2654435761u + seed; x ^= x >> 16; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
    p[i] = __float2half(((float) (x & 0xffff) / 32768.0f - 1.0f) * scale);
}
__global__ void f16_to_f32(const __half* in, float* out, long n) {
    long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}
__global__ void count_diff(const unsigned* a, const unsigned* b, long n, unsigned* out) {
    long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && a[i] != b[i]) atomicAdd(out, 1u);
}
static double ms(cudaEvent_t a, cudaEvent_t b) { float t = 0; cudaEventElapsedTime(&t, a, b); return t; }
#define CK(x) do { auto s_ = (x); if (s_ != 0) { std::fprintf(stderr, "FAIL %s = %d line %d\n", #x, (int) s_, __LINE__); std::exit(1); } } while (0)

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    struct Shape { const char* name; long N, K; };
    const Shape shapes[] = {{"gdn qkv", 10240, 2560}, {"gdn gate z", 6144, 2560}, {"qsa q", 12288, 2560}, {"gdn out", 2560, 6144}, {"qsa out", 2560, 4096},
                            {"qsa k/v", 512, 2560}, {"shexp gate/up", 640, 2560}};
    std::vector<long> Ts;
    for (long t = 1; t <= 16; ++t) Ts.push_back(t);
    for (long t = 24; t <= 512; t += 24) Ts.push_back(t);
    for (long t = 560; t <= 8192; t += 253) Ts.push_back(t);
    for (long t = 7030; t <= 7056; ++t) Ts.push_back(t);
    for (long t : {1024L, 2048L, 4096L, 8192L, 1000L, 3000L, 5000L}) Ts.push_back(t);
    const long TMAX = 8192;
    cublasHandle_t h; CK(cublasCreate(&h));
    cudaStream_t st; CK(cudaStreamCreate(&st)); CK(cublasSetStream(h, st));
    void* ws; CK(cudaMalloc(&ws, 32u << 20)); CK(cublasSetWorkspace(h, ws, 32u << 20));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    unsigned* cnt; CK(cudaMalloc(&cnt, 4));
    for (const Shape& s : shapes) {
        const long N = s.N, K = s.K;
        __half *W16, *X16; float *Wf, *Xf, *Ya, *Yb;
        CK(cudaMalloc(&W16, N * K * 2)); CK(cudaMalloc(&X16, TMAX * K * 2)); CK(cudaMalloc(&Wf, N * K * 4)); CK(cudaMalloc(&Xf, TMAX * K * 4));
        CK(cudaMalloc(&Ya, TMAX * N * 4)); CK(cudaMalloc(&Yb, TMAX * N * 4));
        fill_f16<<<(unsigned) ((N * K + 255) / 256), 256, 0, st>>>(W16, N * K, 1u, 0.05f);
        fill_f16<<<(unsigned) ((TMAX * K + 255) / 256), 256, 0, st>>>(X16, TMAX * K, 7u, 2.0f);
        f16_to_f32<<<(unsigned) ((N * K + 255) / 256), 256, 0, st>>>(W16, Wf, N * K);
        f16_to_f32<<<(unsigned) ((TMAX * K + 255) / 256), 256, 0, st>>>(X16, Xf, TMAX * K);
        const float one = 1.0f, zero = 0.0f;
        int n_equal = 0, n_all = 0, n_slow = 0;
        double sum_a = 0, sum_b = 0;
        for (long T : Ts) {
            double ta = 1e30, tb = 1e30;
            for (int r = 0; r < 3; ++r) {
                cudaEventRecord(e0, st);
                CK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &one, W16, CUDA_R_16F, (int) K, X16, CUDA_R_16F, (int) K,
                                &zero, Ya, CUDA_R_32F, (int) N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                cudaEventRecord(e1, st); CK(cudaEventSynchronize(e1));
                if (r) ta = fmin(ta, ms(e0, e1));
                cudaEventRecord(e0, st);
                CK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &one, Wf, (int) K, Xf, (int) K, &zero, Yb, (int) N));
                cudaEventRecord(e1, st); CK(cudaEventSynchronize(e1));
                if (r) tb = fmin(tb, ms(e0, e1));
            }
            CK(cudaMemsetAsync(cnt, 0, 4, st));
            count_diff<<<(unsigned) ((T * N + 255) / 256), 256, 0, st>>>((const unsigned*) Ya, (const unsigned*) Yb, T * N, cnt);
            unsigned nd = 0; CK(cudaMemcpyAsync(&nd, cnt, 4, cudaMemcpyDeviceToHost, st)); CK(cudaStreamSynchronize(st));
            const double fl = 2.0 * N * K * T / 1e12, fa = fl / (ta / 1e3), fb = fl / (tb / 1e3);
            ++n_all; n_equal += nd == 0; n_slow += fa < 6.0 && T >= 256; sum_a += ta; sum_b += tb;
            std::printf("SWEEP %-14s N %5ld K %5ld T %5ld | f16 %8.3f ms %5.2f TFLOPS | f32 %8.3f ms %5.2f TFLOPS | %5.2fx | %s\n", s.name, N, K, T, ta, fa, tb,
                        fb, ta / tb, nd == 0 ? "bitwise equal" : "DIFFERENT");
            if (nd) std::printf("       %u of %ld outputs differ\n", nd, T * N);
        }
        std::printf("SUMMARY %-14s: %d of %d T values bitwise equal; f16 below 6 TFLOPS at %d of the T >= 256; total f16 %.0f ms, f32 %.0f ms\n", s.name, n_equal,
                    n_all, n_slow, sum_a, sum_b);
        cudaFree(W16); cudaFree(X16); cudaFree(Wf); cudaFree(Xf); cudaFree(Ya); cudaFree(Yb);
    }
    return 0;
}
