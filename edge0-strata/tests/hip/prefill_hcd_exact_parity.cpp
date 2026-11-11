// STRATA_HCD_EXACT's kernel against hipBLASLt, bit for bit.  Gemm::bf16_hcd_exact (the hyper-connection read's down
// projection, N 320 x K 10240, BF16 in, FP32 out) reproduces hipBLASLt solutions 1176 / 1177's k order, StaggerU
// included; it launches only when the tuning table makes hipBLASLt take one of them for the shape.  This test runs both
// for several chunk sizes T on the same random inputs (the activations at the token stride the engine uses, K + 64) and
// requires every output to be equal: a hipBLASLt update that renumbers its solutions or changes their order shows up
// here as a mismatch (or as the kernel declining, which is reported).
//
// Needs STRATA_HIPBLASLT_TUNING (the gfx1151 table in tools/hip) on a gfx11 card: exit 77 (SKIP) otherwise, or when no
// T is taken by the kernel at all.
#include <cuda_runtime.h>

#include "strata/kernels/bf16_bits.hpp"
#include "strata/prefill/gemm.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define HIP_CHECK(call) do { \
    const hipError_t status_ = (call); \
    if (status_ != hipSuccess) { \
        std::fprintf(stderr, "HIP failure at %s:%d: %s: %s\n", __FILE__, __LINE__, #call, hipGetErrorString(status_)); \
        std::exit(2); \
    } \
} while (0)

int main() {
    if (std::getenv("STRATA_HIPBLASLT_TUNING") == nullptr) {
        std::printf("SKIP: STRATA_HIPBLASLT_TUNING is not set (the kernel runs only where the table picks solution 1176 / 1177)\n");
        return 77;
    }
    hipDeviceProp_t prop{};
    int dev = 0;
    HIP_CHECK(hipGetDevice(&dev));
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    if (std::strncmp(prop.gcnArchName, "gfx11", 5) != 0) {
        std::printf("SKIP: %s is not a gfx11 card\n", prop.gcnArchName);
        return 77;
    }
    constexpr int64_t N = 320, K = 10240, LDX = K + 64;
    const int64_t Ts[] = {64, 1000, 2047, 2048, 3333, 4096, 6000, 8191, 16384};
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    strata::prefill::Gemm gemm;
    std::string err;
    if (!gemm.init(stream, 1 << 20, err)) {
        std::fprintf(stderr, "Gemm::init: %s\n", err.c_str());
        return 2;
    }
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint16_t> w((size_t) N * K);
    for (auto& v : w) v = strata::kernels::bf16_from_f32(0.05f * dist(rng));
    uint16_t* dw = nullptr;
    HIP_CHECK(hipMalloc((void**) &dw, w.size() * 2));
    HIP_CHECK(hipMemcpy(dw, w.data(), w.size() * 2, hipMemcpyHostToDevice));
    int ran = 0, bad = 0, declined = 0;
    for (const int64_t T : Ts) {
        std::vector<uint16_t> x((size_t) T * LDX, 0);
        for (int64_t t = 0; t < T; ++t)
            for (int64_t k = 0; k < K; ++k) x[(size_t) t * LDX + k] = strata::kernels::bf16_from_f32(dist(rng));
        uint16_t* dx = nullptr;
        float *dy_ref = nullptr, *dy = nullptr;
        HIP_CHECK(hipMalloc((void**) &dx, x.size() * 2));
        HIP_CHECK(hipMalloc((void**) &dy_ref, (size_t) T * N * 4));
        HIP_CHECK(hipMalloc((void**) &dy, (size_t) T * N * 4));
        HIP_CHECK(hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(dy_ref, 0xff, (size_t) T * N * 4));
        HIP_CHECK(hipMemset(dy, 0xee, (size_t) T * N * 4));
        gemm.bf16(dx, dw, dy_ref, T, N, K, 0, 0.0f, LDX);   // hipBLASLt (the table's solution), the same stride
        HIP_CHECK(hipStreamSynchronize(stream));
        if (!gemm.bf16_hcd_exact(dx, LDX, dw, dy, T, N, K)) {
            std::printf("T %5lld: the kernel declines (the table does not pick solution 1176 / 1177 here): hipBLASLt runs it\n", (long long) T);
            ++declined;
        } else {
            HIP_CHECK(hipStreamSynchronize(stream));
            std::vector<float> a((size_t) T * N), b((size_t) T * N);
            HIP_CHECK(hipMemcpy(a.data(), dy_ref, a.size() * 4, hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(b.data(), dy, b.size() * 4, hipMemcpyDeviceToHost));
            int64_t diff = 0;
            for (size_t i = 0; i < a.size(); ++i)
                if (std::memcmp(&a[i], &b[i], 4) != 0) ++diff;
            std::printf("T %5lld: %lld of %lld outputs differ\n", (long long) T, (long long) diff, (long long) a.size());
            if (diff) ++bad;
            ++ran;
        }
        HIP_CHECK(hipFree(dx));
        HIP_CHECK(hipFree(dy_ref));
        HIP_CHECK(hipFree(dy));
    }
    HIP_CHECK(hipFree(dw));
    if (ran == 0) {
        std::printf("SKIP: the table sends no tested T to solution 1176 / 1177\n");
        return 77;
    }
    std::printf("%s: %d chunk sizes compared, %d declined, %d with differences\n", bad ? "FAIL" : "PASS", ran, declined, bad);
    return bad ? 1 : 0;
}
