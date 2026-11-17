// Opt-in HIP smoke test for the rocBLAS solution table of the prompt path's FP16-out GEMMs (gfx103x); it needs
// STRATA_ROCBLAS_TUNING (SKIP otherwise).  It checks that the table loads for this device and rocBLAS build, and that
// Gemm's FP16 path (set_f16_io, the route the table serves) matches an FP32-out hipBLASEx reference on the table's
// first two shapes, at their token bucket and at a smaller odd T that resolves to the same row.  Like the hipBLASLt
// test it does not check that rocBLAS ran the table's solution: an id the library refuses falls back to the default
// kernel and the test still passes; STRATA_ROCBLAS_VERBOSE=1 prints the "tuned_launches=... fallbacks=..." summary.
#include <cuda_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#include <rocblas/rocblas.h>

#include "strata/prefill/gemm.hpp"
#include "rocblas_tuning.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
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

#define HIPBLAS_CHECK(call) do { \
    const hipblasStatus_t status_ = (call); \
    if (status_ != HIPBLAS_STATUS_SUCCESS) { \
        std::fprintf(stderr, "hipBLAS failure at %s:%d: %s: %d\n", __FILE__, __LINE__, #call, (int) status_); \
        std::exit(2); \
    } \
} while (0)

namespace {

struct DeviceBuffer {
    void* p = nullptr;
    explicit DeviceBuffer(size_t bytes) { if (bytes) HIP_CHECK(hipMalloc(&p, bytes)); }
    ~DeviceBuffer() { if (p) (void) hipFree(p); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

uint16_t f16_bits(float value) {
    const __half half = __float2half_rn(value);
    uint16_t bits = 0;
    std::memcpy(&bits, &half, sizeof(bits));
    return bits;
}

// Y[T, N] (fp32, row stride ldy, at output_offset floats into the buffer) = X[T, K] . W[N, K]^T, Gemm's FP16 route
// against hipBLASEx with FP32 out.  The FP16 route rounds its outputs to FP16 (that is the measured, accepted cost of
// prompt_f16(), docs/AMD_HIP.md): the tolerance is a few FP16 ulps of the largest output.
bool run_case(strata::prefill::Gemm& gemm, hipblasHandle_t blas, hipStream_t stream, int t, int n, int k, int ldy,
              int output_offset, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
    const size_t x_count = (size_t) t * k, w_count = (size_t) n * k;
    std::vector<uint16_t> x(x_count), w(w_count);
    for (auto& value : x) value = f16_bits(dist(rng));
    for (auto& value : w) value = f16_bits(dist(rng));
    const size_t output_count = (size_t) output_offset + (size_t) t * ldy + 8;
    std::vector<float> initial(output_count, -777.25f);

    DeviceBuffer dx(x_count * 2), dw(w_count * 2), dy(output_count * 4), dref(output_count * 4);
    HIP_CHECK(hipMemcpyAsync(dx.p, x.data(), x_count * 2, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(dw.p, w.data(), w_count * 2, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(dy.p, initial.data(), output_count * 4, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(dref.p, initial.data(), output_count * 4, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    const float alpha = 1.0f, beta = 0.0f;
    HIPBLAS_CHECK(hipblasGemmEx(blas, HIPBLAS_OP_T, HIPBLAS_OP_N, n, t, k, &alpha, dw.p, HIP_R_16F, k, dx.p, HIP_R_16F,
                                k, &beta, (float*) dref.p + output_offset, HIP_R_32F, ldy, HIPBLAS_COMPUTE_32F,
                                HIPBLAS_GEMM_DEFAULT));
    gemm.f16((const uint16_t*) dx.p, (const uint16_t*) dw.p, (float*) dy.p + output_offset, t, n, k, ldy, beta);
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<float> got(output_count), ref(output_count);
    HIP_CHECK(hipMemcpy(got.data(), dy.p, output_count * 4, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(ref.data(), dref.p, output_count * 4, hipMemcpyDeviceToHost));

    std::vector<uint8_t> active(output_count, 0);
    double diff2 = 0.0, ref2 = 0.0, max_abs = 0.0, max_ref = 0.0;
    for (int row = 0; row < t; ++row) {
        for (int col = 0; col < n; ++col) {
            const size_t index = (size_t) output_offset + (size_t) row * ldy + col;
            active[index] = 1;
            if (!std::isfinite(got[index]) || !std::isfinite(ref[index])) {
                std::fprintf(stderr, "non-finite output T=%d N=%d K=%d index=%zu\n", t, n, k, index);
                return false;
            }
            const double delta = (double) got[index] - ref[index];
            diff2 += delta * delta;
            ref2 += (double) ref[index] * ref[index];
            max_abs = std::max(max_abs, std::abs(delta));
            max_ref = std::max(max_ref, (double) std::abs(ref[index]));
        }
    }
    for (size_t i = 0; i < output_count; ++i) {
        if (!active[i] && (got[i] != initial[i] || ref[i] != initial[i])) {
            std::fprintf(stderr, "output touched padding/guard T=%d N=%d K=%d ldy=%d index=%zu got=%.9g\n", t, n, k, ldy,
                         i, got[i]);
            return false;
        }
    }
    const double relative_l2 = std::sqrt(diff2 / std::max(ref2, 1e-300));
    const double ulp = std::ldexp(1.0, (int) std::floor(std::log2(std::max(max_ref, 1e-30))) - 10);
    const bool passed = relative_l2 <= 2e-3 && max_abs <= 4.0 * ulp;
    std::printf("case T=%d N=%d K=%d ldy=%d output_offset=%d relative_l2=%.3g max_abs=%.3g (4 ulp = %.3g) %s\n", t, n, k,
                ldy, output_offset, relative_l2, max_abs, 4.0 * ulp, passed ? "PASS" : "FAIL");
    return passed;
}

}  // namespace

int main() {
    const char* tuning_path = std::getenv("STRATA_ROCBLAS_TUNING");
    if (!tuning_path || !*tuning_path) {
        std::fprintf(stderr, "SKIP: set STRATA_ROCBLAS_TUNING to a tuning table (or the tools/hip directory) for this GPU and rocBLAS build\n");
        return 77;
    }
    if (!strata::prefill::prompt_f16()) {
        std::fprintf(stderr, "SKIP: the FP16 prompt GEMMs (prompt_f16) are off on this device; the table serves that route only\n");
        return 77;
    }

    int device = 0;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, device));
    std::string arch(properties.gcnArchName);
    if (const auto suffix = arch.find(':'); suffix != std::string::npos) arch.resize(suffix);
    char version[128] = {0};
    if (rocblas_get_version_string(version, sizeof version) != rocblas_status_success) {
        std::fprintf(stderr, "rocblas_get_version_string failed\n");
        return 2;
    }
    std::filesystem::path file(tuning_path);
    std::error_code ec;
    if (std::filesystem::is_directory(file, ec)) {   // the engine's rule: <arch>-rocblas-<version>.txt, trailing dots dropped
        std::string tag(version);
        while (!tag.empty() && tag.back() == '.') tag.pop_back();
        file /= arch + "-rocblas-" + tag + ".txt";
    }
    strata::prefill::rocblas_tuning::Table table;
    std::string error;
    if (!table.load(file.string(), arch, version, error)) {
        std::fprintf(stderr, "tuning table rejected: %s (%s)\n", error.c_str(), file.string().c_str());
        return 1;
    }
    if (table.rows().empty()) {
        std::fprintf(stderr, "tuning table has no rows\n");
        return 1;
    }

    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    bool ok = true;
    int cases = 0;
    {
        std::string init_error;
        strata::prefill::Gemm gemm;
        if (!gemm.init((void*) stream, 0, init_error)) {
            std::fprintf(stderr, "Gemm init failed: %s\n", init_error.c_str());
            HIP_CHECK(hipStreamDestroy(stream));
            return 2;
        }
        gemm.set_f16_io(true);
        hipblasHandle_t blas = nullptr;
        HIPBLAS_CHECK(hipblasCreate(&blas));
        HIPBLAS_CHECK(hipblasSetStream(blas, stream));

        // the table's first two distinct shapes: at their bucket (an exact row) and at a smaller odd T (the closest
        // row, a non-tile-multiple tail), the second with the output inside a larger buffer
        std::vector<const strata::prefill::rocblas_tuning::Row*> picked;
        for (const auto& row : table.rows()) {
            if (!row.tuned) continue;
            bool seen = false;
            for (const auto* p : picked) seen = seen || (p->n == row.n && p->k == row.k && p->ldy == row.ldy);
            if (!seen) picked.push_back(&row);
            if (picked.size() == 2) break;
        }
        uint32_t seed = 201;
        for (const auto* row : picked) {
            ok &= run_case(gemm, blas, stream, row->t_bucket, row->n, row->k, row->ldy, 0, seed++);
            ok &= run_case(gemm, blas, stream, std::max(1, row->t_bucket - 37), row->n, row->k, row->ldy, 7, seed++);
            cases += 2;
        }
        HIPBLAS_CHECK(hipblasDestroy(blas));
    }
    HIP_CHECK(hipStreamDestroy(stream));
    std::printf("smoke test: %d cases on the first %zu shapes of the table's %zu rows; %s\n", cases,
                std::min<size_t>(2, table.rows().size()), table.rows().size(),
                ok ? "outputs match hipBLASEx within FP16 rounding (solution ids are not verified; see STRATA_ROCBLAS_VERBOSE)"
                   : "FAILED");
    return ok ? 0 : 1;
}
