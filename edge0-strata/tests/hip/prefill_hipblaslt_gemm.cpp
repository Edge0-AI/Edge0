// Opt-in HIP smoke test for the calibrated hipBLASLt prefill route; it needs STRATA_HIPBLASLT_TUNING (SKIP otherwise).
// It checks that the table loads for this device and hipBLASLt version, that the two rows it uses exist (bf16
// N=48 K=2560 ldy=96 and f16 N=512 K=2560 ldy=512, T bucket 4096), and that Gemm's output matches hipBLASEx on those
// two rows. It does NOT check the other rows, and it does not check that hipBLASLt ran the table's solution: an id
// the library rejects falls back to hipBLASEx and the test still passes. Set STRATA_HIPBLASLT_VERBOSE=1 to read the
// "hipBLASLt summary launches=... fallbacks=..." line.
#include <cuda_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>

#include "strata/kernels/bf16_bits.hpp"
#include "strata/prefill/gemm.hpp"
#include "hipblaslt_tuning.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

struct DeviceBuffer {
    void* p = nullptr;
    explicit DeviceBuffer(size_t bytes) { if (bytes) HIP_CHECK(hipMalloc(&p, bytes)); }
    ~DeviceBuffer() { if (p) (void) hipFree(p); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

uint16_t encode(float value, bool bf16) {
    if (bf16) return strata::kernels::bf16_from_f32(value);
    const __half half = __float2half_rn(value);
    uint16_t bits = 0;
    std::memcpy(&bits, &half, sizeof(bits));
    return bits;
}

bool run_case(strata::prefill::Gemm& gemm, hipblasHandle_t blas, hipStream_t stream, bool bf16, int t, int n,
              int k, int ldy, int output_offset, float beta, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
    const size_t x_count = (size_t) t * k;
    const size_t w_count = (size_t) n * k;
    std::vector<uint16_t> x(x_count), w(w_count);
    for (auto& value : x) value = encode(dist(rng), bf16);
    for (auto& value : w) value = encode(dist(rng), bf16);

    const size_t output_count = (size_t) output_offset + (size_t) t * ldy + 8;
    std::vector<float> initial(output_count, -777.25f);
    for (int row = 0; row < t; ++row) {
        for (int col = 0; col < n; ++col) {
            initial[(size_t) output_offset + (size_t) row * ldy + col] =
                (float) ((row * 17 + col * 3) % 29 - 14) * 0.03125f;
        }
    }

    DeviceBuffer dx(x_count * sizeof(uint16_t));
    DeviceBuffer dw(w_count * sizeof(uint16_t));
    DeviceBuffer dy(initial.size() * sizeof(float));
    DeviceBuffer dref(initial.size() * sizeof(float));
    HIP_CHECK(hipMemcpyAsync(dx.p, x.data(), x_count * sizeof(uint16_t), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(dw.p, w.data(), w_count * sizeof(uint16_t), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(dy.p, initial.data(), initial.size() * sizeof(float), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(dref.p, initial.data(), initial.size() * sizeof(float), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    const hipDataType type = bf16 ? HIP_R_16BF : HIP_R_16F;
    const float alpha = 1.0f;
    HIPBLAS_CHECK(hipblasGemmEx(blas, HIPBLAS_OP_T, HIPBLAS_OP_N, n, t, k, &alpha, dw.p, type, k, dx.p, type, k,
                                &beta, (float*) dref.p + output_offset, HIP_R_32F, ldy, HIPBLAS_COMPUTE_32F,
                                HIPBLAS_GEMM_DEFAULT));
    if (bf16) {
        gemm.bf16((const uint16_t*) dx.p, (const uint16_t*) dw.p, (float*) dy.p + output_offset, t, n, k, ldy,
                  beta);
    } else {
        gemm.f16((const uint16_t*) dx.p, (const uint16_t*) dw.p, (float*) dy.p + output_offset, t, n, k, ldy,
                 beta);
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<float> got(initial.size()), ref(initial.size());
    HIP_CHECK(hipMemcpy(got.data(), dy.p, got.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(ref.data(), dref.p, ref.size() * sizeof(float), hipMemcpyDeviceToHost));

    std::vector<uint8_t> active(initial.size(), 0);
    double diff2 = 0.0, ref2 = 0.0, max_abs = 0.0;
    for (int row = 0; row < t; ++row) {
        for (int col = 0; col < n; ++col) {
            const size_t index = (size_t) output_offset + (size_t) row * ldy + col;
            active[index] = 1;
            if (!std::isfinite(got[index]) || !std::isfinite(ref[index])) {
                std::fprintf(stderr, "non-finite output dtype=%s T=%d N=%d K=%d beta=%.1f index=%zu\n",
                             bf16 ? "bf16" : "f16", t, n, k, beta, index);
                return false;
            }
            const double delta = (double) got[index] - ref[index];
            diff2 += delta * delta;
            ref2 += (double) ref[index] * ref[index];
            max_abs = std::max(max_abs, std::abs(delta));
        }
    }
    for (size_t i = 0; i < initial.size(); ++i) {
        if (!active[i] && (got[i] != initial[i] || ref[i] != initial[i])) {
            std::fprintf(stderr, "output touched padding/guard dtype=%s T=%d N=%d K=%d beta=%.1f index=%zu got=%.9g\n",
                         bf16 ? "bf16" : "f16", t, n, k, beta, i, got[i]);
            return false;
        }
    }
    const double relative_l2 = std::sqrt(diff2 / std::max(ref2, 1e-300));
    const bool passed = relative_l2 <= 1e-4 && max_abs <= 5e-3;
    std::printf("case dtype=%s T=%d N=%d K=%d ldy=%d output_offset=%d beta=%.1f relative_l2=%.9g max_abs=%.9g %s\n",
                bf16 ? "bf16" : "f16", t, n, k, ldy, output_offset, beta, relative_l2, max_abs,
                passed ? "PASS" : "FAIL");
    return passed;
}

int main() {
    const char* tuning_path = std::getenv("STRATA_HIPBLASLT_TUNING");
    if (!tuning_path || !*tuning_path) {
        std::fprintf(stderr, "SKIP: set STRATA_HIPBLASLT_TUNING to a tuning table for this GPU and hipBLASLt version\n");
        return 77;
    }

    int device = 0;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, device));
    std::string arch(properties.gcnArchName);
    const auto suffix = arch.find(':');
    if (suffix != std::string::npos) arch.resize(suffix);
    hipblasLtHandle_t lt = nullptr;
    HIPBLAS_CHECK(hipblasLtCreate(&lt));
    int version = 0;
    HIPBLAS_CHECK(hipblasLtGetVersion(lt, &version));
    HIPBLAS_CHECK(hipblasLtDestroy(lt));

    strata::prefill::hipblaslt::TuningTable table;
    std::string error;
    if (!table.load(tuning_path, arch, version, error)) {
        std::fprintf(stderr, "tuning table rejected: %s\n", error.c_str());
        return 1;
    }
    if (!table.closest(strata::prefill::hipblaslt::InputType::bf16, 48, 2560, 96, 4096) ||
        !table.closest(strata::prefill::hipblaslt::InputType::f16, 512, 2560, 512, 4096)) {
        std::fprintf(stderr, "tuning table lacks the rows this smoke test uses (bf16 N=48 K=2560 ldy=96, f16 N=512 K=2560 ldy=512)\n");
        return 1;
    }

    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    bool ok = true;
    {
        std::string init_error;
        strata::prefill::Gemm gemm;
        if (!gemm.init((void*) stream, 0, init_error)) {
            std::fprintf(stderr, "Gemm init failed: %s\n", init_error.c_str());
            HIP_CHECK(hipStreamDestroy(stream));
            return 2;
        }
        hipblasHandle_t blas = nullptr;
        HIPBLAS_CHECK(hipblasCreate(&blas));
        HIPBLAS_CHECK(hipblasSetStream(blas, stream));

        // T=4096 is an exact calibrated bucket; T=37 resolves to the same T=4096 row (the closest bucket), validates the
        // actual shape before launch, and tests a non-tile-multiple tail. So the four cases use two table rows.
        ok &= run_case(gemm, blas, stream, true, 4096, 48, 2560, 96, 0, 0.0f, 101);
        ok &= run_case(gemm, blas, stream, true, 37, 48, 2560, 96, 48, 1.0f, 102);
        ok &= run_case(gemm, blas, stream, false, 4096, 512, 2560, 512, 0, 0.0f, 103);
        ok &= run_case(gemm, blas, stream, false, 37, 512, 2560, 512, 7, 1.0f, 104);
        HIPBLAS_CHECK(hipblasDestroy(blas));
    }
    HIP_CHECK(hipStreamDestroy(stream));
    std::printf("smoke test: 4 cases on 2 of the table's %zu rows; %s\n", table.rows().size(),
                ok ? "outputs match hipBLASEx (solution ids are not verified; see STRATA_HIPBLASLT_VERBOSE)"
                   : "FAILED");
    return ok ? 0 : 1;
}
