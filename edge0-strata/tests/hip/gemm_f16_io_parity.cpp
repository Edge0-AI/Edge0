// #835: the FP16-in / FP16-out prompt GEMMs (Gemm::set_f16_io, STRATA_HIP_PROMPT_F16=1) against a CPU reference.
// f16() and bf16() (its weights BF16, its activations the FP16 image) write FP16 into Y's own rows and widen them in
// place; the columns past N must stay untouched and a beta = 1 call must keep the ordinary FP32 path.  The tolerance
// is FP16's: one rounding of each output.
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include "strata/prefill/gemm.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CHECK(call) do { const auto e = (call); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(e)); std::exit(2); } } while (0)

struct Buffer {
    void* p = nullptr;
    explicit Buffer(size_t bytes) { CHECK(hipMalloc(&p, bytes)); }
    ~Buffer() { if (p) (void) hipFree(p); }
};

// bf16_w: the weights are BF16 (the bf16() entry); the activations are always the FP16 image on this path.
bool run(strata::prefill::Gemm& gemm, hipStream_t stream, bool bf16_w, int t, int n, int k, int ldy, float beta) {
    std::mt19937 rng(835 + t + n + (bf16_w ? 1 : 0));
    std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
    std::vector<uint16_t> x((size_t) t * k), w((size_t) n * k);
    std::vector<float> xf(x.size()), wf(w.size());
    for (size_t i = 0; i < x.size(); ++i) {
        const __half v = __float2half_rn(dist(rng));
        std::memcpy(&x[i], &v, 2);
        xf[i] = __half2float(v);
    }
    for (size_t i = 0; i < w.size(); ++i) {
        if (bf16_w) {
            // round to nearest even by hand (hip_bfloat16's host constructor gave zeros on ROCm 7.14)
            const float f = dist(rng);
            uint32_t bits;
            std::memcpy(&bits, &f, 4);
            bits += 0x7fffu + ((bits >> 16) & 1u);
            w[i] = (uint16_t) (bits >> 16);
            const uint32_t wide = static_cast<uint32_t>(w[i]) << 16;
            std::memcpy(&wf[i], &wide, 4);
        } else {
            const __half v = __float2half_rn(dist(rng));
            std::memcpy(&w[i], &v, 2);
            wf[i] = __half2float(v);
        }
    }
    constexpr int offset = 5;
    std::vector<float> initial(offset + (size_t) t * ldy + 8, -777.25f);
    for (int r = 0; r < t; ++r)
        for (int c = 0; c < n; ++c) initial[offset + (size_t) r * ldy + c] = (r + c) % 17 * 0.03125f;
    Buffer dx(x.size() * 2), dw(w.size() * 2), dy(initial.size() * 4);
    CHECK(hipMemcpyAsync(dx.p, x.data(), x.size() * 2, hipMemcpyHostToDevice, stream));
    CHECK(hipMemcpyAsync(dw.p, w.data(), w.size() * 2, hipMemcpyHostToDevice, stream));
    CHECK(hipMemcpyAsync(dy.p, initial.data(), initial.size() * 4, hipMemcpyHostToDevice, stream));
    if (bf16_w) gemm.bf16((const uint16_t*) dx.p, (const uint16_t*) dw.p, (float*) dy.p + offset, t, n, k, ldy, beta);
    else gemm.f16((const uint16_t*) dx.p, (const uint16_t*) dw.p, (float*) dy.p + offset, t, n, k, ldy, beta);
    CHECK(hipStreamSynchronize(stream));
    std::vector<float> got(initial.size());
    CHECK(hipMemcpy(got.data(), dy.p, got.size() * 4, hipMemcpyDeviceToHost));
    std::vector<bool> active(got.size());
    double diff2 = 0, ref2 = 0, maximum = 0;
    bool ok = true;
    for (int r = 0; r < t; ++r)
        for (int c = 0; c < n; ++c) {
            const size_t j = offset + (size_t) r * ldy + c;
            active[j] = true;
            double ref = beta * initial[j];
            for (int i = 0; i < k; ++i) ref += (double) xf[(size_t) r * k + i] * wf[(size_t) c * k + i];
            const double d = (double) got[j] - ref;
            ok = ok && std::isfinite(got[j]);
            diff2 += d * d;
            ref2 += ref * ref;
            maximum = std::max(maximum, std::abs(d));
        }
    for (size_t j = 0; j < got.size(); ++j)
        if (!active[j]) ok = ok && got[j] == initial[j];
    const double rel = std::sqrt(diff2 / std::max(ref2, 1e-300));
    // one FP16 rounding of the output (beta = 0), or the ordinary FP32 path (beta = 1)
    ok = ok && rel < (beta == 0.0f ? 2e-3 : 1e-4);
    ok = ok && ref2 > 0;   // an all-zero reference would pass anything
    std::printf("%s %s T=%d N=%d K=%d ldy=%d beta=%.1f rel_l2=%.3g max_abs=%.3g ref_rms=%.3g\n", ok ? "PASS" : "FAIL",
                bf16_w ? "BF16W" : "F16", t, n, k, ldy, beta, rel, maximum, std::sqrt(ref2 / ((double) t * n)));
    return ok;
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    hipStream_t stream;
    CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    bool ok = true;
    {
        strata::prefill::Gemm gemm;
        std::string error;
        if (!gemm.init(stream, 2560 * 640, error)) { std::fprintf(stderr, "%s\n", error.c_str()); return 2; }
        gemm.set_f16_io(true);
        ok = run(gemm, stream, false, 32, 640, 2560, 648, 0) && ok;     // N < ldy: the columns past N stay untouched
        ok = run(gemm, stream, false, 64, 2560, 640, 2560, 0) && ok;
        ok = run(gemm, stream, false, 17, 96, 2560, 96, 0) && ok;       // T and N not multiples of the block
        ok = run(gemm, stream, true, 16, 96, 2560, 100, 0) && ok;       // BF16 weights through the FP16 scratch
        ok = run(gemm, stream, true, 48, 640, 2560, 640, 0) && ok;
        ok = run(gemm, stream, false, 64, 640, 640, 640, 1) && ok;      // beta = 1 keeps the FP32 path
    }
    CHECK(hipStreamDestroy(stream));
    return ok ? 0 : 1;
}
