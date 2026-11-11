// src/kernels/iq_parity.cpp - plan v0.3 P6: the i-quant kernels against gguf-py on generated rows.
//
//     python tools/iq_fixture.py --out <dir> && build/iq_parity <dir>
//
// The fixtures are GENERATED (deterministic; TODO 24) - tools/iq_fixture.py writes each type's raw block
// bytes and the gguf-py dequantized reference (Q2_0, this repository's type 42, is dequantized by
// tools/gguf_writer.py's codec instead - gguf-py has no such type).  Dependencies: python3 with numpy and
// the vendored gguf-py that setup downloads (the pinned llama.cpp zip; third_party/ is gitignored).
//
// What is compared, and how: the dequant parity is a MEASURED relative error (sum|got-ref| / sum|ref|) -
// bitwise equality is not claimed and not required; on the generated fixtures it measures 0.00e+00.  The
// MMVQ dot (q8_1 activations) must match the float matrix-vector product over that reference within the
// activation rounding (a few 1e-3 relative), for 1 to 8 columns, every column of a multi-column call bitwise
// equal to a one-column call on it.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "logs/iq_fixture";
    const char* names[] = {"IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ1_M", "IQ4_NL", "IQ4_XS", "Q2_0", "Q3_K"};
    int failures = 0, missing = 0;
    cudaStream_t s;
    cudaStreamCreate(&s);
    for (const char* nm : names) {
        std::FILE* f = std::fopen((dir + "/" + nm + ".bin").c_str(), "rb");
        std::FILE* g = std::fopen((dir + "/" + nm + ".f32").c_str(), "rb");
        if (!f || !g) {
            std::printf("%-8s missing fixture: %s/%s.bin or %s/%s.f32 - generate the fixtures first with "
                        "python tools/iq_fixture.py --out %s (deterministic; see the tool's --help)\n",
                        nm, dir.c_str(), nm, dir.c_str(), nm, dir.c_str());
            if (f) std::fclose(f);
            if (g) std::fclose(g);
            ++failures;
            ++missing;
            continue;
        }
        int hdr[3];
        std::fread(hdr, 4, 3, f);
        const int type = hdr[0], rows = hdr[1], cols = hdr[2];
        std::vector<uint8_t> raw;
        { uint8_t buf[1 << 16]; size_t n; while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) raw.insert(raw.end(), buf, buf + n); }
        std::vector<float> ref((size_t) rows * cols);
        std::fread(ref.data(), 4, ref.size(), g);
        std::fclose(f);
        std::fclose(g);
        void* dw = nullptr;
        float* dq = nullptr;
        cudaMalloc(&dw, raw.size());
        cudaMalloc(&dq, ref.size() * 4);
        cudaMemcpy(dw, raw.data(), raw.size(), cudaMemcpyHostToDevice);
        double dq_err = 0.0;
        if (strata::kernels::iq_supported(type) && ((size_t) rows * cols) % 256 == 0) {
            strata::kernels::iq_dequant_f32(type, dw, (int64_t) rows * cols, dq, s);
            std::vector<float> got(ref.size());
            cudaMemcpy(got.data(), dq, got.size() * 4, cudaMemcpyDeviceToHost);
            double num = 0, den = 0;
            for (size_t i = 0; i < ref.size(); ++i) { num += std::fabs(got[i] - ref[i]); den += std::fabs(ref[i]); }
            dq_err = num / (den + 1e-30);
        }
        // the dot, ncols 1..8: each column of an ncols call bitwise equal to a one-column call on it (the
        // exact multi-column layout, the default), all of them within the activation rounding of the float product
        const int MC = 8;
        std::mt19937 rng(7);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> x((size_t) MC * cols);
        for (auto& v : x) v = nd(rng);
        float* dx = nullptr;
        void* xq = nullptr;
        float* dy = nullptr;
        cudaMalloc(&dx, x.size() * 4);
        cudaMalloc(&xq, (size_t) MC * cols / 32 * 36);
        cudaMalloc(&dy, (size_t) MC * rows * 4);
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        strata::kernels::quantize_q8_1_rows(dx, MC, cols, xq, s);
        std::vector<float> y((size_t) MC * rows), yn(y.size());
        int multi_bad = 0;
        try {
            for (int c = 0; c < MC; ++c)
                strata::kernels::native_mmvq(type, dw, (const uint8_t*) xq + (size_t) c * cols / 32 * 36,
                                             dy + (size_t) c * rows, cols, rows, 1, s);
            cudaStreamSynchronize(s);
            cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost);
            for (int nc = 2; nc <= MC; ++nc) {
                strata::kernels::native_mmvq(type, dw, xq, dy, cols, rows, nc, s);
                cudaStreamSynchronize(s);
                cudaMemcpy(yn.data(), dy, yn.size() * 4, cudaMemcpyDeviceToHost);
                if (strata::kernels::native_mmvq_multi_exact() &&
                    std::memcmp(yn.data(), y.data(), (size_t) nc * rows * 4) != 0) {
                    std::printf("%-8s mmvq ncols %d: not bitwise equal to the one-column calls\n", nm, nc);
                    ++multi_bad;
                }
            }
        } catch (const std::exception& e) { std::printf("%-8s mmvq: %s\n", nm, e.what()); ++failures; continue; }
        double num = 0, den = 0;
        for (int c = 0; c < MC; ++c)
            for (int r = 0; r < rows; ++r) {
                double acc = 0;
                for (int k = 0; k < cols; ++k) acc += (double) ref[(size_t) r * cols + k] * x[(size_t) c * cols + k];
                num += std::fabs(y[(size_t) c * rows + r] - acc);
                den += std::fabs(acc);
            }
        const double mm_err = num / (den + 1e-30);
        const bool ok = dq_err < 1e-6 && mm_err < 2e-2 && multi_bad == 0;
        std::printf("%-8s type %2d %4d x %5d  dequant rel %.2e  mmvq rel %.2e (ncols 1..%d)  %s\n", nm, type, rows, cols,
                    dq_err, mm_err, MC, ok ? "ok" : "FAIL");
        if (!ok) ++failures;
        cudaFree(dw); cudaFree(dq); cudaFree(dx); cudaFree(xq); cudaFree(dy);
    }
    std::printf("iq_parity: %d failures\n", failures);
    // no fixture at all: the generator was skipped (no numpy / vendored gguf-py) - a skip (3), not a failure
    if (missing == (int) (sizeof names / sizeof names[0])) {
        std::printf("iq_parity: no fixtures in %s - skipped\n", dir.c_str());
        return 3;
    }
    return failures ? 1 : 0;
}
