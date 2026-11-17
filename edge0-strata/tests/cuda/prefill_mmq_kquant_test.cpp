// prefill_mmq_kquant_test - the prompt path's MMQ products for the expert formats of Unsloth's UD-Q4_K_XL and
// UD-Q6_K_XL (gate/up Q4_K / Q5_K / Q6_K at 1280 x 2560, down Q5_1 and Q8_0 at 2560 x 640), on synthetic weights
// quantized by ggml's own
// reference quantizers, against a double-precision product of ggml's dequantized weights.  MMQ rounds the activations
// to q8_1 by design, so this is a screen for layout, stride, expert-bound and row-id errors (the bounds of
// tests/hip/prefill_mmq_parity.cpp), not a bit-exactness test.  Several experts per product, permuted rows, an
// all-zero row.  The MMQ path covers gate/up Q4_K / Q5_K / Q6_K and down Q5_1 / Q8_0 (UD-Q4_K_XL / UD-Q6_K_XL).
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace mmq = strata::prefill::mmq;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n) { ck(cudaMalloc(&p, n), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
};

std::vector<uint8_t> make_matrix(ggml_type t, int64_t rows, int64_t cols, int expert, int trial) {
    const auto* tr = ggml_get_type_traits(t);
    if (tr == nullptr || tr->from_float_ref == nullptr) throw std::runtime_error("no reference quantizer");
    const size_t rb = ggml_row_size(t, cols);
    std::vector<uint8_t> out((size_t) rows * rb);
    std::vector<float> row((size_t) cols);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t k = 0; k < cols; ++k) {
            const int code = (int) ((k * 37 + r * 19 + expert * 23 + trial * 11) % 101) - 50;
            row[(size_t) k] = 0.025f * std::sin(0.013f * (float) (k + 1) + 0.17f * (float) r + 0.31f * (float) expert) +
                              0.0015f * (float) code;
        }
        tr->from_float_ref(row.data(), out.data() + (size_t) r * rb, cols);
    }
    return out;
}

std::vector<float> make_x(int rows, int64_t cols, int trial) {
    std::vector<float> x((size_t) rows * (size_t) cols);
    for (int r = 0; r < rows; ++r)
        for (int64_t k = 0; k < cols; ++k)
            x[(size_t) r * (size_t) cols + (size_t) k] =
                0.45f * std::sin(0.009f * (float) (k + 1) + 0.23f * (float) r + 0.07f * (float) trial) +
                0.17f * std::cos(0.021f * (float) (k + 3) - 0.19f * (float) r) +
                0.002f * (float) (((k * 7 + r * 13 + trial * 5) % 19) - 9);
    std::fill(x.end() - cols, x.end(), 0.0f);   // an all-zero row: exactly zero out
    return x;
}

std::vector<int32_t> perm(int n, int shift, bool reverse) {
    std::vector<int32_t> p((size_t) n);
    for (int i = 0; i < n; ++i) p[(size_t) i] = (int32_t) (((reverse ? n - 1 - i : i) + shift) % n);
    return p;
}

// one grouped product: experts side by side at one stride (a zeroed tail after them, as prefill.cpp stages them)
void product(mmq::Context& ctx, cudaStream_t s, const char* name, ggml_type t, int64_t out_rows, int64_t cols,
             const std::vector<int>& counts, int trial) {
    if (!mmq::supported((int) t)) throw std::runtime_error(std::string(name) + ": not covered by this build");
    const int n = (int) counts.size(), rows = std::accumulate(counts.begin(), counts.end(), 0);
    std::vector<int32_t> bounds((size_t) n + 1, 0);
    for (int e = 0; e < n; ++e) bounds[(size_t) e + 1] = bounds[(size_t) e] + counts[(size_t) e];
    const auto src = perm(rows, trial + 1, false), dst = perm(rows, trial + 2, true);
    const std::vector<float> x = make_x(rows, cols, trial);
    const size_t eb = mmq::matrix_bytes((int) t, out_rows, cols);
    if (eb != (size_t) out_rows * ggml_row_size(t, cols)) throw std::runtime_error(std::string(name) + ": matrix_bytes");
    std::vector<std::vector<uint8_t>> ws;
    std::vector<uint8_t> w((size_t) n * eb + 4096, 0);
    for (int e = 0; e < n; ++e) {
        ws.push_back(make_matrix(t, out_rows, cols, e, trial));
        std::copy(ws.back().begin(), ws.back().end(), w.begin() + (ptrdiff_t) ((size_t) e * eb));
    }
    Dev dx(x.size() * 4), dsrc((size_t) rows * 4), ddst((size_t) rows * 4), db(bounds.size() * 4), dw(w.size()),
        dxq(mmq::q8_bytes(rows, cols)), dy((size_t) rows * (size_t) out_rows * 4);
    ck(cudaMemcpy(dx.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(dsrc.p, src.data(), src.size() * 4, cudaMemcpyHostToDevice), "src");
    ck(cudaMemcpy(ddst.p, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice), "dst");
    ck(cudaMemcpy(db.p, bounds.data(), bounds.size() * 4, cudaMemcpyHostToDevice), "bounds");
    ck(cudaMemcpy(dw.p, w.data(), w.size(), cudaMemcpyHostToDevice), "w");
    ck(cudaMemset(dy.p, 0xff, (size_t) rows * (size_t) out_rows * 4), "sentinel");
    // the copies and the sentinel above run on the legacy stream, which a non-blocking stream does not wait for: a
    // pageable cudaMemcpy/cudaMemset can still be in flight when the kernels start (a 128-SM 4090 caught it)
    ck(cudaDeviceSynchronize(), "inputs");
    mmq::quantize((const float*) dx.p, (const int32_t*) dsrc.p, dxq.p, (int) t, cols, cols, rows, s);
    mmq::Product p;
    p.w = dw.p; p.type = (int) t; p.w_rows = out_rows; p.w_cols = cols; p.expert_bytes = eb; p.n = n;
    p.xq = dxq.p; p.bounds = (const int32_t*) db.p; p.ids = (const int32_t*) ddst.p; p.total_rows = rows;
    p.max_rows = *std::max_element(counts.begin(), counts.end()); p.dst = (float*) dy.p; p.ld_dst = out_rows;
    ctx.run(p, s);
    ck(cudaGetLastError(), "launch");
    ck(cudaStreamSynchronize(s), "sync");
    std::vector<float> got((size_t) rows * (size_t) out_rows);
    ck(cudaMemcpy(got.data(), dy.p, got.size() * 4, cudaMemcpyDeviceToHost), "y");

    const auto* tr = ggml_get_type_traits(t);
    std::vector<float> ref(got.size(), 0.0f), wd((size_t) out_rows * (size_t) cols);
    for (int e = 0; e < n; ++e) {
        const size_t rb = ggml_row_size(t, cols);
        for (int64_t o = 0; o < out_rows; ++o)
            tr->to_float(ws[(size_t) e].data() + (size_t) o * rb, wd.data() + (size_t) o * (size_t) cols, cols);
        for (int r = bounds[(size_t) e]; r < bounds[(size_t) e + 1]; ++r) {
            const float* xr = x.data() + (size_t) src[(size_t) r] * (size_t) cols;
            for (int64_t o = 0; o < out_rows; ++o) {
                double acc = 0;
                const float* wr = wd.data() + (size_t) o * (size_t) cols;
                for (int64_t k = 0; k < cols; ++k) acc += (double) wr[k] * xr[k];
                ref[(size_t) dst[(size_t) r] * (size_t) out_rows + (size_t) o] = (float) acc;
            }
        }
    }
    double err2 = 0, ref2 = 0, max_abs = 0, zero_max = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i])) throw std::runtime_error(std::string(name) + ": unwritten or non-finite output");
        const double d = (double) got[i] - ref[i];
        err2 += d * d; ref2 += (double) ref[i] * ref[i]; max_abs = std::max(max_abs, std::fabs(d));
    }
    for (int r = 0; r < rows; ++r)
        if (src[(size_t) r] == rows - 1)   // the zero row
            for (int64_t o = 0; o < out_rows; ++o)
                zero_max = std::max(zero_max, (double) std::fabs(got[(size_t) dst[(size_t) r] * (size_t) out_rows + (size_t) o]));
    const double rms = std::sqrt(ref2 / (double) ref.size()), rel = std::sqrt(err2 / (double) ref.size()) / rms;
    std::printf("%-22s rows %2d experts %d  ref_rms %.4g  rel_l2 %.5f  max/rms %.4f  zero row %.3g\n", name, rows, n, rms,
                rel, max_abs / rms, zero_max);
    if (rel > 0.04 || max_abs / rms > 0.35 || zero_max > std::max(1e-5, 1e-4 * rms))
        throw std::runtime_error(std::string(name) + ": outside the screen");
}
}  // namespace

int main() {
    try {
        if (!mmq::built()) { std::printf("no MMQ in this build\n"); return 1; }
        cudaStream_t s = nullptr;
        ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        {
            mmq::Context ctx;
            const std::vector<std::vector<int>> batches{{1, 3, 3}, {4, 1, 2}, {2, 3}, {17}};
            int trial = 0;
            for (const auto& counts : batches) {
                const std::string tag = "-" + std::to_string(trial);
                product(ctx, s, ("Q4_K gate/up" + tag).c_str(), GGML_TYPE_Q4_K, 1280, 2560, counts, trial);
                product(ctx, s, ("Q5_K gate/up" + tag).c_str(), GGML_TYPE_Q5_K, 1280, 2560, counts, trial + 5);
                product(ctx, s, ("Q6_K gate/up" + tag).c_str(), GGML_TYPE_Q6_K, 1280, 2560, counts, trial + 17);
                product(ctx, s, ("Q5_1 down" + tag).c_str(), GGML_TYPE_Q5_1, 2560, 640, counts, trial + 9);
                product(ctx, s, ("Q8_0 down" + tag).c_str(), GGML_TYPE_Q8_0, 2560, 640, counts, trial + 13);
                ++trial;
            }
        }
        ck(cudaStreamDestroy(s), "destroy");
        std::printf("prefill MMQ K-quant screen passed\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "prefill MMQ K-quant screen failed: %s\n", e.what());
        return 1;
    }
}
