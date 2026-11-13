// The latent D projection (latent_dense_d, moe_fused.cu and moe_fused_q3k.cu) on a prompt whose last 64-row tile is
// only partly filled.  A row's products depend only on that row, so every row of a launch with M rows must be bit
// for bit the row of a launch over two full tiles, and nothing past row M may be written.  A launch that does not
// finish within the timeout is reported as a hang: lanes leaving the epilogue before its full-warp shuffles
// deadlocked the warp on sm_121 (DGX Spark), with the engine's watchdog firing at layer 0 of the prompt.
//
//     lat_dense_d_tail_test            every M in 1..127 except 64, both kernels, Q8_0 and Q4_0 weights
#include "strata/kernels/iq_kernels.hpp"
#include "strata/prefill/moe_fused.hpp"
#include "strata/prefill/moe_fused_q3k.hpp"

#include <cuda_runtime.h>
#include <ggml.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

namespace fused = strata::prefill::fused;
namespace fused_q3k = strata::prefill::fused_q3k;

#define CK(e) do { const cudaError_t r_ = (e); if (r_ != cudaSuccess) { \
    std::printf("FAIL %s: %s\n", #e, cudaGetErrorString(r_)); std::exit(1); } } while (0)

namespace {

constexpr int K = 2560;          // the layer width (Edge0 145B)
constexpr int G = 8;             // the projection groups
constexpr int64_t ELL = 1792;    // the latent width
constexpr int64_t MREF = 128;    // the reference: two full tiles
constexpr size_t XROW = (size_t) (ELL / 32) * 36;   // one xl row of q8_1 blocks
constexpr uint8_t FILL = 0xA5;

// a launch that has not finished after `seconds` is a hang; the device cannot be recovered, so the test ends
void finish_or_fail(cudaStream_t s, const char* what, int64_t m) {
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        const cudaError_t r = cudaStreamQuery(s);
        if (r == cudaSuccess) return;
        if (r != cudaErrorNotReady) {
            std::printf("FAIL %s M=%lld: %s\n", what, (long long) m, cudaGetErrorString(r));
            std::exit(1);
        }
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(10)) {
            std::printf("FAIL %s M=%lld: the launch did not finish in 10 s (hang)\n", what, (long long) m);
            std::fflush(stdout);
            std::_Exit(1);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

struct Case {
    const char* name;
    ggml_type wtype;
    int fmt;
    bool q3k;
    bool colscale;
};

}  // namespace

int main() {
    if (!fused::latent_fused_available()) {
        std::printf("SKIP: the latent fused kernels need sm_80 or newer\n");
        return 0;
    }
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);

    // the activations: MREF rows, quantized on the device as the prompt path does (quantize_q8_1_rows)
    std::vector<float> x((size_t) MREF * K);
    for (auto& v : x) v = nd(rng);
    float* dx = nullptr;
    uint8_t *dxq = nullptr, *dw = nullptr, *dout = nullptr;
    float* dinv = nullptr;
    const size_t xq_bytes = (size_t) MREF * (K / 32) * 36;
    const size_t out_bytes = (size_t) MREF * G * XROW;
    CK(cudaMalloc(&dx, x.size() * sizeof(float)));
    CK(cudaMalloc(&dxq, xq_bytes));
    CK(cudaMalloc(&dout, out_bytes));
    CK(cudaMalloc(&dinv, (size_t) ELL * sizeof(float)));
    CK(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    cudaStream_t s = nullptr;
    CK(cudaStreamCreate(&s));
    strata::kernels::quantize_q8_1_rows(dx, MREF, K, dxq, s);
    CK(cudaStreamSynchronize(s));

    // a column scale for the q3k kernel's colscaled pack (1/s, one float an ell column)
    std::vector<float> inv((size_t) ELL);
    std::uniform_real_distribution<float> ud(0.5f, 2.0f);
    for (auto& v : inv) v = ud(rng);
    CK(cudaMemcpy(dinv, inv.data(), inv.size() * sizeof(float), cudaMemcpyHostToDevice));

    // the weights: G*ELL rows of K, quantized from real values (never random bytes: a random fp16 scale is not finite)
    std::vector<float> w((size_t) G * ELL * K);
    std::normal_distribution<float> nw(0.f, 0.02f);
    for (auto& v : w) v = nw(rng);

    const Case cases[] = {
        {"fused Q8_0", GGML_TYPE_Q8_0, 8, false, false},
        {"fused Q4_0", GGML_TYPE_Q4_0, 4, false, false},
        {"q3k Q8_0", GGML_TYPE_Q8_0, 8, true, false},
        {"q3k Q8_0 colscale", GGML_TYPE_Q8_0, 8, true, true},
    };
    std::vector<uint8_t> ref(out_bytes), got(out_bytes);
    int failures = 0;
    for (const Case& c : cases) {
        const size_t wbytes = ggml_row_size(c.wtype, K) * (size_t) G * ELL;
        std::vector<uint8_t> wq(wbytes);
        const size_t used = ggml_quantize_chunk(c.wtype, w.data(), wq.data(), 0, G * ELL, K, nullptr);
        if (used != wbytes) {
            std::printf("FAIL %s: quantized %zu bytes, expected %zu\n", c.name, used, wbytes);
            return 1;
        }
        CK(cudaMalloc(&dw, wbytes));
        CK(cudaMemcpy(dw, wq.data(), wbytes, cudaMemcpyHostToDevice));
        auto run = [&](int64_t m) {
            CK(cudaMemsetAsync(dout, FILL, out_bytes, s));
            if (c.q3k)
                fused_q3k::latent_dense_d(dw, dxq, m, K, G, ELL, dout, s, c.fmt, c.colscale ? dinv : nullptr);
            else
                fused::latent_dense_d(dw, dxq, m, K, G, ELL, dout, s, c.fmt);
            finish_or_fail(s, c.name, m);
        };
        run(MREF);
        CK(cudaMemcpy(ref.data(), dout, out_bytes, cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int64_t m = 1; m < MREF; ++m) {
            if (m == 64) continue;   // a full tile
            run(m);
            CK(cudaMemcpy(got.data(), dout, out_bytes, cudaMemcpyDeviceToHost));
            const size_t live = (size_t) m * G * XROW;
            if (std::memcmp(got.data(), ref.data(), live) != 0) {
                size_t i = 0;
                while (got[i] == ref[i]) ++i;
                std::printf("FAIL %s M=%lld: row %zu differs from the full-tile run\n", c.name, (long long) m,
                            i / (G * XROW));
                ++bad;
                continue;
            }
            for (size_t i = live; i < out_bytes; ++i)
                if (got[i] != FILL) {
                    std::printf("FAIL %s M=%lld: byte %zu past the last row was written\n", c.name, (long long) m, i);
                    ++bad;
                    break;
                }
        }
        std::printf("%s %s: M = 1..127, rows identical to the full-tile run, nothing written past row M\n",
                    bad ? "FAIL" : "ok  ", c.name);
        failures += bad;
        CK(cudaFree(dw));
    }
    CK(cudaStreamDestroy(s));
    std::printf(failures ? "FAIL\n" : "PASS\n");
    return failures ? 1 : 0;
}
