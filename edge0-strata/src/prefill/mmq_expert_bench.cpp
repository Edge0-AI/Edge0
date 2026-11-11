// src/prefill/mmq_expert_bench.cpp - the prompt path's routed-expert products (MMQ) timed at the engine's shapes.
//
//     [STRATA_PREFILL_NVFP4=w4a8|w4a4] mmq_expert_bench [rows per expert ...]
//
// One MMQ group as the prompt path launches it: 16 experts, R rows each, in one activation array in expert order.
// gate/up [1280 x 2560] and down [2560 x 640] in NVFP4, down in Q8_0 (the GPTQ + Q8_0-down pack's 27 layers), and
// UD-IQ4_XS's IQ3_S gate/up and IQ4_NL down for comparison.
// Synthetic weights of the right format (valid scales), random activations of row-varying magnitude.  Per product:
// the median of 25 runs, and the rate in int8-equivalent TOPS (2 x rows x out x in).  Default R: 12 (a 600-token
// chunk: 600 x 10 / 512), 160 (8K), 626 (32K).
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace mmq = strata::prefill::mmq;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

// valid synthetic weights: NVFP4 blocks are {e4m3 scales, e2m1 codes}; Q8_0 {f16 d, int8 q[32]}.  Random bytes with
// the scale fields overwritten by small positive values (an e4m3 / f16 in the model's range).
std::vector<uint8_t> weights(ggml_type t, int64_t rows, int64_t cols, int experts, std::mt19937& rng) {
    const size_t per = mmq::matrix_bytes(t, rows, cols);
    std::vector<uint8_t> w((size_t) experts * per + 4096);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& b : w) b = (uint8_t) byte(rng);
    const size_t bs = ggml_type_size(t), nblk = (size_t) experts * per / bs;
    for (size_t i = 0; i < nblk; ++i) {
        uint8_t* blk = w.data() + i * bs;
        if (t != GGML_TYPE_NVFP4) {                                   // Q8_0, IQ3_S, IQ4_NL: an f16 d leads the block
            const uint16_t d = 0x2000 + (uint16_t) byte(rng);         // f16 ~ 0.0078..0.016
            std::memcpy(blk, &d, 2);
        } else {                                                      // NVFP4: the scales lead the block
            const int nsc = (int) (bs - (size_t) ggml_blck_size(t) / 2);
            for (int s = 0; s < nsc; ++s) blk[s] = (uint8_t) (0x30 + byte(rng) % 16);   // e4m3 ~ 0.5..1
        }
    }
    return w;
}

struct Case {
    const char* name;
    ggml_type t;
    int64_t rows, cols;
};

double run_case(const Case& c, int R, std::mt19937& rng, float* ms_out) {
    constexpr int E = 16;
    const int T = E * R;
    const size_t per = mmq::matrix_bytes(c.t, c.rows, c.cols);
    std::vector<uint8_t> w = weights(c.t, c.rows, c.cols, E, rng);
    std::vector<float> x((size_t) T * c.cols);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> mag(0.1f, 10.f);
    for (int t = 0; t < T; ++t) { const float m = mag(rng); for (int64_t k = 0; k < c.cols; ++k) x[(size_t) t * c.cols + k] = nd(rng) * m; }
    std::vector<int32_t> bounds(E + 1);
    for (int e = 0; e <= E; ++e) bounds[e] = e * R;

    uint8_t* dw; float *dx, *dd, *dys; void* dq; int32_t *dids, *db;
    ck(cudaMalloc(&dw, w.size()), "w"); ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice), "w");
    ck(cudaMalloc(&dx, x.size() * 4), "x"); ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMalloc(&dq, mmq::q8_bytes(T, c.cols)), "xq");
    ck(cudaMalloc(&dys, (size_t) T * 4), "ys");
    ck(cudaMalloc(&dd, (size_t) T * c.rows * 4), "dst");
    ck(cudaMalloc(&dids, (size_t) T * 4), "ids");
    ck(cudaMalloc(&db, (E + 1) * 4), "bounds");
    ck(cudaMemcpy(db, bounds.data(), (E + 1) * 4, cudaMemcpyHostToDevice), "bounds");
    mmq::iota(dids, T, nullptr);
    mmq::quantize(dx, nullptr, dq, c.t, c.cols, c.cols, T, nullptr, dys);
    mmq::Context ctx;
    mmq::Product p;
    p.w = dw; p.type = c.t; p.w_rows = c.rows; p.w_cols = c.cols; p.expert_bytes = per; p.n = E;
    p.xq = dq; p.bounds = db; p.ids = dids; p.total_rows = T; p.max_rows = R; p.dst = dd; p.ld_dst = c.rows;
    p.y_scale = mmq::fp4_activations(c.t) ? dys : nullptr;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    std::vector<float> ts;
    for (int i = 0; i < 28; ++i) {
        cudaEventRecord(e0);
        ctx.run(p, nullptr);
        cudaEventRecord(e1);
        ck(cudaEventSynchronize(e1), "run");
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (i >= 3) ts.push_back(ms);
    }
    std::sort(ts.begin(), ts.end());
    const float ms = ts[ts.size() / 2];
    if (ms_out) *ms_out = ms;
    cudaFree(dw); cudaFree(dx); cudaFree(dq); cudaFree(dys); cudaFree(dd); cudaFree(dids); cudaFree(db);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return 2.0 * T * (double) c.rows * c.cols / (ms * 1e9);
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<int> Rs;
    for (int i = 1; i < argc; ++i) Rs.push_back(std::atoi(argv[i]));
    if (Rs.empty()) Rs = {12, 160, 626};
    const char* mode = std::getenv("STRATA_PREFILL_NVFP4");
    std::printf("mmq_expert_bench: mode %s (FP4 activations: %s), 16 experts a launch, medians of 25\n",
                mode ? mode : "w4a8", mmq::fp4_activations(GGML_TYPE_NVFP4) ? "yes" : "no");
    const Case cases[] = {{"nvfp4 gate/up", GGML_TYPE_NVFP4, 1280, 2560},
                          {"nvfp4 down", GGML_TYPE_NVFP4, 2560, 640},
                          {"q8_0 down", GGML_TYPE_Q8_0, 2560, 640},
                          {"iq3_s gate/up", GGML_TYPE_IQ3_S, 1280, 2560},
                          {"iq4_nl down", GGML_TYPE_IQ4_NL, 2560, 640}};
    std::mt19937 rng(7);
    for (int R : Rs)
        for (const Case& c : cases) {
            float ms = 0;
            const double tops = run_case(c, R, rng, &ms);
            std::printf("  R %4d  %-14s %8.3f ms  %7.1f TOPS\n", R, c.name, ms, tops);
        }
    return 0;
}
