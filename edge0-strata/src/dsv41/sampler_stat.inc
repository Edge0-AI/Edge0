// sampler_stat.inc - a statistics harness for the sampler (sample_tokens / sample_draft / sample_verify): the same text is compiled
// into the Metal engine (--sampler-test) and into a CUDA standalone, so the host sampler (Metal) and the CUDA kernel can be compared
// sample by sample (same Philox counters -> identical picks when the arithmetic agrees) and against the chain's own distribution.
//   sampler_stat ROWS.bin N_VOCAB N_SAMPLES OUTPREFIX     ROWS.bin: 4 float32 rows of N_VOCAB: peaked, flat, penalty case, draft
// Writes OUTPREFIX_A.bin (peaked, top_k 20 top_p 0.95 T 0.8), _B.bin (flat, top_k 64, T 1), _C.bin (penalties on a 64-token history,
// top_k 20 top_p 0.9 T 0.7), _D.bin (speculative: the draft tokens, then the verify's outputs, int32 each).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include "strata/dsv41/sampler.hpp"

namespace sampler_stat {
using strata::dsv41k::SamplerParams;
using strata::dsv41k::SpecList;

inline int run(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: sampler_stat ROWS.bin N_VOCAB N_SAMPLES OUTPREFIX\n");
        return 2;
    }
    const int V = std::atoi(argv[2]), N = std::atoi(argv[3]);
    const std::string pre = argv[4];
    std::vector<float> rows((size_t) 4 * V);
    {
        std::FILE* f = std::fopen(argv[1], "rb");
        if (f == nullptr || std::fread(rows.data(), 4, rows.size(), f) != rows.size()) {
            std::fprintf(stderr, "cannot read %s\n", argv[1]);
            return 2;
        }
        std::fclose(f);
    }
    const int B = 500;   // rows a launch
    float* dl = nullptr;
    float* dq = nullptr;
    int* dout = nullptr;
    int* dd = nullptr;
    int* dhist = nullptr;
    SpecList* dql = nullptr;
    cudaMalloc((void**) &dl, (size_t) B * V * 4);
    cudaMalloc((void**) &dq, (size_t) B * V * 4);
    cudaMalloc((void**) &dout, (size_t) B * 4);
    cudaMalloc((void**) &dd, (size_t) B * 4);
    cudaMalloc((void**) &dql, sizeof(SpecList) * (size_t) B);
    std::vector<int> hist(64);
    for (int i = 0; i < 64; ++i) hist[(size_t) i] = (int) (((long long) i * 37 + 11) % V);
    // the history is per row (history_len ints a row): the same 64 for every row of a launch
    std::vector<int> hist_rows((size_t) B * 64);
    for (int r = 0; r < B; ++r)
        for (int i = 0; i < 64; ++i) hist_rows[(size_t) r * 64 + (size_t) i] = hist[(size_t) i];
    cudaMalloc((void**) &dhist, hist_rows.size() * 4);
    cudaMemcpy(dhist, hist_rows.data(), hist_rows.size() * 4, cudaMemcpyHostToDevice);

    auto fill = [&](float* dev, const float* row) {
        std::vector<float> m((size_t) B * V);
        for (int r = 0; r < B; ++r) std::memcpy(m.data() + (size_t) r * V, row, (size_t) V * 4);
        cudaMemcpy(dev, m.data(), m.size() * 4, cudaMemcpyHostToDevice);
    };
    auto write = [&](const char* tag, const std::vector<int>& v) {
        std::FILE* f = std::fopen((pre + "_" + tag + ".bin").c_str(), "wb");
        std::fwrite(v.data(), 4, v.size(), f);
        std::fclose(f);
    };
    std::vector<int> res(B);
    for (int c = 0; c < 4; ++c) {
        const float* row = rows.data() + (size_t) (c == 3 ? 0 : c) * V;
        SamplerParams p;
        p.seed = 12345;
        const char* tag = "A";
        if (c == 0) { p.top_k = 20; p.top_p = 0.95f; p.temperature = 0.8f; }
        if (c == 1) { p.top_k = 0; p.top_p = 1.0f; p.temperature = 1.0f; tag = "B"; }
        if (c == 2) {
            p.top_k = 20; p.top_p = 0.9f; p.temperature = 0.7f; tag = "C";
            p.penalty_last_n = 64; p.penalty_repeat = 1.3f; p.penalty_freq = 0.1f; p.penalty_present = 0.1f;
        }
        if (c == 3) { p.top_k = 20; p.top_p = 0.95f; p.temperature = 0.8f; tag = "D"; }
        fill(dl, row);
        if (c == 3) fill(dq, rows.data() + (size_t) 3 * V);
        std::vector<int> all, all_d;
        for (int base = 0; base < N; base += B) {
            const int n = std::min(B, N - base);
            p.counter = (uint64_t) base;
            if (c == 3) {
                strata::dsv41k::sample_draft(dq, n, V, p, dql, dd, nullptr);
                strata::dsv41k::sample_verify(dl, n, V, p, dql, dd, dout, nullptr);
                cudaDeviceSynchronize();
                cudaMemcpy(res.data(), dd, (size_t) n * 4, cudaMemcpyDeviceToHost);
                all_d.insert(all_d.end(), res.begin(), res.begin() + n);
            } else {
                strata::dsv41k::sample_tokens(dl, n, V, c == 2 ? dhist : nullptr, c == 2 ? 64 : 0, p, dout, nullptr);
                cudaDeviceSynchronize();
            }
            cudaMemcpy(res.data(), dout, (size_t) n * 4, cudaMemcpyDeviceToHost);
            all.insert(all.end(), res.begin(), res.begin() + n);
        }
        if (c == 3) all.insert(all.begin(), all_d.begin(), all_d.end());   // drafts first, then the verify's outputs
        write(tag, all);
        std::fprintf(stderr, "sampler_stat: case %s done (%zu values)\n", tag, all.size());
    }
    return 0;
}
}  // namespace sampler_stat
