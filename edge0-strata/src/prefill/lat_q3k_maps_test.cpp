// src/prefill/lat_q3k_maps_test.cpp - M4's unit gate for the two entry maps the engine needs (native_expert_grouped's
// semantics): ent_tok[e] = the activation row entry e reads (the engine's m.tok: several entries share one xl row),
// ent_dst[e] = the output row it writes (m.ids_identity).  hq stays entry-indexed.
//   1. ent_tok: an input whose rows are shared (two entries a row, the engine's shape) gives the same answer as the
//      same data laid out one row an entry with no map;
//   2. ent_dst: a permutation of the output rows;
//   3. with both maps, fused == gu(ent_tok) -> swiglu -> down(ent_dst) byte for byte.
#include "strata/prefill/lat_q3k_fused.hpp"
#include "strata/prefill/lat_q3k_gu.hpp"
#include "strata/prefill/lat_q3k_swiglu.hpp"
#include "strata/prefill/lat_q3k_down.hpp"

#include <cuda_runtime.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(e) do { const cudaError_t r_ = (e); if (r_ != cudaSuccess) { \
    std::printf("FAIL %s: %s\n", #e, cudaGetErrorString(r_)); std::exit(1); } } while (0)

namespace {
constexpr int64_t ELL = 1280, FF = 640;
constexpr int64_t GUB = 1280 * 550, DNB = ELL * FF / 32 * 18;
constexpr int kXq = (int) (ELL / 32) * 36;
constexpr int kHq = (int) strata::prefill::lat_q3k_fused_hbuf_row;
constexpr int G = 64, EPG = 4, E = G * EPG;

long long n_diff(const float* a, const float* b, size_t n) {
    long long d = 0;
    for (size_t i = 0; i < n; ++i) d += (a[i] != b[i]);
    return d;
}
}  // namespace

int main() {
    std::mt19937 rng(61);
    std::normal_distribution<float> nw(0.f, 0.05f), nd(0.f, 1.f);
    constexpr int NBLOB = 4;

    std::vector<uint8_t> blob((size_t) NBLOB * (GUB + DNB));
    for (int i = 0; i < NBLOB; ++i) {
        uint8_t* b = blob.data() + (size_t) i * (GUB + DNB);
        std::vector<float> w((size_t) 1280 * ELL);
        for (auto& v : w) v = nw(rng);
        if ((size_t) ggml_quantize_chunk(GGML_TYPE_Q3_K, w.data(), b, 0, 1280, ELL, nullptr) != (size_t) GUB) {
            std::printf("FAIL gu quantize\n"); return 1;
        }
        std::vector<float> wd((size_t) ELL * FF);
        for (auto& v : wd) v = nw(rng);
        if ((size_t) ggml_quantize_chunk(GGML_TYPE_IQ4_NL, wd.data(), b + GUB, 0, ELL, FF, nullptr) != (size_t) DNB) {
            std::printf("FAIL dn quantize\n"); return 1;
        }
    }
    const size_t EB = (size_t) (GUB + DNB);
    uint8_t* dall = nullptr;
    CK(cudaMalloc(&dall, blob.size()));
    CK(cudaMemcpy(dall, blob.data(), blob.size(), cudaMemcpyHostToDevice));

    // blobs of the groups: group g (whose row range is [4g, 4g+4)) gets expert (4g) % NBLOB
    std::vector<const uint8_t*> bp((size_t) G);
    for (int g = 0; g < G; ++g) bp[(size_t) g] = dall + (size_t) ((EPG * g) % NBLOB) * EB;
    const uint8_t** dblob = nullptr;
    CK(cudaMalloc(&dblob, (size_t) G * sizeof(uint8_t*)));
    CK(cudaMemcpy(dblob, bp.data(), (size_t) G * sizeof(uint8_t*), cudaMemcpyHostToDevice));

    // NTOK token rows, two entries each (the engine: entries of one token share the xl row)
    const int NTOK = E / 2;
    std::vector<uint8_t> xqt((size_t) NTOK * kXq);
    {
        std::uniform_int_distribution<int> u8(-127, 127);
        for (int t = 0; t < NTOK; ++t)
            for (int b = 0; b < (int) (ELL / 32); ++b) {
                uint8_t* blk = xqt.data() + (size_t) t * kXq + 36 * b;
                const uint16_t d16 = ggml_fp32_to_fp16(0.01f + 1e-4f * (t % 71)), s16 = 0;
                std::memcpy(blk, &d16, 2); std::memcpy(blk + 2, &s16, 2);
                for (int i = 0; i < 32; ++i) blk[4 + i] = (uint8_t) u8(rng);
            }
    }
    uint8_t* dxqt = nullptr;
    CK(cudaMalloc(&dxqt, xqt.size()));
    CK(cudaMemcpy(dxqt, xqt.data(), xqt.size(), cudaMemcpyHostToDevice));
    // the same data laid out one row an entry
    std::vector<uint8_t> xqf((size_t) E * kXq);
    for (int e = 0; e < E; ++e) std::memcpy(xqf.data() + (size_t) e * kXq, xqt.data() + (size_t) (e / 2) * kXq, kXq);
    uint8_t* dxqf = nullptr;
    CK(cudaMalloc(&dxqf, xqf.size()));
    CK(cudaMemcpy(dxqf, xqf.data(), xqf.size(), cudaMemcpyHostToDevice));

    int32_t *dgs = nullptr, *dcnt = nullptr, *dtok = nullptr, *ddst = nullptr, *dng = nullptr;
    CK(cudaMalloc(&dgs, (size_t) (G + 1) * 4));
    CK(cudaMalloc(&dcnt, (size_t) G * 4));
    CK(cudaMalloc(&dtok, (size_t) E * 4));
    CK(cudaMalloc(&ddst, (size_t) E * 4));
    CK(cudaMalloc(&dng, 4));
    std::vector<int32_t> gs((size_t) G + 1), cnt((size_t) G, EPG), tok((size_t) E), dst((size_t) E);
    for (int g = 0; g <= G; ++g) gs[(size_t) g] = g * EPG;
    for (int e = 0; e < E; ++e) { tok[(size_t) e] = e / 2; dst[(size_t) e] = e; }
    // a permutation of the output rows
    std::vector<int> pi((size_t) E);
    for (int e = 0; e < E; ++e) pi[(size_t) e] = e;
    std::mt19937 sh(11);
    std::shuffle(pi.begin(), pi.end(), sh);
    std::vector<int32_t> dstp((size_t) E);
    for (int e = 0; e < E; ++e) dstp[(size_t) e] = pi[(size_t) e];
    CK(cudaMemcpy(dgs, gs.data(), gs.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dcnt, cnt.data(), cnt.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dtok, tok.data(), tok.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ddst, dstp.data(), dstp.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dng, &G, 4, cudaMemcpyHostToDevice));

    float* dgate = nullptr; float* dup = nullptr;
    CK(cudaMalloc(&dgate, (size_t) E * FF * 4));
    CK(cudaMalloc(&dup, (size_t) E * FF * 4));
    uint8_t* dhq = nullptr; uint8_t* dhbuf = nullptr;
    CK(cudaMalloc(&dhq, (size_t) E * kHq));
    CK(cudaMalloc(&dhbuf, (size_t) E * kHq));
    float* dy = nullptr; float* dy2 = nullptr;
    CK(cudaMalloc(&dy, (size_t) E * ELL * 4));
    CK(cudaMalloc(&dy2, (size_t) E * ELL * 4));

    // ---- 1: no maps, the one-row-an-entry layout --------------------------------------------------------------
    CK(cudaMemset(dy, 0, (size_t) E * ELL * 4));
    strata::prefill::lat_q3k_fused(dblob, dgs, G, dxqf, dhbuf, dy, ELL, nullptr, dcnt, nullptr, nullptr);
    CK(cudaDeviceSynchronize());
    std::vector<float> y1((size_t) E * ELL);
    CK(cudaMemcpy(y1.data(), dy, y1.size() * 4, cudaMemcpyDeviceToHost));

    // ---- 2: ent_tok, the shared rows ---------------------------------------------------------------------------
    CK(cudaMemset(dy2, 0, (size_t) E * ELL * 4));
    strata::prefill::lat_q3k_fused(dblob, dgs, G, dxqt, dhbuf, dy2, ELL, nullptr, dcnt, dtok, nullptr);
    CK(cudaDeviceSynchronize());
    std::vector<float> y2((size_t) E * ELL);
    CK(cudaMemcpy(y2.data(), dy2, y2.size() * 4, cudaMemcpyDeviceToHost));
    std::printf("ent_tok : %lld of %zu floats differ from the one-row-an-entry layout <- must be 0\n",
                n_diff(y1.data(), y2.data(), y1.size()), y1.size());
    if (n_diff(y1.data(), y2.data(), y1.size()) != 0) { std::printf("FAIL ent_tok\n"); return 1; }

    // ---- 3: ent_dst, a permutation of the output rows ----------------------------------------------------------
    CK(cudaMemset(dy2, 0, (size_t) E * ELL * 4));
    strata::prefill::lat_q3k_fused(dblob, dgs, G, dxqt, dhbuf, dy2, ELL, nullptr, dcnt, dtok, ddst);
    CK(cudaDeviceSynchronize());
    std::vector<float> y3((size_t) E * ELL);
    CK(cudaMemcpy(y3.data(), dy2, y3.size() * 4, cudaMemcpyDeviceToHost));
    long long bad = 0;
    for (int e = 0; e < E; ++e)
        for (int r = 0; r < ELL; ++r)
            bad += (y3[(size_t) pi[(size_t) e] * ELL + r] != y1[(size_t) e * ELL + r]);
    std::printf("ent_dst : %lld of %zu floats differ at the permuted rows               <- must be 0\n", bad, y1.size());
    if (bad != 0) { std::printf("FAIL ent_dst\n"); return 1; }

    // ---- 4: with both maps, fused == gu -> swiglu -> down ------------------------------------------------------
    {
        std::vector<float> yser((size_t) E * ELL);
        CK(cudaMemset(dy, 0, (size_t) E * ELL * 4));
        strata::prefill::lat_q3k_gu(dblob, dgs, G, dxqt, dgate, dup, FF, nullptr, dcnt, dtok);
        strata::prefill::lat_q3k_swiglu_q8(dgate, dup, E, dhq, nullptr, nullptr);
        strata::prefill::lat_q3k_down(dblob, dgs, G, dhq, dy, ELL, nullptr, dcnt, ddst);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(yser.data(), dy, yser.size() * 4, cudaMemcpyDeviceToHost));
        const long long d = n_diff(yser.data(), y3.data(), yser.size());
        std::printf("fused vs gu -> swiglu -> down (both maps): %lld of %zu floats differ  <- must be 0\n", d,
                    yser.size());
        if (d) { std::printf("FAIL map chain\n"); return 1; }
    }

    std::printf("PASS: M4 maps (ent_tok shared rows, ent_dst permutation, fused == serial)\n");
    return 0;
}
