// src/kernels/kv_q8_parity.cpp - plan v0.3 P7: INT8 KV append/gather against a host reference (GPU, no model).
//
// Appends random K/V cells at scattered positions through a non-identity page table into both the INT8 pools
// (kv_append_q8_step) and the FP16 pools (kv_append_step), gathers random cell sets from both, and checks:
//   1. INT8 codes and FP16 scales are BITWISE equal to the host reference (same f16_from_f32 rounding);
//   2. the INT8 gather is BITWISE equal to the host dequantization of those codes;
//   3. against the FP16 path, the INT8 values differ by at most half a quantization step of their group plus the
//      fp16 rounding of both sides (0.624 steps).
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
int g_fail = 0;
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* dalloc(size_t n) { T* p = nullptr; ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc"); ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset"); return p; }
}  // namespace

// A decode-shaped microbench: a 32k pool, the 2,051-cell selection the QSA decoder gathers per layer, random
// cells (so the reads miss L2).  Reports the gather's effective traffic per call and GB/s.  Not part of the test.
int bench() {
    k::QsaShapes s = k::qsa_real_shapes();
    const int H = (int) s.n_head_kv, D = (int) s.head_dim, P = (int) s.page_size, G = D / k::KV_Q8_GROUP;
    const int ctx = 32768, pages = ctx / P;
    const long long rows = (long long) pages * H * P;
    const int max_ids = (int) k::qsa_selection_width(ctx, s);
    std::mt19937 rng(7);
    std::vector<int8_t> kq((size_t) rows * D), vq((size_t) rows * D);
    std::vector<uint16_t> ks((size_t) rows * G), vs((size_t) rows * G);
    for (auto& x : kq) x = (int8_t) (rng() & 0xFF);
    for (auto& x : vq) x = (int8_t) (rng() & 0xFF);
    for (auto& x : ks) x = (uint16_t) (0x3000 | (rng() & 0x3FF));
    for (auto& x : vs) x = (uint16_t) (0x3000 | (rng() & 0x3FF));
    std::vector<int32_t> table(pages);
    for (int i = 0; i < pages; ++i) table[i] = i;
    std::shuffle(table.begin(), table.end(), rng);
    std::vector<int32_t> ids(max_ids);
    for (auto& x : ids) x = (int32_t) (rng() % ctx);
    int8_t* d_kq = dalloc<int8_t>(kq.size()); ck(cudaMemcpy(d_kq, kq.data(), kq.size(), cudaMemcpyHostToDevice), "up");
    int8_t* d_vq = dalloc<int8_t>(vq.size()); ck(cudaMemcpy(d_vq, vq.data(), vq.size(), cudaMemcpyHostToDevice), "up");
    uint16_t* d_ks = dalloc<uint16_t>(ks.size()); ck(cudaMemcpy(d_ks, ks.data(), ks.size() * 2, cudaMemcpyHostToDevice), "up");
    uint16_t* d_vs = dalloc<uint16_t>(vs.size()); ck(cudaMemcpy(d_vs, vs.data(), vs.size() * 2, cudaMemcpyHostToDevice), "up");
    int32_t* d_table = dalloc<int32_t>(pages); ck(cudaMemcpy(d_table, table.data(), pages * 4, cudaMemcpyHostToDevice), "up");
    int32_t* d_ids = dalloc<int32_t>(max_ids); ck(cudaMemcpy(d_ids, ids.data(), max_ids * 4, cudaMemcpyHostToDevice), "up");
    int32_t* d_step = dalloc<int32_t>(k::kStepCount);
    int32_t hstep[k::kStepCount] = {0, 0, 0, max_ids};
    ck(cudaMemcpy(d_step, hstep, sizeof hstep, cudaMemcpyHostToDevice), "up");
    uint16_t* k8 = dalloc<uint16_t>((size_t) max_ids * H * D);
    uint16_t* v8 = dalloc<uint16_t>((size_t) max_ids * H * D);
    for (int i = 0; i < 5; ++i) k::kv_gather_q8_step(d_kq, d_vq, d_ks, d_vs, d_table, d_ids, d_step, max_ids, s, k8, v8, nullptr);
    ck(cudaDeviceSynchronize(), "warmup");
    const int reps = 200;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int r = 0; r < reps; ++r) k::kv_gather_q8_step(d_kq, d_vq, d_ks, d_vs, d_table, d_ids, d_step, max_ids, s, k8, v8, nullptr);
    cudaEventRecord(e1); ck(cudaEventSynchronize(e1), "time");
    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
    const double bytes = (double) max_ids * H * D * 2                       // K + V codes
                       + (double) max_ids * H * G * 2 * 2                    // K + V scales
                       + (double) max_ids * H * D * 2 * 2;                   // K + V fp16 scratch write
    const double us = ms * 1000.0 / reps;
    std::printf("kv_gather_q8 bench: %d ids, %.1f us/call, %.1f GB/s (%.2f MB moved)\n", max_ids, us,
                bytes / (us * 1e-6) / 1e9, bytes / 1e6);
    return 0;
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) if (!std::strcmp(argv[i], "--bench")) return bench();
    k::QsaShapes s = k::qsa_real_shapes();
    s.page_size = 64;                                                              // several pages in a small pool
    const int H = (int) s.n_head_kv, D = (int) s.head_dim, P = (int) s.page_size, G = D / k::KV_Q8_GROUP;
    const int pages = 8, cells = pages * P;
    std::mt19937 rng(17);
    std::vector<int32_t> table(pages);
    for (int i = 0; i < pages; ++i) table[i] = (i * 5 + 3) % pages;            // a non-identity permutation
    int32_t* d_table = dalloc<int32_t>(pages);
    ck(cudaMemcpy(d_table, table.data(), pages * 4, cudaMemcpyHostToDevice), "table");
    int8_t *kq = dalloc<int8_t>((size_t) cells * H * D), *vq = dalloc<int8_t>((size_t) cells * H * D);
    uint16_t *ks = dalloc<uint16_t>((size_t) cells * H * G), *vs = dalloc<uint16_t>((size_t) cells * H * G);
    uint16_t *kp = dalloc<uint16_t>((size_t) cells * H * D), *vp = dalloc<uint16_t>((size_t) cells * H * D);
    float *kcur = dalloc<float>(H * D), *vcur = dalloc<float>(H * D);
    int32_t* step = dalloc<int32_t>(k::kStepCount);
    std::vector<std::vector<float>> hk(cells), hv(cells);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<int> positions(cells);
    for (int i = 0; i < cells; ++i) positions[i] = i;
    std::shuffle(positions.begin(), positions.end(), rng);
    const int n_fill = cells - 37;                                                  // leave some cells empty
    for (int n = 0; n < n_fill; ++n) {
        const int pos = positions[n];
        std::vector<float> kv(H * D), vv(H * D);
        const float scale = (n % 7 == 0) ? 1e-3f : (n % 11 == 0 ? 40.f : 1.f);    // small and large groups
        for (auto& x : kv) x = nd(rng) * scale;
        for (auto& x : vv) x = nd(rng) * scale;
        if (n % 13 == 0) std::fill(kv.begin(), kv.begin() + 64, 0.f);              // an all-zero group
        hk[pos] = kv; hv[pos] = vv;
        int32_t hstep[k::kStepCount] = {pos, pos + 1, 0, 0};
        ck(cudaMemcpy(step, hstep, sizeof hstep, cudaMemcpyHostToDevice), "step");
        ck(cudaMemcpy(kcur, kv.data(), kv.size() * 4, cudaMemcpyHostToDevice), "k");
        ck(cudaMemcpy(vcur, vv.data(), vv.size() * 4, cudaMemcpyHostToDevice), "v");
        k::kv_append_q8_step(kq, vq, ks, vs, d_table, step, kcur, vcur, s, nullptr);
        k::kv_append_step(kp, vp, d_table, step, kcur, vcur, s, nullptr);
        ck(cudaDeviceSynchronize(), "append");
    }
    // 1. codes and scales bitwise vs the host reference
    std::vector<int8_t> hkq((size_t) cells * H * D), hvq(hkq.size());
    std::vector<uint16_t> hks((size_t) cells * H * G), hvs(hks.size());
    ck(cudaMemcpy(hkq.data(), kq, hkq.size(), cudaMemcpyDeviceToHost), "d2h");
    ck(cudaMemcpy(hvq.data(), vq, hvq.size(), cudaMemcpyDeviceToHost), "d2h");
    ck(cudaMemcpy(hks.data(), ks, hks.size() * 2, cudaMemcpyDeviceToHost), "d2h");
    ck(cudaMemcpy(hvs.data(), vs, hvs.size() * 2, cudaMemcpyDeviceToHost), "d2h");
    long bad_codes = 0;
    for (int n = 0; n < n_fill; ++n) {
        const int pos = positions[n];
        for (int kvsel = 0; kvsel < 2; ++kvsel)
            for (int h = 0; h < H; ++h)
                for (int g = 0; g < G; ++g) {
                    const float* x = (kvsel ? hv[pos] : hk[pos]).data() + h * D + g * 64;
                    float amax = 0.f;
                    for (int t = 0; t < 64; ++t) amax = std::fmax(amax, std::fabs(x[t]));
                    const uint16_t sb = k::f16_from_f32(amax / 127.0f);
                    const float sf = k::f32_from_f16(sb);
                    const long long row = ((long long) table[pos / P] * H + h) * P + pos % P;
                    if ((kvsel ? hvs : hks)[row * G + g] != sb) ++bad_codes;
                    for (int t = 0; t < 64; ++t) {
                        int q = sf > 0.f ? (int) std::nearbyint(x[t] / sf) : 0;
                        q = std::clamp(q, -127, 127);
                        if ((kvsel ? hvq : hkq)[row * D + g * 64 + t] != (int8_t) q) ++bad_codes;
                    }
                }
    }
    if (bad_codes) { std::fprintf(stderr, "FAIL: %ld codes/scales differ from the host reference\n", bad_codes); ++g_fail; }
    // 2 + 3. gather random selections from both paths
    const int max_ids = 200;
    int32_t* d_ids = dalloc<int32_t>(max_ids);
    uint16_t *k8 = dalloc<uint16_t>((size_t) max_ids * H * D), *v8 = dalloc<uint16_t>((size_t) max_ids * H * D);
    uint16_t *k16 = dalloc<uint16_t>((size_t) max_ids * H * D), *v16 = dalloc<uint16_t>((size_t) max_ids * H * D);
    double worst = 0.0;
    for (int trial = 0; trial < 20; ++trial) {
        const int n_ids = 1 + (int) (rng() % max_ids);
        std::vector<int32_t> ids(n_ids);
        for (auto& id : ids) id = positions[rng() % n_fill];
        ck(cudaMemcpy(d_ids, ids.data(), n_ids * 4, cudaMemcpyHostToDevice), "ids");
        int32_t hstep[k::kStepCount] = {0, 0, 0, n_ids};
        ck(cudaMemcpy(step, hstep, sizeof hstep, cudaMemcpyHostToDevice), "step");
        k::kv_gather_q8_step(kq, vq, ks, vs, d_table, d_ids, step, max_ids, s, k8, v8, nullptr);
        k::kv_gather_step(kp, vp, d_table, d_ids, step, max_ids, s, k16, v16, nullptr);
        ck(cudaDeviceSynchronize(), "gather");
        std::vector<uint16_t> a8((size_t) n_ids * H * D), b8(a8.size()), a16(a8.size()), b16(a8.size());
        ck(cudaMemcpy(a8.data(), k8, a8.size() * 2, cudaMemcpyDeviceToHost), "d2h");
        ck(cudaMemcpy(b8.data(), v8, b8.size() * 2, cudaMemcpyDeviceToHost), "d2h");
        ck(cudaMemcpy(a16.data(), k16, a16.size() * 2, cudaMemcpyDeviceToHost), "d2h");
        ck(cudaMemcpy(b16.data(), v16, b16.size() * 2, cudaMemcpyDeviceToHost), "d2h");
        for (int j = 0; j < n_ids; ++j)
            for (int h = 0; h < H; ++h) {
                const long long row = ((long long) table[ids[j] / P] * H + h) * P + ids[j] % P;
                for (int d = 0; d < D; ++d) {
                    const size_t o = ((size_t) j * H + h) * D + d;
                    for (int kvsel = 0; kvsel < 2; ++kvsel) {
                        const float sf = k::f32_from_f16((kvsel ? hvs : hks)[row * G + d / 64]);
                        const int8_t q = (kvsel ? hvq : hkq)[row * D + d];
                        const uint16_t want = k::f16_from_f32((float) q * sf);
                        const uint16_t got = (kvsel ? b8 : a8)[o];
                        if (got != want) { if (g_fail < 5) std::fprintf(stderr, "FAIL: gather id %d h %d d %d\n", j, h, d); ++g_fail; }
                        const float ref = k::f32_from_f16((kvsel ? b16 : a16)[o]);
                        const double err = std::fabs(k::f32_from_f16(got) - ref) / (sf > 0 ? sf : 1.0);
                        worst = std::max(worst, err);
                    }
                }
            }
    }
    // |x - q*sf| <= sf/2 (the stored scale keeps |x|/sf <= 127.06, so no clamping), plus the fp16 rounding of each
    // side: at most 2^-11 relative of a value up to 127*sf, i.e. 127/2048 = 0.062 steps each -> 0.5 + 0.124
    if (!(worst <= 0.5 + 2.0 * 127.0 / 2048.0 + 1e-3)) { std::fprintf(stderr, "FAIL: INT8 vs FP16 error %.3f steps\n", worst); ++g_fail; }
    std::printf("kv_q8_parity: %s (worst INT8-vs-FP16 error %.3f quantization steps)\n", g_fail ? "FAILED" : "OK", worst);
    return g_fail ? 1 : 0;
}
