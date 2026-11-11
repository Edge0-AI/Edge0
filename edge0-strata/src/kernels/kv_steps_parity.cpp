// src/kernels/kv_steps_parity.cpp - the batched K/V append (#783 PR-d): kv_append_q8_steps / kv_append_q4_steps for
// n = 1..8 cells in one launch against n calls of the single-cell append, memcmp on every pool byte.
//
// Cells land at scattered positions through a non-identity page table; the K/V rows are random with small, large and
// all-zero groups. Three layouts: INT8 K and V, Q4_0 K and V, and the K8V4 hybrid's folded call (the K pool passed as
// both halves, one plane). GPU, synthetic, no model.
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
int g_fail = 0;
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}
template <typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset");
    return p;
}
template <typename T>
std::vector<T> fetch(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "d2h");
    return h;
}
template <typename T>
void zero(T* d, size_t n) { ck(cudaMemset(d, 0, n * sizeof(T) + 64), "zero"); }
bool same(const void* a, const void* b, size_t bytes) { return std::memcmp(a, b, bytes) == 0; }
}  // namespace

int main() {
    k::QsaShapes s = k::qsa_real_shapes();
    s.page_size = 64;
    const int H = (int) s.n_head_kv, D = (int) s.head_dim, P = (int) s.page_size, G = D / k::KV_Q8_GROUP;
    const int pages = 8, cells = pages * P;
    std::mt19937 rng(783);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<int32_t> table(pages);
    for (int i = 0; i < pages; ++i) table[i] = (i * 5 + 3) % pages;
    int32_t* d_table = dalloc<int32_t>(pages);
    ck(cudaMemcpy(d_table, table.data(), pages * 4, cudaMemcpyHostToDevice), "table");

    const size_t q8n = (size_t) cells * H * D, q8s = (size_t) cells * H * G, q4n = (size_t) cells * H * D;  // q4: <= 144 B per cell-head
    int8_t *kq1 = dalloc<int8_t>(q8n), *vq1 = dalloc<int8_t>(q8n), *kq2 = dalloc<int8_t>(q8n), *vq2 = dalloc<int8_t>(q8n);
    uint16_t *ks1 = dalloc<uint16_t>(q8s), *vs1 = dalloc<uint16_t>(q8s), *ks2 = dalloc<uint16_t>(q8s), *vs2 = dalloc<uint16_t>(q8s);
    uint8_t *k41 = dalloc<uint8_t>(q4n), *v41 = dalloc<uint8_t>(q4n), *k42 = dalloc<uint8_t>(q4n), *v42 = dalloc<uint8_t>(q4n);
    uint8_t *h41 = dalloc<uint8_t>(q4n), *h42 = dalloc<uint8_t>(q4n);
    int8_t *hq1 = dalloc<int8_t>(q8n), *hq2 = dalloc<int8_t>(q8n);
    uint16_t *hs1 = dalloc<uint16_t>(q8s), *hs2 = dalloc<uint16_t>(q8s);

    int total_cases = 0;
    for (int n = 1; n <= 8; ++n) {
        for (int round = 0; round < 3; ++round) {
            zero(kq1, q8n); zero(vq1, q8n); zero(kq2, q8n); zero(vq2, q8n);
            zero(ks1, q8s); zero(vs1, q8s); zero(ks2, q8s); zero(vs2, q8s);
            zero(k41, q4n); zero(v41, q4n); zero(k42, q4n); zero(v42, q4n); zero(h41, q4n); zero(h42, q4n);
            zero(hq1, q8n); zero(hq2, q8n); zero(hs1, q8s); zero(hs2, q8s);
            std::vector<int> positions(cells);
            for (int i = 0; i < cells; ++i) positions[i] = i;
            std::shuffle(positions.begin(), positions.end(), rng);
            std::vector<float> hk((size_t) n * H * D), hv(hk.size());
            for (int t = 0; t < n; ++t) {
                const float scale = (t % 3 == 0) ? 1e-3f : (t % 5 == 0 ? 40.f : 1.f);
                for (int i = 0; i < H * D; ++i) {
                    hk[(size_t) t * H * D + i] = nd(rng) * scale;
                    hv[(size_t) t * H * D + i] = nd(rng) * scale;
                }
                if (t % 4 == 1) std::fill(hk.begin() + (size_t) t * H * D, hk.begin() + (size_t) t * H * D + 64, 0.f);
            }
            std::vector<int32_t> hstep((size_t) n * k::kStepCount, 0);
            for (int t = 0; t < n; ++t) {
                hstep[(size_t) t * k::kStepCount + k::kStepPos] = positions[t];
                hstep[(size_t) t * k::kStepCount + 1] = positions[t] + 1;
            }
            float *d_k = dalloc<float>(hk.size()), *d_v = dalloc<float>(hv.size());
            int32_t* d_step = dalloc<int32_t>(hstep.size());
            ck(cudaMemcpy(d_k, hk.data(), hk.size() * 4, cudaMemcpyHostToDevice), "k");
            ck(cudaMemcpy(d_v, hv.data(), hv.size() * 4, cudaMemcpyHostToDevice), "v");
            ck(cudaMemcpy(d_step, hstep.data(), hstep.size() * 4, cudaMemcpyHostToDevice), "step");

            for (int t = 0; t < n; ++t) {
                const int32_t* st = d_step + (size_t) t * k::kStepCount;
                const float* kc = d_k + (size_t) t * H * D;
                const float* vc = d_v + (size_t) t * H * D;
                k::kv_append_q8_step(kq1, vq1, ks1, vs1, d_table, st, kc, vc, s, nullptr);
                k::kv_append_q4_step(k41, v41, d_table, st, kc, vc, s, nullptr);
                k::kv_append_q8_step(hq1, hq1, hs1, hs1, d_table, st, kc, kc, s, nullptr);      // the hybrid's folded K call
                k::kv_append_q4_step(h41, h41, d_table, st, vc, vc, s, nullptr);                // ... and its folded V call
            }
            k::kv_append_q8_steps(kq2, vq2, ks2, vs2, d_table, d_step, k::kStepCount, d_k, d_v, H * D, n, s, nullptr, nullptr);
            k::kv_append_q4_steps(k42, v42, d_table, d_step, k::kStepCount, n, d_k, d_v, s, nullptr);
            k::kv_append_q8_steps(hq2, hq2, hs2, hs2, d_table, d_step, k::kStepCount, d_k, d_k, H * D, n, s, nullptr, nullptr);
            k::kv_append_q4_steps(h42, h42, d_table, d_step, k::kStepCount, n, d_v, d_v, s, nullptr);
            ck(cudaDeviceSynchronize(), "sync");

            struct Chk { const char* name; bool ok; };
            const Chk chks[] = {
                {"INT8 K/V codes", same(fetch(kq1, q8n).data(), fetch(kq2, q8n).data(), q8n) && same(fetch(vq1, q8n).data(), fetch(vq2, q8n).data(), q8n)},
                {"INT8 K/V scales", same(fetch(ks1, q8s).data(), fetch(ks2, q8s).data(), q8s * 2) && same(fetch(vs1, q8s).data(), fetch(vs2, q8s).data(), q8s * 2)},
                {"Q4_0 K/V", same(fetch(k41, q4n).data(), fetch(k42, q4n).data(), q4n) && same(fetch(v41, q4n).data(), fetch(v42, q4n).data(), q4n)},
                {"hybrid K (folded INT8)", same(fetch(hq1, q8n).data(), fetch(hq2, q8n).data(), q8n) && same(fetch(hs1, q8s).data(), fetch(hs2, q8s).data(), q8s * 2)},
                {"hybrid V (folded Q4_0)", same(fetch(h41, q4n).data(), fetch(h42, q4n).data(), q4n)},
            };
            for (const Chk& c : chks) {
                ++total_cases;
                if (!c.ok) {
                    std::printf("    *** n=%d round %d: %s differ from the per-cell appends ***\n", n, round, c.name);
                    ++g_fail;
                }
            }
            cudaFree(d_k); cudaFree(d_v); cudaFree(d_step);
        }
    }
    std::printf("kv_steps: %d checks, %d failures\n", total_cases, g_fail);
    if (g_fail) return 1;
    std::printf("kv_steps_parity OK\n");
    return 0;
}
