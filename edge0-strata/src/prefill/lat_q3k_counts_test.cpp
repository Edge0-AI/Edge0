// src/prefill/lat_q3k_counts_test.cpp - M3b: explicit per-group counts for the three kernels.
// The engine's flush writes each group's rows as one run, but consecutive groups' runs are NOT adjacent (off[]
// follows the ring slots under stream_all), so a kernel that derives a group's end from starts[g + 1] reads the wrong
// rows.  counts[g] fixes the end; starts[] then need only give each group's beginning.
// Gates:
//   1. counts == nullptr is bit for bit the old behaviour;
//   2. with counts, a deliberately scrambled / non-monotonic starts[] gives the same bytes as the ordered layout -
//      and the same scrambled input under counts == nullptr does NOT (that is the defect being fixed);
//   3. fused == gu -> swiglu -> down (memcmp) in both arrangements;
//   4. the four shapes, fused against native's whole step (epg = 200 must stay <= native).
#include "strata/prefill/lat_q3k_fused.hpp"
#include "strata/prefill/lat_q3k_gu.hpp"
#include "strata/prefill/lat_q3k_swiglu.hpp"
#include "strata/prefill/lat_q3k_down.hpp"
#include "strata/kernels/iq_kernels.hpp"

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
constexpr int kXq = (int) (ELL / 32) * 36;   // 1440 B an entry
constexpr int kHq = (int) strata::prefill::lat_q3k_fused_hbuf_row;   // 800 B an entry
constexpr int kMaxE = 4500, kMaxG = 1024, G = 128, EPG = 4, E = G * EPG;

long long n_diff(const float* a, const float* b, size_t n) {
    long long d = 0;
    for (size_t i = 0; i < n; ++i) d += (a[i] != b[i]);
    return d;
}
long long n_diff8(const uint8_t* a, const uint8_t* b, size_t n) {
    long long d = 0;
    for (size_t i = 0; i < n; ++i) d += (a[i] != b[i]);
    return d;
}
}  // namespace

int main() {
    std::mt19937 rng(53);
    std::normal_distribution<float> nw(0.f, 0.05f), nd(0.f, 1.f);

    // NB different experts (not replicas): a group that reads the wrong row range then writes the WRONG weights, which
    // is the whole point of the counts - with one replica byte-identical every wrong range still lands right
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
    const auto L = strata::kernels::native_expert_layout(GGML_TYPE_Q3_K, GGML_TYPE_IQ4_NL, ELL, FF);
    if (L.bytes != EB || L.down_off != 704000 || L.tail_off != 0) {
        std::printf("FAIL layout\n"); return 1;
    }
    uint8_t* dall = nullptr;
    CK(cudaMalloc(&dall, blob.size()));
    CK(cudaMemcpy(dall, blob.data(), blob.size(), cudaMemcpyHostToDevice));
    // expert for the canonical row range r: r % NBLOB
    std::vector<const uint8_t*> bp((size_t) kMaxG);
    for (int g = 0; g < kMaxG; ++g) bp[(size_t) g] = dall + (size_t) (g % NBLOB) * EB;
    const uint8_t** dblob = nullptr;
    CK(cudaMalloc(&dblob, (size_t) kMaxG * sizeof(uint8_t*)));
    CK(cudaMemcpy(dblob, bp.data(), (size_t) kMaxG * sizeof(uint8_t*), cudaMemcpyHostToDevice));

    // a different x row per entry, so a group reading the wrong rows shows up
    std::vector<float> xs((size_t) E * ELL);
    for (auto& v : xs) v = nd(rng);
    std::vector<uint8_t> xq((size_t) kMaxE * kXq);
    {
        std::uniform_int_distribution<int> u8(-127, 127);
        for (int e = 0; e < kMaxE; ++e)
            for (int b = 0; b < (int) (ELL / 32); ++b) {
                uint8_t* blk = xq.data() + (size_t) e * kXq + 36 * b;
                float amax = 0.f, sum = 0.f;
                if (e < E) {
                    for (int i = 0; i < 32; ++i) {
                        const float v = xs[(size_t) e * ELL + 32 * b + i];
                        amax = std::fmax(amax, std::fabs(v)); sum += v;
                    }
                }
                const float d = amax / 127.0f;
                const uint16_t d16 = ggml_fp32_to_fp16(amax != 0.f ? d : 0.01f), s16 = ggml_fp32_to_fp16(sum);
                std::memcpy(blk, &d16, 2); std::memcpy(blk + 2, &s16, 2);
                for (int i = 0; i < 32; ++i)
                    blk[4 + i] = amax != 0.f
                                     ? (uint8_t) (int8_t) std::lround(xs[(size_t) e * ELL + 32 * b + i] / d)
                                     : (uint8_t) u8(rng);
            }
    }
    uint8_t* dxq = nullptr;
    CK(cudaMalloc(&dxq, xq.size()));
    CK(cudaMemcpy(dxq, xq.data(), xq.size(), cudaMemcpyHostToDevice));

    int32_t *dgs = nullptr, *dcnt = nullptr, *ded = nullptr, *det = nullptr, *dng = nullptr;
    CK(cudaMalloc(&dgs, (size_t) (kMaxG + 1) * 4));
    CK(cudaMalloc(&dcnt, (size_t) kMaxG * 4));
    CK(cudaMalloc(&ded, (size_t) kMaxE * 4));
    CK(cudaMalloc(&det, (size_t) kMaxE * 4));
    CK(cudaMalloc(&dng, 4));

    float* dgate = nullptr; float* dup = nullptr;
    CK(cudaMalloc(&dgate, (size_t) kMaxE * FF * 4));
    CK(cudaMalloc(&dup, (size_t) kMaxE * FF * 4));
    uint8_t* dhq = nullptr; uint8_t* dhbuf = nullptr;
    CK(cudaMalloc(&dhq, (size_t) kMaxE * kHq));
    CK(cudaMalloc(&dhbuf, (size_t) kMaxE * kHq));
    float* dy = nullptr; float* dy2 = nullptr;
    CK(cudaMalloc(&dy, (size_t) kMaxE * ELL * 4));
    CK(cudaMalloc(&dy2, (size_t) kMaxE * ELL * 4));

    std::vector<int32_t> id((size_t) E), ngv(1, G);
    for (int e = 0; e < E; ++e) id[(size_t) e] = e;
    CK(cudaMemcpy(ded, id.data(), id.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(det, id.data(), id.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dng, ngv.data(), 4, cudaMemcpyHostToDevice));

    auto run_fused = [&](const std::vector<int32_t>& starts, const std::vector<int32_t>* counts, uint8_t* hq_out,
                         float* y_out) {
        CK(cudaMemcpy(dgs, starts.data(), starts.size() * 4, cudaMemcpyHostToDevice));
        if (counts != nullptr) CK(cudaMemcpy(dcnt, counts->data(), counts->size() * 4, cudaMemcpyHostToDevice));
        strata::prefill::lat_q3k_fused(dblob, dgs, G, dxq, hq_out, y_out, ELL, nullptr,
                                       counts != nullptr ? dcnt : nullptr);
        CK(cudaDeviceSynchronize());
    };

    // ---- A: the ordered layout, counts == nullptr (the frozen behaviour) ---------------------------------------
    std::vector<int32_t> stA((size_t) G + 1), cntA((size_t) G);
    for (int g = 0; g <= G; ++g) stA[(size_t) g] = g * EPG;
    for (int g = 0; g < G; ++g) cntA[(size_t) g] = EPG;
    run_fused(stA, nullptr, dhbuf, dy);
    std::vector<float> yA((size_t) E * ELL);
    std::vector<uint8_t> hA((size_t) E * kHq);
    CK(cudaMemcpy(yA.data(), dy, yA.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hA.data(), dhbuf, hA.size(), cudaMemcpyDeviceToHost));

    // ---- B: same layout, counts given -> must be bit for bit A -------------------------------------------------
    run_fused(stA, &cntA, dhbuf, dy2);
    std::vector<float> yB((size_t) E * ELL);
    std::vector<uint8_t> hB((size_t) E * kHq);
    CK(cudaMemcpy(yB.data(), dy2, yB.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hB.data(), dhbuf, hB.size(), cudaMemcpyDeviceToHost));
    std::printf("gate 1  ordered, counts given   : hbuf %lld B differ, out %lld floats differ (of %zu)\n",
                n_diff8(hA.data(), hB.data(), hA.size()), n_diff(yA.data(), yB.data(), yA.size()), yA.size());
    if (n_diff8(hA.data(), hB.data(), hA.size()) || n_diff(yA.data(), yB.data(), yA.size())) {
        std::printf("FAIL gate 1\n"); return 1;
    }

    // ---- C: scrambled, non-monotonic starts[]: the same (blob, rows) pairs in another group order ---------------
    // group g gets entry rows [4 pi(g), 4 pi(g) + 4) and the blob of pi(g); starts[] is scrambled on purpose
    std::vector<int> pi((size_t) G);
    for (int g = 0; g < G; ++g) pi[(size_t) g] = g;
    std::mt19937 sh(7);
    std::shuffle(pi.begin(), pi.end(), sh);
    std::vector<int32_t> stC((size_t) G + 1), cntC((size_t) G, EPG);
    std::vector<const uint8_t*> bpC((size_t) kMaxG);
    for (int g = 0; g < kMaxG; ++g) bpC[(size_t) g] = bp[(size_t) g];
    for (int g = 0; g < G; ++g) {
        stC[(size_t) g] = EPG * pi[(size_t) g];                  // scrambled, not increasing
        bpC[(size_t) g] = dall + (size_t) (pi[(size_t) g] % NBLOB) * EB;   // that range's own expert
    }
    stC[(size_t) G] = EPG * G;                                   // a value no group reads
    const uint8_t** dblobC = nullptr;
    CK(cudaMalloc(&dblobC, (size_t) kMaxG * sizeof(uint8_t*)));
    CK(cudaMemcpy(dblobC, bpC.data(), (size_t) kMaxG * sizeof(uint8_t*), cudaMemcpyHostToDevice));

    CK(cudaMemcpy(dgs, stC.data(), stC.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dcnt, cntC.data(), cntC.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(dhbuf, 0, (size_t) E * kHq));
    CK(cudaMemset(dy, 0, (size_t) E * ELL * 4));
    strata::prefill::lat_q3k_fused(dblobC, dgs, G, dxq, dhbuf, dy, ELL, nullptr, dcnt);
    CK(cudaDeviceSynchronize());
    std::vector<float> yC((size_t) E * ELL);
    std::vector<uint8_t> hC((size_t) E * kHq);
    CK(cudaMemcpy(yC.data(), dy, yC.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hC.data(), dhbuf, hC.size(), cudaMemcpyDeviceToHost));
    const long long hd = n_diff8(hA.data(), hC.data(), hA.size()), yd = n_diff(yA.data(), yC.data(), yA.size());
    std::printf("gate 2  scrambled starts + counts: hbuf %lld B differ, out %lld floats differ  <- must be 0\n", hd, yd);
    if (hd || yd) { std::printf("FAIL gate 2 (counts)\n"); return 1; }

    // the same scrambled input WITHOUT counts: the defect this fixes (group g's end taken from starts[g + 1])
    CK(cudaMemset(dhbuf, 0, (size_t) E * kHq));
    CK(cudaMemset(dy2, 0, (size_t) E * ELL * 4));
    strata::prefill::lat_q3k_fused(dblobC, dgs, G, dxq, dhbuf, dy2, ELL, nullptr, nullptr);
    CK(cudaDeviceSynchronize());
    std::vector<float> yD((size_t) E * ELL);
    CK(cudaMemcpy(yD.data(), dy2, yD.size() * 4, cudaMemcpyDeviceToHost));
    std::printf("        same input, counts == nullptr: %lld of %zu floats differ from the ordered output (the defect)\n",
                n_diff(yA.data(), yD.data(), yA.size()), yA.size());

    // ---- gate 3: fused == gu -> swiglu -> down, in both arrangements --------------------------------------------
    {
        run_fused(stA, &cntA, dhbuf, dy2);
        std::vector<float> yser((size_t) E * ELL);
        strata::prefill::lat_q3k_gu(dblob, dgs, G, dxq, dgate, dup, FF, nullptr, dcnt);
        strata::prefill::lat_q3k_swiglu_q8(dgate, dup, E, dhq, nullptr, nullptr);
        strata::prefill::lat_q3k_down(dblob, dgs, G, dhq, dy, ELL, nullptr, dcnt);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(yser.data(), dy, yser.size() * 4, cudaMemcpyDeviceToHost));
        std::vector<float> yfu((size_t) E * ELL);
        CK(cudaMemcpy(yfu.data(), dy2, yfu.size() * 4, cudaMemcpyDeviceToHost));
        const long long d = n_diff(yser.data(), yfu.data(), yser.size());
        std::printf("gate 3  fused vs gu -> swiglu -> down (counts): %lld of %zu floats differ  <- must be 0\n", d,
                    yser.size());
        if (d) { std::printf("FAIL gate 3\n"); return 1; }
    }

    // ---- gate 4: the four shapes, fused (with counts) against native's whole step -------------------------------
    {
        void* scratch = nullptr;
        CK(cudaMalloc(&scratch, strata::kernels::native_expert_scratch_bytes(kMaxE, FF)));
        std::printf("\nthe shape sweep (whole A stage, us/entry, counts given; native from starts):\n");
        const int shapes[][2] = {{4, 1000}, {32, 128}, {200, 20}, {1400, 3}};
        double us200 = 0, nat200 = 0;
        for (const auto& sh : shapes) {
            const int epg = sh[0], G2 = sh[1], E2 = epg * G2;
            std::vector<int32_t> g2((size_t) G2 + 1), c2((size_t) G2, epg);
            for (int g = 0; g <= G2; ++g) g2[(size_t) g] = g * epg;
            CK(cudaMemcpy(dgs, g2.data(), g2.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dcnt, c2.data(), c2.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dng, &G2, 4, cudaMemcpyHostToDevice));
            const int iters = E2 >= 4000 ? 20 : 50;
            strata::kernels::native_expert_set_mode(-1, 0);
            auto tm = [&] {
                strata::prefill::lat_q3k_fused(dblob, dgs, G2, dxq, dhbuf, dy, ELL, nullptr, dcnt);
                CK(cudaDeviceSynchronize());
                cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
                CK(cudaEventRecord(t0));
                for (int i = 0; i < iters; ++i)
                    strata::prefill::lat_q3k_fused(dblob, dgs, G2, dxq, dhbuf, dy, ELL, nullptr, dcnt);
                CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
                float ms = 0; CK(cudaEventElapsedTime(&ms, t0, t1));
                CK(cudaEventDestroy(t0)); CK(cudaEventDestroy(t1));
                return ms / iters;
            };
            auto tn = [&] {
                strata::kernels::native_expert_grouped(L, (const unsigned long long*) dblob, dgs, dng, ded, det, G2, E2,
                                                       dxq, scratch, dy, nullptr, 0);
                CK(cudaDeviceSynchronize());
                cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
                CK(cudaEventRecord(t0));
                for (int i = 0; i < iters; ++i)
                    strata::kernels::native_expert_grouped(L, (const unsigned long long*) dblob, dgs, dng, ded, det,
                                                           G2, E2, dxq, scratch, dy, nullptr, 0);
                CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
                float ms = 0; CK(cudaEventElapsedTime(&ms, t0, t1));
                CK(cudaEventDestroy(t0)); CK(cudaEventDestroy(t1));
                return ms / iters;
            };
            const double a = tm(), b = tn();
            if (epg == 200) { us200 = a * 1000.0 / E2; nat200 = b * 1000.0 / E2; }
            std::printf("  epg %4d  groups %4d  E %5d | fused %.3f ms %6.3f us/entry | native %.3f ms %6.3f us/entry | "
                        "fused/native %.2fx\n", epg, G2, E2, a, a * 1000.0 / E2, b, b * 1000.0 / E2, b > 0 ? a / b : 0.0);
        }
        std::printf("epg200: fused %.4f vs native %.4f us/entry  (4K projection: %.3f ms)\n", us200, nat200,
                    us200 * 2.03e6 / 1000.0);
        if (!(us200 <= nat200)) { std::printf("FAIL gate 4 perf\n"); return 1; }
    }

    std::printf("PASS: M3b (explicit counts; nullptr unchanged, scrambled starts identical)\n");
    return 0;
}
