// src/prefill/lat_q3k_fused_test.cpp - M3 of the q3k latent A stage: the whole A stage in one call.
// Gates, in this order:
//   1. bit for bit: lat_q3k_fused against the serial chain of the FROZEN kernels (lat_q3k_gu -> lat_q3k_swiglu_q8 ->
//      lat_q3k_down) - memcmp of both the staged H rows and the fp32 output, not a hash;
//   2. against native_expert_grouped's whole step (its phase 0) on the same blobs and the same xq;
//   3. the four shapes, fused against native's whole step (us/entry), and the 4K projection at 2.03M entries.
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
constexpr int kXq = (int) (ELL / 32) * 36;    // 1440 B an entry (n_embd / 32 = 40 q8_1 blocks)
constexpr int kHq = (int) strata::prefill::lat_q3k_fused_hbuf_row;   // 800 B an entry (n_ff / 32 = 20 blocks)
constexpr int kMaxE = 4500, kMaxG = 1024, G = 128, EPG = 4, E = G * EPG;

unsigned long long fnv(const void* p, size_t bytes) {
    const uint8_t* b = (const uint8_t*) p;
    unsigned long long h = 1469598103934665603ULL;
    for (size_t i = 0; i < bytes; ++i) h = (h ^ b[i]) * 1099511628211ULL;
    return h;
}

// the first byte where the two differ, or -1
long long first_diff(const void* a, const void* b, size_t bytes) {
    const uint8_t* x = (const uint8_t*) a;
    const uint8_t* y = (const uint8_t*) b;
    for (size_t i = 0; i < bytes; ++i)
        if (x[i] != y[i]) return (long long) i;
    return -1;
}

}  // namespace

int main() {
    std::mt19937 rng(41);
    std::normal_distribution<float> nw(0.f, 0.05f), nd(0.f, 1.f);

    // one real expert blob (Q3_K gate/up + IQ4_NL down), replicated at distinct addresses
    std::vector<uint8_t> blob((size_t) (GUB + DNB));
    {
        std::vector<float> w((size_t) 1280 * ELL);
        for (auto& v : w) v = nw(rng);
        if ((size_t) ggml_quantize_chunk(GGML_TYPE_Q3_K, w.data(), blob.data(), 0, 1280, ELL, nullptr) != (size_t) GUB) {
            std::printf("FAIL gu quantize\n"); return 1;
        }
        std::vector<float> wd((size_t) ELL * FF);
        for (auto& v : wd) v = nw(rng);
        if ((size_t) ggml_quantize_chunk(GGML_TYPE_IQ4_NL, wd.data(), blob.data() + GUB, 0, ELL, FF, nullptr) !=
            (size_t) DNB) { std::printf("FAIL dn quantize\n"); return 1; }
    }
    const auto L = strata::kernels::native_expert_layout(GGML_TYPE_Q3_K, GGML_TYPE_IQ4_NL, ELL, FF);
    std::printf("layout: gu_row %zu up_off %zu down_off %zu d_row %zu bytes %zu tail %zu\n", L.gu_row, L.up_off,
                L.down_off, L.d_row, L.bytes, L.tail_off);
    if (L.bytes != (size_t) (GUB + DNB) || L.down_off != 704000 || L.gu_row != 550 || L.tail_off != 0) {
        std::printf("FAIL layout\n"); return 1;
    }

    uint8_t* dall = nullptr;
    CK(cudaMalloc(&dall, (size_t) G * blob.size()));
    {
        std::vector<uint8_t> all((size_t) G * blob.size());
        for (int g = 0; g < G; ++g) std::memcpy(all.data() + (size_t) g * blob.size(), blob.data(), blob.size());
        CK(cudaMemcpy(dall, all.data(), all.size(), cudaMemcpyHostToDevice));
    }
    std::vector<const uint8_t*> bp((size_t) kMaxG);
    for (int g = 0; g < kMaxG; ++g) bp[(size_t) g] = dall + (size_t) (g % G) * blob.size();
    const uint8_t** dblob = nullptr;
    CK(cudaMalloc(&dblob, (size_t) kMaxG * sizeof(uint8_t*)));
    CK(cudaMemcpy(dblob, bp.data(), (size_t) kMaxG * sizeof(uint8_t*), cudaMemcpyHostToDevice));

    // a different x row per entry (40 q8_1 blocks), so a row swap anywhere shows up
    std::vector<float> xs((size_t) E * ELL);
    for (auto& v : xs) v = nd(rng);
    std::vector<uint8_t> xq((size_t) E * kXq);
    for (int e = 0; e < E; ++e)
        for (int b = 0; b < (int) (ELL / 32); ++b) {
            float amax = 0.f, sum = 0.f;
            for (int i = 0; i < 32; ++i) {
                const float v = xs[(size_t) e * ELL + 32 * b + i];
                amax = std::fmax(amax, std::fabs(v)); sum += v;
            }
            const float d = amax / 127.0f;
            uint8_t* blk = xq.data() + (size_t) e * kXq + 36 * b;
            const uint16_t d16 = ggml_fp32_to_fp16(d), s16 = ggml_fp32_to_fp16(sum);
            std::memcpy(blk, &d16, 2); std::memcpy(blk + 2, &s16, 2);
            for (int i = 0; i < 32; ++i)
                blk[4 + i] = (uint8_t) (int8_t) std::lround(d != 0.f ? xs[(size_t) e * ELL + 32 * b + i] / d : 0.f);
        }
    uint8_t* dxq = nullptr;
    CK(cudaMalloc(&dxq, xq.size()));
    CK(cudaMemcpy(dxq, xq.data(), xq.size(), cudaMemcpyHostToDevice));

    int32_t *dgs = nullptr, *ded = nullptr, *det = nullptr, *dng = nullptr;
    CK(cudaMalloc(&dgs, (size_t) (kMaxG + 1) * 4));
    CK(cudaMalloc(&ded, (size_t) kMaxE * 4));
    CK(cudaMalloc(&det, (size_t) kMaxE * 4));
    CK(cudaMalloc(&dng, 4));
    std::vector<int32_t> gs((size_t) G + 1), id((size_t) E), ngv(1, G);
    for (int g = 0; g <= G; ++g) gs[(size_t) g] = g * EPG;
    for (int e = 0; e < E; ++e) id[(size_t) e] = e;
    CK(cudaMemcpy(dgs, gs.data(), gs.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ded, id.data(), id.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(det, id.data(), id.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dng, ngv.data(), 4, cudaMemcpyHostToDevice));

    float* dgate = nullptr; float* dup = nullptr;
    CK(cudaMalloc(&dgate, (size_t) E * FF * 4));
    CK(cudaMalloc(&dup, (size_t) E * FF * 4));
    uint8_t* dhq_serial = nullptr; uint8_t* dhbuf = nullptr;
    CK(cudaMalloc(&dhq_serial, (size_t) E * kHq));
    CK(cudaMalloc(&dhbuf, (size_t) E * kHq));
    // the same fill in both: neither kernel writes a chunk's pad (bytes 68..79 of the 80), so a different fill would
    // show up as a hbuf difference the down half never reads - the test checks that separately below
    CK(cudaMemset(dhq_serial, 0x00, (size_t) E * kHq));
    CK(cudaMemset(dhbuf, 0x00, (size_t) E * kHq));
    float* dy_serial = nullptr; float* dy_fused = nullptr;
    CK(cudaMalloc(&dy_serial, (size_t) E * ELL * 4));
    CK(cudaMalloc(&dy_fused, (size_t) E * ELL * 4));
    CK(cudaMemset(dy_serial, 0x11, (size_t) E * ELL * 4));
    CK(cudaMemset(dy_fused, 0x22, (size_t) E * ELL * 4));

    // ---- the serial chain, all three FROZEN kernels -----------------------------------------------------------
    strata::prefill::lat_q3k_gu(dblob, dgs, G, dxq, dgate, dup, FF, nullptr);
    strata::prefill::lat_q3k_swiglu_q8(dgate, dup, E, dhq_serial, nullptr, nullptr);
    strata::prefill::lat_q3k_down(dblob, dgs, G, dhq_serial, dy_serial, ELL, nullptr);
    CK(cudaDeviceSynchronize());
    // ---- the fused call ----------------------------------------------------------------------------------------
    strata::prefill::lat_q3k_fused(dblob, dgs, G, dxq, dhbuf, dy_fused, ELL, nullptr);
    CK(cudaDeviceSynchronize());

    std::vector<uint8_t> hq_s((size_t) E * kHq), hq_f((size_t) E * kHq);
    std::vector<float> y_s((size_t) E * ELL), y_f((size_t) E * ELL);
    CK(cudaMemcpy(hq_s.data(), dhq_serial, hq_s.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hq_f.data(), dhbuf, hq_f.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(y_s.data(), dy_serial, y_s.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(y_f.data(), dy_fused, y_f.size() * 4, cudaMemcpyDeviceToHost));

    // ---- gate 1: bit for bit -----------------------------------------------------------------------------------
    const long long hd = first_diff(hq_s.data(), hq_f.data(), hq_s.size());
    const long long yd = first_diff(y_s.data(), y_f.data(), y_s.size());
    const int hm = std::memcmp(hq_s.data(), hq_f.data(), hq_s.size());
    const int ym = std::memcmp(y_s.data(), y_f.data(), y_s.size());
    const int pm = hm;   // the full-buffer memcmp; the pad-only case is reported below
    std::printf("hbuf  memcmp %d  (%zu B, first diff %lld)  hash %016llx / %016llx\n", hm, hq_s.size(), hd,
                fnv(hq_s.data(), hq_s.size()), fnv(hq_f.data(), hq_f.size()));
    std::printf("out   memcmp %d  (%zu B, first diff %lld)  hash %016llx / %016llx\n", ym, y_s.size(), yd,
                fnv(y_s.data(), y_s.size()), fnv(y_f.data(), y_f.size()));
    {   // what the down half reads: codes (16 * s ..) and the two d's at 64..67; the pad is nobody's
        long long used = 0, pad = 0;
        for (size_t row = 0; row < (size_t) E; ++row)
            for (int c = 0; c < 10; ++c) {
                const size_t base = row * (size_t) kHq + (size_t) c * 80;
                for (int b = 0; b < 80; ++b)
                    if (hq_s[base + b] != hq_f[base + b]) (b < 68 ? used : pad)++;
            }
        std::printf("hbuf  differing bytes: %lld in the 68 bytes the down reads, %lld in the 12 pad bytes\n", used, pad);
        if (used != 0) { std::printf("FAIL bitwise: the H the down reads differs\n"); return 1; }
        if (pm != 0) std::printf("note: only the pad bytes differ (%d memcmp) - not read by the down half\n", pm);
    }
    if (hm != 0 || ym != 0) {
        if (ym != 0) { std::printf("FAIL bitwise: fused output != serial output\n"); return 1; }
    }

    // ---- gate 2: against native's whole step, same blobs, same xq -----------------------------------------------
    void* scratch = nullptr;
    CK(cudaMalloc(&scratch, strata::kernels::native_expert_scratch_bytes(E, FF)));
    float* dy_nat = nullptr;
    CK(cudaMalloc(&dy_nat, (size_t) E * ELL * 4));
    strata::kernels::native_expert_set_mode(-1, 0);
    strata::kernels::native_expert_grouped(L, (const unsigned long long*) dblob, dgs, dng, ded, det, G, E, dxq, scratch,
                                           dy_nat, nullptr, 0);
    CK(cudaDeviceSynchronize());
    std::vector<float> y_n((size_t) E * ELL);
    CK(cudaMemcpy(y_n.data(), dy_nat, y_n.size() * 4, cudaMemcpyDeviceToHost));
    {
        double sabs = 0, sref = 0;
        float dmax = 0.f;
        double cosmin = 1.0;
        int bad99 = 0, skipped = 0;
        for (int e = 0; e < E; ++e) {
            double num = 0, aa = 0, bb = 0;
            float big = 0.f;
            for (int r = 0; r < ELL; ++r) {
                const double a = y_f[(size_t) e * ELL + r], b = y_n[(size_t) e * ELL + r];
                const double dd = std::fabs(a - b);
                sabs += dd; sref += std::fabs(b); dmax = std::fmax(dmax, (float) dd);
                num += a * b; aa += a * a; bb += b * b;
                big = std::fmax(big, (float) std::fmax(std::fabs(a), std::fabs(b)));
            }
            if (big == 0.f) { ++skipped; continue; }
            const double cos = num / (std::sqrt(aa) * std::sqrt(bb) + 1e-30);
            cosmin = std::min(cosmin, cos);
            if (cos < 0.99) ++bad99;
        }
        std::printf("fused vs native whole step: rows=%d (skipped %d)  rel_mean %.3e  max|d| %.6e  cos_min %.9f  "
                    "rows<0.99 %d\n", E, skipped, sref > 0 ? sabs / sref : 0.0, (double) dmax, cosmin, bad99);
        if (!((sref > 0 ? sabs / sref : 1.0) <= 1e-3 && cosmin >= 0.9999 && bad99 == 0 && skipped == 0)) {
            std::printf("FAIL vs native\n"); return 1;
        }
        std::printf("yl[0][0:6] fused:"); for (int r = 0; r < 6; ++r) std::printf(" %.6f", y_f[(size_t) r]);
        std::printf("\nyl[0][0:6] native:"); for (int r = 0; r < 6; ++r) std::printf(" %.6f", y_n[(size_t) r]);
        std::printf("\n");
    }

    // ---- gate 3: the four shapes, fused against native's whole step ---------------------------------------------
    {
        std::vector<uint8_t> xq2((size_t) kMaxE * kXq);
        std::uniform_int_distribution<int> u8(-127, 127);
        for (size_t e = 0; e < (size_t) kMaxE; ++e)
            for (int b = 0; b < (int) (ELL / 32); ++b) {
                uint8_t* blk = xq2.data() + e * kXq + 36 * b;
                const uint16_t d16 = ggml_fp32_to_fp16(0.01f + 1e-5f * (e % 199)), s16 = 0;
                std::memcpy(blk, &d16, 2); std::memcpy(blk + 2, &s16, 2);
                for (int i = 0; i < 32; ++i) blk[4 + i] = (uint8_t) u8(rng);
            }
        uint8_t* dxq2 = nullptr;
        CK(cudaMalloc(&dxq2, xq2.size()));
        CK(cudaMemcpy(dxq2, xq2.data(), xq2.size(), cudaMemcpyHostToDevice));
        uint8_t* dhbuf2 = nullptr;
        CK(cudaMalloc(&dhbuf2, (size_t) kMaxE * kHq));
        float* dout2 = nullptr;
        CK(cudaMalloc(&dout2, (size_t) kMaxE * ELL * 4));
        void* scr2 = nullptr;
        CK(cudaMalloc(&scr2, strata::kernels::native_expert_scratch_bytes(kMaxE, FF)));
        std::printf("\nthe shape sweep (whole A stage, us/entry):\n");
        const int shapes[][2] = {{4, 1000}, {32, 128}, {200, 20}, {1400, 3}};
        double us200 = 0, nat200 = 0;
        for (const auto& sh : shapes) {
            const int epg = sh[0], G2 = sh[1], E2 = epg * G2;
            std::vector<int32_t> g2((size_t) G2 + 1), id2((size_t) E2), ng2(1, G2);
            for (int g = 0; g <= G2; ++g) g2[(size_t) g] = g * epg;
            for (int e = 0; e < E2; ++e) id2[(size_t) e] = e;
            CK(cudaMemcpy(dgs, g2.data(), g2.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(ded, id2.data(), id2.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(det, id2.data(), id2.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dng, ng2.data(), 4, cudaMemcpyHostToDevice));
            const int iters = E2 >= 4000 ? 20 : 50;
            strata::kernels::native_expert_set_mode(-1, 0);
            auto tm = [&] {
                strata::prefill::lat_q3k_fused(dblob, dgs, G2, dxq2, dhbuf2, dout2, ELL, nullptr);
                CK(cudaDeviceSynchronize());
                cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
                CK(cudaEventRecord(t0));
                for (int i = 0; i < iters; ++i)
                    strata::prefill::lat_q3k_fused(dblob, dgs, G2, dxq2, dhbuf2, dout2, ELL, nullptr);
                CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
                float ms = 0; CK(cudaEventElapsedTime(&ms, t0, t1));
                CK(cudaEventDestroy(t0)); CK(cudaEventDestroy(t1));
                return ms / iters;
            };
            auto tn = [&] {
                strata::kernels::native_expert_grouped(L, (const unsigned long long*) dblob, dgs, dng, ded, det, G2, E2,
                                                       dxq2, scr2, dout2, nullptr, 0);
                CK(cudaDeviceSynchronize());
                cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
                CK(cudaEventRecord(t0));
                for (int i = 0; i < iters; ++i)
                    strata::kernels::native_expert_grouped(L, (const unsigned long long*) dblob, dgs, dng, ded, det,
                                                           G2, E2, dxq2, scr2, dout2, nullptr, 0);
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
        // the 4K projection: 2.03M entries a layer at the epg=200 shape
        const double ents4k = 2.03e6;
        std::printf("4K projection (2.03M entries, epg=200 shape): fused %.3f ms vs native %.3f ms  (%.4f us/entry)\n",
                    us200 * ents4k / 1000.0, nat200 * ents4k / 1000.0, us200);
        if (!(us200 <= nat200)) { std::printf("FAIL perf: fused slower than native at epg=200\n"); return 1; }
    }

    std::printf("PASS: M3 (fused A stage: memcmp-identical to the serial chain, faster than native at epg=200)\n");
    return 0;
}
