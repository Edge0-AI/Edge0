// src/prefill/lat_q3k_down_test.cpp - M1 of the q3k latent A stage: the down half (IQ4_NL, K = 640) against
// native_expert_grouped's own down, on the same staged H it produced (its phase knob stops it after the q8_1 of H,
// then runs only the down).  Two things are checked:
//   1. numbers: 128 groups x 4 entries of real quantized weights at 128 distinct blob addresses, against the
//      reference and against a double-precision CPU dot of the same weights and the same H row;
//   2. speed: the engine's shapes (entries per group 4 / 32 / 200 / 1400, ~4000 entries a call) against the
//      reference's down on the same data.
// Acceptance (q3k_fused/BRIEF.md): rel_mean <= 1e-3, cos_min >= 0.9999, no non-zero row under 0.99.
#include "strata/prefill/lat_q3k_down.hpp"
#include "strata/prefill/moe_mmq.hpp"
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

namespace mmq = strata::prefill::mmq;
#define CK(e) do { const cudaError_t r_ = (e); if (r_ != cudaSuccess) { \
    std::printf("FAIL %s: %s\n", #e, cudaGetErrorString(r_)); std::exit(1); } } while (0)

namespace {

constexpr int64_t ELL = 1280, FF = 640;
constexpr int kNB = (int) (FF / 32);        // a plain q8_1 H row: 20 blocks of 36 B
constexpr int kXNB = (int) (ELL / 32);      // the gate/up input row: n_embd / 32 = 40 blocks of 36 B
constexpr int kChunks = (int) (FF / 64);    // the staged H row: 10 chunks of 80 B
constexpr int kMaxE = 4500, kMaxG = 1024;   // the sweep's largest call (entries, groups)

// One q8_1 block as the engine's quantizer stores it: {half d, half s, int8 qs[32]}.
struct H8 { uint16_t d, s; int8_t qs[32]; };

// A plain q8_1 H row (20 x 36 B) into the staged activation rows lat_q3k_down reads: per 64 values, the two 32-value
// blocks' 64 codes in q8_1_store's order (a sub-block's byte p holds its value 4 * (p & 3) + (p >> 2)), then the
// blocks' d's at 64.  This is the layout moe_fused.cu's latent gate/up epilogue writes for the down to read.
void stage_h(const std::vector<H8>& q8, int64_t rows, std::vector<uint8_t>& out) {
    out.assign((size_t) rows * kChunks * 80, 0);
    for (int64_t r = 0; r < rows; ++r) {
        for (int kb = 0; kb < kChunks; ++kb) {
            uint8_t* dst = out.data() + ((size_t) r * kChunks + kb) * 80;
            for (int b = 0; b < 2; ++b) {
                const H8& blk = q8[(size_t) r * kNB + 2 * kb + b];
                std::memcpy(dst + 64 + 2 * b, &blk.d, 2);
                for (int h = 0; h < 2; ++h)
                    for (int j = 0; j < 16; ++j)
                        dst[16 * (2 * b + h) + 4 * (j & 3) + (j >> 2)] = (uint8_t) blk.qs[16 * h + j];
            }
        }
    }
}

// The IQ4_NL codebook (ggml-common.h, kvalues_iq4nl) and a double-precision dot of the same weights against the
// same q8_1 H row - the arbiter between this kernel and native_expert_grouped when the two disagree.
const int8_t kIQ4[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

double cpu_dot(const uint8_t* dn, const H8* hq_row, int row) {
    const uint8_t* wr = dn + (size_t) row * 360;
    double s = 0;
    for (int b = 0; b < 20; ++b) {
        const uint8_t* blk = wr + 18 * b;
        uint16_t dh;
        std::memcpy(&dh, blk, 2);
        const double dw = (double) ggml_fp16_to_fp32(dh);
        const double d8 = (double) ggml_fp16_to_fp32(hq_row[b].d);
        for (int j = 0; j < 16; ++j) {
            const double wl = dw * (double) kIQ4[blk[2 + j] & 0xF], wh = dw * (double) kIQ4[blk[2 + j] >> 4];
            s += wl * (d8 * (double) hq_row[b].qs[j]) + wh * (d8 * (double) hq_row[b].qs[16 + j]);
        }
    }
    return s;
}

unsigned long long fnv(const std::vector<float>& v) {
    unsigned long long h = 1469598103934665603ULL;
    for (float f : v) { uint32_t u; std::memcpy(&u, &f, 4); h = (h ^ u) * 1099511628211ULL; }
    return h;
}

}  // namespace

int main() {
    constexpr int G = 128, EPG = 4, E = G * EPG;
    const size_t gub = mmq::matrix_bytes(GGML_TYPE_Q3_K, 1280, ELL);    // gate 640 rows + up 640 rows
    const size_t dnb = mmq::matrix_bytes(GGML_TYPE_IQ4_NL, ELL, FF);    // down 1280 x 640
    std::printf("one expert: gu %zu B (Q3_K 1280x1280), dn %zu B (IQ4_NL 1280x640)\n", gub, dnb);

    // the blob: real quantized weights (never random bytes - a random 6-bit scale is 1e-30 .. 1e+30)
    std::mt19937 rng(11);
    std::normal_distribution<float> nw(0.f, 0.05f), nd(0.f, 1.f);
    std::vector<uint8_t> blob(gub + dnb);
    {
        std::vector<float> w((size_t) 1280 * ELL);
        for (auto& v : w) v = nw(rng);
        if ((size_t) ggml_quantize_chunk(GGML_TYPE_Q3_K, w.data(), blob.data(), 0, 1280, ELL, nullptr) != gub) {
            std::printf("FAIL gu quantize\n"); return 1;
        }
        std::vector<float> wd((size_t) ELL * FF);
        for (auto& v : wd) v = nw(rng);
        if ((size_t) ggml_quantize_chunk(GGML_TYPE_IQ4_NL, wd.data(), blob.data() + gub, 0, ELL, FF, nullptr) != dnb) {
            std::printf("FAIL dn quantize\n"); return 1;
        }
    }
    const auto L = strata::kernels::native_expert_layout(GGML_TYPE_Q3_K, GGML_TYPE_IQ4_NL, ELL, FF);
    std::printf("layout: gu_row %zu up_off %zu down_off %zu d_row %zu bytes %zu tail %zu\n", L.gu_row, L.up_off,
                L.down_off, L.d_row, L.bytes, L.tail_off);
    if (L.bytes != gub + dnb || L.down_off != 704000 || L.d_row != 360 || L.gu_row != 550 || L.tail_off != 0) {
        std::printf("FAIL layout != the q3k pack's (704000 / 360 / 550 / no tail)\n");
        return 1;
    }

    // 128 experts: replicas of this blob at distinct addresses (the reads are DRAM traffic), 4 entries each
    uint8_t* dall = nullptr;
    CK(cudaMalloc(&dall, (size_t) G * L.bytes));
    {
        std::vector<uint8_t> one(L.bytes);
        std::memcpy(one.data(), blob.data(), gub);
        std::memcpy(one.data() + L.down_off, blob.data() + gub, dnb);
        std::vector<uint8_t> all((size_t) G * L.bytes);
        for (int g = 0; g < G; ++g) std::memcpy(all.data() + (size_t) g * L.bytes, one.data(), L.bytes);
        CK(cudaMemcpy(dall, all.data(), all.size(), cudaMemcpyHostToDevice));
    }
    std::vector<const uint8_t*> bp((size_t) kMaxG);
    for (int g = 0; g < kMaxG; ++g) bp[(size_t) g] = dall + (size_t) (g % G) * L.bytes;
    const uint8_t** dblob = nullptr;
    CK(cudaMalloc(&dblob, (size_t) kMaxG * sizeof(uint8_t*)));
    CK(cudaMemcpy(dblob, bp.data(), (size_t) kMaxG * sizeof(uint8_t*), cudaMemcpyHostToDevice));

    // the gate/up input: a different x row per entry, so a row swap anywhere shows up (40 q8_1 blocks = n_embd 1280)
    std::vector<float> xs((size_t) E * ELL);
    for (auto& v : xs) v = nd(rng);
    std::vector<H8> hx((size_t) E * kXNB);
    for (int e = 0; e < E; ++e)
        for (int b = 0; b < kXNB; ++b) {
            float amax = 0.f, sum = 0.f;
            for (int i = 0; i < 32; ++i) {
                const float v = xs[(size_t) e * ELL + 32 * b + i];
                amax = std::fmax(amax, std::fabs(v)); sum += v;
            }
            const float d = amax / 127.0f;
            H8& o = hx[(size_t) e * kXNB + b];
            o.d = ggml_fp32_to_fp16(d);
            o.s = ggml_fp32_to_fp16(sum);
            for (int i = 0; i < 32; ++i)
                o.qs[i] = (int8_t) std::lround(d != 0.f ? xs[(size_t) e * ELL + 32 * b + i] / d : 0.f);
        }
    uint8_t* dxq = nullptr;
    CK(cudaMalloc(&dxq, hx.size() * sizeof(H8)));
    CK(cudaMemcpy(dxq, hx.data(), hx.size() * sizeof(H8), cudaMemcpyHostToDevice));

    int32_t *dgs = nullptr, *dng = nullptr, *ded = nullptr, *det = nullptr;
    CK(cudaMalloc(&dgs, (size_t) (G + 1) * sizeof(int32_t)));
    CK(cudaMalloc(&dng, sizeof(int32_t)));
    CK(cudaMalloc(&ded, (size_t) E * sizeof(int32_t)));
    CK(cudaMalloc(&det, (size_t) E * sizeof(int32_t)));
    std::vector<int32_t> gs((size_t) G + 1), ed((size_t) E), et((size_t) E), ng(1, G);
    for (int g = 0; g <= G; ++g) gs[(size_t) g] = g * EPG;
    for (int e = 0; e < E; ++e) { ed[(size_t) e] = e; et[(size_t) e] = e; }
    CK(cudaMemcpy(dgs, gs.data(), gs.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dng, ng.data(), 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ded, ed.data(), ed.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(det, et.data(), et.size() * 4, cudaMemcpyHostToDevice));

    void* scratch = nullptr;
    const size_t sb = strata::kernels::native_expert_scratch_bytes(E, FF);
    CK(cudaMalloc(&scratch, sb));
    float* dout = nullptr;
    CK(cudaMalloc(&dout, (size_t) E * ELL * sizeof(float)));
    std::printf("native: %d groups x %d entries, scratch %zu B\n", G, EPG, sb);

    // the reference: phase 1 = gate/up + SwiGLU + q8_1 (H), phase 2 = the down of that H, phase 0 = both.  All three
    // read the same buffers, so 0 and 1+2 must match bitwise - it is the check that H is the one we compare against.
    strata::kernels::native_expert_set_mode(-1, 1);
    strata::kernels::native_expert_grouped(L, (const unsigned long long*) dblob, dgs, dng, ded, det, G, E, dxq, scratch,
                                           dout, nullptr, 0);
    CK(cudaDeviceSynchronize());
    const size_t fa = ((size_t) E * FF * sizeof(float) + 255) & ~(size_t) 255;
    std::vector<H8> hq_host((size_t) E * kNB);
    CK(cudaMemcpy(hq_host.data(), (const uint8_t*) scratch + 3 * fa, hq_host.size() * sizeof(H8),
                  cudaMemcpyDeviceToHost));

    strata::kernels::native_expert_set_mode(-1, 2);
    strata::kernels::native_expert_grouped(L, (const unsigned long long*) dblob, dgs, dng, ded, det, G, E, dxq, scratch,
                                           dout, nullptr, 0);
    CK(cudaDeviceSynchronize());
    std::vector<float> ref((size_t) E * ELL);
    CK(cudaMemcpy(ref.data(), dout, ref.size() * 4, cudaMemcpyDeviceToHost));

    strata::kernels::native_expert_set_mode(-1, 0);
    CK(cudaMemset(dout, 0, (size_t) E * ELL * sizeof(float)));
    strata::kernels::native_expert_grouped(L, (const unsigned long long*) dblob, dgs, dng, ded, det, G, E, dxq, scratch,
                                           dout, nullptr, 0);
    CK(cudaDeviceSynchronize());
    std::vector<float> full((size_t) E * ELL);
    CK(cudaMemcpy(full.data(), dout, full.size() * 4, cudaMemcpyDeviceToHost));
    std::printf("phase check: full %016llx  gu+h+down %016llx  %s\n", fnv(full), fnv(ref),
                fnv(full) == fnv(ref) ? "same" : "DIFFERENT");

    // the staged H, and the M1 down on it
    std::vector<uint8_t> staged;
    stage_h(hq_host, E, staged);
    uint8_t* dstaged = nullptr;
    CK(cudaMalloc(&dstaged, staged.size()));
    CK(cudaMemcpy(dstaged, staged.data(), staged.size(), cudaMemcpyHostToDevice));
    float* dmine = nullptr;
    CK(cudaMalloc(&dmine, (size_t) E * ELL * sizeof(float)));
    CK(cudaMemset(dmine, 0, (size_t) E * ELL * sizeof(float)));
    strata::prefill::lat_q3k_down(dblob, dgs, G, dstaged, dmine, ELL, nullptr);
    CK(cudaDeviceSynchronize());
    std::vector<float> mine((size_t) E * ELL);
    CK(cudaMemcpy(mine.data(), dmine, mine.size() * 4, cudaMemcpyDeviceToHost));

    // ---- the comparison ---------------------------------------------------------------------------------------
    // A row both sides leave at zero (no data for that entry) is not a difference: it is skipped.
    double sabs = 0, sref = 0;
    float dmax = 0.f;
    double cosmin = 1.0;
    int bad99 = 0, bad9999 = 0, skipped = 0;
    for (int e = 0; e < E; ++e) {
        double num = 0, aa = 0, bb = 0;
        float big = 0.f;
        for (int r = 0; r < ELL; ++r) {
            const float a = mine[(size_t) e * ELL + r], b = ref[(size_t) e * ELL + r];
            const double d = std::fabs((double) a - b);
            sabs += d; sref += std::fabs((double) b);
            dmax = std::fmax(dmax, (float) d);
            num += (double) a * b; aa += (double) a * a; bb += (double) b * b;
            big = std::fmax(big, std::fmax(std::fabs(a), std::fabs(b)));
        }
        if (big == 0.f) { ++skipped; continue; }
        const double cos = num / (std::sqrt(aa) * std::sqrt(bb) + 1e-30);
        cosmin = std::min(cosmin, cos);
        if (cos < 0.99) ++bad99;
        if (cos < 0.9999) ++bad9999;
    }
    std::printf("rows=%d (skipped %d all-zero)  rel_mean=%.3e  max|d|=%.6e  cos_min=%.9f  rows<0.99=%d  rows<0.9999=%d\n",
                E, skipped, sref > 0 ? sabs / sref : 0.0, (double) dmax, cosmin, bad99, bad9999);
    std::printf("row0 mine:"); for (int r = 0; r < 8; ++r) std::printf(" %.6f", mine[(size_t) r]);
    std::printf("\nrow0 ref :"); for (int r = 0; r < 8; ++r) std::printf(" %.6f", ref[(size_t) r]);
    std::printf("\n");
    {   // the double-precision arbiter, entry 0: the same weights, the same H row
        const uint8_t* dn = blob.data() + gub;
        double cd = 0, csum = 0, ccc = 0, crr = 0, nrc = 0;
        for (int r = 0; r < ELL; ++r) {
            const double c = cpu_dot(dn, &hq_host[0], r), a = ref[(size_t) r];
            ccc += c * c; cd += c * a; crr += a * a; csum += std::fabs(a); nrc += std::fabs(c - a);
        }
        double cm = 0, nm = 0, csum2 = 0, mm = 0;
        for (int r = 0; r < ELL; ++r) {
            const double c = cpu_dot(dn, &hq_host[0], r), b = mine[(size_t) r];
            cm += c * b; nm += std::fabs(c - b); csum2 += std::fabs(c); mm += b * b;
        }
        std::printf("entry0 cpu vs ref : rel_mean %.3e cos %.9f\n", nrc / csum,
                    cd / (std::sqrt(ccc) * std::sqrt(crr) + 1e-30));
        std::printf("entry0 cpu vs mine: rel_mean %.3e cos %.9f\n", nm / csum2,
                    cm / (std::sqrt(ccc) * std::sqrt(mm) + 1e-30));
    }

    // ---- the engine's shapes: entries per group 4 / 32 / 200 / 1400, ~4000 entries a call ----------------------
    {
        const int shapes[][2] = {{4, 1000}, {32, 128}, {200, 20}, {1400, 3}};
        std::vector<H8> plain((size_t) kMaxE * kNB);       // a valid H for the reference to read
        for (int e = 0; e < kMaxE; ++e)
            for (int b = 0; b < kNB; ++b) {
                H8& o = plain[(size_t) e * kNB + b];
                o.d = ggml_fp32_to_fp16(0.004f + 1e-5f * (e % 211));
                o.s = 0;
                for (int i = 0; i < 32; ++i) o.qs[i] = (int8_t) (((e * 7 + b * 13 + i * 3) % 255) - 127);
            }
        std::vector<uint8_t> stAll;
        stage_h(plain, kMaxE, stAll);
        uint8_t* dstage = nullptr; CK(cudaMalloc(&dstage, stAll.size()));
        CK(cudaMemcpy(dstage, stAll.data(), stAll.size(), cudaMemcpyHostToDevice));
        float* dsw = nullptr; CK(cudaMalloc(&dsw, (size_t) kMaxE * ELL * sizeof(float)));
        void* scr2 = nullptr; CK(cudaMalloc(&scr2, strata::kernels::native_expert_scratch_bytes(kMaxE, FF)));
        int32_t *dgs2 = nullptr, *ded2 = nullptr, *det2 = nullptr, *dng2 = nullptr;
        CK(cudaMalloc(&dgs2, (size_t) (kMaxG + 1) * 4));
        CK(cudaMalloc(&ded2, (size_t) kMaxE * 4));
        CK(cudaMalloc(&det2, (size_t) kMaxE * 4));
        CK(cudaMalloc(&dng2, 4));
        std::printf("\nthe shape sweep (one down, us/entry):\n");
        double worst = 0;
        for (const auto& sh : shapes) {
            const int epg = sh[0], G2 = sh[1], E2 = epg * G2;
            std::vector<int32_t> g2((size_t) G2 + 1), d2((size_t) E2), t2((size_t) E2), ng2(1, G2);
            for (int g = 0; g <= G2; ++g) g2[(size_t) g] = g * epg;
            for (int e = 0; e < E2; ++e) { d2[(size_t) e] = e; t2[(size_t) e] = e; }
            CK(cudaMemcpy(dgs2, g2.data(), g2.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(ded2, d2.data(), d2.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(det2, t2.data(), t2.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dng2, ng2.data(), 4, cudaMemcpyHostToDevice));
            const size_t fa2 = ((size_t) E2 * FF * sizeof(float) + 255) & ~(size_t) 255;
            CK(cudaMemcpy((uint8_t*) scr2 + 3 * fa2, plain.data(), (size_t) E2 * kNB * sizeof(H8),
                          cudaMemcpyHostToDevice));
            const int iters = E2 >= 4000 ? 20 : 50;
            strata::kernels::native_expert_set_mode(-1, 2);
            auto time2 = [&](bool use_mine) {
                if (use_mine) strata::prefill::lat_q3k_down(dblob, dgs2, G2, dstage, dsw, ELL, nullptr);
                else strata::kernels::native_expert_grouped(L, (const unsigned long long*) dblob, dgs2, dng2, ded2, det2,
                                                            G2, E2, dxq, scr2, dsw, nullptr, 0);
                CK(cudaDeviceSynchronize());
                cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
                CK(cudaEventRecord(t0));
                for (int i = 0; i < iters; ++i) {
                    if (use_mine) strata::prefill::lat_q3k_down(dblob, dgs2, G2, dstage, dsw, ELL, nullptr);
                    else strata::kernels::native_expert_grouped(L, (const unsigned long long*) dblob, dgs2, dng2, ded2,
                                                                det2, G2, E2, dxq, scr2, dsw, nullptr, 0);
                }
                CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
                float ms = 0; CK(cudaEventElapsedTime(&ms, t0, t1));
                CK(cudaEventDestroy(t0)); CK(cudaEventDestroy(t1));
                return ms / iters;
            };
            const double tm = time2(true), tr = time2(false);
            const double r = tr > 0 ? tm / tr : 0.0;
            worst = std::max(worst, r);
            std::printf("  epg %4d  groups %4d  E %5d | mine %.3f ms %6.3f us/entry | native %.3f ms %6.3f us/entry | "
                        "mine/native %.2fx\n", epg, G2, E2, tm, tm * 1000.0 / E2, tr, tr * 1000.0 / E2, r);
        }
        std::printf("worst shape: mine/native %.2fx\n", worst);
    }

    const bool ok = (sref > 0 ? sabs / sref : 1.0) <= 1e-3 && cosmin >= 0.9999 && bad99 == 0;
    std::printf("%s: M1 (IQ4_NL down, %d groups x %d entries)\n", ok ? "PASS" : "FAIL", G, EPG);
    return ok ? 0 : 1;
}
