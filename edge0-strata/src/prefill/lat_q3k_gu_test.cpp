// src/prefill/lat_q3k_gu_test.cpp - M2a of the q3k latent A stage: gate/up (Q3_K, K = 1280) against a
// double-precision host dequantization of the same blob bytes and the same q8_1 activations, plus the decode gate:
// one row's super-block, its 16 sub-blocks' codes and sub-scales against the host's.
// Acceptance (q3k_fused/BRIEF.md, M2a): rel_mean <= 1e-3, cos_min >= 0.9999, no row under 0.99, all data real;
// the decode gate: codes exactly equal, sub-scales within 1e-6 relative.
#include "strata/prefill/lat_q3k_gu.hpp"
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
constexpr int kXNB = (int) (ELL / 32);          // the gate/up input row: 40 q8_1 blocks of 36 B
constexpr int kRowBytes = 550;                  // a Q3_K gate/up row: 5 x 110 B
constexpr int kSB = 5;                          // super-blocks in a row
constexpr int kChunks = 20;
constexpr int kStagedRow = strata::prefill::lat_q3k_gu_staged_row;

struct H8 { uint16_t d, s; int8_t qs[32]; };

// ggml-quants.c:1305 dequantize_row_q3_K, transcribed: 16 sub-blocks per 128-value half, the 12-byte 6-bit scales
// (sub-block s: the low (s < 8) or high nibble of byte s & 7, and the two high bits 2 * (s / 4) of byte 8 + (s & 3)).
void host_q3k_row(const uint8_t* row, float* y) {
    for (int i = 0; i < kSB; ++i) {
        const uint8_t* b = row + 110 * i;
        const float d_all = ggml_fp16_to_fp32(*(const uint16_t*) (b + 108));
        int sc[16];
        for (int s = 0; s < 16; ++s) {
            const int nib = (b[96 + (s & 7)] >> (4 * (s >> 3))) & 0xF;
            const int hi2 = (b[96 + 8 + (s & 3)] >> (2 * (s >> 2))) & 3;
            sc[s] = nib | (hi2 << 4);
        }
        for (int s = 0; s < 16; ++s) {
            const float dl = d_all * (float) (sc[s] - 32);
            const int t = (s & 7) >> 1, m = t + 4 * (s >> 3), qb = 32 * (s >> 3) + 16 * (s & 1);
            const int hb = 16 * (s & 1);   // the hmask is not advanced across the halves (ggml: hm[l] / hm[l + 16])
            for (int l = 0; l < 16; ++l) {
                const int q = b[32 + qb + l], h = b[hb + l];
                const int code = ((q >> (2 * t)) & 3) - (((h >> m) & 1) ? 0 : 4);
                y[(size_t) i * 256 + 16 * s + l] = dl * (float) code;
            }
        }
    }
}

// One sub-block's 16 codes and its scale (the host's view of the same bytes).
void host_q3k_sub(const uint8_t* row, int sb, int sub, int* codes, double* scale) {
    const uint8_t* b = row + 110 * sb;
    const int s = sub;   // 0..15 within the super-block
    const int nib = (b[96 + (s & 7)] >> (4 * (s >> 3))) & 0xF;
    const int hi2 = (b[96 + 8 + (s & 3)] >> (2 * (s >> 2))) & 3;
    *scale = (double) ggml_fp16_to_fp32(*(const uint16_t*) (b + 108)) * (double) ((nib | (hi2 << 4)) - 32);
    const int t = (s & 7) >> 1, m = t + 4 * (s >> 3), qb = 32 * (s >> 3) + 16 * (s & 1);
    const int hb = 16 * (s & 1);
    for (int l = 0; l < 16; ++l) {
        const int q = b[32 + qb + l], h = b[hb + l];
        codes[l] = ((q >> (2 * t)) & 3) - (((h >> m) & 1) ? 0 : 4);
    }
}

}  // namespace

int main() {
    constexpr int G = 128, EPG = 4, E = G * EPG;
    const size_t gub = (size_t) 1280 * kRowBytes;   // 1280 Q3_K rows (gate 640 + up 640)
    const size_t dnb = (size_t) ELL * FF / 32 * 18;
    std::printf("one expert: gu %zu B (Q3_K 1280x1280), dn %zu B (IQ4_NL 1280x640)\n", gub, dnb);

    std::mt19937 rng(11);
    std::normal_distribution<float> nw(0.f, 0.05f), nd(0.f, 1.f);
    std::vector<uint8_t> blob(gub + dnb);
    std::vector<float> wgu((size_t) 1280 * ELL);
    {
        for (auto& v : wgu) v = nw(rng);
        if ((size_t) ggml_quantize_chunk(GGML_TYPE_Q3_K, wgu.data(), blob.data(), 0, 1280, ELL, nullptr) != gub) {
            std::printf("FAIL gu quantize\n"); return 1;
        }
        std::vector<float> wd((size_t) ELL * FF);
        for (auto& v : wd) v = nw(rng);
        if ((size_t) ggml_quantize_chunk(GGML_TYPE_IQ4_NL, wd.data(), blob.data() + gub, 0, ELL, FF, nullptr) != dnb) {
            std::printf("FAIL dn quantize\n"); return 1;
        }
    }

    // the host dequantization against ggml's own quantizer: if my reading of the block were wrong, this would be
    // garbage rather than Q3_K's ~1 % quantization noise
    {
        std::vector<float> deq((size_t) ELL);
        double num = 0, den = 0, aa = 0, bb = 0;
        for (int r = 0; r < 1280; ++r) {
            host_q3k_row(blob.data() + (size_t) r * kRowBytes, deq.data());
            for (int k = 0; k < ELL; ++k) {
                const double a = deq[(size_t) k], b = wgu[(size_t) r * ELL + k];
                num += a * b; aa += a * a; bb += b * b; den += std::fabs(b);
            }
        }
        const double cs = num / (std::sqrt(aa) * std::sqrt(bb) + 1e-30);
        std::printf("host dequant vs the quantized source: cos %.6f (Q3_K's own 3.44-bit noise)\n", cs);
        if (!(cs > 0.98)) { std::printf("FAIL host dequant (structure, not precision)\n"); return 1; }
    }

    // 128 experts: replicas at distinct addresses, 4 entries each
    uint8_t* dall = nullptr;
    CK(cudaMalloc(&dall, (size_t) G * blob.size()));
    {
        std::vector<uint8_t> all((size_t) G * blob.size());
        for (int g = 0; g < G; ++g) std::memcpy(all.data() + (size_t) g * blob.size(), blob.data(), blob.size());
        CK(cudaMemcpy(dall, all.data(), all.size(), cudaMemcpyHostToDevice));
    }
    std::vector<const uint8_t*> bp((size_t) G);
    for (int g = 0; g < G; ++g) bp[(size_t) g] = dall + (size_t) g * blob.size();
    const uint8_t** dblob = nullptr;
    CK(cudaMalloc(&dblob, (size_t) G * sizeof(uint8_t*)));
    CK(cudaMemcpy(dblob, bp.data(), (size_t) G * sizeof(uint8_t*), cudaMemcpyHostToDevice));
    int32_t *dgs = nullptr, *dng = nullptr;
    CK(cudaMalloc(&dgs, (size_t) (G + 1) * sizeof(int32_t)));
    CK(cudaMalloc(&dng, sizeof(int32_t)));
    std::vector<int32_t> gs((size_t) G + 1), ng(1, G);
    for (int g = 0; g <= G; ++g) gs[(size_t) g] = g * EPG;
    CK(cudaMemcpy(dgs, gs.data(), gs.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dng, ng.data(), 4, cudaMemcpyHostToDevice));

    // ---- the decode gate: one gate row's super-block, the kernel's staged rows against the host's bytes ---------
    uint8_t* drow = nullptr;
    CK(cudaMalloc(&drow, kRowBytes));
    CK(cudaMemcpy(drow, blob.data(), kRowBytes, cudaMemcpyHostToDevice));
    std::vector<uint8_t> dump((size_t) kChunks * kStagedRow);
    uint8_t* ddump = nullptr;
    CK(cudaMalloc(&ddump, dump.size()));
    strata::prefill::lat_q3k_gu_dump(drow, 0, ddump, nullptr);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(dump.data(), ddump, dump.size(), cudaMemcpyDeviceToHost));
    {
        const int sb = 2;                       // the super-block under the microscope
        int worst_code = 0;
        double worst_sc = 0, worst_sc_rel = 0;
        int shown = 0;
        for (int cl = 0; cl < 4; ++cl) {
            const uint8_t* st = dump.data() + (size_t) (4 * sb + cl) * kStagedRow;
            for (int sub = 0; sub < 4; ++sub) {
                int codes[16];
                double sc = 0;
                host_q3k_sub(blob.data(), sb, 4 * cl + sub, codes, &sc);
                // the staged order: word k byte c is the sub-block's value 4c + k, so byte p is value 4*(p & 3)+(p>>2)
                int mine[16];
                int lo = 99, hi = -99;
                for (int p = 0; p < 16; ++p) mine[4 * (p & 3) + (p >> 2)] = (int8_t) st[16 * sub + p];
                for (int l = 0; l < 16; ++l) {
                    worst_code = std::max(worst_code, std::abs(mine[l] - codes[l]));
                    lo = std::min(lo, mine[l]); hi = std::max(hi, mine[l]);
                }
                float scm = 0;
                std::memcpy(&scm, st + 64 + 4 * sub, 4);
                worst_sc = std::max(worst_sc, std::fabs((double) scm - sc));
                worst_sc_rel = std::max(worst_sc_rel, std::fabs((double) scm - sc) / (std::fabs(sc) + 1e-30));
                if (shown < 2) {
                    std::printf("  sb %d sub %2d: codes[", sb, 4 * cl + sub);
                    for (int l = 0; l < 16; ++l) std::printf(" %2d", mine[l]);
                    std::printf(" ] host["); 
                    for (int l = 0; l < 16; ++l) std::printf(" %2d", codes[l]);
                    std::printf(" ] in [%d,%d]  scale %.9g host %.9g\n", lo, hi, (double) scm, sc);
                    ++shown;
                }
            }
        }
        std::printf("decode gate: 16 sub-blocks of sb %d -> max code |diff| %d, max scale |diff| %.3e (rel %.3e)\n",
                    sb, worst_code, worst_sc, worst_sc_rel);
        if (worst_code != 0 || worst_sc_rel > 1e-6) { std::printf("FAIL decode gate\n"); return 1; }
    }

    // ---- the products -----------------------------------------------------------------------------------------
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

    float* dgate = nullptr;
    float* dup = nullptr;
    CK(cudaMalloc(&dgate, (size_t) E * FF * sizeof(float)));
    CK(cudaMalloc(&dup, (size_t) E * FF * sizeof(float)));
    CK(cudaMemset(dgate, 0, (size_t) E * FF * sizeof(float)));
    CK(cudaMemset(dup, 0, (size_t) E * FF * sizeof(float)));
    strata::prefill::lat_q3k_gu(dblob, dgs, G, dxq, dgate, dup, FF, nullptr);
    CK(cudaDeviceSynchronize());
    std::vector<float> gate((size_t) E * FF), up((size_t) E * FF);
    CK(cudaMemcpy(gate.data(), dgate, gate.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(up.data(), dup, up.size() * 4, cudaMemcpyDeviceToHost));

    // the host reference: the activations the kernel reads (d8 * code) against the dequantized rows, in double
    std::vector<double> xh((size_t) E * ELL);
    for (int e = 0; e < E; ++e)
        for (int b = 0; b < kXNB; ++b) {
            const double d8 = (double) ggml_fp16_to_fp32(hx[(size_t) e * kXNB + b].d);
            for (int i = 0; i < 32; ++i)
                xh[(size_t) e * ELL + 32 * b + i] = d8 * (double) hx[(size_t) e * kXNB + b].qs[i];
        }
    std::vector<float> wrow((size_t) ELL);
    std::vector<double> ref((size_t) 2 * E * FF);
    for (int r = 0; r < 1280; ++r) {
        const bool is_up = r >= FF;
        host_q3k_row(blob.data() + (size_t) r * kRowBytes, wrow.data());
        const size_t proj = is_up ? 1 : 0;
        const int f = r % FF;
        for (int e = 0; e < E; ++e) {
            const double* xr = xh.data() + (size_t) e * ELL;
            double s = 0;
            for (int k = 0; k < ELL; ++k) s += (double) wrow[(size_t) k] * xr[k];
            ref[((proj * E) + e) * FF + f] = s;
        }
    }

    // native_expert_grouped's gate/up only (its phase 3) on the same xq: the cross-check of the host reference
    std::vector<float> ngate((size_t) E * FF), nup((size_t) E * FF);
    {
        const auto L = strata::kernels::native_expert_layout(GGML_TYPE_Q3_K, GGML_TYPE_IQ4_NL, ELL, FF);
        void* nscratch = nullptr;
        CK(cudaMalloc(&nscratch, strata::kernels::native_expert_scratch_bytes(E, FF)));
        int32_t* ded = nullptr; int32_t* det = nullptr; int32_t* dnn = nullptr;
        CK(cudaMalloc(&ded, (size_t) E * 4)); CK(cudaMalloc(&det, (size_t) E * 4)); CK(cudaMalloc(&dnn, 4));
        std::vector<int32_t> id((size_t) E), ngv(1, G);
        for (int e = 0; e < E; ++e) id[(size_t) e] = e;
        CK(cudaMemcpy(ded, id.data(), id.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(det, id.data(), id.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dnn, ngv.data(), 4, cudaMemcpyHostToDevice));
        strata::kernels::native_expert_set_mode(-1, 3);
        strata::kernels::native_expert_grouped(L, (const unsigned long long*) dblob, dgs, dnn, ded, det, G, E, dxq,
                                               nscratch, dgate, nullptr, 0);
        CK(cudaDeviceSynchronize());
        const size_t fa = ((size_t) E * FF * sizeof(float) + 255) & ~(size_t) 255;
        CK(cudaMemcpy(ngate.data(), nscratch, (size_t) E * FF * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(nup.data(), (const uint8_t*) nscratch + fa, (size_t) E * FF * 4, cudaMemcpyDeviceToHost));
        strata::kernels::native_expert_set_mode(-1, 0);
        auto cmp = [&](const std::vector<float>& a, const double* b, const char* tag) {
            double num = 0, den = 0, aa = 0, bb = 0;
            for (int e = 0; e < E; ++e)
                for (int f = 0; f < FF; ++f) {
                    const double x = a[(size_t) e * FF + f], y = b[(size_t) e * FF + f];
                    num += x * y; aa += x * x; bb += y * y; den += std::fabs(y);
                }
            double dabs = 0;
            for (int e = 0; e < E; ++e)
                for (int f = 0; f < FF; ++f)
                    dabs += std::fabs((double) a[(size_t) e * FF + f] - b[(size_t) e * FF + f]);
            std::printf("%s vs host: cos %.9f  rel_mean %.3e\n", tag, num / (std::sqrt(aa) * std::sqrt(bb) + 1e-30),
                        den > 0 ? dabs / den : 0.0);
        };
        cmp(ngate, ref.data(), "native gate");
        cmp(nup, ref.data() + (size_t) E * FF, "native up  ");
    }

    // compared row by row: an (entry, projection)'s 640 values
    double sabs = 0, sref = 0;
    float dmax = 0.f;
    double cosmin = 1.0;
    int bad99 = 0, skipped = 0;
    for (int proj = 0; proj < 2; ++proj)
        for (int e = 0; e < E; ++e) {
            const float* m = (proj ? up.data() : gate.data()) + (size_t) e * FF;
            const double* rf = ref.data() + ((size_t) proj * E + e) * FF;
            double num = 0, aa = 0, bb = 0;
            for (int f = 0; f < FF; ++f) {
                const double d = std::fabs((double) m[f] - rf[f]);
                sabs += d; sref += std::fabs(rf[f]);
                dmax = std::fmax(dmax, (float) d);
                num += (double) m[f] * rf[f]; aa += (double) m[f] * m[f]; bb += rf[f] * rf[f];
            }
            if (aa == 0 && bb == 0) { ++skipped; continue; }
            const double cos = num / (std::sqrt(aa) * std::sqrt(bb) + 1e-30);
            cosmin = std::min(cosmin, cos);
            if (cos < 0.99) ++bad99;
        }
    std::printf("rows=%d (skipped %d)  rel_mean=%.3e  max|d|=%.6e  cos_min=%.9f  rows<0.99=%d\n", 2 * E, skipped,
                sref > 0 ? sabs / sref : 0.0, (double) dmax, cosmin, bad99);
    std::printf("gate[0][0:8] mine:"); for (int f = 0; f < 8; ++f) std::printf(" %.6f", gate[(size_t) f]);
    std::printf("\ngate[0][0:8] host:"); for (int f = 0; f < 8; ++f) std::printf(" %.6f", ref[(size_t) f]);
    std::printf("\nup  [0][0:8] mine:"); for (int f = 0; f < 8; ++f) std::printf(" %.6f", up[(size_t) f]);
    std::printf("\nup  [0][0:8] host:"); for (int f = 0; f < 8; ++f) std::printf(" %.6f", ref[(size_t) (E * FF) + f]);
    std::printf("\n");

    // ---- timing (informational) -------------------------------------------------------------------------------
    {
        auto t = [&] {
            strata::prefill::lat_q3k_gu(dblob, dgs, G, dxq, dgate, dup, FF, nullptr);
            CK(cudaDeviceSynchronize());
            cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
            CK(cudaEventRecord(t0));
            for (int i = 0; i < 20; ++i) strata::prefill::lat_q3k_gu(dblob, dgs, G, dxq, dgate, dup, FF, nullptr);
            CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
            float ms = 0; CK(cudaEventElapsedTime(&ms, t0, t1));
            CK(cudaEventDestroy(t0)); CK(cudaEventDestroy(t1));
            return ms / 20.0;
        };
        const double ms = t();
        // the same shape through native_expert_grouped's gate/up only (phase 3), informational as well
        const auto L2 = strata::kernels::native_expert_layout(GGML_TYPE_Q3_K, GGML_TYPE_IQ4_NL, ELL, FF);
        void* nsc = nullptr;
        CK(cudaMalloc(&nsc, strata::kernels::native_expert_scratch_bytes(E, FF)));
        int32_t* ded = nullptr; int32_t* det = nullptr; int32_t* dnn = nullptr;
        CK(cudaMalloc(&ded, (size_t) E * 4)); CK(cudaMalloc(&det, (size_t) E * 4)); CK(cudaMalloc(&dnn, 4));
        std::vector<int32_t> id2((size_t) E), ngv2(1, G);
        for (int e = 0; e < E; ++e) id2[(size_t) e] = e;
        CK(cudaMemcpy(ded, id2.data(), id2.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(det, id2.data(), id2.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dnn, ngv2.data(), 4, cudaMemcpyHostToDevice));
        strata::kernels::native_expert_set_mode(-1, 3);
        auto tn = [&] {
            strata::kernels::native_expert_grouped(L2, (const unsigned long long*) dblob, dgs, dnn, ded, det, G, E, dxq,
                                                   nsc, dgate, nullptr, 0);
            CK(cudaDeviceSynchronize());
            cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
            CK(cudaEventRecord(t0));
            for (int i = 0; i < 20; ++i)
                strata::kernels::native_expert_grouped(L2, (const unsigned long long*) dblob, dgs, dnn, ded, det, G, E,
                                                       dxq, nsc, dgate, nullptr, 0);
            CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
            float ms2 = 0; CK(cudaEventElapsedTime(&ms2, t0, t1));
            CK(cudaEventDestroy(t0)); CK(cudaEventDestroy(t1));
            return ms2 / 20.0;
        };
        const double msn = tn();
        std::printf("M2a gu: mine %.3f ms  %.2f us/entry | native gate/up only %.3f ms  %.2f us/entry (128 groups x 4 "
                    "entries, informational)\n", ms, ms * 1000.0 / E, msn, msn * 1000.0 / E);
    }

    const bool ok = (sref > 0 ? sabs / sref : 1.0) <= 1e-3 && cosmin >= 0.9999 && bad99 == 0;
    std::printf("%s: M2a (Q3_K gate/up)\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
