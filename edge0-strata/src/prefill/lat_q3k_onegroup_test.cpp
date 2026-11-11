// src/prefill/lat_q3k_onegroup_test.cpp - the shape the engine actually runs when it walks one expert at a time:
// ONE group with thousands of rows (so the launcher's blockIdx.z fill is active), and the boundaries around a tile.
// The reference is native_expert_grouped, which has no z fill at all - the serial chain of my own kernels cannot be
// the reference here, because the gu and the down both carry the same z fill and a defect in it cancels out.
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
#include <string>
#include <vector>

#define CK(e) do { const cudaError_t r_ = (e); if (r_ != cudaSuccess) { \
    std::printf("FAIL %s: %s\n", #e, cudaGetErrorString(r_)); std::exit(1); } } while (0)

namespace {
constexpr int64_t ELL = 1280, FF = 640;
constexpr int64_t GUB = 1280 * 550, DNB = ELL * FF / 32 * 18;
constexpr int kXq = (int) (ELL / 32) * 36;
constexpr int kHq = (int) strata::prefill::lat_q3k_fused_hbuf_row;
constexpr int NBLOB = 4;
constexpr int E = 42450;          // the engine's entry space (the dumps' off[NG])
constexpr int NTOK = E;           // one xl row an entry here (the engine shares rows; the maps are tested elsewhere)

long long n_diff(const float* a, const float* b, size_t n) {
    long long d = 0;
    for (size_t i = 0; i < n; ++i) d += (a[i] != b[i]);
    return d;
}
}  // namespace

int main() {
    std::mt19937 rng(83);
    std::normal_distribution<float> nw(0.f, 0.05f);
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
    std::vector<const uint8_t*> bp((size_t) 8);
    for (int g = 0; g < 8; ++g) bp[(size_t) g] = dall + (size_t) (g % NBLOB) * EB;
    const uint8_t** dblob = nullptr;
    CK(cudaMalloc(&dblob, 8 * sizeof(uint8_t*)));
    CK(cudaMemcpy(dblob, bp.data(), 8 * sizeof(uint8_t*), cudaMemcpyHostToDevice));

    std::vector<uint8_t> xq((size_t) NTOK * kXq);
    {
        std::uniform_int_distribution<int> u8(-127, 127);
        for (size_t i = 0; i < xq.size(); ++i) xq[i] = (uint8_t) u8(rng);
        for (int t = 0; t < NTOK; ++t)
            for (int k = 0; k < (int) (ELL / 32); ++k) {
                uint8_t* b = xq.data() + (size_t) t * kXq + 36 * k;
                const uint16_t d16 = ggml_fp32_to_fp16(0.01f + 1e-4f * ((t + k) % 97)), s16 = 0;
                std::memcpy(b, &d16, 2); std::memcpy(b + 2, &s16, 2);
            }
    }
    uint8_t* dxq = nullptr;
    CK(cudaMalloc(&dxq, xq.size()));
    CK(cudaMemcpy(dxq, xq.data(), xq.size(), cudaMemcpyHostToDevice));

    int32_t *dgs = nullptr, *dcnt = nullptr, *dident = nullptr, *dng = nullptr;
    CK(cudaMalloc(&dgs, 9 * 4)); CK(cudaMalloc(&dcnt, 8 * 4));
    CK(cudaMalloc(&dident, (size_t) E * 4)); CK(cudaMalloc(&dng, 4));
    std::vector<int32_t> ident((size_t) E);
    for (int e = 0; e < E; ++e) ident[(size_t) e] = e;
    CK(cudaMemcpy(dident, ident.data(), ident.size() * 4, cudaMemcpyHostToDevice));
    int32_t* dident_tok = nullptr;   // native dereferences ent_tok as well: the identity, so entry e reads row e
    CK(cudaMalloc(&dident_tok, (size_t) E * 4));
    CK(cudaMemcpy(dident_tok, ident.data(), ident.size() * 4, cudaMemcpyHostToDevice));

    float* dgate = nullptr; float* dup = nullptr;
    CK(cudaMalloc(&dgate, (size_t) E * FF * 4)); CK(cudaMalloc(&dup, (size_t) E * FF * 4));
    uint8_t* dhq = nullptr; uint8_t* dhbuf = nullptr;
    CK(cudaMalloc(&dhq, (size_t) E * kHq)); CK(cudaMalloc(&dhbuf, (size_t) E * kHq));
    float *dy = nullptr, *dy2 = nullptr, *dyn = nullptr;
    CK(cudaMalloc(&dy, (size_t) E * ELL * 4)); CK(cudaMalloc(&dy2, (size_t) E * ELL * 4));
    CK(cudaMalloc(&dyn, (size_t) E * ELL * 4));
    const auto L = strata::kernels::native_expert_layout(GGML_TYPE_Q3_K, GGML_TYPE_IQ4_NL, ELL, FF);
    void* scratch = nullptr;
    CK(cudaMalloc(&scratch, strata::kernels::native_expert_scratch_bytes(E, FF)));

    std::vector<float> ys((size_t) E * ELL), yf((size_t) E * ELL), yn((size_t) E * ELL);

    // one case: ng groups with (starts, counts); every row zeroed first so an unwritten row shows
    auto run_case = [&](const char* tag, int ng, const int32_t* starts, const int32_t* counts, bool vs_native) -> int {
        CK(cudaMemcpy(dgs, starts, (size_t) (ng + 1) * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dcnt, counts, (size_t) ng * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dng, &ng, 4, cudaMemcpyHostToDevice));
        int total = 0;
        for (int g = 0; g < ng; ++g) total += counts[g];
        // fused
        CK(cudaMemset(dy2, 0, (size_t) E * ELL * 4));
        strata::prefill::lat_q3k_fused(dblob, dgs, ng, dxq, dhbuf, dy2, ELL, nullptr, dcnt, nullptr, dident);
        // serial, same buffers
        CK(cudaMemset(dy, 0, (size_t) E * ELL * 4));
        strata::prefill::lat_q3k_gu(dblob, dgs, ng, dxq, dgate, dup, FF, nullptr, dcnt, nullptr);
        strata::prefill::lat_q3k_swiglu_q8(dgate, dup, E, dhq, nullptr, nullptr);
        strata::prefill::lat_q3k_down(dblob, dgs, ng, dhq, dy, ELL, nullptr, dcnt, dident);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(ys.data(), dy, ys.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(yf.data(), dy2, yf.size() * 4, cudaMemcpyDeviceToHost));
        const long long dser = n_diff(ys.data(), yf.data(), ys.size());
        double rel = 0, cosmin = 1.0;
        int bad = 0, first_bad = -1, first_rel = -1;
        if (vs_native) {
            strata::kernels::native_expert_set_mode(-1, 0);
            CK(cudaMemset(dyn, 0, (size_t) E * ELL * 4));
            strata::kernels::native_expert_grouped(L, (const unsigned long long*) dblob, dgs, dng, dident, dident_tok,
                                                   ng, E, dxq, scratch, dyn, nullptr, 0);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(yn.data(), dyn, yn.size() * 4, cudaMemcpyDeviceToHost));
            double sabs = 0, sref = 0;
            for (int g = 0; g < ng; ++g)
                for (int e = starts[g]; e < starts[g] + counts[g]; ++e) {
                    double num = 0, aa = 0, bb = 0;
                    for (int r = 0; r < ELL; ++r) {
                        const double a = yf[(size_t) e * ELL + r], b = yn[(size_t) e * ELL + r];
                        sabs += std::fabs(a - b); sref += std::fabs(b);
                        num += a * b; aa += a * a; bb += b * b;
                    }
                    const double c = num / (std::sqrt(aa) * std::sqrt(bb) + 1e-30);
                    cosmin = std::min(cosmin, c);
                    if (c < 0.99) {
                        ++bad;
                        if (first_bad < 0) first_bad = e - starts[g];
                        if (first_rel < 0 && std::fabs(yf[(size_t) e * ELL] - yn[(size_t) e * ELL]) > 1e-9)
                            first_rel = e - starts[g];
                    }
                }
            rel = sref > 0 ? sabs / sref : 0.0;
        }
        std::printf("%-28s ng=%d rows=%5d | fused vs serial: %lld diff | vs native: rel %.3e cos_min %.9f bad %d "
                    "(first bad row-in-group %d)\n", tag, ng, total, dser, rel, cosmin, bad, first_bad);
        const bool ok = dser == 0 && (!vs_native || (rel <= 1e-3 && cosmin >= 0.9999 && bad == 0));
        if (!ok) {
            std::printf("   FAIL %s\n", tag);
            for (int g = 0; g < ng; ++g) {
                long long dg = 0;
                int fb = -1;
                for (int e = starts[g]; e < starts[g] + counts[g]; ++e) {
                    const long long d = n_diff(yf.data() + (size_t) e * ELL, ys.data() + (size_t) e * ELL, ELL);
                    if (d && fb < 0) fb = e - starts[g];
                    dg += d;
                }
                std::printf("   group %d (start %d, cnt %d): %lld floats differ from serial, first at row %d\n", g,
                            starts[g], counts[g], dg, fb);
            }
        }
        return ok ? 0 : 1;
    };

    int fails = 0;
    // the engine's real big groups, one group a launch (what the per-expert walk does)
    {
        const int32_t s1[2] = {32028, 32028 + 2749}, c1[1] = {2749};
        fails += run_case("nb=1 cnt=2749 (group 327)", 1, s1, c1, true);
        const int32_t s2[2] = {34777, 34777 + 400}, c2[1] = {400};
        fails += run_case("nb=1 cnt=400 (group 334)", 1, s2, c2, true);
        const int32_t s3[2] = {37469, 37469 + 2376}, c3[1] = {2376};
        fails += run_case("nb=1 cnt=2376 (group 434)", 1, s3, c3, true);
        const int32_t s4[3] = {30567, 31035, 31076}, c4[2] = {468, 41};
        fails += run_case("nb=2 cnt={468,41} (253/254)", 2, s4, c4, true);
    }
    // the boundaries around a tile, and the counts the engine really reaches
    {
        const int sizes[] = {1, 2, 63, 64, 65, 127, 128, 129, 255, 256, 257, 468, 1231, 2748, 2749, 3098};
        for (int c : sizes) {
            const int32_t s[2] = {0, c}, ct[1] = {c};
            char tag[64];
            std::snprintf(tag, sizeof tag, "nb=1 sweep cnt=%d", c);
            fails += run_case(tag, 1, s, ct, (c == 2749 || c == 3098));
        }
    }
    std::printf("%s: one-group / z-fill / tile boundaries (fails %d)\n", fails == 0 ? "PASS" : "FAIL", fails);
    return fails == 0 ? 0 : 1;
}
