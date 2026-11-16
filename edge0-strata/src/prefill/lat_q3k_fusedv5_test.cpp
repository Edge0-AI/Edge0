// src/prefill/lat_q3k_fusedv5_test.cpp - the equivalence gate of moe_fused_q3k.cu (v5's kernel with the q3k decode)
// against the frozen chain lat_q3k_gu -> lat_q3k_swiglu_q8 -> lat_q3k_down, on the same input, byte for byte:
//   shape 1: 128 groups x 4 entries (the unit tests' shape)
//   shape 2: the engine's 512 groups with the dumped off[]/cnt[] (42450 entries, off[] not monotonic)
#include "strata/prefill/moe_fused_q3k.hpp"
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
constexpr int NG = 512, E = 42450;          // the engine's entry space (the dumps' off[NG])
const std::string DUMP = "/home/user/lat-s26/q3k/dump_ref/";

template <class T>
std::vector<T> rd(const char* name, size_t expect) {
    std::vector<T> v(expect);
    std::FILE* f = std::fopen((DUMP + name).c_str(), "rb");
    if (f == nullptr) { std::printf("FAIL cannot open %s\n", name); std::exit(1); }
    const size_t n = std::fread(v.data(), sizeof(T), expect, f);
    std::fclose(f);
    if (n != expect) { std::printf("FAIL %s: %zu of %zu\n", name, n, expect); std::exit(1); }
    return v;
}

template <class T>
void upl(T* dst, const std::vector<T>& src) {
    CK(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
}
}  // namespace

static void step(const char* m) { std::printf("[step] %s\n", m); std::fflush(stdout); }

int main() {
    std::mt19937 rng(101);
    step("start");
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
    step("blobs quantized");
    const size_t EB = (size_t) (GUB + DNB);
    uint8_t* dall = nullptr;
    CK(cudaMalloc(&dall, blob.size()));
    CK(cudaMemcpy(dall, blob.data(), blob.size(), cudaMemcpyHostToDevice));

    // 512 experts, one blob each (the v5 batch holds the whole layer)
    std::vector<const uint8_t*> bp((size_t) NG);
    for (int g = 0; g < NG; ++g) bp[(size_t) g] = dall + (size_t) (g % NBLOB) * EB;
    strata::prefill::fused::WideBatch wb;
    wb.e0 = 0; wb.e1 = NG;
    for (int g = 0; g < NG; ++g) wb.blob[g] = bp[(size_t) g];
    const uint8_t** dblob = nullptr;
    CK(cudaMalloc(&dblob, (size_t) NG * sizeof(uint8_t*)));
    CK(cudaMemcpy(dblob, bp.data(), (size_t) NG * sizeof(uint8_t*), cudaMemcpyHostToDevice));

    step("blobs on device");
    std::vector<uint8_t> xq((size_t) E * kXq);
    {
        std::uniform_int_distribution<int> u8(-127, 127);
        for (size_t i = 0; i < xq.size(); ++i) xq[i] = (uint8_t) u8(rng);
        for (int t = 0; t < E; ++t)
            for (int k = 0; k < (int) (ELL / 32); ++k) {
                uint8_t* b = xq.data() + (size_t) t * kXq + 36 * k;
                const uint16_t d16 = ggml_fp32_to_fp16(0.01f + 1e-4f * ((t + k) % 97)), s16 = 0;
                std::memcpy(b, &d16, 2); std::memcpy(b + 2, &s16, 2);
            }
    }
    uint8_t* dxq = nullptr;
    CK(cudaMalloc(&dxq, xq.size()));
    CK(cudaMemcpy(dxq, xq.data(), xq.size(), cudaMemcpyHostToDevice));
    std::vector<int32_t> ident((size_t) E);
    for (int e = 0; e < E; ++e) ident[(size_t) e] = e;
    int32_t* dident = nullptr;
    CK(cudaMalloc(&dident, (size_t) E * 4));
    upl(dident, ident);

    step("xq ready");
    int32_t *dgs = nullptr, *dcnt = nullptr, *dng = nullptr;
    CK(cudaMalloc(&dgs, (size_t) (NG + 1) * 4));
    CK(cudaMalloc(&dcnt, (size_t) NG * 4));
    CK(cudaMalloc(&dng, 4));
    float* dgate = nullptr; float* dup = nullptr;
    CK(cudaMalloc(&dgate, (size_t) E * FF * 4)); CK(cudaMalloc(&dup, (size_t) E * FF * 4));
    uint8_t* dhq = nullptr; uint8_t* dhbuf = nullptr;
    CK(cudaMalloc(&dhq, (size_t) E * kHq)); CK(cudaMalloc(&dhbuf, (size_t) E * kHq));
    float* dref = nullptr; float* dv5 = nullptr;
    CK(cudaMalloc(&dref, (size_t) E * ELL * 4)); CK(cudaMalloc(&dv5, (size_t) E * ELL * 4));
    // the v5 path's grouping tables (group_latent: the same counts/offsets, the 64-row tile list per expert)
    void* htables = nullptr;
    CK(cudaMalloc(&htables, strata::prefill::fused::latent_group_bytes(E, NG)));
    void* dtables = nullptr;
    CK(cudaMalloc(&dtables, strata::prefill::fused::latent_group_bytes(E, NG)));
    const strata::prefill::fused::LatentGeom lg{1280, 640, 550, 360, 352000, 704000, 0};

    std::vector<float> yref((size_t) E * ELL), yv5((size_t) E * ELL);
    std::vector<uint8_t> href((size_t) E * kHq), hv5((size_t) E * kHq);

    step("buffers ready");
    auto run_shape = [&](const char* tag, const std::vector<int32_t>& off, const std::vector<int32_t>& cnt) -> int {
        step(tag);
        const int ng = (int) cnt.size();
        upl(dgs, off); upl(dcnt, cnt);
        CK(cudaMemcpy(dng, &ng, 4, cudaMemcpyHostToDevice));
        int total = 0;
        for (int c : cnt) total += c;
        step("  reference");
        // reference: the frozen chain
        CK(cudaMemset(dhq, 0, (size_t) E * kHq));
        CK(cudaMemset(dref, 0, (size_t) E * ELL * 4));
        strata::prefill::lat_q3k_gu(dblob, dgs, ng, dxq, dgate, dup, FF, nullptr, dcnt, nullptr);
        strata::prefill::lat_q3k_swiglu_q8(dgate, dup, E, dhq, nullptr, nullptr);
        strata::prefill::lat_q3k_down(dblob, dgs, ng, dhq, dref, ELL, nullptr, dcnt, nullptr);
        step("  tables");
        // under test: moe_fused_q3k (v5's kernel + the q3k decode)
        CK(cudaMemset(htables, 0, strata::prefill::fused::latent_group_bytes(E, NG)));
        {
            std::vector<char> t(strata::prefill::fused::latent_group_bytes(E, NG));
            strata::prefill::fused::group_latent(cnt.data(), off.data(), NG, t.data(), t.size());
            CK(cudaMemcpy(dtables, t.data(), t.size(), cudaMemcpyHostToDevice));
        }
        CK(cudaMemset(dhbuf, 0, (size_t) E * kHq));
        CK(cudaMemset(dv5, 0, (size_t) E * ELL * 4));
        step("  v5 launch");
        strata::prefill::fused_q3k::experts_latent_q3k(wb, lg, NG, E, dtables, dxq, dident, dhbuf, dv5, nullptr);
        CK(cudaDeviceSynchronize());
        step("  copies");
        CK(cudaMemcpy(yref.data(), dref, yref.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(yv5.data(), dv5, yv5.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(href.data(), dhq, href.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hv5.data(), dhbuf, hv5.size(), cudaMemcpyDeviceToHost));

        long long hdiff = 0, odiff = 0, h_first = -1, o_first = -1;
        double omax = 0, oabs = 0, oref = 0;
        std::vector<char> mine((size_t) E, 0);
        for (int g = 0; g < ng; ++g)
            for (int e = off[(size_t) g]; e < off[(size_t) g] + cnt[(size_t) g]; ++e) mine[(size_t) e] = 1;
        for (int e = 0; e < E; ++e) {
            if (!mine[(size_t) e]) continue;
            for (size_t b = 0; b < (size_t) kHq; ++b) {
                const size_t at = (size_t) e * kHq + b;
                if (href[at] != hv5[at]) { ++hdiff; if (h_first < 0) h_first = (long long) at; }
            }
            for (int r = 0; r < ELL; ++r) {
                const size_t at = (size_t) e * ELL + r;
                const double a = yref[at], b2 = yv5[at];
                if (a != b2) {
                    ++odiff; if (o_first < 0) o_first = (long long) at;
                    const double d = std::fabs(a - b2);
                    omax = std::max(omax, d); oabs += d;
                }
                oref += std::fabs(b2);
            }
        }
        {   // is either side non-finite, and what does the first difference look like?
            long long nf_ref = 0, nf_v5 = 0;
            for (int e = 0; e < E; ++e) {
                if (!mine[(size_t) e]) continue;
                for (int r = 0; r < ELL; ++r) {
                    const float a = yref[(size_t) e * ELL + r], b2 = yv5[(size_t) e * ELL + r];
                    if (!std::isfinite(a)) ++nf_ref;
                    if (!std::isfinite(b2)) ++nf_v5;
                }
            }
            std::printf("%-30s ng=%d rows=%d  non-finite: ref %lld, v5 %lld\n", tag, ng, total, nf_ref, nf_v5);
            if (o_first >= 0) {
                const int e = (int) (o_first / ELL), r = (int) (o_first % ELL);
                std::printf("   first diff at entry %d row %d: ref %.6g  v5 %.6g   (ref[0:4] %.6g %.6g %.6g %.6g)\n", e, r,
                            (double) yref[(size_t) o_first], (double) yv5[(size_t) o_first],
                            (double) yref[(size_t) e * ELL], (double) yref[(size_t) e * ELL + 1],
                            (double) yref[(size_t) e * ELL + 2], (double) yref[(size_t) e * ELL + 3]);
            }
        }
        std::printf("   hbuf: differing %lld / %lld B  first at byte %lld\n", hdiff, (long long) total * kHq, h_first);
        std::printf("   out : differing %lld / %lld values  max|d| %.3e  rel %.3e  first at value %lld\n", odiff,
                    (long long) total * ELL, omax, oref > 0 ? oabs / oref : 0.0, o_first);
        return (hdiff == 0 && odiff == 0) ? 0 : 1;
    };

    int fails = 0;
    {   // shape 1: 128 groups x 4 entries - padded to the pack's 512 experts, which is what group_latent walks
        std::vector<int32_t> off((size_t) NG + 1), cnt((size_t) NG, 0);
        for (int g = 0; g < 128; ++g) cnt[(size_t) g] = 4;
        for (int g = 0; g <= NG; ++g) off[(size_t) g] = 4 * std::min(g, 128);
        fails += run_shape("128 groups x 4 entries", off, cnt);
    }
    {   // shape 2: the engine's own table (already 512 experts)
        std::vector<int32_t> off = rd<int32_t>("g-off.bin", NG + 1);
        std::vector<int32_t> cnt = rd<int32_t>("g-cnt.bin", NG);
        fails += run_shape("engine table (512 groups)", off, cnt);
    }

    // timing (reported only after the gate passes)
    if (fails == 0) {
        std::vector<int32_t> off = rd<int32_t>("g-off.bin", NG + 1);
        std::vector<int32_t> cnt = rd<int32_t>("g-cnt.bin", NG);
        upl(dgs, off); upl(dcnt, cnt);
        CK(cudaMemcpy(dng, &NG, 4, cudaMemcpyHostToDevice));
        cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
        strata::prefill::fused_q3k::experts_latent_q3k(wb, lg, NG, E, dtables, dxq, dident, dhbuf, dv5, nullptr);
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(t0));
        for (int i = 0; i < 20; ++i)
            strata::prefill::fused_q3k::experts_latent_q3k(wb, lg, NG, E, dtables, dxq, dident, dhbuf, dv5, nullptr);
        CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, t0, t1));
        std::printf("experts_latent_q3k: %.3f ms / 20 = %.4f ms  %.4f us/entry (engine shape, %d entries)\n", ms,
                    ms / 20.0, ms / 20.0 * 1000.0 / E, E);
        std::printf("frozen chain for comparison:");
        CK(cudaEventRecord(t0));
        for (int i = 0; i < 20; ++i) {
            strata::prefill::lat_q3k_gu(dblob, dgs, NG, dxq, dgate, dup, FF, nullptr, dcnt, nullptr);
            strata::prefill::lat_q3k_swiglu_q8(dgate, dup, E, dhq, nullptr, nullptr);
            strata::prefill::lat_q3k_down(dblob, dgs, NG, dhq, dref, ELL, nullptr, dcnt, nullptr);
        }
        CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
        CK(cudaEventElapsedTime(&ms, t0, t1));
        std::printf(" %.4f ms  %.4f us/entry\n", ms / 20.0, ms / 20.0 * 1000.0 / E);
        CK(cudaEventDestroy(t0)); CK(cudaEventDestroy(t1));
    }

    std::printf("%s: moe_fused_q3k equivalent to the frozen chain (%d shapes differ)\n", fails == 0 ? "PASS" : "FAIL",
                fails);
    return fails == 0 ? 0 : 1;
}
