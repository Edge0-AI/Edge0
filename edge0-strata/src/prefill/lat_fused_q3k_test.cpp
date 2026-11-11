// The golden reference for the q3k latent A stage: ONE expert, its real blob layout, through the MMQ products
// that are already verified against the default path (rel_mean 1.4e-3, cos_min 0.99997).  The fused latent kernel's
// Q3_K/IQ4_NL dot has to reproduce these numbers; this prints them so the port has a target.
#include "strata/prefill/moe_mmq.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>
#include <ggml.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cstring>

namespace mmq = strata::prefill::mmq;
#define CK(e) do { const cudaError_t r_ = (e); if (r_ != cudaSuccess) { \
    std::printf("FAIL %s: %s\n", #e, cudaGetErrorString(r_)); std::exit(1); } } while (0)

int main(int argc, char** argv) {
    // argv[1]: 0 = the default IQ4_NL down, 1 = STRATA_LAT_DOWN_S26 (the S26 rows)
    const int s26 = argc > 1 ? std::atoi(argv[1]) : 0;
    if (s26) setenv("STRATA_LAT_DOWN_S26", "1", 1);
    constexpr int64_t ELL = 1280, FF = 640;
    const size_t gub = mmq::matrix_bytes(GGML_TYPE_Q3_K, 1280, ELL);    // gate 640 rows + up 640 rows
    const size_t dnb = mmq::matrix_bytes(GGML_TYPE_IQ4_NL, ELL, FF);    // down 1280 x 640
    std::printf("one expert: gu %zu B (Q3_K 1280x1280), dn %zu B (IQ4_NL 1280x640)\n", gub, dnb);

    std::mt19937 rng(11);
    std::normal_distribution<float> nw(0.f, 0.05f);
    // VALID weights: quantize a real weight matrix, never random bytes (a random 6-bit scale is 1e-30..1e+30)
    std::vector<uint8_t> blob(gub + dnb);
    {
        std::vector<float> w((size_t) 1280 * ELL);
        for (auto& v : w) v = nw(rng);
        const int64_t used = ggml_quantize_chunk(GGML_TYPE_Q3_K, w.data(), blob.data(), 0, 1280, ELL, nullptr);
        if (used != (int64_t) gub) { std::printf("FAIL gu quantize %lld != %zu\n", (long long) used, gub); return 1; }
        std::vector<float> wd((size_t) ELL * FF);
        for (auto& v : wd) v = nw(rng);
        const int64_t used2 = ggml_quantize_chunk(GGML_TYPE_IQ4_NL, wd.data(), blob.data() + gub, 0, ELL, FF, nullptr);
        if (used2 != (int64_t) dnb) { std::printf("FAIL dn quantize %lld != %zu\n", (long long) used2, dnb); return 1; }
    }
    std::vector<float> x((size_t) ELL);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) v = nd(rng);

    uint8_t *dgu = nullptr, *ddn = nullptr, *dxq = nullptr, *dhq = nullptr;
    float *dgo = nullptr, *dho = nullptr, *ddo = nullptr;
    int32_t* db = nullptr; int32_t* did = nullptr;
    CK(cudaMalloc(&dgu, gub)); CK(cudaMalloc(&ddn, dnb));
    CK(cudaMalloc(&dxq, mmq::q8_bytes(1, ELL))); CK(cudaMalloc(&dhq, mmq::q8_bytes(1, FF)));
    CK(cudaMalloc(&dgo, 1280 * sizeof(float))); CK(cudaMalloc(&dho, FF * sizeof(float)));
    CK(cudaMalloc(&ddo, ELL * sizeof(float)));
    CK(cudaMalloc(&db, 2 * sizeof(int32_t))); CK(cudaMalloc(&did, sizeof(int32_t)));

    CK(cudaMemcpy(dgu, blob.data(), gub, cudaMemcpyHostToDevice));            // gate|up, the blob's own order
    CK(cudaMemcpy(ddn, blob.data() + gub, dnb, cudaMemcpyHostToDevice));
    const int32_t b01[2] = {0, 1};
    const int32_t id0 = 0;
    CK(cudaMemcpy(db, b01, 2 * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(did, &id0, 4, cudaMemcpyHostToDevice));
    mmq::quantize(x.data(), nullptr, dxq, GGML_TYPE_Q3_K, ELL, ELL, 1, nullptr);

    mmq::Context ctx;
    mmq::Product gu;
    gu.w = dgu; gu.type = GGML_TYPE_Q3_K; gu.w_rows = 1280; gu.w_cols = ELL; gu.expert_bytes = gub;
    gu.n = 1; gu.xq = dxq; gu.bounds = db; gu.ids = did; gu.total_rows = 1; gu.max_rows = 1;
    gu.dst = dgo; gu.ld_dst = 1280;
    ctx.run(gu, nullptr);
    mmq::swiglu(dgo, dho, 1, FF, false, nullptr);
    mmq::quantize(dho, nullptr, dhq, GGML_TYPE_IQ4_NL, FF, FF, 1, nullptr);
    mmq::Product dn;
    dn.w = ddn; dn.type = GGML_TYPE_IQ4_NL; dn.w_rows = ELL; dn.w_cols = FF; dn.expert_bytes = dnb;
    dn.n = 1; dn.xq = dhq; dn.bounds = db; dn.ids = did; dn.total_rows = 1; dn.max_rows = 1;
    dn.dst = ddo; dn.ld_dst = ELL;
    ctx.run(dn, nullptr);
    CK(cudaDeviceSynchronize());

    std::vector<float> y((size_t) ELL), g((size_t) 1280), h((size_t) FF);
    CK(cudaMemcpy(g.data(), dgo, 1280 * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h.data(), dho, FF * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(y.data(), ddo, ELL * 4, cudaMemcpyDeviceToHost));
    double sg = 0, sh = 0, sy = 0;
    for (float v : g) sg += v;
    for (float v : h) sh += v;
    for (float v : y) sy += v;
    std::printf("GU  first8:"); for (int i = 0; i < 8; ++i) std::printf(" %.6f", g[(size_t) i]);
    std::printf("  sum %.6f\nH   first8:", sg);
    for (int i = 0; i < 8; ++i) std::printf(" %.6f", h[(size_t) i]);
    std::printf("  sum %.6f\nyl  first8:", sh);
    for (int i = 0; i < 8; ++i) std::printf(" %.6f", y[(size_t) i]);
    std::printf("  sum %.6f\n", sy);

    // ---- the native grouped kernels at the same latent layout ------------------------------------------------
    // 128 experts (replicas of this one blob, distinct addresses so the reads are DRAM traffic), 4 entries each:
    // the engine's shape, not one big group.  native_expert_grouped runs gate/up (Q3_K), SwiGLU + q8_1, then the
    // down; the phase knob stops it before the down, so down = all - nodown.  Every input is identical between the
    // two runs, so `out` must match bitwise.
    {
        const int G = 128, EPG = 4, E = G * EPG;
        const auto L = strata::kernels::native_expert_layout(GGML_TYPE_Q3_K, GGML_TYPE_IQ4_NL, ELL, FF);
        std::printf("layout: gu_row %zu up_off %zu down_off %zu d_row %zu bytes %zu\n", L.gu_row, L.up_off,
                    L.down_off, L.d_row, L.bytes);
        if (L.bytes != gub + dnb || L.up_off != gub / 2 || L.down_off != gub) {
            std::printf("FAIL layout != the pack's (bytes %zu vs %zu)\n", L.bytes, gub + dnb);
            return 1;
        }
        std::vector<uint8_t> one(L.bytes);
        std::memcpy(one.data(), blob.data(), gub);
        std::memcpy(one.data() + L.down_off, blob.data() + gub, dnb);
        std::vector<uint8_t> all((size_t) G * L.bytes);
        for (int g = 0; g < G; ++g) std::memcpy(all.data() + (size_t) g * L.bytes, one.data(), L.bytes);
        uint8_t* dall = nullptr; CK(cudaMalloc(&dall, all.size()));
        CK(cudaMemcpy(dall, all.data(), all.size(), cudaMemcpyHostToDevice));
        std::vector<unsigned long long> gp((size_t) G);
        for (int g = 0; g < G; ++g) gp[(size_t) g] = (unsigned long long) (dall + (size_t) g * L.bytes);

        // q8_1 rows of x (every entry the same x; the down's input is produced inside the call, so only X is ours)
        struct H8 { uint16_t d, s; int8_t qs[32]; };
        std::vector<H8> hx((size_t) E * (ELL / 32));
        for (int e = 0; e < E; ++e)
            for (int b = 0; b < ELL / 32; ++b) {
                float amax = 0.f, sum = 0.f;
                for (int i = 0; i < 32; ++i) {
                    const float v = x[(size_t) 32 * b + i];
                    amax = std::fmax(amax, std::fabs(v)); sum += v;
                }
                const float d = amax / 127.0f;
                H8& o = hx[(size_t) e * (ELL / 32) + b];
                o.d = ggml_fp32_to_fp16(d);
                o.s = ggml_fp32_to_fp16(sum);
                for (int i = 0; i < 32; ++i)
                    o.qs[i] = (int8_t) std::lround(d != 0.f ? x[(size_t) 32 * b + i] / d : 0.f);
            }
        uint8_t* dxq2 = nullptr;
        CK(cudaMalloc(&dxq2, hx.size() * sizeof(H8)));
        CK(cudaMemcpy(dxq2, hx.data(), hx.size() * sizeof(H8), cudaMemcpyHostToDevice));

        unsigned long long* dgp = nullptr;
        int32_t *dgs = nullptr, *dng = nullptr, *ded = nullptr, *det = nullptr;
        CK(cudaMalloc(&dgp, (size_t) G * sizeof(unsigned long long)));
        CK(cudaMalloc(&dgs, (size_t) (G + 1) * sizeof(int32_t)));
        CK(cudaMalloc(&dng, sizeof(int32_t)));
        CK(cudaMalloc(&ded, (size_t) E * sizeof(int32_t)));
        CK(cudaMalloc(&det, (size_t) E * sizeof(int32_t)));
        std::vector<int32_t> gs((size_t) G + 1), ed((size_t) E), et((size_t) E), ng(1, G);
        for (int g = 0; g <= G; ++g) gs[(size_t) g] = g * EPG;
        for (int e = 0; e < E; ++e) { ed[(size_t) e] = e; et[(size_t) e] = e; }
        CK(cudaMemcpy(dgp, gp.data(), gp.size() * 8, cudaMemcpyHostToDevice));
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

        // the same expert set as ONE call per `group_batch` groups: gs[] tiles exactly (gs[g] = g*EPG), so this is
        // the API's own contract.  batch 1 = what the engine's per-expert walk does.
        int32_t* dngk = nullptr;
        CK(cudaMalloc(&dngk, sizeof(int32_t)));
        auto run = [&](int phase, int group_batch) {
            strata::kernels::native_expert_set_mode(-1, phase);
            if (group_batch <= 0 || group_batch >= G) {
                strata::kernels::native_expert_grouped(L, dgp, dgs, dng, ded, det, G, E, dxq2, scratch, dout, nullptr, 0);
                return;
            }
            for (int q0 = 0; q0 < G; q0 += group_batch) {
                const int k = std::min(group_batch, G - q0);
                CK(cudaMemcpy(dngk, &k, 4, cudaMemcpyHostToDevice));
                strata::kernels::native_expert_grouped(L, dgp + q0, dgs + q0, dngk, ded, det, k, E, dxq2, scratch,
                                                       dout, nullptr, 0);
            }
        };
        auto time_it = [&](int phase, int gb) {
            run(phase, gb); CK(cudaDeviceSynchronize());
            cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
            CK(cudaEventRecord(t0));
            for (int i = 0; i < 50; ++i) run(phase, gb);
            CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
            float ms = 0; CK(cudaEventElapsedTime(&ms, t0, t1));
            CK(cudaEventDestroy(t0)); CK(cudaEventDestroy(t1));
            return ms / 50.0;
        };
        {   // the batching question first: the same groups, one call per gb groups
            unsigned long long ref = 0;
            for (int gb : {1, 2, 4, 8, 16, 64, 128}) {
                CK(cudaMemset(dout, 0, (size_t) E * ELL * sizeof(float)));
                run(0, gb);
                CK(cudaDeviceSynchronize());
                std::vector<float> oo((size_t) E * ELL);
                CK(cudaMemcpy(oo.data(), dout, oo.size() * 4, cudaMemcpyDeviceToHost));
                unsigned long long h = 1469598103934665603ULL;
                for (float v : oo) { uint32_t u; std::memcpy(&u, &v, 4); h = (h ^ u) * 1099511628211ULL; }
                const double us = time_it(0, gb) * 1000.0 / E;
                if (gb == 1) ref = h;
                std::printf("group_batch %3d: hash %016llx %s  %.2f us/entry\n", gb, h,
                            h == ref ? "same" : "DIFFERENT", us);
            }
        }
        const double t_nodown = time_it(1, 1);   // gate/up + SwiGLU + q8_1
        const double t_all = time_it(0, G);      // ... + the down
        std::vector<float> o((size_t) E * ELL);
        CK(cudaMemcpy(o.data(), dout, o.size() * 4, cudaMemcpyDeviceToHost));
        unsigned long long hsh = 1469598103934665603ULL;
        for (float v : o) { uint32_t u; std::memcpy(&u, &v, 4); hsh = (hsh ^ u) * 1099511628211ULL; }
        double om = 0; for (float v : o) om += v;
        double num = 0, den = 0;
        for (int i = 0; i < ELL; ++i) { num += std::fabs(o[(size_t) i] - y[(size_t) i]); den += std::fabs(y[(size_t) i]); }
        std::printf("s26=%d  all %.3f ms  nodown %.3f ms  down %.3f ms (%.1f us/entry)  hash %016llx  sum %.6f  row0 rel %.2e\n",
                    s26, t_all, t_nodown, t_all - t_nodown, (t_all - t_nodown) * 1000.0 / E, hsh, om,
                    den > 0 ? num / den : 0.0);
    }
    bool finite = true;
    for (float v : y) finite = finite && std::isfinite(v);
    std::printf("%s: the golden reference for one expert (all finite: %d)\n", finite ? "PASS" : "FAIL", (int) finite);
    return finite ? 0 : 1;
}
