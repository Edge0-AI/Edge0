// Minimal: is the fused A stage's 11x gap between the unit test and the engine a cache effect?
// Same kernel, same shape (64 groups x 83 entries), two buffer sizes: small (xq/out/hbuf fit in L2) and big
// (the engine's 42,510 entries: xq 61 MB, out 218 MB, hbuf 34 MB).  Only the buffer size changes.
#include "strata/prefill/lat_q3k_fused.hpp"

#include <cuda_runtime.h>
#include <ggml.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(e) do { const cudaError_t r_ = (e); if (r_ != cudaSuccess) { \
    std::printf("FAIL %s: %s\n", #e, cudaGetErrorString(r_)); std::exit(1); } } while (0)

int main() {
    constexpr int64_t ELL = 1280, FF = 640;
    constexpr int64_t GUB = 1280 * 550, DNB = 1280 * 360;      // one expert's blob
    constexpr int64_t BLOB = GUB + DNB;
    constexpr int kHq = (int) strata::prefill::lat_q3k_fused_hbuf_row;   // 800
    const int64_t gub = GUB, dnb = DNB, blob = BLOB;

    // 64 distinct real experts (74 MB, past L2), each with its own quantized weights
    std::mt19937 rng(11);
    std::normal_distribution<float> nw(0.f, 0.05f);
    std::vector<uint8_t> one((size_t) blob);
    {
        std::vector<float> w((size_t) 1280 * ELL);
        for (auto& v : w) v = nw(rng);
        if (ggml_quantize_chunk(GGML_TYPE_Q3_K, w.data(), one.data(), 0, 1280, ELL, nullptr) != gub) { std::printf("FAIL q3k\n"); return 1; }
        std::vector<float> wd((size_t) ELL * FF);
        for (auto& v : wd) v = nw(rng);
        if (ggml_quantize_chunk(GGML_TYPE_IQ4_NL, wd.data(), one.data() + gub, 0, ELL, FF, nullptr) != dnb) { std::printf("FAIL iq4\n"); return 1; }
    }
    std::vector<uint8_t> blobs((size_t) 64 * blob);
    for (int g = 0; g < 64; ++g) std::memcpy(blobs.data() + (size_t) g * blob, one.data(), blob);
    uint8_t* dblobs = nullptr; CK(cudaMalloc(&dblobs, blobs.size()));
    CK(cudaMemcpy(dblobs, blobs.data(), blobs.size(), cudaMemcpyHostToDevice));
    std::vector<unsigned long long> gp((size_t) 64);
    for (int g = 0; g < 64; ++g) gp[(size_t) g] = (unsigned long long) (dblobs + (size_t) g * blob);
    unsigned long long* dgp = nullptr; CK(cudaMalloc(&dgp, gp.size() * 8));
    CK(cudaMemcpy(dgp, gp.data(), gp.size() * 8, cudaMemcpyHostToDevice));

    // one valid q8_1 activation row (40 blocks x 36 B), replicated
    std::vector<float> x((size_t) ELL);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) v = nd(rng);
    struct H8 { uint16_t d, s; int8_t qs[32]; };
    std::vector<H8> row((size_t) ELL / 32);
    for (int b = 0; b < ELL / 32; ++b) {
        float amax = 0.f, sum = 0.f;
        for (int i = 0; i < 32; ++i) { const float v = x[(size_t) 32 * b + i]; amax = std::fmax(amax, std::fabs(v)); sum += v; }
        const float d = amax / 127.0f;
        row[(size_t) b].d = ggml_fp32_to_fp16(d);
        row[(size_t) b].s = ggml_fp32_to_fp16(sum);
        for (int i = 0; i < 32; ++i) row[(size_t) b].qs[i] = (int8_t) std::lround(d != 0.f ? x[(size_t) 32 * b + i] / d : 0.f);
    }

    int dev = 0, sms = 0;
    cudaGetDevice(&dev); cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);

    auto bench = [&](const char* tag, int nbig, bool spread) {
        // 64 groups x 83 entries inside a buffer of nbig entries
        std::vector<int32_t> starts((size_t) 65), cnt((size_t) 64), tok((size_t) nbig), ident((size_t) nbig);
        for (int g = 0; g < 64; ++g) starts[(size_t) g] = spread ? g * (nbig / 64) : g * 83;
        starts[64] = spread ? nbig : 64 * 83;
        for (int g = 0; g < 64; ++g) cnt[(size_t) g] = 83;
        const int ntok = nbig < 33960 ? nbig : 33960;
        for (int e = 0; e < nbig; ++e) { tok[(size_t) e] = (int32_t) ((e * 7) % ntok); ident[(size_t) e] = e; }
        int32_t* ds = nullptr; int32_t* dc = nullptr; int32_t* dt = nullptr; int32_t* di = nullptr;
        CK(cudaMalloc(&ds, 65 * 4)); CK(cudaMalloc(&dc, 64 * 4)); CK(cudaMalloc(&dt, (size_t) nbig * 4)); CK(cudaMalloc(&di, (size_t) nbig * 4));
        CK(cudaMemcpy(ds, starts.data(), 65 * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dc, cnt.data(), 64 * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dt, tok.data(), (size_t) nbig * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(di, ident.data(), (size_t) nbig * 4, cudaMemcpyHostToDevice));
        std::vector<H8> xq((size_t) nbig * (ELL / 32));
        for (int e = 0; e < nbig; ++e) std::memcpy(xq.data() + (size_t) e * (ELL / 32), row.data(), row.size() * sizeof(H8));
        uint8_t* dxq = nullptr; void* dhbuf = nullptr; float* dout = nullptr;
        CK(cudaMalloc(&dxq, xq.size() * sizeof(H8)));
        CK(cudaMalloc(&dhbuf, (size_t) nbig * kHq));
        CK(cudaMalloc(&dout, (size_t) nbig * ELL * 4));
        CK(cudaMemcpy(dxq, xq.data(), xq.size() * sizeof(H8), cudaMemcpyHostToDevice));
        auto run = [&] { strata::prefill::lat_q3k_fused((const uint8_t* const*) dgp, ds, 64, dxq, dhbuf, dout, ELL, nullptr, dc, dt, di); };
        run(); CK(cudaDeviceSynchronize());
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        const int it = 20;
        CK(cudaEventRecord(e0));
        for (int i = 0; i < it; ++i) run();
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
        const double entries = 64.0 * 83.0 * it;
        std::printf("%-28s nbig %6d  %.3f ms / %d iters  = %.4f us/entry  (xq %.1f MB, out %.1f MB)\n", tag, nbig,
                    ms / it, it, ms * 1000.0 / entries, (double) nbig * 1440 / 1e6, (double) nbig * ELL * 4 / 1e6);
        cudaFree(ds); cudaFree(dc); cudaFree(dt); cudaFree(di); cudaFree(dxq); cudaFree(dhbuf); cudaFree(dout);
    };

    // the engine's own group-length distribution (g-cnt.bin dumped from a 4K run), in batches of 64 groups
    {
        std::FILE* f = std::fopen("/home/user/lat-s26/q3k/cnt.bin", "rb");
        if (f != nullptr) {
            std::vector<int32_t> real((size_t) 512);
            const size_t got = std::fread(real.data(), 4, 512, f); std::fclose(f);
            if (got == 512) {
                int64_t tot = 0; int32_t mx = 0;
                for (int32_t c : real) { tot += c; mx = std::max(mx, c); }
                std::printf("engine cnt: 512 groups, sum %lld, max %d\n", (long long) tot, mx);
                std::vector<int32_t> starts((size_t) 513);
                starts[0] = 0;
                for (int g = 0; g < 512; ++g) starts[(size_t) g + 1] = starts[(size_t) g] + real[(size_t) g];
                const int nbig = (int) starts[512];
                // buffers sized for the whole layer
                std::vector<int32_t> tok((size_t) nbig), ident((size_t) nbig);
                const int ntok = nbig < 33960 ? nbig : 33960;
                for (int e = 0; e < nbig; ++e) { tok[(size_t) e] = (int32_t) ((e * 7) % ntok); ident[(size_t) e] = e; }
                int32_t* ds = nullptr; int32_t* dc = nullptr; int32_t* dt = nullptr; int32_t* di = nullptr;
                CK(cudaMalloc(&ds, 513 * 4)); CK(cudaMalloc(&dc, 512 * 4));
                CK(cudaMalloc(&dt, (size_t) nbig * 4)); CK(cudaMalloc(&di, (size_t) nbig * 4));
                CK(cudaMemcpy(ds, starts.data(), 513 * 4, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(dc, real.data(), 512 * 4, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(dt, tok.data(), (size_t) nbig * 4, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(di, ident.data(), (size_t) nbig * 4, cudaMemcpyHostToDevice));
                std::vector<H8> xq((size_t) nbig * (ELL / 32));
                for (int e = 0; e < nbig; ++e) std::memcpy(xq.data() + (size_t) e * (ELL / 32), row.data(), row.size() * sizeof(H8));
                uint8_t* dxq = nullptr; void* dhbuf = nullptr; float* dout = nullptr;
                CK(cudaMalloc(&dxq, xq.size() * sizeof(H8))); CK(cudaMalloc(&dhbuf, (size_t) nbig * kHq));
                CK(cudaMalloc(&dout, (size_t) nbig * ELL * 4));
                CK(cudaMemcpy(dxq, xq.data(), xq.size() * sizeof(H8), cudaMemcpyHostToDevice));
                auto run_all = [&] {
                    for (int q0 = 0; q0 < 512; q0 += 64) {
                        const int nb = std::min(64, 512 - q0);
                        strata::prefill::lat_q3k_fused((const uint8_t* const*) dgp, ds + q0, nb, dxq, dhbuf, dout, ELL,
                                                       nullptr, dc + q0, dt, di);
                    }
                };
                run_all(); CK(cudaDeviceSynchronize());
                cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
                const int it = 20;
                CK(cudaEventRecord(e0));
                for (int i = 0; i < it; ++i) run_all();
                CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
                float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
                std::printf("%-28s nbig %6d  %.3f ms / %d iters  = %.4f us/entry  (engine cnt, 8 batches of 64)\n",
                            "engine cnt distribution", nbig, ms / it, it, ms * 1000.0 / ((double) tot * it));
                cudaFree(ds); cudaFree(dc); cudaFree(dt); cudaFree(di); cudaFree(dxq); cudaFree(dhbuf); cudaFree(dout);
            }
        }
    }
    // the same engine distribution, flattened into pseudo-groups of <= 64 rows (host-side tile split, no kernel change)
    {
        std::FILE* f = std::fopen("/home/user/lat-s26/q3k/cnt.bin", "rb");
        if (f != nullptr) {
            std::vector<int32_t> real((size_t) 512);
            if (std::fread(real.data(), 4, 512, f) == 512) {
                std::fclose(f);
                std::vector<int32_t> ps, pc; std::vector<int> pg;
                int64_t tot = 0; int32_t base = 0;
                for (int g = 0; g < 512; ++g) {
                    for (int32_t off = 0; off < real[(size_t) g]; off += 64) {
                        ps.push_back(base + off);
                        pc.push_back(std::min<int32_t>(64, real[(size_t) g] - off));
                        pg.push_back(g);
                        tot += pc.back();
                    }
                    base += real[(size_t) g];
                }
                const int np = (int) ps.size(), nbig = (int) base;
                std::vector<unsigned long long> pgp((size_t) np);
                for (int i = 0; i < np; ++i) pgp[(size_t) i] = gp[(size_t) (pg[(size_t) i] % 64)];
                unsigned long long* dpgp = nullptr;
                int32_t* ds = nullptr; int32_t* dc = nullptr; int32_t* dt = nullptr; int32_t* di = nullptr;
                std::vector<int32_t> tok((size_t) nbig), ident((size_t) nbig);
                const int ntok = nbig < 33960 ? nbig : 33960;
                for (int e = 0; e < nbig; ++e) { tok[(size_t) e] = (int32_t) ((e * 7) % ntok); ident[(size_t) e] = e; }
                CK(cudaMalloc(&dpgp, (size_t) np * 8)); CK(cudaMalloc(&ds, (size_t) np * 4)); CK(cudaMalloc(&dc, (size_t) np * 4));
                CK(cudaMalloc(&dt, (size_t) nbig * 4)); CK(cudaMalloc(&di, (size_t) nbig * 4));
                CK(cudaMemcpy(dpgp, pgp.data(), (size_t) np * 8, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(ds, ps.data(), (size_t) np * 4, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(dc, pc.data(), (size_t) np * 4, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(dt, tok.data(), (size_t) nbig * 4, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(di, ident.data(), (size_t) nbig * 4, cudaMemcpyHostToDevice));
                std::vector<H8> xq((size_t) nbig * (ELL / 32));
                for (int e = 0; e < nbig; ++e) std::memcpy(xq.data() + (size_t) e * (ELL / 32), row.data(), row.size() * sizeof(H8));
                uint8_t* dxq = nullptr; void* dhbuf = nullptr; float* dout = nullptr;
                CK(cudaMalloc(&dxq, xq.size() * sizeof(H8))); CK(cudaMalloc(&dhbuf, (size_t) nbig * kHq));
                CK(cudaMalloc(&dout, (size_t) nbig * ELL * 4));
                CK(cudaMemcpy(dxq, xq.data(), xq.size() * sizeof(H8), cudaMemcpyHostToDevice));
                for (int pb : {64, 240, 480}) {
                    auto run_all = [&] {
                        for (int q0 = 0; q0 < np; q0 += pb) {
                            const int nb = std::min(pb, np - q0);
                            strata::prefill::lat_q3k_fused((const uint8_t* const*) (dpgp + q0), ds + q0, nb, dxq, dhbuf,
                                                           dout, ELL, nullptr, dc + q0, dt, di);
                        }
                    };
                    run_all(); CK(cudaDeviceSynchronize());
                    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
                    const int it = 20;
                    CK(cudaEventRecord(e0));
                    for (int i = 0; i < it; ++i) run_all();
                    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
                    float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
                    std::printf("%-28s tiles %5d  batch %3d  %.3f ms = %.4f us/entry\n", "engine cnt, tile-flattened", np, pb,
                                ms / it, ms * 1000.0 / ((double) tot * it));
                    cudaEventDestroy(e0); cudaEventDestroy(e1);
                }
                cudaFree(dpgp); cudaFree(ds); cudaFree(dc); cudaFree(dt); cudaFree(di); cudaFree(dxq); cudaFree(dhbuf); cudaFree(dout);
            }
        }
    }
    bench("small (compact, fits L2)", 64 * 83, false);
    bench("big (engine spread)", 42510, true);
    bench("big (compact, same size)", 42510, false);
    std::printf("%s\n", "PASS: cache test");
    return 0;
}
