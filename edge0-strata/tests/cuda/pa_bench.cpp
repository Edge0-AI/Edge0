// pa_bench - the prompt path's sparse attention (qsa_prompt_attn_batch) on a recorded selection: a STRATA_QSA_DUMP
// record of one QSA layer and chunk (int32 {layer, p0, T, cap}, T * cap cells, then T * kStepCount steps, written
// with STRATA_QSA_DUMP_STEPS=1), random q and int8 K/V (the values do not change the work), one page per 4 cells.
// usage: pa_bench DUMP [reps]   - prints the median time and a hash of the output (for A/B of kernel variants)
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); return 1; } } while (0)

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: pa_bench DUMP [reps]\n"); return 2; }
    const int reps = argc > 2 ? std::atoi(argv[2]) : 5;
    std::FILE* f = std::fopen(argv[1], "rb");
    if (!f) { std::perror(argv[1]); return 1; }
    int32_t hdr[4];
    if (std::fread(hdr, 4, 4, f) != 4) return 1;
    const int64_t p0 = hdr[1], T = hdr[2], cap = hdr[3];
    std::vector<int32_t> ids((size_t) (T * cap)), steps((size_t) (T * strata::kernels::kStepCount));
    if (std::fread(ids.data(), 4, ids.size(), f) != ids.size() || std::fread(steps.data(), 4, steps.size(), f) != steps.size()) {
        std::fprintf(stderr, "short dump (written without STRATA_QSA_DUMP_STEPS?)\n");
        return 1;
    }
    std::fclose(f);
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    const int64_t cells = p0 + T, HD = s.head_dim, KVH = s.n_head_kv, NH = s.n_head;
    const int64_t pages = (cells + s.page_size - 1) / s.page_size;
    std::printf("layer %d p0 %lld T %lld cap %lld, %lld cells, %lld heads / %lld KV heads x %lld\n", hdr[0], (long long) p0,
                (long long) T, (long long) cap, (long long) cells, (long long) NH, (long long) KVH, (long long) HD);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> u8(-127, 127);
    std::vector<float> q((size_t) (T * NH * HD));
    for (auto& x : q) x = nd(rng);
    const size_t kv_n = (size_t) (pages * KVH * s.page_size * HD);
    std::vector<int8_t> kq(kv_n), vq(kv_n);
    for (auto& x : kq) x = (int8_t) u8(rng);
    for (auto& x : vq) x = (int8_t) u8(rng);
    const size_t sc_n = (size_t) (pages * KVH * s.page_size * (HD / 64));
    std::vector<uint16_t> ks(sc_n), vs(sc_n);
    for (size_t i = 0; i < sc_n; ++i) { ks[i] = 0x2000 + (uint16_t) (i % 512); vs[i] = 0x2000 + (uint16_t) ((i * 7) % 512); }
    std::vector<int32_t> pt((size_t) pages);
    for (int64_t i = 0; i < pages; ++i) pt[(size_t) i] = (int32_t) i;

    float *dq, *dattn;
    int8_t *dk, *dv;
    uint16_t *dks, *dvs;
    int32_t *dids, *dsteps, *dpt;
    CK(cudaMalloc(&dq, q.size() * 4));
    CK(cudaMalloc(&dattn, (size_t) (T * NH * HD) * 4));
    CK(cudaMalloc(&dk, kv_n));
    CK(cudaMalloc(&dv, kv_n));
    CK(cudaMalloc(&dks, sc_n * 2));
    CK(cudaMalloc(&dvs, sc_n * 2));
    CK(cudaMalloc(&dids, ids.size() * 4));
    CK(cudaMalloc(&dsteps, steps.size() * 4));
    CK(cudaMalloc(&dpt, pt.size() * 4));
    CK(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dk, kq.data(), kv_n, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dv, vq.data(), kv_n, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dks, ks.data(), sc_n * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dvs, vs.data(), sc_n * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsteps, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dpt, pt.data(), pt.size() * 4, cudaMemcpyHostToDevice));
    strata::kernels::QsaAttnPools pools;
    pools.k_q = dk; pools.v_q = dv; pools.k_scale = dks; pools.v_scale = dvs; pools.page_table = dpt;
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    std::vector<float> ms;
    for (int r = 0; r < reps + 1; ++r) {
        CK(cudaEventRecord(e0, st));
        if (!strata::kernels::qsa_prompt_attn_batch(dq, pools, dids, dsteps, cap, s, dattn, T, st)) {
            std::fprintf(stderr, "qsa_prompt_attn_batch declined\n");
            return 1;
        }
        CK(cudaEventRecord(e1, st));
        CK(cudaEventSynchronize(e1));
        float t = 0;
        CK(cudaEventElapsedTime(&t, e0, e1));
        if (r > 0) ms.push_back(t);
    }
    std::sort(ms.begin(), ms.end());
    std::vector<float> out((size_t) (T * NH * HD));
    CK(cudaMemcpy(out.data(), dattn, out.size() * 4, cudaMemcpyDeviceToHost));
    uint64_t h = 1469598103934665603ull;
    for (float x : out) { uint32_t b; std::memcpy(&b, &x, 4); h = (h ^ b) * 1099511628211ull; }
    double ss = 0;
    for (float x : out) ss += (double) x * x;
    std::printf("prompt attention: median %.3f ms (min %.3f) over %d; output hash %016llx, rms %.6g\n", ms[ms.size() / 2],
                ms[0], reps, (unsigned long long) h, std::sqrt(ss / (double) out.size()));
    if (const char* rf = std::getenv("PA_BENCH_REF")) {   // another variant's PA_BENCH_OUT: relative RMS and worst difference
        std::vector<float> ref(out.size());
        if (std::FILE* g = std::fopen(rf, "rb")) {
            const size_t got = std::fread(ref.data(), 4, ref.size(), g);
            std::fclose(g);
            if (got == ref.size()) {
                double e2 = 0, r2 = 0, worst = 0;
                for (size_t i = 0; i < out.size(); ++i) {
                    const double d = (double) out[i] - ref[i];
                    e2 += d * d; r2 += (double) ref[i] * ref[i];
                    worst = std::max(worst, std::fabs(d));
                }
                std::printf("against %s: relative RMS %.3e, worst |diff| %.3e (reference rms %.4g)\n", rf, std::sqrt(e2 / std::max(r2, 1e-30)),
                            worst, std::sqrt(r2 / (double) out.size()));
            } else std::printf("PA_BENCH_REF %s: short file\n", rf);
        }
    }
    if (const char* o = std::getenv("PA_BENCH_OUT")) {   // the output, for an accuracy comparison of two variants
        if (std::FILE* g = std::fopen(o, "wb")) { std::fwrite(out.data(), 4, out.size(), g); std::fclose(g); }
    }
    return 0;
}
