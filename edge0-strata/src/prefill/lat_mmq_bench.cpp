// src/prefill/lat_mmq_bench.cpp - the latent A stage at the engine's own shapes and group sizes: gate/up Q3_K
// [1280 x 1280], down IQ4_NL [1280 x 640], ELL = 1280, n experts a launch, R rows each.  Every stage timed alone.
//
//     lat_mmq_bench [R ...]        default R = 95 (a 4K chunk: 42450 entries / 512 experts / 15.6 a group)
//
// The gather is the engine's per-expert `gather_native` into the MMQ group buffers, with a plain D2D memcpy of the
// same bytes as the reference: the question is whether that copy or the products are the A stage's cost.
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace mmq = strata::prefill::mmq;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

std::vector<uint8_t> weights(ggml_type t, int64_t rows, int64_t cols, int experts, std::mt19937& rng) {
    const size_t per = mmq::matrix_bytes(t, rows, cols), bs = ggml_type_size(t);
    std::vector<uint8_t> w((size_t) experts * per);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& b : w) b = (uint8_t) byte(rng);
    for (size_t i = 0; i < w.size() / bs; ++i) {
        const uint16_t d = (uint16_t) (0x2800 + byte(rng) % 0x800);
        std::memcpy(w.data() + i * bs + (t == GGML_TYPE_Q3_K ? bs - 2 : 0), &d, 2);
    }
    return w;
}

cudaStream_t g_s = nullptr;   // the engine's compute stream is not the default one

float med(void (*fire)(void*), void* arg) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    std::vector<float> ts;
    for (int i = 0; i < 26; ++i) {
        cudaEventRecord(a, g_s);
        fire(arg);
        cudaEventRecord(b, g_s);
        ck(cudaEventSynchronize(b), "sync");
        float ms = 0; cudaEventElapsedTime(&ms, a, b);
        if (i >= 3) ts.push_back(ms);
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    std::sort(ts.begin(), ts.end());
    return ts[ts.size() / 2];
}

const int64_t ELL = 1280, FF = 640;
const size_t GU_B = 1280 * 550;          // Q3_K [1280, 1280]: 5 blocks of 256 x 110 B
const size_t DN_B = 1280 * 20 * 18;      // IQ4_NL [1280, 640]: 20 blocks of 32 x 18 B

struct Ctx {
    int n = 16;
    int64_t R = 95;
    std::vector<uint8_t> wgu, wdn;
    std::vector<float> xl;
    std::vector<int32_t> bounds;
    void *dgu = nullptr, *ddn = nullptr, *dxl = nullptr, *dxq = nullptr, *dgo = nullptr, *dho = nullptr;
    void *dhq = nullptr, *ddo = nullptr, *db = nullptr, *dids = nullptr, *ggu = nullptr, *gdn = nullptr;
    mmq::Context ctx;
    mmq::Product gu, dn;
    mmq::Product g2s, d2s;   // the engine's geometry, timed
};

void fire_gather(void* a) {   // one launch per expert, as the engine does it
    Ctx* c = (Ctx*) a;
    for (int q = 0; q < c->n; ++q)
        mmq::gather_native(c->ddn, (const uint8_t*) c->ddn + DN_B + GU_B / 2, GU_B / 2,
                           (const uint8_t*) c->ddn + DN_B + GU_B, DN_B,
                           (uint8_t*) c->ggu + (size_t) q * GU_B, (uint8_t*) c->gdn + (size_t) q * DN_B, g_s);
}
void fire_gather_group(void* a) {   // one launch for the whole group
    Ctx* c = (Ctx*) a;
    mmq::GatherGroup g;
    g.first = 0;
    g.n = std::min(c->n, mmq::kGatherGroupMax);
    for (int q = 0; q < g.n; ++q) g.blob[q] = (const uint8_t*) c->ddn;
    mmq::gather_native_group(g, DN_B + GU_B / 2, GU_B / 2, DN_B + GU_B, DN_B, c->ggu, GU_B, c->gdn, DN_B, g_s);
}
void fire_memcpy(void* a) {
    Ctx* c = (Ctx*) a;
    for (int q = 0; q < c->n; ++q) {
        ck(cudaMemcpyAsync((uint8_t*) c->ggu + (size_t) q * GU_B, c->ddn, GU_B, cudaMemcpyDeviceToDevice, nullptr), "cp");
        ck(cudaMemcpyAsync((uint8_t*) c->gdn + (size_t) q * DN_B, c->ddn, DN_B, cudaMemcpyDeviceToDevice, nullptr), "cp2");
    }
}
void fire_quant(void* a) {
    Ctx* c = (Ctx*) a;
    mmq::quantize(c->xl.data(), nullptr, c->dxq, GGML_TYPE_Q3_K, ELL, ELL, (int64_t) c->n * c->R, nullptr);
}
void fire_gu(void* a) { Ctx* c = (Ctx*) a; c->ctx.run(c->gu, g_s); }
void fire_swiglu(void* a) {
    Ctx* c = (Ctx*) a;
    mmq::swiglu((const float*) c->dgo, (float*) c->dho, (int64_t) c->n * c->R, 640, false, nullptr);
}
void fire_quanth(void* a) {
    Ctx* c = (Ctx*) a;
    mmq::quantize((const float*) c->dho, nullptr, c->dhq, GGML_TYPE_IQ4_NL, 640, 640, (int64_t) c->n * c->R, nullptr);
}
void fire_dn(void* a) { Ctx* c = (Ctx*) a; c->ctx.run(c->dn, g_s); }

}  // namespace

int main(int argc, char** argv) {
    std::vector<int64_t> Rs;
    for (int i = 1; i < argc; ++i) Rs.push_back(std::atoll(argv[i]));
    if (Rs.empty()) Rs = {95};
    std::printf("lat_mmq_bench: gu Q3_K [1280 x 1280] (%.0f KB), dn IQ4_NL [1280 x 640] (%.0f KB), medians of 23\n",
                GU_B / 1024.0, DN_B / 1024.0);
    ck(cudaStreamCreate(&g_s), "stream");
    std::mt19937 rng(7);
    for (int64_t R : Rs)
        for (int n : {1, 4, 16}) {
            Ctx c;
            c.n = n; c.R = R;
            const int64_t T = (int64_t) n * R;
            c.wgu = weights(GGML_TYPE_Q3_K, 1280, ELL, n, rng);
            c.wdn = weights(GGML_TYPE_IQ4_NL, ELL, FF, n, rng);
            std::normal_distribution<float> nd(0.f, 1.f);
            c.xl.resize((size_t) T * ELL);
            for (auto& v : c.xl) v = nd(rng);
            c.bounds.resize((size_t) n + 1);
            for (int e = 0; e <= n; ++e) c.bounds[(size_t) e] = (int32_t) (e * R);
            auto alloc = [](void** p, size_t bytes) { ck(cudaMalloc(p, bytes ? bytes : 16), "alloc"); };
            alloc(&c.dgu, c.wgu.size());
            alloc(&c.ddn, c.wdn.size() + GU_B + DN_B);   // room for the "expert blob" the gather reads
            alloc(&c.ggu, (size_t) n * GU_B);
            alloc(&c.gdn, (size_t) n * DN_B);
            alloc(&c.dxl, c.xl.size() * 4);
            alloc(&c.dxq, mmq::q8_bytes(T, ELL));
            alloc(&c.dgo, (size_t) T * 1280 * 4);
            alloc(&c.dho, (size_t) T * 640 * 4);
            alloc(&c.dhq, mmq::q8_bytes(T, 640));
            alloc(&c.ddo, (size_t) T * ELL * 4);
            alloc(&c.db, ((size_t) n + 1) * 4);
            alloc(&c.dids, (size_t) (T + 1024) * 4);
            ck(cudaMemcpy(c.dgu, c.wgu.data(), c.wgu.size(), cudaMemcpyHostToDevice), "w");
            ck(cudaMemcpy(c.ddn, c.wdn.data(), c.wdn.size(), cudaMemcpyHostToDevice), "w2");
            ck(cudaMemcpy(c.dxl, c.xl.data(), c.xl.size() * 4, cudaMemcpyHostToDevice), "x");
            ck(cudaMemcpy(c.db, c.bounds.data(), ((size_t) n + 1) * 4, cudaMemcpyHostToDevice), "b");
            mmq::iota((int32_t*) c.dids, T + 1024, nullptr);
            mmq::quantize(c.xl.data(), nullptr, c.dxq, GGML_TYPE_Q3_K, ELL, ELL, T, nullptr);
            mmq::swiglu((const float*) c.dgo, (float*) c.dho, T, 640, false, nullptr);
            mmq::quantize((const float*) c.dho, nullptr, c.dhq, GGML_TYPE_IQ4_NL, 640, 640, T, nullptr);
            c.gu.w = c.dgu; c.gu.type = GGML_TYPE_Q3_K; c.gu.w_rows = 1280; c.gu.w_cols = ELL;
            c.gu.expert_bytes = GU_B; c.gu.n = n; c.gu.xq = c.dxq; c.gu.bounds = (const int32_t*) c.db;
            c.gu.ids = (const int32_t*) c.dids; c.gu.total_rows = T; c.gu.max_rows = R;
            c.gu.dst = (float*) c.dgo; c.gu.ld_dst = 1280;
            c.dn.w = c.ddn; c.dn.type = GGML_TYPE_IQ4_NL; c.dn.w_rows = ELL; c.dn.w_cols = 640;
            c.dn.expert_bytes = DN_B; c.dn.n = n; c.dn.xq = c.dhq; c.dn.bounds = (const int32_t*) c.db;
            c.dn.ids = (const int32_t*) c.dids; c.dn.total_rows = T; c.dn.max_rows = R;
            c.dn.dst = (float*) c.ddo; c.dn.ld_dst = ELL;
            if (n == 16) {   // the engine's own layer-0 group: uneven bounds, maxr 370, total_rows 1278
                const int32_t ub[17] = {0, 266, 267, 637, 827, 828, 1019, 1206, 1207, 1212, 1213, 1216, 1218, 1219,
                                        1263, 1277, 1278};
                ck(cudaMemcpy(c.db, ub, 17 * 4, cudaMemcpyHostToDevice), "ub");
                const int64_t UN = 1278;
                ck(cudaStreamSynchronize(g_s), "pre");
                mmq::Product g2 = c.gu;
                g2.total_rows = UN; g2.max_rows = 370;
                c.ctx.run(g2, g_s);
                std::printf("  UNEVEN gu n=16 nr=%lld maxr=370: %s\n", (long long) UN,
                            cudaGetErrorString(cudaStreamSynchronize(g_s)));
                mmq::Product d2 = c.dn;
                d2.total_rows = UN; d2.max_rows = 370;
                c.ctx.run(d2, g_s);
                c.g2s = g2; c.d2s = d2;
                {
                    auto tg = [](void* q) { Ctx* x = (Ctx*) q; x->ctx.run(x->g2s, g_s); };
                    auto td = [](void* q) { Ctx* x = (Ctx*) q; x->ctx.run(x->d2s, g_s); };
                    const float tg2 = med(tg, &c), td2 = med(td, &c);
                    std::printf("  UNEVEN TIMED n=16 nr=1278 maxr=370: gu %7.1f us | dn %7.1f us  (dn/gu %.1fx)\n",
                                tg2 * 1000, td2 * 1000, tg2 > 0 ? td2 / tg2 : 0.0);
                }
                std::printf("  UNEVEN dn n=16 nr=%lld maxr=370: %s\n", (long long) UN,
                            cudaGetErrorString(cudaStreamSynchronize(g_s)));
                ck(cudaMemcpy(c.db, c.bounds.data(), ((size_t) n + 1) * 4, cudaMemcpyHostToDevice), "ub2");
            }
            const double mb = (double) n * (GU_B + DN_B) / 1e6;
            const float tg = med(fire_gather, &c), tgg = med(fire_gather_group, &c), tc = med(fire_memcpy, &c);
            const float tq = med(fire_quant, &c), tgu = med(fire_gu, &c), ts = med(fire_swiglu, &c);
            const float th = med(fire_quanth, &c), td = med(fire_dn, &c);
            std::printf("  R %3lld n %2d rows %5lld | gather %8.1f us (%5.0f GB/s per-exp) | gather_group %7.1f us"
                        " (%5.0f GB/s) | memcpy %7.1f us (%5.0f GB/s) [%.1f MB]\n",
                        (long long) R, n, (long long) T, tg * 1000, mb / (tg * 1e3) / 1e3, tgg * 1000,
                        mb / (tgg * 1e3) / 1e3, tc * 1000, mb / (tc * 1e3) / 1e3, mb);
            std::printf("                    | quantize %6.1f | gu %7.1f | swiglu %5.1f | quantH %5.1f | dn %7.1f us"
                        "  (sum %.1f us)\n", tq * 1000, tgu * 1000, ts * 1000, th * 1000, td * 1000,
                        (tq + tgu + ts + th + td) * 1000);
            for (void* p : {c.dgu, c.ddn, c.ggu, c.gdn, c.dxl, c.dxq, c.dgo, c.dho, c.dhq, c.ddo, c.db, c.dids})
                cudaFree(p);
        }
    return 0;
}
