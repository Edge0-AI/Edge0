// src/kernels/latent_a_u_check.cpp - parity: strata::latent::a_u (A+U from xl) vs strata::latent::run_layer (D+A+U).
//
// run_layer is the already-validated D->A->U reference (latent_module_verify checks it against a Python float
// reference). a_u is the new A+U-only slice a peer/helper will call once the D rows are shared. This test runs
// run_layer on a synthetic layer, reuses its q8_1 xl (b.xlq), feeds that to a_u, and compares the two H-wide row
// sets. Both share the A kernel (native_expert_grouped), so the only numerical difference is the U kernel
// (latent_u_grouped vs run_layer's iq_mmvq gather): the parity gap must stay tiny (relL2 < 2e-3).
//
// usage: STRATA_LATENT_DENSE_FMT=<fmt> latent_a_u_check <dense.bin> <experts.bin> <ELL> <blob_bytes>
//   e.g.  STRATA_LATENT_DENSE_FMT=8 latent_a_u_check latent_dense_q8.bin experts.bin 1792 1935376
#include "strata/latent/latent_moe.hpp"
#include "strata/core/latent_dense.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}
template <typename T> void dev(T*& p, size_t bytes) { ck(cudaMalloc((void**) &p, bytes), "cudaMalloc"); }
double rel_l2(const float* a, const float* b, int n) {
    double num = 0, den = 0;
    for (int i = 0; i < n; ++i) { const double d = (double) a[i] - b[i]; num += d * d; den += (double) b[i] * b[i]; }
    return den > 0 ? std::sqrt(num / den) : (num > 0 ? 1e30 : 0.0);
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: latent_a_u_check <dense.bin> <experts.bin> <ELL> <blob_bytes>\n");
        return 2;
    }
    const std::string dense_path = argv[1], pack_path = argv[2];
    const int ELL = std::atoi(argv[3]);
    const int64_t blob_bytes = std::atoll(argv[4]);
    const char* fmt_env = std::getenv("STRATA_LATENT_DENSE_FMT");
    const int fmt = fmt_env && *fmt_env ? std::atoi(fmt_env) : 8;
    if (fmt != 4 && fmt != 8) { std::fprintf(stderr, "STRATA_LATENT_DENSE_FMT must be 4 or 8\n"); return 2; }

    strata::latent::Geometry g;
    g.H = 2560; g.ELL = ELL; g.I = 640; g.G = 8; g.E = 10; g.K = 10;   // E=10: experts 0..9 only
    const int T = 1, cap = T * g.K;

    strata::core::LatentDense dense;
    std::string err;
    if (!dense.open(dense_path, 48, 8, g.H, g.ELL, err)) { std::fprintf(stderr, "dense open: %s\n", err.c_str()); return 2; }
    strata::core::LatentA pack;
    if (!pack.open(pack_path, 48, 512, blob_bytes, err)) { std::fprintf(stderr, "pack open: %s\n", err.c_str()); return 2; }

    std::vector<int32_t> labels((size_t) g.E);
    for (size_t e = 0; e < labels.size(); ++e) labels[e] = (int32_t) (e % g.G);
    std::vector<int32_t> topk((size_t) cap);
    for (int i = 0; i < cap; ++i) topk[(size_t) i] = i;   // experts 0..9 in route order

    auto plan = strata::latent::build_plan(g, topk.data(), labels.data(), T);

    std::vector<float> x((size_t) T * g.H);
    for (size_t i = 0; i < x.size(); ++i) x[i] = ((i * 1103515245u + 12345u) % 1000u) / 1000.0f - 0.5f;

    // upload host state
    float *d_x = nullptr, *d_d = nullptr; uint8_t* d_u = nullptr; int32_t *d_labels = nullptr, *d_topk = nullptr;
    unsigned long long* d_ptr = nullptr; int32_t *d_start = nullptr, *d_dst = nullptr, *d_tok = nullptr, *d_n = nullptr;
    dev(d_x, (size_t) T * g.H * 4);
    dev(d_d, dense.d_bytes());
    dev(d_u, dense.u_bytes());
    dev(d_labels, (size_t) g.E * 4);
    dev(d_topk, (size_t) cap * 4);
    dev(d_ptr, (size_t) g.E * sizeof(unsigned long long));
    dev(d_start, plan.start.size() * 4);
    dev(d_dst, plan.dst.size() * 4);
    dev(d_tok, plan.tok.size() * 4);
    dev(d_n, 4);
    ck(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(d_d, dense.d(0), dense.d_bytes(), cudaMemcpyHostToDevice), "D");
    ck(cudaMemcpy(d_u, dense.u(0), dense.u_bytes(), cudaMemcpyHostToDevice), "U");
    ck(cudaMemcpy(d_labels, labels.data(), labels.size() * 4, cudaMemcpyHostToDevice), "labels");
    ck(cudaMemcpy(d_topk, topk.data(), topk.size() * 4, cudaMemcpyHostToDevice), "topk");
    for (int e = 0; e < g.E; ++e) {
        uint8_t* b = nullptr;
        dev(b, (size_t) blob_bytes);
        ck(cudaMemcpy(b, pack.blob(0, e), (size_t) blob_bytes, cudaMemcpyHostToDevice), "blob");
        ck(cudaMemcpy(d_ptr + e, &b, sizeof(b), cudaMemcpyHostToDevice), "blob ptr");
    }
    ck(cudaMemcpy(d_start, plan.start.data(), plan.start.size() * 4, cudaMemcpyHostToDevice), "start");
    ck(cudaMemcpy(d_dst, plan.dst.data(), plan.dst.size() * 4, cudaMemcpyHostToDevice), "dst");
    ck(cudaMemcpy(d_tok, plan.tok.data(), plan.tok.size() * 4, cudaMemcpyHostToDevice), "tok");
    int32_t n = plan.cap; ck(cudaMemcpy(d_n, &n, 4, cudaMemcpyHostToDevice), "n");

    // run_layer reference buffers
    strata::latent::Inputs in;
    in.d_q8 = (const uint8_t*) d_d; in.u_q8 = d_u; in.x = d_x; in.topk = topk.data(); in.labels = labels.data();
    in.T = T; in.fmt = fmt;
    strata::latent::DevicePlan dp;
    dp.ptr = d_ptr; dp.start = d_start; dp.n = d_n; dp.dst = d_dst; dp.tok = d_tok; dp.cap = cap; dp.groups = g.E;
    strata::latent::Buffers b;
    dev(b.xq, (size_t) T * (g.H / 32) * 36);
    dev(b.xlq, (size_t) T * g.G * (ELL / 32) * 36);
    dev(b.uq, (size_t) cap * (ELL / 32) * 36);
    dev(b.scratch, strata::kernels::native_expert_scratch_bytes(cap, g.I));
    dev(b.xl, (size_t) T * g.G * ELL * 4);
    dev(b.yl, (size_t) cap * ELL * 4);
    dev(b.uin, (size_t) cap * ELL * 4);
    dev(b.uout, (size_t) cap * g.H * 4);
    dev(b.y, (size_t) cap * g.H * 4);
    if (!strata::latent::run_layer(g, in, dp, b, 0)) { std::fprintf(stderr, "run_layer failed\n"); return 2; }

    // a_u: same xlq, same plan; separate yl/scratch/uq/out
    strata::latent::AUBuffers au;
    au.xlq = b.xlq; au.ptr = d_ptr; au.start = d_start; au.n = d_n; au.dst = d_dst; au.tok = d_tok;
    au.ids = d_topk; au.labels = d_labels; au.u = d_u;
    au.gu_type = 40; au.d_type = 40;
    au.ELL = ELL; au.I = g.I; au.G = g.G; au.E = g.E; au.H = g.H; au.fmt = fmt;
    au.cap_entries = cap; au.cap_groups = g.E;
    dev(au.yl, (size_t) cap * ELL * 4);
    dev(au.scratch, strata::kernels::native_expert_scratch_bytes(cap, g.I));
    dev(au.uq, (size_t) cap * (ELL / 32) * 36);
    dev(au.out, (size_t) cap * g.H * 4);
    if (!strata::latent::a_u(au, 0)) { std::fprintf(stderr, "a_u failed\n"); return 2; }
    ck(cudaDeviceSynchronize(), "sync");

    std::vector<float> ref((size_t) cap * g.H), got((size_t) cap * g.H);
    ck(cudaMemcpy(ref.data(), b.y, ref.size() * 4, cudaMemcpyDeviceToHost), "y D2H");
    ck(cudaMemcpy(got.data(), au.out, got.size() * 4, cudaMemcpyDeviceToHost), "out D2H");
    const double rl = rel_l2(got.data(), ref.data(), (int) got.size());
    std::printf("ELL=%d fmt=%d relL2=%.3e (threshold 2e-3)\n", ELL, fmt, rl);
    if (rl >= 2e-3) { std::fprintf(stderr, "FAIL: parity gap too large\n"); return 1; }
    std::printf("PASS: a_u matches run_layer\n");
    return 0;
}