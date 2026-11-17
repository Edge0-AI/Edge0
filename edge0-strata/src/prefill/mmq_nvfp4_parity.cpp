// src/prefill/mmq_nvfp4_parity.cpp - the prompt path's NVFP4 expert products (MMQ) against an exact reference.
//
//     STRATA_PREFILL_NVFP4=w4a8|w4a4 mmq_nvfp4_parity pack/experts.bin [rows ...]
//     STRATA_PREFILL_NVFP4=w4a8|w4a4 mmq_nvfp4_parity pack/experts.bin --group
//     STRATA_PREFILL_NVFP4=w4a8|w4a4 mmq_nvfp4_parity pack/experts.bin --layers [expert]
//     STRATA_PREFILL_NVFP4=w4a8|w4a4 mmq_nvfp4_parity pack/experts.bin --real moe_input.bin layer
//
// --group: the prompt path's launch shape - 20 experts of layer 0 with 0..60 rows each, one activation array in
// expert order, gate/up in two launches (experts 0-15, then 16-19 through the same absolute bounds, 16 in), and the
// second group's down from its own rows through relative bounds.
// --layers: one expert (default 7) of every one of the 48 layers, 64 rows - scales differ by layer.
// --real: a layer's MoE input from a prompt (STRATA_DUMP_MOE_INPUT) through its 12 most-routed experts, gate/up on
// the real rows and down on the hidden they produce; also an FP16-activation product (the fp16 path) and the rows'
// outlier ratio (max |x| / rms x).
//
// One real expert (layer 0, expert 0): gate/up [1280, 2560] and down [2560, 640], as the prompt path gathers them.
// Activations are random rows; the reference dequantizes the weights with ggml's own NVFP4 decode and sums in double.
// Reported per product: relative Frobenius error and the worst element relative to the row's largest output.
// 8-bit activations land near 0.5%; FP4 activations near 10% - the difference between the two modes.
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>
#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <random>
#include <vector>

namespace mmq = strata::prefill::mmq;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

// one product: x [T, cols] times w [rows, cols]^T
float half_round(float v) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtps_ph(_mm_set_ss(v), 0))); }

int product(const uint8_t* w_host, int64_t rows, int64_t cols, int T, const char* what, std::mt19937& rng,
            const std::vector<float>* given = nullptr, double* rel_out = nullptr, double* rel16_out = nullptr,
            std::vector<double>* ref_out = nullptr) {
    const size_t wbytes = mmq::matrix_bytes(GGML_TYPE_NVFP4, rows, cols);
    std::vector<float> wf((size_t) rows * cols);
    ggml_get_type_traits(GGML_TYPE_NVFP4)->to_float(w_host, wf.data(), rows * cols);
    std::vector<float> x((size_t) T * cols);
    if (given) {
        x = *given;
    } else {
        std::normal_distribution<float> nd(0.f, 1.f);
        std::uniform_real_distribution<float> mag(0.1f, 10.f);
        for (int t = 0; t < T; ++t) {
            const float m = mag(rng);                                 // rows of different magnitude, like a residual
            for (int64_t k = 0; k < cols; ++k) x[(size_t) t * cols + k] = nd(rng) * m;
        }
    }
    std::vector<double> ref((size_t) T * rows);
    for (int t = 0; t < T; ++t)
        for (int64_t r = 0; r < rows; ++r) {
            double s = 0;
            const float* a = x.data() + (size_t) t * cols;
            const float* b = wf.data() + (size_t) r * cols;
            for (int64_t k = 0; k < cols; ++k) s += (double) a[k] * b[k];
            ref[(size_t) t * rows + r] = s;
        }

    uint8_t* dw = nullptr; float* dx = nullptr; void* dq = nullptr; float* dys = nullptr; float* dd = nullptr;
    int32_t* dids = nullptr; int32_t* dbounds = nullptr;
    ck(cudaMalloc(&dw, wbytes + 4096), "w");
    ck(cudaMemset(dw, 0, wbytes + 4096), "w");
    ck(cudaMemcpy(dw, w_host, wbytes, cudaMemcpyHostToDevice), "w");
    ck(cudaMalloc(&dx, x.size() * sizeof(float)), "x");
    ck(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "x");
    ck(cudaMalloc(&dq, mmq::q8_bytes(T, cols)), "xq");
    ck(cudaMalloc(&dys, T * sizeof(float)), "yscale");
    ck(cudaMalloc(&dd, (size_t) T * rows * sizeof(float)), "dst");
    ck(cudaMalloc(&dids, T * sizeof(int32_t)), "ids");
    ck(cudaMalloc(&dbounds, 2 * sizeof(int32_t)), "bounds");
    const int32_t bounds[2] = {0, T};
    ck(cudaMemcpy(dbounds, bounds, sizeof bounds, cudaMemcpyHostToDevice), "bounds");
    mmq::iota(dids, T, nullptr);
    mmq::quantize(dx, nullptr, dq, GGML_TYPE_NVFP4, cols, cols, T, nullptr, dys);
    mmq::Context ctx;
    mmq::Product p;
    p.w = dw; p.type = GGML_TYPE_NVFP4; p.w_rows = rows; p.w_cols = cols; p.expert_bytes = wbytes; p.n = 1;
    p.xq = dq; p.bounds = dbounds; p.ids = dids; p.total_rows = T; p.max_rows = T; p.dst = dd; p.ld_dst = rows;
    p.y_scale = mmq::fp4_activations(GGML_TYPE_NVFP4) ? dys : nullptr;
    ctx.run(p, nullptr);
    ck(cudaDeviceSynchronize(), "run");
    std::vector<float> got((size_t) T * rows);
    ck(cudaMemcpy(got.data(), dd, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "dst");
    cudaFree(dw); cudaFree(dx); cudaFree(dq); cudaFree(dys); cudaFree(dd); cudaFree(dids); cudaFree(dbounds);

    double num = 0, den = 0, worst = 0;
    for (int t = 0; t < T; ++t) {
        double rmax = 0;
        for (int64_t r = 0; r < rows; ++r) rmax = std::fmax(rmax, std::fabs(ref[(size_t) t * rows + r]));
        for (int64_t r = 0; r < rows; ++r) {
            const double e = (double) got[(size_t) t * rows + r] - ref[(size_t) t * rows + r];
            num += e * e;
            den += ref[(size_t) t * rows + r] * ref[(size_t) t * rows + r];
            worst = std::fmax(worst, std::fabs(e) / (rmax > 0 ? rmax : 1));
        }
    }
    const double rel = std::sqrt(num / den);
    if (rel_out) *rel_out = rel;
    if (ref_out) *ref_out = ref;
    if (rel16_out) {                                                  // the fp16 path: activations rounded to FP16
        double n16 = 0;
        for (int t = 0; t < T; ++t) {
            std::vector<float> xh(cols);
            for (int64_t k = 0; k < cols; ++k) xh[k] = half_round(x[(size_t) t * cols + k]);
            for (int64_t r = 0; r < rows; ++r) {
                double s = 0;
                for (int64_t k = 0; k < cols; ++k) s += (double) xh[k] * wf[(size_t) r * cols + k];
                const double e = s - ref[(size_t) t * rows + r];
                n16 += e * e;
            }
        }
        *rel16_out = std::sqrt(n16 / den);
        return rel > 0.2 ? 1 : 0;
    }
    std::printf("%-8s T %4d  [%lld x %lld]  rel. error %.4f%%  worst %.4f%% of the row max%s\n", what, T,
                (long long) rows, (long long) cols, 100 * rel, 100 * worst, rel > 0.05 ? "  <-- FP4-level or wrong" : "");
    return rel > 0.2 ? 1 : 0;
}

double dot_ref(const float* a, const float* b, int64_t n) {
    double s = 0;
    for (int64_t k = 0; k < n; ++k) s += (double) a[k] * b[k];
    return s;
}

struct Err {
    double num = 0, den = 0;
    void add(double got, double ref) { num += (got - ref) * (got - ref); den += ref * ref; }
    double rel() const { return den > 0 ? std::sqrt(num / den) : 0; }
};

int group(const char* path) {
    constexpr int64_t N = 2560, FF = 640;
    constexpr int E = 20;
    constexpr size_t STRIDE = 2764816;                                 // one layer-0 blob in experts.bin
    const size_t gub = mmq::matrix_bytes(GGML_TYPE_NVFP4, 2 * FF, N), db = mmq::matrix_bytes(GGML_TYPE_NVFP4, N, FF);
    std::vector<uint8_t> all((size_t) E * STRIDE);
    std::ifstream f(path, std::ios::binary);
    if (!f.read((char*) all.data(), (std::streamsize) all.size())) { std::fprintf(stderr, "cannot read %s\n", path); return 2; }
    std::vector<uint8_t> G((size_t) E * gub + 4096, 0), D((size_t) E * db + 4096, 0);
    for (int e = 0; e < E; ++e) {
        std::memcpy(G.data() + e * gub, all.data() + e * STRIDE, gub);
        std::memcpy(D.data() + e * db, all.data() + e * STRIDE + gub, db);
    }
    std::mt19937 rng(5);
    std::uniform_int_distribution<int> cntd(0, 60);
    std::vector<int32_t> bounds(E + 1, 0);
    int maxA = 0, maxB = 0;
    for (int e = 0; e < E; ++e) {
        const int c = (e % 7 == 3) ? 0 : cntd(rng);
        bounds[e + 1] = bounds[e] + c;
        if (e < 16) maxA = std::max(maxA, c); else maxB = std::max(maxB, c);
    }
    const int T = bounds[E];
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> mag(0.1f, 10.f);
    std::vector<float> x((size_t) T * N);
    for (int t = 0; t < T; ++t) { const float m = mag(rng); for (int64_t k = 0; k < N; ++k) x[(size_t) t * N + k] = nd(rng) * m; }
    // group B's down input: its own rows of FF values
    const int rB = bounds[16], nB = T - rB;
    std::vector<float> h((size_t) std::max(nB, 1) * FF);
    for (auto& v : h) v = nd(rng);
    std::vector<int32_t> relB(5);
    for (int i = 0; i <= 4; ++i) relB[i] = bounds[16 + i] - rB;

    uint8_t *dG, *dD; float *dx, *dh, *dgu, *ddn, *dys, *dhs; void *dxq, *dhq; int32_t *dids, *dbounds, *drel;
    ck(cudaMalloc(&dG, G.size()), "G"); ck(cudaMemcpy(dG, G.data(), G.size(), cudaMemcpyHostToDevice), "G");
    ck(cudaMalloc(&dD, D.size()), "D"); ck(cudaMemcpy(dD, D.data(), D.size(), cudaMemcpyHostToDevice), "D");
    ck(cudaMalloc(&dx, x.size() * 4), "x"); ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMalloc(&dh, h.size() * 4), "h"); ck(cudaMemcpy(dh, h.data(), h.size() * 4, cudaMemcpyHostToDevice), "h");
    ck(cudaMalloc(&dgu, (size_t) T * 2 * FF * 4), "gu"); ck(cudaMemset(dgu, 0, (size_t) T * 2 * FF * 4), "gu");
    ck(cudaMalloc(&ddn, (size_t) std::max(nB, 1) * N * 4), "dn");
    ck(cudaMalloc(&dxq, mmq::q8_bytes(T, N)), "xq"); ck(cudaMalloc(&dhq, mmq::q8_bytes(std::max(nB, 1), FF)), "hq");
    ck(cudaMalloc(&dys, T * 4), "ys"); ck(cudaMalloc(&dhs, std::max(nB, 1) * 4), "hs");
    ck(cudaMalloc(&dids, T * 4), "ids"); ck(cudaMalloc(&dbounds, (E + 1) * 4), "b"); ck(cudaMalloc(&drel, 5 * 4), "rel");
    ck(cudaMemcpy(dbounds, bounds.data(), (E + 1) * 4, cudaMemcpyHostToDevice), "b");
    ck(cudaMemcpy(drel, relB.data(), 5 * 4, cudaMemcpyHostToDevice), "rel");
    mmq::iota(dids, T, nullptr);
    const bool fp4 = mmq::fp4_activations(GGML_TYPE_NVFP4);
    mmq::quantize(dx, nullptr, dxq, GGML_TYPE_NVFP4, N, N, T, nullptr, dys);
    mmq::Context ctx;
    for (int grp = 0; grp < 2; ++grp) {
        mmq::Product p;
        p.w = dG + (grp ? 16 * gub : 0); p.type = GGML_TYPE_NVFP4; p.w_rows = 2 * FF; p.w_cols = N; p.expert_bytes = gub;
        p.n = grp ? 4 : 16; p.xq = dxq; p.bounds = dbounds + (grp ? 16 : 0); p.ids = dids; p.total_rows = T;
        p.max_rows = grp ? maxB : maxA; p.dst = dgu; p.ld_dst = 2 * FF; p.y_scale = fp4 ? dys : nullptr;
        ctx.run(p, nullptr);
    }
    mmq::quantize(dh, nullptr, dhq, GGML_TYPE_NVFP4, FF, FF, nB, nullptr, dhs);
    {
        mmq::Product p;
        p.w = dD + 16 * db; p.type = GGML_TYPE_NVFP4; p.w_rows = N; p.w_cols = FF; p.expert_bytes = db; p.n = 4;
        p.xq = dhq; p.bounds = drel; p.ids = dids; p.total_rows = nB; p.max_rows = maxB; p.dst = ddn; p.ld_dst = N;
        p.y_scale = fp4 ? dhs : nullptr;
        ctx.run(p, nullptr);
    }
    ck(cudaDeviceSynchronize(), "run");
    std::vector<float> gu((size_t) T * 2 * FF), dn((size_t) std::max(nB, 1) * N);
    ck(cudaMemcpy(gu.data(), dgu, gu.size() * 4, cudaMemcpyDeviceToHost), "gu");
    ck(cudaMemcpy(dn.data(), ddn, dn.size() * 4, cudaMemcpyDeviceToHost), "dn");
    cudaFree(dG); cudaFree(dD); cudaFree(dx); cudaFree(dh); cudaFree(dgu); cudaFree(ddn); cudaFree(dxq); cudaFree(dhq);
    cudaFree(dys); cudaFree(dhs); cudaFree(dids); cudaFree(dbounds); cudaFree(drel);

    Err ea, eb, ed;
    std::vector<float> wf((size_t) 2 * FF * N), wd((size_t) N * FF);
    for (int e = 0; e < E; ++e) {
        ggml_get_type_traits(GGML_TYPE_NVFP4)->to_float(G.data() + e * gub, wf.data(), 2 * FF * N);
        for (int t = bounds[e]; t < bounds[e + 1]; ++t)
            for (int64_t r = 0; r < 2 * FF; ++r)
                (e < 16 ? ea : eb).add(gu[(size_t) t * 2 * FF + r], dot_ref(x.data() + (size_t) t * N, wf.data() + r * N, N));
        if (e >= 16) {
            ggml_get_type_traits(GGML_TYPE_NVFP4)->to_float(D.data() + e * db, wd.data(), N * FF);
            for (int t = bounds[e]; t < bounds[e + 1]; ++t)
                for (int64_t r = 0; r < N; ++r)
                    ed.add(dn[(size_t) (t - rB) * N + r], dot_ref(h.data() + (size_t) (t - rB) * FF, wd.data() + r * FF, FF));
        }
    }
    std::printf("group: %d rows over %d experts (max %d / %d per expert)\n", T, E, maxA, maxB);
    std::printf("  gate/up, experts 0-15 (bounds from 0)      rel. error %.4f%%\n", 100 * ea.rel());
    std::printf("  gate/up, experts 16-19 (bounds from 16)    rel. error %.4f%%\n", 100 * eb.rel());
    std::printf("  down, experts 16-19 (relative bounds)      rel. error %.4f%%\n", 100 * ed.rel());
    const double worst = std::max(ea.rel(), std::max(eb.rel(), ed.rel()));
    std::printf("RESULT: %s\n", worst > 0.2 ? "WRONG (error beyond 20%)" : "ok");
    return worst > 0.2 ? 1 : 0;
}

int layers(const char* path, int e) {
    constexpr int64_t N = 2560, FF = 640;
    constexpr size_t STRIDE = 2764816, LAYER = 512 * STRIDE;
    const size_t gub = mmq::matrix_bytes(GGML_TYPE_NVFP4, 2 * FF, N), db = mmq::matrix_bytes(GGML_TYPE_NVFP4, N, FF);
    std::vector<uint8_t> blob(gub + db);
    std::ifstream f(path, std::ios::binary);
    std::mt19937 rng(3);
    int bad = 0;
    for (int l = 0; l < 48; ++l) {
        f.seekg((std::streamoff) (l * LAYER + (size_t) e * STRIDE));
        if (!f.read((char*) blob.data(), (std::streamsize) blob.size())) { std::fprintf(stderr, "cannot read layer %d\n", l); return 2; }
        char name[32];
        std::snprintf(name, sizeof name, "L%02d gu", l);
        bad += product(blob.data(), 2 * FF, N, 64, name, rng);
        std::snprintf(name, sizeof name, "L%02d dn", l);
        bad += product(blob.data() + gub, N, FF, 64, name, rng);
    }
    std::printf("RESULT: %s\n", bad ? "WRONG (error beyond 20%)" : "ok");
    return bad ? 1 : 0;
}

int real(const char* experts_bin, const char* dump, int layer) {
    std::ifstream d(dump, std::ios::binary);
    int64_t hdr[3];
    if (!d.read((char*) hdr, sizeof hdr)) { std::fprintf(stderr, "cannot read %s\n", dump); return 2; }
    const int64_t T = hdr[0], N = hdr[1], K = hdr[2], FF = 640;
    std::vector<float> x((size_t) (T * N));
    std::vector<int32_t> ids((size_t) (T * K));
    d.read((char*) x.data(), (std::streamsize) (x.size() * 4));
    d.read((char*) ids.data(), (std::streamsize) (ids.size() * 4));
    // the rows' outlier ratio
    std::vector<double> ratio;
    for (int64_t t = 0; t < T; ++t) {
        double mx = 0, ss = 0;
        for (int64_t k = 0; k < N; ++k) { const double v = x[(size_t) (t * N + k)]; mx = std::fmax(mx, std::fabs(v)); ss += v * v; }
        ratio.push_back(mx / std::sqrt(ss / N + 1e-30));
    }
    std::sort(ratio.begin(), ratio.end());
    std::printf("layer %d: %lld tokens, max|x| / rms: median %.1f, p99 %.1f, max %.1f\n", layer, (long long) T,
                ratio[ratio.size() / 2], ratio[ratio.size() * 99 / 100], ratio.back());
    std::vector<int> cnt(512, 0);
    for (int32_t e : ids) if (e >= 0 && e < 512) ++cnt[e];
    std::vector<int> order(512);
    for (int i = 0; i < 512; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return cnt[a] > cnt[b]; });
    constexpr size_t STRIDE = 2764816, LAYER = 512 * STRIDE;
    const size_t gub = mmq::matrix_bytes(GGML_TYPE_NVFP4, 2 * FF, N);
    std::ifstream f(experts_bin, std::ios::binary);
    std::vector<uint8_t> blob(STRIDE);
    std::mt19937 rng(1);
    double s_gu = 0, s_gu16 = 0, s_dn = 0, s_dn16 = 0;
    int n = 0;
    std::printf("%6s %5s  %10s %10s  %10s %10s\n", "expert", "rows", "gu mmq", "gu fp16", "down mmq", "down fp16");
    for (int i = 0; i < 12; ++i) {
        const int e = order[i];
        f.seekg((std::streamoff) (layer * LAYER + (size_t) e * STRIDE));
        f.read((char*) blob.data(), (std::streamsize) blob.size());
        float tail[4];
        std::memcpy(tail, blob.data() + STRIDE - 16, sizeof tail);
        std::vector<float> xe;
        for (int64_t t = 0; t < T; ++t)
            for (int64_t k = 0; k < K; ++k)
                if (ids[(size_t) (t * K + k)] == e) xe.insert(xe.end(), x.begin() + t * N, x.begin() + (t + 1) * N);
        const int Te = (int) (xe.size() / N);
        double gu = 0, gu16 = 0, dn = 0, dn16 = 0;
        std::vector<double> ref;
        product(blob.data(), 2 * FF, N, Te, "gu", rng, &xe, &gu, &gu16, &ref);
        std::vector<float> h((size_t) Te * FF);                      // the hidden the exact gate/up gives
        for (int t = 0; t < Te; ++t)
            for (int64_t r = 0; r < FF; ++r) {
                const double g = ref[(size_t) t * 2 * FF + r] * tail[0], u = ref[(size_t) t * 2 * FF + FF + r] * tail[1];
                h[(size_t) t * FF + r] = (float) (g / (1 + std::exp(-g)) * u);
            }
        product(blob.data() + gub, N, FF, Te, "down", rng, &h, &dn, &dn16);
        std::printf("%6d %5d  %9.3f%% %9.3f%%  %9.3f%% %9.3f%%\n", e, Te, 100 * gu, 100 * gu16, 100 * dn, 100 * dn16);
        s_gu += gu; s_gu16 += gu16; s_dn += dn; s_dn16 += dn16; ++n;
    }
    std::printf("%6s %5s  %9.3f%% %9.3f%%  %9.3f%% %9.3f%%\n", "mean", "", 100 * s_gu / n, 100 * s_gu16 / n,
                100 * s_dn / n, 100 * s_dn16 / n);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: mmq_nvfp4_parity pack/experts.bin [rows ...]\n"); return 2; }
    constexpr int64_t N = 2560, FF = 640;
    const size_t gu_bytes = mmq::matrix_bytes(GGML_TYPE_NVFP4, 2 * FF, N), d_bytes = mmq::matrix_bytes(GGML_TYPE_NVFP4, N, FF);
    std::vector<uint8_t> blob(gu_bytes + d_bytes);
    std::ifstream f(argv[1], std::ios::binary);
    if (!f.read((char*) blob.data(), (std::streamsize) blob.size())) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 2; }
    if (argc >= 3 && std::string(argv[2]) == "--group") return group(argv[1]);
    if (argc >= 5 && std::string(argv[2]) == "--real") return real(argv[1], argv[3], std::atoi(argv[4]));
    if (argc >= 3 && std::string(argv[2]) == "--layers") return layers(argv[1], argc >= 4 ? std::atoi(argv[3]) : 7);
    const char* mode = std::getenv("STRATA_PREFILL_NVFP4");
    std::printf("mode %s, FP4 activations: %s\n", mode ? mode : "default", mmq::fp4_activations(GGML_TYPE_NVFP4) ? "yes" : "no");
    std::vector<int> Ts;
    for (int i = 2; i < argc; ++i) Ts.push_back(std::atoi(argv[i]));
    if (Ts.empty()) Ts = {1, 7, 37, 128, 300};
    std::mt19937 rng(11);
    int bad = 0;
    for (int T : Ts) {
        bad += product(blob.data(), 2 * FF, N, T, "gate/up", rng);
        bad += product(blob.data() + gu_bytes, N, FF, T, "down", rng);
    }
    std::printf("RESULT: %s\n", bad ? "WRONG (error beyond 20%)" : "ok");
    return bad ? 1 : 0;
}
