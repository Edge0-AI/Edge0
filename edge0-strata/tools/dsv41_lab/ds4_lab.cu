// tools/dsv41_lab/ds4_lab.cu - parity + speed of the DeepSeek-V4.1 device kernels (src/dsv41/ds4_kernels.cu, the file the port
// ports, linked unmodified) on synthetic tensors at the real shapes: dim 5120, hc 4 (mHC 24 x 20480, 20 Sinkhorn iterations),
// 64 heads x 512 (MQA, K = V), 128-row window + sink, top-512 compressed rows (E2M1 + E4M3 scales), indexer 32 x 128,
// router 384 / 6, Engram rows 256 (FP8 + E8M0).  References are plain double-precision C++ written from the kernels' documented
// math (DeepSeek model.py), not from the kernels.
//   ds4_lab [--tokens 1,2,3,4,5] [--only name,name] [--ctx 8192] [--reps 100]
#include "../../src/dsv41/ds4_kernels.cuh"
#include "attn_split.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

using namespace strata::ds4;

static uint64_t g_rng = 0x2545f4914f6cdd1dull;
static uint64_t rnd() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return g_rng; }
static float urand() { return (float) ((rnd() >> 40) * (1.0 / 16777216.0)); }              // [0, 1)
static float srand_() { return urand() * 2.f - 1.f; }                                      // [-1, 1)
static float nrand() { float u = 0; for (int i = 0; i < 12; ++i) u += urand(); return u - 6.f; }

static cudaStream_t g_s;
static int g_fails = 0;
static int g_reps = 100;

template <class T> struct Dev {
    T* p = nullptr; size_t n = 0;
    Dev() {}
    explicit Dev(const std::vector<T>& h) { set(h); }
    void set(const std::vector<T>& h) { if (p) cudaFree(p); n = h.size(); CK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T))); CK(cudaMemcpy(p, h.data(), n * sizeof(T), cudaMemcpyHostToDevice)); }
    void alloc(size_t cnt) { if (p) cudaFree(p); n = cnt; CK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T))); CK(cudaMemset(p, 0, n * sizeof(T))); }
    std::vector<T> get() const { std::vector<T> h(n); CK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return h; }
    ~Dev() { if (p) cudaFree(p); }
};

static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t) (u >> 16); }
static float bf2f(uint16_t b) { uint32_t u = (uint32_t) b << 16; float f; std::memcpy(&f, &u, 4); return f; }
static float e4m3_val(uint8_t q) {
    const int e = (q >> 3) & 15, m = q & 7; float v;
    if (e == 15 && m == 7) v = NAN; else if (e == 0) v = m * (1.f / 512.f); else v = std::ldexp(1.f + m / 8.f, e - 7);
    return (q & 0x80) ? -v : v;
}
static const float kE2m1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
static float e2m1_val(int c) { return (c & 8) ? -kE2m1[c & 7] : kE2m1[c & 7]; }

// ---- timing: n launches of `fn` captured into a graph (no launch gaps), replayed
static double time_us(const std::function<void()>& fn) {
    cudaGraph_t g; cudaGraphExec_t ge;
    const int inner = 20;
    CK(cudaStreamBeginCapture(g_s, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < inner; ++i) fn();
    CK(cudaStreamEndCapture(g_s, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    CK(cudaGraphLaunch(ge, g_s)); CK(cudaStreamSynchronize(g_s));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int outer = std::max(1, g_reps / inner);
    cudaEventRecord(e0, g_s);
    for (int i = 0; i < outer; ++i) CK(cudaGraphLaunch(ge, g_s));
    cudaEventRecord(e1, g_s); CK(cudaEventSynchronize(e1));
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
    return ms * 1e3 / (outer * inner);
}

static void report(const char* name, int T, double err, double tol, double us, const char* extra = "") {
    const bool ok = err <= tol;
    g_fails += !ok;
    std::printf("%-22s T=%-2d  err %.2e (tol %.0e) %-4s  %9.2f us  %s\n", name, T, err, tol, ok ? "ok" : "FAIL", us, extra);
}
static double max_rel(const std::vector<float>& got, const std::vector<double>& ref) {   // max |got-ref| / (max|ref| + tiny)
    double mx = 0, e = 0;
    for (double r : ref) mx = std::max(mx, std::fabs(r));
    for (size_t i = 0; i < ref.size(); ++i) e = std::max(e, std::fabs(got[i] - ref[i]));
    return e / (mx + 1e-30);
}

// ============================================================================================================ mHC
static void test_mhc(int T) {
    const int dim = 5120, hc = 4, hcdim = hc * dim, n_mix = 24, iters = 20;
    const float eps = 1e-6f, hc_eps = 1e-6f;
    std::vector<float> h((size_t) T * hcdim), fn((size_t) n_mix * hcdim), scale = {0.3f, 0.5f, 0.7f}, base(24);
    for (auto& v : h) v = nrand();
    for (auto& v : fn) v = nrand() * 0.01f;
    for (auto& v : base) v = srand_();
    Dev<float> dh(h), dfn(fn), dscale(scale), dbase(base), dmix, dpre, dpost, dcomb;
    dmix.alloc((size_t) T * 32); dpre.alloc((size_t) T * 8); dpost.alloc((size_t) T * 8); dcomb.alloc((size_t) T * 64);
    // hc_mix
    hc_mix(dh.p, dfn.p, hcdim, n_mix, T, eps, dmix.p, g_s); CK(cudaStreamSynchronize(g_s));
    auto mix = dmix.get();
    std::vector<double> rmix((size_t) T * 32, 0.0), gm((size_t) T * 32, 0.0);
    for (int t = 0; t < T; ++t) {
        double ss = 0; for (int k = 0; k < hcdim; ++k) ss += (double) h[(size_t) t * hcdim + k] * h[(size_t) t * hcdim + k];
        const double inv = 1.0 / std::sqrt(ss / hcdim + eps);
        for (int m = 0; m < n_mix; ++m) { double a = 0; for (int k = 0; k < hcdim; ++k) a += (double) h[(size_t) t * hcdim + k] * fn[(size_t) m * hcdim + k]; rmix[(size_t) t * 32 + m] = a * inv; gm[(size_t) t * 32 + m] = mix[(size_t) t * 32 + m]; }
    }
    std::vector<float> mixg(mix.begin(), mix.end());
    { std::vector<double> r2; std::vector<float> g2; for (int t = 0; t < T; ++t) for (int m = 0; m < n_mix; ++m) { r2.push_back(rmix[(size_t) t * 32 + m]); g2.push_back(mix[(size_t) t * 32 + m]); }
      report("hc_mix", T, max_rel(g2, r2), 1e-5, time_us([&] { hc_mix(dh.p, dfn.p, hcdim, n_mix, T, eps, dmix.p, g_s); }), "24 x 20480 dots + norm"); }
    // hc_split (Sinkhorn)
    hc_split(dmix.p, dscale.p, dbase.p, hc, iters, hc_eps, T, dpre.p, dpost.p, dcomb.p, g_s); CK(cudaStreamSynchronize(g_s));
    auto pre = dpre.get(), post = dpost.get(), comb = dcomb.get();
    {
        std::vector<double> rr; std::vector<float> gg;
        for (int t = 0; t < T; ++t) {
            const float* mx = mix.data() + (size_t) t * 32;
            for (int k = 0; k < hc; ++k) { rr.push_back(1.0 / (1.0 + std::exp(-((double) mx[k] * scale[0] + base[k]))) + hc_eps); gg.push_back(pre[(size_t) t * 8 + k]); }
            for (int k = 0; k < hc; ++k) { rr.push_back(2.0 / (1.0 + std::exp(-((double) mx[hc + k] * scale[1] + base[hc + k])))); gg.push_back(post[(size_t) t * 8 + k]); }
            double c[4][4];
            for (int j = 0; j < 4; ++j) { double m = -1e300; for (int k = 0; k < 4; ++k) { c[j][k] = (double) mx[8 + j * 4 + k] * scale[2] + base[8 + j * 4 + k]; m = std::max(m, c[j][k]); }
                double s = 0; for (int k = 0; k < 4; ++k) { c[j][k] = std::exp(c[j][k] - m); s += c[j][k]; } for (int k = 0; k < 4; ++k) c[j][k] = c[j][k] / s + hc_eps; }
            auto colnorm = [&] { for (int k = 0; k < 4; ++k) { double s = 0; for (int j = 0; j < 4; ++j) s += c[j][k]; for (int j = 0; j < 4; ++j) c[j][k] /= s + hc_eps; } };
            auto rownorm = [&] { for (int j = 0; j < 4; ++j) { double s = 0; for (int k = 0; k < 4; ++k) s += c[j][k]; for (int k = 0; k < 4; ++k) c[j][k] /= s + hc_eps; } };
            colnorm();
            for (int it = 1; it < iters; ++it) { rownorm(); colnorm(); }
            for (int j = 0; j < 4; ++j) for (int k = 0; k < 4; ++k) { rr.push_back(c[j][k]); gg.push_back(comb[(size_t) t * 64 + j * 4 + k]); }
        }
        report("hc_split sinkhorn", T, max_rel(gg, rr), 2e-5, time_us([&] { hc_split(dmix.p, dscale.p, dbase.p, hc, iters, hc_eps, T, dpre.p, dpost.p, dcomb.p, g_s); }), "20 iterations");
    }
    // hc_pre_norm: x = rmsnorm(sum_s pre[s] h[s]) * w
    std::vector<float> w(dim); for (auto& v : w) v = 0.5f + urand();
    Dev<float> dw(w), dx; dx.alloc((size_t) T * dim);
    hc_pre_norm(dh.p, dpre.p, dw.p, dim, hc, T, eps, dx.p, g_s); CK(cudaStreamSynchronize(g_s));
    {
        auto x = dx.get(); std::vector<double> rr; std::vector<float> gg;
        for (int t = 0; t < T; ++t) {
            std::vector<double> a(dim); double ss = 0;
            for (int e = 0; e < dim; ++e) { double s = 0; for (int k = 0; k < hc; ++k) s += (double) pre[(size_t) t * 8 + k] * h[((size_t) t * hc + k) * dim + e]; a[e] = s; ss += s * s; }
            const double inv = 1.0 / std::sqrt(ss / dim + eps);
            for (int e = 0; e < dim; ++e) { rr.push_back(a[e] * inv * w[e]); gg.push_back(x[(size_t) t * dim + e]); }
        }
        report("hc_pre_norm", T, max_rel(gg, rr), 2e-5, time_us([&] { hc_pre_norm(dh.p, dpre.p, dw.p, dim, hc, T, eps, dx.p, g_s); }));
    }
    // hc_post / hc_post_inplace: out[k][e] = post[k] y[e] + sum_j comb[j][k] res[j][e]
    std::vector<float> y((size_t) T * dim); for (auto& v : y) v = nrand();
    Dev<float> dy(y), dout; dout.alloc((size_t) T * hcdim);
    hc_post(dy.p, dh.p, dpost.p, dcomb.p, dim, hc, T, dout.p, g_s); CK(cudaStreamSynchronize(g_s));
    {
        auto o = dout.get(); std::vector<double> rr; std::vector<float> gg;
        for (int t = 0; t < T; ++t) for (int k = 0; k < hc; ++k) for (int e = 0; e < dim; ++e) {
            double a = (double) post[(size_t) t * 8 + k] * y[(size_t) t * dim + e];
            for (int j = 0; j < hc; ++j) a += (double) comb[(size_t) t * 64 + j * hc + k] * h[((size_t) t * hc + j) * dim + e];
            rr.push_back(a); gg.push_back(o[((size_t) t * hc + k) * dim + e]);
        }
        const double err = max_rel(gg, rr);
        report("hc_post", T, err, 2e-6, time_us([&] { hc_post(dy.p, dh.p, dpost.p, dcomb.p, dim, hc, T, dout.p, g_s); }));
        Dev<float> dh2(h);
        hc_post_inplace(dy.p, dh2.p, dpost.p, dcomb.p, dim, hc, T, g_s); CK(cudaStreamSynchronize(g_s));
        auto o2 = dh2.get(); std::vector<float> gg2(o2.begin(), o2.begin() + (size_t) T * hcdim);
        report("hc_post_inplace", T, max_rel(gg2, rr), 2e-6, time_us([&] { hc_post_inplace(dy.p, dh2.p, dpost.p, dcomb.p, dim, hc, T, g_s); }), "(timing re-applies in place)");
    }
}

// ============================================================================================================ router
static void test_router(int T) {
    const int n = 384, k = 6; const float rs = 1.5f;
    std::vector<float> logits((size_t) T * 1024, 0.f), bias(n);
    for (int t = 0; t < T; ++t) for (int i = 0; i < n; ++i) logits[(size_t) t * 1024 + i] = nrand() * 2.f;
    for (auto& v : bias) v = srand_() * 0.3f;
    Dev<float> dl(logits), db(bias), dw; Dev<int> di; dw.alloc((size_t) T * kPicks); di.alloc((size_t) T * kPicks);
    route(dl.p, db.p, n, k, T, rs, di.p, dw.p, g_s); CK(cudaStreamSynchronize(g_s));
    auto gi = di.get(); auto gw = dw.get();
    int bad_idx = 0; std::vector<double> rr; std::vector<float> gg;
    for (int t = 0; t < T; ++t) {
        std::vector<double> sc(n), bs(n);
        for (int i = 0; i < n; ++i) { const double l = logits[(size_t) t * 1024 + i]; sc[i] = std::sqrt(l > 20 ? l : std::log1p(std::exp(l))); bs[i] = sc[i] + bias[i]; }
        std::vector<int> ord(n); for (int i = 0; i < n; ++i) ord[i] = i;
        std::partial_sort(ord.begin(), ord.begin() + k, ord.end(), [&](int a, int b) { return bs[a] > bs[b] || (bs[a] == bs[b] && a < b); });
        double s = 0; for (int r = 0; r < k; ++r) s += sc[ord[r]];
        for (int r = 0; r < k; ++r) { bad_idx += gi[(size_t) t * kPicks + r] != ord[r]; rr.push_back(sc[ord[r]] / (s + 1e-20) * rs); gg.push_back(gw[(size_t) t * kPicks + r]); }
    }
    char ex[64]; std::snprintf(ex, sizeof ex, "%d / %d picks differ", bad_idx, T * k);
    report("route (sqrt-softplus)", T, bad_idx ? 1.0 : max_rel(gg, rr), 1e-5, time_us([&] { route(dl.p, db.p, n, k, T, rs, di.p, dw.p, g_s); }), ex);
}

// ============================================================================================================ compressor
static void test_compress(int T, int ratio) {
    const int d = 512; const float eps = 1e-6f;
    // state pre-filled for the slots before p0; p0 chosen so that the last token of the window completes a group
    const int p0 = ratio * 3 + (ratio - T);
    std::vector<float> kv((size_t) T * d), sc((size_t) T * d), kst((size_t) ratio * d), sst((size_t) ratio * d), w(d);
    for (auto& v : kv) v = nrand(); for (auto& v : sc) v = nrand() * 2.f; for (auto& v : kst) v = nrand(); for (auto& v : sst) v = nrand() * 2.f;
    for (auto& v : w) v = 0.5f + urand();
    Dev<float> dkv(kv), dsc(sc), dk(kst), ds(sst), dw(w), dlat; dlat.alloc((size_t) d * 2);
    compress_win(dkv.p, dsc.p, dk.p, ds.p, ratio, p0, T, d, dw.p, eps, dlat.p, g_s); CK(cudaStreamSynchronize(g_s));
    auto lat = dlat.get();
    // reference: slots (p0+t)%ratio overwritten by the window's tokens, group = all `ratio` slots
    std::vector<float> k2 = kst, s2 = sst;
    for (int t = 0; t < T; ++t) { const int slot = (p0 + t) % ratio; for (int e = 0; e < d; ++e) { k2[(size_t) slot * d + e] = kv[(size_t) t * d + e]; s2[(size_t) slot * d + e] = sc[(size_t) t * d + e]; } }
    std::vector<double> rr(d); double ss = 0;
    for (int e = 0; e < d; ++e) { double m = -1e300; for (int s = 0; s < ratio; ++s) m = std::max(m, (double) s2[(size_t) s * d + e]);
        double den = 0; for (int s = 0; s < ratio; ++s) den += std::exp(s2[(size_t) s * d + e] - m);
        double a = 0; for (int s = 0; s < ratio; ++s) a += (double) k2[(size_t) s * d + e] * std::exp(s2[(size_t) s * d + e] - m) / den; rr[e] = a; ss += a * a; }
    const double inv = 1.0 / std::sqrt(ss / d + eps); for (int e = 0; e < d; ++e) rr[e] = rr[e] * inv * w[e];
    char nm[32]; std::snprintf(nm, sizeof nm, "compress_win r=%d", ratio);
    std::vector<float> gg(lat.begin(), lat.begin() + d);
    report(nm, T, max_rel(gg, rr), 5e-6, time_us([&] { compress_win(dkv.p, dsc.p, dk.p, ds.p, ratio, p0, T, d, dw.p, eps, dlat.p, g_s); }), "d 512, pooling window");
}

// ============================================================================================================ indexer
static void test_indexer(int T, int n) {
    const int heads = 32, hd = 128, topk_k = 512;
    const size_t rb = kv_row_bytes(kKvFp4p, hd);
    std::vector<uint8_t> keys((size_t) n * rb, 0);
    std::vector<float> kdec((size_t) n * hd);
    for (int p = 0; p < n; ++p) {
        uint8_t* r = &keys[(size_t) p * rb];
        for (int j = 0; j < hd / 2; ++j) r[j] = (uint8_t) (rnd() & 0xff);
        for (int g = 0; g < hd / 32; ++g) r[hd / 2 + g] = (uint8_t) (120 + (rnd() % 5));   // 2^-7 .. 2^-3
        for (int e = 0; e < hd; ++e) { const int code = (r[e / 2] >> (4 * (e & 1))) & 15; kdec[(size_t) p * hd + e] = e2m1_val(code) * std::ldexp(1.f, (int) r[hd / 2 + e / 32] - 127); }
    }
    std::vector<float> q((size_t) T * heads * hd), w((size_t) T * 64, 0.f);
    for (auto& v : q) v = nrand(); for (int t = 0; t < T; ++t) for (int h = 0; h < heads; ++h) w[(size_t) t * 64 + h] = srand_();
    Dev<uint8_t> dk(keys); Dev<float> dq(q), dw(w), dout; dout.alloc((size_t) T * n);
    Rows lim; for (int t = 0; t < kWin; ++t) { lim.n[t] = n; lim.k[t] = topk_k; }
    index_score(dq.p, dw.p, dk.p, heads, hd, T, n, nullptr, 0, 1, lim, dout.p, n, g_s); CK(cudaStreamSynchronize(g_s));
    auto sc = dout.get();
    std::vector<double> rr; std::vector<float> gg;
    for (int t = 0; t < T; ++t) for (int p = 0; p < n; p += std::max(1, n / 512)) {
        double s = 0; for (int h = 0; h < heads; ++h) { double a = 0; for (int e = 0; e < hd; ++e) a += (double) q[((size_t) t * heads + h) * hd + e] * kdec[(size_t) p * hd + e]; s += std::max(a, 0.0) * w[(size_t) t * 64 + h]; }
        rr.push_back(s); gg.push_back(sc[(size_t) t * n + p]);
    }
    char ex[96]; std::snprintf(ex, sizeof ex, "%d keys, %.1f MB of keys", n, n * (double) rb / 1e6);
    const double us = time_us([&] { index_score(dq.p, dw.p, dk.p, heads, hd, T, n, nullptr, 0, 1, lim, dout.p, n, g_s); });
    report("index_score", T, max_rel(gg, rr), 3e-5, us, ex);
    // top-512 of the device's own scores: exact set compare
    Rows tr; for (int t = 0; t < kWin; ++t) { tr.n[t] = n; tr.k[t] = std::min(topk_k, n); }
    Dev<int> dsel; dsel.alloc((size_t) T * topk_k);
    topk(dout.p, n, T, tr, nullptr, 0, 1, dsel.p, topk_k, g_s); CK(cudaStreamSynchronize(g_s));
    auto sel = dsel.get(); int bad = 0;
    for (int t = 0; t < T; ++t) {
        std::vector<int> ord(n); for (int i = 0; i < n; ++i) ord[i] = i;
        std::nth_element(ord.begin(), ord.begin() + topk_k, ord.end(), [&](int a, int b) { return sc[(size_t) t * n + a] > sc[(size_t) t * n + b]; });
        std::vector<char> in(n, 0); for (int i = 0; i < topk_k; ++i) in[ord[i]] = 1;
        const float kth = sc[(size_t) t * n + *std::min_element(ord.begin(), ord.begin() + topk_k, [&](int a, int b) { return sc[(size_t) t * n + a] < sc[(size_t) t * n + b]; })];
        for (int i = 0; i < topk_k; ++i) { const int p = sel[(size_t) t * topk_k + i]; if (p < 0 || p >= n || (!in[p] && sc[(size_t) t * n + p] != kth)) ++bad; }
    }
    std::snprintf(ex, sizeof ex, "%d wrong picks of %d", bad, T * topk_k);
    report("topk 512", T, bad ? 1.0 : 0.0, 0.0, time_us([&] { topk(dout.p, n, T, tr, nullptr, 0, 1, dsel.p, topk_k, g_s); }), ex);
}

// ============================================================================================================ attention
static void test_attn(int T, int ncache) {
    const int heads = 64, d = 512, nwin = 128, nsel = 512;
    const float scale = 1.f / std::sqrt((float) d);
    const size_t rb = kv_row_bytes(kKvFp4, d);   // 304
    std::vector<uint8_t> ckv((size_t) ncache * rb, 0);
    std::vector<float> cdec((size_t) ncache * d);
    for (int p = 0; p < ncache; ++p) {
        uint8_t* r = &ckv[(size_t) p * rb];
        for (int j = 0; j < d / 2; ++j) r[j] = (uint8_t) (rnd() & 0xff);
        for (int g = 0; g < d / 16; ++g) r[d / 2 + g] = (uint8_t) ((7 << 3) | (rnd() & 7)) - (uint8_t) (rnd() % 3 * 8);   // ~1 .. 0.25
        for (int e = 0; e < d; ++e) { const int code = (r[e / 2] >> (4 * (e & 1))) & 15; cdec[(size_t) p * d + e] = e2m1_val(code) * e4m3_val(r[d / 2 + e / 16]); }
    }
    // window buffer: nwin + T - 1 rows (each query sees nwin of them)
    const int wrows = nwin + T;
    std::vector<uint16_t> win((size_t) wrows * d); std::vector<float> wdec((size_t) wrows * d);
    for (size_t i = 0; i < win.size(); ++i) { win[i] = f2bf(nrand()); wdec[i] = bf2f(win[i]); }
    std::vector<float> q((size_t) T * heads * d), sink(heads);
    for (auto& v : q) v = nrand(); for (auto& v : sink) v = srand_();
    std::vector<int> sel((size_t) T * nsel);
    for (int t = 0; t < T; ++t) for (int i = 0; i < nsel; ++i) sel[(size_t) t * nsel + i] = (int) (rnd() % ncache);
    AttnRows ar; for (int t = 0; t < kWin; ++t) { ar.win0[t] = t < T ? t + 1 : 0; ar.nwin[t] = nwin; ar.nsel[t] = nsel; }
    Dev<uint8_t> dck(ckv); Dev<uint16_t> dwin(win); Dev<float> dq(q), dsink(sink), dout; Dev<int> dsel(sel); dout.alloc((size_t) T * heads * d);
    Dev<uint16_t> dscr; dscr.alloc((size_t) kAttnScratchT * (nsel + 8) * d);
    AttnScratch scr; scr.rows = dscr.p; scr.max_t = kAttnScratchT; scr.cap = nsel + 8;
    sparse_attn(dq.p, dwin.p, dck.p, dsel.p, nsel, ar, dsink.p, heads, d, T, scale, dout.p, g_s, kKvFp4, 0, &scr); CK(cudaStreamSynchronize(g_s));
    auto o = dout.get(); std::vector<double> rr; std::vector<float> gg;
    for (int t = 0; t < T; t += std::max(1, T / 2)) for (int h = 0; h < heads; h += 7) {
        std::vector<double> s; std::vector<const float*> rows;
        for (int j = 0; j < nwin; ++j) rows.push_back(&wdec[(size_t) (ar.win0[t] + j) * d]);
        for (int i = 0; i < nsel; ++i) rows.push_back(&cdec[(size_t) sel[(size_t) t * nsel + i] * d]);
        double m = sink[h];
        for (auto* rw : rows) { double a = 0; for (int e = 0; e < d; ++e) a += (double) q[((size_t) t * heads + h) * d + e] * rw[e]; s.push_back(a * scale); m = std::max(m, s.back()); }
        double den = std::exp(sink[h] - m); for (double v : s) den += std::exp(v - m);
        std::vector<double> acc(d, 0.0);
        for (size_t j = 0; j < rows.size(); ++j) { const double p = std::exp(s[j] - m) / den; for (int e = 0; e < d; ++e) acc[e] += p * rows[j][e]; }
        for (int e = 0; e < d; ++e) { rr.push_back(acc[e]); gg.push_back(o[((size_t) t * heads + h) * d + e]); }
    }
    char ex[96]; std::snprintf(ex, sizeof ex, "64 heads x 512, 128 win + 512 sel, cache %d rows", ncache);
    report("sparse_attn (fp4 ckv)", T, max_rel(gg, rr), 1e-4, time_us([&] { sparse_attn(dq.p, dwin.p, dck.p, dsel.p, nsel, ar, dsink.p, heads, d, T, scale, dout.p, g_s, kKvFp4, 0, &scr); }), ex);
    // split-KV version (tools/dsv41_lab/attn_split.cuh), same inputs
    {
        const int total = nwin + nsel;
        auto run = [&](const char* nm, auto launch_fn, int R) {
            const int nsplit = (total + R - 1) / R;
            Dev<float> part, ml; part.alloc((size_t) T * heads * nsplit * d); ml.alloc((size_t) T * heads * nsplit * 2);
            strata::ds4lab::AttnSplitScratch ss; ss.part = part.p; ss.ml = ml.p; ss.nsplit = nsplit; ss.max_t = T;
            Dev<float> o3; o3.alloc((size_t) T * heads * d);
            auto go = [&] { launch_fn(dq.p, dwin.p, dck.p, rb, dsel.p, nsel, ar, dsink.p, heads, T, scale, o3.p, ss, total, g_s); };
            go(); CK(cudaStreamSynchronize(g_s)); CK(cudaGetLastError());
            auto oo = o3.get(); std::vector<float> g3;
            for (int t = 0; t < T; t += std::max(1, T / 2)) for (int h = 0; h < heads; h += 7) for (int e = 0; e < d; ++e) g3.push_back(oo[((size_t) t * heads + h) * d + e]);
            report(nm, T, max_rel(g3, rr), 1e-4, time_us(go), ex);
        };
        run("attn_split G4 R40", [](auto... a) { strata::ds4lab::sparse_attn_split<4, 40>(a...); }, 40);
        run("attn_split G8 R40", [](auto... a) { strata::ds4lab::sparse_attn_split<8, 40>(a...); }, 40);
        run("attn_split G4 R20", [](auto... a) { strata::ds4lab::sparse_attn_split<4, 20>(a...); }, 20);
        run("attn_split G2 R32", [](auto... a) { strata::ds4lab::sparse_attn_split<2, 32>(a...); }, 32);
    }
    // the packed-rows kernel (no scratch): used from 8 queries on, timed here at this T for comparison
    sparse_attn(dq.p, dwin.p, dck.p, dsel.p, nsel, ar, dsink.p, heads, d, T, scale, dout.p, g_s, kKvFp4, 0, nullptr); CK(cudaStreamSynchronize(g_s));
    { auto o2 = dout.get(); std::vector<float> g2; for (int t = 0; t < T; t += std::max(1, T / 2)) for (int h = 0; h < heads; h += 7) for (int e = 0; e < d; ++e) g2.push_back(o2[((size_t) t * heads + h) * d + e]);
      report("sparse_attn (packed)", T, max_rel(g2, rr), 1e-4, time_us([&] { sparse_attn(dq.p, dwin.p, dck.p, dsel.p, nsel, ar, dsink.p, heads, d, T, scale, dout.p, g_s, kKvFp4, 0, nullptr); })); }
}

// ============================================================================================================ Engram
static void test_engram(int T) {
    const int cols = 24, hd = 256, hc = 4, dim = 5120; const float eps = 1e-6f;
    // dequant: n rows FP8 E4M3 + E8M0 scales
    const int n = T * cols;
    std::vector<uint8_t> rows((size_t) n * hd), scales((size_t) n * (hd / 32));
    for (auto& v : rows) { do v = (uint8_t) (rnd() & 0xff); while ((v & 0x7f) == 0x7f); }
    for (auto& v : scales) v = (uint8_t) (115 + rnd() % 8);
    Dev<uint8_t> dr(rows), dsc(scales); Dev<float> dout; dout.alloc((size_t) n * hd);
    engram_dequant(dr.p, dsc.p, n, hd, dout.p, g_s); CK(cudaStreamSynchronize(g_s));
    auto o = dout.get(); size_t bad = 0;
    for (int r = 0; r < n; ++r) for (int e = 0; e < hd; ++e) { const float ref = e4m3_val(rows[(size_t) r * hd + e]) * std::ldexp(1.f, (int) scales[(size_t) r * (hd / 32) + e / 32] - 127); if (std::memcmp(&ref, &o[(size_t) r * hd + e], 4) != 0) ++bad; }
    char ex[64]; std::snprintf(ex, sizeof ex, "%zu of %zu values differ (bit exact)", bad, (size_t) n * hd);
    report("engram_dequant", T, bad ? 1.0 : 0.0, 0.0, time_us([&] { engram_dequant(dr.p, dsc.p, n, hd, dout.p, g_s); }), ex);
    // gate
    std::vector<float> h((size_t) T * hc * dim), kv((size_t) T * (hc + 1) * dim), w((size_t) hc * dim);
    for (auto& v : h) v = nrand(); for (auto& v : kv) v = nrand(); for (auto& v : w) v = 0.5f + urand();
    Dev<float> dh(h), dkv(kv), dw(w);
    engram_gate(dh.p, dkv.p, dw.p, hc, dim, T, eps, g_s); CK(cudaStreamSynchronize(g_s));
    auto go = dh.get(); std::vector<double> rr; std::vector<float> gg;
    for (int t = 0; t < T; ++t) for (int k = 0; k < hc; ++k) {
        const float* hk = &h[((size_t) t * hc + k) * dim]; const float* key = &kv[(size_t) t * (hc + 1) * dim + (size_t) k * dim]; const float* val = &kv[(size_t) t * (hc + 1) * dim + (size_t) hc * dim];
        double sh = 0, sk = 0, sd = 0; for (int e = 0; e < dim; ++e) { sh += (double) hk[e] * hk[e]; sk += (double) key[e] * key[e]; sd += (double) hk[e] * w[(size_t) k * dim + e] * key[e]; }
        const double rstd = 1.0 / std::sqrt(sh / dim + eps) / std::sqrt(sk / dim + eps);
        const double dot = sd * rstd / std::sqrt((double) dim); const double sq = std::copysign(std::sqrt(std::max(std::fabs(dot), 1e-6)), dot); const double gate = 1.0 / (1.0 + std::exp(-sq));
        for (int e = 0; e < dim; ++e) { rr.push_back(hk[e] + gate * val[e]); gg.push_back(go[((size_t) t * hc + k) * dim + e]); }
    }
    Dev<float> dh2(h);
    report("engram_gate", T, max_rel(gg, rr), 2e-6, time_us([&] { engram_gate(dh2.p, dkv.p, dw.p, hc, dim, T, eps, g_s); }), "(timing re-applies in place)");
    // the n-gram row gather: 24 hash-picked rows of 256 per token = the host/SSD side; on the device only the gather of
    // x rows (gather_rows) - T x 5120 floats
    Gather gt; for (int i = 0; i < kWin * kPicks; ++i) gt.src[i] = (int) (rnd() % T);
    Dev<float> dx; { std::vector<float> x((size_t) T * dim); for (auto& v : x) v = nrand(); dx.set(x); } Dev<float> dg; dg.alloc((size_t) T * kPicks * dim);
    gather_rows(dx.p, gt, T * kPicks > kWin * kPicks ? kWin * kPicks : T * kPicks, dim, dg.p, g_s); CK(cudaStreamSynchronize(g_s));
    auto go2 = dg.get(); size_t badg = 0; const int ng = std::min(T * kPicks, kWin * kPicks);
    std::vector<float> xs = dx.get();
    for (int r = 0; r < ng; ++r) for (int c = 0; c < dim; ++c) badg += go2[(size_t) r * dim + c] != xs[(size_t) gt.src[r] * dim + c];
    std::snprintf(ex, sizeof ex, "%zu mismatches, %d rows", badg, ng);
    report("gather_rows", T, badg ? 1.0 : 0.0, 0.0, time_us([&] { gather_rows(dx.p, gt, ng, dim, dg.p, g_s); }), ex);
}

// ============================================================================================================ swiglu + combine
static void test_swiglu(int T) {
    const int n = T * 2304 * 6; const float lim = 10.f;
    std::vector<float> g(n), u(n); for (auto& v : g) v = nrand() * 6.f; for (auto& v : u) v = nrand() * 6.f;
    Dev<float> dg(g), du(u), dh; dh.alloc(n);
    swiglu(dg.p, du.p, n, lim, dh.p, g_s); CK(cudaStreamSynchronize(g_s));
    auto h = dh.get(); std::vector<double> rr; for (int i = 0; i < n; ++i) { const double gv = std::min((double) g[i], (double) lim), uv = std::min(std::max((double) u[i], -(double) lim), (double) lim); rr.push_back(gv / (1.0 + std::exp(-gv)) * uv); }
    report("swiglu clamp 10", T, max_rel(h, rr), 3e-6, time_us([&] { swiglu(dg.p, du.p, n, lim, dh.p, g_s); }), "T x 6 experts x 2304");
}

int main(int argc, char** argv) {
    std::vector<int> Ts = {1, 2, 3, 4, 5};
    std::string only; int ctx = 8192;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--tokens" && i + 1 < argc) { Ts.clear(); for (const char* p = argv[++i]; *p;) { Ts.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; } }
        else if (a == "--only" && i + 1 < argc) only = std::string(",") + argv[++i] + ",";
        else if (a == "--ctx" && i + 1 < argc) ctx = std::atoi(argv[++i]);
        else if (a == "--reps" && i + 1 < argc) g_reps = std::atoi(argv[++i]);
        else if (a == "--dev" && i + 1 < argc) CK(cudaSetDevice(std::atoi(argv[++i])));
    }
    cudaDeviceProp prop; int dev; CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
    std::printf("device %s sm_%d%d; kernels: src/dsv41/ds4_kernels.cu unmodified; graph-replayed timings (no launch gaps)\n", prop.name, prop.major, prop.minor);
    CK(cudaStreamCreate(&g_s));
    const int ncache = ctx / 4;   // compressed rows at ratio 4
    auto want = [&](const char* n) { return only.empty() || only.find(std::string(",") + n + ",") != std::string::npos; };
    for (int T : Ts) {
        if (want("mhc")) test_mhc(T);
        if (want("router")) test_router(T);
        if (want("compress")) { test_compress(T, 4); test_compress(T, 128); }
        if (want("indexer")) test_indexer(T, ncache);
        if (want("attn")) test_attn(T, ncache);
        if (want("engram")) test_engram(T);
        if (want("swiglu")) test_swiglu(T);
        std::printf("\n");
    }
    std::printf("%s (%d failures)\n", g_fails ? "FAILED" : "ALL PARITY OK", g_fails);
    return g_fails ? 1 : 0;
}
