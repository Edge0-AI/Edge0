// Strata. The trellis routed experts of the DeepSeek-V4.1 path (ds4_tq.cu on the device, ds4_tq_cpu.cpp on the CPU lane) against
// a double reference on host-decoded weights (tools/trellis/tqenc: decode_blocks), for the six types (TQK6 / TQK7 / TQK5X /
// TQK4X / TQK5T / TQK4T): the token path, the window path (grouped entries), the prompt sets (gate/up and down) and the CPU
// lane, plus the two lanes against each other.  Random legal streams: any bit pattern is one.
//   tq_expert_test [--type 145,146,...]
#include "../../src/dsv41/ds4_gguf_cpu.hpp"
#include "../../src/dsv41/ds4_iq.cuh"
#include "../../tools/trellis/tqenc.hpp"

#include "strata/dsv41/hip_or_cuda.h"   // HIP names on a HIP build, the CUDA runtime otherwise

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace iq = strata::ds4::gguf::iq;
namespace gg = strata::ds4::gguf;
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); std::exit(2); } } while (0)

template <typename T> static T* up(const std::vector<T>& v) {
    T* d; CK(hipMalloc(&d, std::max<size_t>(1, v.size() * sizeof(T)))); CK(hipMemcpy(d, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice)); return d;
}
template <typename T> static std::vector<T> down_(const T* d, size_t n) {
    std::vector<T> v(n); CK(hipMemcpy(v.data(), d, n * sizeof(T), hipMemcpyDeviceToHost)); return v;
}
static float h2f(uint16_t h) {
    const int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
    float f = e == 0 ? std::ldexp((float) m, -24) : std::ldexp(1.0f + m / 1024.0f, e - 15);
    return s ? -f : f;
}
static uint16_t f2h(float f) {   // round to nearest even, normal range
    uint32_t u; std::memcpy(&u, &f, 4);
    const uint32_t s = (u >> 16) & 0x8000u; int e = (int) ((u >> 23) & 0xff) - 127 + 15; uint32_t m = u & 0x7fffffu;
    uint32_t hm = m >> 13; const uint32_t rem = m & 0x1fffu;
    uint32_t v = ((uint32_t) e << 10) | hm;
    if (rem > 0x1000u || (rem == 0x1000u && (v & 1))) ++v;
    return (uint16_t) (s | v);
}

struct Q8 { std::vector<int8_t> q; std::vector<float> d; std::vector<float> xhat; };
// the engine's q8_quant: sc = max|x| / 127 per 32, q = round-half-even(x / sc)
static Q8 quant(const float* x, int k) {
    Q8 r; r.q.resize(k); r.d.resize(k / 32); r.xhat.resize(k);
    for (int b = 0; b < k / 32; ++b) {
        float am = 0; for (int j = 0; j < 32; ++j) am = std::max(am, std::fabs(x[b * 32 + j]));
        const float sc = am / 127.0f; r.d[b] = sc;
        for (int j = 0; j < 32; ++j) { const int q = am > 0 ? (int) std::nearbyint(x[b * 32 + j] / sc) : 0; r.q[b * 32 + j] = (int8_t) q; r.xhat[b * 32 + j] = (float) q * sc; }
    }
    return r;
}
static double swiglu(double g, double u, double lim) {
    if (lim > 0) { g = std::min(g, lim); u = std::min(std::max(u, -lim), lim); }
    return g / (1.0 + std::exp(-g)) * u;
}

struct Mat { std::vector<uint8_t> bytes; std::vector<float> w; int rows, k; };
static Mat make_mat(std::mt19937_64& rng, int type, int K, int rows, int k) {
    Mat m; m.rows = rows; m.k = k;
    const int bsz = 2 + 4 * K, nblk = k / 128;
    m.bytes.resize((size_t) rows * nblk * bsz);
    for (auto& b : m.bytes) b = (uint8_t) rng();
    for (int64_t b = 0; b < (int64_t) rows * nblk; ++b) { const uint16_t d = f2h(0.05f + 0.0005f * (float) (rng() % 600)); std::memcpy(&m.bytes[(size_t) b * bsz], &d, 2); }
    m.w.resize((size_t) rows * k);
    tqenc::decode_blocks(type, m.bytes.data(), (int64_t) rows * nblk, m.w.data());
    return m;
}
static double dotd(const float* w, const float* x, int k, double* sabs = nullptr) {
    double s = 0, sa = 0; for (int i = 0; i < k; ++i) { const double t = (double) w[i] * x[i]; s += t; sa += std::fabs(t); } if (sabs) *sabs = sa; return s;
}

// one decode token of the DeepSeek shape (n_embd 5120, n_ff 2304, top-6 experts, distinct blobs) through the token path: gate/up +
// down launches, hipEvent time per layer; and a 5-entry verify window of one expert group
static void bench(int type) {
    const int K = iq::tq_kbits(type), E = 5120, FF = 2304, NE = 6;
    const size_t rb_g = (size_t) (E / 128) * (2 + 4 * K), rb_d = (size_t) (FF / 128) * (2 + 4 * K);
    auto al = [](size_t n) { return (n + 255) / 256 * 256; };
    const size_t off[3] = {0, al(FF * rb_g), al(FF * rb_g) * 2}, blob = off[2] + al((size_t) E * rb_d);
    std::mt19937_64 rng(5);
    std::vector<uint8_t> one(blob);
    for (auto& b : one) b = (uint8_t) rng();
    std::vector<unsigned long long> plan(NE);
    for (int e = 0; e < NE; ++e) { uint8_t* d; CK(hipMalloc(&d, blob)); CK(hipMemcpy(d, one.data(), blob, hipMemcpyHostToDevice)); plan[e] = (unsigned long long) d; }
    std::vector<int8_t> xq(E, 3), hq((size_t) NE * FF, 3); std::vector<float> xd(E / 32, 0.01f), hd((size_t) NE * FF / 32, 0.01f);
    iq::Args a{};
    a.plan_ptr = up(plan); a.n_embd = E; a.n_ff = FF; a.limit = 10.0f; a.xq = up(xq); a.xd = up(xd); a.hq = up(hq); a.hd = up(hd); a.tg = a.td = type;
    for (int r = 0; r < 3; ++r) a.tr[r] = off[r];
    float *dh, *dp; CK(hipMalloc(&dh, (size_t) NE * FF * 4)); CK(hipMalloc(&dp, (size_t) NE * E * 4)); a.h = dh; a.part = dp;
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    for (int rep = 0; rep < 5; ++rep) { iq::gate_up(a, NE, 1, 0); iq::down(a, NE, 1, 0); }
    CK(hipDeviceSynchronize());
    float ms_gu = 0, ms_dn = 0;
    for (int part = 0; part < 2; ++part) {
        hipEventRecord(e0);
        for (int rep = 0; rep < 100; ++rep) { if (part == 0) iq::gate_up(a, NE, 1, 0); else iq::down(a, NE, 1, 0); }
        hipEventRecord(e1); hipEventSynchronize(e1);
        float ms; hipEventElapsedTime(&ms, e0, e1);
        (part == 0 ? ms_gu : ms_dn) = ms / 100;
    }
    const double bytes = (double) NE * blob;
    std::printf("bench %-6s token path, 6 experts: gate/up %.0f us, down %.0f us, layer %.0f us (%.0f GB/s, %.2f Tw/s), x40 = %.1f ms per token\n", iq::name(type), ms_gu * 1e3,
                ms_dn * 1e3, (ms_gu + ms_dn) * 1e3, bytes / ((ms_gu + ms_dn) * 1e-3) / 1e9, (double) NE * 3 * FF * E / ((ms_gu + ms_dn) * 1e-3) / 1e12, 40.0 * (ms_gu + ms_dn));
}

int main(int argc, char** argv) {
    std::vector<int> types = {145, 146, 224, 225, 226, 227};
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--bench") { for (int t : {225, 224, 145, 146}) bench(t); return 0; }
        else if (std::string(argv[i]) == "--type" && i + 1 < argc) { types.clear(); for (const char* p = argv[++i]; *p;) { types.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; } }
    std::mt19937_64 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    int fails = 0;
    auto report = [&](const char* what, int type, double worst, double tol) {
        const bool ok = worst <= tol;
        std::printf("%-6s %-22s worst %.2e (tol %.0e) %s\n", iq::name(type), what, worst, tol, ok ? "ok" : "FAIL");
        fails += !ok;
    };
    for (int type : types) {
        const int K = iq::tq_kbits(type);
        const int E = 512, FF = 256, NEXP = 3, T = 4;
        const float lim = 7.0f;
        const size_t rb_g = (size_t) (E / 128) * (2 + 4 * K), rb_d = (size_t) (FF / 128) * (2 + 4 * K);
        // experts: blob = gate | up | down, each 256-aligned
        auto al = [](size_t n) { return (n + 255) / 256 * 256; };
        const size_t off[3] = {0, al(FF * rb_g), al(FF * rb_g) * 2}, blob = off[2] + al((size_t) E * rb_d);
        std::vector<Mat> G(NEXP), U(NEXP), D(NEXP);
        // h of row r of expert e for the input xv (F32 values), and the size of the operands it was made of
        auto refh = [&](int e, int r, const float* xv, double& scale) {
            double sg, su;
            const double g = dotd(&G[e].w[(size_t) r * E], xv, E, &sg), u = dotd(&U[e].w[(size_t) r * E], xv, E, &su);
            scale = sg + su + 1e-30;
            return swiglu(g, u, lim);
        };
        std::vector<uint8_t> blobs((size_t) NEXP * blob, 0);
        for (int e = 0; e < NEXP; ++e) {
            G[e] = make_mat(rng, type, K, FF, E); U[e] = make_mat(rng, type, K, FF, E); D[e] = make_mat(rng, type, K, E, FF);
            std::memcpy(&blobs[e * blob + off[0]], G[e].bytes.data(), G[e].bytes.size());
            std::memcpy(&blobs[e * blob + off[1]], U[e].bytes.data(), U[e].bytes.size());
            std::memcpy(&blobs[e * blob + off[2]], D[e].bytes.data(), D[e].bytes.size());
        }
        uint8_t* dblob = up(blobs);
        // tokens
        std::vector<float> x((size_t) T * E);
        for (auto& v : x) v = nd(rng) * (1.0f + 3.0f * (rng() % 7 == 0));
        std::vector<Q8> qx(T);
        std::vector<int8_t> xq((size_t) T * E); std::vector<float> xd((size_t) T * E / 32);
        for (int t = 0; t < T; ++t) { qx[t] = quant(&x[(size_t) t * E], E); std::memcpy(&xq[(size_t) t * E], qx[t].q.data(), E); std::memcpy(&xd[(size_t) t * E / 32], qx[t].d.data(), E / 32 * 4); }
        const int8_t* dxq = up(xq); const float* dxd = up(xd);

        // ---- token path: plan = 3 experts, input row 0
        {
            std::vector<unsigned long long> plan(NEXP);
            for (int e = 0; e < NEXP; ++e) plan[e] = (unsigned long long) (dblob + e * blob);
            iq::Args a{};
            a.plan_ptr = up(plan); a.n_embd = E; a.n_ff = FF; a.limit = lim; a.xq = dxq; a.xd = dxd; a.tg = a.td = type;
            for (int r = 0; r < 3; ++r) a.tr[r] = off[r];
            float* dh; CK(hipMalloc(&dh, (size_t) NEXP * FF * 4)); a.h = dh;
            if (!iq::gate_up(a, NEXP, 1, 0)) { std::printf("gate_up refused\n"); return 1; }
            CK(hipDeviceSynchronize());
            const auto h = down_(dh, (size_t) NEXP * FF);
            double worst = 0;
            std::vector<float> href((size_t) NEXP * FF);
            for (int e = 0; e < NEXP; ++e)
                for (int r = 0; r < FF; ++r) {
                    double sc;
                    const double ref = refh(e, r, qx[0].xhat.data(), sc);
                    href[(size_t) e * FF + r] = (float) ref;
                    worst = std::max(worst, std::fabs(h[(size_t) e * FF + r] - ref) / sc);
                }
            report("token gate_up", type, worst, 1e-5);
            // down over the reference h, quantized
            std::vector<int8_t> hq((size_t) NEXP * FF); std::vector<float> hd((size_t) NEXP * FF / 32);
            std::vector<Q8> qh(NEXP);
            for (int e = 0; e < NEXP; ++e) { qh[e] = quant(&href[(size_t) e * FF], FF); std::memcpy(&hq[(size_t) e * FF], qh[e].q.data(), FF); std::memcpy(&hd[(size_t) e * FF / 32], qh[e].d.data(), FF / 32 * 4); }
            a.hq = up(hq); a.hd = up(hd);
            float* dp; CK(hipMalloc(&dp, (size_t) NEXP * E * 4)); a.part = dp;
            if (!iq::down(a, NEXP, 1, 0)) { std::printf("down refused\n"); return 1; }
            CK(hipDeviceSynchronize());
            const auto part = down_(dp, (size_t) NEXP * E);
            worst = 0;
            for (int e = 0; e < NEXP; ++e)
                for (int r = 0; r < E; ++r) {
                    double sa; const double ref = dotd(&D[e].w[(size_t) r * FF], qh[e].xhat.data(), FF, &sa);
                    worst = std::max(worst, std::fabs(part[(size_t) e * E + r] - ref) / (sa + 1e-30));
                }
            report("token down", type, worst, 1e-6);

            // ---- the CPU lane on the same experts and inputs; against the reference and the device
            gg::IqCpu fm; std::string err;
            if (!gg::iq_cpu_fmt(type, type, E, FF, fm, err)) { std::printf("cpu fmt: %s\n", err.c_str()); ++fails; }
            else {
                std::vector<uint8_t> act(fm.act_bytes), hact(fm.h_bytes);
                gg::iq_cpu_quant(fm.gu_act, x.data(), E, act.data());
                bool same = std::memcmp(act.data(), qx[0].d.data(), E / 32 * 4) == 0 && std::memcmp(act.data() + E / 32 * 4, qx[0].q.data(), E) == 0;
                std::printf("%-6s cpu activation == device quantizer: %s\n", iq::name(type), same ? "ok" : "FAIL");
                fails += !same;
                double wg = 0, wd = 0, wx = 0;
                for (int e = 0; e < NEXP; ++e) {
                    std::vector<float> hc(FF), yc(E);
                    float* hp[1] = {hc.data()}; float* yp[1] = {yc.data()};
                    const void* ap[1] = {act.data()};
                    gg::iq_cpu_gate_up(fm, &blobs[e * blob + off[0]], &blobs[e * blob + off[1]], ap, 1, hp, 0, FF, lim);
                    for (int r = 0; r < FF; ++r) { double sc; refh(e, r, qx[0].xhat.data(), sc); wg = std::max(wg, std::fabs(hc[r] - href[(size_t) e * FF + r]) / sc); wx = std::max(wx, std::fabs(hc[r] - h[(size_t) e * FF + r]) / sc); }
                    gg::iq_cpu_quant(fm.d_act, &href[(size_t) e * FF], FF, hact.data());
                    const void* hpp[1] = {hact.data()};
                    gg::iq_cpu_down(fm, &blobs[e * blob + off[2]], hpp, 1, yp, 0, E);
                    for (int r = 0; r < E; ++r) { double sa; const double ref = dotd(&D[e].w[(size_t) r * FF], qh[e].xhat.data(), FF, &sa); wd = std::max(wd, std::fabs(yc[r] - ref) / (sa + 1e-30)); }
                }
                report("cpu gate_up vs ref", type, wg, 1e-5);
                report("cpu down vs ref", type, wd, 1e-6);
                report("cpu vs device gate_up", type, wx, 1e-5);
            }
        }

        // ---- window path: groups of entries (entry e = t * 8 + i reads input row e >> 3)
        {
            std::vector<unsigned long long> grp = {(unsigned long long) (dblob + 0 * blob), (unsigned long long) (dblob + 2 * blob)};
            std::vector<int> start = {0, 3, 5}, ent = {1, 10, 19, 2, 8}, ng = {2};
            iq::Args a{};
            a.grp_ptr = up(grp); a.grp_start = up(start); a.grp_ent = up(ent); a.n_groups = up(ng);
            a.n_embd = E; a.n_ff = FF; a.limit = lim; a.xq = dxq; a.xd = dxd; a.tg = a.td = type;
            for (int r = 0; r < 3; ++r) a.tr[r] = off[r];
            const int NE = 24;
            float* dh; CK(hipMalloc(&dh, (size_t) NE * FF * 4)); CK(hipMemset(dh, 0, (size_t) NE * FF * 4)); a.h = dh;
            iq::gate_up(a, 2, 3, 0);
            CK(hipDeviceSynchronize());
            const auto h = down_(dh, (size_t) NE * FF);
            double worst = 0;
            for (int gi = 0; gi < 2; ++gi)
                for (int j = start[gi]; j < start[gi + 1]; ++j) {
                    const int en = ent[j], t = en >> 3, e = gi == 0 ? 0 : 2;
                    for (int r = 0; r < FF; ++r) {
                        double sc;
                        const double ref = refh(e, r, qx[t].xhat.data(), sc);
                        worst = std::max(worst, std::fabs(h[(size_t) en * FF + r] - ref) / sc);
                    }
                }
            report("window gate_up", type, worst, 1e-5);
        }

        // ---- the prompt sets: expert 0 owns sorted rows [0, 37), expert 2 rows [37, 42); input rows idx[sorted]
        {
            const int NR = 42;
            std::vector<int> bounds = {0, 37, 37, 42}, idx(NR);
            for (int i = 0; i < NR; ++i) idx[i] = i % T;
            iq::SetArgs q{};
            q.wbase = dblob; q.stride = blob; for (int r = 0; r < 3; ++r) q.tr[r] = off[r];
            q.bounds = up(bounds); q.row0 = 0; q.max_rows = 37; q.n_exp = 3; q.n_embd = E; q.n_ff = FF; q.limit = lim;
            q.tg = q.td = type; q.xf = up(x); q.idx = up(idx); q.ldx = E;
            uint16_t* dh16; CK(hipMalloc(&dh16, (size_t) NR * FF * 2)); q.h = dh16; q.ldo = FF;
            iq::set_gate_up(q, 0);
            CK(hipDeviceSynchronize());
            const auto h16 = down_(dh16, (size_t) NR * FF);
            double worst = 0;
            for (int e = 0; e < 3; ++e)
                for (int lr = bounds[e]; lr < bounds[e + 1]; ++lr)
                    for (int r = 0; r < FF; ++r) {
                        double sc;
                        const double ref = refh(e, r, &x[(size_t) idx[lr] * E], sc);
                        worst = std::max(worst, std::fabs(h2f(h16[(size_t) lr * FF + r]) - ref) / sc);
                    }
            report("set gate_up (f16 out)", type, worst, 1e-3);
            // down over those f16 h rows
            q.xh = dh16; q.ldx = FF;
            float* dy; CK(hipMalloc(&dy, (size_t) NR * E * 4)); q.y = dy; q.ldo = E;
            iq::set_down(q, 0);
            CK(hipDeviceSynchronize());
            const auto y = down_(dy, (size_t) NR * E);
            worst = 0;
            for (int e = 0; e < 3; ++e)
                for (int lr = bounds[e]; lr < bounds[e + 1]; ++lr) {
                    std::vector<float> hf(FF);
                    for (int c = 0; c < FF; ++c) hf[c] = h2f(h16[(size_t) lr * FF + c]);
                    for (int r = 0; r < E; ++r) {
                        double sa; const double ref = dotd(&D[e].w[(size_t) r * FF], hf.data(), FF, &sa);
                        worst = std::max(worst, std::fabs(y[(size_t) lr * E + r] - ref) / (sa + 1e-30));
                    }
                }
            report("set down", type, worst, 1e-6);
        }
    }
    std::printf("%s\n", fails ? "FAILED" : "ALL OK");
    return fails;
}
