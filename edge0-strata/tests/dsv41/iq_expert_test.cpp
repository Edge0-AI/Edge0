// Strata. The i-quant routed experts of the DeepSeek-V4.1 path (ds4_iq.cu on the device, ds4_iq_cpu.cpp on the CPU lane)
// against ggml's own decode of the same bytes plus a double reference: the token path (gate/up, down), the window path
// (grouped entries), the prompt sets (gate/up and down), and the CPU lane on the same experts.  Random legal blocks: any
// bit pattern with a sane F16 scale is one.  This is the H2 gate for a new type (IQ2_XS first): the kernel's integer sums
// must equal ggml's weights times the same activations.
//   iq_expert_test [--type 16,17,18,21,22,23,29] [--bench] [--iq256 0|1]
#include "../../src/dsv41/ds4_gguf_cpu.hpp"
#include "../../src/dsv41/ds4_iq.cuh"

#include "strata/dsv41/hip_or_cuda.h"   // HIP names on a HIP build, the CUDA runtime otherwise

#include "ggml.h"   // ggml_get_type_traits(type)->to_float: ggml's decode of the same blocks

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
static double dotd(const float* w, const float* x, int k, double* sabs = nullptr) {
    double s = 0, sa = 0; for (int i = 0; i < k; ++i) { const double t = (double) w[i] * x[i]; s += t; sa += std::fabs(t); } if (sabs) *sabs = sa; return s;
}
// an activation buffer of a ggml type (Q8_K for the i-quant rows) decoded by ggml itself
static std::vector<float> act_values(int type, const void* act, int k) {
    std::vector<float> v((size_t) k);
    if (type == 15) {   // Q8_K: float d, int8 qs[256], int16 bsums[16] - no to_float (an intermediate type)
        const int nb = k / 256;
        const size_t bs = ggml_row_size((ggml_type) type, k) / (size_t) nb;
        for (int b = 0; b < nb; ++b) {
            const uint8_t* p = (const uint8_t*) act + (size_t) b * bs;
            float d; std::memcpy(&d, p, 4);
            const int8_t* qs = (const int8_t*) (p + 4);
            for (int j = 0; j < 256; ++j) v[(size_t) b * 256 + j] = d * (float) qs[j];
        }
        return v;
    }
    const ggml_type_traits* t = ggml_get_type_traits((ggml_type) type);
    if (t == nullptr || t->to_float == nullptr) { std::printf("no to_float for activation type %d\n", type); std::exit(2); }
    t->to_float(act, v.data(), k);
    return v;
}

struct Mat { std::vector<uint8_t> bytes; std::vector<float> w; int rows = 0, k = 0; };
// random legal blocks of `type` for a rows x k matrix, decoded by ggml itself
static Mat make_mat(std::mt19937_64& rng, int type, int rows, int k) {
    Mat m; m.rows = rows; m.k = k;
    const int bv = iq::block_values(type), bb = iq::block_bytes(type);
    const size_t nblk = (size_t) rows * (k / bv);
    m.bytes.resize(nblk * (size_t) bb);
    for (auto& b : m.bytes) b = (uint8_t) rng();
    for (size_t b = 0; b < nblk; ++b) {   // a sane F16 scale: at offset 0 (every type here but IQ1_M has d there)
        const uint16_t d = f2h(0.02f + 0.0002f * (float) (rng() % 600));
        uint8_t* p = &m.bytes[b * (size_t) bb];
        if (type == 29) {   // block_iq1_m has no d field: its F16 scale is the top nibbles of its four u16 scales
            p[49] = (uint8_t) ((p[49] & 0x0f) | ((d & 0x000f) << 4));
            p[51] = (uint8_t) ((p[51] & 0x0f) | (d & 0x00f0));
            p[53] = (uint8_t) ((p[53] & 0x0f) | ((d & 0x0f00) >> 4));
            p[55] = (uint8_t) ((p[55] & 0x0f) | ((d & 0xf000) >> 8));
        } else {
            std::memcpy(p, &d, 2);
        }
    }
    m.w.resize((size_t) rows * k);
    const ggml_type_traits* tr = ggml_get_type_traits((ggml_type) type);
    if (tr == nullptr || tr->to_float == nullptr) { std::printf("ggml has no to_float for type %d\n", type); std::exit(2); }
    const size_t rb = (size_t) (k / bv) * (size_t) bb;
    for (int r = 0; r < rows; ++r) tr->to_float(&m.bytes[(size_t) r * rb], &m.w[(size_t) r * k], k);
    return m;
}

// one decode token of the DeepSeek shape (n_embd 5120, n_ff 2304, top-6 experts): gate/up + down launches, per-layer time
static void bench(int type) {
    const int E = 5120, FF = 2304, NE = 6;
    const size_t rb_g = iq::row_bytes(type, E), rb_d = iq::row_bytes(type, FF);
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
    std::printf("bench %-7s 6 experts x 40 layers: gate/up %.0f us, down %.0f us, layer %.0f us -> %.0f GB/s, %.2f Tw/s, 40 layers %.1f ms/token\n",
                iq::name(type), ms_gu * 1e3, ms_dn * 1e3, (ms_gu + ms_dn) * 1e3,
                bytes / ((ms_gu + ms_dn) * 1e-3) / 1e9,
                (double) NE * 3 * FF * E / ((ms_gu + ms_dn) * 1e-3) / 1e12, 40.0 * (ms_gu + ms_dn));
}


// --dump F / --cmp F: every device output of a run as raw bits, to a file / against one (a kernel change that keeps the sums
// exact and the float order must leave all of them identical)
static FILE* g_dump = nullptr; static FILE* g_cmp = nullptr; static long g_cmp_bad = 0, g_cmp_n = 0;
static void bits(const char* what, int type, const void* p, size_t bytes) {
    if (!g_dump && !g_cmp) return;
    if (g_dump) { std::fwrite(p, 1, bytes, g_dump); return; }
    std::vector<uint8_t> ref(bytes);
    if (std::fread(ref.data(), 1, bytes, g_cmp) != bytes) { std::printf("cmp: golden file too short at %s %s\n", iq::name(type), what); ++g_cmp_bad; return; }
    ++g_cmp_n;
    if (std::memcmp(ref.data(), p, bytes) != 0) {
        size_t bad = 0, first = 0; for (size_t i = 0; i < bytes; ++i) if (ref[i] != ((const uint8_t*) p)[i]) { if (!bad) first = i; ++bad; }
        std::printf("%-7s %-22s BITS DIFFER: %zu of %zu bytes (first at %zu)\n", iq::name(type), what, bad, bytes, first);
        ++g_cmp_bad;
    }
}
// the real shapes (n_embd 5120, n_ff 2304): the plan path (6 groups of one entry) and the grouped path (windows of 3 and 2)
static void real_shape_bits(int type) {
    std::mt19937_64 rng(1000 + type);   // (its own stream: the main loop's random experts are the same with or without this)
    const int E = 5120, FF = 2304, NG = 2, NEnt = 12;
    const size_t rb_g = iq::row_bytes(type, E), rb_d = iq::row_bytes(type, FF);
    auto al = [](size_t n) { return (n + 255) / 256 * 256; };
    const size_t off[3] = {0, al(FF * rb_g), al(FF * rb_g) * 2}, blob = off[2] + al((size_t) E * rb_d);
    std::vector<uint8_t> hb((size_t) NG * blob);
    for (auto& b : hb) b = (uint8_t) rng();
    for (size_t b = 0; b < hb.size() / (size_t) iq::block_bytes(type); ++b) {   // sane F16 scales (the same placement as make_mat)
        const uint16_t d = f2h(0.02f + 0.0002f * (float) (rng() % 600));
        uint8_t* p = &hb[b * (size_t) iq::block_bytes(type)];
        if (type == 29) { p[49] = (uint8_t) ((p[49] & 0x0f) | ((d & 0x000f) << 4)); p[51] = (uint8_t) ((p[51] & 0x0f) | (d & 0x00f0)); p[53] = (uint8_t) ((p[53] & 0x0f) | ((d & 0x0f00) >> 4)); p[55] = (uint8_t) ((p[55] & 0x0f) | ((d & 0xf000) >> 8)); }
        else std::memcpy(p, &d, 2);
    }
    uint8_t* dblob = up(hb);
    std::vector<int8_t> xq(2 * E), hq((size_t) NEnt * FF); std::vector<float> xd(2 * E / 32), hd((size_t) NEnt * FF / 32);
    for (auto& v : xq) v = (int8_t) ((int) (rng() % 255) - 127);
    for (auto& v : hq) v = (int8_t) ((int) (rng() % 255) - 127);
    for (auto& v : xd) v = 0.002f + 0.0001f * (float) (rng() % 100);
    for (auto& v : hd) v = 0.002f + 0.0001f * (float) (rng() % 100);
    iq::Args a{};
    a.n_embd = E; a.n_ff = FF; a.limit = 10.0f; a.xq = up(xq); a.xd = up(xd); a.hq = up(hq); a.hd = up(hd); a.tg = a.td = type;
    for (int r = 0; r < 3; ++r) a.tr[r] = off[r];
    float *dh, *dp; CK(hipMalloc(&dh, (size_t) NEnt * FF * 4)); CK(hipMalloc(&dp, (size_t) NEnt * E * 4)); a.h = dh; a.part = dp;
    std::vector<unsigned long long> plan = {(unsigned long long) dblob, (unsigned long long) (dblob + blob), 0ull, (unsigned long long) dblob};
    a.plan_ptr = up(plan);
    CK(hipMemset(dh, 0, (size_t) NEnt * FF * 4)); CK(hipMemset(dp, 0, (size_t) NEnt * E * 4));
    if (!iq::gate_up(a, 4, 1, 0) || !iq::down(a, 4, 1, 0)) { std::printf("real shape: refused\n"); return; }
    CK(hipDeviceSynchronize());
    { auto h = down_(dh, (size_t) NEnt * FF); bits("real plan h", type, h.data(), h.size() * 4); auto p = down_(dp, (size_t) NEnt * E); bits("real plan part", type, p.data(), p.size() * 4); }
    std::vector<unsigned long long> gp = {(unsigned long long) dblob, (unsigned long long) (dblob + blob)};
    std::vector<int> gs = {0, 3, 5}, ge = {0, 1, 2, 3, 4}, ng = {NG};
    a.plan_ptr = nullptr; a.grp_ptr = up(gp); a.grp_start = up(gs); a.grp_ent = up(ge); a.n_groups = up(ng);
    CK(hipMemset(dh, 0, (size_t) NEnt * FF * 4)); CK(hipMemset(dp, 0, (size_t) NEnt * E * 4));
    if (!iq::gate_up(a, NG, 3, 0) || !iq::down(a, NG, 3, 0)) { std::printf("real shape grouped: refused\n"); return; }
    CK(hipDeviceSynchronize());
    { auto h = down_(dh, (size_t) NEnt * FF); bits("real grp h", type, h.data(), h.size() * 4); auto p = down_(dp, (size_t) NEnt * E); bits("real grp part", type, p.data(), p.size() * 4); }
    // more windows: a bound of 7 groups with 3 live (dynamic count), 5/1/3 entries (ne_max 5: the 8-entry kernels), then 9 groups of
    // one entry with 4 live; entries index the h/hq rows, their input row is entry >> 3
    {
        std::vector<unsigned long long> gp2 = {(unsigned long long) dblob, (unsigned long long) (dblob + blob), (unsigned long long) dblob, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull};
        std::vector<int> gs2 = {0, 5, 6, 9, 9, 9, 9, 9, 9, 9}, ge2 = {3, 1, 4, 0, 2, 7, 5, 6, 8, 11, 10, 9}, ng2 = {3};
        a.grp_ptr = up(gp2); a.grp_start = up(gs2); a.grp_ent = up(ge2); a.n_groups = up(ng2);
        CK(hipMemset(dh, 0, (size_t) NEnt * FF * 4)); CK(hipMemset(dp, 0, (size_t) NEnt * E * 4));
        if (!iq::gate_up(a, 7, 5, 0) || !iq::down(a, 7, 5, 0)) { std::printf("real shape window 2: refused\n"); return; }
        CK(hipDeviceSynchronize());
        { auto h = down_(dh, (size_t) NEnt * FF); bits("real win2 h", type, h.data(), h.size() * 4); auto p = down_(dp, (size_t) NEnt * E); bits("real win2 part", type, p.data(), p.size() * 4); }
        std::vector<int> gs3 = {0, 1, 2, 3, 4, 4, 4, 4, 4, 4}, ng3 = {4};
        std::vector<unsigned long long> gp3 = {(unsigned long long) (dblob + blob), (unsigned long long) dblob, (unsigned long long) (dblob + blob), (unsigned long long) dblob, 0ull, 0ull, 0ull, 0ull, 0ull};
        a.grp_ptr = up(gp3); a.grp_start = up(gs3); a.n_groups = up(ng3);
        CK(hipMemset(dh, 0, (size_t) NEnt * FF * 4)); CK(hipMemset(dp, 0, (size_t) NEnt * E * 4));
        if (!iq::gate_up(a, 9, 1, 0) || !iq::down(a, 9, 1, 0)) { std::printf("real shape window 3: refused\n"); return; }
        CK(hipDeviceSynchronize());
        { auto h = down_(dh, (size_t) NEnt * FF); bits("real win3 h", type, h.data(), h.size() * 4); auto p = down_(dp, (size_t) NEnt * E); bits("real win3 part", type, p.data(), p.size() * 4); }
    }
    CK(hipFree(dblob));
}

// the same DeepSeek decode token over a working set far past L2 (>= 640 MiB of distinct expert blobs, rotated launch by launch),
// so the rate is DRAM's, not L2's: gate/up and down timed apart; ne = entries a group (1: the decode token)
static void bench_big(int type, int mib) {
    const int E = 5120, FF = 2304, NE = 6;
    const size_t rb_g = iq::row_bytes(type, E), rb_d = iq::row_bytes(type, FF);
    auto al = [](size_t n) { return (n + 255) / 256 * 256; };
    const size_t off[3] = {0, al(FF * rb_g), al(FF * rb_g) * 2}, blob = off[2] + al((size_t) E * rb_d);
    const int L = std::max(2, (int) (((size_t) mib << 20) / (NE * blob)) + 1);   // layers rotated
    std::mt19937_64 rng(5);
    std::vector<uint8_t> one(blob);
    for (auto& b : one) b = (uint8_t) rng();
    std::vector<unsigned long long> plan((size_t) L * NE);
    for (auto& p : plan) { uint8_t* d; CK(hipMalloc(&d, blob)); CK(hipMemcpy(d, one.data(), blob, hipMemcpyHostToDevice)); p = (unsigned long long) d; }
    const int NW = std::getenv("DS4_BENCH_WIN") ? std::max(1, std::min(16, std::atoi(std::getenv("DS4_BENCH_WIN")))) : 0;   // window path: NW entries a group
    const int NEN = std::max(1, NW) * NE;   // entries
    std::vector<int8_t> xq((size_t) 8 * E, 3), hq((size_t) NEN * FF, 3); std::vector<float> xd((size_t) 8 * E / 32, 0.01f), hd((size_t) NEN * FF / 32, 0.01f);
    for (size_t i = 0; i < xq.size(); ++i) xq[i] = (int8_t) ((int) (rng() % 255) - 127);
    for (size_t i = 0; i < hq.size(); ++i) hq[i] = (int8_t) ((int) (rng() % 255) - 127);
    if (const char* e = std::getenv("DS4_BENCH_ACTIVE")) { const int na = std::atoi(e); for (int l = 0; l < L; ++l) for (int k = na; k < NE; ++k) plan[(size_t) l * NE + k] = 0; }   // (empty plan entries)
    unsigned long long* dplan = up(plan);
    iq::Args a{};
    a.n_embd = E; a.n_ff = FF; a.limit = 10.0f; a.xq = up(xq); a.xd = up(xd); a.hq = up(hq); a.hd = up(hd); a.tg = a.td = type;
    for (int r = 0; r < 3; ++r) a.tr[r] = off[r];
    float *dh, *dp; CK(hipMalloc(&dh, (size_t) NEN * FF * 4)); CK(hipMalloc(&dp, (size_t) NEN * E * 4)); a.h = dh; a.part = dp;
    std::vector<int> gstart(NE + 1), gent(NEN), ngrp(1, NE);
    for (int g = 0; g <= NE; ++g) gstart[g] = g * std::max(1, NW);
    for (int i = 0; i < NEN; ++i) gent[i] = i;
    a.grp_start = up(gstart); a.grp_ent = up(gent); a.n_groups = up(ngrp);
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    double us[2] = {0, 0};
    for (int part = 0; part < 2; ++part) {
        auto run = [&](int reps) {
            hipEventRecord(e0);
            for (int rep = 0; rep < reps; ++rep) {
                if (NW) a.grp_ptr = dplan + (size_t) (rep % L) * NE; else a.plan_ptr = dplan + (size_t) (rep % L) * NE;
                if (part == 0) iq::gate_up(a, NE, std::max(1, NW), 0); else iq::down(a, NE, std::max(1, NW), 0);
            }
            hipEventRecord(e1); hipEventSynchronize(e1);
            float ms; hipEventElapsedTime(&ms, e0, e1);
            return (double) ms;
        };
        double ms = 0; int reps = 2 * L;
        while (ms < 300.0) { ms = run(reps); reps *= 2; }   // warm-up: clocks ramp, ~300 ms
        const int n = std::max(L, (int) (reps / 2 * 40.0 / ms));   // ~40 ms a window
        double best = 1e30;
        for (int w = 0; w < 7; ++w) best = std::min(best, run(n) * 1e3 / n);   // the best of 7 windows (clocks/thermals wander)
        us[part] = best;
    }
    const double bg = (double) NE * (off[2]) , bd = (double) NE * al((size_t) E * rb_d);   // bytes read a launch
    std::printf("big %-7s ws %4.0f MiB: gate/up %6.1f us (%5.0f GB/s)  down %6.1f us (%5.0f GB/s)  layer %6.1f us -> %5.0f GB/s, %.2f Tw/s\n",
                iq::name(type), (double) L * NE * blob / 1048576.0, us[0], bg / us[0] * 1e-3, us[1], bd / us[1] * 1e-3, us[0] + us[1],
                (bg + bd) / (us[0] + us[1]) * 1e-3, (double) NE * 3 * FF * E / ((us[0] + us[1]) * 1e-6) / 1e12);
    for (auto p : plan) if (p) hipFree((void*) p);
}

// the light sets (experts of <= 16 rows: DSpark verify windows, short prompts): n_exp experts of `rows` rows each, gate/up and down
static void bench_light(int type, int nexp, int rows, bool tiles = false) {
    const int E = 5120, FF = 2304;
    const size_t rb_g = iq::row_bytes(type, E), rb_d = iq::row_bytes(type, FF);
    auto al = [](size_t n) { return (n + 255) / 256 * 256; };
    const size_t off[3] = {0, al(FF * rb_g), al(FF * rb_g) * 2}, blob = off[2] + al((size_t) E * rb_d);
    std::mt19937_64 rng(7);
    std::vector<uint8_t> one(blob);
    for (auto& b : one) b = (uint8_t) rng();
    uint8_t* w; CK(hipMalloc(&w, blob * nexp));
    for (int e = 0; e < nexp; ++e) CK(hipMemcpy(w + (size_t) e * blob, one.data(), blob, hipMemcpyHostToDevice));
    const int NR = nexp * rows, T = 8;
    if (tiles && rows <= 16) { std::printf("tiles: rows > 16 only\n"); return; }
    std::vector<int> bounds(nexp + 1), idx(NR);
    for (int e = 0; e <= nexp; ++e) bounds[e] = e * rows;
    for (int i = 0; i < NR; ++i) idx[i] = i % T;
    std::vector<float> x((size_t) T * E);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) v = nd(rng);
    std::vector<uint16_t> h((size_t) NR * FF, 0x3c00);
    iq::SetArgs q{};
    q.wbase = w; q.stride = blob; for (int r = 0; r < 3; ++r) q.tr[r] = off[r];
    q.bounds = up(bounds); q.row0 = 0; q.max_rows = rows; q.skip_le = 0; q.n_exp = nexp; q.n_embd = E; q.n_ff = FF; q.limit = 7.0f;
    q.tg = q.td = type; q.xf = up(x); q.idx = up(idx); q.ldx = E;
    uint16_t* dh; CK(hipMalloc(&dh, (size_t) NR * FF * 2)); q.h = dh; q.ldo = FF;
    q.xh = up(h);
    float* dy; CK(hipMalloc(&dy, (size_t) NR * E * 4));
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    double us[2] = {0, 0};
    for (int part = 0; part < 2; ++part) {
        auto run = [&](int reps) {
            hipEventRecord(e0);
            for (int rep = 0; rep < reps; ++rep) {
                if (part == 0) { q.h = dh; q.ldo = FF; if (tiles) iq::set_gate_up(q, 0); else iq::light_gate_up(q, 0); }
                else { q.y = dy; q.ldo = E; q.ldx = FF; if (tiles) iq::set_down(q, 0); else iq::light_down(q, 0); }
            }
            hipEventRecord(e1); hipEventSynchronize(e1);
            float ms; hipEventElapsedTime(&ms, e0, e1);
            return (double) ms;
        };
        double ms = 0; int reps = 4;
        while (ms < 200.0) { ms = run(reps); reps *= 2; }
        const int n = std::max(2, (int) (reps / 2 * 40.0 / ms));
        double best = 1e30;
        for (int wnd = 0; wnd < 5; ++wnd) best = std::min(best, run(n) * 1e3 / n);
        us[part] = best;
    }
    const double bytes = (double) nexp * (off[2] + al((size_t) E * rb_d));
    std::printf("%s %-7s %3d experts x %2d rows: gate/up %7.1f us  down %7.1f us  -> %6.0f GB/s of weights, %.2f Tw/s, %.1f TMAC/s\n", tiles ? "tiles" : "light", iq::name(type), nexp, rows,
                us[0], us[1], bytes / ((us[0] + us[1]) * 1e-6) / 1e9, (double) nexp * 3 * FF * E / ((us[0] + us[1]) * 1e-6) / 1e12, (double) NR * 3 * FF * E / ((us[0] + us[1]) * 1e-6) / 1e12);
    hipFree(w);
}

int main(int argc, char** argv) {
    std::vector<int> types = {16, 17, 22, 18, 21, 23, 29};
    std::vector<int> devs;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--bench-tiles") {   // [types] [n_exp] [rows list]: the prompt sets' tensor-core tiles
            std::vector<int> bt, rl = {32, 128};
            int nexp = 48;
            if (i + 1 < argc && argv[i + 1][0] != '-') { for (const char* p = argv[++i]; *p;) { bt.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; } }
            else bt = types;
            if (i + 1 < argc && argv[i + 1][0] != '-') nexp = std::atoi(argv[++i]);
            if (i + 1 < argc && argv[i + 1][0] != '-') { rl.clear(); for (const char* p = argv[++i]; *p;) { rl.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; } }
            for (int t : bt) for (int r : rl) bench_light(t, nexp, r, true);
            return 0;
        } else if (std::string(argv[i]) == "--bench-light") {   // [types] [n_exp] [rows list]
            std::vector<int> bt, rl = {1, 2, 4};
            int nexp = 24;
            if (i + 1 < argc && argv[i + 1][0] != '-') { for (const char* p = argv[++i]; *p;) { bt.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; } }
            else bt = types;
            if (i + 1 < argc && argv[i + 1][0] != '-') nexp = std::atoi(argv[++i]);
            if (i + 1 < argc && argv[i + 1][0] != '-') { rl.clear(); for (const char* p = argv[++i]; *p;) { rl.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; } }
            for (int t : bt) for (int r : rl) bench_light(t, nexp, r);
            return 0;
        } else if (std::string(argv[i]) == "--bench-big") {   // [types] [MiB]
            std::vector<int> bt; int mib = 640;
            if (i + 1 < argc && argv[i + 1][0] != '-') { for (const char* p = argv[++i]; *p;) { bt.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; } }
            else bt = types;
            if (i + 1 < argc && argv[i + 1][0] != '-') mib = std::atoi(argv[++i]);
            for (int t : bt) bench_big(t, mib);
            return 0;
        } else if (std::string(argv[i]) == "--bench") {
            std::vector<int> bt;
            if (i + 1 < argc && argv[i + 1][0] != '-') { for (const char* p = argv[++i]; *p;) { bt.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; } }
            else bt = types;
            for (int t : bt) bench(t);
            return 0;
        } else if (std::string(argv[i]) == "--devices" && i + 1 < argc) {   // every check on each device in turn, in this process;
            devs.clear(); for (const char* p = argv[++i]; *p;) { devs.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; }   // the later ones bit-compared to the first
        } else if (std::string(argv[i]) == "--dump" && i + 1 < argc) {
            g_dump = std::fopen(argv[++i], "wb"); if (!g_dump) { std::printf("cannot write %s\n", argv[i]); return 2; }
        } else if (std::string(argv[i]) == "--cmp" && i + 1 < argc) {
            g_cmp = std::fopen(argv[++i], "rb"); if (!g_cmp) { std::printf("cannot read %s\n", argv[i]); return 2; }
        } else if (std::string(argv[i]) == "--type" && i + 1 < argc) {
            types.clear(); for (const char* p = argv[++i]; *p;) { types.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; }
        } else if (std::string(argv[i]) == "--iq256" && i + 1 < argc) {
            gg::gg_cpu_set_iq256(std::atoi(argv[++i]));
        }
    }
    int fails = 0;
    if (devs.size() > 1) {   // the first device writes its outputs, the others compare to them
        if (g_dump || g_cmp) { std::printf("--devices takes neither --dump nor --cmp\n"); return 2; }
    }
    for (size_t di = 0; di < std::max<size_t>(1, devs.size()); ++di) {
    if (!devs.empty()) {
        CK(hipSetDevice(devs[di]));
        if (devs.size() > 1) {
            if (g_dump) { std::fclose(g_dump); g_dump = nullptr; }
            if (di == 0) g_dump = std::fopen("iq_devices.bin", "wb"); else { g_cmp = std::fopen("iq_devices.bin", "rb"); g_cmp_n = g_cmp_bad = 0; }
        }
        std::printf("---- device %d\n", devs[di]);
    }
    std::mt19937_64 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto report = [&](const char* what, int type, double worst, double tol) {
        const bool ok = worst <= tol;
        std::printf("%-7s %-22s worst %.2e (tol %.0e) %s\n", iq::name(type), what, worst, tol, ok ? "ok" : "FAIL");
        fails += !ok;
    };
    for (int type : types) {
        if (!iq::supported(type) && !iq::known(type)) { std::printf("%-7s not a type this engine knows\n", iq::name(type)); ++fails; continue; }
        const bool dev = iq::supported(type);   // (IQ1_S / IQ1_M / IQ4_NL: a known layout, no device kernel - CPU lane only)
        const int E = 512, FF = 256, NEXP = 3, T = 4;
        const float lim = 7.0f;
        const size_t rb_g = iq::row_bytes(type, E), rb_d = iq::row_bytes(type, FF);
        auto al = [](size_t n) { return (n + 255) / 256 * 256; };
        const size_t off[3] = {0, al(FF * rb_g), al(FF * rb_g) * 2}, blob = off[2] + al((size_t) E * rb_d);
        std::vector<Mat> G(NEXP), U(NEXP), D(NEXP);
        auto refh = [&](int e, int r, const float* xv, double& scale) {
            double sg, su;
            const double g = dotd(&G[e].w[(size_t) r * E], xv, E, &sg), u = dotd(&U[e].w[(size_t) r * E], xv, E, &su);
            scale = sg + su + 1e-30;
            return swiglu(g, u, lim);
        };
        std::vector<uint8_t> blobs((size_t) NEXP * blob, 0);
        for (int e = 0; e < NEXP; ++e) {
            G[e] = make_mat(rng, type, FF, E); U[e] = make_mat(rng, type, FF, E); D[e] = make_mat(rng, type, E, FF);
            std::memcpy(&blobs[e * blob + off[0]], G[e].bytes.data(), G[e].bytes.size());
            std::memcpy(&blobs[e * blob + off[1]], U[e].bytes.data(), U[e].bytes.size());
            std::memcpy(&blobs[e * blob + off[2]], D[e].bytes.data(), D[e].bytes.size());
        }
        uint8_t* dblob = up(blobs);
        std::vector<float> x((size_t) T * E);
        for (auto& v : x) v = nd(rng) * (1.0f + 3.0f * (rng() % 7 == 0));
        std::vector<Q8> qx(T);
        std::vector<int8_t> xq((size_t) T * E); std::vector<float> xd((size_t) T * E / 32);
        for (int t = 0; t < T; ++t) { qx[t] = quant(&x[(size_t) t * E], E); std::memcpy(&xq[(size_t) t * E], qx[t].q.data(), E); std::memcpy(&xd[(size_t) t * E / 32], qx[t].d.data(), E / 32 * 4); }
        const int8_t* dxq = up(xq); const float* dxd = up(xd);

        // ---- the reference h of every expert's rows for token 0 (the CPU lane and the device share it)
        std::vector<float> href((size_t) NEXP * FF);
        std::vector<Q8> qh(NEXP);
        for (int e = 0; e < NEXP; ++e)
            for (int r = 0; r < FF; ++r) {
                double sc;
                href[(size_t) e * FF + r] = (float) refh(e, r, qx[0].xhat.data(), sc);
            }
        for (int e = 0; e < NEXP; ++e) qh[e] = quant(&href[(size_t) e * FF], FF);

        // ---- token path (device; skipped for a type with no device kernel)
        if (dev) {
            std::vector<unsigned long long> plan(NEXP);
            for (int e = 0; e < NEXP; ++e) plan[e] = (unsigned long long) (dblob + e * blob);
            iq::Args a{};
            a.plan_ptr = up(plan); a.n_embd = E; a.n_ff = FF; a.limit = lim; a.xq = dxq; a.xd = dxd; a.tg = a.td = type;
            for (int r = 0; r < 3; ++r) a.tr[r] = off[r];
            float* dh; CK(hipMalloc(&dh, (size_t) NEXP * FF * 4)); a.h = dh;
            if (!iq::gate_up(a, NEXP, 1, 0)) { std::printf("gate_up refused\n"); return 1; }
            CK(hipDeviceSynchronize());
            const auto h = down_(dh, (size_t) NEXP * FF);
            bits("token h", type, h.data(), h.size() * 4);
            double worst = 0;
            for (int e = 0; e < NEXP; ++e)
                for (int r = 0; r < FF; ++r) {
                    double sc;
                    const double ref = refh(e, r, qx[0].xhat.data(), sc);
                    worst = std::max(worst, std::fabs(h[(size_t) e * FF + r] - ref) / sc);
                }
            report("token gate_up", type, worst, 1e-5);
            std::vector<int8_t> hq((size_t) NEXP * FF); std::vector<float> hd((size_t) NEXP * FF / 32);
            for (int e = 0; e < NEXP; ++e) { std::memcpy(&hq[(size_t) e * FF], qh[e].q.data(), FF); std::memcpy(&hd[(size_t) e * FF / 32], qh[e].d.data(), FF / 32 * 4); }
            a.hq = up(hq); a.hd = up(hd);
            float* dp; CK(hipMalloc(&dp, (size_t) NEXP * E * 4)); a.part = dp;
            if (!iq::down(a, NEXP, 1, 0)) { std::printf("down refused\n"); return 1; }
            CK(hipDeviceSynchronize());
            const auto part = down_(dp, (size_t) NEXP * E);
            bits("token part", type, part.data(), part.size() * 4);
            worst = 0;
            for (int e = 0; e < NEXP; ++e)
                for (int r = 0; r < E; ++r) {
                    double sa; const double ref = dotd(&D[e].w[(size_t) r * FF], qh[e].xhat.data(), FF, &sa);
                    worst = std::max(worst, std::fabs(part[(size_t) e * E + r] - ref) / (sa + 1e-30));
                }
            report("token down", type, worst, 1e-6);

            // ---- the prompt sets: expert 0 owns rows [0, 37), expert 2 [37, 42)
            const int NR = 42;
            std::vector<int> bounds = {0, 37, 37, 42}, idx(NR);
            for (int i = 0; i < NR; ++i) idx[i] = i % T;
            iq::SetArgs q{};
            q.wbase = dblob; q.stride = blob; for (int r = 0; r < 3; ++r) q.tr[r] = off[r];
            q.bounds = up(bounds); q.row0 = 0; q.max_rows = 37; q.n_exp = 3; q.n_embd = E; q.n_ff = FF; q.limit = lim;
            q.tg = q.td = type; q.xf = up(x); q.idx = up(idx); q.ldx = E;
            uint16_t* dh16; CK(hipMalloc(&dh16, (size_t) NR * FF * 2)); q.h = dh16; q.ldo = FF;
            if (!iq::set_gate_up(q, 0)) { std::printf("set_gate_up refused\n"); return 1; }
            CK(hipDeviceSynchronize());
            const auto h16 = down_(dh16, (size_t) NR * FF);
            bits("set h16", type, h16.data(), h16.size() * 2);
            double w2 = 0;
            for (int e = 0; e < 3; ++e)
                for (int lr = bounds[e]; lr < bounds[e + 1]; ++lr)
                    for (int r = 0; r < FF; ++r) {
                        double sc;
                        const double ref = refh(e, r, &x[(size_t) idx[lr] * E], sc);
                        w2 = std::max(w2, std::fabs((double) h2f(h16[(size_t) lr * FF + r]) - ref) / sc);
                    }
            // (the prompt sets quantise the activations, so this worst case is data dependent: IQ1_S measured 7.1e-3 .. 1.08e-2 over
            // different random-data orders, the same on the older build, with every other check far inside; IQ1_S gets 1.5e-2)
            report("set gate_up (f16)", type, w2, type == 19 ? 1.5e-2 : 1e-2);
            q.xh = dh16; q.ldx = FF;
            float* dy; CK(hipMalloc(&dy, (size_t) NR * E * 4)); q.y = dy; q.ldo = E;
            if (!iq::set_down(q, 0)) { std::printf("set_down refused\n"); return 1; }
            CK(hipDeviceSynchronize());
            const auto y = down_(dy, (size_t) NR * E);
            bits("set y", type, y.data(), y.size() * 4);
            real_shape_bits(type);
            w2 = 0;
            for (int e = 0; e < 3; ++e)
                for (int lr = bounds[e]; lr < bounds[e + 1]; ++lr) {
                    std::vector<float> hf(FF);
                    for (int c = 0; c < FF; ++c) hf[c] = h2f(h16[(size_t) lr * FF + c]);
                    for (int r = 0; r < E; ++r) {
                        double sa; const double ref = dotd(&D[e].w[(size_t) r * FF], hf.data(), FF, &sa);
                        w2 = std::max(w2, std::fabs(y[(size_t) lr * E + r] - ref) / (sa + 1e-30));
                    }
                }
            // (the prompt sets' tiles take F16 operands for the i-quants: the tile's own rounding, not the token path's)
            report("set down", type, w2, 3e-3);

            // ---- the matrix-core tiles (HIP gfx11 / gfx12) forced on, against the scalar tiles forced on the same inputs: each within the
            // tolerances above of the CPU reference, the two within the same tolerance of each other, and the forced-on run must really
            // have launched them (STRATA_DS4_HIPK is not needed)
            if (iq::hipk_force(1)) {
                const long long n0 = iq::hipk_launches();
                iq::SetArgs qm = q;
                uint16_t* mh; CK(hipMalloc(&mh, (size_t) NR * FF * 2)); float* my; CK(hipMalloc(&my, (size_t) NR * E * 4));
                qm.h = mh; qm.ldx = E; qm.ldo = FF; qm.xh = nullptr; qm.y = nullptr;
                if (!iq::set_gate_up(qm, 0)) { std::printf("matrix-core set_gate_up refused\n"); return 1; }
                qm.xh = mh; qm.ldx = FF; qm.y = my; qm.ldo = E;
                if (!iq::set_down(qm, 0)) { std::printf("matrix-core set_down refused\n"); return 1; }
                CK(hipDeviceSynchronize());
                const long long taken = iq::hipk_launches() - n0;
                iq::hipk_force(0);
                const long long n1 = iq::hipk_launches();
                uint16_t* sh; CK(hipMalloc(&sh, (size_t) NR * FF * 2)); float* sy; CK(hipMalloc(&sy, (size_t) NR * E * 4));
                iq::SetArgs qs = q;
                qs.h = sh; qs.ldx = E; qs.ldo = FF; qs.xh = nullptr; qs.y = nullptr;
                if (!iq::set_gate_up(qs, 0)) { std::printf("scalar set_gate_up refused\n"); return 1; }
                qs.xh = sh; qs.ldx = FF; qs.y = sy; qs.ldo = E;
                if (!iq::set_down(qs, 0)) { std::printf("scalar set_down refused\n"); return 1; }
                CK(hipDeviceSynchronize());
                const long long scal = iq::hipk_launches() - n1;
                iq::hipk_force(-1);
                const auto hm = down_(mh, (size_t) NR * FF), hs = down_(sh, (size_t) NR * FF);
                const auto ym = down_(my, (size_t) NR * E), ys = down_(sy, (size_t) NR * E);
                double wg = 0, wd = 0, wx = 0, wy = 0;
                for (int e = 0; e < 3; ++e)
                    for (int lr = bounds[e]; lr < bounds[e + 1]; ++lr) {
                        std::vector<float> hf(FF);
                        for (int c = 0; c < FF; ++c) hf[c] = h2f(hm[(size_t) lr * FF + c]);
                        for (int r = 0; r < FF; ++r) {
                            double sc;
                            const double ref = refh(e, r, &x[(size_t) idx[lr] * E], sc);
                            wg = std::max(wg, std::fabs((double) h2f(hm[(size_t) lr * FF + r]) - ref) / sc);
                            wx = std::max(wx, std::fabs((double) h2f(hm[(size_t) lr * FF + r]) - (double) h2f(hs[(size_t) lr * FF + r])) / sc);
                        }
                        for (int r = 0; r < E; ++r) {
                            double sa; const double ref = dotd(&D[e].w[(size_t) r * FF], hf.data(), FF, &sa);
                            wd = std::max(wd, std::fabs(ym[(size_t) lr * E + r] - ref) / (sa + 1e-30));
                            wy = std::max(wy, std::fabs((double) ym[(size_t) lr * E + r] - (double) ys[(size_t) lr * E + r]) / (sa + 1e-30));
                        }
                    }
                std::printf("%-7s matrix-core tiles taken %lld (scalar run: %lld)\n", iq::name(type), taken, scal);
                if (taken < 2 || scal != 0) { std::printf("%-7s FAIL: the matrix-core tiles did not run as forced\n", iq::name(type)); ++fails; }
                report("wmma gate_up vs CPU", type, wg, type == 19 ? 1.5e-2 : 1e-2);
                report("wmma down vs CPU", type, wd, 3e-3);
                report("wmma vs scalar h", type, wx, type == 19 ? 1.5e-2 : 1e-2);
                report("wmma vs scalar y", type, wy, 3e-3);
            }
        }

        // ---- the CPU lane on the same experts and inputs (ggml-cpu's rows, or Strata's AVX2 ones with DS4_GG_IQ256=1)
        {
            gg::IqCpu fm; std::string err;
            if (!gg::iq_cpu_fmt(type, type, E, FF, fm, err)) { std::printf("%-7s cpu lane: %s\n", iq::name(type), err.c_str()); ++fails; }
            else {
                std::vector<uint8_t> act(fm.act_bytes), hact(fm.h_bytes);
                gg::iq_cpu_quant(fm.gu_act, x.data(), E, act.data());
                // (the CPU lane rounds the activation to its own type - Q8_K, one F32 scale per 256 values - so the
                //  reference is that quantized activation, not the device's per-32 one)
                const std::vector<float> xhat = act_values(fm.gu_act, act.data(), E);
                double wg = 0, wd = 0;
                for (int e = 0; e < NEXP; ++e) {
                    std::vector<float> hc(FF), yc(E);
                    float* hp[1] = {hc.data()}; float* yp[1] = {yc.data()};
                    const void* ap[1] = {act.data()};
                    gg::iq_cpu_gate_up(fm, &blobs[e * blob + off[0]], &blobs[e * blob + off[1]], ap, 1, hp, 0, FF, lim);
                    for (int r = 0; r < FF; ++r) {
                        double sg, su;
                        const double g = dotd(&G[e].w[(size_t) r * E], xhat.data(), E, &sg);
                        const double u = dotd(&U[e].w[(size_t) r * E], xhat.data(), E, &su);
                        wg = std::max(wg, std::fabs(hc[r] - swiglu(g, u, lim)) / (sg + su + 1e-30));
                    }
                    gg::iq_cpu_quant(fm.d_act, &href[(size_t) e * FF], FF, hact.data());
                    const std::vector<float> hhat = act_values(fm.d_act, hact.data(), FF);
                    const void* hpp[1] = {hact.data()};
                    gg::iq_cpu_down(fm, &blobs[e * blob + off[2]], hpp, 1, yp, 0, E);
                    for (int r = 0; r < E; ++r) { double sa; const double ref = dotd(&D[e].w[(size_t) r * FF], hhat.data(), FF, &sa); wd = std::max(wd, std::fabs(yc[r] - ref) / (sa + 1e-30)); }
                }
                report("cpu gate_up vs ref", type, wg, 1e-5);
                report("cpu down vs ref", type, wd, 1e-5);
                std::printf("%-7s cpu rows: %s\n", iq::name(type), gg::iq_cpu_own_rows(type) ? "Strata AVX2" : "ggml-cpu");
            }
        }
        CK(hipFree(dblob));
    }
    if (g_cmp && devs.size() > 1) { std::fclose(g_cmp); g_cmp = nullptr; std::printf("device %d vs device %d: %ld buffers, %ld differ\n", devs[di], devs[0], g_cmp_n, g_cmp_bad); fails += g_cmp_bad != 0; }
    }
    if (g_cmp) { std::printf("bit-compare vs golden: %ld buffers, %ld differ -> %s\n", g_cmp_n, g_cmp_bad, g_cmp_bad ? "DIFFERENT" : "IDENTICAL"); fails += g_cmp_bad != 0; }
    if (g_dump) std::fclose(g_dump);
    std::printf("%s\n", fails ? "FAILED" : "ALL OK");
    return fails;
}
