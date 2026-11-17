// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// tests/ds4_mx_bench.cpp - how fast does this CPU run DeepSeek-V4.1-Flash's MXFP4 experts on the CPU lane?
//
//   ds4_mx_bench [--threads 1,6,12] [--T 1,2,4,8,16] [--experts 16] [--per-layer 3] [--secs 0.4] [--pf 0]
//                [--kernels float,avx2,avxvnni,avx512,avx512vnni,ggml,scalar] [--file PATH] [--make-file PATH]
//
// Experts of the real shape (gate / up 2304 rows x 5120, down 5120 x 2304, MXFP4: 18.8 MB each) with random codes in
// anonymous memory (huge pages where the OS gives them), or in a file's mapping (--file: 4 KiB page-cache pages, as the
// engine reads the GGUF; --make-file writes one).  A "layer" is --per-layer experts, as the lane runs them: the gate /
// up rows of every expert in 64-row jobs on the pool, then (a barrier) their down rows; the layers walk through the
// experts, so the weights come from RAM, not the cache.  Prints the weights' GB/s for each kernel x tokens x threads
// (T tokens a call: each row read once for up to 8 of them), the int8 tiers' agreement with the scalar tier, and the
// cost of the activation quant and of madvise(WILLNEED) on cached pages.
#include "../../src/dsv41/ds4_gguf_cpu.hpp"
#include "../../src/dsv41/ds4_pool.hpp"

#if defined(STRATA_NATIVE_EXPERTS)
#include "ggml-cpu.h"
#include "ggml.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace gg = strata::ds4::gguf;
namespace os = strata::ds4::os;
using strata::ds4::Pool;

namespace {

constexpr int E = 5120, FF = 2304, RB = 64;
const size_t kGr = (size_t) E / 32 * 17, kDr = (size_t) FF / 32 * 17;   // row bytes
const size_t kMat = kGr * FF;                                           // gate (= up = down) bytes
const size_t kExpert = 3 * kMat;

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
std::vector<int> ints(const std::string& s) {
    std::vector<int> v;
    size_t a = 0;
    while (a < s.size()) {
        size_t b = s.find(',', a);
        if (b == std::string::npos) b = s.size();
        v.push_back(std::atoi(s.substr(a, b - a).c_str()));
        a = b + 1;
    }
    return v;
}
std::vector<std::string> words(const std::string& s) {
    std::vector<std::string> v;
    size_t a = 0;
    while (a < s.size()) {
        size_t b = s.find(',', a);
        if (b == std::string::npos) b = s.size();
        v.push_back(s.substr(a, b - a));
        a = b + 1;
    }
    return v;
}

// the logical CPUs, one of each core first, then their second threads (pinning order)
std::vector<int> cpu_order() {
    std::vector<int> first, rest;
#if defined(_WIN32)
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    std::vector<uint8_t> buf(len);
    if (len > 0 && GetLogicalProcessorInformationEx(RelationProcessorCore,
                                                    (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(), &len)) {
        for (size_t o = 0; o < buf.size();) {
            auto* e = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) (buf.data() + o);
            if (e->Processor.GroupCount > 0 && e->Processor.GroupMask[0].Group == 0) {
                const KAFFINITY m = e->Processor.GroupMask[0].Mask;
                bool f = true;
                for (int b = 0; b < 64; ++b)
                    if ((m >> b) & 1) {
                        (f ? first : rest).push_back(b);
                        f = false;
                    }
            }
            o += e->Size;
        }
    }
#else
    for (int c = 0; c < 1024; ++c) {
        std::FILE* f = std::fopen(("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list").c_str(), "r");
        if (f == nullptr) break;
        int s0 = -1;
        if (std::fscanf(f, "%d", &s0) != 1) s0 = c;
        std::fclose(f);
        (s0 == c ? first : rest).push_back(c);
    }
#endif
    if (first.empty())
        for (unsigned c = 0; c < std::max(1u, std::thread::hardware_concurrency()); ++c) first.push_back((int) c);
    std::sort(first.begin(), first.end());
    std::sort(rest.begin(), rest.end());
    first.insert(first.end(), rest.begin(), rest.end());
    return first;
}

void fill_mx(uint8_t* p, size_t bytes, uint64_t seed) {   // random MXFP4 blocks, scales 2^-14 .. 2^-6 as weights'
    std::mt19937_64 rng(seed);
    for (size_t b = 0; b + 17 <= bytes; b += 17) {
        const uint64_t r = rng(), s = rng();
        p[b] = (uint8_t) (114 + r % 9);
        std::memcpy(p + b + 1, &r, 8);
        std::memcpy(p + b + 9, &s, 8);
    }
}

enum Kern { kFloat = 100, kGgml = 101 };
int kern_id(const std::string& s) {
    if (s == "float") return kFloat;
    if (s == "ggml") return kGgml;
    for (int t = 0; t < gg::kMxTiers; ++t)
        if (s == gg::mx_tier_name(t)) return t;
    return -1;
}

struct Acts {   // T tokens' activations of E and FF values in every form a kernel takes
    int T = 0;
    std::vector<float> xf, hf;                  // F32 (rounded as q8_round) [T][E], [T][FF]
    std::vector<uint8_t> xq, hq;                // mx_quant's [T][2E], [T][2FF]
    std::vector<uint8_t> x8, h8;                // ggml's block_q8_0 [T][E / 32 * 34]
    void make(int t, uint64_t seed) {
        T = t;
        std::mt19937 rng((unsigned) seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        xf.resize((size_t) T * E);
        hf.resize((size_t) T * FF);
        for (auto& v : xf) v = nd(rng);
        for (auto& v : hf) v = 0.05f * nd(rng);
        xq.resize((size_t) T * gg::mx_act_bytes(E));
        hq.resize((size_t) T * gg::mx_act_bytes(FF));
        x8.resize((size_t) T * E / 32 * 34);
        h8.resize((size_t) T * FF / 32 * 34);
        for (int i = 0; i < T; ++i) {
            gg::mx_quant(&xf[(size_t) i * E], E, &xq[(size_t) i * gg::mx_act_bytes(E)]);
            gg::mx_quant(&hf[(size_t) i * FF], FF, &hq[(size_t) i * gg::mx_act_bytes(FF)]);
#if defined(STRATA_NATIVE_EXPERTS)
            const auto* q8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
            q8->from_float(&xf[(size_t) i * E], &x8[(size_t) i * E / 32 * 34], E);
            q8->from_float(&hf[(size_t) i * FF], &h8[(size_t) i * FF / 32 * 34], FF);
#endif
            gg::q8_round(&xf[(size_t) i * E], E, &xf[(size_t) i * E]);
            gg::q8_round(&hf[(size_t) i * FF], FF, &hf[(size_t) i * FF]);
        }
    }
};

// rows [r0, r1) of one matrix (k values a row) against the T activations, kernel `kern`
void rows(int kern, const uint8_t* w, size_t rb, int k, int r0, int r1, const Acts& a, float* const* y, int pf) {
    const bool down = k == FF;
    if (kern == kFloat) {
        for (int i0 = 0; i0 < a.T; i0 += 8) {
            const float* xp[8];
            const int n = std::min(8, a.T - i0);
            for (int i = 0; i < n; ++i) xp[i] = (down ? a.hf.data() : a.xf.data()) + (size_t) (i0 + i) * k;
            gg::mx_cpu_rows(w, rb, k, r0, r1, xp, n, y + i0);
        }
        return;
    }
    if (kern == kGgml) {
#if defined(STRATA_NATIVE_EXPERTS)
        static const ggml_vec_dot_t dot = ggml_get_type_traits_cpu(GGML_TYPE_MXFP4)->vec_dot;
        const uint8_t* x8 = down ? a.h8.data() : a.x8.data();
        for (int r = r0; r < r1; ++r)
            for (int i = 0; i < a.T; ++i) dot(k, &y[i][r - r0], 0, w + (size_t) r * rb, 0, x8 + (size_t) i * k / 32 * 34, 0, 1);
#endif
        return;
    }
    const void* ap[16];
    for (int i = 0; i < a.T; ++i) ap[i] = (down ? a.hq.data() : a.xq.data()) + (size_t) i * gg::mx_act_bytes(k);
    gg::mx_rows_tier(kern, w, rb, k, r0, r1, ap, a.T, y, pf);
}

struct Bench {
    const uint8_t* mem = nullptr;
    int n_exp = 16, per_layer = 3;
    double secs = 0.4;
    // GB/s of weights with `pool` (its size the threads), kernel kern, the T tokens of a
    double run(Pool& pool, int kern, const Acts& a, int pf) {
        const int GB = FF / RB, DB = E / RB;
        std::vector<std::vector<float>> ys((size_t) pool.size() * 64);
        int layer = 0;
        size_t bytes = 0;
        auto one_layer = [&] {
            const int e0 = (layer * per_layer) % n_exp;
            std::function<void(int)> fgu = [&](int j) {
                const int e = (e0 + j / (2 * GB)) % n_exp, m = (j / GB) % 2, b = j % GB;
                float yb[16][RB];
                float* yp[16];
                for (int i = 0; i < a.T; ++i) yp[i] = yb[i];
                rows(kern, mem + (size_t) e * kExpert + (size_t) m * kMat, kGr, E, b * RB, b * RB + RB, a, yp, pf);
            };
            pool.run(per_layer * 2 * GB, fgu);
            std::function<void(int)> fd = [&](int j) {
                const int e = (e0 + j / DB) % n_exp, b = j % DB;
                float yb[16][RB];
                float* yp[16];
                for (int i = 0; i < a.T; ++i) yp[i] = yb[i];
                rows(kern, mem + (size_t) e * kExpert + 2 * kMat, kDr, FF, b * RB, b * RB + RB, a, yp, pf);
            };
            pool.run(per_layer * DB, fd);
            bytes += (size_t) per_layer * kExpert;
            ++layer;
        };
        for (int w = 0; w < 2; ++w) one_layer();   // (warm: the threads, the code)
        bytes = 0;
        const double t0 = now_s();
        double t = t0;
        while (t - t0 < secs) {
            one_layer();
            t = now_s();
        }
        return bytes / (t - t0) / 1e9;
    }
};


}  // namespace

int main(int argc, char** argv) {
    std::vector<int> threads = {1}, Ts = {1, 2, 4, 8, 16}, pfs = {0};
    std::vector<std::string> kerns = {"float", "avx2", "avxvnni", "avx512", "avx512vnni", "ggml"};
    std::string file, make_file;
    Bench B;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--threads") threads = ints(v);
        else if (a == "--T") Ts = ints(v);
        else if (a == "--pf") pfs = ints(v);
        else if (a == "--experts") B.n_exp = std::atoi(v.c_str());
        else if (a == "--per-layer") B.per_layer = std::atoi(v.c_str());
        else if (a == "--secs") B.secs = std::atof(v.c_str());
        else if (a == "--kernels") kerns = words(v);
        else if (a == "--file") file = v;
        else if (a == "--make-file") make_file = v;
    }
#if defined(STRATA_NATIVE_EXPERTS)
    ggml_cpu_init();
#endif
    const size_t total = (size_t) B.n_exp * kExpert;
    if (!make_file.empty()) {
        std::FILE* f = std::fopen(make_file.c_str(), "wb");
        if (f == nullptr) return 1;
        std::vector<uint8_t> buf(kExpert);
        for (int e = 0; e < B.n_exp; ++e) {
            fill_mx(buf.data(), buf.size(), 1000 + (uint64_t) e);
            std::fwrite(buf.data(), 1, buf.size(), f);
        }
        std::fclose(f);
        std::printf("wrote %s: %d experts, %.0f MB\n", make_file.c_str(), B.n_exp, total / 1e6);
        return 0;
    }
    os::fd_t fd = os::kBadFd;
    if (!file.empty()) {
        fd = os::open_ro(file);
        if (fd == os::kBadFd || os::file_size(fd) < total) {
            std::fprintf(stderr, "%s: cannot open, or smaller than %d experts\n", file.c_str(), B.n_exp);
            return 1;
        }
        B.mem = (const uint8_t*) os::map_ro(fd, total);
        volatile uint64_t s = 0;   // (into the page cache and the mapping)
        for (size_t o = 0; o < total; o += 4096) s += B.mem[o];
    } else {
        uint8_t* m = (uint8_t*) os::anon_alloc(total, -1, 1);
        if (m == nullptr) return 1;
        for (int e = 0; e < B.n_exp; ++e) fill_mx(m + (size_t) e * kExpert, kExpert, 1000 + (uint64_t) e);
        B.mem = m;
    }
    std::printf("CPU tiers:");
    for (int t = 0; t < gg::kMxTiers; ++t)
        if (gg::mx_tier_ok(t)) std::printf(" %s", gg::mx_tier_name(t));
    std::printf(" (mx_rows takes %s: %s); weights %s, %d experts (%.0f MB), %d a layer\n", gg::mx_tier_name(gg::mx_tier()),
                gg::mx_tier_report(), file.empty() ? "anonymous memory" : file.c_str(), B.n_exp, total / 1e6, B.per_layer);

    // ---- the int8 tiers against the scalar tier, and the old F32 path against both (T = 3, 64 rows of each kind)
    {
        Acts a;
        a.make(3, 7);
        for (int down = 0; down < 2; ++down) {
            const int k = down ? FF : E;
            const size_t rb = down ? kDr : kGr;
            const uint8_t* w = B.mem + (down ? 2 * kMat : 0);
            float ref[3][RB], got[3][RB];
            float* rp[3] = {ref[0], ref[1], ref[2]};
            float* gp[3] = {got[0], got[1], got[2]};
            rows(gg::kMxScalar, w, rb, k, 0, RB, a, rp, 0);
            for (int t = 0; t < gg::kMxTiers + 1; ++t) {
                const int kern = t < gg::kMxTiers ? t : kFloat;
                if (kern != kFloat && (kern == gg::kMxScalar || !gg::mx_tier_ok(kern))) continue;
                rows(kern, w, rb, k, 0, RB, a, gp, 0);
                double md = 0, mx = 0;
                for (int i = 0; i < 3; ++i)
                    for (int r = 0; r < RB; ++r) {
                        md = std::max(md, (double) std::fabs(got[i][r] - ref[i][r]));
                        mx = std::max(mx, (double) std::fabs(ref[i][r]));
                    }
                std::printf("  %-6s %-10s vs scalar: max |diff| %.3g (max |y| %.3g, rel %.2g)%s\n", down ? "down" : "gate",
                            kern == kFloat ? "float" : gg::mx_tier_name(kern), md, mx, md / std::max(mx, 1e-30),
                            kern != kFloat && std::memcmp(got, ref, sizeof got) == 0 ? "  bit-identical" : "");
            }
        }
        // the quant's two forms: the same bytes
        std::vector<uint8_t> q1(gg::mx_act_bytes(E)), q2(gg::mx_act_bytes(E));
        gg::mx_quant_blocks(a.xf.data(), 0, E / 32, q1.data(), E);
        gg::mx_quant_blocks_scalar(a.xf.data(), 0, E / 32, q2.data(), E);
        std::printf("  quant: vector vs scalar bytes %s\n", q1 == q2 ? "identical" : "DIFFER");
    }

    // ---- the quant and madvise costs
    {
        std::vector<float> x(E), r(E);
        std::mt19937 rng(3);
        std::normal_distribution<float> nd;
        for (auto& v : x) v = nd(rng);
        std::vector<uint8_t> q(gg::mx_act_bytes(E));
        const int N = 20000;
        double t0 = now_s();
        for (int i = 0; i < N; ++i) gg::mx_quant(x.data(), E, q.data());
        const double tq = (now_s() - t0) / N;
        t0 = now_s();
        for (int i = 0; i < N / 10; ++i) gg::q8_round(x.data(), E, r.data());
        const double tr = (now_s() - t0) / (N / 10);
        std::printf("activation of %d: mx_quant %.2f us, q8_round (old, scalar) %.2f us\n", E, tq * 1e6, tr * 1e6);
        if (!file.empty()) {
            t0 = now_s();
            int n = 0;
            for (int e = 0; e < B.n_exp; ++e)
                for (int m = 0; m < 3; ++m, ++n)
                    os::prefetch({{B.mem + (size_t) e * kExpert + (size_t) m * kMat, kMat}});
            std::printf("madvise(WILLNEED) of a cached expert matrix (%.1f MB): %.1f us\n", kMat / 1e6,
                        (now_s() - t0) / n * 1e6);
            t0 = now_s();
            int nres = 0;
            for (int rep = 0; rep < 10; ++rep)
                for (int e = 0; e < B.n_exp; ++e)
                    for (int m = 0; m < 3; ++m)
                        nres += os::resident_sampled(B.mem + (size_t) e * kExpert + (size_t) m * kMat, kMat);
            std::printf("os::resident_sampled (8 pages) of an expert matrix: %.2f us (%d of %d resident)\n",
                        (now_s() - t0) / (10 * n) * 1e6, nres / 10, n);
        }
    }

    // ---- the table
    const std::vector<int> order = cpu_order();
    std::printf("\n%-11s %3s %4s %5s %9s %10s\n", "kernel", "T", "thr", "pf", "GB/s", "GB/s/thr");
    for (int nt : threads) {
        nt = std::max(1, std::min(nt, (int) order.size()));
        Pool pool(nt - 1, std::vector<int>(order.begin() + 1, order.begin() + nt), 20000);
        os::pin_self(order[0]);
        for (const std::string& ks : kerns) {
            const int kern = kern_id(ks);
            if (kern < 0 || (kern < gg::kMxTiers && !gg::mx_tier_ok(kern))) continue;
#if !defined(STRATA_NATIVE_EXPERTS)
            if (kern == kGgml) continue;
#endif
            for (int T : Ts) {
                if (T < 1 || T > 16) continue;
                Acts a;
                a.make(T, 11 + (uint64_t) T);
                for (int pf : pfs) {
                    if (pf != 0 && (kern == kFloat || kern == kGgml || kern == gg::kMxScalar)) continue;
                    const double g = B.run(pool, kern, a, pf);
                    std::printf("%-11s %3d %4d %5d %9.2f %10.2f\n", ks.c_str(), T, nt, pf, g, g / nt);
                    std::fflush(stdout);
                }
            }
        }
    }
    return 0;
}
