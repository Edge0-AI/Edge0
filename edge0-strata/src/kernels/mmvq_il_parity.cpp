// src/kernels/mmvq_il_parity.cpp - native_mmvq_il (2-4 columns from the interleaved q8_1 copy, fork F4) against
// native_mmvq's multi-column kernels, bitwise, for every rows-a-warp choice; --bench times both per call.
//
//     build/mmvq_il_parity [--bench [--emit-table] [--hot] [--gap MB] [--persist] [--ab] [--pf a,b,..] [--rounds N]]
//
// --pf a,b,..: the P1 / P2 kernel variants (native_mmvq_il_pf: 0 = today's kernel, 6 hoisted group loads, 3 + P2's exact
// integer forms, 4 = 3 at 5 blocks a SM, 2 / 5 = 3 / 4 with a ring) are each bit-checked against native_mmvq, and with
// --bench timed interleaved per (case, T, rows): N rounds
// (default 5) of 100 calls each, variant after variant, the median of the rounds printed per variant (PFBENCH lines; the
// table cells take the first listed variant).  Without --pf every call takes STRATA_MMVQ_IL_PF (unset: today's kernel).
//
// --hot times one copy of the weights (they sit in L2 after the first call: the ceiling of an L2
// prefetch); --gap MB reads MB megabytes of OTHER weights (native_mmvq over other copies) before every timed call, as
// the window's GEMVs do between a prefetch and its consumer (a memset gap was tried first: its write-backs drain
// during the timed call and make every arm slower than cold); --persist puts the timed weights in a persisting L2
// set-aside (cudaLimitPersistingL2CacheSize + an access-policy window on the stream) so the gap's reads stream past
// them.  --ab runs the arms cold / hot / hot+gap / hot+gap+persist / cold per case in ONE process (the clocks ramp
// over the first ~100 ms of a run, so arms in separate runs are not comparable).  MMVQ_IL_TMIN / MMVQ_IL_TMAX bound T.
//
// --emit-table (with --bench) prints the rows-a-warp table this card measured, in the source form of kIlRows in
// native_mmvq.cu: per cell the rows count (1/2/4) that takes at least 3% off native_mmvq's time, else 0.
//
// The same Q8_1 bytes feed both paths (quantize_q8_1_rows, then native_q8_1_interleave). Weights are random bytes with
// the block scales rewritten as normal fp16 values, so every output is finite (a NaN would compare equal whatever
// produced it). The bench cycles through enough copies of the weights to miss the L2 cache, as a layer's call does.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/core/defaults_026.hpp"
#include "strata/core/sm120_series.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
using namespace strata::kernels;

struct Cell { int type, T, n_out, pf; double ref; double il[5]; };   // pf: the variant the il[] times ran (-1: the environment's)
std::vector<Cell> g_cells;
bool g_hot = false, g_persist = false, g_ab = false;
std::vector<int> g_pf;   // --pf: the variants to check and time; empty: the environment's
int g_rounds = 5;
std::size_t g_gap_bytes = 0;
std::size_t g_gap_now = 0;   // the gap in force for the current arm (--ab) or g_gap_bytes
// time `reps` calls of `fn` on `s`; with --gap each call is preceded by `gap` (other weights' reads) and timed on its own
template <class F, class G>
double time_us(F fn, int reps, cudaStream_t s, G gap) {
    cudaEvent_t e0, e1;
    (void) cudaEventCreate(&e0);
    (void) cudaEventCreate(&e1);
    float total = 0;
    for (int rep = 0; rep < 2; ++rep) {   // rep 0 warms, rep 1 times
        total = 0;
        if (g_gap_now == 0) {
            (void) cudaEventRecord(e0, s);
            for (int i = 0; i < reps; ++i) fn(i);
            (void) cudaEventRecord(e1, s);
            (void) cudaEventSynchronize(e1);
            (void) cudaEventElapsedTime(&total, e0, e1);
        } else {
            for (int i = 0; i < reps; ++i) {
                gap();
                (void) cudaEventRecord(e0, s);
                fn(i);
                (void) cudaEventRecord(e1, s);
                (void) cudaEventSynchronize(e1);
                float ms = 0;
                (void) cudaEventElapsedTime(&ms, e0, e1);
                total += ms;
            }
        }
    }
    (void) cudaEventDestroy(e0);
    (void) cudaEventDestroy(e1);
    return 1000.0 * total / reps;
}
// --persist: the matrix `w` (wbytes) persisting in L2 for every kernel on `s`; off: the window cleared
void persist_window(cudaStream_t s, void* w, std::size_t wbytes, bool on) {
#if !defined(STRATA_HIP_GFX906)
    cudaStreamAttrValue v{};
    v.accessPolicyWindow.base_ptr = on ? w : nullptr;
    v.accessPolicyWindow.num_bytes = on ? wbytes : 0;
    v.accessPolicyWindow.hitRatio = 1.0f;
    v.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    v.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    if (cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &v) != cudaSuccess) {
        std::printf("--persist: the access policy window was refused: %s\n", cudaGetErrorString(cudaGetLastError()));
    }
#else
    (void) s; (void) w; (void) wbytes;   // (no L2 access policy window in the gfx906 compat layer)
    std::printf("--persist: not available in this build (%s)\n", on ? "on" : "off");
#endif
}
struct Case {
    const char* name;
    int type, n_in, n_out, block_elems, block_bytes, scale_at[2];
};
// the engine's dense shapes: qkv 10240 / gate 6144 / out 2560 (GDN), q 12288 / k,v 512 / o 2560 (QSA), the head
const Case CASES[] = {
    {"IQ4_XS 2560x10240", 23, 2560, 10240, 256, 136, {0, -1}},
    {"IQ4_XS 2560x6144", 23, 2560, 6144, 256, 136, {0, -1}},
    {"IQ4_XS 6144x2560", 23, 6144, 2560, 256, 136, {0, -1}},
    {"IQ4_XS 2560x512", 23, 2560, 512, 256, 136, {0, -1}},
    {"Q4_K 2560x10240", 12, 2560, 10240, 256, 144, {0, 2}},
    {"Q4_K 6144x2560", 12, 6144, 2560, 256, 144, {0, 2}},
    {"Q4_K odd rows 2560x2051", 12, 2560, 2051, 256, 144, {0, 2}},
    {"Q5_K 2560x12288", 13, 2560, 12288, 256, 176, {0, 2}},
    {"Q5_K 2560x6144", 13, 2560, 6144, 256, 176, {0, 2}},
    {"Q5_K 6144x2560", 13, 6144, 2560, 256, 176, {0, 2}},
    {"Q5_K head 2560x248320", 13, 2560, 248320, 256, 176, {0, 2}},
    {"Q6_K 2560x12288", 14, 2560, 12288, 256, 210, {208, -1}},
    {"Q6_K 2560x10240", 14, 2560, 10240, 256, 210, {208, -1}},
    {"Q6_K 6144x2560", 14, 6144, 2560, 256, 210, {208, -1}},
    {"Q6_K odd rows 2560x2051", 14, 2560, 2051, 256, 210, {208, -1}},
    {"Q6_K 2560x512", 14, 2560, 512, 256, 210, {208, -1}},
    {"IQ4_XS 2560x12288", 23, 2560, 12288, 256, 136, {0, -1}},
    {"IQ4_XS 2560x640", 23, 2560, 640, 256, 136, {0, -1}},
    {"Q4_K 2560x6144", 12, 2560, 6144, 256, 144, {0, 2}},
    {"Q4_K 2560x640", 12, 2560, 640, 256, 144, {0, 2}},
    {"Q4_K 2560x512", 12, 2560, 512, 256, 144, {0, 2}},
    {"Q5_K 2560x10240", 13, 2560, 10240, 256, 176, {0, 2}},
    {"Q5_K 2560x640", 13, 2560, 640, 256, 176, {0, 2}},
    {"Q5_K 2560x512", 13, 2560, 512, 256, 176, {0, 2}},
    {"Q6_K 2560x6144", 14, 2560, 6144, 256, 210, {208, -1}},
    {"Q6_K 2560x640", 14, 2560, 640, 256, 210, {208, -1}},
    {"Q3_K 2560x10240", 11, 2560, 10240, 256, 110, {108, -1}},
    {"Q3_K 2560x12288", 11, 2560, 12288, 256, 110, {108, -1}},
    {"Q3_K 2560x6144", 11, 2560, 6144, 256, 110, {108, -1}},
    {"Q3_K 6144x2560", 11, 6144, 2560, 256, 110, {108, -1}},
    {"Q3_K 2560x640", 11, 2560, 640, 256, 110, {108, -1}},
    {"Q3_K 2560x512", 11, 2560, 512, 256, 110, {108, -1}},
    {"Q3_K 2560x2051 odd", 11, 2560, 2051, 256, 110, {108, -1}},
};

uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}
bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

int run_case(const Case& c, bool bench, cudaStream_t s) {
    if (std::strstr(c.name, "odd") != nullptr) bench = false;   // parity-only shape: not a timing cell of the table
    if (c.type == 13 && c.n_out > 100000 && !bench) { /* the head: parity too, just big */ }
    const std::size_t wbytes = native_mmvq_weight_bytes(c.type, c.n_in, c.n_out);
    std::mt19937 rng(77u + (unsigned) c.type * 31u + (unsigned) c.n_out);
    std::vector<uint8_t> w(wbytes);
    for (auto& b : w) b = (uint8_t) (rng() & 0xff);
    const std::size_t n_blocks = (std::size_t) c.n_out * (std::size_t) (c.n_in / c.block_elems);
    if (n_blocks * (std::size_t) c.block_bytes != wbytes) {
        std::printf("%s: weight bytes mismatch\n", c.name);
        return 1;
    }
    for (std::size_t k = 0; k < n_blocks; ++k)
        for (int at : c.scale_at)
            if (at >= 0) {
                const uint16_t h = sane_half(rng);
                std::memcpy(&w[k * (std::size_t) c.block_bytes + (std::size_t) at], &h, 2);
            }
    // --hot: one copy (timed hot); with --gap, 1 + the copies the gap reads (dw[1..]) - the timed call always reads dw[0]
    const int gap_copies = g_gap_bytes > 0 ? (int) std::min<std::size_t>(23, (g_gap_bytes + wbytes - 1) / wbytes) : 0;
    const int copies = bench && (!g_hot || g_ab) ? (int) std::max<std::size_t>(1, std::min<std::size_t>(24, (192u << 20) / wbytes)) : 1 + gap_copies;
    std::vector<void*> dw(copies, nullptr);
    for (int i = 0; i < copies; ++i) {
        if (!ck(cudaMalloc(&dw[i], wbytes), "malloc w") || !ck(cudaMemcpy(dw[i], w.data(), wbytes, cudaMemcpyHostToDevice), "copy w"))
            return 1;
    }
    int bad = 0;
    if (bench && g_persist && !g_ab) persist_window(s, dw[0], wbytes, true);
    g_gap_now = g_gap_bytes;
    const int t_max = std::getenv("MMVQ_IL_TMAX") ? std::atoi(std::getenv("MMVQ_IL_TMAX")) : 8;   // 147-verify: 5-8 columns
    const int t_min = std::getenv("MMVQ_IL_TMIN") ? std::atoi(std::getenv("MMVQ_IL_TMIN")) : 1;
    for (int T = t_min; T <= t_max; ++T) {
        std::vector<float> x((std::size_t) T * c.n_in);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : x) v = nd(rng);
        float* dx = nullptr;
        void *xq = nullptr, *xil = nullptr;
        float *yref = nullptr, *yil = nullptr;
        const std::size_t ybytes = (std::size_t) T * c.n_out * 4;
        if (!ck(cudaMalloc(&dx, x.size() * 4), "malloc x") ||
            !ck(cudaMalloc(&xq, native_q8_1_bytes(c.n_in, T)), "malloc xq") ||
            !ck(cudaMalloc(&xil, native_q8_1_il_bytes(c.n_in, T)), "malloc xil") ||
            !ck(cudaMalloc((void**) &yref, ybytes), "malloc yref") || !ck(cudaMalloc((void**) &yil, ybytes), "malloc yil"))
            return 1;
        ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "copy x");
        quantize_q8_1_rows(dx, T, c.n_in, xq, s);
        if (T >= 2) native_q8_1_interleave(xq, xil, c.n_in, T, s);
        native_mmvq(c.type, dw[0], xq, yref, c.n_in, c.n_out, T, s);
        ck(cudaStreamSynchronize(s), "ref");
        auto gap = [&]() {   // --gap: other weights' reads between the timed calls (the window's traffic)
            for (int j = 1; j <= gap_copies; ++j) native_mmvq(c.type, dw[j], xq, yref, c.n_in, c.n_out, T, s);
        };
        std::vector<uint32_t> a((std::size_t) T * c.n_out), b(a.size());
        ck(cudaMemcpy(a.data(), yref, ybytes, cudaMemcpyDeviceToHost), "read ref");
        double best_ref = 0, best_il[5] = {0, 0, 0, 0, 0};
        std::vector<std::array<double, 5>> pf_il(g_pf.size(), std::array<double, 5>{0, 0, 0, 0, 0});   // per --pf variant
        int table_r = 0;
        struct Arm { const char* name; bool hot; std::size_t gap; bool persist; };
        const Arm arms_ab[] = {{"cold", false, 0, false}, {"hot", true, 0, false}, {"hot+gap", true, g_gap_bytes, false},
                               {"hot+gap+persist", true, g_gap_bytes, true}, {"cold2", false, 0, false}};
        const Arm arms_one[] = {{g_hot ? "hot" : "cold", g_hot, g_gap_bytes, g_persist}};
        const Arm* arms = g_ab ? arms_ab : arms_one;
        const int n_arms = g_ab ? 5 : 1;
        for (int ai = 0; ai < (bench ? n_arms : 1); ++ai) {
        if (bench) {
            g_hot = arms[ai].hot;
            g_gap_now = arms[ai].gap;
            if (arms[ai].persist != g_persist) { g_persist = arms[ai].persist; persist_window(s, dw[0], wbytes, g_persist); }
        }
        const std::vector<int> pfs = g_pf.empty() ? std::vector<int>{-1} : g_pf;
        for (int r : {0, 1, 2, 4}) {
            native_mmvq_il_tune(r);
            for (int pf : pfs) {
                native_mmvq_il_pf(pf);
                cudaMemset(yil, 0xff, ybytes);
                native_mmvq_il(c.type, dw[0], xq, xil, yil, c.n_in, c.n_out, T, s);
                ck(cudaStreamSynchronize(s), "il");
                ck(cudaMemcpy(b.data(), yil, ybytes, cudaMemcpyDeviceToHost), "read il");
                long long diff = 0, nonfinite = 0;
                for (std::size_t i = 0; i < a.size(); ++i) {
                    float fa;
                    std::memcpy(&fa, &a[i], 4);
                    if (!std::isfinite(fa)) ++nonfinite;
                    if (a[i] != b[i]) ++diff;
                }
                if (diff != 0 || nonfinite != 0) {
                    ++bad;
                    std::printf("FAIL %-26s T=%d rows=%d pf=%d: %lld bits differ, %lld non-finite of %zu\n", c.name, T, r, pf, diff,
                                nonfinite, a.size());
                }
            }
            native_mmvq_il_pf(-1);
            if (bench && r != 0) {
                auto il_call = [&](int i) { native_mmvq_il(c.type, dw[g_hot ? 0 : i % copies], xq, xil, yil, c.n_in, c.n_out, T, s); };
                if (g_pf.empty()) {
                    best_il[r] = time_us(il_call, 200, s, gap);
                } else {   // the variants interleaved, round after round; the median of each one's rounds
                    std::vector<std::vector<double>> t(g_pf.size());
                    for (int round = 0; round < g_rounds; ++round)
                        for (std::size_t k = 0; k < g_pf.size(); ++k) {
                            native_mmvq_il_pf(g_pf[k]);
                            t[k].push_back(time_us(il_call, 100, s, gap));
                        }
                    native_mmvq_il_pf(-1);
                    std::printf("PFBENCH %-26s T=%d rows=%d:", c.name, T, r);
                    for (std::size_t k = 0; k < g_pf.size(); ++k) {
                        std::sort(t[k].begin(), t[k].end());
                        const double med = t[k][t[k].size() / 2];
                        if (k == 0) best_il[r] = med;
                        pf_il[k][r] = med;
                        std::printf("  pf%d %.1f us (min %.1f max %.1f)", g_pf[k], med, t[k].front(), t[k].back());
                    }
                    std::printf("\n");
                }
            }
        }
        native_mmvq_il_tune(0);
        if (bench) {
            best_ref = time_us([&](int i) { native_mmvq(c.type, dw[g_hot ? 0 : i % copies], xq, yref, c.n_in, c.n_out, T, s); }, 200, s, gap);
            (void) table_r;
            if (g_pf.empty()) g_cells.push_back({c.type, T, c.n_out, -1, best_ref, {0, best_il[1], best_il[2], 0, best_il[4]}});
            for (std::size_t k = 0; k < g_pf.size(); ++k)
                g_cells.push_back({c.type, T, c.n_out, g_pf[k], best_ref, {0, pf_il[k][1], pf_il[k][2], 0, pf_il[k][4]}});
            std::printf("BENCH %-26s T=%d %-16s multi %.1f us | il rows1 %.1f rows2 %.1f rows4 %.1f us\n", c.name, T,
                        arms[ai].name, best_ref, best_il[1], best_il[2], best_il[4]);
        }
        }   // arms
        cudaFree(dx); cudaFree(xq); cudaFree(xil); cudaFree(yref); cudaFree(yil);
    }
    if (bench && g_persist) persist_window(s, dw[0], wbytes, false);
    for (void* p : dw) cudaFree(p);
    return bad;
}

// the table as kIlRows prints it: per (type, ncols, n_out class) the fastest rows count if >= 3% under native_mmvq, else 0.
// A cell with several measured shapes takes the worst (smallest) gain of the best choice, as the shipped table does for
// two cards; a class with no measured shape prints the value of its neighbour below it (or 0).
void emit_table() {
    const int types[] = {23, 12, 13, 14, 11};
    const char* names[] = {"IQ4_XS", "Q4_K", "Q5_K", "Q6_K", "Q3_K"};
    std::printf("// measured on this card by mmvq_il_parity --bench --emit-table\n");
    // per cell: the rows count whose WORST ratio to native_mmvq over the cell's measured shapes is the best, taken only
    // when that worst ratio is under 0.97 (else 0); a class with no measured shape takes its lower neighbour's value
    // pf_sel: the cells of that variant (the environment's when --pf was not given, else 0 = today's kernel); -2: every
    // listed variant, the answer then pf << 4 | rows (the kIlPf form), 0 when nothing beats native_mmvq by 3%
    auto pick_cell = [&](int type, int nc, int cls, int prev, int pf_sel) {
        double best_worst = 1e9;
        int best = -1;
        bool any = false;
        for (int pf : (pf_sel == -2 ? (g_pf.empty() ? std::vector<int>{-1} : g_pf) : std::vector<int>{pf_sel}))
            for (int r : {1, 2, 4}) {
                double worst = 0;
                bool have = false;
                for (const Cell& c : g_cells) {
                    if (c.type != type || c.T != nc || c.pf != pf) continue;
                    const int cc = c.n_out < 2048 ? 0 : c.n_out < 4096 ? 1 : c.n_out < 8192 ? 2 : c.n_out < 12288 ? 3 : 4;
                    if (cc != cls || c.il[r] <= 0) continue;
                    have = true;
                    worst = std::max(worst, c.il[r] / c.ref);
                }
                any = any || have;
                if (have && worst < best_worst) { best_worst = worst; best = pf_sel == -2 ? (std::max(pf, 0) << 4 | r) : r; }
            }
        if (!any) return prev;
        return best_worst < 0.97 ? best : 0;
    };
    const int pf_today = g_pf.empty() ? -1 : 0;
    for (int ti = 0; ti < 5; ++ti) {
        std::printf("    {%d, {", types[ti]);
        for (int nc = 2; nc <= 4; ++nc) {
            int prev = 0;
            std::printf("{");
            for (int cls = 0; cls < 5; ++cls) {
                prev = pick_cell(types[ti], nc, cls, prev, pf_today);
                std::printf("%d%s", prev, cls < 4 ? ", " : "");
            }
            std::printf("}%s", nc < 4 ? ", " : "");
        }
        std::printf("}},   // %s\n", names[ti]);
    }
    std::printf("// 5-8 columns (kIlRows58_<cc>; rows 4 runs as 2 there):\n");
    for (int ti = 0; ti < 5; ++ti) {
        std::printf("    {%d, {", types[ti]);
        for (int nc = 5; nc <= 8; ++nc) {
            int prev = 0;
            std::printf("{");
            for (int cls = 0; cls < 5; ++cls) {
                prev = pick_cell(types[ti], nc, cls, prev, pf_today);
                std::printf("%d%s", prev, cls < 4 ? ", " : "");
            }
            std::printf("}%s", nc < 8 ? ", " : "");
        }
        std::printf("}},   // %s\n", names[ti]);
    }
    if (!g_pf.empty()) {   // the joint (variant, rows) table over the listed variants: kIlPf_80 / kIlPf58_80
        std::printf("// variant << 4 | rows over --pf (kIlPf_<cc>), 2-4 columns:\n");
        for (int ti = 0; ti < 5; ++ti) {
            std::printf("    {%d, {", types[ti]);
            for (int nc = 2; nc <= 4; ++nc) {
                int prev = 0;
                std::printf("{");
                for (int cls = 0; cls < 5; ++cls) {
                    prev = pick_cell(types[ti], nc, cls, prev, -2);
                    std::printf("0x%02x%s", prev, cls < 4 ? ", " : "");
                }
                std::printf("}%s", nc < 4 ? ", " : "");
            }
            std::printf("}},   // %s\n", names[ti]);
        }
        std::printf("// 5-8 columns (kIlPf58_<cc>):\n");
        for (int ti = 0; ti < 5; ++ti) {
            std::printf("    {%d, {", types[ti]);
            for (int nc = 5; nc <= 8; ++nc) {
                int prev = 0;
                std::printf("{");
                for (int cls = 0; cls < 5; ++cls) {
                    prev = pick_cell(types[ti], nc, cls, prev, -2);
                    std::printf("0x%02x%s", prev, cls < 4 ? ", " : "");
                }
                std::printf("}%s", nc < 8 ? ", " : "");
            }
            std::printf("}},   // %s\n", names[ti]);
        }
    }
    std::printf("// one column (kIl1RowsShared):\n");
    for (int ti = 0; ti < 5; ++ti) {
        std::printf("    {%d, {", types[ti]);
        int prev = 0;
        for (int cls = 0; cls < 5; ++cls) {
            prev = pick_cell(types[ti], 1, cls, prev, pf_today);
            std::printf("%d%s", prev, cls < 4 ? ", " : "");
        }
        std::printf("}},   // %s\n", names[ti]);
    }
}

// the per-architecture lookup, no GPU needed for its answers
// The two Ada runs sm_89's table was merged from (mmvq_il_parity --bench --emit-table, issue 3453): by type IQ4_XS, Q4_K,
// Q5_K, Q6_K, Q3_K, then 2/3/4 columns, then the five n_out classes.
const int kAda4070[5][3][5] = {
    {{0, 0, 0, 1, 0}, {1, 1, 1, 1, 1}, {2, 2, 1, 1, 1}}, {{0, 0, 1, 1, 1}, {0, 0, 1, 1, 1}, {0, 1, 1, 1, 1}},
    {{0, 0, 1, 1, 1}, {0, 0, 1, 1, 1}, {1, 2, 1, 2, 2}}, {{0, 0, 0, 0, 0}, {0, 0, 1, 2, 2}, {0, 1, 4, 1, 1}},
    {{1, 1, 1, 2, 1}, {1, 1, 1, 1, 1}, {2, 1, 1, 1, 1}}};
const int kAda4080[5][3][5] = {
    {{0, 0, 0, 0, 0}, {0, 0, 1, 1, 1}, {0, 1, 1, 1, 1}}, {{0, 0, 1, 1, 1}, {0, 0, 1, 1, 1}, {0, 0, 1, 1, 1}},
    {{0, 0, 1, 1, 0}, {0, 0, 1, 1, 0}, {0, 0, 1, 1, 1}}, {{0, 0, 0, 1, 0}, {0, 0, 1, 1, 1}, {0, 0, 1, 1, 1}},
    {{0, 0, 1, 1, 1}, {0, 1, 1, 1, 1}, {0, 1, 1, 1, 1}}};
int check_tables() {
    int bad = 0;
    const char* e89 = std::getenv("STRATA_MMVQ_IL_89");
    // sm_89's own tables are ON by default (#3453, 0.2.7) unless STRATA_MMVQ_IL_89=0 or STRATA_DEFAULTS=026
    const bool opt89 = strata::mmvq_il_89_resolve(e89, false, 89, strata::qwen_defaults_026());
    for (int type : {23, 12, 13, 14, 11})
        for (int nc = 2; nc <= 4; ++nc)
            for (int n_out : {512, 2560, 6144, 10240, 12288, 248320}) {
                const int a = native_mmvq_il_rows_for(86, type, nc, n_out);
                if (a != 0 && a != 1 && a != 2 && a != 4) { ++bad; std::printf("FAIL table: rows %d\n", a); }
                // sm_89 (#3453): the cells where an RTX 4070 and an RTX 4080 gave the same rows count, else 0
                const int cls = n_out < 2048 ? 0 : n_out < 4096 ? 1 : n_out < 8192 ? 2 : n_out < 12288 ? 3 : 4;
                const int ti = type == 23 ? 0 : type == 12 ? 1 : type == 13 ? 2 : type == 14 ? 3 : 4;
                const int b = native_mmvq_il_rows_for(89, type, nc, n_out);
                // tables on (the default): the agreement; off (=0 or STRATA_DEFAULTS=026): what sm_89 answered before #3453 (the shared table, which is
                // sm_86's except Q3_K, a type the shared table has no row for)
                const int want = opt89 ? (kAda4070[ti][nc - 2][cls] == kAda4080[ti][nc - 2][cls] ? kAda4070[ti][nc - 2][cls] : 0)
                                       : (type == 11 ? 0 : a);
                if (b != want) {
                    ++bad;
                    std::printf("FAIL table: sm_89 is %d, %s says %d at type %d ncols %d n_out %d\n", b,
                                opt89 ? "the 4070/4080 agreement" : "the shared table", want, type, nc, n_out);
                }
            }
    // Volta has its own table (PR 1401): Q6_K at 2 columns, 2560 rows -> 1 row a warp; the Ampere table says 0 there
    if (native_mmvq_il_rows_for(70, 14, 2, 2560) != 1 || native_mmvq_il_rows_for(86, 14, 2, 2560) != 0) {
        ++bad;
        std::printf("FAIL table: the sm_70 table is not the V100's\n");
    }
    // Pascal has its own table too (the P100's): Q6_K, 3 columns, 12288 rows -> 4 rows a warp; sm_75 still has none (the shared one)
    if (native_mmvq_il_rows_for(60, 14, 3, 12288) != 4 || native_mmvq_il_rows_for(60, 23, 2, 512) != 0) {
        ++bad;
        std::printf("FAIL table: the sm_60 table is not the P100's\n");
    }
    // sm_89 one column: only Q3_K with the tables on (the 4070 and the 4080 agree), else none; wider than 4: the plain kernels
    if (native_mmvq_il_rows_for(89, 11, 1, 2560) != 0 || native_mmvq_il_rows_for(89, 11, 1, 6144) != (opt89 ? 1 : 0) ||
        native_mmvq_il_rows_for(89, 12, 1, 6144) != 0) {
        ++bad;
        std::printf("FAIL table: the sm_89 one-column row is not Q3_K only\n");
    }
    // sm_80 has its own tables (kIlRows_80 / kIlRows58_80 / kIl1Rows_80: a CMP 170HX's); the key 81 is what STRATA_MMVQ_IL_SM80=0
    // gives an sm_80 card: no table of its own, so the shared one (and no one-column table).  Every answer is 0, 1, 2 or 4.
    {
        bool differs = false;
        for (int type : {23, 12, 13, 14, 11})
            for (int nc = 1; nc <= 8; ++nc)
                for (int n_out : {512, 2560, 6144, 10240, 12288, 248320}) {
                    const int a = native_mmvq_il_rows_for(80, type, nc, n_out), b = native_mmvq_il_rows_for(81, type, nc, n_out);
                    if (a != 0 && a != 1 && a != 2 && a != 4) { ++bad; std::printf("FAIL table: sm_80 rows %d\n", a); }
                    differs = differs || a != b;
                }
        if (!differs) { ++bad; std::printf("FAIL table: sm_80 answers like the shared table\n"); }
        // spot checks read off the source tables: Q6_K at 3 columns and 2560 rows (class 1) is {0, 2, 2, 2, 2}[1] = 2; one-column
        // IQ4_XS at 6144 rows (class 2) is {0, 0, 1, 4, 1}[2] = 1 on sm_80 and 0 (no one-column table) on the key 81
        if (native_mmvq_il_rows_for(80, 14, 3, 2560) != 2 || native_mmvq_il_rows_for(80, 23, 1, 6144) != 1 ||
            native_mmvq_il_rows_for(81, 23, 1, 6144) != 0) {
            ++bad;
            std::printf("FAIL table: the sm_80 table is not the CMP 170HX's\n");
        }
    }
    if (native_mmvq_il_rows_for(89, 99, 2, 2560) != 0 || native_mmvq_il_rows_for(89, 12, 5, 2560) != 0 ||
        native_mmvq_il_rows_for(86, 12, 5, 2560) != 0 || native_mmvq_il_rows_for(120, 12, 9, 2560) != 0) {
        ++bad;
        std::printf("FAIL table: an unknown type or ncols must be 0\n");
    }
    return bad;
}
}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    const bool emit = bench && argc > 2 && std::string(argv[2]) == "--emit-table";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--hot") g_hot = true;
        else if (a == "--ab") g_ab = true;
        else if (a == "--persist") g_persist = true;
        else if (a == "--gap" && i + 1 < argc) g_gap_bytes = (std::size_t) std::atoi(argv[++i]) << 20;
        else if (a == "--rounds" && i + 1 < argc) g_rounds = std::atoi(argv[++i]);
        else if (a == "--pf" && i + 1 < argc) {
            const std::string list = argv[++i];
            for (std::size_t p = 0; p < list.size();) {
                std::size_t q = list.find(',', p);
                if (q == std::string::npos) q = list.size();
                g_pf.push_back(std::atoi(list.substr(p, q - p).c_str()));
                p = q + 1;
            }
        }
    }
    cudaStream_t s;
    cudaStreamCreate(&s);
#if !defined(STRATA_HIP_GFX906)
    if (g_persist || g_ab) {
        int dev = 0, l2 = 0, maxp = 0, maxw = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev);
        cudaDeviceGetAttribute(&maxp, cudaDevAttrMaxPersistingL2CacheSize, dev);
        cudaDeviceGetAttribute(&maxw, cudaDevAttrMaxAccessPolicyWindowSize, dev);
        std::size_t want = std::min<std::size_t>(20u << 20, (std::size_t) maxp), got = 0;
        const cudaError_t e = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, want);
        cudaDeviceGetLimit(&got, cudaLimitPersistingL2CacheSize);
        std::printf("--persist: L2 %d MiB, max persisting %d MiB, max window %d MiB; set-aside asked %zu MiB -> %s, granted %zu MiB\n",
                    l2 >> 20, maxp >> 20, maxw >> 20, want >> 20, cudaGetErrorString(e), got >> 20);
    }
#endif
    if (bench) std::printf("bench arms: %s%s%s\n", g_hot ? "hot (one copy) " : "cold (192 MB rotation) ",
                           g_gap_bytes ? ("gap " + std::to_string(g_gap_bytes >> 20) + " MB ").c_str() : "", g_persist ? "persist" : "");
    int bad = 0;
    bad += check_tables();
    if (bench) {   // ~150 ms of launches so the clocks are up before the first timed case
        void* wu = nullptr; float* yu = nullptr; void* xu = nullptr;
        if (cudaMalloc(&wu, 6144u * 2560u) == cudaSuccess && cudaMalloc((void**) &yu, 2560 * 4) == cudaSuccess &&
            cudaMalloc(&xu, native_q8_1_bytes(6144, 1)) == cudaSuccess) {
            (void) cudaMemset(wu, 0, 6144u * 2560u); (void) cudaMemset(xu, 0, native_q8_1_bytes(6144, 1));
            for (int i = 0; i < 6000; ++i) native_mmvq(12, wu, xu, yu, 6144, 2560, 1, s);
            (void) cudaStreamSynchronize(s);
            (void) cudaFree(wu); (void) cudaFree(yu); (void) cudaFree(xu);
        }
    }
    const char* only = std::getenv("MMVQ_IL_ONLY");   // 143-dense: only the cases whose name contains this
    for (const Case& c : CASES)
        if (only == nullptr || std::strstr(c.name, only) != nullptr) bad += run_case(c, bench, s);
    if (emit) emit_table();
    std::printf("%s\n", bad ? "mmvq_il_parity: FAIL" : "mmvq_il_parity: OK (every case, T 1-8, rows 1/2/4 and the table, bitwise)");
    return bad ? 1 : 0;
}
