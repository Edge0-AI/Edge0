// src/kernels/native_grouped_parity.cpp - native_expert_grouped's group stride (grid_groups) and its fused SwiGLU +
// q8_1 pass, against the launches before them (native_grouped_set_v1: a block row per possible group, SwiGLU and the
// q8_1 quantization as two kernels over all cap_entries).  Synthetic blobs, no model.
//
//     build/native_grouped_parity           the checks (a ctest; needs the GPU)
//     build/native_grouped_parity --bench   the checks, then v1/new timings of 48 calls in a graph (a window's layers)
//
// Every gate/up format with every down format; calls of 0..cap groups of 1..T entries each (an empty call is what the
// verify window's PCIe call usually is), entries from 0 and from past 0 (the PCIe call's follow the VRAM call's),
// scattered destinations, stale scratch, grid_groups 0 (= cap), 1, 2, 3, 4 and above cap: the output must be BITWISE
// equal to v1's, the rows the call must not write included (both start from the same NaN pattern).
//
// Random bytes are valid codes for every format here (every grid index is in range); only the fp16 block scales are
// set, small enough that the SwiGLU outputs keep a finite fp16 q8_1 scale.
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

namespace {

int g_fail = 0;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

template<typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 256), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 256), "memset");
    return p;
}

const char* name_of(int t) {
    switch (t) {
        case 16: return "IQ2_XXS";
        case 17: return "IQ2_XS";
        case 18: return "IQ3_XXS";
        case 20: return "IQ4_NL";
        case 21: return "IQ3_S";
        case 22: return "IQ2_S";
        case 23: return "IQ4_XS";
        case 29: return "IQ1_M";
        case 42: return "Q2_0";
        case 12: return "Q4_K";
        case 13: return "Q5_K";
        case 14: return "Q6_K";
        case 7: return "Q5_1";
        case 8: return "Q8_0";
        default: return "?";
    }
}
int block_values(int t) { return t == 20 || t == 7 || t == 8 ? 32 : t == 42 ? 64 : 256; }

// `rows` rows of `n` values of format t: random bytes, then a finite fp16 scale in every block
std::vector<uint8_t> random_rows(int t, int64_t rows, int64_t n, std::mt19937& rng) {
    const size_t rb = k::iq_row_bytes(t, n), bs = k::iq_row_bytes(t, block_values(t));
    std::vector<uint8_t> w((size_t) rows * rb);
    std::uniform_int_distribution<int> byte(0, 255), ex(2, 8), man(0, 1023), sgn(0, 1);
    for (auto& b : w) b = (uint8_t) byte(rng);
    for (size_t o = 0; o < w.size(); o += bs) {
        if (t == 29) {
            // IQ1_M: the fp16 scale is the top nibbles of its four scale words (bytes 48..55), the sign and the
            // exponent's top bits in the last: 0x1 / 0x2 there (0x9 / 0xA negative) keeps it in 2^-11 .. 2^-3
            const uint8_t nib = (uint8_t) ((sgn(rng) ? 0x8 : 0x0) | (1 + (byte(rng) & 1)));
            w[o + 55] = (uint8_t) ((w[o + 55] & 0x0F) | (nib << 4));
        } else {   // the other formats start their block with the fp16 scale: 2^-13 .. 2^-6
            // (Q4_K / Q5_K: 2^-14 .. 2^-11, their 6-bit sub-scales multiply it by up to 63)
            const int e = (t == 12 || t == 13) ? 1 + ex(rng) % 4 : ex(rng);
            const uint16_t h = (uint16_t) ((sgn(rng) ? 0x8000 : 0) | (e << 10) | man(rng));
            std::memcpy(&w[o], &h, 2);
            if (t == 12 || t == 13 || t == 7) {   // Q4_K / Q5_K: dmin, Q5_1: m - the second fp16, as small
                const int em = (t == 7) ? ex(rng) : 1 + ex(rng) % 4;
                const uint16_t m = (uint16_t) ((sgn(rng) ? 0x8000 : 0) | (em << 10) | man(rng));
                std::memcpy(&w[o + 2], &m, 2);
            }
        }
    }
    return w;
}

// one window's worth of expert calls on one format pair: blobs, T tokens' q8_1 activations, cap = T * K entries
struct Setup {
    k::NativeExpertLayout L;
    int T = 0, K = 0, cap = 0, n_blobs = 0;
    int active_lo = 0, active_hi = 0;
    size_t slot = 0;
    uint8_t* blobs = nullptr;
    uint8_t* xq = nullptr;
    uint8_t* scr = nullptr;
    size_t scr_bytes = 0;
    float* out = nullptr;
    size_t out_floats = 0;
    unsigned long long* ptr = nullptr;
    int32_t *start = nullptr, *n = nullptr, *dst = nullptr, *tok = nullptr;

    Setup(int gu, int dt, int64_t H, int64_t FF, int tokens, int k_per_token, int blobs_n, std::mt19937& rng,
          cudaStream_t s) {
        L = k::native_expert_layout(gu, dt, H, FF);
        T = tokens; K = k_per_token; cap = T * K; n_blobs = blobs_n;
        slot = (L.bytes + 255) & ~(size_t) 255;
        blobs = dalloc<uint8_t>(slot * (size_t) n_blobs);
        for (int b = 0; b < n_blobs; ++b) {
            // gate rows | up rows (format gu, H values each) | down rows (format dt, FF values each)
            auto blob = random_rows(gu, 2 * FF, H, rng);
            const auto down = random_rows(dt, H, FF, rng);
            blob.insert(blob.end(), down.begin(), down.end());
            if (blob.size() != L.bytes) { std::fprintf(stderr, "blob %zu != %zu bytes\n", blob.size(), L.bytes); std::exit(2); }
            ck(cudaMemcpy(blobs + slot * (size_t) b, blob.data(), blob.size(), cudaMemcpyHostToDevice), "blob");
        }
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> x((size_t) T * H);
        for (auto& v : x) v = nd(rng);
        float* dx = dalloc<float>(x.size());
        ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
        xq = dalloc<uint8_t>((size_t) T * (H / 32) * 36);
        k::quantize_q8_1_rows(dx, T, H, xq, s);
        ck(cudaStreamSynchronize(s), "quantize");
        cudaFree(dx);
        scr_bytes = k::native_expert_scratch_bytes(cap, FF);
        scr = dalloc<uint8_t>(scr_bytes);
        out_floats = (size_t) cap * H;
        out = dalloc<float>(out_floats);
        ptr = dalloc<unsigned long long>(cap);
        start = dalloc<int32_t>(cap + 1);
        n = dalloc<int32_t>(1);
        dst = dalloc<int32_t>(cap);
        tok = dalloc<int32_t>(cap);
    }
    ~Setup() {
        cudaFree(blobs); cudaFree(xq); cudaFree(scr); cudaFree(out);
        cudaFree(ptr); cudaFree(start); cudaFree(n); cudaFree(dst); cudaFree(tok);
    }
    // a plan: `groups` groups of 1..T entries from entry `base` on (as many as fit in cap)
    int plan(int groups, int base, std::mt19937& rng) {
        std::uniform_int_distribution<int> size(1, T);
        std::vector<int> sizes;
        int used = base;
        for (int g = 0; g < groups; ++g) {
            const int m = std::min(size(rng), cap - used);
            if (m <= 0) break;
            sizes.push_back(m);
            used += m;
        }
        return plan_sizes(sizes, base, rng);
    }
    // groups of these sizes from entry `base` on, each its own blob (as long as there are enough: a window's groups
    // are different experts); scattered destinations, random tokens
    int plan_sizes(const std::vector<int>& sizes, int base, std::mt19937& rng) {
        std::uniform_int_distribution<int> tk(0, T - 1);
        std::vector<int> order(n_blobs);
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);
        std::vector<unsigned long long> p;
        std::vector<int32_t> st(1, base), tk_v, ds(cap);
        for (const int m : sizes) {
            p.push_back((unsigned long long) (blobs + slot * (size_t) order[p.size() % order.size()]));
            st.push_back(st.back() + m);
        }
        if (st.back() > cap) { std::fprintf(stderr, "plan: %d entries > cap %d\n", st.back(), cap); std::exit(2); }
        const int ng = (int) p.size();
        active_lo = base;
        active_hi = st.back();
        std::iota(ds.begin(), ds.end(), 0);
        std::shuffle(ds.begin(), ds.end(), rng);                  // scattered rows of `out`
        for (int e = 0; e < cap; ++e) tk_v.push_back(tk(rng));
        if (ng) ck(cudaMemcpy(ptr, p.data(), p.size() * 8, cudaMemcpyHostToDevice), "ptr");
        ck(cudaMemcpy(start, st.data(), st.size() * 4, cudaMemcpyHostToDevice), "start");
        ck(cudaMemcpy(n, &ng, 4, cudaMemcpyHostToDevice), "n");
        ck(cudaMemcpy(dst, ds.data(), ds.size() * 4, cudaMemcpyHostToDevice), "dst");
        ck(cudaMemcpy(tok, tk_v.data(), tk_v.size() * 4, cudaMemcpyHostToDevice), "tok");
        return ng;
    }
    void call(int64_t grid_groups, cudaStream_t s) {
        k::native_expert_grouped(L, ptr, start, n, dst, tok, cap, cap, xq, scr, out, s, grid_groups);
    }
    // `out` after one call from the NaN pattern, the scratch holding `junk` (what earlier calls left there)
    std::vector<uint32_t> result(bool v1, int64_t grid_groups, int junk, cudaStream_t s) {
        ck(cudaMemset(out, 0xFF, out_floats * 4), "memset");
        ck(cudaMemset(scr, junk, scr_bytes), "memset");
        k::native_grouped_set_v1(v1);
        call(grid_groups, s);
        k::native_grouped_set_v1(false);
        ck(cudaStreamSynchronize(s), "sync");
        std::vector<uint32_t> o(out_floats);
        ck(cudaMemcpy(o.data(), out, o.size() * 4, cudaMemcpyDeviceToHost), "out");
        // Fused SwiGLU does not write h or inactive Q8 rows. Compare only the
        // active Q8_1 bytes, appended after the full output (including padding).
        const size_t fa = ((size_t) cap * L.n_ff * sizeof(float) + 255) & ~(size_t) 255;
        const size_t qrow = (size_t) (L.n_ff / 32) * 36;
        const size_t qb = (size_t) (active_hi - active_lo) * qrow;
        const size_t old = o.size();
        o.resize(old + qb / sizeof(uint32_t));
        if (qb)
            ck(cudaMemcpy(o.data() + old, scr + 3 * fa + active_lo * qrow, qb, cudaMemcpyDeviceToHost), "active hq");
        return o;
    }
};

void check(int gu, int dt, int64_t H, int64_t FF, cudaStream_t s, std::mt19937& rng) {
    Setup S(gu, dt, H, FF, 6, 4, 9, rng, s);   // 6 tokens x 4 experts: cap 24 entries
    int calls = 0, bad = 0;
    size_t written = 0;
    for (const int base : {0, 7}) {
        for (const int groups : {0, 1, 2, 3, 5, 8, 24}) {
            const int ng = S.plan(groups, base, rng);
            const auto ref = S.result(true, 0, 0x00, s);
            size_t w = 0;
            for (size_t i = 0; i < S.out_floats; ++i) w += ref[i] != 0xFFFFFFFFu;
            written += w;
            for (const int64_t gy : {(int64_t) 0, (int64_t) 1, (int64_t) 2, (int64_t) 3, (int64_t) 4, (int64_t) S.cap + 3}) {
                const auto got = S.result(false, gy, (calls & 1) ? 0x7F : 0x00, s);
                ++calls;
                size_t diff = 0;
                for (size_t i = 0; i < ref.size(); ++i) diff += ref[i] != got[i];
                if (diff) {
                    std::printf("  %-8s/%-7s base %d, %d groups, grid_groups %lld: %zu of %zu output/Q8 words differ  FAIL\n",
                                name_of(gu), name_of(dt), base, ng, (long long) gy, diff, ref.size());
                    ++bad;
                }
            }
        }
    }
    std::printf("%-8s/%-7s %5lld x %4lld  %d calls: %s (%zu output floats written in all)\n", name_of(gu), name_of(dt),
                (long long) H, (long long) FF, calls, bad ? "FAIL" : "bitwise equal to v1", written);
    if (written == 0) { std::printf("  nothing was written: the check checked nothing  FAIL\n"); ++bad; }
    g_fail += bad;
}

// ------------------------------------------------------------------------------------------------ --bench
// A window's 48 calls, one per layer, each with ITS OWN experts as the layers have: copies of the Setup's blobs in one
// arena, so the weights stream from DRAM as in the engine.  (48 calls re-reading the same 28 experts - 73 MB, about an
// Ada L2 - would read part of them from L2; the engine's layers share none.)
struct Layers {
    uint8_t* arena = nullptr;
    std::vector<unsigned long long*> ptr;           // per layer: its groups' blobs (device)
    Layers(Setup& S, int ng) {
        const int per = std::max(ng, 1);
        arena = dalloc<uint8_t>(S.slot * (size_t) (48 * per));
        for (int l = 0; l < 48; ++l) {
            std::vector<unsigned long long> p;
            for (int j = 0; j < per; ++j) {
                uint8_t* d = arena + S.slot * (size_t) (l * per + j);
                ck(cudaMemcpy(d, S.blobs + S.slot * (size_t) ((l * per + j) % S.n_blobs), S.L.bytes,
                              cudaMemcpyDeviceToDevice), "blob copy");
                p.push_back((unsigned long long) d);
            }
            unsigned long long* dp = dalloc<unsigned long long>(p.size());
            ck(cudaMemcpy(dp, p.data(), p.size() * 8, cudaMemcpyHostToDevice), "ptr");
            ptr.push_back(dp);
        }
    }
    Layers(const Layers&) = delete;
    Layers& operator=(const Layers&) = delete;
    ~Layers() {
        for (auto* p : ptr) cudaFree(p);
        cudaFree(arena);
    }
};

// the 48 calls of one variant captured in one graph
cudaGraphExec_t make_graph(cudaStream_t s, Setup& S, const Layers& Ls, bool v1, int64_t gy) {
    k::native_grouped_set_v1(v1);
    cudaGraph_t g;
    cudaGraphExec_t ge;
    ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "capture");
    for (int l = 0; l < 48; ++l)
        k::native_expert_grouped(S.L, Ls.ptr[l], S.start, S.n, S.dst, S.tok, S.cap, S.cap, S.xq, S.scr, S.out, s, gy);
    ck(cudaStreamEndCapture(s, &g), "capture");
    ck(cudaGraphInstantiate(&ge, g, 0), "instantiate");
    ck(cudaGraphDestroy(g), "graph");
    k::native_grouped_set_v1(false);
    return ge;
}

// one launch of the graph: microseconds per call
float launch_us(cudaStream_t s, cudaGraphExec_t ge, cudaEvent_t e0, cudaEvent_t e1) {
    ck(cudaEventRecord(e0, s), "record");
    ck(cudaGraphLaunch(ge, s), "launch");
    ck(cudaEventRecord(e1, s), "record");
    ck(cudaEventSynchronize(e1), "bench");
    float ms = 0;
    ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
    return 1e3f * ms / 48.0f;
}

float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// The two variants' launches ALTERNATE (v1-new, new-v1, ...) after a warm-up, and each reports its median: a card
// under sustained load steps its clock down after a fraction of a second (an RTX 4080 SUPER went from 80 to 89 us
// per gate/up call within one run), and whichever variant ran before the step would look faster.
constexpr int kWarmPairs = 20, kPairs = 40;

void bench(cudaStream_t s, std::mt19937& rng) {
    std::printf("\n--bench: microseconds per call, a window's 48 calls in a graph, each layer with its own experts (the\n"
                "weights stream from DRAM as in the engine); v1 / new (grid_groups): medians of %d launches each,\n"
                "alternating, after %d warm-up pairs - idle GPU assumed\n", kPairs, kWarmPairs);
    // a verify window's shape: 4 tokens x 10 experts, 2560 x 768 IQ3_S / IQ4_XS experts
    Setup S(21, 23, 2560, 768, 4, 10, 32, rng, s);
    std::vector<int> vram(16, 1);
    vram.insert(vram.end(), 12, 2);                                   // 28 groups, 40 entries
    const struct { const char* what; std::vector<int> sizes; int64_t gy; } rows[] = {
        {"no group (the PCIe call at pcie_frac 0)", {}, 4},
        {"1 group of 2 entries (a PCIe call)", {2}, 4},
        {"28 groups, 40 entries (a VRAM call)", vram, 0},
    };
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "event");
    ck(cudaEventCreate(&e1), "event");
    for (const auto& r : rows) {
        const int ng = S.plan_sizes(r.sizes, 0, rng);
        const Layers Ls(S, ng);                                       // 48 x 28 experts: 3.7 GB for the VRAM call
        cudaGraphExec_t ga = make_graph(s, S, Ls, true, 0), gb = make_graph(s, S, Ls, false, r.gy);
        std::vector<float> ta, tb;
        for (int i = 0; i < kWarmPairs + kPairs; ++i) {
            const bool a_first = i % 2 == 0;
            const float x = launch_us(s, a_first ? ga : gb, e0, e1), y = launch_us(s, a_first ? gb : ga, e0, e1);
            if (i < kWarmPairs) continue;
            ta.push_back(a_first ? x : y);
            tb.push_back(a_first ? y : x);
        }
        ck(cudaGraphExecDestroy(ga), "graph");
        ck(cudaGraphExecDestroy(gb), "graph");
        std::printf("  %-42s %8.2f / %8.2f us (grid_groups %lld)\n", r.what, median(ta), median(tb), (long long) r.gy);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
}

}  // namespace

int main(int argc, char** argv) {
    const bool do_bench = argc > 1 && std::string(argv[1]) == "--bench";
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    std::mt19937 rng(316);
    for (int gu : {16, 17, 18, 21, 22, 23, 29, 42, 12, 13,
#ifdef STRATA_Q6K_EXPERTS
                   14,
#endif
                   8})        // STRATA_GU_FMTS
        for (int dt : {20, 23, 42, 7, 8}) check(gu, dt, 512, 256, s, rng);   // STRATA_D_FMTS; IQ4_XS: n_ff % 256
    check(42, 42, 2560, 640, s, rng);
    check(21, 20, 2560, 640, s, rng);                                   // a model's shapes
    check(21, 23, 2560, 768, s, rng);
    if (do_bench) bench(s, rng);
    std::printf("native_grouped_parity: %d failures\n", g_fail);
    cudaStreamDestroy(s);
    return g_fail ? 1 : 0;
}
