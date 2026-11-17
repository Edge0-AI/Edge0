// src/kernels/decode_cluster_parity.cpp - S19: the decode kernels on thread-block clusters (sm_90+) against the ones
// they replace, bit for bit (GPU, synthetic, no model).
//
//   QSA top-k : qsa_block_topk_cluster vs qsa_block_topk's one-CTA dispatch (the register kernel up to 33,792 blocks,
//               block_topk_kernel above) and vs qsa_block_topk_ref - every id of every row of the `cap`-wide buffer,
//               including the entries past the width that neither may write.
//   argmax    : sample_greedy_cluster vs sample_tokens' one-block greedy kernel.
//
// Inputs: random scores like a real indexer's, and adversarial ones - all equal, a few distinct values (ties at the
// threshold across the CTAs of a cluster), NaN, +-inf, -0.0 against +0.0, denormals, monotone rows, all NaN; contexts
// from below the selection width (the identity) to 262,144 cells, every tail weight (n_kv mod 4), 1-16 queries; and
// one CUDA graph that captures both cluster launches, replayed with new inputs.
//
//   decode_cluster_parity --selftest     parity (exit 0 = identical everywhere); skipped (exit 0) below sm_90
//   decode_cluster_parity --bench        per-call times, old vs new, at 32K / 128K / 262K
//
// The old paths are reached through the public dispatchers with STRATA_QSA_CLUSTER=0 / STRATA_ARGMAX_MULTI=0, set
// here before their first call (they read the variables once).
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/sampler.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

template <typename T> struct Dev {
    T* p = nullptr;
    size_t n = 0;
    explicit Dev(size_t count) : n(count) { ck(cudaMalloc(&p, count * sizeof(T) + 64), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    void put(const std::vector<T>& h) {
        ck(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "up");
        // #548 #536: a pageable cudaMemcpy can return before its DMA lands, and the graph replays and the timing
        // cases run on a cudaStreamNonBlocking stream that does not wait for the legacy stream - so wait here
        ck(cudaDeviceSynchronize(), "up landed");
    }
    std::vector<T> get() const {
        std::vector<T> h(n);
        ck(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "down");
        return h;
    }
};

const float kNaN = std::numeric_limits<float>::quiet_NaN(), kInf = std::numeric_limits<float>::infinity();

// ---- QSA top-k fixtures
enum Pattern { kRandom, kAllEqual, kFewValues, kNaNs, kInfs, kSignedZero, kDenormal, kAscending, kDescending, kAllNaN,
               kQuantized, kPatterns };
const char* pattern_name(int p) {
    static const char* n[] = {"random", "all-equal", "few-values", "nan", "inf", "signed-zero", "denormal", "ascending",
                              "descending", "all-nan", "quantized"};
    return n[p];
}

// one query's row of block scores (blocks 0..n_bid); the tail block gets the +1e9 the score kernel gives it
void fill_row(float* row, int64_t n_kv, int64_t n_bid, int pattern, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> pick(0, 99);
    for (int64_t b = 0; b <= n_bid; ++b) {
        float v = 0.f;
        const float g = nd(rng);
        switch (pattern) {
            case kRandom: v = std::fmax(0.f, g) * 3.f + std::fmax(0.f, nd(rng)); break;   // relu sums: exact zeros too
            case kAllEqual: v = 1.f; break;
            case kFewValues: v = (float) (pick(rng) % 4) * 0.5f; break;
            case kNaNs: v = pick(rng) < 15 ? kNaN : std::fabs(g); break;
            case kInfs: {
                const int r = pick(rng);
                v = r < 5 ? kInf : r < 15 ? -kInf : r < 30 ? -std::fabs(g) : g;
                break;
            }
            case kSignedZero: v = pick(rng) < 50 ? -0.0f : (pick(rng) < 50 ? 0.0f : -std::fabs(g) * 1e-3f); break;
            case kDenormal: v = (float) (pick(rng) % 7) * 1e-41f; break;
            case kAscending: v = (float) b; break;
            case kDescending: v = (float) (n_bid - b); break;
            case kAllNaN: v = kNaN; break;
            case kQuantized: v = std::round(std::fabs(g) * 8.f) / 8.f; break;
        }
        row[b] = v;
    }
    if (n_kv % 4 != 0 && pattern != kAllNaN && pattern != kAllEqual) row[n_bid] += 1e9f;
}

struct TopkCase {
    int64_t ctx;         // the last query's n_kv
    int64_t capacity;    // --max-context: max_blocks = capacity / 4 + 2 (the engine's rule)
    int nq;
    int pattern;
};

int run_topk_case(const TopkCase& c, std::mt19937& rng, bool verbose) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t max_blocks = c.capacity / 4 + 2, cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    std::vector<float> sc((size_t) (c.nq * max_blocks), 0.f);
    std::vector<int32_t> steps((size_t) c.nq * k::kStepCount);
    for (int i = 0; i < c.nq; ++i) {
        const int64_t n_kv = c.ctx - (c.nq - 1 - i);   // the window's queries, one cell apart, the last at ctx
        int32_t* st = steps.data() + (size_t) i * k::kStepCount;
        st[k::kStepPos] = (int32_t) (n_kv - 1);
        st[k::kStepNKv] = (int32_t) n_kv;
        st[k::kStepNBid] = (int32_t) (n_kv / s.idx_block);
        st[k::kStepWidth] = (int32_t) k::qsa_selection_width(n_kv, s);
        fill_row(sc.data() + (size_t) i * max_blocks, n_kv, n_kv / s.idx_block, c.pattern, rng);
    }
    Dev<float> d_sc(sc.size());
    const size_t n_ids = (size_t) (c.nq * cap);
    Dev<int32_t> d_st(steps.size()), d_new(n_ids), d_old(n_ids), d_ref(n_ids);
    d_sc.put(sc);
    d_st.put(steps);
    ck(cudaMemset(d_new.p, 0xA5, d_new.n * 4), "memset");
    ck(cudaMemset(d_old.p, 0xA5, d_old.n * 4), "memset");
    ck(cudaMemset(d_ref.p, 0xA5, d_ref.n * 4), "memset");
    if (!k::qsa_block_topk_cluster(d_sc.p, d_st.p, c.nq, max_blocks, cap, s, d_new.p, nullptr)) {
        std::printf("FAIL topk ctx %lld cap %lld nq %d: the cluster kernel did not run\n", (long long) c.ctx,
                    (long long) c.capacity, c.nq);
        return 1;
    }
    k::qsa_block_topk(d_sc.p, d_st.p, c.nq, max_blocks, cap, s, d_old.p, nullptr);
    k::qsa_block_topk_ref(d_sc.p, d_st.p, c.nq, max_blocks, cap, s, d_ref.p, nullptr);
    ck(cudaDeviceSynchronize(), "topk sync");
    const std::vector<int32_t> a = d_new.get(), b = d_old.get(), r = d_ref.get();
    int64_t diff_old = 0, diff_ref = 0, first = -1;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) { ++diff_old; if (first < 0) first = (int64_t) i; }
        if (a[i] != r[i]) ++diff_ref;
    }
    const bool ok = diff_old == 0 && diff_ref == 0;
    if (!ok || verbose)
        std::printf("%s topk ctx %7lld cap %7lld nq %2d %-11s: %lld ids differ from the one-CTA dispatch, %lld from "
                    "the reference%s\n", ok ? "ok  " : "FAIL", (long long) c.ctx, (long long) c.capacity, c.nq,
                    pattern_name(c.pattern), (long long) diff_old, (long long) diff_ref,
                    first >= 0 ? (" (first at " + std::to_string(first) + ")").c_str() : "");
    return ok ? 0 : 1;
}

// ---- argmax fixtures
enum AmPattern { aRandom, aTies, aAllEqual, aAllNegInf, aAllNaN, aNaNMax, aPosInf, aSignedZero, aLast, aFirst,
                 aDenormal, aTiesAcross, aPatterns };
const char* am_name(int p) {
    static const char* n[] = {"random", "ties", "all-equal", "all--inf", "all-nan", "nan+max", "+inf", "signed-zero",
                              "max-last", "max-first", "denormal", "ties-across"};
    return n[p];
}

void fill_logits(float* l, int nv, int pattern, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 3.f);
    std::uniform_int_distribution<int> at(0, nv - 1);
    for (int v = 0; v < nv; ++v) l[v] = nd(rng);
    switch (pattern) {
        case aRandom: break;
        case aTies: { const int n = 1 + at(rng) % 9; for (int i = 0; i < n; ++i) l[at(rng)] = 100.f; break; }
        case aAllEqual: for (int v = 0; v < nv; ++v) l[v] = 0.25f; break;
        case aAllNegInf: for (int v = 0; v < nv; ++v) l[v] = -kInf; break;
        case aAllNaN: for (int v = 0; v < nv; ++v) l[v] = kNaN; break;
        case aNaNMax:
            for (int v = 0; v < nv; ++v) if (at(rng) % 3 == 0) l[v] = kNaN;
            l[at(rng)] = 50.f;
            l[at(rng)] = 50.f;
            break;
        case aPosInf: for (int i = 0; i < 4; ++i) l[at(rng)] = kInf; break;
        case aSignedZero:
            for (int v = 0; v < nv; ++v) l[v] = -std::fabs(l[v]) - 1.f;
            for (int i = 0; i < 6; ++i) l[at(rng)] = (i & 1) ? 0.0f : -0.0f;
            break;
        case aLast: l[nv - 1] = 1e30f; break;
        case aFirst: l[0] = 1e30f; if (nv > 1) l[nv - 1] = 1e30f; break;
        case aDenormal: for (int v = 0; v < nv; ++v) l[v] = (float) (at(rng) % 5) * 1e-42f - 1e-30f * (v & 1); break;
        case aTiesAcross:   // the same maximum in several CTAs' and threads' ranges (8 x 1024 stride), plus -inf, NaN
            for (int v = 0; v < nv; v += 1 + at(rng) % 9000) l[v] = 7.f;
            for (int v = 3; v < nv; v += 11) l[v] = kNaN;
            for (int v = 5; v < nv; v += 13) l[v] = -kInf;
            break;
    }
}

int run_argmax_case(int nv, int T, int pattern, std::mt19937& rng, bool verbose) {
    std::vector<float> l((size_t) nv * T);
    for (int t = 0; t < T; ++t) fill_logits(l.data() + (size_t) t * nv, nv, t % 2 == 0 ? pattern : aRandom, rng);
    Dev<float> d_l(l.size());
    Dev<int> d_new((size_t) T), d_old((size_t) T);
    d_l.put(l);
    ck(cudaMemset(d_new.p, 0xA5, (size_t) T * 4), "memset");
    ck(cudaMemset(d_old.p, 0x5A, (size_t) T * 4), "memset");
    k::SamplerParams sp;
    sp.greedy = true;
    sp.temperature = 0.f;
    if (!k::sample_greedy_cluster(d_l.p, T, nv, d_new.p, nullptr)) {
        std::printf("FAIL argmax nv %d T %d: the cluster kernel did not run\n", nv, T);
        return 1;
    }
    k::sample_tokens(d_l.p, T, nv, nullptr, 0, sp, d_old.p, nullptr);
    ck(cudaDeviceSynchronize(), "argmax sync");
    const std::vector<int> a = d_new.get(), b = d_old.get();
    // and an independent host answer: the lowest index of the largest non-NaN value above -inf, else 0
    int bad = 0;
    for (int t = 0; t < T; ++t) {
        const float* r = l.data() + (size_t) t * nv;
        int hb = -1;
        for (int v = 0; v < nv; ++v)
            if (r[v] > -kInf && (hb < 0 || r[v] > r[hb])) hb = v;
        if (hb < 0) hb = 0;
        if (a[(size_t) t] != b[(size_t) t] || a[(size_t) t] != hb) ++bad;
    }
    if (bad || verbose)
        std::printf("%s argmax nv %6d T %d %-11s: %d rows differ (new %d, old %d)\n", bad ? "FAIL" : "ok  ", nv, T,
                    am_name(pattern), bad, a[0], b[0]);
    return bad ? 1 : 0;
}

// both cluster launches captured in one graph, replayed over new inputs written into the same buffers
int run_graph_case(std::mt19937& rng) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int nq = 5, nv = 248320;
    const int64_t capacity = 131072 + 272, max_blocks = capacity / 4 + 2;
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    Dev<float> d_sc((size_t) (nq * max_blocks)), d_l((size_t) nv * nq);
    Dev<int32_t> d_st((size_t) nq * k::kStepCount), d_new((size_t) (nq * cap)), d_old((size_t) (nq * cap));
    Dev<int> a_new((size_t) nq), a_old((size_t) nq);
    ck(cudaMemset(d_new.p, 0, d_new.n * 4), "memset");   // the entries past a row's width: written by neither
    ck(cudaMemset(d_old.p, 0, d_old.n * 4), "memset");
    cudaStream_t cs;
    ck(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "stream");
    cudaGraph_t g = nullptr;
    cudaGraphExec_t ge = nullptr;
    ck(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal), "begin capture");
    const bool l1 = k::qsa_block_topk_cluster(d_sc.p, d_st.p, nq, max_blocks, cap, s, d_new.p, cs);
    const bool l2 = k::sample_greedy_cluster(d_l.p, nq, nv, a_new.p, cs);
    ck(cudaStreamEndCapture(cs, &g), "end capture");
    if (!l1 && !l2) {   // no cluster code here (a card or a build below sm_90): nothing to compare
        if (g) cudaGraphDestroy(g);
        cudaStreamDestroy(cs);
        return -1;
    }
    if (!l1 || !l2) { std::printf("FAIL graph: one cluster kernel did not run\n"); return 1; }
    ck(cudaGraphInstantiate(&ge, g, 0), "instantiate");
    int fails = 0;
    for (int rep = 0; rep < 6; ++rep) {
        const int64_t ctx = 100000 + rep * 4001 + rep % 4;
        std::vector<float> sc((size_t) (nq * max_blocks), 0.f), l((size_t) nv * nq);
        std::vector<int32_t> steps((size_t) nq * k::kStepCount);
        for (int i = 0; i < nq; ++i) {
            const int64_t n_kv = ctx - (nq - 1 - i);
            int32_t* st = steps.data() + (size_t) i * k::kStepCount;
            st[k::kStepPos] = (int32_t) (n_kv - 1);
            st[k::kStepNKv] = (int32_t) n_kv;
            st[k::kStepNBid] = (int32_t) (n_kv / 4);
            st[k::kStepWidth] = (int32_t) k::qsa_selection_width(n_kv, s);
            fill_row(sc.data() + (size_t) i * max_blocks, n_kv, n_kv / 4, rep % 2 ? kFewValues : kRandom, rng);
            fill_logits(l.data() + (size_t) i * nv, nv, rep % 2 ? aTiesAcross : aRandom, rng);
        }
        d_sc.put(sc);
        d_st.put(steps);
        d_l.put(l);
        ck(cudaGraphLaunch(ge, cs), "graph launch");
        k::qsa_block_topk(d_sc.p, d_st.p, nq, max_blocks, cap, s, d_old.p, cs);
        k::SamplerParams sp;
        sp.greedy = true;
        k::sample_tokens(d_l.p, nq, nv, nullptr, 0, sp, a_old.p, cs);
        ck(cudaStreamSynchronize(cs), "graph sync");
        const auto x = d_new.get(), y = d_old.get();
        const auto u = a_new.get(), v = a_old.get();
        int64_t d1 = 0, d2 = 0;
        for (size_t i = 0; i < x.size(); ++i) d1 += x[i] != y[i];
        for (size_t i = 0; i < u.size(); ++i) d2 += u[i] != v[i];
        if (d1 || d2) ++fails;
        std::printf("%s graph replay %d (ctx %lld): %lld ids, %lld tokens differ\n", d1 || d2 ? "FAIL" : "ok  ", rep,
                    (long long) ctx, (long long) d1, (long long) d2);
    }
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
    cudaStreamDestroy(cs);
    return fails;
}

// back-to-back launches on `cs` (the engine's are too: one stream, inside a graph)
template <typename F> float time_us(cudaStream_t cs, F&& f, int reps = 300) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int i = 0; i < 5; ++i) f();
    cudaEventRecord(a, cs);
    for (int i = 0; i < reps; ++i) f();
    cudaEventRecord(b, cs);
    ck(cudaEventSynchronize(b), "time");
    float ms = 0.f;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms * 1000.f / (float) reps;
}

int bench() {
    std::mt19937 rng(11);
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    cudaStream_t cs;
    ck(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "stream");
    std::printf("QSA top-k, per call (us); capacity = context + 272 as the speed runs set --max-context\n");
    for (const int64_t ctx : {32768LL, 131072LL, 262144LL - 272}) {
        for (const int nq : {1, 5}) {
            const int64_t capacity = ctx + 272, max_blocks = capacity / 4 + 2;
            std::vector<float> sc((size_t) (nq * max_blocks), 0.f);
            std::vector<int32_t> steps((size_t) nq * k::kStepCount);
            for (int i = 0; i < nq; ++i) {
                const int64_t n_kv = ctx - (nq - 1 - i);
                int32_t* st = steps.data() + (size_t) i * k::kStepCount;
                st[k::kStepPos] = (int32_t) (n_kv - 1);
                st[k::kStepNKv] = (int32_t) n_kv;
                st[k::kStepNBid] = (int32_t) (n_kv / 4);
                st[k::kStepWidth] = (int32_t) k::qsa_selection_width(n_kv, s);
                fill_row(sc.data() + (size_t) i * max_blocks, n_kv, n_kv / 4, kRandom, rng);
            }
            Dev<float> d_sc(sc.size());
            Dev<int32_t> d_st(steps.size()), d_ids((size_t) (nq * cap));
            d_sc.put(sc);
            d_st.put(steps);
            const float* sp = d_sc.p;
            const int32_t* st = d_st.p;
            int32_t* ids = d_ids.p;
            const float t_old = time_us(cs, [&] { k::qsa_block_topk(sp, st, nq, max_blocks, cap, s, ids, cs); });
            const float t_ref = time_us(cs, [&] { k::qsa_block_topk_ref(sp, st, nq, max_blocks, cap, s, ids, cs); });
            const float t_new = time_us(cs, [&] { k::qsa_block_topk_cluster(sp, st, nq, max_blocks, cap, s, ids, cs); });
            std::printf("  ctx %6lld nq %d: one-CTA dispatch %7.1f  (ref %7.1f)  cluster %6.1f  -> %.1fx\n",
                        (long long) ctx, nq, t_old, t_ref, t_new, t_old / t_new);
        }
    }
    std::printf("argmax, per call (us)\n");
    for (const int nv : {248320, 32768}) {
        for (const int T : {1, 5}) {
            std::vector<float> l((size_t) nv * T);
            for (int t = 0; t < T; ++t) fill_logits(l.data() + (size_t) t * nv, nv, aRandom, rng);
            Dev<float> d_l(l.size());
            Dev<int> d_o((size_t) T);
            d_l.put(l);
            k::SamplerParams sp;
            sp.greedy = true;
            const float t_old = time_us(cs, [&] { k::sample_tokens(d_l.p, T, nv, nullptr, 0, sp, d_o.p, cs); });
            const float t_new = time_us(cs, [&] { k::sample_greedy_cluster(d_l.p, T, nv, d_o.p, cs); });
            std::printf("  n_vocab %6d T %d: one block %6.1f  cluster %5.1f  -> %.1fx\n", nv, T, t_old, t_new,
                        t_old / t_new);
        }
    }
    cudaStreamDestroy(cs);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    set_env("STRATA_QSA_CLUSTER", "0");     // the dispatchers: the one-CTA kernels (the cluster ones called directly)
    set_env("STRATA_ARGMAX_MULTI", "0");
    const bool do_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    const bool verbose = argc > 2 && std::strcmp(argv[2], "-v") == 0;
    int dev = 0, major = 0;
    ck(cudaGetDevice(&dev), "device");
    ck(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev), "cc");
    std::mt19937 rng(20261002);
    // first, so that both kernels' first calls (their per-device checks and opt-ins) happen inside a stream capture,
    // as in the engine's decode graphs
    const int gfails = run_graph_case(rng);
    if (gfails < 0) {
        std::printf("SKIP: no thread-block clusters on this card / build (compute capability %d)\n", major);
        return 0;
    }
    if (do_bench) return bench();
    int fails = 0, cases = 0;
    // contexts: below the width (identity), at it, just past it, odd tails, the register kernel's range, past it
    // (block_topk_kernel), up to 262,144; capacities: tight (the speed runs) and 262,144 (a long --max-context)
    const int64_t ctxs[] = {7, 2050, 2051, 2052, 2053, 2054, 2055, 3001, 9000, 32768, 32769, 65538, 100003,
                            131072, 135170, 135171, 200001, 262142, 262143, 262144};
    for (const int64_t ctx : ctxs) {
        for (const int64_t capacity : {ctx + 272, (int64_t) 262144 + 272}) {
            if (capacity < ctx) continue;
            for (int p = 0; p < kPatterns; ++p) {
                const int nq = (p % 3 == 0) ? 1 : (p % 3 == 1 ? 5 : 16);
                if (nq > ctx) continue;
                fails += run_topk_case({ctx, capacity, nq, p}, rng, verbose);
                ++cases;
            }
        }
    }
    std::printf("QSA top-k: %d cases, %d failed\n", cases, fails);
    int afails = 0, acases = 0;
    for (const int nv : {248320, 248321, 248319, 32771, 24576, 8193, 1000, 33, 1}) {
        for (int p = 0; p < aPatterns; ++p) {
            for (const int T : {1, 5}) {
                afails += run_argmax_case(nv, T, p, rng, verbose);
                ++acases;
            }
        }
    }
    std::printf("argmax: %d cases, %d failed\n", acases, afails);
    const int total = fails + afails + gfails;
    std::printf("%s\n", total == 0 ? "PASS: the cluster kernels are bitwise identical" : "FAIL");
    return total == 0 ? 0 : 1;
}
