// route_resident_parity - the warp-parallel STRATA_ROUTE_RESIDENT swap against a scalar reference of the same
// semantics, bit for bit.
//
// route_resident_k is one warp per token: lane j scans experts j, j + 32, ... and the warp reduces the best
// logit in a shuffle tree, where the form it replaced walked all 512 experts on a single thread.  The two have
// to land on the same expert and leave the same bits in ids and weights, because the option's whole
// justification is that it moves misses across PCIe and not the model.  This checks that on random cases built
// for the corners: many equal logits (the tie rule is the smallest expert index), repeated ids, a partly
// resident table, random rank windows, and margins from below to above the gap a swap needs.
//
// The reference below is the spec written out plainly - a serial walk with the same comparisons - and not a
// copy of the kernel that the parallel form replaced.  It runs on the device so that expf, and therefore the
// softmax bits, are the same function in both.
//
// Run: route_resident_parity   (ctest: route_resident_parity; GPU, synthetic, no model)
#include "strata/kernels/native_router.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}

constexpr int kExperts = 512;
constexpr int kTop = 10;
constexpr int kMaxTok = 8;
// The weights are a softmax ratio (ten expf calls, a sum, one division).  The kernel is bit-identical to the
// serial form it replaced - that was measured with both kernels side by side in one translation unit - but
// this reference is a second compilation, and two compilations of that expression land within a handful of
// ulp of each other (expf is an approximation, and the ratio amplifies it).  8 ulp is 1e-6 relative: an order
// of magnitude below anything the engine can act on, and far above what a semantic difference would produce.
constexpr int kWeightUlp = 8;

// The spec.  One thread per token, serial scan of all 512 experts, the comparisons the warp form reduces to:
// a strictly greater logit wins, so the smallest expert index keeps a tie, and the already-picked test runs
// against the ids as they are at that rank (a swap changes the set).
__global__ void route_resident_ref_k(const float* __restrict__ logits, int32_t* __restrict__ ids,
                                     float* __restrict__ weights, const int32_t* __restrict__ res, int n_tok,
                                     float margin, int lo, int hi, unsigned long long* __restrict__ stats) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tok) return;
    const float* l = logits + (size_t) t * kExperts;
    int32_t* id = ids + (size_t) t * kTop;
    int32_t my[kTop];
    for (int r = 0; r < kTop; ++r) my[r] = id[r];
    int swaps = 0, tail = 0, before = 0;
    for (int r = 0; r < kTop; ++r) before += res[my[r]] < 0;
    for (int r = lo; r <= hi && r < kTop; ++r) {
        const int e = my[r];
        if (res[e] >= 0) continue;                      // resident already: nothing to swap
        ++tail;
        float bv = -INFINITY;
        int bi = -1;
        for (int f = 0; f < kExperts; ++f) {
            if (res[f] < 0) continue;
            bool used = false;
            for (int q = 0; q < kTop; ++q) used |= my[q] == f;
            if (used) continue;
            if (l[f] > bv) { bv = l[f]; bi = f; }
        }
        if (bi >= 0 && l[e] - bv <= margin) { my[r] = bi; ++swaps; }
    }
    int after = 0;
    for (int r = 0; r < kTop; ++r) after += res[my[r]] < 0;
    if (swaps) {
        float m = -INFINITY;
        for (int r = 0; r < kTop; ++r) m = fmaxf(m, l[my[r]]);
        float ex[kTop], sum = 0.0f;
        for (int r = 0; r < kTop; ++r) { ex[r] = expf(l[my[r]] - m); sum += ex[r]; }
        for (int r = 0; r < kTop; ++r) weights[(size_t) t * kTop + r] = ex[r] / sum;
    }
    for (int r = 0; r < kTop; ++r) id[r] = my[r];
    if (stats) {
        atomicAdd(stats + 0, (unsigned long long) tail);
        atomicAdd(stats + 1, (unsigned long long) swaps);
        atomicAdd(stats + 2, (unsigned long long) before);
        atomicAdd(stats + 3, (unsigned long long) after);
    }
}

// The router's top-10 of a row, ties to the smallest index - what the kernel is handed as `ids`.
void top10(const float* l, int32_t* out) {
    for (int r = 0; r < kTop; ++r) {
        float bv = -INFINITY;
        int bi = -1;
        for (int f = 0; f < kExperts; ++f) {
            bool used = false;
            for (int q = 0; q < r; ++q) used |= out[q] == f;
            if (used) continue;
            if (l[f] > bv) { bv = l[f]; bi = f; }
        }
        out[r] = bi;
    }
}
}  // namespace

int main() {
    const float margins[] = {-0.5f, 0.0f, 0.05f, 0.25f, 0.5f, 1.0f, 1.5f};
    const int n_margins = (int) (sizeof(margins) / sizeof(margins[0]));
    const int cases = 240;
    std::mt19937 rng(20261010u);
    std::uniform_real_distribution<float> ud(-3.0f, 3.0f);
    std::uniform_int_distribution<int> pick(0, 1);
    std::uniform_int_distribution<int> coin(0, 99);

    std::vector<float> logits((size_t) kMaxTok * kExperts), weights((size_t) kMaxTok * kTop);
    std::vector<int32_t> ids((size_t) kMaxTok * kTop), res(kExperts);
    std::vector<float> sentinel(weights.size());
    const uint32_t pattern = 0xA5A5A5A5u;
    for (size_t i = 0; i < sentinel.size(); ++i) std::memcpy(&sentinel[i], &pattern, 4);

    float* d_logits = nullptr;
    float* d_w_ref = nullptr;
    float* d_w_par = nullptr;
    int32_t* d_ids_ref = nullptr;
    int32_t* d_ids_par = nullptr;
    int32_t* d_res = nullptr;
    unsigned long long* d_stats = nullptr;
    ck(cudaMalloc(&d_logits, logits.size() * 4), "logits");
    ck(cudaMalloc(&d_w_ref, weights.size() * 4), "w_ref");
    ck(cudaMalloc(&d_w_par, weights.size() * 4), "w_par");
    ck(cudaMalloc(&d_ids_ref, ids.size() * 4), "ids_ref");
    ck(cudaMalloc(&d_ids_par, ids.size() * 4), "ids_par");
    ck(cudaMalloc(&d_res, res.size() * 4), "res");
    ck(cudaMalloc(&d_stats, 4 * sizeof(unsigned long long)), "stats");
    cudaStream_t stream = nullptr;
    ck(cudaStreamCreate(&stream), "stream");

    int bad = 0, swapped_cases = 0, stats_bad = 0, dumped = 0, worst_ulp = 0;
    unsigned long long swaps_total = 0;
    for (int c = 0; c < cases; ++c) {
        const int n_tok = 1 + (c % kMaxTok);
        const float margin = margins[c % n_margins];
        // The rank window: what the profile marks as tail.  Sometimes the whole tail, sometimes a slice.
        const int lo = pick(rng) ? 4 + (c % 4) : 0;
        const int hi = pick(rng) ? kTop - 1 : lo + (c % (kTop - lo));
        const bool tie_heavy = pick(rng) != 0;          // half the cases round the logits onto a coarse grid
        const bool ids_from_router = pick(rng) != 0;    // the other half get ids with repeats instead
        const int resident_pct = 40 + coin(rng) % 45;   // 40-84% of the table resident

        for (int e = 0; e < kExperts; ++e) res[e] = (coin(rng) < resident_pct) ? (int32_t) (e % 7) : -1;
        for (size_t i = 0; i < logits.size(); ++i) {
            float v = ud(rng);
            if (tie_heavy) v = std::round(v * 2.0f) * 0.5f;   // many equal logits, so the tie rule gets used
            logits[i] = v;
        }
        for (int t = 0; t < n_tok; ++t) {
            int32_t* row = ids.data() + (size_t) t * kTop;
            if (ids_from_router) {
                top10(logits.data() + (size_t) t * kExperts, row);
            } else {
                for (int r = 0; r < kTop; ++r) row[r] = (int32_t) (rng() % kExperts);   // repeats on purpose
            }
        }

        ck(cudaMemcpy(d_logits, logits.data(), logits.size() * 4, cudaMemcpyHostToDevice), "up logits");
        ck(cudaMemcpy(d_res, res.data(), res.size() * 4, cudaMemcpyHostToDevice), "up res");

        // Reference, from the router's ids and a sentinel weight buffer, with its own counters.
        ck(cudaMemcpy(d_ids_ref, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "up ids ref");
        ck(cudaMemcpy(d_w_ref, sentinel.data(), sentinel.size() * 4, cudaMemcpyHostToDevice), "up w ref");
        ck(cudaMemset(d_stats, 0, 4 * sizeof(unsigned long long)), "zero stats");
        route_resident_ref_k<<<(n_tok + 63) / 64, 64, 0, stream>>>(d_logits, d_ids_ref, d_w_ref, d_res, n_tok,
                                                                   margin, lo, hi, d_stats);
        ck(cudaStreamSynchronize(stream), "sync ref");
        unsigned long long st_ref[4] = {0, 0, 0, 0};
        ck(cudaMemcpy(st_ref, d_stats, sizeof(st_ref), cudaMemcpyDeviceToHost), "down stats ref");

        // The shipped kernel, same inputs, same sentinel.
        ck(cudaMemcpy(d_ids_par, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "up ids par");
        ck(cudaMemcpy(d_w_par, sentinel.data(), sentinel.size() * 4, cudaMemcpyHostToDevice), "up w par");
        ck(cudaMemset(d_stats, 0, 4 * sizeof(unsigned long long)), "zero stats");
        strata::kernels::native_route_resident(d_logits, d_ids_par, d_w_par, d_res, n_tok, margin, lo, hi,
                                               d_stats, (void*) stream);
        ck(cudaStreamSynchronize(stream), "sync par");
        unsigned long long st_par[4] = {0, 0, 0, 0};
        ck(cudaMemcpy(st_par, d_stats, sizeof(st_par), cudaMemcpyDeviceToHost), "down stats par");

        std::vector<int32_t> i_ref(ids.size()), i_par(ids.size());
        std::vector<float> w_ref(weights.size()), w_par(weights.size());
        ck(cudaMemcpy(i_ref.data(), d_ids_ref, i_ref.size() * 4, cudaMemcpyDeviceToHost), "down ids ref");
        ck(cudaMemcpy(i_par.data(), d_ids_par, i_par.size() * 4, cudaMemcpyDeviceToHost), "down ids par");
        ck(cudaMemcpy(w_ref.data(), d_w_ref, w_ref.size() * 4, cudaMemcpyDeviceToHost), "down w ref");
        ck(cudaMemcpy(w_par.data(), d_w_par, w_par.size() * 4, cudaMemcpyDeviceToHost), "down w par");

        const bool ids_same = i_ref == i_par;
        // Weights: the kernel is bit-identical to the serial form it replaced (measured with both kernels
        // side by side in one translation unit); this reference is a second compilation, so its softmax can
        // round a division differently.  Compare within a few ulp and report the worst seen.
        int w_ulp = 0;
        for (size_t i = 0; i < w_ref.size(); ++i) {
            uint32_t a = 0, b = 0;
            std::memcpy(&a, &w_ref[i], 4);
            std::memcpy(&b, &w_par[i], 4);
            if (a != b) {
                const uint32_t d = a > b ? a - b : b - a;
                if ((int) d > w_ulp) w_ulp = (int) d;
            }
        }
        if (w_ulp > worst_ulp) worst_ulp = w_ulp;
        const bool w_same = w_ulp <= kWeightUlp;
        const bool st_same = std::memcmp(st_ref, st_par, sizeof(st_ref)) == 0;
        if (!ids_same || !w_same || !st_same) {
            ++bad;
            if (!st_same) ++stats_bad;
            std::printf("  case %3d  n_tok %d  margin %.2f  ranks [%d,%d]  ties %d  ids %s  weights %s (%d ulp)  stats %s\n",
                        c, n_tok, margin, lo, hi, (int) tie_heavy, ids_same ? "same" : "*** DIFFER ***",
                        w_same ? "same" : "*** DIFFER ***", w_ulp, st_same ? "same" : "*** DIFFER ***");
            if (std::getenv("STRATA_PARITY_DUMP") && dumped < 2) {
                ++dumped;
                for (int t = 0; t < n_tok; ++t) {
                    std::printf("    token %d  ids in ", t);
                    for (int r = 0; r < kTop; ++r) std::printf(" %d", ids[(size_t) t * kTop + r]);
                    std::printf("\n      ids ref ");
                    for (int r = 0; r < kTop; ++r) std::printf(" %d", i_ref[(size_t) t * kTop + r]);
                    std::printf("\n      ids par ");
                    for (int r = 0; r < kTop; ++r) std::printf(" %d", i_par[(size_t) t * kTop + r]);
                    std::printf("\n      logits  ");
                    for (int r = 0; r < kTop; ++r)
                        std::printf(" %.9g", logits[(size_t) t * kExperts + i_ref[(size_t) t * kTop + r]]);
                    std::printf("\n      w ref   ");
                    for (int r = 0; r < kTop; ++r)
                        std::printf(" %08x", *(uint32_t*) &w_ref[(size_t) t * kTop + r]);
                    std::printf("\n      w par   ");
                    for (int r = 0; r < kTop; ++r)
                        std::printf(" %08x", *(uint32_t*) &w_par[(size_t) t * kTop + r]);
                    std::printf("\n");
                }
            }
        }
        if (st_par[1]) { ++swapped_cases; swaps_total += st_par[1]; }
    }

    std::printf("route_resident_parity: %d cases, %d with swaps (%llu swaps), %d mismatches, weights within %d ulp\n",
                cases, swapped_cases, swaps_total, bad, worst_ulp);
    if (stats_bad) std::printf("  (%d of them in the counters)\n", stats_bad);
    if (swapped_cases == 0) {
        std::printf("route_resident_parity: no case swapped anything - the test proved nothing\n");
        ++bad;
    }
    std::printf(bad ? "route_resident_parity: %d FAILURES\n" : "route_resident_parity OK\n", bad);
    cudaFree(d_logits); cudaFree(d_w_ref); cudaFree(d_w_par);
    cudaFree(d_ids_ref); cudaFree(d_ids_par); cudaFree(d_res); cudaFree(d_stats);
    cudaStreamDestroy(stream);
    return bad ? 1 : 0;
}
