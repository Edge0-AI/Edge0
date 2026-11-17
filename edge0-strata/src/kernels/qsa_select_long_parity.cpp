// src/kernels/qsa_select_long_parity.cpp - STRATA_SELECT_LONG's decode select (PR #3820): qsa_block_scores_long
// (block_scores_dx_kernel) and qsa_block_topk_long (the multi-CTA top-k) against qsa_block_scores and qsa_block_topk,
// bit for bit (GPU, synthetic data, no model).
//
// Needs the CUDA build with STRATA_EXPERIMENTAL_SM60 (elsewhere the _long calls forward to the plain ones and there is
// nothing to compare; the test skips with 77) and STRATA_SCORES_DX=1 STRATA_TOPK_MULTI=1 in the environment, which make
// the new kernels run on any card (on compute capability 6.x they are the default).  STRATA_TOPK_MULTI_N=<n> fixes the
// CTAs per query (the ctest "n3" case), otherwise one per 2,048 blocks of capacity.
//
//   scores: random pooled keys, queries and a dead key; nq 1..8 queries with their own n_bid (a decode window's
//           consecutive positions, so tail blocks of 0..3 cells), n_bid 0 / 1 / 31 / 32 / 33 / 1000 / 70001 ...; the
//           scores of every block 0..n_bid of every query are compared by memcmp (the tail block n_bid included).
//   top-k:  scores that are random, all equal, 7 distinct values (many ties across CTAs, so the "first eq_budget cells
//           at the threshold" rule is exercised) and random with a boosted tail block; contexts 1,000 (width equals
//           n_kv: the identity) up to 524,288 cells; nq 1, 2, 5, 8, 16; capacity max_blocks = the largest context / 4 + 2.
//           The selected ids of every query are compared on the first `width` entries. The multi-CTA scratch is filled
//           with junk before every call (the contract says no initial value) and must have been written (the new
//           kernels ran; a forwarded call would leave it untouched).
//
//   qsa_select_long_parity [--device N]     exit 0: all cases identical; 77: no GPU or a build without the kernels
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_select_long_parity: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc((void**) &p, n * sizeof(T) + 64), "cudaMalloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "cudaMemset");
    return p;
}
template <typename T> T* upload(const std::vector<T>& h) {
    T* d = dalloc<T>(h.size());
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}

// steps of a decode window: query i at position pos0 + i
std::vector<int32_t> make_steps(int nq, int64_t pos0, const k::QsaShapes& s) {
    std::vector<int32_t> st((size_t) nq * k::kStepCount, 0);
    for (int i = 0; i < nq; ++i) {
        int32_t* p = st.data() + (size_t) i * k::kStepCount;
        const int64_t pos = pos0 + i;
        p[k::kStepPos] = (int32_t) pos;
        p[k::kStepNKv] = (int32_t) (pos + 1);
        p[k::kStepNBid] = (int32_t) ((pos + 1) / s.idx_block);
        p[k::kStepWidth] = (int32_t) k::qsa_selection_width(pos + 1, s);
    }
    return st;
}

long long g_cases = 0, g_failed = 0;

void check_scores(const k::QsaShapes& s, int nq, int64_t n_bid0, std::mt19937& rng) {
    const int64_t D = s.idx_dim, H = s.idx_n_head;
    const int64_t max_blocks = n_bid0 + nq + 8;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> pooled((size_t) (max_blocks * D)), dead((size_t) D), q((size_t) (nq * H * D));
    for (auto& x : pooled) x = nd(rng);
    for (auto& x : dead) x = nd(rng);
    for (auto& x : q) x = nd(rng);
    // the window's last query has n_bid0 blocks; the others one position earlier each (their tails differ)
    const std::vector<int32_t> steps = make_steps(nq, n_bid0 * s.idx_block - 1 + (int64_t) (rng() % 4u) - (nq - 1), s);
    float *dp = upload(pooled), *dd = upload(dead), *dq = upload(q);
    int32_t* dst = upload(steps);
    float* a = dalloc<float>((size_t) (nq * max_blocks));
    float* b = dalloc<float>((size_t) (nq * max_blocks));
    ck(cudaMemset(a, 0xFF, (size_t) (nq * max_blocks) * 4), "fill a");
    ck(cudaMemset(b, 0xFF, (size_t) (nq * max_blocks) * 4), "fill b");
    k::qsa_block_scores(dp, dd, dq, dst, nq, max_blocks, s, a, nullptr);
    k::qsa_block_scores_long(dp, dd, dq, dst, nq, max_blocks, s, b, nullptr, -1);
    ck(cudaDeviceSynchronize(), "scores");
    std::vector<float> ha((size_t) (nq * max_blocks)), hb(ha.size());
    ck(cudaMemcpy(ha.data(), a, ha.size() * 4, cudaMemcpyDeviceToHost), "read a");
    ck(cudaMemcpy(hb.data(), b, hb.size() * 4, cudaMemcpyDeviceToHost), "read b");
    bool ok = true;
    for (int i = 0; i < nq && ok; ++i) {
        const int64_t nb = steps[(size_t) i * k::kStepCount + k::kStepNBid];
        for (int64_t j = 0; j <= nb; ++j) {   // block n_bid is the tail
            if (std::memcmp(&ha[(size_t) (i * max_blocks + j)], &hb[(size_t) (i * max_blocks + j)], 4) != 0) {
                std::printf("FAIL scores nq %d n_bid0 %lld: query %d block %lld: %.9g vs %.9g\n", nq, (long long) n_bid0, i,
                            (long long) j, ha[(size_t) (i * max_blocks + j)], hb[(size_t) (i * max_blocks + j)]);
                ok = false;
                break;
            }
        }
    }
    ++g_cases;
    if (!ok) ++g_failed;
    for (void* p : {(void*) dp, (void*) dd, (void*) dq, (void*) dst, (void*) a, (void*) b}) cudaFree(p);
}

void check_topk(const k::QsaShapes& s, int nq, int64_t ctx, int mode, int64_t max_blocks, std::mt19937& rng, bool& multi_ran) {
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    const std::vector<int32_t> steps = make_steps(nq, ctx - nq, s);
    std::vector<float> sc((size_t) (nq * max_blocks), -999.0f);
    std::uniform_real_distribution<float> ur(0.0f, 200.0f);
    for (int i = 0; i < nq; ++i) {
        const int32_t nb = steps[(size_t) i * k::kStepCount + k::kStepNBid];
        const bool tail = steps[(size_t) i * k::kStepCount + k::kStepNKv] % s.idx_block != 0;
        for (int64_t j = 0; j <= nb; ++j) {
            float v = mode == 1 ? 3.0f : mode == 2 ? (float) (rng() % 7u) : ur(rng);
            sc[(size_t) (i * max_blocks + j)] = v;
        }
        if (tail && mode != 1) sc[(size_t) (i * max_blocks + nb)] += 1e9f;
    }
    float* ds = upload(sc);
    int32_t* dst = upload(steps);
    int32_t* ia = dalloc<int32_t>((size_t) (nq * cap));
    int32_t* ib = dalloc<int32_t>((size_t) (nq * cap));
    const int64_t scratch_ints = k::qsa_topk_multi_scratch_ints(nq);
    int32_t* scratch = dalloc<int32_t>((size_t) scratch_ints);
    ck(cudaMemset(scratch, 0xFF, (size_t) scratch_ints * 4), "scratch junk");
    ck(cudaMemset(ia, 0xFF, (size_t) (nq * cap) * 4), "fill ia");
    ck(cudaMemset(ib, 0xFF, (size_t) (nq * cap) * 4), "fill ib");
    k::qsa_block_topk(ds, dst, nq, max_blocks, cap, s, ia, nullptr);
    k::qsa_block_topk_long(ds, dst, nq, max_blocks, cap, s, ib, nullptr, -1, scratch);
    ck(cudaDeviceSynchronize(), "topk");
    std::vector<int32_t> ha((size_t) (nq * cap)), hb(ha.size()), hs((size_t) scratch_ints);
    ck(cudaMemcpy(ha.data(), ia, ha.size() * 4, cudaMemcpyDeviceToHost), "read ia");
    ck(cudaMemcpy(hb.data(), ib, hb.size() * 4, cudaMemcpyDeviceToHost), "read ib");
    ck(cudaMemcpy(hs.data(), scratch, hs.size() * 4, cudaMemcpyDeviceToHost), "read scratch");
    // the multi kernels write the scratch (pass 0's histogram rows) when the contexts are above the identity width
    const bool touched = std::any_of(hs.begin(), hs.end(), [](int32_t v) { return v != -1; });
    if (touched) multi_ran = true;
    bool ok = true;
    for (int i = 0; i < nq && ok; ++i) {
        const int64_t w = steps[(size_t) i * k::kStepCount + k::kStepWidth];
        if (!std::equal(ha.begin() + i * cap, ha.begin() + i * cap + w, hb.begin() + i * cap)) {
            size_t first = 0;
            while (ha[(size_t) (i * cap) + first] == hb[(size_t) (i * cap) + first]) ++first;
            std::printf("FAIL topk nq %d ctx %lld mode %d max_blocks %lld: query %d entry %zu: %d vs %d\n", nq,
                        (long long) ctx, mode, (long long) max_blocks, i, first, ha[(size_t) (i * cap) + first],
                        hb[(size_t) (i * cap) + first]);
            ok = false;
        }
    }
    ++g_cases;
    if (!ok) ++g_failed;
    for (void* p : {(void*) ds, (void*) dst, (void*) ia, (void*) ib, (void*) scratch}) cudaFree(p);
}
}  // namespace

int main(int argc, char** argv) {
    int device = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) device = std::atoi(argv[++i]);
        else {
            std::fprintf(stderr, "usage: qsa_select_long_parity [--device N]\n");
            return 2;
        }
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: no GPU");
        return 77;
    }
    ck(cudaSetDevice(device), "cudaSetDevice");
    if (k::qsa_topk_multi_scratch_ints(1) <= 0) {
        std::puts("SKIP: this build has no multi-CTA select (CUDA with STRATA_EXPERIMENTAL_SM60)");
        return 77;
    }
    const char* dx = std::getenv("STRATA_SCORES_DX");
    const char* mt = std::getenv("STRATA_TOPK_MULTI");
    if (dx == nullptr || dx[0] != '1' || mt == nullptr || mt[0] != '1') {
        std::fprintf(stderr, "qsa_select_long_parity: set STRATA_SCORES_DX=1 and STRATA_TOPK_MULTI=1 (the kernels are "
                             "otherwise default only on compute capability 6.x)\n");
        return 2;
    }
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    std::printf("qsa_select_long_parity on device %d (%s, %d.%d), STRATA_TOPK_MULTI_N=%s\n", device, prop.name, prop.major,
                prop.minor, std::getenv("STRATA_TOPK_MULTI_N") ? std::getenv("STRATA_TOPK_MULTI_N") : "(per capacity)");

    const k::QsaShapes s = k::qsa_real_shapes();
    std::mt19937 rng(20261026u);
    for (int nq : {1, 2, 3, 5, 8})
        for (int64_t nb : {(int64_t) 8, (int64_t) 31, (int64_t) 32, (int64_t) 33, (int64_t) 1000, (int64_t) 70001, (int64_t) 131072})
            check_scores(s, nq, nb, rng);

    bool multi_ran = false;
    const int64_t max_ctx = 524288, max_blocks = max_ctx / s.idx_block + 2;
    for (int nq : {1, 2, 5, 8, 16})
        for (int64_t ctx : {(int64_t) 1000, (int64_t) 2051, (int64_t) 2055, (int64_t) 8192, (int64_t) 70000, (int64_t) 135164,
                            (int64_t) 135168, (int64_t) 300003, max_ctx})
            for (int mode = 0; mode < 3; ++mode)
                check_topk(s, nq, ctx, mode, max_blocks, rng, multi_ran);
    // a smaller capacity than the context asks for (max_blocks only bounds the row stride)
    for (int nq : {1, 4}) check_topk(s, nq, 40000, 0, 40000 / s.idx_block + 2, rng, multi_ran);
    if (!multi_ran) {
        std::printf("FAIL: the multi-CTA top-k never wrote its scratch: the new kernels did not run\n");
        ++g_failed;
    }
    std::printf("qsa_select_long_parity: %lld cases, %lld differ -> %s\n", g_cases, g_failed, g_failed == 0 ? "PASS" : "FAIL");
    return g_failed == 0 ? 0 : 1;
}
