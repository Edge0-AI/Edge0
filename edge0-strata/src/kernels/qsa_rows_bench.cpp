// src/kernels/qsa_rows_bench.cpp - QSAROWS (docs/QSAROWS.md): what the T rows of a verification window cost in a QSA
// layer's decode kernels at a long context, row by row.  GPU, synthetic data at the real geometry (24 query heads x
// 256, 2 KV heads, indexer 4 x 128, blocks of 4 cells, top-k 2048), INT8 pools, no model.
//
// The hypothesis under test: "the decode attention of the QSA layers works row by row, re-reading K/V for every row of
// the window, so 8 layers x ~0.25 ms explain the ~2.5 ms a row the stage costs".  The kernels measured here are the
// ones verify.cpp launches for a window of T rows at the usual long-context options (--max-context 262144, --kv int8):
//
//   scores   qsa_block_scores   (block_scores_multi_kernel: every pooled key read ONCE for the T queries)
//   topk     qsa_block_topk     (one CTA per query; max_blocks 65538 -> block_topk_wide_kernel)
//   attn     qsa_decode_attn_batch over each row's own selection, cap 2051 (33 chunks x 2 KV heads x T blocks)
//   append   native_qsa_indexer_append_steps for the T raw indexer keys (one CTA, the steps in sequence)
//   dense    qsa_decode_attn_batch over a 32768-cell window per row: the drafter's shape, which the hypothesis's
//            "~240 us a query" comes from (the drafter runs it with ONE query per step: capture_round's record_rest)
//
// --rows: block_scores_rows_kernel (STRATA_QSA_ROWS=1, docs/QSAROWS.md) against the multi kernel, scores and ids bit for
// bit, and its microseconds beside the reference's.
//
//   qsa_rows_bench [--device N] [--iters 100] [--ctx 131072] [--max-context 262144] [--rows] [--no-dense]
//   exit 0 (77: no GPU; 1: a --rows parity failure)
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/rope_scaling.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace k = strata::kernels;

namespace {
cudaStream_t st = nullptr;
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_rows_bench: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc((void**) &p, n * sizeof(T) + 256), "cudaMalloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 256), "cudaMemset");
    return p;
}
template <typename T> T* upload(const std::vector<T>& h) {
    T* d = dalloc<T>(h.size());
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
template <typename T> std::vector<T> download(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}
uint16_t f16_scale_q8(uint32_t r) { return (uint16_t) ((9u << 10) | (r & 0x3FFu)); }

struct Timer {
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    Timer() { ck(cudaEventCreate(&e0), "event"); ck(cudaEventCreate(&e1), "event"); }
    ~Timer() { cudaEventDestroy(e0); cudaEventDestroy(e1); }
    // microseconds per call over `iters` back-to-back calls, warmed; the median of three such runs
    template <typename F> double us(F&& f, int iters) {
        double r[3];
        for (int a = 0; a < 3; ++a) {
            for (int i = 0; i < 4; ++i) f();
            ck(cudaDeviceSynchronize(), "warm");
            ck(cudaEventRecord(e0, st), "record");
            for (int i = 0; i < iters; ++i) f();
            ck(cudaEventRecord(e1, st), "record");
            ck(cudaEventSynchronize(e1), "sync");
            float ms = 0.0f;
            ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
            r[a] = 1000.0 * ms / iters;
        }
        std::sort(r, r + 3);
        return r[1];
    }
};

constexpr int NH = 24, HD = 256, NKV = 2, IQ = 4, ID = 128;
constexpr int MAXT = 4;

struct Table {   // us[T] for T = 1..MAXT
    double us[MAXT + 1] = {};
    void print(const char* what) const {
        std::printf("  %-44s", what);
        for (int T = 1; T <= MAXT; ++T) std::printf(" T=%d %8.1f", T, us[T]);
        std::printf("  | +%.1f us a row\n", (us[MAXT] - us[1]) / (MAXT - 1));
    }
};
}  // namespace

int main(int argc, char** argv) {
    int device = 0, iters = 100;
    int64_t ctx = 131072, max_ctx = 262144;
    bool rows = false, dense = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) device = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) ctx = std::atoll(argv[++i]);
        else if (std::strcmp(argv[i], "--max-context") == 0 && i + 1 < argc) max_ctx = std::atoll(argv[++i]);
        else if (std::strcmp(argv[i], "--rows") == 0) rows = true;
        else if (std::strcmp(argv[i], "--no-dense") == 0) dense = false;
        else {
            std::fprintf(stderr, "usage: qsa_rows_bench [--device N] [--iters 100] [--ctx 131072] [--max-context 262144] "
                                 "[--rows] [--no-dense]\n");
            return 2;
        }
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: no GPU");
        return 77;
    }
    ck(cudaSetDevice(device), "cudaSetDevice");
    ck(cudaStreamCreate(&st), "stream");
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t PS = s.page_size;
    const int64_t max_cells = (max_ctx + PS - 1) / PS * PS;
    const int64_t max_blocks = max_cells / s.idx_block + 2;                       // verify.cpp: max_blocks_
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);              // verify.cpp: cap_ (2051)
    const int64_t cells = (ctx + MAXT + 64 + PS - 1) / PS * PS;                     // the pools' rows: resident cells
    const int64_t pages = cells / PS, prows = pages * NKV * PS;
    const int64_t dense_win = 32768, dense_cap = dense_win;                       // the drafter's --mtp-window
    std::printf("qsa_rows_bench on device %d (%s, %d.%d), %d iterations: context %lld, capacity %lld cells (max_blocks "
                "%lld), selection cap %lld, int8 pools over %lld cells%s\n",
                device, prop.name, prop.major, prop.minor, iters, (long long) ctx, (long long) max_cells,
                (long long) max_blocks, (long long) cap, (long long) cells,
                std::getenv("STRATA_QSA_ROWS") ? ", STRATA_QSA_ROWS set" : "");
    std::mt19937 rng(20261012u);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // ---- INT8 K/V pools, every page resident, identity page table
    std::vector<int8_t> c8((size_t) (prows * HD));
    for (auto& x : c8) x = (int8_t) ((int) (rng() % 255u) - 127);
    int8_t* k8 = upload(c8);
    for (auto& x : c8) x = (int8_t) ((int) (rng() % 255u) - 127);
    int8_t* v8 = upload(c8);
    c8.clear(); c8.shrink_to_fit();
    std::vector<uint16_t> sc((size_t) (prows * (HD / k::KV_Q8_GROUP)));
    for (auto& x : sc) x = f16_scale_q8(rng());
    uint16_t* k8s = upload(sc);
    for (auto& x : sc) x = f16_scale_q8(rng());
    uint16_t* v8s = upload(sc);
    std::vector<int32_t> tab((size_t) pages);
    for (int64_t i = 0; i < pages; ++i) tab[(size_t) i] = (int32_t) i;
    int32_t* table = upload(tab);
    k::QsaAttnPools pools;
    pools.page_table = table;
    pools.k_q = k8; pools.v_q = v8; pools.k_scale = k8s; pools.v_scale = v8s;

    // ---- the indexer's state: pooled keys (unit-ish rows), the spare, a tail, the block position
    std::vector<float> hp((size_t) (max_blocks * ID));
    for (auto& x : hp) x = nd(rng) * 0.09f;
    float* pooled = upload(hp);
    hp.clear(); hp.shrink_to_fit();
    std::vector<float> hd((size_t) ID);
    for (auto& x : hd) x = nd(rng) * 0.09f;
    float* dead = upload(hd);
    float* tail = dalloc<float>((size_t) (s.idx_block - 1) * ID);
    int32_t* block_pos = dalloc<int32_t>(1);
    std::vector<float> hg((size_t) ID, 1.0f);
    float* gamma = upload(hg);
    // the T rows' indexer queries: one query perturbed a little per row, so the selections overlap as a window's do
    std::vector<float> hq0((size_t) (IQ * ID)), hqi((size_t) (MAXT * IQ * ID));
    for (auto& x : hq0) x = nd(rng);
    for (int t = 0; t < MAXT; ++t)
        for (int i = 0; i < IQ * ID; ++i) hqi[(size_t) t * IQ * ID + i] = hq0[(size_t) i] + 0.15f * nd(rng);
    float* q_idx = upload(hqi);
    std::vector<float> hraw((size_t) (MAXT * ID));
    for (auto& x : hraw) x = nd(rng);
    float* raw = upload(hraw);
    // the attention queries [MAXT, NH, HD]
    std::vector<float> hq((size_t) (MAXT * NH * HD));
    for (auto& x : hq) x = nd(rng);
    float* q = upload(hq);
    // step records for rows 0..MAXT-1 at cells ctx .. ctx + MAXT - 1 (verify: pos, n_kv, n_bid, width)
    std::vector<int32_t> hst((size_t) (MAXT * k::kStepCount));
    for (int t = 0; t < MAXT; ++t) {
        const int64_t pos = ctx + t, n_kv = pos + 1;
        hst[(size_t) t * k::kStepCount + k::kStepPos] = (int32_t) pos;
        hst[(size_t) t * k::kStepCount + k::kStepNKv] = (int32_t) n_kv;
        hst[(size_t) t * k::kStepCount + k::kStepNBid] = (int32_t) (n_kv / s.idx_block);
        hst[(size_t) t * k::kStepCount + k::kStepWidth] = (int32_t) k::qsa_selection_width(n_kv, s);
    }
    int32_t* steps = upload(hst);
    // the drafter's dense window: the same cells, width = the window
    std::vector<int32_t> hsd(hst);
    for (int t = 0; t < MAXT; ++t) hsd[(size_t) t * k::kStepCount + k::kStepWidth] = (int32_t) dense_win;
    int32_t* steps_d = upload(hsd);
    std::vector<int32_t> hwid((size_t) (MAXT * dense_cap));
    for (int t = 0; t < MAXT; ++t)
        for (int64_t j = 0; j < dense_win; ++j) hwid[(size_t) t * dense_cap + j] = (int32_t) (ctx + t + 1 - dense_win + j);
    int32_t* ids_d = upload(hwid);
    hwid.clear(); hwid.shrink_to_fit();

    float* scores = dalloc<float>((size_t) (MAXT * max_blocks));
    int32_t* ids = dalloc<int32_t>((size_t) (MAXT * cap));
    const size_t scr_floats = (size_t) k::qsa_decode_attn_scratch_floats(cap, s);
    float* scratch = dalloc<float>(scr_floats * MAXT);
    float* attn = dalloc<float>((size_t) (MAXT * NH * HD));
    const size_t scr_floats_d = (size_t) k::qsa_decode_attn_scratch_floats(dense_cap, s);
    float* scratch_d = dalloc<float>(scr_floats_d * MAXT);
    float* attn_d = dalloc<float>((size_t) (MAXT * NH * HD));
    const k::QsaIndexerBuffers ib{tail, dead, pooled, block_pos};
    const k::RopeScaling scaling;

    // ---- the selections once (the attention's inputs), and how much the rows' selections overlap
    k::qsa_block_scores(pooled, dead, q_idx, steps, MAXT, max_blocks, s, scores, st);
    k::qsa_block_topk(scores, steps, MAXT, max_blocks, cap, s, ids, st);
    ck(cudaDeviceSynchronize(), "selection");
    {
        const std::vector<int32_t> hid = download(ids, (size_t) (MAXT * cap));
        std::printf("  selections: width %d a row;", hst[k::kStepWidth]);
        for (int T = 2; T <= MAXT; ++T) {
            std::set<int32_t> u;
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < hst[(size_t) t * k::kStepCount + k::kStepWidth]; ++j) u.insert(hid[(size_t) t * cap + j]);
            std::printf(" union of %d rows %zu cells (%.2fx one row)", T, u.size(), (double) u.size() / hst[k::kStepWidth]);
        }
        std::puts("");
    }

    Timer tm;
    Table t_scores, t_topk, t_attn, t_append, t_layer, t_dense;
    for (int T = 1; T <= MAXT; ++T) {
        t_scores.us[T] = tm.us([&] { k::qsa_block_scores(pooled, dead, q_idx, steps, T, max_blocks, s, scores, st); }, iters);
        t_topk.us[T] = tm.us([&] { k::qsa_block_topk(scores, steps, T, max_blocks, cap, s, ids, st); }, iters);
        t_attn.us[T] = tm.us([&] { k::qsa_decode_attn_batch(q, pools, ids, steps, cap, s, scratch, attn, T, st); }, iters);
        // the append writes the tail / pooled / block_pos the scores read; its own cells (ctx .. ctx + T - 1) land in
        // the tail (ctx % 4 == 0) and, at T == 4, complete block ctx / 4: the same state every call
        t_append.us[T] = tm.us([&] {
            k::native_qsa_indexer_append_steps(raw, steps + k::kStepPos, (int) k::kStepCount, T, 0, gamma, 1e-6f, ib, s,
                                               max_cells, scaling, st);
        }, iters);
        t_layer.us[T] = tm.us([&] {
            k::native_qsa_indexer_append_steps(raw, steps + k::kStepPos, (int) k::kStepCount, T, 0, gamma, 1e-6f, ib, s,
                                               max_cells, scaling, st);
            k::qsa_block_scores(pooled, dead, q_idx, steps, T, max_blocks, s, scores, st);
            k::qsa_block_topk(scores, steps, T, max_blocks, cap, s, ids, st);
            k::qsa_decode_attn_batch(q, pools, ids, steps, cap, s, scratch, attn, T, st);
        }, iters);
        if (dense)
            t_dense.us[T] = tm.us([&] {
                k::qsa_decode_attn_batch(q, pools, ids_d, steps_d, dense_cap, s, scratch_d, attn_d, T, st);
            }, iters);
    }
    std::puts("-- a QSA layer's decode kernels at the verification window's shape (us a call, median of 3 runs)");
    t_scores.print("scores  qsa_block_scores (multi)");
    t_topk.print("topk    qsa_block_topk");
    t_attn.print("attn    qsa_decode_attn_batch cap 2051");
    t_append.print("append  native_qsa_indexer_append_steps");
    t_layer.print("layer   append + scores + topk + attn");
    if (dense) {
        std::puts("-- the drafter's dense window (32768 cells a row): the shape the hypothesis's ~240 us a query comes from");
        t_dense.print("dense   qsa_decode_attn_batch cap 32768");
    }
    const double per_row = (t_layer.us[MAXT] - t_layer.us[1]) / (MAXT - 1);
    std::printf("-- per row: %.1f us a QSA layer -> x4 layers (stage 0) %.2f ms, x8 layers (stage 1) %.2f ms\n", per_row,
                4 * per_row / 1000.0, 8 * per_row / 1000.0);

    bool ok = true;
    if (rows) {
        // QSAROWS: block_scores_rows_kernel (STRATA_QSA_ROWS=1) against block_scores_multi_kernel, bit for bit on the scores
        // and on the ids the top-k draws from them, at every T; then the microseconds of the two
        std::puts("-- STRATA_QSA_ROWS: qsa_block_scores_rows against qsa_block_scores (multi), scores and ids bit for bit");
        float* scores_b = dalloc<float>((size_t) (MAXT * max_blocks));
        int32_t* ids_b = dalloc<int32_t>((size_t) (MAXT * cap));
        long long cases = 0, failed = 0;
        Table t_rows;
        for (int T = 1; T <= MAXT; ++T) {
            ck(cudaMemset(scores, 0xFF, (size_t) MAXT * max_blocks * 4), "fill");
            ck(cudaMemset(scores_b, 0xFF, (size_t) MAXT * max_blocks * 4), "fill");
            ck(cudaMemset(ids, 0x7F, (size_t) MAXT * cap * 4), "fill");
            ck(cudaMemset(ids_b, 0x7F, (size_t) MAXT * cap * 4), "fill");
            k::qsa_block_scores(pooled, dead, q_idx, steps, T, max_blocks, s, scores, st);
            if (!k::qsa_block_scores_rows(pooled, dead, q_idx, steps, T, max_blocks, s, scores_b, st)) {
                std::printf("  FAIL T=%d: qsa_block_scores_rows refused the call\n", T);
                ++cases; ++failed;
                continue;
            }
            k::qsa_block_topk(scores, steps, T, max_blocks, cap, s, ids, st);
            k::qsa_block_topk(scores_b, steps, T, max_blocks, cap, s, ids_b, st);
            ck(cudaDeviceSynchronize(), "rows sync");
            const std::vector<float> a = download(scores, (size_t) T * max_blocks), b = download(scores_b, (size_t) T * max_blocks);
            const std::vector<int32_t> ia = download(ids, (size_t) T * cap), ib2 = download(ids_b, (size_t) T * cap);
            ++cases;
            const bool same_s = std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
            const bool same_i = std::memcmp(ia.data(), ib2.data(), ia.size() * 4) == 0;
            if (!same_s || !same_i) {
                ++failed;
                size_t count = 0, first = a.size();
                for (size_t i = 0; i < a.size(); ++i)
                    if (std::memcmp(&a[i], &b[i], 4) != 0) { if (first == a.size()) first = i; ++count; }
                std::printf("  FAIL T=%d: scores %s (%zu of %zu differ, first at query %zu block %zu), ids %s\n", T,
                            same_s ? "same" : "DIFFER", count, a.size(), first / (size_t) max_blocks, first % (size_t) max_blocks,
                            same_i ? "same" : "DIFFER");
            }
            t_rows.us[T] = tm.us([&] { k::qsa_block_scores_rows(pooled, dead, q_idx, steps, T, max_blocks, s, scores_b, st); },
                                 iters);
        }
        // a short context too (n_bid small, the tail block with cells): rows at cells 100..103
        {
            std::vector<int32_t> hs2(hst);
            for (int t = 0; t < MAXT; ++t) {
                const int64_t pos = 100 + t, n_kv = pos + 1;
                hs2[(size_t) t * k::kStepCount + k::kStepPos] = (int32_t) pos;
                hs2[(size_t) t * k::kStepCount + k::kStepNKv] = (int32_t) n_kv;
                hs2[(size_t) t * k::kStepCount + k::kStepNBid] = (int32_t) (n_kv / s.idx_block);
                hs2[(size_t) t * k::kStepCount + k::kStepWidth] = (int32_t) k::qsa_selection_width(n_kv, s);
            }
            int32_t* steps2 = upload(hs2);
            for (int T = 1; T <= MAXT; ++T) {
                ck(cudaMemset(scores, 0xFF, (size_t) MAXT * max_blocks * 4), "fill");
                ck(cudaMemset(scores_b, 0xFF, (size_t) MAXT * max_blocks * 4), "fill");
                k::qsa_block_scores(pooled, dead, q_idx, steps2, T, max_blocks, s, scores, st);
                k::qsa_block_scores_rows(pooled, dead, q_idx, steps2, T, max_blocks, s, scores_b, st);
                ck(cudaDeviceSynchronize(), "rows sync");
                const std::vector<float> a = download(scores, (size_t) T * max_blocks), b = download(scores_b, (size_t) T * max_blocks);
                ++cases;
                if (std::memcmp(a.data(), b.data(), a.size() * 4) != 0) { ++failed; std::printf("  FAIL short context T=%d\n", T); }
            }
            ck(cudaFree(steps2), "free");
        }
        std::printf("  parity: %lld cases, %lld differ -> %s\n", cases, failed, failed ? "FAIL" : "bit-identical");
        ok = failed == 0;
        t_scores.print("scores  qsa_block_scores (multi, reference)");
        t_rows.print("scores  qsa_block_scores_rows");
        ck(cudaFree(scores_b), "free");
        ck(cudaFree(ids_b), "free");
    }
    for (void* p : {(void*) k8, (void*) v8, (void*) k8s, (void*) v8s, (void*) table, (void*) pooled, (void*) dead,
                    (void*) tail, (void*) block_pos, (void*) gamma, (void*) q_idx, (void*) raw, (void*) q, (void*) steps,
                    (void*) steps_d, (void*) ids_d, (void*) scores, (void*) ids, (void*) scratch, (void*) attn,
                    (void*) scratch_d, (void*) attn_d})
        ck(cudaFree(p), "free");
    std::printf("qsa_rows_bench: %s\n", ok ? "done" : "FAIL");
    return ok ? 0 : 1;
}
