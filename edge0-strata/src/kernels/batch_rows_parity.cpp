// src/kernels/batch_rows_parity.cpp - the batch window's one-launch-for-all-rows kernels (#2234, STRATA_BATCH_ATTN_ROWS /
// STRATA_BATCH_GDN_GROUPS) against the per-row launches they replace, bitwise (GPU, synthetic, no model).
//
//   1. qsa_block_scores_rows   == qsa_block_scores, one call per row, each row with its OWN indexer keys;
//   2. qsa_decode_attn_batch_rows == qsa_decode_attn_batch, one call per row, each row over its OWN K/V pools
//      (int8 codes + scales, and fp16);
//   3. gdn_step_norm_groups    == gdn_step_norm_multi, one call per slot group, in the window (all outputs, the state
//      untouched) and in the 1-row commit (the state written after `keep` tokens).
// Every case compares with memcmp, not a tolerance. Usage: batch_rows_parity
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* up(const std::vector<T>& h) {
    T* d = nullptr;
    ck(cudaMalloc(&d, h.size() * sizeof(T) + 64), "malloc");
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
template <typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}
uint16_t f2h(float f) { __half h = __float2half(f); return *reinterpret_cast<uint16_t*>(&h); }

int bad = 0;
void report(const char* what, bool same) {
    std::printf("  %-64s %s\n", what, same ? "bitwise" : "*** DIFFERS ***");
    if (!same) ++bad;
}

// ---- 1. the block scores ----
void scores_case(int rows) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t max_blocks = 3000;
    std::mt19937 rng(11 + rows);
    std::normal_distribution<float> nd(0.f, 1.f);
    k::QsaScoreRows sr;
    sr.n = rows;
    std::vector<const float*> pooled(rows), dead(rows);
    std::vector<float> q((size_t) rows * s.idx_n_head * s.idx_dim);
    for (auto& x : q) x = nd(rng);
    std::vector<int32_t> steps((size_t) rows * k::kStepCount);
    for (int r = 0; r < rows; ++r) {
        std::vector<float> p((size_t) max_blocks * s.idx_dim), d((size_t) s.idx_dim);
        for (auto& x : p) x = nd(rng);
        for (auto& x : d) x = nd(rng);
        pooled[r] = up(p);
        dead[r] = up(d);
        sr.pooled[r] = pooled[r];
        sr.dead[r] = dead[r];
        const int64_t pos = 700 + 997 * r + (r % 3) * 5;   // different lengths, and an incomplete block at the tail
        steps[(size_t) r * k::kStepCount + k::kStepPos] = (int32_t) pos;
        steps[(size_t) r * k::kStepCount + k::kStepNKv] = (int32_t) (pos + 1);
        steps[(size_t) r * k::kStepCount + k::kStepNBid] = (int32_t) ((pos + 1) / s.idx_block);
        steps[(size_t) r * k::kStepCount + k::kStepWidth] = (int32_t) k::qsa_selection_width(pos + 1, s);
    }
    const float* d_q = up(q);
    const int32_t* d_steps = up(steps);
    float *d_rows = nullptr, *d_ref = nullptr;
    const size_t n = (size_t) rows * max_blocks;
    ck(cudaMalloc(&d_rows, n * 4), "malloc");
    ck(cudaMalloc(&d_ref, n * 4), "malloc");
    ck(cudaMemset(d_rows, 0, n * 4), "memset");
    ck(cudaMemset(d_ref, 0, n * 4), "memset");
    k::qsa_block_scores_rows(sr, d_q, d_steps, rows, max_blocks, s, d_rows, nullptr);
    for (int r = 0; r < rows; ++r)
        k::qsa_block_scores(pooled[r], dead[r], d_q + (size_t) r * s.idx_n_head * s.idx_dim, d_steps + (size_t) r * k::kStepCount, 1,
                            max_blocks, s, d_ref + (size_t) r * max_blocks, nullptr);
    ck(cudaDeviceSynchronize(), "sync");
    const std::vector<float> a = down(d_rows, n), b = down(d_ref, n);
    char name[96];
    std::snprintf(name, sizeof name, "qsa_block_scores_rows, %d rows (own keys each)", rows);
    report(name, std::memcmp(a.data(), b.data(), n * 4) == 0);
    // the window selects with one top-k launch over all rows (row-independent) where the per-row path ran one per row
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    int32_t *i_rows = nullptr, *i_ref = nullptr;
    ck(cudaMalloc(&i_rows, (size_t) rows * cap * 4), "malloc");
    ck(cudaMalloc(&i_ref, (size_t) rows * cap * 4), "malloc");
    ck(cudaMemset(i_rows, 0, (size_t) rows * cap * 4), "memset");
    ck(cudaMemset(i_ref, 0, (size_t) rows * cap * 4), "memset");
    k::qsa_block_topk(d_ref, d_steps, rows, max_blocks, cap, s, i_rows, nullptr);
    for (int r = 0; r < rows; ++r)
        k::qsa_block_topk(d_ref + (size_t) r * max_blocks, d_steps + (size_t) r * k::kStepCount, 1, max_blocks, cap, s,
                          i_ref + (size_t) r * cap, nullptr);
    ck(cudaDeviceSynchronize(), "sync");
    const std::vector<int32_t> ta = down(i_rows, (size_t) rows * cap), tb = down(i_ref, (size_t) rows * cap);
    bool tsame = true;
    for (int r = 0; r < rows; ++r) {
        const int64_t w = steps[(size_t) r * k::kStepCount + k::kStepWidth];
        tsame = tsame && std::memcmp(ta.data() + (size_t) r * cap, tb.data() + (size_t) r * cap, (size_t) w * 4) == 0;
    }
    std::snprintf(name, sizeof name, "qsa_block_topk over %d rows == one call per row", rows);
    report(name, tsame);
}

// ---- 2. the decode attention ----
void attn_case(int rows, bool int8) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t HD = s.head_dim, NKV = s.n_head_kv, NH = s.n_head, PS = s.page_size;
    const int64_t ctx_max = 5000, pages = (ctx_max + PS - 1) / PS, prow = pages * NKV * PS;
    const int64_t cap = k::qsa_selection_width(ctx_max, s);
    std::mt19937 rng(23 + rows + int8);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> sc(0.005f, 0.03f);
    k::QsaAttnPoolsRows pr;
    pr.n = rows;
    std::vector<float> q((size_t) rows * NH * HD);
    for (auto& x : q) x = nd(rng) * 2.0f;
    std::vector<int32_t> ids((size_t) rows * cap, 0), steps((size_t) rows * k::kStepCount, 0);
    for (int r = 0; r < rows; ++r) {
        k::QsaAttnPools& pl = pr.p[r];
        if (int8) {
            std::vector<int8_t> kq((size_t) prow * HD), vq((size_t) prow * HD);
            std::vector<uint16_t> ks((size_t) prow * (HD / 64)), vs(ks.size());
            for (auto& x : kq) x = (int8_t) code(rng);
            for (auto& x : vq) x = (int8_t) code(rng);
            for (auto& x : ks) x = f2h(sc(rng));
            for (auto& x : vs) x = f2h(sc(rng));
            pl.k_q = up(kq); pl.v_q = up(vq); pl.k_scale = up(ks); pl.v_scale = up(vs);
        } else {
            std::vector<uint16_t> kh((size_t) prow * HD), vh(kh.size());
            for (auto& x : kh) x = f2h(nd(rng) * 1.5f);
            for (auto& x : vh) x = f2h(nd(rng));
            pl.k_pool = up(kh); pl.v_pool = up(vh);
        }
        std::vector<int32_t> table(pages);
        for (int64_t i = 0; i < pages; ++i) table[i] = (int32_t) i;
        std::shuffle(table.begin(), table.end(), rng);   // each row's pool has its own page order
        pl.page_table = up(table);
        const int64_t nkv = 900 + 641 * r + (r % 2) * 3;
        const int64_t w = k::qsa_selection_width(nkv, s);
        std::vector<int32_t> all((size_t) nkv);
        for (int64_t c = 0; c < nkv; ++c) all[c] = (int32_t) c;
        std::shuffle(all.begin(), all.end(), rng);
        std::sort(all.begin(), all.begin() + w);
        std::copy(all.begin(), all.begin() + w, ids.begin() + (size_t) r * cap);
        steps[(size_t) r * k::kStepCount + k::kStepPos] = (int32_t) (nkv - 1);
        steps[(size_t) r * k::kStepCount + k::kStepNKv] = (int32_t) nkv;
        steps[(size_t) r * k::kStepCount + k::kStepNBid] = (int32_t) (nkv / s.idx_block);
        steps[(size_t) r * k::kStepCount + k::kStepWidth] = (int32_t) w;
    }
    const float* d_q = up(q);
    const int32_t* d_ids = up(ids);
    const int32_t* d_steps = up(steps);
    const int64_t scr = (int64_t) k::qsa_decode_attn_scratch_floats(cap, s);
    float *scratch = nullptr, *d_rows = nullptr, *d_ref = nullptr;
    ck(cudaMalloc(&scratch, (size_t) rows * scr * 4), "malloc");
    const size_t n = (size_t) rows * NH * HD;
    ck(cudaMalloc(&d_rows, n * 4), "malloc");
    ck(cudaMalloc(&d_ref, n * 4), "malloc");
    ck(cudaMemset(d_rows, 0, n * 4), "memset");
    ck(cudaMemset(d_ref, 0, n * 4), "memset");
    k::qsa_decode_attn_batch_rows(d_q, pr, d_ids, d_steps, cap, s, scratch, d_rows, rows, nullptr);
    for (int r = 0; r < rows; ++r)
        k::qsa_decode_attn_batch(d_q + (size_t) r * NH * HD, pr.p[r], d_ids + (size_t) r * cap, d_steps + (size_t) r * k::kStepCount, cap,
                                 s, scratch + (size_t) r * scr, d_ref + (size_t) r * NH * HD, 1, nullptr);
    ck(cudaDeviceSynchronize(), "sync");
    const std::vector<float> a = down(d_rows, n), b = down(d_ref, n);
    char name[96];
    std::snprintf(name, sizeof name, "qsa_decode_attn_batch_rows, %d rows, %s pools (own pools each)", rows, int8 ? "int8" : "fp16");
    report(name, std::memcmp(a.data(), b.data(), n * 4) == 0);
}

// ---- 3. the GDN recurrence ----
void gdn_case(const std::vector<int>& ntok, bool commit) {
    const int S = 128, h_k = 16, h_v = 48;   // the artifact's geometry
    const int qk = S * h_k, C = 2 * qk + S * h_v, vd = S * h_v;
    const float eps = 1e-6f;
    int T = 0;
    for (int n : ntok) T += n;
    const int G = (int) ntok.size();
    std::mt19937 rng(31 + T * 7 + G + commit);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> h((size_t) T * C), gate((size_t) T * h_v), beta((size_t) T * h_v), z((size_t) T * vd), gamma(S);
    for (auto& v : h) v = nd(rng) * 0.1f;
    for (auto& v : gate) v = -(0.5f + 3.0f * (float) (rng() % 100) / 100.0f);
    for (auto& v : beta) v = (float) (rng() % 100) / 100.0f;
    for (auto& v : z) v = nd(rng);
    for (auto& v : gamma) v = 1.0f + 0.1f * nd(rng);
    const size_t st_n = (size_t) S * h_v * S;
    std::vector<std::vector<float>> st0(G);
    std::vector<float*> st_g(G), st_r(G);
    std::vector<int32_t*> keep(G);
    k::GdnGroups gg;
    gg.n = G;
    int first = 0;
    for (int i = 0; i < G; ++i) {
        st0[i].resize(st_n);
        for (auto& v : st0[i]) v = nd(rng) * 0.1f;
        st_g[i] = up(st0[i]);
        st_r[i] = up(st0[i]);
        const int32_t kp = commit ? ntok[i] : 0;   // a commit group keeps all of its rows here
        keep[i] = up(std::vector<int32_t>{kp});
        gg.state[i] = st_g[i];
        gg.keep[i] = commit ? keep[i] : nullptr;
        gg.first[i] = first;
        gg.ntok[i] = ntok[i];
        first += ntok[i];
    }
    const float* d_h = up(h);
    const float* d_gate = up(gate);
    const float* d_beta = up(beta);
    const float* d_z = up(z);
    const float* d_gm = up(gamma);
    float *d_y = nullptr, *d_yr = nullptr;
    ck(cudaMalloc(&d_y, (size_t) T * vd * 4), "malloc");
    ck(cudaMalloc(&d_yr, (size_t) T * vd * 4), "malloc");
    ck(cudaMemset(d_y, 0, (size_t) T * vd * 4), "memset");
    ck(cudaMemset(d_yr, 0, (size_t) T * vd * 4), "memset");
    k::gdn_step_norm_groups(gg, d_h, C, d_gate, d_beta, d_z, d_gm, eps, d_y, h_k, h_v, commit, nullptr);
    for (int i = 0; i < G; ++i) {
        const int f = gg.first[i];
        // the engine's per-slot call: a window group runs its rows with all outputs; a commit group (1 row) keeps them
        k::gdn_step_norm_multi(st_r[i], d_h + (size_t) f * C, C, d_gate + (size_t) f * h_v, d_beta + (size_t) f * h_v,
                               d_z + (size_t) f * vd, d_gm, eps, d_yr + (size_t) f * vd, h_k, h_v, ntok[i],
                               commit ? keep[i] : nullptr, nullptr, 0);
    }
    ck(cudaDeviceSynchronize(), "sync");
    bool same = true;
    for (int i = 0; i < G; ++i) {
        const std::vector<float> a = down(st_g[i], st_n), b = down(st_r[i], st_n);
        same = same && std::memcmp(a.data(), b.data(), st_n * 4) == 0;
        if (!commit) same = same && std::memcmp(a.data(), st0[i].data(), st_n * 4) == 0;   // a window leaves the state alone
    }
    if (!commit) {
        const std::vector<float> a = down(d_y, (size_t) T * vd), b = down(d_yr, (size_t) T * vd);
        same = same && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
    }
    char name[96], rows[64] = "";
    for (int i = 0; i < G && i < 8; ++i) std::snprintf(rows + std::strlen(rows), sizeof rows - std::strlen(rows), "%s%d", i ? "," : "", ntok[i]);
    std::snprintf(name, sizeof name, "gdn_step_norm_groups %s, groups of %s rows", commit ? "commit" : "window", rows);
    report(name, same);
}
}  // namespace

int main() {
    std::printf("batch rows parity (#2234 kernels against the per-row launches)\n");
    for (int r : {1, 2, 4, 8}) scores_case(r);
    for (int r : {1, 3, 8}) {
        attn_case(r, true);
        attn_case(r, false);
    }
    gdn_case({1}, false);
    gdn_case({1, 1, 1, 1}, false);
    gdn_case({2, 2, 2, 2}, false);
    gdn_case({1, 3, 2, 1, 1}, false);
    gdn_case({1, 1, 1, 1, 1, 1, 1, 1}, false);
    gdn_case({1}, true);
    gdn_case({1, 1, 1, 1}, true);
    gdn_case({1, 1, 1, 1, 1, 1, 1, 1}, true);
    std::printf("%s\n", bad ? "batch_rows_parity: FAILED" : "batch_rows_parity: ok");
    return bad ? 1 : 0;
}
