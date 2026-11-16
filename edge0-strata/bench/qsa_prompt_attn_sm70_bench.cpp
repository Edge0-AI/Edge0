// bench/qsa_prompt_attn_sm70_bench.cpp - the Volta (sm_70) prompt attention against the FP32 kernel it replaces.
//
// Same synthetic pools and prompt-shaped selections as src/kernels/qsa_prompt_attn_parity.cpp (GPU, no model), but
// timing is the point: for each KV format (int8, fp16) and a sweep of query counts it times
// `qsa_decode_attn_batch` (the FP32 fallback, in batches of 32 as prefill.cpp calls it) against
// `qsa_prompt_attn_batch` with CUDA events, and reports the output difference.  Run it on one card:
//   flock /tmp/opencode/gpu.lock -c 'CUDA_VISIBLE_DEVICES=0 ./qsa_prompt_attn_sm70_bench [ctx=32768] [nq=2048] [reps=10]'
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
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
uint16_t f2h(float f) { __half h = __float2half(f); return *reinterpret_cast<uint16_t*>(&h); }

// one synthetic prompt: the widest selection the shape has (2,051 cells at 32K), a recent window plus older cells
// that drift slowly from query to query, as qsa_prompt_attn_parity builds them.  fmt 1 int8 KV, 0 fp16 KV,
// 2 hybrid K8V4 (int8 K, q4_0 V - the Q2_0 engine's pools)
void build(int fmt, int64_t ctx, int64_t nq, std::vector<int8_t>& kq, std::vector<int8_t>& vq, std::vector<uint16_t>& ks,
           std::vector<uint16_t>& vs, std::vector<uint16_t>& kh, std::vector<uint16_t>& vh, std::vector<uint8_t>& vq4,
           std::vector<int32_t>& table, std::vector<int32_t>& ids, std::vector<int32_t>& steps, std::vector<float>& q,
           int64_t& cap) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t HD = s.head_dim, NKV = s.n_head_kv, NH = s.n_head, PS = s.page_size;
    const int64_t pages = (ctx + PS - 1) / PS, rows = pages * NKV * PS;
    std::mt19937 rng(1234 + fmt);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> sc(0.005f, 0.03f);
    if (fmt == 1 || fmt == 2) {
        kq.resize(rows * HD); ks.resize(rows * 4);
        for (auto& x : kq) x = (int8_t) code(rng);
        for (auto& x : ks) x = f2h(sc(rng));
        if (fmt == 1) {
            vq.resize(rows * HD); vs.resize(rows * 4);
            for (auto& x : vq) x = (int8_t) code(rng);
            for (auto& x : vs) x = f2h(sc(rng));
        } else {
            vq4.resize((size_t) (rows * (HD / 32) * 18));   // q4_0 blocks: a finite fp16 scale + random nibbles
            for (size_t i = 0; i < vq4.size(); i += 18) {
                const uint16_t d = f2h(sc(rng));
                vq4[i] = (uint8_t) d; vq4[i + 1] = (uint8_t) (d >> 8);
                for (int j = 2; j < 18; ++j) vq4[i + j] = (uint8_t) rng();
            }
        }
    } else {
        kh.resize(rows * HD); vh.resize(rows * HD);
        for (auto& x : kh) x = f2h(nd(rng) * 1.5f);
        for (auto& x : vh) x = f2h(nd(rng));
    }
    table.resize(pages);
    for (int64_t i = 0; i < pages; ++i) table[i] = (int32_t) i;
    std::shuffle(table.begin(), table.end(), rng);
    cap = k::qsa_selection_width(ctx, s);
    ids.assign((size_t) (nq * cap), 0);
    steps.assign((size_t) (nq * k::kStepCount), 0);
    q.assign((size_t) (nq * NH * HD), 0.f);
    for (auto& x : q) x = nd(rng) * 2.0f;
    std::vector<int32_t> old_cells;
    for (int64_t i = 0; i < nq; ++i) {
        const int64_t pos = ctx - nq + i, nkv = pos + 1;
        const int64_t w = k::qsa_selection_width(nkv, s);
        int32_t* sel = ids.data() + i * cap;
        steps[i * k::kStepCount + k::kStepWidth] = (int32_t) w;
        if (w == nkv) {
            for (int64_t c = 0; c < w; ++c) sel[c] = (int32_t) c;
            continue;
        }
        const int64_t recent = 512, older = w - recent;
        if ((int64_t) old_cells.size() != older) {
            std::vector<int32_t> all((size_t) (nkv - recent));
            for (int64_t c = 0; c < nkv - recent; ++c) all[c] = (int32_t) c;
            std::shuffle(all.begin(), all.end(), rng);
            old_cells.assign(all.begin(), all.begin() + older);
        } else {
            std::uniform_int_distribution<int64_t> pick(0, older - 1), any(0, nkv - recent - 1);
            for (int r = 0; r < older / 32; ++r) {
                const int32_t c = (int32_t) any(rng);
                if (std::find(old_cells.begin(), old_cells.end(), c) == old_cells.end()) old_cells[pick(rng)] = c;
            }
        }
        std::vector<int32_t> v(old_cells);
        for (int64_t c = nkv - recent; c < nkv; ++c) v.push_back((int32_t) c);
        std::sort(v.begin(), v.end());
        std::copy(v.begin(), v.end(), sel);
    }
}

float time_ms(const std::function<void()>& f, int reps) {
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "event");
    ck(cudaEventCreate(&e1), "event");
    f();
    ck(cudaDeviceSynchronize(), "warm");
    ck(cudaEventRecord(e0), "event");
    for (int r = 0; r < reps; ++r) f();
    ck(cudaEventRecord(e1), "event");
    ck(cudaEventSynchronize(e1), "time");
    float ms = 0;
    ck(cudaEventElapsedTime(&ms, e0, e1), "time");
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ms / reps;
}

void run(int fmt, int64_t ctx, int64_t nq, int reps) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t HD = s.head_dim, NH = s.n_head;
    std::vector<int8_t> kq, vq;
    std::vector<uint16_t> ks, vs, kh, vh;
    std::vector<uint8_t> vq4;
    std::vector<int32_t> table, ids, steps;
    std::vector<float> q;
    int64_t cap = 0;
    build(fmt, ctx, nq, kq, vq, ks, vs, kh, vh, vq4, table, ids, steps, q, cap);
    k::QsaAttnPools pl;
    if (fmt == 1) { pl.k_q = up(kq); pl.v_q = up(vq); pl.k_scale = up(ks); pl.v_scale = up(vs); }
    else if (fmt == 2) { pl.k_q = up(kq); pl.k_scale = up(ks); pl.v_q4 = up(vq4); }
    else { pl.k_pool = up(kh); pl.v_pool = up(vh); }
    pl.page_table = up(table);
    const int32_t* d_ids = up(ids);
    const int32_t* d_steps = up(steps);
    const float* d_q = up(q);
    float *d_old = nullptr, *d_new = nullptr, *scratch = nullptr;
    const int64_t batch = 32;
    ck(cudaMalloc(&d_old, nq * NH * HD * 4), "malloc");
    ck(cudaMalloc(&d_new, nq * NH * HD * 4), "malloc");
    ck(cudaMalloc(&scratch, batch * k::qsa_decode_attn_scratch_floats(cap, s) * 4), "malloc");
    std::printf("# %s KV, ctx %lld, selection %lld cells, %lld queries of %lld heads x %lld dims\n",
                fmt == 1 ? "int8" : fmt == 2 ? "int8+q4 V (K8V4)" : "fp16", (long long) ctx, (long long) cap,
                (long long) nq, (long long) NH, (long long) HD);
    const int64_t counts[] = {16, 64, 256, 1024, 2048, 4096};
    for (int64_t nqi : counts) {
        if (nqi > nq) continue;
        auto old_run = [&]() {
            for (int64_t t0 = 0; t0 < nqi; t0 += batch)
                k::qsa_decode_attn_batch(d_q + t0 * NH * HD, pl, d_ids + t0 * cap, d_steps + t0 * k::kStepCount, cap,
                                         s, scratch, d_old + t0 * NH * HD, std::min(batch, nqi - t0), nullptr);
        };
        auto new_run = [&]() {
            if (!k::qsa_prompt_attn_batch(d_q, pl, d_ids, d_steps, cap, s, d_new, nqi, nullptr)) {
                std::fprintf(stderr, "qsa_prompt_attn_batch refused the pools\n");
                std::exit(2);
            }
        };
        old_run(); new_run();
        ck(cudaDeviceSynchronize(), "run");
        std::vector<float> o((size_t) (nqi * NH * HD)), nw(o.size());
        ck(cudaMemcpy(o.data(), d_old, o.size() * 4, cudaMemcpyDeviceToHost), "down");
        ck(cudaMemcpy(nw.data(), d_new, nw.size() * 4, cudaMemcpyDeviceToHost), "down");
        double diff = 0, scale = 0;
        for (size_t i = 0; i < o.size(); ++i) {
            diff = std::max(diff, (double) std::fabs(o[i] - nw[i]));
            scale = std::max(scale, (double) std::fabs(o[i]));
        }
        const float ms_old = time_ms(old_run, reps), ms_new = time_ms(new_run, reps);
        std::printf("q %5lld: fp32 %8.3f ms   volta %8.3f ms   %5.2fx   max|old-new| %.3g (%.2g of scale)\n",
                    (long long) nqi, ms_old, ms_new, ms_old / ms_new, diff, scale > 0 ? diff / scale : 0.0);
    }
    cudaFree((void*) d_ids); cudaFree((void*) d_steps); cudaFree((void*) d_q); cudaFree(d_old); cudaFree(d_new);
    cudaFree(scratch);
    cudaFree((void*) pl.k_q); cudaFree((void*) pl.v_q); cudaFree((void*) pl.k_scale); cudaFree((void*) pl.v_scale);
    cudaFree((void*) pl.v_q4);
    cudaFree((void*) pl.k_pool); cudaFree((void*) pl.v_pool); cudaFree((void*) pl.page_table);
}
}  // namespace

int main(int argc, char** argv) {
    const int64_t ctx = argc > 1 ? std::atoll(argv[1]) : 32768;
    const int64_t nq = argc > 2 ? std::atoll(argv[2]) : 2048;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 10;
    run(1, ctx, nq, reps);
    run(2, ctx, nq, reps);
    run(0, ctx, nq, reps);
    return 0;
}
