// src/kernels/qsa_prompt_attn_parity.cpp - perf-review D-1: the tensor-core prompt attention (qsa_prompt_attn.hpp)
// against the FP32 kernel it replaces (`qsa_decode_attn_batch`) and an FP64 host reference (GPU, synthetic, no model).
//
// Int8, FP16 and Q4_0 pools with random codes, scales and queries; selections shaped like the prompt path's (the 2,051
// widest, a recent window plus older cells that drift slowly from one query to the next, so neighbours share most
// of them as they do in a real prompt; short contexts take every cell). Checks:
//   1. against FP64, the new kernel's error is no larger than a small multiple of the old kernel's (both FP32 math);
//   2. the new and old outputs agree to a relative 1e-4 of the output scale;
// then times both over a prompt chunk (the old one in batches of 32, as prefill.cpp calls it).
// HIP builds (S6): the same checks for the RDNA4 matrix-core kernel (STRATA_HIP_WMMA), skipped (77) off gfx12.
// Usage: qsa_prompt_attn_parity [context=32768] [queries=2048] [reps=5]
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/kv_q4.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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
float h2f(uint16_t b) { __half h; *reinterpret_cast<uint16_t*>(&h) = b; return __half2float(h); }
uint16_t f2h(float f) { __half h = __float2half(f); return *reinterpret_cast<uint16_t*>(&h); }

int run(int fmt, int64_t ctx, int64_t nq, int reps) {   // fmt 1 int8, 0 fp16, 2 q4_0
#if !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
    if (fmt == 2) {   // mode 4 (Q4_0 KV) runs on sm_80 and newer only: below that the dispatcher keeps the old kernel
        int dev = 0, major = 0;
        ck(cudaGetDevice(&dev), "device");
        ck(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev), "cc");
        if (major < 8) {
            std::printf("SKIP q4_0 ctx %lld: the tensor-core kernel takes Q4_0 KV on sm_80+ only (this device: sm_%d)\n",
                        (long long) ctx, major);
            return 0;
        }
    }
#endif
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t HD = s.head_dim, NKV = s.n_head_kv, NH = s.n_head, PS = s.page_size;
    const int64_t pages = (ctx + PS - 1) / PS, rows = pages * NKV * PS;
    std::mt19937 rng(1234 + fmt);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> sc(0.005f, 0.03f);
    // pools; page table a shuffled permutation (the readers must follow it)
    std::vector<int8_t> kq, vq;
    std::vector<uint16_t> ks, vs, kh, vh;
    std::vector<uint8_t> k4, v4;   // q4_0: 8 blocks of {fp16 d, 16 code bytes} per row (kv_q4.hpp)
    constexpr int64_t B4 = 18, ROW4 = 8 * B4;
    if (fmt == 2) {
        std::uniform_int_distribution<int> byte(0, 255);
        // the scales as a q4_0 pool holds them: signed (ggml's d = max / -8) and spread over two decades, so a chunk's
        // cells mix signs and magnitudes (a positive-only bound on them overflowed FP16 on real K/V)
        std::uniform_real_distribution<float> lg(std::log(0.01f), std::log(1.0f));
        auto sc4 = [&](std::mt19937& r) { return std::exp(lg(r)) * (byte(r) & 1 ? -1.0f : 1.0f); };
        k4.resize(rows * ROW4); v4.resize(rows * ROW4);
        for (auto* pool : {&k4, &v4})
            for (int64_t b = 0; b < rows * 8; ++b) {
                const uint16_t d = f2h(sc4(rng));
                std::memcpy(pool->data() + b * B4, &d, 2);
                for (int j = 0; j < 16; ++j) (*pool)[b * B4 + 2 + j] = (uint8_t) byte(rng);
            }
    } else if (fmt == 1) {
        kq.resize(rows * HD); vq.resize(rows * HD); ks.resize(rows * 4); vs.resize(rows * 4);
        for (auto& x : kq) x = (int8_t) code(rng);
        for (auto& x : vq) x = (int8_t) code(rng);
        for (auto& x : ks) x = f2h(sc(rng));
        for (auto& x : vs) x = f2h(sc(rng));
    } else {
        kh.resize(rows * HD); vh.resize(rows * HD);
        for (auto& x : kh) x = f2h(nd(rng) * 1.5f);
        for (auto& x : vh) x = f2h(nd(rng));
    }
    std::vector<int32_t> table(pages);
    for (int64_t i = 0; i < pages; ++i) table[i] = (int32_t) i;
    std::shuffle(table.begin(), table.end(), rng);
    // queries at positions ctx - nq .. ctx - 1
    const int64_t cap = k::qsa_selection_width(ctx, s);
    std::vector<int32_t> ids((size_t) (nq * cap), 0), steps((size_t) (nq * k::kStepCount), 0);
    std::vector<float> q((size_t) (nq * NH * HD));
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
        if ((int64_t) old_cells.size() != older) {   // the older cells: a set that drifts ~3% per query
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
    // QA_SAME=1 (S23 probe): every query reads the same first-w cells - the best case of any K/V sharing between queries
    // (all loads hit the caches); QA_SAME=2: query i reads a window of w cells at i / 4 * 64 (neighbours share ~90%)
    if (const char* sm = std::getenv("QA_SAME")) {
        const int mode = std::atoi(sm);
        for (int64_t i = 0; i < nq; ++i) {
            const int64_t w = steps[i * k::kStepCount + k::kStepWidth], nkv = ctx - nq + i + 1;
            if (w == nkv) continue;
            if (mode == 3 || mode == 4) {   // 3: w random cells per query (no overlap beyond chance); 4: a 4-query group shares
                // one random set of w - 512 cells (the union of 4 queries = 1x), the last 512 cells recent
                std::mt19937 r3((unsigned) (mode == 3 ? i : i / 4) * 2654435761u + 7);
                std::vector<int32_t> all((size_t) (nkv - 512));
                for (int64_t c = 0; c < nkv - 512; ++c) all[c] = (int32_t) c;
                std::shuffle(all.begin(), all.end(), r3);
                std::vector<int32_t> v(all.begin(), all.begin() + (w - 512));
                for (int64_t c = nkv - 512; c < nkv; ++c) v.push_back((int32_t) c);
                std::sort(v.begin(), v.end());
                std::copy(v.begin(), v.end(), ids.data() + i * cap);
                continue;
            }
            const int64_t base = mode == 1 ? 0 : std::min<int64_t>(i / 4 * 64, nkv - w);
            for (int64_t c = 0; c < w; ++c) ids[i * cap + c] = (int32_t) (base + c);
        }
    }
    // QA_DUMP=<file> QA_LAYER=n (S23 probe): the engine's own selections (STRATA_QSA_DUMP records {layer, pos0, T, cap} +
    // T*cap cells): the last record of that layer, whose T must be nq and pos0 + T ctx
    if (const char* df = std::getenv("QA_DUMP")) {
        const int want = std::getenv("QA_LAYER") ? std::atoi(std::getenv("QA_LAYER")) : 3;
        std::FILE* f = std::fopen(df, "rb");
        if (!f) { std::fprintf(stderr, "QA_DUMP: cannot open\n"); std::exit(2); }
        int32_t h[4];
        long at = -1;
        while (std::fread(h, 4, 4, f) == 4) {
            const long data = std::ftell(f);
            if (h[0] == want && h[2] == nq && h[1] + h[2] == ctx && h[3] == cap) at = data;
            std::fseek(f, (long) h[2] * h[3] * 4, SEEK_CUR);
        }
        if (at < 0) { std::fprintf(stderr, "QA_DUMP: no record for layer %d, T %lld, ctx %lld\n", want, (long long) nq, (long long) ctx); std::exit(2); }
        std::fseek(f, at, SEEK_SET);
        if (std::fread(ids.data(), 4, ids.size(), f) != ids.size()) std::exit(2);
        std::fclose(f);
        for (int grp : {2, 4, 8, 16}) {   // the union of `grp` neighbouring queries' cells against grp x the width
            double su = 0, sw = 0;
            for (int64_t i0 = 0; i0 + grp <= nq; i0 += grp * 16) {
                std::vector<int32_t> u;
                for (int g2 = 0; g2 < grp; ++g2) {
                    const int64_t w = steps[(i0 + g2) * k::kStepCount + k::kStepWidth];
                    u.insert(u.end(), ids.begin() + (i0 + g2) * cap, ids.begin() + (i0 + g2) * cap + w);
                    sw += (double) w;
                }
                std::sort(u.begin(), u.end());
                su += (double) (std::unique(u.begin(), u.end()) - u.begin());
            }
            std::printf("QA_DUMP overlap: %d-query groups read %.2fx the cells of one query (%.1f%% of %d x)\n", grp,
                        su * grp / sw, 100.0 * su / sw, grp);
        }
    }
    k::QsaAttnPools pl;
    if (fmt == 2) { pl.k_q4 = up(k4); pl.v_q4 = up(v4); }
    else if (fmt == 1) { pl.k_q = up(kq); pl.v_q = up(vq); pl.k_scale = up(ks); pl.v_scale = up(vs); }
    else { pl.k_pool = up(kh); pl.v_pool = up(vh); }
    // a q4_0 value: block d / 32 of the row, element j = d % 32 in the low nibble of byte j (j < 16), else the high
    // nibble of byte j - 16, minus 8, times the block's scale
    auto q4v = [&](const std::vector<uint8_t>& pool, int64_t row, int64_t d) {
        const uint8_t* blk = pool.data() + row * ROW4 + (d / 32) * B4;
        uint16_t sc;
        std::memcpy(&sc, blk, 2);
        const int j = (int) (d % 32);
        const int code = j < 16 ? (blk[2 + j] & 0x0F) : (blk[2 + j - 16] >> 4);
        return (double) (code - 8) * h2f(sc);
    };
    pl.page_table = up(table);
    const int32_t* d_ids = up(ids);
    const int32_t* d_steps = up(steps);
    const float* d_q = up(q);
    float *d_old = nullptr, *d_new = nullptr, *scratch = nullptr;
    const int64_t batch = 32;
    ck(cudaMalloc(&d_old, nq * NH * HD * 4), "malloc");
    ck(cudaMalloc(&d_new, nq * NH * HD * 4), "malloc");
    ck(cudaMalloc(&scratch, batch * k::qsa_decode_attn_scratch_floats(cap, s) * 4), "malloc");
    auto old_run = [&]() {
        for (int64_t t0 = 0; t0 < nq; t0 += batch)
            k::qsa_decode_attn_batch(d_q + t0 * NH * HD, pl, d_ids + t0 * cap, d_steps + t0 * k::kStepCount, cap, s,
                                     scratch, d_old + t0 * NH * HD, std::min(batch, nq - t0), nullptr);
    };
    // int8 KV: v2 here, and the opt-in INT8-MMA kernel (STRATA_PROMPT_ATTN_IMMA) checked and timed below beside it
    if (fmt == 1) k::qsa_prompt_attn_set_imma(0);
    auto new_run = [&]() {
        if (!k::qsa_prompt_attn_batch(d_q, pl, d_ids, d_steps, cap, s, d_new, nq, nullptr)) {
            std::fprintf(stderr, "qsa_prompt_attn_batch refused the pools\n");
            std::exit(2);
        }
    };
    float* d_im = nullptr;
    ck(cudaMalloc(&d_im, nq * NH * HD * 4), "malloc");
    auto imma_run = [&]() {
        k::qsa_prompt_attn_set_imma(1);
        const bool ok = k::qsa_prompt_attn_batch(d_q, pl, d_ids, d_steps, cap, s, d_im, nq, nullptr);
        k::qsa_prompt_attn_set_imma(0);
        if (!ok) { std::fprintf(stderr, "qsa_prompt_attn_batch refused the pools (imma)\n"); std::exit(2); }
    };
    old_run();
    new_run();
    if (fmt == 1) imma_run();
    ck(cudaDeviceSynchronize(), "run");
    std::vector<float> o((size_t) (nq * NH * HD)), nw(o.size()), im(fmt == 1 ? o.size() : 0);
    if (fmt == 1) ck(cudaMemcpy(im.data(), d_im, im.size() * 4, cudaMemcpyDeviceToHost), "down");
    ck(cudaMemcpy(o.data(), d_old, o.size() * 4, cudaMemcpyDeviceToHost), "down");
    ck(cudaMemcpy(nw.data(), d_new, nw.size() * 4, cudaMemcpyDeviceToHost), "down");
    {   // a hash of the new kernel's output bits (bitwise A/B of two builds / switches)
        uint64_t hh = 1469598103934665603ull;
        for (float x : nw) { uint32_t u; std::memcpy(&u, &x, 4); hh = (hh ^ u) * 1099511628211ull; }
        std::printf("NEWHASH %016llx\n", (unsigned long long) hh);
    }
    // 1. FP64 reference on a sample of queries
    double err_old = 0, err_new = 0, err_im = 0, ref_scale = 0;
    for (int64_t i = 0; i < nq; i += std::max<int64_t>(1, nq / 16)) {
        const int64_t w = steps[i * k::kStepCount + k::kStepWidth];
        const int32_t* sel = ids.data() + i * cap;
        for (int64_t h = 0; h < NH; ++h) {
            const int64_t kvh = h / (NH / NKV);
            std::vector<double> sco((size_t) w);
            double mx = -1e300;
            for (int64_t c = 0; c < w; ++c) {
                const int64_t cell = sel[c], row = ((int64_t) table[cell / PS] * NKV + kvh) * PS + cell % PS;
                double a = 0;
                for (int64_t d = 0; d < HD; ++d) {
                    const double kv = fmt == 2 ? q4v(k4, row, d)
                                      : fmt == 1 ? (double) kq[row * HD + d] * h2f(ks[row * 4 + d / 64]) : h2f(kh[row * HD + d]);
                    a += (double) q[(i * NH + h) * HD + d] * kv;
                }
                sco[c] = a / 16.0;
                mx = std::max(mx, sco[c]);
            }
            double l = 0;
            for (auto& x : sco) { x = std::exp(x - mx); l += x; }
            for (int64_t d = 0; d < HD; ++d) {
                double a = 0;
                for (int64_t c = 0; c < w; ++c) {
                    const int64_t cell = sel[c], row = ((int64_t) table[cell / PS] * NKV + kvh) * PS + cell % PS;
                    const double vv = fmt == 2 ? q4v(v4, row, d)
                                      : fmt == 1 ? (double) vq[row * HD + d] * h2f(vs[row * 4 + d / 64]) : h2f(vh[row * HD + d]);
                    a += sco[c] * vv;
                }
                const double r = a / l;
                const size_t at = (size_t) ((i * NH + h) * HD + d);
                ref_scale = std::max(ref_scale, std::fabs(r));
                err_old = std::max(err_old, std::fabs(o[at] - r));
                err_new = std::max(err_new, std::fabs(nw[at] - r));
                if (fmt == 1) err_im = std::max(err_im, std::fabs(im[at] - r));
            }
        }
    }
    // 2. new vs old everywhere
    double diff = 0, scale = 0;
    for (size_t i = 0; i < o.size(); ++i) {
        diff = std::max(diff, (double) std::fabs(o[i] - nw[i]));
        scale = std::max(scale, (double) std::fabs(o[i]));
    }
    // 3. speed
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    float ms_old = 0, ms_new = 0;
    cudaEventRecord(e0);
    for (int r = 0; r < reps; ++r) old_run();
    cudaEventRecord(e1);
    ck(cudaEventSynchronize(e1), "time");
    cudaEventElapsedTime(&ms_old, e0, e1);
    cudaEventRecord(e0);
    for (int r = 0; r < reps; ++r) new_run();
    cudaEventRecord(e1);
    ck(cudaEventSynchronize(e1), "time");
    cudaEventElapsedTime(&ms_new, e0, e1);
    bool ok_im = true;
    if (fmt == 1) {   // the INT8-MMA kernel: the same bounds as v2's, against FP64 and against v2
        double d_iv = 0;
        for (size_t i = 0; i < o.size(); ++i) d_iv = std::max(d_iv, (double) std::fabs(im[i] - nw[i]));
        float ms_im = 0, ms_v2 = 0;
        for (int round = 0; round < 2; ++round) {   // alternating, v2 then imma, twice; the second round counts
            cudaEventRecord(e0);
            for (int r = 0; r < reps; ++r) new_run();
            cudaEventRecord(e1);
            ck(cudaEventSynchronize(e1), "time");
            cudaEventElapsedTime(&ms_v2, e0, e1);
            cudaEventRecord(e0);
            for (int r = 0; r < reps; ++r) imma_run();
            cudaEventRecord(e1);
            ck(cudaEventSynchronize(e1), "time");
            cudaEventElapsedTime(&ms_im, e0, e1);
        }
        ok_im = err_im <= std::max(4.0 * err_old, 1e-6 * ref_scale) && d_iv <= 1e-4 * scale;
        std::printf("%s int8 IMMA ctx %lld, %lld queries: vs FP64 %.3g (v2 %.3g, FP32 %.3g); IMMA vs v2 %.3g (%.2g of scale); "
                    "v2 %.3f -> IMMA %.3f ms per chunk (%.2fx)\n", ok_im ? "PASS" : "FAIL", (long long) ctx, (long long) nq,
                    err_im, err_new, err_old, d_iv, d_iv / scale, ms_v2 / reps, ms_im / reps, ms_v2 / ms_im);
    }
    k::qsa_prompt_attn_set_imma(-1);
    cudaFree(d_im);
    const bool ok1 = err_new <= std::max(4.0 * err_old, 1e-6 * ref_scale) && ok_im;
    const bool ok2 = diff <= 1e-4 * scale;
    uint64_t sum = 1469598103934665603ull;   // the new kernel's output bits: two builds of it compare by this
    for (float f : nw) {
        uint32_t b;
        std::memcpy(&b, &f, 4);
        sum = (sum ^ b) * 1099511628211ull;
    }
    std::printf("%s %s ctx %lld, %lld queries: vs FP64 old %.3g new %.3g (output scale %.3g); new vs old %.3g (%.2g of "
                "scale); %.3f -> %.3f ms per chunk (%.2fx); new output checksum %016llx\n",
                ok1 && ok2 ? "PASS" : "FAIL", fmt == 2 ? "q4_0" : fmt == 1 ? "int8" : "fp16", (long long) ctx, (long long) nq, err_old,
                err_new, ref_scale, diff, diff / scale, ms_old / reps, ms_new / reps, ms_old / ms_new, (unsigned long long) sum);
    cudaFree((void*) d_ids); cudaFree((void*) d_steps); cudaFree((void*) d_q); cudaFree(d_old); cudaFree(d_new);
    cudaFree(scratch);
    cudaFree((void*) pl.k_q); cudaFree((void*) pl.v_q); cudaFree((void*) pl.k_scale); cudaFree((void*) pl.v_scale);
    cudaFree((void*) pl.k_pool); cudaFree((void*) pl.v_pool); cudaFree((void*) pl.page_table);
    cudaFree((void*) pl.k_q4); cudaFree((void*) pl.v_q4);
    return ok1 && ok2 ? 0 : 1;
}
}  // namespace

int main(int argc, char** argv) {
#if defined(__HIP_PLATFORM_AMD__)
    // S6: on AMD the kernel under test is the RDNA4 matrix-core one (opt-in in the engine); other cards skip
    {
        int dev = 0;
        hipDeviceProp_t prop{};
        if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&prop, dev) != hipSuccess) return 2;
        if (std::strncmp(prop.gcnArchName, "gfx12", 5) != 0 && std::strncmp(prop.gcnArchName, "gfx11", 5) != 0) {
            std::printf("SKIP: %s is not gfx11 / gfx12 (the matrix-core prompt attention)\n", prop.gcnArchName);
            return 77;
        }
#if defined(_WIN32)
        _putenv_s("STRATA_HIP_WMMA", "1");
#else
        setenv("STRATA_HIP_WMMA", "1", 1);
#endif
    }
#endif
    const int64_t ctx = argc > 1 ? std::atoll(argv[1]) : 32768;
    const int64_t nq = argc > 2 ? std::atoll(argv[2]) : 2048;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 5;
    int fails = 0;
    fails += run(1, ctx, nq, reps);
#if !defined(__HIP_PLATFORM_AMD__)
    fails += run(0, ctx, nq, reps);   // FP16 KV: the RDNA4 kernel takes int8 KV only
    fails += run(2, ctx, nq, reps);   // Q4_0 KV (mode 4)
    fails += run(2, 1500, std::min<int64_t>(nq, 1500), reps);
#endif
    fails += run(1, 1500, std::min<int64_t>(nq, 1500), reps);   // short context: the selection is every cell
    fails += run(1, 2100, std::min<int64_t>(nq, 256), reps);    // the identity-to-sparse edge
    std::printf("FAILURES: %d\n", fails);
    return fails;
}
