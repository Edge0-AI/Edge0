// prefill_fused_moe_test - #136: the fused int8 expert kernels of the prompt path (moe_fused.hpp, STRATA_PF_FUSED=1)
// on random Strata Q2_0 expert blobs and random routing, against
//   - a double-precision reference: the dequantized weights times the FP32 activations (gate/up, SwiGLU, down), and
//   - the MMQ path in prefill.cpp's sequence (gather_strata_q2 into 16-expert groups, q8_1 of the slots' activations,
//     gate/up, SwiGLU, q8_1 of H, down).
// Both round the activations and H to int8 per 32 values, differently, so this is not a bitwise test: the fused
// path's error against the reference has to be comparable to MMQ's own (at most 1.5x its RMS and 2x its worst row).
// Part 1 (reference): 256 tokens over 64 experts, skewed routing - experts with 0 rows, with one, with several 64-row
// tiles - an all-zero token, per-expert blobs at unrelated addresses, three launches of different expert ranges.
// Part 2 (timing): one layer at a real chunk (2048 tokens, 512 experts, top 10) on both paths, with their agreement.
// Exit 77 without a CUDA device of sm_80 or newer.
#include "strata/prefill/moe_fused.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
namespace mmq = strata::prefill::mmq;
namespace fused = strata::prefill::fused;

constexpr int N = 2560, FF = 640, K = 10, GROUP = 16, Q2 = 42;   // GGML_TYPE_Q2_0
constexpr size_t BLOB = 1382400, O_D_CODES = (size_t) 1280 * 640, O_GU_SC = O_D_CODES + (size_t) 2560 * 160,
                 O_D_SC = O_GU_SC + (size_t) 1280 * 40 * 2;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n) { ck(cudaMalloc(&p, n), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    template <typename T> T* as() const { return (T*) p; }
};

// a Strata Q2_0 blob: random codes, scales in [0.004, 0.03] (a 2-bit expert's range)
std::vector<uint8_t> make_blob(std::mt19937& rng) {
    std::vector<uint8_t> b(BLOB);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> sc(0.004f, 0.03f);
    for (size_t i = 0; i < O_GU_SC; ++i) b[i] = (uint8_t) byte(rng);
    for (size_t i = O_GU_SC; i < BLOB; i += 2) {
        const ggml_fp16_t h = ggml_fp32_to_fp16(sc(rng));
        std::memcpy(&b[i], &h, 2);
    }
    return b;
}

// the blob's weights as floats: gate/up [1280][2560] (interleaved rows), down [2560][640]
void dequant(const std::vector<uint8_t>& b, std::vector<float>& gu, std::vector<float>& dn) {
    auto row = [&](size_t codes, size_t ld, size_t scales, int nb, int r, float* out) {
        for (int k = 0; k < nb * 64; ++k) {
            ggml_fp16_t h;
            std::memcpy(&h, &b[scales + ((size_t) r * nb + k / 64) * 2], 2);
            const int q = (b[codes + (size_t) r * ld + k / 4] >> (2 * (k % 4))) & 3;
            out[k] = ggml_fp16_to_fp32(h) * (float) (q - 1);
        }
    };
    gu.resize((size_t) 1280 * N);
    dn.resize((size_t) N * FF);
    for (int r = 0; r < 1280; ++r) row(0, 640, O_GU_SC, 40, r, gu.data() + (size_t) r * N);
    for (int r = 0; r < N; ++r) row(O_D_CODES, 160, O_D_SC, 10, r, dn.data() + (size_t) r * FF);
}

// activations: N(0, 1) with a few large values (the hidden state has outliers); token `zero` all zero
std::vector<float> make_x(int T, std::mt19937& rng, int zero) {
    std::vector<float> x((size_t) T * N);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> pick(0, 199);
    for (float& v : x) v = nd(rng) * (pick(rng) == 0 ? 12.0f : 1.0f);
    if (zero >= 0) std::fill(x.begin() + (size_t) zero * N, x.begin() + (size_t) (zero + 1) * N, 0.0f);
    return x;
}

// K distinct experts per token; `hot` > 0 skews the first choices towards the lowest ids
std::vector<int32_t> make_ids(int T, int E, std::mt19937& rng, int hot, int unused) {
    std::vector<int32_t> ids((size_t) T * K);
    std::uniform_int_distribution<int> any(0, E - 1 - unused);
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < K; ++k) {
            int e;
            bool dup;
            do {
                e = (hot > 0 && k < 3 && (int) (rng() % 4) != 0) ? (int) (rng() % (unsigned) hot) : any(rng);
                dup = false;
                for (int j = 0; j < k; ++j) dup |= ids[(size_t) t * K + j] == e;
            } while (dup);
            ids[(size_t) t * K + k] = e;
        }
    return ids;
}

struct Routing {
    std::vector<int32_t> cnt, off, src;   // per expert; sorted row -> token (rows in pair order within an expert)
    std::vector<int32_t> row_of;          // pair -> sorted row
};
Routing sort_rows(const std::vector<int32_t>& ids, int E) {
    Routing r;
    r.cnt.assign((size_t) E, 0);
    for (int32_t e : ids) ++r.cnt[(size_t) e];
    r.off.assign((size_t) E + 1, 0);
    for (int e = 0; e < E; ++e) r.off[(size_t) e + 1] = r.off[(size_t) e] + r.cnt[(size_t) e];
    std::vector<int32_t> fill(r.off.begin(), r.off.end() - 1);
    r.src.resize(ids.size());
    r.row_of.resize(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        const int32_t p = fill[(size_t) ids[i]]++;
        r.row_of[i] = p;
        r.src[(size_t) p] = (int32_t) (i / K);
    }
    return r;
}

// MMQ, as prefill.cpp runs it: per group of 16 routed experts a gather, gate/up, SwiGLU, q8_1 of H, down.  Returns
// Dm [rows][N] in sorted-row order.
struct MmqBufs {
    Dev xq, gu, h, hq, dm, ident, bounds, grp_gu, grp_d;
    MmqBufs(int64_t rows, int E)
        : xq(mmq::q8_bytes(rows, N)), gu((size_t) rows * 1280 * 4), h((size_t) rows * FF * 4),
          hq(mmq::q8_bytes(rows, FF)),
          dm((size_t) rows * N * 4), ident((size_t) rows * 4), bounds((size_t) (2 * (E + E / GROUP + 2)) * 4),
          grp_gu(GROUP * mmq::matrix_bytes(Q2, 1280, N) + 4096), grp_d(GROUP * mmq::matrix_bytes(Q2, N, FF) + 4096) {}
};
void run_mmq(mmq::Context& ctx, MmqBufs& b, const Routing& r, const float* x_dev, const int32_t* src_dev,
             const std::vector<const uint8_t*>& blob, int64_t rows, cudaStream_t s) {
    const size_t gub = mmq::matrix_bytes(Q2, 1280, N), db = mmq::matrix_bytes(Q2, N, FF);
    const int E = (int) r.cnt.size();
    std::vector<int32_t> order;
    for (int e = 0; e < E; ++e) if (r.cnt[(size_t) e] > 0) order.push_back(e);
    const size_t n = order.size(), ng = (n + GROUP - 1) / GROUP;
    std::vector<int32_t> bh(n + 1 + ng * (GROUP + 1));
    for (size_t j = 0; j < n; ++j) bh[j] = r.off[(size_t) order[j]];
    bh[n] = (int32_t) rows;
    for (size_t g = 0; g < ng; ++g)
        for (size_t i = 0; i <= GROUP; ++i)
            bh[n + 1 + g * (GROUP + 1) + i] = bh[std::min(n, g * GROUP + i)] - bh[g * GROUP];
    ck(cudaMemcpyAsync(b.bounds.p, bh.data(), bh.size() * 4, cudaMemcpyHostToDevice, s), "bounds");
    mmq::iota(b.ident.as<int32_t>(), rows, s);
    mmq::quantize(x_dev, src_dev, b.xq.p, Q2, N, N, rows, s);
    for (size_t j = 0; j < n; ++j) {
        const size_t q = j % GROUP;
        mmq::gather_strata_q2(blob[(size_t) order[j]], b.grp_gu.as<uint8_t>() + q * gub, b.grp_d.as<uint8_t>() + q * db,
                              s);
        if (q + 1 < GROUP && j + 1 < n) continue;
        const size_t j0 = j - q, g = j0 / GROUP;
        const int ngx = (int) (q + 1);
        const int64_t r0 = bh[j0], nr = bh[j + 1] - r0;
        int64_t maxr = 0;
        for (size_t i = j0; i <= j; ++i) maxr = std::max<int64_t>(maxr, r.cnt[(size_t) order[i]]);
        ck(cudaMemsetAsync(b.grp_gu.as<uint8_t>() + (size_t) ngx * gub, 0, 4096, s), "tail");
        ck(cudaMemsetAsync(b.grp_d.as<uint8_t>() + (size_t) ngx * db, 0, 4096, s), "tail");
        mmq::Product gu;
        gu.w = b.grp_gu.p; gu.type = Q2; gu.w_rows = 1280; gu.w_cols = N; gu.expert_bytes = gub;
        gu.n = ngx; gu.xq = b.xq.p; gu.bounds = b.bounds.as<int32_t>() + j0; gu.ids = b.ident.as<int32_t>();
        gu.total_rows = rows; gu.max_rows = maxr; gu.dst = b.gu.as<float>(); gu.ld_dst = 1280;
        ctx.run(gu, s);
        mmq::swiglu(b.gu.as<float>() + r0 * 1280, b.h.as<float>() + r0 * FF, nr, FF, true, s);
        mmq::quantize(b.h.as<float>() + r0 * FF, nullptr, b.hq.p, Q2, FF, FF, nr, s);
        mmq::Product dn;
        dn.w = b.grp_d.p; dn.type = Q2; dn.w_rows = N; dn.w_cols = FF; dn.expert_bytes = db;
        dn.n = ngx; dn.xq = b.hq.p; dn.bounds = b.bounds.as<int32_t>() + n + 1 + g * (GROUP + 1);
        dn.ids = b.ident.as<int32_t>(); dn.total_rows = nr; dn.max_rows = maxr; dn.dst = b.dm.as<float>() + r0 * N;
        dn.ld_dst = N;
        ctx.run(dn, s);
    }
}

// the fused path: quantize, group on the device, then launches over expert ranges `cuts` (e.g. {0, 20, 64})
struct FusedBufs {
    Dev xa, ha, dm, scratch, slot, src, ids;
    FusedBufs(int T, int64_t rows, int E)
        : xa(fused::act_bytes(T, N)), ha(fused::act_bytes(rows, FF)), dm((size_t) rows * N * 4),
          scratch(fused::group_bytes(rows, E)), slot((size_t) rows * 4), src((size_t) rows * 4),
          ids((size_t) rows * 4) {}
};
void run_fused(FusedBufs& b, const float* x_dev, int T, int E, const std::vector<int>& cuts,
               const std::vector<const uint8_t*>& blob, cudaStream_t s) {
    const int64_t rows = (int64_t) T * K;
    fused::quantize_act(x_dev, T, N, b.xa.p, s);
    fused::group(b.ids.as<int32_t>(), rows, K, E, b.scratch.p, b.slot.as<int32_t>(), b.src.as<int32_t>(), s);
    for (size_t c = 0; c + 1 < cuts.size(); ++c) {
        for (int e0 = cuts[c]; e0 < cuts[c + 1]; e0 += fused::kMaxBatch) {
            fused::Batch bt;
            bt.e0 = e0;
            bt.e1 = std::min(cuts[c + 1], e0 + fused::kMaxBatch);
            for (int e = bt.e0; e < bt.e1; ++e) bt.blob[e - bt.e0] = blob[(size_t) e];
            fused::experts(bt, E, rows, b.scratch.p, b.xa.p, b.src.as<int32_t>(), b.ha.p, b.dm.as<float>(), s);
        }
    }
}

struct Err {
    double rms = 0, worst = 0;   // RMS error / RMS of the reference; the worst row's max |error| / its max |ref|
};
// y_a[row_a(i)] against y_b[row_b(i)] for every pair i (rows of N values)
template <typename RA, typename RB>
Err compare(const std::vector<float>& a, RA row_a, const std::vector<float>& b, RB row_b, size_t pairs) {
    double e2 = 0, r2 = 0;
    Err out;
    for (size_t i = 0; i < pairs; ++i) {
        const float* ya = a.data() + (size_t) row_a(i) * N;
        const float* yb = b.data() + (size_t) row_b(i) * N;
        double me = 0, mr = 0;
        for (int o = 0; o < N; ++o) {
            if (!std::isfinite(ya[o])) throw std::runtime_error("a non-finite output");
            const double d = (double) ya[o] - yb[o];
            e2 += d * d;
            r2 += (double) yb[o] * yb[o];
            me = std::max(me, std::fabs(d));
            mr = std::max(mr, (double) std::fabs(yb[o]));
        }
        if (mr > 0) out.worst = std::max(out.worst, me / mr);
        else if (me > 0) out.worst = 1e30;   // a zero row must stay exactly zero
    }
    out.rms = r2 > 0 ? std::sqrt(e2 / r2) : 0;
    return out;
}

std::vector<float> download(const Dev& d, size_t n) {
    std::vector<float> h(n);
    ck(cudaMemcpy(h.data(), d.p, n * 4, cudaMemcpyDeviceToHost), "download");
    return h;
}
std::vector<int32_t> download_i(const Dev& d, size_t n) {
    std::vector<int32_t> h(n);
    ck(cudaMemcpy(h.data(), d.p, n * 4, cudaMemcpyDeviceToHost), "download");
    return h;
}

// part 1: against the double-precision reference
void reference_part(cudaStream_t s) {
    constexpr int T = 256, E = 64, ZERO = 77;
    std::mt19937 rng(136);
    std::vector<std::vector<uint8_t>> host((size_t) E);
    std::vector<std::unique_ptr<Dev>> dev;
    std::vector<const uint8_t*> blob((size_t) E);
    for (int e = 0; e < E; ++e) {
        host[(size_t) e] = make_blob(rng);
        dev.push_back(std::make_unique<Dev>(BLOB + 4096 * (size_t) (e % 3)));   // unrelated addresses
        uint8_t* p = dev.back()->as<uint8_t>() + 256 * (size_t) (e % 3);       // 16-byte aligned, not 512
        ck(cudaMemcpy(p, host[(size_t) e].data(), BLOB, cudaMemcpyHostToDevice), "blob");
        blob[(size_t) e] = p;
    }
    const std::vector<float> x = make_x(T, rng, ZERO);
    const std::vector<int32_t> ids = make_ids(T, E, rng, 2, 4);   // experts 0, 1 hot (several tiles), 60-63 unused
    const int64_t rows = (int64_t) T * K;
    const Routing r = sort_rows(ids, E);
    std::printf("reference part: %d tokens, %d experts, rows per expert: max %d, experts without rows %d\n", T, E,
                *std::max_element(r.cnt.begin(), r.cnt.end()), (int) std::count(r.cnt.begin(), r.cnt.end(), 0));

    Dev x_dev(x.size() * 4), src_dev((size_t) rows * 4);
    ck(cudaMemcpy(x_dev.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(src_dev.p, r.src.data(), r.src.size() * 4, cudaMemcpyHostToDevice), "src");
    mmq::Context ctx;
    MmqBufs mb(rows, E);
    // the uploads (legacy stream) must land before the non-blocking stream reads them (S20 found the kernels could run
    // on all-zero routing otherwise - every pair on one expert: fast and wrong)
    ck(cudaDeviceSynchronize(), "uploads");
    run_mmq(ctx, mb, r, x_dev.as<float>(), src_dev.as<int32_t>(), blob, rows, s);
    FusedBufs fb(T, rows, E);
    ck(cudaMemcpy(fb.ids.p, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "ids");
    ck(cudaMemset(fb.dm.p, 0xff, (size_t) rows * N * 4), "sentinel");   // an unwritten row is NaN
    ck(cudaDeviceSynchronize(), "uploads");
    run_fused(fb, x_dev.as<float>(), T, E, {0, 1, 20, E}, blob, s);
    ck(cudaStreamSynchronize(s), "sync");
    const std::vector<float> y_mmq = download(mb.dm, (size_t) rows * N), y_f = download(fb.dm, (size_t) rows * N);
    const std::vector<int32_t> slot = download_i(fb.slot, (size_t) rows), fsrc = download_i(fb.src, (size_t) rows);
    for (int64_t i = 0; i < rows; ++i)
        if (fsrc[(size_t) slot[(size_t) i]] != (int32_t) (i / K)) throw std::runtime_error("group: slot/src disagree");

    // the reference, an expert at a time on all cores
    std::vector<float> ref((size_t) rows * N);
    std::vector<std::thread> th;
    const int nth = std::max(1, std::min(16, (int) std::thread::hardware_concurrency()));
    for (int w = 0; w < nth; ++w)
        th.emplace_back([&, w] {
            std::vector<float> gu, dn, h(FF);
            for (int e = w; e < E; e += nth) {
                if (r.cnt[(size_t) e] == 0) continue;
                dequant(host[(size_t) e], gu, dn);
                for (int64_t i = 0; i < rows; ++i) {
                    if (ids[(size_t) i] != e) continue;
                    const float* xr = x.data() + (size_t) (i / K) * N;
                    for (int f = 0; f < FF; ++f) {
                        double gt = 0, up = 0;
                        const float *wg = gu.data() + (size_t) (2 * f) * N, *wu = wg + N;
                        for (int k = 0; k < N; ++k) { gt += (double) wg[k] * xr[k]; up += (double) wu[k] * xr[k]; }
                        h[(size_t) f] = (float) (gt / (1.0 + std::exp(-gt)) * up);
                    }
                    for (int o = 0; o < N; ++o) {
                        double a = 0;
                        const float* wd = dn.data() + (size_t) o * FF;
                        for (int f = 0; f < FF; ++f) a += (double) wd[f] * h[(size_t) f];
                        ref[(size_t) i * N + o] = (float) a;
                    }
                }
            }
        });
    for (auto& t : th) t.join();

    auto pair = [](size_t i) { return (int64_t) i; };
    auto mrow = [&](size_t i) { return (int64_t) r.row_of[i]; };
    auto frow = [&](size_t i) { return (int64_t) slot[i]; };
    const Err em = compare(y_mmq, mrow, ref, pair, (size_t) rows);
    const Err ef = compare(y_f, frow, ref, pair, (size_t) rows);
    const Err efm = compare(y_f, frow, y_mmq, mrow, (size_t) rows);
    std::printf("  MMQ   vs FP32 reference: rel RMS %.3e  worst row max rel %.3e\n", em.rms, em.worst);
    std::printf("  fused vs FP32 reference: rel RMS %.3e  worst row max rel %.3e\n", ef.rms, ef.worst);
    std::printf("  fused vs MMQ           : rel RMS %.3e  worst row max rel %.3e\n", efm.rms, efm.worst);
    double zmax = 0;
    for (int k = 0; k < K; ++k)
        for (int o = 0; o < N; ++o)
            zmax = std::max(zmax, (double) std::fabs(y_f[(size_t) slot[(size_t) (ZERO * K + k)] * N + o]));
    if (zmax != 0) throw std::runtime_error("the all-zero token's outputs are not zero");
    if (ef.rms > 1.5 * em.rms || ef.worst > 2.0 * em.worst)
        throw std::runtime_error("the fused path's error is not comparable to MMQ's");
}

// part 2: one layer at a real chunk, timed
void timing_part(cudaStream_t s) {
    constexpr int T = 2048, E = 512;
    std::mt19937 rng(1360);
    const int64_t rows = (int64_t) T * K;
    // 512 distinct blobs (708 MB) - the products read each once per tile, as the engine does
    Dev blobs((size_t) E * BLOB);
    {
        std::vector<uint8_t> b = make_blob(rng);
        for (int e = 0; e < E; ++e) {
            b[(size_t) e * 7919 % O_GU_SC] ^= (uint8_t) e;   // not all equal
            ck(cudaMemcpy(blobs.as<uint8_t>() + (size_t) e * BLOB, b.data(), BLOB, cudaMemcpyHostToDevice), "blob");
        }
    }
    std::vector<const uint8_t*> blob((size_t) E);
    for (int e = 0; e < E; ++e) blob[(size_t) e] = blobs.as<uint8_t>() + (size_t) e * BLOB;
    const std::vector<float> x = make_x(T, rng, -1);
    const std::vector<int32_t> ids = make_ids(T, E, rng, 0, 0);
    const Routing r = sort_rows(ids, E);
    Dev x_dev(x.size() * 4), src_dev((size_t) rows * 4);
    ck(cudaMemcpy(x_dev.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(src_dev.p, r.src.data(), r.src.size() * 4, cudaMemcpyHostToDevice), "src");
    mmq::Context ctx;
    MmqBufs mb(rows, E);
    FusedBufs fb(T, rows, E);
    ck(cudaMemcpy(fb.ids.p, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "ids");
    ck(cudaDeviceSynchronize(), "uploads");   // the legacy-stream uploads before the timed non-blocking stream
    cudaEvent_t a, b;
    ck(cudaEventCreate(&a), "event");
    ck(cudaEventCreate(&b), "event");
    auto time = [&](auto&& f) {
        std::vector<float> ms;
        for (int rep = 0; rep < 6; ++rep) {
            ck(cudaEventRecord(a, s), "record");
            f();
            ck(cudaEventRecord(b, s), "record");
            ck(cudaEventSynchronize(b), "sync");
            float t = 0;
            ck(cudaEventElapsedTime(&t, a, b), "elapsed");
            if (rep > 0) ms.push_back(t);   // the first is the warm-up
        }
        std::sort(ms.begin(), ms.end());
        return ms[ms.size() / 2];
    };
    // the engine's batches: up to 128 experts per launch (the streamed ring decides; 4 launches here)
    const float t_mmq = time([&] { run_mmq(ctx, mb, r, x_dev.as<float>(), src_dev.as<int32_t>(), blob, rows, s); });
    const float t_f = time([&] { run_fused(fb, x_dev.as<float>(), T, E, {0, E}, blob, s); });
    const float t_f1 = time([&] {   // one launch per 16 experts, MMQ's group size
        std::vector<int> cuts;
        for (int e = 0; e <= E; e += 16) cuts.push_back(e);
        run_fused(fb, x_dev.as<float>(), T, E, cuts, blob, s);
    });
    const std::vector<float> y_mmq = download(mb.dm, (size_t) rows * N), y_f = download(fb.dm, (size_t) rows * N);
    const std::vector<int32_t> slot = download_i(fb.slot, (size_t) rows);
    const Err efm = compare(y_f, [&](size_t i) { return (int64_t) slot[i]; }, y_mmq,
                            [&](size_t i) { return (int64_t) r.row_of[i]; }, (size_t) rows);
    std::printf("timing part: %d tokens x top %d over %d experts (%.1f rows per expert), one layer:\n", T, K, E,
                (double) rows / E);
    std::printf("  MMQ path (gathers, q8_1, products, SwiGLU, 32 groups): %.3f ms\n", t_mmq);
    std::printf("  fused path (quantize, group, 4 launches of 128):       %.3f ms  (%.2fx)\n", t_f, t_mmq / t_f);
    std::printf("  fused path, launches of 16 experts:                    %.3f ms\n", t_f1);
    std::printf("  fused vs MMQ: rel RMS %.3e  worst row max rel %.3e\n", efm.rms, efm.worst);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    if (efm.rms > 0.05) throw std::runtime_error("the paths disagree at the real shape");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) { std::printf("no CUDA device: skipped\n"); return 77; }
        if (!fused::available()) { std::printf("the fused kernels need sm_80 or newer: skipped\n"); return 77; }
        if (!mmq::built()) { std::printf("no MMQ in this build\n"); return 1; }
        cudaStream_t s = nullptr;
        ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        reference_part(s);
        if (!(argc > 1 && std::strcmp(argv[1], "--no-timing") == 0)) timing_part(s);
        ck(cudaStreamDestroy(s), "destroy");
        std::printf("prefill fused MoE parity passed\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "prefill fused MoE parity failed: %s\n", e.what());
        return 1;
    }
}
