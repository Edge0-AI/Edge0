// prefill_fused_iq_test - #136: the fused int8 expert kernels for the native packs (moe_fused_iq.hpp, STRATA_PF_FUSED=1)
// on random native expert blobs ([gate rows | up rows | down rows], GGUF blocks) and random routing, per format pair,
// against
//   - a double-precision reference: the weights dequantized by ggml's own to_float, times the FP32 activations
//     (gate/up, SwiGLU, down), and
//   - the MMQ path in prefill.cpp's sequence (gather_native into 16-expert groups, q8_1 of the slots' activations,
//     gate/up, SwiGLU, q8_1 of H, down).
// Both round the activations and H to int8 per 32 values, differently, so this is not a bitwise test: the fused
// path's error against the reference has to be comparable to MMQ's own (at most 1.5x its RMS and 2x its worst row).
// The pairs cover every format the kernels take: gate/up IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S IQ4_XS, down Q2_0 IQ4_NL.
// Part 1 (reference), per pair: 256 tokens over 64 experts, skewed routing - experts with 0 rows, with one, with
// several 64-row tiles - an all-zero token, per-expert blobs at unrelated (2-byte aligned) addresses, three launches of
// different expert ranges.  Part 2 (timing): one layer at the engine's chunks (2048, 3584, 8192 tokens; 512 experts,
// top 10) on both paths and on MMQ's products alone (no gathers), for the IQ2_XS, IQ3_XXS and IQ3_S packs' most
// common layers, with the grouping checked and a sample of rows against the reference.  --no-ref, --no-timing,
// --only=NAME, --chunks=A,B, --unset-pf-fused (STRATA_PF_FUSED left unset instead of set to 1: the engine's default, for the
// STRATA_PF_FUSED_Q2N=1 ctest entry).  Exit 77 without a CUDA device of sm_80 or newer.
#include "strata/core/emulate.hpp"
#include "strata/prefill/moe_fused_iq.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

constexpr int N = 2560, FF = 640, K = 10, GROUP = 16;

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

struct Pair {
    ggml_type gu, d;
    const char* name;
};
// MMQ covers this pair (the HIP build has no Q4_K / Q5_K / Q5_1 MMQ: those pairs are checked against the reference only)
bool mmq_ok(const Pair& p) { return mmq::supported((int) p.gu) && mmq::supported((int) p.d); }

struct Geo {
    size_t gu_row, d_row, up_off, down_off, bytes;
    explicit Geo(const Pair& p) {
        gu_row = ggml_row_size(p.gu, N);
        d_row = ggml_row_size(p.d, FF);
        up_off = gu_row * FF;
        down_off = 2 * up_off;
        bytes = down_off + d_row * N;
    }
    fused::NativeGeom native(const Pair& p) const {
        fused::NativeGeom g;
        g.gu_type = (int) p.gu; g.d_type = (int) p.d; g.gu_row = gu_row; g.d_row = d_row; g.up_off = up_off;
        g.down_off = down_off;
        return g;
    }
};

// random blocks with a sane fp16 scale `d` at each block's start (every format here begins with it): scales chosen so
// STRATA_PF_F16=1: the covered formats run on F16 weights (moe_f16_kq.cu, native_f16_kernel), not the int8 kernels
bool f16_on() {
    const char* v = std::getenv("STRATA_PF_F16");
    // STRATA_PF_FUSED_F16=1 / STRATA_PF_FUSED_WS=1 (the int8 kernels' f16 mma form, #3779): F16 products too, so the same limits
    const char* f = std::getenv("STRATA_PF_FUSED_F16");
    const char* w = std::getenv("STRATA_PF_FUSED_WS");
    return (v != nullptr && v[0] == '1') || (f != nullptr && f[0] == '1') || (w != nullptr && w[0] == '1');
}

// that the weights' typical size is that of a 2-3 bit expert (a few hundredths)
void fill_matrix(uint8_t* m, ggml_type t, int64_t rows, int64_t cols, std::mt19937& rng) {
    const size_t bs = ggml_type_size(t), nb = (size_t) rows * (size_t) (cols / ggml_blck_size(t));
    std::uniform_int_distribution<int> byte(0, 255);
    const float lo = t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K ? 0.00004f : t == GGML_TYPE_Q5_1 ? 0.0006f
                   : t == GGML_TYPE_Q8_0 ? 0.0001f : t == GGML_TYPE_IQ3_S ? 0.0004f : t == GGML_TYPE_IQ4_XS ? 0.00005f
                   : t == GGML_TYPE_IQ4_NL ? 0.0002f : t == GGML_TYPE_Q2_0 ? 0.004f : 0.0008f;
    std::uniform_real_distribution<float> sc(lo, 6.0f * lo);
    for (size_t i = 0; i < nb * bs; ++i) m[i] = (uint8_t) byte(rng);
    for (size_t b = 0; b < nb; ++b) {
        const ggml_fp16_t h = ggml_fp32_to_fp16(sc(rng));
        std::memcpy(m + b * bs, &h, 2);
        if (t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K || t == GGML_TYPE_Q5_1) {   // the minimum's scale, either sign
            const ggml_fp16_t h2 = ggml_fp32_to_fp16(sc(rng) * ((rng() & 1) ? 1.0f : -1.0f) * (t == GGML_TYPE_Q5_1 ? 4.0f : 1.0f));
            std::memcpy(m + b * bs + 2, &h2, 2);
        }
    }
}
std::vector<uint8_t> make_blob(const Pair& p, const Geo& g, std::mt19937& rng) {
    std::vector<uint8_t> b(g.bytes);
    fill_matrix(b.data(), p.gu, 2 * FF, N, rng);
    fill_matrix(b.data() + g.down_off, p.d, N, FF, rng);
    return b;
}

// ggml's dequantized weights: gate [640][2560], up [640][2560], down [2560][640]
void dequant(const Pair& p, const Geo& g, const std::vector<uint8_t>& b, std::vector<float>& gu, std::vector<float>& dn) {
    const auto* tg = ggml_get_type_traits(p.gu);
    const auto* td = ggml_get_type_traits(p.d);
    gu.resize((size_t) 2 * FF * N);
    dn.resize((size_t) N * FF);
    for (int r = 0; r < 2 * FF; ++r) tg->to_float(b.data() + (size_t) r * g.gu_row, gu.data() + (size_t) r * N, N);
    for (int r = 0; r < N; ++r) td->to_float(b.data() + g.down_off + (size_t) r * g.d_row, dn.data() + (size_t) r * FF, FF);
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

// MMQ, as prefill.cpp runs a native layer: per group of 16 routed experts a gather, gate/up, SwiGLU (gate rows, then
// up rows), q8_1 of H, down.  Returns Dm [rows][N] in sorted-row order.
struct MmqBufs {
    Dev xq, gu, h, hq, dm, ident, bounds, grp_gu, grp_d;
    MmqBufs(const Pair& p, int64_t rows, int E)
        : xq(mmq::q8_bytes(rows, N)), gu((size_t) rows * 1280 * 4), h((size_t) rows * FF * 4),
          hq(mmq::q8_bytes(rows, FF)), dm((size_t) rows * N * 4), ident((size_t) rows * 4),
          bounds((size_t) (2 * (E + E / GROUP + 2)) * 4), grp_gu(GROUP * mmq::matrix_bytes(p.gu, 1280, N) + 4096),
          grp_d(GROUP * mmq::matrix_bytes(p.d, N, FF) + 4096) {}
};
// `direct`: no gathers - the products read the blobs where they are (they must lie at a fixed stride, every expert
// with rows: the timing part's layout), the cost of MMQ's own kernels alone
void run_mmq(const Pair& p, const Geo& geo, mmq::Context& ctx, MmqBufs& b, const Routing& r, const float* x_dev,
             const int32_t* src_dev, const std::vector<const uint8_t*>& blob, int64_t rows, cudaStream_t s,
             bool direct = false, size_t stride = 0) {
    const size_t gub = mmq::matrix_bytes(p.gu, 1280, N), db = mmq::matrix_bytes(p.d, N, FF);
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
    mmq::quantize(x_dev, src_dev, b.xq.p, (int) p.gu, N, N, rows, s);
    for (size_t j = 0; j < n; ++j) {
        const size_t q = j % GROUP;
        const uint8_t* bl = blob[(size_t) order[j]];
        if (!direct)
            mmq::gather_native(bl, bl + geo.up_off, gub / 2, bl + geo.down_off, db, b.grp_gu.as<uint8_t>() + q * gub,
                               b.grp_d.as<uint8_t>() + q * db, s);
        if (q + 1 < GROUP && j + 1 < n) continue;
        const size_t j0 = j - q, g = j0 / GROUP;
        const int ngx = (int) (q + 1);
        const int64_t r0 = bh[j0], nr = bh[j + 1] - r0;
        int64_t maxr = 0;
        for (size_t i = j0; i <= j; ++i) maxr = std::max<int64_t>(maxr, r.cnt[(size_t) order[i]]);
        if (!direct) {
            ck(cudaMemsetAsync(b.grp_gu.as<uint8_t>() + (size_t) ngx * gub, 0, 4096, s), "tail");
            ck(cudaMemsetAsync(b.grp_d.as<uint8_t>() + (size_t) ngx * db, 0, 4096, s), "tail");
        }
        const uint8_t* b0 = blob[(size_t) order[j0]];
        mmq::Product gu;
        gu.w = direct ? (const void*) b0 : b.grp_gu.p; gu.type = (int) p.gu; gu.w_rows = 1280; gu.w_cols = N;
        gu.expert_bytes = direct ? stride : gub;
        gu.n = ngx; gu.xq = b.xq.p; gu.bounds = b.bounds.as<int32_t>() + j0; gu.ids = b.ident.as<int32_t>();
        gu.total_rows = rows; gu.max_rows = maxr; gu.dst = b.gu.as<float>(); gu.ld_dst = 1280;
        ctx.run(gu, s);
        mmq::swiglu(b.gu.as<float>() + r0 * 1280, b.h.as<float>() + r0 * FF, nr, FF, false, s);
        mmq::quantize(b.h.as<float>() + r0 * FF, nullptr, b.hq.p, (int) p.d, FF, FF, nr, s);
        mmq::Product dn;
        dn.w = direct ? (const void*) (b0 + geo.down_off) : b.grp_d.p; dn.type = (int) p.d; dn.w_rows = N; dn.w_cols = FF;
        dn.expert_bytes = direct ? stride : db;
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
void run_fused(const Pair& p, const Geo& geo, FusedBufs& b, const float* x_dev, int T, int E,
               const std::vector<int>& cuts, const std::vector<const uint8_t*>& blob, cudaStream_t s) {
    const int64_t rows = (int64_t) T * K;
    const fused::NativeGeom ng = geo.native(p);
    fused::quantize_act_native(x_dev, T, N, b.xa.p, s);
    fused::group(b.ids.as<int32_t>(), rows, K, E, b.scratch.p, b.slot.as<int32_t>(), b.src.as<int32_t>(), s);
    for (size_t c = 0; c + 1 < cuts.size(); ++c) {
        for (int e0 = cuts[c]; e0 < cuts[c + 1]; e0 += fused::kMaxBatch) {
            fused::Batch bt;
            bt.e0 = e0;
            bt.e1 = std::min(cuts[c + 1], e0 + fused::kMaxBatch);
            for (int e = bt.e0; e < bt.e1; ++e) bt.blob[e - bt.e0] = blob[(size_t) e];
            fused::experts_native(bt, ng, E, rows, b.scratch.p, b.xa.p, b.src.as<int32_t>(), b.ha.p, b.dm.as<float>(), s);
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

// the kernels' int8 rounding per 32 values (quant_act_nat_kernel / the H epilogue), as floats: codes * (amax / 127)
void quant32(const float* in, float* out, int n) {
    for (int b = 0; b < n; b += 32) {
        float am = 0.0f;
        for (int j = 0; j < 32; ++j) am = std::max(am, std::fabs(in[b + j]));
        const float inv = am > 0.0f ? 127.0f / am : 0.0f, d = am / 127.0f;
        for (int j = 0; j < 32; ++j) out[b + j] = (float) std::nearbyint(in[b + j] * inv) * d;
    }
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
void reference_part(const Pair& p, cudaStream_t s) {
    constexpr int T = 256, E = 64, ZERO = 77;
    const Geo geo(p);
    std::mt19937 rng(136 + (unsigned) p.gu * 7 + (unsigned) p.d);
    std::vector<std::vector<uint8_t>> host((size_t) E);
    std::vector<std::unique_ptr<Dev>> dev;
    std::vector<const uint8_t*> blob((size_t) E);
    for (int e = 0; e < E; ++e) {
        host[(size_t) e] = make_blob(p, geo, rng);
        dev.push_back(std::make_unique<Dev>(geo.bytes + 4096 * (size_t) (e % 3) + 64));   // unrelated addresses
        uint8_t* q = dev.back()->as<uint8_t>() + 256 * (size_t) (e % 3) + 2 * (size_t) (e % 2);   // 2-byte aligned
        ck(cudaMemcpy(q, host[(size_t) e].data(), geo.bytes, cudaMemcpyHostToDevice), "blob");
        blob[(size_t) e] = q;
    }
    const std::vector<float> x = make_x(T, rng, ZERO);
    const std::vector<int32_t> ids = make_ids(T, E, rng, 2, 4);   // experts 0, 1 hot (several tiles), 60-63 unused
    const int64_t rows = (int64_t) T * K;
    const Routing r = sort_rows(ids, E);
    std::printf("%s - reference part: %d tokens, %d experts, rows per expert: max %d, experts without rows %d\n", p.name,
                T, E, *std::max_element(r.cnt.begin(), r.cnt.end()), (int) std::count(r.cnt.begin(), r.cnt.end(), 0));

    Dev x_dev(x.size() * 4), src_dev((size_t) rows * 4);
    ck(cudaMemcpy(x_dev.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(src_dev.p, r.src.data(), r.src.size() * 4, cudaMemcpyHostToDevice), "src");
    // (the uploads above are on the legacy stream and the work below on a non-blocking one: a cudaMemcpy from pageable
    // memory may return before its DMA lands - on Windows it can wait in the queue until the next synchronous call)
    ck(cudaDeviceSynchronize(), "uploads");
    const bool mq = mmq_ok(p);
    mmq::Context ctx;
    MmqBufs* mbp = mq ? new MmqBufs(p, rows, E) : nullptr;
    if (mq) run_mmq(p, geo, ctx, *mbp, r, x_dev.as<float>(), src_dev.as<int32_t>(), blob, rows, s);
    FusedBufs fb(T, rows, E);
    ck(cudaMemcpy(fb.ids.p, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "ids");
    ck(cudaMemset(fb.dm.p, 0xff, (size_t) rows * N * 4), "sentinel");   // an unwritten row is NaN
    ck(cudaDeviceSynchronize(), "uploads");
    run_fused(p, geo, fb, x_dev.as<float>(), T, E, {0, 1, 20, E}, blob, s);
    ck(cudaStreamSynchronize(s), "sync");
    const std::vector<float> y_mmq = mq ? download(mbp->dm, (size_t) rows * N) : std::vector<float>((size_t) rows * N, 0.0f),
                             y_f = download(fb.dm, (size_t) rows * N);
    delete mbp;
    const std::vector<int32_t> slot = download_i(fb.slot, (size_t) rows), fsrc = download_i(fb.src, (size_t) rows);
    for (int64_t i = 0; i < rows; ++i)
        if (fsrc[(size_t) slot[(size_t) i]] != (int32_t) (i / K)) throw std::runtime_error("group: slot/src disagree");

    // the reference, an expert at a time on all cores
    std::vector<float> ref((size_t) rows * N), ref2((size_t) rows * N);   // ref2: the same with the kernels' int8 rounding
    std::vector<std::thread> th;
    const int nth = std::max(1, std::min(16, (int) std::thread::hardware_concurrency()));
    for (int w = 0; w < nth; ++w)
        th.emplace_back([&, w] {
            std::vector<float> gu, dn, h(FF), h2(FF), hq(FF), xq(N);
            for (int e = w; e < E; e += nth) {
                if (r.cnt[(size_t) e] == 0) continue;
                dequant(p, geo, host[(size_t) e], gu, dn);
                for (int64_t i = 0; i < rows; ++i) {
                    if (ids[(size_t) i] != e) continue;
                    const float* xr = x.data() + (size_t) (i / K) * N;
                    for (int f = 0; f < FF; ++f) {
                        double gt = 0, up = 0;
                        const float *wg = gu.data() + (size_t) f * N, *wu = gu.data() + (size_t) (FF + f) * N;
                        for (int k = 0; k < N; ++k) { gt += (double) wg[k] * xr[k]; up += (double) wu[k] * xr[k]; }
                        h[(size_t) f] = (float) (gt / (1.0 + std::exp(-gt)) * up);
                    }
                    for (int o = 0; o < N; ++o) {
                        double a = 0;
                        const float* wd = dn.data() + (size_t) o * FF;
                        for (int f = 0; f < FF; ++f) a += (double) wd[f] * h[(size_t) f];
                        ref[(size_t) i * N + o] = (float) a;
                    }
                    quant32(xr, xq.data(), N);
                    for (int f = 0; f < FF; ++f) {
                        double gt = 0, up = 0;
                        const float *wg = gu.data() + (size_t) f * N, *wu = gu.data() + (size_t) (FF + f) * N;
                        for (int k = 0; k < N; ++k) { gt += (double) wg[k] * xq[(size_t) k]; up += (double) wu[k] * xq[(size_t) k]; }
                        h2[(size_t) f] = (float) (gt / (1.0 + std::exp(-gt)) * up);
                    }
                    quant32(h2.data(), hq.data(), FF);
                    for (int o = 0; o < N; ++o) {
                        double a = 0;
                        const float* wd = dn.data() + (size_t) o * FF;
                        for (int f = 0; f < FF; ++f) a += (double) wd[f] * hq[(size_t) f];
                        ref2[(size_t) i * N + o] = (float) a;
                    }
                }
            }
        });
    for (auto& t : th) t.join();

    auto pair = [](size_t i) { return (int64_t) i; };
    auto mrow = [&](size_t i) { return (int64_t) r.row_of[i]; };
    auto frow = [&](size_t i) { return (int64_t) slot[i]; };
    const Err em = mq ? compare(y_mmq, mrow, ref, pair, (size_t) rows) : Err{};
    const Err ef = compare(y_f, frow, ref, pair, (size_t) rows);
    const Err efm = mq ? compare(y_f, frow, y_mmq, mrow, (size_t) rows) : Err{};
    const Err e2 = compare(y_f, frow, ref2, pair, (size_t) rows);
    std::printf("  fused vs the int8-rounded model (same activation / H rounding, double sums): rel RMS %.3e  worst %.3e\n",
                e2.rms, e2.worst);
    // STRATA_PF_F16=1: H's int8 rounding meets values a few F16 ulps from the int8 model's (f16_on); the FP32 reference
    // checks below stay as they are
    if (e2.rms > (f16_on() ? 5e-3 : 2e-3) || e2.worst > 3e-2)
        throw std::runtime_error(std::string(p.name) + ": the fused path differs from its own arithmetic's model");
    if (mq) std::printf("  MMQ   vs FP32 reference: rel RMS %.3e  worst row max rel %.3e\n", em.rms, em.worst);
    std::printf("  fused vs FP32 reference: rel RMS %.3e  worst row max rel %.3e\n", ef.rms, ef.worst);
    if (mq) std::printf("  fused vs MMQ           : rel RMS %.3e  worst row max rel %.3e\n", efm.rms, efm.worst);
    double zmax = 0;
    for (int k = 0; k < K; ++k)
        for (int o = 0; o < N; ++o)
            zmax = std::max(zmax, (double) std::fabs(y_f[(size_t) slot[(size_t) (ZERO * K + k)] * N + o]));
    if (zmax != 0) throw std::runtime_error(std::string(p.name) + ": the all-zero token's outputs are not zero");
    if (!mq) {   // no MMQ here: the int8 activations' own error (the IQ pairs' fused path sits at ~1e-2 rel RMS)
        if (ef.rms > 0.03 || ef.worst > 0.25)
            throw std::runtime_error(std::string(p.name) + ": the fused path's error against the reference is too large");
        return;
    }
    if (ef.rms > 1.5 * em.rms || ef.worst > 2.0 * em.worst)
        throw std::runtime_error(std::string(p.name) + ": the fused path's error is not comparable to MMQ's");
}

// part 2: one layer at a real chunk, timed
void timing_part(const Pair& p, int T, cudaStream_t s) {
    static const int E = [] { const char* v = std::getenv("S20_E"); return v != nullptr && std::atoi(v) > 0 ? std::atoi(v) : 512; }();   // S20_E=256: the Coder pack
    const Geo geo(p);
    std::mt19937 rng(1360);
    const int64_t rows = (int64_t) T * K;
    // 512 distinct blobs - the products read each once per tile, as the engine does
    // at a stride of whole blocks of both formats (MMQ's expert stride counts blocks), + MMQ's read past the end
    size_t unit = ggml_type_size(p.gu);
    while (unit % ggml_type_size(p.d) != 0 || unit % 16 != 0) unit += ggml_type_size(p.gu);
    const size_t stride = (geo.bytes + unit - 1) / unit * unit;
    Dev blobs((size_t) E * stride + 4096);
    ck(cudaMemset(blobs.p, 0, (size_t) E * stride + 4096), "zero");
    {
        std::vector<uint8_t> b = make_blob(p, geo, rng);
        for (int e = 0; e < E; ++e) {
            b[4 + (size_t) (e % 16)] ^= (uint8_t) e;   // not all equal (inside the first block, not its scale)
            ck(cudaMemcpy(blobs.as<uint8_t>() + (size_t) e * stride, b.data(), geo.bytes, cudaMemcpyHostToDevice),
               "blob");
        }
    }
    std::vector<const uint8_t*> blob((size_t) E);
    for (int e = 0; e < E; ++e) blob[(size_t) e] = blobs.as<uint8_t>() + (size_t) e * stride;
    const std::vector<float> x = make_x(T, rng, -1);
    const std::vector<int32_t> ids = make_ids(T, E, rng, 0, 0);
    const Routing r = sort_rows(ids, E);
    Dev x_dev(x.size() * 4), src_dev((size_t) rows * 4);
    ck(cudaMemcpy(x_dev.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(src_dev.p, r.src.data(), r.src.size() * 4, cudaMemcpyHostToDevice), "src");
    const bool mq = mmq_ok(p);
    mmq::Context ctx;
    MmqBufs* mbp = mq ? new MmqBufs(p, rows, E) : nullptr;
    MmqBufs& mb = *mbp;   // (only used when mq)
    FusedBufs fb(T, rows, E);
    ck(cudaMemcpy(fb.ids.p, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "ids");
    ck(cudaDeviceSynchronize(), "uploads");   // (see reference_part)
    cudaEvent_t a, b;
    ck(cudaEventCreate(&a), "event");
    ck(cudaEventCreate(&b), "event");
    auto once = [&](auto&& f) {
        ck(cudaEventRecord(a, s), "record");
        f();
        ck(cudaEventRecord(b, s), "record");
        ck(cudaEventSynchronize(b), "sync");
        float t = 0;
        ck(cudaEventElapsedTime(&t, a, b), "elapsed");
        return t;
    };
    auto mmq_run = [&] { run_mmq(p, geo, ctx, mb, r, x_dev.as<float>(), src_dev.as<int32_t>(), blob, rows, s); };
    auto fused_run = [&] { run_fused(p, geo, fb, x_dev.as<float>(), T, E, {0, E}, blob, s); };
    const bool all_routed = mq && std::count(r.cnt.begin(), r.cnt.end(), 0) == 0;
    auto direct_run = [&] {
        run_mmq(p, geo, ctx, mb, r, x_dev.as<float>(), src_dev.as<int32_t>(), blob, rows, s, true, stride);
    };
    if (std::getenv("S20_IDSDIAG")) {   // which step overwrites the routing ids on the device
        auto check = [&](const char* after) {
            ck(cudaStreamSynchronize(s), "sync");
            const std::vector<int32_t> d = download_i(fb.ids, (size_t) rows);
            int64_t bad = 0;
            for (size_t i = 0; i < (size_t) rows; ++i) bad += d[i] != ids[i];
            std::printf("    ids on the device after %s: %lld of %lld differ (fb.ids %p, mb.dm %p..%p, fb.dm %p..%p)\n",
                        after, (long long) bad, (long long) rows, fb.ids.p, mb.dm.p,
                        (void*) (mb.dm.as<uint8_t>() + (size_t) rows * N * 4), fb.dm.p,
                        (void*) (fb.dm.as<uint8_t>() + (size_t) rows * N * 4));
        };
        check("the upload");
        mmq_run();
        check("an MMQ run");
        fused_run();
        check("a fused run");
        if (all_routed) {
            direct_run();
            check("an MMQ run without gathers");
        }
    }
    // ~1 s of both first (the clocks ramp up), then alternating rounds; the medians
    const bool no_mmq = !mq || std::getenv("S20_NOMMQ") != nullptr;   // (debug: the fused runs alone)
    for (float spent = 0; spent < 1000.0f;) spent += (no_mmq ? 0.0f : once(mmq_run)) + once(fused_run);
    std::vector<float> tm, tf, td;
    for (int rep = 0; rep < 8; ++rep) {
        tm.push_back(no_mmq ? 1.0f : once(mmq_run));
        tf.push_back(once(fused_run));
        if (all_routed && !no_mmq) td.push_back(once(direct_run));
    }
    if (td.empty()) td.push_back(0.0f);
    std::sort(tm.begin(), tm.end());
    std::sort(tf.begin(), tf.end());
    std::sort(td.begin(), td.end());
    const float t_mmq = tm[tm.size() / 2], t_f = tf[tf.size() / 2];
    if (all_routed)
        std::printf("  (MMQ's products alone, no gathers: %.3f ms - fused %.2fx)\n", td[td.size() / 2],
                    td[td.size() / 2] / t_f);
    std::vector<float> tf2;   // back to back
    for (int rep = 0; rep < 8; ++rep) tf2.push_back(once(fused_run));
    std::sort(tf2.begin(), tf2.end());
    std::printf("  (fused back to back: median %.3f ms, min %.3f; alternating min %.3f; MMQ min %.3f)\n", tf2[4],
                tf2[0], tf[0], tm[0]);
    const std::vector<float> y_mmq = mq ? download(mb.dm, (size_t) rows * N) : std::vector<float>((size_t) rows * N, 0.0f),
                             y_f = download(fb.dm, (size_t) rows * N);
    const std::vector<int32_t> slot = download_i(fb.slot, (size_t) rows);
    {   // (Aurora S23) a hash of the fused output's bits, to compare two builds bit for bit
        uint64_t hsh = 1469598103934665603ull;
        for (size_t i = 0; i < (size_t) rows; ++i)   // in pair order: the rows of an expert are placed in any order
            for (size_t c = 0; c < (size_t) N; ++c) { uint32_t u; std::memcpy(&u, &y_f[(size_t) slot[i] * N + c], 4); hsh = (hsh ^ u) * 1099511628211ull; }
        std::printf("  fused output bits hash %016llx\n", (unsigned long long) hsh);
    }
    const Err efm = mq ? compare(y_f, [&](size_t i) { return (int64_t) slot[i]; }, y_mmq,
                                 [&](size_t i) { return (int64_t) r.row_of[i]; }, (size_t) rows) : Err{};
    std::printf("%s - timing part: %d tokens x top %d over %d experts, one layer: MMQ path %.3f ms, fused %.3f ms "
                "(%.2fx); fused vs MMQ rel RMS %.3e worst row %.3e\n",
                p.name, T, K, E, t_mmq, t_f, t_mmq / t_f, efm.rms, efm.worst);
    {   // the grouping: every pair in its expert's rows (the near-identical blobs here would hide a wrong expert)
        const std::vector<int32_t> fsrc = download_i(fb.src, (size_t) rows);
        int64_t bad = 0, bad_src = 0;
        for (size_t i = 0; i < (size_t) rows; ++i) {
            const int e = ids[i];
            bad += slot[i] < r.off[(size_t) e] || slot[i] >= r.off[(size_t) e + 1];
            bad_src += fsrc[(size_t) slot[i]] != (int32_t) (i / K);
        }
        if (bad || bad_src) {
            std::vector<int32_t> tbl((size_t) 4 * E + 2);
            ck(cudaMemcpy(tbl.data(), fb.scratch.p, tbl.size() * 4, cudaMemcpyDeviceToHost), "tables");
            int64_t bad_cnt = 0, bad_off = 0, bad_ids = 0;
            const std::vector<int32_t> d_ids = download_i(fb.ids, (size_t) rows);
            for (size_t i = 0; i < (size_t) rows; ++i) bad_ids += d_ids[i] != ids[i];
            std::printf("  the routing ids on the device: %lld of %lld differ (first %d %d %d)\n", (long long) bad_ids,
                        (long long) rows, d_ids[0], d_ids[1], d_ids[2]);
            const int32_t smin = *std::min_element(slot.begin(), slot.end()), smax = *std::max_element(slot.begin(), slot.end());
            std::printf("  slots %d..%d; off[0..3] %d %d %d %d off[E] %d; fill[0..3] %d %d %d %d; ts[0..3] %d %d %d %d "
                        "ts[E] %d\n", smin, smax, tbl[E], tbl[E + 1], tbl[E + 2], tbl[E + 3], tbl[2 * E], tbl[2 * E + 1],
                        tbl[2 * E + 2], tbl[2 * E + 3], tbl[2 * E + 4], tbl[3 * E + 1], tbl[3 * E + 2], tbl[3 * E + 3],
                        tbl[3 * E + 4], tbl[4 * E + 1]);
            // the group again on these ids, synchronously
            fused::group(fb.ids.as<int32_t>(), rows, K, E, fb.scratch.p, fb.slot.as<int32_t>(), fb.src.as<int32_t>(), s);
            ck(cudaStreamSynchronize(s), "sync");
            std::vector<int32_t> t2((size_t) 4 * E + 2);
            ck(cudaMemcpy(t2.data(), fb.scratch.p, t2.size() * 4, cudaMemcpyDeviceToHost), "tables");
            std::printf("  group() again, synchronized: cnt[0..3] %d %d %d %d off[E] %d\n", t2[0], t2[1], t2[2], t2[3],
                        t2[(size_t) 2 * E]);
            for (int e = 0; e < E; ++e) {
                bad_cnt += tbl[(size_t) e] != r.cnt[(size_t) e];
                bad_off += tbl[(size_t) E + e] != r.off[(size_t) e];
            }
            std::printf("  GROUPING WRONG: %lld pairs outside their expert's rows, %lld src mismatches; tables: %lld counts "
                        "and %lld offsets differ from the host's (cnt[0..3] %d %d %d %d vs %d %d %d %d)\n",
                        (long long) bad, (long long) bad_src, (long long) bad_cnt, (long long) bad_off, tbl[0], tbl[1],
                        tbl[2], tbl[3], r.cnt[0], r.cnt[1], r.cnt[2], r.cnt[3]);
            throw std::runtime_error(std::string(p.name) + ": the grouping is wrong");
        }
    }
    if (std::getenv("S20_ROWDIAG")) {   // the pairs whose rows differ most: expert, its rows, the row's place
        const std::vector<int32_t> fsrc = download_i(fb.src, (size_t) rows);
        std::vector<std::pair<double, size_t>> worst;
        for (size_t i = 0; i < (size_t) rows; ++i) {
            const float* a = y_f.data() + (size_t) slot[i] * N;
            const float* m = y_mmq.data() + (size_t) r.row_of[i] * N;
            double me = 0, mr = 0;
            for (int o = 0; o < N; ++o) { me = std::max(me, (double) std::fabs(a[o] - m[o])); mr = std::max(mr, (double) std::fabs(m[o])); }
            worst.push_back({mr > 0 ? me / mr : me, i});
        }
        std::sort(worst.rbegin(), worst.rend());
        for (int k = 0; k < 12; ++k) {
            const size_t i = worst[(size_t) k].second;
            const int e = ids[i];
            // the expert's rows start at its first fused row: the smallest slot of its pairs
            int32_t first = INT32_MAX;
            for (size_t j = 0; j < (size_t) rows; ++j) if (ids[j] == e) first = std::min(first, slot[j]);
            std::printf("    pair %zu token %zu expert %d (%d rows) fused row %d (+%d in the expert) max rel %.3e\n", i,
                        i / K, e, r.cnt[(size_t) e], slot[i], slot[i] - first, worst[(size_t) k].first);
        }
        (void) fsrc;
    }
    // both against the double-precision reference on a sample of 96 pairs
    {
        std::vector<size_t> pick;
        for (size_t i = 0; i < 96; ++i) pick.push_back((i * 2654435761u) % (size_t) rows);
        std::vector<float> ref(pick.size() * N), ym(pick.size() * N), yf(pick.size() * N);
        std::vector<uint8_t> hb(geo.bytes);
        std::vector<float> gu, dn, h(FF);
        for (size_t j = 0; j < pick.size(); ++j) {
            const size_t i = pick[j];
            const int e = ids[i];
            ck(cudaMemcpy(hb.data(), blob[(size_t) e], geo.bytes, cudaMemcpyDeviceToHost), "blob back");
            dequant(p, geo, hb, gu, dn);
            const float* xr = x.data() + (i / K) * N;
            for (int f = 0; f < FF; ++f) {
                double gt = 0, up = 0;
                const float *wg = gu.data() + (size_t) f * N, *wu = gu.data() + (size_t) (FF + f) * N;
                for (int k = 0; k < N; ++k) { gt += (double) wg[k] * xr[k]; up += (double) wu[k] * xr[k]; }
                h[(size_t) f] = (float) (gt / (1.0 + std::exp(-gt)) * up);
            }
            for (int o = 0; o < N; ++o) {
                double a = 0;
                const float* wd = dn.data() + (size_t) o * FF;
                for (int f = 0; f < FF; ++f) a += (double) wd[f] * h[(size_t) f];
                ref[j * N + o] = (float) a;
            }
            if (mq) std::copy_n(y_mmq.begin() + (ptrdiff_t) ((size_t) r.row_of[i] * N), N, ym.begin() + (ptrdiff_t) (j * N));
            std::copy_n(y_f.begin() + (ptrdiff_t) ((size_t) slot[i] * N), N, yf.begin() + (ptrdiff_t) (j * N));
        }
        auto id = [](size_t j) { return (int64_t) j; };
        const Err em = mq ? compare(ym, id, ref, id, pick.size()) : Err{}, ef = compare(yf, id, ref, id, pick.size());
        std::printf("  sample of %zu pairs vs FP32 reference: MMQ rel RMS %.3e worst %.3e, fused rel RMS %.3e worst "
                    "%.3e\n", pick.size(), em.rms, em.worst, ef.rms, ef.worst);
        if (!mq ? (ef.rms > 0.03 || ef.worst > 0.25) : (ef.rms > 1.5 * em.rms || ef.worst > 2.0 * em.worst))
            throw std::runtime_error(std::string(p.name) + ": the fused path's error is not comparable to MMQ's (timing)");
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    delete mbp;
    // the two int8 paths at the real shape, every tile of the layer: R9700 rel RMS 1.3-1.9e-4, worst row 2.9e-3-8.3e-3
    // (STRATA_PF_F16=1: F16 products against MMQ's int8, gfx1151 rel RMS 2.5-2.7e-3, worst row 6.4e-3-8.8e-3)
    if (mq && !no_mmq && (efm.rms > (f16_on() ? 5e-3 : 1e-3) || efm.worst > 2e-2))
        throw std::runtime_error(std::string(p.name) + ": the paths disagree at the real shape");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        // --unset-pf-fused: STRATA_PF_FUSED unset, as the engine's default (the ctest entry for the opt-in STRATA_PF_FUSED_Q2N=1,
        // whose Q2_0 / Q2_0 pair the engine takes with the old variable unset or at 1)
        bool unset_pf = false;
        for (int i = 1; i < argc; ++i)
            if (std::strcmp(argv[i], "--unset-pf-fused") == 0) unset_pf = true;
#ifdef _WIN32
        if (unset_pf) _putenv_s("STRATA_PF_FUSED", ""); else _putenv_s("STRATA_PF_FUSED", "1");   // (an empty value removes it)
#else
        if (unset_pf) unsetenv("STRATA_PF_FUSED"); else setenv("STRATA_PF_FUSED", "1", 1);
        setenv("STRATA_PF_FUSED_KQ", "1", 1);   // (the HIP build's Q4_K / Q5_K / Q5_1 / Q8_0 pairs)
#endif
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) { std::printf("no CUDA device: skipped\n"); return 77; }
        // gfx12: only the native kernels exist (fused::available() is the Q2_0 pack's), so they decide there
        if (!fused::available() && !fused::native_supported(GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL)) {
            std::printf("the fused kernels need sm_80 or newer (or gfx11 / gfx12 matrix cores): skipped\n");
            return 77;
        }
        // with STRATA_PF_FUSED unset nothing is set up below sm_80 (Turing's pair is STRATA_PF_FUSED=1 STRATA_PF_FUSED_SM75=1's)
        if (unset_pf) {
            int major = 0;
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
            if (strata::cc_major_of(major) < 8) { std::printf("STRATA_PF_FUSED unset: nothing is set up below sm_80 (skipped)\n"); return 77; }
        }
        // the opt-in kernel sets (STRATA_PF_FUSED_NT=256, _F16=1, _WS=1): a card that cannot set them up (shared memory,
        // occupancy) falls back to the 512-thread kernels, and a run of those would pass for the wrong kernels - skip it
        {
            const char* nt = std::getenv("STRATA_PF_FUSED_NT");
            const char* f16 = std::getenv("STRATA_PF_FUSED_F16");
            const char* ws = std::getenv("STRATA_PF_FUSED_WS");
            const bool asked = (nt && std::atoi(nt) == 256) || (f16 && f16[0] == '1') || (ws && ws[0] == '1');
            if (asked && !fused::native_wide()) {
                std::printf("STRATA_PF_FUSED_NT / _F16 / _WS: those kernels are not set up on this device (skipped)\n");
                return 77;
            }
        }
        if (!mmq::built()) { std::printf("no MMQ in this build\n"); return 1; }
        const std::vector<Pair> pairs = {
            {GGML_TYPE_IQ2_S, GGML_TYPE_Q2_0, "IQ2_S / Q2_0"},       {GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_0, "IQ2_XXS / Q2_0"},
            {GGML_TYPE_IQ2_XS, GGML_TYPE_IQ4_NL, "IQ2_XS / IQ4_NL"}, {GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL, "IQ3_XXS / IQ4_NL"},
            {GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, "IQ3_S / IQ4_NL"},   {GGML_TYPE_IQ4_XS, GGML_TYPE_Q2_0, "IQ4_XS / Q2_0"},
            // (HIP only) UD-Q4_K_XL's: 47 layers Q4_K / Q5_1, 4 Q4_K / Q8_0, and the one Q5_K layer
            {GGML_TYPE_Q4_K, GGML_TYPE_Q5_1, "Q4_K / Q5_1"},         {GGML_TYPE_Q4_K, GGML_TYPE_Q8_0, "Q4_K / Q8_0"},
            {GGML_TYPE_Q5_K, GGML_TYPE_Q5_1, "Q5_K / Q5_1"},         {GGML_TYPE_Q5_K, GGML_TYPE_Q8_0, "Q5_K / Q8_0"},
            // the Coder IQ1_M pack's other pairs (appended: the default timing set takes pairs 0, 1, 4, 3 by index)
            {GGML_TYPE_IQ2_S, GGML_TYPE_IQ4_NL, "IQ2_S / IQ4_NL"},   {GGML_TYPE_IQ3_XXS, GGML_TYPE_Q2_0, "IQ3_XXS / Q2_0"},
            {GGML_TYPE_IQ3_S, GGML_TYPE_Q2_0, "IQ3_S / Q2_0"},       {GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL, "IQ4_XS / IQ4_NL"},
            // the canonical Q2_0 pack (gate/up and down Q2_0; CUDA)
            {GGML_TYPE_Q2_0, GGML_TYPE_Q2_0, "Q2_0 / Q2_0"}};
        const char* only = nullptr;   // --only=NAME: the pairs whose name contains NAME
        for (int i = 1; i < argc; ++i)
            if (std::strncmp(argv[i], "--only=", 7) == 0) only = argv[i] + 7;
        bool timing = true, ref = true;   // --no-timing, --no-ref: skip a part
        std::vector<int> chunks = {2048, 3584, 8192};   // --chunks=A,B: the timing part's chunk sizes
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--no-timing") == 0) timing = false;
            if (std::strcmp(argv[i], "--no-ref") == 0) ref = false;
            if (std::strncmp(argv[i], "--chunks=", 9) == 0) {
                chunks.clear();
                for (const char* c = argv[i] + 9; *c;) {
                    chunks.push_back(std::atoi(c));
                    while (*c && *c != ',') ++c;
                    if (*c == ',') ++c;
                }
            }
        }
        cudaStream_t s = nullptr;
        ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        for (const Pair& p : pairs) {
            if (only ? std::string(p.name).find(only) == std::string::npos : p.gu == GGML_TYPE_Q4_K || p.gu == GGML_TYPE_Q5_K) continue;
            if (!fused::native_supported((int) p.gu, (int) p.d)) {
                // the Q2_0 / Q2_0 pair is the Turing opt-in's (STRATA_PF_FUSED_SM75=1; the only pair set up there) or, on sm_80+,
                // STRATA_PF_FUSED_Q2N=1's: skip what this device does not take (--only=NAME still insists)
                const bool turing_pair = p.gu == GGML_TYPE_Q2_0 && p.d == GGML_TYPE_Q2_0;
                int major = 0;
                cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
                if ((major < 8 || turing_pair) && !only) { std::printf("%s: not set up on this device (skipped)\n", p.name); continue; }
                throw std::runtime_error(std::string(p.name) + ": not covered by the native kernels");
            }
            if (ref) reference_part(p, s);
        }
        if (timing) {
            // the IQ2_XS pack's layers (34 of 48 IQ2_S / Q2_0, 11 IQ2_XXS / Q2_0), the IQ3_S pack's most common, IQ3_XXS's
            for (const Pair& p : only ? pairs : std::vector<Pair>{pairs[0], pairs[1], pairs[4], pairs[3]}) {
                if (only && std::string(p.name).find(only) == std::string::npos) continue;
                if (!fused::native_supported((int) p.gu, (int) p.d)) continue;   // (Turing: the Q2_0 pair only)
                for (int T : chunks) timing_part(p, T, s);
            }
        }
        ck(cudaStreamDestroy(s), "destroy");
        std::printf("prefill fused MoE (native formats) parity passed\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "prefill fused MoE (native formats) parity failed: %s\n", e.what());
        return 1;
    }
}
