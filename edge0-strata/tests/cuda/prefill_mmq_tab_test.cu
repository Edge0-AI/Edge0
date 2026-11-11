// prefill_mmq_tab_test - STRATA_MMQ_DIRECT: the pointer-table MMQ (src/prefill/moe_mmq_tab.cuh, Product::tab) bit for
// bit against the gathered MMQ the prompt path runs today (experts copied `expert_bytes` apart, a zeroed 4096-byte
// tail after the last one).
//
// Part 1, per product type (IQ3_S gate/up [1280, 2560], IQ4_NL and Q8_0 down [2560, 640], and every other type with a
// pointer-table instance, at its pack shape): random valid expert matrices at scattered 256-aligned addresses, each
// followed by a guard filled with an fp16 NaN pattern (0xFF 0x7C); prep_group zeroes the over-read bytes, then the
// table launch; memcmp of the whole output (pre-filled with a NaN pattern in both runs) against the gathered launch.
// Sweep: 1..16 experts with mixed row counts {0,1,7,8,9,31,64,100,512,1000}; max_rows from 1 to 1000 so that every
// tile width J from 8 to 128 is taken (and max_rows above the largest count); few-tile launches (stream-k grid of
// nsm blocks with the fixup kernel) and many-tile ones (one block per tile).
// Part 2, negative control: the same table launch with the guards NOT zeroed must give non-finite values in a down
// product (proves the guard handling is load-bearing; skipped for types that read nothing past a matrix).
// Part 3, mixed groups as prefill.cpp's walk builds them: native expert blobs (gate|up|down) for IQ3_S + IQ4_NL and
// IQ3_S + Q8_0, some read in place (direct), the others gathered at compact positions c != q - one at a time
// (gather_native) or by gather_native_group with a mid-group flush (first > 0) - including groups whose last entry is
// direct and groups with no gathered entry; both products bitwise equal to all-gathered.
// Part 4, STRATA_MMQ_GROUP_SORT: a layer's experts in groups of 16 as prefill.cpp runs them (activations quantized
// in group order through the src map, absolute bounds, max_rows = the group's largest count), once in ID order and
// once with the full groups in row-count order (mmq_resident_sort_rows, the partial group kept): every token's
// product row bitwise equal, when mmq::whole_tiles says so for this card (else only reported). Sorting the partial
// group too (STRATA_MMQ_RESIDENT_SORT_NE) is reported, not required.
// Exit 77 without a CUDA device of sm_80 or newer or without the pointer-table build.
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include "../../src/prefill/mmq_resident_sort.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace mmq = strata::prefill::mmq;

constexpr int G = mmq::kGatherGroupMax;
constexpr size_t TAIL = 4096;
constexpr int64_t kIdsPad = 2048 + 128;   // > the largest max_rows of the sweep plus one tile

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Dev {
    void* p = nullptr;
    size_t n = 0;
    explicit Dev(size_t bytes) : n(bytes) { ck(cudaMalloc(&p, bytes), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    uint8_t* b() const { return (uint8_t*) p; }
};

// random blocks with a sane fp16 scale at each block's start (every format here begins with it)
void fill_matrix(uint8_t* m, ggml_type t, int64_t rows, int64_t cols, std::mt19937& rng) {
    const size_t bs = ggml_type_size(t), nb = (size_t) rows * (size_t) (cols / ggml_blck_size(t));
    std::uniform_int_distribution<int> byte(0, 255);
    const float lo = t == GGML_TYPE_Q8_0 ? 0.0001f : t == GGML_TYPE_IQ3_S ? 0.0004f : t == GGML_TYPE_IQ4_XS ? 0.00005f
                   : t == GGML_TYPE_IQ4_NL ? 0.0002f : t == GGML_TYPE_Q2_0 ? 0.004f : 0.0008f;
    std::uniform_real_distribution<float> sc(lo, 6.0f * lo);
    for (size_t i = 0; i < nb * bs; ++i) m[i] = (uint8_t) byte(rng);
    for (size_t b = 0; b < nb; ++b) {
        const ggml_fp16_t h = ggml_fp32_to_fp16(sc(rng));
        std::memcpy(m + b * bs, &h, 2);
    }
}

// host -> device, finished before any work on the (non-blocking) test stream starts: a pageable cudaMemcpy returns
// before its DMA lands, and the legacy stream does not order it against a cudaStreamNonBlocking stream
void upload(void* dst, const void* src, size_t n, const char* what) {
    ck(cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice), what);
    ck(cudaDeviceSynchronize(), what);
}

// the fp16 NaN pattern a guard holds before prep_group (0x7CFF little-endian at every even offset)
void fill_nan(uint8_t* dev, size_t n) {
    std::vector<uint8_t> h(n);
    for (size_t i = 0; i < n; ++i) h[i] = (i & 1) ? 0x7C : 0xFF;
    upload(dev, h.data(), n, "fill_nan");
}

struct Shape {
    ggml_type t;
    int64_t rows, cols;   // the weight matrix [rows, cols]
    const char* role;
};

// activations for `total` rows of `cols`, quantized for type t (q8_1), plus an identity map and the bounds
struct Acts {
    Dev x, xq, ids, bounds;
    int64_t total;
    Acts(ggml_type t, int64_t cols, const std::vector<int>& cnt, std::mt19937& rng, cudaStream_t s)
        : x((size_t) std::max<int64_t>(1, sum(cnt)) * cols * 4), xq(mmq::q8_bytes(std::max<int64_t>(1, sum(cnt)), cols)),
          ids((size_t) (sum(cnt) + kIdsPad) * 4), bounds((cnt.size() + 1) * 4), total(sum(cnt)) {
        std::vector<float> hx((size_t) std::max<int64_t>(1, total) * cols);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto& v : hx) v = nd(rng);
        upload(x.p, hx.data(), hx.size() * 4, "x");
        std::vector<int32_t> hb(cnt.size() + 1, 0);
        for (size_t i = 0; i < cnt.size(); ++i) hb[i + 1] = hb[i] + cnt[i];
        upload(bounds.p, hb.data(), hb.size() * 4, "bounds");
        // MMQ loads a whole tile's J ids from an expert's first row, and its stream-k fixup does so for every column
        // tile of max_rows, also past the last row (upstream behaviour, harmless values): padded for memcheck
        mmq::iota((int32_t*) ids.p, total + kIdsPad, s);
        if (total > 0) mmq::quantize((const float*) x.p, nullptr, xq.p, (int) t, cols, cols, total, s);
        ck(cudaStreamSynchronize(s), "acts");
    }
    static int64_t sum(const std::vector<int>& c) {
        int64_t n = 0;
        for (int v : c) n += v;
        return n;
    }
};

mmq::Product product(const Shape& sh, const Acts& a, int n, int64_t max_rows, float* dst) {
    mmq::Product p;
    p.type = (int) sh.t; p.w_rows = sh.rows; p.w_cols = sh.cols;
    p.n = n; p.xq = a.xq.p; p.bounds = (const int32_t*) a.bounds.p; p.ids = (const int32_t*) a.ids.p;
    p.total_rows = a.total; p.max_rows = max_rows; p.dst = dst; p.ld_dst = sh.rows;
    return p;
}

int g_cases = 0, g_fail = 0;

bool same_bits(const Dev& a, const Dev& b, size_t n, const char* what) {
    std::vector<uint8_t> ha(n), hb(n);
    ck(cudaMemcpy(ha.data(), a.p, n, cudaMemcpyDeviceToHost), "dtoh a");
    ck(cudaMemcpy(hb.data(), b.p, n, cudaMemcpyDeviceToHost), "dtoh b");
    ++g_cases;
    if (std::memcmp(ha.data(), hb.data(), n) == 0) return true;
    size_t i = 0;
    while (i < n && ha[i] == hb[i]) ++i;
    std::printf("  FAIL %s: first differing byte %zu of %zu\n", what, i, n);
    if (std::getenv("TAB_TEST_DEBUG")) {
        const float* fa = (const float*) ha.data();
        const float* fb = (const float*) hb.data();
        size_t nd = 0;
        for (size_t k = 0; k < n / 4; ++k)
            if (std::memcmp(fa + k, fb + k, 4) != 0) {
                if (nd < 6) std::printf("    float %zu: %.9g vs %.9g\n", k, fa[k], fb[k]);
                ++nd;
            }
        std::printf("    %zu floats differ\n", nd);
    }
    ++g_fail;
    return false;
}

bool any_nonfinite(const Dev& d, size_t floats) {
    std::vector<float> h(floats);
    ck(cudaMemcpy(h.data(), d.p, floats * 4, cudaMemcpyDeviceToHost), "dtoh");
    for (float v : h)
        if (!std::isfinite(v)) return true;
    return false;
}

// Part 1 + 2: one product, `cnt` rows per expert
void product_case(mmq::Context& ctx, const Shape& sh, const std::vector<int>& cnt, int64_t max_rows, std::mt19937& rng,
                  cudaStream_t s, bool negative, const char* label) {
    const int n = (int) cnt.size();
    const size_t eb = mmq::matrix_bytes((int) sh.t, sh.rows, sh.cols);
    const size_t over = mmq::overread_bytes((int) sh.t, sh.cols);
    Acts a(sh.t, sh.cols, cnt, rng, s);
    // the experts' matrices on the host
    std::vector<std::vector<uint8_t>> w((size_t) n, std::vector<uint8_t>(eb));
    for (auto& m : w) fill_matrix(m.data(), sh.t, sh.rows, sh.cols, rng);
    // gathered: back to back, a zeroed tail, the space after it NaN-patterned
    Dev grp((size_t) n * eb + TAIL + 4096);
    fill_nan(grp.b(), grp.n);
    for (int z = 0; z < n; ++z) upload(grp.b() + (size_t) z * eb, w[(size_t) z].data(), eb, "grp");
    ck(cudaMemset(grp.b() + (size_t) n * eb, 0, TAIL), "tail");
    ck(cudaDeviceSynchronize(), "tail");
    // in place: scattered 256-aligned slots in one arena (shuffled order, random gaps), each with a NaN guard
    const size_t stride = (eb + mmq::kSlotGuard + 255) / 256 * 256;
    std::vector<size_t> pos((size_t) n);
    for (int z = 0; z < n; ++z) pos[(size_t) z] = (size_t) z;
    std::shuffle(pos.begin(), pos.end(), rng);
    std::uniform_int_distribution<int> gap(0, 8);
    Dev arena((size_t) n * (stride + 8 * 256) + 4096);
    fill_nan(arena.b(), arena.n);
    mmq::WeightTab tab;
    mmq::GuardTab guards;
    size_t at = 0;
    std::vector<size_t> where((size_t) n);
    for (int k = 0; k < n; ++k) {
        at += (size_t) gap(rng) * 256;
        where[pos[(size_t) k]] = at;
        at += stride;
    }
    for (int z = 0; z < n; ++z) {
        uint8_t* p = arena.b() + where[(size_t) z];
        upload(p, w[(size_t) z].data(), eb, "arena");
        tab.p[z] = p;
        guards.p[guards.n++] = p + eb;
    }
    const size_t out = (size_t) std::max<int64_t>(1, a.total) * (size_t) sh.rows * 4;
    Dev ref(out), got(out);
    fill_nan(ref.b(), out);
    fill_nan(got.b(), out);
    mmq::Product pr = product(sh, a, n, max_rows, (float*) ref.p);
    pr.w = grp.p; pr.expert_bytes = eb;
    ctx.run(pr, s);
    if (!negative) mmq::prep_group(guards, over, nullptr, nullptr, 0, s);
    mmq::Product pt = product(sh, a, n, max_rows, (float*) got.p);
    pt.w = nullptr; pt.expert_bytes = 0; pt.tab = &tab;
    ctx.run(pt, s);
    ck(cudaStreamSynchronize(s), "run");
    char what[256];
    std::snprintf(what, sizeof what, "%s %s n=%d rows=%lld max_rows=%lld (%s)", ggml_type_name(sh.t), sh.role, n,
                  (long long) a.total, (long long) max_rows, label);
    if (!negative) {
        same_bits(ref, got, out, what);
        return;
    }
    ++g_cases;
    if (!any_nonfinite(got, out / 4)) {
        std::printf("  FAIL negative control %s: no non-finite value with the guards left as NaN\n", what);
        ++g_fail;
    } else if (any_nonfinite(ref, (size_t) a.total * (size_t) sh.rows)) {
        std::printf("  FAIL negative control %s: the gathered reference is not finite\n", what);
        ++g_fail;
    }
}

std::vector<int> random_counts(int n, std::mt19937& rng) {
    static const int kRows[] = {0, 1, 7, 8, 9, 31, 64, 100, 512, 1000};
    std::uniform_int_distribution<int> pick(0, 9);
    std::vector<int> c((size_t) n);
    for (auto& v : c) v = kRows[pick(rng)];
    return c;
}
int64_t max_of(const std::vector<int>& c) {
    int64_t m = 0;
    for (int v : c) m = std::max<int64_t>(m, v);
    return m;
}

void sweep(mmq::Context& ctx, const Shape& sh, bool full, std::mt19937& rng, cudaStream_t s) {
    const int f0 = g_fail, c0 = g_cases;
    // 1..16 experts, mixed counts (max_rows = the largest count, as prefill.cpp passes it)
    for (int n = 1; n <= G; ++n) {
        if (!full && n != 1 && n != 3 && n != 8 && n != G) continue;
        for (int rep = 0; rep < (full ? 3 : 1); ++rep) {
            std::vector<int> c = random_counts(n, rng);
            if (max_of(c) == 0) c[0] = 1;
            product_case(ctx, sh, c, max_of(c), rng, s, false, "mixed");
        }
    }
    // every J: one expert with m rows, the others fewer; and max_rows above the largest count
    static const int kMax[] = {1, 7, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 56, 64, 65, 72, 80, 88, 96, 100, 104,
                               112, 120, 127, 128, 129, 300, 1000};
    for (int m : kMax) {
        if (!full && m % 8 != 0 && m != 1 && m != 129) continue;
        for (int n : {1, 5, 16}) {
            if (!full && n == 5) continue;
            std::uniform_int_distribution<int> fewer(0, m);
            std::vector<int> c((size_t) n);
            for (auto& v : c) v = fewer(rng);
            c[(size_t) (rng() % (unsigned) n)] = m;
            product_case(ctx, sh, c, m, rng, s, false, "J sweep");
        }
    }
    for (int n : {1, 4, 16}) {
        std::vector<int> c = random_counts(n, rng);
        if (max_of(c) == 0) c[0] = 9;
        product_case(ctx, sh, c, max_of(c) + 37, rng, s, false, "max_rows above the largest count");
    }
    // experts with 0 rows at the ends and in the middle
    product_case(ctx, sh, {0, 0, 5, 0, 1000, 0, 0}, 1000, rng, s, false, "empty experts");
    product_case(ctx, sh, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}, 1, rng, s, false, "only the last expert");
    std::printf("%-8s %-7s %4d cases, %s\n", ggml_type_name(sh.t), sh.role, g_cases - c0,
                g_fail == f0 ? "all bitwise identical" : "FAILURES");
    // negative control: a down product that reads past its matrix, guards not zeroed
    if (mmq::overread_bytes((int) sh.t, sh.cols) > 0) {
        const int f1 = g_fail;
        product_case(ctx, sh, {3, 64, 0, 17}, 64, rng, s, true, "negative");
        product_case(ctx, sh, {200}, 200, rng, s, true, "negative");
        std::printf("%-8s %-7s negative control (guards not zeroed): %s\n", ggml_type_name(sh.t), sh.role,
                    g_fail == f1 ? "non-finite output, as required" : "FAILED");
    }
}

// Part 3: groups as prefill.cpp's walk builds them, both products
struct Pair {
    ggml_type gu, d;
};
void group_case(mmq::Context& ctx, const Pair& pr, const std::vector<int>& cnt, const std::vector<char>& direct,
                bool group_gather, int flush_at, std::mt19937& rng, cudaStream_t s, const char* label) {
    const int n = (int) cnt.size();
    const Shape gs{pr.gu, 1280, 2560, "gate/up"}, ds{pr.d, 2560, 640, "down"};
    const size_t gub = mmq::matrix_bytes((int) pr.gu, 1280, 2560), db = mmq::matrix_bytes((int) pr.d, 2560, 640);
    const size_t up_off = gub / 2, down_off = gub, bytes = gub + db;
    const size_t over = mmq::overread_bytes((int) pr.d, 640);
    // native blobs, each in its own place: direct ones in "cache slots" with a guard, gathered ones elsewhere
    std::vector<std::vector<uint8_t>> blob((size_t) n, std::vector<uint8_t>(bytes));
    for (auto& b : blob) {
        fill_matrix(b.data(), pr.gu, 1280, 2560, rng);
        fill_matrix(b.data() + down_off, pr.d, 2560, 640, rng);
    }
    const size_t stride = (bytes + mmq::kSlotGuard + 255) / 256 * 256;
    Dev arena((size_t) n * (stride + 4 * 256) + 4096);
    fill_nan(arena.b(), arena.n);
    std::vector<uint8_t*> at((size_t) n);
    size_t o = 0;
    for (int z = 0; z < n; ++z) {
        o += (size_t) (rng() % 4) * 256;
        at[(size_t) z] = arena.b() + o;
        o += stride;
        upload(at[(size_t) z], blob[(size_t) z].data(), bytes, "blob");
    }
    Acts ag(pr.gu, 2560, cnt, rng, s), ad(pr.d, 640, cnt, rng, s);
    const int64_t mr = std::max<int64_t>(1, max_of(cnt));
    const size_t out_g = (size_t) std::max<int64_t>(1, ag.total) * 1280 * 4, out_d = (size_t) std::max<int64_t>(1, ad.total) * 2560 * 4;
    // reference: today's walk (gather at q, two tail memsets)
    Dev rgu((size_t) G * gub + TAIL), rd((size_t) G * db + TAIL), ref_g(out_g), ref_d(out_d);
    fill_nan(rgu.b(), rgu.n);
    fill_nan(rd.b(), rd.n);
    fill_nan(ref_g.b(), out_g);
    fill_nan(ref_d.b(), out_d);
    for (int q = 0; q < n; ++q) {
        const uint8_t* b = at[(size_t) q];
        mmq::gather_native(b, b + up_off, gub / 2, b + down_off, db, rgu.b() + (size_t) q * gub, rd.b() + (size_t) q * db, s);
    }
    ck(cudaMemsetAsync(rgu.b() + (size_t) n * gub, 0, TAIL, s), "memset");
    ck(cudaMemsetAsync(rd.b() + (size_t) n * db, 0, TAIL, s), "memset");
    mmq::Product p = product(gs, ag, n, mr, (float*) ref_g.p);
    p.w = rgu.p; p.expert_bytes = gub;
    ctx.run(p, s);
    p = product(ds, ad, n, mr, (float*) ref_d.p);
    p.w = rd.p; p.expert_bytes = db;
    ctx.run(p, s);
    // direct: prefill.cpp's compute() in direct mode
    Dev tgu_buf((size_t) G * gub + TAIL), td_buf((size_t) G * db + TAIL), got_g(out_g), got_d(out_d);
    fill_nan(tgu_buf.b(), tgu_buf.n);
    fill_nan(td_buf.b(), td_buf.n);
    fill_nan(got_g.b(), out_g);
    fill_nan(got_d.b(), out_d);
    mmq::WeightTab tgu, td;
    mmq::GuardTab gd;
    mmq::GatherGroup gg;
    int ns = 0;
    auto flush = [&]() {
        if (gg.n <= gg.first) return;
        if (!mmq::gather_native_group(gg, up_off, gub / 2, down_off, db, tgu_buf.p, gub, td_buf.p, db, s))
            throw std::runtime_error("gather_native_group refused an aligned group");
        gg.first = gg.n;
    };
    for (int q = 0; q < n; ++q) {
        uint8_t* b = at[(size_t) q];
        if (direct[(size_t) q]) {
            tgu.p[q] = b;
            td.p[q] = b + down_off;
            gd.p[gd.n++] = b + bytes;
        } else {
            const int c = ns++;
            tgu.p[q] = tgu_buf.b() + (size_t) c * gub;
            td.p[q] = td_buf.b() + (size_t) c * db;
            if (group_gather) {
                gg.blob[c] = b;
                gg.n = c + 1;
            } else {
                mmq::gather_native(b, b + up_off, gub / 2, b + down_off, db, tgu_buf.b() + (size_t) c * gub,
                                   td_buf.b() + (size_t) c * db, s);
            }
        }
        if (group_gather && q == flush_at) flush();   // release_to's mid-group flush
    }
    if (group_gather) flush();
    mmq::prep_group(gd, over, ns ? tgu_buf.b() + (size_t) ns * gub : nullptr, ns ? td_buf.b() + (size_t) ns * db : nullptr,
                    TAIL, s);
    p = product(gs, ag, n, mr, (float*) got_g.p);
    p.tab = &tgu;
    ctx.run(p, s);
    p = product(ds, ad, n, mr, (float*) got_d.p);
    p.tab = &td;
    ctx.run(p, s);
    ck(cudaStreamSynchronize(s), "group");
    char what[256];
    std::snprintf(what, sizeof what, "%s+%s group n=%d direct=%d %s (%s)", ggml_type_name(pr.gu), ggml_type_name(pr.d), n,
                  gd.n, group_gather ? "gather_native_group" : "gather_native", label);
    std::string w1 = std::string(what) + " gate/up", w2 = std::string(what) + " down";
    same_bits(ref_g, got_g, out_g, w1.c_str());
    same_bits(ref_d, got_d, out_d, w2.c_str());
}


// Part 4: one layer's experts in groups of G; `span` as mmq_resident_sort_span gives it (0: ID order)
struct Layer {
    std::vector<int32_t> ids, count, off, order, slot, src;
};
Layer route(const std::vector<int>& cnt, std::mt19937& rng) {
    Layer L;
    for (size_t e = 0; e < cnt.size(); ++e)
        for (int r = 0; r < cnt[e]; ++r) L.ids.push_back((int32_t) e);
    std::shuffle(L.ids.begin(), L.ids.end(), rng);   // one route per token (K = 1), experts interleaved
    L.count.assign(cnt.begin(), cnt.end());
    L.off.assign(cnt.size() + 1, 0);
    for (size_t e = 0; e < cnt.size(); ++e) L.off[e + 1] = L.off[e] + cnt[e];
    for (size_t e = 0; e < cnt.size(); ++e)
        if (cnt[e] > 0) L.order.push_back((int32_t) e);
    L.slot.assign(L.ids.size(), 0);
    L.src.assign(L.ids.size(), 0);
    std::vector<int32_t> fill(L.off.begin(), L.off.end() - 1);
    for (size_t i = 0; i < L.ids.size(); ++i) {
        const int32_t p = fill[(size_t) L.ids[i]]++;
        L.slot[i] = p;
        L.src[(size_t) p] = (int32_t) i;
    }
    return L;
}

// the layer's product rows in the layer's row order (row p = token src[p])
std::vector<float> run_layer(mmq::Context& ctx, const Shape& sh, const Layer& L, const std::vector<std::vector<uint8_t>>& w,
                             const Dev& x, cudaStream_t s) {
    const int64_t T = (int64_t) L.ids.size();
    const size_t eb = mmq::matrix_bytes((int) sh.t, sh.rows, sh.cols), n = L.order.size();
    Dev src((size_t) T * 4), xq(mmq::q8_bytes(T, sh.cols)), ids((size_t) (T + kIdsPad) * 4), bounds((n + 1) * 4);
    Dev grp((size_t) G * eb + TAIL), out((size_t) T * sh.rows * 4);
    upload(src.p, L.src.data(), (size_t) T * 4, "src");
    std::vector<int32_t> hb(n + 1);
    for (size_t j = 0; j < n; ++j) hb[j] = L.off[(size_t) L.order[j]];
    hb[n] = (int32_t) T;
    upload(bounds.p, hb.data(), hb.size() * 4, "bounds");
    fill_nan(out.b(), out.n);
    mmq::iota((int32_t*) ids.p, T + kIdsPad, s);
    mmq::quantize((const float*) x.p, (const int32_t*) src.p, xq.p, (int) sh.t, sh.cols, sh.cols, T, s);
    for (size_t j0 = 0; j0 < n; j0 += G) {
        const int ng = (int) std::min<size_t>(G, n - j0);
        int64_t maxr = 0;
        ck(cudaStreamSynchronize(s), "group");   // grp is rewritten below
        for (int q = 0; q < ng; ++q) {
            const int32_t e = L.order[j0 + (size_t) q];
            maxr = std::max<int64_t>(maxr, L.count[(size_t) e]);
            upload(grp.b() + (size_t) q * eb, w[(size_t) e].data(), eb, "grp");
        }
        ck(cudaMemset(grp.b() + (size_t) ng * eb, 0, TAIL), "tail");
        ck(cudaDeviceSynchronize(), "tail");
        mmq::Product p;
        p.w = grp.p; p.type = (int) sh.t; p.w_rows = sh.rows; p.w_cols = sh.cols; p.expert_bytes = eb;
        p.n = ng; p.xq = xq.p; p.bounds = (const int32_t*) bounds.p + j0; p.ids = (const int32_t*) ids.p;
        p.total_rows = T; p.max_rows = maxr; p.dst = (float*) out.p; p.ld_dst = sh.rows;
        ctx.run(p, s);
    }
    ck(cudaStreamSynchronize(s), "layer");
    std::vector<float> h((size_t) T * sh.rows);
    ck(cudaMemcpy(h.data(), out.p, h.size() * 4, cudaMemcpyDeviceToHost), "dtoh layer");
    return h;
}

// tokens whose product row differs between two orders of the same layer
int64_t rows_differing(const Layer& a, const std::vector<float>& ra, const Layer& b, const std::vector<float>& rb, int64_t cols) {
    int64_t nd = 0;
    for (size_t t = 0; t < a.ids.size(); ++t)
        if (std::memcmp(ra.data() + (size_t) a.slot[t] * cols, rb.data() + (size_t) b.slot[t] * cols, (size_t) cols * 4) != 0) ++nd;
    return nd;
}

void order_case(mmq::Context& ctx, const Shape& sh, const std::vector<int>& cnt, bool whole, std::mt19937& rng, cudaStream_t s,
                const char* label) {
    const size_t eb = mmq::matrix_bytes((int) sh.t, sh.rows, sh.cols);
    std::vector<std::vector<uint8_t>> w(cnt.size(), std::vector<uint8_t>(eb));
    for (auto& m : w) fill_matrix(m.data(), sh.t, sh.rows, sh.cols, rng);
    Layer id = route(cnt, rng);
    const int64_t T = (int64_t) id.ids.size();
    Dev x((size_t) T * sh.cols * 4);
    {
        std::vector<float> hx((size_t) T * sh.cols);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto& v : hx) v = nd(rng);
        upload(x.p, hx.data(), hx.size() * 4, "x");
    }
    Layer full = id, all = id;
    namespace d = strata::prefill::detail;
    d::mmq_resident_sort_rows(full.ids.data(), T, 1, full.count, full.off, full.order, full.slot.data(), full.src.data(),
                              d::mmq_resident_sort_span(full.order.size(), G, true));
    d::mmq_resident_sort_rows(all.ids.data(), T, 1, all.count, all.off, all.order, all.slot.data(), all.src.data(),
                              d::mmq_resident_sort_span(all.order.size(), G, false));
    const size_t n = id.order.size(), tail = n % G;
    // the partial group's entries and row range stay where they were
    bool kept = std::equal(id.order.end() - (ptrdiff_t) tail, id.order.end(), full.order.end() - (ptrdiff_t) tail);
    for (size_t j = n - tail; j < n && kept; ++j) kept = full.off[(size_t) full.order[j]] == id.off[(size_t) id.order[j]];
    const std::vector<float> r_id = run_layer(ctx, sh, id, w, x, s);
    const std::vector<float> r_full = run_layer(ctx, sh, full, w, x, s);
    const std::vector<float> r_all = run_layer(ctx, sh, all, w, x, s);
    const int64_t d_full = rows_differing(id, r_id, full, r_full, sh.rows), d_all = rows_differing(id, r_id, all, r_all, sh.rows);
    ++g_cases;
    const bool ok = kept && (!whole || d_full == 0);
    if (!ok) ++g_fail;
    std::printf("  %s %-7s %s: %zu experts (%zu in the partial group), %lld rows: full groups sorted %lld rows differ%s, "
                "all sorted %lld rows differ%s\n", ggml_type_name(sh.t), sh.role, label, n, tail, (long long) T,
                (long long) d_full, ok ? "" : (kept ? " FAIL" : " FAIL (partial group moved)"), (long long) d_all,
                whole ? "" : " (whole_tiles false on this card: reported only)");
}

std::vector<int> skewed_counts(int n, int lo, int hi, std::mt19937& rng) {
    // prompt-like: most experts few rows, some many (log-uniform)
    std::uniform_real_distribution<double> u(std::log((double) lo), std::log((double) hi + 1));
    std::vector<int> c((size_t) n);
    for (auto& v : c) v = std::max(lo, (int) std::exp(u(rng)));
    return c;
}

}  // namespace

int main() {
    try {
        int dev = 0;
        cudaDeviceProp prop{};
        if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess || prop.major < 8) {
            std::printf("needs a CUDA device of sm_80 or newer: skipped\n");
            return 77;
        }
        if (!mmq::built() || !mmq::tab_built()) {
            std::printf("no pointer-table MMQ in this build: skipped\n");
            return 77;
        }
        std::printf("%s (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
        cudaStream_t s;
        ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        mmq::Context ctx;
        std::mt19937 rng(20261011);
        // the pack's three formats, fully; every other type with an instance, a shorter sweep
        const Shape main_shapes[] = {{GGML_TYPE_IQ3_S, 1280, 2560, "gate/up"},
                                     {GGML_TYPE_IQ4_NL, 2560, 640, "down"},
                                     {GGML_TYPE_Q8_0, 2560, 640, "down"}};
        // the types without a pointer table (the optional Q5_0 / K-quant instances) must keep the gather
        for (const ggml_type t : {GGML_TYPE_Q5_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q5_1, GGML_TYPE_Q6_K}) {
            if (mmq::tab_fits((int) t, 1280) || mmq::tab_fits((int) t, 2560)) {
                std::printf("FAIL: %s reports a pointer table (only the nine tested types have one)\n", ggml_type_name(t));
                return 1;
            }
        }
        // the over-read each guard must cover: within kSlotGuard, and on sm_86 the values the guards were sized for
        // (256-value K tiles: 640 columns are 20 blocks of 32, so the last tile reads 4 blocks more - 4 x 18 B of
        // IQ4_NL, 4 x 34 B of Q8_0; 2560 columns end on a tile)
        const size_t want_over[] = {0, 72, 136};
        for (size_t i = 0; i < 3; ++i) {
            const Shape& sh = main_shapes[i];
            const size_t ov = mmq::overread_bytes((int) sh.t, sh.cols);
            if (ov > mmq::kSlotGuard || (prop.major == 8 && prop.minor == 6 && ov != want_over[i])) {
                std::printf("FAIL: %s over-read %zu B (want %zu, at most %zu)\n", ggml_type_name(sh.t), ov, want_over[i],
                            mmq::kSlotGuard);
                return 1;
            }
        }
        for (const Shape& sh : main_shapes) {
            if (!mmq::tab_fits((int) sh.t, sh.rows)) {
                std::printf("FAIL: no pointer-table MMQ for %s\n", ggml_type_name(sh.t));
                return 1;
            }
            std::printf("%-8s %-7s over-read past the matrix: %zu B\n", ggml_type_name(sh.t), sh.role,
                        mmq::overread_bytes((int) sh.t, sh.cols));
            sweep(ctx, sh, true, rng, s);
        }
        const Shape other[] = {{GGML_TYPE_IQ2_XXS, 1280, 2560, "gate/up"}, {GGML_TYPE_IQ2_XS, 1280, 2560, "gate/up"},
                               {GGML_TYPE_IQ2_S, 1280, 2560, "gate/up"},   {GGML_TYPE_IQ3_XXS, 1280, 2560, "gate/up"},
                               {GGML_TYPE_IQ4_XS, 1280, 2560, "gate/up"},  {GGML_TYPE_Q8_0, 1280, 2560, "gate/up"},
                               {GGML_TYPE_IQ4_NL, 1280, 2560, "gate/up"},  {GGML_TYPE_Q2_0, 2560, 640, "down"}};
        for (const Shape& sh : other) {
            if (!mmq::tab_fits((int) sh.t, sh.rows)) {   // all nine types have an instance (CMake's tab types)
                std::printf("FAIL: no pointer-table MMQ for %s %s\n", ggml_type_name(sh.t), sh.role);
                return 1;
            }
            sweep(ctx, sh, false, rng, s);
        }
        // Part 3
        const int f0 = g_fail, c0 = g_cases;
        for (const Pair pr : {Pair{GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL}, Pair{GGML_TYPE_IQ3_S, GGML_TYPE_Q8_0}}) {
            for (int gg = 0; gg < 2; ++gg) {
                const bool group_gather = gg == 1;
                // alternating direct / gathered, last entry direct
                group_case(ctx, pr, {5, 0, 31, 64, 1, 100, 8, 9}, {0, 1, 0, 1, 1, 0, 0, 1}, group_gather, 3, rng, s,
                           "interleaved, last direct");
                // a full group, the first entries gathered, then mostly direct; flush after entry 5
                group_case(ctx, pr, {7, 8, 9, 31, 64, 100, 1, 0, 512, 3, 17, 33, 2, 64, 90, 128},
                           {0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0}, group_gather, 5, rng, s,
                           "full group, last gathered");
                // every entry direct (ns == 0: no tails), and every entry gathered
                group_case(ctx, pr, {12, 40, 0, 7}, {1, 1, 1, 1}, group_gather, -1, rng, s, "all direct");
                group_case(ctx, pr, {12, 40, 0, 7}, {0, 0, 0, 0}, group_gather, 1, rng, s, "all gathered");
                // one entry
                group_case(ctx, pr, {1000}, {1}, group_gather, -1, rng, s, "one direct entry");
                // random mixes
                for (int r = 0; r < 4; ++r) {
                    const int n = 1 + (int) (rng() % G);
                    std::vector<int> c = random_counts(n, rng);
                    if (max_of(c) == 0) c[0] = 1;
                    std::vector<char> d((size_t) n);
                    for (auto& v : d) v = (char) (rng() & 1);
                    group_case(ctx, pr, c, d, group_gather, (int) (rng() % (unsigned) n), rng, s, "random mix");
                }
            }
        }
        std::printf("mixed groups: %d comparisons, %s\n", g_cases - c0, g_fail == f0 ? "all bitwise identical" : "FAILURES");
        // Part 4
        const int f4 = g_fail, c4 = g_cases;
        const Shape order_shapes[] = {{GGML_TYPE_IQ3_S, 1280, 2560, "gate/up"}, {GGML_TYPE_IQ4_XS, 1280, 2560, "gate/up"},
                                      {GGML_TYPE_IQ4_NL, 2560, 640, "down"},   {GGML_TYPE_Q8_0, 2560, 640, "down"}};
        for (const Shape& sh : order_shapes) {
            const bool whole = mmq::whole_tiles((int) sh.t, sh.rows, G);
            std::printf("%-8s %-7s whole_tiles(%d experts): %s\n", ggml_type_name(sh.t), sh.role, G, whole ? "yes" : "no");
            order_case(ctx, sh, skewed_counts(147, 1, 40, rng), whole, rng, s, "150-token prompt");
            order_case(ctx, sh, skewed_counts(219, 1, 160, rng), whole, rng, s, "512-token chunk");
            order_case(ctx, sh, skewed_counts(48, 1, 300, rng), whole, rng, s, "no partial group");
            order_case(ctx, sh, skewed_counts(11, 1, 300, rng), whole, rng, s, "partial group only");
        }
        std::printf("group order: %d comparisons, %s\n", g_cases - c4, g_fail == f4 ? "all bitwise identical" : "FAILURES");
        cudaStreamDestroy(s);
        std::printf("%d comparisons, %d failed\n", g_cases, g_fail);
        return g_fail == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("error: %s\n", e.what());
        return 1;
    }
}
