// src/kernels/s2_expert_grouped_parity.cpp - the rewritten expert kernels against the previous ones,
// BITWISE (GPU, synthetic, no model).
//
//     build/s2_expert_grouped_parity            the checks below; exit 1 on any difference
//     build/s2_expert_grouped_parity --bench    previous vs new kernel timings on engine-like shapes
//
// Random Q2_0 expert blobs (random codes, fp16 scales in a realistic range) and random Q8_0 activations.  Every entry
// point of `s2_expert_grouped.hpp` that gained a new kernel runs twice from identical inputs and sentinel-filled
// outputs - `moe_grouped_select_old(1)`, then `(0)` - and the two runs are compared byte for byte:
//   1. `moe_hit_grouped_s2`          per hit, host count; with fp32 activation scales and with the blocks' fp16 `d`
//   2. `moe_hit_grouped_s2_dev`      device count below the capacity (the rows past it must stay untouched)
//   3. `moe_hit_grouped_s2_multi`    verify window: each hit reads its own token's activation row
//   4. `moe_grouped_s2`              host-built groups of 1..8 entries, one of them a mapped HOST blob (the PCIe
//                                    share), and more group capacity than groups
//   5. `moe_group_resident` + `moe_grouped_s2`   the MTP layer's shape, with and without fp32 scales
//   6. an activation row that is only 2-byte aligned: the new path must fall back to the previous kernels
//   7. a slot stride that is not a multiple of 8 (BLOB + 4), and 8. an arena 4 bytes past an aligned address: the
//      per-hit path must fall back as well (its codes are read as uint2)
// Every run asks `moe_grouped_last_path` which kernels it launched, so a new path that silently declined cannot pass
// as "the previous kernels against themselves": forced new must report the new kernels, and 6-8 the previous ones.
// (A misaligned scratch is not exercised: the previous kernels store floats there, so it is not a legal input.)
// The output AND the scratch (gate/up, the quantized intermediate and its scales) are compared, so a failure names
// the projection.  Bitwise equality with the previous kernels is the contract; on top of it a double-precision
// host reference of every up row and every down row checks that the fixture computes an expert at all - two
// kernels agreeing on garbage would otherwise pass.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

namespace {
// The blob geometry, `cpu/expert.hpp`'s (restated as in the kernel).
constexpr int H = 2560, FF = 640, NCH_GU = H / 32, NCH_D = FF / 32;
constexpr size_t ROW_GU = H / 4, ROW_D = FF / 4, SC_GU = H / 64, SC_D = FF / 64;
constexpr size_t O_D_CODES = 2ull * FF * ROW_GU;
constexpr size_t O_GU_SCALES = O_D_CODES + (size_t) H * ROW_D;
constexpr size_t O_D_SCALES = O_GU_SCALES + 2ull * FF * SC_GU * 2;
constexpr size_t BLOB = O_D_SCALES + (size_t) H * SC_D * 2;
static_assert(BLOB == 1382400, "the Q2_0 expert blob is 1,382,400 bytes");
constexpr size_t XROW = (size_t) NCH_GU * 34;   // one token's activation row of block_q8_0

int g_fail = 0;

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 256), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 256), "memset");
    return p;
}
template <typename T> void up(T* d, const std::vector<T>& h) {
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "h2d");
}
template <typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "d2h");
    return h;
}

float f16f(const uint8_t* p) { return k::f32_from_f16((uint16_t) (p[0] | (p[1] << 8))); }
void put16(uint8_t* p, float f) {
    const uint16_t h = k::f16_from_f32(f);
    p[0] = (uint8_t) h;
    p[1] = (uint8_t) (h >> 8);
}

// Random codes; fp16 scales of either sign in [1e-3, 3e-2] (a real pack's range, so no overflow downstream).
void fill_blob(uint8_t* b, std::mt19937& rng) {
    for (size_t i = 0; i < O_GU_SCALES; ++i) b[i] = (uint8_t) rng();
    std::uniform_real_distribution<float> mag(1e-3f, 3e-2f);
    for (size_t i = O_GU_SCALES; i < BLOB; i += 2) put16(b + i, (rng() & 1 ? -1.0f : 1.0f) * mag(rng));
}

// `n_tok` activation rows: per 32-chunk an fp16 `d` then 32 int8 in [-127, 127] (what `quantize_q8_0` writes),
// plus the fp32 scales `quantize_q8_0_scaled` would pass alongside.  Some chunks are saturated on purpose.
void fill_x(std::vector<uint8_t>& x, std::vector<float>& xs, int n_tok, std::mt19937& rng) {
    x.assign((size_t) n_tok * XROW, 0);
    xs.assign((size_t) n_tok * NCH_GU, 0.0f);
    std::uniform_real_distribution<float> sc(2e-3f, 5e-2f);
    for (size_t c = 0; c < (size_t) n_tok * NCH_GU; ++c) {
        uint8_t* b = x.data() + c * 34;
        xs[c] = sc(rng);
        put16(b, xs[c]);
        const int mode = (int) (rng() % 16);
        for (int j = 0; j < 32; ++j) {
            const int v = mode == 0 ? 127 : mode == 1 ? -127 : (int) (rng() % 255) - 127;
            b[2 + j] = (uint8_t) (int8_t) v;
        }
    }
}

// Double-precision reference of one row against one activation row: sum_c dw * dx * sum_e (code_e - 1) * x_e.
// `mag` receives sum |term|, the scale the tolerance is relative to.
double row_ref(const uint8_t* codes, const uint8_t* scales, const uint8_t* xrow, const float* xs, int n_chunks,
               double* mag) {
    double acc = 0.0, a = 0.0;
    for (int c = 0; c < n_chunks; ++c) {
        const uint8_t* xb = xrow + (size_t) c * 34;
        const double dx = xs ? (double) xs[c] : (double) f16f(xb);
        const double dw = (double) f16f(scales + (size_t) (c >> 1) * 2);
        int s = 0;
        for (int e = 0; e < 32; ++e) {
            const int code = (codes[(size_t) c * 8 + e / 4] >> (2 * (e % 4))) & 3;
            s += (code - 1) * (int) (int8_t) xb[2 + e];
        }
        const double t = dw * dx * (double) s;
        acc += t;
        a += std::fabs(t);
    }
    *mag = a;
    return acc;
}

bool close_to(float got, double ref, double mag) {
    return std::isfinite(got) && std::fabs((double) got - ref) <= 4e-5 * mag + 1e-30;
}

struct Scratch {
    void* p = nullptr;
    uint64_t bytes = 0;
    int64_t cap = 0;
    uint64_t gu_bytes() const { return ((uint64_t) cap * 2 * FF * 4 + 15) & ~15ull; }
    uint64_t q8_bytes() const { return ((uint64_t) cap * NCH_D * 34 + 15) & ~15ull; }
};

// One entry point, run with the previous kernels then with the new ones from identical sentinel-filled buffers;
// every differing float of the output and byte of the scratch is counted.  `keep` receives the new run.
struct Run {
    std::vector<float> out;
    std::vector<uint8_t> scratch;
};

bool twice(const char* name, float* d_out, size_t out_floats, const Scratch& s,
           const std::function<void()>& call, Run* keep, bool expect_new = true) {
    Run r[2];
    for (int old = 1; old >= 0; --old) {
        ck(cudaMemset(d_out, 0xA5, out_floats * 4), "sentinel out");
        ck(cudaMemset(s.p, 0x5A, s.bytes), "sentinel scratch");
        k::moe_grouped_select_old(old);
        (void) k::moe_grouped_last_path();   // clear
        call();
        const int path = k::moe_grouped_last_path();
        const int want = old ? 0 : (expect_new ? 1 : 0);
        if (path != want) {
            std::printf("  %-44s forced %s: launched %s kernels, expected %s\n", name, old ? "previous" : "new",
                        path == 1 ? "the new" : path == 0 ? "the previous" : "no", want ? "the new" : "the previous");
            ++g_fail;
        }
        ck(cudaGetLastError(), name);
        ck(cudaDeviceSynchronize(), name);
        Run& x = r[old ? 0 : 1];
        x.out = down(d_out, out_floats);
        x.scratch = down((const uint8_t*) s.p, (size_t) s.bytes);
    }
    k::moe_grouped_select_old(-1);
    const uint64_t gu = s.gu_bytes(), q8 = s.q8_bytes();
    size_t diff_out = 0, diff_gu = 0, diff_q8 = 0, diff_hs = 0;
    for (size_t i = 0; i < out_floats; ++i)
        diff_out += std::memcmp(&r[0].out[i], &r[1].out[i], 4) != 0;
    for (size_t i = 0; i < (size_t) s.bytes; ++i) {
        if (r[0].scratch[i] == r[1].scratch[i]) continue;
        if (i < gu) ++diff_gu;
        else if (i < gu + q8) ++diff_q8;
        else ++diff_hs;
    }
    const bool ok = diff_out + diff_gu + diff_q8 + diff_hs == 0;
    std::printf("  %-44s %s", name, ok ? "bitwise identical" : "DIFFERS");
    if (!ok)
        std::printf(" (out %zu floats, gate/up %zu B, intermediate %zu B, its scales %zu B)", diff_out, diff_gu,
                    diff_q8, diff_hs);
    std::printf("\n");
    if (!ok) ++g_fail;
    if (keep) *keep = std::move(r[1]);
    return ok;
}

// The host reference on a finished run: every up row (the second half of gate/up, which the swiglu leaves alone)
// against the activation row it read, and every written down row against the intermediate the run produced.
struct Entry {
    const uint8_t* blob;   // host copy of the entry's blob
    int tok;               // activation row it reads
    int dst;               // output row it writes
};
void reference(const char* name, const Run& r, const Scratch& s, const std::vector<Entry>& ent,
               const std::vector<uint8_t>& x, const std::vector<float>* xs) {
    const float* gu = (const float*) r.scratch.data();
    const uint8_t* hq = r.scratch.data() + s.gu_bytes();
    const float* hs = (const float*) (r.scratch.data() + s.gu_bytes() + s.q8_bytes());
    long checked = 0, bad = 0;
    double worst = 0.0;
    for (size_t e = 0; e < ent.size(); ++e) {
        const uint8_t* b = ent[e].blob;
        const uint8_t* xrow = x.data() + (size_t) ent[e].tok * XROW;
        const float* xsr = xs ? xs->data() + (size_t) ent[e].tok * NCH_GU : nullptr;
        for (int rr = 0; rr < FF; rr += 7) {   // up row rr = row-slot 2rr + 1
            double mag;
            const double ref = row_ref(b + (size_t) (2 * rr + 1) * ROW_GU, b + O_GU_SCALES + (size_t) (2 * rr + 1) * SC_GU * 2,
                                       xrow, xsr, NCH_GU, &mag);
            const float got = gu[(size_t) s.cap * FF + e * FF + (size_t) rr];
            ++checked;
            if (!close_to(got, ref, mag)) ++bad;
            if (mag > 0) worst = std::max(worst, std::fabs((double) got - ref) / mag);
        }
        const uint8_t* hrow = hq + e * (size_t) NCH_D * 34;
        const float* hsr = xs ? hs + e * (size_t) NCH_D : nullptr;
        for (int rr = 0; rr < H; rr += 13) {
            double mag;
            const double ref = row_ref(b + O_D_CODES + (size_t) rr * ROW_D, b + O_D_SCALES + (size_t) rr * SC_D * 2,
                                       hrow, hsr, NCH_D, &mag);
            const float got = r.out[(size_t) ent[e].dst * H + (size_t) rr];
            ++checked;
            if (!close_to(got, ref, mag)) ++bad;
            if (mag > 0) worst = std::max(worst, std::fabs((double) got - ref) / mag);
        }
    }
    std::printf("  %-44s host reference: %ld rows, %ld outside tolerance, worst %.2e of sum|term|\n", name, checked,
                bad, worst);
    if (bad || checked == 0) ++g_fail;
}

struct Fixture {
    int nb = 0;
    std::vector<uint8_t> h;   // nb blobs
    uint8_t* d = nullptr;
    const uint8_t* hb(int i) const { return h.data() + (size_t) i * BLOB; }
};

Fixture make_blobs(int nb, std::mt19937& rng) {
    Fixture f;
    f.nb = nb;
    f.h.resize((size_t) nb * BLOB);
    for (int i = 0; i < nb; ++i) fill_blob(f.h.data() + (size_t) i * BLOB, rng);
    f.d = dalloc<uint8_t>((size_t) nb * BLOB);
    up(f.d, f.h);
    return f;
}

Scratch make_scratch(int64_t cap) {
    Scratch s;
    s.cap = cap;
    s.bytes = k::moe_hit_grouped_scratch_bytes(cap, H, FF);
    s.p = dalloc<uint8_t>(s.bytes);
    return s;
}

// ---------------------------------------------------------------- the checks
void check_all() {
    std::mt19937 rng(20260928);
    const Fixture fx = make_blobs(12, rng);
    constexpr int K = 10;

    // Activations for up to 8 tokens, device copies with and without the fp32 scales.
    std::vector<uint8_t> x;
    std::vector<float> xs;
    fill_x(x, xs, 8, rng);
    uint8_t* d_x = dalloc<uint8_t>(x.size());
    float* d_xs = dalloc<float>(xs.size());
    up(d_x, x);
    up(d_xs, xs);

    // ---- 1. moe_hit_grouped_s2: 9 hits of 10 routed rows, slots and destinations shuffled
    {
        const int n_hits = 9;
        std::vector<int32_t> slot(n_hits), dst(K);
        for (int i = 0; i < n_hits; ++i) slot[i] = (int32_t) ((i * 5 + 3) % fx.nb);
        std::iota(dst.begin(), dst.end(), 0);
        std::shuffle(dst.begin(), dst.end(), rng);
        dst.resize(n_hits);
        int32_t* d_slot = dalloc<int32_t>(n_hits);
        int32_t* d_dst = dalloc<int32_t>(n_hits);
        up(d_slot, slot);
        up(d_dst, dst);
        const Scratch s = make_scratch(n_hits);
        float* d_out = dalloc<float>((size_t) K * H);
        std::vector<Entry> ent;
        for (int i = 0; i < n_hits; ++i) ent.push_back({fx.hb(slot[i]), 0, dst[i]});
        for (int scaled = 1; scaled >= 0; --scaled) {
            Run r;
            const std::string name = std::string("moe_hit_grouped_s2") + (scaled ? " (fp32 scales)" : " (fp16 d)");
            twice(name.c_str(), d_out, (size_t) K * H, s, [&] {
                k::moe_hit_grouped_s2(fx.d, d_slot, d_dst, n_hits, (int64_t) BLOB, d_x, s.p, d_out, nullptr,
                                      scaled ? d_xs : nullptr);
            }, &r);
            reference(name.c_str(), r, s, ent, x, scaled ? &xs : nullptr);
        }
    }

    // ---- 2. moe_hit_grouped_s2_dev: capacity 10, device count 7
    {
        const int cap = 10, count = 7;
        std::vector<int32_t> slot(cap), dst(cap);
        for (int i = 0; i < cap; ++i) slot[i] = (int32_t) ((i * 7 + 1) % fx.nb);
        std::iota(dst.begin(), dst.end(), 0);
        std::shuffle(dst.begin(), dst.end(), rng);
        int32_t* d_slot = dalloc<int32_t>(cap);
        int32_t* d_dst = dalloc<int32_t>(cap);
        int32_t* d_count = dalloc<int32_t>(1);
        up(d_slot, slot);
        up(d_dst, dst);
        up(d_count, std::vector<int32_t>{count});
        const Scratch s = make_scratch(cap);
        float* d_out = dalloc<float>((size_t) K * H);
        std::vector<Entry> ent;
        for (int i = 0; i < count; ++i) ent.push_back({fx.hb(slot[i]), 0, dst[i]});
        Run r;
        twice("moe_hit_grouped_s2_dev (count 7 of 10)", d_out, (size_t) K * H, s, [&] {
            k::moe_hit_grouped_s2_dev(fx.d, d_slot, d_dst, d_count, cap, (int64_t) BLOB, d_x, s.p, d_out, nullptr,
                                      d_xs);
        }, &r);
        reference("moe_hit_grouped_s2_dev (count 7 of 10)", r, s, ent, x, &xs);
        // the rows of the hits past the count keep the sentinel
        long touched = 0;
        for (int i = count; i < cap; ++i)
            for (int c = 0; c < H; ++c) {
                uint32_t v;
                std::memcpy(&v, &r.out[(size_t) dst[i] * H + c], 4);
                touched += v != 0xA5A5A5A5u;
            }
        std::printf("  %-44s rows past the count written: %ld\n", "moe_hit_grouped_s2_dev (count 7 of 10)", touched);
        if (touched) ++g_fail;
    }

    // ---- 3. moe_hit_grouped_s2_multi: 4 tokens x 10 routed entries, 29 hits
    {
        const int T = 4, n = T * K, count = 29;
        std::vector<int32_t> ent_ids(n);
        std::iota(ent_ids.begin(), ent_ids.end(), 0);
        std::shuffle(ent_ids.begin(), ent_ids.end(), rng);
        std::vector<int32_t> slot(n), dst(n);
        for (int i = 0; i < n; ++i) {
            dst[i] = ent_ids[i];
            slot[i] = (int32_t) (rng() % fx.nb);
        }
        std::sort(dst.begin(), dst.begin() + count);   // hit order is routing order, as `moe_hit_select_multi` makes it
        int32_t* d_slot = dalloc<int32_t>(n);
        int32_t* d_dst = dalloc<int32_t>(n);
        int32_t* d_count = dalloc<int32_t>(1);
        up(d_slot, slot);
        up(d_dst, dst);
        up(d_count, std::vector<int32_t>{count});
        const Scratch s = make_scratch(n);
        float* d_out = dalloc<float>((size_t) n * H);
        std::vector<Entry> ent;
        for (int i = 0; i < count; ++i) ent.push_back({fx.hb(slot[i]), dst[i] / K, dst[i]});
        Run r;
        twice("moe_hit_grouped_s2_multi (4 tokens, 29 hits)", d_out, (size_t) n * H, s, [&] {
            k::moe_hit_grouped_s2_multi(fx.d, d_slot, d_dst, d_count, n, (int64_t) BLOB, d_x, d_xs, K, s.p, d_out,
                                        nullptr);
        }, &r);
        reference("moe_hit_grouped_s2_multi (4 tokens, 29 hits)", r, s, ent, x, &xs);
    }

    // ---- 4. moe_grouped_s2: 8 tokens x 10 entries in groups of 1..8, one group on a mapped host blob
    {
        const int T = 8, n = T * K;
        uint8_t* h_host = nullptr;
        uint8_t* d_host = nullptr;
        ck(cudaHostAlloc((void**) &h_host, BLOB, cudaHostAllocMapped), "host blob");
        std::memcpy(h_host, fx.hb(fx.nb - 1), BLOB);
        ck(cudaHostGetDevicePointer((void**) &d_host, h_host, 0), "host blob ptr");
        // group sizes: 8, 1, 5, 2, 7, 3, 1, 6, 4, ... up to n entries
        const int sizes[] = {8, 1, 5, 2, 7, 3, 1, 6, 4, 8, 1, 2, 3, 4, 5, 6, 7, 1, 1, 5};
        std::vector<int32_t> order(n);   // the output rows, in group order
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);
        std::vector<unsigned long long> gptr;
        std::vector<int32_t> gstart{0}, edst, etok;
        std::vector<Entry> ent;
        int at = 0, gi = 0;
        while (at < n) {
            const int sz = std::min(sizes[gi % 20], n - at);
            const bool host = gi == 3;
            const int b = gi % (fx.nb - 1);
            gptr.push_back((unsigned long long) (host ? d_host : fx.d + (size_t) b * BLOB));
            for (int j = 0; j < sz; ++j, ++at) {
                edst.push_back(order[at]);
                etok.push_back(order[at] / K);
                ent.push_back({host ? h_host : fx.hb(b), order[at] / K, order[at]});
            }
            gstart.push_back(at);
            ++gi;
        }
        const int n_groups = gi, cap_groups = n_groups + 3;
        gstart.resize((size_t) cap_groups + 1, at);
        gptr.resize((size_t) cap_groups, gptr[0]);
        unsigned long long* d_gptr = dalloc<unsigned long long>(gptr.size());
        int32_t* d_gstart = dalloc<int32_t>(gstart.size());
        int32_t* d_ng = dalloc<int32_t>(1);
        int32_t* d_edst = dalloc<int32_t>(n);
        int32_t* d_etok = dalloc<int32_t>(n);
        up(d_gptr, gptr);
        up(d_gstart, gstart);
        up(d_ng, std::vector<int32_t>{n_groups});
        up(d_edst, edst);
        up(d_etok, etok);
        const Scratch s = make_scratch(n);
        float* d_out = dalloc<float>((size_t) n * H);
        for (int scaled = 1; scaled >= 0; --scaled) {
            Run r;
            const std::string name = std::string("moe_grouped_s2 (") + std::to_string(n_groups) + " groups" +
                                     (scaled ? ", fp32 scales)" : ", fp16 d)");
            twice(name.c_str(), d_out, (size_t) n * H, s, [&] {
                k::moe_grouped_s2(d_gptr, d_gstart, d_ng, d_edst, d_etok, cap_groups, n, d_x, scaled ? d_xs : nullptr,
                                  s.p, d_out, nullptr);
            }, &r);
            reference(name.c_str(), r, s, ent, x, scaled ? &xs : nullptr);
        }
        ck(cudaFreeHost(h_host), "free host blob");
    }

    // ---- 5. moe_group_resident + moe_grouped_s2: the MTP layer (experts resident at base + id * BLOB)
    {
        const int T = 5, n = T * K;
        std::vector<int32_t> ids(n);
        for (int t = 0; t < T; ++t) {   // distinct within a token, shared across tokens
            std::vector<int32_t> pick(fx.nb);
            std::iota(pick.begin(), pick.end(), 0);
            std::shuffle(pick.begin(), pick.end(), rng);
            for (int j = 0; j < K; ++j) ids[(size_t) t * K + j] = pick[j];
        }
        int32_t* d_ids = dalloc<int32_t>(n);
        up(d_ids, ids);
        unsigned long long* d_gptr = dalloc<unsigned long long>(n);
        int32_t* d_gstart = dalloc<int32_t>(n + 1);
        int32_t* d_counts = dalloc<int32_t>(2);
        int32_t* d_edst = dalloc<int32_t>(n);
        int32_t* d_etok = dalloc<int32_t>(n);
        k::moe_group_resident(d_ids, n, K, fx.d, (int64_t) BLOB, d_gptr, d_gstart, d_counts, d_edst, d_etok, nullptr);
        ck(cudaDeviceSynchronize(), "group_resident");
        const std::vector<int32_t> edst = down(d_edst, n), etok = down(d_etok, n);
        std::vector<Entry> ent(n);   // entry e: dst = routing index, blob = its expert
        for (int e = 0; e < n; ++e) ent[e] = {fx.hb(ids[(size_t) edst[e]]), etok[e], edst[e]};
        const Scratch s = make_scratch(n);
        float* d_out = dalloc<float>((size_t) n * H);
        for (int scaled = 1; scaled >= 0; --scaled) {
            Run r;
            const std::string name = std::string("moe_group_resident + moe_grouped_s2") + (scaled ? " (fp32)" : " (fp16)");
            twice(name.c_str(), d_out, (size_t) n * H, s, [&] {
                k::moe_grouped_s2(d_gptr, d_gstart, d_counts, d_edst, d_etok, n, n, d_x, scaled ? d_xs : nullptr, s.p,
                                  d_out, nullptr);
            }, &r);
            reference(name.c_str(), r, s, ent, x, scaled ? &xs : nullptr);
        }
    }

    // ---- 6. a 2-byte aligned activation row: the new path must decline and the previous kernels run
    {
        uint8_t* d_x2 = dalloc<uint8_t>(x.size() + 2);
        ck(cudaMemcpy(d_x2 + 2, x.data(), XROW, cudaMemcpyHostToDevice), "x2");
        const int n_hits = 3;
        std::vector<int32_t> slot{2, 5, 7}, dst{4, 0, 9};
        int32_t* d_slot = dalloc<int32_t>(n_hits);
        int32_t* d_dst = dalloc<int32_t>(n_hits);
        up(d_slot, slot);
        up(d_dst, dst);
        const Scratch s = make_scratch(n_hits);
        float* d_out = dalloc<float>((size_t) K * H);
        std::vector<Entry> ent;
        for (int i = 0; i < n_hits; ++i) ent.push_back({fx.hb(slot[i]), 0, dst[i]});
        Run r;
        twice("moe_hit_grouped_s2 (2-byte aligned x)", d_out, (size_t) K * H, s, [&] {
            k::moe_hit_grouped_s2(fx.d, d_slot, d_dst, n_hits, (int64_t) BLOB, d_x2 + 2, s.p, d_out, nullptr, d_xs);
        }, &r, /*expect_new=*/false);
        reference("moe_hit_grouped_s2 (2-byte aligned x)", r, s, ent, x, &xs);
    }

    // ---- 7. slots BLOB + 4 bytes apart, and 8. an arena at an address that is 4 mod 8: per-hit fallback
    {
        constexpr size_t STRIDE = BLOB + 4;
        static_assert(STRIDE % 8 != 0, "the stride must defeat the uint2 code loads");
        const int nb = 8;
        uint8_t* d_arena = dalloc<uint8_t>((size_t) nb * STRIDE + 8);
        const int n_hits = 4;
        std::vector<int32_t> slot{6, 1, 3, 0}, dst{2, 7, 5, 8};
        int32_t* d_slot = dalloc<int32_t>(n_hits);
        int32_t* d_dst = dalloc<int32_t>(n_hits);
        up(d_slot, slot);
        up(d_dst, dst);
        const Scratch s = make_scratch(n_hits);
        float* d_out = dalloc<float>((size_t) K * H);
        std::vector<Entry> ent;
        for (int i = 0; i < n_hits; ++i) ent.push_back({fx.hb(slot[i]), 0, dst[i]});
        // One allocation for both layouts: the BLOB + 4 strided copies first, then the same blobs BLOB apart from
        // `d_arena + 4` over them.
        for (int i = 0; i < nb; ++i)
            ck(cudaMemcpy(d_arena + (size_t) i * STRIDE, fx.hb(i), BLOB, cudaMemcpyHostToDevice), "arena");
        {
            Run r;
            twice("moe_hit_grouped_s2 (slot stride BLOB + 4)", d_out, (size_t) K * H, s, [&] {
                k::moe_hit_grouped_s2(d_arena, d_slot, d_dst, n_hits, (int64_t) STRIDE, d_x, s.p, d_out, nullptr,
                                      d_xs);
            }, &r, /*expect_new=*/false);
            reference("moe_hit_grouped_s2 (slot stride BLOB + 4)", r, s, ent, x, &xs);
        }
        for (int i = 0; i < nb; ++i)
            ck(cudaMemcpy(d_arena + 4 + (size_t) i * BLOB, fx.hb(i), BLOB, cudaMemcpyHostToDevice), "arena");
        {
            Run r;
            twice("moe_hit_grouped_s2 (arena 4 mod 8)", d_out, (size_t) K * H, s, [&] {
                k::moe_hit_grouped_s2(d_arena + 4, d_slot, d_dst, n_hits, (int64_t) BLOB, d_x, s.p, d_out, nullptr,
                                      d_xs);
            }, &r, /*expect_new=*/false);
            reference("moe_hit_grouped_s2 (arena 4 mod 8)", r, s, ent, x, &xs);
        }
    }
}

// ---------------------------------------------------------------- --bench
// Engine-like shapes with the blobs cycled through a set larger than the L2, so each call reads its codes from
// DRAM as the engine's do.  Reports microseconds per call and the codes' effective bandwidth.
void bench() {
    int dev = 0;
    cudaDeviceProp prop{};
    ck(cudaGetDevice(&dev), "dev");
    ck(cudaGetDeviceProperties(&prop, dev), "props");
    const int nb = std::max(48, (int) (3ull * (size_t) prop.l2CacheSize / BLOB) + 8);
    std::printf("bench: %s, L2 %d MB, %d blobs (%.0f MB) cycled\n", prop.name, prop.l2CacheSize >> 20, nb,
                nb * (double) BLOB / 1e6);
    std::mt19937 rng(7);
    Fixture fx;
    fx.nb = nb;
    fx.d = dalloc<uint8_t>((size_t) nb * BLOB);
    {
        std::vector<uint8_t> b(BLOB);
        for (int i = 0; i < nb; ++i) {
            fill_blob(b.data(), rng);
            ck(cudaMemcpy(fx.d + (size_t) i * BLOB, b.data(), BLOB, cudaMemcpyHostToDevice), "blob");
        }
    }
    std::vector<uint8_t> x;
    std::vector<float> xs;
    fill_x(x, xs, 8, rng);
    uint8_t* d_x = dalloc<uint8_t>(x.size());
    float* d_xs = dalloc<float>(xs.size());
    up(d_x, x);
    up(d_xs, xs);
    constexpr int K = 10, ITERS = 200, SETS = 16;
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "ev");
    ck(cudaEventCreate(&e1), "ev");
    auto time = [&](const std::function<void(int)>& call) {
        call(0);
        ck(cudaDeviceSynchronize(), "warm");
        ck(cudaEventRecord(e0), "rec");
        for (int i = 0; i < ITERS; ++i) call(i % SETS);
        ck(cudaEventRecord(e1), "rec");
        ck(cudaEventSynchronize(e1), "sync");
        float ms = 0;
        ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
        return 1000.0 * ms / ITERS;
    };
    auto ab = [&](const char* name, double blobs_per_call, const std::function<void(int)>& call) {
        double us[2];
        for (int old = 1; old >= 0; --old) {
            k::moe_grouped_select_old(old);
            us[old ? 0 : 1] = time(call);
        }
        k::moe_grouped_select_old(-1);
        const double gb = blobs_per_call * (double) BLOB / 1e3;   // bytes / us = MB/s ... / 1e3 -> GB/s
        std::printf("  %-40s previous %8.1f us (%6.0f GB/s)   new %8.1f us (%6.0f GB/s)   x%.2f\n", name, us[0],
                    gb / us[0], us[1], gb / us[1], us[0] / us[1]);
    };

    // per-hit path, decode: 9 hits of 10 (5090-like residency), 5 of 10, and the low-residency 1-3 hits, where the
    // new kernels' grid (80 gate/up blocks per hit) is below one block per SM - see `STRATA_GROUPED_PAIR_MIN_HITS`
    for (const int n_hits : {9, 5, 3, 2, 1}) {
        std::vector<int32_t> slots((size_t) SETS * n_hits), dst(K);
        for (auto& v : slots) v = (int32_t) (rng() % nb);
        std::iota(dst.begin(), dst.end(), 0);
        int32_t* d_slots = dalloc<int32_t>(slots.size());
        int32_t* d_dst = dalloc<int32_t>(K);
        up(d_slots, slots);
        up(d_dst, dst);
        const Scratch s = make_scratch(n_hits);
        float* d_out = dalloc<float>((size_t) K * H);
        const std::string name = "moe_hit_grouped_s2, " + std::to_string(n_hits) + " hits";
        ab(name.c_str(), n_hits, [&](int set) {
            k::moe_hit_grouped_s2(fx.d, d_slots + (size_t) set * n_hits, d_dst, n_hits, (int64_t) BLOB, d_x, s.p,
                                  d_out, nullptr, d_xs);
        });
    }
    // grouped path, verify window: T tokens x 10, routed with `share` = the chance an entry reuses an expert already
    // routed by an earlier token of the window
    struct Shape { int T; double share; const char* what; };
    for (const Shape sh : {Shape{4, 0.0, "4 tokens, no sharing"}, Shape{4, 0.3, "4 tokens, 30% shared"},
                           Shape{8, 0.3, "8 tokens, 30% shared"}, Shape{8, 1.0, "8 tokens, all shared (ne 8)"}}) {
        const int n = sh.T * K;
        std::vector<unsigned long long> gptr((size_t) SETS * n);
        std::vector<int32_t> gstart((size_t) SETS * (n + 1)), ng(SETS), edst((size_t) SETS * n), etok((size_t) SETS * n);
        double groups = 0;
        std::uniform_real_distribution<double> u(0, 1);
        for (int set = 0; set < SETS; ++set) {
            std::vector<std::vector<int>> g;       // per group: routing indices
            std::vector<int> gexp;                 // per group: blob
            for (int t = 0; t < sh.T; ++t) {
                std::vector<char> used(g.size(), 0);
                for (int j = 0; j < K; ++j) {
                    int pick = -1;
                    if (t > 0 && u(rng) < sh.share)
                        for (size_t q = 0; q < g.size(); ++q)
                            if (!used[q] && (int) g[q].size() < 8) { pick = (int) q; break; }
                    if (pick < 0) {
                        pick = (int) g.size();
                        g.emplace_back();
                        gexp.push_back((int) (rng() % nb));
                        used.push_back(0);
                    }
                    used[(size_t) pick] = 1;
                    g[(size_t) pick].push_back(t * K + j);
                }
            }
            int at = 0;
            for (size_t q = 0; q < g.size(); ++q) {
                gptr[(size_t) set * n + q] = (unsigned long long) (fx.d + (size_t) gexp[q] * BLOB);
                gstart[(size_t) set * (n + 1) + q] = at;
                for (int e : g[q]) {
                    edst[(size_t) set * n + at] = e;
                    etok[(size_t) set * n + at] = e / K;
                    ++at;
                }
            }
            for (size_t q = g.size(); q <= (size_t) n; ++q) gstart[(size_t) set * (n + 1) + q] = at;
            ng[set] = (int32_t) g.size();
            groups += (double) g.size() / SETS;
        }
        unsigned long long* d_gptr = dalloc<unsigned long long>(gptr.size());
        int32_t* d_gstart = dalloc<int32_t>(gstart.size());
        int32_t* d_ng = dalloc<int32_t>(SETS);
        int32_t* d_edst = dalloc<int32_t>(edst.size());
        int32_t* d_etok = dalloc<int32_t>(etok.size());
        up(d_gptr, gptr);
        up(d_gstart, gstart);
        up(d_ng, ng);
        up(d_edst, edst);
        up(d_etok, etok);
        const Scratch s = make_scratch(n);
        float* d_out = dalloc<float>((size_t) n * H);
        char name[96];
        std::snprintf(name, sizeof name, "moe_grouped_s2, %s (%.0f groups)", sh.what, groups);
        ab(name, groups, [&](int set) {
            k::moe_grouped_s2(d_gptr + (size_t) set * n, d_gstart + (size_t) set * (n + 1), d_ng + set,
                              d_edst + (size_t) set * n, d_etok + (size_t) set * n, n, n, d_x, d_xs, s.p, d_out,
                              nullptr);
        });
    }
}
}  // namespace

int main(int argc, char** argv) {
    bool do_bench = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0) do_bench = true;
        else if (std::strcmp(argv[i], "--selftest") != 0) {
            std::fprintf(stderr, "usage: s2_expert_grouped_parity [--selftest] [--bench]\n");
            return 2;
        }
    }
    std::printf("s2_expert_grouped_parity: new expert kernels vs the previous ones, bitwise\n");
    check_all();
    if (do_bench) bench();
    std::printf("s2_expert_grouped_parity: %d failures\n%s\n", g_fail, g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
