// src/kernels/kv_window_parity.cpp - the verify window's KV calls on a STREAMED layer, as the engine makes them (GPU, no model).
//
// kv_stream_parity appends one cell per call. The engine's verify window (and the MTP drafter's catch-up) appends n = 2..6
// cells in ONE launch (kv_append_*_steps, a 3D grid with the host copy as a kernel argument), resolves the n selections
// together, attends n queries in one batched launch, and does all of it inside a captured graph on a non-blocking
// stream. This test drives exactly that sequence for each KV format on a layer that must evict, and on a ring over a host
// copy (the drafter's layout), and checks
//   1. the host copy equals a fully resident reference pool, byte for byte, for every cell written;
//   2. every resident slot equals the reference block it holds (the slots never go stale);
//   3. the streamed attention of each window equals the resident one, bitwise (n queries);
//   4. the ring's attention (last cells, through `block % n_slots`) equals the resident one;
//   5. nothing faulted (a stray store or load of the append / copy / attention kernels ends the run in a CUDA error).
// Every window runs twice per format: as plain launches and replayed from a graph captured per window size. Issue #2403
// (q4_0 KV + KV streaming + MTP) is the case this was written for.
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset");
    return p;
}
struct HostBuf {   // pinned, mapped: `d` is what kernels get, `h` what the test reads
    void* h = nullptr;
    void* d = nullptr;
};
HostBuf halloc(size_t bytes) {
    HostBuf b;
    ck(cudaHostAlloc(&b.h, bytes + 64, cudaHostAllocMapped), "hostalloc");
    std::memset(b.h, 0, bytes + 64);
    ck(cudaHostGetDevicePointer(&b.d, b.h, 0), "devptr");
    return b;
}

constexpr int kMaxWin = 6;   // --spec 4 windows are <= 5 tokens, the drafter's catch-up T <= 6

// One KV array of a pool set: the pointer (device or host-mapped), the host-readable twin, bytes per page (block).
struct Arr {
    void* d = nullptr;
    void* h = nullptr;   // non-null for host copies
    size_t per_block = 0;
};

struct Pools {   // one pool set of `pages` pages in the layout of format `fmt`
    k::KvHostPools p;
    std::vector<Arr> arrs;
    void alloc(int64_t pages, const k::QsaShapes& s, int fmt, bool host) {
        const size_t rows_per_page = (size_t) s.n_head_kv * s.page_size, rows = (size_t) pages * rows_per_page;
        auto mk = [&](size_t per_row, auto** field) {
            using T = std::remove_pointer_t<std::remove_pointer_t<decltype(field)>>;
            Arr a;
            a.per_block = rows_per_page * per_row;
            if (host) { HostBuf b = halloc(rows * per_row); a.d = b.d; a.h = b.h; }
            else a.d = dalloc<uint8_t>(rows * per_row);
            *field = (T*) a.d;
            arrs.push_back(a);
        };
        const size_t q4 = k::kv_q4_bytes_per_head((int) s.head_dim);
        const size_t codes = (size_t) s.head_dim, scales = (size_t) (s.head_dim / k::KV_Q8_GROUP) * 2;
        if (fmt == k::kKvHybrid) {
            mk(codes, &p.k_q); mk(scales, &p.k_scale); mk(q4, &p.v_q4);
        } else if (fmt == k::kKvQ4) {
            mk(q4, &p.k_q4); mk(q4, &p.v_q4);
        } else {   // kKvInt8
            mk(codes, &p.k_q); mk(codes, &p.v_q); mk(scales, &p.k_scale); mk(scales, &p.v_scale);
        }
    }
    k::QsaAttnPools attn(const int32_t* table) const {
        k::QsaAttnPools a;
        a.k_pool = p.k_pool; a.v_pool = p.v_pool; a.k_q = p.k_q; a.v_q = p.v_q; a.k_scale = p.k_scale;
        a.v_scale = p.v_scale; a.k_q4 = p.k_q4; a.v_q4 = p.v_q4; a.page_table = table;
        return a;
    }
};

// The appends of one window, as verify.cpp / mtp.cpp make them.
void append_window(const Pools& pl, const int32_t* table, const int32_t* steps, const float* kc, const float* vc, int n,
                   const k::QsaShapes& s, int fmt, const k::KvHostPools* host, cudaStream_t cs) {
    const int cur = (int) (s.n_head_kv * s.head_dim);
    if (fmt == k::kKvHybrid) {   // the unused half's lanes folded onto the used pool (layer.cpp), the host copy by halves
        const k::KvHostPools hk = host ? k::kv_hybrid_k_half(*host) : k::KvHostPools{},
                             hv = host ? k::kv_hybrid_v_half(*host) : k::KvHostPools{};
        k::kv_append_q8_steps(pl.p.k_q, pl.p.k_q, pl.p.k_scale, pl.p.k_scale, table, steps, k::kStepCount, kc, kc, cur, n, s, cs,
                              host ? &hk : nullptr);
        k::kv_append_q4_steps(pl.p.v_q4, pl.p.v_q4, table, steps, k::kStepCount, n, vc, vc, s, cs, host ? &hv : nullptr);
    } else if (fmt == k::kKvQ4) {
        k::kv_append_q4_steps(pl.p.k_q4, pl.p.v_q4, table, steps, k::kStepCount, n, kc, vc, s, cs, host);
    } else {
        k::kv_append_q8_steps(pl.p.k_q, pl.p.v_q, pl.p.k_scale, pl.p.v_scale, table, steps, k::kStepCount, kc, vc, cur, n, s, cs,
                              host);
    }
}

// A selection like qsa_block_topk's: ascending cells, whole blocks, the tail block's cells, and one partial block.
std::vector<int32_t> selection(int64_t n_kv, int64_t width, std::mt19937& rng) {
    std::vector<int32_t> ids;
    if (n_kv <= width) {
        for (int64_t c = 0; c < n_kv; ++c) ids.push_back((int32_t) c);
        return ids;
    }
    const int64_t n_bid = n_kv / 4, tail = n_kv - n_bid * 4;
    std::set<int64_t> blocks;
    const int64_t want_full = (width - tail) / 4, part = (width - tail) % 4;
    std::uniform_int_distribution<int64_t> any(0, n_bid - 1), recent(std::max<int64_t>(0, n_bid - 2048), n_bid - 1);
    while ((int64_t) blocks.size() < want_full + (part ? 1 : 0)) blocks.insert(rng() % 2 ? recent(rng) : any(rng));
    int64_t partial_block = part ? *std::next(blocks.begin(), (long) (rng() % blocks.size())) : -1;
    for (int64_t b : blocks) {
        const int64_t take = b == partial_block ? part : 4;
        for (int64_t i = 0; i < take; ++i) ids.push_back((int32_t) (b * 4 + i));
    }
    for (int64_t c = n_bid * 4; c < n_kv; ++c) ids.push_back((int32_t) c);
    std::sort(ids.begin(), ids.end());
    return ids;
}

int g_bad = 0;
void fail(const char* fmt_name, const char* what, long long a = 0, long long b = 0) {
    if (g_bad++ < 12) std::fprintf(stderr, "  FAIL %s: %s (%lld, %lld)\n", fmt_name, what, a, b);
}

bool run(int fmt, bool graphs) {
    const char* name = fmt == k::kKvHybrid ? "k8v4" : fmt == k::kKvQ4 ? "q4_0" : "int8";
    k::QsaShapes s = k::qsa_real_shapes();
    const int64_t N = 40000, n_blocks = (N + 3) / 4;
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    const int64_t n_slots = kMaxWin * (cap / 4 + 2) + 700;   // holds one window's selections; far fewer than n_blocks
    const int64_t R = 1500;                                  // the drafter-like ring
    const int64_t H = s.n_head_kv, D = s.head_dim, NH = s.n_head;
    std::mt19937 rng(11 + 5 * fmt + graphs);
    std::normal_distribution<float> nd(0.f, 1.f);

    cudaStream_t cs;
    ck(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "stream");
    Pools ref, slots, host, ring, ring_host;
    ref.alloc(n_blocks, s, fmt, false);
    slots.alloc(n_slots, s, fmt, false);
    host.alloc(n_blocks, s, fmt, true);
    ring.alloc(R, s, fmt, false);
    ring_host.alloc(n_blocks, s, fmt, true);
    int32_t* ident = dalloc<int32_t>(n_blocks);
    int32_t* rtab = dalloc<int32_t>(n_blocks);
    {
        std::vector<int32_t> t(n_blocks);
        for (int64_t i = 0; i < n_blocks; ++i) t[i] = (int32_t) i;
        ck(cudaMemcpy(ident, t.data(), n_blocks * 4, cudaMemcpyHostToDevice), "ident");
        k::kv_ring_table(rtab, n_blocks, R, nullptr);
    }
    k::KvStreamMap m;
    m.page_table = dalloc<int32_t>(n_blocks);
    m.slot_block = dalloc<int32_t>(n_slots); m.slot_stamp = dalloc<int32_t>(n_slots); m.slot_ref = dalloc<int32_t>(n_slots);
    m.miss_block = dalloc<int32_t>(n_slots); m.miss_slot = dalloc<int32_t>(n_slots);
    m.ctl = dalloc<int32_t>(k::kKvCtlInts);
    m.n_blocks = n_blocks; m.n_slots = n_slots;
    k::kv_stream_reset(m, nullptr);
    ck(cudaDeviceSynchronize(), "setup");

    float* kc = dalloc<float>(kMaxWin * H * D);
    float* vc = dalloc<float>(kMaxWin * H * D);
    int32_t* steps = dalloc<int32_t>(kMaxWin * k::kStepCount);
    int32_t* ids = dalloc<int32_t>(kMaxWin * cap);
    int32_t* ring_ids = dalloc<int32_t>(cap);
    int32_t* ring_step = dalloc<int32_t>(k::kStepCount);
    float* q = dalloc<float>(kMaxWin * NH * D);
    const uint64_t scr = k::qsa_decode_attn_scratch_floats(cap, s);
    float* scratch = dalloc<float>(kMaxWin * scr);
    float* out_ref = dalloc<float>(kMaxWin * NH * D);
    float* out_str = dalloc<float>(kMaxWin * NH * D);
    float* scratch2 = dalloc<float>(scr);
    float* out_rref = dalloc<float>(NH * D);
    float* out_rng = dalloc<float>(NH * D);
    const k::KvHostPools* hp = &host.p;
    const k::KvHostPools* rhp = &ring_host.p;

    // per window size n: the appends (resident reference, streamed + host copy, ring + its host copy) and, for a window
    // that is also attended, the streamed resolve + attention, each captured once
    cudaGraphExec_t g_app[kMaxWin + 1] = {}, g_att[kMaxWin + 1] = {};
    auto record_app = [&](int n) {
        append_window(ref, ident, steps, kc, vc, n, s, fmt, nullptr, cs);
        append_window(slots, m.page_table, steps, kc, vc, n, s, fmt, hp, cs);
        append_window(ring, rtab, steps, kc, vc, n, s, fmt, rhp, cs);
    };
    auto record_att = [&](int n) {
        k::kv_stream_resolve(m, slots.attn(m.page_table), host.p, fmt, ids, steps, n, cap, s, cs);
        k::qsa_decode_attn_batch(q, slots.attn(m.page_table), ids, steps, cap, s, scratch, out_str, n, cs);
    };
    if (graphs) {
        for (int n = 1; n <= kMaxWin; ++n) {
            cudaGraph_t g = nullptr;
            ck(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal), "begin capture");
            record_app(n);
            ck(cudaStreamEndCapture(cs, &g), "end capture");
            ck(cudaGraphInstantiate(&g_app[n], g, 0), "instantiate append");
            (void) cudaGraphDestroy(g);
            ck(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal), "begin capture");
            record_att(n);
            ck(cudaStreamEndCapture(cs, &g), "end capture");
            ck(cudaGraphInstantiate(&g_att[n], g, 0), "instantiate attention");
            (void) cudaGraphDestroy(g);
        }
    }

    std::vector<float> hk(kMaxWin * H * D), hv(kMaxWin * H * D), hq(kMaxWin * NH * D);
    std::vector<float> a((size_t) kMaxWin * NH * D), b2((size_t) kMaxWin * NH * D);
    int windows = 0, attended = 0;
    int64_t pos = 0;
    while (pos < N) {
        const int n = (int) std::min<int64_t>(N - pos, 1 + (int64_t) (rng() % kMaxWin));
        for (int j = 0; j < n; ++j)
            for (int64_t i = 0; i < H * D; ++i) {
                hk[j * H * D + i] = nd(rng) * (((pos + j) % 17 == 0) ? 30.f : 1.f);
                hv[j * H * D + i] = nd(rng);
            }
        std::vector<int32_t> st((size_t) n * k::kStepCount);
        for (int j = 0; j < n; ++j) {
            const int64_t p = pos + j;
            st[j * 4 + k::kStepPos] = (int32_t) p; st[j * 4 + k::kStepNKv] = (int32_t) (p + 1);
            st[j * 4 + k::kStepNBid] = (int32_t) ((p + 1) / 4); st[j * 4 + k::kStepWidth] = (int32_t) std::min<int64_t>(p + 1, cap);
        }
        ck(cudaMemcpyAsync(kc, hk.data(), (size_t) n * H * D * 4, cudaMemcpyHostToDevice, cs), "k");
        ck(cudaMemcpyAsync(vc, hv.data(), (size_t) n * H * D * 4, cudaMemcpyHostToDevice, cs), "v");
        ck(cudaMemcpyAsync(steps, st.data(), st.size() * 4, cudaMemcpyHostToDevice, cs), "steps");
        ck(cudaStreamSynchronize(cs), "upload");   // pageable sources: the copies above must be done before the vectors change
        if (graphs) ck(cudaGraphLaunch(g_app[n], cs), "graph append");
        else record_app(n);
        ++windows;
        pos += n;
        if (windows % 37 != 0 && pos < N) continue;
        // attend the window just appended: n queries at its n positions, each with a selection of its own
        std::vector<int32_t> hids((size_t) n * cap, 0);
        for (int j = 0; j < n; ++j) {
            const int64_t n_kv = pos - (n - 1 - j), width = std::min<int64_t>(n_kv, cap);
            const std::vector<int32_t> sel = selection(n_kv, width, rng);
            if ((int64_t) sel.size() != width) { std::fprintf(stderr, "selection size\n"); std::exit(2); }
            std::copy(sel.begin(), sel.end(), hids.begin() + (size_t) j * cap);
        }
        for (auto& x : hq) x = nd(rng);
        ck(cudaMemcpyAsync(ids, hids.data(), hids.size() * 4, cudaMemcpyHostToDevice, cs), "ids");
        ck(cudaMemcpyAsync(q, hq.data(), (size_t) n * NH * D * 4, cudaMemcpyHostToDevice, cs), "q");
        ck(cudaStreamSynchronize(cs), "upload2");
        k::qsa_decode_attn_batch(q, ref.attn(ident), ids, steps, cap, s, scratch, out_ref, n, cs);
        if (graphs) ck(cudaGraphLaunch(g_att[n], cs), "graph attention");
        else record_att(n);
        // the ring: the last cells of the last query, through block % R (the drafter reads its window only)
        {
            const int64_t n_kv = pos, width = std::min<int64_t>(n_kv, cap);
            std::vector<int32_t> rid((size_t) cap, 0);
            for (int64_t i = 0; i < width; ++i) rid[i] = (int32_t) (n_kv - width + i);
            const int32_t rs[4] = {(int32_t) (n_kv - 1), (int32_t) n_kv, (int32_t) (n_kv / 4), (int32_t) width};
            ck(cudaMemcpyAsync(ring_ids, rid.data(), rid.size() * 4, cudaMemcpyHostToDevice, cs), "rid");
            ck(cudaMemcpyAsync(ring_step, rs, sizeof(rs), cudaMemcpyHostToDevice, cs), "rs");
            ck(cudaStreamSynchronize(cs), "upload3");
            k::qsa_decode_attn_batch(q, ref.attn(ident), ring_ids, ring_step, cap, s, scratch2, out_rref, 1, cs);
            k::qsa_decode_attn_batch(q, ring.attn(rtab), ring_ids, ring_step, cap, s, scratch2, out_rng, 1, cs);
        }
        ck(cudaStreamSynchronize(cs), "window");
        ck(cudaGetLastError(), "window (last error)");
        // 3. the streamed attention of the n queries equals the resident one
        ck(cudaMemcpy(a.data(), out_ref, (size_t) n * NH * D * 4, cudaMemcpyDeviceToHost), "a");
        ck(cudaMemcpy(b2.data(), out_str, (size_t) n * NH * D * 4, cudaMemcpyDeviceToHost), "b");
        if (std::memcmp(a.data(), b2.data(), (size_t) n * NH * D * 4) != 0) fail(name, "streamed attention differs", pos, n);
        // 4. the ring reads the same values as the resident pool
        std::vector<float> r(NH * D), rr(NH * D);
        ck(cudaMemcpy(r.data(), out_rng, (size_t) NH * D * 4, cudaMemcpyDeviceToHost), "r");
        ck(cudaMemcpy(rr.data(), out_rref, (size_t) NH * D * 4, cudaMemcpyDeviceToHost), "rr");
        if (std::memcmp(rr.data(), r.data(), (size_t) NH * D * 4) != 0) fail(name, "ring attention differs", pos, n);
        ++attended;
    }
    // 1. the host copies equal the resident reference, byte for byte (all cells written)
    long long host_bad = 0, slot_bad = 0, resident = 0;
    for (int ai = 0; ai < (int) ref.arrs.size(); ++ai) {
        const size_t per = ref.arrs[ai].per_block;
        std::vector<uint8_t> rb((size_t) n_blocks * per);
        ck(cudaMemcpy(rb.data(), ref.arrs[ai].d, rb.size(), cudaMemcpyDeviceToHost), "ref d2h");
        const size_t written = (size_t) ((N + 3) / 4) * per;
        host_bad += std::memcmp(rb.data(), host.arrs[ai].h, written) != 0;
        host_bad += std::memcmp(rb.data(), ring_host.arrs[ai].h, written) != 0;
        // 2. every resident slot equals the block it holds
        std::vector<uint8_t> sb((size_t) n_slots * per);
        ck(cudaMemcpy(sb.data(), slots.arrs[ai].d, sb.size(), cudaMemcpyDeviceToHost), "slots d2h");
        std::vector<int32_t> pt(n_blocks);
        ck(cudaMemcpy(pt.data(), m.page_table, n_blocks * 4, cudaMemcpyDeviceToHost), "pt");
        for (int64_t bk = 0; bk < n_blocks; ++bk)
            if (pt[bk] >= 0) {
                if (ai == 0) ++resident;
                slot_bad += std::memcmp(sb.data() + (size_t) pt[bk] * per, rb.data() + (size_t) bk * per, per) != 0;
            }
    }
    if (host_bad) fail(name, "a host copy differs from the resident reference", host_bad);
    if (slot_bad) fail(name, "a resident slot differs from its block", slot_bad);
    const k::KvStreamCounters c = k::kv_stream_counters(m);
    std::printf("  %-5s %-6s: %d windows (%d attended), %llu block lookups, %llu misses, %lld resident, overflow %d, host/ring-host bad %lld, stale slots %lld\n",
                name, graphs ? "graph" : "direct", windows, attended, (unsigned long long) c.lookups, (unsigned long long) c.misses,
                resident, (int) c.overflow, host_bad, slot_bad);
    if (c.overflow || c.misses == 0 || c.misses >= c.lookups) fail(name, "the stream counters are off", (long long) c.misses, (long long) c.lookups);
    for (int n = 1; n <= kMaxWin; ++n) {
        if (g_app[n]) (void) cudaGraphExecDestroy(g_app[n]);
        if (g_att[n]) (void) cudaGraphExecDestroy(g_att[n]);
    }
    (void) cudaStreamDestroy(cs);
    return g_bad == 0;
}
}  // namespace

int main() {
    std::printf("kv_window_parity: verify-window KV calls on a streamed layer and on a ring, plain launches and graph replays\n");
    bool ok = true;
    for (int fmt : {k::kKvQ4, k::kKvInt8, k::kKvHybrid})
        for (bool graphs : {false, true}) ok = run(fmt, graphs) && ok;
    std::printf(ok && g_bad == 0 ? "PASS\n" : "FAIL\n");
    return ok && g_bad == 0 ? 0 : 1;
}
