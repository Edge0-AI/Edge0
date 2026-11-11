// Recovered from the Claude Daily merge parity harness (lab/wip-0809), extended to the Daily's chunk kernel.
// src/kernels/qsa_merge_daily_parity.cpp - the Daily's decode attention (STRATA_ATTN_DAILY=1: attn_chunk_kernel_pre75
// and attn_merge_v2_kernel) against upstream's (attn_chunk_kernel, attn_merge_kernel), bitwise (GPU, synthetic, no
// model).
//
// The reference is upstream's pair: one chunk pass with attn_chunk_kernel (qsa_decode_attn_chunks_only, daily_chunk 0)
// and attn_merge_kernel (qsa_decode_attn_merge_only, v2 0).  Against it, memcmp of: attn_merge_v2_kernel on the same
// partials; the production entry points (qsa_decode_attn_batch, and _step at n_q 1) as the environment sets them; the
// Daily's chunk kernel (daily_chunk 1, NaN-filled scratch again) merged by both mergers.  Kernels are chosen by the
// arguments, never by the environment, except for the production entries.  Covers:
//   - the four KV formats the chunk kernel reads: fp16, int8, q4_0 (rotated or not, the merge cannot tell), k8v4;
//   - page tables with unresident (-1) pages: scattered single pages, and one run of 64 pages (256 consecutive cells)
//     that the window-shaped selections land on, so whole chunks inside the width are masked (m = -FLT_MAX, l = 0,
//     acc = 0) and some queries are masked entirely (output 0);
//   - widths {-1, 0, 1, 63, 64, 65, 1023, 1024, 1025, 2051, 4096, 16383, 16385, 32768, INT_MAX} per query (1023-1025 and 16383 /
//     16385 sit on the v2 merge's 16-chunk block edges), all queries alike and mixed within one batch; a width above
//     aligned cap (2112/32768) is bounded by the chunk grid; an unaligned cap2051 keeps width<=cap;
//   - n_q 1..6, caps2051/2112 (main selection and its rounded capacity) and32768 (the drafter's
//     default --mtp-window);
//   - selections shaped like the drafter's window (consecutive cells, window_ids), like the indexer's (ascending, with
//     gaps) and unordered with repeats.
// The scratch is filled with NaN before every chunk pass, so a merge that read a partial the chunk pass did not write
// (a chunk past the width, say) would differ from the other one or produce a NaN.  ~260 MB of VRAM.
//
//   qsa_merge_daily_parity [--device N]           the sweep; exit 0 when every case is bit-identical (77: no GPU)
//   qsa_merge_daily_parity [--device N] --bench   plus each merge's time per call at target/drafter caps2051/32768
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_merge_daily_parity: %s: %s\n", what, cudaGetErrorString(e));
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

// fp16 bits drawn directly: a random sign, exponent 11..15 (|x| in [2^-4, 2)), a random mantissa; finite by
// construction. Scales are positive: INT8 [1/64, 1/32) (codes up to 127 give |x| < 4), Q4_0 [1/4, 1/2).
uint16_t f16_value(uint32_t r) { return (uint16_t) ((r & 0x8000u) | ((11u + (r >> 16) % 5u) << 10) | (r & 0x3FFu)); }
uint16_t f16_scale_q8(uint32_t r) { return (uint16_t) ((9u << 10) | (r & 0x3FFu)); }
uint16_t f16_scale_q4(uint32_t r) { return (uint16_t) ((13u << 10) | (r & 0x3FFu)); }

constexpr int kFmtF16 = 0, kFmtInt8 = 1, kFmtQ4 = 2, kFmtK8V4 = 3;
const char* fmt_name(int f) { return f == kFmtF16 ? "fp16" : f == kFmtInt8 ? "int8" : f == kFmtQ4 ? "q4_0" : "k8v4"; }

struct DevicePools {   // every format's pools over the same physical rows; `view` picks what a KV mode reads
    uint16_t *k16 = nullptr, *v16 = nullptr, *k8s = nullptr, *v8s = nullptr;
    int8_t *k8 = nullptr, *v8 = nullptr;
    uint8_t *k4 = nullptr, *v4 = nullptr;
    k::QsaAttnPools view(int fmt, const int32_t* table) const {
        k::QsaAttnPools p;
        p.page_table = table;
        if (fmt == kFmtF16) { p.k_pool = k16; p.v_pool = v16; }
        else if (fmt == kFmtInt8) { p.k_q = k8; p.v_q = v8; p.k_scale = k8s; p.v_scale = v8s; }
        else if (fmt == kFmtQ4) { p.k_q4 = k4; p.v_q4 = v4; }
        else { p.k_q = k8; p.k_scale = k8s; p.v_q4 = v4; }   // K8V4: INT8 K, Q4_0 V (qsa_decode_attn.cu kv_mode 3)
        return p;
    }
};

DevicePools make_pools(int64_t rows, int64_t D, std::mt19937& rng) {
    DevicePools p;
    {
        std::vector<uint16_t> h((size_t) (rows * D));
        for (auto& x : h) x = f16_value(rng());
        p.k16 = upload(h);
        for (auto& x : h) x = f16_value(rng());
        p.v16 = upload(h);
    }
    {
        std::vector<int8_t> c((size_t) (rows * D));
        for (auto& x : c) x = (int8_t) ((int) (rng() % 255u) - 127);
        p.k8 = upload(c);
        for (auto& x : c) x = (int8_t) ((int) (rng() % 255u) - 127);
        p.v8 = upload(c);
        std::vector<uint16_t> sc((size_t) (rows * (D / k::KV_Q8_GROUP)));
        for (auto& x : sc) x = f16_scale_q8(rng());
        p.k8s = upload(sc);
        for (auto& x : sc) x = f16_scale_q8(rng());
        p.v8s = upload(sc);
    }
    {
        const size_t blocks = (size_t) (rows * (D / k::QK4_0));
        std::vector<uint8_t> b(blocks * sizeof(k::block_q4_0));
        for (int side = 0; side < 2; ++side) {
            for (size_t i = 0; i < blocks; ++i) {
                k::block_q4_0 blk;
                blk.d = f16_scale_q4(rng());
                for (auto& q : blk.qs) q = (uint8_t) (rng() & 0xFFu);
                std::memcpy(b.data() + i * sizeof(k::block_q4_0), &blk, sizeof(blk));
            }
            (side == 0 ? p.k4 : p.v4) = upload(b);
        }
    }
    return p;
}

struct Case {
    int fmt = 0;
    int64_t cap = 0;
    int n_q = 0;
    std::vector<int32_t> widths;
};

// the cells this harness can address, and where the 64 unresident pages in a row sit (cells [kHole, kHole + 256))
constexpr int64_t kCells = 40960;
constexpr int64_t kHole = 8192;

// query `i`'s `cap` selection ids. Every entry is a valid cell: a width above the cap makes the chunk kernel read
// all `cap` of them.
void selection(int32_t* ids, int64_t cap, int width, int shape, std::mt19937& rng) {
    const int64_t w = std::max<int64_t>(0, std::min<int64_t>(width, cap));
    if (shape == 0) {   // the drafter's window: consecutive cells (window_ids), often across the unresident run
        int64_t start;
        if (rng() % 2 == 0) start = std::max<int64_t>(0, kHole - 64 * (int64_t) (rng() % 8));   // chunk-aligned on it
        else start = (int64_t) (rng() % (uint32_t) (kCells - std::max<int64_t>(w, 1)));
        if (start + w > kCells) start = kCells - w;
        for (int64_t j = 0; j < cap; ++j) ids[j] = (int32_t) ((start + j) % kCells);
    } else if (shape == 1) {   // the indexer's: ascending with gaps
        std::vector<int32_t> v((size_t) cap);
        for (auto& x : v) x = (int32_t) (rng() % (uint32_t) kCells);
        std::sort(v.begin(), v.begin() + w);
        std::copy(v.begin(), v.end(), ids);
    } else {   // unordered, with repeats
        for (int64_t j = 0; j < cap; ++j) ids[j] = (int32_t) (rng() % (uint32_t) kCells);
    }
}

struct Harness {
    k::QsaShapes s = k::qsa_real_shapes();
    DevicePools pools;
    int32_t* table = nullptr;
    float *q = nullptr, *scratch = nullptr, *attn_old = nullptr, *attn_new = nullptr;
    int32_t *ids = nullptr, *steps = nullptr;
    int64_t max_q = 0, max_cap = 0;

    void init(std::mt19937& rng, int64_t max_q_, int64_t max_cap_) {
        max_q = max_q_;
        max_cap = max_cap_;
        const int64_t PS = s.page_size, pages = kCells / PS, H = s.n_head_kv, D = s.head_dim;
        // logical page -> physical page: a permutation, ~3% scattered unresident pages, and the 64-page hole
        std::vector<int32_t> tab((size_t) pages);
        for (int64_t i = 0; i < pages; ++i) tab[(size_t) i] = (int32_t) i;
        std::shuffle(tab.begin(), tab.end(), rng);
        for (int64_t i = 0; i < pages; ++i)
            if (rng() % 32u == 0) tab[(size_t) i] = -1;
        for (int64_t i = kHole / PS; i < (kHole + 256) / PS; ++i) tab[(size_t) i] = -1;
        table = upload(tab);
        pools = make_pools(pages * H * PS, D, rng);
        q = dalloc<float>((size_t) (max_q * s.n_head * D));
        ids = dalloc<int32_t>((size_t) (max_q * max_cap));
        steps = dalloc<int32_t>((size_t) (max_q * k::kStepCount));
        scratch = dalloc<float>((size_t) (k::qsa_decode_attn_scratch_floats(max_cap, s) * (uint64_t) max_q));
        attn_old = dalloc<float>((size_t) (max_q * s.n_head * D));
        attn_new = dalloc<float>((size_t) (max_q * s.n_head * D));
    }

    ~Harness() {
        for (void* p : {static_cast<void*>(table), static_cast<void*>(q), static_cast<void*>(ids),
                       static_cast<void*>(steps), static_cast<void*>(scratch), static_cast<void*>(attn_old),
                       static_cast<void*>(attn_new), static_cast<void*>(pools.k16), static_cast<void*>(pools.v16),
                       static_cast<void*>(pools.k8), static_cast<void*>(pools.v8), static_cast<void*>(pools.k8s),
                       static_cast<void*>(pools.v8s), static_cast<void*>(pools.k4), static_cast<void*>(pools.v4)})
            if (p) ck(cudaFree(p), "free harness");
    }

    // uploads the case's queries, selections and widths, and runs ONE chunk pass into NaN-filled scratch
    void prepare(const Case& c, std::mt19937& rng) {
        const int64_t D = s.head_dim, NH = s.n_head;
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> hq((size_t) (c.n_q * NH * D));
        for (auto& x : hq) x = nd(rng);
        ck(cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice), "q");
        std::vector<int32_t> hid((size_t) (c.n_q * c.cap));
        std::vector<int32_t> hst((size_t) (c.n_q * k::kStepCount), 0);
        for (int i = 0; i < c.n_q; ++i) {
            selection(hid.data() + (size_t) i * c.cap, c.cap, c.widths[(size_t) i], (int) (rng() % 3u), rng);
            hst[(size_t) i * k::kStepCount + k::kStepWidth] = c.widths[(size_t) i];
        }
        ck(cudaMemcpy(ids, hid.data(), hid.size() * 4, cudaMemcpyHostToDevice), "ids");
        ck(cudaMemcpy(steps, hst.data(), hst.size() * 4, cudaMemcpyHostToDevice), "steps");
        const size_t scratch_bytes = (size_t) k::qsa_decode_attn_scratch_floats(c.cap, s) * (size_t) c.n_q * 4;
        ck(cudaMemset(scratch, 0xFF, scratch_bytes), "scratch NaN fill");
        ck(cudaMemset(attn_old, 0xFF, (size_t) (c.n_q * NH * D) * 4), "attn_old fill");
        ck(cudaMemset(attn_new, 0x7F, (size_t) (c.n_q * NH * D) * 4), "attn_new fill");
        k::qsa_decode_attn_chunks_only(q, pools.view(c.fmt, table), ids, steps, c.cap, s, scratch, c.n_q, 0, nullptr);
    }

    // true when the two merges' outputs are bit-identical (and the original's has no NaN)
    bool check(const Case& c, long long& nonzero) {
        const int64_t D = s.head_dim, NH = s.n_head;
        k::qsa_decode_attn_merge_only(steps, c.cap, s, scratch, attn_old, c.n_q, 0, nullptr);
        k::qsa_decode_attn_merge_only(steps, c.cap, s, scratch, attn_new, c.n_q, 1, nullptr);
        ck(cudaDeviceSynchronize(), "merge sync");
        const size_t n = (size_t) (c.n_q * NH * D);
        std::vector<float> a(n), b(n);
        ck(cudaMemcpy(a.data(), attn_old, n * 4, cudaMemcpyDeviceToHost), "attn_old");
        ck(cudaMemcpy(b.data(), attn_new, n * 4, cudaMemcpyDeviceToHost), "attn_new");
        nonzero = 0;
        long long nan = 0;
        for (size_t i = 0; i < n; ++i) {
            if (a[i] != 0.0f) ++nonzero;
            if (!std::isfinite(a[i])) ++nan;
        }
        const bool same = std::memcmp(a.data(), b.data(), n * 4) == 0;
        if (!same || nan) {
            size_t first = 0;
            while (first < n && std::memcmp(&a[first], &b[first], 4) == 0) ++first;
            std::printf("FAIL %s cap %lld n_q %d widths", fmt_name(c.fmt), (long long) c.cap, c.n_q);
            for (int32_t w : c.widths) std::printf(" %d", w);
            if (first < n) {
                uint32_t ua, ub;
                std::memcpy(&ua, &a[first], 4);
                std::memcpy(&ub, &b[first], 4);
                std::printf(": first difference at query %lld head %lld dim %lld: %.9g (0x%08x) vs %.9g (0x%08x)",
                            (long long) (first / (NH * D)), (long long) (first / D % NH), (long long) (first % D),
                            a[first], ua, b[first], ub);
            }
            if (nan) std::printf(" (%lld nonfinite values in the original's output)", nan);
            std::printf("\n");
        }
        // Exercise production entry points too: either selected merger must equal the reference partials.
        bool entry_same = true;
        k::qsa_decode_attn_batch(q, pools.view(c.fmt, table), ids, steps, c.cap, s, scratch, attn_new, c.n_q, nullptr);
        ck(cudaDeviceSynchronize(), "batch entry sync");
        ck(cudaMemcpy(b.data(), attn_new, n * 4, cudaMemcpyDeviceToHost), "batch entry output");
        if (std::memcmp(a.data(), b.data(), n * 4) != 0) {
            std::puts("FAIL production batch differs from merge reference");
            entry_same = false;
        }
        if (c.n_q == 1) {
            k::qsa_decode_attn_step(q, pools.view(c.fmt, table), ids, steps, c.cap, s, scratch, attn_new, nullptr);
            ck(cudaDeviceSynchronize(), "step entry sync");
            ck(cudaMemcpy(b.data(), attn_new, n * 4, cudaMemcpyDeviceToHost), "step entry output");
            if (std::memcmp(a.data(), b.data(), n * 4) != 0) {
                std::puts("FAIL production step differs from merge reference");
                entry_same = false;
            }
        }
        // The Daily's chunk kernel on the same inputs, into NaN-filled scratch, merged by both mergers.
        bool daily_same = true;
        const size_t scratch_bytes = (size_t) k::qsa_decode_attn_scratch_floats(c.cap, s) * (size_t) c.n_q * 4;
        ck(cudaMemset(scratch, 0xFF, scratch_bytes), "scratch NaN refill");
        k::qsa_decode_attn_chunks_only(q, pools.view(c.fmt, table), ids, steps, c.cap, s, scratch, c.n_q, 1, nullptr);
        for (int v2 = 0; v2 < 2; ++v2) {
            ck(cudaMemset(attn_new, 0x7F, n * 4), "attn_new refill");
            k::qsa_decode_attn_merge_only(steps, c.cap, s, scratch, attn_new, c.n_q, v2, nullptr);
            ck(cudaDeviceSynchronize(), "daily chunk sync");
            ck(cudaMemcpy(b.data(), attn_new, n * 4, cudaMemcpyDeviceToHost), "daily chunk output");
            if (std::memcmp(a.data(), b.data(), n * 4) != 0) {
                std::printf("FAIL Daily chunk kernel + %s differs from upstream reference\n",
                            v2 ? "attn_merge_v2_kernel" : "attn_merge_kernel");
                daily_same = false;
            }
        }
        return same && nan == 0 && entry_same && daily_same;
    }
};

void bench(Harness& hx, std::mt19937& rng) {
    std::printf("\n--bench: CUDA events over 200 calls per arm, each arm warmed outside timing, same buffers.\n"
                "chunks BAB: B1(Daily chunk) -> A(upstream chunk) -> B2(Daily chunk); merges on hot partials:\n"
                "BAB: B1(v2) -> A(upstream merger) -> B2(v2).\n");
    std::printf("%-6s %6s %4s %6s | %9s %9s %9s %9s | %9s %9s %9s %9s\n", "fmt", "cap", "n_q", "width",
                "chB1 us", "chA us", "chB2 us", "meanB/A", "B1 us", "A us", "B2 us", "meanB/A");
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "event");
    ck(cudaEventCreate(&e1), "event");
    const int iters = 200;
    for (int64_t cap : {(int64_t) 2051, (int64_t) 32768})
    for (int fmt : {kFmtQ4, kFmtInt8, kFmtF16}) {
        for (int n_q : {1, 2, 3, 4}) {
            for (int width : {64, 2051, 16384, 31000, 32768}) {
                Case c;
                c.fmt = fmt;
                if (width > cap) continue;
                c.cap = cap;
                c.n_q = n_q;
                c.widths.assign((size_t) n_q, width);
                hx.prepare(c, rng);
                ck(cudaDeviceSynchronize(), "bench prepare");
                double us_ch[3] = {};
                for (int arm = 0; arm < 3; ++arm) {   // B1 -> A -> B2: Daily, upstream, Daily chunk kernel
                    const int daily = arm == 1 ? 0 : 1;
                    auto chunks = [&] {
                        k::qsa_decode_attn_chunks_only(hx.q, hx.pools.view(fmt, hx.table), hx.ids, hx.steps, c.cap,
                                                       hx.s, hx.scratch, n_q, daily, nullptr);
                    };
                    for (int i = 0; i < iters; ++i) chunks();
                    ck(cudaDeviceSynchronize(), "chunk warm");
                    ck(cudaEventRecord(e0, nullptr), "record");
                    for (int i = 0; i < iters; ++i) chunks();
                    ck(cudaEventRecord(e1, nullptr), "record");
                    ck(cudaEventSynchronize(e1), "sync");
                    float ms_chunks = 0.0f;
                    ck(cudaEventElapsedTime(&ms_chunks, e0, e1), "elapsed");
                    us_ch[arm] = 1000.0 * ms_chunks / iters;
                }
                const double ratio_ch = us_ch[1] > 0.0 ? (us_ch[0] + us_ch[2]) / (2.0 * us_ch[1]) : std::nan("");
                double us[3] = {};
                for (int arm = 0; arm < 3; ++arm) {   // B1 -> A -> B2 at this identical shape
                    const int v2 = arm == 1 ? 0 : 1;
                    auto merge = [&] {
                        k::qsa_decode_attn_merge_only(hx.steps, c.cap, hx.s, hx.scratch, hx.attn_new, n_q, v2, nullptr);
                    };
                    for (int i = 0; i < iters; ++i) merge();
                    ck(cudaDeviceSynchronize(), "merge arm warm");
                    ck(cudaEventRecord(e0, nullptr), "record");
                    for (int i = 0; i < iters; ++i) merge();
                    ck(cudaEventRecord(e1, nullptr), "record");
                    ck(cudaEventSynchronize(e1), "sync");
                    float ms = 0.0f;
                    ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
                    us[arm] = 1000.0 * ms / iters;
                }
                const double ratio = us[1] > 0.0 ? (us[0] + us[2]) / (2.0 * us[1]) : std::nan("");
                std::printf("%-6s %6lld %4d %6d | %9.2f %9.2f %9.2f %9.4f | %9.2f %9.2f %9.2f %9.4f\n",
                            fmt_name(fmt), (long long) cap, n_q, width, us_ch[0], us_ch[1], us_ch[2], ratio_ch, us[0],
                            us[1], us[2], ratio);
            }
        }
    }
    ck(cudaEventDestroy(e0), "destroy event");
    ck(cudaEventDestroy(e1), "destroy event");
}
}  // namespace

int main(int argc, char** argv) {
    int device = 0;
    bool do_bench = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0) do_bench = true;
        else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) device = std::atoi(argv[++i]);
        else {
            std::fprintf(stderr, "usage: qsa_merge_daily_parity [--device N] [--bench]\n");
            return 2;
        }
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: no GPU");
        return 77;
    }
    ck(cudaSetDevice(device), "cudaSetDevice");
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    std::printf("qsa_merge_daily_parity on device %d (%s, %d.%d)\n", device, prop.name, prop.major, prop.minor);

    std::mt19937 rng(20261004u);
    Harness hx;
    hx.init(rng, 7, 32768);

    // 1023 / 1024 / 1025 and 16383 / 16385: the v2 merge's U = 16 chunks x 64 cells block edges
    const int32_t widths[] = {-1, 0, 1, 63, 64, 65, 1023, 1024, 1025, 2051, 4096, 16383, 16384, 16385, 32768, std::numeric_limits<int32_t>::max()};
    const int n_widths = (int) (sizeof(widths) / sizeof(widths[0]));
    long long cases = 0, failed = 0, nonzero_total = 0;
    for (int fmt = 0; fmt < 4; ++fmt) {
        long long fmt_cases = 0, fmt_failed = 0;
        for (int64_t cap : {(int64_t) 2051, (int64_t) 2112, (int64_t) 32768}) {
            for (int n_q = 1; n_q <= 6; ++n_q) {
                // every width for all queries alike, then three batches of mixed widths
                for (int wc = 0; wc < n_widths + 3; ++wc) {
                    Case c;
                    c.fmt = fmt;
                    c.cap = cap;
                    c.n_q = n_q;
                    for (int i = 0; i < n_q; ++i)
                        c.widths.push_back(wc < n_widths ? widths[wc] : widths[rng() % (uint32_t) n_widths]);
                    // The unchanged chunk contract requires width<=cap for an unaligned last chunk.
                    // Overflow widths remain tested at aligned caps, where the grid bounds the reads.
                    if (cap % 64 != 0 && std::any_of(c.widths.begin(), c.widths.end(),
                            [cap](int32_t w) { return (int64_t) w > cap; })) continue;
                    hx.prepare(c, rng);
                    long long nz = 0;
                    const bool ok = hx.check(c, nz);
                    nonzero_total += nz;
                    ++cases;
                    ++fmt_cases;
                    if (!ok) { ++failed; ++fmt_failed; }
                }
            }
        }
        std::printf("[%s] %lld cases, %s (%lld differ)\n", fmt_name(fmt), fmt_cases, fmt_failed ? "FAIL" : "bit-identical",
                    fmt_failed);
    }
    std::printf("qsa_merge_daily_parity: %lld cases, %lld differ, %lld nonzero output values in all -> %s\n", cases, failed,
                nonzero_total, failed == 0 && nonzero_total > 0 ? "PASS" : "FAIL");
    if (do_bench) bench(hx, rng);
    return failed == 0 && nonzero_total > 0 ? 0 : 1;
}
