// src/kernels/qsa_pattn_parity.cpp - STRATA_PATTN_SM60's score and value kernels (qsa_pattn.cu, PR #3943) against the
// decode kernels' output, bitwise (GPU, synthetic data, no model).
//
// The reference is what the prompt path reads on sm_6x today: the chunk pass of attn_chunk_kernel_pre75 (the Daily's
// chunk kernel, qsa_decode_attn_chunks_only with daily_chunk 1, which runs on any card) and attn_merge_kernel
// (qsa_decode_attn_merge_only, v2 0).  The candidate is qsa_pattn_batch_any_arch (no compute-capability test), so the
// same bytes can be checked on an Ampere card too: the two paths do the same operations in the same order, and a
// pass there shows the indexing, masking and merge are right.  The last word is a run on a Pascal card (P100): there
// the checked entry qsa_pattn_batch must take the shapes and the production qsa_decode_attn_batch must give the same
// bytes as the reference.
//
// Data: random int8 K/V codes, fp16 scales in a safe range, normal queries, a shuffled page table with ~3% unresident
// (-1) pages and one run of 64 unresident pages (so whole chunks are masked and some queries read nothing), selections
// shaped like the drafter's window, the indexer's (ascending with gaps) and unordered with repeats, caps 64 / 130 /
// 2051 / 2112 (1, 3 and 33 chunks), widths 0 .. cap, n_q up to 40 with a scratch that holds only 2 queries of the
// decode layout (so qsa_pattn_batch runs several batches).  The pattn scratch and the output are filled with
// junk before each call, so a read of a block that was not written shows up as a difference.
//
//   qsa_pattn_parity [--device N]     exit 0: every case is bit-identical; 77: no GPU or a build without the kernels
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_pattn.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_pattn_parity: %s: %s\n", what, cudaGetErrorString(e));
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
// positive fp16 scale in [1/64, 1/32): codes up to 127 give |x| < 4
uint16_t f16_scale_q8(uint32_t r) { return (uint16_t) ((9u << 10) | (r & 0x3FFu)); }

constexpr int64_t kCells = 40960;   // the cells this harness can address
constexpr int64_t kHole = 8192;     // 64 unresident pages in a row: cells [kHole, kHole + 256)

void selection(int32_t* ids, int64_t cap, int width, int shape, std::mt19937& rng) {
    const int64_t w = std::max<int64_t>(0, std::min<int64_t>(width, cap));
    if (shape == 0) {   // the drafter's window: consecutive cells, often across the unresident run
        int64_t start;
        if (rng() % 2 == 0) start = std::max<int64_t>(0, kHole - 64 * (int64_t) (rng() % 8));
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
    k::QsaAttnPools pools;
    int32_t* table = nullptr;
    float *q = nullptr, *scratch_ref = nullptr, *scratch_new = nullptr, *attn_ref = nullptr, *attn_new = nullptr;
    int32_t *ids = nullptr, *steps = nullptr;
    int64_t max_q = 0, max_cap = 0;
    std::vector<void*> to_free;

    template <typename T> T* keep(T* p) { to_free.push_back(p); return p; }

    void init(std::mt19937& rng, int64_t max_q_, int64_t max_cap_) {
        max_q = max_q_;
        max_cap = max_cap_;
        const int64_t PS = s.page_size, pages = kCells / PS, H = s.n_head_kv, D = s.head_dim;
        std::vector<int32_t> tab((size_t) pages);   // logical page -> physical page: a permutation, with holes
        for (int64_t i = 0; i < pages; ++i) tab[(size_t) i] = (int32_t) i;
        std::shuffle(tab.begin(), tab.end(), rng);
        for (int64_t i = 0; i < pages; ++i)
            if (rng() % 32u == 0) tab[(size_t) i] = -1;
        for (int64_t i = kHole / PS; i < (kHole + 256) / PS; ++i) tab[(size_t) i] = -1;
        table = keep(upload(tab));
        const int64_t rows = pages * H * PS;
        pools.page_table = table;
        std::vector<int8_t> c((size_t) (rows * D));
        for (auto& x : c) x = (int8_t) ((int) (rng() % 255u) - 127);
        pools.k_q = keep(upload(c));
        for (auto& x : c) x = (int8_t) ((int) (rng() % 255u) - 127);
        pools.v_q = keep(upload(c));
        std::vector<uint16_t> sc((size_t) (rows * (D / 64)));
        for (auto& x : sc) x = f16_scale_q8(rng());
        pools.k_scale = keep(upload(sc));
        for (auto& x : sc) x = f16_scale_q8(rng());
        pools.v_scale = keep(upload(sc));
        q = keep(dalloc<float>((size_t) (max_q * s.n_head * D)));
        ids = keep(dalloc<int32_t>((size_t) (max_q * max_cap)));
        steps = keep(dalloc<int32_t>((size_t) (max_q * k::kStepCount)));
        const size_t sf = (size_t) (k::qsa_decode_attn_scratch_floats(max_cap, s) * (uint64_t) max_q);
        scratch_ref = keep(dalloc<float>(sf));
        scratch_new = keep(dalloc<float>(sf));
        attn_ref = keep(dalloc<float>((size_t) (max_q * s.n_head * D)));
        attn_new = keep(dalloc<float>((size_t) (max_q * s.n_head * D)));
    }
    ~Harness() {
        for (void* p : to_free) cudaFree(p);
    }

    // returns 0 identical, 1 differs, -1 the candidate declined (build without the kernels)
    int run(int64_t cap, int n_q, int scr_q, const std::vector<int32_t>& widths, std::mt19937& rng, bool on_pascal,
            long long& nonzero) {
        const int64_t D = s.head_dim, NH = s.n_head;
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> hq((size_t) (n_q * NH * D));
        for (auto& x : hq) x = nd(rng);
        ck(cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice), "q");
        std::vector<int32_t> hid((size_t) (n_q * cap));
        std::vector<int32_t> hst((size_t) (n_q * k::kStepCount), 0);
        for (int i = 0; i < n_q; ++i) {
            selection(hid.data() + (size_t) i * cap, cap, widths[(size_t) i], (int) (rng() % 3u), rng);
            hst[(size_t) i * k::kStepCount + k::kStepWidth] = widths[(size_t) i];
        }
        ck(cudaMemcpy(ids, hid.data(), hid.size() * 4, cudaMemcpyHostToDevice), "ids");
        ck(cudaMemcpy(steps, hst.data(), hst.size() * 4, cudaMemcpyHostToDevice), "steps");
        const size_t n = (size_t) (n_q * NH * D);
        const size_t sbytes = (size_t) k::qsa_decode_attn_scratch_floats(cap, s) * (size_t) n_q * 4;

        // the reference: the pre-sm_75 chunk kernel and attn_merge_kernel
        ck(cudaMemset(scratch_ref, 0xFF, sbytes), "scratch_ref fill");
        ck(cudaMemset(attn_ref, 0xFF, n * 4), "attn_ref fill");
        k::qsa_decode_attn_chunks_only(q, pools, ids, steps, cap, s, scratch_ref, n_q, 1, nullptr);
        k::qsa_decode_attn_merge_only(steps, cap, s, scratch_ref, attn_ref, n_q, 0, nullptr);
        ck(cudaDeviceSynchronize(), "reference");
        std::vector<float> a(n), b(n);
        ck(cudaMemcpy(a.data(), attn_ref, n * 4, cudaMemcpyDeviceToHost), "read reference");
        nonzero = 0;
        for (size_t i = 0; i < n; ++i) {
            if (!std::isfinite(a[i])) { std::printf("FAIL: the reference has a non-finite value\n"); return 1; }
            if (a[i] != 0.0f) ++nonzero;
        }

        // the candidate: the scratch holds scr_q queries of the decode layout (the caller's attn_batch), junk in it
        ck(cudaMemset(scratch_new, 0xFF, sbytes), "scratch_new fill");
        ck(cudaMemset(attn_new, 0x7F, n * 4), "attn_new fill");
        const uint64_t scratch_floats = (uint64_t) scr_q * k::qsa_decode_attn_scratch_floats(cap, s);
        const bool took = k::qsa_pattn_batch_any_arch(q, pools, ids, steps, cap, s, scratch_new, scratch_floats, attn_new,
                                                      n_q, nullptr);
        if (!took) return -1;
        ck(cudaDeviceSynchronize(), "candidate");
        ck(cudaMemcpy(b.data(), attn_new, n * 4, cudaMemcpyDeviceToHost), "read candidate");
        bool same = std::memcmp(a.data(), b.data(), n * 4) == 0;
        if (!same) {
            size_t first = 0;
            while (first < n && std::memcmp(&a[first], &b[first], 4) == 0) ++first;
            std::printf("FAIL cap %lld n_q %d scratch %d queries, widths", (long long) cap, n_q, scr_q);
            for (int32_t w : widths) std::printf(" %d", w);
            std::printf(": first difference at query %lld head %lld dim %lld: %.9g vs %.9g\n",
                        (long long) (first / (NH * D)), (long long) (first / D % NH), (long long) (first % D), a[first],
                        b[first]);
        }
        if (on_pascal) {   // the checked entry takes the shapes, and the production path is the reference
            ck(cudaMemset(scratch_new, 0xFF, sbytes), "scratch_new refill");
            ck(cudaMemset(attn_new, 0x7F, n * 4), "attn_new refill");
            if (!k::qsa_pattn_batch(q, pools, ids, steps, cap, s, scratch_new, scratch_floats, attn_new, n_q, nullptr)) {
                std::printf("FAIL: qsa_pattn_batch declined on a compute capability 6 card (cap %lld n_q %d)\n",
                            (long long) cap, n_q);
                same = false;
            } else {
                ck(cudaDeviceSynchronize(), "checked entry");
                ck(cudaMemcpy(b.data(), attn_new, n * 4, cudaMemcpyDeviceToHost), "read checked entry");
                if (std::memcmp(a.data(), b.data(), n * 4) != 0) {
                    std::printf("FAIL: qsa_pattn_batch differs from the reference (cap %lld n_q %d)\n", (long long) cap, n_q);
                    same = false;
                }
            }
            ck(cudaMemset(scratch_new, 0xFF, sbytes), "scratch_new refill");
            k::qsa_decode_attn_batch(q, pools, ids, steps, cap, s, scratch_new, attn_new, n_q, nullptr);
            ck(cudaDeviceSynchronize(), "production");
            ck(cudaMemcpy(b.data(), attn_new, n * 4, cudaMemcpyDeviceToHost), "read production");
            if (std::memcmp(a.data(), b.data(), n * 4) != 0) {
                std::printf("FAIL: qsa_decode_attn_batch differs from the reference (cap %lld n_q %d)\n", (long long) cap,
                            n_q);
                same = false;
            }
        }
        return same ? 0 : 1;
    }
};
}  // namespace

int main(int argc, char** argv) {
    int device = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) device = std::atoi(argv[++i]);
        else {
            std::fprintf(stderr, "usage: qsa_pattn_parity [--device N]\n");
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
    const bool on_pascal = prop.major == 6;
    std::printf("qsa_pattn_parity on device %d (%s, %d.%d)%s\n", device, prop.name, prop.major, prop.minor,
                on_pascal ? ": the checked entry and the production path are compared too" : "");

    std::mt19937 rng(20261026u);
    Harness hx;
    hx.init(rng, 40, 2112);

    const int32_t base_widths[] = {0, 1, 63, 64, 65, 1023, 1024, 1025, 2051, 2112};
    const int n_base = (int) (sizeof(base_widths) / sizeof(base_widths[0]));
    long long cases = 0, failed = 0, nonzero_total = 0;
    for (int64_t cap : {(int64_t) 64, (int64_t) 130, (int64_t) 2051, (int64_t) 2112}) {
        for (int n_q : {1, 2, 3, 7, 40}) {
            for (int scr_q : {n_q, 2}) {
                if (scr_q > n_q) continue;
                for (int wc = 0; wc < n_base + 3; ++wc) {
                    std::vector<int32_t> widths;
                    for (int i = 0; i < n_q; ++i)
                        widths.push_back(std::min<int32_t>((int32_t) cap, wc < n_base ? base_widths[wc] : base_widths[rng() % (uint32_t) n_base]));
                    long long nz = 0;
                    const int r = hx.run(cap, n_q, scr_q, widths, rng, on_pascal, nz);
                    if (r < 0) {
                        std::puts("SKIP: this build has no sm_6x kernels (CUDA with STRATA_EXPERIMENTAL_SM60)");
                        return 77;
                    }
                    nonzero_total += nz;
                    ++cases;
                    if (r != 0) ++failed;
                }
            }
        }
    }
    std::printf("qsa_pattn_parity: %lld cases, %lld differ, %lld nonzero output values in all -> %s\n", cases, failed,
                nonzero_total, failed == 0 && nonzero_total > 0 ? "PASS" : "FAIL");
    return failed == 0 && nonzero_total > 0 ? 0 : 1;
}
