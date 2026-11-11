// Standalone HIP INT8 QSA query-LDS-swizzle component probe.
// Adapted from the cell-probability probe; input generation and full-byte dump are unchanged.
// Build/link in the project with strata_kernels + hip::host (or its HIP runtime target).
// Usage: gfx906_qsa_query_swizzle_probe [nq=32] [ctx=65536] [reps=20] [output.bin] [--masked-pages]
// Output path may instead be supplied by STRATA_QSA_PROBE_OUT. It is required.
// Run candidate environment settings in SEPARATE processes: the dispatcher caches them.
// No prompt-attention entry point, architecture skip, model loading, or repository writes.
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <numeric>
#include <vector>

namespace k = strata::kernels;
namespace {
constexpr int64_t kBatch = 32;
constexpr size_t kGuard = 64;
constexpr uint64_t kSeed = 0x7161736150524f42ull;

[[noreturn]] void fail(const char* why) {
    std::fprintf(stderr, "qsa_query_swizzle_probe: %s\n", why);
    std::exit(2);
}
void ck(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
        std::exit(2);
    }
}
#define HIP_CHECK(x) ck((x), #x)

// Fixed integer generator and exact binary fractions avoid library-dependent
// normal distributions and never inspect the candidate environment setting.
struct Rng {
    uint64_t state = kSeed;
    uint32_t next() {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return uint32_t((state * 2685821657736338717ull) >> 32);
    }
    size_t below(size_t n) { return size_t(next()) % n; }
    template <class T> void shuffle(std::vector<T>& v) {
        for (size_t n = v.size(); n > 1; --n) std::swap(v[n - 1], v[below(n)]);
    }
};

template <class T> struct Device {
    T* p = nullptr;
    explicit Device(size_t n) { HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&p), n * sizeof(T))); }
    explicit Device(const std::vector<T>& v) : Device(v.size()) {
        HIP_CHECK(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    }
    ~Device() { if (p) (void) hipFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};

uint64_t hash_bytes(uint64_t hash, const void* ptr, size_t bytes) {
    const auto* p = static_cast<const unsigned char*>(ptr);
    for (size_t i = 0; i < bytes; ++i) hash = (hash ^ p[i]) * 1099511628211ull;
    return hash;
}
template <class T> void hash_vector(uint64_t& hash, const std::vector<T>& v) {
    hash = hash_bytes(hash, v.data(), v.size() * sizeof(T));
}
int64_t positive(const char* text) {
    errno = 0;
    char* end = nullptr;
    const long long n = std::strtoll(text, &end, 10);
    if (errno || end == text || *end || n <= 0 || n > INT32_MAX)
        fail("nq, ctx, and reps must be positive int32 values");
    return int64_t(n);
}
const char* env(const char* name) {
    const char* value = std::getenv(name);
    return value ? value : "<unset>";
}

std::vector<float> read_checked(const Device<float>& storage, size_t count) {
    std::vector<float> host(count + 2 * kGuard);
    HIP_CHECK(hipMemcpy(host.data(), storage.p, host.size() * sizeof(float), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < host.size(); ++i) {
        if (i >= kGuard && i < kGuard + count) {
            if (!std::isfinite(host[i])) {
                std::fprintf(stderr, "nonfinite or unwritten output at float %zu\n", i - kGuard);
                fail("output check failed");
            }
        } else {
            uint32_t bits;
            std::memcpy(&bits, &host[i], sizeof(bits));
            if (bits != UINT32_MAX) fail("output guard changed");
        }
    }
    return std::vector<float>(host.begin() + kGuard, host.begin() + kGuard + count);
}

int run(int64_t nq, int64_t ctx, int reps, const char* output, bool masked) {
    if (ctx < nq) fail("ctx must be >= nq (query positions are ctx-nq through ctx-1)");
    const char* lane = std::getenv("STRATA_ATTN_LANECELL");
    if (lane && lane[0] == '1') fail("unset STRATA_ATTN_LANECELL or set it to 0 for this probe");
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                  "probe requires IEEE binary32 output");
    const k::QsaShapes s = k::qsa_real_shapes();
    // Match prefill.cpp: use the model's actual 4-cell page layout.
    if (s.n_head != 24 || s.n_head_kv != 2 || s.head_dim != 256 || s.page_size != 4)
        fail("unexpected model geometry");
    const int64_t hd = s.head_dim, nh = s.n_head, nkvh = s.n_head_kv, ps = s.page_size;
    const int64_t pages = (ctx + ps - 1) / ps, rows = pages * nkvh * ps;
    const int64_t cap = k::qsa_selection_width(ctx, s);
    const size_t count = size_t(nq) * size_t(nh * hd);
    Rng rng;
    std::vector<int8_t> kq(size_t(rows * hd)), vq(kq.size());
    std::vector<uint16_t> ks(size_t(rows * (hd / 64))), vs(ks.size());
    for (auto* v : {&kq, &vq})
        for (auto& x : *v) x = int8_t(int(rng.next() % 255) - 127);
    // Positive finite binary16 scales, directly encoded: [2^-8, 2^-5).
    for (auto* v : {&ks, &vs})
        for (auto& x : *v) x = uint16_t(0x1c00u + rng.next() % 0x0c00u);
    std::vector<int32_t> table(size_t(pages), 0);
    std::iota(table.begin(), table.end(), 0);
    rng.shuffle(table);
    const auto physical = table;
    if (masked)
        for (int64_t page = 1; page < pages; page += 3) table[size_t(page)] = -1;

    // The decode kernel reads ids[c0+t] only for t<n_here, and returns before
    // reading IDs for empty chunks. Unused row slots are nevertheless zeroed;
    // a final 64-cell initialized guard also avoids an exact allocation end.
    // cap remains the actual row stride; no per-row padding is passed as cap.
    std::vector<int32_t> ids(size_t(nq * cap) + kGuard, 0);
    std::vector<int32_t> steps(size_t(nq * k::kStepCount), 0);
    std::vector<float> query(count);
    for (auto& x : query) x = float(int(rng.next() % 4096) - 2048) * (1.0f / 1024.0f);
    std::vector<int32_t> old_cells;
    for (int64_t i = 0; i < nq; ++i) {
        const int64_t pos = ctx - nq + i, nkv = pos + 1;
        const int64_t width = k::qsa_selection_width(nkv, s);
        k::qsa_step_fill(steps.data() + i * k::kStepCount, pos, s);
        int32_t* sel = ids.data() + i * cap;
        if (width == nkv) {
            std::iota(sel, sel + width, 0);
        } else {
            // Same selection pattern as qsa_prompt_attn_parity: a recent
            // 512-cell window plus a sorted older set drifting ~3% per query.
            constexpr int64_t recent = 512;
            const int64_t older = width - recent;
            if (int64_t(old_cells.size()) != older) {
                std::vector<int32_t> all(size_t(nkv - recent), 0);
                std::iota(all.begin(), all.end(), 0);
                rng.shuffle(all);
                old_cells.assign(all.begin(), all.begin() + older);
            } else {
                for (int64_t r = 0; r < older / 32; ++r) {
                    const int32_t cell = int32_t(rng.below(size_t(nkv - recent)));
                    if (std::find(old_cells.begin(), old_cells.end(), cell) == old_cells.end())
                        old_cells[rng.below(size_t(older))] = cell;
                }
            }
            std::copy(old_cells.begin(), old_cells.end(), sel);
            std::iota(sel + older, sel + width, int32_t(nkv - recent));
            std::sort(sel, sel + width);
        }
        bool valid_page = false;
        for (int64_t c = 0; c < width; ++c) {
            if (sel[c] < 0 || sel[c] >= nkv || (c && sel[c - 1] >= sel[c]))
                fail("selection must be unique, sorted, and causal");
            valid_page |= table[size_t(sel[c] / ps)] >= 0;
        }
        // Restore a selected logical page if necessary, never allow an
        // all-masked query to make a vacuous zero-output equality check pass.
        if (!valid_page) {
            const size_t page = size_t(sel[0] / ps);
            table[page] = physical[page];
        }
    }
    size_t masked_selected = 0;
    for (int64_t i = 0; i < nq; ++i)
        for (int32_t c = 0; c < steps[size_t(i * k::kStepCount + k::kStepWidth)]; ++c)
            masked_selected += table[size_t(ids[size_t(i * cap + c)] / ps)] < 0;
    uint64_t input_hash = 14695981039346656037ull;
    hash_vector(input_hash, kq); hash_vector(input_hash, vq);
    hash_vector(input_hash, ks); hash_vector(input_hash, vs);
    hash_vector(input_hash, table); hash_vector(input_hash, ids);
    hash_vector(input_hash, steps); hash_vector(input_hash, query);

    Device<int8_t> dk(kq), dv(vq);
    Device<uint16_t> dks(ks), dvs(vs);
    Device<int32_t> dtable(table), dids(ids), dsteps(steps);
    Device<float> dq(query), dout(count + 2 * kGuard);
    const uint64_t stride = k::qsa_decode_attn_scratch_floats(cap, s);
    Device<float> scratch(size_t(std::min(kBatch, nq)) * size_t(stride));
    k::QsaAttnPools pools;
    pools.k_q = dk.p; pools.v_q = dv.p; pools.k_scale = dks.p; pools.v_scale = dvs.p;
    pools.page_table = dtable.p;
    HIP_CHECK(hipMemset(dout.p, 0xff, (count + 2 * kGuard) * sizeof(float)));
    HIP_CHECK(hipMemset(scratch.p, 0xff, size_t(std::min(kBatch, nq)) * size_t(stride) * sizeof(float)));
    auto launch = [&] {
        for (int64_t t = 0; t < nq; t += kBatch)
            k::qsa_decode_attn_batch(dq.p + t * nh * hd, pools, dids.p + t * cap,
                dsteps.p + t * k::kStepCount, cap, s, scratch.p, dout.p + kGuard + t * nh * hd,
                std::min(kBatch, nq - t), nullptr);
        HIP_CHECK(hipGetLastError());
    };
    launch();
    HIP_CHECK(hipDeviceSynchronize());
    const auto first = read_checked(dout, count);
    constexpr int warmups = 3;
    for (int i = 0; i < warmups; ++i) launch();
    HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t begin = nullptr, end = nullptr;
    HIP_CHECK(hipEventCreate(&begin)); HIP_CHECK(hipEventCreate(&end));
    std::vector<float> times(size_t(reps), 0.0f);
    for (int i = 0; i < reps; ++i) {
        // Excluded from event duration; the last timed invocation must also
        // overwrite every logical output element instead of inheriting it.
        HIP_CHECK(hipMemsetAsync(dout.p + kGuard, 0xff, count * sizeof(float), nullptr));
        HIP_CHECK(hipEventRecord(begin, nullptr));
        launch();
        HIP_CHECK(hipEventRecord(end, nullptr));
        HIP_CHECK(hipEventSynchronize(end));
        HIP_CHECK(hipEventElapsedTime(&times[size_t(i)], begin, end));
    }
    const auto result = read_checked(dout, count);
    if (std::memcmp(first.data(), result.data(), count * sizeof(float)) != 0)
        fail("first and final invocation differ byte-for-byte within this process");
    const uint64_t output_hash = hash_bytes(14695981039346656037ull, result.data(), count * sizeof(float));
    std::FILE* file = std::fopen(output, "wb");
    if (!file) fail("cannot open output file");
    const size_t written = std::fwrite(result.data(), sizeof(float), count, file);
    const int close_status = std::fclose(file);
    if (written != count || close_status != 0) fail("cannot write complete output file");
    std::sort(times.begin(), times.end());
    const float median = reps % 2 ? times[size_t(reps / 2)]
        : 0.5f * (times[size_t(reps / 2 - 1)] + times[size_t(reps / 2)]);
    int device = 0;
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDevice(&device)); HIP_CHECK(hipGetDeviceProperties(&prop, device));
    std::printf("device=%s arch=%s nq=%lld ctx=%lld cap=%lld page_size=%lld batch=%lld reps=%d warmups=%d\n",
        prop.name, prop.gcnArchName, (long long)nq, (long long)ctx, (long long)cap,
        (long long)ps, (long long)kBatch, reps, warmups);
    std::printf("QUERY_SWIZZLE=%s LANECELL=%s seed=%016llx masked_pages=%zu masked_selected=%zu\n",
        env("STRATA_GFX906_ATTN_QUERY_SWIZZLE"), env("STRATA_ATTN_LANECELL"), (unsigned long long)kSeed,
        size_t(std::count(table.begin(), table.end(), -1)), masked_selected);
    std::printf("INPUT_FNV1A64=%016llx OUTPUT_FNV1A64=%016llx output_floats=%zu output_bytes=%zu\n",
        (unsigned long long)input_hash, (unsigned long long)output_hash, count, count * sizeof(float));
    std::printf("PASS finite_written_guards_and_repeat median_ms=%.6f min_ms=%.6f max_ms=%.6f us_per_query=%.6f output=%s\n",
        median, times.front(), times.back(), double(median) * 1000.0 / double(nq), output);
    HIP_CHECK(hipEventDestroy(begin)); HIP_CHECK(hipEventDestroy(end));
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        std::vector<const char*> args;
        bool masked = false;
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--masked-pages") == 0) masked = true;
            else args.push_back(argv[i]);
        }
        if (args.size() > 4) fail("usage: probe [nq] [ctx] [reps] [output.bin] [--masked-pages]");
        const int64_t nq = args.size() > 0 ? positive(args[0]) : 32;
        const int64_t ctx = args.size() > 1 ? positive(args[1]) : 65536;
        const int reps = args.size() > 2 ? int(positive(args[2])) : 20;
        const char* output = args.size() > 3 ? args[3] : std::getenv("STRATA_QSA_PROBE_OUT");
        if (!output || !*output) fail("pass output.bin as fourth positional argument or set STRATA_QSA_PROBE_OUT");
        return run(nq, ctx, reps, output, masked);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "qsa_query_swizzle_probe: %s\n", e.what());
        return 2;
    }
}
