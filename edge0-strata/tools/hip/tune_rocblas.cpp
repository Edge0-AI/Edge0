// tools/hip/tune_rocblas.cpp - a rocBLAS tuning table for the prompt path's FP16-out GEMMs (gfx103x, prompt_f16()).
//
// On RDNA2 the prompt path's dense products run as rocBLAS FP16 in -> FP16 out (docs/AMD_HIP.md, "RDNA2").  rocBLAS
// picks its kernel from a table by shape, and for some shapes that pick is far from the best kernel the library has:
// on an RX 6900 XT, N=10240 K=2560 (the GDN input projection) below T=1152 tokens runs at 5 TFLOPS where the best of
// rocBLAS's own 237 solutions for the same shape runs at 32 (E270, 2026-10-05).  This tool enumerates the library's
// solutions for each shape (rocblas_gemm_ex_get_solutions), times them, checks each against the default kernel's
// output, and writes the rows where a solution beats the default by at least --min-gain (1.05 = 5%) as a table the
// engine reads through STRATA_ROCBLAS_TUNING (src/prefill/rocblas_tuning.hpp; Gemm::f16_inplace).
//
// Solution indices are valid only for one GPU architecture and one rocBLAS build (its version string, tweak hash
// included), so a table is calibrated on the card with the ROCm the engine runs with:
//
//   cmake --build build-hip --target tune_rocblas
//   ./build-hip/tune_rocblas --tuning-out tools/hip/<arch>-rocblas-<version>.txt
//
// Defaults: the engine's 14 FP16 dense shapes at T = 256 ... 8192 (--tokens); --case T,N,K,ldy adds one shape.
#define ROCBLAS_BETA_FEATURES_API
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {
#define HIP_CHECK(call) do { const hipError_t e = (call); if (e != hipSuccess) { \
    std::fprintf(stderr, "HIP %s:%d: %s: %s\n", __FILE__, __LINE__, #call, hipGetErrorString(e)); std::exit(2); } } while (0)
#define RB_CHECK(call) do { const rocblas_status e = (call); if (e != rocblas_status_success) { \
    std::fprintf(stderr, "rocBLAS %s:%d: %s: status=%d\n", __FILE__, __LINE__, #call, (int) e); std::exit(2); } } while (0)

struct Buffer {
    void* p = nullptr;
    explicit Buffer(size_t bytes) { if (bytes) HIP_CHECK(hipMalloc(&p, bytes)); }
    ~Buffer() { if (p) (void) hipFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

struct Shape { int t, n, k, ldy; };
struct Options {
    std::vector<Shape> shapes;
    std::vector<int> tokens{256, 512, 768, 1024, 1536, 2048, 4096, 8192};
    std::string tuning_out;
    double min_gain = 1.05;
    bool explicit_shapes = false;
};
struct Err { double rel_l2 = 0, max_abs = 0; bool finite = true, ok = true; size_t padding_writes = 0; };
struct Row { Shape s; bool tuned; int solution; float default_ms, best_ms; };

constexpr int WARMUPS = 2, REPS = 3, TOP = 8;
constexpr uint16_t CANARY = 0x7BFF;   // the largest finite FP16 (65504): never a product of (-1, 1) inputs

uint16_t f16_bits(float x) { const __half h = __float2half_rn(x); uint16_t b; std::memcpy(&b, &h, 2); return b; }
float f16_value(uint16_t b) { __half h; std::memcpy(&h, &b, 2); return __half2float(h); }

std::vector<std::string> csv(const std::string& s) {
    std::vector<std::string> out; size_t b = 0;
    for (;;) {
        const size_t p = s.find(',', b);
        out.push_back(s.substr(b, p == std::string::npos ? p : p - b));
        if (p == std::string::npos) return out;
        b = p + 1;
    }
}
int positive(const std::string& s, const char* label) {
    char* e = nullptr; const long v = std::strtol(s.c_str(), &e, 10);
    if (s.empty() || !e || *e || v < 1 || v > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "invalid %s: %s\n", label, s.c_str()); std::exit(2);
    }
    return (int) v;
}
void usage(const char* p) {
    std::printf("Usage: %s [--case T,N,K,ldy]... [--tokens T1,T2,...] [--min-gain 1.05] [--tuning-out PATH]\n", p);
    std::printf("Defaults: the engine's 14 FP16 dense shapes (ldy = N) at T = 256,512,768,1024,1536,2048,4096,8192.\n");
}
Options options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing option value\n"); std::exit(2); }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") { usage(argv[0]); std::exit(0); }
        else if (a == "--case") {
            const auto f = csv(next());
            if (f.size() != 4) { std::fprintf(stderr, "--case expects T,N,K,ldy\n"); std::exit(2); }
            const Shape s{positive(f[0], "T"), positive(f[1], "N"), positive(f[2], "K"), positive(f[3], "ldy")};
            if (s.ldy < s.n) { std::fprintf(stderr, "ldy must be >= N\n"); std::exit(2); }
            o.shapes.push_back(s); o.explicit_shapes = true;
        } else if (a == "--tokens") {
            o.tokens.clear();
            for (const auto& x : csv(next())) o.tokens.push_back(positive(x, "token bucket"));
        } else if (a == "--min-gain") { o.min_gain = std::atof(next().c_str()); if (o.min_gain < 1.0) o.min_gain = 1.0; }
        else if (a == "--tuning-out") o.tuning_out = next();
        else { std::fprintf(stderr, "unknown option: %s\n", a.c_str()); usage(argv[0]); std::exit(2); }
    }
    if (!o.explicit_shapes) {
        // Qwen3.8-Flash-Next's dense FP16 products, prompt_f16() on (the hc read, GDN and QSA projections, the router
        // and the shared expert), ldy = N for every one of them.  N K as the engine calls Gemm (ROCBLAS_LAYER=2).
        const int nk[][2] = {{10240, 2560}, {12288, 2560}, {6144, 2560}, {2560, 6144}, {2560, 640}, {2560, 2560},
                             {640, 2560}, {512, 2560}, {128, 2560}, {48, 2560}, {320, 10240}, {10240, 320},
                             {4, 10240}, {1, 2560}};
        for (const auto& p : nk) for (int t : o.tokens) o.shapes.push_back({t, p[0], p[1], p[0]});
    }
    return o;
}

float time_ms(hipStream_t stream, hipEvent_t a, hipEvent_t b, const std::function<rocblas_status()>& f, int reps,
              bool& ok) {
    HIP_CHECK(hipEventRecord(a, stream));
    for (int i = 0; i < reps; ++i) if (f() != rocblas_status_success) { ok = false; return 0; }
    HIP_CHECK(hipEventRecord(b, stream));
    HIP_CHECK(hipEventSynchronize(b));
    float ms = 0; HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    ok = true;
    return ms / reps;
}

// The candidate's FP16 output against the default kernel's, over the N x T results; the rest of each 2*ldy-half column
// (what the engine's widen_rows_f16 does not cover) must still hold the canary.
Err compare(const std::vector<uint16_t>& ref, const std::vector<uint16_t>& got, const Shape& s) {
    Err e; long double d2 = 0, r2 = 0; float max_ref = 0;
    const size_t ldc = (size_t) s.ldy * 2;
    for (int col = 0; col < s.t; ++col) {
        for (int row = 0; row < s.n; ++row) {
            const float r = f16_value(ref[col * ldc + row]), g = f16_value(got[col * ldc + row]);
            if (!std::isfinite(r) || !std::isfinite(g)) { e.finite = false; continue; }
            const double d = (double) g - r;
            d2 += (long double) d * d; r2 += (long double) r * r;
            e.max_abs = std::max(e.max_abs, std::abs(d)); max_ref = std::max(max_ref, std::abs(r));
        }
        for (size_t row = s.n; row < ldc; ++row) if (got[col * ldc + row] != CANARY) ++e.padding_writes;
    }
    e.rel_l2 = std::sqrt((double) (d2 / std::max(r2, 1e-300L)));
    // the two kernels sum K products in different orders, then both round to FP16: a few last-bit differences
    const float ulp = std::ldexp(1.0f, (int) std::floor(std::log2(std::max(max_ref, 1e-30f))) - 10);
    e.ok = e.finite && e.padding_writes == 0 && e.rel_l2 <= 2e-3 && e.max_abs <= 4.0 * ulp;
    return e;
}
}  // namespace

int main(int argc, char** argv) {
    const Options o = options(argc, argv);
    rocblas_handle h; RB_CHECK(rocblas_create_handle(&h));
    hipStream_t stream; HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    RB_CHECK(rocblas_set_stream(h, stream));
    char ver[128] = {0}; RB_CHECK(rocblas_get_version_string(ver, sizeof ver));
    int device = 0; hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDevice(&device)); HIP_CHECK(hipGetDeviceProperties(&prop, device));
    std::string arch(prop.gcnArchName); if (const auto c = arch.find(':'); c != std::string::npos) arch.resize(c);
    std::printf("meta device_arch=%s rocblas_version=%s shapes=%zu min_gain=%.3f\n", arch.c_str(), ver, o.shapes.size(), o.min_gain);
    hipEvent_t ea, eb; HIP_CHECK(hipEventCreate(&ea)); HIP_CHECK(hipEventCreate(&eb));

    std::vector<Row> rows;
    for (size_t ci = 0; ci < o.shapes.size(); ++ci) {
        const Shape& s = o.shapes[ci];
        const size_t na = (size_t) s.n * s.k, nb = (size_t) s.t * s.k, ldc = (size_t) s.ldy * 2, nc = ldc * s.t;
        std::vector<uint16_t> ha(na), hb(nb), canary(nc, CANARY);
        std::mt19937 rng(0x5A17C3u + (uint32_t) ci * 0x10001u);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& x : ha) x = f16_bits(dist(rng));
        for (auto& x : hb) x = f16_bits(dist(rng));
        Buffer da(na * 2), db(nb * 2), dc(nc * 2);
        HIP_CHECK(hipMemcpy(da.p, ha.data(), na * 2, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(db.p, hb.data(), nb * 2, hipMemcpyHostToDevice));
        const float alpha = 1.0f, beta = 0.0f;
        // the engine's call (Gemm::f16_inplace): C[N x T] = W^T . X, FP16 out, ldc = 2 * ldy halves
        auto gemm = [&](rocblas_gemm_algo algo, int32_t sol) {
            return rocblas_gemm_ex(h, rocblas_operation_transpose, rocblas_operation_none, s.n, s.t, s.k, &alpha,
                                   da.p, rocblas_datatype_f16_r, s.k, db.p, rocblas_datatype_f16_r, s.k, &beta, dc.p,
                                   rocblas_datatype_f16_r, (int) ldc, dc.p, rocblas_datatype_f16_r, (int) ldc,
                                   rocblas_datatype_f32_r, algo, sol, 0);
        };
        bool ok = true;
        HIP_CHECK(hipMemcpy(dc.p, canary.data(), nc * 2, hipMemcpyHostToDevice));
        for (int i = 0; i < WARMUPS; ++i) RB_CHECK(gemm(rocblas_gemm_algo_standard, 0));
        const float default_ms = time_ms(stream, ea, eb, [&] { return gemm(rocblas_gemm_algo_standard, 0); }, REPS, ok);
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<uint16_t> ref(nc);
        HIP_CHECK(hipMemcpy(ref.data(), dc.p, nc * 2, hipMemcpyDeviceToHost));
        const Err self = compare(ref, ref, s);
        std::printf("case T=%d N=%d K=%d ldy=%d default_ms=%.4f default_padding_writes=%zu\n", s.t, s.n, s.k, s.ldy,
                    default_ms, self.padding_writes);

        rocblas_int count = 0;
        auto solutions = [&](rocblas_int* list, rocblas_int* n) {
            return rocblas_gemm_ex_get_solutions(h, rocblas_operation_transpose, rocblas_operation_none, s.n, s.t, s.k,
                                                 &alpha, da.p, rocblas_datatype_f16_r, s.k, db.p, rocblas_datatype_f16_r,
                                                 s.k, &beta, dc.p, rocblas_datatype_f16_r, (int) ldc, dc.p,
                                                 rocblas_datatype_f16_r, (int) ldc, rocblas_datatype_f32_r,
                                                 rocblas_gemm_algo_solution_index, 0, list, n);
        };
        if (solutions(nullptr, &count) != rocblas_status_success || count <= 0) {
            std::printf("no_solutions T=%d N=%d K=%d ldy=%d\n", s.t, s.n, s.k, s.ldy);
            continue;
        }
        std::vector<rocblas_int> ids(count);
        RB_CHECK(solutions(ids.data(), &count));
        // pass 1: one timed call each; pass 2: the TOP fastest get the full measurement and the output check
        std::vector<std::pair<float, rocblas_int>> quick;
        for (rocblas_int id : ids) {
            if (gemm(rocblas_gemm_algo_solution_index, id) != rocblas_status_success) continue;
            const float ms = time_ms(stream, ea, eb, [&] { return gemm(rocblas_gemm_algo_solution_index, id); }, 1, ok);
            if (ok) quick.push_back({ms, id});
        }
        std::sort(quick.begin(), quick.end());
        rocblas_int best = 0; float best_ms = std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < quick.size() && i < (size_t) TOP; ++i) {
            const rocblas_int id = quick[i].second;
            HIP_CHECK(hipMemcpy(dc.p, canary.data(), nc * 2, hipMemcpyHostToDevice));
            for (int w = 0; w < WARMUPS; ++w) if (gemm(rocblas_gemm_algo_solution_index, id) != rocblas_status_success) { ok = false; break; }
            if (!ok) continue;
            const float ms = time_ms(stream, ea, eb, [&] { return gemm(rocblas_gemm_algo_solution_index, id); }, REPS, ok);
            if (!ok) continue;
            HIP_CHECK(hipDeviceSynchronize());
            std::vector<uint16_t> got(nc);
            HIP_CHECK(hipMemcpy(got.data(), dc.p, nc * 2, hipMemcpyDeviceToHost));
            Err e = compare(ref, got, s);
            // the engine applies a row to every T nearest its bucket, and a kernel can refuse a T it was not
            // measured at (a tile or grid predicate): the solution must also run at odd, smaller token counts
            bool accepts_neighbors = true;
            for (int tn : {s.t - 1, s.t - 37, s.t - 63, s.t / 2 + 1}) {
                if (tn < 1) continue;
                const rocblas_status st = rocblas_gemm_ex(
                    h, rocblas_operation_transpose, rocblas_operation_none, s.n, tn, s.k, &alpha, da.p,
                    rocblas_datatype_f16_r, s.k, db.p, rocblas_datatype_f16_r, s.k, &beta, dc.p, rocblas_datatype_f16_r,
                    (int) ldc, dc.p, rocblas_datatype_f16_r, (int) ldc, rocblas_datatype_f32_r,
                    rocblas_gemm_algo_solution_index, id, 0);
                if (st != rocblas_status_success) { accepts_neighbors = false; break; }
            }
            HIP_CHECK(hipDeviceSynchronize());
            e.ok = e.ok && accepts_neighbors;
            std::printf("  solution=%d ms=%.4f gain=%.2f relative_l2=%.3g max_abs=%.3g padding_writes=%zu %s%s\n", (int) id,
                        ms, default_ms / ms, e.rel_l2, e.max_abs, e.padding_writes, e.ok ? "ok" : "REJECT",
                        accepts_neighbors ? "" : " (refuses a smaller T)");
            if (e.ok && ms < best_ms) { best = id; best_ms = ms; }
        }
        if (std::isfinite(best_ms) && default_ms / best_ms >= o.min_gain) {
            std::printf("best T=%d N=%d K=%d ldy=%d solution=%d ms=%.4f default_ms=%.4f gain=%.2f\n", s.t, s.n, s.k, s.ldy,
                        (int) best, best_ms, default_ms, default_ms / best_ms);
            rows.push_back({s, true, (int) best, default_ms, best_ms});
        } else {
            // written too: the engine applies a row to the token counts nearest its bucket, and a `default` row keeps
            // a neighbouring bucket's solution from reaching token counts where the default kernel measured best
            std::printf("keep_default T=%d N=%d K=%d ldy=%d (%zu solutions; best %.4f ms vs default %.4f)\n", s.t, s.n, s.k,
                        s.ldy, quick.size(), best_ms, default_ms);
            rows.push_back({s, false, 0, default_ms, best_ms});
        }
    }

    if (!o.tuning_out.empty()) {
        std::ofstream f(o.tuning_out);
        if (!f) { std::fprintf(stderr, "cannot write tuning output: %s\n", o.tuning_out.c_str()); return 2; }
        f << "# rocBLAS solution indices for the prompt path's FP16-out GEMMs; valid for this architecture and rocBLAS build only\n";
        f << "# rows: type N K ldy T solution   (type f16o: FP16 in, FP16 out, FP32 compute; Gemm::f16_inplace)\n";
        f << "# solution: rocBLAS's solution index, or `default` where its own choice measured best at that bucket\n";
        f << "STRATA_ROCBLAS_TUNING_V1 " << arch << " " << ver << "\n";
        size_t tuned = 0;
        for (const auto& r : rows) {
            f << "f16o " << r.s.n << " " << r.s.k << " " << r.s.ldy << " " << r.s.t << " ";
            if (r.tuned) { f << r.solution << "   # " << r.default_ms << " ms -> " << r.best_ms << " ms\n"; ++tuned; }
            else f << "default   # " << r.default_ms << " ms\n";
        }
        std::printf("wrote %zu rows (%zu with a solution) to %s\n", rows.size(), tuned, o.tuning_out.c_str());
    }
    HIP_CHECK(hipEventDestroy(ea)); HIP_CHECK(hipEventDestroy(eb));
    HIP_CHECK(hipStreamDestroy(stream)); RB_CHECK(rocblas_destroy_handle(h));
    return 0;
}
