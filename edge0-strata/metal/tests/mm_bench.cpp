// metal/tests/mm_bench.cpp - matrix throughput (TFLOP/s) of MPP matmul2d and simdgroup_matrix, fp16/bf16/fp32
// operands, fp32 results, on a square 4096^3 and on prompt-expert shapes; checks one product against the CPU.
//   mm_bench
#include "mtl_rt.h"
#include "ref_quant.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
struct MmArgs { int M, N, K; };
double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
}  // namespace

int main() {
    mtl::Context cx;
    std::printf("device %s, metal4 %d\n", cx.device_name().c_str(), cx.supports_metal4());
    const int lib = cx.compile({"mm_bench.metal"}, "#define NO_GGML_COMMON 1\n");
    const int MAX = 4096;
    mtl::Buffer A = cx.alloc((size_t)MAX * MAX * 4), B = cx.alloc((size_t)MAX * MAX * 4), C = cx.alloc((size_t)MAX * MAX * 4);
    ref::Rng r(1);
    // fill as half and float views of the same random values (each kernel reads its own type)
    auto fill = [&](int elem) {
        for (size_t i = 0; i < (size_t)MAX * MAX; ++i) {
            const float v = r.sym();
            if (elem == 4) { A.as<float>()[i] = v; B.as<float>()[i] = r.sym(); }
            else if (elem == 2) { A.as<uint16_t>()[i] = ref::f32_to_f16(v); B.as<uint16_t>()[i] = ref::f32_to_f16(r.sym()); }
            else { A.as<uint16_t>()[i] = ref::f32_to_bf16(v); B.as<uint16_t>()[i] = ref::f32_to_bf16(r.sym()); }
        }
    };
    struct K { std::string name; int elem; int tm, tn, sg; bool mpp; };
    const std::vector<K> ks = {
        {"mpp_mm_half_64_32_4", 2, 64, 32, 4, true},  {"mpp_mm_half_64_64_4", 2, 64, 64, 4, true},
        {"mpp_mm_half_128_64_4", 2, 128, 64, 4, true}, {"mpp_mm_half_32_64_4", 2, 32, 64, 4, true},
        {"mpp_mm_half_64_128_4", 2, 64, 128, 4, true}, {"mpp_mm_half_128_128_4", 2, 128, 128, 4, true},
        {"mpp_mm_half_16_64_2", 2, 16, 64, 2, true},   {"mpp_mm_bfloat_64_64_4", 3, 64, 64, 4, true},
        {"mpp_mm_bfloat_128_64_4", 3, 128, 64, 4, true}, {"mpp_mm_float_64_32_4", 4, 64, 32, 4, true},
        {"mpp_mm_float_64_64_4", 4, 64, 64, 4, true},  {"mpp_mm_float_128_64_4", 4, 128, 64, 4, true},
        {"sg_mm_half", 2, 64, 64, 4, false},           {"sg_mm_float", 4, 64, 64, 4, false},
        {"mpp_mm_half_64_16_2", 2, 64, 16, 2, true},   {"mpp_mm_half_64_16_4", 2, 64, 16, 4, true},
        {"mpp_mm_half_128_16_4", 2, 128, 16, 4, true}, {"mpp_mm_half_128_32_4", 2, 128, 32, 4, true},
        {"mpp_mm_half_64_32_2", 2, 64, 32, 2, true},   {"mpp_mm_half_32_16_2", 2, 32, 16, 2, true},
        {"mpp_mm_half_32_32_2", 2, 32, 32, 2, true},   {"mpp_mm_half_16_32_1", 2, 16, 32, 1, true},
        {"mpp_mm_half_16_16_1", 2, 16, 16, 1, true},   {"mpp_mm_half_128_48_4", 2, 128, 48, 4, true},
        {"mpp_mm_half_64_48_4", 2, 64, 48, 4, true}};
    struct Shape { const char* name; int M, N, K; };
    const std::vector<Shape> shapes = {{"square 4096^3", 4096, 4096, 4096},
                                       {"expert gu 128 tok x 1280 x 2560", 128, 1280, 2560},
                                       {"expert down 128 tok x 2560 x 640", 128, 2560, 640},
                                       {"dense qkv 2048 tok x 10240 x 2560", 2048, 4096, 2560},
                                       {"expert gu 16 tok x 1280 x 2560", 16, 1280, 2560},
                                       {"expert gu^T 1280 x 48 tok x 2560", 1280, 48, 2560},
                                       {"expert gu^T 1280 x 16 tok x 2560", 1280, 16, 2560},
                                       {"many gu^T 12800 x 48 tok x 2560", 12800, 48, 2560},
                                       {"many gu 48 tok x 12800 x 2560", 48, 12800, 2560}};
    int filled = 0;
    const char* only = std::getenv("MMB_ONLY");   // a substring of the kernel names to run
    for (const K& k : ks) {
        if (only && k.name.find(only) == std::string::npos) continue;
        if (filled != k.elem) { fill(k.elem); filled = k.elem; }
        mtl::Pipeline p = cx.pipeline(lib, k.name);
        // correctness on a small product (M = tm*2, N = tn*2, K = 256) against the CPU
        {
            const int M = k.tm * 2, N = k.tn * 2, KK = 256;
            MmArgs a{M, N, KK};
            mtl::Dispatch d;
            d.p = &p;
            d.args = {mtl::Arg::B(A), mtl::Arg::B(B), mtl::Arg::B(C), mtl::Arg::V(a)};
            d.grid[0] = N / k.tn;
            d.grid[1] = M / k.tm;
            d.tg[0] = 32 * k.sg;
            cx.run({d});
            double se = 0, sr = 0;
            for (int i = 0; i < M; ++i)
                for (int j = 0; j < N; ++j) {
                    double ref = 0;
                    for (int q = 0; q < KK; ++q) {
                        auto val = [&](const mtl::Buffer& X, size_t idx) -> double {
                            if (k.elem == 4) return X.as<float>()[idx];
                            if (k.elem == 2) return ref::f16_to_f32(X.as<uint16_t>()[idx]);
                            return ref::bf16_to_f32(X.as<uint16_t>()[idx]);
                        };
                        ref += val(A, (size_t)i * KK + q) * val(B, (size_t)j * KK + q);
                    }
                    const double e = C.as<float>()[i * N + j] - ref;
                    se += e * e;
                    sr += ref * ref;
                }
            std::printf("%-26s check rel L2 %.2e\n", k.name.c_str(), std::sqrt(se / sr));
        }
        for (const Shape& s : shapes) {
            if (s.M % k.tm || s.N % k.tn) continue;
            MmArgs a{s.M, s.N, s.K};
            mtl::Dispatch d;
            d.p = &p;
            d.args = {mtl::Arg::B(A), mtl::Arg::B(B), mtl::Arg::B(C), mtl::Arg::V(a)};
            d.grid[0] = s.N / k.tn;
            d.grid[1] = s.M / k.tm;
            d.tg[0] = 32 * k.sg;
            const int reps = std::max(1, (int)(2e10 / (2.0 * s.M * s.N * s.K)));
            std::vector<mtl::Dispatch> ds(std::min(reps, 200), d);
            cx.warm(ds, 0.2);
            std::vector<double> t;
            for (int i = 0; i < 5; ++i) t.push_back(cx.run(ds) / ds.size());
            const double m = median(t);
            std::printf("%-26s %-36s %8.1f us  %6.2f TFLOP/s\n", k.name.c_str(), s.name, m * 1e6,
                        2.0 * s.M * s.N * s.K / m / 1e12);
        }
    }
    return 0;
}
