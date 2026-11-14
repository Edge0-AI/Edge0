// metal/tests/dense_mm_parity.cpp - dense prompt projections (dense_mm.metal): dequantize to fp16 once, then the
// MPP GEMM. Parity vs a reference with the same fp16 operands (and the exact float one), every dense format and
// awkward row counts; then the speed on the packs' shapes for a 2048-token chunk.
//   dense_mm_parity            parity
//   dense_mm_parity --bench    speed
#include "mtl_rt.h"
#include "ref_quant.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
struct DqArgs { uint32_t n_in, n_out, row_bytes, perm; };
struct GemmArgs { int M, N, K, pad; };
double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
float h16(float v) { return ref::f16_to_f32(ref::f32_to_f16(v)); }
int unit_of(int t) {
    switch (t) {
        case ref::Q2_0: case ref::IQ3_S: case ref::BF16: case ref::F32: return 16;
        default: return 32;
    }
}
}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    mtl::Context cx;
    const int lib = cx.compile({"quant.metal", "units.metal", "mma.metal", "dense_mm.metal"});
    std::printf("device %s\n", cx.device_name().c_str());
    mtl::Pipeline pact = cx.pipeline(lib, "act_f16");
    auto run_all = [&](int ty, const mtl::Buffer& W, size_t woff, int n_in, int n_out, int T, const mtl::Buffer& x,
                       const mtl::Buffer& xh, const mtl::Buffer& wh, const mtl::Buffer& y, const std::string& gk,
                       std::vector<mtl::Dispatch>& ds, int which) {
        static std::vector<mtl::Pipeline*> keep;
        mtl::Pipeline* pdq = new mtl::Pipeline(cx.pipeline(lib, "dq_f16_" + std::to_string(ty)));
        mtl::Pipeline* pg = new mtl::Pipeline(cx.pipeline(lib, gk));
        keep.push_back(pdq);
        keep.push_back(pg);
        int tm = 64, tn = 64;
        std::sscanf(gk.c_str(), "gemm_f16_%d_%d", &tm, &tn);
        if (which & 1) {
            const uint32_t n = (uint32_t)T * n_in;
            mtl::Dispatch d; d.p = &pact; d.args = {mtl::Arg::B(x), mtl::Arg::B(xh), mtl::Arg::V(n)};
            d.grid[0] = (n + 255) / 256; d.tg[0] = 256; ds.push_back(d);
        }
        if (which & 2) {
            DqArgs a{(uint32_t)n_in, (uint32_t)n_out, (uint32_t)ref::row_bytes(ty, n_in)};
            mtl::Dispatch d; d.p = pdq; d.args = {mtl::Arg::B(W, woff), mtl::Arg::B(wh), mtl::Arg::V(a)};
            d.grid[0] = (n_in / unit_of(ty) + 63) / 64; d.grid[1] = n_out; d.tg[0] = 64; ds.push_back(d);
        }
        if (which & 4) {
            GemmArgs g{T, n_out, n_in, 0};
            mtl::Dispatch d; d.p = pg; d.args = {mtl::Arg::B(xh), mtl::Arg::B(wh), mtl::Arg::B(y), mtl::Arg::V(g)};
            d.grid[0] = (n_out + tn - 1) / tn; d.grid[1] = (T + tm - 1) / tm; d.tg[0] = 128; ds.push_back(d);
        }
    };
    const std::vector<int> types = {ref::F32, ref::BF16, ref::Q8_0, ref::Q4_0, ref::Q5_0, ref::IQ4_NL, ref::IQ4_XS, ref::Q2_0,
                                    ref::IQ3_S, ref::Q3_K, ref::Q4_K, ref::Q5_K, ref::Q6_K};
    if (!bench) {
        int fails = 0;
        for (int ty : types)
            for (int n_in : {2560, 640, 6144}) {
                if (n_in % 256 && ref::block_elems(ty) == 256) continue;
                for (int n_out : {100, 48}) {
                    const int T = 72;
                    const size_t rb = ref::row_bytes(ty, n_in);
                    mtl::Buffer W = cx.alloc(rb * n_out), x = cx.alloc(4 * (size_t)T * n_in), xh = cx.alloc(2 * (size_t)T * n_in),
                                wh = cx.alloc(2 * (size_t)n_out * n_in), y = cx.alloc(4 * (size_t)T * n_out);
                    ref::random_blocks(ty, W.as<uint8_t>(), n_out, n_in, 5 + ty + n_in, 0.02f);
                    ref::Rng r(3);
                    for (size_t i = 0; i < (size_t)T * n_in; ++i) x.as<float>()[i] = r.sym();
                    std::vector<mtl::Dispatch> ds;
                    run_all(ty, W, 0, n_in, n_out, T, x, xh, wh, y, "gemm_f16_64_64_4", ds, 7);
                    cx.run(ds);
                    std::vector<float> wf((size_t)n_in * n_out);
                    ref::dequant(ty, W.ptr, wf.data(), wf.size());
                    double se16 = 0, se = 0, sr = 0;
                    for (int t = 0; t < T; ++t)
                        for (int o = 0; o < n_out; ++o) {
                            double a16 = 0, ae = 0;
                            for (int k = 0; k < n_in; ++k) {
                                const float xv = x.as<float>()[(size_t)t * n_in + k], wv = wf[(size_t)o * n_in + k];
                                a16 += (double)h16(xv) * h16(wv);
                                ae += (double)xv * wv;
                            }
                            const double g = y.as<float>()[(size_t)t * n_out + o];
                            se16 += (g - a16) * (g - a16);
                            se += (g - ae) * (g - ae);
                            sr += ae * ae;
                        }
                    const double r16 = std::sqrt(se16 / sr), re = std::sqrt(se / sr);
                    const bool ok = r16 < 1e-5;
                    fails += !ok;
                    std::printf("dense mm %-7s %5d -> %4d, T %d: vs fp16 operands %.2e, vs exact %.2e  %s\n",
                                ref::type_name(ty), n_in, n_out, T, r16, re, ok ? "PASS" : "FAIL");
                }
            }
        std::printf("%s\n", fails ? "SOME FAILED" : "ALL PASS");
        return fails ? 1 : 0;
    }
    struct Shape { const char* name; int type, n_in, n_out; };
    const std::vector<Shape> shapes = {{"gdn qkv", ref::IQ4_XS, 2560, 10240}, {"gdn qkv", ref::Q3_K, 2560, 10240},
                                       {"gdn gate", ref::Q4_K, 2560, 6144},   {"gdn out", ref::Q6_K, 6144, 2560},
                                       {"qsa q", ref::Q2_0, 2560, 12288},     {"hc down", ref::BF16, 10240, 324},
                                       {"hc up", ref::BF16, 320, 10240},      {"router", ref::BF16, 2560, 512},
                                       {"alpha", ref::BF16, 2560, 48}};
    const int T = 2048;
    for (const Shape& s : shapes) {
        if (s.n_in % 128) { std::printf("%-9s %-6s: n_in %d not a multiple of 128 (needs the 32-chunk path)\n", s.name, ref::type_name(s.type), s.n_in); continue; }
        const size_t rb = ref::row_bytes(s.type, s.n_in);
        mtl::Buffer W = cx.alloc(rb * s.n_out), x = cx.alloc(4 * (size_t)T * s.n_in), xh = cx.alloc(2 * (size_t)T * s.n_in),
                    wh = cx.alloc(2 * (size_t)s.n_out * s.n_in), y = cx.alloc(4 * (size_t)T * s.n_out);
        ref::random_blocks(s.type, W.as<uint8_t>(), s.n_out, s.n_in, 3, 0.02f);
        auto timeit = [&](int which, const std::string& gk) {
            std::vector<mtl::Dispatch> ds;
            for (int i = 0; i < 4; ++i) run_all(s.type, W, 0, s.n_in, s.n_out, T, x, xh, wh, y, gk, ds, which);
            cx.warm(ds, 0.2);
            std::vector<double> t;
            for (int i = 0; i < 5; ++i) t.push_back(cx.run(ds) / 4);
            return median(t);
        };
        const double tdq = timeit(2, "gemm_f16_64_64_4"), tact = timeit(1, "gemm_f16_64_64_4");
        double best = 1e9;
        std::string bk;
        for (const char* gk : {"gemm_f16_64_64_4", "gemm_f16_128_64_4", "gemm_f16_64_128_4", "gemm_f16_64_32_4"}) {
            const double tg = timeit(4, gk);
            if (tg < best) { best = tg; bk = gk; }
        }
        const double fl = 2.0 * T * s.n_in * s.n_out;
        std::printf("%-9s %-6s %5d -> %5d, T %d: dequant %7.1f us, act %6.1f us, GEMM %7.1f us (%5.1f TFLOP/s, %s) -> %.2f us a token\n",
                    s.name, ref::type_name(s.type), s.n_in, s.n_out, T, tdq * 1e6, tact * 1e6, best * 1e6, fl / best / 1e12,
                    bk.c_str(), (tdq + best) / T * 1e6);
    }
    return 0;
}
