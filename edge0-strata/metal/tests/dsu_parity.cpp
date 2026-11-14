// metal/tests/dsu_parity.cpp - the kernels Unsloth's UD-IQ1_S / UD-IQ1_M DeepSeek sets need on the Metal engine, against the CPU
// references in ref_quant.cpp (transcribed from ggml's dequantize_row_*; dsu_ref_vs_ggml checks that transcription against ggml itself):
//   1. ds4_iq1s_mvn1/2/4/8 (IQ1_S expert matvec; IQ1_M and IQ2_XXS as controls) and ds4_iq_mv_k (Q6_K dense matvec): the real engine
//      kernels run on int8 activations; relative L2 error against the double-precision dot of the dequantized weights.
//   2. W8<T_IQ1_S> (the prompt tile's weight loader, ds4_mm_gu_19 / ds4_mm_dn_19): bit-exact against the half-rounded reference.
//   3. ds4_dq_q6k (the dense prompt matrices' Q6_K -> half): bit-exact against the half-rounded reference.
#include "mtl_rt.h"
#include "ref_quant.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
int fails = 0;
void report(const char* what, bool ok, double err) {
    std::printf("%-62s err %.3e  %s\n", what, err, ok ? "PASS" : "FAIL");
    fails += !ok;
}
size_t al(size_t v) { return (v + 255) & ~size_t(255); }

// activations as the engine's q8 quantiser: per 32 values d = amax / 127, q = round(x / d)
void quant_x(const std::vector<float>& x, int k, int ncol, std::vector<int8_t>& q, std::vector<float>& d) {
    q.assign((size_t)ncol * k, 0);
    d.assign((size_t)ncol * k / 32, 0.f);
    for (int c = 0; c < ncol; ++c)
        for (int g = 0; g < k / 32; ++g) {
            float am = 0;
            for (int i = 0; i < 32; ++i) am = std::fmax(am, std::fabs(x[(size_t)c * k + g * 32 + i]));
            const float dd = am / 127.f;
            d[(size_t)c * (k / 32) + g] = dd;
            for (int i = 0; i < 32; ++i)
                q[(size_t)c * k + g * 32 + i] = dd == 0 ? 0 : (int8_t)std::lround(x[(size_t)c * k + g * 32 + i] / dd);
        }
}
}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    mtl::Context cx;
    std::printf("device %s\n", cx.device_name().c_str());
    const int lib = cx.compile({"../src/engine/msl/ds4_iq.metal", "../tests/dsu_job.metal"}, "#define DSU_MKJOB 1\n");
    mtl::Pipeline mk = cx.pipeline(lib, "dsu_mkjob");

    struct Case { const char* kern; int type; int nc; };
    const std::vector<Case> dummy;
    const std::vector<Case> cases = {{"ds4_iq1s_mvn1", 19, 1}, {"ds4_iq1s_mvn2", 19, 2}, {"ds4_iq1s_mvn4", 19, 4}, {"ds4_iq1s_mvn8", 19, 8},
                                     {"ds4_iq1m_mvn4", 29, 4}, {"ds4_iq2xxs_mvn4", 16, 4}, {"ds4_iq_mv_k", 14, 8}, {"ds4_iq_mv_k", 14, 1},
                                     {"ds4_q6k_mvn1", 14, 1}, {"ds4_q6k_mvn2", 14, 2}, {"ds4_q6k_mvn4", 14, 4}, {"ds4_q6k_mvn8", 14, 8}};
    if (bench) {   // dense decode matvec speed on a real attention-sized matrix: Q5_K (the 2.00 set's) against Q6_K (the Unsloth sets')
        for (const Case& cs : std::vector<Case>{{"ds4_q5k_mvn1", 13, 1}, {"ds4_iq_mv_k", 14, 1}, {"ds4_iq1s_mvn1", 19, 1}, {"ds4_iq2xxs_mvn1", 16, 1}, {"ds4_iq_mv_k", 14, 8}, {"ds4_q5k_mvn8", 13, 8}, {"ds4_q6k_mvn1", 14, 1}, {"ds4_q6k_mvn8", 14, 8}}) {
            const int k = 4096, n = 16384, nc = cs.nc;
            const size_t rb = ref::row_bytes(cs.type, k);
            std::vector<uint8_t> w(rb * n);
            ref::random_blocks(cs.type, w.data(), n, k, 3, 0.05f);
            const size_t o_w = al(80), o_xq = al(o_w + w.size()), o_xd = al(o_xq + (size_t)nc * k), o_y = al(o_xd + (size_t)nc * k / 8);
            mtl::Buffer mem = cx.alloc(al(o_y + (size_t)nc * n * 4)), offs = cx.alloc(64), args = cx.alloc(64);
            std::memcpy(mem.as<uint8_t>() + o_w, w.data(), w.size());
            std::memset(mem.as<uint8_t>() + o_xq, 3, (size_t)nc * k);
            unsigned long ov[5] = {0, o_w, o_xq, o_xd, o_y};
            int av[5] = {cs.type, k, n, nc, (int)rb};
            std::memcpy(offs.ptr, ov, sizeof ov);
            std::memcpy(args.ptr, av, sizeof av);
            mtl::Pipeline pk = cx.pipeline(lib, cs.kern);
            mtl::Dispatch d0;
            d0.p = &mk; d0.args = {mtl::Arg::B(mem), mtl::Arg::B(offs), mtl::Arg::B(args)}; d0.grid[0] = 1; d0.tg[0] = 1;
            mtl::Dispatch d1;
            d1.p = &pk; d1.args = {mtl::Arg::B(mem, 0)}; d1.grid[0] = (n + 7) / 8; d1.grid[1] = 1; d1.tg[0] = 256;
            cx.run({d0});
            std::vector<mtl::Dispatch> ds(20, d1);
            cx.warm(ds, 0.3);
            double best = 1e9;
            for (int i = 0; i < 7; ++i) best = std::fmin(best, cx.run(ds) / 20);
            std::printf("bench %-16s %-7s ncol %d: %.1f us  %.0f GB/s of weights\n", cs.kern, ref::type_name(cs.type), nc, best * 1e6, (double)w.size() / best / 1e9);
        }
        return 0;
    }
    for (const Case& cs : cases) {
        for (int k : {512, 4096}) {
            const int n = 83;   // not a multiple of 8: the last threadgroup's tail rows
            const int nc = cs.nc;
            const size_t rb = ref::row_bytes(cs.type, k);
            std::vector<uint8_t> w(rb * n);
            ref::random_blocks(cs.type, w.data(), n, k, 11 + cs.type + k, 0.05f);
            ref::Rng r(77);
            std::vector<float> x((size_t)nc * k);
            for (auto& v : x) v = r.sym() * 3.f;
            std::vector<int8_t> xq;
            std::vector<float> xd;
            quant_x(x, k, nc, xq, xd);
            const size_t o_job = 0, o_w = al(80), o_xq = al(o_w + w.size()), o_xd = al(o_xq + xq.size()), o_y = al(o_xd + xd.size() * 4);
            const size_t total = al(o_y + (size_t)nc * n * 4);
            mtl::Buffer mem = cx.alloc(total), offs = cx.alloc(64), args = cx.alloc(64);
            std::memcpy(mem.as<uint8_t>() + o_w, w.data(), w.size());
            std::memcpy(mem.as<uint8_t>() + o_xq, xq.data(), xq.size());
            std::memcpy(mem.as<uint8_t>() + o_xd, xd.data(), xd.size() * 4);
            unsigned long ov[5] = {o_job, o_w, o_xq, o_xd, o_y};
            int av[5] = {cs.type, k, n, nc, (int)rb};
            std::memcpy(offs.ptr, ov, sizeof ov);
            std::memcpy(args.ptr, av, sizeof av);
            mtl::Pipeline pk = cx.pipeline(lib, cs.kern);
            mtl::Dispatch d0;
            d0.p = &mk; d0.args = {mtl::Arg::B(mem), mtl::Arg::B(offs), mtl::Arg::B(args)}; d0.grid[0] = 1; d0.tg[0] = 1;
            mtl::Dispatch d1;
            d1.p = &pk; d1.args = {mtl::Arg::B(mem, o_job)}; d1.grid[0] = (n + 7) / 8; d1.grid[1] = 1; d1.tg[0] = 256;
            cx.run({d0, d1});
            std::vector<float> wf((size_t)n * k);
            ref::dequant(cs.type, w.data(), wf.data(), wf.size());
            double se = 0, sr = 0;
            const float* y = (const float*)(mem.as<uint8_t>() + o_y);
            for (int c = 0; c < nc; ++c)
                for (int row = 0; row < n; ++row) {
                    double acc = 0;
                    for (int i = 0; i < k; ++i)
                        acc += (double)wf[(size_t)row * k + i] * ((double)xq[(size_t)c * k + i] * xd[(size_t)c * (k / 32) + i / 32]);
                    const double g = y[(size_t)c * n + row];
                    se += (g - acc) * (g - acc);
                    sr += acc * acc;
                }
            const double e = std::sqrt(se / sr);
            char nm[128];
            std::snprintf(nm, sizeof nm, "%s %s k=%d n=%d ncol=%d", cs.kern, ref::type_name(cs.type), k, n, nc);
            report(nm, e < 2e-6, e);
        }
    }

    // 1b. the engine's own prompt-tile library compiles with the IQ1_S instantiation
    {
        const int libm = cx.compile({"quant.metal", "units.metal", "mma.metal", "moe_mm.metal", "../src/engine/msl/ds4_mm.metal"});
        for (const char* fn : {"ds4_mm_gu_19", "ds4_mm_dn_19", "ds4_mm_gu_29", "ds4_mm_gu_16"}) {
            mtl::Pipeline pp = cx.pipeline(libm, fn);
            report(fn, pp.handle != nullptr, 0);
        }
    }

    // 2. the tile loader
    {
        const int libw = cx.compile({"quant.metal", "units.metal", "mma.metal", "moe_mm.metal", "../tests/dsu_job.metal"}, "#define DSU_W8 1\n");
        for (int ty : {19, 29}) {
            const int k = 2048, n = 37;
            const size_t rb = ref::row_bytes(ty, k);
            mtl::Buffer W = cx.alloc(rb * n), out = cx.alloc((size_t)n * k * 2);
            ref::random_blocks(ty, W.as<uint8_t>(), n, k, 5 + ty, 0.05f);
            mtl::Pipeline pw = cx.pipeline(libw, ty == 19 ? "dsu_w8_iq1s" : "dsu_w8_iq1m");
            mtl::Dispatch d;
            d.p = &pw; d.args = {mtl::Arg::B(W), mtl::Arg::B(out), mtl::Arg::V((uint32_t)k), mtl::Arg::V((uint32_t)rb)};
            d.grid[0] = (k / 32 * 4 + 63) / 64; d.grid[1] = n; d.tg[0] = 64;
            cx.run({d});
            std::vector<float> wf((size_t)n * k);
            ref::dequant(ty, W.ptr, wf.data(), wf.size());
            size_t bad = 0;
            const uint16_t* o = out.as<uint16_t>();
            for (size_t i = 0; i < wf.size(); ++i)
                if (o[i] != ref::f32_to_f16(wf[i])) ++bad;
            char nm[112];
            std::snprintf(nm, sizeof nm, "tile loader W8<%s>, %zu values, mismatching half bits", ref::type_name(ty), wf.size());
            report(nm, bad == 0, (double)bad);
        }
    }

    // 3. Q6_K -> half
    {
        const int libd = cx.compile({"../src/engine/msl/ds4_init.metal"});
        mtl::Pipeline pq = cx.pipeline(libd, "ds4_dq_q6k");
        for (int k : {256, 2048}) {
            const int n = 45;
            const size_t rb = ref::row_bytes(14, k);
            mtl::Buffer W = cx.alloc(rb * n), out = cx.alloc((size_t)n * k * 2);
            ref::random_blocks(14, W.as<uint8_t>(), n, k, 3 + k, 0.05f);
            mtl::Dispatch d;
            const unsigned long ld = rb;
            const uint32_t uk = k, un = n;
            d.p = &pq; d.args = {mtl::Arg::B(W), mtl::Arg::V(ld), mtl::Arg::V(uk), mtl::Arg::V(un), mtl::Arg::B(out)};
            d.grid[0] = (uint32_t)(((size_t)n * (k / 256) * 64 + 255) / 256); d.tg[0] = 256;
            cx.run({d});
            std::vector<float> wf((size_t)n * k);
            ref::dequant(14, W.ptr, wf.data(), wf.size());
            size_t bad = 0;
            const uint16_t* o = out.as<uint16_t>();
            for (size_t i = 0; i < wf.size(); ++i)
                if (o[i] != ref::f32_to_f16(wf[i])) ++bad;
            char nm[96];
            std::snprintf(nm, sizeof nm, "ds4_dq_q6k k=%d n=%d, mismatching half bits", k, n);
            report(nm, bad == 0, (double)bad);
        }
    }
    std::printf("%s\n", fails ? "SOME FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
