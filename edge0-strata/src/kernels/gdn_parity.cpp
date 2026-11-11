// src/kernels/gdn_parity.cpp - P2.S2's test for the GDN recurrence, the conv and the two norms.
//
// `ref/gdn.py` carries eleven PROPERTY checks, several of which exist because a rival reading of the SOURCE
// produces well-formed output.  The four that the kernels here can be wrong about are re-pinned below, each
// asserted OBSERVABLE against a deliberately-wrong computation before the kernel is judged:
//
//   1. MODULO head pairing, h % h_k, not INTERLEAVE.  With H_k=2 and H_v=4 the two give [0,1,0,1] and
//      [0,0,1,1]: they differ at h=1 and h=2.  Both produce a full-rank state of the right shape.
//   2. Decay applied BEFORE the rank-1 update.  Applying it after is a one-line move that keeps every shape
//      and every magnitude.
//   3. The conv's SLIDE DIRECTION.  `kernel[0]` must read the OLDEST state row and the new input must land in
//      the LAST state row; reversing either is invisible on a symmetric state, so the state rows here are
//      labelled so each position is identifiable.
//   4. `l2_norm`'s `+ eps` is an absolute floor on the SQUARED NORM, not on the mean.  The mean reading
//      differs by sqrt(S) = 11.3x at S = 128.  `ref/gdn.py` PROPERTY 1 instruments the derivation, so it is
//      not taken on trust there either.
//
// It also checks the STATE LAYOUT, which is the one place these kernels intentionally differ from the
// reference: (S, h_v, S) with j fastest instead of (S, S, h_v).  A layout mix-up is silent, so the state is
// filled with a value that encodes its own coordinates.
#include "strata/kernels/gdn.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

float sigmoid_f(float x) { return 1.0f / (1.0f + std::exp(-x)); }

/// The reference's `gdn_step`, with switches for the rival readings.
///
/// `state` is passed in the REFERENCE layout (S, S, h_v) - i.e. (i, j, h) - so that this function is a
/// transcription of `ref/gdn.py` and not of the kernel.
void ref_step(std::vector<double>& state, const std::vector<float>& q, const std::vector<float>& k,
              const std::vector<float>& v, const std::vector<float>& gate, const std::vector<float>& beta,
              int S, int h_k, int h_v, std::vector<float>& o, bool modulo_heads, bool decay_first) {
    auto at = [&](int i, int j, int h) -> double& { return state[(size_t) (i * S + j) * h_v + h]; };
    std::vector<float> oo((size_t) h_v * S, 0.0f);
    for (int h = 0; h < h_v; ++h) {
        const int src = modulo_heads ? (h % h_k) : (h / (h_v / h_k));
        const double dec = std::exp((double) gate[(size_t) h]);
        const double b = (double) beta[(size_t) h];
        for (int j = 0; j < S; ++j) {
            if (decay_first) for (int i = 0; i < S; ++i) at(i, j, h) *= dec;
            double sk = 0;
            for (int i = 0; i < S; ++i) sk += at(i, j, h) * (double) k[(size_t) (src * S + i)];
            const double d = ((double) v[(size_t) (h * S + j)] - sk) * b;
            for (int i = 0; i < S; ++i) at(i, j, h) += (double) k[(size_t) (src * S + i)] * d;
            if (!decay_first) for (int i = 0; i < S; ++i) at(i, j, h) *= dec;
            double dot = 0;
            for (int i = 0; i < S; ++i) dot += at(i, j, h) * (double) q[(size_t) (src * S + i)];
            oo[(size_t) (h * S + j)] = (float) dot;
        }
    }
    o = oo;
}

/// `ref/gdn.py::conv_step` in its own (d_conv, C) row-major orientation.
void ref_conv(std::vector<float>& cs, const std::vector<float>& x, const std::vector<float>& kern, int C,
              int dc, std::vector<float>& out) {
    out.assign((size_t) C, 0.0f);
    for (int c = 0; c < C; ++c) {
        double acc = 0;
        for (int i = 0; i < dc - 1; ++i) acc += (double) cs[(size_t) (i * C + c)] * (double) kern[(size_t) (i * C + c)];
        acc += (double) x[(size_t) c] * (double) kern[(size_t) ((dc - 1) * C + c)];
        out[(size_t) c] = (float) acc;
        for (int i = 0; i < dc - 2; ++i) cs[(size_t) (i * C + c)] = cs[(size_t) ((i + 1) * C + c)];
        cs[(size_t) ((dc - 2) * C + c)] = x[(size_t) c];
    }
}

double rel_l1(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0, m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += std::fabs((double) a[i] - (double) b[i]);
        m += std::fabs((double) a[i]);
    }
    return d / (m > 1e-30 ? m : 1e-30);
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // (progress lines survive a crash)
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: gdn_parity [--selftest]\n"); return 2; }
    }

    const int S = 128, h_k = 16, h_v = 48;      // the artifact's real geometry
    std::mt19937 rng(2024);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    int bad = 0;

    // ================= 1. the recurrence =================
    {
        // The state encodes its own coordinates, so a layout mix-up cannot pass: reading (i,j,h) where the
        // buffer holds (i,h,j) gives a DIFFERENT number, not a wrong-but-plausible one.
        std::vector<double> st_ref((size_t) S * S * h_v);
        std::vector<float> st_dev((size_t) S * h_v * S);
        for (int i = 0; i < S; ++i)
            for (int j = 0; j < S; ++j)
                for (int h = 0; h < h_v; ++h) {
                    const float val = (float) (0.001 * (i + 2 * j + 3 * h) - 1.0);
                    st_ref[(size_t) (i * S + j) * h_v + h] = val;
                    st_dev[(size_t) (i * h_v + h) * S + j] = val;   // (S, h_v, S), j fastest
                }

        std::vector<float> q((size_t) h_k * S), k((size_t) h_k * S), v((size_t) h_v * S), gate((size_t) h_v),
            beta((size_t) h_v);
        for (auto& x : q) x = gauss(rng) * 0.1f;
        for (auto& x : k) x = gauss(rng) * 0.1f;
        for (auto& x : v) x = gauss(rng);
        for (auto& x : gate) x = -(2.0f + 4.0f * (float) (rng() % 100) / 100.0f);   // dec in (0.0025, 0.135]
        for (auto& x : beta) x = (float) (rng() % 100) / 100.0f;

        std::vector<float> want_o;
        std::vector<double> st_mod = st_ref;
        ref_step(st_mod, q, k, v, gate, beta, S, h_k, h_v, want_o, true, true);

        // the traps
        {
            std::vector<double> a = st_ref;
            std::vector<float> oo;
            ref_step(a, q, k, v, gate, beta, S, h_k, h_v, oo, false, true);
            const double rel = rel_l1(want_o, oo);
            std::printf("  %-42s %-4s (%.2f%% apart)\n", "modulo vs interleave head pairing",
                        rel > 0.05 ? "yes" : "*** NO ***", rel * 100);
            if (!(rel > 0.05)) ++bad;
        }
        {
            std::vector<double> a = st_ref;
            std::vector<float> oo;
            ref_step(a, q, k, v, gate, beta, S, h_k, h_v, oo, true, false);
            const double rel = rel_l1(want_o, oo);
            std::printf("  %-42s %-4s (%.2f%% apart)\n", "decay before vs after the update",
                        rel > 0.05 ? "yes" : "*** NO ***", rel * 100);
            // The two readings differ by exactly `k (x) (beta * v * (1 - dec))`, so the fixture is only able
            // to see the trap when `dec` is far from 1 and `beta` is far from 0.  The first version used
            // gate in [-0.99, 0] - dec in [0.37, 1] - and managed 3.25%, which is a threshold the wrong
            // reading nearly clears.  Deep decay is also the realistic regime: `ssm_a = -exp(A_log) < 0` and
            // softplus is positive, so gate is comfortably negative in the real model.
            if (!(rel > 0.05)) ++bad;
        }

        // the kernel
        float *d_st = nullptr, *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_g = nullptr, *d_b = nullptr,
              *d_o = nullptr;
        check(cudaMalloc(&d_st, st_dev.size() * 4), "st");
        check(cudaMalloc(&d_q, q.size() * 4), "q");
        check(cudaMalloc(&d_k, k.size() * 4), "k");
        check(cudaMalloc(&d_v, v.size() * 4), "v");
        check(cudaMalloc(&d_g, gate.size() * 4), "g");
        check(cudaMalloc(&d_b, beta.size() * 4), "b");
        check(cudaMalloc(&d_o, (size_t) h_v * S * 4), "o");
        check(cudaMemcpy(d_st, st_dev.data(), st_dev.size() * 4, cudaMemcpyHostToDevice), "cst");
        check(cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice), "cq");
        check(cudaMemcpy(d_k, k.data(), k.size() * 4, cudaMemcpyHostToDevice), "ck");
        check(cudaMemcpy(d_v, v.data(), v.size() * 4, cudaMemcpyHostToDevice), "cv");
        check(cudaMemcpy(d_g, gate.data(), gate.size() * 4, cudaMemcpyHostToDevice), "cg");
        check(cudaMemcpy(d_b, beta.data(), beta.size() * 4, cudaMemcpyHostToDevice), "cb");

        strata::kernels::GdnShapes sh{S, h_k, h_v};
        strata::kernels::gdn_step(d_st, d_q, d_k, d_v, d_g, d_b, d_o, sh, nullptr);

        std::vector<float> got_o((size_t) h_v * S), got_st(st_dev.size());
        check(cudaMemcpy(got_o.data(), d_o, got_o.size() * 4, cudaMemcpyDeviceToHost), "cgo");
        check(cudaMemcpy(got_st.data(), d_st, got_st.size() * 4, cudaMemcpyDeviceToHost), "cgst");

        // o, and the STATE read back through the declared layout
        const double rel_o = rel_l1(want_o, got_o);
        double rel_st = 0, mag_st = 0;
        for (int i = 0; i < S; ++i)
            for (int j = 0; j < S; ++j)
                for (int h = 0; h < h_v; ++h) {
                    const size_t dev = (size_t) (i * h_v + h) * S + j;
                    const double w = st_mod[(size_t) (i * S + j) * h_v + h];
                    rel_st += std::fabs(w - (double) got_st[dev]);
                    mag_st += std::fabs(w);
                }
        rel_st /= (mag_st > 1e-30 ? mag_st : 1e-30);
        std::printf("\n  %-42s rel %.3e\n", "o vs reference", rel_o);
        std::printf("  %-42s rel %.3e\n", "state (read via (S,h_v,S)) vs reference", rel_st);
        // The kernel accumulates in f32 and the reference in f64, and the state is a RECURRENCE over 128
        // terms, so 1e-5 is the arithmetic's floor here and not a structural allowance.
        if (!(rel_o <= 1e-5)) { std::printf("    *** o over 1e-5 ***\n"); ++bad; }
        if (!(rel_st <= 1e-5)) { std::printf("    *** state over 1e-5 ***\n"); ++bad; }

        // a zero state stays zero, and a zero gate/beta leaves the state untouched (the reference's
        // PROPERTY 4 setup) - a cheap independent invariant that no layout can fake
        {
            std::vector<float> zero(st_dev.size(), 0.0f), zg((size_t) h_v, 0.0f), zb((size_t) h_v, 0.0f);
            check(cudaMemcpy(d_st, zero.data(), zero.size() * 4, cudaMemcpyHostToDevice), "z0");
            check(cudaMemcpy(d_g, zg.data(), zg.size() * 4, cudaMemcpyHostToDevice), "z1");
            check(cudaMemcpy(d_b, zb.data(), zb.size() * 4, cudaMemcpyHostToDevice), "z2");
            strata::kernels::gdn_step(d_st, d_q, d_k, d_v, d_g, d_b, d_o, sh, nullptr);
            std::vector<float> zz(zero.size());
            check(cudaMemcpy(zz.data(), d_st, zz.size() * 4, cudaMemcpyDeviceToHost), "z3");
            double nz = 0;
            for (float x : zz) nz += std::fabs(x);
            const bool ok = nz == 0.0;
            std::printf("  %-42s %s\n", "gate=beta=0 from a zero state is a no-op", ok ? "yes" : "*** NO ***");
            if (!ok) ++bad;
        }

        // ephemeral state: unchanged when the state is untouched but beta=1 — the delta rule writing k*d
        cudaFree(d_st); cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_g); cudaFree(d_b); cudaFree(d_o);
    }

    // ================= 2. the conv =================
    {
        const int C = 24, dc = 4;      // small, so the host reference is a plain loop
        // TWO DIFFERENT LAYOUTS MEET HERE, and this is where the first version of this test failed at 0.136.
        // `ref/gdn.py` carries the conv state as (d_conv-1, C) ROW-MAJOR - (i, c) at i*C + c - because it is
        // numpy.  The kernel carries it as (C, d_conv-1) with c fastest - c*(dc-1) + i - because that is
        // ggml's `ne = {d_conv-1, C}` with ne0 fast, and it is a runtime buffer so either is legal.  The
        // mapping between them is written out below rather than assumed, in both directions.
        std::vector<float> cs_ref((size_t) (dc - 1) * C), x((size_t) C), kern((size_t) dc * C);
        for (int i = 0; i < dc - 1; ++i)
            for (int c = 0; c < C; ++c) cs_ref[(size_t) (i * C + c)] = (float) (10 * (i + 1) + c);  // labelled
        for (int c = 0; c < C; ++c) x[(size_t) c] = (float) (100 + c);
        for (size_t i = 0; i < kern.size(); ++i) kern[i] = gauss(rng);

        auto to_dev_layout = [&](const std::vector<float>& r) {
            std::vector<float> d(r.size());
            for (int i = 0; i < dc - 1; ++i)
                for (int c = 0; c < C; ++c) d[(size_t) (c * (dc - 1) + i)] = r[(size_t) (i * C + c)];
            return d;
        };
        auto to_ref_layout = [&](const std::vector<float>& d) {
            std::vector<float> r(d.size());
            for (int i = 0; i < dc - 1; ++i)
                for (int c = 0; c < C; ++c) r[(size_t) (i * C + c)] = d[(size_t) (c * (dc - 1) + i)];
            return r;
        };

        std::vector<float> cs = to_dev_layout(cs_ref);
        std::vector<float> want;
        std::vector<float> cs_want_ref = cs_ref;
        ref_conv(cs_want_ref, x, kern, C, dc, want);
        const std::vector<float> cs_want = to_dev_layout(cs_want_ref);

        // THE LAYOUT: the manifest's `ssm_conv1d.weight` has ne = [d_conv, C], so ne0 = d_conv is fast.
        // The kernel takes `kW[c*dc + i]`; the reference's `kern` is (dc, C) row-major.  The transcription
        // below is the ONLY place the two meet.
        std::vector<float> kW((size_t) C * dc);
        for (int c = 0; c < C; ++c)
            for (int i = 0; i < dc; ++i) kW[(size_t) (c * dc + i)] = kern[(size_t) (i * C + c)];
        // and the rival reading, which must differ or the fixture cannot see a transposed weight
        std::vector<float> kW_wrong((size_t) C * dc);
        for (size_t i = 0; i < kern.size(); ++i) kW_wrong[i] = kern[i];
        {
            // apply the WRONG layout by hand: tap i of channel c is kW[c*dc+i] read as kW[i*C+c]
            std::vector<float> cs2 = cs;
            std::vector<float> out_bad((size_t) C, 0.0f);
            for (int c = 0; c < C; ++c) {
                double a = 0;
                for (int i = 0; i < dc - 1; ++i) a += (double) cs2[(size_t) (i * C + c)] * (double) kW_wrong[(size_t) (c * dc + i)];
                a += (double) x[(size_t) c] * (double) kW_wrong[(size_t) (c * dc + dc - 1)];
                out_bad[(size_t) c] = (float) a;
            }
            const double rel = rel_l1(want, out_bad);
            std::printf("\n  %-42s %-4s (%.1f%% apart)\n", "conv weight layout is observable",
                        rel > 0.05 ? "yes" : "*** NO ***", rel * 100);
            if (!(rel > 0.05)) ++bad;
        }

        float *d_cs = nullptr, *d_x = nullptr, *d_w = nullptr, *d_o = nullptr;
        check(cudaMalloc(&d_cs, cs.size() * 4), "cs");
        check(cudaMalloc(&d_x, x.size() * 4), "cx");
        check(cudaMalloc(&d_w, kW.size() * 4), "cw");
        check(cudaMalloc(&d_o, (size_t) C * 4), "co");
        check(cudaMemcpy(d_cs, cs.data(), cs.size() * 4, cudaMemcpyHostToDevice), "ccs");
        check(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "ccx");
        check(cudaMemcpy(d_w, kW.data(), kW.size() * 4, cudaMemcpyHostToDevice), "ccw");
        strata::kernels::gdn_conv_step(d_cs, d_x, d_w, d_o, C, dc, nullptr);
        std::vector<float> got((size_t) C), got_cs(cs.size());
        check(cudaMemcpy(got.data(), d_o, got.size() * 4, cudaMemcpyDeviceToHost), "cgo");
        check(cudaMemcpy(got_cs.data(), d_cs, got_cs.size() * 4, cudaMemcpyDeviceToHost), "cgs");
        const double rel_out = rel_l1(want, got), rel_cs = rel_l1(cs_want, got_cs);
        std::printf("  %-42s rel %.3e\n", "conv out vs reference", rel_out);
        std::printf("  %-42s rel %.3e\n", "conv state slide (mapped) vs reference", rel_cs);
        if (!(rel_out <= 1e-6)) { std::printf("    *** conv out ***\n"); ++bad; }
        if (!(rel_cs <= 1e-6)) { std::printf("    *** conv state ***\n"); ++bad; }
        cudaFree(d_cs); cudaFree(d_x); cudaFree(d_w); cudaFree(d_o);
    }

    // ================= 3. l2_norm =================
    {
        const int rows = 16, cols = 128;
        std::vector<float> x((size_t) rows * cols), x_ref;
        for (auto& val : x) val = gauss(rng) * 3.0f;
        x_ref = x;
        const float eps = 1e-6f;
        for (int r = 0; r < rows; ++r) {
            double acc = 0;
            for (int i = 0; i < cols; ++i) acc += (double) x_ref[(size_t) (r * cols + i)] * (double) x_ref[(size_t) (r * cols + i)];
            const float inv = (float) (1.0 / std::sqrt(acc + (double) eps));
            for (int i = 0; i < cols; ++i) x_ref[(size_t) (r * cols + i)] *= inv;
        }
        // The rival reading puts THE SAME eps on the MEAN instead of on the squared norm, which at
        // ||x||^2 ~ 1152 and cols = 128 changes the divisor from sqrt(1152) to sqrt(9) - a factor of 11.3, so
        // the trap is large here.  It would be a much smaller correction on a near-zero row, where the floor
        // is the whole answer; that regime is not what this fixture exercises.
        std::vector<float> x_mean = x;
        for (int r = 0; r < rows; ++r) {
            double acc = 0;
            for (int i = 0; i < cols; ++i) acc += (double) x_mean[(size_t) (r * cols + i)] * (double) x_mean[(size_t) (r * cols + i)];
            const float inv = (float) (1.0 / std::sqrt(acc / cols + (double) eps));
            for (int i = 0; i < cols; ++i) x_mean[(size_t) (r * cols + i)] *= inv;
        }
        const double rel_trap = rel_l1(x_ref, x_mean);
        std::printf("\n  %-42s %-4s (%.1f%% apart)\n", "eps on sum vs on mean is observable",
                    rel_trap > 0.05 ? "yes" : "*** NO ***", rel_trap * 100);
        if (!(rel_trap > 0.05)) ++bad;

        float* d_x = nullptr;
        check(cudaMalloc(&d_x, x.size() * 4), "lx");
        check(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "lcx");
        strata::kernels::gdn_l2_norm(d_x, rows, cols, eps, nullptr);
        std::vector<float> got(x.size());
        check(cudaMemcpy(got.data(), d_x, got.size() * 4, cudaMemcpyDeviceToHost), "lcg");
        const double rel = rel_l1(x_ref, got);
        std::printf("  %-42s rel %.3e\n", "l2_norm vs reference", rel);
        if (!(rel <= 1e-6)) ++bad;
        cudaFree(d_x);
    }

    // ================= 4. the closing norm =================
    {
        const int S2 = 128, hv2 = 8;
        std::vector<float> o((size_t) hv2 * S2), z((size_t) hv2 * S2), sn((size_t) S2), y_ref((size_t) hv2 * S2);
        for (auto& val : o) val = gauss(rng);
        for (auto& val : z) val = gauss(rng);
        for (auto& val : sn) val = 1.0f + 0.1f * gauss(rng);
        const float eps = 1e-6f;
        for (int h = 0; h < hv2; ++h) {
            double acc = 0;
            for (int i = 0; i < S2; ++i) acc += (double) o[(size_t) (h * S2 + i)] * (double) o[(size_t) (h * S2 + i)];
            const float inv = (float) (1.0 / std::sqrt(acc / S2 + (double) eps));
            for (int i = 0; i < S2; ++i)
                y_ref[(size_t) (h * S2 + i)] = o[(size_t) (h * S2 + i)] * inv * sn[(size_t) i] *
                                              sigmoid_f(z[(size_t) (h * S2 + i)]);
        }
        float *d_o = nullptr, *d_z = nullptr, *d_sn = nullptr, *d_y = nullptr;
        check(cudaMalloc(&d_o, o.size() * 4), "no");
        check(cudaMalloc(&d_z, z.size() * 4), "nz");
        check(cudaMalloc(&d_sn, sn.size() * 4), "ns");
        check(cudaMalloc(&d_y, y_ref.size() * 4), "ny");
        check(cudaMemcpy(d_o, o.data(), o.size() * 4, cudaMemcpyHostToDevice), "nco");
        check(cudaMemcpy(d_z, z.data(), z.size() * 4, cudaMemcpyHostToDevice), "ncz");
        check(cudaMemcpy(d_sn, sn.data(), sn.size() * 4, cudaMemcpyHostToDevice), "ncs");
        strata::kernels::gdn_out_norm(d_o, d_z, d_sn, d_y, hv2, S2, eps, nullptr);
        std::vector<float> got(y_ref.size());
        check(cudaMemcpy(got.data(), d_y, got.size() * 4, cudaMemcpyDeviceToHost), "ncy");
        const double rel = rel_l1(y_ref, got);
        std::printf("\n  %-42s rel %.3e\n", "out norm (rms * ssm_norm * sigmoid z)", rel);
        if (!(rel <= 1e-6)) ++bad;

        // SiLU instead of sigmoid is the qwen3.5 reading; `ref/gdn.py` says this artifact does NOT use it
        std::vector<float> y_silu(y_ref.size());
        for (int h = 0; h < hv2; ++h) {
            double acc = 0;
            for (int i = 0; i < S2; ++i) acc += (double) o[(size_t) (h * S2 + i)] * (double) o[(size_t) (h * S2 + i)];
            const float inv = (float) (1.0 / std::sqrt(acc / S2 + (double) eps));
            for (int i = 0; i < S2; ++i) {
                const float zz = z[(size_t) (h * S2 + i)];
                y_silu[(size_t) (h * S2 + i)] =
                    o[(size_t) (h * S2 + i)] * inv * sn[(size_t) i] * (zz / (1.0f + std::exp(-zz)));
            }
        }
        const double rel_silu = rel_l1(y_ref, y_silu);
        std::printf("  %-42s %-4s (%.2f%% apart)\n", "sigmoid vs SiLU on the gate is observable",
                    rel_silu > 0.05 ? "yes" : "*** NO ***", rel_silu * 100);
        if (!(rel_silu > 0.05)) ++bad;
        cudaFree(d_o); cudaFree(d_z); cudaFree(d_sn); cudaFree(d_y);
    }

    // ================= 5. the verify-window kernels, T = 1..8, bitwise against the single-token kernels =================
    // gdn_ab_multi and gdn_step_norm_multi are documented as bitwise `fused_gdn_ab` / `fused_gdn_step_norm` per
    // token (verify_kernels.hpp). Every T from 1 to kVerifyMaxT is its own exact-T instantiation, so each is run,
    // and every output and the state are compared with memcmp (not a tolerance).
    {
        const int n_embd = 2560, qk = S * h_k, C = 2 * qk + S * h_v, vd = S * h_v;
        const float eps = 1e-6f;
        auto bf16_of = [](float f) {
            uint32_t u;
            std::memcpy(&u, &f, 4);
            return (uint16_t) (u >> 16);
        };
        std::vector<uint16_t> wa((size_t) h_v * n_embd), wb((size_t) h_v * n_embd);
        for (auto& w : wa) w = bf16_of(gauss(rng) * 0.05f);
        for (auto& w : wb) w = bf16_of(gauss(rng) * 0.05f);
        std::vector<float> dt((size_t) h_v), ssm_a((size_t) h_v), gamma((size_t) S);
        for (auto& x : dt) x = gauss(rng);
        for (auto& x : ssm_a) x = -(0.1f + (float) (rng() % 100) / 50.0f);
        for (auto& x : gamma) x = 1.0f + 0.1f * gauss(rng);
        std::vector<float> st0((size_t) S * h_v * S);
        for (auto& x : st0) x = gauss(rng) * 0.1f;

        auto dev = [&](size_t bytes) {
            void* p = nullptr;
            check(cudaMalloc(&p, bytes), "m5");
            return p;
        };
        auto up = [&](const void* h, size_t bytes) {
            void* p = dev(bytes);
            check(cudaMemcpy(p, h, bytes, cudaMemcpyHostToDevice), "u5");
            return p;
        };
        auto down = [&](std::vector<float>& h, const void* d) {
            check(cudaMemcpy(h.data(), d, h.size() * 4, cudaMemcpyDeviceToHost), "d5");
        };
        auto d_wa = (uint16_t*) up(wa.data(), wa.size() * 2);
        auto d_wb = (uint16_t*) up(wb.data(), wb.size() * 2);
        auto d_dt = (float*) up(dt.data(), dt.size() * 4);
        auto d_sa = (float*) up(ssm_a.data(), ssm_a.size() * 4);
        auto d_gm = (float*) up(gamma.data(), gamma.size() * 4);
        int ab_bad = 0, sn_bad = 0, ab_n = 0, sn_n = 0;
        for (int T = 1; T <= strata::kernels::kVerifyMaxT; ++T) {
            // --- alpha/beta
            std::vector<float> x((size_t) T * n_embd);
            for (auto& v : x) v = gauss(rng);
            auto d_x = (float*) up(x.data(), x.size() * 4);
            auto d_g = (float*) dev((size_t) T * h_v * 4), d_b = (float*) dev((size_t) T * h_v * 4);
            auto d_gr = (float*) dev((size_t) T * h_v * 4), d_br = (float*) dev((size_t) T * h_v * 4);
            strata::kernels::gdn_ab_multi(d_x, d_wa, d_wb, d_dt, d_sa, d_g, d_b, n_embd, h_v, T, nullptr);
            for (int t = 0; t < T; ++t)
                strata::kernels::fused_gdn_ab(d_x + (size_t) t * n_embd, d_wa, d_wb, d_dt, d_sa, d_gr + (size_t) t * h_v,
                                              d_br + (size_t) t * h_v, n_embd, h_v, nullptr);
            std::vector<float> g((size_t) T * h_v), b(g.size()), gr(g.size()), br(g.size());
            down(g, d_g); down(b, d_b); down(gr, d_gr); down(br, d_br);
            ++ab_n;
            if (std::memcmp(g.data(), gr.data(), g.size() * 4) != 0 || std::memcmp(b.data(), br.data(), b.size() * 4) != 0) {
                std::printf("    *** gdn_ab_multi T=%d differs from fused_gdn_ab (memcmp) ***\n", T);
                ++ab_bad;
            }
            cudaFree(d_x); cudaFree(d_g); cudaFree(d_b); cudaFree(d_gr); cudaFree(d_br);

            // --- the recurrence + output norm
            std::vector<float> h((size_t) T * C), gate((size_t) T * h_v), beta((size_t) T * h_v), z((size_t) T * vd);
            for (auto& v : h) v = gauss(rng) * 0.1f;
            for (auto& v : gate) v = -(0.5f + 3.0f * (float) (rng() % 100) / 100.0f);
            for (auto& v : beta) v = (float) (rng() % 100) / 100.0f;
            for (auto& v : z) v = gauss(rng);
            auto d_h = (float*) up(h.data(), h.size() * 4);
            auto d_gate = (float*) up(gate.data(), gate.size() * 4);
            auto d_beta = (float*) up(beta.data(), beta.size() * 4);
            auto d_z = (float*) up(z.data(), z.size() * 4);
            auto d_y = (float*) dev((size_t) T * vd * 4);
            auto d_yr = (float*) dev((size_t) T * vd * 4);
            auto d_st = (float*) dev(st0.size() * 4);
            auto d_sr = (float*) dev(st0.size() * 4);
            int32_t* d_nk = (int32_t*) dev(4);
            // the reference chain (the single-token kernel), n tokens, into d_sr / d_yr
            auto run_ref = [&](int n) {
                check(cudaMemcpy(d_sr, st0.data(), st0.size() * 4, cudaMemcpyHostToDevice), "r0");
                for (int t = 0; t < n; ++t)
                    strata::kernels::fused_gdn_step_norm(d_sr, d_h + (size_t) t * C, d_h + (size_t) t * C + qk,
                                                         d_h + (size_t) t * C + 2 * qk, d_gate + (size_t) t * h_v,
                                                         d_beta + (size_t) t * h_v, d_z + (size_t) t * vd, d_gm, eps,
                                                         d_yr + (size_t) t * vd, h_k, h_v, nullptr);
            };
            std::vector<float> y((size_t) T * vd), yr(y.size()), st(st0.size()), sr(st0.size());
            // (name, n_keep tokens or -1 = verify half, t_out_begin)
            struct Case { const char* name; int n_keep; int tb; };
            std::vector<Case> cases = {{"verify", -1, 0}, {"verify tb", -1, T > 1 ? T / 2 : 0}, {"commit", T, 0},
                                       {"commit tb", T, T / 2}, {"commit state only", T, T}};
            if (T > 1) cases.push_back({"commit n-1", T - 1, T - 1});
            for (const Case& c : cases) {
                check(cudaMemcpy(d_st, st0.data(), st0.size() * 4, cudaMemcpyHostToDevice), "s0");
                check(cudaMemset(d_y, 0, (size_t) T * vd * 4), "y0");
                check(cudaMemset(d_yr, 0, (size_t) T * vd * 4), "yr0");
                const int n = c.n_keep < 0 ? T : c.n_keep;
                if (c.n_keep >= 0) check(cudaMemcpy(d_nk, &c.n_keep, 4, cudaMemcpyHostToDevice), "nk");
                strata::kernels::gdn_step_norm_multi(d_st, d_h, C, d_gate, d_beta, d_z, d_gm, eps, d_y, h_k, h_v, T,
                                                     c.n_keep >= 0 ? d_nk : nullptr, nullptr, c.tb);
                run_ref(n);
                down(y, d_y); down(yr, d_yr); down(st, d_st); down(sr, d_sr);
                bool ok = true;
                size_t y_bad = 0, y_all = 0;
                for (int t = c.tb; t < n; ++t) {
                    for (int i = 0; i < vd; ++i, ++y_all)
                        if (std::memcmp(&y[(size_t) t * vd + i], &yr[(size_t) t * vd + i], 4) != 0) ++y_bad;
                }
                if (y_bad) ok = false;
                // the verify half leaves the state untouched; the commit half writes the state after n tokens
                const std::vector<float>& want_st = c.n_keep < 0 ? st0 : sr;
                if (std::memcmp(st.data(), want_st.data(), st.size() * 4) != 0) ok = false;
                ++sn_n;
                // #3948 (STRATA_QFUSE): the same call with a q8_1 destination also writes the outputs' q8_1 image, row t - tb
                // at (t - tb) * value_dim: the bytes native_quantize_q8_1 makes of y rows [tb, n), the outputs and the state
                // unchanged by asking for it, and nothing written past those rows (the engine's second group has tb > 0)
                if (c.tb < n) {
                    if (std::getenv("STRATA_GDN_PARITY_TRACE")) std::printf("    trace: T=%d '%s' with a q8_1 destination\n", T, c.name);
                    const size_t qb = (size_t) (vd / 32) * 36, rows = (size_t) (n - c.tb);
                    void *d_q = dev((size_t) T * qb), *d_qr = dev((size_t) T * qb);
                    check(cudaMemset(d_q, 0x5a, (size_t) T * qb), "q0");
                    check(cudaMemcpy(d_st, st0.data(), st0.size() * 4, cudaMemcpyHostToDevice), "s1");
                    std::vector<float> y2(y.size()), st2(st0.size());
                    check(cudaMemset(d_y, 0, (size_t) T * vd * 4), "y1");
                    strata::kernels::gdn_step_norm_multi(d_st, d_h, C, d_gate, d_beta, d_z, d_gm, eps, d_y, h_k, h_v, T,
                                                         c.n_keep >= 0 ? d_nk : nullptr, nullptr, c.tb, d_q);
                    static cudaStream_t qstream = nullptr;   // (native_quantize_q8_1 refuses the null stream; a blocking stream orders with the legacy one)
                    if (qstream == nullptr) check(cudaStreamCreate(&qstream), "qstream");
                    strata::kernels::native_quantize_q8_1(d_y + (size_t) c.tb * vd, d_qr, vd, (int) rows, qstream);
                    check(cudaDeviceSynchronize(), "q sync");
                    std::vector<uint8_t> hq((size_t) T * qb), hr((size_t) T * qb);
                    check(cudaMemcpy(hq.data(), d_q, hq.size(), cudaMemcpyDeviceToHost), "q read");
                    check(cudaMemcpy(hr.data(), d_qr, rows * qb, cudaMemcpyDeviceToHost), "qr read");
                    down(y2, d_y); down(st2, d_st);
                    bool q_ok = std::memcmp(hq.data(), hr.data(), rows * qb) == 0;
                    for (size_t i = rows * qb; q_ok && i < hq.size(); ++i) q_ok = hq[i] == 0x5a;
                    for (int t = c.tb; q_ok && t < n; ++t)
                        q_ok = std::memcmp(&y2[(size_t) t * vd], &y[(size_t) t * vd], (size_t) vd * 4) == 0;
                    q_ok = q_ok && std::memcmp(st2.data(), st.data(), st.size() * 4) == 0;
                    if (!q_ok) {
                        std::printf("    *** gdn_step_norm_multi T=%d '%s' with a q8_1 destination: bytes differ from native_quantize_q8_1 or outputs/state changed ***\n", T, c.name);
                        ++sn_bad;
                    }
                    ++sn_n;
                    cudaFree(d_q); cudaFree(d_qr);
                }
                if (!ok) {
                    std::printf("    *** gdn_step_norm_multi T=%d '%s' differs from fused_gdn_step_norm (memcmp): %zu of %zu y words%s ***\n",
                                T, c.name, y_bad, y_all,
                                std::memcmp(st.data(), (c.n_keep < 0 ? st0 : sr).data(), st.size() * 4) != 0 ? ", state too" : "");
                    ++sn_bad;
                }
            }
            cudaFree(d_h); cudaFree(d_gate); cudaFree(d_beta); cudaFree(d_z); cudaFree(d_y); cudaFree(d_yr); cudaFree(d_st);
            cudaFree(d_sr); cudaFree(d_nk);
        }
        cudaFree(d_wa); cudaFree(d_wb); cudaFree(d_dt); cudaFree(d_sa); cudaFree(d_gm);
        std::printf("\n  %-42s %s (%d cases)\n", "gdn_ab_multi T=1..8 == fused_gdn_ab", ab_bad ? "*** NO ***" : "bitwise", ab_n);
        std::printf("  %-42s %s (%d cases)\n", "gdn_step_norm_multi T=1..8 == single-token", sn_bad ? "*** NO ***" : "bitwise", sn_n);
        bad += ab_bad + sn_bad;
    }

    // ================= 6. the layer-batched commit (engine-latency), bitwise against the per-layer calls =================
    // gdn_commit_layers runs every layer's conv-history commit and state commit in one launch each; the per-layer loop
    // (gdn_conv_commit + gdn_step_norm_multi with n_keep, no outputs) is the reference.  Several layers with their own
    // state, rows and gate/beta; n_keep 1..T; memcmp on the whole state arena (states and conv histories).
    {
        const int qk = S * h_k, C = 2 * qk + S * h_v, L = 5, MT = strata::kernels::kVerifyMaxT;
        const int64_t conv_off = (int64_t) S * h_v * S, stride = conv_off + (int64_t) C * 3;
        std::vector<float> st0((size_t) (stride * L)), hb((size_t) L * MT * C), qkv((size_t) L * MT * C),
            gt((size_t) L * MT * h_v), bt((size_t) L * MT * h_v), zdummy((size_t) MT * S * h_v), gm(S, 1.0f);
        for (auto& x : st0) x = gauss(rng) * 0.1f;
        for (auto& x : hb) x = gauss(rng) * 0.1f;
        for (auto& x : qkv) x = gauss(rng);
        for (auto& x : gt) x = -(0.5f + (float) (rng() % 100) / 50.0f);
        for (auto& x : bt) x = (float) (rng() % 100) / 100.0f;
        float *d_a = nullptr, *d_b = nullptr, *d_h = nullptr, *d_qkv = nullptr, *d_g = nullptr, *d_bt = nullptr,
              *d_z = nullptr, *d_y = nullptr, *d_gm = nullptr;
        int32_t* d_nk = nullptr;
        check(cudaMalloc(&d_a, st0.size() * 4), "la"); check(cudaMalloc(&d_b, st0.size() * 4), "lb");
        check(cudaMalloc(&d_h, hb.size() * 4), "lh"); check(cudaMalloc(&d_qkv, qkv.size() * 4), "lq");
        check(cudaMalloc(&d_g, gt.size() * 4), "lg"); check(cudaMalloc(&d_bt, bt.size() * 4), "lbt");
        check(cudaMalloc(&d_z, zdummy.size() * 4), "lz"); check(cudaMalloc(&d_y, zdummy.size() * 4), "ly");
        check(cudaMalloc(&d_gm, gm.size() * 4), "lgm"); check(cudaMalloc(&d_nk, 4), "lnk");
        check(cudaMemcpy(d_h, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice), "lch");
        check(cudaMemcpy(d_qkv, qkv.data(), qkv.size() * 4, cudaMemcpyHostToDevice), "lcq");
        check(cudaMemcpy(d_g, gt.data(), gt.size() * 4, cudaMemcpyHostToDevice), "lcg");
        check(cudaMemcpy(d_bt, bt.data(), bt.size() * 4, cudaMemcpyHostToDevice), "lcb");
        check(cudaMemcpy(d_gm, gm.data(), gm.size() * 4, cudaMemcpyHostToDevice), "lcm");
        check(cudaMemset(d_z, 0, zdummy.size() * 4), "lz0");
        int lb_bad = 0, lb_n = 0;
        bool launched = true;
        for (int nk = 1; nk <= MT && launched; ++nk) {
            check(cudaMemcpy(d_nk, &nk, 4, cudaMemcpyHostToDevice), "lnk1");
            check(cudaMemcpy(d_a, st0.data(), st0.size() * 4, cudaMemcpyHostToDevice), "lca");
            check(cudaMemcpy(d_b, st0.data(), st0.size() * 4, cudaMemcpyHostToDevice), "lcb2");
            for (int l = 0; l < L; ++l) {   // the reference: the per-layer calls
                float* stl = d_a + (size_t) l * stride;
                strata::kernels::gdn_conv_commit(stl + conv_off, d_qkv + (size_t) l * MT * C, C, d_nk, nullptr);
                strata::kernels::gdn_step_norm_multi(stl, d_h + (size_t) l * MT * C, C, d_g + (size_t) l * MT * h_v,
                                                     d_bt + (size_t) l * MT * h_v, d_z, d_gm, 1e-6f, d_y, h_k, h_v, MT, d_nk,
                                                     nullptr, MT);
            }
            launched = strata::kernels::gdn_commit_layers(d_b, stride, conv_off, d_qkv, (int64_t) MT * C, d_h, (int64_t) MT * C,
                                                          d_g, d_bt, (int64_t) MT * h_v, C, h_k, h_v, MT, L, d_nk, nullptr);
            if (!launched) break;
            check(cudaDeviceSynchronize(), "lsync");
            std::vector<float> a(st0.size()), b(st0.size());
            check(cudaMemcpy(a.data(), d_a, a.size() * 4, cudaMemcpyDeviceToHost), "lda");
            check(cudaMemcpy(b.data(), d_b, b.size() * 4, cudaMemcpyDeviceToHost), "ldb");
            ++lb_n;
            if (std::memcmp(a.data(), b.data(), a.size() * 4) != 0 || std::memcmp(a.data(), st0.data(), a.size() * 4) == 0) {
                std::printf("    *** gdn_commit_layers n_keep=%d differs from the per-layer commit (or wrote nothing) ***\n", nk);
                ++lb_bad;
            }
        }
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_h); cudaFree(d_qkv); cudaFree(d_g); cudaFree(d_bt); cudaFree(d_z);
        cudaFree(d_y); cudaFree(d_gm); cudaFree(d_nk);
        if (launched)
            std::printf("  %-42s %s (%d cases, %d layers)\n", "gdn_commit_layers == per-layer commit", lb_bad ? "*** NO ***" : "bitwise",
                        lb_n, L);
        else
            std::printf("  %-42s skipped (the per-layer commit is not the state-only kernel here)\n", "gdn_commit_layers");
        bad += lb_bad;
    }

    std::printf("\ngdn: %d failures\n", bad);
    if (bad) return 1;
    if (selftest) std::printf("gdn_parity OK\n");
    return 0;
}
