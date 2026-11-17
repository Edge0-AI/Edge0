// src/kernels/rope_parity.cpp - P2.S2's parity test for NEOX partial RoPE.
//
// TWO CHECKS, deliberately separated so each can be tight:
//
//   1. THE TABLE against the float64 reference - `theta ** (-2i/n_rot)`, `pos * inv`, `cos`/`sin`, all in
//      double exactly as `ref/qsa.py::rope_freqs` does.  Compared as float32 after the cast, so a mismatch
//      here is a real arithmetic difference and not a formatting one.
//   2. THE ROTATION against a host rotation written from `ref/qsa.py::rope_neox`, using the SAME table the
//      kernel is given.  Both sides then perform identical float32 operations, so this is BIT-EXACT - and a
//      bit-exact rotation plus a table that matches the reference means the composition matches too.
//
// A single end-to-end comparison would have had to carry one loose tolerance for both, and the interesting
// failure - the NEOX pairing being wrong - is a PERMUTATION that a loose tolerance over all 256 dims would
// happily accept.
//
// Check 4 extends the same discipline to the SCALED tables (rope scaling: none/linear/YaRN).  The rotation
// kernel cannot see scaling - it lives in the table contents - so each variant's TABLE is held to a float64
// transcription of ggml's `rope_yarn` spec (the ramp helper is shared, the `rope_neox_pair` convention), and
// the two properties a tolerance could never fake get their own structural checks: at position 0 every pair's
// angle is 0, so YaRN's mscale stands naked in cos_tab[0], and at a far position the first pair must match
// EXTRAPOLATION while the last matches INTERPOLATION.  An observability assertion closes it: if the scaled
// and unscaled tables were indistinguishable, every green number above would be vacuous.
//
// Check 5 holds the TWO PATHS together: the table path's float64 host trig and the native path's float32
// fast-math device trig must answer to the same `RopeScaling`, yarn and none alike, so one cache never
// mixes two rotations.
#include "strata/kernels/rope.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_rope.hpp"

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

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: rope_parity [--selftest]\n"); return 2; }
    }

    const int n_rot = 64;          // the artifact's rope.dimension_count
    const double theta = 1.0e7;    // rope.freq_base
    const int head_dim = 256;
    const int rows = 24 * 8;       // 24 heads over 8 positions, so several positions exercise the table
    const int max_pos = 32;
    const int half = n_rot / 2;
    int bad = 0;

    // ---- 1. the table against the float64 reference
    std::vector<float> hcos((size_t) max_pos * half), hsin((size_t) max_pos * half);
    strata::kernels::build_rope_table(n_rot, theta, max_pos, hcos.data(), hsin.data());
    long long table_bad = 0;
    double table_worst = 0.0;
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            const double inv = std::pow(theta, -2.0 * (double) i / (double) n_rot);
            const double ang = (double) p * inv;
            const float rc = (float) std::cos(ang), rs = (float) std::sin(ang);
            if (std::memcmp(&rc, &hcos[(size_t) p * half + i], 4) != 0) ++table_bad;
            if (std::memcmp(&rs, &hsin[(size_t) p * half + i], 4) != 0) ++table_bad;
            table_worst = std::max(table_worst,
                                   std::fabs((double) rc - (double) hcos[(size_t) p * half + i]));
        }
    }
    std::printf("  rope table vs float64 reference   %s (%lld of %d entries differ, worst %.1e)\n",
                table_bad ? "*** WRONG ***" : "bit-exact", table_bad, max_pos * half * 2, table_worst);
    bad += (int) table_bad;

    // ---- 2. the rotation, against the same spec in float32 with the SAME table
    std::mt19937 rng(11);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> x((size_t) rows * head_dim);
    for (auto& v : x) v = gauss(rng);
    std::vector<int> pos((size_t) rows);
    for (int r = 0; r < rows; ++r) pos[(size_t) r] = r % max_pos;

    std::vector<float> ref(x.size());
    for (int r = 0; r < rows; ++r) {
        const float* xr = &x[(size_t) r * head_dim];
        float* orow = &ref[(size_t) r * head_dim];
        for (int d = 0; d < head_dim; ++d) orow[d] = xr[d];          // the tail passes through
        for (int i = 0; i < half; ++i) {
            const float c = hcos[(size_t) pos[(size_t) r] * half + i];
            const float s = hsin[(size_t) pos[(size_t) r] * half + i];
            const float a = xr[i], b = xr[half + i];
            orow[i] = a * c - b * s;
            orow[half + i] = a * s + b * c;
        }
    }

    float *d_x = nullptr, *d_out = nullptr, *d_cos = nullptr, *d_sin = nullptr;
    int* d_pos = nullptr;
    check(cudaMalloc(&d_x, x.size() * sizeof(float)), "malloc x");
    check(cudaMalloc(&d_out, ref.size() * sizeof(float)), "malloc out");
    check(cudaMalloc(&d_cos, hcos.size() * sizeof(float)), "malloc cos");
    check(cudaMalloc(&d_sin, hsin.size() * sizeof(float)), "malloc sin");
    check(cudaMalloc(&d_pos, pos.size() * sizeof(int)), "malloc pos");
    check(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "copy x");
    check(cudaMemcpy(d_cos, hcos.data(), hcos.size() * sizeof(float), cudaMemcpyHostToDevice), "copy cos");
    check(cudaMemcpy(d_sin, hsin.data(), hsin.size() * sizeof(float), cudaMemcpyHostToDevice), "copy sin");
    check(cudaMemcpy(d_pos, pos.data(), pos.size() * sizeof(int), cudaMemcpyHostToDevice), "copy pos");
    strata::kernels::rope_neox_apply(d_x, d_out, rows, head_dim, n_rot, d_cos, d_sin, d_pos, nullptr);
    std::vector<float> got(ref.size());
    check(cudaMemcpy(got.data(), d_out, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "back");

    // A RELATIVE TOLERANCE, NOT BIT EQUALITY.  *c - b*s is one of the expressions the compiler is free to
    // contract into an FMA, and the host and device compilers choose differently - so a handful of values
    // differ in the last bit.  Round 167 established this the hard way: a bit-equality expectation that a
    // kernel cannot meet is a test that reports a correct kernel as broken.  The residual is reported, and if
    // it were structural (a wrong pairing, a wrong table entry) it would be O(1), not O(1 ULP).
    long long rot_bad = 0;
    double worst = 0.0;
    long long first_bad = -1;
    // The denominator is the ROW'S INPUT MAGNITUDE, not the element's own value.  *c - b*s cancels, so an
    // element whose result is near zero reports a large relative error for a 1-ULP absolute one - which is the
    // same metric mistake round 169 found on Q4_K's dot products, where the fix was to measure against
    // sum|term| instead of the result.  Here the natural scale of the computation is the largest input in the
    // row, because every output element is a combination of two inputs with |cos|,|sin| <= 1.
    std::vector<double> row_scale((size_t) rows, 0.0);
    for (int r = 0; r < rows; ++r) {
        double m = 0;
        for (int d = 0; d < head_dim; ++d) m = std::max(m, (double) std::fabs(x[(size_t) r * head_dim + d]));
        row_scale[(size_t) r] = m > 1e-30 ? m : 1e-30;
    }
    for (size_t i = 0; i < ref.size(); ++i) {
        const double a = ref[i], b = got[i];
        const double rel = std::fabs(a - b) / row_scale[i / (size_t) head_dim];
        if (!(rel <= worst)) worst = rel;
        if (!(rel <= 1e-6)) { if (first_bad < 0) first_bad = (long long) i; ++rot_bad; }
    }
    std::printf("  rope rotation vs host reference   %s (%lld of %zu over 1e-6, worst rel %.3e)\n",
                rot_bad ? "*** WRONG ***" : "agrees", rot_bad, ref.size(), worst);
    if (first_bad >= 0) std::printf("    first over-tolerance at element %lld\n", first_bad);
    bad += (int) rot_bad;

    // ---- 3. THE PAIRING, asserted structurally rather than as a tolerance.  NEOX pairs (i, i+half); the
    // adjacent-pair convention would pair (2i, 2i+1).  Feed an input that is 1 at dim 0 and 0 elsewhere, and
    // check WHICH dims move: under NEOX only dims 0 and half may change, under the adjacent convention only
    // dims 0 and 1 may.  A tolerance over all 256 dims cannot tell the two apart.
    {
        std::vector<float> e((size_t) head_dim, 0.0f);
        e[0] = 1.0f;
        std::vector<float> o((size_t) head_dim, 0.0f);
        check(cudaMemcpy(d_x, e.data(), e.size() * sizeof(float), cudaMemcpyHostToDevice), "copy e");
        // position 7, NOT position 0: at pos 0 sin is exactly 0, so dim half legitimately does not move and
        // the check would report the correct kernel as wrong.  The first version made exactly that mistake.
        int p7 = 7;
        check(cudaMemcpy(d_pos, &p7, sizeof(int), cudaMemcpyHostToDevice), "copy p7");
        strata::kernels::rope_neox_apply(d_x, d_out, 1, head_dim, n_rot, d_cos, d_sin, d_pos, nullptr);
        check(cudaMemcpy(o.data(), d_out, o.size() * sizeof(float), cudaMemcpyDeviceToHost), "back e");
        int moved[4] = {0, 0, 0, 0};       // dims 0, 1, half, half+1
        const float* chk[4] = {&e[0], &e[1], &e[half], &e[half + 1]};
        (void) chk;
        moved[0] = (o[0] != 0.0f || o[half] != 0.0f) ? 1 : 0;
        const bool adjacent_moved = (o[1] != 0.0f);
        const bool neox_moved = (o[half] != 0.0f);
        std::printf("  pairing: dim 0 set -> NEOX moves dim %d (o[half]=%.6f), adjacent would move dim 1 "
                    "(o[1]=%.6f)   %s\n", half, (double) o[half], (double) o[1],
                    (neox_moved && !adjacent_moved) ? "NEOX confirmed" : "*** WRONG CONVENTION ***");
        if (!(neox_moved && !adjacent_moved)) ++bad;
    }

    // ---- 4. THE SCALED TABLES (rope_scaling.hpp).  Scaling lives in the table contents, so each variant is
    // a table check: a float64 transcription of the ggml spec, bit-exact after the float32 cast, plus the
    // structural and observability checks the tolerances cannot cover.
    {
        // 4a. type None IS the five-argument builder above - bit for bit.
        std::vector<float> nc((size_t) max_pos * half), ns((size_t) max_pos * half);
        strata::kernels::RopeScaling none;
        none.freq_base = theta;
        strata::kernels::build_rope_table(n_rot, none, max_pos, nc.data(), ns.data());
        const long long none_bad =
            (long long) (std::memcmp(nc.data(), hcos.data(), nc.size() * 4) != 0) +
            (long long) (std::memcmp(ns.data(), hsin.data(), ns.size() * 4) != 0);
        std::printf("  scaled table, None vs unscaled builder  %s\n", none_bad ? "*** WRONG ***" : "bit-identical");
        bad += (int) none_bad;

        // 4b. LINEAR, factor 4: `ang = p * inv / 4` in float64.  The factor is a power of two on purpose -
        // dividing by it is exact, so no multiplication-order rounding can sneak between spec and builder.
        strata::kernels::RopeScaling lin;
        lin.type = strata::kernels::RopeScalingType::Linear;
        lin.freq_base = theta;
        lin.factor = 4.0;
        std::vector<float> lc((size_t) max_pos * half), ls((size_t) max_pos * half);
        strata::kernels::build_rope_table(n_rot, lin, max_pos, lc.data(), ls.data());
        long long lin_bad = 0;
        for (int p = 0; p < max_pos; ++p) {
            for (int i = 0; i < half; ++i) {
                const double inv = std::pow(theta, -2.0 * (double) i / (double) n_rot);
                const double ang = (double) p * inv / 4.0;
                const float rc = (float) std::cos(ang), rs = (float) std::sin(ang);
                if (std::memcmp(&rc, &lc[(size_t) p * half + i], 4) != 0) ++lin_bad;
                if (std::memcmp(&rs, &ls[(size_t) p * half + i], 4) != 0) ++lin_bad;
            }
        }
        std::printf("  scaled table, linear factor 4          %s (%lld of %d entries differ)\n",
                    lin_bad ? "*** WRONG ***" : "bit-exact", lin_bad, max_pos * half * 2);
        bad += (int) lin_bad;

        // 4c. THE INTERPOLATION CLAIM ITSELF: linear(4) at position p is `none` at p/4 - the same table row,
        // bit for bit, at every position divisible by the factor.
        std::vector<float> wc((size_t) 4 * max_pos * half), ws((size_t) 4 * max_pos * half);
        strata::kernels::build_rope_table(n_rot, lin, 4 * max_pos, wc.data(), ws.data());
        long long equiv_bad = 0;
        for (int p = 0; p < 4 * max_pos; p += 4) {
            for (int i = 0; i < half; ++i) {
                if (std::memcmp(&wc[(size_t) p * half + i], &hcos[(size_t) (p / 4) * half + i], 4) != 0) ++equiv_bad;
                if (std::memcmp(&ws[(size_t) p * half + i], &hsin[(size_t) (p / 4) * half + i], 4) != 0) ++equiv_bad;
            }
        }
        std::printf("  linear(4) @ p == none @ p/4            %s (%lld of %d rows differ)\n",
                    equiv_bad ? "*** WRONG ***" : "bit-identical", equiv_bad, max_pos);
        bad += (int) equiv_bad;

        // 4d. YARN vs a float64 transcription of ggml's rope_yarn (factor 4, the default correction knobs).
        // The ramp helper is shared with the builder (the rope_neox_pair convention: one definition, not two
        // transcriptions of it); the interpolation mix and the mscale formula are written here from the ggml
        // source lines, which is the spec being tested.
        strata::kernels::RopeScaling yarn;
        yarn.type = strata::kernels::RopeScalingType::YaRN;
        yarn.freq_base = theta;
        yarn.factor = 4.0;
        yarn.ext_factor = 1.0;
        std::vector<float> yc((size_t) max_pos * half), ys((size_t) max_pos * half);
        strata::kernels::build_rope_table(n_rot, yarn, max_pos, yc.data(), ys.data());
        const double fs = yarn.freq_scale();          // 0.25
        const double ms = yarn.mscale();              // attn_factor * (1 + 0.1*ln(4))
        double cd[2];
        yarn.corr_dims(n_rot, cd);
        long long yarn_bad = 0;
        for (int p = 0; p < max_pos; ++p) {
            for (int i = 0; i < half; ++i) {
                const double inv = std::pow(theta, -2.0 * (double) i / (double) n_rot);
                const double extrap = (double) p * inv;
                const double interp = fs * extrap;
                const double ramp = (double) strata::kernels::rope_yarn_ramp((float) cd[0], (float) cd[1], i) *
                                    yarn.ext_factor;
                const double ang = interp * (1.0 - ramp) + extrap * ramp;
                const float rc = (float) (std::cos(ang) * ms), rs = (float) (std::sin(ang) * ms);
                if (std::memcmp(&rc, &yc[(size_t) p * half + i], 4) != 0) ++yarn_bad;
                if (std::memcmp(&rs, &ys[(size_t) p * half + i], 4) != 0) ++yarn_bad;
            }
        }
        std::printf("  scaled table, yarn factor 4            %s (%lld of %d entries differ)\n",
                    yarn_bad ? "*** WRONG ***" : "bit-exact", yarn_bad, max_pos * half * 2);
        bad += (int) yarn_bad;

        // 4e. THE STRUCTURE a tolerance cannot fake.  At position 0 every pair's angle is 0, so YaRN's whole
        // magnitude correction stands naked: cos_tab[0] == mscale.  At the far end of the table the first
        // pair (the highest trained frequency) must match EXTRAPOLATION and the last INTERPOLATION.  The two
        // halves need different observables: at the first pair cos separates the hypotheses outright, but at
        // the last they differ by ~4 microradians here, invisible to cos near 1 - sin sees it, because near
        // zero sin IS the angle.  That asymmetry is the YaRN design: interpolation happens where angles are
        // small.  Both hypotheses go through the same mix formula the builder uses, so the bit comparison is
        // spec against spec, not shortcut against implementation.
        {
            const float msv = (float) ms;
            const bool zero_ok = std::memcmp(&msv, &yc[0], 4) == 0 && ys[0] == 0.0f;
            const int far = max_pos - 1;
            const double inv_f = 1.0;   // pair 0: theta ** 0
            const double extrap_f = (double) far * inv_f, interp_f = fs * extrap_f;
            const double inv_l = std::pow(theta, -2.0 * (double) (half - 1) / (double) n_rot);
            const double extrap_l = (double) far * inv_l, interp_l = fs * extrap_l;
            const float ef = (float) (std::cos(interp_f * (1.0 - 1.0) + extrap_f * 1.0) * ms);
            const float itf = (float) (std::cos(interp_f * (1.0 - 0.0) + extrap_f * 0.0) * ms);
            const float sl_e = (float) (std::sin(interp_l * (1.0 - 1.0) + extrap_l * 1.0) * ms);
            const float sl_i = (float) (std::sin(interp_l * (1.0 - 0.0) + extrap_l * 0.0) * ms);
            const float got_f = yc[(size_t) far * half], got_l = ys[(size_t) far * half + (half - 1)];
            const bool first_extrapolates = std::memcmp(&got_f, &ef, 4) == 0;
            const bool last_interpolates = std::memcmp(&got_l, &sl_i, 4) == 0;
            const bool distinguishable = std::fabs((double) ef - (double) itf) > 1e-3 &&
                                         std::fabs((double) sl_e - (double) sl_i) > 1e-8;
            std::printf("  yarn structure: cos_tab[0]==mscale %.4f, first pair extrapolates, last interpolates   "
                        "%s\n", (double) msv,
                        zero_ok && first_extrapolates && last_interpolates && distinguishable
                            ? "confirmed"
                            : "*** WRONG ***");
            if (!(zero_ok && first_extrapolates && last_interpolates && distinguishable)) ++bad;
        }

        // 4f. THE OBSERVABILITY assertion: the fixtures must SEE scaling.  If the builder ignored its config,
        // linear(2) would equal `none` everywhere and 4b-4e would be green on a broken builder.
        {
            strata::kernels::RopeScaling lin2;
            lin2.type = strata::kernels::RopeScalingType::Linear;
            lin2.freq_base = theta;
            lin2.factor = 2.0;
            std::vector<float> l2c((size_t) 101 * half), l2s((size_t) 101 * half);
            strata::kernels::build_rope_table(n_rot, lin2, 101, l2c.data(), l2s.data());
            const float c_none = (float) std::cos(100.0);      // pair 0, position 100, unscaled: ang = 100
            const float c_lin = l2c[(size_t) 100 * half];      // pair 0, position 100, linear(2): ang = 50
            const bool sees = std::fabs((double) c_none - (double) c_lin) > 1e-3;
            std::printf("  observability: none vs linear(2) at position 100   %s (|%.4f - %.4f|)\n",
                        sees ? "visible" : "*** VACUUM ***", (double) c_none, (double) c_lin);
            if (!sees) ++bad;
        }

        // 4g. THE ROTATION under a scaled table.  The kernel is table-agnostic and must not know or care:
        // the host reference is the same rope_neox_pair walk as check 2, over the YaRN table.
        std::vector<float> ref2(x.size());
        for (int r = 0; r < rows; ++r) {
            const float* xr = &x[(size_t) r * head_dim];
            float* orow = &ref2[(size_t) r * head_dim];
            for (int d = 0; d < head_dim; ++d) orow[d] = xr[d];
            for (int i = 0; i < half; ++i) {
                float c = yc[(size_t) pos[(size_t) r] * half + i], s = ys[(size_t) pos[(size_t) r] * half + i];
                strata::kernels::rope_neox_pair(xr[i], xr[half + i], c, s, orow[i], orow[half + i]);
            }
        }
        check(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "copy x");
        check(cudaMemcpy(d_cos, yc.data(), yc.size() * sizeof(float), cudaMemcpyHostToDevice), "copy ycos");
        check(cudaMemcpy(d_sin, ys.data(), ys.size() * sizeof(float), cudaMemcpyHostToDevice), "copy ysin");
        check(cudaMemcpy(d_pos, pos.data(), pos.size() * sizeof(int), cudaMemcpyHostToDevice), "copy pos");
        strata::kernels::rope_neox_apply(d_x, d_out, rows, head_dim, n_rot, d_cos, d_sin, d_pos, nullptr);
        std::vector<float> got2(ref2.size());
        check(cudaMemcpy(got2.data(), d_out, got2.size() * sizeof(float), cudaMemcpyDeviceToHost), "back");
        long long rot2_bad = 0;
        double worst2 = 0.0;
        for (size_t i = 0; i < ref2.size(); ++i) {
            const double a = ref2[i], b = got2[i];
            const double rel = std::fabs(a - b) / row_scale[i / (size_t) head_dim];
            worst2 = std::max(worst2, rel);
            if (!(rel <= 1e-6)) ++rot2_bad;
        }
        std::printf("  rope rotation over the yarn table      %s (%lld of %zu over 1e-6, worst rel %.3e)\n",
                    rot2_bad ? "*** WRONG ***" : "agrees", rot2_bad, ref2.size(), worst2);
        bad += (int) rot2_bad;
    }

    // ---- 5. THE TWO PATHS MUST AGREE.  The table path (the default) computes cos/sin on the host in float64;
    // the native path (`--native-rope`) computes the same angles on device in float32 under `--use_fast_math`.
    // They answer to ONE config - `RopeScaling` - and a disagreement between them would put differently-rotated
    // K into the cache depending on which path ran.  The tolerance is NOT the 1e-6 of checks 2/4g: the device
    // side takes fast-math trig at angles up to ~2048 rad, where the fp32 range reduction alone is worth ~1e-4,
    // so the bar is the row-magnitude-relative 3e-3 and the test keeps the positions where that holds.  (The
    // engine's default is the table path precisely because float64 host trig has no such floor.)
    {
        const int npos = 2048;
        strata::kernels::RopeScaling yarn;
        yarn.type = strata::kernels::RopeScalingType::YaRN;
        yarn.factor = 2.0;
        yarn.ext_factor = 1.0;
        std::vector<float> sc((size_t) npos * half), ss((size_t) npos * half);
        strata::kernels::build_rope_table(n_rot, yarn, npos, sc.data(), ss.data());
        std::mt19937 rng2(23);
        std::normal_distribution<float> gauss2(0.0f, 1.0f);
        const int nrows = 24 * 16;             // 16 positions spread over the table's range
        std::vector<float> x2((size_t) nrows * head_dim);
        for (auto& v : x2) v = gauss2(rng2);
        std::vector<int> pos2((size_t) nrows);
        for (int r = 0; r < nrows; ++r) pos2[(size_t) r] = (r * 127) % npos;

        float *d_x2 = nullptr, *d_t2 = nullptr, *d_n2 = nullptr, *d_c2 = nullptr, *d_s2 = nullptr;
        int* d_p2 = nullptr;
        check(cudaMalloc(&d_x2, x2.size() * sizeof(float)), "malloc x2");
        check(cudaMalloc(&d_t2, x2.size() * sizeof(float)), "malloc t2");
        check(cudaMalloc(&d_n2, x2.size() * sizeof(float)), "malloc n2");
        check(cudaMalloc(&d_c2, sc.size() * sizeof(float)), "malloc c2");
        check(cudaMalloc(&d_s2, ss.size() * sizeof(float)), "malloc s2");
        check(cudaMalloc(&d_p2, pos2.size() * sizeof(int)), "malloc p2");
        check(cudaMemcpy(d_x2, x2.data(), x2.size() * sizeof(float), cudaMemcpyHostToDevice), "copy x2");
        check(cudaMemcpy(d_c2, sc.data(), sc.size() * sizeof(float), cudaMemcpyHostToDevice), "copy c2");
        check(cudaMemcpy(d_s2, ss.data(), ss.size() * sizeof(float), cudaMemcpyHostToDevice), "copy s2");
        check(cudaMemcpy(d_p2, pos2.data(), pos2.size() * sizeof(int), cudaMemcpyHostToDevice), "copy p2");
        // The native path demands an explicit stream - a null one is refused by validation, not defaulted.
        cudaStream_t cs5 = nullptr;
        check(cudaStreamCreate(&cs5), "stream5");
        strata::kernels::rope_neox_apply(d_x2, d_t2, nrows, head_dim, n_rot, d_c2, d_s2, d_p2, nullptr);
        strata::kernels::native_rope_apply(d_x2, d_n2, nrows, head_dim, n_rot, yarn, d_p2, cs5);
        check(cudaStreamSynchronize(cs5), "sync5");
        std::vector<float> got_t(x2.size()), got_n(x2.size());
        check(cudaMemcpy(got_t.data(), d_t2, got_t.size() * sizeof(float), cudaMemcpyDeviceToHost), "back t2");
        check(cudaMemcpy(got_n.data(), d_n2, got_n.size() * sizeof(float), cudaMemcpyDeviceToHost), "back n2");
        long long agree_bad = 0;
        double worst = 0.0;
        for (int r = 0; r < nrows; ++r) {
            double m = 0;
            for (int d = 0; d < head_dim; ++d) m = std::max(m, (double) std::fabs(x2[(size_t) r * head_dim + d]));
            const double scale = m > 1e-30 ? m : 1e-30;
            for (int d = 0; d < head_dim; ++d) {
                const double rel = std::fabs((double) got_t[(size_t) r * head_dim + d] -
                                             (double) got_n[(size_t) r * head_dim + d]) / scale;
                worst = std::max(worst, rel);
                if (!(rel <= 3e-3)) ++agree_bad;
            }
        }
        std::printf("  native path vs table path, yarn factor 2   %s (%lld of %d over 3e-3, worst rel %.3e)\n",
                    agree_bad ? "*** WRONG ***" : "agrees", agree_bad, nrows * head_dim, worst);
        bad += (int) agree_bad;

        // and the None config: the native path against the UNSCALED table - the identity this feature must
        // not disturb.
        strata::kernels::RopeScaling none5;               // all defaults: type None, freq_scale 1, mscale 1
        strata::kernels::native_rope_apply(d_x2, d_n2, nrows, head_dim, n_rot, none5, d_p2, cs5);
        check(cudaStreamSynchronize(cs5), "sync5b");
        check(cudaMemcpy(got_n.data(), d_n2, got_n.size() * sizeof(float), cudaMemcpyDeviceToHost), "back n2b");
        std::vector<float> nc5((size_t) npos * half), ns5((size_t) npos * half);
        strata::kernels::build_rope_table(n_rot, none5, npos, nc5.data(), ns5.data());
        check(cudaMemcpy(d_c2, nc5.data(), nc5.size() * sizeof(float), cudaMemcpyHostToDevice), "copy c2b");
        check(cudaMemcpy(d_s2, ns5.data(), ns5.size() * sizeof(float), cudaMemcpyHostToDevice), "copy s2b");
        strata::kernels::rope_neox_apply(d_x2, d_t2, nrows, head_dim, n_rot, d_c2, d_s2, d_p2, nullptr);
        check(cudaMemcpy(got_t.data(), d_t2, got_t.size() * sizeof(float), cudaMemcpyDeviceToHost), "back t2b");
        long long none_bad5 = 0;
        double worst5 = 0.0;
        for (int r = 0; r < nrows; ++r) {
            double m = 0;
            for (int d = 0; d < head_dim; ++d) m = std::max(m, (double) std::fabs(x2[(size_t) r * head_dim + d]));
            const double scale = m > 1e-30 ? m : 1e-30;
            for (int d = 0; d < head_dim; ++d) {
                const double rel = std::fabs((double) got_t[(size_t) r * head_dim + d] -
                                             (double) got_n[(size_t) r * head_dim + d]) / scale;
                worst5 = std::max(worst5, rel);
                if (!(rel <= 3e-3)) ++none_bad5;
            }
        }
        std::printf("  native path vs table path, none            %s (%lld of %d over 3e-3, worst rel %.3e)\n",
                    none_bad5 ? "*** WRONG ***" : "agrees", none_bad5, nrows * head_dim, worst5);
        bad += (int) none_bad5;

        // ---- 6. FUSED RMSNorm + RoPE (`native_qsa_rms_norm_rope`) MUST BE BIT-IDENTICAL to
        // `native_qsa_rms_norm_weighted` followed by `native_rope_apply` for both contiguous
        // (`in_stride == head_dim`) and strided Q/gate split (`in_stride == 2 * head_dim`) inputs,
        // across supported head_dims (128, 256) and both unscaled and YaRN scaling.
        // The kernel is off by default on the HIP build until this check passes there (native_norm_rope_usable): the
        // check then reports SKIPPED, and STRATA_NORM_ROPE=1 runs it (gfx1151, Aurora, 0.1.40: it does not pass).
        const bool run6 = strata::kernels::native_norm_rope_usable(256, 64);
        if (!run6)
            std::printf("  native_qsa_rms_norm_rope                   SKIPPED (the fused kernel is not enabled on this build)\n");
        for (int hd6 : {128, 256}) {
            if (!run6) break;
            const int nr6 = 64;
            const int rows6 = 96;
            std::vector<float> x6_strided((size_t) rows6 * 2 * hd6);
            std::vector<float> gamma6((size_t) hd6);
            std::vector<int> pos6((size_t) rows6);
            for (auto& v : x6_strided) v = gauss2(rng2);
            for (auto& g : gamma6) g = 0.5f + std::fabs(gauss2(rng2));
            for (int r = 0; r < rows6; ++r) pos6[(size_t) r] = (r * 53 + 1) % npos;

            float *d_xs6 = nullptr, *d_g6 = nullptr, *d_ref6 = nullptr, *d_fus6 = nullptr;
            int* d_pos6 = nullptr;
            check(cudaMalloc(&d_xs6, x6_strided.size() * sizeof(float)), "malloc xs6");
            check(cudaMalloc(&d_g6, gamma6.size() * sizeof(float)), "malloc g6");
            check(cudaMalloc(&d_ref6, (size_t) rows6 * hd6 * sizeof(float)), "malloc ref6");
            check(cudaMalloc(&d_fus6, (size_t) rows6 * hd6 * sizeof(float)), "malloc fus6");
            check(cudaMalloc(&d_pos6, pos6.size() * sizeof(int)), "malloc pos6");
            check(cudaMemcpy(d_xs6, x6_strided.data(), x6_strided.size() * sizeof(float), cudaMemcpyHostToDevice), "copy xs6");
            check(cudaMemcpy(d_g6, gamma6.data(), gamma6.size() * sizeof(float), cudaMemcpyHostToDevice), "copy g6");
            check(cudaMemcpy(d_pos6, pos6.data(), pos6.size() * sizeof(int), cudaMemcpyHostToDevice), "copy pos6");

            long long fbad = 0;
            for (const auto& sc6 : {none5, yarn}) {
                // (a) strided Q/gate split (in_stride = 2 * hd6) vs cudaMemcpy2DAsync + norm + rope
                check(cudaMemcpy2DAsync(d_ref6, (size_t) hd6 * sizeof(float), d_xs6, (size_t) 2 * hd6 * sizeof(float),
                                        (size_t) hd6 * sizeof(float), (size_t) rows6, cudaMemcpyDeviceToDevice, cs5),
                      "split6");
                strata::kernels::native_qsa_rms_norm_weighted(d_ref6, d_g6, d_ref6, hd6, rows6, 1e-6f, cs5);
                strata::kernels::native_rope_apply(d_ref6, d_ref6, rows6, hd6, nr6, sc6, d_pos6, cs5);

                strata::kernels::native_qsa_rms_norm_rope(d_xs6, 2 * hd6, d_g6, d_fus6, rows6, hd6, nr6, 1e-6f, sc6,
                                                          d_pos6, cs5);
                check(cudaStreamSynchronize(cs5), "sync6a");
                std::vector<float> href((size_t) rows6 * hd6), hfus((size_t) rows6 * hd6);
                check(cudaMemcpy(href.data(), d_ref6, href.size() * sizeof(float), cudaMemcpyDeviceToHost), "back ref6a");
                check(cudaMemcpy(hfus.data(), d_fus6, hfus.size() * sizeof(float), cudaMemcpyDeviceToHost), "back fus6a");
                for (size_t i = 0; i < href.size(); ++i) {
                    if (std::memcmp(&href[i], &hfus[i], 4) != 0) ++fbad;
                }

                // (b) contiguous in-place (in_stride = hd6, output == input)
                check(cudaMemcpy2DAsync(d_ref6, (size_t) hd6 * sizeof(float), d_xs6, (size_t) 2 * hd6 * sizeof(float),
                                        (size_t) hd6 * sizeof(float), (size_t) rows6, cudaMemcpyDeviceToDevice, cs5),
                      "contig6_ref");
                check(cudaMemcpy(d_fus6, d_ref6, (size_t) rows6 * hd6 * sizeof(float), cudaMemcpyDeviceToDevice),
                      "contig6_fus");
                strata::kernels::native_qsa_rms_norm_weighted(d_ref6, d_g6, d_ref6, hd6, rows6, 1e-6f, cs5);
                strata::kernels::native_rope_apply(d_ref6, d_ref6, rows6, hd6, nr6, sc6, d_pos6, cs5);
                strata::kernels::native_qsa_rms_norm_rope(d_fus6, hd6, d_g6, d_fus6, rows6, hd6, nr6, 1e-6f, sc6,
                                                          d_pos6, cs5);
                check(cudaStreamSynchronize(cs5), "sync6b");
                check(cudaMemcpy(href.data(), d_ref6, href.size() * sizeof(float), cudaMemcpyDeviceToHost), "back ref6b");
                check(cudaMemcpy(hfus.data(), d_fus6, hfus.size() * sizeof(float), cudaMemcpyDeviceToHost), "back fus6b");
                for (size_t i = 0; i < href.size(); ++i) {
                    if (std::memcmp(&href[i], &hfus[i], 4) != 0) ++fbad;
                }
            }
            std::printf("  native_qsa_rms_norm_rope hd=%3d nr=%2d      %s (%lld of %d elements differ)\n",
                        hd6, nr6, fbad ? "*** WRONG ***" : "bit-identical", fbad, rows6 * hd6 * 4);
            bad += (int) fbad;
            cudaFree(d_xs6);
            cudaFree(d_g6);
            cudaFree(d_ref6);
            cudaFree(d_fus6);
            cudaFree(d_pos6);
        }
    }

    std::printf("\nrope: %d failures\n", bad);
    if (bad) return 1;
    if (selftest) std::printf("rope_parity OK\n");
    return 0;
}
