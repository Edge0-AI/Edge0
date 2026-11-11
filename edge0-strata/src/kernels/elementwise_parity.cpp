// src/kernels/elementwise_parity.cpp - P2.S5's test for the layer glue.
//
// The ops are small; the CONVENTIONS in them are not, and each one is a place where a plausible reading
// gives a plausible number:
//
//   1. `softplus` HAS A LARGE-x BRANCH.  `log1p(exp(x))` overflows f32 at x > 88 and loses relative precision
//      well before that; `ggml_compute_softplus_f32` returns `x` above 20.  A rival without the branch is
//      asserted observable, because a gate that saturates to +inf makes `exp(gate)` inf and the whole
//      recurrence NaN - which reads as a state bug and not as a missing branch.
//   2. `ssm_a` IS NEGATIVE.  `gate = softplus(...) * ssm_a`, so `exp(gate) < 1` and the state DECAYS.  A
//      fixture with a positive `ssm_a` would still produce finite output and the recurrence would blow up
//      instead of decaying, so the sign is checked as a property and not assumed.
//   3. `silu` IS COMPUTED IN DOUBLE then cast, because `ref/gdn.py`'s numpy does.  f32 `expf` differs in the
//      last bits, and the test measures that rather than asserting it away.
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/f16_bits.hpp"

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
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: elementwise_parity [--selftest]\n"); return 2; }
    }
    int bad = 0;
    const int64_t H_V = 48;

    // ---- 1. gdn_gate, with the reference's own softplus
    {
        const int n = (int) H_V;
        std::mt19937 rng(11);
        std::normal_distribution<float> g(0.0f, 1.0f);
        std::vector<float> alpha(n), dt(n), a(n), want(n);
        for (int i = 0; i < n; ++i) {
            // A MIXTURE THAT ACTUALLY CROSSES THE BRANCH.  The first version drew alpha from a normal with
            // sigma 3, so the largest value was about 9 and `softplus` never took its `x > 20` path - the
            // "branch is observable" check then reported **0.00% apart, 0 non-finite**, which is the fixture
            // saying it cannot see the thing it was written to see.  Every third head is now large.
            const bool large = (i % 3) == 0;
            // Up to ~120, so the fixture spans WELL PAST f32's `exp` overflow at 88.  A fixture that stopped
            // at 88 would show the branch as unobservable and would be right: `log1pf(expf(x))` equals `x` to
            // f32 precision for the whole range 20..88, so ggml's threshold of 20 is CONSERVATIVE and the
            // behaviour only actually changes where `expf` overflows.
            alpha[i] = large ? (22.0f + 6.0f * (float) (i % 17)) : g(rng) * 3.0f;
            dt[i] = g(rng) * 0.5f;
            // NEGATIVE, as the artifact's `ssm_a = -exp(A_log)` is.  Checked below as a property.
            a[i] = -(std::fabs(g(rng)) + 0.1f);
            const double x = (double) alpha[i] + (double) dt[i];
            const double sp = x > 20.0 ? x : std::log1p(std::exp(x));
            want[(size_t) i] = (float) (sp * (double) a[i]);
        }
        // the fixture must EXERCISE the branch, or check 1 below measures nothing
        {
            int over = 0, past_overflow = 0;
            for (int i = 0; i < n; ++i) {
                const float x = alpha[(size_t) i] + dt[(size_t) i];
                if (x > 20.0f) ++over;
                if (x > 88.0f) ++past_overflow;
            }
            std::printf("  %-44s %s (%d above 20, %d above 88)\n", "the fixture crosses the branch",
                        over ? "yes" : "*** NO ***", over, past_overflow);
            if (!over) ++bad;
        }
        // the sign property, asserted rather than assumed
        int positive = 0;
        for (int i = 0; i < n; ++i) if (a[(size_t) i] >= 0.0f) ++positive;
        std::printf("  %-44s %s (%d of %d non-negative)\n", "the fixture's ssm_a is negative",
                    positive ? "*** NO ***" : "yes", positive, n);
        if (positive) ++bad;

        float *d_a = nullptr, *d_dt = nullptr, *d_sa = nullptr, *d_g = nullptr;
        check(cudaMalloc(&d_a, n * 4), "a");
        check(cudaMalloc(&d_dt, n * 4), "dt");
        check(cudaMalloc(&d_sa, n * 4), "sa");
        check(cudaMalloc(&d_g, n * 4), "g");
        check(cudaMemcpy(d_a, alpha.data(), n * 4, cudaMemcpyHostToDevice), "ca");
        check(cudaMemcpy(d_dt, dt.data(), n * 4, cudaMemcpyHostToDevice), "cd");
        check(cudaMemcpy(d_sa, a.data(), n * 4, cudaMemcpyHostToDevice), "cs");
        strata::kernels::gdn_gate(d_a, d_dt, d_sa, d_g, 1, H_V, nullptr);
        std::vector<float> got((size_t) n);
        check(cudaMemcpy(got.data(), d_g, n * 4, cudaMemcpyDeviceToHost), "cg");

        const double rel = rel_l1(want, got);
        std::printf("  %-44s rel %.3e\n", "gdn_gate vs the reference", rel);
        // double softplus and double multiply on both sides, then one cast
        if (!(rel <= 1e-6)) { std::printf("    *** over 1e-6 ***\n"); ++bad; }

        // TRAP: no large-x branch.  `log1p(exp(x))` in f32, with no `x > 20` path.
        //
        // AND THE MEASUREMENT SAYS WHERE THE BRANCH ACTUALLY MATTERS.  The first version of this fixture
        // spanned 20..88 and reported **0 heads differ**: `log1pf(expf(x))` agrees with `x` to f32 precision
        // over that whole range, so ggml's threshold of 20 is CONSERVATIVE - the branch changes the answer only
        // where `expf` OVERFLOWS, at about 88.  The check reports the smallest x at which the two readings
        // part company, so the number is in the output rather than in this comment.
        {
            int wrong = 0, differing = 0;
            float first_differ = -1.0f;
            std::vector<float> rival((size_t) n);
            for (int i = 0; i < n; ++i) {
                const float x = alpha[(size_t) i] + dt[(size_t) i];
                const float sp = std::log1p(std::exp(x));       // no branch, f32
                rival[(size_t) i] = sp * a[(size_t) i];
                if (!std::isfinite(sp)) ++wrong;
                const float branched = x > 20.0f ? x : std::log1pf(std::exp(x));
                if (sp != branched) {
                    ++differing;
                    if (first_differ < 0.0f || x < first_differ) first_differ = x;
                }
            }
            const double r = rel_l1(want, rival);
            const bool visible = r > 0.05 || wrong > 0;
            std::printf("  %-44s %s (%.2f%% apart, %d non-finite, %d heads differ, first at x = %.1f)\n",
                        "the softplus large-x branch is observable", visible ? "yes" : "*** NO ***",
                        r * 100, wrong, differing, (double) first_differ);
            if (!visible) ++bad;
        }
        // PROPERTY: exp(gate) < 1 for every element, which is what makes the state decay
        {
            int over = 0;
            for (float v : got) if (!(std::exp(v) < 1.0f)) ++over;
            std::printf("  %-44s %s (%d of %d not < 1)\n", "exp(gate) < 1 for every head",
                        over ? "*** NO ***" : "yes", over, n);
            if (over) ++bad;
        }
        cudaFree(d_a); cudaFree(d_dt); cudaFree(d_sa); cudaFree(d_g);
    }

    // ---- 2. silu, in double then cast
    {
        const int n = 4096;
        std::mt19937 rng(22);
        std::normal_distribution<float> g(0.0f, 3.0f);
        std::vector<float> x((size_t) n), want((size_t) n);
        for (int i = 0; i < n; ++i) {
            x[(size_t) i] = g(rng);
            const double v = (double) x[(size_t) i];
            want[(size_t) i] = (float) (v / (1.0 + std::exp(-v)));
        }
        float* d_x = nullptr;
        check(cudaMalloc(&d_x, n * 4), "sx");
        check(cudaMemcpy(d_x, x.data(), n * 4, cudaMemcpyHostToDevice), "csx");
        strata::kernels::silu_inplace(d_x, n, nullptr);
        std::vector<float> got((size_t) n);
        check(cudaMemcpy(got.data(), d_x, n * 4, cudaMemcpyDeviceToHost), "csg");
        const double rel = rel_l1(want, got);
        std::printf("\n  %-44s rel %.3e\n", "silu (double) vs the reference", rel);
        if (!(rel <= 1e-7)) { std::printf("    *** over 1e-7 ***\n"); ++bad; }
        cudaFree(d_x);
    }

    // ---- 3. scale and the f32->f16 bridge
    {
        const int n = 1024;
        std::vector<float> x((size_t) n), want((size_t) n);
        for (int i = 0; i < n; ++i) x[(size_t) i] = (float) (i - n / 2) * 0.013f;
        const float s = 1.0f / std::sqrt(128.0f);
        for (int i = 0; i < n; ++i) want[(size_t) i] = x[(size_t) i] * s;

        float* d_x = nullptr;
        uint16_t* d_h = nullptr;
        check(cudaMalloc(&d_x, n * 4), "ex");
        check(cudaMalloc(&d_h, n * 2), "eh");
        check(cudaMemcpy(d_x, x.data(), n * 4, cudaMemcpyHostToDevice), "cex");
        strata::kernels::scale_inplace(d_x, n, s, nullptr);
        std::vector<float> got((size_t) n);
        check(cudaMemcpy(got.data(), d_x, n * 4, cudaMemcpyDeviceToHost), "ceg");
        int diff = 0;
        for (int i = 0; i < n; ++i) if (got[(size_t) i] != want[(size_t) i]) ++diff;
        std::printf("  %-44s %d of %d differ\n", "scale_inplace is exact", diff, n);
        if (diff) ++bad;

        strata::kernels::f32_to_f16_bulk(d_x, d_h, n, nullptr);
        std::vector<uint16_t> h((size_t) n);
        check(cudaMemcpy(h.data(), d_h, n * 2, cudaMemcpyDeviceToHost), "ceh");
        int hbad = 0;
        for (int i = 0; i < n; ++i)
            if (h[(size_t) i] != strata::kernels::f16_from_f32(got[(size_t) i])) ++hbad;
        std::printf("  %-44s %d of %d differ\n", "f32_to_f16_bulk uses the shared conversion", hbad, n);
        if (hbad) ++bad;
        cudaFree(d_x); cudaFree(d_h);
    }

    // ---- 4. rms_norm_weighted, and the TWO RIVAL READINGS it exists to be told apart from.
    //
    // This kernel's whole reason for being a separate entry point from `gdn_l2_norm` is that the two differ by
    // one `/ cols`, and that a swap produces a well-scaled, plausible tensor.  So the test computes BOTH wrong
    // readings and requires each to differ from the right one by more than the tolerance before it judges the
    // kernel - the same discipline the shared-expert test uses for silu-on-gate.
    {
        const int64_t rows = 24, cols = 256;          // the real `attn_q`-norm shape
        const float eps = 1e-6f;
        std::vector<float> x((size_t) (rows * cols));
        for (size_t i = 0; i < x.size(); ++i) x[i] = (float) std::sin((double) i * 0.017) * 1.7f + 0.4f;
        std::vector<float> w((size_t) cols);
        for (int64_t c = 0; c < cols; ++c) w[(size_t) c] = 0.5f + 0.002f * (float) (c % 97);

        // the oracle: f64, mean, times w
        auto ref = [&](bool use_sum, bool use_w) {
            std::vector<double> y((size_t) (rows * cols));
            for (int64_t r = 0; r < rows; ++r) {
                double ss = 0;
                for (int64_t c = 0; c < cols; ++c) {
                    const double v = x[(size_t) (r * cols + c)];
                    ss += v * v;
                }
                const double den = std::sqrt((use_sum ? ss : ss / (double) cols) + (double) eps);
                for (int64_t c = 0; c < cols; ++c) {
                    const double v = x[(size_t) (r * cols + c)];
                    y[(size_t) (r * cols + c)] = (v / den) * (use_w ? (double) w[(size_t) c] : 1.0);
                }
            }
            std::vector<float> out((size_t) (rows * cols));
            for (size_t i = 0; i < out.size(); ++i) out[i] = (float) y[i];
            return out;
        };
        const std::vector<float> want = ref(false, true);
        const std::vector<float> rival_sum = ref(true, false);    // what gdn_l2_norm computes
        const std::vector<float> rival_now = ref(false, false);   // the (1+w)-vs-w / no-weight confusion

        auto rel = [&](const std::vector<float>& a, const std::vector<float>& b) {
            double d = 0, m = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                d += std::fabs((double) a[i] - (double) b[i]);
                m += std::fabs((double) a[i]);
            }
            return d / (m > 1e-30 ? m : 1e-30);
        };
        const double r_sum = rel(want, rival_sum), r_now = rel(want, rival_now);
        std::printf("  %-44s %.2f%% apart\n", "sum-vs-mean is observable", r_sum * 100);
        std::printf("  %-44s %.2f%% apart\n", "with-w-vs-without-w is observable", r_now * 100);
        // `sqrt(cols)` = 16 for the first and the weight's own spread for the second; a fixture that cannot
        // separate them is not testing the kernel that was written.
        if (!(r_sum > 0.5)) { std::printf("  *** the sum reading is NOT observable ***\n"); ++bad; }
        if (!(r_now > 0.5)) { std::printf("  *** the no-weight reading is NOT observable ***\n"); ++bad; }

        float* d_x = nullptr;
        float* d_w = nullptr;
        check(cudaMalloc(&d_x, x.size() * 4), "m rmsx");
        check(cudaMalloc(&d_w, w.size() * 4), "m rmsw");
        check(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "c rmsx");
        check(cudaMemcpy(d_w, w.data(), w.size() * 4, cudaMemcpyHostToDevice), "c rmsw");
        strata::kernels::rms_norm_weighted(d_x, d_w, rows, cols, eps, nullptr);
        std::vector<float> got((size_t) (rows * cols));
        check(cudaMemcpy(got.data(), d_x, got.size() * 4, cudaMemcpyDeviceToHost), "c rmsg");

        const double r_got = rel(want, got);
        double worst = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            const double m = std::fabs((double) want[i]);
            const double e = std::fabs((double) got[i] - (double) want[i]) / (m > 1e-30 ? m : 1e-30);
            if (e > worst) worst = e;
        }
        int nonfinite = 0;
        for (float v : got) if (!std::isfinite(v)) ++nonfinite;
        std::printf("  %-44s rel %.3e  worst %.3e  nonfinite %d\n", "rms_norm_weighted vs f64 oracle", r_got,
                    worst, nonfinite);
        if (nonfinite) ++bad;
        // The kernel accumulates in f32 over 256 terms; 1e-6 is loose enough for that and far tighter than the
        // 16x the rival readings are off by.
        if (!(r_got < 1e-6)) { std::printf("  *** rms_norm_weighted is WRONG ***\n"); ++bad; }
        // and a null `w` must be legal, because `ref/qsa.py` allows it.
        //
        // THE INPUT HAS TO BE RE-UPLOADED FIRST.  The kernel is IN PLACE, so `d_x` currently holds the
        // NORMALISED tensor from the call above; running it again normalises a second time and every element
        // differs.  That is exactly what the first version of this check reported - "6144 of 6144 differ" -
        // and it is a fixture bug rather than a kernel bug: the oracle reads the ORIGINAL `x`, so the two
        // sides were never looking at the same input.
        const std::vector<float> want_null = ref(false, false);
        check(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "c rmsx2");
        strata::kernels::rms_norm_weighted(d_x, nullptr, rows, cols, eps, nullptr);
        check(cudaMemcpy(got.data(), d_x, got.size() * 4, cudaMemcpyDeviceToHost), "c rmsn");
        int null_bad = 0;
        for (size_t i = 0; i < got.size(); ++i)
            if (std::fabs((double) got[i] - (double) want_null[i]) > 1e-6) ++null_bad;
        std::printf("  %-44s %d of %zu differ\n", "a null weight is legal", null_bad, got.size());
        if (null_bad) ++bad;
        cudaFree(d_x);
        cudaFree(d_w);
    }

    // Packed embedding rows: every code width, scale-group size, optional offset,
    // row boundary and partial CUDA block. Compare bits with a scalar CPU decoder.
    // Volatile materializes the multiply so this oracle cannot silently use FMA.
    {
        cudaStream_t stream;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "embedding stream");
        std::mt19937 rng(77);
        std::uniform_real_distribution<float> values(-2.0f, 2.0f);
        int mismatches = 0, guards = 0, fma_diff = 0, cases = 0;
        for (const int bits : {2, 4, 8}) {
            const int bias = bits == 2 ? -1 : bits == 4 ? -7 : -16;
            const int per_byte = 8 / bits;
            const unsigned mask = (1u << bits) - 1u;
            for (const int group : {16, 32, 64}) {
                for (const int n : {320, 2560}) {
                    const int row_bytes = n / per_byte, row_groups = n / group;
                    std::vector<uint8_t> codes((size_t) 3 * row_bytes, 0);
                    std::vector<float> scales((size_t) 3 * row_groups), offsets(scales.size());
                    std::vector<int> unpacked((size_t) 3 * n);
                    for (size_t i = 0; i < unpacked.size(); ++i) {
                        const unsigned code = (unsigned) (i * 13 + i / n * 7) & mask;
                        unpacked[i] = (int) code;
                        codes[i / per_byte] |= (uint8_t) (code << ((i % per_byte) * bits));
                    }
                    for (size_t i = 0; i < scales.size(); ++i) {
                        scales[i] = values(rng);
                        offsets[i] = values(rng);
                    }
                    scales[0] = -0.0f;
                    offsets[0] = 0.0f;
                    uint8_t* dc = nullptr;
                    float *ds = nullptr, *dof = nullptr, *out = nullptr;
                    check(cudaMalloc(&dc, codes.size()), "embedding codes");
                    check(cudaMalloc(&ds, scales.size() * sizeof(float)), "embedding scales");
                    check(cudaMalloc(&dof, offsets.size() * sizeof(float)), "embedding offsets");
                    check(cudaMalloc(&out, ((size_t) n + 2) * sizeof(float)), "embedding output");
                    check(cudaMemcpyAsync(dc, codes.data(), codes.size(), cudaMemcpyHostToDevice, stream), "embedding codes upload");
                    check(cudaMemcpyAsync(ds, scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "embedding scales upload");
                    check(cudaMemcpyAsync(dof, offsets.data(), offsets.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "embedding offsets upload");
                    check(cudaStreamSynchronize(stream), "embedding upload sync");

                    for (const bool with_offset : {false, true}) {
                        for (int row = 0; row < 3; ++row) {
                            std::vector<float> want((size_t) n), got((size_t) n + 2, -12345.0f);
                            for (int i = 0; i < n; ++i) {
                                const size_t gi = (size_t) row * row_groups + i / group;
                                const float code = (float) (unpacked[(size_t) row * n + i] + bias);
                                volatile float product = code * scales[gi];
                                const float offset = with_offset ? offsets[gi] : 0.0f;
                                want[(size_t) i] = product + offset;
                                const float fused = std::fma(code, scales[gi], offset);
                                if (std::memcmp(&fused, &want[(size_t) i], sizeof(float)) != 0) ++fma_diff;
                            }
                            check(cudaMemcpyAsync(out, got.data(), got.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "embedding output guards");
                            strata::kernels::embedding_gather(dc + (size_t) row * row_bytes,
                                ds + (size_t) row * row_groups,
                                with_offset ? dof + (size_t) row * row_groups : nullptr,
                                n, bits, bias, group, out + 1, stream);
                            check(cudaMemcpyAsync(got.data(), out, got.size() * sizeof(float), cudaMemcpyDeviceToHost, stream), "embedding result");
                            check(cudaStreamSynchronize(stream), "embedding sync");
                            for (int i = 0; i < n; ++i) {
                                if (std::memcmp(&want[(size_t) i], &got[(size_t) i + 1], sizeof(float)) != 0) ++mismatches;
                            }
                            if (got.front() != -12345.0f || got.back() != -12345.0f) ++guards;
                            ++cases;

                            // Capturing the gather proves that it has no hidden synchronization.
                            // A following scale also checks that it uses the caller's stream.
                            if (row == 1) {
                                cudaGraph_t graph;
                                cudaGraphExec_t exec;
                                check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "embedding capture");
                                strata::kernels::embedding_gather(dc + (size_t) row * row_bytes,
                                    ds + (size_t) row * row_groups,
                                    with_offset ? dof + (size_t) row * row_groups : nullptr,
                                    n, bits, bias, group, out + 1, stream);
                                strata::kernels::scale_inplace(out + 1, n, 2.0f, stream);
                                check(cudaStreamEndCapture(stream, &graph), "embedding capture end");
                                check(cudaGraphInstantiate(&exec, graph, 0), "embedding instantiate");
                                check(cudaGraphLaunch(exec, stream), "embedding replay");
                                check(cudaMemcpyAsync(got.data(), out, got.size() * sizeof(float), cudaMemcpyDeviceToHost, stream), "embedding graph result");
                                check(cudaStreamSynchronize(stream), "embedding graph sync");
                                for (int i = 0; i < n; ++i) {
                                    const float expected = want[(size_t) i] * 2.0f;
                                    if (std::memcmp(&expected, &got[(size_t) i + 1], sizeof(float)) != 0) ++mismatches;
                                }
                                if (got.front() != -12345.0f || got.back() != -12345.0f) ++guards;
                                check(cudaGraphExecDestroy(exec), "embedding exec destroy");
                                check(cudaGraphDestroy(graph), "embedding graph destroy");
                            }
                        }
                    }
                    cudaFree(dc); cudaFree(ds); cudaFree(dof); cudaFree(out);
                }
            }
        }
        check(cudaStreamDestroy(stream), "embedding stream destroy");
        std::printf("  embedding gather: %d row cases, %d bit mismatches, %d guard failures, %d FMA differences\n",
                    cases, mismatches, guards, fma_diff);
        if (mismatches || guards || fma_diff == 0) ++bad;
    }

    // ---- THE f32 -> bf16 CONVERSIONS KEEP A NaN A NaN.  `f32_to_bf16_bulk` (the header's `bf16_from_f32`
    // on the device) and the prompt path's dequantizer (its own `f2bf`) against ggml_compute_fp32_to_bf16's rule;
    // before the fix 0x7FFFFFFF came back as -0 and 0x7F800001 as +inf.  `bf16_bits_test` covers every f32 on
    // the host; this is the same header compiled by nvcc, plus the second copy in dequant_bf16.cu.
    {
        auto ggml_bf16 = [](uint32_t u) -> uint16_t {
            if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 64);
            return (uint16_t) ((u + (0x7fffu + ((u >> 16) & 1u))) >> 16);
        };
        std::vector<uint32_t> bits = {0x7FFFFFFFu, 0xFFFFFFFFu, 0x7F800001u, 0xFF800001u, 0x7FC00000u, 0x7FBFFFFFu,
                                      0x7F800000u, 0xFF800000u, 0x7F7FFFFFu, 0x00000000u, 0x80000000u, 0x3F808000u,
                                      0x3F818000u, 0x00008000u, 0x7F7F8000u};
        std::mt19937 rng(13);
        while (bits.size() < 4096) bits.push_back((uint32_t) rng());
        const size_t n = bits.size();
        float* dx = nullptr;
        uint16_t* dy = nullptr;
        check(cudaMalloc(&dx, n * 4), "bf16 x");
        check(cudaMalloc(&dy, n * 2), "bf16 y");
        check(cudaMemcpy(dx, bits.data(), n * 4, cudaMemcpyHostToDevice), "bf16 cx");
        strata::kernels::f32_to_bf16_bulk(dx, dy, (int64_t) n, nullptr);
        std::vector<uint16_t> got(n);
        check(cudaMemcpy(got.data(), dy, n * 2, cudaMemcpyDeviceToHost), "bf16 cy");
        int wrong = 0;
        for (size_t i = 0; i < n; ++i) wrong += got[i] != ggml_bf16(bits[i]);

        // A Q8_0 block (type 8) whose fp16 scale is a NaN: every dequantized value is NaN * q, a NaN, and it must
        // still be one in BF16.  A second block with an ordinary scale checks the finite path did not move.
        std::vector<uint8_t> blk(2 * 34, 0);
        blk[0] = 0x00; blk[1] = 0x7E;                        // fp16 quiet NaN
        blk[34] = 0x00; blk[35] = 0x3C;                      // fp16 1.0
        for (int j = 0; j < 32; ++j) {
            blk[(size_t) (2 + j)] = (uint8_t) (int8_t) (j - 16);
            blk[(size_t) (36 + j)] = (uint8_t) (int8_t) (3 * j - 50);
        }
        uint8_t* db = nullptr;
        uint16_t* dq = nullptr;
        check(cudaMalloc(&db, blk.size()), "q8 blk");
        check(cudaMalloc(&dq, 64 * 2), "q8 out");
        check(cudaMemcpy(db, blk.data(), blk.size(), cudaMemcpyHostToDevice), "q8 cblk");
        strata::kernels::dequant_bf16(8, db, 0, 2, 32, dq, nullptr);
        check(cudaDeviceSynchronize(), "dequant_bf16");
        std::vector<uint16_t> q(64);
        check(cudaMemcpy(q.data(), dq, 64 * 2, cudaMemcpyDeviceToHost), "q8 cout");
        int dq_wrong = 0;
        for (int j = 0; j < 32; ++j) {
            if (!((q[(size_t) j] & 0x7FFFu) > 0x7F80u)) ++dq_wrong;           // row 0: NaN scale -> NaN
            const float v = (float) (3 * j - 50);                                // row 1: d = 1.0 -> the integer
            uint32_t vb;
            std::memcpy(&vb, &v, 4);
            if (q[(size_t) (32 + j)] != ggml_bf16(vb)) ++dq_wrong;
        }
        std::printf("  f32 -> bf16 (NaN kept): bulk %d of %zu differ from ggml, dequant_bf16 %d of 64 wrong\n",
                    wrong, n, dq_wrong);
        if (wrong || dq_wrong) ++bad;
        cudaFree(dx); cudaFree(dy); cudaFree(db); cudaFree(dq);
    }

    std::printf("\nelementwise: %d failures\n", bad);
    if (bad) return 1;
    if (selftest) std::printf("elementwise_parity OK\n");
    return 0;
}
