// src/kernels/gr_parity.cpp - P2.S2's test for the gated residual / hyper-connection.
//
// `ref/gr.py` opens by listing the details "a prose reading gets wrong", and every one of them is a reading
// that has the RIGHT SHAPES and produces a PLAUSIBLE number.  So each is computed here the wrong way round and
// required to differ materially BEFORE the kernel is judged against the right one.  A test that only does the
// second half passes against either reading:
//
//   1. PER-STREAM RMSNorm.  The source comment reads ambiguously; the code reduces over ne[0] = n_embd, i.e.
//      one RMS per stream.  The rival reading is one RMS over the whole hc*n_embd stack.
//   2. `/ hc` INSIDE the silu: `lo = silu(proj / hc)`.  Moving it outside is a one-line change that keeps
//      every shape.
//   3. SiLU on `lo` and SIGMOID on the gate, a few lines apart in the same function.  Swapping them is the
//      obvious slip, and both are monotone saturating functions so the magnitudes stay comparable.
//   4. `mean` over the stream axis, not `sum`.  A factor of hc = 4, which is exactly the kind of error that
//      looks like a scale problem rather than a structural one.
//   5. ACTIVATION PRECISION. The historical BF16-rounded CPU contract and the pinned CUDA single-token
//      FP32-activation contract must be distinguishable. Both are tested against explicit references;
//      neither is selected merely because the weight is BF16. The scalar fixture isolates xn and lo
//      rounding from reduction order, and graph replay must preserve the precision chosen at capture.
//   6. `gr_write`'s `2*sigmoid`, which centres the gate on 1 so a ZERO injection is a plain residual add.
//      Asserted as a property, not as a value, because that is what the source comment claims.
#include "strata/kernels/gr.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <algorithm>
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

/// The same rule the kernel uses, on the host - `ref/quant.py::bf16`.
float to_bf16(float f) {
    uint32_t i;
    std::memcpy(&i, &f, 4);
    i = (i + ((i >> 16) & 1u) + 0x7FFFu) & 0xFFFF0000u;
    float o;
    std::memcpy(&o, &i, 4);
    return o;
}

uint16_t bf16_bits(float f) {
    uint32_t i;
    std::memcpy(&i, &f, 4);
    return (uint16_t) (i >> 16);
}

struct Opts {
    bool per_stream_norm = true;   ///< false: one RMS over the whole stack (WRONG)
    bool scale_inside_silu = true; ///< false: silu(proj)/hc instead of silu(proj/hc) (WRONG)
    bool sigmoid_on_gate = true;   ///< false: silu on the gate and sigmoid on lo (WRONG)
    bool mean_over_streams = true; ///< false: sum over streams instead of mean (WRONG)
    bool round_activation = true;  ///< false: pinned CUDA single-token BF16 MMVF contract
};

/// `ref/gr.py::gr_read`, transcribed with switches for the rival readings.
void reference(const std::vector<float>& R, const std::vector<float>& w_norm,
               const std::vector<float>& w_down, const std::vector<float>& w_up,
               const std::vector<float>& w_inject, float eps, long long n_embd, long long hc, long long hc_lr,
               const Opts& o, std::vector<float>& mixed, std::vector<float>& inject) {
    const long long hc_dim = hc * n_embd;
    std::vector<float> xn((size_t) hc_dim);
    if (o.per_stream_norm) {
        for (long long c = 0; c < hc; ++c) {
            double ms = 0;
            for (long long d = 0; d < n_embd; ++d) ms += (double) R[(size_t) (c * n_embd + d)] * R[(size_t) (c * n_embd + d)];
            ms /= (double) n_embd;
            const float rs = (float) (1.0 / std::sqrt(ms + (double) eps));
            for (long long d = 0; d < n_embd; ++d) {
                const size_t i = (size_t) (c * n_embd + d);
                xn[i] = R[i] * rs * w_norm[i];
            }
        }
    } else {
        double ms = 0;
        for (long long i = 0; i < hc_dim; ++i) ms += (double) R[(size_t) i] * R[(size_t) i];
        ms /= (double) hc_dim;
        const float rs = (float) (1.0 / std::sqrt(ms + (double) eps));
        for (long long i = 0; i < hc_dim; ++i) xn[(size_t) i] = R[(size_t) i] * rs * w_norm[(size_t) i];
    }

    std::vector<float> act((size_t) hc_dim);
    for (long long i = 0; i < hc_dim; ++i) act[(size_t) i] = o.round_activation ? to_bf16(xn[(size_t) i]) : xn[(size_t) i];

    // lo
    std::vector<float> lo((size_t) hc_lr);
    for (long long k = 0; k < hc_lr; ++k) {
        double a = 0;
        for (long long i = 0; i < hc_dim; ++i) a += (double) act[(size_t) i] * (double) w_down[(size_t) (k * hc_dim + i)];
        const float p = (float) a;
        if (o.sigmoid_on_gate) lo[(size_t) k] = o.scale_inside_silu ? (p / (float) hc) / (1.0f + std::exp(-(p / (float) hc)))
                                                                  : (p / (1.0f + std::exp(-p))) / (float) hc;
        else                   lo[(size_t) k] = o.scale_inside_silu ? 1.0f / (1.0f + std::exp(-(p / (float) hc)))
                                                                    : (1.0f / (1.0f + std::exp(-p))) / (float) hc;
    }
    std::vector<float> lq((size_t) hc_lr);
    for (long long k = 0; k < hc_lr; ++k)
        lq[(size_t) k] = o.round_activation ? to_bf16(lo[(size_t) k]) : lo[(size_t) k];

    // gate and the gated mean
    mixed.assign((size_t) n_embd, 0.0f);
    for (long long d = 0; d < n_embd; ++d) {
        float m = 0.0f;
        for (long long c = 0; c < hc; ++c) {
            const long long i = c * n_embd + d;
            double a = 0;
            for (long long k = 0; k < hc_lr; ++k) a += (double) lq[(size_t) k] * (double) w_up[(size_t) (i * hc_lr + k)];
            const float g = (float) a;
            const float s = o.sigmoid_on_gate ? 1.0f / (1.0f + std::exp(-g)) : g / (1.0f + std::exp(-g));
            m += xn[(size_t) i] * s;
        }
        mixed[(size_t) d] = o.mean_over_streams ? m / (float) hc : m;
    }

    inject.assign((size_t) hc, 0.0f);
    for (long long c = 0; c < hc; ++c) {
        double a = 0;
        for (long long i = 0; i < hc_dim; ++i) a += (double) act[(size_t) i] * (double) w_inject[(size_t) (c * hc_dim + i)];
        inject[(size_t) c] = (float) a;
    }
}

/// Normalised L1 difference.  `mag_out` returns the mean |a|, for reporting.
double rel_diff(const std::vector<float>& a, const std::vector<float>& b, double* mag_out = nullptr) {
    double d = 0, mag = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += std::fabs((double) a[i] - (double) b[i]);
        mag += std::fabs((double) a[i]);
    }
    if (mag_out) *mag_out = mag / (double) (a.empty() ? 1 : a.size());
    return d / (mag > 1e-30 ? mag : 1e-30);
}

/// Relative error against the magnitude of the TERMS, not of the result.
///
/// `R + block_out * w` cancels wherever block_out*w is close to -R, and a plain |want-got|/|want| then
/// reports the CONDITION NUMBER instead of the arithmetic.  This is the same metric mistake the project has
/// now made four times: rounds 169 (Q4_K), 189 (RoPE), 194 (shared_expert) and this one.  For a sum, the
/// denominator is the sum of the term magnitudes.
double rel_terms(double want, double got, double term_a, double term_b) {
    const double den = std::fabs(term_a) + std::fabs(term_b);
    return std::fabs(want - got) / (den > 1e-30 ? den : 1e-30);
}

const char* activation_mode_name(int mode) {
    return mode == 2 ? "native pinned GR" : (mode == 1 ? "FP32" : "BF16");
}
void select_activation_mode(int mode) {
    strata::kernels::gr_set_native_mmvf(mode == 2);
    // In mode 2 the precision-only switch is deliberately false: native MMVF must imply FP32 by itself.
    strata::kernels::gr_set_fp32_activations(mode == 1);
}

// A diagonal GR fixture makes activation precision observable without summation-order ambiguity.
// Two-wide matrices satisfy native MMVF's pair ABI, while each nonzero dot has only one nonzero product.
// At R=1 and eps=0, xn=gamma exactly; any rounding in xn or lo comes from the selected activation contract.
int scalar_activation_contract() {
    using namespace strata::kernels;
    const GrShapes sh{2, 1, 2};
    float *d_R = nullptr, *d_norm = nullptr, *d_mixed = nullptr, *d_inject = nullptr;
    uint16_t* d_weights = nullptr;
    void* d_scratch = nullptr;
    check(cudaMalloc(&d_R, 2 * sizeof(float)), "scalar R");
    check(cudaMalloc(&d_norm, 2 * sizeof(float)), "scalar norm");
    check(cudaMalloc(&d_mixed, 2 * sizeof(float)), "scalar mixed");
    check(cudaMalloc(&d_inject, sizeof(float)), "scalar inject");
    check(cudaMalloc(&d_weights, 10 * sizeof(uint16_t)), "scalar weights");
    check(cudaMalloc(&d_scratch, gr_workspace_bytes(sh)), "scalar scratch");
    GrWorkspace ws;
    gr_workspace_init(sh, d_scratch, ws);
    const float ones[] = {1.0f, 1.0f};
    const uint16_t weights[] = {
        bf16_bits(0.5f), 0, 0, 0,             // down: lo[0]=silu(xn[0]/2), lo[1]=0
        bf16_bits(1.5f), 0, bf16_bits(1.5f), 0, // both gate rows consume lo[0]
        bf16_bits(2.0f), 0                     // injection: 2*xn[0]
    };
    check(cudaMemcpy(d_R, ones, sizeof(ones), cudaMemcpyHostToDevice), "scalar upload R");
    check(cudaMemcpy(d_weights, weights, sizeof(weights), cudaMemcpyHostToDevice), "scalar upload weights");
    // the uploads run on the legacy stream from pageable memory, which the non-blocking `stream` does not wait for
    check(cudaDeviceSynchronize(), "scalar setup");
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "scalar stream");
    int bad = 0;
    for (float gamma : {1.0f, 1.00390625f}) {
        const float gammas[] = {gamma, gamma};
        check(cudaMemcpy(d_norm, gammas, sizeof(gammas), cudaMemcpyHostToDevice), "scalar upload norm");
        check(cudaDeviceSynchronize(), "scalar norm sync");
        float mixed_by_mode[3][2] = {}, inject_by_mode[3] = {};
        for (int mode = 0; mode < 3; ++mode) {
            select_activation_mode(mode);
            gr_read(d_R, d_norm, d_weights, d_weights + 4, d_weights + 8, 0.0f,
                    sh, ws, d_mixed, d_inject, stream);
            check(cudaStreamSynchronize(stream), "scalar warmup");
            cudaGraph_t graph;
            cudaGraphExec_t executable;
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "scalar capture begin");
            gr_read(d_R, d_norm, d_weights, d_weights + 4, d_weights + 8, 0.0f,
                    sh, ws, d_mixed, d_inject, stream);
            check(cudaStreamEndCapture(stream, &graph), "scalar capture end");
            check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "scalar instantiate");
            // Changing both host options after capture cannot change the graph's selected kernels.
            select_activation_mode((mode + 1) % 3);
            // Warmup results must not make a missing/no-op captured launch pass.
            // Poison the outputs and intermediate workspace before replay.
            check(cudaMemsetAsync(d_mixed, 0xa5, 2 * sizeof(float), stream), "scalar poison mixed");
            check(cudaMemsetAsync(d_inject, 0xa5, sizeof(float), stream), "scalar poison inject");
            check(cudaMemsetAsync(d_scratch, 0xa5, ws.bytes, stream), "scalar poison workspace");
            check(cudaGraphLaunch(executable, stream), "scalar replay");
            check(cudaStreamSynchronize(stream), "scalar replay sync");
            check(cudaMemcpy(mixed_by_mode[mode], d_mixed, 2 * sizeof(float), cudaMemcpyDeviceToHost), "scalar mixed copy");
            check(cudaMemcpy(&inject_by_mode[mode], d_inject, sizeof(float), cudaMemcpyDeviceToHost), "scalar inject copy");
            const float activation = mode ? gamma : to_bf16(gamma);
            const float projection = activation * 0.5f;
            const float lo = projection / (1.0f + std::exp(-projection));
            const float gate = (mode ? lo : to_bf16(lo)) * 1.5f;
            const float expected_mixed = gamma / (1.0f + std::exp(-gate));
            const float expected_inject = activation * 2.0f;
            // Each mode has its own activation contract and bounded scalar-formula
            // check. Native fast exp/div need not match ordinary FP32 postops bit
            // for bit: native_gr_postops_parity establishes that arithmetic with
            // actual pinned CUDA graphs. The nonzero projection/injection here
            // has only one exactly representable product and remains exact.
            const bool ok = std::fabs(mixed_by_mode[mode][0] - expected_mixed) <= 2e-7f &&
                            mixed_by_mode[mode][0] == mixed_by_mode[mode][1] &&
                            inject_by_mode[mode] == expected_inject;
            std::printf("  scalar gamma %.8f %s activation/capture %s (mixed %.9f, ref %.9f)\n",
                        gamma, activation_mode_name(mode), ok ? "pass" : "FAIL",
                        mixed_by_mode[mode][0], expected_mixed);
            if (!ok) ++bad;
            check(cudaGraphExecDestroy(executable), "scalar graph exec destroy");
            check(cudaGraphDestroy(graph), "scalar graph destroy");

            const float sentinel = -73.25f;
            check(cudaMemcpy(d_inject, &sentinel, sizeof(float), cudaMemcpyHostToDevice), "scalar sentinel");
            check(cudaDeviceSynchronize(), "scalar sentinel sync");   // a late sentinel would hide a write to d_inject
            select_activation_mode(mode);
            gr_read(d_R, d_norm, d_weights, d_weights + 4, nullptr, 0.0f,
                    sh, ws, d_mixed, d_inject, stream);
            check(cudaStreamSynchronize(stream), "scalar final mixer sync");
            float final_mixed[2] = {}, untouched = 0.0f;
            check(cudaMemcpy(final_mixed, d_mixed, sizeof(final_mixed), cudaMemcpyDeviceToHost), "scalar final mixed");
            check(cudaMemcpy(&untouched, d_inject, sizeof(float), cudaMemcpyDeviceToHost), "scalar sentinel copy");
            const bool final_ok = std::memcmp(final_mixed, mixed_by_mode[mode], sizeof(final_mixed)) == 0 && untouched == sentinel;
            std::printf("  scalar final mixer %s %s\n", activation_mode_name(mode), final_ok ? "pass" : "FAIL");
            if (!final_ok) ++bad;
        }
        if (std::fabs(mixed_by_mode[1][0] - mixed_by_mode[0][0]) <= 1e-5f ||
            (gamma != 1.0f && inject_by_mode[0] == inject_by_mode[1])) {
            std::printf("  scalar precision fixture is not observable\n");
            ++bad;
        }
        if (inject_by_mode[1] != inject_by_mode[2]) {
            std::printf("  native MMVF changed the exact scalar FP32 injection projection\n");
            ++bad;
        }
    }
    select_activation_mode(0);
    check(cudaStreamDestroy(stream), "scalar stream destroy");
    cudaFree(d_R); cudaFree(d_norm); cudaFree(d_mixed); cudaFree(d_inject);
    cudaFree(d_weights); cudaFree(d_scratch);
    return bad;
}

int fused_multi_lds_parity(const float* d_norm, const uint16_t* d_down, const uint16_t* d_up,
                           const uint16_t* d_inject, float eps) {
    using namespace strata::kernels;
    constexpr int N = 2560, HC = 4, LR = 320, D = N * HC, T = kFusedGrMaxT;
    std::mt19937 rng(0x6f8a);
    std::normal_distribution<float> normal(0.0f, 0.3f);
    std::vector<float> r((size_t) T * D), bo((size_t) T * N), inj((size_t) T * HC);
    for (auto& x : r) x = normal(rng);
    for (auto& x : bo) x = normal(rng);
    for (auto& x : inj) x = normal(rng);

    float *d_r = nullptr, *d_r_out = nullptr, *d_bo = nullptr, *d_inj = nullptr;
    float *d_lo = nullptr, *d_rs = nullptr, *d_inj_out = nullptr, *d_mixed = nullptr, *d_xn = nullptr;
    check(cudaMalloc(&d_r, r.size() * sizeof(float)), "multi R");
    check(cudaMalloc(&d_r_out, r.size() * sizeof(float)), "multi R_out");
    check(cudaMalloc(&d_bo, bo.size() * sizeof(float)), "multi bo");
    check(cudaMalloc(&d_inj, inj.size() * sizeof(float)), "multi inj");
    check(cudaMalloc(&d_lo, (size_t) T * LR * sizeof(float)), "multi lo");
    check(cudaMalloc(&d_rs, (size_t) T * HC * sizeof(float)), "multi rs");
    check(cudaMalloc(&d_inj_out, (size_t) T * HC * sizeof(float)), "multi injection");
    check(cudaMalloc(&d_mixed, (size_t) T * N * sizeof(float)), "multi mixed");
    check(cudaMalloc(&d_xn, (size_t) T * D * sizeof(float)), "multi xn");
    check(cudaMemcpy(d_r, r.data(), r.size() * sizeof(float), cudaMemcpyHostToDevice), "multi copy R");
    check(cudaMemcpy(d_bo, bo.data(), bo.size() * sizeof(float), cudaMemcpyHostToDevice), "multi copy bo");
    check(cudaMemcpy(d_inj, inj.data(), inj.size() * sizeof(float), cudaMemcpyHostToDevice), "multi copy inj");

    std::vector<FusedGrArgs> args(T);
    for (int t = 0; t < T; ++t) {
        auto& a = args[t];
        a.R = d_r + (size_t) t * D;
        a.R_out = d_r_out + (size_t) t * D;
        a.apply = true;
        a.bo_prev = d_bo + (size_t) t * N;
        a.inj_prev = d_inj + (size_t) t * HC;
        a.w_norm = d_norm;
        a.w_down = d_down;
        a.w_up = d_up;
        a.w_inject = d_inject;
        a.eps = eps;
        a.lo = d_lo + (size_t) t * LR;
        a.rs = d_rs + (size_t) t * HC;
        a.inject_out = d_inj_out + (size_t) t * HC;
        a.mixed = d_mixed + (size_t) t * N;
    }

    struct Snapshot {
        std::vector<float> r_out, lo, rs, inject, mixed;
    };
    auto snapshot = [&]() {
        Snapshot s;
        s.r_out.resize((size_t) T * D);
        s.lo.resize((size_t) T * LR);
        s.rs.resize((size_t) T * HC);
        s.inject.resize((size_t) T * HC);
        s.mixed.resize((size_t) T * N);
        check(cudaMemcpy(s.r_out.data(), d_r_out, s.r_out.size() * sizeof(float), cudaMemcpyDeviceToHost), "multi read R_out");
        check(cudaMemcpy(s.lo.data(), d_lo, s.lo.size() * sizeof(float), cudaMemcpyDeviceToHost), "multi read lo");
        check(cudaMemcpy(s.rs.data(), d_rs, s.rs.size() * sizeof(float), cudaMemcpyDeviceToHost), "multi read rs");
        check(cudaMemcpy(s.inject.data(), d_inj_out, s.inject.size() * sizeof(float), cudaMemcpyDeviceToHost), "multi read inject");
        check(cudaMemcpy(s.mixed.data(), d_mixed, s.mixed.size() * sizeof(float), cudaMemcpyDeviceToHost), "multi read mixed");
        return s;
    };
    // the split read (STRATA_GR_V3=1) sums in another order than the single-token kernel: equal within float
    // rounding, not to the bit, so it is compared with a relative tolerance; the default kernels bit for bit
    static const bool v3 = [] { const char* v = std::getenv("STRATA_GR_V3"); return v != nullptr && std::atoi(v) != 0; }();
    auto close = [](const std::vector<float>& x, const std::vector<float>& y) {
        double worst = 0.0, mag = 1e-30;
        for (size_t i = 0; i < x.size(); ++i) {
            worst = std::max(worst, (double) std::fabs(x[i] - y[i]));
            mag = std::max(mag, (double) std::fabs(y[i]));
        }
        if (worst > 2e-6 * mag)
            std::printf("    tolerance: worst %.3e of max |ref| %.3e (rel %.3e), n %zu\n", worst, mag, worst / mag, x.size());
        return worst <= 2e-6 * mag;
    };
    auto same = [&](const Snapshot& a, const Snapshot& b) {
        if (v3)   // `lo` is the default kernels' workspace between down and up; the split read keeps it in shared memory
            return close(a.r_out, b.r_out) && close(a.rs, b.rs) && close(a.inject, b.inject) && close(a.mixed, b.mixed);
        return std::memcmp(a.r_out.data(), b.r_out.data(), a.r_out.size() * sizeof(float)) == 0 &&
               std::memcmp(a.lo.data(), b.lo.data(), a.lo.size() * sizeof(float)) == 0 &&
               std::memcmp(a.rs.data(), b.rs.data(), a.rs.size() * sizeof(float)) == 0 &&
               std::memcmp(a.inject.data(), b.inject.data(), a.inject.size() * sizeof(float)) == 0 &&
               std::memcmp(a.mixed.data(), b.mixed.data(), a.mixed.size() * sizeof(float)) == 0;
    };

    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "multi stream");
    // the uploads above (and the caller's weights) run on the legacy stream, which the non-blocking `stream` does not wait for
    check(cudaDeviceSynchronize(), "multi setup");
    // Max T forces the HIP kernel's full dynamic-LDS request: 8 * 1280 * sizeof(float) = 40 KiB.
    fused_gr_read_multi(args.data(), T, d_xn, stream);
    check(cudaStreamSynchronize(stream), "multi max-T sync");
    const Snapshot multi = snapshot();
    for (int t = 0; t < T; ++t) fused_gr_read(args[t], stream);
    check(cudaStreamSynchronize(stream), "single reference sync");
    const Snapshot single = snapshot();
    int bad = 0;
    if (!same(multi, single)) {
        std::printf("  fused GR multi max-T differs from single-token calls\n");
        ++bad;
    }

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "multi graph begin");
    fused_gr_read_multi(args.data(), T, d_xn, stream);
    check(cudaStreamEndCapture(stream, &graph), "multi graph end");
    check(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0), "multi graph instantiate");
    check(cudaGraphLaunch(graph_exec, stream), "multi graph initial replay");
    check(cudaStreamSynchronize(stream), "multi graph initial sync");
    const Snapshot captured = snapshot();
    if (!same(multi, captured)) {
        std::printf("  fused GR multi captured graph differs from direct max-T call\n");
        ++bad;
    }

    // Reuse the same captured pointers with new payloads; then independently run the single-token path again.
    for (size_t i = 0; i < r.size(); ++i) r[i] = -0.7f * r[i] + 0.001f * (float) (i % 17);
    for (size_t i = 0; i < bo.size(); ++i) bo[i] = -0.4f * bo[i] + 0.02f;
    for (size_t i = 0; i < inj.size(); ++i) inj[i] += 0.3f;
    check(cudaMemcpyAsync(d_r, r.data(), r.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "multi replay R");
    check(cudaMemcpyAsync(d_bo, bo.data(), bo.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "multi replay bo");
    check(cudaMemcpyAsync(d_inj, inj.data(), inj.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "multi replay inj");
    check(cudaGraphLaunch(graph_exec, stream), "multi graph changed replay");
    check(cudaStreamSynchronize(stream), "multi graph changed sync");
    const Snapshot replay = snapshot();
    if (std::memcmp(multi.mixed.data(), replay.mixed.data(), multi.mixed.size() * sizeof(float)) == 0) {
        std::printf("  fused GR graph replay ignored changed inputs\n");
        ++bad;
    }
    for (int t = 0; t < T; ++t) fused_gr_read(args[t], stream);
    check(cudaStreamSynchronize(stream), "changed single reference sync");
    if (!same(replay, snapshot())) {
        std::printf("  fused GR changed graph replay differs from single-token calls\n");
        ++bad;
    }


    // S26 STRATA_QFUSE: the read's own q8_1 image of `mixed` must be the bytes native_quantize_q8_1 writes from it -
    // for every T (1..8), directly and through a captured graph replayed twice (the group counters must reset)
    {
        uint8_t *d_q = nullptr, *d_ref = nullptr;
        unsigned* d_cnt = nullptr;
        const size_t qbytes = (size_t) T * (N / 32) * 36;
        check(cudaMalloc(&d_q, qbytes), "qfuse q8");
        check(cudaMalloc(&d_ref, qbytes), "qfuse ref");
        check(cudaMalloc(&d_cnt, (N / 32) * sizeof(unsigned)), "qfuse counters");
        check(cudaMemset(d_cnt, 0, (N / 32) * sizeof(unsigned)), "qfuse counters zero");
        int qbad = 0;
        std::vector<uint8_t> hq(qbytes), hr(qbytes);
        for (int tt = 1; tt <= T && !qbad; ++tt) {
            std::vector<FusedGrArgs> qa(args.begin(), args.begin() + tt);
            for (int t = 0; t < tt; ++t) { qa[t].q8_mixed = d_q + (size_t) t * (N / 32) * 36; qa[t].q8_cnt = d_cnt; }
            for (int rep = 0; rep < 3 && !qbad; ++rep) {
                check(cudaMemsetAsync(d_q, 0x5a, qbytes, stream), "qfuse poison");
                bool wrote = false;
                cudaGraph_t qg = nullptr;
                cudaGraphExec_t qx = nullptr;
                if (rep == 0) {
                    wrote = fused_gr_read_multi(qa.data(), tt, d_xn, stream);
                } else {   // a captured read, replayed (twice: rep 1 and 2 use fresh captures, each replayed twice)
                    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "qfuse begin");
                    wrote = fused_gr_read_multi(qa.data(), tt, d_xn, stream);
                    check(cudaStreamEndCapture(stream, &qg), "qfuse end");
                    check(cudaGraphInstantiate(&qx, qg, nullptr, nullptr, 0), "qfuse instantiate");
                    check(cudaGraphLaunch(qx, stream), "qfuse replay 1");
                    check(cudaMemsetAsync(d_q, 0x5a, qbytes, stream), "qfuse poison 2");
                    check(cudaGraphLaunch(qx, stream), "qfuse replay 2");
                }
                strata::kernels::native_quantize_q8_1(d_mixed, d_ref, N, tt, stream);
                check(cudaStreamSynchronize(stream), "qfuse sync");
                if (qx) { cudaGraphExecDestroy(qx); cudaGraphDestroy(qg); }
                if (v3) { std::printf("  QFUSE: the v3 read writes no q8_1 (%s)\n", wrote ? "WRONG: it says it did" : "ok"); qbad += wrote; break; }
                check(cudaMemcpy(hq.data(), d_q, (size_t) tt * (N / 32) * 36, cudaMemcpyDeviceToHost), "qfuse read");
                check(cudaMemcpy(hr.data(), d_ref, (size_t) tt * (N / 32) * 36, cudaMemcpyDeviceToHost), "qfuse ref read");
                if (!wrote || std::memcmp(hq.data(), hr.data(), (size_t) tt * (N / 32) * 36) != 0) {
                    std::printf("  QFUSE: T=%d rep %d: %s\n", tt, rep, wrote ? "q8_1 bytes differ" : "not written");
                    ++qbad;
                }
            }
        }
        std::printf("  fused GR read + q8_1 (STRATA_QFUSE), T 1..8, direct and graph replays: %s\n", qbad ? "FAIL" : "pass");
        bad += qbad;
        cudaFree(d_q); cudaFree(d_ref); cudaFree(d_cnt);
    }

    // #3948: STRATA_QFUSE in a verify window of TWO GROUPS (verify.cpp gr_read_group with G = 2): the window's tokens are
    // read in two calls one after the other, the same group counters, each call with its own tokens' destinations - the
    // attention half's xq_ is indexed from the group's first token, the expert half's nat_xq_ by the window's token.
    // Fresh random inputs every repetition, 200 repetitions for each T = 2..8 (a race that fires one launch in a hundred
    // shows), direct and as a graph replay. After every repetition: both groups' q8_1 bytes equal what
    // native_quantize_q8_1 makes of `mixed`, nothing outside the written ranges changed, and the counters are back at zero.
    if (!v3) {
        // (STRATA_GRP_REPS=N more repetitions, STRATA_GRP_KEEP_GOING=1 counts every failure instead of stopping at the first: to measure a rare one)
        const int REPS = std::getenv("STRATA_GRP_REPS") ? std::max(1, std::atoi(std::getenv("STRATA_GRP_REPS"))) : 200;
        const bool keep_going = std::getenv("STRATA_GRP_KEEP_GOING") != nullptr;
        const size_t B = (size_t) (N / 32) * 36;   // one token's q8_1 image
        uint8_t *d_qa = nullptr, *d_qb = nullptr, *d_qabs = nullptr, *d_ref2 = nullptr;
        unsigned* d_cnt2 = nullptr;
        check(cudaMalloc(&d_qa, (size_t) T * B), "qfuse2 qa");
        check(cudaMalloc(&d_qb, (size_t) T * B), "qfuse2 qb");
        check(cudaMalloc(&d_qabs, (size_t) T * B), "qfuse2 qabs");
        check(cudaMalloc(&d_ref2, (size_t) T * B), "qfuse2 ref");
        check(cudaMalloc(&d_cnt2, (N / 32) * sizeof(unsigned)), "qfuse2 counters");
        check(cudaMemset(d_cnt2, 0, (N / 32) * sizeof(unsigned)), "qfuse2 counters zero");
        std::mt19937 prng(0x3948u);
        std::normal_distribution<float> pn(0.0f, 0.3f);
        std::vector<uint8_t> g_all((size_t) T * B), g_ref((size_t) T * B), g_a((size_t) T * B), g_b((size_t) T * B);
        std::vector<unsigned> h_cnt(N / 32);
        int g2bad = 0;
        long runs = 0;
        for (int tt = 2; tt <= T && (!g2bad || keep_going); ++tt) {
            const int h = (tt + 1) / 2;   // verify.cpp: tb_/te_ = {0, (T + 1) / 2}, {(T + 1) / 2, T}
            std::vector<FusedGrArgs> rel1(args.begin(), args.begin() + h), rel2(args.begin() + h, args.begin() + tt);
            std::vector<FusedGrArgs> abs1 = rel1, abs2 = rel2;
            for (int t = 0; t < h; ++t) {
                rel1[t].q8_mixed = d_qa + (size_t) t * B;     rel1[t].q8_cnt = d_cnt2;
                abs1[t].q8_mixed = d_qabs + (size_t) t * B;   abs1[t].q8_cnt = d_cnt2;
            }
            for (int t = h; t < tt; ++t) {
                rel2[t - h].q8_mixed = d_qb + (size_t) (t - h) * B;   rel2[t - h].q8_cnt = d_cnt2;
                abs2[t - h].q8_mixed = d_qabs + (size_t) t * B;       abs2[t - h].q8_cnt = d_cnt2;
            }
            cudaGraph_t gg = nullptr;
            cudaGraphExec_t gx = nullptr;
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "qfuse2 begin");
            const bool w1 = fused_gr_read_multi(rel1.data(), h, d_xn, stream);
            const bool w2 = fused_gr_read_multi(rel2.data(), tt - h, d_xn + (size_t) h * D, stream);
            check(cudaStreamEndCapture(stream, &gg), "qfuse2 end");
            check(cudaGraphInstantiate(&gx, gg, nullptr, nullptr, 0), "qfuse2 instantiate");
            if (!w1 || !w2) { std::printf("  QFUSE two groups: T=%d: a group did not write its q8_1\n", tt); ++g2bad; }
            for (int rep = 0; rep < REPS && (!g2bad || keep_going); ++rep) {
                for (auto& x : r) x = pn(prng);
                for (auto& x : bo) x = pn(prng);
                for (auto& x : inj) x = pn(prng);
                check(cudaMemcpy(d_r, r.data(), r.size() * sizeof(float), cudaMemcpyHostToDevice), "qfuse2 R");
                check(cudaMemcpy(d_bo, bo.data(), bo.size() * sizeof(float), cudaMemcpyHostToDevice), "qfuse2 bo");
                check(cudaMemcpy(d_inj, inj.data(), inj.size() * sizeof(float), cudaMemcpyHostToDevice), "qfuse2 inj");
                check(cudaMemsetAsync(d_qa, 0x5a, (size_t) T * B, stream), "qfuse2 poison a");
                check(cudaMemsetAsync(d_qb, 0x5a, (size_t) T * B, stream), "qfuse2 poison b");
                check(cudaMemsetAsync(d_qabs, 0x5a, (size_t) T * B, stream), "qfuse2 poison abs");
                const int mode = rep % 3;   // 0: group-relative, direct; 1: window-indexed, direct; 2: group-relative, graph replay
                if (mode == 0) {
                    fused_gr_read_multi(rel1.data(), h, d_xn, stream);
                    fused_gr_read_multi(rel2.data(), tt - h, d_xn + (size_t) h * D, stream);
                } else if (mode == 1) {
                    fused_gr_read_multi(abs1.data(), h, d_xn, stream);
                    fused_gr_read_multi(abs2.data(), tt - h, d_xn + (size_t) h * D, stream);
                } else {
                    check(cudaGraphLaunch(gx, stream), "qfuse2 replay");
                }
                strata::kernels::native_quantize_q8_1(d_mixed, d_ref2, N, tt, stream);
                check(cudaStreamSynchronize(stream), "qfuse2 sync");
                ++runs;
                check(cudaMemcpy(g_ref.data(), d_ref2, (size_t) tt * B, cudaMemcpyDeviceToHost), "qfuse2 read ref");
                check(cudaMemcpy(h_cnt.data(), d_cnt2, h_cnt.size() * sizeof(unsigned), cudaMemcpyDeviceToHost), "qfuse2 read counters");
                for (size_t i = 0; i < h_cnt.size(); ++i)
                    if (h_cnt[i] != 0) {
                        std::printf("  QFUSE two groups: T=%d rep %d: counter %zu left at %u\n", tt, rep, i, h_cnt[i]);
                        ++g2bad;
                        break;
                    }
                if (mode == 1) {
                    check(cudaMemcpy(g_all.data(), d_qabs, (size_t) T * B, cudaMemcpyDeviceToHost), "qfuse2 read abs");
                    if (std::memcmp(g_all.data(), g_ref.data(), (size_t) tt * B) != 0) {
                        for (int t = 0; t < tt; ++t)
                            if (std::memcmp(g_all.data() + (size_t) t * B, g_ref.data() + (size_t) t * B, B) != 0) {
                                std::printf("  QFUSE two groups (window-indexed): T=%d rep %d: token %d (group %d) differs\n", tt, rep, t, t < h ? 0 : 1);
                                int nd = 0;   // where: block = 32 columns, bytes 0-3 the (scale, sum) pair, 4-35 the codes
                                for (size_t o = 0; o < B && nd < 6; ++o)
                                    if (g_all[(size_t) t * B + o] != g_ref[(size_t) t * B + o]) {
                                        std::printf("    block %zu byte %zu (%s): got 0x%02x, separate quantise 0x%02x\n", o / 36, o % 36,
                                                    o % 36 < 4 ? "scale/sum" : "code", g_all[(size_t) t * B + o], g_ref[(size_t) t * B + o]);
                                        ++nd;
                                    }
                                break;
                            }
                        ++g2bad;
                    }
                    for (size_t i = (size_t) tt * B; i < (size_t) T * B; ++i)
                        if (g_all[i] != 0x5a) { std::printf("  QFUSE two groups: T=%d rep %d: wrote past the window at byte %zu\n", tt, rep, i); ++g2bad; break; }
                } else {
                    check(cudaMemcpy(g_a.data(), d_qa, (size_t) T * B, cudaMemcpyDeviceToHost), "qfuse2 read a");
                    check(cudaMemcpy(g_b.data(), d_qb, (size_t) T * B, cudaMemcpyDeviceToHost), "qfuse2 read b");
                    if (std::memcmp(g_a.data(), g_ref.data(), (size_t) h * B) != 0) {
                        std::printf("  QFUSE two groups (group-relative, %s): T=%d rep %d: group 0 differs\n", mode == 2 ? "graph" : "direct", tt, rep);
                        for (size_t o = 0, nd = 0; o < (size_t) h * B && nd < 6; ++o)
                            if (g_a[o] != g_ref[o]) {
                                std::printf("    token %zu block %zu byte %zu (%s): got 0x%02x, separate quantise 0x%02x\n", o / B, (o % B) / 36,
                                            o % 36, o % 36 < 4 ? "scale/sum" : "code", g_a[o], g_ref[o]);
                                ++nd;
                            }
                        ++g2bad;
                    }
                    if (std::memcmp(g_b.data(), g_ref.data() + (size_t) h * B, (size_t) (tt - h) * B) != 0) {
                        std::printf("  QFUSE two groups (group-relative, %s): T=%d rep %d: group 1 differs\n", mode == 2 ? "graph" : "direct", tt, rep);
                        ++g2bad;
                    }
                    for (size_t i = (size_t) h * B; i < (size_t) T * B; ++i)
                        if (g_a[i] != 0x5a) { std::printf("  QFUSE two groups: T=%d rep %d: group 0 wrote past its tokens at byte %zu\n", tt, rep, i); ++g2bad; break; }
                    for (size_t i = (size_t) (tt - h) * B; i < (size_t) T * B; ++i)
                        if (g_b[i] != 0x5a) { std::printf("  QFUSE two groups: T=%d rep %d: group 1 wrote past its tokens at byte %zu\n", tt, rep, i); ++g2bad; break; }
                }
            }
            cudaGraphExecDestroy(gx);
            cudaGraphDestroy(gg);
        }
        // the inputs the rest of the test (and its caller) expect again
        check(cudaDeviceSynchronize(), "qfuse2 done");
        std::printf("  fused GR read + q8_1 (STRATA_QFUSE), a window of two groups, T 2..8, %ld runs (%d random inputs each T), "
                    "group-relative and window-indexed, direct and graph: %s\n", runs, REPS, g2bad ? "FAIL" : "pass");
        bad += g2bad;
        cudaFree(d_qa); cudaFree(d_qb); cudaFree(d_qabs); cudaFree(d_ref2); cudaFree(d_cnt2);
    }
    std::printf("  fused GR multi max-T=8 LDS launch and changing graph replay %s\n",
                bad == 0 ? "pass" : "FAIL");
    check(cudaGraphExecDestroy(graph_exec), "multi graph exec destroy");
    check(cudaGraphDestroy(graph), "multi graph destroy");
    check(cudaStreamDestroy(stream), "multi stream destroy");
    check(cudaFree(d_xn), "free multi xn");
    check(cudaFree(d_mixed), "free multi mixed");
    check(cudaFree(d_inj_out), "free multi injection");
    check(cudaFree(d_rs), "free multi rs");
    check(cudaFree(d_lo), "free multi lo");
    check(cudaFree(d_inj), "free multi inj");
    check(cudaFree(d_bo), "free multi bo");
    check(cudaFree(d_r_out), "free multi R_out");
    check(cudaFree(d_r), "free multi R");
    return bad;
}

}  // namespace

int main(int argc, char** argv) {
    // this test compares the default kernels bit for bit; the split read (a default on a 12.0 card since 0.2.7) is STRATA_GR_V3=1 below
#if defined(_WIN32)
    if (std::getenv("STRATA_GR_V3") == nullptr) _putenv_s("STRATA_GR_V3", "0");
#else
    if (std::getenv("STRATA_GR_V3") == nullptr) setenv("STRATA_GR_V3", "0", 1);
#endif
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: gr_parity [--selftest]\n"); return 2; }
    }

    const long long n_embd = 256, hc = 4, hc_lr = 32;
    const long long hc_dim = hc * n_embd;
    const float eps = 1e-6f;
    std::mt19937 rng(1234);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    // Weights are BF16-VALUED f32, which is what the pack holds for a BF16 source type.  Generating plain f32
    // weights would make the kernel (which stores 16 bits) differ from the reference by the rounding itself,
    // and the test would then be measuring the fixture.
    auto bf16_weight = [&](size_t n, float sigma) {
        std::vector<float> v(n);
        for (auto& x : v) x = to_bf16(gauss(rng) * sigma);
        return v;
    };
    std::vector<float> R((size_t) hc_dim);
    for (auto& x : R) x = gauss(rng);
    // STREAM SCALES 1, 4, 16, 64.  Without them trap 1 is only ~5% observable, because four streams of i.i.d.
    // normal noise have nearly the same RMS by accident - so a whole-stack RMS and a per-stream RMS agree to
    // within the noise, and the fixture would be asserting a distinction it cannot see.  Giving the streams
    // different magnitudes is what makes "per-stream" a claim with content.
    for (long long c = 0; c < hc; ++c)
        for (long long d = 0; d < n_embd; ++d) R[(size_t) (c * n_embd + d)] *= std::pow(4.0f, (float) c);
    // gamma stored as (1 + w), so values near 1 - a reader expecting a plain scale is the one this catches
    std::vector<float> w_norm((size_t) hc_dim);
    for (auto& x : w_norm) x = 1.0f + 0.1f * gauss(rng);
    std::vector<float> w_down = bf16_weight((size_t) (hc_lr * hc_dim), 0.2f);
    std::vector<float> w_up = bf16_weight((size_t) (hc_dim * hc_lr), 0.2f);
    std::vector<float> w_inject = bf16_weight((size_t) (hc * hc_dim), 0.05f);

    // NO PERMUTATION: the kernel takes both matrices exactly as the manifest (and `ref/gr.py`) store them.
    // `w_down` is (hc_lr, hc_dim) row-major and `w_up` is (hc_dim, hc_lr) row-major, so the bit-copy below is
    // the whole conversion.  An earlier version of this kernel wanted both TRANSPOSED - the test built them
    // that way and asserted the orientation observable - and that is gone because the warp-per-row mapping
    // makes it unnecessary.  What replaces it is the general orientation trap further down: a weight array
    // read with the wrong index expression must still produce a visibly different answer.
    std::vector<uint16_t> q_down(w_down.size()), q_up(w_up.size()), q_inject(w_inject.size());
    for (size_t i = 0; i < w_down.size(); ++i) q_down[i] = bf16_bits(w_down[i]);
    for (size_t i = 0; i < w_up.size(); ++i) q_up[i] = bf16_bits(w_up[i]);
    for (size_t i = 0; i < w_inject.size(); ++i) q_inject[i] = bf16_bits(w_inject[i]);

    // ---- device side
    float *d_R = nullptr, *d_norm = nullptr, *d_mixed = nullptr, *d_inject = nullptr;
    uint16_t *d_down = nullptr, *d_up = nullptr, *d_inj = nullptr;
    check(cudaMalloc(&d_R, R.size() * sizeof(float)), "m R");
    check(cudaMalloc(&d_norm, w_norm.size() * sizeof(float)), "m norm");
    check(cudaMalloc(&d_down, q_down.size() * sizeof(uint16_t)), "m down");
    check(cudaMalloc(&d_up, q_up.size() * sizeof(uint16_t)), "m up");
    check(cudaMalloc(&d_inj, q_inject.size() * sizeof(uint16_t)), "m inj");
    check(cudaMalloc(&d_mixed, (size_t) n_embd * sizeof(float)), "m mixed");
    check(cudaMalloc(&d_inject, (size_t) hc * sizeof(float)), "m inject");
    check(cudaMemcpy(d_R, R.data(), R.size() * sizeof(float), cudaMemcpyHostToDevice), "c R");
    check(cudaMemcpy(d_norm, w_norm.data(), w_norm.size() * sizeof(float), cudaMemcpyHostToDevice), "c norm");
    check(cudaMemcpy(d_down, q_down.data(), q_down.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "c down");
    check(cudaMemcpy(d_up, q_up.data(), q_up.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "c up");
    check(cudaMemcpy(d_inj, q_inject.data(), q_inject.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "c inj");

    const strata::kernels::GrShapes sh{n_embd, hc, hc_lr};
    // DEVICE memory: the workspace is written by the kernel.
    void* d_ws_raw = nullptr;
    check(cudaMalloc(&d_ws_raw, strata::kernels::gr_workspace_bytes(sh)), "m ws");
    strata::kernels::GrWorkspace ws;
    strata::kernels::gr_workspace_init(sh, d_ws_raw, ws);
    strata::kernels::gr_read(d_R, d_norm, d_down, d_up, d_inj, eps, sh, ws, d_mixed, d_inject, nullptr);
    std::vector<float> got_mixed((size_t) n_embd), got_inject((size_t) hc);
    check(cudaMemcpy(got_mixed.data(), d_mixed, got_mixed.size() * sizeof(float), cudaMemcpyDeviceToHost), "c mixed");
    check(cudaMemcpy(got_inject.data(), d_inject, got_inject.size() * sizeof(float), cudaMemcpyDeviceToHost), "c inject");

    int bad = 0;

    // ---- the traps, asserted OBSERVABLE before the kernel is judged against the right reading
    Opts right;
    std::vector<float> want_mixed, want_inject;
    reference(R, w_norm, w_down, w_up, w_inject, eps, n_embd, hc, hc_lr, right, want_mixed, want_inject);

    struct Trap { const char* name; Opts o; double floor; };
    const Trap traps[] = {
        {"per-stream RMSNorm vs whole-stack", {false, true, true, true, true}, 0.05},
        {"/hc inside the silu vs outside", {true, false, true, true, true}, 0.05},
        {"SiLU on lo / sigmoid on gate, swapped", {true, true, false, true, true}, 0.05},
        {"mean over streams vs sum", {true, true, true, false, true}, 0.50},
        {"BF16 vs FP32 activation contracts", {true, true, true, true, false}, 1e-4},
    };
    for (const Trap& t : traps) {
        std::vector<float> m, i;
        reference(R, w_norm, w_down, w_up, w_inject, eps, n_embd, hc, hc_lr, t.o, m, i);
        const double rel = rel_diff(want_mixed, m);
        const bool observable = rel > t.floor;
        std::printf("  %-40s %-4s (%.4f%% apart, floor %.3f%%)\n", t.name, observable ? "yes" : "*** NO ***",
                    rel * 100, t.floor * 100);
        if (!observable) {
            std::printf("      *** the fixture cannot see this trap, so passing proves nothing about it ***\n");
            ++bad;
        }
    }

    // ---- the kernel against the correct reference
    double mag = 0;
    const double rel_mixed = rel_diff(want_mixed, got_mixed, &mag);
    std::printf("\n  %-40s worst %.3e (mean |ref| %.4f)\n", "mixed vs reference", rel_mixed, mag);

    // The tolerance is set by SUMMATION ORDER, not by the structure: both sides round the activation to bf16
    // and both weights are bf16-valued, so every product is exact in f32 and only the order differs.  A
    // structural error is several orders of magnitude larger - the trap floors above say so.
    if (!(rel_mixed <= 1e-4)) {
        std::printf("    *** over 1e-4, which summation order does not explain - look for a structural bug ***\n");
        ++bad;
    }

    const double rel_inject = rel_diff(want_inject, got_inject);
    std::printf("  %-40s rel %.3e\n", "inject vs reference", rel_inject);
    if (!(rel_inject <= 1e-4)) {
        // Say WHICH elements and by how much: a whole-array L1 hides whether this is one stream wrong (an
        // indexing bug) or all of them slightly wrong (a precision bug), and those need different responses.
        std::printf("    want:");
        for (long long c = 0; c < hc; ++c) std::printf(" %12.6f", (double) want_inject[(size_t) c]);
        std::printf("\n    got :");
        for (long long c = 0; c < hc; ++c) std::printf(" %12.6f", (double) got_inject[(size_t) c]);
        std::printf("\n");
        ++bad;
    }

    // Same geometry and weights with the CUDA single-token activation contract.  The scalar fixture
    // below is the independent precision check; this case exercises multi-stream reductions and indexing.
    {
        Opts fp32;
        fp32.round_activation = false;
        std::vector<float> wm, wi, gm((size_t) n_embd), gi((size_t) hc);
        reference(R, w_norm, w_down, w_up, w_inject, eps, n_embd, hc, hc_lr, fp32, wm, wi);
        strata::kernels::gr_set_fp32_activations(true);
        strata::kernels::gr_read(d_R, d_norm, d_down, d_up, d_inj, eps, sh, ws, d_mixed, d_inject, nullptr);
        check(cudaMemcpy(gm.data(), d_mixed, gm.size() * sizeof(float), cudaMemcpyDeviceToHost), "FP32 mixed");
        check(cudaMemcpy(gi.data(), d_inject, gi.size() * sizeof(float), cudaMemcpyDeviceToHost), "FP32 inject");
        strata::kernels::gr_set_fp32_activations(false);
        const double rm = rel_diff(wm, gm), ri = rel_diff(wi, gi);
        const bool ok = rm <= 1e-4 && ri <= 1e-4;
        std::printf("  FP32 activations vs reference: %s (mixed %.3e, inject %.3e)\n",
                    ok ? "pass" : "FAIL", rm, ri);
        if (!ok) ++bad;
        strata::kernels::gr_read(d_R, d_norm, d_down, d_up, d_inj, eps, sh, ws, d_mixed, d_inject, nullptr);
        check(cudaMemcpy(gm.data(), d_mixed, gm.size() * sizeof(float), cudaMemcpyDeviceToHost), "restored mixed");
        check(cudaMemcpy(gi.data(), d_inject, gi.size() * sizeof(float), cudaMemcpyDeviceToHost), "restored inject");
        const bool restored = std::memcmp(gm.data(), got_mixed.data(), gm.size() * sizeof(float)) == 0 &&
                              std::memcmp(gi.data(), got_inject.data(), gi.size() * sizeof(float)) == 0;
        std::printf("  restoring default activation contract: %s\n", restored ? "byte-identical" : "FAIL");
        if (!restored) ++bad;
    }

    // ---- THE WEIGHT ORIENTATION MUST BE OBSERVABLE.  The kernel now takes both matrices exactly as the
    // manifest stores them, so there is no permutation for a loader to forget - but the INDEX EXPRESSIONS
    // inside the kernel are still a place to be wrong, and a wrong one gives right-shaped, wrong-valued
    // output.  Feed transposed copies through the same code path and require the answer to differ.
    {
        uint16_t *d_down_bad = nullptr, *d_up_bad = nullptr;
        check(cudaMalloc(&d_down_bad, q_down.size() * sizeof(uint16_t)), "m down_bad");
        check(cudaMalloc(&d_up_bad, q_up.size() * sizeof(uint16_t)), "m up_bad");
        std::vector<uint16_t> tr_down(q_down.size()), tr_up(q_up.size());
        for (long long k = 0; k < hc_lr; ++k)
            for (long long i = 0; i < hc_dim; ++i)
                tr_down[(size_t) (i * hc_lr + k)] = q_down[(size_t) (k * hc_dim + i)];
        for (long long i = 0; i < hc_dim; ++i)
            for (long long k = 0; k < hc_lr; ++k)
                tr_up[(size_t) (k * hc_dim + i)] = q_up[(size_t) (i * hc_lr + k)];
        check(cudaMemcpy(d_down_bad, tr_down.data(), tr_down.size() * 2, cudaMemcpyHostToDevice), "cb d");
        check(cudaMemcpy(d_up_bad, tr_up.data(), tr_up.size() * 2, cudaMemcpyHostToDevice), "cb u");
        std::vector<float> bad_mixed((size_t) n_embd);
        float* d_bad = nullptr;
        check(cudaMalloc(&d_bad, bad_mixed.size() * sizeof(float)), "m bad");
        strata::kernels::gr_read(d_R, d_norm, d_down_bad, d_up_bad, d_inj, eps, sh, ws, d_bad, d_inject,
                                 nullptr);
        check(cudaMemcpy(bad_mixed.data(), d_bad, bad_mixed.size() * sizeof(float), cudaMemcpyDeviceToHost), "cb b");
        const double rel = rel_diff(want_mixed, bad_mixed);
        const bool visible = rel > 0.05;
        std::printf("  %-40s %s (%.1f%% apart)\n", "weight orientation is observable",
                    visible ? "yes" : "*** NO ***", rel * 100);
        if (!visible) ++bad;
        cudaFree(d_down_bad); cudaFree(d_up_bad); cudaFree(d_bad);
    }

    // ---- the FINAL mixer passes w_inject = nullptr, and then nothing may be written
    float sentinel = -12345.0f;
    check(cudaMemcpy(d_inject, &sentinel, sizeof(float), cudaMemcpyHostToDevice), "c sentinel");
    strata::kernels::gr_read(d_R, d_norm, d_down, d_up, nullptr, eps, sh, ws, d_mixed, d_inject, nullptr);
    float after = 0.0f;
    check(cudaMemcpy(&after, d_inject, sizeof(float), cudaMemcpyDeviceToHost), "c after");
    const bool untouched = (after == sentinel);
    std::printf("  %-40s %s\n", "null w_inject writes nothing", untouched ? "yes" : "*** NO ***");
    if (!untouched) ++bad;

    // ---- gr_write: the `2*sigmoid` centring, asserted as the PROPERTY the source comment claims
    std::vector<float> block_out((size_t) n_embd);
    for (auto& x : block_out) x = gauss(rng);
    std::vector<float> zero_inj((size_t) hc, 0.0f);
    float *d_Rw = nullptr, *d_bo = nullptr, *d_zi = nullptr, *d_outw = nullptr;
    check(cudaMalloc(&d_Rw, R.size() * sizeof(float)), "m Rw");
    check(cudaMalloc(&d_bo, block_out.size() * sizeof(float)), "m bo");
    check(cudaMalloc(&d_zi, zero_inj.size() * sizeof(float)), "m zi");
    check(cudaMalloc(&d_outw, R.size() * sizeof(float)), "m outw");
    check(cudaMemcpy(d_Rw, R.data(), R.size() * sizeof(float), cudaMemcpyHostToDevice), "c Rw");
    check(cudaMemcpy(d_bo, block_out.data(), block_out.size() * sizeof(float), cudaMemcpyHostToDevice), "c bo");
    check(cudaMemcpy(d_zi, zero_inj.data(), zero_inj.size() * sizeof(float), cudaMemcpyHostToDevice), "c zi");

    strata::kernels::gr_write(d_Rw, d_bo, d_zi, sh, d_outw, nullptr);
    std::vector<float> got_write((size_t) hc_dim);
    check(cudaMemcpy(got_write.data(), d_outw, got_write.size() * sizeof(float), cudaMemcpyDeviceToHost), "c outw");

    double worst_plain = 0;
    for (long long c = 0; c < hc; ++c)
        for (long long d = 0; d < n_embd; ++d) {
            const size_t i = (size_t) (c * n_embd + d);
            const double want = (double) R[i] + (double) block_out[(size_t) d];
            worst_plain = std::fmax(worst_plain, rel_terms(want, (double) got_write[i], (double) R[i],
                                                           (double) block_out[(size_t) d]));
        }
    const bool plain = worst_plain < 1e-6;
    std::printf("  %-40s %s (worst rel %.3e)\n", "zero injection -> plain residual add",
                plain ? "yes" : "*** NO ***", worst_plain);
    if (!plain) ++bad;

    // the rival reading: sigmoid without the 2 halves the update, and must be visible.
    //
    // The comparison is on the DELTA `out - R`, not on `out`.  The claim is about the gate WEIGHT, and `R`
    // now reaches 64 in the last stream, so measuring `R + block_out*w` against the wrong reading dilutes a
    // 100% error in `w` down to 2.27% - which is how this check first passed as "not observable" while being
    // perfectly observable.  Measure the quantity the claim is about.
    {
        std::vector<float> got_delta((size_t) hc_dim), want_delta((size_t) hc_dim), wrong_delta((size_t) hc_dim);
        for (long long c = 0; c < hc; ++c)
            for (long long d = 0; d < n_embd; ++d) {
                const size_t i = (size_t) (c * n_embd + d);
                got_delta[i] = (float) ((double) got_write[i] - (double) R[i]);
                want_delta[i] = block_out[(size_t) d];        // 2*sigmoid(0) = 1.0
                wrong_delta[i] = block_out[(size_t) d] * 0.5f; // 1*sigmoid(0) = 0.5
            }
        const double rel = rel_diff(want_delta, wrong_delta);
        const double rel_got = rel_diff(want_delta, got_delta);
        const bool visible = rel > 0.05 && rel_got < 0.05;
        std::printf("  %-40s %s (readings %.0f%% apart, kernel within %.2e)\n",
                    "1*sigmoid vs 2*sigmoid is observable", visible ? "yes" : "*** NO ***", rel * 100, rel_got);
        if (!visible) ++bad;
    }

    // a non-zero injection: every stream receives the SAME block output, differing only by weight
    std::vector<float> inj((size_t) hc);
    for (auto& x : inj) x = gauss(rng) * 3.0f;
    check(cudaMemcpy(d_zi, inj.data(), inj.size() * sizeof(float), cudaMemcpyHostToDevice), "c inj2");
    strata::kernels::gr_write(d_Rw, d_bo, d_zi, sh, d_outw, nullptr);
    check(cudaMemcpy(got_write.data(), d_outw, got_write.size() * sizeof(float), cudaMemcpyDeviceToHost), "c outw2");
    double worst_stream = 0;
    for (long long c = 0; c < hc; ++c) {
        const double w = 2.0 / (1.0 + std::exp(-(double) inj[(size_t) c] / (double) hc));
        for (long long d = 0; d < n_embd; ++d) {
            const size_t i = (size_t) (c * n_embd + d);
            const double want = (double) R[i] + (double) block_out[(size_t) d] * w;
            worst_stream = std::fmax(worst_stream, rel_terms(want, (double) got_write[i], (double) R[i],
                                                             (double) block_out[(size_t) d] * w));
        }
    }
    const bool streams_ok = worst_stream < 1e-6;
    std::printf("  %-40s %s (worst rel %.3e)\n", "every stream adds the same block output",
                streams_ok ? "yes" : "*** NO ***", worst_stream);
    if (!streams_ok) ++bad;

    // ---- THE REAL DIMENSIONS. Exercise all activation/projection variants at n_embd=2560, hc=4, hc_lr=320.
    // Workspace sizing and every full weight row must agree with the model geometry.
    {
        const long long rn = 2560, rhc = 4, rlr = 320, rdim = rhc * rn;
        std::mt19937 rrng(99);
        std::normal_distribution<float> rg(0.0f, 1.0f);
        auto rbf16 = [&](size_t n, float sigma) {
            std::vector<float> v(n);
            for (auto& x : v) x = to_bf16(rg(rrng) * sigma);
            return v;
        };
        std::vector<float> rR((size_t) rdim);
        for (auto& x : rR) x = rg(rrng);
        for (long long c = 0; c < rhc; ++c)
            for (long long d = 0; d < rn; ++d) rR[(size_t) (c * rn + d)] *= std::pow(4.0f, (float) c);
        std::vector<float> rnorm((size_t) rdim);
        for (auto& x : rnorm) x = 1.0f + 0.1f * rg(rrng);
        std::vector<float> rdown = rbf16((size_t) (rlr * rdim), 0.2f);
        std::vector<float> rup = rbf16((size_t) (rdim * rlr), 0.2f);
        std::vector<float> rinj = rbf16((size_t) (rhc * rdim), 0.05f);

        std::vector<uint16_t> qd(rdown.size()), qu(rup.size()), qi(rinj.size());
        for (size_t i = 0; i < rdown.size(); ++i) qd[i] = bf16_bits(rdown[i]);
        for (size_t i = 0; i < rup.size(); ++i) qu[i] = bf16_bits(rup[i]);
        for (size_t i = 0; i < rinj.size(); ++i) qi[i] = bf16_bits(rinj[i]);

        float *dR = nullptr, *dN = nullptr, *dM = nullptr, *dI = nullptr;
        uint16_t *dD = nullptr, *dU = nullptr, *dJ = nullptr;
        check(cudaMalloc(&dR, rR.size() * 4), "rR");
        check(cudaMalloc(&dN, rnorm.size() * 4), "rN");
        check(cudaMalloc(&dD, qd.size() * 2), "rD");
        check(cudaMalloc(&dU, qu.size() * 2), "rU");
        check(cudaMalloc(&dJ, qi.size() * 2), "rJ");
        check(cudaMalloc(&dM, (size_t) rn * 4), "rM");
        check(cudaMalloc(&dI, (size_t) rhc * 4), "rI");
        check(cudaMemcpy(dR, rR.data(), rR.size() * 4, cudaMemcpyHostToDevice), "crR");
        check(cudaMemcpy(dN, rnorm.data(), rnorm.size() * 4, cudaMemcpyHostToDevice), "crN");
        check(cudaMemcpy(dD, qd.data(), qd.size() * 2, cudaMemcpyHostToDevice), "crD");
        check(cudaMemcpy(dU, qu.data(), qu.size() * 2, cudaMemcpyHostToDevice), "crU");
        check(cudaMemcpy(dJ, qi.data(), qi.size() * 2, cudaMemcpyHostToDevice), "crJ");

        const strata::kernels::GrShapes rsh{rn, rhc, rlr};
        void* rws_raw = nullptr;
        check(cudaMalloc(&rws_raw, strata::kernels::gr_workspace_bytes(rsh)), "m rws");
        strata::kernels::GrWorkspace rws;
        strata::kernels::gr_workspace_init(rsh, rws_raw, rws);
        for (int mode = 0; mode < 3; ++mode) {
            select_activation_mode(mode);
            strata::kernels::gr_read(dR, dN, dD, dU, dJ, eps, rsh, rws, dM, dI, nullptr);
            std::vector<float> gm((size_t) rn), gi((size_t) rhc);
            check(cudaMemcpy(gm.data(), dM, gm.size() * 4, cudaMemcpyDeviceToHost), "cgm");
            check(cudaMemcpy(gi.data(), dI, gi.size() * 4, cudaMemcpyDeviceToHost), "cgi");

            Opts precision;
            precision.round_activation = mode == 0;
            std::vector<float> wm, wi;
            reference(rR, rnorm, rdown, rup, rinj, eps, rn, rhc, rlr, precision, wm, wi);
            const double rm = rel_diff(wm, gm), ri = rel_diff(wi, gi);
            const bool ok = rm <= 1e-4 && ri <= 1e-4;
            std::printf("\n  real dims 2560/4/320 %s %s (mixed %.3e, inject %.3e)\n",
                        activation_mode_name(mode), ok ? "pass" : "*** FAIL ***", rm, ri);
            if (!ok) ++bad;
        }
        bad += fused_multi_lds_parity(dN, dD, dU, dJ, eps);
        select_activation_mode(0);
        cudaFree(rws_raw);
        cudaFree(dR); cudaFree(dN); cudaFree(dD); cudaFree(dU); cudaFree(dJ); cudaFree(dM); cudaFree(dI);
    }

    bad += scalar_activation_contract();
    std::printf("\ngr_read/gr_write: %d failures\n", bad);
    if (bad) return 1;
    if (selftest) std::printf("gr_parity OK\n");
    return 0;
}
