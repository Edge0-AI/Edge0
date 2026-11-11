// Exercise the public opt-in dispatch, including tails, padded rows and a nonzero carried state.  The FP32
// recurrence is the all-head reference; short cases also use an independent double-precision CPU recurrence.
#include "strata/prefill/kernels.hpp"
#include "strata/kernels/gfx_arch.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {
constexpr int S = 128, HK = 16, HV = 48, C = 10240, ROW = HV * S;
constexpr uint16_t SENTINEL = 0x7bff;
void ck(cudaError_t e, const char* where) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(e)); std::exit(2); }
}
void setting(const char* key, const char* value) {
#ifdef _WIN32
    if (_putenv_s(key, value)) std::exit(2);
#else
    if (setenv(key, value, 1)) std::exit(2);
#endif
}
template<class X> X* upload(const std::vector<X>& v) {
    X* p = nullptr;
    ck(cudaMalloc(&p, v.size() * sizeof(X)), "malloc");
    ck(cudaMemcpy(p, v.data(), v.size() * sizeof(X), cudaMemcpyHostToDevice), "upload");
    return p;
}
template<class X> std::vector<X> download(const X* p, size_t n) {
    std::vector<X> v(n);
    ck(cudaMemcpy(v.data(), p, n * sizeof(X), cudaMemcpyDeviceToHost), "download");
    return v;
}
float half(uint16_t bits) { __half_raw x; x.x = bits; return __half2float(__half(x)); }
struct Error {
    double square = 0, reference = 0, max_error = 0, max_reference = 0;
    bool finite = true;
    void add(double a, double b) {
        finite &= std::isfinite(a) && std::isfinite(b);
        square += (a - b) * (a - b); reference += a * a;
        max_error = std::max(max_error, std::fabs(a - b)); max_reference = std::max(max_reference, std::fabs(a));
    }
    double rms() const { return std::sqrt(square / std::max(reference, 1e-30)); }
    bool ok() const { return finite && rms() <= 0.003 && max_error <= 0.02 * max_reference + 1e-5; }
};
struct Input {
    int64_t n;
    std::vector<float> h, gate, beta, z, gamma, state;
    Input(int64_t t, float decay, float scale, bool no_update) : n(t), h(std::max<int64_t>(t, 1) * C),
        gate(std::max<int64_t>(t, 1) * HV), beta(gate.size()), z(std::max<int64_t>(t, 1) * ROW),
        gamma(S), state(S * ROW) {
        std::mt19937 rng(11);
        std::normal_distribution<float> normal(0, 1);
        std::uniform_real_distribution<float> uniform(0, 1);
        for (auto& x : h) x = normal(rng);
        for (int64_t i = 0; i < t; ++i) {
            float* row = h.data() + i * C;
            for (int head = 0; head < 2 * HK; ++head) {
                double norm = 0;
                for (int d = 0; d < S; ++d) norm += double(row[head * S + d]) * row[head * S + d];
                for (int d = 0; d < S; ++d) row[head * S + d] /= std::sqrt(norm + 1e-6);
            }
            for (int d = 2 * HK * S; d < C; ++d) row[d] *= scale;
        }
        for (auto& x : gate) x = -decay * uniform(rng);
        for (auto& x : beta) x = no_update ? 0 : uniform(rng);
        for (auto& x : z) x = normal(rng);
        for (auto& x : gamma) x = 0.5f + uniform(rng);
        for (auto& x : state) x = 0.1f * scale * normal(rng);
    }
};
struct Output { std::vector<uint16_t> y; std::vector<float> state; };
Output run(const Input& in, int ld, bool reference, int64_t split = 0) {
    auto h = upload(in.h), g = upload(in.gate), b = upload(in.beta), z = upload(in.z), gm = upload(in.gamma);
    auto state = upload(in.state);
    auto y = upload(std::vector<uint16_t>(std::max<int64_t>(in.n, 1) * ld + 8, SENTINEL));
    // The chunked path owns y as temporary storage; only y16 and state are observable outputs.
    auto scratch = upload(std::vector<float>(std::max<int64_t>(in.n, 1) * ROW));
    auto call = [&](int64_t start, int64_t n) {
        if (reference)
            strata::prefill::gdn_recurrence_variant(5, state, h + start * C, g + start * HV, b + start * HV,
                z + start * ROW, gm, 1e-6f, scratch, y + start * ld, n, nullptr, ld);
        else
            strata::prefill::gdn_recurrence(state, h + start * C, g + start * HV, b + start * HV,
                z + start * ROW, gm, 1e-6f, scratch, y + start * ld, n, nullptr, ld);
        ck(cudaDeviceSynchronize(), "recurrence");
    };
    if (split) { call(0, split); call(split, in.n - split); }
    else call(0, in.n);
    Output out{download(y, std::max<int64_t>(in.n, 1) * ld + 8), download(state, in.state.size())};
    ck(cudaFree(h), "free"); ck(cudaFree(g), "free"); ck(cudaFree(b), "free"); ck(cudaFree(z), "free");
    ck(cudaFree(gm), "free"); ck(cudaFree(state), "free"); ck(cudaFree(y), "free"); ck(cudaFree(scratch), "free");
    return out;
}
bool compare(const Input& in, int ld, const Output& ref, const Output& got, bool exact, const char* label) {
    Error y, state;
    bool guards = true, bits = true;
    for (int64_t t = 0; t < in.n; ++t) {
        for (int d = 0; d < ROW; ++d) {
            y.add(half(ref.y[t * ld + d]), half(got.y[t * ld + d]));
            bits &= ref.y[t * ld + d] == got.y[t * ld + d];
        }
        for (int d = ROW; d < ld; ++d) guards &= got.y[t * ld + d] == SENTINEL;
    }
    for (size_t i = in.n * ld; i < got.y.size(); ++i) guards &= got.y[i] == SENTINEL;
    for (size_t i = 0; i < got.state.size(); ++i) {
        state.add(ref.state[i], got.state[i]);
        bits &= std::memcmp(&ref.state[i], &got.state[i], sizeof(float)) == 0;
    }
    const bool ok = guards && (exact ? bits : y.ok() && state.ok());
    std::printf("%s %s T=%lld ld=%d exact=%d y_rms=%.3e y_max=%.3e state_rms=%.3e guards=%d\n",
        ok ? "PASS" : "FAIL", label, (long long) in.n, ld, exact, y.rms(), y.max_error, state.rms(), guards);
    return ok;
}
bool cpu_check(const Input& in, int ld, const Output& got) {
    Error y, state;
    for (int vh : {0, 23, 47}) {
        const int qh = vh % HK;
        std::vector<double> s(S * S), output(S), delta(S);
        for (int r = 0; r < S; ++r)
            for (int c = 0; c < S; ++c) s[r * S + c] = in.state[r * ROW + vh * S + c];
        for (int64_t t = 0; t < in.n; ++t) {
            const float* row = in.h.data() + t * C;
            const float* q = row + qh * S;
            const float* k = row + HK * S + qh * S;
            const float* v = row + 2 * HK * S + vh * S;
            const double decay = std::exp(in.gate[t * HV + vh]), beta = in.beta[t * HV + vh];
            for (auto& x : s) x *= decay;
            for (int c = 0; c < S; ++c) {
                double prediction = 0;
                for (int r = 0; r < S; ++r) prediction += k[r] * s[r * S + c];
                delta[c] = beta * (v[c] - prediction);
            }
            double norm = 0;
            for (int c = 0; c < S; ++c) {
                double o = 0;
                for (int r = 0; r < S; ++r) { s[r * S + c] += k[r] * delta[c]; o += q[r] * s[r * S + c]; }
                output[c] = o / std::sqrt(double(S)); norm += output[c] * output[c];
            }
            for (int c = 0; c < S; ++c) {
                const double value = output[c] / std::sqrt(norm / S + 1e-6) * in.gamma[c]
                    / (1 + std::exp(-in.z[t * ROW + vh * S + c]));
                y.add(value, half(got.y[t * ld + vh * S + c]));
            }
        }
        for (int r = 0; r < S; ++r)
            for (int c = 0; c < S; ++c) state.add(s[r * S + c], got.state[r * ROW + vh * S + c]);
    }
    const bool ok = y.ok() && state.ok();
    std::printf("%s CPU T=%lld y_rms=%.3e state_rms=%.3e\n", ok ? "PASS" : "FAIL", (long long) in.n, y.rms(), state.rms());
    return ok;
}
// One path's time for one layer's recurrence + norm (GPU-event median of 7 runs after 2 warmups, the same start state each
// run).  STRATA_GDN_CHUNKED is read once per process, so --bench (this path) and --bench-default (gfx1151's default: the
// existing kernels with STRATA_GDN_PP=2 and STRATA_GDN_NOY=1) are separate runs of this binary.
void bench(const char* path) {
    for (int64_t t : {64, 256, 2048, 16384, 32768}) {
        const Input in(t, 2, 1, false);
        auto h = upload(in.h), g = upload(in.gate), b = upload(in.beta), z = upload(in.z), gm = upload(in.gamma);
        auto state = upload(in.state);
        auto y = upload(std::vector<uint16_t>(t * (ROW + 8)));
        auto scratch = upload(std::vector<float>(t * ROW));
        cudaEvent_t start, end;
        ck(cudaEventCreate(&start), "event"); ck(cudaEventCreate(&end), "event");
        std::vector<float> times;
        for (int round = -2; round < 7; ++round) {
            ck(cudaMemcpy(state, in.state.data(), in.state.size() * sizeof(float), cudaMemcpyHostToDevice), "reset");
            ck(cudaEventRecord(start), "event");
            strata::prefill::gdn_recurrence(state, h, g, b, z, gm, 1e-6f, scratch, y, t, nullptr, ROW + 8);
            ck(cudaEventRecord(end), "event"); ck(cudaEventSynchronize(end), "time");
            float ms = 0;
            ck(cudaEventElapsedTime(&ms, start, end), "time");
            if (round >= 0) times.push_back(ms);
        }
        std::sort(times.begin(), times.end());
        std::printf("BENCH path=%s T=%lld ms=%.6f repeats=7\n", path, (long long) t, times[3]);
        ck(cudaEventDestroy(start), "event"); ck(cudaEventDestroy(end), "event");
        ck(cudaFree(h), "free"); ck(cudaFree(g), "free"); ck(cudaFree(b), "free"); ck(cudaFree(z), "free");
        ck(cudaFree(gm), "free"); ck(cudaFree(state), "free"); ck(cudaFree(y), "free"); ck(cudaFree(scratch), "free");
    }
}
}
int main(int argc, char** argv) {
    const bool bench_default = argc > 1 && std::strcmp(argv[1], "--bench-default") == 0;
    const bool off = argc > 1 && (std::strcmp(argv[1], "--off") == 0 || std::strcmp(argv[1], "--default") == 0 || bench_default);
    setting("STRATA_GDN_CHUNKED", off ? "0" : "1");
    setting("STRATA_GDN_HEAD", "1"); setting("STRATA_GDN_PP", "2");
    if (argc > 1 && std::strncmp(argv[1], "--bench", 7) == 0) setting("STRATA_GDN_NOY", "1");   // gfx1151's default
#ifdef _WIN32
    _putenv_s("STRATA_GDN_REC_HEADS", "");
    if (argc > 1 && std::strcmp(argv[1], "--default") == 0) _putenv_s("STRATA_GDN_CHUNKED", "");
#else
    unsetenv("STRATA_GDN_REC_HEADS");
    if (argc > 1 && std::strcmp(argv[1], "--default") == 0) unsetenv("STRATA_GDN_CHUNKED");
#endif
    int device = 0;
    cudaDeviceProp props{};
    ck(cudaGetDevice(&device), "device"); ck(cudaGetDeviceProperties(&props, device), "properties");
    const bool wmma = strata::kernels::gfx_arch_is_gfx11_wmma(props.gcnArchName) || strata::kernels::gfx_arch_is_gfx12_wmma(props.gcnArchName);
    std::printf("device=%s chunked=%d wmma=%d\n", props.gcnArchName, !off, wmma);
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) { bench("chunked"); return 0; }
    if (bench_default) { bench("default"); return 0; }
    int failures = 0;
    for (int64_t t : {0, 1, 15, 16, 17, 63, 64, 65, 79, 80, 81, 257, 2051, 16385}) {
        if (off && t > 2051) continue;
        const Input in(t, 2, 1, false);
        for (int padding : {0, 8, 1}) {
            if (t > 257 && padding != 8) continue;
            const int ld = ROW + padding;
            const auto ref = run(in, ld, true), got = run(in, ld, false);
            const bool exact = off || !wmma || t < 64 || ld % 8;
            failures += !compare(in, ld, ref, got, exact, "dispatch");
            if (t == 65 && padding == 8) failures += !cpu_check(in, ld, got);
            if (t == 257 && padding == 8) failures += !compare(in, ld, got, run(in, ld, false, 129), false, "continuation");
        }
    }
    if (!off) {
        for (const float decay : {0.f, 0.01f}) {
            const Input in(2051, decay, 16, false);
            failures += !compare(in, ROW + 8, run(in, ROW + 8, true), run(in, ROW + 8, false), !wmma, "weak-decay");
        }
        const Input in(65, 0.01f, 1, true);
        const auto got = run(in, ROW, false);
        failures += !compare(in, ROW, run(in, ROW, true), got, !wmma, "beta-zero");
        failures += !cpu_check(in, ROW, got);
    }
    std::printf("FAILURES: %d\n", failures);
    return failures ? 1 : 0;
}
