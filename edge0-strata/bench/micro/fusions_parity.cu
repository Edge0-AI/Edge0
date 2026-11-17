// src/kernels/fusions_parity.cpp - the V100 fusion work against its own unfused pipeline, BITWISE, plus
// the before/after kernel timings (CUDA events).
//
//     build/fusions_parity            every bitwise check below; exit 1 on any difference
//     build/fusions_parity --bench    fused vs unfused kernel times at decode shapes (M=1 and M=8)
//
// WHAT IS CHECKED (synthetic data, GPU, no model):
//   1. `swilu_quantize_q8_0` / `_scaled` against `swiglu` + `quantize_q8_0` / `_scaled` - the S2 expert
//      path's intermediate.  Products, q8 bytes and the fp32 scales all bitwise.
//   2. `swilu_quantize_q8_0` / `_q8_K` with the LEGACY silu against `shared_expert`'s swiglu kernel +
//      `quantize_q8_0` / `quantize_q8_K`.
//   3. `quantize_act_images` against the three standalone quantizers (q8_0 bytes, q8_K bytes, bf16 bits).
//   4. `bf16_gemv_pair` against two `bf16_gemv_split(32)` calls (the warp accumulation it replicates).
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s_gemv.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

namespace {

int g_fail = 0;

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T)), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T)), "memset");
    return p;
}
template <typename T> void up(T* d, const std::vector<T>& h) {
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "h2d");
}
template <typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "d2h");
    return h;
}

void report(const char* name, bool ok) {
    std::printf("  %-52s %s\n", name, ok ? "bitwise identical" : "DIFFERS");
    if (!ok) ++g_fail;
}

// the `gemv` dispatch `shared_expert`'s lambda uses (defined later; declared for `run_bench`)
void gemv_dispatch(const k::SForm& f, const uint8_t* codes, const float* scales, const float* off,
                   const uint8_t* x80, const uint8_t* x8k, float* y, long long n_in, long long n_out,
                   void* stream);

// ---- the unfused SwiGLU references, one thread per element (the shapes the standalone kernels had) ----
__global__ void swilu_legacy_kernel(const float* g, const float* u, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double x = (double) g[i];
    out[i] = (float) (x / (1.0 + exp(-x))) * u[i];
}
__global__ void swilu_fast_kernel(const float* g, const float* u, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = (g[i] / (1.0f + __expf(-g[i]))) * u[i];
}
__global__ void swilu_native_kernel(const float* g, const float* u, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __fdividef(g[i], 1.0f + __expf(-g[i])) * u[i];
}

void fill_pairs(std::vector<float>& g, std::vector<float>& u, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> gv(-6.0f, 6.0f), uv(-3.0f, 3.0f);
    for (size_t i = 0; i < g.size(); ++i) { g[i] = gv(rng); u[i] = uv(rng); }
}

void check_swilu_quantize(int kind, bool scaled, const char* name) {
    const long long n = 640 * 4;                       // FF x a few hits, multiple of 256
    std::vector<float> hg(n), hu(n);
    fill_pairs(hg, hu, 7u + (unsigned) kind * 13 + (unsigned) scaled);
    float *dg = dalloc<float>(n), *du = dalloc<float>(n), *dfused = dalloc<float>(n), *dref = dalloc<float>(n);
    up(dg, hg); up(du, hu);
    const size_t nb80 = (size_t) (n / 32);
    uint8_t* db80 = dalloc<uint8_t>(nb80 * 34 + 256);
    uint8_t* db80r = dalloc<uint8_t>(nb80 * 34 + 256);
    float* ds = dalloc<float>(nb80 + 16);
    float* dsr = dalloc<float>(nb80 + 16);

    // fused
    if (scaled) k::swilu_quantize_q8_0_scaled(dg, du, dfused, n, kind, db80, ds, nullptr);
    else k::swilu_quantize_q8_0(dg, du, dfused, n, kind, db80, nullptr);
    // unfused: standalone swilu then quantize
    const int blocks = (int) ((n + 127) / 128);
    if (kind == 0) swilu_legacy_kernel<<<blocks, 128>>>(dg, du, dref, (int) n);
    else swilu_fast_kernel<<<blocks, 128>>>(dg, du, dref, (int) n);
    ck(cudaGetLastError(), "swilu ref");
    if (scaled) k::quantize_q8_0_scaled(dref, db80r, dsr, n, nullptr);
    else k::quantize_q8_0(dref, db80r, n, nullptr);
    ck(cudaDeviceSynchronize(), "sync");

    const std::vector<float> p1 = down(dfused, n), p2 = down(dref, n);
    const std::vector<uint8_t> q1 = down(db80, nb80 * 34), q2 = down(db80r, nb80 * 34);
    bool ok = std::memcmp(p1.data(), p2.data(), n * sizeof(float)) == 0 &&
              std::memcmp(q1.data(), q2.data(), nb80 * 34) == 0;
    if (scaled) {
        const std::vector<float> s1 = down(ds, nb80), s2 = down(dsr, nb80);
        ok = ok && std::memcmp(s1.data(), s2.data(), nb80 * sizeof(float)) == 0;
    }
    report(name, ok);
    cudaFree(dg); cudaFree(du); cudaFree(dfused); cudaFree(dref); cudaFree(db80); cudaFree(db80r);
    cudaFree(ds); cudaFree(dsr);
}

void check_swilu_quantize_q8_K() {
    const long long n = 256 * 3;
    std::vector<float> hg(n), hu(n);
    fill_pairs(hg, hu, 99u);
    float *dg = dalloc<float>(n), *du = dalloc<float>(n), *df = dalloc<float>(n), *dr = dalloc<float>(n);
    up(dg, hg); up(du, hu);
    const size_t nbK = (size_t) (n / 256);
    uint8_t* dK = dalloc<uint8_t>(nbK * 292 + 256);
    uint8_t* dKr = dalloc<uint8_t>(nbK * 292 + 256);
    k::swilu_quantize_q8_K(dg, du, df, n, 0, dK, nullptr);
    const int blocks = (int) ((n + 127) / 128);
    swilu_legacy_kernel<<<blocks, 128>>>(dg, du, dr, (int) n);
    k::quantize_q8_K(dr, dKr, n, nullptr);
    ck(cudaDeviceSynchronize(), "sync");
    bool ok = down(df, n) == down(dr, n) && down(dK, nbK * 292) == down(dKr, nbK * 292);
    report("swilu_quantize_q8_K (legacy silu)", ok);
    cudaFree(dg); cudaFree(du); cudaFree(df); cudaFree(dr); cudaFree(dK); cudaFree(dKr);
}

void check_images() {
    const long long n = 2560;
    std::vector<float> hx(n);
    std::mt19937 rng(4u);
    std::uniform_real_distribution<float> d(-4.0f, 4.0f);
    for (auto& v : hx) v = d(rng);
    float* dx = dalloc<float>(n);
    up(dx, hx);
    const size_t nb80 = (size_t) (n / 32), nbK = (size_t) (n / 256);
    uint8_t* i80 = dalloc<uint8_t>(nb80 * 34 + 256);
    uint8_t* iK = dalloc<uint8_t>(nbK * 292 + 256);
    uint16_t* i16 = dalloc<uint16_t>(n + 8);
    uint8_t* r80 = dalloc<uint8_t>(nb80 * 34 + 256);
    uint8_t* rK = dalloc<uint8_t>(nbK * 292 + 256);
    k::quantize_act_images(dx, n, i80, iK, i16, nullptr);
    k::quantize_q8_0(dx, r80, n, nullptr);
    k::quantize_q8_K(dx, rK, n, nullptr);
    ck(cudaDeviceSynchronize(), "sync");
    // the standalone bf16 conversion is `bf16_from_f32` per element (`f32_to_bf16_bulk`)
    std::vector<uint16_t> hb(n);
    for (long long i = 0; i < n; ++i) hb[(size_t) i] = k::bf16_from_f32(hx[(size_t) i]);
    const std::vector<uint16_t> got16 = down(i16, n);
    report("quantize_act_images (q8_0 + q8_K + bf16)",
           down(i80, nb80 * 34) == down(r80, nb80 * 34) && down(iK, nbK * 292) == down(rK, nbK * 292) &&
           std::memcmp(got16.data(), hb.data(), n * sizeof(uint16_t)) == 0);
    cudaFree(dx); cudaFree(i80); cudaFree(iK); cudaFree(i16); cudaFree(r80); cudaFree(rK);
}

void check_pair() {
    const long long n_in = 2560, n1 = 48, n2 = 48;    // ssm_alpha / ssm_beta shapes
    std::vector<uint16_t> hx(n_in), hw1(n_in * n1), hw2(n_in * n2);
    std::mt19937 rng(11u);
    std::uniform_real_distribution<float> d(-2.0f, 2.0f);
    auto put = [&](std::vector<uint16_t>& v) {
        for (auto& t : v) t = k::bf16_from_f32(d(rng));
    };
    put(hx); put(hw1); put(hw2);
    uint16_t *dx = dalloc<uint16_t>(n_in), *dw1 = dalloc<uint16_t>(n_in * n1), *dw2 = dalloc<uint16_t>(n_in * n2);
    up(dx, hx); up(dw1, hw1); up(dw2, hw2);
    float *dy1 = dalloc<float>(n1 + 8), *dy2 = dalloc<float>(n2 + 8), *dr1 = dalloc<float>(n1 + 8),
          *dr2 = dalloc<float>(n2 + 8);
    k::bf16_gemv_pair(dx, dw1, dy1, n1, dw2, dy2, n2, n_in, nullptr);
    k::bf16_gemv_split(dx, dw1, dr1, n_in, n1, 32, nullptr);
    k::bf16_gemv_split(dx, dw2, dr2, n_in, n2, 32, nullptr);
    ck(cudaDeviceSynchronize(), "sync");
    report("bf16_gemv_pair vs 2x bf16_gemv_split(32)",
           down(dy1, n1) == down(dr1, n1) && down(dy2, n2) == down(dr2, n2));
    cudaFree(dx); cudaFree(dw1); cudaFree(dw2); cudaFree(dy1); cudaFree(dy2); cudaFree(dr1); cudaFree(dr2);
}

// ---- micro bench: fused vs unfused kernel time at decode shapes, CUDA events -------------------------
//
// The decode geometry is the model's: n_embd 2560, FF 640 for the experts (H/FF from `cpu/expert.hpp`).
// M = 1 is the single-token decode; M = 8 runs the M=8 pair shapes (8 columns of the same weights).

float timed(cudaStream_t s, const std::function<void()>& f, int reps) {
    cudaEvent_t a, b;
    ck(cudaEventCreate(&a), "ev"); ck(cudaEventCreate(&b), "ev");
    f(); ck(cudaStreamSynchronize(s), "sync");               // warmup
    ck(cudaEventRecord(a, s), "rec");
    for (int i = 0; i < reps; ++i) f();
    ck(cudaEventRecord(b, s), "rec");
    ck(cudaEventSynchronize(b), "sync");
    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / reps;
}

void bench_pair(const char* what, const std::function<void()>& before, const std::function<void()>& after,
                int reps) {
    cudaStream_t s = nullptr;
    const float t0 = timed(s, before, reps), t1 = timed(s, after, reps);
    std::printf("  %-44s unfused %8.1f us   fused %8.1f us   %+.1f%%\n", what, t0 * 1e3f, t1 * 1e3f,
                (t1 / t0 - 1.0f) * 100.0f);
}

void run_bench() {
    std::printf("\nfusions bench: decode shapes (n_embd 2560, FF 640), CUDA events, %s\n",
                "stream = default, 200 reps");
    const int reps = 200;

    for (int m : {1, 8}) {
        const long long pairs = (long long) 640 * m;             // one expert hit's rows x tokens
        std::vector<float> hg(pairs), hu(pairs);
        fill_pairs(hg, hu, 5u + (unsigned) m);
        float *dg = dalloc<float>(pairs), *du = dalloc<float>(pairs), *dout = dalloc<float>(pairs);
        up(dg, hg); up(du, hu);
        const size_t nb = (size_t) (pairs / 32);
        uint8_t* d80 = dalloc<uint8_t>(nb * 34 + 256);
        float* ds = dalloc<float>(nb + 16);
        char label[96];
        std::snprintf(label, sizeof label, "expert intermediate  M=%d  (swilu + q8_0)", m);
        bench_pair(label,
                   [&] {
                       const int blocks = (int) ((pairs + 127) / 128);
                       swilu_fast_kernel<<<blocks, 128>>>(dg, du, dout, (int) pairs);
                       k::quantize_q8_0(dout, d80, pairs, nullptr);
                   },
                   [&] { k::swilu_quantize_q8_0(dg, du, dout, pairs, 1, d80, nullptr); }, reps);
        cudaFree(dg); cudaFree(du); cudaFree(dout); cudaFree(d80); cudaFree(ds);
    }

    {
        const long long n = 2560;
        std::vector<float> hx(n);
        std::mt19937 rng(21u);
        std::uniform_real_distribution<float> d(-4.0f, 4.0f);
        for (auto& v : hx) v = d(rng);
        float* dx = dalloc<float>(n);
        up(dx, hx);
        const size_t nb80 = (size_t) (n / 32), nbK = (size_t) (n / 256);
        uint8_t* o80 = dalloc<uint8_t>(nb80 * 34 + 256);
        uint8_t* oK = dalloc<uint8_t>(nbK * 292 + 256);
        uint16_t* o16 = dalloc<uint16_t>(n + 8);
        bench_pair("layer activation images  (q8_0 + q8_K + bf16)",
                   [&] {
                       k::quantize_q8_0(dx, o80, n, nullptr);
                       k::quantize_q8_K(dx, oK, n, nullptr);
                       k::f32_to_bf16_bulk(dx, o16, n, nullptr);
                   },
                   [&] { k::quantize_act_images(dx, n, o80, oK, o16, nullptr); }, reps);
        cudaFree(dx); cudaFree(o80); cudaFree(oK); cudaFree(o16);
    }

    {
        // (a) gate/up dual GEMV + SwiGLU: one kernel vs gemv(gate) + gemv(up) + swilu, shared-expert shapes
        const long long n_in = 2560, n_out = 640;
        const long long n_groups = n_in / 64;
        std::vector<uint8_t> hcg((size_t) n_out * n_in / 2), hcu((size_t) n_out * n_in / 2);
        std::vector<float> hsg((size_t) n_out * n_groups), hsu((size_t) n_out * n_groups);
        std::vector<float> hog((size_t) n_out * n_groups), hou((size_t) n_out * n_groups);
        std::vector<float> hx(n_in);
        std::mt19937 rng(55u);
        for (auto& v : hcg) v = (uint8_t) rng();
        for (auto& v : hcu) v = (uint8_t) rng();
        std::uniform_real_distribution<float> sc(1e-3f, 3e-2f), xv(-4.0f, 4.0f), ov(-1e-2f, 1e-2f);
        for (auto& v : hsg) v = sc(rng);
        for (auto& v : hsu) v = sc(rng);
        for (auto& v : hog) v = ov(rng);
        for (auto& v : hou) v = ov(rng);
        for (auto& v : hx) v = xv(rng);
        float* dx2 = dalloc<float>(n_in);
        up(dx2, hx);
        uint8_t* dx80 = dalloc<uint8_t>((size_t) (n_in / 32) * 34 + 256);
        k::quantize_q8_0(dx2, dx80, n_in, nullptr);
        uint8_t *dcg = dalloc<uint8_t>(hcg.size()), *dcu = dalloc<uint8_t>(hcu.size());
        float *dsg = dalloc<float>(hsg.size()), *dsu = dalloc<float>(hsu.size());
        float *dog = dalloc<float>(hog.size()), *dou = dalloc<float>(hou.size());
        up(dcg, hcg); up(dcu, hcu); up(dsg, hsg); up(dsu, hsu); up(dog, hog); up(dou, hou);
        float *d_gate = dalloc<float>(n_out + 8), *d_up = dalloc<float>(n_out + 8), *d_pair = dalloc<float>(n_out + 8);
        const k::SForm fg{4, -8, 64, k::Codebook::Affine, true, 0};
        const k::SForm fu{4, -8, 64, k::Codebook::Affine, true, 0};
        bench_pair("dual GEMV + swilu (gate,up)  FF=640",
                   [&] {
                       gemv_dispatch(fg, dcg, dsg, dog, dx80, nullptr, d_gate, n_in, n_out, nullptr);
                       gemv_dispatch(fu, dcu, dsu, dou, dx80, nullptr, d_up, n_in, n_out, nullptr);
                       const int bl = (int) ((n_out + 127) / 128);
                       swilu_legacy_kernel<<<bl, 128>>>(d_gate, d_up, d_pair, (int) n_out);
                   },
                   [&] {
                       k::s_gemv_pair_silu(dx80, nullptr, fg, dcg, dsg, dog, fu, dcu, dsu, dou, d_pair,
                                           n_in, n_out, 32, 0, nullptr);
                   }, reps);
        cudaFree(dx2); cudaFree(dx80); cudaFree(dcg); cudaFree(dcu); cudaFree(dsg); cudaFree(dsu);
        cudaFree(dog); cudaFree(dou); cudaFree(d_gate); cudaFree(d_up); cudaFree(d_pair);
    }

    for (int m : {1, 8}) {
        const long long n_in = 2560, n1 = 48, n2 = 48;
        std::vector<uint16_t> hx(n_in), hw1(n_in * n1), hw2(n_in * n2);
        std::mt19937 rng(31u + (unsigned) m);
        std::uniform_real_distribution<float> d(-2.0f, 2.0f);
        auto put = [&](std::vector<uint16_t>& v) { for (auto& t : v) t = k::bf16_from_f32(d(rng)); };
        put(hx); put(hw1); put(hw2);
        uint16_t *dx = dalloc<uint16_t>(n_in), *dw1 = dalloc<uint16_t>(n_in * n1), *dw2 = dalloc<uint16_t>(n_in * n2);
        up(dx, hx); up(dw1, hw1); up(dw2, hw2);
        float *dy1 = dalloc<float>(n1 + 8), *dy2 = dalloc<float>(n2 + 8);
        char label[96];
        std::snprintf(label, sizeof label, "bf16 pair (alpha,beta)   M=%d", m);
        bench_pair(label,
                   [&] {
                       k::bf16_gemv_split(dx, dw1, dy1, n_in, n1, 32, nullptr);
                       k::bf16_gemv_split(dx, dw2, dy2, n_in, n2, 32, nullptr);
                   },
                   [&] { k::bf16_gemv_pair(dx, dw1, dy1, n1, dw2, dy2, n2, n_in, nullptr); }, reps);
        cudaFree(dx); cudaFree(dw1); cudaFree(dw2); cudaFree(dy1); cudaFree(dy2);
    }
}

// ---- (a) the gate/up dual GEMV + SwiGLU kernel against its unfused pipeline, bitwise ----------------
// The unfused pipeline is the one `shared_expert` ran: a standalone GEMV per projection (dispatching the
// same way its `gemv` lambda does) then the standalone SwiGLU kernel.  Both silu kinds are checked.

void gemv_dispatch(const k::SForm& f, const uint8_t* codes, const float* scales, const float* off,
                   const uint8_t* x80, const uint8_t* x8k, float* y, long long n_in, long long n_out,
                   void* stream) {
    if (f.code_bits == 2) {
        std::fprintf(stderr, "gemv_dispatch: code_bits 2 goes through s2_gemv_q8 - not exercised here\n");
        std::exit(2);
    }
    if (f.act_kind == 1) k::s_gemv_q8k_split(x8k, codes, scales, off, y, n_in, n_out, f, stream);
    else k::s_gemv_q8_0_split(x80, codes, scales, off, y, n_in, n_out, f, stream);
}

void check_dual_gemv(int swilu_kind, const char* name) {
    const long long n_in = 2560, n_out = 128;
    const long long n_groups = n_in / 64;
    std::vector<uint8_t> hcg((size_t) n_out * n_in / 2), hcu((size_t) n_out * n_in / 2);  // S4 codes
    std::vector<float> hsg((size_t) n_out * n_groups), hsu((size_t) n_out * n_groups);
    std::vector<float> hog((size_t) n_out * n_groups), hou((size_t) n_out * n_groups);
    std::vector<float> hx(n_in);
    std::mt19937 rng(77u + (unsigned) swilu_kind);
    for (auto& v : hcg) v = (uint8_t) rng();
    for (auto& v : hcu) v = (uint8_t) rng();
    std::uniform_real_distribution<float> sc(1e-3f, 3e-2f), xv(-4.0f, 4.0f), ov(-1e-2f, 1e-2f);
    for (auto& v : hsg) v = sc(rng);
    for (auto& v : hsu) v = sc(rng);
    for (auto& v : hog) v = ov(rng);
    for (auto& v : hou) v = ov(rng);
    for (auto& v : hx) v = xv(rng);

    // Q8_0 activation image of x (both sides use it)
    const size_t nb80 = (size_t) (n_in / 32);
    std::vector<uint8_t> hx80(nb80 * 34 + 256);
    uint8_t* dx80 = dalloc<uint8_t>(hx80.size());
    float* dx = dalloc<float>(n_in);
    up(dx, hx);
    k::quantize_q8_0(dx, dx80, n_in, nullptr);
    ck(cudaDeviceSynchronize(), "sync");

    uint8_t *dcg = dalloc<uint8_t>(hcg.size()), *dcu = dalloc<uint8_t>(hcu.size());
    float *dsg = dalloc<float>(hsg.size()), *dsu = dalloc<float>(hsu.size());
    float *dog = dalloc<float>(hog.size()), *dou = dalloc<float>(hou.size());
    up(dcg, hcg); up(dcu, hcu); up(dsg, hsg); up(dsu, hsu); up(dog, hog); up(dou, hou);
    float* d_pair = dalloc<float>(n_out + 8);
    float* d_gate = dalloc<float>(n_out + 8), *d_up = dalloc<float>(n_out + 8), *d_ref = dalloc<float>(n_out + 8);

    const k::SForm fg{4, -8, 64, k::Codebook::Affine, true, 0};
    const k::SForm fu{4, -8, 64, k::Codebook::Affine, true, 0};
    const bool ok_launch = k::s_gemv_pair_silu(dx80, nullptr, fg, dcg, dsg, dog, fu, dcu, dsu, dou,
                                               d_pair, n_in, n_out, 32, swilu_kind, nullptr);
    // unfused: the same two GEMVs the `gemv` lambda would run, then the standalone swilu
    gemv_dispatch(fg, dcg, dsg, dog, dx80, nullptr, d_gate, n_in, n_out, nullptr);
    gemv_dispatch(fu, dcu, dsu, dou, dx80, nullptr, d_up, n_in, n_out, nullptr);
    const int blocks = (int) ((n_out + 127) / 128);
    if (swilu_kind == 0) swilu_legacy_kernel<<<blocks, 128>>>(d_gate, d_up, d_ref, (int) n_out);
    else if (swilu_kind == 1) swilu_fast_kernel<<<blocks, 128>>>(d_gate, d_up, d_ref, (int) n_out);
    else swilu_native_kernel<<<blocks, 128>>>(d_gate, d_up, d_ref, (int) n_out);
    ck(cudaGetLastError(), "swilu ref");
    ck(cudaDeviceSynchronize(), "sync");
    report(name, ok_launch && down(d_pair, n_out) == down(d_ref, n_out));

    cudaFree(dx80); cudaFree(dx); cudaFree(dcg); cudaFree(dcu); cudaFree(dsg); cudaFree(dsu);
    cudaFree(dog); cudaFree(dou); cudaFree(d_pair); cudaFree(d_gate); cudaFree(d_up); cudaFree(d_ref);
}

}  // namespace

int main(int argc, char** argv) {
    bool bench = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--bench") bench = true;
        else { std::fprintf(stderr, "usage: fusions_parity [--bench]\n"); return 2; }
    std::printf("fusions_parity: V100 fusions vs their unfused pipelines, bitwise\n");
    check_swilu_quantize(1, false, "swilu_quantize_q8_0 (fast silu, expert path)");
    check_swilu_quantize(1, true, "swilu_quantize_q8_0_scaled (fast silu, expert path)");
    check_swilu_quantize(0, false, "swilu_quantize_q8_0 (legacy silu, shared expert)");
    check_swilu_quantize_q8_K();
    check_images();
    check_pair();
    check_dual_gemv(0, "s_gemv_pair_silu (legacy silu) vs gemv+gemv+swilu");
    check_dual_gemv(1, "s_gemv_pair_silu (fast silu) vs gemv+gemv+swilu");
    check_dual_gemv(2, "s_gemv_pair_silu (native silu) vs gemv+gemv+swilu");
    if (bench) run_bench();
    std::printf(g_fail ? "fusions_parity: %d failures\n" : "fusions_parity: 0 failures\n", g_fail);
    return g_fail ? 1 : 0;
}
