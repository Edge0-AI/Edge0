// bench_tc_gemv.cu - ISOLATED benchmark: BF16 GEMV baseline vs Tensor Core (mma.m16n8k8, BF16->FP16) on sm_75.
// Nothing in the engine changes. Shapes are the REAL Strata decode GEMVs (BF16 weights, n_in = n_embd = 2560):
//   ssm_alpha / ssm_beta   [2560, 48]   project_bf16(split=true)  -> bf16_gemv_split(TPR=32) -> warp kernel
//   indexer.k_proj         [2560, 128]  project_bf16(split=false) -> bf16_gemv -> warp kernel
//   indexer.q_proj / ffn_gate_inp / mlp.gate [2560, 512]  warp / split(TPR=32) -> warp kernel
//   ple_value              [2560, 2560] warp
// Baseline = bf16_gemv_warp_kernel transcribed verbatim from src/kernels/cuda/bf16_gemv.cu (production).
// TC variant = same maths; A = x broadcast down m16 (single-token GEMV), k-chunk = 8;
//   (a) W preconverted bf16->fp16 once, x converted in-kernel (timed)   [tc_preconv]
//   (b) fully inline: W read as bf16 and converted in-register           [tc_inline]
// Numerics: fp64 host reference; max abs err for baseline and both TC variants; bf16->fp16 exactness scan.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>

#define CHECK(e) do { cudaError_t err_ = (e); if (err_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); exit(1); } } while (0)

__device__ __forceinline__ float bf16_f32(uint16_t v) { return __int_as_float((int) ((unsigned) v << 16)); }
static float bf16_f32_host(uint16_t v) { unsigned u = (unsigned) v << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t f32_bf16_host(float f) {
    unsigned u, r; memcpy(&u, &f, 4);
    r = ((u >> 16) ^ (u >> 15)) & 1; u += 0x7fff + r;
    return (uint16_t) (u >> 16);
}

// ============ baseline: production warp kernel (verbatim semantics from bf16_gemv.cu) ============
__global__ void gemv_warp(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
                          float* __restrict__ y, long long n_in, long long n_out) {
    const long long o = (long long) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (o >= n_out) return;
    const int lane = threadIdx.x & 31;
    const uint16_t* row = w + o * n_in;
    float acc = 0.0f;
    for (long long i = lane; i < n_in; i += 32)
        acc += bf16_f32(x[i]) * bf16_f32(row[i]);
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    if (lane == 0) y[o] = acc;
}

// ============ TC: A(x) broadcast, k-split over blocks, partial D row 0 -> scratch ============
// A fragment (m16n8k8, .f16): lane l: a0 = A[m=l>>2][k=2(l&3)], a1 = A[m][k=2(l&3)+1],
//                               a2/a3 = same cols at row m+8.  Broadcast => a2=a0, a3=a1.
// B fragment (.f16, col-major): lane l: b0 = B[k=2(l&3)][n=(l>>2)], b1 = B[k=2(l&3)+1][n].
//   B[k][n] = W[n][k], so lane l loads W[n][2(l&3)], W[n][2(l&3)+1] - coalesced within the row.
// D fragment: c0/c1 = D[m=l>>2][cols 2(l&3), +1], c2/c3 = row m+8.  All m rows identical -> row 0.
__global__ void tc_preconv(const uint16_t* __restrict__ x_b, const uint16_t* __restrict__ w_f,
                           float* __restrict__ partials, long long n_in, long long n_out, int ks) {
    const int tile = blockIdx.x, ks_i = blockIdx.y, lane = threadIdx.x & 31;
    const long long n_base = (long long) tile * 8;
    const long long k_per = n_in / (long long) ks;
    const long long k0 = (long long) ks_i * k_per;
    const int t2 = 2 * (lane & 3);
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    for (long long kk0 = k0; kk0 < k0 + k_per; kk0 += 8) {
        // A (2 x f16x2): a0 = {A[m][2t], A[m+8][2t]}, a1 = {A[m][2t+1], A[m+8][2t+1]} (broadcast -> dup halves)
        unsigned xh = (unsigned) x_b[kk0 + t2] | ((unsigned) x_b[kk0 + t2 + 1] << 16);
        // fp16 conversion in-register: bf16 -> f32 -> f16 (exact within fp16 range)
        __half2 ha = __floats2half2_rn(bf16_f32((uint16_t) (xh & 0xFFFF)), bf16_f32((uint16_t) (xh >> 16)));
        __half2 hb = __floats2half2_rn(bf16_f32((uint16_t) (xh & 0xFFFF)), bf16_f32((uint16_t) (xh >> 16)));
        unsigned a0 = *reinterpret_cast<unsigned*>(&ha);
        unsigned a1 = *reinterpret_cast<unsigned*>(&hb);
        // B is already fp16 here: load as __half2 directly (bf16 path must NOT convert twice)
        __half2 wbh = *reinterpret_cast<const __half2*>(&w_f[(n_base + (lane >> 2)) * n_in + kk0 + t2]);
        unsigned b0 = *reinterpret_cast<unsigned*>(&wbh);
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                     : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                     : "r"(a0), "r"(a1), "r"(b0));
    }
    if (lane < 4) { float* p = partials + (size_t) (tile * ks + ks_i) * 8; p[t2] = c0; p[t2 + 1] = c1; }
}

// same, but W stays bf16 and converts in-register (fully fair per-call cost)
__global__ void tc_inline(const uint16_t* __restrict__ x_b, const uint16_t* __restrict__ w_b,
                          float* __restrict__ partials, long long n_in, long long n_out, int ks) {
    const int tile = blockIdx.x, ks_i = blockIdx.y, lane = threadIdx.x & 31;
    const long long n_base = (long long) tile * 8;
    const long long k_per = n_in / (long long) ks;
    const long long k0 = (long long) ks_i * k_per;
    const int t2 = 2 * (lane & 3);
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    for (long long kk0 = k0; kk0 < k0 + k_per; kk0 += 8) {
        unsigned xh = (unsigned) x_b[kk0 + t2] | ((unsigned) x_b[kk0 + t2 + 1] << 16);
        __half2 ha = __floats2half2_rn(bf16_f32((uint16_t) (xh & 0xFFFF)), bf16_f32((uint16_t) (xh >> 16)));
        __half2 hb = __floats2half2_rn(bf16_f32((uint16_t) (xh & 0xFFFF)), bf16_f32((uint16_t) (xh >> 16)));
        unsigned a0 = *reinterpret_cast<unsigned*>(&ha);
        unsigned a1 = *reinterpret_cast<unsigned*>(&hb);
        const uint16_t* wbv = w_b + (n_base + (lane >> 2)) * n_in + kk0 + t2;
        __half2 wbh = __floats2half2_rn(bf16_f32(wbv[0]), bf16_f32(wbv[1]));
        unsigned b0 = *reinterpret_cast<unsigned*>(&wbh);
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                     : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                     : "r"(a0), "r"(a1), "r"(b0));
    }
    if (lane < 4) { float* p = partials + (size_t) (tile * ks + ks_i) * 8; p[t2] = c0; p[t2 + 1] = c1; }
}

__global__ void tc_reduce(const float* __restrict__ partials, float* __restrict__ y, long long n_out, int ks) {
    const long long o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n_out) return;
    float s = 0.f;
    for (int q = 0; q < ks; ++q) s += partials[(size_t) ((o >> 3) * ks + q) * 8 + (o & 7)];
    y[o] = s;
}

// variant with the OTHER candidate A layout: a0 = {A[gid][2t], A[gid+8][2t]} (same k-col, two rows),
// a1 = same for col 2t+1. Broadcast => both halves of a0 hold x[2t], both halves of a1 hold x[2t+1].
__global__ void tc_dupcol(const uint16_t* __restrict__ x_b, const uint16_t* __restrict__ w_b,
                          float* __restrict__ partials, long long n_in, long long n_out, int ks) {
    const int tile = blockIdx.x, ks_i = blockIdx.y, lane = threadIdx.x & 31;
    const long long n_base = (long long) tile * 8;
    const long long k_per = n_in / (long long) ks;
    const long long k0 = (long long) ks_i * k_per;
    const int t2 = 2 * (lane & 3);
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    for (long long kk0 = k0; kk0 < k0 + k_per; kk0 += 8) {
        __half2 ha = __floats2half2_rn(bf16_f32(x_b[kk0 + t2]), bf16_f32(x_b[kk0 + t2]));
        __half2 hb = __floats2half2_rn(bf16_f32(x_b[kk0 + t2 + 1]), bf16_f32(x_b[kk0 + t2 + 1]));
        unsigned a0 = *reinterpret_cast<unsigned*>(&ha);
        unsigned a1 = *reinterpret_cast<unsigned*>(&hb);
        const uint16_t* wbv = w_b + (n_base + (lane >> 2)) * n_in + kk0 + t2;
        __half2 wbh = __floats2half2_rn(bf16_f32(wbv[0]), bf16_f32(wbv[1]));
        unsigned b0 = *reinterpret_cast<unsigned*>(&wbh);
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                     : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                     : "r"(a0), "r"(a1), "r"(b0));
    }
    if (lane < 4) { float* p = partials + (size_t) (tile * ks + ks_i) * 8; p[t2] = c0; p[t2 + 1] = c1; }
}

// FUSED, ks=1 only: with a single k-fragment the partial IS the answer -> write y directly, no reduce kernel,
// no scratch. Fully inline bf16 W conversion (fairest per-call TC cost).
__global__ void tc_fused(const uint16_t* __restrict__ x_b, const uint16_t* __restrict__ w_b,
                         float* __restrict__ y, long long n_in, long long n_out) {
    const int tile = blockIdx.x, lane = threadIdx.x & 31;
    const long long n_base = (long long) tile * 8;
    const int t2 = 2 * (lane & 3);
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    for (long long kk0 = 0; kk0 < n_in; kk0 += 8) {
        __half2 ha = __floats2half2_rn(bf16_f32(x_b[kk0 + t2]), bf16_f32(x_b[kk0 + t2 + 1]));
        __half2 hb = ha;   // broadcast: row gid+8 = row gid (single-token GEMV)
        unsigned a0 = *reinterpret_cast<unsigned*>(&ha);
        unsigned a1 = *reinterpret_cast<unsigned*>(&hb);
        const uint16_t* wbv = w_b + (n_base + (lane >> 2)) * n_in + kk0 + t2;
        __half2 wbh = __floats2half2_rn(bf16_f32(wbv[0]), bf16_f32(wbv[1]));
        unsigned b0 = *reinterpret_cast<unsigned*>(&wbh);
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                     : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                     : "r"(a0), "r"(a1), "r"(b0));
    }
    if (lane < 4) { y[n_base + t2] = c0; y[n_base + t2 + 1] = c1; }
}

int main() {
    CHECK(cudaSetDevice(0));
    cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s, SMs %d, capability %d.%d\n", prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    struct Shape { const char* name; long long n_in, n_out; };
    Shape shapes[] = {
        {"ssm_alpha/beta [2560,48]", 2560, 48},
        {"indexer.k_proj [2560,128]", 2560, 128},
        {"q_proj/router [2560,512]", 2560, 512},
        {"ple_value [2560,2560]", 2560, 2560},
    };
    auto rng = [](unsigned& s) { s = s * 1664525u + 1013904223u; return (float) ((s >> 8) & 0xFFFF) / 65535.0f - 0.5f; };
    const int ITERS = 2000, WARM = 300;

    printf("\n%-26s %13s | %s\n", "shape", "baseline us", "tc variants (pc = W preconverted, in = inline conversion, fused = no reduce), speedup vs baseline");
    printf("%s\n", std::string(150, '-').c_str());

    for (auto& sh : shapes) {
        const long long n_in = sh.n_in, n_out = sh.n_out;
        const size_t wn = (size_t) n_out * n_in;
        std::vector<uint16_t> hw(wn), hx(n_in), hf(wn);
        std::vector<float> hyb(n_out), hypc(n_out), hyin(n_out), hyfu(n_out);
        std::vector<double> href(n_out);
        unsigned seed = 991u + (unsigned) (n_out & 0xFFF);
        for (size_t i = 0; i < wn; ++i) hw[i] = f32_bf16_host(rng(seed) * 0.3f);
        for (long long i = 0; i < n_in; ++i) hx[i] = f32_bf16_host(rng(seed) * 3.0f);
        for (long long o = 0; o < n_out; ++o) {
            double s = 0.0;
            for (long long i = 0; i < n_in; ++i) s += (double) bf16_f32_host(hx[i]) * (double) bf16_f32_host(hw[(size_t) o * n_in + i]);
            href[o] = s;
        }
        // preconvert W to fp16 (bf16 -> fp16 is exact within fp16 range)
        for (size_t i = 0; i < wn; ++i) { __half h = __float2half_rn(bf16_f32_host(hw[i])); memcpy(&hf[i], &h, 2); }

        uint16_t *dx, *dw, *dwf; float *dy, *dyp;
        CHECK(cudaMalloc(&dx, n_in * 2));
        CHECK(cudaMalloc(&dw, wn * 2)); CHECK(cudaMalloc(&dwf, wn * 2));
        CHECK(cudaMalloc(&dy, n_out * 4));
        CHECK(cudaMalloc(&dyp, (size_t)(n_out / 8) * 65 * 8 * 4));
        CHECK(cudaMemcpy(dx, hx.data(), n_in * 2, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dw, hw.data(), wn * 2, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dwf, hf.data(), wn * 2, cudaMemcpyHostToDevice));

        // one-time W conversion cost (host->device, reported separately; the engine would fold this
        // into the weight load exactly like Gemm::bf16 does for sm<80)
        {
            cudaEvent_t e0, e1; CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
            CHECK(cudaEventRecord(e0));
            CHECK(cudaMemcpy(dwf, hf.data(), wn * 2, cudaMemcpyHostToDevice));
            CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
            float ms = 0; CHECK(cudaEventElapsedTime(&ms, e0, e1));
            printf("%-26s one-time W convert: %.2f ms (%.1f MB), tiles=%lld\n", sh.name, ms, wn * 2 / 1e6, n_out / 8);
        }

        int tiles = (int) (n_out / 8);
        // valid ks must divide n_in so each k-chunk is a multiple of 8; sweep ALL of them, report the best
        const int cands[11] = {1, 2, 4, 5, 8, 10, 16, 20, 32, 40, 64};

        auto timeit = [&](const char* tag, const std::function<void()>& fn) -> double {
            for (int i = 0; i < WARM; ++i) fn();
            CHECK(cudaDeviceSynchronize());
            double best = 1e30;
            for (int rep = 0; rep < 3; ++rep) {
                cudaEvent_t e0, e1; CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
                CHECK(cudaEventRecord(e0));
                for (int i = 0; i < ITERS; ++i) fn();
                CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
                float ms = 0; CHECK(cudaEventElapsedTime(&ms, e0, e1));
                double us = ms * 1000.0 / ITERS;
                if (us < best) best = us;
            }
            (void) tag; return best;
        };

        double t_base = timeit("base", [&]{ gemv_warp<<<(unsigned) ((n_out + 7) / 8), 256>>>(dx, dw, dy, n_in, n_out); });
        gemv_warp<<<(unsigned) ((n_out + 7) / 8), 256>>>(dx, dw, dy, n_in, n_out);
        CHECK(cudaMemcpy(hyb.data(), dy, n_out * 4, cudaMemcpyDeviceToHost));

        // occupancy sweep over k-chunk count for the preconverted-W TC kernel
        int best_ks = 1; double t_pc = 1e30;
        for (int i = 0; i < 11; ++i) {
            const int ksv = cands[i];
            if (n_in % ksv != 0) continue;
            dim3 g((unsigned) tiles, (unsigned) ksv);
            double t = timeit("pcs", [&]{ tc_preconv<<<g, 32>>>(dx, dwf, dyp, n_in, n_out, ksv);
                                          tc_reduce<<<(unsigned) ((n_out + 255) / 256), 256>>>(dyp, dy, n_out, ksv); });
            if (t < t_pc) { t_pc = t; best_ks = ksv; }
        }
        dim3 gtc((unsigned) tiles, (unsigned) best_ks);
        tc_preconv<<<gtc, 32>>>(dx, dwf, dyp, n_in, n_out, best_ks);
        tc_reduce<<<(unsigned) ((n_out + 255) / 256), 256>>>(dyp, dy, n_out, best_ks);
        CHECK(cudaMemcpy(hypc.data(), dy, n_out * 4, cudaMemcpyDeviceToHost));

        double t_ti = timeit("tci", [&]{ tc_inline<<<gtc, 32>>>(dx, dw, dyp, n_in, n_out, best_ks);
                                         tc_reduce<<<(unsigned) ((n_out + 255) / 256), 256>>>(dyp, dy, n_out, best_ks); });
        tc_inline<<<gtc, 32>>>(dx, dw, dyp, n_in, n_out, best_ks);
        tc_reduce<<<(unsigned) ((n_out + 255) / 256), 256>>>(dyp, dy, n_out, best_ks);
        CHECK(cudaMemcpy(hyin.data(), dy, n_out * 4, cudaMemcpyDeviceToHost));

        double t_fu = timeit("tcf", [&]{ tc_fused<<<(unsigned) tiles, 32>>>(dx, dw, dy, n_in, n_out); });
        tc_fused<<<(unsigned) tiles, 32>>>(dx, dw, dy, n_in, n_out);
        CHECK(cudaMemcpy(hyfu.data(), dy, n_out * 4, cudaMemcpyDeviceToHost));

        double best_tc = t_pc; const char* best_var = "preconv";
        if (t_ti < best_tc) { best_tc = t_ti; best_var = "inline"; }
        if (t_fu < best_tc) { best_tc = t_fu; best_var = "fused"; }

        double e_tc = 0, e_base = 0, sc = 0;
        if (strcmp(best_var, "preconv") == 0) {
            for (long long o = 0; o < n_out; ++o) { e_tc = std::max(e_tc, std::fabs((double) hypc[o] - href[o])); }
        } else if (strcmp(best_var, "inline") == 0) {
            for (long long o = 0; o < n_out; ++o) { e_tc = std::max(e_tc, std::fabs((double) hyin[o] - href[o])); }
        } else {
            for (long long o = 0; o < n_out; ++o) { e_tc = std::max(e_tc, std::fabs((double) hyfu[o] - href[o])); }
        }
        for (long long o = 0; o < n_out; ++o) {
            e_base = std::max(e_base, std::fabs((double) hyb[o] - href[o]));
            sc = std::max(sc, std::fabs(href[o]));
        }
        double gb_base = (double) (wn * 2) / t_base / 1e3;
        double gb_tc = (double) (wn * 2) / best_tc / 1e3;
        printf("%-26s %13.3f | pc %13.3f (ks=%d) | in %13.3f | fused %13.3f | BEST %s %7.3fx | maxerr tc %.2e base %.2e  (scale %.1f, BW base %.0f tc %.0f GB/s)\n",
               sh.name, t_base, t_pc, best_ks, t_ti, t_fu, best_var, t_base / best_tc,
               e_tc, e_base, sc, gb_base, gb_tc);

        CHECK(cudaFree(dx)); CHECK(cudaFree(dw)); CHECK(cudaFree(dwf)); CHECK(cudaFree(dy)); CHECK(cudaFree(dyp));
    }

    {   // bf16 -> fp16 exactness scan
        unsigned seed = 777u; double worst = 0; int bad = 0, n = 2000000;
        for (int i = 0; i < n; ++i) {
            float f = (rng(seed) * 2.0f - 1.0f) * 100.0f;
            uint16_t b = f32_bf16_host(f);
            __half h = __float2half_rn(bf16_f32_host(b));
            if ((double) __half2float(h) != (double) bf16_f32_host(b)) {
                bad++; worst = std::max(worst, std::fabs((double) __half2float(h) - (double) bf16_f32_host(b)));
            }
        }
        printf("\nbf16->fp16 conversion exactness: %d / %d inexact, worst %.3g\n", bad, n, worst);
    }
    printf("BENCH_DONE\n");
    return 0;
}
