// tools/dsv41_lab/tqk_lab.cu - parity + speed of the trellis expert kernels (tqk_gemv.cuh) with synthetic weights.
//   tqk_lab [--k 4,5,6,7] [--nb 1,2,3,4,5] [--reps 200] [--quick]
// Parity: dequant vs tqenc::decode_blocks (bit exact), GEMV vs a double reference on the host-decoded weights.
// Bench: one decode token of a routed layer = top-6 experts x (gate, up: 2304 x 5120; down: 5120 x 2304).
#include "tqk_gemv.cuh"
#include "tqk_gemv3.cuh"
#include "tqk_gemv_c.cuh"
#include "tqk_repack.hpp"
#include "../trellis/tqenc.hpp"
#include "strata/kernels/cpu/tq_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

using namespace tqlab;

static uint64_t g_rng = 0x9e3779b97f4a7c15ull;
static uint64_t rnd() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return g_rng; }
static float frnd() { return (float) ((rnd() >> 40) * (1.0 / 16777216.0)) * 2.f - 1.f; }

static int type_of(int K) { return K == 4 ? 225 : K == 5 ? 224 : K == 6 ? 145 : 146; }

// random legal weights: any bit pattern is a legal stream; scale = a normal-ish fp16 in [0.02, 0.5]
static void fill_random(std::vector<uint8_t>& w, int K, int64_t nblocks) {
    const int bsz = 2 + 4 * K;
    w.resize((size_t) nblocks * bsz);
    for (size_t i = 0; i < w.size(); i += 8) { uint64_t r = rnd(); std::memcpy(&w[i], &r, std::min<size_t>(8, w.size() - i)); }
    for (int64_t b = 0; b < nblocks; ++b) {
        const uint16_t d = tqenc::float_to_half(0.02f + 0.48f * (frnd() * 0.5f + 0.5f));
        std::memcpy(&w[(size_t) b * bsz], &d, 2);
    }
}

__global__ void rd_kernel(const float4* p, size_t n4, float* o) {
    float s = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n4; i += (size_t) gridDim.x * blockDim.x) { float4 v = p[i]; s += v.x + v.y + v.z + v.w; }
    if (s == 1234.5f) *o = s;
}
struct KFn { const void* f; int threads; int rows_per_cta; int lutkind; int tql = 0; };   // lutkind 0: float2 table, 1: u32 half2 table
template <int K, int NB> static KFn pick(int variant) {
    switch (variant) {
        case 1: return {(const void*) gemv_v1<K, NB, 1>, 256, 8, 0};
        case 2: return {(const void*) gemv_v1<K, NB, 2>, 256, 16, 0};
        case 3: return {(const void*) gemv_v2<K, NB, 1, 1>, 128, 64, 1};
        case 4: return {(const void*) gemv_v2<K, NB, 2, 1>, 128, 128, 1};
        case 5: return {(const void*) gemv_v2<K, NB, 4, 1>, 128, 256, 1};
        case 6: return {(const void*) gemv_v2<K, NB, 2, 0>, 128, 128, 1};
        case 7: return {(const void*) gemv_v2<K, NB, 4, 0>, 128, 256, 1};
        case 8: return {(const void*) gemv_v3<K, NB, 1, 1>, 128, 64, 1, 1};
        case 9: return {(const void*) gemv_v3<K, NB, 2, 1>, 128, 128, 1, 1};
        case 10: return {(const void*) gemv_v3<K, NB, 4, 1>, 128, 256, 1, 1};
        case 11: return {(const void*) gemv_v3<K, NB, 2, 0>, 128, 128, 1, 1};
        case 12: return {(const void*) gemv_v3c<K, NB, 4, 1, 0>, 128, 256, 1, 1};
        case 13: return {(const void*) gemv_v3c<K, NB, 4, 1, 1>, 128, 256, 1, 1};
        case 14: return {(const void*) gemv_v3c<K, NB, 2, 1, 0>, 128, 128, 1, 1};
        case 15: return {(const void*) gemv_v3c<K, NB, 2, 1, 1>, 128, 128, 1, 1};
        case 16: return {(const void*) gemv_v3c<K, NB, 2, 0, 0>, 128, 128, 1, 1};
        case 17: return {(const void*) gemv_v3c<K, NB, 4, 0, 0>, 128, 256, 1, 1};
        default: return {(const void*) gemv_v3c<K, NB, 1, 0, 0>, 128, 64, 1, 1};
    }
}
template <int K> static KFn pick_nb(int nb, int variant) {
    switch (nb) {
        case 1: return pick<K, 1>(variant);
        case 2: return pick<K, 2>(variant);
        case 3: return pick<K, 3>(variant);
        case 4: return pick<K, 4>(variant);
        default: return pick<K, 5>(variant);
    }
}
static KFn pick_k(int K, int nb, int variant) {
    switch (K) {
        case 4: return pick_nb<4>(nb, variant);
        case 5: return pick_nb<5>(nb, variant);
        case 6: return pick_nb<6>(nb, variant);
        default: return pick_nb<7>(nb, variant);
    }
}
static const float2* g_lut_f2; static const uint32_t* g_lut_u32;
static void launch(const KFn& k, const Job* jd, int njobs, int rows, int in, cudaStream_t s = 0) {
    dim3 grid((rows + k.rows_per_cta - 1) / k.rows_per_cta, njobs);
    const void* lut = k.lutkind ? (const void*) g_lut_u32 : (const void*) g_lut_f2;
    void* args[] = {(void*) &jd, (void*) &lut, (void*) &in};
    cudaLaunchKernel(k.f, grid, dim3(k.threads), args, 0, s);
}

static float* dev_f32(const std::vector<float>& h) {
    float* d; CK(cudaMalloc(&d, h.size() * 4)); CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice)); return d;
}

int main(int argc, char** argv) {
    std::vector<int> Ks = {4, 5, 6, 7}, NBs = {1, 2, 3, 4, 5};
    int reps = 200, variant = 2, dev = 0;
    bool quick = false;
    auto parse = [](const char* s) { std::vector<int> v; for (const char* p = s; *p;) { v.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; } return v; };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--k" && i + 1 < argc) Ks = parse(argv[++i]);
        else if (a == "--nb" && i + 1 < argc) NBs = parse(argv[++i]);
        else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (a == "--variant" && i + 1 < argc) variant = std::atoi(argv[++i]);
        else if (a == "--dev" && i + 1 < argc) dev = std::atoi(argv[++i]);
        else if (a == "--quick") quick = true;
    }
    CK(cudaSetDevice(dev));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
    std::printf("device %s sm_%d%d, %d SMs, mem clock %d kHz bus %d bit\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                prop.memoryClockRate, prop.memoryBusWidth);

    // point table (x, y) as the CPU decoder sees it
    const float* pt = tqenc::point_table();
    float2* lut; CK(cudaMalloc(&lut, 2048 * sizeof(float2))); CK(cudaMemcpy(lut, pt, 2048 * sizeof(float2), cudaMemcpyHostToDevice));
    g_lut_f2 = lut;
    {   // half2 table: must be exact (the claim the tensor-core kernels rest on)
        std::vector<uint32_t> hl(2048); int inexact = 0;
        for (int i = 0; i < 2048; ++i) {
            const uint16_t a = tqenc::float_to_half(pt[2 * i]), b = tqenc::float_to_half(pt[2 * i + 1]);
            inexact += tqenc::half_to_float(a) != pt[2 * i] || tqenc::half_to_float(b) != pt[2 * i + 1];
            hl[i] = a | ((uint32_t) b << 16);
        }
        std::printf("fp16 point table inexact entries: %d\n", inexact);
        uint32_t* d; CK(cudaMalloc(&d, 8192)); CK(cudaMemcpy(d, hl.data(), 8192, cudaMemcpyHostToDevice)); g_lut_u32 = d;
    }

    // streaming bandwidth baseline: read-only sum over 512 MB
    {
        const size_t n = 128u << 20;   // floats
        float4* buf; CK(cudaMalloc(&buf, n * 4)); CK(cudaMemset(buf, 0, n * 4));
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        float* o; CK(cudaMalloc(&o, 4));
        rd_kernel<<<prop.multiProcessorCount * 8, 256>>>(buf, n / 4, o); CK(cudaDeviceSynchronize());
        cudaEventRecord(e0);
        for (int i = 0; i < 10; ++i) rd_kernel<<<prop.multiProcessorCount * 8, 256>>>(buf, n / 4, o);
        cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        std::printf("stream-read baseline: %.1f GB/s (spec 448)\n", 10.0 * n * 4 / (ms * 1e-3) / 1e9);
        cudaFree(buf); cudaFree(o);
    }

    int fails = 0;
    for (int K : Ks) {
        const int type = type_of(K), bsz = 2 + 4 * K;
        std::printf("\n== K=%d (type %d, %d B / 128 weights, %.3f bpw) ==\n", K, type, bsz, bsz * 8 / 128.0);
        // ---- parity 1: dequant bit exact vs the CPU decoder (random streams + encoded N(0,1) rows)
        {
            const int64_t nblocks = 4096;
            std::vector<uint8_t> w; fill_random(w, K, nblocks);
            // second half: really encoded Gaussian weights
            const int64_t nenc = 256;
            std::vector<float> g(nenc * 128); for (auto& v : g) { float u = 0; for (int i = 0; i < 12; ++i) u += frnd() * 0.5f + 0.5f; v = u - 6.f; }
            std::vector<uint8_t> enc(nenc * bsz);
            tqenc::Params p; double err = 0; int64_t nf = 0, zb = 0;
            err = tqenc::encode_blocks(type, g.data(), nullptr, nenc, enc.data(), p, 4, nullptr, &nf, &zb);
            std::memcpy(&w[(size_t) (nblocks - nenc) * bsz], enc.data(), enc.size());
            std::vector<float> ref((size_t) nblocks * 128);
            tqenc::decode_blocks(type, w.data(), nblocks, ref.data());
            uint8_t* dw; float* dout; CK(cudaMalloc(&dw, w.size())); CK(cudaMalloc(&dout, ref.size() * 4));
            CK(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice));
            const int thr = 256; const int64_t units = nblocks * 8;
            auto fn = K == 4 ? dequant_f32<4> : K == 5 ? dequant_f32<5> : K == 6 ? dequant_f32<6> : dequant_f32<7>;
            fn<<<(int) ((units + thr - 1) / thr), thr>>>(dw, lut, dout, nblocks); CK(cudaGetLastError());
            std::vector<float> got(ref.size()); CK(cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost));
            size_t bad = 0; for (size_t i = 0; i < got.size(); ++i) if (std::memcmp(&got[i], &ref[i], 4) != 0) ++bad;
            double sq = 0, ee = 0; for (int64_t i = 0; i < nenc * 128; ++i) { const double r = got[(size_t) (nblocks - nenc) * 128 + i]; sq += (double) g[i] * g[i]; ee += (r - g[i]) * (r - g[i]); }
            std::printf("dequant: %zu / %zu weights differ from tqenc::decode_blocks (bit exact %s); encoded N(0,1) SQNR %.2f dB (encoder err %.3f)\n",
                        bad, got.size(), bad ? "NO" : "yes", 10 * std::log10(sq / ee), err / (nenc * 128));
            fails += (bad != 0) && !std::getenv("TQE_CODE");   // (the table kernel is meaningless under a computed code)
            // ---- parity 2: GEMV vs double reference on the decoded weights, rows x in small
            for (int nb : NBs) {
                const int rows = 256, in = 768, nblk = in / 128;
                std::vector<uint8_t> mw; fill_random(mw, K, (int64_t) rows * nblk);
                std::vector<float> dec((size_t) rows * in); tqenc::decode_blocks(type, mw.data(), (int64_t) rows * nblk, dec.data());
                uint8_t* mwd; CK(cudaMalloc(&mwd, mw.size())); CK(cudaMemcpy(mwd, mw.data(), mw.size(), cudaMemcpyHostToDevice));
                const KFn kf = pick_k(K, nb, variant); const uint64_t* scd = nullptr; uint8_t* wd = mwd;
                if (kf.tql) { std::vector<uint32_t> words; std::vector<uint64_t> sc; repack_tql(K, mw.data(), rows, nblk, words, sc);
                    CK(cudaMalloc(&wd, words.size() * 4)); CK(cudaMemcpy(wd, words.data(), words.size() * 4, cudaMemcpyHostToDevice));
                    uint64_t* t; CK(cudaMalloc(&t, sc.size() * 8)); CK(cudaMemcpy(t, sc.data(), sc.size() * 8, cudaMemcpyHostToDevice)); scd = t; }
                auto gpu_run = [&](const std::vector<float>& x) {
                    float* xd = dev_f32(x); float* yd; CK(cudaMalloc(&yd, (size_t) nb * rows * 4));
                    float* xj = xd;
                    if (kf.tql) { const size_t n = (size_t) nb * (in / 128) * 32; uint4* xb; CK(cudaMalloc(&xb, n * 16)); prep_b<<<(int) ((n + 255) / 256), 256>>>(xd, in, nb, in, xb); CK(cudaDeviceSynchronize()); xj = (float*) xb; }
                    Job j{wd, xj, yd, rows, in, rows, scd, in}; Job* jd; CK(cudaMalloc(&jd, sizeof(Job))); CK(cudaMemcpy(jd, &j, sizeof(Job), cudaMemcpyHostToDevice));
                    launch(kf, jd, 1, rows, in); CK(cudaGetLastError());
                    std::vector<float> y((size_t) nb * rows); CK(cudaMemcpy(y.data(), yd, y.size() * 4, cudaMemcpyDeviceToHost));
                    cudaFree(xd); cudaFree(yd); cudaFree(jd); if (kf.tql) cudaFree(xj);
                    return y;
                };
                // (1) random F32 activations vs a double reference on the decoded weights
                std::vector<float> x((size_t) nb * in); for (auto& v : x) v = frnd();
                std::vector<double> ref((size_t) nb * rows), refabs((size_t) nb * rows);
                for (int c = 0; c < nb; ++c) for (int r = 0; r < rows; ++r) { double s = 0, sa = 0; for (int i = 0; i < in; ++i) { const double t = (double) dec[(size_t) r * in + i] * x[(size_t) c * in + i]; s += t; sa += std::fabs(t); } ref[(size_t) c * rows + r] = s; refabs[(size_t) c * rows + r] = sa; }
                std::vector<float> y = gpu_run(x);
                double worst = 0; for (size_t i = 0; i < y.size(); ++i) worst = std::max(worst, std::fabs(y[i] - ref[i]) / (refabs[i] + 1e-30));
                const bool ok = worst < 2e-6 || (variant >= 16 && variant <= 18 && worst < 1e-4);
                // (2) the CPU kernel of the repo (tq_cpu scalar and AVX2, Q8_0 activations): same weights; the GPU runs on the dequantised Q8_0 activations
                std::vector<uint8_t> q8((size_t) nb * (in / 32) * 34); std::vector<float> xq((size_t) nb * in);
                for (int c = 0; c < nb; ++c) for (int b = 0; b < in / 32; ++b) {
                    float amax = 0; for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[(size_t) c * in + b * 32 + i]));
                    const uint16_t dh = tqenc::float_to_half(amax / 127.f); const float df = amax / 127.f, id = df ? 1.f / df : 0.f, dr = tqenc::half_to_float(dh);
                    uint8_t* blk = &q8[((size_t) c * (in / 32) + b) * 34]; std::memcpy(blk, &dh, 2);
                    for (int i = 0; i < 32; ++i) { const int qv = (int) std::lround(x[(size_t) c * in + b * 32 + i] * id); blk[2 + i] = (uint8_t) (int8_t) qv; xq[(size_t) c * in + b * 32 + i] = qv * dr; }
                }
                std::vector<float> ycpu_s((size_t) nb * rows), ycpu_a((size_t) nb * rows);
                const bool have_avx2 = strata::kernels::cpu::tq_isa_ok(strata::kernels::cpu::TqIsa::avx2);
                { const void* act[8]; float* os[8]; float* oa[8];
                  for (int c = 0; c < nb; ++c) { act[c] = &q8[(size_t) c * (in / 32) * 34]; os[c] = &ycpu_s[(size_t) c * rows]; oa[c] = &ycpu_a[(size_t) c * rows]; }
                  const size_t row_bytes = (size_t) nblk * bsz;
                  strata::kernels::cpu::tq_rows(strata::kernels::cpu::TqIsa::scalar, type, mw.data(), row_bytes, in, act, nb, os, 0, rows);
                  if (have_avx2) strata::kernels::cpu::tq_rows(strata::kernels::cpu::TqIsa::avx2, type, mw.data(), row_bytes, in, act, nb, oa, 0, rows); }
                std::vector<float> yq = gpu_run(xq);
                std::vector<double> refq((size_t) nb * rows);
                for (int c = 0; c < nb; ++c) for (int r = 0; r < rows; ++r) { double s = 0, sa = 0; for (int i = 0; i < in; ++i) { const double t = (double) dec[(size_t) r * in + i] * xq[(size_t) c * in + i]; s += t; sa += std::fabs(t); } refq[(size_t) c * rows + r] = s; refabs[(size_t) c * rows + r] = sa; }
                double w_gc = 0, w_cr = 0, w_ac = 0;
                for (size_t i = 0; i < yq.size(); ++i) { const double den = refabs[i] + 1e-30; w_gc = std::max(w_gc, std::fabs(yq[i] - ycpu_s[i]) / den); w_cr = std::max(w_cr, std::fabs(ycpu_s[i] - refq[i]) / den); if (have_avx2) w_ac = std::max(w_ac, std::fabs(ycpu_a[i] - ycpu_s[i]) / den); }
                std::printf("gemv nb=%d: GPU vs double ref %.2e %s | vs CPU kernel of the repo (Q8_0 acts): GPU-CPU %.2e, CPU-double %.2e, avx2-scalar %.2e (all / sum|w x|)\n", nb, worst, ok ? "ok" : "FAIL", w_gc, w_cr, w_ac);
                fails += !ok || (w_gc > 5e-6 && !std::getenv("TQE_CODE"));
                cudaFree(mwd); if (kf.tql) { cudaFree(wd); }
            }
            cudaFree(dw); cudaFree(dout);
        }
        if (quick) continue;
        // ---- bench: top-6 experts of a layer, gate/up 2304 x 5120 (12 matrices), down 5120 x 2304 (6 matrices)
        const int E = 6;
        const int rows_gu = 2304, in_gu = 5120, rows_dn = 5120, in_dn = 2304;
        const size_t mat_gu = (size_t) rows_gu * (in_gu / 128) * bsz, mat_dn = (size_t) rows_dn * (in_dn / 128) * bsz;
        std::vector<uint8_t> hw;
        fill_random(hw, K, (int64_t) rows_gu * (in_gu / 128));
        uint8_t* big; CK(cudaMalloc(&big, mat_gu * 2 * E + mat_dn * E));
        // distinct bytes per matrix (so nothing is cached across matrices): reuse one random block with different scales is
        // not needed, the memory system does not see values; upload the same bytes to each slot.
        std::vector<Job> hj_gu, hj_dn;
        const int maxnb = *std::max_element(NBs.begin(), NBs.end());
        std::vector<float> xg((size_t) maxnb * in_gu), xdn((size_t) E * maxnb * in_dn);
        for (auto& v : xg) v = frnd(); for (auto& v : xdn) v = frnd();
        float* xg_d = dev_f32(xg); float* xdn_d = dev_f32(xdn);
        float* yg_d; CK(cudaMalloc(&yg_d, (size_t) 2 * E * maxnb * rows_gu * 4)); float* yd_d; CK(cudaMalloc(&yd_d, (size_t) E * maxnb * rows_dn * 4));
        for (int e = 0; e < E; ++e) {
            for (int m = 0; m < 2; ++m) {
                uint8_t* p = big + (size_t) (e * 2 + m) * mat_gu;
                CK(cudaMemcpy(p, hw.data(), mat_gu, cudaMemcpyHostToDevice));
                hj_gu.push_back(Job{p, xg_d, yg_d + (size_t) (e * 2 + m) * maxnb * rows_gu, rows_gu, in_gu, rows_gu, (const uint64_t*) (p + (size_t) rows_gu * (in_gu / 128) * 4 * K)});
            }
            uint8_t* p = big + 2 * E * mat_gu + (size_t) e * mat_dn;
            std::vector<uint8_t> hd; fill_random(hd, K, (int64_t) rows_dn * (in_dn / 128));
            CK(cudaMemcpy(p, hd.data(), mat_dn, cudaMemcpyHostToDevice));
            hj_dn.push_back(Job{p, xdn_d + (size_t) e * maxnb * in_dn, yd_d + (size_t) e * maxnb * rows_dn, rows_dn, in_dn, rows_dn, (const uint64_t*) (p + (size_t) rows_dn * (in_dn / 128) * 4 * K)});
        }
        Job *jgu, *jdn; CK(cudaMalloc(&jgu, sizeof(Job) * hj_gu.size())); CK(cudaMalloc(&jdn, sizeof(Job) * hj_dn.size()));
        CK(cudaMemcpy(jgu, hj_gu.data(), sizeof(Job) * hj_gu.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(jdn, hj_dn.data(), sizeof(Job) * hj_dn.size(), cudaMemcpyHostToDevice));
        const double bytes_layer = (double) (mat_gu * 2 + mat_dn) * E, weights_layer = (double) (rows_gu * (double) in_gu * 2 + rows_dn * (double) in_dn) * E;
        // flush L2 between reps would only matter for tiny sets: 6 x 3 matrices exceed the 4 MB L2 many times over.
        for (int nb : NBs) {
            KFn f = pick_k(K, nb, variant);
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            Job* jall = nullptr; uint4* xb_gu = nullptr; uint4* xb_dn = nullptr;
            if (f.tql) {   // one launch for all 18 matrices; activations prepared once (prep_b)
                const size_t gu_n = (size_t) nb * (in_gu / 128) * 32, dn_n = (size_t) nb * (in_dn / 128) * 32;
                CK(cudaMalloc(&xb_gu, gu_n * 16)); CK(cudaMalloc(&xb_dn, dn_n * 16 * E));
                prep_b<<<(int) ((gu_n + 255) / 256), 256>>>(xg_d, in_gu, nb, in_gu, xb_gu);
                for (int e = 0; e < E; ++e) prep_b<<<(int) ((dn_n + 255) / 256), 256>>>(xdn_d + (size_t) e * maxnb * in_dn, in_dn, nb, in_dn, xb_dn + e * dn_n);
                std::vector<Job> all = hj_gu; all.insert(all.end(), hj_dn.begin(), hj_dn.end());
                for (size_t i = 0; i < all.size(); ++i) { const bool gu = i < hj_gu.size(); all[i].in = gu ? in_gu : in_dn; all[i].x = (const float*) (gu ? xb_gu : xb_dn + (i - hj_gu.size()) * dn_n); }
                CK(cudaMalloc(&jall, sizeof(Job) * all.size())); CK(cudaMemcpy(jall, all.data(), sizeof(Job) * all.size(), cudaMemcpyHostToDevice));
                CK(cudaDeviceSynchronize());
            }
            auto one = [&]() { if (f.tql) launch(f, jall, 3 * E, rows_dn, 0); else { launch(f, jgu, 2 * E, rows_gu, in_gu); launch(f, jdn, E, rows_dn, in_dn); } };
            for (int w = 0; w < 5; ++w) one();
            CK(cudaDeviceSynchronize()); cudaEventRecord(e0);
            for (int r = 0; r < reps; ++r) one();
            cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
            float ms; cudaEventElapsedTime(&ms, e0, e1); const double us = ms * 1e3 / reps;
            const double gbs = bytes_layer / (us * 1e-6) / 1e9, roof_us = bytes_layer / 448e9 * 1e6;
            std::printf("bench nb=%d: %8.1f us / layer (top-6 gate+up+down)  %6.1f GB/s  %6.2f Gweights/s  roof(448 GB/s) %7.1f us  = %4.1f%% of roof  | 43 layers: %.2f ms\n",
                        nb, us, gbs, weights_layer / (us * 1e-6) / 1e9, roof_us, 100 * roof_us / us, us * 43 / 1e3);
            if (jall) { cudaFree(jall); cudaFree(xb_gu); cudaFree(xb_dn); }
        }
        cudaFree(big); cudaFree(jgu); cudaFree(jdn); cudaFree(xg_d); cudaFree(xdn_d); cudaFree(yg_d); cudaFree(yd_d);
    }
    std::printf("\n%s (%d failures)\n", fails ? "FAILED" : "ALL PARITY OK", fails);
    return fails ? 1 : 0;
}

// tq_cpu.cpp needs these probes from expert_layout.cpp: answer from the CPU itself
namespace strata::kernels::cpu {
bool cpu_avx2_ok() { return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c"); }
bool cpu_avx512_ok() { return false; }
}
