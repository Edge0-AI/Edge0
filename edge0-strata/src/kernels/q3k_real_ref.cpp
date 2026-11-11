// q3k_real_ref.cpp - CPU-official vs GPU: the Q3_K gate/up (and full native_expert_grouped) on a REAL
// v7_l1280_q3k expert blob.  The reference uses the OFFICIAL ggml dequant (computed on HPC-1 by
// iq3tool_q3k, linked ggml-quants.c) for the weights, and the engine's own quantize_q8_1_rows for the
// activations - so only the Q3_K (gu) / IQ4_NL (dn) GPU dot products are under test.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace k = strata::kernels;

static void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA %s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}

static std::vector<float> readf(const char* p, size_t n) {
    std::vector<float> v(n);
    FILE* f = std::fopen(p, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", p); std::exit(2); }
    std::fread(v.data(), 4, n, f);
    std::fclose(f);
    return v;
}

// q8_1 blocks (fp16 d, fp16 s, 32 int8) back to doubles
static std::vector<double> dequant_q8_1(const std::vector<uint8_t>& q, size_t n) {
    std::vector<double> v(n);
    for (size_t b = 0; b < n / 32; ++b) {
        uint16_t dh;
        std::memcpy(&dh, &q[b * 36], 2);
        const double d = (double) k::f32_from_f16(dh);
        for (int i = 0; i < 32; ++i) v[b * 32 + i] = d * (int8_t) q[b * 36 + 4 + i];
    }
    return v;
}

int main() {
    const int64_t H = 1280, FF = 640;   // expert_embd 1280, expert_ff 640
    const int gu_type = 11, dn_type = 20;
    const size_t gu_bytes = 704000, dn_bytes = 460800, blob_bytes = gu_bytes + dn_bytes;

    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");

    // ---- official ggml dequant (produced on HPC-1 by iq3tool_q3k dequant) ----
    const std::vector<float> wf_gu = readf("/home/user/ref_l0e0_gu.f32", (size_t)(2 * FF) * H);   // 1280x1280
    const std::vector<float> wf_dn = readf("/home/user/ref_l0e0_dn.f32", (size_t) H * FF);         // 1280x640

    // ---- real Q3_K expert blob (layer 0, expert 0) ----
    std::vector<uint8_t> blob(blob_bytes);
    {
        FILE* f = std::fopen("/home/user/models/v7_l1280_q3k_pack/latent_a_pack/experts.bin", "rb");
        if (!f) { std::fprintf(stderr, "cannot open experts.bin\n"); return 2; }
        std::fread(blob.data(), 1, blob_bytes, f);
        std::fclose(f);
    }

    // ---- a small deterministic activation (NOT lossless; the reference reuses the engine's q8_1) ----
    std::vector<float> x((size_t) H);
    for (int j = 0; j < H; ++j) x[(size_t) j] = (float)(0.6 * std::sin(0.13 * (double) j) + 0.2 * std::cos(0.07 * (double) j));

    // ---- device side ----
    uint8_t* dblob = nullptr; ck(cudaMalloc((void**) &dblob, blob_bytes), "blob");
    ck(cudaMemcpy(dblob, blob.data(), blob_bytes, cudaMemcpyHostToDevice), "h2d blob");

    // ---- GPU dequant vs OFFICIAL dequant (localizes dequant vs dot vs down) ----
    {
        float* gpu_gu = nullptr; ck(cudaMalloc((void**) &gpu_gu, (size_t)(2 * FF) * H * 4), "gpu_gu");
        k::iq_dequant_f32(11, dblob, (int64_t)(2 * FF) * H, gpu_gu, s);
        std::vector<float> gg((size_t)(2 * FF) * H);
        ck(cudaMemcpy(gg.data(), gpu_gu, gg.size() * 4, cudaMemcpyDeviceToHost), "d2h gpu_gu");
        double mg = 0, mg_rel = 0; int bg = 0;
        for (size_t i = 0; i < gg.size(); ++i) {
            double d = std::fabs((double) gg[i] - (double) wf_gu[i]);
            mg = std::max(mg, d); mg_rel += d / (1.0 + std::fabs((double) wf_gu[i]));
            if (d > 1e-3 * (1.0 + std::fabs((double) wf_gu[i]))) ++bg;
        }
        std::printf("  [gu dequant] max_abs %.6f mean_rel %.2e bad %zu/%zu\n", mg, mg_rel / gg.size(), (size_t) bg, gg.size());

        float* gpu_dn = nullptr; ck(cudaMalloc((void**) &gpu_dn, (size_t) H * FF * 4), "gpu_dn");
        k::iq_dequant_f32(20, dblob + gu_bytes, (int64_t) H * FF, gpu_dn, s);
        std::vector<float> gd((size_t) H * FF);
        ck(cudaMemcpy(gd.data(), gpu_dn, gd.size() * 4, cudaMemcpyDeviceToHost), "d2h gpu_dn");
        double md = 0, md_rel = 0; int bd = 0;
        for (size_t i = 0; i < gd.size(); ++i) {
            double d = std::fabs((double) gd[i] - (double) wf_dn[i]);
            md = std::max(md, d); md_rel += d / (1.0 + std::fabs((double) wf_dn[i]));
            if (d > 1e-3 * (1.0 + std::fabs((double) wf_dn[i]))) ++bd;
        }
        std::printf("  [dn dequant] max_abs %.6f mean_rel %.2e bad %zu/%zu\n", md, md_rel / gd.size(), (size_t) bd, gd.size());
        cudaFree(gpu_gu); cudaFree(gpu_dn);
    }
    unsigned long long* dptr = nullptr; ck(cudaMalloc((void**) &dptr, 8), "ptr");
    unsigned long long ptr = (unsigned long long) dblob;
    ck(cudaMemcpy(dptr, &ptr, 8, cudaMemcpyHostToDevice), "h2d ptr");
    int32_t *dstart = nullptr, *dn_g = nullptr, *ddst = nullptr, *dtok = nullptr;
    ck(cudaMalloc((void**) &dstart, 8), "start"); ck(cudaMalloc((void**) &dn_g, 4), "ng");
    ck(cudaMalloc((void**) &ddst, 4), "dst"); ck(cudaMalloc((void**) &dtok, 4), "tok");
    int32_t start[2] = {0, 1}, ng = 1, dst = 0, tok = 0;
    ck(cudaMemcpy(dstart, start, 8, cudaMemcpyHostToDevice), "h2d start");
    ck(cudaMemcpy(dn_g, &ng, 4, cudaMemcpyHostToDevice), "h2d ng");
    ck(cudaMemcpy(ddst, &dst, 4, cudaMemcpyHostToDevice), "h2d dst");
    ck(cudaMemcpy(dtok, &tok, 4, cudaMemcpyHostToDevice), "h2d tok");

    float* dx = nullptr; ck(cudaMalloc((void**) &dx, (size_t) H * 4), "x");
    ck(cudaMemcpy(dx, x.data(), (size_t) H * 4, cudaMemcpyHostToDevice), "h2d x");
    const size_t xq_bytes = (size_t)(H / 32) * 36;
    uint8_t* dxq = nullptr; ck(cudaMalloc((void**) &dxq, xq_bytes), "xq");
    k::quantize_q8_1_rows(dx, 1, H, dxq, s);

    const k::NativeExpertLayout L = k::native_expert_layout(gu_type, dn_type, H, FF);
    uint8_t* dscr = nullptr; ck(cudaMalloc((void**) &dscr, k::native_expert_scratch_bytes(1, FF)), "scratch");
    float* dout = nullptr; ck(cudaMalloc((void**) &dout, (size_t) H * 4), "out");
    k::native_expert_grouped(L, dptr, dstart, dn_g, ddst, dtok, 1, 1, dxq, dscr, dout, s);
    ck(cudaStreamSynchronize(s), "sync");

    std::vector<float> out_gpu((size_t) H);
    ck(cudaMemcpy(out_gpu.data(), dout, (size_t) H * 4, cudaMemcpyDeviceToHost), "d2h out");

    // ---- reference (official weights x engine q8_1) ----
    std::vector<uint8_t> xq(xq_bytes);
    ck(cudaMemcpy(xq.data(), dxq, xq_bytes, cudaMemcpyDeviceToHost), "d2h xq");
    const std::vector<double> xd = dequant_q8_1(xq, (size_t) H);

    std::vector<double> gate((size_t) FF), up((size_t) FF);
    for (int r = 0; r < FF; ++r) {
        double g = 0, u = 0;
        for (int c = 0; c < H; ++c) {
            g += (double) wf_gu[(size_t) r * H + c] * xd[(size_t) c];
            u += (double) wf_gu[((size_t) FF + r) * H + c] * xd[(size_t) c];
        }
        gate[(size_t) r] = g; up[(size_t) r] = u;
    }

    std::vector<float> hf((size_t) FF);
    for (int r = 0; r < FF; ++r) {
        const double silu = gate[(size_t) r] / (1.0 + std::exp(-gate[(size_t) r]));
        hf[(size_t) r] = (float)(silu * up[(size_t) r]);
    }
    float* dh = nullptr; ck(cudaMalloc((void**) &dh, (size_t) FF * 4), "h");
    ck(cudaMemcpy(dh, hf.data(), (size_t) FF * 4, cudaMemcpyHostToDevice), "h2d h");
    const size_t hq_bytes = (size_t)(FF / 32) * 36;
    uint8_t* dhq = nullptr; ck(cudaMalloc((void**) &dhq, hq_bytes), "hq");
    k::quantize_q8_1_rows(dh, 1, FF, dhq, s);
    std::vector<uint8_t> hq(hq_bytes);
    ck(cudaMemcpy(hq.data(), dhq, hq_bytes, cudaMemcpyDeviceToHost), "d2h hq");
    const std::vector<double> hd = dequant_q8_1(hq, (size_t) FF);

    std::vector<double> out_ref((size_t) H);
    for (int r = 0; r < H; ++r) {
        double o = 0;
        for (int c = 0; c < FF; ++c) o += (double) wf_dn[(size_t) r * FF + c] * hd[(size_t) c];
        out_ref[(size_t) r] = o;
    }

    // ---- compare ----
    double max_abs = 0, sum_abs = 0, sum_ref = 0;
    int bad = 0;
    for (int r = 0; r < H; ++r) {
        const double d = std::fabs((double) out_gpu[(size_t) r] - out_ref[(size_t) r]);
        max_abs = std::max(max_abs, d);
        sum_abs += d;
        sum_ref += std::fabs(out_ref[(size_t) r]);
        if (d > 1e-3 * (1.0 + std::fabs(out_ref[(size_t) r]))) ++bad;
    }
    std::printf("q3k_real_ref: H=%lld FF=%lld\n", (long long) H, (long long) FF);
    std::printf("  gate rms %.4f  up rms %.4f  h rms %.4f\n",
                std::sqrt([&]{double s2=0;for(double v:gate)s2+=v*v;return s2/(double)FF;}( )),
                std::sqrt([&]{double s2=0;for(double v:up)s2+=v*v;return s2/(double)FF;}( )),
                std::sqrt([&]{double s2=0;for(double v:out_ref)s2+=v*v;return s2/(double)H;}( )));
    std::printf("  out: gpu[0..4] = %.5f %.5f %.5f %.5f %.5f\n", out_gpu[0], out_gpu[1], out_gpu[2], out_gpu[3], out_gpu[4]);
    std::printf("  out: ref[0..4] = %.5f %.5f %.5f %.5f %.5f\n", out_ref[0], out_ref[1], out_ref[2], out_ref[3], out_ref[4]);
    std::printf("  max_abs_diff %.6f  mean_abs_diff %.6f  mean_ref %.6f  bad_rows %d/%d\n",
                max_abs, sum_abs / (double) H, sum_ref / (double) H, bad, (int) H);
    const bool ok = max_abs < 1e-3 * (1.0 + sum_ref / (double) H);
    std::printf("q3k_real_ref: %s\n", ok ? "MATCH (GPU == official reference)" : "MISMATCH (GPU differs from official)");

    cudaFree(dblob); cudaFree(dptr); cudaFree(dstart); cudaFree(dn_g); cudaFree(ddst); cudaFree(dtok);
    cudaFree(dx); cudaFree(dxq); cudaFree(dscr); cudaFree(dout); cudaFree(dh); cudaFree(dhq);
    cudaStreamDestroy(s);
    return ok ? 0 : 1;
}
