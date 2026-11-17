// src/kernels/latent_u_check.cpp - parity: latent_u_grouped (GPU) vs the engine's per-route U.
//
// Part 1 (existing): latent_u_grouped vs iq_mmvq(8) per-route reference, worst relL2.
// Part 2 (new, knife-0): the SAME Q8_1 activation feeds BOTH the CPU kernel (lat_q8_0_q8_1_rows,
// the current mode-7 path) and the GPU kernel (latent_u_grouped, the new path).  Both use the same
// U weight (latent_dense_q8.bin, layer 0), so the delta is accumulation-order only.  Multi-seed +
// multi-group: reports max/mean/p99 of the per-row delta and per-route relL2 (the quality-gate basis).
#include "strata/core/latent_dense.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/latent.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// scalar replica of expert_source.cpp:4861 lat_quant_q8_1 (per 32: d(f16), s(f16), qs[32])
static void lat_quant_q8_1_scalar(const float* x, int n, uint8_t* o) {
    const int nb = n / 32;
    for (int b = 0; b < nb; ++b) {
        const float* xb = x + b * 32;
        uint8_t* ob = o + b * 36;
        float amax = 0.0f, sum = 0.0f;
        for (int j = 0; j < 32; ++j) { const float v = xb[j]; amax = std::fmax(amax, std::fabs(v)); sum += v; }
        const float d32 = amax / 127.0f;
        const float d = std::fabs(d32) > 65504.0f ? std::copysign(65504.0f, d32) : d32;
        const uint16_t dh = strata::kernels::f16_from_f32(d);
        const uint16_t sh = strata::kernels::f16_from_f32(sum);
        ob[0] = (uint8_t)(dh & 0xFFu); ob[1] = (uint8_t)(dh >> 8);
        ob[2] = (uint8_t)(sh & 0xFFu); ob[3] = (uint8_t)(sh >> 8);
        int8_t* q = (int8_t*)(ob + 4);
        for (int j = 0; j < 32; ++j) {
            float v = amax == 0.0f ? 0.0f : std::roundf(xb[j] / d);
            v = v > 127.0f ? 127.0f : (v < -127.0f ? -127.0f : v);
            q[j] = (int8_t) v;
        }
    }
}

int main() {
    const int G = 8, E = 512, cap = 10, ELL = 1792, H = 2560;
    size_t urow = (size_t) (ELL / 32) * 34;   // Q4DU: overwritten from the pack below
    const size_t xlq_row = (size_t) (ELL / 32) * 36;
    
    const char* dense_path = std::getenv("STRATA_LATENT_DENSE");
    const char* labels_path = std::getenv("STRATA_LATENT_LABELS");
    if (dense_path == nullptr || dense_path[0] == 0 || labels_path == nullptr || labels_path[0] == 0) {
        std::fprintf(stderr, "usage: STRATA_LATENT_DENSE=<latent_dense_q8.bin> STRATA_LATENT_LABELS=<labels.bin> latent_u_check\n");
        return 2;
    }
    strata::core::LatentDense dense;
    std::string err;
    if (!dense.open(dense_path, 48, G, H, ELL, err)) { std::fprintf(stderr, "dense: %s\n", err.c_str()); return 2; }
    urow = dense.u_row_bytes();   // Q4DU: 34 (Q8_0) or 18 (Q4_0) per block, from the pack
    const int fmt = dense.fmt();
    std::fprintf(stderr, "latent_u_check: fmt=%d urow=%zu\n", fmt, urow);
    if (dense.u_bytes() != (size_t) G * H * urow) { std::fprintf(stderr, "u_bytes %zu != %zu\n", dense.u_bytes(), (size_t) G * H * urow); return 2; }
    std::vector<int32_t> labels((size_t) 48 * E);
    {
        std::FILE* f = std::fopen(labels_path, "rb");
        if (f == nullptr) { std::perror("labels"); return 2; }
        const size_t got = std::fread(labels.data(), 4, labels.size(), f);
        std::fclose(f);
        if (got != labels.size()) { std::fprintf(stderr, "labels short\n"); return 2; }
    }

    // device allocs (shared by both parts)
    float *out_k = nullptr, *out_ref = nullptr, *yl_d = nullptr;
    uint8_t *u_d = nullptr, *ylq = nullptr, *uq = nullptr, *hq_d = nullptr;
    int32_t *ids_d = nullptr, *labels_d = nullptr;
    float *out_cpu = nullptr, *out_gpu = nullptr;
    const auto chk = [&](const char* what, cudaError_t e) {
        if (e == cudaSuccess) return true;
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        return false;
    };
    // Q4DU-fix: HARD-FAIL semantics.  A kernel launch or stream error (e.g. an illegal address
    // left in the stream by a wrong row width) must abort with a clear message - never keep
    // computing and print a misleading relL2 (the "false green" failure mode).
    const auto gpu_check = [&](const char* what) {
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "latent_u_check: FATAL %s: %s\n", what, cudaGetErrorString(e));
            std::exit(1);
        }
    };
    // Q4DU-fix: the outputs must not be all-zero or NaN (a dead stream produces zeros).
    const auto out_guard = [&](const char* what, const float* v, size_t n) {
        bool all_zero = true;
        for (size_t i = 0; i < n; ++i) {
            if (!std::isfinite(v[i])) { std::fprintf(stderr, "latent_u_check: FATAL %s: NaN/Inf at %zu\n", what, i); std::exit(1); }
            if (v[i] != 0.0f) all_zero = false;
        }
        if (all_zero) { std::fprintf(stderr, "latent_u_check: FATAL %s: all-zero output (dead stream?)\n", what); std::exit(1); }
    };
    bool ok = chk("malloc u", cudaMalloc((void**) &u_d, (size_t) G * H * urow)) &&
              chk("malloc yl", cudaMalloc((void**) &yl_d, (size_t) cap * ELL * 4)) &&
              chk("malloc ylq", cudaMalloc((void**) &ylq, (size_t) cap * xlq_row)) &&
              chk("malloc uq", cudaMalloc((void**) &uq, xlq_row)) &&
              chk("malloc hq", cudaMalloc((void**) &hq_d, (size_t) cap * xlq_row)) &&
              chk("malloc out_k", cudaMalloc((void**) &out_k, (size_t) cap * H * 4)) &&
              chk("malloc out_ref", cudaMalloc((void**) &out_ref, (size_t) cap * H * 4)) &&
              chk("malloc out_gpu", cudaMalloc((void**) &out_gpu, (size_t) cap * H * 4)) &&
              chk("malloc ids", cudaMalloc((void**) &ids_d, (size_t) cap * 4)) &&
              chk("malloc labels", cudaMalloc((void**) &labels_d, (size_t) 48 * E * 4));
    if (!ok) return 2;
    cudaMemcpy(u_d, dense.u(0), (size_t) G * H * urow, cudaMemcpyHostToDevice);
    cudaMemcpy(labels_d, labels.data(), (size_t) 48 * E * 4, cudaMemcpyHostToDevice);

    cudaStream_t cs = nullptr;
    cudaStreamCreate(&cs);

    const uint8_t* u_host = dense.u(0);

    // ---- Part 1 (existing): GPU latent_u_grouped vs iq_mmvq(8) reference ----
    {
        std::vector<float> h_yl((size_t) cap * ELL);
        std::vector<int32_t> h_ids(cap);
        uint32_t s = 1;
        auto rnd = [&s]() { s = s * 1664525u + 1013904223u; return s; };
        for (float& v : h_yl) v = ((int) (rnd() % 2001) - 1000) / 1000.0f;
        for (int r = 0; r < cap; ++r) h_ids[(size_t) r] = (int) (rnd() % 512);

        cudaMemcpy(yl_d, h_yl.data(), (size_t) cap * ELL * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(ids_d, h_ids.data(), (size_t) cap * 4, cudaMemcpyHostToDevice);

        strata::kernels::quantize_q8_1_rows(yl_d, cap, ELL, ylq, cs);
        strata::kernels::latent_u_grouped(u_d, ylq, ids_d, labels_d, E, cap, ELL, H, out_k, cs, fmt);   // Q4DU
        gpu_check("part1 grouped launch");
        for (int r = 0; r < cap; ++r) {
            const int g = labels[(size_t) h_ids[(size_t) r]];
            cudaMemcpyAsync(uq, ylq + (size_t) r * xlq_row, xlq_row, cudaMemcpyDeviceToDevice, cs);
            strata::kernels::iq_mmvq(strata::kernels::lat_dense_fmt_to_type(fmt), u_d + (size_t) g * (size_t) H * urow, uq, out_ref + (size_t) r * H, ELL, H, 1, cs);   // Q4DU
        }
        const cudaError_t se1 = cudaStreamSynchronize(cs);
        if (se1 != cudaSuccess) { std::fprintf(stderr, "latent_u_check: FATAL part1 sync: %s\n", cudaGetErrorString(se1)); std::exit(1); }
        std::vector<float> h_k((size_t) cap * H), h_r((size_t) cap * H);
        cudaMemcpy(h_k.data(), out_k, (size_t) cap * H * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_r.data(), out_ref, (size_t) cap * H * 4, cudaMemcpyDeviceToHost);
        out_guard("part1 grouped output", h_k.data(), h_k.size());
        out_guard("part1 mmvq reference", h_r.data(), h_r.size());
        double worst = 0;
        for (int r = 0; r < cap; ++r) {
            double num = 0, den = 0;
            for (int h = 0; h < H; ++h) {
                const double d = (double) h_k[(size_t) r * H + h] - h_r[(size_t) r * H + h];
                num += d * d; den += (double) h_r[(size_t) r * H + h] * h_r[(size_t) r * H + h];
            }
            worst = std::max(worst, std::sqrt(num / std::max(den, 1e-30)));
        }
        std::fprintf(stderr, "part1 GPU-vs-iq_mmvq worst relL2 = %g\n", worst);
    }

    // ---- Part 2 (new, knife-0): CPU lat_q8_0_q8_1_rows vs GPU latent_u_grouped, same hq, multi-seed ----
    {
        const int n_seeds = 3;
        std::vector<double> row_delta;      // |cpu - gpu| per output row
        std::vector<double> route_rel;      // per-route relL2
        for (int seed = 0; seed < n_seeds; ++seed) {
            std::vector<float> h_yl((size_t) cap * ELL);
            std::vector<int32_t> h_ids(cap);
            uint32_t s = (uint32_t) (0x9e3779b9u ^ ((uint32_t) seed * 0x1000003u));
            auto rnd = [&s]() { s = s * 1664525u + 1013904223u; return s; };
            for (float& v : h_yl) v = ((int) (rnd() % 2001) - 1000) / 1000.0f;
            for (int r = 0; r < cap; ++r) h_ids[(size_t) r] = (int) (rnd() % 512);

            // CPU hq quantization (the engine's lat_quant_q8_1 scalar replica)
            std::vector<uint8_t> h_hq((size_t) cap * xlq_row);
            for (int r = 0; r < cap; ++r)
                lat_quant_q8_1_scalar(h_yl.data() + (size_t) r * ELL, ELL, h_hq.data() + (size_t) r * xlq_row);

            // CPU path: per route, the group g = labels[ids[r]], U[g] . hq[r]
            std::vector<float> h_cpu((size_t) cap * H);
            for (int r = 0; r < cap; ++r) {
                const int g = labels[(size_t) h_ids[(size_t) r]];
                const void* act_arr[1] = { h_hq.data() + (size_t) r * xlq_row };
                float* out_arr[1] = { h_cpu.data() + (size_t) r * H };
                if (fmt == 4)
                    strata::kernels::cpu::lat_q4_0_q8_1_rows(u_host + (size_t) g * (size_t) H * urow, urow, ELL,
                                                             act_arr, 1, out_arr, 0, H);
                else
                    strata::kernels::cpu::lat_q8_0_q8_1_rows(u_host + (size_t) g * (size_t) H * urow, urow, ELL,
                                                             act_arr, 1, out_arr, 0, H);
            }

            // GPU path: upload the SAME hq, latent_u_grouped
            cudaMemcpy(hq_d, h_hq.data(), (size_t) cap * xlq_row, cudaMemcpyHostToDevice);
            cudaMemcpy(ids_d, h_ids.data(), (size_t) cap * 4, cudaMemcpyHostToDevice);
            strata::kernels::latent_u_grouped(u_d, hq_d, ids_d, labels_d, E, cap, ELL, H, out_gpu, cs, fmt);
            gpu_check("part2 grouped launch");
            const cudaError_t se2 = cudaStreamSynchronize(cs);
            if (se2 != cudaSuccess) { std::fprintf(stderr, "latent_u_check: FATAL part2 sync: %s\n", cudaGetErrorString(se2)); std::exit(1); }
            std::vector<float> h_gpu((size_t) cap * H);
            cudaMemcpy(h_gpu.data(), out_gpu, (size_t) cap * H * 4, cudaMemcpyDeviceToHost);
            out_guard("part2 gpu output", h_gpu.data(), h_gpu.size());
            out_guard("part2 cpu output", h_cpu.data(), h_cpu.size());
            for (int r = 0; r < cap; ++r) {
                double num = 0, den = 0;
                for (int h = 0; h < H; ++h) {
                    const double d = (double) h_cpu[(size_t) r * H + h] - h_gpu[(size_t) r * H + h];
                    row_delta.push_back(std::fabs(d));
                    num += d * d; den += (double) h_gpu[(size_t) r * H + h] * h_gpu[(size_t) r * H + h];
                }
                route_rel.push_back(std::sqrt(num / std::max(den, 1e-30)));
            }
        }

        std::sort(row_delta.begin(), row_delta.end());
        std::sort(route_rel.begin(), route_rel.end());
        const size_t nrow = row_delta.size(), nrel = route_rel.size();
        double dsum = 0; for (double v : row_delta) dsum += v;
        double rsum = 0; for (double v : route_rel) rsum += v;
        std::fprintf(stderr, "part2 CPU-vs-GPU (seeds=%d, rows=%zu):\n", n_seeds, nrow);
        std::fprintf(stderr, "  abs delta max %.3e mean %.3e p99 %.3e\n",
                     row_delta.back(), dsum / (double) nrow, row_delta[(size_t) (0.99 * (nrow - 1))]);
        std::fprintf(stderr, "  relL2   max %.3e mean %.3e p99 %.3e\n",
                     route_rel.back(), rsum / (double) nrel, route_rel[(size_t) (0.99 * (nrel - 1))]);
        const bool pass = route_rel.back() < 1e-4;
        std::fprintf(stderr, "%s\n", pass ? "RESULT: ok" : "RESULT: FAIL");
        return pass ? 0 : 1;
    }
}
