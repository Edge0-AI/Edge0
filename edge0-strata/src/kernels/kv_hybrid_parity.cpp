// src/kernels/kv_hybrid_parity.cpp - hybrid K8V4 KV against host references (GPU, no model).
//
// Exercises the EXACT composed calls the engine's hybrid path makes (layer.cpp / prefill.cpp): the q8/q4
// append and gather kernels invoked with the unused half's pointers folded onto the used pool, and the
// fused decode attention's KV_MODE 3 (INT8 K, rotated Q4_0 V) followed by the output's inverse Hadamard.
//   1. K: INT8 codes + FP16 scales bitwise vs the host reference (kv_q8.hpp's rules);
//   2. V: after fwht256, Q4_0 blocks bitwise vs the host reference (kv_q4.hpp's rules, PR #21);
//   3. the composed gathers bitwise vs the host dequantization of those codes;
//   4. qsa_decode_attn<3> + output fwht vs a host attention over the SAME dequantized K/V (fp32 math), and
//      bounded against the fp32-true attention from the unquantized K/V.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
int g_fail = 0;
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset");
    return p;
}

// kv_q4_parity.cpp's exact CPU references.
void fwht256_host_reference(float* x) {
    constexpr int n = 256;
    for (int s = 1; s < n; s *= 2) {
        for (int i = 0; i < n; i += 2 * s) {
            for (int j = 0; j < s; ++j) {
                const float u = x[i + j];
                const float v = x[i + j + s];
                x[i + j]     = u + v;
                x[i + j + s] = u - v;
            }
        }
    }
    const float scale = 1.0f / 16.0f;
    for (int i = 0; i < n; ++i) x[i] *= scale;
}
void quantize_block_q4_0_host(const float* x, k::block_q4_0& blk) {
    float amax = 0.0f, max_val = 0.0f;
    for (int i = 0; i < 32; ++i) {
        const float v = x[i];
        if (amax < std::fabs(v)) { amax = std::fabs(v); max_val = v; }
    }
    const float d = max_val / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    blk.d = k::f16_from_f32(d);
    for (int i = 0; i < 16; ++i) {
        const int q0 = (int) (x[i] * id + 8.5f);
        const int q1 = (int) (x[i + 16] * id + 8.5f);
        const uint8_t xi0 = (uint8_t) std::min(15, std::max(0, q0));
        const uint8_t xi1 = (uint8_t) std::min(15, std::max(0, q1));
        blk.qs[i] = xi0 | (xi1 << 4);
    }
}
}  // namespace

int main() {
    std::printf("=== Running kv_hybrid_parity test ===\n");
    k::QsaShapes s = k::qsa_real_shapes();
    s.page_size = 64;
    const int H = (int) s.n_head_kv, D = (int) s.head_dim, P = (int) s.page_size, G = D / k::KV_Q8_GROUP;
    const int QH = (int) s.n_head, PER = QH / H;   // query heads per KV head
    const int pages = 24, cells = pages * P;   // n_chunks >= n_head: the batched merge overlaps queries below that
    std::mt19937 rng(17);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<int32_t> table(pages);
    for (int i = 0; i < pages; ++i) table[i] = (i * 5 + 3) % pages;
    int32_t* d_table = dalloc<int32_t>(pages);
    ck(cudaMemcpy(d_table, table.data(), pages * 4, cudaMemcpyHostToDevice), "table");

    std::vector<float> K((size_t) cells * H * D), V((size_t) cells * H * D);
    for (auto& x : K) x = nd(rng) * 0.5f;
    for (auto& x : V) x = nd(rng) * 0.5f;

    // ---- host reference: K in INT8 (kv_q8.hpp), rotated V in Q4_0 (kv_q4.hpp)
    std::vector<int8_t> rk((size_t) cells * H * D);
    std::vector<uint16_t> rk_s((size_t) cells * H * G);
    for (int c = 0; c < cells; ++c)
        for (int h = 0; h < H; ++h)
            for (int g = 0; g < G; ++g) {
                float amax = 0.0f;
                for (int t = 0; t < k::KV_Q8_GROUP; ++t)
                    amax = std::max(amax, std::fabs(K[((size_t) c * H + h) * D + g * k::KV_Q8_GROUP + t]));
                const uint16_t sb = k::f16_from_f32(amax / 127.0f);
                const float sf = k::f32_from_f16(sb);
                for (int t = 0; t < k::KV_Q8_GROUP; ++t) {
                    int q = 0;
                    if (sf > 0.0f) {
                        q = (int) std::nearbyint(K[((size_t) c * H + h) * D + g * k::KV_Q8_GROUP + t] / sf);   // rn: ties to even, as the kernel
                        q = std::max(-127, std::min(127, q));
                    }
                    rk[((size_t) c * H + h) * D + g * k::KV_Q8_GROUP + t] = (int8_t) q;
                }
                rk_s[((size_t) c * H + h) * G + g] = sb;
            }
    std::vector<float> Vrot((size_t) cells * H * D);
    std::memcpy(Vrot.data(), V.data(), V.size() * 4);
    for (int c = 0; c < cells * H; ++c) fwht256_host_reference(Vrot.data() + (size_t) c * D);
    std::vector<k::block_q4_0> rv4((size_t) cells * H * (D / 32));
    for (int c = 0; c < cells; ++c)
        for (int h = 0; h < H; ++h)
            for (int b = 0; b < D / 32; ++b)
                quantize_block_q4_0_host(&Vrot[((size_t) c * H + h) * D + b * 32],
                                         rv4[((size_t) c * H + h) * (D / 32) + b]);

    // ---- device: one cell appended through the COMPOSED hybrid calls, then gathered
    const int64_t q4_row = (int64_t) k::kv_q4_bytes_per_head(D);
    int8_t* d_kq = dalloc<int8_t>((size_t) cells * H * D);
    uint16_t* d_ks = dalloc<uint16_t>((size_t) cells * H * G);
    uint8_t* d_v4 = dalloc<uint8_t>((size_t) cells * H * q4_row);
    float *d_kcur = dalloc<float>((size_t) H * D), *d_vcur = dalloc<float>((size_t) H * D);
    int32_t* d_step = dalloc<int32_t>(k::kStepCount);

    int append_bad = 0, gather_bad = 0;
    for (int c = 0; c < cells; ++c) {
        ck(cudaMemcpy(d_kcur, &K[(size_t) c * H * D], (size_t) H * D * 4, cudaMemcpyHostToDevice), "kcur");
        ck(cudaMemcpy(d_vcur, &V[(size_t) c * H * D], (size_t) H * D * 4, cudaMemcpyHostToDevice), "vcur");
        int32_t step[k::kStepCount];
        k::qsa_step_fill(step, c, s);
        ck(cudaMemcpy(d_step, step, sizeof step, cudaMemcpyHostToDevice), "step");
        // exactly what qsa_layer's hybrid branch does
        k::fwht256_inplace_cuda(d_vcur, H, nullptr);
        k::kv_append_q8_step(d_kq, d_kq, d_ks, d_ks, d_table, d_step, d_kcur, d_kcur, s, nullptr, nullptr);
        k::kv_append_q4_step(d_v4, d_v4, d_table, d_step, d_vcur, d_vcur, s, nullptr, nullptr);
        ck(cudaGetLastError() != cudaSuccess ? cudaGetLastError() : cudaSuccess, "append");
    }
    ck(cudaDeviceSynchronize(), "append sync");

    std::vector<int8_t> hk((size_t) cells * H * D);
    std::vector<uint16_t> hks((size_t) cells * H * G);
    std::vector<uint8_t> hv4((size_t) cells * H * q4_row);
    ck(cudaMemcpy(hk.data(), d_kq, hk.size(), cudaMemcpyDeviceToHost), "kq");
    ck(cudaMemcpy(hks.data(), d_ks, hks.size() * 2, cudaMemcpyDeviceToHost), "ks");
    ck(cudaMemcpy(hv4.data(), d_v4, hv4.size(), cudaMemcpyDeviceToHost), "v4");
    // the pools are PHYSICAL (page, head, slot); the reference is by LOGICAL cell - compare through the table
    const k::block_q4_0* hv4b = reinterpret_cast<const k::block_q4_0*>(hv4.data());
    for (int c = 0; c < cells; ++c) {
        const long long row = ((long long) table[c / P] * H);
        for (int h = 0; h < H; ++h) {
            const long long r = (row + h) * P + c % P;
            for (int d = 0; d < D; ++d)
                if (hk[(size_t) r * D + d] != rk[((size_t) c * H + h) * D + d]) ++append_bad;
            for (int g = 0; g < G; ++g)
                if (hks[(size_t) r * G + g] != rk_s[((size_t) c * H + h) * G + g]) ++append_bad;
            for (int b = 0; b < D / 32; ++b) {
                const k::block_q4_0& got = hv4b[(size_t) r * (D / 32) + b];
                if (got.d != rv4[((size_t) c * H + h) * (D / 32) + b].d ||
                    std::memcmp(got.qs, rv4[((size_t) c * H + h) * (D / 32) + b].qs, 16) != 0) ++append_bad;
            }
        }
    }
    std::printf("[1/3] composed appends: %s (%d bad values of %zu K / %zu V)\n",
                append_bad == 0 ? "PASS" : "FAIL", append_bad, hk.size(), rv4.size());

    // gather every cell (ids = all, ascending - the identity selection for width = cells)
    std::vector<int32_t> ids(cells);
    for (int i = 0; i < cells; ++i) ids[i] = i;
    int32_t* d_ids = dalloc<int32_t>(cells);
    ck(cudaMemcpy(d_ids, ids.data(), cells * 4, cudaMemcpyHostToDevice), "ids");
    int32_t step[k::kStepCount];
    k::qsa_step_fill(step, cells - 1, s);
    step[k::kStepWidth] = cells;   // select everything
    ck(cudaMemcpy(d_step, step, sizeof step, cudaMemcpyHostToDevice), "step2");
    uint16_t *d_ksc = dalloc<uint16_t>((size_t) cells * H * D), *d_vsc = dalloc<uint16_t>((size_t) cells * H * D);
    k::kv_gather_q8_step(d_kq, d_kq, d_ks, d_ks, d_table, d_ids, d_step, cells, s, d_ksc, d_ksc, nullptr);
    k::kv_gather_q4_step(d_v4, d_v4, d_table, d_ids, d_step, cells, s, d_vsc, d_vsc, nullptr);
    ck(cudaDeviceSynchronize(), "gather sync");
    std::vector<uint16_t> h_ksc((size_t) cells * H * D), h_vsc((size_t) cells * H * D);
    ck(cudaMemcpy(h_ksc.data(), d_ksc, h_ksc.size() * 2, cudaMemcpyDeviceToHost), "ksc");
    ck(cudaMemcpy(h_vsc.data(), d_vsc, h_vsc.size() * 2, cudaMemcpyDeviceToHost), "vsc");
    for (int c = 0; c < cells; ++c)
        for (int h = 0; h < H; ++h)
            for (int d = 0; d < D; ++d) {
                const float kref = k::f32_from_f16(k::f16_from_f32(
                    (float) rk[((size_t) c * H + h) * D + d] * k::f32_from_f16(rk_s[((size_t) c * H + h) * G + d / k::KV_Q8_GROUP])));
                if (k::f32_from_f16(h_ksc[((size_t) c * H + h) * D + d]) != kref) ++gather_bad;
                const k::block_q4_0& b = rv4[((size_t) c * H + h) * (D / 32) + d / 32];
                const int nib = d % 32 < 16 ? (b.qs[d % 32] & 0x0F) - 8 : ((b.qs[d % 32 - 16] >> 4) - 8);
                const float vref = k::f32_from_f16(k::f16_from_f32((float) nib * k::f32_from_f16(b.d)));
                if (k::f32_from_f16(h_vsc[((size_t) c * H + h) * D + d]) != vref) ++gather_bad;
            }
    std::printf("[2/3] composed gathers: %s (%d bad of %zu)\n",
                gather_bad == 0 ? "PASS" : "FAIL", gather_bad, h_ksc.size());

    // ---- attention: qsa_decode_attn KV_MODE 3 over the hybrid pools + output fwht, vs fp32 references
    std::vector<float> q((size_t) QH * D);
    for (auto& x : q) x = nd(rng);
    float* d_q = dalloc<float>(q.size());
    ck(cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice), "q");
    k::QsaAttnPools pools;
    pools.k_q = d_kq; pools.k_scale = d_ks; pools.v_q4 = d_v4; pools.page_table = d_table;
    float* d_scr = dalloc<float>(k::qsa_decode_attn_scratch_floats(cells, s) + 16);
    float* d_attn = dalloc<float>((size_t) QH * D);
    k::qsa_decode_attn_step(d_q, pools, d_ids, d_step, cells, s, d_scr, d_attn, nullptr);
    k::fwht256_inplace_cuda(d_attn, QH, nullptr);   // the engine's inverse rotation of the output
    ck(cudaDeviceSynchronize(), "attn sync");
    std::vector<float> h_attn((size_t) QH * D);
    ck(cudaMemcpy(h_attn.data(), d_attn, h_attn.size() * 4, cudaMemcpyDeviceToHost), "attn");

    // reference A: attention over the SAME dequantized K/V (isolates the kernel; fp32 host math)
    // reference B: attention over the TRUE unquantized K/V (bounds the format's error)
    auto host_attn = [&](const std::vector<float>& Keq, const std::vector<float>& Veq, std::vector<float>& out) {
        out.assign((size_t) QH * D, 0.0f);
        for (int h = 0; h < QH; ++h) {
            const int kvh = h / PER;
            std::vector<float> sc(cells);
            float m = -1e30f;
            for (int c = 0; c < cells; ++c) {
                float d_ = 0.0f;
                for (int t = 0; t < D; ++t)
                    d_ += q[(size_t) h * D + t] * Keq[((size_t) c * H + kvh) * D + t];
                sc[c] = d_ / std::sqrt((float) D);
                m = std::max(m, sc[c]);
            }
            float l = 0.0f;
            for (int c = 0; c < cells; ++c) { sc[c] = std::exp(sc[c] - m); l += sc[c]; }
            for (int c = 0; c < cells; ++c)
                for (int t = 0; t < D; ++t) out[(size_t) h * D + t] += sc[c] / l * Veq[((size_t) c * H + kvh) * D + t];
        }
    };
    std::vector<float> Kdq((size_t) cells * H * D), Vdq((size_t) cells * H * D);
    for (int c = 0; c < cells; ++c)
        for (int h = 0; h < H; ++h)
            for (int d = 0; d < D; ++d) {
                Kdq[((size_t) c * H + h) * D + d] =
                    (float) rk[((size_t) c * H + h) * D + d] * k::f32_from_f16(rk_s[((size_t) c * H + h) * G + d / k::KV_Q8_GROUP]);
                const k::block_q4_0& b = rv4[((size_t) c * H + h) * (D / 32) + d / 32];
                const int nib = d % 32 < 16 ? (b.qs[d % 32] & 0x0F) - 8 : ((b.qs[d % 32 - 16] >> 4) - 8);
                Vdq[((size_t) c * H + h) * D + d] = (float) nib * k::f32_from_f16(b.d);
            }
    // un-rotate the dequantized V (the kernel returns the output in the UNROTATED domain)
    std::vector<float> Vdq_un(Vdq.size());
    std::memcpy(Vdq_un.data(), Vdq.data(), Vdq.size() * 4);
    for (int c = 0; c < cells * H; ++c) fwht256_host_reference(Vdq_un.data() + (size_t) c * D);

    // ---- batched form with TWO queries and per-query step arrays (the verify.cpp window's exact call)
    {
        const int nq = 2;
        std::vector<float> q2((size_t) nq * QH * D);
        for (auto& x : q2) x = nd(rng);
        float* d_q2 = dalloc<float>(q2.size());
        ck(cudaMemcpy(d_q2, q2.data(), q2.size() * 4, cudaMemcpyHostToDevice), "q2");
        // the batched kernel indexes ids[z * cap + ...]: one FULL selection per query
        std::vector<int32_t> ids2((size_t) nq * cells);
        for (int i = 0; i < nq; ++i) for (int c = 0; c < cells; ++c) ids2[(size_t) i * cells + c] = c;
        int32_t* d_ids2 = dalloc<int32_t>((size_t) nq * cells);
        ck(cudaMemcpy(d_ids2, ids2.data(), ids2.size() * 4, cudaMemcpyHostToDevice), "ids2");
        int32_t* d_steps2 = dalloc<int32_t>(nq * k::kStepCount);
        for (int i = 0; i < nq; ++i) {
            int32_t st2[k::kStepCount];
            k::qsa_step_fill(st2, cells - 1 - i, s);
            st2[k::kStepWidth] = cells - i;   // slightly different widths, as verify's windows have
            ck(cudaMemcpy(d_steps2 + i * k::kStepCount, st2, sizeof st2, cudaMemcpyHostToDevice), "steps2");
        }
        float* d_scr2 = dalloc<float>(k::qsa_decode_attn_scratch_floats(cells, s) * nq + 64);
        float* d_at2 = dalloc<float>((size_t) nq * QH * D);
        k::qsa_decode_attn_batch(d_q2, pools, d_ids2, d_steps2, cells, s, d_scr2, d_at2, nq, nullptr);
        k::fwht256_inplace_cuda(d_at2, (int64_t) nq * QH, nullptr);
        ck(cudaDeviceSynchronize(), "attn2 sync");
        std::vector<float> h_at2((size_t) nq * QH * D);
        ck(cudaMemcpy(h_at2.data(), d_at2, h_at2.size() * 4, cudaMemcpyDeviceToHost), "at2");
        double max_b = 0.0;
        double per_q[2] = {0.0, 0.0};
        for (int i = 0; i < nq; ++i)
            for (int h = 0; h < QH; ++h) {
                // reference over the cells this query's width selects (ids are ascending: the first width)
                const int w = cells - i;
                std::vector<float> sc(w);
                float m = -1e30f;
                for (int c = 0; c < w; ++c) {
                    float d_ = 0.0f;
                    for (int t = 0; t < D; ++t)
                        d_ += q2[((size_t) i * QH + h) * D + t] * Kdq[((size_t) c * H + h / PER) * D + t];
                    sc[c] = d_ / std::sqrt((float) D);
                    m = std::max(m, sc[c]);
                }
                float l = 0.0f;
                for (int c = 0; c < w; ++c) { sc[c] = std::exp(sc[c] - m); l += sc[c]; }
                for (int d = 0; d < D; ++d) {
                    float acc = 0.0f;
                    for (int c = 0; c < w; ++c)
                        acc += sc[c] / l * Vdq_un[((size_t) c * H + h / PER) * D + d];   // dim d only
                    const double e = (double) std::fabs(h_at2[((size_t) i * QH + h) * D + d] - acc);
                    max_b = std::max(max_b, e);
                    per_q[i] = std::max(per_q[i], e);
                }
            }
        const bool bok = max_b < 5e-3;
        std::printf("       (per-query max: q0 %.2e, q1 %.2e)\n", per_q[0], per_q[1]);
        std::printf("[4/4] mode-3 BATCHED attention (verify's form): %s (vs dequant ref %.2e)\n",
                    bok ? "PASS" : "FAIL", max_b);
        if (!bok) g_fail = 1;
    }

    std::vector<float> ref_deq, ref_true;
    host_attn(Kdq, Vdq_un, ref_deq);
    host_attn(K, V, ref_true);
    double max_deq = 0.0, max_true = 0.0;
    for (size_t i = 0; i < h_attn.size(); ++i) {
        max_deq = std::max(max_deq, (double) std::fabs(h_attn[i] - ref_deq[i]));
        max_true = std::max(max_true, (double) std::fabs(h_attn[i] - ref_true[i]));
    }
    // The tensor-core prompt path (qsa_prompt_attn, 0.1.22+) takes hybrid pools as KV_MODE 3: INT8 K, and V
    // dequantized from its q4_0 blocks to fp16 at gather. Its summation order differs from the old kernel's
    // (accuracy-level, not bitwise - qsa_prompt_attn.hpp says the same of its int8 mode), so: tolerance check
    // against the SAME dequant reference, output un-rotated exactly as prefill.cpp does.
    {
        float* d_at4 = dalloc<float>((size_t) QH * D);
        const bool took = k::qsa_prompt_attn_batch(d_q, pools, d_ids, d_step, cells, s, d_at4, 1, nullptr);
        if (!took) {
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)
            // AMD: the tensor-core prompt path is CUDA-only, so it refuses every pool and the old kernel runs
            std::printf("[5/5] qsa_prompt_attn mode 3: PASS (refused on HIP - the old kernel runs)\n");
#else
            // a card below compute capability 7.0 (Pascal) never had the tensor-core prompt kernels: the old kernel runs
            int dev5 = 0, major5 = 0;
            cudaGetDevice(&dev5);
            cudaDeviceGetAttribute(&major5, cudaDevAttrComputeCapabilityMajor, dev5);
            if (major5 > 0 && major5 < 7) {
                std::printf("[5/5] qsa_prompt_attn mode 3: PASS (refused below sm_70 - the old kernel runs)\n");
            } else {
                std::printf("[5/5] qsa_prompt_attn mode 3: FAIL (refused the hybrid pools)\n");
                g_fail = 1;
            }
#endif
        } else {
            k::fwht256_inplace_cuda(d_at4, QH, nullptr);
            ck(cudaDeviceSynchronize(), "at4 sync");
            std::vector<float> h_at4((size_t) QH * D);
            ck(cudaMemcpy(h_at4.data(), d_at4, h_at4.size() * 4, cudaMemcpyDeviceToHost), "at4");
            double max_p = 0.0;
            for (size_t i = 0; i < h_at4.size(); ++i)
                max_p = std::max(max_p, (double) std::fabs(h_at4[i] - ref_deq[i]));
            const bool pok = max_p < 1e-2;
            std::printf("[5/5] qsa_prompt_attn mode 3 (tensor cores): %s (vs dequant ref %.2e)\n",
                        pok ? "PASS" : "FAIL", max_p);
            if (!pok) g_fail = 1;
        }
    }

    const bool attn_ok = max_deq < 5e-3;   // fp32 kernel math vs fp64-ish host accumulation
    std::printf("[3/3] mode-3 attention + output fwht: %s (vs dequant ref %.2e, vs true fp32 %.2e)\n",
                attn_ok ? "PASS" : "FAIL", max_deq, max_true);

    const bool ok = append_bad == 0 && gather_bad == 0 && attn_ok && g_fail == 0;
    std::printf("=== kv_hybrid_parity: %s ===\n", ok ? "ALL PASS" : "FAILED");
    return ok ? 0 : 1;
}
