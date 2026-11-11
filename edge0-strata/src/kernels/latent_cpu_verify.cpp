// src/kernels/latent_cpu_verify.cpp - CPU latent expert path, per-stage vs the test bundle's
// Python float32 reference.  Isolates D (Q8_0 matvec), A (native NVFP4 gu+down), U (Q8_0 matvec).
//
//     latent_cpu_verify <layerL_test.bin> <latent_a_pack> <latent_dense_q8.bin> <layer>
//
// Each stage is fed the *reference* output of the previous stage so the errors don't compound:
//   xl = D . x              (compare ref_xl)
//   yl = A(xl := ref_xl)    (compare ref_yl)   -- A = swiglu(A_gu . xl) -> A_dn . ff
//   y  = U(yl := ref_yl)    (compare ref_y)
//   out = combine           (compare ref_out, from my own xl/yl/y)
#include "strata/core/latent_dense.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

// ---- copied verbatim from expert_source.cpp (latent_cpu_run's helpers) ----
static inline float latent_f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t e = (h >> 10) & 0x1Fu, m = h & 0x3FFu, f;
    if (e == 0) {
        if (m == 0) f = sign << 31;
        else { e = 127 - 15 + 1; while (!(m & 0x400u)) { m <<= 1; --e; } m &= 0x3FFu; f = (sign << 31) | (e << 23) | (m << 13); }
    } else if (e == 31) f = (sign << 31) | 0x7F800000u | (m << 13);
    else f = (sign << 31) | ((e - 15 + 127) << 23) | (m << 13);
    float out; std::memcpy(&out, &f, 4); return out;
}
static void q8_0_matvec_f32(const uint8_t* W, size_t row_bytes, int nrows, int ncols, const float* x, float* y) {
    const int nb = ncols / 32;
    for (int r = 0; r < nrows; ++r) {
        const uint8_t* row = W + (size_t) r * row_bytes;
        float acc = 0.f;
        for (int b = 0; b < nb; ++b) {
            const uint8_t* blk = row + (size_t) b * 34;
            const float d = latent_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
            const int8_t* q = (const int8_t*)(blk + 2);
            const float* xb = x + (size_t) b * 32;
            float s = 0.f;
            for (int j = 0; j < 32; ++j) s += (float) q[j] * xb[j];
            acc += d * s;
        }
        y[r] = acc;
    }
}
// ---- end copied ----

namespace {
template<typename T>
std::vector<T> read_n(std::ifstream& f, size_t n, const char* what) {
    std::vector<T> v(n);
    if (!f.read((char*) v.data(), (std::streamsize) (n * sizeof(T)))) {
        std::fprintf(stderr, "short read: %s\n", what);
        std::exit(1);
    }
    return v;
}
double rel_l2(const float* got, const float* ref, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) {
        const double d = (double) got[i] - (double) ref[i];
        num += d * d; den += (double) ref[i] * (double) ref[i];
    }
    return std::sqrt(num / (den + 1e-30));
}
float max_abs(const float* got, const float* ref, size_t n) {
    float m = 0;
    for (size_t i = 0; i < n; ++i) m = std::fmax(m, std::fabs(got[i] - ref[i]));
    return m;
}
void report(const char* name, const std::vector<float>& got, const std::vector<float>& ref) {
    std::printf("  %-4s relL2 %.4f%%  max_abs %.6g\n", name,
                100 * rel_l2(got.data(), ref.data(), got.size()),
                max_abs(got.data(), ref.data(), got.size()));
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage: %s test.bin latent_a_pack dense_q8.bin layer\n", argv[0]); return 2; }
    const int layer = std::atoi(argv[4]);
    std::ifstream f(argv[1], std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
    char magic[4];
    f.read(magic, 4);
    if (std::memcmp(magic, "LAT0", 4) != 0) { std::fprintf(stderr, "bad magic\n"); return 2; }
    int32_t hdr[7];
    if (!f.read((char*) hdr, sizeof hdr)) return 2;
    const int T = hdr[0], K = hdr[1], G = hdr[2], E = hdr[3], H = hdr[4], ELL = hdr[5], I = hdr[6];
    std::printf("layer=%d T=%d K=%d G=%d E=%d H=%d ELL=%d I=%d\n", layer, T, K, G, E, H, ELL, I);
    if (T * K > 128) { std::fprintf(stderr, "cap too large\n"); return 2; }

    const auto x = read_n<float>(f, (size_t) T * H, "x");
    const auto topk = read_n<int32_t>(f, (size_t) T * K, "topk");
    const auto w = read_n<float>(f, (size_t) T * K, "w");
    const auto labels = read_n<int32_t>(f, (size_t) E, "labels");
    const size_t drow = (size_t) H / 32 * 34, urow = (size_t) ELL / 32 * 34;
    f.seekg((std::streamoff) ((size_t) G * ELL * drow + (size_t) G * H * urow), std::ios::cur);
    const auto ref_xl = read_n<float>(f, (size_t) T * G * ELL, "ref_xl");
    const auto ref_yl = read_n<float>(f, (size_t) T * K * ELL, "ref_yl");
    const auto ref_y = read_n<float>(f, (size_t) T * K * H, "ref_y");
    const auto ref_out = read_n<float>(f, (size_t) T * H, "ref_out");

    std::string err;
    if (!cpu::expert_layout_load(argv[2], 48, 512, err)) { std::fprintf(stderr, "layout: %s\n", err.c_str()); return 2; }
    strata::core::FileExpertSource src;
    if (!src.open(argv[2], 48, 512, err)) { std::fprintf(stderr, "source: %s\n", err.c_str()); return 2; }
    strata::core::LatentDense dense;
    if (!dense.open(argv[3], 48, G, H, ELL, err)) { std::fprintf(stderr, "dense: %s\n", err.c_str()); return 2; }

    const cpu::NativeFmt& nf = cpu::expert_layout().fmt[(size_t) layer];
    std::printf("NativeFmt: gu_type=%d d_type=%d n_embd=%lld n_ff=%lld gu_row=%zu up_off=%zu down_off=%zu act_bytes=%zu h_bytes=%zu tail_off=%zu\n",
                nf.gu_type, nf.d_type, (long long) nf.n_embd, (long long) nf.n_ff,
                nf.gu_row, nf.up_off, nf.down_off, nf.act_bytes, nf.h_bytes, nf.tail_off);

    const uint8_t* D = dense.d(layer);
    const uint8_t* U = dense.u(layer);

    // ---- D stage: xl = D . x, compare ref_xl ----
    std::vector<float> my_xl((size_t) T * G * ELL);
    for (int t = 0; t < T; ++t)
        for (int g = 0; g < G; ++g)
            q8_0_matvec_f32(D + (size_t) g * ELL * drow, drow, ELL, H, x.data() + (size_t) t * H,
                            my_xl.data() + ((size_t) t * G + g) * ELL);
    report("xl", my_xl, ref_xl);

    // ---- A stage: yl = A(ref_xl), compare ref_yl (isolated: feed ref_xl, not my_xl) ----
    std::vector<float> my_yl((size_t) T * K * ELL);
    {
        std::vector<uint8_t> act((size_t) cpu::kNativeActBytes), hq((size_t) cpu::kNativeHBytes);
        std::vector<float> ff((size_t) I);
        const void* actp = act.data(); float* ffp = ff.data();
        const void* hqp = hq.data(); float* hp = nullptr;
        for (int i = 0; i < T * K; ++i) {
            const int e = topk[i], t = i / K, g = labels[(size_t) e];
            const uint8_t* blob = src.blob(layer, e);
            if (blob == nullptr) { std::fprintf(stderr, "no blob for e=%d\n", e); return 2; }
            const float* xl_tg = ref_xl.data() + ((size_t) t * G + g) * ELL;
            cpu::native_quant_act(nf, xl_tg, act.data());
            cpu::native_gu_rows(nf, blob, &actp, 1, &ffp, 0, I);
            cpu::native_quant_h(nf, ff.data(), hq.data());
            hp = my_yl.data() + (size_t) i * ELL;
            cpu::native_down_rows(nf, blob, &hqp, 1, &hp, 0, ELL);
        }
    }
    report("yl", my_yl, ref_yl);

    // ---- U stage: y = U(ref_yl), compare ref_y (isolated: feed ref_yl, not my_yl) ----
    std::vector<float> my_y((size_t) T * K * H);
    for (int i = 0; i < T * K; ++i) {
        const int e = topk[i], g = labels[(size_t) e];
        q8_0_matvec_f32(U + (size_t) g * H * urow, urow, H, ELL, ref_yl.data() + (size_t) i * ELL,
                        my_y.data() + (size_t) i * H);
    }
    report("y", my_y, ref_y);

    // ---- combine (using my own xl/yl/y): out[t] = sum_k w[r] * y[r] ----
    std::vector<float> my_out((size_t) T * H, 0.f);
    for (int i = 0; i < T * K; ++i) {
        const int t = i / K;
        const float wi = w[(size_t) i];
        for (int c = 0; c < H; ++c) my_out[(size_t) t * H + c] += wi * my_y[(size_t) i * H + c];
    }
    report("out", my_out, ref_out);

    std::printf("DONE\n");
    return 0;
}
