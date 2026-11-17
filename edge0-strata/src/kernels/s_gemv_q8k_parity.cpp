// src/kernels/s_gemv_q8k_parity.cpp - the Q8_K activation path of `s_gemv`.
//
// THE POINT OF THIS KERNEL IS A CONTRACT, so the test is built around the contract rather than around the
// arithmetic:
//
//   1. the kernel against a host reference that dequantizes the SAME Q8_K bytes and the SAME S-form planes,
//      in double - so what is under test is the indexing, not the format;
//   2. the Q8_K activation is produced by the ACTUAL `quantize_q8_K`, which is byte-exact against ggml
//      (round 198).  Building the activation by hand here would make this a test of two transcriptions;
//   3. **THE Q8_K-vs-FP16 GAP IS MEASURED**, because that is the number the whole activation-contract
//      decision rests on and it has only ever been measured on Q2_0 weights (0.6175%) and on BF16 ones
//      (0.141-0.156%).  A K-quant weight is a third case and it is the one the dense projections use.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s_gemv.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

using strata::kernels::SForm;
using strata::kernels::Codebook;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

/// The host decode, transcribed from `docs/pack-format.md`'s rule: `value = cb[code] * scale + offset`, with
/// the integer bias applied INSIDE the integer domain before the one multiply.
const signed char kIq4Nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

double weight_at(const std::vector<uint8_t>& codes, const std::vector<float>& scales,
                 const std::vector<float>& offset, const SForm& f, long long n_in, long long o, long long i) {
    const int per_byte = 8 / f.code_bits;
    const long long codes_per_row = n_in / per_byte;
    const long long n_groups = n_in / f.group_elems;
    const uint8_t* c = codes.data() + o * codes_per_row;
    const int code = (c[i / per_byte] >> ((int) (i % per_byte) * f.code_bits)) & ((1 << f.code_bits) - 1);
    const double v = (f.codebook == Codebook::Iq4Nl) ? (double) kIq4Nl[code & 0x0F]
                                                     : (double) (code + f.code_bias);
    const long long g = i / f.group_elems;
    double w = v * (double) scales[(size_t) (o * n_groups + g)];
    if (f.has_offset) w += (double) offset[(size_t) (o * n_groups + g)];
    return w;
}

/// Dequantize a block_q8_K buffer on the host: 292 bytes per 256 elements.
double q8k_at(const std::vector<uint8_t>& x, long long i) {
    const uint8_t* blk = x.data() + (size_t) (i / 256) * 292;
    float d;
    std::memcpy(&d, blk, 4);
    const int8_t q = ((const int8_t*) (blk + 4))[i % 256];
    return (double) d * (double) q;
}

struct Case {
    const char* what;
    int bits;
    int bias;
    int group;
    Codebook cb;
    bool has_offset;
};

/// THE RAGGED LAST BLOCK.  The split kernels give each warp one row and a block eight warps, so with
/// `n_out % 8 != 0` the last block holds warps that have no row.  Those warps used to return BEFORE the
/// codebook barrier - undefined behaviour that every real shape (a multiple of 8) avoided.  Here `n_out` is
/// deliberately ragged, every row is checked against the host reference over the same bytes, and a guard
/// band past `n_out` must come back untouched: a warp without a row must neither hang the block nor write.
/// (`compute-sanitizer --tool synccheck` on this binary is the direct check of the barrier itself.)
int ragged_rows(const Case& cs, long long n_out) {
    const long long n_in = 512;
    const long long guard = 64;
    std::mt19937 rng(77 + (unsigned) n_out);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<float> xf((size_t) n_in);
    for (auto& v : xf) v = g(rng);
    SForm f;
    f.code_bits = cs.bits;
    f.code_bias = cs.bias;
    f.group_elems = cs.group;
    f.codebook = cs.cb;
    f.has_offset = cs.has_offset;
    const int per_byte = 8 / cs.bits;
    std::vector<uint8_t> codes((size_t) (n_out * n_in / per_byte));
    for (auto& v : codes) v = (uint8_t) (rng() & 0xFF);
    const long long n_groups = n_in / cs.group;
    std::vector<float> scales((size_t) (n_out * n_groups)), offs((size_t) (n_out * n_groups));
    for (auto& v : scales) v = 0.01f + 0.001f * (float) (rng() % 20);
    for (auto& v : offs) v = 0.2f * g(rng);

    float* d_xf = nullptr;
    uint8_t *d_xk = nullptr, *d_x0 = nullptr, *d_codes = nullptr;
    float *d_scales = nullptr, *d_offs = nullptr, *d_y = nullptr;
    std::vector<uint8_t> xk((size_t) (n_in / 256) * 292), x0((size_t) (n_in / 32) * 34);
    check(cudaMalloc(&d_xf, (size_t) n_in * 4), "rxf");
    check(cudaMalloc(&d_xk, xk.size()), "rxk");
    check(cudaMalloc(&d_x0, x0.size()), "rx0");
    check(cudaMalloc(&d_codes, codes.size()), "rcodes");
    check(cudaMalloc(&d_scales, scales.size() * 4), "rscales");
    check(cudaMalloc(&d_offs, offs.size() * 4), "roffs");
    check(cudaMalloc(&d_y, (size_t) (n_out + guard) * 4), "ry");
    check(cudaMemcpy(d_xf, xf.data(), (size_t) n_in * 4, cudaMemcpyHostToDevice), "rcxf");
    check(cudaMemcpy(d_codes, codes.data(), codes.size(), cudaMemcpyHostToDevice), "rcc");
    check(cudaMemcpy(d_scales, scales.data(), scales.size() * 4, cudaMemcpyHostToDevice), "rcs");
    check(cudaMemcpy(d_offs, offs.data(), offs.size() * 4, cudaMemcpyHostToDevice), "rco");
    strata::kernels::quantize_q8_K(d_xf, d_xk, n_in, nullptr);
    strata::kernels::quantize_q8_0(d_xf, d_x0, n_in, nullptr);
    check(cudaMemcpy(xk.data(), d_xk, xk.size(), cudaMemcpyDeviceToHost), "rcxk");
    check(cudaMemcpy(x0.data(), d_x0, x0.size(), cudaMemcpyDeviceToHost), "rcx0");

    int bad = 0;
    for (int kind = 0; kind < 2; ++kind) {           // 0: Q8_K activation, 1: Q8_0 activation
        std::vector<float> want((size_t) n_out);
        std::vector<double> mag((size_t) n_out);   // sum |w_i a_i| per row: a row whose sum cancels near 0
        for (long long o = 0; o < n_out; ++o) {    // must not turn f32 rounding into a relative failure
            double acc = 0, am = 0;
            for (long long i = 0; i < n_in; ++i) {
                double a;
                if (kind == 0) {
                    a = q8k_at(xk, i);
                } else {
                    const uint8_t* blk = x0.data() + (size_t) (i / 32) * 34;
                    uint16_t dbits;
                    std::memcpy(&dbits, blk, 2);
                    a = (double) strata::kernels::f32_from_f16(dbits) * (double) ((const int8_t*) (blk + 2))[i % 32];
                }
                const double wa = weight_at(codes, scales, offs, f, n_in, o, i) * a;
                acc += wa;
                am += std::fabs(wa);
            }
            want[(size_t) o] = (float) acc;
            mag[(size_t) o] = am;
        }
        // the guard band holds a NaN pattern; any write there shows up as a changed bit
        std::vector<uint32_t> sentinel((size_t) (n_out + guard), 0x7FC0DEADu);
        check(cudaMemcpy(d_y, sentinel.data(), sentinel.size() * 4, cudaMemcpyHostToDevice), "rsent");
        if (kind == 0)
            strata::kernels::s_gemv_q8k_split(d_xk, d_codes, d_scales, cs.has_offset ? d_offs : nullptr, d_y, n_in,
                                              n_out, f, nullptr);
        else
            strata::kernels::s_gemv_q8_0_split(d_x0, d_codes, d_scales, cs.has_offset ? d_offs : nullptr, d_y, n_in,
                                               n_out, f, nullptr);
        check(cudaDeviceSynchronize(), "ragged split");
        std::vector<uint32_t> got_bits(sentinel.size());
        check(cudaMemcpy(got_bits.data(), d_y, got_bits.size() * 4, cudaMemcpyDeviceToHost), "rcy");
        double m = 0, d = 0;
        for (long long o = 0; o < n_out; ++o) {
            float v;
            std::memcpy(&v, &got_bits[(size_t) o], 4);
            m += mag[(size_t) o];
            d += std::fabs((double) want[(size_t) o] - (double) v);
        }
        long long touched = 0;
        for (long long o = n_out; o < n_out + guard; ++o) touched += got_bits[(size_t) o] != 0x7FC0DEADu;
        const double rel = d / (m > 1e-30 ? m : 1e-30);
        const bool ok = rel <= 1e-5 && touched == 0;
        std::printf("  %-28s %s n_out %3lld: rel %.3e, guard words written %lld  %s\n", cs.what,
                    kind == 0 ? "Q8_K" : "Q8_0", n_out, rel, touched, ok ? "ok" : "*** WRONG ***");
        if (!ok) ++bad;
    }
    cudaFree(d_xf); cudaFree(d_xk); cudaFree(d_x0); cudaFree(d_codes); cudaFree(d_scales); cudaFree(d_offs);
    cudaFree(d_y);
    return bad;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: s_gemv_q8k_parity [--selftest]\n"); return 2; }
    }
    int bad = 0;

    // The forms the DENSE weights actually use, taken from the manifest rather than invented:
    //   attn_qkv.weight   IQ4_XS  S4  group 32, no offset   (the GDN mixer's qkv)
    //   attn_gate.weight  Q4_K    S4  group 32, offset      (the GDN mixer's z)
    //   ssm_out.weight    Q5_K    S8  group 32, offset      (the GDN mixer's output)
    const Case cases[] = {
        {"IQ4_XS  S4 g32 no-offset", 4, -8, 32, Codebook::Affine, false},
        {"Q4_K    S4 g32 offset", 4, 0, 32, Codebook::Affine, true},
        {"Q5_K    S8 g32 offset", 8, 0, 32, Codebook::Affine, true},
    };

    for (const Case& cs : cases) {
        const long long n_in = 512, n_out = 64;   // 2 Q8_K blocks wide, small enough for a host reference
        std::mt19937 rng(2024);
        std::normal_distribution<float> g(0.0f, 1.0f);
        std::vector<float> xf((size_t) n_in);
        for (auto& v : xf) v = g(rng);

        SForm f;
        f.code_bits = cs.bits;
        f.code_bias = cs.bias;
        f.group_elems = cs.group;
        f.codebook = cs.cb;
        f.has_offset = cs.has_offset;

        const int per_byte = 8 / cs.bits;
        std::vector<uint8_t> codes((size_t) (n_out * n_in / per_byte));
        for (auto& v : codes) v = (uint8_t) (rng() & 0xFF);
        const long long n_groups = n_in / cs.group;
        std::vector<float> scales((size_t) (n_out * n_groups)), offs((size_t) (n_out * n_groups));
        for (auto& v : scales) v = 0.01f + 0.001f * (float) (rng() % 20);
        for (auto& v : offs) v = 0.2f * g(rng);

        // ---- the activation, from the REAL quantizer
        std::vector<uint8_t> xq8k((size_t) (n_in / 256) * 292);
        float* d_xf = nullptr;
        uint8_t* d_xq = nullptr;
        check(cudaMalloc(&d_xf, (size_t) n_in * 4), "xf");
        check(cudaMalloc(&d_xq, xq8k.size()), "xq");
        check(cudaMemcpy(d_xf, xf.data(), (size_t) n_in * 4, cudaMemcpyHostToDevice), "cxf");
        strata::kernels::quantize_q8_K(d_xf, d_xq, n_in, nullptr);
        check(cudaMemcpy(xq8k.data(), d_xq, xq8k.size(), cudaMemcpyDeviceToHost), "cxq");

        // ---- host reference in double, over the same bytes
        std::vector<float> want((size_t) n_out, 0.0f);
        for (long long o = 0; o < n_out; ++o) {
            double acc = 0;
            for (long long i = 0; i < n_in; ++i)
                acc += weight_at(codes, scales, offs, f, n_in, o, i) * q8k_at(xq8k, i);
            want[(size_t) o] = (float) acc;
        }

        // ---- device
        uint8_t* d_codes = nullptr;
        float *d_scales = nullptr, *d_offs = nullptr, *d_y = nullptr;
        check(cudaMalloc(&d_codes, codes.size()), "codes");
        check(cudaMalloc(&d_scales, scales.size() * 4), "scales");
        check(cudaMalloc(&d_offs, offs.size() * 4), "offs");
        check(cudaMalloc(&d_y, (size_t) n_out * 4), "y");
        check(cudaMemcpy(d_codes, codes.data(), codes.size(), cudaMemcpyHostToDevice), "cc");
        check(cudaMemcpy(d_scales, scales.data(), scales.size() * 4, cudaMemcpyHostToDevice), "cs");
        check(cudaMemcpy(d_offs, offs.data(), offs.size() * 4, cudaMemcpyHostToDevice), "co");

        std::vector<float> naive((size_t) n_out), warp((size_t) n_out);
        strata::kernels::s_gemv_q8k(d_xq, d_codes, d_scales, cs.has_offset ? d_offs : nullptr, d_y, n_in, n_out,
                                    f, nullptr);
        check(cudaMemcpy(naive.data(), d_y, (size_t) n_out * 4, cudaMemcpyDeviceToHost), "cy1");
        strata::kernels::s_gemv_q8k_split(d_xq, d_codes, d_scales, cs.has_offset ? d_offs : nullptr, d_y, n_in,
                                          n_out, f, nullptr);
        check(cudaMemcpy(warp.data(), d_y, (size_t) n_out * 4, cudaMemcpyDeviceToHost), "cy2");

        double mag = 0, d1 = 0, d2 = 0;
        for (long long o = 0; o < n_out; ++o) {
            mag += std::fabs((double) want[(size_t) o]);
            d1 += std::fabs((double) want[(size_t) o] - (double) naive[(size_t) o]);
            d2 += std::fabs((double) want[(size_t) o] - (double) warp[(size_t) o]);
        }
        std::printf("  %-28s naive %.3e   warp %.3e\n", cs.what, d1 / mag, d2 / mag);
        // The kernel accumulates in f32, the reference in f64, over 512 terms with values of similar size.
        // 1e-5 is the arithmetic's floor, not a structural allowance.
        if (!(d1 / mag <= 1e-5 && d2 / mag <= 1e-5)) { std::printf("    *** over 1e-5 ***\n"); ++bad; }

        // ---- 2b. THE Q8_0 ACTIVATION, on the SAME weight planes.  `s_gemv_q8_0_split` exists because the
        // legacy block formats want Q8_0 and `ffn_down_shexp` cannot use Q8_K at all (n_in = 640 is not a
        // multiple of 256).  It shares every line with the Q8_K kernel except the activation loader, so what
        // this checks is the LOADER - the 34-byte block, the fp16 `d`, and the element index inside it.
        {
            std::vector<uint8_t> xq0((size_t) (n_in / 32) * 34);
            uint8_t* d_x0 = nullptr;
            check(cudaMalloc(&d_x0, xq0.size()), "x0");
            strata::kernels::quantize_q8_0(d_xf, d_x0, n_in, nullptr);
            check(cudaMemcpy(xq0.data(), d_x0, xq0.size(), cudaMemcpyDeviceToHost), "cx0");

            // the host reference over the SAME Q8_0 bytes, with the same weight decode
            std::vector<float> want0((size_t) n_out, 0.0f);
            for (long long o = 0; o < n_out; ++o) {
                double acc = 0;
                for (long long i = 0; i < n_in; ++i) {
                    const uint8_t* blk = xq0.data() + (size_t) (i / 32) * 34;
                    uint16_t dbits;
                    std::memcpy(&dbits, blk, 2);
                    const double d = (double) strata::kernels::f32_from_f16(dbits);
                    const int8_t q = ((const int8_t*) (blk + 2))[i % 32];
                    acc += weight_at(codes, scales, offs, f, n_in, o, i) * d * (double) q;
                }
                want0[(size_t) o] = (float) acc;
            }
            strata::kernels::s_gemv_q8_0_split(d_x0, d_codes, d_scales, cs.has_offset ? d_offs : nullptr, d_y,
                                               n_in, n_out, f, nullptr);
            std::vector<float> got0((size_t) n_out);
            check(cudaMemcpy(got0.data(), d_y, (size_t) n_out * 4, cudaMemcpyDeviceToHost), "cy0");
            double m0 = 0, dd0 = 0;
            for (long long o = 0; o < n_out; ++o) {
                m0 += std::fabs((double) want0[(size_t) o]);
                dd0 += std::fabs((double) want0[(size_t) o] - (double) got0[(size_t) o]);
            }
            std::printf("  %-28s Q8_0 act %.3e\n", cs.what, dd0 / (m0 > 1e-30 ? m0 : 1e-30));
            if (!(dd0 / (m0 > 1e-30 ? m0 : 1e-30) <= 1e-5)) {
                std::printf("    *** the Q8_0 activation loader is WRONG ***\n");
                ++bad;
            }
            check(cudaFree(d_x0), "fx0");
        }

        // ---- 3. THE Q8_K vs FP16 GAP, on a K-quant weight - the case the contract decision needs
        {
            std::vector<uint16_t> x16((size_t) n_in);
            for (long long i = 0; i < n_in; ++i) x16[(size_t) i] = strata::kernels::f16_from_f32(xf[(size_t) i]);
            uint16_t* d_x16 = nullptr;
            check(cudaMalloc(&d_x16, (size_t) n_in * 2), "x16");
            check(cudaMemcpy(d_x16, x16.data(), (size_t) n_in * 2, cudaMemcpyHostToDevice), "cx16");
            std::vector<float> fp16out((size_t) n_out);
            strata::kernels::s_gemv(d_x16, d_codes, d_scales, cs.has_offset ? d_offs : nullptr, d_y, n_in, n_out,
                                    f);
            check(cudaMemcpy(fp16out.data(), d_y, (size_t) n_out * 4, cudaMemcpyDeviceToHost), "cy3");
            double dd = 0;
            for (long long o = 0; o < n_out; ++o)
                dd += std::fabs((double) naive[(size_t) o] - (double) fp16out[(size_t) o]);
            const double rel = dd / mag;
            const bool visible = rel > 1e-3;
            std::printf("      %-24s %s (%.4f%% apart, tolerance 0.1%%)\n", "Q8_K vs fp16 activation",
                        visible ? "yes" : "*** NO ***", rel * 100);
            if (!visible) ++bad;
            cudaFree(d_x16);
        }
        cudaFree(d_xf); cudaFree(d_xq); cudaFree(d_codes); cudaFree(d_scales); cudaFree(d_offs); cudaFree(d_y);
    }

    // ---- 4. output widths that are NOT a multiple of the eight rows per block
    std::printf("\n");
    for (const Case& cs : cases)
        for (long long n_out : {1LL, 7LL, 9LL, 61LL}) bad += ragged_rows(cs, n_out);

    std::printf("\ns_gemv_q8k: %d failures\n", bad);
    if (bad) return 1;
    if (selftest) std::printf("s_gemv_q8k_parity OK\n");
    return 0;
}
