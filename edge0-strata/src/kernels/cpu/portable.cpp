// src/kernels/cpu/portable.cpp - the CPU kernel symbols on a non-x86 build (aarch64: DGX Spark / GB10, Grace).
//
// Replaces expert.cpp, q2_avx2.cpp, iq_avx2.cpp, iq_avx512.cpp, kq_avx2.cpp and kq_avx1.cpp, which are AVX-only.
// What a native (GGUF) pack needs from this file is the activation quantizer (expert.cpp's scalar loop, bitwise
// the same as its AVX-512 version) and the BF16 router dot (NEON).  The i-quant / k-quant multi-token kernels
// report themselves unsupported, so native_expert.cpp takes ggml-cpu's vec_dot, which has NEON paths.  The
// canonical Q2_0 pack (its own blob layout) is refused: those kernels are AVX-512 only.  A native pack's Q2_0
// rows are not routed here: q2_native_kernels() is false when cpu_avx2_ok() is, so they use ggml-cpu too.
// Local port of the approach in https://github.com/Niko1221/Strata/pull/1077 onto main e8ca9af.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/kq_avx1.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx1.hpp"
#include "strata/kernels/cpu/nvfp4_avx2.hpp"
#include "strata/kernels/cpu/nvfp4_avx512.hpp"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace strata::kernels::cpu {
namespace {

std::atomic<bool> oracle_q8_0{false};

[[noreturn]] void unsupported(const char* what) {
    std::fprintf(stderr,
                 "strata: %s is x86-only (AVX2/AVX-512) and this build has no portable version.\n"
                 "        On this CPU use a native GGUF pack (IQ2_XS, IQ3_S, IQ4_NL, Q4_K, ...).\n",
                 what);
    std::abort();
}

inline float bf16f(uint16_t h) {
    const uint32_t b = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

}  // namespace





void act_quant_q8_1(const float* x, int n, ActQ& a) {
    // The Q8_0 oracle contract (expert.cpp's quantize_oracle_q8_0) is an AVX rounding.  The legacy scalar loop
    // below is what the engine uses; the oracle flag would silently quantize differently here.
    if (oracle_q8_0.load(std::memory_order_relaxed)) unsupported("the Q8_0 oracle activation quantizer");
    // expert.cpp's scalar loop: round half away from zero, clamp to +-127.
    a.nchunks = n / QKA;
    for (int k = 0; k < a.nchunks; ++k) {
        const float* xb = x + k * QKA;
        float amax = 0.f;
        for (int j = 0; j < QKA; ++j) amax = std::fmax(amax, std::fabs(xb[j]));
        const float s = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = s > 0.f ? 1.f / s : 0.f;
        int32_t sum = 0;
        int8_t* q = a.q + k * QKA;
        for (int j = 0; j < QKA; ++j) {
            const float t = xb[j] * inv;
            const float r = t + (t >= 0.f ? 0.5f : -0.5f);
            int v = (int) r;
            v = v < -127 ? -127 : (v > 127 ? 127 : v);
            q[j] = (int8_t) v;
            sum += v;
        }
        a.scale[k] = s;
        a.sum[k] = sum;
        a.hx[k] = s * (float) sum;
    }
}

void act_quant_q8_1_avx2(const float* x, int n, ActQ& a) { act_quant_q8_1(x, n, a); }

void s2_expert_vnni(const uint8_t*, const float*, float*, ExpertScratch&) { unsupported("the Q2_0 expert kernel"); }
void s2_expert_vnni_q(const uint8_t*, const ActQ&, float*, ExpertScratch&) { unsupported("the Q2_0 expert kernel"); }
void s2_expert_gu_rows(const uint8_t*, const ActQ&, float*, int, int) { unsupported("the Q2_0 expert kernel"); }
void s2_expert_down_rows(const uint8_t*, const ActQ&, float*, int, int) { unsupported("the Q2_0 expert kernel"); }
void s2_expert_gu_rows_multi(const uint8_t*, const ActQ* const*, int, float* const*, int, int) {
    unsupported("the Q2_0 expert kernel");
}
void s2_expert_down_rows_multi(const uint8_t*, const ActQ* const*, int, float* const*, int, int) {
    unsupported("the Q2_0 expert kernel");
}
void s2_expert_vnni_multi(const uint8_t*, const ActQ* const*, int, float* const*, ExpertScratchMulti&) {
    unsupported("the Q2_0 expert kernel");
}
void q2_0_gguf_rows_multi(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    unsupported("the Q2_0 GGUF row kernel");
}
void q2_0_gguf_rows_multi_avx2(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    unsupported("the Q2_0 GGUF row kernel");
}
bool q2_bitplane_enabled() { return false; }
void q2_0_gguf_rows_multi_avx2_legacy(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    unsupported("the Q2_0 GGUF row kernel");
}
void q2_0_gguf_rows_multi_avx2_v(bool, const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    unsupported("the Q2_0 GGUF row kernel");
}
void s2_expert_scalar(const uint8_t*, const float*, float*, bool) { unsupported("the Q2_0 scalar oracle"); }

void iq512_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX-512 i-quant kernel");
}
void iq512_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX-512 i-quant kernel");
}
bool iq512_gather_on() noexcept { return false; }
bool iq512x_supported(int) noexcept { return false; }
void iq512x_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX-512 exact i-quant kernel");
}
bool iq256x_supported(int) noexcept { return false; }   // (the DeepSeek CPU lane's multi-token rows: ggml-cpu's NEON dots instead)
void iq256x_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX2 multi-token i-quant rows");
}
int iq256_variant() noexcept { return 0; }
int iq256_variants() noexcept { return 0; }
void iq256_gu_rows_v(int, int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX-2 i-quant kernel");
}
void iq256_rows_v(int, int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX-2 i-quant kernel");
}
void iq256_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX-2 i-quant kernel");
}
void iq256_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX-2 i-quant kernel");
}
void iq4nl256_down_rows_v(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX-2 IQ4_NL kernel");
}
void iq4nl256_down_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX-2 IQ4_NL kernel");
}
void q8k_quant_avx2(const float*, void*, int64_t) { unsupported("the AVX-2 Q8_K quantizer"); }
void kq256_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX-2 k-quant kernel");
}
void kq256_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX-2 k-quant kernel");
}
void bf16_rows_dot_multi_avx1(const uint16_t*, int, int, const float*, int, float*) {
    unsupported("the AVX1 router dot");
}

void bf16_rows_dot(const uint16_t* w, int rows, int cols, const float* x, float* out) {
    bf16_rows_dot_multi(w, rows, cols, x, 1, out);
}

void bf16_rows_dot_multi(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out) {
    for (int r = 0; r < rows; ++r) {
        const uint16_t* wr = w + (size_t) r * (size_t) cols;
        for (int t = 0; t < nt; ++t) {
            const float* xt = x + (size_t) t * cols;
            int c = 0;
            float s = 0.f;
#if defined(__ARM_NEON)
            float32x4_t acc0 = vdupq_n_f32(0.f), acc1 = vdupq_n_f32(0.f);
            for (; c + 8 <= cols; c += 8) {
                const uint16x8_t h = vld1q_u16(wr + c);
                const float32x4_t w0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h), 16));
                const float32x4_t w1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(h), 16));
                acc0 = vfmaq_f32(acc0, w0, vld1q_f32(xt + c));
                acc1 = vfmaq_f32(acc1, w1, vld1q_f32(xt + c + 4));
            }
            s = vaddvq_f32(vaddq_f32(acc0, acc1));
#endif
            for (; c < cols; ++c) s += bf16f(wr[c]) * xt[c];
            out[(size_t) t * rows + r] = s;
        }
    }
}


// aarch64 has none of the x86 kernels: these names are referenced by shared code but reached only behind
// the x86 feature checks (all of them report unsupported here).
[[noreturn]] static void cpu_kernel_unreachable(const char* name) {
    std::fprintf(stderr, "strata: %s is an x86 kernel and cannot run on this CPU\n", name);
    std::abort();
}
void act_quant_q8_1_avx1(const float* x, int n, ActQ& a) { cpu_kernel_unreachable("act_quant_q8_1_avx1"); }
void iq128_down_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt, float* const* out, int r0, int r1) { cpu_kernel_unreachable("iq128_down_rows"); }
bool iq128_down_supported(int ggml_type) noexcept { return false; }
void iq128_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt, float* const* ff, int r0, int r1) { cpu_kernel_unreachable("iq128_gu_rows"); }
bool iq128_supported(int ggml_type) noexcept { return false; }
void iq256_gu_rows_exact_one(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* act, float* ff, int r0, int r1) { cpu_kernel_unreachable("iq256_gu_rows_exact_one"); }
bool nvfp4_256_fits(int n) { return false; }
void nvfp4_256_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt, float* const* ff, int r0, int r1, float s_gate, float s_up) { cpu_kernel_unreachable("nvfp4_256_gu_rows"); }
void nvfp4_256_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0, int r1, float scale) { cpu_kernel_unreachable("nvfp4_256_rows"); }
bool nvfp4_512_fits(int n) { return false; }
void nvfp4_512_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt, float* const* ff, int r0, int r1, float s_gate, float s_up) { cpu_kernel_unreachable("nvfp4_512_gu_rows"); }
void nvfp4_512_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0, int r1, float scale) { cpu_kernel_unreachable("nvfp4_512_rows"); }
void q2_0_gguf_rows_multi_avx1_legacy(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt, float* const* out, int r0, int r1) { cpu_kernel_unreachable("q2_0_gguf_rows_multi_avx1_legacy"); }
void q2_0_gguf_rows_multi_avx1(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt, float* const* out, int r0, int r1) { cpu_kernel_unreachable("q2_0_gguf_rows_multi_avx1"); }
void q8k_quant_avx1(const float* x, void* y, int64_t n) { cpu_kernel_unreachable("q8k_quant_avx1"); }
// (the trellis rows are tq_cpu.cpp's: its scalar kernel runs here, as in the Metal build)

}  // namespace strata::kernels::cpu
