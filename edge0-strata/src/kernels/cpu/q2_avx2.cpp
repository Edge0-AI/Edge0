// src/kernels/cpu/q2_avx2.cpp - plan v0.3 P6: the Q2_0 expert rows and the activation quantizer for CPUs
// without AVX-512 (Intel Core 12th-14th gen and Core Ultra, AMD Zen 2/3).
//
// Compiled with AVX2 only, so nothing here can fault on those CPUs; the row kernel's AVX-VNNI copy (vpdpbusd, the
// same sums) is compiled under its own target attribute and called only where cpu_avxvnni_ok().  The arithmetic is
// the AVX-512 kernels': codes 0..3 against the int8 activation per 32-value chunk, times the weight scale and the
// chunk scale, minus the weight scale times the chunk's `hx` (the -1 code offset); the quantizer is the scalar rule,
// bit for bit.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <immintrin.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

// STRATA_AVXVNNI (CMake: the compiler has the AVX-VNNI intrinsics), as in iq_avx2.cpp
#if !defined(STRATA_AVXVNNI)
#define STRATA_AVXVNNI 0
#endif
#if STRATA_AVXVNNI && (defined(__GNUC__) || defined(__clang__))
#define STRATA_AVXVNNI_FN __attribute__((target("avxvnni")))
#else
#define STRATA_AVXVNNI_FN
#endif

namespace strata::kernels::cpu {
namespace {

inline float h2f(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h)));
}

// 16 bytes of 2-bit codes (value i in byte i/4, bits 2*(i%4)) -> 64 codes in value order, two 32-byte vectors
inline void unpack64(const uint8_t* codes, __m256i& lo, __m256i& hi) {
    const __m128i b = _mm_loadu_si128((const __m128i*) codes);
    const __m128i m3 = _mm_set1_epi8(3);
    const __m128i c0 = _mm_and_si128(b, m3);
    const __m128i c1 = _mm_and_si128(_mm_srli_epi16(b, 2), m3);
    const __m128i c2 = _mm_and_si128(_mm_srli_epi16(b, 4), m3);
    const __m128i c3 = _mm_and_si128(_mm_srli_epi16(b, 6), m3);
    const __m128i a0 = _mm_unpacklo_epi8(c0, c1), a1 = _mm_unpacklo_epi8(c2, c3);   // bytes 0..7
    const __m128i b0 = _mm_unpackhi_epi8(c0, c1), b1 = _mm_unpackhi_epi8(c2, c3);   // bytes 8..15
    lo = _mm256_set_m128i(_mm_unpackhi_epi16(a0, a1), _mm_unpacklo_epi16(a0, a1));  // values 0..31
    hi = _mm256_set_m128i(_mm_unpackhi_epi16(b0, b1), _mm_unpacklo_epi16(b0, b1));  // values 32..63
}

// ---- the row kernel, twice: the AVX2 form and the AVX-VNNI one (q2_avx2_rows.inl)
namespace plain {
#define STRATA_ROWS_VNNI 0
#define STRATA_ROWS_FN
#include "q2_avx2_rows.inl"
}  // namespace plain
#if STRATA_AVXVNNI
namespace vnni {
#define STRATA_ROWS_VNNI 1
#define STRATA_ROWS_FN STRATA_AVXVNNI_FN
#include "q2_avx2_rows.inl"
}  // namespace vnni
#endif

// ================================ the "bit-plane" row dot (opt-in: STRATA_Q2_BITPLANE=1) ================================
//
// The kernel above spends, per 32-value chunk and token, a maddubs, a madd, a convert, a scalar d*scale product
// broadcast into an FMA, and a scalar correction FMA; plus a 14-op interleave to unpack 64 codes.  Here:
//
//   * NO INTERLEAVE.  Byte i of a block's 16 code bytes holds the codes of values 4i..4i+3, code k at bits 2k.
//     `(bytes >> 2k) & 3` is therefore "plane k": the codes of values 4i+k, i = 0..15, in byte order.  The
//     activation is permuted ONCE per token (act_quant_q8_1_avx2 -> ActQ::qp) into the same plane order, so
//     plane k meets its activations with no shuffle.  Two blocks share one ymm (one per 128-bit lane).
//   * THE FOUR PLANES ARE SUMMED IN int16, then ONE madd.  |pair| <= 2*3*127 = 762, four planes <= 3048: no
//     saturation.  After the madd, int32 lanes 0-1 of each 128-bit half are values 0..31 of the block (one
//     activation chunk) and lanes 2-3 values 32..63 (the next): the per-chunk scale is a precomputed per-token
//     lane pattern (ActQ::pscale), not a broadcast.
//   * THE -1 CODE OFFSET IS SUBTRACTED IN THE INTEGERS: each lane minus the exact sum of the activations it
//     paired with (ActQ::psum), so sum((c-1)q) is exact and the scalar correction chain is gone.
//   * ONE FMA per two blocks per token: acc += (d0 x4 | d1 x4) * (float(isum) * pscale).
//
// The integer part is exact; only the float summation order differs from the kernel above (last bits).  Every
// token's operations are the same whatever the group width, so a token's rows are bitwise the same alone or in a
// verify window.  It changes the last bits of a Q2_0 expert on AVX2-only CPUs, so it is OFF by default
// (STRATA_Q2_BITPLANE=1 turns it on; unset: the kernel above, exactly as before).
template <int NT>
inline void row_bp_acc(const uint8_t* row, const ActQ* const* a, int npairs, __m256* acc) {
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i didx = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1);
    for (int p = 0; p < npairs; ++p) {
        const uint8_t* blk = row + (size_t) p * 36;
        uint16_t d0, d1;
        std::memcpy(&d0, blk, 2);
        std::memcpy(&d1, blk + 18, 2);
        const __m128 d01 = _mm_cvtph_ps(_mm_cvtsi32_si128((int) ((uint32_t) d0 | ((uint32_t) d1 << 16))));
        const __m256 dp = _mm256_permutevar8x32_ps(_mm256_castps128_ps256(d01), didx);
        const __m256i c = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) (blk + 2))),
                                                  _mm_loadu_si128((const __m128i*) (blk + 20)), 1);
        const __m256i c0 = _mm256_and_si256(c, m3);
        const __m256i c1 = _mm256_and_si256(_mm256_srli_epi16(c, 2), m3);
        const __m256i c2 = _mm256_and_si256(_mm256_srli_epi16(c, 4), m3);
        const __m256i c3 = _mm256_and_si256(_mm256_srli_epi16(c, 6), m3);
        for (int t = 0; t < NT; ++t) {
            const int8_t* q = a[t]->qp + (size_t) p * 128;
            __m256i s = _mm256_maddubs_epi16(c0, _mm256_load_si256((const __m256i*) q));
            s = _mm256_add_epi16(s, _mm256_maddubs_epi16(c1, _mm256_load_si256((const __m256i*) (q + 32))));
            s = _mm256_add_epi16(s, _mm256_maddubs_epi16(c2, _mm256_load_si256((const __m256i*) (q + 64))));
            s = _mm256_add_epi16(s, _mm256_maddubs_epi16(c3, _mm256_load_si256((const __m256i*) (q + 96))));
            const __m256i i32 = _mm256_sub_epi32(_mm256_madd_epi16(s, ones),
                                                 _mm256_load_si256((const __m256i*) (a[t]->psum + p * 8)));
            const __m256 v = _mm256_mul_ps(_mm256_cvtepi32_ps(i32), _mm256_load_ps(a[t]->pscale + p * 8));
            acc[t] = _mm256_fmadd_ps(dp, v, acc[t]);
        }
    }
}

// Every row is reduced with ONE tree, ((x0+x1)+(x2+x3)) + ((x4+x5)+(x6+x7)), whether alone or four at a time (the
// hadd form below computes exactly that per row), so a row's bits do not depend on how the rows were cut into tasks.
inline float reduce1(__m256 v) {
    __m256 h = _mm256_hadd_ps(v, v);
    h = _mm256_hadd_ps(h, h);
    return _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(h), _mm256_extractf128_ps(h, 1)));
}
inline __m128 reduce4(__m256 a0, __m256 a1, __m256 a2, __m256 a3) {
    const __m256 h = _mm256_hadd_ps(_mm256_hadd_ps(a0, a1), _mm256_hadd_ps(a2, a3));
    return _mm_add_ps(_mm256_castps256_ps128(h), _mm256_extractf128_ps(h, 1));
}

template <int NT>
void rows_bp(const uint8_t* w, size_t row_bytes, int npairs, const ActQ* const* a, float* const* out, int r0, int r1) {
    int r = r0;
    for (; r + 4 <= r1; r += 4) {
        __m256 acc[4][NT];
        for (int k = 0; k < 4; ++k) row_bp_acc<NT>(w + (size_t) (r + k) * row_bytes, a, npairs, acc[k]);
        for (int t = 0; t < NT; ++t) _mm_storeu_ps(out[t] + r, reduce4(acc[0][t], acc[1][t], acc[2][t], acc[3][t]));
    }
    for (; r < r1; ++r) {
        __m256 acc[NT];
        row_bp_acc<NT>(w + (size_t) r * row_bytes, a, npairs, acc);
        for (int t = 0; t < NT; ++t) out[t][r] = reduce1(acc[t]);
    }
}

const bool kBitplane = [] {
    const char* e = std::getenv("STRATA_Q2_BITPLANE");
    return e != nullptr && e[0] == '1';
}();

}  // namespace

bool q2_bitplane_enabled() { return kBitplane; }

void q2_0_gguf_rows_multi_avx2_v(bool vnni_rows, const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a,
                                 int nt, float* const* out, int r0, int r1) {
#if STRATA_AVXVNNI
    if (vnni_rows) {
        vnni::rows_nt(w, row_bytes, nblocks, a, nt, out, r0, r1);
        return;
    }
#endif
    (void) vnni_rows;
    plain::rows_nt(w, row_bytes, nblocks, a, nt, out, r0, r1);
}

void q2_0_gguf_rows_multi_avx2_legacy(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                                      float* const* out, int r0, int r1) {
    q2_0_gguf_rows_multi_avx2_v(STRATA_AVXVNNI && cpu_avxvnni_ok(), w, row_bytes, nblocks, a, nt, out, r0, r1);
}

void q2_0_gguf_rows_multi_avx2(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                               float* const* out, int r0, int r1) {
    bool bp = kBitplane && nblocks % 2 == 0;
    for (int t = 0; t < nt && bp; ++t) bp = a[t]->bp_pairs == nblocks / 2;
    if (!bp) {
        q2_0_gguf_rows_multi_avx2_legacy(w, row_bytes, nblocks, a, nt, out, r0, r1);
        return;
    }
    const int np = nblocks / 2;
    switch (nt) {
        case 1: rows_bp<1>(w, row_bytes, np, a, out, r0, r1); break;
        case 2: rows_bp<2>(w, row_bytes, np, a, out, r0, r1); break;
        case 3: rows_bp<3>(w, row_bytes, np, a, out, r0, r1); break;
        case 4: rows_bp<4>(w, row_bytes, np, a, out, r0, r1); break;
        default:
            for (int t0 = 0; t0 < nt; t0 += 4) {
                const int k = nt - t0 < 4 ? nt - t0 : 4;
                q2_0_gguf_rows_multi_avx2(w, row_bytes, nblocks, a + t0, k, out + t0, r0, r1);
            }
    }
}

namespace {
// ActQ::qp / psum / pscale from q and scale (see expert.hpp).  One 64-value block: four 16-byte groups, each
// shuffled to plane-major (4 bytes per plane), then a 4x4 dword transpose gives the four 16-byte planes.
void bitplane_image(ActQ& a) {
    if (a.nchunks % 4 != 0) { a.bp_pairs = 0; return; }
    const int nblk = a.nchunks / 2;
    const __m128i shuf = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    const __m128i bias = _mm_set1_epi8((char) 0x80);
    for (int b = 0; b < nblk; ++b) {
        const int8_t* q = a.q + b * 64;
        __m128i s[4];
        for (int g = 0; g < 4; ++g) {
            const __m128i v = _mm_loadu_si128((const __m128i*) (q + 16 * g));
            s[g] = _mm_shuffle_epi8(v, shuf);
            // sum of 16 signed bytes: sad of (q ^ 0x80) = sum(q + 128), two 8-byte halves
            const __m128i sad = _mm_sad_epu8(_mm_xor_si128(v, bias), _mm_setzero_si128());
            a.psum[(b >> 1) * 8 + (b & 1) * 4 + g] =
                _mm_cvtsi128_si32(sad) + _mm_extract_epi16(sad, 4) - 16 * 128;
        }
        const __m128i t0 = _mm_unpacklo_epi32(s[0], s[1]), t1 = _mm_unpacklo_epi32(s[2], s[3]);
        const __m128i t2 = _mm_unpackhi_epi32(s[0], s[1]), t3 = _mm_unpackhi_epi32(s[2], s[3]);
        int8_t* dst = a.qp + (b >> 1) * 128 + (b & 1) * 16;
        _mm_store_si128((__m128i*) (dst), _mm_unpacklo_epi64(t0, t1));
        _mm_store_si128((__m128i*) (dst + 32), _mm_unpackhi_epi64(t0, t1));
        _mm_store_si128((__m128i*) (dst + 64), _mm_unpacklo_epi64(t2, t3));
        _mm_store_si128((__m128i*) (dst + 96), _mm_unpackhi_epi64(t2, t3));
    }
    for (int c = 0; c < a.nchunks; ++c) {
        a.pscale[2 * c] = a.scale[c];
        a.pscale[2 * c + 1] = a.scale[c];
    }
    a.bp_pairs = nblk / 2;
}
}  // namespace

void act_quant_q8_1_avx2(const float* x, int n, ActQ& a) {
    a.nchunks = n / QKA;
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 half = _mm256_set1_ps(0.5f), mhalf = _mm256_set1_ps(-0.5f), zero = _mm256_setzero_ps();
    const __m256i lo = _mm256_set1_epi32(-127), hi = _mm256_set1_epi32(127);
    for (int k = 0; k < a.nchunks; ++k) {
        const float* xb = x + k * QKA;
        __m256 v[4];
        __m256 m = _mm256_setzero_ps();
        for (int i = 0; i < 4; ++i) {
            v[i] = _mm256_loadu_ps(xb + 8 * i);
            m = _mm256_max_ps(m, _mm256_and_ps(v[i], absmask));
        }
        __m128 h = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        h = _mm_max_ps(h, _mm_movehl_ps(h, h));
        const float amax = _mm_cvtss_f32(_mm_max_ss(h, _mm_movehdup_ps(h)));
        const float s = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = s > 0.f ? 1.f / s : 0.f;
        const __m256 vinv = _mm256_set1_ps(inv);
        __m256i sum = _mm256_setzero_si256();
        alignas(32) int32_t qi[QKA];
        for (int i = 0; i < 4; ++i) {
            const __m256 t = _mm256_mul_ps(v[i], vinv);
            const __m256 r = _mm256_add_ps(t, _mm256_blendv_ps(mhalf, half, _mm256_cmp_ps(t, zero, _CMP_GE_OQ)));
            __m256i q = _mm256_cvttps_epi32(r);
            q = _mm256_min_epi32(_mm256_max_epi32(q, lo), hi);
            sum = _mm256_add_epi32(sum, q);
            _mm256_store_si256((__m256i*) (qi + 8 * i), q);
        }
        for (int j = 0; j < QKA; ++j) a.q[k * QKA + j] = (int8_t) qi[j];
        __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
        s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));
        s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xB1));
        const int32_t total = _mm_cvtsi128_si32(s4);
        a.scale[k] = s;
        a.sum[k] = total;
        a.hx[k] = s * (float) total;
    }
    if (kBitplane) bitplane_image(a);   // the image costs a pass per token: only when its kernel will read it
    else a.bp_pairs = 0;
}

}  // namespace strata::kernels::cpu
