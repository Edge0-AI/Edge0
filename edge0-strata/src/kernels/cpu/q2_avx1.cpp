// src/kernels/cpu/q2_avx1.cpp - the Q2_0 expert rows and the activation quantizer for CPUs that have AVX but
// no AVX2 (Sandy Bridge through Ivy Bridge Xeons; issue #623, #1375), the third rung of q2_rows_any /
// act_quant_any below the AVX-2 one (issue #1699: those dispatchers used to hand AVX2 code to an AVX1 CPU).
//
// The arithmetic is q2_avx2.cpp's: codes 0..3 against the int8 activation per 32-value chunk, times the weight
// scale and the chunk scale, minus the weight scale times the chunk's `hx`.  128-bit integer lanes (maddubs and
// madd are SSSE3/SSE2), 256-bit float accumulation with mul+add (no FMA), fp16 scales converted in software (no
// F16C).  The legacy row kernel keeps q2_avx2.cpp's lane pairing and reduction tree exactly, so the only
// difference from an AVX2 CPU is its FMA becoming mul+add (last bits, unavoidable without FMA); the bit-plane
// kernel's float summation order is its own (last bits, as documented for the AVX-2 one), and every token's
// operations are the same whatever the window width (#152).
//
// Compiled with -msse4.2 -mavx -mno-avx2 -mno-fma -mno-f16c (CMakeLists.txt, the iq_avx1.cpp pattern) and called
// only behind cpu_avx1_ok() && !cpu_avx2_ok(), so no CPU that lacks those features can reach it - and no CPU that
// has AVX2 changes what it runs.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(__AVX2__) || defined(__FMA__) || defined(__F16C__)
#error "q2_avx1.cpp must be compiled without AVX2, FMA and F16C (see the per-source flags in CMakeLists.txt)"
#endif

namespace strata::kernels::cpu {
namespace {

// fp16 -> fp32 without F16C (the written-out conversion, subnormals included; iq_avx1.cpp's copy).
inline float h2f(uint16_t h) {
    const uint32_t sign = (uint32_t) (h >> 15) & 1u;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, f;
    if (exp == 0) {
        if (man == 0) {
            f = sign << 31;
        } else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400u)) { man <<= 1; --exp; }
            man &= 0x3FFu;
            f = (sign << 31) | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000u | (man << 13);
    } else {
        f = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

// 16 bytes of 2-bit codes -> 64 codes in value order, four 16-byte vectors: v0 values 0..15, v1 16..31,
// v2 32..47, v3 48..63 (q2_avx2.cpp's unpack64 cut to 128-bit lanes; the same interleave).
inline void unpack64(const uint8_t* codes, __m128i& v0, __m128i& v1, __m128i& v2, __m128i& v3) {
    const __m128i b = _mm_loadu_si128((const __m128i*) codes);
    const __m128i m3 = _mm_set1_epi8(3);
    const __m128i c0 = _mm_and_si128(b, m3);
    const __m128i c1 = _mm_and_si128(_mm_srli_epi16(b, 2), m3);
    const __m128i c2 = _mm_and_si128(_mm_srli_epi16(b, 4), m3);
    const __m128i c3 = _mm_and_si128(_mm_srli_epi16(b, 6), m3);
    const __m128i a0 = _mm_unpacklo_epi8(c0, c1), a1 = _mm_unpacklo_epi8(c2, c3);   // bytes 0..7
    const __m128i b0 = _mm_unpackhi_epi8(c0, c1), b1 = _mm_unpackhi_epi8(c2, c3);   // bytes 8..15
    v0 = _mm_unpacklo_epi16(a0, a1);   // values 0..15
    v1 = _mm_unpackhi_epi16(a0, a1);   // values 16..31
    v2 = _mm_unpacklo_epi16(b0, b1);   // values 32..47
    v3 = _mm_unpackhi_epi16(b0, b1);   // values 48..63
}

// 16 codes against 16 activations, four products per int32 lane (q2_avx2_rows.inl's dot4 at 128-bit;
// maddubs cannot saturate: a pair is at most 2 * 3 * 128).
inline __m128i dot4(__m128i codes, __m128i act) {
    return _mm_madd_epi16(_mm_maddubs_epi16(codes, act), _mm_set1_epi16(1));
}

// The legacy row kernel.  The accumulator pairing matches q2_avx2.cpp's lane pairing: each 256-bit lane takes
// the s0 contribution (values 0..31) and the s1 contribution (values 32..63) into the SAME lanes, block by
// block, and the reduction tree is the same one ((L0+L4)+(L2+L6)) + ((L1+L5)+(L3+L7)) - the only difference
// from an AVX2 CPU is that kernel's FMA being a mul+add here (last bits).
template <int NT>
inline void row_multi(const uint8_t* row, const ActQ* const* a, int nblocks, float* res) {
    __m256 acc[NT];
    float corr[NT];
    for (int t = 0; t < NT; ++t) { acc[t] = _mm256_setzero_ps(); corr[t] = 0.f; }
    for (int b = 0; b < nblocks; ++b) {
        const uint8_t* blk = row + (size_t) b * 18;
        const float d = h2f((uint16_t) (blk[0] | (blk[1] << 8)));
        __m128i v0, v1, v2, v3;
        unpack64(blk + 2, v0, v1, v2, v3);
        for (int t = 0; t < NT; ++t) {
            const int8_t* q = a[t]->q + b * 64;
            // s0 = values 0..31: lanes 0..3 of accA's 128-bit halves; s1 = values 32..63 into the SAME lanes
            const __m128i s0a = dot4(v0, _mm_loadu_si128((const __m128i*) q));        // values 0..15
            const __m128i s0b = dot4(v1, _mm_loadu_si128((const __m128i*) (q + 16))); // values 16..31
            const __m128i s1a = dot4(v2, _mm_loadu_si128((const __m128i*) (q + 32))); // values 32..47
            const __m128i s1b = dot4(v3, _mm_loadu_si128((const __m128i*) (q + 48))); // values 48..63
            const __m256 s0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_cvtepi32_ps(s0a)), _mm_cvtepi32_ps(s0b), 1);
            const __m256 s1 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_cvtepi32_ps(s1a)), _mm_cvtepi32_ps(s1b), 1);
            // exactly the AVX-2 kernel's two fmadds: the whole s0 (values 0..31) by scale[2b], the whole s1
            // (values 32..63) by scale[2b+1], into the same lanes - lane pairing and reduce tree follow suit
            acc[t] = _mm256_add_ps(acc[t], _mm256_mul_ps(_mm256_set1_ps(d * a[t]->scale[2 * b]), s0));
            acc[t] = _mm256_add_ps(acc[t], _mm256_mul_ps(_mm256_set1_ps(d * a[t]->scale[2 * b + 1]), s1));
            corr[t] += d * (a[t]->hx[2 * b] + a[t]->hx[2 * b + 1]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        const __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc[t]), _mm256_extractf128_ps(acc[t], 1));
        const __m128 s = _mm_add_ps(h, _mm_movehl_ps(h, h));
        res[t] = _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s))) - corr[t];
    }
}

template <int NT>
void rows(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, float* const* out, int r0, int r1) {
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_multi<NT>(w + (size_t) r * row_bytes, a, nblocks, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

// ---- the bit-plane row kernel (STRATA_Q2_BITPLANE=1), 128-bit lanes: one 64-value block at a time instead of
// a pair per 256-bit register.  The integer part is exact; the float summation order is this kernel's own (the
// same statement as for the AVX-2 one), and every row reduces with the same tree alone or in a group (#152).
template <int NT>
inline void row_bp_acc(const uint8_t* row, const ActQ* const* a, int npairs, __m128* acc) {
    for (int t = 0; t < NT; ++t) { acc[2 * t] = _mm_setzero_ps(); acc[2 * t + 1] = _mm_setzero_ps(); }
    const __m128i m3 = _mm_set1_epi8(3);
    const __m128i ones = _mm_set1_epi16(1);
    for (int p = 0; p < npairs; ++p) {
        const uint8_t* blk = row + (size_t) p * 36;
        for (int h = 0; h < 2; ++h) {
            uint16_t dh;
            std::memcpy(&dh, blk + h * 18, 2);          // the block's fp16 scale: bytes 0..1 (block 1 at +18)
            const __m128 d = _mm_set1_ps(h2f(dh));
            const __m128i c = _mm_loadu_si128((const __m128i*) (blk + 2 + h * 18));   // its 16 code bytes
            const __m128i c0 = _mm_and_si128(c, m3);
            const __m128i c1 = _mm_and_si128(_mm_srli_epi16(c, 2), m3);
            const __m128i c2 = _mm_and_si128(_mm_srli_epi16(c, 4), m3);
            const __m128i c3 = _mm_and_si128(_mm_srli_epi16(c, 6), m3);
            for (int t = 0; t < NT; ++t) {
                const int8_t* q = a[t]->qp + (size_t) p * 128 + h * 16;
                __m128i s = _mm_maddubs_epi16(c0, _mm_load_si128((const __m128i*) q));
                s = _mm_add_epi16(s, _mm_maddubs_epi16(c1, _mm_load_si128((const __m128i*) (q + 32))));
                s = _mm_add_epi16(s, _mm_maddubs_epi16(c2, _mm_load_si128((const __m128i*) (q + 64))));
                s = _mm_add_epi16(s, _mm_maddubs_epi16(c3, _mm_load_si128((const __m128i*) (q + 96))));
                const __m128i i32 = _mm_sub_epi32(_mm_madd_epi16(s, ones),
                                                  _mm_load_si128((const __m128i*) (a[t]->psum + p * 8 + h * 4)));
                const __m128 v = _mm_mul_ps(_mm_cvtepi32_ps(i32),
                                            _mm_load_ps(a[t]->pscale + p * 8 + h * 4));
                __m128& x = acc[2 * t + h];
                x = _mm_add_ps(x, _mm_mul_ps(d, v));
            }
        }
    }
}

inline float reduce1(__m128 a, __m128 b) {   // ((v0+v1)+(v2+v3)) + ((v4+v5)+(v6+v7)), the AVX-2 kernel's tree
    __m128 ha = _mm_hadd_ps(a, a);                        // [a0+a1, a2+a3, a0+a1, a2+a3]
    ha = _mm_add_ss(ha, _mm_movehdup_ps(ha));             // (v0+v1)+(v2+v3)
    __m128 hb = _mm_hadd_ps(b, b);
    hb = _mm_add_ss(hb, _mm_movehdup_ps(hb));             // (v4+v5)+(v6+v7)
    return _mm_cvtss_f32(_mm_add_ss(ha, hb));
}

template <int NT>
void rows_bp(const uint8_t* w, size_t row_bytes, int npairs, const ActQ* const* a, float* const* out, int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
        __m128 acc[2 * NT];
        row_bp_acc<NT>(w + (size_t) r * row_bytes, a, npairs, acc);
        for (int t = 0; t < NT; ++t) out[t][r] = reduce1(acc[2 * t], acc[2 * t + 1]);
    }
}

const bool kBitplane = [] {
    const char* e = std::getenv("STRATA_Q2_BITPLANE");
    return e != nullptr && e[0] == '1';
}();

// ActQ::qp / psum / pscale from q and scale: q2_avx2.cpp's bitplane_image, which is already pure SSE2/SSSE3
// (shuffle_epi8, sad, unpack) - copied rather than shared so q2_avx2.cpp stays untouched.
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

void q2_0_gguf_rows_multi_avx1_legacy(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                                      float* const* out, int r0, int r1) {
    switch (nt) {
        case 1: rows<1>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 2: rows<2>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 3: rows<3>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 4: rows<4>(w, row_bytes, nblocks, a, out, r0, r1); break;
        default:
            for (int t0 = 0; t0 < nt; t0 += 4) {
                const int k = nt - t0 < 4 ? nt - t0 : 4;
                q2_0_gguf_rows_multi_avx1_legacy(w, row_bytes, nblocks, a + t0, k, out + t0, r0, r1);
            }
    }
}

void q2_0_gguf_rows_multi_avx1(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                               float* const* out, int r0, int r1) {
    bool bp = kBitplane && nblocks % 2 == 0;
    for (int t = 0; t < nt && bp; ++t) bp = a[t]->bp_pairs == nblocks / 2;
    if (!bp) {
        q2_0_gguf_rows_multi_avx1_legacy(w, row_bytes, nblocks, a, nt, out, r0, r1);
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
                q2_0_gguf_rows_multi_avx1(w, row_bytes, nblocks, a + t0, k, out + t0, r0, r1);
            }
    }
}

void act_quant_q8_1_avx1(const float* x, int n, ActQ& a) {
    a.nchunks = n / QKA;
    const __m128 absmask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
    const __m128 half = _mm_set1_ps(0.5f), mhalf = _mm_set1_ps(-0.5f), zero = _mm_setzero_ps();
    const __m128i lo = _mm_set1_epi32(-127), hi = _mm_set1_epi32(127);
    for (int k = 0; k < a.nchunks; ++k) {
        const float* xb = x + k * QKA;
        __m128 v[8];
        __m128 m = _mm_setzero_ps();
        for (int i = 0; i < 8; ++i) {
            v[i] = _mm_loadu_ps(xb + 4 * i);
            m = _mm_max_ps(m, _mm_and_ps(v[i], absmask));
        }
        m = _mm_max_ps(m, _mm_movehl_ps(m, m));
        const float amax = _mm_cvtss_f32(_mm_max_ss(m, _mm_movehdup_ps(m)));
        const float s = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = s > 0.f ? 1.f / s : 0.f;
        const __m128 vinv = _mm_set1_ps(inv);
        __m128i sum = _mm_setzero_si128();
        alignas(16) int32_t qi[QKA];
        for (int i = 0; i < 8; ++i) {
            const __m128 t = _mm_mul_ps(v[i], vinv);
            const __m128 r = _mm_add_ps(t, _mm_blendv_ps(mhalf, half, _mm_cmp_ps(t, zero, _CMP_GE_OQ)));
            __m128i q = _mm_cvttps_epi32(r);
            q = _mm_min_epi32(_mm_max_epi32(q, lo), hi);
            sum = _mm_add_epi32(sum, q);
            _mm_store_si128((__m128i*) (qi + 4 * i), q);
        }
        for (int j = 0; j < QKA; ++j) a.q[k * QKA + j] = (int8_t) qi[j];
        __m128i s4 = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4E));
        s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xB1));
        const int32_t total = _mm_cvtsi128_si32(s4);
        a.scale[k] = s;
        a.sum[k] = total;
        a.hx[k] = s * (float) total;
    }
    if (kBitplane) bitplane_image(a);
    else a.bp_pairs = 0;
}

}  // namespace strata::kernels::cpu
