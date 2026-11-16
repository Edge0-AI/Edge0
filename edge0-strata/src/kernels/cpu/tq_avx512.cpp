// src/kernels/cpu/tq_avx512.cpp - the trellis expert rows (TQ2_T / TQK6 / TQK7) on AVX-512 F/BW/VL/VNNI
// (tq_cpu.hpp).  Reached behind cpu_avx512_ok() only.
//
// Per 128-weight block: the AVX2 kernel's decode (scalar states, scalar 8-byte state-table loads, one ZMM of 32
// int16 weights per Q8_0 sub-block); per token four VPDPWSSD against the unpacked int16 activations, an unpack/add
// tree to the four exact sub-block sums S[k] in one XMM, and the scalar kernel's float steps lane for lane.  The
// weights are decoded once per block for all the tokens of the group.
//
// Measured on Zen 5 (cycles per block): the decode bounds it, so one token is no faster than AVX2 (~48); each extra
// token of a group costs ~10% less than on AVX2.  ggml-cpu's AVX-512 decode for these types - all states from two
// VPERMB and the table rows by VPGATHERQQ - was 20-30% slower here: Zen gathers are slow, and so are Intel ones
// under the Gather Data Sampling microcode mitigation.
#include "strata/kernels/cpu/tq_cpu.hpp"

#include "tq_states.hpp"

#include <immintrin.h>

namespace strata::kernels::cpu::tq_detail {
namespace {

// four ZMM of 16 int32 partial sums -> [S0, S1, S2, S3] (exact integer adds)
inline __m128i reduce4(__m512i p0, __m512i p1, __m512i p2, __m512i p3) {
    const __m512i a = _mm512_add_epi32(_mm512_unpacklo_epi32(p0, p1), _mm512_unpackhi_epi32(p0, p1));
    const __m512i b = _mm512_add_epi32(_mm512_unpacklo_epi32(p2, p3), _mm512_unpackhi_epi32(p2, p3));
    const __m512i c = _mm512_add_epi32(_mm512_unpacklo_epi64(a, b), _mm512_unpackhi_epi64(a, b));
    // c: per 128-bit lane [p0, p1, p2, p3] partial sums; add the four lanes
    const __m256i d = _mm256_add_epi32(_mm512_castsi512_si256(c), _mm512_extracti64x4_epi64(c, 1));
    return _mm_add_epi32(_mm256_castsi256_si128(d), _mm256_extracti128_si256(d, 1));
}

// four state-table rows (8 bytes each) into one YMM: scalar loads, broadcast and blended in; two make a ZMM
inline __m256i rows4(const long long* tab, uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3) {
    __m256i v = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) (tab + s0)));
    v = _mm256_blend_epi32(v, _mm256_set1_epi64x(tab[s1]), 0x0C);
    v = _mm256_blend_epi32(v, _mm256_set1_epi64x(tab[s2]), 0x30);
    return _mm256_blend_epi32(v, _mm256_set1_epi64x(tab[s3]), 0xC0);
}

template <int KB>
inline __m512i rows8(const long long* tab, const uint8_t* qs, uint64_t wr, int t0) {
    const __m256i lo = rows4(tab, tq_state<KB>(qs, wr, t0), tq_state<KB>(qs, wr, t0 + 1), tq_state<KB>(qs, wr, t0 + 2),
                             tq_state<KB>(qs, wr, t0 + 3));
    const __m256i hi = rows4(tab, tq_state<KB>(qs, wr, t0 + 4), tq_state<KB>(qs, wr, t0 + 5),
                             tq_state<KB>(qs, wr, t0 + 6), tq_state<KB>(qs, wr, t0 + 7));
    return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}

template <int KB>
void rows_t(const int16_t* table, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt, float* const* out, int r0,
            int r1) {
    const int nb = n / 128;
    constexpr size_t bsz = 2 + 4 * KB;
    const long long* tab = (const long long*) table;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* x = w + (size_t) r * row_bytes;
        __m128 acc[kTqMaxTokens];
        for (int t = 0; t < nt; ++t) acc[t] = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* blk = x + (size_t) i * bsz;
            const uint8_t* qs = blk + 2;
            const uint64_t wr = tq_wrap<KB>(qs);
            // states 8k .. 8k+7 -> weights 32k .. 32k+31 = Q8_0 sub-block k
            const __m512i w0 = rows8<KB>(tab, qs, wr, 0), w1 = rows8<KB>(tab, qs, wr, 8);
            const __m512i w2 = rows8<KB>(tab, qs, wr, 16), w3 = rows8<KB>(tab, qs, wr, 24);
            const __m128 dx = _mm_set1_ps(tq_half_to_float(tq_ld16(blk)));
            for (int t = 0; t < nt; ++t) {
                const __m512i* q = (const __m512i*) (a[t]->q + 128 * i);
                const __m512i z = _mm512_setzero_si512();
                const __m128i s = reduce4(_mm512_dpwssd_epi32(z, w0, _mm512_load_si512(q + 0)),
                                          _mm512_dpwssd_epi32(z, w1, _mm512_load_si512(q + 1)),
                                          _mm512_dpwssd_epi32(z, w2, _mm512_load_si512(q + 2)),
                                          _mm512_dpwssd_epi32(z, w3, _mm512_load_si512(q + 3)));
                const __m128 sc = _mm_mul_ps(dx, _mm_loadu_ps(a[t]->d + 4 * i));
                acc[t] = _mm_fmadd_ps(_mm_cvtepi32_ps(s), sc, acc[t]);
            }
        }
        for (int t = 0; t < nt; ++t) {
            alignas(16) float v[4];
            _mm_store_ps(v, acc[t]);
            out[t][r] = ((v[0] + v[1]) + (v[2] + v[3])) * (1.0f / 8192.0f);
        }
    }
}

}  // namespace

void tq_rows_avx512(int kb, const int16_t* table, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                    float* const* out, int r0, int r1) {
    if (kb == 8) rows_t<8>(table, w, row_bytes, n, a, nt, out, r0, r1);
    else if (kb == 6) rows_t<6>(table, w, row_bytes, n, a, nt, out, r0, r1);
    else if (kb == 5) rows_t<5>(table, w, row_bytes, n, a, nt, out, r0, r1);   // experimental TQK5X
    else if (kb == 4) rows_t<4>(table, w, row_bytes, n, a, nt, out, r0, r1);   // experimental TQK4X
    else rows_t<7>(table, w, row_bytes, n, a, nt, out, r0, r1);
}

}  // namespace strata::kernels::cpu::tq_detail
