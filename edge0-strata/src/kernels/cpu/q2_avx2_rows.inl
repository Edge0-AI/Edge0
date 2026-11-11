// src/kernels/cpu/q2_avx2_rows.inl - the Q2_0 row kernel of q2_avx2.cpp, included there twice like
// iq_avx2_rows.inl: the AVX2 form (STRATA_ROWS_VNNI 0, namespace plain) and the AVX-VNNI one (1, namespace vnni), every
// function under STRATA_ROWS_FN (GCC/Clang's target("avxvnni") in the VNNI copy).  Not a header.

// The 2-bit codes (0..3) times the int8 activation, four products per int32 lane.  vpdpbusd is one uop where maddubs
// + madd(ones) take two on the multiply ports; maddubs cannot saturate (a pair is at most 2 * 3 * 128), so the sums
// are the same.
STRATA_ROWS_FN inline __m256i dot4(__m256i codes, __m256i act) {
#if STRATA_ROWS_VNNI
    return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), codes, act);
#else
    return _mm256_madd_epi16(_mm256_maddubs_epi16(codes, act), _mm256_set1_epi16(1));
#endif
}

template <int NT> STRATA_ROWS_FN
inline void row_multi(const uint8_t* row, const ActQ* const* a, int nblocks, float* res) {
    __m256 acc[NT];
    float corr[NT];
    for (int t = 0; t < NT; ++t) { acc[t] = _mm256_setzero_ps(); corr[t] = 0.f; }
    for (int b = 0; b < nblocks; ++b) {
        const uint8_t* blk = row + (size_t) b * 18;
        const float d = h2f(blk);
        __m256i lo, hi;
        unpack64(blk + 2, lo, hi);
        for (int t = 0; t < NT; ++t) {
            const int8_t* q = a[t]->q + b * 64;
            const __m256i s0 = dot4(lo, _mm256_loadu_si256((const __m256i*) q));
            const __m256i s1 = dot4(hi, _mm256_loadu_si256((const __m256i*) (q + 32)));
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * a[t]->scale[2 * b]), _mm256_cvtepi32_ps(s0), acc[t]);
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * a[t]->scale[2 * b + 1]), _mm256_cvtepi32_ps(s1), acc[t]);
            corr[t] += d * (a[t]->hx[2 * b] + a[t]->hx[2 * b + 1]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        const __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc[t]), _mm256_extractf128_ps(acc[t], 1));
        const __m128 s = _mm_add_ps(h, _mm_movehl_ps(h, h));
        res[t] = _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s))) - corr[t];
    }
}

template <int NT> STRATA_ROWS_FN
void rows(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, float* const* out, int r0, int r1) {
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_multi<NT>(w + (size_t) r * row_bytes, a, nblocks, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

STRATA_ROWS_FN void rows_nt(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                            float* const* out, int r0, int r1) {
    switch (nt) {
        case 1: rows<1>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 2: rows<2>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 3: rows<3>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 4: rows<4>(w, row_bytes, nblocks, a, out, r0, r1); break;
        default:
            for (int t0 = 0; t0 < nt; t0 += 4) {
                const int k = nt - t0 < 4 ? nt - t0 : 4;
                rows_nt(w, row_bytes, nblocks, a + t0, k, out + t0, r0, r1);
            }
    }
}

#undef STRATA_ROWS_VNNI
#undef STRATA_ROWS_FN
