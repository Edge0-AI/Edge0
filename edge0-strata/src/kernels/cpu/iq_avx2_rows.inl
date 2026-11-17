// src/kernels/cpu/iq_avx2_rows.inl - the row kernels of iq_avx2.cpp.  Not a header: iq_avx2.cpp includes it twice,
//
//   STRATA_ROWS_VNNI 0   the AVX2 forms (namespace plain);
//   STRATA_ROWS_VNNI 1   the same loops with AVX-VNNI (namespace vnni, built with STRATA_AVXVNNI): vpdpwssd for each
//                        madd + add, vpdpbusd for maddubs + madd(ones).  Integer sums, so the same bits.
//
// Every function here is STRATA_ROWS_FN: nothing in the plain copy, GCC/Clang's target("avxvnni") in the VNNI copy, so
// the AVX-VNNI intrinsics compile in these functions only and the plain copy stays AVX2 (-mavxvnni on the whole file
// would let the compiler use the extension anywhere).  MSVC takes the intrinsics under /arch:AVX2.  The VNNI copy is
// called only where cpu_avxvnni_ok().

// acc + madd(p16, sc).  vpdpwssd is the madd and the add in one uop; both wrap the same way.
STRATA_ROWS_FN inline __m256i madd_add(__m256i acc, __m256i p16, __m256i sc) {
#if STRATA_ROWS_VNNI
    return _mm256_dpwssd_avx_epi32(acc, p16, sc);
#else
    return _mm256_add_epi32(acc, _mm256_madd_epi16(p16, sc));
#endif
}

// Four u8 x s8 products per int32 lane.  vpdpbusd does not saturate where maddubs would, but maddubs cannot here: the
// u8 side is an IQ4_NL |w| <= 127, so a pair is at most 2 * 127 * 128.
STRATA_ROWS_FN inline __m256i dot4u(__m256i u8, __m256i s8) {
#if STRATA_ROWS_VNNI
    return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), u8, s8);
#else
    return _mm256_madd_epi16(_mm256_maddubs_epi16(u8, s8), _mm256_set1_epi16(1));
#endif
}

STRATA_ROWS_FN inline float ggml_hsum8(__m256 x) {
    __m128 s = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

template <int NT, bool GGML = false> STRATA_ROWS_FN
inline void row_dot_iq2xs(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res);

template <int TY> STRATA_ROWS_FN
inline float row_dot_ggml_one(const uint8_t* row, int nblocks, const block_q8_K* y) {
    if constexpr (TY == 17) {
        const block_q8_K* act[1] = {y};
        float result = 0.0f;
        row_dot_iq2xs<1, true>(row, nblocks, act, &result);
        return result;
    }
    __m256 accum = _mm256_setzero_ps();
    const int pf = prefetch_distance();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * Fmt32<TY>::bytes;
        rows_ahead(blk, pf);
        __m256i sums[2] = {_mm256_setzero_si256(), _mm256_setzero_si256()};
        for (int j = 0; j < 4; ++j)
            for (int half = 0; half < 2; ++half) {
                __m256i g, sign, scale;
                Fmt32<TY>::decode(blk, j, half, g, sign, scale);
                const __m256i act = _mm256_loadu_si256((const __m256i*) (y[i].qs + 64 * j + 32 * half));
                const __m256i signed_act = _mm256_sign_epi8(act, sign);
                sums[half] = madd_add(sums[half], _mm256_maddubs_epi16(g, signed_act), scale);
            }
        const float d = h2f(u16(blk)) * y[i].d;
        accum = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sums[0], sums[1])), accum);
    }
    if constexpr (TY == 16 || TY == 17 || TY == 22 || TY == 122) return 0.125f * ggml_hsum8(accum);
    else if constexpr (TY == 18 || TY == 118) return 0.25f * ggml_hsum8(accum);
    else return ggml_hsum8(accum);
}

template <int TY> STRATA_ROWS_FN
void gu_ggml_one(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const block_q8_K* y, float* ff,
                 int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
        const float g = row_dot_ggml_one<TY>(blob + (size_t) r * gu_row, n / QK_K, y);
        const float u = row_dot_ggml_one<TY>(blob + up_off + (size_t) r * gu_row, n / QK_K, y);
        ff[r] = (g / (1.f + std::exp(-g))) * u;
    }
}

template <int TY, int NT, bool EX = false> STRATA_ROWS_FN
inline void row_dot(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    const int pf = prefetch_distance();
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * Fmt32<TY>::bytes;
        rows_ahead(blk, pf);
        // EX: independent integer accumulators a token (A of them) cut the dependent madd+add chain from eight steps to 8/A -
        // few tokens have too little else in flight to hide it (integer sums, the same bits)
        constexpr int A = EX ? (NT == 1 ? 4 : NT == 2 ? 2 : 1) : 1;
        __m256i acci[NT][A];
        for (int t = 0; t < NT; ++t)
            for (int a = 0; a < A; ++a) acci[t][a] = _mm256_setzero_si256();
        for (int j = 0; j < 4; ++j) {
            for (int half = 0; half < 2; ++half) {
                __m256i g, sgn, sc;
                Fmt32<TY>::decode(blk, j, half, g, sgn, sc);
                const int off = 64 * j + 32 * half;
                for (int t = 0; t < NT; ++t) {
                    const __m256i yv = _mm256_loadu_si256((const __m256i*) (y[t][i].qs + off));
                    const __m256i ys = _mm256_sign_epi8(yv, sgn);
                    acci[t][(2 * j + half) % A] = madd_add(acci[t][(2 * j + half) % A], _mm256_maddubs_epi16(g, ys), sc);
                }
            }
        }
        for (int t = 0; t < NT; ++t)
            for (int a = 1; a < A; ++a) acci[t][0] = _mm256_add_epi32(acci[t][0], acci[t][a]);
        // EX (the DeepSeek CPU lane's rows): ggml's float order, so a token's dot is the same bits as ggml-cpu's whatever
        // number of tokens share the pass - block scale f16(d) * y.d, the format's constant K after the sum, ggml's hsum.
        // Without EX the order of the main engine's rows (unchanged).
        const float dx = EX ? h2f(u16(blk)) : h2f(u16(blk)) * Fmt32<TY>::K;
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(acci[t][0]), accf[t]);
    }
    for (int t = 0; t < NT; ++t) res[t] = EX ? Fmt32<TY>::K * ggml_hsum8(accf[t]) : hsum8(accf[t]);
}

// ---- IQ2_XS (17) with ggml's vectorized sign decode (arch/x86/quants.c, ggml_vec_dot_iq2_xs_q8_K):
// the 7-bit sign indices never touch ksigns_iq2xs - the 8th sign bit is reconstructed by parity (two
// shifts, a xor and ggml's bit_helper pshufb) and the per-half sign vectors come out of shuffles of one
// 32-byte load.  The 8 scale bytes become all 16 half-scales with ggml's unpack trick.  The grid lookups
// stay scalar loads into set_epi64x (ggml does the same).  Every token still only pays a load, a sign,
// a maddubs, a madd and an add per half, into alternating accumulators.
template <int NT, bool GGML> STRATA_ROWS_FN
inline void row_dot_iq2xs(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    static const uint8_t bit_sel[32] = {
        1, 2, 4, 8, 16, 32, 64, (uint8_t) 0x80, 1, 2, 4, 8, 16, 32, 64, (uint8_t) 0x80,
        1, 2, 4, 8, 16, 32, 64, (uint8_t) 0x80, 1, 2, 4, 8, 16, 32, 64, (uint8_t) 0x80 };
    static const char shuf_even[32] = {   // sign bits of u16 0,1,2,3 (even bytes of the sign register)
        0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2,
        4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6 };
    static const char shuf_odd[32] = {    // sign bits of u16 4,5,6,7
        8, 8, 8, 8, 8, 8, 8, 8, 10, 10, 10, 10, 10, 10, 10, 10,
        12, 12, 12, 12, 12, 12, 12, 12, 14, 14, 14, 14, 14, 14, 14, 14 };
    static const uint8_t bit_help[32] = {  // 0x80 when the 4-bit index has odd popcount (parity bit 7)
        0x00, (uint8_t) 0x80, (uint8_t) 0x80, 0x00, (uint8_t) 0x80, 0x00, 0x00, (uint8_t) 0x80,
        (uint8_t) 0x80, 0x00, 0x00, (uint8_t) 0x80, 0x00, (uint8_t) 0x80, (uint8_t) 0x80, 0x00,
        0x00, (uint8_t) 0x80, (uint8_t) 0x80, 0x00, (uint8_t) 0x80, 0x00, 0x00, (uint8_t) 0x80,
        (uint8_t) 0x80, 0x00, 0x00, (uint8_t) 0x80, 0x00, (uint8_t) 0x80, (uint8_t) 0x80, 0x00 };
    static const uint8_t k_sc_shuffle[128] = {  // half h -> scale bytes 2h (lanes 0-7), 2h+1 (lanes 8-15)
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
        8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11,
        12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13,
        14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15 };
    const __m256i bsel = _mm256_loadu_si256((const __m256i*) bit_sel);
    const __m256i shev = _mm256_loadu_si256((const __m256i*) shuf_even);
    const __m256i shod = _mm256_loadu_si256((const __m256i*) shuf_odd);
    const __m256i bhelp = _mm256_loadu_si256((const __m256i*) bit_help);
    const __m256i m511 = _mm256_set1_epi16(511);
    const __m128i m4 = _mm_set1_epi8(0xf), m1 = _mm_set1_epi8(1);
    const __m256i one8 = _mm256_set1_epi8(1);
    const int pf = prefetch_distance();

    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * 74;
        rows_ahead(blk, pf);
        // the 8 scale bytes -> 16 half-scales of 2*s+1, interleaved [a0, b0, a1, b1, ...] (ggml's unpack)
        __m128i st = _mm_set1_epi64x((long long) u64(blk + 66));
        st = _mm_unpacklo_epi8(_mm_and_si128(st, m4), _mm_and_si128(_mm_srli_epi16(st, 4), m4));
        const __m128i scales = _mm_add_epi8(_mm_slli_epi16(st, 1), m1);
        constexpr int CH = GGML ? 4 : 2;   // (EX: four independent integer chains a token)
        __m256i acc[NT][CH];
        for (int t = 0; t < NT; ++t)
            for (int c = 0; c < CH; ++c) acc[t][c] = _mm256_setzero_si256();
        for (int jj = 0; jj < 2; ++jj) {   // 128 values (16 u16) per step
            const __m256i q2 = _mm256_loadu_si256((const __m256i*) (blk + 2 + 32 * jj));
            const __m256i p7 = _mm256_srli_epi16(q2, 9);              // 7 sign bits per u16, even bytes
            const __m256i p3 = _mm256_srli_epi16(q2, 13);             // sign index bits 4-6
            const __m256i fsb = _mm256_or_si256(p7,
                _mm256_shuffle_epi8(bhelp, _mm256_xor_si256(p7, p3)));  // + the parity sign bit 7
            const __m256i s01 = _mm256_broadcastsi128_si256(_mm256_castsi256_si128(fsb)); // u16 0-7
            const __m256i s23 = _mm256_broadcastsi128_si256(_mm256_extracti128_si256(fsb, 1)); // u16 8-15
            alignas(32) uint16_t gi[16];
            _mm256_store_si256((__m256i*) gi, _mm256_and_si256(q2, m511));
            for (int h = 0; h < 4; ++h) {   // the four 32-value halves of this group
                const int H = 4 * jj + h;   // global half index -> scales byte pair
                const __m256i g = _mm256_set_epi64x((long long) iq2xs_grid[gi[4 * h + 3]],
                                                   (long long) iq2xs_grid[gi[4 * h + 2]],
                                                   (long long) iq2xs_grid[gi[4 * h + 1]],
                                                   (long long) iq2xs_grid[gi[4 * h]]);
                const __m256i sb = _mm256_shuffle_epi8(h < 2 ? s01 : s23, (h & 1) ? shod : shev);
                const __m256i sgn = _mm256_or_si256(
                    _mm256_cmpeq_epi8(_mm256_and_si256(sb, bsel), bsel), one8);
                const __m256i sc = _mm256_cvtepi8_epi16(
                    _mm_shuffle_epi8(scales, _mm_loadu_si128((const __m128i*) k_sc_shuffle + H)));
                const int off = 128 * jj + 32 * h;
                for (int t = 0; t < NT; ++t) {
                    const __m256i yv = _mm256_loadu_si256((const __m256i*) (y[t][i].qs + off));
                    const __m256i ys = _mm256_sign_epi8(yv, sgn);
                    acc[t][h % CH] = madd_add(acc[t][h % CH], _mm256_maddubs_epi16(g, ys), sc);
                }
            }
        }
        for (int t = 0; t < NT; ++t) {
            const float d = h2f(u16(blk)) * (GGML ? y[t][i].d : 0.125f);
            const float scale = GGML ? d : d * y[t][i].d;
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(scale),
                _mm256_cvtepi32_ps(CH == 4 ? _mm256_add_epi32(_mm256_add_epi32(acc[t][0], acc[t][1]), _mm256_add_epi32(acc[t][2], acc[t][3]))
                                                       : _mm256_add_epi32(acc[t][0], acc[t][1])), accf[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = GGML ? 0.125f * ggml_hsum8(accf[t]) : hsum8(accf[t]);
}

// ---- IQ1_M (29): 56-byte blocks (qs[32], qh[16], scales[8] holding the 3-bit sub-scales and, in their top nibbles, the
// block's f16 d), ggml's ggml_vec_dot_iq1_m_q8_K (AVX2): s = hsum(sum d*sumi1) + 0.125 * hsum(sum d*sumi2), sumi1 the
// grid dot, sumi2 the +-1 delta dot.  Both are kept lane for lane as ggml has them, so the bits are ggml's.
//  - the grid values are -1/0/1 bytes: |g| and g's sign (the two halves of ggml's mul_add_epi8) are taken once per weight chunk;
//  - the delta dot needs no weight bytes beyond one sign a group of 8: sign(q8, delta) summed in pairs and times the scale is
//    (pair sums of q8) times (+-scale), so the pair sums of every activation chunk are made once per call (iq1m_pre_build) and
//    a token's delta dot is one madd per 32 values;
//  - the indices' high bits and the delta signs of eight groups come out of one 32-bit load by variable shifts (ggml's
//    non-BMI2 path branches on every bit).

// the pair sums maddubs(1, q8) of every 32-value chunk of an activation: 16 int16 a chunk, 8 chunks a block
STRATA_ROWS_FN inline void iq1m_pre_build(const block_q8_K* y, int nblocks, int16_t* p) {
    const __m256i one8 = _mm256_set1_epi8(1);
    for (int i = 0; i < nblocks; ++i)
        for (int c = 0; c < 8; ++c)
            _mm256_storeu_si256((__m256i*) (p + ((size_t) i * 8 + c) * 16),
                                _mm256_maddubs_epi16(one8, _mm256_loadu_si256((const __m256i*) (y[i].qs + 32 * c))));
}

template <int NT> STRATA_ROWS_FN
inline void row_dot_iq1m(const uint8_t* row, int nblocks, const block_q8_K* const* y, const int16_t* const* P, float* res) {
    const __m256i mask = _mm256_set1_epi16(0x7);
    const __m256i mone = _mm256_set1_epi16(1);
    const __m256i mtwo8 = _mm256_set1_epi8(2);
    const __m256i scales_shift = _mm256_set_epi64x(9, 3, 6, 0);
    const __m256i nib_shift = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
    const __m256i m7 = _mm256_set1_epi32(7), m8 = _mm256_set1_epi32(8);
    const int pf = prefetch_distance();
    __m256 acc1[NT], acc2[NT];
    for (int t = 0; t < NT; ++t) acc1[t] = acc2[t] = _mm256_setzero_ps();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * 56;
        rows_ahead(blk, pf);
        const uint8_t* qs = blk;
        const uint8_t* qh = blk + 32;
        const uint8_t* scb = blk + 48;
        const uint16_t sc0 = u16(scb), sc1 = u16(scb + 2), sc2 = u16(scb + 4), sc3 = u16(scb + 6);
        const uint16_t d16 = (uint16_t) ((sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000));
        __m256i scales = _mm256_set1_epi64x((long long) u64(scb));
        scales = _mm256_srlv_epi64(scales, scales_shift);
        scales = _mm256_add_epi16(_mm256_slli_epi16(_mm256_and_si256(scales, mask), 1), mone);
        __m256i sidx1 = _mm256_set1_epi16(0x0100);
        __m256i sidx2 = _mm256_add_epi8(sidx1, _mm256_set1_epi8(8));
        // the block's four 64-value steps decoded once into a small stack frame, then each token runs over them with two
        // accumulators in registers (NT tokens' worth of accumulators and decode together do not fit 16 registers)
        struct Dec { __m256i a1, g1, a2, g2, sa, sb, ssa, ssb; };
        Dec dec[4];
        for (int it = 0; it < 4; ++it) {
            const __m256i v8 = _mm256_srlv_epi32(_mm256_set1_epi32((int) u32(qh + 4 * it)), nib_shift);   // group k's qh nibble
            const __m256i idx = _mm256_or_si256(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (qs + 8 * it))),
                                                _mm256_slli_epi32(_mm256_and_si256(v8, m7), 8));
            alignas(32) int32_t ix[8];
            _mm256_store_si256((__m256i*) ix, idx);
            const __m256i g1 = _mm256_set_epi64x((long long) iq1s_grid[ix[3]], (long long) iq1s_grid[ix[2]],
                                                 (long long) iq1s_grid[ix[1]], (long long) iq1s_grid[ix[0]]);
            const __m256i g2 = _mm256_set_epi64x((long long) iq1s_grid[ix[7]], (long long) iq1s_grid[ix[6]],
                                                 (long long) iq1s_grid[ix[5]], (long long) iq1s_grid[ix[4]]);
            const __m256i neg = _mm256_cmpeq_epi32(_mm256_and_si256(v8, m8), m8);   // delta -1: all ones a group
            const __m256i n1 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(neg));   // 4 int16 lanes a group of 8 values
            const __m256i n2 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(neg, 1));
            const __m256i sc_a = _mm256_shuffle_epi8(scales, sidx1);
            const __m256i sc_b = _mm256_shuffle_epi8(scales, sidx2);
            dec[it] = Dec{_mm256_sign_epi8(g1, g1), g1, _mm256_sign_epi8(g2, g2), g2, sc_a, sc_b,
                          _mm256_sign_epi16(sc_a, _mm256_or_si256(n1, mone)), _mm256_sign_epi16(sc_b, _mm256_or_si256(n2, mone))};
            sidx1 = _mm256_add_epi8(sidx1, mtwo8);
            sidx2 = _mm256_add_epi8(sidx2, mtwo8);
        }
        const float dx = h2f(d16);
        for (int t = 0; t < NT; ++t) {
            __m256i s1 = _mm256_setzero_si256(), s2 = _mm256_setzero_si256();
            for (int it = 0; it < 4; ++it) {
                const Dec& dc = dec[it];
                const __m256i q8a = _mm256_loadu_si256((const __m256i*) (y[t][i].qs + 64 * it));
                const __m256i q8b = _mm256_loadu_si256((const __m256i*) (y[t][i].qs + 64 * it + 32));
                const __m256i dot1 = _mm256_maddubs_epi16(dc.a1, _mm256_sign_epi8(q8a, dc.g1));
                const __m256i dot2 = _mm256_maddubs_epi16(dc.a2, _mm256_sign_epi8(q8b, dc.g2));
                s1 = _mm256_add_epi32(s1, _mm256_add_epi32(_mm256_madd_epi16(dot1, dc.sa), _mm256_madd_epi16(dot2, dc.sb)));
                const int16_t* pp = P[t] + ((size_t) i * 8 + 2 * it) * 16;
                s2 = _mm256_add_epi32(s2, _mm256_add_epi32(_mm256_madd_epi16(_mm256_loadu_si256((const __m256i*) pp), dc.ssa),
                                                           _mm256_madd_epi16(_mm256_loadu_si256((const __m256i*) (pp + 16)), dc.ssb)));
            }
            const __m256 d = _mm256_set1_ps(y[t][i].d * dx);
            acc1[t] = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(s1), acc1[t]);
            acc2[t] = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(s2), acc2[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = ggml_hsum8(acc1[t]) + 0.125f * ggml_hsum8(acc2[t]);
}

template <int TY, int NT, bool EX = false> STRATA_ROWS_FN
inline void row_dot_any(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res,
                        const int16_t* const* aux = nullptr) {
    if constexpr (TY == 29) row_dot_iq1m<NT>(row, nblocks, y, aux, res);
    else if constexpr (TY == 17) row_dot_iq2xs<NT, EX>(row, nblocks, y, res);
    else                    row_dot<TY, NT, EX>(row, nblocks, y, res);
}

template <int TY, int NT, bool EX> STRATA_ROWS_FN
void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, float* const* ff,
             int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    const int nb = n / QK_K;
    const int16_t* aux[NT] = {};
    if constexpr (TY == 29) {   // (IQ1_M's activation pair sums, once for the rows of this call)
        static thread_local std::vector<int16_t> pre;
        pre.resize((size_t) NT * nb * 128);
        for (int t = 0; t < NT; ++t) {
            iq1m_pre_build(y[t], nb, pre.data() + (size_t) t * nb * 128);
            aux[t] = pre.data() + (size_t) t * nb * 128;
        }
    }
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot_any<TY, NT, EX>(blob + (size_t) r * gu_row, nb, y, g, aux);
        row_dot_any<TY, NT, EX>(blob + up_off + (size_t) r * gu_row, nb, y, u, aux);
        for (int t = 0; t < NT; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
    }
}

template <int TY, int NT, bool EX> STRATA_ROWS_FN
void dot_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    const int nb = n / QK_K;
    const int16_t* aux[NT] = {};
    if constexpr (TY == 29) {   // (IQ1_M's activation pair sums, once for the rows of this call)
        static thread_local std::vector<int16_t> pre;
        pre.resize((size_t) NT * nb * 128);
        for (int t = 0; t < NT; ++t) {
            iq1m_pre_build(y[t], nb, pre.data() + (size_t) t * nb * 128);
            aux[t] = pre.data() + (size_t) t * nb * 128;
        }
    }
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot_any<TY, NT, EX>(w + (size_t) r * row_bytes, nb, y, res, aux);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

// ---- many tokens an expert (the prompt: a chunk's tokens routed to one expert), EX rows only.
// The weights of a 256-value block are decoded once into signed int8 (the grid magnitude times the sign) and every token of the
// call applies them.  The activations come as u8 (q8 + 128, made once per call for all the rows): maddubs(act_u8, w_s8) needs no
// per-token sign, and sum(w * q) = sum(w * (q + 128)) - 128 * sum(w) with the second sum made once per weight block (corr), so the
// integer sums are exactly ggml's, lane for lane (no int16 overflow: |w| <= 62 for these grids, 255 * 62 * 2 < 32767).
// A token costs, per 32 values, one maddubs, one madd and one add (the single-token form has a sign on top).
constexpr int kWideTokens = 64;

STRATA_ROWS_FN inline void wide_prep(const block_q8_K* y, int nb, uint8_t* u) {
    const __m256i flip = _mm256_set1_epi8((char) 0x80);
    for (int i = 0; i < nb; ++i)
        for (int c = 0; c < 8; ++c)
            _mm256_storeu_si256((__m256i*) (u + (size_t) i * 256 + 32 * c),
                                _mm256_xor_si256(_mm256_loadu_si256((const __m256i*) (y[i].qs + 32 * c)), flip));
}

template <int TY> STRATA_ROWS_FN
void dot_rows_wide(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int T, float* const* out, int r0, int r1) {
    // tokens in tiles of kTile and rows in groups of kRows: a tile's activation slice of one block (kTile * 256 B) and the group's
    // accumulators (kRows * kTile * 32 B) stay in L1 while the group's rows go past, the tile's whole activation in L2
#ifndef STRATA_WIDE_TILE
#define STRATA_WIDE_TILE 16
#endif
#ifndef STRATA_WIDE_ROWS
#define STRATA_WIDE_ROWS 8
#endif
    constexpr int kTile = STRATA_WIDE_TILE, kRows = STRATA_WIDE_ROWS;
    const int nb = n / QK_K;
    static thread_local std::vector<uint8_t> ubuf;
    static thread_local std::vector<float> accb;
    ubuf.resize((size_t) (T < kTile ? T : kTile) * n);
    accb.resize((size_t) kRows * kTile * 8);
    const __m256i one8 = _mm256_set1_epi8(1);
    const int pf = prefetch_distance();
    for (int t0 = 0; t0 < T; t0 += kTile) {
        const int tt = T - t0 < kTile ? T - t0 : kTile;
        const block_q8_K* y[kTile];
        for (int t = 0; t < tt; ++t) {
            y[t] = (const block_q8_K*) act[t0 + t];
            wide_prep(y[t], nb, ubuf.data() + (size_t) t * n);
        }
        for (int rg = r0; rg < r1; rg += kRows) {
            const int rr = r1 - rg < kRows ? r1 - rg : kRows;
            for (int k = 0; k < rr * tt; ++k) _mm256_storeu_ps(&accb[(size_t) k * 8], _mm256_setzero_ps());
            for (int i = 0; i < nb; ++i) {
                for (int q = 0; q < rr; ++q) {
                    const uint8_t* blk = w + (size_t) (rg + q) * row_bytes + (size_t) i * Fmt32<TY>::bytes;
                    rows_ahead(blk, pf);
                    __m256i w8[8], sc[8];
                    __m256i corr = _mm256_setzero_si256();
                    for (int j = 0; j < 4; ++j)
                        for (int half = 0; half < 2; ++half) {
                            __m256i g, sgn, s;
                            Fmt32<TY>::decode(blk, j, half, g, sgn, s);
                            const int k = 2 * j + half;
                            w8[k] = _mm256_sign_epi8(g, sgn);
                            sc[k] = s;
                            corr = madd_add(corr, _mm256_maddubs_epi16(one8, w8[k]), s);
                        }
                    const __m256i corr128 = _mm256_slli_epi32(corr, 7);
                    const float dx = h2f(u16(blk));
                    float* accq = &accb[(size_t) q * tt * 8];
                    for (int t = 0; t < tt; ++t) {
                        const uint8_t* u = ubuf.data() + (size_t) t * n + (size_t) i * 256;
                        __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
                        for (int k = 0; k < 8; k += 2) {
                            a0 = madd_add(a0, _mm256_maddubs_epi16(_mm256_loadu_si256((const __m256i*) (u + 32 * k)), w8[k]), sc[k]);
                            a1 = madd_add(a1, _mm256_maddubs_epi16(_mm256_loadu_si256((const __m256i*) (u + 32 * k + 32)), w8[k + 1]), sc[k + 1]);
                        }
                        const __m256i s = _mm256_sub_epi32(_mm256_add_epi32(a0, a1), corr128);
                        float* ap = accq + (size_t) t * 8;
                        _mm256_storeu_ps(ap, _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(s), _mm256_loadu_ps(ap)));
                    }
                }
            }
            for (int q = 0; q < rr; ++q)
                for (int t = 0; t < tt; ++t)
                    out[t0 + t][rg + q] = Fmt32<TY>::K * ggml_hsum8(_mm256_loadu_ps(&accb[((size_t) q * tt + t) * 8]));
        }
    }
}

// IQ1_M the same way: its grid bytes are already signed (-1, 0, 1), and the delta dot keeps the pair sums of the activation
// (maddubs(1, q8), made once per tile).  Two float accumulators a (row, token), ggml's sum d*sumi1 and sum d*sumi2.
STRATA_ROWS_FN
void dot_rows_wide_iq1m(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int T, float* const* out, int r0, int r1) {
    constexpr int kTile = STRATA_WIDE_TILE, kRows = STRATA_WIDE_ROWS;
    const int nb = n / QK_K;
    static thread_local std::vector<uint8_t> ubuf;
    static thread_local std::vector<int16_t> pbuf;
    static thread_local std::vector<float> accb;
    ubuf.resize((size_t) (T < kTile ? T : kTile) * n);
    pbuf.resize((size_t) (T < kTile ? T : kTile) * n);
    accb.resize((size_t) kRows * kTile * 16);
    const __m256i one8 = _mm256_set1_epi8(1);
    const __m256i mask = _mm256_set1_epi16(0x7);
    const __m256i mone = _mm256_set1_epi16(1);
    const __m256i mtwo8 = _mm256_set1_epi8(2);
    const __m256i scales_shift = _mm256_set_epi64x(9, 3, 6, 0);
    const __m256i nib_shift = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
    const __m256i m7 = _mm256_set1_epi32(7), m8 = _mm256_set1_epi32(8);
    const int pf = prefetch_distance();
    for (int t0 = 0; t0 < T; t0 += kTile) {
        const int tt = T - t0 < kTile ? T - t0 : kTile;
        const block_q8_K* y[kTile];
        for (int t = 0; t < tt; ++t) {
            y[t] = (const block_q8_K*) act[t0 + t];
            wide_prep(y[t], nb, ubuf.data() + (size_t) t * n);
            iq1m_pre_build(y[t], nb, pbuf.data() + (size_t) t * n);
        }
        for (int rg = r0; rg < r1; rg += kRows) {
            const int rr = r1 - rg < kRows ? r1 - rg : kRows;
            for (int k = 0; k < rr * tt * 2; ++k) _mm256_storeu_ps(&accb[(size_t) k * 8], _mm256_setzero_ps());
            for (int i = 0; i < nb; ++i) {
                for (int q = 0; q < rr; ++q) {
                    const uint8_t* blk = w + (size_t) (rg + q) * row_bytes + (size_t) i * 56;
                    rows_ahead(blk, pf);
                    const uint8_t* qs = blk;
                    const uint8_t* qh = blk + 32;
                    const uint8_t* scb = blk + 48;
                    const uint16_t sc0 = u16(scb), sc1 = u16(scb + 2), sc2 = u16(scb + 4), sc3 = u16(scb + 6);
                    const uint16_t d16 = (uint16_t) ((sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000));
                    __m256i scales = _mm256_set1_epi64x((long long) u64(scb));
                    scales = _mm256_srlv_epi64(scales, scales_shift);
                    scales = _mm256_add_epi16(_mm256_slli_epi16(_mm256_and_si256(scales, mask), 1), mone);
                    __m256i sidx1 = _mm256_set1_epi16(0x0100);
                    __m256i sidx2 = _mm256_add_epi8(sidx1, _mm256_set1_epi8(8));
                    __m256i g1[4], g2[4], sa[4], sb[4], ssa[4], ssb[4];
                    __m256i corr = _mm256_setzero_si256();
                    for (int it = 0; it < 4; ++it) {
                        const __m256i v8 = _mm256_srlv_epi32(_mm256_set1_epi32((int) u32(qh + 4 * it)), nib_shift);
                        const __m256i idx = _mm256_or_si256(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (qs + 8 * it))),
                                                            _mm256_slli_epi32(_mm256_and_si256(v8, m7), 8));
                        alignas(32) int32_t ix[8];
                        _mm256_store_si256((__m256i*) ix, idx);
                        g1[it] = _mm256_set_epi64x((long long) iq1s_grid[ix[3]], (long long) iq1s_grid[ix[2]],
                                                   (long long) iq1s_grid[ix[1]], (long long) iq1s_grid[ix[0]]);
                        g2[it] = _mm256_set_epi64x((long long) iq1s_grid[ix[7]], (long long) iq1s_grid[ix[6]],
                                                   (long long) iq1s_grid[ix[5]], (long long) iq1s_grid[ix[4]]);
                        const __m256i neg = _mm256_cmpeq_epi32(_mm256_and_si256(v8, m8), m8);
                        const __m256i n1 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(neg));
                        const __m256i n2 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(neg, 1));
                        sa[it] = _mm256_shuffle_epi8(scales, sidx1);
                        sb[it] = _mm256_shuffle_epi8(scales, sidx2);
                        ssa[it] = _mm256_sign_epi16(sa[it], _mm256_or_si256(n1, mone));
                        ssb[it] = _mm256_sign_epi16(sb[it], _mm256_or_si256(n2, mone));
                        sidx1 = _mm256_add_epi8(sidx1, mtwo8);
                        sidx2 = _mm256_add_epi8(sidx2, mtwo8);
                        corr = madd_add(madd_add(corr, _mm256_maddubs_epi16(one8, g1[it]), sa[it]), _mm256_maddubs_epi16(one8, g2[it]), sb[it]);
                    }
                    const __m256i corr128 = _mm256_slli_epi32(corr, 7);
                    const float dx = h2f(d16);
                    float* accq = &accb[(size_t) q * tt * 16];
                    for (int t = 0; t < tt; ++t) {
                        const uint8_t* u = ubuf.data() + (size_t) t * n + (size_t) i * 256;
                        const int16_t* pp = pbuf.data() + (size_t) t * n + (size_t) i * 128;
                        __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
                        __m256i b0 = _mm256_setzero_si256(), b1 = _mm256_setzero_si256();
                        for (int it = 0; it < 4; ++it) {
                            a0 = madd_add(a0, _mm256_maddubs_epi16(_mm256_loadu_si256((const __m256i*) (u + 64 * it)), g1[it]), sa[it]);
                            a1 = madd_add(a1, _mm256_maddubs_epi16(_mm256_loadu_si256((const __m256i*) (u + 64 * it + 32)), g2[it]), sb[it]);
                            b0 = madd_add(b0, _mm256_loadu_si256((const __m256i*) (pp + 32 * it)), ssa[it]);
                            b1 = madd_add(b1, _mm256_loadu_si256((const __m256i*) (pp + 32 * it + 16)), ssb[it]);
                        }
                        const __m256i s1 = _mm256_sub_epi32(_mm256_add_epi32(a0, a1), corr128);
                        const __m256i s2 = _mm256_add_epi32(b0, b1);
                        const __m256 d = _mm256_set1_ps(y[t][i].d * dx);
                        float* ap = accq + (size_t) t * 16;
                        _mm256_storeu_ps(ap, _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(s1), _mm256_loadu_ps(ap)));
                        _mm256_storeu_ps(ap + 8, _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(s2), _mm256_loadu_ps(ap + 8)));
                    }
                }
            }
            for (int q = 0; q < rr; ++q)
                for (int t = 0; t < tt; ++t) {
                    const float* ap = &accb[((size_t) q * tt + t) * 16];
                    out[t0 + t][rg + q] = ggml_hsum8(_mm256_loadu_ps(ap)) + 0.125f * ggml_hsum8(_mm256_loadu_ps(ap + 8));
                }
        }
    }
}

template <int TY, bool EX> STRATA_ROWS_FN
void gu_rows_nt(int nt, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                float* const* ff, int r0, int r1) {
    switch (nt) {
        case 1: gu_rows<TY, 1, EX>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 2: gu_rows<TY, 2, EX>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 3: gu_rows<TY, 3, EX>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 4: gu_rows<TY, 4, EX>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 5: gu_rows<TY, 5, EX>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 6: gu_rows<TY, 6, EX>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 7: gu_rows<TY, 7, EX>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: gu_rows<TY, 8, EX>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    }
}

template <int TY, bool EX> STRATA_ROWS_FN
void dot_rows_nt(int nt, const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0,
                 int r1) {
    if constexpr (EX) {
        if (nt >= 7) {   // (a prompt chunk's tokens: one decode of a block for all of them)
            if constexpr (TY == 29) dot_rows_wide_iq1m(w, row_bytes, n, act, nt, out, r0, r1);
            else dot_rows_wide<TY>(w, row_bytes, n, act, nt, out, r0, r1);
            return;
        }
    }
    switch (nt) {
        case 1: dot_rows<TY, 1, EX>(w, row_bytes, n, act, out, r0, r1); break;
        case 2: dot_rows<TY, 2, EX>(w, row_bytes, n, act, out, r0, r1); break;
        case 3: dot_rows<TY, 3, EX>(w, row_bytes, n, act, out, r0, r1); break;
        case 4: dot_rows<TY, 4, EX>(w, row_bytes, n, act, out, r0, r1); break;
        case 5: if constexpr (EX) { dot_rows<TY, 5, EX>(w, row_bytes, n, act, out, r0, r1); break; }   // (the DeepSeek lane's verify
        [[fallthrough]];                                                                              //  rounds: one decode of 5 or 6)
        case 6: if constexpr (EX) { dot_rows<TY, 6, EX>(w, row_bytes, n, act, out, r0, r1); break; }
        [[fallthrough]];
        default: for (int t0 = 0; t0 < nt; t0 += 4) {
            const int k = nt - t0 < 4 ? nt - t0 : 4;
            dot_rows_nt<TY, EX>(k, w, row_bytes, n, act + t0, out + t0, r0, r1);
        }
    }
}

// The decode of each format for a variant: the gathered one (kIq256Gather) for IQ3_XXS, IQ3_S and IQ2_S.
template <bool G, bool EX = false> STRATA_ROWS_FN
void gu_type(int type, int nt, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
             float* const* ff, int r0, int r1) {
    switch (type) {
        case 16: gu_rows_nt<16, EX>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 17: gu_rows_nt<17, EX>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 18: gu_rows_nt<G ? 118 : 18, EX>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 21: gu_rows_nt<G ? 121 : 21, EX>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 22: gu_rows_nt<G ? 122 : 22, EX>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 23: gu_rows_nt<23, EX>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 29: gu_rows_nt<29, EX>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: break;
    }
}

template <bool G, bool EX = false> STRATA_ROWS_FN
void dot_type(int type, int nt, const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out,
              int r0, int r1) {
    switch (type) {
        case 16: dot_rows_nt<16, EX>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 17: dot_rows_nt<17, EX>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 18: dot_rows_nt<G ? 118 : 18, EX>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 21: dot_rows_nt<G ? 121 : 21, EX>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 22: dot_rows_nt<G ? 122 : 22, EX>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 23: dot_rows_nt<23, EX>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 29: dot_rows_nt<29, EX>(nt, w, row_bytes, n, act, out, r0, r1); break;
        default: break;
    }
}

// ---- IQ4_NL (type 20): 32-value blocks of 18 bytes (f16 d + 16 nibble bytes) against Q8_0 activations.
// ggml-cpu's own AVX-2 dot (ggml_vec_dot_iq4_nl_q8_0) is single-token: every token redoes the nibble ->
// kvalues_iq4nl pshufb decode and the |w| half of the signed x signed product.  Here both are computed once
// per block; every token then costs a sign, a maddubs, a madd and an fmadd.  The arithmetic is ggml's - only
// the order of the float additions differs.
template <int NT> STRATA_ROWS_FN
void iq4nl_rows(const uint8_t* w, size_t row_bytes, int n, const block_q8_0* const* y, float* const* out,
                int r0, int r1) {
    const __m128i values = _mm_loadu_si128((const __m128i*) kvalues_iq4nl);
    const __m128i m4b = _mm_set1_epi8(0x0f);
    const int nb = n / QK4_NL;
    const int pf = prefetch_distance();
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * row_bytes;
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
        for (int ib = 0; ib < nb; ++ib) {
            const uint8_t* blk = row + (size_t) ib * sizeof(block_iq4_nl);
            rows_ahead(blk, pf);
            const __m128i bits = _mm_loadu_si128((const __m128i*) (blk + 2));
            const __m128i lo = _mm_and_si128(bits, m4b);                      // values 0..15
            const __m128i hi = _mm_and_si128(_mm_srli_epi16(bits, 4), m4b);    // values 16..31
            const __m256i q4 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_shuffle_epi8(values, lo)),
                                                       _mm_shuffle_epi8(values, hi), 1);
            const __m256i aq = _mm256_sign_epi8(q4, q4);   // |w|: the unsigned operand of maddubs
            const float dx = h2f(u16(blk));
            for (int t = 0; t < NT; ++t) {
                const block_q8_0& b = y[t][ib];
                const __m256i q8 = _mm256_loadu_si256((const __m256i*) b.qs);
                const __m256i p = dot4u(aq, _mm256_sign_epi8(q8, q4));
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * h2f(b.d)), _mm256_cvtepi32_ps(p), accf[t]);
            }
        }
        for (int t = 0; t < NT; ++t) out[t][r] = hsum8(accf[t]);
    }
}

STRATA_ROWS_FN void iq4nl_nt(const uint8_t* w, size_t row_bytes, int n, const block_q8_0* const* y, int nt,
                             float* const* out, int r0, int r1) {
    switch (nt) {
        case 1: iq4nl_rows<1>(w, row_bytes, n, y, out, r0, r1); break;
        case 2: iq4nl_rows<2>(w, row_bytes, n, y, out, r0, r1); break;
        case 3: iq4nl_rows<3>(w, row_bytes, n, y, out, r0, r1); break;
        case 4: iq4nl_rows<4>(w, row_bytes, n, y, out, r0, r1); break;
        default: for (int t0 = 0; t0 < nt; t0 += 4) {
            const int k = nt - t0 < 4 ? nt - t0 : 4;
            iq4nl_nt(w, row_bytes, n, y + t0, k, out + t0, r0, r1);
        }
    }
}

#undef STRATA_ROWS_VNNI
#undef STRATA_ROWS_FN
