// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_mx_kern.inc - the CPU lane's int8 MXFP4 rows (ds4_gguf_cpu.hpp: mx_rows), one body for the x86 tiers.
// ds4_gguf_cpu.cpp includes it once a tier, inside its unnamed namespace, after <immintrin.h>, with
//   MXK_NS    the tier's namespace
//   MXK_ATTR  the tier's instruction set as a function attribute (GCC / Clang: target(...); MSVC: nothing - it takes
//             any intrinsic) on every function here, so the file is built for the baseline and each tier's code runs
//             only where ds4_gguf_cpu.cpp's CPU check says it can
//   MXK_256 or MXK_512, and MXK_VNNI (vpdpbusd; else vpmaddubsw + vpmaddwd)
// and gets MXK_NS::rows(...) (and with MXK_256 and no MXK_VNNI, MXK_NS::quant(...)); the macros are undefined at the end.
//
// The arithmetic is the device's (ds4_gguf.cu, mx_dot) - a block's 32 codes times the int8 activations summed exactly
// in integers, times (weight scale x activation scale) into F32 sums - in ONE order for every tier, the scalar one
// included (ds4_gguf_cpu.cpp's rows_q_scalar), so every tier gives the same bits and the choice of tier never changes
// an answer:
//   * the blocks go in groups of 8; block j of a group adds into accumulator L(j), L = {0, 4, 1, 5, 2, 6, 3, 7} (the
//     order a 256-bit hadd tree leaves the 8 blocks' totals in: lanes = blocks 0, 2, 4, 6, 1, 3, 5, 7):
//     acc[L(j)] = fma(float(total_j), dw_j * dx_j, acc[L(j)]), the groups in turn;
//   * the row is ((acc0 + acc4) + (acc2 + acc6)) + ((acc1 + acc5) + (acc3 + acc7)) (hsum256's tree).
// The codes come out of a shuffle table as w + 12 (0..24, unsigned bytes), so the products go through vpmaddubsw /
// vpdpbusd as u8 x s8; each block's total then gets the activation's -12 x (sum of its 32 values) back, exactly.
//
// The activation of a call (mx_act_bytes(k): k % 256 == 0 here; quant() / ds4_gguf_cpu.cpp's quant_scalar):
//   [0, k)            int8 values in quads of 4 blocks, quad g at 128 g: the 4 blocks' values 0..15 (block j at
//                     +16 j), then their 16..31 - one 256-bit step takes a pair of blocks, one 512-bit step a quad
//   [k, k + k/8)      F32 scale of each block, group G (8 blocks) at 8 G in lane order (blocks 0, 2, 4, 6, 1, 3, 5, 7)
//   [k + k/8, + k/8)  int32 -12 x (sum of the block's 32 values), the same order
namespace MXK_NS {

constexpr int kBlockBytes = 17;   // E8M0 scale + 16 code bytes

MXK_ATTR inline float hsum256(__m256 v) {
    const __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    const __m128 b = _mm_add_ps(a, _mm_movehl_ps(a, a));
    return _mm_cvtss_f32(_mm_add_ss(b, _mm_shuffle_ps(b, b, 1)));
}
// the 8 weight scales of a group (its first block at p), in lane order
MXK_ATTR inline __m256 group_dw(const uint8_t* p, const float* e8) {
    return _mm256_setr_ps(e8[p[0]], e8[p[2 * kBlockBytes]], e8[p[4 * kBlockBytes]], e8[p[6 * kBlockBytes]],
                          e8[p[kBlockBytes]], e8[p[3 * kBlockBytes]], e8[p[5 * kBlockBytes]], e8[p[7 * kBlockBytes]]);
}

#if defined(MXK_256)
// ---------------------------------------------------------------- 256-bit: a pair of blocks a step
// a pair's codes (a1: values 0..15 of both blocks, a2: 16..31) times x1 / x2: 8 int32 lanes, block A's 4 then B's 4
MXK_ATTR inline __m256i pair_dot(__m256i a1, __m256i a2, __m256i x1, __m256i x2) {
#if defined(MXK_VNNI)
    return _mm256_dpbusd_avx_epi32(_mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), a1, x1), a2, x2);
#else
    const __m256i p = _mm256_add_epi16(_mm256_maddubs_epi16(a1, x1), _mm256_maddubs_epi16(a2, x2));
    return _mm256_madd_epi16(p, _mm256_set1_epi16(1));
#endif
}

template <int N>
MXK_ATTR inline void rows_n(const uint8_t* w, size_t rb, int k, int r0, int r1, const void* const* act, float* const* y,
                            const float* e8, int pf) {
    const __m256i lut = _mm256_setr_epi8(12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0,   //
                                         12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0);
    const __m256i m4 = _mm256_set1_epi8(0x0f);
    const int ng = k / 256;
    const int8_t* xq[N];
    const float* xd[N];
    const int32_t* xc[N];
    for (int i = 0; i < N; ++i) {
        const uint8_t* a = (const uint8_t*) act[i];
        xq[i] = (const int8_t*) a;
        xd[i] = (const float*) (a + k);
        xc[i] = (const int32_t*) (a + k + k / 8);
    }
    for (int r = r0; r < r1; ++r) {
        const uint8_t* p = w + (size_t) r * rb;
        __m256 acc[N];
        for (int i = 0; i < N; ++i) acc[i] = _mm256_setzero_ps();
        for (int G = 0; G < ng; ++G, p += 8 * kBlockBytes) {
            if (pf > 0) _mm_prefetch((const char*) p + pf, _MM_HINT_T0);
            const __m256 dw = group_dw(p, e8);
            // the group's 4 pairs of blocks (2 q, 2 q + 1) decoded once; then each activation in turn: its 4 pair dots,
            // the hadd tree to the 8 blocks' totals (lanes: blocks 0 2 4 6 | 1 3 5 7), the FMA - only the accumulators
            // stay live from one activation to the next (16 YMM registers on AVX2)
            __m256i a1[4], a2[4];
            for (int q = 0; q < 4; ++q) {
                const uint8_t* pb = p + 2 * q * kBlockBytes;
                const __m256i c = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) (pb + 1))),
                                                          _mm_loadu_si128((const __m128i*) (pb + 18)), 1);
                a1[q] = _mm256_shuffle_epi8(lut, _mm256_and_si256(c, m4));
                a2[q] = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(c, 4), m4));
            }
            const int o = G * 256;   // (quads 2 G and 2 G + 1: pair q at (q / 2) * 128 + (q % 2) * 32)
            for (int i = 0; i < N; ++i) {
                const int8_t* x = xq[i] + o;
                const __m256i s0 = pair_dot(a1[0], a2[0], _mm256_loadu_si256((const __m256i*) x),
                                            _mm256_loadu_si256((const __m256i*) (x + 64)));
                const __m256i s1 = pair_dot(a1[1], a2[1], _mm256_loadu_si256((const __m256i*) (x + 32)),
                                            _mm256_loadu_si256((const __m256i*) (x + 96)));
                const __m256i s2 = pair_dot(a1[2], a2[2], _mm256_loadu_si256((const __m256i*) (x + 128)),
                                            _mm256_loadu_si256((const __m256i*) (x + 192)));
                const __m256i s3 = pair_dot(a1[3], a2[3], _mm256_loadu_si256((const __m256i*) (x + 160)),
                                            _mm256_loadu_si256((const __m256i*) (x + 224)));
                const __m256i u = _mm256_hadd_epi32(_mm256_hadd_epi32(s0, s1), _mm256_hadd_epi32(s2, s3));
                const __m256i tot = _mm256_add_epi32(u, _mm256_loadu_si256((const __m256i*) (xc[i] + 8 * G)));
                acc[i] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(tot), _mm256_mul_ps(dw, _mm256_loadu_ps(xd[i] + 8 * G)), acc[i]);
            }
        }
        for (int i = 0; i < N; ++i) y[i][r - r0] = hsum256(acc[i]);
    }
}

#if !defined(MXK_VNNI)
// blocks [b0, b1) of x (x at block b0's first value) into act, the bytes of quant_scalar (ds4_gguf_cpu.cpp): the same
// IEEE division and round-half-even conversion, the same max
MXK_ATTR inline void quant(const float* x, int b0, int b1, uint8_t* act, int k) {
    int8_t* q = (int8_t*) act;
    float* dx = (float*) (act + k);
    int32_t* corr = (int32_t*) (act + k + k / 8);
    const __m256 sign = _mm256_set1_ps(-0.0f);
    const __m256i order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (int b = b0; b < b1; ++b, x += 32) {
        const __m256 v0 = _mm256_loadu_ps(x), v1 = _mm256_loadu_ps(x + 8);
        const __m256 v2 = _mm256_loadu_ps(x + 16), v3 = _mm256_loadu_ps(x + 24);
        const __m256 m = _mm256_max_ps(_mm256_max_ps(_mm256_andnot_ps(sign, v0), _mm256_andnot_ps(sign, v1)),
                                       _mm256_max_ps(_mm256_andnot_ps(sign, v2), _mm256_andnot_ps(sign, v3)));
        __m128 m2 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        m2 = _mm_max_ps(m2, _mm_movehl_ps(m2, m2));
        m2 = _mm_max_ss(m2, _mm_shuffle_ps(m2, m2, 1));
        const float am = _mm_cvtss_f32(m2);
        const float sc = am / 127.0f;
        int8_t* lo = q + (b >> 2) * 128 + 16 * (b & 3);
        int8_t* hi = lo + 64;
        int s = 0;
        if (am > 0.0f) {
            const __m256 vs = _mm256_set1_ps(sc);
            const __m256i q0 = _mm256_cvtps_epi32(_mm256_div_ps(v0, vs)), q1 = _mm256_cvtps_epi32(_mm256_div_ps(v1, vs));
            const __m256i q2 = _mm256_cvtps_epi32(_mm256_div_ps(v2, vs)), q3 = _mm256_cvtps_epi32(_mm256_div_ps(v3, vs));
            const __m256i pk = _mm256_permutevar8x32_epi32(
                _mm256_packs_epi16(_mm256_packs_epi32(q0, q1), _mm256_packs_epi32(q2, q3)), order);
            _mm_storeu_si128((__m128i*) lo, _mm256_castsi256_si128(pk));
            _mm_storeu_si128((__m128i*) hi, _mm256_extracti128_si256(pk, 1));
            __m256i t = _mm256_add_epi32(_mm256_add_epi32(q0, q1), _mm256_add_epi32(q2, q3));
            __m128i t2 = _mm_add_epi32(_mm256_castsi256_si128(t), _mm256_extracti128_si256(t, 1));
            t2 = _mm_add_epi32(t2, _mm_shuffle_epi32(t2, 0x4E));
            t2 = _mm_add_epi32(t2, _mm_shuffle_epi32(t2, 0xB1));
            s = _mm_cvtsi128_si32(t2);
        } else {
            _mm_storeu_si128((__m128i*) lo, _mm_setzero_si128());
            _mm_storeu_si128((__m128i*) hi, _mm_setzero_si128());
        }
        const int j = b & 7, lane = 8 * (b >> 3) + (j & 1) * 4 + (j >> 1);
        dx[lane] = sc;
        corr[lane] = -12 * s;
    }
}
#endif
#endif  // MXK_256

#if defined(MXK_512)
// ---------------------------------------------------------------- 512-bit: a quad of blocks a step
MXK_ATTR inline __m512i quad_dot(__m512i a1, __m512i a2, __m512i x1, __m512i x2) {
#if defined(MXK_VNNI)
    return _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), a1, x1), a2, x2);
#else
    const __m512i p = _mm512_add_epi16(_mm512_maddubs_epi16(a1, x1), _mm512_maddubs_epi16(a2, x2));
    return _mm512_madd_epi16(p, _mm512_set1_epi16(1));
#endif
}

template <int N>
MXK_ATTR inline void rows_n(const uint8_t* w, size_t rb, int k, int r0, int r1, const void* const* act, float* const* y,
                            const float* e8, int pf) {
    const __m512i lut = _mm512_broadcast_i32x4(_mm_setr_epi8(12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0));
    const __m512i m4 = _mm512_set1_epi8(0x0f);
    // a quad's 16 lanes after adjacent-pair sums: block j's halves in lanes 4 j and 4 j + 2; the group's two quads ->
    // the 8 blocks in lane order (0 2 4 6 1 3 5 7), first halves (pa) and second halves (pb)
    const __m512i pa = _mm512_setr_epi32(0, 8, 16, 24, 4, 12, 20, 28, 0, 0, 0, 0, 0, 0, 0, 0);
    const __m512i pb = _mm512_setr_epi32(2, 10, 18, 26, 6, 14, 22, 30, 0, 0, 0, 0, 0, 0, 0, 0);
    const int ng = k / 256;
    const int8_t* xq[N];
    const float* xd[N];
    const int32_t* xc[N];
    for (int i = 0; i < N; ++i) {
        const uint8_t* a = (const uint8_t*) act[i];
        xq[i] = (const int8_t*) a;
        xd[i] = (const float*) (a + k);
        xc[i] = (const int32_t*) (a + k + k / 8);
    }
    for (int r = r0; r < r1; ++r) {
        const uint8_t* p = w + (size_t) r * rb;
        __m256 acc[N];
        for (int i = 0; i < N; ++i) acc[i] = _mm256_setzero_ps();
        for (int G = 0; G < ng; ++G, p += 8 * kBlockBytes) {
            if (pf > 0) _mm_prefetch((const char*) p + pf, _MM_HINT_T0);
            const __m256 dw = group_dw(p, e8);
            __m512i h0[N];
            for (int qd = 0; qd < 2; ++qd) {   // the group's quads: blocks 4 qd .. 4 qd + 3
                const uint8_t* pq = p + 4 * qd * kBlockBytes;
                __m512i c = _mm512_castsi128_si512(_mm_loadu_si128((const __m128i*) (pq + 1)));
                c = _mm512_inserti32x4(c, _mm_loadu_si128((const __m128i*) (pq + 18)), 1);
                c = _mm512_inserti32x4(c, _mm_loadu_si128((const __m128i*) (pq + 35)), 2);
                c = _mm512_inserti32x4(c, _mm_loadu_si128((const __m128i*) (pq + 52)), 3);
                const __m512i a1 = _mm512_shuffle_epi8(lut, _mm512_and_si512(c, m4));
                const __m512i a2 = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(c, 4), m4));
                const int o = (2 * G + qd) * 128;
                for (int i = 0; i < N; ++i) {
                    const __m512i s = quad_dot(a1, a2, _mm512_loadu_si512((const void*) (xq[i] + o)),
                                               _mm512_loadu_si512((const void*) (xq[i] + o + 64)));
                    const __m512i h = _mm512_add_epi32(s, _mm512_shuffle_epi32(s, (_MM_PERM_ENUM) 0xB1));
                    if (qd == 0) {
                        h0[i] = h;
                        continue;
                    }
                    const __m512i ta = _mm512_permutex2var_epi32(h0[i], pa, h), tb = _mm512_permutex2var_epi32(h0[i], pb, h);
                    const __m256i tot = _mm256_add_epi32(
                        _mm256_add_epi32(_mm512_castsi512_si256(ta), _mm512_castsi512_si256(tb)),
                        _mm256_loadu_si256((const __m256i*) (xc[i] + 8 * G)));
                    acc[i] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(tot), _mm256_mul_ps(dw, _mm256_loadu_ps(xd[i] + 8 * G)), acc[i]);
                }
            }
        }
        for (int i = 0; i < N; ++i) y[i][r - r0] = hsum256(acc[i]);
    }
}
#endif  // MXK_512

// y[i][r - r0] for every i < n, in calls of up to 8 activations; e8 the 256 E8M0 scales as F32 (2^(e - 128)), pf the
// bytes to prefetch ahead in a row (0: none)
MXK_ATTR inline void rows(const uint8_t* w, size_t rb, int k, int r0, int r1, const void* const* act, int n,
                          float* const* y, const float* e8, int pf) {
    for (int i0 = 0; i0 < n; i0 += 8) {
        float* yy[8];
        const int c = n - i0 < 8 ? n - i0 : 8;
        for (int i = 0; i < c; ++i) yy[i] = y[i0 + i];
        switch (c) {
            case 1: rows_n<1>(w, rb, k, r0, r1, act + i0, yy, e8, pf); break;
            case 2: rows_n<2>(w, rb, k, r0, r1, act + i0, yy, e8, pf); break;
            case 3: rows_n<3>(w, rb, k, r0, r1, act + i0, yy, e8, pf); break;
            case 4: rows_n<4>(w, rb, k, r0, r1, act + i0, yy, e8, pf); break;
            case 5: rows_n<5>(w, rb, k, r0, r1, act + i0, yy, e8, pf); break;
            case 6: rows_n<6>(w, rb, k, r0, r1, act + i0, yy, e8, pf); break;
            case 7: rows_n<7>(w, rb, k, r0, r1, act + i0, yy, e8, pf); break;
            default: rows_n<8>(w, rb, k, r0, r1, act + i0, yy, e8, pf); break;
        }
    }
}

}  // namespace MXK_NS

#undef MXK_NS
#undef MXK_ATTR
#undef MXK_256
#undef MXK_512
#undef MXK_VNNI
