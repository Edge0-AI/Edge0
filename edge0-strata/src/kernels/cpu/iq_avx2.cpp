// src/kernels/cpu/iq_avx2.cpp - the i-quant expert rows in 256-bit lanes, several tokens at once.
//
// The AVX-512 kernels' (iq_avx512.cpp) multi-token scheme on the CPUs without AVX-512 (AMD Zen 2/3, Intel
// Core 12th-14th gen and Core Ultra): each 32-value chunk is decoded once - grid lookups, one sign vector,
// one scale vector - and every token of the verify window applies them with a load, a `sign`, a `maddubs`,
// a `madd` and an `add`.  ggml-cpu's own AVX2 dot products for these formats are single-token: every token
// re-does the codebook lookups and the sign expansion.  The sign vector is ggml's bit_selector pattern
// (ggml-cpu/arch/x86/quants.c): pshufb-broadcast of each sign byte, AND with the bit selector, CMPEQ, OR
// with one, `vpsignb`.  The arithmetic is ggml's (ggml-cpu/quants.c, the `_generic` references) - only the
// order of the float additions differs.
//
// Formats: IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18), IQ3_S (21), IQ2_S (22), IQ4_XS (23).  IQ1_M stays on ggml-cpu.
//
// IQ3_XXS, IQ3_S and IQ2_S also have a gathered decode (Fmt32<118>, <121>, <122>): the grid indices built in a
// register and read with one vpgather, the same words into the same lanes, so the same bits.  Where it is faster
// depends on the core, so STRATA_IQ256_GATHER picks it per thread (expert_layout.hpp, cpu_gather_fast_here).
//
// Where the CPU has AVX-VNNI (cpu_avxvnni_ok: Alder Lake, Sapphire Rapids and later) the row kernels take a second
// copy with vpdpwssd / vpdpbusd (iq_avx2_rows.inl), the same integer sums.
//
// This file is compiled for AVX2 and must not run anything before the engine's cpu_avx2_ok() check: no runtime
// initializer at namespace scope (#391, 016ea2e), the switches are read on first use.
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

// STRATA_AVXVNNI (CMake: the compiler has the AVX-VNNI intrinsics): GCC and Clang compile them only inside functions
// with target("avxvnni") - the VNNI copy of iq_avx2_rows.inl.
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

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }
inline uint32_t u32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint16_t u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline uint64_t u64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

// 32 sign bits -> 32 bytes of -1 (bit set) / +1: ggml's bit_selector pattern, shared by all tokens.
inline __m256i sgn_vec(uint32_t m) {
    const __m128i bm = _mm_set1_epi32((int) m);
    const __m128i lo = _mm_shuffle_epi8(bm, _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1));
    const __m128i hi = _mm_shuffle_epi8(bm, _mm_setr_epi8(2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3));
    const __m256i bits = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
    const __m256i sel = _mm256_setr_epi8(1, 2, 4, 8, 16, 32, 64, (char) 0x80,
                                         1, 2, 4, 8, 16, 32, 64, (char) 0x80,
                                         1, 2, 4, 8, 16, 32, 64, (char) 0x80,
                                         1, 2, 4, 8, 16, 32, 64, (char) 0x80);
    const __m256i nz = _mm256_cmpeq_epi8(_mm256_and_si256(bits, sel), sel);
    return _mm256_or_si256(nz, _mm256_set1_epi8(1));
}

// The same sign vector read in place from the half's four sign bytes (IQ3_S, IQ2_S): the bytes come in as one
// broadcast load (vpbroadcastd from memory, a load uop) and ONE pshufb expands them, instead of sgn_vec's two
// pshufb + inserti128 - three uops on the single shuffle port of Haswell-class cores, one here.
inline __m256i sgn_vec_at(const uint8_t* p) {
    const __m256i bits = _mm256_shuffle_epi8(_mm256_set1_epi32((int) u32(p)),
                                             _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                                              2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3));
    const __m256i sel = _mm256_setr_epi8(1, 2, 4, 8, 16, 32, 64, (char) 0x80,
                                         1, 2, 4, 8, 16, 32, 64, (char) 0x80,
                                         1, 2, 4, 8, 16, 32, 64, (char) 0x80,
                                         1, 2, 4, 8, 16, 32, 64, (char) 0x80);
    const __m256i nz = _mm256_cmpeq_epi8(_mm256_and_si256(bits, sel), sel);
    return _mm256_or_si256(nz, _mm256_set1_epi8(1));
}

// The two 16-value scales of one 32-value half (IQ2_XS, IQ2_S): int16 lanes 0-7 = a (values 0-15),
// 8-15 = b (values 16-31).  NOT a broadcast dword [a,b] - that would alternate the scales lane by lane.
inline __m256i sc16(int a, int b) {
    return _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_set1_epi16((short) a)),
                                   _mm_set1_epi16((short) b), 1);
}
// One scale for the whole 32-value half (IQ2_XXS, IQ3_XXS, IQ3_S).
inline __m256i sc32(int s) { return _mm256_set1_epi16(s); }

// keven_signs_q2xs (ggml keeps it static in arch/x86/quants.c): one u64 per 7-bit sign index, byte k
// = 0xFF when bit k of ksigns_iq2xs[i] is set, 0x01 otherwise.  With this a whole 32-value sign vector
// is four scalar loads and a set_epi64x - no ksigns byte packing chain and no bit_selector expansion.
//
// Computed at COMPILE time (constexpr), so this file has no static constructor at all.  This TU is compiled
// for AVX2, and a runtime constructor here runs before main() on every CPU: MSVC turned its variable shift
// into BMI2 `shlx` (#391, demetree: strata.exe exited 0xC000001D before printing anything on a Sandy Bridge
// Xeon) and GCC vectorised the loop into AVX2 (`vpbroadcastb`, found on an AVX-only Xeon E5 by the
// Strata_Dirigo fork).  ksigns_iq2xs[i] is i's 7 bits plus an even-parity bit 7 (ggml-common.h); a const
// array is not usable in a constant expression, so the byte is derived the same way here, and
// native_expert_parity checks these kernels against ggml-cpu's.
struct EvenSigns {
    uint64_t v[128];
    constexpr EvenSigns() : v{} {
        for (int i = 0; i < 128; ++i) {
            int par = 0;
            for (int k = 0; k < 7; ++k) par ^= (i >> k) & 1;
            const int s = i | (par << 7);          // == ksigns_iq2xs[i]
            uint64_t r = 0;
            for (int k = 0; k < 8; ++k) r |= (uint64_t) (((s >> k) & 1) ? 0xFF : 0x01) << (8 * k);
            v[i] = r;
        }
    }
};
static constexpr EvenSigns even_signs{};
static_assert(even_signs.v[0] == 0x0101010101010101ull && even_signs.v[1] == 0xFF010101010101FFull,
              "keven_signs_q2xs: byte k = 0xFF when bit k of ksigns_iq2xs[i] is set");

// IQ3_S / IQ2_S grid indices: the high index bits of a 32-value half (one qh byte) spread one index per byte,
// so a single punpcklbw puts them next to the low index bytes and every grid lookup is a 16-bit load plus the
// table load - instead of a shift, a mask and an or per index.  IQ3_S: byte k = bit k (the 9th bit of index k).
// IQ2_S: byte k = bits 2k,2k+1 (bits 8-9 of index k), k < 4.  constexpr for the same reason as EvenSigns.
struct HiSpread {
    uint64_t iq3s[256];
    uint64_t iq2s[256];
    constexpr HiSpread() : iq3s{}, iq2s{} {
        for (int h = 0; h < 256; ++h) {
            uint64_t r3 = 0, r2 = 0;
            for (int k = 0; k < 8; ++k) r3 |= (uint64_t) ((h >> k) & 1) << (8 * k);
            for (int k = 0; k < 4; ++k) r2 |= (uint64_t) ((h >> (2 * k)) & 3) << (8 * k);
            iq3s[h] = r3;
            iq2s[h] = r2;
        }
    }
};
static constexpr HiSpread hi_spread{};
static_assert(hi_spread.iq3s[0x81] == 0x0100000000000001ull && hi_spread.iq2s[0xE4] == 0x0000000003020100ull,
              "hi_spread: one high-bit group per index byte");

// The (2s+1) scale vectors as tables (IQ3_S: one 4-bit scale per 32-value half; IQ2_S: one scale byte holding
// the two 16-value scales of the half) - one load instead of the shift/or/broadcast chain of sc32()/sc16().
struct ScaleVecs {
    int16_t s32[16][16];    // all 16 lanes = 2s+1
    int16_t s16[256][16];   // lanes 0-7 = 2*(b&15)+1, lanes 8-15 = 2*(b>>4)+1 (the layout sc16() builds)
    constexpr ScaleVecs() : s32{}, s16{} {
        for (int s = 0; s < 16; ++s)
            for (int l = 0; l < 16; ++l) s32[s][l] = (int16_t) (2 * s + 1);
        for (int b = 0; b < 256; ++b)
            for (int l = 0; l < 8; ++l) {
                s16[b][l] = (int16_t) (2 * (b & 15) + 1);
                s16[b][l + 8] = (int16_t) (2 * (b >> 4) + 1);
            }
    }
};
static constexpr ScaleVecs scale_vecs{};
static_assert(scale_vecs.s32[15][0] == 31 && scale_vecs.s16[0x2F][7] == 31 && scale_vecs.s16[0x2F][8] == 5,
              "scale_vecs: 2s+1, low nibble in lanes 0-7");

inline __m256i scale_vec(const int16_t* lanes) { return _mm256_loadu_si256((const __m256i*) lanes); }

// The low index bytes q[0..7] interleaved with the spread high bits: eight 16-bit grid indices in sp[].
inline void grid_indices(const uint8_t* q, uint64_t hi, uint16_t* sp) {
    _mm_storeu_si128((__m128i*) sp, _mm_unpacklo_epi8(_mm_cvtsi64_si128((long long) u64(q)),
                                                      _mm_cvtsi64_si128((long long) hi)));
}

inline float hsum8(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

// E-2 (iq_avx512.cpp) on the AVX-2 path: the pool streams the expert rows from DRAM at ~25 GB/s (4 KB pages
// when large pages are refused), so ask for the bytes a few blocks before the decode needs them - 2048 B is
// two gate/up rows ahead, a row is ~1 KB.  Same switch as the AVX-512 kernels: STRATA_IQ_PREFETCH is the
// distance in bytes, 0 = off, default 2048.  Measured on a Zen 3 5700X3D (no AVX-512) on IQ3_S decode: -4% on
// the gate/up phase, +1.0 GB/s over the rows, -1.3% ms/round end to end.  The non-temporal hint measured
// worse than T0 at the same distance, so this keeps T0.
int prefetch_distance() {
    static const int d = [] {
        const char* v = std::getenv("STRATA_IQ_PREFETCH");
        return v ? std::atoi(v) : 2048;
    }();
    return d;
}

inline void rows_ahead(const uint8_t* blk, int pf) {
    if (pf <= 0) return;
    _mm_prefetch((const char*) blk + pf, _MM_HINT_T0);
    _mm_prefetch((const char*) blk + pf + 64, _MM_HINT_T0);
}

// ---- per format: one 32-value half (values 64*j + 32*half .. +31) -> grid magnitudes, sign vector, scales
template <int TY> struct Fmt32;

template <> struct Fmt32<16> {   // IQ2_XXS: d, qs[32] u16
    static constexpr int bytes = 66;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn, __m256i& sc) {
        const uint8_t* q = b + 2 + 16 * j + 8 * half;
        const uint32_t w0 = u32(q), w1 = u32(q + 4);
        g = _mm256_set_epi64x((long long) iq2xxs_grid[w0 >> 24], (long long) iq2xxs_grid[(w0 >> 16) & 255],
                              (long long) iq2xxs_grid[(w0 >> 8) & 255], (long long) iq2xxs_grid[w0 & 255]);
        sgn = _mm256_set_epi64x((long long) even_signs.v[(w1 >> 21) & 127], (long long) even_signs.v[(w1 >> 14) & 127],
                                (long long) even_signs.v[(w1 >> 7) & 127], (long long) even_signs.v[w1 & 127]);
        sc = sc32(2 * (int) (w1 >> 28) + 1);
    }
};

template <> struct Fmt32<17> {   // IQ2_XS: d, qs[32] u16 (9-bit grid index + 7-bit sign index), scales[8]
    static constexpr int bytes = 74;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn, __m256i& sc) {
        uint16_t v[8];
        std::memcpy(v, b + 2 + 16 * j, 16);
        const int o = 4 * half;
        g = _mm256_set_epi64x((long long) iq2xs_grid[v[o + 3] & 511], (long long) iq2xs_grid[v[o + 2] & 511],
                              (long long) iq2xs_grid[v[o + 1] & 511], (long long) iq2xs_grid[v[o] & 511]);
        uint32_t s = 0;
        for (int l = 0; l < 4; ++l) s |= (uint32_t) ksigns_iq2xs[v[o + l] >> 9] << (8 * l);
        sgn = sgn_vec(s);
        const uint8_t sb = b[66 + 2 * j + half];
        sc = sc16(2 * (sb & 15) + 1, 2 * (sb >> 4) + 1);
    }
};

template <> struct Fmt32<22> {   // IQ2_S: d, qs[64] (32 grid bytes, 32 sign bytes), qh[8], scales[8]
    static constexpr int bytes = 82;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn, __m256i& sc) {
        // grid_indices reads 8 index bytes; only the first 4 are this half's (the rest stay inside the block)
        alignas(16) uint16_t sp[8];
        grid_indices(b + 2 + 8 * j + 4 * half, hi_spread.iq2s[b[66 + 2 * j + half]], sp);
        g = _mm256_set_epi64x((long long) iq2s_grid[sp[3]], (long long) iq2s_grid[sp[2]],
                              (long long) iq2s_grid[sp[1]], (long long) iq2s_grid[sp[0]]);
        sgn = sgn_vec_at(b + 2 + 32 + 8 * j + 4 * half);
        sc = scale_vec(scale_vecs.s16[b[74 + 2 * j + half]]);
    }
};

template <> struct Fmt32<18> {   // IQ3_XXS: d, qs[64] grid bytes, 8 x u32 (4 x 7-bit sign index + 4-bit scale)
    static constexpr int bytes = 98;
    static constexpr float K = 0.25f;
    static inline void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn, __m256i& sc) {
        const uint8_t* q = b + 2 + 16 * j + 8 * half;
        g = _mm256_set_epi32((int) iq3xxs_grid[q[7]], (int) iq3xxs_grid[q[6]], (int) iq3xxs_grid[q[5]], (int) iq3xxs_grid[q[4]],
                             (int) iq3xxs_grid[q[3]], (int) iq3xxs_grid[q[2]], (int) iq3xxs_grid[q[1]], (int) iq3xxs_grid[q[0]]);
        const uint32_t w = u32(b + 2 + 64 + 8 * j + 4 * half);
        sgn = _mm256_set_epi64x((long long) even_signs.v[(w >> 21) & 127], (long long) even_signs.v[(w >> 14) & 127],
                                (long long) even_signs.v[(w >> 7) & 127], (long long) even_signs.v[w & 127]);
        sc = sc32(2 * (int) (w >> 28) + 1);
    }
};

template <> struct Fmt32<21> {   // IQ3_S: d, qs[64], qh[8], signs[32], scales[4]
    static constexpr int bytes = 110;
    static constexpr float K = 1.0f;
    static inline void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn, __m256i& sc) {
        const uint8_t* q = b + 2 + 16 * j + 8 * half;
        const uint32_t h = b[66 + 2 * j + half];
        alignas(16) uint16_t sp[8];
        grid_indices(q, hi_spread.iq3s[h], sp);
#define G3(k) (int) iq3s_grid[sp[k]]
        g = _mm256_set_epi32(G3(7), G3(6), G3(5), G3(4), G3(3), G3(2), G3(1), G3(0));
#undef G3
        sgn = sgn_vec_at(b + 74 + 8 * j + 4 * half);
        const uint8_t s = b[106 + j];
        sc = scale_vec(scale_vecs.s32[half ? s >> 4 : s & 15]);
    }
};

// ---- the gathered decodes (STRATA_IQ256_GATHER, see the top of the file): each its format's Fmt32 bit for bit
// The 32 sign bits of a half from the j's 8 sign bytes: one broadcast and one shuffle (sgn_vec: two and an insert).
inline __m256i sgn_half(const uint8_t* m8, int half) {
    const __m256i sb = _mm256_shuffle_epi8(
        _mm256_set1_epi64x((long long) u64(m8)),
        half ? _mm256_setr_epi8(4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7)
             : _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3));
    const __m256i sel = _mm256_set1_epi64x((long long) 0x8040201008040201ull);
    return _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(sb, sel), sel), _mm256_set1_epi8(1));
}
// The four 7-bit even-sign indices of a u32 (bits 0-6, 7-13, 14-20, 21-27) -> their four masks, one gather.
inline __m256i even_signs4(uint32_t w) {
    const __m128i idx = _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32((int) w), _mm_setr_epi32(0, 7, 14, 21)),
                                      _mm_set1_epi32(127));
    return _mm256_i32gather_epi64((const long long*) even_signs.v, idx, 8);
}

// IQ3_S: the eight 9-bit grid indices built in one register (8 low bits from qs, the 9th from bit k of qh) and read
// with one gather; the scalar form spends ~7 shuffle-port uops per half on vpinsrd and runs the index arithmetic on
// the scalar ports.  On a 14900KF's P-cores, gate/up rows of an IQ3_S expert at three tokens: 0.40 -> 0.27 ms.
template <> struct Fmt32<121> {
    static constexpr int bytes = 110;
    static constexpr float K = 1.0f;
    static inline void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn, __m256i& sc) {
        const uint8_t* q = b + 2 + 16 * j + 8 * half;
        const uint32_t h = b[66 + 2 * j + half];
        const __m256i lo = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) q));
        const __m256i hb = _mm256_and_si256(_mm256_sllv_epi32(_mm256_set1_epi32((int) h),
                                                              _mm256_setr_epi32(8, 7, 6, 5, 4, 3, 2, 1)),
                                            _mm256_set1_epi32(0x100));
        g = _mm256_i32gather_epi32((const int*) iq3s_grid, _mm256_or_si256(lo, hb), 4);
        sgn = sgn_half(b + 74 + 8 * j, half);
        const uint8_t s = b[106 + j];
        sc = sc32(half ? 2 * (s >> 4) + 1 : 2 * (s & 15) + 1);
    }
};

// IQ3_XXS: the grid entries and the four even-sign masks each by one gather.  P-core gate/up rows at three tokens:
// 0.26 -> 0.24 ms.
template <> struct Fmt32<118> {
    static constexpr int bytes = 98;
    static constexpr float K = 0.25f;
    static inline void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn, __m256i& sc) {
        const uint8_t* q = b + 2 + 16 * j + 8 * half;
        g = _mm256_i32gather_epi32((const int*) iq3xxs_grid, _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) q)), 4);
        const uint32_t w = u32(b + 2 + 64 + 8 * j + 4 * half);
        sgn = even_signs4(w);
        sc = sc32(2 * (int) (w >> 28) + 1);
    }
};

// IQ2_S: the four 10-bit grid indices (qs byte k, bits 2k and 2k+1 of qh at bits 8-9) by one 64-bit gather and the
// explicit sign bits from one broadcast.  P-core gate/up rows at three tokens: 0.31 -> 0.27 ms.  IQ2_XXS measured 4%
// slower gathered (its four 8-byte grid loads are few enough already) and IQ2_XS has its own sign decode, so both
// keep their Fmt32 everywhere.
template <> struct Fmt32<122> {
    static constexpr int bytes = 82;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn, __m256i& sc) {
        const uint8_t* qs = b + 2 + 8 * j + 4 * half;
        const uint32_t h = b[66 + 2 * j + half];
        const __m128i idx = _mm_or_si128(
            _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) u32(qs))),
            _mm_and_si128(_mm_sllv_epi32(_mm_set1_epi32((int) h), _mm_setr_epi32(8, 6, 4, 2)), _mm_set1_epi32(0x300)));
        g = _mm256_i32gather_epi64((const long long*) iq2s_grid, idx, 8);
        sgn = sgn_half(b + 2 + 32 + 8 * j, half);
        const uint8_t sb = b[74 + 2 * j + half];
        sc = sc16(2 * (sb & 15) + 1, 2 * (sb >> 4) + 1);
    }
};

template <> struct Fmt32<23> {   // IQ4_XS: d, scales_h, scales_l[4], qs[128] - 136 B, 8 signed sub-scales
    // ggml's ggml_vec_dot_iq4_xs_q8_K: the same 16-value codebook as IQ4_NL, but each of the eight 32-value
    // sub-blocks carries its own 6-bit scale, read as two nibbles of scales_l[p] plus two bits of scales_h, and
    // used SIGNED as (ls - 32).  The scale therefore folds into the int16 operand of madd_epi16 instead of
    // becoming a float multiply per sub-block, and the codebook sign is carried the way iq4nl_rows does it.
    static constexpr int bytes = 136;
    static constexpr float K = 1.0f;
    static inline void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn, __m256i& sc) {
        const int H = 2 * j + half;   // the eight 32-value sub-blocks, in the order the activation bytes come
        const __m128i values = _mm_loadu_si128((const __m128i*) kvalues_iq4nl);
        const __m128i bits = _mm_loadu_si128((const __m128i*) (b + 8 + 16 * H));
        const __m128i m4 = _mm_set1_epi8(0x0f);
        const __m256i q4 = _mm256_inserti128_si256(
            _mm256_castsi128_si256(_mm_shuffle_epi8(values, _mm_and_si128(bits, m4))),
            _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(bits, 4), m4)), 1);
        g   = _mm256_sign_epi8(q4, q4);                       // |w|, the unsigned operand of maddubs
        sgn = _mm256_sign_epi8(_mm256_set1_epi8(1), q4);      // w's sign, applied to the activation
        const int ls = ((b[4 + (H >> 1)] >> (4 * (H & 1))) & 0xf) | (((u16(b + 2) >> (2 * H)) & 3) << 4);
        sc = _mm256_set1_epi16((short) (ls - 32));
    }
};

// ---- the row kernels, twice: the AVX2 forms and the AVX-VNNI ones (iq_avx2_rows.inl)
namespace plain {
#define STRATA_ROWS_VNNI 0
#define STRATA_ROWS_FN
#include "iq_avx2_rows.inl"
}  // namespace plain
#if STRATA_AVXVNNI
namespace vnni {
#define STRATA_ROWS_VNNI 1
#define STRATA_ROWS_FN STRATA_AVXVNNI_FN
#include "iq_avx2_rows.inl"
}  // namespace vnni
#endif

bool vnni_on() { return STRATA_AVXVNNI && cpu_avxvnni_ok(); }

}  // namespace

// ggml's quantize_row_q8_K_ref (ggml-quants.c), which is also what ggml-cpu runs on x86 (no SIMD version there),
// in AVX-2: the same per-value arithmetic - one IEEE multiply by the same iscale, the same 1.5*2^23 rounding add,
// the same clamp - so the bytes are identical (q8k_quant_parity checks it).  The block's sign-carrying max is the
// FIRST value of the largest magnitude, as in the scalar loop; NaNs are skipped as there (`ax > amax` is false for
// them, and max_ps returns its second operand for a NaN first one).  A zero block leaves bsums as the reference
// does.  A verify window quantizes up to 6 tokens per layer on the host before the pool can start.
void q8k_quant_avx2(const float* x, void* vy, int64_t k) {
    block_q8_K* y = (block_q8_K*) vy;
    const int64_t nb = k / QK_K;
    const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 magic = _mm256_set1_ps(12582912.f);
    const __m256i mant = _mm256_set1_epi32(0x007fffff), off = _mm256_set1_epi32(0x00400000);
    const __m256i c127 = _mm256_set1_epi32(127);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (int64_t i = 0; i < nb; ++i, x += QK_K) {
        __m256 m = _mm256_setzero_ps();
        for (int j = 0; j < QK_K; j += 8) m = _mm256_max_ps(_mm256_and_ps(_mm256_loadu_ps(x + j), absm), m);
        __m128 h = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        h = _mm_max_ps(h, _mm_movehl_ps(h, h));
        h = _mm_max_ss(h, _mm_movehdup_ps(h));
        const float amax = _mm_cvtss_f32(h);
        if (!amax) {
            y[i].d = 0;
            std::memset(y[i].qs, 0, QK_K);
            continue;
        }
        float mx = 0.f;
        const __m256 va = _mm256_set1_ps(amax);
        for (int j = 0; j < QK_K; j += 8) {
            const int msk = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_and_ps(_mm256_loadu_ps(x + j), absm), va, _CMP_EQ_OQ));
            if (msk) {
                int b = 0;
                while (!((msk >> b) & 1)) ++b;
                mx = x[j + b];
                break;
            }
        }
        const float iscale = -127.f / mx;
        const __m256 vis = _mm256_set1_ps(iscale);
        for (int j = 0; j < QK_K; j += 32) {
            __m256i v[4];
            for (int q = 0; q < 4; ++q) {
                __m256 p = _mm256_mul_ps(vis, _mm256_loadu_ps(x + j + 8 * q));
#if defined(__GNUC__) && !defined(__clang__)
                // GCC fuses a multiply and an add into an FMA by default (-ffp-contract=fast); the reference rounds
                // the product before the add, so the product must stay a separate value
                __asm__("" : "+x"(p));
#endif
                const __m256 t = _mm256_add_ps(p, magic);
                v[q] = _mm256_min_epi32(_mm256_sub_epi32(_mm256_and_si256(_mm256_castps_si256(t), mant), off), c127);
                // the reference stores MIN(127, v) into an int8 (keeps the low byte), and its bsums add those bytes:
                // the same here, so even a degenerate block (iscale overflowing to inf) gives the same bytes
                v[q] = _mm256_srai_epi32(_mm256_slli_epi32(v[q], 24), 24);
            }
            for (int q = 0; q < 2; ++q) {   // two 16-value sums
                const __m256i s = _mm256_add_epi32(v[2 * q], v[2 * q + 1]);
                __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
                s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));
                s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xB1));
                y[i].bsums[j / 16 + q] = (int16_t) _mm_cvtsi128_si32(s4);
            }
            const __m256i p16a = _mm256_packs_epi32(v[0], v[1]), p16b = _mm256_packs_epi32(v[2], v[3]);
            const __m256i p8 = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(p16a, p16b), perm);
            _mm256_storeu_si256((__m256i*) (y[i].qs + j), p8);
        }
        y[i].d = 1 / iscale;
    }
}

void iq256_gu_rows_exact_one(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                             const void* act, float* ff, int r0, int r1) {
    const auto* y = (const block_q8_K*) act;
#if STRATA_AVXVNNI
    if (vnni_on()) {
        if (type == 16) vnni::gu_ggml_one<16>(blob, gu_row, up_off, n, y, ff, r0, r1);
        else if (type == 17) vnni::gu_ggml_one<17>(blob, gu_row, up_off, n, y, ff, r0, r1);
        else if (type == 18) {
            if (iq256_variant() & kIq256Gather) vnni::gu_ggml_one<118>(blob, gu_row, up_off, n, y, ff, r0, r1);
            else vnni::gu_ggml_one<18>(blob, gu_row, up_off, n, y, ff, r0, r1);
        } else if (type == 21) {
            if (iq256_variant() & kIq256Gather) vnni::gu_ggml_one<121>(blob, gu_row, up_off, n, y, ff, r0, r1);
            else vnni::gu_ggml_one<21>(blob, gu_row, up_off, n, y, ff, r0, r1);
        } else if (type == 22) {
            if (iq256_variant() & kIq256Gather) vnni::gu_ggml_one<122>(blob, gu_row, up_off, n, y, ff, r0, r1);
            else vnni::gu_ggml_one<22>(blob, gu_row, up_off, n, y, ff, r0, r1);
        }
        return;
    }
#endif
    if (type == 16) plain::gu_ggml_one<16>(blob, gu_row, up_off, n, y, ff, r0, r1);
    else if (type == 17) plain::gu_ggml_one<17>(blob, gu_row, up_off, n, y, ff, r0, r1);
    else if (type == 18) {
        if (iq256_variant() & kIq256Gather) plain::gu_ggml_one<118>(blob, gu_row, up_off, n, y, ff, r0, r1);
        else plain::gu_ggml_one<18>(blob, gu_row, up_off, n, y, ff, r0, r1);
    } else if (type == 21) {
        if (iq256_variant() & kIq256Gather) plain::gu_ggml_one<121>(blob, gu_row, up_off, n, y, ff, r0, r1);
        else plain::gu_ggml_one<21>(blob, gu_row, up_off, n, y, ff, r0, r1);
    } else if (type == 22) {
        if (iq256_variant() & kIq256Gather) plain::gu_ggml_one<122>(blob, gu_row, up_off, n, y, ff, r0, r1);
        else plain::gu_ggml_one<22>(blob, gu_row, up_off, n, y, ff, r0, r1);
    }
}

int iq256_variant() noexcept {
    // STRATA_IQ256_GATHER=0/1 for every core; unset, where the calling thread's core gathers faster (per thread: a
    // hybrid CPU's pool runs on P- and E-cores).  AVX-VNNI where cpu_avxvnni_ok().  The same bits either way.
    const int s = iq256_gather_setting();
    return ((s >= 0 ? s == 1 : cpu_gather_fast_here()) ? kIq256Gather : 0) | (vnni_on() ? kIq256Vnni : 0);
}

int iq256_variants() noexcept { return kIq256Gather | (vnni_on() ? kIq256Vnni : 0); }

void iq256_gu_rows_v(int variant, int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                     const void* const* act, int nt, float* const* ff, int r0, int r1) {
    const bool g = (variant & kIq256Gather) != 0;
#if STRATA_AVXVNNI
    if (variant & kIq256Vnni) {
        if (g) vnni::gu_type<true>(type, nt, blob, gu_row, up_off, n, act, ff, r0, r1);
        else vnni::gu_type<false>(type, nt, blob, gu_row, up_off, n, act, ff, r0, r1);
        return;
    }
#endif
    if (g) plain::gu_type<true>(type, nt, blob, gu_row, up_off, n, act, ff, r0, r1);
    else plain::gu_type<false>(type, nt, blob, gu_row, up_off, n, act, ff, r0, r1);
}

void iq256_rows_v(int variant, int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                  float* const* out, int r0, int r1) {
    const bool g = (variant & kIq256Gather) != 0;
#if STRATA_AVXVNNI
    if (variant & kIq256Vnni) {
        if (g) vnni::dot_type<true>(type, nt, w, row_bytes, n, act, out, r0, r1);
        else vnni::dot_type<false>(type, nt, w, row_bytes, n, act, out, r0, r1);
        return;
    }
#endif
    if (g) plain::dot_type<true>(type, nt, w, row_bytes, n, act, out, r0, r1);
    else plain::dot_type<false>(type, nt, w, row_bytes, n, act, out, r0, r1);
}

void iq256_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1) {
    iq256_gu_rows_v(iq256_variant(), type, blob, gu_row, up_off, n, act, nt, ff, r0, r1);
}

void iq256_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                int r0, int r1) {
    iq256_rows_v(iq256_variant(), type, w, row_bytes, n, act, nt, out, r0, r1);
}

bool iq256x_supported(int type) noexcept { return type == 16 || type == 17 || type == 18 || type == 21 || type == 22 || type == 29; }

void iq256x_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                 int r0, int r1) {
    const int variant = iq256_variant();
    const bool g = (variant & kIq256Gather) != 0;
#if STRATA_AVXVNNI
    if (variant & kIq256Vnni) {
        if (g) vnni::dot_type<true, true>(type, nt, w, row_bytes, n, act, out, r0, r1);
        else vnni::dot_type<false, true>(type, nt, w, row_bytes, n, act, out, r0, r1);
        return;
    }
#endif
    if (g) plain::dot_type<true, true>(type, nt, w, row_bytes, n, act, out, r0, r1);
    else plain::dot_type<false, true>(type, nt, w, row_bytes, n, act, out, r0, r1);
}

void iq4nl256_down_rows_v(int variant, const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt,
                          float* const* out, int r0, int r1) {
    const block_q8_0* y[8];
    for (int t = 0; t < nt; ++t) y[t] = (const block_q8_0*) hq[t];
#if STRATA_AVXVNNI
    if (variant & kIq256Vnni) {
        vnni::iq4nl_nt(w, row_bytes, n, y, nt, out, r0, r1);
        return;
    }
#endif
    (void) variant;
    plain::iq4nl_nt(w, row_bytes, n, y, nt, out, r0, r1);
}

void iq4nl256_down_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt, float* const* out,
                        int r0, int r1) {
    iq4nl256_down_rows_v(vnni_on() ? kIq256Vnni : 0, w, row_bytes, n, hq, nt, out, r0, r1);
}


// Singleton-equivalent down accumulation; legacy/VNNI kernels above keep their order.
namespace {
namespace singleton_iq4nl {

float h2f(uint16_t x) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x)));
}
float hsum(__m256 x) {
    __m128 s = _mm256_extractf128_ps(x, 1);
    s = _mm_add_ps(s, _mm256_castps256_ps128(x));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}
template <int NT>
void rows(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, float* const* out, int r0, int r1) {
    const __m128i values = _mm_loadu_si128((const __m128i*)kvalues_iq4nl), mask = _mm_set1_epi8(15);
    const __m256i ones = _mm256_set1_epi16(1);
    const int nb = n / QK4_NL;
    for (int r = r0; r < r1; ++r) {
        auto* x = (const block_iq4_nl*)(w + r * row_bytes);
        __m256 even[NT], odd[NT];
        for (int t = 0; t < NT; ++t) {
            even[t] = _mm256_setzero_ps();
            odd[t] = _mm256_setzero_ps();
        }
        int ib = 0;
        for (; ib + 1 < nb; ib += 2) {
            const __m128i b0 = _mm_loadu_si128((const __m128i*)x[ib].qs),
                          b1 = _mm_loadu_si128((const __m128i*)x[ib + 1].qs);
            const __m256i w0 =
                _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_shuffle_epi8(values, _mm_and_si128(b0, mask))),
                                        _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(b0, 4), mask)), 1);
            const __m256i w1 =
                _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_shuffle_epi8(values, _mm_and_si128(b1, mask))),
                                        _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(b1, 4), mask)), 1);
            const __m256i a0 = _mm256_sign_epi8(w0, w0), a1 = _mm256_sign_epi8(w1, w1);
            const float d0 = h2f(x[ib].d), d1 = h2f(x[ib + 1].d);
            for (int t = 0; t < NT; ++t) {
                auto* y = (const block_q8_0*)hq[t];
                auto p0 = _mm256_madd_epi16(
                    _mm256_maddubs_epi16(a0, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)y[ib].qs), w0)), ones);
                auto p1 = _mm256_madd_epi16(
                    _mm256_maddubs_epi16(a1, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)y[ib + 1].qs), w1)),
                    ones);
                even[t] = _mm256_fmadd_ps(_mm256_set1_ps(h2f(y[ib].d) * d0), _mm256_cvtepi32_ps(p0), even[t]);
                odd[t] = _mm256_fmadd_ps(_mm256_set1_ps(h2f(y[ib + 1].d) * d1), _mm256_cvtepi32_ps(p1), odd[t]);
            }
        }
        for (int t = 0; t < NT; ++t) {
            float s = hsum(_mm256_add_ps(even[t], odd[t]));
            if (ib < nb) {
                auto* y = (const block_q8_0*)hq[t];
                int a = 0, b = 0;
                for (int j = 0; j < QK4_NL / 2; ++j) {
                    a += kvalues_iq4nl[x[ib].qs[j] & 15] * y[ib].qs[j];
                    b += kvalues_iq4nl[x[ib].qs[j] >> 4] * y[ib].qs[j + QK4_NL / 2];
                }
                const float d = h2f(y[ib].d) * h2f(x[ib].d);
                s += d * (a + b);
            }
            out[t][r] = s;
        }
    }
}
} // namespace singleton_iq4nl
} // namespace
void iq4nl_reference_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt, float* const* out,
                          int r0, int r1) {
    for (int at = 0; at < nt; at += 4) {
        int k = nt - at < 4 ? nt - at : 4;
        switch (k) {
        case 1:
            singleton_iq4nl::rows<1>(w, row_bytes, n, hq + at, out + at, r0, r1);
            break;
        case 2:
            singleton_iq4nl::rows<2>(w, row_bytes, n, hq + at, out + at, r0, r1);
            break;
        case 3:
            singleton_iq4nl::rows<3>(w, row_bytes, n, hq + at, out + at, r0, r1);
            break;
        case 4:
            singleton_iq4nl::rows<4>(w, row_bytes, n, hq + at, out + at, r0, r1);
            break;
        }
    }
}

} // namespace strata::kernels::cpu
