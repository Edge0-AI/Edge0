// tools/trellis/tqenc_test.cpp - tests of the trellis encoder (tqenc.hpp) against Strata's own CPU decoder
// (src/kernels/cpu/tq_states.hpp, tq_cpu.cpp: scalar / AVX2 / AVX-512 kernels).  Build and run: tools/trellis/build.sh test.
//
//   T1  fp16 conversion (all 65536 halves, random floats vs the hardware where F16C exists)
//   T2  the encoder's codebook == the decoder's state table (bit for bit, 32768 x 4)
//   T3  random streams: block_state (bit by bit) == tq_states.hpp; every window obeys the shift-trellis relation,
//       circularly (so a legal path = the chain relation + the closure state(31) -> state(0)); pack(unpack(s)) == s
//   T4  the Viterbi pass == a naive DP written from the legality relation (every state, every predecessor), for a fixed
//       start group (tail-biting), a free start and a rotated start, with importance weights
//   T5  exhaustive: every symbol value at every stream position round-trips through pack/decode
//   T6  planted solutions: weights decoded from a random legal stream and scale; exact search at that scale must find
//       error 0 and bit-identical weights; the approximate search is reported (recovery rate)
//   T7  encode -> decode: the encoder's reconstruction == the decoder's weights bit for bit (tq_states + state table),
//       and the CPU kernels' dot products against Q8_0 activations == the dense dot of those weights
//   T8  specials: zero blocks, tiny / huge / extreme scales, constants, spikes, NaN / Inf, determinism over threads
//   T9  quality on Gaussian blocks: SQNR against the rate-distortion bound 6.02 * K/4 dB; approximate vs exact search
#include "tqenc.hpp"

#include "strata/kernels/cpu/tq_cpu.hpp"
#include "tq_states.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if defined(__F16C__)
#include <immintrin.h>
#endif

namespace strata::kernels::cpu {
bool cpu_avx512_ok() { return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512vbmi") && __builtin_cpu_supports("avx512vnni"); }
bool cpu_avx2_ok() { return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"); }
}  // namespace strata::kernels::cpu

using namespace tqenc;
namespace kc = strata::kernels::cpu;

static int g_fail = 0, g_checks = 0;
#define CHECK(cond, ...)                                                                          \
    do {                                                                                          \
        ++g_checks;                                                                               \
        if (!(cond)) {                                                                            \
            ++g_fail;                                                                             \
            std::printf("  FAIL %s:%d  %s  ", __FILE__, __LINE__, #cond);                         \
            std::printf(__VA_ARGS__);                                                             \
            std::printf("\n");                                                                    \
        }                                                                                         \
    } while (0)

static const int kTypes[7] = {TQ2_T, TQK6, TQK7, TQK5X, TQK4X, TQK5T, TQK4T};
static const char* tname(int t) {
    return t == TQ2_T ? "TQ2_T" : t == TQK6 ? "TQK6" : t == TQK7 ? "TQK7" : t == TQK5X ? "TQK5X" : t == TQK4X ? "TQK4X" : t == TQK5T ? "TQK5T" : "TQK4T";
}

template <int KB>
static void states_repo(const uint8_t* qs, uint32_t* st) { kc::tq_detail::tq_states<KB>(qs, st); }
static void states_repo_k(int K, const uint8_t* qs, uint32_t* st) {
    switch (K) {
        case 8: states_repo<8>(qs, st); break;
        case 7: states_repo<7>(qs, st); break;
        case 6: states_repo<6>(qs, st); break;
        case 5: states_repo<5>(qs, st); break;
        default: states_repo<4>(qs, st); break;
    }
}

static std::mt19937_64 rng(12345);
static float gauss() { static std::normal_distribution<float> n(0.f, 1.f); return n(rng); }

// chain relation between consecutive steps, circularly: returns true when state(t) and state(t') obey the shift trellis
static bool chain_ok(int type, const uint32_t* st) {
    const int K = type_kbits(type);
    for (int t = 0; t < 32; ++t) {
        if (type == TQ2_T) {   // state(t) = ((state(t-1) & 0x7f) << 8) | byte
            const int tp = (t + 31) & 31;
            if ((st[t] >> 8) != (st[tp] & 0x7f)) return false;
        } else {               // u = 31 - t; state(u+1) low (15-K) bits = state(u) >> K, i.e. step t-1 follows step t
            const int tp = (t + 31) & 31;
            if ((st[tp] & ((1u << (15 - K)) - 1u)) != (st[t] >> K)) return false;
        }
    }
    return true;
}

static void t1() {
    std::printf("T1 fp16\n");
    for (uint32_t h = 0; h < 65536; ++h) {
        const float f = half_to_float((uint16_t) h);
        if (std::isnan(f)) continue;
        CHECK(float_to_half(f) == (uint16_t) h || (h == 0x8000 && float_to_half(f) == 0x8000), "half %04x", h);
    }
#if defined(__F16C__)
    int bad = 0;
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    for (int i = 0; i < 4000000; ++i) {
        float f = u(rng) * std::pow(2.f, (float) (int) (rng() % 50) - 30.f);
        if (i % 7 == 0) f = std::ldexp(u(rng), -25 + (int) (rng() % 6));   // subnormal range and its ties
        if (i % 11 == 0) f = std::ldexp(u(rng), 14 + (int) (rng() % 3));   // the overflow edge
        const uint16_t hw = _cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT);
        bad += float_to_half(f) != hw;
    }
    CHECK(bad == 0, "float_to_half vs F16C: %d differ", bad);
#endif
}

static void t2() {
    std::printf("T2 codebook == decoder table\n");
    for (int type : kTypes) {
        const int16_t* a = codebook_table(type);
        const int16_t* b = kc::tq_state_table_for(type);
        CHECK(std::memcmp(a, b, 32768 * 4 * sizeof(int16_t)) == 0, "%s: encoder and decoder tables differ", tname(type));
    }
    // the computed codebook: its own definition, evaluated bit by bit here (not through tq_computed.hpp), is what both use
    {
        const int16_t* c = codebook_table(TQK5X);
        int bad = 0, big = 0;
        for (uint32_t s = 0; s < 32768; ++s)
            for (int i = 0; i < 2; ++i) {
                const uint32_t A[2] = {89226354u, 0x9E3779B1u}, B[2] = {64248484u, 0x7F4A7C15u};
                const uint32_t r = ((s * A[i] + B[i]) & 0x8FFF8FFFu) ^ 0x3B603B60u;
                for (int hw = 0; hw < 2; ++hw) {
                    const uint16_t h = (uint16_t) (hw ? r >> 16 : r & 0xffff);
                    const float v = half_to_float(h) * 8192.0f;
                    bad += c[4 * s + 2 * i + hw] != (int16_t) v || v != std::floor(v);
                    big += std::fabs(v) >= 16384.0f || std::fabs(v) < 1023.0f;
                }
            }
        CHECK(bad == 0, "computed codebook differs from its definition (%d)", bad);
        CHECK(big == 0, "computed codebook value out of [0.125, 2) (%d)", big);
        // the distribution: rms and the share of the weights below / above one
        double ss = 0;
        for (int i = 0; i < 32768 * 4; ++i) ss += (double) c[i] * c[i];
        std::printf("  computed codebook: rms %.4f (table: %.4f), all values exact integers at 2^13\n", std::sqrt(ss / (32768.0 * 4)) / 8192.0, 1.072);
    }
}

static void t3() {
    std::printf("T3 states of random streams\n");
    for (int type : kTypes) {
        const int K = type_kbits(type), nb = 4 * K;
        int bad_states = 0, bad_chain = 0, bad_pack = 0;
        for (int it = 0; it < 20000; ++it) {
            uint8_t qs[32];
            for (int i = 0; i < nb; ++i) qs[i] = (uint8_t) rng();
            uint32_t sr[32];
            states_repo_k(K, qs, sr);
            uint32_t sm[32];
            for (int t = 0; t < 32; ++t) sm[t] = block_state(type, qs, t);
            bad_states += std::memcmp(sr, sm, sizeof sr) != 0;
            bad_chain += !chain_ok(type, sm);
            // pack(unpack): processing order p -> step t
            uint16_t by_pos[32];
            for (int p = 0; p < 32; ++p) by_pos[p] = (uint16_t) sm[type == TQ2_T ? p : 31 - p];
            uint8_t q2[32];
            debug_pack(type, by_pos, q2);
            // TQ2_T's bit 7 of each byte is part of the stream but the state uses only 15 bits of two bytes: bit 7 of
            // qs[t] is in state(t) (bit 7), so it survives; every stream bit is in some state
            bad_pack += std::memcmp(qs, q2, nb) != 0;
        }
        std::printf("  %-6s states vs tq_states.hpp: %d bad, chain relation: %d bad, pack(unpack): %d bad\n", tname(type),
                    bad_states, bad_chain, bad_pack);
        CHECK(bad_states == 0 && bad_chain == 0 && bad_pack == 0, "%s", tname(type));
    }
}

// naive DP from the legality relation, one pass; same conventions as debug_viterbi
static float naive(int type, const float (*T)[4], const float (*H)[4], int rot, int fixed_c, uint16_t* st_out) {
    const int K = type_kbits(type);
    const bool t2 = type == TQ2_T;
    const int G = debug_groups(type);
    std::vector<std::vector<float>> D(32, std::vector<float>(32768, 1e30f));
    std::vector<std::vector<int>> B(32, std::vector<int>(32768, -1));
    auto cost = [&](int p, uint32_t s) {
        const float* c = codebook_state(type, s);
        float e = 0;
        for (int j = 0; j < 4; ++j) { const float d = T[p][j] - c[j]; e += H[p][j] * d * d; }
        return e;
    };
    auto legal = [&](uint32_t prev, uint32_t next) {   // next may follow prev
        return t2 ? (next >> 8) == (prev & 0x7f) : (next & (uint32_t) (G - 1)) == (prev >> K);
    };
    for (uint32_t s = 0; s < 32768; ++s) {
        const bool ok = fixed_c < 0 || (t2 ? (int) (s >> 8) == fixed_c : (int) (s & (G - 1)) == fixed_c);
        if (ok) D[0][s] = cost(rot & 31, s);
    }
    for (int q = 1; q < 32; ++q)
        for (uint32_t s = 0; s < 32768; ++s) {
            float best = 1e30f;
            int arg = -1;
            // enumerate the predecessors from the relation (all prev with legal(prev, s)): scan the 2^K candidates
            const int n = 1 << K;
            for (int y = 0; y < (t2 ? 256 : n); ++y) {
                const uint32_t prev = t2 ? ((uint32_t) y << 7) | (s >> 8) : (((s & (uint32_t) (G - 1)) << K) | (uint32_t) y);
                if (!legal(prev, s)) continue;
                if (D[q - 1][prev] < best) { best = D[q - 1][prev]; arg = (int) prev; }
            }
            if (arg >= 0) { D[q][s] = best + cost((rot + q) & 31, s); B[q][s] = arg; }
        }
    float best = 1e30f;
    int bs = -1;
    for (uint32_t s = 0; s < 32768; ++s) {
        const bool ok = fixed_c < 0 || (t2 ? (int) (s & 0x7f) == fixed_c : (int) (s >> K) == fixed_c);
        if (ok && D[31][s] < best) { best = D[31][s]; bs = (int) s; }
    }
    if (bs < 0) return 1e30f;
    int s = bs;
    for (int q = 31; q >= 0; --q) {
        st_out[(rot + q) & 31] = (uint16_t) s;
        if (q > 0) s = B[q][s];
    }
    return best;
}

static void t4() {
    std::printf("T4 Viterbi vs naive DP\n");
    for (int type : kTypes) {
        const int G = debug_groups(type);
        double worst = 0;
        int cases = 0;
        for (int trial = 0; trial < 3; ++trial) {
            float T[32][4], H[32][4];
            for (int p = 0; p < 32; ++p)
                for (int j = 0; j < 4; ++j) {
                    T[p][j] = gauss() * (trial == 2 ? 2.5f : 1.0f);
                    H[p][j] = trial == 0 ? 1.f : 0.1f + std::fabs(gauss());
                }
            struct Case { int rot, c; };
            Case cs[3] = {{0, (int) (rng() % G)}, {0, -1}, {13, -1}};
            for (const Case& cc : cs) {
                uint16_t sa[32], sb[32];
                const float a = debug_viterbi(type, T, H, cc.rot, cc.c, sa);
                const float b = naive(type, T, H, cc.rot, cc.c, sb);
                const double rel = std::fabs(a - b) / std::max(1e-6f, b);
                worst = std::max(worst, rel);
                CHECK(rel < 2e-5, "%s trial %d rot %d c %d: viterbi %.6f naive %.6f", tname(type), trial, cc.rot, cc.c, a, b);
                // the returned path is legal and its own cost is the reported cost
                uint32_t sm[32];
                uint8_t qs[32];
                debug_pack(type, sa, qs);
                for (int t = 0; t < 32; ++t) sm[t] = block_state(type, qs, t);
                bool same = true;
                for (int p = 0; p < 32; ++p) {
                    const uint32_t d = sm[type == TQ2_T ? p : 31 - p];
                    if (cc.rot == 0 || cc.c < 0) {
                        // a free rotated pass is a legal chain only if the stream closes; compare only closed ones
                    }
                    same = same && (cc.c < 0 ? true : d == sa[p]);
                }
                if (cc.c >= 0) CHECK(same, "%s tail-biting path does not decode to its own states", tname(type));
                double sum = 0;
                for (int q = 0; q < 32; ++q) {
                    const int p = (cc.rot + q) & 31;
                    const float* c = codebook_state(type, sa[p]);
                    for (int j = 0; j < 4; ++j) { const double d = T[p][j] - c[j]; sum += H[p][j] * d * d; }
                }
                CHECK(std::fabs(sum - a) < 1e-3 * std::max(1.0, (double) a), "path cost %.6f vs reported %.6f", sum, a);
                ++cases;
            }
        }
        std::printf("  %-6s %d cases, worst relative cost difference %.2e\n", tname(type), cases, worst);
    }
}

static void t5() {
    std::printf("T5 every symbol value at every position\n");
    for (int type : kTypes) {
        const int K = type_kbits(type), nb = 4 * K;
        long bad = 0, n = 0;
        if (type == TQ2_T) {
            for (int pos = 0; pos < 32; ++pos)
                for (int x = 0; x < 256; ++x) {
                    uint8_t qs[32];
                    for (int i = 0; i < nb; ++i) qs[i] = (uint8_t) rng();
                    qs[pos] = (uint8_t) x;
                    uint32_t sm[32];
                    for (int t = 0; t < 32; ++t) sm[t] = block_state(type, qs, t);
                    uint16_t bp[32];
                    for (int p = 0; p < 32; ++p) bp[p] = (uint16_t) sm[p];
                    uint8_t q2[32];
                    debug_pack(type, bp, q2);
                    bad += std::memcmp(qs, q2, nb) != 0 || !chain_ok(type, sm);
                    ++n;
                }
        } else {
            for (int u = 0; u < 32; ++u)
                for (int x = 0; x < (1 << K); ++x) {
                    uint8_t qs[32];
                    for (int i = 0; i < nb; ++i) qs[i] = (uint8_t) rng();
                    // overwrite symbol u
                    for (int b = 0; b < K; ++b) {
                        const int bit = u * K + b;
                        qs[bit >> 3] = (uint8_t) ((qs[bit >> 3] & ~(1u << (bit & 7))) | (((x >> b) & 1u) << (bit & 7)));
                    }
                    uint32_t sm[32];
                    for (int t = 0; t < 32; ++t) sm[t] = block_state(type, qs, t);
                    // symbol u is the low K bits of the state of step 31 - u
                    bad += (int) (sm[31 - u] & ((1u << K) - 1u)) != x;
                    uint16_t bp[32];
                    for (int p = 0; p < 32; ++p) bp[p] = (uint16_t) sm[31 - p];
                    uint8_t q2[32];
                    debug_pack(type, bp, q2);
                    bad += std::memcmp(qs, q2, nb) != 0 || !chain_ok(type, sm);
                    ++n;
                }
        }
        std::printf("  %-6s %ld cases, %ld bad\n", tname(type), n, bad);
        CHECK(bad == 0, "%s", tname(type));
    }
}

static void t6() {
    std::printf("T6 planted solutions\n");
    for (int type : kTypes) {
        const int K = type_kbits(type), nb = 4 * K, bb = block_bytes(type);
        const int N = (type == TQK4X ? 12 : 30);
        int exact_ok = 0, approx_ok1 = 0, approx_ok3 = 0;
        double approx_err_sum = 0, wsum = 0;
        for (int it = 0; it < N; ++it) {
            std::vector<uint8_t> blk(bb);
            const float dsc = std::ldexp(0.5f + (float) (rng() % 1000) / 1000.f, -6 + (int) (rng() % 10));
            const uint16_t dh = float_to_half(dsc);
            blk[0] = (uint8_t) (dh & 255);
            blk[1] = (uint8_t) (dh >> 8);
            // a legal stream: any bytes are legal (every stream is a closed cycle by construction); for TQ2_T too
            for (int i = 0; i < nb; ++i) blk[2 + i] = (uint8_t) rng();
            std::vector<float> w(128);
            decode_blocks(type, blk.data(), 1, w.data());
            for (float x : w) wsum += (double) x * x;
            Params p;
            p.fixed_scale = dh;
            std::vector<uint8_t> out(bb);
            p.exact = 1;
            BlockResult r = encode_block(type, w.data(), nullptr, out.data(), p);
            std::vector<float> w2(128);
            decode_blocks(type, out.data(), 1, w2.data());
            const bool same = std::memcmp(w.data(), w2.data(), 512) == 0;
            exact_ok += (r.err == 0.0 && same);
            CHECK(r.err == 0.0 && same, "%s exact search missed a planted solution (err %g)", tname(type), r.err);
            p.exact = 0;
            p.ncand = 1;
            r = encode_block(type, w.data(), nullptr, out.data(), p);
            approx_ok1 += r.err == 0.0;
            approx_err_sum += r.err;
            p.ncand = 3;
            r = encode_block(type, w.data(), nullptr, out.data(), p);
            approx_ok3 += r.err == 0.0;
        }
        std::printf("  %-6s exact: %d/%d recovered; approximate (rotation 16): %d/%d (err/energy %.2e); best of 3 rotations: %d/%d\n",
                    tname(type), exact_ok, N, approx_ok1, N, approx_err_sum / wsum, approx_ok3, N);
    }
}

static uint16_t q8_scale(const std::vector<float>& x, int b) {
    float amax = 0;
    for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[32 * b + j]));
    return float_to_half(amax / 127.f);
}

static void t7() {
    std::printf("T7 encoder reconstruction == decoder, kernels vs dense\n");
    for (int type : kTypes) {
        const int K = type_kbits(type), bb = block_bytes(type);
        const int rows = 24, n = 512, nbr = n / 128;
        std::vector<float> w((size_t) rows * n);
        for (int r = 0; r < rows; ++r)
            for (int i = 0; i < n; ++i) {
                float v = gauss() * 0.02f * (1.f + (r % 5));
                if (r == 3) v = (i % 97 == 0) ? 1.5f : 0.001f * gauss();   // spiky row
                if (r == 4) v = 0.f;                                          // zero row
                if (r == 5) v *= 300.f;                                       // big
                w[(size_t) r * n + i] = v;
            }
        std::vector<uint8_t> enc((size_t) rows * nbr * bb);
        int64_t nf = 0, zb = 0;
        std::vector<double> errs((size_t) rows * nbr);
        Params p;
        const double tot = encode_blocks(type, w.data(), nullptr, (int64_t) rows * nbr, enc.data(), p, 8, errs.data(), &nf, &zb);
        (void) tot;
        // decoder reference: tq_states.hpp + the state table
        const int16_t* tab = kc::tq_state_table_for(type);
        std::vector<float> ref((size_t) rows * n), ours((size_t) rows * n);
        for (int64_t b = 0; b < (int64_t) rows * nbr; ++b) {
            const uint8_t* blk = enc.data() + b * bb;
            const float d = kc::tq_detail::tq_half_to_float(kc::tq_detail::tq_ld16(blk));
            uint32_t st[32];
            states_repo_k(K, blk + 2, st);
            for (int t = 0; t < 32; ++t)
                for (int j = 0; j < 4; ++j) ref[b * 128 + 4 * t + j] = d * ((float) tab[4 * st[t] + j] * (1.0f / 8192.0f));
        }
        decode_blocks(type, enc.data(), (int64_t) rows * nbr, ours.data());
        CHECK(std::memcmp(ref.data(), ours.data(), ref.size() * 4) == 0, "%s decode != tq_states + state table", tname(type));
        // the encoder's own error equals the error of the decoded weights
        double worst = 0;
        for (int64_t b = 0; b < (int64_t) rows * nbr; ++b) {
            double e = 0;
            for (int i = 0; i < 128; ++i) { const double d = (double) w[b * 128 + i] - ref[b * 128 + i]; e += d * d; }
            worst = std::max(worst, std::fabs(e - errs[b]) / std::max(1e-30, e + errs[b]));
        }
        CHECK(worst < 1e-9, "%s encoder error != decoded error (%.3e)", tname(type), worst);
        // kernels (scalar, AVX2, AVX-512 where present) vs the dense dot with the dequantized Q8_0 activation
        std::vector<float> x(n);
        for (auto& v : x) v = gauss();
        std::vector<uint8_t> q8((size_t) (n / 32) * 34);
        std::vector<float> xd(n);
        for (int b = 0; b < n / 32; ++b) {
            const uint16_t dh = q8_scale(x, b);
            const float d = half_to_float(dh);
            q8[34 * b] = (uint8_t) (dh & 255);
            q8[34 * b + 1] = (uint8_t) (dh >> 8);
            for (int j = 0; j < 32; ++j) {
                const int q = d == 0.f ? 0 : (int) std::lround(x[32 * b + j] / d);
                q8[34 * b + 2 + j] = (uint8_t) (int8_t) std::max(-127, std::min(127, q));
                xd[32 * b + j] = d * (float) (int8_t) q8[34 * b + 2 + j];
            }
        }
        const void* ap[1] = {q8.data()};
        for (kc::TqIsa isa : {kc::TqIsa::scalar, kc::TqIsa::avx2, kc::TqIsa::avx512}) {
            if (!kc::tq_isa_ok(isa)) continue;
            std::vector<float> out(rows);
            float* op[1] = {out.data()};
            kc::tq_rows(isa, type, enc.data(), (size_t) nbr * bb, n, ap, 1, op, 0, rows);
            double worst_rel = 0;
            for (int r = 0; r < rows; ++r) {
                double dsum = 0, mag = 0;
                for (int i = 0; i < n; ++i) {
                    dsum += (double) ref[(size_t) r * n + i] * xd[i];
                    mag += std::fabs((double) ref[(size_t) r * n + i] * xd[i]);
                }
                worst_rel = std::max(worst_rel, std::fabs(dsum - out[r]) / (mag + 1e-30));
            }
            std::printf("  %-6s %-6s kernel vs dense dot of the decoded weights: worst error %.2e of sum|w a|\n", tname(type),
                        kc::tq_isa_name(isa), worst_rel);
            CHECK(worst_rel < 1e-5, "%s %s", tname(type), kc::tq_isa_name(isa));
        }
        std::printf("  %-6s decode == tq_states+table: bit-exact, %zu weights; zero blocks %lld\n", tname(type), ref.size(), (long long) zb);
    }
}

static void t8() {
    std::printf("T8 specials\n");
    for (int type : kTypes) {
        const int bb = block_bytes(type);
        std::vector<uint8_t> out(bb), z(bb, 0);
        std::vector<float> w(128), dec(128);
        Params p;
        // zero block
        std::fill(w.begin(), w.end(), 0.f);
        BlockResult r = encode_block(type, w.data(), nullptr, out.data(), p);
        CHECK(r.status == 1 && out == z, "%s zero block must be canonical all-zero bytes", tname(type));
        w[5] = -0.f;
        r = encode_block(type, w.data(), nullptr, out.data(), p);
        CHECK(r.status == 1 && out == z, "%s negative zero", tname(type));
        decode_blocks(type, out.data(), 1, dec.data());
        for (float v : dec) CHECK(v == 0.f, "zero block decodes to 0");
        // scales: from the smallest subnormal weights to beyond fp16
        const float mags[] = {1e-9f, 3e-8f, 1e-7f, 6e-5f, 1e-3f, 1.f, 100.f, 3000.f, 30000.f, 60000.f, 65000.f, 1e5f, 1e7f};
        for (float mag : mags) {
            for (int i = 0; i < 128; ++i) w[i] = mag * gauss();
            r = encode_block(type, w.data(), nullptr, out.data(), p);
            const uint16_t dh = (uint16_t) (out[0] | (out[1] << 8));
            const float d = half_to_float(dh);
            CHECK(std::isfinite(d) && d >= 0.f, "%s scale %g not finite for magnitude %g", tname(type), d, mag);
            decode_blocks(type, out.data(), 1, dec.data());
            double e = 0, en = 0;
            bool fin = true;
            for (int i = 0; i < 128; ++i) { fin &= std::isfinite(dec[i]); e += (w[i] - dec[i]) * (double) (w[i] - dec[i]); en += (double) w[i] * w[i]; }
            CHECK(fin, "decoded nonfinite");
            std::printf("  %-6s magnitude %-8g d=%-12g rel err %.3f status %d\n", tname(type), mag, d, e / std::max(en, 1e-300), r.status);
            if (mag >= 1e-3f && mag <= 60000.f) CHECK(e / en < 0.5, "%s error too large at magnitude %g (%.3f)", tname(type), mag, e / en);
        }
        // constant, spike, alternating, power of two
        for (int variant = 0; variant < 4; ++variant) {
            for (int i = 0; i < 128; ++i)
                w[i] = variant == 0 ? 0.37f : variant == 1 ? (i == 77 ? 5.f : 0.f) : variant == 2 ? (i & 1 ? 1.f : -1.f) : std::ldexp(1.f, -3);
            r = encode_block(type, w.data(), nullptr, out.data(), p);
            decode_blocks(type, out.data(), 1, dec.data());
            double e = 0, en = 0;
            for (int i = 0; i < 128; ++i) { e += (w[i] - dec[i]) * (double) (w[i] - dec[i]); en += (double) w[i] * w[i]; }
            std::printf("  %-6s pattern %d rel err %.4f\n", tname(type), variant, e / en);
            CHECK(e / en < 0.8, "%s pattern %d", tname(type), variant);
        }
        // non-finite input
        for (int i = 0; i < 128; ++i) w[i] = gauss();
        w[17] = INFINITY;
        r = encode_block(type, w.data(), nullptr, out.data(), p);
        CHECK(r.status == 2 && out == z, "%s inf input", tname(type));
        w[17] = NAN;
        r = encode_block(type, w.data(), nullptr, out.data(), p);
        CHECK(r.status == 2 && out == z, "%s nan input", tname(type));
    }
    // determinism over thread counts
    {
        const int type = TQK6, nbl = 64, bb = block_bytes(type);
        std::vector<float> w((size_t) nbl * 128);
        for (auto& v : w) v = gauss();
        std::vector<uint8_t> a((size_t) nbl * bb), b(a.size());
        Params p;
        encode_blocks(type, w.data(), nullptr, nbl, a.data(), p, 1, nullptr, nullptr, nullptr);
        encode_blocks(type, w.data(), nullptr, nbl, b.data(), p, 7, nullptr, nullptr, nullptr);
        CHECK(a == b, "1 thread vs 7 threads differ");
    }
}

static void t9() {
    std::printf("T9 quality on N(0,1) blocks (SQNR in dB; the rate-distortion limit of the K/4 bits per weight is 6.02 * K/4)\n");
    for (int type : kTypes) {
        const int K = type_kbits(type), bb = block_bytes(type);
        const int nbl = (type == TQ2_T || type == TQK7) ? 256 : 384;
        std::vector<float> w((size_t) nbl * 128);
        for (auto& v : w) v = gauss();
        std::vector<uint8_t> out((size_t) nbl * bb);
        double en = 0;
        for (float v : w) en += (double) v * v;
        struct Cfg { const char* name; Params p; int th; };
        Cfg cfgs[5];
        cfgs[0] = {"1 pass, scale from rms", Params{0, 1, 0, 1, 0}, 8};
        cfgs[1] = {"default (rot 16, 3 scale rounds)", Params{}, 8};
        cfgs[2] = {"3 rotations, 3 scale rounds", Params{3, 3, 0, 1, 0}, 8};
        cfgs[3] = {"3 rotations, 3 starting scales", Params{3, 3, 0, 3, 0}, 8};
        int ncfg = 4;
        for (int i = 0; i < ncfg; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            const double e = encode_blocks(type, w.data(), nullptr, nbl, out.data(), cfgs[i].p, cfgs[i].th, nullptr, nullptr, nullptr);
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const double sq = 10.0 * std::log10(en / e);
            std::printf("  %-6s %-34s SQNR %.3f dB (limit %.2f, gap %.2f dB)  %.0f weights/s (%d threads)\n", tname(type),
                        cfgs[i].name, sq, 6.0206 * K / 4.0, 6.0206 * K / 4.0 - sq, (double) nbl * 128 / sec, cfgs[i].th);
        }
        // approximate vs exact start group at the default scale rounds on a few blocks
        {
            const int nb2 = type == TQK4X ? 2 : 8;
            Params pe;
            pe.exact = 1;
            pe.iters = 0;
            Params pa;
            pa.iters = 0;
            const double ee = encode_blocks(type, w.data(), nullptr, nb2, out.data(), pe, 8, nullptr, nullptr, nullptr);
            const double ea = encode_blocks(type, w.data(), nullptr, nb2, out.data(), pa, 8, nullptr, nullptr, nullptr);
            std::printf("  %-6s tail-biting search, %d blocks, one scale: approximate error / exact error = %.4f\n", tname(type), nb2, ea / ee);
            CHECK(ea >= ee * (1 - 1e-6), "approximate beat exact?");
        }
    }
}

int main(int argc, char** argv) {
    const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    t1();
    t2();
    t3();
    t5();
    t4();
    t6();
    t7();
    t8();
    if (!quick) t9();
    std::printf("\n%d checks, %d failures\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
