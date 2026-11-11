// src/kernels/cpu/tqp_cpu_test.cpp - the trellis CPU rows (tq_cpu.hpp) on synthetic blocks, with no ggml and no model:
//
//   * the pair-trellis types TQP5 / TQP6 / TQP7 (ggml 147-149, Gyro-L): the scalar kernel against a reference written
//     here from the format definition (state t = the 16 stream bits that start at bit (63 - t) * K, LSB first,
//     circular over 64 * K bits; the step's two weights are the first pair of the state's codebook entry), summed in
//     double; then every ISA this CPU has (AVX2, AVX-512 takes the AVX2 pair kernel) bit for bit equal to the scalar
//     kernel, for groups of 1..8 tokens, one row range or two, and for the fused gate/up rows;
//   * the older trellis types (144-146): AVX2 / AVX-512 bit for bit equal to the scalar kernel (the pair types were
//     added next to them: this shows they still agree), and their block sizes and K unchanged;
//   * the type predicates (tq_supported, tq_kbits, tq_block_bytes) for every id 0..299: the answers the code gave
//     before the pair types, plus 147-149.
//
// Random stream bytes are valid codes (any 15-bit state indexes the codebook).  CPU only.  Exit 0 = pass.
#include "strata/kernels/cpu/tq_cpu.hpp"

#include "tq_states.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::kernels::cpu;

namespace {

int g_fail = 0;

void fail(const char* what, int a = 0, int b = 0) {
    std::printf("  FAIL: %s %d %d\n", what, a, b);
    ++g_fail;
}

// ---- the type predicates as they were before the pair types (the old formulas, copied)
bool old_supported(int t) {
    return t == 144 || t == 145 || t == 146 || t == kTqTypeK5X || t == kTqTypeK4X || t == kTqTypeK5T || t == kTqTypeK4T;
}
int old_kbits(int t) {
    return t == 144 ? 8 : t == 145 ? 6 : t == 146 ? 7 : (t == kTqTypeK5X || t == kTqTypeK5T) ? 5 : (t == kTqTypeK4X || t == kTqTypeK4T) ? 4 : 0;
}

void check_predicates() {
    for (int t = 0; t < 300; ++t) {
        const bool pair = t >= 147 && t <= 149;
        if (tq_supported(t) != (old_supported(t) || pair)) fail("tq_supported", t);
        const int want_k = pair ? (t == 147 ? 5 : t == 148 ? 6 : 7) : old_kbits(t);
        if (tq_kbits(t) != want_k) fail("tq_kbits", t, tq_kbits(t));
        const size_t want_b = pair ? (size_t) (2 + 8 * want_k) : (size_t) (2 + 4 * want_k);
        if (tq_block_bytes(t) != want_b) fail("tq_block_bytes", t, (int) tq_block_bytes(t));
    }
    if (tq_block_bytes(144) != 34 || tq_block_bytes(145) != 26 || tq_block_bytes(146) != 30) fail("sizes of 144-146");
    if (tq_block_bytes(147) != 42 || tq_block_bytes(148) != 50 || tq_block_bytes(149) != 58) fail("sizes of 147-149");
    std::printf("predicates (ids 0..299): %s\n", g_fail ? "FAIL" : "ok");
}

// ---- synthetic data
uint16_t random_half(std::mt19937& rng) {
    std::uniform_int_distribution<int> ex(2, 8), man(0, 1023), sgn(0, 3);
    return (uint16_t) ((sgn(rng) == 0 ? 0x8000 : 0) | (ex(rng) << 10) | man(rng));   // 2^-13 .. 2^-6
}

std::vector<uint8_t> random_rows(int type, int rows, int n, std::mt19937& rng, size_t& row_bytes) {
    const size_t bs = tq_block_bytes(type);
    row_bytes = bs * (size_t) (n / 128);
    std::vector<uint8_t> w(row_bytes * (size_t) rows);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& b : w) b = (uint8_t) byte(rng);
    for (size_t o = 0; o < w.size(); o += bs) {
        const uint16_t h = random_half(rng);
        std::memcpy(&w[o], &h, 2);
    }
    return w;
}

// n values as Q8_0 blocks (fp16 d, 32 x int8): 34 bytes per 32 values
std::vector<uint8_t> random_q8_0(int n, std::mt19937& rng, std::vector<float>& d, std::vector<int>& q) {
    std::vector<uint8_t> a((size_t) (n / 32) * 34);
    std::uniform_int_distribution<int> val(-127, 127), ex(8, 13), man(0, 1023);
    d.assign((size_t) n / 32, 0.f);
    q.assign((size_t) n, 0);
    for (int b = 0; b < n / 32; ++b) {
        const uint16_t h = (uint16_t) ((ex(rng) << 10) | man(rng));
        std::memcpy(&a[(size_t) b * 34], &h, 2);
        d[(size_t) b] = strata::kernels::cpu::tq_detail::tq_half_to_float(h);
        for (int j = 0; j < 32; ++j) {
            const int v = val(rng);
            q[(size_t) b * 32 + j] = v;
            a[(size_t) b * 34 + 2 + j] = (uint8_t) (int8_t) v;
        }
    }
    return a;
}

// ---- the reference for the pair types, from the format definition (bit by bit, not the window code of the kernels)
uint32_t ref_state(const uint8_t* qs, int K, int t) {
    const int nbits = 64 * K;
    uint32_t s = 0;
    for (int b = 0; b < 16; ++b) {
        const int pos = ((63 - t) * K + b) % nbits;
        s |= (uint32_t) ((qs[pos >> 3] >> (pos & 7)) & 1) << b;
    }
    return s & 0x7FFFu;
}

// w . a in double; `bound` = the sum of the absolute terms (the scale of the float rounding allowed)
double ref_dot(int type, const uint8_t* row, int n, const std::vector<float>& d, const std::vector<int>& q, double& bound) {
    const int K = tq_kbits(type);
    const int16_t* tab = tq_state_table();
    const size_t bs = tq_block_bytes(type);
    double sum = 0, abs_sum = 0;
    for (int i = 0; i < n / 128; ++i) {
        const uint8_t* blk = row + (size_t) i * bs;
        uint16_t hd;
        std::memcpy(&hd, blk, 2);
        const double dw = strata::kernels::cpu::tq_detail::tq_half_to_float(hd);
        uint32_t st[64];
        for (int t = 0; t < 64; ++t) st[t] = ref_state(blk + 2, K, t);
        for (int k = 0; k < 4; ++k) {
            long long s = 0, sa = 0;
            for (int j = 0; j < 32; ++j) {
                const int v = 32 * k + j;                    // weight v of the block: step v / 2, element v & 1 of the first pair
                const int w16 = tab[4 * st[v >> 1] + (v & 1)];
                s += (long long) w16 * q[(size_t) 128 * i + v];
                sa += (long long) std::abs(w16) * std::abs(q[(size_t) 128 * i + v]);
            }
            const double sc = dw * d[(size_t) 4 * i + k];
            sum += sc * (double) s;
            abs_sum += std::fabs(sc) * (double) sa;
        }
    }
    bound = abs_sum / 8192.0;
    return sum / 8192.0;
}

bool same_bits(const float* a, const float* b, size_t n) { return std::memcmp(a, b, n * sizeof(float)) == 0; }

constexpr int NT = kTqMaxTokens;

struct Case {
    int type, rows, n;
    std::vector<uint8_t> w;
    size_t rb = 0;
    std::vector<std::vector<uint8_t>> act;
    std::vector<std::vector<float>> d;
    std::vector<std::vector<int>> q;
    const void* ap[NT];
    Case(int type_, int rows_, int n_, uint32_t seed) : type(type_), rows(rows_), n(n_) {
        std::mt19937 rng(seed);
        w = random_rows(type, rows, n, rng, rb);
        act.resize(NT); d.resize(NT); q.resize(NT);
        for (int t = 0; t < NT; ++t) {
            act[t] = random_q8_0(n, rng, d[t], q[t]);
            ap[t] = act[t].data();
        }
    }
    void run(TqIsa isa, int nt, const void* const* a, std::vector<float>& o) const {
        o.assign((size_t) nt * rows, 0.f);
        float* op[NT];
        for (int t = 0; t < nt; ++t) op[t] = o.data() + (size_t) t * rows;
        tq_rows(isa, type, w.data(), rb, n, a, nt, op, 0, rows / 3);       // two row ranges, as the pool splits them
        tq_rows(isa, type, w.data(), rb, n, a, nt, op, rows / 3, rows);
    }
};

void check_type(int type, bool reference) {
    Case c(type, 37, 1280, 1000u + (uint32_t) type);
    std::vector<float> ref, got;
    c.run(TqIsa::scalar, NT, c.ap, ref);
    int bad = 0;
    if (reference) {   // the pair types: the scalar kernel against the definition
        double worst = 0;
        for (int t = 0; t < NT; ++t)
            for (int r = 0; r < c.rows; ++r) {
                double bound = 0;
                const double want = ref_dot(type, c.w.data() + (size_t) r * c.rb, c.n, c.d[t], c.q[t], bound);
                const double err = std::fabs((double) ref[(size_t) t * c.rows + r] - want);
                worst = std::fmax(worst, bound > 0 ? err / bound : err);
                if (err > 2e-5 * bound + 1e-30) ++bad;
            }
        std::printf("type %d scalar vs the definition (double): %d of %d outside 2e-5 of the term sum, worst %.2e\n", type, bad,
                    NT * c.rows, worst);
        if (bad) fail("scalar kernel differs from the definition", type);
    }
    for (TqIsa isa : {TqIsa::avx2, TqIsa::avx512}) {
        if (!tq_isa_ok(isa)) { std::printf("type %d %-6s not on this CPU: skipped\n", type, tq_isa_name(isa)); continue; }
        bool ok = true;
        for (int nt = 1; nt <= NT; ++nt)
            for (int t0 = 0; t0 + nt <= NT; t0 += nt) {
                c.run(isa, nt, c.ap + t0, got);
                ok = ok && same_bits(got.data(), ref.data() + (size_t) t0 * c.rows, (size_t) nt * c.rows);
            }
        std::printf("type %d %-6s groups of 1..%d tokens vs the scalar kernel: %s\n", type, tq_isa_name(isa), NT,
                    ok ? "bitwise equal" : "DIFFERENT");
        if (!ok) fail("ISA differs from scalar", type);
    }
    // the fused gate/up rows: gate rows at the blob, up rows after them
    {
        std::vector<uint8_t> blob = c.w;
        std::mt19937 rng(77u + (uint32_t) type);
        size_t rb2 = 0;
        const auto up = random_rows(type, c.rows, c.n, rng, rb2);
        blob.insert(blob.end(), up.begin(), up.end());
        const size_t up_off = c.w.size();
        std::vector<float> ff0((size_t) NT * c.rows), ff1((size_t) NT * c.rows);
        float* p0[NT]; float* p1[NT];
        for (int t = 0; t < NT; ++t) { p0[t] = ff0.data() + (size_t) t * c.rows; p1[t] = ff1.data() + (size_t) t * c.rows; }
        tq_gu_rows(TqIsa::scalar, type, blob.data(), c.rb, up_off, c.n, c.ap, NT, p0, 0, c.rows);
        bool ok = true;
        for (TqIsa isa : {TqIsa::avx2, TqIsa::avx512}) {
            if (!tq_isa_ok(isa)) continue;
            tq_gu_rows(isa, type, blob.data(), c.rb, up_off, c.n, c.ap, NT, p1, 0, c.rows);
            ok = ok && same_bits(ff0.data(), ff1.data(), ff0.size());
        }
        std::printf("type %d gate/up rows, every ISA vs scalar: %s\n", type, ok ? "bitwise equal" : "DIFFERENT");
        if (!ok) fail("gate/up rows differ", type);
    }
}

}  // namespace

int main() {
    check_predicates();
    for (int type : {147, 148, 149}) check_type(type, true);
    for (int type : {144, 145, 146}) check_type(type, false);
    std::printf("tqp_cpu_test: %d failures\n", g_fail);
    return g_fail ? 1 : 0;
}
