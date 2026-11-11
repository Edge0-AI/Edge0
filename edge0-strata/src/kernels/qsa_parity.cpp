// src/kernels/qsa_parity.cpp - P2.S2's test for the QSA cache, indexer, selection and attention.
//
// `ref/qsa.py` carries eleven PROPERTY checks and `ref/model.py::_qsa` is the layer as the model runs it.  The
// readings a prose transcription gets WRONG, each of which produces a well-formed result of the right shape,
// are computed here the wrong way round AND required to differ MATERIALLY before the kernel is judged against
// the right one - a test that only does the second half passes against either reading:
//
//   1. `pooled[b]` is rotated at the block's FIRST cell.  The reference's PROPERTY 4 pins it against the
//      block's LAST cell; both give a unit-norm vector and the same downstream shape.
//   2. Pool THEN norm (`rms_norm(mean(x))`) and not norm THEN pool (`mean(rms_norm(x))`).  `ref/model.py` L358
//      is explicit, and the two differ whenever the four cells have different magnitudes.
//   3. The spare slot's key is `rms_norm(raw[0])` - cell ZERO - and not the last raw key, and the tail cells
//      map to THAT slot and not to the last complete block.  A three-row raw tail cannot produce it unless
//      cell 0's key is kept, which is what makes this a real design constraint rather than a detail.
//   4. The indexer's Relu is PER HEAD, summed afterwards.  `relu(sum_h dot_h)` is the rival reading and
//      differs whenever any head's dot is negative.
//   5. The q -> kv head map is `q / 12` (integer division), NOT `q % 2`.  `ref/qsa.py` PROPERTY 7 tests the
//      same thing; with 24 and 2 the two give [0 x12, 1 x12] and [0,1,0,1,...].
//   6. The attention reads ONLY the selected cells.  The source writes -inf into the mask and zeros at the
//      top-k indices, so an unselected cell has weight exactly zero even when it would score highest -
//      PROPERTY 8's fixture, reproduced here with its own numbers.
//   7. `attn *= sigmoid(gate)` with the gate taken from the SECOND half of each head's 2*head_dim block, and
//      the rounding to fp16 happening AFTER the multiply (one rounding, not two).
//   8. The KV cache is PAGED and the page table is not assumed to be the identity: the pool is written and
//      read back at page_size 1, 4 and 512 with a REVERSED table, and the kernel's own reading of the layout
//      is compared against two rivals (linear, i.e. the table ignored; and cell-major inside the page).
//
// THE METRIC.  Both sums here - the indexer's sum of four relu'd dots and the attention's weighted sum of
// value rows - cancel, so a plain |want-got|/|want| reports the condition number instead of the arithmetic.
// The project has made that mistake four times (rounds 169, 189, 194, 196); where a sum is compared below, the
// denominator is the magnitude of the TERMS.
//
// WHAT SETS EACH TOLERANCE, because a tolerance nobody can explain is one nobody can tighten:
//   * the fp16 converter, the pool contents, the gathered scratch and the spare indexer key: EXACT.  The only
//     arithmetic is a conversion, and section 0 checks that conversion against numpy's float16 over 395
//     vectors covering ties in both directions, subnormals and overflow - an oracle sharing no code with it -
//     plus all 65,536 fp16 patterns in the reverse direction.
//   * the pooled block keys: 1e-6.  The pooling and the norm are double on both sides and are bit-exact
//     BEFORE the rotation; the residual is `rope_neox_apply` (f32, reused rather than re-implemented) against
//     the reference's float64 rotation, the class round 189 measured at ~1e-7.
//   * the indexer scores: 1e-6 against Sum|dot_h| + |bias|, set by the fp32 storage of the pooled keys -
//     2^-24 per element over a 128-term dot.
//   * the selected ids: EXACT.  `docs/capture-format.md` L127 gives `indexer_ids` no tolerance.
//   * the attention against a reference reading the SAME fp16 cache: 1e-5, set by f32 accumulation over
//     256-term dots plus `expf`'s 2 ulp.  This is the KERNEL's number.
//   * the attention against a reference reading the f32 keys: REPORTED and then checked against the phase's
//     1e-3 for FP16 paths.  That difference is the FP16 CACHE's cost, not the kernel's, and it is kept
//     separate so a kernel bug cannot hide inside it.
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"

#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/rope.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {

using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_f16;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

/// A device buffer with the boilerplate folded in.  A test that allocates twenty buffers by hand spends its
/// attention on memcpy arguments instead of on the kernel.
template <typename T>
struct Dev {
    T* p = nullptr;
    Dev() = default;
    explicit Dev(size_t n) { alloc(n); }
    ~Dev() { if (p) cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    void alloc(size_t n) {
        if (p) { cudaFree(p); p = nullptr; }
        if (n) check(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc");
    }
    void put(const std::vector<T>& v) {
        if (!p) alloc(v.size());
        check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "H2D");
    }
    std::vector<T> get(size_t n) const {
        std::vector<T> v(n);
        check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "D2H");
        return v;
    }
};

std::vector<double> as_d(const std::vector<float>& v) { return std::vector<double>(v.begin(), v.end()); }
std::vector<float> as_f(const std::vector<double>& v) {
    std::vector<float> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = (float) v[i];
    return o;
}

/// Relative error against the magnitude of the TERMS.  See the header comment.
double rel_terms(double want, double got, double terms) {
    return std::fabs(want - got) / (terms > 1e-30 ? terms : 1e-30);
}

double rel_l1(const std::vector<double>& a, const std::vector<double>& b) {
    double d = 0, m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += std::fabs(a[i] - b[i]);
        m += std::fabs(a[i]);
    }
    return d / (m > 1e-30 ? m : 1e-30);
}

int g_bad = 0;
std::string pct(double rel) {
    char b[32];
    std::snprintf(b, sizeof b, "%.2f%% apart", rel * 100.0);
    return b;
}
void verdict(const std::string& name, bool ok, const std::string& detail = "") {
    std::printf("  %-48s %-6s %s\n", name.c_str(), ok ? "yes" : "*** NO ***", detail.c_str());
    if (!ok) ++g_bad;
}
void report(const std::string& name, double v, const char* unit) {
    std::printf("  %-48s %s %.3e\n", name.c_str(), unit, v);
}
void require(const std::string& name, bool ok, const std::string& detail = "") {
    std::printf("  %-48s %-6s %s\n", name.c_str(), ok ? "ok" : "*** BAD ***", detail.c_str());
    if (!ok) ++g_bad;
}

// ================= the reference: ref/qsa.py + ref/model.py, in double =================

/// Which of the rival readings to compute.  The defaults are the SOURCE's reading.
struct Alt {
    bool rotate_at_block_start = true;  ///< false: rotate at the block's LAST cell         (WRONG)
    bool pool_then_norm = true;         ///< false: mean of the per-cell norms              (WRONG)
    bool dead_from_first = true;        ///< false: the spare key from the LAST raw key     (WRONG)
    bool rope_raw = false;              ///< true:  rotate the raw keys before pooling      (WRONG)
    bool relu_per_head = true;          ///< false: relu(sum of the head dots)              (WRONG)
    bool use_relu = true;               ///< false: no relu at all                          (WRONG)
    bool use_bias = true;               ///< false: drop the per-block bias                 (WRONG)
    bool kv_divide = true;              ///< false: q % n_head_kv instead of q / 12          (WRONG)
    bool apply_scale = true;            ///< false: no 1/sqrt(head_dim)                      (WRONG)
    bool gate_second_half = true;       ///< false: the gate from the FIRST half             (WRONG)
    bool gate_sigmoid = true;           ///< false: silu instead of sigmoid                  (WRONG)
};

/// `x*y + z` with the product ROUNDED first, i.e. no FMA contraction.
///
/// MSVC contracts `ss += v*v` into an FMA even under /fp:precise, so the reference was MORE accurate than
/// the kernel's explicit `__fadd_rn(__fmul_rn(v, v))` and the spare key came out one bit different in 7 of
/// 128 components.  The kernel's arithmetic is what is under test, so the reference REPRODUCES it instead of
/// beating it; `volatile` is what stops the compiler re-fusing the two operations.
double mul_add_plain(double x, double y, double z) {
    const volatile double p = x * y;
    return z + p;
}

/// `ref/qsa.py::rms_norm` in double, in place.  `w` may be empty (w = None).
void ref_rms_norm(std::vector<double>& x, const std::vector<float>& w, float eps) {
    double ss = 0;
    for (double v : x) ss = mul_add_plain(v, v, ss);
    const double inv = 1.0 / std::sqrt(ss / (double) x.size() + (double) eps);
    for (size_t i = 0; i < x.size(); ++i) x[i] = x[i] * inv * (w.empty() ? 1.0 : (double) w[i]);
}

/// `ref/qsa.py::rope_neox` in double: NEOX pairs (i, i + n_rot/2), partial over the first n_rot dims.
///
/// `terms`, when given, receives the magnitude of each component's TERMS - `|a c| + |b s|` where the
/// rotation subtracts - so a comparison can divide by that instead of by the result, which cancels.
void ref_rope_neox(std::vector<double>& x, double pos, int n_rot, double theta,
                   std::vector<double>* terms = nullptr) {
    const int half = n_rot / 2;
    std::vector<double> a(x.begin(), x.begin() + half), b(x.begin() + half, x.begin() + n_rot);
    if (terms) {
        terms->assign(x.size(), 0.0);
        for (size_t i = (size_t) n_rot; i < x.size(); ++i) (*terms)[i] = std::fabs(x[i]);
    }
    for (int i = 0; i < half; ++i) {
        const double inv = std::pow(theta, -2.0 * (double) i / (double) n_rot);
        const double ang = pos * inv, c = std::cos(ang), s = std::sin(ang);
        x[i] = a[i] * c - b[i] * s;
        x[half + i] = a[i] * s + b[i] * c;
        if (terms) {
            (*terms)[i] = std::fabs(a[i] * c) + std::fabs(b[i] * s);
            (*terms)[half + i] = std::fabs(a[i] * s) + std::fabs(b[i] * c);
        }
    }
}

const int64_t REF_R = 4;    ///< IDX_BLOCK, checked against the shapes at the top of main

/// `QsaCache`, holding f64 rows - the reference never sees an fp16 key.
struct RefCache {
    std::vector<std::vector<double>> k, v, raw;
    std::vector<double> pos;
    int64_t size() const { return (int64_t) pos.size(); }
    void append(std::vector<double> kk, std::vector<double> vv, std::vector<double> rr, double p) {
        k.push_back(std::move(kk));
        v.push_back(std::move(vv));
        raw.push_back(std::move(rr));
        pos.push_back(p);
    }
};

/// `QsaCache.pooled_raw(r)` - BEFORE the norm, which is the only place the 1/r divisor is observable.
std::vector<double> ref_pooled_raw(const RefCache& c) {
    const int64_t n_bid = c.size() / REF_R, dim = (int64_t) c.raw[0].size();
    std::vector<double> out((size_t) n_bid * dim, 0.0);
    for (int64_t b = 0; b < n_bid; ++b)
        for (int64_t i = 0; i < REF_R; ++i)
            for (int64_t d = 0; d < dim; ++d) out[(size_t) b * dim + d] += c.raw[(size_t) (b * REF_R + i)][(size_t) d];
    for (auto& x : out) x /= (double) REF_R;
    return out;
}

/// `pooled_keys` + `dead_pooled_key`, in the readback order `indexer_scores` uses: (n_bid + 1, idx_dim).
std::vector<double> ref_pooled(const RefCache& c, const std::vector<float>& w_kn, float eps, int n_rot,
                               double theta, const Alt& o,
                               std::vector<double>* terms_out = nullptr) {
    const int64_t n_bid = c.size() / REF_R, dim = (int64_t) c.raw[0].size();
    std::vector<double> pooled((size_t) (n_bid + 1) * dim, 0.0);
    std::vector<std::vector<double>> rows(c.raw);
    if (o.rope_raw)
        for (size_t j = 0; j < rows.size(); ++j) ref_rope_neox(rows[j], c.pos[j], n_rot, theta);

    for (int64_t b = 0; b < n_bid; ++b) {
        std::vector<double> m((size_t) dim, 0.0);
        if (o.pool_then_norm) {
            for (int64_t i = 0; i < REF_R; ++i)
                for (int64_t d = 0; d < dim; ++d)
                    m[(size_t) d] += rows[(size_t) (b * REF_R + i)][(size_t) d] / (double) REF_R;
            ref_rms_norm(m, w_kn, eps);
        } else {
            for (int64_t i = 0; i < REF_R; ++i) {
                std::vector<double> one = rows[(size_t) (b * REF_R + i)];
                ref_rms_norm(one, w_kn, eps);
                for (int64_t d = 0; d < dim; ++d) m[(size_t) d] += one[(size_t) d] / (double) REF_R;
            }
        }
        std::vector<double> terms;
        ref_rope_neox(m, o.rotate_at_block_start ? c.pos[(size_t) (b * REF_R)]
                                                 : c.pos[(size_t) (b * REF_R + REF_R - 1)],
                      n_rot, theta, terms_out ? &terms : nullptr);
        for (int64_t d = 0; d < dim; ++d) pooled[(size_t) b * dim + d] = m[(size_t) d];
        if (terms_out) {
            if (b == 0) terms_out->assign((size_t) (n_bid + 1) * dim, 0.0);
            for (int64_t d = 0; d < dim; ++d) (*terms_out)[(size_t) b * dim + d] = terms[(size_t) d];
        }
    }
    std::vector<double> dead = rows[o.dead_from_first ? 0 : rows.size() - 1];
    ref_rms_norm(dead, w_kn, eps);
    ref_rope_neox(dead, 0.0, n_rot, theta);      // the identity, but computed the reference's way
    for (int64_t d = 0; d < dim; ++d) pooled[(size_t) n_bid * dim + d] = dead[(size_t) d];
    return pooled;
}

/// `ref/qsa.py::indexer_scores` - per CELL, after the `cell_block` mapping.
std::vector<double> ref_indexer_scores(const std::vector<double>& pooled, int64_t n_bid,
                                       const std::vector<double>& q_idx, int64_t idx_n_head, int64_t idx_dim,
                                       const std::vector<float>& bias, int64_t n_kv, const Alt& o) {
    std::vector<double> per_block((size_t) n_bid + 1, 0.0);
    for (int64_t b = 0; b <= n_bid; ++b) {
        double s = 0;
        for (int64_t h = 0; h < idx_n_head; ++h) {
            double dot = 0;
            for (int64_t d = 0; d < idx_dim; ++d)
                dot += pooled[(size_t) b * idx_dim + d] * q_idx[(size_t) h * idx_dim + d];
            if (!o.use_relu) s += dot;
            else if (o.relu_per_head) s += dot > 0 ? dot : 0.0;
            else s += dot;
        }
        if (o.use_relu && !o.relu_per_head) s = s > 0 ? s : 0.0;
        if (o.use_bias && !bias.empty()) s += (double) bias[(size_t) b];
        if (b == n_bid && n_kv % REF_R != 0) s = (float) s + 1e9f;
        per_block[(size_t) b] = s;
    }
    std::vector<double> cell((size_t) n_kv, 0.0);
    for (int64_t j = 0; j < n_kv; ++j) cell[(size_t) j] = per_block[(size_t) (j / REF_R < n_bid ? j / REF_R : n_bid)];
    return cell;
}

/// `ref/qsa.py::select_cells`, spelled with a stable sort rather than the kernel's threshold search, so the
/// two are independent implementations of one rule: score DESCENDING, ties by ASCENDING index.
std::vector<int32_t> ref_select(const std::vector<double>& scores, int64_t n_kv, int64_t top_k, bool dense) {
    std::vector<int32_t> idx((size_t) n_kv);
    for (int64_t j = 0; j < n_kv; ++j) idx[(size_t) j] = (int32_t) j;
    if (dense) return idx;
    const int64_t width = std::min<int64_t>(n_kv, top_k + REF_R - 1);
    std::stable_sort(idx.begin(), idx.end(), [&](int32_t a, int32_t b) {
        const double sa = scores[(size_t) a], sb = scores[(size_t) b];
        if (sa != sb) return sa > sb;
        return a < b;
    });
    idx.resize((size_t) width);
    std::sort(idx.begin(), idx.end());
    return idx;
}

/// `ref/qsa.py::qsa_attend` over an EXPLICIT selected set, reading K/V from whichever table the caller passes
/// so that one piece of code runs on the f64 keys and on the fp16 cache's exact f32 values.
std::vector<double> ref_attend(const std::vector<double>& q, int64_t n_head, int64_t n_head_kv, int64_t head_dim,
                               const std::vector<int32_t>& ids, const std::vector<std::vector<double>>& k,
                               const std::vector<std::vector<double>>& v, const Alt& o,
                               std::vector<double>* weights_out) {
    const double scale = o.apply_scale ? 1.0 / std::sqrt((double) head_dim) : 1.0;
    std::vector<double> out((size_t) n_head * head_dim, 0.0);
    if (weights_out) weights_out->assign((size_t) n_head * ids.size(), 0.0);
    for (int64_t h = 0; h < n_head; ++h) {
        const int64_t kv = o.kv_divide ? h / (n_head / n_head_kv) : h % n_head_kv;
        std::vector<double> sc(ids.size());
        for (size_t j = 0; j < ids.size(); ++j) {
            double dot = 0;
            for (int64_t d = 0; d < head_dim; ++d)
                dot += k[(size_t) ids[j]][(size_t) (kv * head_dim + d)] * q[(size_t) (h * head_dim + d)];
            sc[j] = dot * scale;
        }
        const double mx = *std::max_element(sc.begin(), sc.end());
        double sum = 0;
        for (auto& s : sc) { s = std::exp(s - mx); sum += s; }
        for (auto& s : sc) s /= sum;
        if (weights_out)
            for (size_t j = 0; j < ids.size(); ++j) (*weights_out)[(size_t) h * ids.size() + j] = sc[j];
        for (int64_t d = 0; d < head_dim; ++d) {
            double acc = 0;
            for (size_t j = 0; j < ids.size(); ++j) acc += sc[j] * v[(size_t) ids[j]][(size_t) (kv * head_dim + d)];
            out[(size_t) (h * head_dim + d)] = acc;
        }
    }
    return out;
}

/// `ref/model.py::_qsa`'s last two lines.
std::vector<double> ref_gate(const std::vector<double>& attn, const std::vector<double>& q_full, int64_t n_head,
                             int64_t head_dim, const Alt& o) {
    std::vector<double> out((size_t) n_head * head_dim, 0.0);
    for (int64_t h = 0; h < n_head; ++h)
        for (int64_t d = 0; d < head_dim; ++d) {
            const int64_t gi = o.gate_second_half ? h * 2 * head_dim + head_dim + d : h * 2 * head_dim + d;
            const double g = q_full[(size_t) gi];
            const double s = o.gate_sigmoid ? 1.0 / (1.0 + std::exp(-g)) : g / (1.0 + std::exp(-g));
            out[(size_t) (h * head_dim + d)] = attn[(size_t) (h * head_dim + d)] * s;
        }
    return out;
}

/// {f32 bits, the fp16 bits numpy produces for it} interleaved.  Generated by
/// `scripts/_gen_f16_oracle.py` from `np.float16(np.float32(x))`: an oracle that shares no code with
/// f16_bits.hpp, which is the whole point of the table.
const uint32_t kOracle[] = {
#include "qsa_oracle_vectors.inc"
};
constexpr int kOracleN = (int) (sizeof(kOracle) / sizeof(kOracle[0])) / 2;

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // a crash must not swallow what was printed already
    bool selftest = false, dense = false, bench = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") selftest = true;
        else if (a == "--qsa-dense") dense = true;
        else if (a == "--bench") bench = true;
        else {
            std::fprintf(stderr, "usage: qsa_parity [--selftest] [--qsa-dense] [--bench]\n"
                                 "  --qsa-dense  attend to EVERY cell instead of the indexer's selection\n"
                                 "               (P1.S4's isolation switch; the sparse selection is still run\n"
                                 "                and checked, against the reference's dense/sparse property)\n"
                                 "  --bench      time the kernels at 32K of context\n");
            return 2;
        }
    }

    const strata::kernels::QsaShapes S = strata::kernels::qsa_real_shapes();
    const int64_t NH = S.n_head, NKV = S.n_head_kv, HD = S.head_dim, IDXN = S.idx_n_head, IDXD = S.idx_dim,
                  R = S.idx_block, TOPK = S.idx_top_k;
    const int64_t MAXC = strata::kernels::kTopkMaxCells;
    const float EPS = strata::kernels::qsa_rms_eps();
    const double THETA = strata::kernels::qsa_freq_base();
    std::mt19937 rng(20260916u);
    std::normal_distribution<double> gauss(0.0, 1.0);
    auto rnd = [&](double s) { return (float) (gauss(rng) * s); };

    std::printf("qsa_parity: n_head %lld  n_head_kv %lld  head_dim %lld  indexer %lldx%lld r=%lld "
                "top_k %lld  page (from the shapes) %lld   [%s]\n\n",
                (long long) NH, (long long) NKV, (long long) HD, (long long) IDXN, (long long) IDXD,
                (long long) R, (long long) TOPK, (long long) S.page_size, dense ? "DENSE" : "sparse");
    require("the fixture's r is the artifact's IDX_BLOCK", R == REF_R, "r = " + std::to_string(R));

    // ================= 0. the fp16 converter, against numpy =================
    {
        int mism = 0, subnormal = 0;
        for (int i = 0; i < kOracleN; ++i) {
            float f;
            const uint32_t fb = kOracle[2 * i];
            std::memcpy(&f, &fb, 4);
            const uint16_t want = (uint16_t) kOracle[2 * i + 1];
            const uint16_t got = f16_from_f32(f);
            if (got != want) {
                if (mism < 5) std::printf("    f32 0x%08X -> got 0x%04X, numpy says 0x%04X\n", fb, got, want);
                ++mism;
            }
            if ((want & 0x7C00u) == 0u && (want & 0x3FFu) != 0u) ++subnormal;
        }
        require("f32->f16 matches numpy's float16 on every vector", mism == 0,
                std::to_string(kOracleN) + " vectors, " + std::to_string(subnormal) + " subnormal");
        int bad_rev = 0;
        for (int h = 0; h < 65536; ++h) {
            const int ex = (h >> 10) & 0x1F, man = h & 0x3FF;
            if (ex == 31) continue;                    // inf/NaN come from the vector table above
            const double mag = ex == 0 ? (double) man * std::ldexp(1.0, -24)
                                       : (double) (1024 + man) * std::ldexp(1.0, ex - 25);
            const double want = (h & 0x8000) ? -mag : mag;
            if ((double) f32_from_f16((uint16_t) h) != want) {
                if (bad_rev < 5) std::printf("    f16 0x%04X -> %.9g, want %.9g\n", h, (double) f32_from_f16((uint16_t) h), want);
                ++bad_rev;
            }
        }
        require("f16->f32 is exact on all 65,536 bit patterns", bad_rev == 0,
                std::to_string(bad_rev) + " wrong");
    }

    // ================= 1. kv_append: the paged pool, three page granules =================
    const int64_t TCELLS = 1100;                     // 275 indexer blocks, and 3 pages at page_size 512
    std::vector<float> kcur_all((size_t) TCELLS * NKV * HD), vcur_all((size_t) TCELLS * NKV * HD);
    for (auto& x : kcur_all) x = rnd(1.0);
    for (auto& x : vcur_all) x = rnd(1.0);
    // EVERY pointer this kernel set takes is a DEVICE pointer - the append sources, the indexer weights
    // and the rope tables alike.  The first draft passed `&kcur_all[...]`, `w_kn.data()` and
    // `cos_tab.data()` straight through, and every kernel that dereferenced one took an illegal access.
    Dev<float> dkcur_all, dvcur_all;
    dkcur_all.put(kcur_all);
    dvcur_all.put(vcur_all);

    struct PageCase {
        int64_t page_size = 0, n_pages = 0;
        std::vector<int32_t> table;
    };
    std::vector<PageCase> pages;
    for (int64_t ps : {(int64_t) 1, (int64_t) 4, (int64_t) 512}) {
        PageCase c;
        c.page_size = ps;
        c.n_pages = (TCELLS + ps - 1) / ps;
        c.table.resize((size_t) c.n_pages);
        for (int64_t i = 0; i < c.n_pages; ++i) c.table[(size_t) i] = (int32_t) (c.n_pages - 1 - i);   // REVERSED
        pages.push_back(std::move(c));
    }
    const std::vector<int32_t> ids = {0, 1, 2, 3, 5, 8, 13, 21, 34, 39, 1099, 700};

    for (PageCase& c : pages) {
        // THE PAGE GRANULE IS A PROPERTY OF THE SHAPES, and the kernel takes it from there rather than from an
        // argument.  The first version of this fixture left it at 512 while the table described 1-cell pages,
        // so the kernel indexed a 1100-page table as if it had 3 pages and ran off the pool - which is what an
        // illegal memory access reported.  A test that feeds one geometry and expects another is measuring
        // itself; the fix is to build the shapes per case.
        strata::kernels::QsaShapes Sc = S;
        Sc.page_size = c.page_size;
        const size_t pool_n = (size_t) c.n_pages * NKV * c.page_size * HD;
        Dev<uint16_t> dk(pool_n), dv(pool_n);
        Dev<int32_t> dtab;
        dtab.put(c.table);
        for (int64_t t = 0; t < TCELLS; ++t)
            strata::kernels::kv_append(dk.p, dv.p, dtab.p, t, dkcur_all.p + (size_t) t * NKV * HD,
                                       dvcur_all.p + (size_t) t * NKV * HD, Sc, nullptr);
        const std::vector<uint16_t> hk = dk.get(pool_n), hv = dv.get(pool_n);

        int mism = 0, mism_linear = 0, mism_head_major = 0;
        const int64_t rows_avail = (int64_t) (pool_n / (size_t) HD);
        // A rival layout's address is HYPOTHETICAL: where it would put the value may be outside the pool,
        // and reading there is a host access violation rather than a mismatch.  Off the end counts as
        // "the value is not where the rival would put it", which is the claim being tested.
        auto elsewhere = [&](int64_t r) { return r < 0 || r >= rows_avail; };
        for (int64_t cell = 0; cell < TCELLS; ++cell)
            for (int64_t h = 0; h < NKV; ++h)
                for (int64_t d = 0; d < HD; ++d) {
                    const size_t src = (size_t) (cell * NKV + h) * HD + d;
                    const uint16_t wk = f16_from_f32(kcur_all[src]), wv = f16_from_f32(vcur_all[src]);
                    // the DECLARED layout: [page][kv_head][page_size][head_dim]
                    const int64_t row = ((int64_t) c.table[(size_t) (cell / c.page_size)] * NKV + h) * c.page_size +
                                        (cell % c.page_size);
                    if (hk[(size_t) row * HD + d] != wk || hv[(size_t) row * HD + d] != wv) ++mism;
                    // rival (a): the page table ignored, i.e. written at the logical address
                    const int64_t rl = (cell * NKV + h) * c.page_size + (cell % c.page_size);
                    if (elsewhere(rl) || hk[(size_t) rl * HD + d] != wk) ++mism_linear;
                    // rival (b): cell-major inside the page instead of head-major
                    const int64_t rh = ((int64_t) c.table[(size_t) (cell / c.page_size)] * c.page_size +
                                        (cell % c.page_size)) * NKV + h;
                    if (elsewhere(rh) || hk[(size_t) rh * HD + d] != wk) ++mism_head_major;
                }
        const double n_tot = (double) (TCELLS * NKV * HD);
        require("kv_append at page_size " + std::to_string(c.page_size) + ": the pool is BIT-EXACT",
                mism == 0, std::to_string(mism) + " of " + std::to_string((long long) n_tot) + " bits wrong");
        verdict("  the page table is observable at page_size " + std::to_string(c.page_size),
                (double) mism_linear / n_tot > 0.5,
                pct((double) mism_linear / n_tot) + " (linear vs declared)");
        // The head-major rival is IDENTICAL to the declared layout when page_size == 1 (both reduce to
        // `cell * n_head_kv + h`), so the separation is only asserted where the layouts actually differ.
        if (c.page_size >= 4) {
            verdict("  head-major-in-page is observable at page_size " + std::to_string(c.page_size),
                    (double) mism_head_major / n_tot > 0.5,
                    pct((double) mism_head_major / n_tot) + " (cell-major vs head-major)");
        }

        Dev<int32_t> dids;
        dids.put(ids);
        const size_t scr_n = ids.size() * NKV * HD;
        Dev<uint16_t> dks(scr_n), dvs(scr_n);
        strata::kernels::kv_gather(dk.p, dv.p, dtab.p, dids.p, (int64_t) ids.size(), Sc, dks.p, dvs.p, nullptr);
        const std::vector<uint16_t> gk = dks.get(scr_n), gv = dvs.get(scr_n);
        int gm = 0;
        for (size_t j = 0; j < ids.size(); ++j)
            for (int64_t h = 0; h < NKV; ++h)
                for (int64_t d = 0; d < HD; ++d) {
                    const size_t src = (size_t) (ids[j] * NKV + h) * HD + d;
                    if (gk[(size_t) (j * NKV + h) * HD + d] != f16_from_f32(kcur_all[src])) ++gm;
                    if (gv[(size_t) (j * NKV + h) * HD + d] != f16_from_f32(vcur_all[src])) ++gm;
                }
        require("  kv_gather is BIT-EXACT at page_size " + std::to_string(c.page_size), gm == 0,
                std::to_string(gm) + " wrong bits");
    }

    // ================= 2. indexer_key_append =================
    const int64_t NT = 10;                           // 2 complete blocks + a 2-cell tail
    // The cells sit at POSITIONS 100..109, not at 0..9: `QsaCache.pos` is what the reference rotates by,
    // and a kernel that assumes position == cell index is wrong for any sequence that does not start at 0
    // (prefix reuse, phase 5).  With base 100 the two readings are 514% apart, which is asserted below.
    const int32_t POS_BASE = 100;
    RefCache rc;
    std::vector<float> raw_flat;
    for (int64_t t = 0; t < NT; ++t) {
        std::vector<double> kk((size_t) (NKV * HD)), vv((size_t) (NKV * HD)), rr((size_t) IDXD);
        for (int64_t i = 0; i < NKV * HD; ++i) { kk[(size_t) i] = gauss(rng); vv[(size_t) i] = gauss(rng); }
        // cell MAGNITUDES 1, 4, 16, 64 - for the same reason round 196 gave its residual streams different
        // scales: four i.i.d. rows of normal noise have nearly the same norm by accident, so "pool then norm"
        // and "norm then pool" would agree to within the noise and the fixture could not see its own trap.
        for (int64_t d = 0; d < IDXD; ++d) rr[(size_t) d] = gauss(rng) * std::pow(4.0, (double) (t % 4));
        for (int64_t d = 0; d < IDXD; ++d) raw_flat.push_back((float) rr[(size_t) d]);
        // The reference gets the f32 PROMOTION of what the device gets, not the f64 original.  Feeding it
        // the f64 values left a 6e-8 relative difference, which is invisible in a score but crosses an f32
        // rounding boundary in one value in four - and the spare key is asserted BIT-EXACT.
        for (int64_t d = 0; d < IDXD; ++d) rr[(size_t) d] = (double) raw_flat[(size_t) t * IDXD + d];
        rc.append(kk, vv, rr, (double) (100 + t));
    }
    std::vector<float> w_kn((size_t) IDXD);
    for (auto& x : w_kn) x = 1.0f + 0.1f * rnd(1.0);
    std::vector<float> cos_tab((size_t) (2000 * S.n_rot / 2)), sin_tab(cos_tab.size());
    strata::kernels::build_rope_table((int) S.n_rot, THETA, 2000, cos_tab.data(), sin_tab.data());
    Dev<float> dw_kn, dcos, dsin;
    dw_kn.put(w_kn);
    dcos.put(cos_tab);
    dsin.put(sin_tab);

    const int64_t n_bid = NT / R;
    const size_t pooled_n = (size_t) (n_bid + 1) * IDXD;
    Dev<float> dpooled(pooled_n), ddead((size_t) IDXD), dtail((size_t) (R - 1) * IDXD);
    Dev<int32_t> dblockpos(1);      // the rotation position, which only the DEVICE may read
    // The POSITION is a device value now, because a host scalar would be baked into a captured graph.  One
    // `cudaMemcpy` per cell here; the layer keeps this buffer and updates it per token.
    Dev<int32_t> dpos(1);
    {
        strata::kernels::QsaIndexerBuffers bufs{dtail.p, ddead.p, dpooled.p, dblockpos.p};
        Dev<float> draw((size_t) IDXD);
        for (int64_t t = 0; t < NT; ++t) {      // one cell at a time: the tail is a ring and the spare row MOVES
            check(cudaMemcpy(draw.p, &raw_flat[(size_t) t * IDXD], (size_t) IDXD * 4, cudaMemcpyHostToDevice),
                  "raw");
            const int32_t tpos = (int32_t) t;
            check(cudaMemcpy(dpos.p, &tpos, 4, cudaMemcpyHostToDevice), "pos");
            strata::kernels::indexer_key_append(draw.p, dpos.p, POS_BASE, dw_kn.p, EPS, bufs, S, dcos.p, dsin.p,
                                                nullptr);
        }
        const std::vector<float> got = dpooled.get(pooled_n), got_dead = ddead.get((size_t) IDXD);
        std::vector<double> terms;
        const std::vector<double> want = ref_pooled(rc, w_kn, EPS, (int) S.n_rot, THETA, Alt{}, &terms);
        const std::vector<double> want_raw = ref_pooled_raw(rc);

        // (a) the spare slot: BIT-EXACT, because rope at position 0 is the identity in both (cos = 1, sin = 0
        //     exactly), so the residual the pooled rows carry from the f32 rope cannot appear here.
        int dead_bad = 0;
        for (int64_t d = 0; d < IDXD; ++d)
            if (got_dead[(size_t) d] != (float) want[(size_t) n_bid * IDXD + d]) ++dead_bad;
        require("the spare slot's key (rms_norm of cell 0) is BIT-EXACT", dead_bad == 0,
                std::to_string(dead_bad) + " of " + std::to_string(IDXD) + " wrong");
        require("the spare slot is at row n_bid and nowhere else", n_bid == 2,
                "10 cells / r=4 -> 2 complete blocks");

        // (b) the pooled blocks against the reference
        // THE DENOMINATOR IS THE ROPE'S TERMS, not the result: `a*cos - b*sin` cancels, and the kernel's
        // rotation runs in f32 over an f32 table (6e-8 per term) while the reference's runs in f64.  Dividing
        // by the result reports the cancellation, which is what made this read 1.067e-06 against a 1e-6 bar.
        double worst = 0;
        for (int64_t b = 0; b < n_bid; ++b)
            for (int64_t d = 0; d < IDXD; ++d) {
                const double wv = want[(size_t) b * IDXD + d];
                const double tm = terms[(size_t) b * IDXD + d];
                worst = std::max(worst, rel_terms(wv, (double) got[(size_t) b * IDXD + d], tm));
            }
        report("pooled block keys vs the reference", worst, "rel/|rope terms|");
        if (!(worst <= 1e-6)) { std::printf("    *** over 1e-6 ***\n"); ++g_bad; }

        // (c) THE INCOMPLETE TAIL IS NOT POOLED: the two tail cells take the SPARE slot's score, which is cell
        //     0's key - not their own mean.  A rival reading that pools them would still fill every row.
        {
            std::vector<double> tm((size_t) IDXD, 0.0);
            for (int64_t i = n_bid * R; i < NT; ++i)
                for (int64_t d = 0; d < IDXD; ++d) tm[(size_t) d] += rc.raw[(size_t) i][(size_t) d] / 2.0;
            ref_rms_norm(tm, w_kn, EPS);
            double num = 0, den = 0;
            for (int64_t d = 0; d < IDXD; ++d) {
                num += std::fabs(tm[(size_t) d] - (double) got_dead[(size_t) d]);
                den += std::fabs(tm[(size_t) d]);
            }
            verdict("the incomplete tail is NOT pooled into the spare slot", num / den > 0.5, pct(num / den));
        }

        auto sep = [&](const std::string& name, const Alt& a) {
            const std::vector<double> alt = ref_pooled(rc, w_kn, EPS, (int) S.n_rot, THETA, a);
            double num = 0, den = 0;
            // Rows 0..n_bid INCLUSIVE: the spare slot is row n_bid and three of the four rivals change
            // nothing else, so leaving it out made "the spare key from the LAST raw key" separate 0.00%.
            for (int64_t b = 0; b <= n_bid; ++b)
                for (int64_t d = 0; d < IDXD; ++d) {
                    num += std::fabs(alt[(size_t) b * IDXD + d] - want[(size_t) b * IDXD + d]);
                    den += std::fabs(want[(size_t) b * IDXD + d]);
                }
            verdict(name, num / den > 0.05, pct(num / den));
        };
        Alt a1; a1.rotate_at_block_start = false;
        sep("rotating at the block's LAST cell is observable", a1);
        Alt a2; a2.pool_then_norm = false;
        sep("norm-then-pool vs pool-then-norm is observable", a2);
        Alt a3; a3.dead_from_first = false;
        sep("the spare key from the LAST raw key is observable", a3);
        Alt a4; a4.rope_raw = true;
        sep("rotating the RAW indexer keys is observable", a4);
        {   // THE POSITION BASE.  The same reference with the cells at 0..9 instead of 100..109 - i.e.
            // what a kernel that equates a cell's position with its index would compute.
            RefCache flat = rc;
            for (int64_t j = 0; j < NT; ++j) flat.pos[(size_t) j] = (double) j;
            const std::vector<double> alt = ref_pooled(flat, w_kn, EPS, (int) S.n_rot, THETA, Alt{});
            double num = 0, den = 0;
            for (int64_t b = 0; b <= n_bid; ++b)
                for (int64_t d = 0; d < IDXD; ++d) {
                    num += std::fabs(alt[(size_t) b * IDXD + d] - want[(size_t) b * IDXD + d]);
                    den += std::fabs(want[(size_t) b * IDXD + d]);
                }
            verdict("the rotation's POSITION BASE (pos_base) is observable", num / den > 0.05,
                    pct(num / den));
        }

        // (d) THE TRAP THIS FIXTURE CANNOT SEE, measured and reported rather than hidden.  `rms_norm` divides
        //     by sqrt(mean(x^2) + eps), so the 1/r in the mean cancels EXCEPT through eps: mean and sum differ
        //     by ~eps/(2*mean(x^2)) relative.  `ref/qsa.py` L463-465 makes the same point about its own test.
        {
            std::vector<double> psum = want_raw;
            for (auto& x : psum) x *= (double) R;
            std::vector<double> nm = want_raw, ns = psum;
            for (int64_t b = 0; b < n_bid; ++b) {
                std::vector<double> m(nm.begin() + (size_t) b * IDXD, nm.begin() + (size_t) (b + 1) * IDXD);
                std::vector<double> s(ns.begin() + (size_t) b * IDXD, ns.begin() + (size_t) (b + 1) * IDXD);
                ref_rms_norm(m, w_kn, EPS);
                ref_rms_norm(s, w_kn, EPS);
                for (int64_t d = 0; d < IDXD; ++d) {
                    nm[(size_t) b * IDXD + d] = m[(size_t) d];
                    ns[(size_t) b * IDXD + d] = s[(size_t) d];
                }
            }
            double num = 0, den = 0;
            for (size_t i = 0; i < want_raw.size(); ++i) {
                num += std::fabs(psum[i] - want_raw[i]);
                den += std::fabs(want_raw[i]);
            }
            verdict("the 1/r divisor IS observable in the RAW pooled value", num / den > 0.5,
                    pct(num / den) + " (sum vs mean, before any norm)");
            std::printf("  %-48s      %.3e  <-- below EVERY score tolerance; the divisor cannot be\n",
                        "the same divisor AFTER the norm", rel_l1(ns, nm));
            std::printf("  %-48s      %s\n", "", "checked downstream at all, which is why the raw value is the");
            std::printf("  %-48s      %s\n", "", "only place the mean/sum question is answerable.");
        }
    }

    // ================= 3. qsa_index =================
    std::vector<float> q_idx((size_t) (IDXN * IDXD));
    for (auto& x : q_idx) x = rnd(1.0);
    // A bias of ~0.4 against scores of ~20 separates "applies the bias" from "drops it" by 3.4%, which
    // is not a fixture that can see its own trap. This is a guard for the kernel's optional additional
    // bias, separate from the mandatory incomplete-tail bias; it is sized to
    // be comparable with the scores so that dropping it is material.
    std::vector<float> bias((size_t) (n_bid + 1));
    for (auto& x : bias) x = rnd(8.0f);
    {
        const std::vector<float> got_pooled = dpooled.get(pooled_n);
        const std::vector<double> p64 = as_d(got_pooled);
        const std::vector<double> q_idx64 = as_d(q_idx);
        const int64_t n_kv = NT;
        Dev<float> dp, dqi, dbias, dsc;
        dp.put(got_pooled);
        dqi.put(q_idx);
        dbias.put(bias);
        dsc.alloc((size_t) n_kv);
        strata::kernels::qsa_index(dp.p, n_bid, dqi.p, dbias.p, S, n_kv, dsc.p, nullptr);
        const std::vector<float> got = dsc.get((size_t) n_kv);
        const std::vector<double> want =
            ref_indexer_scores(p64, n_bid, q_idx64, IDXN, IDXD, bias, n_kv, Alt{});

        // The score is a SUM of four relu'd dots plus the bias, so the denominator is the magnitude of the
        // TERMS - |dot| bounds a relu'd dot from above and is what is used here.
        double worst = 0;
        for (int64_t j = 0; j < n_kv; ++j) {
            const int64_t cb = std::min(j / R, n_bid);
            double terms = std::fabs((double) bias[(size_t) cb]);
            if (cb == n_bid && n_kv % R != 0) terms += 1e9;
            for (int64_t h = 0; h < IDXN; ++h) {
                double dot = 0;
                for (int64_t d = 0; d < IDXD; ++d) dot += p64[(size_t) cb * IDXD + d] * q_idx64[(size_t) h * IDXD + d];
                terms += std::fabs(dot);
            }
            worst = std::max(worst, rel_terms(want[(size_t) j], (double) got[(size_t) j], terms));
        }
        report("indexer cell scores vs the reference", worst, "rel/Sum|terms|");
        if (!(worst <= 1e-6)) { std::printf("    *** over 1e-6 ***\n"); ++g_bad; }

        require("cells of one block share its score (0..3)",
                got[0] == got[1] && got[1] == got[2] && got[2] == got[3] && got[0] != got[4]);
        require("cells 8,9 (the tail) take the SPARE slot's score, not block 1's",
                got[8] == got[9] && got[9] != got[7], "got[8] == got[9] != got[7]");

        auto sep_scores = [&](const std::string& name, const Alt& a) {
            const std::vector<double> alt = ref_indexer_scores(p64, n_bid, q_idx64, IDXN, IDXD, bias, n_kv, a);
            double num = 0, den = 0;
            // The mandatory tail bias must not swamp the score-arithmetic fixture.
            for (int64_t j = 0; j < n_bid * R; ++j) {
                num += std::fabs(alt[(size_t) j] - want[(size_t) j]);
                den += std::fabs(want[(size_t) j]);
            }
            verdict(name, num / den > 0.05, pct(num / den));
        };
        Alt a5; a5.relu_per_head = false;
        sep_scores("relu PER HEAD vs relu(sum of heads) is observable", a5);
        Alt a6; a6.use_relu = false;
        sep_scores("dropping the relu is observable", a6);
        Alt a7; a7.use_bias = false;
        sep_scores("dropping the per-block bias is observable", a7);
    }

    // Causal decode's incomplete tail must survive even when its dot-product score
    // ties every completed block. The oracle supplies +1e9 in set_input_qsa;
    // without it, ascending-index top-k drops every tail after the sparse switch.
    // Replay one captured graph across block/page boundaries to catch stale bias.
    {
        const int64_t max_n = 20480;
        const int64_t max_blocks = max_n / R + 1;
        const int64_t cap = strata::kernels::qsa_selection_width(max_n, S);
        Dev<float> pooled, query, scores((size_t) max_n);
        Dev<int32_t> ids((size_t) cap), step(strata::kernels::kStepCount);
        pooled.put(std::vector<float>((size_t) max_blocks * IDXD, 0.0f));
        query.put(std::vector<float>((size_t) IDXN * IDXD, 0.0f));

        cudaStream_t stream;
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        check(cudaStreamCreate(&stream), "tail stream");
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "tail capture begin");
        strata::kernels::qsa_index_step(pooled.p, query.p, nullptr, S, step.p, max_blocks, scores.p, stream);
        strata::kernels::topk_512_step(scores.p, S, cap, step.p, ids.p, stream);
        check(cudaStreamEndCapture(stream, &graph), "tail capture end");
        check(cudaGraphInstantiate(&exec, graph, 0), "tail instantiate");

        const std::vector<int64_t> counts = {
            1, 2, 3, 4, 511, 512, 513, 2046, 2047, 2048, 2049, 2050, 2051,
            2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059, 2060,
            4095, 4096, 4097, 8191, 8192, 8193, 20477, 20478, 20479, 20480};
        for (const int64_t n : counts) {
            const int64_t width = strata::kernels::qsa_selection_width(n, S);
            const int64_t tail = n % R;
            std::vector<int32_t> want;
            for (int64_t j = 0; j < width - tail; ++j) want.push_back((int32_t) j);
            for (int64_t j = n - tail; j < n; ++j) want.push_back((int32_t) j);

            strata::kernels::qsa_index(pooled.p, n / R, query.p, nullptr, S, n, scores.p, nullptr);
            strata::kernels::topk_512(scores.p, n, S, cap, ids.p, nullptr);
            require("tail static n=" + std::to_string(n), ids.get((size_t) width) == want);

            std::vector<int32_t> values(strata::kernels::kStepCount);
            strata::kernels::qsa_step_fill(values.data(), n - 1, S);
            step.put(values);
            check(cudaGraphLaunch(exec, stream), "tail replay");
            check(cudaStreamSynchronize(stream), "tail replay sync");
            require("tail captured n=" + std::to_string(n), ids.get((size_t) width) == want);
            const auto actual_scores = scores.get((size_t) n);
            bool exact_scores = true;
            for (int64_t j = 0; j < n; ++j) {
                const float expected = j >= n - tail ? 1e9f : 0.0f;
                exact_scores = exact_scores && actual_scores[(size_t) j] == expected;
            }
            require("  only incomplete cells receive the bias", exact_scores);
        }
        check(cudaGraphExecDestroy(exec), "tail exec destroy");
        check(cudaGraphDestroy(graph), "tail graph destroy");
        check(cudaStreamDestroy(stream), "tail stream destroy");
    }

    // ================= 4. topk_512, against an independent sort =================
    {
        const int64_t CAP = strata::kernels::qsa_selection_width(MAXC, S);
        Dev<float> dsc;
        Dev<int32_t> dids;
        dsc.alloc((size_t) MAXC);
        dids.alloc((size_t) CAP);
        std::vector<float> sc((size_t) MAXC);

        auto one = [&](const std::string& name, int64_t n_kv) {
            dsc.put(std::vector<float>(sc.begin(), sc.begin() + n_kv));
            strata::kernels::topk_512(dsc.p, n_kv, S, CAP, dids.p, nullptr);
            const int64_t width = strata::kernels::qsa_selection_width(n_kv, S);
            const std::vector<int32_t> g = dids.get((size_t) width);
            const std::vector<int32_t> want = ref_select(as_d(std::vector<float>(sc.begin(), sc.begin() + n_kv)),
                                                         n_kv, TOPK, false);
            const bool ok = g == want;
            require(name, ok, std::to_string(width) + " ids, width = min(n_kv, top_k + r - 1)");
            if (!ok) {
                for (size_t i = 0; i < std::min(g.size(), want.size()); ++i)
                    if (g[i] != want[i]) {
                        std::printf("    first difference at %zu: got %d, want %d\n", i, g[i], want[i]);
                        break;
                    }
            }
            bool asc = true;
            for (size_t i = 1; i < g.size(); ++i) if (g[i] <= g[i - 1]) asc = false;
            require("  ...ascending and unique", asc);
            return g;
        };

        for (auto& x : sc) x = rnd(1.0);
        one("topk_512 n_kv = 1", 1);
        one("topk_512 n_kv = 7 (random)", 7);
        one("topk_512 n_kv = 2048 (the plan's threshold)", 2048);
        one("topk_512 n_kv = 2051 (the reference's own bound)", 2051);
        one("topk_512 n_kv = 2052 (two past the plan's number)", 2052);
        one("topk_512 n_kv = 5000 (random)", 5000);
        one("topk_512 n_kv = 32768 (= kTopkMaxCells)", MAXC);

        // ALL-EQUAL scores: every cell ties, so the answer must be 0..width-1 and any other tie rule shows up
        // on the first element.  ggml does not specify the rule; the reference and the kernel both say
        // ascending index, and this is the case that distinguishes them.
        std::fill(sc.begin(), sc.begin() + 2052, 0.25f);
        one("topk_512 with 2052 EQUAL scores (every cell ties)", 2052);

        // a tie BLOCK straddling the cut: 100 cells share the cut score and only some of them fit
        for (int64_t j = 0; j < 5000; ++j) sc[(size_t) j] = (float) (-(double) j);
        for (int64_t j = 2051 - 50; j < 2051 + 50; ++j) sc[(size_t) j] = -1000.0f;
        one("topk_512 with a 100-cell tie straddling the cut", 5000);

        // THE IDENTITY BELOW THE BOUND.  `select_cells` returns `arange` whenever width == n_kv, which is what
        // "selection skipped when context <= 2,048" means - and it holds to 2,051, not 2,048.
        for (int64_t n : {(int64_t) 2048, (int64_t) 2051}) {
            for (int64_t j = 0; j < n; ++j) sc[(size_t) j] = rnd(1.0);
            const std::vector<int32_t> g = one(std::string("topk_512 n_kv = ") + std::to_string(n) +
                                                   " selects every cell (identity)",
                                               n);
            bool ident = (int64_t) g.size() == n;
            for (int64_t j = 0; j < n && ident; ++j) if (g[(size_t) j] != j) ident = false;
            require("  ...and it really is 0,1,2,...", ident);
        }
        {   // one past the bound the identity must FAIL, or the two checks above prove nothing
            for (int64_t j = 0; j < 2052; ++j) sc[(size_t) j] = rnd(1.0);
            const std::vector<int32_t> g = one("topk_512 n_kv = 2052 is NOT the identity", 2052);
            std::vector<char> kept(2052, 0);
            for (int32_t id : g) kept[(size_t) id] = 1;
            int64_t missing = 0;
            for (int64_t j = 0; j < 2052; ++j) if (!kept[(size_t) j]) ++missing;
            int64_t argmin = 0;
            for (int64_t j = 1; j < 2052; ++j) if (sc[(size_t) j] < sc[(size_t) argmin]) argmin = j;
            require("  ...exactly one cell is dropped, and it is the smallest score",
                    missing == 1 && !kept[(size_t) argmin], std::to_string(missing) + " dropped");
        }
    }

    // ================= 5. qsa_attend on the loud-cell fixture =================
    // `ref/qsa.py` PROPERTY 8's own numbers: a key with 100 in its first dim against zeros, a value of 10
    // against 1, and q[:,0] = 1.  The logit gap is 100/sqrt(256) = 6.25, so the loud cell takes ~99% of its
    // head's weight - and exactly zero when it is not in the selected set.
    {
        const int64_t N = 8;
        std::vector<std::vector<double>> k2, v2;
        std::vector<float> k2f, v2f;
        for (int64_t j = 0; j < N; ++j) {
            std::vector<double> kk((size_t) (NKV * HD), 0.0), vv((size_t) (NKV * HD), 0.0);
            for (int64_t h = 0; h < NKV; ++h) vv[(size_t) (h * HD)] = 1.0;
            kk[0] = j == N - 1 ? 100.0 : 0.0;
            vv[0] = j == N - 1 ? 10.0 : 1.0;
            for (int64_t i = 0; i < NKV * HD; ++i) { k2f.push_back((float) kk[(size_t) i]); v2f.push_back((float) vv[(size_t) i]); }
            k2.push_back(kk);
            v2.push_back(vv);
        }
        std::vector<double> q((size_t) (NH * HD), 0.0);
        for (int64_t h = 0; h < NH; ++h) q[(size_t) (h * HD)] = 1.0;

        Dev<float> dq;
        dq.put(as_f(q));
        const size_t scr = (size_t) N * NKV * HD;
        std::vector<uint16_t> kbits(scr), vbits(scr);
        for (size_t i = 0; i < scr; ++i) { kbits[i] = f16_from_f32(k2f[i]); vbits[i] = f16_from_f32(v2f[i]); }
        Dev<uint16_t> dks(scr), dvs(scr);
        dks.put(kbits);
        dvs.put(vbits);
        Dev<float> dattn((size_t) NH * HD), dwt((size_t) NH * N);

        std::vector<int32_t> all(N), no_loud(N - 1);
        for (int64_t j = 0; j < N; ++j) all[(size_t) j] = (int32_t) j;
        for (int64_t j = 0; j < N - 1; ++j) no_loud[(size_t) j] = (int32_t) j;

        // the scratch is written directly here (no pool), so this section is about the attention alone
        strata::kernels::qsa_attend(dq.p, dks.p, dvs.p, N, S, dattn.p, dwt.p, nullptr);
        const std::vector<double> out_all = as_d(dattn.get((size_t) NH * HD));
        strata::kernels::qsa_attend(dq.p, dks.p, dvs.p, N - 1, S, dattn.p, nullptr, nullptr);
        const std::vector<double> out_sel = as_d(dattn.get((size_t) NH * HD));
        const std::vector<double> wts = as_d(dwt.get((size_t) NH * N));

        verdict("with every cell selected the loud cell dominates its head 0", out_all[0] > 5.0,
                "out[0][0] = " + std::to_string(out_all[0]).substr(0, 6));
        require("with the loud cell unselected it contributes EXACTLY nothing",
                std::fabs(out_sel[0] - 1.0) < 1e-6, "out[0][0] = " + std::to_string(out_sel[0]).substr(0, 8));
        double worst_sum = 0;
        for (int64_t h = 0; h < NH; ++h) {
            double s = 0;
            for (int64_t j = 0; j < N; ++j) s += wts[(size_t) (h * N + j)];
            worst_sum = std::max(worst_sum, std::fabs(s - 1.0));
        }
        report("the returned weights sum to 1 per head", worst_sum, "max |1 - sum|");

        auto sep_attend = [&](const std::string& name, const Alt& a) {
            const std::vector<double> alt = ref_attend(q, NH, NKV, HD, all, k2, v2, a, nullptr);
            double num = 0, den = 0;
            for (size_t i = 0; i < alt.size(); ++i) {
                num += std::fabs(alt[i] - out_all[i]);
                den += std::fabs(alt[i]);
            }
            verdict(name, num / den > 0.05, pct(num / den));
        };
        Alt a8; a8.kv_divide = false;
        sep_attend("q % 2 vs q / 12 for the kv head is observable", a8);
        // The scale trap is NOT asserted on this fixture: its softmax is saturated (the loud cell holds 99%
        // of the weight, so removing a 16x logit scale moves it to 100% and the output by 1.09%).  It is
        // asserted on the random 2100-cell fixture below, where the logits are O(10) and the temperature
        // actually matters.  This is round 196's lesson about a fixture that cannot see its own trap.
        std::printf("  %-48s %-6s %s\n", "  (the scale trap is measured on the random fixture)", "--",
                    "this fixture's softmax is saturated");
        {   // the mask: the same reference with and without the loud cell in the selected set
            const std::vector<double> with_loud = ref_attend(q, NH, NKV, HD, all, k2, v2, Alt{}, nullptr);
            const std::vector<double> without = ref_attend(q, NH, NKV, HD, no_loud, k2, v2, Alt{}, nullptr);
            double num = 0, den = 0;
            for (size_t i = 0; i < with_loud.size(); ++i) {
                num += std::fabs(with_loud[i] - without[i]);
                den += std::fabs(with_loud[i]);
            }
            verdict("attending to the unselected cell is observable", num / den > 0.05, pct(num / den));
        }
    }

    // ================= 6. the whole layer at n_kv = 2100, so the selection is genuinely sparse =============
    int64_t width = 0;
    {
        const int64_t T = 2100;                      // > 2051, so `width` really truncates
        width = strata::kernels::qsa_selection_width(T, S);
        std::vector<float> pk((size_t) T * NKV * HD), pv((size_t) T * NKV * HD), praw((size_t) T * IDXD);
        std::vector<std::vector<double>> k16((size_t) T), v16((size_t) T);
        RefCache pc;
        std::vector<float> q((size_t) NH * HD);
        for (auto& x : q) x = rnd(1.0);
        for (int64_t t = 0; t < T; ++t) {
            std::vector<double> kk((size_t) (NKV * HD)), vv((size_t) (NKV * HD)), rr((size_t) IDXD);
            for (int64_t i = 0; i < NKV * HD; ++i) {
                kk[(size_t) i] = gauss(rng);
                vv[(size_t) i] = gauss(rng);
                pk[(size_t) t * NKV * HD + i] = (float) kk[(size_t) i];
                pv[(size_t) t * NKV * HD + i] = (float) vv[(size_t) i];
            }
            for (int64_t d = 0; d < IDXD; ++d) {
                rr[(size_t) d] = gauss(rng);
                praw[(size_t) t * IDXD + d] = (float) rr[(size_t) d];
            }
            for (int64_t i = 0; i < NKV * HD; ++i) {
                kk[(size_t) i] = (double) pk[(size_t) t * NKV * HD + i];       // the f32 the device got
                vv[(size_t) i] = (double) pv[(size_t) t * NKV * HD + i];
            }
            for (int64_t d = 0; d < IDXD; ++d) rr[(size_t) d] = (double) praw[(size_t) t * IDXD + d];
            pc.append(kk, vv, rr, (double) t);
            k16[(size_t) t].resize((size_t) (NKV * HD));
            v16[(size_t) t].resize((size_t) (NKV * HD));
            for (int64_t i = 0; i < NKV * HD; ++i) {
                k16[(size_t) t][(size_t) i] = (double) f32_from_f16(f16_from_f32(pk[(size_t) t * NKV * HD + i]));
                v16[(size_t) t][(size_t) i] = (double) f32_from_f16(f16_from_f32(pv[(size_t) t * NKV * HD + i]));
            }
        }
        const int64_t p_nbid = T / R;                // 525 complete blocks; the spare key sits at row 525
        const int64_t page_size = 64, n_pages = (T + page_size - 1) / page_size;
        strata::kernels::QsaShapes Sp = S;           // the granule lives in the shapes: see section 1
        Sp.page_size = page_size;
        std::vector<int32_t> table((size_t) n_pages);
        for (int64_t i = 0; i < n_pages; ++i) table[(size_t) i] = (int32_t) (n_pages - 1 - i);   // REVERSED

        Dev<int32_t> dtab;
        dtab.put(table);
        Dev<float> dpk_all, dpv_all;
        dpk_all.put(pk);
        dpv_all.put(pv);
        Dev<uint16_t> dpk((size_t) n_pages * NKV * page_size * HD), dpv((size_t) n_pages * NKV * page_size * HD);
        for (int64_t t = 0; t < T; ++t)
            strata::kernels::kv_append(dpk.p, dpv.p, dtab.p, t, dpk_all.p + (size_t) t * NKV * HD,
                                       dpv_all.p + (size_t) t * NKV * HD, Sp, nullptr);

        const size_t ppooled_n = (size_t) (p_nbid + 1) * IDXD;
        Dev<float> dpp((size_t) ppooled_n), dpdead((size_t) IDXD), dptail((size_t) (R - 1) * IDXD);
        Dev<int32_t> dblockpos2(1);
        Dev<int32_t> dpos2(1);
        Dev<float> draw((size_t) IDXD);
        strata::kernels::QsaIndexerBuffers bufs{dptail.p, dpdead.p, dpp.p, dblockpos2.p};
        for (int64_t t = 0; t < T; ++t) {
            check(cudaMemcpy(draw.p, &praw[(size_t) t * IDXD], (size_t) IDXD * 4, cudaMemcpyHostToDevice), "raw");
            const int32_t tpos = (int32_t) t;
            check(cudaMemcpy(dpos2.p, &tpos, 4, cudaMemcpyHostToDevice), "pos2");
            strata::kernels::indexer_key_append(draw.p, dpos2.p, 0, dw_kn.p, EPS, bufs, S, dcos.p, dsin.p,
                                                nullptr);
        }
        Dev<float> dqidx, dsc;
        dqidx.put(q_idx);
        dsc.alloc((size_t) T);
        strata::kernels::qsa_index(dpp.p, p_nbid, dqidx.p, nullptr, S, T, dsc.p, nullptr);

        Dev<int32_t> dsel((size_t) T);
        strata::kernels::topk_512(dsc.p, T, S, T, dsel.p, nullptr);
        const std::vector<int32_t> sel_sparse = dsel.get((size_t) width);
        const std::vector<float> pool32 = dpp.get(ppooled_n);
        const std::vector<double> pool64 = as_d(pool32);
        const std::vector<double> q_idx64 = as_d(q_idx);
        const std::vector<double> sc_want =
            ref_indexer_scores(pool64, p_nbid, q_idx64, IDXN, IDXD, {}, T, Alt{});
        const std::vector<int32_t> sel_want = ref_select(sc_want, T, TOPK, false);
        require("the selected set matches the reference (2100 cells, width 2051)", sel_sparse == sel_want,
                std::to_string(width) + " ids");

        // HOW CLOSE IS THE CUT?  `indexer_ids` is an EXACT capture key, so a boundary gap of a few fp32 ulp
        // would make it sensitive to the accumulation order.  The gap is measured in ulp of the cut score.
        {
            std::vector<char> issel((size_t) T, 0);
            for (int32_t id : sel_sparse) issel[(size_t) id] = 1;
            double lo = 1e300, hi = -1e300;
            for (int32_t id : sel_sparse) lo = std::min(lo, sc_want[(size_t) id]);
            for (int64_t j = 0; j < T; ++j) if (!issel[(size_t) j]) hi = std::max(hi, sc_want[(size_t) j]);
            if (lo == hi) {
                // The cut sits inside a group of cells that all score exactly the same - here 0.0, a relu
                // signal with all four head dots negative - so nothing but the ascending-index rule decides
                // which of them survive.  A gap metric is undefined there; the tie is the finding.
                int64_t tied_in = 0, tied_out = 0;
                for (int32_t id : sel_sparse) if (sc_want[(size_t) id] == lo) ++tied_in;
                for (int64_t j = 0; j < T; ++j) if (!issel[(size_t) j] && sc_want[(size_t) j] == lo) ++tied_out;
                std::printf("  %-48s      THE CUT IS A TIE at %.6f: %lld selected, %lld rejected\n",
                            "the cut", lo, (long long) tied_in, (long long) tied_out);
                require("  ...and the kernel picks the same ones as the reference", sel_sparse == sel_want);
            } else {
                report("the selection boundary gap, in fp32 ulp of the cut",
                       (lo - hi) / (std::fabs(lo) * 1.2e-7), "gap/ulp");
                std::printf("  %-48s      last selected %.6f, first rejected %.6f\n", "the cut", lo, hi);
            }
        }

        // the pipeline.  `--qsa-dense` swaps the SELECTION for every cell and keeps everything else; the sparse
        // selection above is still compared against the reference either way.
        std::vector<int32_t> sel = sel_sparse;
        if (dense) {
            sel.resize((size_t) T);
            for (int64_t j = 0; j < T; ++j) sel[(size_t) j] = (int32_t) j;
        }
        Dev<int32_t> dsel2;
        dsel2.put(sel);
        const int64_t nsel = (int64_t) sel.size();
        Dev<uint16_t> dks((size_t) nsel * NKV * HD), dvs((size_t) nsel * NKV * HD);
        strata::kernels::kv_gather(dpk.p, dpv.p, dtab.p, dsel2.p, nsel, Sp, dks.p, dvs.p, nullptr);
        Dev<float> dqq, dattn((size_t) NH * HD), dwt((size_t) NH * nsel);
        dqq.put(q);
        strata::kernels::qsa_attend(dqq.p, dks.p, dvs.p, nsel, S, dattn.p, dwt.p, nullptr);
        const std::vector<double> attn = as_d(dattn.get((size_t) NH * HD));
        const std::vector<double> wts = as_d(dwt.get((size_t) NH * nsel));

        const std::vector<double> q64 = as_d(q);
        const std::vector<double> attn_f64 = ref_attend(q64, NH, NKV, HD, sel, pc.k, pc.v, Alt{}, nullptr);
        const std::vector<double> attn_f16 = ref_attend(q64, NH, NKV, HD, sel, k16, v16, Alt{}, nullptr);
        double num = 0, den = 0;
        for (size_t i = 0; i < attn.size(); ++i) {
            num += std::fabs(attn[i] - attn_f16[i]);
            den += std::fabs(attn_f16[i]);
        }
        report("qsa_attend vs the reference on the SAME fp16 cache", num / den, "rel");
        {   // the SCALE trap, on data whose softmax is not saturated
            Alt a; a.apply_scale = false;
            const std::vector<double> alt = ref_attend(q64, NH, NKV, HD, sel, k16, v16, a, nullptr);
            double n2 = 0, d2 = 0;
            for (size_t i = 0; i < alt.size(); ++i) {
                n2 += std::fabs(alt[i] - attn_f16[i]);
                d2 += std::fabs(attn_f16[i]);
            }
            verdict("omitting 1/sqrt(head_dim) is observable (random fixture)", n2 / d2 > 0.05, pct(n2 / d2));
        }
        if (!(num / den <= 1e-5)) { std::printf("    *** over 1e-5 ***\n"); ++g_bad; }
        const double fp16_cost = rel_l1(attn_f64, attn_f16);
        report("the fp16 KV cache's OWN cost (f32 keys vs fp16 keys)", fp16_cost, "rel");
        if (!(fp16_cost <= 1e-3)) {
            std::printf("    *** the phase's 1e-3 for FP16 paths is exceeded by the CACHE ***\n");
            ++g_bad;
        }
        {
            std::vector<double> w_ref;
            ref_attend(q64, NH, NKV, HD, sel, k16, v16, Alt{}, &w_ref);
            double wn = 0, wd = 0;
            for (int64_t j = 0; j < nsel; ++j) {
                wn += std::fabs(wts[(size_t) j] - w_ref[(size_t) j]);
                wd += std::fabs(w_ref[(size_t) j]);
            }
            report("the returned weights vs the reference (head 0)", wn / wd, "rel");
        }

        // the gate
        std::vector<float> q_full((size_t) (NH * 2 * HD));
        for (auto& x : q_full) x = rnd(2.0);
        Dev<float> dqf, dattnf;
        dqf.put(q_full);
        dattnf.put(as_f(attn));
        Dev<uint16_t> dgate((size_t) NH * HD);
        strata::kernels::qsa_gate_apply(dattnf.p, dqf.p, S, dgate.p, nullptr);
        const std::vector<uint16_t> gate16 = dgate.get((size_t) NH * HD);
        const std::vector<double> qf64 = as_d(q_full);
        const std::vector<double> gwant = ref_gate(attn, qf64, NH, HD, Alt{});
        double gnum = 0, gden = 0;
        int gbits = 0;
        for (size_t i = 0; i < gwant.size(); ++i) {
            gnum += std::fabs((double) f32_from_f16(gate16[i]) - gwant[i]);
            gden += std::fabs(gwant[i]);
            if (gate16[i] != f16_from_f32((float) gwant[i])) ++gbits;
        }
        report("qsa_gate_apply vs the double reference", gnum / gden, "rel (f16 out quant ~2.4e-4)");
        if (!(gnum / gden <= 1e-3)) { std::printf("    *** over 1e-3 ***\n"); ++g_bad; }
        std::printf("  %-48s      %d of %lld (CUDA's double exp vs libm's, in the last bits)\n",
                    "fp16 gate bits differing from the reference", gbits, (long long) gwant.size());

        auto sep_gate = [&](const std::string& name, const Alt& a) {
            const std::vector<double> alt = ref_gate(attn, qf64, NH, HD, a);
            double n2 = 0, d2 = 0;
            for (size_t i = 0; i < alt.size(); ++i) {
                n2 += std::fabs(alt[i] - gwant[i]);
                d2 += std::fabs(gwant[i]);
            }
            verdict(name, n2 / d2 > 0.05, pct(n2 / d2));
        };
        Alt a10; a10.gate_second_half = false;
        sep_gate("the gate from the FIRST half is observable", a10);
        Alt a11; a11.gate_sigmoid = false;
        sep_gate("silu instead of sigmoid on the QSA gate is observable", a11);

        // DENSE vs SPARSE.  Below the bound the width IS n_kv, so the two paths must agree EXACTLY (P1.S4's
        // isolation property, `ref/qsa.py` PROPERTY 9b); above it they must differ, or the switch is a no-op.
        require("below the bound, dense and sparse are the SAME selection",
                ref_select(sc_want, 2000, TOPK, false) == ref_select(sc_want, 2000, TOPK, true));
        {
            // BOTH SIDES COMPUTED HERE, from their own id lists and independently of `--qsa-dense`.  The first
            // version compared against the pipeline's `attn`, which in dense mode IS the dense attention - so the
            // switch silently turned the check into a comparison of a thing with itself and reported 0.00%.
            const std::vector<int32_t> dense_all = ref_select(sc_want, T, TOPK, true);
            auto attend_ids = [&](const std::vector<int32_t>& ids) {
                Dev<int32_t> did;
                did.put(ids);
                Dev<uint16_t> dk2((size_t) ids.size() * NKV * HD), dv2((size_t) ids.size() * NKV * HD);
                strata::kernels::kv_gather(dpk.p, dpv.p, dtab.p, did.p, (int64_t) ids.size(), Sp, dk2.p, dv2.p,
                                           nullptr);
                Dev<float> da((size_t) NH * HD);
                strata::kernels::qsa_attend(dqq.p, dk2.p, dv2.p, (int64_t) ids.size(), S, da.p, nullptr, nullptr);
                return as_d(da.get((size_t) NH * HD));
            };
            const std::vector<double> dense_attn = attend_ids(dense_all);
            const std::vector<double> sparse_attn = attend_ids(sel_sparse);
            double dn = 0, dd = 0;
            for (size_t i = 0; i < sparse_attn.size(); ++i) {
                dn += std::fabs(dense_attn[i] - sparse_attn[i]);
                dd += std::fabs(sparse_attn[i]);
            }
            verdict("dense differs from sparse at n_kv = 2100 (> 2051)", dn / dd > 0.05, pct(dn / dd));
        }
    }

    // ================= 7. the empty selection =================
    {
        Dev<float> dq((size_t) NH * HD), dattn((size_t) NH * HD);
        dq.put(std::vector<float>((size_t) NH * HD, 1.0f));
        dattn.put(std::vector<float>((size_t) NH * HD, 7.0f));
        strata::kernels::qsa_attend(dq.p, nullptr, nullptr, 0, S, dattn.p, nullptr, nullptr);
        const std::vector<float> got = dattn.get((size_t) NH * HD);
        bool allz = true;
        for (float x : got) if (x != 0.0f) allz = false;
        require("an EMPTY selection gives a zero attention, not a NaN", allz, "all 6144 elements zero");
    }

    // ================= 8. --bench: what one QSA layer costs at 32K of context =================
    if (bench) {
        const int64_t T = MAXC;                      // 32,768 cells = 32K of context
        const int64_t nbid = T / R;
        const int64_t ps = S.page_size, n_pages = (T + ps - 1) / ps;
        const int64_t w = strata::kernels::qsa_selection_width(T, S);
        std::printf("\n  --bench at n_kv = %lld, page_size %lld, selection width %lld\n", (long long) T,
                    (long long) ps, (long long) w);
        std::vector<int32_t> table((size_t) n_pages);
        for (int64_t i = 0; i < n_pages; ++i) table[(size_t) i] = (int32_t) i;
        Dev<int32_t> dtab;
        dtab.put(table);
        Dev<uint16_t> dpk((size_t) n_pages * NKV * ps * HD), dpv((size_t) n_pages * NKV * ps * HD);
        std::vector<float> kz((size_t) NKV * HD), vz((size_t) NKV * HD);
        for (auto& x : kz) x = rnd(1.0);
        for (auto& x : vz) x = rnd(1.0);
        Dev<float> dkz, dvz;
        dkz.put(kz);
        dvz.put(vz);
        strata::kernels::kv_append(dpk.p, dpv.p, dtab.p, 0, dkz.p, dvz.p, S, nullptr);

        const size_t ppooled_n = (size_t) (nbid + 1) * IDXD;
        Dev<float> dpp((size_t) ppooled_n), dsc((size_t) T);
        std::vector<float> pz(ppooled_n), sz((size_t) T);
        for (auto& x : pz) x = rnd(0.1f);
        for (auto& x : sz) x = std::fabs(rnd(1.0f));
        dpp.put(pz);
        dsc.put(sz);
        Dev<float> dqidx;
        dqidx.put(q_idx);
        Dev<int32_t> dsel((size_t) w);
        Dev<uint16_t> dks((size_t) w * NKV * HD), dvs((size_t) w * NKV * HD);
        Dev<float> dq((size_t) NH * HD), dattn((size_t) NH * HD), dwt((size_t) NH * w);
        std::vector<float> qv((size_t) NH * HD);
        for (auto& x : qv) x = rnd(1.0);
        dq.put(qv);

        auto timeit = [&](const std::string& name, int reps, const std::function<void()>& fn) {
            cudaEvent_t a, b;
            cudaEventCreate(&a);
            cudaEventCreate(&b);
            fn();
            cudaDeviceSynchronize();
            cudaEventRecord(a);
            for (int i = 0; i < reps; ++i) fn();
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            float ms = 0;
            cudaEventElapsedTime(&ms, a, b);
            std::printf("  %-48s %8.1f us\n", name.c_str(), ms * 1000.0f / (float) reps);
            cudaEventDestroy(a);
            cudaEventDestroy(b);
        };
        const int reps = 20;
        timeit("qsa_index  (8193 pooled rows x 4 heads)", reps,
               [&] { strata::kernels::qsa_index(dpp.p, nbid, dqidx.p, nullptr, S, T, dsc.p, nullptr); });
        timeit("topk_512   (32768 cells)", reps,
               [&] { strata::kernels::topk_512(dsc.p, T, S, w, dsel.p, nullptr); });
        strata::kernels::topk_512(dsc.p, T, S, w, dsel.p, nullptr);
        timeit("kv_gather  (2051 ids through the page table)", reps,
               [&] { strata::kernels::kv_gather(dpk.p, dpv.p, dtab.p, dsel.p, w, S, dks.p, dvs.p, nullptr); });
        timeit("qsa_attend (2051 selected, 24 heads)", reps,
               [&] { strata::kernels::qsa_attend(dq.p, dks.p, dvs.p, w, S, dattn.p, dwt.p, nullptr); });
        timeit("kv_append  (one cell, 2 heads x 256)", 200,
               [&] { strata::kernels::kv_append(dpk.p, dpv.p, dtab.p, 0, dkz.p, dvz.p, S, nullptr); });
        timeit("ONE QSA LAYER (index+topk+gather+attend)", reps, [&] {
            strata::kernels::qsa_index(dpp.p, nbid, dqidx.p, nullptr, S, T, dsc.p, nullptr);
            strata::kernels::topk_512(dsc.p, T, S, w, dsel.p, nullptr);
            strata::kernels::kv_gather(dpk.p, dpv.p, dtab.p, dsel.p, w, S, dks.p, dvs.p, nullptr);
            strata::kernels::qsa_attend(dq.p, dks.p, dvs.p, w, S, dattn.p, dwt.p, nullptr);
        });
    }

    // ================= 6. indexer_key_append IS CAPTURABLE =================
    //
    // THE PROPERTY THE DEVICE-POSITION REFACTOR WAS FOR, TESTED DIRECTLY rather than argued.  `indexer_key_append`
    // took `int64_t pos` as a host scalar, which a CUDA graph bakes in at capture time - so a captured call
    // would replay the first token's position forever: `slot = pos % r` never advances, no block completes, and
    // the indexer keeps pooling cells 0-3 for the whole sequence.
    //
    // The test captures ONE call, then replays the SAME graph twice with `pos_dev` and `raw` changed between
    // them, and requires the second replay to land in a DIFFERENT pooled row with the reference's value.  If the
    // position were still baked in, both replays would write row 0 and the pooled rows would be identical.
    //
    // That is the negative control built into the assertion: "row 1 is correct" alone could also pass for a
    // graph that reread the position by accident, but "row 0 unchanged AND row 1 correct AND row 1 != row 0"
    // cannot pass for a baked-in position.
    {
        std::printf("\n-- indexer_key_append under graph capture\n");
        const int R2 = 4;
        const int64_t NCELL = 8;                       // two complete blocks of r=4
        const size_t IDXD2 = (size_t) IDXD;
        std::vector<float> raw2((size_t) NCELL * IDXD2);
        for (size_t i = 0; i < raw2.size(); ++i) raw2[i] = (float) std::sin((double) i * 0.013) * 2.0f - 0.3f;

        const int64_t nb2 = NCELL / R2;
        Dev<float> pooled2((size_t) (nb2 + 2) * IDXD2), dead2(IDXD2), tail2((size_t) (R2 - 1) * IDXD2);
        Dev<int32_t> bpos2(1), pos2(1);
        Dev<float> raw2d(IDXD2);
        strata::kernels::QsaIndexerBuffers bufs2{tail2.p, dead2.p, pooled2.p, bpos2.p};

        // capture ONE call, on its own stream
        cudaStream_t cs = nullptr;
        check(cudaStreamCreate(&cs), "cs");
        check(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal), "begincap");
        strata::kernels::indexer_key_append(raw2d.p, pos2.p, POS_BASE, dw_kn.p, EPS, bufs2, S, dcos.p, dsin.p,
                                            (void*) cs);
        cudaGraph_t g2 = nullptr;
        check(cudaStreamEndCapture(cs, &g2), "endcap");
        check(cudaStreamDestroy(cs), "csd");
        size_t nodes2 = 0;
        check(cudaGraphGetNodes(g2, nullptr, &nodes2), "nodes");
        cudaGraphExec_t ex2 = nullptr;
        check(cudaGraphInstantiate(&ex2, g2, 0), "inst");

        // replay A: cells 0..3, which completes block 0
        for (int t = 0; t < 4; ++t) {
            check(cudaMemcpy(raw2d.p, &raw2[(size_t) t * IDXD2], IDXD2 * 4, cudaMemcpyHostToDevice), "r2");
            const int32_t tp = t;
            check(cudaMemcpy(pos2.p, &tp, 4, cudaMemcpyHostToDevice), "p2");
            check(cudaGraphLaunch(ex2, nullptr), "launchA");
        }
        const std::vector<float> afterA = pooled2.get((size_t) (nb2 + 2) * IDXD2);

        // replay B: cells 4..7 through the SAME graph, which completes block 1
        for (int t = 4; t < 8; ++t) {
            check(cudaMemcpy(raw2d.p, &raw2[(size_t) t * IDXD2], IDXD2 * 4, cudaMemcpyHostToDevice), "r3");
            const int32_t tp = t;
            check(cudaMemcpy(pos2.p, &tp, 4, cudaMemcpyHostToDevice), "p3");
            check(cudaGraphLaunch(ex2, nullptr), "launchB");
        }
        check(cudaDeviceSynchronize(), "sync2");
        const std::vector<float> afterB = pooled2.get((size_t) (nb2 + 2) * IDXD2);

        std::printf("  %-44s %zu nodes for one call\n", "the pooled kernel is capturable", nodes2);

        // row 0 must be UNTOUCHED by the second block's replays
        int row0_same = 0;
        for (size_t d = 0; d < IDXD2; ++d)
            if (afterA[d] == afterB[d]) ++row0_same;
        std::printf("  %-44s %d of %zu unchanged\n", "the first block's row survives replay B", row0_same,
                    IDXD2);
        if (row0_same != (int) IDXD2) ++g_bad;

        // row 1 must now be FILLED by the captured graph - the whole point
        int row1_nonzero = 0;
        for (size_t d = 0; d < IDXD2; ++d)
            if (afterB[IDXD2 + d] != 0.0f) ++row1_nonzero;
        std::printf("  %-44s %d of %zu non-zero\n", "the SECOND block's row arrives through the same graph",
                    row1_nonzero, IDXD2);
        if (row1_nonzero != (int) IDXD2) ++g_bad;

        // and the two rows must DIFFER, which a baked-in position cannot produce
        int rows_differ = 0;
        for (size_t d = 0; d < IDXD2; ++d)
            if (afterB[d] != afterB[IDXD2 + d]) ++rows_differ;
        std::printf("  %-44s %d of %zu differ\n", "the two pooled rows are different keys", rows_differ,
                    IDXD2);
        // A baked-in position would have written row 0 a second time and left row 1 at zero, so these two
        // assertions together are the negative control for the refactor.
        if (rows_differ != (int) IDXD2) ++g_bad;

        check(cudaGraphExecDestroy(ex2), "exd");
        check(cudaGraphDestroy(g2), "gd");
    }

    // ================= the NATIVE indexer's batched append vs the sequential one =================
    // The prefill path appends a chunk in one native_qsa_indexer_append_batch call, the decode path appends
    // token by token - and one cache must never be able to tell which path filled it (rope_scaling.hpp's
    // one-rotation rule).  So the batch's whole end state must be BIT-IDENTICAL to the sequential appends',
    // under all three scaling configs: none (today's rotation), linear 2 (pure angle shrink, no correction)
    // and yarn 2 (corr-dims ramp and the mscale magnitude) - both kernels must take the same RopeScaling.
    // "Bit-identical" is asserted on the bit representations (memcmp of the state words): float != would
    // pass a NaN-payload or -0.0 flip that the claim does not cover.  Eight cells: exactly two completed
    // blocks, so the paths' end states agree on block_pos too (a mid-block chunk end leaves the sequential
    // path pointing at the open block's base and the batch at the last completed one's - upstream C-2's own
    // semantics, the same for every scaling; see native_qsa_indexer.hpp's block_pos contract).
    {
        std::printf("\n-- the native indexer's batched append vs the sequential one\n");
        const int64_t NB = 8;                          // exactly two complete blocks of r=4
        const int32_t BASE = 100;                      // positions 100..107, not cell indices 0..7
        const int64_t MC = 1024;                       // the capacity both sides validate against
        using RST = strata::kernels::RopeScalingType;
        struct Variant { const char* name; RST type; double factor; double ext; };
        const Variant variants[] = {{"none", RST::None, 1.0, 0.0},
                                    {"linear 2", RST::Linear, 2.0, 0.0},
                                    {"yarn 2", RST::YaRN, 2.0, 1.0}};
        std::vector<float> raws((size_t) NB * IDXD);
        for (int64_t t = 0; t < NB; ++t)
            for (int64_t d = 0; d < IDXD; ++d)
                raws[(size_t) t * IDXD + d] = (float) (gauss(rng) * std::pow(4.0, (double) (t % 4)));
        cudaStream_t cs = nullptr;
        check(cudaStreamCreate(&cs), "cs");
        std::vector<float> none_pooled;
        for (const Variant& var : variants) {
            strata::kernels::RopeScaling sc;
            sc.type = var.type;
            sc.factor = var.factor;
            sc.ext_factor = var.ext;
            const size_t prows = (size_t) (MC / R + 1) * IDXD, trows = (size_t) (R - 1) * IDXD;
            Dev<float> pooledA(prows), deadA(IDXD), tailA(trows), pooledB(prows), deadB(IDXD), tailB(trows);
            Dev<int32_t> bposA(1), bposB(1), dpos(1);
            for (Dev<float>* dp : {&pooledA, &deadA, &tailA, &pooledB, &deadB, &tailB})
                check(cudaMemset(dp->p, 0, dp == &pooledA || dp == &pooledB ? prows * 4
                                : dp == &deadA || dp == &deadB ? IDXD * 4 : trows * 4), "zero");
            check(cudaMemset(bposA.p, 0, 4), "zero"); check(cudaMemset(bposB.p, 0, 4), "zero");
            Dev<float> draw((size_t) IDXD), drawAll;
            drawAll.put(raws);
            strata::kernels::QsaIndexerBuffers bufsA{tailA.p, deadA.p, pooledA.p, bposA.p};
            strata::kernels::QsaIndexerBuffers bufsB{tailB.p, deadB.p, pooledB.p, bposB.p};
            for (int64_t t = 0; t < NB; ++t) {         // the sequential side: one cell, its device position
                check(cudaMemcpy(draw.p, &raws[(size_t) t * IDXD], (size_t) IDXD * 4, cudaMemcpyHostToDevice), "raw");
                const int32_t tp = (int32_t) t;
                check(cudaMemcpy(dpos.p, &tp, 4, cudaMemcpyHostToDevice), "pos");
                strata::kernels::native_qsa_indexer_append(draw.p, dpos.p, BASE, dw_kn.p, EPS, bufsA, S, MC, sc, cs);
            }
            strata::kernels::native_qsa_indexer_append_batch(drawAll.p, NB, 0, BASE, dw_kn.p, EPS, bufsB, S, MC, sc, cs);
            // #783 PR-d: native_qsa_indexer_append_steps - the verify window's n cells in one launch (positions at a
            // stride: kStepCount in a window, 1 in the commit graph) - leaves the same buffers as the sequential calls
            for (const int stride : {1, 5}) {
                Dev<float> pooledC(prows), deadC(IDXD), tailC(trows);
                Dev<int32_t> bposC(1), dposC((size_t) NB * stride);
                check(cudaMemset(pooledC.p, 0, prows * 4), "zero");
                check(cudaMemset(deadC.p, 0, IDXD * 4), "zero");
                check(cudaMemset(tailC.p, 0, trows * 4), "zero");
                check(cudaMemset(bposC.p, 0, 4), "zero");
                std::vector<int32_t> hp((size_t) NB * stride, -1);
                for (int64_t t = 0; t < NB; ++t) hp[(size_t) t * stride] = (int32_t) t;
                check(cudaMemcpy(dposC.p, hp.data(), hp.size() * 4, cudaMemcpyHostToDevice), "pos");
                strata::kernels::QsaIndexerBuffers bufsC{tailC.p, deadC.p, pooledC.p, bposC.p};
                strata::kernels::native_qsa_indexer_append_steps(drawAll.p, dposC.p, stride, (int) NB, BASE, dw_kn.p, EPS,
                                                                 bufsC, S, MC, sc, cs);
                check(cudaStreamSynchronize(cs), "sync");
                const std::vector<float> pc = pooledC.get(prows), dc = deadC.get((size_t) IDXD), tc = tailC.get(trows);
                const std::vector<int32_t> bc = bposC.get(1);
                const std::vector<float> pa2 = pooledA.get(prows), da2 = deadA.get((size_t) IDXD), ta2 = tailA.get(trows);
                const std::vector<int32_t> ba2 = bposA.get(1);
                const bool same = std::memcmp(pc.data(), pa2.data(), pc.size() * 4) == 0 &&
                                  std::memcmp(dc.data(), da2.data(), dc.size() * 4) == 0 &&
                                  std::memcmp(tc.data(), ta2.data(), tc.size() * 4) == 0 && bc[0] == ba2[0];
                std::printf("  %-44s %s\n", (std::string("append_steps (stride ") + std::to_string(stride) + ") vs sequential, " + var.name).c_str(),
                            same ? "bit-identical" : "*** WRONG ***");
                if (!same) ++g_bad;
            }
            // engine-latency: native_qsa_indexer_append_steps_layers - the commit's tail restore + append for several
            // layers in one launch - against the per-layer copy + append_steps (3 layers, each its own raw rows, tail
            // snapshot and buffers; cells 2..NB+1 so the first completed block reads the restored tail)
            {
                constexpr int NL = 3;
                Dev<float> pr[NL], dr[NL], tr[NL], pl[NL], dl[NL], tl[NL], src[NL];
                Dev<int32_t> br[NL], bl[NL];
                for (int i = 0; i < NL; ++i) {
                    pr[i].alloc(prows); dr[i].alloc((size_t) IDXD); tr[i].alloc(trows);
                    pl[i].alloc(prows); dl[i].alloc((size_t) IDXD); tl[i].alloc(trows);
                    br[i].alloc(1); bl[i].alloc(1);
                    std::vector<float> ts(trows);
                    for (auto& x : ts) x = (float) gauss(rng);
                    src[i].put(ts);
                }
                for (int i = 0; i < NL; ++i) {
                    check(cudaMemset(pr[i].p, 0, prows * 4), "z"); check(cudaMemset(pl[i].p, 0, prows * 4), "z");
                    check(cudaMemset(dr[i].p, 0, IDXD * 4), "z"); check(cudaMemset(dl[i].p, 0, IDXD * 4), "z");
                    check(cudaMemset(tr[i].p, 0, trows * 4), "z"); check(cudaMemset(tl[i].p, 0, trows * 4), "z");
                    check(cudaMemset(br[i].p, 0, 4), "z"); check(cudaMemset(bl[i].p, 0, 4), "z");
                }
                std::vector<float> rawL((size_t) NL * NB * IDXD);
                for (auto& x : rawL) x = (float) gauss(rng);
                Dev<float> drawL;
                drawL.put(rawL);
                Dev<int32_t> dposL((size_t) NB);
                std::vector<int32_t> hp((size_t) NB);
                for (int64_t t = 0; t < NB; ++t) hp[(size_t) t] = (int32_t) (2 + t);
                check(cudaMemcpy(dposL.p, hp.data(), hp.size() * 4, cudaMemcpyHostToDevice), "pos");
                strata::kernels::QsaIndexerLayers il;
                for (int i = 0; i < NL; ++i) {   // the reference: copy + append_steps, layer by layer
                    check(cudaMemcpyAsync(tr[i].p, src[i].p, trows * 4, cudaMemcpyDeviceToDevice, cs), "cp");
                    strata::kernels::QsaIndexerBuffers b{tr[i].p, dr[i].p, pr[i].p, br[i].p};
                    strata::kernels::native_qsa_indexer_append_steps(drawL.p + (size_t) i * NB * IDXD, dposL.p, 1, (int) NB, BASE,
                                                                     dw_kn.p, EPS, b, S, MC, sc, cs);
                    il.gamma[i] = dw_kn.p; il.tail_src[i] = src[i].p; il.tail[i] = tl[i].p; il.dead[i] = dl[i].p;
                    il.pooled[i] = pl[i].p; il.block_pos[i] = bl[i].p;
                }
                strata::kernels::native_qsa_indexer_append_steps_layers(drawL.p, (int64_t) NB * IDXD, dposL.p, 1, (int) NB, BASE,
                                                                        EPS, il, NL, S, MC, sc, cs);
                check(cudaStreamSynchronize(cs), "sync");
                bool same = true;
                for (int i = 0; i < NL; ++i) {
                    const std::vector<float> a1 = pr[i].get(prows), b1 = pl[i].get(prows), a2 = dr[i].get((size_t) IDXD),
                                             b2 = dl[i].get((size_t) IDXD), a3 = tr[i].get(trows), b3 = tl[i].get(trows);
                    const std::vector<int32_t> a4 = br[i].get(1), b4 = bl[i].get(1);
                    same = same && std::memcmp(a1.data(), b1.data(), a1.size() * 4) == 0 &&
                           std::memcmp(a2.data(), b2.data(), a2.size() * 4) == 0 &&
                           std::memcmp(a3.data(), b3.data(), a3.size() * 4) == 0 && a4[0] == b4[0];
                }
                std::printf("  %-44s %s\n", (std::string("append_steps_layers (3) vs per layer, ") + var.name).c_str(),
                            same ? "bit-identical" : "*** WRONG ***");
                if (!same) ++g_bad;
            }
            check(cudaStreamSynchronize(cs), "sync");
            const std::vector<float> pa = pooledA.get(prows), pb = pooledB.get(prows);
            const std::vector<float> da = deadA.get((size_t) IDXD), db = deadB.get((size_t) IDXD);
            const std::vector<float> ta = tailA.get(trows), tbb = tailB.get(trows);
            const std::vector<int32_t> ba = bposA.get(1), bb = bposB.get(1);
            long long bad = 0;
            for (size_t i = 0; i < pa.size(); ++i) bad += std::memcmp(&pa[i], &pb[i], 4) != 0;
            for (size_t i = 0; i < da.size(); ++i) bad += std::memcmp(&da[i], &db[i], 4) != 0;
            for (size_t i = 0; i < ta.size(); ++i) bad += std::memcmp(&ta[i], &tbb[i], 4) != 0;
            bad += std::memcmp(&ba[0], &bb[0], 4) != 0;
            std::printf("  %-44s %s (%lld of %zu state words differ)\n",
                        (std::string("batch vs sequential, ") + var.name).c_str(),
                        bad ? "*** WRONG ***" : "bit-identical", bad, pa.size() + da.size() + ta.size() + 1);
            if (bad) ++g_bad;
            if (var.type == RST::None) none_pooled = pa;
            else {
                int moved = 0;
                for (size_t i = 0; i < pa.size(); ++i) moved += std::memcmp(&pa[i], &none_pooled[i], 4) != 0;
                std::printf("  %-44s %d of %zu pooled words differ from the none baseline\n",
                            (std::string("the scaling is visible, ") + var.name).c_str(), moved, pa.size());
                if (!moved) { std::printf("    *** the batched append ignored its scaling ***\n"); ++g_bad; }
            }
        }
        check(cudaStreamDestroy(cs), "csd");
    }

    // ================= the spare (position 0) under rope scaling: the table indexer vs the native one =========
    // The spare key (`dead`, and `pooled[0]` until the first block completes) rotates at angle 0.  Its sine is
    // 0 in every scaling, so the rotation is `v * cos_tab[0][pair]`: exactly `v` unscaled (row 0 is (1, 0), the
    // bit-exact value section 2 asserts), `v * mscale` under YaRN - the magnitude every other pooled row and
    // the native kernel's spare carry.  The table kernel used to skip the rotation at position 0, which left
    // the spare the one unscaled key of a scaled cache.  Asserted per variant: the table spare is BIT-EXACTLY
    // the unscaled spare times row 0 on the rotated dims and the unscaled spare past them; `pooled[0]` equals
    // `dead`; the native spare carries the same factor (to f32 rounding: its mscale is formed in float); and
    // under YaRN the factor is observable (row 0 is not 1).
    {
        std::printf("\n-- the spare key (position 0) under rope scaling: table indexer vs native indexer\n");
        using RST = strata::kernels::RopeScalingType;
        struct Variant { const char* name; RST type; double factor; double ext; };
        const Variant variants[] = {{"none", RST::None, 1.0, 0.0},
                                    {"linear 2", RST::Linear, 2.0, 0.0},
                                    {"yarn 2", RST::YaRN, 2.0, 1.0}};
        const int64_t MC = 64, HALF = S.n_rot / 2, NROT = S.n_rot;
        std::vector<float> raw0((size_t) IDXD);
        for (int64_t d = 0; d < IDXD; ++d) raw0[(size_t) d] = rnd(3.0);
        Dev<float> draw0;
        draw0.put(raw0);
        Dev<int32_t> dpos0(1);
        const int32_t zero = 0;
        check(cudaMemcpy(dpos0.p, &zero, 4, cudaMemcpyHostToDevice), "pos0");
        cudaStream_t cs = nullptr;
        check(cudaStreamCreate(&cs), "cs");
        const size_t prows = (size_t) (MC / R + 1) * IDXD, trows = (size_t) (R - 1) * IDXD;
        // one cell at position 0 through either kernel; returns (dead, pooled[0])
        auto spare = [&](bool native, const strata::kernels::RopeScaling& sc, std::vector<float>& dead,
                         std::vector<float>& pooled0) {
            Dev<float> pooled(prows), dd((size_t) IDXD), tail(trows);
            Dev<int32_t> bpos(1);
            check(cudaMemset(pooled.p, 0, prows * 4), "zero");
            check(cudaMemset(dd.p, 0, (size_t) IDXD * 4), "zero");
            check(cudaMemset(tail.p, 0, trows * 4), "zero");
            check(cudaMemset(bpos.p, 0, 4), "zero");
            strata::kernels::QsaIndexerBuffers bufs{tail.p, dd.p, pooled.p, bpos.p};
            if (native) {
                strata::kernels::native_qsa_indexer_append(draw0.p, dpos0.p, 0, dw_kn.p, EPS, bufs, S, MC, sc, cs);
            } else {
                std::vector<float> tc((size_t) (MC * HALF)), ts(tc.size());
                strata::kernels::build_rope_table((int) NROT, sc, (int) MC, tc.data(), ts.data());
                Dev<float> dc, dsn;
                dc.put(tc);
                dsn.put(ts);
                strata::kernels::indexer_key_append(draw0.p, dpos0.p, 0, dw_kn.p, EPS, bufs, S, dc.p, dsn.p, cs);
            }
            check(cudaStreamSynchronize(cs), "sync");
            dead = dd.get((size_t) IDXD);
            pooled0 = pooled.get((size_t) IDXD);
        };
        std::vector<float> t_none, t_none_p, n_none, n_none_p;
        for (const Variant& var : variants) {
            strata::kernels::RopeScaling sc;
            sc.type = var.type;
            sc.factor = var.factor;
            sc.ext_factor = var.ext;
            std::vector<float> row0c((size_t) (MC * HALF)), row0s(row0c.size());
            strata::kernels::build_rope_table((int) NROT, sc, (int) MC, row0c.data(), row0s.data());
            std::vector<float> td, tp, nd, np;
            spare(false, sc, td, tp);
            spare(true, sc, nd, np);
            if (var.type == RST::None) { t_none = td; t_none_p = tp; n_none = nd; n_none_p = np; }
            long long tbad = 0, pbad = 0, sin_bad = 0;
            double nworst = 0;
            for (int64_t i = 0; i < HALF; ++i) sin_bad += row0s[(size_t) i] != 0.0f;
            for (int64_t d = 0; d < IDXD; ++d) {
                const float c0 = d < NROT ? row0c[(size_t) (d % HALF)] : 1.0f;
                const float want = d < NROT ? t_none[(size_t) d] * c0 : t_none[(size_t) d];
                tbad += std::memcmp(&td[(size_t) d], &want, 4) != 0;
                pbad += std::memcmp(&tp[(size_t) d], &td[(size_t) d], 4) != 0;
                const double nwant = (double) n_none[(size_t) d] * (double) c0;
                const double den = std::max(std::fabs(nwant), 1e-6);
                nworst = std::max(nworst, std::fabs((double) nd[(size_t) d] - nwant) / den);
            }
            require(std::string("  ") + var.name + ": row 0 of the table is (c, 0)", sin_bad == 0,
                    std::to_string(sin_bad) + " nonzero sines");
            require(std::string("  ") + var.name + ": the table spare is the unscaled spare times row 0, BIT-EXACT",
                    tbad == 0, std::to_string(tbad) + " of " + std::to_string(IDXD) + " wrong");
            require(std::string("  ") + var.name + ": the table's pooled[0] is its spare", pbad == 0,
                    std::to_string(pbad) + " wrong");
            require(std::string("  ") + var.name + ": the native spare carries the same row-0 factor", nworst <= 4e-7,
                    "worst relative " + std::to_string(nworst));
            if (var.type == RST::YaRN) {
                int moved = 0;
                for (int64_t d = 0; d < NROT; ++d) moved += std::memcmp(&td[(size_t) d], &t_none[(size_t) d], 4) != 0;
                require("  yarn 2: the magnitude correction is observable on the table spare",
                        row0c[0] != 1.0f && moved > NROT / 2,
                        "row0 cos " + std::to_string(row0c[0]) + ", " + std::to_string(moved) + " dims moved");
            }
        }
        check(cudaStreamDestroy(cs), "csd");
    }

    std::printf("\nqsa: %d failures\n", g_bad);
    if (g_bad) return 1;
    if (selftest) std::printf("qsa_parity OK\n");
    return 0;
}
