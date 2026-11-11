// prefill_ws_parity_test - the weight-stationary expert products of PR #3752 (STRATA_MMQ_WS, GP100) against the per-group
// MMQ they replace, byte for byte, on random experts (GPU, synthetic data, no model).
//
// A layer of E resident experts (random gate/up [1280 x 2560] and down [2560 x 640] blocks with sane fp16 scales, laid out
// as the expert cache holds them: gate rows then up rows, the down matrix at down_off) gets random activations, quantized
// as the prompt path does (mmq::quantize), and a routing of random row counts per expert (1 .. 300 rows, so u spans the
// measured range, some experts with one row).  The reference is mmq::Context::run per expert (gate/up, then
// swiglu + quantize, then down), i.e. what Prefill does per 16-expert group; the candidate is ws::gate_up, ONE
// mmq::swiglu_quant over all rows and ws::down.  Compared: GU (floats), Hq (the q8_1 bytes) and Dm (floats) of every row.
// The down is also run alone on the reference Hq, so a difference points at the product. For the IQ4_NL down the run
// repeats with STRATA_WS_FAST on (ws::set_fast(1)); the same Dm is required.
//
// The kernels were written for GP100 (the integer sums are exact, the float epilogue follows MMQ's sm_60 SASS), so the
// verdict is only given on a compute capability 6.0 card; on another card the kernels are run too (ws::set_mode(1))
// and the counts printed for information, with exit 77.  Needs the CUDA build with STRATA_EXPERIMENTAL_SM60.
//
//   prefill_ws_parity_test [--device N] [--any]    exit 0: all identical (cc 6.0); 77: other card; 1: a difference
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/moe_ws.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace mmq = strata::prefill::mmq;
namespace ws = strata::prefill::ws;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Dev {
    void* p = nullptr;
    size_t n = 0;
    explicit Dev(size_t bytes) : n(bytes) {
        ck(cudaMalloc(&p, bytes + 256), "cudaMalloc");
        ck(cudaMemset(p, 0, bytes + 256), "cudaMemset");
    }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    template <class T> T* as() const { return (T*) p; }
    template <class T> void up(const std::vector<T>& h) { ck(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload"); }
    template <class T> std::vector<T> down(size_t count) const {
        std::vector<T> h(count);
        ck(cudaMemcpy(h.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost), "download");
        return h;
    }
};

constexpr int64_t kNEmbd = 2560, kNFF = 640, kGuRows = 1280;

// random blocks of `type` for a [rows x cols] matrix; every block starts with its fp16 scale d: a finite value in
// [2^-6, 2^-5) (all the formats used here put d first)
std::vector<uint8_t> random_matrix(ggml_type type, int64_t rows, int64_t cols, std::mt19937& rng) {
    const size_t bytes = mmq::matrix_bytes((int) type, rows, cols);
    const size_t bs = ggml_type_size(type);
    std::vector<uint8_t> w(bytes);
    for (auto& b : w) b = (uint8_t) rng();
    for (size_t o = 0; o + bs <= bytes; o += bs) {
        const uint16_t d = (uint16_t) ((9u << 10) | (rng() & 0x3FFu));
        std::memcpy(&w[o], &d, 2);
    }
    return w;
}

size_t count_diff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, size_t n) {
    size_t d = 0;
    for (size_t i = 0; i < n; ++i) d += a[i] != b[i];
    return d;
}

// the largest |a - b| over the largest |a|, both float arrays in byte form (how far apart two results are in rounding terms)
double max_norm_err(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, size_t bytes) {
    const float* x = (const float*) a.data();
    const float* y = (const float*) b.data();
    double mx = 0.0, err = 0.0;
    for (size_t i = 0; i < bytes / 4; ++i) {
        mx = std::max(mx, (double) std::fabs(x[i]));
        err = std::max(err, (double) std::fabs(x[i] - y[i]));
    }
    return mx > 0.0 ? err / mx : err;
}

struct Result {
    size_t gu = 0, hq = 0, dm = 0, dm_ref_hq = 0, dm_fast = 0, fast_vs_default = 0;
    double e_gu = 0, e_dm = 0, e_dm_rh = 0;   // normalized max errors against the reference
    bool ran = true;
};

Result run_case(ggml_type gt, ggml_type dt, const std::vector<int32_t>& cnt, uint32_t seed, bool test_fast) {
    std::mt19937 rng(seed);
    const int E = (int) cnt.size();
    std::vector<int32_t> bounds(E + 1, 0);
    for (int j = 0; j < E; ++j) bounds[j + 1] = bounds[j] + cnt[j];
    const int64_t rows = bounds[E];
    const size_t gub = mmq::matrix_bytes((int) gt, kGuRows, kNEmbd), db = mmq::matrix_bytes((int) dt, kNEmbd, kNFF);
    const size_t stride = (gub + db + 255) / 256 * 256;
    // the layer's experts, "resident" in one buffer at 256-byte aligned offsets (the expert cache's slots)
    std::vector<uint8_t> blob((size_t) E * stride, 0);
    std::vector<int64_t> xoff(E);
    for (int j = 0; j < E; ++j) {
        xoff[j] = (int64_t) j * (int64_t) stride;
        const auto g = random_matrix(gt, kGuRows, kNEmbd, rng), d = random_matrix(dt, kNEmbd, kNFF, rng);
        std::memcpy(&blob[(size_t) xoff[j]], g.data(), gub);
        std::memcpy(&blob[(size_t) xoff[j] + gub], d.data(), db);
    }
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t) (rows * kNEmbd));
    for (auto& v : x) v = nd(rng) * 0.7f;

    const size_t xqb = mmq::q8_bytes(rows, kNEmbd), hqb = mmq::q8_bytes(rows, kNFF);
    Dev dblob(blob.size()), dx(x.size() * 4), dxq(xqb), dbnd((size_t) (E + 1) * 4), dxoff((size_t) E * 8);
    Dev dident((size_t) rows * 4), dgu_ref((size_t) rows * kGuRows * 4), dgu((size_t) rows * kGuRows * 4);
    Dev dh((size_t) rows * kNFF * 4), dhq_ref(hqb), dhq(hqb), ddm_ref((size_t) rows * kNEmbd * 4), ddm((size_t) rows * kNEmbd * 4),
        ddm_rh((size_t) rows * kNEmbd * 4), ddm_fast((size_t) rows * kNEmbd * 4);
    dblob.up(blob); dx.up(x); dbnd.up(bounds); dxoff.up(xoff);
    std::vector<int32_t> ident((size_t) rows);
    for (int64_t i = 0; i < rows; ++i) ident[(size_t) i] = (int32_t) i;
    dident.up(ident);
    mmq::quantize(dx.as<float>(), nullptr, dxq.p, (int) gt, kNEmbd, kNEmbd, rows, nullptr);

    // the reference: per expert, as a one-expert group (absolute rows)
    mmq::Context ctx;
    for (int j = 0; j < E; ++j) {
        mmq::Product gu;
        gu.w = dblob.as<uint8_t>() + xoff[j]; gu.type = (int) gt; gu.w_rows = kGuRows; gu.w_cols = kNEmbd; gu.expert_bytes = gub;
        gu.n = 1; gu.xq = dxq.p; gu.bounds = dbnd.as<int32_t>() + j; gu.ids = dident.as<int32_t>();
        gu.total_rows = rows; gu.max_rows = cnt[j]; gu.dst = dgu_ref.as<float>(); gu.ld_dst = kGuRows;
        ctx.run(gu, nullptr);
    }
    mmq::swiglu(dgu_ref.as<float>(), dh.as<float>(), rows, kNFF, false, nullptr);
    mmq::quantize(dh.as<float>(), nullptr, dhq_ref.p, (int) dt, kNFF, kNFF, rows, nullptr);
    for (int j = 0; j < E; ++j) {
        mmq::Product dn;
        dn.w = dblob.as<uint8_t>() + xoff[j] + gub; dn.type = (int) dt; dn.w_rows = kNEmbd; dn.w_cols = kNFF; dn.expert_bytes = db;
        dn.n = 1; dn.xq = dhq_ref.p; dn.bounds = dbnd.as<int32_t>() + j; dn.ids = dident.as<int32_t>();
        dn.total_rows = rows; dn.max_rows = cnt[j]; dn.dst = ddm_ref.as<float>(); dn.ld_dst = kNEmbd;
        ctx.run(dn, nullptr);
    }
    ck(cudaDeviceSynchronize(), "reference");

    // the candidate
    std::vector<int32_t> order(E);
    for (int j = 0; j < E; ++j) order[j] = j;
    std::vector<ws::Chunk> chunks(ws::max_chunks(rows, E));
    const size_t nch = ws::chunks(order.data(), (size_t) E, cnt.data(), bounds.data(), chunks.data());
    chunks.resize(nch);
    Dev dch(std::max<size_t>(1, nch) * sizeof(ws::Chunk));
    dch.up(chunks);
    ws::Layer L;
    L.gu_type = (int) gt; L.d_type = (int) dt; L.w = dblob.p; L.xoff = dxoff.as<int64_t>(); L.down_off = (int64_t) gub;
    L.chunks = dch.as<ws::Chunk>(); L.nchunks = (int64_t) nch; L.rows = rows; L.n = E; L.xq = dxq.p; L.gu = dgu.as<float>();
    L.hq = dhq.p; L.dm = ddm.as<float>();
    Result r;
    if (!ws::gate_up(L, nullptr)) { r.ran = false; return r; }
    mmq::swiglu_quant(dgu.as<float>(), dhq.p, rows, false, nullptr);
    ws::down(L, nullptr);
    ck(cudaDeviceSynchronize(), "candidate");
    // the down alone on the reference H (a difference above this one is the gate/up's)
    ws::Layer L2 = L;
    L2.hq = dhq_ref.p; L2.dm = ddm_rh.as<float>();
    ws::down(L2, nullptr);
    if (test_fast) {
        ws::set_fast(1);
        ws::Layer L3 = L;
        L3.dm = ddm_fast.as<float>();
        ws::down(L3, nullptr);
        ws::set_fast(-1);
    }
    ck(cudaDeviceSynchronize(), "candidate (down alone, fast)");

    const size_t gun = (size_t) rows * kGuRows * 4, dmn = (size_t) rows * kNEmbd * 4;
    const auto gur = dgu_ref.down<uint8_t>(gun), gun_c = dgu.down<uint8_t>(gun);
    r.gu = count_diff(gur, gun_c, gun);
    r.e_gu = max_norm_err(gur, gun_c, gun);
    r.hq = count_diff(dhq_ref.down<uint8_t>(hqb), dhq.down<uint8_t>(hqb), hqb);
    const auto dmr = ddm_ref.down<uint8_t>(dmn);
    const auto dmc = ddm.down<uint8_t>(dmn), dmh = ddm_rh.down<uint8_t>(dmn);
    r.dm = count_diff(dmr, dmc, dmn);
    r.dm_ref_hq = count_diff(dmr, dmh, dmn);
    r.e_dm = max_norm_err(dmr, dmc, dmn);
    r.e_dm_rh = max_norm_err(dmr, dmh, dmn);
    if (test_fast) {
        const auto dmf = ddm_fast.down<uint8_t>(dmn);
        r.dm_fast = count_diff(dmr, dmf, dmn);
        r.fast_vs_default = count_diff(dmc, dmf, dmn);   // STRATA_WS_FAST: the same bits as the default kernel, on any card
    }
    return r;
}

std::vector<int32_t> counts(int E, int lo, int hi, std::mt19937& rng) {
    std::vector<int32_t> c(E);
    for (auto& v : c) v = lo + (int32_t) (rng() % (uint32_t) (hi - lo + 1));
    return c;
}
}  // namespace

int main(int argc, char** argv) {
    int device = 0;
    bool any = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) device = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--any") == 0) any = true;
        else {
            std::fprintf(stderr, "usage: prefill_ws_parity_test [--device N] [--any]\n");
            return 2;
        }
    }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::puts("prefill_ws_parity_test: no CUDA device (skipped)");
        return 77;
    }
    try {
        ck(cudaSetDevice(device), "cudaSetDevice");
        cudaDeviceProp prop{};
        ck(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
        const bool gp100 = prop.major == 6 && prop.minor == 0;
        std::printf("prefill_ws_parity_test on %s (%d.%d)%s\n", prop.name, prop.major, prop.minor,
                    gp100 ? "" : ": not a GP100, so the counts are for information only");
        if (!gp100 && !any) {
            std::puts("prefill_ws_parity_test: not a compute capability 6.0 card (skipped; --any runs it for information)");
            return 77;
        }
        ws::set_mode(1);
        std::mt19937 rng(20261026u);
        const ggml_type gus[] = {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S};
        const ggml_type dns[] = {GGML_TYPE_Q2_0, GGML_TYPE_IQ4_NL};
        size_t total_diff = 0, total_fast = 0;
        double worst_err = 0.0;
        int cases = 0, declined = 0;
        for (ggml_type gt : gus)
            for (ggml_type dt : dns) {
                // a spread of rows per expert (u from ~2 to ~150), and a tiny layer with one-row experts
                const std::vector<std::vector<int32_t>> shapes = {counts(24, 1, 12, rng), counts(24, 5, 40, rng),
                                                                  counts(20, 60, 300, rng), {1, 2, 65, 1, 64, 63}};
                for (const auto& cnt : shapes) {
                    const Result r = run_case(gt, dt, cnt, 1000u + (uint32_t) cases, dt == GGML_TYPE_IQ4_NL);
                    ++cases;
                    if (!r.ran) { ++declined; std::printf("  %-8s -> %-6s %zu experts: ws declined (not faster here)\n", ggml_type_name(gt), ggml_type_name(dt), cnt.size()); continue; }
                    const size_t bad = r.gu + r.hq + r.dm + r.dm_ref_hq + r.dm_fast;
                    total_diff += gp100 ? bad : 0;
                    total_fast += r.fast_vs_default;
                    worst_err = std::max(worst_err, std::max(r.e_gu, r.e_dm_rh));   // Dm after the full chain also carries int8 flips of H
                    std::printf("  %-8s -> %-6s %2zu experts %5d rows: bytes differ GU %zu  Hq %zu  Dm %zu  Dm(down alone) %zu  Dm(fast) %zu"
                                "  fast vs default %zu | max error / max value: GU %.2e  Dm %.2e  Dm(down alone) %.2e%s\n",
                                ggml_type_name(gt), ggml_type_name(dt), cnt.size(),
                                (int) std::accumulate(cnt.begin(), cnt.end(), 0), r.gu, r.hq, r.dm, r.dm_ref_hq, r.dm_fast,
                                r.fast_vs_default, r.e_gu, r.e_dm, r.e_dm_rh, gp100 && bad ? "  <-- FAIL" : "");
                }
            }
        if (!gp100) {   // no verdict on this card: the float epilogue of its MMQ is not the one the kernels follow
            std::printf("prefill_ws_parity_test (information, %s): %d cases (%d declined), STRATA_WS_FAST vs default: %zu differing bytes, "
                        "largest GU / down-alone error against this card's MMQ %.2e of the largest value -> %s\n", prop.name, cases, declined,
                        total_fast, worst_err, total_fast == 0 && worst_err < 1e-5 && declined < cases ? "consistent" : "INCONSISTENT");
            return total_fast == 0 && worst_err < 1e-5 && declined < cases ? 77 : 1;
        }
        std::printf("prefill_ws_parity_test: %d cases (%d declined), %zu differing bytes, STRATA_WS_FAST vs default %zu -> %s\n", cases,
                    declined, total_diff, total_fast, total_diff == 0 && total_fast == 0 && declined < cases ? "PASS" : "FAIL");
        return total_diff == 0 && total_fast == 0 && declined < cases ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "prefill_ws_parity_test: %s\n", e.what());
        return 1;
    }
}
