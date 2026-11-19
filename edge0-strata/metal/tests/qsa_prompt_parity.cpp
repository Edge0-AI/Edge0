// metal/tests/qsa_prompt_parity.cpp - qsa_attn_mm (qsa_prompt.metal, the matrix units, fp16 operands) against qsa_attn
// (fp32, itself checked by qsa_parity), both followed by qsa_merge: the attention output of T query tokens on random
// int8 caches, within the budget (all cells, causal) and past it (2051 selected cells per token); then the speed.
//   qsa_prompt_parity [--bench]
#include "coop_probe.h"
#include "mtl_rt.h"
#include "ref_quant.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

namespace {
constexpr int QH = 24, QKV = 2, QD = 256, QB = 2051, NCH = 33;
struct QsaArgs { uint32_t pos0, pos_base, max_cells, n_tok; float eps; };
double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const bool noflash = std::getenv("QSA_NOFLASH") != nullptr;
    mtl::Context cx;
    const int lib = cx.compile({"qsa.metal", "qsa_prompt.metal"}, "#define NO_GGML_COMMON 1\n");
    std::printf("device %s\n", cx.device_name().c_str());
    const int T = bench ? 2048 : 96, MAXC = 40000;
    mtl::Buffer kq = cx.alloc((size_t)MAXC * QKV * QD), vq = cx.alloc((size_t)MAXC * QKV * QD), ksc = cx.alloc(2 * (size_t)MAXC * QKV * 4),
                vsc = cx.alloc(2 * (size_t)MAXC * QKV * 4), qcur = cx.alloc(4 * (size_t)T * QH * QD), qh = cx.alloc(2 * (size_t)T * QH * QD),
                qfull = cx.alloc(4 * (size_t)T * QH * 2 * QD), ids = cx.alloc(4 * (size_t)T * QB),
                part = cx.alloc(4 * (size_t)T * NCH * QH * (QD + 2)), xa = cx.alloc(4 * ((size_t)4 * T * QH * QD + (size_t)T * QH * QD / 16)),
                xb = cx.alloc(4 * ((size_t)4 * T * QH * QD + (size_t)T * QH * QD / 16));
    ref::Rng r(5);
    for (size_t i = 0; i < kq.bytes; ++i) { kq.as<int8_t>()[i] = (int8_t)((int)(r.next() % 255) - 127); vq.as<int8_t>()[i] = (int8_t)((int)(r.next() % 255) - 127); }
    for (size_t i = 0; i < ksc.bytes / 2; ++i) { ksc.as<uint16_t>()[i] = ref::f32_to_f16(0.004f + 0.004f * r.uni()); vsc.as<uint16_t>()[i] = ref::f32_to_f16(0.004f + 0.004f * r.uni()); }
    for (size_t i = 0; i < (size_t)T * QH * QD; ++i) qcur.as<float>()[i] = r.sym() * 3.0f;
    for (size_t i = 0; i < (size_t)T * QH * 2 * QD; ++i) qfull.as<float>()[i] = r.sym();
    mtl::Pipeline pa = cx.pipeline(lib, "qsa_attn"), pm = cx.pipeline(lib, "qsa_merge"), pmm = cx.pipeline(lib, "qsa_attn_mm", mtl::Constants().b(7, coop_k4(cx)).i(8, std::getenv("QSA_STOP") ? std::atoi(std::getenv("QSA_STOP")) : 0)),
                  pq = cx.pipeline(lib, "qsa_q_f16"), pf = cx.pipeline(lib, "qsa_flash", mtl::Constants().b(7, coop_k4(cx)));
    mtl::Buffer xf = cx.alloc(2 * (size_t)T * QH * QD);
    std::printf("pipelines: qsa_attn_mm max threads %d, tg mem %d; qsa_flash max threads %d, tg mem %d\n", pmm.max_threads,
                pmm.static_tg_mem, pf.max_threads, pf.static_tg_mem);
    int fails = 0;
    for (uint32_t pos0 : {0u, 30000u}) {
        // ids: within the budget the identity; past it 2051 distinct ascending cells < n_kv
        for (int t = 0; t < T; ++t) {
            const uint32_t n_kv = pos0 + t + 1;
            int* o = ids.as<int>() + (size_t)t * QB;
            if (n_kv <= (uint32_t)QB) { for (uint32_t c = 0; c < n_kv; ++c) o[c] = (int)c; continue; }
            std::set<int> s;
            while ((int)s.size() < QB) s.insert((int)(r.next() % n_kv));
            int j = 0;
            for (int c : s) o[j++] = c;
        }
        QsaArgs a{pos0, 0, (uint32_t)MAXC, (uint32_t)T, 1e-6f};
        auto D = [&](const mtl::Pipeline& p, std::vector<mtl::Arg> args, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t tg) {
            mtl::Dispatch d; d.p = &p; d.args = std::move(args); d.grid[0] = gx; d.grid[1] = gy; d.grid[2] = gz; d.tg[0] = tg; return d;
        };
        const uint32_t nq = T * QH * QD;
        std::vector<mtl::Dispatch> ref_ds = {
            D(pa, {mtl::Arg::B(qcur), mtl::Arg::B(kq), mtl::Arg::B(vq), mtl::Arg::B(ksc), mtl::Arg::B(vsc), mtl::Arg::B(ids), mtl::Arg::B(part), mtl::Arg::V(a)}, NCH, QKV, T, 256),
            D(pm, {mtl::Arg::B(part), mtl::Arg::B(qfull), mtl::Arg::B(xa), mtl::Arg::V(a)}, QH, T, 1, 256)};
        std::vector<mtl::Dispatch> mm_ds = {
            D(pq, {mtl::Arg::B(qcur), mtl::Arg::B(qh), mtl::Arg::V(nq)}, (nq + 255) / 256, 1, 1, 256),
            D(pmm, {mtl::Arg::B(qh), mtl::Arg::B(kq), mtl::Arg::B(vq), mtl::Arg::B(ksc), mtl::Arg::B(vsc), mtl::Arg::B(ids), mtl::Arg::B(part), mtl::Arg::V(a)}, NCH, QKV, T, 32),
            D(pm, {mtl::Arg::B(part), mtl::Arg::B(qfull), mtl::Arg::B(xb), mtl::Arg::V(a)}, QH, T, 1, 256)};
        const mtl::Dispatch fl = D(pf, {mtl::Arg::B(qh), mtl::Arg::B(kq), mtl::Arg::B(vq), mtl::Arg::B(ksc), mtl::Arg::B(vsc),
                                        mtl::Arg::B(ids), mtl::Arg::B(qfull), mtl::Arg::B(xf), mtl::Arg::V(a)}, QKV, T, 1, 128);
        if (bench) {
            auto timeit = [&](const std::vector<mtl::Dispatch>& ds) {
                cx.warm(ds, 0.3);
                std::vector<double> t;
                for (int i = 0; i < 5; ++i) t.push_back(cx.run(ds));
                return median(t);
            };
            std::printf("pos0 %5u, T %d: qsa_attn %.2f ms, qsa_attn_mm %.2f ms, merge %.2f ms; qsa_flash (no merge) %.2f ms\n", pos0,
                        T, timeit({ref_ds[0]}) * 1e3, timeit({mm_ds[1]}) * 1e3, timeit({ref_ds[1]}) * 1e3, timeit({fl}) * 1e3);
            continue;
        }
        cx.run(ref_ds);
        std::fill(part.as<float>(), part.as<float>() + part.bytes / 4, std::nanf(""));   // a kernel that writes nothing shows
        cx.run(mm_ds);
        if (std::getenv("QSA_DEBUG")) {
            // how much of the partials the matrix-unit kernel wrote, and token 5's first chunk (m, l, acc[0..3])
            size_t written = 0, nonfinite = 0;
            const float* pp = part.as<float>();
            for (size_t i = 0; i < (size_t)T * NCH * QH * (QD + 2); ++i) {
                if (!std::isnan(pp[i])) ++written;
                if (!std::isfinite(pp[i]) && !std::isnan(pp[i])) ++nonfinite;
            }
            const float* q = pp + ((size_t)5 * NCH * QH) * (QD + 2);
            std::printf("debug pos0 %u: %zu of %zu partial floats written (%zu inf); t5 h0: m %g l %g acc %g %g %g %g\n", pos0,
                        written, (size_t)T * NCH * QH * (QD + 2), nonfinite, q[QD], q[QD + 1], q[0], q[1], q[2], q[3]);
        }
        double se = 0, sr = 0;
        for (size_t i = 0; i < (size_t)T * QH * QD; ++i) {
            const double e = xb.as<float>()[i] - xa.as<float>()[i];
            se += e * e;
            sr += (double)xa.as<float>()[i] * xa.as<float>()[i];
        }
        const double rel = std::sqrt(se / sr);
        const bool ok = rel < 3e-3;
        fails += !ok;
        std::printf("pos0 %5u, T %d: qsa_attn_mm vs qsa_attn (after merge) rel L2 %.2e  %s\n", pos0, T, rel, ok ? "PASS" : "FAIL");
        if (noflash) continue;
        cx.run({fl});
        double sf = 0;
        for (size_t i = 0; i < (size_t)T * QH * QD; ++i) {
            const double e = ref::f16_to_f32(xf.as<uint16_t>()[i]) - xa.as<float>()[i];
            sf += e * e;
        }
        const double relf = std::sqrt(sf / sr);
        const bool okf = relf < 3e-3;
        fails += !okf;
        std::printf("pos0 %5u, T %d: qsa_flash (fp16 out) vs qsa_attn + merge rel L2 %.2e  %s\n", pos0, T, relf, okf ? "PASS" : "FAIL");
    }
    // qsa_score_mm vs qsa_score at pos0 30000: the scores of every block <= n_bid
    {
        const uint32_t pos0 = 30000, MB = MAXC / 4 + 2, nb = (pos0 + T) / 4 + 1;
        mtl::Buffer qidx = cx.alloc(4 * (size_t)T * 4 * 128), pooled = cx.alloc(4 * (size_t)MB * 128), dead = cx.alloc(4 * 128),
                    sa = cx.alloc(4 * (size_t)T * MB), sb = cx.alloc(4 * (size_t)T * MB);
        for (size_t i = 0; i < (size_t)T * 4 * 128; ++i) qidx.as<float>()[i] = r.sym();
        for (size_t i = 0; i < (size_t)MB * 128; ++i) pooled.as<float>()[i] = r.sym();
        for (int i = 0; i < 128; ++i) dead.as<float>()[i] = r.sym();
        QsaArgs a{pos0, 0, (uint32_t)MAXC, (uint32_t)T, 1e-6f};
        mtl::Pipeline ps = cx.pipeline(lib, "qsa_score"), psm = cx.pipeline(lib, "qsa_score_mm");
        mtl::Dispatch d1; d1.p = &ps; d1.args = {mtl::Arg::B(qidx), mtl::Arg::B(pooled), mtl::Arg::B(dead), mtl::Arg::B(sa), mtl::Arg::V(a)};
        d1.grid[0] = (nb + 7) / 8; d1.grid[1] = T; d1.tg[0] = 256;
        mtl::Dispatch d2; d2.p = &psm; d2.args = {mtl::Arg::B(qidx), mtl::Arg::B(pooled), mtl::Arg::B(dead), mtl::Arg::B(sb), mtl::Arg::V(a)};
        d2.grid[0] = (nb + 63) / 64; d2.grid[1] = (T + 15) / 16; d2.tg[0] = 128;
        if (bench) {
            auto timeit = [&](const mtl::Dispatch& d) {
                cx.warm({d}, 0.3);
                std::vector<double> t;
                for (int i = 0; i < 5; ++i) t.push_back(cx.run({d}));
                return median(t);
            };
            std::printf("score pos0 %u, T %d: qsa_score %.2f ms, qsa_score_mm %.2f ms\n", pos0, T, timeit(d1) * 1e3, timeit(d2) * 1e3);
        } else {
            cx.run({d1});
            cx.run({d2});
            double worst = 0;
            for (int t = 0; t < T; ++t) {
                const uint32_t nbid = (pos0 + t + 1) / 4;
                for (uint32_t b = 0; b <= nbid; ++b) {
                    const double x = sa.as<float>()[(size_t)t * MB + b], y = sb.as<float>()[(size_t)t * MB + b];
                    worst = std::max(worst, std::fabs(x - y) / (std::fabs(x) + 1.0));   // scores are O(10)
                }
            }
            const bool ok = worst < 1e-5;
            fails += !ok;
            std::printf("score pos0 %u, T %d: qsa_score_mm vs qsa_score worst rel %.2e  %s\n", pos0, T, worst, ok ? "PASS" : "FAIL");
        }
    }
    if (!bench) std::printf("%s\n", fails ? "SOME FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
