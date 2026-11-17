// metal/tests/route_parity.cpp - moe_route (softmax + top-10 + the expert plan) against a CPU replica of
// native_router.cu's rule, for 1..8 tokens; logits with deliberate ties.
#include "mtl_rt.h"
#include "ref_quant.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <vector>

namespace {
constexpr int P_NG = 0, P_MAXE = 128, P_GEXP = 16, P_GSTART = P_GEXP + P_MAXE, P_ETOK = P_GSTART + P_MAXE + 4,
              P_EW = P_ETOK + P_MAXE, P_TOKENT = P_EW + P_MAXE, P_INTS = P_TOKENT + P_MAXE;
struct MoeArgs {
    uint64_t gate_off, up_off, down_off, gu_stride, d_stride;
    uint32_t gu_row, d_row, n_embd, n_ff, n_tok, topk, n_expert, gx_gu, gx_dn, pad;
};
}  // namespace

int main() {
    mtl::Context cx;
    const int lib = cx.compile({"quant.metal", "units.metal", "moe.metal"});
    mtl::Pipeline p = cx.pipeline(lib, "moe_route");
    mtl::Buffer lg = cx.alloc(4 * 8 * 512), ids = cx.alloc(4 * 80), w = cx.alloc(4 * 80), plan = cx.alloc(4 * P_INTS);
    ref::Rng r(9);
    int fails = 0;
    double worst_w = 0;
    for (int trial = 0; trial < 200; ++trial) {
        const int nt = 1 + trial % 8;
        for (int i = 0; i < nt * 512; ++i) {
            float v = r.sym() * 4.0f;
            if (trial % 3 == 0) v = std::round(v * 4.0f) / 4.0f;   // many exact ties
            lg.as<float>()[i] = v;
        }
        MoeArgs a{};
        a.n_tok = nt;
        a.topk = 10;
        a.n_expert = 512;
        a.gx_gu = 1;
        a.gx_dn = 1;
        mtl::Dispatch d;
        d.p = &p;
        d.args = {mtl::Arg::B(lg), mtl::Arg::B(ids), mtl::Arg::B(w), mtl::Arg::B(plan), mtl::Arg::V(a)};
        d.tg[0] = 512;
        cx.run({d});
        for (int t = 0; t < nt; ++t) {
            const float* l = lg.as<float>() + t * 512;
            float mx = -INFINITY;
            for (int e = 0; e < 512; ++e) mx = std::max(mx, l[e]);
            std::vector<double> pr(512);
            double s = 0;
            for (int e = 0; e < 512; ++e) s += (pr[e] = std::exp((double)l[e] - mx));
            for (int e = 0; e < 512; ++e) pr[e] /= s;
            // top 10: largest probability, ties -> lower index (probabilities of equal logits are equal)
            std::vector<int> order(512);
            for (int e = 0; e < 512; ++e) order[e] = e;
            std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return l[x] > l[y]; });
            double ssum = 0;
            for (int k = 0; k < 10; ++k) ssum += pr[order[k]];
            for (int k = 0; k < 10; ++k) {
                if (ids.as<int>()[t * 10 + k] != order[k]) { ++fails; break; }
                const double ref = pr[order[k]] / std::max(ssum, 6.103515625e-5);
                worst_w = std::max(worst_w, std::fabs(w.as<float>()[t * 10 + k] - ref) / ref);
            }
        }
        // the plan
        const int* P = plan.as<int>();
        std::set<int> distinct(ids.as<int>(), ids.as<int>() + nt * 10);
        bool ok = P[P_NG] == (int)distinct.size();
        int gi = 0;
        for (int e : distinct) {
            if (P[P_GEXP + gi] != e) ok = false;
            int j = P[P_GSTART + gi];
            for (int i = 0; i < nt * 10; ++i)
                if (ids.as<int>()[i] == e) {
                    if (P[P_ETOK + j] != i / 10 || P[P_TOKENT + i] != j) ok = false;
                    ++j;
                }
            if (j != P[P_GSTART + gi + 1]) ok = false;
            ++gi;
        }
        fails += !ok;
    }
    std::printf("route parity, 200 windows of 1..8 tokens (a third with exact ties): ids/plan mismatches %d, worst weight "
                "rel err %.2e  %s\n", fails, worst_w, fails == 0 && worst_w < 2e-6 ? "PASS" : "FAIL");
    return fails ? 1 : 0;
}
