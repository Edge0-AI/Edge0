#include "strata/latent/latent_moe.hpp"

#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <cstring>
#include <vector>

namespace strata::latent {

Plan build_plan(const Geometry& g, const int32_t* topk, const int32_t* labels, int T) {
    Plan p;
    p.cap = T * g.K;
    p.groups = g.E;
    p.start.assign((size_t) g.E + 1, 0);
    p.dst.resize((size_t) p.cap);
    p.tok.resize((size_t) p.cap);
    std::vector<int32_t> pos((size_t) g.E, 0);
    for (int r = 0; r < p.cap; ++r) p.start[(size_t) topk[r] + 1]++;
    for (int e = 0; e < g.E; ++e) p.start[(size_t) e + 1] += p.start[(size_t) e];
    pos = std::vector<int32_t>(p.start.begin(), p.start.end() - 1);
    for (int r = 0; r < p.cap; ++r) {
        const int e = topk[r];
        const int q = pos[(size_t) e]++;
        p.dst[(size_t) q] = r;
        p.tok[(size_t) q] = (r / g.K) * g.G + labels[e];
    }
    return p;
}

bool run_layer(const Geometry& g, const Inputs& in, const DevicePlan& plan, const Buffers& b, void* stream) {
    using namespace strata::kernels;
    if (!in.d_q8 || !in.u_q8 || !in.x || (!in.route_group && (!in.topk || !in.labels)) || !plan.ptr ||
        !plan.start || !plan.n ||
        !plan.dst || !plan.tok || !b.xq || !b.xlq || !b.uq || !b.scratch || !b.xl || !b.yl || !b.uin ||
        !b.uout || !b.y)
        return false;

    // D: [T,H] Q8 x [G*ELL,H]^T -> [T,G*ELL] (Q4DU: fmt 8 Q8_0 / 4 Q4_0, from Inputs)
    quantize_q8_1_rows(in.x, in.T, (int) g.H, b.xq, stream);
    iq_mmvq(strata::kernels::lat_dense_fmt_to_type(in.fmt), in.d_q8, b.xq, b.xl, (int) g.H, (int) (g.G * g.ELL), in.T, stream);
    quantize_q8_1_rows(b.xl, in.T * g.G, (int) g.ELL, b.xlq, stream);

    // A: every expert is a group; activation row is token*G + group.
    const NativeExpertLayout L = native_expert_layout(40, 40, (int) g.ELL, (int) g.I);
    if (!native_expert_supported(40, 40, (int) g.ELL, (int) g.I)) return false;
    native_expert_grouped(L, plan.ptr, plan.start, plan.n, plan.dst, plan.tok, plan.groups, plan.cap, b.xlq,
                          b.scratch, b.yl, stream, plan.groups);

    // U: per group, gather the route rows, Q8 GEMM, scatter back.
    const size_t urow = (size_t) (g.ELL / 32) * (in.fmt == 4 ? 18 : 34);   // Q4DU
    std::vector<int32_t> routes;
    routes.reserve((size_t) plan.cap);
    for (int grp = 0; grp < g.G; ++grp) {
        routes.clear();
        for (int r = 0; r < plan.cap; ++r) {
            const int rg = in.route_group ? in.route_group[r] : in.labels[in.topk[r]];
            if (rg == grp) routes.push_back(r);
        }
        const int n = (int) routes.size();
        if (!n) continue;
        for (int i = 0; i < n; ++i)
            cudaMemcpyAsync(b.uin + (size_t) i * g.ELL, b.yl + (size_t) routes[i] * g.ELL, (size_t) g.ELL * 4,
                            cudaMemcpyDeviceToDevice, (cudaStream_t) stream);
        quantize_q8_1_rows(b.uin, n, (int) g.ELL, b.uq, stream);
        iq_mmvq(strata::kernels::lat_dense_fmt_to_type(in.fmt), in.u_q8 + (size_t) grp * (size_t) g.H * urow, b.uq, b.uout, (int) g.ELL, (int) g.H, n, stream);
        for (int i = 0; i < n; ++i)
            cudaMemcpyAsync(b.y + (size_t) routes[i] * g.H, b.uout + (size_t) i * g.H, (size_t) g.H * 4,
                            cudaMemcpyDeviceToDevice, (cudaStream_t) stream);
    }
    return true;
}

void combine(const Geometry& g, const float* w, const float* y, float* out, int T) {
    std::memset(out, 0, (size_t) T * g.H * sizeof(float));
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < g.K; ++j)
            for (int h = 0; h < g.H; ++h)
                out[(size_t) t * g.H + h] += w[(size_t) t * g.K + j] * y[(size_t) (t * g.K + j) * g.H + h];
}

}  // namespace strata::latent
