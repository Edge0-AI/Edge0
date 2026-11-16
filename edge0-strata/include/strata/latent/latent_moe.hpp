// include/strata/latent/latent_moe.hpp - the latent MoE layer: D -> A -> U -> combine.
//
// The caller owns every allocation.  The module only launches kernels, so it can be
// dropped into the existing verify/cache path without changing buffer ownership.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::latent {

struct Geometry {
    int64_t H = 0;    ///< hidden width (2560)
    int64_t ELL = 0;  ///< latent width (1792)
    int64_t I = 0;    ///< latent expert width (640)
    int G = 0;        ///< latent groups (8)
    int E = 0;        ///< experts (512)
    int K = 0;        ///< top-k (10)
};

/// Host-side route plan, grouped by expert, valid for one (layer, batch group).
struct Plan {
    int cap = 0;
    int groups = 0;
    std::vector<int32_t> start;  ///< E + 1
    std::vector<int32_t> dst;    ///< cap: route index
    std::vector<int32_t> tok;    ///< cap: activation row token*G + group
};

Plan build_plan(const Geometry& g, const int32_t* topk, const int32_t* labels, int T);

/// The device side of Plan, uploaded by the caller.
struct DevicePlan {
    const unsigned long long* ptr = nullptr;
    const int32_t* start = nullptr;
    const int32_t* n = nullptr;
    const int32_t* dst = nullptr;
    const int32_t* tok = nullptr;
    int cap = 0;
    int groups = 0;
};

struct Inputs {
    const uint8_t* d_q8 = nullptr;        ///< [G*ELL, H] Q8_0 rows
    const uint8_t* u_q8 = nullptr;        ///< [G*H, ELL] Q8_0 rows
    const float* x = nullptr;             ///< [T, H]
    const int32_t* topk = nullptr;        ///< [T*K]
    const int32_t* labels = nullptr;      ///< [E], optional when route_group is set
    const int32_t* route_group = nullptr; ///< [cap] host group per route, optional
    const float* w = nullptr;             ///< [T*K]
    int T = 0;
    int fmt = 8;                          ///< Q4DU: the D/U pack's format (8 = Q8_0, 4 = Q4_0)
};

struct Buffers {
    void* xq = nullptr;        ///< T * (H/32) * 36
    void* xlq = nullptr;       ///< T*G * (ELL/32) * 36
    void* uq = nullptr;        ///< cap * (ELL/32) * 36
    void* scratch = nullptr;   ///< native_expert_scratch_bytes(cap, I)
    float* xl = nullptr;       ///< [T*G, ELL]
    float* yl = nullptr;       ///< [cap, ELL]
    float* uin = nullptr;      ///< [cap, ELL]
    float* uout = nullptr;     ///< [cap, H]
    float* y = nullptr;        ///< [cap, H]
};

/// D (Q8) -> q8_1 -> A (native NVFP4 grouped) -> U (Q8) -> y.
bool run_layer(const Geometry& g, const Inputs& in, const DevicePlan& plan, const Buffers& b, void* stream);

/// A -> U for entries computed off the primary. Mirrors the verify path's latent A+U exactly:
/// `native_expert_grouped` over each entry's q8_1 xl row (tok = token*G + labels[id]), then
/// `latent_u_grouped` with this layer's U and labels. Unlike run_layer it does NOT run the D stage (those f32 xl
/// rows are already on the primary and shared as mapped host memory); `xlq` is the caller's q8_1 of those rows and
/// `tok` is already rewritten. `out` gets one H-wide row per entry. gu_type/d_type come from the pack
/// (NVFP4 = 40,40; q3k = 11,20); fmt 8 (Q8_0) or 4 (Q4_0). Every buffer is owned by the caller.
struct AUBuffers {
    const void* xlq = nullptr;                 ///< q8_1 xl rows, (T*G) x ELL, row = token*G + group
    const unsigned long long* ptr = nullptr;   ///< blob device ptr per group
    const int32_t* start = nullptr;            ///< groups + 1
    const int32_t* n = nullptr;                ///< pointer to the group count (int32)
    const int32_t* dst = nullptr;              ///< entry -> compact output row
    const int32_t* tok = nullptr;              ///< entry -> xl row (already token*G + label)
    const int32_t* ids = nullptr;              ///< entry -> expert id
    const int32_t* labels = nullptr;           ///< device labels for this layer [E]
    const uint8_t* u = nullptr;                ///< device U for this layer [G*H, ELL]
    int64_t gu_type = 40, d_type = 40;
    int ELL = 0, I = 0, G = 0, E = 0, H = 0, fmt = 8;
    int cap_entries = 0, cap_groups = 0;
    float* yl = nullptr;                       ///< [cap_entries, ELL] f32 scratch
    void* scratch = nullptr;                   ///< native_expert_scratch_bytes(cap_entries, I)
    void* uq = nullptr;                        ///< [cap_entries, ELL] q8_1 scratch
    float* out = nullptr;                      ///< [cap_entries, H]
};

bool a_u(const AUBuffers& a, void* stream);

/// Host combine: out[t] = sum_j w[t,j] * y[t*K+j].
void combine(const Geometry& g, const float* w, const float* y, float* out, int T);

}  // namespace strata::latent
