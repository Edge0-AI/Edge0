// metal/kernels/moe.metal - routed experts of one layer for a verify window of 1..8 tokens, grouped by expert:
//
//   moe_plan    one threadgroup: the window's top-k ids -> groups (distinct experts, by id) with their entries
//               (token, weight), each (token, k)'s entry index, and the indirect grid sizes of the next two kernels.
//               Deterministic (no atomics).
//   moe_gu      per group, a block of FFN rows: gate and up rows read once, applied to every entry of the group
//               (the shared input x is read from cache), h = silu(gate) * up per entry.
//   moe_down    per group, a block of output rows: ye[entry] = weight * (down . h[entry]).
//   moe_combine y[token] = sum over k of ye[entry(token, k)], in k order.
//
// The plan is one int buffer (see P_* below).  Expert addressing: MoeArgs (a pack blob or the GGUF tensors):
// [gate rows | up rows | down rows] in GGUF blocks (tools/iq_pack.py's experts.bin layout).

#define P_NG 0
#define P_IGU 4
#define P_IDN 8
#define P_MAXE 128
#define P_GEXP 16
#define P_GSTART (P_GEXP + P_MAXE)
#define P_ETOK (P_GSTART + P_MAXE + 4)
#define P_EW (P_ETOK + P_MAXE)
#define P_TOKENT (P_EW + P_MAXE)
#define P_INTS (P_TOKENT + P_MAXE)

// Expert e's matrices: gate at W + gate_off + e * gu_stride, up at W + up_off + e * gu_stride, down at
// W + down_off + e * d_stride (a pack's blobs: gate_off 0, strides = the blob; a GGUF: the three tensors).
struct MoeArgs {
    ulong gate_off;
    ulong up_off;
    ulong down_off;
    ulong gu_stride;
    ulong d_stride;
    uint gu_row;     // bytes per gate/up row
    uint d_row;      // bytes per down row
    uint n_embd;
    uint n_ff;
    uint n_tok;
    uint topk;
    uint n_expert;
    uint gx_gu;      // threadgroups along the rows for moe_gu (n_ff / rows per threadgroup)
    uint gx_dn;      // ... for moe_down (n_embd / rows per threadgroup)
    uint pad;
};

// The SSD-streaming arena (metal/src/engine/stream.cpp): with function constant 20 set, expert e of the layer lies in
// an arena slot, smap[e] = (chunk << 28) | its 16 KiB page in that chunk; the chunks are W (buffer 0) and buffers 7..9
// (gate_off / up_off / down_off are then offsets inside the slot).  The same block is in moe_mm.metal.
#ifndef STRATA_ARENA_METAL
#define STRATA_ARENA_METAL
constant bool kArenaFc [[function_constant(20)]];
constant bool kArena = is_function_constant_defined(kArenaFc) && kArenaFc;
static inline device const uchar* arena_slot(uint v, device const uchar* W, device const uchar* W1, device const uchar* W2,
                                             device const uchar* W3) {
    const uint c = v >> 28;
    device const uchar* b = c == 0 ? W : (c == 1 ? W1 : (c == 2 ? W2 : W3));
    return b + (ulong)(v & 0x0FFFFFFFu) * 16384ul;
}
#endif

static inline void plan_body(device const int* ids, device const float* w, device int* plan, constant MoeArgs& a,
                             ushort tid, ushort nth, ushort sg, ushort lane, threadgroup int* s_grp,
                             threadgroup int* s_ent) {
    const int ne = int(a.n_tok * a.topk);
    // one thread per expert id (nth == n_expert, at most 1024)
    const int e = tid;
    int cnt = 0;
    if (e < int(a.n_expert))
        for (int i = 0; i < ne; ++i) cnt += ids[i] == e;
    const int has = cnt > 0;
    const int pg = simd_prefix_exclusive_sum(has), pe = simd_prefix_exclusive_sum(cnt);
    if (lane == 31) { s_grp[sg] = pg + has; s_ent[sg] = pe + cnt; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        int og = 0, oe = 0;
        const int nsg = (nth + 31) / 32;
        for (int i = 0; i < nsg; ++i) {
            const int g = s_grp[i], n = s_ent[i];
            s_grp[i] = og;
            s_ent[i] = oe;
            og += g;
            oe += n;
        }
        plan[P_NG] = og;
        plan[P_IGU] = int(a.gx_gu); plan[P_IGU + 1] = og; plan[P_IGU + 2] = 1;
        plan[P_IDN] = int(a.gx_dn); plan[P_IDN + 1] = og; plan[P_IDN + 2] = 1;
        plan[P_GSTART + og] = oe;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (has) {
        const int gi = s_grp[sg] + pg, off = s_ent[sg] + pe;
        plan[P_GEXP + gi] = e;
        plan[P_GSTART + gi] = off;
        int j = 0;
        for (int i = 0; i < ne; ++i)
            if (ids[i] == e) {
                plan[P_ETOK + off + j] = i / int(a.topk);
                plan[P_EW + off + j] = as_type<int>(w[i]);
                plan[P_TOKENT + i] = off + j;
                ++j;
            }
    }
}

kernel void moe_plan(device const int* ids [[buffer(0)]], device const float* w [[buffer(1)]],
                     device int* plan [[buffer(2)]], constant MoeArgs& a [[buffer(3)]],
                     ushort tid [[thread_position_in_threadgroup]], ushort nth [[threads_per_threadgroup]],
                     ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup int s_grp[32], s_ent[32];
    plan_body(ids, w, plan, a, tid, nth, sg, lane, s_grp, s_ent);
}

// The router's tail for the window, then the plan, in one threadgroup of n_expert (512) threads: simdgroup t routes
// token t exactly as native_router.cu (pinned llama.cpp topk-moe): softmax over the 512 logits, top 10 by
// probability (equal values: the lower expert index), weights = p / max(sum of the selected, 2^-14), the sum taken
// by winning lane in selection order and reduced with the same xor tree.
constant bool MOE_ROUTE_ONLY [[function_constant(28)]];
kernel void moe_route(device const float* logits [[buffer(0)]], device int* ids [[buffer(1)]],
                      device float* w [[buffer(2)]], device int* plan [[buffer(3)]], constant MoeArgs& a [[buffer(4)]],
                      ushort tid [[thread_position_in_threadgroup]], ushort nth [[threads_per_threadgroup]],
                      ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup int s_grp[32], s_ent[32];
    if (sg < a.n_tok) {
        device const float* l = logits + sg * 512;
        float v[16];
        for (int i = 0; i < 16; ++i) v[i] = l[lane + 32 * i];
        float mx = -INFINITY;
        for (int i = 0; i < 16; ++i) mx = max(mx, v[i]);
        for (ushort m = 16; m > 0; m >>= 1) mx = max(mx, simd_shuffle_xor(mx, m));
        float sum = 0.0f;
        for (int i = 0; i < 16; ++i) {
            v[i] = exp(v[i] - mx);
            sum += v[i];
        }
        for (ushort m = 16; m > 0; m >>= 1) sum += simd_shuffle_xor(sum, m);
        const float rcp = 1.0f / sum;
        for (int i = 0; i < 16; ++i) {
            v[i] *= rcp;
            if (isnan(v[i])) v[i] = -FLT_MAX;
        }
        float selected = 0.0f, ssum = 0.0f;
        for (int rank = 0; rank < 10; ++rank) {
            float best = v[0];
            int ex = lane;
            for (int i = 1; i < 16; ++i)
                if (v[i] > best) { best = v[i]; ex = lane + 32 * i; }
            for (ushort m = 16; m > 0; m >>= 1) {
                const float ob = simd_shuffle_xor(best, m);
                const int oe = simd_shuffle_xor(ex, m);
                if (ob > best || (ob == best && oe < ex)) { best = ob; ex = oe; }
            }
            if ((ex & 31) == lane) {
                v[ex / 32] = -INFINITY;
                ids[sg * 10 + rank] = ex;
                ssum += best;
            }
            if (rank == lane) selected = best;
        }
        for (ushort m = 16; m > 0; m >>= 1) ssum += simd_shuffle_xor(ssum, m);
        ssum = max(ssum, 6.103515625e-5f);
        const float inv = 1.0f / ssum;
        if (lane < 10) w[sg * 10 + lane] = selected * inv;
    }
    if (!is_function_constant_defined(MOE_ROUTE_ONLY) || !MOE_ROUTE_ONLY) {
        threadgroup_barrier(mem_flags::mem_device);
        plan_body(ids, w, plan, a, tid, nth, sg, lane, s_grp, s_ent);
    }
}

// ---- gate + up + SwiGLU -------------------------------------------------------------------------------------------
// One pass over a group's rows for all its NE entries (1..8: an expert appears once per token); NE is uniform per
// threadgroup, so the kernel switches once into a body specialised for it.
template <int F, int LPR, int NR, int NE>
static inline void gu_body(device const uchar* g0, device const uchar* u0, device const float* x,
                           device const float* xs16, device float* h, device const int* etok, int c, int row0, int li,
                           constant MoeArgs& a, threadgroup const uint* tab) {
    constexpr int UNIT = fmt<F>::UNIT;
    float ag[NR][NE], au[NR][NE];
    for (int r = 0; r < NR; ++r)
        for (int t = 0; t < NE; ++t) ag[r][t] = au[r][t] = 0.0f;
    const int U = int(a.n_embd) / UNIT;
    for (int u = li; u < U; u += LPR) {
        float qg[NR][UNIT], qu[NR][UNIT];
        float4 cg[NR], cu[NR];
        for (int r = 0; r < NR; ++r) {
            Dec<F>::run(g0 + r * a.gu_row, u, qg[r], cg[r], tab);
            Dec<F>::run(u0 + r * a.gu_row, u, qu[r], cu[r], tab);
        }
        for (int t = 0; t < NE; ++t) {
            const int tok = etok[t];
            float xs[UNIT], xsum[2];
            X<F>::load(x + tok * int(a.n_embd), xs16 + tok * int(a.n_embd / 16), u, xs, xsum);
            for (int r = 0; r < NR; ++r) {
                ag[r][t] += unit_combine<UNIT>(qg[r], cg[r], xs, xsum);
                au[r][t] += unit_combine<UNIT>(qu[r], cu[r], xs, xsum);
            }
        }
    }
    for (int r = 0; r < NR; ++r)
        for (int t = 0; t < NE; ++t) {
            const float gv = lane_sum<LPR>(ag[r][t]), uv = lane_sum<LPR>(au[r][t]);
            if (li == 0) h[(c + t) * int(a.n_ff) + row0 + r] = gv / (1.0f + exp(-gv)) * uv;
        }
}

// bench probe (function constant 22): 1 = no table load (wrong values for the i-quants)
constant int MOE_PROBE [[function_constant(22)]];
constant int moe_probe = is_function_constant_defined(MOE_PROBE) ? MOE_PROBE : 0;

template <int F, int LPR, int NR, int SG>
kernel void moe_gu(device const uchar* W [[buffer(0)]], device const float* x [[buffer(1)]],
                   device float* h [[buffer(2)]], device const int* plan [[buffer(3)]],
                   constant MoeArgs& a [[buffer(4)]], device const float* xs16 [[buffer(5)]],
                   device const uint* smap [[buffer(6)]], device const uchar* W1 [[buffer(7)]],
                   device const uchar* W2 [[buffer(8)]], device const uchar* W3 [[buffer(9)]],
                   uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                   ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int RG = 32 / LPR;
    threadgroup uint tab[fmt<F>::TAB];
    if (moe_probe != 1) {
        load_tab<F>(tab, tid, SG * 32);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const int g = int(tg.y);
    if (g >= plan[P_NG]) return;
    const int li = lane % LPR;
    const int row0 = ((int(tg.x) * SG + sg) * RG + lane / LPR) * NR;
    const ulong ex = (ulong)plan[P_GEXP + g];
    const int c = plan[P_GSTART + g], ne = plan[P_GSTART + g + 1] - c;
    device const int* etok = plan + P_ETOK + c;
    device const uchar* E = kArena ? arena_slot(smap[ex], W, W1, W2, W3) : W + ex * a.gu_stride;
    device const uchar* g0 = E + a.gate_off + (ulong)row0 * a.gu_row;
    device const uchar* u0 = E + a.up_off + (ulong)row0 * a.gu_row;
    switch (ne) {
        case 1: gu_body<F, LPR, NR, 1>(g0, u0, x, xs16, h, etok, c, row0, li, a, tab); break;
        case 2: gu_body<F, LPR, NR, 2>(g0, u0, x, xs16, h, etok, c, row0, li, a, tab); break;
        case 3: gu_body<F, LPR, NR, 3>(g0, u0, x, xs16, h, etok, c, row0, li, a, tab); break;
        case 4: gu_body<F, LPR, NR, 4>(g0, u0, x, xs16, h, etok, c, row0, li, a, tab); break;
        default:
            for (int k = 0; k < ne; k += 4) {
                const int m = min(4, ne - k);
                if (m == 4) gu_body<F, LPR, NR, 4>(g0, u0, x, xs16, h, etok + k, c + k, row0, li, a, tab);
                else if (m == 3) gu_body<F, LPR, NR, 3>(g0, u0, x, xs16, h, etok + k, c + k, row0, li, a, tab);
                else if (m == 2) gu_body<F, LPR, NR, 2>(g0, u0, x, xs16, h, etok + k, c + k, row0, li, a, tab);
                else gu_body<F, LPR, NR, 1>(g0, u0, x, xs16, h, etok + k, c + k, row0, li, a, tab);
            }
    }
}

// ---- down ---------------------------------------------------------------------------------------------------------
template <int F, int LPR, int NR, int NE>
static inline void dn_body(device const uchar* d0, device const float* h, device const float* hs16, device float* ye,
                           device const int* plan, int c, int row0, int li, constant MoeArgs& a,
                           threadgroup const uint* tab) {
    constexpr int UNIT = fmt<F>::UNIT;
    float ac[NR][NE];
    for (int r = 0; r < NR; ++r)
        for (int t = 0; t < NE; ++t) ac[r][t] = 0.0f;
    const int U = int(a.n_ff) / UNIT;
    for (int u = li; u < U; u += LPR) {
        float q[NR][UNIT];
        float4 sc[NR];
        for (int r = 0; r < NR; ++r) Dec<F>::run(d0 + r * a.d_row, u, q[r], sc[r], tab);
        for (int t = 0; t < NE; ++t) {
            float xs[UNIT], xsum[2];
            X<F>::load(h + (c + t) * int(a.n_ff), hs16 + (c + t) * int(a.n_ff / 16), u, xs, xsum);
            for (int r = 0; r < NR; ++r) ac[r][t] += unit_combine<UNIT>(q[r], sc[r], xs, xsum);
        }
    }
    for (int r = 0; r < NR; ++r)
        for (int t = 0; t < NE; ++t) {
            const float v = lane_sum<LPR>(ac[r][t]);
            if (li == 0) ye[(c + t) * int(a.n_embd) + row0 + r] = as_type<float>(plan[P_EW + c + t]) * v;
        }
}

template <int F, int LPR, int NR, int SG>
kernel void moe_down(device const uchar* W [[buffer(0)]], device const float* h [[buffer(1)]],
                     device float* ye [[buffer(2)]], device const int* plan [[buffer(3)]],
                     constant MoeArgs& a [[buffer(4)]], device const float* hs16 [[buffer(5)]],
                     device const uint* smap [[buffer(6)]], device const uchar* W1 [[buffer(7)]],
                     device const uchar* W2 [[buffer(8)]], device const uchar* W3 [[buffer(9)]],
                     uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                     ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int RG = 32 / LPR;
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, SG * 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int g = int(tg.y);
    if (g >= plan[P_NG]) return;
    const int li = lane % LPR;
    const int row0 = ((int(tg.x) * SG + sg) * RG + lane / LPR) * NR;
    const ulong ex = (ulong)plan[P_GEXP + g];
    const int c = plan[P_GSTART + g], ne = plan[P_GSTART + g + 1] - c;
    device const uchar* d0 = (kArena ? arena_slot(smap[ex], W, W1, W2, W3) : W + ex * a.d_stride) + a.down_off + (ulong)row0 * a.d_row;
    switch (ne) {
        case 1: dn_body<F, LPR, NR, 1>(d0, h, hs16, ye, plan, c, row0, li, a, tab); break;
        case 2: dn_body<F, LPR, NR, 2>(d0, h, hs16, ye, plan, c, row0, li, a, tab); break;
        case 3: dn_body<F, LPR, NR, 3>(d0, h, hs16, ye, plan, c, row0, li, a, tab); break;
        case 4: dn_body<F, LPR, NR, 4>(d0, h, hs16, ye, plan, c, row0, li, a, tab); break;
        default:
            for (int k = 0; k < ne; k += 4) {
                const int m = min(4, ne - k);
                if (m == 4) dn_body<F, LPR, NR, 4>(d0, h, hs16, ye, plan, c + k, row0, li, a, tab);
                else if (m == 3) dn_body<F, LPR, NR, 3>(d0, h, hs16, ye, plan, c + k, row0, li, a, tab);
                else if (m == 2) dn_body<F, LPR, NR, 2>(d0, h, hs16, ye, plan, c + k, row0, li, a, tab);
                else dn_body<F, LPR, NR, 1>(d0, h, hs16, ye, plan, c + k, row0, li, a, tab);
            }
    }
}

kernel void moe_combine(device const float* ye [[buffer(0)]], device float* y [[buffer(1)]],
                        device const int* plan [[buffer(2)]], constant MoeArgs& a [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
    const uint t = gid / a.n_embd, k = gid % a.n_embd;
    if (t >= a.n_tok) return;
    float s = 0.0f;
    for (uint j = 0; j < a.topk; ++j) s += ye[plan[P_TOKENT + t * a.topk + j] * int(a.n_embd) + k];
    y[gid] = s;
}

// ---- instantiations: moe_gu_<type>_<LPR>_<NR>_<SG>, moe_down_<type>_<LPR>_<NR>_<SG> -------------------------------
#define GU(F, L, R, S) template [[host_name("moe_gu_" #F "_" #L "_" #R "_" #S)]] kernel decltype(moe_gu<F, L, R, S>) moe_gu<F, L, R, S>;
#define DN(F, L, R, S) template [[host_name("moe_down_" #F "_" #L "_" #R "_" #S)]] kernel decltype(moe_down<F, L, R, S>) moe_down<F, L, R, S>;
#define GU_SET(F) GU(F, 32, 1, 4) GU(F, 32, 2, 4) GU(F, 32, 4, 4) GU(F, 32, 2, 2) GU(F, 32, 2, 8) GU(F, 32, 1, 8) \
                  GU(F, 16, 1, 4) GU(F, 16, 2, 4) GU(F, 8, 1, 4) GU(F, 8, 2, 4) GU(F, 8, 1, 2)
#define DN_SET(F) DN(F, 8, 1, 4) DN(F, 8, 2, 4) DN(F, 8, 4, 4) DN(F, 8, 1, 8) DN(F, 8, 2, 8) DN(F, 4, 1, 4) DN(F, 4, 2, 4) \
                  DN(F, 4, 4, 4) DN(F, 16, 1, 4) DN(F, 16, 2, 4) DN(F, 32, 1, 4) DN(F, 2, 2, 4)
GU_SET(T_Q2_0)
GU_SET(T_IQ3_XXS)
GU_SET(T_IQ2_XXS)
GU_SET(T_IQ2_XS)
GU_SET(T_IQ2_S)
GU_SET(T_IQ3_S)
GU_SET(T_IQ4_XS)   // the IQ3_S pack's gate/up in one layer
GU_SET(T_Q4_K) GU_SET(T_Q5_K) GU_SET(T_Q6_K) GU_SET(T_Q8_0)   // the UD-Q4/5/6_K_XL packs
DN_SET(T_Q5_1) DN_SET(T_Q8_0)
DN_SET(T_Q2_0)
DN_SET(T_Q2_0R)
DN_SET(T_IQ4_NL)
// Gyro (agentionai's trellis experts): TQK6 / TQ2_T gate/up, TQK7 / TQ2_T down (the CUDA engine's STRATA_GU_FMTS / D_FMTS)
GU_SET(T_TQK6) GU_SET(T_TQ2T)
DN_SET(T_TQK7) DN_SET(T_TQ2T)
