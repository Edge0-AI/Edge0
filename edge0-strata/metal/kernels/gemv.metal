// metal/kernels/gemv.metal - y[t][r] = W[r] . x[t] for a window of 1..8 tokens (dense projections, the shared
// expert, the LM head), any format of quant.metal/units.metal.
//
// Shape of a threadgroup: SG simdgroups; KS of them split the row's units (KS = 1: no split), the other SG/KS cover
// different rows.  In a simdgroup, LPR lanes share a row and 32/LPR rows run side by side; each lane holds NR rows.
// The weights of a unit are decoded once and applied to every token.  Partial sums of a K split are added in a
// fixed order (deterministic).

struct GemvArgs {
    uint n_in;
    uint n_out;
    uint n_tok;
    uint row_bytes;
    uint x_stride;   // floats between tokens' x rows
    uint y_stride;   // floats between tokens' y rows
};

constant int NTOK [[function_constant(0)]];   // tokens of the window (1..8): the token loops are specialised
// a bench probe (function constant 9): 1 = tokens past the first reuse token 0's x (no load; wrong values),
// 2 = tokens past the first skip the combine (wrong values)
constant int GEMV_PROBE [[function_constant(9)]];
constant int gemv_probe = is_function_constant_defined(GEMV_PROBE) ? GEMV_PROBE : 0;

template <int F, int LPR, int NR, int SG, int KS>
kernel void gemv(device const uchar* W [[buffer(0)]], device const float* xp [[buffer(1)]],
                 device const float* s16 [[buffer(2)]], device float* y [[buffer(3)]], constant GemvArgs& a [[buffer(4)]],
                 uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                 ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int UNIT = fmt<F>::UNIT, RS = SG / KS, RG = 32 / LPR;
    threadgroup uint tab[fmt<F>::TAB];
    threadgroup float red[KS > 1 ? KS * RS * RG * NR * 8 : 1];
    load_tab<F>(tab, tid, SG * 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int ks = sg / RS, rs = sg % RS;
    const int li = lane % LPR;
    const int lrow0 = (rs * RG + lane / LPR) * NR;
    const int row0 = int(tg.x) * RS * RG * NR + lrow0;
    const int U = int(a.n_in) / UNIT;
    const int u0 = ks * U / KS, u1 = (ks + 1) * U / KS;
    float acc[NR][8];
    for (int r = 0; r < NR; ++r)
        for (int t = 0; t < NTOK; ++t) acc[r][t] = 0.0f;
    device const uchar* rp[NR];
    for (int r = 0; r < NR; ++r) rp[r] = W + (ulong)min(row0 + r, int(a.n_out) - 1) * a.row_bytes;
    const int s16n = int(a.n_in) / 16;
    if (gemv_probe == 3) {
        // rows outer, tokens inner: one row's decoded unit in registers at a time (x re-read per row from L1)
        for (int u = u0 + li; u < u1; u += LPR)
            for (int r = 0; r < NR; ++r) {
                float q[UNIT];
                float4 sc;
                Dec<F>::run(rp[r], u, q, sc, tab);
                for (int t = 0; t < NTOK; ++t) {
                    float xs[UNIT], xsum[2];
                    X<F>::load(xp + t * int(a.x_stride), s16 + t * s16n, u, xs, xsum);
                    acc[r][t] += unit_combine<UNIT>(q, sc, xs, xsum);
                }
            }
    } else
    for (int u = u0 + li; u < u1; u += LPR) {
        float q[NR][UNIT];
        float4 sc[NR];
        for (int r = 0; r < NR; ++r) Dec<F>::run(rp[r], u, q[r], sc[r], tab);
        float xs[UNIT], xsum[2];
        for (int t = 0; t < NTOK; ++t) {
            if (gemv_probe != 1 || t == 0) X<F>::load(xp + t * int(a.x_stride), s16 + t * s16n, u, xs, xsum);
            if (gemv_probe != 2 || t == 0)
                for (int r = 0; r < NR; ++r) acc[r][t] += unit_combine<UNIT>(q[r], sc[r], xs, xsum);
        }
    }
    constexpr int RPT = RS * RG * NR;
    for (int r = 0; r < NR; ++r)
        for (int t = 0; t < NTOK; ++t) {
            const float v = lane_sum<LPR>(acc[r][t]);
            if (KS == 1) {
                if (li == 0 && row0 + r < int(a.n_out)) y[t * int(a.y_stride) + row0 + r] = v;
            } else if (li == 0) {
                red[(ks * RPT + lrow0 + r) * 8 + t] = v;
            }
        }
    if (KS > 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ks == 0 && li == 0)
            for (int r = 0; r < NR; ++r)
                for (int t = 0; t < NTOK; ++t) {
                    float v = 0.0f;
                    for (int k = 0; k < KS; ++k) v += red[(k * RPT + lrow0 + r) * 8 + t];
                    if (row0 + r < int(a.n_out)) y[t * int(a.y_stride) + row0 + r] = v;
                }
    }
}

#define GV(F, L, R, S, K) template [[host_name("gemv_" #F "_" #L "_" #R "_" #S "_" #K)]] kernel decltype(gemv<F, L, R, S, K>) gemv<F, L, R, S, K>;
#define GV_SET(F) GV(F, 32, 1, 4, 1) GV(F, 32, 2, 4, 1) GV(F, 16, 1, 4, 1) GV(F, 16, 2, 4, 1) GV(F, 8, 1, 4, 1) \
                  GV(F, 8, 2, 4, 1) GV(F, 4, 2, 4, 1) GV(F, 4, 1, 4, 1) GV(F, 32, 1, 4, 4) GV(F, 32, 2, 4, 2) \
                  GV(F, 32, 1, 8, 8) GV(F, 16, 1, 8, 4) GV(F, 32, 1, 8, 2) GV(F, 8, 1, 8, 1)                   GV(F, 1, 1, 4, 1) GV(F, 1, 2, 4, 1) GV(F, 2, 1, 4, 1) GV(F, 2, 2, 4, 1) GV(F, 1, 1, 2, 1) GV(F, 4, 4, 4, 1)
#ifndef GEMV_TYPES
#define GEMV_TYPES GV_SET(T_F32) GV_SET(T_BF16) GV_SET(T_Q8_0) GV_SET(T_Q4_0) GV_SET(T_Q5_0) GV_SET(T_IQ4_NL) GV_SET(T_IQ4_XS) \
                   GV_SET(T_Q2_0) GV_SET(T_IQ3_S) GV_SET(T_Q3_K) GV_SET(T_Q4_K) GV_SET(T_Q5_K) GV_SET(T_Q6_K)
#endif
GEMV_TYPES

// The shared expert's gate and up in one dispatch: h[t][r] = silu(Wg[r] . x[t]) * (Wu[r] . x[t]), each with its own
// format (and so its own prepared activation kind); h is written raw (its consumer reads raw activations).
// KS = 1; tiles as gemv.  Buffers: Wg, xg (Wg's kind), Wu, xu (Wu's kind), s16, h, args (row_bytes = Wg's), rb_up.
template <int F1, int F2, int LPR, int SG>
kernel void gemv_pair(device const uchar* Wg [[buffer(0)]], device const float* xg [[buffer(1)]],
                      device const uchar* Wu [[buffer(2)]], device const float* xu [[buffer(3)]],
                      device const float* s16 [[buffer(4)]], device float* h [[buffer(5)]],
                      constant GemvArgs& a [[buffer(6)]], constant uint& rb_up [[buffer(7)]],
                      uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int U1 = fmt<F1>::UNIT, U2 = fmt<F2>::UNIT, RG = 32 / LPR;
    constexpr int TB = fmt<F1>::TAB > fmt<F2>::TAB ? fmt<F1>::TAB : fmt<F2>::TAB;
    threadgroup uint tab1[fmt<F1>::TAB];
    threadgroup uint tab2[fmt<F2>::TAB];
    (void)TB;
    load_tab<F1>(tab1, tid, SG * 32);
    load_tab<F2>(tab2, tid, SG * 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int li = lane % LPR;
    const int row = (int(tg.x) * SG + sg) * RG + lane / LPR;
    const int rr = min(row, int(a.n_out) - 1);
    device const uchar* rg = Wg + ulong(rr) * a.row_bytes;
    device const uchar* ru = Wu + ulong(rr) * rb_up;
    const int s16n = int(a.n_in) / 16;
    float ag[8], au[8];
    for (int t = 0; t < 8; ++t) ag[t] = au[t] = 0.0f;
    for (int u = li; u < int(a.n_in) / U1; u += LPR) {
        float q[U1];
        float4 sc;
        Dec<F1>::run(rg, u, q, sc, tab1);
        for (int t = 0; t < NTOK; ++t) {
            float xs[U1], xsum[2];
            X<F1>::load(xg + t * int(a.x_stride), s16 + t * s16n, u, xs, xsum);
            ag[t] += unit_combine<U1>(q, sc, xs, xsum);
        }
    }
    for (int u = li; u < int(a.n_in) / U2; u += LPR) {
        float q[U2];
        float4 sc;
        Dec<F2>::run(ru, u, q, sc, tab2);
        for (int t = 0; t < NTOK; ++t) {
            float xs[U2], xsum[2];
            X<F2>::load(xu + t * int(a.x_stride), s16 + t * s16n, u, xs, xsum);
            au[t] += unit_combine<U2>(q, sc, xs, xsum);
        }
    }
    for (int t = 0; t < NTOK; ++t) {
        const float gv = lane_sum<LPR>(ag[t]), uv = lane_sum<LPR>(au[t]);
        if (li == 0 && row < int(a.n_out)) h[t * int(a.y_stride) + row] = gv / (1.0f + exp(-gv)) * uv;
    }
}

#define GP_(F1, F2) template [[host_name("gemv_pair_" #F1 "_" #F2)]] kernel decltype(gemv_pair<F1, F2, 32, 4>) gemv_pair<F1, F2, 32, 4>;
#define GP(F1, F2) GP_(F1, F2)
#define GP_ROW(F1) GP(F1, T_IQ3_S) GP(F1, T_IQ4_XS) GP(F1, T_Q2_0) GP(F1, T_Q3_K) GP(F1, T_Q4_K) GP(F1, T_Q5_K) GP(F1, T_Q6_K)
GP_ROW(T_IQ3_S) GP_ROW(T_IQ4_XS) GP_ROW(T_Q2_0) GP_ROW(T_Q3_K) GP_ROW(T_Q4_K) GP_ROW(T_Q5_K) GP_ROW(T_Q6_K)
// Q2_0 with raw activations (the shared expert's down reading the pair's raw h)
GV_SET(T_Q2_0R)

// gemv_cat: two matrices of the same format and n_in over the same x in one dispatch - rows [0, n_split) are W's
// (into y), rows [n_split, n_out) are W2's (into y2, from its row 0).  Each row's arithmetic is gemv's (KS = 1), so the
// outputs are bitwise the two separate GEMVs'.  (GDN's ssm_alpha + ssm_beta: 48 + 48 rows.)
// Buffers: W, xp, s16, y, args (n_out = both), W2, y2, n_split.
template <int F, int LPR, int NR, int SG>
kernel void gemv_cat(device const uchar* W [[buffer(0)]], device const float* xp [[buffer(1)]],
                     device const float* s16 [[buffer(2)]], device float* y [[buffer(3)]], constant GemvArgs& a [[buffer(4)]],
                     device const uchar* W2 [[buffer(5)]], device float* y2 [[buffer(6)]], constant uint& n_split [[buffer(7)]],
                     uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                     ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int UNIT = fmt<F>::UNIT, RG = 32 / LPR;
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, SG * 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int li = lane % LPR;
    const int row0 = int(tg.x) * SG * RG * NR + (sg * RG + lane / LPR) * NR;
    const int U = int(a.n_in) / UNIT, sp = int(n_split);
    float acc[NR][8];
    for (int r = 0; r < NR; ++r)
        for (int t = 0; t < NTOK; ++t) acc[r][t] = 0.0f;
    device const uchar* rp[NR];
    for (int r = 0; r < NR; ++r) {
        const int row = min(row0 + r, int(a.n_out) - 1);
        rp[r] = row < sp ? W + (ulong)row * a.row_bytes : W2 + (ulong)(row - sp) * a.row_bytes;
    }
    const int s16n = int(a.n_in) / 16;
    for (int u = li; u < U; u += LPR) {
        float q[NR][UNIT];
        float4 sc[NR];
        for (int r = 0; r < NR; ++r) Dec<F>::run(rp[r], u, q[r], sc[r], tab);
        for (int t = 0; t < NTOK; ++t) {
            float xs[UNIT], xsum[2];
            X<F>::load(xp + t * int(a.x_stride), s16 + t * s16n, u, xs, xsum);
            for (int r = 0; r < NR; ++r) acc[r][t] += unit_combine<UNIT>(q[r], sc[r], xs, xsum);
        }
    }
    for (int r = 0; r < NR; ++r)
        for (int t = 0; t < NTOK; ++t) {
            const float v = lane_sum<LPR>(acc[r][t]);
            const int row = row0 + r;
            if (li == 0 && row < int(a.n_out)) {
                if (row < sp) y[t * int(a.y_stride) + row] = v;
                else y2[t * int(a.y_stride) + row - sp] = v;
            }
        }
}
#define GVC_(F, L, R, S) template [[host_name("gemvc_" #F "_" #L "_" #R "_" #S)]] kernel decltype(gemv_cat<F, L, R, S>) gemv_cat<F, L, R, S>;
#define GVC(F, L, R, S) GVC_(F, L, R, S)
GVC(T_BF16, 32, 1, 4) GVC(T_BF16, 32, 1, 8) GVC(T_BF16, 16, 1, 4) GVC(T_BF16, 32, 2, 4)
GVC(T_F32, 32, 1, 4) GVC(T_F32, 32, 1, 8) GVC(T_F32, 16, 1, 4) GVC(T_F32, 32, 2, 4)

// gemv_x: gemv (KS = 1) with the window's prepared x (the format's plane, NTOK rows, and its per-16 sums) staged in
// threadgroup memory once per threadgroup, so the per-unit x reads of every token come from threadgroup memory
// instead of L1 (on the M4 the x traffic through L1 - 4 B a weight a token, shared by the rows in a simdgroup - is
// what a window token costs).  Same tile meaning and the same arithmetic as gemv_<F>_<LPR>_<NR>_<SG>_1: bitwise.
// Dynamic threadgroup memory 0: 4 * NTOK * (n_in + n_in / 16) bytes.
template <int F, int LPR, int NR, int SG>
kernel void gemv_x(device const uchar* W [[buffer(0)]], device const float* xp [[buffer(1)]],
                   device const float* s16 [[buffer(2)]], device float* y [[buffer(3)]], constant GemvArgs& a [[buffer(4)]],
                   threadgroup float* xt [[threadgroup(0)]],
                   uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                   ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int UNIT = fmt<F>::UNIT, RG = 32 / LPR;
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, SG * 32);
    const int n = int(a.n_in), s16n = n / 16;
    threadgroup float* st = xt + NTOK * n;
    for (int t = 0; t < NTOK; ++t) {
        device const float4* src = (device const float4*)(xp + t * int(a.x_stride));
        threadgroup float4* dst = (threadgroup float4*)(xt + t * n);
        for (int i = tid; i < n / 4; i += SG * 32) dst[i] = src[i];
        for (int i = tid; i < s16n; i += SG * 32) st[t * s16n + i] = s16[t * s16n + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int li = lane % LPR;
    const int row0 = int(tg.x) * SG * RG * NR + (sg * RG + lane / LPR) * NR;
    const int U = n / UNIT;
    float acc[NR][8];
    for (int r = 0; r < NR; ++r)
        for (int t = 0; t < NTOK; ++t) acc[r][t] = 0.0f;
    device const uchar* rp[NR];
    for (int r = 0; r < NR; ++r) rp[r] = W + (ulong)min(row0 + r, int(a.n_out) - 1) * a.row_bytes;
    for (int u = li; u < U; u += LPR) {
        float q[NR][UNIT];
        float4 sc[NR];
        for (int r = 0; r < NR; ++r) Dec<F>::run(rp[r], u, q[r], sc[r], tab);
        for (int t = 0; t < NTOK; ++t) {
            float xs[UNIT], xsum[2];
            X<F>::load((threadgroup const float*)(xt + t * n), (threadgroup const float*)(st + t * s16n), u, xs, xsum);
            for (int r = 0; r < NR; ++r) acc[r][t] += unit_combine<UNIT>(q[r], sc[r], xs, xsum);
        }
    }
    for (int r = 0; r < NR; ++r)
        for (int t = 0; t < NTOK; ++t) {
            const float v = lane_sum<LPR>(acc[r][t]);
            if (li == 0 && row0 + r < int(a.n_out)) y[t * int(a.y_stride) + row0 + r] = v;
        }
}
#define GVX(F, L, R, S) template [[host_name("gemvx_" #F "_" #L "_" #R "_" #S)]] kernel decltype(gemv_x<F, L, R, S>) gemv_x<F, L, R, S>;
#define GVX_SET(F) GVX(F, 32, 1, 4) GVX(F, 32, 2, 4) GVX(F, 16, 1, 4) GVX(F, 16, 2, 4) GVX(F, 8, 1, 4) GVX(F, 8, 2, 4) \
                   GVX(F, 4, 2, 4) GVX(F, 16, 2, 8) GVX(F, 8, 2, 8) GVX(F, 16, 4, 4) GVX(F, 32, 2, 8) GVX(F, 2, 2, 4)
GVX_SET(T_BF16) GVX_SET(T_Q8_0) GVX_SET(T_Q4_0) GVX_SET(T_Q5_0) GVX_SET(T_IQ4_NL) GVX_SET(T_IQ4_XS) GVX_SET(T_Q2_0)
GVX_SET(T_IQ3_S) GVX_SET(T_Q3_K) GVX_SET(T_Q4_K) GVX_SET(T_Q5_K) GVX_SET(T_Q6_K)
