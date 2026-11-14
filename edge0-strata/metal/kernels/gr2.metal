// metal/kernels/gr2.metal - the hyper-connection read in TWO dispatches (after quant/units/gr.metal), for BF16 HC
// weights (the exact path) or a Q8_0 copy of them (the STRATA_HC_Q8 option):
//
//   gr_down3  grid (row blocks, hc streams).  The RMSNorm is per stream, so a threadgroup owns one stream c: it forms
//             R'_c = R_c + bo * 2 sigmoid(inj_prev_c / hc) (10 KB a token), its sum of squares and rs_c itself,
//             stages xn_c = (R'_c * w_norm_c) * rs_c in threadgroup memory once, and every simdgroup dots its rows'
//             stream-c quarter with it: D[t][row][c].  Row block 0 also writes R'_c to R_out (ping-pong, never in
//             place) and rs_c.  Rows 0..hc_lr-1 are w_down, hc_lr.. the inject rows.
//   gr_up3    lo[k] = silu(sum_c D[t][k][c] / hc) formed in threadgroup memory, then as gr_up with xn = R' * w_norm
//             * rs on the fly, writing the prepared activations.
//   inj_prev of the NEXT read = sum_c D[t][hc_lr + c'][c] of this one (gr_down3 sums it, in c order).
//
// Per element the arithmetic is gr_norm's (fma(bo, g, R), then (R' * w) * rs); the K order differs (per stream, then
// the 4 streams in order).

#ifndef P_TOKENT
#define P_TOKENT 532   // moe.metal's plan layout
#endif

struct GrArgs3 {
    uint n_embd;
    uint hc;
    uint hc_lr;
    uint n_tok;      // the prepared planes' row count (the plane stride)
    uint apply;
    float eps;
    uint rb_down;    // bytes per row of w_down / w_inject
    uint rb_up;      // bytes per row of w_up
    uint comb;       // 1: bo is not given; it is the MoE combine (moe_combine_sh's arithmetic) of buffers 10..13
    uint topk;
};

// bo[t][d], given or the MoE combine: sum_k ye[entry(t, k)][d] (k in order) + sigmoid(sg[t]) * ysh[t][d]
static inline float bo_at(device const float* bo, device const int* plan, device const float* ye,
                          device const float* ysh, device const float* sg, constant GrArgs3& a, int t, int d) {
    const int n = int(a.n_embd);
    if (!a.comb) return bo[t * n + d];
    float s = 0.0f;
    for (uint j = 0; j < a.topk; ++j) s += ye[plan[P_TOKENT + t * int(a.topk) + j] * n + d];
    const float gs = 1.0f / (1.0f + exp(-sg[t]));
    return fma(gs, ysh[t * n + d], s);
}

template <int F, int TT, int RPS>
kernel void gr_down3(device const uchar* w_down [[buffer(0)]], device const uchar* w_inject [[buffer(1)]],
                     device const float* Rin [[buffer(2)]], device float* Rout [[buffer(3)]],
                     device const float* bo [[buffer(4)]], device const float* Dprev [[buffer(5)]],
                     device const float* wn [[buffer(6)]], device float* D [[buffer(7)]],
                     device float* rs_out [[buffer(8)]], constant GrArgs3& a [[buffer(9)]],
                     device const int* plan [[buffer(10)]], device const float* ye [[buffer(11)]],
                     device const float* ysh [[buffer(12)]], device const float* sgv [[buffer(13)]],
                     uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                     ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int UNIT = fmt<F>::UNIT;
    threadgroup float xs_tg[TT][2560];
    threadgroup float part[8][TT];
    threadgroup float g_tg[TT], rs_tg[TT];
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, 256);
    const int n = int(a.n_embd), hc = int(a.hc), NIN = hc * n, c = int(tg.y);
    const int nrows = int(a.hc_lr) + hc;
    for (int t0 = 0; t0 < GR_NTOK; t0 += TT) {
        const int T = min(TT, GR_NTOK - t0);
        if (tid < T) {
            float g = 0.0f;
            if (a.apply) {
                device const float* dp = Dprev + ((t0 + tid) * nrows + int(a.hc_lr) + c) * hc;
                const float ip = ((dp[0] + dp[1]) + dp[2]) + dp[3];
                g = 2.0f / (1.0f + exp(-ip / float(hc)));
            }
            g_tg[tid] = g;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int t = 0; t < T; ++t) {
            device const float* rt = Rin + (t0 + t) * NIN + c * n;
            float ss = 0.0f;
            for (int d = tid; d < n; d += 256) {
                float v = rt[d];
                if (a.apply) v = fma(bo_at(bo, plan, ye, ysh, sgv, a, t0 + t, d), g_tg[t], v);
                xs_tg[t][d] = v;
                ss = fma(v, v, ss);
                if (tg.x == 0) Rout[(t0 + t) * NIN + c * n + d] = v;
            }
            ss = simd_sum(ss);
            if (lane == 0) part[sg][t] = ss;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < T) {
            float tot = 0.0f;
            for (int k = 0; k < 8; ++k) tot += part[k][tid];
            const float rsv = rsqrt(tot / float(n) + a.eps);
            rs_tg[tid] = rsv;
            if (tg.x == 0) rs_out[(t0 + tid) * hc + c] = rsv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int t = 0; t < T; ++t)
            for (int d = tid; d < n; d += 256) xs_tg[t][d] = xs_tg[t][d] * wn[c * n + d] * rs_tg[t];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // the rows: simdgroup sg holds RPS rows; lanes split the stream's units
        int rows[RPS];
        device const uchar* wr[RPS];
        for (int k = 0; k < RPS; ++k) {
            rows[k] = (int(tg.x) * 8 + sg) * RPS + k;
            const int rr = min(rows[k], nrows - 1);
            wr[k] = rr < int(a.hc_lr) ? w_down + ulong(rr) * a.rb_down : w_inject + ulong(rr - int(a.hc_lr)) * a.rb_down;
        }
        float acc[RPS][TT];
        for (int k = 0; k < RPS; ++k)
            for (int t = 0; t < TT; ++t) acc[k][t] = 0.0f;
        const int U = n / UNIT, u0 = c * U;
        for (int u = lane; u < U; u += 32) {
            float q[RPS][UNIT];
            float4 sc[RPS];
            for (int k = 0; k < RPS; ++k) Dec<F>::run(wr[k], u0 + u, q[k], sc[k], tab);
            for (int t = 0; t < T; ++t) {
                float xs[UNIT], xsum[2] = {0.0f, 0.0f};
                for (int e = 0; e < UNIT; ++e) xs[e] = xs_tg[t][u * UNIT + e];
                for (int k = 0; k < RPS; ++k) acc[k][t] += unit_combine<UNIT>(q[k], sc[k], xs, xsum);
            }
        }
        for (int k = 0; k < RPS; ++k)
            for (int t = 0; t < T; ++t) {
                const float v = simd_sum(acc[k][t]);
                if (lane == 0 && rows[k] < nrows) D[((t0 + t) * nrows + rows[k]) * hc + c] = v;
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// grid (n_embd / 16), 256 threads.
template <int F>
kernel void gr_up3(device const uchar* w_up [[buffer(0)]], device const float* D [[buffer(1)]],
                   device const float* Rn [[buffer(2)]], device float* xp [[buffer(3)]],
                   constant GrArgs3& a [[buffer(4)]], device const float* wn [[buffer(5)]],
                   device const float* rs [[buffer(6)]],
                   uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                   ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int UNIT = fmt<F>::UNIT;
    threadgroup float lo_tg[8][320];
    threadgroup float gate[8][4][16];
    threadgroup float msum[8][16];
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, 256);
    const int n = int(a.n_embd), hc = int(a.hc), lr = int(a.hc_lr), nrows = lr + hc;
    for (int e = tid; e < GR_NTOK * lr; e += 256) {
        const int t = e / lr, k = e - t * lr;
        device const float* dp = D + (t * nrows + k) * hc;
        const float z = (((dp[0] + dp[1]) + dp[2]) + dp[3]) / float(hc);
        lo_tg[t][k] = z / (1.0f + exp(-z));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int c = sg >> 1, dl = 8 * (sg & 1) + lane / 4, li = lane & 3;
    const int d = int(tg.x) * 16 + dl;
    device const uchar* w = w_up + ulong(c * n + d) * a.rb_up;
    const int U = lr / UNIT;
    float acc[8];
    for (int t = 0; t < 8; ++t) acc[t] = 0.0f;
    for (int u = li; u < U; u += 4) {
        float q[UNIT];
        float4 sc;
        Dec<F>::run(w, u, q, sc, tab);
        for (int t = 0; t < GR_NTOK; ++t) {
            float xs[UNIT], xsum[2] = {0.0f, 0.0f};
            for (int e = 0; e < UNIT; ++e) xs[e] = lo_tg[t][UNIT * u + e];
            acc[t] += unit_combine<UNIT>(q, sc, xs, xsum);
        }
    }
    for (int t = 0; t < GR_NTOK; ++t) {
        const float v = lane_sum<4>(acc[t]);
        if (li == 0) gate[t][c][dl] = 1.0f / (1.0f + exp(-v));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int t = tid / 16, j = tid & 15;
    if (t < GR_NTOK) {
        const int dd = int(tg.x) * 16 + j;
        float m = 0.0f;
        for (int cc = 0; cc < hc; ++cc) {
            const float xn = Rn[(t * hc + cc) * n + dd] * wn[cc * n + dd] * rs[t * hc + cc];
            m = fma(xn, gate[t][cc][j], m);
        }
        m = m / float(hc);
        const uint plane = a.n_tok * uint(n);
        device float* raw = xp + t * n + dd;
        const float q2s[8] = {1.0f, 0.25f, 0.0625f, 0.015625f, 0.00390625f, 0.0009765625f, 0.000244140625f,
                              0.00006103515625f};
        const float p4s[4] = {1.0f, 0.00390625f, 1.52587890625e-05f, 5.9604644775390625e-08f};
        raw[0] = m;
        raw[plane] = m * q2s[j & 7];
        raw[2 * plane] = m * p4s[j & 3];
        raw[3 * plane] = (j & 1) ? m * 0.00390625f : m;
        msum[t][j] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < GR_NTOK && j == 0) {
        float s0 = 0.0f, s1 = 0.0f;
        for (int k = 0; k < 16; k += 2) { s0 += msum[t][k]; s1 += msum[t][k + 1]; }
        xp[4 * a.n_tok * uint(n) + t * (n / 16) + tg.x] = s0 + s1;
    }
}

#define GD3(F, T, R) template [[host_name("gr_down3_" #F "_" #T "_" #R)]] kernel decltype(gr_down3<F, T, R>) gr_down3<F, T, R>;
GD3(30, 1, 1) GD3(30, 2, 1) GD3(30, 3, 1) GD3(30, 1, 2) GD3(30, 2, 2) GD3(30, 3, 2)
GD3(8, 1, 1) GD3(8, 2, 1) GD3(8, 3, 1) GD3(8, 1, 2) GD3(8, 2, 2) GD3(8, 3, 2)
template [[host_name("gr_up3_30")]] kernel decltype(gr_up3<T_BF16>) gr_up3<T_BF16>;
template [[host_name("gr_up3_8")]] kernel decltype(gr_up3<T_Q8_0>) gr_up3<T_Q8_0>;

// BF16 only: as gr_down3, but each lane loads its rows' weights (5 units of 16 per row and lane: 2560 = 160 units)
// into registers before the norm phase, so the weight stream is in flight while the threadgroup forms xn.
template <int TT, int RPS>
kernel void gr_down3b(device const uchar* w_down [[buffer(0)]], device const uchar* w_inject [[buffer(1)]],
                      device const float* Rin [[buffer(2)]], device float* Rout [[buffer(3)]],
                      device const float* bo [[buffer(4)]], device const float* Dprev [[buffer(5)]],
                      device const float* wn [[buffer(6)]], device float* D [[buffer(7)]],
                      device float* rs_out [[buffer(8)]], constant GrArgs3& a [[buffer(9)]],
                      device const int* plan [[buffer(10)]], device const float* ye [[buffer(11)]],
                      device const float* ysh [[buffer(12)]], device const float* sgv [[buffer(13)]],
                      uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                      ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float xs_tg[TT][2560];
    threadgroup float part[8][TT];
    threadgroup float g_tg[TT], rs_tg[TT];
    const int n = int(a.n_embd), hc = int(a.hc), NIN = hc * n, c = int(tg.y);
    const int nrows = int(a.hc_lr) + hc;
    int rows[RPS];
    uint4 wr[RPS][5][2];
    for (int k = 0; k < RPS; ++k) {
        rows[k] = (int(tg.x) * 8 + sg) * RPS + k;
        const int rr = min(rows[k], nrows - 1);
        device const uchar* w = rr < int(a.hc_lr) ? w_down + ulong(rr) * a.rb_down : w_inject + ulong(rr - int(a.hc_lr)) * a.rb_down;
        device const uint4* p = (device const uint4*)(w + 2 * (c * n));
        for (int i = 0; i < 5; ++i) {
            wr[k][i][0] = p[2 * (lane + 32 * i)];
            wr[k][i][1] = p[2 * (lane + 32 * i) + 1];
        }
    }
    for (int t0 = 0; t0 < GR_NTOK; t0 += TT) {
        const int T = min(TT, GR_NTOK - t0);
        if (tid < T) {
            float g = 0.0f;
            if (a.apply) {
                device const float* dp = Dprev + ((t0 + tid) * nrows + int(a.hc_lr) + c) * hc;
                const float ip = ((dp[0] + dp[1]) + dp[2]) + dp[3];
                g = 2.0f / (1.0f + exp(-ip / float(hc)));
            }
            g_tg[tid] = g;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int t = 0; t < T; ++t) {
            device const float* rt = Rin + (t0 + t) * NIN + c * n;
            float ss = 0.0f;
            for (int d = tid; d < n; d += 256) {
                float v = rt[d];
                if (a.apply) v = fma(bo_at(bo, plan, ye, ysh, sgv, a, t0 + t, d), g_tg[t], v);
                xs_tg[t][d] = v;
                ss = fma(v, v, ss);
                if (tg.x == 0) Rout[(t0 + t) * NIN + c * n + d] = v;
            }
            ss = simd_sum(ss);
            if (lane == 0) part[sg][t] = ss;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < T) {
            float tot = 0.0f;
            for (int k = 0; k < 8; ++k) tot += part[k][tid];
            const float rsv = rsqrt(tot / float(n) + a.eps);
            rs_tg[tid] = rsv;
            if (tg.x == 0) rs_out[(t0 + tid) * hc + c] = rsv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int t = 0; t < T; ++t)
            for (int d = tid; d < n; d += 256) xs_tg[t][d] = xs_tg[t][d] * wn[c * n + d] * rs_tg[t];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float acc[RPS][TT];
        for (int k = 0; k < RPS; ++k)
            for (int t = 0; t < TT; ++t) acc[k][t] = 0.0f;
        for (int i = 0; i < 5; ++i) {
            const int u = lane + 32 * i;
            for (int k = 0; k < RPS; ++k) {
                const uint v[8] = {wr[k][i][0].x, wr[k][i][0].y, wr[k][i][0].z, wr[k][i][0].w,
                                   wr[k][i][1].x, wr[k][i][1].y, wr[k][i][1].z, wr[k][i][1].w};
                float q[16];
                for (int e = 0; e < 8; ++e) {
                    q[2 * e] = as_type<float>(v[e] << 16);
                    q[2 * e + 1] = as_type<float>(v[e] & 0xFFFF0000u);
                }
                const float4 sc = float4(1.0f, 1.0f, 0.0f, 0.0f);
                for (int t = 0; t < T; ++t) {
                    float xs[16], xsum[2] = {0.0f, 0.0f};
                    for (int e = 0; e < 16; ++e) xs[e] = xs_tg[t][u * 16 + e];
                    acc[k][t] += unit_combine<16>(q, sc, xs, xsum);
                }
            }
        }
        for (int k = 0; k < RPS; ++k)
            for (int t = 0; t < T; ++t) {
                const float v = simd_sum(acc[k][t]);
                if (lane == 0 && rows[k] < nrows) D[((t0 + t) * nrows + rows[k]) * hc + c] = v;
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
#define GD3B(T, R) template [[host_name("gr_down3b_" #T "_" #R)]] kernel decltype(gr_down3b<T, R>) gr_down3b<T, R>;
GD3B(1, 1) GD3B(2, 1) GD3B(3, 1) GD3B(1, 2) GD3B(2, 2) GD3B(3, 2)

// gr_norm_comb: gr_norm (gr.metal) with the sublayer output bo formed on the fly as the MoE combine - bo[t][d] =
// sum_k ye[entry(t, k)][d] (k in order) + sigmoid(sg[t]) * ysh[t][d], moe_combine_sh's arithmetic - so the combine's
// dispatch goes away; R', xn bitwise gr_norm's after moe_combine_sh.  One threadgroup per (stream c, token t).
kernel void gr_norm_comb(device float* R [[buffer(0)]], device const float* ye [[buffer(1)]],
                         device const float* inj [[buffer(2)]], device const float* w_norm [[buffer(3)]],
                         device float* xn [[buffer(4)]], constant GrArgs& a [[buffer(5)]],
                         device const int* plan [[buffer(6)]], device const float* ysh [[buffer(7)]],
                         device const float* sgv [[buffer(8)]], constant uint& topk [[buffer(9)]],
                         uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    const uint c = tg.x, t = tg.y, n = a.n_embd;
    device float* r = R + (t * a.hc + c) * n;
    const float g = a.apply ? 2.0f / (1.0f + exp(-inj[t * a.hc + c] / float(a.hc))) : 0.0f;
    const float gs = 1.0f / (1.0f + exp(-sgv[t]));
    float ss = 0.0f;
    for (uint d = tid; d < n; d += 256) {
        float v = r[d];
        if (a.apply) {
            float s = 0.0f;
            for (uint j = 0; j < topk; ++j) s += ye[plan[P_TOKENT + t * topk + j] * int(n) + d];
            const float bo = fma(gs, ysh[t * n + d], s);
            v = fma(bo, g, v);
            r[d] = v;
        }
        ss = fma(v, v, ss);
    }
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += part[i];
    const float rs = rsqrt(tot / float(n) + a.eps);
    device float* o = xn + (t * a.hc + c) * n;
    device const float* w = w_norm + c * n;
    for (uint d = tid; d < n; d += 256) o[d] = r[d] * w[d] * rs;
}
