// metal/kernels/gdn.metal - one GDN (gated delta-net) layer's recurrence for a window of tokens, fused end to end:
//
//   conv (4 taps, history [channels][3] oldest first) -> SiLU, on q (16 heads), k (16), v (48 heads) of 128
//   q, k: L2 norm as native_gdn_l2_norm (rsqrt(sum/128 + eps/128), then x 1/sqrt(128), two roundings)
//   gate = exp(softplus(alpha + dt) * a)  (softplus = v > 20 ? v : log1p(exp(v))),  beta = sigmoid(beta)
//   kv_j = sum_i S[i][j] k[i];  delta_j = (v_j - gate * kv_j) * beta;  S = gate * S + k[i] delta_j
//   o_j = (sum_i S[i][j] q[i]) / sqrt(128);  y_j = o_j * rsqrt(mean(o^2) + eps) * gamma_j * sigmoid(z_j)
//
// State layout as Strata's: S[(i * h_v + h) * 128 + j].  Value head h reads q/k head h % h_k.
// One threadgroup per value head, 512 threads = 16 simdgroups: simdgroup s holds columns j = 32 (s % 4) + lane and
// rows i = 32 (s / 4) .. + 32 of the head's 128 x 128 state in registers.  GDN_NTOK tokens run in order; with
// `commit` the state and the conv history are written back after the last one (a verify window runs without commit,
// the fold of m accepted tokens is the same kernel with GDN_NTOK = m and commit).  The conv history is read from
// `hist` and the committed one written to `hist_out` (ping-pong: the q/k channels are shared by three threadgroups).

constant int GDN_NTOK [[function_constant(2)]];
// GDN_ROWS (function constant 31, with GDN_NTOK = 1): the grid is (h_v, rows) and row y is a DIFFERENT sequence (a batch slot) with its own
// state and conv history; rinfo as in qsa.metal (header [5] state stride, [6] history stride per slot and parity; row y at [8 + 4 y] =
// {slot, cell, history parity, 0}).  hist / hist_out are then one buffer (hist), the parities side by side per slot.  The inputs and the
// output are row y of the window's arrays.  Off: exactly the plain kernel.
constant bool GDN_ROWS_C [[function_constant(31)]];
constant bool gdn_rows = is_function_constant_defined(GDN_ROWS_C) && GDN_ROWS_C;

struct GdnArgs {
    uint h_k;        // 16
    uint h_v;        // 48
    uint channels;   // 10240 = (2 h_k + h_v) * 128
    uint commit;     // 0: verify (no write); 1: commit after the last token; 0x100 | k (1 <= k <= GDN_NTOK): commit
                     // the state and the conv history after the first k tokens, in place, while every token's y is
                     // still computed (a window whose first k rows are kept) - bitwise the commit of a k-token step
    float eps;       // 1e-6
    uint plane;      // 0: y only; else y is the raw plane of prepared activations (xprep layout, n = h_v * 128,
                     // `plane` floats between planes) and the other planes and the per-16 sums are written too
};

// The kernel is launched with IB * 128 threads (512 for gdn_step, 1024 for gdn_step8).  Without the attribute the compiler
// is free to use as many registers as the 128-row state needs and a pipeline can end up with
// maxTotalThreadsPerThreadgroup below the launch size: an M2 Max gave 448 for the GDN_NTOK >= 2 specializations, and
// Metal then skips the dispatch without an error (#2323).  The attribute makes the compiler fit the launch size.
template <int IB>
[[max_total_threads_per_threadgroup(IB * 128)]]
kernel void gdn_step_t(device float* state [[buffer(0)]], device float* hist [[buffer(1)]],
                     device const float* qkv [[buffer(2)]], device const float* alpha [[buffer(3)]],
                     device const float* beta [[buffer(4)]], device const float* z [[buffer(5)]],
                     device const float* conv_w [[buffer(6)]], device const float* dt [[buffer(7)]],
                     device const float* ssm_a [[buffer(8)]], device const float* gamma [[buffer(9)]],
                     device float* y [[buffer(10)]], constant GdnArgs& a [[buffer(11)]],
                     device float* hist_out [[buffer(12)]],
                     device const uint* rinfo [[buffer(13), function_constant(gdn_rows)]],
                     uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                     ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int S = 128;
    threadgroup float qs[S], ks[S], vs[S];
    constexpr int RPT = S / IB;   // state rows per thread
    threadgroup float red[IB][S];
    threadgroup float part[32];
    threadgroup float gb[2];
    threadgroup float ys[S];
    const int h = int(tg.x), hk = h % int(a.h_k), hv = int(a.h_v);
    const int cb = sg & 3, ib = sg >> 2;
    const int j = 32 * cb + lane, i0 = RPT * ib;
    const float scale = 1.0f / sqrt(float(S));
    const int row0 = gdn_rows ? int(tg.y) : 0;
    if (gdn_rows) {
        const ulong sl = rinfo[8 + 4 * row0], par = rinfo[8 + 4 * row0 + 2];
        state += sl * rinfo[5];
        hist_out = hist + (sl * 2 + (1 - par)) * rinfo[6];
        hist += (sl * 2 + par) * rinfo[6];
    }

    // the channel this thread convolves: 0..127 q of head hk, 128..255 k of head hk, 256..383 v of head h
    int ch = -1;
    if (tid < 128) ch = hk * S + tid;
    else if (tid < 256) ch = int(a.h_k) * S + hk * S + (tid - 128);
    else if (tid < 384) ch = 2 * int(a.h_k) * S + h * S + (tid - 256);
    float hst[3] = {0.0f, 0.0f, 0.0f}, cw[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (ch >= 0) {
        for (int k = 0; k < 3; ++k) hst[k] = hist[ch * 3 + k];
        for (int k = 0; k < 4; ++k) cw[k] = conv_w[ch * 4 + k];
    }
    float st[RPT];
    for (int r = 0; r < RPT; ++r) st[r] = state[((i0 + r) * hv + h) * S + j];

    const int commit_at = a.commit == 1u ? GDN_NTOK : (a.commit >> 8) ? int(a.commit & 0xFFu) : 0;
    for (int t = 0; t < GDN_NTOK; ++t) {
        // conv + SiLU
        const int tr = t + row0;
        float cv = 0.0f;
        if (ch >= 0) {
            const float in = qkv[tr * int(a.channels) + ch];
            const float vals[4] = {hst[0], hst[1], hst[2], in};
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) s = fma(vals[k], cw[k], s);
            s = s + 0.0f;
            cv = s / (1.0f + exp(-s));
            hst[0] = hst[1];
            hst[1] = hst[2];
            hst[2] = in;
        }
        // L2 norms of q (threads 0..127 = simdgroups 0..3) and k (128..255 = 4..7)
        const float ssq = simd_sum(cv * cv);
        if (lane == 0) part[sg] = ssq;
        if (tid == 0) {
            const float v = alpha[tr * hv + h] + dt[h];
            // log1p(e^v) (Kahan: log(u) * e / (u - 1), u = 1 + e; exact where 1 + e rounds to 1)
            const float e = exp(v), u = 1.0f + e;
            const float sp = v > 20.0f ? v : (u == 1.0f ? e : log(u) * e / (u - 1.0f));
            gb[0] = exp(sp * ssm_a[h]);
            gb[1] = 1.0f / (1.0f + exp(-beta[tr * hv + h]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 256) {
            const int base = tid < 128 ? 0 : 4;
            const float tot = (part[base] + part[base + 1]) + (part[base + 2] + part[base + 3]);
            const float rs = rsqrt(tot / float(S) + a.eps / float(S));
            const float nv = (rs * cv) * scale;
            if (tid < 128) qs[tid] = nv;
            else ks[tid - 128] = nv;
        } else if (tid < 384) {
            vs[tid - 256] = cv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float g = gb[0], be = gb[1];
        // kv_j
        float kv = 0.0f;
        for (int r = 0; r < RPT; ++r) kv = fma(st[r], ks[i0 + r], kv);
        red[ib][j] = kv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        kv = 0.0f;
        for (int k = 0; k < IB; ++k) kv += red[k][j];
        const float delta = (vs[j] - g * kv) * be;
        float o = 0.0f;
        for (int r = 0; r < RPT; ++r) {
            st[r] = fma(g, st[r], ks[i0 + r] * delta);
            o = fma(st[r], qs[i0 + r], o);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        red[ib][j] = o;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // the readout and the output norm: column j's o (every i-block thread of the column holds the same value)
        o = 0.0f;
        for (int k = 0; k < IB; ++k) o += red[k][j];
        o = o * scale;
        const float osq = ib == 0 ? simd_sum(o * o) : 0.0f;
        if (ib == 0 && lane == 0) part[cb] = osq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ib == 0) {
            const float tot = (part[0] + part[1]) + (part[2] + part[3]);
            const float rs = rsqrt(tot / float(S) + a.eps);
            const float zz = z[tr * hv * S + h * S + j];
            const float yv = ((rs * o) * gamma[j]) * (1.0f / (1.0f + exp(-zz)));
            y[tr * hv * S + h * S + j] = yv;
            if (a.plane) {
                const float q2s[8] = {1.0f, 0.25f, 0.0625f, 0.015625f, 0.00390625f, 0.0009765625f, 0.000244140625f,
                                      0.00006103515625f};
                const float p4s[4] = {1.0f, 0.00390625f, 1.52587890625e-05f, 5.9604644775390625e-08f};
                device float* raw = y + tr * hv * S + h * S + j;
                const int jj = j & 15;
                raw[a.plane] = yv * q2s[jj & 7];
                raw[2 * a.plane] = yv * p4s[jj & 3];
                raw[3 * a.plane] = (jj & 1) ? yv * 0.00390625f : yv;
                ys[j] = yv;
            }
        }
        if (t + 1 == commit_at) {
            // the state and the conv history after token t (registers: st after this token's update, hst after its
            // shift); the state is read only before the loop, so in place is safe.  q/k history: the threadgroups
            // h < h_k own their q/k head (h == hk); v history: every head its own - to hist_out, NOT in place: a q/k
            // channel is read by the three threadgroups h % h_k at their start, and a threadgroup may start after its
            // owner has finished (seen on the 10-core M4); the caller swaps the two
            for (int r = 0; r < RPT; ++r) state[((i0 + r) * hv + h) * S + j] = st[r];
            if (ch >= 0 && (tid >= 256 || h < int(a.h_k)))
                for (int k = 0; k < 3; ++k) hist_out[ch * 3 + k] = hst[k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (a.plane && ib == 0 && (j & 15) == 0) {
            float s0 = 0.0f, s1 = 0.0f;
            for (int k = 0; k < 16; k += 2) { s0 += ys[j + k]; s1 += ys[j + k + 1]; }
            y[4 * a.plane + tr * (hv * S / 16) + (h * S + j) / 16] = s0 + s1;
        }
    }
}

template [[host_name("gdn_step")]] kernel decltype(gdn_step_t<4>) gdn_step_t<4>;
template [[host_name("gdn_step8")]] kernel decltype(gdn_step_t<8>) gdn_step_t<8>;
