// metal/kernels/gdn_prompt.metal - a GDN layer's recurrence for a PROMPT chunk of T tokens, in three passes: the work
// that does not touch the state runs parallel over the tokens; only the state update is a scan, and the columns of a
// head's state are independent in it (column j needs k, q, v_j, the gate and beta), so a lane owns one column.
//
//   gdn_pre    per (token, head): conv (4 taps; history oldest first) -> SiLU; q, k L2 norm * 1/sqrt(128) (the decode
//              step's arithmetic); kq = k . q per key head; gate = exp(softplus(alpha + dt) * a), beta = sigmoid(beta);
//              the chunk's last 3 conv inputs -> hist_out
//   gdn_scan   per (value head, 32 columns x CB simdgroups): the state column in registers (128 floats a lane);
//              per token a = S^T k, b = S^T q (old state), delta = (v - g a) beta, o = (g b + delta kq) / sqrt(128)
//              (= S_new^T q / sqrt(128)), S = g S + k delta^T; q and k staged per TB tokens in threadgroup memory
//   gdn_post   per (token, head): y = o * rsqrt(mean(o^2) + eps) * gamma * sigmoid(z) -> fp16 (the out GEMM's operand)
//              and, with `f32`, fp32
//
// State layout as Strata's: S[(i * h_v + h) * 128 + j].  Value head h reads q/k head h % h_k.
// The kernels are new entries (nothing of gdn.metal changes).

struct GdnPArgs {
    uint n_tok;
    uint h_k;        // 16
    uint h_v;        // 48
    uint channels;   // 10240
    float eps;       // 1e-6
    uint f32;        // gdn_post: also write fp32 y
};

// function constant 7 (optional): the row stride in floats of ONE fused output [qkv | z | alpha | beta] of the four input
// projections (16480).  gdn_pre then reads qkv / alpha / beta and gdn_post reads z at that stride (the buffers bound at offsets);
// undefined (every other caller): the separate buffers of channels / h_v / h_v * 128 floats a row.
constant int GDN_LD [[function_constant(7)]];
constant bool gdn_ld_on = is_function_constant_defined(GDN_LD);

static inline float gp_conv(device const float* qkv, device const float* hist, device const float* cw, int ch, int t,
                            int channels) {
    float s = 0.0f;
    for (int k = 0; k < 4; ++k) {
        const int src = t - 3 + k;
        const float x = src >= 0 ? qkv[src * channels + ch] : hist[ch * 3 + (src + 3)];
        s = fma(x, cw[ch * 4 + k], s);
    }
    s = s + 0.0f;
    return s / (1.0f + exp(-s));
}

// grid (h_k + h_v / 2, T), 256 threads.  x < h_k: q (threads 0..127) and k (128..255) of key head x; else value heads
// 2 (x - h_k) and + 1.
kernel void gdn_pre(device const float* qkv [[buffer(0)]], device const float* hist [[buffer(1)]],
                    device const float* conv_w [[buffer(2)]], device const float* alpha [[buffer(3)]],
                    device const float* beta [[buffer(4)]], device const float* dt [[buffer(5)]],
                    device const float* ssm_a [[buffer(6)]], device float* qn [[buffer(7)]], device float* kn [[buffer(8)]],
                    device float* vc [[buffer(9)]], device float2* gb [[buffer(10)]], device float* kq [[buffer(11)]],
                    device float* hist_out [[buffer(12)]], constant GdnPArgs& a [[buffer(13)]],
                    uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int S = 128;
    threadgroup float part[8];
    threadgroup float prod[S];
    const int x = int(tg.x), t = int(tg.y), hk_n = int(a.h_k), hv = int(a.h_v), C = int(a.channels), T = int(a.n_tok);
    int ch;
    if (x < hk_n) ch = tid < 128 ? x * S + tid : hk_n * S + x * S + (tid - 128);
    else ch = 2 * hk_n * S + (2 * (x - hk_n) + (tid >> 7)) * S + (tid & 127);
    const int QL = gdn_ld_on ? GDN_LD : C;
    const float cv = gp_conv(qkv, hist, conv_w, ch, t, QL);
    if (t == T - 1)
        for (int k = 0; k < 3; ++k) {
            const int src = T - 3 + k;
            hist_out[ch * 3 + k] = src >= 0 ? qkv[src * QL + ch] : hist[ch * 3 + (src + 3)];
        }
    if (x < hk_n) {
        const float ssq = simd_sum(cv * cv);
        if (lane == 0) part[sg] = ssq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int base = tid < 128 ? 0 : 4;
        const float tot = (part[base] + part[base + 1]) + (part[base + 2] + part[base + 3]);
        const float rs = rsqrt(tot / float(S) + a.eps / float(S));
        const float nv = (rs * cv) * (1.0f / sqrt(float(S)));
        if (tid < 128) {
            qn[(t * hk_n + x) * S + tid] = nv;
            prod[tid] = nv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid >= 128) {
            kn[(t * hk_n + x) * S + tid - 128] = nv;
            const float p = simd_sum(nv * prod[tid - 128]);
            if (lane == 0) part[sg] = p;   // simdgroups 4..7 (q's slots 0..3 are read already)
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) kq[t * hk_n + x] = (part[4] + part[5]) + (part[6] + part[7]);
    } else {
        const int h = 2 * (x - hk_n) + (tid >> 7);
        vc[(t * hv + h) * S + (tid & 127)] = cv;
        if ((tid & 127) == 0) {
            const int AL = gdn_ld_on ? GDN_LD : hv;
            const float v = alpha[t * AL + h] + dt[h];
            const float e = exp(v), u = 1.0f + e;
            const float sp = v > 20.0f ? v : (u == 1.0f ? e : log(u) * e / (u - 1.0f));
            gb[t * hv + h] = float2(exp(sp * ssm_a[h]), 1.0f / (1.0f + exp(-beta[t * AL + h])));
        }
    }
}

// grid (h_v * 4 / CB), CB * 32 threads; TB tokens of q and k staged at a time
template <int CB, int TB>
kernel void gdn_scan_t(device float* state [[buffer(0)]], device const float4* qn [[buffer(1)]],
                       device const float4* kn [[buffer(2)]], device const float* vc [[buffer(3)]],
                       device const float2* gb [[buffer(4)]], device const float* kq [[buffer(5)]],
                       device float* o [[buffer(6)]], constant GdnPArgs& a [[buffer(7)]],
                       uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                       ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int S = 128, NB = 4 / CB;
    threadgroup float4 qs[TB][32], ks[TB][32];
    const int h = int(tg.x) / NB, hk = h % int(a.h_k), hv = int(a.h_v), hkn = int(a.h_k);
    const int j = ((int(tg.x) % NB) * CB + sg) * 32 + lane;
    const int T = int(a.n_tok);
    const float scale = 1.0f / sqrt(float(S));
    float st[S];
#pragma clang loop unroll(full)
    for (int i = 0; i < S; ++i) st[i] = state[(i * hv + h) * S + j];
    for (int t0 = 0; t0 < T; t0 += TB) {
        const int nb = min(TB, T - t0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx = tid; idx < nb * 32; idx += CB * 32) {
            const int tt = idx >> 5, w = idx & 31;
            qs[tt][w] = qn[((t0 + tt) * hkn + hk) * 32 + w];
            ks[tt][w] = kn[((t0 + tt) * hkn + hk) * 32 + w];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int tt = 0; tt < nb; ++tt) {
            const int t = t0 + tt;
            const float2 g_b = gb[t * hv + h];
            const float kqv = kq[t * hkn + hk];
            const float v = vc[(t * hv + h) * S + j];
            float ka = 0.0f, kb = 0.0f, qa = 0.0f, qb = 0.0f;
#pragma clang loop unroll(full)
            for (int w = 0; w < 32; w += 2) {
                const float4 k0 = ks[tt][w], k1 = ks[tt][w + 1], q0 = qs[tt][w], q1 = qs[tt][w + 1];
                ka = fma(st[4 * w], k0.x, ka); ka = fma(st[4 * w + 1], k0.y, ka);
                ka = fma(st[4 * w + 2], k0.z, ka); ka = fma(st[4 * w + 3], k0.w, ka);
                kb = fma(st[4 * w + 4], k1.x, kb); kb = fma(st[4 * w + 5], k1.y, kb);
                kb = fma(st[4 * w + 6], k1.z, kb); kb = fma(st[4 * w + 7], k1.w, kb);
                qa = fma(st[4 * w], q0.x, qa); qa = fma(st[4 * w + 1], q0.y, qa);
                qa = fma(st[4 * w + 2], q0.z, qa); qa = fma(st[4 * w + 3], q0.w, qa);
                qb = fma(st[4 * w + 4], q1.x, qb); qb = fma(st[4 * w + 5], q1.y, qb);
                qb = fma(st[4 * w + 6], q1.z, qb); qb = fma(st[4 * w + 7], q1.w, qb);
            }
            const float g = g_b.x, kv = ka + kb, qv = qa + qb;
            const float delta = (v - g * kv) * g_b.y;
            o[(t * hv + h) * S + j] = fma(g, qv, delta * kqv) * scale;
#pragma clang loop unroll(full)
            for (int w = 0; w < 32; ++w) {
                const float4 k4 = ks[tt][w];
                st[4 * w] = fma(g, st[4 * w], k4.x * delta);
                st[4 * w + 1] = fma(g, st[4 * w + 1], k4.y * delta);
                st[4 * w + 2] = fma(g, st[4 * w + 2], k4.z * delta);
                st[4 * w + 3] = fma(g, st[4 * w + 3], k4.w * delta);
            }
        }
    }
#pragma clang loop unroll(full)
    for (int i = 0; i < S; ++i) state[(i * hv + h) * S + j] = st[i];
}

// a probe for the bench (function constant 6): 1 = no per-token exchange (wrong results), 2 = no state update
constant int GDN_PROBE [[function_constant(6)]];
constant int gdn_probe = is_function_constant_defined(GDN_PROBE) ? GDN_PROBE : 0;

// the rows of a column split over IB simdgroups (128 / IB rows a lane, fewer registers): grid (h_v * 4), IB * 32
// threads; lane = column (32 columns a threadgroup); per token the IB partial dots are exchanged through threadgroup
// memory (double-buffered: one barrier a token)
template <int IB, int TB>
kernel void gdn_scan_r(device float* state [[buffer(0)]], device const float4* qn [[buffer(1)]],
                       device const float4* kn [[buffer(2)]], device const float* vc [[buffer(3)]],
                       device const float2* gb [[buffer(4)]], device const float* kq [[buffer(5)]],
                       device float* o [[buffer(6)]], constant GdnPArgs& a [[buffer(7)]],
                       uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                       ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int S = 128, R = S / IB, R4 = R / 4;
    threadgroup float4 qs[TB][32], ks[TB][32];
    threadgroup float2 red[2][IB][32];
    threadgroup float vs[TB][32];
    threadgroup float4 gk[TB];   // gate, beta, kq
    const int h = int(tg.x) / 4, hk = h % int(a.h_k), hv = int(a.h_v), hkn = int(a.h_k);
    const int j = (int(tg.x) % 4) * 32 + lane, i0 = sg * R;
    const int T = int(a.n_tok);
    const float scale = 1.0f / sqrt(float(S));
    float st[R];
#pragma clang loop unroll(full)
    for (int i = 0; i < R; ++i) st[i] = state[((i0 + i) * hv + h) * S + j];
    for (int t0 = 0; t0 < T; t0 += TB) {
        const int nb = min(TB, T - t0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx = tid; idx < nb * 32; idx += IB * 32) {
            const int tt = idx >> 5, w = idx & 31;
            qs[tt][w] = qn[((t0 + tt) * hkn + hk) * 32 + w];
            ks[tt][w] = kn[((t0 + tt) * hkn + hk) * 32 + w];
            vs[tt][w] = vc[((t0 + tt) * hv + h) * S + (j - lane) + w];
            if (w == 0) {
                const float2 g_b = gb[(t0 + tt) * hv + h];
                gk[tt] = float4(g_b.x, g_b.y, kq[(t0 + tt) * hkn + hk], 0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int tt = 0; tt < nb; ++tt) {
            const int t = t0 + tt, slot = t & 1;
            float ka = 0.0f, qa = 0.0f;
#pragma clang loop unroll(full)
            for (int w = 0; w < R4; ++w) {
                const float4 k4 = ks[tt][i0 / 4 + w], q4 = qs[tt][i0 / 4 + w];
                ka = fma(st[4 * w], k4.x, ka); ka = fma(st[4 * w + 1], k4.y, ka);
                ka = fma(st[4 * w + 2], k4.z, ka); ka = fma(st[4 * w + 3], k4.w, ka);
                qa = fma(st[4 * w], q4.x, qa); qa = fma(st[4 * w + 1], q4.y, qa);
                qa = fma(st[4 * w + 2], q4.z, qa); qa = fma(st[4 * w + 3], q4.w, qa);
            }
            const float4 g_b = gk[tt];
            const float kqv = g_b.z;
            const float v = vs[tt][lane];
            float2 s = float2(ka, qa);
            if (gdn_probe != 1) {
                red[slot][sg][lane] = s;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                s = red[slot][0][lane];
#pragma clang loop unroll(full)
                for (int b = 1; b < IB; ++b) s += red[slot][b][lane];
            }
            const float g = g_b.x;
            const float delta = (v - g * s.x) * g_b.y;
            if (sg == 0) o[(t * hv + h) * S + j] = fma(g, s.y, delta * kqv) * scale;
            if (gdn_probe != 2)
#pragma clang loop unroll(full)
            for (int w = 0; w < R4; ++w) {
                const float4 k4 = ks[tt][i0 / 4 + w];
                st[4 * w] = fma(g, st[4 * w], k4.x * delta);
                st[4 * w + 1] = fma(g, st[4 * w + 1], k4.y * delta);
                st[4 * w + 2] = fma(g, st[4 * w + 2], k4.z * delta);
                st[4 * w + 3] = fma(g, st[4 * w + 3], k4.w * delta);
            }
        }
    }
#pragma clang loop unroll(full)
    for (int i = 0; i < R; ++i) state[((i0 + i) * hv + h) * S + j] = st[i];
}

// gdn_scan_r<8, 16> with the 16 state rows of a lane in four float4 variables (no array): a register-allocation check
kernel void gdn_scan_v(device float* state [[buffer(0)]], device const float4* qn [[buffer(1)]],
                       device const float4* kn [[buffer(2)]], device const float* vc [[buffer(3)]],
                       device const float2* gb [[buffer(4)]], device const float* kq [[buffer(5)]],
                       device float* o [[buffer(6)]], constant GdnPArgs& a [[buffer(7)]],
                       uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                       ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int S = 128, IB = 8, TB = 16;
    threadgroup float4 qs[TB][32], ks[TB][32];
    threadgroup float2 red[2][IB][32];
    threadgroup float vs[TB][32];
    threadgroup float4 gk[TB];
    const int h = int(tg.x) / 4, hk = h % int(a.h_k), hv = int(a.h_v), hkn = int(a.h_k);
    const int j = (int(tg.x) % 4) * 32 + lane, i0 = sg * 16, w0 = sg * 4;
    const int T = int(a.n_tok);
    const float scale = 1.0f / sqrt(float(S));
    device float* sp = state + (i0 * hv + h) * S + j;
    const int rs = hv * S;
    float4 s0 = float4(sp[0], sp[rs], sp[2 * rs], sp[3 * rs]);
    float4 s1 = float4(sp[4 * rs], sp[5 * rs], sp[6 * rs], sp[7 * rs]);
    float4 s2 = float4(sp[8 * rs], sp[9 * rs], sp[10 * rs], sp[11 * rs]);
    float4 s3 = float4(sp[12 * rs], sp[13 * rs], sp[14 * rs], sp[15 * rs]);
    for (int t0 = 0; t0 < T; t0 += TB) {
        const int nb = min(TB, T - t0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx = tid; idx < nb * 32; idx += IB * 32) {
            const int tt = idx >> 5, w = idx & 31;
            qs[tt][w] = qn[((t0 + tt) * hkn + hk) * 32 + w];
            ks[tt][w] = kn[((t0 + tt) * hkn + hk) * 32 + w];
            vs[tt][w] = vc[((t0 + tt) * hv + h) * S + (j - lane) + w];
            if (w == 0) {
                const float2 g_b = gb[(t0 + tt) * hv + h];
                gk[tt] = float4(g_b.x, g_b.y, kq[(t0 + tt) * hkn + hk], 0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int tt = 0; tt < nb; ++tt) {
            const int t = t0 + tt, slot = t & 1;
            const float4 k0 = ks[tt][w0], k1 = ks[tt][w0 + 1], k2 = ks[tt][w0 + 2], k3 = ks[tt][w0 + 3];
            const float4 q0 = qs[tt][w0], q1 = qs[tt][w0 + 1], q2 = qs[tt][w0 + 2], q3 = qs[tt][w0 + 3];
            const float ka = (dot(s0, k0) + dot(s1, k1)) + (dot(s2, k2) + dot(s3, k3));
            const float qa = (dot(s0, q0) + dot(s1, q1)) + (dot(s2, q2) + dot(s3, q3));
            red[slot][sg][lane] = float2(ka, qa);
            const float4 g_b = gk[tt];
            const float v = vs[tt][lane];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float2 r = red[slot][0][lane];
            for (int b = 1; b < IB; ++b) r += red[slot][b][lane];
            const float g = g_b.x;
            const float delta = (v - g * r.x) * g_b.y;
            if (sg == 0) o[(t * hv + h) * S + j] = fma(g, r.y, delta * g_b.z) * scale;
            s0 = fma(float4(g), s0, k0 * delta);
            s1 = fma(float4(g), s1, k1 * delta);
            s2 = fma(float4(g), s2, k2 * delta);
            s3 = fma(float4(g), s3, k3 * delta);
        }
    }
    sp[0] = s0.x; sp[rs] = s0.y; sp[2 * rs] = s0.z; sp[3 * rs] = s0.w;
    sp[4 * rs] = s1.x; sp[5 * rs] = s1.y; sp[6 * rs] = s1.z; sp[7 * rs] = s1.w;
    sp[8 * rs] = s2.x; sp[9 * rs] = s2.y; sp[10 * rs] = s2.z; sp[11 * rs] = s2.w;
    sp[12 * rs] = s3.x; sp[13 * rs] = s3.y; sp[14 * rs] = s3.z; sp[15 * rs] = s3.w;
}

// grid (h_v, T), 128 threads
kernel void gdn_post(device const float* o [[buffer(0)]], device const float* z [[buffer(1)]],
                     device const float* gamma [[buffer(2)]], device half* yh [[buffer(3)]], device float* y [[buffer(4)]],
                     constant GdnPArgs& a [[buffer(5)]],
                     uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                     ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[4];
    const int h = int(tg.x), t = int(tg.y), hv = int(a.h_v);
    const uint i = uint(t * hv + h) * 128 + tid;
    const float ov = o[i];
    const float ss = simd_sum(ov * ov);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float tot = (part[0] + part[1]) + (part[2] + part[3]);
    const float rs = rsqrt(tot / 128.0f + a.eps);
    const float zv = gdn_ld_on ? z[uint(t) * uint(GDN_LD) + uint(h) * 128u + tid] : z[i];
    const float yv = ((rs * ov) * gamma[tid]) * (1.0f / (1.0f + exp(-zv)));
    yh[i] = half(yv);
    if (a.f32) y[i] = yv;
}

template [[host_name("gdn_scan_1")]] kernel decltype(gdn_scan_t<1, 16>) gdn_scan_t<1, 16>;
template [[host_name("gdn_scan_2")]] kernel decltype(gdn_scan_t<2, 16>) gdn_scan_t<2, 16>;
template [[host_name("gdn_scan_4")]] kernel decltype(gdn_scan_t<4, 16>) gdn_scan_t<4, 16>;
template [[host_name("gdn_scanr_2")]] kernel decltype(gdn_scan_r<2, 16>) gdn_scan_r<2, 16>;
template [[host_name("gdn_scanr_4")]] kernel decltype(gdn_scan_r<4, 16>) gdn_scan_r<4, 16>;
template [[host_name("gdn_scanr_8")]] kernel decltype(gdn_scan_r<8, 16>) gdn_scan_r<8, 16>;
template [[host_name("gdn_scanr_4_8")]] kernel decltype(gdn_scan_r<4, 8>) gdn_scan_r<4, 8>;
template [[host_name("gdn_scanr_4_4")]] kernel decltype(gdn_scan_r<4, 4>) gdn_scan_r<4, 4>;
template [[host_name("gdn_scanr_4_2")]] kernel decltype(gdn_scan_r<4, 2>) gdn_scan_r<4, 2>;
template [[host_name("gdn_scanr_8_8")]] kernel decltype(gdn_scan_r<8, 8>) gdn_scan_r<8, 8>;
template [[host_name("gdn_scanr_8_4")]] kernel decltype(gdn_scan_r<8, 4>) gdn_scan_r<8, 4>;
template [[host_name("gdn_scanr_2_8")]] kernel decltype(gdn_scan_r<2, 8>) gdn_scan_r<2, 8>;
template [[host_name("gdn_scanr_2_4")]] kernel decltype(gdn_scan_r<2, 4>) gdn_scan_r<2, 4>;

// ---- gdn_scanf: gdn_scan_r<4, TB> with gdn_post folded in.  One threadgroup a value head (grid h_v), 16 simdgroups =
// 4 column blocks x 4 row blocks, 512 threads.  The scan is gdn_scan_r's, token for token (the same operations in the
// same order, so o is the same bits); a column block's row-block-0 simdgroup holds o for its 32 columns, writes them and
// their simd_sum of squares to threadgroup memory, and once every TB tokens the whole threadgroup runs gdn_post's
// arithmetic on those tokens (the sum of the four partials in gdn_post's order: the same bits as gdn_post).  Outputs:
// y (fp16) and, with `f32`, fp32 y; o itself never reaches device memory.
template <int TB>
kernel void gdn_scan_f(device float* state [[buffer(0)]], device const float4* qn [[buffer(1)]],
                       device const float4* kn [[buffer(2)]], device const float* vc [[buffer(3)]],
                       device const float2* gb [[buffer(4)]], device const float* kq [[buffer(5)]],
                       device const float* z [[buffer(6)]], device const float* gamma [[buffer(7)]],
                       device half* yh [[buffer(8)]], device float* y32 [[buffer(9)]], constant GdnPArgs& a [[buffer(10)]],
                       uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                       ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int S = 128, IB = 4, R = S / IB, R4 = R / 4;
    threadgroup float4 qs[TB][32], ks[TB][32];
    threadgroup float2 red[2][16][32];
    threadgroup float vs[TB][128];
    threadgroup float4 gk[TB];
    threadgroup float ot[TB][128];
    threadgroup float osq[TB][4];
    const int h = int(tg.x), hk = h % int(a.h_k), hv = int(a.h_v), hkn = int(a.h_k);
    const int cb = sg >> 2, ib = sg & 3;
    const int j = cb * 32 + lane, i0 = ib * R;
    const int T = int(a.n_tok);
    const float scale = 1.0f / sqrt(float(S));
    float st[R];
#pragma clang loop unroll(full)
    for (int i = 0; i < R; ++i) st[i] = state[((i0 + i) * hv + h) * S + j];
    for (int t0 = 0; t0 < T; t0 += TB) {
        const int nb = min(TB, T - t0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx = tid; idx < nb * 32; idx += 512) {
            const int tt = idx >> 5, w = idx & 31;
            qs[tt][w] = qn[((t0 + tt) * hkn + hk) * 32 + w];
            ks[tt][w] = kn[((t0 + tt) * hkn + hk) * 32 + w];
        }
        for (int idx = tid; idx < nb * 128; idx += 512) {
            const int tt = idx >> 7, w = idx & 127;
            vs[tt][w] = vc[((t0 + tt) * hv + h) * S + w];
        }
        for (int tt = tid; tt < nb; tt += 512) {
            const float2 g_b = gb[(t0 + tt) * hv + h];
            gk[tt] = float4(g_b.x, g_b.y, kq[(t0 + tt) * hkn + hk], 0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int tt = 0; tt < nb; ++tt) {
            const int t = t0 + tt, slot = t & 1;
            float ka = 0.0f, qa = 0.0f;
#pragma clang loop unroll(full)
            for (int w = 0; w < R4; ++w) {
                const float4 k4 = ks[tt][i0 / 4 + w], q4 = qs[tt][i0 / 4 + w];
                ka = fma(st[4 * w], k4.x, ka); ka = fma(st[4 * w + 1], k4.y, ka);
                ka = fma(st[4 * w + 2], k4.z, ka); ka = fma(st[4 * w + 3], k4.w, ka);
                qa = fma(st[4 * w], q4.x, qa); qa = fma(st[4 * w + 1], q4.y, qa);
                qa = fma(st[4 * w + 2], q4.z, qa); qa = fma(st[4 * w + 3], q4.w, qa);
            }
            const float4 g_b = gk[tt];
            const float v = vs[tt][j];
            red[slot][sg][lane] = float2(ka, qa);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float2 s = red[slot][cb * 4][lane];
#pragma clang loop unroll(full)
            for (int b = 1; b < IB; ++b) s += red[slot][cb * 4 + b][lane];
            const float g = g_b.x;
            const float delta = (v - g * s.x) * g_b.y;
            if (ib == 0) {
                const float ov = fma(g, s.y, delta * g_b.z) * scale;
                ot[tt][j] = ov;
                const float ss = simd_sum(ov * ov);
                if (lane == 0) osq[tt][cb] = ss;
            }
#pragma clang loop unroll(full)
            for (int w = 0; w < R4; ++w) {
                const float4 k4 = ks[tt][i0 / 4 + w];
                st[4 * w] = fma(g, st[4 * w], k4.x * delta);
                st[4 * w + 1] = fma(g, st[4 * w + 1], k4.y * delta);
                st[4 * w + 2] = fma(g, st[4 * w + 2], k4.z * delta);
                st[4 * w + 3] = fma(g, st[4 * w + 3], k4.w * delta);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // gdn_post for these tokens
        for (int idx = tid; idx < nb * 128; idx += 512) {
            const int tt = idx >> 7, c = idx & 127;
            const float tot = (osq[tt][0] + osq[tt][1]) + (osq[tt][2] + osq[tt][3]);
            const float rs = rsqrt(tot / 128.0f + a.eps);
            const ulong i = ulong((t0 + tt) * hv + h) * 128 + c;
            const float ov = ot[tt][c];
            const float yv = ((rs * ov) * gamma[c]) * (1.0f / (1.0f + exp(-z[i])));
            yh[i] = half(yv);
            if (a.f32) y32[i] = yv;
        }
    }
#pragma clang loop unroll(full)
    for (int i = 0; i < R; ++i) state[((i0 + i) * hv + h) * S + j] = st[i];
}
template [[host_name("gdn_scanf_8")]] kernel decltype(gdn_scan_f<8>) gdn_scan_f<8>;
template [[host_name("gdn_scanf_4")]] kernel decltype(gdn_scan_f<4>) gdn_scan_f<4>;
template [[host_name("gdn_scanf_2")]] kernel decltype(gdn_scan_f<2>) gdn_scan_f<2>;

// ---- the chunked form (chunks of GC = 32 tokens), the WY / UT form of the gated delta rule.  With G_t the product
// of the chunk's gates up to t (log space: lg_t) and S0 the state entering the chunk:
//   (I + A) D = B,  A_ts = beta_t (k_t . k_s) G_t / G_s (s < t),  B = [beta v | beta G k]   ->   D = [U | W]
//   delta = U - W S0,   o_t = G_t S0^T q_t + sum_{s <= t} (q_t . k_s) G_t / G_s delta_s,
//   S_L = G_L S0 + sum_s G_L / G_s k_s delta_s^T
// (the sequential recurrence's algebra; the rounding differs).  Three kernels:
//   gdn_cprep   per (head, chunk): lg, A, the forward substitution for U and W (a thread per column of B)
//   gdn_cstate  per (head, 32 state columns): the chunks in order - delta, the inter-chunk output, the state update
//   gdn_cout    per (head, chunk): o += the intra-chunk part, times 1/sqrt(128)
constant constexpr int GC = 32;

// grid (h_v, chunks), 256 threads.  U, W: [h][chunk][GC][128]; lgo: [h][chunk][GC]
kernel void gdn_cprep(device const float* qn [[buffer(0)]], device const float* kn [[buffer(1)]],
                      device const float* vc [[buffer(2)]], device const float2* gb [[buffer(3)]],
                      device float* U [[buffer(4)]], device float* W [[buffer(5)]], device float* lgo [[buffer(6)]],
                      constant GdnPArgs& a [[buffer(7)]],
                      uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int S = 128;
    threadgroup float A[GC][GC];
    threadgroup float lg[GC], bt[GC];
    const int h = int(tg.x), c = int(tg.y), hv = int(a.h_v), hkn = int(a.h_k), hk = h % hkn, T = int(a.n_tok);
    const int NC = (T + GC - 1) / GC, t0 = c * GC, nb = min(GC, T - t0);
    if (tid == 0) {
        float s = 0.0f;
        for (int t = 0; t < GC; ++t) {
            if (t < nb) s += max(log(gb[(t0 + t) * hv + h].x), -80.0f);
            lg[t] = s;
            bt[t] = t < nb ? gb[(t0 + t) * hv + h].y : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = tid; idx < GC * GC; idx += 256) {
        const int t = idx / GC, s = idx % GC;
        float v = 0.0f;
        if (s < t && t < nb) {
            device const float4* kt = (device const float4*)(kn + ((t0 + t) * hkn + hk) * S);
            device const float4* ks = (device const float4*)(kn + ((t0 + s) * hkn + hk) * S);
            float d = 0.0f;
            for (int i = 0; i < S / 4; ++i) d += dot(kt[i], ks[i]);
            v = bt[t] * d * exp(lg[t] - lg[s]);
        }
        A[t][s] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // forward substitution, column j of B a thread (threads 0..127: beta v; 128..255: beta G k)
    const int j = tid & 127;
    const bool isw = tid >= 128;
    float x[GC];
    for (int t = 0; t < GC; ++t) {
        float b = 0.0f;
        if (t < nb) b = isw ? bt[t] * exp(lg[t]) * kn[((t0 + t) * hkn + hk) * S + j] : bt[t] * vc[((t0 + t) * hv + h) * S + j];
        for (int s = 0; s < t; ++s) b = fma(-A[t][s], x[s], b);
        x[t] = b;
    }
    device float* o = (isw ? W : U) + (ulong(h) * NC + c) * GC * S + j;
    for (int t = 0; t < GC; ++t) o[t * S] = x[t];
    if (tid < GC) lgo[(h * NC + c) * GC + tid] = lg[tid];
}

// grid (h_v * 4), 256 threads = 8 row blocks of 16 x 32 columns; the state's 128 x 32 block in registers (16 a thread)
kernel void gdn_cstate(device float* state [[buffer(0)]], device const float* qn [[buffer(1)]],
                       device const float* kn [[buffer(2)]], device const float* U [[buffer(3)]],
                       device const float* W [[buffer(4)]], device const float* lgo [[buffer(5)]],
                       device float* delta [[buffer(6)]], device float* o [[buffer(7)]], constant GdnPArgs& a [[buffer(8)]],
                       uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                       ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int S = 128, RB = 16;
    threadgroup float red[8][16][32];
    threadgroup float dl[GC][32];
    const int h = int(tg.x) / 4, hv = int(a.h_v), hkn = int(a.h_k), hk = h % hkn, T = int(a.n_tok);
    const int NC = (T + GC - 1) / GC;
    const int j = (int(tg.x) % 4) * 32 + lane, i0 = sg * RB;
    float st[RB];
    for (int r = 0; r < RB; ++r) st[r] = state[((i0 + r) * hv + h) * S + j];
    for (int c = 0; c < NC; ++c) {
        const int t0 = c * GC, nb = min(GC, T - t0);
        device const float* Uc = U + (ulong(h) * NC + c) * GC * S;
        device const float* Wc = W + (ulong(h) * NC + c) * GC * S;
        device const float* lg = lgo + (h * NC + c) * GC;
        // pass 0: delta = U - W S0;  pass 1: o_inter = G_t S0^T q_t;  16 tokens at a time
        for (int pass = 0; pass < 2; ++pass) {
            for (int hf = 0; hf < 2; ++hf) {
                float p[16];
                for (int u = 0; u < 16; ++u) {
                    const int t = hf * 16 + u;
                    float s = 0.0f;
                    if (t < nb) {
                        device const float* v = pass == 0 ? Wc + t * S + i0 : qn + ((t0 + t) * hkn + hk) * S + i0;
                        for (int r = 0; r < RB; ++r) s = fma(v[r], st[r], s);
                    }
                    p[u] = s;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int u = 0; u < 16; ++u) red[sg][u][lane] = p[u];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int u = sg * 2; u < sg * 2 + 2; ++u) {
                    const int t = hf * 16 + u;
                    float s = 0.0f;
                    for (int b = 0; b < 8; ++b) s += red[b][u][lane];
                    if (t < nb) {
                        if (pass == 0) {
                            const float d = Uc[t * S + j] - s;
                            dl[t][lane] = d;
                            delta[((ulong(h) * NC + c) * GC + t) * S + j] = d;
                        } else {
                            o[((t0 + t) * hv + h) * S + j] = exp(lg[t]) * s;
                        }
                    } else if (pass == 0) {
                        dl[t][lane] = 0.0f;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = G_L S0 + sum_s G_L / G_s k_s delta_s^T
        const float lgl = lg[nb - 1], gl = exp(lgl);
        for (int r = 0; r < RB; ++r) st[r] *= gl;
        for (int s = 0; s < nb; ++s) {
            const float f = exp(lgl - lg[s]) * dl[s][lane];
            device const float* ks = kn + ((t0 + s) * hkn + hk) * S + i0;
            for (int r = 0; r < RB; ++r) st[r] = fma(ks[r], f, st[r]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int r = 0; r < RB; ++r) state[((i0 + r) * hv + h) * S + j] = st[r];
}

// grid (h_v, chunks), 128 threads (column j): o_t = (o_inter_t + sum_{s <= t} (q_t . k_s) G_t / G_s delta_s) / sqrt(128)
kernel void gdn_cout(device const float* qn [[buffer(0)]], device const float* kn [[buffer(1)]],
                     device const float* delta [[buffer(2)]], device const float* lgo [[buffer(3)]],
                     device float* o [[buffer(4)]], constant GdnPArgs& a [[buffer(5)]],
                     uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int S = 128;
    threadgroup float M[GC][GC];
    const int h = int(tg.x), c = int(tg.y), hv = int(a.h_v), hkn = int(a.h_k), hk = h % hkn, T = int(a.n_tok);
    const int NC = (T + GC - 1) / GC, t0 = c * GC, nb = min(GC, T - t0);
    device const float* lg = lgo + (h * NC + c) * GC;
    for (int idx = tid; idx < GC * GC; idx += 128) {
        const int t = idx / GC, s = idx % GC;
        float v = 0.0f;
        if (s <= t && t < nb) {
            device const float4* qt = (device const float4*)(qn + ((t0 + t) * hkn + hk) * S);
            device const float4* ks = (device const float4*)(kn + ((t0 + s) * hkn + hk) * S);
            float d = 0.0f;
            for (int i = 0; i < S / 4; ++i) d += dot(qt[i], ks[i]);
            v = d * exp(lg[t] - lg[s]);
        }
        M[t][s] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int j = tid;
    device const float* dc = delta + (ulong(h) * NC + c) * GC * S + j;
    float dv[GC];
    for (int s = 0; s < GC; ++s) dv[s] = s < nb ? dc[s * S] : 0.0f;
    const float scale = 1.0f / sqrt(float(S));
    for (int t = 0; t < nb; ++t) {
        float s = o[((t0 + t) * hv + h) * S + j];
        for (int u = 0; u <= t; ++u) s = fma(M[t][u], dv[u], s);
        o[((t0 + t) * hv + h) * S + j] = s * scale;
    }
}

// ---- the chunked form v2: the same algebra as gdn_cprep / gdn_cstate / gdn_cout (WY / UT, chunks of GC = 32), organised for the
// GPU.  Staging the chunk's k in threadgroup memory removes the redundant device loads of gdn_cprep, the state update, the
// inter-chunk and the intra-chunk products run as 8x8 simdgroup_matrix products (fp32), and the intra-chunk part is folded into
// the state kernel (no gdn_cout, no delta buffer).  The chunk's q.k matrix M (with the decay factors) comes from the prep.
//   gdn_cprep2   per (head, chunk): lg, A, the forward substitution for U and W, and M_ts = (q_t . k_s) G_t / G_s (s <= t)
//   gdn_cstate2  per (head, 32 state columns): the chunks in order - delta = U - W S, o = (G q S + M delta) / sqrt(128),
//                S = G_L S + sum_s (G_L / G_s) k_s delta_s^T.  The q and k rows of the last chunk's 32 tokens are read in
//                full (rows past n_tok are read and multiplied by zero): the caller's buffers hold a multiple of 32 rows.
kernel void gdn_cprep2(device const float* qn [[buffer(0)]], device const float* kn [[buffer(1)]],
                       device const float* vc [[buffer(2)]], device const float2* gb [[buffer(3)]],
                       device float* U [[buffer(4)]], device float* W [[buffer(5)]], device float* lgo [[buffer(6)]],
                       device float* Mo [[buffer(7)]], constant GdnPArgs& a [[buffer(8)]],
                       uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int S = 128;
    threadgroup float4 Ks[GC * 32];
    threadgroup float4 A4[GC * 8];
    threadgroup float* A = (threadgroup float*)A4;
    threadgroup float lg[GC], bt[GC];
    const int h = int(tg.x), c = int(tg.y), hv = int(a.h_v), hkn = int(a.h_k), hk = h % hkn, T = int(a.n_tok);
    const int NC = (T + GC - 1) / GC, t0 = c * GC, nb = min(GC, T - t0);
    if (tid == 0) {
        float s = 0.0f;
        for (int t = 0; t < GC; ++t) {
            if (t < nb) s += max(log(gb[(t0 + t) * hv + h].x), -80.0f);
            lg[t] = s;
            bt[t] = t < nb ? gb[(t0 + t) * hv + h].y : 0.0f;
        }
    }
    device const float4* kn4 = (device const float4*)kn;
    for (int idx = tid; idx < GC * 32; idx += 256) {
        const int t = idx >> 5, w = idx & 31;
        Ks[idx] = t < nb ? kn4[((t0 + t) * hkn + hk) * 32 + w] : float4(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = tid; idx < GC * GC; idx += 256) {
        const int t = idx / GC, s = idx % GC;
        float av = 0.0f, mv = 0.0f;
        if (t < nb && s <= t && gdn_probe != 1) {
            device const float4* qt = (device const float4*)(qn + ((t0 + t) * hkn + hk) * S);
            float dm = 0.0f, da = 0.0f;
            for (int i = 0; i < S / 4; ++i) {
                const float4 ks = Ks[s * 32 + i];
                dm += dot(qt[i], ks);
                da += dot(Ks[t * 32 + i], ks);
            }
            const float f = exp(lg[t] - lg[s]);
            mv = dm * f;
            if (s < t) av = bt[t] * da * f;
        }
        A[t * GC + s] = av;
        Mo[((ulong(h) * NC + c) * GC + t) * GC + s] = mv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int j = tid & 127;
    const bool isw = tid >= 128;
    threadgroup const float* Kf = (threadgroup const float*)Ks;
    float x[GC];
#pragma clang loop unroll(full)
    for (int t = 0; t < GC; ++t) x[t] = 0.0f;
#pragma clang loop unroll(full)
    for (int t = 0; t < GC; ++t) {
        float b = 0.0f;
        if (t < nb) b = isw ? bt[t] * exp(lg[t]) * Kf[t * S + j] : bt[t] * vc[((t0 + t) * hv + h) * S + j];
#pragma clang loop unroll(full)
        for (int s4 = 0; s4 < (t + 3) / 4; ++s4) {
            const float4 ar = A4[t * 8 + s4];
            b = fma(-ar.x, x[4 * s4], b);
            b = fma(-ar.y, x[4 * s4 + 1], b);
            b = fma(-ar.z, x[4 * s4 + 2], b);
            b = fma(-ar.w, x[4 * s4 + 3], b);
        }
        x[t] = b;
    }
    device float* o = (isw ? W : U) + (ulong(h) * NC + c) * GC * S + j;
#pragma clang loop unroll(full)
    for (int t = 0; t < GC; ++t) o[t * S] = x[t];
    if (tid < GC) lgo[(h * NC + c) * GC + tid] = lg[tid];
}


// grid (h_v * 4), 128 threads = 4 simdgroups.  A simdgroup owns rows 32 sg .. 32 sg + 31 of the 128 x 32 state block as 4 x 4
// simdgroup matrices that stay in registers for the whole call; each chunk the block is stored to threadgroup memory (Sm) once,
// where the two products that need all 128 rows (W S and Q S, sharing the S tiles) read it.
kernel void gdn_cstate2(device float* state [[buffer(0)]], device const float* qn [[buffer(1)]],
                        device const float* kn [[buffer(2)]], device const float* U [[buffer(3)]],
                        device const float* W [[buffer(4)]], device const float* lgo [[buffer(5)]],
                        device const float* Mo [[buffer(6)]], device float* o [[buffer(7)]], constant GdnPArgs& a [[buffer(8)]],
                        uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int S = 128;
    threadgroup float Sm[S][32];
    threadgroup float Dl[GC][32];
    threadgroup float X[GC][32];
    threadgroup float Y[GC][32];
    const int h = int(tg.x) / 4, hv = int(a.h_v), hkn = int(a.h_k), hk = h % hkn, T = int(a.n_tok);
    const int NC = (T + GC - 1) / GC, j0 = (int(tg.x) % 4) * 32;
    const float scale = 1.0f / sqrt(float(S));
    simdgroup_float8x8 st[4][4];
    for (int rt = 0; rt < 4; ++rt)
        for (int jt = 0; jt < 4; ++jt)
            simdgroup_load(st[rt][jt], state + ((32 * sg + 8 * rt) * hv + h) * S + j0 + 8 * jt, hv * S);
    for (int c = 0; c < NC; ++c) {
        const int t0 = c * GC, nb = min(GC, T - t0);
        device const float* Uc = U + (ulong(h) * NC + c) * GC * S;
        device const float* Wc = W + (ulong(h) * NC + c) * GC * S;
        device const float* Mc = Mo + (ulong(h) * NC + c) * GC * GC;
        device const float* lg = lgo + (h * NC + c) * GC;
        for (int rt = 0; rt < 4; ++rt)
            for (int jt = 0; jt < 4; ++jt) simdgroup_store(st[rt][jt], &Sm[32 * sg + 8 * rt][8 * jt], 32);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // P1 = W S0 (into X) and P2 = Q S0 (into Y), the S tiles read once for both
        simdgroup_float8x8 acc1[4], acc2[4];
        for (int jt = 0; jt < 4; ++jt) {
            acc1[jt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            acc2[jt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
        device const float* qrow = qn + ((t0 + 8 * sg) * hkn + hk) * S;
        if (gdn_probe != 1)
#pragma clang loop unroll(full)
        for (int kb = 0; kb < 16; ++kb) {
            simdgroup_float8x8 aw, aq, bs;
            simdgroup_load(aw, Wc + (8 * sg) * S + 8 * kb, S);
            simdgroup_load(aq, qrow + 8 * kb, hkn * S);
            for (int jt = 0; jt < 4; ++jt) {
                simdgroup_load(bs, &Sm[8 * kb][8 * jt], 32);
                simdgroup_multiply_accumulate(acc1[jt], aw, bs, acc1[jt]);
                simdgroup_multiply_accumulate(acc2[jt], aq, bs, acc2[jt]);
            }
        }
        for (int jt = 0; jt < 4; ++jt) {
            simdgroup_store(acc1[jt], &X[8 * sg][8 * jt], 32);
            simdgroup_store(acc2[jt], &Y[8 * sg][8 * jt], 32);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx = tid; idx < GC * 32; idx += 128) {
            const int t = idx >> 5, j = idx & 31;
            Dl[t][j] = t < nb ? Uc[t * S + j0 + j] - X[t][j] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // intra = M delta (into X)
        for (int jt = 0; jt < 4; ++jt) acc1[jt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        if (gdn_probe != 2)
#pragma clang loop unroll(full)
        for (int kb = 0; kb < 4; ++kb) {
            simdgroup_float8x8 am, bd;
            simdgroup_load(am, Mc + (8 * sg) * GC + 8 * kb, GC);
            for (int jt = 0; jt < 4; ++jt) {
                simdgroup_load(bd, &Dl[8 * kb][8 * jt], 32);
                simdgroup_multiply_accumulate(acc1[jt], am, bd, acc1[jt]);
            }
        }
        for (int jt = 0; jt < 4; ++jt) simdgroup_store(acc1[jt], &X[8 * sg][8 * jt], 32);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float lgl = lg[nb - 1], gl = exp(lgl);
        if (gdn_probe != 4)
        for (int idx = tid; idx < GC * 32; idx += 128) {
            const int t = idx >> 5, j = idx & 31;
            if (t < nb) {
                o[((t0 + t) * hv + h) * S + j0 + j] = fma(exp(lg[t]), Y[t][j], X[t][j]) * scale;
                Dl[t][j] *= exp(lgl - lg[t]);
            } else {
                Dl[t][j] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = G_L S + K^T delta'
        for (int rt = 0; rt < 4; ++rt)
            for (int jt = 0; jt < 4; ++jt) {
                st[rt][jt].thread_elements()[0] *= gl;
                st[rt][jt].thread_elements()[1] *= gl;
            }
        if (gdn_probe != 3)
#pragma clang loop unroll(full)
        for (int kb = 0; kb < 4; ++kb) {
            simdgroup_float8x8 bd[4];
            for (int jt = 0; jt < 4; ++jt) simdgroup_load(bd[jt], &Dl[8 * kb][8 * jt], 32);
            for (int rt = 0; rt < 4; ++rt) {
                simdgroup_float8x8 ak;
                simdgroup_load(ak, kn + ((t0 + 8 * kb) * hkn + hk) * S + 32 * sg + 8 * rt, hkn * S, ulong2(0, 0), true);
                for (int jt = 0; jt < 4; ++jt) simdgroup_multiply_accumulate(st[rt][jt], ak, bd[jt], st[rt][jt]);
            }
        }
    }
    for (int rt = 0; rt < 4; ++rt)
        for (int jt = 0; jt < 4; ++jt)
            simdgroup_store(st[rt][jt], state + ((32 * sg + 8 * rt) * hv + h) * S + j0 + 8 * jt, hv * S);
}
